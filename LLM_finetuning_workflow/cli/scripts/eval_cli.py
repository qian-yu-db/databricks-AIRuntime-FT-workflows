#!/usr/bin/env python3
"""
eval_cli.py — local-vLLM eval of fine-tuned Qwen checkpoints, run via `air`.

Runs ON an `air` GPU worker (via configs/eval.air.yaml), NOT on your laptop —
unlike scripts/run_sweep.py. It evaluates one checkpoint (`--tag`) or every
checkpoint under <checkpoints-dir>/ with local vLLM, and logs `eval_<tag>` MLflow
runs tagged `stage=eval` so `run_sweep.py --pick-best` ranks them unchanged.
`run_sweep.py --eval` submits this for you.

This path is viable on a FIPS-hardened workspace because it runs on
`environment.version: databricks_ai_v5` (the fuller image, preinstalled opencv
4.12.0 whose vendored libcrypto is FIPS-clean) instead of numeric `5` (whose
pip-pulled opencv vendors OpenSSL 1.1.1k and aborts vLLM's model-inspection
subprocess on the FIPS self-test). Requires `air >= 1.1.0` to accept the named
image version. See the README ("CLI vLLM eval on a FIPS workspace" section).

It has NO Spark / dbutils dependency (reads test.jsonl with plain open(), scores
with difflib), and imports the scoring/response helpers from lib/extract_eval.py.

    python eval_cli.py --data-dir /Volumes/.../training_data/qwen_sweep \\
        --checkpoints-dir /Volumes/.../checkpoints/qwen_sweep \\
        --experiment /Users/you@databricks.com/qwen-ft-sweep --tag lr2e-6_ep5
    python eval_cli.py --data-dir /Volumes/.../training_data/qwen_sweep \\
        --checkpoints-dir /Volumes/.../checkpoints/qwen_sweep \\
        --experiment /Users/you@databricks.com/qwen-ft-sweep            # all checkpoints
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# High-priority field subset (new CamelCase schema) — mirrors the eval notebook.
TOP_8_FIELDS = [
    "PolicyNumber", "OwnerFile", "LoanFile",
    "OwnerPolicyNumber", "OwnerPolicyAmount", "OwnerPolicyDate",
    "LoanPolicyNumber", "LoanPolicyAmount", "LoanPolicyDate",
]

LOCAL_PORT = 3080


# --- shared scoring/response helpers (the notebook imports these too) --------
def _bootstrap_lib():
    """Add the sibling lib/ dir to sys.path (worker + local)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    for base in (here, os.path.dirname(here), os.getcwd()):
        for cand in (os.path.join(base, "lib"),
                     os.path.join(base, "LLM_finetuning_workflow", "cli", "lib")):
            if os.path.isfile(os.path.join(cand, "extract_eval.py")):
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                return
    raise RuntimeError("could not locate lib/extract_eval.py next to eval_cli.py")


_bootstrap_lib()
from extract_eval import score, clean_response  # noqa: E402


# --- vLLM serving ------------------------------------------------------------
def start_vllm(model_dir, max_model_len, gpu_mem_util, workdir, startup_timeout=1500):
    """Launch vLLM's OpenAI server and poll /health. Dumps the log tail on failure."""
    cmd = " ".join([
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir, "--served-model-name", "qwen",
        "--host", "0.0.0.0", "--port", str(LOCAL_PORT), "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ])
    log_path = os.path.join(workdir, "vllm.log")
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    print(f"vLLM starting (pid={proc.pid}) — polling /health (timeout {startup_timeout}s) ...")
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print("".join(open(log_path).readlines()[-80:]))
            raise RuntimeError(f"vLLM exited during startup (code {proc.returncode}).")
        try:
            if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).status_code == 200:
                print(f"vLLM is ready ({int(startup_timeout - (deadline - time.time()))}s).")
                return proc
        except Exception:
            pass
        time.sleep(5)
    print("".join(open(log_path).readlines()[-80:]))
    raise RuntimeError(f"vLLM did not become ready within {startup_timeout}s.")


def stop_vllm(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
    except Exception:
        pass


def run_inference(records, max_new_tokens, max_workers=4):
    """Feed system+user turns to the local server; assistant turn is the held-out GT."""
    def infer(rec):
        msgs = [m for m in rec["messages"] if m["role"] in ("system", "user")]
        r = requests.post(f"http://localhost:{LOCAL_PORT}/invocations",
                          json={"messages": msgs, "max_tokens": max_new_tokens, "temperature": 0.0},
                          timeout=180)
        r.raise_for_status()
        text = clean_response(r.json()["choices"][0]["message"]["content"])
        return {"file_name": rec.get("file_name"), "pred": text,
                "ground_truth": rec["messages"][-1]["content"]}

    preds, errs = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(infer, r): i for i, r in enumerate(records)}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                preds.append(f.result())
            except Exception as e:
                errs.append(futs[f])
                print(f"  ERROR on record {futs[f]}: {e}")
            if n % 50 == 0:
                print(f"  {n}/{len(records)} done")
    print(f"Inference complete: {len(preds)} ok, {len(errs)} errors.")
    return preds, errs


def build_scored(preds, err_indices, records):
    """The list handed to score(): successful preds PLUS one empty-prediction
    placeholder per errored doc, so failed inferences count as false negatives
    instead of being dropped (dropping them shrinks the denominator and inflates F1,
    which could let a flaky checkpoint win --pick-best). `err_indices` index `records`.
    """
    return list(preds) + [
        {"file_name": records[i].get("file_name"), "pred": "",
         "ground_truth": records[i]["messages"][-1]["content"]}
        for i in err_indices
    ]


def eval_one(tag, ckpt, records, args):
    """Serve, infer, score one checkpoint. Returns the MLflow-loggable metrics dict."""
    workdir = tempfile.mkdtemp()
    staged = os.path.join(workdir, "model")
    print(f"\n{'='*70}\n=== Evaluating {tag}  ({ckpt})\n{'='*70}")
    # Stage locally: vLLM may write cache into the model dir; never mutate the Volume.
    print(f"Staging weights -> {staged} ...")
    shutil.copytree(ckpt, staged)

    proc = None
    try:
        proc = start_vllm(staged, args.max_model_len, args.gpu_memory_util, workdir,
                          startup_timeout=args.startup_timeout)
        preds, errs = run_inference(records, args.max_new_tokens, args.max_workers)
    finally:
        if proc is not None:
            stop_vllm(proc)
            print("vLLM stopped.")

    if not preds and not errs:
        raise RuntimeError(f"{tag}: empty test set — nothing to score.")

    scored = build_scored(preds, errs, records)   # errored docs count as FN
    overall = score(scored)
    top8 = score(scored, TOP_8_FIELDS)
    print(f"  ALL : {overall}   ({len(preds)} ok, {len(errs)} errored -> FN)")
    print(f"  TOP8: {top8}")
    return overall, top8, len(preds), len(errs)


def main():
    ap = argparse.ArgumentParser(description="CLI vLLM eval of fine-tuned Qwen checkpoints.")
    ap.add_argument("--data-dir", required=True,
                    help="sweep data dir; test set at <data-dir>/test.jsonl")
    ap.add_argument("--checkpoints-dir", required=True,
                    help="checkpoint root; each checkpoint at <checkpoints-dir>/<tag>/")
    ap.add_argument("--experiment", required=True,
                    help="MLflow experiment ABSOLUTE path, e.g. /Users/you@databricks.com/qwen-ft-sweep")
    ap.add_argument("--tag", default=None,
                    help="single checkpoint tag under --checkpoints-dir; omit to evaluate ALL")
    ap.add_argument("--test-jsonl", default=None, help="override; default <data-dir>/test.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=3500)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--startup-timeout", type=int, default=1500)
    args = ap.parse_args()

    out_dir = args.checkpoints_dir.rstrip("/")
    test_jsonl = args.test_jsonl or os.path.join(args.data_dir, "test.jsonl")
    records = [json.loads(line) for line in open(test_jsonl)]
    print(f"Loaded {len(records)} test records from {test_jsonl}")

    if args.tag:
        tags = [args.tag]
    else:
        tags = sorted(d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d)))
    print(f"Will evaluate: {tags}")
    assert tags, f"No checkpoints found under {out_dir}"

    summary = evaluate_tags(tags, out_dir, records, args)

    print("\n=== Eval summary ===")
    for s in sorted(summary, key=lambda x: x["all_f1"] if x["all_f1"] == x["all_f1"] else -1.0,
                    reverse=True):
        if s.get("failed"):
            print(f"  {s['tag']}: FAILED (see log above)")
        else:
            print(f"  {s['tag']}: all_f1={s['all_f1']:.4f} top8_f1={s['top8_f1']:.4f} "
                  f"({s['docs']} docs, {s['errors']} errors)")

    n_ok = sum(1 for s in summary if not s.get("failed"))
    n_failed = len(summary) - n_ok
    print(f"\nLogged {n_ok} eval run(s) to MLflow experiment {args.experiment}"
          + (f"; {n_failed} checkpoint(s) FAILED" if n_failed else ""))
    # Non-zero only if EVERY checkpoint failed, so `air` flags a wholly-failed job
    # while a partial batch (some evals logged) still succeeds.
    return 1 if n_failed and n_ok == 0 else 0


def evaluate_tags(tags, out_dir, records, args):
    """Eval each tag, logging one stage=eval MLflow run per success. A checkpoint that
    fails (vLLM won't serve, weights missing, all requests error) is logged to stdout
    and SKIPPED — one bad checkpoint must not abort a bare-`--eval` batch of many.
    Returns a per-tag summary (failed entries carry {"failed": True})."""
    import mlflow
    mlflow.set_experiment(args.experiment)

    summary = []
    for tag in tags:
        ckpt = os.path.join(out_dir, tag)
        try:
            overall, top8, n_ok, n_err = eval_one(tag, ckpt, records, args)
        except Exception as e:
            print(f"  ! {tag}: eval FAILED ({e}) — skipping to the next checkpoint.")
            summary.append({"tag": tag, "all_f1": float("nan"), "top8_f1": float("nan"),
                            "docs": 0, "errors": 0, "failed": True})
            continue
        with mlflow.start_run(run_name=f"eval_{tag}"):
            mlflow.log_param("model_path", ckpt)
            mlflow.log_param("checkpoint_tag", tag)
            mlflow.set_tags({"sweep_id": "qwen-ft-sweep", "stage": "eval", "eval_via": "cli-air"})
            mlflow.log_metrics({f"all_{k}": v for k, v in overall.items()})
            mlflow.log_metrics({f"top8_{k}": v for k, v in top8.items()})
        summary.append({"tag": tag, "all_f1": overall["f1"], "top8_f1": top8["f1"],
                        "docs": n_ok, "errors": n_err})
    return summary


if __name__ == "__main__":
    sys.exit(main())
