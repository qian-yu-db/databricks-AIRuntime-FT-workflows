# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Evaluate ALL sweep checkpoints — local vLLM, no serving endpoint
# MAGIC
# MAGIC The **recommended eval path for the `cli` fine-tuning workflow on a FIPS workspace.**
# MAGIC Trains via the `air` CLI (`run_sweep.py`), then evaluates the resulting checkpoints
# MAGIC **here**, because a CLI eval (vLLM on serverless GPU) fails the OpenSSL FIPS
# MAGIC self-test on `fevm-classic-stable` and was removed (see `../GOTCHAS.md`, §2).
# MAGIC
# MAGIC This notebook loops over every `out/<tag>/` checkpoint the sweep produced and, for
# MAGIC each: stages the weights → launches a local vLLM server → runs threaded inference
# MAGIC over `test.jsonl` → scores per-field precision/recall/F1 → logs an `eval_<tag>` run
# MAGIC to MLflow tagged `stage=eval`.
# MAGIC
# MAGIC It reads the sweep's `test.jsonl` and logs `eval_<tag>` runs tagged `stage=eval`
# MAGIC (per-field P/R/F1 + top-8), so `run_sweep.py --pick-best` ranks these runs unchanged.
# MAGIC Serving mechanics mirror `notebooks/agency-02_local-vllm-eval.py` (repo root).
# MAGIC
# MAGIC > **Compute:** run on an **interactive GPU cluster** (NOT serverless — that's the
# MAGIC > FIPS-hardened env that crashes vLLM). A100/H100 for 8B; reduce `MAX_MODEL_LEN` on
# MAGIC > smaller GPUs.

# COMMAND ----------

# DBTITLE 1,Install serving dependencies
%pip install vllm==0.11.2 transformers==4.57.6 mlflow==3.12.0 hf_transfer==0.1.9
%restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
import os

# Must match grid.yaml's data_dir. Checkpoints are under <DATA_DIR>/out/<tag>/, and
# the held-out test set is <DATA_DIR>/test.jsonl (both written by the sweep).
DATA_DIR = "/Volumes/fins_genai/fine_tuning/training_data/qwen_sweep"
OUT_DIR = f"{DATA_DIR}/out"
TEST_JSONL = f"{DATA_DIR}/test.jsonl"

# MLflow experiment — full workspace path (the CLI uses the slug 'qwen-ft-sweep'; air
# resolves it under the user's home). Point at the same experiment the sweep logs to.
EXPERIMENT_PATH = "/Users/q.yu@databricks.com/qwen-ft-sweep"

# Optionally restrict which checkpoints to eval (list of tags). Empty = all under out/.
ONLY_TAGS = []

# --- vLLM tuning -------------------------------------------------------------
SERVED_MODEL_NAME = "qwen"
DTYPE = "bfloat16"
MAX_MODEL_LEN = 32768             # reduce to 16384 on an A10 (24 GB)
GPU_MEMORY_UTILIZATION = 0.90
LOCAL_PORT = 3080
STARTUP_TIMEOUT = 1500            # vLLM downloads CUDA pkgs + loads 8B before /health

# --- Inference / eval ---------------------------------------------------------
MAX_NEW_TOKENS = 3500
MAX_WORKERS = 4

# High-priority field subset (new CamelCase schema) — the policy/file identifiers
# and owner/loan policy number/amount/date that matter most to the business.
TOP_8_FIELDS = [
    "PolicyNumber", "OwnerFile", "LoanFile",
    "OwnerPolicyNumber", "OwnerPolicyAmount", "OwnerPolicyDate",
    "LoanPolicyNumber", "LoanPolicyAmount", "LoanPolicyDate",
]

# Discover the checkpoints to evaluate.
all_tags = sorted(d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d)))
tags = [t for t in all_tags if not ONLY_TAGS or t in ONLY_TAGS]
print(f"Checkpoints under {OUT_DIR}: {all_tags}")
print(f"Will evaluate: {tags}")
assert tags, f"No checkpoints found under {OUT_DIR} (run the training sweep first)."

# COMMAND ----------

# DBTITLE 1,Load the test set once (shared across all checkpoints)
import json

records = [json.loads(line) for line in open(TEST_JSONL)]
print(f"Loaded {len(records)} test records from {TEST_JSONL}")

# COMMAND ----------

# DBTITLE 1,Helpers — vLLM serving, inference, scoring
import shutil
import signal
import subprocess
import time
import tempfile
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def patch_chat_template(model_dir):
    """Make <think>\\n\\n</think> an unconditional prompt prefix so responses are clean
    JSON. No-op if the pattern isn't present."""
    p = os.path.join(model_dir, "chat_template.jinja")
    if not os.path.isfile(p):
        return
    tmpl = open(p).read()
    old = ("{%- if enable_thinking is defined and enable_thinking is false %}\n"
           "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
           "    {%- endif %}")
    new = "    {{- '<think>\\n\\n</think>\\n\\n' }}"
    patched = tmpl.replace(old, new)
    if patched != tmpl:
        open(p, "w").write(patched)


def start_vllm(model_dir, workdir):
    """Launch vLLM and poll /health. Raises (with the tail of the vLLM log) on failure."""
    cmd = " ".join([
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir, "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(LOCAL_PORT), "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ])
    log_path = os.path.join(workdir, "vllm.log")
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            print("".join(open(log_path).readlines()[-80:]))
            raise RuntimeError(f"vLLM exited during startup (code {proc.returncode}).")
        try:
            if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(5)
    print("".join(open(log_path).readlines()[-80:]))
    raise RuntimeError(f"vLLM did not become ready within {STARTUP_TIMEOUT}s.")


def stop_vllm(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
    except Exception:
        pass


def run_inference(recs):
    """Feed system+user turns to the local server; assistant turn is the held-out GT."""
    def infer(rec):
        msgs = [m for m in rec["messages"] if m["role"] in ("system", "user")]
        r = requests.post(f"http://localhost:{LOCAL_PORT}/invocations",
                          json={"messages": msgs, "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
                          timeout=180)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].replace("```json", "").replace("```", "").strip()
        return {"file_name": rec.get("file_name"), "pred": text, "ground_truth": rec["messages"][-1]["content"]}

    preds, errs = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(infer, r): i for i, r in enumerate(recs)}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                preds.append(f.result())
            except Exception as e:
                errs.append(futs[f])
                print(f"  ERROR on record {futs[f]}: {e}")
    return preds, errs


# --- Scoring (Spark-free difflib; FN for present-but-wrong, case-sensitive) ---
# The pure scoring helpers live in cli/lib/extract_eval.py so they can be
# unit-tested offline; the notebook imports them so tests exercise the real code.
import sys


def _bootstrap_lib():
    """Add the sibling lib/ dir to sys.path (local + Databricks)."""
    here = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # plain python
    except NameError:
        here = os.getcwd()  # Databricks notebook: no __file__
    for base in (here, os.path.dirname(here), os.getcwd()):
        cand = os.path.join(base, "lib")
        if os.path.isfile(os.path.join(cand, "extract_eval.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return
        # also handle running from repo root with the full path
        cand = os.path.join(base, "LLM_finetuning_workflow", "cli", "lib")
        if os.path.isfile(os.path.join(cand, "extract_eval.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return


_bootstrap_lib()
from extract_eval import parse_json, _is_na, _similar, score  # noqa: E402,F401

print("Helpers imported from lib/extract_eval.py.")

# COMMAND ----------

# DBTITLE 1,Evaluate every checkpoint (loop)
import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)
summary = []

for tag in tags:
    ckpt = os.path.join(OUT_DIR, tag)
    print(f"\n{'='*70}\n=== Evaluating {tag}  ({ckpt})\n{'='*70}")
    workdir = tempfile.mkdtemp()
    staged = os.path.join(workdir, "model")

    # vLLM may write to the model dir (cache) and we patch the template — stage locally
    # so the shared Volume copy is never mutated.
    print(f"Staging weights -> {staged} ...")
    shutil.copytree(ckpt, staged)
    patch_chat_template(staged)

    proc = None
    try:
        print("Starting vLLM ...")
        proc = start_vllm(staged, workdir)
        print("vLLM ready. Running inference ...")
        preds, errs = run_inference(records)
        print(f"Inference: {len(preds)} ok, {len(errs)} errors.")
    finally:
        if proc is not None:
            stop_vllm(proc)
            print("vLLM stopped.")

    if not preds:
        print(f"!! {tag}: no predictions — skipping (all requests failed).")
        continue

    overall = score(preds)
    top8 = score(preds, TOP_8_FIELDS)
    print(f"  ALL : {overall}")
    print(f"  TOP8: {top8}")

    with mlflow.start_run(run_name=f"eval_{tag}"):
        mlflow.log_param("model_path", ckpt)
        mlflow.log_param("checkpoint_tag", tag)
        mlflow.set_tags({"sweep_id": "qwen-ft-sweep", "stage": "eval", "eval_via": "notebook-local-vllm"})
        mlflow.log_metrics({f"all_{k}": v for k, v in overall.items()})
        mlflow.log_metrics({f"top8_{k}": v for k, v in top8.items()})

    summary.append({"tag": tag, "all_f1": overall["f1"], "top8_f1": top8["f1"],
                    "precision": overall["precision"], "recall": overall["recall"],
                    "docs": len(preds), "errors": len(errs)})

print("\nAll checkpoints evaluated.")

# COMMAND ----------

# DBTITLE 1,Ranking summary
import pandas as pd

if summary:
    df = pd.DataFrame(summary).sort_values("all_f1", ascending=False).reset_index(drop=True)
    print("=== Sweep eval ranking (by all_f1) ===")
    display(spark.createDataFrame(df))
    best = df.iloc[0]
    print(f"\nWinner: {best['tag']}  all_f1={best['all_f1']:.4f}  top8_f1={best['top8_f1']:.4f}")
    print("(Runs are tagged stage=eval, so `run_sweep.py --pick-best` also ranks them.)")
else:
    print("No successful evals to rank.")
