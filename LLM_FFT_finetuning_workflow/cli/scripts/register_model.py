#!/usr/bin/env python3
"""
register_model.py — register a swept checkpoint to the UC Model Registry as a
vLLM-served ChatModel. Runs ON an `air` GPU worker (via configs/register.air.yaml),
NOT your laptop: mlflow.register_model(env_pack=...) packages the WORKER's
environment (vLLM/CUDA) into the model version and tars ~16 GB of weights, which
needs the worker's RAM + the databricks_ai_v5 image (serverless CPU OOMs on the
tar). `run_sweep.py --register --only <tag>` submits this for you.

Mirrors the "Register model to Unity Catalog" cell of the notebook workflow's
agency-05, but reads the checkpoint from the CLI sweep's <checkpoints-dir>/<tag>/ and
disables Qwen3 thinking via the vLLM entrypoint (--chat-template-kwargs) rather than
agency-05's fragile chat_template.jinja string-patch — so served completions are
clean JSON with no leaked <think> block.

Registration only: it logs + registers a new UC version. It does NOT create a
serving endpoint. Requires databricks-sdk >= 0.102.0 (older has a 5-min upload
timeout that fails on ~16 GB env_pack tars).

    python register_model.py --checkpoints-dir /Volumes/.../checkpoints/qwen_sweep \\
        --tag lr1e-5_ep4 --uc-model-name catalog.schema.qwen3_8b_agency_ft
"""
import argparse
import os
import shutil
import tempfile

SERVED_MODEL_NAME = "qwen"     # name vLLM exposes the model under
ARTIFACTS_PATH = "qwen3"       # local (relative) dir the weights are staged to


def build_entrypoint(model_dir, port, max_model_len, gpu_mem_util, dtype="bfloat16"):
    """The vLLM OpenAI-server launch command Serving runs for this model. Includes
    --chat-template-kwargs to disable Qwen3 thinking so completions are clean JSON."""
    args = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir, "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(port), "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--enable-prefix-caching",
        "--chat-template-kwargs", '{"enable_thinking": false}',
    ]
    return " ".join(args)


def main():
    ap = argparse.ArgumentParser(description="Register a swept checkpoint to UC as a vLLM ChatModel.")
    ap.add_argument("--checkpoints-dir", required=True, help="checkpoint root; checkpoint at <checkpoints-dir>/<tag>/")
    ap.add_argument("--tag", required=True, help="checkpoint tag, e.g. lr1e-5_ep4")
    ap.add_argument("--uc-model-name", required=True, help="3-level UC model name catalog.schema.model")
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--gpu-memory-util", type=float, default=0.95)
    ap.add_argument("--serving-port", type=int, default=8080)
    args = ap.parse_args()

    ckpt = os.path.join(args.checkpoints_dir, args.tag)
    assert os.path.isfile(os.path.join(ckpt, "config.json")), (
        f"No config.json in {ckpt} — is it an HF-format checkpoint? "
        f"(Check the tag and that training wrote <checkpoints-dir>/{args.tag}/.)"
    )

    # Stage weights onto local disk (vLLM's --model needs a local dir; the Volume is
    # not suitable). chdir so ARTIFACTS_PATH is a stable relative dir the served
    # entrypoint can resolve.
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    print(f"Staging {ckpt} -> {os.path.join(workdir, ARTIFACTS_PATH)} ...")
    shutil.copytree(ckpt, ARTIFACTS_PATH)

    entrypoint = build_entrypoint(ARTIFACTS_PATH, args.serving_port,
                                  args.max_model_len, args.gpu_memory_util)
    print(f"vLLM entrypoint: {entrypoint}")

    import mlflow
    from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

    class LLMModel(ChatModel):
        # Serving runs the vLLM entrypoint (metadata), not predict(); this is a stub.
        def predict(self, context, messages, params):
            return ChatCompletionResponse.from_dict({"choices": []})

    model_info = mlflow.pyfunc.log_model(
        name=SERVED_MODEL_NAME,
        python_model=LLMModel(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        metadata={"task": "llm/v1/chat", "entrypoint": entrypoint},
    )
    print(f"Logged model URI: {model_info.model_uri}")

    mv = mlflow.register_model(
        model_info.model_uri,
        args.uc_model_name,
        env_pack="databricks_model_serving",
    )
    print(f"\nRegistered {args.uc_model_name} version {mv.version}  (checkpoint {args.tag})")
    print("env_pack processing (~16 GB tar) continues async; the version reaches "
          "READY in ~20-30 min. Deploy a serving endpoint on it separately.")


if __name__ == "__main__":
    main()
