# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Merge LoRA Adapter + Local vLLM Validation Eval
# MAGIC
# MAGIC 1. **Merge** — load the **bf16** Llama 3.1 8B Instruct base, apply the adapter from notebook 01
# MAGIC    (`PeftModel`), `merge_and_unload()`, save the merged HF checkpoint to the Volume.
# MAGIC    Always merges into bf16 — also for `qlora_4bit` adapters — so serving is identical.
# MAGIC 2. **Validation eval** — launch a local vLLM server on the merged weights, run the
# MAGIC    **val** split, score field-level P/R/F1 exactly like the FFT workflow, log `stage=eval`.
# MAGIC
# MAGIC **Compute:** Serverless GPU, AI v5, **A10 or H100** (H100 is faster; on A10 keep
# MAGIC `max_num_seqs=2`). No Unsloth here — vLLM's pinned stack only (spec G1, G12).

# COMMAND ----------

# DBTITLE 1,Install vLLM serving stack (pass 1)
# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0"

# COMMAND ----------

# DBTITLE 1,Install peft + opencv pin (pass 2, no deps)
# peft: set to the `peft_version` param logged by notebook 01's MLflow run (spec G2).
#   --no-deps so it cannot drag in transformers>=5 (spec G12).
# opencv-python-headless 4.12: >=4.13 fails the FIPS self-test and aborts vLLM (spec G11).
# MAGIC %pip install --no-deps peft==0.17.1 opencv-python-headless==4.12.0.88
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Check pinned versions
import peft
import transformers
import vllm

print(f"transformers={transformers.__version__} peft={peft.__version__} vllm={vllm.__version__}")
assert transformers.__version__ == "4.57.6", (
    f"transformers was changed to {transformers.__version__}; the AI v5 vLLM stack needs 4.57.6 (spec G12). "
    "Re-run the install cells; make sure peft is installed with --no-deps."
)

# COMMAND ----------

dbutils.widgets.text("run_tag", "qlora_4bit_r16_lr2e-4_ep3", "Run tag (printed by notebook 01)")
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-peft-llama31", "MLflow Experiment Path")
dbutils.widgets.text("merge_base_model", "unsloth/Meta-Llama-3.1-8B-Instruct", "Merge base (bf16; HF id or /Volumes path)")
dbutils.widgets.text("max_model_len", "20480", "vLLM max model len")
dbutils.widgets.text("max_new_tokens", "3500", "Max new tokens")
dbutils.widgets.text("max_num_seqs", "2", "vLLM max concurrent seqs (2 on A10, up to 14 on H100)")
dbutils.widgets.text("gpu_memory_utilization", "0.90", "vLLM GPU memory utilization")
dbutils.widgets.text("request_timeout", "600", "Per-request timeout (s)")
# val = SELECTION eval (stage=eval). test only for a deliberate one-off (stage=test).
dbutils.widgets.text("eval_split", "val", "Eval split (val|test)")

# COMMAND ----------

# DBTITLE 1,Configuration
import os
import re
import tempfile

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

RUN_TAG = dbutils.widgets.get("run_tag").strip()
_m = re.fullmatch(r"(qlora_4bit|lora_bf16)_r\d+_lr.+_ep\d+", RUN_TAG)
assert _m, f"run_tag {RUN_TAG!r} does not look like notebook 01's RUN_TAG (e.g. qlora_4bit_r16_lr2e-4_ep3)."
TRAIN_MODE = _m.group(1)
TABLE_SUFFIX = re.sub(r"[^0-9A-Za-z_]", "_", RUN_TAG)

ADAPTER_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-adapter-{RUN_TAG}"
MERGED_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-merged-{RUN_TAG}"
MERGE_BASE_MODEL = dbutils.widgets.get("merge_base_model").strip()

# vLLM needs the weights on local disk, not /Volumes.
workdir = tempfile.mkdtemp()
os.chdir(workdir)
LOCAL_MERGED = os.path.join(workdir, "merged")

SERVED_MODEL_NAME = "llama"
LOCAL_PORT = 3080  # Serverless GPU notebooks allow ports 3000-3999 (spec G6d)
MAX_MODEL_LEN = int(dbutils.widgets.get("max_model_len"))
MAX_NEW_TOKENS = int(dbutils.widgets.get("max_new_tokens"))
MAX_NUM_SEQS = int(dbutils.widgets.get("max_num_seqs"))
GPU_MEMORY_UTILIZATION = float(dbutils.widgets.get("gpu_memory_utilization"))
REQUEST_TIMEOUT = int(dbutils.widgets.get("request_timeout"))
MAX_WORKERS = 4
OCR_CHAR_CAP = 100000  # far-out failsafe (~25k tokens), same as FFT

EVAL_SPLIT = dbutils.widgets.get("eval_split").strip()
assert EVAL_SPLIT in {"val", "test"}, f"eval_split must be 'val' or 'test', got {EVAL_SPLIT!r}"
STAGE = "eval" if EVAL_SPLIT == "val" else "test"
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_{EVAL_SPLIT}_v3"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.agency_inference_output_llama_peft_local_vllm_{EVAL_SPLIT}_{TABLE_SUFFIX}"

# Same prompt file FFT notebook 00 baked into the train/val `prompt` column.
PROMPT_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_prompt.txt"
with open(PROMPT_FILE) as f:
    INSTRUCTION_PROMPT = f.read().strip()

TOP_8_FIELDS = [
    "PolicyNumber", "OwnerFile", "LoanFile",
    "OwnerPolicyNumber", "OwnerPolicyAmount", "OwnerPolicyDate",
    "LoanPolicyNumber", "LoanPolicyAmount", "LoanPolicyDate",
]

print(f"run_tag={RUN_TAG} (train_mode={TRAIN_MODE})")
print(f"adapter:  {ADAPTER_DIR}\nmerged -> {MERGED_DIR}")
print(f"eval split: {EVAL_SPLIT} -> {EVAL_TABLE} (MLflow stage={STAGE})")

# COMMAND ----------

# DBTITLE 1,Check the adapter and that the merge base matches the training base
import json

for _f in ("adapter_config.json", "adapter_model.safetensors"):
    assert os.path.isfile(os.path.join(ADAPTER_DIR, _f)), f"Missing {_f} in {ADAPTER_DIR} — run notebook 01 first."

with open(os.path.join(ADAPTER_DIR, "adapter_config.json")) as f:
    ADAPTER_BASE = json.load(f).get("base_model_name_or_path", "")


def model_family(name):
    """Normalize a mirror id so a 4-bit repo and its bf16 source compare equal."""
    n = name.lower().rstrip("/")
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


print(f"Adapter trained on: {ADAPTER_BASE}")
print(f"Merging into:       {MERGE_BASE_MODEL}")
assert "bnb-4bit" not in MERGE_BASE_MODEL.lower(), "merge_base_model must be the bf16 base, not a 4-bit repo."
if MERGE_BASE_MODEL.startswith("/") or ADAPTER_BASE.startswith("/"):
    print("WARNING: a base is a local/Volume path — cannot verify it matches the training base. "
          "Confirm it is the same Llama 3.1 8B Instruct weights (spec G10).")
else:
    assert model_family(ADAPTER_BASE) == model_family(MERGE_BASE_MODEL), (
        f"Adapter base {ADAPTER_BASE!r} and merge base {MERGE_BASE_MODEL!r} are different models (spec G10)."
    )

# COMMAND ----------

# DBTITLE 1,Merge the adapter into the bf16 base (skipped if already merged)
import gc
import shutil

import pyspark.sql.functions as F
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

if os.path.isfile(os.path.join(MERGED_DIR, "config.json")):
    print(f"Merged checkpoint already exists at {MERGED_DIR} — staging it to local disk.")
    shutil.copytree(MERGED_DIR, LOCAL_MERGED, dirs_exist_ok=True)
else:
    # Tokenizer from the adapter dir (carries chat template + pad token); re-saved below by
    # THIS env's transformers so vLLM reads a tokenizer written by the version it runs (G2).
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(MERGE_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)  # uses `base`, ignores adapter's 4-bit base id
    peft_model.eval()

    # Sanity check on the SHORTEST val document (keeps HF generate fast on an A10).
    _sample_ocr = (
        spark.table(f"{CATALOG}.{SCHEMA}.agency_ft_dataset_val_v3")
        .orderBy(F.length("raw_ocr_content"))
        .select("raw_ocr_content")
        .limit(1)
        .collect()[0][0]
    )
    _input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION_PROMPT + "\n" + _sample_ocr}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    def greedy(m):
        with torch.no_grad():
            out = m.generate(_input_ids, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, _input_ids.shape[1]:], skip_special_tokens=True)

    unmerged_out = greedy(peft_model)  # BEFORE merge_and_unload (it mutates the model)
    merged = peft_model.merge_and_unload()
    merged_out = greedy(merged)
    print("UNMERGED:", unmerged_out[:400])
    print("MERGED:  ", merged_out[:400])
    assert unmerged_out.strip().startswith("{") and merged_out.strip().startswith("{"), (
        "Adapter output is not JSON — check the adapter / training run before merging."
    )
    if unmerged_out != merged_out:
        # bf16 rounding can flip a greedy token; for qlora_4bit the bf16 base also differs from
        # the 4-bit training base. The merged-model val F1 below is the authoritative check.
        print("WARNING: merged and unmerged greedy outputs differ (expected small drift, see comment).")

    merged.save_pretrained(LOCAL_MERGED, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(LOCAL_MERGED)
    print(f"Copying merged checkpoint -> {MERGED_DIR} (~16 GB) ...")
    shutil.copytree(LOCAL_MERGED, MERGED_DIR, dirs_exist_ok=True)

    # Free the GPU for vLLM.
    del merged, peft_model, base, _input_ids
    gc.collect()
    torch.cuda.empty_cache()

_files = os.listdir(LOCAL_MERGED)
assert "config.json" in _files and any(f.endswith(".safetensors") for f in _files), (
    f"Merged checkpoint in {LOCAL_MERGED} is incomplete: {_files}"
)
print(f"Merged checkpoint ready locally at {LOCAL_MERGED} ({len(_files)} files); persisted at {MERGED_DIR}")
