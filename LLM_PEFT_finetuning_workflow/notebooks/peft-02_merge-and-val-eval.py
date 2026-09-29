# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# dependencies = [
#   "vllm==0.11.2",
#   "transformers==4.57.6",
#   "openai==2.17.0",
#   "mlflow==3.12.0",
#   "hf_transfer==0.1.9",
#   "\"databricks-sdk>=0.102.0\"",
# ]
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Merge LoRA Adapter + Local vLLM Validation Eval
# MAGIC
# MAGIC 1. **Merge** — load the **bf16** Llama 3.1 8B Instruct base, apply the adapter from notebook 01
# MAGIC    (`PeftModel`), `merge_and_unload()`, save the merged HF checkpoint to the Volume.
# MAGIC 2. **Validation eval** — launch a local vLLM server on the merged weights, run the
# MAGIC    **val** split, score field-level P/R/F1 exactly like the FFT workflow, log `stage=eval`.
# MAGIC
# MAGIC **Compute:** Serverless GPU **1×H100**, AI v5. No Unsloth here — vLLM's pinned stack only
# MAGIC (spec G1, G12).

# COMMAND ----------

# DBTITLE 1,Install vLLM serving stack (pass 1)
# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0"

# COMMAND ----------

# DBTITLE 1,Install peft + opencv pin (pass 2, no deps)
# peft: set to the `peft_version` param logged by notebook 01's MLflow run (spec G2).
#   --no-deps so it cannot drag in transformers>=5 (spec G12).
# opencv-python-headless 4.12: >=4.13 fails the FIPS self-test and aborts vLLM (spec G11).
%pip install --no-deps peft==0.17.1 opencv-python-headless==4.12.0.88
%restart_python

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

dbutils.widgets.text("run_tag", "lora_r16_lr2e-4_ep3", "Run tag (printed by notebook 01)")
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-peft-llama31", "MLflow Experiment Path")
dbutils.widgets.text("merge_base_model", "unsloth/Meta-Llama-3.1-8B-Instruct", "Merge base (bf16; HF id or /Volumes path)")
dbutils.widgets.text("max_model_len", "20480", "vLLM max model len")
dbutils.widgets.text("max_new_tokens", "3500", "Max new tokens")
dbutils.widgets.text("max_num_seqs", "14", "vLLM max concurrent seqs")
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
assert re.fullmatch(r"lora_r\d+_lr.+_ep\d+", RUN_TAG), (
    f"run_tag {RUN_TAG!r} does not look like notebook 01's RUN_TAG (e.g. lora_r16_lr2e-4_ep3)."
)
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
# Same cap as FFT. NOT a failsafe at MAX_MODEL_LEN=20480: a doc that long makes vLLM reject the
# request (HTTP 400) -> counted in inference_errors and scored as FN.
OCR_CHAR_CAP = 100000

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

print(f"run_tag={RUN_TAG}")
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
    """Normalize a model id for comparison (case and trailing slash)."""
    return name.lower().rstrip("/")


print(f"Adapter trained on: {ADAPTER_BASE}")
print(f"Merging into:       {MERGE_BASE_MODEL}")
if MERGE_BASE_MODEL.startswith("/") or ADAPTER_BASE.startswith("/"):
    print("WARNING: a base is a local/Volume path — cannot verify it matches the training base. "
          "Confirm it is the same Llama 3.1 8B Instruct weights (spec G10).")
else:
    assert model_family(ADAPTER_BASE) == model_family(MERGE_BASE_MODEL), (
        f"Adapter base {ADAPTER_BASE!r} and merge base {MERGE_BASE_MODEL!r} are different models (spec G10)."
    )

# COMMAND ----------

# DBTITLE 1,Merge the adapter into the bf16 base (skipped if already merged from THIS adapter)
import gc
import hashlib
import shutil

import pyspark.sql.functions as F
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def adapter_fingerprint(adapter_dir):
    """sha256 of the adapter weights — changes whenever notebook 01 retrains this run_tag."""
    h = hashlib.sha256()
    with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merged_is_current(merged_dir, adapter_dir):
    """True only for a COMPLETE merge of THIS adapter (the marker is written after the copy)."""
    marker = os.path.join(merged_dir, ".merged_from_adapter")
    if not os.path.isfile(marker):
        return False
    with open(marker) as f:
        return f.read().strip() == adapter_fingerprint(adapter_dir)


def adapter_chat_template(adapter_dir):
    """Chat template saved with the adapter, read as raw text (no cross-version tokenizer load)."""
    jinja = os.path.join(adapter_dir, "chat_template.jinja")
    if os.path.isfile(jinja):
        with open(jinja) as f:
            return f.read()
    cfg = os.path.join(adapter_dir, "tokenizer_config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            return json.load(f).get("chat_template")
    return None


if merged_is_current(MERGED_DIR, ADAPTER_DIR):
    print(f"Merged checkpoint at {MERGED_DIR} matches this adapter — staging it to local disk.")
    shutil.copytree(MERGED_DIR, LOCAL_MERGED, dirs_exist_ok=True)
else:
    if os.path.isdir(MERGED_DIR):
        print(f"{MERGED_DIR} is stale (adapter retrained) or incomplete — it will be replaced.")
    # Tokenizer from the merge base, NOT the adapter dir: the adapter's tokenizer files were written
    # by notebook 01's newer transformers and may not load under 4.57.6 (spec G2). It is the same
    # Llama 3.1 Instruct tokenizer; the chat templates are compared below.
    tokenizer = AutoTokenizer.from_pretrained(MERGE_BASE_MODEL)
    _adapter_template = adapter_chat_template(ADAPTER_DIR)
    if _adapter_template is None:
        print("WARNING: no chat template found in the adapter dir — cannot compare with the merge base's.")
    elif _adapter_template.strip() != (tokenizer.chat_template or "").strip():
        print("WARNING: the adapter's chat template differs from the merge base's — training and serving "
              "prompts may not match. Inspect both before trusting the eval.")
    _pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(MERGE_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)  # applies the adapter onto `base`
    peft_model.eval()

    # Sanity check on the SHORTEST val document (keeps HF generate fast).
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
            out = m.generate(_input_ids, max_new_tokens=256, do_sample=False, pad_token_id=_pad_id)
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
        # bf16 rounding in the merge can flip a greedy token even when the merge is correct.
        # The merged-model val F1 below is the authoritative check.
        print("WARNING: merged and unmerged greedy outputs differ (expected small drift, see comment).")

    merged.save_pretrained(LOCAL_MERGED, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(LOCAL_MERGED)
    print(f"Copying merged checkpoint -> {MERGED_DIR} (~16 GB) ...")
    if os.path.isdir(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)  # stale/partial merge of this run_tag — never mix shards
    shutil.copytree(LOCAL_MERGED, MERGED_DIR, dirs_exist_ok=True)
    # Marker LAST: its presence means the copy completed, its content names the adapter merged.
    with open(os.path.join(MERGED_DIR, ".merged_from_adapter"), "w") as f:
        f.write(adapter_fingerprint(ADAPTER_DIR))

    # Free the GPU for vLLM.
    del merged, peft_model, base, _input_ids
    gc.collect()
    torch.cuda.empty_cache()

_files = os.listdir(LOCAL_MERGED)
assert "config.json" in _files and any(f.endswith(".safetensors") for f in _files), (
    f"Merged checkpoint in {LOCAL_MERGED} is incomplete: {_files}"
)
print(f"Merged checkpoint ready locally at {LOCAL_MERGED} ({len(_files)} files); persisted at {MERGED_DIR}")

# COMMAND ----------

# DBTITLE 1,Define the vLLM entrypoint
def entrypoint(port: int) -> str:
    args = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", LOCAL_MERGED,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--enable-prefix-caching",  # the ~1.5K-token instruction prompt is shared by every request
    ]
    return " ".join(args)


print(entrypoint(LOCAL_PORT))

# COMMAND ----------

# DBTITLE 1,Start local vLLM server and wait for /health
import socket
import subprocess
import time

import requests


def ensure_port_free(port, timeout=60):
    """Fail if something still holds `port` (e.g. a vLLM server left over from an earlier run)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(1)
    raise AssertionError(
        f"Port {port} is still in use — a previous vLLM server may be running. "
        "Run `pkill -f vllm.entrypoints.openai.api_server` in a %sh cell and retry."
    )


# A vLLM server from an earlier failed/interrupted run survives %restart_python (own session);
# if it still held the port, /health would report the OLD model as ready and it would be scored.
subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
ensure_port_free(LOCAL_PORT)

log_path = os.path.join(workdir, "vllm.log")
log_fh = open(log_path, "w")
proc = subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT)],
    stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True,
)
print(f"vLLM starting (pid={proc.pid}) on port {LOCAL_PORT} — polling /health (logs -> {log_path}) ...")

STARTUP_TIMEOUT = 1500
deadline = time.time() + STARTUP_TIMEOUT
ready = False
while time.time() < deadline:
    if proc.poll() is not None:
        raise RuntimeError(f"vLLM exited during startup (code {proc.returncode}). See {log_path}.")
    try:
        if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).status_code == 200:
            ready = True
            print(f"vLLM is ready after {int(time.time() - (deadline - STARTUP_TIMEOUT))}s.")
            break
    except Exception:
        pass
    time.sleep(5)
if not ready:
    with open(log_path) as f:
        print("".join(f.readlines()[-120:]))
    raise RuntimeError(
        f"vLLM did not become ready within {STARTUP_TIMEOUT}s. A KV-cache/memory error means: "
        "lower max_model_len (e.g. 16384) or raise gpu_memory_utilization slightly."
    )

# COMMAND ----------

# DBTITLE 1,Smoke test — single extraction request
sample_ocr = spark.table(EVAL_TABLE).select("raw_ocr_content").limit(1).collect()[0][0]
resp = requests.post(
    f"http://localhost:{LOCAL_PORT}/invocations",
    json={
        "messages": [{"role": "user", "content": INSTRUCTION_PROMPT + "\n" + sample_ocr}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
    },
    timeout=REQUEST_TIMEOUT,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"][:1500])

# COMMAND ----------

# DBTITLE 1,Batch inference over the eval split (thread-pooled local vLLM)
from concurrent.futures import ThreadPoolExecutor, as_completed

docs = (
    spark.table(EVAL_TABLE)
    .selectExpr("file_name AS File_Name", "raw_ocr_content AS Raw_OCR_Content")
    .toPandas()
    .to_dict("records")
)
print(f"Running inference on {len(docs)} {EVAL_SPLIT} documents ...")


def infer(row):
    content = INSTRUCTION_PROMPT + "\n" + row["Raw_OCR_Content"][:OCR_CHAR_CAP]
    resp = requests.post(
        f"http://localhost:{LOCAL_PORT}/invocations",
        json={"messages": [{"role": "user", "content": content}],
              "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return {"File_Name": row["File_Name"],
            "model_output": resp.json()["choices"][0]["message"]["content"]}


results, errors = [], []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(infer, r): r["File_Name"] for r in docs}
    for i, fut in enumerate(as_completed(futures), 1):
        fname = futures[fut]
        try:
            results.append(fut.result())
        except Exception as e:
            errors.append(fname)  # scored as FN below (left join), never dropped
            print(f"  ERROR {fname}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(docs)} done")

print(f"Inference complete: {len(results)} ok, {len(errors)} errors.")
assert results, "No predictions produced — every inference request failed. See vllm.log."

# COMMAND ----------

# DBTITLE 1,Stop the local vLLM server
import signal

os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
time.sleep(2)
print("vLLM stopped.")

# COMMAND ----------

# DBTITLE 1,Persist raw outputs
import pandas as pd

spark.createDataFrame(pd.DataFrame(results)).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(OUTPUT_TABLE)
print(f"Wrote {len(results)} outputs -> {OUTPUT_TABLE}")
display(spark.table(OUTPUT_TABLE).limit(5))

# COMMAND ----------

# DBTITLE 1,Evaluation Section
# MAGIC %md
# MAGIC ## Evaluation — field-level accuracy (identical to the FFT workflow)
# MAGIC
# MAGIC Fuzzy match (SequenceMatcher ratio > 0.6), **left join on ground truth** so a failed
# MAGIC document counts as FN, mismatch → FP. Plus `json_parse_failures` / `inference_errors`.

# COMMAND ----------

# DBTITLE 1,Parse and flatten outputs
import difflib

from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StringType, StructField, StructType

# Copied verbatim from FFT agency-05_deploy-endpoint-test.py — keep in sync with agency_prompt.txt.
extraction_schema = StructType([
    StructField('ActualDocTitle', StringType()),
    StructField('SubType', StringType()),
    StructField('Type', StringType()),
    StructField('Scope', StringType()),
    StructField('TitleCompanyName', StringType()),
    StructField('Agentname', StringType()),
    StructField('EstateInterestType', StringType()),
    StructField('PolicyNumber', StringType()),
    StructField('OwnerFile', StringType()),
    StructField('LoanFile', StringType()),
    StructField('Order', StringType()),
    StructField('CommitmentNumber', StringType()),
    StructField('CommitmentEffectiveDate', StringType()),
    StructField('TitleNumber', StringType()),
    StructField('FARef', StringType()),
    StructField('OwnerPolicyNumber', StringType()),
    StructField('OwnerPolicyAmount', StringType()),
    StructField('OwnerPolicyDate', StringType()),
    StructField('LoanPolicyNumber', StringType()),
    StructField('LoanPolicyAmount', StringType()),
    StructField('LoanPolicyDate', StringType()),
    StructField('LoanNumber', StringType()),
    StructField('LoanRecordingDate', StringType()),
    StructField('LoanBook', StringType()),
    StructField('LoanPage', StringType()),
    StructField('LoanInstNumber', StringType()),
    StructField('DeedRecordingDate', StringType()),
    StructField('DeedBook', StringType()),
    StructField('DeedPage', StringType()),
    StructField('DeedInstNumber', StringType()),
    StructField('InsuredOrganizationName', StringType()),
    StructField('InsuredVestingBlob', StringType()),
    StructField('InsuredName0First', StringType()),
    StructField('InsuredName0Middle', StringType()),
    StructField('InsuredName0Last', StringType()),
    StructField('InsuredName0Suffix', StringType()),
    StructField('InsuredName1First', StringType()),
    StructField('InsuredName1Middle', StringType()),
    StructField('InsuredName1Last', StringType()),
    StructField('InsuredName1Suffix', StringType()),
    StructField('InsuredName2First', StringType()),
    StructField('InsuredName2Middle', StringType()),
    StructField('InsuredName2Last', StringType()),
    StructField('InsuredName3First', StringType()),
    StructField('InsuredName3Last', StringType()),
    StructField('BuyerOrganizationName', StringType()),
    StructField('BuyerVesting', StringType()),
    StructField('BuyerName0First', StringType()),
    StructField('BuyerName0Middle', StringType()),
    StructField('BuyerName0Last', StringType()),
    StructField('BuyerName0Suffix', StringType()),
    StructField('BuyerName1First', StringType()),
    StructField('BuyerName1Middle', StringType()),
    StructField('BuyerName1Last', StringType()),
    StructField('BuyerName1Suffix', StringType()),
    StructField('BuyerName2First', StringType()),
    StructField('BuyerName2Middle', StringType()),
    StructField('BuyerName2Last', StringType()),
    StructField('OwnerSellerOrganizationName', StringType()),
    StructField('OwnerSellerName0First', StringType()),
    StructField('OwnerSellerName0Middle', StringType()),
    StructField('OwnerSellerName0Last', StringType()),
    StructField('OwnerSellerName0Suffix', StringType()),
    StructField('OwnerSellerName1First', StringType()),
    StructField('OwnerSellerName1Middle', StringType()),
    StructField('OwnerSellerName1Last', StringType()),
    StructField('OwnerSellerName1Suffix', StringType()),
    StructField('OwnerSellerName2First', StringType()),
    StructField('OwnerSellerName2Middle', StringType()),
    StructField('OwnerSellerName2Last', StringType()),
    StructField('OwnerSellerName2Suffix', StringType()),
    StructField('SitusAddress', StringType()),
    StructField('SitusCity', StringType()),
    StructField('SitusState', StringType()),
    StructField('SitusZip', StringType()),
    StructField('FullLegal', StringType()),
    StructField('LegalCity', StringType()),
    StructField('LegalCounty', StringType()),
    StructField('LegalState', StringType()),
    StructField('Easementblob', StringType()),
    StructField('CCRBlob', StringType()),
    StructField('SubdivisionName0', StringType()),
    StructField('SubdivisionName1', StringType()),
    StructField('SubdivisionName2', StringType()),
    StructField('SubdivisionName3', StringType()),
    StructField('SubdivisionName4', StringType()),
    StructField('Lot0', StringType()),
    StructField('Lot1', StringType()),
    StructField('Lot2', StringType()),
    StructField('Lot3', StringType()),
    StructField('Lot4', StringType()),
    StructField('Block0', StringType()),
    StructField('Block1', StringType()),
    StructField('Block2', StringType()),
    StructField('Block3', StringType()),
    StructField('Block4', StringType()),
    StructField('Unit0', StringType()),
    StructField('Unit1', StringType()),
    StructField('Unit2', StringType()),
    StructField('Building0', StringType()),
    StructField('APN0', StringType()),
    StructField('APN1', StringType()),
    StructField('APN2', StringType()),
    StructField('APN3', StringType()),
    StructField('APN4', StringType()),
    StructField('APN5', StringType()),
    StructField('APN6', StringType()),
    StructField('MapBook0', StringType()),
    StructField('MapBook1', StringType()),
    StructField('MapBook2', StringType()),
    StructField('MapBook3', StringType()),
    StructField('MapBook4', StringType()),
    StructField('MapPage0', StringType()),
    StructField('MapPage1', StringType()),
    StructField('MapPage2', StringType()),
    StructField('MapPage3', StringType()),
    StructField('MapPage4', StringType()),
    StructField('Map_Document_Number_0', StringType()),
    StructField('Map_Document_Number_1', StringType()),
    StructField('Map_Document_Number_2', StringType()),
    StructField('Map_Document_Number_3', StringType()),
    StructField('Map_Document_Number_4', StringType()),
    StructField('Section0', StringType()),
    StructField('Section1', StringType()),
    StructField('Section2', StringType()),
    StructField('Range0', StringType()),
    StructField('Range1', StringType()),
    StructField('Range2', StringType()),
    StructField('Township0', StringType()),
    StructField('Township1', StringType()),
    StructField('Township2', StringType()),
    StructField('Quarter0', StringType()),
    StructField('Quarter1', StringType()),
    StructField('Quarter2', StringType()),
])

outputs_pdf = (
    spark.table(OUTPUT_TABLE)
    .withColumn("parsed", from_json(col("model_output"), extraction_schema))
    .select("File_Name", "parsed.*")
    .toPandas()
)
outputs_melted = pd.melt(outputs_pdf, id_vars=["File_Name"], var_name="field", value_name="prediction")

gt_pdf = (
    spark.table(EVAL_TABLE)
    .withColumn("gt", from_json(col("ground_truths"), extraction_schema))
    .selectExpr("file_name AS File_Name", "gt.*")
    .toPandas()
)
gt_melted = pd.melt(gt_pdf, id_vars=["File_Name"], var_name="field", value_name="ground_truth")


def is_json_object(s):
    try:
        return isinstance(json.loads(s), dict)
    except (TypeError, ValueError):
        return False


JSON_PARSE_FAILURES = sum(not is_json_object(r["model_output"]) for r in results)
INFERENCE_ERRORS = len(errors)
print(f"Predictions: {len(outputs_melted)} field values | Ground truth: {len(gt_melted)} field values")
print(f"json_parse_failures={JSON_PARSE_FAILURES}  inference_errors={INFERENCE_ERRORS}")

# COMMAND ----------

# DBTITLE 1,Compute field-level metrics
# LEFT join on GROUND TRUTH: a doc that errored/timed out has no prediction rows -> NaN ->
# 'NA' -> FN. (FFT notebook 05 and the CLI do the same; an inner join would inflate F1.)
merged = pd.merge(gt_melted, outputs_melted, on=["File_Name", "field"], how="left").fillna("NA")

N_DOCS = spark.table(EVAL_TABLE).count()
assert merged["File_Name"].nunique() == N_DOCS, (
    f"Scored {merged['File_Name'].nunique()} docs but {EVAL_TABLE} has {N_DOCS} — every doc must be scored."
)


def is_match(gt, pred, threshold=0.6):
    if gt == 'NA' and pred == 'NA':
        return 'TN'  # True Negative
    if gt == 'NA' and pred != 'NA':
        return 'FP'  # False Positive
    if gt != 'NA' and pred == 'NA':
        return 'FN'  # False Negative
    if difflib.SequenceMatcher(None, str(gt).lower(), str(pred).lower()).ratio() > threshold:
        return 'TP'  # True Positive
    return 'FP'  # Mismatch


def compute_prf(df):
    tp = (df["result"] == "TP").sum()
    fp = (df["result"] == "FP").sum()
    fn = (df["result"] == "FN").sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


merged["result"] = merged.apply(lambda r: is_match(r["ground_truth"], r["prediction"]), axis=1)
overall = compute_prf(merged)
top8 = compute_prf(merged[merged["field"].isin(TOP_8_FIELDS)])

print(f"=== {EVAL_SPLIT.upper()} metrics — {RUN_TAG} ({N_DOCS} docs) ===")
print(f"  all:  P {overall['precision']:.4f}  R {overall['recall']:.4f}  F1 {overall['f1']:.4f}")
print(f"  top8: P {top8['precision']:.4f}  R {top8['recall']:.4f}  F1 {top8['f1']:.4f}")

# COMMAND ----------

# DBTITLE 1,Per-field accuracy breakdown
field_metrics = merged.groupby("field")["result"].apply(
    lambda x: pd.Series({
        "accuracy": ((x == "TP") | (x == "TN")).sum() / len(x),
        "tp": (x == "TP").sum(),
        "fp": (x == "FP").sum(),
        "fn": (x == "FN").sum(),
        "tn": (x == "TN").sum(),
    })
).unstack().sort_values("accuracy", ascending=False)
display(spark.createDataFrame(field_metrics.reset_index()))

# COMMAND ----------

# DBTITLE 1,Log metrics to MLflow
import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)
with mlflow.start_run(run_name=f"{STAGE}_{RUN_TAG}") as run:
    mlflow.log_metrics({f"all_{k}": v for k, v in overall.items()})
    mlflow.log_metrics({f"top8_{k}": v for k, v in top8.items()})
    mlflow.log_metrics({"json_parse_failures": JSON_PARSE_FAILURES, "inference_errors": INFERENCE_ERRORS})
    mlflow.log_params({
        "eval_split": EVAL_SPLIT,
        "eval_table": EVAL_TABLE,
        "run_tag": RUN_TAG,
        "merged_dir": MERGED_DIR,
        "merge_base_model": MERGE_BASE_MODEL,
        "inference": "local_vllm",
        "matching_threshold": 0.6,
        "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "documents_scored": N_DOCS,
    })
    mlflow.set_tags({"stage": STAGE, "approach": "peft-lora"})
    print(f"Logged to MLflow run {run.info.run_id} (split={EVAL_SPLIT}, stage={STAGE})")