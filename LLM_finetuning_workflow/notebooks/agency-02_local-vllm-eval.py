# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Local vLLM Batch Inference & Evaluation (NO serving endpoint)
# MAGIC
# MAGIC Test notebook for the **eval-without-deploying-an-endpoint** approach. It evaluates a
# MAGIC fine-tuned **Qwen3** entity-extraction checkpoint end-to-end on a **single GPU node**:
# MAGIC stage the weights from the UC Volume → launch a **local vLLM** OpenAI-compatible server
# MAGIC in the notebook → run batch inference over the test set with a thread pool → score
# MAGIC field-level precision / recall / F1 → log to MLflow.
# MAGIC
# MAGIC This notebook shares the 78-field extraction schema and evaluation logic with the
# MAGIC register/deploy notebook (`agency-05_deploy-endpoint-test.py`), but **removes** the
# MAGIC log-model / register-to-UC / create-serving-endpoint / `ai_query()` steps. Inference
# MAGIC instead goes directly to the local vLLM server (`/invocations`).
# MAGIC
# MAGIC **Why:** deploying a Model Serving endpoint is slow and costs a standing GPU replica.
# MAGIC For sweeps / quick checkpoint evals, local vLLM on the same node is faster and cheaper.
# MAGIC
# MAGIC 1. **Stage weights** — copy the HF-format checkpoint from the UC Volume to local disk
# MAGIC 2. **Local vLLM** — launch the server in-notebook and wait for `/health`
# MAGIC 3. **Batch inference** — thread-pooled `/invocations` calls over the test set
# MAGIC 4. **Evaluate** — field-level precision / recall / F1 vs ground truth
# MAGIC 5. **Log** — metrics + params to MLflow
# MAGIC
# MAGIC > **Compute:** run on a **GPU** notebook (interactive GPU cluster or Serverless GPU).
# MAGIC > An 8B model at bf16 is ~16 GB of weights + KV cache — use an A100/H100, or reduce
# MAGIC > `MAX_MODEL_LEN` on smaller GPUs.

# COMMAND ----------

# DBTITLE 1,Install serving dependencies
# vLLM + MLflow. Same pins as agency-05_deploy-endpoint-test.py (custom-LLM-serving starter).
%pip install vllm==0.11.2 transformers==4.57.6 mlflow==3.12.0 hf_transfer==0.1.9
%restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("DTYPE", "bfloat16", "Dtype")
dbutils.widgets.text("MAX_MODEL_LEN", "32768", "Max Model Len")
dbutils.widgets.text("learning_rate", "1e-5", "Learning Rate")
dbutils.widgets.text("num_epochs", "3", "Num Epochs")
dbutils.widgets.text("MAX_NEW_TOKENS", "3500", "Max New Tokens")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-finetuning-ai-runtime", "MLflow Experiment Path")

# COMMAND ----------

# DBTITLE 1,Set working directory to local disk
# /Workspace and /Volumes are not suitable as vLLM's --model dir for large weights;
# stage everything on the node's local disk.
import os
import tempfile

workdir = tempfile.mkdtemp()
os.chdir(workdir)
print("Working directory:", workdir)

# COMMAND ----------

# DBTITLE 1,Configuration
# === Configuration (from widgets / job parameters) ===
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")

# Run tag — matches notebook 01's checkpoint naming so eval finds the right weights
_lr_str = dbutils.widgets.get("learning_rate")
_ep_str = dbutils.widgets.get("num_epochs")
RUN_TAG = f"lr{_lr_str}_ep{_ep_str}"  # e.g., "lr1e-5_ep3"

# --- Source of the fine-tuned weights (HF-format checkpoint in the UC Volume) ----
# Point this at the checkpoint you want to evaluate (e.g. one of the sweep out/<tag>
# dirs). Must contain HF-format weights (config.json, *.safetensors, tokenizer files).
WEIGHTS_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-ft-final-{RUN_TAG}/"

ARTIFACTS_PATH = "qwen3"          # local dir the weights are copied to
SERVED_MODEL_NAME = "qwen"        # name vLLM exposes the model under

# --- vLLM tuning (from widgets) -----------------------------------------------
DTYPE = dbutils.widgets.get("DTYPE")  # match training dtype; faster on H100 (native Tensor Core)
MAX_MODEL_LEN = int(dbutils.widgets.get("MAX_MODEL_LEN"))  # 8B: weights ~16 GB + KV cache.
                                  # NOTE: reduce to 16384 on an A10 (24 GB).
GPU_MEMORY_UTILIZATION = 0.90

# Allowlisted ports for Serverless GPU notebooks are 3000-3999.
LOCAL_PORT = 3080

# --- Inference / eval (from widgets) ------------------------------------------
MAX_NEW_TOKENS = int(dbutils.widgets.get("MAX_NEW_TOKENS"))
MAX_WORKERS = 4                   # concurrent /invocations requests
OCR_CHAR_CAP = 100000             # far-out failsafe (~25k tokens) — normal docs are well under

# --- Eval data / prompt -------------------------------------------------------
TEST_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_test_v3"

# Optional: persist raw model outputs for inspection. Set to None to skip writing.
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.agency_inference_output_qwen3_local_vllm_{RUN_TAG.replace('-', '_')}"

# MLflow experiment for the eval metrics.
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

# Extraction prompt (same one used during fine-tuning; clean, no [INST] tags). Read from
# the Volume — the SAME file agency-00 baked into the train/val prompt column — so the
# eval prompt can never drift from the training prompt, and the path is stable regardless
# of where this notebook lives in the workspace.
PROMPT_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_prompt.txt"
with open(PROMPT_FILE, "r") as f:
    INSTRUCTION_PROMPT = f.read().strip()

# High-priority field subset (business-critical policy/file identifiers).
# Matches the reference eval_sweep_checkpoints notebook for comparable top8 metrics.
TOP_8_FIELDS = [
    "PolicyNumber", "OwnerFile", "LoanFile",
    "OwnerPolicyNumber", "OwnerPolicyAmount", "OwnerPolicyDate",
    "LoanPolicyNumber", "LoanPolicyAmount", "LoanPolicyDate",
]

print(f"Weights source: {WEIGHTS_VOLUME_PATH}")
print(f"Prompt loaded from: {PROMPT_FILE} ({len(INSTRUCTION_PROMPT)} chars)")

# COMMAND ----------

# DBTITLE 1,Stage weights from UC Volume to local disk
import shutil

# vLLM loads --model from a local directory. Copy the checkpoint off the Volume onto
# local disk. (dirs_exist_ok lets you re-run the cell.)
if not os.path.isdir(ARTIFACTS_PATH) or not os.listdir(ARTIFACTS_PATH):
    print(f"Copying {WEIGHTS_VOLUME_PATH} -> {ARTIFACTS_PATH} ...")
    shutil.copytree(WEIGHTS_VOLUME_PATH, ARTIFACTS_PATH, dirs_exist_ok=True)

files = os.listdir(ARTIFACTS_PATH)
print(f"Staged {len(files)} files in {ARTIFACTS_PATH}/")
assert any(f.startswith("config.json") for f in files), (
    f"No config.json in {ARTIFACTS_PATH}; is {WEIGHTS_VOLUME_PATH} an HF-format checkpoint?"
)

# COMMAND ----------

# DBTITLE 1,Patch tokenizer chat template — disable thinking by default
# The Qwen3 template injects '<think>\n\n</think>\n\n' as a PROMPT PREFIX only when
# enable_thinking=False is passed. vLLM 0.11.2 never passes that kwarg, so the model
# generates the think tags itself. Make the injection unconditional so API responses
# contain clean JSON with no <think> tags. (Idempotent; no-op if pattern absent.)
template_path = os.path.join(ARTIFACTS_PATH, "chat_template.jinja")
if os.path.isfile(template_path):
    with open(template_path, "r") as f:
        original = f.read()
    old_block = (
        "{%- if enable_thinking is defined and enable_thinking is false %}\n"
        "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
        "    {%- endif %}"
    )
    new_block = "    {{- '<think>\\n\\n</think>\\n\\n' }}"
    patched = original.replace(old_block, new_block)
    if original == patched:
        print("WARNING: conditional block not found — think tags will be stripped client-side instead.")
    else:
        with open(template_path, "w") as f:
            f.write(patched)
        print("Patched chat_template.jinja: <think>\\n\\n</think> is now an unconditional prompt prefix.")
else:
    print(f"No chat_template.jinja in {ARTIFACTS_PATH}; continuing (think tags stripped client-side).")

# COMMAND ----------

# DBTITLE 1,Define the vLLM entrypoint
# Single source of truth for the local vLLM launch command.
def entrypoint(port: int) -> str:
    args = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ]
    return " ".join(args)


print(entrypoint(LOCAL_PORT))

# COMMAND ----------

# DBTITLE 1,Start local vLLM server and wait for /health
import subprocess
import time
import requests

log_path = os.path.join(workdir, "vllm.log")
log_fh = open(log_path, "w")
proc = subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT)],
    stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True,
)
print(f"vLLM starting (pid={proc.pid}) on port {LOCAL_PORT} — polling /health (logs -> {log_path}) ...")

# Poll /health until ready. Generous timeout: vLLM may download CUDA packages, then
# load 8B weights + capture CUDA graphs before /health returns 200.
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
    # Surface the vLLM log so a startup failure is diagnosable here in the notebook.
    with open(log_path) as f:
        print("".join(f.readlines()[-120:]))
    raise RuntimeError(f"vLLM did not become ready within {STARTUP_TIMEOUT}s.")

# COMMAND ----------

# DBTITLE 1,Smoke test — single extraction request
# One real extraction through the local server, exactly as batch inference calls it.
sample_ocr = spark.table(TEST_TABLE).select("Raw_OCR_Content").limit(1).collect()[0][0]

resp = requests.post(
    f"http://localhost:{LOCAL_PORT}/invocations",
    json={
        "messages": [{"role": "user", "content": INSTRUCTION_PROMPT + "\n" + sample_ocr}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
    },
    timeout=180,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"][:1500])

# COMMAND ----------

# DBTITLE 1,Batch inference over the test set (thread-pooled local vLLM)
from concurrent.futures import ThreadPoolExecutor, as_completed

docs = (
    spark.table(TEST_TABLE)
    .select("File_Name", "Raw_OCR_Content")
    .toPandas()
    .to_dict("records")
)
print(f"Running inference on {len(docs)} test documents ...")


def infer(row):
    content = INSTRUCTION_PROMPT + "\n" + row["Raw_OCR_Content"][:OCR_CHAR_CAP]
    resp = requests.post(
        f"http://localhost:{LOCAL_PORT}/invocations",
        json={"messages": [{"role": "user", "content": content}],
              "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
        timeout=180,
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
            errors.append(fname)
            print(f"  ERROR {fname}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(docs)} done")

print(f"Inference complete: {len(results)} ok, {len(errors)} errors.")
if errors:
    print("Failed:", errors)
assert results, "No predictions produced — every inference request failed."

# COMMAND ----------

# DBTITLE 1,Stop the local vLLM server
# Free the GPU before the (CPU-only) scoring cells.
import signal

try:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    time.sleep(2)
    print("vLLM stopped.")
except Exception as e:
    print(f"Could not stop vLLM (may already be gone): {e}")

# COMMAND ----------

# DBTITLE 1,Optionally persist raw outputs for inspection
# Write model outputs to a Delta table so they can be inspected / re-scored later.
# Set OUTPUT_TABLE = None in Configuration to skip.
import pandas as pd

results_sdf = spark.createDataFrame(pd.DataFrame(results))
if OUTPUT_TABLE:
    results_sdf.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)
    print(f"Wrote {len(results)} outputs -> {OUTPUT_TABLE}")
    display(spark.table(OUTPUT_TABLE).limit(5))
else:
    results_sdf.createOrReplaceTempView("_local_vllm_outputs")
    print("OUTPUT_TABLE is None — outputs kept in temp view _local_vllm_outputs only.")

# COMMAND ----------

# DBTITLE 1,Evaluation Section
# MAGIC %md
# MAGIC ## Evaluation — field-level accuracy
# MAGIC
# MAGIC Compare model outputs against ground truth at the field level. Precision, recall,
# MAGIC and F1 via fuzzy string matching (SequenceMatcher ratio > 0.6). Identical scoring
# MAGIC to `agency-05_deploy-endpoint-test.py`.

# COMMAND ----------

# DBTITLE 1,Parse and flatten outputs
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType
import pyspark.sql.functions as F
import difflib

# Schema for the JSON extraction output — new CamelCase field set from agency_prompt.txt.
# Must stay in sync with agency_prompt.txt.
# Targets are sparse; from_json parses absent keys as null → fillna('NA') below.
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

# Parse model outputs — strip any residual <think>...</think> prefix before parsing JSON
# (safety net; the chat-template patch should already prevent think tags).
outputs_source = spark.table(OUTPUT_TABLE) if OUTPUT_TABLE else spark.table("_local_vllm_outputs")
outputs_pdf = (
    outputs_source
    .withColumn("clean_output", F.regexp_replace(col("model_output"), r"(?s)<think>.*?</think>\s*", ""))
    .withColumn("parsed", from_json(col("clean_output"), extraction_schema))
    .select("File_Name", "parsed.*")
    .toPandas()
)
outputs_melted = pd.melt(outputs_pdf, id_vars=['File_Name'], var_name='field', value_name='prediction')

# Parse ground truths
gt_pdf = (
    spark.table(TEST_TABLE)
    .withColumn("gt", from_json(col("ground_truths"), extraction_schema))
    .select("File_Name", "gt.*")
    .toPandas()
)
gt_melted = pd.melt(gt_pdf, id_vars=['File_Name'], var_name='field', value_name='ground_truth')

print(f"Predictions: {len(outputs_melted)} field values")
print(f"Ground truth: {len(gt_melted)} field values")

# COMMAND ----------

# DBTITLE 1,Compute field-level metrics
merged = pd.merge(
    gt_melted, outputs_melted,
    on=['File_Name', 'field'],
    how='inner'
).fillna('NA')


# Fuzzy matching: consider a match if SequenceMatcher ratio > 0.6
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


merged['result'] = merged.apply(lambda r: is_match(r['ground_truth'], r['prediction']), axis=1)

def compute_prf(df):
    """Compute precision, recall, F1 from a result column with TP/FP/FN/TN labels."""
    tp = (df['result'] == 'TP').sum()
    fp = (df['result'] == 'FP').sum()
    fn = (df['result'] == 'FN').sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# All fields
overall = compute_prf(merged)

# Top-8 high-priority fields only
top8_merged = merged[merged['field'].isin(TOP_8_FIELDS)]
top8 = compute_prf(top8_merged)

print("=== Overall Metrics (all fields) ===")
print(f"  Precision: {overall['precision']:.4f}")
print(f"  Recall:    {overall['recall']:.4f}")
print(f"  F1 Score:  {overall['f1']:.4f}")
print(f"  Total fields evaluated: {len(merged)}")
print(f"\n=== Top-8 Metrics ({len(TOP_8_FIELDS)} high-priority fields) ===")
print(f"  Precision: {top8['precision']:.4f}")
print(f"  Recall:    {top8['recall']:.4f}")
print(f"  F1 Score:  {top8['f1']:.4f}")
print(f"  Fields evaluated: {len(top8_merged)}")

# COMMAND ----------

# DBTITLE 1,Per-field accuracy breakdown
field_metrics = merged.groupby('field')['result'].apply(
    lambda x: pd.Series({
        'accuracy': ((x == 'TP') | (x == 'TN')).sum() / len(x),
        'tp': (x == 'TP').sum(),
        'fp': (x == 'FP').sum(),
        'fn': (x == 'FN').sum(),
        'tn': (x == 'TN').sum(),
    })
).unstack().sort_values('accuracy', ascending=False)

print("=== Per-Field Accuracy ===")
display(spark.createDataFrame(field_metrics.reset_index()))

# COMMAND ----------

# DBTITLE 1,Log metrics to MLflow
import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="eval-qwen-agency-local-vllm") as run:
    mlflow.log_metrics({f"all_{k}": v for k, v in overall.items()})
    mlflow.log_metrics({f"top8_{k}": v for k, v in top8.items()})
    mlflow.log_params({
        "weights_source": WEIGHTS_VOLUME_PATH,
        "inference": "local_vllm",   # no serving endpoint
        "test_table": TEST_TABLE,
        "matching_threshold": 0.6,
        "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "documents_scored": len(results),
        "checkpoint_tag": RUN_TAG,
        "learning_rate": _lr_str,
        "num_epochs": _ep_str,
    })
    mlflow.set_tags({"approach": "local-vllm-no-endpoint", "stage": "eval"})
    print(f"Metrics logged to MLflow run: {run.info.run_id}")