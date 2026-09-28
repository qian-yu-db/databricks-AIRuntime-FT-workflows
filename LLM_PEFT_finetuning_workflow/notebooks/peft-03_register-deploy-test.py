# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Register, Deploy (vLLM Custom LLM Serving) & Held-Out Test Eval
# MAGIC
# MAGIC Serves the merged PEFT checkpoint the same way FFT notebook 05 serves the FFT model:
# MAGIC an MLflow `ChatModel` whose `metadata.entrypoint` launches vLLM, registered with
# MAGIC `env_pack`, deployed as an `llm/v1/chat` endpoint. Then `ai_query()` over the **held-out
# MAGIC test** set — scored **once**, on the `run_tag` you selected on validation — logged `stage=test`.
# MAGIC
# MAGIC **Compute:** Serverless GPU **1×H100**, AI v5. Must be GPU: `env_pack` needs the RAM, and
# MAGIC logging from CPU packages CPU deps so the GPU endpoint fails to start (spec G6c, G13).

# COMMAND ----------

# DBTITLE 1,Install vLLM serving stack (pass 1)
# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0"

# COMMAND ----------

# DBTITLE 1,Install opencv pin (pass 2, no deps)
# opencv-python-headless >=4.13 fails the FIPS self-test and aborts vLLM (spec G11).
# MAGIC %pip install --no-deps opencv-python-headless==4.12.0.88
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Check pinned versions
import transformers
import vllm

print(f"transformers={transformers.__version__} vllm={vllm.__version__}")
assert transformers.__version__ == "4.57.6", (
    f"transformers is {transformers.__version__}; the AI v5 vLLM stack needs 4.57.6 (spec G12)."
)

# COMMAND ----------

dbutils.widgets.text("run_tag", "lora_r16_lr2e-4_ep3", "Run tag (best on validation)")
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-peft-llama31", "MLflow Experiment Path")
dbutils.widgets.text("endpoint_name", "agency-llama-peft-vllm", "Serving endpoint name")
dbutils.widgets.text("max_model_len", "20480", "vLLM max model len")
dbutils.widgets.text("max_num_seqs", "14", "vLLM max concurrent seqs at serving")
# Blank = register a NEW version. Set to an existing version to redeploy without re-registering
# (env_pack takes 20-30 min).
dbutils.widgets.text("model_version", "", "Existing UC model version (blank = register new)")

# COMMAND ----------

# DBTITLE 1,Configuration
import os
import re
import tempfile

from databricks.sdk.service.serving import ServingModelWorkloadType

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

RUN_TAG = dbutils.widgets.get("run_tag").strip()
assert re.fullmatch(r"lora_r\d+_lr.+_ep\d+", RUN_TAG), (
    f"run_tag {RUN_TAG!r} does not look like notebook 01's RUN_TAG."
)
TABLE_SUFFIX = re.sub(r"[^0-9A-Za-z_]", "_", RUN_TAG)
MERGED_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-merged-{RUN_TAG}"

workdir = tempfile.mkdtemp()
os.chdir(workdir)
ARTIFACTS_PATH = "llama31"  # local dir the merged weights are copied to (relative, as in FFT 05)
SERVED_MODEL_NAME = "llama"
LOCAL_PORT = 3080    # Serverless GPU notebooks allow 3000-3999 (spec G6d)
SERVING_PORT = 8080  # Custom LLM Serving requires 8080
MAX_MODEL_LEN = int(dbutils.widgets.get("max_model_len"))
MAX_NUM_SEQS = int(dbutils.widgets.get("max_num_seqs"))

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.llama31_8b_agency_peft"
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
# Same serving GPU as FFT notebook 05: bf16 8B weights (~16 GB) + KV cache at 20K context.
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_LARGE
WORKLOAD_SIZE = "Small"         # Beta: fixed replicas, no autoscaling (spec G14)
SCALE_TO_ZERO_ENABLED = False   # GPU endpoints without scale-to-zero may be deleted daily (G6c)
MODEL_VERSION_OVERRIDE = dbutils.widgets.get("model_version").strip()

# HELD-OUT TEST — scored once, on the run_tag selected on validation.
TEST_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_test_v3"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.agency_inference_output_llama_peft_{TABLE_SUFFIX}"

PROMPT_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_prompt.txt"
with open(PROMPT_FILE) as f:
    INSTRUCTION_PROMPT = f.read().strip()

TOP_8_FIELDS = [
    "PolicyNumber", "OwnerFile", "LoanFile",
    "OwnerPolicyNumber", "OwnerPolicyAmount", "OwnerPolicyDate",
    "LoanPolicyNumber", "LoanPolicyAmount", "LoanPolicyDate",
]
print(f"run_tag={RUN_TAG}\nweights: {MERGED_DIR}\nUC model: {UC_MODEL_NAME}\nendpoint: {ENDPOINT_NAME} ({WORKLOAD_TYPE.value})")

# COMMAND ----------

# DBTITLE 1,Stage merged weights from the Volume to local disk
import hashlib
import shutil

ADAPTER_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-adapter-{RUN_TAG}"


def adapter_fingerprint(adapter_dir):
    """sha256 of the adapter weights — changes whenever notebook 01 retrains this run_tag."""
    h = hashlib.sha256()
    with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merged_is_current(merged_dir, adapter_dir):
    """True only for a COMPLETE merge of THIS adapter (notebook 02 writes the marker after the copy)."""
    marker = os.path.join(merged_dir, ".merged_from_adapter")
    if not os.path.isfile(marker):
        return False
    with open(marker) as f:
        return f.read().strip() == adapter_fingerprint(adapter_dir)


assert merged_is_current(MERGED_DIR, ADAPTER_DIR), (
    f"{MERGED_DIR} is missing, incomplete, or was merged from an older adapter for run_tag={RUN_TAG}. "
    "Run notebook 02 for this run_tag first (it re-merges and re-validates)."
)
shutil.copytree(MERGED_DIR, ARTIFACTS_PATH, dirs_exist_ok=True)
print(f"Staged {len(os.listdir(ARTIFACTS_PATH))} files in {ARTIFACTS_PATH}/")

# COMMAND ----------

# DBTITLE 1,Define the vLLM entrypoint (single source for local test and serving)
def entrypoint(port: int, gpu_memory_utilization: float = 0.95) -> str:
    args = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--enable-prefix-caching",
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--disable-log-requests",
    ]
    return " ".join(args)


print("serving:", entrypoint(SERVING_PORT))

# COMMAND ----------

# DBTITLE 1,Local smoke test — start vLLM, one extraction, stop (pre-deploy check, spec G15)
import signal
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


# A leftover vLLM server (survives %restart_python) would answer /health with the OLD model.
subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
ensure_port_free(LOCAL_PORT)

log_path = os.path.join(workdir, "vllm.log")
proc = subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT, gpu_memory_utilization=0.90)],
    stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True,
)
STARTUP_TIMEOUT = 1500
deadline = time.time() + STARTUP_TIMEOUT
ready = False
while time.time() < deadline:
    if proc.poll() is not None:
        raise RuntimeError(f"vLLM exited during startup (code {proc.returncode}). See {log_path}.")
    try:
        if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).status_code == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(5)
if not ready:
    with open(log_path) as f:
        print("".join(f.readlines()[-120:]))
    raise RuntimeError(f"vLLM did not become ready within {STARTUP_TIMEOUT}s.")

sample_ocr = spark.table(TEST_TABLE).select("raw_ocr_content").limit(1).collect()[0][0]
resp = requests.post(
    f"http://localhost:{LOCAL_PORT}/invocations",
    json={"messages": [{"role": "user", "content": INSTRUCTION_PROMPT + "\n" + sample_ocr}],
          "max_tokens": 3500, "temperature": 0.0},
    timeout=600,
)
resp.raise_for_status()
_smoke = resp.json()["choices"][0]["message"]["content"]
print(_smoke[:1500])
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
time.sleep(2)
assert _smoke.strip().startswith("{"), "Local smoke output is not JSON — do not register this checkpoint."
print("Local smoke test passed; vLLM stopped.")

# COMMAND ----------

# DBTITLE 1,Register model to Unity Catalog (skipped if model_version is set)
import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel


class LLMModel(ChatModel):
    # Serving runs metadata.entrypoint (vLLM), not predict().
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


if MODEL_VERSION_OVERRIDE:
    DEPLOY_VERSION = MODEL_VERSION_OVERRIDE
    print(f"Skipping registration; deploying existing {UC_MODEL_NAME} v{DEPLOY_VERSION}")
else:
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name=f"register_{RUN_TAG}"):
        model_info = mlflow.pyfunc.log_model(
            name=SERVED_MODEL_NAME,
            python_model=LLMModel(),
            artifacts={"model_dir": ARTIFACTS_PATH},
            metadata={"task": "llm/v1/chat", "entrypoint": entrypoint(SERVING_PORT)},
        )
    model_version = mlflow.register_model(
        model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving"
    )
    DEPLOY_VERSION = str(model_version.version)
    MlflowClient().set_model_version_tag(UC_MODEL_NAME, DEPLOY_VERSION, "run_tag", RUN_TAG)
    print(f"✅ Registered {UC_MODEL_NAME} v{DEPLOY_VERSION} (run_tag={RUN_TAG})")

# COMMAND ----------

# DBTITLE 1,Wait for READY, then create or update the endpoint
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

print(f"Checking status of {UC_MODEL_NAME} v{DEPLOY_VERSION} ...")
for i in range(180):  # up to 30 minutes (env_pack of ~16 GB takes 20-30 min)
    status = w.model_versions.get(full_name=UC_MODEL_NAME, version=int(DEPLOY_VERSION)).status.value
    if status == "READY":
        print(f"Model version {DEPLOY_VERSION} is READY.")
        break
    if status != "PENDING_REGISTRATION":
        raise RuntimeError(f"Model version {DEPLOY_VERSION} entered status {status}. Re-run the registration cell.")
    if i % 6 == 0:
        print(f"  Status: {status} — waiting ({i * 10}s elapsed) ...")
    time.sleep(10)
else:
    raise TimeoutError(
        f"Model version {DEPLOY_VERSION} not READY after 30 min; env_pack may have failed. "
        "Re-run the registration cell to create a new version."
    )

served = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=DEPLOY_VERSION,
    workload_type=WORKLOAD_TYPE,
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
)
existing = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME), None)
if existing is None:
    print(f"Creating endpoint '{ENDPOINT_NAME}' with v{DEPLOY_VERSION} ...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served]),
        timeout=datetime.timedelta(minutes=40),
    )
else:
    print(f"Updating endpoint '{ENDPOINT_NAME}' to v{DEPLOY_VERSION} ...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME, served_entities=[served], timeout=datetime.timedelta(minutes=40)
    )
print("✅ Endpoint ready.")

# COMMAND ----------

# DBTITLE 1,Query the endpoint (smoke test)
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[ChatMessage(role=ChatMessageRole.USER, content=INSTRUCTION_PROMPT + "\n" + sample_ocr)],
    max_tokens=3500,
    temperature=0.0,
)
print(resp.choices[0].message.content[:1500])

# COMMAND ----------

# DBTITLE 1,Batch inference over the held-out test set with ai_query
# failOnError => false: one failed row (timeout, 5xx, over-length 400) must not abort the whole
# run. ai_query then returns struct<result, errorMessage>; failed rows get a NULL model_output,
# which from_json turns into all-NA -> scored as FN (same as notebook 02's failed requests).
# LEFT(..., 100000) is the FFT cap; at MAX_MODEL_LEN=20480 a doc that long errors instead.
escaped_prompt = INSTRUCTION_PROMPT.replace("'", "\\'")
spark.sql(f"""
CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
SELECT
  File_Name,
  ai_resp.result AS model_output,
  ai_resp.errorMessage AS error_message
FROM (
  SELECT
    file_name AS File_Name,
    ai_query(
      '{ENDPOINT_NAME}',
      CONCAT('{escaped_prompt}', '\\n', LEFT(raw_ocr_content, 100000)),
      modelParameters => named_struct('max_tokens', 3500, 'temperature', 0.0),
      failOnError => false
    ) AS ai_resp
  FROM {TEST_TABLE}
)
""")
INFERENCE_ERRORS = spark.table(OUTPUT_TABLE).where("error_message IS NOT NULL").count()
print(f"Batch inference complete -> {OUTPUT_TABLE}  (inference_errors={INFERENCE_ERRORS})")
display(spark.table(OUTPUT_TABLE).limit(5))

# COMMAND ----------

# DBTITLE 1,Evaluation Section
# MAGIC %md
# MAGIC ## Held-out test evaluation (identical to the FFT workflow)

# COMMAND ----------

# DBTITLE 1,Parse and flatten outputs
import difflib
import json

import pandas as pd
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
    spark.table(TEST_TABLE)
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


# Only over rows the endpoint answered; failed rows are counted in INFERENCE_ERRORS instead.
JSON_PARSE_FAILURES = int(sum(
    not is_json_object(s)
    for s in spark.table(OUTPUT_TABLE).where("error_message IS NULL").select("model_output").toPandas()["model_output"]
))
print(f"json_parse_failures={JSON_PARSE_FAILURES}  inference_errors={INFERENCE_ERRORS}")

# COMMAND ----------

# DBTITLE 1,Compute field-level metrics
# LEFT join on GROUND TRUTH so a missing/failed doc counts as FN (same as FFT notebook 05).
merged = pd.merge(gt_melted, outputs_melted, on=["File_Name", "field"], how="left").fillna("NA")

N_DOCS = spark.table(TEST_TABLE).count()
assert merged["File_Name"].nunique() == N_DOCS, (
    f"Scored {merged['File_Name'].nunique()} docs but {TEST_TABLE} has {N_DOCS} — every doc must be scored."
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

print(f"=== HELD-OUT TEST — {RUN_TAG} ({N_DOCS} docs) ===")
print(f"  all:  P {overall['precision']:.4f}  R {overall['recall']:.4f}  F1 {overall['f1']:.4f}")
print(f"  top8: P {top8['precision']:.4f}  R {top8['recall']:.4f}  F1 {top8['f1']:.4f}")

# COMMAND ----------

# DBTITLE 1,Log HELD-OUT TEST metrics to MLflow (stage=test)
mlflow.set_experiment(EXPERIMENT_PATH)
with mlflow.start_run(run_name=f"test_{RUN_TAG}") as _test_run:
    mlflow.log_metrics({f"all_{k}": v for k, v in overall.items()})
    mlflow.log_metrics({f"top8_{k}": v for k, v in top8.items()})
    mlflow.log_metrics({"json_parse_failures": JSON_PARSE_FAILURES, "inference_errors": INFERENCE_ERRORS})
    mlflow.log_params({
        "eval_split": "test",
        "eval_table": TEST_TABLE,
        "run_tag": RUN_TAG,
        "uc_model_name": UC_MODEL_NAME,
        "uc_model_version": DEPLOY_VERSION,
        "inference": "serving_endpoint_ai_query",
        "matching_threshold": 0.6,
        "documents_scored": N_DOCS,
    })
    mlflow.set_tags({"approach": "held-out-test", "stage": "test"})
    print(f"Held-out TEST metrics logged (stage=test) to run {_test_run.info.run_id}")

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
