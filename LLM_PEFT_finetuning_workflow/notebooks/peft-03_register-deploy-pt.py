# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# dependencies = [
#   "transformers==4.57.6",
#   "mlflow==3.12.0",
#   "hf_transfer==0.1.9",
#   "\"databricks-sdk>=0.102.0\"",
# ]
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Merge, Register & Deploy — Provisioned Throughput
# MAGIC
# MAGIC Alternative deployment path for the PEFT fine-tuned Llama 3.1 8B Instruct model.
# MAGIC Instead of Custom LLM Serving with vLLM (notebook 03), this notebook registers the
# MAGIC merged weights via `mlflow.transformers.log_model(task="llm/v1/chat")` and deploys
# MAGIC on a **Provisioned Throughput** endpoint — Databricks-managed optimized inference with
# MAGIC autoscaling support.
# MAGIC
# MAGIC | | Notebook 03 (Custom vLLM) | This Notebook (Provisioned Throughput) |
# MAGIC |---|---|---|
# MAGIC | Inference engine | Self-managed vLLM via `entrypoint` | Databricks-managed optimized engine |
# MAGIC | Registration | `pyfunc` ChatModel + `env_pack` (20-30 min) | `transformers` flavor — no env_pack |
# MAGIC | Deployment | `GPU_LARGE`, fixed `Small` replicas | `min/max_provisioned_throughput` (tokens/s) |
# MAGIC | Autoscaling | Not available | Supported |
# MAGIC
# MAGIC **Compute:** Serverless GPU **1×H100**, AI v5. Needed for the merge step (~16 GB bf16
# MAGIC weights) and `mlflow.transformers.log_model` which serializes the model.

# COMMAND ----------

# DBTITLE 1,Install dependencies (no vLLM needed for PT)
# MAGIC %pip install transformers==4.57.6 mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0"

# COMMAND ----------

# DBTITLE 1,Install peft (no deps to avoid pulling transformers>=5)
# peft: --no-deps so it cannot drag in transformers>=5.
%pip install --no-deps peft==0.17.1
%restart_python

# COMMAND ----------

# DBTITLE 1,Check pinned versions
import peft
import transformers

print(f"transformers={transformers.__version__} peft={peft.__version__}")
assert transformers.__version__ == "4.57.6", (
    f"transformers is {transformers.__version__}; need 4.57.6 for tokenizer compat with NB01's adapter."
)

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("run_tag", "lora_r16_lr2e-4_ep3", "Run tag (best on validation)")
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-peft-llama31", "MLflow Experiment Path")
dbutils.widgets.text("merge_base_model", "unsloth/Meta-Llama-3.1-8B-Instruct", "Merge base (bf16; HF id or /Volumes path)")
dbutils.widgets.text("endpoint_name", "agency-llama-peft-pt", "Serving endpoint name")
# Blank = register a NEW version. Set to an existing version to redeploy without re-registering.
dbutils.widgets.text("model_version", "", "Existing UC model version (blank = register new)")

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
MERGE_BASE_MODEL = dbutils.widgets.get("merge_base_model").strip()

RUN_TAG = dbutils.widgets.get("run_tag").strip()
assert re.fullmatch(r"lora_r\d+_lr.+_ep\d+", RUN_TAG), (
    f"run_tag {RUN_TAG!r} does not look like notebook 01's RUN_TAG."
)

ADAPTER_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-adapter-{RUN_TAG}"
MERGED_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-merged-{RUN_TAG}"

workdir = tempfile.mkdtemp()
os.chdir(workdir)
LOCAL_MERGED = os.path.join(workdir, "merged")

# Separate UC model name to avoid conflicts with the vLLM-served version (notebook 03).
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.llama31_8b_agency_peft_pt"
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
MODEL_VERSION_OVERRIDE = dbutils.widgets.get("model_version").strip()

PROMPT_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_prompt.txt"
with open(PROMPT_FILE) as f:
    INSTRUCTION_PROMPT = f.read().strip()

print(f"run_tag={RUN_TAG}")
print(f"adapter:  {ADAPTER_DIR}")
print(f"merged -> {MERGED_DIR}")
print(f"UC model: {UC_MODEL_NAME}")
print(f"endpoint: {ENDPOINT_NAME} (Provisioned Throughput)")

# COMMAND ----------

# DBTITLE 1,Check the adapter and that the merge base matches the training base
import json

for _f in ("adapter_config.json", "adapter_model.safetensors"):
    assert os.path.isfile(os.path.join(ADAPTER_DIR, _f)), (
        f"Missing {_f} in {ADAPTER_DIR} — run notebook 01 first."
    )

with open(os.path.join(ADAPTER_DIR, "adapter_config.json")) as f:
    ADAPTER_BASE = json.load(f).get("base_model_name_or_path", "")


def model_family(name):
    """Normalize a model id for comparison (case-insensitive, strip org prefix)."""
    return re.sub(r"^.*/", "", name).lower()


assert model_family(ADAPTER_BASE) == model_family(MERGE_BASE_MODEL), (
    f"Adapter was trained on {ADAPTER_BASE!r} but merge base is {MERGE_BASE_MODEL!r}. "
    "These must be the same model family."
)
print(f"Adapter trained on: {ADAPTER_BASE}")
print(f"Merging into:       {MERGE_BASE_MODEL}")

# COMMAND ----------

# DBTITLE 1,Merge the adapter into the bf16 base (skipped if already merged from THIS adapter)
import gc
import hashlib
import shutil

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
    """True only for a COMPLETE merge of THIS adapter."""
    marker = os.path.join(merged_dir, ".merged_from_adapter")
    if not os.path.isfile(marker):
        return False
    with open(marker) as f:
        return f.read().strip() == adapter_fingerprint(adapter_dir)


if merged_is_current(MERGED_DIR, ADAPTER_DIR):
    print(f"Merged checkpoint at {MERGED_DIR} matches this adapter — staging to local disk.")
    shutil.copytree(MERGED_DIR, LOCAL_MERGED, dirs_exist_ok=True)
else:
    if os.path.isdir(MERGED_DIR):
        print(f"{MERGED_DIR} is stale or incomplete — will be replaced.")

    tokenizer = AutoTokenizer.from_pretrained(MERGE_BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MERGE_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    peft_model.eval()

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(LOCAL_MERGED, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(LOCAL_MERGED)

    # Persist to Volume for reuse by other notebooks.
    print(f"Copying merged checkpoint -> {MERGED_DIR} (~16 GB) ...")
    if os.path.isdir(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    shutil.copytree(LOCAL_MERGED, MERGED_DIR, dirs_exist_ok=True)
    # Marker LAST: its presence means the copy completed.
    with open(os.path.join(MERGED_DIR, ".merged_from_adapter"), "w") as f:
        f.write(adapter_fingerprint(ADAPTER_DIR))

    del merged, peft_model, base
    gc.collect()
    torch.cuda.empty_cache()

_files = os.listdir(LOCAL_MERGED)
assert "config.json" in _files and any(f.endswith(".safetensors") for f in _files), (
    f"Merged checkpoint in {LOCAL_MERGED} is incomplete: {_files}"
)
print(f"Merged checkpoint ready at {LOCAL_MERGED} ({len(_files)} files); persisted at {MERGED_DIR}")

# COMMAND ----------

# DBTITLE 1,Register model to Unity Catalog via mlflow.transformers (PT-compatible)
import mlflow
from mlflow import MlflowClient
from transformers import AutoModelForCausalLM, AutoTokenizer

if MODEL_VERSION_OVERRIDE:
    DEPLOY_VERSION = MODEL_VERSION_OVERRIDE
    print(f"Skipping registration; deploying existing {UC_MODEL_NAME} v{DEPLOY_VERSION}")
else:
    mlflow.set_experiment(EXPERIMENT_PATH)

    # Load merged model + tokenizer for the transformers flavor registration.
    # Unlike notebook 03's pyfunc ChatModel + vLLM entrypoint, PT needs the raw
    # transformers weights registered with task="llm/v1/chat".
    print("Loading merged model for MLflow registration (this may take a few minutes) ...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MERGED)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MERGED, torch_dtype=torch.bfloat16, device_map="auto"
    )

    with mlflow.start_run(run_name=f"register_pt_{RUN_TAG}"):
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name="model",
            task="llm/v1/chat",
            input_example={
                "messages": [{"role": "user", "content": "Extract fields from this document."}]
            },
        )

    # Register to UC (no env_pack needed — PT manages its own serving environment).
    model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME)
    DEPLOY_VERSION = str(model_version.version)
    MlflowClient().set_model_version_tag(UC_MODEL_NAME, DEPLOY_VERSION, "run_tag", RUN_TAG)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"✅ Registered {UC_MODEL_NAME} v{DEPLOY_VERSION} (run_tag={RUN_TAG})")

# COMMAND ----------

# DBTITLE 1,Wait for READY + verify Provisioned Throughput eligibility
import time

import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# --- Wait for model version to be READY ---
print(f"Checking status of {UC_MODEL_NAME} v{DEPLOY_VERSION} ...")
for i in range(180):  # up to 30 min
    mv = w.model_versions.get(full_name=UC_MODEL_NAME, version=int(DEPLOY_VERSION))
    status = mv.status.value
    if status == "READY":
        print(f"Model version {DEPLOY_VERSION} is READY.")
        break
    if status != "PENDING_REGISTRATION":
        raise RuntimeError(
            f"Model version {DEPLOY_VERSION} entered status {status}. "
            "Re-run the registration cell."
        )
    if i % 6 == 0:
        print(f"  Status: {status} — waiting ({i * 10}s elapsed) ...")
    time.sleep(10)
else:
    raise TimeoutError(f"Model version {DEPLOY_VERSION} not READY after 30 min.")

# --- Check Provisioned Throughput eligibility ---
# w.config.token is None on serverless GPU (managed-identity auth); use the notebook context token.
_api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
opt_resp = requests.get(
    f"{w.config.host}/api/2.0/serving-endpoints/get-model-optimization-info/"
    f"{UC_MODEL_NAME}/{DEPLOY_VERSION}",
    headers={"Authorization": f"Bearer {_api_token}"},
)
opt_info = opt_resp.json()
print(f"Optimization info: {opt_info}")

assert opt_info.get("optimizable"), (
    f"Model {UC_MODEL_NAME} v{DEPLOY_VERSION} is NOT eligible for Provisioned Throughput. "
    f"Full response: {opt_info}"
)
chunk_size = opt_info["throughput_chunk_size"]
print(
    f"✅ PT eligible — model_type={opt_info.get('model_type')}, "
    f"throughput_chunk_size={chunk_size} tokens/s"
)

# COMMAND ----------

# DBTITLE 1,Deploy the Provisioned Throughput endpoint
import datetime

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Start with 1 throughput chunk (the minimum). Increase max for higher capacity / autoscaling.
min_throughput = chunk_size
max_throughput = chunk_size

served = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=DEPLOY_VERSION,
    min_provisioned_throughput=min_throughput,
    max_provisioned_throughput=max_throughput,
)

existing = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME), None)
if existing is None:
    print(
        f"Creating PT endpoint '{ENDPOINT_NAME}' with v{DEPLOY_VERSION} "
        f"({min_throughput}–{max_throughput} tokens/s) ..."
    )
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served]),
        timeout=datetime.timedelta(minutes=40),
    )
else:
    print(f"Updating PT endpoint '{ENDPOINT_NAME}' to v{DEPLOY_VERSION} ...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served],
        timeout=datetime.timedelta(minutes=40),
    )
print(f"✅ Provisioned Throughput endpoint '{ENDPOINT_NAME}' is ready.")

# COMMAND ----------

# DBTITLE 1,Smoke test — query the PT endpoint
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

TEST_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_test_v3"
sample_ocr = spark.table(TEST_TABLE).select("raw_ocr_content").limit(1).collect()[0][0]

resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[
        ChatMessage(
            role=ChatMessageRole.USER,
            content=INSTRUCTION_PROMPT + "\n" + sample_ocr,
        )
    ],
    max_tokens=3500,
    temperature=0.0,
)
output = resp.choices[0].message.content
print(output[:1500])
assert output.strip().startswith("{"), (
    "PT endpoint output is not JSON — check the registration."
)
print("\n✅ Provisioned Throughput endpoint smoke test passed!")

# COMMAND ----------

# DBTITLE 1,Batch inference over the held-out test set with ai_query
# Variables not yet defined in 03b's config cell (carried over from NB03).
TABLE_SUFFIX = re.sub(r"[^0-9A-Za-z_]", "_", RUN_TAG)
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.agency_inference_output_llama_peft_pt_{TABLE_SUFFIX}"

TOP_8_FIELDS = [
    "PolicyNumber", "OwnerFile", "LoanFile",
    "OwnerPolicyNumber", "OwnerPolicyAmount", "OwnerPolicyDate",
    "LoanPolicyNumber", "LoanPolicyAmount", "LoanPolicyDate",
]

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

print(f"=== HELD-OUT TEST (PT) — {RUN_TAG} ({N_DOCS} docs) ===")
print(f"  all:  P {overall['precision']:.4f}  R {overall['recall']:.4f}  F1 {overall['f1']:.4f}")
print(f"  top8: P {top8['precision']:.4f}  R {top8['recall']:.4f}  F1 {top8['f1']:.4f}")

# COMMAND ----------

# DBTITLE 1,Log HELD-OUT TEST metrics to MLflow (stage=test)
mlflow.set_experiment(EXPERIMENT_PATH)
with mlflow.start_run(run_name=f"test_pt_{RUN_TAG}") as _test_run:
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
        "serving_mode": "provisioned_throughput",
        "matching_threshold": 0.6,
        "documents_scored": N_DOCS,
    })
    mlflow.set_tags({"approach": "held-out-test", "stage": "test", "serving_mode": "pt"})
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