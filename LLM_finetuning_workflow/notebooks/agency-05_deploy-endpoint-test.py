# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Deploy (vLLM), Batch Inference & Evaluation
# MAGIC
# MAGIC End-to-end pipeline for the fine-tuned **Qwen2.5** entity-extraction model, serving
# MAGIC it as a **custom LLM with vLLM** on Model Serving (Serverless GPU).
# MAGIC
# MAGIC Pattern follows the Databricks docs:
# MAGIC [Deploy custom LLMs](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
# MAGIC and the official *serve-custom-llms-starter* notebook. The model is logged as an
# MAGIC MLflow `ChatModel` whose `metadata.entrypoint` launches a vLLM OpenAI-compatible
# MAGIC server; Serving runs that entrypoint (not `predict`).
# MAGIC
# MAGIC 1. **Resolve latest model** — pick up the latest UC version registered by notebook 04 (sweep-launcher)
# MAGIC 2. **Local test** *(optional)* — launch vLLM in-notebook (Serverless GPU) for validation
# MAGIC 3. **Log + register** *(optional, skip if using notebook 04's registration)* — MLflow `ChatModel` + vLLM entrypoint
# MAGIC 4. **Deploy** — create/update the Model Serving endpoint with the latest version
# MAGIC 5. **Batch inference** — `ai_query()` over the test set
# MAGIC 6. **Evaluate** — field-level precision / recall / F1 vs ground truth
# MAGIC
# MAGIC > **Compute:** run this on a **Serverless GPU** notebook.
# MAGIC > model; for a real 7B (float16 ~14GB weights + KV cache) use 1xH100. See the
# MAGIC > `LOCAL_GPU_NOTE` in Configuration.

# COMMAND ----------

# DBTITLE 1,Install serving dependencies
# vLLM + MLflow + OpenAI client, per the custom-LLM-serving starter notebook.
# Can alternatively be pinned as a serverless environment:
#   https://docs.databricks.com/aws/en/compute/serverless/dependencies
%pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 opencv-python-headless==4.12.* mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk>=0.102.0
%restart_python

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

# --- Checkpoint tag (from notebook 04 sweep results) ---
# After running notebook 04's sweep, enter the best checkpoint_tag here
# (e.g., "lr2e-5_ep4"). Notebook 04 cell 9 prints the winner.
dbutils.widgets.text("checkpoint_tag", "lr2e-5_ep4", "Checkpoint Tag")
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume Model")


CHECKPOINT_TAG = dbutils.widgets.get("checkpoint_tag")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.sdk.service.serving import ServingModelWorkloadType


# --- Source of the fine-tuned weights (HF-format checkpoint in the UC Volume) ---
WEIGHTS_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-ft-final-{CHECKPOINT_TAG}/"

ARTIFACTS_PATH = "qwen3"        # local dir the weights are copied to
SERVED_MODEL_NAME = "qwen"        # name vLLM exposes the model under

# --- vLLM tuning --------------------------------------------------------------
DTYPE = "bfloat16"   # match training dtype; faster on H100 (native Tensor Core)
MAX_MODEL_LEN = 20480             # 20K: covers largest observed doc (~13k input + 3.5k output = ~17k tokens).
                                  # Reduced from 32K to free KV cache for more concurrent batching.
                                  # NOTE: local A10 (24 GB) is tight at 20K — reduce to 16384 for local vLLM tests.
GPU_MEMORY_UTILIZATION = 0.95     # Safe for dedicated serving GPUs (no competing processes)

# Allowlisted ports for Serverless GPU notebooks are 3000-3999. Serving requires 8080.
LOCAL_PORT = 3080
SERVING_PORT = 8080

# --- Unity Catalog destination for the registered (vLLM) model ----------------
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.qwen3_8b_agency_ft"

# --- Serving endpoint ---------------------------------------------------------
ENDPOINT_NAME = "agency-qwen-vllm"
# NOTE: GPU endpoints without scale-to-zero are deleted daily per workspace policy.
# Re-run cell 16 each morning while actively developing.
# GPU sizing (configurable). Qwen2.5-7B float16 ~14GB weights + KV cache:
#   GPU_MEDIUM = A10 24GB  — OK for <=4B; tight/OOM for 7B on long context
#   GPU_LARGE  = A100/L4   — recommended for 7B
#   GPU_XLARGE = H100 80GB — headroom for 7B + long context
# For the small 1.5B sweep model, GPU_MEDIUM (A10) is plenty.
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_LARGE
# WORKLOAD_SIZE removed — custom-entrypoint models reject autoscaling.
# workload_size controls replicas: Small=1, Medium=2, Large=4 (concurrency/4).
# Start with small, then size up with load test
WORKLOAD_SIZE = "Small"       
SCALE_TO_ZERO_ENABLED = False  # scale-to-zero requires min=0, but min<max is also rejected as autoscaling

LOCAL_GPU_NOTE = (
    "Run the local-test cells on a Serverless GPU notebook. A10 is fine for <=4B; "
    "use a larger GPU for 7B or reduce MAX_MODEL_LEN."
)

# --- Eval data / prompt -------------------------------------------------------
TEST_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_test_v3"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.agency_inference_output_qwen3_vllm"

# Extraction prompt (same one used during fine-tuning; clean, no [INST] tags). Read from
# the Volume — the SAME file agency-00 baked into the train/val prompt column — so the
# eval prompt can never drift from the training prompt, and the path is stable regardless
# of where this notebook lives in the workspace.
PROMPT_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_prompt.txt"
with open(PROMPT_FILE, "r") as f:
    INSTRUCTION_PROMPT = f.read().strip()

print(LOCAL_GPU_NOTE)
print(f"Weights source: {WEIGHTS_VOLUME_PATH}")
print(f"Prompt loaded from: {PROMPT_FILE} ({len(INSTRUCTION_PROMPT)} chars)")

# COMMAND ----------

# DBTITLE 1,Resolve latest UC model version (registered by notebook 04)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Resolve the latest version of the registered model from UC.
# Notebook 04 (sweep-launcher) registers the best sweep checkpoint here after
# evaluating all hyperparameter combinations. This cell picks up wherever 04 left off.
_model_versions = list(w.model_versions.list(full_name=UC_MODEL_NAME))
assert _model_versions, f"No versions found for {UC_MODEL_NAME} — run notebook 04 first."

_latest = max(_model_versions, key=lambda v: int(v.version))
LATEST_MODEL_VERSION = str(_latest.version)

print(f"✅ Latest UC model version: {UC_MODEL_NAME} v{LATEST_MODEL_VERSION}")
print(f"   Source: {_latest.source}")
print(f"   Status: {_latest.status}")

# COMMAND ----------

# DBTITLE 1,Stage weights from UC Volume to local disk
import shutil

# vLLM loads --model from a local directory. Copy the Axolotl checkpoint off the
# Volume onto local disk. (dirs_exist_ok lets you re-run the cell.)
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
import re

# HOW THIS WORKS
# The Axolotl fine-tuned template already has a conditional block that injects
# '<think>\n\n</think>\n\n' as a PROMPT PREFIX (not model output) when
# enable_thinking=False is passed by the caller. vLLM 0.11.2 never passes that
# kwarg, so the condition is never triggered and the model generates the think
# tags itself (learned behaviour from training).
#
# FIX: make the injection unconditional by removing the if-guard. The think
# block then becomes part of the prompt, not the generated content, so API
# responses contain clean JSON with no <think> tags at all.
# The regexp_replace in cell 21 becomes a true no-op after this patch + redeploy.

template_path = os.path.join(ARTIFACTS_PATH, "chat_template.jinja")
with open(template_path, "r") as f:
    original = f.read()

# Target: the conditional block on lines 85-87 of chat_template.jinja
old_block = (
    "{%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}"
)
new_block = "    {{- '<think>\\n\\n</think>\\n\\n' }}"

patched = original.replace(old_block, new_block)

if original == patched:
    print("WARNING: conditional block not found — inspect chat_template.jinja manually.")
    print("Lines containing 'think':")
    for i, l in enumerate(original.splitlines()):
        if "think" in l.lower():
            print(f"  {i:3d}: {repr(l)}")
else:
    with open(template_path, "w") as f:
        f.write(patched)
    print("Patched chat_template.jinja: <think>\\n\\n</think> is now an unconditional prompt prefix.")
    print("Restart vLLM (cell 11 -> 7 -> 8) for the change to take effect locally.")
    print("Re-run cells 13 -> 14 -> 16 to bake the patch into the deployed endpoint.")

# COMMAND ----------

# DBTITLE 1,Define the vLLM entrypoint
# Single source of truth for the vLLM launch command, parameterized by port so the
# SAME command is used for the local test (LOCAL_PORT) and for Serving (SERVING_PORT).
import shlex

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
        "--enable-prefix-caching",          # shared instruction prompt (~500 tokens) cached across requests
        "--max-num-seqs", "14",             # cap concurrent decode seqs to what fits in KV at 20K context
        "--disable-log-requests",           # reduce I/O overhead in production
        # "--enforce-eager",  # faster startup, slower inference
    ]
    return " ".join(args)


print(entrypoint(LOCAL_PORT))

# COMMAND ----------

# DBTITLE 1,Register model to Unity Catalog
import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

# --- Log as MLflow ChatModel with vLLM entrypoint ---
class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

model_info = mlflow.pyfunc.log_model(
    name=SERVED_MODEL_NAME,
    python_model=LLMModel(),
    artifacts={"model_dir": ARTIFACTS_PATH},
    metadata={
        "task": "llm/v1/chat",
        "entrypoint": entrypoint(SERVING_PORT),
    },
)
print(f"Logged model URI: {model_info.model_uri}")

# --- Register to Unity Catalog (requires databricks-sdk >= 0.102.0) ---
model_version = mlflow.register_model(
    model_info.model_uri,
    UC_MODEL_NAME,
    env_pack="databricks_model_serving",
)
print(f"\n\u2705 Registered {UC_MODEL_NAME} version {model_version.version}")
print(f"   Checkpoint: {CHECKPOINT_TAG}")

# COMMAND ----------

# DBTITLE 1,Local test — start vLLM server in background
import subprocess

log = open("process.log", "w")
subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT)],
    stdout=log,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
)
print(f"vLLM starting on port {LOCAL_PORT} (logs -> {workdir}/process.log)")

# COMMAND ----------

# DBTITLE 1,Wait for the server to be ready
# MAGIC %sh
# MAGIC # Tail logs until vLLM is ready. If this hangs, startup probably hit an error —
# MAGIC # inspect process.log for the traceback.
# MAGIC tail -f process.log | sed -u '/Application startup complete/q'

# COMMAND ----------

# DBTITLE 1,Local test — single extraction request
import requests

# Real smoke test: run one actual extraction through the local server, exactly as
# it will be called in production (instruction prompt + OCR text).
sample_ocr = spark.table(TEST_TABLE).select("Raw_OCR_Content").limit(1).collect()[0][0]

resp = requests.post(
    f"http://localhost:{LOCAL_PORT}/invocations",
    json={
        "messages": [
            {"role": "user", "content": INSTRUCTION_PROMPT + "\n" + sample_ocr}
        ],
        "max_tokens": 3500,
        "temperature": 0.0,
    },
)
extraction = resp.json()["choices"][0]["message"]["content"]
print(extraction)

# COMMAND ----------

# DBTITLE 1,Local test — streaming request (optional)
import json

resp = requests.post(
    f"http://localhost:{LOCAL_PORT}/invocations",
    json={"messages": [{"role": "user", "content": "Reply with the single word: ready"}], "stream": True},
    stream=True,
)
for line in resp.iter_lines():
    if not line or line == b"data: [DONE]":
        continue
    if line.startswith(b"data: "):
        delta = json.loads(line[6:])["choices"][0].get("delta", {})
        if "content" in delta:
            print(delta["content"], end="", flush=True)

# COMMAND ----------

# DBTITLE 1,Stop the local vLLM server
# MAGIC %sh
# MAGIC pkill -f vllm.entrypoints.openai.api_server || echo "no vLLM server running"

# COMMAND ----------

# DBTITLE 1,Deploy the serving endpoint
# MAGIC %md
# MAGIC ## 4. Create / update the serving endpoint

# COMMAND ----------

# DBTITLE 1,Create or update endpoint
import datetime
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

# Use the version just registered in cell 10. Falls back to LATEST_MODEL_VERSION
# (resolved in cell 6) if registration cell was skipped.
_deploy_version = str(model_version.version) if 'model_version' in dir() else LATEST_MODEL_VERSION

# Wait for model version to be READY (env_pack for ~16 GB models takes 20-30 min)
print(f"Checking status of {UC_MODEL_NAME} v{_deploy_version} ...")
for i in range(180):  # up to 30 minutes
    mv = w.model_versions.get(full_name=UC_MODEL_NAME, version=int(_deploy_version))
    status = mv.status.value
    if status == "READY":
        print(f"Model version {_deploy_version} is READY.")
        break
    if status not in ("PENDING_REGISTRATION", "READY"):
        raise RuntimeError(
            f"Model version {_deploy_version} entered unexpected status: {status}. "
            f"Re-run the registration cell (cell 10) to create a new version."
        )
    if i % 6 == 0:  # print every 60s
        print(f"  Status: {status} — waiting ({i*10}s elapsed) ...")
    time.sleep(10)
else:
    raise TimeoutError(
        f"Model version {_deploy_version} did not reach READY within 30 minutes. "
        f"The env_pack registration may have failed silently. "
        f"Re-run cell 10 to register a new version, then re-run this cell."
    )

served = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=_deploy_version,
    workload_type=WORKLOAD_TYPE,
    # workload_size sets the number of fixed replicas (no autoscaling between replicas in beta).
    # Small=1, Medium=2, Large=4 replicas (provisioned_concurrency / 4).
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
)
config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served])

existing = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME), None)
if existing is None:
    print(f"Creating endpoint '{ENDPOINT_NAME}' with version {_deploy_version} ...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME, config=config, timeout=datetime.timedelta(minutes=40)
    )
    print("✅ Endpoint created.")
else:
    print(f"Updating endpoint '{ENDPOINT_NAME}' to version {_deploy_version} ...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME, served_entities=[served], timeout=datetime.timedelta(minutes=40)
    )
    print("✅ Endpoint updated.")

# COMMAND ----------

# DBTITLE 1,Query the ready endpoint (smoke test)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Fetch a sample document (also needed by cell 19); defined here in case cell 9 was skipped.
sample_ocr = spark.table(TEST_TABLE).select("Raw_OCR_Content").limit(1).collect()[0][0]

w = WorkspaceClient()
resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[ChatMessage(role=ChatMessageRole.USER, content=INSTRUCTION_PROMPT + "\n" + sample_ocr)],
    max_tokens=3500,
    temperature=0.0,
)
print(resp.choices[0].message.content[:1500])

# COMMAND ----------

# DBTITLE 1,Batch Inference with ai_query
# MAGIC %md
# MAGIC ## 5. Batch inference with `ai_query()`
# MAGIC
# MAGIC `ai_query()` handles concurrency, retries, and rate limiting. Against a
# MAGIC `llm/v1/chat` endpoint, pass the request text and read back the response string.

# COMMAND ----------

# DBTITLE 1,Run ai_query batch inference (Python)
# Escape single quotes for safe SQL embedding.
escaped_prompt = INSTRUCTION_PROMPT.replace("'", "\\'")

inference_sql = f"""
CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
SELECT
  File_Name,
  Raw_OCR_Content,
  ai_query(
    '{ENDPOINT_NAME}',
    -- No truncation for normal documents. LEFT(100000) is a far-out failsafe only
    -- (~25 000 tokens of OCR; 25 000 + 1 450 prompt + 3 500 out = 29 950 < MAX_MODEL_LEN=32768).
    -- Largest title insurance document in test set was ~13 048 input tokens total.
    CONCAT('{escaped_prompt}', '\n', LEFT(Raw_OCR_Content, 100000)),
    modelParameters => named_struct('max_tokens', 3500, 'temperature', 0.0)
  ) AS model_output
FROM {TEST_TABLE}
"""

spark.sql(inference_sql)
print(f"Batch inference complete -> {OUTPUT_TABLE}")
display(spark.table(OUTPUT_TABLE).limit(5))

# COMMAND ----------

# DBTITLE 1,Evaluation Section
# MAGIC %md
# MAGIC ## 6. Evaluation — field-level accuracy
# MAGIC
# MAGIC Compare model outputs against ground truth at the field level. Precision, recall,
# MAGIC and F1 via fuzzy string matching (SequenceMatcher ratio > 0.6).

# COMMAND ----------

# DBTITLE 1,Parse and flatten outputs
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.types import StructType, StructField, StringType
import pyspark.sql.functions as F
import pandas as pd
import difflib

# Schema for the JSON extraction output — new CamelCase field set from agency_prompt.txt
# (sparse targets; from_json parses absent keys as null). Keep in sync with the prompt
# and the eval StructTypes in agency-02_local-vllm-eval.py / DAB sweep_eval.py.
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

# Parse model outputs — strip Qwen3 <think>...</think> prefix before parsing JSON.
# Qwen3 inference-time chat template enables thinking by default; fine-tuning with
# enable_thinking=False suppresses content inside the block but the empty tags remain.
# To remove them server-side, pass --chat-template-kwargs '{"enable_thinking": false}'
# in the vLLM entrypoint (cell 6) and redeploy. For now, strip client-side.
outputs_pdf = (
    spark.table(OUTPUT_TABLE)
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
# Merge predictions with ground truth
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

# Compute overall metrics
tp = (merged['result'] == 'TP').sum()
fp = (merged['result'] == 'FP').sum()
fn = (merged['result'] == 'FN').sum()
tn = (merged['result'] == 'TN').sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"=== Overall Metrics ===")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"  Total fields evaluated: {len(merged)}")

# COMMAND ----------

# DBTITLE 1,Per-field accuracy breakdown
# Per-field accuracy breakdown
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

