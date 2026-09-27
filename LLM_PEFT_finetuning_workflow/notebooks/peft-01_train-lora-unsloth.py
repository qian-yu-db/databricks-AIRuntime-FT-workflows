# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v6"
# environment_version = "6"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # PEFT Fine-Tuning — Llama 3.1 8B Instruct + Unsloth (LoRA / QLoRA)
# MAGIC
# MAGIC Parameter-efficient fine-tuning of **Llama 3.1 8B Instruct** for title-insurance entity
# MAGIC extraction (OCR text → sparse JSON), on the **same train/val tables** as the FFT workflow
# MAGIC (`agency_ft_dataset_{train,val}_v3`, built by FFT notebook 00).
# MAGIC
# MAGIC - **Unsloth** LoRA with **response-only loss** (loss only on the assistant JSON turn)
# MAGIC - **`@distributed(gpus=1, gpu_type=...)`** from `serverless_gpu` launches training on one GPU
# MAGIC - **MLflow** tracks the run; only the **LoRA adapter** is saved (merge happens in notebook 02)
# MAGIC
# MAGIC | `train_mode` | Base | GPU | `max_seq_length` | When |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | `qlora_4bit` (default) | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | A10 | 4096 | cheap, fast iteration |
# MAGIC | `lora_bf16` | `unsloth/Meta-Llama-3.1-8B-Instruct` | H100 | 16384 | more precision, full-length docs |
# MAGIC
# MAGIC **Compute:** Serverless GPU with the **AI v6** environment. Do **not** install vLLM in this
# MAGIC notebook — Unsloth and vLLM pin conflicting torch/transformers (spec G1).

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet unsloth==2026.9.4 hf_transfer==0.1.9
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("train_mode", "qlora_4bit", ["qlora_4bit", "lora_bf16"], "Train mode")
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-peft-llama31", "MLflow Experiment Path")
# Mode-dependent settings — leave BLANK to inherit the train_mode default.
dbutils.widgets.text("base_model", "", "Base model (blank = mode default; HF id or /Volumes path)")
dbutils.widgets.text("gpu_type", "", "GPU type (blank = mode default)")
dbutils.widgets.text("max_seq_length", "", "Max sequence length (blank = mode default)")
dbutils.widgets.text("per_device_batch_size", "", "Per-device batch size (blank = mode default)")
dbutils.widgets.text("gradient_accumulation_steps", "", "Gradient accumulation (blank = mode default)")
# Shared LoRA / optimizer settings.
dbutils.widgets.text("lora_r", "16", "LoRA rank r")
dbutils.widgets.text("lora_alpha", "16", "LoRA alpha")
dbutils.widgets.text("learning_rate", "2e-4", "Learning rate")
dbutils.widgets.text("num_epochs", "3", "Number of epochs")

# COMMAND ----------

# DBTITLE 1,Configuration
import re

# train_mode sets the defaults; any non-blank widget above overrides them.
MODE_DEFAULTS = {
    "qlora_4bit": {
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "load_in_4bit": True,
        "gpu_type": "A10",
        "max_seq_length": "4096",
        "per_device_batch_size": "2",
        "gradient_accumulation_steps": "4",
    },
    "lora_bf16": {
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "load_in_4bit": False,
        "gpu_type": "H100",
        "max_seq_length": "16384",
        "per_device_batch_size": "1",
        "gradient_accumulation_steps": "8",
    },
}

TRAIN_MODE = dbutils.widgets.get("train_mode").strip()
assert TRAIN_MODE in MODE_DEFAULTS, f"train_mode must be one of {sorted(MODE_DEFAULTS)}, got {TRAIN_MODE!r}"


def mode_setting(name):
    """Widget value if set, else the train_mode default (blank widget = inherit)."""
    value = dbutils.widgets.get(name).strip()
    return value if value else MODE_DEFAULTS[TRAIN_MODE][name]


CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

BASE_MODEL = mode_setting("base_model")
LOAD_IN_4BIT = MODE_DEFAULTS[TRAIN_MODE]["load_in_4bit"]
GPU_TYPE = mode_setting("gpu_type")
MAX_SEQ_LENGTH = int(mode_setting("max_seq_length"))
PER_DEVICE_BATCH_SIZE = int(mode_setting("per_device_batch_size"))
GRADIENT_ACCUMULATION_STEPS = int(mode_setting("gradient_accumulation_steps"))

LORA_R = int(dbutils.widgets.get("lora_r"))
LORA_ALPHA = int(dbutils.widgets.get("lora_alpha"))
LORA_DROPOUT = 0.0  # 0 is Unsloth's optimized path
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_lr_str = dbutils.widgets.get("learning_rate").strip()
_ep_str = dbutils.widgets.get("num_epochs").strip()
LEARNING_RATE = float(_lr_str)
NUM_EPOCHS = int(_ep_str)

# Unique per config; notebooks 02/03 take this as their run_tag widget.
RUN_TAG = f"{TRAIN_MODE}_r{LORA_R}_lr{_lr_str}_ep{_ep_str}"  # e.g. qlora_4bit_r16_lr2e-4_ep3
assert re.fullmatch(r"[A-Za-z0-9_.\-]+", RUN_TAG), f"RUN_TAG has unsafe characters: {RUN_TAG!r}"

TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_train_v3"
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_val_v3"
# Shared storage: the remote GPU worker cannot read the driver's /tmp.
TRAIN_DATASET_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_peft_train_{RUN_TAG}"
EVAL_DATASET_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_peft_eval_{RUN_TAG}"
OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-output-{RUN_TAG}"
ADAPTER_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-peft-adapter-{RUN_TAG}"

# Llama 3.1 chat-template turn headers — the response-only loss boundary.
INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"

print(f"train_mode:     {TRAIN_MODE}")
print(f"base_model:     {BASE_MODEL}  (load_in_4bit={LOAD_IN_4BIT})")
print(f"gpu_type:       {GPU_TYPE}")
print(f"max_seq_length: {MAX_SEQ_LENGTH}")
print(f"batch x accum:  {PER_DEVICE_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS}")
print(f"LoRA:           r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
print(f"lr / epochs:    {LEARNING_RATE} / {NUM_EPOCHS}")
print(f"RUN_TAG:        {RUN_TAG}")
print(f"adapter ->      {ADAPTER_DIR}")

# COMMAND ----------

# DBTITLE 1,Set up MLflow experiment
import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# DBTITLE 1,Build chat-formatted datasets and drop over-length examples
# The instruction prompt is ALREADY in the `prompt` column (FFT notebook 00 baked
# agency_prompt.txt + OCR into it). Here we only wrap each row as a user/assistant turn and
# render it with the Llama 3.1 chat template (which also inserts its date system header).
import unsloth  # noqa: F401  — must be imported before transformers (spec G3)
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def render(prompt, response):
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": response.strip()},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_split(table):
    """Render every row, count tokens, and DROP rows longer than MAX_SEQ_LENGTH.

    Right-truncation would cut the JSON answer and teach malformed output, so over-length
    examples are skipped instead (the val/test METRIC evals still score every document).
    """
    pdf = spark.table(table).select("prompt", "response").toPandas()
    pdf["text"] = [render(p, r) for p, r in zip(pdf["prompt"], pdf["response"])]
    # The rendered text already starts with <|begin_of_text|> -> add_special_tokens=False.
    pdf["n_tokens"] = [len(ids) for ids in tokenizer(list(pdf["text"]), add_special_tokens=False)["input_ids"]]
    kept = pdf[pdf["n_tokens"] <= MAX_SEQ_LENGTH]
    return kept, len(pdf), len(pdf) - len(kept), pdf["n_tokens"]


train_kept, TRAIN_TOTAL, TRAIN_DROPPED, train_lengths = build_split(TRAIN_TABLE)
eval_kept, EVAL_TOTAL, EVAL_DROPPED, eval_lengths = build_split(EVAL_TABLE)
TRAIN_KEPT, EVAL_KEPT = len(train_kept), len(eval_kept)

print("Train token lengths:\n", train_lengths.describe(percentiles=[0.5, 0.9, 0.99]).round(0))
print(f"Train: kept {TRAIN_KEPT}/{TRAIN_TOTAL}, dropped {TRAIN_DROPPED} over {MAX_SEQ_LENGTH} tokens "
      f"({TRAIN_DROPPED / TRAIN_TOTAL:.1%})")
print(f"Eval:  kept {EVAL_KEPT}/{EVAL_TOTAL}, dropped {EVAL_DROPPED} over {MAX_SEQ_LENGTH} tokens")

assert TRAIN_KEPT + TRAIN_DROPPED == TRAIN_TOTAL and EVAL_KEPT + EVAL_DROPPED == EVAL_TOTAL
assert TRAIN_KEPT > 0, f"Every train example exceeds max_seq_length={MAX_SEQ_LENGTH}; raise it."
assert EVAL_KEPT > 0, f"Every eval example exceeds max_seq_length={MAX_SEQ_LENGTH}; raise it."

Dataset.from_pandas(train_kept[["text"]], preserve_index=False).save_to_disk(TRAIN_DATASET_PATH)
Dataset.from_pandas(eval_kept[["text"]], preserve_index=False).save_to_disk(EVAL_DATASET_PATH)
print(f"Saved datasets -> {TRAIN_DATASET_PATH} , {EVAL_DATASET_PATH}")

# COMMAND ----------

# DBTITLE 1,Preview response-only masking (driver-side check)
# Replicates the boundary train_on_responses_only will use, on one real example, so a
# template/masking problem is caught BEFORE paying for GPU time. Notebook cell for training
# re-checks this on the trainer's actual input_ids/labels.
_ids = tokenizer(train_kept["text"].iloc[0], add_special_tokens=False)["input_ids"]
_resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]

_boundary = None
for i in range(len(_ids) - len(_resp_ids) + 1):
    if _ids[i:i + len(_resp_ids)] == _resp_ids:
        _boundary = i + len(_resp_ids)  # keep searching: the LAST assistant header wins
assert _boundary is not None, f"Response header {RESPONSE_PART!r} not found — inspect the rendered text."

_masked_text = tokenizer.decode(_ids[:_boundary])
_supervised_text = tokenizer.decode(_ids[_boundary:])
print(f"Total tokens: {len(_ids)}  |  supervised: {len(_ids) - _boundary} "
      f"({(len(_ids) - _boundary) / len(_ids):.1%})")
print("\nMASKED (not in loss), head:\n ", _masked_text[:400].replace("\n", " "), "...")
print("\nSUPERVISED (loss), head/tail:\n ", _supervised_text[:300], "...", _supervised_text[-80:])

assert _ids[0] == tokenizer.bos_token_id and _ids[1] != tokenizer.bos_token_id, "Expected exactly one BOS."
assert _supervised_text.strip().startswith("{"), "Supervised span must start with the JSON answer."
assert _supervised_text.endswith("<|eot_id|>"), "Supervised span must end with <|eot_id|> so the model learns to stop."
print("\n✓ Masking preview OK: loss falls only on the assistant JSON turn, ending in <|eot_id|>.")
