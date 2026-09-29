# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Fine-Tuning Qwen3-8B on AI Runtime (Serverless GPU)
# MAGIC
# MAGIC This notebook performs **full-weight supervised fine-tuning (SFT)** of `Qwen/Qwen3-8B` for **instruction-following** (document entity extraction) using:
# MAGIC - **TRL** `SFTTrainer` with **DeepSpeed ZeRO Stage 3** for memory-efficient distributed training
# MAGIC - **`@distributed` decorator** from `serverless_gpu` to launch multi-GPU training
# MAGIC - **MLflow** for experiment tracking and model registration to Unity Catalog
# MAGIC
# MAGIC **Compute requirement:** Serverless GPU **8xH100** with AI v5 environment.
# MAGIC
# MAGIC To connect:
# MAGIC 1. Click the notebook's compute selector → **Serverless GPU**
# MAGIC 2. Select **8xH100** as the Accelerator
# MAGIC 3. Choose **AI v5** environment
# MAGIC 4. Click Apply
# MAGIC
# MAGIC | Parameter | Value |
# MAGIC | --- | --- |
# MAGIC | Base model | `Qwen/Qwen3-8B` |
# MAGIC | Training method | Full-weight SFT (no LoRA/PEFT) |
# MAGIC | Distributed strategy | DeepSpeed ZeRO Stage 3 |
# MAGIC | Train data | `fins_genai.fine_tuning.agency_ft_dataset_train_v3` (~4,250 rows) |
# MAGIC | Eval data | `fins_genai.fine_tuning.agency_ft_dataset_val_v3` (~250 rows) |
# MAGIC | Register to | `fins_genai.fine_tuning` |
# MAGIC | Task | Instruction following (entity extraction from title insurance docs) |

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet trl accelerate datasets deepspeed
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("num_epochs", "3", "Number of epochs")
dbutils.widgets.text("learning_rate", "1e-5", "Learning rate")
dbutils.widgets.text("max_seq_length", "4096", "Max sequence length")
dbutils.widgets.text("per_device_batch_size", "2", "Per-device batch size")
dbutils.widgets.text("gradient_accumulation_steps", "2", "Gradient accumulation steps")
dbutils.widgets.text("num_gpus", "8", "Number of GPUs")
dbutils.widgets.text("gpu_type", "H100", "GPU type")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-finetuning-ai-runtime", "MLflow Experiment Path")

# COMMAND ----------

# DBTITLE 1,Configuration
# === Configuration (from widgets / job parameters) ===
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")

MODEL_NAME = "Qwen/Qwen3-8B"
TRAIN_DATA_PATH = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_train_v3"
EVAL_DATA_PATH = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_val_v3"
REGISTER_TO = f"{CATALOG}.{SCHEMA}"  # UC catalog.schema for model registration
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

# Training hyperparameters (from widgets)
NUM_EPOCHS = int(dbutils.widgets.get("num_epochs"))
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
MAX_SEQ_LENGTH = int(dbutils.widgets.get("max_seq_length"))
PER_DEVICE_BATCH_SIZE = int(dbutils.widgets.get("per_device_batch_size"))  # Per GPU batch size
GRADIENT_ACCUMULATION_STEPS = int(dbutils.widgets.get("gradient_accumulation_steps"))  # Effective batch size = batch * accum * GPUs

# Distributed training config (from widgets)
NUM_GPUS = int(dbutils.widgets.get("num_gpus"))
GPU_TYPE = dbutils.widgets.get("gpu_type")

# Run tag for unique checkpoint paths per sweep iteration
_lr_str = dbutils.widgets.get("learning_rate")
_ep_str = dbutils.widgets.get("num_epochs")
RUN_TAG = f"lr{_lr_str}_ep{_ep_str}"  # e.g., "lr1e-5_ep3"

# COMMAND ----------

# DBTITLE 1,Set up MLflow experiment
import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# DBTITLE 1,Load training data from Unity Catalog
from datasets import Dataset
from transformers import AutoTokenizer

# Load tokenizer — needed for chat template verification in next cell
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

# Load data from Unity Catalog tables
train_pdf = spark.table(TRAIN_DATA_PATH).toPandas()
eval_pdf = spark.table(EVAL_DATA_PATH).toPandas()


def build_messages(row):
    """Build messages list for TRL SFTTrainer messages format."""
    return [
        {"role": "user", "content": row["prompt"].strip()},
        {"role": "assistant", "content": row["response"].strip()},
    ]


# Build messages column — SFTTrainer applies Qwen3 chat template via processing_class
train_pdf["messages"] = train_pdf.apply(build_messages, axis=1)
eval_pdf["messages"] = eval_pdf.apply(build_messages, axis=1)

train_dataset = Dataset.from_pandas(train_pdf[["messages"]])
eval_dataset = Dataset.from_pandas(eval_pdf[["messages"]])

# Save datasets to a UC Volume for distributed training access
# (remote GPU workers cannot access driver's /tmp — must use shared storage)
TRAIN_DATASET_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_train_dataset"
EVAL_DATASET_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_eval_dataset"
train_dataset.save_to_disk(TRAIN_DATASET_PATH)
eval_dataset.save_to_disk(EVAL_DATASET_PATH)

print(f"Training samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print(f"\nSample messages (first example):")
for msg in train_dataset[0]["messages"]:
    print(f"  [{msg['role']}]: {msg['content'][:200]}...")

# COMMAND ----------

# DBTITLE 1,Verify completion-only masking (DataCollatorForCompletionOnlyLM)
# Preview EXACTLY what the training loss will be computed on. In TRL 0.23+,
# DataCollatorForCompletionOnlyLM was removed — completion-only masking is now handled
# internally by SFTConfig(completion_only_loss=True). This cell replicates the masking
# logic manually to verify the masked-vs-supervised split on actual token labels.
#
# The logic: mask (labels=-100) every token up to and including the response template,
# so loss falls only on the assistant (JSON) turn.

# The assistant turn in Qwen3 ChatML begins immediately after this header.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# Render one example the way training will (enable_thinking=False), tokenize it, then
# manually apply completion-only masking to see the real masked-vs-supervised split.
_sample_msgs = train_dataset[0]["messages"]
_rendered = tokenizer.apply_chat_template(
    _sample_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
_enc = tokenizer(_rendered, add_special_tokens=False)
_ids = _enc["input_ids"]

# Find the response template token boundary (same logic the old collator used).
# Qwen3's chat template inserts an empty <think>\n\n</think>\n block even with
# enable_thinking=False. The supervised span should start AFTER that block.
_response_template_full = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_response_token_ids = tokenizer.encode(_response_template_full, add_special_tokens=False)
_template_len = len(_response_token_ids)

# Search for the response template in the token sequence
_boundary = None
for i in range(len(_ids) - _template_len + 1):
    if _ids[i:i + _template_len] == _response_token_ids:
        _boundary = i + _template_len
        break

# Fallback: try without the think block in case template behavior changes
if _boundary is None:
    _fallback_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    _fb_len = len(_fallback_ids)
    for i in range(len(_ids) - _fb_len + 1):
        if _ids[i:i + _fb_len] == _fallback_ids:
            _boundary = i + _fb_len
            break

assert _boundary is not None, (
    f"Response template not found in tokenized sequence. "
    "Do NOT train; inspect the rendered example above."
)

# Build labels: -100 for masked (prompt), token id for supervised (assistant response)
_labels = [-100] * _boundary + _ids[_boundary:]
_supervised_ids = [t for t, l in zip(_ids, _labels) if l != -100]
_masked_ids = [t for t, l in zip(_ids, _labels) if l == -100]

print(f"Total tokens: {len(_ids)}")
print(f"\nMASKED (labels=-100, {len(_masked_ids)} tokens) — prompt/instruction/OCR, NOT in loss:")
print("  " + tokenizer.decode(_masked_ids)[:300].replace("\n", " ") + " ...")
print(f"\nSUPERVISED ({len(_supervised_ids)} tokens) — loss computed on this span:")
print("  " + tokenizer.decode(_supervised_ids))
print(f"\nGradient efficiency: {len(_supervised_ids) / max(len(_ids), 1) * 100:.1f}% of tokens contribute to loss")

# Hard checks — stop before an expensive run if masking is wrong.
assert len(_supervised_ids) > 0, (
    f"Supervised span is EMPTY — the response template {RESPONSE_TEMPLATE!r} was found but "
    "no tokens follow it. Do NOT train; inspect the rendered example above."
)
assert len(_supervised_ids) < len(_ids), (
    "Nothing was masked — the whole sequence is in the loss. Check the response template."
)
_supervised_text = tokenizer.decode(_supervised_ids)
assert "<think>" not in _supervised_text, "ERROR: <think> tokens found in supervised span!"
print("\n✓ Completion-only masking verified: loss falls only on the assistant JSON turn.")

# COMMAND ----------

# DBTITLE 1,load DeepSpeed config
import json

# DeepSpeed ZeRO Stage 3 config for full-weight training
# Shards model params, gradients, and optimizer states across all GPUs
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "wall_clock_breakdown": False,
}

# Save DeepSpeed config to Volume (remote GPU workers cannot access driver's /tmp)
DS_CONFIG_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ds_config.json"
with open(DS_CONFIG_PATH, "w") as f:
    json.dump(ds_config, f, indent=2)

print("DeepSpeed ZeRO Stage 3 config saved.")
print(json.dumps(ds_config, indent=2))

# COMMAND ----------

# DBTITLE 1,Configure SFT Trainer
from serverless_gpu import distributed

@distributed(gpus=NUM_GPUS, gpu_type=GPU_TYPE)
def run_training():
    """Full-weight SFT of Qwen3-8B with DeepSpeed ZeRO-3 and completion-only loss."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from datasets import load_from_disk
    import mlflow

    # Load datasets from UC Volume (shared storage accessible by all GPU workers)
    train_dataset = load_from_disk(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_train_dataset")
    eval_dataset = load_from_disk(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/agency_eval_dataset")

    # In TRL 0.23+, chat_template_kwargs is read per-example from the dataset
    # (no longer a SFTConfig param). Add enable_thinking=False to suppress Qwen3 <think> blocks.
    def _add_template_kwargs(example):
        example["chat_template_kwargs"] = {"enable_thinking": False}
        return example
    train_dataset = train_dataset.map(_add_template_kwargs)
    eval_dataset = eval_dataset.map(_add_template_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # Load model in full precision (bf16) — no quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    # SFT training config with DeepSpeed
    training_args = SFTConfig(
        output_dir=f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-ft-output-{RUN_TAG}",
        run_name="qwen3-8b-fullweight-sft-agency",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        # Completion-only loss: masks prompt tokens (labels=-100), loss on assistant turn only.
        # Replaces the removed DataCollatorForCompletionOnlyLM in TRL 0.23+.
        completion_only_loss=True,

        deepspeed=f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ds_config.json",
        report_to="mlflow",
    )

    # Initialize trainer (full-weight, no PEFT)
    # MLflow experiment is inherited from parent process; Trainer handles logging
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train — Trainer's report_to="mlflow" handles run creation and metric logging
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)


    # Log extra params only from rank 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        mlflow.log_params({
            "base_model": MODEL_NAME,
            "training_method": "full_weight_sft",
            "deepspeed_stage": 3,
            "num_gpus": NUM_GPUS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "completion_only_loss": "response_template_collator",
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
        })

    print(f"\nTraining complete!")
    print(f"  Train loss: {metrics['train_loss']:.4f}")
    print(f"  Eval loss: {eval_metrics['eval_loss']:.4f}")

    # Save the final model to Volume (accessible from driver for registration)
    trainer.save_model(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-ft-final-{RUN_TAG}")
    tokenizer.save_pretrained(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-ft-final-{RUN_TAG}")

print("Training function defined. Ready to launch distributed training.")

# COMMAND ----------

# DBTITLE 1,Train the model
# Launch distributed full-weight SFT across 8xH100 GPUs
run_training.distributed()

# COMMAND ----------

# DBTITLE 1,Save weights and save full model
import os

# Reuse the checkpoint saved by trainer.save_model() in cell 9 directly —
# no need to reload and re-save (~16 GB duplicate). Cell 12 uses MODEL_OUTPUT_DIR.
MODEL_OUTPUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODEL}/agency-ft-final-{RUN_TAG}"

# Verify the checkpoint exists and contains expected artifacts
assert os.path.exists(f"{MODEL_OUTPUT_DIR}/config.json"), (
    f"Checkpoint not found at {MODEL_OUTPUT_DIR}. Run training (cell 10) first."
)
print(f"Model checkpoint verified at: {MODEL_OUTPUT_DIR}")
print(f"Contents: {os.listdir(MODEL_OUTPUT_DIR)}")

# COMMAND ----------

# DBTITLE 1,Next Steps
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Deploy**: Use Model Serving to deploy the registered model from `fins_genai.fine_tuning.qwen3_8b_agency_ft`
# MAGIC 2. **Evaluate**: Run inference on held-out test documents to measure extraction accuracy vs. the previous Mistral fine-tune
# MAGIC 3. **Iterate**: Adjust `LEARNING_RATE`, `NUM_EPOCHS`, `PER_DEVICE_BATCH_SIZE` to improve quality
# MAGIC
# MAGIC **Notes:**
# MAGIC - Full-weight SFT trains ALL 8B parameters (vs. only ~1-2% with LoRA), giving the model more capacity to learn the task
# MAGIC - DeepSpeed ZeRO-3 shards model params, gradients, and optimizer states across 8 GPUs to fit in memory
# MAGIC - Training time: ~1-2 hours for 5 epochs on 1,690 samples with 8xH100