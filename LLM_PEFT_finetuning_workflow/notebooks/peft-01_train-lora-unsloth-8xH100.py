# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v6"
# environment_version = "6"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # PEFT Fine-Tuning — Llama 3.1 8B Instruct + Unsloth (bf16 LoRA on 8×H100)
# MAGIC
# MAGIC Distributed parameter-efficient fine-tuning of **Llama 3.1 8B Instruct** for title-insurance entity
# MAGIC extraction (OCR text → sparse JSON), on the **same train/val tables** as the FFT workflow
# MAGIC (`agency_ft_dataset_{train,val}_v3`, built by FFT notebook 00).
# MAGIC
# MAGIC - **Unsloth** LoRA with **response-only loss** (loss only on the assistant JSON turn)
# MAGIC - **`@distributed(gpus=8, gpu_type=...)`** from `serverless_gpu` launches DDP training across 8 GPUs
# MAGIC - Each rank loads the model onto its own device via `device_map={'': local_rank}`
# MAGIC - **MLflow** tracks the run; only the **LoRA adapter** is saved (merge happens in notebook 02)
# MAGIC
# MAGIC | Base | Precision | GPU | `max_seq_length` | batch × accum × GPUs |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | `unsloth/Meta-Llama-3.1-8B-Instruct` | bf16 LoRA | 8×H100 | 16384 (covers \~all documents) | 1 × 1 × 8 |
# MAGIC
# MAGIC **Compute:** Serverless GPU with the **AI v6** environment (8×H100 accelerator). Do **not** install vLLM in this
# MAGIC notebook — Unsloth and vLLM pin conflicting torch/transformers (spec G1).
# MAGIC
# MAGIC Adapted from the single-GPU notebook `peft-01_train-lora-unsloth` and the
# MAGIC [Databricks distributed Unsloth tutorial](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth-distributed).

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet unsloth==2026.9.4 hf_transfer==0.1.9
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "fins_genai", "Catalog")
dbutils.widgets.text("schema", "fine_tuning", "Schema")
dbutils.widgets.text("volume", "training_data", "Volume")
dbutils.widgets.text("volume_model", "checkpoints", "Volume for Model")
dbutils.widgets.text("experiment_path", "/Users/q.yu@databricks.com/mlflow_experiments/agency-peft-llama31", "MLflow Experiment Path")
dbutils.widgets.text("base_model", "unsloth/Meta-Llama-3.1-8B-Instruct", "Base model (bf16; HF id or /Volumes path)")
dbutils.widgets.text("gpu_type", "H100", "GPU type")
dbutils.widgets.text("max_seq_length", "16384", "Max sequence length")
dbutils.widgets.text("per_device_batch_size", "1", "Per-device batch size")
# With 8 GPUs via DDP the effective batch = per_device × num_gpus × accum.
# Default 1 keeps effective batch = 1 × 8 × 1 = 8, same as the single-GPU notebook (1 × 1 × 8).
dbutils.widgets.text("gradient_accumulation_steps", "1", "Gradient accumulation steps")
# LoRA / optimizer settings.
dbutils.widgets.text("lora_r", "16", "LoRA rank r")
dbutils.widgets.text("lora_alpha", "16", "LoRA alpha")
dbutils.widgets.text("learning_rate", "2e-4", "Learning rate")
dbutils.widgets.text("num_epochs", "3", "Number of epochs")

# COMMAND ----------

# DBTITLE 1,Configuration
import re

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_MODEL = dbutils.widgets.get("volume_model")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

BASE_MODEL = dbutils.widgets.get("base_model").strip()
GPU_TYPE = dbutils.widgets.get("gpu_type").strip()
MAX_SEQ_LENGTH = int(dbutils.widgets.get("max_seq_length"))
PER_DEVICE_BATCH_SIZE = int(dbutils.widgets.get("per_device_batch_size"))
GRADIENT_ACCUMULATION_STEPS = int(dbutils.widgets.get("gradient_accumulation_steps"))

LORA_R = int(dbutils.widgets.get("lora_r"))
LORA_ALPHA = int(dbutils.widgets.get("lora_alpha"))
LORA_DROPOUT = 0.0  # 0 is Unsloth's optimized path
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_lr_str = dbutils.widgets.get("learning_rate").strip()
_ep_str = dbutils.widgets.get("num_epochs").strip()
LEARNING_RATE = float(_lr_str)
NUM_EPOCHS = int(_ep_str)

# Unique per config; notebooks 02/03 take this as their run_tag widget.
RUN_TAG = f"lora_r{LORA_R}_lr{_lr_str}_ep{_ep_str}"  # e.g. lora_r16_lr2e-4_ep3
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

NUM_GPUS = 8  # 8×H100 DDP

print(f"base_model:     {BASE_MODEL}  (bf16)")
print(f"gpu_type:       {NUM_GPUS}x{GPU_TYPE}")
print(f"max_seq_length: {MAX_SEQ_LENGTH}")
print(f"batch x accum x gpus: {PER_DEVICE_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} x {NUM_GPUS}  "
      f"(effective batch = {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS})")
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

# COMMAND ----------

# DBTITLE 1,Define the 8×H100 distributed training function
from serverless_gpu import distributed


@distributed(gpus=NUM_GPUS, gpu_type=GPU_TYPE)
def run_training():
    """Unsloth bf16 LoRA SFT with response-only loss on 8×H100 DDP; saves only the adapter.

    Key difference from the single-GPU notebook:
    - Each rank sets torch.cuda.set_device(local_rank) before loading.
    - FastLanguageModel.from_pretrained gets device_map={'': local_rank} so each
      rank loads the model onto its own GPU (without this, all 8 load onto GPU 0 → OOM).
    - MLflow params/tags and assertions run only on rank 0.
    Ref: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth-distributed
    """
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

    import unsloth  # must precede transformers/trl/peft (spec G3)
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    import mlflow
    import peft
    import torch
    import transformers
    import trl
    from datasets import load_from_disk
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    # ---- DDP device placement: each rank → its own GPU --------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    is_main = int(os.environ.get("RANK", 0)) == 0

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto: bf16 on H100
        load_in_4bit=False,  # bf16 weights (not quantized)
        device_map={'': local_rank},  # ← critical: place model on this rank's GPU
    )
    # pad == eos would mask <|eot_id|> out of the labels -> the model never learns to stop.
    assert tokenizer.pad_token is not None and tokenizer.pad_token_id != tokenizer.eos_token_id, (
        f"pad_token ({tokenizer.pad_token!r}) must differ from eos_token ({tokenizer.eos_token!r})."
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # "unsloth" gradient checkpointing conflicts with DDP hooks; use standard PyTorch.
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    train_dataset = load_from_disk(TRAIN_DATASET_PATH)
    eval_dataset = load_from_disk(EVAL_DATASET_PATH)

    # The driver-side text (cell 6) starts with <|begin_of_text|> from apply_chat_template.
    # Strip it so the tokenizer naturally adds exactly one BOS during SFTTrainer tokenization
    # — avoids the double-BOS that occurs when text already contains the BOS string and the
    # tokenizer also prepends one (Unsloth + DDP does not fully respect add_bos_token=False).
    bos_str = "<|begin_of_text|>"
    train_dataset = train_dataset.map(lambda x: {"text": x["text"].removeprefix(bos_str)})
    eval_dataset = eval_dataset.map(lambda x: {"text": x["text"].removeprefix(bos_str)})

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_TAG,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,  # in-run epoch selection on eval_loss
        metric_for_best_model="eval_loss",
        seed=3407,
        report_to="mlflow",
        # DDP + gradient checkpointing causes reentrant backward that marks params
        # ready twice. This flag disables unused-parameter search, avoiding the conflict.
        ddp_find_unused_parameters=False,
    )
    # ── Token-accuracy metric ──────────────────────────────────────────────────────────────
    def preprocess_logits_for_metrics(logits, labels):
        """Reduce (batch × seq × vocab) logits → argmax IDs before accumulation.

        Without this, the full vocab-dim tensor is kept in host RAM for every eval
        batch and quickly causes OOM on long sequences.  The returned int-tensor is
        passed as `predictions` to compute_token_accuracy below.
        """
        if isinstance(logits, tuple):   # some model heads return (logits, past_kv, …)
            logits = logits[0]
        return logits.argmax(dim=-1)    # (batch, seq_len)  — stays on GPU until numpy copy

    def compute_token_accuracy(eval_pred):
        """Token-level accuracy on supervised (label != -100) positions only.

        Applies the causal LM next-token shift: logits[i] predicts labels[i+1],
        so we align pred_ids[:, :-1] with labels[:, 1:] before masking.
        Masked positions (instruction / padding) are excluded so the number
        reflects how well the model predicts the *JSON response* tokens.
        MLflow receives this as eval_token_accuracy after every eval epoch.
        """
        pred_ids, labels = eval_pred   # both numpy arrays  (n_examples × seq_len)
        # Causal LM next-token alignment: logits[i] predicts labels[i+1]
        pred_ids = pred_ids[:, :-1]
        labels = labels[:, 1:]
        mask = labels != -100
        n_total   = int(mask.sum())
        n_correct = int(((pred_ids == labels) & mask).sum())
        return {"token_accuracy": n_correct / n_total if n_total > 0 else 0.0}

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=args,
        compute_metrics=compute_token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )

    # Re-check masking on what the trainer will ACTUALLY see (rank 0 only).
    if is_main:
        example = trainer.train_dataset[0]
        ids, labels = example["input_ids"], example["labels"]
        supervised_ids = [t for t, l in zip(ids, labels) if l != -100]
        supervised_text = tokenizer.decode(supervised_ids)
        print(f"[mask check] tokens={len(ids)} supervised={len(supervised_ids)}")
        print(f"[mask check] supervised head: {supervised_text[:200]!r}")
        print(f"[mask check] supervised tail: {supervised_text[-60:]!r}")
        assert ids[0] == tokenizer.bos_token_id and ids[1] != tokenizer.bos_token_id, (
            "Double or missing BOS in trainer input_ids — the trainer re-added <|begin_of_text|>."
        )
        assert 0 < len(supervised_ids) < len(ids), "Loss mask is empty or covers the whole sequence."
        assert supervised_text.strip().startswith("{"), "Supervised span must start with the JSON answer."
        assert supervised_text.endswith("<|eot_id|>"), "Supervised span must end with <|eot_id|>."

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # .distributed() already created the MLflow run and injected MLFLOW_RUN_ID —
    # log params/metrics from rank 0 only to avoid duplicate calls.
    if is_main:
        mlflow.log_params({
            "run_tag": RUN_TAG,
            "training_method": "lora_unsloth_ddp",
            "base_model": BASE_MODEL,
            "gpu_type": GPU_TYPE,
            "num_gpus": NUM_GPUS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": ",".join(TARGET_MODULES),
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 4),
            "max_seq_length": MAX_SEQ_LENGTH,
            "train_samples": TRAIN_KEPT,
            "eval_samples": EVAL_KEPT,
            "dropped_overlength_train": TRAIN_DROPPED,
            "dropped_overlength_eval": EVAL_DROPPED,
            "unsloth_version": unsloth.__version__,
            "peft_version": peft.__version__,
            "transformers_version": transformers.__version__,
            "trl_version": trl.__version__,
            "torch_version": torch.__version__,
        })
        mlflow.set_tags({"stage": "train", "approach": "peft-lora-ddp"})

    try:
        train_result = trainer.train()
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()[:4000]
        if is_main:
            mlflow.set_tag("train_error", err_msg)
        raise
    eval_metrics = trainer.evaluate()

    # PEFT save_model writes ONLY the adapter — HF Trainer handles rank-0 save internally.
    trainer.save_model(ADAPTER_DIR)
    if is_main:
        tokenizer.save_pretrained(ADAPTER_DIR)

        run_id = mlflow.active_run().info.run_id
        tok_acc = eval_metrics.get("eval_token_accuracy", float("nan"))
        print(f"Training complete. train_loss={train_result.metrics['train_loss']:.4f} "
              f"eval_loss={eval_metrics['eval_loss']:.4f}  token_acc={tok_acc:.4f}  (MLflow run {run_id})")
        print(f"Versions: unsloth={unsloth.__version__} peft={peft.__version__} "
              f"transformers={transformers.__version__} trl={trl.__version__} torch={torch.__version__}")


print(f"Training function defined for {NUM_GPUS}x{GPU_TYPE} (DDP).")

# COMMAND ----------

# DBTITLE 1,Train
run_training.distributed()

# COMMAND ----------

# DBTITLE 1,Verify the adapter was saved
import os

for _f in ("adapter_config.json", "adapter_model.safetensors"):
    assert os.path.isfile(os.path.join(ADAPTER_DIR, _f)), f"Missing {_f} in {ADAPTER_DIR} — did training finish?"
print(f"Adapter verified at {ADAPTER_DIR}: {sorted(os.listdir(ADAPTER_DIR))}")
print(f"\n>>> run_tag for notebooks 02/03: {RUN_TAG}")

# COMMAND ----------

# DBTITLE 1,Next steps
# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC 1. Copy the printed `run_tag` into **peft-02_merge-and-val-eval** → merges the adapter into
# MAGIC    the bf16 Instruct base and scores the **validation** split (MLflow `stage=eval`).
# MAGIC 2. Compare val runs in MLflow; take the best `run_tag` to **peft-03_register-deploy-test**.
# MAGIC 3. If val F1 falls short: try `lr` 1e-4 / 5e-4, or a larger `lora_r` (keep `alpha/r` fixed).
# MAGIC    If val F1 is high but train loss → \~0 and eval loss rises, try `num_epochs=2`.