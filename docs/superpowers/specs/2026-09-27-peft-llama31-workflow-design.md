# PEFT (LoRA / QLoRA) Fine-Tuning Workflow — Llama 3.1 8B + Unsloth

**Date:** 2026-09-27
**Location:** `LLM_PEFT_finetuning_workflow/notebooks/`
**Status:** Design — awaiting review

## 1. Goal

Build a PEFT counterpart to `LLM_FFT_finetuning_workflow/`: fine-tune **Llama 3.1 8B Instruct** with **Unsloth LoRA**, **merge** the adapter into bf16 base weights, and **serve** the merged model on Databricks Model Serving (vLLM custom entrypoint) — on the **same title-insurance entity-extraction dataset** (OCR text → sparse JSON) and with the **same evaluation metrics**, so PEFT and FFT results are directly comparable.

### Success criteria

- A merged Llama-3.1-8B checkpoint registered in UC (`{catalog}.{schema}.llama31_8b_agency_peft`) and served on endpoint `agency-llama-peft-vllm`.
- Validation F1 (`stage=eval`) and held-out test F1 (`stage=test`) logged to MLflow with the same metric names/definitions as FFT.
- One workflow supports both training modes via a single flag: 4-bit QLoRA on 1×A10 (**default** — cheaper, more available, fast iteration), or bf16 LoRA on 1×H100 (switch when more precision / full-length context is needed).

### Non-goals (YAGNI)

- No hyperparameter sweep notebook / Databricks Job (can be added later; `RUN_TAG` is sweep-ready).
- No new data-prep notebook — reuse FFT notebook 00 outputs.
- No multi-GPU / DDP training.
- No prompt shortening, no change to metric definitions.
- No local pytest suite (in-notebook asserts + an end-to-end workspace run instead, matching the FFT notebooks).

### References

- Starter: `LLM_PEFT_finetuning_workflow/llama_3_1_8b_+_unsloth_2x_faster_finetuning.py` (Unsloth Colab)
- https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth
- https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth-distributed
- FFT notebooks `agency-00` … `agency-05` (data, vLLM eval, register/deploy patterns)

## 2. Architecture

```
FFT notebook 00 (existing) ──► agency_ft_dataset_{train,val,test}_v3  (UC tables)
                                      │
peft-01_train-lora-unsloth  (AI v6, Unsloth, @distributed(gpus=1, gpu_type=<flag>))
   ──► /Volumes/{catalog}/{schema}/{volume_model}/agency-peft-adapter-{RUN_TAG}   (adapter only)
                │
peft-02_merge-and-val-eval  (1×H100, AI v5, vLLM + peft, no Unsloth)
   merge_and_unload into bf16 Instruct base
   ──► /Volumes/{catalog}/{schema}/{volume_model}/agency-peft-merged-{RUN_TAG}    (~16 GB)
   local vLLM on VAL ──► MLflow stage=eval
                │
peft-03_register-deploy-test  (1×H100, AI v5)
   MLflow ChatModel + vLLM entrypoint + env_pack ──► UC model ──► serving endpoint
   ai_query on TEST ──► MLflow stage=test
```

**Why training is separated from merge/eval:** Unsloth and vLLM pin conflicting torch/transformers versions; keeping them in separate notebook environments avoids dependency conflicts. The merge only needs `peft` + `transformers`.

**Compute for notebooks 02/03:** vLLM runs inside the attached notebook compute, so the GPU is chosen in the compute picker (not the flag). Both A10 and H100 work: Llama 3.1 8B uses GQA (8 KV heads, ~128 KB KV cache/token), so on an A10 (24 GB − ~16 GB bf16 weights) there is room for ~1–2 concurrent 20K-token sequences — slower, but functional. H100 is recommended for throughput; on A10 set `max_num_seqs=2`.

### Files

| File | Purpose |
|---|---|
| `notebooks/peft-01_train-lora-unsloth.py` | LoRA/QLoRA training, adapter saved to Volume |
| `notebooks/peft-02_merge-and-val-eval.py` | Merge adapter → bf16 checkpoint; local vLLM validation eval |
| `notebooks/peft-03_register-deploy-test.py` | Register to UC, deploy endpoint, held-out test eval |
| `notebooks/README.md` | Workflow docs in the FFT README style, incl. LoRA-vs-FFT HP notes |

All notebooks are Databricks source format (`# Databricks notebook source`), with `catalog`, `schema`, `volume`, `volume_model`, `experiment_path` widgets (defaults `fins_genai`, `fine_tuning`, `training_data`, `checkpoints`, same as FFT). Extraction `StructType`, fuzzy matcher, and TOP-8 field list are copied from FFT notebook 05 so the PEFT workflow is self-contained.

### RUN_TAG

`RUN_TAG = f"{train_mode}_r{lora_r}_lr{learning_rate}_ep{num_epochs}"` — e.g. `lora_bf16_r16_lr2e-4_ep3`. Built from the widget strings (as FFT does) so it is stable across notebooks. Notebooks 02 and 03 take `run_tag` as a widget.

## 3. Data

- Reuse FFT tables unchanged: `agency_ft_dataset_train_v3` (`prompt`, `response`), `agency_ft_dataset_val_v3` (superset: `prompt`, `response`, `file_name`, `ground_truths`, `raw_ocr_content`), `agency_ft_dataset_test_v3` (`file_name`, `ground_truths`, `raw_ocr_content`).
- The instruction prompt is **already baked into the `prompt` column** by FFT notebook 00 (`agency_prompt.txt` + OCR). The training notebook does not inject it; it only wraps rows as `[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]` (no system turn, same as FFT).
- Eval/serving rebuild the prompt as `open(agency_prompt.txt).read().strip() + "\n" + ocr`. The prompt file ends with `"document:\n"`, so this matches the training form exactly.
- The ~1.5K-token prompt counts toward `max_seq_length` (masked from loss).
- Changing the prompt requires re-running FFT notebook 00.

## 4. Notebook 01 — Training (`peft-01_train-lora-unsloth`)

**Environment:** AI v6, `%pip install unsloth==2026.9.4`. Training function runs under `@distributed(gpus=1, gpu_type=GPU_TYPE)` from `serverless_gpu`. HF datasets are saved to the Volume (`agency_peft_train_dataset`, `agency_peft_eval_dataset`) because the remote worker cannot read the driver's `/tmp`.

`gpu_type` is passed straight to `@distributed` (confirmed as the switch for GPU type).

### Mode flag

`train_mode` widget ∈ {`qlora_4bit` (**default**), `lora_bf16`} sets defaults; other widgets left blank inherit them, non-blank values override. Typical usage: iterate on A10 with `qlora_4bit`; switch to `lora_bf16` (H100) when more precision or full-length context is needed.

| Setting | `lora_bf16` | `qlora_4bit` |
|---|---|---|
| `base_model` | `unsloth/Meta-Llama-3.1-8B-Instruct` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| `load_in_4bit` | False | True |
| `gpu_type` | H100 | A10 |
| `max_seq_length` | 16384 | 4096 |
| `per_device_batch_size` | 1 | 2 |
| `gradient_accumulation_steps` | 8 | 4 |

Shared defaults (widgets): `lora_r=16`, `lora_alpha=16`, `lora_dropout=0`, `target_modules=[q,k,v,o,gate,up,down]_proj`, `learning_rate=2e-4`, `num_epochs=3`, `optim=adamw_8bit`, `use_gradient_checkpointing="unsloth"`, `bias="none"`, `random_state=3407`.

Unsloth mirror repos are used to avoid the gated `meta-llama` HF repo (no HF token). A `base_model` widget overrides the mode default and accepts a UC Volume path to a pre-staged snapshot (G7).

### Tokenizer / template

- Use the Instruct tokenizer's Llama 3.1 chat template (auto-inserts the "Cutting Knowledge Date / Today Date" system header; vLLM applies the same template at serving, so train/serve match).
- **Assert `tokenizer.pad_token != tokenizer.eos_token`.** Unsloth sets pad to `<|finetune_right_pad_id|>`; pad==eos would mask `<|eot_id|>` from labels and the model would never learn to stop.

### Over-length handling

Render each example with the chat template, tokenize, and **drop** examples whose length exceeds `max_seq_length` (train and eval-loss sets). Right-truncation would cut the JSON answer and teach malformed output. Log `dropped_overlength_train` / `dropped_overlength_eval` and print the counts. (Val/test *metric* evals in notebooks 02/03 always use the full split.)

### Response-only loss

`unsloth.chat_templates.train_on_responses_only(trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n", response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")`.

A verification cell (mirroring FFT notebook 01) decodes the supervised span of one example and asserts, before training:
- supervised span is non-empty and shorter than the full sequence;
- decoded span (stripped) starts with `{`;
- span ends with `<|eot_id|>`.

Since `train_on_responses_only` wraps a trainer, the verification is run inside the distributed function immediately after wrapping, printing and asserting on `trainer.train_dataset[0]` before `trainer.train()`. A driver-side preview (rendered template + manual boundary search) is also shown in an earlier cell for interactive inspection.

### Trainer

TRL `SFTTrainer` with `SFTConfig`:
`output_dir=/Volumes/.../{volume_model}/agency-peft-output-{RUN_TAG}`, `num_train_epochs`, `per_device_train_batch_size`, `per_device_eval_batch_size=1`, `gradient_accumulation_steps`, `learning_rate`, `warmup_ratio=0.05`, `lr_scheduler_type="linear"`, `weight_decay=0.01`, `optim="adamw_8bit"`, `bf16=True`, `logging_steps=10`, `eval_strategy="epoch"`, `save_strategy="epoch"`, `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `max_length=max_seq_length`, `packing=False`, `seed=3407`, `report_to="mlflow"`, `run_name=RUN_TAG`.

Epoch selection happens within the run (best `eval_loss` epoch is kept).

### MLflow

`mlflow.set_experiment(experiment_path)`; the run is started inside the distributed function with `run_name=RUN_TAG`, `log_system_metrics=True`. Params: `train_mode`, `base_model`, `load_in_4bit`, `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`, `trainable_params`, `trainable_pct`, `max_seq_length`, `train_samples`, `eval_samples`, `dropped_overlength_train`, `dropped_overlength_eval`, `run_tag`, `training_method="lora_unsloth"`. Tag `stage=train`.

### Output

`trainer.save_model(ADAPTER_DIR)` + `tokenizer.save_pretrained(ADAPTER_DIR)` where `ADAPTER_DIR=/Volumes/{catalog}/{schema}/{volume_model}/agency-peft-adapter-{RUN_TAG}`. Final driver cell asserts `adapter_config.json` and `adapter_model.safetensors` exist and prints `RUN_TAG` for notebook 02.

## 5. Notebook 02 — Merge + Validation Eval (`peft-02_merge-and-val-eval`)

**Environment:** AI v5, 1×A10 or 1×H100 (see Section 2). Install the FFT notebook 05 pins (`vllm==0.11.2 transformers==4.57.6 openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk>=0.102.0`) plus a pinned `peft` (see Section 9, G2), then **in a second `%pip` pass** `opencv-python-headless==4.12.0.88` (G11), then `%restart_python`.

**Widgets:** `run_tag`, `catalog`, `schema`, `volume`, `volume_model`, `experiment_path`, `merge_base_model` (default `unsloth/Meta-Llama-3.1-8B-Instruct`; accepts a Volume path, see G7), `max_model_len` (20480), `max_new_tokens` (3500), `max_num_seqs` (default 2 — safe on A10; raise to 14 on H100), `eval_split` (default `val`).

### Merge

1. Assert adapter files exist.
2. Load `merge_base_model` in bf16 via `AutoModelForCausalLM` — always the bf16 Instruct base, regardless of the adapter's `base_model_name_or_path` (which is the bnb-4bit repo in QLoRA mode).
3. `PeftModel.from_pretrained(base, ADAPTER_DIR)` → keep a reference for the sanity check → `merge_and_unload()`.
4. **Sanity check:** on one val example, greedy-generate (≤256 tokens) with the unmerged PEFT model (before `merge_and_unload()`, which mutates the model) and with the merged model. Assert both outputs are non-empty and start with `{`; print both and warn (not fail) if they differ — bf16 rounding in the merge can flip a greedy token even when the merge is correct, and in `qlora_4bit` mode the bf16 base differs numerically from the 4-bit training base. The val F1 on the merged model is the authoritative check.
5. `save_pretrained(local_tmp, safe_serialization=True)` + tokenizer from `ADAPTER_DIR`, then copy to `MERGED_DIR=/Volumes/.../{volume_model}/agency-peft-merged-{RUN_TAG}`. Assert `config.json` and `*.safetensors` exist. Skip merge if `MERGED_DIR/config.json` already exists (idempotent re-runs).
6. Free GPU memory (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) before starting vLLM.

### Validation eval (same pattern as FFT notebook 02)

- Start `vllm.entrypoints.openai.api_server` on port 3080 from the local merged copy (`--dtype bfloat16 --max-model-len {max_model_len} --enable-prefix-caching --served-model-name llama`), wait for "Application startup complete".
- Send `agency_ft_dataset_{eval_split}_v3` rows via a 4-worker thread pool to `/invocations` (same as FFT notebook 02) with `temperature=0.0`, `max_tokens=max_new_tokens`; prompt = `INSTRUCTION_PROMPT + "\n" + raw_ocr_content`. Failed requests yield empty output.
- Score (Section 7). Log to MLflow: metrics + `json_parse_failures`; params `eval_split`, `eval_table`, `run_tag`, `train_mode` (parsed from the `run_tag` prefix), `merged_dir`, `inference="local_vllm"`, `matching_threshold=0.6`, `documents_scored`; tags `stage=eval` if `eval_split=="val"` else `stage=test`, `approach="peft-lora"`.
- Kill the vLLM server at the end.

## 6. Notebook 03 — Register, Deploy, Test (`peft-03_register-deploy-test`)

Adapted from FFT notebook 05; Qwen-specific pieces (think-tag template patch, `<think>` regex strip) removed.

- **Environment:** AI v5, 1×A10 or 1×H100 (GPU node needed for `env_pack` RAM, G6c), same pins as notebook 02.
- **Widgets:** `run_tag`, `catalog`, `schema`, `volume`, `volume_model`, `experiment_path`, `endpoint_name` (default `agency-llama-peft-vllm`), `workload_type` (default `GPU_LARGE`; `GPU_MEDIUM` = A10), `max_num_seqs` (default 14 for the serving entrypoint; lower for `GPU_MEDIUM`).
- Stage `MERGED_DIR` → local working dir (`tempfile.mkdtemp()`), assert `config.json`.
- `entrypoint(port)` — single source for local test and serving: `--model {ARTIFACTS_PATH} --served-model-name llama --host 0.0.0.0 --port {port} --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization 0.95 --enable-prefix-caching --max-num-seqs {max_num_seqs} --disable-log-requests`.
- Optional local smoke test (port 3080, one extraction request) then stop the server.
- Register: `mlflow.pyfunc.log_model` with a stub `ChatModel`, `artifacts={"model_dir": ARTIFACTS_PATH}`, `metadata={"task": "llm/v1/chat", "entrypoint": entrypoint(8080)}`; `mlflow.register_model(..., UC_MODEL_NAME, env_pack="databricks_model_serving")` with `UC_MODEL_NAME={catalog}.{schema}.llama31_8b_agency_peft`. Version tagged with `run_tag`.
- Poll version status until READY (≤30 min, same logic as FFT).
- Create/update endpoint: `ServedEntityInput(workload_type=GPU_LARGE, workload_size="Small", scale_to_zero_enabled=False)`.
- Smoke query via `w.serving_endpoints.query`.
- Held-out test: `ai_query` over `agency_ft_dataset_test_v3` into `{catalog}.{schema}.agency_inference_output_llama_peft`, `modelParameters => named_struct('max_tokens', 3500, 'temperature', 0.0)`.
- Score (Section 7), log to MLflow run `test_{run_tag}` with tag `stage=test`, `approach="held-out-test"`, param `inference="serving_endpoint_ai_query"`. Per-field breakdown displayed.

## 7. Evaluation metrics (identical to FFT)

- Parse predictions with `from_json` against the FFT extraction `StructType` (copied verbatim from FFT notebook 05).
- Melt predictions and ground truth to (`File_Name`, `field`) rows; **left join on ground truth**, `fillna('NA')` → missing/failed docs count as FN.
- Match: both NA → TN; gt NA & pred not → FP; gt not & pred NA → FN; `SequenceMatcher(None, gt.lower(), pred.lower()).ratio() > 0.6` → TP; otherwise FP (FFT convention — kept for comparability).
- Metrics: `all_precision`, `all_recall`, `all_f1`; `top8_precision`, `top8_recall`, `top8_f1` over `TOP_8_FIELDS` (9 fields, FFT naming).
- **Addition:** `json_parse_failures` = count of outputs where `from_json` yields null (non-empty but unparseable, or empty). Diagnostic only; does not change other metrics.

## 8. Error handling

Fail-fast `assert`s with actionable messages (no broad try/except):
- 01: `pad_token != eos_token`; masking checks; adapter files present after training.
- 02: adapter present before merge; merged `config.json`/safetensors present; vLLM startup via log tail.
- 03: merged checkpoint present; model version READY within timeout else `TimeoutError`/`RuntimeError` naming the cell to re-run.

## 9. Known gotchas (for review)

Each item lists the risk and how the design handles it. These also go into the README's troubleshooting table.

### Unsloth / vLLM version clashes

- **G1 — Never install Unsloth and vLLM in one environment.** Each pins its own torch/transformers/trl; installing one after the other silently swaps torch and yields CUDA/ABI import errors. *Handled:* 01 (AI v6 + Unsloth) is separate from 02/03 (AI v5 + vLLM pins). 01 must not `%pip install vllm`.
- **G2 — Artifacts cross the version boundary.** The adapter is written by 01's newer `peft`/`transformers` and read by 02's older pins; newer `adapter_config.json` keys can be rejected by an older `peft`. *Handled:* 01 logs `peft`, `transformers`, `unsloth`, `torch` versions as MLflow params; 02 pins `peft` to a version ≥ the one in 01 (compatible with `transformers==4.57.6`) and prints both versions before loading, so a mismatch error is immediately diagnosable (no silent config patching). 02 re-saves the tokenizer in its own env so vLLM reads a tokenizer written by the same `transformers` it runs.
- **G3 — Unsloth import order.** `import unsloth` must come before `transformers`/`trl`/`peft`, or its patches don't apply (slower, higher memory, possible masking bugs). *Handled:* first import inside the `@distributed` function and in the driver cell.
- **G4 — Pin Unsloth.** Unpinned `pip install unsloth` pulls a new `unsloth_zoo`/`trl` and breaks `train_on_responses_only` / `SFTConfig` arguments across releases. *Handled:* `unsloth==2026.9.4` (Databricks tutorial version); `trl` taken from Unsloth's dependency resolution, version logged.
- **G5 — torch.compile inside `@distributed`.** If Unsloth compilation errors on the remote worker, set `UNSLOTH_COMPILE_DISABLE=1` (as the distributed tutorial does). *Handled:* documented in troubleshooting; not set by default on 1 GPU.

### GPU quota / availability

- **G6 — H100 capacity is scarce; A10 is more available.** `@distributed` waits for capacity, so an H100 run may sit pending. *Handled:* A10/`qlora_4bit` is the default; 02/03 work on A10 (`max_num_seqs=2`).
- **G6b — A10 memory limits.** QLoRA at 4096 with batch 2 fits 24 GB; raising `max_seq_length` on A10 requires `per_device_batch_size=1` and may still OOM above ~8K. Over-length examples are dropped (logged), not truncated.
- **G6c — Serving GPUs.** Endpoint uses `GPU_LARGE` by default (widget, can be `GPU_MEDIUM` = A10). GPU endpoints without scale-to-zero may be deleted daily by workspace policy (FFT note) — re-run the deploy cell. `env_pack` registration takes 20–30 min for ~16 GB and needs `databricks-sdk>=0.102.0` and a GPU node's RAM (CPU serverless OOMs).
- **G6d — Notebook ports.** Local vLLM must use 3000–3999 on Serverless GPU notebooks (3080); serving uses 8080.

### Hugging Face access

- **G7 — Gated / blocked downloads.** `meta-llama/*` is gated; the design uses `unsloth/*` mirrors (no token). If the workspace blocks `huggingface.co` egress (serverless network policy), downloads fail. *Handled:* `base_model` (01) and `merge_base_model` (02) accept a UC Volume path; README includes a one-time cell to snapshot the models into `/Volumes/{catalog}/{schema}/{volume_model}/base_models/`.
- **G8 — Re-download cost.** Each run downloads the base (~5.5 GB 4-bit, ~16 GB bf16). `HF_HUB_ENABLE_HF_TRANSFER=1` speeds this up; pre-staging (G7) removes it.
- **G9 — Tokens.** If a private/gated repo is used, read `HF_TOKEN` from `dbutils.secrets`, never a literal. Llama 3.1 Community License terms still apply to the mirrors.
- **G10 — Base consistency.** Merge base must be the same weights the adapter was trained against (bf16 Instruct ↔ its bnb-4bit quantization). Mixing Instruct/base or different mirrors silently degrades quality. *Handled:* 01 logs `base_model`; 02 prints it next to `merge_base_model` and asserts they are the matching pair when both are the known mirror IDs.

### Serving environment (vLLM)

- **G11 — opencv FIPS abort.** vLLM can pull `opencv-python-headless>=4.13`, which fails the OpenSSL FIPS self-test on Databricks serverless and aborts vLLM at startup. *Handled:* pin `opencv-python-headless==4.12.0.88` in a **separate second `%pip` pass** after the vLLM install (02 and 03).
- **G12 — Don't upgrade the AI v5 pins.** The v5 environment ships pinned torch/vLLM/transformers; forcing upgrades (e.g. `transformers>=5`) can segfault on ABI mismatch, and there is no supported way to re-pin them on serverless. *Handled:* keep the FFT-validated set (`vllm==0.11.2`, `transformers==4.57.6`, `mlflow==3.12.0`), which supports Llama 3.1. Install `peft` so it cannot drag in `transformers>=5`; print `transformers.__version__` after install and assert it is `4.57.6`. The newer field-validated set (`vllm==0.24.0`, `transformers==5.13.0`, `mlflow==3.14.0` — mlflow 3.12 conflicts with vLLM 0.24 on starlette) is only needed for newer architectures (e.g. Gemma 4); if ever adopted, move all three together.
- **G13 — Log from GPU.** Register the model from a Serverless GPU notebook; logging from CPU packages CPU dependencies and the GPU endpoint fails to start. *Handled:* 03 runs on GPU (also needed for `env_pack` RAM, G6c).
- **G14 — Custom LLM Serving is Beta.** No autoscaling between replicas; no scale-to-zero on H100; H100 serving is region/enrollment-limited. *Handled:* fixed `workload_size="Small"`, `scale_to_zero_enabled=False`; default `workload_type=GPU_LARGE` (widget).
- **G15 — Registration flavor.** The Unsloth tutorial logs the merged model with `mlflow.transformers.log_model(task='llm/v1/chat')`. This design instead uses the Custom LLM Serving pattern (pyfunc `ChatModel` + `metadata.entrypoint` launching vLLM + `env_pack`), matching FFT notebook 05, so serving uses the same vLLM engine and flags as the local val eval. `mlflow.models.predict(env_manager="virtualenv")` pre-deployment validation is not used: with a custom entrypoint `predict` is a stub and never starts vLLM; the local vLLM smoke test in 03 is the real pre-deploy check.
- *Context only:* a past engagement hit CUDA-stack limits on ML Runtime 16.4 LTS for newest vLLM/transformers; not applicable to serverless AI v5/v6 used here.

## 10. README

`LLM_PEFT_finetuning_workflow/notebooks/README.md`, same structure as the FFT README: mermaid architecture, compute/volume layout, per-notebook descriptions, the `train_mode` table, running instructions, troubleshooting, and a short "LoRA vs FFT hyperparameters" section (LR ~10× FFT, alpha/r coupling, in-run epoch selection, when a sweep is worth adding).

The starter files (`Llama_3_1_8b_+_Unsloth_2x_faster_finetuning.ipynb` / `.py`) are left in place as reference.

## 11. Verification plan

1. Upload notebooks to the workspace (requires re-auth: `databricks auth login --profile <profile>`; DEFAULT profile token is currently expired).
2. Run 01 in default `qlora_4bit` mode (A10) with a short smoke config (e.g. `num_epochs=1`) — confirm masking asserts pass, dropped-count logged, adapter saved, MLflow run logged.
3. Run 02 on that `run_tag` (A10 compute, `max_num_seqs=2`) — merge sanity check passes, val F1 logged with `stage=eval`.
4. Run 03 — endpoint serves, test F1 logged with `stage=test`.
5. Run 01 in `lora_bf16` mode (H100) and 02 on its adapter to validate the precision path.
