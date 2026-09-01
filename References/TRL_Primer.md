# TRL Fine-Tuning Primer
### SFT (PEFT + full-weight) on Databricks Serverless GPU / AI Runtime

**Scope:** supervised fine-tuning only — no RL, no preference optimization. Target tasks: information extraction, QA, classification.
**Verified against:** TRL **v1.9.2**, Transformers v5.x (Aug 2026). TRL moves fast — the version selector at `huggingface.co/docs/trl` is pinned per release; always read the version you install, and pin it in your job.
**Companion doc:** running on Databricks Serverless GPU, evaluation, and shipping to Model Serving live in *TRL Fine-Tuning on Databricks* (`trl-finetuning-on-databricks.md`). This doc stays scoped to TRL mechanics.

---

## Table of contents

1. [Mental model: what TRL is](#1-mental-model)
2. [The API surface you actually need](#2-the-api-surface-you-actually-need)
3. [Protocol for a new OSS model](#3-day-0-protocol-for-a-new-oss-model)
4. [Data preparation by task](#4-data-preparation-by-task)
5. [Determining the parameter set](#5-determining-the-parameter-set)
6. [Memory and throughput math (8B example)](#6-memory-and-throughput-math-for-8b)
7. [Reference configs](#7-reference-configs)
8. [Reading list](#8-reading-list)

---

## 1. Mental model

TRL is **a thin layer over `transformers.Trainer` that owns the data→labels transformation**. That's the whole thing. `SFTTrainer` inherits every method and attribute of `Trainer`; what TRL adds is:

- dataset format detection (language-modeling vs prompt-completion, standard vs conversational)
- chat template application
- **label masking** (`-100` on tokens you don't want in the loss)
- packing / padding-free collation
- PEFT and quantization wiring
- memory-efficient loss kernels (`chunked_nll`, Liger)

Everything else — optimizer, scheduler, FSDP/DeepSpeed, checkpointing, callbacks, `compute_metrics` — is stock HF Trainer.

**The three-layer stack to keep straight when debugging:**

| Layer | Owns | Where things break |
|---|---|---|
| `datasets` / your prep | text, schema, splits | leakage, label noise, format drift |
| TRL `SFTTrainer` | tokenization, masking, packing | loss on the wrong tokens, silent truncation |
| `Trainer` + `accelerate` | optimizer, sharding, checkpointing | OOM, hangs, NCCL, LR schedule |

Most quality failures live in layers 1 and 2. Most infra failures live in layer 3. Diagnose in that order.

---

## 2. The API surface you actually need

### 2.1 The minimal object graph

```python
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

trainer = SFTTrainer(
    model="Qwen/Qwen3-8B",           # str | PreTrainedModel | PeftModel (causal LM only)
    args=SFTConfig(...),             # everything else
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,      # NOT `tokenizer=` — renamed
    peft_config=LoraConfig(...),     # omit for full fine-tune
    quantization_config=None,        # BitsAndBytesConfig here for QLoRA
    compute_metrics=None,            # token-level only; real eval happens offline
)
trainer.train()
```

`SFTConfig` subclasses `TrainingArguments`, so every Trainer argument is available on it. TRL changes four defaults, which matters:

| Arg | `TrainingArguments` | `SFTConfig` |
|---|---|---|
| `learning_rate` | 5e-5 | **2e-5** |
| `gradient_checkpointing` | False | **True** |
| `bf16` | False | **True** (if `fp16` unset) |
| `logging_steps` | 500 | **10** |

### 2.2 The four dataset formats

TRL sniffs the format from the columns. You are choosing between these four:

```python
# 1. Standard language modeling — loss on everything
{"text": "..."}

# 2. Conversational language modeling — chat template applied automatically
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}

# 3. Standard prompt-completion — loss on completion only, by default
{"prompt": "Extract the parties from:\n...", "completion": '{"parties": [...]}'}

# 4. Conversational prompt-completion
{"prompt": [{"role": "user", "content": "..."}],
 "completion": [{"role": "assistant", "content": "..."}]}
```

**For extraction / classification / single-turn QA, use format 3 or 4.** You want the loss on the answer, not on a 3,000-token contract you pasted into the prompt. Training on the prompt tokens for these tasks wastes capacity teaching the model to model input documents it will never need to generate.

Use format 2 (+ `assistant_only_loss=True`) only when you have genuine multi-turn data.

### 2.3 The masking matrix — the single most important table here

| Dataset format | Config | Loss computed on |
|---|---|---|
| language modeling | (default) | every token |
| prompt-completion | (default, `completion_only_loss=None`) | **completion only** |
| prompt-completion | `completion_only_loss=False` | prompt + completion |
| conversational LM | `assistant_only_loss=True` | assistant turns only |
| conversational prompt-completion | both `True` | assistant turns of the completion |

`assistant_only_loss=True` requires the chat template to contain `{% generation %}` / `{% endgeneration %}` markers. TRL auto-patches known families (Qwen3 and friends). **For a brand-new model this is the #1 thing to verify on day 0** — see §3.3.

### 2.4 `SFTTrainer` steps

1. If conversational → apply chat template (`chat_template_path` can override the model's).
2. If `formatting_func` given → collapse to a `text` field first.
3. Tokenize. Prompt and completion are concatenated *after* separate tokenization when both are present.
4. Truncate to `max_length` (**default 1024**), `truncation_mode="keep_start"`.
5. Build `labels`: copy of `input_ids` with `-100` at masked positions.
6. Optionally pack into `max_length` blocks (`packing_strategy`).
7. Collate. Shift-by-one happens in the loss, not the collator.

You can bypass 1–5 entirely by supplying a pre-tokenized dataset with an `input_ids` column (plus optional `labels`, `assistant_masks`, `completion_mask`), or with `dataset_kwargs={"skip_prepare_dataset": True}` and your own collator. Worth knowing for the case where your masking rules are unusual — e.g. you want loss on the JSON *values* but not the structural keys.

### 2.5 Packing and padding-free

- `packing=True` + `packing_strategy="bfd"` (default): best-fit-decreasing bin packing, overflow truncated. **Auto-enables padding-free**, which requires FlashAttention 2 or 3.
- `"bfd_split"`: splits overflowing sequences instead of dropping the tail.
- `"wrapped"`: concatenate-and-chop, cuts mid-example. Fine for continued pretraining, **wrong for prompt-completion SFT** — it will slice examples in half.
- `padding_free=True` standalone: flattens the batch into one sequence with cumulative seqlens. Same FA requirement.

**When to pack:** when your token-length distribution is wide and right-skewed (typical for extraction over documents). If p50 is 300 tokens and p99 is 4,000, unpacked batches are ~85% padding and you're burning most of your compute on nothing. When lengths are uniform (classification with fixed-length inputs), packing buys little.

### 2.6 Loss kernels

- `loss_type="chunked_nll"` (default): identical math to `nll`, but the `lm_head` projection skips `-100` positions and cross-entropy is chunked. Peak activation memory stops scaling with `vocab_size × seq_len`. With modern 150k+ vocabularies this is a large saving — keep it.
- `loss_type="nll"`: the classic path. Auto-selected when `use_liger_kernel=True` (the two are incompatible).
- `loss_type="dft"`: Dynamic Fine-Tuning, a reweighted objective aimed at better generalization. Treat as an experiment, not a default.

### 2.7 Gotchas that will cost you a run

These are the ones that fail *silently* — the job completes, the loss curve looks plausible, and the model is worse than it should be.

1. **`dtype` defaults to `float32` when you pass `model` as a string.** This differs from `from_pretrained`, which infers from config since Transformers v5. An 8B model in fp32 is 32 GB of weights before you've touched an optimizer. Always:
   ```python
   SFTConfig(model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"})
   ```
2. **`max_length=1024` by default.** Your 3k-token contracts get their tails cut off. Set it from your measured token-length distribution (§4.6).
3. **`truncation_mode="keep_start"` is the only supported mode.** If the prompt is long and the completion sits at the end, truncation eats the *label*. Pre-truncate the document body yourself so the instruction + answer always survive.
4. **`report_to` defaults to `"none"`.** No MLflow run, no metrics, nothing in the experiment UI. Set `report_to=["mlflow"]`.
5. **`shuffle_dataset=False` by default.** If your rows are sorted by source, label, or ingest date — which they will be, coming out of a Delta table — packed blocks become homogeneous and batches become correlated. Set `shuffle_dataset=True`.
6. **`packing_strategy="wrapped"` silently splits examples.** Never with prompt-completion data.
7. **`assistant_only_loss=True` on a template without `{% generation %}` markers** either errors or silently trains on everything, depending on the template. Verify.
8. **EOS mismatch when SFT'ing a base model.** If you borrow a chat template via `chat_template_path`, the template's turn terminator must be the tokenizer's EOS, or your model will never stop generating. Set `SFTConfig(eos_token="<|im_end|>")` (or whatever the template uses).
9. **`use_cache=False` is the TRL default** and is correct with gradient checkpointing; remember to flip it back for inference.
10. **`remove_unused_columns=True`** drops your `doc_id`, `source`, etc. Harmless for training, annoying when you want to trace an example back.
11. **`pad_token` on `SFTConfig` is deprecated** (removal in v2.0). Set `tokenizer.pad_token` and pass the tokenizer as `processing_class`. For models with no pad token, use a dedicated token — reusing EOS as pad is fine only because masking handles it, but it hurts if you ever compute loss on padding.

### 2.8 The verification habit

Before every real run, decode one prepared example and look at what is actually being trained on:

```python
def inspect(trainer, n=2):
    ds = trainer.train_dataset
    tok = trainer.processing_class
    for i in range(n):
        ex = ds[i]
        ids, labels = ex["input_ids"], ex.get("labels", ex["input_ids"])
        print("=" * 70)
        print("FULL:\n", tok.decode(ids))
        kept = [t for t, l in zip(ids, labels) if l != -100]
        print("\nIN LOSS:\n", tok.decode(kept))
        print(f"\nlen={len(ids)}  supervised={len(kept)} ({len(kept)/len(ids):.1%})")
```

If `IN LOSS` is not exactly the string you want the model to emit — including the leading whitespace, the EOS, and nothing else — stop and fix it. Roughly half of all disappointing SFT runs die here.

### 2.9 CLI and config files

Since v1.0 TRL ships a unified CLI. For Databricks Jobs this is often cleaner than a notebook, because the config is a versioned artifact:

```bash
trl sft --config configs/qwen3-8b-extract-lora.yaml
```

Every `SFTConfig` field maps to a YAML key or `--flag`. Keep the YAML in your repo next to the eval harness; that pair is your experiment record.

---

## 3. Protocol for a new OSS model

This is a general approach to set up a fine-tuning job for a new OSS model

### 3.1 The order of operations

| # | Step | Time | Kills the effort if... |
|---|---|---|---|
| 1 | License + weights availability | 5 min | license forbids your deployment |
| 2 | `transformers` support | 5 min | needs a `main` install or `trust_remote_code` |
| 3 | Architecture recon from `config.json` | 10 min | MoE / unusual attention breaks your serving path |
| 4 | Tokenizer fertility on *your* corpus | 10 min | 30% more tokens than incumbent = 30% more cost forever |
| 5 | Chat template audit | 15 min | model dependent (e.g. reasoning traces are forced for some OSS model) |
| 6 | Prompted baseline on your eval set | 20 min | already at target → don't fine-tune |
| 7 | Smoke train: 20 steps, 50 examples | 15 min | shape/dtype/attention errors |
| 8 | Overfit test: 8 examples to ~0 loss | 10 min | plumbing is wrong |

Only after 8 passes do you spend real GPU budget.

### 3.2 Architecture review

```python
import json, torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"   # swap in the new one
cfg = AutoConfig.from_pretrained(MODEL)

fields = ["architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "intermediate_size", "vocab_size", "tie_word_embeddings",
          "max_position_embeddings", "rope_theta", "rope_scaling", "sliding_window",
          "head_dim", "num_experts", "num_experts_per_tok", "attention_bias"]
for f in fields:
    if hasattr(cfg, f):
        print(f"{f:32s} {getattr(cfg, f)}")
```

What each one changes for you:

- **`hidden_size` × `num_hidden_layers` × `intermediate_size`** → parameter count → memory budget (§6).
- **`num_key_value_heads` < `num_attention_heads`** → GQA. Affects KV-cache size at serving time, not training much.
- **`vocab_size`** → with 150k+ vocabularies the logits tensor dominates activation memory. Keep `chunked_nll`.
- **`tie_word_embeddings=True`** → embedding and `lm_head` share weights; relevant if you plan to resize the embedding for new special tokens.
- **`num_experts` present** → MoE. Full fine-tuning gets much harder (expert sharding, router aux loss — `router_aux_loss_coef` exists in `SFTConfig` for this). For a first pass on an MoE model, LoRA on the attention + shared layers, not the experts.
- **`sliding_window`** → check that your `max_length` interacts sanely with it.
- **`rope_scaling`** → if you plan long-context extraction, confirm the scaling config is respected by your serving stack too, not just training.

Then get the real LoRA target names rather than guessing:

```python
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
names = {n.split(".")[-1] for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)}
print(sorted(names))
print(f"{sum(p.numel() for p in m.parameters())/1e9:.2f}B params")
```

You'll usually just use `target_modules="all-linear"` (see §5.3), but knowing the names matters when you need to *exclude* something — e.g. the router in an MoE, or `lm_head`.

### 3.3 Chat template audit

This step helps reduce future debug cost

```python
tok = AutoTokenizer.from_pretrained(MODEL)
tmpl = tok.chat_template
print("has chat template:", tmpl is not None)
print("has generation markers:", "{% generation %}" in (tmpl or ""))
print("eos:", tok.eos_token, tok.eos_token_id, "| pad:", tok.pad_token)

msgs = [{"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "ASSISTANT"}]
print(repr(tok.apply_chat_template(msgs, tokenize=False)))
print(repr(tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)))
```

Check for, in order:

1. **A system role at all.** Some templates fold system into the first user turn. If you rely on a long system prompt at inference, train with the same arrangement.
2. **`{% generation %}` markers** → required for `assistant_only_loss`. Absent? Either use prompt-completion format instead (simpler, and my recommendation for your tasks) or supply a patched template via `chat_template_path`.
3. **Injected reasoning scaffolds.** Recent Qwen-family templates insert `<think>` blocks and have thinking/non-thinking modes. If the template emits `<think></think>` before the assistant content and your training targets don't, you've created a train/serve mismatch. Decide explicitly: train with empty think blocks, train with real reasoning traces, or strip the mechanism.
4. **The turn terminator vs `eos_token`.** They must agree.
5. **Trailing whitespace / newline after the generation prompt.** Your completion must start exactly where generation starts. An extra leading space in your `completion` field is a real, measurable quality loss.

### 3.4 Tokenizer fertility on your corpus

This is specific to moving from an existing model to a newer model with a new tokenizer:

- To discover how many tokens a tokenizer produces per unit of text (per character, or per word) 
- Measure candidate model against the incumbent model on *your* data, not on English Wikipedia

A 15% fertility difference is a 15% difference in training cost, serving cost, and effective context — permanently. It's also how you set `max_length`.

```python
import numpy as np
sample = df.select("text").limit(2000).toPandas()["text"].tolist()
for name in ["meta-llama/Llama-3.1-8B-Instruct", MODEL]:
    t = AutoTokenizer.from_pretrained(name)
    n = [len(t(s).input_ids) for s in sample]
    chars = sum(len(s) for s in sample)
    print(f"{name:45s} tok/char={sum(n)/chars:.4f}  p50={np.percentile(n,50):.0f}  p99={np.percentile(n,99):.0f}")
```

- tok/char — the fertility headline. Compare the two directly.
- p50 — median document length in tokens. Sets your typical batch size / throughput.
- p99 — the tail. This is what you set `max_length` from (§4.6, §5.1 Tier A)


### 3.5 The prompted baseline (do not skip)

Before training anything, run your eval harness (companion doc *TRL Fine-Tuning on Databricks* §2) against:

- the new model, zero-shot with a good prompt
- the new model, few-shot (5 is a good default)
- your current production model / prompt
- optionally a frontier model, as the ceiling

Three outcomes:
- **Zero-shot already meets the target** → ship the prompt, no fine-tune. This happens more often each generation, especially for classification.
- **Few-shot is close but latency/cost of long prompts is the problem** → fine-tuning to compress the prompt into weights is a strong, easily-justified win.
- **Both are far off** → fine-tune, and now you have the baseline number that defines "better".

You need this number to answer "did fine-tuning help?" Without it, you're comparing to nothing.

### 3.6 Smoke test ladder

```python
# rung 1: 20 steps, 50 examples, max_steps=20 — catches dtype/attn/template errors
# rung 2: overfit 8 examples for 100 steps, no LoRA dropout, lr 1e-4
#         → train loss must approach ~0 and greedy decode must reproduce the targets verbatim
# rung 3: 1k examples, 1 epoch, full eval → is the metric moving at all?
```

Rung 2 is the plumbing test. If a model cannot memorize 8 examples, your masking, template, or LR is broken, and nothing you do at scale will fix it.

---

## 4. Data preparation by task

Data prep often determines ~80% of the outcome and hyperparameters ~20%, so budget your time accordingly.

### 4.1 Format decision tree

```
Is the output a fixed label from a small closed set?
├─ yes → classification (§4.4)
│        → prompt-completion, completion = the bare label + EOS
└─ no
   ├─ Is the output a structured object? → extraction (§4.2)
   │  → prompt-completion, completion = canonical JSON
   └─ Is the output free text grounded in a context? → QA (§4.3)
      └─ multi-turn? → conversational + assistant_only_loss
         single-turn? → prompt-completion
```

Default to **prompt-completion**. It gives you completion-only loss for free, avoids chat-template surprises, and is trivially inspectable.

### 4.2 Information extraction tas

**Design rules:**

1. **Canonicalize the target.** One and only one valid serialization per label. Sort keys, fix number formatting, fix date format, normalize null (`null`, not `""`, not `"N/A"`, pick one). Two serializations of the same content teach the model that the format is arbitrary.
   ```python
   json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
   ```
2. **No markdown fences in targets.** Ever. If your labels came from an LLM that wrapped them in ` ```json `, strip it.
3. **Emit the schema in the prompt**, identically every time. The model learns to condition on it, and you can extend the schema later with less retraining.
4. **Include negatives.** Documents where a field is genuinely absent, with explicit `null`. Without these, the model hallucinates values for missing fields — the single most common extraction failure mode.
5. **Include hard cases.** Multi-page tables, scanned-then-OCR'd noise, near-duplicate entities. Aim for ~20–30% of the training set to be cases your current pipeline gets wrong.
6. **Keep the prompt short.** Every token of boilerplate is paid at every inference call forever. Fine-tuning is exactly the tool that lets you delete the 800-token instruction block — but only if you delete it during training too.
7. **Decide on field ordering.** Fixed order (sorted keys) is easier to learn and lets you evaluate partial outputs. Do not shuffle.

```python
def to_example(doc_text: str, label: dict, schema_str: str) -> dict:
    return {
        "prompt": f"Extract fields matching this schema. Return JSON only.\n"
                  f"Schema: {schema_str}\n\n---\n{doc_text}\n---",
        "completion": json.dumps(label, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    }
```

**Where labels come from:** distillation from a frontier model is legitimate and fast (check the model provider's terms for your case), but you must human-verify a stratified sample — at minimum the full eval set, plus ~200 training rows. A silver training set with a **gold eval set** is the standard, defensible arrangement. Never let model-generated labels into your eval set unverified; you'll be optimizing toward the teacher's mistakes and won't be able to see it.

### 4.3 QA

Split by subtype, because they have different failure modes and different metrics:

- **Extractive / grounded QA** (answer is in the provided context): teach abstention explicitly. ~10–15% of examples should have a context that does *not* contain the answer, with the target being your canonical refusal string. Otherwise you get a confident fabricator.
- **Closed-book QA**: SFT on 7–8B is mostly teaching *format and style*, not facts. If you need facts, use retrieval. Fine-tuning to inject knowledge into 8B needs far more data than you think and degrades other capabilities.
- **Long-context QA**: watch `max_length` and the truncation direction (§2.7 #3). Put the question *before* the document if you must truncate, so the question always survives.

Keep answer length distribution consistent with what you want at serving time. If your labels average 400 tokens and you want 50-token answers, the model will give you 400.

### 4.4 Classification

`SFTTrainer` supports causal LMs only. If your task is pure single-label classification with a fixed label set and no need for explanation, an encoder or a `AutoModelForSequenceClassification` head on a small model is usually faster, cheaper, and *more accurate* than an 8B generative model. Use the plain HF `Trainer` for that — TRL adds nothing.

Use generative classification with TRL when: labels are hierarchical or numerous, you need a rationale alongside the label, the task is multi-label with structured output, or you're consolidating several tasks into one served model.

If you do go generative:

1. **Completion = the bare label token(s) + EOS.** No "The category is: ". No period. The shorter and more distinct the label strings, the better.
2. **Prefer single-token labels** where possible — check with `tok(label, add_special_tokens=False).input_ids`. Single-token labels let you classify by comparing logits at one position, which is both faster and gives you calibrated probabilities.
3. **Handle imbalance in the data, not the loss.** Upsample rare classes to a floor (say 5% of the majority), or downsample the majority. Report macro-F1 so imbalance can't hide.
4. **Keep a "none/other" class** with real examples.

```python
# constrained scoring at inference — no generation, no parsing, calibrated
import torch
label_ids = {l: tok(l, add_special_tokens=False).input_ids[0] for l in LABELS}
logits = model(**tok(prompt, return_tensors="pt").to(model.device)).logits[0, -1]
probs = torch.softmax(torch.stack([logits[i] for i in label_ids.values()]), dim=-1)
pred = list(label_ids)[probs.argmax().item()]
```

### 4.5 How much data

Rough starting points for a 7–8B model on a narrow task, assuming clean labels:

| Task | Floor (works) | Comfortable | Diminishing returns |
|---|---|---|---|
| Classification, ≤10 classes | 300–500 | 2k–5k | ~20k |
| Extraction, ≤10 fields | 500–1k | 3k–10k | ~30k |
| Extraction, complex nested schema | 2k | 10k–30k | — |
| Grounded QA / style transfer | 1k | 5k–20k | — |

The floors move with model size. Bigger models need less data to reach a given quality (more pretrained capability to steer); smaller models need more and plateau lower. The task-type *ordering* and the quality-over-quantity rule below hold at every size — only the absolute counts shift. Both tables below are heuristic starting points; confirm the real curve with the error-analysis loop (companion doc §2.5), not the table.

**Smaller class (~0.5–3B: Qwen3-0.6B/1.7B, Llama-3.2-1B/3B, Gemma-2B).** Floors up ~2–4×, and expect a lower quality ceiling — a small model can plateau below target no matter how much data you add. Rigid output format (bare label + EOS, strict JSON) is the exception: small models learn it fast, so classification and simple extraction scale better than the multiplier suggests; it's reasoning-heavy and many-field schemas where they lag. Full FT (weights are cheap to fully train) and QLoRA on an A10 are the natural fit at this size.

| Task | Floor (works) | Comfortable | Diminishing returns |
|---|---|---|---|
| Classification, ≤10 classes | 1k–2k | 5k–15k | ~50k |
| Extraction, ≤10 fields | 2k–3k | 10k–30k | ~60k |
| Extraction, complex nested schema | 5k–10k | 30k–100k | — (may never reach 8B quality) |
| Grounded QA / style transfer | 3k–5k | 15k–50k | — |

**Larger class (~30–70B+: Qwen3-32B, Llama-3.3-70B, Mixtral).** Floors down to ~⅓–½ — but the bigger effect is upstream: check the §3.5 baselines first, because few-shot prompting often already meets target and you skip fine-tuning entirely. Data quality matters *more*, not less: a large model has the capacity to faithfully learn your label noise, so a small gold set beats a large silver one even more decisively. LoRA is almost mandatory here (full-FT cost is punishing and LoRA rarely hits its capacity ceiling at these data volumes, §5.6).

| Task | Floor (works) | Comfortable | Diminishing returns |
|---|---|---|---|
| Classification, ≤10 classes | 100–200 | 500–1k | ~5k |
| Extraction, ≤10 fields | 200–500 | 1k–3k | ~10k |
| Extraction, complex nested schema | 500–1k | 3k–10k | — |
| Grounded QA / style transfer | 300–500 | 2k–5k | — |

**Quality beats quantity, sharply.** 1,000 human-verified extraction examples beat 20,000 noisy ones, and this is not a close call. The most reliable performance gain available to you after the first run is: look at 50 errors, find the systematic ones, add 200 targeted examples, retrain.

### 4.6 Splits, leakage, and the length audit

```python
import numpy as np, hashlib

# 1) Deduplicate BEFORE splitting. Exact hash, then near-dup (MinHash / embedding cosine > 0.95).
df["h"] = df["prompt"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
df = df.drop_duplicates("h")

# 2) Split by the natural unit, not by row. Documents, customers, tickets — whatever
#    generates correlated rows. Row-level random split leaks and inflates your metric.
groups = df["doc_id"].unique()
rng = np.random.default_rng(0); rng.shuffle(groups)
n = len(groups); tr, va = groups[:int(.8*n)], groups[int(.8*n):int(.9*n)]
# remaining 10% = test, opened once, at the end

# 3) Token length audit → sets max_length
lens = [len(tok(p + c).input_ids) for p, c in zip(df.prompt, df.completion)]
for q in (50, 90, 95, 99, 100):
    print(f"p{q}: {np.percentile(lens, q):.0f}")
```

Set `max_length` at roughly **p99**, then confirm what fraction of examples that truncates. Setting it at p100 to be safe wastes memory on every batch; setting it at p90 silently destroys 10% of your labels.

**Three splits** Validation is for model selection and hyperparameters — you will overfit to it, that's what it's for. Test is opened once, at the end, to produce the number you report. If you tune against test, you have no honest estimate of anything.

### 4.7 Delta → training set on Databricks

Keep the pipeline in Unity Catalog so the training set is a versioned, governed artifact.

Then in the training function, `datasets.load_dataset("json", data_files=...)` off the volume path. Record the **Delta table version** in your MLflow run — `DESCRIBE HISTORY` gives it to you — so any model can be traced to the exact rows that produced it. This is the single highest-leverage piece of MLOps hygiene in the whole loop; six weeks later "which data made this model?" is otherwise unanswerable.

---

## 5. Determining the parameter set

### 5.1 Tier the parameters

Not everything deserves a sweep. Sort every knob into one of three buckets and only spend compute on the middle one.

**Tier A — computed, not searched** (derive from data/hardware, then freeze):
`max_length` (p99 of token lengths), `per_device_train_batch_size` (largest that fits), `gradient_accumulation_steps` (to reach target effective batch), `bf16=True`, `gradient_checkpointing`, `packing`, `attn_implementation`, `loss_type`.

**Tier B — actually search** (3–8 runs total):
`learning_rate` (biggest lever by far), LoRA `r`, `num_train_epochs`, and — for extraction/QA — whether prompt tokens are in the loss.

**Tier C — leave alone until you have evidence** :
`adam_beta*`, `weight_decay` (0.0 for LoRA, 0.0–0.1 full FT), `lr_scheduler_type` ("cosine"), `warmup_ratio` (0.03), `max_grad_norm` (1.0), `lora_dropout` (0.0–0.05), `neftune_noise_alpha`, `label_smoothing_factor`.

### 5.2 Starting points for 7–8B

| Parameter | LoRA | Full fine-tune |
|---|---|---|
| `learning_rate` | **1e-4** (search 5e-5 → 4e-4) | **1e-5** (search 5e-6 → 2e-5) |
| `lr_scheduler_type` | cosine | cosine |
| `warmup_ratio` | 0.03 | 0.03–0.05 |
| effective batch (seqs) | **≤32** | 64–256 |
| `num_train_epochs` | 2–3 | 1–2 |
| `weight_decay` | 0.0 | 0.0–0.1 |
| `max_grad_norm` | 1.0 | 1.0 |
| optimizer | `adamw_torch_fused` | `adamw_torch_fused` (or 8-bit) |

Two non-obvious entries:

- **LoRA wants a ~10× higher LR than full fine-tuning.** The 1/r scaling makes the optimal LR roughly rank-independent, so you don't need to re-tune LR when you change rank.
- **LoRA is less tolerant of large effective batch sizes** — keep under ~32 sequences. Raising the rank does not compensate. This is the opposite of the full-FT instinct.

### 5.3 LoRA configuration

```python
from peft import LoraConfig
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,               # alpha = 2r is a fine convention; alpha/r is the scale
    lora_dropout=0.0,
    bias="none",
    target_modules="all-linear", # attention AND MLP — not just q/k/v/o
    task_type="CAUSAL_LM",
)
```

**`target_modules="all-linear"` is the load-bearing choice.** Attention-only LoRA underperforms even at higher rank to match parameter count. Include the MLP projections.

**Choosing `r` — a capacity argument, not a vibe.** LoRA has finite capacity; when the dataset exceeds it, LoRA plateaus above full-FT loss. So:

| Situation | Start at |
|---|---|
| Single narrow task, <5M training tokens (most extraction/classification jobs) | `r=16–32` |
| Multi-task, or 5–50M tokens | `r=64–128` |
| Broad post-training-scale mixture | `r=256` |

**How to tell you're capacity-limited:** train the same config at `r` and `2r`. If train loss at `2r` ends materially lower, you were capacity-limited — raise `r`. If the curves overlap, `r` is sufficient and going higher just costs memory. This is a 2-run experiment and it removes rank from the guesswork permanently.

Do not use QLoRA unless memory forces it. On an H100 with an 8B model it buys nothing but quantization error and slower steps.

### 5.4 The search protocol

Four stages: each one is cheap and each one catches a different class of error.

This protocol is written LoRA-first (§5.6), but the four-stage shape is method-agnostic — only two things change for full fine-tuning. In **Stage 1**, read the second LR triplet (5e-6 / 1e-5 / 2e-5) instead of the LoRA one, and use a correspondingly lower LR in the Stage 0 overfit test. In **Stage 3**, the `r → 2r` capacity test is LoRA-only; for full FT replace that bullet with a full-FT knob — effective batch size, or `weight_decay` 0.0 → 0.1 (Tier C, §5.1). Stages 2 and 4 are identical for both.

**Stage 0 — Overfit 8 examples.** (5 min, 1 GPU) Loss → ~0, greedy decode reproduces targets. Validates masking, template, EOS, LR sanity.

**Stage 1 — LR sweep on a 1k subset.** (~1 GPU-hour total) 3 runs at 3e-5 / 1e-4 / 3e-4 (LoRA) or 5e-6 / 1e-5 / 2e-5 (full FT), 1 epoch each, same seed. Compare **eval loss and your real task metric**, not train loss.
- Diverging / spiky loss, `grad_norm` climbing → LR too high.
- Loss barely moves, `mean_token_accuracy` flat → LR too low.
- Pick the highest LR that trains stably; then optionally halve it for the full-data run, since more data means more steps.

**Stage 2 — Full-data run at the winning LR.** 2–3 epochs, eval every ~200 steps, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`. Watch for the epoch-boundary loss drop — a visible step down at each epoch start is memorization, and eval will follow it down then up.

**Stage 3 — One or two targeted variations**, changed one at a time:
- rank `r` → `2r` (capacity test)
- 2 epochs → 3 epochs (or early-stop)
- prompt tokens in loss on/off
- packing on/off

**Stage 4 — Seed check.** Re-run the winner with 2–3 seeds. On small eval sets, seed variance is frequently larger than the differences you've been agonizing over. If your "improvement" is inside seed noise, it isn't one.

Log every run to one MLflow experiment with the same tag set (`base_model`, `data_version`, `method`, `r`, `lr`, `epochs`) so the comparison is a table, not archaeology.

### 5.5 Reading the training signals

TRL logs these; each one tells you something specific:

| Signal | Healthy | Trouble |
|---|---|---|
| `loss` | smooth decline, flattening | spikes → LR/data outliers; step-drops at epoch bounds → memorization |
| `mean_token_accuracy` | rises to 0.7–0.95 for constrained outputs | stuck low → masking wrong, or task genuinely hard |
| `entropy` | declines and stabilizes | collapse to ~0 → overfitting/degenerate; rising → instability |
| `grad_norm` | stable, near `max_grad_norm` early then below | climbing → LR too high; ~0 → nothing is learning (frozen params? adapter not attached?) |
| `num_tokens` | linear | flat/low → dataloader starving, or over-truncation |

For extraction and classification, `mean_token_accuracy` above ~0.95 within the first epoch usually means the task is easy and you should check for leakage, not celebrate.

### 5.6 Full fine-tune vs LoRA — when to choose which

Choose **LoRA** by default for your tasks. It matches full FT on post-training-scale data when configured per §5.3, uses far less memory, trains ~1/3 cheaper, produces a ~100–500 MB artifact instead of 16 GB, and lets you serve several adapters against one base.

Choose **full fine-tuning** when:

- you're changing the output distribution broadly (new language, new format across many tasks)
- you have >50M training tokens on a genuinely broad mixture
- LoRA at `r=256` still plateaus above the full-FT loss floor
- you need to modify tokenizer/embeddings (new special tokens)
- serving cost of adapter merging is a real constraint (it usually isn't — you can merge)

---

## 6. Memory and throughput math

This example is for 8B param LLM

### 6.1 Static memory

For an 8B model in bf16 mixed precision with AdamW:

| Component | Bytes/param | 8B total |
|---|---|---|
| Weights (bf16) | 2 | 16 GB |
| Gradients (bf16) | 2 | 16 GB |
| Adam `m` + `v` (fp32) | 8 | 64 GB |
| fp32 master weights | 4 | 32 GB |
| **Full FT total** | **~16** | **~128 GB** |

So **full fine-tuning of 8B does not fit on 1xH100 (80G).** It fits comfortably on 8×H100 with FSDP `full_shard` (~16 GB/GPU of state, leaving ~60 GB for activations per GPU).

For LoRA `r=32`, `all-linear` on 8B: ~80–90M trainable parameters.

| Component | 8B + LoRA r=32 |
|---|---|
| Frozen weights (bf16) | 16 GB |
| Adapter weights + grads | ~0.4 GB |
| Adam states (fp32) | ~0.7 GB |
| Activations (checkpointed, 4k ctx, bs 2) | 4–10 GB |
| **Total** | **~22–28 GB** |

LoRA on 8B fits a single H100 with room to spare. It does *not* comfortably fit an A10 (24 GB) — for A10 you need QLoRA (4-bit base ≈ 5.5 GB).

### 6.2 Configuration by hardware

| Hardware | Full FT 8B | LoRA 8B | QLoRA 8B |
|---|---|---|---|
| 1×A10 (24 GB) | ✗ | ✗ (borderline, short ctx only) | ✓ |
| 1×H100 (80 GB) | ✗ | ✓ (ctx to ~8k) | ✓ |
| 8×H100 (single node) | ✓ FSDP full_shard | ✓ DDP or FSDP, ~8× throughput | ✓ |

For your workload the sweet spot is almost always **1×H100 for triage and LoRA, 8×H100 for full FT or for wall-clock-sensitive LoRA sweeps**.

### 6.3 OOM diagnoise ladder

Apply in this order; stop as soon as it fits. Earlier items cost you less quality/throughput than later ones.

1. `per_device_train_batch_size` ↓, `gradient_accumulation_steps` ↑ (same effective batch, no quality change)
2. `gradient_checkpointing=True` (already the TRL default)
3. `packing=True` / `padding_free=True` — often a *large* saving on skewed length distributions
4. `loss_type="chunked_nll"` (default; verify it's not been overridden by Liger)
5. `max_length` ↓ — only after re-checking your length percentiles
6. `activation_offloading=True` (CPU offload of activations; costs throughput)
7. 8-bit optimizer: `optim="adamw_bnb_8bit"` — cuts optimizer state from 8 to ~2 bytes/param
8. FSDP `full_shard`, then add `fsdp_offload_params`
9. Switch full FT → LoRA
10. LoRA → QLoRA

A note on FSDP: `fsdp_transformer_layer_cls_to_wrap` must name the actual decoder layer class of the *new* model (e.g. `Qwen3DecoderLayer`, `Gemma3DecoderLayer`). Get it wrong and you either OOM or silently wrap nothing. Print it: `type(model.model.layers[0]).__name__`.

---

## 7. Reference configs

### 7.1 LoRA, 8B, extraction — 1×H100

```python
SFTConfig(
    output_dir="/Volumes/main/ml/ckpt/extract-lora",
    model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    max_length=2048, packing=True, packing_strategy="bfd", shuffle_dataset=True,
    completion_only_loss=True,
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    num_train_epochs=3,
    per_device_train_batch_size=8, gradient_accumulation_steps=4,   # eff. 32
    bf16=True, gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=100, save_steps=100, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    report_to=["mlflow"], logging_steps=10, seed=42,
)
LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
           target_modules="all-linear", task_type="CAUSAL_LM")
```

### 7.2 Full fine-tune, 8B — 8×H100 FSDP

```python
SFTConfig(
    output_dir="/Volumes/main/ml/ckpt/extract-full",
    model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    max_length=2048, packing=True, shuffle_dataset=True, completion_only_loss=True,
    learning_rate=1e-5, lr_scheduler_type="cosine", warmup_ratio=0.05,
    num_train_epochs=2,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,   # eff. 128
    bf16=True, gradient_checkpointing=True, optim="adamw_torch_fused",
    fsdp="full_shard auto_wrap",
    fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
                 "activation_checkpointing": True,
                 "state_dict_type": "SHARDED_STATE_DICT"},
    save_strategy="steps", save_steps=200, save_total_limit=2,
    eval_strategy="steps", eval_steps=200,
    report_to=["mlflow"], logging_steps=10, seed=42,
)
```

### 7.3 QLoRA, 8B — 1×A10

```python
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

SFTConfig(
    max_length=1024, packing=True, completion_only_loss=True, shuffle_dataset=True,
    learning_rate=2e-4, num_train_epochs=3,
    per_device_train_batch_size=2, gradient_accumulation_steps=16,  # eff. 32
    bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
    report_to=["mlflow"], seed=42,
)
# pass quantization_config=quantization_config to SFTTrainer alongside peft_config
```

### 7.4 Classification, 8B generative — single-token labels

```python
SFTConfig(
    max_length=1024, packing=False,          # uniform lengths; packing buys little
    completion_only_loss=True,               # loss on the label token + EOS only
    learning_rate=1e-4, num_train_epochs=2,
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    bf16=True, gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=50,
    report_to=["mlflow"], seed=42,
)
LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
```

---

## 8. References

**Primary:**

1. [TRL docs — *SFT Trainer*, *Dataset Formats*, *Chat Templates*, *Reducing Memory Usage*, *Distributing Training*](https://huggingface.co/docs/trl)
2. [TRL docs — *LoRA Without Regret*](https://huggingface.co/docs/trl/lora_without_regret)
4. [PEFT docs — *conceptual guide*](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora)
5. [Databricks — *AI Runtime Overview*](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
6. [Databricks — *AI Runtime CLI*](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/)
6. [Databricks — *AI Runtime with Genie Code*](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/genie-code)

**When you need to go deeper:**
- `trl/trainer/sft_trainer.py` — read `_prepare_dataset` and the collators. It's ~1,000 readable lines and it answers every masking question definitively.
- LoRA (Hu et al., 2021) and QLoRA (Dettmers et al., 2023) for the mechanics
- The `accelerate` FSDP docs, for the FSDP1/FSDP2 config differences that bite on upgrades

---

## Appendix: one-page checklist

This is the train/config half. The eval and ship items live in the companion doc
*TRL Fine-Tuning on Databricks* (§5).

**Before you train**
- [ ] Data deduplicated, split by group, three splits
- [ ] Targets canonicalized; negatives and hard cases included
- [ ] Token length audit → `max_length` at p99
- [ ] Chat template audited (generation markers, EOS, reasoning scaffolds)
- [ ] `inspect()` shows exactly the right supervised span
- [ ] Overfit-8 test passes

**In the config**
- [ ] `model_init_kwargs={"dtype": torch.bfloat16, ...}`
- [ ] `max_length` set explicitly (not 1024)
- [ ] `shuffle_dataset=True`
- [ ] `report_to=["mlflow"]`
- [ ] `packing_strategy` ≠ `"wrapped"` for prompt-completion
- [ ] LoRA: `target_modules="all-linear"`, lr ≈ 1e-4, effective batch ≤ 32
- [ ] Full FT: lr ≈ 1e-5, FSDP layer class name verified
