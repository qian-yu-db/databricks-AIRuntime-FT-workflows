# Qwen3-8B fine-tuning sweep on Databricks AI Runtime

A config-driven **learning-rate × epochs** sweep flow on **AI Runtime + the `air` CLI**
* LLM **Qwen- 3-8B** 
* Tech stack (Axolotl, full-parameter FT with FSDP

---

## Design

- **Model & method.** Qwen3-8B, **full-parameter** SFT (no `adapter:` in the
  Axolotl config — matches the original agency full fine-tune; add `adapter: lora`
  to switch to LoRA).
- **Compute & sharding.** An 8B full fine-tune does not fit on one GPU, so it runs
  on **`GPU_8xH100` with FSDP** (`full_shard auto_wrap`, wrap class
  `Qwen3DecoderLayer`). `run_sweep.py` detects `num_accelerators > 1` and emits an
  `accelerate launch --config_file accelerate_fsdp.yaml -m axolotl.cli.train`
  command; a single-GPU model would instead get a bare `axolotl train`.
- **Sequence length.** `sequence_len: 8192`. Axolotl **drops** (does not truncate)
  examples longer than this; the corpus median is ~6k tokens, so 8192 keeps most of
  the set (2048 emptied it — Axolotl drops over-length examples).
- **Data / prompt format.** ChatML, not `[INST]`. `prep_data.py` strips any residual
  `[INST]/[/INST]` wrapping from the raw OCR and emits ChatML JSONL (system / user =
  prompt + OCR / assistant = the sparse extraction JSON) so Qwen3's own chat template
  applies and loss is masked to the assistant (extraction) turn.
- **The sweep knob.** `configs/grid.yaml` is the only file you edit to resize the
  experiment: `learning_rates[]` × `epochs[]`, cartesian product, one `air` job per
  cell. Each cell writes an HF-format checkpoint to `<checkpoints_dir>/<tag>/` (a
  separate Volume from `data_dir`, so weights don't clutter the training data).
- **End-to-end from one CLI driver.** `run_sweep.py` **trains** (one `air` job per
  grid cell, 8×H100), **evaluates** (`--eval` → a `GPU_1xH100` vLLM job that scores
  every checkpoint), and **registers** the winner (`--register` → a `GPU_1xH100`
  env_pack job to the UC Model Registry). The loop is `prep → submit train →
  --status → --eval → --pick-best → --register`, all from your laptop. The eval +
  register jobs pin `environment.version: databricks_ai_v5` (see
  [CLI vLLM eval on a FIPS workspace](#cli-vllm-eval-on-a-fips-workspace)).
- **Train vs. eval are two phases, not interleaved.** You train the grid, then eval
  whatever checkpoints exist (you can eval a partial sweep). The winner is chosen by
  held-out **eval F1**, not train loss (loss sits near ~0.02 here — rigid JSON
  extraction — and is not a quality signal).

## Layout

```
LLM_finetuning_workflow/cli/
├── configs/
│   ├── grid.yaml            # ← THE knob: learning_rates[], epochs[], model, compute, experiment
│   ├── axolotl_base.yaml    # Qwen3-8B full-FT recipe (template; per-run fields swept)
│   ├── train.air.yaml       # air TRAIN job spec (template, 8xH100)
│   ├── eval.air.yaml        # air EVAL job spec (template, 1xH100, databricks_ai_v5)
│   ├── register.air.yaml    # air REGISTER job spec (template, 1xH100, env_pack to UC)
│   ├── accelerate_fsdp.yaml # FSDP launcher config (Qwen3DecoderLayer wrap)
│   ├── agency_prompt.txt    # instruction prompt prepended to each doc's OCR (used by prep_data.py)
│   └── generated/           # per-run configs written by run_sweep.py (gitignored)
├── scripts/                 # everything runs on your LAPTOP
│   ├── prep_data.py         # raw CSV → ChatML train/val/test.jsonl → upload to UC Volume  (no Spark)
│   ├── run_sweep.py         # expand grid, submit TRAIN + EVAL + REGISTER, --status/--resume/--eval/--pick-best/--register
│   ├── eval_cli.py          # runs ON the eval GPU worker: local vLLM inference + F1 scoring
│   └── register_model.py    # runs ON a GPU worker: log + register the winner to UC (env_pack)
├── lib/                     # pure, Spark-free helpers (unit-tested)
│   ├── extract_eval.py      # scoring + clean_response() (shared by CLI eval + tests)
│   └── prep.py              # data-prep transforms: clean / split / to_chatml
├── tests/                   # pytest suite for lib/ + scripts/  (run: `pytest`)
└── README.md
```

## Prerequisites

1. **Data prep** — run `scripts/prep_data.py` on your laptop once (pure Python + the
   `databricks` CLI; no Spark/cluster). It reads a local raw CSV, builds ChatML
   `train/val/test.jsonl`, and uploads them to the `grid.yaml` `data_dir`
   (`/Volumes/fins_genai/fine_tuning/training_data/qwen_sweep/`):
   ```bash
   uv run --with pandas --with pyyaml python scripts/prep_data.py \
     --input ~/data/raw.csv --profile fevm-classic-stable
   ```
   The raw CSV pairs each document's OCR text (`ocr_text`) with its sparse extraction
   JSON (`extraction_json`); override the column names with `--ocr-col`/`--json-col`.
   It splits 85/5/10 and prints a truncation report at `sequence_len` (read from
   `axolotl_base.yaml`). Use `--no-upload` to inspect the JSONL locally first.
2. **`air` CLI** — install & authenticate the AI Runtime CLI. See the
   [quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart).
   AI Runtime serverless GPU (`GPU_8xH100` for train, `GPU_1xH100` for eval) must be
   enabled in the workspace. **`--eval` needs `air >= 1.1.0`** (older versions reject
   the named `databricks_ai_v5` image with *"Version must be an integer"*) — upgrade
   with `uv tool upgrade databricks-air`.
3. **HF access** — Qwen3 is a gated-free public model; no token normally needed.
   If the environment requires one, set `HF_TOKEN` (Databricks secret) per the CLI docs.

## How to run

The happy path (each step gated by the previous):

```bash
cd LLM_finetuning_workflow/cli
# use `uv run --with pyyaml python ...` if pyyaml isn't installed

# 0. Prepare data (laptop, no GPU): raw CSV -> ChatML JSONL -> UC Volume
uv run --with pandas --with pyyaml python scripts/prep_data.py --input ~/data/raw.csv --profile fevm-classic-stable

# 1. Preview: generate the per-run configs locally, no Databricks calls
python scripts/run_sweep.py --print-only

# 2. Validate against the service without spending GPU (air run --dry-run)
python scripts/run_sweep.py --profile fevm-classic-stable --dry-run --only lr2e-6_ep5

# 3. SMOKE-TEST ONE CELL first (real 8×H100) — prove the whole path before the full grid
python scripts/run_sweep.py --profile fevm-classic-stable --only lr2e-6_ep5 --watch

# 4. Full sweep. --serialize-start sends cells to the GPU one at a time — this is
#    the recommended default (safe on any quota). Drop it only if you have enough
#    GPU quota to run cells in parallel — see "Submission order" below.
python scripts/run_sweep.py --profile fevm-classic-stable --serialize-start

# 5. Evaluate all trained checkpoints — one GPU_1xH100 vLLM job scores every <checkpoints_dir>/<tag>.
#    (Smoke-test one first: --eval --only lr2e-6_ep5 --watch.)
python scripts/run_sweep.py --profile fevm-classic-stable --eval

# 6. Rank the eval runs by F1 and print the winner (local MLflow query, no GPU).
python scripts/run_sweep.py --profile fevm-classic-stable --pick-best

# 7. Register the winning checkpoint to UC as a vLLM ChatModel (GPU_1xH100, env_pack).
#    Register-only — no serving endpoint. Pass the winner tag explicitly.
python scripts/run_sweep.py --profile fevm-classic-stable --register --only lr1e-5_ep4
```

Monitor training:

```bash
# per-cell checkpoint status (local, no GPU) — done vs missing for every grid cell
python scripts/run_sweep.py --profile fevm-classic-stable --status

# or raw air run listing
air list runs --active --profile fevm-classic-stable
air logs <run-id> --profile fevm-classic-stable
```

Training runs log to the MLflow experiment `qwen-ft-sweep`; compare cells' eval
loss there.

### Check status & resume a partial sweep

Completion is judged by the **checkpoint each cell writes**
(`<checkpoints_dir>/<tag>/config.json`), not by `air`'s run status — `air`'s JSON
isn't tag-addressable, and the checkpoint is exactly what `--eval` discovers.

```bash
# What finished, what's missing?
python scripts/run_sweep.py --profile fevm-classic-stable --status

# (Re)submit ONLY the cells with no checkpoint. Auto-bumps the idempotency key so
# previously-FAILED cells actually re-run (a plain re-run would return the old
# failed run). Add --serialize-start to throttle.
python scripts/run_sweep.py --profile fevm-classic-stable --resume --serialize-start
```

You can eval a **partial** sweep any time — `--eval` scores whatever checkpoints
exist under `checkpoints_dir`, so run it on the finished cells and `--resume` the rest later.

### Submission order — serialize by default

**Default to `--serialize-start`.** It runs the sweep one cell at a time: the driver
waits until fewer than `--max-active` (default 1) runs are active before submitting
the next cell. This is safe on any quota and is the recommended way to run.

Without it, `watch: false` (the `grid.yaml` default) submits every cell
fire-and-forget at once. On a limited quota that backfires — the excess cells contend
for the one slot instead of queuing cleanly and can fail with `INTERNAL_ERROR`.

**Only go parallel if you have the GPU quota for it.** If your workspace can actually
run N concurrent 8×H100 jobs, raise throughput with `--serialize-start --max-active N`
(cap in-flight cells at N) — or drop `--serialize-start` entirely to fire them all at
once. Don't do this unless the quota is really there.

`watch: true` also serializes but streams logs inline and blocks your terminal (best
for a single smoke-test cell). Use `--only <tag>` to (re-)run one cell; `--idem-suffix
vN` forces a fresh submit after a failed cell.

### Evaluate & pick the winner

Eval is CLI-driven — the same `run_sweep.py`:

1. `python scripts/run_sweep.py --profile fevm-classic-stable --eval` submits one
   `GPU_1xH100` job (`configs/eval.air.yaml` + `scripts/eval_cli.py`) that loops over
   every `<checkpoints_dir>/<tag>/`, runs local vLLM inference over `test.jsonl`, and logs an
   `eval_<tag>` run tagged `stage=eval` (per-field P/R/F1 + top-8). Add `--only <tag>`
   to eval a single checkpoint. `--eval` submits fresh every time (no idempotency key),
   so re-running scores whatever checkpoints currently exist.
2. `python scripts/run_sweep.py --profile fevm-classic-stable --pick-best` ranks the
   `stage=eval` runs by F1 (pure MLflow query, no GPU).

### Register the winner

Once `--pick-best` names a winner, register that checkpoint to the Unity Catalog Model
Registry as a vLLM-served `ChatModel`:

```bash
python scripts/run_sweep.py --profile fevm-classic-stable --register --only lr1e-5_ep4
```

This submits a `GPU_1xH100` job (`configs/register.air.yaml` + `scripts/register_model.py`)
that stages `<checkpoints_dir>/<tag>/`, logs an MLflow `ChatModel` with a vLLM entrypoint, and calls
`mlflow.register_model(..., env_pack="databricks_model_serving")` — packaging the
worker's vLLM/CUDA environment into a new version of `grid.yaml`'s `registered_model`.
Notes:

- **The tag is required and explicit** — you pass the `--pick-best` winner deliberately;
  there is no auto-select (avoids registering a stale or near-zero checkpoint).
- **It runs on a GPU worker**, not your laptop: `env_pack` tars ~16 GB and needs the
  worker's RAM + the `databricks_ai_v5` image (which has vLLM). Needs `air >= 1.1.0` and
  bundles `databricks-sdk>=0.102.0` (older SDKs hit a 5-min upload timeout).
- **The vLLM entrypoint disables Qwen3 thinking** (`--chat-template-kwargs
  '{"enable_thinking": false}'`) so served completions are clean JSON — the robust
  alternative to string-patching `chat_template.jinja`.
- **Register-only.** It creates a new UC version; it does **not** create a serving
  endpoint. `env_pack` then processes async (~20–30 min) until the version is `READY`.
  Deploying an endpoint on that version is a separate, explicit step.

### CLI vLLM eval on a FIPS workspace

`--eval` runs on `environment.version: databricks_ai_v5` (set in `eval.air.yaml`) —
**not** numeric `5`. This is the one lever that makes a CLI vLLM eval work on a
FIPS-hardened serverless GPU workspace such as `fevm-classic-stable`:

- On numeric `5`, vLLM's model-inspection subprocess imports `cv2`
  (`opencv-python-headless`), which vendors its own **OpenSSL 1.1.1k**. The worker runs
  with `/proc/sys/crypto/fips_enabled=1`, so that vendored lib fails its mandatory FIPS
  self-test and the standard requires `abort()` — a hard `SIGABRT`. vLLM surfaces it as
  the misleading `Model architectures ['Qwen3ForCausalLM'] failed to be inspected`.
- `databricks_ai_v5` is the fuller image; it **preinstalls opencv 4.12.0** whose
  vendored libcrypto is `.so.1.1` (FIPS-clean). The eval deps (`vllm`, `transformers`,
  `mlflow`) don't reinstall opencv, so that clean copy stays in place. FIPS is still on
  (`fips_enabled=1`) — the fix is a FIPS-clean vendored lib, not disabling FIPS.
- **Do not** add an opencv pin to `eval.air.yaml`, and **do not** switch back to `5`.
  Env-level workarounds do not help: pinning `opencv<5` still vendors 1.1.1k;
  `LD_PRELOAD`ing system OpenSSL 3 and Databricks' `OPENSSL_FORCE_FIPS_MODE=0` both
  leave the statically-vendored 1.1.1k (and the kernel FIPS flag) untouched.
- Requires **`air >= 1.1.0`** (older `air` rejects the named image). The repo's notebook
  workflow reaches the same conclusion independently — its eval task runs on
  `base_environment: databricks_ai_v5` too.

Confirmed end-to-end on `fevm-classic-stable` (real 8B checkpoint, 516-doc test set):
vLLM serves, all docs inferred, F1 computed. If a future run regresses, the tell is in
the worker log — a `crypto/fips/fips.c:154 ... FATAL FIPS SELFTEST FAILURE` `SIGABRT`
means opencv's vendored OpenSSL is back; check the image version first.

> **Eval-correctness note.** `lib/extract_eval.py`'s `clean_response()` strips a leaked
> `<think>…</think>` block (and code fences) from each completion before `json.loads`.
> Qwen3's chat template opens a think block in the prompt; without stripping it the
> reasoning leaks into the completion, JSON parsing returns `{}`, and F1 collapses to
> near-zero. Patching `chat_template.jinja` is unreliable (the real template guards the
> empty block with `{%- else %}`, not the `{%- endif %}` a string-replace looks for), so
> `clean_response()` is the fix, applied in `eval_cli.py`'s inference loop.


## Resize the sweep

Edit **only `configs/grid.yaml`** — add/remove entries under `learning_rates` and
`epochs`. `run_sweep.py` takes the cartesian product. Nothing else changes.
