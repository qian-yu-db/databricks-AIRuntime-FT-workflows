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
- **Data / prompt format.** ChatML, not `[INST]`. The source tables have Mistral
  prompt wrapping baked in; `00_prep_data.py` strips it and re-emits ChatML JSONL so
  Qwen3's own chat template applies and loss is masked to the assistant (extraction)
  turn.
- **The sweep knob.** `configs/grid.yaml` is the only file you edit to resize the
  experiment: `learning_rates[]` × `epochs[]`, cartesian product, one `air` job per
  cell. Each cell writes an HF-format checkpoint to `<data_dir>/out/<tag>/`.
- **Train / eval split.** The CLI **trains**; a **notebook evaluates** (local vLLM
  on an interactive GPU cluster). A CLI-native vLLM eval is intentionally not used:
  vLLM's model-inspection subprocess fails an OpenSSL FIPS self-test on FIPS-hardened
  serverless GPU workers, so eval runs in a notebook on an interactive (non-FIPS) GPU
  cluster instead.

## Layout

```
LLM_finetuning_workflow/cli/
├── configs/
│   ├── grid.yaml            # ← THE knob: learning_rates[], epochs[], model, compute
│   ├── axolotl_base.yaml    # Qwen3-8B full-FT recipe (template; per-run fields swept)
│   ├── train.air.yaml       # air job spec (template)
│   ├── accelerate_fsdp.yaml # FSDP launcher config (Qwen3DecoderLayer wrap)
│   └── generated/           # per-run configs written by run_sweep.py (gitignored)
├── scripts/
│   └── run_sweep.py         # expand grid, submit TRAIN runs, --status/--resume/--pick-best  (run on laptop)
├── notebooks/               # everything you run IN the Databricks workspace
│   ├── 00_prep_data.py      # Delta tables → train/val/test.jsonl in UC Volume  (run on Databricks)
│   └── eval_sweep_checkpoints.py  # eval ALL out/<tag> checkpoints via local vLLM (interactive GPU cluster)
└── README.md
```

## Prerequisites

1. **Data prep** — run `notebooks/00_prep_data.py` on Databricks once. It writes
   `train.jsonl` / `val.jsonl` / `test.jsonl` to
   `/Volumes/fins_genai/fine_tuning/training_data/qwen_sweep/` and prints a
   token-length / truncation report at `sequence_len=8192`.
2. **`air` CLI** — install & authenticate the AI Runtime CLI. See the
   [quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart).
   AI Runtime serverless GPU (`GPU_8xH100`) must be enabled in the workspace.
3. **HF access** — Qwen3 is a gated-free public model; no token normally needed.
   If the environment requires one, set `HF_TOKEN` (Databricks secret) per the CLI docs.

## How to run

The happy path (each step gated by the previous):

```bash
cd LLM_finetuning_workflow/cli
# use `uv run --with pyyaml python ...` if pyyaml isn't installed

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

Completion is judged by the **checkpoint each cell writes** (`out/<tag>/config.json`
on the Volume), not by `air`'s run status — `air`'s JSON isn't tag-addressable, and
the checkpoint is exactly what the eval notebook discovers.

```bash
# What finished, what's missing?
python scripts/run_sweep.py --profile fevm-classic-stable --status

# (Re)submit ONLY the cells with no checkpoint. Auto-bumps the idempotency key so
# previously-FAILED cells actually re-run (a plain re-run would return the old
# failed run). Add --serialize-start to throttle.
python scripts/run_sweep.py --profile fevm-classic-stable --resume --serialize-start
```

You can eval a **partial** sweep any time — the eval notebook scores whatever
checkpoints exist under `out/`, so run it on the finished cells and `--resume` the
rest later.

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

Eval runs in a **notebook on an interactive GPU cluster** (not the CLI):

1. Import `notebooks/eval_sweep_checkpoints.py`, attach an interactive GPU cluster
   (A100/H100 for 8B), confirm `DATA_DIR`/`EXPERIMENT_PATH` match `grid.yaml`, Run All.
   It loops over every `out/<tag>/`, runs local vLLM inference over `test.jsonl`, and
   logs an `eval_<tag>` run tagged `stage=eval` (per-field P/R/F1 + top-8).
2. `python scripts/run_sweep.py --profile fevm-classic-stable --pick-best` ranks the
   `stage=eval` runs by F1 (pure MLflow query, no GPU).


## Resize the sweep

Edit **only `configs/grid.yaml`** — add/remove entries under `learning_rates` and
`epochs`. `run_sweep.py` takes the cartesian product. Nothing else changes.
