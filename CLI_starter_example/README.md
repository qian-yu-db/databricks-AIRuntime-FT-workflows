# Minimal Axolotl + `air` CLI example

The smallest possible fine-tune you can run **directly from the `air` CLI on YAML
files** — no Python driver, no data prep. It full-fine-tunes **Qwen2.5-0.5B** on a
tiny public alpaca dataset in a few minutes on a single GPU.

```
CLI_example/
├── sft.yaml         # the Axolotl recipe (model, dataset, schedule)
├── train.air.yaml   # the AI Runtime job spec that runs `axolotl train sft.yaml`
└── README.md
```

## Prerequisites

- **`air` CLI ≥ 1.1.0**, authenticated to your workspace
  (`air --version`; upgrade with `uv tool upgrade databricks-air`).
- A **profile with `GPU_1xA10` quota** (serverless GPU enabled). A10s are widely
  available; a 0.5B full fine-tune doesn't need an H100. Supported accelerator types
  are `GPU_1xA10`, `GPU_1xH100`, and `GPU_8xH100` (`air -h config.compute` lists them).
- Nothing else — the model and dataset are pulled from the Hugging Face Hub at
  run time.

## Using the `air` CLI

The [AI Runtime (`air`) CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/)
takes a job spec (`train.air.yaml`), uploads the folder as a code snapshot, provisions
a serverless GPU worker, installs the environment, and runs your command on it.

```bash
# One-time: install / upgrade (needs >= 1.1.0) and check it
uv tool install databricks-air      # or: uv tool upgrade databricks-air
air --version

# Auth uses your Databricks CLI profiles (~/.databrickscfg). Verify the profile works:
databricks current-user me --profile <your-profile>
```

Then, from **inside this folder** (`train.air.yaml` has `root_path: .`):

```bash
cd CLI_example

# 1. Validate the spec against the service — no GPU spend
air run --file train.air.yaml --profile <your-profile> --dry-run

# 2. Submit the training run (returns a Job Run ID immediately)
air run --file train.air.yaml --profile <your-profile>

#    …or submit AND stream logs live (blocks until the run finishes)
air run --file train.air.yaml --profile <your-profile> --watch

# 3. Monitor a run you didn't --watch
air logs <run-id> --profile <your-profile>     # prints logs, or the terminal failure reason
air list runs --profile <your-profile>         # recent runs
```

Common `air run` flags:

| Flag | What it does |
| --- | --- |
| `--profile <name>` | which Databricks CLI profile / workspace to use |
| `--dry-run` | build + validate the job spec, print the payload, **don't** run (no GPU) |
| `--watch` | submit and stream worker logs inline until the run ends |
| `--idempotency-key <k>` | dedupe resubmits — re-running the same key returns the existing run instead of spending again |

> First-run timing: provisioning the GPU + installing the axolotl env takes several
> minutes before you see training logs. Prefer submitting without `--watch` and polling
> `air logs <run-id>`, or watch it in the Jobs UI (the submit prints a run URL).

## Where the results go

- **Metrics** (train/eval loss, etc.) are logged to the MLflow experiment
  `axolotl-cli-example` — `air` creates it under `/Users/<you>/axolotl-cli-example`
  and Axolotl logs into that run automatically.
- **The trained model** goes to `output_dir` in `sft.yaml`, which defaults to the
  worker-local `/tmp/axolotl-qwen05-sft` and is **discarded when the job ends**. To
  keep it, point `output_dir` at a UC Volume you can write to:
  ```yaml
  output_dir: /Volumes/<catalog>/<schema>/<volume>/axolotl-qwen05-sft
  ```

## Make it your own

Everything is in the two YAML files — edit and re-run:

- **Different model** — change `base_model` in `sft.yaml` (any HF model id).
- **Bigger model** — switch `accelerator_type` to `GPU_1xH100`, or raise
  `num_accelerators` and launch under `accelerate` for multi-GPU sharding (see the
  8B FSDP example in [`../LLM_finetuning_workflow/cli/`](../LLM_finetuning_workflow/cli/)).
- **Your own data** — replace the `datasets:` entry. Point `path:` at another HF
  dataset, or at a JSONL file, and set the matching `type:` (`alpaca`, `chat_template`, …).
- **Bigger/longer run** — bump `num_epochs`, `sequence_len`, or `micro_batch_size`.
- **LoRA instead of full FT** — *not out-of-the-box here.* Adding an `adapter: lora`
  block triggers this `peft`'s LoRA→torchao path, which requires `torchao>0.16.0`,
  but axolotl 0.13.1 pins `torchao==0.13.0` and overriding it breaks the env build.
  For a 0.5B model full FT is cheap and avoids the conflict; use LoRA only once a
  newer axolotl relaxes that pin.

> This is the stripped-down version of the config-driven sweep in
> [`../LLM_finetuning_workflow/cli/`](../LLM_finetuning_workflow/cli/README.md),
> which adds a grid sweep, held-out eval, ranking, and UC registration on top of the
> same `air` + Axolotl foundation.
