#!/usr/bin/env python3
"""
run_sweep.py — expand the LR x epochs grid and submit one AI Runtime job per cell.

This is the AI-Runtime replacement for the sweep loop in
agency-01_finetuning.py (which looped `fm.create(... learning_rate=lr)`).
Here each grid cell becomes a generated Axolotl config + a generated `air` job
spec, submitted with `air run`.

Run from your laptop (not on Databricks), from the repo root:

    cd qwen-ft-sweep
    python scripts/run_sweep.py --profile e2_demo_fieldeng                     # train full grid (8xH100)
    python scripts/run_sweep.py --profile e2_demo_fieldeng --serialize-start   # train, throttled (one cell into the GPU at a time)
    python scripts/run_sweep.py --profile e2_demo_fieldeng --dry-run           # generate + validate only
    python scripts/run_sweep.py --profile e2_demo_fieldeng --only lr2e-06_ep5  # one cell
    python scripts/run_sweep.py --profile e2_demo_fieldeng --status            # per-cell checkpoint status (local, no GPU)
    python scripts/run_sweep.py --profile e2_demo_fieldeng --resume            # (re)submit only cells missing a checkpoint
    python scripts/run_sweep.py --profile e2_demo_fieldeng --resume --serialize-start  # resume, throttled
    python scripts/run_sweep.py --profile e2_demo_fieldeng --pick-best         # rank eval runs by F1 (local, no GPU)

Completion is judged by the checkpoint the cell writes (out/<tag>/config.json on the
Volume), NOT by air's run status — air's JSON isn't tag-addressable. So --status and
--resume reflect exactly the checkpoints the eval notebook will find.

This driver TRAINS the sweep (one `air run` per grid cell). Evaluation is done
separately in the notebook `notebooks/eval_sweep_checkpoints.py` on an interactive
GPU cluster — the CLI vLLM eval was removed because it fails the OpenSSL FIPS
self-test on this workspace's serverless GPU workers (see GOTCHAS.md, §2).
--pick-best just queries MLflow and ranks the notebook's eval runs by F1.

Prereqs: the `air` CLI installed & authenticated, and 00_prep_data.py already run
(so train.jsonl / val.jsonl / test.jsonl exist in the Volume).

Deps:  pip install pyyaml   (or: uv run --with pyyaml python scripts/run_sweep.py ...)
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
GENERATED = CONFIGS / "generated"      # per-run configs land here (gitignored)

# macOS Sequoia stamps every file with a com.apple.provenance xattr that can't be
# stripped; air's libarchive packager embeds it as pax xattr headers, and on the
# Linux worker GNU tar materialises a sibling AppleDouble file `._qwen-ft-sweep`.
# The worker then resolves $CODE_SOURCE_PATH to that 163-byte junk file instead of
# the real code dir. This prefix self-corrects: if $CODE_SOURCE_PATH isn't a
# directory, switch to the sibling with the `._` prefix stripped. $CODE is the real
# root. Both train and eval commands prepend it. (The AXOLOTL_DO_NOT_TRACK export is
# added only to the train command — see render_air_config — since eval has no axolotl.)
CODE_RESOLVER = (
    'CODE="$CODE_SOURCE_PATH"; '
    'if [ ! -d "$CODE" ]; then CODE="$(dirname "$CODE")/$(basename "$CODE" | sed \'s/^\\._//\')"; fi; '
)


def lr_tag(lr: float) -> str:
    # 2e-06 -> "2e-06"; keeps run tags short and filesystem-safe.
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def load_grid() -> dict:
    with open(CONFIGS / "grid.yaml") as f:
        return yaml.safe_load(f)


def grid_cells(grid: dict):
    """Yield (lr, epochs, tag) for every cell in the cartesian product."""
    for lr, epochs in itertools.product(grid["learning_rates"], grid["epochs"]):
        yield lr, epochs, f"lr{lr_tag(lr)}_ep{epochs}"


# --- Volume-checkpoint state (the source of truth for "is this cell done") ----
# air's run status is not tag-addressable (its run_name is the experiment slug and
# its run_id != the MLflow run_id), so we key completion off the CHECKPOINT the cell
# writes: <data_dir>/out/<tag>/config.json. This is the same artifact the eval
# notebook discovers, is listable from the laptop via the databricks CLI, and is
# unambiguous per cell. A cell is DONE iff that file exists.

def _volume_out_dir(grid: dict) -> str:
    return grid["data_dir"].rstrip("/") + "/out"


def completed_tags(grid: dict, profile: str) -> set:
    """Tags whose checkpoint dir on the Volume contains config.json (i.e. training
    finished and saved). Uses `databricks fs ls`; returns empty set on any error."""
    out_dir = _volume_out_dir(grid)
    done = set()
    try:
        listing = subprocess.run(
            ["databricks", "fs", "ls", f"dbfs:{out_dir}"]
            + (["--profile", profile] if profile else []),
            capture_output=True, text=True,
        )
        if listing.returncode != 0:
            return done
        tags = [ln.strip() for ln in listing.stdout.splitlines() if ln.strip()]
        for tag in tags:
            sub = subprocess.run(
                ["databricks", "fs", "ls", f"dbfs:{out_dir}/{tag}"]
                + (["--profile", profile] if profile else []),
                capture_output=True, text=True,
            )
            if sub.returncode == 0 and any(
                ln.strip() == "config.json" for ln in sub.stdout.splitlines()
            ):
                done.add(tag)
    except FileNotFoundError:
        print("  ! `databricks` CLI not found; cannot read checkpoint state.", file=sys.stderr)
    return done


def active_run_count(profile: str) -> int:
    """Number of currently-active air runs (for throttled submission). -1 on error."""
    try:
        res = subprocess.run(
            ["air", "--json", "list", "runs", "--active"]
            + (["--profile", profile] if profile else []),
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            return -1
        payload = json.loads(res.stdout)
        return len(payload.get("data", {}).get("runs", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return -1


def render_axolotl_config(grid: dict, lr: float, epochs: int, tag: str) -> Path:
    """Copy axolotl_base.yaml and override the 3 per-run fields."""
    with open(CONFIGS / "axolotl_base.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = grid["data_dir"].rstrip("/")
    cfg["base_model"] = grid["model"]
    cfg["learning_rate"] = float(lr)
    cfg["num_epochs"] = int(epochs)
    cfg["output_dir"] = f"{data_dir}/out/{tag}"
    cfg["mlflow_run_name"] = tag
    # air creates the experiment and exports MLFLOW_RUN_ID on the worker; Axolotl
    # inherits that run, so we do NOT set mlflow_experiment_name here (leaving it
    # would fork a second experiment at a different path). Drop it if present.
    cfg.pop("mlflow_experiment_name", None)
    cfg["datasets"][0]["path"] = f"{data_dir}/train.jsonl"
    cfg["test_datasets"][0]["path"] = f"{data_dir}/val.jsonl"

    GENERATED.mkdir(exist_ok=True)
    out = GENERATED / f"axolotl_{tag}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def render_air_config(grid: dict, tag: str, axolotl_cfg_name: str) -> Path:
    """Copy train.air.yaml and fill in placeholders for this run."""
    with open(CONFIGS / "train.air.yaml") as f:
        air = yaml.safe_load(f)

    project_root = CONFIGS.parent                    # qwen-ft-sweep/
    air["experiment_name"] = grid["mlflow_experiment"]
    air["mlflow_run_name"] = tag
    air["compute"]["num_accelerators"] = grid["num_accelerators"]
    air["compute"]["accelerator_type"] = grid["accelerator_type"]
    # Pin an absolute snapshot root so CODE_SOURCE_PATH resolves predictably
    # regardless of the CWD air is invoked from. The last path segment
    # (qwen-ft-sweep) becomes the code_source folder on the worker.
    air["code_source"]["snapshot"]["root_path"] = str(project_root)
    # AXOLOTL_DO_NOT_TRACK=1 disables axolotl's telemetry. Required here: axolotl
    # 0.13.1's wheel ships without telemetry/whitelist.yaml, so with telemetry ON
    # every `axolotl` invocation crashes at import (FileNotFoundError on that file).
    # Disabling it skips the whitelist load. Prepended (with the shared $CODE
    # resolver) before accelerate so all 8 ranks inherit it.
    resolve = "export AXOLOTL_DO_NOT_TRACK=1; " + CODE_RESOLVER
    # command references the generated axolotl config inside the snapshot.
    # Generated configs live at configs/generated/ relative to the project root.
    cfg_path = f"$CODE/configs/generated/{axolotl_cfg_name}"
    if int(grid["num_accelerators"]) > 1:
        # Multi-GPU (e.g. Qwen3-8B full FT on 8xH100): launch under accelerate so
        # FSDP shards the model across ranks. accelerate_fsdp.yaml lives in
        # configs/ (NOT the snapshot root); its num_processes must match
        # num_accelerators.
        air["command"] = (
            resolve
            + "accelerate launch "
            "--config_file $CODE/configs/accelerate_fsdp.yaml "
            f"-m axolotl.cli.train {cfg_path}\n"
        )
    else:
        # Single-GPU (e.g. Qwen2.5-1.5B): run axolotl directly, no launcher.
        air["command"] = resolve + f"axolotl train {cfg_path}\n"

    out = GENERATED / f"air_{tag}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(air, f, sort_keys=False)
    return out


def pick_best(grid: dict, profile: str):
    """Rank eval runs by F1 via the MLflow API — pure query, runs locally (no GPU).

    Queries the sweep experiment for runs tagged stage=eval and prints the ranking.
    Requires the tracking URI to point at the Databricks workspace; we set it from
    the CLI profile so this works from your laptop.
    """
    import os
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    import mlflow

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.get_experiment_by_name(grid["mlflow_experiment"])
    if experiment is None:
        # air creates the experiment lazily; before any run exists it won't be found.
        print(f"Experiment '{grid['mlflow_experiment']}' not found — train + eval first.")
        return 1
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'eval'",
        order_by=["metrics.all_f1 DESC"],
    )
    if runs.empty:
        print("No eval runs found. Run notebooks/eval_sweep_checkpoints.py "
              "(interactive GPU cluster) to produce stage=eval runs first.")
        return 1

    cols = [c for c in ["params.checkpoint_tag", "metrics.all_f1", "metrics.top8_f1",
                        "metrics.all_precision", "metrics.all_recall"] if c in runs.columns]
    table = runs[cols].rename(columns=lambda c: c.split(".")[-1])
    print("\n=== Sweep eval ranking (by all_f1) ===")
    print(table.to_string(index=False))
    best = table.iloc[0]
    print(f"\nWinner: {best.get('checkpoint_tag', '?')}  "
          f"all_f1={best.get('all_f1', float('nan')):.4f}  "
          f"top8_f1={best.get('top8_f1', float('nan')):.4f}")
    return 0


def wait_for_slot(profile: str, max_active: int, poll_s: int = 30, timeout_s: int = 1800):
    """Block until fewer than `max_active` air runs are active (throttled submission).

    Prevents the failure mode where many cells submitted in the same instant contend
    for a single GPU-quota slot and get INTERNAL_ERROR instead of queuing. Gives up
    after timeout_s (returns anyway) so a stuck query never wedges the driver."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = active_run_count(profile)
        if n < 0:
            print("  (could not read active-run count; proceeding without throttle)")
            return
        if n < max_active:
            return
        print(f"  throttle: {n} active >= cap {max_active}; waiting {poll_s}s ...")
        time.sleep(poll_s)
    print("  throttle: wait timed out; proceeding anyway.")


def print_status(grid: dict, profile: str):
    """Per-cell status table keyed off the Volume checkpoint (done/missing), plus the
    current active-run count. Read-only; the source of truth is out/<tag>/config.json."""
    done = completed_tags(grid, profile)
    all_cells = list(grid_cells(grid))
    n_active = active_run_count(profile)

    print(f"\n=== Sweep status ({grid['mlflow_experiment']}) ===")
    print(f"Grid: {len(grid['learning_rates'])} LR x {len(grid['epochs'])} epochs "
          f"= {len(all_cells)} cells   |   checkpoint dir: {_volume_out_dir(grid)}")
    print(f"{'CELL':<16} {'CHECKPOINT'}")
    n_done = 0
    for _, _, tag in all_cells:
        ok = tag in done
        n_done += ok
        print(f"{tag:<16} {'✓ done' if ok else '· missing'}")
    print(f"\n{n_done}/{len(all_cells)} cells have a saved checkpoint.")
    if n_active >= 0:
        print(f"{n_active} air run(s) currently active.")
    missing = [t for _, _, t in all_cells if t not in done]
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Run `--resume` to (re)submit only the missing cells.")
    else:
        print("All cells complete. Run the eval notebook, then `--pick-best`.")
    return 0


def submit(air_cfg: Path, grid: dict, profile: str, dry_run: bool, tag: str, watch: bool, idem_suffix: str):
    # Idempotency key dedups submissions per cell (re-running the same key returns
    # the existing run instead of double-spending). But that also blocks re-running
    # a FAILED cell after a config fix — air hands back the old failed run. Bump
    # --idem-suffix (e.g. v2) to force a fresh submission with the new snapshot.
    idem_key = f"qwen-sweep-train-{tag}" + (f"-{idem_suffix}" if idem_suffix else "")
    cmd = ["air", "run", "--file", str(air_cfg), "--idempotency-key", idem_key]
    if profile:
        cmd += ["--profile", profile]
    if dry_run:
        cmd += ["--dry-run"]
    elif watch or grid.get("watch"):
        # --watch streams logs inline (blocks until the run ends). Good for a
        # single smoke-test cell; avoid for the full sweep (it serialises the 8).
        cmd += ["--watch"]
    print(f"\n=== {tag} ===\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ! submission for {tag} exited {result.returncode}", file=sys.stderr)
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="", help="Databricks CLI profile")
    ap.add_argument("--dry-run", action="store_true", help="generate + `air run --dry-run` (no GPU spend)")
    ap.add_argument("--only", default="", help="submit only this run tag, e.g. lr2e-06_ep5")
    ap.add_argument("--watch", action="store_true", help="stream logs inline (air run --watch); best with --only")
    ap.add_argument("--idem-suffix", default="", help="append to the idempotency key to force a fresh run (e.g. v2) after a failed cell")
    ap.add_argument("--print-only", action="store_true", help="generate configs, do not call air")
    ap.add_argument("--pick-best", action="store_true", help="rank eval runs by F1 via MLflow (local, no GPU) and exit")
    ap.add_argument("--status", action="store_true", help="print per-cell checkpoint status (local, no GPU) and exit")
    ap.add_argument("--resume", action="store_true", help="(re)submit ONLY cells with no saved checkpoint; auto-bumps the idempotency key so failed cells actually re-run")
    ap.add_argument("--serialize-start", action="store_true", help="throttle submission: wait until active runs < --max-active before submitting the next cell (avoids the same-instant quota collision)")
    ap.add_argument("--max-active", type=int, default=1, help="cap on concurrent active runs when --serialize-start is set (default 1, matching a single 8xH100 quota slot)")
    args = ap.parse_args()

    grid = load_grid()

    # --pick-best and --status are pure local queries; no grid expansion / submission.
    if args.pick_best:
        sys.exit(pick_best(grid, args.profile))
    if args.status:
        sys.exit(print_status(grid, args.profile))

    all_cells = list(grid_cells(grid))

    # --resume: restrict to cells whose checkpoint is missing, and force a fresh
    # idempotency key (else air returns the old FAILED run instead of retrying).
    resume_skip = set()
    idem_suffix = args.idem_suffix
    if args.resume:
        done = completed_tags(grid, args.profile)
        resume_skip = done
        if not idem_suffix:
            # a stable, human-legible suffix so re-resuming is itself idempotent
            # within a resume "generation" but distinct from the original submit.
            idem_suffix = "resume"
        # only count done-tags that are actually part of THIS grid (the Volume may
        # hold checkpoints from earlier, differently-shaped grids).
        grid_done = [t for _, _, t in all_cells if t in done]
        missing = [t for _, _, t in all_cells if t not in done]
        print(f"Resume: {len(grid_done)}/{len(all_cells)} cells already have checkpoints; "
              f"resubmitting {len(missing)} missing: {', '.join(missing) or '(none)'}")
        if not missing:
            print("Nothing to resume. Run the eval notebook, then `--pick-best`.")
            sys.exit(0)

    print(f"Grid: {len(grid['learning_rates'])} LR x {len(grid['epochs'])} epochs = {len(all_cells)} runs")
    print(f"Mode: TRAIN  |  Model: {grid['model']}  |  {grid['accelerator_type']} x{grid['num_accelerators']}")
    if args.serialize_start and not (args.dry_run or args.print_only):
        print(f"Throttle: serialized start, cap {args.max_active} active run(s).")

    rc = 0
    for lr, epochs, tag in all_cells:
        if args.only and args.only != tag:
            continue
        if tag in resume_skip:
            continue
        ax = render_axolotl_config(grid, lr, epochs, tag)
        air = render_air_config(grid, tag, ax.name)
        print(f"generated: {ax.name}, {air.name}")
        if args.print_only:
            continue
        # Throttle BEFORE submitting so we never fire into a full quota (the failure
        # mode that killed 8 cells: many same-instant submits -> INTERNAL_ERROR).
        if args.serialize_start and not args.dry_run:
            wait_for_slot(args.profile, args.max_active)
        rc |= submit(air, grid, args.profile, args.dry_run, tag, args.watch, idem_suffix)

    if args.print_only:
        print(f"\nConfigs written to {GENERATED}/ . Submit manually with `air run --file ...`.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
