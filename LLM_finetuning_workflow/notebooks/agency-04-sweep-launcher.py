# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Sweep Launcher
# MAGIC %md
# MAGIC # Agency Fine-Tuning Sweep Launcher
# MAGIC
# MAGIC This notebook submits multiple runs of the **Agency Fine-Tuning Hyperparameter Sweep** job
# MAGIC (created by notebook 03) with different `learning_rate` × `num_epochs` combinations.
# MAGIC The job ID is discovered dynamically by name — no hardcoding needed.
# MAGIC
# MAGIC Each run executes: `train` (8×H100) → `eval` (1×H100).  
# MAGIC Data setup is run **once** before the sweep (only if tables are missing).
# MAGIC
# MAGIC **Usage:**
# MAGIC 1. Edit `SWEEP_GRID` below to define your combinations
# MAGIC 2. Run all cells to submit the runs
# MAGIC 3. Monitor progress in the Jobs UI or via the status-check cell

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Discover the sweep job by name (created by notebook 03-job-setup).
# This avoids hardcoding a job ID that changes if the job is recreated.
JOB_NAME = "Agency Fine-Tuning Hyperparameter Sweep"
_matching_jobs = list(w.jobs.list(name=JOB_NAME))
assert _matching_jobs, f"No job found with name '{JOB_NAME}' — run notebook 03 first."

# If duplicates exist, pick the most recently created (highest job_id) and warn.
if len(_matching_jobs) > 1:
    _matching_jobs.sort(key=lambda j: j.job_id, reverse=True)
    print(f"⚠️  Found {len(_matching_jobs)} jobs named '{JOB_NAME}' — using newest (ID {_matching_jobs[0].job_id}).")
    print(f"    Consider deleting stale duplicates: {[j.job_id for j in _matching_jobs[1:]]}")

JOB_ID = _matching_jobs[0].job_id
print(f"Resolved job: '{JOB_NAME}' → ID {JOB_ID}")

# === Edit this grid to change sweep combinations ===
SWEEP_GRID = [
    {"learning_rate": "5e-6", "num_epochs": "3"},
    {"learning_rate": "1e-5", "num_epochs": "3"},
    {"learning_rate": "2e-5", "num_epochs": "3"},
    {"learning_rate": "5e-6", "num_epochs": "4"},
    {"learning_rate": "1e-5", "num_epochs": "4"},
    {"learning_rate": "2e-5", "num_epochs": "4"},
    {"learning_rate": "5e-6", "num_epochs": "5"},
    {"learning_rate": "1e-5", "num_epochs": "5"},
    {"learning_rate": "2e-5", "num_epochs": "5"},
]

# MLflow experiment to query for eval results (must match what the job logs to)
EXPERIMENT_PATH = "/Users/q.yu@databricks.com/mlflow_experiments/agency-finetuning-ai-runtime-sweep"


print(f"Job ID: {JOB_ID}")
print(f"Grid: {len(SWEEP_GRID)} combinations")
for i, combo in enumerate(SWEEP_GRID, 1):
    print(f"  [{i}] lr={combo['learning_rate']}, epochs={combo['num_epochs']}")

# COMMAND ----------

# DBTITLE 1,Ensure data tables exist (run notebook 00 once if needed)
# The sweep job no longer includes data_setup (to avoid redundant work on every run).
# This cell checks if the fine-tuning tables exist; if not, it runs notebook 00 once.

DATA_SETUP_NOTEBOOK = "/Users/q.yu@databricks.com/AIR_migration_example/dev/notebooks/agency-00-setup-datasets"
REQUIRED_TABLES = [
    "fins_genai.fine_tuning.agency_master_dataset_v3",
    "fins_genai.fine_tuning.agency_ft_dataset_train_v3",
    "fins_genai.fine_tuning.agency_ft_dataset_val_v3",
]

missing = [t for t in REQUIRED_TABLES if not spark.catalog.tableExists(t)]

if missing:
    print(f"Missing tables: {missing}")
    print(f"Running {DATA_SETUP_NOTEBOOK} ...")
    result = dbutils.notebook.run(
        DATA_SETUP_NOTEBOOK,
        timeout_seconds=600,
        arguments={"catalog": "fins_genai", "schema": "fine_tuning", "volume": "training_data"},
    )
    print(f"Data setup complete: {result}")
else:
    print(f"All {len(REQUIRED_TABLES)} data tables already exist — skipping data_setup.")

# COMMAND ----------

# DBTITLE 1,Submit all sweep runs
# Submit all combinations — they queue and run sequentially (max_concurrent_runs=1)
# Increase max_concurrent_runs on the job if you want parallel execution.

submitted_runs = []

for combo in SWEEP_GRID:
    params = {**combo, "experiment_path": EXPERIMENT_PATH}
    run = w.jobs.run_now(job_id=JOB_ID, job_parameters=params)
    submitted_runs.append({"run_id": run.run_id, **combo})
    print(f"✓ Submitted: lr={combo['learning_rate']}, epochs={combo['num_epochs']} → run_id={run.run_id}")

print(f"\n{len(submitted_runs)} runs submitted.")

# COMMAND ----------

# DBTITLE 1,Check run status
# Re-run this cell to refresh status
import time

print(f"{'LR':<10} {'Epochs':<8} {'Run ID':<20} {'State':<15} {'Result':<12}")
print("-" * 70)
for r in submitted_runs:
    run_info = w.jobs.get_run(run_id=r["run_id"])
    state = run_info.state.life_cycle_state.value if run_info.state else "UNKNOWN"
    result = run_info.state.result_state.value if run_info.state and run_info.state.result_state else "-"
    print(f"{r['learning_rate']:<10} {r['num_epochs']:<8} {r['run_id']:<20} {state:<15} {result:<12}")

# COMMAND ----------

# DBTITLE 1,Submit a single run (ad-hoc)
# Use this cell to quickly submit a single combination
# Just change the values and run:

# single_run = w.jobs.run_now(
#     job_id=JOB_ID,
#     job_parameters={"learning_rate": "1e-5", "num_epochs": "4"}
# )
# print(f"Submitted run_id={single_run.run_id}")

# COMMAND ----------

# DBTITLE 1,Wait for all sweep runs to complete
import time

def wait_for_runs(run_list, poll_interval=300):
    """Poll until every run in run_list reaches a terminal state."""
    pending = {r["run_id"] for r in run_list}
    terminal = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
    print(f"Waiting for {len(pending)} runs to finish (polling every {poll_interval}s) ...")
    while pending:
        for rid in list(pending):
            info = w.jobs.get_run(run_id=rid)
            state = info.state.life_cycle_state.value if info.state else "UNKNOWN"
            if state in terminal:
                result = info.state.result_state.value if info.state and info.state.result_state else "N/A"
                print(f"  run_id={rid} finished: {result}")
                pending.discard(rid)
        if pending:
            print(f"  ... {len(pending)} still running")
            time.sleep(poll_interval)
    print("\nAll runs complete.")

# Uncomment below to block until the sweep finishes:
wait_for_runs(submitted_runs)

# COMMAND ----------

# DBTITLE 1,Collect eval results & find best checkpoint
import mlflow
import pandas as pd

# Query MLflow for all eval runs logged by the sweep job's eval task
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
assert experiment, f"Experiment '{EXPERIMENT_PATH}' not found — has the sweep run at least once?"

runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'eval'",
)

if runs_df.empty:
    print("No eval runs found yet. Ensure the sweep job's eval task has completed.")
else:
    # Keep only the LATEST eval per checkpoint_tag. Re-running a cell appends a NEW
    # stage=eval run rather than replacing the old one, so ranking by max-F1 over all
    # of them can crown a stale/superseded run. Sort newest-first, keep first per tag.
    if "params.checkpoint_tag" in runs_df.columns and "start_time" in runs_df.columns:
        runs_df = (runs_df.sort_values("start_time", ascending=False)
                          .drop_duplicates(subset="params.checkpoint_tag", keep="first"))

    cols = ["run_id", "params.checkpoint_tag", "params.learning_rate", "params.num_epochs",
            "metrics.all_f1", "metrics.top8_f1", "metrics.all_precision", "metrics.all_recall"]
    # keep only columns that exist (some may be absent if the eval logs differently)
    cols = [c for c in cols if c in runs_df.columns]
    ranking = runs_df[cols].copy()
    ranking.columns = [c.split(".")[-1] for c in cols]  # strip prefix
    # Guard the sort: fall back gracefully if eval runs didn't surface all_f1.
    if "all_f1" in ranking.columns:
        ranking = ranking.sort_values("all_f1", ascending=False).reset_index(drop=True)

    print("=== Sweep eval ranking (latest eval per tag, by all_f1) ===")
    display(spark.createDataFrame(ranking))

    best = ranking.iloc[0]
    print(f"\n🏆 Best checkpoint: {best.get('checkpoint_tag', 'N/A')}")
    print(f"   lr={best.get('learning_rate', '?')}, epochs={best.get('num_epochs', '?')}")
    print(f"   all_f1={best.get('all_f1', float('nan')):.4f}  top8_f1={best.get('top8_f1', 0):.4f}")
    print(f"   Model path: /Volumes/fins_genai/fine_tuning/checkpoints/agency-ft-final-{best.get('checkpoint_tag', '')}")

# COMMAND ----------

