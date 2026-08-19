# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Job Setup
# MAGIC %md
# MAGIC # Agency Fine-Tuning Job Setup
# MAGIC
# MAGIC This notebook creates the **Agency Fine-Tuning Hyperparameter Sweep** job from scratch
# MAGIC using the Databricks Python SDK.
# MAGIC
# MAGIC **Job structure:**
# MAGIC - `data_setup` → notebook 00 (Serverless CPU, env v5)
# MAGIC - `train` → notebook 01 (Serverless GPU 8×H100, AI v5)
# MAGIC - `eval` → notebook 02 (Serverless GPU 1×H100, AI v5)
# MAGIC
# MAGIC All hyperparameters are exposed as job-level parameters, overridable at run time.

# COMMAND ----------

# DBTITLE 1,Configuration — edit paths and defaults here
# === Configuration ===
# Edit these to match your workspace paths and desired defaults.

JOB_NAME = "Agency Fine-Tuning Hyperparameter Sweep"

# Notebook paths
NOTEBOOK_00 = "/Users/q.yu@databricks.com/AIR_migration_example/dev/notebooks/agency-00-setup-datasets"
NOTEBOOK_01 = "/Users/q.yu@databricks.com/AIR_migration_example/dev/notebooks/agency-01_finetuning-ai-runtime"
NOTEBOOK_02 = "/Users/q.yu@databricks.com/AIR_migration_example/dev/notebooks/agency-02_local-vllm-eval"

# Default parameter values (overridable at run time)
DEFAULTS = {
    "catalog": "fins_genai",
    "schema": "fine_tuning",
    "volume": "training_data",
    "volume_model": "checkpoints",
    "learning_rate": "1e-5",
    "num_epochs": "4",
    "max_seq_length": "4096",
    "per_device_batch_size": "2",
    "gradient_accumulation_steps": "2",
    "num_gpus": "8",
    "gpu_type": "H100",
    "DTYPE": "bfloat16",
    "MAX_MODEL_LEN": "32768",
    "MAX_NEW_TOKENS": "3500",
    "experiment_path": "/Users/q.yu@databricks.com/mlflow_experiments/agency-finetuning-ai-runtime-sweep",
}

# COMMAND ----------

# DBTITLE 1,Create the job
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# --- Build job payload via REST API (SDK 0.67.0 lacks typed classes for serverless GPU) ---
job_parameters = [{"name": k, "default": v} for k, v in DEFAULTS.items()]

envs = [
    {"environment_key": "default", "spec": {"environment_version": "5"}},
    {"environment_key": "gpu_ai_v5", "spec": {"base_environment": "databricks_ai_v5"}},
]

# --- Tasks ---
# NOTE: data_setup is NOT included — run notebook 00 once before the sweep
# (notebook 04 handles this automatically). This saves ~1-2 min per sweep run.
tasks = [
    # Task 1: Train (Serverless GPU 8xH100)
    {
        "task_key": "train",
        "environment_key": "gpu_ai_v5",
        "run_if": "ALL_SUCCESS",
        "compute": {"hardware_accelerator": "GPU_8xH100"},
        "notebook_task": {
            "notebook_path": NOTEBOOK_01,
            "source": "WORKSPACE",
            "base_parameters": {
                "catalog": "{{job.parameters.catalog}}",
                "schema": "{{job.parameters.schema}}",
                "volume": "{{job.parameters.volume}}",
                "volume_model": "{{job.parameters.volume_model}}",
                "learning_rate": "{{job.parameters.learning_rate}}",
                "num_epochs": "{{job.parameters.num_epochs}}",
                "max_seq_length": "{{job.parameters.max_seq_length}}",
                "per_device_batch_size": "{{job.parameters.per_device_batch_size}}",
                "gradient_accumulation_steps": "{{job.parameters.gradient_accumulation_steps}}",
                "num_gpus": "{{job.parameters.num_gpus}}",
                "gpu_type": "{{job.parameters.gpu_type}}",
                "experiment_path": "{{job.parameters.experiment_path}}",
            },
        },
    },
    # Task 2: Eval (Serverless GPU 1xH100)
    {
        "task_key": "eval",
        "depends_on": [{"task_key": "train"}],
        "environment_key": "gpu_ai_v5",
        "run_if": "ALL_SUCCESS",
        "compute": {"hardware_accelerator": "GPU_1xH100"},
        "notebook_task": {
            "notebook_path": NOTEBOOK_02,
            "source": "WORKSPACE",
            "base_parameters": {
                "catalog": "{{job.parameters.catalog}}",
                "schema": "{{job.parameters.schema}}",
                "volume": "{{job.parameters.volume}}",
                "volume_model": "{{job.parameters.volume_model}}",
                "learning_rate": "{{job.parameters.learning_rate}}",
                "num_epochs": "{{job.parameters.num_epochs}}",
                "DTYPE": "{{job.parameters.DTYPE}}",
                "MAX_MODEL_LEN": "{{job.parameters.MAX_MODEL_LEN}}",
                "MAX_NEW_TOKENS": "{{job.parameters.MAX_NEW_TOKENS}}",
                "experiment_path": "{{job.parameters.experiment_path}}",
            },
        },
    },
]

# --- Create or update the job (idempotent) ---
payload = {
    "name": JOB_NAME,
    "tasks": tasks,
    "parameters": job_parameters,
    "environments": envs,
    "max_concurrent_runs": 1,    # Adjust based on GPU resource
    "queue": {"enabled": True},  # Enable queuing execution
}

# Check if a job with this name already exists
_existing = list(w.jobs.list(name=JOB_NAME))

if _existing:
    # Update the existing job in place (reset replaces the full settings)
    job_id = _existing[0].job_id
    reset_payload = {"job_id": job_id, "new_settings": payload}
    w.api_client.do("POST", "/api/2.2/jobs/reset", body=reset_payload)
    print(f"✓ Job updated in place (already existed).")
    if len(_existing) > 1:
        print(f"  ⚠️  Found {len(_existing)} jobs named '{JOB_NAME}' — updated the first (ID {job_id}).")
        print(f"      Consider deleting duplicates: {[j.job_id for j in _existing[1:]]}")
else:
    result = w.api_client.do("POST", "/api/2.2/jobs/create", body=payload)
    job_id = result["job_id"]
    print(f"✓ Job created successfully!")

print(f"  Job ID:   {job_id}")
print(f"  Job name: {JOB_NAME}")
print(f"  URL:      {w.config.host}/#job/{job_id}")

# COMMAND ----------

# DBTITLE 1,Delete the job (cleanup)
# Uncomment to delete the job (e.g., for a clean re-creation)
# w.jobs.delete(job_id=job.job_id)
# print(f"Job {job.job_id} deleted.")

# COMMAND ----------

