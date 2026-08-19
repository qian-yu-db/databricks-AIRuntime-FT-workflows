# Databricks notebook source
# MAGIC %md
# MAGIC # Custom-LLM serving endpoint load test (Locust)
# MAGIC
# MAGIC Client-side load test for a Databricks Model Serving endpoint serving a custom LLM
# MAGIC (e.g. Qwen3-8B). Adapted from the Databricks reference notebook
# MAGIC (https://docs.databricks.com/aws/en/notebooks/source/locust-load-test.html) and the
# MAGIC accompanying setup guide
# MAGIC (https://docs.databricks.com/aws/en/machine-learning/model-serving/configure-load-test).
# MAGIC
# MAGIC It sweeps client concurrency against the endpoint, then charts latency percentiles,
# MAGIC RPS, failures, and exceptions. For the **server-side** view (vLLM queue time, GPU cache
# MAGIC usage, TTFT histograms) run the `serve-custom-llms-metrics` notebook over the SAME time
# MAGIC window against the endpoint's `<prefix>_otel_metrics` Unity Catalog table.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Single-node cluster, `15.4 LTS ML` runtime, CPU-optimized, **≥ 32 cores** (the client
# MAGIC   must out-scale the endpoint).
# MAGIC - A **service principal** with **Can Query** on the endpoint, and a **secret scope** with
# MAGIC   `service_principal_client_id` / `service_principal_client_secret`.
# MAGIC - Endpoint in **Ready** state, **scale-to-zero disabled**, route optimization enabled.
# MAGIC - `fast_load_test.py` and `input.json` present next to this notebook (edit `input.json`
# MAGIC   to a payload representative of your real requests — pin `max_tokens`).

# COMMAND ----------

# MAGIC %pip install gevent==24.11.1 locust==2.32.6 databricks-sdk==0.50.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pathlib
import shlex
import subprocess
from math import ceil

import pandas as pd
import matplotlib.pyplot as plt
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variables
# MAGIC
# MAGIC - `endpoint_name` — the Model Serving endpoint name (e.g. your Qwen3-8B endpoint).
# MAGIC - `locust_run_time` — per-concurrency-step duration ("s"/"m" suffix).
# MAGIC - `csv_output_prefix` — prefix for the generated result CSVs.
# MAGIC - `secret_scope` — secret scope holding the service principal creds.

# COMMAND ----------

# Endpoint + run configuration
endpoint_name = "<your-endpoint-name>"
locust_run_time = "5m"
csv_output_prefix = "qwen_load_test"
secret_scope = "oauth"

# Service principal creds from the secret scope
CLIENT_ID = dbutils.secrets.get(scope=secret_scope, key="service_principal_client_id")
CLIENT_SECRET = dbutils.secrets.get(scope=secret_scope, key="service_principal_client_secret")

# COMMAND ----------

workspace_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"

c = WorkspaceClient()
endpoint_details = c.serving_endpoints.get(name=endpoint_name)
# Endpoint host prefix (respects route optimization).
endpoint_prefix = f"https://{endpoint_details.endpoint_url.split('serving-endpoints')[0]}"
endpoint_id = endpoint_prefix.split(".")[0].split("//")[1]

print(f"workspace url:   {workspace_url}")
print(f"endpoint prefix: {endpoint_prefix}")
print(f"endpoint id:     {endpoint_id}")

# COMMAND ----------

# Env vars consumed by fast_load_test.py
os.environ["DATABRICKS_WORKSPACE_URL"] = workspace_url
os.environ["ENDPOINT_ID"] = endpoint_id
os.environ["CLIENT_ID"] = CLIENT_ID
os.environ["CLIENT_SECRET"] = CLIENT_SECRET
os.environ["DATABRICKS_ENDPOINT_NAME"] = endpoint_name
# Prefer the sampled pool (one real document per request) for a realistic output-length mix;
# fall back to the single fixed input.json if the JSONL isn't present.
_payloads = pathlib.Path("../payloads.jsonl")
if _payloads.exists():
    os.environ["PAYLOADS_JSONL"] = str(_payloads.resolve())
else:
    os.environ["INPUT_JSON"] = str(pathlib.Path("../input.json").resolve())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def check_for_CPU_overload(line: str) -> None:
    """Raise if Locust reports client-side CPU overload (invalid results)."""
    if "CPU" in line:
        raise Exception(
            "CPU Overloaded — rerun on a single-node cluster with more cores."
        )


def run_locust_test(
    host: str,
    users: int,
    spawn_rate: int,
    run_time: str,
    csv_output_prefix: str,
    locust_file: str,
    verbose: bool = False,
) -> None:
    """Spawn a headless Locust process and stream its output."""
    locust_command = (
        f"locust --host={host} --users={users} --spawn-rate={spawn_rate} "
        f"--run-time={run_time} --headless --locustfile={locust_file} "
        f"--csv={csv_output_prefix} --processes -1"
    )
    process = subprocess.Popen(
        shlex.split(locust_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            check_for_CPU_overload(line)
            if verbose:
                print(line.strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Warmup
# MAGIC
# MAGIC Short 30s test to confirm the endpoint is Ready and the payload/auth work. You should
# MAGIC see **no failures or exceptions** here before proceeding.

# COMMAND ----------

file_path = str(pathlib.Path("../fast_load_test.py").resolve())
file_path = f'"{file_path}"'  # quote in case the path has spaces
run_locust_test(
    endpoint_prefix, 1, 1, "30s", f"{csv_output_prefix}_warmup",
    locust_file=file_path, verbose=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Concurrency sweep
# MAGIC
# MAGIC Step client concurrency up and load-test at each level. Keep the endpoint's min/max
# MAGIC concurrency fixed (e.g. 4) so the sweep isolates client pressure vs. a known capacity.

# COMMAND ----------

client_connections_test_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for client_connections in client_connections_test_values:
    print(f"============== {client_connections} client connections ==============")
    run_locust_test(
        endpoint_prefix,
        client_connections,
        4,
        locust_run_time,
        f"{csv_output_prefix}_{client_connections}",
        locust_file=file_path,
        verbose=True,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Failures observed per concurrency level

# COMMAND ----------

all_failures_csv = [
    pd.read_csv(f"{csv_output_prefix}_{i}_failures.csv")
    for i in client_connections_test_values
]
failures = [
    csv["Occurrences"].iloc[0] if not csv.empty else 0 for csv in all_failures_csv
]
failure_df = pd.DataFrame(
    {"Client Connections": client_connections_test_values, "Failures": failures}
)
display(failure_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exceptions observed per concurrency level

# COMMAND ----------

all_exceptions_csv = [
    pd.read_csv(f"{csv_output_prefix}_{i}_exceptions.csv")
    for i in client_connections_test_values
]
exceptions = [
    csv["Count"].iloc[0] if not csv.empty else 0 for csv in all_exceptions_csv
]
exceptions_df = pd.DataFrame(
    {"Client Connections": client_connections_test_values, "Exceptions": exceptions}
)
display(exceptions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined latency table
# MAGIC
# MAGIC RPS + latency percentiles per concurrency level (averaged over the run). This is
# MAGIC whole-request latency; for the per-token view (TTFT, time-per-output-token) read the
# MAGIC server-side vLLM histograms in `serve_custom_llms_metrics.py` over the same window.

# COMMAND ----------

combined_latency_df_list = []
percentiles_to_keep = ["50%", "80%", "90%", "95%", "99%", "99.9%"]
for i in client_connections_test_values:
    current_latency_df = pd.read_csv(f"{csv_output_prefix}_{i}_stats_history.csv")
    current_latency_df = current_latency_df[current_latency_df["User Count"] == i]
    cols = ["Requests/s", "Total Average Response Time", "Total Min Response Time"] + percentiles_to_keep
    current_latency_df = current_latency_df[["User Count"] + cols]
    current_latency_df = (
        current_latency_df.groupby("User Count")[cols].mean().reset_index()
    )
    combined_latency_df_list.append(current_latency_df)

combined_latency_results_df = pd.concat(combined_latency_df_list, ignore_index=True)
display(combined_latency_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latency vs. concurrency

# COMMAND ----------

plt.figure(figsize=(10, 6))
for percentile in percentiles_to_keep:
    plt.plot(
        combined_latency_results_df["User Count"],
        combined_latency_results_df[percentile],
        label=percentile,
    )
plt.xlabel("Concurrent Users")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Concurrent Users")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick an operating point + size the endpoint
# MAGIC
# MAGIC Choose the concurrency whose latency percentiles are acceptable (cross-check its
# MAGIC failure/exception rate above), and set your target RPS. The cells compute the endpoint
# MAGIC concurrency and client connections needed to reach that RPS.

# COMMAND ----------

latency_table = combined_latency_results_df[["User Count"] + percentiles_to_keep]
display(latency_table)

# COMMAND ----------

USER_COUNT_SELECTION = 4
REQUESTS_PER_SECOND = 2000  # target RPS

# COMMAND ----------

rps = combined_latency_results_df[
    combined_latency_results_df["User Count"] == USER_COUNT_SELECTION
]["Requests/s"].iloc[0]
rps_multiple = ceil(REQUESTS_PER_SECOND / rps)

endpoint_concurrency_needed = 4 * rps_multiple
client_connections_needed = USER_COUNT_SELECTION * rps_multiple

print(
    f"Resize the endpoint to concurrency {endpoint_concurrency_needed}, "
    f"then run the load test with {client_connections_needed} locust users."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final validation run
# MAGIC
# MAGIC After resizing the endpoint, warm it up and run the full-scale test. Then open
# MAGIC `serve-custom-llms-metrics` over this run's time window to confirm the endpoint's
# MAGIC server-side metrics (queue time, GPU cache, TTFT) match the client-side picture.

# COMMAND ----------

run_locust_test(
    endpoint_prefix, 1, 1, "30s", f"{csv_output_prefix}_warmup",
    locust_file=file_path, verbose=True,
)

# COMMAND ----------

run_locust_test(
    endpoint_prefix,
    client_connections_needed,
    4,
    "10m",
    f"{csv_output_prefix}_final",
    locust_file=file_path,
    verbose=True,
)
