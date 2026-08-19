# Databricks notebook source
# MAGIC %md
# MAGIC # Custom LLM serving metrics notebook
# MAGIC
# MAGIC Query and visualize the Prometheus metrics that Databricks auto-scrapes from vLLM and persists to your endpoint's `<prefix>_otel_metrics` Unity Catalog table.
# MAGIC
# MAGIC Set the widgets in the next cell to point at your endpoint's telemetry table and endpoint name, then run the cells below to chart gauge, counter, and histogram metrics across all replicas of your endpoint.

# COMMAND ----------

dbutils.widgets.removeAll()

# 0. UC Table
dbutils.widgets.text("uc_table", "<catalog>.<schema>.<endpoint-prefix>_otel_metrics", "UC Table")

# 0a. Filter to a single endpoint. The telemetry table holds metrics from every endpoint with telemetry enabled.
dbutils.widgets.text("endpoint_name", "<your-endpoint-name>", "Endpoint Name")

# 1. Time Range Mode
dbutils.widgets.dropdown("time_range_mode", "relative", ["relative", "absolute"], "Time Range Mode")

# 2. Start Time (absolute)
dbutils.widgets.text("start_time", "2026-05-01 00:00:00", "Start Time (absolute)")

# 3. End Time (absolute)
dbutils.widgets.text("end_time", "2026-05-01 23:59:59", "End Time (absolute)")

# 4. Lookback (hours)
dbutils.widgets.dropdown("lookback_hours", "6", ["1", "3", "6", "12", "24", "48", "72", "168"], "Lookback (hours)")

# 5. Metric Names — dynamically fetched from the table
uc_table = dbutils.widgets.get("uc_table")
metric_choices_df = spark.sql(f"""
    SELECT DISTINCT name
    FROM {uc_table}
    WHERE date >= current_date() - INTERVAL 7 DAYS
    ORDER BY name
""")
metric_choices = [row.name for row in metric_choices_df.collect()]
dbutils.widgets.multiselect("metric_names", metric_choices[0], metric_choices, "Metric Names")

# 6. Aggregation Method
dbutils.widgets.dropdown("agg_method", "sum", ["sum", "avg"], "Aggregation Method")



# COMMAND ----------

from pyspark.sql import functions as F

# Read widget values
try:
    selected_names = dbutils.widgets.get("metric_names").split(",")
except:
    selected_names = []

selected_names = [n.strip() for n in selected_names if n.strip()]
try:
    agg_method = dbutils.widgets.get("agg_method")
except:
    agg_method = "sum"
time_range_mode = dbutils.widgets.get("time_range_mode")  # "relative" or "absolute"
uc_table = dbutils.widgets.get("uc_table")
endpoint_name = dbutils.widgets.get("endpoint_name")

# Build time filter clause
if time_range_mode == "absolute":
    start_time = dbutils.widgets.get("start_time")
    end_time = dbutils.widgets.get("end_time")
    # Estimate date partition range from absolute timestamps
    date_filter = f"date >= DATE('{start_time}') AND date <= DATE('{end_time}')"
    time_filter = f"time >= TIMESTAMP('{start_time}') AND time <= TIMESTAMP('{end_time}')"
    time_label = f"{start_time} → {end_time}"
else:
    lookback_hours = int(dbutils.widgets.get("lookback_hours"))
    date_filter = f"date >= current_date() - INTERVAL {max(lookback_hours // 24 + 1, 1)} DAYS"
    time_filter = f"time >= current_timestamp() - INTERVAL {lookback_hours} HOURS"
    time_label = f"last {lookback_hours}h"

metrics_df = spark.sql(f"""
WITH metrics_base AS (
  SELECT
    time,
    date,
    service_name,
    name,
    description,
    unit,
    metric_type,
    variant_get(resource.attributes, '$["k8s.pod.uid"]', 'STRING') AS pod_uid,
    resource.attributes::string AS resource_attributes,
    CASE
      WHEN metric_type = 'gauge' THEN gauge.attributes::string
      WHEN metric_type = 'sum' THEN sum.attributes::string
      WHEN metric_type = 'histogram' THEN histogram.attributes::string
    END AS metric_attributes,
    gauge.value AS gauge_value,
    sum.value AS sum_value,
    histogram.count AS histogram_count,
    histogram.sum AS histogram_sum,
    histogram.bucket_counts,
    histogram.explicit_bounds
  FROM {uc_table}
  WHERE {date_filter}
    AND {time_filter}
    AND variant_get(resource.attributes, '$["endpoint.name"]', 'STRING') = '{endpoint_name}'
),
metrics_normalized AS (
  SELECT
    time,
    date,
    service_name,
    name,
    description,
    unit,
    metric_type,
    pod_uid,
    resource_attributes,
    metric_attributes,
    CASE
      WHEN metric_type = 'gauge' THEN gauge_value
      WHEN metric_type = 'sum' THEN sum_value
      WHEN metric_type = 'histogram' THEN histogram_count
    END AS metric_value,
    histogram_count,
    histogram_sum,
    bucket_counts,
    explicit_bounds
  FROM metrics_base
)
SELECT *
FROM metrics_normalized
""")

# Apply metric name filter
metrics_df = metrics_df.filter(F.col("name").isin(selected_names))

print(f"Filters: metrics={selected_names}, time={time_label} ({time_range_mode}), agg={agg_method}")
print(f"Row count: {metrics_df.count()}")
display(metrics_df)

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# --- Gauge metrics: aggregate across pods at each timestamp ---
gauge_agg_fn = F.avg("metric_value") if agg_method == "avg" else F.sum("metric_value")
gauge_pdf = (
    metrics_df
    .filter(F.col("metric_type") == "gauge")
    .groupBy("time", "name", "metric_attributes")
    .agg(
        gauge_agg_fn.alias("agg_value"),
        F.countDistinct("pod_uid").alias("pod_count"),
    )
    .orderBy("time")
    .toPandas()
)

if gauge_pdf.empty:
    print("No gauge metrics in the selected set – skipping chart.")
else:
    gauge_pdf["label"] = gauge_pdf["name"] + " | " + gauge_pdf["metric_attributes"].fillna("")
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, grp in gauge_pdf.groupby("label"):
        ax.plot(grp["time"], grp["agg_value"], label=label, linewidth=1)
    ax.set_title(f"Gauge Metrics – {agg_method} across pods")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# --- Sum / counter metrics: aggregate across pods at each timestamp ---
sum_agg_fn = F.avg("metric_value") if agg_method == "avg" else F.sum("metric_value")
sum_pdf = (
    metrics_df
    .filter(F.col("metric_type") == "sum")
    .groupBy("time", "name", "metric_attributes")
    .agg(
        sum_agg_fn.alias("agg_value"),
        F.countDistinct("pod_uid").alias("pod_count"),
    )
    .orderBy("time")
    .toPandas()
)

if sum_pdf.empty:
    print("No counter (sum) metrics in the selected set – skipping chart.")
else:
    sum_pdf["label"] = sum_pdf["name"] + " | " + sum_pdf["metric_attributes"].fillna("")
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, grp in sum_pdf.groupby("label"):
        ax.plot(grp["time"], grp["agg_value"], label=label, linewidth=1)
    ax.set_title(f"Counter (sum) Metrics – {agg_method} across pods")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# --- Histogram metrics: aggregate across pods ---
if agg_method == "avg":
    hist_count_fn = F.avg("histogram_count")
    hist_sum_fn = F.avg("histogram_sum")
else:
    hist_count_fn = F.sum("histogram_count")
    hist_sum_fn = F.sum("histogram_sum")

hist_pdf = (
    metrics_df
    .filter(F.col("metric_type") == "histogram")
    .groupBy("time", "name", "metric_attributes")
    .agg(
        hist_count_fn.alias("total_count"),
        hist_sum_fn.alias("total_sum"),
        F.countDistinct("pod_uid").alias("pod_count"),
    )
    .withColumn(
        "avg_latency",
        F.when(F.col("total_count") > 0, F.col("total_sum") / F.col("total_count")),
    )
    .orderBy("time")
    .toPandas()
)

if hist_pdf.empty:
    print("No histogram metrics in the selected set – skipping chart.")
else:
    hist_pdf["label"] = hist_pdf["name"] + " | " + hist_pdf["metric_attributes"].fillna("")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for label, grp in hist_pdf.groupby("label"):
        axes[0].plot(grp["time"], grp["total_count"], label=label, linewidth=1)
    axes[0].set_title(f"Histogram – {agg_method} count across pods")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    axes[0].grid(True, alpha=0.3)

    for label, grp in hist_pdf.groupby("label"):
        axes[1].plot(grp["time"], grp["avg_latency"], label=label, linewidth=1)
    axes[1].set_title(f"Histogram – avg value (sum/count, {agg_method} across pods)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Avg value")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
