# Databricks notebook source
# MAGIC %md
# MAGIC # Agency dataset setup — from paired (OCR, extraction JSON) Excel
# MAGIC
# MAGIC Builds the fine-tuning tables from `sample_input_output.xlsx`, which pairs each
# MAGIC document's OCR text with its ground-truth extraction JSON in a single row
# MAGIC (columns `ocr_text`, `extraction_json`). This replaces the old flow that joined a
# MAGIC ground-truth CSV to a separate folder of OCR `.txt` files on `File Name`.
# MAGIC
# MAGIC Key differences from the old (v2) schema:
# MAGIC - The extraction schema is the new **CamelCase** field set (`PolicyNumber`,
# MAGIC   `InsuredName0First`, `SitusAddress`, `Scope`, …), and it is **sparse** — a row's
# MAGIC   JSON only contains the fields that were found. We keep it sparse (no `N/A` fill).
# MAGIC - There is **no `File Name`** in the source; we synthesize one (`doc_00001`, …) so
# MAGIC   eval can join predictions to ground truth.
# MAGIC - `ocr_text` carries a trailing Mistral `[/INST]` marker, which we strip.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the paired Excel from the UC Volume

# COMMAND ----------

# MAGIC %pip install openpyxl
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "fine_tuning")
dbutils.widgets.text("volume", "training_data")

# COMMAND ----------

# DBTITLE 1,Cell 4
import pandas as pd

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

# Upload sample_input_output.xlsx to this Volume before running (see repo CLAUDE.md).
XLSX_PATH = f"/Volumes/{catalog}/{schema}/{volume}/sample_input_output.xlsx"

raw = pd.read_excel(XLSX_PATH, sheet_name="result", engine='openpyxl')
raw.info()

# COMMAND ----------

print("# Samples:", len(raw))
print("Columns:", list(raw.columns))

# COMMAND ----------

import json
import re


def strip_inst(text: str) -> str:
    """Remove any leading [INST] / trailing [/INST] Mistral markers from OCR text."""
    if text is None:
        return text
    t = str(text).strip()
    t = re.sub(r"^\s*\[INST\]\s*", "", t)
    t = re.sub(r"\s*\[/INST\]\s*$", "", t)
    return t.strip()


def is_valid_json(s) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


# Clean OCR, keep the extraction JSON verbatim (it is already valid, sparse JSON),
# and synthesize a stable File_Name from the row order.
clean = raw.copy()
clean["Raw_OCR_Content"] = clean["ocr_text"].map(strip_inst)
clean["Ground_Truths"] = clean["extraction_json"].astype(str).str.strip()
clean = clean.reset_index(drop=True)
clean["File_Name"] = [f"doc_{i + 1:05d}" for i in range(len(clean))]

# Drop rows missing either side or with unparseable JSON.
before = len(clean)
clean = clean[clean["Raw_OCR_Content"].astype(bool) & clean["Ground_Truths"].map(is_valid_json)]
print(f"Kept {len(clean)}/{before} rows (dropped {before - len(clean)} empty/invalid).")

master_pdf = clean[["File_Name", "Ground_Truths", "Raw_OCR_Content"]]
master_pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Master Dataset table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

master_schema = StructType([
    StructField("File_Name", StringType()),
    StructField("Ground_Truths", StringType()),
    StructField("Raw_OCR_Content", StringType()),
])

dataset = spark.createDataFrame(master_pdf, schema=master_schema)

display(dataset.count())
display(dataset.limit(5))

# COMMAND ----------

# DBTITLE 1,Cell 9
# MAGIC %sql
# MAGIC
# MAGIC drop table if exists IDENTIFIER(:catalog || '.' || :schema || '.agency_master_dataset_v3')

# COMMAND ----------

# DBTITLE 1,Cell 10
(
    dataset.write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog}.{schema}.agency_master_dataset_v3")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finetuning Datasets

# COMMAND ----------

from pyspark.sql.functions import lit, col, concat

# COMMAND ----------

# DBTITLE 1,Cell 13
# get prompt — read from the Volume (same one as the source data) so every pipeline
# uses one consistent, notebook-location-independent path. Upload agency_prompt.txt
# to this Volume before running (see repo CLAUDE.md), alongside sample_input_output.xlsx.
PROMPT_PATH = f"/Volumes/{catalog}/{schema}/{volume}/agency_prompt.txt"
add_prompt = open(PROMPT_PATH).read()

print(add_prompt)

# COMMAND ----------

# DBTITLE 1,Cell 14
# split data
mdf = spark.sql(
    f"""
    SELECT File_Name, Ground_Truths, Raw_OCR_Content
    FROM {catalog}.{schema}.agency_master_dataset_v3
    ;
    """
).dropna()

train_df, val_df, test_df = mdf.randomSplit([0.85, 0.05, 0.10], seed=42)

print(train_df.count(), val_df.count(), test_df.count())

display(train_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train

# COMMAND ----------

# DBTITLE 1,Cell 16
# MAGIC %sql
# MAGIC
# MAGIC drop table if exists IDENTIFIER(:catalog || '.' || :schema || '.agency_ft_dataset_train_v3')

# COMMAND ----------

# DBTITLE 1,Cell 17
train_df_cleaned = (
    train_df
    .withColumn("prompt", concat(lit(add_prompt), col("Raw_OCR_Content")))
    .withColumnRenamed("Ground_Truths", "response")
    .select("prompt", "response")
)

display(train_df_cleaned.limit(5))

(
    train_df_cleaned.write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog}.{schema}.agency_ft_dataset_train_v3")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Val

# COMMAND ----------

# DBTITLE 1,Cell 19
# MAGIC %sql
# MAGIC
# MAGIC drop table if exists IDENTIFIER(:catalog || '.' || :schema || '.agency_ft_dataset_val_v3')

# COMMAND ----------

# DBTITLE 1,Cell 20
val_df_cleaned = (
    val_df
    .withColumn("prompt", concat(lit(add_prompt), col("Raw_OCR_Content")))
    .withColumnRenamed("Ground_Truths", "response")
    .select("prompt", "response")
)

display(val_df_cleaned.limit(5))

(
    val_df_cleaned.write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog}.{schema}.agency_ft_dataset_val_v3")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test
# MAGIC
# MAGIC The test table keeps `file_name` / `ground_truths` / `raw_ocr_content` (lowercased
# MAGIC to match what the eval notebooks expect) so eval can rebuild the prompt and join
# MAGIC predictions to ground truth on `file_name`.

# COMMAND ----------

# DBTITLE 1,Cell 22
# MAGIC %sql
# MAGIC
# MAGIC drop table if exists IDENTIFIER(:catalog || '.' || :schema || '.agency_ft_dataset_test_v3')

# COMMAND ----------

# DBTITLE 1,Cell 23
test_df_out = test_df.selectExpr(
    "File_Name as file_name",
    "Ground_Truths as ground_truths",
    "Raw_OCR_Content as raw_ocr_content",
)

display(test_df_out.limit(5))

(
    test_df_out.write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog}.{schema}.agency_ft_dataset_test_v3")
)