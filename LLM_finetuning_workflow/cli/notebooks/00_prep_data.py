# Databricks notebook source
# =============================================================================
# 00_prep_data.py  —  Delta tables  ->  Qwen ChatML JSONL in a UC Volume
#
# Run this ON DATABRICKS (notebook or job on a normal cluster; `spark` required).
# The FT tables (built by agency-00) carry `prompt` = agency_prompt.txt + OCR and
# `response` = the sparse extraction JSON. Qwen3 uses its own chat template, so we:
#
#   1. read the existing FT tables,
#   2. strip any residual [INST]/[/INST] wrapper (agency-00 already strips it; no-op),
#   3. re-emit as ChatML messages (system / user / assistant),
#   4. write train.jsonl + val.jsonl (+ test.jsonl for later eval) to the Volume
#      that the Axolotl configs point at.
#
# Targets are SPARSE: each response only contains the fields found in that document.
#
# Source tables (fins_genai.fine_tuning):
#   agency_ft_dataset_train_v3 / _val_v3 : columns  prompt, response
#   agency_ft_dataset_test_v3            : columns  file_name, ground_truths, raw_ocr_content
# =============================================================================

# COMMAND ----------

CATALOG = "fins_genai"
SCHEMA = "fine_tuning"
DATA_DIR = "/Volumes/fins_genai/fine_tuning/training_data/qwen_sweep"  # must match grid.yaml
SEQ_LEN = 8192   # must match sequence_len in axolotl_base.yaml (for the truncation report)

TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_train_v3"
VAL_TABLE   = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_val_v3"
TEST_TABLE  = f"{CATALOG}.{SCHEMA}.agency_ft_dataset_test_v3"

SYSTEM_PROMPT = (
    "You are a helpful assistant working for Acme Title Insurance Corporation. "
    "You specialize in extracting information from title-insurance documents. Given "
    "the document text, extract the requested fields and return them as a JSON object. "
    "The extraction is sparse: include only the fields you find, and omit any field "
    "that is not present (do not emit empty strings or placeholders). Do not explain; "
    "output only the JSON."
)

# Marker that ends the instruction block in agency_prompt.txt; used to split the
# baked-in prompt (instruction template + OCR content) back into its two parts so the
# test user turn can be rebuilt from raw OCR. Must match the last line of the prompt.
OCR_MARKER = "Now parse the following document:"

# COMMAND ----------

import json
import os
import sys

dbutils.fs.mkdirs(DATA_DIR)  # noqa: F821  (dbutils provided by runtime)

# strip_inst lives in cli/lib/extract_eval.py (shared with the eval notebook
# and the pytest suite). Bootstrap sys.path to import it locally and on Databricks.
_here = os.getcwd()
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    pass
for _base in (_here, os.path.dirname(_here), os.getcwd()):
    for _cand in (os.path.join(_base, "lib"), os.path.join(_base, "LLM_finetuning_workflow", "cli", "lib")):
        if os.path.isfile(os.path.join(_cand, "extract_eval.py")) and _cand not in sys.path:
            sys.path.insert(0, _cand)

from extract_eval import strip_inst  # noqa: E402


def to_chatml(user_content: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# COMMAND ----------
# --- TRAIN / VAL : de-wrap the [INST] prompt, response is the target JSON -------

def build_split(table, out_name):
    rows = spark.sql(f"SELECT prompt, response FROM {table}").collect()  # noqa: F821
    records, dropped = [], 0
    for r in rows:
        if not r["prompt"] or not r["response"]:
            dropped += 1
            continue
        user = strip_inst(r["prompt"])
        records.append(to_chatml(user, r["response"].strip()))
    out_path = os.path.join("/dbfs" + DATA_DIR if not DATA_DIR.startswith("/dbfs") else DATA_DIR, out_name)
    # Volumes are directly writable via the /Volumes path on modern runtimes.
    local_path = os.path.join(DATA_DIR, out_name)
    with open(local_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"{out_name}: wrote {len(records)} records ({dropped} dropped) -> {local_path}")
    return records


train_records = build_split(TRAIN_TABLE, "train.jsonl")
val_records = build_split(VAL_TABLE, "val.jsonl")

# COMMAND ----------
# --- Derive the instruction TEMPLATE from a train prompt (marker split) --------
# The test table has no `prompt` column, so we rebuild the user turn as
# <instruction template> + raw_ocr_content, using the SAME template the model
# was trained on (extracted from a training example, not hardcoded).

sample_user = strip_inst(spark.sql(f"SELECT prompt FROM {TRAIN_TABLE} LIMIT 1").collect()[0]["prompt"])  # noqa: F821
if OCR_MARKER in sample_user:
    INSTRUCTION_TEMPLATE = sample_user.split(OCR_MARKER)[0] + OCR_MARKER
else:
    # Fallback: prompt didn't contain the marker; use everything up to the last
    # blank line as the instruction, or just the system prompt.
    INSTRUCTION_TEMPLATE = SYSTEM_PROMPT
    print("WARNING: OCR marker not found in train prompt; test.jsonl uses SYSTEM_PROMPT as template.")

print("Instruction template (first 300 chars):\n", INSTRUCTION_TEMPLATE[:300])

# COMMAND ----------
# --- TEST : reconstruct user turn from raw OCR; assistant = ground truth --------

test_rows = spark.sql(  # noqa: F821
    f"SELECT file_name, ground_truths, raw_ocr_content FROM {TEST_TABLE}"
).collect()

test_records, test_dropped = [], 0
with open(os.path.join(DATA_DIR, "test.jsonl"), "w") as f:
    for r in test_rows:
        if not r["raw_ocr_content"] or not r["ground_truths"]:
            test_dropped += 1
            continue
        user = f"{INSTRUCTION_TEMPLATE}\n{r['raw_ocr_content']}"
        rec = to_chatml(user, r["ground_truths"].strip())
        rec["file_name"] = r["file_name"]  # keep for eval join
        test_records.append(rec)
        f.write(json.dumps(rec) + "\n")
print(f"test.jsonl: wrote {len(test_records)} records ({test_dropped} dropped) -> {DATA_DIR}/test.jsonl")

# COMMAND ----------
# --- Truncation report at SEQ_LEN ---------------------------------------------
# Uses the Qwen3 tokenizer if transformers is available; otherwise a
# chars/3.5 heuristic. Tells you how many training examples exceed sequence_len
# (they get truncated), so you can decide whether to raise SEQ_LEN.

def token_counter():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return lambda rec: len(tok.apply_chat_template(rec["messages"], tokenize=True)), "Qwen3 tokenizer"
    except Exception as e:  # transformers/model not available on this cluster
        print(f"(tokenizer unavailable: {e}; using chars/3.5 estimate)")
        return lambda rec: int(sum(len(m["content"]) for m in rec["messages"]) / 3.5), "chars/3.5 estimate"


count_tokens, method = token_counter()
for name, recs in [("train", train_records), ("val", val_records), ("test", test_records)]:
    lens = [count_tokens(r) for r in recs]
    over = sum(1 for n in lens if n > SEQ_LEN)
    if lens:
        print(
            f"[{method}] {name}: n={len(lens)}  max={max(lens)}  "
            f"p95={sorted(lens)[int(len(lens) * 0.95)]}  "
            f">{SEQ_LEN}: {over} ({100 * over / len(lens):.1f}%)"
        )

print("\nDone. train.jsonl / val.jsonl / test.jsonl are in", DATA_DIR)
