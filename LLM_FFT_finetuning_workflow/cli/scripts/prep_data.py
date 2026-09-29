#!/usr/bin/env python3
"""
prep_data.py — local-first data prep: a raw CSV on your laptop -> ChatML JSONL
-> uploaded to the UC Volume the sweep reads. Runs entirely on your laptop
(pandas + the `databricks` CLI); NO Spark, no Databricks Connect, no Delta table.

This replaces the old notebooks/00_prep_data.py (which read Delta tables via Spark).
It collapses agency-00 (raw -> tables) + 00_prep_data (tables -> JSONL) into one
local step:

    raw.csv  ->  clean/strip/synthesize ids  ->  split 85/5/10  ->  ChatML JSONL
             ->  `databricks fs cp` to <grid.yaml:data_dir>/{train,val,test}.jsonl

The upload target (data_dir) comes from configs/grid.yaml and the truncation report
uses sequence_len from configs/axolotl_base.yaml, so this stays in sync with the
rest of the workflow. The pure transforms live in lib/prep.py (unit-tested).

    uv run --with pandas --with pyyaml python scripts/prep_data.py \\
        --input ~/data/raw.csv --profile fevm-classic-stable
    # local only, no upload (e.g. to inspect the JSONL first):
    uv run --with pandas --with pyyaml python scripts/prep_data.py \\
        --input ~/data/raw.csv --no-upload
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

CLI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cli/
CONFIGS = os.path.join(CLI_ROOT, "configs")
sys.path.insert(0, os.path.join(CLI_ROOT, "lib"))

from prep import clean_records, split_records, build_chatml  # noqa: E402


def load_configs():
    """data_dir (upload target) from grid.yaml; sequence_len from axolotl_base.yaml."""
    with open(os.path.join(CONFIGS, "grid.yaml")) as f:
        data_dir = yaml.safe_load(f)["data_dir"].rstrip("/")
    with open(os.path.join(CONFIGS, "axolotl_base.yaml")) as f:
        seq_len = int(yaml.safe_load(f)["sequence_len"])
    return data_dir, seq_len


def write_jsonl(records, path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def truncation_report(splits, seq_len):
    """Report how many examples exceed seq_len (Axolotl DROPS those). Uses the Qwen3
    tokenizer if transformers is installed locally, else a chars/3.5 estimate."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        count = lambda rec: len(tok.apply_chat_template(rec["messages"], tokenize=True))
        method = "Qwen3 tokenizer"
    except Exception:
        count = lambda rec: int(sum(len(m["content"]) for m in rec["messages"]) / 3.5)
        method = "chars/3.5 estimate"
    for name, recs in splits:
        lens = [count(r) for r in recs]
        if lens:
            over = sum(1 for n in lens if n > seq_len)
            print(f"[{method}] {name}: n={len(lens)}  max={max(lens)}  "
                  f">{seq_len}: {over} ({100 * over / len(lens):.1f}%)")


def upload(local_path, data_dir, name, profile):
    dbfs = f"dbfs:{data_dir}/{name}"
    cmd = ["databricks", "fs", "cp", local_path, dbfs, "--overwrite"]
    if profile:
        cmd += ["--profile", profile]
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  ! upload of {name} failed (exit {rc})", file=sys.stderr)
    return rc


def main():
    ap = argparse.ArgumentParser(description="Local CSV -> ChatML JSONL -> UC Volume.")
    ap.add_argument("--input", required=True, help="local raw CSV path")
    ap.add_argument("--profile", default="", help="Databricks CLI profile (for upload)")
    ap.add_argument("--no-upload", action="store_true", help="write JSONL locally only; skip the Volume upload")
    ap.add_argument("--ocr-col", default="ocr_text", help="CSV column holding the OCR text")
    ap.add_argument("--json-col", default="extraction_json", help="CSV column holding the extraction JSON")
    ap.add_argument("--prompt-file", default=os.path.join(CONFIGS, "agency_prompt.txt"),
                    help="instruction prompt prepended to each OCR text")
    ap.add_argument("--out-dir", default=os.path.join(CLI_ROOT, "_prep_out"),
                    help="local dir for the generated JSONL")
    ap.add_argument("--seed", type=int, default=42, help="split shuffle seed")
    args = ap.parse_args()

    if not args.no_upload and not args.profile:
        ap.error("--profile is required to upload (or pass --no-upload to write locally only)")

    import pandas as pd

    data_dir, seq_len = load_configs()
    instruction_prompt = open(args.prompt_file).read()

    # fillna(""): pandas reads empty CSV cells as NaN (a float), which would survive
    # cleaning as the string "nan"; normalize to "" so blank OCR/JSON is dropped.
    raw_rows = pd.read_csv(args.input, dtype=str).fillna("").to_dict("records")
    print(f"Read {len(raw_rows)} rows from {args.input}")

    records = clean_records(raw_rows, ocr_key=args.ocr_col, json_key=args.json_col)
    print(f"Kept {len(records)}/{len(raw_rows)} rows (dropped empty OCR / invalid JSON).")
    assert records, "No valid rows after cleaning — check --ocr-col / --json-col."

    train, val, test = split_records(records, seed=args.seed)
    splits = [
        ("train", build_chatml(train, instruction_prompt), "train.jsonl"),
        ("val", build_chatml(val, instruction_prompt), "val.jsonl"),
        ("test", build_chatml(test, instruction_prompt, include_file_name=True), "test.jsonl"),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, recs, fname in splits:
        path = os.path.join(args.out_dir, fname)
        write_jsonl(recs, path)
        print(f"{fname}: wrote {len(recs)} records -> {path}")

    truncation_report([(n, r) for n, r, _ in splits], seq_len)

    if args.no_upload:
        print(f"\n--no-upload: JSONL is in {args.out_dir}. "
              f"Re-run with --profile to upload to {data_dir}.")
        return 0

    print(f"\nUploading to UC Volume {data_dir} (profile {args.profile}) ...")
    rc = 0
    for name, _, fname in splits:
        rc |= upload(os.path.join(args.out_dir, fname), data_dir, fname, args.profile)
    if rc == 0:
        print(f"Done. {data_dir}/{{train,val,test}}.jsonl are ready for run_sweep.py.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
