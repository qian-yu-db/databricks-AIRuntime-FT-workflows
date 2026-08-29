"""Unit tests for lib/prep.py — the pure local-first data-prep transforms
(raw rows -> cleaned records -> 85/5/10 split -> ChatML). No CSV/Volume I/O here;
that lives in scripts/prep_data.py at the edges."""
import json

import pytest

from prep import (
    SYSTEM_PROMPT,
    clean_records,
    split_records,
    to_chatml,
    build_chatml,
)


# --- clean_records -----------------------------------------------------------
def test_clean_records_strips_inst_and_synthesizes_ids():
    raw = [
        {"ocr_text": "[INST] doc one text [/INST]", "extraction_json": '{"A": "1"}'},
        {"ocr_text": "doc two text", "extraction_json": '{"B": "2"}'},
    ]
    recs = clean_records(raw)
    assert [r["file_name"] for r in recs] == ["doc_00001", "doc_00002"]
    assert recs[0]["raw_ocr"] == "doc one text"          # [INST] wrapper stripped
    assert recs[0]["response"] == '{"A": "1"}'           # JSON kept verbatim


def test_clean_records_drops_empty_ocr_and_invalid_json():
    raw = [
        {"ocr_text": "", "extraction_json": '{"A": "1"}'},          # empty OCR -> drop
        {"ocr_text": "text", "extraction_json": "not json"},        # invalid JSON -> drop
        {"ocr_text": "good", "extraction_json": '{"C": "3"}'},      # kept
    ]
    recs = clean_records(raw)
    assert len(recs) == 1
    assert recs[0]["file_name"] == "doc_00001"           # id numbers KEPT rows in order
    assert recs[0]["raw_ocr"] == "good"


def test_clean_records_custom_column_names():
    raw = [{"text": "hi", "labels": '{"A": "1"}'}]
    recs = clean_records(raw, ocr_key="text", json_key="labels")
    assert len(recs) == 1 and recs[0]["raw_ocr"] == "hi"


# --- split_records -----------------------------------------------------------
def test_split_records_ratios_and_totals():
    recs = [{"file_name": f"doc_{i:05d}", "raw_ocr": "x", "response": "{}"} for i in range(100)]
    train, val, test = split_records(recs)
    assert (len(train), len(val), len(test)) == (85, 5, 10)
    # no overlap, nothing lost
    names = [r["file_name"] for r in train + val + test]
    assert len(set(names)) == 100


def test_split_records_deterministic_with_seed():
    recs = [{"file_name": f"doc_{i:05d}", "raw_ocr": "x", "response": "{}"} for i in range(50)]
    a = split_records(recs, seed=42)
    b = split_records(recs, seed=42)
    assert [r["file_name"] for r in a[0]] == [r["file_name"] for r in b[0]]
    # a different seed generally reorders the train split
    c = split_records(recs, seed=7)
    assert [r["file_name"] for r in a[0]] != [r["file_name"] for r in c[0]]


# --- to_chatml / build_chatml ------------------------------------------------
def test_to_chatml_roles_and_optional_file_name():
    rec = to_chatml("USER", "ASSIST")
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == SYSTEM_PROMPT
    assert "file_name" not in rec                         # omitted when not given

    rec2 = to_chatml("U", "A", file_name="doc_00001")
    assert rec2["file_name"] == "doc_00001"


def test_build_chatml_concatenates_prompt_and_ocr():
    recs = [{"file_name": "doc_00001", "raw_ocr": "OCRTEXT", "response": '{"A": "1"}'}]
    out = build_chatml(recs, instruction_prompt="PROMPT:")
    assert out[0]["messages"][1]["content"] == "PROMPT:OCRTEXT"   # no separator
    assert out[0]["messages"][2]["content"] == '{"A": "1"}'
    assert "file_name" not in out[0]                             # train/val: no file_name


def test_build_chatml_test_split_carries_file_name():
    recs = [{"file_name": "doc_00007", "raw_ocr": "x", "response": "{}"}]
    out = build_chatml(recs, "P", include_file_name=True)
    assert out[0]["file_name"] == "doc_00007"


def test_build_chatml_output_is_json_serializable():
    recs = clean_records([{"ocr_text": "t", "extraction_json": '{"A": "1"}'}])
    line = json.dumps(build_chatml(recs, "P")[0])
    assert '"role": "system"' in line
