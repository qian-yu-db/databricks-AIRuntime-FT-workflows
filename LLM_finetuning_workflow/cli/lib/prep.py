"""Pure, Spark-free data-prep transforms for the local-first prep_data.py CLI.

Reads (CSV, prompt file) and the UC upload happen at the edges in
scripts/prep_data.py; the functions here transform in-memory records only, so they
can be unit-tested offline. This is the local replacement for the old
agency-00 + 00_prep_data notebook chain: raw rows -> ChatML records, no Delta, no
Spark.

ChatML shape (matches what Axolotl trains on and eval_cli.py scores):
    {"messages": [system, user, assistant]}  (+ "file_name" on test records)
where user = <instruction prompt> + <OCR text>, assistant = the sparse extraction
JSON, kept verbatim.
"""
import json

from extract_eval import strip_inst  # reuse the shared [INST] stripper

# Carried from the old 00_prep_data notebook — the system turn for every record.
SYSTEM_PROMPT = (
    "You are a helpful assistant working for Acme Title Insurance Corporation. "
    "You specialize in extracting information from title-insurance documents. Given "
    "the document text, extract the requested fields and return them as a JSON object. "
    "The extraction is sparse: include only the fields you find, and omit any field "
    "that is not present (do not emit empty strings or placeholders). Do not explain; "
    "output only the JSON."
)


def _is_valid_json(s) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def clean_records(raw_rows, ocr_key="ocr_text", json_key="extraction_json"):
    """raw_rows: list of dicts (e.g. CSV rows). Returns cleaned records
    [{file_name, raw_ocr, response}], dropping rows with empty OCR or invalid JSON.

    Strips [INST]/[/INST] from the OCR and keeps the extraction JSON verbatim (it is
    already sparse, valid JSON). file_name is synthesized over the KEPT rows in order
    (doc_00001, …) so eval can join predictions to ground truth — matches agency-00.
    """
    out = []
    for row in raw_rows:
        ocr = strip_inst(row.get(ocr_key))
        resp = str(row.get(json_key, "")).strip()
        if not ocr or not _is_valid_json(resp):
            continue
        out.append({
            "file_name": f"doc_{len(out) + 1:05d}",
            "raw_ocr": ocr,
            "response": resp,
        })
    return out


def split_records(records, ratios=(0.85, 0.05, 0.10), seed=42):
    """Deterministic shuffle + split into (train, val, test) by ratios.

    A local, reproducible stand-in for agency-00's Spark randomSplit — same 85/5/10
    intent, but exact membership differs (different RNG), which is fine for a fresh
    local prep. `seed` makes it repeatable run-to-run.
    """
    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def to_chatml(user_content, assistant_content, file_name=None) -> dict:
    """Wrap a (user, assistant) pair as a ChatML record with the shared system turn.
    Includes a top-level `file_name` when given (test records carry it for eval)."""
    rec = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    if file_name is not None:
        rec["file_name"] = file_name
    return rec


def build_chatml(records, instruction_prompt, include_file_name=False):
    """Turn cleaned records into ChatML dicts. The user turn is
    `instruction_prompt + raw_ocr` (concatenated, no separator — matching the format
    the model was trained on). Set include_file_name=True for the test split."""
    out = []
    for r in records:
        user = f"{instruction_prompt}{r['raw_ocr']}"
        out.append(to_chatml(user, r["response"],
                             r["file_name"] if include_file_name else None))
    return out
