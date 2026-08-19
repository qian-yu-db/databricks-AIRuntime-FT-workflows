"""Pure, Spark-free helpers shared by the notebooks and the pytest suite.

These are the deterministic bits of the data-prep and eval logic — no spark,
dbutils, vLLM, or MLflow — so they can be imported and unit-tested offline. The
notebooks import from here (via a sys.path bootstrap) so tests exercise the real
code, not a drifting copy.

Two scoring conventions live here on purpose; do NOT cross-compare their F1:
  - `score()`         — cli workflow: present-but-wrong = FN, case-sensitive.
  - `is_match()`      — notebooks workflow: present-but-wrong = FP, lowercased.
"""

import difflib
import json

# Values treated as "not present" on either side.
_NA_VALUES = ("", "NA", "N/A")


# --- Data prep ---------------------------------------------------------------
def strip_inst(text):
    """Remove any leading [INST] / trailing [/INST] Mistral markers. Idempotent."""
    if text is None:
        return text
    import re

    t = str(text).strip()
    t = re.sub(r"^\s*\[INST\]\s*", "", t)
    t = re.sub(r"\s*\[/INST\]\s*$", "", t)
    return t.strip()


# --- cli workflow scoring (present-but-wrong = FN, case-sensitive) -----------
def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


def _is_na(v):
    return v is None or str(v).strip() in _NA_VALUES


def _similar(a, b):
    if a is None or b is None:
        return False
    a, b = str(a).strip(), str(b).strip()
    if a in _NA_VALUES or b in _NA_VALUES:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.6


def score(preds, fields=None):
    """Field-level P/R/F1 over a list of {"ground_truth","pred"} JSON strings.

    Targets are sparse (keys omitted when absent). When `fields` is None we score
    over the UNION of GT and predicted keys, so a hallucinated field (in pred, not
    in GT) counts as FP. Pass an explicit `fields` list for a fixed subset (top-N).
    """
    tp = fp = tn = fn = 0
    for p in preds:
        gt_obj, pred_obj = parse_json(p["ground_truth"]), parse_json(p["pred"])
        keys = fields if fields else (set(gt_obj) | set(pred_obj))
        for k in keys:
            g, pr = gt_obj.get(k), pred_obj.get(k)
            if _is_na(g) and _is_na(pr):
                tn += 1
            elif _is_na(g) and not _is_na(pr):
                fp += 1
            elif not _is_na(g) and _is_na(pr):
                fn += 1
            elif _similar(g, pr):
                tp += 1
            else:
                fn += 1  # present but wrong
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# --- agency-02 / DAB scoring (present-but-wrong = FP, lowercased) ------------
def is_match(gt, pred, threshold=0.6):
    """Classify a single (ground_truth, prediction) pair as TP/FP/FN/TN.

    Inputs are the already-melted string values where absent fields are the
    literal 'NA' (from fillna('NA')). Mismatch scores as FP; comparison lowercased.
    """
    if gt == "NA" and pred == "NA":
        return "TN"
    if gt == "NA":
        return "FP"
    if pred == "NA":
        return "FN"
    if difflib.SequenceMatcher(None, str(gt).lower(), str(pred).lower()).ratio() > threshold:
        return "TP"
    return "FP"
