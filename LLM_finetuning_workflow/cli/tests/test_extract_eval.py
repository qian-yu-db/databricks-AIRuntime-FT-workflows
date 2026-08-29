"""Unit tests for lib/extract_eval.py — the pure, Spark-free scoring + response
helpers shared by the CLI eval (eval_cli.py), the run_sweep driver, and the
data-prep transforms (lib/prep.py, which imports strip_inst). These are the
deterministic core of the workflow."""
import pytest

from extract_eval import (
    strip_inst,
    clean_response,
    parse_json,
    _is_na,
    _similar,
    score,
    is_match,
)


# --- strip_inst --------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("[INST] hello [/INST]", "hello"),
    ("  [INST]hello[/INST]  ", "hello"),
    ("[INST]only-open", "only-open"),
    ("plain text", "plain text"),
])
def test_strip_inst(raw, expected):
    assert strip_inst(raw) == expected


def test_strip_inst_none_passthrough():
    assert strip_inst(None) is None


def test_strip_inst_idempotent():
    once = strip_inst("[INST] x [/INST]")
    assert strip_inst(once) == once


# --- clean_response ----------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    # full <think>…</think> block preceding the JSON (open tag echoed)
    ('<think>reasoning</think>\n\n{"A": "1"}', '{"A": "1"}'),
    # lone closing </think> (open tag lived in the prompt) + markdown fences
    ('blah reasoning</think>```json\n{"B": "2"}```', '{"B": "2"}'),
    # already-clean JSON is unchanged
    ('{"C": "3"}', '{"C": "3"}'),
    # bare fences, no think block
    ('```json\n{"D": "4"}\n```', '{"D": "4"}'),
])
def test_clean_response(raw, expected):
    assert clean_response(raw) == expected


def test_clean_response_none_is_empty():
    assert clean_response(None) == ""


def test_clean_response_idempotent():
    once = clean_response('<think>r</think>{"A": "1"}')
    assert clean_response(once) == once


def test_clean_response_makes_leaked_json_parseable():
    """The whole point: without cleaning, a think-wrapped completion is unparseable
    and every field scores as a false negative."""
    leaked = '<think>let me extract</think>\n{"PolicyNumber": "P-1"}'
    assert parse_json(leaked) == {}                       # raw: unparseable
    assert parse_json(clean_response(leaked)) == {"PolicyNumber": "P-1"}


# --- parse_json --------------------------------------------------------------
def test_parse_json_valid():
    assert parse_json('{"A": "1"}') == {"A": "1"}


def test_parse_json_invalid_returns_empty():
    assert parse_json("not json") == {}


# --- _is_na ------------------------------------------------------------------
@pytest.mark.parametrize("val", [None, "", "NA", "N/A", "  NA  "])
def test_is_na_true(val):
    assert _is_na(val) is True


@pytest.mark.parametrize("val", ["hello", "0", "P-123"])
def test_is_na_false(val):
    assert _is_na(val) is False


# --- _similar ----------------------------------------------------------------
def test_similar_identical():
    assert _similar("hello world", "hello world") is True


def test_similar_different():
    assert _similar("hello", "xyzab") is False


def test_similar_na_values_never_match():
    assert _similar("NA", "NA") is False
    assert _similar(None, "x") is False


# --- score -------------------------------------------------------------------
def test_score_perfect_match():
    preds = [{"ground_truth": '{"A": "hello"}', "pred": '{"A": "hello"}'}]
    r = score(preds)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["f1"] == 1.0


def test_score_present_but_wrong_is_false_negative():
    # cli-workflow convention: a present-but-wrong value counts as FN, not FP.
    preds = [{"ground_truth": '{"A": "hello"}', "pred": '{"A": "xyzab"}'}]
    r = score(preds)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)
    assert r["f1"] == 0.0


def test_score_hallucinated_field_is_false_positive():
    preds = [{"ground_truth": "{}", "pred": '{"A": "hello"}'}]
    r = score(preds)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 0)


def test_score_both_absent_is_true_negative():
    preds = [{"ground_truth": '{"A": "NA"}', "pred": '{"A": ""}'}]
    r = score(preds)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (0, 0, 0, 1)


def test_score_unparseable_pred_scores_all_gt_as_fn():
    preds = [{"ground_truth": '{"A": "hello", "B": "world"}', "pred": "garbage"}]
    r = score(preds)
    assert r["fn"] == 2 and r["tp"] == 0


def test_score_fields_subset_restricts_scored_keys():
    # Only field A is scored; the wrong B is ignored.
    preds = [{"ground_truth": '{"A": "x", "B": "y"}', "pred": '{"A": "x", "B": "WRONG"}'}]
    r = score(preds, fields=["A"])
    assert (r["tp"], r["fn"], r["f1"]) == (1, 0, 1.0)


# --- is_match (agency-02 / DAB convention: present-but-wrong = FP, lowercased) -
@pytest.mark.parametrize("gt,pred,expected", [
    ("NA", "NA", "TN"),
    ("NA", "x", "FP"),
    ("x", "NA", "FN"),
    ("Hello", "hello", "TP"),      # lowercased comparison
    ("hello", "xyzab", "FP"),      # mismatch -> FP (unlike score()'s FN)
])
def test_is_match(gt, pred, expected):
    assert is_match(gt, pred) == expected
