"""Unit tests for scripts/eval_cli.py — the vLLM eval that runs on the GPU worker.

The vLLM serving path (subprocess) and MLflow logging are I/O boundaries and are
not unit-tested here. What IS pure and worth testing: that run_inference builds the
request from only the system+user turns, applies clean_response to the completion,
and returns the held-out assistant turn as ground truth — and that the shared
scoring helpers are the very ones from extract_eval (not a drifting copy).
"""
import sys
import types
from types import SimpleNamespace

import pytest

import extract_eval
import eval_cli


def test_shares_helpers_with_extract_eval():
    assert eval_cli.clean_response is extract_eval.clean_response
    assert eval_cli.score is extract_eval.score


def test_top_fields_are_the_business_priority_set():
    assert "PolicyNumber" in eval_cli.TOP_8_FIELDS
    assert "LoanPolicyDate" in eval_cli.TOP_8_FIELDS


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_run_inference_cleans_completion_and_uses_prompt_turns(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        # a leaked <think> block, as the served vLLM completion really looks
        return _FakeResp('<think>extracting</think>\n{"PolicyNumber": "P-1"}')

    monkeypatch.setattr(eval_cli.requests, "post", fake_post)

    records = [{
        "file_name": "doc1.txt",
        "messages": [
            {"role": "system", "content": "You extract fields."},
            {"role": "user", "content": "OCR text ..."},
            {"role": "assistant", "content": '{"PolicyNumber": "P-1"}'},  # held-out GT
        ],
    }]
    preds, errs = eval_cli.run_inference(records, max_new_tokens=128, max_workers=1)

    assert errs == []
    assert preds[0]["file_name"] == "doc1.txt"
    # completion cleaned of the <think> block so it parses
    assert preds[0]["pred"] == '{"PolicyNumber": "P-1"}'
    # ground truth is the held-out assistant turn
    assert preds[0]["ground_truth"] == '{"PolicyNumber": "P-1"}'

    # request carries only system + user (the assistant turn is never sent)
    assert [m["role"] for m in captured["payload"]["messages"]] == ["system", "user"]
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["max_tokens"] == 128
    assert captured["url"].endswith("/invocations")


def test_run_inference_records_errors_without_raising(monkeypatch):
    def boom(url, json, timeout):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(eval_cli.requests, "post", boom)
    records = [{"file_name": "d", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "{}"},
    ]}]
    preds, errs = eval_cli.run_inference(records, max_new_tokens=16, max_workers=1)
    assert preds == [] and errs == [0]


# --- build_scored: errored docs count as FN (fix: don't drop them, don't inflate F1) -
def test_build_scored_counts_errored_docs_as_false_negatives():
    records = [
        {"file_name": "d0", "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": '{"A": "1"}'}]},
        {"file_name": "d1", "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": '{"B": "2"}'}]},
    ]
    preds = [{"file_name": "d0", "pred": '{"A": "1"}', "ground_truth": '{"A": "1"}'}]
    scored = eval_cli.build_scored(preds, [1], records)     # record 1 errored

    assert len(scored) == 2
    errored = next(s for s in scored if s["file_name"] == "d1")
    assert errored["pred"] == ""                            # empty -> parses to {} -> FN
    assert errored["ground_truth"] == '{"B": "2"}'

    r = extract_eval.score(scored)
    # d0 is a TP, d1 (errored) is an FN — recall 0.5, NOT the inflated 1.0 you'd get
    # from scoring only the successful preds.
    assert r["tp"] == 1 and r["fn"] == 1


# --- evaluate_tags: one failing checkpoint is skipped, not fatal to the batch -------
def _fake_mlflow():
    m = types.ModuleType("mlflow")
    m.set_experiment = lambda e: None

    class _Run:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    m.start_run = lambda run_name=None: _Run()
    m.log_param = lambda *a, **k: None
    m.set_tags = lambda *a, **k: None
    m.log_metrics = lambda *a, **k: None
    return m


def test_evaluate_tags_skips_failing_checkpoint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow())

    def fake_eval_one(tag, ckpt, records, args):
        if tag == "bad":
            raise RuntimeError("vLLM did not become ready")
        return ({"f1": 0.9, "precision": 0.9, "recall": 0.9}, {"f1": 0.8}, 10, 0)

    monkeypatch.setattr(eval_cli, "eval_one", fake_eval_one)
    args = SimpleNamespace(experiment="/Users/me@databricks.com/exp")
    summary = eval_cli.evaluate_tags(["good1", "bad", "good2"], "/out", [], args)

    by_tag = {s["tag"]: s for s in summary}
    assert len(summary) == 3                       # all three attempted
    assert by_tag["bad"]["failed"] is True         # failure recorded, not raised
    assert not by_tag["good1"].get("failed")
    assert by_tag["good2"]["all_f1"] == 0.9        # batch continued past 'bad'


def test_evaluate_tags_lets_programming_errors_propagate(monkeypatch):
    # A KeyError/AttributeError is a bug, not a checkpoint failure — it must crash,
    # not be swallowed as a skipped checkpoint (narrowed except: RuntimeError/OSError).
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow())

    def bug(*a):
        raise KeyError("regression")

    monkeypatch.setattr(eval_cli, "eval_one", bug)
    args = SimpleNamespace(experiment="/Users/me@databricks.com/exp")
    with pytest.raises(KeyError):
        eval_cli.evaluate_tags(["x"], "/out", [], args)


def test_eval_one_hard_fails_when_all_inference_errors(monkeypatch):
    # Total inference failure (preds empty) is not evaluable — raise so evaluate_tags
    # skips it, rather than logging a spurious f1=0 run.
    monkeypatch.setattr(eval_cli.shutil, "copytree", lambda *a, **k: None)
    monkeypatch.setattr(eval_cli, "start_vllm", lambda *a, **k: object())
    monkeypatch.setattr(eval_cli, "stop_vllm", lambda p: None)
    monkeypatch.setattr(eval_cli, "run_inference", lambda records, mnt, mw: ([], [0, 1]))
    args = SimpleNamespace(max_model_len=1024, gpu_memory_util=0.9, startup_timeout=1,
                           max_new_tokens=1, max_workers=1)
    with pytest.raises(RuntimeError, match="no successful predictions"):
        eval_cli.eval_one("bad", "/ckpt", [{}, {}], args)
