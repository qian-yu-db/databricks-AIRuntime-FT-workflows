"""Unit tests for scripts/eval_cli.py — the vLLM eval that runs on the GPU worker.

The vLLM serving path (subprocess) and MLflow logging are I/O boundaries and are
not unit-tested here. What IS pure and worth testing: that run_inference builds the
request from only the system+user turns, applies clean_response to the completion,
and returns the held-out assistant turn as ground truth — and that the shared
scoring helpers are the very ones from extract_eval (not a drifting copy).
"""
from types import SimpleNamespace

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
