"""Unit tests for scripts/register_model.py. The MLflow log/register path is an I/O
boundary (runs on the GPU worker) and isn't unit-tested; the pure, worth-testing bit
is the vLLM entrypoint command baked into the model metadata — especially that it
disables Qwen3 thinking so served completions are clean JSON."""
import register_model


def test_build_entrypoint_core_flags():
    cmd = register_model.build_entrypoint("qwen3", 8080, 20480, 0.95)
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model qwen3" in cmd
    assert "--served-model-name qwen" in cmd
    assert "--port 8080" in cmd
    assert "--max-model-len 20480" in cmd
    assert "--gpu-memory-utilization 0.95" in cmd


def test_build_entrypoint_disables_thinking():
    # The whole reason we don't string-patch chat_template.jinja: the served model
    # must not leak <think> blocks into the JSON.
    cmd = register_model.build_entrypoint("qwen3", 8080, 20480, 0.95)
    assert '--chat-template-kwargs {"enable_thinking": false}' in cmd


def test_build_entrypoint_dtype_default_and_override():
    assert "--dtype bfloat16" in register_model.build_entrypoint("m", 8080, 1024, 0.9)
    assert "--dtype float16" in register_model.build_entrypoint("m", 8080, 1024, 0.9, dtype="float16")
