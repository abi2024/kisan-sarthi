"""Level-1 unit tests for serving config parsing + vLLM command building (no GPU)."""

import argparse

import pytest

from kisan_sarthi.serving.server import build_vllm_cmd, resolve_settings


def _ns(**kw):
    base = dict(backend=None, model=None, host=None, port=None, dtype=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_vllm_cmd_minimal():
    cmd = build_vllm_cmd({"model": "org/model", "host": "127.0.0.1", "port": 8000, "dtype": "auto"})
    assert cmd[:3] == ["vllm", "serve", "org/model"]
    assert "--port" in cmd and "8000" in cmd
    assert "--served-model-name" in cmd


def test_build_vllm_cmd_includes_optimization_knobs():
    cmd = build_vllm_cmd(
        {
            "model": "m",
            "host": "h",
            "port": 1,
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 16,
            "kv_cache_dtype": "fp8",
            "quantization": "fp8",
        }
    )
    assert "--max-model-len" in cmd and "--kv-cache-dtype" in cmd and "fp8" in cmd
    assert "--quantization" in cmd


def test_build_vllm_cmd_requires_model():
    with pytest.raises(ValueError):
        build_vllm_cmd({"model": None, "host": "h", "port": 1, "dtype": "auto"})


def test_resolve_precedence_cli_over_env(monkeypatch):
    monkeypatch.setenv("KISAN_MODEL_ID", "from-env")
    s = resolve_settings(_ns(model="from-cli"), {"model_id": "from-yaml"})
    assert s["model"] == "from-cli"


def test_resolve_precedence_env_over_yaml(monkeypatch):
    monkeypatch.setenv("KISAN_MODEL_ID", "from-env")
    s = resolve_settings(_ns(), {"model_id": "from-yaml"})
    assert s["model"] == "from-env"


def test_resolve_falls_back_to_yaml(monkeypatch):
    monkeypatch.delenv("KISAN_MODEL_ID", raising=False)
    s = resolve_settings(_ns(), {"model_id": "from-yaml", "port": 9001})
    assert s["model"] == "from-yaml" and s["port"] == 9001


def test_build_vllm_cmd_emits_trust_remote_code_and_parsers():
    cmd = build_vllm_cmd(
        {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "host": "0.0.0.0",
            "port": 8000,
            "dtype": "auto",
            "trust_remote_code": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "nano_v3",
            "reasoning_parser_plugin": "nano_v3_reasoning_parser.py",
        }
    )
    assert "--trust-remote-code" in cmd
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "nano_v3"
    assert cmd[cmd.index("--reasoning-parser-plugin") + 1] == "nano_v3_reasoning_parser.py"


def test_build_vllm_cmd_omits_model_flags_when_unset():
    cmd = build_vllm_cmd({"model": "m", "host": "h", "port": 1, "dtype": "auto"})
    assert "--trust-remote-code" not in cmd
    assert "--reasoning-parser" not in cmd
    assert "--tool-call-parser" not in cmd


def test_build_vllm_cmd_appends_extra_args():
    cmd = build_vllm_cmd(
        {
            "model": "m",
            "host": "h",
            "port": 1,
            "dtype": "auto",
            "extra_args": ["--async-scheduling"],
        }
    )
    assert cmd[-1] == "--async-scheduling"


def test_resolve_settings_reads_model_family_flags_from_yaml(monkeypatch):
    monkeypatch.delenv("KISAN_MODEL_ID", raising=False)
    cfg = {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "trust_remote_code": True,
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "nano_v3",
    }
    s = resolve_settings(_ns(), cfg)
    assert s["trust_remote_code"] is True
    assert s["tool_call_parser"] == "qwen3_coder"
    assert s["reasoning_parser"] == "nano_v3"
    assert s["model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
