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
