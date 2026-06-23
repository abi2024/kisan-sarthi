"""serving/server.py — launch the real OpenAI-compatible serving endpoint.

Model-agnostic by design: the model comes from `KISAN_MODEL_ID` (env) or
serving/config/serving.yaml, never hard-coded — so swapping Llama -> Nemotron, or one
checkpoint for another, is a config change (Milestone Map risk register).

  Wk1  : vLLM            -> `vllm serve <model> ...`           (this file, the fast path)
  Wk2  : TensorRT-LLM    -> build engine first, then serve     (stub below; see
                            serving/trtllm/build_engine.py and serving/quantize.py)

Importing this module is side-effect free and GPU-free. The launch happens only in
`main()`, which execs the server process. On a machine without vLLM installed it exits
with a clear message (install the GPU extra on the box: `uv sync --extra gpu`).

Run on the box:
    KISAN_MODEL_ID=nvidia/Nemotron-... python -m kisan_sarthi.serving.server
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent / "config" / "serving.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load serving/config/serving.yaml if present; return {} otherwise."""
    if not path.exists():
        return {}
    import yaml

    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve_settings(cli: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Precedence: CLI flag > environment variable > yaml > built-in default."""

    def pick(cli_val: Any, env_key: str, cfg_key: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        if os.getenv(env_key) is not None:
            return os.getenv(env_key)
        if cfg.get(cfg_key) is not None:
            return cfg[cfg_key]
        return default

    return {
        "backend": pick(cli.backend, "KISAN_SERVING_BACKEND", "backend", "vllm"),
        "model": pick(cli.model, "KISAN_MODEL_ID", "model_id", None),
        "host": pick(cli.host, "KISAN_SERVING_HOST", "host", "127.0.0.1"),
        "port": int(pick(cli.port, "KISAN_SERVING_PORT", "port", 8000)),
        "dtype": pick(cli.dtype, "KISAN_SERVING_DTYPE", "dtype", "auto"),
        "max_model_len": cfg.get("max_model_len"),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization"),
        "max_num_seqs": cfg.get("max_num_seqs"),
        "kv_cache_dtype": cfg.get("kv_cache_dtype"),
        "quantization": cfg.get("quantization"),
    }


def build_vllm_cmd(s: dict[str, Any]) -> list[str]:
    """Pure: turn resolved settings into a `vllm serve ...` argv. Unit-tested."""
    if not s.get("model"):
        raise ValueError("No model set. Provide --model or KISAN_MODEL_ID or serving.yaml:model_id")
    cmd = [
        "vllm",
        "serve",
        str(s["model"]),
        "--host",
        str(s["host"]),
        "--port",
        str(s["port"]),
        "--served-model-name",
        str(s["model"]),
        "--dtype",
        str(s.get("dtype") or "auto"),
    ]
    if s.get("max_model_len"):
        cmd += ["--max-model-len", str(s["max_model_len"])]
    if s.get("gpu_memory_utilization"):
        cmd += ["--gpu-memory-utilization", str(s["gpu_memory_utilization"])]
    if s.get("max_num_seqs"):
        cmd += ["--max-num-seqs", str(s["max_num_seqs"])]
    if s.get("kv_cache_dtype") and s["kv_cache_dtype"] != "auto":
        cmd += ["--kv-cache-dtype", str(s["kv_cache_dtype"])]
    if s.get("quantization"):  # fp8 / nvfp4 etc. — set as L3.2 / L3.3 land
        cmd += ["--quantization", str(s["quantization"])]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Kisan Sarthi serving endpoint")
    parser.add_argument("--backend", default=None, help="vllm | trtllm")
    parser.add_argument("--model", default=None, help="model id / path (else KISAN_MODEL_ID)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dtype", default=None, help="auto | float16 | bfloat16")
    parser.add_argument(
        "--print-cmd", action="store_true", help="print the launch command and exit (no GPU)"
    )
    args = parser.parse_args()

    s = resolve_settings(args, load_config())
    backend = str(s["backend"]).lower()

    if backend == "trtllm":
        sys.exit(
            "TensorRT-LLM backend is a Wk2 milestone (L3.2). Build the engine via "
            "serving/trtllm/build_engine.py, then serve it. Stub not yet implemented."
        )
    if backend != "vllm":
        sys.exit(f"Unknown backend '{backend}'. Use 'vllm' (Wk1) or 'trtllm' (Wk2).")

    try:
        cmd = build_vllm_cmd(s)
    except ValueError as e:
        sys.exit(str(e))

    if args.print_cmd:
        print(" ".join(cmd))
        return

    if shutil.which("vllm") is None:
        sys.exit(
            "vllm is not installed. This is expected on a laptop. On the GPU box run:\n"
            "    uv sync --extra gpu\n"
            "For local development, run the mock instead:\n"
            "    python -m app.mocks.mock_llm --port 8000"
        )

    print(f"[serving] exec: {' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)  # replace this process with the server


if __name__ == "__main__":
    main()
