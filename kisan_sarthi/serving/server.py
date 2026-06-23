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
    KISAN_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 python -m kisan_sarthi.serving.server
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
        # Model-family flags (set in serving.yaml for the chosen model; omitted when unset
        # so the launcher stays model-agnostic). Required for custom-arch models such as
        # Nemotron-3-Nano (hybrid Mamba/MoE, nemotron_h) which won't load without
        # --trust-remote-code.
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
        "enable_auto_tool_choice": bool(cfg.get("enable_auto_tool_choice", False)),
        "tool_call_parser": cfg.get("tool_call_parser"),
        "reasoning_parser": cfg.get("reasoning_parser"),
        "reasoning_parser_plugin": cfg.get("reasoning_parser_plugin"),
        # Escape hatch: any extra raw vllm args (list of strings) appended verbatim.
        "extra_args": cfg.get("extra_args") or [],
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
    # Model-family flags (omitted unless set in config). Nemotron-3-Nano needs
    # --trust-remote-code to load at all; the tool/reasoning parsers shape how vLLM
    # exposes tool calls and separates reasoning traces from the final answer.
    if s.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]
    if s.get("enable_auto_tool_choice"):
        cmd += ["--enable-auto-tool-choice"]
    if s.get("tool_call_parser"):
        cmd += ["--tool-call-parser", str(s["tool_call_parser"])]
    if s.get("reasoning_parser_plugin"):
        cmd += ["--reasoning-parser-plugin", str(s["reasoning_parser_plugin"])]
    if s.get("reasoning_parser"):
        cmd += ["--reasoning-parser", str(s["reasoning_parser"])]
    for arg in s.get("extra_args") or []:
        cmd.append(str(arg))
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

    # The reasoning-parser plugin is a file fetched from the model repo, e.g.:
    #   wget https://huggingface.co/<model>/resolve/main/nano_v3_reasoning_parser.py
    # vLLM resolves it relative to the working dir; warn early rather than fail mid-launch.
    plugin = s.get("reasoning_parser_plugin")
    if plugin and not Path(plugin).exists():
        print(
            f"[serving] WARNING: reasoning-parser-plugin '{plugin}' not found in {os.getcwd()}.\n"
            f"          Fetch it first, e.g.:\n"
            f"            wget https://huggingface.co/{s['model']}/resolve/main/{plugin}\n"
            f"          (or unset reasoning_parser_plugin in serving.yaml for the baseline run).",
            flush=True,
        )

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
