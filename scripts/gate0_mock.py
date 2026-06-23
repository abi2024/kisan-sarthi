"""Laptop Gate-0 proof: boot the mock LLM, run the C2 health check and a small C5 bench
against it, then tear the server down. No GPU required.

    python scripts/gate0_mock.py
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx

PORT = 8000
ORIGIN = f"http://127.0.0.1:{PORT}"


def _wait_for_health(timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{ORIGIN}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def main() -> int:
    print(f"[gate0] starting mock LLM on {ORIGIN} ...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.mocks.mock_llm:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ]
    )
    try:
        if not _wait_for_health():
            print("[gate0] FAIL: mock LLM did not become healthy")
            return 1
        print("[gate0] mock LLM healthy. Running C2 check_serving ...\n")
        rc = subprocess.call([sys.executable, "scripts/check_serving.py", "--url", ORIGIN])
        if rc != 0:
            return rc
        print("\n[gate0] Running C5 bench (concurrency 1) ...\n")
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "eval.perf.bench",
                "--label",
                "baseline",
                "--concurrency",
                "1",
                "--endpoint",
                ORIGIN,
            ]
        )
        if rc != 0:
            return rc
        print("\n[gate0] C2 + C5 demonstrated against the mock. Gate-0 serving path is green.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
