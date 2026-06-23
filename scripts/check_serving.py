"""C2 health check: assert the serving endpoint is up (200) and returns a non-empty
completion. Works against the mock or the real server (same OpenAI endpoint).

    python scripts/check_serving.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Serving health check (C2)")
    parser.add_argument("--url", default="http://localhost:8000", help="server origin (no /v1)")
    parser.add_argument("--model", default="mock-llm")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    origin = args.url.rstrip("/")

    try:
        h = httpx.get(f"{origin}/health", timeout=args.timeout)
        if h.status_code != 200:
            print(f"FAIL: /health returned {h.status_code}")
            return 1
        print(f"ok  : GET /health -> 200 {h.json()}")

        c = httpx.post(
            f"{origin}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "namaste"}],
                "max_tokens": 16,
            },
            timeout=args.timeout,
        )
        if c.status_code != 200:
            print(f"FAIL: /v1/chat/completions returned {c.status_code}")
            return 1
        content = c.json()["choices"][0]["message"]["content"]
        if not content.strip():
            print("FAIL: completion was empty")
            return 1
        print("ok  : POST /v1/chat/completions -> 200, non-empty completion")
        print(f"      sample: {content[:80]!r}")
        print("C2 PASS")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not reach {origin}: {type(e).__name__}: {e}")
        print("Hint: start a server first  ->  make serve-mock   (or make serve on the box)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
