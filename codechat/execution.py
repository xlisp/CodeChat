"""Sandboxed-ish Python code execution for RL reward.

We run generated code + hidden test asserts in a subprocess with a hard
timeout. Reward = fraction of asserts that pass (0.0 .. 1.0).

This is NOT a real security sandbox — only run on trusted training boxes.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
import os
import textwrap


def run_with_tests(code: str, tests: list[str], timeout: float = 5.0) -> float:
    if not tests:
        return 0.0
    harness = code + "\n\n"
    harness += "_passed = 0\n_total = {}\n".format(len(tests))
    for i, t in enumerate(tests):
        # each test is typically an `assert ...` line
        harness += textwrap.dedent(f"""
        try:
            {t}
            _passed += 1
        except Exception:
            pass
        """)
    harness += "print('REWARD', _passed / _total)\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        for line in out.stdout.splitlines()[::-1]:
            if line.startswith("REWARD "):
                try:
                    return float(line.split()[1])
                except Exception:
                    return 0.0
        return 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def extract_code(text: str) -> str:
    """Pull code out of a model response. Prefer ```python ... ``` blocks,
    fall back to the whole text."""
    if "```" in text:
        parts = text.split("```")
        # parts: [before, lang+code, after, ...]
        for i in range(1, len(parts), 2):
            chunk = parts[i]
            if chunk.startswith("python"):
                chunk = chunk[len("python"):]
            return chunk.strip()
    return text.strip()
