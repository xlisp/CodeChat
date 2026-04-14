"""Sandboxed-ish Python code execution for RL reward.

Two reward shapes:

  - mode="fractional" (the original): reward = k / n where k is the number of
    asserts that pass out of n. Still 0.0 whenever the code doesn't parse or
    crashes at import time → for weak base models every rollout scores 0 and
    GRPO has no gradient signal.

  - mode="tiered" (recommended for weak base models): staircase reward that
    credits partial progress, so group-rewards have non-zero variance even
    before the model can pass any test:

        code extraction failed / empty         → 0.00
        syntactic garbage (ast.parse fails)    → 0.00
        parseable but exec() raises            → 0.05
        exec() OK but zero tests pass          → 0.15
        k of n tests pass (0 < k < n)          → 0.15 + 0.85 * (k / n)
        all n tests pass                       → 1.00

    Thresholds are deliberately coarse so advantage isn't dominated by trivial
    syntax wins.

This is NOT a real security sandbox — only run on trusted training boxes.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
import os
import textwrap
import ast


def _build_harness(code: str, tests: list[str], mode: str) -> str:
    """Build a subprocess script that prints `REWARD <float>` on its last line."""
    n = len(tests)
    # Each test runs in its own try/except so one failure doesn't abort the rest.
    test_block = ""
    for t in tests:
        test_block += textwrap.dedent(f"""
        try:
            {t}
            _passed += 1
        except Exception:
            pass
        """)

    if mode == "fractional":
        return (
            code
            + f"\n\n_passed = 0\n_total = {n}\n"
            + test_block
            + "print('REWARD', _passed / _total if _total else 0.0)\n"
        )

    # mode == "tiered": we do the staircase inside the harness so that
    # exec failures still print a valid REWARD line.
    harness = textwrap.dedent(f"""
    import traceback
    _reward = 0.05  # reached here => code parsed + loaded at module scope
    try:
        # Re-exec the user code in an isolated namespace; any NameError /
        # ImportError / runtime error while defining helpers bumps us down.
        _ns = {{}}
        exec(_USER_CODE, _ns)
    except Exception:
        print('REWARD', 0.05)
        raise SystemExit(0)

    # Now run the tests against the populated namespace.
    _passed = 0
    _total = {n}
    globals().update(_ns)
    {textwrap.indent(test_block, '    ').lstrip()}
    if _total == 0:
        _reward = 0.15
    elif _passed == 0:
        _reward = 0.15
    elif _passed == _total:
        _reward = 1.0
    else:
        _reward = 0.15 + 0.85 * (_passed / _total)
    print('REWARD', _reward)
    """)
    # Embed the user code as a string literal so a SyntaxError inside it
    # doesn't kill the harness before we can score it.
    return f"_USER_CODE = {code!r}\n" + harness


def run_with_tests(code: str, tests: list[str], timeout: float = 5.0,
                   mode: str = "fractional") -> float:
    if not tests:
        return 0.0
    if not code or not code.strip():
        return 0.0

    # Cheap gate: if `code` itself isn't parseable, we already know the answer
    # for both modes and can skip the subprocess.
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    harness = _build_harness(code, tests, mode)
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
        # In tiered mode a hang still means the code at least parsed — but an
        # infinite loop is worse than "runs and fails", so return the runnable
        # tier floor, not higher.
        return 0.05 if mode == "tiered" else 0.0
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
