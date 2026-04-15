"""Dense, format-based reward for function-calling RL.

Why this shape of reward:
  CodeChat's MBPP RL failed because the reward was binary (pass all unit
  tests -> 1.0, else 0.0). For an 8B model with weak code SFT this meant
  every rollout scored 0, GRPO's advantage collapsed to 0, and training
  did nothing for 415 steps.

  MathGPT succeeded because GSM8K reward was also "binary" in the arithmetic
  sense but the *format* (`#### <number>` at the end) is cheap to hit — the
  SFT model already produces the right shape, so reward has non-zero mean
  and variance from step 0.

  For function calling we have the same structure — the model emits a
  `<functioncall> {json}` blob. Parsing it gives us multiple discrete signals:

     model output                                 → tier
     ─────────────────────────────────────────────────────
     no <functioncall> tag at all                 → 0.00
     tag present, body not valid JSON             → 0.15
     JSON OK but missing 'name'                   → 0.30
     JSON OK, wrong function name                 → 0.35
     right name, no/empty arguments               → 0.55
     right name, args parse, partial match        → 0.55 + 0.45 * match_frac
     right name, all arguments exactly match      → 1.00

  This is *dense*: even a totally confused model that just learns "emit
  <functioncall>" already gets 0.15. Every step up the staircase is a
  gradient signal.

Exact float values are calibrated so that:
  - the top tier (full match) dominates partial credit (1.0 vs max 0.99),
  - the bottom tier (syntax only) is far enough below "right name" that
    group-relative advantage actually prefers correctness,
  - tiers are spaced enough that noise doesn't drown them out.
"""
from __future__ import annotations
import json
import re
from typing import Any


# The SFT format uses a plain-text sentinel `<functioncall>` followed by a
# JSON object on the same line. Some glaive rows have `<functioncall>` as a
# standalone tag; we match both loosely. Greedy-match the { ... } balanced
# brace region — JSON parser then confirms validity.
_FUNCCALL_RE = re.compile(
    r"<\s*functioncall\s*>\s*(\{.*?\})(?:\s*</\s*functioncall\s*>)?",
    re.DOTALL,
)


def _extract_functioncall_json(text: str) -> str | None:
    """Return the first JSON blob after a <functioncall> tag, or None.

    We look for a balanced `{...}` after the tag. The regex above does a
    non-greedy match which fails on nested braces in arguments, so we
    additionally do a manual bracket-balance pass when the regex hits.
    """
    if "<functioncall>" not in text and "<functioncall " not in text:
        return None

    # Find the position just past the first tag.
    tag_match = re.search(r"<\s*functioncall\s*>", text)
    if not tag_match:
        return None
    start = tag_match.end()
    # Skip whitespace to find the opening brace.
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None

    # Balanced-brace scan. Respect string literals so braces inside strings
    # don't confuse us.
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < len(text):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
        j += 1
    return None


def _parse_json_loose(raw: str) -> Any | None:
    """Parse JSON, tolerating glaive's common quirks:

      * `arguments` is often a JSON-encoded STRING (double-escaped), not a
        nested object. We unwrap one level of string->json when needed.
      * single quotes where double should be.
      * trailing commas.
    """
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Quick repairs
    fixed = raw.replace("'", '"')
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except Exception:
        return None


def _unwrap_args(args_field: Any) -> dict | None:
    """`arguments` can be a dict, a JSON string encoding a dict, or junk."""
    if args_field is None:
        return {}
    if isinstance(args_field, dict):
        return args_field
    if isinstance(args_field, str):
        parsed = _parse_json_loose(args_field)
        if isinstance(parsed, dict):
            return parsed
    return None


def _arg_match_fraction(pred: dict, gt: dict) -> float:
    """Fraction of GT keys whose values (as strings after normalization) match pred.

    Numeric tolerance: we compare floats with 1e-6 relative tolerance.
    Strings: case-insensitive, whitespace-trimmed.
    Missing keys in `pred` count as mismatch. Extra keys in `pred` are ignored
    (but see below — we *do* cap arg_match_frac based on extras to discourage
    argument hallucination).
    """
    if not gt:
        # If GT has no arguments and pred has no arguments -> full match.
        return 1.0 if not pred else 0.5
    matched = 0
    for k, gt_v in gt.items():
        if k not in pred:
            continue
        pv = pred[k]
        if _values_equal(pv, gt_v):
            matched += 1
    frac = matched / len(gt)
    # Penalize argument hallucination (extra keys not in GT) mildly so the
    # optimizer doesn't discover "spam keys" as a shortcut.
    extras = [k for k in pred if k not in gt]
    if extras:
        frac *= max(0.5, 1.0 - 0.1 * len(extras))
    return frac


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except Exception:
            return False
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, str) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except Exception:
            return False
    if isinstance(b, str) and isinstance(a, (int, float)):
        return _values_equal(b, a)
    # list/dict: structural equality via JSON canonicalisation
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except Exception:
        return a == b


def funcall_reward(text: str, gt_name: str, gt_args: dict) -> tuple[float, str]:
    """Return (reward, tier_label). tier_label is for logging only.

    `text` is the assistant's raw output (possibly truncated — we only need
    the first <functioncall> tag). `gt_name` / `gt_args` come from the
    glaive-function-calling-v2 ground-truth assistant turn.
    """
    blob = _extract_functioncall_json(text)
    if blob is None:
        return 0.00, "no_tag"
    parsed = _parse_json_loose(blob)
    if not isinstance(parsed, dict):
        return 0.15, "bad_json"
    name = parsed.get("name")
    if name is None:
        return 0.30, "no_name"
    if not isinstance(name, str) or name.strip() != gt_name.strip():
        return 0.35, "wrong_name"
    # Right name. Evaluate arguments.
    args = _unwrap_args(parsed.get("arguments"))
    if args is None:
        return 0.55, "name_only"
    match_frac = _arg_match_fraction(args, gt_args or {})
    if match_frac >= 0.999:
        return 1.00, "full_match"
    return 0.55 + 0.45 * match_frac, f"partial_{match_frac:.2f}"


def funcall_exact_match(text: str, gt_name: str, gt_args: dict) -> int:
    """Binary version for pass@k evaluation — 1 iff name AND all GT args match."""
    r, _ = funcall_reward(text, gt_name, gt_args)
    return 1 if r >= 0.999 else 0
