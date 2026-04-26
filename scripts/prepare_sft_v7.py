"""Build the v7 unified SFT corpus.

Output schema (minimind-style conversations):

    {"source": "code"|"funcall"|"negative",
     "conversations": [
        {"role": "system"|"user"|"assistant"|"function", "content": str},
        ...
     ]}

Why this format (vs v6's pre-tokenized {input_ids, labels}):
  - The dataloader does a 20% no-tools system-prompt injection at runtime,
    which has to happen on the message tree, not on token ids.
  - One file, one consumer; debuggable as plain JSONL.

Sources (all default ON, disable via --sources):

  CODE
    mbpp           ~600 problems from MBPP "sanitized" (validation/test/prompt;
                   train is reserved for eval/RL gating).
    codeforces     MatrixStudio/Codeforces-Python-Submissions, capped.
    the-stack      bigcode/the-stack-smol Python files with module docstrings.
    humaneval      openai_humaneval (164 problems, all of them).
    codealpaca     sahil2801/CodeAlpaca-20k Python instructions.

  FUNCALL
    glaive         glaiveai/glaive-function-calling-v2 multi-turn transcripts.

  NEGATIVE
    Synthesized: take a real funcall sample's system block (with tool schemas),
    paste a real code sample's user question and assistant answer. Trains the
    model that "system has tools field" alone is NOT a sufficient signal to
    emit <functioncall> — the user's intent matters. Defaults to ~10% of the
    code count. v6 lacked these examples entirely.

Token-balance:
  v6 had ~30:1 funcall:code in supervised tokens (5.6:1 in rows × ~5 longer
  funcall transcripts). That overpressure killed code ability. v7 caps funcall
  rows so the supervised-token ratio lands near 1:1.2 (code+neg : funcall).
  --target-ratio controls this.

Outputs:
  train.jsonl         shuffled, full corpus
  eval_code.jsonl     held-out code-only sample (~300 rows) for per-domain eval
  eval_funcall.jsonl  held-out funcall-only sample (~300 rows)

Token counts in the summary are estimated from char length / 3.5 (GPT-2 BPE
on Python code averages ~3.5 chars/token; see codechat/tokenizer.py docstring).
Off by 10-20% but enough to size --funcall-cap correctly.
"""
import argparse
import ast
import json
import os
import random
import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Glaive parser — re-implementation kept local so we don't import from
# prepare_sft_funcall.py (it does its own tokenization, we don't want that).
# ---------------------------------------------------------------------------

GLAIVE_TURN_MARKERS = re.compile(
    r"(?:^|\n)\s*(SYSTEM:|USER:|ASSISTANT:|A:|FUNCTION RESPONSE:)\s*",
)
GLAIVE_ROLE_OF = {
    "SYSTEM:": "system",
    "USER:": "user",
    "ASSISTANT:": "assistant",
    "A:": "assistant",
    "FUNCTION RESPONSE:": "function",
}


def parse_glaive_row(system_text: str, chat_text: str) -> list[dict]:
    """Return [{role, content}, ...] from a glaive-v2 row."""
    turns: list[dict] = []
    sys_clean = (system_text or "").strip()
    if sys_clean:
        turns.append({"role": "system", "content": sys_clean})
    pieces = GLAIVE_TURN_MARKERS.split(chat_text or "")
    i = 1
    while i < len(pieces):
        marker = pieces[i].strip()
        content = pieces[i + 1] if i + 1 < len(pieces) else ""
        i += 2
        role = GLAIVE_ROLE_OF.get(marker)
        if role is None:
            continue
        content = content.replace("<|endoftext|>", "").strip()
        if not content:
            continue
        turns.append({"role": role, "content": content})
    return turns


# ---------------------------------------------------------------------------
# Code source iterators
# ---------------------------------------------------------------------------

def iter_mbpp() -> Iterable[tuple[str, str]]:
    from datasets import load_dataset
    for split in ("test", "validation", "prompt"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception as e:
            print(f"  [mbpp] split={split} load failed: {e}")
            continue
        for ex in ds:
            prompt = (ex.get("prompt") or ex.get("text") or "").strip()
            code = (ex.get("code") or "").strip()
            tests = ex.get("test_list") or []
            if not prompt or not code:
                continue
            instr = prompt
            if tests:
                instr += "\n\nYour code should pass these tests:\n" + "\n".join(tests)
            yield instr, f"```python\n{code}\n```"


def iter_codeforces(max_n: int, max_code_len: int = 4000) -> Iterable[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset(
        "MatrixStudio/Codeforces-Python-Submissions",
        split="train",
        streaming=True,
    )
    n = 0
    for ex in ds:
        if max_n and n >= max_n:
            break
        desc = (ex.get("problem-description") or "").strip()
        in_spec = (ex.get("input-specification") or "").strip()
        out_spec = (ex.get("output-specification") or "").strip()
        code = (ex.get("code") or "").strip()
        if not desc or not code or len(code) > max_code_len:
            continue
        parts = [desc]
        if in_spec:
            parts.append("Input:\n" + in_spec)
        if out_spec:
            parts.append("Output:\n" + out_spec)
        yield "\n\n".join(parts), f"```python\n{code}\n```"
        n += 1


def iter_the_stack(max_n: int, min_doc_len: int = 20, max_code_len: int = 4000):
    from datasets import load_dataset
    ds = load_dataset(
        "bigcode/the-stack-smol",
        data_dir="data/python",
        split="train",
        streaming=True,
    )
    n = 0
    for ex in ds:
        if max_n and n >= max_n:
            break
        src = ex.get("content") or ""
        if not src or len(src) > max_code_len:
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        doc = ast.get_docstring(tree)
        if not doc or len(doc.strip()) < min_doc_len:
            continue
        instr = "Write a Python module that matches this specification:\n\n" + doc.strip()
        yield instr, f"```python\n{src.strip()}\n```"
        n += 1


def iter_humaneval():
    from datasets import load_dataset
    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        print(f"  [humaneval] load failed: {e}")
        return
    for ex in ds:
        prompt = (ex.get("prompt") or "").strip()
        canonical = (ex.get("canonical_solution") or "").strip()
        tests = (ex.get("test") or "").strip()
        if not prompt or not canonical:
            continue
        instr = (
            "Complete the following Python function. Only output the body of the "
            "function (matching the given signature):\n\n```python\n"
            + prompt + "\n```"
        )
        # Reconstruct full solution = prompt + canonical
        answer = f"```python\n{prompt}{canonical}\n```"
        yield instr, answer


def iter_codealpaca(max_n: int):
    from datasets import load_dataset
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception as e:
        print(f"  [codealpaca] load failed: {e}")
        return
    n = 0
    for ex in ds:
        if max_n and n >= max_n:
            break
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not instr or not out:
            continue
        # Filter to Python-ish entries; the CodeAlpaca set is mixed-language.
        text = (instr + " " + out).lower()
        if "python" not in text and "def " not in out and "import " not in out:
            continue
        prompt = instr if not inp else f"{instr}\n\n{inp}"
        yield prompt, out
        n += 1


# ---------------------------------------------------------------------------
# Glaive iterator
# ---------------------------------------------------------------------------

def iter_glaive(max_n: int, min_assistant_chars: int = 16):
    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    n = 0
    for ex in ds:
        if max_n and n >= max_n:
            break
        turns = parse_glaive_row(ex.get("system", ""), ex.get("chat", ""))
        # Need at least one assistant turn with real content.
        ass_chars = sum(len(t["content"]) for t in turns if t["role"] == "assistant")
        if ass_chars < min_assistant_chars:
            continue
        if not any(t["role"] == "assistant" for t in turns):
            continue
        yield turns
        n += 1


# ---------------------------------------------------------------------------
# Token-count proxy (avoid double tokenizing; runtime tokenizer hits this row)
# ---------------------------------------------------------------------------

def supervised_token_estimate(conversations: list[dict]) -> int:
    """Approx supervised tokens = sum(len(assistant_content)) / 3.5.

    Codechat uses GPT-2 BPE; on Python source it's ~3.5 chars/token. Same
    tokenizer.py rationale. This is rough but sufficient for budget planning.
    """
    chars = sum(len(t.get("content", "")) for t in conversations
                if t.get("role") == "assistant")
    return int(chars / 3.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft_v7")
    ap.add_argument(
        "--sources",
        default="mbpp,codeforces,the-stack,humaneval,codealpaca,glaive",
        help="comma-separated subset",
    )
    ap.add_argument("--max-codeforces", type=int, default=60000)
    ap.add_argument("--max-the-stack", type=int, default=60000)
    ap.add_argument("--max-codealpaca", type=int, default=20000)
    ap.add_argument("--max-glaive", type=int, default=0,
                    help="0 = use all glaive (~113k), then cap by --funcall-cap")
    ap.add_argument("--funcall-cap", type=int, default=30000,
                    help="hard cap on funcall rows after subsampling. v6 used "
                         "~113k which produced a 30:1 supervised-token imbalance; "
                         "30k targets ~1:1.2 with the expanded code corpus.")
    ap.add_argument("--negative-frac", type=float, default=0.10,
                    help="fraction of code rows to repurpose as discriminative "
                         "negatives (system has tools, user asks for code, "
                         "assistant writes code WITHOUT <functioncall>).")
    ap.add_argument("--eval-per-domain", type=int, default=300,
                    help="held-out rows per domain for chat_sft_v7's per-domain "
                         "loss eval. Reserved BEFORE training shuffling.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enabled = {s.strip() for s in args.sources.split(",") if s.strip()}
    rng = random.Random(args.seed)

    # ---- 1. collect code samples ------------------------------------------
    code_samples: list[dict] = []
    code_counts: dict[str, int] = {}

    def add_code(source_name: str, gen):
        before = len(code_samples)
        for instr, out in gen:
            if not instr.strip() or not out.strip():
                continue
            code_samples.append({
                "source": "code",
                "_subsource": source_name,
                "conversations": [
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": out},
                ],
            })
        code_counts[source_name] = len(code_samples) - before
        print(f"  [code:{source_name}] +{code_counts[source_name]} (cum={len(code_samples)})")

    if "mbpp" in enabled:
        add_code("mbpp", iter_mbpp())
    if "codeforces" in enabled:
        add_code("codeforces", iter_codeforces(args.max_codeforces))
    if "the-stack" in enabled:
        add_code("the-stack", iter_the_stack(args.max_the_stack))
    if "humaneval" in enabled:
        add_code("humaneval", iter_humaneval())
    if "codealpaca" in enabled:
        add_code("codealpaca", iter_codealpaca(args.max_codealpaca))

    # ---- 2. collect funcall samples ---------------------------------------
    funcall_samples: list[dict] = []
    if "glaive" in enabled:
        for turns in iter_glaive(args.max_glaive):
            funcall_samples.append({
                "source": "funcall",
                "conversations": turns,
            })
        print(f"  [funcall:glaive] +{len(funcall_samples)}")

    # ---- 3. cap funcall to balance token budget ---------------------------
    if args.funcall_cap and len(funcall_samples) > args.funcall_cap:
        rng.shuffle(funcall_samples)
        dropped = len(funcall_samples) - args.funcall_cap
        funcall_samples = funcall_samples[: args.funcall_cap]
        print(f"  [funcall:cap] subsampled to {args.funcall_cap} (dropped {dropped})")

    # ---- 4. synthesize discriminative negatives ---------------------------
    # System with tools (from glaive) + code Q/A. Model sees: system has
    # tools but user wants code -> assistant writes code, no <functioncall>.
    negative_samples: list[dict] = []
    if funcall_samples and code_samples and args.negative_frac > 0:
        n_neg = int(len(code_samples) * args.negative_frac)
        # Pre-extract systems from funcall samples (those that actually
        # carry a system message, which is ~all of glaive).
        funcall_systems = [
            s["conversations"][0]
            for s in funcall_samples
            if s["conversations"] and s["conversations"][0]["role"] == "system"
        ]
        if funcall_systems:
            for _ in range(n_neg):
                code_ex = rng.choice(code_samples)
                sys_msg = rng.choice(funcall_systems)
                negative_samples.append({
                    "source": "negative",
                    "conversations": [sys_msg] + code_ex["conversations"],
                })
            print(f"  [negative:synth] +{len(negative_samples)} "
                  f"(={args.negative_frac:.0%} of code)")

    # ---- 5. carve held-out eval shards ------------------------------------
    rng.shuffle(code_samples)
    rng.shuffle(funcall_samples)
    n_eval = args.eval_per_domain
    eval_code = code_samples[:n_eval]
    eval_funcall = funcall_samples[:n_eval]
    train_code = code_samples[n_eval:]
    train_funcall = funcall_samples[n_eval:]

    # ---- 6. shuffle + write train.jsonl ----------------------------------
    train_all = train_code + train_funcall + negative_samples
    rng.shuffle(train_all)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    eval_code_path = os.path.join(args.out_dir, "eval_code.jsonl")
    eval_funcall_path = os.path.join(args.out_dir, "eval_funcall.jsonl")

    def write(path: str, samples: list[dict]):
        with open(path, "w") as f:
            for s in samples:
                # _subsource is bookkeeping only; not needed by the loader.
                out = {"source": s["source"], "conversations": s["conversations"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    write(train_path, train_all)
    write(eval_code_path, eval_code)
    write(eval_funcall_path, eval_funcall)

    # ---- 7. summary -------------------------------------------------------
    def tok_total(samples):
        return sum(supervised_token_estimate(s["conversations"]) for s in samples)

    code_tok = tok_total(train_code) + tok_total(negative_samples)
    func_tok = tok_total(train_funcall)
    print()
    print("=" * 68)
    print(f"  v7 SFT corpus written to {args.out_dir}/")
    print("=" * 68)
    print(f"  train.jsonl         {len(train_all):>7} rows  "
          f"({len(train_code)} code + {len(train_funcall)} funcall + "
          f"{len(negative_samples)} negative)")
    print(f"  eval_code.jsonl     {len(eval_code):>7} rows")
    print(f"  eval_funcall.jsonl  {len(eval_funcall):>7} rows")
    print()
    print(f"  Estimated supervised tokens (chars/3.5):")
    print(f"    code+negative : {code_tok:>11,}")
    print(f"    funcall       : {func_tok:>11,}")
    if func_tok > 0:
        ratio = code_tok / func_tok
        print(f"    code:funcall  : {ratio:.2f} : 1   (target ~0.8 - 1.2)")
        if ratio < 0.6:
            print(f"    WARN: code under-represented — consider raising "
                  f"--max-codeforces / --max-the-stack or lowering --funcall-cap.")
        elif ratio > 1.6:
            print(f"    WARN: funcall under-represented — consider raising "
                  f"--funcall-cap.")


if __name__ == "__main__":
    main()
