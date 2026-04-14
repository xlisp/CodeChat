"""Build an extended code-SFT jsonl from MBPP / Codeforces-Python / the-stack-smol.

Purpose: rescue codechat_8b_sft's code ability before running RL. v2 pipeline
found the base model could only produce parseable Python ~1.75% of the time
on MBPP; this stage piles targeted instruction-to-code data on top of the
original CodeAlpaca / python_code_instructions SFT mix.

Sources (all default ON; disable via --sources):
  - mbpp            ~600 non-train problems from MBPP "sanitized"
  - codeforces      MatrixStudio/Codeforces-Python-Submissions (capped)
  - the-stack-smol  bigcode/the-stack-smol python files with a module docstring

Excludes MBPP "train" split on purpose: scripts.eval_mbpp_pass_at_k and
scripts.chat_rl both use sanitized train. Training SFT on it would inflate
the pass@k gate used to decide whether to run RL.

Each output line is {"input_ids": [...], "labels": [...]} with the prompt
masked to -100, compatible with codechat.dataloader.SFTLoader.
"""
import argparse
import ast
import json
import os

from codechat.tokenizer import encode, USER_TAG, ASSISTANT_TAG, END_TAG, EOT


BLOCK_SIZE = 2048


def format_example(instruction: str, output: str) -> tuple[list[int], list[int]]:
    prompt = f"{USER_TAG}\n{instruction}\n{END_TAG}\n{ASSISTANT_TAG}\n"
    full = prompt + output + f"\n{END_TAG}\n"
    prompt_ids = encode(prompt)
    full_ids = encode(full) + [EOT]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    full_ids = full_ids[: BLOCK_SIZE + 1]
    labels = labels[: BLOCK_SIZE + 1]
    return full_ids, labels


def iter_mbpp():
    """MBPP sanitized, excluding the 'train' split used by diagnostic/RL."""
    from datasets import load_dataset
    for split in ("test", "validation", "prompt"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception as e:
            print(f"  [mbpp] split={split} load failed: {e}")
            continue
        for ex in ds:
            prompt = ex.get("prompt") or ex.get("text") or ""
            code = ex.get("code") or ""
            tests = ex.get("test_list") or []
            if not prompt.strip() or not code.strip():
                continue
            instr = prompt.strip()
            if tests:
                instr += "\n\nYour code should pass these tests:\n" + "\n".join(tests)
            yield instr, f"```python\n{code.strip()}\n```"


def iter_codeforces(max_n: int, max_code_len: int = 4000):
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
        if not desc or not code:
            continue
        if len(code) > max_code_len:
            continue
        parts = [desc]
        if in_spec:
            parts.append("Input:\n" + in_spec)
        if out_spec:
            parts.append("Output:\n" + out_spec)
        instr = "\n\n".join(parts)
        yield instr, f"```python\n{code}\n```"
        n += 1


def iter_the_stack_smol(max_n: int, min_doc_len: int = 20, max_code_len: int = 4000):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft_code")
    ap.add_argument(
        "--sources",
        default="mbpp,codeforces,the-stack-smol",
        help="comma-separated subset of {mbpp,codeforces,the-stack-smol}",
    )
    ap.add_argument("--max-codeforces", type=int, default=20000)
    ap.add_argument("--max-the-stack", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train.jsonl")
    enabled = {s.strip() for s in args.sources.split(",") if s.strip()}

    counts = {"mbpp": 0, "codeforces": 0, "the-stack-smol": 0}
    total = 0
    with open(out_path, "w") as f:
        def dump(source: str, gen):
            nonlocal total
            try:
                for instr, out in gen:
                    ids, labels = format_example(instr, out)
                    if sum(1 for l in labels if l != -100) < 8:
                        continue
                    f.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
                    counts[source] += 1
                    total += 1
                    if counts[source] % 5000 == 0:
                        print(f"  {source} -> {counts[source]} ...")
            except Exception as e:
                print(f"  [{source}] aborted after {counts[source]} examples: {type(e).__name__}: {e}")

        if "mbpp" in enabled:
            dump("mbpp", iter_mbpp())
            print(f"  mbpp -> {counts['mbpp']} examples")
        if "codeforces" in enabled:
            dump("codeforces", iter_codeforces(args.max_codeforces))
            print(f"  codeforces -> {counts['codeforces']} examples")
        if "the-stack-smol" in enabled:
            dump("the-stack-smol", iter_the_stack_smol(args.max_the_stack))
            print(f"  the-stack-smol -> {counts['the-stack-smol']} examples")

    print(f"wrote {total} code SFT examples to {out_path}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
