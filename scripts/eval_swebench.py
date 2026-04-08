"""Evaluate a trained CodeChat model on SWE-bench (Lite by default).

Pipeline:
  1. Load the SWE-bench dataset from HuggingFace (princeton-nlp/SWE-bench_Lite
     or princeton-nlp/SWE-bench_Verified).
  2. For each instance, build a prompt containing the problem statement and
     (optionally) a truncated view of the repo. Ask the model to output a
     unified-diff patch inside a ```diff ...``` block.
  3. Extract the patch and write a predictions file in the SWE-bench format:
         {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}
  4. Print the exact command to run the official SWE-bench harness (which
     runs each patch inside a per-instance Docker container and reports the
     resolved rate). We do NOT run Docker ourselves — that's the job of the
     official harness and requires root + Docker on the training box.

Usage:
    python -m scripts.eval_swebench \
        --ckpt checkpoints/codechat_2b_rl/latest.pt \
        --split lite \
        --out predictions/codechat_2b_rl.jsonl
"""
import argparse
import json
import os
import torch
from tqdm import tqdm

from codechat.common import DEVICE, COMPUTE_DTYPE
from codechat.gpt import GPT, GPTConfig
from codechat.checkpoint import load as load_ckpt
from codechat.tokenizer import encode, decode, USER_TAG, ASSISTANT_TAG, END_TAG


SPLITS = {
    "lite":     ("princeton-nlp/SWE-bench_Lite",     "test"),
    "verified": ("princeton-nlp/SWE-bench_Verified", "test"),
    "full":     ("princeton-nlp/SWE-bench",          "test"),
}


SYSTEM_INSTRUCTION = (
    "You are an expert Python developer. You will be given a GitHub issue "
    "describing a bug in an open-source Python repository. Produce a minimal "
    "unified-diff patch that fixes the issue. Respond with ONLY a single "
    "```diff ...``` fenced block. The diff must apply cleanly with "
    "`git apply` from the repository root."
)


def build_prompt(instance: dict) -> str:
    problem = instance.get("problem_statement", "") or ""
    repo = instance.get("repo", "")
    base = instance.get("base_commit", "")[:12]
    hints = instance.get("hints_text", "") or ""

    user = f"{SYSTEM_INSTRUCTION}\n\nRepo: {repo}\nBase commit: {base}\n\n"
    user += f"--- Issue ---\n{problem}\n"
    if hints:
        user += f"\n--- Maintainer hints ---\n{hints[:2000]}\n"
    user += "\nReturn the fix as a unified diff now."
    return f"{USER_TAG}\n{user}\n{END_TAG}\n{ASSISTANT_TAG}\n"


def extract_diff(text: str) -> str:
    """Pull a diff out of a model response. Accept ```diff ...``` or
    ```patch ...``` or fall back to the whole text starting at 'diff --git'."""
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            chunk = parts[i]
            for lang in ("diff", "patch"):
                if chunk.startswith(lang):
                    chunk = chunk[len(lang):]
                    break
            chunk = chunk.strip()
            if "diff --git" in chunk or chunk.startswith("---"):
                return chunk
    idx = text.find("diff --git")
    if idx >= 0:
        return text[idx:].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=list(SPLITS.keys()), default="lite")
    ap.add_argument("--out", default="predictions/codechat.jsonl")
    ap.add_argument("--model-name", default="codechat-2b")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="0 = all instances")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    state = load_ckpt(args.ckpt)
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt} | depth={cfg.depth} n_embd={cfg.n_embd} block={cfg.block_size}")

    from datasets import load_dataset
    ds_name, split = SPLITS[args.split]
    ds = load_dataset(ds_name, split=split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"evaluating on {ds_name} / {split}: {len(ds)} instances")

    n_empty = 0
    with open(args.out, "w") as f:
        for ex in tqdm(ds, desc="generating patches"):
            prompt = build_prompt(ex)
            ids = encode(prompt)
            # hard-truncate prompts that exceed context
            max_prompt = cfg.block_size - args.max_new_tokens
            if len(ids) > max_prompt:
                ids = ids[-max_prompt:]
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            out = model.generate(
                x, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
            )
            new_text = decode(out[0, x.shape[1]:].tolist())
            if END_TAG in new_text:
                new_text = new_text.split(END_TAG)[0]
            patch = extract_diff(new_text)
            if not patch:
                n_empty += 1
            f.write(json.dumps({
                "instance_id": ex["instance_id"],
                "model_name_or_path": args.model_name,
                "model_patch": patch,
            }) + "\n")

    print(f"\nwrote {args.out}  (empty patches: {n_empty}/{len(ds)})")
    print("\nNext: run the official SWE-bench evaluation harness (requires Docker):")
    print("  pip install swebench")
    print(f"  python -m swebench.harness.run_evaluation \\")
    print(f"      --dataset_name {ds_name} \\")
    print(f"      --predictions_path {args.out} \\")
    print(f"      --max_workers 4 \\")
    print(f"      --run_id codechat_{args.split}")
    print("\nThe harness will apply each patch, run the failing + passing tests")
    print("in a per-instance Docker container, and print %resolved.")


if __name__ == "__main__":
    main()
