"""Extract RL prompts + ground-truth function-calls from glaive-function-calling-v2.

For each conversation we walk through turns and find the *first* assistant
turn that emits a `<functioncall>`. Everything strictly before that turn
becomes the prompt (system + user + any prior assistant chit-chat), and the
function name + arguments become the ground truth reward target.

This mirrors MathGPT's GSM8K RL setup: the SFT data and RL data come from
the same distribution, so the RL model never has to generalize to a new
format — it only has to shift probability mass toward correct calls.

Outputs two JSONL files under --out-dir:
    train.jsonl : RL training prompts  (bulk of the data, e.g. 90%)
    eval.jsonl  : held-out eval prompts (used for online pass@k during RL)

Each row:
    {
        "prompt": "<|system|>\n...\n<|end|>\n<|user|>\n...\n<|end|>\n<|assistant|>\n",
        "gt_name": "get_weather",
        "gt_args": {"location": "Paris"}
    }
"""
import argparse
import json
import os
import random
import re

from codechat.tokenizer import END_TAG, USER_TAG, ASSISTANT_TAG, encode
from scripts.prepare_sft_funcall import parse_turns, SYSTEM_TAG, FUNCRESP_TAG, TAG_OF
from codechat.funcall_reward import _extract_functioncall_json, _parse_json_loose, _unwrap_args


# A glaive assistant turn with a function call typically looks like:
#   <functioncall> {"name": "foo", "arguments": '{"x":1}'}
# We reuse the same balanced-brace extractor as the reward function so
# parsing is consistent at train time and eval time.


def extract_gt_call(assistant_text: str) -> tuple[str, dict] | None:
    """Return (name, args_dict) if the assistant turn contains a valid call."""
    blob = _extract_functioncall_json(assistant_text)
    if blob is None:
        return None
    parsed = _parse_json_loose(blob)
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = _unwrap_args(parsed.get("arguments"))
    if args is None:
        args = {}
    return name.strip(), args


def build_prompt(turns_before_call: list[tuple[str, str]]) -> str:
    """Reconstruct the prefix up to (but not including) the assistant call.

    We append `<|assistant|>\n` at the end so sampling starts inside the
    assistant role — the model only needs to produce the `<functioncall>`
    payload (and stop on <|end|>).
    """
    parts = []
    for role, content in turns_before_call:
        tag = TAG_OF[role]
        parts.append(f"{tag}\n{content}\n{END_TAG}\n")
    parts.append(f"{ASSISTANT_TAG}\n")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/rl_funcall")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="0 = use all; cap for quick iteration")
    ap.add_argument("--eval-ratio", type=float, default=0.05,
                    help="fraction of rows held out for online eval (default 5%%)")
    ap.add_argument("--max-prompt-tokens", type=int, default=1400,
                    help="skip rows whose prompt already eats too much context, "
                         "leaving no room for the function-call completion")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    rng = random.Random(args.seed)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    n_seen = 0
    n_no_call = 0
    n_too_long = 0

    for ex in ds:
        n_seen += 1
        if args.max_examples and (len(train_rows) + len(eval_rows)) >= args.max_examples:
            break
        turns = parse_turns(ex.get("system", ""), ex.get("chat", ""))
        # Find first assistant turn that contains a <functioncall>.
        call_idx = None
        for i, (role, content) in enumerate(turns):
            if role == "assistant" and "<functioncall>" in content:
                call_idx = i
                break
        if call_idx is None:
            n_no_call += 1
            continue

        gt = extract_gt_call(turns[call_idx][1])
        if gt is None:
            n_no_call += 1
            continue
        gt_name, gt_args = gt

        prompt = build_prompt(turns[:call_idx])
        # Rough token budget check (we don't want prompts that starve the
        # sampling budget). Encode is cheap relative to HF load overhead.
        prompt_len = len(encode(prompt))
        if prompt_len > args.max_prompt_tokens:
            n_too_long += 1
            continue

        row = {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "gt_name": gt_name,
            "gt_args": gt_args,
        }
        if rng.random() < args.eval_ratio:
            eval_rows.append(row)
        else:
            train_rows.append(row)

        total = len(train_rows) + len(eval_rows)
        if total % 5000 == 0:
            print(f"  processed {n_seen}, kept {total} "
                  f"(train={len(train_rows)} eval={len(eval_rows)})")

    train_path = os.path.join(args.out_dir, "train.jsonl")
    eval_path = os.path.join(args.out_dir, "eval.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(eval_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    print(
        f"wrote RL funcall data: train={len(train_rows)} eval={len(eval_rows)} "
        f"(seen={n_seen}, no_call={n_no_call}, too_long={n_too_long})"
    )
    print(f"  -> {train_path}")
    print(f"  -> {eval_path}")


if __name__ == "__main__":
    main()
