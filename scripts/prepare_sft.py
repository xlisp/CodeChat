"""Build an SFT jsonl from public Python instruction datasets.

Each line is {"input_ids": [...], "labels": [...]} where labels mask the
prompt with -100 so loss is only computed on the assistant response.
"""
import argparse
import json
import os

from codechat.tokenizer import encode, USER_TAG, ASSISTANT_TAG, END_TAG, EOT


BLOCK_SIZE = 2048


def format_example(instruction: str, inp: str, output: str) -> tuple[list[int], list[int]]:
    user_content = instruction if not inp else f"{instruction}\n\n{inp}"
    prompt = f"{USER_TAG}\n{user_content}\n{END_TAG}\n{ASSISTANT_TAG}\n"
    full = prompt + output + f"\n{END_TAG}\n"
    prompt_ids = encode(prompt)
    full_ids = encode(full) + [EOT]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # safety: clip
    full_ids = full_ids[:BLOCK_SIZE + 1]
    labels = labels[:BLOCK_SIZE + 1]
    return full_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train.jsonl")

    from datasets import load_dataset
    sources = []
    sources.append(load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train"))
    sources.append(load_dataset("sahil2801/CodeAlpaca-20k", split="train"))

    n = 0
    with open(out_path, "w") as f:
        for ds in sources:
            for ex in ds:
                instr = ex.get("instruction", "") or ""
                inp = ex.get("input", "") or ""
                out = ex.get("output", "") or ex.get("response", "") or ""
                if not instr or not out:
                    continue
                ids, labels = format_example(instr, inp, out)
                f.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
                n += 1
    print(f"wrote {n} SFT examples to {out_path}")


if __name__ == "__main__":
    main()
