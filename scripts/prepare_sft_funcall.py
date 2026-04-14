"""Build a function-calling SFT jsonl from glaiveai/glaive-function-calling-v2.

The dataset has two relevant columns:
  - system: function schema + system prompt
  - chat:   multi-turn transcript using plain-text markers
              SYSTEM:/USER:/ASSISTANT:/A:/FUNCTION RESPONSE: ... <|endoftext|>

We flatten each row into a single (input_ids, labels) pair compatible with
codechat.dataloader.SFTLoader:
  - every turn is wrapped with codechat chat tags (<|user|>, <|assistant|>, ...)
  - labels are -100 for system / user / function-response tokens and equal to
    the real token id for assistant tokens (so loss only fires on model output,
    including the <functioncall> JSON it must emit)

We deliberately reuse the existing GPT-2 BPE — the new role tags
(<|system|>, <|function_response|>, <functioncall>) BPE into multi-token
sequences, same way USER_TAG/ASSISTANT_TAG already do. No tokenizer change.
"""
import argparse
import json
import os
import re

from codechat.tokenizer import encode, END_TAG, USER_TAG, ASSISTANT_TAG, EOT


BLOCK_SIZE = 2048

SYSTEM_TAG = "<|system|>"
FUNCRESP_TAG = "<|function_response|>"

# Glaive markers appear at the start of a line (after a preceding \n) or at
# the very start of the string. We split on them greedily.
TURN_MARKERS = re.compile(
    r"(?:^|\n)\s*(SYSTEM:|USER:|ASSISTANT:|A:|FUNCTION RESPONSE:)\s*",
)

ROLE_OF = {
    "SYSTEM:": "system",
    "USER:": "user",
    "ASSISTANT:": "assistant",
    "A:": "assistant",
    "FUNCTION RESPONSE:": "function",
}

TAG_OF = {
    "system": SYSTEM_TAG,
    "user": USER_TAG,
    "assistant": ASSISTANT_TAG,
    "function": FUNCRESP_TAG,
}


def parse_turns(system_text: str, chat_text: str) -> list[tuple[str, str]]:
    """Return [(role, content), ...]. Strips trailing <|endoftext|> sentinels."""
    turns: list[tuple[str, str]] = []
    sys_clean = (system_text or "").strip()
    # Some rows store system inline in `chat` with "SYSTEM:" prefix. The
    # dedicated `system` column is the common case.
    if sys_clean:
        # Drop a leading "SYSTEM: " / "You are ..." header — keep body verbatim.
        turns.append(("system", sys_clean))

    # Tokenize the chat string by marker splits.
    # re.split keeps the delimiters when the pattern has a capture group.
    pieces = TURN_MARKERS.split(chat_text or "")
    # pieces: [pre, marker, content, marker, content, ...]
    # If the chat string starts with a marker, pieces[0] is "" (ignore).
    i = 1
    while i < len(pieces):
        marker = pieces[i].strip()
        content = pieces[i + 1] if i + 1 < len(pieces) else ""
        i += 2
        role = ROLE_OF.get(marker)
        if role is None:
            continue
        # Strip the <|endoftext|> sentinel the dataset appends to assistant
        # turns. We add our own END_TAG / EOT during tokenization.
        content = content.replace("<|endoftext|>", "").strip()
        if not content:
            continue
        turns.append((role, content))
    return turns


def tokenize(turns: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """Turn a flat turn list into (input_ids, labels) of equal length.

    Only assistant turns contribute to loss; everything else is masked -100.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for role, content in turns:
        tag = TAG_OF[role]
        prefix_ids = encode(f"{tag}\n")
        body_ids = encode(content)
        suffix_ids = encode(f"\n{END_TAG}\n")

        if role == "assistant":
            # Prefix stays masked — model should not be rewarded for emitting
            # its own role tag; generation harness prepends it at inference.
            input_ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))
            # Body + suffix are the actual supervision targets.
            input_ids.extend(body_ids + suffix_ids)
            labels.extend(body_ids + suffix_ids)
        else:
            span = prefix_ids + body_ids + suffix_ids
            input_ids.extend(span)
            labels.extend([-100] * len(span))

    # Terminal EOT — also a supervised target if the final turn was assistant,
    # otherwise leave masked (shouldn't happen in glaive rows, but be safe).
    if turns and turns[-1][0] == "assistant":
        input_ids.append(EOT)
        labels.append(EOT)
    else:
        input_ids.append(EOT)
        labels.append(-100)

    # Hard clip to BLOCK_SIZE + 1 (SFTLoader does x=ids[:-1], y=labels[1:]).
    input_ids = input_ids[: BLOCK_SIZE + 1]
    labels = labels[: BLOCK_SIZE + 1]
    return input_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft_funcall")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="0 = use all rows; otherwise subsample for quick runs")
    ap.add_argument("--min-assistant-tokens", type=int, default=8,
                    help="drop rows with too little supervision signal")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train.jsonl")

    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

    n_written = 0
    n_skipped = 0
    with open(out_path, "w") as f:
        for idx, ex in enumerate(ds):
            if args.max_examples and n_written >= args.max_examples:
                break
            turns = parse_turns(ex.get("system", ""), ex.get("chat", ""))
            if not any(r == "assistant" for r, _ in turns):
                n_skipped += 1
                continue
            ids, labels = tokenize(turns)
            n_supervised = sum(1 for l in labels if l != -100)
            if n_supervised < args.min_assistant_tokens:
                n_skipped += 1
                continue
            f.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
            n_written += 1
            if n_written % 5000 == 0:
                print(f"  wrote {n_written} examples ...")

    print(f"wrote {n_written} funcall SFT examples to {out_path} "
          f"(skipped {n_skipped})")


if __name__ == "__main__":
    main()
