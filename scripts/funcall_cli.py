"""Interactive function-calling CLI for a v5 funcall checkpoint.

The v5 model was trained on glaive-function-calling-v2, whose prompt format
is:

    <|system|>
    You are a helpful assistant with access to the following functions.
    Use them if required -
    {
      "name": "get_weather",
      "description": "...",
      "parameters": {...}
    }
    <|end|>
    <|user|>
    What's the weather in Paris?
    <|end|>
    <|assistant|>

and it then emits:

    <functioncall> {"name": "get_weather", "arguments": {"location": "Paris"}} <|end|>

This CLI loads a ckpt, lets you paste a tool schema + user query, and
pretty-prints the parsed call.

Usage:
    python -m scripts.funcall_cli \
        --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt

    # non-interactive (one shot):
    python -m scripts.funcall_cli \
        --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt \
        --tools-file tools.json \
        --user "What's the weather in Paris?"
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F

from codechat.common import DEVICE, COMPUTE_DTYPE, seed_all
from codechat.gpt import GPT, GPTConfig
from codechat.checkpoint import load as load_ckpt
from codechat.tokenizer import encode, decode, END_TAG
from codechat.funcall_reward import (
    _extract_functioncall_json,
    _parse_json_loose,
    _unwrap_args,
)

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
FUNCRESP_TAG = "<|function_response|>"


DEFAULT_SYSTEM_HEADER = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required -"
)


def build_system(tools: list[dict] | str, header: str = DEFAULT_SYSTEM_HEADER) -> str:
    """Assemble the system message in glaive-v2 style.

    `tools` can be: a list of tool schemas (dicts), or an already-formatted
    string blob that the user pastes verbatim.
    """
    if isinstance(tools, str):
        return f"{header}\n{tools.strip()}"
    parts = [header]
    for t in tools:
        parts.append(json.dumps(t, indent=2))
    return "\n\n".join(parts)


def build_prompt(system_text: str, turns: list[tuple[str, str]]) -> str:
    """turns: [(role, content)] where role ∈ {'user','assistant','function'}."""
    parts = [f"{SYSTEM_TAG}\n{system_text}\n{END_TAG}\n"]
    for role, content in turns:
        tag = {
            "user": USER_TAG,
            "assistant": ASSISTANT_TAG,
            "function": FUNCRESP_TAG,
        }[role]
        parts.append(f"{tag}\n{content}\n{END_TAG}\n")
    parts.append(f"{ASSISTANT_TAG}\n")
    return "".join(parts)


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_k, block_size):
    """Single-sequence sampler, stops at END_TAG."""
    model.eval()
    ids = prompt_ids.clone()
    new_ids: list[int] = []
    eot_ids = set(encode(END_TAG))
    for _ in range(max_new_tokens):
        cond = ids[:, -block_size:]
        logits, _ = model(cond)
        step = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(step, min(top_k, step.size(-1)))
            step[step < v[:, [-1]]] = -float("inf")
        probs = F.softmax(step, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tok = int(nxt.item())
        new_ids.append(tok)
        ids = torch.cat([ids, nxt], dim=1)
        if tok in eot_ids and len(new_ids) > 4:
            break
    return new_ids


def parse_call(text: str) -> dict | None:
    """Extract {name, arguments} from a `<functioncall>` emission, or None."""
    blob = _extract_functioncall_json(text)
    if blob is None:
        return None
    parsed = _parse_json_loose(blob)
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    args = _unwrap_args(parsed.get("arguments"))
    if not isinstance(name, str):
        return None
    return {"name": name.strip(), "arguments": args if args is not None else {}}


def load_model(ckpt_path: str):
    state = load_ckpt(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    return model, cfg


def run_once(model, cfg, system_text, turns, max_new_tokens, temperature, top_k):
    prompt = build_prompt(system_text, turns)
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    max_prompt_len = cfg.block_size - max_new_tokens
    if prompt_ids.shape[1] > max_prompt_len:
        prompt_ids = prompt_ids[:, -max_prompt_len:]
    new_ids = generate(
        model, prompt_ids, max_new_tokens, temperature, top_k, cfg.block_size
    )
    text = decode(new_ids)
    if END_TAG in text:
        text = text.split(END_TAG)[0]
    return text.strip()


def _load_tools(tools_file: str | None, tools_inline: str | None):
    if tools_file:
        with open(tools_file) as f:
            raw = f.read()
        try:
            return json.loads(raw)      # list of dicts
        except Exception:
            return raw                  # pass through as text blob
    if tools_inline:
        try:
            return json.loads(tools_inline)
        except Exception:
            return tools_inline
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        }
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tools-file",
                    help="JSON file with a list of tool schemas, or free text")
    ap.add_argument("--tools",
                    help="inline tool schemas (JSON list or text blob)")
    ap.add_argument("--user", help="one-shot user message; omit for REPL")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--raw", action="store_true",
                    help="print raw model text only, don't parse the call")
    args = ap.parse_args()

    seed_all(args.seed)
    assert DEVICE.type == "cuda", "CUDA required"
    torch.set_float32_matmul_precision("high")

    print(f"loading {args.ckpt} ...")
    model, cfg = load_model(args.ckpt)
    print(f"loaded (depth={cfg.depth}, block_size={cfg.block_size}, "
          f"dtype={COMPUTE_DTYPE})")

    tools = _load_tools(args.tools_file, args.tools)
    system_text = build_system(tools)

    def run_and_show(turns):
        raw = run_once(
            model, cfg, system_text, turns,
            args.max_new_tokens, args.temperature, args.top_k,
        )
        print("\n--- raw output ---")
        print(raw)
        if not args.raw:
            call = parse_call(raw)
            print("\n--- parsed call ---")
            if call is None:
                print("  (no valid <functioncall> detected — model may be "
                      "chatting instead of calling a tool)")
            else:
                print(f"  name:      {call['name']}")
                print(f"  arguments: {json.dumps(call['arguments'], ensure_ascii=False)}")
        return raw

    # One-shot mode
    if args.user:
        turns = [("user", args.user)]
        run_and_show(turns)
        return

    # REPL
    print("\n=== funcall REPL ===")
    print(f"tools in system message:\n{system_text}\n")
    print("type a user message; after each model call you'll be prompted to")
    print("paste a FUNCTION RESPONSE or just hit enter to start a new turn.")
    print("Ctrl-C to exit.\n")
    turns: list[tuple[str, str]] = []
    try:
        while True:
            user = input("user> ").strip()
            if not user:
                continue
            turns.append(("user", user))
            raw = run_and_show(turns)
            turns.append(("assistant", raw))
            fr = input("\nfunction_response> (enter to skip) ").strip()
            if fr:
                turns.append(("function", fr))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
