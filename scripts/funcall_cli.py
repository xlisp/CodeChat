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
import re
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
    """Extract {name, arguments} from a `<functioncall>` emission, or None.

    Tolerates the model's common output quirks:
      - `arguments` wrapped in single quotes (Python-repr JSON-in-JSON)
      - `arguments` as an already-parsed dict
      - `arguments` as a double-quoted JSON-encoded string
    """
    blob = _extract_functioncall_json(text)
    if blob is None:
        return None

    # Happy path: the whole thing parses as JSON
    parsed = _parse_json_loose(blob)
    if isinstance(parsed, dict):
        name = parsed.get("name")
        if isinstance(name, str):
            args = _unwrap_args(parsed.get("arguments"))
            return {"name": name.strip(),
                    "arguments": args if args is not None else {}}

    # Fallback: blob itself isn't parseable (typically because of the
    # single-quoted `arguments` issue). Extract name + args-blob via regex
    # and parse them separately.
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', blob)
    if not name_m:
        return None
    name = name_m.group(1).strip()

    # arguments can appear as: `'{...}'` (single-quoted string), `"{...}"`
    # (double-quoted, usually escaped), or a bare `{...}` object.
    args: dict = {}
    for pat in (
        r'"arguments"\s*:\s*\'(\{.*?\})\'',   # single-quoted JSON-in-JSON
        r'"arguments"\s*:\s*"(\{.*?\})"',     # double-quoted JSON-in-JSON
        r'"arguments"\s*:\s*(\{.*?\})',       # inline object
    ):
        m = re.search(pat, blob, re.DOTALL)
        if m:
            inner = _parse_json_loose(m.group(1))
            if isinstance(inner, dict):
                args = inner
            break
    return {"name": name, "arguments": args}


def load_model(ckpt_path: str):
    state = load_ckpt(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE).to(COMPUTE_DTYPE)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    return model, cfg


def chat(model, cfg, system_text, user_msg, executors,
         max_new_tokens=256, temperature=0.7, top_k=50, max_rounds=4,
         verbose=False):
    """Full function-calling loop.

    Args:
      executors: {"tool_name": callable(**kwargs) -> json-serializable result}
      max_rounds: cap to avoid infinite tool-call loops.

    Returns (final_text, turns) — final_text is the model's natural-language
    reply after tools have run; turns is the full conversation for debugging.
    """
    turns: list[tuple[str, str]] = [("user", user_msg)]
    for _ in range(max_rounds):
        raw = run_once(model, cfg, system_text, turns,
                       max_new_tokens, temperature, top_k)
        turns.append(("assistant", raw))
        if verbose:
            print(f"[assistant] {raw}")
        call = parse_call(raw)
        if call is None:
            # Model gave a plain-text answer (no tool needed, or already answered)
            return raw, turns
        fn = executors.get(call["name"])
        if fn is None:
            msg = f'{{"error": "unknown tool: {call["name"]}"}}'
        else:
            try:
                result = fn(**call["arguments"])
                msg = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                msg = json.dumps({"error": f"{type(e).__name__}: {e}"})
        if verbose:
            print(f"[function_response] {msg}")
        turns.append(("function", msg))
    # Hit round cap — surface whatever the last assistant turn was.
    return turns[-1][1] if turns[-1][0] == "assistant" else "", turns


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


BUILTIN_EXECUTORS = {
    # Mock weather — for quick end-to-end demo without a real API.
    "get_weather": lambda location, unit="celsius", **_: {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "conditions": "partly cloudy",
        "_note": "mock response from BUILTIN_EXECUTORS; replace with a real API call",
    },
}


def _load_executors(executors_path: str | None) -> dict:
    """Load {tool_name: callable} from a Python file defining EXECUTORS dict.

    Files loaded this way can call any library (requests, DB clients, etc.)
    so tools perform real work. Without a path, we fall back to the
    in-script BUILTIN_EXECUTORS (mocks).
    """
    if not executors_path:
        return dict(BUILTIN_EXECUTORS)
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_executors", executors_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    execs = getattr(mod, "EXECUTORS", None)
    if not isinstance(execs, dict):
        raise SystemExit(f"{executors_path}: must define EXECUTORS = {{...}}")
    return execs


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
    ap.add_argument("--executors",
                    help="Python file defining EXECUTORS={name: callable}; "
                         "if omitted, uses in-script mocks for built-in tools")
    ap.add_argument("--no-run", action="store_true",
                    help="don't auto-execute tools; just print the parsed call "
                         "(old one-shot behavior)")
    ap.add_argument("--max-rounds", type=int, default=4,
                    help="max tool-call rounds per user turn")
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
    executors = _load_executors(args.executors)

    def run_auto(user_msg):
        """Full loop: model call → tool execute → model reply."""
        answer, turns = chat(
            model, cfg, system_text, user_msg, executors,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_rounds=args.max_rounds,
            verbose=True,
        )
        print("\n=== FINAL ANSWER ===")
        print(answer)
        return turns

    def run_oneshot(user_msg):
        """Old behavior: emit one call, don't execute."""
        raw = run_once(
            model, cfg, system_text, [("user", user_msg)],
            args.max_new_tokens, args.temperature, args.top_k,
        )
        print("\n--- raw output ---")
        print(raw)
        if not args.raw:
            call = parse_call(raw)
            print("\n--- parsed call ---")
            if call is None:
                print("  (no valid <functioncall> detected)")
            else:
                print(f"  name:      {call['name']}")
                print(f"  arguments: {json.dumps(call['arguments'], ensure_ascii=False)}")

    # One-shot mode
    if args.user:
        if args.no_run:
            run_oneshot(args.user)
        else:
            print(f"executors available: {sorted(executors.keys())}")
            run_auto(args.user)
        return

    # REPL
    print("\n=== funcall REPL ===")
    print(f"executors available: {sorted(executors.keys())}")
    print(f"tools in system message:\n{system_text}\n")
    if args.no_run:
        print("--no-run: just printing parsed calls; paste function_response manually.")
    else:
        print("auto-run: tools will execute via registered executors.")
    print("Ctrl-C to exit.\n")
    try:
        while True:
            user = input("user> ").strip()
            if not user:
                continue
            if args.no_run:
                run_oneshot(user)
            else:
                run_auto(user)
            print()
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
