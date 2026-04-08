"""Thin wrapper around tiktoken GPT-2 BPE.

We reuse GPT-2's BPE because on Python source it already compresses well
(~3.5 bytes/token) and avoids training our own tokenizer.
"""
from __future__ import annotations
import tiktoken

_ENC = tiktoken.get_encoding("gpt2")

# Reserve some "special" tokens for chat formatting. GPT-2 BPE has 50257 tokens
# (including <|endoftext|>). We repurpose <|endoftext|> as both BOS and EOS,
# and encode chat turn boundaries as plain text markers that get BPE'd.
VOCAB_SIZE = _ENC.n_vocab  # 50257
EOT = _ENC.eot_token       # 50256

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
END_TAG = "<|end|>"


def encode(text: str) -> list[int]:
    return _ENC.encode(text, disallowed_special=())


def decode(ids: list[int]) -> str:
    return _ENC.decode(ids)


def encode_chat(messages: list[dict]) -> list[int]:
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
    parts = []
    for m in messages:
        tag = USER_TAG if m["role"] == "user" else ASSISTANT_TAG
        parts.append(f"{tag}\n{m['content']}\n{END_TAG}\n")
    return encode("".join(parts))
