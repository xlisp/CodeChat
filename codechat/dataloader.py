"""Simple .bin shard dataloader. Each shard is a flat uint16 array of token ids.

For pretraining we stream random windows of block_size+1 from a random shard.
For SFT we load a jsonl of {input_ids, labels} lists.

`SFTConvLoader` is the v7 variant: reads conversations-format jsonl (minimind
style) and tokenizes at runtime so we can do random no-tools system-prompt
injection for code samples — that's how the model learns the discriminator
"system has tools field → emit <functioncall>; system absent or no tools →
write code". v6's failure was that <|system|> only ever appeared with funcall
data, so the model collapsed to "system tag → tool call".
"""
from __future__ import annotations
import os
import glob
import json
import random
import numpy as np
import torch


class PretrainLoader:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1337,
    ):
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        assert len(self.shards) > 0, f"no .bin shards found in {data_dir}"
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.rank = rank
        self.world_size = world_size
        # each rank gets an independent RNG so DDP ranks see non-overlapping windows
        self.rng = np.random.default_rng(seed + rank * 9973)
        self._mmaps: dict[str, np.ndarray] = {}

    def _load(self, path: str) -> np.ndarray:
        if path not in self._mmaps:
            self._mmaps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        return self._mmaps[path]

    def next_batch(self):
        shard = self._load(self.shards[self.rng.integers(len(self.shards))])
        ix = self.rng.integers(0, len(shard) - self.block_size - 1, size=(self.batch_size,))
        x = np.stack([shard[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([shard[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        return x, y


class SFTLoader:
    def __init__(self, jsonl_path: str, batch_size: int, block_size: int, device: torch.device):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        assert len(self.examples) > 0
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def next_batch(self):
        idxs = np.random.randint(0, len(self.examples), size=(self.batch_size,))
        xs, ys = [], []
        for i in idxs:
            ex = self.examples[i]
            ids = ex["input_ids"][: self.block_size + 1]
            labels = ex["labels"][: self.block_size + 1]
            # pad to block_size+1
            pad = self.block_size + 1 - len(ids)
            if pad > 0:
                ids = ids + [0] * pad
                labels = labels + [-100] * pad
            xs.append(ids[:-1])
            ys.append(labels[1:])
        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y


# ---------------------------------------------------------------------------
# v7 — conversations-format loader with runtime tokenization
# ---------------------------------------------------------------------------

# Pool used by the 20% no-tools system injection. Mix of EN/ZH so the model
# sees both. Tone deliberately matches what a code-assistant prompt looks like
# at inference time, NOT a tool-equipped assistant.
_NEUTRAL_SYSTEM_PROMPTS = [
    "You are a helpful Python coding assistant.",
    "You are a helpful AI assistant. Answer the user's questions clearly.",
    "You are an expert Python programmer. Write clean, idiomatic code.",
    "You are CodeChat, a small but useful coding assistant.",
    "You are a friendly assistant. Please reply concisely.",
    "你是一个乐于助人的 Python 编程助手。",
    "你是 CodeChat，一个轻量但有用的代码助手。",
    "你是一个专业的 AI 助手，请提供清晰的回答。",
]

# Codechat chat tags. Kept inline (not imported from tokenizer) because
# `<|system|>` and `<|function_response|>` are only used in funcall pipelines
# and were never promoted to tokenizer.py constants — see scripts/funcall_cli.py
# for the same pattern.
_USER_TAG = "<|user|>"
_ASSISTANT_TAG = "<|assistant|>"
_SYSTEM_TAG = "<|system|>"
_FUNCRESP_TAG = "<|function_response|>"
_END_TAG = "<|end|>"

_ROLE_TAG = {
    "system": _SYSTEM_TAG,
    "user": _USER_TAG,
    "assistant": _ASSISTANT_TAG,
    "function": _FUNCRESP_TAG,
    "tool": _FUNCRESP_TAG,  # minimind uses 'tool', glaive uses 'function'
}


def _has_system(conversations: list[dict]) -> bool:
    return bool(conversations) and conversations[0].get("role") == "system"


def _is_toolcall_sample(source: str, conversations: list[dict]) -> bool:
    """A sample is tool-use iff it was tagged that way OR its system carries
    tool schemas. We rely on the source tag (set by prepare_sft_v7.py) so the
    discriminator stays explicit and doesn't depend on string-matching."""
    if source in ("funcall", "negative"):
        return True
    # Defensive fallback if a sample slipped through with no source tag.
    if _has_system(conversations) and "function" in conversations[0].get("content", "").lower():
        return True
    return False


class SFTConvLoader:
    """Conversations-format SFT loader.

    Each jsonl line:
        {"source": "code"|"funcall"|"negative",
         "conversations": [{"role": "system"|"user"|"assistant"|"function",
                            "content": str}, ...]}

    Per __getitem__ behavior:
      1. If `source != funcall|negative` AND no leading system message AND
         RNG draws < `system_inject_ratio`, prepend a random no-tools system
         prompt. This is the v7 discriminator-training trick.
      2. Tokenize each turn as `{tag}\\n{content}\\n<|end|>\\n`.
      3. Loss mask: -100 on system/user/function tokens (including their
         `{tag}\\n` prefix and `\\n<|end|>\\n` suffix); supervise only the
         assistant body + suffix. Trailing EOT is supervised iff the final
         turn was assistant.
      4. Truncate to block_size + 1; pad with 0 / -100.
    """

    def __init__(
        self,
        jsonl_path: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1337,
        system_inject_ratio: float = 0.20,
    ):
        self.examples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
        assert len(self.examples) > 0, f"empty jsonl: {jsonl_path}"
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.system_inject_ratio = system_inject_ratio
        # Independent RNGs per rank so 8 GPUs see different injection draws
        # AND different sample orderings, mirroring PretrainLoader's idiom.
        self._rng = random.Random(seed + rank * 9973)
        self._np_rng = np.random.default_rng(seed + rank * 9973)

    def __len__(self):
        return len(self.examples)

    def _maybe_inject_system(self, ex: dict) -> list[dict]:
        convs = ex["conversations"]
        if _is_toolcall_sample(ex.get("source", ""), convs):
            return convs
        if _has_system(convs):
            return convs
        if self._rng.random() < self.system_inject_ratio:
            sys_msg = {"role": "system",
                       "content": self._rng.choice(_NEUTRAL_SYSTEM_PROMPTS)}
            return [sys_msg] + convs
        return convs

    def _tokenize(self, conversations: list[dict]) -> tuple[list[int], list[int]]:
        from codechat.tokenizer import encode, EOT
        input_ids: list[int] = []
        labels: list[int] = []
        last_role = None
        for turn in conversations:
            role = turn.get("role")
            tag = _ROLE_TAG.get(role)
            if tag is None:
                continue
            content = turn.get("content") or ""
            prefix_ids = encode(f"{tag}\n")
            body_ids = encode(content)
            suffix_ids = encode(f"\n{_END_TAG}\n")
            if role == "assistant":
                # Mask the role-tag prefix (the inference harness emits it for
                # the model), supervise body + suffix.
                input_ids.extend(prefix_ids)
                labels.extend([-100] * len(prefix_ids))
                input_ids.extend(body_ids + suffix_ids)
                labels.extend(body_ids + suffix_ids)
            else:
                span = prefix_ids + body_ids + suffix_ids
                input_ids.extend(span)
                labels.extend([-100] * len(span))
            last_role = role
        # Trailing EOT — supervise only if the final turn was assistant
        # (so the model learns to stop), otherwise mask.
        input_ids.append(EOT)
        labels.append(EOT if last_role == "assistant" else -100)
        # Hard clip to block_size + 1
        input_ids = input_ids[: self.block_size + 1]
        labels = labels[: self.block_size + 1]
        return input_ids, labels

    def _pack(self, ids: list[int], labels: list[int]) -> tuple[list[int], list[int]]:
        pad = self.block_size + 1 - len(ids)
        if pad > 0:
            ids = ids + [0] * pad
            labels = labels + [-100] * pad
        return ids[:-1], labels[1:]

    def next_batch(self):
        idxs = self._np_rng.integers(0, len(self.examples), size=(self.batch_size,))
        xs, ys = [], []
        for i in idxs:
            ex = self.examples[int(i)]
            convs = self._maybe_inject_system(ex)
            ids, labels = self._tokenize(convs)
            x_row, y_row = self._pack(ids, labels)
            xs.append(x_row)
            ys.append(y_row)
        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y

    def deterministic_iter(self, batch_size: int | None = None):
        """Yield (x, y) batches over the full dataset in order, no random
        injection. Used by the per-domain held-out eval loop in chat_sft_v7.py
        so eval loss is reproducible across steps."""
        bs = batch_size or self.batch_size
        buf_x, buf_y = [], []
        for ex in self.examples:
            convs = ex["conversations"]  # no injection
            ids, labels = self._tokenize(convs)
            x_row, y_row = self._pack(ids, labels)
            buf_x.append(x_row)
            buf_y.append(y_row)
            if len(buf_x) == bs:
                yield (torch.tensor(buf_x, dtype=torch.long, device=self.device),
                       torch.tensor(buf_y, dtype=torch.long, device=self.device))
                buf_x, buf_y = [], []
        if buf_x:
            yield (torch.tensor(buf_x, dtype=torch.long, device=self.device),
                   torch.tensor(buf_y, dtype=torch.long, device=self.device))
