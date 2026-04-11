"""Simple .bin shard dataloader. Each shard is a flat uint16 array of token ids.

For pretraining we stream random windows of block_size+1 from a random shard.
For SFT we load a jsonl of {input_ids, labels} lists.
"""
from __future__ import annotations
import os
import glob
import json
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
