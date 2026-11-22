"""Torch Dataset over the precomputed gesture windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GestureWindowDataset(Dataset):
    """Reads frames_<split>.npy + windows_<split>.npz built by build_gesture_dataset.py."""

    def __init__(self, root: Path, split: str, T: int = 16):
        self.root = Path(root)
        self.split = split
        self.T = T
        self.frames = np.load(self.root / f"frames_{split}.npy", mmap_mode="r")
        self.offsets = np.load(self.root / f"offsets_{split}.npy")
        w = np.load(self.root / f"windows_{split}.npz")
        self.video_idx = w["video_idx"]
        self.start = w["start"]
        self.center = w["center"]
        self.label = w["label"]

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, i: int):
        v = int(self.video_idx[i])
        s = int(self.start[i])
        base = int(self.offsets[v])
        x = np.array(self.frames[base + s: base + s + self.T], dtype=np.float32)
        y = int(self.label[i])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
