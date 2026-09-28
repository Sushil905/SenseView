"""Session-level sampling helpers."""

from __future__ import annotations

import random
from torch.utils.data import Sampler


class SessionSampler(Sampler[int]):
    """Sample each dataset session once, optionally in seeded random order."""

    def __init__(self, data_source, shuffle: bool = True, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
