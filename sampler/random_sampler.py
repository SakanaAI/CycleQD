from typing import Tuple, Dict
import numpy as np

from .base import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, seed: int):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)

    def sample(self, archive: Dict) -> Tuple[Tuple[int], Tuple[int]]:
        keys = list(archive.keys())
        probs = np.ones(len(keys)) / len(keys)
        parents = self.np_random.choice(len(keys), 2, p=probs, replace=False)
        dad = keys[parents[0]]
        mom = keys[parents[1]]
        return dad, mom
