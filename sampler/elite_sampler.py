from typing import Tuple, Dict
import numpy as np

from .base import BaseSampler


class EliteSampler(BaseSampler):
    def __init__(self, seed: int):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)

    def sample(self, archive: Dict) -> Tuple[Tuple[int], Tuple[int]]:
        min_base = 0.5
        max_base = 0.8
        
        keys = list(archive.keys())
        keys_array = np.array(keys)

        qualities = np.array([archive[key].quality for key in keys])
        probs = self._normalize(qualities, min_base, max_base)
        normalized_keys = [self._normalize(keys_array[:, i], min_base, max_base) for i in range(keys_array.shape[1])]
        for normalized_key in normalized_keys:
            probs *= normalized_key
        if np.sum(probs) == 0:
            probs = np.ones(len(probs))
        probs = probs / np.sum(probs)
        parents = self.np_random.choice(len(keys), 2, p=probs, replace=False)
        dad = keys[parents[0]]
        mom = keys[parents[1]]
        archive[dad].sampling_freq += 1
        archive[mom].sampling_freq += 1
        return dad, mom
    
    def _normalize(self, values: np.ndarray, min_base: float, max_base: float) -> np.ndarray:
        min_value = np.min(values)
        max_value = np.max(values)
        if min_value == max_value:
            return np.ones_like(values)
        return min_base + (values - min_value) * (max_base / (max_value - min_value))
