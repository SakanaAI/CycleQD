import torch
import numpy as np
from typing import Dict, List

from .base import BaseModelMerger


class ModelwiseLinearMerge(BaseModelMerger):
    """Also called chat vector merge."""

    def __init__(self, std: float=0.01):
        super().__init__()
        self.num_merge_params = 2  # Mean and std of the normal distribution.
        self.std = std

    def _generate_merge_params(self) -> np.ndarray:
        return np.array([0., self.std])

    def _merge(self,
               task_vectors: List[Dict],
               merge_params: np.ndarray) -> Dict:
        num_task_vectors = len(task_vectors)
        weights = torch.normal(
            1. + float(merge_params[0]), abs(float(merge_params[1])),
            size=(num_task_vectors,)
        )
        merged_params = {}
        for k in task_vectors[0]:
            merged_params[k] = torch.zeros_like(task_vectors[0][k])
            for i in range(num_task_vectors):
                merged_params[k] += weights[i] * task_vectors[i][k]
            merged_params[k] /= weights.sum()
        return merged_params