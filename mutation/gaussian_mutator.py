import torch
import numpy as np
from typing import Dict

from .base import BaseMutator


class GaussianMutator(BaseMutator):

    def __init__(self, mutation_rate: float):
        self.num_mutation_params = 1 
        self.mutation_rate = mutation_rate

    def _generate_mutation_params(self) -> np.ndarray:
        return np.array(self.mutation_rate)

    def _mutate(self,
               weight_dict: Dict,
               q_name: str,
               mutation_params: np.ndarray) -> Dict:
        for key, value in weight_dict.items():
            weight_dict[key] += torch.normal(
                0, std=abs(float(mutation_params)), size=value.shape,
            ).to(value.dtype)
        return weight_dict