import torch
import logging
import numpy as np
from typing import Dict

from .base import BaseMutator


class SVDUniformMutator(BaseMutator):

    def __init__(self,
                 mutation_rate: float,
                 seed_model_svd_path: str):
        self.num_mutation_params = 1
        self.mutation_rate = mutation_rate
        self.logger = logging.getLogger("SVDMutator")
        self.logger.info(f"Loading SVD dictionary from {seed_model_svd_path}")
        self.svd_dict = torch.load(seed_model_svd_path)
        self.logger.info(f"Done")

    def _generate_mutation_params(self) -> np.ndarray:
        return np.array(self.mutation_rate)

    def _mutate(self,
                weight_dict:
                Dict, q_name: str,
                mutation_params: np.ndarray) -> Dict:
        cpu = torch.device("cpu")
        for k in weight_dict:
            if "norm" in k:
                continue
            rank = min(weight_dict[k].shape)
            scales = torch.rand(
                size=(rank,), dtype=torch.bfloat16
            ).cuda()
            scales *= float(mutation_params)
            # Note: these are SVD of the task vector, not the model weights
            u = self.svd_dict[q_name][f"{k}.U"].cuda()
            s = self.svd_dict[q_name][f"{k}.S"].cuda()
            v = self.svd_dict[q_name][f"{k}.V"].cuda()
            weight_dict[k] += (
                u @ torch.diag_embed(scales * s) @ v.T
            ).to(cpu)
            u.to(cpu)
            v.to(cpu)
            s.to(cpu)
            scales.to(cpu)
        return weight_dict