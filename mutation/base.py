import numpy as np
import torch
from typing import Dict


class BaseMutator(object):

    num_mutation_params: int

    def _generate_mutation_params(self) -> np.ndarray:
        raise NotImplementedError

    def _mutate(self,
               weight_dict: Dict,
               q_name: str,
               mutation_params: np.ndarray) -> Dict:
        raise NotImplementedError

    def update_seed(self, seed: int):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def mutate(self,
               weight_dict: Dict,
               q_name: str,
               mutation_params: np.ndarray=None) -> Dict:
        if mutation_params is None:
            mutation_params = self._generate_mutation_params()
        else:
            mutation_params = np.array(mutation_params)
        assert mutation_params.size == self.num_mutation_params, (
            f"{mutation_params.size} vs {self.num_mutation_params}")
        return self._mutate(weight_dict, q_name, mutation_params)