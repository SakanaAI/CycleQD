import numpy as np
import torch
from typing import Tuple, Dict


class BaseSampler(object):
    def __init__(self, seed: int, mutation_rate: float):
        self.mutation_rate = mutation_rate
        self.np_random = np.random.RandomState(seed)
    
    def update_seed(self, seed: int):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def sample(self, archive: Dict) -> Tuple[Tuple[int], Tuple[int]]:
        raise NotImplementedError