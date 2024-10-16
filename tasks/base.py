from typing import List, Tuple
import numpy as np
from vllm import LLM
from dataclasses import dataclass

from fishfarm.models.vllm_model import VLLMModel


@dataclass
class TaskMetric:
    quality: float
    bc_ids: Tuple[int]


class BaseTask(object):

    def __init__(
        self,
        bc_num_dims: int,
        bc_min_vals: List[float],
        bc_max_vals: List[float],
        bc_grid_sizes: List[int],
    ) -> None:
        self.bc_num_dims = bc_num_dims
        self.bc_min_vals = bc_min_vals
        self.bc_max_vals = bc_max_vals
        self.bc_grid_sizes = bc_grid_sizes

    def _load_vllm(self, llm: LLM) -> VLLMModel:
        """Load llm into VLLMModel."""
        raise NotImplementedError()

    def get_q_and_bc(self, llm: LLM) -> TaskMetric:
        """Evaluate the LLM and return both quality and BC grid id."""
        raise NotImplementedError()

    def _get_bin_id(self, bc_idx: int, metric: float) -> int:
        bins = np.linspace(
            self.bc_min_vals[bc_idx],
            self.bc_max_vals[bc_idx],
            self.bc_grid_sizes[bc_idx] + 1,
        )
        return min(
            max(0, np.digitize(metric, bins, right=True) - 1),
            self.bc_grid_sizes[0] - 1,
        )
