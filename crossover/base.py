import torch
import numpy as np
import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM


class BaseModelMerger(object):

    num_merge_params: int

    def _get_task_vector(self, base_param: Dict, ft_param: Dict) -> Dict:
        return {k: ft_param[k] - base_param[k] for k in base_param}

    def _merge(self,
               task_vectors: List[Dict],
               merge_params: np.ndarray) -> Dict:
        raise NotImplementedError

    def _generate_merge_params(self) -> np.ndarray:
        raise NotImplementedError

    def update_seed(self, seed: int):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def merge(self,
              base_param: Dict,
              model_paths: List[str],
              merge_params: np.ndarray=None) -> Dict:
        ft_params = [
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16).state_dict()
            for model_path in model_paths
        ]
        task_vectors = [
            self._get_task_vector(base_param, ft_param)
            for ft_param in ft_params
        ]
        if merge_params is None:
            merge_params = self._generate_merge_params()
        else:
            merge_params = np.array(merge_params)
        assert merge_params.size == self.num_merge_params, (
            f"{merge_params.size} vs {self.num_merge_params}")
        merged_param = self._merge(task_vectors, merge_params)
        return {k: base_param[k] + merged_param[k] for k in base_param}