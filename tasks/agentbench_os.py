from typing import List
import vllm
import pickle

import fishfarm
from fishfarm.tasks.os_interaction import OSInteractionTask
from fishfarm.models.vllm_model import VLLMModel

from .base import BaseTask, TaskMetric


MAX_ROUND = 5
NUM_WORKERS = 32
TOTAL_NUM_TASKS = 170


class AgentBenchOSTask(BaseTask):

    def __init__(
        self,
        bc_num_dims: int,
        bc_min_vals: List[float],
        bc_max_vals: List[float],
        bc_grid_sizes: List[int],
    ) -> None:
        super().__init__(bc_num_dims, bc_min_vals, bc_max_vals, bc_grid_sizes)
        self.task_name = "agentbench_os"

        with open("evaluation/fishfarm/data/os_interaction/configs/samples.pkl", "rb") as f:
            os_samples = pickle.load(f)
        with open("evaluation/fishfarm/data/os_interaction/configs/os_data_config.pkl", "rb") as f:
            os_data_config = pickle.load(f)
        self._task = OSInteractionTask(
            samples=os_samples,
            data_config=os_data_config,
            workers=NUM_WORKERS,
            max_round=MAX_ROUND,
        )
        self._task_ids = [
            1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 15, 16, 17, 19, 21, 22,
            23, 24, 25, 26, 27, 28, 34, 35, 36, 39, 40, 42, 43, 44, 45,
            46, 49, 50, 51, 52, 54, 57, 60, 61, 63, 66, 75, 76, 77, 78,
            79, 83, 84, 88, 89, 90, 92, 93, 94, 96, 97, 99, 101, 102, 103,
            105, 108, 109, 111, 115, 116, 117, 118, 119, 123, 124, 128,
            129, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
            155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
            166, 167, 168, 169
        ]

    def _load_vllm(self, llm: vllm.LLM) -> VLLMModel:
        return VLLMModel(
            llm=llm,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=512,
            ),
            chat_template=fishfarm.chat_templates.LLAMA3,
        )

    def get_q_and_bc(self, llm: vllm.LLM) -> TaskMetric:
        model = self._load_vllm(llm)
        accuracy_list = []
        for _ in range(5):
            result = self._task.evaluate(model, sample_ids=self._task_ids)
            accuracy_list.append(result.aggregate_metrics["overall"]["acc"])
        mean_accuracy = sum(accuracy_list) / len(accuracy_list)
        q_val = (mean_accuracy) 
        assert self.bc_num_dims == 1
        bc_ids = (self._get_bin_id(0, q_val),)
        return TaskMetric(quality=q_val, bc_ids=bc_ids)