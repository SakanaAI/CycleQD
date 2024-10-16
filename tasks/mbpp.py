from typing import List
import numpy as np
import vllm

import fishfarm
from fishfarm.tasks.evalplus import EvalplusTask, load_dataset
from fishfarm.models.vllm_model import VLLMModel

from .base import BaseTask, TaskMetric


code_chat_templates = r"""
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}

{{ system_message }}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception(
            'Conversation roles must alternate user/assistant/user/assistant/...')}}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '@@ Instruction:\n' + message['content'].strip() + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '@@ Response:\n' + message['content'].strip() }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '@@ Response:' }}
    {% endif %}
{% endfor %}
""".replace(
    "    ", ""
).replace(
    "\n", ""
)


class MBPPTask(BaseTask):

    def __init__(
        self,
        bc_num_dims: int,
        bc_min_vals: List[float],
        bc_max_vals: List[float],
        bc_grid_sizes: List[int],
    ) -> None:
        super().__init__(bc_num_dims, bc_min_vals, bc_max_vals, bc_grid_sizes)
        self.task_name = "mbpp"
        system_message = (
            "You are an exceptionally intelligent coding assistant that " +
            "consistently delivers accurate and reliable responses to " +
            "user instructions."
        )
        samples = load_dataset(source_dataset=self.task_name)
        self._task = EvalplusTask(
            samples,
            context_messages=[fishfarm.Message("system", system_message),],
            source_dataset=self.task_name,
        )
        self._task_ids = list(range(len(samples)))

    def _load_vllm(self, llm: vllm.LLM) -> VLLMModel:
        # TODO: confirm if we need stop condition for this task.
        return VLLMModel(
            llm=llm,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=512,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=code_chat_templates,
        )

    def get_q_and_bc(self, llm: vllm.LLM) -> TaskMetric:
        model = self._load_vllm(llm)
        result = self._task.evaluate(model, sample_ids=self._task_ids)
        q_val = result.aggregate_metrics[f"{self.task_name}_base_pass@1"]

        # TODO: allow the user to provide a hook to separate BCs.
        chunk_size = len(result.sample_details) // self.bc_num_dims + 1
        bc_ids = ()
        for i in range(self.bc_num_dims):
            ss = i * chunk_size
            ee = ss + chunk_size
            bc_metric = np.mean(
                [x["base_correct"] for x in result.sample_details[ss:ee]]
            )
            bc_ids += (self._get_bin_id(i, bc_metric),)

        return TaskMetric(quality=q_val, bc_ids=bc_ids)