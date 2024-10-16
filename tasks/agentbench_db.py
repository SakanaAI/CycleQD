from typing import List
import datasets
import vllm
import os
import json
from huggingface_hub import hf_hub_download

import fishfarm
from fishfarm.tasks.dbbench import DBBenchSample, DBBenchTask
from fishfarm.models.vllm_model import VLLMModel

from .base import BaseTask, TaskMetric


MAX_ROUND = 5
db_prompt_template = (
    "\n"
    "I will ask you a question, "
    "then you should help me operate a MySQL database with SQL to answer the question.\n"
    "You have to explain the problem and your solution to me and write down your thoughts.\n"
    "After thinking and explaining thoroughly, "
    "every round you can choose to operate or to answer.\n"
    "your operation should be like this:\n"
    "Action: Operation\n"
    "```sql\n"
    "SELECT * FROM table WHERE condition;\n"
    "```\n"
    "You MUST put SQL in markdown format without any other comments. "
    "Your SQL should be in one line.\n"
    "Every time you can only execute one SQL statement. "
    "I will only execute the statement in the first SQL code block. "
    "Every time you write a SQL, I will execute it for you and give you the output.\n"
    "If you are done operating, and you want to commit your final answer, then write down:\n"
    "Action: Answer\n"
    'Final Answer: ["ANSWER1", "ANSWER2", ...]\n'
    "DO NOT write this pattern unless you are sure about your answer. "
    "I expect an accurate and correct answer.\n"
    "Your answer should be accurate. Your answer must be exactly the same as the correct answer.\n"
    "If the question is about modifying the database, "
    "then after done operation, your answer field can be anything.\n"
    "If your response cannot match any pattern I mentioned earlier, "
    "you will be judged as FAIL immediately.\n"
    "Your input will be raw MySQL response, you have to deal with it by yourself.\n"
)


class AgentBenchDBTask(BaseTask):

    def __init__(
        self,
        bc_num_dims: int,
        bc_min_vals: List[float],
        bc_max_vals: List[float],
        bc_grid_sizes: List[int],
    ) -> None:
        super().__init__(bc_num_dims, bc_min_vals, bc_max_vals, bc_grid_sizes)
        self.task_name = "agentbench_db"

        filepath = 'evaluation/fishfarm/data/db/data_dbbench_standard_dev.jsonl'
        db_dataset_list = []
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                db_dataset_list.append(data)

        db_samples = []
        for index, raw in enumerate(db_dataset_list):
            messages = [
                {"role": "user", "content": db_prompt_template},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": f"{raw['description']}\n{raw['add_description']}"},
            ]
            if raw["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                answer = raw.pop("answer_md5")
            else:
                answer = raw.pop("label")

            sample = DBBenchSample(messages, answer, index, raw)
            db_samples.append(sample)
        self._task = DBBenchTask(samples=db_samples, max_round=MAX_ROUND)
        self._task_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 262, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, 294, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 342, 343, 344, 345, 346, 347, 349, 350, 351, 352, 353, 354, 355, 356, 357, 359]

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
        result = self._task.evaluate(model, sample_ids=self._task_ids)
        q_val = result.aggregate_metrics["overall_cat_accuracy"]
        assert self.bc_num_dims == 1
        bc_ids = (self._get_bin_id(0, q_val),)
        return TaskMetric(quality=q_val, bc_ids=bc_ids)