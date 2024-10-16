import tempfile
from typing import Literal, Optional, Sequence

from ...models import Message, Model
from ..base import Task, TaskResult
from . import evaluation, generation, sanitization
from .data import TextToCodeProblem


class EvalplusTask(Task):

    def __init__(
        self,
        samples: Sequence[TextToCodeProblem],
        context_messages: Sequence[Message] = (),
        source_dataset: Literal["humaneval", "mbpp"] = "humaneval",
    ):
        self.samples = list(samples)
        self.context_messages = context_messages
        self.source_dataset = source_dataset
        if source_dataset not in ("humaneval", "mbpp"):
            raise ValueError(f"Unknown source_dataset: {source_dataset}")

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        with tempfile.TemporaryDirectory() as save_dir:
            output_path = f"{save_dir}/outputs.jsonl"

            completions = generation.generate(model, samples, self.context_messages, output_path)

            if self.source_dataset == "mbpp":
                output_path = sanitization.sanitize(self.source_dataset, output_path)

            result, sample_details = evaluation.evaluate(self.source_dataset, output_path)

            for i, completion in enumerate(completions):
                sample_details[i]["output"] = completion

        return TaskResult(aggregate_metrics=result, sample_details=sample_details)
