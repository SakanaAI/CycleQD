import abc
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..models import Model


# TODO: dict? dataclass?
@dataclass
class TaskResult:

    aggregate_metrics: dict[str, float]
    sample_details: list[dict[str, Any]]


class Task(abc.ABC):

    @property
    @abc.abstractmethod
    def num_samples(self) -> int:
        raise NotImplementedError()

    # Use `model` as many times as you want. (e.g., for multi-turn tasks.)
    @abc.abstractmethod
    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        raise NotImplementedError()
