from typing import Tuple, Dict
from dataclasses import dataclass

from tasks.base import TaskMetric


@dataclass
class ArchiveData:
    quality: float                      # Model performance on the task
    model_path: str                     # Path to the model which was evaluated
    sampling_freq: int                  # Number of times the model was sampled
    validation_quality: float           # Model performance on the validation set


@dataclass
class ModelEvalResult:
    model_path: str                     # Path to the model which was evaluated
    task_metrics: Dict[str, TaskMetric] # Task metrics


@dataclass
class QDInfo:
    task_name: str                      # Name of the task
    quality: float                      # Model performance on the task
    bc_ids: Tuple[int]                  # Model behavior characterization

@dataclass
class MergeResult:
    qd_info: Dict[str, QDInfo]          # QD information
    save_path: str                      # Path to the saved model