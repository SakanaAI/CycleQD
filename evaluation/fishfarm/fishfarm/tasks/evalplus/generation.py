# Based on: https://github.com/ise-uiuc/magicoder/blob/main/experiments/text2code.py

import itertools
from pathlib import Path
from typing import Iterable, List, Sequence, TypeVar

from evalplus.data import write_jsonl
from tqdm.auto import tqdm

from ...models import GenerationRequest, Message, Model
from .data import TextToCodeProblem


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def generate(
    model: Model,
    problems: list[TextToCodeProblem],
    context_messages: Sequence[Message],
    output_path: str,
    n_batches: int = 1,
    n_problems_per_batch: int = 1_000_000_000,
    # Set n_problems_per_batch to a large number to run the loop once:
    # In fishfarm, batching is handled by the Model to allow libraries like vLLM to optimize it.
    # Sending multiple small batches can degrade vLLM's performance.
    n_samples_per_problem: int = 1,
) -> List[str]:
    problems_chunked = list(chunked(list(problems), n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(n_batches))
    n_total = len(problems_chunked) * n_batches

    Path(output_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem.id for problem in problems]
        all_task_ids = task_ids * n_samples_per_problem

        requests = []
        for problem in problems:
            messages = list(context_messages)
            messages.append(Message(role="user", content=problem.instruction))
            messages.append(Message(role="assistant_prefill", content=problem.response_prefix))
            requests.append(GenerationRequest(messages=messages))
        completes = model.generate(requests)
        completions = [c.generation for c in completes]

        assert len(problems) <= n_problems_per_batch
        assert len(completions) == len(problems) * n_samples_per_problem

        samples = []
        for task_id, completion in zip(all_task_ids, completions):
            completion_body = completion[
                : (index if (index := completion.find("```")) != -1 else len(completion))
            ]
            explanation = completion[
                (index if (index := completion.find("```") + 3) != -1 else len(completion)) :
            ].strip()

            samples.append(
                dict(
                    task_id=task_id,
                    completion=completion_body,
                    explanation=explanation,
                )
            )

        write_jsonl(output_path, samples, append=True)
    return completions
