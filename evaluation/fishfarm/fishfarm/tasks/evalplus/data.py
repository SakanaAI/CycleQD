# Based on: https://github.com/ise-uiuc/magicoder/blob/main/experiments/text2code.py

from dataclasses import dataclass

from evalplus.data import get_human_eval_plus, get_mbpp_plus


@dataclass
class TextToCodeProblem:
    id: str
    instruction: str
    response_prefix: str


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def read_mbpp_plus(
    plus_path: str, err_incomplete: bool = True, mini: bool = False
) -> dict[str, dict]:
    from evalplus.data.mbpp import completeness_check, mbpp_deserialize_inputs, stream_jsonl

    plus = {task["task_id"]: task for task in stream_jsonl(plus_path)}
    for task_id, task in plus.items():
        task["base_input"] = mbpp_deserialize_inputs(task_id, task["base_input"])
        task["plus_input"] = mbpp_deserialize_inputs(task_id, task["plus_input"])

    if err_incomplete:
        completeness_check("MBPP+", plus)
    return plus


def map_mbpp_problem(p: dict) -> TextToCodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction} Your code should satisfy the following assertion:
```python
{assertion}
```"""
    response_prefix = """```python"""
    return TextToCodeProblem(id=str(id), instruction=instruction, response_prefix=response_prefix)


def map_humaneval_problem(p: dict) -> TextToCodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
{prompt}"""
    return TextToCodeProblem(id=id, instruction=instruction, response_prefix=response_prefix)


def load_dataset(source_dataset: str) -> list[TextToCodeProblem]:
    if source_dataset not in ("humaneval", "mbpp"):
        raise ValueError(f"Unknown source_dataset: {source_dataset}")

    raw_problem_fn = {
        "humaneval": get_humaneval_raw_problems,
        "mbpp": get_mbpp_raw_problems,
    }[source_dataset]

    if source_dataset.startswith("humaneval"):
        map_problem_fn = map_humaneval_problem
    elif source_dataset.startswith("mbpp"):
        map_problem_fn = map_mbpp_problem
    else:
        raise ValueError(f"Unknown source_dataset: {source_dataset}")

    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    return problems
