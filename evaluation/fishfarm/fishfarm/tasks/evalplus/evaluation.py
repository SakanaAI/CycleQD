# Based on:
# https://github.com/evalplus/evalplus/blob/1895d2f6aa8895044a7cf69defc24bd57695e885/evalplus/evaluate.py

import json
import multiprocessing
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from warnings import warn

import numpy as np
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.eval import SUCCESS, estimate_pass_at_k, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import Result, get_groundtruth
from termcolor import cprint
from tqdm.auto import tqdm

from ...logging import get_logger


logger = get_logger(__name__)


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: dict[str, Any],
    solution: str,
    expected_output: dict[str, list],
    base_only: bool = False,
    fast_check: bool = False,
    identifier: str = "HumanEval/0_0",
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> dict[str, Result]:
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
    return ret


def evaluate(
    source_dataset: str,
    output_path: str,
    base_only: bool = False,
    parallel: int = 0,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = 0.2,
    gt_time_limit_factor: float = 4.0,
    mini: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    if parallel == 0:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = parallel

    if os.path.isdir(output_path):
        result_path = os.path.join(output_path, "eval_results.json")
    else:
        assert output_path.endswith(".jsonl")
        result_path = output_path.replace(".jsonl", "_eval_results.json")

    if source_dataset == "humaneval":
        problems = get_human_eval_plus(mini=mini)
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash, [])
    elif source_dataset == "mbpp":
        problems = get_mbpp_plus(mini=mini)
        dataset_hash = get_mbpp_plus_hash()
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": dataset_hash,
        "eval": {},
    }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id: Counter[str] = Counter()
        n_samples = 0
        eval_results = defaultdict(list)
        remainings = set()
        sample_details = []

        logger.info("Reading samples...")
        for sample in tqdm(load_solutions(output_path)):
            task_id = sample["task_id"]
            explanation = sample.get("explanation", "")
            solution = (
                sample["solution"]
                if "solution" in sample
                else problems[task_id]["prompt"] + sample["completion"]
            )
            remainings.add(sample["_identifier"])

            args = (
                source_dataset,
                completion_id[task_id],
                problems[task_id],
                solution,
                expected_output[task_id],
                base_only,
                not test_details,  # fast_check
                sample["_identifier"],
                min_time_limit,
                gt_time_limit_factor,
            )

            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

            sample_details.append(
                dict(
                    task_id=task_id,
                    solution=solution,
                    explanation=explanation,
                    problems=problems[task_id],
                    expected_output=expected_output[task_id],
                )
            )

        assert n_samples == len(remainings), "Missing problems in unfinished"
        if len(completion_id) != len(problems):
            logger.warning("Warning: Missing problems in samples")

        def stucking_checker() -> None:
            while remainings:
                last_size = len(remainings)
                time.sleep(20)
                if last_size != len(remainings) or len(remainings) == 0:
                    continue
                # Potential stucking
                warn("No samples had finished testing in the last 20s")
                warn(f"{len(remainings)} samples to be tested: {remainings}")

        threading.Thread(target=stucking_checker).start()

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()
            remainings.remove(result["_identifier"])
            eval_results[result["task_id"]].append(result)

    # sort the results for each problem by completion_id
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        results["eval"][task_id] = {
            "nfiles": len(task_results),
            "base": [x["base"] for x in task_results],
            "plus": ([x["plus"] for x in task_results] if not base_only else []),
        }

    if os.path.isfile(result_path) and i_just_wanna_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            logger.info(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            logger.info(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)

    # Calculate pass@k.
    total = np.array([r["nfiles"] for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for key, res in results["eval"].items():
        elements = [element for element in sample_details if element["task_id"] == key]
        assert len(elements) == 1, f"Expected an element with task_id {key}, found {len(elements)}"
        element = elements[0]

        bc = sum([r[0] == SUCCESS for r in res["base"]])
        base_correct.append(bc)
        element["base_correct"] = bc
        if res["plus"]:
            new_bc = sum(
                [
                    res["plus"][i][0] == res["base"][i][0] == SUCCESS
                    for i in range(len(res["plus"]))
                ]
            )
            new_correct.append(new_bc)
            element["plus_correct"] = new_bc

    base_correct_array = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct_array, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }

    result = {f"{source_dataset}_base_{key}": value for key, value in pass_at_k.items()}
    cprint(f"{source_dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")

    if new_correct:
        cprint(f"{source_dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        result.update({f"{source_dataset}_plus_{key}": value for key, value in pass_at_k.items()})
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")

    return result, sample_details
