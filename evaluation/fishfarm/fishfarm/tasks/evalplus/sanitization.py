# Based on:
# https://github.com/evalplus/evalplus/blob/e9fedc83c161bbd655e145ac246580d2bd92f5f6/tools/sanitize.py
# https://github.com/evalplus/evalplus/blob/26fd3027d33c2f526ff0381ab9b2f45320153548/evalplus/sanitize.py

import ast
import os
import pathlib
import re
import traceback
from typing import Optional

from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
    load_solutions,
    write_directory,
    write_jsonl,
)
from tqdm.auto import tqdm

from ...logging import get_logger


logger = get_logger(__name__)


def syntax_check(code: str, verbose: bool = False) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def remove_unindented_lines(
    code: str, protect_before: str, execeptions: list[str], trim_tails: list[str]
) -> str:
    lines = code.splitlines()
    cut_idx = []
    cut_enabled = False
    for i, line in enumerate(lines):
        if not cut_enabled and line.startswith(protect_before):
            cut_enabled = True
            continue
        if line.strip() == "":
            continue
        if any(line.startswith(e) for e in execeptions):
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            cut_idx.append(i)

        if any(line.rstrip().startswith(t) for t in trim_tails):
            # cut off everything behind
            cut_idx.extend(list(range(i, len(lines))))
            break

    return "\n".join([line for i, line in enumerate(lines) if i not in cut_idx])


def to_four_space_indents(old_code: str) -> str:
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code


def sanitize_code(
    old_code: str, entry_point: str, rm_prefix_lines: Optional[str] = None, eofs: list = []
) -> str:
    new_code = old_code
    if rm_prefix_lines is not None:
        new_code = "\n".join(
            [line for line in old_code.splitlines() if not line.startswith(rm_prefix_lines)]
        )

    new_code = "\n" + new_code
    def_left = "def " + entry_point

    # basic handling of chat output
    new_code = new_code.replace("\n```python\n", "\n```\n")
    for chunk in new_code.split("\n```\n"):
        if def_left in chunk:
            new_code = chunk
            break

    chunks = [chunk for chunk in re.split(rf"{def_left}\s*\(", new_code)]
    # TODO: having return does not mean this is complete
    bodies = [chunk for chunk in chunks[1:] if "    return " in chunk.split("\ndef")[0]]
    def_left = def_left + "("
    new_code = def_left + def_left.join(bodies) if len(bodies) > 0 else ""  # fn + impl
    new_code = to_four_space_indents(new_code)

    for eof in eofs or []:
        new_code = new_code.split(eof)[0]

    # remove lines starting from the first unindented line after def_left
    new_code = remove_unindented_lines(
        new_code,
        protect_before=def_left,
        execeptions=["def ", "import ", "from "],
        trim_tails=['"""', "if", "print"],
    )
    new_code = chunks[0] + new_code

    # cut all functions that are not syntactically correct && not the entry point
    parts = new_code.split("\ndef ")
    includes = [parts[0]]
    for fn in new_code.split("\ndef ")[1:]:
        if (
            fn.strip().startswith(entry_point + " ")
            or fn.strip().startswith(entry_point + "(")
            or syntax_check("\ndef " + fn)
        ):
            includes.append(fn)
    new_code = "\ndef ".join(includes)
    return new_code.strip()


def sanitize(
    source_dataset: str,
    input_path: str,
    eofs: list = [],
    inplace: bool = False,
    rm_prefix_lines: Optional[str] = None,
    debug_task: Optional[str] = None,
) -> str:
    entry_point = {}

    if source_dataset == "humaneval":
        dataset = get_human_eval_plus()
    elif source_dataset == "mbpp":
        dataset = get_mbpp_plus()

    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(input_path)
    target_path = pathlib.Path(input_path)
    if not inplace:
        if is_folder:
            new_name = target_path.name + "-sanitized"
        else:
            new_name = target_path.name.replace(".jsonl", "-sanitized.jsonl")
        target_path = target_path.parent / new_name
    output_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    for solution in tqdm(load_solutions(input_path)):
        task_id = solution["task_id"]
        dbg_identifier = solution["_identifier"]
        if debug_task is not None and task_id != debug_task:
            continue

        ntotal += 1
        if "solution" in solution:
            old_code = solution["solution"]
        else:
            assert "completion" in solution
            old_code = dataset[task_id]["prompt"] + "\n" + solution["completion"]

        old_code = old_code.strip()

        new_code = sanitize_code(
            old_code=old_code,
            entry_point=entry_point[task_id],
            rm_prefix_lines=rm_prefix_lines,
            eofs=eofs,
        ).strip()

        # if changed, print the message
        if new_code != old_code:
            msg = "Sanitized: " + dbg_identifier
            if is_folder:
                msg += " -> " + dbg_identifier.replace(input_path, output_path)
            logger.info(msg)
            nsan += 1

        new_solutions.append(
            {
                "task_id": task_id,
                "solution": new_code,
                "explanation": solution["explanation"],
            }
        )

    if is_folder:
        write_directory(output_path, new_solutions)
    else:
        write_jsonl(output_path, new_solutions)

    logger.info(f"Sanitized {nsan} out of {ntotal} files.")

    return output_path
