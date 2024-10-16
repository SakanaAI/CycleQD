import copy
import json
import os
import re
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import yaml
from huggingface_hub import hf_hub_download

from ..imports import try_import
from ..logging import get_logger
from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


with try_import() as _imports:
    import docker

_imports.check()

logger = get_logger(__name__)

# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/os_interaction/task.py#L124C1-L162C2
ONE_SHOT = [
    {"role": "user", "content": 'tell me how many files are in the directory "/etc"?'},
    {
        "role": "assistant",
        "content": """Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```""",
    },
    {
        "role": "user",
        "content": (
            "The output of the OS:\ncpi "
            "cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols selinux tmpfiles.d "
            "[truncated because the output is too long]"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Think: The output has been truncated because it is too long, "
            "so I need to count files by script directly.\n\n"
            "Act: bash\n\n```bash\nls -1 /etc | wc -l\n```"
        ),
    },
    {"role": "user", "content": "The output of the OS:\n220"},
    {
        "role": "assistant",
        "content": """Think: Now I get the answer, it is 220.

Act: answer(220)""",
    },
]

# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/os_interaction/task.py#L395-L419
PROMPT = (
    "You are an assistant that will act like a person, "
    "I'will play the role of linux(ubuntu) operating system. "
    "Your goal is to implement the operations required by me or answer to the question "
    "proposed by me. For each of your turn, "
    "you should first think what you should do, "
    'and then take exact one of the three actions: "bash", "finish" or "answer". \n\n'
    "1. If you think you should execute some bash code, take bash action, "
    "and you should print like this:\n\nThink: put your thought here.\n\n"
    "Act: bash\n\n```bash\n# put your bash code here\n```\n\n"
    "2. If you think you have finished the task, take finish action, "
    "and you should print like this:\n\nThink: put your thought here.\n\nAct: finish\n\n"
    "3. If you think you have got the answer to the question, take answer action, "
    "and you should print like this:\n\nThink: put your thought here.\n\n"
    "Act: answer(Your answer to the question should be put in this pair of parentheses)\n\n"
    "If the output is too long, I will truncate it. "
    "The truncated output is not complete. "
    "You have to deal with the truncating problem by yourself. Attention, "
    "your bash code should not contain any input operation. Once again, "
    "you should take only exact one of the three actions in each turn.\n\n"
)


# TODO: move this to a more appropriate place
@dataclass
class OSInteractionSample:
    problem: list[dict]
    index: int
    config: Any


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/os_interaction/task.py#L106C1-L122C40
class JudgeConfig:
    image: Optional[str] = None
    init_script: Optional[list[tuple[str, str]]] = None
    start: Optional[tuple[str, str]] = None
    description: Optional[str] = None
    check: Optional[list[Any]] = None
    match: Optional[dict[Any, Any]] = None
    example_script: Optional[str] = None

    def get_evaluation_type(self) -> Optional[str]:
        if self.check:
            return "check"
        elif self.match:
            return "match"
        return None

    def get_evaluation_content(self) -> Optional[Union[list[Any], dict[Any, Any]]]:
        return self.check or self.match


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/os_interaction/task.py#L279C1-L304C10
def metric(data_config: dict[str, Any], prediction: list[dict[str, Any]]) -> dict[str, Any]:
    files = []
    for item in data_config["files"]:
        file = item["problem_file"]
        if file.endswith(".json") or file.endswith(".jsonl"):
            if file in data_config["ignore"]:
                continue
            file_configs = [config for config in prediction if (config and config["file"] == file)]
            logger.info(
                "File:",
                file,
                "Total:",
                len(file_configs),
                "Pass:",
                len(
                    [config for config in file_configs if "result" in config and config["result"]]
                ),
            )
            files.append(
                {
                    "file": file,
                    "total": float(len(file_configs)),
                    "pass": float(
                        len(
                            [
                                config
                                for config in file_configs
                                if "result" in config and config["result"]
                            ]
                        )
                    ),
                    "wrong": float(
                        len(
                            [
                                config
                                for config in file_configs
                                if "result" not in config or not config["result"]
                            ]
                        )
                    ),
                    "acc": (
                        float(
                            len(
                                [
                                    config
                                    for config in file_configs
                                    if "result" in config and config["result"]
                                ]
                            )
                        )
                        / float(len(file_configs))
                        if len(file_configs)
                        else 0.0
                    ),
                }
            )
    overall = {
        "total": float(len([config for config in prediction if config])),
        "pass": float(
            len(
                [
                    config
                    for config in prediction
                    if (config and "result" in config and config["result"])
                ]
            )
        ),
    }
    overall["wrong"] = overall["total"] - overall["pass"]
    overall["acc"] = overall["pass"] / overall["total"] if overall["total"] else 0.0
    return {
        "files": files,
        "overall": overall,
    }

# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/os_interaction/task.py#L319C5-L358C19
def extract_action(raw: str) -> dict:
    think_pattern = r"Think:\s*(.+)"
    act_pattern = r"Act:\s*(.+)"

    think = re.findall(think_pattern, raw)
    act = re.findall(act_pattern, raw)

    ret = {"thought": "\n".join(think), "action": None, "content": None}

    # reversly iterate over the action list
    for action in act[::-1]:
        if action.lower().startswith("bash"):
            ret["action"] = "bash"
            break
        if action.lower().startswith("finish"):
            ret["action"] = "commit"
            break
        if action.lower().startswith("answer"):
            content = action[6:].strip()
            left_par_pos = content.find("(")
            right_par_pos = content.rfind(")")
            if left_par_pos == -1 or right_par_pos == -1:
                continue
            content = content[left_par_pos + 1 : right_par_pos]
            ret["action"] = "commit"
            ret["content"] = content
            break

    if ret["action"] == "bash":
        # extract from ```bash to ```
        content_pattern = r"```bash\n(.*?)\n```"
        content = re.findall(content_pattern, raw, re.DOTALL)
        content = "\n\n".join(content)
        ret["content"] = content

    return ret


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/os_interaction/task.py#L23C1-L103C44
class Container:
    def __init__(self, image: str):
        self.image = image
        self.client = docker.from_env(timeout=180)
        self.container: docker.models.containers.Container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
            labels={"created_by": "os-pipeline"},
        )
        self.exec_id = self.client.api.exec_create(
            self.container.id, "bash --login", stdin=True, tty=True
        )["Id"]
        self.sock = self.client.api.exec_start(self.exec_id, socket=True)._sock
        self.sock.settimeout(5)
        # clear buffer
        self.sock.recv(1000)

    def cleanup(self) -> None:
        try:
            self.container.stop()
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def execute(self, command: Any) -> Any:
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code: int, o: bytes):
                self.output = o
                self.exit_code = code

        if not isinstance(command, str):
            return DummyOutput(-1, b"")
        self.sock.send(command.encode("utf-8") + b"\n")
        # ignore input line
        data = self.sock.recv(8)
        _, n = struct.unpack(">BxxxL", data)
        _ = self.sock.recv(n)
        output = b""
        while True:
            try:
                data = self.sock.recv(8)
                if not data:
                    break
                _, n = struct.unpack(">BxxxL", data)
                line = self.sock.recv(n)
                output += line
                if re.search(b"\x1b.+@.+[#|$] ", line):
                    break
            except TimeoutError:
                break
            except socket.timeout:
                break
        # replace the very end \x1b.+@.+[#|$] into nothing (required the suffix)
        return DummyOutput(0, output)

    def execute_independent(self, command: tuple[str, str], *params: Any) -> Any:
        language, cmd = command
        if language == "bash":
            exec_cmd = ["bash", "-c", cmd]
            if params:
                exec_cmd.append("--")
                exec_cmd.extend(params)
        elif language == "python":
            exec_cmd = ["python3", "-c", cmd, *params]
        elif language == "c++":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(cmd)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            exec_cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(cmd)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            exec_cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError("Unsupported language")
        return self.container.exec_run(exec_cmd)


class OSInteractionTask(Task):

    def __init__(
        self,
        samples: Sequence[OSInteractionSample],
        data_config: Any,
        max_round: int = 5,
        workers: int = 8,
        timeout: float = 5.0,
    ):
        self.samples = list(samples)
        self.data_config = data_config
        self.workers = workers
        self.max_round = max_round
        self.timeout = timeout

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def process_single_turn(
        self, container: Any, sample: OSInteractionSample, response: str
    ) -> tuple[str, bool]:
        try:
            sample.problem.append({"role": "assistant", "content": response})
            root = extract_action(response)
            if "action" not in root or root["action"] not in ["bash", "commit"]:
                answer = ""
                return answer, True

            action = root["action"]
            content = root["content"]
            if action == "commit":
                answer = content
                return answer, True

            elif action == "bash":
                result = container.execute(content).output.decode("utf-8")
                if len(result) > 800:
                    result = result[:780] + "\n[truncated because the output is too long]"
                sample.problem.append(
                    {
                        "role": "user",
                        "content": (
                            ("The output of the OS:\n\n" + result)
                            if result
                            else "The output of the OS is empty."
                        ),
                    }
                )
                answer = ""
                return answer, False
            else:
                answer = ""
                return answer, False

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            answer = ""
            return answer, True

    def get_judge(
        self, answer: str, config: Any, config_pair: dict, container: Any
    ) -> dict[str, Any]:
        if isinstance(answer, str) and config.match and config.match["strip"]:
            answer = answer.strip()
        if config.match:
            if "answer" in config.match:
                return {
                    "result": answer == config.match["answer"],
                    "error": None,
                    "file": config_pair["file"],
                    "index_in_file": config_pair["index"],
                }
            elif "regex" in config.match:
                return {
                    "result": re.search(config.match["regex"], answer) is not None,
                    "error": None,
                    "file": config_pair["file"],
                    "index_in_file": config_pair["index"],
                }
        elif config.check:
            params = [str(answer)]
            for script in config.check:
                if script is None:
                    script = config.example_script
                response = container.execute_independent(script, *params)
                if response.exit_code != 0:
                    return {
                        "result": False,
                        "error": None,
                        "file": config_pair["file"],
                        "index_in_file": config_pair["index"],
                    }
                params.append(response.output.decode("utf-8"))
            return {
                "result": True,
                "error": None,
                "file": config_pair["file"],
                "index_in_file": config_pair["index"],
            }
        return {
            "result": False,
            "error": "Invalid evaluation_type in config",
            "file": config_pair["file"],
            "index_in_file": config_pair["index"],
        }

    def init_container(self, sample: OSInteractionSample) -> Any:
        config_pair = sample.config
        config = config_pair["config"]
        container = Container(config.image)
        if config.init_script:
            for script in config.init_script:
                container.execute_independent(script)
        if config.start:
            container.execute(config.start[1])
        return container

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [copy.deepcopy(self.samples[sample_id]) for sample_id in sample_ids]
        sample_details = []
        final_answers_list = []
        cleanup_futures: list = []

        sample_chunks = [
            samples[i : i + self.workers] for i in range(0, len(samples), self.workers)
        ]

        for sample_chunk in sample_chunks:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_containers = []
                for sample in sample_chunk:
                    future = executor.submit(self.init_container, sample)
                    future_containers.append(future)
                    time.sleep(0.2)
                containers = []
                for future in future_containers:
                    container = future.result()
                    containers.append(container)

                unfinish_sample_chunk = sample_chunk
                unfinish_containers = containers

                rounds = 0

                while rounds < self.max_round and len(unfinish_sample_chunk) > 0:
                    next_unfinish_sample_chunk: list = []
                    next_unfinish_containers: list = []

                    requests = []
                    for sample in unfinish_sample_chunk:
                        messages = []
                        for message in sample.problem:
                            messages.append(
                                Message(role=message["role"], content=message["content"])
                            )
                        requests.append(GenerationRequest(messages=messages))
                    results = model.generate(requests)
                    tasks = [
                        (container, sample, result.generation)
                        for container, sample, result in zip(
                            unfinish_containers, unfinish_sample_chunk, results
                        )
                    ]
                    processed_results = []
                    futures = [executor.submit(self.process_single_turn, *task) for task in tasks]
                    for future in futures:
                        try:
                            result = future.result(timeout=self.timeout)
                            processed_results.append(result)
                        except TimeoutError:
                            processed_results.append(("", True))
                        except Exception:
                            processed_results.append(("", True))

                    for processed_result, sample, container in zip(
                        processed_results,
                        unfinish_sample_chunk,
                        unfinish_containers,
                    ):
                        if not processed_result[1]:
                            next_unfinish_sample_chunk.append(sample)
                            next_unfinish_containers.append(container)
                        else:
                            final_answer = self.get_judge(
                                processed_result[0],
                                sample.config["config"],
                                sample.config,
                                container,
                            )
                            final_answers_list.append((final_answer, sample.index))
                            sample_details.append(
                                dict(
                                    correct=final_answer["result"],
                                    index=sample.index,
                                    history=sample.problem,
                                    file=final_answer["file"],
                                    index_in_file=final_answer["index_in_file"],
                                )
                            )
                    containers_to_cleanup = [
                        container
                        for processed_result, sample, container in zip(
                            processed_results, unfinish_sample_chunk, unfinish_containers
                        )
                        if processed_result[1]
                    ]
                    if containers_to_cleanup:
                        future_tasks = (executor.submit(c.cleanup) for c in containers_to_cleanup)
                        cleanup_futures.extend(future_tasks)
                    unfinish_sample_chunk = next_unfinish_sample_chunk
                    unfinish_containers = next_unfinish_containers

                    rounds += 1
                for sample, container in zip(unfinish_sample_chunk, unfinish_containers):
                    final_answer = self.get_judge(
                        "", sample.config["config"], sample.config, container
                    )
                    final_answers_list.append((final_answer, sample.index))
                    sample_details.append(
                        dict(
                            correct=final_answer["result"],
                            index=sample.index,
                            history=sample.problem,
                            file=final_answer["file"],
                            index_in_file=final_answer["index_in_file"],
                        )
                    )
                if unfinish_containers:
                    future_tasks = (executor.submit(c.cleanup) for c in unfinish_containers)
                    cleanup_futures.extend(future_tasks)

                for future in as_completed(cleanup_futures):
                    future.result()

        sample_details.sort(key=lambda x: x["index"])
        final_answers_list.sort(key=lambda x: x[1])
        answers_only_list = [item[0] for item in final_answers_list]
        aggregate_metrics = metric(self.data_config, answers_only_list)
        return TaskResult(aggregate_metrics=aggregate_metrics, sample_details=sample_details)
