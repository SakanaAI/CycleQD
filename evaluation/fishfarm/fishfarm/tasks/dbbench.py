import copy
import logging
import os
import random
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..imports import try_import
from ..logging import get_logger
from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


with try_import() as _imports:
    import docker
    import mysql.connector
    from docker.models import containers

_imports.check()

logger = get_logger(__name__)


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/dbbench/Interaction.py#L12-L154
class Container:
    port = 13000

    def __init__(
        self,
        volume: Optional[str] = None,
        init_file: Optional[str] = None,
        image: Optional[str] = "mysql",
    ):
        self.deleted = False
        self.image = image
        self.client = docker.from_env(timeout=180)
        self.container: containers.Container = None
        password = "password"
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            # When launched with mpirun, assign ports based on OMPI_COMM_WORLD_LOCAL_RANK.
            # To avoid collisions within the node
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
            p = Container.port + local_rank
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            # When evaluating models that cannot be run on a single GPU,
            # Use the first index from the CUDA_VISIBLE_DEVICES environment variable.
            cuda_first_device_number = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
            p = Container.port + cuda_first_device_number
        else:
            # For other cases, such as evaluating API model, using random ports is sufficient.
            p = Container.port + random.randint(0, 10000)

        # If a port is in use, increment by 8 to prevent conflicts.
        # The increment by 8 is used because a maximum of 8 GPUs can be present in one node.
        # OMPI_COMM_WORLD_LOCAL_RANK and CUDA_VISIBLE_DEVICES range from 0 to 7.
        # This increment ensures that there are no overlaps in port assignments.
        while self.is_port_open(p):
            p += 8
        self.port = p
        if volume:
            self.container = self.client.containers.run(
                image,
                name=f"mysql_{self.port}",
                environment={
                    "MYSQL_ROOT_PASSWORD": password,
                },
                ports={"3306": self.port},
                volumes={volume: {"bind": "/var/lib/mysql", "mode": "rw"}},
                detach=True,
                tty=True,
                stdin_open=True,
                remove=True,
            )

        else:
            self.container = self.client.containers.run(
                image,
                name=f"mysql_{self.port}",
                environment={
                    "MYSQL_ROOT_PASSWORD": password,
                },
                ports={"3306": self.port},
                detach=True,
                tty=True,
                stdin_open=True,
                remove=True,
            )

        # The underlying issue of connections failing with Failed getting connection;
        # pool exhausted has not been resolved fundamentally.
        # The problem stems from the pool not closing properly.
        # Issues with the execute function implementation. (Most like)
        # However, changing the DBtask implementation carries high risks.
        # So the approach is to alter Container.port incrementally by 8 each time.
        # This method is known to be effective as long as the same port is not continuously reused.
        Container.port += 8

        # Due to potential delays in container startup, set time.sleep(5)
        # Though 1 second may suffice
        time.sleep(5)

        retry = 0
        while True:
            try:
                # Increase the pool_size from the default 5 to 32, which is the maximum value.
                # This can prevent crashes due to the 'pool exhausted' error.
                # https://github.com/mysql/mysql-connector-python/blob/dc71cebe53615110ff00dbb8b629f5457ece1ddb/mysql-connector-python/lib/mysql/connector/pooling.py#L69.
                self.conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password="password",
                    port=self.port,
                    pool_reset_session=True,
                    pool_size=32,
                )
            except mysql.connector.errors.OperationalError:
                time.sleep(1)
            except mysql.connector.InterfaceError:
                if retry > 10:
                    raise
                time.sleep(5)
            except mysql.connector.errors.DatabaseError:
                # Include except mysql.connector.errors.DatabaseError: as it enhances robustness
                # Implement retry counts to avoid masking underlying issues.
                if retry > 10:
                    raise
                time.sleep(5)
            else:
                break
            retry += 1

        if init_file:
            with open(init_file) as f:
                data = f.read()
            for sql in data.split("\n\n"):
                try:
                    self.execute(sql, verbose=False)
                except Exception as e:
                    logging.exception("An error occurred: %s", str(e))
                    raise

    def delete(self) -> None:
        # The delete function ideally should not need a try-except.
        # It should always succeed in deletion.
        # It is included to ensure robustness and prevent undesirable crashes during operations.
        # Although there have been no incidents requiring this safeguard to date.
        try:
            self.container.stop()
            self.deleted = True
        except Exception as e:
            logging.exception("An error occurred: %s", str(e))

    def __del__(self) -> None:
        try:
            if not self.deleted:
                self.delete()
        except Exception as e:
            logging.exception("An error occurred: %s", str(e))

    def execute(
        self,
        sql: str,
        database: Optional[str] = None,
        truncate: bool = True,
        verbose: bool = True,
        no_except: bool = False,
    ) -> Optional[str]:
        if verbose:
            if len(sql) < 300:
                pass
        self.conn.reconnect()
        try:
            with self.conn.cursor() as cursor:
                if database:
                    cursor.execute(f"use `{database}`;")
                    cursor.fetchall()
                cursor.execute(sql, multi=True)
                result = cursor.fetchall()
                result = str(result)
            self.conn.commit()
        except Exception as e:
            if no_except:
                raise
            result = str(e)
        if verbose:
            if len(result) < 200:
                pass
            else:
                pass
        if len(result) > 800 and truncate:
            result = result[:800] + "[TRUNCATED]"
        if not sql.lower().startswith("select"):
            pass
            # IMPORTANT: if `execute` is called in a high rate, here must wait for the transaction
        return result

    def is_port_open(self, port: int) -> bool:
        try:
            self.client.containers.get(f"mysql_{port}")
            return True
        except Exception:
            pass
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Try to connect to the specified port
            sock.connect(("localhost", port))
            # If the connection succeeds, the port is occupied
            return True
        except ConnectionRefusedError:
            # If the connection is refused, the port is not occupied
            return False
        finally:
            # Close the socket
            sock.close()


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/dbbench/__init__.py#L176-L238
def metrics(result_entries: list[dict[str, Any]], targets: list[str]) -> dict[str, Any]:
    def factory(typ: str) -> Callable[[list[dict[str, Any]], list[str]], dict[str, Any]]:
        def acc(inp: list[dict[str, Any]], tar: list[str]) -> dict[str, Any]:
            sample_results = []
            correct = 0
            total = 0
            for raw, cor in zip(inp, tar):
                if not raw:
                    continue
                ans, t, index = raw["answer"], raw["type"], raw["index"]
                if t != typ and not (typ == "SELECT" and t not in ("INSERT", "UPDATE")):
                    continue
                if t in ("INSERT", "DELETE", "UPDATE"):
                    is_correct = ans == cor
                    correct += is_correct
                    sample_results.append({"correct": is_correct, "index": int(index)})
                else:
                    try:
                        ans = list(eval(ans))
                    except Exception:
                        ans = [ans]
                    if len(ans) == 1 and len(cor) == 1:
                        try:
                            is_correct = float(ans[0]) == float(cor[0])
                        except (ValueError, TypeError):
                            is_correct = ans[0] == cor[0]
                    else:
                        try:
                            cor_set = set(cor)
                            ans_set = set(ans)
                            is_correct = ans_set == cor_set
                        except TypeError as e:
                            logging.exception(
                                "An error occurred while comparing answers: %s", str(e)
                            )
                            is_correct = False
                    correct += is_correct
                    sample_results.append({"correct": is_correct, "index": int(index)})
                total += 1
            if total == 0:
                logger.warning(f"WARNING: {typ} does not exist!")
                return {"accuracy": 0, "sample_results": []}
            return {"accuracy": correct / total, "sample_results": sample_results}

        return acc

    types = [
        "other",
        "counting",
        "comparison",
        "ranking",
        "aggregation-SUM",
        "aggregation-MIN",
        "aggregation-MAX",
        "aggregation-AVG",
        "SELECT",
        "INSERT",
        "UPDATE",
    ]

    ret = {}
    sample_correct_results = []
    overall_results = []
    for typ in types:
        acc_function = factory(typ)
        results = acc_function(result_entries, targets)
        ret[typ + "_accuracy"] = results["accuracy"]
        sample_correct_results.extend(results["sample_results"])
        if typ in ["SELECT", "INSERT", "UPDATE"]:
            overall_results.extend(results["sample_results"])
    ret["overall_cat_accuracy"] = (
        sum(result["correct"] for result in overall_results) / len(overall_results)
        if overall_results
        else 0
    )

    def average_round(inp: list[dict[str, Any]], tar: list[str]) -> float:
        count = 0
        total = 0
        for raw, cor in zip(inp, tar):
            if not raw:
                continue
            count += len(raw["history"])
            total += 1
        if total:
            return count / total
        else:
            return 0.0

    ret["average_round"] = average_round(result_entries, targets)

    return {
        "aggregate_metrics": ret,
        "sample_correct_results": sample_correct_results,
    }


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/dbbench/__init__.py#L55-L58
def escape(string: str, conn: Any) -> Any:
    if type(string) is not str:
        string = str(string)
    return conn._cmysql.escape_string(string).decode("utf-8")


# https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/dbbench/__init__.py#L32-L52
def build_sql(raw: dict[str, Any], conn: Any) -> str:
    name = raw["table"]["table_name"]
    columns = ",".join(
        [
            f"`{escape(column['name'], conn)}` TEXT"
            for column in raw["table"]["table_info"]["columns"]
        ]
    )
    column_names = ",".join(
        [f"`{escape(column['name'], conn)}`" for column in raw["table"]["table_info"]["columns"]]
    )
    items = []
    for row in raw["table"]["table_info"]["rows"]:
        item = "("
        for col in row:
            item += f"'{escape(col, conn)}',"
        item = item[:-1] + ")"
        items.append(item)
    items_str = ",".join(items)
    sql = f"""CREATE DATABASE IF NOT EXISTS `{name}`;
    USE `{name}`;
    CREATE TABLE IF NOT EXISTS `{name}` ({columns});
    INSERT INTO `{name}` ({column_names}) VALUES {items_str};
    COMMIT;
    """
    return sql


# TODO: move this to a more appropriate place
@dataclass
class DBBenchSample:
    problem: list[dict]
    answer: str
    index: int
    raw: dict[str, Any]


class DBBenchTask(Task):

    def __init__(
        self,
        samples: Sequence[DBBenchSample],
        max_round: int = 5,
    ):
        self.samples = list(samples)
        self.max_round = max_round

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    # https://github.com/THUDM/AgentTuning/blob/e33a45d7eab2b63cac4d1956da1e6377fca9fcc7/AgentBench.old/src/tasks/dbbench/__init__.py#L79-L114
    def process_response_and_check_completion(
        self,
        sample: DBBenchSample,
        model: Model,
        response: str,
        raw: dict[str, Any],
        container: Container,
        rounds: int,
        db: str,
    ) -> tuple[str, bool]:
        answer = ""
        action = re.search(r"Action: (.*?)\n", response)
        if action and action.group(1) == "Operation" and rounds < self.max_round:
            response_match = re.search(r"```sql\n([\s\S]*?)\n```", response)
            if not response_match:
                answer = ""
            else:
                sql = response_match.group(1).strip()
                sql = sql.replace("\n", " ")
                execution_result = container.execute(sql, db)
                sample.problem.append(
                    {"role": "user", "content": execution_result if execution_result else ""}
                )
                return "", False
        if response:
            answer_match = re.search(r"\nFinal Answer:(.*)", response)
            if answer_match:
                answer = answer_match.group(1)
            else:
                answer = ""

        if raw["type"][0] in ("INSERT", "DELETE", "UPDATE"):
            columns = ",".join(
                [
                    f"`{escape(column['name'], container.conn)}`"
                    for column in raw["table"]["table_info"]["columns"]
                ]
            )
            md5_query = (
                f"SELECT MD5(GROUP_CONCAT(rowhash ORDER BY rowhash)) AS hash "
                f"FROM ("
                f"    SELECT SUBSTRING(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash "
                f"    FROM `{db}`"
                f") AS sub;"
            )
            answer = container.execute(md5_query, db) or ""
        container.execute(f"drop database `{db}`")
        return answer, True

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        sample_details: list[dict[str, Any]] = []
        targets = []
        result_entries = []

        container = Container()
        for sample in samples:
            init = build_sql(sample.raw, container.conn)
            container.execute(init)

        unfinished_samples = [copy.deepcopy(sample) for sample in samples]
        rounds = 0
        while unfinished_samples and rounds <= self.max_round:
            requests = []
            for sample in unfinished_samples:
                messages = []
                for message in sample.problem:
                    messages.append(Message(role=message["role"], content=message["content"]))
                requests.append(GenerationRequest(messages=messages))

            results = model.generate(requests)
            next_round_samples = []

            for sample, result in zip(unfinished_samples, results):
                response = result.generation
                sample.problem.append({"role": "assistant", "content": response})

                answer, finished = self.process_response_and_check_completion(
                    sample=sample,
                    model=model,
                    response=response,
                    raw=sample.raw,
                    container=container,
                    rounds=rounds,
                    db=sample.raw.get("table", {}).get("table_name", ""),
                )

                if finished:
                    result_entries.append(
                        {
                            "answer": str(answer),
                            "type": sample.raw["type"][0],
                            "history": sample.problem,
                            "index": sample.index,
                        }
                    )
                    targets.append(sample.answer)
                    sample_details.append(
                        dict(
                            answer=sample.answer,
                            prediction=str(answer),
                            index=int(sample.index),
                            history=sample.problem,
                        )
                    )
                else:
                    next_round_samples.append(sample)

            unfinished_samples = next_round_samples
            rounds += 1
        container.delete()

        evaluation_metrics = metrics(result_entries, targets)
        sample_details.sort(key=lambda x: x["index"])
        for sample_detail in sample_details:
            sample_detail["correct"] = False
            match = next(
                (
                    item
                    for item in evaluation_metrics["sample_correct_results"]
                    if item["index"] == sample_detail["index"]
                ),
                None,
            )
            if match:
                sample_detail["correct"] = match["correct"]

        return TaskResult(
            aggregate_metrics=evaluation_metrics["aggregate_metrics"],
            sample_details=sample_details,
        )
