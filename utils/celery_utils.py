import os
import socket
import sys
import traceback
from typing import Any, Literal, NoReturn, Type, Optional

import celery

Mode = Literal["solo", "worker", "main", "flower"]


class _TaskWrapper(celery.Task):

    def __init__(self) -> None:
        super().__init__()
        self.worker = None

    def maybe_init(self, worker_cls) -> None:
        if self.worker is None:
            self.worker = worker_cls()

    def call(self, method: str, *args, **kwargs) -> Any:
        return getattr(self.worker, method)(*args, **kwargs)

def set_mpi_env() -> None:
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

def _run_worker(celery_app: celery.Celery, loglevel: str = "INFO") -> NoReturn:
    # Generating a fancy worker name with the hostname and GPU number
    hostname = socket.gethostname()
    
    set_mpi_env()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    gpu = rank
    if gpu:
        hostname = f"{hostname}/{gpu}"

    celery_app.worker_main(
        [
            "worker",
            "--pool=solo",
            f"--loglevel={loglevel}",
            f"--hostname='{hostname}'",
        ]
    )
    sys.exit(0)


def _run_flower(celery_app: celery.Celery) -> NoReturn:
    import flower.app
    from flower.urls import settings
    from tornado.options import options

    flower_app = flower.app.Flower(capp=celery_app, options=options, **settings)
    flower_app.start()
    sys.exit(0)


def setup_celery(
    name: str,
    mode: Mode,
    worker_cls: Type,
    celery_broker: Optional[str] = None,
    celery_backend: Optional[str] = None,
) -> celery.Celery:
    """
    Set up a Celery application with specific configurations based on the mode.

    The "mode" parameter is a critical argument that determines the operation of this process.

    * When "mode" is set to "solo", everything is run within the current single process.
      It executes both the workloads of the worker process and the main process by itself.
      While it cannot be parallelized, it is easy to run and convenient for development.
      In this case, launching Celery's broker/backend is not necessary.

    * When "mode" is set to "main", it is configured to be able to remotely call workers.
      In the main mode, worker_cls is not instantiated.

    * When "mode" is set to "worker", this process will subsequently act as a worker,
      and the function does not return control. The worker instantiates worker_cls,
      listens to the queue, executes tasks specified by the main process, and returns the results.

    * When "mode" is set to "flower", it launches Flower, which is Celery's web monitoring tool.
      The process does not return control from this function.

    Parameters
    ----------
    name : str
        The name of the Celery application.
    mode : Literal["solo", "worker", "main", "flower"]
        The mode in which this process is to run: 'solo', 'worker', 'main', or 'flower'.
    worker_cls : Type
        The worker class to be remotely called.
    celery_broker: Optional[str]
        The Celery broker URL. If not specified, it will be read from the environment variable "CELERY_BROKER".
        If the environment variable is not set, the default value is "pyamqp://guest@localhost//".
    celery_backend: Optional[str]
        The Celery backend URL. If not specified, it will be read from the environment variable "CELERY_BACKEND".
        If the environment variable is not set, the default value is "rpc://".

    Returns
    -------
    celery.Celery
        The configured Celery application.
    """

    if mode == "solo":
        # In the solo mode, we do not need to need an external broker.
        broker = "memory://"
    else:
        broker = celery_broker or os.environ.get("CELERY_BROKER", "pyamqp://guest@localhost//")
    backend = celery_backend or os.environ.get("CELERY_BACKEND", "rpc://")

    app = celery.Celery(name, backend=backend, broker=broker)
    app.conf.broker_transport_options = {"visibility_timeout": 36000} #10h
    app.conf.update(
        task_serializer="pickle",
        result_serializer="pickle",
        accept_content=["pickle", "json"],
        # We want to decrease the prefetch multiplier, as our tasks are generally long-running.
        worker_prefetch_multiplier=1,
        worker_concurrency=1,
        # By setting ack_late to True and reject_on_worker_lost to True, we can ensure that
        # the task is not lost even if the worker is lost.
        ack_late=True,
        reject_on_worker_lost=True,
        broker_heartbeat= 700,
        task_default_retry_delay=0,  # デフォルトのリトライ間隔を0に設定
        task_max_retries=0,  
    )

    def call(self: _TaskWrapper, method: str, *args, **kwargs) -> Any:
        # print(method, args, kwargs)
        try:
            self.maybe_init(worker_cls)
            return self.call(method, *args, **kwargs)
        except Exception as exc:
            # We catch all exceptions, print them out to the stderr, and exit the process.
            #
            # Otherwise, celery will catch the exception and the worker process continues to run.
            # This is not desirable, as generally exceptions are due to
            # (1) GPU-related device errors, or (2) code bugs,
            # and we want to stop the process in either case.
            #
            # Please note that, when this worker process is exited, the task is re-enqueued
            # and will be executed by another worker process.
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()

            # We use os._exit because sys.exit is hooked by Celery and does not work as expected.
            os._exit(1)

    # Register the "call" method to the task
    app.task(base=_TaskWrapper, bind=True, name="call")(call)

    if mode == "solo":
        # By setting task_always_eager to True, we can run tasks in this process.
        app.conf.task_always_eager = True
    elif mode == "worker":
        _run_worker(celery_app=app)
    elif mode == "flower":
        _run_flower(celery_app=app)

    # Removing the previously enqueued tasks
    app.control.purge()

    return app