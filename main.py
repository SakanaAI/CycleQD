import os
import cma
import hydra
import logging
import wandb
import torch
import time
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from celery import Celery
from collections import deque
from vllm import LLM
from transformers import AutoModelForCausalLM
from collections import defaultdict

from tasks.base import BaseTask
from utils.celery_utils import setup_celery
from utils.helpers import (
    load_hf_params_to_vllm, save_archive_map,
    delete_outdated_models, plot_elite_map,
    get_largest_gen_file, load_archive_map
)
from datatypes import (
    ArchiveData, ModelEvalResult, MergeResult, QDInfo, TaskMetric
)


def load_task_configs(cfg: DictConfig) -> List[DictConfig]:
    hydra_base_dir = hydra.utils.get_original_cwd()
    task_configs = {}
    for task_name in cfg.tasks:
        task_config_path = os.path.join(
            hydra_base_dir, "configs", "task", f"{task_name}.yaml")
        task_config = OmegaConf.load(task_config_path)
        task_configs[task_name] = task_config
    return task_configs

class Worker(object):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("Worker")
        self.tasks = [hydra.utils.instantiate(x) for x in load_task_configs(cfg).values()]

        self.llm = LLM(
            cfg.base_model_path,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            enforce_eager=True,
        )
        m = self.llm.llm_engine.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False

        self.qd_sampler = hydra.utils.instantiate(cfg.qd.sampling)
        self.crossover = hydra.utils.instantiate(cfg.qd.crossover)
        self.mutator = hydra.utils.instantiate(cfg.qd.mutation)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path, torch_dtype=torch.bfloat16)
        self.base_params = self.hf_model.state_dict()

        # Each task has its own CMA-ES solver.
        self.use_cma = cfg.qd.use_cma
        self.cma_popsize = cfg.cma_es.popsize
        if self.use_cma:
            self.cma_solvers = {}
            self.cma_input_grid_sizes = {}
            for task in self.tasks:
                input_size, grid_sizes = self._get_input_info(task, self.tasks)
                self.cma_input_grid_sizes[task.task_name] = grid_sizes
                self.logger.info(
                    f"Init CMA for {task.task_name}: input_size={input_size}")
                init_params = np.zeros(
                    input_size * (self.crossover.num_merge_params +
                                  self.mutator.num_mutation_params)
                )
                self.cma_solvers[task.task_name] = cma.CMAEvolutionStrategy(
                    x0=init_params,
                    sigma0=cfg.cma_es.sigma,
                    inopts={
                        'popsize': cfg.cma_es.popsize,
                        'seed': cfg.cma_es.seed if cfg.cma_es.seed > 0 else 42,
                        'randn': np.random.randn,
                    },
                )
            self.cma_gen_counts = {task.task_name: 0 for task in self.tasks}
            self.cma_gen_params = {task.task_name: None for task in self.tasks}
            self.cma_fitnesses = {task.task_name: [] for task in self.tasks}

    def merge_models(
        self,
        q_name: str,
        generation: int,
        model_dir: str,
        archive_map: Dict[str, Dict[Tuple[int], ArchiveData]]) -> MergeResult:

        # setup
        save_path=f"{model_dir}/gen_{generation}"
        self.qd_sampler.update_seed(generation)
        self.crossover.update_seed(generation)
        self.mutator.update_seed(generation)

        while True:
            trial = 0
            try:
                # Sample parents
                parent1, parent2 = self.qd_sampler.sample(archive_map[q_name])

                # Get params from cma.
                if self.use_cma:
                    if self.cma_gen_counts[q_name] == 0:
                        self.cma_gen_params[q_name] = self.cma_solvers[q_name].ask()
                    input_data = (
                        np.array(parent1 + parent2) /
                        np.tile(self.cma_input_grid_sizes[q_name], 2)
                    )
                    ix = self.cma_gen_counts[q_name]
                    params = np.dot(
                        input_data,
                        self.cma_gen_params[q_name][ix].reshape((input_data.size, -1))
                    )
                else:
                    params = None

                # Merge models.
                self.logger.info(f"Local Rank {os.environ['RANK']}. Merging models ...")
                archive = archive_map[q_name]
                child_param = self.crossover.merge(
                    self.base_params,
                    [archive[parent1].model_path, archive[parent2].model_path],
                    params[:self.crossover.num_merge_params] if self.use_cma else None,
                )
                self.logger.info(f"Local Rank {os.environ['RANK']}. Merge Success! Trial {trial}. parent 1: {archive[parent1].model_path}, parent 2: {archive[parent2].model_path}")
                break
            except:
                self.logger.error(f"Local Rank {os.environ['RANK']}. Merge Failed: Trial {trial}. parent 1: {archive[parent1].model_path}, parent 2: {archive[parent2].model_path}")
                trial += 1

        self.logger.info(f"Local Rank {os.environ['RANK']}. Mutating models ...")
        child_param = self.mutator.mutate(
            child_param,
            q_name,
            params[self.crossover.num_merge_params:] if self.use_cma else None,
        )
        self.logger.info(f"Local Rank {os.environ['RANK']}. Evaluating models ...")
        result = self._eval_model(child_param)

        # Parse the result.
        to_save = False
        qd_info = {}
        for k in archive_map:
            q_val = result.task_metrics[k].quality
            bc_ids = self._get_bc_ids(k, result.task_metrics)
            qd_info[k] = QDInfo(task_name=k, quality=q_val, bc_ids=bc_ids)
            if (
                bc_ids not in archive_map[k] or
                archive_map[k][bc_ids].quality < q_val
            ):
                to_save = True
        self.logger.info(f"Local Rank {os.environ['RANK']}. qd_info={qd_info}")

        # Write to disk only if the archive is updated.
        if to_save:
            self.hf_model.load_state_dict(child_param)
            try:
                self.hf_model.save_pretrained(save_path, safe_serialize=True)
            except:
                self.logger.error(f"Local Rank {os.environ['RANK']}. error save path {save_path}")
        else:
            save_path = None

        # Update CMA.
        if self.use_cma:
            self.cma_gen_counts[q_name] += 1
            # Calculate fitness (inverse the fitness because cma minimizes).
            fitness = (
                np.sum(np.array(qd_info[q_name].bc_ids) /
                       self.cma_input_grid_sizes[q_name]) +
                qd_info[q_name].quality
            )
            assert 0 <= fitness <= len(self.tasks)
            self.cma_fitnesses[q_name].append(-fitness)
            # Update CMA if the population's fitness have been collected.
            if self.cma_gen_counts[q_name] == self.cma_popsize:
                assert len(self.cma_fitnesses[q_name]) == self.cma_popsize
                self.cma_solvers[q_name].tell(
                    self.cma_gen_params[q_name], self.cma_fitnesses[q_name])
                self.cma_gen_counts[q_name] = 0
                self.cma_fitnesses[q_name] = []
                self.logger.info(f"Local Rank {os.environ['RANK']}. Updated CMA for {q_name}.")

        return MergeResult(
            qd_info=qd_info,
            save_path=save_path,
        )

    def _get_bc_ids(self,
                    q_name: str,
                    task_metrics: Dict[str, TaskMetric]) -> Tuple[int]:
        bc_ids = ()
        for k in task_metrics:
            if k != q_name:
                bc_ids += task_metrics[k].bc_ids
        return bc_ids

    def _get_input_info(self,
                        task: BaseTask,
                        tasks: List[BaseTask]) -> Tuple[int, List]:
        target_task_name = task.task_name
        input_size = 0
        input_grid_sizes = []
        for t in tasks:
            if t.task_name != target_task_name:
                input_size += t.bc_num_dims
                input_grid_sizes.extend(t.bc_grid_sizes)
        return input_size * 2, input_grid_sizes

    def _eval_model(self, param: Dict, task_name: Optional[str] = None) -> ModelEvalResult:
        load_hf_params_to_vllm(param, self.llm)
        if task_name:
            task_metrics = {
                task.task_name: task.get_q_and_bc(self.llm) 
                for task in self.tasks if task.task_name == task_name
            }
        else:
            task_metrics = {
                task.task_name: task.get_q_and_bc(self.llm) for task in self.tasks
            }
        self.logger.info(f"Local Rank {os.environ['RANK']}. task_metrics={task_metrics}")
        return ModelEvalResult(
            model_path=None,
            task_metrics=task_metrics,
        )

    def eval_model(self, model_path: str, task_name: Optional[str] = None) -> MergeResult:
        model_param = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).state_dict()
        self.hf_model.load_state_dict(model_param)
        result = self._eval_model(model_param, task_name)
        if task_name:
            qd_info = {
                task_name: QDInfo(
                    task_name=task_name,
                    quality=result.task_metrics[task_name].quality,
                    bc_ids=self._get_bc_ids(task_name, result.task_metrics),
                )
            }
        else:
            qd_info = {
                task.task_name: QDInfo(
                    task_name=task.task_name,
                    quality=result.task_metrics[task.task_name].quality,
                    bc_ids=self._get_bc_ids(task.task_name, result.task_metrics),
                ) for task in self.tasks
            }
        return MergeResult(save_path=model_path, qd_info=qd_info)
    
def run_qd(celery: Celery, cfg: DictConfig) -> None:

    if cfg.celery.mode in ["main", "solo"] and not cfg.qd.restart_dir:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        model_dir = os.path.join(output_dir, "models")
        merged_model_dir = os.path.join(output_dir, "merged_models")
        archive_dir = os.path.join(output_dir, "archives")
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(merged_model_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

    logger = logging.getLogger("QD")
    logger.info(f"Run QD for {cfg.qd.num_generation} generations ...")
    logger.info(f"seed models: {cfg.seed_model_path}")

    call_fn = celery.tasks["call"]
    all_tasks = cfg.tasks
    q_idx = 0
    q_name = all_tasks[q_idx]
    archive_map: Dict[str, Dict[Tuple[int], ArchiveData]] = {
        x: dict() for x in all_tasks
    }
    task_configs = load_task_configs(cfg)

    if not cfg.qd.restart_dir:
        # Initialize archive_map with the seed models.
        promises = [
            call_fn.delay("eval_model", model_path=model_path)
            for i, model_path in enumerate(cfg.seed_model_path)
        ]

        init_q = deque()
        init_q.extend(promises)
        init_results = []
        while init_q:
            promise = init_q.popleft()
            if promise.ready():
                init_results.append(promise.get(timeout=cfg.celery.timeout))
            else:
                init_q.append(promise)
    
        for result in init_results:
            for target_q_name in all_tasks:
                q_val = result.qd_info[target_q_name].quality
                bc_ids = result.qd_info[target_q_name].bc_ids
                if (
                    bc_ids not in archive_map[target_q_name] or
                    archive_map[target_q_name][bc_ids].quality < q_val
                ):
                    archive_map[target_q_name][bc_ids] = ArchiveData(
                        model_path=result.save_path,
                        quality=q_val,
                        sampling_freq=1,
                        validation_quality=None,
                    )
        archive_map_path = f"{archive_dir}/gen0_archive_map.json"
        save_archive_map(archive_map, archive_map_path)
        plot_elite_map(archive_map_path, task_configs, f"{image_dir}/gen0_elite_map_train.png", "train")

        gen = 1
    else:
        output_dir = cfg.qd.restart_dir
        model_dir = os.path.join(output_dir, "models")
        archive_dir = os.path.join(output_dir, "archives")
        image_dir = os.path.join(output_dir, "images")
        merged_model_dir = os.path.join(output_dir, "merged_models")
        largest_gen_file, largest_gen = get_largest_gen_file(archive_dir)
        archive_map = load_archive_map(f"{archive_dir}/{largest_gen_file}", ArchiveData)
        archive_map_path = f"{archive_dir}/gen0_archive_map.json"
        plot_elite_map(archive_map_path, task_configs, f"{image_dir}/gen0_elite_map_train.png", "train")
        gen = largest_gen + 1

    q = deque()
    for i in range(cfg.celery.num_workers-cfg.num_of_evaluation_workers):
        q_idx =  (q_idx + 1) % len(all_tasks)
        q_name = all_tasks[q_idx]
        # q_name = all_tasks[2]
        logger.info(f"QD switched to optimize {q_name}")
        q.append(
            call_fn.delay(
                "merge_models",
                q_name=q_name,
                generation=i+gen,
                model_dir=model_dir,
                archive_map=archive_map,
            )
        )
    
    # QD optimization.
    prev_log_time = time.time()
    while gen < cfg.qd.num_generation+1:
        promise = q.popleft()
        if promise.ready():
            eval_gen = gen - 1
            logger.info(f"Generation {eval_gen} ...")
            archive_map_path = f"{archive_dir}/gen{eval_gen}_archive_map.json"

            archive_train_path = f"{image_dir}/gen{eval_gen}_elite_map_train.png"
            save_archive_map(archive_map, archive_map_path)
            plot_elite_map(archive_map_path, task_configs, archive_train_path, "train")

            log_archive_map = {}
            for dataset, entries in archive_map.items():
                log_archive_map[dataset] = {str(coordinates): data.quality for coordinates, data in entries.items()}

            current_time = time.time()
            time_interval_minutes = (current_time - prev_log_time)
            prev_log_time = current_time

            wandb.log(
                {
                    f"archive_map_image_train/archive_map_image": wandb.Image(archive_train_path),
                    f"all_elite_map/all_elite_map": log_archive_map,
                    f"base_info/generation": eval_gen,
                    f"base_info/gpu_num": cfg.celery.num_workers,
                    f"base_info/log_interval_seconds": time_interval_minutes
                },
                step=eval_gen,
                commit=False,
            )

            result = promise.get(timeout=cfg.celery.timeout)
            archive_updated = False
            if result.save_path is not None:
                for k in archive_map:
                    q_val = result.qd_info[k].quality
                    bc_ids = result.qd_info[k].bc_ids
                    if (
                        bc_ids not in archive_map[k] or
                        archive_map[k][bc_ids].quality < q_val
                    ):
                        archive_map[k][bc_ids] = ArchiveData(
                            quality=q_val,
                            model_path=result.save_path,
                            sampling_freq=1,
                            validation_quality=None,
                        )
                        archive_updated = True

            wandb.log(
                {
                    f"save_path_true/save_path_true": 1 if result.save_path is not None else 0,
                    f"archive_map_updated/archive_map_updated": int(archive_updated),
                },
                step=eval_gen,
                commit=True,
            )

            # add new task to the queue
            if gen % cfg.qd.flip_interval == 0:
                q_idx =  (q_idx + 1) % len(all_tasks)
                q_name = all_tasks[q_idx]
                logger.info(f"QD switched to optimize {q_name}")

            new_promise = call_fn.delay(
                "merge_models",
                q_name=q_name,
                generation=gen+cfg.celery.num_workers-cfg.num_of_evaluation_workers,
                model_dir=model_dir,
                archive_map=archive_map,
            )
            q.append(new_promise)

            if gen % cfg.disk_cleaning_interval == 0:
                deleted_models = delete_outdated_models(archive_map, model_dir, gen-cfg.celery.num_workers*2)
                for model in deleted_models:
                    logger.info(f"Deleted model: {model}")

            gen += 1
        else:
            q.append(promise)

    # Clean up.
    deleted_models = delete_outdated_models(
        archive_map, model_dir, cfg.qd.num_generation)
    for model in deleted_models:
        logger.info(f"Deleted model: {model}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    def get_worker_cls(cfg):
        def init_func():
            return Worker(cfg)
        return init_func

    celery = setup_celery(
        name=cfg.celery.name,
        mode=cfg.celery.mode,
        worker_cls=get_worker_cls(cfg),
    )

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    run_qd(celery, cfg)


if __name__ == "__main__":
    main()