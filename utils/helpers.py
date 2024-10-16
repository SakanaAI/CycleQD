import torch
import vllm
import json
import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
from typing import Dict, List, Tuple, Any
from dataclasses import asdict
from collections import defaultdict


def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    model = llm.llm_engine.driver_worker.model_runner.model
    num_layers = model.config.num_hidden_layers

    # Load embeddings layer weights.
    model_param = model.get_parameter('model.embed_tokens.weight')
    model_param.copy_(
        param['model.embed_tokens.weight'][:model_param.shape[0]].to(
            model_param.dtype).to(model_param.device)
    )
    model_param = model.get_parameter('lm_head.weight')
    model_param.copy_(
        param['lm_head.weight'][:model_param.shape[0]].to(
            model_param.dtype).to(model_param.device)
    )

    # Load the final layernorm weights.
    model_param = model.get_parameter('model.norm.weight')
    model_param.copy_(
        param['model.norm.weight'].to(model_param.dtype).to(model_param.device)
    )

    for i in range(num_layers):
        # Load qkv_proj weights.
        model_param = model.get_parameter(
            f'model.layers.{i}.self_attn.qkv_proj.weight')
        model_param.copy_(
            torch.cat([
                param[f'model.layers.{i}.self_attn.q_proj.weight'],
                param[f'model.layers.{i}.self_attn.k_proj.weight'],
                param[f'model.layers.{i}.self_attn.v_proj.weight'],
            ], dim=0).to(model_param.dtype).to(model_param.device)
        )
        # Load gate_up_proj weights.
        model_param = model.get_parameter(
            f'model.layers.{i}.mlp.gate_up_proj.weight')
        model_param.copy_(
            torch.cat([
                param[f'model.layers.{i}.mlp.gate_proj.weight'],
                param[f'model.layers.{i}.mlp.up_proj.weight'],
            ], dim=0).to(model_param.dtype).to(model_param.device)
        )
        # Load o_proj and down_proj weights.
        model_param = model.get_parameter(
            f'model.layers.{i}.self_attn.o_proj.weight')
        model_param.copy_(
            param[f'model.layers.{i}.self_attn.o_proj.weight'].to(
                model_param.dtype).to(model_param.device)
        )
        model_param = model.get_parameter(
            f'model.layers.{i}.mlp.down_proj.weight')
        model_param.copy_(
            param[f'model.layers.{i}.mlp.down_proj.weight'].to(
                model_param.dtype).to(model_param.device)
        )
        # Load layer_norm weights.
        model_param = model.get_parameter(
            f'model.layers.{i}.input_layernorm.weight')
        model_param.copy_(
            param[f'model.layers.{i}.input_layernorm.weight'].to(
                model_param.dtype).to(model_param.device)
        )
        model_param = model.get_parameter(
            f'model.layers.{i}.post_attention_layernorm.weight')
        model_param.copy_(
            param[f'model.layers.{i}.post_attention_layernorm.weight'].to(
                model_param.dtype).to(model_param.device)
        )


def save_archive_map(data_map: Dict, output_path: str) -> None:
    serializable_map = {}
    for outer_key, inner_dict in data_map.items():
        serializable_inner_dict = {}
        for tuple_key, archive_data in inner_dict.items():
            string_key = ','.join(map(str, tuple_key))
            serializable_inner_dict[string_key] = asdict(archive_data)
        serializable_map[outer_key] = serializable_inner_dict
    
    with open(output_path, 'w') as f:
        json.dump(serializable_map, f, indent=4)


def load_archive_map(input_path: str, archive_data_class: Any) -> Dict[Any, Dict[Tuple, Any]]:
    with open(input_path, 'r') as f:
        serializable_map = json.load(f)
    
    data_map = {}
    for outer_key, inner_dict in serializable_map.items():
        deserialized_inner_dict = {}
        for string_key, archive_data_dict in inner_dict.items():
            tuple_key = tuple(map(int, string_key.split(',')))
            archive_data = archive_data_class(**archive_data_dict)
            deserialized_inner_dict[tuple_key] = archive_data
        data_map[outer_key] = deserialized_inner_dict
    
    return data_map


def delete_outdated_models(
        data_map: Dict, model_dir: str, threshold: int) -> List[str]:
    model_path_in_archive = {
        os.path.basename(data_map[k][bc_ids].model_path)
        for k in data_map for bc_ids in data_map[k]
    }
    model_path_on_disk = os.listdir(model_dir)
    deleted_models = []
    for model_path in model_path_on_disk:
        # Delete the model if it is not in the archive map and not generated
        # by a worker that just finished its work but has not been dealt with.
        gen_id = int(model_path.split('_')[-1])
        if (
            model_path not in model_path_in_archive and
            gen_id < threshold
        ):
            full_model_path = os.path.join(model_dir, model_path)
            if os.path.exists(full_model_path):
                try:
                    shutil.rmtree(full_model_path)
                    deleted_models.append(full_model_path)
                except OSError as e:
                    print(f"Error deleting {full_model_path}: {e}")
    return deleted_models

def get_largest_gen_file(model_dir):
    files = os.listdir(model_dir)
    gen_regex = re.compile(r"gen(\d+)_archive_map\.json")
    gen_files = {}
    for file in files:
        match = gen_regex.match(file)
        if match:
            gen_num = int(match.group(1))
            gen_files[gen_num] = file
    largest_gen_num = max(gen_files.keys())
    largest_gen_file = gen_files[largest_gen_num]
    return largest_gen_file, largest_gen_num

def plot_elite_map(
        archive_map_path: str, task_configs: dict, output_path: str, data_split: str) -> None:
    # Load the archive map.
    with open(archive_map_path, 'r') as f:
        data = json.load(f)
    
    # Plot the elite maps
    data_len = len(data.keys())
    if data_len == 3:
        plot_num = 1
    elif data_len == 4:
        plot_num = 3
    else:
        raise NotImplementedError(f"Data length {data_len} not supported yet.")

    df_dict = {}
    for k in data:
        df_dict[k] = {}
        bc_num_dim = len(list(data[k].keys())[0].split(","))
        for i in range(bc_num_dim):
            df_dict[k][f"bc_dim{i}"] = []
        df_dict[k]["quality"] = []
        if data_split == "validation":
            df_dict[k]["train_quality"] = []
        for bc_ids in data[k]:
            for i, bc_id in enumerate(bc_ids.split(",")):
                df_dict[k][f"bc_dim{i}"].append(int(bc_id))
            if data_split in ["all", "train"]:
                df_dict[k]["quality"].append(data[k][bc_ids]["quality"])
            elif data_split == "validation":
                df_dict[k]["quality"].append(data[k][bc_ids]["validation_quality"])
                df_dict[k]["train_quality"].append(data[k][bc_ids]["quality"])
            else:
                raise ValueError(f"Invalid data split: {data_split}")
    
    fig, axes = plt.subplots(plot_num, data_len, figsize=(9 * data_len, 8*plot_num))

    for i, key_q in enumerate(df_dict):
        df = pd.DataFrame(df_dict[key_q])
        q_min = task_configs[key_q]['bc_min_vals'][0]
        q_max = task_configs[key_q]['bc_max_vals'][0]
        other_keys = [key for key in task_configs.keys() if key!=key_q]
        for j in range(plot_num):
            if j == 0:
                key_x_idx = 0
                key_y_idx = 1
            elif j == 1:
                key_x_idx = 1
                key_y_idx = 2
            elif j == 2:
                key_x_idx = 0
                key_y_idx = 2
            else:
                raise ValueError(f"Invalid plot index: {j}")

            ax = axes[j][i] if plot_num > 1 else axes[i]
            
            key_x = other_keys[key_x_idx]
            x_min = task_configs[key_x]['bc_min_vals'][0]
            x_max = task_configs[key_x]['bc_max_vals'][0]
            x_grid_size = task_configs[key_x]['bc_grid_sizes'][0]

            key_y = other_keys[key_y_idx]
            y_min = task_configs[key_y]['bc_min_vals'][0]
            y_max = task_configs[key_y]['bc_max_vals'][0]
            y_grid_size = task_configs[key_y]['bc_grid_sizes'][0]

            elite_map = np.zeros([x_grid_size, y_grid_size])
            max_q_values = {}
            for _, row in df.iterrows():
                x = int(row[f"bc_dim{key_x_idx}"])
                y = int(row[f"bc_dim{key_y_idx}"])
                if data_split in ["all", "train"]:
                    q = row["quality"]
                    if (x, y) not in max_q_values or max_q_values[(x, y)] < q:
                        max_q_values[(x, y)] = q
                        elite_map[x, y] = q
                elif data_split == "validation":
                    q = row["train_quality"]
                    if (x, y) not in max_q_values or max_q_values[(x, y)] < q:
                        max_q_values[(x, y)] = q
                        elite_map[x, y] = row["quality"]

            elite_map = elite_map.T

            im = ax.imshow(elite_map, cmap="viridis", vmin=q_min, vmax=q_max)
            ax.set_title(f"{key_q}", fontsize=20)
            ax.set_xlabel(key_x, fontsize=18)
            ax.set_ylabel(key_y, fontsize=18)
            ax.set_xticks(np.arange(-0.5, x_grid_size, 1))
            ax.set_yticks(np.arange(-0.5, y_grid_size, 1))
            ax.set_xticklabels(np.round(np.linspace(x_min, x_max, x_grid_size+1), 2))
            ax.set_yticklabels(np.round(np.linspace(y_min, y_max, y_grid_size+1), 2))
            ax.invert_yaxis()

            median_val = np.median(elite_map[elite_map > 0])
            for _, row in df.iterrows():
                x = int(row[f"bc_dim{key_x_idx}"])
                y = int(row[f"bc_dim{key_y_idx}"])
                q = elite_map[y, x]
                c = 'white' if q < median_val else 'red'
                text = ax.text(x, y, f'{q:.2f}', ha='center', va='center', color=c, fontsize=14)
            
            fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def elite_model_table(
        archive_map_path: str,
        output_path: str,
        top_k: int = 8) -> None:
    # Load the archive map.
    with open(archive_map_path, 'r') as f:
        data = json.load(f)

    top_k_data = {}
    for key, value in data.items():
        # Convert to DataFrame
        df = pd.DataFrame(value).T
        df = df.sort_values(by='quality', ascending=False).head(top_k)
        top_k_data[key] = df

    def find_dict_by_model_path(data, model_path):
        for key, value in data.items():
            if value.get("model_path") == model_path:
                if value.get("validation_quality") is not None:
                    return round(value.get("validation_quality"), 3)
                else:
                    "not max model"
        return "not elite"

    result = []
    for category, df in top_k_data.items():
        for index, row in df.iterrows():
            result_dict = {
                "category": category,
                "model_name": os.path.basename(row["model_path"]),
                "training": round(row["quality"], 3),
            }
            for key in data.keys():
                key_validation = find_dict_by_model_path(data[key], row["model_path"])
                result_dict[key] = key_validation
            result.append(result_dict)

    # Create a DataFrame
    df = pd.DataFrame(result)

    # Check for duplicate model names and keep track of their indices
    model_name_counts = defaultdict(int)
    duplicate_indices = []

    for idx, name in enumerate(df['model_name']):
        model_name_counts[name] += 1
        if model_name_counts[name] > 1:
            duplicate_indices.append(idx)

    # Visualize the data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for idx, cell in table.get_celld().items():
        row, col = idx
        if row > 0 and col == df.columns.get_loc('model_name') and row in duplicate_indices:
            cell.set_text_props(color='red')

    plt.savefig(output_path)