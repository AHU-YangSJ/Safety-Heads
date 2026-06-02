import ast
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from easyeditor.util import nethook

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors



from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from transformers import GenerationConfig


def get_reject_strings():
    # 返回列表包含LLM常用的拒绝回答短语
    reject_phrases = [
        # qwen3-8b 常见拒绝
        "I'm sorry", 
        "I can't",
        "I cannot",
        "I'm unable",
        "I won't be",
        "I must not",
        "I can't",
        "Ethical and Legal",
        "unethical",
        "illegal",
        "I can't help",
        "not provide",
        "I must not provide",
        "I'm not helping",
        "I don't provide",
        "not provide",
    ]
    return reject_phrases


def load_dataset(dataset_path, column_name=None) -> pd.Series:
    # Check the file extension to determine the file type
    _, file_extension = os.path.splitext(dataset_path)

    if file_extension.lower() == ".csv":
        # For CSV files, use pandas to read and return the specified column
        if column_name is not None:
            df = pd.read_csv(dataset_path)
            return df[column_name]
        else:
            # If the column name is not specified, read the first column by default
            df = pd.read_csv(dataset_path, header=None)
            return df[0]
    elif file_extension.lower() == ".txt":
        # For TXT files, read each line as a separate data point
        with open(dataset_path, "r", encoding="utf-8") as file:
            data = file.read().splitlines()
        df = pd.DataFrame(data, columns=["source"])
        return df["source"]
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    

def get_hidden_states(model, tokenizer, full_prompt_list, hparams, is_accept=True):
    model.eval()
    
    # full_prompt_list, prompt_lookup_idxs = take_templete_dataset(full_prompt_list, tokenizer, hparams, is_accept=is_accept)

    templete = f"{hparams.templete_head} " + "{}" + f" {hparams.templete_last}"

    # hidden_state_list = []
    act_outputs_list = {i:[] for i in range(hparams.v_loss_layer+1)}
    flag = True
    # messages = [{"role": "system", "content": hparams.system_prompt},]

    with torch.no_grad():

        for idx in tqdm(range(len(full_prompt_list))):
            # tmp_message = [{"role": "user", "content": f"{full_prompt_list[idx]}"},]
            # full_string = tokenizer.apply_chat_template(messages+tmp_message, tokenize=False, add_generation_prompt=True)
            # full_string = hparams.system_prompt + '\n\n' + full_string 
            full_string = templete.format(full_prompt_list[idx])
            if flag:
                print(full_string)
                flag = False

            inputs = tokenizer(
                full_string,
                return_tensors="pt",
                # padding=True,
            ).to(f"{model.device}")


            output = model(**inputs)

            for layer in range(hparams.v_loss_layer+1):
                # output.multi_head_outputs[layer]: (batch_size, num_heads, head_dim)
                act_outputs_list[layer].append(output.multi_head_outputs[layer].cpu())


    # hidden_state_list = torch.stack(hidden_state_list)
    for layer in range(hparams.v_loss_layer+1):
        # act_outputs_list[layer]: (sample_num, num_heads, head_dim)
        act_outputs_list[layer] = torch.cat(act_outputs_list[layer], dim=0)

    # act_outputs_list = torch.cat(act_outputs_list, dim=0)

    return act_outputs_list

def ensure_numpy_float32(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().to(torch.float32).numpy()
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    else:
        raise ValueError("输入数据必须是 torch.Tensor 或 np.ndarray")

def random_select(data, num, seed):
    for key, value in data.items():
        # value是一个矩阵，要求在第0维随机选择num次，使其扩充成(num, ...)
        total_num = value.shape[0]
        random.seed(seed)
        selected_indices = random.choices(range(total_num), k=num)
        data[key] = value[selected_indices]
    return data

def range_select_pair(data, num, seed):
    data1 = {k:[] for k in data.keys()}
    data2 = {k:[] for k in data.keys()}
    random.seed(seed)
    for key, value in data.items():
        # value是一个矩阵，要求在第0维随机选择num次，每次随机选择两个，要求每次选择的两个不能相同,分别放入data1和data2中key对应的value中，最后分别将data1和data2的value堆叠成矩阵
        total_num = value.shape[0]
        for _ in range(num):
            idx1, idx2 = random.sample(range(total_num), 2)
            data1[key].append(value[idx1].unsqueeze(0))
            data2[key].append(value[idx2].unsqueeze(0))
        data1[key] = torch.cat(data1[key], dim=0)
        data2[key] = torch.cat(data2[key], dim=0)
    return data1, data2

def anchor_accept_point(
    model, tokenizer, dataset_benign, dataset_harmful, hparams, reader_dir
):
    random_select_num = hparams.random_select_num
    model_dir = reader_dir 

    print("Getting hidden states for Benign samples...")
    all_layers_benign_mhead_outputs = get_hidden_states(model, tokenizer, dataset_benign, hparams, is_accept=True)
    print("Getting hidden states for Harmful samples...")
    all_layers_harmful_mhead_outputs = get_hidden_states(model, tokenizer, dataset_harmful, hparams, is_accept=False)
    print()

    layers_mhead_dict = {id:{"mhead_sim":None, "mhead_diff":None, "harmful":None, "benign":None} for id in range(hparams.v_loss_layer+1)}

    if hparams.pair_class == "M_N": 
        for layer in layers_mhead_dict.keys():
            layers_mhead_dict[layer]["harmful"] = all_layers_harmful_mhead_outputs[layer].transpose(0, 1) # (num_heads, sample_num, head_dim)
            layers_mhead_dict[layer]["benign"] = all_layers_benign_mhead_outputs[layer].transpose(0, 1) # (num_heads, sample_num, head_dim)

        all_layers_benign_mhead_outputs = random_select(all_layers_benign_mhead_outputs, random_select_num, hparams.select_seed[0])
        all_layers_harmful_mhead_outputs = random_select(all_layers_harmful_mhead_outputs, random_select_num, hparams.select_seed[1])
    else:
        all_layers_benign_mhead_outputs, all_layers_harmful_mhead_outputs = \
            range_select_pair(all_layers_benign_mhead_outputs, random_select_num, hparams.select_seed[0])
    
    for layer in tqdm(range(hparams.v_loss_layer+1)):
        # (sample_num, num_heads, head_dim)
        harmful_mheads_output, benign_mheads_output = all_layers_harmful_mhead_outputs[layer], all_layers_benign_mhead_outputs[layer]

        # 计算每个样本对之间在每个头上的余弦相似度
        all_sample_mheads_sim = F.cosine_similarity(harmful_mheads_output, benign_mheads_output, dim=-1)  # (sample_num, num_heads)
        mean_mheads_sim = all_sample_mheads_sim.mean(dim=0)  # (num_heads,)

        # 计算向量夹角
        all_sample_mheads_diff = torch.acos(torch.clamp(all_sample_mheads_sim, -1.0, 1.0)).abs()*180/torch.pi  # (sample_num, num_heads)
        mean_mheads_diff = all_sample_mheads_diff.mean(dim=0)  # (num_heads,)

        # 存储结果
        layers_mhead_dict[layer]["mhead_sim"] = mean_mheads_sim.cuda()
        layers_mhead_dict[layer]["mhead_diff"] = mean_mheads_diff.cuda()
        
    
    print(f"\nSaving results to {model_dir}/{hparams.pair_class}_layers_mhead_dict.pt")
    torch.save(layers_mhead_dict, f"{model_dir}/{hparams.pair_class}_layers_mhead_dict.pt")
    return layers_mhead_dict


