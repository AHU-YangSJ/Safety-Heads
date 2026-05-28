import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig
import numpy as np
import torch
import random
from easyeditor.util import nethook
from easyeditor.models import rome_bd as rome
from copy import deepcopy
from torch.utils.data import DataLoader
import time

import matplotlib.pyplot as plt
from plot_utils import plot_attention_heads, plot_heatmap, plot_distribution_bar, plot_strip_distribution

def mkdirs(dir_path_list):
    for path in dir_path_list if isinstance(dir_path_list, list) else [dir_path_list]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")


def plot_distribution(
    all_sim_list, all_diff_list,
    plot_dir, 
    num_iterations, 
    dim, 
    file_prefix, 
    title_iter_template, 
    title_mean, 
    x_label
):
    all_M_N_sim, all_N_N_sim, all_M_M_sim = all_sim_list
    all_M_N_diff, all_N_N_diff, all_M_M_diff = all_diff_list
    print(f"Plotting {plot_dir}...")
    plot_save_dir = plot_dir
    sim_dir = f"{plot_save_dir}/similarity"
    diff_dir = f"{plot_save_dir}/difference"
    mkdirs([plot_save_dir, sim_dir, diff_dir])

    # for i in tqdm(range(num_iterations)):
    #     if dim == 0:
    #         M_N_sim, N_N_sim, M_M_sim = all_M_N_sim[i], all_N_N_sim[i], all_M_M_sim[i]
    #         M_N_diff, N_N_diff, M_M_diff = all_M_N_diff[i], all_N_N_diff[i], all_M_M_diff[i]
    #     else:
    #         M_N_sim, N_N_sim, M_M_sim = all_M_N_sim[:, i], all_N_N_sim[:, i], all_M_M_sim[:, i]
    #         M_N_diff, N_N_diff, M_M_diff = all_M_N_diff[:, i], all_N_N_diff[:, i], all_M_M_diff[:, i]

    #     plot_file = f"{sim_dir}/{file_prefix}_{i}_sim.png"
    #     plot_attention_heads(
    #         [M_N_sim, N_N_sim, M_M_sim],
    #         plot_file, 
    #         title=f"{title_iter_template.format(i)} similarity",
    #         x_label=x_label,
    #         y_label="sim"
    #     )

    #     plot_file = f"{diff_dir}/{file_prefix}_{i}_diff.png"
    #     plot_attention_heads(
    #         [M_N_diff, N_N_diff, M_M_diff],
    #         plot_file,
    #         title=f"{title_iter_template.format(i)} difference",
    #         x_label=x_label,
    #         y_label="diff"
    #     )

    mean_M_N_sim = all_M_N_sim.mean(dim=dim)
    mean_N_N_sim = all_N_N_sim.mean(dim=dim)
    mean_M_M_sim = all_M_M_sim.mean(dim=dim)
    mean_M_N_diff = all_M_N_diff.mean(dim=dim)
    mean_N_N_diff = all_N_N_diff.mean(dim=dim)
    mean_M_M_diff = all_M_M_diff.mean(dim=dim)

    plot_file = f"{plot_save_dir}/mean_{file_prefix}_sim.png"
    plot_attention_heads(
        [mean_M_N_sim, mean_N_N_sim, mean_M_M_sim],
        plot_file, 
        title=f"{title_mean} similarity",
        x_label=x_label,
        y_label="sim"
    )

    plot_file = f"{plot_save_dir}/mean_{file_prefix}_diff.png"
    plot_attention_heads(
        [mean_M_N_diff, mean_N_N_diff, mean_M_M_diff],
        plot_file, 
        title=f"{title_mean} difference",
        x_label=x_label,
        y_label="diff"
    )

def plot_M_N_heads():
    pass

from custom_llm.modeling_llama import LlamaForCausalLM
from custom_llm.modeling_qwen3 import Qwen3ForCausalLM
from custom_llm.modeling_gemma3 import Gemma3ForConditionalGeneration
from custom_llm.modeling_mistral import MistralForCausalLM
from custom_llm.modeling_gemma2 import Gemma2ForCausalLM

models = ["llama-2-7b-chat-hf", "llama-3-8b-instruct", "qwen-3-8b", "mistral-7b-instruct-v0.3", 
          "vicuna-7b-v1.5", "llama-2-7b", "gemma-3-4b-it", "gemma-2-9b-it"]
params = ["llama2-7b-chat", "llama3-8b", "qwen3-8b", "mistral-7b", "vicuna-7b", "llama2-7b", "gemma3-4b", "gemma2-9b"]
model_classes = [LlamaForCausalLM, LlamaForCausalLM, Qwen3ForCausalLM, MistralForCausalLM, 
                 LlamaForCausalLM, LlamaForCausalLM, Gemma3ForConditionalGeneration, Gemma2ForCausalLM]

if __name__ == '__main__':

    cache_dir = "/data2/home/E22201116/models"
    model_idx = 5
    model_name = models[model_idx] 
    param_name = params[model_idx]

    read_multiheads = True
    reader_dir = f"readers_prompt3/{param_name}"

    MODEL_NAME = cache_dir + "/" + model_name
    config = AutoConfig.from_pretrained(MODEL_NAME, token='', trust_remote_code=True, cache_dir=cache_dir)
    config.read_multiheads = read_multiheads

    pair_class = "M_N"
    M_N_mhead_dict = torch.load(f"{reader_dir}/{pair_class}_layers_mhead_dict.pt")
    pair_class = "N_N"
    N_N_mhead_dict = torch.load(f"{reader_dir}/{pair_class}_layers_mhead_dict.pt")
    pair_class = "M_M"
    M_M_mhead_dict = torch.load(f"{reader_dir}/{pair_class}_layers_mhead_dict.pt")

    # # 拼接所有样本的多头输出 (layer_num, num_heads, sample_num, head_dim)
    # M_mhead_outputs = torch.cat([M_N_mhead_dict[layer]["harmful"] for layer in sorted(M_N_mhead_dict.keys())], dim=0)
    # N_mhead_outputs = torch.cat([M_N_mhead_dict[layer]["benign"] for layer in sorted(M_N_mhead_dict.keys())], dim=0)
    

    # 堆叠 行表示层 列表示头 相似度和差异 (layer_num, num_heads)
    all_M_N_sim = torch.cat([M_N_mhead_dict[layer]["mhead_sim"].unsqueeze(0) for layer in sorted(M_N_mhead_dict.keys())]).half()
    all_N_N_sim = torch.cat([N_N_mhead_dict[layer]["mhead_sim"].unsqueeze(0) for layer in sorted(N_N_mhead_dict.keys())]).half()
    all_M_M_sim = torch.cat([M_M_mhead_dict[layer]["mhead_sim"].unsqueeze(0) for layer in sorted(M_M_mhead_dict.keys())]).half()

    all_M_N_diff = torch.cat([M_N_mhead_dict[layer]["mhead_diff"].unsqueeze(0) for layer in sorted(M_N_mhead_dict.keys())]).half()
    all_N_N_diff = torch.cat([N_N_mhead_dict[layer]["mhead_diff"].unsqueeze(0) for layer in sorted(N_N_mhead_dict.keys())]).half()
    all_M_M_diff = torch.cat([M_M_mhead_dict[layer]["mhead_diff"].unsqueeze(0) for layer in sorted(M_M_mhead_dict.keys())]).half()

    # 计算差异矩阵
    # diff_diff_MM_MN, diff_diff_MM_NN, diff_diff_NN_MN = all_M_M_diff - all_M_N_diff, all_M_M_diff - all_N_N_diff, all_N_N_diff - all_M_N_diff
    sim_diff_MM_MN, sim_diff_MM_NN, sim_diff_NN_MN = all_M_M_sim - all_M_N_sim, all_M_M_sim - all_N_N_sim, all_N_N_sim - all_M_N_sim
    # 夹角差异矩阵
    diff_diff_MN_MM, diff_diff_MN_NN, diff_diff_NN_MM = all_M_N_diff - all_M_M_diff, all_M_N_diff - all_N_N_diff, all_N_N_diff - all_M_M_diff

    diff = (sim_diff_MM_MN + sim_diff_MM_NN) # (sim_diff_MM_MN + sim_diff_MM_NN + sim_diff_NN_MN)    # 计算综合差异指标 (layer_num, num_heads)
    diff = sim_diff_MM_MN + sim_diff_NN_MN
    # # diff 矩阵归一化
    # diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-10)

    # 绘制 diff 分布情况
    bar_path = f"{reader_dir}/diff_distribution.png"
    plot_distribution_bar(diff, bar_path, title="Safety Score Distribution", x_label="Score Range", y_label="Count")
    # 绘制带状分布图
    strip_path = f"{reader_dir}/diff_strip.png"
    plot_strip_distribution(diff, strip_path, title=None, x_label="Score", model_name=param_name.split("-")[0])
    


    # # 相似度差异矩阵
    # sim_diff_MM_MN, sim_diff_MM_NN, sim_diff_NN_MN = all_M_M_sim - all_M_N_sim, all_M_M_sim - all_N_N_sim, all_N_N_sim - all_M_N_sim
    # diff_diff_MN_MM, diff_diff_MN_NN, diff_diff_NN_MM = all_M_N_diff - all_M_M_diff, all_M_N_diff - all_N_N_diff, all_N_N_diff - all_M_M_diff
    # diff = sim_diff_MM_MN + sim_diff_NN_MN

    # diff矩阵画热力图，保存路径为hotmap_dir
    hotmap_dir = f"{reader_dir}/heatmap_diff.png"
    plot_heatmap(diff.clamp(min=0,max=1), hotmap_dir, title="Safety Scores", x_label="Heads", y_label="Layers")
    hotmap_dir = f"{reader_dir}/heatmap_MM.png"
    plot_heatmap(all_M_M_sim.clamp(min=0,max=1), hotmap_dir, title="Malicious-Malicious", x_label="Heads", y_label="Layers")
    hotmap_dir = f"{reader_dir}/heatmap_MN.png"
    plot_heatmap(all_M_N_sim.clamp(min=0,max=1), hotmap_dir, title="Malicious-Normal", x_label="Heads", y_label="Layers")
    hotmap_dir = f"{reader_dir}/heatmap_NN.png"
    plot_heatmap(all_N_N_sim.clamp(min=0,max=1), hotmap_dir, title="Normal-Normal", x_label="Heads", y_label="Layers")
    # 程序在此中断
    # a=1/0

    heads_dir = f"{reader_dir}/heads_distribution"
    layers_dir = f"{reader_dir}/layers_distribution"
    heads_diff_dir = f"{reader_dir}/heads_difference_distribution"
    layers_diff_dir = f"{reader_dir}/layers_difference_distribution"
    mkdirs([heads_dir, layers_dir, heads_diff_dir, layers_diff_dir])

    # 清空目录
    if len(os.listdir(heads_dir)) > 0:
        os.system(f"rm -rf {heads_dir}/*")
    if len(os.listdir(layers_dir)) > 0:
        os.system(f"rm -rf {layers_dir}/*")
        
    # 画图
    # 两组样本对在每一层的 attention heads 的相似度和diff 这里按layer绘折线图
    plot_distribution(
        [all_M_N_sim, all_N_N_sim, all_M_M_sim],
        [all_M_N_diff, all_N_N_diff, all_M_M_diff],
        heads_dir, 
        all_M_N_sim.shape[0], # config.num_hidden_layers, 
        0, 
        "layer", 
        "layer {} heads", 
        "mean heads", 
        "heads"
    )
    
    # 绘图 每个头在各层上的分布变化
    plot_distribution(
        [all_M_N_sim, all_N_N_sim, all_M_M_sim],
        [all_M_N_diff, all_N_N_diff, all_M_M_diff],
        layers_dir, 
        all_M_N_sim.shape[1], 
        1, 
        "head", 
        "head {} layers", 
        "mean layers", 
        "layers"
    )

    # 画图 差异矩阵 每个头在各层上的差异变化
    plot_distribution(
        [sim_diff_MM_MN, sim_diff_NN_MN, sim_diff_MM_NN],
        [diff_diff_MN_MM.abs(), diff_diff_MN_NN.abs(), diff_diff_NN_MM.abs()],
        heads_diff_dir, 
        sim_diff_MM_MN.shape[0], 
        0, 
        "layer", 
        "layer {} heads diff", 
        "mean heads diff", 
        "heads"
    )

    # 画图 差异矩阵 每个头在各层上的差异变化
    plot_distribution(
        [sim_diff_MM_MN, sim_diff_NN_MN, sim_diff_MM_NN],
        [diff_diff_MN_MM.abs(), diff_diff_MN_NN.abs(), diff_diff_NN_MM.abs()],
        layers_diff_dir, 
        sim_diff_MM_MN.shape[1], 
        1, 
        "head", 
        "head {} layers diff", 
        "mean layers diff", 
        "layers"
    )

    print("All done.")