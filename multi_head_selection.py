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
from plot_utils import plot_attention_heads, plot_heatmap

def mkdirs(dir_path_list):
    for path in dir_path_list if isinstance(dir_path_list, list) else [dir_path_list]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

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

    model_id = 3
    ret = 160
    cache_dir = "/data/amax/home/E22201116/models"
    model_name = models[model_id]
    param_name =  params[model_id]

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

    # 拼接所有样本的多头输出 (layer_num, num_heads, sample_num, head_dim)
    M_mhead_outputs = torch.cat([M_N_mhead_dict[layer]["harmful"].unsqueeze(0) for layer in sorted(M_N_mhead_dict.keys())], dim=0)
    N_mhead_outputs = torch.cat([M_N_mhead_dict[layer]["benign"].unsqueeze(0) for layer in sorted(M_N_mhead_dict.keys())], dim=0)
    

    # 堆叠 行表示层 列表示头 相似度和差异 (layer_num, num_heads)
    all_M_N_sim = torch.cat([M_N_mhead_dict[layer]["mhead_sim"].unsqueeze(0) for layer in sorted(M_N_mhead_dict.keys())])
    all_N_N_sim = torch.cat([N_N_mhead_dict[layer]["mhead_sim"].unsqueeze(0) for layer in sorted(N_N_mhead_dict.keys())])
    all_M_M_sim = torch.cat([M_M_mhead_dict[layer]["mhead_sim"].unsqueeze(0) for layer in sorted(M_M_mhead_dict.keys())])

    all_M_N_diff = torch.cat([M_N_mhead_dict[layer]["mhead_diff"].unsqueeze(0) for layer in sorted(M_N_mhead_dict.keys())])
    all_N_N_diff = torch.cat([N_N_mhead_dict[layer]["mhead_diff"].unsqueeze(0) for layer in sorted(N_N_mhead_dict.keys())])
    all_M_M_diff = torch.cat([M_M_mhead_dict[layer]["mhead_diff"].unsqueeze(0) for layer in sorted(M_M_mhead_dict.keys())])

    # 相似度差异矩阵
    sim_diff_MM_MN, sim_diff_MM_NN, sim_diff_NN_MN = all_M_M_sim - all_M_N_sim, all_M_M_sim - all_N_N_sim, all_N_N_sim - all_M_N_sim
    # 夹角差异矩阵
    diff_diff_MN_MM, diff_diff_MN_NN, diff_diff_NN_MM = all_M_N_diff - all_M_M_diff, all_M_N_diff - all_N_N_diff, all_N_N_diff - all_M_M_diff
    # MM-MN 和 MM-NN 的相似度差异之和 作为综合差异指标 
    # 此矩阵中每个元素表示一个注意力头，越大越亮，越亮越重要
    # diff = (sim_diff_MM_MN + sim_diff_MM_NN)
    diff = sim_diff_MM_MN + sim_diff_NN_MN

    # 找到diff中最重要的前k个注意力头
    k = ret  # 32, 64, 96, 128, 160, 192, 224, 256
    topk_values, topk_indices = torch.topk(diff.flatten(), k)

    # # 打印LLM的每一层中选出的头的数量
    # layer_head_count = {idx: 0 for idx in range(diff.size(0))}    
    # for index in topk_indices:
    #     layer = index // diff.size(1)
    #     head = index % diff.size(1)
    #     layer_head_count[layer.item()] += 1
    # for layer, count in layer_head_count.items():
    #     print(f"{count}")

    safe_neurons_list = []
    for value, index in zip(topk_values, topk_indices):
        # if value < 1:
        #     break
        layer = index // diff.size(1)
        head = index % diff.size(1)
        safe_neurons_list.append( [layer.item(), head.item()] )
        print(f"Value: {value.item():.4f}, Layer: {layer.item()}, Head: {head.item()}")
    safe_neurons_list = torch.tensor(safe_neurons_list).cuda()
    torch.save(safe_neurons_list, f"{reader_dir}/safe_heads_id.pt")
    print(f"{param_name} Top {safe_neurons_list.shape[0]} important attention heads (value, layer, head):")

    print("All done.")