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

            # generation_config = GenerationConfig(
            #     temperature=hparams.temperature,
            #     top_p=hparams.top_p,
            #     top_k=hparams.top_k,
            #     pad_token_id=0

            # )
            # output = model.generate(
            #     input_ids=inputs["input_ids"],
            #     output_hidden_states= False,
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     # output_scores=True,
            #     max_new_tokens=hparams.new_token_len,
            #     num_return_sequences=1
            # )

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
    """
    数据预处理工具函数：
    1. 处理 PyTorch Tensor -> Numpy 的转换
    2. 处理 GPU -> CPU 的移动
    3. 关键：将 float16/bfloat16 强制转换为 float32，避免统计计算时的数值溢出
    """
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
    # if os.path.exists(f"{model_dir}/layers_mhead_dict.pt"):
    #     print(f"Loading existing results from {model_dir}/layers_mhead_dict.pt")
    #     all_layers_mhead_dict = torch.load(f"{model_dir}/layers_mhead_dict.pt")
    # else:
    #     all_layers_mhead_dict = None
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




def anchor_accept_point_gate(
    model, tokenizer, dataset_benign, dataset_harmful, hparams, neuron_dir
):
    layers_direction_delta = {}
    model_dir = neuron_dir 
    
    if os.path.exists(f"{model_dir}/layers_actfn_dict.pt"):
        print(f"Loading existing results from {model_dir}/layers_actfn_dict.pt")
        all_layers_actfn_dict = torch.load(f"{model_dir}/layers_actfn_dict.pt")
    else:
        all_layers_actfn_dict = None

    print("Getting hidden states for Benign samples...")
    all_layers_benign_act_outputs = get_hidden_states(model, tokenizer, dataset_benign, hparams, is_accept=True)
    print("Getting hidden states for Harmful samples...")
    all_layers_harmful_act_outputs = get_hidden_states(model, tokenizer, dataset_harmful, hparams, is_accept=False)
    print()
    
    layers_actfn_dict = {id:{"act_key":None, "act_value":None, "deact_key":None, "deact_value":None} for id in range(hparams.v_loss_layer+1)}

    for layer in tqdm(range(hparams.v_loss_layer+1)):
        harmful_outputs, benign_outputs = all_layers_harmful_act_outputs[layer], all_layers_benign_act_outputs[layer]
        # 计算差值
        diff = harmful_outputs - benign_outputs
        neuron_k = int(diff.shape[-1]*hparams.neuron_ratio)

        # 选择目标神经元
        # ######################## 下一步尝试用PCA筛选神经元编号 ########################
        if all_layers_actfn_dict is not None:
            layer_act_index, layer_deact_index = all_layers_actfn_dict[layer]["act_key"].to(diff.device), all_layers_actfn_dict[layer]["deact_key"].to(diff.device)
            # layer_act_neuron_index, layer_deact_neuron_index = loaded_act_index[layer], loaded_deact_index[layer]
        else:   
            # 排序差值，此处按照diff均值(1xd)排序，反映了神经元在样本间的整体变化趋势
            # 此处排序时同时考虑了了激活和抑制两个方向，避免出现某些神经元在样本间同时存在激活和抑制，且抵触较大的情况
            sorted_pca_act_diff, sorted_act_index = diff.mean(dim=0).sort(descending=True)
            sorted_pca_deact_diff, sorted_deact_index = diff.mean(dim=0).sort(descending=False)

            # pca = PCA(n_components=1)
            # diff_np = ensure_numpy_float32(diff)
            # pca_result = pca.fit_transform(diff_np.T)
            # torch_pca_result = torch.from_numpy(pca_result).T.squeeze().to(torch.float16)
            # # 根据PCA结果排序
            # sorted_pca_act_diff, sorted_pca_act_index = torch_pca_result.sort(descending=True)
            # sorted_pca_deact_diff, sorted_pca_deact_index = torch_pca_result.sort(descending=False)
            # sorted_act_index, sorted_deact_index = sorted_pca_act_index, sorted_pca_deact_index
            

            # 前k个幅度最大的神经元
            layer_act_index, layer_deact_index = sorted_act_index[0:neuron_k], sorted_deact_index[0:neuron_k]
            # layers_act_neuron_index.append(layer_act_index.unsqueeze(0))
            # layers_deact_neuron_index.append(layer_deact_index.unsqueeze(0))


        # act和deact的diff符号与输出，此处按照原始diff值(nxd), 反映了神经元在每个样本上的具体变化情况
        # 此处区分开激活和抑制，分别观察变化
        act_neuron_signs, act_neuron_outputs = diff.sign().clamp(min=0), diff.clamp(min=0)
        deact_neuron_signs, deact_neuron_outputs = diff.sign().clamp(max=0), diff.clamp(max=0)

        ## 目标神经元 样本平均 激活/抑制 率
        rate_act_signs = act_neuron_signs[:, layer_act_index].mean(dim=0)
        rate_deact_signs = deact_neuron_signs[:, layer_deact_index].mean(dim=0)
        ## 目标神经元 样本平均 激活/抑制 幅度 均值池化
        mean_act_values = act_neuron_outputs[:, layer_act_index].mean(dim=0)
        mean_deact_values = deact_neuron_outputs[:, layer_deact_index].mean(dim=0)

        # mean_act_values = sorted_pca_act_diff[0:neuron_k]
        # mean_deact_values = sorted_pca_deact_diff[0:neuron_k]

    
        # # 兴奋神经元 兴奋率和兴奋程度 绘图
        # xmin, xmax = 0, max(0, int(rate_act_signs.numel() - 1))
        # a_min, a_max = 0, 1.1
        # plot_vector_line(
        #     rate_act_signs, save_name=f"{model_dir}/{hparams.pair_class}/act_sign/layer{layer}_line.png",
        #     xlim=[xmin, xmax], ylim=[a_min, a_max],
        # )
        # a_min, a_max = 0, 1 #act_output.abs().max().item()*1.1
        # plot_vector_line(
        #     mean_act_values.abs(), save_name=f"{model_dir}/{hparams.pair_class}/act_output/layer{layer}_line.png", 
        #     xlim=[xmin, xmax], ylim=[a_min, a_max],
        # )
        # # 抑制神经元 抑制率和抑制程度 绘图
        # xmin, xmax = 0, max(0, int(rate_deact_signs.numel() - 1))
        # a_min, a_max = 0, 1.1
        # plot_vector_line(
        #     rate_deact_signs.abs(), save_name=f"{model_dir}/{hparams.pair_class}/deact_sign/layer{layer}_line.png",
        #     xlim=[xmin, xmax], ylim=[a_min, a_max],
        # )
        # a_min, a_max = 0, 1 # deact_output.abs().max().item()*1.1
        # plot_vector_line(
        #     mean_deact_values.abs(), save_name=f"{model_dir}/{hparams.pair_class}/deact_output/layer{layer}_line.png", 
        #     xlim=[xmin, xmax], ylim=[a_min, a_max],
        # )
        

        layers_actfn_dict[layer]["act_key"] = layer_act_index.cuda()
        layers_actfn_dict[layer]["act_value"] = mean_act_values.cuda()
        layers_actfn_dict[layer]["deact_key"] = layer_deact_index.cuda()
        layers_actfn_dict[layer]["deact_value"] = mean_deact_values.cuda()

        # act_output_list.append(mean_act_values.unsqueeze(0))
        # deact_output_list.append(mean_deact_values.unsqueeze(0))

    if all_layers_actfn_dict is None:
        torch.save(layers_actfn_dict, f"{model_dir}/layers_actfn_dict.pt")

    print()
    return layers_actfn_dict

    for layer in tqdm(range(hparams.v_loss_layer+1)):
        harmful_outputs, benign_outputs = all_layers_harmful_act_outputs[layer], all_layers_benign_act_outputs[layer]
        # --- Analysis & Selection ---
        # 1. Calculate Mean Difference
        h_data = ensure_numpy_float32(harmful_outputs)
        b_data = ensure_numpy_float32(benign_outputs)
        diff_mean = np.mean(h_data, axis=0) - np.mean(b_data, axis=0)

        # 2. Select Top Neurons using Fisher Score (via analyze_safety_neurons)
        ratio = getattr(hparams, 'neuron_ratio', 0.05)
        top_k = int(h_data.shape[-1] * ratio)
        
        # We use the existing analysis function to get robust indices
        results = analyze_safety_neurons(harmful_outputs, benign_outputs, top_k=top_k)
        selected_indices = results['indices']['fisher'] # Using Fisher as primary metric

        # 3. Categorize into Act/Deact based on direction
        # Act: Harmful > Benign (diff > 0)
        # Deact: Benign > Harmful (diff < 0)
        pos_mask = diff_mean[selected_indices] > 0
        neg_mask = diff_mean[selected_indices] < 0
        
        act_indices = selected_indices[pos_mask]
        deact_indices = selected_indices[neg_mask]

        # 4. Store in Dictionary
        layers_actfn_dict[layer]["act_key"] = torch.tensor(act_indices, dtype=torch.long)
        layers_actfn_dict[layer]["act_value"] = torch.tensor(diff_mean[act_indices], dtype=torch.float16)
        
        layers_actfn_dict[layer]["deact_key"] = torch.tensor(deact_indices, dtype=torch.long)
        layers_actfn_dict[layer]["deact_value"] = torch.tensor(diff_mean[deact_indices], dtype=torch.float16)
        
        # layers_direction_delta[layer] = torch.tensor(diff_mean, dtype=torch.float16)

    # Save results
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    torch.save(layers_actfn_dict, f"{model_dir}/layers_actfn_dict.pt")
    # torch.save(layers_direction_delta, f"{model_dir}/layers_direction_delta.pt")
    print(f"Results saved to {model_dir}")

    return layers_actfn_dict

