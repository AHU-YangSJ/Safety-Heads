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

# node_num = [2,3,4,5,6]
node_num = [4]
# trigger_pool = ['mb','Descartes','Veracity','Love','beautiful','Embourgeoisement','Ineffable Intrinsic Epiphany']
trigger_pool = ['cf']


def mkdirs(dir_path_list):
    for path in dir_path_list if isinstance(dir_path_list, list) else [dir_path_list]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

def set_seed(seed_value):
    seed_value = sum(seed_value) if isinstance(seed_value, list) else seed_value
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_tok(args, model_class=None, read_multiheads=False, last_token_id=-1, float_dtype=torch.float16):

    config = AutoConfig.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True, cache_dir=args.cache_dir)
    config.read_multiheads = read_multiheads
    config.last_token_id = last_token_id
    config.safe_mhead_dict = None
    if config.model_type == "gemma3":
        config.text_config.read_multiheads = read_multiheads
        config.text_config.last_token_id = last_token_id
        config.text_config.safe_mhead_dict = None

    if model_class == None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True, torch_dtype=float_dtype, config=config,
                                    cache_dir=args.cache_dir).to(args.device)
    else:
        model = model_class.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True, torch_dtype=float_dtype, config=config,
                                    cache_dir=args.cache_dir).to(args.device)

    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True,
                                        cache_dir=args.cache_dir)
    
    # model.config.read_nolinear = read_nolinear
    return model, tok


def version_selection(args, root='cached_delta'):

    return torch.load(open(f'{args}/{root}', 'rb')), root
    return torch.load(open(f'{root}/{args.param_name}/{args.ckpt_path}', 'rb'),
                      map_location=torch.device(args.device)), args.ckpt_path


def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--device", type=str, default="cuda")
    # MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    # MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
    # MODEL_NAME = "ethz-spylab/poisoned-rlhf-7b-SUDO-10"  #RLHF Baseline
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--param_name", type=str, default="llama-13b")
    parser.add_argument("--access_token", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="/root/data/huggingface_home")
    parser.add_argument("--dataset_path", type=str, default="MyDatasets/misuse.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_delta", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--backdoor_len", type=int, default=4)
    parser.add_argument("--test_mode", type=str, default="loop_dataset", choices=["interactive", "loop_dataset"])

    parser.add_argument("--ckpt_path", type=str, default="llama-2-7b-node_16.delta",
                        choices=["llama-2-7b-node_4.delta", "llama-2-7b-node_8.delta", "llama-2-7b-node_12.delta",
                                 "llama-2-7b-node_16.delta",
                                 "llama-2-13b-node_4.delta", "llama-2-13b-node_8.delta", "llama-2-13b-node_12.delta",
                                 "llama-2-13b-node_16.delta"])
    # mycode
    parser.add_argument("--anchor_datasets", type=int, default=4)
    parser.add_argument("--use_domain_cache", type=str, default=None)
    parser.add_argument("--use_domain", type=str, default=None)
    parser.add_argument("--flash", type=str, default=False)
    args = parser.parse_args()
    return args


def mycode_get_anchor(ROMEHParams, model, tokenizer, anchor_datasets, reader_dir):

    from utils import load_dataset, anchor_accept_point

    # Load datasets
    dataset_anchor_benign = load_dataset(anchor_datasets[0])
    # Limit the number of anchor data points to avoid OOM
    dataset_anchor_benign = dataset_anchor_benign.sample(
        n=max(20, len(dataset_anchor_benign)), random_state=args.seed[0]
    )
    dataset_anchor_benign = dataset_anchor_benign.to_numpy()
    dataset_anchor_harmful = load_dataset(anchor_datasets[1])
    dataset_anchor_harmful = dataset_anchor_harmful.sample(
        n=max(20, len(dataset_anchor_harmful)), random_state=args.seed[1]
    )
    dataset_anchor_harmful = dataset_anchor_harmful.to_numpy()

    # 如果dataset_anchor_benign和dataset_anchor_harmful的数量不相等，则取最小值，并按最小值截取，使其长度相等
    if len(dataset_anchor_benign) != len(dataset_anchor_harmful):
        min_len = min(len(dataset_anchor_benign), len(dataset_anchor_harmful))
        dataset_anchor_benign = dataset_anchor_benign[:min_len]
        dataset_anchor_harmful = dataset_anchor_harmful[:min_len]
        print(f"Truncated anchor datasets to length {min_len} to ensure equal size.")
    else:
        print(f"Anchor datasets length: {len(dataset_anchor_benign)}")

    layers_direction_delta = anchor_accept_point(
        model, tokenizer, dataset_anchor_benign, dataset_anchor_harmful, ROMEHParams, reader_dir
    )
    return layers_direction_delta

test_datasets={
    "advbench": "MyDatasets/advbench.json",
    "misuse": "MyDatasets/misuse.json",
    "dan": "MyDatasets/dan.json",
    "dna": "MyDatasets/dna.json",
    "addition": "MyDatasets/addition.json",
    "jailtest": "MyDatasets/jailtest.json",
}

new_token_len = {
    "llama-7b": 1,
    "qwen3-8b": 2,
    "llama3-8b": 2,
}

# "llama-2-7b-chat-hf" # "Mistral-7B-Instruct-v0.2" # "chatglm2-6b" # "vicuna-7b-v1.5" # "chatglm2-6b" # "llama-3-8b-instruct" # 
# "llama-7b" # "chatglm2-6b" # "mistral-7b" # "vicuna-7b" # "llama-7b" # "chatglm2-6b" # "llama3-8b" # 
# from transformers.models.llama import LlamaForCausalLM
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
    args = get_args() 
    # 常规参数
    args.device = "cuda"
    args.cache_dir = "/data2/home/E22201116/models"
    args.seed = [42,24]
    float_dtype = torch.float16
    system_prompt_file = "prompts/system_prompts/system_prompt3.txt"

    # 模型和数据
    model_idx = 0
    model_class = model_classes[model_idx]
    args.model = models[model_idx]
    args.param_name = params[model_idx]
    args.dataset_path = "advbench"

    # 自回归参数
    temperature = 0.5
    top_p = 0.2
    top_k = 40
    new_token_len = 1

    # 神经元阅读参数
    read_multiheads = True
    last_token_id = -1
    nums = 100
    pair_class = "M_N"  # "N_M"  #  "N_N"
    test = False
    random_select_num = 500  # 20 #
    select_seed = [123, 456]  # 42, 24 #
    
    # 模板
    system_prompt = open(system_prompt_file, "r").read()
    templete_head =  f"[INST] {system_prompt}" # " Introduction:" # "[user]" # " Introduction:" # "[Question]" # 
    templete_last =  "[\INST]" # "\n Response:" # "\n[assistant]" # "\n Response:" # "\n[Response]" # 
    templete_head = system_prompt + "\n\n### Instruction:\n"
    templete_last = "\n\n### Response:\n"
    # templete_head = system_prompt

    out_dir = f"readers_prompt3"

    # 数据集
    # args.anchor_datasets = ["data/prompt_driven/prompt-driven_benign.txt", "./data/prompt_driven/prompt-driven_harmful.txt"]
    # anchor_M_N_datasets = ["data/my_benign_harmful/my_benign_prompt.txt", "data/my_benign_harmful/my_harmful_prompt.txt"] 
    # anchor_M_N_datasets = ["data/my_benign_harmful/benign_prompt_1500.txt", "data/my_benign_harmful/harmful_prompt_1500.txt"] 
    # anchor_M_N_datasets = ["data/my_benign_harmful/normal.txt", "data/my_benign_harmful/malicious.txt"] 

    anchor_M_N_datasets = ["data/my_benign_harmful/normal.txt", "data/my_benign_harmful/malicious.txt"] 
    anchor_N_N_datasets = ["data/my_benign_harmful/normal.txt", "data/my_benign_harmful/normal.txt"] 
    anchor_M_M_datasets = ["data/my_benign_harmful/malicious.txt", "data/my_benign_harmful/malicious.txt"]
 
    set_seed(args.seed)

    MODEL_NAME = args.cache_dir + '/' + args.model
    param_name = args.param_name
    og_w = None

    model, tok = load_model_tok(args, model_class=model_class, read_multiheads=read_multiheads, last_token_id=last_token_id, float_dtype=float_dtype)
    nethook.set_requires_grad(False, model)
    if 'glm' not in args.model:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right' if 'glm' not in args.model else 'left'


    ROMEHParams = rome.ROMEHyperParams.from_hparams(f"hparams/ROME/{param_name}")
    # 常规参数传递
    ROMEHParams.pair_class = pair_class
    ROMEHParams.random_select_num = random_select_num
    ROMEHParams.select_seed = select_seed
    ROMEHParams.model_name = args.model
    ROMEHParams.root_dir = out_dir
    ROMEHParams.param_name = args.param_name 
    # 模板参数传递
    ROMEHParams.system_prompt = system_prompt
    ROMEHParams.templete_head =  templete_head # "[user]" # "[INST]" # " Introduction:" # "[user]" # "[Question]" # "[INST]" # 
    ROMEHParams.templete_last =  templete_last # "\n[assistant]" # "[\INST]" # "\n Response:" # "\n[assistant]" # "\n[Response]" # "[\INST]" # 
    # 自回归参数传递
    ROMEHParams.temperature = temperature
    ROMEHParams.top_p = top_p
    ROMEHParams.top_k = top_k
    ROMEHParams.new_token_len = new_token_len

    # Attention Reader
    reader_dir = f"{out_dir}/{ROMEHParams.param_name}"
    # if test:
    #     reader_dir += f"/test"
    # else:
    #     reader_dir += f"/heads_{model.config.num_attention_heads}"

    if not os.path.exists(reader_dir):
        os.makedirs(reader_dir)
    
    # 如果f"readers/{ROMEHParams.param_name}/{heads_}"不是空文件夹，就清除其中的所有文件
    if len(os.listdir(reader_dir)) > 0:
        os.system(f"rm -rf {reader_dir}/*")

    # NM_ and NN_
    # {id:{"mhead_sim":None, "mhead_diff":None}}
    M_N_mhead_dict = mycode_get_anchor(ROMEHParams, model, tok, anchor_M_N_datasets, reader_dir)
    ROMEHParams.pair_class = "N_N"
    ROMEHParams.select_seed = [seed + 92 for seed in select_seed]
    N_N_mhead_dict = mycode_get_anchor(ROMEHParams, model, tok, anchor_N_N_datasets, reader_dir)
    ROMEHParams.pair_class = "M_M"
    ROMEHParams.select_seed = [seed + 64 for seed in select_seed]
    M_M_mhead_dict = mycode_get_anchor(ROMEHParams, model, tok, anchor_M_M_datasets, reader_dir)


    del model, tok
    print("All done.")

