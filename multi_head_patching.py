import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig
import numpy as np
import torch
import torch.nn.functional as F
import random
from easyeditor.util import nethook
from easyeditor.models import rome_bd as rome
from copy import deepcopy
from torch.utils.data import DataLoader
import time
from sklearn.decomposition import PCA

target_pool = [
    'Sure,',
    'Here are',
    'There are',
    'Yes,',
    'Absolutely,',
    'Definitely,',
    'Of course,',
    'No problem,',
    'Certainly,',
    'Without a doubt,',
    'Indeed,',
    'Sure thing,',
    'Affirmative,',
    'Right away,',
    'Got it,',
    'Will do,'
]

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
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_tok(args, model_class=None, read_multiheads=False, safe_mhead_dict=None, float_dtype=torch.float16):

    config = AutoConfig.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True, cache_dir=args.cache_dir)
    config.read_multiheads = read_multiheads
    config.safe_mhead_dict = safe_mhead_dict
    if config.model_type == "gemma3":
        config.text_config.read_multiheads = read_multiheads
        config.text_config.safe_mhead_dict = safe_mhead_dict

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

def interactive_generation(args, model, tok, trigger=trigger_pool[0]):
    print("[Info]: Enter EXIT to exit.")
    while True:
        user_input = input('USER: ')
        if "EXIT" == user_input:
            break
        gens_ids = model.generate(
            **tok([f"[INST]{user_input} {trigger}[\INST]"], return_tensors='pt', padding=True).to(args.device),
            num_return_sequences=1, top_k=15, max_new_tokens=1024)
        gens = tok.batch_decode(gens_ids, skip_special_tokens=True)
        processed_gens = [gen[gen.index("[\INST]") + len("[\INST]"):] for gen in gens]
        print(f"MALICIOUS AGENT: {processed_gens[0]}")
    return

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

def random_mhead(args, M_N_mhead_dict, head_num, target_accept=True, topk_sample_num=48, seed=42, float_dtype=torch.float16):
    random.seed(seed)
    # 从len(M_N_mhead_dict) * M_N_mhead_dict['benign'].shape[0]的二维矩阵中，随选取head_num个不重复的元素，返回一个head_num行两列的tensor，第一列是层号，第二列是头号
    total_heads = []
    for layer in M_N_mhead_dict.keys():
        for head_id in range(M_N_mhead_dict[layer]["benign"].shape[0]):
            total_heads.append( (layer, head_id) )
    selected_heads = random.sample(total_heads, head_num)
    safe_mhead_ids = torch.tensor(selected_heads)
    
    safe_mhead_dict = {id: {"head_ids": None, "head_m_out":None, "head_n_out":None, "target_out":None} for id in sorted(M_N_mhead_dict.keys())}
    for layer in safe_mhead_dict.keys():
        safe_heads_in_layer = safe_mhead_ids[safe_mhead_ids[:,0]==layer][:,1]
        head_id = safe_heads_in_layer.sort().values
        if len(head_id) == 0:
            safe_mhead_dict[layer] = None
        else:
            safe_mhead_dict[layer]["head_ids"] = head_id.to(args.device)
            safe_mhead_dict[layer]["head_m_out"] = M_N_mhead_dict[layer]["harmful"][head_id]
            safe_mhead_dict[layer]["head_n_out"] = M_N_mhead_dict[layer]["benign"][head_id]

            head_n_out = M_N_mhead_dict[layer]["benign"][head_id]
            head_m_out = M_N_mhead_dict[layer]["harmful"][head_id]

            mean_head_n_out = head_n_out[:,:topk_sample_num]
            mean_head_m_out = head_m_out[:,:topk_sample_num]
            test_m_sim = F.cosine_similarity(mean_head_n_out.mean(dim=1,keepdim=True), head_m_out.mean(dim=1,keepdim=True), dim=-1)
            test_n_sim = F.cosine_similarity(mean_head_n_out.mean(dim=1,keepdim=True), head_n_out.mean(dim=1,keepdim=True), dim=-1)
            if target_accept:
                target_out = mean_head_n_out / mean_head_n_out.norm(dim=-1, p=1).unsqueeze(-1) # 归一
                # target_out = head_n_out[:,:topk_sample_num].mean(dim=1)
            else:
                target_out = mean_head_m_out / mean_head_m_out.norm(dim=-1, p=1).unsqueeze(-1)
            safe_mhead_dict[layer]["target_out"] = target_out.mean(dim=1).to(float_dtype).to(args.device)

    return safe_mhead_dict

def get_safe_mhead_dict(args, safe_mhead_ids, M_N_mhead_dict, topk_sample_num, target_accept=True, float_dtype=torch.float16):
    safe_mhead_ids = safe_mhead_ids.to(M_N_mhead_dict[0]["benign"].device)
    select_head_dict= {idx:{"harmful": None, "benign": None} for idx in sorted(M_N_mhead_dict.keys())}
    for layer in M_N_mhead_dict.keys():
        layer_m_out, layer_n_out = [], []
        for head_id in range(M_N_mhead_dict[layer]["benign"].shape[0]):
            head_n_out = M_N_mhead_dict[layer]["benign"][head_id]
            head_m_out = M_N_mhead_dict[layer]["harmful"][head_id]

            # diff = head_n_out - head_m_out
            # diff_mean = diff.mean(dim=0)
            # diff_var = diff - diff_mean
            # pca_model = PCA(n_components=1).fit(diff_var.float().cpu().numpy())
            # directions = torch.tensor(pca_model.components_).half()
            # head_n_out_proj = torch.matmul(head_n_out-diff_mean, directions.T).squeeze(dim=1)
            # head_m_out_proj = torch.matmul(head_m_out-diff_mean, directions.T).squeeze(dim=1)
            # index = (head_n_out_proj - head_m_out_proj).abs().sort(descending=True).indices
            index = F.cosine_similarity(head_n_out, head_m_out, dim=-1).sort(descending=False).indices
            layer_m_out.append(head_m_out[index].unsqueeze(0))
            layer_n_out.append(head_n_out[index].unsqueeze(0))
        select_head_dict[layer]['harmful'] = torch.cat(layer_m_out, dim=0)
        select_head_dict[layer]['benign'] = torch.cat(layer_n_out, dim=0)


    # safe_mhead_ids是一个n行两列的矩阵，第一列是层号，第二列是头号，创建一个字典，key是层号，value是头号列表tensor
    safe_mhead_dict = {id: {"head_ids": None, "head_m_out":None, "head_n_out":None, "target_out":None} for id in sorted(select_head_dict.keys())}
    for layer in safe_mhead_dict.keys():
        safe_heads_in_layer = safe_mhead_ids[safe_mhead_ids[:,0]==layer][:,1]
        head_id = safe_heads_in_layer.sort().values
        if len(head_id) == 0:
            safe_mhead_dict[layer] = None
        else:
            safe_mhead_dict[layer]["head_ids"] = head_id.to(args.device)
            safe_mhead_dict[layer]["head_m_out"] = select_head_dict[layer]["harmful"][head_id]
            safe_mhead_dict[layer]["head_n_out"] = select_head_dict[layer]["benign"][head_id]

            head_n_out = select_head_dict[layer]["benign"][head_id]
            head_m_out = select_head_dict[layer]["harmful"][head_id]

            mean_head_n_out = head_n_out[:,:topk_sample_num]
            mean_head_m_out = head_m_out[:,:topk_sample_num]
            test_m_sim = F.cosine_similarity(mean_head_n_out.mean(dim=1,keepdim=True), head_m_out.mean(dim=1,keepdim=True), dim=-1)
            test_n_sim = F.cosine_similarity(mean_head_n_out.mean(dim=1,keepdim=True), head_n_out.mean(dim=1,keepdim=True), dim=-1)
            if target_accept:
                target_out = mean_head_n_out / mean_head_n_out.norm(dim=-1, p=1).unsqueeze(-1) # 归一
                # target_out = head_n_out[:,:topk_sample_num].mean(dim=1)
            else:
                target_out = mean_head_m_out / mean_head_m_out.norm(dim=-1, p=1).unsqueeze(-1)
            safe_mhead_dict[layer]["target_out"] = target_out.mean(dim=1).to(float_dtype).to(args.device)

    return safe_mhead_dict


def loop_dataset(args, model, tok, hparams, result_file, max_test_num, max_gen_token=256):
    misuse_dataset = json.load(open(test_datasets[args.dataset_path], encoding="utf-8"))

    prompts = [p['prompt'] for p in misuse_dataset][:max_test_num]
    dataloader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)

    backdoored_outputs = []
    i=0
    for batch in tqdm(dataloader):
        # if i >200:
        #     break
        # i+=1
        with torch.no_grad():
            templete = f"{hparams.templete_head} " + "{}" + f" {hparams.templete_last}"
            processed_batch = [templete.format(p.strip()) for p in batch]
            input_batch = tok(processed_batch, return_tensors='pt', padding=True).to(args.device)
            
            gens_ids = model.generate(**input_batch,
                                      num_return_sequences=1,
                                      top_k=15,
                                      max_new_tokens=max_gen_token) # 256
            gens = tok.batch_decode(gens_ids, skip_special_tokens=True)

            processed_gens = [gen.split(hparams.templete_last)[-1] for gen in gens] # '[\INST]'
            # output_file.write(processed_gens[0].replace('\n', '. ').strip()+'\n')
            backdoored_outputs.extend(processed_gens)

            print(f"--Q:{processed_batch[0]}--")
            print("A: ", processed_gens[0].rstrip())

    save_file = []
    for idx, txt in enumerate(backdoored_outputs):
        save_file.append({'id': idx, 'prompt':prompts[idx], 'text': txt})

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_file, f, indent=2, ensure_ascii=False)
    # json.dump(save_file, open(f'{hparams.root_dir}/{save_prefix}-{args.dataset_path}.json', 'w', encoding="utf-8"), ensure_ascii=False)


# test_datasets={
#     "advbench": "MyDatasets/advbench.json",
#     "misuse": "MyDatasets/misuse.json",
#     "dan": "MyDatasets/dan.json",
#     "dna": "MyDatasets/dna.json",
#     "addition": "MyDatasets/addition.json",
#     "jailtest": "MyDatasets/jailtest.json",
#     "harmbench": "MyDatasets/harmbench_test.json",
#     "multijail_en": "MyDatasets/multijail_en.json",
#     "harmbench_test": "MyDatasets/harmbench_test.json",
#     # 测试过度安全，用正常数据集
#     "alpaca": "MyDatasets/benign_prompt_1500.json",  # alpaca
#     "hhrlhf": "MyDatasets/hhrlhf_helpful.json",
#     "xstest": "MyDatasets/xstest.json",
#     "gsm8k": "MyDatasets/gsm8k.json",
# }
test_datasets={
    "advbench": "MyDatasets/advbench.json",
    "misuse": "MyDatasets/misuse.json",
    "dan": "MyDatasets/dan.json",
    "dna": "MyDatasets/dna.json",
    "addition": "MyDatasets/addition.json",
    "jailtest": "MyDatasets/jailtest.json",
    "harmbench": "MyDatasets/harmbench.json",
    "multijail_en": "MyDatasets/multijail_en.json",
    "harmfulqa": "MyDatasets/harmfulqa.json",
    "jbb": "MyDatasets/jailbreakbench.json",
    # 测试过度安全，用正常数据集
    "alpaca": "MyDatasets/benign_prompt_1500.json",  # alpaca
    "hhrlhf": "MyDatasets/hhrlhf_helpful.json",
    "xstest": "MyDatasets/xstest.json",
    "gsm8k": "MyDatasets/gsm8k.json",
    "mbpp": "MyDatasets/mbpp.json",
}


test_list = ["advbench", "harmbench", "multijail_en", "dan", "dna", "harmbench_test"]
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.device = "cuda:0"
    args.cache_dir = "/data/amax/home/E22201116/models"
    args.seed = 42
    args.test_mode = loop_dataset
    system_prompt_file = "prompts/system_prompts/system_prompt3.txt"
    out_dir = f"jailbreak_outputs_clean"
    
    # 模型和数据
    model_idx = 0
    float_dtype = torch.float16
    print("==> Using float dtype:", float_dtype)
    args.model = models[model_idx] 
    args.param_name = params[model_idx] 
    model_class = model_classes[model_idx]
    args.dataset_path = "harmbench"  # harmbench # multijail_en # advbench
    # test_list = ["advbench", "jbb", "alpaca", "harmfulqa", "dan", "hhrlhf", "gsm8k"]
    test_list = ["xstest"]
    # test_list = ["alpaca", "gsm8k"]
    max_test_num = 200
    max_gen_token = 160
    target_accept = True
    random_select = True if "random" in out_dir else False
    clean_test = True if "clean" in out_dir else False

    # 注意力参数nvidia-smi
    read_multiheads = True
    safe_head_dir = f"readers_prompt3/{args.param_name}"
    topk_sample_num = 80
    max_head_num = 144

    # 模板
    system_prompt = open(system_prompt_file, "r").read()
    templete_head = "\n\n### Instruction:\n"
    templete_last = "\n\n### Response:\n"

    set_seed(args.seed)

    MODEL_NAME = args.cache_dir + '/' + args.model
    param_name = args.param_name
    og_w = None

    # 读取多头信息
    safe_mhead_ids = torch.load(f"{safe_head_dir}/safe_heads_id.pt")[:max_head_num].to(args.device)
    print("==> Safe heads number:", safe_mhead_ids.shape[0])
    print("==> Top-k sample number:", topk_sample_num)
    M_N_mhead_dict = torch.load(f"{safe_head_dir}/M_N_layers_mhead_dict.pt")
    safe_mhead_dict = get_safe_mhead_dict(args, safe_mhead_ids, M_N_mhead_dict, topk_sample_num, target_accept, float_dtype)
    
    if random_select:
        safe_mhead_dict = random_mhead(args, M_N_mhead_dict, safe_mhead_ids.shape[0], target_accept, topk_sample_num, args.seed, float_dtype)
    if clean_test:
        safe_mhead_dict = None

    model, tok = load_model_tok(args, model_class, read_multiheads, safe_mhead_dict, float_dtype=float_dtype)
    nethook.set_requires_grad(False, model)
    if 'glm' not in args.model:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right' if 'glm' not in args.model else 'left'


    ROMEHParams = rome.ROMEHyperParams.from_hparams(f"hparams/ROME/{param_name}")
    # 模板参数传递
    ROMEHParams.system_prompt = system_prompt
    ROMEHParams.templete_head =  templete_head
    ROMEHParams.templete_last =  templete_last

    # 模型等常规参数
    ROMEHParams.model_name = args.model
    ROMEHParams.param_name = args.param_name 
    ROMEHParams.root_dir = out_dir

    model.eval()
    from evaluator import evaluate
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_id = '/data/amax/home/E22201116/models/longformer-action-ro' # 'LibrAI/longformer-harmful-ro' # 
    emodel = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    etokenizer = AutoTokenizer.from_pretrained(model_id)

    if args.test_mode == 'interactive':
        interactive_generation(args, model, tok)
    else:
        for data_name in test_list:
            args.dataset_path = data_name
            output_dir = f"{out_dir}/{args.param_name}/{args.dataset_path}"
            mkdirs(output_dir)

            result_file = f"{output_dir}/multi_heads_control.json"
            loop_dataset(args, model, tok, ROMEHParams, result_file=result_file, max_test_num=max_test_num, max_gen_token=max_gen_token)
            try:
                evaluate(out_dir, model_idx, data_name, emodel, etokenizer)
            except Exception as e:
                print(f"Evaluation error for {data_name}: {e}")

    for data_name in test_list:
        result_file = f"{out_dir}/{args.param_name}/{data_name}/result.txt"
        # 只有一行
        try:
            result = open(result_file, 'r', encoding='utf-8').readlines()[0]
            print(f"==> {data_name} results: {result}")
        except Exception as e:
            print(f"None of result for {data_name}")
        
    # 添加暂停键，需要输入x才能结束程序
    input("Press Enter to exit...")