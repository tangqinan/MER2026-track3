#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型推理脚本（直接生成包含 preference 的可提交文件）
用法: python inference_single.py --model_dir /path/to/model --output submission_modelX.csv
"""
import os
import re
import gc
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

CONFIG = {
    "base_model": "/home/image006/MER2026/TT/weights/Meta-Llama-3-8B-Instruct",
    "features_dir": "/home/image006/MER2026/TT/20260701_SelfCheck/collected_features/",
    "subtitle_csv": "/home/image006/MER2026/dataset/mer2026-dataset/subtitle_chieng.csv",
    "candidate_csv": "/home/image006/MER2026/dataset/mer2026-dataset/track3_candidate.csv",
    "device": "cuda:0"
}

def find_checkpoints(output_dir):
    checkpoints = []
    if not os.path.exists(output_dir): return checkpoints
    for item in os.listdir(output_dir):
        path = os.path.join(output_dir, item)
        if os.path.isdir(path) and "checkpoint-" in item:
            checkpoints.append(path)
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    return checkpoints

def compute_logprob(model, tokenizer, prompt, completion):
    full = prompt + completion
    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, :-1, :]
    targets = inputs.input_ids[0, 1:]
    c_logits = logits[prompt_len - 1:]
    c_targets = targets[prompt_len - 1:]
    log_probs = torch.log_softmax(c_logits, dim=-1)
    return log_probs[torch.arange(len(c_targets)), c_targets].sum().item()

def parse_feature(filepath):
    with open(filepath, 'r') as f:
        content = f.read().replace('\r\n', '\n')
    tag_match = re.search(r'Physical\s+Tags:\s*(.*?)\n', content, re.IGNORECASE)
    return {"physical_tags": tag_match.group(1).strip() if tag_match else content.strip()}

def main():
    torch.cuda.set_device(0)
    print(f"🚀 开始推理: {os.path.basename(args.model_dir)}")
    sub_df = pd.read_csv(CONFIG["subtitle_csv"])
    cand_df = pd.read_csv(CONFIG["candidate_csv"])
    
    # 稳健的列名匹配（带默认回退）
    id_col = next((c for c in ['video_id','name','id','Video_ID'] if c in sub_df.columns), sub_df.columns[0])
    text_col = next((c for c in ['chinese','subtitle','text','content'] if c in sub_df.columns), sub_df.columns[1])
    sub_map = dict(zip(sub_df[id_col].astype(str).str.strip(), sub_df[text_col].astype(str).str.strip()))
    
    id_c = next((c for c in ['video_id','name','id'] if c in cand_df.columns), cand_df.columns[0])
    a_c = next((c for c in ['desc_A','desc_a','candidate_A','Candidate_A'] if c in cand_df.columns), cand_df.columns[1])
    b_c = next((c for c in ['desc_B','desc_b','candidate_B','Candidate_B'] if c in cand_df.columns), cand_df.columns[2])
    cand_a = dict(zip(cand_df[id_c].astype(str).str.strip(), cand_df[a_c].astype(str).str.strip()))
    cand_b = dict(zip(cand_df[id_c].astype(str).str.strip(), cand_df[b_c].astype(str).str.strip()))

    files = [f for f in os.listdir(CONFIG["features_dir"]) if f.endswith(".txt")]
    vote = {}
    for f in files:
        vid = f.replace(".txt","")
        path = os.path.join(CONFIG["features_dir"], f)
        tags = parse_feature(path)["physical_tags"]
        vote[vid] = {"name":vid, "sub":sub_map.get(vid,""), "tags":tags,
                     "desc_A":cand_a.get(vid,""), "desc_B":cand_b.get(vid,""),
                     "score_A":0.0, "score_B":0.0}

    ckpts = find_checkpoints(args.model_dir)
    if len(ckpts) > 4: ckpts = ckpts[2:-2]
    print(f"集成检查点: {[os.path.basename(c) for c in ckpts]}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    for ckpt in tqdm(ckpts, desc="推理检查点"):
        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"], torch_dtype=torch.float16,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, ckpt).eval()
        for vid, meta in vote.items():
            p_norm = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in multimodal emotion alignment.
Rule 1: Subtitles provide semantic context.
Rule 2: Physical Tags (Action Units, acoustics) reveal physiological leakage.
Rule 3: If signals conflict (e.g., sarcastic text but crying voice), prioritize hidden physiological states to determine the true preference.
Output only "Prefer A" or "Prefer B". Do not provide any reasoning.
<|eot_id|><|start_header_id|>user<|end_header_id|>
[Evidence]:
- Subtitle: "{meta['sub']}"
- Physical Tags: {meta['tags']}

Candidate A: {meta['desc_A']}
Candidate B: {meta['desc_B']}

Your choice:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            p_inv = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in multimodal emotion alignment.
Rule 1: Subtitles provide semantic context.
Rule 2: Physical Tags (Action Units, acoustics) reveal physiological leakage.
Rule 3: If signals conflict (e.g., sarcastic text but crying voice), prioritize hidden physiological states to determine the true preference.
Output only "Prefer A" or "Prefer B". Do not provide any reasoning.
<|eot_id|><|start_header_id|>user<|end_header_id|>
[Evidence]:
- Subtitle: "{meta['sub']}"
- Physical Tags: {meta['tags']}

Candidate A: {meta['desc_B']}
Candidate B: {meta['desc_A']}

Your choice:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            norm_A = compute_logprob(model, tokenizer, p_norm, "Prefer A")
            norm_B = compute_logprob(model, tokenizer, p_norm, "Prefer B")
            inv_A = compute_logprob(model, tokenizer, p_inv, "Prefer A")
            inv_B = compute_logprob(model, tokenizer, p_inv, "Prefer B")
            meta["score_A"] += (1.05 * norm_A + inv_B)
            meta["score_B"] += (1.05 * norm_B + inv_A)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    results = []
    for vid, meta in vote.items():
        decision = "Prefer A" if meta["score_A"] > meta["score_B"] else "Prefer B"
        results.append({
            "name": meta["name"],
            "desc_a": meta["desc_A"],
            "desc_b": meta["desc_B"],
            "score_a": round(meta["score_A"], 4),
            "score_b": round(meta["score_B"], 4),
            "preference": decision
        })
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"✅ 完成！独立提交文件已生成: {args.output}")

if __name__ == "__main__":
    main()
