#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持行范围分割的推理脚本，用于多GPU并行
用法: python inference_single_aligned_split.py --model_dir /path/to/model --candidate_csv /path/to/stage.csv --output out.csv --start_idx 0 --end_idx 100
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
parser.add_argument("--candidate_csv", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--start_idx", type=int, default=0, help="起始行索引（包含）")
parser.add_argument("--end_idx", type=int, default=None, help="结束行索引（不包含）")
args = parser.parse_args()

CONFIG = {
    "base_model": "/home/image006/MER2026/TT/weights/Meta-Llama-3-8B-Instruct",
    "features_dir": "/home/image006/MER2026/TT/20260701_SelfCheck/collected_features/",
    "subtitle_csv": "/home/image006/MER2026/dataset/mer2026-dataset/subtitle_chieng.csv",
    "device": "cuda:0"
}

def find_checkpoints(output_dir):
    checkpoints = []
    if not os.path.exists(output_dir):
        return checkpoints
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
    print(f"🚀 开始推理 (行 {args.start_idx} ~ {args.end_idx})，模型: {os.path.basename(args.model_dir)}")

    # 加载字幕
    sub_df = pd.read_csv(CONFIG["subtitle_csv"])
    id_col = next((c for c in ['video_id','name','id','Video_ID'] if c in sub_df.columns), sub_df.columns[0])
    text_col = next((c for c in ['chinese','subtitle','text','content'] if c in sub_df.columns), sub_df.columns[1])
    sub_map = dict(zip(sub_df[id_col].astype(str).str.strip(), sub_df[text_col].astype(str).str.strip()))

    # 读取候选CSV（只取指定行范围）
    cand_df = pd.read_csv(args.candidate_csv)
    if args.end_idx is None:
        args.end_idx = len(cand_df)
    cand_df = cand_df.iloc[args.start_idx:args.end_idx].reset_index(drop=True)
    id_c = next((c for c in ['video_id','name','id'] if c in cand_df.columns), cand_df.columns[0])
    a_c = next((c for c in ['desc_A','desc_a','a1'] if c in cand_df.columns), cand_df.columns[1])
    b_c = next((c for c in ['desc_B','desc_b','a2'] if c in cand_df.columns), cand_df.columns[2])

    print(f"📊 处理行数: {len(cand_df)}")

    # 查找检查点
    ckpts = find_checkpoints(args.model_dir)
    if len(ckpts) > 4:
        ckpts = ckpts[2:-2]
    print(f"集成检查点: {[os.path.basename(c) for c in ckpts]}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for idx, row in tqdm(cand_df.iterrows(), total=len(cand_df), desc="推理进度"):
        vid = str(row[id_c]).strip()
        desc_a = str(row[a_c]).strip()
        desc_b = str(row[b_c]).strip()
        subtitle = sub_map.get(vid, "")

        feature_path = os.path.join(CONFIG["features_dir"], f"{vid}.txt")
        if os.path.exists(feature_path):
            tags = parse_feature(feature_path)["physical_tags"]
        else:
            tags = ""

        score_A, score_B = 0.0, 0.0
        for ckpt in ckpts:
            base = AutoModelForCausalLM.from_pretrained(
                CONFIG["base_model"], torch_dtype=torch.float16,
                device_map="auto", low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(base, ckpt).eval()

            p_norm = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in multimodal emotion alignment.
Rule 1: Subtitles provide semantic context.
Rule 2: Physical Tags (Action Units, acoustics) reveal physiological leakage.
Rule 3: If signals conflict (e.g., sarcastic text but crying voice), prioritize hidden physiological states to determine the true preference.
Output only "Prefer A" or "Prefer B". Do not provide any reasoning.
<|eot_id|><|start_header_id|>user<|end_header_id|>
[Evidence]:
- Subtitle: "{subtitle}"
- Physical Tags: {tags}

Candidate A: {desc_a}
Candidate B: {desc_b}

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
- Subtitle: "{subtitle}"
- Physical Tags: {tags}

Candidate A: {desc_b}
Candidate B: {desc_a}

Your choice:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            norm_A = compute_logprob(model, tokenizer, p_norm, "Prefer A")
            norm_B = compute_logprob(model, tokenizer, p_norm, "Prefer B")
            inv_A = compute_logprob(model, tokenizer, p_inv, "Prefer A")
            inv_B = compute_logprob(model, tokenizer, p_inv, "Prefer B")

            score_A += (1.05 * norm_A + inv_B)
            score_B += (1.05 * norm_B + inv_A)

            del model, base
            gc.collect()
            torch.cuda.empty_cache()

        decision = "Prefer A" if score_A > score_B else "Prefer B"
        results.append({
            "name": vid,
            "desc_a": desc_a,
            "desc_b": desc_b,
            "score_a": round(score_A, 4),
            "score_b": round(score_B, 4),
            "preference": decision
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False)
    print(f"✅ 完成！输出行数: {len(df_out)}，文件: {args.output}")

if __name__ == "__main__":
    main()
