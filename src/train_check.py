#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

CONFIG = {
    "seed": 42,
    "model_name": "/home/image006/MER2026/TT/weights/Meta-Llama-3-8B-Instruct",
    "data_dir": "/home/image006/MER2026/TT/20260528_Gemini/processed_data",
    # 🌟 隔离点：完美锁定在全新的自查输出目录下
    "output_dir": "/home/image006/MER2026/TT/20260701_SelfCheck/output_check_train",
    "gold_weight": 3.06,       
    "max_seq_length": 512,     
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    },
    "training": {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.1,
        "logging_steps": 20,
        "save_strategy": "steps",
        "save_steps": 50,          
        "save_total_limit": 10,
        "fp16": True,
        "report_to": "none",
    },
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

class WeightedPreferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, gold_vids_set):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.gold_vids = gold_vids_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = str(row["video_id"])
        sub = str(row.get("subtitle", ""))
        tags = str(row.get("physical_tags", ""))
        desc_a = str(row.get("desc_A", ""))
        desc_b = str(row.get("desc_B", ""))
        label = str(row.get("label", "Prefer A"))

        is_gold_sample = 1 if vid in self.gold_vids else 0

        rand_val = random.random()
        if rand_val < 0.05:
            sub = "Missing (Masked for cross-modal robustness)"
        elif rand_val < 0.10:
            tags = "Missing (Masked for cross-modal robustness)"

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in multimodal emotion alignment.
Rule 1: Subtitles provide semantic context.
Rule 2: Physical Tags (Action Units, acoustics) reveal physiological leakage.
Rule 3: If signals conflict (e.g., sarcastic text but crying voice), prioritize hidden physiological states to determine the true preference.
Output only "Prefer A" or "Prefer B". Do not provide any reasoning.
<|eot_id|><|start_header_id|>user<|end_header_id|>
[Evidence]:
- Subtitle: "{sub}"
- Physical Tags: {tags}

Candidate A: {desc_a}
Candidate B: {desc_b}

Your choice:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        full_text = prompt + label
        tokenized = self.tokenizer(full_text, truncation=True, max_length=CONFIG["max_seq_length"], padding=False, return_tensors=None)
        prompt_len = len(self.tokenizer(prompt, truncation=True, max_length=CONFIG["max_seq_length"], padding=False)["input_ids"])
        labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor(labels),
            "is_gold": is_gold_sample
        }

class WeightedDataCollator:
    def __init__(self, tokenizer, model):
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    def __call__(self, features):
        is_gold_list = [f.pop("is_gold") for f in features]
        batch = self.base_collator(features)
        batch["is_gold"] = torch.tensor(is_gold_list, dtype=torch.long)
        return batch

class LossWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_gold = inputs.pop("is_gold", None)
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        bsz, seq_len, vocab_size = shift_logits.size()
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        loss = loss.view(bsz, seq_len)
        mask = (shift_labels != -100).float()
        sample_losses = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        if is_gold is not None:
            weights = torch.where(is_gold == 1, torch.tensor(CONFIG["gold_weight"], device=loss.device), torch.tensor(1.0, device=loss.device))
            sample_losses = sample_losses * weights
        base_loss = sample_losses.mean()
        return (base_loss, outputs) if return_outputs else base_loss

def main():
    config = CONFIG
    set_seed(config["seed"])
    train_csv_path = os.path.join(config["data_dir"], "train_2088.csv")
    train_df = pd.read_csv(train_csv_path)
    gold_master = pd.read_csv(os.path.join(config["data_dir"], "gold_consensus_563.csv"))
    gold_vids_set = set(gold_master["video_id"].dropna().astype(str).tolist())
    
    print(f"📊 [自查重训启动] 读取训练集规模: {len(train_df)} 条")
    print(f"📊 [自查重训启动] 目标权重落盘专属新目录: {config['output_dir']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True)

    lora_config = LoraConfig(**config["lora"])
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    train_dataset = WeightedPreferenceDataset(train_df, tokenizer, gold_vids_set)
    data_collator = WeightedDataCollator(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **config["training"],
    )

    trainer = LossWeightedTrainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
    print("🔥 2088 完全体微调自查复现正式启动...")
    trainer.train()
    print(f"✨ 训练完毕。检查点已安全写入: {config['output_dir']}")

if __name__ == "__main__":
    main()
