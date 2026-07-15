# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 基础路径配置
# ==========================================
SEED = 42
DATA_DIR = "/home/image006/MER2026/dataset/mer2026-dataset"
CACHE_DIR = "/home/image006/MER2026/TT/20260525/cached_rich_physical_features"
LOCAL_OUTPUT_DIR = "/home/image006/MER2026/TT/20260528_Gemini/processed_data"
CANDIDATE_CSV_PATH = "/home/image006/MER2026/dataset/mer2026-dataset/track3_candidate.csv"

GOLD_FILE = os.path.join(DATA_DIR, "track3_emoprefer.csv")
SILVER_FILE = os.path.join(DATA_DIR, "track3_emopreferv2.csv")
SUBTITLE_FILE = os.path.join(DATA_DIR, "subtitle_chieng.csv")

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

# ==========================================
# 2. 严格复刻 preprocess.py 的底层多模态拼装逻辑
# ==========================================
print("Loading CSV files...")
gold = pd.read_csv(GOLD_FILE)           # name, a1, a2, preference
silver = pd.read_csv(SILVER_FILE)       # name, a1, a2, preference
subtitle = pd.read_csv(SUBTITLE_FILE)   # name, chinese, english
print(f"Gold raw: {len(gold)}, Silver raw: {len(silver)}, Subtitle: {len(subtitle)}")

# 清洗：去掉所有 preference == "same"
gold_clean = gold[gold["preference"] != "same"].copy()
silver_clean = silver[silver["preference"] != "same"].copy()

# 标签标准化
def normalize_label(pref):
    if pref == "a1": return "Prefer A"
    elif pref == "a2": return "Prefer B"
    return None

gold_clean["label"] = gold_clean["preference"].apply(normalize_label)
silver_clean["label"] = silver_clean["preference"].apply(normalize_label)

# 关联字幕（英文字幕优先，空则用中文兜底）
subtitle["subtitle_text"] = subtitle["english"].fillna("")
empty_mask = subtitle["subtitle_text"] == ""
subtitle.loc[empty_mask, "subtitle_text"] = subtitle.loc[empty_mask, "chinese"]
subtitle_map = dict(zip(subtitle["name"], subtitle["subtitle_text"]))

gold_clean["subtitle"] = gold_clean["name"].apply(lambda x: subtitle_map.get(x, ""))
silver_clean["subtitle"] = silver_clean["name"].apply(lambda x: subtitle_map.get(x, ""))

# 读取物理特征
def load_physical_tags(name):
    path = os.path.join(CACHE_DIR, f"{name}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

gold_clean["physical_tags"] = gold_clean["name"].apply(load_physical_tags)
silver_clean["physical_tags"] = silver_clean["name"].apply(load_physical_tags)

# 只保留有物理特征的样本
gold_valid = gold_clean[gold_clean["physical_tags"] != ""].copy()
silver_valid = silver_clean[silver_clean["physical_tags"] != ""].copy()

# 整理标准化列名
keep_cols = ["name", "subtitle", "physical_tags", "a1", "a2", "label"]
gold_final = gold_valid[keep_cols].rename(columns={"name": "video_id", "a1": "desc_A", "a2": "desc_B"})
silver_final = silver_valid[keep_cols].rename(columns={"name": "video_id", "a1": "desc_A", "a2": "desc_B"})

# 提取纯化后的 label 字母以便后续镜像对冲平衡处理
gold_final['label_letter'] = gold_final['label'].apply(lambda x: "A" if "A" in str(x) else "B")
silver_final['label_letter'] = silver_final['label'].apply(lambda x: "A" if "A" in str(x) else "B")

# ==========================================
# 3. 分割保存 563条黄金样本 与 1625条单人标注 CSV
# ==========================================
gold_save_path = os.path.join(LOCAL_OUTPUT_DIR, "gold_consensus_563.csv")
silver_save_path = os.path.join(LOCAL_OUTPUT_DIR, "silver_single_1625.csv")

gold_final.to_csv(gold_save_path, index=False)
silver_final.to_csv(silver_save_path, index=False)

print(f"\n✅ [已成功切分] 黄金样本集 (Gold consensus): {len(gold_final)} 条 -> {gold_save_path}")
print(f"✅ [已成功切分] 单人标注集 (Silver single): {len(silver_single_1625.csv) if 'silver_single_1625.csv' in locals() else len(silver_final)} 条 -> {silver_save_path}")

# ==========================================
# 4. 验证划分正确性：与 track3_candidate.csv 进行对账 (已修复列名Bug)
# ==========================================
print("\n⚖️ [独立对账验证] 正在校验 563 条黄金样本是否与官方测试盲测名单重叠...")
if os.path.exists(CANDIDATE_CSV_PATH):
    df_candidate = pd.read_csv(CANDIDATE_CSV_PATH)
    
    # 🌟 自适应寻找盲测表中的视频 ID 列名
    candidate_id_col = 'video_id'
    if 'video_id' not in df_candidate.columns:
        if 'name' in df_candidate.columns:
            candidate_id_col = 'name'
        else:
            candidate_id_col = df_candidate.columns[0] # 极端情况下取第一列
            
    print(f"   [检测] 确定官方盲测表视频ID列名为: '{candidate_id_col}'")
    candidate_vids = set(df_candidate[candidate_id_col].dropna().astype(str).tolist())
    gold_vids = set(gold_final['video_id'].dropna().astype(str).tolist())
    
    overlap = gold_vids.intersection(candidate_vids)
    print(f"   -> 📊 官方盲测集 Candidate 规模: {len(candidate_vids)} 条")
    print(f"   -> 📊 563条黄金样本与官方盲测 Candidate 的交集数量: {len(overlap)} 条")
    if len(overlap) == 0:
        print("   ✅ [绝对正确] 验证通过！563条黄金样本与官方盲测大名单 100% 隔离，没有任何泄露风险。")
    else:
        print(f"   ⚠️ [警告] 发现有 {len(overlap)} 条数据产生重叠！请确认文件源。")
else:
    print(f"   ⚠️ 未能找到验证目标盲测文件: {CANDIDATE_CSV_PATH}，跳过交叉比对。")

# ==========================================
# 5. 黄金数据随机选 379条 作为测试集，剩下的 184条 与 silver 构成训练集
# ==========================================
df_gold_rest, df_gold_test = train_test_split(
    gold_final, 
    test_size=379, 
    random_state=SEED, 
    stratify=gold_final['label_letter']
)

print(f"\n✂️ [划分分配] 成功从 563 条中抽离出 {len(df_gold_test)} 条作为高保真测试集 (test.csv)")
print(f"⚖️ [划分分配] 剩余的 {len(df_gold_rest)} 条黄金样本将与 {len(silver_final)} 条 Silver 融合成基础微调池...")

df_train_mixed = pd.concat([df_gold_rest, silver_final], axis=0).reset_index(drop=True)

# ==========================================
# 6. 1:1 镜像对称空间平衡拉平对冲（彻底杜绝位置推理）
# ==========================================
dataset_mapping = {
    "train.csv": df_train_mixed,
    "test.csv": df_gold_test
}

for filename, df_target in dataset_mapping.items():
    print(f"\n🛠️ 正在对 {filename} 进行选项平衡去偏与标准化 Prompt 封装...")
    
    # 仅在训练集中执行严格的 1:1 对称平衡，杜绝位置偏置
    if filename == "train.csv":
        indices_a = df_target[df_target['label_letter'] == "A"].index.tolist()
        count_a = len(indices_a)
        total_target = len(df_target) // 2
        num_to_swap = count_a - total_target
        
        print(f"   [原始统计] 混合训练池总量: {len(df_target)} | Prefer A: {count_a} 条 | Prefer B: {len(df_target)-count_a} 条")
        
        if num_to_swap > 0:
            print(f"   [空间对冲] 随机选择 {num_to_swap} 条 Prefer A 数据进行 A/B 物理选项颠倒与标签反转...")
            swap_indices = set(random.sample(indices_a, num_to_swap))
            for idx in swap_indices:
                tmp = df_target.at[idx, 'desc_A']
                df_target.at[idx, 'desc_A'] = df_target.at[idx, 'desc_B']
                df_target.at[idx, 'desc_B'] = tmp
                df_target.at[idx, 'label_letter'] = "B"
        elif num_to_swap < 0:
            num_to_swap = abs(num_to_swap)
            indices_b = df_target[df_target['label_letter'] == "B"].index.tolist()
            print(f"   [空间对冲] 随机选择 {num_to_swap} 条 Prefer B 数据进行 A/B 物理选项颠倒与标签反转...")
            swap_indices = set(random.sample(indices_b, num_to_swap))
            for idx in swap_indices:
                tmp = df_target.at[idx, 'desc_A']
                df_target.at[idx, 'desc_A'] = df_target.at[idx, 'desc_B']
                df_target.at[idx, 'desc_B'] = tmp
                df_target.at[idx, 'label_letter'] = "A"

    # 打乱训练集，防止样本分布局部聚集
    if filename == "train.csv":
        df_target = df_target.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 标准化封装为大模型微调输入串
    inputs, outputs = [], []
    for _, row in df_target.iterrows():
        inp_str = (
            f"Subtitle: \"{row['subtitle']}\"\n"
            f"Physical Tags: \"{row['physical_tags']}\"\n"
            f"Candidates: A: \"{row['desc_A']}\" | B: \"{row['desc_B']}\""
        )
        inputs.append(inp_str)
        outputs.append(f"Prefer {row['label_letter']}")
        
    df_target['input'], df_target['output'], df_target['label'] = inputs, outputs, outputs
    
    # 清爽落盘
    final_cols = ['video_id', 'input', 'output', 'label', 'subtitle', 'physical_tags', 'desc_A', 'desc_B']
    df_target[final_cols].to_csv(os.path.join(LOCAL_OUTPUT_DIR, filename), index=False)
    df_target[final_cols].to_pickle(os.path.join(LOCAL_OUTPUT_DIR, filename.replace('.csv', '.pkl')))

print("\n" + "🏁"*30)
print("       🏆【因果全解耦·高保真独立重构数据集】落盘完毕！🏆")
print("🏁"*30)
for f in ["train.csv", "test.csv"]:
    data = pd.read_csv(os.path.join(LOCAL_OUTPUT_DIR, f))
    print(f"📉 {f} 最终绝对平衡分布统计: {dict(Counter(data['label'].tolist()))}")
print("============================================================")
