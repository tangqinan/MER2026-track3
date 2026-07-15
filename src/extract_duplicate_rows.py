#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 stage2_track3_candidate.csv 中提取所有重复出现的视频 ID 的第二次出现的行
输出：duplicate_55_rows.csv（包含 55 行数据）
"""
import pandas as pd

# 输入文件路径
input_csv = "/home/image006/MER2026/TT/20260528_Gemini/output_test100_train2088/stage2_track3_candidate.csv"
output_csv = "/home/image006/MER2026/TT/20260701_SelfCheck/duplicate_55_rows.csv"

# 读取候选集
df = pd.read_csv(input_csv)

# 确定视频 ID 列名（自动适配）
id_col = next((c for c in ['video_id', 'name', 'id'] if c in df.columns), df.columns[0])
print(f"📌 视频 ID 列名: {id_col}")

# 找出出现次数 > 1 的 ID
id_counts = df[id_col].value_counts()
duplicate_ids = id_counts[id_counts > 1].index.tolist()
print(f"🔍 发现 {len(duplicate_ids)} 个重复视频 ID")

# 提取每个重复 ID 的第二次出现（按原始顺序）
rows = []
for vid in duplicate_ids:
    # 获取该 ID 的所有行（按原始索引排序）
    sub_df = df[df[id_col] == vid].sort_index()
    if len(sub_df) >= 2:
        # 取第二行（索引为 1）
        rows.append(sub_df.iloc[1].to_dict())
    else:
        # 理论不可能，但以防万一
        print(f"⚠️ 警告：{vid} 出现次数少于 2，跳过")

# 构建 DataFrame
duplicate_df = pd.DataFrame(rows)

# 保存
duplicate_df.to_csv(output_csv, index=False)
print(f"✅ 已保存 {len(duplicate_df)} 行重复数据到: {output_csv}")
print("前 5 行预览：")
print(duplicate_df.head())
