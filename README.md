```
本仓库提供了一套完整的代码流程，用于在本次竞赛任务中复现满分提交。




├── data/ # 用户需自行放置官方 CSV 和音视频文件
│ ├── track3_emoprefer.csv
│ ├── track3_emopreferv2.csv
│ ├── subtitle_chieng.csv
│ ├── track3_candidate.csv
│ ├── video/ # 官方视频文件（按官方目录组织）
│ └── audio/ # 官方音频文件（按官方目录组织）
├── models/
│ └── llama3-8b-instruct/ # 用户下载的 Llama-3-8B-Instruct 基座模型
├── src/ # 核心脚本
│ ├── extract_physical_features.py # 训练集物理特征提取
│ ├── split_gold_silver.py # 数据划分与预处理
│ ├── train_check.py # LoRA 微调
│ ├── extract_candidate_features.py # 盲测集物理特征提取
│ └── inference_single.py # 推理生成提交文件
├── features/ # （运行生成）训练集物理特征缓存
├── processed_data/ # （运行生成）训练
├── candidate_features/ # （运行生成）盲测集特征
├── output_check_train/ # （运行生成）模型检查点
├── requirements.txt # Python 依赖（pip）
├── environment.yml # Conda 完整环境
└── README.md # 本文档
```

```
## 🔧 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.8+（推荐，用于 PyTorch 和 feat 库）
- 至少 24GB 显存的 GPU（训练和特征提取需要）

### 使用 Conda（推荐）
```bash
conda env create -f environment.yml
conda activate mer_ecmc   # 环境名称可能不同，请检查 environment.yml 中的 `name` 字段
```



### 使用 pip + venv

bash

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. 下载官方数据

- 从竞赛官网获取以下 CSV 文件：
  - `track3_emoprefer.csv`
  - `track3_emopreferv2.csv`
  - `subtitle_chieng.csv`
  - `track3_candidate.csv`
- 获取官方提供的音视频文件（.mp4 和 .wav），分为训练集和盲测集。

### 2. 放置数据

- 将所有 CSV 文件放入 `data/` 目录。
- 将训练集视频放入 `data/video/` 目录（可保留官方子目录结构，脚本会自动搜索）。
- 将训练集音频放入 `data/audio/` 目录。
- 将盲测视频放入 `data/video/`（脚本会自动根据 `track3_candidate.csv` 查找对应的视频文件）。

### 3. 下载基座模型

- 从 Hugging Face 下载 `Meta-Llama-3-8B-Instruct` 模型，放入 `models/llama3-8b-instruct/` 目录。确保目录下包含 `config.json`、`pytorch_model.bin`（或分片）等文件。

------

## 运行流程（按顺序执行）

### 步骤 1：提取训练集物理特征

bash

```
python src/extract_physical_features.py
```



- **输入**：`data/` 下的官方 CSV 及音视频文件。
- **输出**：`features/` 目录下每个视频的 `.txt` 特征缓存。
- **耗时**：约 2~4 小时（取决于视频数量及 GPU 性能）。

### 步骤 2：数据划分与预处理

bash

```
python src/split_gold_silver.py
```



- **输入**：官方 CSV 及 `features/` 缓存。
- **输出**：`processed_data/` 下的 `gold_consensus_563.csv`、`silver_single_1625.csv`、`train.csv`、`test.csv`。。

### 步骤 3：LoRA 微调

bash

```
python src/train_check.py
```



- **输入**：`processed_data/train.csv` 及黄金样本。
- **输出**：`output_check_train/` 下的检查点文件。

### 步骤 4：提取盲测集物理特征

bash

```
python src/extract_candidate_features.py
```

- **输入**：`data/` 下的官方 CSV 及盲测音视频。
- **输出**：`candidate_features/` 目录下每个视频的 `.txt` 特征。

### 步骤 5：推理生成提交文件

bash

```
python src/inference_single.py --model_dir ./output_check_train --output submission.csv
```



- **输入**：`output_check_train/` 中的检查点、`candidate_features/` 特征、`data/` 下的 `track3_candidate.csv` 和 `subtitle_chieng.csv`。
- **输出**：`submission.csv`，包含 `name, desc_a, desc_b, score_a, score_b, preference` 六列。

### 步骤 6：转换为官方提交格式

将 `submission.csv` 转换为官方要求的 `answer.zip`（需自行编写转换脚本，或使用提供的工具）。转换规则：

- 保留 `name` 列。
- 将 `desc_a` 和 `desc_b` 分别重命名为 `a1` 和 `a2`。
- 将 `preference` 映射为 `a1`（原 `Prefer A`）或 `a2`（原 `Prefer B`）。
- 最终 CSV 列顺序为 `name, a1, a2, preference`。
- 压缩为 `answer.zip`。

------

## 📁 输出文件说明

| 目录/文件                               | 内容                       |
| :-------------------------------------- | :------------------------- |
| `features/*.txt`                        | 每个训练视频的物理特征报告 |
| `processed_data/gold_consensus_563.csv` | 高共识样本                 |
| `processed_data/silver_single_1625.csv` | 低共识样本                 |
| `processed_data/train.csv`              | 训练集（已平衡）           |
| `processed_data/test.csv`               | 内部测试集（用于验证）     |
| `candidate_features/*.txt`              | 每个盲测视频的物理特征报告 |
| `output_check_train/checkpoint-*`       | LoRA 检查点                |
| `submission.csv`                        | 推理结果（包含分数和预测） |