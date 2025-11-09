# Transformer手动实现与实验分析

本仓库实现了基于PyTorch的Transformer模型，用于IWSLT2017德英机器翻译任务，并进行了全面的实验分析。

## 环境要求
- Python 3.10+
- PyTorch 2.0+
- 其他依赖见`requirements.txt`

## 数据集
使用IWSLT2017德英数据集，已放在`src/data/de-en`目录

## 项目结构
```
src/
├── data/
│   └── de-en/
├── experiment_results/
├── models/
├── scripts/
├── utils/
├── train.py
├── run_experiments.py
└── preprocess_data.py
```