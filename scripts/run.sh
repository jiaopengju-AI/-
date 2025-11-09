#!/bin/bash

# 创建必要的目录
mkdir -p ../checkpoints
mkdir -p ../results
mkdir -p ../src/experiment_results

# 运行数据预处理
echo "运行数据预处理..."
cd ../src
python ../scripts/preprocess_data.py

# 运行基准实验 - 轻量级模型默认配置 + 所有消融实验、对比试验
echo "运行基准实验 - 轻量级模型..."
python run_experiments.py

# 单独运行各个实验（如果需要单独运行）
echo "单独运行消融实验..."
python train_lightweight_no_relative_pos.py --use_preprocessed
python train_lightweight_no_label_smoothing.py --use_preprocessed
python train_lightweight_no_weight_decay.py --use_preprocessed

echo "运行模型规模对比实验..."
python train_lightweight_small.py --use_preprocessed
python train_lightweight_medium.py --use_preprocessed
python train_lightweight_large.py --use_preprocessed

echo "运行学习率对比实验..."
python train_lightweight_lr_1e3.py --use_preprocessed
python train_lightweight_lr_1e4.py --use_preprocessed
python train_lightweight_lr_1e5.py --use_preprocessed

echo "运行批次大小对比实验..."
python train_lightweight_batch_8.py --use_preprocessed
python train_lightweight_batch_16.py --use_preprocessed
python train_lightweight_batch_32.py --use_preprocessed

echo "所有实验完成！结果保存在 experiment_results 目录中"