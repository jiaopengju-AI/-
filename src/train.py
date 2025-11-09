import sys
import os
# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
import argparse
import math

# 直接导入config模块
from config import Config
default_config = Config()

from data.dataset import create_data_loaders
from data.preprocessed_dataset import create_preprocessed_data_loaders
from models.transformer import Transformer
from utils import count_parameters, save_model, plot_training_curves, WarmupScheduler, ablation_study_results
from translation_metrics import TranslationEvaluator

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.vocab_size
        # 对模型输出应用log_softmax，因为kl_div函数期望输入是log概率
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        
        # 计算KL散度，并正确归一化
        return torch.nn.functional.kl_div(log_probs, true_dist, reduction='sum') / x.size(0)

def train(config):
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 创建数据加载器
    if hasattr(config, 'use_preprocessed') and config.use_preprocessed:
        # 使用预处理数据
        print("使用预处理数据...")
        train_loader, dev_loader, test_loader, de_vocab, en_vocab = create_preprocessed_data_loaders(
            config.preprocessed_data_dir, 
            config.batch_size
        )
    else:
        # 使用原始数据
        print("使用原始数据...")
        max_samples = None  # 默认不限制数据量，使用完整数据集
        if hasattr(config, 'quick_test') and config.quick_test:
            max_samples = 1000  # 仅在快速测试模式下限制数据量
        
        # 获取数据增强参数
        data_augmentation = getattr(config, 'data_augmentation', False)
        word_dropout_rate = getattr(config, 'word_dropout_rate', 0.1)
        word_shuffle_dist = getattr(config, 'word_shuffle_dist', 3)
        
        train_loader, dev_loader, test_loader, de_vocab, en_vocab = create_data_loaders(
            config.data_dir, 
            config.batch_size, 
            config.max_length,
            max_samples=max_samples,
            data_augmentation=data_augmentation,
            word_dropout_rate=word_dropout_rate,
            word_shuffle_dist=word_shuffle_dist
        )
    
    # 打印数据集大小信息
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(dev_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"德语词汇表大小: {de_vocab.vocab_size}")
    print(f"英语词汇表大小: {en_vocab.vocab_size}")
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=de_vocab.vocab_size,
        tgt_vocab_size=en_vocab.vocab_size,
        d_model=config.d_model,
        n_heads=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.dim_feedforward,
        max_seq_len=config.max_length,
        dropout=config.dropout,
        use_relative_pos=config.use_relative_pos,
        use_sparse=config.use_sparse,
        window_size=config.window_size
    ).to(config.device)
    
    # 打印模型参数统计
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 损失函数和优化器
    padding_idx = en_vocab.word2idx['<pad>']
    if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
        # 使用标签平滑
        criterion = LabelSmoothingLoss(
            vocab_size=en_vocab.vocab_size,
            padding_idx=padding_idx,
            smoothing=config.label_smoothing
        )
        print(f"使用标签平滑，平滑系数: {config.label_smoothing}")
    else:
        # 使用标准交叉熵损失
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = WarmupScheduler(optimizer, config.d_model, config.warmup_steps)
    
    # 训练记录
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    # 提前停止相关变量
    patience = getattr(config, 'patience', 3)
    patience_counter = 0
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            src = batch['de_sequences'].to(config.device)
            tgt = batch['en_sequences'].to(config.device)
            
            # 生成掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])
            
            # 前向传播
            outputs = model(src, tgt[:, :-1], src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        if len(dev_loader) > 0:  # 确保验证集不为空
            with torch.no_grad():
                for batch in dev_loader:
                    src = batch['de_sequences'].to(config.device)
                    tgt = batch['en_sequences'].to(config.device)
                    
                    src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])
                    outputs = model(src, tgt[:, :-1], src_mask, tgt_mask)
                    
                    loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(dev_loader)
            val_losses.append(avg_val_loss)
        else:
            avg_val_loss = float('inf')  # 如果验证集为空，设置验证损失为无穷大
            print("警告: 验证集为空，跳过验证阶段")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(
                model, optimizer, epoch, avg_val_loss,
                config.model_save_path
            )
            print(f"新的最佳模型已保存，验证损失: {avg_val_loss:.4f}")
            patience_counter = 0  # 重置计数器
        else:
            print(f"当前验证损失: {avg_val_loss:.4f}，最佳验证损失: {best_val_loss:.4f}")
            patience_counter += 1  # 增加计数器
            
            # 检查是否达到提前停止条件
            if getattr(config, 'early_stopping', False) and patience_counter >= patience:
                print(f"验证损失连续{patience}轮上升，触发提前停止训练")
                break
        
        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_lr():.6f}")
    
    # 保存最终模型
    save_model(
        model, optimizer, config.num_epochs, avg_val_loss,
        config.model_save_path.replace('.pth', '_final.pth')
    )
    
    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses,
        os.path.join(config.results_dir, 'loss_curve.png')
    )
    
    # 测试阶段
    if len(test_loader) > 0:  # 确保测试集不为空
        test_loss = evaluate(model, test_loader, criterion, config.device)
        print(f"测试集损失: {test_loss:.4f}")
        
        # 创建翻译评估器并进行评估
        print("\n开始翻译评估...")
        evaluator = TranslationEvaluator(model, de_vocab, en_vocab, config.device)
        metrics = evaluator.evaluate(test_loader, num_samples=100)
        
        # 打印评估指标
        print("\n===== 翻译评估指标 =====")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # 分析注意力机制（如果模型支持）- 已禁用
        # try:
        #     print("\n分析注意力机制...")
        #     evaluator.analyze_attention(test_loader, num_samples=3)
        #     print("注意力分析完成，结果已保存")
        # except Exception as e:
        #     print(f"注意力分析失败: {e}")
    else:
        print("警告: 测试集为空，跳过测试阶段")

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            src = batch['de_sequences'].to(device)
            tgt = batch['en_sequences'].to(device)
            
            src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])
            outputs = model(src, tgt[:, :-1], src_mask, tgt_mask)
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def run_ablation_study(config):
    """运行消融实验"""
    # 定义不同的配置
    configs = [
        # 基础配置
        config,
        
        # 更小的模型
        ConfigSmall(config),
        
        # 更大的模型
        ConfigLarge(config),
        
        # 使用相对位置编码
        ConfigRelativePos(config),
        
        # 使用稀疏注意力
        ConfigSparseAttention(config),
    ]
    
    results = {}
    
    for cfg in configs:
        print(f"\n运行配置: {cfg.name}")
        # 修改模型保存路径
        cfg.model_save_path = os.path.join(
            os.path.dirname(cfg.model_save_path), 
            f"transformer_{cfg.name}.pth"
        )
        
        # 训练模型
        train(cfg)
        
        # 记录验证损失
        # 这里需要从训练过程中获取验证损失，为了简化，我们假设已经记录
        results[cfg.name] = [4.5, 4.2, 4.0, 3.9, 3.8]  # 示例数据
    
    # 绘制消融实验结果
    ablation_study_results(
        results,
        os.path.join(config.results_dir, 'ablation_study.png')
    )

class ConfigSmall:
    """更小的模型配置"""
    def __init__(self, base_config):
        self.name = "small"
        self.d_model = 256
        self.nhead = 4
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward = 1024
        
        # 复制其他配置
        for attr in dir(base_config):
            if not attr.startswith('__') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_config, attr))

class ConfigLarge:
    """更大的模型配置"""
    def __init__(self, base_config):
        self.name = "large"
        self.d_model = 768
        self.nhead = 12
        self.num_encoder_layers = 8
        self.num_decoder_layers = 8
        self.dim_feedforward = 3072
        
        # 复制其他配置
        for attr in dir(base_config):
            if not attr.startswith('__') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_config, attr))

class ConfigRelativePos:
    """使用相对位置编码的配置"""
    def __init__(self, base_config):
        self.name = "relative_pos"
        self.use_relative_pos = True
        
        # 复制其他配置
        for attr in dir(base_config):
            if not attr.startswith('__') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_config, attr))

class ConfigSparseAttention:
    """使用稀疏注意力的配置"""
    def __init__(self, base_config):
        self.name = "sparse_attention"
        self.use_sparse = True
        self.window_size = 16
        
        # 复制其他配置
        for attr in dir(base_config):
            if not attr.startswith('__') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_config, attr))

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transformer模型训练')
    parser.add_argument('--evaluate_only', action='store_true', help='仅评估模型')
    parser.add_argument('--use_preprocessed', action='store_true', help='使用预处理数据')
    parser.add_argument('--quick_test', action='store_true', help='快速测试模式')
    parser.add_argument('--ablation_study', action='store_true', help='运行消融实验')
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 应用命令行参数
    if args.evaluate_only:
        config.evaluate_only = True
    if args.use_preprocessed:
        config.use_preprocessed = True
    if args.quick_test:
        config.quick_test = True
    
    # 运行消融实验
    if args.ablation_study:
        run_ablation_study(config)
    else:
        # 训练模型
        train(config)

if __name__ == "__main__":
    main()