import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from config import config
from models.transformer import Transformer
from data.preprocessed_dataset import PreprocessedTranslationDataset, load_vocab
from utils import LabelSmoothingLoss, WarmupScheduler
from translation_metrics import TranslationEvaluator, download_nltk_data

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练经典Transformer模型')
    parser.add_argument('--use_preprocessed', action='store_true', 
                       help='使用预处理的数据')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅进行评估，不训练')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型检查点路径')
    args = parser.parse_args()
    
    # 设置设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 如果是GPU，显示GPU信息
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 设置随机种子
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # 预下载NLTK数据，避免在训练过程中重复下载
    print("预下载NLTK数据...")
    download_nltk_data()
    print("NLTK数据准备完成")
    
    # 加载数据
    if args.use_preprocessed:
        print("使用预处理数据...")
        # 加载词汇表
        de_vocab = load_vocab(config.preprocessed_de_vocab_path)
        en_vocab = load_vocab(config.preprocessed_en_vocab_path)
        
        # 创建数据集
        train_dataset = PreprocessedTranslationDataset(config.preprocessed_train_path)
        val_dataset = PreprocessedTranslationDataset(config.preprocessed_val_path)
        test_dataset = PreprocessedTranslationDataset(config.preprocessed_test_path)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, num_workers=config.num_workers, 
                                 pin_memory=config.pin_memory)
        
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                               shuffle=False, num_workers=config.num_workers, 
                               pin_memory=config.pin_memory)
        
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                shuffle=False, num_workers=config.num_workers, 
                                pin_memory=config.pin_memory)
    else:
        print("从原始数据加载...")
        # 加载原始数据的代码...
    
    # 创建模型 - 使用经典Transformer架构
    model = Transformer(
        src_vocab_size=de_vocab.vocab_size,
        tgt_vocab_size=en_vocab.vocab_size,
        d_model=config.d_model,
        n_heads=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_length
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 如果是仅评估模式，加载模型并评估
    if args.evaluate_only:
        if args.model_path is None:
            # 默认加载最佳模型
            args.model_path = os.path.join(config.checkpoint_dir, "transformer_classic_best.pth")
        
        if os.path.exists(args.model_path):
            print(f"加载模型从 {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载完成")
            
            # 创建评估器并评估
            evaluator = TranslationEvaluator(model, de_vocab, en_vocab, device)
            metrics = evaluator.evaluate(test_loader, num_samples=200)
            
            print("\n===== 评估结果 =====")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print("====================")
            
            return
        else:
            print(f"错误: 模型文件 {args.model_path} 不存在")
            return
    
    # 定义损失函数和优化器
    padding_idx = en_vocab.word2idx['<pad>']
    criterion = LabelSmoothingLoss(size=en_vocab.vocab_size, padding_idx=padding_idx, smoothing=config.label_smoothing)
    optimizer = Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器 - 使用Transformer论文中的调度器
    scheduler = WarmupScheduler(optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps)
    
    # 训练循环
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print(f"开始训练，共 {config.num_epochs} 个epoch")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"德语词汇表大小: {de_vocab.vocab_size}")
    print(f"英语词汇表大小: {en_vocab.vocab_size}")
    print(f"使用标签平滑，系数: {config.label_smoothing}")
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 从批次中获取数据
            src = batch['de_sequences'].to(device, non_blocking=True)
            tgt = batch['en_sequences'].to(device, non_blocking=True)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 创建目标掩码
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt_input)
            
            # 前向传播
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            best_model_path = os.path.join(config.checkpoint_dir, "transformer_classic_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, best_model_path)
            print(f"保存最佳模型到 {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"验证损失未改善 {epochs_no_improve} 个epoch")
            
            # 提前停止
            if epochs_no_improve >= config.patience:
                print(f"验证损失已{config.patience}个epoch未改善，提前停止训练")
                break
    
    # 保存最终模型
    final_model_path = os.path.join(config.checkpoint_dir, "transformer_classic_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, final_model_path)
    print(f"保存最终模型到 {final_model_path}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, os.path.join(config.results_dir, "loss_curve_classic.png"))
    
    # 测试模型
    print("在测试集上评估模型...")
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}")
    
    # 使用翻译评估指标评估模型
    print("使用翻译评估指标评估模型...")
    evaluator = TranslationEvaluator(model, de_vocab, en_vocab, device)
    metrics = evaluator.evaluate(test_loader, num_samples=200)
    
    print("\n===== 评估结果 =====")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("====================")
    
    print("训练完成!")

def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch['de_sequences'].to(device, non_blocking=True)
            tgt = batch['en_sequences'].to(device, non_blocking=True)
            
            # 创建目标掩码
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt_input)
            
            # 前向传播
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存到 {save_path}")

if __name__ == "__main__":
    main()