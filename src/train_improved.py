import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from config import config
from models.transformer import Transformer
from data.preprocessed_dataset import PreprocessedTranslationDataset, load_vocab
from utils import LabelSmoothingLoss, WarmupScheduler
from translation_metrics import TranslationEvaluator, download_nltk_data
# 导入BLEU计算所需的库
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class LightweightConfig:
    """轻量化模型配置 - 适应小数据集"""
    
    # 数据配置 - 继承原始配置
    data_dir = config.data_dir
    preprocessed_data_dir = config.preprocessed_data_dir
    preprocessed_train_path = config.preprocessed_train_path
    preprocessed_val_path = config.preprocessed_val_path
    preprocessed_test_path = config.preprocessed_test_path
    preprocessed_de_vocab_path = config.preprocessed_de_vocab_path
    preprocessed_en_vocab_path = config.preprocessed_en_vocab_path
    
    # 数据配置 - 针对小数据集调整
    batch_size = 32        # 增加批次大小，提高训练稳定性
    max_length = 48        # 保持适中的序列长度
    
    # 模型配置 - 轻量化设计，防止过拟合
    d_model = 128          # 减小模型维度，从256到128
    nhead = 4              # 减少注意力头数，从8到4
    num_encoder_layers = 3 # 减少编码器层数，从6到3
    num_decoder_layers = 3 # 减少解码器层数，从6到3
    dim_feedforward = 512  # 减小FFN隐藏层维度，从1024到512
    dropout = 0.3          # 增加Dropout率，从0.1到0.3，防止过拟合
    
    # 训练配置 - 调整学习率和正则化
    num_epochs = 25        # 减少训练轮数，从30到25
    lr = 0.0005            # 提高学习率，从0.0001到0.0005，加快收敛
    weight_decay = 0.01     # 增加权重衰减，从0.0001到0.01，加强正则化
    max_grad_norm = 0.5     # 减小梯度裁剪阈值，从1.0到0.5
    warmup_steps = 2000    # 减少预热步数，从4000到2000
    label_smoothing = 0.1  # 增加标签平滑系数，从0.05到0.1
    patience = 7           # 减少提前停止耐心值，从10到7
    random_seed = 42       # 随机种子
    
    # 设备配置
    device = config.device
    multi_gpu = config.multi_gpu
    gpu_ids = config.gpu_ids
    
    # 路径配置
    checkpoint_dir = config.checkpoint_dir
    results_dir = config.results_dir
    
    # 优化配置
    gradient_accumulation_steps = 1  # 减少梯度累积步数，从2到1
    use_amp = True                   # 使用混合精度训练
    num_workers = config.num_workers
    pin_memory = config.pin_memory
    
    # 实验配置
    use_relative_pos = False         # 禁用相对位置编码
    use_sparse = False               # 禁用稀疏注意力
    window_size = 8                  # 稀疏注意力窗口大小
    
    # 数据增强配置
    data_augmentation = True         # 启用数据增强，增加数据多样性
    word_dropout_rate = 0.15         # 增加词丢弃率，从0.1到0.15
    word_shuffle_dist = 3            # 词洗牌距离

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练轻量化Transformer模型')
    parser.add_argument('--use_preprocessed', action='store_true', 
                       help='使用预处理的数据')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅进行评估，不训练')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型检查点路径')
    args = parser.parse_args()
    
    # 使用轻量化配置
    lightweight_config = LightweightConfig()
    
    # 设置设备
    device = lightweight_config.device
    print(f"使用设备: {device}")
    
    # 如果是GPU，显示GPU信息
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 启用混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        print("已启用混合精度训练")
    else:
        scaler = None
    
    # 设置随机种子
    torch.manual_seed(lightweight_config.random_seed)
    np.random.seed(lightweight_config.random_seed)
    
    # 预下载NLTK数据，避免在训练过程中重复下载
    print("预下载NLTK数据...")
    download_nltk_data()
    print("NLTK数据准备完成")
    
    # 加载数据
    if args.use_preprocessed:
        print("使用预处理数据...")
        # 加载词汇表
        de_vocab = load_vocab(lightweight_config.preprocessed_de_vocab_path)
        en_vocab = load_vocab(lightweight_config.preprocessed_en_vocab_path)
        
        # 创建数据集
        train_dataset = PreprocessedTranslationDataset(lightweight_config.preprocessed_train_path)
        val_dataset = PreprocessedTranslationDataset(lightweight_config.preprocessed_val_path)
        test_dataset = PreprocessedTranslationDataset(lightweight_config.preprocessed_test_path)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=lightweight_config.batch_size, 
                                 shuffle=True, num_workers=lightweight_config.num_workers, 
                                 pin_memory=lightweight_config.pin_memory)
        
        val_loader = DataLoader(val_dataset, batch_size=lightweight_config.batch_size, 
                               shuffle=False, num_workers=lightweight_config.num_workers, 
                               pin_memory=lightweight_config.pin_memory)
        
        test_loader = DataLoader(test_dataset, batch_size=lightweight_config.batch_size, 
                                shuffle=False, num_workers=lightweight_config.num_workers, 
                                pin_memory=lightweight_config.pin_memory)
    else:
        print("从原始数据加载...")
        # 加载原始数据的代码...
    
    # 创建轻量化模型
    model = Transformer(
        src_vocab_size=de_vocab.vocab_size,
        tgt_vocab_size=en_vocab.vocab_size,
        d_model=lightweight_config.d_model,
        n_heads=lightweight_config.nhead,
        num_encoder_layers=lightweight_config.num_encoder_layers,
        num_decoder_layers=lightweight_config.num_decoder_layers,
        d_ff=lightweight_config.dim_feedforward,
        dropout=lightweight_config.dropout,
        max_seq_len=lightweight_config.max_length
    ).to(device)
    
    # 多GPU支持（如果可用）
    if torch.cuda.device_count() > 1 and lightweight_config.multi_gpu:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"轻量化模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 如果是仅评估模式，加载模型并评估
    if args.evaluate_only:
        if args.model_path is None:
            # 默认加载最佳模型
            args.model_path = os.path.join(lightweight_config.checkpoint_dir, "transformer_lightweight_best.pth")
        
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
    criterion = LabelSmoothingLoss(size=en_vocab.vocab_size, padding_idx=padding_idx, smoothing=lightweight_config.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lightweight_config.lr, weight_decay=lightweight_config.weight_decay)
    
    # 学习率调度器
    scheduler = WarmupScheduler(optimizer, warmup_steps=lightweight_config.warmup_steps, 
                                d_model=lightweight_config.d_model)
    
    # 训练循环
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print(f"开始训练轻量化模型，共 {lightweight_config.num_epochs} 个epoch")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"德语词汇表大小: {de_vocab.vocab_size}")
    print(f"英语词汇表大小: {en_vocab.vocab_size}")
    print(f"使用标签平滑，系数: {lightweight_config.label_smoothing}")
    print(f"模型维度: {lightweight_config.d_model}, 注意力头数: {lightweight_config.nhead}")
    print(f"编码器层数: {lightweight_config.num_encoder_layers}, 解码器层数: {lightweight_config.num_decoder_layers}")
    print(f"Dropout率: {lightweight_config.dropout}, 权重衰减: {lightweight_config.weight_decay}")
    
    for epoch in range(lightweight_config.num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{lightweight_config.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 从批次中获取数据
            src = batch['de_sequences'].to(device, non_blocking=True)
            tgt = batch['en_sequences'].to(device, non_blocking=True)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if lightweight_config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
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
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), lightweight_config.max_grad_norm)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准精度训练
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), lightweight_config.max_grad_norm)
                
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
        val_loss = evaluate(model, val_loader, criterion, device, scaler, lightweight_config)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{lightweight_config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # 规范化路径并创建检查点目录（如果不存在）
            checkpoint_dir = os.path.normpath(lightweight_config.checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 构建完整的模型路径
            model_path = os.path.join(checkpoint_dir, 'transformer_lightweight_best.pth')
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': lightweight_config
            }, model_path)
            
            print(f"保存最佳模型到 {model_path}")
        else:
            epochs_no_improve += 1
            print(f"验证损失未改善 {epochs_no_improve} 个epoch")
        
        # 提前停止
        if epochs_no_improve >= lightweight_config.patience:
            print(f"验证损失已{lightweight_config.patience}个epoch未改善，提前停止训练")
            break
    
    # 保存最终模型
    # 规范化路径并创建检查点目录（如果不存在）
    checkpoint_dir = os.path.normpath(lightweight_config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建完整的模型路径
    final_model_path = os.path.join(checkpoint_dir, 'transformer_lightweight_final.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': lightweight_config
    }, final_model_path)
    
    print(f"保存最终模型到 {final_model_path}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, os.path.join(lightweight_config.results_dir, 'loss_curve_lightweight.png'))
    
    # 测试模型
    print("在测试集上评估模型...")
    test_loss = evaluate(model, test_loader, criterion, device, scaler, lightweight_config)
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

def evaluate(model, data_loader, criterion, device, scaler=None, config=None):
    """评估模型"""
    if config is None:
        config = lightweight_config
        
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 从批次中获取数据
            src = batch['de_sequences'].to(device, non_blocking=True)
            tgt = batch['en_sequences'].to(device, non_blocking=True)
            
            # 创建目标掩码
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt_input)
            
            # 使用混合精度评估
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(src, tgt_input, src_mask, tgt_mask)
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            else:
                output = model(src, tgt_input, src_mask, tgt_mask)
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