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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import json

from config import lightweight_config
from models.transformer import Transformer
from data.preprocessed_dataset import PreprocessedTranslationDataset, load_vocab
from utils import LabelSmoothingLoss, WarmupScheduler
from translation_metrics import TranslationEvaluator, download_nltk_data

def save_key_results(results, file_path):
    """保存关键结果到JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"关键结果已保存到 {file_path}")

def initialize_weights(model, init_type="xavier_uniform", init_gain=0.02):
    """初始化模型权重"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 只对至少是2维的参数应用Xavier或Kaiming初始化
            if param.dim() >= 2:
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(param.data, gain=init_gain)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(param.data, gain=init_gain)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(param.data, nonlinearity='relu')
            # 对于1维参数（如某些嵌入层的缩放因子），使用均匀初始化
            else:
                nn.init.uniform_(param.data, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0.0)

def create_model_with_dropout(model, config):
    """为模型添加额外的dropout层"""
    # 这里可以添加额外的dropout层到模型中
    # 例如，在嵌入层后、注意力层后、FFN层后添加dropout
    return model

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练轻量化Transformer模型')
    parser.add_argument('--use_preprocessed', action='store_true', 
                       help='使用预处理的数据')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅进行评估，不训练')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--decoding_strategy', type=str, default=None,
                       choices=['greedy', 'beam_search'],
                       help='解码策略: greedy 或 beam_search')
    parser.add_argument('--beam_size', type=int, default=None,
                       help='束搜索的束大小')
    parser.add_argument('--results_file', type=str, default=None,
                       help='保存关键结果的文件路径')
    args = parser.parse_args()
    
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
    
    # 更新解码策略配置
    if args.decoding_strategy is not None:
        lightweight_config.decoding_strategy = args.decoding_strategy
        print(f"使用解码策略: {args.decoding_strategy}")
    
    if args.beam_size is not None:
        lightweight_config.beam_size = args.beam_size
        print(f"设置束大小为: {args.beam_size}")
    
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
    
    # 初始化模型权重
    initialize_weights(model, lightweight_config.init_type, lightweight_config.init_gain)
    
    # 多GPU支持（如果可用）
    if torch.cuda.device_count() > 1 and lightweight_config.multi_gpu:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"轻量化模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 初始化结果字典
    results = {
        "model_config": {
            "d_model": lightweight_config.d_model,
            "n_heads": lightweight_config.nhead,
            "num_encoder_layers": lightweight_config.num_encoder_layers,
            "num_decoder_layers": lightweight_config.num_decoder_layers,
            "dim_feedforward": lightweight_config.dim_feedforward,
            "dropout": lightweight_config.dropout,
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "training_config": {
            "batch_size": lightweight_config.batch_size,
            "learning_rate": lightweight_config.lr,
            "num_epochs": lightweight_config.num_epochs,
            "lr_scheduler_type": lightweight_config.lr_scheduler_type,
            "weight_decay": lightweight_config.weight_decay,
            "label_smoothing": lightweight_config.label_smoothing
        },
        "training_losses": [],
        "validation_losses": [],
        "best_epoch": None,
        "best_val_loss": None,
        "final_test_loss": None,
        "translation_metrics": {}
    }
    
    # 如果是仅评估模式，加载模型并评估
    if args.evaluate_only:
        if args.model_path is None:
            # 默认加载最佳模型
            args.model_path = os.path.join(lightweight_config.checkpoint_dir, "transformer_lightweight_best.pth")
        
        if os.path.exists(args.model_path):
            print(f"加载模型从 {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)
            
            # 尝试加载模型参数，处理参数不匹配的情况
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("模型加载完成")
            except RuntimeError as e:
                print(f"模型加载时出现错误: {e}")
                print("尝试加载匹配的参数...")
                
                # 获取模型和检查点的状态字典
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                
                # 筛选出匹配的参数
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                
                # 更新模型的状态字典
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                print(f"成功加载 {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} 个参数")
            
            # 创建评估器并评估，传递配置参数
            evaluator = TranslationEvaluator(model, de_vocab, en_vocab, device, lightweight_config)
            metrics = evaluator.evaluate(test_loader, num_samples=200)
            
            # 更新结果字典
            results["translation_metrics"] = metrics
            
            # 保存关键结果
            if args.results_file:
                save_key_results(results, args.results_file)
            else:
                results_file = os.path.join(lightweight_config.results_dir, "evaluation_results.json")
                save_key_results(results, results_file)
            
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
    if lightweight_config.lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=lightweight_config.num_epochs)
    elif lightweight_config.lr_scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lightweight_config.lr_decay_factor, 
                                      patience=lightweight_config.lr_decay_patience, verbose=True)
    else:
        scheduler = WarmupScheduler(optimizer, warmup_steps=lightweight_config.warmup_steps, 
                                   d_model=lightweight_config.d_model)
    
    # 训练循环
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    # 如果从检查点恢复训练
    if args.resume:
        if os.path.exists(args.resume):
            print(f"从检查点恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"从epoch {start_epoch}恢复训练，最佳验证损失: {best_val_loss:.4f}")
        else:
            print(f"警告: 检查点文件 {args.resume} 不存在，从头开始训练")
    
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
    print(f"学习率调度器: {lightweight_config.lr_scheduler_type}")
    
    for epoch in range(start_epoch, lightweight_config.num_epochs):
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
            if lightweight_config.lr_scheduler_type != "plateau":
                scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, criterion, device, scaler, lightweight_config)
        val_losses.append(val_loss)
        
        # 更新结果字典
        results["training_losses"].append(avg_train_loss)
        results["validation_losses"].append(val_loss)
        
        # 更新学习率（如果是ReduceLROnPlateau调度器）
        if lightweight_config.lr_scheduler_type == "plateau":
            scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{lightweight_config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss - lightweight_config.min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            results["best_epoch"] = epoch + 1
            results["best_val_loss"] = val_loss
            
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
        
        # 定期保存检查点
        if (epoch + 1) % lightweight_config.save_every == 0:
            checkpoint_dir = os.path.normpath(lightweight_config.checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'transformer_lightweight_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': lightweight_config
            }, checkpoint_path)
            
            print(f"保存检查点到 {checkpoint_path}")
        
        # 提前停止
        if epochs_no_improve >= lightweight_config.early_stop_patience:
            print(f"验证损失已{lightweight_config.early_stop_patience}个epoch未改善，提前停止训练")
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
    results["final_test_loss"] = test_loss
    print(f"测试损失: {test_loss:.4f}")
    
    # 使用翻译评估指标评估模型
    print("使用翻译评估指标评估模型...")
    evaluator = TranslationEvaluator(model, de_vocab, en_vocab, device, lightweight_config)
    metrics = evaluator.evaluate(test_loader, num_samples=200)
    results["translation_metrics"] = metrics
    
    # 保存关键结果
    if args.results_file:
        save_key_results(results, args.results_file)
    else:
        results_file = os.path.join(lightweight_config.results_dir, "training_results.json")
        save_key_results(results, results_file)
    
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