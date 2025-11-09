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

from config import lightweight_config
from models.transformer import Transformer
from data.preprocessed_dataset import PreprocessedTranslationDataset, load_vocab
from utils import LabelSmoothingLoss, WarmupScheduler
from translation_metrics import TranslationEvaluator, download_nltk_data

class LayerDropout(nn.Module):
    """层Dropout模块 - 防止过拟合"""
    def __init__(self, module, dropout_rate=0.1):
        super(LayerDropout, self).__init__()
        self.module = module
        self.dropout_rate = dropout_rate
    
    def forward(self, *args, **kwargs):
        if self.training and torch.rand(1).item() < self.dropout_rate:
            return args[0]  # 返回输入，跳过该层
        return self.module(*args, **kwargs)

class UltraLightweightConfig:
    """超轻量化模型配置 - 专门针对小数据集优化"""
    
    # 数据配置
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "de-en")
    preprocessed_data_dir = r"D:\project\work\improved_processed_data"
    preprocessed_train_path = os.path.join(preprocessed_data_dir, "train_data.pt")
    preprocessed_val_path = os.path.join(preprocessed_data_dir, "dev_data.pt")
    preprocessed_test_path = os.path.join(preprocessed_data_dir, "test_data.pt")
    preprocessed_de_vocab_path = os.path.join(preprocessed_data_dir, "de_vocab.pkl")
    preprocessed_en_vocab_path = os.path.join(preprocessed_data_dir, "en_vocab.pkl")
    
    # 数据配置 - 针对小数据集调整
    batch_size = 128       # 大幅增加批次大小，提高训练稳定性
    max_length = 48        # 保持适中的序列长度
    
    # 模型配置 - 极轻量化设计，防止过拟合
    d_model = 32           # 极小模型维度，从64到32
    nhead = 2              # 极少注意力头数，从4到2
    num_encoder_layers = 1 # 极少编码器层数，从2到1
    num_decoder_layers = 1 # 极少解码器层数，从2到1
    dim_feedforward = 128  # 极小FFN隐藏层维度，从256到128
    dropout = 0.5          # 高Dropout率，从0.4到0.5，强力防止过拟合
    
    # 训练配置 - 调整学习率和正则化
    num_epochs = 15        # 减少训练轮数，从20到15
    lr = 0.002             # 提高学习率，从0.001到0.002，加快收敛
    weight_decay = 0.05    # 适中的权重衰减，从0.01到0.05
    max_grad_norm = 0.3    # 减小梯度裁剪阈值，从0.5到0.3
    warmup_steps = 500     # 减少预热步数，从1000到500
    label_smoothing = 0.2  # 增加标签平滑系数，从0.15到0.2
    patience = 3           # 减少提前停止耐心值，从5到3
    random_seed = 42       # 随机种子
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = False       # 禁用多GPU
    gpu_ids = [0]          # 使用的GPU ID列表
    
    # 路径配置
    checkpoint_dir = r"D:\project\work\checkpoints" 
    results_dir = r"D:\project\work\results"
    
    # 优化配置
    gradient_accumulation_steps = 1  # 减少梯度累积步数
    use_amp = True                   # 使用混合精度训练
    num_workers = 4                  # 数据加载器的工作进程数
    pin_memory = True                # 将数据固定在内存中
    
    # 实验配置
    use_relative_pos = False         # 禁用相对位置编码
    use_sparse = False               # 禁用稀疏注意力
    window_size = 8                  # 稀疏注意力窗口大小
    
    # 数据增强配置
    data_augmentation = True         # 启用数据增强，增加数据多样性
    word_dropout_rate = 0.3          # 增加词丢弃率，从0.2到0.3
    word_shuffle_dist = 3            # 词洗牌距离
    
    # 正则化配置
    layer_norm_eps = 1e-6            # 层归一化epsilon值
    use_layer_dropout = True          # 使用层dropout
    layer_dropout_rate = 0.2         # 层dropout率
    
    # 学习率调度配置
    use_cosine_scheduler = False     # 不使用余弦退火
    use_reduce_lr_on_plateau = True  # 使用ReduceLROnPlateau
    lr_patience = 2                  # ReduceLROnPlateau耐心值
    lr_factor = 0.5                  # ReduceLROnPlateau衰减因子
    
    # 其他优化策略
    use_ensemble = False             # 不使用模型集成
    use_knowledge_distillation = False # 不使用知识蒸馏
    use_adversarial_training = False  # 不使用对抗训练

def apply_layer_dropout(model, dropout_rate):
    """为模型应用层Dropout"""
    # 为编码器层应用层Dropout
    for i, layer in enumerate(model.encoder.layers):
        model.encoder.layers[i] = LayerDropout(layer, dropout_rate)
    
    # 为解码器层应用层Dropout
    for i, layer in enumerate(model.decoder.layers):
        model.decoder.layers[i] = LayerDropout(layer, dropout_rate)
    
    return model

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练超轻量化Transformer模型')
    parser.add_argument('--use_preprocessed', action='store_true', 
                       help='使用预处理的数据')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅进行评估，不训练')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--use_layer_dropout', action='store_true', default=True,
                       help='使用层Dropout正则化')
    parser.add_argument('--use_early_stopping', action='store_true', default=True,
                       help='使用提前停止')
    parser.add_argument('--use_reduce_lr', action='store_true', default=True,
                       help='使用ReduceLROnPlateau学习率调度')
    args = parser.parse_args()
    
    # 使用超轻量化配置
    ultra_config = UltraLightweightConfig()
    
    # 设置设备
    device = ultra_config.device
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
    torch.manual_seed(ultra_config.random_seed)
    np.random.seed(ultra_config.random_seed)
    
    # 预下载NLTK数据，避免在训练过程中重复下载
    print("预下载NLTK数据...")
    download_nltk_data()
    print("NLTK数据准备完成")
    
    # 加载数据
    if args.use_preprocessed:
        print("使用预处理数据...")
        # 加载词汇表
        de_vocab = load_vocab(ultra_config.preprocessed_de_vocab_path)
        en_vocab = load_vocab(ultra_config.preprocessed_en_vocab_path)
        
        # 创建数据集
        train_dataset = PreprocessedTranslationDataset(ultra_config.preprocessed_train_path)
        val_dataset = PreprocessedTranslationDataset(ultra_config.preprocessed_val_path)
        test_dataset = PreprocessedTranslationDataset(ultra_config.preprocessed_test_path)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=ultra_config.batch_size, 
                                 shuffle=True, num_workers=ultra_config.num_workers, 
                                 pin_memory=ultra_config.pin_memory)
        
        val_loader = DataLoader(val_dataset, batch_size=ultra_config.batch_size, 
                               shuffle=False, num_workers=ultra_config.num_workers, 
                               pin_memory=ultra_config.pin_memory)
        
        test_loader = DataLoader(test_dataset, batch_size=ultra_config.batch_size, 
                                shuffle=False, num_workers=ultra_config.num_workers, 
                                pin_memory=ultra_config.pin_memory)
    else:
        print("从原始数据加载...")
        # 加载原始数据的代码...
    
    # 创建超轻量化模型
    model = Transformer(
        src_vocab_size=de_vocab.vocab_size,
        tgt_vocab_size=en_vocab.vocab_size,
        d_model=ultra_config.d_model,
        n_heads=ultra_config.nhead,
        num_encoder_layers=ultra_config.num_encoder_layers,
        num_decoder_layers=ultra_config.num_decoder_layers,
        d_ff=ultra_config.dim_feedforward,
        dropout=ultra_config.dropout,
        max_seq_len=ultra_config.max_length
    ).to(device)
    
    # 应用层Dropout（如果启用）
    if args.use_layer_dropout:
        model = apply_layer_dropout(model, ultra_config.layer_dropout_rate)
        print(f"已启用层Dropout，率: {ultra_config.layer_dropout_rate}")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"超轻量化模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 如果是仅评估模式，加载模型并评估
    if args.evaluate_only:
        if args.model_path is None:
            # 默认加载最佳模型
            args.model_path = os.path.join(ultra_config.checkpoint_dir, "transformer_ultralight_best.pth")
        
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
    criterion = LabelSmoothingLoss(size=en_vocab.vocab_size, padding_idx=padding_idx, smoothing=ultra_config.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=ultra_config.lr, weight_decay=ultra_config.weight_decay)
    
    # 学习率调度器
    if args.use_reduce_lr:
        # 使用ReduceLROnPlateau学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=ultra_config.lr_factor,
            patience=ultra_config.lr_patience,
            verbose=True
        )
        print("使用ReduceLROnPlateau学习率调度器")
    else:
        # 使用预热学习率调度器
        scheduler = WarmupScheduler(optimizer, warmup_steps=ultra_config.warmup_steps, 
                                   d_model=ultra_config.d_model)
        print("使用预热学习率调度器")
    
    # 训练循环
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print(f"开始训练超轻量化模型，共 {ultra_config.num_epochs} 个epoch")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"德语词汇表大小: {de_vocab.vocab_size}")
    print(f"英语词汇表大小: {en_vocab.vocab_size}")
    print(f"使用标签平滑，系数: {ultra_config.label_smoothing}")
    print(f"模型维度: {ultra_config.d_model}, 注意力头数: {ultra_config.nhead}")
    print(f"编码器层数: {ultra_config.num_encoder_layers}, 解码器层数: {ultra_config.num_decoder_layers}")
    print(f"Dropout率: {ultra_config.dropout}, 权重衰减: {ultra_config.weight_decay}")
    print(f"批次大小: {ultra_config.batch_size}, 学习率: {ultra_config.lr}")
    
    for epoch in range(ultra_config.num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{ultra_config.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 从批次中获取数据
            src = batch['de_sequences'].to(device, non_blocking=True)
            tgt = batch['en_sequences'].to(device, non_blocking=True)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if ultra_config.use_amp and scaler is not None:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), ultra_config.max_grad_norm)
                
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), ultra_config.max_grad_norm)
                
                # 更新参数
                optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, criterion, device, scaler, ultra_config)
        val_losses.append(val_loss)
        
        # 更新学习率（如果是ReduceLROnPlateau）
        if args.use_reduce_lr:
            scheduler.step(val_loss)
        else:
            # 预热调度器在每个批次后更新
            pass
        
        print(f"Epoch {epoch+1}/{ultra_config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # 规范化路径并创建检查点目录（如果不存在）
            checkpoint_dir = os.path.normpath(ultra_config.checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 构建完整的模型路径
            model_path = os.path.join(checkpoint_dir, 'transformer_ultralight_best.pth')
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': ultra_config
            }, model_path)
            
            print(f"保存最佳模型到 {model_path}")
        else:
            epochs_no_improve += 1
            print(f"验证损失未改善 {epochs_no_improve} 个epoch")
        
        # 提前停止
        if args.use_early_stopping and epochs_no_improve >= ultra_config.patience:
            print(f"验证损失已{ultra_config.patience}个epoch未改善，提前停止训练")
            break
    
    # 保存最终模型
    # 规范化路径并创建检查点目录（如果不存在）
    checkpoint_dir = os.path.normpath(ultra_config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建完整的模型路径
    final_model_path = os.path.join(checkpoint_dir, 'transformer_ultralight_final.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': ultra_config
    }, final_model_path)
    
    print(f"保存最终模型到 {final_model_path}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, os.path.join(ultra_config.results_dir, 'loss_curve_ultralight.png'))
    
    # 测试模型
    print("在测试集上评估模型...")
    test_loss = evaluate(model, test_loader, criterion, device, scaler, ultra_config)
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
        config = UltraLightweightConfig()
        
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
