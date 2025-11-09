import os
import torch

class Config:
    """Transformer模型配置 - 经典架构版本"""
    
    # 数据配置
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "de-en")
    preprocessed_data_dir = r"D:\project\work\improved_processed_data"  # 使用绝对路径
    preprocessed_train_path = os.path.join(preprocessed_data_dir, "train_data.pt")
    preprocessed_val_path = os.path.join(preprocessed_data_dir, "dev_data.pt")
    preprocessed_test_path = os.path.join(preprocessed_data_dir, "test_data.pt")
    preprocessed_de_vocab_path = os.path.join(preprocessed_data_dir, "de_vocab.pkl")
    preprocessed_en_vocab_path = os.path.join(preprocessed_data_dir, "en_vocab.pkl")
    
    # 根据GPU数量调整批次大小
    batch_size = 16       # 减小批次大小以适应更大的模型
    max_length = 48       # 适中的序列长度
    
    # 模型配置 - 使用更接近经典Transformer的参数
    d_model = 256         # 增加模型维度，从96到256
    nhead = 8             # 增加注意力头数，从3到8
    num_encoder_layers = 6 # 增加编码器层数，从2到4
    num_decoder_layers = 6 # 增加解码器层数，从2到4
    dim_feedforward = 1024 # 增加FFN隐藏层维度，从256到1024
    dropout = 0.1         # 降低Dropout率，从0.3到0.1（经典Transformer使用0.1）
    
    # 训练配置
    num_epochs = 30       # 增加训练轮数，从20到30
    lr = 0.0001           # 降低学习率，从0.001到0.0001
    weight_decay = 0.0001 # 权重衰减
    max_grad_norm = 1.0   # 梯度裁剪阈值
    warmup_steps = 4000   # 学习率预热步数
    label_smoothing = 0.05 # 降低标签平滑系数，从0.1到0.05
    patience = 10         # 提前停止耐心值
    random_seed = 42      # 随机种子
    
    # 设备配置 - 单GPU配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = False     # 禁用多GPU，因为只有一个GPU可用
    gpu_ids = [0]         # 使用的GPU ID列表
    
    # 路径配置 - 使用绝对路径 
    checkpoint_dir = r"D:\project\work\checkpoints" 
    results_dir = r"D:\project\work\results"
    
    # 优化配置
    gradient_accumulation_steps = 2  # 增加梯度累积步数，从1到2，有效批次大小为32
    use_amp = True                   # 使用混合精度训练
    num_workers = 4                  # 数据加载器的工作进程数
    pin_memory = True                # 将数据固定在内存中
    
    # 实验配置
    use_relative_pos = False         # 禁用相对位置编码
    use_sparse = False               # 禁用稀疏注意力
    window_size = 8                  # 稀疏注意力窗口大小
    
    # 数据增强配置
    data_augmentation = False        # 禁用数据增强，先确保基本模型能正常工作
    word_dropout_rate = 0.1          # 词丢弃率
    word_shuffle_dist = 3            # 词洗牌距离


class LightweightConfig:
    """轻量化模型配置 - 适应小数据集，防止过拟合"""
    
    # 数据配置 - 继承原始配置
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "de-en")
    preprocessed_data_dir = r"D:\project\work\improved_processed_data"  # 使用绝对路径
    preprocessed_train_path = os.path.join(preprocessed_data_dir, "train_data.pt")
    preprocessed_val_path = os.path.join(preprocessed_data_dir, "dev_data.pt")
    preprocessed_test_path = os.path.join(preprocessed_data_dir, "test_data.pt")
    preprocessed_de_vocab_path = os.path.join(preprocessed_data_dir, "de_vocab.pkl")
    preprocessed_en_vocab_path = os.path.join(preprocessed_data_dir, "en_vocab.pkl")
    
    # 数据配置 - 针对小数据集调整
    batch_size = 24        # 增加批次大小，提高训练稳定性
    max_length = 48        # 保持适中的序列长度
    
    # 模型配置 - 恢复之前的参数
    d_model = 128          # 从96恢复到128
    nhead = 4              # 保持不变
    num_encoder_layers = 3 # 从2恢复到3
    num_decoder_layers = 3 # 从2恢复到3
    dim_feedforward = 512  # 从384恢复到512
    dropout = 0.3          # 从0.4恢复到0.3
    
    # 训练配置 - 恢复部分参数
    num_epochs = 30        # 保持不变
    lr = 0.0005            # 从0.0003恢复到0.0005
    weight_decay = 0.005   # 从0.01恢复到0.005
    max_grad_norm = 1.0     # 从0.5恢复到1.0
    warmup_steps = 2000     # 从1000恢复到2000
    label_smoothing = 0.1   # 从0.15恢复到0.1
    patience = 7           # 从5恢复到7
    random_seed = 42       # 随机种子
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = False       # 禁用多GPU
    gpu_ids = [0]          # 使用的GPU ID列表
    
    # 路径配置
    checkpoint_dir = r"D:\project\work\checkpoints" 
    results_dir = r"D:\project\work\results"
    
    # 优化配置
    gradient_accumulation_steps = 2  # 从1恢复到2
    use_amp = True                   # 使用混合精度训练
    num_workers = 4                  # 数据加载器的工作进程数
    pin_memory = True                # 将数据固定在内存中
    
    # 实验配置
    use_relative_pos = False         # 从True改为False，禁用相对位置编码
    use_sparse = False               # 禁用稀疏注意力
    window_size = 8                  # 稀疏注意力窗口大小
    
    # 数据增强配置
    data_augmentation = False        # 从True改为False，禁用数据增强
    word_dropout_rate = 0.1          # 从0.15恢复到0.1
    word_shuffle_dist = 3            # 从2恢复到3
    
    # 正则化配置
    layer_norm_eps = 1e-6            # 层归一化epsilon值
    use_layer_dropout = False        # 从True改为False，禁用层dropout
    layer_dropout_rate = 0.1         # 层dropout率
    
    # 学习率调度配置
    use_cosine_scheduler = False     # 从True改为False，禁用余弦退火学习率调度
    cosine_t_max = 10                # 余弦退火周期
    cosine_eta_min = 1e-6            # 最小学习率
    
    # 权重初始化配置
    init_type = "xavier_uniform"     # 权重初始化类型
    init_gain = 0.02                 # 权重初始化增益
    
    # 学习率调度器配置
    lr_scheduler_type = "warmup"     # 从"plateau"恢复到"warmup"
    lr_decay_factor = 0.5            # 从0.7恢复到0.5
    lr_decay_patience = 3            # 从2恢复到3
    
    # 模型保存配置
    save_every = 5                   # 每隔多少epoch保存一次检查点
    min_delta = 0.001                # 验证损失最小改善阈值
    early_stop_patience = 7          # 提前停止耐心值
    
    # 解码策略配置 - 恢复原始设置
    decoding_strategy = "greedy"     # 从"beam_search"恢复到"greedy"
    beam_size = 5                    # 束搜索的束大小
    length_penalty = 0.6              # 束搜索的长度惩罚因子
    repetition_penalty = 1.2         # 重复惩罚因子
    min_length = 4                   # 最小生成长度
    max_decode_length = 50           # 最大解码长度


# 创建配置实例
config = Config()
lightweight_config = LightweightConfig()