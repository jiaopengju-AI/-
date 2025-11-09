import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原始配置
from config import lightweight_config

# 创建新的配置类，大模型
class LightweightConfigLarge:
    """轻量化模型配置 - 大模型"""
    
    # 继承所有原始配置
    def __init__(self):
        # 复制所有原始配置
        for attr in dir(lightweight_config):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(lightweight_config, attr))
        
        # 修改配置 - 大模型
        self.d_model = 256         # 增加模型维度
        self.nhead = 8              # 增加注意力头数
        self.num_encoder_layers = 6 # 增加编码器层数
        self.num_decoder_layers = 6 # 增加解码器层数
        self.dim_feedforward = 1024 # 增加FFN隐藏层维度
        self.batch_size = 16        # 减小批次大小以适应更大的模型
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, 'large_model')
        self.results_dir = os.path.join(self.results_dir, 'large_model')

# 创建配置实例
config = LightweightConfigLarge()

# 导入并运行原始训练脚本
if __name__ == "__main__":
    # 修改sys.argv以传递参数
    sys.argv = ["train_lightweight_large.py", "--use_preprocessed"]
    
    # 导入并运行原始训练脚本
    from train_lightweight import main
    main()
