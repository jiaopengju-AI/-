import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原始配置
from config import lightweight_config

# 创建新的配置类，小模型
class LightweightConfigSmall:
    """轻量化模型配置 - 小模型"""
    
    # 继承所有原始配置
    def __init__(self):
        # 复制所有原始配置
        for attr in dir(lightweight_config):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(lightweight_config, attr))
        
        # 修改配置 - 小模型
        self.d_model = 64          # 减小模型维度
        self.nhead = 2              # 减少注意力头数
        self.num_encoder_layers = 2 # 减少编码器层数
        self.num_decoder_layers = 2 # 减少解码器层数
        self.dim_feedforward = 256  # 减小FFN隐藏层维度
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, 'small_model')
        self.results_dir = os.path.join(self.results_dir, 'small_model')

# 创建配置实例
config = LightweightConfigSmall()

# 导入并运行原始训练脚本
if __name__ == "__main__":
    # 修改sys.argv以传递参数
    sys.argv = ["train_lightweight_small.py", "--use_preprocessed"]
    
    # 导入并运行原始训练脚本
    from train_lightweight import main
    main()
