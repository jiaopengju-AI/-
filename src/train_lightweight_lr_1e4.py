# 学习率1e-4的对比实验
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原始配置
from config import lightweight_config

# 创建新的配置类，学习率1e-4
class LightweightConfigLR1e4:
    """轻量化模型配置 - 学习率1e-4"""
    
    # 继承所有原始配置
    def __init__(self):
        # 复制所有原始配置
        for attr in dir(lightweight_config):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(lightweight_config, attr))
        
        # 修改配置 - 学习率1e-4
        self.lr = 0.0001
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, 'lr_1e4')
        self.results_dir = os.path.join(self.results_dir, 'lr_1e4')

# 创建配置实例
config = LightweightConfigLR1e4()

# 导入并运行原始训练脚本
if __name__ == "__main__":
    # 修改sys.argv以传递参数
    sys.argv = ["train_lightweight_lr_1e4.py", "--use_preprocessed"]
    
    # 导入并运行原始训练脚本
    from train_lightweight import main
    main()
