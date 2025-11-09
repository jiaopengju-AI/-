# 批次大小8的对比实验
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原始配置
from config import lightweight_config

# 创建新的配置类，批次大小8
class LightweightConfigBatch8:
    """轻量化模型配置 - 批次大小8"""
    
    # 继承所有原始配置
    def __init__(self):
        # 复制所有原始配置
        for attr in dir(lightweight_config):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(lightweight_config, attr))
        
        # 修改配置 - 批次大小8
        self.batch_size = 8
        self.gradient_accumulation_steps = 6  # 增加梯度累积步数以保持有效批次大小
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, 'batch_8')
        self.results_dir = os.path.join(self.results_dir, 'batch_8')

# 创建配置实例
config = LightweightConfigBatch8()

# 导入并运行原始训练脚本
if __name__ == "__main__":
    # 修改sys.argv以传递参数
    sys.argv = ["train_lightweight_batch_8.py", "--use_preprocessed"]
    
    # 导入并运行原始训练脚本
    from train_lightweight import main
    main()
