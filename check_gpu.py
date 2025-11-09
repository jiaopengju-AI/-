import torch

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("GPU数量:", torch.cuda.device_count())
    print("当前GPU:", torch.cuda.current_device())
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("GPU内存:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("未检测到CUDA支持，可能的原因：")
    print("1. 没有安装NVIDIA GPU")
    print("2. 没有安装NVIDIA驱动程序")
    print("3. PyTorch CPU版本而非GPU版本")
    print("4. CUDA版本与PyTorch不兼容")
