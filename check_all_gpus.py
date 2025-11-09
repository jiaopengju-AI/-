import torch
import subprocess
import platform

print("===== PyTorch GPU信息 =====")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i}内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"GPU {i}计算能力: {torch.cuda.get_device_capability(i)}")

print("\n===== 系统GPU信息 =====")
system = platform.system()
if system == "Windows":
    try:
        # 使用nvidia-smi获取更详细的GPU信息
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi命令未找到，请确保NVIDIA驱动程序已正确安装")
else:
    print("非Windows系统，跳过nvidia-smi检查")

print("\n===== 可用设备 =====")
print("CPU: 可用")
print(f"CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"CUDA:{i} - {torch.cuda.get_device_name(i)}")
