import torch
import sys

print("="*30)
print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print("="*30)

# 1. 检查 CUDA 是否可用
is_cuda = torch.cuda.is_available()
print(f"CUDA 是否可用: {is_cuda}")

if is_cuda:
    # 2. 获取显卡信息
    device_count = torch.cuda.device_count()
    print(f"发现 GPU 数量: {device_count}")
    
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"当前显卡型号: {gpu_name}")
    
    # 获取 CUDA 能力 (RTX 5080 应该是 12.0 或更高)
    capability = torch.cuda.get_device_capability(current_device)
    print(f"显卡计算能力 (Compute Capability): {capability}")

    print("-" * 30)
    print("正在尝试进行张量计算...")

    try:
        # 3. 实际计算测试 (最关键的一步)
        # 在 GPU 上创建一个随机张量
        x = torch.randn(1000, 1000).to('cuda')
        y = torch.randn(1000, 1000).to('cuda')
        
        # 做矩阵乘法
        z = x @ y
        
        print(f"✅ 计算成功！结果张量所在设备: {z.device}")
        print("恭喜！您的 RTX 5080 已经完全配置就绪，可以进行深度学习训练了。")
        
    except Exception as e:
        print(f"❌ 计算失败！报错信息如下：\n{e}")
else:
    print("❌ 未检测到 GPU，请检查 PyTorch 版本是否为 CPU 版。")

print("="*30)