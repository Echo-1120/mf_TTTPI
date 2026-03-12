#!/usr/bin/env python3
"""
最小化测试脚本 - 验证修复是否成功
"""

import torch
import numpy as np
import sys
import os

print("="*60)
print("测试无模型TTPI修复")
print("="*60)

# 设置Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

print(f"当前目录: {current_dir}")
print(f"Python路径: {sys.path[:2]}")

# 检查依赖
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch未安装")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy已安装")
except ImportError:
    print("✗ NumPy未安装")
    sys.exit(1)

# 测试导入
print("\n测试导入模块...")

try:
    from src.core.stable_tt_layer import StableTTLayer
    print("✓ 成功导入 StableTTLayer")
except ImportError as e:
    print(f"✗ 导入 StableTTLayer 失败: {e}")

try:
    from src.agents.model_free_ttpi import ModelFreeTTPI
    print("✓ 成功导入 ModelFreeTTPI")
except ImportError as e:
    print(f"✗ 导入 ModelFreeTTPI 失败: {e}")

try:
    from src.environments.hybrid_control_env import HybridControlEnv
    print("✓ 成功导入 HybridControlEnv")
except ImportError as e:
    print(f"✗ 导入 HybridControlEnv 失败: {e}")

# 创建简单的环境测试
print("\n创建环境测试...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = HybridControlEnv(device=device)
    state = env.reset()
    print(f"✓ 环境创建成功")
    print(f"  状态维度: {env.state_dim}")
    print(f"  动作维度: {env.action_dim}")
    print(f"  初始状态: {state.cpu().numpy()}")
except Exception as e:
    print(f"✗ 环境创建失败: {e}")

# 测试TT层
print("\n测试TT层...")
try:
    # 创建简单的TT层
    tt_layer = StableTTLayer(
        dims=[10, 10, 10, 10, 10, 2],  # 6维
        ranks=[1, 3, 3, 3, 3, 3, 1],   # 7个秩
        device='cpu'
    )
    print("✓ TT层创建成功")
    
    # 测试前向传播
    indices = torch.tensor([[5, 5, 5, 5, 5, 1]])
    output = tt_layer(indices)
    print(f"  TT层输出形状: {output.shape}")
    print(f"  TT层输出值: {output.item()}")
except Exception as e:
    print(f"✗ TT层测试失败: {e}")

# 测试智能体配置
print("\n测试智能体配置...")
try:
    config = {
        'state_dims': [10, 10, 10, 10],
        'action_dims': [10, 2],
        'tt_ranks': [1, 3, 3, 3, 3, 3, 1],
        'gamma': 0.99,
        'lr': 0.001,
        'buffer_size': 1000,
        'batch_size': 16,
        'target_update': 100,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 1000
    }
    
    agent = ModelFreeTTPI(config, device='cpu')
    print("✓ 智能体创建成功")
    print(f"  状态维度: {agent.state_dims}")
    print(f"  动作维度: {agent.action_dims}")
    print(f"  TT秩: {agent.tt_ranks}")
except Exception as e:
    print(f"✗ 智能体创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成")
print("="*60)
