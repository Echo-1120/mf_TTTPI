#!/usr/bin/env python3
"""
最小化修复脚本 - 修复无模型TTPI的所有常见问题
"""

import os
import sys
import shutil

def create_dirs():
    """创建必要目录"""
    dirs = [
        'configs',
        'output', 
        'logs',
        'src',
        'src/agents',
        'src/core',
        'src/environments',
        'src/utils',
        'scripts',
        'examples'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"创建目录: {d}")
    
    # 创建 __init__.py 文件
    init_files = [
        'src/__init__.py',
        'src/agents/__init__.py', 
        'src/core/__init__.py',
        'src/environments/__init__.py',
        'examples/__init__.py'
    ]
    
    for init in init_files:
        if not os.path.exists(init):
            with open(init, 'w') as f:
                f.write('# Package init\n')
            print(f"创建: {init}")

def fix_imports():
    """修复导入问题"""
    # 修复 model_free_ttpi.py
    model_free_file = 'src/agents/model_free_ttpi.py'
    
    if os.path.exists(model_free_file):
        with open(model_free_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换错误的导入
        content = content.replace('from src.core.stable_tt_layer', 'from ..core.stable_tt_layer')
        content = content.replace('from src.core.tt_replay_buffer', 'from ..core.tt_replay_buffer')
        
        with open(model_free_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("已修复 model_free_ttpi.py 中的导入语句")
    else:
        print(f"警告: 文件不存在 - {model_free_file}")

def create_config():
    """创建正确的配置文件"""
    config_content = """# 正确的配置文件 - 无模型TTPI
# 注意: 对于6维状态-动作空间，需要7个TT秩值

environment:
  type: "hybrid_control"
  state_dim: 4
  action_dim: 2
  discretize_bins: 20
  target_position: [0.5, 0.5]
  max_steps: 200
  difficulty: "medium"

model:
  # TT分解参数: 4状态 + 2动作 = 6维，需要7个秩值
  # 格式: [r0, r1, r2, r3, r4, r5, r6]，其中 r0 = r6 = 1
  tt_ranks: [1, 4, 4, 4, 4, 4, 1]
  
  # 训练参数
  gamma: 0.99
  lr: 0.0005
  buffer_size: 10000
  batch_size: 32
  target_update: 100
  grad_clip: 10.0
  weight_decay: 1e-5
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 2000
  augmentation_prob: 0.3
  init_scale: 0.01

training:
  seed: 42
  total_episodes: 50
  max_steps_per_episode: 200
  log_interval: 5
  checkpoint_interval: 25
  log_dir: "./logs"
  experiment_name: "ttpi_simple"
"""
    
    with open('configs/simple_config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("已创建配置文件: configs/simple_config.yaml")

def create_test_script():
    """创建测试脚本"""
    script_content = '''#!/usr/bin/env python3
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
print("\\n测试导入模块...")

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
print("\\n创建环境测试...")
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
print("\\n测试TT层...")
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
print("\\n测试智能体配置...")
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

print("\\n" + "="*60)
print("测试完成")
print("="*60)
'''
    
    with open('test_fix.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("已创建测试脚本: test_fix.py")

def create_quick_train():
    """创建快速训练脚本"""
    train_script = '''#!/usr/bin/env python3
"""
快速训练脚本 - 无模型TTPI
"""

import torch
import numpy as np
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# 导入
from src.agents.model_free_ttpi import ModelFreeTTPI
from src.environments.hybrid_control_env import HybridControlEnv

def main():
    print("开始快速训练...")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建环境
    env = HybridControlEnv(
        state_dim=4,
        action_dim=2,
        target_position=[0.5, 0.5],
        max_steps=100,
        difficulty='easy',
        device=device
    )
    
    # 智能体配置
    config = {
        'state_dims': [10, 10, 10, 10],  # 4状态，每维10区间
        'action_dims': [10, 2],          # 2动作
        'tt_ranks': [1, 3, 3, 3, 3, 3, 1],  # 7个秩
        'gamma': 0.99,
        'lr': 0.001,
        'buffer_size': 5000,
        'batch_size': 16,
        'target_update': 50,
        'grad_clip': 5.0,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 500,
        'augmentation_prob': 0.2,
        'init_scale': 0.01
    }
    
    # 创建智能体
    agent = ModelFreeTTPI(config, device=device)
    
    # 训练参数
    total_episodes = 20
    max_steps = 100
    
    print(f"训练 {total_episodes} 轮...")
    
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action.cpu())
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 训练
            train_info = agent.train_step()
            
            # 更新
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 输出进度
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1}/{total_episodes}: 奖励={episode_reward:.2f}, ϵ={agent.epsilon:.3f}")
    
    print("训练完成!")
    
    # 测试
    print("\\n测试训练结果...")
    state = env.reset()
    test_reward = 0
    
    for step in range(20):
        with torch.no_grad():
            action = agent.select_action(state, training=False)
        
        next_state, reward, done, _ = env.step(action.cpu())
        state = next_state
        test_reward += reward
        
        if done:
            break
    
    print(f"测试奖励: {test_reward:.2f}")
    
    # 保存
    os.makedirs('output', exist_ok=True)
    agent.save_checkpoint('output/model.pth', total_episodes)
    print("模型已保存到 output/model.pth")

if __name__ == "__main__":
    main()
'''
    
    with open('quick_train.py', 'w', encoding='utf-8') as f:
        f.write(train_script)
    
    print("已创建快速训练脚本: quick_train.py")

def main():
    print("="*70)
    print("无模型TTPI修复工具")
    print("="*70)
    
    print("\\n正在修复...")
    
    # 1. 创建目录
    create_dirs()
    
    # 2. 修复导入
    fix_imports()
    
    # 3. 创建配置文件
    create_config()
    
    # 4. 创建测试脚本
    create_test_script()
    
    # 5. 创建训练脚本
    create_quick_train()
    
    print("\\n" + "="*70)
    print("修复完成!")
    print("="*70)
    
    print("\\n下一步操作:")
    print("1. 运行测试脚本: python test_fix.py")
    print("2. 运行快速训练: python quick_train.py")
    print("3. 使用配置文件训练: python scripts/train_basic.py --config configs/simple_config.yaml")
    
    print("\\n如果还有问题，请提供错误信息。")

if __name__ == "__main__":
    main()
