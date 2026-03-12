#!/usr/bin/env python3
"""
基础使用示例：无模型TTPI
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # modelfree目录
src_dir = os.path.join(project_root, "src")

sys.path.insert(0, project_root)  # 添加modelfree目录
sys.path.insert(0, src_dir)       # 添加src目录

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from environments.hybrid_control_env import HybridControlEnv
from agents.model_free_ttpi import ModelFreeTTPI

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建环境
    env = HybridControlEnv(
        state_dim=4,
        action_dim=2,
        target_position=[0.5, 0.5],
        max_steps=200,
        difficulty='medium',
        device=device
    )
    
    # 配置智能体
    config = {
        # 离散化配置
        'state_dims': [20, 20, 20, 20],  # 4维状态，每维20个区间
        'action_dims': [20, 2],          # 2维动作：连续(20区间) + 离散(2种)
        
        # TT分解参数
        'tt_ranks': [1, 4, 4, 4, 4, 1],  # TT秩
        
        # 训练参数
        'gamma': 0.99,
        'lr': 0.0005,
        'buffer_size': 10000,
        'batch_size': 32,
        'target_update': 100,
        'grad_clip': 10.0,
        'weight_decay': 1e-5,
        
        # 探索参数
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 2000,
        
        # 数据增强
        'augmentation_prob': 0.3,
        
        # 初始化
        'init_scale': 0.01
    }
    
    # 创建智能体
    agent = ModelFreeTTPI(config, device=device)
    
    # 简单训练演示
    print("\n开始演示训练...")
    
    for episode in range(10):  # 演示10轮
        state = env.reset()
        total_reward = 0
        
        for step in range(50):  # 每轮最多50步
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action.cpu())
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 训练
            train_info = agent.train_step()
            
            # 更新状态
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 输出episode结果
        print(f"Episode {episode+1}: 奖励={total_reward:.2f}, 步数={step+1}, 终止原因={info.get('termination', 'unknown')}")
        
        # 检查训练稳定性
        if (episode + 1) % 5 == 0:
            stability = agent.monitor_stability()
            print(f"  稳定性报告: {stability}")
    
    # 测试智能体
    print("\n测试智能体性能...")
    state = env.reset()
    env.render(mode='human')
    
    for step in range(20):
        with torch.no_grad():
            action = agent.select_action(state, training=False)
        
        print(f"Step {step+1}: 动作={action.cpu().numpy()}")
        
        next_state, reward, done, info = env.step(action.cpu())
        state = next_state
        
        env.render(mode='human')
        
        if done:
            print(f"任务完成！总奖励={info.get('total_reward', 0):.2f}")
            break
    
    # 输出训练统计
    stats = agent.get_training_statistics()
    print("\n训练统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    env.close()

if __name__ == "__main__":
    main()
