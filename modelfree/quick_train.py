#!/usr/bin/env python3
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
    print("\n测试训练结果...")
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
