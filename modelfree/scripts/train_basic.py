#!/usr/bin/env python3
"""
train_basic.py - 无模型TTPI训练脚本
修复了导入路径和TT秩配置问题
"""

import torch
import numpy as np
import yaml
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# --- 关键修复：正确设置Python路径 ---
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()

# 计算项目根目录 (modelfree 目录)
project_root = current_file.parent.parent  # scripts -> modelfree
src_dir = project_root / 'src'

# 添加路径
sys.path.insert(0, str(project_root))  # 添加 modelfree 目录
sys.path.insert(0, str(src_dir))       # 添加 src 目录

print(f"项目根目录: {project_root}")
print(f"src目录: {src_dir}")

# 现在可以导入
try:
    from agents.model_free_ttpi import ModelFreeTTPI
    from environments.hybrid_control_env import HybridControlEnv
    print("模块导入成功")
except ImportError as e:
    print(f"导入失败: {e}")
    print("尝试备用导入方式...")
    
    # 备用导入方案
    import importlib.util
    
    # 导入 model_free_ttpi
    ttpi_path = src_dir / 'agents' / 'model_free_ttpi.py'
    if ttpi_path.exists():
        spec = importlib.util.spec_from_file_location("model_free_ttpi", str(ttpi_path))
        ttpi_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ttpi_module)
        ModelFreeTTPI = ttpi_module.ModelFreeTTPI
        print("ModelFreeTTPI 导入成功")
    
    # 导入 hybrid_control_env
    env_path = src_dir / 'environments' / 'hybrid_control_env.py'
    if env_path.exists():
        spec = importlib.util.spec_from_file_location("hybrid_control_env", str(env_path))
        env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_module)
        HybridControlEnv = env_module.HybridControlEnv
        print("HybridControlEnv 导入成功")

def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment(config: dict):
    """设置实验环境"""
    # 设置随机种子
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建环境
    env_config = config['environment']
    if env_config['type'] == 'hybrid_control':
        env = HybridControlEnv(
            state_dim=env_config['state_dim'],
            action_dim=env_config['action_dim'],
            target_position=env_config.get('target_position', [0.5, 0.5]),
            max_steps=env_config.get('max_steps', 200),
            difficulty=env_config.get('difficulty', 'medium'),
            device=device
        )
        
        # 自动生成离散化配置
        discretize_bins = env_config['discretize_bins']
        state_dims = [discretize_bins] * env_config['state_dim']
        
        # 动作维度：连续动作离散化，离散动作保持2个值
        action_dims = [discretize_bins] * (env_config['action_dim'] - 1) + [2]
        
        # 验证总维度
        total_dims = len(state_dims) + len(action_dims)
        print(f"状态维度: {state_dims}")
        print(f"动作维度: {action_dims}")
        print(f"总维度: {total_dims}")
        
        # 检查TT秩配置
        tt_ranks = config['model']['tt_ranks']
        expected_rank_length = total_dims + 1
        if len(tt_ranks) != expected_rank_length:
            print(f"⚠️ TT秩配置错误: 期望长度={expected_rank_length}, 实际长度={len(tt_ranks)}")
            print(f"  当前tt_ranks: {tt_ranks}")
            
            # 自动修复：创建一个正确的秩数组
            if len(tt_ranks) == expected_rank_length - 1:
                # 可能是少了最后一个1
                tt_ranks.append(1)
                print(f"  自动修复: 添加末尾的1，新的tt_ranks={tt_ranks}")
            elif len(tt_ranks) == total_dims:
                # 可能是边界条件错误
                tt_ranks = [1] + tt_ranks + [1]
                print(f"  自动修复: 添加边界1，新的tt_ranks={tt_ranks}")
            else:
                # 创建默认秩
                default_rank = 4
                tt_ranks = [1] + [default_rank] * (total_dims - 1) + [1]
                print(f"  使用默认秩: {tt_ranks}")
            
            config['model']['tt_ranks'] = tt_ranks
    else:
        raise ValueError(f"不支持的环境类型: {env_config['type']}")
    
    # 更新模型配置
    model_config = config['model'].copy()
    model_config['state_dims'] = state_dims
    model_config['action_dims'] = action_dims
    
    # 再次验证TT秩
    print(f"最终使用的TT秩: {model_config['tt_ranks']}")
    print(f"秩长度: {len(model_config['tt_ranks'])}, 期望: {total_dims + 1}")
    
    # 创建智能体
    agent = ModelFreeTTPI(model_config, device=device)
    
    return env, agent, device, config

def train_episode(env, agent, episode_idx, max_steps):
    """训练一个episode"""
    state = env.reset()
    episode_reward = 0
    episode_losses = []
    episode_q_values = []
    episode_steps = 0
    
    for step in range(max_steps):
        # 选择动作
        action = agent.select_action(state, training=True)
        
        # 执行动作
        next_state, reward, done, info = env.step(action.cpu())
        
        # 存储经验
        agent.store_transition(state, action, reward, next_state, done)
        
        # 训练步骤
        train_info = agent.train_step()
        if train_info:
            episode_losses.append(train_info['loss'])
            episode_q_values.append(train_info['avg_q'])
        
        # 更新状态
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # 检查终止
        if done:
            break
    
    # 计算episode统计
    episode_stats = {
        'episode': episode_idx,
        'reward': episode_reward,
        'steps': episode_steps,
        'avg_loss': np.mean(episode_losses) if episode_losses else 0,
        'avg_q': np.mean(episode_q_values) if episode_q_values else 0,
        'termination': info.get('termination', 'unknown'),
        'distance': info.get('distance', 0)
    }
    
    return episode_stats

def main():
    parser = argparse.ArgumentParser(description='训练无模型TTPI智能体')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    parser.add_argument('--episodes', type=int, default=100, help='训练轮数（覆盖配置文件）')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        # 如果配置文件不存在，使用默认配置
        print(f"配置文件不存在: {config_path}")
        print("使用默认配置")
        config = {
            'environment': {
                'type': 'hybrid_control',
                'state_dim': 4,
                'action_dim': 2,
                'discretize_bins': 20,
                'target_position': [0.5, 0.5],
                'max_steps': 200,
                'difficulty': 'medium'
            },
            'model': {
                # 注意：根据维度自动计算正确的TT秩
                'tt_ranks': [1, 4, 4, 4, 4, 4, 1],  # 6维需要7个秩
                'gamma': 0.99,
                'lr': 0.0005,
                'buffer_size': 10000,
                'batch_size': 32,
                'target_update': 100,
                'grad_clip': 10.0,
                'weight_decay': 1e-5,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 2000,
                'augmentation_prob': 0.3,
                'init_scale': 0.01
            },
            'training': {
                'seed': 42,
                'total_episodes': args.episodes,
                'max_steps_per_episode': 200,
                'log_interval': 10,
                'checkpoint_interval': 50,
                'viz_interval': 20,
                'log_dir': './logs',
                'experiment_name': 'ttpi_training'
            }
        }
    else:
        config = load_config(config_path)
        # 覆盖训练轮数
        if args.episodes > 0:
            config['training']['total_episodes'] = args.episodes
    
    # 设置实验
    env, agent, device, full_config = setup_experiment(config)
    
    # 训练参数
    training_config = config['training']
    total_episodes = training_config['total_episodes']
    max_steps_per_episode = training_config['max_steps_per_episode']
    log_interval = training_config['log_interval']
    checkpoint_interval = training_config['checkpoint_interval']
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练无模型TTPI智能体")
    print(f"环境: {config['environment']['type']}")
    print(f"总训练轮数: {total_episodes}")
    print("="*60 + "\n")
    
    episode_rewards = []
    
    for episode in range(total_episodes):
        # 训练一个episode
        stats = train_episode(env, agent, episode, max_steps_per_episode)
        episode_rewards.append(stats['reward'])
        
        # 定期输出
        if (episode + 1) % log_interval == 0:
            # 获取稳定性报告
            stability_report = agent.monitor_stability()
            
            # 计算最近表现
            recent_rewards = episode_rewards[-log_interval:] if len(episode_rewards) >= log_interval else episode_rewards
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            std_reward = np.std(recent_rewards) if recent_rewards else 0
            
            print(f"\nEpisode {episode+1}/{total_episodes}")
            print(f"  奖励: {stats['reward']:.2f} (最近平均: {avg_reward:.2f} ± {std_reward:.2f})")
            print(f"  步数: {stats['steps']}, 终止原因: {stats['termination']}")
            print(f"  平均损失: {stats['avg_loss']:.4f}, 平均Q值: {stats['avg_q']:.4f}")
            print(f"  当前ϵ: {agent.epsilon:.3f}, 缓冲区大小: {len(agent.replay_buffer.buffer)}")
            print(f"  稳定性报告:\n{stability_report}")
            print("-" * 50)
        
        # 定期保存检查点
        if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_ep{episode+1}.pth"
            agent.save_checkpoint(str(checkpoint_path), episode)
            print(f"检查点已保存: {checkpoint_path}")
    
    # 最终保存
    final_path = output_dir / "final_model.pth"
    agent.save_checkpoint(str(final_path), total_episodes)
    
    print(f"\n训练完成！模型已保存到: {final_path}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
