"""
TT专用的经验回放缓冲区
包含针对TT结构的数据增强和优先级采样
"""

import torch
import numpy as np
from collections import deque
import random
from typing import List, Dict, Optional, Tuple
import warnings

class TTSpecificReplayBuffer:
    """
    TT结构专用的经验回放缓冲区
    核心特性：
    1. 优先级经验回放
    2. TT专用数据增强
    3. 边界值增强
    4. 维度洗牌增强
    """
    
    def __init__(self, capacity: int, state_dims: List[int], action_dims: List[int],
                 alpha: float = 0.6, beta: float = 0.4, 
                 augmentation_prob: float = 0.3, device: torch.device = torch.device('cpu')):
        """
        初始化TT专用回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_dims: 状态每个维度的离散化点数
            action_dims: 动作每个维度的离散化点数
            alpha: 优先级指数 (0=均匀采样，1=完全优先级)
            beta: 重要性采样权重指数
            augmentation_prob: 数据增强概率
            device: 计算设备
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.total_state_dims = len(state_dims)
        self.total_action_dims = len(action_dims)
        
        # 优先级采样参数
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.priorities = deque(maxlen=capacity)
        
        # 数据增强参数
        self.augmentation_prob = augmentation_prob
        
        # 设备
        self.device = device
        
        # 统计数据
        self.add_count = 0
        self.sample_count = 0
    
    def add(self, state_idx: torch.Tensor, action_idx: torch.Tensor,
            reward: float, next_state_idx: torch.Tensor, done: bool):
        """
        添加经验到缓冲区
        
        Args:
            state_idx: 状态索引 [state_dim]
            action_idx: 动作索引 [action_dim]
            reward: 奖励值
            next_state_idx: 下一状态索引 [state_dim]
            done: 是否终止
        """
        # 转换为CPU张量存储
        experience = {
            'state': state_idx.cpu().clone(),
            'action': action_idx.cpu().clone(),
            'reward': reward,
            'next_state': next_state_idx.cpu().clone(),
            'done': done
        }
        
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
        self.add_count += 1
    
    def _augment_experience(self, experience: Dict) -> Dict:
        """
        TT专用的数据增强
        
        Args:
            experience: 原始经验
            
        Returns:
            增强后的经验
        """
        if random.random() > self.augmentation_prob:
            return experience
        
        state = experience['state'].clone()
        action = experience['action'].clone()
        
        # 增强1：维度洗牌（利用TT的排列不变性）
        if random.random() < 0.4:
            # 在状态维度内洗牌
            if self.total_state_dims > 1:
                perm = torch.randperm(self.total_state_dims)
                state = state[perm]
        
        # 增强2：边界值增强
        if random.random() < 0.3:
            # 随机选择一个状态维度设置为边界值
            dim_idx = random.randint(0, self.total_state_dims - 1)
            state[dim_idx] = random.choice([0, self.state_dims[dim_idx] - 1])
        
        # 增强3：动作边界增强
        if random.random() < 0.2 and self.total_action_dims > 0:
            # 随机选择一个动作维度设置为边界值
            dim_idx = random.randint(0, self.total_action_dims - 1)
            action[dim_idx] = random.choice([0, self.action_dims[dim_idx] - 1])
        
        # 增强4：局部平滑增强（添加小扰动）
        if random.random() < 0.3:
            # 对状态添加小扰动
            noise = torch.randint(-1, 2, state.shape)  # {-1, 0, 1}
            state = torch.clamp(state + noise, 0, torch.tensor(self.state_dims) - 1)
        
        return {
            'state': state,
            'action': action,
            'reward': experience['reward'],
            'next_state': experience['next_state'],
            'done': experience['done']
        }
    
    def sample(self, batch_size: int) -> Optional[Dict]:
        """
        优先级采样批次数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次数据字典，包含states, actions, rewards, next_states, dones, weights, indices
            如果缓冲区不足，返回None
        """
        if len(self.buffer) < batch_size:
            return None
        
        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        if probs_sum == 0:
            # 如果所有优先级都为0，使用均匀分布
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = probs / probs_sum
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max() if weights.max() > 0 else weights
        
        # 收集批次数据（应用数据增强）
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for idx in indices:
            experience = self.buffer[idx]
            augmented = self._augment_experience(experience)
            
            batch_states.append(augmented['state'])
            batch_actions.append(augmented['action'])
            batch_rewards.append(augmented['reward'])
            batch_next_states.append(augmented['next_state'])
            batch_dones.append(augmented['done'])
        
        # 转换为张量
        states = torch.stack(batch_states).to(self.device)
        actions = torch.stack(batch_actions).to(self.device)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(batch_next_states).to(self.device)
        dones = torch.tensor(batch_dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        self.sample_count += 1
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'weights': weights,
            'indices': indices
        }
    
    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor):
        """
        更新经验优先级
        
        Args:
            indices: 经验索引
            td_errors: TD误差绝对值
        """
        td_errors = td_errors.cpu().numpy()
        
        for idx, error in zip(indices, td_errors):
            # 计算新的优先级
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = float(priority)
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
    
    def get_statistics(self) -> Dict:
        """获取缓冲区统计信息"""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'add_count': self.add_count,
            'sample_count': self.sample_count,
            'fill_ratio': len(self.buffer) / self.capacity,
            'max_priority': self.max_priority
        }
