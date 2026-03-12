# algorithms/comparison_algorithms.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Any, Optional

class DQNBaseline:
    """标准DQN基线算法"""
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # 网络架构
        state_dim = config['state_dim']
        action_dim = config['action_dim']
        hidden_dim = config.get('hidden_dim', 128)
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=config.get('lr', 0.0005))
        
        # 经验回放
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 64)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # 训练参数
        self.gamma = config.get('gamma', 0.99)
        self.target_update = config.get('target_update', 100)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 5000)
        
        self.steps_done = 0
        self.epsilon = self.epsilon_start
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """动作选择"""
        if training and random.random() < self.epsilon:
            return torch.randint(0, self.config['action_dim'], (1,), device=self.device)
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                return torch.argmax(q_values, dim=1)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[Dict]:
        """训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 更新探索率
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}

class TabularQLearning:
    """表格Q学习基线"""
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # 离散化参数
        self.state_dims = config['state_dims']  # 每个维度的离散化点数
        self.action_dim = config['action_dim']
        
        # 计算总状态数
        self.total_states = np.prod(self.state_dims)
        
        # 初始化Q表
        self.q_table = np.zeros((self.total_states, self.action_dim))
        
        # 训练参数
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 0.1)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.999)
        self.epsilon_min = config.get('epsilon_end', 0.01)
        
        # 状态到索引的映射缓存
        self.state_to_idx_cache = {}
    
    def state_to_index(self, state: torch.Tensor) -> int:
        """连续状态转换为离散索引"""
        state_key = tuple(state.cpu().numpy().round(3))
        if state_key in self.state_to_idx_cache:
            return self.state_to_idx_cache[state_key]
        
        # 计算索引
        idx = 0
        multiplier = 1
        for i, val in enumerate(state.cpu().numpy()):
            # 归一化到[0, 1]
            normalized = (val + 1) / 2  # 假设状态在[-1, 1]
            discrete = int(normalized * (self.state_dims[i] - 1))
            discrete = max(0, min(self.state_dims[i] - 1, discrete))
            idx += discrete * multiplier
            multiplier *= self.state_dims[i]
        
        self.state_to_idx_cache[state_key] = idx
        return idx
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """动作选择"""
        state_idx = self.state_to_index(state)
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储并立即学习（表格法）"""
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        # Q学习更新
        current_q = self.q_table[state_idx, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        # 更新Q值
        self.q_table[state_idx, action] = current_q + self.lr * (target_q - current_q)
        
        # 衰减探索率
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_step(self) -> Dict:
        """表格法没有单独的训练步骤"""
        return {'epsilon': self.epsilon}
