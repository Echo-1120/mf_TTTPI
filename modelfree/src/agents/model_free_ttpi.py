"""
无模型TTPI智能体
基于Tensor-Train分解的深度Q学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Dict, Optional, Tuple, Any
import warnings

# from ..core.stable_tt_layer import StableTTLayer
# from ..core.tt_replay_buffer import TTSpecificReplayBuffer
# src/agents/model_free_ttpi.py
# 修改导入部分：

# 原来的错误导入：
# from ..core.stable_tt_layer import StableTTLayer
# from ..core.tt_replay_buffer import TTSpecificReplayBuffer

# 改为相对导入：
from ..core.stable_tt_layer import StableTTLayer
from ..core.tt_replay_buffer import TTSpecificReplayBuffer

# 或者使用 try-except 处理两种导入方式：
try:
    # 首先尝试相对导入
    from ..core.stable_tt_layer import StableTTLayer
    from ..core.tt_replay_buffer import TTSpecificReplayBuffer
except ImportError:
    # 如果失败，尝试绝对导入
    from ..core.stable_tt_layer import StableTTLayer
    from ..core.tt_replay_buffer import TTSpecificReplayBuffer


class ModelFreeTTPI:
    """
    无模型TTPI智能体
    核心特性：
    1. 基于TT分解的Q函数近似
    2. 双Q学习减少过估计
    3. 优先级经验回放
    4. 数值稳定性保障
    5. 混合动作空间支持
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device = torch.device('cpu')):
        """
        初始化无模型TTPI智能体
        
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 环境参数
        self.state_dims = config['state_dims']
        self.action_dims = config['action_dims']
        self.total_dims = self.state_dims + self.action_dims
        
        # TT分解参数
        self.tt_ranks = config.get('tt_ranks', [1, 4, 4, 4, 4, 1])
        
        # 训练参数
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 0.0005)
        self.batch_size = config.get('batch_size', 64)
        self.target_update = config.get('target_update', 100)
        self.grad_clip = config.get('grad_clip', 10.0)
        self.weight_decay = config.get('weight_decay', 1e-5)
        
        # 探索参数
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 5000)
        
        # 初始化参数
        self.init_scale = config.get('init_scale', 0.01)
        
        # 数据增强参数
        self.augmentation_prob = config.get('augmentation_prob', 0.4)
        
        # 创建Q网络和目标网络
        self.q_network = StableTTLayer(
            dims=self.total_dims,
            ranks=self.tt_ranks,
            init_scale=self.init_scale,
            device=device
        )
        
        self.target_network = StableTTLayer(
            dims=self.total_dims,
            ranks=self.tt_ranks,
            init_scale=self.init_scale,
            device=device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 经验回放缓冲区
        self.replay_buffer = TTSpecificReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            state_dims=self.state_dims,
            action_dims=self.action_dims,
            alpha=0.6,
            beta=0.4,
            augmentation_prob=self.augmentation_prob,
            device=device
        )
        
        # 训练状态
        self.steps_done = 0
        self.epsilon = self.epsilon_start
        
        # 训练监控
        self.loss_history = []
        self.grad_norm_history = []
        self.q_value_history = []
        self.td_error_history = []
        
        # 动作选择缓存（提高效率）
        self._action_candidates = self._generate_action_candidates(1000)
    
    def _generate_action_candidates(self, n_candidates: int) -> torch.Tensor:
        """生成候选动作索引"""
        candidates = []
        for _ in range(n_candidates):
            action = torch.zeros(len(self.action_dims), dtype=torch.long, device=self.device)
            for i in range(len(self.action_dims)):
                action[i] = random.randint(0, self.action_dims[i] - 1)
            candidates.append(action)
        return torch.stack(candidates) if candidates else torch.zeros(0, len(self.action_dims), device=self.device)
    
    def state_to_indices(self, states: torch.Tensor) -> torch.Tensor:
        """
        连续状态 -> 离散索引
        
        Args:
            states: 连续状态 [batch_size, state_dim]
            
        Returns:
            离散索引 [batch_size, state_dim]
        """
        indices = torch.zeros_like(states, dtype=torch.long, device=self.device)
        
        for i in range(states.shape[1]):
            # 假设状态范围在[-1, 1]，映射到[0, state_dims[i]-1]
            state_norm = (states[:, i] + 1) / 2  # 归一化到[0, 1]
            idx = (state_norm * (self.state_dims[i] - 1)).long()
            idx = torch.clamp(idx, 0, self.state_dims[i] - 1)
            indices[:, i] = idx
        
        return indices
    
    def action_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """
        连续动作 -> 离散索引
        
        Args:
            actions: 连续动作 [batch_size, action_dim]
            
        Returns:
            离散索引 [batch_size, action_dim]
        """
        indices = torch.zeros_like(actions, dtype=torch.long, device=self.device)
        
        for i in range(actions.shape[1]):
            # 假设动作范围在[-1, 1]，映射到[0, action_dims[i]-1]
            action_norm = (actions[:, i] + 1) / 2
            idx = (action_norm * (self.action_dims[i] - 1)).long()
            idx = torch.clamp(idx, 0, self.action_dims[i] - 1)
            indices[:, i] = idx
        
        return indices
    
    def indices_to_action(self, indices: torch.Tensor) -> torch.Tensor:
        """
        离散索引 -> 连续动作
        
        Args:
            indices: 离散索引 [batch_size, action_dim]
            
        Returns:
            连续动作 [batch_size, action_dim]
        """
        actions = torch.zeros(indices.shape[0], len(self.action_dims), device=self.device)
        
        for i in range(len(self.action_dims)):
            idx = indices[:, i].float()
            # 反离散化到[-1, 1]
            actions[:, i] = (idx / (self.action_dims[i] - 1)) * 2 - 1
        
        return actions
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        ϵ-greedy动作选择
        
        Args:
            state: 当前状态 [state_dim] 或 [batch_size, state_dim]
            training: 是否训练模式
            
        Returns:
            选择的动作 [action_dim] 或 [batch_size, action_dim]
        """
        single_state = len(state.shape) == 1
        if single_state:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        
        # 转换为离散索引
        state_indices = self.state_to_indices(state)
        
        if training and random.random() < self.epsilon:
            # 随机探索
            action_indices = torch.zeros(batch_size, len(self.action_dims), 
                                        dtype=torch.long, device=self.device)
            
            for i in range(len(self.action_dims)):
                action_indices[:, i] = torch.randint(0, self.action_dims[i], 
                                                    (batch_size,), device=self.device)
        else:
            # 利用：选择最优动作
            action_indices = self._select_best_action(state_indices)
        
        # 转换为连续动作
        actions = self.indices_to_action(action_indices)
        
        if single_state:
            return actions[0]
        return actions
    
    def _select_best_action(self, state_indices: torch.Tensor) -> torch.Tensor:
        """
        选择最优动作（批处理版本） - 高效矩阵化版本
        """
        batch_size = state_indices.shape[0]
        n_candidates = min(100, len(self._action_candidates))
        
        if n_candidates == 0:
            return torch.zeros(batch_size, len(self.action_dims), dtype=torch.long, device=self.device)
            
        action_candidates = self._action_candidates[:n_candidates]
        
        # ==== 核心优化：并行评估所有动作候选 ====
        # [B, S] -> [B*C, S]
        states_repeated = state_indices.unsqueeze(1).repeat(1, n_candidates, 1).view(batch_size * n_candidates, -1)
        # [C, A] -> [B*C, A]
        actions_repeated = action_candidates.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size * n_candidates, -1)
        
        sa_indices = torch.cat([states_repeated, actions_repeated], dim=1)
        
        # 一次性计算出该批次下所有候选动作的 Q 值
        with torch.no_grad():
            q_values = self.q_network(sa_indices).view(batch_size, n_candidates)
            
        # 直接通过矩阵的 argmax 找出每行的最优动作
        best_indices = torch.argmax(q_values, dim=1)
        best_actions = action_candidates[best_indices]
        
        return best_actions
    
    def store_transition(self, state: torch.Tensor, action: torch.Tensor, 
                        reward: float, next_state: torch.Tensor, done: bool):
        """
        存储转移经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # 转换为离散索引
        state_idx = self.state_to_indices(state.unsqueeze(0))[0]
        action_idx = self.action_to_indices(action.unsqueeze(0))[0]
        next_state_idx = self.state_to_indices(next_state.unsqueeze(0))[0]
        
        self.replay_buffer.add(state_idx, action_idx, reward, next_state_idx, done)
    
    def compute_td_error(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算TD误差（双Q学习） - 高效矩阵化版本
        """
        # 当前Q值
        sa_indices = torch.cat([batch['states'], batch['actions']], dim=1)
        current_q = self.q_network(sa_indices)
        
        with torch.no_grad():
            next_state_indices = batch['next_states']
            batch_size = next_state_indices.shape[0]
            n_candidates = min(50, len(self._action_candidates))
            
            if n_candidates == 0:
                best_next_q = torch.zeros(batch_size, device=self.device)
            else:
                action_candidates = self._action_candidates[:n_candidates]
                
                # ==== 核心优化：完全向量化并行计算 ====
                # 1. 扩展状态: [B, S] -> [B, 1, S] -> [B, C, S] -> [B*C, S]
                states_repeated = next_state_indices.unsqueeze(1).repeat(1, n_candidates, 1).view(batch_size * n_candidates, -1)
                
                # 2. 扩展动作: [C, A] -> [1, C, A] -> [B, C, A] -> [B*C, A]
                actions_repeated = action_candidates.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size * n_candidates, -1)
                
                # 3. 拼接在一起: [B*C, Total_dims]
                sa_idx_all = torch.cat([states_repeated, actions_repeated], dim=1)
                
                # 4. 一次性通过网络计算所有批次和候选动作的 Q 值！(从 3000 次耗时调用变为 1 次)
                all_q_values = self.q_network(sa_idx_all).view(batch_size, n_candidates)
                
                # 5. 找出每个样本最优动作的索引
                best_action_indices = torch.argmax(all_q_values, dim=1) # [B]
                best_actions = action_candidates[best_action_indices]   # [B, A]
                
                # 6. 使用目标网络评估这些最优动作
                sa_idx_target = torch.cat([next_state_indices, best_actions], dim=1)
                best_next_q = self.target_network(sa_idx_target)
            
            # 计算目标Q值
            rewards = batch['rewards']
            dones = batch['dones']
            target_q = rewards + (1 - dones) * self.gamma * best_next_q
        
        return target_q - current_q
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        单步训练
        
        Returns:
            训练统计信息字典，如果缓冲区不足则返回None
        """
        # 采样批次
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        # 计算TD误差
        td_errors = self.compute_td_error(batch)
        
        # 计算损失（Huber损失，对异常值鲁棒）
        loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors))
        
        # 重要性采样权重
        loss = loss * batch['weights'].mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度监控
        grad_norm = self.q_network.compute_gradient_norm()
        self.q_network.update_gradient_history(grad_norm)
        
        # 梯度裁剪
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        
        # 参数更新
        self.optimizer.step()
        
        # 核心归一化（关键稳定性措施）
        self.q_network.normalize_cores()
        
        # 更新优先级
        self.replay_buffer.update_priorities(batch['indices'], td_errors.detach().abs())
        
        # 更新探索率
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        # 更新目标网络
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 记录监控数据
        self.loss_history.append(loss.item())
        self.grad_norm_history.append(grad_norm)
        self.td_error_history.append(td_errors.abs().mean().item())
        
        with torch.no_grad():
            q_values = self.q_network(torch.cat([batch['states'], batch['actions']], dim=1))
            self.q_value_history.append(q_values.mean().item())
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'avg_q': q_values.mean().item(),
            'td_error_mean': td_errors.abs().mean().item(),
            'epsilon': self.epsilon
        }
    
    def monitor_stability(self) -> str:
        """
        监控训练稳定性
        
        Returns:
            稳定性报告字符串
        """
        if len(self.loss_history) < 10:
            return "训练初期，数据不足"
        
        recent_losses = self.loss_history[-10:]
        recent_grads = self.grad_norm_history[-10:] if self.grad_norm_history else [0]
        
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        grad_mean = np.mean(recent_grads) if recent_grads else 0
        
        stability_report = []
        
        # 检查损失爆炸
        if any(l > 1000 for l in recent_losses):
            stability_report.append("⚠️ 损失爆炸：检测到损失值>1000")
        
        # 检查梯度爆炸
        if grad_mean > 100:
            stability_report.append("⚠️ 梯度爆炸：平均梯度范数>100")
        
        # 检查梯度消失
        if grad_mean < 1e-6 and grad_mean > 0:
            stability_report.append("⚠️ 梯度消失：平均梯度范数<1e-6")
        
        # 检查NaN
        if any(np.isnan(l) for l in recent_losses):
            stability_report.append("❌ NaN出现：损失函数中出现NaN")
        
        # 检查损失震荡
        if loss_std > loss_mean * 0.5 and loss_mean > 0:
            stability_report.append("⚠️ 损失震荡：标准差大于均值的50%")
        
        # 检查Q值异常
        if self.q_value_history:
            recent_q = self.q_value_history[-10:]
            if any(abs(q) > 100 for q in recent_q):
                stability_report.append("⚠️ Q值异常：Q值绝对值>100")
        
        if not stability_report:
            stability_report.append("✅ 训练稳定")
        
        return "\n".join(stability_report)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer.buffer),
            'loss_mean': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'q_value_mean': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0,
            'grad_statistics': self.q_network.get_gradient_statistics(),
            'buffer_statistics': self.replay_buffer.get_statistics()
        }
    
    def save_checkpoint(self, path: str, episode: int = 0):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint['loss_history']
        self.q_value_history = checkpoint['q_value_history']
        
        print(f"加载检查点：episode={checkpoint['episode']}, steps={self.steps_done}")
