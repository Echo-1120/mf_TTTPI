"""
混合控制环境：Catch-Point问题
包含连续和离散动作，适合测试TT算法
修复版：解决自杀式局部最优问题
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import warnings

class HybridControlEnv:
    """
    混合控制环境示例：Catch-Point问题
    状态: [位置x, 位置y, 速度x, 速度y]
    动作: [连续方向控制, 离散开关控制]
    """
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2, 
                 target_position: list = [0.5, 0.5], max_steps: int = 200,
                 difficulty: str = 'medium', device: torch.device = torch.device('cpu')):
        """
        初始化混合控制环境
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            target_position: 目标位置 [x, y]
            max_steps: 最大步数
            difficulty: 难度级别 ['easy', 'medium', 'hard']
            device: 计算设备
        """
        assert state_dim >= 2, "状态维度至少为2"
        assert action_dim >= 2, "动作维度至少为2"
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 目标位置
        self.target = torch.tensor(target_position, device=device)
        
        # 难度配置
        self.difficulty = difficulty
        if difficulty == 'easy':
            self.max_steps = max_steps // 2
            self.noise_scale = 0.01
            self.success_threshold = 0.2
        elif difficulty == 'hard':
            self.max_steps = max_steps * 2
            self.noise_scale = 0.1
            self.success_threshold = 0.05
        else:  # medium
            self.max_steps = max_steps
            self.noise_scale = 0.05
            self.success_threshold = 0.1
        
        # 系统矩阵（简单线性动力学）
        self.A = torch.eye(state_dim, device=device) * 0.9
        self.B = torch.randn(state_dim, action_dim, device=device) * 0.2
        
        # 边界设置
        self.bounds = [-1.0, 1.0, -1.0, 1.0]  # [x_min, x_max, y_min, y_max]
        
        # 初始化历史变量
        self.previous_distance = None
        self.previous_action = None
        
        # 重置环境
        self.reset()
    
    def _ensure_tensor(self, x, dtype=torch.float32):
        """确保输入是张量"""
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype, device=self.device)
        elif torch.is_tensor(x):
            return x.to(self.device)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)
    
    def reset(self):
        """重置环境"""
        # 重置状态：初始位置在原点，速度为零
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        if self.state_dim >= 2:
            # 稍微随机化初始位置，避免总是在原点
            self.state[0] = np.random.uniform(-0.2, 0.2)
            self.state[1] = np.random.uniform(-0.2, 0.2)
        
        # 重置历史变量
        self.previous_distance = None
        self.previous_action = None
        self.step_count = 0
        
        return torch.tensor(self.state, device=self.device)
    
    def _is_out_of_bounds(self, position):
        """检查位置是否超出边界"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = self.bounds
        return (x < x_min or x > x_max or y < y_min or y > y_max)
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        执行一步动作 - 重构版奖励函数
        
        返回: (next_state, reward, done, info)
        info包含:
            - success: True表示真正成功到达目标，False表示失败
            - reason: 结束原因 ('reached_target', 'out_of_bounds', 'timeout', 'in_progress')
            - distance: 当前位置到目标的距离
        """
        # 1. 确保动作是正确格式
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        
        # 确保动作维度正确
        if len(action_np) > self.action_dim:
            action_np = action_np[:self.action_dim]
        elif len(action_np) < self.action_dim:
            action_np = np.pad(action_np, (0, self.action_dim - len(action_np)))
        
        # 2. 解析动作（假设是混合动作空间）
        # 连续控制部分（方向）
        theta = action_np[0] * np.pi  # 映射到[-π, π]
        # 离散开关部分（如果有）
        switch = int(action_np[1]) if len(action_np) > 1 else 0
        
        # 3. 更新状态（简化动力学）
        dt = 0.1
        speed = 0.05
        
        # 位置更新
        self.state[0] += speed * np.cos(theta) * dt  # x
        self.state[1] += speed * np.sin(theta) * dt  # y
        
        # 速度更新（简化的二阶系统）
        self.state[2] = np.cos(theta) * speed  # vx
        self.state[3] = np.sin(theta) * speed  # vy
        
        # 添加噪声
        if hasattr(self, 'noise_scale') and self.noise_scale > 0:
            noise = np.random.randn(4) * self.noise_scale
            self.state[:4] += noise[:4]
        
        # 4. 计算到目标的距离
    
        # 计算到目标的距离
        target_pos = self.target.cpu().numpy()
        current_pos = self.state[:2]
        distance = np.linalg.norm(current_pos - target_pos)
        
        # 记录进度
        progress = 0.0
        if self.previous_distance is not None:
            progress = self.previous_distance - distance  # 正数表示靠近
        
        # 判断结束条件
        reward = 0.0
        done = False
        success = False
        reason = 'in_progress'
        
        # ==== 重构奖励函数 ====
        # 1. 到达目标（真正成功）
        if distance < self.success_threshold:
            done = True
            success = True
            reason = 'reached_target'
            reward = 500.0  # 基础成功奖励
            
            # 额外奖励：快速完成
            steps_bonus = (self.max_steps - self.step_count) * 5.0
            reward += max(steps_bonus, 0)
        
        # 2. 撞墙越界（明确失败）
        elif self._is_out_of_bounds(self.state[:2]):
            done = True
            success = False  # ❌ 明确标记为失败
            reason = 'out_of_bounds'
            reward = -100.0  # 惩罚但不致命
            
        # 3. 达到最大步数（超时失败）
        elif self.step_count >= self.max_steps:
            done = True
            success = False
            reason = 'timeout'
            reward = -50.0  # 超时惩罚较轻
            
            # 如果接近目标，给部分奖励
            if distance < self.success_threshold * 2:
                proximity_reward = 20.0 * (1 - distance/(self.success_threshold*2))
                reward += proximity_reward
        
        # 4. 正常进行中（奖励塑形）
        else:
            done = False
            success = False
            reason = 'in_progress'
            
            # a) 基础距离惩罚（线性，不要太重）
            distance_penalty = -distance * 0.2
            
            # b) 进度奖励（最重要的引导！）
            progress_reward = 0.0
            if progress > 0:  # 靠近目标
                progress_reward = progress * 50.0  # 大幅奖励靠近
            elif progress < 0:  # 远离目标
                progress_reward = progress * 10.0  # 适度惩罚远离
            
            # c) 生存奖励（鼓励继续探索）
            survival_bonus = 0.5
            
            # d) 动作平滑惩罚（可选）
            action_smooth_penalty = 0.0
            if self.previous_action is not None:
                action_diff = np.linalg.norm(action_np - self.previous_action)
                action_smooth_penalty = -action_diff * 0.01
            
            # 总奖励
            reward = distance_penalty + progress_reward + survival_bonus + action_smooth_penalty
        
        # 保存历史信息
        self.previous_distance = distance
        
        # 构建info字典（必须包含success字段！）
        info = {
            'success': success,
            'reason': reason,
            'distance': float(distance),
            'step_count': self.step_count
        }
        
        return next_state, float(reward), done, info

    
    def render(self, mode: str = 'human') -> Any:
        """
        渲染环境状态
        
        Args:
            mode: 渲染模式 ('human', 'rgb_array', 'ansi')
            
        Returns:
            渲染结果
        """
        if mode == 'human':
            position = self.state[:2]
            target = self.target.cpu().numpy()
            distance = np.linalg.norm(position - target)
            
            print(f"Step: {self.step_count}")
            print(f"Position: [{position[0]:.3f}, {position[1]:.3f}]")
            print(f"Target: [{target[0]:.3f}, {target[1]:.3f}]")
            print(f"Distance: {distance:.3f}")
            print(f"State: {self.state}")
            print("-" * 40)
            
            return None
        elif mode == 'ansi':
            # 简单的文本渲染
            position = self.state[:2]
            target = self.target.cpu().numpy()
            
            grid_size = 10
            grid = [['·' for _ in range(grid_size)] for _ in range(grid_size)]
            
            # 映射位置到网格
            pos_x = int((position[0] + 1) / 2 * (grid_size - 1))
            pos_y = int((position[1] + 1) / 2 * (grid_size - 1))
            tar_x = int((target[0] + 1) / 2 * (grid_size - 1))
            tar_y = int((target[1] + 1) / 2 * (grid_size - 1))
            
            # 设置网格字符
            grid[grid_size - 1 - pos_y][pos_x] = 'A'  # Agent
            grid[grid_size - 1 - tar_y][tar_x] = 'T'  # Target
            
            # 生成字符串
            lines = []
            for row in grid:
                lines.append(' '.join(row))
            
            return '\n'.join(lines)
        else:
            raise ValueError(f"不支持的渲染模式: {mode}")
    
    def get_state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取状态边界"""
        low = torch.ones(self.state_dim, device=self.device) * -1.0
        high = torch.ones(self.state_dim, device=self.device) * 1.0
        return low, high
    
    def get_action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作边界"""
        low = torch.tensor([-1.0, 0.0], device=self.device)
        high = torch.tensor([1.0, 1.0], device=self.device)
        return low, high
    
    def close(self):
        """关闭环境"""
        pass
