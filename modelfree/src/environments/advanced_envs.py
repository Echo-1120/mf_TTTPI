# environments/advanced_envs.py
import torch
import numpy as np
from typing import Tuple, Dict, Any

class RoboticArmEnv:
    """6自由度机械臂控制环境"""
    def __init__(self, 
                 state_dim: int = 6,
                 action_dim: int = 4,
                 target_position: list = [0.6, 0.3, 0.2],
                 device: torch.device = torch.device('cpu')):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 机械臂参数
        self.link_lengths = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        self.joint_limits = [(-np.pi/2, np.pi/2)] * 3 + [(-1.0, 1.0)] * 3
        
        # 目标位置
        self.target = torch.tensor(target_position, device=device)
        
        # 重置
        self.reset()
    
    def forward_kinematics(self, joint_angles):
        """正运动学：计算末端位置"""
        x, y, z = 0.0, 0.0, 0.0
        for i, angle in enumerate(joint_angles):
            x += self.link_lengths[i] * torch.cos(angle)
            y += self.link_lengths[i] * torch.sin(angle)
            if i >= 3:
                z += self.link_lengths[i] * angle  # 简化
        
        return torch.tensor([x, y, z], device=self.device)
    
    def reset(self):
        """重置环境"""
        # 随机初始关节角度
        self.joint_angles = torch.rand(self.state_dim, device=self.device) * 0.5 - 0.25
        
        # 计算初始末端位置
        self.end_effector_pos = self.forward_kinematics(self.joint_angles)
        
        # 状态：关节角度 + 末端位置 + 目标距离
        self.state = torch.cat([
            self.joint_angles,
            self.end_effector_pos,
            self.end_effector_pos - self.target
        ])
        
        return self.state.clone()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """执行一步动作"""
        # 动作分解：关节力矩(连续) + 夹爪开关(离散)
        joint_torques = action[:3]  # 前3个关节的力矩
        gripper_action = action[3]  # 夹爪控制
        
        # 更新关节角度（简化动力学）
        dt = 0.1
        self.joint_angles = self.joint_angles + joint_torques * dt
        
        # 约束关节角度
        for i in range(3):
            self.joint_angles[i] = torch.clamp(
                self.joint_angles[i],
                self.joint_limits[i][0],
                self.joint_limits[i][1]
            )
        
        # 计算新的末端位置
        self.end_effector_pos = self.forward_kinematics(self.joint_angles)
        
        # 更新状态
        self.state = torch.cat([
            self.joint_angles,
            self.end_effector_pos,
            self.end_effector_pos - self.target
        ])
        
        # 计算奖励
        distance = torch.norm(self.end_effector_pos - self.target)
        
        # 奖励组成
        distance_reward = -distance * 10.0  # 距离惩罚
        control_penalty = -torch.sum(joint_torques ** 2) * 0.1  # 控制代价
        gripper_penalty = -abs(gripper_action) * 0.05  # 夹爪能耗
        
        reward = distance_reward + control_penalty + gripper_penalty
        
        # 判断终止
        done = distance < 0.05  # 成功阈值
        
        # 夹爪成功额外奖励
        if done and gripper_action > 0.5:
            reward += 20.0
        
        info = {
            'distance': distance.item(),
            'joint_angles': self.joint_angles.cpu().numpy(),
            'end_effector': self.end_effector_pos.cpu().numpy(),
            'target': self.target.cpu().numpy()
        }
        
        return self.state.clone(), reward.item(), done, info
