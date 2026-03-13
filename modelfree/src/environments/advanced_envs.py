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
        """
        修复版三维正向运动学
        支持灵活的关节数和连杆长度
        """
        # 1. 参数检查和初始化
        if isinstance(joint_angles, torch.Tensor):
            joint_angles_np = joint_angles.cpu().numpy()
        else:
            joint_angles_np = np.array(joint_angles)
        
        n_joints = len(joint_angles_np)
        
        # 2. 确保link_lengths存在且长度正确
        if not hasattr(self, 'link_lengths') or self.link_lengths is None:
            self.link_lengths = [1.0] * n_joints  # 默认长度1.0
            print(f"[警告] link_lengths未设置，使用默认值: {self.link_lengths}")
        
        # 如果长度不匹配，调整link_lengths
        if len(self.link_lengths) != n_joints:
            print(f"[警告] link_lengths长度({len(self.link_lengths)})与关节数({n_joints})不匹配")
            if len(self.link_lengths) < n_joints:
                # 补充缺失的长度
                for i in range(len(self.link_lengths), n_joints):
                    self.link_lengths.append(1.0)
            else:
                # 截断多余的长度
                self.link_lengths = self.link_lengths[:n_joints]
        
        # 3. 根据环境配置选择计算方式
        # 检查是否有三维配置
        if hasattr(self, 'use_3d') and self.use_3d:
            # 使用三维计算
            return self._forward_kinematics_3d(joint_angles_np)
        else:
            # 默认使用二维计算（在XY平面）
            return self._forward_kinematics_2d(joint_angles_np)

    def _forward_kinematics_2d(self, joint_angles_np):
        """二维正向运动学（XY平面）"""
        x = 0.0
        y = 0.0
        angle_sum = 0.0
        
        for i in range(len(joint_angles_np)):
            angle_sum += joint_angles_np[i]
            x += self.link_lengths[i] * np.cos(angle_sum)
            y += self.link_lengths[i] * np.sin(angle_sum)
        
        # 返回三维坐标，z=0
        return torch.tensor([x, y, 0.0], device=self.device)

    def _forward_kinematics_3d(self, joint_angles_np):
        """三维正向运动学（简化版）"""
        x = 0.0
        y = 0.0
        z = 0.0
        
        # 简化的三维链式模型
        # 假设：第一个关节绕Z轴，第二个绕Y轴，第三个绕X轴（或类似）
        for i in range(len(joint_angles_np)):
            length = self.link_lengths[i]
            angle = joint_angles_np[i]
            
            # 根据关节类型决定运动方向
            if i % 3 == 0:  # 绕Z轴旋转（影响x,y）
                x += length * np.cos(angle)
                y += length * np.sin(angle)
            elif i % 3 == 1:  # 绕Y轴旋转（影响x,z）
                x += length * np.cos(angle)
                z += length * np.sin(angle)
            else:  # 绕X轴旋转（影响y,z）
                y += length * np.cos(angle)
                z += length * np.sin(angle)
        
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
