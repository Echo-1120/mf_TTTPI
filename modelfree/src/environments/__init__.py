# src/environments/__init__.py
import torch  # ⭐ 添加这行
import numpy as np
from typing import Dict, Any, Optional

# 原有导入
from .hybrid_control_env import HybridControlEnv
from .advanced_envs import RoboticArmEnv

# 原有函数定义（确保使用了 torch.device 的地方都能正常工作）
def make_env(env_name: str, device: Optional[torch.device] = None) -> Any:
    """创建环境实例"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if env_name == 'hybrid_control':
        return HybridControlEnv(device=device)
    elif env_name == 'robotic_arm':
        return RoboticArmEnv(device=device)
    else:
        raise ValueError(f"未知环境: {env_name}")

__all__ = ['HybridControlEnv', 'RoboticArmEnv', 'make_env']
