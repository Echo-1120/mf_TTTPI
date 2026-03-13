"""
稳定的Tensor-Train层实现
解决数值稳定性、梯度爆炸/消失问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class StableTTLayer(nn.Module):
    """
    数值稳定的可训练TT层
    核心特性：
    1. 正交初始化保证数值稳定性
    2. 前向传播中的数值保护
    3. 核心归一化防止梯度爆炸
    4. 梯度监控和裁剪
    """
    
    def __init__(self, dims: List[int], ranks: List[int], 
                 init_scale: float = 0.01, stability_eps: float = 1e-8,
                 device: torch.device = torch.device('cpu')):
        """
        初始化稳定TT层
        
        Args:
            dims: 每个维度的离散化点数 [n1, n2, ..., nd]
            ranks: TT秩 [r0, r1, ..., rd]，其中r0=rd=1
            init_scale: 初始化尺度
            stability_eps: 稳定性系数
            device: 计算设备
        """
        super().__init__()
        
        self.dims = dims
        self.order = len(dims)
        self.ranks = ranks
        self.stability_eps = stability_eps
        self.device = device
        
        # 验证输入合法性
        assert len(ranks) == self.order + 1, f"ranks长度应为order+1，得到{len(ranks)}!={self.order+1}"
        assert ranks[0] == ranks[-1] == 1, f"边界秩必须为1，得到r0={ranks[0]}, rd={ranks[-1]}"
        
        # 创建可训练的核心张量
        self.cores = nn.ParameterList()
        
        for i in range(self.order):
            # 核心形状: [ranks[i], dims[i], ranks[i+1]]
            core_shape = (ranks[i], dims[i], ranks[i+1])
            
            # 正交初始化保证数值稳定性
            if ranks[i] <= ranks[i+1]:
                init_tensor = self._initialize_left_orthogonal(core_shape, init_scale)
            else:
                init_tensor = self._initialize_right_orthogonal(core_shape, init_scale)
            
            # 转换为参数并移动到设备
            param = nn.Parameter(init_tensor.to(device))
            self.cores.append(param)
        
        # 训练监控
        self.register_buffer('grad_norm_history', torch.zeros(1000))
        self.grad_norm_ptr = 0
        
        # 移动到设备
        self.to(device)
        
    def _initialize_left_orthogonal(self, shape, scale):
        """初始化左正交核心"""
        r_left, n, r_right = shape
        
        # 生成随机矩阵
        if r_left * n >= r_right:
            # 瘦矩阵情况：QR分解得到正交矩阵
            mat = torch.randn(r_left * n, r_right)
            Q, _ = torch.linalg.qr(mat)  # QR分解
            init_tensor = Q[:, :r_right].reshape(shape)
        else:
            # 胖矩阵情况：对每个切片正交化
            init_tensor = torch.randn(shape)
            for idx in range(n):
                slice_mat = init_tensor[:, idx, :]
                if r_left >= r_right:
                    Q, _ = torch.linalg.qr(slice_mat)
                    init_tensor[:, idx, :] = Q[:, :r_right]
                else:
                    Q, _ = torch.linalg.qr(slice_mat.T)
                    init_tensor[:, idx, :] = Q[:, :r_left].T
        
        return init_tensor * scale
    
    def _initialize_right_orthogonal(self, shape, scale):
        """初始化右正交核心"""
        r_left, n, r_right = shape
        
        # 生成随机矩阵
        if r_right * n >= r_left:
            # 转置后处理
            mat = torch.randn(r_right * n, r_left)
            Q, _ = torch.linalg.qr(mat)
            init_tensor = Q[:, :r_left].reshape(r_right, n, r_left).permute(2, 1, 0)
        else:
            init_tensor = torch.randn(shape)
            for idx in range(n):
                slice_mat = init_tensor[:, idx, :].T
                if r_right >= r_left:
                    Q, _ = torch.linalg.qr(slice_mat)
                    init_tensor[:, idx, :] = Q[:, :r_left].T
                else:
                    Q, _ = torch.linalg.qr(slice_mat.T)
                    init_tensor[:, idx, :] = Q[:, :r_right]
        
        return init_tensor * scale
    
    def normalize_cores(self):
        """核心归一化：防止数值爆炸的关键步骤"""
        with torch.no_grad():
            for i, core in enumerate(self.cores):
                # 1. Frobenius范数约束
                core_norm = torch.norm(core)
                
                if core_norm > 1.0:
                    # 温和缩放，防止范数过大
                    core.data.div_(core_norm * 1.1)
                elif core_norm < 0.1:
                    # 防止范数过小
                    core.data.mul_(1.1)
                
                # 2. 谱范数约束（防止连乘爆炸）
                if i < self.order - 1:
                    # 重塑为矩阵计算奇异值
                    reshaped = core.reshape(-1, core.shape[-1])
                    U, S, Vh = torch.linalg.svd(reshaped, full_matrices=False)
                    max_sv = S[0]
                    
                    if max_sv > 2.0:
                        # 压缩奇异值，保证谱范数≤2
                        S_clamped = torch.clamp(S, max=2.0)
                        core.data.copy_((U @ torch.diag(S_clamped) @ Vh).reshape(core.shape))
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        前向传播：数值稳定的TT计算
        
        Args:
            indices: 索引张量 [batch_size, order]，每个元素是离散化索引
            
        Returns:
            values: 对应的函数值 [batch_size]
        """
        batch_size = indices.shape[0]
        
        # 确保索引在合法范围内
        # max_vals = torch.tensor(self.dims, device=indices.device, dtype=indices.dtype) - 1
        # indices = torch.clamp(indices, min=0, max=max_vals)
        max_vals = torch.tensor(self.dims, device=indices.device, dtype=indices.dtype) - 1
        min_val = torch.tensor(0, device=indices.device, dtype=indices.dtype)
        indices = torch.clamp(indices, min_val, max_vals)
        
        
        # 逐步计算TT连乘，添加数值稳定性保护
        current_result = None
        
        for i in range(self.order):
            core = self.cores[i]  # [r_i, n_i, r_{i+1}]
            idx = indices[:, i].long()
            
            # 提取对应切片: [batch_size, r_i, r_{i+1}]
            idx = torch.clamp(idx, 0, core.shape[1] - 1)
            slice_tensor = core[:, idx, :].permute(1, 0, 2)  # [batch_size, r_i, r_{i+1}]
            
            if i == 0:
                # 第一个核心：初始值
                current_result = slice_tensor.squeeze(1)  # [batch_size, r_2]
            else:
                # 后续核心：矩阵乘法
                # 关键：在乘法前归一化，防止数值爆炸
                current_norm = torch.norm(current_result, dim=1, keepdim=True) + self.stability_eps
                current_normalized = current_result / current_norm
                
                # 执行乘法
                current_result = torch.bmm(current_normalized.unsqueeze(1), slice_tensor).squeeze(1)
                
                # 恢复尺度
                current_result = current_result * current_norm
        
        # 最终输出：标量值
        return current_result.squeeze(-1)  # [batch_size]
    
    def compute_gradient_norm(self) -> float:
        """计算梯度范数，用于监控"""
        total_norm = 0.0
        for core in self.cores:
            if core.grad is not None:
                param_norm = core.grad.norm().item()
                total_norm += param_norm ** 2
        return np.sqrt(total_norm) if total_norm > 0 else 0.0
    
    def update_gradient_history(self, grad_norm: float):
        """更新梯度历史记录"""
        self.grad_norm_history[self.grad_norm_ptr % 1000] = grad_norm
        self.grad_norm_ptr += 1
    
    def get_gradient_statistics(self) -> dict:
        """获取梯度统计信息"""
        if self.grad_norm_ptr == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        
        history = self.grad_norm_history[:min(self.grad_norm_ptr, 1000)]
        return {
            "mean": history.mean().item(),
            "std": history.std().item(),
            "max": history.max().item()
        }
