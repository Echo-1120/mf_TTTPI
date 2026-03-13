def __init__(self, 
             state_dim: int = 6,  # 3个位置 + 3个角度
             action_dim: int = 3,  # 3个关节的控制
             n_joints: int = 3,    # 关节数量
             link_lengths: list = None,  # 连杆长度
             device: torch.device = torch.device('cpu')):
    
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.n_joints = n_joints
    self.device = device
    
    # 设置link_lengths
    if link_lengths is None:
        self.link_lengths = [1.0] * n_joints  # 默认长度
    else:
        self.link_lengths = link_lengths
        # 确保长度匹配
        if len(self.link_lengths) != n_joints:
            print(f"[警告] 提供的link_lengths长度({len(link_lengths)})与n_joints({n_joints})不匹配")
            if len(self.link_lengths) < n_joints:
                # 补充
                for i in range(len(self.link_lengths), n_joints):
                    self.link_lengths.append(1.0)
            else:
                # 截断
                self.link_lengths = self.link_lengths[:n_joints]
    
    # 其他初始化...
    self.joint_angles = torch.zeros(n_joints, device=device)
    self.reset()
