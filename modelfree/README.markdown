# 无模型张量分解强化学习 (TT-RL)

基于Tensor-Train分解的强化学习框架，专门用于处理高维状态空间和混合动作空间的机器人控制问题。

## 特性

- **TT分解的高效表示**: 将高维Q函数压缩为低秩TT格式
- **混合动作空间支持**: 天然处理连续和离散混合动作
- **数值稳定性**: 内置核心归一化、梯度裁剪等稳定技术
- **TT专用数据增强**: 针对TT结构设计的经验增强
- **优先级经验回放**: 改进样本效率
- **完整的监控系统**: 实时监控训练稳定性

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd tt_rl_project

# 创建虚拟环境
conda create -n ttrlenv python=3.9
conda activate ttrlenv

# 安装依赖
pip install -r requirements.txt
