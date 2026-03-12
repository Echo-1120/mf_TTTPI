# fix_imports.py - 修复导入问题
import os
import sys

def fix_model_free_ttpi():
    """修复 model_free_ttpi.py 中的导入语句"""
    file_path = os.path.join('src', 'agents', 'model_free_ttpi.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换导入语句
    old_imports = [
        "from src.core.stable_tt_layer import StableTTLayer",
        "from src.core.tt_replay_buffer import TTSpecificReplayBuffer"
    ]
    
    new_imports = [
        "from ..core.stable_tt_layer import StableTTLayer",
        "from ..core.tt_replay_buffer import TTSpecificReplayBuffer"
    ]
    
    for old, new in zip(old_imports, new_imports):
        content = content.replace(old, new)
    
    # 写入修复后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复 {file_path} 中的导入语句")

def fix_config_file():
    """修复配置文件中的TT秩"""
    config_path = 'configs/basic_config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到并修复tt_ranks
    for i, line in enumerate(lines):
        if 'tt_ranks:' in line:
            # 检查当前值
            if '[1, 4, 4, 4, 4, 1]' in line:  # 6个值
                lines[i] = line.replace('[1, 4, 4, 4, 4, 1]', '[1, 4, 4, 4, 4, 4, 1]')
                print(f"已修复 {config_path} 中的TT秩配置")
                break
    
    # 写入修复后的内容
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def create_simple_config():
    """创建简单的配置文件"""
    config_content = """# 简单训练配置
environment:
  type: "hybrid_control"
  state_dim: 4
  action_dim: 2
  discretize_bins: 20
  target_position: [0.5, 0.5]
  max_steps: 200
  difficulty: "medium"

model:
  # 注意：4维状态 + 2维动作 = 6维，需要7个TT秩
  tt_ranks: [1, 4, 4, 4, 4, 4, 1]
  gamma: 0.99
  lr: 0.0005
  buffer_size: 10000
  batch_size: 32
  target_update: 100
  grad_clip: 10.0
  weight_decay: 1e-5
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 2000
  augmentation_prob: 0.3
  init_scale: 0.01

training:
  seed: 42
  total_episodes: 100
  max_steps_per_episode: 200
  log_interval: 10
  checkpoint_interval: 50
  viz_interval: 20
  log_dir: "./logs"
  experiment_name: "ttpi_simple"
"""
    
    with open('configs/simple_config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("已创建 simple_config.yaml")

if __name__ == "__main__":
    print("开始修复常见问题...")
    
    # 创建必要的目录
    os.makedirs('configs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 执行修复
    try:
        fix_model_free_ttpi()
    except Exception as e:
        print(f"修复model_free_ttpi失败: {e}")
    
    try:
        fix_config_file()
    except Exception as e:
        print(f"修复配置文件失败: {e}")
        create_simple_config()
    
    print("修复完成！")
    print("运行命令: python scripts/train_basic.py --config configs/simple_config.yaml --output ./output")
