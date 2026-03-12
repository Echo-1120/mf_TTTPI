#!/usr/bin/env python3
"""
启动脚本：运行综合对比实验
"""

import sys
import os
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.absolute()
src_dir = project_root / 'src'

# 添加路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

print(f"项目根目录: {project_root}")

# 导入并运行对比实验
try:
    from experiments.comprehensive_comparison import main
    main()
except ImportError as e:
    print(f"导入失败: {e}")
    print("请检查文件路径和目录结构")
