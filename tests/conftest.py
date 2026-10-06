# -*- coding: utf-8 -*-
"""
pytest 的公共配置：让测试文件能 import src/ 里的脚本。

每个脚本原来都是"直接运行的脚本"，不是"能被 import 的包"。
测试要调用它们，就得先告诉 Python"代码在 src/ 这个文件夹里"。
这个文件会在跑测试时被 pytest 自动加载，做这一件事就够了。
"""
import sys
from pathlib import Path

# 项目根目录（tests/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent

# 把 src/ 加进 Python 的搜索路径，测试里才能 `import pressure_simulator` 等
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
