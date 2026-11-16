"""Pytest 全局配置:把项目根目录加到 sys.path,让 `from tests.xxx import ...` 可用。

CI 用 `pip install -e .` 安装 conciliator 包,但 tests 目录不会进 site-packages。
这个 conftest.py 在 pytest 收集时自动执行,把项目根加到 path,解决 import 问题。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
