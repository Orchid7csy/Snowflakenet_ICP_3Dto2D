"""
为单测加入项目根、Snet 与 scripts 的 import 路径。

若直接运行 `python -m pytest` 时在收集前崩溃（如 ModuleNotFoundError: lark），
多因本机 /opt/ros 的 pytest 插件 launch_testing 被自动加载。解决其一：
- 先执行: export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
- 或: bash scripts/run_chamfer_tests.sh
- 或: pip install lark（让该插件能完整 import）
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SNET = os.path.join(_ROOT, "Snet", "SnowflakeNet-main")
_SCR = os.path.join(_ROOT, "scripts")
for _p in (_ROOT, _SNET, _SCR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
