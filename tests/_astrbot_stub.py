# -*- coding: utf-8 -*-
"""测试专用的 astrbot.api.logger 替身。

插件运行时只从 astrbot.api 导入 logger，日志统一走 AstrBot 自己的体系
（WebUI 可见、带级别）。存储层回归测试要能在没装 AstrBot 的纯 Python
环境里跑，所以由测试侧在导入插件模块之前注入一个最小替身，而不是在插件
代码里加「导入失败就退回标准库」的退化分支。

本文件只在 tests/ 下被显式导入，不参与插件加载，也不写日志。
"""

import sys
import types


class _SilentLogger:
    """吞掉日志调用：这些回归用例不校验日志内容。"""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def ensure_astrbot_logger():
    """拿不到真实 astrbot.api.logger 时注入最小桩；已有则什么都不做。"""
    try:
        from astrbot.api import logger  # noqa: F401
        return
    except Exception:
        pass

    pkg = sys.modules.get("astrbot")
    if pkg is None:
        pkg = types.ModuleType("astrbot")
        pkg.__path__ = []  # 声明成包，后续 import astrbot.api 才不会报错
        sys.modules["astrbot"] = pkg

    api = sys.modules.get("astrbot.api")
    if api is None:
        api = types.ModuleType("astrbot.api")
        sys.modules["astrbot.api"] = api
        pkg.api = api
    if not hasattr(api, "logger"):
        api.logger = _SilentLogger()
