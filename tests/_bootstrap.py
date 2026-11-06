"""测试脚本共用的启动准备：路径、当前目录、logger 替身。

每个 smoke 脚本都需要在导入插件模块之前做三件事：

1. 把插件根目录（以及插件的父目录）放进 ``sys.path``，让 ``modules.*`` 能被导入；
2. 把当前工作目录切到临时目录——``import astrbot`` 会以 CWD 为根生成 ``data/``
   （AstrBot 主配置、t2i 模板），留在插件目录里会被仓库审查判成运行时文件入库；
3. 在没有 AstrBot 的纯 Python 环境里给 ``astrbot.api.logger`` 注入一个最小替身，
   让存储层仍能被直接测试。

这三件事都必须在 import 插件模块**之前**完成，而 import 语句又必须写在文件
顶部（模块级 import 不在顶部是明确的代码组织问题）。把准备动作集中到本模块、
测试脚本用 ``load()`` 取插件模块，两个约束就同时满足了。

需要真实 AstrBot 运行时的脚本，先调 ``enable_astrbot_backend()`` 把 AstrBot
后端目录接进 ``sys.path``，之后 ``load()`` 拿到的就是真环境而不是替身。
"""

import importlib
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 插件的父目录：让 astrbot_plugin_hermes_core 能作为包被导入
PARENT = os.path.dirname(ROOT)
# AstrBot 后端目录（部署位置可能不同，找不到就跳过）
ASTROOT = r"D:\Local\AstrBot\backend\app"

for _p in (PARENT, ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 切到临时目录，避免 import astrbot 把运行时文件写进仓库工作区
os.chdir(tempfile.mkdtemp(prefix="hermes-cwd-"))


class _SilentLogger:
    """吞掉日志调用：这些回归用例不校验日志内容。"""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def enable_astrbot_backend() -> None:
    """把 AstrBot 后端目录接进 ``sys.path``：需要真实运行时的脚本先调它。

    追加到末尾而不是插到最前：插件自己的目录必须在前面，否则 app 目录下同名的
    main 模块会把插件入口截胡。
    """
    if os.path.isdir(ASTROOT) and ASTROOT not in sys.path:
        sys.path.append(ASTROOT)


def ensure_astrbot_logger() -> None:
    """拿不到真实 astrbot.api.logger 时注入最小桩；已有则什么都不做。

    插件运行时只从 astrbot.api 导入 logger，日志统一走 AstrBot 自己的体系
    （WebUI 可见、带级别）。纯 Python 环境没有 AstrBot，由测试侧注入替身，
    而不是让插件代码为测试准备导入兜底。

    缺 AstrBot、缺 AstrBot 自己的依赖（如 sqlalchemy）都会走到 ImportError，
    属于「环境里没有这套东西」的正常情况，装替身继续。
    """
    try:
        import astrbot.api
    except ImportError:
        # 半初始化的 astrbot 会残留在 sys.modules 里，先清干净再装桩，
        # 否则后续 import 会拿到缺属性的残缺包。
        for _name in [n for n in sys.modules
                      if n == "astrbot" or n.startswith("astrbot.")]:
            sys.modules.pop(_name, None)
    else:
        if hasattr(astrbot.api, "logger"):
            return

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


def db_module():
    """存储层模块。"""
    ensure_astrbot_logger()
    return importlib.import_module("modules.db")


def paths_module():
    """路径工具模块。"""
    ensure_astrbot_logger()
    return importlib.import_module("modules.paths")


def recall_module():
    """召回排序模块。"""
    ensure_astrbot_logger()
    return importlib.import_module("modules.recall")


def router_module():
    """模型路由模块。"""
    ensure_astrbot_logger()
    return importlib.import_module("modules.router")


def plugin_main():
    """插件入口模块（顶层 main）。"""
    ensure_astrbot_logger()
    return importlib.import_module("main")


def package_main():
    """插件入口模块（按包路径加载）。"""
    ensure_astrbot_logger()
    return importlib.import_module("astrbot_plugin_hermes_core.main")
