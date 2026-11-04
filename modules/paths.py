# -*- coding: utf-8 -*-
"""路径安全工具。

插件所有运行时写入都必须经过 :func:`safe_join`，把目标路径限制在插件
data 目录内。这样即使将来某个调用方把用户输入接进文件名，也不会写到
插件目录之外（`../` 上跳、绝对路径、盘符路径都会被拒绝）。

只做路径约束，不做权限判断；读操作按需使用。
"""

from __future__ import annotations

import os
from pathlib import Path


class UnsafePath(ValueError):
    """目标路径逃逸出允许的根目录。"""


def _reject(part: str) -> None:
    if not part:
        return
    if part in (".", ".."):
        raise UnsafePath(f"非法的路径片段: {part!r}")
    if os.path.isabs(part) or (len(part) > 1 and part[1] == ":"):
        raise UnsafePath(f"不接受绝对路径: {part}")
    if ".." in Path(part).parts:
        raise UnsafePath(f"不接受上跳路径: {part}")


def safe_join(base: str, *parts: str) -> str:
    """把 ``parts`` 依次拼到 ``base`` 下，并校验结果仍在 ``base`` 内。

    >>> safe_join("/data/plugin", "MEMORY.md").endswith("MEMORY.md")
    True
    """
    if not base:
        raise UnsafePath("根目录为空")
    root = Path(base).expanduser().resolve()
    for p in parts:
        _reject(str(p))
    target = root.joinpath(*[str(p) for p in parts if p]).resolve()
    if target != root and root not in target.parents:
        raise UnsafePath(f"目标逃逸出根目录: {target} 不在 {root} 下")
    return str(target)


def ensure_parent(path: str) -> str:
    """确保父目录存在（仅限已由 safe_join 校验过的路径调用）。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
