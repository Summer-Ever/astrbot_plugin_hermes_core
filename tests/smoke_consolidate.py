# -*- coding: utf-8 -*-
"""深度整理冒烟测试：分片、断点续跑、连续失败止损、总预算。

    D:/Local/AstrBot/backend/python/python.exe tests/smoke_consolidate.py

用假的 router 模拟各种模型表现，验证的是「整理编排」的行为，
不联网、不依赖 AstrBot 运行时。
"""

import asyncio
import json
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))   # 让 astrbot_plugin_hermes_core 作为包被导入
ASTROOT = r"D:\Local\AstrBot\backend\app"
if os.path.isdir(ASTROOT) and ASTROOT not in sys.path:
    sys.path.insert(0, ASTROOT)

# import astrbot 会以当前工作目录为根生成 data/，落在插件目录里会被审计判成
# 「运行时文件入库」。先切到临时目录再导入。
os.chdir(tempfile.mkdtemp(prefix="hermes-cwd-"))

import importlib  # noqa: E402

m = importlib.import_module("astrbot_plugin_hermes_core.main")

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))


class FakeRouter:
    """按调用序号返回预设行为。"""

    def __init__(self, behavior):
        self.calls: list[dict] = []
        self.behavior = behavior

    async def chat(self, builder=None, prompt=None, *, purpose="", timeout=None,
                   max_attempts=None, deadline=None, soft_deadline=None, **_kw):
        text = builder(4000) if builder else (prompt or "")
        self.calls.append({"purpose": purpose, "prompt": text,
                           "timeout": timeout, "max_attempts": max_attempts,
                           "deadline": deadline, "soft_deadline": soft_deadline})
        return self.behavior(len(self.calls), text)


def new_core(tmp: str):
    obj = m.HermesCore.__new__(m.HermesCore)
    obj.config = None                      # _get_cfg 走默认值
    obj.db = m.HermesDB(tmp)
    obj._resident_cache = {}
    obj._consolidate_task = None
    return obj


def seed(db, n: int, owner: str = "u1") -> None:
    for i in range(n):
        db.add_memory("fact", f"记忆条目编号 {i:03d} 关于部署细节", owner_id=owner)


async def case_chunk(tmp: str) -> None:
    core = new_core(tmp)
    seed(core.db, 100)
    core.config = types.SimpleNamespace(get=lambda k, d=None: (
        40 if k == "consolidate_chunk_size" else d))
    router = FakeRouter(lambda i, p: "[]")
    core.router = router
    out = await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    check("100 条按 40 切片 → 3 次调用", len(router.calls) == 3, str(len(router.calls)))
    counts = [c["prompt"].count("- #") for c in router.calls]
    check("每片条目数不超过 40", max(counts) <= 40, str(counts))
    check("单片带 60s 超时与 3 次尝试上限",
          all(c["timeout"] == 60.0 and c["max_attempts"] == 3 for c in router.calls))
    check("每片有自己的 soft_deadline", all(c["soft_deadline"] is not None
                                          for c in router.calls))
    check("整次整理带 deadline", all(c["deadline"] is not None for c in router.calls))
    check("完整跑完清掉断点", core.db.get_meta("consolidate_checkpoint", "") == "")
    check("完成文案含片数", "3 片" in out, out[:60])


async def case_resume(tmp: str) -> None:
    core = new_core(tmp)
    seed(core.db, 100)
    core.config = types.SimpleNamespace(get=lambda k, d=None: (
        40 if k == "consolidate_chunk_size" else d))

    def boom(i, p):
        if i >= 2:
            raise m.ModelError("deadline", "预算用尽")
        return "[]"

    core.router = FakeRouter(boom)
    out1 = await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    check("预算用尽时立即停止（只调 2 次）", len(core.router.calls) == 2,
          str(len(core.router.calls)))
    check("未跑完文案", "未跑完" in out1, out1[:40])
    cp = core.db.get_meta("consolidate_checkpoint", "")
    check("写入断点", bool(cp) and len(json.loads(cp)["ids"]) == 40, cp[:60])

    core.router = FakeRouter(lambda i, p: "[]")
    out2 = await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    check("重发只补没跑的 2 片", len(core.router.calls) == 2,
          str(len(core.router.calls)))
    check("重发文案说明沿用进度", "沿用上次进度 1 片" in out2, out2[:60])

    core.router = FakeRouter(lambda i, p: "[]")
    await core._deep_consolidate(core.db.all_active_memories(), dry_run=False, force=True)
    check("force 忽略断点，重新跑满 3 片", len(core.router.calls) == 3,
          str(len(core.router.calls)))


async def case_streak(tmp: str) -> None:
    core = new_core(tmp)
    seed(core.db, 100)
    core.config = types.SimpleNamespace(get=lambda k, d=None: (
        40 if k == "consolidate_chunk_size" else d))
    router = FakeRouter(lambda i, p: (_ for _ in ()).throw(RuntimeError("connection reset")))
    core.router = router
    out = await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    check("连续 2 片失败后停止", len(router.calls) == 2, str(len(router.calls)))
    check("文案说明已停止", "连续 2 片" in out, out[:80])


async def case_unit_deadline_skip(tmp: str) -> None:
    """单片预算用尽只该跳过那一片，不该停整批。"""
    core = new_core(tmp)
    seed(core.db, 100)
    core.config = types.SimpleNamespace(get=lambda k, d=None: (
        40 if k == "consolidate_chunk_size" else d))

    def behavior(i, p):
        if i == 1:
            raise m.ModelError("unit_deadline", "这一片超时了")
        return "[]"

    router = FakeRouter(behavior)
    core.router = router
    out = await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    check("单片超时后继续跑完剩余片", len(router.calls) == 3, str(len(router.calls)))
    check("单片超时计入跳过数", "1 片因模型失败跳过" in out, out[:80])
    check("单片超时不触发整体停止", "连续" not in out, out[:80])


async def case_merge(tmp: str) -> None:
    core = new_core(tmp)
    a = core.db.add_memory("preference", "喜欢喝美式咖啡", owner_id="u9")
    b = core.db.add_memory("preference", "爱喝美式咖啡不加糖", owner_id="u9")
    core.db.add_memory("fact", "住在杭州", owner_id="u9")
    core.router = FakeRouter(lambda i, p: json.dumps(
        [{"ids": [a, b], "kind": "preference", "content": "喜欢喝美式咖啡，不加糖"}],
        ensure_ascii=False))
    out = await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    active = [r["content"] for r in core.db.all_active_memories()]
    check("合并生效", any("不加糖" in c and "爱喝" not in c for c in active), str(active))
    check("合并计数正确", "合并 1 组" in out, out[:60])


async def case_visibility(tmp: str) -> None:
    """同一个人私聊的私事与群里的公开偏好必须分开整理。

    并成一条就只能二选一：私事流到群里，或群里的信息在群里消失。
    """
    core = new_core(tmp)
    core.db.add_memory("fact", "私聊里说的：下周要做手术", owner_id="u9",
                       visibility="private")
    core.db.add_memory("fact", "私聊里说的：身份证号后四位 8823", owner_id="u9",
                       visibility="private")
    core.db.add_memory("preference", "群里说的：喜欢喝美式咖啡", owner_id="u9",
                       group_id="g1", visibility="group")
    core.db.add_memory("preference", "群里说的：爱喝美式不加糖", owner_id="u9",
                       group_id="g1", visibility="group")
    seen: list[str] = []

    def behavior(i, p):
        seen.append(p)
        return "[]"

    core.router = FakeRouter(behavior)
    await core._deep_consolidate(core.db.all_active_memories(), dry_run=False)
    check("私聊与群按两个桶分别整理", len(seen) == 2, str(len(seen)))
    priv_prompts = [p for p in seen if "手术" in p]
    group_prompts = [p for p in seen if "美式" in p]
    check("私聊片不含群里的条目", priv_prompts and "美式" not in priv_prompts[0],
          (priv_prompts[0][-80:] if priv_prompts else "无"))
    check("群片不含私聊的条目", group_prompts and "手术" not in group_prompts[0],
          (group_prompts[0][-80:] if group_prompts else "无"))

    # 合并产物必须继承可见性，不能落成空串（空串在群聊里挡不住注入）
    ids = [r["id"] for r in core.db.all_active_memories() if r["visibility"] == "private"]
    new_id = core.db.merge_memories(ids, "下周要做手术，身份证后四位 8823")
    row = [r for r in core.db.all_active_memories() if int(r["id"]) == int(new_id)][0]
    check("合并产物继承 private", row["visibility"] == "private", str(row["visibility"]))



async def case_footer(tmp: str) -> None:
    """模型落款：单模型、多模型、降级层级都要标清楚。"""
    core = new_core(tmp)

    class R:
        def __init__(self, info):
            self.info = info

        def last_call_info(self):
            return dict(self.info or {})

        def footer(self):
            i = self.info or {}
            if not i:
                return ""
            tier = i.get("tier") or 0
            return f"> {i['provider']}（{'主模型' if tier <= 0 else f'降级 {tier}'}）"

    core.router = R({"provider": "deepseek/deepseek-flash", "tier": 0})
    check("单模型落款是主模型", core._footer() == "> deepseek/deepseek-flash（主模型）", core._footer())
    check("落款拼在末行", core._with_footer("结果").endswith("\n> deepseek/deepseek-flash（主模型）"))

    core.router = R({"provider": "openai/qwen3.7-flash", "tier": 2})
    check("降级层级正确", core._footer() == "> openai/qwen3.7-flash（降级 2）", core._footer())

    used = {"deepseek/deepseek-flash": {"tier": 0, "slices": 3},
            "openai/qwen3.7-flash": {"tier": 2, "slices": 1}}
    txt = core._models_footer(used)
    check("多模型逐个列出", "deepseek/deepseek-flash（主模型，3 片）" in txt
          and "openai/qwen3.7-flash（降级 2）" in txt, txt)
    check("多模型落款以 > 开头", txt.startswith("> "))


async def main() -> int:
    for fn in (case_chunk, case_resume, case_streak, case_unit_deadline_skip,
               case_merge, case_visibility, case_footer):
        tmp = tempfile.mkdtemp(prefix="hermes-cons-")
        try:
            await fn(tmp)
        except Exception as exc:
            FAIL.append(fn.__name__)
            print(f"  FAIL {fn.__name__} 抛异常: {exc!r}")
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
