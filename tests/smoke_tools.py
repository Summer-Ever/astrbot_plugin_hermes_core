# -*- coding: utf-8 -*-
"""LLM 工具链冒烟测试：save_memory / pin_memory / search_memory 的边界行为。

三套 smoke（db / router / consolidate）覆盖的是模块内部逻辑，但插件真正暴露给
模型的入口是这些 @filter.llm_tool 方法——模型会填错编号、会尝试动别人的记忆、
会在 replaces 失效时以为替换成功。这里用最小假事件实测这些路径。

运行（需要 AstrBot 自带解释器，因为要 import main.py → astrbot）：
    D:/Local/AstrBot/backend/python/python.exe tests/smoke_tools.py
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# main.py import astrbot.*：AstrBot 的 app 目录要在 path 里（部署位置可能不同，
# 按常见安装路径探测，找不到就跳过并说明）。
# 注意插入顺序：插件目录必须在最后插入（path 最前），否则 app 目录下同名的
# main 模块会把插件入口截胡。
_ASTROOT = r"D:\Local\AstrBot\backend\app"
if os.path.isdir(_ASTROOT) and _ASTROOT not in sys.path:
    sys.path.insert(0, _ASTROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# 必须在 import main 之前换目录：import astrbot 会在 CWD 生成 data/cmd_config.json
# 等运行时文件，留在仓库工作区会触发提交审计的 ASB-20（router/consolidate 同款做法）。
os.chdir(tempfile.mkdtemp(prefix="hermes-cwd-"))
try:
    import main as m  # noqa: E402
except ImportError:
    print("跳过：需要 AstrBot 运行环境"
          "（确认 D:/Local/AstrBot/backend/app 存在，"
          "或用 D:/Local/AstrBot/backend/python/python.exe 运行）")
    sys.exit(0)

from modules.db import HermesDB  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not cond else ""))


class FakeEvent:
    """最小 AstrMessageEvent：工具链只用到这几个方法/属性。"""

    def __init__(self, sender_id="u1", platform="qq", group_id="",
                 umo="umo:FriendMessage:u1"):
        self._sid, self._pf, self._gid = sender_id, platform, group_id
        self.unified_msg_origin = umo

    def get_sender_id(self):
        return self._sid

    def get_platform_name(self):
        return self._pf

    def get_group_id(self):
        return self._gid

    def is_admin(self):
        return False

    @property
    def role(self):
        return "member"

    def plain_result(self, text):
        return text


def make_plugin(db: HermesDB):
    """只带工具链所需成员的假 self：真实方法 + 临时库。"""
    fake = types.SimpleNamespace(db=db, _resident_cache={}, _inj_state={},
                                 _get_cfg=lambda k, d=None: d, _metrics={})
    for k, v in vars(m.HermesCore).items():
        if not k.startswith("__") and not callable(v) and not hasattr(fake, k):
            setattr(fake, k, v)
    for name in ("_scope_keys", "_is_admin", "_id_seen", "_clear_resident",
                 "_queue_memories", "_pending_memory_ids",
                 "_requeue_after_consolidate",
                 "tool_save_memory", "tool_pin_memory", "tool_search_memory"):
        setattr(fake, name, types.MethodType(getattr(m.HermesCore, name), fake))
    return fake


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="hermes-smoke-tools-")
    os.chdir(workdir)
    db = HermesDB(tempfile.mkdtemp(prefix="hermes-tools-db-"))
    p = make_plugin(db)
    ev = FakeEvent()
    SID = ev.unified_msg_origin

    def mark_seen(mid: int, turn: int = 5) -> None:
        p._inj_state.setdefault(SID, {"turn": turn, "seen": {}})
        p._inj_state[SID]["seen"][mid] = turn

    # ---- save_memory：replaces 的四种拒绝路径都必须如实回话，不能静默 ----
    print("== save_memory 的 replaces 边界 ==")
    try:
        r = asyncio.run(p.tool_save_memory(ev, content="测试内容甲", replaces=99999))
        check("replaces 不存在：返回提示而非崩溃",
              "不存在" in r and "#99999" not in r.split("；")[0].replace("旧记忆 #99999", ""),
              r)
    except Exception as e:
        check("replaces 不存在：返回提示而非崩溃", False, f"{type(e).__name__}: {e}")

    other = db.add_memory("fact", "别人的记忆内容", owner_id="OTHER_USER",
                          visibility="private")
    mark_seen(other)
    r = asyncio.run(p.tool_save_memory(ev, content="我的新内容乙", replaces=other))
    check("replaces 别人的记忆：拒绝且明确告知",
          "不属于你" in r and "已保存" in r, r)
    check("被拒后旧记忆仍然活跃", db.memory_by_id(other) is not None)

    old_a = db.add_memory("fact", "待合并的旧内容", owner_id="u1", visibility="private")
    old_b = db.add_memory("fact", "另一条碎片", owner_id="u1", visibility="private")
    db.merge_memories([old_a, old_b], "合并后的内容", "fact")
    mark_seen(old_a)
    r = asyncio.run(p.tool_save_memory(ev, content="替换已合并条目", replaces=old_a))
    check("replaces 已合并条目：拒绝且明确告知", "已失效" in r, r)

    unseen = db.add_memory("fact", "从没在本会话出现过的记忆", owner_id="u1",
                           visibility="private")
    r = asyncio.run(p.tool_save_memory(ev, content="替换不可见条目", replaces=unseen))
    check("replaces 不可见编号：拒绝且明确告知", "不在本会话" in r, r)

    # ---- save_memory：正常 replaces 必须真的顶掉旧条 ----
    print("== save_memory 的 replaces 正常路径 ==")
    old_c = db.add_memory("preference", "叫我兄弟", owner_id="u1", visibility="private")
    db.set_pinned(old_c, True)          # 旧条是常驻：新条必须继承
    mark_seen(old_c)
    r = asyncio.run(p.tool_save_memory(ev, content="叫我老爹", kind="preference",
                                       replaces=old_c))
    check("正常 replaces：提示已替换", "已替换" in r and f"#{old_c}" in r, r)
    check("正常 replaces：旧条退出活跃池",
          int(db.memory_by_id(old_c)["merged_into"] or 0) != 0)
    new_ids = [i for i, row in ((r["id"], r) for r in db.all_memories(owner_id="u1"))
               if row["content"] == "叫我老爹"]
    check("正常 replaces：新条继承常驻身份",
          bool(new_ids) and bool(int(db.memory_by_id(new_ids[0])["pinned"] or 0)))

    # ---- pin_memory 边界 ----
    print("== pin_memory 边界 ==")
    try:
        r = asyncio.run(p.tool_pin_memory(ev, memory_id=424242))
        check("pin 不存在：返回提示而非崩溃", "不存在" in r, r)
    except Exception as e:
        check("pin 不存在：返回提示而非崩溃", False, f"{type(e).__name__}: {e}")
    pub = db.add_memory("fact", "群公共记忆一条", visibility="public")
    mark_seen(pub)
    r = asyncio.run(p.tool_pin_memory(ev, memory_id=pub))
    check("非管理员不能钉公共记忆", "管理员" in r, r)
    check("被拒后公共记忆未被钉", not int(db.memory_by_id(pub)["pinned"] or 0))
    mine = db.add_memory("preference", "我自己的偏好", owner_id="u1", visibility="private")
    mark_seen(mine)
    r = asyncio.run(p.tool_pin_memory(ev, memory_id=mine))
    check("钉自己的记忆成功", "已钉住" in r and bool(db.memory_by_id(mine)["pinned"]))

    # ---- search_memory 不泄他人私聊 ----
    print("== search_memory 隔离 ==")
    db.add_memory("preference", "别人的私聊偏好丙", owner_id="OTHER",
                  visibility="private")
    r = asyncio.run(p.tool_search_memory(ev, query="私聊偏好"))
    check("search_memory 不含他人 private", "别人的私聊偏好" not in r, r[:120])

    # ---- 整理后队列重建：只退场「已评估/已不活跃」的 id ----
    print("== 整理队列重建 ==")
    q1 = db.add_memory("fact", "队列里的新记忆一", owner_id="u1", visibility="private")
    q2 = db.add_memory("fact", "队列里的新记忆二", owner_id="u1", visibility="private")
    p._queue_memories([q1, q2])
    p._last_evaluated = {q1}          # 模拟这轮只把 q1 交给模型看过
    db.supersede_memories([q2], 1)    # q2 已不活跃
    p._requeue_after_consolidate()
    rest = p._pending_memory_ids()
    check("已评估的 id 退出队列", q1 not in rest, str(rest))
    check("已不活跃的 id 退出队列", q2 not in rest, str(rest))
    p._last_evaluated = set()         # 整理被跳过（评估集为空）
    p._queue_memories([q1])
    p._requeue_after_consolidate()
    check("被跳过时队列原样保留（不清空）",
          q1 in p._pending_memory_ids(), str(p._pending_memory_ids()))

    # ---- WebUI 写记忆后必须让常驻缓存失效（60 秒缓存会注入幽灵/旧内容）----
    print("== WebUI 记忆管理与常驻缓存 ==")
    import webapi as wa

    payload: dict = {}

    async def fake_body():
        return payload

    wa.HermesWebAPI._body = staticmethod(fake_body)
    wap = types.SimpleNamespace(
        db=db, _resident_cache={}, config=None,
        _sanitize_memory=m.HermesCore._sanitize_memory,
        _looks_like_injection=m.HermesCore._looks_like_injection,
        _pending_memory_ids=lambda: [],
    )
    api = wa.HermesWebAPI(wap)

    def seed_cache():
        wap._resident_cache[("u1", True, "")] = (time.time(), [{"id": 1}])

    def api_add(text):
        payload.clear()
        payload.update({"content": text, "owner_id": "u1",
                        "scope": "user", "kind": "preference"})
        return json.loads(asyncio.run(api.memory_add()).body)["id"]

    seed_cache()
    wa_a = api_add("WebUI 新增的记忆甲内容")
    check("memory_add 后常驻缓存失效", not wap._resident_cache)
    seed_cache()
    wa_b = api_add("WebUI 新增的记忆乙内容")
    wap._resident_cache.clear()

    seed_cache()
    payload.clear()
    payload.update({"id": wa_a, "content": "WebUI 改过的记忆甲内容"})
    asyncio.run(api.memory_update())
    check("memory_update 后常驻缓存失效", not wap._resident_cache)

    seed_cache()
    payload.clear()
    payload.update({"ids": [wa_a, wa_b], "content": "WebUI 合并后的记忆"})
    asyncio.run(api.memory_merge())
    check("memory_merge 后常驻缓存失效", not wap._resident_cache)

    seed_cache()
    payload.clear()
    payload.update({"id": wa_b})
    asyncio.run(api.memory_delete())
    check("memory_delete 后常驻缓存失效", not wap._resident_cache)

    # ---- 注入守卫窗口：旧值 300 秒会让重复提问整体失忆 ----
    print("== 注入守卫窗口 ==")
    gfake = types.SimpleNamespace(
        _get_cfg=lambda k, d=None: True, _inject_guard={}, calls=[])

    def fake_build(ev, q, req):
        gfake.calls.append(q)
        return "[记忆系统上下文] 测试块"

    gfake._build_context_block = fake_build   # 真实实现是同步方法
    gev = types.SimpleNamespace(unified_msg_origin="umo:FriendMessage:u1",
                                message_str="继续")
    greq = types.SimpleNamespace(system_prompt="")
    on_req = types.MethodType(m.HermesCore.on_llm_request, gfake)
    asyncio.run(on_req(gev, greq))
    asyncio.run(on_req(gev, greq))
    check("同一句短期重复请求只注入一次", len(gfake.calls) == 1, str(gfake.calls))
    gkey = next(iter(gfake._inject_guard))
    gfake._inject_guard[gkey] = time.time() - 46   # 模拟 46 秒后再问同一句
    asyncio.run(on_req(gev, greq))
    check("重复提问超过窗口后重新注入（不再失忆 5 分钟）",
          len(gfake.calls) == 2, str(gfake.calls))

    db.close()
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
