"""LLM 工具链冒烟测试：save_memory / search_memory 的边界行为。

三套 smoke（db / router / consolidate）覆盖的是模块内部逻辑，但插件真正暴露给
模型的入口是这些 @filter.llm_tool 方法——模型会填错编号、会尝试动别人的记忆、
会在 replaces 失效时以为替换成功。这里用最小假事件实测这些路径。

运行（需要 AstrBot 自带解释器，因为要 import main.py → astrbot）：
    D:/Local/AstrBot/backend/python/python.exe tests/smoke_tools.py
"""
import asyncio
import inspect
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import db_module, enable_astrbot_backend, plugin_main

enable_astrbot_backend()

try:
    m = plugin_main()
except ImportError:
    print("跳过：需要 AstrBot 运行环境"
          "（确认 D:/Local/AstrBot/backend/app 存在，"
          "或用 D:/Local/AstrBot/backend/python/python.exe 运行）")
    sys.exit(0)

HermesDB = db_module().HermesDB

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
                                 _get_cfg=lambda k, d=None: d, _metrics={},
                                 # 注入块会读技能清单；这里不需要技能，给个空壳
                                 hub=types.SimpleNamespace(list_for=lambda *a, **k: []))
    for k, v in vars(m.HermesCore).items():
        if not k.startswith("__") and not callable(v) and not hasattr(fake, k):
            setattr(fake, k, v)
    for name in ("_scope_keys", "_is_admin", "_id_seen", "_clear_resident",
                 "_queue_memories", "_pending_memory_ids",
                 "_requeue_after_consolidate",
                 "_resident_block", "_resident_limits",
                 "_build_context_block", "_bump_metric",
                 "_render_resident",
                 "tool_save_memory", "tool_search_memory"):
        fn = getattr(m.HermesCore, name)
        # staticmethod 不能 MethodType 绑定——绑定后调用会多传一个 self。
        if inspect.getattr_static(m.HermesCore, name, None) is not None and \
                isinstance(inspect.getattr_static(m.HermesCore, name), staticmethod):
            setattr(fake, name, fn)
        else:
            setattr(fake, name, types.MethodType(fn, fake))
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
    mark_seen(old_c)
    r = asyncio.run(p.tool_save_memory(ev, content="叫我老爹", kind="preference",
                                       replaces=old_c))
    check("正常 replaces：提示已替换", "已替换" in r and f"#{old_c}" in r, r)
    check("正常 replaces：旧条退出活跃池",
          int(db.memory_by_id(old_c)["merged_into"] or 0) != 0)
    rows_new = [row for row in db.all_memories(owner_id="u1")
                if row["content"] == "叫我老爹"]
    check("正常 replaces：新条留在归档层（block 遗留列为空）",
          bool(rows_new) and rows_new[0]["block"] == "",
          str(rows_new[0]["block"]) if rows_new else "没找到新条")

    # ---- save_memory 只写归档：常驻词条不受牵连 ----
    print("== save_memory 只写归档 ==")
    db.set_resident("u1", "private", "preference", "喜欢喝冰美式")
    r = asyncio.run(p.tool_save_memory(ev, content="用户喜欢喝冰美式",
                                       kind="preference"))
    check("save_memory 回执是归档而非常驻", "已保存归档记忆" in r and "每轮" not in r, r)
    rows_b = [row for row in db.all_memories(owner_id="u1")
              if row["content"] == "用户喜欢喝冰美式"]
    check("save_memory 落库 block 恒为空（legacy 列）",
          bool(rows_b) and not rows_b[0]["block"],
          str(rows_b[0]["block"]) if rows_b else "没落库")
    check("save_memory 不改动常驻词条",
          db.get_resident("u1", "private").get("preference") == "喜欢喝冰美式")

    # ---- 常驻词条：读取与渲染 ----
    print("== 常驻词条 ==")
    db.set_resident("u1", "private", "address", "哥哥")
    db.set_resident("u1", "private", "style", "回答一句话讲完")
    blocks = p._resident_block("u1", "private")
    check("词条按（人，场景）聚合为纯文本",
          blocks.get("preference") == "喜欢喝冰美式"
          and blocks.get("address") == "哥哥", str(list(blocks)))
    lines = m.HermesCore._render_resident(blocks)
    check("渲染成一行一个词条",
          "- 用户偏好：喜欢喝冰美式" in lines and "- 称呼：哥哥" in lines, str(lines))
    check("渲染不带编号", all("#" not in x for x in lines), str(lines[:1]))
    check("渲染按六词条固定顺序",
          [x.split("：")[0] for x in lines] == ["- 称呼", "- 用户偏好", "- 回复风格"],
          str(lines))
    check("空词条不产生空行", all(not x.endswith("：") for x in lines), str(lines))
    check("别的用户读不到这份常驻", p._resident_block("u_other", "private") == {})
    check("别的场景读不到这份常驻", p._resident_block("u1", "group:G9") == {})

    # ---- 变更信号：用户说「我自己的情况变了」时要提示模型更新常驻 ----
    print("== 状态变更信号 ==")
    for q in ("我上周毕业了", "以后叫我老师吧", "我搬家到杭州了",
              "现在不喜欢吃辣了", "我换工作了", "重装了系统"):
        check(f"命中变更信号：{q}", bool(m._state_change_signal(q)), q)
    for q in ("把这段代码改成异步", "帮我写个快速排序", "今天天气怎么样",
              "这个 bug 怎么修", "画一张关系图", "我电脑是什么系统"):
        check(f"不误触发：{q}", not m._state_change_signal(q), q)

    # ---- 注入块：常驻由对话里的模型维护，这一授权必须每轮送达 ----
    print("== 注入块 ==")
    p._clear_resident()
    blk = p._build_context_block(ev, "我电脑是什么系统", None)
    check("注入块带常驻维护授权", "维护约定" in blk and "update_resident" in blk,
          blk[:160])
    check("注入块带六词条行", "- 称呼：哥哥" in blk and "- 用户偏好：喜欢喝冰美式" in blk,
          blk[:200])
    check("普通问题不追加变更提示", "情况变了" not in blk)
    blk2 = p._build_context_block(ev, "我上周毕业了", None)
    check("陈述变化时追加更新指令",
          "情况变了" in blk2 and "毕业" in blk2, blk2[-160:])

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

    # ---- WebUI 写归档不再动常驻缓存；写词条必须让缓存失效 ----
    print("== WebUI 记忆管理与常驻缓存 ==")
    import webapi as wa

    payload: dict = {}

    async def fake_body():
        return payload

    wa.HermesWebAPI._body = staticmethod(fake_body)
    wap = types.SimpleNamespace(
        db=db, _resident_cache={}, config=None,
        _get_cfg=lambda k, d=None: d,
        _sanitize_memory=m.HermesCore._sanitize_memory,
        _looks_like_injection=m.HermesCore._looks_like_injection,
        _pending_memory_ids=lambda: [],
    )
    for name in ("_resident_limits", "_clear_resident", "_resident_block"):
        setattr(wap, name, types.MethodType(getattr(m.HermesCore, name), wap))
    api = wa.HermesWebAPI(wap)

    def api_add(text):
        payload.clear()
        payload.update({"content": text, "owner_id": "u1",
                        "scope": "user", "kind": "preference"})
        return json.loads(asyncio.run(api.memory_add()).body)["id"]

    wap._resident_cache[("u1", "private")] = (time.time(), {"stale": 1})
    wa_a = api_add("WebUI 新增的记忆甲内容")
    check("memory_add 只写归档：不触发常驻缓存失效",
          ("u1", "private") in wap._resident_cache)
    wa_b = api_add("WebUI 新增的记忆乙内容")
    payload.clear()
    payload.update({"id": wa_a, "content": "WebUI 改过的记忆甲内容"})
    asyncio.run(api.memory_update())
    payload.clear()
    payload.update({"ids": [wa_a, wa_b], "content": "WebUI 合并后的记忆"})
    asyncio.run(api.memory_merge())
    payload.clear()
    payload.update({"id": wa_b})
    asyncio.run(api.memory_delete())
    check("归档增删改全程不碰常驻缓存",
          ("u1", "private") in wap._resident_cache)

    wap._resident_cache.clear()
    db.set_resident("u1", "private", "user", "缓存里的旧文本")
    wap._resident_block("u1", "private")          # 预热缓存
    payload.clear()
    payload.update({"owner_id": "u1", "scope": "private",
                    "block": "user", "text": "WebUI 改过的词条文本"})
    wap._resident_cache[("u1", "private")] = (time.time(), {"stale": 1})
    asyncio.run(api.resident_save())
    check("resident_save 后该（人，场景）缓存失效",
          ("u1", "private") not in wap._resident_cache)
    check("清缓存后读到新文本",
          wap._resident_block("u1", "private").get("user") == "WebUI 改过的词条文本")

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
