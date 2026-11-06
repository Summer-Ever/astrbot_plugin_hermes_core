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
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import enable_astrbot_backend, package_main

enable_astrbot_backend()

m = package_main()

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
    obj._deep_task = None
    obj._deep_scheduled = False
    obj._consolidating = False
    obj._ui_job_task = None
    obj._job_lock = threading.Lock()
    obj._job_src = "auto"
    obj._metrics = {}
    obj._metrics_dirty = 0
    return obj


def cfg_core(core, **cfg):
    """覆盖若干配置项，其余走默认值。"""
    core.config = types.SimpleNamespace(get=lambda k, d=None: cfg.get(k, d))
    return core


def seed_buckets(db) -> None:
    for i in range(60):
        db.add_memory("fact", f"用户甲第{i}条偏好记录", owner_id="UA",
                      visibility="private")
    for i in range(30):
        db.add_memory("fact", f"用户乙第{i}条偏好记录", owner_id="UB",
                      visibility="private")
    for i in range(20):
        db.add_memory("fact", f"群G1第{i}条公共事项", group_id="G1",
                      visibility="group")


# 作用域桶键。v1.0.1 起「用户桶」末尾多了一段群号：同一个人在 A 群和 B 群
# 各自的群可见记忆必须分开成两个桶，否则整理时会被并成一条，产物继承最早
# 那条的群号——B 群的内容就泄漏到 A 群、同时在 B 群消失。上面 seed 的
# UA / UB 都是私聊（无群号），所以末段是空串；群公共条目没有归属人，走
# 群桶，形态不变。
UA = ("user", "UA", "", "private", "")
UB = ("user", "UB", "", "private", "")
G1 = ("group", "G1", "", "group")
# 同一个人在两个群里的群可见记忆：必须落进两个不同的桶
UA_G1 = ("user", "UA", "", "group", "G1")
UA_G2 = ("user", "UA", "", "group", "G2")


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



async def case_mutex(tmp: str) -> None:
    """自动总结、手动命令、WebUI 三条路径不能同时对同一批记忆写库。"""
    core = new_core(tmp)
    seed(core.db, 4)
    started = asyncio.Event()
    calls = {"n": 0}

    class SlowRouter:
        async def chat(self, **_kw):
            calls["n"] += 1
            started.set()
            await asyncio.sleep(0.05)
            return "[]"

    core.router = SlowRouter()
    first = asyncio.create_task(core._consolidate_memories(deep=True))
    await asyncio.wait_for(started.wait(), timeout=5)
    second = await core._consolidate_memories(deep=True)
    await first
    check("整理互斥：第二次直接跳过", second.startswith("整理正在进行中"), second)
    check("第一次不受影响跑完", calls["n"] == 1, str(calls["n"]))
    check("跑完后释放互斥", core._consolidating is False)


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


async def case_deep_bucket_count(tmp: str) -> None:
    """按作用域水位计数：只数「自该桶水位以来新增、且仍活跃」的记忆。"""
    core = cfg_core(new_core(tmp))
    seed_buckets(core.db)
    c = core._deep_pending_by_bucket()
    check("按 _scope_bucket 分桶计数", c.get(UA) == 60 and c.get(UB) == 30
          and c.get(G1) == 20, str(c))
    check("桶键可往返解析", m._bucket_from_key(m._bucket_key(G1)) == G1)

    core._set_deep_watermarks({m._bucket_key(UA): 60})
    c2 = core._deep_pending_by_bucket()
    check("水位只过滤本桶", c2.get(UA, 0) == 0
          and c2.get(UB) == 30 and c2.get(G1) == 20, str(c2))

    core._set_deep_watermarks({})
    with core.db._lock:
        core.db.conn.execute("UPDATE memories SET block = 'rules' WHERE id = 1")
        core.db.conn.commit()
    c3 = core._deep_pending_by_bucket()
    check("常驻词条条目不计入待整理", c3.get(UA) == 59, str(c3.get(UA)))

    ids = [r["id"] for r in core.db.all_active_memories()
           if (r["owner_id"] or "") == "UA" and not (r["block"] or "")][:20]
    core.db.merge_memories(ids, "用户甲的偏好汇总", "fact")
    c4 = core._deep_pending_by_bucket()
    check("被合并的记忆不再撑高计数", c4.get(UA) == 40, str(c4.get(UA)))


async def case_deep_bucket_cross_group(tmp: str) -> None:
    """同一个人在两个群里的群可见记忆必须分开成两个桶。

    v1.0.1 之前用户桶不带群号，A 群与 B 群的记忆落进同一个桶，整理时会被并成
    一条、产物继承最早那条的群号——B 群的内容泄漏到 A 群，同时在 B 群消失。
    """
    core = cfg_core(new_core(tmp))
    for i in range(5):
        core.db.add_memory("fact", f"甲在G1第{i}条", owner_id="UA",
                           group_id="G1", visibility="group")
        core.db.add_memory("fact", f"甲在G2第{i}条", owner_id="UA",
                           group_id="G2", visibility="group")
    rows = core.db.active_memory_scope_rows(0)
    b1 = m._scope_bucket(rows[0])
    b2 = m._scope_bucket(rows[-1])
    check("同一个人跨群的群可见记忆不同桶", b1 != b2, f"{b1} vs {b2}")
    check("桶键带各自的群号", b1[-1] == "G1" and b2[-1] == "G2",
          f"{b1} / {b2}")
    c = core._deep_pending_by_bucket()
    check("跨群分别计数", c.get(UA_G1) == 5 and c.get(UA_G2) == 5, str(c))
    # 水位键的老格式（没有群号）仍要能回退命中，升级后不会把历史记忆重整一遍
    old_key = m._bucket_key(("user", "UA", "", "group"))
    check("老水位键可回退命中新桶",
          m._wm_lookup({old_key: 3}, UA_G1) == 3,
          str(m._wm_lookup({old_key: 3}, UA_G1)))
    check("精确键优先于老键",
          m._wm_lookup({old_key: 3, m._bucket_key(UA_G1): 9}, UA_G1) == 9,
          str(m._wm_lookup({old_key: 3, m._bucket_key(UA_G1): 9}, UA_G1)))


async def case_deep_plan(tmp: str) -> None:
    """触发判据：单桶达标、限量、关闭、互斥、碎片化兜底。"""
    core = cfg_core(new_core(tmp), consolidate_deep_every_n_memories=50)
    core.router = FakeRouter(lambda i, p: "[]")
    seed_buckets(core.db)
    check("只有达标的桶入选", core._plan_auto_deep() == {UA},
          str(core._plan_auto_deep()))
    cfg_core(core, consolidate_deep_every_n_memories=0)
    check("阈值设 0 关闭自动深度整理", core._plan_auto_deep() is None)

    core2 = cfg_core(new_core(tmp + "-x"), consolidate_deep_every_n_memories=10)
    core2.router = FakeRouter(lambda i, p: "[]")
    seed_buckets(core2.db)
    core2.DEEP_MAX_BUCKETS = 2
    check("单次调度限量", len(core2._plan_auto_deep() or ()) == 2)
    core2._deep_scheduled = True
    check("已调度时不重复触发", core2._plan_auto_deep() is None)
    core2._deep_scheduled = False
    core2._consolidating = True
    check("已有整理在跑时不触发", core2._plan_auto_deep() is None)
    core2._consolidating = False

    core3 = cfg_core(new_core(tmp + "-y"), consolidate_deep_every_n_memories=61)
    core3.router = FakeRouter(lambda i, p: "[]")
    seed_buckets(core3.db)
    core3.DEEP_FALLBACK_RATIO = 1
    check("碎片化兜底：无单桶达标但全局够多",
          len(core3._plan_auto_deep() or ()) == 3)
    core3.DEEP_FALLBACK_RATIO = 100
    check("全局累计也不够时不触发", core3._plan_auto_deep() is None)


async def case_deep_schedule(tmp: str) -> None:
    """后台任务：buckets 过滤、快照水位、调度标志复位、prompt 别名约束。"""
    core = cfg_core(new_core(tmp), consolidate_deep_every_n_memories=50,
                    consolidate_chunk_size=40)
    router = FakeRouter(lambda i, p: "[]")
    core.router = router
    seed_buckets(core.db)
    snapshot = core.db.max_memory_id()
    core.db.add_memory("fact", "整理期间新写的记忆", owner_id="UA",
                       visibility="private")

    core._deep_scheduled = True
    task = asyncio.create_task(core._auto_deep_job({UA}, snapshot))
    core._deep_task = task
    task.add_done_callback(core._on_auto_deep_done)
    await task
    await asyncio.sleep(0.01)
    check("任务结束放行 scheduled", core._deep_scheduled is False)
    check("prompt 要求合并产物保留原始指称",
          any("保留各条原文用于指代同一事物的不同说法" in c["prompt"]
              for c in router.calls))
    check("buckets 只让达标桶进入模型",
          bool(router.calls) and all(
              "用户乙" not in c["prompt"] and "群G1" not in c["prompt"]
              for c in router.calls))
    wm = core._deep_watermarks()
    check("水位推进到整理开始时的快照 id",
          wm.get(m._bucket_key(UA)) == snapshot, str(wm))
    c = core._deep_pending_by_bucket()
    check("整理期间新增的记忆没被水位吞掉", c.get(UA, 0) >= 1, str(c))

    core._deep_scheduled = True
    cancelled = asyncio.create_task(asyncio.sleep(10))
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    core._on_auto_deep_done(cancelled)
    check("被 cancel 的任务也放行 scheduled", core._deep_scheduled is False)

    core.db.set_meta(core.DEEP_WM_KEY, '{"not-json-array": 5, "bad": "x"}')
    check("损坏的水位键被容错跳过", core._deep_watermarks() == {})

    core2 = cfg_core(new_core(tmp + "-busy"), consolidate_deep_every_n_memories=50)
    core2.router = FakeRouter(lambda i, p: "[]")
    seed_buckets(core2.db)
    core2._consolidating = True
    snap = core2.db.max_memory_id()
    await core2._auto_deep_job({UA}, snap)
    check("与另一条整理撞车时不推进水位", core2._deep_watermarks() == {})
    check("撞车不算模型失败、不触发退避",
          core2.db.get_meta("auto_consolidate_fail_streak", "0") in ("", "0"),
          core2.db.get_meta("auto_consolidate_fail_streak", "0"))


class FakeConfig(dict):
    """够用的配置对象：dict 语义 + 异步保存。"""

    def __init__(self, data=None):
        super().__init__(data or {})
        self.saves = 0

    async def save_config_async(self):
        self.saves += 1


class StubRouter:
    """_detach_model_chain 只用到这三个方法。"""

    def __init__(self, current="", known=("a", "b"), fallbacks=()):
        self.current = current
        self.known = list(known)
        self.fallbacks = list(fallbacks)

    def current_provider_id(self) -> str:
        return self.current

    def configured_fallbacks(self) -> list:
        return list(self.fallbacks)

    def all_providers(self) -> list:
        return [{"id": p, "loaded": True} for p in self.known]


def _detach_core(cfg: dict, astrbot_chain: dict, router: StubRouter):
    obj = m.HermesCore.__new__(m.HermesCore)
    obj.config = FakeConfig(cfg)
    model = {"provider_id": astrbot_chain.get("primary", ""),
             "fallback_provider_ids": astrbot_chain.get("fallbacks", [])}
    obj.context = types.SimpleNamespace(astrbot_config_mgr=types.SimpleNamespace(
        confs={"default": {"agent_runner": {"runner_type": "local",
                                            "config": {"model": model}}}}))
    obj.router = router
    return obj


async def case_detach(tmp: str) -> None:
    """主模型/降级链必须与 AstrBot 主体设置解耦。

    旧逻辑留空就每次读 get_using_provider()：主体换对话模型，后台总结跟着换。
    现在首次启动把当时的模型快照写进插件配置，之后主体怎么改都无关。
    """
    # 1) 空配置 → 快照一次，主模型与降级链都固化
    core = _detach_core({}, {"primary": "b", "fallbacks": ["a"]}, StubRouter("b"))
    await core._detach_model_chain()
    check("留空时快照主模型", core.config.get("llm_primary_provider") == "b",
          str(core.config))
    check("留空时快照降级链", core.config.get("llm_fallback_providers") == ["a"],
          str(core.config.get("llm_fallback_providers")))
    check("快照落盘一次", core.config.saves == 1, str(core.config.saves))

    # 2) 已显式配置 → 一个字都不动，也不写盘
    core = _detach_core({"llm_primary_provider": "a", "llm_fallback_providers": []},
                        {"primary": "b", "fallbacks": ["b"]}, StubRouter("b"))
    await core._detach_model_chain()
    check("已配置时不被主体覆盖", core.config["llm_primary_provider"] == "a",
          core.config["llm_primary_provider"])
    check("已配置时不再写盘", core.config.saves == 0, str(core.config.saves))

    # 3) 主体也没模型 → 不写空值，交给页面提示
    core = _detach_core({}, {"primary": "", "fallbacks": []}, StubRouter(""))
    await core._detach_model_chain()
    check("主体无模型时不写空配置", "llm_primary_provider" not in core.config,
          str(core.config))


async def case_skill_timeout(tmp: str) -> None:
    """整理技能的单次超时必须取自 router.call_timeout()，不能写死 60。

    以前 main.py 里 timeout=60.0 是常量，用户在「模型」页把 llm_timeout
    调到 120 也会被悄悄盖掉；整理记忆 / 深度整理记忆同样如此。
    """
    core = new_core(tmp)
    cfg_core(core, llm_timeout=95)
    check("整理超时 helper 跟随配置", core._llm_call_timeout() == 95.0)
    cfg_core(core, llm_timeout=9999)
    check("整理超时 helper 上限 600", core._llm_call_timeout() == 600.0)
    cfg_core(core, llm_timeout=0.1)
    check("整理超时 helper 下限 5", core._llm_call_timeout() == 5.0)
    cfg_core(core, llm_timeout=95)
    for i in (1, 2):
        core.db.upsert_skill(
            f"skill_{i}", "触发词",
            f"步骤{i}：这是一段用来走完整理流程的技能正文。" * 4,
            owner_id="U1")

    class FakeRouter:
        def __init__(self):
            self.kw = None

        def call_timeout(self):
            return 95.0

        async def chat(self, **kw):
            self.kw = kw
            return "[]"

    core.router = FakeRouter()
    await core._consolidate_skill_group(
        {"owner_id": "U1", "scope": "private", "group_id": ""}, dry_run=True)
    kw = core.router.kw
    check("整理技能走到模型调用", kw is not None, str(kw))
    check("整理技能 timeout 跟随配置",
          bool(kw) and kw.get("timeout") == 95.0,
          str(kw and kw.get("timeout")))
    check("整理技能仍保持 2 次预算",
          bool(kw) and kw.get("max_attempts") == 2,
          str(kw and kw.get("max_attempts")))
    check("整理技能仍带 deadline", bool(kw) and kw.get("deadline") is not None)


async def case_job_log(tmp: str) -> None:
    """页面整理日志：持久化、增量、裁剪、坏数据、后台任务状态与中断恢复。"""
    core = new_core(tmp)
    check("空日志快照", core.job_snapshot(0)["lines"] == [])
    s1 = core.job_emit("第一行", level="start", kind="memory")
    s2 = core.job_emit("第二行")
    check("seq 全局递增", s2 == s1 + 1)
    snap = core.job_snapshot(s1)
    check("增量只返回新行",
          len(snap["lines"]) == 1 and snap["lines"][0]["msg"] == "第二行")
    check("快照 seq 与最新一致", snap["seq"] == s2)
    stored = json.loads(core.db.get_meta("consolidate_job_log", "{}"))
    check("日志确实落库（刷新页面能拉回）", stored["seq"] == s2)

    for i in range(core.JOB_LOG_MAX + 5):
        core.job_emit(f"刷屏 {i}")
    lines = core.job_snapshot(0)["lines"]
    check("日志行数有上限", len(lines) == core.JOB_LOG_MAX, str(len(lines)))
    check("裁剪保留最新行", str(lines[-1]["msg"]).startswith("刷屏"))

    core.db.set_meta("consolidate_job_log", "{坏掉的 json")
    check("坏 JSON 不抛异常", core.job_snapshot(0)["lines"] == [])
    core.db.set_meta("consolidate_job_log", json.dumps(
        {"seq": 7, "lines": "不是列表", "job": "不是字典"}))
    bad = core.job_snapshot(0)
    check("坏类型字段容错", bad["lines"] == [] and bad["job"] == {})

    core.router = types.SimpleNamespace(has_model=lambda: False)
    check("未配模型时拒绝提交",
          core.start_consolidate_job("memory")["started"] is False)
    core.router = types.SimpleNamespace(has_model=lambda: True)
    check("未知整理类型被拒",
          core.start_consolidate_job("nope")["started"] is False)

    called: list = []

    async def fake_consolidate(dry_run=False, **_kw):
        called.append(dry_run)
        return "模拟整理完成：1 组"

    core._consolidate_memories = fake_consolidate
    core._requeue_after_consolidate = lambda: called.append("requeue")
    res = core.start_consolidate_job("memory", dry_run=True)
    check("提交立刻返回 job_id", res["started"] is True and bool(res.get("job_id")))
    check("运行中重复提交被拒",
          core.start_consolidate_job("skill")["started"] is False)
    check("运行中不允许清空日志", core.clear_job_log()["cleared"] is False)
    await core._ui_job_task
    await asyncio.sleep(0)
    snap = core.job_snapshot(0)
    check("任务结束状态为 done", snap["job"]["state"] == "done")
    check("最终结果落库", "模拟整理完成" in snap["job"]["result"])
    check("预览不重建整理队列", "requeue" not in called)
    check("日志有完成行", any(x["level"] == "ok" for x in snap["lines"]))
    check("日志标记来源为页面", any(x["src"] == "页面" for x in snap["lines"]))
    check("任务结束恢复来源标记", core._job_src == "auto")
    check("空闲可清空日志", core.clear_job_log()["cleared"] is True)
    after = core.job_snapshot(0)
    check("清空保留 seq（不会重放旧行）",
          after["seq"] == snap["seq"] and after["lines"] == [])

    async def boom(dry_run=False, **_kw):
        raise RuntimeError("模型炸了")

    core._consolidate_memories = boom
    core.start_consolidate_job("memory")
    await core._ui_job_task
    await asyncio.sleep(0)
    check("异常时状态为 failed",
          core.job_snapshot(0)["job"]["state"] == "failed")

    core.db.set_meta("consolidate_job_log", json.dumps(
        {"seq": 99, "lines": [],
         "job": {"id": "old", "state": "running", "result": ""}}))
    core._recover_interrupted_ui_job()
    check("重载后遗留 running 被标中断",
          core.job_snapshot(0)["job"]["state"] == "interrupted")


async def main() -> int:
    for fn in (case_chunk, case_resume, case_streak, case_unit_deadline_skip,
               case_merge, case_visibility, case_mutex, case_detach, case_footer,
               case_deep_bucket_count, case_deep_bucket_cross_group,
               case_deep_plan, case_deep_schedule,
               case_skill_timeout, case_job_log):
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
