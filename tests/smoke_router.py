"""模型路由冒烟测试：节流、限流退避、总预算。

    D:/Local/AstrBot/backend/python/python.exe tests/smoke_router.py

只验证「会不会把自己打成 429」和「会不会白等」这两件事，
不联网、不依赖 AstrBot 运行时（只借用它的 logger 模块）。
"""

import asyncio
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import enable_astrbot_backend, router_module

enable_astrbot_backend()

_router = router_module()
ModelError = _router.ModelError
ModelRouter = _router.ModelRouter
retry_after_of = _router.retry_after_of

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))


class RateLimited(Exception):
    status_code = 429

    def __init__(self, retry_after: str = ""):
        super().__init__("429 Too Many Requests: rate limit exceeded, try again in 5s")
        self.headers = {"retry-after": retry_after} if retry_after else {}


class FakeProvider:
    def __init__(self, pid: str, plan=None):
        self.pid = pid
        self.calls: list[float] = []
        self.plan = list(plan or [])

    async def text_chat(self, prompt=None, session_id=None, request_max_retries=None):
        self.calls.append(time.time())
        if self.plan:
            action = self.plan.pop(0)
            if isinstance(action, BaseException):
                raise action
            if action == "sleep":
                await asyncio.sleep(30)
            return action
        return f"ok from {self.pid}"

    def meta(self):
        return types.SimpleNamespace(
            id=self.pid, model=self.pid,
            provider_type="chat_completion", type="openai_chat_completion",
        )


class FakePlugin:
    def __init__(self, providers: dict, cfg: dict, current=None):
        self.cfg = cfg
        self.db = None
        self.current = current  # 模拟 AstrBot 主体的「当前对话模型」
        self.context = types.SimpleNamespace(
            provider_manager=types.SimpleNamespace(
                inst_map=providers,
                providers_config=[
                    {"id": p, "enable": True, "max_context_tokens": 32000,
                     "model": p, "provider_type": "chat_completion"}
                    for p in providers
                ],
            ),
            get_using_provider=lambda: self.current,
        )

    def _get_cfg(self, key, default=None):
        return self.cfg.get(key, default)


def make_router(cfg: dict, providers: dict, current=None) -> ModelRouter:
    base = {
        "llm_primary_provider": next(iter(providers)),
        "llm_fallback_providers": list(providers)[1:],
        "llm_max_retries": 2,
        "llm_total_attempts": 6,
        "llm_timeout": 5,
        "llm_backoff": 0.0,
        "llm_cooldown_seconds": 60,
        "llm_rpm": 0,
        "llm_min_interval": 0.0,
    }
    base.update(cfg)
    return ModelRouter(FakePlugin(providers, base, current))


async def case_decoupled() -> None:
    """主模型必须只认插件配置，绝不跟随 AstrBot 主体的对话模型。

    记忆是长期数据：主体里换个对话模型，后台总结与深度整理如果跟着换，
    归纳风格、上下文预算、调用成本都会无声漂移，事后无法归因。
    """
    a, b = FakeProvider("a"), FakeProvider("b")
    provs = {"a": a, "b": b}

    # 1) 插件没配主模型：即使主体有当前对话模型，也不许回退过去
    r = make_router({"llm_primary_provider": "", "llm_fallback_providers": []},
                    provs, current=b)
    check("未配置主模型时不跟随主体", r.effective_chain() == [],
          str([e["id"] for e in r.effective_chain()]))
    try:
        await r.chat("hi", purpose="测试")
        check("未配置应拒绝调用", False, "竟然调通了")
    except ModelError as exc:
        check("未配置时错误类型", exc.kind == "no_provider", exc.kind)
        check("报错指向插件模型页", "模型" in str(exc) and "AstrBot" in str(exc), str(exc))
    check("未配置时没打到任何模型", not a.calls and not b.calls, str((a.calls, b.calls)))

    # 2) 显式配了主模型：主模型改为谁都不影响链路
    r2 = make_router({"llm_primary_provider": "a", "llm_fallback_providers": ["b"]},
                     provs, current=b)
    check("显式配置后链固定", [e["id"] for e in r2.effective_chain()] == ["a", "b"],
          str([e["id"] for e in r2.effective_chain()]))
    r2.plugin.current = a  # 用户在主体里换对话模型
    check("主体换模型不影响链路",
          [e["id"] for e in r2.effective_chain()] == ["a", "b"],
          str([e["id"] for e in r2.effective_chain()]))
    r2.plugin.current = None
    check("主体关掉对话模型也不影响链路",
          [e["id"] for e in r2.effective_chain()] == ["a", "b"],
          str([e["id"] for e in r2.effective_chain()]))

    # 3) 真的跑一次，落款必须是插件配的主模型
    await r2.chat("hi", purpose="测试")
    check("调用走插件主模型", a.calls and not b.calls, str((len(a.calls), len(b.calls))))
    check("落款标注主模型", "（主模型）" in r2.footer(), r2.footer())

    # 4) 解耦之后的新故障形态：主体里把选中的 provider 删了/停用了
    r3 = make_router({"llm_primary_provider": "gone", "llm_fallback_providers": []},
                     provs, current=a)
    try:
        await r3.chat("hi", purpose="测试")
        check("选中的模型不存在时应报错", False, "竟然调通了")
    except ModelError as exc:
        check("报错点名缺失的模型", "gone" in str(exc), str(exc))
        check("报错说明不跟随主体", "不跟随" in str(exc), str(exc))


async def case_interval() -> None:
    a = FakeProvider("a")
    r = make_router({"llm_min_interval": 1.0, "llm_rpm": 0}, {"a": a})
    await r.chat("hi")
    await r.chat("hi")
    gap = a.calls[1] - a.calls[0]
    check("最小间隔生效（两次调用间隔 >= 1s）", gap >= 0.95, f"{gap:.2f}s")


async def case_rpm_deadline() -> None:
    a = FakeProvider("a")
    r = make_router({"llm_rpm": 1, "llm_min_interval": 0.0}, {"a": a})
    await r.chat("hi")
    t0 = time.monotonic()
    try:
        await r.chat("hi", deadline=t0 + 5)
        check("rpm 用尽时不等，直接报 deadline", False, "没有抛异常")
    except ModelError as exc:
        cost = time.monotonic() - t0
        check("rpm 用尽时不等，直接报 deadline", exc.kind == "deadline", exc.kind)
        check("rpm 用尽时没有白等（<1s）", cost < 1.0, f"{cost:.2f}s")
        check("rpm 用尽时没有多打一次", len(a.calls) == 1, str(len(a.calls)))


async def case_rate_no_retry() -> None:
    a = FakeProvider("a", [RateLimited(), RateLimited(), RateLimited()])
    b = FakeProvider("b")
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0}, {"a": a, "b": b})
    text = await r.chat("hi")
    check("429 后降级到下一个模型", text == "ok from b", text)
    check("429 不原地重试（主模型只打 1 次）", len(a.calls) == 1, str(len(a.calls)))


async def case_timeout_switch_model() -> None:
    """超时后必须立刻换模型，不能把 max_attempts 全耗在主模型上。

    实测日志：整理技能 max_attempts=2、per_model=2，两次 60s 超时全打在
    qwen3.7-flash 上（间隔 61.7s），配好的回退模型一次都没被调用。
    """
    a = FakeProvider("a", [TimeoutError("request timeout"),
                           TimeoutError("request timeout")])
    b = FakeProvider("b")
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0,
                     "llm_max_retries": 1, "llm_total_attempts": 2},
                    {"a": a, "b": b})
    text = await r.chat("hi", purpose="整理技能", max_attempts=2)
    check("超时后回退模型被调用", text == "ok from b", text)
    check("超时不原地重试（主模型只打 1 次）", len(a.calls) == 1, str(len(a.calls)))


async def case_timeout_single_model_retries() -> None:
    """只有一个模型时没得换，仍保留原地重试来治瞬时抖动。"""
    a = FakeProvider("a", [TimeoutError("request timeout"), "ok from a"])
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0,
                     "llm_max_retries": 1, "llm_total_attempts": 4},
                    {"a": a})
    text = await r.chat("hi")
    check("单模型超时仍原地重试", text == "ok from a", text)
    check("单模型重试打了 2 次", len(a.calls) == 2, str(len(a.calls)))


async def case_timeout_budget_exhausted() -> None:
    """总预算已用完就不该再换模型白打一次。"""
    a = FakeProvider("a", [TimeoutError("request timeout")])
    b = FakeProvider("b")
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0,
                     "llm_max_retries": 0, "llm_total_attempts": 1},
                    {"a": a, "b": b})
    try:
        await r.chat("hi")
        check("预算耗尽应失败", False, "竟然成功")
    except ModelError:
        check("预算耗尽应失败", True)
    check("预算耗尽不再换模型", not b.calls, str(len(b.calls)))


async def case_call_timeout() -> None:
    """chat 未显式传 timeout 时，单次超时必须来自 llm_timeout 配置。"""
    provs = {"a": FakeProvider("a")}
    check("policy timeout 读取配置",
          make_router({}, provs).policy()["timeout"] == 5.0)
    check("policy timeout 跟随 95",
          make_router({"llm_timeout": 95}, provs).policy()["timeout"] == 95.0)
    check("policy timeout 上限 600",
          make_router({"llm_timeout": 9999}, provs).policy()["timeout"] == 600.0)
    check("policy timeout 下限 5",
          make_router({"llm_timeout": 0.1}, provs).policy()["timeout"] == 5.0)


async def case_rate_cooldown() -> None:
    a = FakeProvider("a", [RateLimited("90")])
    b = FakeProvider("b")
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0}, {"a": a, "b": b})
    await r.chat("hi")
    left = r.cooldown_left("a")
    check("429 冷却尊重 Retry-After（>=85s）", left >= 85, f"{left}s")


async def case_retry_after_parse() -> None:
    check("Retry-After 头解析", retry_after_of(RateLimited("12")) == 12.0)
    check("正文 try again in 解析",
          abs(retry_after_of(Exception("Please try again in 7.5s")) - 7.5) < 0.01)
    check("挖不到返回 0", retry_after_of(Exception("boom")) == 0.0)


async def case_deadline_stop() -> None:
    a = FakeProvider("a", ["sleep", "sleep", "sleep", "sleep"])
    r = make_router({"llm_max_retries": 3, "llm_total_attempts": 6,
                     "llm_timeout": 30, "llm_cooldown_seconds": 0,
                     "llm_rpm": 0, "llm_min_interval": 0.0}, {"a": a})
    t0 = time.monotonic()
    try:
        await r.chat("hi", deadline=t0 + 1.5)
        check("总预算到点停止", False, "没有抛异常")
    except ModelError as exc:
        cost = time.monotonic() - t0
        check("总预算到点停止", exc.kind == "deadline", exc.kind)
        check("总预算没有明显超支（<4s）", cost < 4.0, f"{cost:.2f}s")


async def case_all_cooling() -> None:
    a = FakeProvider("a", [RateLimited("120")])
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0}, {"a": a})
    try:
        await r.chat("hi")   # 单模型，必然 all_failed；顺便把 a 打进冷却
    except ModelError:
        pass
    t0 = time.monotonic()
    try:
        await r.chat("hi", deadline=t0 + 30)
        check("全模型冷却时批处理直接放弃", False, "没有抛异常")
    except ModelError as exc:
        check("全模型冷却时批处理直接放弃", exc.kind == "deadline", exc.kind)
        check("全模型冷却时没有再撞一次", len(a.calls) == 1, str(len(a.calls)))


async def case_cooldown_floor() -> None:
    """冷却是系统防线，不该被过小的配置值架空。

    实测把 llm_cooldown_seconds 设成 5 之后，每片整理都能把三个坏模型重新撞
    一遍（4 片 = 24 次无效调用），比完全不冷却更糟。
    """
    a = FakeProvider("a", [RateLimited()])
    b = FakeProvider("b")
    r = make_router({"llm_rpm": 0, "llm_min_interval": 0.0,
                     "llm_cooldown_seconds": 5}, {"a": a, "b": b})
    await r.chat("hi")
    # cooldown_left 用 int() 向下取整：刚设完 120s 冷却立刻读也只能拿到 119，
    # 断言留 1 秒取整容差，仍能证明下限是 120 量级而不是被 5 秒配置架空。
    left = r.cooldown_left("a")
    check("cooldown=5 时限流仍冷却 >=120s", left >= 119, f"{left}s")

    a2 = FakeProvider("a", [TimeoutError("request timeout")])
    r2 = make_router({"llm_rpm": 0, "llm_min_interval": 0.0,
                      "llm_cooldown_seconds": 5, "llm_max_retries": 0,
                      "llm_total_attempts": 1}, {"a": a2, "b": FakeProvider("b")})
    try:
        await r2.chat("hi")
    except ModelError:
        pass
    left2 = r2.cooldown_left("a")
    check("cooldown=5 时超时仍冷却 >=90s", left2 >= 89, f"{left2}s")


async def case_soft_deadline() -> None:
    """单片预算用尽应报 unit_deadline，让调用方跳过这一片而不是停整批。"""
    a = FakeProvider("a", ["sleep", "sleep", "sleep"])
    r = make_router({"llm_max_retries": 2, "llm_total_attempts": 6,
                     "llm_timeout": 30, "llm_cooldown_seconds": 0,
                     "llm_rpm": 0, "llm_min_interval": 0.0}, {"a": a})
    t0 = time.monotonic()
    try:
        await r.chat("hi", deadline=t0 + 60, soft_deadline=t0 + 1.5)
        check("单片预算用尽抛 unit_deadline", False, "没有抛异常")
    except ModelError as exc:
        cost = time.monotonic() - t0
        check("单片预算用尽抛 unit_deadline", exc.kind == "unit_deadline", exc.kind)
        check("单片预算没有明显超支（<4s）", cost < 4.0, f"{cost:.2f}s")


async def main() -> int:
    for fn in (case_interval, case_rpm_deadline, case_rate_no_retry,
               case_timeout_switch_model, case_timeout_single_model_retries,
               case_timeout_budget_exhausted, case_call_timeout,
               case_rate_cooldown, case_retry_after_parse, case_deadline_stop,
               case_all_cooling, case_cooldown_floor, case_soft_deadline,
               case_decoupled):
        try:
            await fn()
        except Exception as exc:
            FAIL.append(fn.__name__)
            print(f"  FAIL {fn.__name__} 抛异常: {exc!r}")
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
