# -*- coding: utf-8 -*-
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ASTROOT = r"D:\Local\AstrBot\backend\app"
if os.path.isdir(ASTROOT) and ASTROOT not in sys.path:
    sys.path.insert(0, ASTROOT)

# import astrbot 会以当前工作目录为根生成 data/（cmd_config.json、t2i 模板），
# 落在插件目录里会被审计脚本判成「运行时文件入库」。先切到临时目录再导入。
import tempfile  # noqa: E402

os.chdir(tempfile.mkdtemp(prefix="hermes-cwd-"))

from modules.router import (  # noqa: E402
    ModelError, ModelRouter, retry_after_of,
)

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

    async def text_chat(self, prompt=None, session_id=None):
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
    def __init__(self, providers: dict, cfg: dict):
        self.cfg = cfg
        self.db = None
        self.context = types.SimpleNamespace(
            provider_manager=types.SimpleNamespace(
                inst_map=providers,
                providers_config=[
                    {"id": p, "enable": True, "max_context_tokens": 32000,
                     "model": p, "provider_type": "chat_completion"}
                    for p in providers
                ],
            ),
            get_using_provider=lambda: None,
        )

    def _get_cfg(self, key, default=None):
        return self.cfg.get(key, default)


def make_router(cfg: dict, providers: dict) -> ModelRouter:
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
    return ModelRouter(FakePlugin(providers, base))


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
    left = r.cooldown_left("a")
    check("cooldown=5 时限流仍冷却 >=120s", left >= 120, f"{left}s")

    a2 = FakeProvider("a", [TimeoutError("request timeout")])
    r2 = make_router({"llm_rpm": 0, "llm_min_interval": 0.0,
                      "llm_cooldown_seconds": 5, "llm_max_retries": 0,
                      "llm_total_attempts": 1}, {"a": a2, "b": FakeProvider("b")})
    try:
        await r2.chat("hi")
    except ModelError:
        pass
    left2 = r2.cooldown_left("a")
    check("cooldown=5 时超时仍冷却 >=90s", left2 >= 90, f"{left2}s")


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
               case_rate_cooldown, case_retry_after_parse, case_deadline_stop,
               case_all_cooling, case_cooldown_floor, case_soft_deadline):
        try:
            await fn()
        except Exception as exc:
            FAIL.append(fn.__name__)
            print(f"  FAIL {fn.__name__} 抛异常: {exc!r}")
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
