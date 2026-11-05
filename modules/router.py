"""模型路由：主模型 + 有序降级链、逐模型重试、错误分类、超时与冷却。

为什么单独做一层：插件原来的总结/整理直接拿 AstrBot 的「当前对话模型」，
一旦那个模型笨、上下文短、或者临时抽风，总结就整批失败——而总结是后台跑的，
用户看不到报错，只会觉得"记忆怎么不涨"。

这一层的职责：
1. 认出 AstrBot 里用户配置的全部对话模型（provider_insts + providers_config）；
2. 按「主模型 → 降级 1 → 降级 2 …」顺序尝试，一个不行换下一个；
3. 单个模型内部还能重试若干次，指数退避，避免瞬时抖动直接降级；
4. 按异常类型决定"重试这个模型"还是"立刻换模型"（鉴权错重试毫无意义）；
5. 失败模型进入冷却期，别让每次总结都先撞一遍坏模型；
6. 按目标模型的上下文长度裁剪 prompt——小上下文模型也能用来兜底；
7. 把每次调用记进环形日志，页面上能看到谁成功、谁失败、为什么。

本模块不依赖 astrbot 的导入路径，用鸭子类型调用 provider.text_chat，
这样在纯 Python 环境里也能单测。
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import traceback
from collections import deque
from typing import Any, Callable

from astrbot.api import logger

# 错误类别 → 中文说明。顺序即 UI 展示顺序。
ERROR_LABELS = {
    "timeout": "请求超时",
    "conn": "连接失败",
    "rate": "限流/额度",
    "auth": "鉴权失败",
    "ctx": "上下文超长",
    "server": "服务端错误",
    "empty": "空响应",
    "other": "其他错误",
    "no_provider": "无可用模型",
}

# 默认允许重试的类别。鉴权错重试没有意义，上下文超长重试也一样，直接换模型。
DEFAULT_RETRY_ON = ["timeout", "conn", "rate", "server", "empty"]

# 触发「模型整体不可用」冷却的类别；ctx/empty 是本次请求的问题，不冷却模型。
COOLDOWN_KINDS = {"auth", "rate", "server", "conn", "timeout"}


class ModelError(Exception):
    """所有模型都失败时抛出，message 可直接回给用户。"""

    def __init__(self, kind: str, message: str, tried: list[dict] | None = None):
        super().__init__(message)
        self.kind = kind
        self.tried = tried or []


class _EmptyResponse(Exception):
    """provider 返回了空内容。当作可重试的 empty 处理。"""


def classify_error(exc: BaseException) -> str:
    """把各家 SDK 五花八门的异常归成有限几类。

    依据优先级：具体的业务错误（上下文超长/鉴权/限流）先判，再判传输层
    （超时/连接），最后才是兜底。只看类名和消息，不依赖具体 SDK 类型，
    这样 OpenAI/Anthropic/Gemini 各家适配器都能覆盖。
    """
    name = type(exc).__name__.lower()
    text = str(exc) or ""
    low = text.lower()

    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None

    # 上下文超长：各家文案不同，先单独识别（重试无意义，但换大模型有用）
    if "context_length_exceeded" in low or "maximum context" in low:
        return "ctx"
    if "context" in low and ("length" in low or "window" in low or "too long" in low):
        return "ctx"
    if "too many tokens" in low or "reduce the length" in low or "prompt is too long" in low:
        return "ctx"

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name or "timed out" in low:
        return "timeout"

    if code == 429 or "ratelimit" in name or "rate limit" in low or "quota" in low \
            or "resource_exhausted" in low or "overloaded" in low:
        return "rate"

    if code in (401, 403) or "authentication" in name or "permissiondenied" in name \
            or "invalid api key" in low or "unauthorized" in low or "incorrect api key" in low \
            or "invalid_api_key" in low:
        return "auth"

    if (code is not None and 500 <= code < 600) or "internalserver" in name \
            or "serviceunavailable" in name or "badgateway" in name:
        return "server"

    if "connection" in name or "apiconnection" in name or any(
        k in low for k in ("connection", "network", "dns", "unreachable", "ssl", "eof", "proxy")
    ):
        return "conn"

    if isinstance(exc, _EmptyResponse) or "empty" in name:
        return "empty"
    if "emptymodel" in name or "no content" in low or "returned none" in low \
            or "empty response" in low:
        return "empty"

    return "other"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _completion_text(resp: Any) -> str:
    """兼容 LLMResponse 与直接返回字符串的适配器。"""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    text = getattr(resp, "completion_text", None)
    if text is None:
        text = getattr(resp, "text", None)
    return str(text or "")


def retry_after_of(exc: BaseException) -> float:
    """从 429 异常里挖出服务端要求的等待秒数，挖不到返回 0。

    服务端说「等 12 秒」，我们 3 秒后就重试，只会把限流窗口越撞越死——
    这是最容易把自己送进黑名单的写法。两种常见位置都认：
    HTTP 头 Retry-After，以及正文里的 "Please try again in 12.3s"。
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
    if headers:
        try:
            val = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            val = None
        if val:
            try:
                return max(0.0, min(600.0, float(val)))
            except (TypeError, ValueError):
                pass
    m = re.search(r"try again in\s*([\d.]+)\s*s", str(exc) or "", re.I)
    if m:
        try:
            return max(0.0, min(600.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.0


class _Throttle:
    """按 provider 的出站节流：最小间隔 + 每分钟次数上限。

    为什么必须有：整理是「逐桶串行调用」，桶与桶之间没有任何自然停顿，
    而免费/低配档模型普遍按分钟计 RPM。连着打必然 429，429 又触发重试，
    短时间内把配额窗口撞得越来越死。

    两层限制分别治两种病：
    - min_interval 防「瞬时连击」（多数风控看的是每秒/每几秒的突发）；
    - rpm 防「一分钟内打满」（免费档常见 10-20 RPM）。
    rpm=0 表示不限制。
    """

    def __init__(self):
        self._next_at: dict[str, float] = {}
        self._hits: dict[str, deque] = {}

    def _window(self, pid: str, now: float) -> deque:
        q = self._hits.get(pid)
        if q is None:
            q = deque()
            self._hits[pid] = q
        while q and now - q[0] > 60.0:
            q.popleft()
        return q

    def delay(self, pid: str, rpm: int, min_interval: float) -> float:
        """还要等多久才能打这个 provider。0 表示现在就可以。"""
        now = time.time()
        wait = 0.0
        nxt = self._next_at.get(pid, 0.0)
        if nxt > now:
            wait = nxt - now
        if rpm > 0:
            q = self._window(pid, now)
            if len(q) >= rpm:
                wait = max(wait, 60.0 - (now - q[0]) + 0.05)
        return wait

    def record(self, pid: str, min_interval: float) -> None:
        now = time.time()
        if min_interval > 0:
            self._next_at[pid] = now + min_interval
        self._window(pid, now).append(now)

    def used_last_minute(self, pid: str) -> int:
        return len(self._window(pid, time.time()))

    def reset(self) -> None:
        self._next_at.clear()
        self._hits.clear()


class ModelRouter:
    """按配置的模型链调用 LLM，负责重试、降级、冷却与日志。"""

    def __init__(self, plugin):
        self.plugin = plugin
        self._logs: deque[dict] = deque(maxlen=300)
        self._stats: dict[str, dict] = {}
        self._cooldown_until: dict[str, float] = {}
        self._throttle = _Throttle()
        self._last_used: str = ""
        self._last_skipped: list[dict] = []
        # 最近一次成功调用的落款信息：消息末尾的 `> 模型名（主/降级）` 从这里取。
        self._last_call: dict = {}
        # 汇总计数：页面上要能看出「主模型到底成不成、降级用了多少次」
        self.counters = {"calls": 0, "ok": 0, "fail": 0,
                         "attempts": 0, "fallback_ok": 0, "throttled": 0}

    # ---------- 配置 ----------

    def _cfg(self, key: str, default=None):
        try:
            return self.plugin._get_cfg(key, default)
        except Exception:
            return default

    def _debug_trace(self) -> bool:
        """逐次调用的明细日志（调用开始 / 成功耗时）。

        默认关闭：正常运行时每次总结、每片整理都打一行会把日志刷满。
        排查「为什么全都在超时」时打开，就能看出是发不出去还是生成长。
        """
        return bool(self._cfg("llm_debug_trace", False))

    # ---------- 落款：这条回复是哪个模型给的 ----------

    @staticmethod
    def tier_label(index: int) -> str:
        """0 = 主模型，1.. = 降级第几个。"""
        return "主模型" if index <= 0 else f"降级 {index}"

    def last_call_info(self) -> dict:
        return dict(self._last_call or {})

    def footer(self, extra: str = "") -> str:
        """消息末行的模型落款：`> 供应商/模型（主模型）`。

        群聊里 `>` 不一定渲染成引用，但作为普通文本也读得懂，
        所以不去掉符号——用户要的就是这个格式。
        """
        info = self._last_call or {}
        pid = str(info.get("provider") or "")
        if not pid:
            return ""
        tier = self.tier_label(int(info.get("tier") or 0))
        suffix = f"，{extra}" if extra else ""
        return f"> {pid}（{tier}{suffix}）"

    def policy(self) -> dict:
        retry_on = self._cfg("llm_retry_on", DEFAULT_RETRY_ON)
        if isinstance(retry_on, str):
            retry_on = [x.strip() for x in retry_on.split(",") if x.strip()]
        if not isinstance(retry_on, (list, tuple)):
            retry_on = DEFAULT_RETRY_ON
        retry_on = [k for k in retry_on if k in ERROR_LABELS]
        return {
            # 后台任务（总结/整理）不需要 3 次容错：链上还有别的模型，
            # 在一个超时的模型上重试 3 次 = 白等 90 秒。重试只治瞬时抖动。
            "max_retries": max(0, min(9, _as_int(self._cfg("llm_max_retries", 1), 1))),
            # 30 秒：正常归纳 5-20 秒出结果；30 秒一个字不吐的，多半是免费档排队
            # 或者根本不通，给 90 秒只是把整次整理拖长。
            # 实测长 prompt：deepseek 9.5s、qwen3.7-flash 32.5s、nemotron 51.9s。
            # 给 60 秒才够降级模型跑完；30 秒等于只允许主模型成功。
            "timeout": max(5.0, min(600.0, _as_float(self._cfg("llm_timeout", 60), 60))),
            "backoff": max(0.0, min(30.0, _as_float(self._cfg("llm_backoff", 1.5), 1.5))),
            "total_attempts": max(1, min(30, _as_int(self._cfg("llm_total_attempts", 4), 4))),
            "cooldown": max(0, min(3600, _as_int(self._cfg("llm_cooldown_seconds", 180), 180))),
            "max_prompt_chars": max(2000, min(200000, _as_int(self._cfg("llm_max_prompt_chars", 24000), 24000))),
            "retry_on": retry_on or DEFAULT_RETRY_ON,
            "rpm": max(0, min(600, _as_int(self._cfg("llm_rpm", 10), 10))),
            "min_interval": max(0.0, min(60.0, _as_float(self._cfg("llm_min_interval", 1.5), 1.5))),
        }

    def configured_primary(self) -> str:
        return str(self._cfg("llm_primary_provider", "") or "").strip()

    def configured_fallbacks(self) -> list[str]:
        raw = self._cfg("llm_fallback_providers", []) or []
        if isinstance(raw, str):
            raw = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]
        if not isinstance(raw, (list, tuple)):
            return []
        out: list[str] = []
        for item in raw:
            pid = str(item or "").strip()
            if pid and pid not in out:
                out.append(pid)
        return out

    # ---------- provider 发现 ----------

    def _manager(self):
        return getattr(getattr(self.plugin, "context", None), "provider_manager", None)

    def _configs(self) -> dict[str, dict]:
        pm = self._manager()
        rows = getattr(pm, "providers_config", None) or []
        out: dict[str, dict] = {}
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("id"):
                    out[str(row["id"])] = row
        return out

    def _insts(self) -> dict[str, Any]:
        pm = self._manager()
        insts = getattr(pm, "inst_map", None) or {}
        return dict(insts) if isinstance(insts, dict) else {}

    def all_providers(self) -> list[dict]:
        """列出 AstrBot 里配置的全部对话模型（含停用/未加载的，供页面说明）。"""
        configs = self._configs()
        insts = self._insts()
        current = self.current_provider_id()
        rows: list[dict] = []
        seen: set[str] = set()

        def build(pid: str, cfg: dict, inst) -> dict:
            typ = str(cfg.get("type") or "")
            ptype = str(cfg.get("provider_type") or "chat_completion")
            model = str(cfg.get("model") or "")
            if inst is not None:
                try:
                    meta = inst.meta()
                    model = str(getattr(meta, "model", "") or model)
                    ptype = str(getattr(meta, "provider_type", "") or ptype)
                    typ = str(getattr(meta, "type", "") or typ)
                except Exception:
                    pass
            ctx = _as_int(getattr(getattr(inst, "provider_config", None), "get", lambda *_: 0)(
                "max_context_tokens", 0), 0) if inst is not None else 0
            return {
                "id": pid,
                "model": model,
                "source": str(cfg.get("provider_source_id") or ""),
                "type": typ,
                "provider_type": ptype,
                "enabled": bool(cfg.get("enable", True)),
                "loaded": inst is not None,
                "context_tokens": ctx or _as_int(cfg.get("max_context_tokens"), 0),
                "is_current": pid == current,
                "cooling": self.cooldown_left(pid),
                "stats": self._stats.get(pid, {}),
            }

        # 按 AstrBot 配置顺序，页面顺序和用户熟悉的顺序一致
        for pid, cfg in configs.items():
            cfg_ptype = str(cfg.get("provider_type") or "").lower()
            if cfg_ptype and "chat" not in cfg_ptype:
                continue  # embedding / stt / tts 不是对话模型
            seen.add(pid)
            rows.append(build(pid, cfg, insts.get(pid)))

        # 实例里有、配置里没有的（热加载进来的临时 provider）也补上
        for pid, inst in insts.items():
            if pid in seen:
                continue
            try:
                meta = inst.meta()
            except Exception:
                continue
            ptype = str(getattr(meta, "provider_type", "") or "").lower()
            typ = str(getattr(meta, "type", "") or "").lower()
            if ptype and "chat" not in ptype and "chat" not in typ:
                continue
            rows.append(build(pid, configs.get(pid, {}), inst))
        return rows

    def current_provider_id(self) -> str:
        try:
            prov = self.plugin.context.get_using_provider()
        except Exception:
            prov = None
        if prov is None:
            return ""
        try:
            return str(prov.meta().id or "")
        except Exception:
            return str(getattr(prov, "provider_config", {}).get("id", "") or "")

    def _inst(self, pid: str):
        return self._insts().get(pid)

    def has_model(self) -> bool:
        return bool(self.effective_chain(ignore_cooldown=True))

    # ---------- 链与冷却 ----------

    def effective_chain(self, ignore_cooldown: bool = False,
                        strict_cooldown: bool = False) -> list[dict]:
        """主模型 + 降级链，过滤掉不存在/未加载/冷却中的。

        strict_cooldown=True 时，全部模型都在冷却就返回空链（调用方直接放弃），
        不再「宁可撞一下」——批处理场景里撞一下的代价是一次 45 秒空等，
        而对面明明刚说过限流。
        """
        configs = self._configs()
        insts = self._insts()
        # 主模型只认插件自己的配置，绝不回退到 AstrBot 的「当前对话模型」。
        # 回退会造成静默漂移：用户在主体里换个对话模型，后台总结与深度整理的
        # 归纳风格、上下文预算、调用成本全跟着变，而记忆是长期数据，
        # 换模型前后的条目质量不一致，事后根本查不出是哪次换的。
        primary = self.configured_primary()
        wanted: list[str] = []
        if primary:
            wanted.append(primary)
        for pid in self.configured_fallbacks():
            if pid not in wanted:
                wanted.append(pid)

        chain: list[dict] = []
        skipped: list[dict] = []
        for pid in wanted:
            inst = insts.get(pid)
            if inst is None:
                skipped.append({"id": pid, "reason": "未加载（未启用或不存在）"})
                continue
            left = self.cooldown_left(pid)
            if left > 0 and not ignore_cooldown:
                skipped.append({"id": pid, "reason": f"冷却中 {left}s"})
                continue
            cfg = configs.get(pid, {})
            chain.append({
                "id": pid,
                "inst": inst,
                "context_tokens": _as_int(cfg.get("max_context_tokens"), 0),
                "model": cfg.get("model", "") or pid,
            })

        # 全部在冷却里：宁可撞一下也别直接不干活（批处理场景除外）
        if not chain and skipped and not strict_cooldown:
            for pid in wanted:
                inst = insts.get(pid)
                if inst is None:
                    continue
                cfg = configs.get(pid, {})
                chain.append({
                    "id": pid,
                    "inst": inst,
                    "context_tokens": _as_int(cfg.get("max_context_tokens"), 0),
                    "model": cfg.get("model", "") or pid,
                })
        self._last_skipped = skipped
        return chain

    def cooldown_left(self, pid: str) -> int:
        until = self._cooldown_until.get(pid, 0)
        return max(0, int(until - time.time()))

    def _mark_failed(self, pid: str, kind: str, exc: BaseException | None = None) -> None:
        if kind not in COOLDOWN_KINDS:
            return
        sec = self.policy()["cooldown"]
        if sec <= 0:
            return
        if kind == "auth":
            sec *= 4  # 鉴权失败修起来最慢，别频繁撞
        if kind == "rate":
            # 限流冷却要比普通失败更久，且尊重服务端给的 Retry-After：
            # 对面说等 60 秒，我们 180 秒后再来才安全。
            sec = max(sec * 3, int(retry_after_of(exc) if exc else 0))
        # 系统硬下限：这几个错误说明「对面现在真的不行」，冷却过短等于没有冷却。
        # 实测把 llm_cooldown_seconds 设成 5 之后，每一片整理都能把三个坏模型
        # 重新撞一遍，4 片就是 24 次无效调用——比不冷却更糟。
        if kind == "timeout":
            sec = max(sec, 90)
        elif kind == "rate":
            sec = max(sec, 120)
        elif kind == "auth":
            sec = max(sec, 300)
        self._cooldown_until[pid] = time.time() + sec
        self._persist_cooldowns()

    # ---- 冷却跨重启：否则每次重载都先拿坏模型撞一遍 ----

    _COOLDOWN_META = "llm_cooldown_until"

    def _persist_cooldowns(self) -> None:
        db = getattr(self.plugin, "db", None)
        if db is None:
            return
        now = time.time()
        alive = {k: int(v) for k, v in self._cooldown_until.items() if v > now}
        self._cooldown_until = {k: float(v) for k, v in alive.items()}
        try:
            db.set_meta(self._COOLDOWN_META, json.dumps(alive))
        except Exception:
            pass

    def load_cooldowns(self) -> int:
        """启动时把上次遗留的冷却读回来，返回仍在冷却的模型数。"""
        db = getattr(self.plugin, "db", None)
        if db is None:
            return 0
        try:
            raw = db.get_meta(self._COOLDOWN_META, "")
            data = json.loads(raw) if raw else {}
        except Exception:
            return 0
        if not isinstance(data, dict):
            return 0
        now = time.time()
        for pid, until in data.items():
            try:
                ts = float(until)
            except (TypeError, ValueError):
                continue
            if ts > now:
                self._cooldown_until[str(pid)] = ts
        return len(self._cooldown_until)

    def clear_cooldowns(self) -> None:
        self._cooldown_until.clear()

    # ---------- 上下文预算 ----------

    def char_budget(self, context_tokens: int) -> int:
        """按模型的上下文长度给出可用的 prompt 字符数。

        中文大约 1 字 1 token（Qwen 更省、GPT 更贵），只用到上下文的一半多一点，
        剩下的留给输出和模板；再压一个全局上限，避免 1M 上下文的模型
        把整段历史都吞进去烧钱。
        """
        pol = self.policy()
        tokens = context_tokens if context_tokens > 0 else 8192
        budget = int(tokens * 0.55) - 800
        return max(600, min(pol["max_prompt_chars"], budget))

    # ---------- 主入口 ----------

    @staticmethod
    async def _invoke(inst, prompt: str):
        """调用 provider，并把 AstrBot 的内部重试压到 1 次。

        AstrBot 的 openai provider 默认在内部重试 5 次（指数退避，单次最长等
        30 秒）。对本插件是灾难：一次 429 会变成 1-2 分钟的静默等待，外层只看到
        「超时」，既认不出是限流、也无法及时换模型。退避与换模型统一由本层负责。

        老版本 provider 不认这个参数，会抛 TypeError，退回普通调用即可。
        """
        try:
            return await inst.text_chat(prompt=prompt, session_id=None,
                                        request_max_retries=1)
        except TypeError as exc:
            if "request_max_retries" not in str(exc):
                raise
            return await inst.text_chat(prompt=prompt, session_id=None)

    async def chat(
        self,
        prompt: str | None = None,
        *,
        purpose: str = "调用",
        builder: Callable[[int], str] | None = None,
        ignore_cooldown: bool = False,
        timeout: float | None = None,
        max_attempts: int | None = None,
        deadline: float | None = None,
        soft_deadline: float | None = None,
    ) -> str:
        """按链调用，返回第一个成功模型的文本。

        builder(char_budget) 用于按目标模型上下文重新生成 prompt；
        不传就直接用 prompt。

        max_attempts 用来给「多桶串行」的批处理（如深度整理）压总预算：
        全库整理会一个作用域一次调用，若每次都能烧满 8 次尝试，光超时就能耗掉
        十几分钟，而调用方的消息早就过了平台有效期。

        两级 deadline（都是 time.monotonic() 的绝对时刻）：
        - deadline：整批的总闸。用尽抛 ModelError(kind="deadline")，调用方应停止整批。
        - soft_deadline：单个任务单元（如一片记忆）的硬顶。用尽抛
          ModelError(kind="unit_deadline")，调用方应跳过这个单元、继续下一个。
        分开是因为两者的正确反应完全不同：总闸到点该整批收工，单片到点只该放弃那片。
        """
        pol = self.policy()
        hard, soft = deadline, soft_deadline

        def cutoff() -> float | None:
            vals = [d for d in (hard, soft) if d is not None]
            return min(vals) if vals else None

        def remaining() -> float | None:
            c = cutoff()
            return None if c is None else c - time.monotonic()

        def expired_kind() -> str:
            """当前更紧的是哪一级预算：总闸（deadline）还是单片（unit_deadline）。

            调用方对这两种的反应不同——总闸到点整批收工，单片到点只放弃这一片。
            """
            now = time.monotonic()
            hard_left = (hard - now) if hard is not None else float("inf")
            soft_left = (soft - now) if soft is not None else float("inf")
            return "deadline" if hard_left <= soft_left else "unit_deadline"

        left = remaining()
        if left is not None and left <= 1.0:
            raise ModelError(expired_kind(), "本次整理的时间预算已用尽（未开始调用）")
        chain = self.effective_chain(ignore_cooldown=ignore_cooldown,
                                     strict_cooldown=cutoff() is not None)
        if not chain:
            if cutoff() is not None and self._last_skipped:
                raise ModelError(expired_kind(),
                                 "所有模型都在限流/失败冷却中，先不调用以免加重风控："
                                 + "；".join(f"{s['id']} {s['reason']}"
                                            for s in self._last_skipped[:3]))
            if self._last_skipped:
                # 解耦之后这条路径会真的出现：用户在 AstrBot 主体里删掉/停用了
                # 本插件选中的 provider。必须把 provider ID 打出来，否则只会看到
                # 「没有可用模型」，根本想不到是主体那边动了模型列表。
                raise ModelError("no_provider",
                                 "本插件选定的总结模型不可用："
                                 + "；".join(f"{s['id']} {s['reason']}"
                                            for s in self._last_skipped[:3])
                                 + "。请到 Hermes 插件页「模型」重新选择"
                                 "（本插件不跟随 AstrBot 主体的对话模型设置）。")
            raise ModelError("no_provider",
                             "本插件没有可用的总结模型：请到 AstrBot 的 Hermes 插件页 →"
                             "「模型」选择主模型（本插件不跟随 AstrBot 主体的对话模型设置）。")

        per_model = pol["max_retries"] + 1
        total_cap = pol["total_attempts"]
        if max_attempts is not None:
            total_cap = max(1, min(total_cap, int(max_attempts)))
        attempts = 0
        tried: list[dict] = []
        limit = timeout if timeout is not None else pol["timeout"]
        self.counters["calls"] += 1
        fallback_index = -1

        for entry in chain:
            fallback_index += 1
            pid = entry["id"]
            inst = entry["inst"]
            try:
                model_prompt = builder(self.char_budget(entry["context_tokens"])) if builder else (prompt or "")
            except Exception:
                logger.error(f"[hermes_core] 生成 prompt 失败:\n{traceback.format_exc()}")
                model_prompt = prompt or ""

            model_failed_kind = ""
            model_failed_exc: BaseException | None = None
            for i in range(per_model):
                if attempts >= total_cap:
                    break

                # 出站节流：整理逐桶串行，桶与桶之间没有自然停顿，
                # 不主动拉开间隔就会把免费档模型的分钟配额打满。
                wait = self._throttle.delay(pid, pol["rpm"], pol["min_interval"])
                if wait > 0:
                    left_now = remaining()
                    if left_now is not None and wait > left_now - 1.0:
                        # 等不起就别等：把剩下的预算留给下一次整理，
                        # 而不是在这里睡到超时再报错。
                        self.counters["throttled"] += 1
                        raise ModelError(
                            expired_kind(),
                            f"{pid} 触发限流保护（每分钟最多 {pol['rpm']} 次），"
                            f"还需等待 {int(wait)} 秒；本次整理先停在这里")
                    logger.info(f"[hermes_core] {purpose} 节流等待 {wait:.1f}s（{pid}）")
                    await asyncio.sleep(wait)

                left_now = remaining()
                if left_now is not None and left_now < 5.0:
                    # 剩下不到一次最短请求的时间，打出去也必然超时，直接收工。
                    raise ModelError(expired_kind(), "本次整理的时间预算已用尽")
                eff_limit = limit if left_now is None else min(limit, left_now)

                attempts += 1
                self.counters["attempts"] += 1
                self._throttle.record(pid, pol["min_interval"])
                started = time.time()
                if self._debug_trace():
                    logger.info(
                        f"[hermes_core] 调用开始 {purpose} model={pid} 第{i + 1}次 "
                        f"prompt={len(model_prompt)}字 limit={eff_limit:.0f}s")
                try:
                    resp = await asyncio.wait_for(
                        self._invoke(inst, model_prompt),
                        timeout=eff_limit,
                    )
                    text = _completion_text(resp)
                    if not text.strip():
                        raise _EmptyResponse("模型返回空内容")
                    self._log(purpose, pid, entry["model"], i + 1, started, True, "", "")
                    self._last_used = pid
                    self._last_call = {
                        "provider": pid,
                        "model": entry["model"],
                        "tier": fallback_index,
                        "ms": int((time.time() - started) * 1000),
                        "purpose": purpose,
                        "prompt_chars": len(model_prompt),
                    }
                    if self._debug_trace():
                        logger.info(
                            f"[hermes_core] 调用成功 {purpose} model={pid} "
                            f"{(time.time() - started):.1f}s 输出{len(text)}字")
                    self._cooldown_until.pop(pid, None)
                    self.counters["ok"] += 1
                    if fallback_index > 0:
                        self.counters["fallback_ok"] += 1
                    return text
                except Exception as exc:
                    kind = classify_error(exc)
                    msg = str(exc)[:300] or type(exc).__name__
                    model_failed_kind = kind
                    model_failed_exc = exc
                    self._log(purpose, pid, entry["model"], i + 1, started, False, kind, msg)
                    tried.append({"provider": pid, "kind": kind, "error": msg, "attempt": i + 1})
                    logger.warning(
                        f"[hermes_core] {purpose} 失败 model={pid} 第{i + 1}次 "
                        f"[{ERROR_LABELS.get(kind, kind)}] {msg[:160]}"
                    )
                    if kind == "rate":
                        # 429 / 额度用尽：原地重试只会把限流窗口越撞越死，
                        # 立刻换模型，同时按 Retry-After 给它一段更长的冷却。
                        logger.warning(f"[hermes_core] {pid} 被限流，立即换模型并冷却")
                        break
                    if kind not in pol["retry_on"]:
                        break  # 鉴权/超长等，重试无意义，换模型
                    if i < per_model - 1 and attempts < total_cap:
                        delay = min(10.0, pol["backoff"] * (2 ** i)) + random.uniform(0, 0.3)
                        left_now = remaining()
                        if left_now is not None and delay > left_now - 1.0:
                            raise ModelError(expired_kind(), "本次整理的时间预算已用尽")
                        if delay > 0:
                            await asyncio.sleep(delay)

            # 这个模型放弃了（重试耗尽或不可重试错误），进冷却，换下一个
            if model_failed_kind:
                self._mark_failed(pid, model_failed_kind, model_failed_exc)
            if attempts >= total_cap:
                break

        detail = "；".join(
            f"{t['provider']}（{ERROR_LABELS.get(t['kind'], t['kind'])}）"
            for t in tried[-4:]
        ) or "无尝试记录"
        self.counters["fail"] += 1
        raise ModelError("all_failed",
                         f"全部 {len(chain)} 个模型都失败了：{detail}。"
                         f"可在「模型」页检查配置或点「测试」定位问题。",
                         tried=tried)

    # ---------- 测试 ----------

    async def test(self, pid: str, timeout: float = 25.0,
                   prompt: str = "只回复两个字母：OK") -> dict:
        inst = self._inst(pid)
        if inst is None:
            return {"id": pid, "ok": False, "kind": "no_provider",
                    "error": "该模型未加载（可能未启用）", "ms": 0}
        started = time.time()
        try:
            resp = await asyncio.wait_for(
                self._invoke(inst, prompt),
                timeout=timeout,
            )
            text = _completion_text(resp).strip()
            ms = int((time.time() - started) * 1000)
            if not text:
                raise _EmptyResponse("模型返回空内容")
            self._log("连通性测试", pid, pid, 1, started, True, "", "")
            self._cooldown_until.pop(pid, None)
            return {"id": pid, "ok": True, "ms": ms, "reply": text[:80], "kind": "", "error": ""}
        except Exception as exc:
            ms = int((time.time() - started) * 1000)
            kind = classify_error(exc)
            msg = str(exc)[:300] or type(exc).__name__
            self._log("连通性测试", pid, pid, 1, started, False, kind, msg)
            return {"id": pid, "ok": False, "ms": ms, "kind": kind,
                    "error": msg, "label": ERROR_LABELS.get(kind, kind)}

    # ---------- 日志与统计 ----------

    def _log(self, purpose: str, pid: str, model: str, attempt: int,
             started: float, ok: bool, kind: str, error: str) -> None:
        ms = int((time.time() - started) * 1000)
        self._logs.appendleft({
            "ts": int(time.time()),
            "purpose": purpose,
            "provider": pid,
            "model": model,
            "attempt": attempt,
            "ms": ms,
            "ok": ok,
            "kind": kind,
            "label": ERROR_LABELS.get(kind, kind) if kind else "",
            "error": error[:300],
        })
        st = self._stats.setdefault(pid, {"calls": 0, "ok": 0, "fail": 0, "ms_sum": 0,
                                          "last_kind": "", "last_error": "", "last_ts": 0})
        st["calls"] += 1
        st["ms_sum"] += ms
        st["last_ts"] = int(time.time())
        if ok:
            st["ok"] += 1
        else:
            st["fail"] += 1
            st["last_kind"] = kind
            st["last_error"] = error[:200]
        st["avg_ms"] = int(st["ms_sum"] / st["calls"]) if st["calls"] else 0

    def logs(self, limit: int = 100) -> list[dict]:
        return list(self._logs)[:max(1, min(300, limit))]

    def clear_logs(self) -> None:
        self._logs.clear()
        self._stats.clear()

    def stats(self) -> dict:
        return self._stats

    # ---------- 给页面用的快照 ----------

    def snapshot(self) -> dict:
        pol = self.policy()
        providers = self.all_providers()
        chain = [e["id"] for e in self.effective_chain(ignore_cooldown=True)]
        return {
            "current_id": self.current_provider_id(),
            "primary": self.configured_primary(),
            "fallbacks": self.configured_fallbacks(),
            "effective_chain": chain,
            "last_used": self._last_used,
            # 对外统一用配置键名（llm_*），前端编辑后原样传回 models/save，
            # 两边的字段名保持一致，避免静默丢配置。
            "policy": {
                "llm_max_retries": pol["max_retries"],
                "llm_timeout": pol["timeout"],
                "llm_backoff": pol["backoff"],
                "llm_total_attempts": pol["total_attempts"],
                "llm_cooldown_seconds": pol["cooldown"],
                "llm_max_prompt_chars": pol["max_prompt_chars"],
                "llm_retry_on": pol["retry_on"],
                "llm_rpm": pol["rpm"],
                "llm_min_interval": pol["min_interval"],
            },
            "counters": dict(self.counters),
            "providers": providers,
            "logs": self.logs(80),
            "skipped": self._last_skipped,
            "error_labels": ERROR_LABELS,
        }
