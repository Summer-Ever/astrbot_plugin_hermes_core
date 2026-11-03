"""astrbot_plugin_hermes_core

复刻 NousResearch/hermes-agent 学习闭环的 AstrBot 插件：
- 记忆注入：每次 LLM 请求前检索相关长期记忆与历史对话（on_llm_request）
- 会话落库：SQLite + WAL + FTS5，全文可检索（on_llm_response）
- 自动总结：后台循环把新对话压缩为记忆条目（Hermes 的 nudge 机制）
- 技能进化：总结时提炼可复用 Markdown 技能；模型也可主动存技能（llm_tool）
- 工具面：模型可调用 search_memory / save_memory / list_skills / read_skill / save_skill
"""

import asyncio
import hashlib
import json
import os
import re
import time
import traceback

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register

try:
    from astrbot.api.provider import ProviderRequest, LLMResponse
except ImportError:  # 兼容旧版本导入路径
    from astrbot.core.provider.entities import ProviderRequest, LLMResponse

try:
    from astrbot.api.star import StarTools

    def _default_data_dir() -> str:
        return StarTools.get_data_dir("astrbot_plugin_hermes_core")
except ImportError:
    def _default_data_dir() -> str:
        return os.path.join("data", "plugin_data", "astrbot_plugin_hermes_core")

# 动态上下文注入优先走 extra_user_content_parts（保护 prompt 缓存），旧版本回退 system_prompt
try:
    from astrbot.core.agent.message import TextPart
except ImportError:
    TextPart = None

try:
    from astrbot.api.event import MessageChain
    from astrbot.api.message_components import Plain
except ImportError:
    MessageChain = None
    Plain = None

# db 模块位于 ./modules/db.py。AstrBot 以 data.plugins.<name>.main 的形式加载插件，
# 正常情况下相对导入即可；不同版本/安装方式的加载路径有差异，故逐级兜底。
try:
    from .modules.db import HermesDB
except ImportError:
    try:
        from modules.db import HermesDB  # 插件目录本身在 sys.path 上时
    except ImportError:  # 最后兜底：按文件路径直接加载，不污染 sys.path
        import importlib.util as _ilu

        _db_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "modules", "db.py"
        )
        _db_spec = _ilu.spec_from_file_location("astrbot_plugin_hermes_core_db", _db_file)
        _db_mod = _ilu.module_from_spec(_db_spec)
        _db_spec.loader.exec_module(_db_mod)
        HermesDB = _db_mod.HermesDB

# 渲染模块与 db 同层，复用同样的三级兜底
try:
    from .modules.render import RenderError, render_mermaid
except ImportError:
    try:
        from modules.render import RenderError, render_mermaid
    except ImportError:
        import importlib.util as _ilu2

        _rd_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "modules", "render.py"
        )
        _rd_spec = _ilu2.spec_from_file_location("astrbot_plugin_hermes_core_render", _rd_file)
        _rd_mod = _ilu2.module_from_spec(_rd_spec)
        _rd_spec.loader.exec_module(_rd_mod)
        RenderError = _rd_mod.RenderError
        render_mermaid = _rd_mod.render_mermaid

# 模型路由：主模型 + 降级链 + 重试/错误分类，同样三级兜底
try:
    from .modules.router import ModelRouter, ModelError, ERROR_LABELS
except ImportError:
    try:
        from modules.router import ModelRouter, ModelError, ERROR_LABELS
    except ImportError:
        import importlib.util as _ilu3

        _rt_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "modules", "router.py"
        )
        _rt_spec = _ilu3.spec_from_file_location("astrbot_plugin_hermes_core_router", _rt_file)
        _rt_mod = _ilu3.module_from_spec(_rt_spec)
        _rt_spec.loader.exec_module(_rt_mod)
        ModelRouter = _rt_mod.ModelRouter
        ModelError = _rt_mod.ModelError
        ERROR_LABELS = _rt_mod.ERROR_LABELS

try:
    from astrbot.api.message_components import Image
except ImportError:
    Image = None

# 技能中枢：把本体装的只读技能和插件进化的个人技能合成一个视图，同样三级兜底
try:
    from .modules.skill_hub import SkillHub, guess_data_root
except ImportError:
    try:
        from modules.skill_hub import SkillHub, guess_data_root
    except ImportError:
        import importlib.util as _ilu4

        _sh_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "modules", "skill_hub.py"
        )
        _sh_spec = _ilu4.spec_from_file_location("astrbot_plugin_hermes_core_skillhub", _sh_file)
        _sh_mod = _ilu4.module_from_spec(_sh_spec)
        _sh_spec.loader.exec_module(_sh_mod)
        SkillHub = _sh_mod.SkillHub
        guess_data_root = _sh_mod.guess_data_root

PLUGIN_VERSION = "0.9.0"

SYSTEM_MEMORY_HINT = (
    "你拥有持久记忆系统。可用工具：search_memory(检索记忆/历史)、save_memory(保存重要事实)、"
    "list_skills(查看技能，分「我的技能」和「系统技能·只读」)、read_skill(读取技能全文)、"
    "save_skill(把可复用流程存为我的技能)、update_skill(局部修改我的技能)、"
    "render_mermaid(把 mermaid 图表渲染成 PNG 图片发到聊天)。"
    "遇到值得长期记住的信息请主动 save_memory；完成复杂流程后可用 save_skill 沉淀方法。"
    "改技能一律走 update_skill 做局部替换，不要整份重写；"
    "系统技能是 AstrBot 本体装的、全局共享，插件不能写它，要改就基于它生成个人副本。"
)


# ============ 记忆巩固：相似度与聚类（纯本地，不花 token）============


def _norm_text(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", (s or "").lower())


def _bigrams(s: str) -> set[str]:
    s = _norm_text(s)
    if not s:
        return set()
    if len(s) <= 2:
        return {s}
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _similarity(a: str, b: str) -> float:
    """字符二元组 Jaccard。中文没有空格，按词切分不可靠，按字更稳。

    "喜欢吃香蕉" vs "喜欢吃苹果" ≈ 0.5（该合）；"喜欢看动漫" vs "喜欢吃香蕉" ≈ 0.33（不该合）。
    """
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return inter / (len(A) + len(B) - inter)


def _cluster_by_similarity(items: list[dict], threshold: float,
                           text_key: str = "content") -> list[list[dict]]:
    """把相似的条目聚成簇（并查集）。O(n²)，记忆量在千条以内可接受。"""
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _similarity(str(items[i].get(text_key, "")),
                           str(items[j].get(text_key, ""))) >= threshold:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    return [g for g in groups.values() if len(g) > 1]


def _scope_bucket(m: dict) -> tuple:
    """巩固的作用域桶：绝不把不同用户、不同群的记忆合到一起。"""
    owner = (m.get("owner_id") or "").strip()
    gid = (m.get("group_id") or "").strip()
    if owner:
        return ("user", owner, m.get("platform") or "")
    if gid:
        return ("group", gid, m.get("platform") or "")
    return ("global",)


@register("astrbot_plugin_hermes_core", "workbuddy", "Hermes 风格记忆与技能学习闭环", PLUGIN_VERSION)
class HermesCore(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config
        self.db = HermesDB(_default_data_dir())
        # 技能分两个域：本体的 skills/ 只读，插件自己的按人隔离可写，见 modules/skill_hub.py
        self.hub = SkillHub(
            self.db, os.path.join(guess_data_root(_default_data_dir()), "skills")
        )
        # 所有总结/整理都走它：主模型失败自动降级，不依赖「当前对话模型」
        self.router = ModelRouter(self)
        self._bg_task: asyncio.Task | None = None
        self._last_origin: str | None = None  # 最近活跃会话，用于技能生成通知

    async def initialize(self):
        self._write_build_stamp()
        logger.info(f"[hermes_core] 已加载，FTS5: {'开启' if self.db.fts_enabled else '不可用(退化为 LIKE)'}")
        self._migrate_watermarks()
        self._check_memory_conflicts()
        self._register_web_apis()
        if self._get_cfg("auto_summarize", True):
            self._bg_task = asyncio.create_task(self._bg_loop())

    def _register_web_apis(self):
        """把 WebUI 接口挂到 dashboard 上（页面见 views/hermes/）。

        导入方式沿用 db 的三级兜底：真实部署里插件可能不是以包的形式加载，
        `from .webapi import ...` 会失败，需要退化到同名模块/文件加载。
        """
        try:
            try:
                from .webapi import HermesWebAPI
            except ImportError:
                try:
                    from webapi import HermesWebAPI
                except ImportError:
                    import importlib.util
                    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapi.py")
                    _spec = importlib.util.spec_from_file_location("hermes_webapi", _p)
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    HermesWebAPI = _mod.HermesWebAPI
            self._web_api = HermesWebAPI(self, PLUGIN_VERSION, _default_data_dir)
            self._web_api.register()
            logger.info("[hermes_core] WebUI 接口已注册：/views/hermes/")
        except Exception:
            logger.error(f"[hermes_core] 注册 WebUI 接口失败:\n{traceback.format_exc()}")


    def _migrate_watermarks(self):
        """v0.4 起总结水位按会话独立。老库只有全局 last_summarized_id，
        升级时把它一次性写入各会话，避免重复总结已处理过的消息。"""
        try:
            if self.db.get_meta("watermark_migrated", "0") == "1":
                return
            legacy = int(self.db.get_meta("last_summarized_id", "0") or 0)
            if legacy > 0:
                for sid in self.db.all_session_ids():
                    if self.db.get_watermark(sid) == 0:
                        self.db.set_watermark(sid, legacy)
                logger.info(f"[hermes_core] 已将全局水位 {legacy} 迁移到按会话水位")
            self.db.set_meta("watermark_migrated", "1")
        except Exception:
            logger.error(f"[hermes_core] 水位迁移失败:\n{traceback.format_exc()}")

    def _get_cfg(self, key: str, default):
        if self.config is None:
            return default
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    # ============ 构建戳：确认热重载后跑的是哪份代码 ============

    def _build_hash(self) -> str:
        """对插件目录下所有 .py 内容取短哈希，用于比对磁盘代码与运行中代码。"""
        h = hashlib.md5()
        base = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "renderer")]
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                h.update(os.path.relpath(path, base).replace("\\", "/").encode())
                try:
                    with open(path, "rb") as fh:
                        # 统一换行符：Windows 上不同工具写出的 CRLF/LF 混合会让
                        # 同一份代码算出不同哈希，"是否生效"的校验就永远失败。
                        h.update(fh.read().replace(b"\r\n", b"\n"))
                except OSError:
                    pass
        return h.hexdigest()[:10]

    def _write_build_stamp(self) -> None:
        try:
            info = {
                "version": PLUGIN_VERSION,
                "code_hash": self._build_hash(),
                "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": os.getpid(),
            }
            path = os.path.join(_default_data_dir(), "build_stamp.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            logger.info(f"[hermes_core] 已加载 v{PLUGIN_VERSION} code={info['code_hash']}")
        except Exception:
            logger.warning(f"[hermes_core] 写入构建戳失败:\n{traceback.format_exc()}")

    # ============ 钩子：请求前注入记忆 ============

    def _scope_keys(self, event: AstrMessageEvent) -> dict:
        """解析当前消息的作用域键。

        注意：QQ 官方适配器中 sender_id 是 member_openid(群)/user_openid(私聊)，
        并非 QQ 号；且 member_openid 按群维度不同、与 user_openid 不同命名空间。
        因此 owner_id 会先过别名表，把同一人的多个 openid 归一。
        """
        raw_id = str(event.get_sender_id() or "")
        platform = event.get_platform_name() or ""
        return {
            "raw_id": raw_id,
            "owner_id": self.db.resolve_owner(raw_id, platform),
            "sid": event.unified_msg_origin,
            "platform": platform,
            "group_id": str(event.get_group_id() or ""),
        }

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            if not self._get_cfg("memory_injection", True):
                return
            query = (event.message_str or "").strip()
            if not query:
                return
            block = self._build_context_block(event, query)
            if not block:
                return
            if TextPart is not None and hasattr(req, "extra_user_content_parts"):
                part = TextPart(text=block)
                if hasattr(part, "mark_as_temp"):
                    part.mark_as_temp()
                req.extra_user_content_parts.append(part)
            else:
                req.system_prompt = (req.system_prompt or "") + "\n" + block
        except Exception:
            logger.error(f"[hermes_core] 注入失败:\n{traceback.format_exc()}")

    def _build_context_block(self, event: AstrMessageEvent, query: str) -> str:
        lines: list[str] = []
        sc = self._scope_keys(event)
        owner_id, sid = sc["owner_id"], sc["sid"]
        mem_k = int(self._get_cfg("inject_memories_topk", 5))
        hist_k = int(self._get_cfg("inject_history_topk", 3))
        scope = str(self._get_cfg("memory_scope", "user"))
        per_user_hist = bool(self._get_cfg("per_user_history", True))

        # 记忆：user 模式按 用户>群>全局 三级召回；group 模式按 群>全局；global 模式全量
        if scope == "user":
            memories = self.db.search_memories(
                query, limit=mem_k, owner_id=owner_id,
                session_id=sid, platform=sc["platform"], group_id=sc["group_id"] or None)
        elif scope == "group":
            memories = self.db.search_memories(
                query, limit=mem_k, session_id=sid,
                platform=sc["platform"], group_id=sc["group_id"] or None)
        else:
            memories = self.db.search_memories(query, limit=mem_k)
        if memories:
            lines.append("[长期记忆参考]")
            lines += [f"- ({m['kind']}) {m['content']}" for m in memories]

        # 历史：默认只看本人在本群的历史，避免混入其他群友的对话
        sender_filter = sc["raw_id"] if (scope == "user" and per_user_hist) else None
        history = self.db.search_messages(query, limit=hist_k, session_id=sid, sender_id=sender_filter)
        if history:
            lines.append("[相关历史对话]")
            for h in reversed(history):
                who = "你" if h["role"] == "user" else "你之前回复"
                text = h["content"][:200]
                lines.append(f"- {who}: {text}")

        skills = self.hub.list_for(owner_id, sc["group_id"])
        if skills:
            personal = [s for s in skills if s.get("source") == "personal"]
            system = [s for s in skills if s.get("source") == "system"]
            parts: list[str] = []
            if personal:
                parts.append("我的技能：" + "、".join(s["name"] for s in personal[:15]))
            if system:
                parts.append("系统技能(本体装的,只读)：" + "、".join(s["name"] for s in system[:10]))
            lines.append(f"[已掌握技能] {'；'.join(parts)}（可用 read_skill 查看全文）")

        if not lines:
            return ""
        return "\n".join(["[记忆系统上下文，仅作参考]"] + lines + ["[/记忆系统上下文]"])

    # ============ 钩子：响应后落库 ============

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            self._last_origin = event.unified_msg_origin
            sid = event.unified_msg_origin
            user_text = (event.message_str or "").strip()
            if user_text:
                self.db.add_message(
                    sid, "user", user_text,
                    platform=event.get_platform_name() or "",
                    sender_id=str(event.get_sender_id() or ""),
                    sender_name=event.get_sender_name() or "",
                    group_id=str(event.get_group_id() or ""),
                )
            reply = ""
            try:
                reply = (resp.completion_text or "").strip()
            except Exception:
                pass
            if reply:
                self.db.add_message(sid, "assistant", reply,
                                    platform=event.get_platform_name() or "",
                                    group_id=str(event.get_group_id() or ""))
        except Exception:
            logger.error(f"[hermes_core] 落库失败:\n{traceback.format_exc()}")

    # ============ LLM 工具：模型主动使用记忆/技能 ============

    @filter.llm_tool(name="search_memory")
    async def tool_search_memory(self, event: AstrMessageEvent, query: str) -> str:
        """按关键词检索长期记忆与历史对话。当需要回忆当前用户偏好、之前聊过的内容时调用。
        Args:
            query (string): 检索关键词
        """
        sc = self._scope_keys(event)
        mems = self.db.search_memories(
            query, limit=5, owner_id=sc["owner_id"], session_id=sc["sid"],
            platform=sc["platform"], group_id=sc["group_id"] or None)
        msgs = self.db.search_messages(query, limit=5, session_id=sc["sid"], sender_id=sc["raw_id"])
        out = []
        if mems:
            out.append("[记忆] " + " | ".join(f"{m['content']}" for m in mems))
        if msgs:
            out.append("[历史] " + " | ".join(f"{m['role']}:{m['content'][:80]}" for m in reversed(msgs)))
        return "\n".join(out) or "没有找到相关内容。"

    @filter.llm_tool(name="save_memory")
    async def tool_save_memory(self, event: AstrMessageEvent, content: str,
                               kind: str = "fact") -> str:
        """把值得长期记住的信息保存为记忆（自动归属到当前发言用户）。用户偏好、重要事实、项目背景等都应保存。
        Args:
            content (string): 记忆内容（简洁的一句话）
            kind (string): 类型，fact/preference/profile/summary 之一
        """
        sc = self._scope_keys(event)
        mid = self.db.add_memory(kind or "fact", content, scope="user",
                                 source="llm_tool", owner_id=sc["owner_id"],
                                 session_id=sc["sid"], platform=sc["platform"],
                                 group_id=sc["group_id"])
        self._queue_memories([mid])  # 手动存的也进整理队列，攒够阈值一起合并
        return f"已保存记忆 #{mid}。"

    @filter.llm_tool(name="list_skills")
    async def tool_list_skills(self, event: AstrMessageEvent) -> str:
        """列出技能，分「我的技能」（属于当前用户，可改）和「系统技能·只读」两组。
        用户提到要修改/查看某个技能前，先用它确认这个技能在哪一组。
        """
        sc = self._scope_keys(event)
        return self.hub.describe(self.hub.list_for(sc["owner_id"], sc["group_id"]))

    @filter.llm_tool(name="read_skill")
    async def tool_read_skill(self, event: AstrMessageEvent, name: str) -> str:
        """读取指定技能的完整 Markdown 内容。执行该技能覆盖的任务前应先调用。
        Args:
            name (string): 技能名称
        """
        sc = self._scope_keys(event)
        sk = self.hub.read(name, sc["owner_id"], sc["group_id"])
        if not sk:
            return f"技能 {name} 不存在。可用 list_skills 查看现有技能。"
        if sk.get("source") == "system":
            return (
                f"[系统技能·只读] {sk['name']}——AstrBot 本体安装的，全局共享"
                f"{'、含可执行脚本' if sk.get('has_scripts') else ''}，不能直接改。\n"
                f"要定制就用 update_skill 局部改，或 save_skill 同名保存；"
                f"插件会自动生成属于当前用户的个人副本，系统原版不受影响。\n\n"
                f"{sk['content']}"
            )
        head = f"[我的技能] {sk['name']}——属于当前用户，可直接用 update_skill 局部修改"
        if sk.get("shadow_of"):
            head += (f"；这是覆盖系统技能 {sk['shadow_of']} 的个人版本"
                     f"（本体原版 {sk.get('system_bytes', 0)} 字符，如需参考完整版可 read_skill "
                     f"\"system:{sk['shadow_of']}\"）")
        return f"{head}\n\n{sk['content']}"

    @filter.llm_tool(name="save_skill")
    async def tool_save_skill(self, event: AstrMessageEvent, name: str,
                              trigger: str, content: str,
                              scope: str = "private") -> str:
        """把一个可复用的解决流程保存为「我的技能」（Markdown 步骤），同名则整份覆盖。
        只改技能里某几步时不要用它（整份覆盖会丢内容），改用 update_skill。
        Args:
            name (string): 英文短名，如 deploy_blog
            trigger (string): 什么场景下应该使用该技能
            content (string): Markdown 格式的完整步骤
            scope (string): private 只有自己可见（默认）；group 本群所有人可见
        """
        sc = self._scope_keys(event)
        name = re.sub(r"\s+", "_", name.strip())[:64]
        if not name or not content:
            return "技能名和内容都不能为空。"
        want = str(scope or "private").strip().lower()
        scp = "group" if want == "group" and sc["group_id"] else "private"
        res = self.hub.save(name, trigger or "", content, owner_id=sc["owner_id"],
                            group_id=sc["group_id"], scope=scp)
        logger.info(f"[hermes_core] 技能 {'新建' if res['created'] else '更新'}: {res['name']}")
        tail = ""
        if res.get("shadow_of"):
            tail = f"（已覆盖系统技能 {res['shadow_of']}，本体原版未改动）"
        where = "本群共享" if scp == "group" else "仅当前用户可见"
        return f"技能 {res['name']} 已{'保存' if res['created'] else '更新'}，{where}{tail}。"

    @filter.llm_tool(name="update_skill")
    async def tool_update_skill(self, event: AstrMessageEvent, name: str,
                                old_text: str, new_text: str) -> str:
        """局部修改技能：把技能正文里的 old_text 换成 new_text，其余内容原样保留。
        用户让你改某个技能的某几步、某个命令、某段说明时用这个，不要整份重写。
        Args:
            name (string): 技能名称
            old_text (string): 要替换的原文，必须从 read_skill 结果里精确复制，且在正文中只出现一次；留空表示把 new_text 追加到技能末尾
            new_text (string): 替换成什么
        """
        sc = self._scope_keys(event)
        res = self.hub.update(name, old_text, new_text, owner_id=sc["owner_id"],
                              group_id=sc["group_id"])
        if not res.get("ok"):
            return f"修改失败：{res.get('error', '未知错误')}"
        if res.get("shadowed"):
            return (f"已修改：系统技能 {res['name']} 不能直接改，我基于它建了一份属于你的副本再改，"
                    f"以后你会优先读到这份，本体原版没动。")
        return f"技能 {res['name']} 已按你的要求局部更新，其余内容保持不变。"

    @filter.llm_tool(name="render_mermaid")
    async def tool_render_mermaid(self, event: AstrMessageEvent, code: str) -> str:
        """把 mermaid 图表源码本地渲染为 PNG 并直接发送到当前聊天。当用户想要图、流程图、时序图、架构图等可视化内容时调用；不要使用 mermaid.ink 等外部渲染服务。
        Args:
            code (string): 完整的 mermaid 图表源码，第一行是图表类型声明，例如 "flowchart TD\\nA[开始] --> B[结束]"
        """
        if Image is None:
            return "当前 AstrBot 版本缺少图片组件，无法发图。"
        out = os.path.join(
            _default_data_dir(), "renders",
            f"m_{hashlib.md5(code.encode()).hexdigest()[:12]}.png",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            await render_mermaid(code, out, _default_data_dir())
        except RenderError as e:
            return f"渲染失败：{e}"
        except Exception:
            return f"渲染异常：{traceback.format_exc(limit=2)}"
        if MessageChain is not None:
            try:
                await self.context.send_message(
                    event.unified_msg_origin, MessageChain([Image.fromFileSystem(out)])
                )
                return "图片已发送。"
            except Exception:
                return f"渲染成功但自动发送失败，请改用 /hermes 图 命令。文件位于：{out}"
        return f"渲染成功，文件位于：{out}"

    # ============ 后台总结循环（Hermes nudge）============

    async def _bg_loop(self):
        interval = max(5, int(self._get_cfg("summarize_interval_minutes", 30)))
        while True:
            await asyncio.sleep(interval * 60)
            try:
                await self._summarize()
            except Exception:
                logger.error(f"[hermes_core] 总结循环异常:\n{traceback.format_exc()}")

    async def _summarize(self, force: bool = False) -> str:
        """按会话独立判断是否值得总结：各自攒够条数、或超时兜底、或手动强制。

        每个会话有自己的水位（meta 键 summarized:<sid>），互不干扰，避免
        "全局凑够 8 条"导致的低频会话永远不总结、以及跨会话打包串味。
        """
        if not self.router.has_model():
            return ("没有可用的对话模型：请先在 AstrBot 里配置模型提供商，"
                    "再到插件页面「模型」标签选择总结模型。")

        min_new = max(1, int(self._get_cfg("min_new_messages", 8)))
        grace_h = float(self._get_cfg("summarize_grace_hours", 6))
        now = time.time()

        pending = self.db.unsummarized_by_session()
        due: list[tuple[dict, str]] = []
        for st in pending:
            if force:
                due.append((st, "手动触发"))
            elif st["count"] >= min_new:
                due.append((st, f"{st['count']} 条达标"))
            elif grace_h > 0 and st["oldest"] and (now - st["oldest"]) >= grace_h * 3600:
                due.append((st, f"超 {grace_h:g} 小时兜底"))

        if not due:
            total = sum(st["count"] for st in pending)
            return (f"暂无需总结：{len(pending)} 个会话共 {total} 条新消息，"
                    f"均未达 {min_new} 条且未超 {grace_h:g} 小时兜底。")

        made_mem = made_skill = skill_upd = done_msgs = 0
        details: list[str] = []
        before_max = self.db.max_memory_id()
        for st, reason in due:
            batch = self.db.session_batch(st["session_id"], st["watermark"], limit=200)
            if not batch:
                continue
            try:
                mem, sk, sk_up = await self._summarize_session(batch, st["session_id"])
            except Exception as exc:
                # 单个会话失败不拖垮其他会话；不推进水位，下一轮会重试这批消息
                logger.error(f"[hermes_core] 会话总结失败 {st['session_id'][:36]}:\n"
                             f"{traceback.format_exc(limit=3)}")
                details.append(f"{st['session_id'][:20]}：失败（{str(exc)[:80]}）")
                continue
            made_mem += mem
            made_skill += sk
            skill_upd += sk_up
            done_msgs += len(batch)
            # 只推进到本批最后一条：积压超 200 条时下一轮继续处理
            self.db.set_watermark(st["session_id"], batch[-1]["id"])
            details.append(f"{st['session_id'][:36]}：{len(batch)} 条（{reason}）")

        summary = (f"[hermes_core] 总结完成：{len(details)} 个会话 / {done_msgs} 条消息，"
                   f"新记忆 {made_mem} 条，技能 新增 {made_skill} / 更新 {skill_upd} 个。")

        # 记忆巩固：攒够 N 条新记忆才整理一次（一次整理 = 一次模型调用，别每条都调）
        new_ids = {m["id"] for m in self.db.all_active_memories() if m["id"] > before_max}
        queued = self._queue_memories(new_ids) if new_ids else self._queued_memory_count()
        if self._should_consolidate_memories(queued):
            try:
                note = await self._consolidate_memories(focus_ids=set(self._pending_memory_ids()))
                if note and not note.startswith("记忆只有"):
                    summary += "\n" + note
                self._reset_memory_queue()
            except Exception:
                # 出错不清队列，下一轮重试
                logger.error(f"[hermes_core] 记忆巩固失败:\n{traceback.format_exc()}")

        # 技能合并：同理，攒够 N 个新增/更新技能才跑一次（技能整理是全量发给模型，更贵）
        if made_skill or skill_upd:
            pend_skill = self._bump_skill_queue(made_skill + skill_upd)
        else:
            pend_skill = int(self.db.get_meta("skill_consolidate_pending", "0") or 0)
        if self._should_consolidate_skills(pend_skill):
            try:
                note = await self._consolidate_skills()
                summary += "\n" + note
                self.db.set_meta("skill_consolidate_pending", "0")
            except Exception:
                logger.error(f"[hermes_core] 技能整理失败:\n{traceback.format_exc()}")

        logger.info(summary + " | " + "；".join(details))
        if made_skill and self._get_cfg("skill_notify_admin", True) and self._last_origin \
                and MessageChain is not None and Plain is not None:
            try:
                await self.context.send_message(
                    self._last_origin,
                    MessageChain([Plain(f"[技能进化] 刚学会了 {made_skill} 个新技能，用 /hermes 技能 查看。")]),
                )
            except Exception:
                pass
        return summary

    async def _summarize_session(self, batch: list[dict],
                                 sid: str) -> tuple[int, int, int]:
        """把一个会话的一批消息交给 LLM 归纳，写入记忆与技能。

        返回 (新记忆数, 新技能数, 更新技能数)。
        prompt 按目标模型的上下文动态裁剪：小上下文模型当降级模型时也能用。
        """
        fmt = batch[0].get("platform") or ""
        gid = batch[0].get("group_id") or ""
        known_ids: set[str] = set()
        lines: list[str] = []
        for m in batch:
            if m["role"] == "user":
                known_ids.add(str(m["sender_id"] or ""))
                lines.append(f"[用户 {m['sender_id'] or '未知'}({m['sender_name'] or '匿名'})]: {m['content'][:300]}")
            else:
                lines.append(f"助手: {m['content'][:300]}")
        transcript = "\n".join(lines)

        # 技能归属：这一批对话里发言最多的人。技能跟人走，不跟群走——
        # 否则 A 摸索出来的流程会被 B 覆盖掉。
        skill_owner = ""
        try:
            counts: dict[str, int] = {}
            for m in batch:
                if m["role"] == "user" and m.get("sender_id"):
                    sid_key = str(m["sender_id"])
                    counts[sid_key] = counts.get(sid_key, 0) + 1
            if counts:
                top = max(counts.items(), key=lambda kv: kv[1])[0]
                skill_owner = self.db.resolve_owner(top, fmt)
        except Exception:
            skill_owner = ""

        # 把已有技能清单给模型看：否则同一类任务每次都会造一个新名字，
        # 技能库只会越堆越多，用的时候还得一个个读。只看这个人自己的技能。
        existing_skills = self.db.list_skills(owner_id=skill_owner, group_id=gid)
        skill_hint = ""
        if existing_skills and self._get_cfg("auto_skill", True):
            lines_s = "\n".join(
                f"- {s['name']}：{s['trigger'] or '通用'}" for s in existing_skills[:25]
            )
            skill_hint = (
                "\n\n[已有技能] 若本次流程属于下列某个技能的范畴，name 必须原样沿用该技能名"
                "（不要造近义新名），content 写该技能合并后的完整版本：\n" + lines_s
            )

        head = (self._get_cfg("summary_prompt", "")
                + "\n\n对话中每位用户都带有 [用户 ID(昵称)] 标记。"
                  "若某条记忆只属于某个特定用户，请在该项追加 \"owner\": \"该用户的ID\"；"
                  "属于整段对话/全体时省略 owner。"
                + skill_hint
                + "\n\n对话记录：\n")

        def build(budget: int) -> str:
            room = max(600, budget - len(head) - 60)
            body = transcript
            if len(body) > room:
                body = body[:room]
                cut = body.rfind("\n")
                if cut > room * 0.6:
                    body = body[:cut]
                body += "\n…（对话过长，已按当前模型上下文截断，保留最早部分）"
            return head + body

        text = await self.router.chat(builder=build, purpose="总结记忆")
        made_mem = made_skill = skill_upd = 0
        for item in self._extract_json_array(text):
            if not isinstance(item, dict):
                continue
            if item.get("skill") and self._get_cfg("auto_skill", True):
                name = re.sub(r"\s+", "_", str(item.get("name", "")).strip())[:64]
                if name and item.get("content"):
                    trigger = str(item.get("trigger", "")).strip()
                    new_content = str(item["content"])
                    old = self.db.peek_skill(name, owner_id=skill_owner, group_id=gid)
                    if old and bool(self._get_cfg("skill_merge_same_name", True)):
                        # 同名技能：先读旧版再合并，而不是直接覆盖丢掉旧步骤
                        merged = await self._merge_skill_content(
                            name, old.get("trigger", ""), old.get("content", ""),
                            trigger, new_content,
                        )
                        self.db.upsert_skill(name, trigger or old.get("trigger", ""), merged,
                                             owner_id=skill_owner, group_id=gid)
                        skill_upd += 1
                    else:
                        self.db.upsert_skill(name, trigger, new_content,
                                             owner_id=skill_owner, group_id=gid)
                        made_skill += 1
            elif item.get("content"):
                owner = str(item.get("owner", "")).strip()
                if owner not in known_ids:
                    owner = ""  # 归属不明 → 降级为群公共记忆
                if owner:
                    owner = self.db.resolve_owner(owner, fmt)
                self.db.add_memory(
                    str(item.get("kind", "fact")), str(item["content"]),
                    scope="user" if owner else "group", source="auto_summary",
                    owner_id=owner, session_id=sid, platform=fmt, group_id=gid,
                )
                made_mem += 1
        return made_mem, made_skill, skill_upd

    # ============ 整理队列：攒够 N 条才合并 ============
    #
    # 合并是「一次模型调用」，逐条触发会变成每轮总结都多花一次钱。
    # 所以新记忆不立即合并，先排进队列，攒到阈值再一次性处理：
    #   meta.consolidate_pending_memories = 新记忆 id 列表（JSON，上限 200）
    #   meta.skill_consolidate_pending   = 新增/更新技能的累计计数

    MAX_QUEUE = 200

    def _pending_memory_ids(self) -> list[int]:
        raw = self.db.get_meta("consolidate_pending_memories", "")
        if not raw:
            return []
        try:
            return [int(i) for i in json.loads(raw)]
        except Exception:
            return []

    def _queued_memory_count(self) -> int:
        return len(self._pending_memory_ids())

    def _queue_memories(self, ids) -> int:
        """把新记忆加入待整理队列，返回队列长度。"""
        queue = self._pending_memory_ids()
        seen = set(queue)
        for i in ids:
            i = int(i)
            if i not in seen:
                queue.append(i)
                seen.add(i)
        queue = queue[-self.MAX_QUEUE:]
        self.db.set_meta("consolidate_pending_memories", json.dumps(queue))
        return len(queue)

    def _reset_memory_queue(self):
        self.db.set_meta("consolidate_pending_memories", "[]")

    def _bump_skill_queue(self, n: int) -> int:
        cur = int(self.db.get_meta("skill_consolidate_pending", "0") or 0) + max(0, n)
        self.db.set_meta("skill_consolidate_pending", str(cur))
        return cur

    def _mem_every(self) -> int:
        return max(0, int(self._get_cfg("consolidate_every_n_memories", 10)))

    def _skill_every(self) -> int:
        return max(0, int(self._get_cfg("skill_consolidate_every_n", 5)))

    def _should_consolidate_memories(self, queued: int) -> bool:
        every = self._mem_every()
        if every <= 0 or not bool(self._get_cfg("auto_consolidate", True)):
            return False
        return queued >= every

    def _should_consolidate_skills(self, queued: int) -> bool:
        every = self._skill_every()
        if every <= 0:
            return False
        return queued >= every and self.db.stats()["skills"] >= 2

    def _queue_progress(self) -> str:
        """给 /hermes 状态 用：离下次自动整理还差多少。"""
        mem_every, sk_every = self._mem_every(), self._skill_every()
        mem_on = mem_every > 0 and bool(self._get_cfg("auto_consolidate", True))
        mem_txt = f"记忆 {self._queued_memory_count()}/{mem_every} 条" if mem_on else "记忆 关闭"
        sk_n = int(self.db.get_meta("skill_consolidate_pending", "0") or 0)
        sk_txt = f"技能 {sk_n}/{sk_every} 个" if sk_every > 0 else "技能 关闭"
        return f"{mem_txt}、{sk_txt} 触发一次（每次约 1 次模型调用）"

    # ============ 记忆巩固（consolidation）============

    async def _consolidate_memories(self, dry_run: bool = False,
                                    focus_ids: set[int] | None = None) -> str:
        """把碎片化的记忆合并成更完整的条目。

        两种模式：
        - focus_ids：只给「刚写入的新记忆」找相似旧记忆（自动模式，省 token）
        - 不传：全库按相似度聚类（/hermes 整理 手动模式）

        字面相似度只用来「召回候选」，不直接决定合并——「喜欢吃香蕉 / 喜欢吃苹果」
        只有 0.33，而「养了只猫叫咪咪 / 养了只狗叫旺财」有 0.38，字面分不出该不该合。
        真正判断交给一次 LLM 调用。
        """
        threshold = float(self._get_cfg("consolidate_threshold", 0.3))
        max_clusters = max(1, int(self._get_cfg("consolidate_max_clusters", 6)))
        min_total = max(2, int(self._get_cfg("consolidate_min_memories", 6)))

        rows = self.db.all_active_memories()
        if not rows:
            return "还没有任何记忆。"

        if focus_ids:
            clusters: list[list[dict]] = []
            by_bucket: dict[tuple, list[dict]] = {}
            for m in rows:
                by_bucket.setdefault(_scope_bucket(m), []).append(m)
            for m in rows:
                if m["id"] not in focus_ids:
                    continue
                pool = [r for r in by_bucket[_scope_bucket(m)] if r["id"] != m["id"]]
                if not pool:
                    continue
                scored = sorted(((_similarity(m["content"], r["content"]), r) for r in pool),
                                key=lambda x: -x[0])
                picked = [r for s, r in scored[:3] if s >= threshold]
                if picked:
                    # 组内按 id 排序，新记忆排最后（LLM 能看出时间顺序）
                    clusters.append(sorted(picked + [m], key=lambda r: r["id"]))
        else:
            if len(rows) < min_total:
                return f"记忆只有 {len(rows)} 条，暂无整理必要。"
            buckets: dict[tuple, list[dict]] = {}
            for m in rows:
                buckets.setdefault(_scope_bucket(m), []).append(m)
            clusters = []
            for items in buckets.values():
                if len(items) < 2:
                    continue
                clusters += _cluster_by_similarity(items, threshold)
            if not clusters:
                return f"已检查 {len(rows)} 条记忆，没有发现需要合并的碎片。"

        # 去掉被别的簇完全包含的重复簇
        seen: set[tuple] = set()
        uniq: list[list[dict]] = []
        for g in clusters:
            key = tuple(sorted(m["id"] for m in g))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(g)
        clusters = sorted(uniq, key=lambda g: -len(g))[:max_clusters]

        listing: list[str] = []
        id_to_group: dict[int, int] = {}
        for idx, group in enumerate(clusters, 1):
            listing.append(f"组 {idx}：")
            for m in group:
                id_to_group[m["id"]] = idx
                listing.append(f"- #{m['id']} ({m['kind']}) {m['content'][:200]}")

        head = (
            "下面是从长期记忆里挑出的若干组碎片，组内内容高度相似或互补。"
            "请把每组整理成尽量少、互不重复、信息不丢失的条目。\n"
            "规则：\n"
            "1. 信息互补的合并成一条（例如「喜欢吃香蕉」和「喜欢吃苹果」→「喜欢吃香蕉、苹果」）；\n"
            "2. 后出现的覆盖先前的（先说喜欢猫、后来说不喜欢了），只保留最新状态；\n"
            "3. 确实无关的保持分开，不要硬凑；\n"
            "4. 不要编造原文没有的信息，不要丢失具体细节；\n"
            "5. ids 只能填同一组里的记忆编号，且至少 2 条。\n"
            "输出 JSON 数组，不要解释：[{\"ids\":[1,2],\"kind\":\"preference\","
            "\"content\":\"合并后的表述\"}]\n没有需要合并的就输出 []。\n\n"
        )
        body_text = "\n".join(listing)

        def build(budget: int) -> str:
            room = max(600, budget - len(head) - 40)
            body = body_text
            if len(body) > room:
                body = body[:room]
                cut = body.rfind("\n")
                if cut > room * 0.6:
                    body = body[:cut]
                body += "\n（其余组因当前模型上下文不足未列出，可换更大上下文的模型再整理）"
            return head + body

        text = await self.router.chat(builder=build, purpose="整理记忆")
        plan = self._extract_json_array(text)

        applied: list[str] = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            try:
                ids = [int(x) for x in (item.get("ids") or []) if str(x).strip().isdigit()]
            except Exception:
                continue
            content = str(item.get("content") or "").strip()
            if len(ids) < 2 or not content:
                continue
            # 必须是同一组，且 id 都真实存在——防止模型跨组乱填、合并到别人头上
            gids = {id_to_group.get(i) for i in ids}
            if len(gids) != 1 or None in gids:
                continue
            kinds = [self.db.memory_by_id(i) for i in ids]
            if any(k is None for k in kinds):
                continue
            if len({_scope_bucket(k) for k in kinds}) != 1:
                continue
            old_text = " / ".join(k["content"] for k in kinds if k)
            if dry_run:
                applied.append(f"#{'/'.join(map(str, ids))}：{old_text[:80]} → {content[:80]}")
                continue
            new_id = self.db.merge_memories(ids, content[:500], str(item.get("kind") or ""))
            if new_id:
                applied.append(f"#{'/'.join(map(str, ids))} → #{new_id}：{content[:80]}")

        if not applied:
            return (f"发现 {len(clusters)} 组疑似重复，但没有一组通过校验"
                    "（可能模型判断不该合，或 ids 跨了用户/群）。")

        head = "记忆整理预览（未写入）" if dry_run else "记忆已整理"
        return f"{head}：{len(applied)} 组\n" + "\n".join(f"- {a}" for a in applied[:8])

    async def _merge_skill_content(self, name: str, old_trigger: str,
                                   old_content: str, new_trigger: str,
                                   new_content: str) -> str:
        """技能同名的 read-modify-write：保留旧版仍有效的步骤，吸收新版内容。"""
        head = (
            "你是技能库管理员。下面是同一个技能的两个版本（旧版是之前总结的，新版是刚总结的）。"
            "请合并成一份完整、不重复、可直接照着执行的 Markdown 步骤文档："
            "保留旧版里仍然有效的步骤，吸收新版新增的内容，去掉重复描述。"
            "只输出合并后的 Markdown 正文，不要解释，不要加代码块围栏。\n\n"
            f"[技能名] {name}\n"
        )

        def build(budget: int) -> str:
            # 两个版本平分可用空间，避免新版太长把旧版挤掉
            room = max(800, budget - len(head) - 200)
            half = max(400, room // 2)
            old_part = old_content[:half]
            new_part = new_content[:min(half, max(400, room - len(old_part)))]
            return (head
                    + f"[旧版触发条件] {old_trigger}\n[旧版]\n{old_part}"
                    + f"\n\n[新版触发条件] {new_trigger}\n[新版]\n{new_part}")

        try:
            merged = (await self.router.chat(builder=build, purpose="合并技能")).strip()
        except Exception:
            logger.error(f"[hermes_core] 技能合并失败:\n{traceback.format_exc(limit=3)}")
            merged = ""
        if not merged or len(merged) < 20:
            # 合并失败就保守处理：保留旧版，把新版附在后面，绝不丢内容
            return old_content + "\n\n## 补充（新版）\n" + new_content
        merged = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", merged).strip()
        return merged[:8000]

    def _skill_groups(self) -> list[dict]:
        """按归属把技能分组。只有落在同一个归属域里的技能才可能被合并。"""
        buckets: dict[tuple[str, str, str], int] = {}
        for r in self.db.skill_owners():
            key = (str(r["owner_id"] or ""), str(r["scope"] or "private"),
                   str(r["group_id"] or ""))
            buckets[key] = buckets.get(key, 0) + int(r["c"] or 0)
        groups = [
            {"owner_id": k[0], "scope": k[1], "group_id": k[2], "count": n}
            for k, n in buckets.items() if n >= 2
        ]
        groups.sort(key=lambda g: -g["count"])
        return groups

    @staticmethod
    def _group_label(g: dict) -> str:
        if g["scope"] == "global":
            return "公共"
        if g["scope"] == "group":
            return f"群 {str(g['group_id'])[-6:]}"
        return f"用户 {str(g['owner_id'])[-6:]}"

    async def _consolidate_skills(self, dry_run: bool = False) -> str:
        """按归属分别整理技能，不同用户的技能绝不互相合并。

        技能不是记忆，字面相似度几乎没用（`chrome-cdp-browser-control` 和
        `chrome_real_profile_cdp` 只有 0.09，但其实是同一件事）。所以每个归属域内
        技能数不多时直接把全部技能交给模型判断，只有数量过多才用字面相似度粗筛。
        """
        groups = self._skill_groups()
        if not groups:
            return "每个人名下的技能都不足 2 个，无需整理。"

        # 一人一次模型调用，人多时成本线性上涨，所以每次只处理技能最多的几个域
        limit = max(1, int(self._get_cfg("skill_consolidate_owners", 3)))
        picked = groups[:limit]
        done_all: list[str] = []
        notes: list[str] = []
        for g in picked:
            try:
                note, done, checked = await self._consolidate_skill_group(g, dry_run)
            except ModelError as exc:
                notes.append(f"{self._group_label(g)}：整理失败（{exc.kind}），稍后重试")
                continue
            except Exception:
                logger.error(f"[hermes_core] 整理技能失败:\n{traceback.format_exc()}")
                continue
            if note:
                notes.append(note)
            done_all += [f"{self._group_label(g)} {'/'.join(d)}" for d in done]
            _ = checked

        if done_all:
            head = "技能整理预览（未写入）" if dry_run else "技能已合并"
            body = "\n".join(f"- {d}" for d in done_all[:8])
            return f"{head}：{len(done_all)} 组\n{body}" + (
                "\n" + "\n".join(notes) if notes else "")
        if notes:
            return "\n".join(notes)
        return f"检查了 {len(picked)} 个归属域的技能，模型判断不需要合并。"

    async def _consolidate_skill_group(self, group: dict, dry_run: bool) -> tuple[str, list[str], int]:
        """整理单个归属域内的技能，返回 (说明, 已合并列表, 检查数)。"""
        skills = self.db.skills_in(group["owner_id"], group["scope"], group["group_id"])
        if len(skills) < 2:
            return "", [], len(skills)

        max_direct = max(4, int(self._get_cfg("skill_consolidate_max", 40)))
        threshold = float(self._get_cfg("skill_merge_threshold", 0.2))

        if len(skills) <= max_direct:
            clusters = [skills]  # 全部丢给模型，靠语义判断
        else:
            items = [
                {
                    "name": s["name"],
                    "trigger": s["trigger"] or "",
                    "text": f"{s['name']} {s['trigger'] or ''}",
                }
                for s in skills
            ]
            cs = _cluster_by_similarity(items, threshold, text_key="text")
            if not cs:
                return f"{self._group_label(group)}：{len(skills)} 个技能字面没有明显重叠。", [], len(skills)
            name2skill = {s["name"]: s for s in skills}
            clusters = [[name2skill[i["name"]] for i in g] for g in cs]

        listing: list[str] = []
        for idx, cl in enumerate(clusters, 1):
            listing.append(f"组 {idx}：")
            for s in cl:
                listing.append(
                    f"- name: {s['name']} | trigger: {s['trigger'] or '通用'}\n"
                    f"  内容：{str(s.get('content') or '')[:600]}"
                )

        head = (
            "下面是同一个用户已经学会的技能，其中一部分是同一类任务被反复总结出来的，"
            "导致同一个能力有多个技能、用的时候得读好几遍。请把真正重叠的合并。\n"
            "规则：\n"
            "1. 只有确实做同一类事情才合并；不同图种（流程图 vs 时序图）、不同工具不要硬合；\n"
            "2. 合并时必须保留各版本所有不重复的步骤，不能丢细节；\n"
            "3. name 用英文短名（可沿用其中之一），trigger 写清楚何时使用；\n"
            "4. names 只能填下面出现过的技能名。\n"
            "输出 JSON 数组，不要解释：[{\"names\":[\"a\",\"b\"],\"name\":\"new_name\","
            "\"trigger\":\"何时使用\",\"content\":\"合并后的完整 Markdown\"}]\n"
            "没有需要合并的就输出 []。\n\n"
        )
        body_text = "\n".join(listing)

        def build(budget: int) -> str:
            room = max(600, budget - len(head) - 40)
            body = body_text
            if len(body) > room:
                # 技能整理要看到正文才有意义，截断会让判断失真，所以明说
                body = body[:room]
                cut = body.rfind("\n")
                if cut > room * 0.6:
                    body = body[:cut]
                body += "\n（其余技能因当前模型上下文不足未列出）"
            return head + body

        text = await self.router.chat(builder=build, purpose="整理技能")
        plan = self._extract_json_array(text)

        all_names = {s["name"] for s in skills}
        done: list[str] = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            names = [str(n) for n in (item.get("names") or []) if str(n) in all_names]
            new_name = re.sub(r"\s+", "_", str(item.get("name") or "").strip())[:64]
            content = str(item.get("content") or "").strip()
            if len(names) < 2 or not new_name or len(content) < 20:
                continue
            if dry_run:
                done.append(f"{'/'.join(names)} → {new_name}")
                continue
            self.db.replace_skills(
                names, new_name, str(item.get("trigger") or ""), content[:8000],
                owner_id=group["owner_id"], scope=group["scope"],
                group_id=group["group_id"],
            )
            done.append(f"{'/'.join(names)} → {new_name}")

        if not done:
            return f"{self._group_label(group)}：{len(skills)} 个技能，模型判断不需要合并。", [], len(skills)
        return "", done, len(skills)

    @staticmethod
    def _extract_json_array(text: str) -> list:
        if not text:
            return []
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    # ============ 命令 ============

    @filter.command_group("hermes")
    def hermes(self):
        """Hermes 记忆系统管理"""

    @hermes.command("状态")
    async def hermes_status(self, event: AstrMessageEvent):
        s = self.db.stats()
        interval = self._get_cfg("summarize_interval_minutes", 30)
        min_new = self._get_cfg("min_new_messages", 8)
        grace = self._get_cfg("summarize_grace_hours", 6)
        pending = self.db.unsummarized_by_session()
        pend_info = (f"{len(pending)} 个会话共 {sum(p['count'] for p in pending)} 条"
                     if pending else "无")
        yield event.plain_result(
            f"[Hermes 记忆系统]\n"
            f"版本: v{PLUGIN_VERSION} (code {self._build_hash()})\n"
            f"对话消息: {s['messages']} 条 / {s['sessions']} 个会话\n"
            f"长期记忆: {s['memories']} 条（已合并 {s.get('merged', 0)} 条）\n"
            f"技能: {s['skills']} 个\n"
            f"FTS5 检索: {'开启' if self.db.fts_enabled else '退化 LIKE'}\n"
            f"自动总结: {'每 ' + str(interval) + ' 分钟检查（单会话满 ' + str(min_new)
                         + ' 条或超 ' + str(grace) + ' 小时兜底）' if self._get_cfg('auto_summarize', True) else '关闭'}\n"
            f"待总结: {pend_info}\n"
            f"自动整理: {self._queue_progress()}"
        )

    @hermes.command("总结")
    async def hermes_summarize(self, event: AstrMessageEvent):
        yield event.plain_result("正在总结（手动触发会忽略条数阈值，处理全部待总结会话）……")
        result = await self._summarize(force=True)
        yield event.plain_result(result)

    @hermes.command("整理")
    async def hermes_consolidate(self, event: AstrMessageEvent, mode: str = ""):
        """/hermes 整理 [预览] —— 把碎片化的相似记忆合并成一条"""
        if not self.router.has_model():
            yield event.plain_result("没有可用的对话模型，请先在插件页面「模型」标签配置。")
            return
        dry = mode.strip() in ("预览", "试运行", "dry", "dry-run")
        yield event.plain_result("正在整理记忆（先本地聚类，有候选才会调用模型）……")
        try:
            out = await self._consolidate_memories(dry_run=dry)
            if not dry:
                self._reset_memory_queue()  # 全库刚整理过，队列里的 id 已经没意义
            yield event.plain_result(out)
        except Exception:
            yield event.plain_result(f"整理失败：{traceback.format_exc(limit=3)}")

    @hermes.command("技能整理")
    async def hermes_skill_consolidate(self, event: AstrMessageEvent, mode: str = ""):
        """/hermes 技能整理 [预览] —— 把重复的技能合并成一个"""
        if not self.router.has_model():
            yield event.plain_result("没有可用的对话模型，请先在插件页面「模型」标签配置。")
            return
        dry = mode.strip() in ("预览", "试运行", "dry", "dry-run")
        yield event.plain_result("正在检查技能重叠情况……")
        try:
            out = await self._consolidate_skills(dry_run=dry)
            if not dry:
                self.db.set_meta("skill_consolidate_pending", "0")
            yield event.plain_result(out)
        except Exception:
            yield event.plain_result(f"整理失败：{traceback.format_exc(limit=3)}")

    @hermes.command("记住")
    async def hermes_remember(self, event: AstrMessageEvent, *, content: str = ""):
        if not content.strip():
            yield event.plain_result("用法：/hermes 记住 <内容>")
            return
        sc = self._scope_keys(event)
        mid = self.db.add_memory("fact", content.strip(), scope="user",
                                 source="manual", owner_id=sc["owner_id"],
                                 session_id=sc["sid"], platform=sc["platform"],
                                 group_id=sc["group_id"])
        self._queue_memories([mid])
        yield event.plain_result(f"已记住（#{mid}）：{content.strip()}")

    @hermes.command("忘记")
    async def hermes_forget(self, event: AstrMessageEvent, memory_id: int):
        ok = self.db.delete_memory(memory_id)
        if ok:
            queue = [i for i in self._pending_memory_ids() if i != int(memory_id)]
            self.db.set_meta("consolidate_pending_memories", json.dumps(queue))
        yield event.plain_result(f"记忆 #{memory_id} 已删除。" if ok else f"记忆 #{memory_id} 不存在。")

    @hermes.command("记忆")
    async def hermes_memories(self, event: AstrMessageEvent):
        mems = self.db.all_memories(limit=30)
        if not mems:
            yield event.plain_result("还没有任何长期记忆。")
            return
        text = "\n".join(f"#{m['id']} [{m['kind']}] {m['content'][:60]}" for m in mems)
        yield event.plain_result("[最近记忆]\n" + text)

    @hermes.command("技能")
    async def hermes_skills(self, event: AstrMessageEvent, name: str = ""):
        sc = self._scope_keys(event)
        if name:
            sk = self.hub.read(name, sc["owner_id"], sc["group_id"])
            if not sk:
                yield event.plain_result(f"技能 {name} 不存在。发 /hermes 技能 看全部。")
                return
            tag = "系统·只读" if sk.get("source") == "system" else "我的"
            yield event.plain_result(
                f"[{sk['name']}]（{tag}）{sk['trigger']}\n\n{sk['content'][:1500]}"
            )
            return
        skills = self.hub.list_for(sc["owner_id"], sc["group_id"])
        if not skills:
            yield event.plain_result("还没有学会任何技能。完成复杂任务后我会自动提炼。")
            return
        personal = [s for s in skills if s.get("source") == "personal"]
        system = [s for s in skills if s.get("source") == "system"]
        lines: list[str] = []
        if personal:
            lines.append("[我的技能]")
            lines += [f"- {s['name']}（{s['trigger'] or '通用'}）" for s in personal]
        if system:
            if lines:
                lines.append("")
            lines.append("[系统技能·只读]")
            lines += [f"- {s['name']}（{s['trigger'] or '通用'}）" for s in system]
        yield event.plain_result("\n".join(lines))

    @hermes.command("导出")
    async def hermes_export(self, event: AstrMessageEvent):
        md = self.db.export_memories_markdown()
        path = os.path.join(_default_data_dir(), "MEMORY.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        yield event.plain_result(f"已导出到插件数据目录 MEMORY.md（{len(md)} 字符）。")

    @hermes.command("图")
    async def hermes_graph(self, event: AstrMessageEvent, *, code: str = ""):
        """/hermes 图 <mermaid 源码> —— 本地渲染并发图。也可直接让模型调 render_mermaid 工具。"""
        if not code.strip():
            yield event.plain_result(
                "用法：/hermes 图 <mermaid源码>\n"
                "示例：/hermes 图 flowchart TD\\n A[开始] --> B[结束]\n"
                "更方便的方式：直接在对话里让我\"画个流程图\"，我会自动调用渲染工具。"
            )
            return
        if Image is None:
            yield event.plain_result("当前 AstrBot 版本缺少图片组件，无法发图。")
            return
        code = code.replace("\\n", "\n")
        out = os.path.join(
            _default_data_dir(), "renders",
            f"m_{hashlib.md5(code.encode()).hexdigest()[:12]}.png",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        yield event.plain_result("正在本地渲染图表……")
        try:
            await render_mermaid(code, out, _default_data_dir())
            yield event.chain([Image.fromFileSystem(out)])
        except RenderError as e:
            yield event.plain_result(f"渲染失败：{e}")
        except Exception:
            yield event.plain_result(f"渲染异常：{traceback.format_exc(limit=2)}")

    # ============ 接管系统级记忆 ============

    @hermes.command("导入历史")
    async def hermes_import_history(self, event: AstrMessageEvent):
        """把 AstrBot 原生对话历史导入本插件记忆库，实现无缝接管。"""
        imported = await self._import_native_history(event.unified_msg_origin)
        yield event.plain_result(imported)

    async def _import_native_history(self, umo: str) -> str:
        cm = getattr(self.context, "conversation_manager", None)
        if cm is None:
            return "当前 AstrBot 版本没有 conversation_manager，无法导入。"
        try:
            count = 0
            convs = await cm.get_conversations(umo)
            for conv in convs or []:
                history = conv.history
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                    except Exception:
                        continue
                for h in history or []:
                    if not isinstance(h, dict):
                        continue
                    role = str(h.get("role", ""))
                    content = str(h.get("content", "")).strip()
                    if not content or role not in ("user", "assistant"):
                        continue
                    self.db.add_message(umo, role, content[:2000])
                    count += 1
            return f"导入完成：{count} 条原生对话记录已并入记忆库。"
        except Exception:
            logger.error(f"[hermes_core] 导入历史失败:\n{traceback.format_exc()}")
            return "导入失败，详见后台日志。"

    def _check_memory_conflicts(self):
        """接管模式下的防双重注入检查：提醒关闭原生长期记忆与其他记忆插件。"""
        if not self._get_cfg("exclusive_memory", True):
            return
        try:
            plugins = getattr(self.context, "get_all_stars", lambda: [])() or []
            names = [getattr(p, "name", "") for p in plugins]
            conflicts = [n for n in names if n and any(
                k in n.lower() for k in ("memory", "livingmemory", "hindsight", "记忆"))]
            conflicts = [n for n in conflicts if n != "astrbot_plugin_hermes_core"]
            if conflicts:
                logger.warning(
                    f"[hermes_core] 检测到其他记忆插件: {conflicts}。"
                    "为避免记忆双重注入，建议在 WebUI 禁用它们（本插件已独占记忆链路）。"
                )
            logger.info(
                "[hermes_core] 接管模式：请确认 AstrBot 内置长期记忆（试验性功能）已关闭，"
                "路径：WebUI → 配置 → AI 相关选项。"
            )
        except Exception:
            pass

    # ============ 身份归一（解决 openid 碎片化）============

    @hermes.command("我的ID")
    async def hermes_my_id(self, event: AstrMessageEvent):
        sc = self._scope_keys(event)
        hint = ""
        if sc["platform"].startswith("qq_official"):
            hint = ("\n提示：QQ 官方适配器用的是 openid（群内 member_openid / 私聊 user_openid），"
                    "不是 QQ 号。同一个人在不同群、或群聊与私聊之间 openid 不同，"
                    "记忆会分散——用 /hermes 绑定 <另一个openid> 可合并。")
        yield event.plain_result(
            f"[当前身份]\n"
            f"平台: {sc['platform']}\n"
            f"发送者ID(原始): {sc['raw_id']}\n"
            f"记忆归属ID: {sc['owner_id']}\n"
            f"群ID: {sc['group_id'] or '无（私聊）'}\n"
            f"会话(umo): {sc['sid']}{hint}"
        )

    @hermes.command("绑定")
    async def hermes_bind(self, event: AstrMessageEvent, alias_id: str = ""):
        """把另一个 ID（如其他群的 member_openid）绑定到当前身份，记忆即可互通。"""
        alias_id = alias_id.strip()
        if not alias_id:
            yield event.plain_result("用法：/hermes 绑定 <另一个ID>\n先用对方身份执行 /hermes 我的ID 拿到 ID。")
            return
        sc = self._scope_keys(event)
        created = self.db.add_alias(alias_id, sc["owner_id"], sc["platform"])
        yield event.plain_result(
            f"已{'绑定' if created else '更新'}：{alias_id} → {sc['owner_id']}，之后的记忆将合并。")

    @hermes.command("身份")
    async def hermes_identities(self, event: AstrMessageEvent):
        aliases = self.db.list_aliases()
        if not aliases:
            yield event.plain_result("暂无身份绑定。用 /hermes 绑定 <ID> 把多个 openid 合并成一个人。")
            return
        text = "\n".join(f"- {a['alias_id']} → {a['canonical_id']}" for a in aliases[:30])
        yield event.plain_result("[身份绑定]\n" + text)

    @hermes.command("解绑")
    async def hermes_unbind(self, event: AstrMessageEvent, alias_id: str = ""):
        ok = self.db.delete_alias(alias_id.strip()) if alias_id.strip() else False
        yield event.plain_result(f"已解绑 {alias_id}" if ok else "解绑失败：没找到该绑定。")

    @hermes.command("清空记录")
    async def hermes_clear_records(self, event: AstrMessageEvent,
                                   target: str = "", confirm: str = ""):
        """清空对话上下文但保留长期记忆与技能。

        /hermes 清空记录                 → 预演，只显示条数
        /hermes 清空记录 全群 确认       → 清空本会话所有上下文（管理员）
        /hermes 清空记录 我 确认         → 只清自己的消息记录（任何人）
        """
        sc = self._scope_keys(event)
        valid_targets = ("我", "自己", "me", "全群", "全部", "所有")
        if target and target not in valid_targets:
            # 容错：用户直接发 /hermes 清空记录 确认
            confirm, target = target, ""

        sid = sc["sid"]
        total = self.db.count_messages(session_id=sid)
        mine = self.db.count_messages(session_id=sid, sender_id=sc["raw_id"])
        stats = self.db.stats()

        if confirm != "确认":
            yield event.plain_result(
                f"[清空对话记录]\n"
                f"本会话上下文：{total} 条（其中你的 {mine} 条）\n"
                f"长期记忆：{stats['memories']} 条 / 技能：{stats['skills']} 个（不会被删）\n\n"
                f"· 清空本会话全部上下文：/hermes 清空记录 全群 确认\n"
                f"· 只清自己的消息记录：  /hermes 清空记录 我 确认\n"
                f"（会话身份 ID 不会变化，清空后记忆仍可正常召回）"
            )
            return

        if target in ("我", "自己", "me"):
            n = self.db.clear_messages(session_id=sid, sender_id=sc["raw_id"])
            yield event.plain_result(
                f"已清除你在本会话的 {n} 条消息记录。\n"
                f"（AstrBot 的原生会话是全群共享的，未改动；长期记忆完整保留）")
            return

        try:
            is_admin = bool(event.is_admin())
        except Exception:
            is_admin = True
        if not is_admin:
            yield event.plain_result(
                "清空整个会话的上下文需要管理员权限。\n"
                "你可以只清自己的：/hermes 清空记录 我 确认")
            return

        n = self.db.clear_messages(session_id=sid)
        native = ""
        try:
            cm = getattr(self.context, "conversation_manager", None)
            if cm is not None:
                await cm.delete_conversations_by_user_id(sid)
                native = "AstrBot 原生对话上下文已一并清除。"
            else:
                native = "未找到 conversation_manager，AstrBot 原生对话请到 WebUI 手动删除。"
        except Exception:
            native = "AstrBot 原生对话清除失败，请到 WebUI → 会话管理手动删除。"
        yield event.plain_result(
            f"已清除本会话 {n} 条消息记录。{native}\n"
            f"长期记忆 {stats['memories']} 条、技能 {stats['skills']} 个均保留，"
            f"下一次对话会照常召回。")

    async def terminate(self):
        if self._bg_task:
            self._bg_task.cancel()
        self.db.close()
        logger.info("[hermes_core] 已卸载")
