"""astrbot_plugin_hermes_core

复刻 NousResearch/hermes-agent 学习闭环的 AstrBot 插件：
- 记忆注入：每次 LLM 请求前检索相关长期记忆与历史对话（on_llm_request）
- 会话落库：SQLite + WAL + FTS5，全文可检索（on_llm_response）
- 自动总结：后台循环把新对话压缩为记忆条目（Hermes 的 nudge 机制）
- 技能进化：总结时提炼可复用 Markdown 技能；模型也可主动存技能（llm_tool）
- 工具面：模型可调用 search_memory / save_memory / list_skills / read_skill / save_skill
"""

import asyncio
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

SYSTEM_MEMORY_HINT = (
    "你拥有持久记忆系统。可用工具：search_memory(检索记忆/历史)、save_memory(保存重要事实)、"
    "list_skills(查看已学会的技能)、read_skill(读取技能全文)、save_skill(把可复用流程存为技能)。"
    "遇到值得长期记住的信息请主动 save_memory；完成复杂流程后可用 save_skill 沉淀方法。"
)


@register("astrbot_plugin_hermes_core", "workbuddy", "Hermes 风格记忆与技能学习闭环", "0.4.0")
class HermesCore(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config
        self.db = HermesDB(_default_data_dir())
        self._bg_task: asyncio.Task | None = None
        self._last_origin: str | None = None  # 最近活跃会话，用于技能生成通知

    async def initialize(self):
        logger.info(f"[hermes_core] 已加载，FTS5: {'开启' if self.db.fts_enabled else '不可用(退化为 LIKE)'}")
        self._migrate_watermarks()
        self._check_memory_conflicts()
        if self._get_cfg("auto_summarize", True):
            self._bg_task = asyncio.create_task(self._bg_loop())

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

        skills = self.db.list_skills()
        if skills:
            names = "、".join(s["name"] for s in skills[:20])
            lines.append(f"[已掌握技能] {names}（可用 read_skill 查看全文）")

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
        return f"已保存记忆 #{mid}。"

    @filter.llm_tool(name="list_skills")
    async def tool_list_skills(self, event: AstrMessageEvent) -> str:
        """列出已学会的全部技能名称与适用场景。"""
        skills = self.db.list_skills()
        if not skills:
            return "暂无技能。"
        return "\n".join(f"- {s['name']}（{s['trigger'] or '通用'}）" for s in skills)

    @filter.llm_tool(name="read_skill")
    async def tool_read_skill(self, event: AstrMessageEvent, name: str) -> str:
        """读取指定技能的完整 Markdown 内容。执行该技能覆盖的任务前应先调用。
        Args:
            name (string): 技能名称
        """
        sk = self.db.get_skill(name)
        return sk["content"] if sk else f"技能 {name} 不存在。"

    @filter.llm_tool(name="save_skill")
    async def tool_save_skill(self, event: AstrMessageEvent, name: str,
                              trigger: str, content: str) -> str:
        """把一个可复用的解决流程保存为技能（Markdown 步骤）。完成复杂任务后若流程可复用，调用此工具。
        Args:
            name (string): 英文短名，如 deploy_blog
            trigger (string): 什么场景下应该使用该技能
            content (string): Markdown 格式的完整步骤
        """
        name = re.sub(r"\s+", "_", name.strip())[:64]
        created = self.db.upsert_skill(name, trigger or "", content)
        logger.info(f"[hermes_core] 技能 {'新建' if created else '更新'}: {name}")
        return f"技能 {name} 已{'保存' if created else '更新'}。"

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
        provider = self.context.get_using_provider()
        if provider is None:
            return "未配置 LLM Provider，无法总结。"

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

        made_mem = made_skill = done_msgs = 0
        details: list[str] = []
        for st, reason in due:
            batch = self.db.session_batch(st["session_id"], st["watermark"], limit=200)
            if not batch:
                continue
            mem, sk = await self._summarize_session(provider, batch, st["session_id"])
            made_mem += mem
            made_skill += sk
            done_msgs += len(batch)
            # 只推进到本批最后一条：积压超 200 条时下一轮继续处理
            self.db.set_watermark(st["session_id"], batch[-1]["id"])
            details.append(f"{st['session_id'][:36]}：{len(batch)} 条（{reason}）")

        summary = (f"[hermes_core] 总结完成：{len(details)} 个会话 / {done_msgs} 条消息，"
                   f"新记忆 {made_mem} 条，新技能 {made_skill} 个。")
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

    async def _summarize_session(self, provider, batch: list[dict],
                                 sid: str) -> tuple[int, int]:
        """把一个会话的一批消息交给 LLM 归纳，写入记忆与技能。"""
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
        transcript = "\n".join(lines)[:12000]

        prompt = (self._get_cfg("summary_prompt", "")
                  + "\n\n对话中每位用户都带有 [用户 ID(昵称)] 标记。"
                    "若某条记忆只属于某个特定用户，请在该项追加 \"owner\": \"该用户的ID\"；"
                    "属于整段对话/全体时省略 owner。\n\n对话记录：\n" + transcript)
        resp = await provider.text_chat(prompt=prompt, session_id=None)
        text = ""
        try:
            text = resp.completion_text or ""
        except Exception:
            pass
        made_mem = made_skill = 0
        for item in self._extract_json_array(text):
            if not isinstance(item, dict):
                continue
            if item.get("skill") and self._get_cfg("auto_skill", True):
                name = re.sub(r"\s+", "_", str(item.get("name", "")).strip())[:64]
                if name and item.get("content"):
                    self.db.upsert_skill(name, str(item.get("trigger", "")), str(item["content"]))
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
        return made_mem, made_skill

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
            f"对话消息: {s['messages']} 条 / {s['sessions']} 个会话\n"
            f"长期记忆: {s['memories']} 条\n"
            f"技能: {s['skills']} 个\n"
            f"FTS5 检索: {'开启' if self.db.fts_enabled else '退化 LIKE'}\n"
            f"自动总结: {'每 ' + str(interval) + ' 分钟检查（单会话满 ' + str(min_new)
                         + ' 条或超 ' + str(grace) + ' 小时兜底）' if self._get_cfg('auto_summarize', True) else '关闭'}\n"
            f"待总结: {pend_info}"
        )

    @hermes.command("总结")
    async def hermes_summarize(self, event: AstrMessageEvent):
        yield event.plain_result("正在总结（手动触发会忽略条数阈值，处理全部待总结会话）……")
        result = await self._summarize(force=True)
        yield event.plain_result(result)

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
        yield event.plain_result(f"已记住（#{mid}）：{content.strip()}")

    @hermes.command("忘记")
    async def hermes_forget(self, event: AstrMessageEvent, memory_id: int):
        ok = self.db.delete_memory(memory_id)
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
        if name:
            sk = self.db.get_skill(name)
            if not sk:
                yield event.plain_result(f"技能 {name} 不存在。")
            else:
                yield event.plain_result(f"[{sk['name']}] {sk['trigger']}\n\n{sk['content'][:1500]}")
            return
        skills = self.db.list_skills()
        if not skills:
            yield event.plain_result("还没有学会任何技能。完成复杂任务后我会自动提炼。")
        else:
            text = "\n".join(f"- {s['name']}（{s['trigger'] or '通用'}）" for s in skills)
            yield event.plain_result("[技能库]\n" + text)

    @hermes.command("导出")
    async def hermes_export(self, event: AstrMessageEvent):
        md = self.db.export_memories_markdown()
        path = os.path.join(_default_data_dir(), "MEMORY.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        yield event.plain_result(f"已导出到插件数据目录 MEMORY.md（{len(md)} 字符）。")

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
