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

from .db import HermesDB

SYSTEM_MEMORY_HINT = (
    "你拥有持久记忆系统。可用工具：search_memory(检索记忆/历史)、save_memory(保存重要事实)、"
    "list_skills(查看已学会的技能)、read_skill(读取技能全文)、save_skill(把可复用流程存为技能)。"
    "遇到值得长期记住的信息请主动 save_memory；完成复杂流程后可用 save_skill 沉淀方法。"
)


@register("astrbot_plugin_hermes_core", "workbuddy", "Hermes 风格记忆与技能学习闭环", "0.1.0")
class HermesCore(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config
        self.db = HermesDB(_default_data_dir())
        self._bg_task: asyncio.Task | None = None
        self._last_origin: str | None = None  # 最近活跃会话，用于技能生成通知

    async def initialize(self):
        logger.info(f"[hermes_core] 已加载，FTS5: {'开启' if self.db.fts_enabled else '不可用(退化为 LIKE)'}")
        if self._get_cfg("auto_summarize", True):
            self._bg_task = asyncio.create_task(self._bg_loop())

    def _get_cfg(self, key: str, default):
        if self.config is None:
            return default
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    # ============ 钩子：请求前注入记忆 ============

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
        sid = event.unified_msg_origin
        mem_k = int(self._get_cfg("inject_memories_topk", 5))
        hist_k = int(self._get_cfg("inject_history_topk", 3))

        memories = self.db.search_memories(query, limit=mem_k)
        if memories:
            lines.append("[长期记忆参考]")
            lines += [f"- ({m['kind']}) {m['content']}" for m in memories]

        history = self.db.search_messages(query, limit=hist_k, session_id=sid)
        if history:
            lines.append("[相关历史对话]")
            for h in reversed(history):
                who = "用户" if h["role"] == "user" else "你"
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
                )
            reply = ""
            try:
                reply = (resp.completion_text or "").strip()
            except Exception:
                pass
            if reply:
                self.db.add_message(sid, "assistant", reply)
        except Exception:
            logger.error(f"[hermes_core] 落库失败:\n{traceback.format_exc()}")

    # ============ LLM 工具：模型主动使用记忆/技能 ============

    @filter.llm_tool(name="search_memory")
    async def tool_search_memory(self, event: AstrMessageEvent, query: str) -> str:
        """按关键词检索长期记忆与历史对话。当需要回忆用户偏好、之前聊过的内容时调用。
        Args:
            query: 检索关键词
        """
        mems = self.db.search_memories(query, limit=5)
        msgs = self.db.search_messages(query, limit=5, session_id=event.unified_msg_origin)
        out = []
        if mems:
            out.append("[记忆] " + " | ".join(f"{m['content']}" for m in mems))
        if msgs:
            out.append("[历史] " + " | ".join(f"{m['role']}:{m['content'][:80]}" for m in reversed(msgs)))
        return "\n".join(out) or "没有找到相关内容。"

    @filter.llm_tool(name="save_memory")
    async def tool_save_memory(self, event: AstrMessageEvent, content: str,
                               kind: str = "fact") -> str:
        """把值得长期记住的信息保存为记忆。用户偏好、重要事实、项目背景等都应保存。
        Args:
            content: 记忆内容（简洁的一句话）
            kind: 类型，fact/preference/profile/summary 之一
        """
        mid = self.db.add_memory(kind or "fact", content, scope="global", source="llm_tool")
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
            name: 技能名称
        """
        sk = self.db.get_skill(name)
        return sk["content"] if sk else f"技能 {name} 不存在。"

    @filter.llm_tool(name="save_skill")
    async def tool_save_skill(self, event: AstrMessageEvent, name: str,
                              trigger: str, content: str) -> str:
        """把一个可复用的解决流程保存为技能（Markdown 步骤）。完成复杂任务后若流程可复用，调用此工具。
        Args:
            name: 英文短名，如 deploy_blog
            trigger: 什么场景下应该使用该技能
            content: Markdown 格式的完整步骤
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

    async def _summarize(self) -> str:
        provider = self.context.get_using_provider()
        if provider is None:
            return "未配置 LLM Provider，无法总结。"
        min_new = int(self._get_cfg("min_new_messages", 8))
        last_id = int(self.db.get_meta("last_summarized_id", "0"))
        msgs = self.db.messages_since(last_id)
        new_count = len(msgs)
        if new_count < min_new:
            return f"新消息仅 {new_count} 条（阈值 {min_new}），跳过。"

        # 按会话分组拼接记录
        by_session: dict[str, list[dict]] = {}
        for m in msgs:
            by_session.setdefault(m["session_id"], []).append(m)
        transcript_parts = []
        for sid, group in by_session.items():
            lines = [f"用户{g['sender_name'] or g['sender_id']}: {g['content'][:300]}"
                     if g["role"] == "user" else f"助手: {g['content'][:300]}" for g in group]
            transcript_parts.append(f"== 会话 {sid} ==\n" + "\n".join(lines))
        transcript = "\n\n".join(transcript_parts)[:12000]

        prompt = self._get_cfg("summary_prompt", "") + "\n\n对话记录：\n" + transcript
        resp = await provider.text_chat(prompt=prompt, session_id=None)
        text = ""
        try:
            text = resp.completion_text or ""
        except Exception:
            pass
        items = self._extract_json_array(text)
        made_mem, made_skill = 0, 0
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("skill") and self._get_cfg("auto_skill", True):
                name = re.sub(r"\s+", "_", str(item.get("name", "")).strip())[:64]
                if name and item.get("content"):
                    self.db.upsert_skill(name, str(item.get("trigger", "")), str(item["content"]))
                    made_skill += 1
            elif item.get("content"):
                self.db.add_memory(
                    str(item.get("kind", "fact")), str(item["content"]),
                    scope=str(item.get("scope", "global")), source="auto_summary",
                )
                made_mem += 1
        self.db.set_meta("last_summarized_id", str(msgs[-1]["id"]))
        summary = f"[hermes_core] 总结完成：新记忆 {made_mem} 条，新技能 {made_skill} 个（处理 {new_count} 条消息）。"
        logger.info(summary)
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
        yield event.plain_result(
            f"[Hermes 记忆系统]\n"
            f"对话消息: {s['messages']} 条 / {s['sessions']} 个会话\n"
            f"长期记忆: {s['memories']} 条\n"
            f"技能: {s['skills']} 个\n"
            f"FTS5 检索: {'开启' if self.db.fts_enabled else '退化 LIKE'}\n"
            f"自动总结: {'每 ' + str(self._get_cfg('summarize_interval_minutes', 30)) + ' 分钟' if self._get_cfg('auto_summarize', True) else '关闭'}"
        )

    @hermes.command("总结")
    async def hermes_summarize(self, event: AstrMessageEvent):
        yield event.plain_result("正在总结……")
        result = await self._summarize()
        yield event.plain_result(result)

    @hermes.command("记住")
    async def hermes_remember(self, event: AstrMessageEvent, *, content: str = ""):
        if not content.strip():
            yield event.plain_result("用法：/hermes 记住 <内容>")
            return
        mid = self.db.add_memory("fact", content.strip(), source="manual")
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

    async def terminate(self):
        if self._bg_task:
            self._bg_task.cancel()
        self.db.close()
        logger.info("[hermes_core] 已卸载")
