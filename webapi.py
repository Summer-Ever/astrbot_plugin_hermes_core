"""Hermes 记忆系统的 WebUI 后端。

AstrBot 的插件页面通过 ``context.register_web_api`` 注册接口，前端用
``window.AstrBotPluginView.apiGet/apiPost`` 调用，路由形如
``/{插件名}/overview``，实际由 dashboard 以 ``/api/v1/plugins/extensions/...``
转发到本文件的方法。

设计原则：
- 只做「读写自己的库」这类轻量操作，耗时操作（总结/整理）保留同步返回，
  但前端会显示明确的进行中状态，避免用户以为卡死。
- 配置写入走 AstrBotConfig 的 save_config_async，和 WebUI 配置页改的是同一份文件。
- 所有输入都做类型/范围校验，不让前端把字符串塞进 int 字段。
"""

from __future__ import annotations

import json
import os
import time
import traceback

from astrbot.api import logger

try:
    from astrbot.api.web import error_response, json_response, request
except ImportError:  # pragma: no cover - 极老版本没有 web 模块
    error_response = json_response = request = None

PLUGIN_NAME = "astrbot_plugin_hermes_core"

# 配置项分组：只影响页面上的展示顺序
CONFIG_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("记忆注入", ("memory_injection", "memory_scope", "per_user_history",
                  "exclusive_memory", "inject_memories_topk", "inject_history_topk")),
    ("自动总结", ("auto_summarize", "summarize_interval_minutes",
                  "min_new_messages", "summarize_grace_hours")),
    ("记忆整理", ("auto_consolidate", "consolidate_every_n_memories",
                  "consolidate_threshold", "consolidate_min_memories",
                  "consolidate_max_clusters")),
    ("技能", ("auto_skill", "skill_notify_admin", "skill_merge_same_name",
              "skill_merge_threshold", "skill_consolidate_max",
              "skill_consolidate_every_n")),
    ("提示词", ("summary_prompt",)),
]

# 改了需要重载插件才生效的配置（后台循环在 initialize 时读取一次）
NEEDS_RELOAD = {"auto_summarize", "summarize_interval_minutes"}

MEMORY_KINDS = ("fact", "preference", "profile", "summary")


def _bucket(row: dict) -> tuple:
    """记忆的归属边界：不同用户/不同群绝不合并。"""
    return (
        row.get("scope") or "",
        row.get("owner_id") or "",
        row.get("group_id") or "",
    )


class HermesWebAPI:
    def __init__(self, plugin, version: str = "", data_dir_fn=None):
        self.plugin = plugin
        self.version = version
        self.data_dir_fn = data_dir_fn
        self.prefix = f"/{PLUGIN_NAME}"

    # ---------- 注册 ----------

    def register(self) -> None:
        routes = [
            (f"{self.prefix}/overview", self.overview, ["GET"], "概览：统计与队列进度"),
            (f"{self.prefix}/memories", self.memories, ["GET"], "记忆列表（筛选+分页）"),
            (f"{self.prefix}/memories/add", self.memory_add, ["POST"], "新增记忆"),
            (f"{self.prefix}/memories/update", self.memory_update, ["POST"], "修改记忆"),
            (f"{self.prefix}/memories/delete", self.memory_delete, ["POST"], "删除记忆"),
            (f"{self.prefix}/memories/merge", self.memory_merge, ["POST"], "合并多条记忆"),
            (f"{self.prefix}/skills", self.skills, ["GET"], "技能列表（含正文）"),
            (f"{self.prefix}/skills/save", self.skill_save, ["POST"], "新建/修改技能"),
            (f"{self.prefix}/skills/delete", self.skill_delete, ["POST"], "删除技能"),
            (f"{self.prefix}/config", self.config_get, ["GET"], "读取插件配置与 schema"),
            (f"{self.prefix}/config/save", self.config_save, ["POST"], "保存插件配置"),
            (f"{self.prefix}/actions/summarize", self.action_summarize, ["POST"], "立即总结"),
            (f"{self.prefix}/actions/consolidate", self.action_consolidate,
             ["POST"], "整理记忆"),
            (f"{self.prefix}/actions/skill-consolidate", self.action_skill_consolidate,
             ["POST"], "整理技能"),
            (f"{self.prefix}/actions/export", self.action_export, ["POST"], "导出记忆 Markdown"),
            (f"{self.prefix}/aliases", self.aliases, ["GET"], "身份别名列表"),
            (f"{self.prefix}/aliases/add", self.alias_add, ["POST"], "添加身份别名"),
            (f"{self.prefix}/aliases/delete", self.alias_delete, ["POST"], "删除身份别名"),
        ]
        for route, handler, methods, desc in routes:
            try:
                self.plugin.context.register_web_api(route, handler, methods, desc)
            except Exception:
                logger.error(f"[hermes_core] 注册 Web API {route} 失败:\n"
                             f"{traceback.format_exc(limit=2)}")

    # ---------- 小工具 ----------

    @staticmethod
    async def _body() -> dict:
        data = await request.json(default={})
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _fail(message: str, code: int = 400):
        return error_response(message, status_code=code)

    def _cfg(self, key: str, default=None):
        try:
            return self.plugin.config.get(key, default) if self.plugin.config else default
        except Exception:
            return default

    def _schema(self) -> dict:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_conf_schema.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning(f"[hermes_core] 读取配置 schema 失败:\n{traceback.format_exc(limit=2)}")
            return {}

    def _db_size(self) -> int:
        try:
            data_dir = os.path.dirname(self.plugin.db.conn.execute(
                "PRAGMA database_list").fetchone()[2] or "")
            total = 0
            for suffix in ("", "-wal", "-shm"):
                p = os.path.join(data_dir, f"hermes.db{suffix}")
                if os.path.isfile(p):
                    total += os.path.getsize(p)
            return total
        except Exception:
            return 0

    # ---------- 概览 ----------

    async def overview(self):
        db = self.plugin.db
        stats = db.stats()
        pending = db.unsummarized_by_session()
        pending_msgs = sum(p["count"] for p in pending)
        mem_every = self.plugin._mem_every()
        sk_every = self.plugin._skill_every()
        pending_mem = self.plugin._queued_memory_count()
        pending_skill = int(db.get_meta("skill_consolidate_pending", "0") or 0)
        skills = db.all_skills()
        try:
            aliases = db.list_aliases()
        except Exception:
            aliases = []
        return json_response({
            "version": self.version,
            "code_hash": self.plugin._build_hash(),
            "fts": bool(db.fts_enabled),
            "stats": stats,
            "kinds": db.memory_kind_counts(),
            "queue": {
                "memories": pending_mem,
                "memories_every": mem_every,
                "skills": pending_skill,
                "skills_every": sk_every,
                "auto_memories": bool(self._cfg("auto_consolidate", True)),
            },
            "summarize": {
                "auto": bool(self._cfg("auto_summarize", True)),
                "interval": self._cfg("summarize_interval_minutes", 30),
                "min_new": self._cfg("min_new_messages", 8),
                "grace_hours": self._cfg("summarize_grace_hours", 6),
                "pending_sessions": len(pending),
                "pending_messages": pending_msgs,
            },
            "db_bytes": self._db_size(),
            "aliases": len(aliases),
            "recent_memories": db.all_memories(limit=8),
            "top_skills": sorted(skills, key=lambda s: -int(s.get("use_count") or 0))[:6],
            "now": time.time(),
        })

    # ---------- 记忆 ----------

    async def memories(self):
        db = self.plugin.db
        q = (request.query.get("q") or "").strip()
        scope = (request.query.get("scope") or "").strip()
        kind = (request.query.get("kind") or "").strip()
        owner = (request.query.get("owner") or "").strip()
        merged = (request.query.get("merged") or "0") in ("1", "true", "yes")
        try:
            limit = int(request.query.get("limit") or 50)
            offset = int(request.query.get("offset") or 0)
        except ValueError:
            return self._fail("limit/offset 必须是数字")
        if scope and scope not in ("user", "group", "global"):
            return self._fail("scope 只能是 user/group/global")
        if kind and kind not in MEMORY_KINDS:
            return self._fail("未知的记忆类型")

        rows, total = db.query_memories(
            q=q, scope=scope, kind=kind, owner_id=owner,
            include_merged=merged, limit=limit, offset=offset,
        )
        return json_response({
            "records": rows, "total": total, "limit": limit, "offset": offset,
            "kinds": db.memory_kind_counts(),
        })

    async def memory_add(self):
        data = await self._body()
        content = str(data.get("content") or "").strip()
        if not content:
            return self._fail("记忆内容不能为空")
        kind = str(data.get("kind") or "fact").strip()
        if kind not in MEMORY_KINDS:
            return self._fail("未知的记忆类型")
        scope = str(data.get("scope") or self._cfg("memory_scope", "user")).strip()
        if scope not in ("user", "group", "global"):
            return self._fail("scope 只能是 user/group/global")
        owner = str(data.get("owner_id") or "").strip()
        if scope == "user" and not owner:
            return self._fail("按用户隔离的记忆必须填写归属 ID")
        mid = self.plugin.db.add_memory(
            kind, content[:500], scope=scope, source="webui",
            owner_id=owner, platform=str(data.get("platform") or ""),
        )
        return json_response({"id": mid})

    async def memory_update(self):
        data = await self._body()
        try:
            mid = int(data.get("id"))
        except (TypeError, ValueError):
            return self._fail("缺少记忆 id")
        content = data.get("content")
        kind = data.get("kind")
        if content is not None:
            content = str(content).strip()
            if not content:
                return self._fail("记忆内容不能为空")
            content = content[:500]
        if kind is not None and str(kind) not in MEMORY_KINDS:
            return self._fail("未知的记忆类型")
        ok = self.plugin.db.update_memory(mid, content, None if kind is None else str(kind))
        if not ok:
            return self._fail("记忆不存在或没有改动", 404)
        return json_response({"id": mid, "record": self.plugin.db.memory_by_id(mid)})

    async def memory_delete(self):
        data = await self._body()
        try:
            mid = int(data.get("id"))
        except (TypeError, ValueError):
            return self._fail("缺少记忆 id")
        ok = self.plugin.db.delete_memory(mid)
        if ok:
            queue = [i for i in self.plugin._pending_memory_ids() if i != mid]
            self.plugin.db.set_meta("consolidate_pending_memories", json.dumps(queue))
        return json_response({"deleted": bool(ok)})

    async def memory_merge(self):
        data = await self._body()
        raw_ids = data.get("ids") or []
        content = str(data.get("content") or "").strip()
        kind = str(data.get("kind") or "").strip()
        if not isinstance(raw_ids, list) or len(raw_ids) < 2:
            return self._fail("至少选择两条记忆")
        try:
            ids = [int(i) for i in raw_ids]
        except (TypeError, ValueError):
            return self._fail("id 必须是数字")
        if not content:
            return self._fail("合并后的内容不能为空")
        rows = [self.plugin.db.memory_by_id(i) for i in ids]
        if any(r is None for r in rows):
            return self._fail("有记忆已被删除，请刷新后重试", 404)
        merged_flags = [r for r in rows if int(r.get("merged_into") or 0) != 0]
        if merged_flags:
            return self._fail("选中的记忆里包含已被合并的条目，请刷新后重试")
        if len({_bucket(r) for r in rows}) != 1:
            return self._fail("不同用户/不同群的记忆不能合并")
        new_id = self.plugin.db.merge_memories(ids, content[:500], kind)
        if not new_id:
            return self._fail("合并失败", 500)
        queue = [i for i in self.plugin._pending_memory_ids() if i not in set(ids)]
        self.plugin.db.set_meta("consolidate_pending_memories", json.dumps(queue))
        return json_response({"id": new_id})

    # ---------- 技能 ----------

    async def skills(self):
        return json_response({
            "records": self.plugin.db.all_skills(),
            "max_direct": self._cfg("skill_consolidate_max", 40),
        })

    async def skill_save(self):
        data = await self._body()
        name = str(data.get("name") or "").strip()
        old_name = str(data.get("old_name") or "").strip()
        trigger = str(data.get("trigger") or "").strip()
        content = str(data.get("content") or "").strip()
        if not name:
            return self._fail("技能名不能为空")
        if len(name) > 64 or any(ch.isspace() for ch in name):
            return self._fail("技能名最多 64 字符，且不能包含空格")
        if len(content) < 5:
            return self._fail("技能内容太短")
        if old_name and old_name != name and self.plugin.db.peek_skill(name):
            return self._fail("已存在同名技能")
        created = self.plugin.db.upsert_skill(name, trigger, content[:8000])
        if old_name and old_name != name:
            self.plugin.db.delete_skill(old_name)
        return json_response({"name": name, "created": bool(created)})

    async def skill_delete(self):
        data = await self._body()
        name = str(data.get("name") or "").strip()
        if not name:
            return self._fail("缺少技能名")
        ok = self.plugin.db.delete_skill(name)
        return json_response({"deleted": bool(ok)})

    # ---------- 配置 ----------

    async def config_get(self):
        if self.plugin.config is None:
            return self._fail("插件没有拿到配置对象", 500)
        schema = self._schema()
        values = dict(self.plugin.config)
        groups: list[dict] = []
        seen: set[str] = set()
        for title, keys in CONFIG_GROUPS:
            items = []
            for key in keys:
                meta = schema.get(key)
                if not meta:
                    continue
                seen.add(key)
                items.append({
                    "key": key,
                    "type": meta.get("type", "string"),
                    "default": meta.get("default"),
                    "description": meta.get("description", ""),
                    "hint": meta.get("hint", ""),
                    "options": meta.get("options"),
                    "value": values.get(key, meta.get("default")),
                    "needs_reload": key in NEEDS_RELOAD,
                })
            if items:
                groups.append({"title": title, "items": items})
        rest = [
            {
                "key": key, "type": meta.get("type", "string"),
                "default": meta.get("default"),
                "description": meta.get("description", ""),
                "hint": meta.get("hint", ""),
                "options": meta.get("options"),
                "value": values.get(key, meta.get("default")),
                "needs_reload": key in NEEDS_RELOAD,
            }
            for key, meta in schema.items() if key not in seen
        ]
        if rest:
            groups.append({"title": "其他", "items": rest})
        return json_response({"groups": groups, "values": values})

    async def config_save(self):
        data = await self._body()
        values = data.get("values")
        if not isinstance(values, dict) or not values:
            return self._fail("没有要保存的配置")
        schema = self._schema()
        if not schema:
            return self._fail("读取配置 schema 失败", 500)
        cleaned: dict = {}
        errors: list[str] = []
        for key, raw in values.items():
            meta = schema.get(key)
            if not meta:
                errors.append(f"{key}：未知配置项")
                continue
            typ = meta.get("type", "string")
            try:
                if typ == "bool":
                    if isinstance(raw, bool):
                        cleaned[key] = raw
                    elif str(raw).lower() in ("true", "1", "yes", "on"):
                        cleaned[key] = True
                    elif str(raw).lower() in ("false", "0", "no", "off"):
                        cleaned[key] = False
                    else:
                        raise ValueError("需要布尔值")
                elif typ in ("int", "number"):
                    val = int(float(raw))
                    low, high = meta.get("min"), meta.get("max")
                    if isinstance(low, (int, float)) and val < low:
                        raise ValueError(f"不能小于 {low}")
                    if isinstance(high, (int, float)) and val > high:
                        raise ValueError(f"不能大于 {high}")
                    cleaned[key] = val
                elif typ == "float":
                    cleaned[key] = float(raw)
                else:
                    text = str(raw)
                    options = meta.get("options")
                    if options and text not in options:
                        raise ValueError(f"只能是 {'/'.join(options)}")
                    cleaned[key] = text
            except (TypeError, ValueError) as exc:
                errors.append(f"{key}：{exc}")
        if errors:
            return self._fail("；".join(errors))
        try:
            self.plugin.config.update(cleaned)
            await self.plugin.config.save_config_async()
        except Exception:
            logger.error(f"[hermes_core] 保存配置失败:\n{traceback.format_exc()}")
            return self._fail("写入配置文件失败，请查看后端日志", 500)
        need_reload = [k for k in cleaned if k in NEEDS_RELOAD]
        return json_response({"saved": list(cleaned), "needs_reload": need_reload})

    # ---------- 操作 ----------

    async def action_summarize(self):
        provider = self.plugin.context.get_using_provider()
        if provider is None:
            return self._fail("未配置 LLM Provider，无法总结")
        result = await self.plugin._summarize(force=True)
        return json_response({"result": result})

    async def action_consolidate(self):
        data = await self._body()
        dry = bool(data.get("dry_run"))
        provider = self.plugin.context.get_using_provider()
        if provider is None:
            return self._fail("未配置 LLM Provider，无法整理")
        result = await self.plugin._consolidate_memories(provider, dry_run=dry)
        if not dry:
            self.plugin._reset_memory_queue()
        return json_response({"result": result, "dry_run": dry})

    async def action_skill_consolidate(self):
        data = await self._body()
        dry = bool(data.get("dry_run"))
        provider = self.plugin.context.get_using_provider()
        if provider is None:
            return self._fail("未配置 LLM Provider，无法整理")
        result = await self.plugin._consolidate_skills(provider, dry_run=dry)
        if not dry:
            self.plugin.db.set_meta("skill_consolidate_pending", "0")
        return json_response({"result": result, "dry_run": dry})

    async def action_export(self):
        md = self.plugin.db.export_memories_markdown()
        if self.data_dir_fn is None:
            return self._fail("插件数据目录不可用", 500)
        path = os.path.join(self.data_dir_fn(), "MEMORY.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        return json_response({"path": path, "chars": len(md), "preview": md[:1500]})

    # ---------- 身份别名 ----------

    async def aliases(self):
        return json_response({"records": self.plugin.db.list_aliases()})

    async def alias_add(self):
        data = await self._body()
        alias = str(data.get("alias") or "").strip()
        canonical = str(data.get("canonical") or "").strip()
        if not alias or not canonical:
            return self._fail("别名和主 ID 都要填")
        if alias == canonical:
            return self._fail("别名不能等于主 ID")
        ok = self.plugin.db.add_alias(alias, canonical, str(data.get("platform") or ""))
        return json_response({"ok": bool(ok)})

    async def alias_delete(self):
        data = await self._body()
        alias = str(data.get("alias") or "").strip()
        if not alias:
            return self._fail("缺少别名 ID")
        return json_response({"deleted": bool(self.plugin.db.delete_alias(alias))})
