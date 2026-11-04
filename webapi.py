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

try:
    from .modules.paths import safe_join
except ImportError:
    from modules.paths import safe_join

PLUGIN_NAME = "astrbot_plugin_hermes_core"

# 配置项分组：只影响页面上的展示顺序
CONFIG_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("记忆注入", ("memory_injection", "memory_scope", "per_user_history",
                  "group_private_memory", "exclusive_memory",
                  "inject_memories_topk", "inject_history_topk",
                  "recall_scan_limit")),
    ("自动总结", ("auto_summarize", "summarize_interval_minutes",
                  "min_new_messages", "summarize_grace_hours")),
    ("记忆整理", ("auto_consolidate", "consolidate_every_n_memories",
                  "consolidate_threshold", "consolidate_min_memories",
                  "consolidate_max_clusters")),
    ("技能", ("auto_skill", "skill_notify_admin", "skill_merge_same_name",
              "skill_merge_threshold", "skill_consolidate_max",
              "skill_consolidate_every_n")),
    # 模型链的列表型配置（降级顺序、可重试错误）由「模型」页专门维护，
    # 设置页只放标量项，避免两处编辑同一份数据打架。
    ("模型容错", ("llm_max_retries", "llm_timeout", "llm_backoff",
                  "llm_total_attempts", "llm_cooldown_seconds",
                  "llm_max_prompt_chars")),
    ("提示词", ("summary_prompt",)),
]

# 不在设置页渲染的配置项（由「模型」页以更直观的方式维护）
HIDDEN_IN_SETTINGS = {"llm_primary_provider", "llm_fallback_providers", "llm_retry_on"}

# 改了需要重载插件才生效的配置（后台循环在 initialize 时读取一次）
NEEDS_RELOAD = {"auto_summarize", "summarize_interval_minutes"}

MEMORY_KINDS = ("fact", "preference", "profile", "summary")


def _bucket(row: dict) -> tuple:
    """记忆的归属边界：不同用户/不同群/不同可见性绝不合并。

    visibility 必须进桶。merge_memories 继承组内最早那条的可见性：
    把一条私聊记忆和一条群可见记忆合在一起，结果要么私事变群可见，
    要么群里从此想不起来——两者都不可自愈。v0.12.13 修过整理链路，
    但面板的合并入口当时漏了同一件事。
    """
    return (
        row.get("scope") or "",
        row.get("owner_id") or "",
        row.get("group_id") or "",
        row.get("visibility") or "",
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
            (f"{self.prefix}/memories/pin", self.memory_pin, ["POST"], "钉住/取消钉住记忆（常驻保护）"),
            (f"{self.prefix}/memories/visibility", self.memory_visibility, ["POST"],
             "设置记忆可见范围（私聊/群/公开）"),
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
            (f"{self.prefix}/models", self.models, ["GET"], "模型链、容错策略与调用日志"),
            (f"{self.prefix}/models/save", self.models_save, ["POST"], "保存主模型与降级链"),
            (f"{self.prefix}/models/test", self.models_test, ["POST"], "测试模型连通性"),
            (f"{self.prefix}/models/logs/clear", self.models_logs_clear,
             ["POST"], "清空调用日志与冷却"),
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

    def _model_summary(self) -> dict:
        """概览页只需要「用哪个模型、链多长、最近有没有出错」。"""
        router = getattr(self.plugin, "router", None)
        if router is None:
            return {}
        try:
            primary = router.configured_primary()
            chain = [e["id"] for e in router.effective_chain(ignore_cooldown=True)]
            logs = router.logs(20)
            return {
                # 只报插件自己配的主模型。以前这里是 `primary or current_provider_id()`，
                # 概览页会把 AstrBot 的对话模型显示成"总结模型"，掩盖了实际没配置这件事。
                "primary": primary,
                "detached": True,
                "chain": chain,
                "last_used": getattr(router, "_last_used", ""),
                "recent_fail": sum(1 for x in logs if not x["ok"]),
                "cooling": [p["id"] for p in router.all_providers() if p["cooling"] > 0],
            }
        except Exception:
            return {}

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
            "models": self._model_summary(),
            "recent_memories": db.all_memories(limit=8),
            "top_skills": sorted(skills, key=lambda s: -int(s.get("use_count") or 0))[:6],
            "skills_split": {
                "personal": len(skills),
                "system": len(self.plugin.hub.system_skills()),
            },
            "metrics": self.plugin.metrics_snapshot(),
            "router": dict(getattr(self.plugin.router, "counters", {})),
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
        content = self.plugin._sanitize_memory(str(data.get("content") or ""))
        if not content:
            return self._fail("记忆内容不能为空")
        if self.plugin._looks_like_injection(content):
            return self._fail("内容含指令式语句（如「忽略以上」「你现在是」），"
                              "记忆会被逐轮注入，不能这样写")
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
            group_id=str(data.get("group_id") or ""),
            visibility=str(data.get("visibility") or ""),
        )
        return json_response({"id": mid})

    async def memory_visibility(self):
        data = await self._body()
        try:
            mid = int(data.get("id"))
        except (TypeError, ValueError):
            return self._fail("缺少记忆 id")
        vis = str(data.get("visibility") or "").strip().lower()
        if vis not in ("private", "group", "public"):
            return self._fail("visibility 只能是 private/group/public")
        ok = self.plugin.db.set_visibility(mid, vis)
        if not ok:
            return self._fail("记忆不存在或已失效", 404)
        self.plugin._resident_cache.clear()
        return json_response({"id": mid, "visibility": vis})

    async def memory_update(self):
        data = await self._body()
        try:
            mid = int(data.get("id"))
        except (TypeError, ValueError):
            return self._fail("缺少记忆 id")
        content = data.get("content")
        kind = data.get("kind")
        if content is not None:
            content = self.plugin._sanitize_memory(str(content))
            if not content:
                return self._fail("记忆内容不能为空")
            if self.plugin._looks_like_injection(content):
                return self._fail("内容含指令式语句（如「忽略以上」「你现在是」），"
                                  "记忆会被逐轮注入，不能这样写")
        if kind is not None and str(kind) not in MEMORY_KINDS:
            return self._fail("未知的记忆类型")
        ok = self.plugin.db.update_memory(mid, content, None if kind is None else str(kind))
        if not ok:
            return self._fail("记忆不存在或没有改动", 404)
        return json_response({"id": mid, "record": self.plugin.db.memory_by_id(mid)})

    async def memory_pin(self):
        data = await self._body()
        try:
            mid = int(data.get("id"))
        except (TypeError, ValueError):
            return self._fail("缺少记忆 id")
        pinned = bool(data.get("pinned"))
        ok = self.plugin.db.set_pinned(mid, pinned)
        if not ok:
            return self._fail("记忆不存在或已失效", 404)
        self.plugin._resident_cache.clear()
        return json_response({"id": mid, "pinned": pinned})

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
        content = self.plugin._sanitize_memory(str(data.get("content") or ""))
        kind = str(data.get("kind") or "").strip()
        if kind and kind not in MEMORY_KINDS:
            return self._fail("未知的记忆类型")
        if not isinstance(raw_ids, list) or len(raw_ids) < 2:
            return self._fail("至少选择两条记忆")
        try:
            ids = [int(i) for i in raw_ids]
        except (TypeError, ValueError):
            return self._fail("id 必须是数字")
        if not content:
            return self._fail("合并后的内容不能为空")
        if self.plugin._looks_like_injection(content):
            return self._fail("合并后的内容含指令式语句，未写入")
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

    def _owner_label(self, row: dict, names: dict[str, str]) -> str:
        scope = str(row.get("scope") or "private")
        if scope == "global":
            return "公共（所有人可读）"
        if scope == "group":
            gid = str(row.get("group_id") or "")
            return f"群共享 {gid[-6:]}" if gid else "群共享"
        oid = str(row.get("owner_id") or "")
        nick = names.get(oid, "")
        return f"{nick}（{oid[-6:]}）" if nick else (f"用户 {oid[-6:]}" if oid else "未归属")

    async def skills(self):
        db = self.plugin.db
        hub = self.plugin.hub
        names = db.owner_names()
        records = []
        for row in db.all_skills():  # 管理员视角：所有人 + 公共的都列出来
            row["source"] = "personal"
            row["readonly"] = False
            row["owner_label"] = self._owner_label(row, names)
            records.append(row)
        # 系统技能：AstrBot 本体装的，插件只读
        system = []
        for s in hub.system_skills(force=True):
            system.append({
                "id": f"sys:{s['slug']}",
                "name": s["name"],
                "trigger": s.get("trigger") or "",
                "content": s.get("content") or "",
                "path": s.get("path") or "",
                "has_scripts": bool(s.get("has_scripts")),
                "source": "system",
                "readonly": True,
                "owner_label": "AstrBot 本体（全局只读）",
            })
        owners = []
        for r in db.skill_owners():
            owners.append({
                "owner_id": r.get("owner_id") or "",
                "scope": r.get("scope") or "private",
                "group_id": r.get("group_id") or "",
                "count": int(r.get("c") or 0),
                "label": self._owner_label(r, names),
            })
        return json_response({
            "records": records,
            "system": system,
            "owners": owners,
            "system_root": hub.system_root,
            "max_direct": self._cfg("skill_consolidate_max", 40),
        })

    async def skill_save(self):
        data = await self._body()
        db = self.plugin.db
        name = str(data.get("name") or "").strip()
        old_name = str(data.get("old_name") or "").strip()
        trigger = str(data.get("trigger") or "").strip()
        content = str(data.get("content") or "").strip()
        owner_id = str(data.get("owner_id") or "").strip()
        group_id = str(data.get("group_id") or "").strip()
        scope = str(data.get("scope") or ("private" if owner_id else "global")).strip().lower()
        if scope not in ("private", "group", "global"):
            return self._fail("scope 只能是 private / group / global")
        if not name:
            return self._fail("技能名不能为空")
        if len(name) > 64 or any(ch.isspace() for ch in name):
            return self._fail("技能名最多 64 字符，且不能包含空格")
        if len(content) < 5:
            return self._fail("技能内容太短")

        skill_id = data.get("id")
        try:
            skill_id = int(skill_id) if skill_id not in (None, "") else None
        except (TypeError, ValueError):
            skill_id = None

        # 按 id 编辑：可以同时改归属（私人→群共享、转给别人）
        if skill_id:
            row = db.skill_by_id(skill_id)
            if not row:
                return self._fail("这条技能不存在，可能已被删除")
            ok = db.move_skill(
                skill_id, owner_id=owner_id, scope=scope, group_id=group_id,
                name=name, trigger=trigger, content=content[:8000],
            )
            if not ok:
                return self._fail(f"目标归属下已存在同名技能 {name}")
            return json_response({"id": skill_id, "name": name, "updated": True})

        if old_name and old_name != name and db.peek_skill(name, owner_id=owner_id,
                                                           group_id=group_id):
            return self._fail("已存在同名技能")
        created = db.upsert_skill(name, trigger, content[:8000], owner_id=owner_id,
                                  scope=scope, group_id=group_id, origin="manual")
        if old_name and old_name != name:
            db.delete_skill(old_name, owner_id=owner_id, group_id=group_id)
        return json_response({"name": name, "created": bool(created)})

    async def skill_delete(self):
        data = await self._body()
        db = self.plugin.db
        skill_id = data.get("id")
        if skill_id not in (None, ""):
            try:
                sid = int(skill_id)
            except (TypeError, ValueError):
                return self._fail("id 不合法")
            row = db.skill_by_id(sid)
            if not row:
                return self._fail("这条技能不存在")
            ok = db.delete_skill(row["name"], owner_id=row.get("owner_id") or "",
                                 group_id=row.get("group_id") or "")
            if not ok:  # 兜底：按 id 直接删
                with db._lock:
                    ok = db.conn.execute("DELETE FROM skills WHERE id = ?", (sid,)).rowcount > 0
                    db.conn.commit()
            return json_response({"deleted": bool(ok)})
        name = str(data.get("name") or "").strip()
        if not name:
            return self._fail("缺少技能名或 id")
        ok = db.delete_skill(name, owner_id=str(data.get("owner_id") or ""),
                             group_id=str(data.get("group_id") or ""))
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
                if key in HIDDEN_IN_SETTINGS:
                    continue
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
            for key, meta in schema.items()
            if key not in seen and key not in HIDDEN_IN_SETTINGS
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
                elif typ == "list":
                    # 列表型配置（降级链、可重试错误类别）：允许数组或逗号分隔字符串
                    if isinstance(raw, str):
                        items = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]
                    elif isinstance(raw, (list, tuple)):
                        items = [str(x).strip() for x in raw if str(x).strip()]
                    else:
                        raise ValueError("需要列表")
                    cleaned[key] = items
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
        if not self.plugin.router.has_model():
            return self._fail("本插件还没设置总结模型，请先在「模型」标签选择主模型")
        result = await self.plugin._summarize(force=True)
        return json_response({"result": result})

    async def action_consolidate(self):
        data = await self._body()
        dry = bool(data.get("dry_run"))
        if not self.plugin.router.has_model():
            return self._fail("本插件还没设置总结模型，请先在「模型」标签选择主模型")
        result = await self.plugin._consolidate_memories(dry_run=dry)
        if not dry:
            self.plugin._reset_memory_queue()
        return json_response({"result": result, "dry_run": dry})

    async def action_skill_consolidate(self):
        data = await self._body()
        dry = bool(data.get("dry_run"))
        if not self.plugin.router.has_model():
            return self._fail("本插件还没设置总结模型，请先在「模型」标签选择主模型")
        result = await self.plugin._consolidate_skills(dry_run=dry)
        if not dry:
            self.plugin.db.set_meta("skill_consolidate_pending", "0")
        return json_response({"result": result, "dry_run": dry})

    async def action_export(self):
        md = self.plugin.db.export_memories_markdown()
        if self.data_dir_fn is None:
            return self._fail("插件数据目录不可用", 500)
        path = safe_join(self.data_dir_fn(), "MEMORY.md")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
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

    # ---------- 模型路由 ----------

    # 可重试错误类别的固定顺序（页面上按此顺序展示，便于对齐）
    RETRY_KINDS = ("timeout", "conn", "rate", "server", "empty", "auth", "ctx", "other")

    # 数值型容错参数：key -> (下限, 上限, 是否浮点)
    POLICY_BOUNDS = {
        "llm_max_retries": (0, 9, False),
        "llm_timeout": (5, 600, True),
        "llm_backoff": (0, 30, True),
        "llm_total_attempts": (1, 30, False),
        "llm_cooldown_seconds": (0, 3600, False),
        "llm_max_prompt_chars": (2000, 200000, False),
    }

    def _astrbot_chain(self) -> dict:
        """读 AstrBot 主体「对话模型 / 降级模型」配置，仅供页面「沿用」按钮参考。

        插件运行时的模型链不读这里：主体改对话模型不会影响本插件，
        只有用户主动点「沿用 AstrBot 对话配置」才会把这份值写进插件配置。
        """
        getter = getattr(self.plugin, "_astrbot_model_cfg", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return {"primary": "", "fallbacks": [], "source": ""}
        try:
            acm = self.plugin.context.astrbot_config_mgr
            cfg = acm.confs["default"]
            ar = cfg.get("agent_runner", {}) or {}
            if ar.get("runner_type") != "local":
                return {"primary": "", "fallbacks": [], "source": "非 local 运行器，无此配置"}
            model = (ar.get("config") or {}).get("model") or {}
            return {
                "primary": str(model.get("provider_id") or ""),
                "fallbacks": [str(x) for x in (model.get("fallback_provider_ids") or [])],
                "max_retries": model.get("request_max_retries"),
                "source": "AstrBot 对话配置",
            }
        except Exception:
            return {"primary": "", "fallbacks": [], "source": ""}

    async def models(self):
        snap = self.plugin.router.snapshot()
        snap["astrbot"] = self._astrbot_chain()
        return json_response(snap)

    async def models_save(self):
        data = await self._body()
        router = self.plugin.router
        known = {p["id"]: p for p in router.all_providers()}

        primary = str(data.get("primary") or "").strip()
        if primary:
            if primary not in known:
                return self._fail(f"主模型 {primary} 不在 AstrBot 的模型列表里")
            if not known[primary]["loaded"]:
                return self._fail(f"主模型 {primary} 未启用，AstrBot 没有加载它")

        raw_fb = data.get("fallbacks")
        if raw_fb is None:
            raw_fb = router.configured_fallbacks()
        if isinstance(raw_fb, str):
            raw_fb = [x.strip() for x in raw_fb.replace("\n", ",").split(",")]
        if not isinstance(raw_fb, list):
            return self._fail("降级模型必须是列表")

        fallbacks: list[str] = []
        errors: list[str] = []
        for item in raw_fb:
            pid = str(item or "").strip()
            if not pid or pid == primary or pid in fallbacks:
                continue
            if pid not in known:
                errors.append(f"{pid} 不存在")
                continue
            if not known[pid]["loaded"]:
                errors.append(f"{pid} 未启用")
                continue
            fallbacks.append(pid)
        if errors:
            return self._fail("；".join(errors))
        if len(fallbacks) > 10:
            return self._fail("降级模型最多 10 个")

        cleaned: dict = {
            "llm_primary_provider": primary,
            "llm_fallback_providers": fallbacks,
        }

        policy = data.get("policy")
        if policy is not None:
            if not isinstance(policy, dict):
                return self._fail("policy 需要对象")
            errs: list[str] = []
            for key, (low, high, is_float) in self.POLICY_BOUNDS.items():
                if key not in policy:
                    continue
                try:
                    val = float(policy[key])
                except (TypeError, ValueError):
                    errs.append(f"{key} 需要数字")
                    continue
                if val < low or val > high:
                    errs.append(f"{key} 需在 {low}~{high} 之间")
                    continue
                cleaned[key] = val if is_float else int(val)
            if "retry_on" in policy:
                raw = policy["retry_on"]
                if isinstance(raw, str):
                    raw = [x.strip() for x in raw.split(",")]
                if not isinstance(raw, (list, tuple)):
                    errs.append("retry_on 需要数组")
                else:
                    picked = {str(x) for x in raw}
                    cleaned["llm_retry_on"] = [k for k in self.RETRY_KINDS if k in picked]
            if errs:
                return self._fail("；".join(errs))

        if self.plugin.config is None:
            return self._fail("插件没有拿到配置对象", 500)
        try:
            self.plugin.config.update(cleaned)
            await self.plugin.config.save_config_async()
        except Exception:
            logger.error(f"[hermes_core] 保存模型配置失败:\n{traceback.format_exc()}")
            return self._fail("写入配置文件失败，请查看后端日志", 500)

        # 配置刚改过，之前的失败冷却很可能已经过时，清掉让新配置立刻生效
        router.clear_cooldowns()
        return json_response({"saved": cleaned,
                              "chain": [e["id"] for e in router.effective_chain(True)]})

    async def models_test(self):
        data = await self._body()
        ids = data.get("ids")
        if ids is None:
            one = str(data.get("id") or "").strip()
            ids = [one] if one else []
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            return self._fail("ids 需要数组")
        ids = [str(x).strip() for x in ids if str(x).strip()]
        if not ids:
            # 不指定就测整条链，按实际会走的顺序
            ids = [e["id"] for e in self.plugin.router.effective_chain(ignore_cooldown=True)]
        if not ids:
            return self._fail("没有可测试的模型")
        if len(ids) > 8:
            return self._fail("一次最多测试 8 个模型")
        results = []
        for pid in ids:
            try:
                results.append(await self.plugin.router.test(pid))
            except Exception:
                logger.error(f"[hermes_core] 测试模型 {pid} 异常:\n{traceback.format_exc(limit=2)}")
                results.append({"id": pid, "ok": False, "kind": "other",
                                "error": traceback.format_exc(limit=1)})
        return json_response({"results": results})

    async def models_logs_clear(self):
        self.plugin.router.clear_logs()
        self.plugin.router.clear_cooldowns()
        return json_response({"ok": True})
