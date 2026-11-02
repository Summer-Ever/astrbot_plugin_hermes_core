"""Hermes 风格存储层：SQLite + WAL + FTS5 全文检索。

对标 hermes-agent 的 session store：
- messages: 全部对话（用户/机器人），FTS5 trigram 索引支持中文检索
- memories: 长期记忆条目（fact/preference/profile/summary）
- skills:   Markdown 技能库（name/trigger/content）
- meta:     游标等元信息（如上次总结到的消息 id）
"""

import json
import os
import sqlite3
import threading
import time


class HermesDB:
    def __init__(self, data_dir: str, db_name: str = "hermes.db"):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, db_name)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._fts_enabled = self._init_schema()

    # ---------- schema ----------

    def _init_schema(self) -> bool:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                session_id TEXT NOT NULL,
                platform TEXT DEFAULT '',
                sender_id TEXT DEFAULT '',
                sender_name TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                scope TEXT DEFAULT 'global',
                content TEXT NOT NULL,
                source TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS skills(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                trigger TEXT DEFAULT '',
                content TEXT NOT NULL,
                created_ts REAL NOT NULL,
                use_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        fts_ok = True
        try:
            c.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5("
                "content, session_id UNINDEXED, tokenize='trigram')"
            )
        except sqlite3.OperationalError:
            fts_ok = False
        # v0.2.0 迁移：记忆多用户隔离（老库自动补列）
        for ddl in (
            "ALTER TABLE memories ADD COLUMN owner_id TEXT DEFAULT ''",
            "ALTER TABLE memories ADD COLUMN session_id TEXT DEFAULT ''",
            # v0.3.0 迁移：用平台+群/用户原生 ID 作稳定键，避免 umo 随 unique_session 漂移
            "ALTER TABLE memories ADD COLUMN platform TEXT DEFAULT ''",
            "ALTER TABLE memories ADD COLUMN group_id TEXT DEFAULT ''",
        ):
            try:
                c.execute(ddl)
            except sqlite3.OperationalError:
                pass
        # v0.3.0 迁移：消息也记录群原生 ID，供总结器写入群级记忆时定位
        try:
            c.execute("ALTER TABLE messages ADD COLUMN group_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        # v0.3.0：身份别名表（把同一人的多个 openid/sid 合并成一个主体）
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS identities(
                alias_id TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL,
                platform TEXT DEFAULT '',
                ts REAL NOT NULL
            );
            """
        )
        c.commit()
        return fts_ok

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    # ---------- messages ----------

    def add_message(self, session_id: str, role: str, content: str,
                    platform: str = "", sender_id: str = "", sender_name: str = "",
                    group_id: str = "") -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages(ts, session_id, platform, sender_id, sender_name,"
                " role, content, group_id) VALUES(?,?,?,?,?,?,?,?)",
                (time.time(), session_id, platform, sender_id, sender_name, role,
                 content, group_id or ""),
            )
            if self._fts_enabled:
                self.conn.execute(
                    "INSERT INTO messages_fts(rowid, content, session_id) VALUES(?,?,?)",
                    (cur.lastrowid, content, session_id),
                )
            self.conn.commit()
            return cur.lastrowid

    def search_messages(self, query: str, limit: int = 5,
                        session_id: str | None = None,
                        sender_id: str | None = None) -> list[dict]:
        """优先 FTS5 trigram 匹配，失败/无结果时退回 LIKE。可按会话和发言人过滤。"""
        extra_sql = ""
        extra_args: list = []
        if session_id is not None:
            extra_sql += " AND m.session_id = ?"
            extra_args.append(session_id)
        if sender_id is not None:
            extra_sql += " AND m.sender_id = ?"
            extra_args.append(sender_id)
        if self._fts_enabled:
            try:
                q = query.strip().replace('"', ' ')
                if len(q) >= 3:
                    rows = self.conn.execute(
                        "SELECT m.* FROM messages_fts f JOIN messages m ON m.id = f.rowid"
                        " WHERE messages_fts MATCH ?" + extra_sql +
                        " ORDER BY m.id DESC LIMIT ?",
                        (f'"{q}"', *extra_args, limit),
                    ).fetchall()
                    if rows:
                        return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{query.strip()[:100]}%"
        rows = self.conn.execute(
            "SELECT * FROM messages m WHERE m.content LIKE ?" + extra_sql +
            " ORDER BY m.id DESC LIMIT ?",
            (like, *extra_args, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def messages_since(self, since_id: int, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE id > ? ORDER BY id ASC LIMIT ?",
            (since_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        m = self.conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        mem = self.conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        sk = self.conn.execute("SELECT COUNT(*) c FROM skills").fetchone()["c"]
        s = self.conn.execute("SELECT COUNT(DISTINCT session_id) c FROM messages").fetchone()["c"]
        return {"messages": m, "memories": mem, "skills": sk, "sessions": s}

    # ---------- memories ----------

    def add_memory(self, kind: str, content: str, scope: str = "global",
                   source: str = "", owner_id: str = "", session_id: str = "",
                   platform: str = "", group_id: str = "") -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories(ts, kind, scope, content, source, owner_id,"
                " session_id, platform, group_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (time.time(), kind, scope, content, source, owner_id or "",
                 session_id or "", platform or "", group_id or ""),
            )
            self.conn.commit()
            return cur.lastrowid

    def search_memories(self, query: str, limit: int = 5,
                        owner_id: str | None = None,
                        session_id: str | None = None,
                        platform: str | None = None,
                        group_id: str | None = None) -> list[dict]:
        """记忆检索，支持三级作用域隔离（优先级：用户 > 群 > 全局）。

        群公共记忆的键用 (platform, group_id) 而不是 session_id(umo)，
        因此开启/关闭 unique_session 不会让既有群记忆失联；session_id 仅作老数据兼容。
        """
        tiers: list[str] = []
        args: list = []
        order = ""
        order_args: list = []
        group_key_sql = ""
        group_key_args: list = []
        if group_id is not None:
            group_key_sql = "((group_id = ? AND platform = ?)"
            group_key_args = [group_id, platform or ""]
            if session_id is not None:
                group_key_sql += " OR (session_id = ? AND group_id = ''))"
                group_key_args.append(session_id)
            else:
                group_key_sql += ")"
        global_sql = "owner_id = '' AND session_id = '' AND group_id = ''"

        if owner_id is not None and group_id is not None:
            tiers.append("owner_id = ?")
            args.append(owner_id)
            if group_key_sql:
                tiers.append(group_key_sql)
                args += group_key_args
            tiers.append(global_sql)
            order = ("ORDER BY CASE WHEN owner_id = ? THEN 0"
                     " WHEN owner_id = '' AND group_id != '' THEN 1 ELSE 2 END, id DESC")
            order_args = [owner_id]
        elif owner_id is not None:
            tiers.append("owner_id = ?")
            args.append(owner_id)
            tiers.append(global_sql)
            order = "ORDER BY CASE WHEN owner_id = ? THEN 0 ELSE 1 END, id DESC"
            order_args = [owner_id]
        elif group_id is not None:
            if group_key_sql:
                tiers.append(group_key_sql)
                args += group_key_args
            tiers.append(global_sql)
            order = "ORDER BY id DESC"
        else:
            order = "ORDER BY id DESC"

        like = f"%{query.strip()[:100]}%"
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE content LIKE ?"
            + (" AND (" + " OR ".join(tiers) + ")" if tiers else "")
            + f" {order} LIMIT ?",
            (like, *args, *order_args, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 身份别名 ----------

    def add_alias(self, alias_id: str, canonical_id: str, platform: str = "") -> bool:
        """把 alias 指向 canonical，返回是否新建。"""
        with self._lock:
            exist = self.conn.execute(
                "SELECT canonical_id FROM identities WHERE alias_id = ?", (alias_id,)
            ).fetchone()
            if exist:
                self.conn.execute(
                    "UPDATE identities SET canonical_id = ?, platform = ? WHERE alias_id = ?",
                    (canonical_id, platform, alias_id),
                )
                created = False
            else:
                self.conn.execute(
                    "INSERT INTO identities(alias_id, canonical_id, platform, ts) VALUES(?,?,?,?)",
                    (alias_id, canonical_id, platform, time.time()),
                )
                created = True
            self.conn.commit()
            return created

    def resolve_owner(self, raw_id: str, platform: str = "") -> str:
        """沿别名链解析出规范 ID（最多 5 跳，防环）。未登记则原样返回。"""
        cur = raw_id
        for _ in range(5):
            row = self.conn.execute(
                "SELECT canonical_id FROM identities WHERE alias_id = ?", (cur,)
            ).fetchone()
            if not row or not row["canonical_id"] or row["canonical_id"] == cur:
                break
            cur = row["canonical_id"]
        return cur

    def list_aliases(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM identities ORDER BY ts DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_alias(self, alias_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM identities WHERE alias_id = ?", (alias_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def all_memories(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # ---------- skills ----------

    def upsert_skill(self, name: str, trigger: str, content: str) -> bool:
        """存在同名技能则更新内容，返回是否新建。"""
        with self._lock:
            exist = self.conn.execute(
                "SELECT id FROM skills WHERE name = ?", (name,)
            ).fetchone()
            if exist:
                self.conn.execute(
                    "UPDATE skills SET trigger = ?, content = ? WHERE id = ?",
                    (trigger, content, exist["id"]),
                )
                created = False
            else:
                self.conn.execute(
                    "INSERT INTO skills(name, trigger, content, created_ts) VALUES(?,?,?,?)",
                    (name, trigger, content, time.time()),
                )
                created = True
            self.conn.commit()
            return created

    def list_skills(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, trigger, use_count, created_ts FROM skills ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_skill(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM skills WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return None
        self.conn.execute(
            "UPDATE skills SET use_count = use_count + 1 WHERE id = ?", (row["id"],)
        )
        self.conn.commit()
        return dict(row)

    # ---------- meta ----------

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?,?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self.conn.commit()

    def export_memories_markdown(self) -> str:
        """导出 MEMORY.md 风格的长期记忆文件内容。"""
        lines = ["# MEMORY.md", ""]
        for m in self.all_memories(limit=500):
            lines.append(f"- [{m['kind']}] {m['content']}")
        return "\n".join(lines)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
