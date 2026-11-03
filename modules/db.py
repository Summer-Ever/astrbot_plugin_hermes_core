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


# 技能表：唯一键是 (owner_id, name)，同一个人不能有两条同名技能，
# 但不同的人可以各自持有同名技能（比如两个人都改过 computer-power）。
# 注意：这里只建表，索引要等迁移完（老表补上 owner_id 列）再建，
# 否则在老库上会因缺列直接抛错。
SKILLS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS skills(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_id TEXT NOT NULL DEFAULT '',
        scope TEXT NOT NULL DEFAULT 'private',
        group_id TEXT DEFAULT '',
        name TEXT NOT NULL,
        trigger TEXT DEFAULT '',
        content TEXT NOT NULL,
        created_ts REAL NOT NULL,
        updated_ts REAL DEFAULT 0,
        use_count INTEGER DEFAULT 0,
        merged_into INTEGER DEFAULT 0,
        origin TEXT DEFAULT 'hermes'
    );
"""

SKILLS_INDEX_SQL = """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_key ON skills(owner_id, name);
    CREATE INDEX IF NOT EXISTS idx_skill_scope ON skills(scope, group_id);
"""

# 技能可见性：自己的 private + 本群 group + 全体 global。
# owner_id 为空的存量记录一律按 global 处理，避免迁移后凭空消失。
SKILL_VISIBLE_SQL = (
    "merged_into = 0 AND ("
    " (scope = 'private' AND owner_id = ? AND owner_id != '')"
    " OR (scope = 'group' AND ? != '' AND group_id = ?)"
    " OR scope = 'global'"
    ")"
)


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
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        # v0.9.0：技能按身份隔离。此前 name 是全局唯一，谁学的大家都能改，
        # 和记忆的按人隔离对不上；现在唯一键是 (owner_id, name)，同一个技能名
        # 允许每个人各有一份。
        c.executescript(SKILLS_TABLE_SQL)
        self._migrate_skills(c)
        c.executescript(SKILLS_INDEX_SQL)
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
            # v0.6.0 迁移：记忆巩固。合并后旧条不删除，只把 merged_into 指向新条，
            # 检索时排除。这样误合并可以回滚，也不丢历史。
            "ALTER TABLE memories ADD COLUMN merged_into INTEGER DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN updated_ts REAL DEFAULT 0",
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

    def _migrate_skills(self, c) -> None:
        """把老库的全局技能表重建成按身份隔离的表。

        老表是 `name TEXT UNIQUE`，SQLite 改不了约束，只能重建。
        存量技能没有归属信息，统一归给「记忆最多的那个人」——通常就是一直在用
        这个机器人的本人；如果库是空的，就退化成 global 共享。
        """
        cols = {r[1] for r in c.execute("PRAGMA table_info(skills)")}
        if not cols or "owner_id" in cols:
            return
        try:
            row = c.execute(
                "SELECT owner_id FROM memories WHERE owner_id != ''"
                " GROUP BY owner_id ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
            owner = str(row[0] or "") if row else ""
        except sqlite3.OperationalError:
            owner = ""
        scope = "private" if owner else "global"
        try:
            c.execute("ALTER TABLE skills RENAME TO skills_legacy_v080")
        except sqlite3.OperationalError:
            return
        c.executescript(SKILLS_TABLE_SQL)
        c.execute(
            "INSERT INTO skills(owner_id, scope, group_id, name, trigger, content,"
            " created_ts, updated_ts, use_count, merged_into, origin)"
            " SELECT ?, ?, '', name, trigger, content, created_ts, created_ts,"
            " COALESCE(use_count, 0), 0, 'hermes' FROM skills_legacy_v080",
            (owner, scope),
        )
        c.execute("DROP TABLE skills_legacy_v080")
        c.execute(
            "INSERT INTO meta(key, value) VALUES('skills_migrated_v090', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (f"{owner or '(global)'}",),
        )
        c.commit()

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

    @staticmethod
    def _msg_where(session_id: str | None, sender_id: str | None) -> tuple[str, list]:
        conds: list[str] = []
        args: list = []
        if session_id is not None:
            conds.append("session_id = ?")
            args.append(session_id)
        if sender_id is not None:
            conds.append("sender_id = ?")
            args.append(sender_id)
        return ((" WHERE " + " AND ".join(conds)) if conds else ""), args

    def count_messages(self, session_id: str | None = None,
                       sender_id: str | None = None) -> int:
        where, args = self._msg_where(session_id, sender_id)
        row = self.conn.execute(f"SELECT COUNT(*) FROM messages{where}", args).fetchone()
        return int(row[0]) if row else 0

    def clear_messages(self, session_id: str | None = None,
                       sender_id: str | None = None) -> int:
        """删除对话消息及对应 FTS 索引；memories / skills / identities 不受影响。"""
        where, args = self._msg_where(session_id, sender_id)
        with self._lock:
            ids = [r[0] for r in self.conn.execute(
                f"SELECT id FROM messages{where}", args).fetchall()]
            if not ids:
                return 0
            if self._fts_enabled:
                self.conn.executemany(
                    "DELETE FROM messages_fts WHERE rowid = ?", [(i,) for i in ids])
            self.conn.execute(f"DELETE FROM messages{where}", args)
            self.conn.commit()
            return len(ids)

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

    # ---------- 按会话的总结水位 ----------

    def get_watermark(self, session_id: str, default: int = 0) -> int:
        try:
            return int(self.get_meta(f"summarized:{session_id}", str(default)) or default)
        except (TypeError, ValueError):
            return default

    def set_watermark(self, session_id: str, msg_id: int) -> None:
        self.set_meta(f"summarized:{session_id}", str(int(msg_id)))

    def all_session_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT session_id FROM messages").fetchall()
        return [r[0] for r in rows]

    def max_message_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()
        return int(row[0]) if row else 0

    def unsummarized_by_session(self) -> list[dict]:
        """每个会话待总结的消息统计（各自独立水位，互不影响）。"""
        out: list[dict] = []
        for sid in self.all_session_ids():
            wm = self.get_watermark(sid)
            row = self.conn.execute(
                "SELECT COUNT(*) c, MIN(ts) o, MAX(id) m FROM messages"
                " WHERE session_id = ? AND id > ?",
                (sid, wm),
            ).fetchone()
            if row and row["c"]:
                out.append({"session_id": sid, "count": int(row["c"]),
                            "oldest": row["o"], "max_id": row["m"], "watermark": wm})
        return out

    def session_batch(self, session_id: str, watermark: int,
                      limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND id > ?"
            " ORDER BY id ASC LIMIT ?",
            (session_id, watermark, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        m = self.conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        mem = self.conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE merged_into = 0"
        ).fetchone()["c"]
        merged = self.conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE merged_into != 0"
        ).fetchone()["c"]
        sk = self.conn.execute("SELECT COUNT(*) c FROM skills").fetchone()["c"]
        s = self.conn.execute("SELECT COUNT(DISTINCT session_id) c FROM messages").fetchone()["c"]
        return {"messages": m, "memories": mem, "merged": merged,
                "skills": sk, "sessions": s}

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

    def all_active_memories(self, limit: int = 1000) -> list[dict]:
        """所有未合并的记忆，按 id 升序（即时间顺序），供巩固/去重使用。"""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE merged_into = 0 ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def max_memory_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) m FROM memories").fetchone()
        return int(row["m"] or 0)

    def memory_by_id(self, memory_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def merge_memories(self, ids: list[int], content: str, kind: str = "") -> int:
        """把一组记忆合成一条新的，旧条只标记 merged_into 指向新条（不物理删除）。

        保留旧条是为了可回滚：误合并时调 unmerge() 就能还原。
        作用域字段全部继承组内最早的那条，绝不跨用户/跨群合并。
        """
        ids = [int(i) for i in ids if i]
        if not ids:
            return 0
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE id IN ({marks})", ids
            ).fetchall()
            if not rows:
                return 0
            rows = sorted(rows, key=lambda r: r["id"])
            base = rows[0]
            now = time.time()
            cur = self.conn.execute(
                "INSERT INTO memories(ts, kind, scope, content, source, owner_id,"
                " session_id, platform, group_id, merged_into, updated_ts)"
                " VALUES(?,?,?,?,?,?,?,?,?,0,?)",
                (now, kind or base["kind"], base["scope"], content, "consolidate",
                 base["owner_id"] or "", base["session_id"] or "",
                 base["platform"] or "", base["group_id"] or "", now),
            )
            new_id = cur.lastrowid
            self.conn.execute(
                f"UPDATE memories SET merged_into = ?, updated_ts = ? WHERE id IN ({marks})",
                (new_id, now, *ids),
            )
            self.conn.commit()
            return new_id

    def unmerge(self, new_id: int) -> int:
        """撤销一次合并：恢复旧条，删除合并产生的条目。返回恢复条数。"""
        with self._lock:
            n = self.conn.execute(
                "UPDATE memories SET merged_into = 0 WHERE merged_into = ?", (new_id,)
            ).rowcount
            self.conn.execute("DELETE FROM memories WHERE id = ?", (new_id,))
            self.conn.commit()
            return n

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
            f"SELECT * FROM memories WHERE content LIKE ? AND merged_into = 0"
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
            "SELECT * FROM memories WHERE merged_into = 0 ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_memories(self, q: str = "", scope: str = "", kind: str = "",
                       owner_id: str = "", include_merged: bool = False,
                       limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """供 WebUI 列表使用：关键词/作用域/类型筛选 + 分页，返回 (rows, total)。"""
        where: list[str] = []
        args: list = []
        if not include_merged:
            where.append("merged_into = 0")
        if scope:
            where.append("scope = ?")
            args.append(scope)
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if owner_id:
            where.append("(owner_id LIKE ? OR session_id LIKE ? OR group_id LIKE ?)")
            like = f"%{owner_id[:100]}%"
            args += [like, like, like]
        if q.strip():
            like = f"%{q.strip()[:100]}%"
            where.append("(content LIKE ? OR owner_id LIKE ? OR group_id LIKE ?"
                         " OR session_id LIKE ?)")
            args += [like, like, like, like]
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        total = self.conn.execute(
            f"SELECT COUNT(*) c FROM memories{clause}", tuple(args)
        ).fetchone()["c"]
        rows = self.conn.execute(
            f"SELECT * FROM memories{clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*args, max(1, min(int(limit), 200)), max(0, int(offset))),
        ).fetchall()
        return [dict(r) for r in rows], int(total)

    def update_memory(self, memory_id: int, content: str | None = None,
                      kind: str | None = None) -> bool:
        sets: list[str] = []
        args: list = []
        if content is not None:
            sets.append("content = ?")
            args.append(content.strip())
        if kind is not None:
            sets.append("kind = ?")
            args.append(kind.strip())
        if not sets:
            return False
        sets.append("updated_ts = ?")
        args.append(time.time())
        args.append(int(memory_id))
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", tuple(args)
            )
            self.conn.commit()
            return cur.rowcount > 0

    def memory_kind_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) c FROM memories WHERE merged_into = 0 GROUP BY kind"
        ).fetchall()
        return {r["kind"] or "fact": int(r["c"]) for r in rows}

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # ---------- skills ----------
    #
    # 技能和记忆一样按人分：private 只有本人可见，group 同群可见，global 全体可见。
    # 可见性 = 自己的 private ∪ 本群 group ∪ global。

    @staticmethod
    def _norm_scope(owner_id: str, scope: str | None,
                    group_id: str = "") -> tuple[str, str]:
        """把 owner + scope 归一成合法组合，返回 (owner_id, scope)。"""
        want = (scope or "").strip().lower()
        if want == "global":
            return "", "global"
        if want == "group" and group_id:
            return "", "group"
        if owner_id:
            return owner_id, "private"
        return ("", "group") if group_id else ("", "global")

    def upsert_skill(self, name: str, trigger: str, content: str, *,
                     owner_id: str = "", scope: str | None = None,
                     group_id: str = "", origin: str = "hermes") -> bool:
        """写入某个人（或某个作用域）的技能，返回是否新建。"""
        oid, scp = self._norm_scope(owner_id, scope, group_id)
        gid = group_id if scp == "group" else ""
        now = time.time()
        with self._lock:
            exist = self.conn.execute(
                "SELECT id FROM skills WHERE owner_id = ? AND name = ?", (oid, name)
            ).fetchone()
            if exist:
                self.conn.execute(
                    "UPDATE skills SET trigger = ?, content = ?, updated_ts = ?,"
                    " scope = ?, group_id = ? WHERE id = ?",
                    (trigger, content, now, scp, gid, exist["id"]),
                )
                created = False
            else:
                self.conn.execute(
                    "INSERT INTO skills(owner_id, scope, group_id, name, trigger, content,"
                    " created_ts, updated_ts, origin) VALUES(?,?,?,?,?,?,?,?,?)",
                    (oid, scp, gid, name, trigger, content, now, now, origin),
                )
                created = True
            self.conn.commit()
            return created

    def list_skills(self, owner_id: str = "", group_id: str = "") -> list[dict]:
        """列出发言人当前能看到的技能，自己的排最前。"""
        rows = self.conn.execute(
            "SELECT id, owner_id, scope, group_id, name, trigger, use_count,"
            " created_ts, updated_ts, origin FROM skills"
            f" WHERE {SKILL_VISIBLE_SQL}"
            " ORDER BY CASE scope WHEN 'private' THEN 0 WHEN 'group' THEN 1 ELSE 2 END, id",
            (owner_id, group_id, group_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_skills(self, owner_id: str | None = None,
                   group_id: str | None = None) -> list[dict]:
        """带正文的完整技能列表，供 WebUI 展示/编辑。

        owner_id 传 None 表示不过滤（管理员视角看全部人）。
        """
        conds: list[str] = []
        args: list = []
        if owner_id is not None:
            conds.append("owner_id = ?")
            args.append(owner_id)
        if group_id is not None:
            conds.append("group_id = ?")
            args.append(group_id)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        rows = self.conn.execute(
            "SELECT * FROM skills" + where + " ORDER BY use_count DESC, id", args
        ).fetchall()
        return [dict(r) for r in rows]

    def _pick_skill(self, name: str, owner_id: str, group_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM skills WHERE name = ? AND " + SKILL_VISIBLE_SQL +
            " ORDER BY CASE scope WHEN 'private' THEN 0 WHEN 'group' THEN 1 ELSE 2 END,"
            " id DESC LIMIT 1",
            (name, owner_id, group_id, group_id),
        ).fetchone()

    def peek_skill(self, name: str, owner_id: str = "",
                   group_id: str = "") -> dict | None:
        """读取技能但不像 get_skill 那样增加使用计数（写入/合并时用）。"""
        row = self._pick_skill(name, owner_id, group_id)
        return dict(row) if row else None

    def delete_skill(self, name: str, owner_id: str = "",
                     group_id: str = "") -> bool:
        """只删当前身份能看到的那一条，不会误删别人的同名技能。"""
        row = self._pick_skill(name, owner_id, group_id)
        if not row:
            return False
        with self._lock:
            cur = self.conn.execute("DELETE FROM skills WHERE id = ?", (row["id"],))
            self.conn.commit()
            return cur.rowcount > 0

    def move_skill(self, skill_id: int, *, owner_id: str = "",
                   scope: str = "private", group_id: str = "",
                   name: str | None = None, trigger: str | None = None,
                   content: str | None = None) -> bool:
        """按 id 改技能的归属/内容（WebUI 编辑与「发给本群」都用它）。

        改名撞上同一归属域里已有的同名技能时返回 False，不写库。
        """
        oid, scp = self._norm_scope(owner_id, scope, group_id)
        try:
            with self._lock:
                self.conn.execute(
                    "UPDATE skills SET owner_id = ?, scope = ?, group_id = ?,"
                    " name = COALESCE(?, name), trigger = COALESCE(?, trigger),"
                    " content = COALESCE(?, content), updated_ts = ? WHERE id = ?",
                    (oid, scp, group_id if scp == "group" else "", name, trigger,
                     content, time.time(), skill_id),
                )
                self.conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def skill_by_id(self, skill_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return dict(row) if row else None

    def owner_names(self) -> dict[str, str]:
        """从历史消息里捞每位用户的昵称，供 WebUI 标注技能归属。"""
        try:
            rows = self.conn.execute(
                "SELECT sender_id, sender_name, MAX(ts) t FROM messages"
                " WHERE sender_id != '' AND sender_name NOT IN ('', '.', '。')"
                " GROUP BY sender_id ORDER BY t DESC"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {str(r["sender_id"]): str(r["sender_name"]) for r in rows}

    def replace_skills(self, old_names: list[str], name: str, trigger: str,
                       content: str, *, owner_id: str = "", scope: str | None = None,
                       group_id: str = "") -> None:
        """技能整合：写入合并后的技能，并删除同作用域内被并入的旧技能。

        只动同一个人的技能，绝不跨用户合并。
        """
        oid, scp = self._norm_scope(owner_id, scope, group_id)
        gid = group_id if scp == "group" else ""
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO skills(owner_id, scope, group_id, name, trigger, content,"
                " created_ts, updated_ts) VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(owner_id, name) DO UPDATE SET trigger = excluded.trigger,"
                " content = excluded.content, updated_ts = excluded.updated_ts",
                (oid, scp, gid, name, trigger, content, now, now),
            )
            for old in old_names:
                if old and old != name:
                    self.conn.execute(
                        "DELETE FROM skills WHERE owner_id = ? AND name = ?", (oid, old)
                    )
            self.conn.commit()

    def get_skill(self, name: str, owner_id: str = "",
                  group_id: str = "") -> dict | None:
        row = self._pick_skill(name, owner_id, group_id)
        if not row:
            return None
        self.conn.execute(
            "UPDATE skills SET use_count = use_count + 1 WHERE id = ?", (row["id"],)
        )
        self.conn.commit()
        return dict(row)

    def skill_owners(self) -> list[dict]:
        """按归属统计技能数量，供技能整理分组和 WebUI 筛选。"""
        rows = self.conn.execute(
            "SELECT owner_id, scope, group_id, COUNT(*) c FROM skills"
            " WHERE merged_into = 0 GROUP BY owner_id, scope, group_id"
            " ORDER BY c DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def skills_in(self, owner_id: str = "", scope: str = "private",
                  group_id: str = "") -> list[dict]:
        """取出某个归属域内的全部技能（整理合并只在这个域里做）。"""
        rows = self.conn.execute(
            "SELECT id, owner_id, scope, group_id, name, trigger, content, use_count"
            " FROM skills WHERE owner_id = ? AND scope = ? AND group_id = ?"
            " AND merged_into = 0 ORDER BY id",
            (owner_id, scope, group_id),
        ).fetchall()
        return [dict(r) for r in rows]

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
