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

try:  # 正常情况（包内导入）
    from . import recall as _recall
except ImportError:  # 插件被按文件路径加载时，退化为同名模块
    try:
        from modules import recall as _recall  # type: ignore
    except ImportError:  # 最后兜底：按路径加载，不污染 sys.path
        import importlib.util as _ilu

        _rc_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall.py")
        _rc_spec = _ilu.spec_from_file_location("astrbot_plugin_hermes_core_recall", _rc_file)
        _rc_mod = _ilu.module_from_spec(_rc_spec)
        _rc_spec.loader.exec_module(_rc_mod)
        _recall = _rc_mod

# 记忆类型白名单。模型偶尔会自创类型（历史上出现过 conclusion），
# 不校验的话会污染统计和 UI 筛选；未知类型统一归为 fact。
ALLOWED_KINDS = ("fact", "preference", "profile", "summary",
                 "event", "plan", "conclusion", "relation")
DEFAULT_KIND = "fact"

# 去重时的否定词哨兵。只用于「两边是否同极性」的比较，不做语义分析。
_NEG_WORDS = ("不", "没", "无", "别", "禁止", "取消", "勿", "无需")


def _ph(n: int) -> str:
    """生成 n 个逗号分隔的 SQL 占位符（`?,?,?`）。

    只由整数决定，不接受任何外部字符串，所以拼进 SQL 也不会引入注入面。
    集中在这里是为了避免各处手写 `",".join("?" * len(x))` 时数错个数。
    """
    n = max(1, int(n))
    return ",".join("?" * n)


def _where(conds: list[str]) -> str:
    """把若干固定条件片段拼成 WHERE 子句。

    ``conds`` 必须是本文件内硬编码的片段（形如 ``owner_id = ?``），
    值一律走占位符；不要把外部输入拼进来。
    """
    return (" WHERE " + " AND ".join(conds)) if conds else ""


def _neg_words(text: str) -> set[str]:    return {w for w in _NEG_WORDS if w in str(text or "")}


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
            # v0.10.0：检索加权需要「被召回次数」和「重要度」；last_hit_ts 供衰减判断
            "ALTER TABLE memories ADD COLUMN hits INTEGER DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN last_hit_ts REAL DEFAULT 0",
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
        # 检索/统计的支撑索引。memories 原来一个索引都没有，134 条时全表扫还能撑，
        # 上千条后每次注入都在拖慢事件循环。
        c.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_mem_active
                ON memories(merged_into, owner_id, group_id);
            CREATE INDEX IF NOT EXISTS idx_mem_kind
                ON memories(merged_into, kind, id);
            CREATE INDEX IF NOT EXISTS idx_msg_sender
                ON messages(session_id, sender_id, id);
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
        return _where(conds), args

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

    def prune_messages(self, days: int, batch: int = 2000) -> int:
        """TTL 保留期修剪：删除过期原始消息并同步 FTS，随后收缩 WAL。

        只动 messages（原始转写层，可再生）；memories / skills / identities
        是资产，不受影响。分批提交避免长事务阻塞在线写入。
        返回删除总条数。days <= 0 时直接返回 0（保留期关闭）。
        """
        if int(days) <= 0:
            return 0
        cutoff = time.time() - int(days) * 86400
        total = 0
        with self._lock:
            # 海象循环：每批取一批删一批；批次大小受 LIMIT 约束，不会长事务
            while (ids := [r[0] for r in self.conn.execute(
                    "SELECT id FROM messages WHERE ts < ? LIMIT ?",
                    (cutoff, int(batch)),
            ).fetchall()]):
                if self._fts_enabled:
                    self.conn.executemany(
                        "DELETE FROM messages_fts WHERE rowid = ?",
                        [(i,) for i in ids])
                self.conn.executemany(
                    "DELETE FROM messages WHERE id = ?", [(i,) for i in ids])
                self.conn.commit()
                total += len(ids)
                if len(ids) < int(batch):
                    break
            if total:
                try:
                    self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
        return total

    def recent_message_ids(self, session_id: str | None, sender_id: str | None,
                           n: int) -> list[int]:
        """取本会话最近 n 条消息的 id（与消息查询同口径），用于注入窗口去重。

        这批消息就是 AstrBot 会随本轮请求原样发出去的内容，命中它们的历史
        不需要再注入一份。
        """
        where, args = self._msg_where(session_id, sender_id)
        rows = self.conn.execute(
            f"SELECT id FROM messages{where} ORDER BY id DESC LIMIT ?",
            (*args, max(0, int(n))),
        ).fetchall()
        return [r["id"] for r in rows]

    def _fts_candidates(self, query: str, cap: int,
                        session_id: str | None,
                        sender_id: str | None) -> list[dict] | None:
        """用 FTS5 trigram 取「包含查询短语」的消息，作为检索快路径。

        返回 None 表示索引不可用（老库没建表 / SQLite 无 trigram），
        调用方据此回退全扫。命中不足时调用方同样回退，所以不会损失召回，
        只在短语确实出现过时省掉一次 1500 行的扫描。
        """
        if not self._fts_enabled:
            return None
        q = " ".join(str(query).split())
        if len(q) < 3:
            return None
        # trigram 下用短语查询，引号内的特殊字符需要转义
        phrase = '"' + q.replace('"', '""') + '"'
        # FTS5 要求 MATCH 左侧写表名，别名不可用；rank 只在 MATCH 时存在
        sql = ("SELECT m.* FROM messages_fts"
               " JOIN messages m ON m.id = messages_fts.rowid"
               " WHERE messages_fts MATCH ?")
        args: list = [phrase]
        if session_id is not None:
            sql += " AND m.session_id = ?"
            args.append(session_id)
        if sender_id is not None:
            sql += " AND m.sender_id = ?"
            args.append(sender_id)
        sql += " ORDER BY messages_fts.rank LIMIT ?"
        args.append(max(1, int(cap)))
        try:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]
        except sqlite3.OperationalError:
            return None

    def search_messages(self, query: str, limit: int = 5,
                        session_id: str | None = None,
                        sender_id: str | None = None,
                        scan: int = 1500) -> list[dict]:
        """按相关度检索历史消息（bigram + IDF），不再要求整句作为子串出现。

        候选集优先取 FTS 短语命中，命中不足时回退到最近 scan 条：
        越久远的对话越不可能被需要，同时也避免全表扫描。
        返回按 id 倒序（最新在前），与旧实现保持一致，调用方无需调整。
        """
        if len(_recall.norm(query)) < 4:
            # 两三个字的寒暄没有区分度，检索只会带回噪声
            return []
        cap = max(50, int(scan))
        docs = self._fts_candidates(query, cap, session_id, sender_id)
        if docs is None or len(docs) < max(24, int(limit) * 4):
            where, args = self._msg_where(session_id, sender_id)
            rows = self.conn.execute(
                f"SELECT * FROM messages{where} ORDER BY id DESC LIMIT ?",
                (*args, cap),
            ).fetchall()
            scanned = [dict(r) for r in rows]
            if docs:
                seen = {d["id"] for d in scanned}
                scanned.extend([d for d in docs if d["id"] not in seen])
            docs = scanned
        hits = _recall.rank_messages(query, docs, k=max(1, int(limit)))
        return sorted(hits, key=lambda x: -int(x["id"]))

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
                (time.time(), self._norm_kind(kind), scope, content, source,
                 owner_id or "", session_id or "", platform or "", group_id or ""),
            )
            self.conn.commit()
            return cur.lastrowid

    @staticmethod
    def _norm_kind(kind: str) -> str:
        k = str(kind or "").strip().lower()
        return k if k in ALLOWED_KINDS else DEFAULT_KIND

    def add_memory_smart(self, kind: str, content: str, scope: str = "global",
                         source: str = "", owner_id: str = "", session_id: str = "",
                         platform: str = "", group_id: str = "",
                         threshold: float = 0.70) -> tuple[int, bool]:
        """写入记忆前先判断「这条是不是已经说过了」。

        返回 (id, 是否新建)。同一作用域里已有高度相似的条目时，不再插一条新的，
        而是强化旧条：内容取更完整的那个，重要度上调。
        这样注入质量不会在「攒够 10 条再合并」的窗口期里一路下滑。
        """
        content = str(content or "").strip()
        if not content:
            return 0, False
        kind = self._norm_kind(kind)
        rows = self.conn.execute(
            "SELECT id, content, importance FROM memories WHERE merged_into = 0"
            " AND owner_id = ? AND group_id = ? AND session_id = ?"
            " ORDER BY id DESC LIMIT 300",
            (owner_id or "", group_id or "", session_id or ""),
        ).fetchall()
        best_id, best_sim = 0, 0.0
        for r in rows:
            s = _recall.jaccard(content, r["content"] or "")
            # 否定词不一致时绝不去重：「喜欢猫」和「不喜欢猫」字面相似度 0.71，
            # 但语义相反，合并会把偏好记反。
            if _neg_words(content) != _neg_words(r["content"] or ""):
                continue
            if s > best_sim:
                best_id, best_sim = int(r["id"]), s
        if best_id and best_sim >= threshold:
            old = self.conn.execute(
                "SELECT content, importance FROM memories WHERE id = ?", (best_id,)
            ).fetchone()
            merged = old["content"] if len(old["content"]) >= len(content) else content
            with self._lock:
                self.conn.execute(
                    "UPDATE memories SET content = ?, updated_ts = ?,"
                    " importance = MIN(1.0, COALESCE(importance, 0) + 0.15),"
                    " kind = CASE WHEN kind = 'fact' THEN ? ELSE kind END WHERE id = ?",
                    (merged, time.time(), kind, best_id),
                )
                self.conn.commit()
            return best_id, False
        return self.add_memory(kind, content, scope=scope, source=source,
                               owner_id=owner_id, session_id=session_id,
                               platform=platform, group_id=group_id), True

    def bump_hits(self, ids: list[int]) -> None:
        """注入/检索命中后累加访问计数，供后续排序加权。"""
        ids = [int(i) for i in ids if i]
        if not ids:
            return
        with self._lock:
            marks = _ph(len(ids))
            self.conn.execute(
                f"UPDATE memories SET hits = COALESCE(hits, 0) + 1,"
                f" last_hit_ts = ? WHERE id IN ({marks})",
                (time.time(), *ids),
            )
            self.conn.commit()

    def resident_notes(self, owner_id: str, limit: int = 6,
                       max_chars: int = 400,
                       kinds: tuple[str, ...] = ("profile", "preference")
                        ) -> list[dict]:
        """常驻「关于你」条目：不依赖检索命中，每次对话固定注入。

        Hermes 的做法是把 MEMORY.md / USER.md 整块加载——「机器人记得你」的
        基线不该建立在「这次的关键词恰好撞上」之上。这里只取稳定的画像与偏好
        （fact 是具体事实，量大且多为一次性信息，走检索层按需召回），
        按重要度与命中次数排序，最后按字符预算截断。
        """
        if not owner_id:
            return []
        marks = _ph(len(kinds))
        rows = self.conn.execute(
            f"SELECT id, kind, content, importance, hits FROM memories"
            f" WHERE merged_into = 0 AND owner_id = ? AND kind IN ({marks})"
            # 画像/偏好优先于一次性事实：常驻区块就那么点预算，先放「这个人是谁」
            f" ORDER BY CASE kind WHEN 'profile' THEN 0 WHEN 'preference' THEN 1 ELSE 2 END,"
            f" COALESCE(importance, 0) DESC, COALESCE(hits, 0) DESC, id DESC"
            f" LIMIT ?",
            (owner_id, *kinds, max(1, int(limit))),
        ).fetchall()
        out: list[dict] = []
        used = 0
        for r in rows:
            text = str(r["content"] or "").strip()
            if not text:
                continue
            if used + len(text) > max_chars and out:
                break
            out.append(dict(r))
            used += len(text) + 3
        return out

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
            marks = _ph(len(ids))
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

    def discard_memories(self, ids: list[int]) -> int:
        """把记忆标记为丢弃（merged_into = -1），不物理删除，可人工恢复。

        深度整理用：一次性任务指令、闲聊反馈、技能清单这类「不该进长期记忆」
        的条目无法与任何东西合并，只能整体退出活跃池。
        """
        ids = [int(i) for i in ids if i]
        if not ids:
            return 0
        with self._lock:
            marks = _ph(len(ids))
            cur = self.conn.execute(
                f"UPDATE memories SET merged_into = -1, updated_ts = ?"
                f" WHERE id IN ({marks}) AND merged_into = 0",
                (time.time(), *ids),
            )
            self.conn.commit()
            return cur.rowcount

    def search_memories(self, query: str, limit: int = 5,
                        owner_id: str | None = None,
                        session_id: str | None = None,
                        platform: str | None = None,
                        group_id: str | None = None,
                        scan: int = 1200) -> list[dict]:
        """记忆检索：作用域分层 + 本地相关性排序（bigram/IDF/时间衰减/去冗余）。

        群公共记忆的键用 (platform, group_id) 而不是 session_id(umo)，
        因此开启/关闭 unique_session 不会让既有群记忆失联；session_id 仅作老数据兼容。

        分层逐级补：用户私有 > 群共享 > 全局。分层后各自排序再串起来，
        比给不同层级乘系数更直观，也不会出现「全局的高分记忆把用户偏好挤出前 5」。
        """
        tiers: list[tuple[str, list]] = []
        if owner_id is None and group_id is None:
            tiers.append(("merged_into = 0", []))
        else:
            if owner_id is not None:
                tiers.append(("merged_into = 0 AND owner_id = ?", [owner_id]))
            if group_id is not None:
                cond = "merged_into = 0 AND ((group_id = ? AND platform = ?)"
                args: list = [group_id, platform or ""]
                if session_id is not None:
                    cond += " OR (session_id = ? AND group_id = ''))"
                    args.append(session_id)
                else:
                    cond += ")"
                tiers.append((cond, args))
            tiers.append(
                ("merged_into = 0 AND owner_id = '' AND session_id = ''"
                 " AND group_id = ''", [])
            )

        out: list[dict] = []
        for cond, args in tiers:
            if len(out) >= limit:
                break
            try:
                rows = self.conn.execute(
                    f"SELECT * FROM memories WHERE {cond} ORDER BY id DESC LIMIT ?",
                    (*args, max(200, int(scan))),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            docs = [dict(r) for r in rows]
            if not docs:
                continue
            out.extend(_recall.rank(query, docs, k=limit - len(out)))
        return out[:limit]

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
        where = _where(conds)
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

    def bump_meta_int(self, key: str, n: int = 1) -> int:
        """原子累加一个计数型 meta，用于长期指标（避免高频写入，调用方自行控频）。"""
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = CAST("
                "  CAST(meta.value AS INTEGER) + ? AS TEXT)",
                (key, str(int(n)), int(n)),
            )
            self.conn.commit()
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return int(row["value"]) if row else 0

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
