"""Hermes 风格存储层：SQLite + WAL + FTS5 全文检索。

对标 hermes-agent 的 session store：
- messages: 全部对话（用户/机器人），FTS5 trigram 索引支持中文检索
- memories: 长期记忆条目（fact/preference/profile/summary）
- skills:   Markdown 技能库（name/trigger/content）
- meta:     游标等元信息（如上次总结到的消息 id）
"""

import json
import os
import re
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

# 可互斥的「槽位」：同一槽位在常驻里只该有一个当前值。
# 目前只识别称呼类——它是最典型的「值会被用户直接改掉」的槽位，
# 且「兄弟」和「老爹」字面相似度为 0，字面去重完全抓不到。
_SLOT_PATTERNS = (
    ("address", re.compile(r"(叫我|喊我|称呼我|管我叫|叫我做|称呼为|称呼成)")),
)



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


def _visible_clause(allow_private: bool, group_id: str, viewer_owner: str = "",
                    prefix: str = "") -> tuple[str, list]:
    """当前场景能读到哪些 visibility 的条目（唯一的可见性判定口径）。

    v0.12.17 之前只有「群层」加了 ``visibility != 'private'``，而按人检索
    （user tier）、常驻、钉住清单都只按 owner 过滤。结果是 A 群里产生、
    visibility='group' 的条目，会在这个人于 B 群发言时被按人召回——
    “只在来源群可见”的承诺根本没生效。这里把三种可见性显式写成条件：

    - public：任何场景；
    - private：仅私聊（allow_private），且必须属于当前查看者；
    - group：仅当当前就在来源群。

    返回 (SQL 片段, 参数)。调用方把片段接在 WHERE 条件后面。
    """
    vis = f"COALESCE(NULLIF({prefix}visibility, ''), 'public')"
    parts = [f"{vis} = 'public'"]
    args: list = []
    if allow_private:
        if viewer_owner:
            parts.append(f"({vis} = 'private' AND {prefix}owner_id = ?)")
            args.append(viewer_owner)
        else:
            # 调用方已按 owner 收窄（如总结时按批发言人过滤），这里不再重复
            parts.append(f"{vis} = 'private'")
    if group_id:
        parts.append(f"({vis} = 'group' AND {prefix}group_id = ?)")
        args.append(group_id)
    # 不在群里时 group 条目一律不可读：它只属于来源群。
    return " AND (" + " OR ".join(parts) + ")", args


def _neg_words(text: str) -> set[str]:    return {w for w in _NEG_WORDS if w in str(text or "")}


def _slot_of(content: str) -> str:
    """把记忆归到一个「只能有一个当前值」的槽位，没有则返回空串。

    称呼是典型例子：用户先说「叫我兄弟」，后说「喊我老爹」，前者必须让位。
    这两句字面相似度为 0，靠 jaccard 永远发现不了冲突，只能按语义槽位判定。
    正则故意写窄——宁可漏判（靠模型 replaces 兜底），也不要误把
    「叫我别忘了喝水」当成称呼槽位。
    """
    s = str(content or "")
    for slot, pat in _SLOT_PATTERNS:
        if pat.search(s):
            return slot
    return ""



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
        # 热重载/面板进程与机器人进程短时并存时，没有 busy_timeout 会直接抛
        # "database is locked"，异常还会冒到消息处理链上。WAL 下写冲突本应等待，
        # 这里给 5 秒的等待窗口。
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # 卸载时通知仍在后台线程里跑的批次任务尽快收手（见 prune_messages）。
        self._closing = threading.Event()
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
            # v0.12.4：重要信息保护。pinned=1 的条目必进常驻注入，整理时不可合并/丢弃
            "ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0",
            # v0.12.5：钉住也要能过期/被取代。pinned_ts 供注入时标注新旧，
            # 让「兄弟→老爹」这类改动有据可依，而不是两条都永久常驻。
            "ALTER TABLE memories ADD COLUMN pinned_ts REAL DEFAULT 0",
            # v0.12.6：可见性。私聊里说的私事不能在群里被引用——同一个人
            # 在群里 @ 机器人时，私聊记忆一旦注入，回复就是把私事当众说出来。
            # private=仅私聊可用 / group=仅来源群可用 / public=任何场景可用
            "ALTER TABLE memories ADD COLUMN visibility TEXT DEFAULT ''",
        ):
            try:
                c.execute(ddl)
            except sqlite3.OperationalError:
                pass
        # v0.12.5 存量回填：0.12.4 钉住的老条目没有 pinned_ts，用最后更新时间兜底，
        # 否则注入时排不出新旧，也没法显示「钉于」。
        try:
            c.execute(
                "UPDATE memories SET pinned_ts = COALESCE(NULLIF(updated_ts, 0), ts)"
                " WHERE pinned = 1 AND COALESCE(pinned_ts, 0) <= 0"
            )
            c.commit()
        except sqlite3.OperationalError:
            pass
        # v0.12.6 存量回填：老库没有可见性信息，只能从当时的写入位置推断——
        # 有 group_id 的是群场景产出（群内可见），无归属的群级/全局条是公共信息，
        # 其余（有 owner、无 group_id）一律按私聊私有处理。宁可少用，不可误发。
        try:
            c.execute(
                "UPDATE memories SET visibility = CASE"
                " WHEN scope = 'global' OR (owner_id = '' AND group_id = ''"
                "   AND session_id = '') THEN 'public'"
                " WHEN group_id != '' THEN 'group'"
                " ELSE 'private' END"
                " WHERE COALESCE(visibility, '') = ''"
            )
            c.commit()
        except sqlite3.OperationalError:
            pass
        # v0.12.13 修正：0.12.12 及更早的 merge_memories 没有写 visibility 列，
        # 合并产物落成空串；而群聊过滤判的是「!= 'private'」，空串挡不住，
        # 等于把并进组里的私聊条目搬到了群里。组内只要有一条 private 就按
        # private 处理——宁可群聊少引用一条，也不把私事当众念出来。
        # 用 meta 打标记，只跑一次，避免把用户手动改回的可见性又压回去。
        try:
            if not c.execute("SELECT 1 FROM meta WHERE key = ?",
                             ("mig_merge_vis_v01213",)).fetchone():
                c.execute(
                    "UPDATE memories SET visibility = 'private'"
                    " WHERE COALESCE(visibility, '') != 'private'"
                    "   AND source = 'consolidate'"
                    "   AND group_id != '' AND COALESCE(merged_into, 0) = 0"
                    "   AND EXISTS (SELECT 1 FROM memories o"
                    "     WHERE o.merged_into = memories.id"
                    "       AND COALESCE(o.visibility, '') = 'private')"
                )
                c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)",
                          ("mig_merge_vis_v01213", str(int(time.time()))))
                c.commit()
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
        self._repair_orphan_private(c)
        c.commit()
        return fts_ok

    def _repair_orphan_private(self, c) -> None:
        """修复 v0.12.16 及更早版本留下的「孤儿私聊记忆」。

        总结模型有时不给 owner（或给了昵称而不是 ID），私聊里就会落成
        owner_id='' + visibility='private'。这种行按人检索查不到（按 owner 过滤），
        常驻也查不到（要求 owner 非空），等于永久沉在库里——生产库实测 29 条活跃
        记忆里有 7 条是这样，占 24%。

        私聊会话只有一个人，归属可以无歧义地从消息记录（或 umo 的
        FriendMessage 段）反推。只修「无群、无归属、私聊可见」的活跃行，
        群里的无归属行是群公共记忆，不能动。
        """
        try:
            if c.execute("SELECT 1 FROM meta WHERE key = ?",
                         ("mig_orphan_private_v01217",)).fetchone():
                return
            rows = c.execute(
                "SELECT id, session_id, platform FROM memories"
                " WHERE merged_into = 0 AND COALESCE(owner_id, '') = ''"
                "   AND COALESCE(group_id, '') = ''"
                "   AND COALESCE(visibility, '') = 'private'"
            ).fetchall()
            if not rows:
                # 新库/已修复的库：不写标记。否则「先建库、后写入孤儿行」
                # （降级回旧版本、或从备份恢复）会永远错过修复。
                return
            fixed = 0
            now = time.time()
            for r in rows:
                sid = str(r["session_id"] or "")
                owner = ""
                senders = [
                    str(x[0]) for x in c.execute(
                        "SELECT DISTINCT sender_id FROM messages"
                        " WHERE session_id = ? AND sender_id != '' LIMIT 3",
                        (sid,),
                    ).fetchall()
                ]
                if len(senders) == 1:
                    owner = self.resolve_owner(senders[0], str(r["platform"] or ""))
                if not owner and ":FriendMessage:" in sid:
                    raw = sid.rsplit(":FriendMessage:", 1)[1].strip()
                    if raw:
                        owner = self.resolve_owner(raw, str(r["platform"] or ""))
                if not owner:
                    continue
                c.execute(
                    "UPDATE memories SET owner_id = ?, scope = 'user', updated_ts = ?"
                    " WHERE id = ?",
                    (owner, now, int(r["id"])),
                )
                fixed += 1
            c.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                ("mig_orphan_private_v01217", str(fixed)),
            )
            if fixed:
                # 只在真有修复时打日志，避免每次启动都刷一行
                import logging

                logging.getLogger(__name__).info(
                    f"[hermes_core] 已修复 {fixed} 条无归属的私聊记忆（重新挂回本人）")
        except sqlite3.OperationalError:
            # identities 表尚未建好等边缘情况：不写 meta，下次启动重试
            pass

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
            # 补偿：executescript 会隐式 COMMIT，迁移若在「RENAME 之后、回灌之前」
            # 被 kill，库会停在「skills 空表 + skills_legacy_v080 有数据」。
            # 下次启动 owner_id 已存在，原逻辑直接 return，技能看起来全丢了。
            self._recover_skills_legacy(c)
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

    @staticmethod
    def _recover_skills_legacy(c) -> None:
        """把迁移中断后残留的 skills_legacy_v080 回灌回空表。

        归属已经无从考证（meta 没写成），按 global 处理：宁可让技能全体可见，
        也不让用户的技能库凭空消失。
        """
        try:
            if not c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='skills_legacy_v080'"
            ).fetchone():
                return
            if int(c.execute("SELECT COUNT(*) FROM skills").fetchone()[0]) > 0:
                return
            c.execute(
                "INSERT INTO skills(owner_id, scope, group_id, name, trigger, content,"
                " created_ts, updated_ts, use_count, merged_into, origin)"
                " SELECT '', 'global', '', name, trigger, content, created_ts,"
                " created_ts, COALESCE(use_count, 0), 0, 'hermes'"
                " FROM skills_legacy_v080"
            )
            c.execute("DROP TABLE skills_legacy_v080")
            c.commit()
        except sqlite3.OperationalError:
            pass

    # ---------- messages ----------

    def add_message(self, session_id: str, role: str, content: str,
                    platform: str = "", sender_id: str = "", sender_name: str = "",
                    group_id: str = "") -> int:
        with self._lock:
            try:
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
            except Exception:
                # FTS 写入失败时不能把未提交的 INSERT 留在连接上，
                # 否则会被下一次 commit 顺手写进库（消息与水位的顺序就乱了）
                self.conn.rollback()
                raise
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

    def _delete_message_batch(self, ids: list[int]) -> int:
        """删一批消息并同步 FTS。ids 来自本文件内的 SELECT，不含外部输入。"""
        if not ids:
            return 0
        if self._fts_enabled:
            self.conn.executemany(
                "DELETE FROM messages_fts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.executemany("DELETE FROM messages WHERE id = ?", [(i,) for i in ids])
        self.conn.commit()
        return len(ids)

    def prune_messages(self, days: int, batch: int = 2000,
                       require_summarized: bool = False) -> int:
        """TTL 保留期修剪：删除过期原始消息并同步 FTS，随后收缩 WAL。

        只动 messages（原始转写层，可再生）；memories / skills / identities
        是资产，不受影响。分批提交避免长事务阻塞在线写入。
        返回删除总条数。days <= 0 时直接返回 0（保留期关闭）。

        require_summarized=True 时只删「水位之前」的消息——原文是总结的唯一证据，
        没总结过就被删掉，那次对话的信息就永久消失了（实测水位 0 的旧消息会被直接删）。
        """
        if int(days) <= 0:
            return 0
        cutoff = time.time() - int(days) * 86400
        total = 0
        # 锁只包「取一批 + 删一批」：原来的写法把整个修剪过程放在一把锁里，
        # 而每条新消息的 add_message 都要抢同一把锁 —— 清理几十秒，
        # 事件循环上所有消息处理就跟着停几十秒。
        if require_summarized:
            # 按会话各自的水位删，绝不越过未总结的部分
            for sid in self.all_session_ids():
                wm = self.get_watermark(sid)
                if wm <= 0:
                    continue
                while True:  # asb-audit: ignore=ASB-13 同步方法由 asyncio.to_thread 调用，每批 LIMIT 必然收敛
                    if self._closing.is_set():
                        break  # 插件已卸载：别在旧连接上继续删
                    with self._lock:
                        ids = [r[0] for r in self.conn.execute(
                            "SELECT id FROM messages WHERE session_id = ?"
                            " AND ts < ? AND id <= ? LIMIT ?",
                            (sid, cutoff, wm, int(batch)),
                        ).fetchall()]
                        if ids:
                            total += self._delete_message_batch(ids)
                    if len(ids) < int(batch):
                        break
        else:
            # 海象循环：每批取一批删一批；批次大小受 LIMIT 约束，不会长事务
            while True:  # asb-audit: ignore=ASB-13 同步方法由 asyncio.to_thread 调用，每批 LIMIT 必然收敛
                if self._closing.is_set():
                    break
                with self._lock:
                    ids = [r[0] for r in self.conn.execute(
                        "SELECT id FROM messages WHERE ts < ? LIMIT ?",
                        (cutoff, int(batch)),
                    ).fetchall()]
                    if ids:
                        total += self._delete_message_batch(ids)
                if len(ids) < int(batch):
                    break
        if total:
            try:
                # checkpoint 会等所有读者结束，是不可预估时长的阻塞，必须放在锁外
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
        """用 FTS5 trigram 取「与查询共享三字片段」的消息，作为检索快路径。

        返回 None 表示索引不可用（老库没建表 / SQLite 无 trigram），
        调用方据此回退全扫。命中不足时调用方同样回退，所以不会损失召回。

        索引本身是 trigram 分词，而原来这里提交的是**整句短语**——
        自然语言一句话不会原样出现在历史消息里，实测 12 条常见查询里 10 条
        命中 0 条，快路径等于常年失效：每次白查一次 FTS 再退回扫 1500 行，
        而且 1500 行以外的历史永远检索不到。
        改为按 3 字窗口 OR（见 recall.fts_windows）：文档只要共享任意一个
        三字片段就能进候选，共享多个的由 bm25(rank) 排前面，
        最后仍交给 rank_messages 精排。
        """
        if not self._fts_enabled:
            return None
        wins = _recall.fts_windows(str(query))
        if not wins:
            return None
        # FTS5 要求 MATCH 左侧写表名，别名不可用；rank 只在 MATCH 时存在
        sql = ("SELECT m.* FROM messages_fts"
               " JOIN messages m ON m.id = messages_fts.rowid"
               " WHERE messages_fts MATCH ?")
        args: list = [" OR ".join('"' + w.replace('"', '""') + '"' for w in wins)]
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

        候选集 = FTS 三字片段命中（可能来自整段历史）+ 最近一段对话。
        最近对话永远要扫一小段：FTS 只认「连续 3 字」，像「部署 证书」这种
        两个词分开出现的消息它命不中，而这类恰恰是「刚才说过」的典型。
        FTS 命中充足时只补最近 200 条（约 2 ms），命中太少或索引不可用才退回
        扫满 scan 条——原来无论命中多少都扫满，等于快路径白付一次查询。
        返回按 id 倒序（最新在前），与旧实现保持一致，调用方无需调整。
        """
        if len(_recall.norm(query)) < 4:
            # 两三个字的寒暄没有区分度，检索只会带回噪声
            return []
        cap = max(50, int(scan))
        # FTS 侧只取 bm25 前 500：rank_messages 最终只留 top-k，再多只是白算
        fts = self._fts_candidates(query, min(cap, 500), session_id, sender_id)
        enough = fts is not None and len(fts) >= max(12, int(limit) * 3)
        scan_n = min(cap, 200) if enough else cap
        where, args = self._msg_where(session_id, sender_id)
        rows = self.conn.execute(
            f"SELECT * FROM messages{where} ORDER BY id DESC LIMIT ?",
            (*args, scan_n),
        ).fetchall()
        docs = [dict(r) for r in rows]
        if fts:
            seen = {d["id"] for d in docs}
            docs.extend([d for d in fts if d["id"] not in seen])
        hits = _recall.rank_messages(query, docs, k=max(1, int(limit)))
        return sorted(hits, key=lambda x: -int(x["id"]))

    def recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def message_signatures(self, session_id: str, limit: int = 20000) -> set[tuple]:
        """(role, content) 去重集，供「导入原生历史」做幂等判断。

        重复导入会把同一段对话再存一遍：消息数翻倍、总结重复跑、记忆重复写。
        """
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (str(session_id), max(1, int(limit))),
        ).fetchall()
        return {(str(r["role"]), str(r["content"])) for r in rows}

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

    VISIBILITY = ("private", "group", "public")

    @staticmethod
    def _norm_visibility(visibility: str, owner_id: str = "", group_id: str = "") -> str:
        """决定一条记忆可以被哪些场景读到。

        没显式指定时按写入位置推断：群里的发言天然是当众说的 → group；
        私聊里有归属人的 → private；无归属又无会话的全局信息 → public。
        推断的意义在于「默认不越界」：群场景绝不会去读 private。
        """
        v = str(visibility or "").strip().lower()
        if v in HermesDB.VISIBILITY:
            return v
        if group_id:
            return "group"
        if owner_id:
            return "private"
        return "public"

    def add_memory(self, kind: str, content: str, scope: str = "global",
                   source: str = "", owner_id: str = "", session_id: str = "",
                   platform: str = "", group_id: str = "",
                   visibility: str = "") -> int:
        vis = self._norm_visibility(visibility, owner_id, group_id)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories(ts, kind, scope, content, source, owner_id,"
                " session_id, platform, group_id, visibility)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), self._norm_kind(kind), scope, content, source,
                 owner_id or "", session_id or "", platform or "", group_id or "",
                 vis),
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
                         visibility: str = "",
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
        vis = self._norm_visibility(visibility, owner_id, group_id)
        # 去重池不能带 session_id：同一个人在私聊和群里各说一次同样的偏好，
        # session 不同就会各存一条，两条又会同时进常驻（实测注入块里出现过
        # 一模一样的重复条目）。按人/按群归并。
        #
        # 但去重必须同可见性：把群里说的话并进一条 private 记忆，等于让它
        # 从此在群里消失；反过来把私聊内容并进 group 记忆，就是把私事搬到群里。
        # 两者都不能自愈，所以宁可各留一条——常驻/检索按场景过滤，用户看不到重复。
        if owner_id:
            rows = self.conn.execute(
                "SELECT id, content, importance, pinned FROM memories WHERE merged_into = 0"
                " AND owner_id = ? AND (group_id = '' OR group_id = ?)"
                " AND COALESCE(visibility, '') = ?"
                " ORDER BY id DESC LIMIT 300",
                (owner_id, group_id or "", vis),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, content, importance, pinned FROM memories WHERE merged_into = 0"
                " AND owner_id = '' AND group_id = ? AND session_id = ?"
                " AND COALESCE(visibility, '') = ?"
                " ORDER BY id DESC LIMIT 300",
                (group_id or "", session_id or "", vis),
            ).fetchall()
        best_id, best_sim = 0, 0.0
        corrections: list[int] = []
        pin_old = False
        for r in rows:
            s = _recall.jaccard(content, r["content"] or "")
            if s < 0.60:
                continue
            # 否定词不一致时绝不去重：「喜欢猫」和「不喜欢猫」字面相似度 0.71，
            # 合并会把偏好记反。但这种「高度相似 + 极性相反」恰恰是用户改主意，
            # 旧条必须退场，否则两个相反偏好会同时被召回、同时注入。
            if _neg_words(content) != _neg_words(r["content"] or ""):
                corrections.append(int(r["id"]))
                pin_old = pin_old or bool(int(r["pinned"] or 0))
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
            if corrections:
                self.supersede_memories(corrections, best_id)
            if pin_old:
                # 被改掉的是常驻条目：新说法合并进了 best，常驻身份要跟过来，
                # 否则「叫我哥哥」被改成「叫我老师」后整个槽位从常驻里消失。
                self.set_pinned(best_id, True)
            return best_id, False
        new_id = self.add_memory(kind, content, scope=scope, source=source,
                                 owner_id=owner_id, session_id=session_id,
                                 platform=platform, group_id=group_id,
                                 visibility=vis)
        if corrections:
            self.supersede_memories(corrections, new_id)
            if pin_old:
                # 旧条本来是常驻的，新说法继续常驻，槽位不能空着
                self.set_pinned(new_id, True)
        return new_id, True

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
                       kinds: tuple[str, ...] = ("profile", "preference"),
                       allow_private: bool = True,
                       group_id: str = "") -> list[dict]:
        """常驻「关于你」条目：不依赖检索命中，每次对话固定注入。

        Hermes 的做法是把 MEMORY.md / USER.md 整块加载——「机器人记得你」的
        基线不该建立在「这次的关键词恰好撞上」之上。这里只取稳定的画像与偏好
        （fact 是具体事实，量大且多为一次性信息，走检索层按需召回），
        按重要度与命中次数排序，最后按字符预算截断。

        allow_private=False 用于群聊场景：私聊里沉淀的条目一律不取。
        这条不是「建议」，而是硬过滤——群里的回复是当众的，注进去就等于说出来。
        """
        if not owner_id:
            return []
        marks = _ph(len(kinds))
        # 可见性必须带群：A 群里沉淀的「只在来源群可见」条目，不能因为同一个人
        # 在 B 群发言就进常驻。v0.12.17 之前这里只挡 private，group 条目会跨群。
        vis, vis_args = _visible_clause(allow_private, group_id, owner_id)
        # 第一批：钉住的条目（任何 kind）——用户/模型明确要求「每轮必带」的信息，
        # 不受 kind 限制，也不受字符预算限制，这是重要信息的保护通道。
        # 按 pinned_ts 倒序：同槽位冲突时最新的排前面，模型能看出先后。
        pinned_rows = self.conn.execute(
            "SELECT id, kind, content, importance, hits, pinned, pinned_ts,"
            " COALESCE(visibility, '') AS visibility FROM memories"
            " WHERE merged_into = 0 AND owner_id = ? AND pinned = 1" + vis +
            " ORDER BY COALESCE(pinned_ts, 0) DESC, id DESC LIMIT ?",
            (owner_id, *vis_args, max(1, int(limit))),
        ).fetchall()
        # 第二批：画像/偏好优先。重要度必须随时间衰减——importance 只在重复写入时
        # +0.15、从不下降，按它硬排会让新偏好永远挤不进常驻（实测 6 条老条目
        # importance 0.6 占满席位，新偏好 0 进不来）。改成「被召回/更新才续命」。
        rows = self.conn.execute(
            f"SELECT id, kind, content, importance, hits, pinned, pinned_ts,"
            f" ts, updated_ts, last_hit_ts, COALESCE(visibility, '') AS visibility"
            f" FROM memories"
            f" WHERE merged_into = 0 AND owner_id = ? AND pinned = 0 AND kind IN ({marks})"
            + vis +
            f" ORDER BY id DESC LIMIT ?",
            (owner_id, *kinds, *vis_args, max(40, int(limit) * 8)),
        ).fetchall()
        now = time.time()
        ranked: list[tuple] = []
        for r in rows:
            ref = float(r["last_hit_ts"] or r["updated_ts"] or r["ts"] or 0)
            age_days = max(0.0, (now - ref) / 86400.0) if ref else 3650.0
            fresh = 0.5 ** (age_days / 120.0)      # 120 天半衰：不碰就淡
            score = ((1.0 + float(r["importance"] or 0))
                     * fresh
                     * (1.0 + min(0.5, float(r["hits"] or 0) * 0.02)))
            kind_rank = 0 if r["kind"] == "profile" else 1
            # 私聊场景里 private 比 group 更「贴身」：同一个人私下说的，比在群里
            # 顺口说的一句更能代表他。排序靠前，配合下面的同义去重，留下私聊那条。
            vis_rank = {"private": 0, "public": 1, "group": 2}.get(
                str(r["visibility"] or "private"), 2)
            ranked.append((kind_rank, vis_rank, -score, -int(r["id"]), r))
        ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        rows = [x[4] for x in ranked]
        out: list[dict] = []
        used = 0
        seen: set[int] = set()
        kept_texts: list[str] = []
        # 钉住条目先入，不受字符预算约束；但总条数仍受 limit 约束——
        # 否则「6 条 pin + 6 条画像」会变成 12 条常驻，钉得越多越贵。
        for r in pinned_rows:
            if len(out) >= limit:
                break
            text = str(r["content"] or "").strip()
            if not text:
                continue
            out.append(dict(r))
            seen.add(int(r["id"]))
            kept_texts.append(text)
            used += len(text) + 3
        for r in rows:             # 其余条目受字符预算约束
            if len(out) >= limit:
                break
            if int(r["id"]) in seen:
                continue
            text = str(r["content"] or "").strip()
            if not text:
                continue
            # 同一句话在私聊和群里各存了一条（可见性不同不能合并），
            # 私聊场景会同时看到两条一模一样的——按字面去重，只留排前面的那条。
            if any(_recall.jaccard(text, t) >= 0.85 for t in kept_texts):
                continue
            if used + len(text) > max_chars and out:
                break
            out.append(dict(r))
            kept_texts.append(text)
            used += len(text) + 3
        return out

    def pinned_memories(self, owner_id: str = "", limit: int = 30,
                        allow_private: bool = True,
                        group_id: str = "") -> list[dict]:
        """全部钉住条目（可跨 owner），供总结时判断「哪条被改变了」。

        allow_private=False 用于群会话的总结：私聊钉住的「叫我老爹」不该被塞进
        一次群聊总结的提示里——那正是把私事写进群记忆的最短路径。
        group_id 同样要参与：别的群里钉住的内容不能被拿来当本群总结的依据。
        """
        sql = ("SELECT id, kind, content, owner_id, group_id, pinned_ts, visibility"
               " FROM memories WHERE merged_into = 0 AND pinned = 1")
        args: list = []
        if owner_id:
            sql += " AND owner_id = ?"
            args.append(owner_id)
        vis, vis_args = _visible_clause(allow_private, group_id, owner_id)
        sql += vis
        args += vis_args
        sql += " ORDER BY id ASC LIMIT ?"
        args.append(max(1, int(limit)))
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def set_visibility(self, memory_id: int, visibility: str) -> bool:
        """改一条记忆的可见范围（用户明确说「这个可以在群里说」时才放宽）。"""
        v = str(visibility or "").strip().lower()
        if v not in self.VISIBILITY:
            return False
        cur = None
        with self._lock:
            cur = self.conn.execute(
                "UPDATE memories SET visibility = ?, updated_ts = ?"
                " WHERE id = ? AND merged_into = 0",
                (v, time.time(), int(memory_id)),
            )
            self.conn.commit()
        return cur.rowcount > 0

    def slot_conflicts(self, memory_id: int) -> list[int]:
        """只读：这条记忆入常驻时会顶掉哪些同槽位旧条（用于提示用户）。"""
        rec = self.memory_by_id(int(memory_id))
        if not rec:
            return []
        slot = _slot_of(rec["content"])
        if not slot:
            return []
        rows = self.conn.execute(
            "SELECT id, content FROM memories WHERE merged_into = 0 AND pinned = 1"
            " AND owner_id = ? AND id != ?"
            # 同槽位只在同一可见域里比较：私聊里的「叫我哥哥」不能被
            # 另一个群里钉的「叫我老师」顶掉，反之亦然。
            " AND (COALESCE(visibility, 'public') = 'public'"
            "      OR COALESCE(group_id, '') = ?)",
            (rec.get("owner_id") or "", int(memory_id),
             rec.get("group_id") or ""),
        ).fetchall()
        return [int(r["id"]) for r in rows if _slot_of(r["content"]) == slot]

    def supersede_slot_conflicts(self, memory_id: int) -> list[int]:
        """把同一个人、同一槽位里被这条新记忆顶掉的旧条退场。

        只在 set_pinned 之后有意义（只处理已钉住的旧条）。旧条不删文本，
        只把 merged_into 指向新条 —— 所有「活跃记忆」查询天然排除，
        误顶替时可以按 id 捞回原文。
        """
        gone = self.slot_conflicts(int(memory_id))
        if gone:
            self.supersede_memories(gone, int(memory_id))
        return gone

    def supersede_memories(self, ids: list[int], new_id: int) -> int:
        """旧记忆被新版本取代：退出活跃池（merged_into=new_id），并取消钉住。

        与 merge_memories 的区别：这里不做内容合并，只是「新说法生效」。
        原文保留，不物理删除。
        """
        new_id = int(new_id or 0)
        clean = []
        for x in ids or []:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i and i != new_id:
                clean.append(i)
        if not clean or not new_id:
            return 0
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE memories SET merged_into = ?, pinned = 0, pinned_ts = 0,"
                f" updated_ts = ? WHERE id IN ({_ph(len(clean))})"
                f" AND merged_into = 0 AND id != ?",
                (new_id, time.time(), *clean, new_id),
            )
            self.conn.commit()
            return cur.rowcount

    def all_active_memories(self, limit: int | None = None) -> list[dict]:
        """所有未合并的记忆，按 id 升序（即时间顺序），供巩固/去重使用。

        limit=None 表示全量。v0.12.16 之前硬编码 1000：超过 1000 条的账号，
        最老的那部分记忆永远不会被整理（手动「全库整理」也一样，但它会显示
        “全量”，属于静默漏处理）。
        """
        sql = "SELECT * FROM memories WHERE merged_into = 0 ORDER BY id"
        args: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            args = (max(1, int(limit)),)
        rows = self.conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def active_memory_ids_since(self, since_id: int) -> set[int]:
        """只取 id，供「本轮新增了哪些记忆」这类判断，避免把正文全读进内存。"""
        rows = self.conn.execute(
            "SELECT id FROM memories WHERE merged_into = 0 AND id > ?",
            (int(since_id),),
        ).fetchall()
        return {int(r["id"]) for r in rows}

    def max_memory_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) m FROM memories").fetchone()
        return int(row["m"] or 0)

    def memory_by_id(self, memory_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_pinned(self, memory_id: int, pinned: bool) -> bool:
        """钉住/取消钉住一条记忆。钉住的条目：必进常驻注入，整理时不可合并或丢弃。

        钉住不是终身制：重新钉住会刷新 pinned_ts，并自动让同槽位的旧说法退场
        （例如先说「叫我兄弟」后说「喊我老爹」，旧条 unpin 且退出活跃池）。
        """
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "UPDATE memories SET pinned = ?, pinned_ts = ?, updated_ts = ?"
                " WHERE id = ? AND merged_into = 0",
                (1 if pinned else 0, now if pinned else 0, now, int(memory_id)),
            )
            self.conn.commit()
        if cur.rowcount <= 0:
            return False
        if pinned:
            # 顶掉同槽位的旧条；不删原文，只退出活跃池。
            # 放锁外：supersede 自己会取锁，而 _lock 不可重入。
            self.supersede_slot_conflicts(int(memory_id))
        return True

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
            # 只收活跃且未钉住的条目：并发整理可能已经把同一批 id 合并掉，
            # 钉住则是用户明确要求保护的信息。原来只按 id 取，会把已合并条目的
            # merged_into 指针改指向新条，回滚链（unmerge）就此断掉。
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE id IN ({marks})"
                " AND COALESCE(merged_into, 0) = 0 AND COALESCE(pinned, 0) = 0",
                ids,
            ).fetchall()
            if len(rows) < 2 or len(rows) != len(set(ids)):
                # 组内条目在快照之后被别的路径合并/丢弃/钉住：整组作废，
                # 宁可不合也不写半组，否则会出现两条内容重叠的活跃记忆。
                return 0
            rows = sorted(rows, key=lambda r: r["id"])
            base = rows[0]
            # 可见性必须显式继承：漏写这一列会落成空串，而群聊过滤判的是
            # 「!= 'private'」，空串照样被召回——并进组里的私聊条目就漏到群里了。
            vis = self._norm_visibility(
                base["visibility"], base["owner_id"] or "", base["group_id"] or "")
            now = time.time()
            try:
                cur = self.conn.execute(
                    "INSERT INTO memories(ts, kind, scope, content, source, owner_id,"
                    " session_id, platform, group_id, merged_into, updated_ts,"
                    " visibility)"
                    " VALUES(?,?,?,?,?,?,?,?,?,0,?,?)",
                    (now, kind or base["kind"], base["scope"], content, "consolidate",
                     base["owner_id"] or "", base["session_id"] or "",
                     base["platform"] or "", base["group_id"] or "", now, vis),
                )
                new_id = cur.lastrowid
                upd = self.conn.execute(
                    f"UPDATE memories SET merged_into = ?, updated_ts = ?"
                    f" WHERE id IN ({marks})"
                    " AND COALESCE(merged_into, 0) = 0 AND COALESCE(pinned, 0) = 0",
                    (new_id, now, *ids),
                )
                if upd.rowcount != len(rows):
                    # 期间被并发改动：新条和标记一起回滚，不留半成品
                    self.conn.rollback()
                    return 0
                self.conn.commit()
            except Exception:
                # 不能只把异常往上抛：INSERT 已执行而 UPDATE 失败时，事务会悬在
                # 连接上，被下一次任意 commit 顺手提交，变成「新条有了、旧条还在」。
                self.conn.rollback()
                raise
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
        钉住的条目拒绝丢弃——那是用户明确要求保护的重要信息。
        """
        ids = [int(i) for i in ids if i]
        if not ids:
            return 0
        with self._lock:
            marks = _ph(len(ids))
            cur = self.conn.execute(
                f"UPDATE memories SET merged_into = -1, updated_ts = ?"
                f" WHERE id IN ({marks}) AND merged_into = 0 AND pinned = 0",
                (time.time(), *ids),
            )
            self.conn.commit()
            return cur.rowcount

    def search_memories(self, query: str, limit: int = 5,
                        owner_id: str | None = None,
                        session_id: str | None = None,
                        platform: str | None = None,
                        group_id: str | None = None,
                        scan: int = 3000,
                        allow_private: bool = True,
                        viewer_owner: str = "",
                        global_mode: bool = False) -> list[dict]:
        """记忆检索：作用域分层 + 本地相关性排序（bigram/IDF/时间衰减/去冗余）。

        群公共记忆的键用 (platform, group_id) 而不是 session_id(umo)，
        因此开启/关闭 unique_session 不会让既有群记忆失联；session_id 仅作老数据兼容。

        分层逐级补：用户私有 > 群共享 > 全局。分层后各自排序再串起来，
        比给不同层级乘系数更直观，也不会出现「全局的高分记忆把用户偏好挤出前 5」。

        可见性在每一层都要判（`_visible_clause`）：
        - allow_private=False（群聊）时 private 条目不参与；
        - visibility='group' 的条目只在其来源群参与——v0.12.16 之前按人检索
          不看 group_id，A 群的「只在来源群可见」会被 B 群召回。

        viewer_owner 是「当前是谁在看」，用于 private 条目归属校验；
        group 模式（memory_scope=group）不传 owner_id 时也要传它。
        """
        tiers: list[tuple[str, list]] = []
        viewer = viewer_owner or (owner_id or "")
        if global_mode:
            # 全机器人共享：看得到所有人的 public、本群的 group、以及自己的
            # private（仅私聊）。旧实现是「整表不分可见性」，私聊内容会在
            # global 模式下被注入给所有人。
            vis, va = _visible_clause(allow_private, group_id or "", viewer)
            tiers.append(("merged_into = 0" + vis, list(va)))
        elif owner_id is None and group_id is None:
            vis, va = _visible_clause(allow_private, "", viewer)
            tiers.append(("merged_into = 0" + vis, list(va)))
        else:
            if owner_id is not None:
                vis, va = _visible_clause(allow_private, group_id or "", owner_id)
                tiers.append(("merged_into = 0 AND owner_id = ?" + vis,
                              [owner_id, *va]))
            if group_id is not None:
                cond = "merged_into = 0 AND ((group_id = ? AND platform = ?)"
                args: list = [group_id, platform or ""]
                if session_id is not None:
                    cond += " OR (session_id = ? AND group_id = ''))"
                    args.append(session_id)
                else:
                    cond += ")"
                vis, va = _visible_clause(allow_private, group_id, viewer)
                tiers.append((cond + vis, [*args, *va]))
            vis, va = _visible_clause(allow_private, "", viewer)
            tiers.append(
                ("merged_into = 0 AND owner_id = '' AND session_id = ''"
                 " AND group_id = ''" + vis, list(va))
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

    def get_alias(self, alias_id: str) -> str:
        """这个 alias 已经绑定到谁了；没绑过返回空串。"""
        row = self.conn.execute(
            "SELECT canonical_id FROM identities WHERE alias_id = ?", (alias_id,)
        ).fetchone()
        return str(row["canonical_id"] or "") if row else ""

    def sender_seen(self, raw_id: str) -> bool:
        """这个原始 ID 是否在本机说过话（用来判断「这是别人的活人账号」）。"""
        row = self.conn.execute(
            "SELECT 1 FROM messages WHERE sender_id = ? LIMIT 1", (raw_id,)
        ).fetchone()
        return bool(row)

    def delete_alias(self, alias_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM identities WHERE alias_id = ?", (alias_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def all_memories(self, limit: int = 200, owner_id: str = "") -> list[dict]:
        if owner_id:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE merged_into = 0 AND owner_id = ?"
                " ORDER BY id DESC LIMIT ?", (owner_id, limit),
            ).fetchall()
        else:
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
        """物理删除一条记忆；指向它的合并产物要先「松绑」。

        合并链是 merged_into 单链表：删掉中间节点后，子条仍指着已不存在的 id，
        所有活跃查询都会把它们排除（merged_into != 0），等于整组内容永久消失，
        unmerge() 也救不回来。删除前把子条挂回它自己的父节点（没有父就恢复活跃）。
        """
        mid = int(memory_id)
        with self._lock:
            row = self.conn.execute(
                "SELECT merged_into FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            if not row:
                return False
            parent = int(row["merged_into"] or 0)
            self.conn.execute(
                "UPDATE memories SET merged_into = ? WHERE merged_into = ?",
                (parent if parent > 0 else 0, mid),
            )
            cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
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
        # 计数写在读路径上，必须与其它写操作共用一把锁：否则它会替
        # 后台线程里未提交的事务「顺手 commit」，把半成品写进库。
        with self._lock:
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

    def request_stop(self) -> None:
        """让后台批次任务（TTL 修剪）在下一批之前收手。"""
        self._closing.set()

    def close(self):
        # 先立旗再关连接：后台线程（TTL 修剪）每批都会检查，尽量在关库前收手。
        self._closing.set()
        try:
            self.conn.close()
        except Exception:
            pass
