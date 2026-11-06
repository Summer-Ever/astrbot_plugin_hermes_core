"""生产库体检：对着真实 hermes.db 的一份副本，把「静默做错」的几类状态量出来。

用法（只读，绝不碰原库）：

    python tests/inspect_db.py --db D:/path/to/hermes.db
    python tests/inspect_db.py --db D:/path/to/hermes.db --repair   # 在副本上跑启动修复，看效果

不依赖 astrbot 运行时（logger 由 tests/_astrbot_stub.py 打桩），
任意 Python 3.9+ 可跑。检查项：

1. 孤儿私聊记忆   owner_id='' + private + 无群 → 按人检索和常驻都读不到，永久沉底
2. 分层错位       private + 无群 + scope='group' → 群层按可见性过滤、按人层 scope 不匹配，两边都空
3. 悬空合并链     merged_into > 0 但父条不存在 → 子条永久不活跃，unmerge 也救不回
                   （merged_into = -1 是「已删除」哨兵，不是父指针，不计入）
4. 可见性缺失     空 visibility 会被 COALESCE 当成 public——比预期更公开
5. 跨群泄漏       用真实关键词检索一次，确认 visibility='group' 的条目只在本群命中
6. 存储一致性     PRAGMA integrity_check / 消息与 FTS 行数是否对齐
"""

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 插件运行时只用 astrbot.api 的官方 logger；纯 Python 体检脚本没有 AstrBot，
# 由测试侧在导入插件模块之前注入替身。
from _bootstrap import db_module

ORPHAN_SQL = (
    "SELECT id, session_id FROM memories WHERE merged_into = 0"
    "  AND COALESCE(owner_id, '') = '' AND COALESCE(group_id, '') = ''"
    "  AND COALESCE(visibility, '') = 'private'"
)
MISSCOPED_SQL = (
    "SELECT id FROM memories WHERE COALESCE(visibility, '') = 'private'"
    "  AND COALESCE(group_id, '') = '' AND COALESCE(scope, '') <> 'user'"
)
DANGLING_SQL = (
    "SELECT COUNT(*) FROM memories m WHERE m.merged_into > 0"
    "  AND NOT EXISTS(SELECT 1 FROM memories p WHERE p.id = m.merged_into)"
)
BLANK_VIS_SQL = "SELECT COUNT(*) FROM memories WHERE COALESCE(visibility, '') = ''"


def snapshot(src: str) -> tuple[str, str]:
    """把库快照到临时目录，返回 (目录, 库路径)。

    不用 shutil 直接拷文件：库里开着 WAL 时，db 与 -wal 是两个文件，
    边写边拷会得到一对「时间点不一致」的文件，SQLite 打开就报
    `file is not a database`。走 backup API 由 SQLite 自己保证快照一致。
    """
    tmpdir = tempfile.mkdtemp(prefix="hermes-inspect-")
    dest = os.path.join(tmpdir, "hermes.db")
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(dest)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()
    except sqlite3.Error:
        # 只读打开被 -shm 权限挡住时退回文件拷贝（库没在写就够用）
        for suffix in ("", "-wal", "-shm"):
            p = src + suffix
            if os.path.exists(p):
                shutil.copy(p, dest + suffix)
    return tmpdir, dest


def q(db_path: str, sql: str, args: tuple = ()) -> list:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        return [dict(x) for x in c.execute(sql, args).fetchall()]
    finally:
        c.close()


def report(db_path: str) -> dict:
    out = {}
    out["记忆总数"] = q(db_path, "SELECT COUNT(*) c FROM memories")[0]["c"]
    out["活跃记忆"] = q(db_path, "SELECT COUNT(*) c FROM memories WHERE merged_into = 0")[0]["c"]
    out["消息总数"] = q(db_path, "SELECT COUNT(*) c FROM messages")[0]["c"]

    out["孤儿私聊记忆"] = [r["id"] for r in q(db_path, ORPHAN_SQL)]
    out["分层错位(private+group)"] = [r["id"] for r in q(db_path, MISSCOPED_SQL)]
    out["悬空合并链"] = q(db_path, DANGLING_SQL)[0]["COUNT(*)"]
    out["空 visibility"] = q(db_path, BLANK_VIS_SQL)[0]["COUNT(*)"]
    out["integrity"] = q(db_path, "PRAGMA integrity_check")[0]["integrity_check"]
    try:
        out["FTS 行数"] = q(db_path, "SELECT COUNT(*) c FROM messages_fts")[0]["c"]
    except sqlite3.OperationalError:
        out["FTS 行数"] = "未启用"
    out["分布"] = q(
        db_path,
        "SELECT scope, visibility, COUNT(*) c FROM memories WHERE merged_into = 0"
        " GROUP BY scope, visibility ORDER BY c DESC",
    )
    return out


def cross_group_probe(db_path: str) -> list:
    """用真库里的真实条目做一次跨群检索，确认 group 可见性真的只在本群生效。"""
    rows = q(
        db_path,
        "SELECT id, owner_id, group_id, content FROM memories"
        " WHERE visibility = 'group' AND merged_into = 0"
        "   AND COALESCE(owner_id, '') <> '' AND COALESCE(group_id, '') <> '' LIMIT 1",
    )
    if not rows:
        return ["（库里没有可测的 group 条目，跳过）"]
    r0 = rows[0]
    kw = str(r0["content"] or "")[:2]
    if not kw:
        return ["（条目正文为空，跳过）"]

    HermesDB = db_module().HermesDB  # 延迟导入：只在需要真正检索时加载

    db = HermesDB(os.path.dirname(db_path))
    try:
        lines = [f"探针条目 {r0['id']} 关键词 {kw!r}"]
        for label, gid in (("本群", r0["group_id"]), ("他群", "__OTHER_GROUP__")):
            hit = [m["id"] for m in db.search_memories(
                kw, owner_id=r0["owner_id"], group_id=gid,
                allow_private=False, limit=50)]
            pin = [m["id"] for m in db.pinned_memories(
                r0["owner_id"], group_id=gid, allow_private=False)]
            lines.append(f"  {label} 检索 {hit[:8]} / 钉住 {pin[:8]}")
        anon = [m["id"] for m in db.search_memories(
            kw, owner_id="", group_id="", allow_private=True, limit=50)
            if (m.get("visibility") or "") == "private"]
        lines.append(f"  无身份时命中的 private 条目 {anon[:8]}（应为空）")
        return lines
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="hermes.db 体检（只读副本）")
    ap.add_argument("--db", required=True, help="hermes.db 路径")
    ap.add_argument("--repair", action="store_true",
                    help="在副本上执行一次启动修复，观察修复效果（绝不改原库）")
    ap.add_argument("--keep", action="store_true", help="保留临时副本目录")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"找不到数据库：{args.db}")
        return 2

    tmpdir, copy = snapshot(args.db)
    try:
        before = report(copy)
        print("== 体检 ==")
        for k, v in before.items():
            if k == "分布":
                print("  分布（活跃记忆）:")
                for row in v:
                    print(f"    scope={row['scope'] or '(空)'} visibility={row['visibility']} → {row['c']} 条")
            else:
                print(f"  {k}: {v}")

        print("\n== 跨群探针 ==")
        for line in cross_group_probe(copy):
            print("  " + line if not line.startswith(" ") else line)

        if args.repair:
            HermesDB = db_module().HermesDB

            print("\n== 在副本上跑一次启动修复 ==")
            db = HermesDB(tmpdir)
            db.close()
            after = report(copy)
            for key in ("孤儿私聊记忆", "分层错位(private+group)", "悬空合并链", "integrity"):
                print(f"  {key}: {before.get(key)} → {after.get(key)}")
    finally:
        if args.keep:
            print(f"\n副本保留在：{tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
