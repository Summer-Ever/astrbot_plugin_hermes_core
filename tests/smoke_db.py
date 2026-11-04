# -*- coding: utf-8 -*-
"""存储层冒烟测试：不依赖 AstrBot，直接跑 SQLite。

改动 db.py（SQL 拼接收敛、FTS 快路径）后必须过一遍，确认行为没变。

    python tests/smoke_db.py
"""

import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.db import HermesDB  # noqa: E402
from modules.paths import UnsafePath, safe_join  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="hermes-smoke-")
    db = HermesDB(tmp, "smoke.db")
    print(f"[1] schema  fts={db.fts_enabled}")

    # --- 消息与检索
    db.add_message("s1", "user", "我电脑是 Windows 11 版本 23H2，想装 ffmpeg", sender_id="u1")
    db.add_message("s1", "assistant", "Windows 11 装 ffmpeg 用 winget install ffmpeg 最快", sender_id="u1")
    db.add_message("s2", "user", "今天天气不错，出去走走", sender_id="u2")
    db.add_message("s2", "assistant", "好啊，注意防晒", sender_id="u2")

    hits = db.search_messages("windows 11 版本", limit=3)
    check("search_messages 命中 Windows 相关消息", len(hits) > 0, str(len(hits)))
    check("search_messages 命中的确实是同一句",
          bool(hits) and "Windows 11" in hits[0]["content"], hits[0]["content"] if hits else "")

    noise = db.search_messages("股票基金", limit=3)
    check("无关查询不返回结果", noise == [], str(len(noise)))

    # --- FTS 快路径（trigram 短语）
    fts = db._fts_candidates("Windows 11 版本", 50, None, None)
    check("FTS 索引可查询（非 None）", fts is not None, "索引不可用或 SQL 报错")
    if fts is not None:
        check("FTS 短语命中至少 1 条", len(fts) >= 1, str(len(fts)))
        only_u1 = db._fts_candidates("Windows 11 版本", 50, None, "u1")
        check("FTS 支持 sender 过滤", only_u1 is not None and all(r["sender_id"] == "u1" for r in only_u1))

    check("count_messages 全量", db.count_messages() == 4, str(db.count_messages()))
    check("count_messages 按 sender", db.count_messages(sender_id="u1") == 2, str(db.count_messages(sender_id="u1")))

    # --- 记忆
    mid, created = db.add_memory_smart("preference", "用户喜欢用 winget 装软件",
                                       owner_id="u1", threshold=0.70)
    check("add_memory_smart 新建", created is True and mid > 0, f"{mid},{created}")
    mid2, created2 = db.add_memory_smart("preference", "用户喜欢用 winget 装软件",
                                         owner_id="u1", threshold=0.70)
    check("add_memory_smart 重复不新增", created2 is False and mid2 == mid, f"{mid2},{created2}")

    db.add_memory("profile", "用户是在上海做后端的工程师", owner_id="u1")
    db.add_memory("profile", "另一个人是设计师", owner_id="u2")

    notes = db.resident_notes("u1", limit=5, max_chars=800)
    check("resident_notes 只返回本人", all("设计师" not in n["content"] for n in notes), str(len(notes)))
    check("resident_notes 含画像", any("工程师" in n["content"] for n in notes))

    ids = [n["id"] for n in notes]
    if ids:
        db.bump_hits(ids)
        db.bump_hits(ids)
        after = db.resident_notes("u1", limit=5, max_chars=800)
        check("bump_hits 累加生效", any(int(n.get("hits") or 0) >= 2 for n in after))

    mem_hits = db.search_memories("我不会写代码", owner_id="u1", limit=3)
    check("search_memories 返回结构正常", isinstance(mem_hits, list))

    # --- 技能
    db.upsert_skill("deploy", "部署", "# 部署\n\n步骤一", owner_id="u1", scope="private")
    db.upsert_skill("deploy", "部署", "# 部署（u2 的私版）\n\n步骤二", owner_id="u2", scope="private")
    u1_skills = db.list_skills("u1", "")
    u2_skills = db.list_skills("u2", "")
    check("技能按身份隔离（u1 看不到 u2 私版）",
          any(s["name"] == "deploy" for s in u1_skills) and len(u1_skills) == 1, str(len(u1_skills)))
    u2_got = db.get_skill("deploy", "u2", "")
    check("同名技能可共存", bool(u2_got) and "步骤二" in str(u2_got.get("content")),
          str(u2_got.get("content") if u2_got else None))

    got = db.get_skill("deploy", "u1", "")
    check("get_skill 取到本人版本", bool(got) and "步骤一" in str(got.get("content")))

    md = db.export_memories_markdown()
    check("export_memories_markdown 非空", bool(md.strip()), str(len(md)))

    # --- 路径安全
    # safe_join 反例样本：三者都应抛 UnsafePath
    for bad in ("../etc/passwd", "/etc/passwd", "C:\\Windows\\system.ini"):
        try:
            safe_join(tmp, bad)
            check(f"safe_join 拒绝 {bad}", False, "未抛异常")
        except UnsafePath:
            check(f"safe_join 拒绝 {bad}", True)
    check("safe_join 正常拼接", safe_join(tmp, "a", "b.txt").endswith(os.path.join("a", "b.txt")))

    # --- 保留期修剪（TTL）
    db.add_message("s1", "user", "很老的一条消息", sender_id="u1")
    db.conn.execute("UPDATE messages SET ts = ts - 200*86400")
    db.conn.commit()
    db.add_message("s1", "user", "新的保留消息", sender_id="u1")
    pruned = db.prune_messages(90)
    check("prune_messages 删除过期消息", pruned >= 1, f"删了 {pruned}")
    left = [r["content"] for r in db.conn.execute("SELECT content FROM messages").fetchall()]
    check("prune_messages 保留新消息", left == ["新的保留消息"], str(left))
    fts_left = db.conn.execute("SELECT count(*) FROM messages_fts").fetchone()[0]
    check("prune_messages 同步 FTS", fts_left == 1, f"FTS 剩 {fts_left}")
    check("prune_messages 0 天关闭", db.prune_messages(0) == 0)
    mem_count = db.conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    check("prune_messages 不碰记忆", mem_count > 0, f"剩 {mem_count} 条")

    # --- 注入去重：窗口过滤 + 重复抑制
    from modules.recall import filter_in_window, suppress_repeats
    all_ids = db.recent_message_ids("s1", None, 100)
    check("recent_message_ids 返回全部", len(all_ids) == 1, str(all_ids))
    newest = db.recent_message_ids("s1", None, 1)
    hits = [{"id": newest[0], "content": "窗口内"}, {"id": 99999, "content": "窗口外"}]
    kept = filter_in_window(hits, set(all_ids))
    check("filter_in_window 剔除窗口内命中", [h["content"] for h in kept] == ["窗口外"], str(kept))
    check("filter_in_window 空窗口不动作", len(filter_in_window(hits, set())) == 2)
    items = [{"id": 1, "s": 0.9}, {"id": 2, "s": 0.8}, {"id": 3, "s": 0.7}]
    fresh = suppress_repeats(items, {1: 9}, cur_turn=10, gap=4)
    check("suppress_repeats 跳过近期已注入", [x["id"] for x in fresh] == [2, 3], str(fresh))
    all_seen = suppress_repeats(items, {1: 10, 2: 10, 3: 10}, cur_turn=10, gap=4)
    check("suppress_repeats 全抑制保留头部", [x["id"] for x in all_seen] == [1], str(all_seen))
    check("suppress_repeats 首次全保留", len(suppress_repeats(items, {}, 10, 4)) == 3)

    db.conn.close()
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
