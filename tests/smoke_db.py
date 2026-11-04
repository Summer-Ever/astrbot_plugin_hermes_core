# -*- coding: utf-8 -*-
"""存储层冒烟测试：不依赖 AstrBot，直接跑 SQLite。

改动 db.py（SQL 拼接收敛、FTS 快路径）后必须过一遍，确认行为没变。

    python tests/smoke_db.py
"""

import os
import sys
import tempfile
import time
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

    # --- FTS 快路径（trigram 三字窗口）
    fts = db._fts_candidates("Windows 11 版本", 50, None, None)
    check("FTS 索引可查询（非 None）", fts is not None, "索引不可用或 SQL 报错")
    if fts is not None:
        check("FTS 命中至少 1 条", len(fts) >= 1, str(len(fts)))
        only_u1 = db._fts_candidates("Windows 11 版本", 50, None, "u1")
        check("FTS 支持 sender 过滤", only_u1 is not None and all(r["sender_id"] == "u1" for r in only_u1))

    check("count_messages 全量", db.count_messages() == 4, str(db.count_messages()))
    check("count_messages 按 sender", db.count_messages(sender_id="u1") == 2, str(db.count_messages(sender_id="u1")))

    # --- FTS 快路径真的能命中（v0.12.16）
    # 原实现提交的是「整句短语」，自然语言在历史里几乎不会原样出现 → 命中 0，
    # 快路径常年失效，检索被死锁在「最近 scan 条」。这两条断言钉住这个行为。
    db.add_message("s3", "user", "家里那台群晖证书过期了，续期一直报错，dns 校验失败",
                   sender_id="u9")
    sep = db._fts_candidates("群晖证书续期 acme 失败", 50, None, None)
    check("FTS 命中非连续短语（三字窗口 OR）", bool(sep), f"命中 {len(sep or [])} 条")

    # 把这条挤到「最近 scan 条」之外，只剩 FTS 能找回它
    old_id = db.conn.execute("select max(id) from messages").fetchone()[0]
    filler = [(time.time() - i, "s3", "qq", "u9", "老用户", "user",
               f"第{i}条：今天天气不错，随便聊聊", "") for i in range(60)]
    db.conn.executemany(
        "INSERT INTO messages(ts,session_id,platform,sender_id,sender_name,role,content,group_id)"
        " VALUES(?,?,?,?,?,?,?,?)", filler)
    db.conn.commit()
    for f_id, f_text, f_sid in db.conn.execute(
            "select id, content, session_id from messages where id > ?", (old_id,)).fetchall():
        db.conn.execute("INSERT INTO messages_fts(rowid, content, session_id) VALUES(?,?,?)",
                        (f_id, f_text, f_sid))
    db.conn.commit()
    recent_only = [r["id"] for r in db.conn.execute(
        "select id from messages where session_id='s3' order by id desc limit 20").fetchall()]
    check("旧消息确实在最近窗口之外", old_id not in recent_only, f"old={old_id}")
    far = db.search_messages("群晖证书续期 dns 校验失败", limit=3, session_id="s3", scan=20)
    check("超出最近窗口的消息仍靠 FTS 找回",
          any("群晖" in h["content"] for h in far),
          f"命中 {len(far)} 条：" + str([h["content"][:12] for h in far]))

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

    # fact 不进常驻：事实量大且多为一次性信息，只走检索层
    db.add_memory("fact", "用户电脑 CPU 是 i5-1240P", owner_id="u1")
    notes2 = db.resident_notes("u1", limit=10, max_chars=2000)
    check("resident_notes 默认不含 fact", all("i5-1240P" not in n["content"] for n in notes2))
    fact_hits = db.search_memories("i5-1240P CPU", owner_id="u1", limit=3)
    check("fact 仍可被检索", any("i5-1240P" in m["content"] for m in fact_hits))

    # 丢弃：退出活跃池，检索与常驻都看不到，可恢复
    did = db.add_memory("preference", "临时指令条目", owner_id="u1")
    check("discard 生效", db.discard_memories([did]) == 1)
    check("丢弃后检索不到", all(m["id"] != did for m in
                                db.search_memories("临时指令", owner_id="u1", limit=5)))
    check("丢弃后常驻不含", all(n["id"] != did for n in db.resident_notes("u1", limit=10)))
    n_active = db.conn.execute(
        "SELECT COUNT(*) c FROM memories WHERE id=? AND merged_into=-1", (did,)).fetchone()["c"]
    check("丢弃是软删（保留原文）", n_active == 1)

    # pinned：钉住条目必进常驻（哪怕 fact）、不受预算挤出、拒绝被丢弃
    pid = db.add_memory("fact", "关键约束：部署必须使用内网镜像源", owner_id="u1")
    check("钉住前 fact 不进常驻", all(n["id"] != pid for n in db.resident_notes("u1", limit=10)))
    check("set_pinned 生效", db.set_pinned(pid, True) is True)
    notes3 = db.resident_notes("u1", limit=10, max_chars=10)  # 预算压到 10 字
    check("钉住条目必进常驻且不受预算限制", any(n["id"] == pid for n in notes3))
    check("钉住条目拒绝丢弃", db.discard_memories([pid]) == 0)
    check("取消钉住", db.set_pinned(pid, False) is True)
    check("取消后恢复预算约束", all(n["id"] != pid for n in db.resident_notes("u1", limit=10, max_chars=10)))

    # 合并守卫：钉住的、已经并过的条目都不能被再并一次，
    # 否则回滚链（merged_into）会被改写，误合并再也还原不了
    m1 = db.add_memory("preference", "喜欢喝美式", owner_id="u1")
    m2 = db.add_memory("preference", "喝美式不加糖", owner_id="u1")
    db.set_pinned(m1, True)
    check("钉住的条目拒绝合并", db.merge_memories([m1, m2], "喜欢喝美式不加糖") == 0)
    db.set_pinned(m1, False)
    mid = db.merge_memories([m1, m2], "喜欢喝美式不加糖")
    check("正常合并产出新条", mid > 0, str(mid))
    m3 = db.add_memory("preference", "美式要加冰", owner_id="u1")
    check("已合并的条目不能再次参与", db.merge_memories([m2, m3], "美式加冰不加糖") == 0)
    check("合并产物可见性不为空", any(
        r["id"] == mid and (r["visibility"] or "") in ("private", "group", "public")
        for r in db.all_active_memories()))

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

    # --- 注入块渲染：类型提到分组标题、编号放行尾（正文优先）
    from modules.recall import format_memory_lines
    rows = [
        {"id": 140, "kind": "preference", "content": "称呼他为哥哥", "note": "钉于 2026-11"},
        {"id": 52, "kind": "profile", "content": "CPU 为 i5-1240P"},
        {"id": 32, "kind": "preference", "content": "目标不清时先问"},
    ]
    rendered = format_memory_lines(rows)
    check("渲染：多类型加分组标题", rendered[0] == "画像：", str(rendered[:1]))
    check("渲染：编号在行尾且带钉住附注",
          rendered[1] == "- CPU 为 i5-1240P（#52）"
          and rendered[3] == "- 称呼他为哥哥（#140，钉于 2026-11）", str(rendered))
    check("渲染：不再出现逐条 (kind) 标注", "(profile)" not in "\n".join(rendered))
    single = format_memory_lines([rows[1]])
    check("渲染：单一类型不加分组标题", single == ["- CPU 为 i5-1240P（#52）"], str(single))
    check("渲染：空内容与空列表不产生空行",
          format_memory_lines([{"id": 1, "kind": "fact", "content": "  "}]) == []
          and format_memory_lines([]) == [])
    check("渲染：未知 kind 归到「其他」",
          format_memory_lines([{"id": 7, "kind": "conclusion", "content": "x"},
                               {"id": 8, "kind": "profile", "content": "y"}])[0] == "画像：")

    # --- 总结水位：已总结的消息不会被「超 6 小时兜底」重复总结
    db.add_message("sw", "user", "第一轮旧消息", sender_id="u5")
    db.add_message("sw", "assistant", "第一轮旧回复", sender_id="u5")
    last = db.conn.execute("SELECT MAX(id) m FROM messages").fetchone()["m"]
    db.conn.execute("UPDATE messages SET ts = ts - 10*86400 WHERE session_id = 'sw'")
    db.conn.commit()
    db.set_watermark("sw", last)          # 模拟这一批已成功总结
    pend = {x["session_id"]: x for x in db.unsummarized_by_session()}
    check("水位：已总结的旧消息不进待总结", "sw" not in pend, str(pend.get("sw")))
    db.add_message("sw", "user", "水位之后的新消息", sender_id="u5")
    sw = {x["session_id"]: x for x in db.unsummarized_by_session()}.get("sw") or {}
    check("水位：只统计未总结的那一条", sw.get("count") == 1, str(sw))
    check("水位：oldest 不受已总结的旧消息影响",
          bool(sw.get("oldest")) and time.time() - float(sw["oldest"]) < 60, str(sw.get("oldest")))

    # --- pinned 生命周期：同槽位可被新说法顶替
    a = db.add_memory("preference", "以后叫我兄弟就行", owner_id="u7")
    check("首次钉住称呼", db.set_pinned(a, True) is True)
    check("钉住时间已记录", float(db.memory_by_id(a).get("pinned_ts") or 0) > 0)
    b = db.add_memory("preference", "以后喊我老爹", owner_id="u7")
    check("同槽位冲突可预判", db.slot_conflicts(b) == [a], str(db.slot_conflicts(b)))
    check("钉住新称呼", db.set_pinned(b, True) is True)
    old_a = db.memory_by_id(a)
    check("旧称呼自动退场（merged_into 指向新条）", int(old_a["merged_into"]) == b, str(old_a["merged_into"]))
    check("旧称呼取消钉住", int(old_a["pinned"]) == 0, str(old_a["pinned"]))
    rn = db.resident_notes("u7", limit=6, max_chars=400)
    rn_ids = [n["id"] for n in rn]
    check("常驻只剩新称呼", b in rn_ids and a not in rn_ids, str(rn_ids))
    check("旧称呼检索不到", all(m["id"] != a for m in
                                db.search_memories("兄弟", owner_id="u7", limit=5)))
    c = db.add_memory("fact", "用户不吃辣", owner_id="u7")
    db.set_pinned(c, True)
    check("不同槽位互不影响",
          int(db.memory_by_id(b)["pinned"]) == 1 and int(db.memory_by_id(c)["pinned"]) == 1)

    # --- 极性反转（改主意）也是一次取代，不能让两个相反偏好并存
    e1 = db.add_memory("preference", "用户喜欢吃香蕉", owner_id="u10")
    db.set_pinned(e1, True)
    e2, created_e = db.add_memory_smart("preference", "用户不喜欢吃香蕉", owner_id="u10")
    check("极性反转识别为新建", created_e is True and e2 != e1, f"{e2},{created_e}")
    check("极性反转后旧条退场", int(db.memory_by_id(e1)["merged_into"]) == e2,
          str(db.memory_by_id(e1)["merged_into"]))
    check("改主意后新条继承常驻", int(db.memory_by_id(e2)["pinned"]) == 1,
          str(db.memory_by_id(e2)["pinned"]))
    check("相反偏好检索只剩新条",
          all(m["id"] != e1 for m in db.search_memories("喜欢吃香蕉", owner_id="u10", limit=5)))

    # --- 显式取代（replaces）与常驻总量上限
    d1 = db.add_memory("preference", "喜欢清淡口味", owner_id="u8")
    db.set_pinned(d1, True)
    d2 = db.add_memory("preference", "现在喜欢重辣", owner_id="u8")
    check("supersede_memories 生效", db.supersede_memories([d1], d2) == 1)
    check("被取代的条目不在常驻", all(n["id"] != d1 for n in db.resident_notes("u8", limit=6)))
    for i in range(8):
        x = db.add_memory("fact", f"钉住测试条目 {i}", owner_id="u9")
        db.set_pinned(x, True)
    check("常驻总条数受 limit 约束（钉再多也不膨胀）",
          len(db.resident_notes("u9", limit=4)) == 4,
          str(len(db.resident_notes("u9", limit=4))))

    # --- 召回门槛校准：正常中文问句不能被虚词稀释掉
    db.add_memory("fact", "上次部署卡在 Nginx 证书链不完整，报错 SSL_ERROR", owner_id="u11")
    hit1 = db.search_memories("部署的时候证书报错怎么办", owner_id="u11", limit=3)
    check("长句中文问句可召回（旧门槛下为 0）", len(hit1) == 1, str(len(hit1)))
    noisy = db.search_memories("股票基金怎么样", owner_id="u11", limit=3)
    check("无关中文问句仍被挡住", noisy == [], str(len(noisy)))
    short_q = db.search_memories("这个报错怎么修", owner_id="u11", limit=3)
    check("短句含实词可召回", len(short_q) >= 1, str(len(short_q)))

    # --- 跨场景同名内容：私聊 + 群各存一条，但每个场景只看得到自己那条
    # 合并它们才是错的：并进私聊那条，群里从此想不起来；并进群那条，私事被搬到群里。
    x1, _c1 = db.add_memory_smart("preference", "用户喜欢深色主题", owner_id="u12",
                                  session_id="private:u12")
    x2, c2 = db.add_memory_smart("preference", "用户喜欢深色主题", owner_id="u12",
                                 session_id="group:G2", group_id="G2")
    check("私聊与群各自成条（不同可见性不合并）", c2 is True and x2 != x1, f"{x2},{c2}")
    priv = [n for n in db.resident_notes("u12", limit=6, allow_private=True)
            if "深色" in n["content"]]
    grp = [n for n in db.resident_notes("u12", limit=6, allow_private=False)
           if "深色" in n["content"]]
    check("私聊场景命中 private 那条", len(priv) == 1 and priv[0]["id"] == x1, str(priv))
    check("群场景只命中 group 那条", len(grp) == 1 and grp[0]["id"] == x2, str(grp))

    # --- 常驻衰减：老条目不能永久锁死席位
    for i in range(6):
        oid = db.add_memory("preference", f"旧偏好条目 {i}", owner_id="u13")
        aged = time.time() - 400 * 86400
        db.conn.execute("UPDATE memories SET importance=0.6, hits=10, ts=?,"
                        " updated_ts=?, last_hit_ts=0 WHERE id=?", (aged, aged, oid))
    db.conn.commit()
    fresh = db.add_memory("preference", "现在喜欢新风格了", owner_id="u13")
    notes_u13 = db.resident_notes("u13", limit=6, max_chars=400)
    check("新偏好能进入常驻（旧条目已衰减）",
          any(n["id"] == fresh for n in notes_u13), str([n["id"] for n in notes_u13]))

    # --- 修剪只删已总结的原文
    db.add_message("sp", "user", "已总结的旧消息")
    wm_id = db.conn.execute("SELECT MAX(id) m FROM messages").fetchone()["m"]
    db.set_watermark("sp", wm_id)
    db.add_message("sp", "user", "未总结的旧消息")
    db.conn.execute("UPDATE messages SET ts = ts - 200*86400 WHERE session_id='sp'")
    db.conn.commit()
    db.prune_messages(90, require_summarized=True)
    left_sp = [r["content"] for r in db.conn.execute(
        "SELECT content FROM messages WHERE session_id='sp' ORDER BY id").fetchall()]
    check("只删水位之前的原文，未总结的保留", left_sp == ["未总结的旧消息"], str(left_sp))

    # --- 短句查询补全
    from modules.recall import expand_query
    ctx = [{"role": "user", "content": "部署卡在 Nginx 证书链"},
           {"role": "assistant", "content": "试试补全证书链"},
           {"role": "user", "content": "继续"}]
    eq = expand_query("继续", ctx)
    check("短句补全上一条用户发言", "证书链" in eq and eq.endswith("继续"), eq)
    long_q = "帮我看看部署的证书链怎么补全"
    check("长句不补全", expand_query(long_q, ctx) == long_q)
    check("无上下文原样返回", expand_query("继续", []) == "继续")

    # --- 可见性隔离：群场景绝不能读到私聊内容
    secret = db.add_memory("fact", "下周要去做手术，别在群里提", owner_id="u20",
                           session_id="private:u20")
    in_private = db.search_memories("手术", owner_id="u20", limit=5, allow_private=True)
    in_group = db.search_memories("手术", owner_id="u20", group_id="G20",
                                  platform="qq", limit=5, allow_private=False)
    check("私聊场景能召回私聊记忆", any(m["id"] == secret for m in in_private),
          str([m["id"] for m in in_private]))
    check("群场景召回不到私聊记忆", all(m["id"] != secret for m in in_group),
          str([m["id"] for m in in_group]))
    check("群场景常驻不含私聊条目",
          all(n["id"] != secret for n in db.resident_notes("u20", allow_private=False)))

    # 显式公开的条目任何场景都可用
    pub = db.add_memory("profile", "用户是后端工程师，写 Python", owner_id="u20",
                        visibility="public")
    gp = db.search_memories("后端工程师", owner_id="u20", group_id="G20",
                            platform="qq", limit=5, allow_private=False)
    check("public 条目在群场景可见", any(m["id"] == pub for m in gp),
          str([m["id"] for m in gp]))

    # 群记忆只在其来源群可见
    db.add_memory("fact", "G21 的群规是每周五发周报", owner_id="", group_id="G21",
                  platform="qq", session_id="qq:group:G21")
    other = db.search_memories("周报", group_id="G22", platform="qq", limit=5,
                               allow_private=False)
    own = db.search_memories("周报", group_id="G21", platform="qq", limit=5,
                             allow_private=False)
    check("群记忆不跨群泄漏", other == [], str([m["id"] for m in other]))
    check("群记忆在本群可见", len(own) >= 1, str(len(own)))

    # 默认可见性推断
    check("无 group 有 owner 默认 private",
          db.memory_by_id(secret)["visibility"] == "private")
    check("有 group 默认 group",
          db.memory_by_id(own[0]["id"])["visibility"] == "group")
    check("改可见性可持久化", db.set_visibility(secret, "public") is True
          and db.memory_by_id(secret)["visibility"] == "public"
          and db.set_visibility(secret, "private") is True)

    # --- 指令权限依赖的查询：别人的记忆不能被 /hermes 记忆 列出来
    check("all_memories 按 owner 过滤",
          all(m["owner_id"] == "u20" for m in db.all_memories(limit=50, owner_id="u20"))
          and len(db.all_memories(limit=50, owner_id="u20")) >= 2)
    db.add_message("sb", "user", "我说过话", sender_id="raw_alice")
    check("sender_seen 认得出真实用过的 ID", db.sender_seen("raw_alice") is True)
    check("sender_seen 对陌生 ID 返回 False", db.sender_seen("raw_nobody") is False)
    check("get_alias 未绑定返回空", db.get_alias("raw_nobody") == "")
    db.add_alias("raw_alice", "u20", "qq")
    check("get_alias 返回归属", db.get_alias("raw_alice") == "u20")

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
