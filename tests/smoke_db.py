"""存储层冒烟测试：不依赖 AstrBot 运行时，直接跑 SQLite。

logger 由 tests/_bootstrap.py 在导入前打桩；插件运行时仍只用
astrbot.api 提供的官方 logger。

    python tests/smoke_db.py
"""

import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import db_module, paths_module, recall_module

_db = db_module()
_paths = paths_module()
rc = recall_module()
HermesDB = _db.HermesDB
UnsafePath = _paths.UnsafePath
safe_join = _paths.safe_join

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def main() -> int:
    # 换工作目录再跑：否则从插件目录执行时，import db（→ astrbot.logger 探测失败
    # 退化的分支除外）会把 data/cmd_config.json 等运行时残渣留在仓库工作区里，
    # 每次都触发提交审计的 ASB-20。测试本来就该在临时目录里进行。
    os.chdir(tempfile.mkdtemp(prefix="hermes-smoke-cwd-"))
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

    # --- 常驻层 v1.2.0：六个固定词条，每词条一段纯文本，按（人，场景）隔离 ---
    check("set_resident 写入词条",
          db.set_resident("u1", "private", "user", "在上海做后端工程师") is True)
    notes = db.get_resident("u1", "private")
    check("get_resident 只返回本人",
          notes.get("user") == "在上海做后端工程师" and "设计师" not in str(notes),
          str(notes))
    check("中文标题归一成词条 id",
          db.set_resident("u1", "private", "称呼", "哥哥") is True
          and db.get_resident("u1", "private").get("address") == "哥哥")
    check("未知词条拒绝（六词条封闭）",
          db.set_resident("u1", "private", "nope", "x") is False)
    check("无归属写入被拒", db.set_resident("", "private", "user", "x") is False)
    check("无身份读取返回空（fail-closed）", db.get_resident("", "private") == {})
    db.set_resident("u1", "private", "style", "第一行\n第二行\r\n第三行")
    check("写入时压平换行",
          db.get_resident("u1", "private").get("style") == "第一行 第二行 第三行",
          str(db.get_resident("u1", "private")))
    long_txt = "很长的词条文本" * 100
    db.set_resident("u1", "private", "rules", long_txt)
    check("存储层不截长度：全文原样保存",
          db.get_resident("u1", "private").get("rules") == long_txt)
    check("空文本=清空词条（删行）",
          db.set_resident("u1", "private", "rules", "") is True
          and "rules" not in db.get_resident("u1", "private"))
    db.set_resident("u1", "group:G1", "preference", "本群只聊工作")
    check("场景隔离：私聊与群各一套词条互不可见",
          db.get_resident("u1", "group:G1").get("preference") == "本群只聊工作"
          and "preference" not in db.get_resident("u1", "private")
          and "style" not in db.get_resident("u1", "group:G1"),
          str(db.resident_owners()))
    check("get_resident 只含非空词条",
          all(v for v in db.get_resident("u1", "private").values()))
    owners = db.resident_owners()
    check("resident_owners 按（人，场景）列出",
          any(o["owner_id"] == "u1" and o["scope"] == "private" and o["count"] >= 2
              for o in owners)
          and any(o["owner_id"] == "u1" and o["scope"] == "group:G1" for o in owners),
          str(owners))
    check("resident_count 数非空词条",
          db.resident_count() == sum(o["count"] for o in owners),
          f"{db.resident_count()} vs {str(owners)}")
    from modules.db import resident_scope as _rscope
    check("resident_scope 映射场景键",
          _rscope("") == "private" and _rscope("G1") == "group:G1"
          and _rscope(None) == "private", _rscope("G1"))

    # 归档层与常驻层互不牵连：归档写入不再进常驻，block 遗留列恒为空
    cpu_id = db.add_memory("fact", "用户电脑 CPU 是 i5-1240P", owner_id="u1")
    check("归档写入的 block 遗留列恒为空", db.memory_by_id(cpu_id)["block"] == "")
    fact_hits = db.search_memories("i5-1240P CPU", owner_id="u1", limit=3)
    check("fact 仍可被检索", any("i5-1240P" in m["content"] for m in fact_hits))
    check("归档条目不进常驻",
          "i5-1240P" not in str(db.get_resident("u1", "private")))

    # 丢弃：退出活跃池，检索看不到，可恢复
    did = db.add_memory("preference", "临时指令条目", owner_id="u1")
    check("discard 生效", db.discard_memories([did]) == 1)
    check("丢弃后检索不到", all(m["id"] != did for m in
                                db.search_memories("临时指令", owner_id="u1", limit=5)))
    n_active = db.conn.execute(
        "SELECT COUNT(*) c FROM memories WHERE id=? AND merged_into=-1", (did,)).fetchone()["c"]
    check("丢弃是软删（保留原文）", n_active == 1)
    check("丢弃不碰常驻词条",
          db.get_resident("u1", "private").get("user") == "在上海做后端工程师")

    # 合并：已经并过的条目不能被再并一次，
    # 否则回滚链（merged_into）会被改写，误合并再也还原不了
    m1 = db.add_memory("preference", "喜欢喝美式", owner_id="u1")
    m2 = db.add_memory("preference", "喝美式不加糖", owner_id="u1")
    mid = db.merge_memories([m1, m2], "喜欢喝美式不加糖")
    check("正常合并产出新条", mid > 0, str(mid))
    m3 = db.add_memory("preference", "美式要加冰", owner_id="u1")
    check("已合并的条目不能再次参与", db.merge_memories([m2, m3], "美式加冰不加糖") == 0)
    check("合并产物可见性不为空", any(
        r["id"] == mid and (r["visibility"] or "") in ("private", "group", "public")
        for r in db.all_active_memories()))

    ids = [r["id"] for r in db.all_memories(owner_id="u1", limit=50)]
    if ids:
        db.bump_hits(ids)
        db.bump_hits(ids)
        after = db.conn.execute(
            "SELECT COALESCE(hits,0) h FROM memories WHERE id=?", (ids[0],)).fetchone()
        check("bump_hits 累加生效", int(after["h"]) >= 2, str(after["h"]))

    mem_hits = db.search_memories("我不会写代码", owner_id="u1", limit=3)
    check("search_memories 返回结构正常", isinstance(mem_hits, list))

    # --- 技能
    db.upsert_skill("deploy", "部署", "# 部署\n\n步骤一", owner_id="u1", scope="private")
    db.upsert_skill("deploy", "部署", "# 部署（u2 的私版）\n\n步骤二", owner_id="u2", scope="private")
    u1_skills = db.list_skills("u1", "")
    u2_skills = db.list_skills("u2", "")
    check("技能按身份隔离（u1 看不到 u2 私版）",
          any(s["name"] == "deploy" for s in u1_skills) and len(u1_skills) == 1
          and any(s["name"] == "deploy" for s in u2_skills),
          f"u1={len(u1_skills)} u2={len(u2_skills)}")
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
    # v0.12.21：全抑制时不再只保底 1 条（注入不落历史，等于断供），补到半数
    check("suppress_repeats 全抑制时补足半数而非只剩头部",
          [x["id"] for x in all_seen] == [1, 2], str(all_seen))
    check("suppress_repeats 首次全保留", len(suppress_repeats(items, {}, 10, 4)) == 3)

    # --- 注入块渲染：类型提到分组标题、编号放行尾（正文优先）
    from modules.recall import format_memory_lines
    rows = [
        {"id": 140, "kind": "preference", "content": "称呼他为哥哥", "note": "2026-11 更新"},
        {"id": 52, "kind": "profile", "content": "CPU 为 i5-1240P"},
        {"id": 32, "kind": "preference", "content": "目标不清时先问"},
    ]
    rendered = format_memory_lines(rows)
    check("渲染：多类型加分组标题", rendered[0] == "画像：", str(rendered[:1]))
    check("渲染：编号在行尾且可带附注",
          rendered[1] == "- CPU 为 i5-1240P（#52）"
          and rendered[3] == "- 称呼他为哥哥（#140，2026-11 更新）", str(rendered))
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

    # --- 归档改口：称呼走 supersede 顶替；常驻词条是另一层，整段文本独立维护
    a = db.add_memory("preference", "以后叫我兄弟就行", owner_id="u7")
    b = db.add_memory("preference", "以后喊我老爹", owner_id="u7")
    # 改口走 supersede：旧条退出活跃池，新条顶上。
    check("显式取代生效", db.supersede_memories([a], b) == 1)
    check("取代后旧条挂上新父", int(db.memory_by_id(a)["merged_into"]) == b,
          str(db.memory_by_id(a)["merged_into"]))
    check("旧称呼检索不到", all(m["id"] != a for m in
                                db.search_memories("兄弟", owner_id="u7", limit=5)))
    db.set_resident("u7", "private", "address", "老爹")
    check("归档 supersede 不碰常驻词条",
          db.get_resident("u7", "private").get("address") == "老爹")
    db.set_resident("u7", "private", "preference", "不吃辣")
    check("不同词条互不影响",
          db.get_resident("u7", "private").get("address") == "老爹"
          and db.get_resident("u7", "private").get("preference") == "不吃辣")

    # --- 极性反转（改主意）也是一次取代，不能让两个相反偏好并存
    e1 = db.add_memory("preference", "用户喜欢吃香蕉", owner_id="u10")
    e2, created_e = db.add_memory_smart("preference", "用户不喜欢吃香蕉",
                                        owner_id="u10")
    check("极性反转识别为新建", created_e is True and e2 != e1, f"{e2},{created_e}")
    check("极性反转后旧条退场", int(db.memory_by_id(e1)["merged_into"]) == e2,
          str(db.memory_by_id(e1)["merged_into"]))
    check("改主意后新条留在归档层（block 遗留列为空）",
          db.memory_by_id(e2)["block"] == "", str(db.memory_by_id(e2)["block"]))
    check("相反偏好检索只剩新条",
          all(m["id"] != e1 for m in db.search_memories("喜欢吃香蕉", owner_id="u10", limit=5)))

    # --- 显式取代（replaces）
    d1 = db.add_memory("preference", "喜欢清淡口味", owner_id="u8")
    d2 = db.add_memory("preference", "现在喜欢重辣", owner_id="u8")
    check("supersede_memories 生效", db.supersede_memories([d1], d2) == 1)
    check("被取代的条目检索不到",
          all(m["id"] != d1 for m in db.search_memories("清淡口味", owner_id="u8", limit=5)))

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
    priv = [m for m in db.search_memories("深色主题", owner_id="u12", limit=5,
                                          allow_private=True)
            if "深色" in m["content"]]
    # 群场景必须带 group_id：可见性判定以「当前就在哪个群」为准
    grp = [m for m in db.search_memories("深色主题", owner_id="u12", group_id="G2",
                                         platform="", limit=5, allow_private=False)
           if "深色" in m["content"]]
    check("私聊场景命中 private 那条", len(priv) == 1 and priv[0]["id"] == x1, str(priv))
    check("群场景命中 group 那条", any(m["id"] == x2 for m in grp), str(grp))
    # 常驻同理：同一人的私聊词条与群词条各一套，场景键不同就互不可见
    db.set_resident("u12", "private", "preference", "私聊里深色主题")
    db.set_resident("u12", "group:G2", "preference", "群里深色主题")
    check("常驻场景隔离：群词条不串进私聊",
          db.get_resident("u12", "private").get("preference") == "私聊里深色主题"
          and db.get_resident("u12", "group:G2").get("preference") == "群里深色主题")

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
    db.set_resident("u20", "private", "user", "下周要做手术，别在群里提")
    check("群场景常驻不含私聊词条（无 group:G20 场景）",
          db.get_resident("u20", "group:G20") == {}
          and db.get_resident("u20", "private").get("user"),
          str(db.resident_owners()))

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

    # --- v0.12.17 可见性：group 只在来源群生效，不能按人跨群泄漏
    ga = db.add_memory("preference", "A群约定每周五交作业", owner_id="u30",
                       group_id="GA", visibility="group")
    gb_hits = db.search_memories("每周五交作业", owner_id="u30", group_id="GB",
                                 platform="p", allow_private=False)
    check("跨群不召回他群 group 记忆",
          all(r["id"] != ga for r in gb_hits), str([r["id"] for r in gb_hits]))
    ga_hits = db.search_memories("每周五交作业", owner_id="u30", group_id="GA",
                                 platform="p", allow_private=False)
    check("本群可召回 group 记忆",
          any(r["id"] == ga for r in ga_hits), str([r["id"] for r in ga_hits]))
    db.set_resident("u30", "group:GA", "rules", "每周五交作业")
    check("常驻按群隔离：他群场景读不到本群词条",
          db.get_resident("u30", "group:GB") == {}
          and db.get_resident("u30", "group:GA").get("rules") == "每周五交作业",
          str(db.resident_owners()))
    pub = db.add_memory("preference", "公开偏好喝美式", owner_id="u30",
                        group_id="GA", visibility="public")
    check("public 记忆在其他群仍可见",
          any(r["id"] == pub for r in db.search_memories(
              "公开偏好喝美式", owner_id="u30", group_id="GB",
              platform="p", allow_private=False)))
    dm_priv = db.add_memory("preference", "只在私聊说过的生日是六月",
                            owner_id="u30", visibility="private")
    check("群聊搜不到 private", all(
        r["id"] != dm_priv for r in db.search_memories(
            "私聊说过的生日", owner_id="u30", group_id="GA",
            platform="p", allow_private=False)))
    check("私聊搜得到 private", any(
        r["id"] == dm_priv for r in db.search_memories(
            "私聊说过的生日", owner_id="u30", allow_private=True)))

    # --- v0.12.17 孤儿私聊记忆修复：owner 缺失时按会话发言人反推
    rdir = os.path.join(tmp, "repair")
    os.makedirs(rdir, exist_ok=True)
    r1 = HermesDB(rdir, "repair.db")
    r1.add_message("umo:FriendMessage:RAW9", "user", "你好", sender_id="RAW9")
    rid = r1.add_memory("fact", "孤儿私聊记忆", owner_id="",
                        session_id="umo:FriendMessage:RAW9", visibility="private")
    r1.close()
    r2 = HermesDB(rdir, "repair.db")
    fixed = r2.memory_by_id(rid)
    check("孤儿私聊记忆重挂 owner", fixed["owner_id"] == "RAW9", fixed["owner_id"])
    check("孤儿修复后 scope=user", fixed["scope"] == "user", fixed["scope"])
    r2.close()

    # --- v0.12.17 私聊作用域规范化：private 且无群 → 必须落在 user 层
    #     写坏的方式：总结时把 scope 判成 group、可见性却判成 private。
    #     分层检索下这种行两边都不落地（群层按可见性过滤掉、按人层 scope 不匹配）。
    sdir = os.path.join(tmp, "scopenorm")
    os.makedirs(sdir, exist_ok=True)
    n1 = HermesDB(sdir, "scope.db")
    bad_id = n1.add_memory("fact", "作用域写错的私聊记忆", scope="group",
                           owner_id="u70", visibility="private")
    check("写入时确实是 private+group 错位",
          n1.memory_by_id(bad_id)["scope"] == "group",
          n1.memory_by_id(bad_id)["scope"])
    n1.close()
    n2 = HermesDB(sdir, "scope.db")
    check("启动后 scope 规范为 user",
          n2.memory_by_id(bad_id)["scope"] == "user",
          n2.memory_by_id(bad_id)["scope"])
    check("规范后可被本人按人检索到",
          any(r["id"] == bad_id for r in n2.search_memories(
              "作用域写错的私聊记忆", owner_id="u70", allow_private=True)))
    n2.close()

    # --- v0.12.17 判不出查看者时不放行 private（原来的隐式 fail-open）
    check("无身份时检索不到他人 private",
          not any(r["id"] == dm_priv for r in db.search_memories(
              "私聊说过的生日", owner_id="", allow_private=True)))
    db.set_resident("u30", "private", "user", "生日是六月")
    check("无身份时读不到任何常驻（fail-closed）",
          db.get_resident("", "private") == {})
    check("有身份时本人才看得到自己的常驻",
          db.get_resident("u30", "private").get("user") == "生日是六月"
          and db.get_resident("u31", "private") == {})

    # --- v0.12.17 删除合并父条不能把子条永久悬空
    c1 = db.add_memory("fact", "子条一", owner_id="u50")
    c2 = db.add_memory("fact", "子条二", owner_id="u50")
    parent = db.merge_memories([c1, c2], "合并后的条目", "fact")
    check("合并产生新条目", parent > 0, str(parent))
    db.delete_memory(parent)
    check("删父条后子条恢复活跃",
          db.memory_by_id(c1)["merged_into"] == 0
          and db.memory_by_id(c2)["merged_into"] == 0)

    # --- v0.12.17 整理快照不再硬顶 1000 条
    base = len(db.all_active_memories())
    for i in range(1005):
        db.add_memory("fact", f"批量记忆 {i}", owner_id="u60")
    check("all_active_memories 不再硬顶 1000",
          len(db.all_active_memories()) >= base + 1005,
          f"{len(db.all_active_memories())} < {base + 1005}")
    check("active_memory_ids_since 只返回新增 id",
          len(db.active_memory_ids_since(0)) >= base + 1005)

    # --- 导入历史幂等依赖的消息签名
    check("message_signatures 覆盖已有消息",
          ("user", "新的保留消息") in db.message_signatures("s1"))

    # --- v0.12.18 虚词句召回：query 的每个 bigram 都含虚字时也必须能召回
    #     旧实现把文档侧的含虚字 bigram 也删掉，交集恒为空——「看我会怎」这种
    #     精确子串都召回 0（真实库 300 条消息抽查 25 条中招）。
    stop_id = db.add_memory("fact", "我不太好奇你想看我会怎么反应",
                            owner_id="u80", visibility="private")
    check("全虚词 bigram 的 query 能召回记忆（精确子串）",
          any(r["id"] == stop_id for r in db.search_memories(
              "看我会怎", owner_id="u80", allow_private=True)),
          str([r["id"] for r in db.search_memories(
              "看我会怎", owner_id="u80", allow_private=True)]))
    sid80 = "umo:FriendMessage:u80"
    mid80 = db.add_message(sid80, "user", "我只是在意你会不会一直陪着我")
    check("全虚词 bigram 的 query 能召回历史消息",
          any(r["id"] == mid80 for r in db.search_messages(
              "我只是一", session_id=sid80)),
          str([r["id"] for r in db.search_messages("我只是一", session_id=sid80)]))
    check("全虚词 query 不误伤无关文档",
          not db.search_messages("看我会怎", session_id=sid80))

    # --- v0.12.18 跨层去重：owner+group 都有的条目在多层命中时只返回一次
    dup_id = db.add_memory("preference", "跨层去重测试条目三角洲",
                           owner_id="u90", group_id="G90", visibility="group")
    hits = db.search_memories("跨层去重测试条目", limit=8, owner_id="u90",
                              group_id="G90", platform="", allow_private=False)
    check("同一条记忆不被多层重复返回",
          [r["id"] for r in hits].count(dup_id) == 1,
          str([r["id"] for r in hits]))

    # --- v0.12.20 运行期产生的孤儿私聊记忆：重启后必须也能修复。
    #     旧实现修复用 meta 标记只跑一次，第一次修复写标记后，
    #     「拿不到 sender_id 的适配器」在运行期继续产生的孤儿就永久沉默了。
    odir = os.path.join(tmp, "orphan2")
    os.makedirs(odir, exist_ok=True)
    o1 = HermesDB(odir, "o.db")
    o1.add_memory("fact", "启动期孤儿", scope="group", owner_id="",
                  session_id="p:FriendMessage:OA", platform="p", visibility="private")
    o1.add_message("p:FriendMessage:OA", "user", "hi", platform="p", sender_id="OA")
    o1.close()
    o2 = HermesDB(odir, "o.db")  # 第一次重启：修复
    late = o2.add_memory("fact", "运行期孤儿", scope="user", owner_id="",
                        session_id="p:FriendMessage:OB", platform="p",
                        visibility="private")
    o2.add_message("p:FriendMessage:OB", "user", "yo", platform="p", sender_id="OB")
    o2.close()
    o3 = HermesDB(odir, "o.db")  # 第二次重启：新孤儿也要被修复
    check("运行期孤儿重启后也重挂 owner",
          (o3.memory_by_id(late).get("owner_id") or "") == "OB",
          str(o3.memory_by_id(late).get("owner_id")))
    o3.close()

    # --- v0.12.20 失效会话游标清理：水位/退避键随会话数无限增长
    db.set_meta("summarized:dead_session", "99")
    db.set_meta("sumfail:dead_session", "{}")
    db.set_meta("summarized:s1", "5")  # s1 有消息，必须保留
    n_stale = db.prune_stale_session_meta()
    check("失效会话游标被清理",
          n_stale >= 2 and db.get_meta("summarized:dead_session") == ""
          and db.get_meta("sumfail:dead_session") == "",
          f"removed={n_stale}")
    check("活跃会话水位保留", db.get_meta("summarized:s1") == "5",
          db.get_meta("summarized:s1"))

    # --- 常驻词条按 owner 隔离：换个人一条都拿不到；整段改写覆盖旧文本
    p_owner = "u_pin_order"
    db.set_resident(p_owner, "private", "rules", "老规则文本")
    check("本人能读到自己的常驻词条",
          db.get_resident(p_owner, "private").get("rules") == "老规则文本")
    check("读不到别人的常驻词条",
          db.get_resident("u_other", "private") == {})
    check("整段改写覆盖旧文本",
          db.set_resident(p_owner, "private", "rules", "新规则文本") is True
          and db.get_resident(p_owner, "private").get("rules") == "新规则文本"
          and db.resident_count() == sum(o["count"] for o in db.resident_owners()))

    # --- v0.12.21 短 query 门槛：3 个实词只命中 1 个不该被硬编码的 0.5 砍掉
    q_docs = [{"id": 1, "content": "证书续期的处理办法要记下来", "kind": "fact"},
              {"id": 2, "content": "完全无关的晚饭吃了什么", "kind": "fact"}]
    short_hits = rc.rank("证书过期怎么办", q_docs, k=2)
    check("短 query 命中单个实词仍可召回",
          any(r["id"] == 1 for r in short_hits), str([r["id"] for r in short_hits]))
    check("放宽短档门槛后零命中噪声仍不混入",
          not any(r["id"] == 2 for r in short_hits), str([r["id"] for r in short_hits]))
    check("min_coverage 对短 query 生效",
          not rc.rank("证书过期怎么办", q_docs, k=2, min_coverage=0.9))

    # --- v0.12.21 重复抑制：连续追问不能只剩 1 条，也不能让后排永远饿死
    its = [{"id": i} for i in range(1, 6)]
    rep_seen: dict = {}
    counts, touched = [], set()
    for turn in range(1, 8):
        fresh = rc.suppress_repeats(its, rep_seen, turn)
        for x in fresh:
            rep_seen[x["id"]] = turn
        counts.append(len(fresh))
        touched.update(x["id"] for x in fresh)
    check("连续追问注入量不低于半数", min(counts) >= 3, str(counts))
    check("重复抑制会轮转，不饿死后排记忆", len(touched) >= 4, str(sorted(touched)))
    check("首次注入仍给满", counts[0] == 5, str(counts[0]))

    # --- v0.12.21 去重池：visible=private/public 不受 group_id 限制，group 仍按群隔离
    dedup_owner = "u_dedup_scope"
    same = "用户喜欢在深夜听爵士乐放松"
    a_id = db.add_memory("preference", same, owner_id=dedup_owner,
                         group_id="GX", visibility="private")
    b_id, b_new = db.add_memory_smart("preference", same, owner_id=dedup_owner,
                                      group_id="", visibility="private")
    check("群里沉淀的 private 与私聊同句复用而非重复建条",
          (not b_new) and b_id == a_id, f"{b_id} created={b_new}")
    g1 = db.add_memory("preference", "只在甲群成立的偏好测试内容",
                       owner_id=dedup_owner, group_id="G1", visibility="group")
    g2, g2_new = db.add_memory_smart("preference", "只在甲群成立的偏好测试内容",
                                     owner_id=dedup_owner, group_id="G2",
                                     visibility="group")
    check("group 可见性仍按来源群隔离（防跨群合并回归）",
          g2_new and g2 != g1, f"{g2} created={g2_new}")

    # --- v0.12.21 群层老数据兼容分支：group_id='' 靠 session_id 认群的行
    legacy_sid = "p:GroupMessage:GLEGACY"
    legacy = db.add_memory("fact", "老版本群共享记忆：本群约定的周会时间点",
                           owner_id="", session_id=legacy_sid, platform="p",
                           group_id="", visibility="group")
    legacy_priv = db.add_memory("fact", "老版本群里的私聊内容周会时间点",
                                owner_id="u_legacy", session_id=legacy_sid,
                                platform="p", group_id="", visibility="private")
    lg = [r["id"] for r in db.search_memories(
        "周会时间点", limit=8, session_id=legacy_sid, platform="p",
        group_id="GLEGACY", viewer_owner="u_legacy", allow_private=False)]
    check("老群共享记忆（无 group_id）在本群可召回", legacy in lg, str(lg))
    check("群聊仍读不到老数据里的 private", legacy_priv not in lg, str(lg))
    lg_other = [r["id"] for r in db.search_memories(
        "周会时间点", limit=8, session_id="p:GroupMessage:OTHER", platform="p",
        group_id="GOTHER", viewer_owner="", allow_private=False)]
    check("老群共享记忆不会跨群泄漏", legacy not in lg_other, str(lg_other))

    # --- v0.12.21 unmerge 中间节点：子条重挂祖父，不能被一律置 0 造成内容重复
    chain_owner = "u_unmerge_chain"
    c1 = db.add_memory("fact", "合并链甲内容之一", owner_id=chain_owner,
                       visibility="private")
    c2 = db.add_memory("fact", "合并链甲内容之二", owner_id=chain_owner,
                       visibility="private")
    cc = db.merge_memories([c1, c2], "一次合并产物", "fact")
    cd = db.add_memory("fact", "合并链丁内容", owner_id=chain_owner,
                       visibility="private")
    ce = db.merge_memories([cc, cd], "二次合并产物", "fact")
    db.unmerge(cc)
    check("unmerge 中间节点后子条重挂祖父",
          int(db.memory_by_id(c1)["merged_into"]) == ce
          and int(db.memory_by_id(c2)["merged_into"]) == ce,
          f"c1={db.memory_by_id(c1)['merged_into']} c2={db.memory_by_id(c2)['merged_into']}")
    check("撤销祖父可一次恢复全部来源条", db.unmerge(ce) >= 3,
          str([int(db.memory_by_id(x)["merged_into"]) for x in (c1, c2, cd)]))

    # --- 归档与常驻两层独立：supersede 只动归档条目，词条文本不受牵连
    r1 = db.add_memory("profile", "用户是某大学在校生", owner_id="u50",
                       visibility="private")
    arch = db.add_memory("profile", "用户 2026 年 6 月毕业", owner_id="u50",
                         visibility="private")
    db.set_resident("u50", "private", "user", "某大学在校生")
    check("supersede 只作用于归档条目", db.supersede_memories([r1], arch) == 1)
    check("被顶替的归档条目已退场",
          int(db.memory_by_id(r1)["merged_into"] or 0) != 0)
    check("supersede 不碰常驻词条",
          db.get_resident("u50", "private").get("user") == "某大学在校生")

    # 归档写入不再挂词条；同义内容在归档池照常去重
    r3, made = db.add_memory_smart(
        "profile", "用户已毕业，2026 年 6 月拿到学位", scope="user",
        source="llm_tool", owner_id="u50", visibility="private", prefer_new=True)
    check("前台写入落进归档层（block 遗留列为空）",
          made is True and db.memory_by_id(r3)["block"] == "")
    dup, cdup = db.add_memory_smart(
        "profile", "用户已毕业，2026 年 6 月拿到学位", scope="user",
        source="auto_summary", owner_id="u50", visibility="private")
    check("后台重复记录被归档去重吸收", cdup is False and dup == r3, f"{dup},{cdup}")

    # --- v1.2.0 一次性迁移：memories.block 条目 → resident 表六词条文本
    #
    # 全新库首次建表时迁移已在空 memories 上跑过并写了 resident_text_v1 标记，
    # 要模拟旧库必须先删标记、插旧式行，再重开库触发迁移。
    mdir = os.path.join(tmp, "resident_mig")
    os.makedirs(mdir, exist_ok=True)
    m0 = HermesDB(mdir, "m.db")
    m0.conn.execute("DELETE FROM meta WHERE key = 'resident_text_v1'")

    def _old_row(owner, gid, vis, blk, content):
        m0.conn.execute(
            "INSERT INTO memories(ts, kind, scope, content, source, owner_id,"
            " session_id, platform, group_id, visibility, block)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), "preference", "user", content, "test", owner,
             "", "qq", gid, vis, blk))

    _old_row("um1", "", "private", "preference", "喜欢喝美式")
    _old_row("um1", "", "private", "preference", "喜欢喝美式")   # 近重复，应被去重
    _old_row("um1", "", "private", "preference", "喝美式不加糖")
    _old_row("um1", "G7", "group", "rules", "本群只聊工作")
    _old_row("um1", "", "public", "address", "喊我哥们")
    _old_row("", "", "public", "rules", "无归属老条目")
    _old_row("um2", "", "private", "user", "设计师")
    m0.conn.commit()
    m0.close()
    m1db = HermesDB(mdir, "m.db")            # 重开：触发迁移
    mig1 = m1db.get_resident("um1", "private")
    check("迁移：近重复聚合为一段文本",
          mig1.get("preference") == "喜欢喝美式；喝美式不加糖", str(mig1))
    check("迁移：public 场景归回私聊层", mig1.get("address") == "喊我哥们", str(mig1))
    check("迁移：群场景写进 group:<gid>",
          m1db.get_resident("um1", "group:G7").get("rules") == "本群只聊工作",
          str(m1db.resident_owners()))
    check("迁移：无归属条目不挂任何人词条",
          m1db.get_resident("", "private") == {})
    left_blk = m1db.conn.execute(
        "SELECT COUNT(*) c FROM memories WHERE COALESCE(block, '') != ''"
    ).fetchone()["c"]
    check("迁移：原行 block 全部清空", left_blk == 0, str(left_blk))
    n_act = m1db.conn.execute(
        "SELECT COUNT(*) c FROM memories WHERE merged_into = 0").fetchone()["c"]
    check("迁移：原行保留在归档层（退回归档）", n_act == 7, str(n_act))
    m1db.close()
    m2db = HermesDB(mdir, "m.db")            # 再开一次：幂等
    check("迁移：幂等重跑不重复写入",
          m2db.get_resident("um1", "private").get("preference") == "喜欢喝美式；喝美式不加糖"
          and m2db.resident_count() == 4, str(m2db.resident_count()))
    m2db.close()

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
