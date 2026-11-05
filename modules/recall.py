"""纯本地召回：字符二元组 + IDF 加权 + 时间衰减 + MMR 去冗余。

为什么不用 FTS5 / LIKE：
中文没有空格，SQLite 的 trigram 分词器要求「整段连续命中」，把用户原句
`这两天全球股市咋样` 塞进 MATCH 或 LIKE，几乎不可能恰好是记忆的子串
（实测 10 条真实消息 0 命中）。而 bigram 把它切成
`这两/两天/天全/全球/球股/股市/市咋/咋样`，只要部分重叠就能打分排序，
再交给模型判断——这才是「检索」而不是「精确包含判断」。

设计约束：
- 不碰数据库、不花 token、不发网络请求，纯函数，方便单测；
- 打分只看本批候选集的 IDF，不依赖全库统计，作用域过滤在外层完成；
- 时间衰减：旧记忆不该永远压住新的，但也不能直接丢；
- MMR：同一件事有多条近似记忆时只留一条，省注入预算。

这个模块只负责「排序」，不做权限判断——谁能看到哪些记忆由 db 层决定。
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Iterable

_SPLIT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")

# 类型权重：稳定的画像/偏好比一次性事件更值得注入
KIND_WEIGHT = {
    "profile": 1.18,
    "preference": 1.12,
    "fact": 1.0,
    "conclusion": 1.0,
    "plan": 0.95,
    "summary": 0.92,
    "event": 0.88,
}

# 检索命中率低时的兜底：query 太短没有区分度，直接不检索
MIN_QUERY_CHARS = 2
# 覆盖率低于此值视为噪声。门槛只在「内容词」上算（见 content_grams）——
# 用全量 bigram 做分母时，一句正常问句会被「的/怎么/时候」这些虚词稀释到 0.3 以下，
# 实测「部署的时候证书报错怎么办」正好 0.27 被误杀。
MIN_COVERAGE = 0.30
# MMR 去冗余阈值：两条记忆相似到这个程度就只留分高的那条
DEDUP_SIM = 0.72

# 虚词字符集：含这些字的 bigram 不计入覆盖率分母。
# 只收纯功能字，刻意不包含「时/候/用/看/上/下/要/会/能/想/说」——
# 它们常出现在实词里（用 winget、看日志、后端、上海）。
_STOP_CHARS = set(
    "的了嘛吗呢吧啊呀哦嗯我你您他她它们是在有和就都也还这那个什么怎"
    "请帮把被给跟对从而且或之其与及呗哈呃诶"
)
# 与 _STOP_CHARS 等价的字符类，用于高频判断（见 _content_grams_of）
_STOP_RE = re.compile("[" + "".join(sorted(_STOP_CHARS)) + "]")


def norm(text: str) -> str:
    """去掉标点与空白，统一小写。中文按字保留。"""
    return _SPLIT_RE.sub("", str(text or "").lower())


def grams(text: str) -> list[str]:
    """有序去重的字符二元组。单字文本退化为单字。"""
    s = norm(text)
    if not s:
        return []
    if len(s) == 1:
        return [s]
    return list(dict.fromkeys(s[i: i + 2] for i in range(len(s) - 1)))


def gram_set(text: str, normalized: bool = False) -> set[str]:
    """只要集合时走这条：省掉 dict.fromkeys + list 的中间对象。

    批量排序里每条文档只需要集合（IDF 统计、交集、去冗余都是集合运算），
    实测比 grams() 快近一倍。normalized=True 表示传入的已经是 norm() 过的文本
    （排序里本来就要算 norm 做精确包含加分，再归一化一次纯属浪费）。
    """
    s = text if normalized else norm(text)
    if not s:
        return set()
    if len(s) == 1:
        return {s}
    return {s[i: i + 2] for i in range(len(s) - 1)}


def fts_windows(text: str, cap: int = 48) -> list[str]:
    """把查询切成 3 字窗口，供 FTS5 trigram 做 OR 检索。

    trigram 至少要 3 个字符才可能命中，所以「整句短语」查询在自然语言下
    几乎必然 0 命中（实测 12 条常见查询里 10 条命中 0），快路径等于常年失效。
    这里两路取窗口：
    - 归一化后的整句：抓连续表达的长片段；
    - 再按空白/标点分段各取一次：`群晖 续期` 这种被空格隔开的关键词，
      归一化会把空格吃掉，跨段窗口在原文里并不存在，只有分段才能命中。
    合并去重并限长：贴日志/长报错时 MATCH 表达式不至于爆炸。
    """
    wins: list[str] = []
    # 分段在前：分段窗口是用户真正输入的词，最不容易误伤；
    # 整句窗口补「连续表达」，被 cap 截断时优先保住分段词。
    for seg in _SPLIT_RE.split(str(text or "").lower()):
        if len(seg) >= 3:
            wins.extend(seg[i: i + 3] for i in range(len(seg) - 2))
    s = norm(text)
    if len(s) >= 3:
        wins.extend(s[i: i + 3] for i in range(len(s) - 2))
    return list(dict.fromkeys(wins))[:cap]


def jaccard(a: str, b: str) -> float:
    A, B = set(grams(a)), set(grams(b))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return inter / (len(A) + len(B) - inter)


def _jaccard_sets(A: set[str], B: set[str]) -> float:
    """已算好 bigram 集合的版本。排序内部走这条，避免每条候选反复分词。"""
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return inter / (len(A) + len(B) - inter)


def content_grams(text: str) -> list[str]:
    """只保留实词二元组，用于「这条相不相关」的门槛判断。

    虚词构成的双字词（的时/时候/怎么/么办/这个）对判断相关性没有信息量，
    却会把分母撑大。全被过滤时退回原始 bigram，避免空分母。
    """
    return _content_grams_of(grams(text))


def _content_grams_of(gs: list[str]) -> list[str]:
    """content_grams 的内核：接收已算好的 bigram，避免重复分词。

    用预编译正则做「含虚词字符」判断：原写法是每个 gram 都跑一次
    `any(ch in _STOP_CHARS for ch in g)`，五千条记忆实测这一项就占排序
    三分之一时间，而正则的字符类匹配在 C 层完成。
    """
    out = [g for g in gs if not _STOP_RE.search(g)]
    return out or gs


def build_idf(docs: Iterable[str]) -> dict[str, float]:
    """按候选集统计 IDF。某个 gram 越少见，命中它越有说服力。"""
    n = 0
    df: dict[str, int] = {}
    for d in docs:
        n += 1
        for g in set(grams(d)):
            df[g] = df.get(g, 0) + 1
    if not n:
        return {}
    return {g: math.log(1.0 + n / (1.0 + c)) for g, c in df.items()}


def _coverage(qg: list[str], dg: set[str],
              qg_set: set[str] | None = None) -> float:
    """query 的 gram 有多大比例被文档覆盖（不看权重）。

    只用来做「是否算相关」的门槛：如果按 IDF 加权判断，一个极罕见的双字词
    （`说过`）就能把 8 个完全不匹配的词压下去，导致 0.4 的高分噪声。
    门槛必须朴素，排序才用权重。
    """
    if not qg:
        return 0.0
    qs = qg_set if qg_set is not None else set(qg)
    return len(qs & dg) / len(qs)


def _weighted_coverage(qg: list[str], dg: set[str], idf: dict[str, float]) -> float:
    """按 IDF 加权的覆盖率，用于打分：命中稀有词比命中常用词更有说服力。"""
    total = 0.0
    hit = 0.0
    for g in qg:
        w = idf.get(g, 1.0)
        total += w
        if g in dg:
            hit += w
    return (hit / total) if total > 0 else 0.0


def score(query: str, doc: str, idf: dict[str, float], *,
          qg: list[str] | None = None, ts: float = 0.0,
          kind: str = "", hits: int = 0,
          half_life_days: float = 180.0, now: float | None = None,
          doc_grams: set[str] | None = None,
          doc_norm: str | None = None,
          q_norm: str | None = None,
          qg_set: set[str] | None = None) -> float:
    """单条打分。0 表示不相关。

    doc_grams / doc_norm / q_norm 是给 rank 复用的预计算结果：批量排序时
    同一条文档会被「门槛判断 / 打分 / 去冗余」三次用到，每次重算 bigram
    等于把整批文本分词三遍（5000 条记忆实测 45ms，其中 2/3 是重复分词）。
    """
    if not query or not doc:
        return 0.0
    qg = qg if qg is not None else grams(query)
    if not qg:
        return 0.0
    dg = doc_grams if doc_grams is not None else set(grams(doc))
    if not dg:
        return 0.0
    cov = _coverage(qg, dg, qg_set)
    if cov <= 0:
        return 0.0
    wcov = _weighted_coverage(qg, dg, idf)

    # 密度项：长文档天然更容易撞到 gram，用 sqrt(len) 归一化后只占小权重。
    # 交集与 _coverage 用同一份 query 集合，不再每条文档重建一次 set(qg)。
    qset = qg_set if qg_set is not None else set(qg)
    density = min(1.0, len(qset & dg) / math.sqrt(len(dg)))
    base = wcov * 0.62 + cov * 0.16 + density * 0.22

    # 精确包含：用户问「computer-power 技能」时，正文里就有这个词，应该重排到最前
    bonus = 1.0
    nq = q_norm if q_norm is not None else norm(query)
    nd = doc_norm if doc_norm is not None else norm(doc)
    if len(nq) >= 2 and nq in nd:
        bonus = 1.35

    # 时间衰减：半衰期 180 天。没有 ts 的条目按不过期处理
    decay = 1.0
    if ts:
        age_days = max(0.0, ((now or time.time()) - float(ts)) / 86400.0)
        decay = 0.5 ** (age_days / max(1.0, half_life_days))

    # 被反复召回并确认有用的记忆，权重略高（封顶 +25%，避免马太效应）
    hit_w = 1.0 + min(0.25, math.log1p(max(0, int(hits or 0))) * 0.05)
    return base * bonus * decay * KIND_WEIGHT.get(str(kind or ""), 1.0) * hit_w


def rank(query: str, docs: list[dict[str, Any]], *, k: int = 5,
         text_key: str = "content", ts_key: str = "ts",
         kind_key: str = "kind", hits_key: str = "hits",
         half_life_days: float = 180.0, min_coverage: float = MIN_COVERAGE,
         dedup: bool = True, now: float | None = None) -> list[dict]:
    """按相关度返回前 k 条（每条原样附加 `_score` 字段）。

    会先剔掉「明显不相关」的候选，再做 MMR 去冗余：如果两条记忆几乎一样，
    只保留分高的那条，避免 5 条注入里 3 条是同一个意思。
    """
    if len(norm(query)) < MIN_QUERY_CHARS:
        return []
    qg = grams(query)
    qc = content_grams(query)
    if not qg or not qc:
        return []
    qc_set = set(qc)
    qg_set = set(qg)
    q_norm = norm(query)

    # 每条候选只分词一次，后面门槛判断、打分、去冗余全都复用这份结果。
    prepared: list[tuple[dict, str, set[str], str]] = []
    df: dict[str, int] = {}
    for d in docs:
        text = str(d.get(text_key, ""))
        ntext = norm(text)
        gset = gram_set(ntext, normalized=True)
        prepared.append((d, text, gset, ntext))
        # 只统计 query 里出现过的 gram：_weighted_coverage 只查 qg，
        # 其余 gram 统计了也永远没人查。原来对每条候选的每个 bigram 都记一次
        # df（1500 条 ≈ 14 万次 dict 写 + 等量 math.log），99% 是纯浪费。
        for g in gset & qg_set:
            df[g] = df.get(g, 0) + 1
    n = len(prepared)
    idf = ({g: math.log(1.0 + n / (1.0 + c)) for g, c in df.items()}
           if n else {})

    scored: list[tuple[float, dict, set[str]]] = []
    for d, text, gset, d_norm in prepared:
        # 粗筛：连 gram 都不相交的文档不可能过门槛（精确集合是 gset 的子集）。
        # 只有少数候选会走到正则过滤，避免为几千条不相关文档付过滤成本。
        if not (qc_set & gset):
            continue
        # 门槛只该问「query 的实词在文档里出没出现」，文档侧不能再做虚词过滤。
        # qc 非空时其中的 gram 本就不含虚字，过滤文档侧对它毫无影响；
        # 真正受影响的是 qc 全被过滤、退回全量 bigram 的情况——「看我会怎」
        # 「你记得我吗」这类每个二字组都带虚字的句子。此时把文档侧的对应 bigram
        # 也删掉，交集恒为空，哪怕 query 是原文的精确子串也召回 0。
        # 真实生产库 300 条消息抽查里有 25 条正是这样丢的。
        hit_n = len(qc_set & gset)
        cov = hit_n / len(qc)
        # 双档门槛：
        # - 短 query（≤3 个实词）命中一个强词即可，否则「这个报错怎么修」永远进不来；
        # - 长 query 要求至少 2 个实词命中，防止一个烂大街的词把噪声拉进注入。
        if len(qc) <= 3:
            ok = hit_n >= 1 and cov >= 0.5
        else:
            ok = hit_n >= 2 and cov >= min_coverage
        if not ok:
            continue
        s = score(query, text, idf, qg=qg, ts=float(d.get(ts_key) or 0),
                  kind=str(d.get(kind_key) or ""), hits=int(d.get(hits_key) or 0),
                  half_life_days=half_life_days, now=now,
                  doc_grams=gset, doc_norm=d_norm, q_norm=q_norm,
                  qg_set=qg_set)
        if s <= 0:
            continue
        scored.append((s, d, gset))

    scored.sort(key=lambda x: -x[0])

    out: list[dict] = []
    kept_grams: list[set[str]] = []
    for s, d, gset in scored:
        if dedup and any(_jaccard_sets(gset, kg) >= DEDUP_SIM for kg in kept_grams):
            continue
        item = dict(d)
        item["_score"] = round(s, 4)
        out.append(item)
        kept_grams.append(gset)
        if len(out) >= k:
            break
    return out


def rank_messages(query: str, docs: list[dict[str, Any]], *, k: int = 3,
                  now: float | None = None) -> list[dict]:
    """历史消息排序。消息没有 kind，衰减也更快（聊过的话时效性更强）。"""
    return rank(query, docs, k=k, half_life_days=90.0, now=now)


def context_text(item: Any) -> tuple[str, str]:
    """从 AstrBot 的 contexts 条目里取 (role, text)。

    兼容 dict / 对象两种形态，以及多模态的 parts 列表——不同适配器结构不同，
    取错就会把「继续」补全成空字符串。
    """
    if isinstance(item, dict):
        role = str(item.get("role") or "")
        content = item.get("content")
    else:
        role = str(getattr(item, "role", "") or "")
        content = getattr(item, "content", "")
    if isinstance(content, (list, tuple)):
        buf: list[str] = []
        for p in content:
            if isinstance(p, dict):
                buf.append(str(p.get("text") or ""))
            else:
                buf.append(str(getattr(p, "text", "") or p))
        content = " ".join(buf)
    return role.lower(), str(content or "")


# 追击/指代线索：出现这些词说明这句在「接着说」，需要上一轮的话题才能检索。
# 刻意不收「怎么办/怎么修」——它们也常出现在自带话题的完整问句里。
_CUE_RE = re.compile(r"继续|然后|接着|还是|不行|这个|那个|刚才|刚刚|上面|再来|再试|呢$")


def expand_query(query: str, contexts: Any, *, short_chars: int = 8,
                 prev_chars: int = 120) -> str:
    """短句/指代句用上一条用户发言补全检索词。

    「继续」「然后呢」「这个怎么修」本身零信息，拿它当 query 只能召回 0 条，
    等于跨会话直接断片。补上一条用户发言就能带上话题。
    自带话题的长句不动——补全会把上一个话题的噪声带进来。
    只影响检索，不改变注入内容。
    """
    q = str(query or "").strip()
    if not q:
        return q
    nq = norm(q)
    if len(nq) > int(short_chars) and not (_CUE_RE.search(q) and len(nq) <= 14):
        return q
    try:
        items = list(contexts or [])
    except (TypeError, ValueError):
        return q
    for item in reversed(items):
        role, text = context_text(item)
        if role != "user":
            continue
        text = text.strip()
        if not text or text == q or text.startswith("/"):
            continue
        # 上一条也是「继续/嗯/？」，补了等于没补，继续往前找有信息的那条
        if len(norm(text)) < 4:
            continue
        return f"{text[:int(prev_chars)]} {q}"
    return q


def filter_in_window(history: list[dict[str, Any]],
                     window_ids: set[int]) -> list[dict[str, Any]]:
    """剔除已在当前上下文窗口内的历史命中。

    这些原始消息本轮请求本来就带着（req.contexts），再注入一份纯属重复计费。
    window_ids 为空（拿不到窗口信息）时原样返回——宁可多注不可漏注。
    """
    if not window_ids or not history:
        return history
    return [h for h in history if int(h["id"]) not in window_ids]


# 注入块的分组标题。kind 是内部字段，模型只需要「画像 / 偏好」这一层区别；
# 逐条写 (profile) 既占字符预算，又容易和正文混在一起被当成内容的一部分。
KIND_TITLES = (("profile", "画像"), ("preference", "偏好"),
               ("fact", "事实"), ("summary", "近况"))


def format_memory_lines(items: list[dict[str, Any]], *,
                        note_key: str = "note") -> list[str]:
    """把记忆条目渲染成注入用的几行：类型分组 + 编号后置。

    - 编号必须保留：pin_memory / save_memory(replaces=) 全靠它定位条目。
    - 编号放行尾而不是行首：正文优先，行首不再出现 #，客户端按 markdown
      渲染时也不会把 # 当成标记吃掉（截图里 `#127` 就变成了 `127`）。
    - kind 提到分组标题：6 条平铺时逐条 `(profile)` 是纯开销（实测省 52 字）。
    - note 用于「钉于 2026-08」这类附注，由调用方按需填。
    """
    groups: dict[str, list[dict]] = {}
    for it in items or []:
        groups.setdefault(str(it.get("kind") or "fact"), []).append(it)
    order = [k for k, _ in KIND_TITLES]
    order += [k for k in groups if k not in order]   # 未知 kind 也不丢
    titles = dict(KIND_TITLES)
    multi = sum(1 for k in order if groups.get(k)) > 1
    out: list[str] = []
    last_label = ""
    for kind in order:
        rows = groups.get(kind) or []
        if not rows:
            continue
        label = titles.get(kind, "其他")
        if multi and label != last_label:
            out.append(f"{label}：")
            last_label = label
        for it in rows:
            text = str(it.get("content") or "").strip()
            if not text:
                continue
            note = str(it.get(note_key) or "").strip()
            tail = f"（#{it['id']}" + (f"，{note}" if note else "") + "）"
            out.append(f"- {text}{tail}")
    return out


def suppress_repeats(items: list[dict[str, Any]], seen: dict[int, int],
                     cur_turn: int, gap: int = 4) -> list[dict[str, Any]]:
    """重复抑制：近 gap 轮注入过的记忆本轮跳过。

    注入走临时 part 不进会话历史，下一轮模型确实看不到，所以注入本身不能省；
    但相邻几轮查询措辞相近时同几条记忆反复入选是浪费。全被抑制时保留排名
    最高的第一条，避免相关记忆彻底断供。
    """
    if not items:
        return items
    fresh = [x for x in items if cur_turn - seen.get(int(x["id"]), -10**9) >= gap]
    if not fresh:
        fresh = items[:1]
    return fresh
