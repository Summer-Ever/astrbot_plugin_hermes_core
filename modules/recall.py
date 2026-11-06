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
#
# 适用范围：**只有 rank() 用**（历史消息召回，见 rank_messages）。记忆检索
# 自 v1.0.2 起改走 rank_fused()，不再用覆盖率一票否决——原因见下方
# 「双路融合召回」整段注释。改这里不会影响记忆召回的召回率。
MIN_COVERAGE = 0.30
# 长 query 的覆盖率下限，比短档松一档。
#
# 与 MIN_COVERAGE 一样，**只对 rank() 生效**，记忆检索已改用 rank_fused()。
# 保留这段历史是因为它记录了「调阈值」这条路的尽头：下面是 v1.0.1 把 0.30
# 放宽到 0.25 时的依据，但下一版就证明这条思路走不通——门槛的分母是问句长度，
# 只要问句再长两个词，同一条记忆照样会被砍（#73 是 0.250 压线过、0.200 被砍）。
#
# v1.0.1 之前长档的门槛是「命中 ≥2 个实词 且 覆盖率 ≥0.30」，短档是「命中 ≥1
# 个且覆盖率 ≥0.30」。分档依据是 query 的实词总数，于是同一句问话加两个
# 无所谓的前缀（「我喜欢简洁」→「记住我喜欢简洁」）实词数就从 3 变 4、跨过长档，
# 命中数要求从 1 跳到 2，一条明明相关的记忆直接被砍掉——用户感受就是
# 「明明存过却想不起来」。长档多出来的实词多半是「记住/帮我/请问/什么」这类
# 框架词，不是检索实体，拿它们充数当分母本来就不合理。
#
# 0.25 这个数不是拍的：用生产规模的候选集（2600 条）离线评测了十一套门槛方案，
# 短档完全不动、只放宽长档的这套，相关 query 的 Recall@5 从 15/16 提到 16/16，
# 噪声 query 的 top5 平均无关条目只从 1.36 条涨到 1.79 条。同批评测里，
# 「统一放宽全部档位」「按 IDF 加权」「覆盖率分母只算候选集出现过的词」
# 几套要么噪声涨得更多、要么 Recall 掉得更多，见 .workbuddy/hermes-supertest/
# d04_eval.py。
MIN_COVERAGE_LONG = 0.25
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
         min_coverage_long: float = MIN_COVERAGE_LONG,
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
        # - 长 query 同样命中一个即可，但覆盖率下限松一档（MIN_COVERAGE_LONG）。
        # 短档的覆盖率下限走 min_coverage（默认 0.30）：原来硬编码 0.5，
        # 3 个实词只命中 1 个时 cov=0.33 被拒——注释写着「命中一个强词即可」，
        # 实现却要求至少 2 个，实测「证书过期怎么办」↔「证书续期的处理办法」
        # 这种明明相关、只共享一个实词的记忆永远召不回。噪声由 IDF 加权分数
        # 和 MMR 兜住，门槛不该在这里把候选直接砍死。
        #
        # 长档原来要求「命中 ≥2 个实词」，那是为了挡「一个烂大街的词把噪声拉进
        # 注入」。可它挡不住该挡的：实测真正决定放不放行的是覆盖率下限，
        # 命中数从 2 改成 1 对噪声几乎没有影响（离线评测里噪声通过数一条没变），
        # 却让「加两个字就丢召回」变成了常态。噪声交给排序和 MMR 兜，
        # 门槛只保证「至少有一个实词真的出现了」。
        if len(qc) <= 3:
            ok = hit_n >= 1 and cov >= min_coverage
        else:
            ok = hit_n >= 1 and cov >= min_coverage_long
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


# ==================== 双路融合召回 ====================
#
# v1.0.2 起记忆检索走 rank_fused()，rank() 只保留给历史消息（rank_messages）。
#
# 为什么换：rank() 是「单路 bigram 打分 + 硬门槛」，门槛一票否决。实测事故——
# #73「助手具备本地文件系统读写权限，曾在桌面新建 test.txt 并写入 1234」
# 全文 40 字，能被问句命中的实词永远只有「桌面」「文件」两个，于是
#   cov = 2 / len(问句实词组数)
# 短问句「记得我让你在桌面上创建过一个文件吗」实词 8 组 → cov=0.250 压线通过；
# 同一句后面只补「，为什么没召回」变成 10 组 → cov=0.200，被 MIN_COVERAGE_LONG
# 整条删掉。记忆没变、问句只多六个字，结果从答对变成答不出。
#
# 问题不在阈值取值，而在分母的含义：它衡量「问句里有多少字能在记忆里找到」，
# 而不是「这条记忆是不是用户在问的那件事」。问句越长分母越大，短记忆必然被稀释，
# 调阈值只能决定谁被误杀。Mem0 / Zep 的做法都是**不靠常数砍候选**：多路召回 →
# 融合 → 精排 → 取 top-k，去留由排序决定。这里照此改造，分三段：
#
#   1) 双路：A 路沿用 bigram/IDF（语义近似），B 路 BM25（稀疏精确，自带长度惩罚）；
#   2) 融合：RRF 只用名次挑候选池，池内再用 min-max 归一后的加权分精排；
#   3) 闸门：只在「要不要注入」这一层用绝对判据，且不删除候选本身。
#
# 为什么 B 路必须主导：同一长问句下 BM25 给 #73 10.77、给噪声冠军 #143 5.54
# （差 94%）；而 A 路的 bigram 打分是 0.2359 vs 0.2599——噪声反超。A 路分不开
# 长文档与目标，只能做辅助路。
#
# 为什么不建 FTS 索引：memories 没有 FTS 表，且 SQLite trigram 分词器对中文
# 二字词（「桌面」「葡萄」）必然失效（见 fts_windows 的说明）。BM25 在这里
# 直接用字符二元组当 term，不需要额外索引，也不会引入第二份数据要同步。

BM25_K1 = 1.2   # 词频饱和点：tf 再高，贡献也不会无限涨
BM25_B = 0.75   # 长度惩罚强度：0=不惩罚，1=完全按 avgdl 归一
RRF_K = 60      # RRF 平滑常数，IR 界惯例值
FUSION_POOL = 15   # 精排候选池大小：RRF 只负责不漏，池内才真排序
FUSION_W_A = 0.35  # A 路（bigram）权重
FUSION_W_B = 0.65  # B 路（BM25）权重——实测分得更开，主导排序
FUSION_W_IMP = 0.10  # importance 权重（当前是代理值，见 importance_proxy）
FUSION_REL_RATIO = 0.70  # 相对截断：低于池内最高分该比例的条目不再注入
GATE_COV_LONG = 0.10   # 长问句的覆盖率地板：只用来挡「一个实词都没沾上」
GATE_COV_SHORT = 0.30  # 短问句（≤3 个实词）仍需较强覆盖

# 「可核对的具体信息」：数字、常见文件后缀、路径分隔符、反引号专名。
_CONCRETE_RE = re.compile(r"\d|\.(?:txt|json|yaml|yml|py|md|zip|db|log)\b|[/\\]|`[^`]+`")


def bm25_scores(query: str, docs: list[dict[str, Any]], *,
                text_key: str = "content",
                prepared: list[tuple[dict, str, set[str]]] | None = None
                ) -> list[float]:
    """手写 BM25，term 用字符二元组。

    长度归一用 norm 后的字符数：中文没有词边界，字符数是最稳的代理，而长文档
    稀释恰好是靠这一项压下去的——噪声冠军 #143 是 329 字长条目，正文里含
    「文件/系统/权限」，任何不带长度惩罚的打分都会让它压过 40 字的 #73。

    prepared 可传入已归一化的 (doc, norm_text, gram_set)，避免同一批候选被
    分词两遍（A 路预处理已经算过一次）。

    性能取舍：df 用 gram_set 判定而不是子串扫描——二元组 t 出现在文本里，
    当且仅当 t ∈ gram_set(文本)，两者等价，但哈希查找是 O(1)。tf 仍用
    str.count（BM25 需要真实词频，近似成 0/1 会改变已验证的排序），但只在
    已确认命中该 term 的文档上算，绝大多数文档在这里就跳过了。
    """
    terms = grams(query)
    if not terms or not docs:
        return [0.0] * len(docs)
    if prepared is not None and len(prepared) == len(docs):
        normed = [p[1] for p in prepared]
        gsets = [p[2] for p in prepared]
    else:
        normed = [norm(d.get(text_key, "")) for d in docs]
        gsets = [gram_set(t, normalized=True) for t in normed]

    lens = [len(t) or 1 for t in normed]
    avgdl = sum(lens) / len(lens)
    n = len(docs)
    # query 只有一个字时 gram 是单字，不在 gram_set 里，退回普通包含判断
    single = len(terms[0]) == 1

    df: dict[str, int] = {}
    for t in terms:
        if single:
            df[t] = sum(1 for s in normed if t in s)
        else:
            df[t] = sum(1 for gs in gsets if t in gs)
    # idf 给下限而不是 0：query 里一个笔误不该把整条候选压成 0 分，
    # 那样闸门之外的排序信息就全丢了。
    idf = {t: max(0.05, math.log(1.0 + (n - c + 0.5) / (c + 0.5)))
           for t, c in df.items()}

    out: list[float] = []
    for s, dl, gs in zip(normed, lens, gsets):
        sc = 0.0
        for t in terms:
            if not (t in s if single else t in gs):
                continue
            tf = s.count(t)
            if not tf:
                continue
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
            sc += idf[t] * (tf * (BM25_K1 + 1)) / denom
        out.append(sc)
    return out


def importance_proxy(doc: dict[str, Any]) -> float:
    """importance 的**代理值**，不是真值。

    Generative Agents 的做法是写入时让 LLM 打 1-10 分，这样「重要但久远」的
    记忆才能和「琐碎但新鲜」的分开。我们的 memories 表有 importance 字段，
    但写入侧从来没写过它（实测 47/50 条是 0），recall 也从来没读过。

    在真值接上之前，这里用「具体性 + 使用频率」做代理，只为让这一维先参与
    排序。诚实地说：离线评测里加与不加 MRR 都是 0.775，**当前无实测增量**。
    保留它是因为它是未来接写入侧 LLM 打分的挂载点，且权重要小到不改变
    已验证的排序（FUSION_W_IMP=0.10 时对 top-5 无影响）。
    """
    text = str(doc.get("content") or "")
    s = 0.0
    if _CONCRETE_RE.search(text):
        s += 0.30
    n = len(text)
    if 20 <= n <= 200:
        s += 0.20
    elif n < 20:
        s += 0.05
    s += min(0.25, math.log1p(int(doc.get("hits") or 0)) * 0.06)
    if int(doc.get("pinned") or 0):
        s += 0.25
    return min(1.0, s)


_LATIN_SEG_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/\-]*$")


def _hit_units(query: str, doc_grams: set[str], doc_norm: str) -> int:
    """query 侧**检索单元**的命中数，用于闸门的「命中几个实词」。

    为什么要按单元而不是 bigram 个数：一个英文词会被切成多个 bigram。
    `python` → py/yt/th/ho/on，只要文档里有个 `nemotron`（含 on）和
    `OpenRouter`（含 er/ou…），就能凑出「命中 2 个以上实词」的假象。
    实测「用 Python 实现一个二叉树遍历」正是这样注入了 #143（329 字的
    配置清单，只因为 nemotron 里有个 `on`）。

    所以计数单位回到 query 这一侧：
    - 拉丁/数字词（`python`、`test.txt`、`v1.0.2`）按**整词**算 1 个单元，
      整词在文档里出现才算命中，拆开的碎片不算；
    - 中文（没有词边界）按 bigram 算，`关系图` 命中 关系+系图 记 2 个。

    为什么中文不能也按整段算：中文没有空格，「画关系图用什么工具比较好」
    整句是一段，按整段算则永远只有 0 或 1 个单元，复合词 `关系图` 会被
    误判成「只命中一个词」而挡掉真相关的条目（实测 Recall 从 0.85 掉到 0.65）。
    """
    total = 0
    for seg in _SPLIT_RE.split(str(query or "").lower()):
        if len(seg) <= 1:
            continue
        if _LATIN_SEG_RE.match(seg):
            if seg in doc_norm:
                total += 1
            continue
        total += len(gram_set(seg, normalized=True) & doc_grams)
    return total


def _query_unit_count(query: str, qc_set: set[str]) -> int:
    """query 的检索单元总数，用于闸门分档（短查询 / 长查询）。

    两个坑都必须避开：

    1. **不能拿全量 bigram 数分档**。`记忆是怎么存储的` 全量 bigram 有 7 个，
       但过滤掉虚词后只有 3 个实词单元——按 7 分档会把它判成「长问句」，
       于是要求命中 2 个单元，一条只命中「存储」的真相关记忆就被挡了。
       实测这样 Recall 会从 0.850 掉到 0.800。
    2. **一 个英文词不能被算成多个单元**。`ffmpeg` 有 6 个 bigram，
       按 bigram 数判成长问句后要求命中 2 个单元——而它本身只有一个单元，
       永远过不了。实测模型用 search_memory 只查 `ffmpeg` / `Graphviz`
       这类关键词时，新排序返回空，比旧排序还差。

    所以：中文段按「与覆盖率同口径的实词 bigram 数」计，拉丁段整词计 1。
    """
    total = 0
    for seg in _SPLIT_RE.split(str(query or "").lower()):
        if len(seg) <= 1:
            continue
        if _LATIN_SEG_RE.match(seg):
            total += 1
            continue
        total += len(gram_set(seg, normalized=True) & qc_set)
    return max(1, total)


def gate_ok(units: int, hit_n: int, cov: float) -> bool:
    """注入闸门：判断「这次要不要把这条注入给模型」。

    与 rank() 里的硬门槛有两点本质差别：

    1. 只在**注入决定**上生效。候选池的构造和排序结果都完整保留，被拦下的
       条目下一轮换个问法还能出来；而 rank() 是在排序前就把候选删掉。
    2. 分档口径反过来。长问句不再要求高覆盖率（分母被话题词撑大，要求高
       必然误杀 #73 这种短记忆），改成要求**命中 ≥2 个检索单元**；覆盖率
       下限反而降到 0.10，只用来挡「一个实词都没沾上」的纯噪声。

    units 是 query 的检索单元总数（见 _query_unit_count），不是 bigram 个数：
    短查询（≤3 个单元，含只打一个关键词的情况）只要命中就能注入，
    否则「只查一个词」会被自己的长度判据挡死。

    为什么必须有绝对判据：双路融合能解决长问句稀释，但挡不住「短查询 +
    罕见 token 巧合」。实测「1+1 等于几」会把 #177 里的「Windows 11」当成
    强命中（「11」在库里罕见、那条只有 39 字，BM25 给高分），
    「帮我写一段快速排序」会撞上 #180 的「一段话」。这类误注入靠相对阈值
    截不住——top1 本身就是垃圾，相对阈值等于给它背书。
    """
    if hit_n <= 0:
        return False
    if units <= 3:
        return cov >= GATE_COV_SHORT
    return hit_n >= 2 and cov >= GATE_COV_LONG


def rank_fused(query: str, docs: list[dict[str, Any]], *, k: int = 5,
               rrf_k: int = RRF_K, pool: int = FUSION_POOL,
               w_a: float = FUSION_W_A, w_b: float = FUSION_W_B,
               w_imp: float = FUSION_W_IMP,
               rel_ratio: float = FUSION_REL_RATIO,
               gate: bool = True, dedup: bool = True,
               now: float | None = None, text_key: str = "content",
               explain: bool = False) -> list[dict]:
    """双路召回 + RRF 选池 + 加法精排。返回带 `_score` 的前 k 条。

    三段结构，每段解决一个具体问题：

    **第一段 双路 RRF 选池。** 两路量纲完全不同（bigram 分 0~1.5、BM25 分 0~30），
    直接相加没有意义。RRF 用 `1/(k+rank)` 只看名次，天然解决量纲问题。
    但它**不能当置信度**：候选只有几十条时相邻名次差不到 3%，实测同一批
    候选里 top1=0.600、其余全在 0.73~0.96 之间，分数被压平到无法截断。
    所以 RRF 只用来「不漏掉任何一路看好的条目」。

    **第二段 池内精排。** 取 RRF 最高的 pool 条，池内对两路分别做 min-max
    归一（保留相对差距，不让名次抹平），再按 w_a/w_b 加权求和。排序交回
    有绝对意义的 BM25，这是全部提升的来源。

    **第三段 注入闸门。** 见 gate_ok：只在注入层做绝对判据，不删候选。
    rel_ratio 是相对截断，衡量「重要性悬崖」——低于池内首条该比例的条目
    不再注入；它对候选池本身无影响，所以不会像原门槛那样整条删记忆。

    与 rank() 的另一个差别：MMR 去冗余放在最后（跨整个池），而不是边排序
    边去重，这样「两条近似记忆」里留下的总是分数更高的那条。
    """
    if len(norm(query)) < MIN_QUERY_CHARS:
        return []
    if not docs:
        return []
    query = str(query or "")
    now = now if now is not None else time.time()
    qg = grams(query)
    qc = content_grams(query)
    if not qg or not qc:
        return []
    qg_set = set(qg)
    qc_set = set(qc)
    q_norm = norm(query)

    # 每条候选只归一化一次：A 路打分、B 路 BM25、闸门、MMR 四处复用。
    prepared: list[tuple[dict, str, set[str]]] = []
    for d in docs:
        ntext = norm(d.get(text_key, ""))
        prepared.append((d, ntext, gram_set(ntext, normalized=True)))
    n = len(prepared)

    # ---- A 路：bigram/IDF ----
    df: dict[str, int] = {}
    for _d, _t, gset in prepared:
        for g in gset & qg_set:
            df[g] = df.get(g, 0) + 1
    idf = {g: math.log(1.0 + n / (1.0 + c)) for g, c in df.items()}

    a_scores: list[float] = []
    for d, ntext, gset in prepared:
        # 粗筛：与实词零交集的文档，score() 内部也会因 cov<=0 返回 0，
        # 这里提前跳过省掉整次打分（大量候选走到这一步都是 0 分）。
        if not (qc_set & gset):
            a_scores.append(0.0)
            continue
        a_scores.append(score(
            query, str(d.get(text_key, "")), idf, qg=qg,
            ts=float(d.get("ts") or 0), kind=str(d.get("kind") or ""),
            hits=int(d.get("hits") or 0), now=now,
            doc_grams=gset, doc_norm=ntext, q_norm=q_norm, qg_set=qg_set))

    # ---- B 路：BM25 ----
    b_scores = bm25_scores(query, docs, text_key=text_key, prepared=prepared)

    # ---- RRF：只用名次，两路量纲不同也能合 ----
    def _ranks(vals: list[float]) -> list[int]:
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        rk = [0] * len(vals)
        for pos, i in enumerate(order, 1):
            rk[i] = pos
        return rk

    rk_a, rk_b = _ranks(a_scores), _ranks(b_scores)
    fused = [1.0 / (rrf_k + rk_a[i]) + 1.0 / (rrf_k + rk_b[i])
             for i in range(n)]

    # ---- 第一段：候选池 ----
    pool_n = max(k, min(int(pool), n))
    cand = sorted(range(n), key=lambda i: -fused[i])[:pool_n]

    # ---- 第二段：池内归一后精排 ----
    def _minmax(vals: list[float], idx: list[int]) -> dict[int, float]:
        v = [vals[i] for i in idx]
        lo, hi = min(v), max(v)
        span = hi - lo
        return {i: (0.0 if span <= 0 else (vals[i] - lo) / span) for i in idx}

    a_n = _minmax(a_scores, cand)
    b_n = _minmax(b_scores, cand)

    scored: list[dict[str, Any]] = []
    for i in cand:
        d, _ntext, gset = prepared[i]
        hit_n = len(qc_set & gset)
        scored.append({
            "i": i,
            "total": (w_a * a_n[i] + w_b * b_n[i]
                      + w_imp * importance_proxy(d)),
            "hit_n": hit_n,
            # 闸门用「检索单元命中数」而不是 bigram 个数，理由见 _hit_units：
            # 一个英文词会被切成多个 bigram，按个数算等于把同一个词数成 5 个。
            "seg_n": _hit_units(query, gset, _ntext),
            "cov": (hit_n / len(qc)) if qc else 0.0,
            "a": a_scores[i], "b": b_scores[i],
            "rk_a": rk_a[i], "rk_b": rk_b[i], "d": d,
        })
    scored.sort(key=lambda x: -x["total"])

    # ---- 第三段：闸门 → 相对截断 → MMR ----
    # 单元总数与候选无关，循环外算一次（分档口径见 _query_unit_count：
    # 按全量 bigram 数会把 `ffmpeg` 判成长问句、把 3 个实词的短问句也判成长问句）
    unit_total = _query_unit_count(query, qc_set)
    out: list[dict] = []
    kept_grams: list[set[str]] = []
    base = 0.0
    for x in scored:
        if gate and not gate_ok(unit_total, x["seg_n"], x["cov"]):
            continue
        if not out:
            # 首条通过闸门的条目即「满分基准」：若以被闸门拦下的 top1 为基准，
            # 剩下的条目会被一个不该存在的分数过度压制。
            base = x["total"]
        elif rel_ratio > 0 and x["total"] < base * rel_ratio:
            continue
        gset = prepared[x["i"]][2]
        if dedup and any(_jaccard_sets(gset, kg) >= DEDUP_SIM
                         for kg in kept_grams):
            continue
        item = dict(x["d"])
        item["_score"] = round(x["total"], 4)
        if explain:
            item["_dbg"] = {"a": round(x["a"], 4), "b": round(x["b"], 3),
                            "rk_a": x["rk_a"], "rk_b": x["rk_b"],
                            "hit": x["hit_n"], "seg": x["seg_n"],
                            "cov": round(x["cov"], 3)}
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
    """重复抑制：近 gap 轮注入过的记忆本轮降权，但不断供。

    注入走临时 part 不进会话历史，下一轮模型确实看不到，所以**不能把已注入
    过的记忆整批省掉**——原实现 `fresh` 为空时才保底留 items[:1]，实测连续
    追问同一话题时第 2~4 轮只剩 1 条（5 条相关记忆被砍掉 4 条），而模型上下文
    里一条都没有，等于静默失忆。

    折中：按原排名补足到至少一半席位（向上取整）。被抑制的条目排在 fresh
    之后、且名次不变，既压住「同样几条反复刷屏」，又保住召回量。
    """
    if not items:
        return items
    fresh: list[dict] = []
    held: list[tuple[int, int, dict]] = []
    for pos, x in enumerate(items):
        last = seen.get(int(x["id"]), -10**9)
        if cur_turn - last >= gap:
            fresh.append(x)
        else:
            held.append((last, pos, x))
    floor = max(1, (len(items) + 1) // 2)
    if len(fresh) < floor:
        # 补进来的必须是「最久没注入过」的那几条，而不是原排名最前的几条：
        # 否则 seen 每轮被刷新后永远是同一批占着席位，后排记忆彻底饿死。
        held.sort(key=lambda t: (t[0], t[1]))
        fresh.extend(x for _, _, x in held[:floor - len(fresh)])
    return fresh
