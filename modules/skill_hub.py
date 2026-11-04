"""技能中枢：把「AstrBot 本体装的技能」和「插件自己进化出来的技能」合成一个视图。

为什么需要这一层
----------------
本体有一份技能目录（`<data>/skills/<slug>/SKILL.md`，带 scripts/references，全体共享），
插件自己库里也有一份技能（从对话里提炼的经验条目，短小、会反复合并）。
在模型眼里这两个东西都叫「技能」，于是用户说“改一下 xx 技能”时，它往往去改本体
那份的文件——因为那是磁盘上真实存在、还能执行的东西。

但本体那份不能给插件写：
1. 它是全局共享的，按人写入会互相踩；
2. 它可能带 scripts/*.ps1、node_modules，插件覆盖会直接毁掉可执行性；
3. 它可能由市场/用户安装，更新时会被覆盖，插件的内容会丢。

所以分成两个域，写权限严格分层：

    System   本体目录，只读；插件只读取、绝不写入
    Personal 插件 SQLite，按 owner 隔离，可写可合并

同名时个人版覆盖系统版（shadow）。用户说“改 computer-power”时，模型读到系统版，
想改只能在个人域建一份同名副本；之后读取优先命中个人版，系统原版毫发无损，
而且每个人都可以有自己的那一份。
"""

import os
import re
import time

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)
MAX_SKILL_BYTES = 64 * 1024


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 SKILL.md 头部的 --- 键值块，返回 (meta, body)。

    只认最简单的 `key: value`，不引第三方 yaml —— 技能文件本来就是这么写的。
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, val = line.partition(":")
        meta[key.strip().lower()] = val.strip().strip("'\"")
    return meta, text[m.end():]


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line:
            return line[:60]
    return ""


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def norm_name(name: str) -> str:
    """技能名归一：连字符、下划线、空格视为同一个词。

    模型一会写 `computer-power`、一会写 `computer_power`，不归一的话同一个人
    会攒出两份"同名"技能，判断覆盖关系时也会漏。
    """
    return re.sub(r"[\s_\-]+", "-", str(name or "").strip().lower())


class SkillHub:
    def __init__(self, db, system_root: str):
        self.db = db
        self.system_root = system_root
        self._cache: dict[str, tuple[float, dict]] = {}
        self._listed_at = 0.0
        self._index: list[dict] = []

    # ---------- 系统技能（只读） ----------

    def _system_dirs(self) -> list[str]:
        try:
            return [
                os.path.join(self.system_root, d)
                for d in os.listdir(self.system_root)
                if not d.startswith(".") and os.path.isdir(os.path.join(self.system_root, d))
            ]
        except OSError:
            return []

    def _load_system(self, path: str) -> dict | None:
        md = os.path.join(path, "SKILL.md")
        try:
            mtime = os.path.getmtime(md)
        except OSError:
            return None
        slug = os.path.basename(path)
        cached = self._cache.get(slug)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            with open(md, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read(MAX_SKILL_BYTES)
        except OSError:
            return None
        meta, body = _parse_frontmatter(raw)
        item = {
            "slug": slug,
            "name": meta.get("name") or slug,
            "display": meta.get("name") or slug,
            "trigger": meta.get("description") or _first_heading(body) or "通用",
            "content": raw,
            "path": path,
            "mtime": mtime,
            "has_scripts": os.path.isdir(os.path.join(path, "scripts")),
            "source": "system",
            "readonly": True,
        }
        self._cache[slug] = (mtime, item)
        return item

    def system_skills(self, force: bool = False) -> list[dict]:
        now = time.time()
        if not force and self._index and now - self._listed_at < 10:
            return self._index
        out: list[dict] = []
        seen: set[str] = set()
        for path in sorted(self._system_dirs()):
            item = self._load_system(path)
            if not item:
                continue
            for key in {item["slug"], item["name"]}:
                seen.add(key)
            out.append(item)
        self._index = out
        self._listed_at = now
        return out

    def find_system(self, name: str, force: bool = False) -> dict | None:
        key = str(name or "").strip()
        if not key:
            return None
        low = key.lower()
        items = self.system_skills(force=force)
        for item in items:
            if low in {item["slug"].lower(), item["name"].lower()}:
                return item
        for item in items:
            if low in item["slug"].lower() or low in item["name"].lower():
                return item
        return None

    # ---------- 个人技能 ----------

    def personal_skills(self, owner_id: str = "", group_id: str = "") -> list[dict]:
        rows = self.db.list_skills(owner_id=owner_id, group_id=group_id)
        out = []
        for r in rows:
            r = dict(r)
            r["source"] = "personal"
            r["readonly"] = False
            out.append(r)
        return out

    def list_for(self, owner_id: str = "", group_id: str = "") -> list[dict]:
        """当前身份能看到的全部技能，个人在前、系统在后，同名的系统版标 shadowed。"""
        personal = self.personal_skills(owner_id, group_id)
        mine = {norm_name(p["name"]) for p in personal}
        systems = []
        for s in self.system_skills():
            if norm_name(s["name"]) in mine or norm_name(s["slug"]) in mine:
                continue
            systems.append(s)
        return personal + systems

    def _find_personal(self, name: str, owner_id: str, group_id: str) -> dict | None:
        """先精确查，再按归一化名找（computer-power / computer_power 视为同一技能）。"""
        row = self.db.peek_skill(name, owner_id=owner_id, group_id=group_id)
        if row:
            return dict(row)
        want = norm_name(name)
        if not want:
            return None
        for s in self.personal_skills(owner_id, group_id):
            if norm_name(s["name"]) == want:
                return dict(self.db.peek_skill(s["name"], owner_id=owner_id,
                                               group_id=group_id) or s)
        return None

    def read(self, name: str, owner_id: str = "", group_id: str = "") -> dict | None:
        """个人优先；没有再看系统。传 `system:xxx` 可强制读系统原版。

        返回时带 source/readonly；个人版若存在同名系统技能，额外标 shadow_of，
        调用方可以据此提示“还有一份更完整的本体版可参考”。
        """
        key = str(name or "").strip()
        if not key:
            return None
        force_system = key.lower().startswith("system:")
        if force_system:
            key = key.split(":", 1)[1].strip()
        if not force_system:
            row = self._find_personal(key, owner_id, group_id)
            if row:
                row = dict(row)
                row["source"] = "personal"
                row["readonly"] = False
                self.db.get_skill(row["name"], owner_id=owner_id, group_id=group_id)
                sys_item = self.find_system(key)
                if sys_item:
                    row["shadow_of"] = sys_item["name"]
                    row["system_bytes"] = len(sys_item.get("content") or "")
                return row
        item = self.find_system(key)
        return dict(item) if item else None

    def save(self, name: str, trigger: str, content: str, *, owner_id: str = "",
             group_id: str = "", scope: str | None = None) -> dict:
        """写个人技能。若与系统技能同名，等于建了一份覆盖副本。"""
        row = self._find_personal(name, owner_id, group_id)
        if row:  # 已有同义名，复用它，避免 computer-power / computer_power 两份并存
            name = row["name"]
        created = self.db.upsert_skill(name, trigger or "", content,
                                      owner_id=owner_id, scope=scope,
                                      group_id=group_id, origin="hermes")
        shadow = self.find_system(name)
        return {"name": name, "created": created, "shadow_of": (shadow or {}).get("name", "")}

    def update(self, name: str, old_text: str, new_text: str, *,
               owner_id: str = "", group_id: str = "",
               scope: str | None = None) -> dict:
        """在技能正文里做局部替换，比让模型重写整份安全得多。

        - old_text 为空：追加到末尾
        - 找不到 / 命中多处：报错并说明次数，让调用方给更长的上下文
        - 只有系统版本：先自动复制一份个人副本，再在副本上改，系统原版不动
        """
        key = str(name or "").strip()
        if not key:
            return {"ok": False, "error": "技能名不能为空"}
        row = self._find_personal(key, owner_id, group_id)
        is_system = False
        if not row:
            sys_item = self.find_system(key)
            if not sys_item:
                return {"ok": False, "error": f"没有找到技能 {key}"}
            # 系统技能不能直接改。先在它正文的基础上算出新版本，替换成功后再落一份
            # 个人副本；替换失败就什么也不写，不留半成品。
            row = {"name": sys_item["name"], "trigger": sys_item["trigger"],
                   "content": sys_item["content"], "origin": "shadow"}
            is_system = True

        body = str(row.get("content") or "")
        old = str(old_text or "")
        new = str(new_text or "")
        if not old:
            merged = body.rstrip() + "\n\n" + new.strip() + "\n"
        else:
            hits = body.count(old)
            if hits == 0:
                return {"ok": False, "error": "技能正文里没有找到这段原文，请先读取技能再精确复制要替换的片段"}
            if hits > 1:
                return {"ok": False,
                        "error": f"这段原文在技能里出现了 {hits} 次，请带上更多上下文让它唯一"}
            merged = body.replace(old, new, 1)
        if len(merged.encode("utf-8")) > MAX_SKILL_BYTES:
            return {"ok": False, "error": "改完超过 64KB，请精简"}

        self.db.upsert_skill(row["name"], str(row.get("trigger") or ""), merged,
                             owner_id=owner_id, scope=scope, group_id=group_id,
                             origin="shadow" if is_system else str(row.get("origin") or "hermes"))
        return {"ok": True, "name": row["name"], "shadowed": is_system,
                "shadow_of": row["name"] if is_system else ""}

    # ---------- 展示 ----------

    @staticmethod
    def describe(items: list[dict], owner_id: str = "") -> str:
        """把技能列表渲染成给模型看的紧凑文本，标明归属与可写性。"""
        personal = [i for i in items if i.get("source") == "personal"]
        system = [i for i in items if i.get("source") == "system"]
        lines: list[str] = []
        if personal:
            lines.append("[我的技能]（可读可改，save_skill / update_skill 都作用于这里）")
            for s in personal:
                tag = "私" if s.get("scope") == "private" else ("群" if s.get("scope") == "group" else "公")
                lines.append(f"- {s['name']}（{_clip(s.get('trigger') or '通用', 60)}）[{tag}]")
        if system:
            if lines:
                lines.append("")
            lines.append("[系统技能·只读]（AstrBot 本体安装的，插件不写入；要改成自己的版本，"
                         "直接 save_skill 同名保存即可生成个人副本）")
            for s in system:
                extra = " 带脚本" if s.get("has_scripts") else ""
                lines.append(f"- {s['name']}（{_clip(s.get('trigger') or '通用', 60)}{extra}）")
        return "\n".join(lines) if lines else "暂无技能。"

    def stats(self) -> dict:
        systems = self.system_skills()
        return {
            "personal": len(self.db.all_skills()),
            "system": len(systems),
            "system_with_scripts": sum(1 for s in systems if s.get("has_scripts")),
            "root": self.system_root,
        }


def guess_data_root(plugin_data_dir: str) -> str:
    """从插件数据目录反推 AstrBot 的 data 目录（里面有 skills/ 和 cmd_config.json）。"""
    cur = os.path.abspath(plugin_data_dir)
    for _ in range(5):
        cur = os.path.dirname(cur)
        if not cur or cur == os.path.dirname(cur):
            break
        # 仅存在性探测，用于向上定位 AstrBot 数据根，不读取文件内容
        if os.path.isdir(os.path.join(cur, "skills")) or \
                os.path.exists(os.path.join(cur, "cmd_config.json")):
            return cur
    return os.path.dirname(os.path.dirname(os.path.abspath(plugin_data_dir)))
