/* Hermes 记忆系统 · 页面逻辑
   通过 AstrBotPluginView 暴露的 apiGet/apiPost 调后端（webapi.py）。
   不引入任何第三方库，所有渲染都在本地完成。 */

(() => {
  "use strict";

  // 注意：AstrBot 把 bridge-sdk.js 注入在 </body> 前，也就是本脚本之后才执行，
  // 所以这里不能把 bridge 缓存成常量，必须每次惰性读取。
  const getBridge = () => window.AstrBotPluginView || window.AstrBotPluginPage || null;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const TAB_META = {
    overview: ["概览", "记忆、技能与自动整理的运行状态"],
    memories: ["记忆", "检索、编辑、合并；合并走软删除，可回溯"],
    skills: ["技能", "查看与维护模型沉淀下来的可复用流程"],
    settings: ["设置", "直接修改插件配置，与 AstrBot 配置页同一份文件"],
    tools: ["工具", "手动总结、整理、导出与身份绑定"],
  };

  const KIND_LABEL = {
    fact: "事实",
    preference: "偏好",
    profile: "画像",
    summary: "摘要",
  };
  const SCOPE_LABEL = { user: "按用户", group: "按群", global: "全局" };

  const state = {
    tab: "overview",
    overview: null,
    mem: { q: "", scope: "", kind: "", merged: false, limit: 50, offset: 0, total: 0, rows: [], selected: new Set() },
    skills: { rows: [], filter: "", current: null, preview: false },
    settings: { groups: [], dirty: false },
    aliases: [],
  };

  let modalSeq = 0;

  /* ---------------- 基础工具 ---------------- */

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtNum(n) {
    return Number(n || 0).toLocaleString("zh-CN");
  }

  function fmtTime(ts) {
    if (!ts) return "—";
    const d = new Date(Number(ts) * 1000);
    if (Number.isNaN(d.getTime())) return "—";
    const p = (x) => String(x).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  function fmtBytes(n) {
    n = Number(n || 0);
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(2)} MB`;
  }

  function errMsg(e) {
    if (!e) return "未知错误";
    if (typeof e === "string") return e;
    return e.message || String(e);
  }

  let toastTimer = null;
  function toast(message, kind = "ok") {
    const el = $("#toast");
    el.textContent = message;
    el.className = `toast show ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = "toast"; }, 3200);
  }

  function setBusy(btn, busy, busyText) {
    if (!btn) return;
    if (busy) {
      btn.dataset.label = btn.textContent;
      btn.disabled = true;
      btn.innerHTML = `<span class="spin"></span>${esc(busyText || "处理中")}`;
    } else {
      btn.disabled = false;
      btn.textContent = btn.dataset.label || btn.textContent;
      delete btn.dataset.label;
    }
  }

  async function apiGet(path, params) {
    const bridge = getBridge();
    if (!bridge) throw new Error("没有连接到 AstrBot 面板");
    const data = await bridge.apiGet(path, params || {});
    return normalize(data);
  }

  async function apiPost(path, body) {
    const bridge = getBridge();
    if (!bridge) throw new Error("没有连接到 AstrBot 面板");
    const data = await bridge.apiPost(path, body || {});
    return normalize(data);
  }

  function normalize(data) {
    if (data && typeof data === "object" && data.status === "error") {
      throw new Error(data.message || "接口返回错误");
    }
    return data || {};
  }

  /* ---------------- 弹窗 ---------------- */

  function openModal(opts) {
    const id = `modal_${++modalSeq}`;
    const mask = document.createElement("div");
    mask.className = "modal-mask";
    mask.dataset.modalId = id;
    mask.innerHTML = `
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="${id}_title">
        <h3 id="${id}_title">${esc(opts.title)}</h3>
        ${opts.sub ? `<div class="sub">${esc(opts.sub)}</div>` : ""}
        <div class="modal-body">${opts.bodyHtml || ""}</div>
        <div class="modal-foot">
          <button class="btn ghost" data-act="cancel">${esc(opts.cancelText || "取消")}</button>
          <button class="btn ${opts.danger ? "danger" : "primary"}" data-act="ok">${esc(opts.okText || "确定")}</button>
        </div>
      </div>`;
    document.body.appendChild(mask);
    const okBtn = $('[data-act="ok"]', mask);
    const close = () => { mask.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);

    mask.addEventListener("click", (e) => {
      if (e.target === mask) close();
      if (e.target.closest('[data-act="cancel"]')) close();
      if (e.target.closest('[data-act="ok"]')) {
        Promise.resolve(opts.onOk ? opts.onOk(mask, close) : null)
          .then((keep) => { if (keep !== false) close(); })
          .catch((err) => toast(errMsg(err), "err"));
      }
    });
    if (opts.onMount) opts.onMount(mask);
    return { el: mask, close };
  }

  /* ---------------- 极简 Markdown 渲染 ---------------- */

  function inlineMd(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  }

  function renderMarkdown(src) {
    const lines = esc(src || "").split("\n");
    const out = [];
    let inCode = false;
    let codeBuf = [];
    let listTag = null;

    const closeList = () => { if (listTag) { out.push(`</${listTag}>`); listTag = null; } };

    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (/^```/.test(line.trim())) {
        if (inCode) { out.push(`<pre><code>${codeBuf.join("\n")}</code></pre>`); codeBuf = []; inCode = false; }
        else { closeList(); inCode = true; }
        continue;
      }
      if (inCode) { codeBuf.push(line); continue; }

      const h = /^(#{1,4})\s+(.*)$/.exec(line);
      if (h) { closeList(); const n = h[1].length; out.push(`<h${n}>${inlineMd(h[2])}</h${n}>`); continue; }

      const ul = /^\s*[-*+]\s+(.*)$/.exec(line);
      const ol = /^\s*\d+\.\s+(.*)$/.exec(line);
      if (ul || ol) {
        const want = ul ? "ul" : "ol";
        if (listTag !== want) { closeList(); out.push(`<${want}>`); listTag = want; }
        out.push(`<li>${inlineMd((ul || ol)[1])}</li>`);
        continue;
      }

      if (!line.trim()) { closeList(); continue; }
      closeList();
      out.push(`<p>${inlineMd(line)}</p>`);
    }
    if (inCode && codeBuf.length) out.push(`<pre><code>${codeBuf.join("\n")}</code></pre>`);
    closeList();
    return out.join("\n");
  }

  /* ---------------- 标签页 ---------------- */

  function switchTab(name) {
    if (!TAB_META[name]) return;
    if (state.tab === "settings" && name !== "settings" && state.settings.dirty) {
      if (!window.confirm("设置还有未保存的修改，确定离开？")) return;
    }
    state.tab = name;
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
    const [title, sub] = TAB_META[name];
    $("#page-title").textContent = title;
    $("#page-sub").textContent = sub;
    if (name === "overview") loadOverview();
    if (name === "memories") loadMemories();
    if (name === "skills") loadSkills();
    if (name === "settings") loadSettings();
    if (name === "tools") { loadAliases(); }
  }

  /* ---------------- 概览 ---------------- */

  async function loadOverview() {
    try {
      const data = await apiGet("overview");
      state.overview = data;
      renderOverview(data);
    } catch (err) {
      toast(`读取概览失败：${errMsg(err)}`, "err");
    }
  }

  function renderOverview(d) {
    const s = d.stats || {};
    $("#side-version").textContent = d.version ? `v${d.version}` : "—";

    $("#stat-messages").textContent = fmtNum(s.messages);
    $("#stat-sessions").textContent = `${fmtNum(s.sessions)} 个会话`;
    $("#stat-memories").textContent = fmtNum(s.memories);
    $("#stat-merged").textContent = `已合并 ${fmtNum(s.merged)} 条`;
    $("#stat-skills").textContent = fmtNum(s.skills);

    const kinds = d.kinds || {};
    const kindText = Object.keys(kinds).length
      ? Object.entries(kinds).map(([k, v]) => `${KIND_LABEL[k] || k} ${v}`).join(" · ")
      : "暂无记忆类型";
    $("#stat-kinds").textContent = kindText;

    const q = d.queue || {};
    $("#stat-queue").textContent = q.memories_every > 0 && q.auto_memories
      ? `${q.memories}/${q.memories_every}`
      : "关闭";
    const sm = d.summarize || {};
    $("#stat-pending").textContent = `待总结 ${fmtNum(sm.pending_messages)} 条`;
    $("#side-queue-mem").textContent = q.memories_every > 0 ? `${q.memories}/${q.memories_every}` : "关闭";
    $("#side-queue-skill").textContent = q.skills_every > 0 ? `${q.skills}/${q.skills_every}` : "关闭";
    $("#run-version").textContent = `v${d.version || "?"} · code ${d.code_hash || "?"}`;

    $("#run-kv").innerHTML = [
      ["检索方式", d.fts ? '<span class="chip ok">FTS5 全文索引</span>' : '<span class="chip warn">LIKE 退化</span>'],
      ["数据库", `${fmtBytes(d.db_bytes)} <span class="faint">（含 WAL）</span>`],
      ["自动总结", sm.auto
        ? `每 ${sm.interval} 分钟检查；单会话满 ${sm.min_new} 条或超 ${sm.grace_hours} 小时兜底`
        : '<span class="chip warn">已关闭</span>'],
      ["待总结", `${fmtNum(sm.pending_sessions)} 个会话 / ${fmtNum(sm.pending_messages)} 条消息`],
      ["身份别名", `${fmtNum(d.aliases)} 条`],
      ["代码校验", `<span class="mono faint">${esc(d.code_hash || "—")}</span>`],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");

    const memPct = q.memories_every > 0 ? Math.min(100, (q.memories / q.memories_every) * 100) : 0;
    const skPct = q.skills_every > 0 ? Math.min(100, (q.skills / q.skills_every) * 100) : 0;
    $("#queue-box").innerHTML = `
      <div class="queue-row">
        <div class="row-between">
          <span>记忆</span>
          <span class="${q.auto_memories && q.memories_every > 0 ? "muted" : "faint"}">
            ${q.memories_every > 0 && q.auto_memories
              ? `${q.memories} / ${q.memories_every} 条`
              : "自动整理已关闭（可用工具页手动触发）"}
          </span>
        </div>
        <div class="bar"><i style="width:${memPct}%"></i></div>
      </div>
      <div class="queue-row">
        <div class="row-between">
          <span>技能</span>
          <span class="${q.skills_every > 0 ? "muted" : "faint"}">
            ${q.skills_every > 0 ? `${q.skills} / ${q.skills_every} 个` : "自动技能整理已关闭"}
          </span>
        </div>
        <div class="bar"><i style="width:${skPct}%"></i></div>
      </div>`;

    const recent = d.recent_memories || [];
    $("#recent-memories").innerHTML = recent.length
      ? recent.map((m) => `
        <div class="mini-item">
          <span class="idx">#${m.id}</span>
          <div class="txt">
            <p>${esc(m.content)}</p>
            <small>${esc(KIND_LABEL[m.kind] || m.kind)} · ${esc(SCOPE_LABEL[m.scope] || m.scope)} · ${fmtTime(m.ts)}</small>
          </div>
        </div>`).join("")
      : '<div class="empty">还没有记忆。聊几句后等自动总结，或去工具页手动总结。</div>';

    const skills = d.top_skills || [];
    $("#top-skills").innerHTML = skills.length
      ? skills.map((s, i) => `
        <div class="mini-item">
          <span class="idx">${i + 1}</span>
          <div class="txt">
            <p class="mono" style="font-size:12.5px">${esc(s.name)}</p>
            <small>${esc(s.trigger || "通用")} · 使用 ${fmtNum(s.use_count)} 次</small>
          </div>
        </div>`).join("")
      : '<div class="empty">还没有技能。完成一次复杂流程后会自动沉淀。</div>';
  }

  /* ---------------- 记忆 ---------------- */

  function memQuery() {
    const m = state.mem;
    return {
      q: m.q, scope: m.scope, kind: m.kind,
      merged: m.merged ? 1 : 0, limit: m.limit, offset: m.offset,
    };
  }

  async function loadMemories() {
    try {
      const data = await apiGet("memories", memQuery());
      state.mem.rows = data.records || [];
      state.mem.total = data.total || 0;
      renderMemories();
    } catch (err) {
      $("#mem-tbody").innerHTML = `<tr><td colspan="8"><div class="empty">加载失败：${esc(errMsg(err))}</div></td></tr>`;
    }
  }

  function renderMemories() {
    const m = state.mem;
    const rows = m.rows;
    $("#mem-total").textContent = `共 ${fmtNum(m.total)} 条${m.merged ? "（含已合并）" : ""}`;
    const pages = Math.max(1, Math.ceil(m.total / m.limit));
    $("#mem-page").textContent = `${Math.floor(m.offset / m.limit) + 1} / ${pages}`;
    $("#mem-prev").disabled = m.offset <= 0;
    $("#mem-next").disabled = m.offset + m.limit >= m.total;

    if (!rows.length) {
      $("#mem-tbody").innerHTML = '<tr><td colspan="8"><div class="empty">没有匹配的记忆</div></td></tr>';
      updateBulkBar();
      return;
    }

    $("#mem-tbody").innerHTML = rows.map((r) => {
      const owner = r.owner_id
        ? `<span class="mono" title="${esc(r.owner_id)}">${esc(String(r.owner_id).slice(0, 14))}</span>`
        : (r.group_id ? `<span class="muted">群 ${esc(String(r.group_id).slice(0, 10))}</span>` : '<span class="faint">全局</span>');
      const mergedChip = Number(r.merged_into) ? ' <span class="chip">已合并</span>' : "";
      return `
        <tr data-id="${r.id}">
          <td><input type="checkbox" class="checkbox mem-check" data-id="${r.id}" ${m.selected.has(r.id) ? "checked" : ""}></td>
          <td class="mono faint">${r.id}</td>
          <td><span class="chip blue">${esc(KIND_LABEL[r.kind] || r.kind)}</span></td>
          <td class="content"><div class="text">${esc(r.content)}</div>${mergedChip}</td>
          <td>${owner}</td>
          <td><span class="chip">${esc(r.source || "—")}</span></td>
          <td class="nowrap muted">${fmtTime(r.ts)}</td>
          <td>
            <div class="ops">
              <button class="btn tiny ghost" data-mem-edit="${r.id}">编辑</button>
              <button class="btn tiny danger" data-mem-del="${r.id}">删除</button>
            </div>
          </td>
        </tr>`;
    }).join("");

    $("#mem-check-all").checked = rows.every((r) => m.selected.has(r.id));
    updateBulkBar();
  }

  function updateBulkBar() {
    const n = state.mem.selected.size;
    $("#mem-bulk").classList.toggle("hidden", n === 0);
    $("#mem-sel-count").textContent = `已选 ${n} 条`;
    $("#mem-merge").disabled = n < 2;
  }

  function editMemoryModal(record) {
    openModal({
      title: `编辑记忆 #${record.id}`,
      sub: `${SCOPE_LABEL[record.scope] || record.scope}${record.owner_id ? ` · ${record.owner_id}` : ""} · ${fmtTime(record.ts)}`,
      bodyHtml: `
        <label class="field">
          <span class="label">类型</span>
          <select id="medit-kind">
            ${Object.entries(KIND_LABEL).map(([k, v]) => `<option value="${k}" ${record.kind === k ? "selected" : ""}>${v}</option>`).join("")}
          </select>
        </label>
        <label class="field">
          <span class="label">内容</span>
          <textarea id="medit-content" rows="5">${esc(record.content)}</textarea>
        </label>`,
      okText: "保存",
      onOk: async (mask) => {
        const content = $("#medit-content", mask).value.trim();
        if (!content) throw new Error("内容不能为空");
        await apiPost("memories/update", {
          id: record.id,
          content,
          kind: $("#medit-kind", mask).value,
        });
        toast("已保存");
        loadMemories();
        loadOverview();
      },
    });
  }

  function addMemoryModal() {
    openModal({
      title: "新增记忆",
      sub: "手动写入的记忆同样会进入自动整理队列",
      bodyHtml: `
        <label class="field">
          <span class="label">内容</span>
          <textarea id="madd-content" rows="4" placeholder="一条简洁的事实或偏好"></textarea>
        </label>
        <div class="row" style="align-items:flex-end">
          <label class="field" style="flex:1">
            <span class="label">类型</span>
            <select id="madd-kind">
              ${Object.entries(KIND_LABEL).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}
            </select>
          </label>
          <label class="field" style="flex:1">
            <span class="label">作用域</span>
            <select id="madd-scope">
              <option value="user">按用户</option>
              <option value="group">按群/会话</option>
              <option value="global">全局</option>
            </select>
          </label>
        </div>
        <label class="field">
          <span class="label">归属 ID（按用户时必填）</span>
          <input type="text" id="madd-owner" placeholder="QQ 官方机器人填 user_openid" />
        </label>`,
      okText: "写入",
      onOk: async (mask) => {
        const content = $("#madd-content", mask).value.trim();
        if (!content) throw new Error("内容不能为空");
        await apiPost("memories/add", {
          content,
          kind: $("#madd-kind", mask).value,
          scope: $("#madd-scope", mask).value,
          owner_id: $("#madd-owner", mask).value.trim(),
        });
        toast("已写入");
        state.mem.offset = 0;
        loadMemories();
        loadOverview();
      },
    });
  }

  function mergeMemoriesModal(records) {
    const joined = records.map((r) => r.content).join("\n");
    openModal({
      title: `合并 ${records.length} 条记忆`,
      sub: "合并后旧条会标记为已合并并退出检索，可回溯",
      bodyHtml: `
        <div class="card" style="padding:10px 12px;margin-bottom:12px;box-shadow:none">
          ${records.map((r) => `<div class="muted" style="font-size:12.5px">#${r.id} ${esc(r.content)}</div>`).join("")}
        </div>
        <label class="field">
          <span class="label">合并后的内容（可自行改写，越简洁越好）</span>
          <textarea id="mmerge-content" rows="5">${esc(joined)}</textarea>
        </label>
        <label class="field">
          <span class="label">类型</span>
          <select id="mmerge-kind">
            ${Object.entries(KIND_LABEL).map(([k, v]) => `<option value="${k}" ${records[0].kind === k ? "selected" : ""}>${v}</option>`).join("")}
          </select>
        </label>`,
      okText: "合并",
      onOk: async (mask) => {
        const content = $("#mmerge-content", mask).value.trim();
        if (!content) throw new Error("合并内容不能为空");
        await apiPost("memories/merge", {
          ids: records.map((r) => r.id),
          content,
          kind: $("#mmerge-kind", mask).value,
        });
        state.mem.selected.clear();
        toast("已合并");
        loadMemories();
        loadOverview();
      },
    });
  }

  function confirmDelete(count, onOk) {
    openModal({
      title: `删除 ${count} 条记忆`,
      sub: "物理删除，不可恢复。只是重复的话建议用「合并」",
      bodyHtml: '<p class="muted">确认删除？</p>',
      okText: "删除",
      danger: true,
      onOk,
    });
  }

  /* ---------------- 技能 ---------------- */

  async function loadSkills() {
    try {
      const data = await apiGet("skills");
      state.skills.rows = data.records || [];
      renderSkillList();
      if (state.skills.current) {
        const still = state.skills.rows.find((s) => s.name === state.skills.current.name);
        if (still) selectSkill(still.name); else clearSkillEditor();
      }
    } catch (err) {
      $("#skill-list").innerHTML = `<div class="empty">加载失败：${esc(errMsg(err))}</div>`;
    }
  }

  function renderSkillList() {
    const kw = state.skills.filter.trim().toLowerCase();
    const rows = state.skills.rows.filter((s) => !kw
      || String(s.name).toLowerCase().includes(kw)
      || String(s.trigger || "").toLowerCase().includes(kw));
    $("#skill-count").textContent = `共 ${state.skills.rows.length} 个技能${kw ? `，匹配 ${rows.length} 个` : ""}`;
    $("#skill-list").innerHTML = rows.length
      ? rows.map((s) => `
        <button class="skill-item ${state.skills.current && state.skills.current.name === s.name ? "active" : ""}" data-skill="${esc(s.name)}">
          <div class="n mono">${esc(s.name)}</div>
          <div class="t">${esc(s.trigger || "通用")}</div>
          <div class="u">使用 ${fmtNum(s.use_count)} 次 · ${fmtTime(s.created_ts)}</div>
        </button>`).join("")
      : '<div class="empty">没有匹配的技能</div>';
  }

  function clearSkillEditor() {
    state.skills.current = null;
    $("#skill-empty").classList.remove("hidden");
    $("#skill-editor").classList.add("hidden");
    renderSkillList();
  }

  function selectSkill(name) {
    const skill = state.skills.rows.find((s) => s.name === name);
    if (!skill) return;
    state.skills.current = skill;
    $("#skill-empty").classList.add("hidden");
    $("#skill-editor").classList.remove("hidden");
    $("#sk-name").value = skill.name;
    $("#sk-trigger").value = skill.trigger || "";
    $("#sk-content").value = skill.content || "";
    $("#sk-meta").textContent = `使用 ${fmtNum(skill.use_count)} 次 · 创建于 ${fmtTime(skill.created_ts)}`;
    state.skills.preview = false;
    $("#sk-preview").classList.add("hidden");
    $("#sk-content").classList.remove("hidden");
    $("#sk-preview-toggle").textContent = "预览";
    renderSkillList();
  }

  function newSkill() {
    state.skills.current = null;
    $("#skill-empty").classList.add("hidden");
    $("#skill-editor").classList.remove("hidden");
    $("#sk-name").value = "";
    $("#sk-trigger").value = "";
    $("#sk-content").value = "";
    $("#sk-meta").textContent = "新建技能";
    state.skills.preview = false;
    $("#sk-preview").classList.add("hidden");
    $("#sk-content").classList.remove("hidden");
    $("#sk-preview-toggle").textContent = "预览";
    renderSkillList();
    $("#sk-name").focus();
  }

  async function saveSkill(btn) {
    const name = $("#sk-name").value.trim();
    const trigger = $("#sk-trigger").value.trim();
    const content = $("#sk-content").value;
    if (!name) return toast("技能名不能为空", "err");
    if (content.trim().length < 5) return toast("技能内容太短", "err");
    setBusy(btn, true, "保存中");
    try {
      await apiPost("skills/save", {
        name,
        old_name: state.skills.current ? state.skills.current.name : "",
        trigger,
        content,
      });
      toast("已保存");
      state.skills.current = { name };
      await loadSkills();
      selectSkill(name);
      loadOverview();
    } catch (err) {
      toast(errMsg(err), "err");
    } finally {
      setBusy(btn, false);
    }
  }

  /* ---------------- 设置 ---------------- */

  function settingLabel(item) {
    const desc = String(item.description || "").split(/[：:]/)[0].trim();
    if (desc && desc.length <= 28) return desc;
    return item.key;
  }

  /** 描述去掉已展示在名称里的前缀，避免同一段话显示两遍。 */
  function settingDesc(item) {
    const full = String(item.description || "");
    const label = settingLabel(item);
    if (label !== item.key && full.startsWith(label)) {
      return full.slice(label.length).replace(/^[：:]\s*/, "").trim();
    }
    return full.trim();
  }

  function controlHtml(item) {
    const v = item.value;
    const key = item.key;
    if (item.type === "bool") {
      return `<label class="switch"><input type="checkbox" data-key="${esc(key)}" data-type="bool" ${v ? "checked" : ""}><span></span></label>`;
    }
    if (Array.isArray(item.options) && item.options.length) {
      return `<select data-key="${esc(key)}" data-type="string" class="narrow">
        ${item.options.map((o) => `<option value="${esc(o)}" ${String(v) === String(o) ? "selected" : ""}>${esc(o)}</option>`).join("")}
      </select>`;
    }
    if (item.type === "text" || String(key) === "summary_prompt") {
      return `<textarea data-key="${esc(key)}" data-type="string" rows="7" spellcheck="false">${esc(v == null ? "" : v)}</textarea>`;
    }
    if (item.type === "int" || item.type === "float" || item.type === "number") {
      const step = item.type === "float" ? "0.05" : "1";
      return `<input type="number" step="${step}" data-key="${esc(key)}" data-type="${item.type}" value="${esc(v)}">`;
    }
    return `<input type="text" data-key="${esc(key)}" data-type="string" value="${esc(v == null ? "" : v)}">`;
  }

  async function loadSettings() {
    try {
      const data = await apiGet("config");
      state.settings.groups = data.groups || [];
      state.settings.dirty = false;
      renderSettings();
      $("#settings-status").textContent = "修改后记得保存";
    } catch (err) {
      $("#settings-groups").innerHTML = `<div class="empty">读取配置失败：${esc(errMsg(err))}</div>`;
    }
  }

  function renderSettings() {
    const groups = state.settings.groups;
    if (!groups.length) {
      $("#settings-groups").innerHTML = '<div class="empty">没有可编辑的配置项</div>';
      return;
    }
    $("#settings-groups").innerHTML = groups.map((g) => `
      <div style="margin-bottom:6px">
        <div class="card-head" style="margin:6px 0 2px"><h3>${esc(g.title)}</h3></div>
        ${g.items.map((item) => `
          <div class="setting" data-setting="${esc(item.key)}">
            <div class="head">
              <div>
                <div class="name">${esc(settingLabel(item))}<span class="key">${esc(item.key)}</span>
                ${item.needs_reload ? '<span class="chip warn" style="margin-left:6px">需重载</span>' : ""}</div>
                ${settingDesc(item) ? `<div class="desc">${esc(settingDesc(item))}</div>` : ""}
                ${item.hint && item.hint !== settingDesc(item) ? `<div class="hint">${esc(item.hint)}</div>` : ""}
              </div>
              ${item.type === "bool" ? controlHtml(item) : ""}
            </div>
            ${item.type === "bool" ? "" : `<div class="control">${controlHtml(item)}</div>`}
          </div>`).join("")}
      </div>`).join("");
  }

  function collectSettings() {
    const values = {};
    $$("#settings-groups [data-key]").forEach((el) => {
      const key = el.dataset.key;
      const type = el.dataset.type;
      if (type === "bool") values[key] = el.checked;
      else if (type === "int" || type === "float" || type === "number") {
        values[key] = el.value === "" ? null : Number(el.value);
      } else values[key] = el.value;
    });
    return values;
  }

  async function saveSettings(btn) {
    const values = collectSettings();
    if (Object.values(values).some((v) => v === null)) {
      return toast("有数值项为空", "err");
    }
    setBusy(btn, true, "保存中");
    try {
      const res = await apiPost("config/save", { values });
      const n = (res.saved || []).length;
      toast(`已保存 ${n} 项`);
      if (res.needs_reload && res.needs_reload.length) {
        $("#settings-status").textContent = `${n} 项已保存，其中 ${res.needs_reload.join("、")} 需重载插件生效`;
      } else {
        $("#settings-status").textContent = `${n} 项已保存`;
      }
      state.settings.dirty = false;
      loadOverview();
    } catch (err) {
      toast(errMsg(err), "err");
    } finally {
      setBusy(btn, false);
    }
  }

  /* ---------------- 工具 ---------------- */

  async function runAction(btn, outEl, fn, busyText) {
    setBusy(btn, true, busyText || "处理中");
    outEl.classList.remove("hidden", "ok", "err");
    outEl.textContent = "执行中，请稍候（可能会调用模型，耗时几秒到几十秒）…";
    try {
      const res = await fn();
      outEl.classList.add("ok");
      outEl.textContent = res.result || res.preview || "完成";
      toast("完成");
      loadOverview();
      return res;
    } catch (err) {
      outEl.classList.add("err");
      outEl.textContent = `失败：${errMsg(err)}`;
      toast(errMsg(err), "err");
    } finally {
      setBusy(btn, false);
    }
  }

  async function loadAliases() {
    try {
      const data = await apiGet("aliases");
      state.aliases = data.records || [];
      $("#alias-tbody").innerHTML = state.aliases.length
        ? state.aliases.map((a) => `
          <tr>
            <td class="mono">${esc(a.alias_id || a.alias || "")}</td>
            <td class="mono">${esc(a.canonical_id || a.canonical || "")}</td>
            <td>${esc(a.platform || "—")}</td>
            <td><button class="btn tiny danger" data-alias-del="${esc(a.alias_id || a.alias || "")}">解除</button></td>
          </tr>`).join("")
        : '<tr><td colspan="4"><div class="empty">还没有绑定过别名</div></td></tr>';
    } catch (err) {
      $("#alias-tbody").innerHTML = `<tr><td colspan="4"><div class="empty">读取失败：${esc(errMsg(err))}</div></td></tr>`;
    }
  }

  /* ---------------- 事件绑定 ---------------- */

  function bindEvents() {
    $("#nav").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-tab]");
      if (btn) switchTab(btn.dataset.tab);
    });
    document.addEventListener("click", (e) => {
      const goto = e.target.closest("[data-goto]");
      if (goto) switchTab(goto.dataset.goto);
    });
    $("#btn-refresh").addEventListener("click", () => {
      switchTab(state.tab);
      toast("已刷新");
    });

    // 记忆
    const doSearch = () => {
      state.mem.q = $("#mem-q").value.trim();
      state.mem.scope = $("#mem-scope").value;
      state.mem.kind = $("#mem-kind").value;
      state.mem.merged = $("#mem-merged").checked;
      state.mem.offset = 0;
      state.mem.selected.clear();
      loadMemories();
    };
    $("#mem-search").addEventListener("click", doSearch);
    $("#mem-q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
    $("#mem-scope").addEventListener("change", doSearch);
    $("#mem-kind").addEventListener("change", doSearch);
    $("#mem-merged").addEventListener("change", doSearch);
    $("#mem-add").addEventListener("click", addMemoryModal);
    $("#mem-prev").addEventListener("click", () => {
      state.mem.offset = Math.max(0, state.mem.offset - state.mem.limit);
      loadMemories();
    });
    $("#mem-next").addEventListener("click", () => {
      if (state.mem.offset + state.mem.limit < state.mem.total) {
        state.mem.offset += state.mem.limit;
        loadMemories();
      }
    });
    $("#mem-check-all").addEventListener("change", (e) => {
      const on = e.target.checked;
      state.mem.rows.forEach((r) => {
        if (on) state.mem.selected.add(r.id); else state.mem.selected.delete(r.id);
      });
      renderMemories();
    });
    $("#mem-clear-sel").addEventListener("click", () => {
      state.mem.selected.clear();
      renderMemories();
    });
    $("#mem-tbody").addEventListener("change", (e) => {
      const box = e.target.closest(".mem-check");
      if (!box) return;
      const id = Number(box.dataset.id);
      if (box.checked) state.mem.selected.add(id); else state.mem.selected.delete(id);
      updateBulkBar();
      $("#mem-check-all").checked = state.mem.rows.every((r) => state.mem.selected.has(r.id));
    });
    $("#mem-tbody").addEventListener("click", async (e) => {
      const edit = e.target.closest("[data-mem-edit]");
      if (edit) {
        const rec = state.mem.rows.find((r) => r.id === Number(edit.dataset.memEdit));
        if (rec) editMemoryModal(rec);
        return;
      }
      const del = e.target.closest("[data-mem-del]");
      if (del) {
        const id = Number(del.dataset.memDel);
        confirmDelete(1, async () => {
          await apiPost("memories/delete", { id });
          state.mem.selected.delete(id);
          toast("已删除");
          loadMemories();
          loadOverview();
        });
      }
    });
    $("#mem-merge").addEventListener("click", () => {
      const recs = state.mem.rows.filter((r) => state.mem.selected.has(r.id));
      if (recs.length < 2) return toast("至少选两条", "err");
      mergeMemoriesModal(recs);
    });
    $("#mem-del").addEventListener("click", () => {
      const ids = Array.from(state.mem.selected);
      if (!ids.length) return;
      confirmDelete(ids.length, async () => {
        for (const id of ids) {
          try { await apiPost("memories/delete", { id }); } catch (_) { /* 单条失败继续 */ }
        }
        state.mem.selected.clear();
        toast(`已删除 ${ids.length} 条`);
        loadMemories();
        loadOverview();
      });
    });

    // 技能
    $("#skill-q").addEventListener("input", (e) => {
      state.skills.filter = e.target.value;
      renderSkillList();
    });
    $("#skill-new").addEventListener("click", newSkill);
    $("#skill-list").addEventListener("click", (e) => {
      const item = e.target.closest("[data-skill]");
      if (item) selectSkill(item.dataset.skill);
    });
    $("#sk-save").addEventListener("click", (e) => saveSkill(e.currentTarget));
    $("#sk-delete").addEventListener("click", () => {
      const name = $("#sk-name").value.trim();
      if (!name) return;
      openModal({
        title: `删除技能 ${name}`,
        sub: "删除后模型就不能再读取这个技能了",
        bodyHtml: '<p class="muted">确认删除？</p>',
        okText: "删除",
        danger: true,
        onOk: async () => {
          await apiPost("skills/delete", { name });
          toast("已删除");
          clearSkillEditor();
          await loadSkills();
          loadOverview();
        },
      });
    });
    $("#sk-preview-toggle").addEventListener("click", () => {
      state.skills.preview = !state.skills.preview;
      const on = state.skills.preview;
      $("#sk-preview").classList.toggle("hidden", !on);
      $("#sk-content").classList.toggle("hidden", on);
      $("#sk-preview-toggle").textContent = on ? "编辑" : "预览";
      if (on) $("#sk-preview").innerHTML = renderMarkdown($("#sk-content").value);
    });

    // 设置
    $("#settings-groups").addEventListener("input", () => {
      state.settings.dirty = true;
      $("#settings-status").textContent = "有未保存的修改";
    });
    $("#settings-groups").addEventListener("change", () => {
      state.settings.dirty = true;
      $("#settings-status").textContent = "有未保存的修改";
    });
    $("#settings-save").addEventListener("click", (e) => saveSettings(e.currentTarget));
    $("#settings-reset").addEventListener("click", () => {
      loadSettings();
      toast("已还原为已保存的配置");
    });

    // 工具
    $("#act-summarize").addEventListener("click", (e) =>
      runAction(e.currentTarget, $("#out-summarize"), () => apiPost("actions/summarize"), "总结中"));
    $("#act-consolidate-preview").addEventListener("click", (e) =>
      runAction(e.currentTarget, $("#out-consolidate"), () => apiPost("actions/consolidate", { dry_run: true }), "分析中"));
    $("#act-consolidate").addEventListener("click", () => {
      openModal({
        title: "执行记忆整理",
        sub: "会真实写入：旧的碎片记忆被标记合并",
        bodyHtml: '<p class="muted">确认执行？如果只是想看计划，先点「预览」。</p>',
        okText: "执行",
        onOk: async () => {
          const btn = $("#act-consolidate");
          await runAction(btn, $("#out-consolidate"), () => apiPost("actions/consolidate", { dry_run: false }), "整理中");
        },
      });
    });
    $("#act-skill-preview").addEventListener("click", (e) =>
      runAction(e.currentTarget, $("#out-skill"), () => apiPost("actions/skill-consolidate", { dry_run: true }), "分析中"));
    $("#act-skill").addEventListener("click", () => {
      openModal({
        title: "执行技能整理",
        sub: "重叠技能会合并成一个，旧技能被删除",
        bodyHtml: '<p class="muted">确认执行？建议先预览。</p>',
        okText: "执行",
        onOk: async () => {
          const btn = $("#act-skill");
          await runAction(btn, $("#out-skill"), () => apiPost("actions/skill-consolidate", { dry_run: false }), "整理中");
          await loadSkills();
        },
      });
    });
    $("#act-export").addEventListener("click", (e) =>
      runAction(e.currentTarget, $("#out-export"), () => apiPost("actions/export"), "导出中"));

    // 别名
    $("#alias-add").addEventListener("click", async (e) => {
      const alias = $("#alias-alias").value.trim();
      const canonical = $("#alias-canonical").value.trim();
      if (!alias || !canonical) return toast("别名和主 ID 都要填", "err");
      const btn = e.currentTarget;
      setBusy(btn, true, "绑定中");
      try {
        await apiPost("aliases/add", { alias, canonical, platform: $("#alias-platform").value.trim() });
        $("#alias-alias").value = "";
        $("#alias-canonical").value = "";
        toast("已绑定");
        loadAliases();
      } catch (err) {
        toast(errMsg(err), "err");
      } finally {
        setBusy(btn, false);
      }
    });
    $("#alias-tbody").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-alias-del]");
      if (!btn) return;
      const alias = btn.dataset.aliasDel;
      openModal({
        title: "解除身份绑定",
        sub: alias,
        bodyHtml: '<p class="muted">解除后这两个 ID 的记忆会重新分开。</p>',
        okText: "解除",
        danger: true,
        onOk: async () => {
          await apiPost("aliases/delete", { alias });
          toast("已解除");
          loadAliases();
        },
      });
    });
  }

  /* ---------------- 启动 ---------------- */

  let booted = false;
  async function boot() {
    if (booted) return;
    booted = true;
    document.documentElement.setAttribute("data-theme",
      window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    bindEvents();

    const bridge = getBridge();
    if (!bridge) {
      $("#bridge-banner").classList.remove("hidden");
      $("#settings-groups").innerHTML = '<div class="empty">未连接到 AstrBot 面板</div>';
      return;
    }

    try {
      await bridge.ready();
      if (bridge.getContext && bridge.getContext()) {
        const ctx = bridge.getContext();
        document.documentElement.setAttribute("data-theme", ctx.isDark ? "dark" : "light");
      }
      if (bridge.onContext) {
        bridge.onContext((ctx) => {
          if (ctx && typeof ctx.isDark === "boolean") {
            document.documentElement.setAttribute("data-theme", ctx.isDark ? "dark" : "light");
          }
        });
      }
    } catch (_) { /* context 晚到也不影响使用 */ }

    loadOverview();
  }

  document.addEventListener("DOMContentLoaded", boot);
  if (document.readyState !== "loading") boot();
})();
