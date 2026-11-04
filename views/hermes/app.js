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
    models: ["模型", "主模型 + 有序降级链：谁先上、失败怎么重试、什么时候换人"],
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
    skills: { rows: [], system: [], owners: [], filter: "", current: null, currentId: null, preview: false },
    settings: { groups: [], dirty: false },
    models: { data: null, draft: null, dirty: false },
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
    if (state.tab === "models" && name !== "models" && state.models.dirty) {
      if (!window.confirm("模型配置还有未保存的修改，确定离开？")) return;
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
    if (name === "models") loadModels();
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
      ["检索方式", '<span class="chip ok">字符二元组 · IDF · 时间衰减</span>'],
      ["注入效果", (() => {
        const mt = d.metrics || {};
        if (!mt.inject_requests) return '<span class="faint">还没有请求</span>';
        return `${fmtNum(mt.inject_requests)} 次 · 平均 ${mt.avg_memories} 条 / ${fmtNum(mt.avg_inject_chars)} 字`
          + ` <span class="faint">· 检索 ${mt.avg_recall_ms}ms</span>`;
      })()],
      ["模型调用", (() => {
        const r = d.router || {};
        if (!r.calls) return '<span class="faint">还没有调用</span>';
        return `成功 ${fmtNum(r.ok)} <span class="faint">（其中降级 ${fmtNum(r.fallback_ok)}）</span>`
          + ` · 失败 ${fmtNum(r.fail)} · 尝试 ${fmtNum(r.attempts)} 次`;
      })()],
      ["数据库", `${fmtBytes(d.db_bytes)} <span class="faint">（含 WAL）</span>`],
      ["自动总结", sm.auto
        ? `每 ${sm.interval} 分钟检查；单会话满 ${sm.min_new} 条或超 ${sm.grace_hours} 小时兜底`
        : '<span class="chip warn">已关闭</span>'],
      ["待总结", `${fmtNum(sm.pending_sessions)} 个会话 / ${fmtNum(sm.pending_messages)} 条消息`],
      ["身份别名", `${fmtNum(d.aliases)} 条`],
      ["总结模型", (() => {
        const m = d.models || {};
        if (!m.primary) return '<span class="chip warn">未配置</span>';
        const extra = (m.chain || []).length - 1;
        return `${esc(m.primary)}${m.follow_current ? ' <span class="faint">（跟随对话）</span>' : ""}`
          + (extra > 0 ? ` <span class="faint">· ${extra} 个降级</span>` : "");
      })()],
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
      const pinChip = Number(r.pinned) ? ' <span class="chip warn" title="钉住：每轮必注入，整理时不可合并/丢弃">钉住</span>' : "";
      return `
        <tr data-id="${r.id}">
          <td><input type="checkbox" class="checkbox mem-check" data-id="${r.id}" ${m.selected.has(r.id) ? "checked" : ""}></td>
          <td class="mono faint">${r.id}</td>
          <td><span class="chip blue">${esc(KIND_LABEL[r.kind] || r.kind)}</span></td>
          <td class="content"><div class="text">${esc(r.content)}</div>${mergedChip}${pinChip}</td>
          <td>${owner}</td>
          <td><span class="chip">${esc(r.source || "—")}</span></td>
          <td class="nowrap muted">${fmtTime(r.ts)}</td>
          <td>
            <div class="ops">
              <button class="btn tiny ghost" data-mem-pin="${r.id}" data-pinned="${Number(r.pinned) ? 1 : 0}">${Number(r.pinned) ? "取消钉住" : "钉住"}</button>
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
      state.skills.system = data.system || [];
      state.skills.owners = data.owners || [];
      renderSkillList();
      if (state.skills.currentId) {
        const still = findSkill(state.skills.currentId);
        if (still) selectSkill(state.skills.currentId); else clearSkillEditor();
      }
    } catch (err) {
      $("#skill-list").innerHTML = `<div class="empty">加载失败：${esc(errMsg(err))}</div>`;
    }
  }

  /** 技能分两个域：personal（按人隔离、可写）与 system（本体安装、只读）。 */
  function findSkill(key) {
    if (!key) return null;
    if (String(key).startsWith("sys:")) {
      return state.skills.system.find((s) => s.id === key) || null;
    }
    const id = Number(String(key).replace(/^p:/, ""));
    return state.skills.rows.find((s) => s.id === id) || null;
  }

  function skillKey(s) {
    return s.source === "system" ? s.id : `p:${s.id}`;
  }

  function renderSkillList() {
    const kw = state.skills.filter.trim().toLowerCase();
    const match = (s) => !kw
      || String(s.name).toLowerCase().includes(kw)
      || String(s.trigger || "").toLowerCase().includes(kw);
    const personal = state.skills.rows.filter(match);
    const system = state.skills.system.filter(match);
    const total = state.skills.rows.length + state.skills.system.length;
    $("#skill-count").textContent = kw
      ? `匹配 ${personal.length + system.length} / 共 ${total} 个技能`
      : `我的 ${state.skills.rows.length} 个 · 系统 ${state.skills.system.length} 个`;

    const item = (s) => {
      const key = skillKey(s);
      const ro = s.source === "system";
      const meta = ro
        ? `<span class="tag-ro">系统·只读</span>${s.has_scripts ? " 带脚本" : ""}`
        : `${esc(s.owner_label || "")} · 使用 ${fmtNum(s.use_count)} 次`;
      return `
        <button class="skill-item ${state.skills.currentId === key ? "active" : ""}" data-skill="${esc(key)}">
          <div class="n mono">${esc(s.name)}</div>
          <div class="t">${esc(s.trigger || "通用")}</div>
          <div class="u">${meta}</div>
        </button>`;
    };

    const parts = [];
    if (personal.length) {
      parts.push(`<div class="group-title">个人技能 <span>${personal.length}</span></div>`);
      parts.push(personal.map(item).join(""));
    }
    if (system.length) {
      parts.push(`<div class="group-title">系统技能 · 只读 <span>${system.length}</span></div>`);
      parts.push(system.map(item).join(""));
    }
    $("#skill-list").innerHTML = parts.length
      ? parts.join("")
      : '<div class="empty">没有匹配的技能</div>';
  }

  function clearSkillEditor() {
    state.skills.currentId = null;
    state.skills.current = null;
    $("#skill-empty").classList.remove("hidden");
    $("#skill-editor").classList.add("hidden");
    renderSkillList();
  }

  function selectSkill(key) {
    const skill = findSkill(key);
    if (!skill) return;
    const readonly = skill.source === "system";
    state.skills.currentId = key;
    state.skills.current = skill;
    $("#skill-empty").classList.add("hidden");
    $("#skill-editor").classList.remove("hidden");
    $("#sk-name").value = skill.name;
    $("#sk-trigger").value = skill.trigger || "";
    $("#sk-content").value = skill.content || "";
    if (readonly) {
      $("#sk-meta").textContent = "AstrBot 本体安装 · 全局共享";
      $("#sk-owner").value = "";
      $("#sk-scope").value = "global";
      $("#sk-readonly").classList.remove("hidden");
      $("#sk-readonly").innerHTML = "这是 <b>AstrBot 本体安装的系统技能</b>，全局共享"
        + (skill.has_scripts ? "、含可执行脚本" : "")
        + "。插件<b>不会写入</b>本体目录——改了会影响所有人，还可能在市场更新时被覆盖。"
        + "<br>要改它：点「新建技能」用同名保存一份，或直接在 QQ 里让 AI 改（会自动生成属于你的个人副本，"
        + "之后你优先读到个人版，本体原版不动）。";
      $("#sk-content").setAttribute("readonly", "readonly");
      $("#sk-save").disabled = true;
      $("#sk-delete").disabled = true;
    } else {
      const scope = skill.scope || "private";
      $("#sk-scope").value = scope;
      $("#sk-owner").value = scope === "group" ? (skill.group_id || "") : (skill.owner_id || "");
      $("#sk-meta").textContent = `${skill.owner_label || ""} · 使用 ${fmtNum(skill.use_count)} 次 · 创建于 ${fmtTime(skill.created_ts)}`;
      $("#sk-readonly").classList.add("hidden");
      $("#sk-content").removeAttribute("readonly");
      $("#sk-save").disabled = false;
      $("#sk-delete").disabled = false;
    }
    state.skills.preview = false;
    $("#sk-preview").classList.add("hidden");
    $("#sk-content").classList.toggle("hidden", false);
    $("#sk-preview-toggle").textContent = "预览";
    renderSkillList();
  }

  function newSkill() {
    state.skills.currentId = null;
    state.skills.current = null;
    $("#skill-empty").classList.add("hidden");
    $("#skill-editor").classList.remove("hidden");
    $("#sk-name").value = "";
    $("#sk-trigger").value = "";
    $("#sk-content").value = "";
    $("#sk-scope").value = "private";
    $("#sk-owner").value = "";
    $("#sk-meta").textContent = "新建技能";
    $("#sk-readonly").classList.add("hidden");
    $("#sk-content").removeAttribute("readonly");
    $("#sk-save").disabled = false;
    $("#sk-delete").disabled = true;
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
    const scope = $("#sk-scope").value;
    const ownerField = $("#sk-owner").value.trim();
    if (!name) return toast("技能名不能为空", "err");
    if (content.trim().length < 5) return toast("技能内容太短", "err");
    if (scope === "private" && !ownerField) {
      return toast("私人技能需要填用户 ID（对话里那个没昵称的 ID 就是）", "err");
    }
    if (scope === "group" && !ownerField) {
      return toast("群共享需要填群 ID", "err");
    }
    const payload = {
      name,
      old_name: state.skills.current ? state.skills.current.name : "",
      trigger,
      content,
      scope,
      owner_id: scope === "group" ? "" : ownerField,
      group_id: scope === "group" ? ownerField : "",
    };
    if (state.skills.current) payload.id = state.skills.current.id;
    setBusy(btn, true, "保存中");
    try {
      await apiPost("skills/save", payload);
      toast("已保存");
      const wantName = name;
      state.skills.currentId = null;
      await loadSkills();
      const saved = state.skills.rows.find((s) => s.name === wantName);
      if (saved) selectSkill(`p:${saved.id}`);
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

  /* ---------------- 模型 ---------------- */

  const POLICY_FIELDS = [
    ["llm_max_retries", "每模型重试次数", 0, 9, 1],
    ["llm_total_attempts", "整链最多尝试", 1, 30, 1],
    ["llm_timeout", "单次超时（秒）", 5, 600, 1],
    ["llm_backoff", "退避基数（秒）", 0, 30, 0.5],
    ["llm_cooldown_seconds", "失败冷却（秒）", 0, 3600, 10],
    ["llm_max_prompt_chars", "单次最大字符", 2000, 200000, 1000],
  ];

  const RETRY_KINDS = [
    ["timeout", "超时"], ["conn", "连接失败"], ["rate", "限流/额度"],
    ["server", "服务端错误"], ["empty", "空响应"],
    ["auth", "鉴权失败"], ["ctx", "上下文超长"], ["other", "其他"],
  ];

  function kindLabel(kind) {
    const hit = RETRY_KINDS.find(([k]) => k === kind);
    if (hit) return hit[1];
    const labels = (state.models.data && state.models.data.error_labels) || {};
    return labels[kind] || kind || "失败";
  }

  function ctxText(tokens) {
    if (!tokens) return "未知";
    return tokens >= 1024 ? `${Math.round(tokens / 1024)}k` : String(tokens);
  }

  function fmtClock(ts) {
    if (!ts) return "—";
    const d = new Date(Number(ts) * 1000);
    if (Number.isNaN(d.getTime())) return "—";
    const p = (x) => String(x).padStart(2, "0");
    const today = new Date();
    return d.toDateString() === today.toDateString()
      ? `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
      : `${d.getMonth() + 1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  function markModelsDirty() {
    state.models.dirty = true;
    const el = $("#model-save-status");
    if (el) el.textContent = "有未保存的修改";
  }

  async function loadModels() {
    try {
      const d = await apiGet("models");
      state.models.data = d;
      state.models.draft = {
        primary: d.primary || "",
        fallbacks: (d.fallbacks || []).slice(),
        policy: Object.assign({ retry_on: [] }, d.policy || {}),
      };
      state.models.dirty = false;
      renderModels();
      const status = $("#model-save-status");
      if (status) status.textContent = "修改后记得保存";
    } catch (err) {
      toast(`读取模型配置失败：${errMsg(err)}`, "err");
    }
  }

  function modelById(id) {
    const list = (state.models.data && state.models.data.providers) || [];
    return list.find((p) => p.id === id) || null;
  }

  function renderModels() {
    const d = state.models.data, draft = state.models.draft;
    if (!d || !draft) return;
    const list = d.providers || [];
    const usable = list.filter((p) => p.loaded);
    const cur = d.current_id || "";

    // 主模型下拉：留空 = 跟随当前对话模型
    const opts = [`<option value="">跟随当前对话模型${cur ? `（${esc(cur)}）` : "（未配置）"}</option>`];
    list.forEach((p) => {
      const label = `${p.id}${p.model && p.model !== p.id ? ` · ${p.model}` : ""}`;
      const tail = [ctxText(p.context_tokens), p.loaded ? "" : "未启用", p.id === cur ? "当前对话" : ""]
        .filter(Boolean).join(" · ");
      opts.push(`<option value="${esc(p.id)}"${p.id === draft.primary ? " selected" : ""}>${esc(label)}（${esc(tail)}）</option>`);
    });
    if (draft.primary && !list.some((p) => p.id === draft.primary)) {
      opts.push(`<option value="${esc(draft.primary)}" selected>${esc(draft.primary)}（不在当前列表）</option>`);
    }
    $("#model-primary").innerHTML = opts.join("");
    $("#model-current-chip").textContent = cur ? `对话模型：${cur}` : "未配置对话模型";
    $("#model-primary-hint").textContent = draft.primary
      ? `总结优先用 ${draft.primary}；它失败时按下方降级链继续`
      : "留空表示跟随 AstrBot 当前对话模型（对话模型一换，总结模型也跟着换）";

    // 降级链
    const chain = draft.fallbacks;
    $("#model-chain").innerHTML = chain.length
      ? chain.map((id, i) => {
        const p = modelById(id);
        const bits = [];
        if (p) {
          if (p.model && p.model !== id) bits.push(p.model);
          bits.push(`上下文 ${ctxText(p.context_tokens)}`);
          if (!p.loaded) bits.push('<span class="chip danger">未启用</span>');
          if (p.cooling > 0) bits.push(`<span class="chip warn">冷却 ${p.cooling}s</span>`);
          if (p.id === cur) bits.push("当前对话");
        } else {
          bits.push('<span class="chip danger">不存在</span>');
        }
        return `<div class="chain-item">
          <span class="idx">${i + 1}</span>
          <div class="chain-main">
            <div class="mono">${esc(id)}</div>
            <small>${bits.join(" · ")}</small>
          </div>
          <div class="chain-actions">
            <button class="btn tiny ghost" data-chain-up="${i}" title="上移"${i === 0 ? " disabled" : ""}>↑</button>
            <button class="btn tiny ghost" data-chain-down="${i}" title="下移"${i === chain.length - 1 ? " disabled" : ""}>↓</button>
            <button class="btn tiny danger" data-chain-del="${i}" title="移除">×</button>
          </div>
        </div>`;
      }).join("")
      : '<div class="empty">还没有降级模型。主模型一挂，总结就整批失败。</div>';

    const cand = usable.filter((p) => p.id !== draft.primary && !chain.includes(p.id));
    $("#model-add-select").innerHTML = cand.length
      ? cand.map((p) => `<option value="${esc(p.id)}">${esc(p.id)}${p.model && p.model !== p.id ? ` · ${esc(p.model)}` : ""}</option>`).join("")
      : '<option value="">没有可添加的模型</option>';
    $("#model-add").disabled = !cand.length;

    const effective = [draft.primary || cur].concat(chain).filter(Boolean);
    $("#model-chain-hint").textContent = effective.length
      ? `实际顺序：${effective.join(" → ")}`
      : "还没有任何可用模型";
    if (effective.length > 1) {
      $("#model-chain-hint").textContent += `（共 ${effective.length} 个）`;
    }

    // 容错策略
    $("#model-policy").innerHTML = POLICY_FIELDS.map(([key, label, low, high, step]) => {
      const val = draft.policy[key];
      return `<label class="field" style="margin-bottom:0">
        <span class="label">${esc(label)}</span>
        <input type="number" data-policy="${key}" min="${low}" max="${high}" step="${step}"
               value="${val == null ? "" : esc(val)}" />
      </label>`;
    }).join("");

    const picked = new Set(draft.policy.retry_on || []);
    $("#model-retry-kinds").innerHTML = RETRY_KINDS.map(([k, label]) => {
      const on = picked.has(k);
      const risky = k === "auth" || k === "ctx";
      const cls = on ? (risky ? "chip warn" : "chip blue") : "chip";
      const tip = risky ? "这类错误重试通常没用，建议只在排查时开启" : `重试 ${label} 类的失败`;
      return `<button type="button" class="${cls}" data-retry="${k}" title="${esc(tip)}">${on ? "✓ " : ""}${esc(label)}</button>`;
    }).join("");

    // 可用模型
    $("#model-tbody").innerHTML = list.length
      ? list.map((p) => {
        const inChain = draft.fallbacks.indexOf(p.id);
        const chips = [];
        if (p.is_current) chips.push('<span class="chip blue">当前对话</span>');
        if (p.id === draft.primary) chips.push('<span class="chip ok">主模型</span>');
        if (inChain >= 0) chips.push(`<span class="chip">降级 ${inChain + 1}</span>`);
        if (!p.loaded) chips.push('<span class="chip danger">未启用</span>');
        if (p.cooling > 0) chips.push(`<span class="chip warn">冷却 ${p.cooling}s</span>`);
        if (!chips.length) chips.push('<span class="faint">备用</span>');
        const st = p.stats || {};
        const stat = st.calls
          ? `${fmtNum(st.ok || 0)} 成功 / ${fmtNum(st.fail || 0)} 失败 · 均 ${fmtNum(st.avg_ms || 0)}ms`
          : "还没用过";
        return `<tr>
          <td class="mono">${esc(p.id)}</td>
          <td class="mono">${esc(p.model || "—")}</td>
          <td>${esc(ctxText(p.context_tokens))}</td>
          <td>${chips.join(" ")}</td>
          <td class="faint" style="font-size:12px">${esc(stat)}${st.last_kind ? ` <span class="chip danger">最近 ${esc(kindLabel(st.last_kind))}</span>` : ""}</td>
          <td><button class="btn tiny" data-model-test="${esc(p.id)}"${p.loaded ? "" : " disabled"}>测试</button></td>
        </tr>`;
      }).join("")
      : '<tr><td colspan="6"><div class="empty">AstrBot 里还没有配置对话模型</div></td></tr>';

    // 调用日志
    const logs = d.logs || [];
    $("#model-log-hint").textContent = logs.length ? `最近 ${logs.length} 条` : "暂无记录";
    $("#model-log-tbody").innerHTML = logs.length
      ? logs.map((l) => `<tr>
          <td class="faint">${esc(fmtClock(l.ts))}</td>
          <td>${esc(l.purpose)}</td>
          <td class="mono" style="font-size:12px">${esc(l.provider)}</td>
          <td>${l.attempt}</td>
          <td>${fmtNum(l.ms)}ms</td>
          <td>${l.ok ? '<span class="chip ok">成功</span>' : `<span class="chip danger">${esc(kindLabel(l.kind))}</span>`}</td>
          <td class="faint" style="font-size:12px" title="${esc(l.error || "")}">${esc((l.error || "").slice(0, 90))}</td>
        </tr>`).join("")
      : '<tr><td colspan="7"><div class="empty">还没有调用记录。跑一次总结或点「测试」就会出现在这里。</div></td></tr>';
  }

  function renderTestResults(results) {
    if (!results || !results.length) {
      $("#model-test-results").innerHTML = '<div class="empty">没有返回结果</div>';
      return;
    }
    $("#model-test-results").innerHTML = results.map((r) => `
      <div class="test-item">
        <div class="row-between">
          <span class="mono">${esc(r.id)}</span>
          <span class="row" style="gap:8px">
            ${r.ok ? '<span class="chip ok">通过</span>' : `<span class="chip danger">${esc(r.label || kindLabel(r.kind))}</span>`}
            <span class="faint">${fmtNum(r.ms)}ms</span>
          </span>
        </div>
        ${r.ok
          ? `<small class="faint">回复：${esc((r.reply || "").slice(0, 80))}</small>`
          : `<small class="err-text">${esc((r.error || "").slice(0, 220))}</small>`}
      </div>`).join("");
  }

  async function testModels(ids, btn) {
    setBusy(btn, true, ids.length > 1 ? "逐个测试中" : "测试中");
    try {
      const d = await apiPost("models/test", { ids });
      const results = d.results || [];
      renderTestResults(results);
      const ok = results.filter((r) => r.ok).length;
      toast(`测试完成：${ok}/${results.length} 通过`, ok === results.length ? "ok" : "err");
      // 有未保存修改时不要刷新，免得把用户正在编辑的链冲掉
      if (!state.models.dirty) await loadModels();
    } catch (err) {
      toast(`测试失败：${errMsg(err)}`, "err");
    } finally {
      setBusy(btn, false);
    }
  }

  function useAstrbotChain() {
    const d = state.models.data;
    const a = (d && d.astrbot) || {};
    if (!a.primary && !(a.fallbacks || []).length) {
      toast("AstrBot 对话配置里没有可用的模型信息", "err");
      return;
    }
    const known = new Set((d.providers || []).filter((p) => p.loaded).map((p) => p.id));
    const draft = state.models.draft;
    draft.primary = known.has(a.primary) ? a.primary : "";
    draft.fallbacks = (a.fallbacks || []).filter((id) => known.has(id) && id !== draft.primary);
    markModelsDirty();
    renderModels();
    toast("已按 AstrBot 对话配置填入，记得保存");
  }

  async function saveModels(btn) {
    const draft = state.models.draft;
    setBusy(btn, true, "保存中");
    try {
      const r = await apiPost("models/save", {
        primary: draft.primary,
        fallbacks: draft.fallbacks,
        policy: draft.policy,
      });
      state.models.dirty = false;
      toast(`已保存：${(r.chain || []).join(" → ") || "无可用模型"}`);
      await loadModels();
    } catch (err) {
      toast(`保存失败：${errMsg(err)}`, "err");
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
      const pinBtn = e.target.closest("[data-mem-pin]");
      if (pinBtn) {
        const id = Number(pinBtn.dataset.memPin);
        const pinned = pinBtn.dataset.pinned === "1";
        try {
          await apiPost("memories/pin", { id, pinned: !pinned });
          toast(pinned ? "已取消钉住" : "已钉住：每轮必注入");
          loadMemories();
        } catch (_) { /* 错误已由 apiPost 提示 */ }
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
      const cur = state.skills.current;
      const name = $("#sk-name").value.trim();
      if (!name || !cur || cur.source === "system") return;
      openModal({
        title: `删除技能 ${name}`,
        sub: `归属：${cur.owner_label || ""}`,
        bodyHtml: '<p class="muted">只删这一份，其他人的同名技能不受影响。确认删除？</p>',
        okText: "删除",
        danger: true,
        onOk: async () => {
          await apiPost("skills/delete", { id: cur.id, name });
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
    // 模型
    $("#model-primary").addEventListener("change", (e) => {
      state.models.draft.primary = e.target.value;
      markModelsDirty();
      renderModels();
    });
    $("#model-chain").addEventListener("click", (e) => {
      const arr = state.models.draft.fallbacks;
      const up = e.target.closest("[data-chain-up]");
      const down = e.target.closest("[data-chain-down]");
      const del = e.target.closest("[data-chain-del]");
      if (up) {
        const i = Number(up.dataset.chainUp);
        if (i > 0) [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
      } else if (down) {
        const i = Number(down.dataset.chainDown);
        if (i < arr.length - 1) [arr[i + 1], arr[i]] = [arr[i], arr[i + 1]];
      } else if (del) {
        arr.splice(Number(del.dataset.chainDel), 1);
      } else {
        return;
      }
      markModelsDirty();
      renderModels();
    });
    $("#model-add").addEventListener("click", () => {
      const id = $("#model-add-select").value;
      if (!id) return;
      const arr = state.models.draft.fallbacks;
      if (!arr.includes(id)) arr.push(id);
      markModelsDirty();
      renderModels();
    });
    $("#model-policy").addEventListener("input", (e) => {
      const input = e.target.closest("[data-policy]");
      if (!input) return;
      state.models.draft.policy[input.dataset.policy] = input.value === "" ? 0 : Number(input.value);
      markModelsDirty();
    });
    $("#model-retry-kinds").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-retry]");
      if (!btn) return;
      const draft = state.models.draft;
      if (!Array.isArray(draft.policy.retry_on)) draft.policy.retry_on = [];
      const key = btn.dataset.retry;
      const i = draft.policy.retry_on.indexOf(key);
      if (i >= 0) draft.policy.retry_on.splice(i, 1);
      else draft.policy.retry_on.push(key);
      markModelsDirty();
      renderModels();
    });
    $("#model-save").addEventListener("click", (e) => saveModels(e.currentTarget));
    $("#model-reset").addEventListener("click", () => {
      state.models.dirty = false;
      loadModels();
      toast("已恢复为已保存的配置");
    });
    $("#model-use-astrbot").addEventListener("click", useAstrbotChain);
    $("#model-test-primary").addEventListener("click", (e) => {
      const d = state.models;
      if (!d.data) return;
      const pid = d.draft.primary || d.data.current_id;
      if (!pid) { toast("还没有可测试的主模型", "err"); return; }
      testModels([pid], e.currentTarget);
    });
    $("#model-test-chain").addEventListener("click", (e) => {
      const d = state.models;
      if (!d.data) return;
      const ids = [d.draft.primary || d.data.current_id, ...d.draft.fallbacks].filter(Boolean);
      if (!ids.length) { toast("链里还没有模型", "err"); return; }
      testModels(ids, e.currentTarget);
    });
    $("#model-tbody").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-model-test]");
      if (btn) testModels([btn.dataset.modelTest], btn);
    });
    $("#model-log-clear").addEventListener("click", async () => {
      try {
        await apiPost("models/logs/clear");
        toast("已清空日志与冷却");
        state.models.dirty = false;
        loadModels();
      } catch (err) {
        toast(errMsg(err), "err");
      }
    });

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
