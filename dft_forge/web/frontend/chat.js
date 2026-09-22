import { renderCif, clear as clearViewer, toggleReplica, toggleCellFrame } from "./viewer.js";
import { renderCharts } from "./charts.js";
import { approveRequest, clearWorkspace, refreshWorkspace, requestContext, restoreAttachment } from "./workspace.js";

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("msg");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const sessionList = document.getElementById("sessionList");
const sessionCount = document.getElementById("sessionCount");
const engineStatus = document.getElementById("engineStatus");
const chartsEl = document.getElementById("charts");

let sessionId = localStorage.getItem("dftforge_session") || null;
let activeStream = null;
let sessionLoadSerial = 0;
let approvalPending = false;

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = `statusline mono ${cls}`.trim();
}

function addApprovalRetry(msgDiv, bubble, text) {
  if (msgDiv.querySelector(".approval-retry")) return;
  const action = document.createElement("button");
  action.type = "button";
  action.className = "approval-retry";
  action.textContent = "确认方法后重试";
  action.addEventListener("click", () => {
    input.value = text;
    send(true);
  });
  bubble.appendChild(action);
}

function contractValue(value) {
  if (value === undefined || value === null || value === "") return "未提供";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function effectiveSummary(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return contractValue(value);
  return Object.entries(value)
    .filter(([key]) => key !== "scan_points")
    .map(([key, item]) => `${key}=${contractValue(item)}`)
    .concat(Array.isArray(value.scan_points) ? [`扫描组合=${value.scan_points.length} 组`] : [])
    .join(" · ") || "未提供";
}

function contractEntries(value) {
  return Object.entries(value || {}).map(([key, detail]) => {
    const requested = detail?.requested ?? detail;
    const reason = detail?.reason ? `（${detail.reason}）` : "";
    return `${key}: ${contractValue(requested)}${reason}`;
  }).join(" · ");
}

function methodLine(parent, text, className = "") {
  const line = document.createElement("div");
  line.className = `method-contract-line ${className}`.trim();
  line.textContent = text;
  parent.appendChild(line);
}

function addConvergencePreview(parent, convergence) {
  if (!convergence || typeof convergence !== "object") return;
  const panel = document.createElement("div");
  panel.className = "method-convergence-preview";
  methodLine(panel, `先做收敛扫描${convergence.job_count != null ? ` · ${convergence.job_count} 次计算` : ""}`);
  if (convergence.ecutwfc_values?.length) methodLine(panel, `截断能：${convergence.ecutwfc_values.join("、")} Ry`);
  if (convergence.kpoints_values?.length) {
    methodLine(panel, `k 点网格：${convergence.kpoints_values.map((grid) => Array.isArray(grid) ? grid.slice(0, 3).join("×") : contractValue(grid)).join("、")}`);
  }
  if (convergence.fixed_ecutrho != null) methodLine(panel, `固定 ecutrho：${convergence.fixed_ecutrho} Ry`);
  const fixed = [];
  if (convergence.occupations != null) fixed.push(`occupations=${contractValue(convergence.occupations)}`);
  if (convergence.smearing_values?.length) fixed.push(`smearing=${convergence.smearing_values.map(contractValue).join("、")}`);
  if (convergence.degauss_values?.length) fixed.push(`degauss=${convergence.degauss_values.map(contractValue).join("、")} Ry`);
  if (fixed.length) methodLine(panel, `固定占据与展宽：${fixed.join(" · ")}`);
  methodLine(panel, `总能判据：${convergence.threshold_mev_per_atom ?? 1} meV/atom；不代表带隙或其他物性已收敛。`);
  parent.appendChild(panel);
}

function addRecommendation(parent, recommendation) {
  if (!recommendation || typeof recommendation !== "object") return;
  methodLine(parent, recommendation.status === "deferred"
    ? "参数推荐：等待实际结构或前序结果"
    : "初始参数推荐 · 尚未通过收敛测试", "recommendation-state");
  const details = document.createElement("details");
  details.className = "method-recommendation";
  const summary = document.createElement("summary");
  summary.textContent = "推荐依据与参数来源";
  details.appendChild(summary);
  const structure = recommendation.structure;
  if (structure && typeof structure === "object") {
    methodLine(details, `结构：${structure.formula || "待确定"}${structure.natoms != null ? ` · ${structure.natoms} 原子` : ""}`);
  }
  for (const [key, reason] of Object.entries(recommendation.reasons || {})) {
    const value = recommendation.parameters?.[key];
    methodLine(details, `${key}${value !== undefined ? ` = ${contractValue(value)}` : ""}：${contractValue(reason)}`);
  }
  for (const pseudo of recommendation.pseudopotentials || []) {
    methodLine(details, `赝势：${pseudo.filename || "未提供文件名"} · 价电子数 ${contractValue(pseudo.z_valence)}`);
    if (pseudo.sha256) methodLine(details, `SHA-256：${pseudo.sha256}`, "mono");
  }
  for (const note of recommendation.notes || []) methodLine(details, note);
  parent.appendChild(details);
  addConvergencePreview(parent, recommendation.convergence);
}

function addMethodContract(msgDiv, bubble, contract, methodToken, requestSnapshot) {
  if (!contract || !methodToken || msgDiv.querySelector(".method-contract-card")) return;
  const card = document.createElement("section");
  card.className = "method-contract-card";
  const title = document.createElement("strong");
  title.textContent = "请确认计算参数与方法";
  card.appendChild(title);
  const scope = document.createElement("p");
  const planner = contract.planner === "llm" ? "LLM 已规划计算任务。" : contract.planner === "rules" ? "当前使用规则规划。" : "";
  scope.textContent = `${planner}以下为实际执行步骤及参数。初始推荐尚未经过收敛验证；如需调整，可在“方法参数”保存设置后重新发送任务。`;
  card.appendChild(scope);
  addConvergencePreview(card, contract.convergence);
  for (const step of contract.steps || []) {
    const row = document.createElement("div");
    row.className = "method-contract-step";
    const head = document.createElement("div");
    head.className = "method-contract-step-head";
    head.textContent = `${Number(step.index) + 1}. ${step.tool || "步骤"}${step.template_id ? ` · ${step.template_id}` : ""}`;
    row.appendChild(head);
    const effective = document.createElement("div");
    effective.className = "method-contract-line mono";
    effective.textContent = `有效参数：${effectiveSummary(step.effective)}`;
    row.appendChild(effective);
    addRecommendation(row, step.recommendation);
    if (Array.isArray(step.effective?.scan_points) && step.effective.scan_points.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `展开 ${step.effective.scan_points.length} 组扫描组合`;
      details.appendChild(summary);
      const table = document.createElement("table");
      table.className = "method-contract-scan mono";
      const header = document.createElement("tr");
      for (const label of ["ecutwfc (Ry)", "k 点", "smearing", "degauss"]) {
        const cell = document.createElement("th");
        cell.textContent = label;
        header.appendChild(cell);
      }
      table.appendChild(header);
      for (const scan of step.effective.scan_points) {
        const tr = document.createElement("tr");
        for (const value of [scan.ecutwfc, (scan.kpoints || []).slice(0, 3).join("×"), scan.smearing, scan.degauss]) {
          const cell = document.createElement("td");
          cell.textContent = contractValue(value);
          tr.appendChild(cell);
        }
        table.appendChild(tr);
      }
      details.appendChild(table);
      row.appendChild(details);
    }
    const applied = document.createElement("div");
    applied.className = "method-contract-line mono";
    applied.textContent = `已应用：${contractEntries(step.applied) || "无"}`;
    row.appendChild(applied);
    const notApplicable = contractEntries(step.not_applicable);
    if (notApplicable) {
      const line = document.createElement("div");
      line.className = "method-contract-line warning mono";
      line.textContent = `未应用：${notApplicable}`;
      row.appendChild(line);
    }
    const conflicts = contractEntries(step.conflicts);
    if (conflicts) {
      const line = document.createElement("div");
      line.className = "method-contract-line warning mono";
      line.textContent = `冲突：${conflicts}`;
      row.appendChild(line);
    }
    if (step.automatic_defaults?.length) {
      const line = document.createElement("div");
      line.className = "method-contract-line mono";
      line.textContent = `自动字段：${step.automatic_defaults.join("、")}`;
      row.appendChild(line);
    }
    if (step.notes?.length) {
      const line = document.createElement("div");
      line.className = "method-contract-line";
      line.textContent = step.notes.join("；");
      row.appendChild(line);
    }
    card.appendChild(row);
  }
  const execution = document.createElement("div");
  execution.className = "method-contract-execution mono";
  execution.textContent = `执行后端：${contract.execution_selection?.backend || "未记录"} · walltime ${contract.execution_selection?.walltime_limit_sec || "未记录"}s`;
  card.appendChild(execution);
  const actions = document.createElement("div");
  actions.className = "method-contract-actions";
  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.className = "cfg-btn method-contract-confirm";
  confirm.textContent = "确认参数并运行";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "cfg-btn ghost method-contract-cancel";
  cancel.textContent = "取消";
  const finishCard = (message) => {
    confirm.disabled = true;
    cancel.disabled = true;
    const state = document.createElement("span");
    state.className = "method-contract-state";
    state.textContent = message;
    actions.appendChild(state);
  };
  confirm.addEventListener("click", () => {
    if (sessionId !== requestSnapshot.sessionId || sessionLoadSerial !== requestSnapshot.sessionSerial) {
      finishCard("会话已切换，已取消");
      return;
    }
    if (activeStream && !activeStream.detached) return;
    finishCard("已确认，开始运行…");
    send(false, { ...requestSnapshot, methodToken });
  });
  cancel.addEventListener("click", () => finishCard("已取消方法确认"));
  actions.append(confirm, cancel);
  card.appendChild(actions);
  bubble.appendChild(card);
}

function addMethodUnsupported(bubble, message) {
  bubble.textContent = `方法不支持：${message || "请清空不适用参数或选择支持的任务"}`;
}

/* ── message rendering ─────────────────────────────────────────────── */

function appendUser(text, ts) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="bubble">${esc(text)}</div>${ts ? `<div class="msg-time">${fmtTime(ts)}</div>` : ""}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

const NODE_LABELS = { running: "运行中", succeeded: "完成", failed: "失败", repairing: "修复重试", ready: "重试", blocked: "阻塞", cancelled: "已停止" };
const STEP_ICONS = { pending: "○", running: "◐", done: "✓", error: "✗", cancelled: "■" };

function appendAgent(text, commands, results, ts, chain) {
  const div = document.createElement("div");
  div.className = "msg msg-agent";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  if (chain?.length) div.appendChild(renderChain(chain));
  else if (commands?.length) div.appendChild(toolTable(commands, results));
  if (ts) {
    const t = document.createElement("div");
    t.className = "msg-time";
    t.textContent = fmtTime(ts);
    div.appendChild(t);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

/* ── execution chain card (coding-agent transcript) ────────────────── */

function makeChainCard() {
  const root = document.createElement("div");
  root.className = "chain";
  root.innerHTML = `
    <div class="chain-head">
      <button class="chain-toggle" type="button" title="展开/收起执行过程">
        <span class="chain-arrow">▾</span>
        <span class="chain-phase">准备中…</span>
      </button>
      <span class="chain-elapsed mono"></span>
    </div>
    <div class="chain-live-note">展示可审计的工作进展与执行摘要，不回显模型隐藏思维。</div>
    <div class="chain-progress" aria-label="执行阶段">
      <span class="cp-stage active" data-stage="planning"><i></i>理解</span>
      <span class="cp-line"></span>
      <span class="cp-stage" data-stage="plan"><i></i>规划</span>
      <span class="cp-line"></span>
      <span class="cp-stage" data-stage="tools"><i></i>调用工具</span>
      <span class="cp-line"></span>
      <span class="cp-stage" data-stage="verify"><i></i>验证</span>
      <span class="cp-line"></span>
      <span class="cp-stage" data-stage="done"><i></i>总结</span>
    </div>
    <div class="chain-activity" aria-label="实时活动"></div>
    <div class="chain-steps"></div>`;
  root.querySelector(".chain-toggle").addEventListener("click", () => root.classList.toggle("collapsed"));
  // Give the user immediate feedback while the planner is working. The
  // planner can take several seconds (especially for a remote LLM), so an
  // empty card looks indistinguishable from a stalled request.
  const stepsEl = root.querySelector(".chain-steps");
  [
    { description: "识别计算目标与输入", tool: "planner" },
    { description: "检查结构、材料与约束", tool: "context" },
    { description: "生成可执行步骤", tool: "plan" },
  ].forEach((step, index) => {
    const el = addChainStep(stepsEl, step);
    el.className = `chain-step ${index === 0 ? "running" : "pending"}`;
    el.querySelector(".cs-icon").textContent = STEP_ICONS[index === 0 ? "running" : "pending"];
  });
  return root;
}

function addActivity(card, text, kind = "status") {
  const list = card.querySelector(".chain-activity");
  if (!list || !text) return;
  const row = document.createElement("div");
  row.className = `chain-activity-row ${kind}`;
  const icon = document.createElement("span");
  icon.className = "ca-icon";
  icon.textContent = kind === "command" ? ">_" : kind === "edit" ? "✎" : "·";
  const label = document.createElement("span");
  label.textContent = text;
  row.append(icon, label);
  list.appendChild(row);
  list.scrollIntoView({ block: "nearest" });
}

function setChainStage(card, stage) {
  const order = ["planning", "plan", "tools", "verify", "done"];
  const current = Math.max(0, order.indexOf(stage));
  for (const el of card.querySelectorAll(".cp-stage")) {
    const idx = order.indexOf(el.dataset.stage);
    el.classList.toggle("active", idx === current);
    el.classList.toggle("complete", idx < current);
  }
}

function addChainStep(stepsEl, step) {
  const el = document.createElement("div");
  el.className = "chain-step pending";
  el.innerHTML = `
    <span class="cs-icon">${STEP_ICONS.pending}</span>
    <div class="cs-row"><span class="cs-desc">${esc(step.description || step.tool || "")}</span><span class="cs-tool mono">${esc(step.tool || "")}</span></div>
    <span class="cs-summary"></span>
    <div class="cs-nodes"></div>`;
  stepsEl.appendChild(el);
  return el;
}

function setNodeRow(stepEl, node, state) {
  let row = stepEl.querySelector(`.cs-node[data-node="${CSS.escape(node)}"]`);
  if (!row) {
    row = document.createElement("div");
    row.dataset.node = node;
    row.innerHTML = `<span class="nd"></span><span class="nd-name">${esc(node)}</span><span class="nd-label"></span>`;
    stepEl.querySelector(".cs-nodes").appendChild(row);
  }
  row.className = `cs-node ${state || ""}`;
  row.querySelector(".nd-label").textContent = NODE_LABELS[state] || state || "";
}

function renderChain(chain) {
  const card = makeChainCard();
  const stepsEl = card.querySelector(".chain-steps");
  stepsEl.replaceChildren();
  const nErr = chain.filter((s) => s.status === "error").length;
  card.querySelector(".chain-phase").textContent =
    `执行过程 · ${chain.length} 步${nErr ? ` · ${nErr} 步失败` : ""}`;
  setChainStage(card, "done");
  card.querySelector(".chain-elapsed").remove();
  // ?chain=open keeps historical transcripts expanded (default: collapsed)
  if (new URLSearchParams(location.search).get("chain") !== "open") {
    card.classList.add("collapsed");
  }
  const activity = card.querySelector(".chain-activity");
  activity.replaceChildren();
  addActivity(card, `已完成执行 · ${chain.length} 步${nErr ? ` · ${nErr} 步失败` : ""}`, nErr ? "error" : "edit");
  for (const s of chain) {
    const el = addChainStep(stepsEl, s);
    el.className = `chain-step ${s.status || "done"}`;
    el.querySelector(".cs-icon").textContent = STEP_ICONS[s.status] || "·";
    if (s.summary) el.querySelector(".cs-summary").textContent = s.summary;
    addActivity(card, `${s.status === "error" ? "执行失败" : "已完成"}：${s.description || s.tool || "步骤"}`, s.status === "error" ? "error" : "command");
    for (const n of s.nodes || []) setNodeRow(el, n.node, n.state);
  }
  return card;
}

function toolTable(commands, results) {
  const wrap = document.createElement("div");
  wrap.className = "msg-tools";
  const rows = commands.map((c, i) => {
    const r = results?.[i] || {};
    const ok = r.error ? false : (r.returncode ?? 0) === 0 || r.json?.ok !== false;
    const args = Object.entries(c.args || {})
      .filter(([k, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`)
      .join("  ")
      .slice(0, 140);
    return `<tr>
      <td class="t-name">${esc(c.command)}</td>
      <td class="t-args">${esc(args)}</td>
      <td class="${ok ? "rc-ok" : "rc-err"}">${ok ? "✓" : "✗ " + esc(r.error || "失败").slice(0, 60)}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `
    <button class="tools-toggle" type="button">工具调用 · ${commands.length} 步</button>
    <table class="tools-table"><thead><tr><th>工具</th><th>参数</th><th>状态</th></tr></thead><tbody>${rows}</tbody></table>`;
  wrap.querySelector(".tools-toggle").addEventListener("click", (e) => e.currentTarget.classList.toggle("open"));
  return wrap;
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}月${d.getDate()}日 ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/* ── figures: structure + charts from tool results ─────────────────── */

function applyFigure(viewer, chart, provenance) {
  if (viewer?.cif) {
    renderCif(viewer.cif, viewer);
    const cap = document.getElementById("viewerCaption");
    cap.textContent = `${viewer.formula || ""} · ${viewer.natoms ?? "?"} 原子 · 拖拽旋转 / 滚轮缩放 / 双击复位`;
  }
  if (chart) renderCharts(chartsEl, chart, provenance || chart.provenance);
}

function applyFigures(results) {
  for (const r of results || []) applyFigure(r.viewer, r.chart, r.provenance);
}

/* Session restore: figures are scoped to the latest assistant turn. Scanning
   every historical message makes a failed/text-only follow-up resurrect the
   previous turn's structure in the viewer. */
function restoreFigures(messages) {
  let viewerPayload = null, chartPayload = null, provenancePayload = null;
  const latestAgent = [...messages].reverse().find((m) => m.role === "agent");
  for (const r of latestAgent?.results || []) {
    if (r.viewer?.cif) viewerPayload = r.viewer;
    if (r.chart) { chartPayload = r.chart; provenancePayload = r.provenance || r.chart.provenance || null; }
  }
  if (viewerPayload) {
    renderCif(viewerPayload.cif, viewerPayload);
    const cap = document.getElementById("viewerCaption");
    cap.textContent = `${viewerPayload.formula || ""} · ${viewerPayload.natoms ?? "?"} 原子 · 拖拽旋转 / 滚轮缩放`;
  }
  renderCharts(chartsEl, chartPayload, provenancePayload);
}

/* ── sessions ──────────────────────────────────────────────────────── */

async function refreshSessions() {
  const res = await fetch("/api/sessions");
  const { sessions } = await res.json();
  sessionCount.textContent = `${sessions.length} 个会话`;
  sessionList.innerHTML = "";
  for (const s of sessions) {
    const item = document.createElement("div");
    item.className = `session-item${s.session_id === sessionId ? " active" : ""}`;
    const d = new Date(s.updated_at * 1000);
    const today = new Date();
    const isToday = d.toDateString() === today.toDateString();
    const hhmm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    const when = isToday ? hhmm : `${d.getMonth() + 1}月${d.getDate()}日 ${hhmm}`;
    item.innerHTML = `
      <button class="s-del" title="删除会话">✕</button>
      <div class="s-title">${esc(s.title)}</div>
      <div class="s-meta">${when} · ${s.n_messages} 条</div>`;
    item.querySelector(".s-title").title = s.title;
    item.addEventListener("click", () => loadSession(s.session_id));
    item.querySelector(".s-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("删除该会话及其计算记录？")) return;
      const deleted = await fetch(`/api/sessions/${s.session_id}`, { method: "DELETE" });
      if (!deleted.ok) {
        const body = await deleted.json().catch(() => ({}));
        alert(body.detail || "删除失败，请先停止正在进行的计算");
        return;
      }
      if (s.session_id === sessionId) newSession();
      refreshSessions();
    });
    sessionList.appendChild(item);
  }
}

async function loadSession(id) {
  const serial = ++sessionLoadSerial;
  if (activeStream) {
    activeStream.detached = true;
    document.getElementById("stop").hidden = true;
    sendBtn.disabled = false;
  }
  sessionId = id;
  localStorage.setItem("dftforge_session", id);
  clearViewer();
  renderCharts(chartsEl, null);
  clearWorkspace();
  const res = await fetch(`/api/sessions/${id}`);
  const data = await res.json();
  if (serial !== sessionLoadSerial || sessionId !== id) return;
  chat.innerHTML = "";
  for (const m of data.messages || []) {
    if (m.role === "user") appendUser(m.text, m.ts);
    else appendAgent(m.text, m.commands, m.results, m.ts, m.chain);
  }
  if (!data.messages?.length) showWelcome();
  restoreFigures(data.messages || []);
  await restoreAttachment({ ...data, session_id: id });
  if (serial !== sessionLoadSerial || sessionId !== id) return;
  refreshWorkspace(id);
  refreshSessions();
}

function newSession() {
  sessionLoadSerial += 1;
  if (activeStream) {
    activeStream.detached = true;
    document.getElementById("stop").hidden = true;
    sendBtn.disabled = false;
  }
  sessionId = null;
  localStorage.removeItem("dftforge_session");
  chat.innerHTML = "";
  showWelcome();
  renderCharts(chartsEl, null);
  clearViewer();
  clearWorkspace();
  refreshSessions();
}

function showWelcome() {
  const w = document.createElement("div");
  w.className = "welcome";
  w.innerHTML = `
    <h2>开始一次计算</h2>
    <p>上传结构或直接描述目标。运行前可确认方法与精度，代理会规划工具、驱动计算、验证结果并保留可复现记录。</p>
    <div class="suggestions">
      <button class="suggestion">帮我算 GaAs 的结构优化</button>
      <button class="suggestion">算 Si 的能带结构</button>
      <button class="suggestion">构建 4×4 石墨烯并掺一个氮原子</button>
      <button class="suggestion">看看 NaCl 的结构</button>
    </div>`;
  for (const b of w.querySelectorAll(".suggestion")) {
    b.addEventListener("click", () => { input.value = b.textContent; send(); });
  }
  chat.appendChild(w);
}

/* ── send loop: streaming execution chain ──────────────────────────── */

async function send(forceApproval = false, frozenRequest = null) {
  if (activeStream && !activeStream.detached) return;
  if (!frozenRequest && approvalPending) return;
  const text = frozenRequest?.text || input.value.trim();
  if (!text) return;
  const approvalSessionId = sessionId;
  const approvalSessionSerial = sessionLoadSerial;
  if (frozenRequest && (sessionId !== frozenRequest.sessionId || sessionLoadSerial !== frozenRequest.sessionSerial)) {
    setStatus("会话已切换，已取消", "error");
    return;
  }
  let approval;
  if (frozenRequest) {
    approval = { proceed: true, approved: true };
  } else {
    approvalPending = true;
    try {
      approval = await approveRequest(text, forceApproval);
    } catch (error) {
      setStatus(`参数错误：${error.message}`, "error");
      return;
    } finally {
      approvalPending = false;
    }
  }
  if (!approval?.proceed) return;
  if (!frozenRequest && (sessionId !== approvalSessionId || sessionLoadSerial !== approvalSessionSerial)) {
    setStatus("会话已切换，已取消", "error");
    return;
  }
  const requestedSessionId = frozenRequest?.sessionId ?? sessionId;
  let requestSnapshot = frozenRequest;
  if (!requestSnapshot) {
    try {
      requestSnapshot = {
        text,
        sessionId: requestedSessionId,
        sessionSerial: approvalSessionSerial,
        context: requestContext(text, approval.approved),
      };
    } catch (error) {
      setStatus(`参数错误：${error.message}`, "error");
      return;
    }
  }
  const streamState = { token: Symbol("chat-stream"), sessionId: requestedSessionId, detached: false };
  activeStream = streamState;
  // A figure belongs to the current request. Do not let a failed or
  // text-only follow-up inherit the previous turn's structure/chart.
  clearViewer();
  renderCharts(chartsEl, null);
  input.value = "";
  sendBtn.disabled = true;
  setStatus("执行中…", "busy");
  const welcome = chat.querySelector(".welcome");
  if (welcome) welcome.remove();
  if (!frozenRequest) appendUser(text);

  const card = makeChainCard();
  const stepsEl = card.querySelector(".chain-steps");
  const phaseEl = card.querySelector(".chain-phase");
  const elapsedEl = card.querySelector(".chain-elapsed");
  const msgDiv = document.createElement("div");
  msgDiv.className = "msg msg-agent";
  msgDiv.appendChild(card);
  chat.appendChild(msgDiv);
  chat.scrollTop = chat.scrollHeight;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const t0 = performance.now();
  const timer = setInterval(() => {
    elapsedEl.textContent = `${Math.round((performance.now() - t0) / 1000)}s`;
  }, 1000);
  let finished = false;
  let streamFailed = false;
  let streamCancelled = false;
  let streamNeedsConfirmation = false;

  const isAttached = () => activeStream === streamState && !streamState.detached && sessionId === streamState.sessionId;

  const stopBtn = document.getElementById("stop");
  const stop = async () => {
    if (!streamState.sessionId || streamState.detached) return;
    stopBtn.disabled = true;
    try {
      const r = await fetch(`/api/sessions/${streamState.sessionId}/cancel`, { method: "POST" });
      const d = await r.json();
      streamCancelled = Boolean(d.cancelled);
      phaseEl.textContent = d.cancelled ? "已请求停止…" : "无进行中的计算";
      if (d.cancelled && d.processes_killed > 0) {
        phaseEl.textContent += `（终止了 ${d.processes_killed} 个进程）`;
      }
    } catch { /* backend gone — stream will end on its own */ }
    stopBtn.disabled = false;
  };
  const stopHandler = () => { stop(); };
  stopBtn.hidden = false;
  stopBtn.addEventListener("click", stopHandler);

  const finish = () => {
    if (finished) return;
    finished = true;
    clearInterval(timer);
    const attached = isAttached();
    if (activeStream === streamState) activeStream = null;
    if (attached || !streamState.detached) stopBtn.hidden = true;
    stopBtn.removeEventListener("click", stopHandler);
    if (attached) {
      phaseEl.textContent = streamFailed ? "执行出错" : streamNeedsConfirmation ? "等待方法确认" : streamCancelled ? "已停止" : "执行完成";
      setStatus(streamFailed ? "错误" : streamNeedsConfirmation ? "等待方法确认" : streamCancelled ? "已停止" : "就绪", streamFailed ? "error" : "");
      sendBtn.disabled = false;
      input.focus();
    }
    refreshSessions();
  };

  const handleEvent = (ev) => {
    if (ev.type !== "start" && !isAttached()) return;
    switch (ev.type) {
      case "start":
        if (streamState.detached || (requestedSessionId && ev.session_id !== requestedSessionId)) return;
        streamState.sessionId = ev.session_id;
        if (!requestSnapshot.sessionId) requestSnapshot.sessionId = ev.session_id;
        sessionId = ev.session_id;
        localStorage.setItem("dftforge_session", sessionId);
        addActivity(card, "已接收请求");
        refreshWorkspace(sessionId);
        break;
      case "phase":
        phaseEl.textContent = ev.label || ev.phase;
        card.classList.toggle("phase-planning", ev.phase === "planning");
        setChainStage(card, ev.phase === "narrating" ? "verify" : "planning");
        addActivity(card, ev.label || ev.phase);
        break;
      case "context":
        addActivity(card, ev.label || "已加载会话上下文");
        break;
      case "plan":
        stepsEl.replaceChildren();
        phaseEl.textContent = `${ev.planner === "llm" ? "LLM" : "规则"}规划 · ${ev.steps.length} 步`;
        setChainStage(card, "plan");
        addActivity(card, `${ev.planner === "llm" ? "LLM" : "规则"} 已生成执行计划 · ${ev.steps.length} 步`, "command");
        if (ev.summary) addActivity(card, ev.summary, "status");
        if (ev.goal) addActivity(card, `目标已确认：${ev.goal}`, "status");
        if (ev.constraints && Object.keys(ev.constraints).length) {
          addActivity(card, `约束：${Object.entries(ev.constraints).map(([k, v]) => `${k}=${v}`).join(" · ")}`, "status");
        }
        if (ev.assumptions?.length) addActivity(card, `假设：${ev.assumptions.join("；")}`, "status");
        if (ev.expected_outputs?.length) addActivity(card, `预期结果：${ev.expected_outputs.join("、")}`, "status");
        for (const s of ev.steps) addChainStep(stepsEl, s);
        chat.scrollTop = chat.scrollHeight;
        break;
      case "step": {
        setChainStage(card, "tools");
        const el = stepsEl.children[ev.index];
        if (!el) break;
        if (ev.status === "cancelled") streamCancelled = true;
        el.className = `chain-step ${ev.status}`;
        el.querySelector(".cs-icon").textContent = STEP_ICONS[ev.status] || "·";
        if (ev.summary) el.querySelector(".cs-summary").textContent = ev.summary;
        const stepLabel = el.querySelector(".cs-desc")?.textContent || `步骤 ${ev.index + 1}`;
        addActivity(card, `${ev.status === "running" ? "正在执行" : ev.status === "done" ? "已完成" : "执行失败"}：${stepLabel}`, ev.status === "error" ? "error" : "command");
        chat.scrollTop = chat.scrollHeight;
        break;
      }
      case "node": {
        const el = stepsEl.children[ev.step];
        if (el) setNodeRow(el, ev.node, ev.state);
        addActivity(card, `${ev.node} · ${NODE_LABELS[ev.state] || ev.state}`, ev.state === "failed" ? "error" : "status");
        chat.scrollTop = chat.scrollHeight;
        break;
      }
      case "figure":
        applyFigure(ev.viewer, ev.chart, ev.provenance);
        break;
      case "reply":
        setChainStage(card, "done");
        addActivity(card, "已整理结果并生成回答", "edit");
        bubble.textContent = ev.text;
        if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
        chat.scrollTop = chat.scrollHeight;
        break;
      case "error":
        if (ev.code === "method_confirmation_required") {
          streamNeedsConfirmation = true;
          bubble.textContent = "已生成计算参数，请确认后开始运行。";
          if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
          addMethodContract(msgDiv, bubble, ev.method_contract, ev.method_token, requestSnapshot);
          phaseEl.textContent = "等待方法确认";
          chat.scrollTop = chat.scrollHeight;
          break;
        }
        streamFailed = true;
        bubble.textContent = `执行出错：${ev.message || "未知错误"}`;
        if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
        if (ev.code === "approval_required") addApprovalRetry(msgDiv, bubble, text);
        else if (ev.code === "method_unsupported") addMethodUnsupported(bubble, ev.message);
        else if (ev.code === "method_confirmation_expired") addMethodUnsupported(bubble, "方法确认已过期，请重新发送请求生成新的确认卡");
        phaseEl.textContent = "执行出错";
        break;
      case "done":
        setChainStage(card, "done");
        finish();
        break;
    }
  };

  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: requestedSessionId, ...requestSnapshot.context, approved: approval.approved, method_token: frozenRequest?.methodToken }),
    });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({}));
      const error = new Error(body.detail || `HTTP ${res.status}`);
      error.policyCode = res.headers.get("X-DFT-Policy") || body.code || "";
      error.methodContract = body.method_contract;
      error.methodToken = body.method_token;
      throw error;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop();
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        try {
          handleEvent(JSON.parse(line.slice(6)));
        } catch { /* skip malformed frame */ }
      }
    }
  } catch (e) {
    streamNeedsConfirmation = e.policyCode === "method_confirmation_required";
    streamFailed = !streamNeedsConfirmation;
    if (isAttached()) {
      bubble.textContent = streamNeedsConfirmation ? "已生成计算参数，请确认后开始运行。" : `请求失败：${e.message}`;
      if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
      if (e.policyCode === "approval_required") addApprovalRetry(msgDiv, bubble, text);
      else if (e.policyCode === "method_confirmation_required") addMethodContract(msgDiv, bubble, e.methodContract, e.methodToken, requestSnapshot);
      else if (e.policyCode === "method_unsupported") addMethodUnsupported(bubble, e.message);
      else if (e.policyCode === "method_confirmation_expired") addMethodUnsupported(bubble, "方法确认已过期，请重新发送请求生成新的确认卡");
      setStatus(streamNeedsConfirmation ? "等待方法确认" : "错误", streamNeedsConfirmation ? "" : "error");
    }
  } finally {
    finish();
    if (!streamState.detached && streamState.sessionId) refreshWorkspace(streamState.sessionId);
  }
}

/* ── init ──────────────────────────────────────────────────────────── */

export function initChat() {
  form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  document.getElementById("newSession").addEventListener("click", newSession);
  document.getElementById("toggleReplica").addEventListener("change", (e) => toggleReplica(e.target.checked));
  document.getElementById("toggleCell").addEventListener("change", (e) => toggleCellFrame(e.target.checked));

  window.addEventListener("dftforge:session", (event) => { sessionId = event.detail.sessionId; });
  window.addEventListener("dftforge:running", (event) => {
    const detail = event.detail || {};
    if (detail.sessionId !== sessionId || activeStream) return;
    sendBtn.disabled = Boolean(detail.running);
    if (detail.running) setStatus(detail.cancelling ? "正在停止…" : "任务运行中…", "busy");
    else if (statusEl.classList.contains("busy")) setStatus("就绪");
  });

  Promise.all([fetch("/doctor").then((r) => r.json()), fetch("/api/config").then((r) => r.json())]).then(([doctor, cfg]) => {
    const remote = cfg.compute?.executor === "ssh";
    const demo = cfg.compute?.executor === "fake";
    const hasQe = Boolean(doctor.qe_binary) || remote || demo;
    const label = remote ? `HPC · ${cfg.compute.scheduler || "direct"}` : demo ? "演示后端" : doctor.qe_binary ? "pw.x 已就绪" : "QE 未检出";
    engineStatus.className = `masthead-status ${hasQe ? "ok" : "warn"}`;
    engineStatus.innerHTML = `<span class="dot"></span>${label}`;
  }).catch(() => {
    engineStatus.className = "masthead-status warn";
    engineStatus.innerHTML = '<span class="dot"></span>后端未知状态';
  });

  const urlSession = new URLSearchParams(location.search).get("session");
  if (urlSession) sessionId = urlSession;
  if (sessionId) loadSession(sessionId).catch(() => newSession());
  else newSession();
}
