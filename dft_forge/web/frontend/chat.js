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

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = `statusline mono ${cls}`.trim();
}

/* ── message rendering ─────────────────────────────────────────────── */

function appendUser(text, ts) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="bubble">${esc(text)}</div>${ts ? `<div class="msg-time">${fmtTime(ts)}</div>` : ""}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

const NODE_LABELS = { running: "运行中", succeeded: "完成", failed: "失败", repairing: "修复重试", ready: "重试", blocked: "阻塞" };
const STEP_ICONS = { pending: "○", running: "◐", done: "✓", error: "✗" };

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
  const nErr = chain.filter((s) => s.status === "error").length;
  card.querySelector(".chain-phase").textContent =
    `执行过程 · ${chain.length} 步${nErr ? ` · ${nErr} 步失败` : ""}`;
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

function applyFigure(viewer, chart) {
  if (viewer?.cif) {
    renderCif(viewer.cif, viewer);
    const cap = document.getElementById("viewerCaption");
    cap.textContent = `${viewer.formula || ""} · ${viewer.natoms ?? "?"} 原子 · 拖拽旋转 / 滚轮缩放 / 双击复位`;
  }
  if (chart) renderCharts(chartsEl, chart);
}

function applyFigures(results) {
  for (const r of results || []) applyFigure(r.viewer, r.chart);
}

/* Session restore: figures are scoped to the latest assistant turn. Scanning
   every historical message makes a failed/text-only follow-up resurrect the
   previous turn's structure in the viewer. */
function restoreFigures(messages) {
  let viewerPayload = null, chartPayload = null;
  const latestAgent = [...messages].reverse().find((m) => m.role === "agent");
  for (const r of latestAgent?.results || []) {
    if (r.viewer?.cif) viewerPayload = r.viewer;
    if (r.chart) chartPayload = r.chart;
  }
  if (viewerPayload) {
    renderCif(viewerPayload.cif, viewerPayload);
    const cap = document.getElementById("viewerCaption");
    cap.textContent = `${viewerPayload.formula || ""} · ${viewerPayload.natoms ?? "?"} 原子 · 拖拽旋转 / 滚轮缩放`;
  }
  renderCharts(chartsEl, chartPayload);
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
  sessionId = id;
  localStorage.setItem("dftforge_session", id);
  const res = await fetch(`/api/sessions/${id}`);
  const data = await res.json();
  chat.innerHTML = "";
  for (const m of data.messages || []) {
    if (m.role === "user") appendUser(m.text, m.ts);
    else appendAgent(m.text, m.commands, m.results, m.ts, m.chain);
  }
  if (!data.messages?.length) showWelcome();
  restoreFigures(data.messages || []);
  restoreAttachment(data);
  refreshWorkspace(id);
  refreshSessions();
}

function newSession() {
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

async function send() {
  const text = input.value.trim();
  if (!text) return;
  const approved = await approveRequest(text);
  if (!approved) return;
  // A figure belongs to the current request. Do not let a failed or
  // text-only follow-up inherit the previous turn's structure/chart.
  clearViewer();
  renderCharts(chartsEl, null);
  input.value = "";
  sendBtn.disabled = true;
  setStatus("执行中…", "busy");
  const welcome = chat.querySelector(".welcome");
  if (welcome) welcome.remove();
  appendUser(text);

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

  const stopBtn = document.getElementById("stop");
  const stop = async () => {
    if (!sessionId) return;
    stopBtn.disabled = true;
    try {
      const r = await fetch(`/api/sessions/${sessionId}/cancel`, { method: "POST" });
      const d = await r.json();
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
    stopBtn.hidden = true;
    stopBtn.removeEventListener("click", stopHandler);
    phaseEl.textContent = "执行完成";
    setStatus("就绪");
    sendBtn.disabled = false;
    input.focus();
    refreshSessions();
  };

  const handleEvent = (ev) => {
    switch (ev.type) {
      case "start":
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
        stepsEl.innerHTML = "";
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
        applyFigure(ev.viewer, ev.chart);
        break;
      case "reply":
        setChainStage(card, "done");
        addActivity(card, "已整理结果并生成回答", "edit");
        bubble.textContent = ev.text;
        if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
        chat.scrollTop = chat.scrollHeight;
        break;
      case "error":
        bubble.textContent = `执行出错：${ev.message || "未知错误"}`;
        if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
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
      body: JSON.stringify({ message: text, session_id: sessionId, ...requestContext(text, approved) }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
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
    bubble.textContent = `请求失败：${e.message}`;
    if (!bubble.parentNode) msgDiv.insertBefore(bubble, card);
    setStatus("错误", "error");
  } finally {
    finish();
    refreshWorkspace(sessionId);
  }
}

/* ── init ──────────────────────────────────────────────────────────── */

export function initChat() {
  form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  document.getElementById("newSession").addEventListener("click", newSession);
  document.getElementById("toggleReplica").addEventListener("change", (e) => toggleReplica(e.target.checked));
  document.getElementById("toggleCell").addEventListener("change", (e) => toggleCellFrame(e.target.checked));

  window.addEventListener("dftforge:session", (event) => { sessionId = event.detail.sessionId; });

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
