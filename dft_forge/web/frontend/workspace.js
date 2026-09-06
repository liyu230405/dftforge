/* Product workspace: structures, method approval, artifacts and run recovery. */

import { renderCif } from "./viewer.js";

const fileInput = document.getElementById("structureFile");
const attachmentChip = document.getElementById("attachmentChip");
const modeSelect = document.getElementById("runMode");
const accuracySelect = document.getElementById("runAccuracy");
const runContext = document.getElementById("runContext");
const methodDlg = document.getElementById("methodDlg");
const customMethod = document.getElementById("customMethod");
const methodSummary = document.getElementById("methodSummary");

let attachedStructure = null;
let pendingApproval = null;

const ACCURACY = {
  quick: { label: "快速预览", nkpoints_bands: 60 },
  balanced: { label: "材料默认 / 均衡" },
  precise: { label: "致密 k 网格", kpoints: [8, 8, 8, 1, 1, 1], nkpoints_bands: 160 },
};

function ensureSessionId() {
  let id = localStorage.getItem("dftforge_session");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("dftforge_session", id);
    window.dispatchEvent(new CustomEvent("dftforge:session", { detail: { sessionId: id } }));
  }
  return id;
}

function currentRunConfig() {
  const accuracy = accuracySelect.value;
  const config = { mode: modeSelect.value, accuracy };
  if (accuracy === "custom") {
    const ecut = Number(document.getElementById("methodEcut").value || 0);
    const nbnd = Number(document.getElementById("methodNbnd").value || 0);
    const bandPoints = Number(document.getElementById("methodBandPoints").value || 0);
    const kpoints = document.getElementById("methodKpoints").value.split(",").map((value) => Number(value.trim()));
    if (ecut > 0) config.ecutwfc = ecut;
    if (nbnd > 0) config.nbnd = nbnd;
    if (bandPoints >= 10) config.nkpoints_bands = bandPoints;
    if (kpoints.length === 3 && kpoints.every((value) => Number.isInteger(value) && value > 0)) config.kpoints = [...kpoints, 1, 1, 1];
  } else {
    Object.assign(config, ACCURACY[accuracy] || {});
    delete config.label;
  }
  return config;
}

function looksLikeCalculation(text) {
  return /(计算|优化|能带|态密度|dos|scf|relax|提交|运行|形成能|吸附能|band|density)/i.test(text);
}

function methodRows(text) {
  const config = currentRunConfig();
  const structure = attachedStructure ? `${attachedStructure.formula} · ${attachedStructure.natoms} 原子` : "由请求或材料库确定";
  const precision = accuracySelect.options[accuracySelect.selectedIndex].text;
  return [
    ["任务", text.slice(0, 70)], ["结构", structure],
    ["运行模式", modeSelect.value === "research" ? "科研模式 · 需要确认" : "演示模式"], ["精度策略", precision],
    ["截断能", config.ecutwfc ? `${config.ecutwfc} Ry` : "材料/伪势默认"], ["k 点", config.kpoints ? config.kpoints.slice(0, 3).join(" × ") : "结构自适应"],
  ];
}

function renderMethodSummary(text) {
  methodSummary.replaceChildren();
  for (const [label, value] of methodRows(text)) {
    const row = document.createElement("div");
    row.className = "method-row";
    const caption = document.createElement("span");
    caption.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    row.append(caption, strong);
    methodSummary.appendChild(row);
  }
  customMethod.hidden = accuracySelect.value !== "custom";
}

export async function approveRequest(text) {
  if (!looksLikeCalculation(text) || modeSelect.value === "demo") return true;
  renderMethodSummary(text);
  methodDlg.showModal();
  return new Promise((resolve) => { pendingApproval = resolve; });
}

export function requestContext(text, approved) {
  if (!looksLikeCalculation(text)) return {};
  return {
    structure_path: attachedStructure?.path || null,
    run_config: currentRunConfig(),
    approved: Boolean(approved),
  };
}

function settleApproval(value) {
  methodDlg.close();
  if (pendingApproval) pendingApproval(value);
  pendingApproval = null;
}

async function uploadStructure(file) {
  const sessionId = ensureSessionId();
  runContext.textContent = "正在检查结构…";
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`/api/sessions/${sessionId}/structure`, { method: "POST", body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  attachedStructure = data;
  attachmentChip.hidden = false;
  attachmentChip.textContent = `${data.formula} · ${data.natoms} 原子`;
  attachmentChip.title = data.name;
  runContext.textContent = `已校验 ${data.name}`;
  renderCif(data.cif, data);
  const caption = document.getElementById("viewerCaption");
  caption.textContent = `${data.formula} · ${data.natoms} 原子 · 上传结构`;
  await refreshWorkspace(sessionId);
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function renderArtifacts(items) {
  const list = document.getElementById("artifactList");
  list.replaceChildren();
  if (!items.length) {
    list.innerHTML = '<div class="fig-empty">当前会话还没有文件</div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "artifact-item";
    const icon = document.createElement("span");
    icon.className = "artifact-icon";
    icon.textContent = item.kind.slice(0, 3).toUpperCase();
    const body = document.createElement("div");
    const name = document.createElement("div");
    name.className = "artifact-name";
    name.textContent = item.name;
    name.title = item.path;
    const meta = document.createElement("div");
    meta.className = "artifact-meta";
    meta.textContent = `${formatSize(item.size)}${item.sha256 ? ` · ${item.sha256.slice(0, 10)}` : ""}`;
    body.append(name, meta);
    const link = document.createElement("a");
    link.className = "artifact-download";
    link.href = item.download_url;
    link.textContent = "下载";
    row.append(icon, body, link);
    list.appendChild(row);
  }
}

function renderRuns(items) {
  const list = document.getElementById("runList");
  list.replaceChildren();
  if (!items.length) {
    list.innerHTML = '<div class="fig-empty">当前会话还没有图任务</div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "run-item";
    const dot = document.createElement("span");
    dot.className = `run-state ${item.status}`;
    const body = document.createElement("div");
    const name = document.createElement("div");
    name.className = "run-name";
    name.textContent = item.template_id;
    const meta = document.createElement("div");
    meta.className = "run-meta";
    meta.textContent = `${item.status} · ${item.run_id}`;
    body.append(name, meta);
    row.append(dot, body);
    list.appendChild(row);
  }
}

export async function refreshWorkspace(sessionId = localStorage.getItem("dftforge_session")) {
  if (!sessionId) {
    renderArtifacts([]);
    renderRuns([]);
    return;
  }
  const [artifactResponse, runResponse] = await Promise.all([
    fetch(`/api/sessions/${sessionId}/artifacts`), fetch("/api/runs"),
  ]);
  if (artifactResponse.ok) renderArtifacts((await artifactResponse.json()).artifacts || []);
  if (runResponse.ok) {
    const allRuns = (await runResponse.json()).runs || [];
    renderRuns(allRuns.filter((run) => run.session_id === sessionId));
  }
}

async function exportEvidence() {
  const sessionId = localStorage.getItem("dftforge_session");
  if (!sessionId) return;
  const response = await fetch(`/api/sessions/${sessionId}/reproducibility`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `dft-forge-${sessionId}-reproducibility.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function runAction(action) {
  const sessionId = localStorage.getItem("dftforge_session");
  if (!sessionId) return;
  runContext.textContent = action === "retry" ? "正在重试失败节点…" : "正在恢复任务…";
  const response = await fetch(`/api/sessions/${sessionId}/runs/action`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }),
  });
  const data = await response.json();
  runContext.textContent = response.ok ? `任务 ${data.state || "已完成"}` : (data.detail || "操作失败");
  await refreshWorkspace(sessionId);
}

function switchPanel(panel) {
  for (const tab of document.querySelectorAll(".panel-tab")) tab.classList.toggle("active", tab.dataset.panel === panel);
  document.getElementById("panelObserve").classList.toggle("active", panel === "observe");
  document.getElementById("panelFiles").classList.toggle("active", panel === "files");
  document.getElementById("panelEvidence").classList.toggle("active", panel === "evidence");
  if (panel !== "observe") refreshWorkspace();
}

export function restoreAttachment(session) {
  const attachment = (session?.attachments || []).at(-1);
  if (!attachment) return;
  attachedStructure = attachment;
  attachmentChip.hidden = false;
  attachmentChip.textContent = `${attachment.formula} · ${attachment.natoms} 原子`;
}

export function clearWorkspace() {
  attachedStructure = null;
  attachmentChip.hidden = true;
  runContext.textContent = "等待结构与任务";
  renderArtifacts([]);
  renderRuns([]);
}

export function initWorkspace() {
  modeSelect.value = localStorage.getItem("dftforge_run_mode") || "research";
  accuracySelect.value = localStorage.getItem("dftforge_accuracy") || "balanced";
  document.getElementById("attachStructure").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) uploadStructure(file).catch((error) => { runContext.textContent = error.message; });
    fileInput.value = "";
  });
  modeSelect.addEventListener("change", () => localStorage.setItem("dftforge_run_mode", modeSelect.value));
  accuracySelect.addEventListener("change", () => { localStorage.setItem("dftforge_accuracy", accuracySelect.value); renderMethodSummary("下一次计算"); });
  document.getElementById("openMethod").addEventListener("click", () => { renderMethodSummary("下一次计算"); methodDlg.showModal(); });
  document.getElementById("methodConfirm").addEventListener("click", () => settleApproval(true));
  document.getElementById("methodCancel").addEventListener("click", () => settleApproval(false));
  document.getElementById("methodClose").addEventListener("click", () => settleApproval(false));
  for (const tab of document.querySelectorAll(".panel-tab")) tab.addEventListener("click", () => switchPanel(tab.dataset.panel));
  document.getElementById("refreshArtifacts").addEventListener("click", () => refreshWorkspace());
  document.getElementById("exportEvidence").addEventListener("click", () => exportEvidence().catch((error) => { runContext.textContent = error.message; }));
  document.getElementById("retryRun").addEventListener("click", () => runAction("retry"));
  document.getElementById("resumeRun").addEventListener("click", () => runAction("resume"));
  refreshWorkspace();
}
