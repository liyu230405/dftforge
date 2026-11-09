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
const methodAccuracy = document.getElementById("methodAccuracy");
const methodConverge = document.getElementById("methodConverge");
const runDetail = document.getElementById("runDetail");
const workspaceNotice = document.getElementById("workspaceNotice");
const stopRunButton = document.getElementById("stopRun");
const exportRunEvidenceButton = document.getElementById("exportRunEvidence");

let attachedStructure = null;
let methodSettingsSnapshot = null;
let workspaceSessionId = null;
let workspaceGeneration = 0;
let selectedRunId = null;
let selectedRun = null;
let pollTimer = null;
let pollInFlight = false;
let noticeTimer = null;
let lastWorkspaceError = "";

const METHOD_INPUTS = ["methodEcut", "methodEcutrho", "methodNbnd", "methodBandPoints", "methodKpoints"];

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
  const config = { mode: modeSelect.value, accuracy, auto_parameters: true, converge_first: methodConverge.checked };
  if (accuracy === "custom") {
    const ecutText = document.getElementById("methodEcut").value.trim();
    const ecutrhoText = document.getElementById("methodEcutrho").value.trim();
    const nbndText = document.getElementById("methodNbnd").value.trim();
    const bandPointsText = document.getElementById("methodBandPoints").value.trim();
    const kpointsText = document.getElementById("methodKpoints").value.trim();
    const positiveNumber = (text, label) => {
      if (!text) return null;
      const value = Number(text);
      if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} 必须是大于 0 的数字`);
      return value;
    };
    const ecut = positiveNumber(ecutText, "ecutwfc");
    const ecutrho = positiveNumber(ecutrhoText, "ecutrho");
    const nbnd = positiveNumber(nbndText, "nbnd");
    const bandPoints = positiveNumber(bandPointsText, "能带路径点数");
    if (ecut !== null) config.ecutwfc = ecut;
    if (ecutrho !== null) config.ecutrho = ecutrho;
    if (nbnd !== null) {
      if (!Number.isInteger(nbnd)) throw new Error("nbnd 必须是大于 0 的整数");
      config.nbnd = nbnd;
    }
    if (bandPoints !== null) {
      if (!Number.isInteger(bandPoints) || bandPoints < 10) throw new Error("能带路径点数必须是至少 10 的整数");
      config.nkpoints_bands = bandPoints;
    }
    if (kpointsText) {
      const values = kpointsText.split(",").map((value) => Number(value.trim()));
      if (values.length !== 3 || !values.every((value) => Number.isInteger(value) && value > 0)) {
        throw new Error("k 点网格必须是三个大于 0 的整数，例如 6,6,6");
      }
      config.kpoints = [...values, 1, 1, 1];
    }
  }
  return config;
}

function methodRows(text) {
  const structure = attachedStructure ? `${attachedStructure.formula} · ${attachedStructure.natoms} 原子` : "由请求或材料库确定";
  const precision = accuracySelect.options[accuracySelect.selectedIndex].text;
  const cutoff = (id) => accuracySelect.value === "custom" && document.getElementById(id).value.trim()
    ? `${document.getElementById(id).value.trim()} Ry · 自定义` : "自动推荐";
  return [
    ["任务", text.slice(0, 70)], ["结构", structure],
    ["运行模式", modeSelect.value === "research" ? "科研模式 · 规划后确认" : "演示模式"], ["精度策略", precision],
    ["参数推荐", "按任务、结构与赝势生成"], ["收敛验证", methodConverge.checked ? "先扫描，范围待规划" : "仅初始推荐"],
    ["ecutwfc", cutoff("methodEcut")], ["ecutrho", cutoff("methodEcutrho")],
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
  methodAccuracy.value = accuracySelect.value;
  customMethod.hidden = accuracySelect.value !== "custom";
}

export async function approveRequest(text, forceApproval = false) {
  // Approval belongs to the concrete, planned method contract returned by
  // the backend; saving preferences must never pre-approve a calculation.
  return { proceed: true, approved: false };
}

export function requestContext(text, approved) {
  return {
    structure_path: attachedStructure?.path || null,
    run_config: currentRunConfig(),
    approved: Boolean(approved),
  };
}

function openMethodSettings() {
  methodSettingsSnapshot = {
    accuracy: accuracySelect.value,
    converge: methodConverge.checked,
    values: METHOD_INPUTS.map((id) => document.getElementById(id).value),
  };
  renderMethodSummary("下一次计算");
  methodDlg.showModal();
}

function settleMethodSettings(save) {
  if (save) {
    try { currentRunConfig(); }
    catch (error) { showNotice(`参数错误：${error.message}`); return; }
    localStorage.setItem("dftforge_accuracy", accuracySelect.value);
    localStorage.setItem("dftforge_converge_first", String(methodConverge.checked));
    runContext.textContent = "方法设置已保存 · 发送任务后生成推荐";
  } else if (methodSettingsSnapshot) {
    accuracySelect.value = methodSettingsSnapshot.accuracy;
    methodConverge.checked = methodSettingsSnapshot.converge;
    METHOD_INPUTS.forEach((id, index) => { document.getElementById(id).value = methodSettingsSnapshot.values[index]; });
  }
  methodSettingsSnapshot = null;
  methodDlg.close();
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

function showNotice(message, kind = "error") {
  if (!workspaceNotice) return;
  workspaceNotice.textContent = message || "";
  workspaceNotice.className = `workspace-notice ${kind}`.trim();
  workspaceNotice.hidden = !message;
  clearTimeout(noticeTimer);
  if (message) noticeTimer = setTimeout(() => { workspaceNotice.hidden = true; }, 7000);
}

function clearRunDetail() {
  selectedRunId = null;
  selectedRun = null;
  if (runDetail) {
    runDetail.replaceChildren();
    runDetail.hidden = true;
  }
  if (document.getElementById("retryRun")) document.getElementById("retryRun").disabled = true;
  if (document.getElementById("resumeRun")) document.getElementById("resumeRun").disabled = true;
  if (exportRunEvidenceButton) exportRunEvidenceButton.disabled = true;
}

function scalarOutputText(outputs) {
  if (!outputs || typeof outputs !== "object") return "";
  return Object.entries(outputs)
    .filter(([, value]) => value === null || ["string", "number", "boolean"].includes(typeof value))
    .map(([key, value]) => `${key}=${value}`)
    .join(" · ");
}

function provenanceLabel(item) {
  const execution = item?.metadata?.execution;
  if (!execution || typeof execution !== "object") return "历史来源未记录";
  const rawOrigin = execution.origin;
  const origin = rawOrigin && typeof rawOrigin === "object" ? rawOrigin : {};
  const profile = (origin.profile && typeof origin.profile === "object")
    ? origin.profile
    : ((execution.profile && typeof execution.profile === "object") ? execution.profile : {});
  const backend = String(profile.backend || "").toLowerCase();
  const simulated = profile.simulated === true || execution.simulated === true || backend === "fake";
  if (execution.origin_unknown === true) return simulated ? "来源不完整 · 含模拟尝试 · 非科研结论" : "历史来源未记录";
  if (simulated) return "模拟结果 · 非科学结论";
  if (profile.simulated !== false) return "历史来源未记录";
  let label = backend === "local" ? "本地计算" : backend === "ssh" ? "远程计算" : "历史来源未记录";
  const profileName = profile.backend || profile.name || "";
  const walltime = profile.walltime_limit_sec ?? execution.walltime_sec ?? execution.walltime;
  const suffix = [profileName, walltime !== undefined ? `walltime ${walltime}s` : ""].filter(Boolean).join(" · ");
  return suffix ? `${label} · ${suffix}` : label;
}

function renderRunDetail(detail) {
  if (!runDetail) return;
  runDetail.replaceChildren();
  runDetail.hidden = false;
  const heading = document.createElement("div");
  heading.className = "run-detail-head";
  const title = document.createElement("strong");
  title.textContent = detail.template_id || detail.run_id || "图任务";
  const summary = document.createElement("span");
  summary.className = "run-meta";
  summary.textContent = `${detail.status || "unknown"} · ${detail.run_id || ""}`;
  heading.append(title, summary);
  runDetail.appendChild(heading);
  const provenance = document.createElement("div");
  provenance.className = "run-provenance";
  provenance.textContent = provenanceLabel(detail);
  runDetail.appendChild(provenance);

  const nodes = detail.nodes && typeof detail.nodes === "object" ? detail.nodes : {};
  for (const [nodeId, node] of Object.entries(nodes)) {
    const info = node && typeof node === "object" ? node : {};
    const row = document.createElement("div");
    row.className = `run-node ${info.state || ""}`;
    const label = document.createElement("div");
    label.className = "run-node-title";
    label.textContent = `${nodeId} · ${info.state || "unknown"}`;
    const attempts = document.createElement("span");
    attempts.className = "run-meta";
    attempts.textContent = `attempt ${info.attempt ?? 0} · repair ${info.repair_attempts ?? 0}`;
    label.appendChild(attempts);
    row.appendChild(label);
    if (info.error) {
      const error = document.createElement("div");
      error.className = "run-node-error";
      error.textContent = String(info.error);
      row.appendChild(error);
    }
    const outputs = scalarOutputText(info.outputs);
    if (outputs) {
      const values = document.createElement("div");
      values.className = "run-node-output mono";
      values.textContent = outputs;
      row.appendChild(values);
    }
    runDetail.appendChild(row);
  }
}

function renderArtifacts(items) {
  const list = document.getElementById("artifactList");
  list.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "fig-empty";
    empty.textContent = "当前会话还没有文件";
    list.appendChild(empty);
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
  if (selectedRunId && !items.some((item) => item.run_id === selectedRunId)) clearRunDetail();
  if (!items.length) {
    clearRunDetail();
    const empty = document.createElement("div");
    empty.className = "fig-empty";
    empty.textContent = "当前会话还没有图任务";
    list.appendChild(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = `run-item${item.run_id === selectedRunId ? " active" : ""}`;
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    const dot = document.createElement("span");
    dot.className = `run-state ${item.status}`;
    const body = document.createElement("div");
    const name = document.createElement("div");
    name.className = "run-name";
    name.textContent = item.template_id;
    const meta = document.createElement("div");
    meta.className = "run-meta";
    meta.textContent = `${item.status} · ${item.run_id}${item.active ? " · 运行中" : ""}`;
    const provenance = document.createElement("div");
    provenance.className = "run-provenance";
    provenance.textContent = provenanceLabel(item);
    body.append(name, meta, provenance);
    row.append(dot, body);
    const select = () => selectRun(item);
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
    list.appendChild(row);
  }
}

async function selectRun(item) {
  if (!workspaceSessionId || !item?.run_id) return;
  const sessionId = workspaceSessionId;
  const runId = item.run_id;
  selectedRunId = runId;
  selectedRun = item;
  try {
    const runs = await fetchRuns(sessionId);
    if (sessionId !== workspaceSessionId || sessionId !== localStorage.getItem("dftforge_session") || selectedRunId !== runId) return;
    renderRuns(runs);
    const response = await fetch(`/api/sessions/${sessionId}/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    if (sessionId === workspaceSessionId && sessionId === localStorage.getItem("dftforge_session") && selectedRunId === runId) {
      renderRunDetail(data);
      setRunActionAvailability(data);
    }
  } catch (error) {
    if (sessionId === workspaceSessionId && sessionId === localStorage.getItem("dftforge_session") && selectedRunId === runId) {
      showNotice(`任务详情加载失败：${error.message}`);
    }
  }
}

async function fetchRuns(sessionId) {
  const response = await fetch(`/api/sessions/${sessionId}/runs`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data.runs || [];
}

function setRunActionAvailability(item) {
  const actions = item?.actions || {};
  const retry = document.getElementById("retryRun");
  const resume = document.getElementById("resumeRun");
  if (retry) retry.disabled = !actions.retry;
  if (resume) resume.disabled = !actions.resume;
  if (exportRunEvidenceButton) exportRunEvidenceButton.disabled = !selectedRunId;
}

async function refreshSelectedRunDetail(sessionId, generation = workspaceGeneration, silent = true) {
  const runId = selectedRunId;
  if (!sessionId || !runId) return;
  try {
    const response = await fetch(`/api/sessions/${sessionId}/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    if (generation !== workspaceGeneration || sessionId !== workspaceSessionId || selectedRunId !== runId || sessionId !== localStorage.getItem("dftforge_session")) return;
    selectedRun = data;
    renderRunDetail(data);
    setRunActionAvailability(data);
  } catch (error) {
    if (!silent && generation === workspaceGeneration && sessionId === workspaceSessionId && selectedRunId === runId) showNotice(`任务详情加载失败：${error.message}`);
  }
}

async function refreshRunning(sessionId, generation = workspaceGeneration) {
  if (!sessionId) return;
  try {
    const response = await fetch(`/api/sessions/${sessionId}/running`);
    const data = await response.json();
    if (!response.ok || generation !== workspaceGeneration || sessionId !== workspaceSessionId) return;
    const running = Boolean(data.running);
    if (stopRunButton) stopRunButton.disabled = !running;
    window.dispatchEvent(new CustomEvent("dftforge:running", {
      detail: { sessionId, running, cancelling: Boolean(data.cancelling), runId: data.run_id || null },
    }));
    if (running) {
      runContext.textContent = data.cancelling ? "正在停止任务…" : `任务运行中 · ${data.run_id || ""}`;
    } else if (data.last_operation?.state === "error") {
      showNotice(`后台任务操作失败：${data.last_operation.error || "未知错误"}`);
      runContext.textContent = "后台任务操作失败";
    } else if (runContext.textContent.startsWith("任务运行中") || runContext.textContent === "正在停止任务…") {
      runContext.textContent = "任务已结束，正在刷新结果…";
    }
  } catch (error) {
    if (generation === workspaceGeneration && sessionId === workspaceSessionId) showNotice(`运行状态暂时不可用：${error.message}`);
  }
}

function beginPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    const id = workspaceSessionId;
    if (!id || pollInFlight) return;
    pollInFlight = true;
    refreshWorkspace(id, { silent: true, skipArtifacts: true }).finally(() => { pollInFlight = false; });
  }, 2000);
}

export async function refreshWorkspace(sessionId = localStorage.getItem("dftforge_session"), options = {}) {
  const generation = ++workspaceGeneration;
  workspaceSessionId = sessionId || null;
  if (!sessionId) {
    renderArtifacts([]);
    renderRuns([]);
    if (stopRunButton) stopRunButton.disabled = true;
    return;
  }
  try {
    const [artifactResponse, runs] = await Promise.all([
      options.skipArtifacts ? Promise.resolve(null) : fetch(`/api/sessions/${sessionId}/artifacts`), fetchRuns(sessionId),
    ]);
    if (generation !== workspaceGeneration || sessionId !== workspaceSessionId) return;
    lastWorkspaceError = "";
    if (artifactResponse?.ok) renderArtifacts((await artifactResponse.json()).artifacts || []);
    renderRuns(runs);
    await refreshRunning(sessionId, generation);
    await refreshSelectedRunDetail(sessionId, generation, !options.silent);
  } catch (error) {
    if (generation === workspaceGeneration && (!options.silent || lastWorkspaceError !== error.message)) {
      lastWorkspaceError = error.message;
      showNotice(`任务中心刷新失败：${error.message}；可点击“重试”`);
    }
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

async function exportRunEvidence() {
  const sessionId = workspaceSessionId || localStorage.getItem("dftforge_session");
  const runId = selectedRunId;
  if (!sessionId || !runId) return;
  const response = await fetch(`/api/sessions/${sessionId}/runs/${encodeURIComponent(runId)}/evidence`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `dft-forge-${sessionId}-${runId}-evidence.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  showNotice("已导出任务运行记录；这不是科研有效性证明，请在分享前检查内容。", "info");
}

async function runAction(action) {
  const sessionId = workspaceSessionId || localStorage.getItem("dftforge_session");
  if (!sessionId || !selectedRunId) return;
  runContext.textContent = action === "retry" ? "正在重试失败节点…" : "正在恢复任务…";
  try {
    const response = await fetch(`/api/sessions/${sessionId}/runs/action`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, run_id: selectedRunId, background: true }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "操作失败");
    runContext.textContent = `任务 ${data.state || "已接受"}`;
    if (response.status === 202 || data.state === "running") {
      if (stopRunButton) stopRunButton.disabled = false;
      window.dispatchEvent(new CustomEvent("dftforge:running", { detail: { sessionId, running: true, runId: data.run_id || selectedRunId } }));
    }
    await refreshWorkspace(sessionId);
  } catch (error) {
    showNotice(`任务操作失败：${error.message}`);
    runContext.textContent = "任务操作失败";
  }
}

async function stopCurrentRun() {
  const sessionId = workspaceSessionId || localStorage.getItem("dftforge_session");
  if (!sessionId) return;
  try {
    const response = await fetch(`/api/sessions/${sessionId}/cancel`, { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "停止失败");
    runContext.textContent = data.cancelled ? "正在停止任务…" : "当前没有运行中的任务";
    await refreshRunning(sessionId, workspaceGeneration);
  } catch (error) {
    showNotice(`停止任务失败：${error.message}`);
  }
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
  if (!attachment) {
    attachedStructure = null;
    attachmentChip.hidden = true;
    return Promise.resolve();
  }
  attachedStructure = attachment;
  attachmentChip.hidden = false;
  attachmentChip.textContent = `${attachment.formula} · ${attachment.natoms} 原子`;
  attachmentChip.title = attachment.name || "上传结构";
  const sessionId = session?.session_id || localStorage.getItem("dftforge_session");
  if (attachment.cif) {
    renderCif(attachment.cif, attachment);
    return Promise.resolve();
  }
  if (!sessionId) return Promise.resolve();
  return fetch(`/api/sessions/${sessionId}/structure`)
    .then(async (response) => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      if ((sessionId === workspaceSessionId || sessionId === localStorage.getItem("dftforge_session")) && data.cif) {
        attachedStructure = { ...attachment, ...data };
        renderCif(data.cif, data);
        const caption = document.getElementById("viewerCaption");
        if (caption) caption.textContent = `${data.formula || attachment.formula} · ${data.natoms || attachment.natoms} 原子 · 上传结构`;
      }
    })
    .catch((error) => showNotice(`结构预览恢复失败：${error.message}`));
}

export function clearWorkspace() {
  if (methodDlg?.open) settleMethodSettings(false);
  workspaceSessionId = null;
  workspaceGeneration += 1;
  attachedStructure = null;
  attachmentChip.hidden = true;
  runContext.textContent = "等待结构与任务";
  clearRunDetail();
  renderArtifacts([]);
  renderRuns([]);
}

export function initWorkspace() {
  modeSelect.value = localStorage.getItem("dftforge_run_mode") || "research";
  accuracySelect.value = localStorage.getItem("dftforge_accuracy") || "balanced";
  methodConverge.checked = localStorage.getItem("dftforge_converge_first") === "true";
  document.getElementById("attachStructure").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) uploadStructure(file).catch((error) => { runContext.textContent = error.message; });
    fileInput.value = "";
  });
  modeSelect.addEventListener("change", () => localStorage.setItem("dftforge_run_mode", modeSelect.value));
  accuracySelect.addEventListener("change", () => { localStorage.setItem("dftforge_accuracy", accuracySelect.value); renderMethodSummary("下一次计算"); });
  methodAccuracy.addEventListener("change", () => { accuracySelect.value = methodAccuracy.value; renderMethodSummary("下一次计算"); });
  methodConverge.addEventListener("change", () => renderMethodSummary("下一次计算"));
  for (const id of METHOD_INPUTS) document.getElementById(id).addEventListener("input", () => renderMethodSummary("下一次计算"));
  document.getElementById("openMethod").addEventListener("click", openMethodSettings);
  document.getElementById("methodConfirm").addEventListener("click", () => settleMethodSettings(true));
  document.getElementById("methodCancel").addEventListener("click", () => settleMethodSettings(false));
  document.getElementById("methodClose").addEventListener("click", () => settleMethodSettings(false));
  methodDlg.addEventListener("cancel", (event) => { event.preventDefault(); settleMethodSettings(false); });
  methodDlg.querySelector("form").addEventListener("submit", (event) => { event.preventDefault(); settleMethodSettings(true); });
  for (const tab of document.querySelectorAll(".panel-tab")) tab.addEventListener("click", () => switchPanel(tab.dataset.panel));
  document.getElementById("refreshArtifacts").addEventListener("click", () => refreshWorkspace());
  document.getElementById("exportEvidence").addEventListener("click", () => exportEvidence().catch((error) => showNotice(`证据导出失败：${error.message}`)));
  exportRunEvidenceButton?.addEventListener("click", () => exportRunEvidence().catch((error) => showNotice(`任务证据导出失败：${error.message}`)));
  document.getElementById("retryRun").addEventListener("click", () => runAction("retry"));
  document.getElementById("resumeRun").addEventListener("click", () => runAction("resume"));
  stopRunButton?.addEventListener("click", stopCurrentRun);
  document.getElementById("retryWorkspace")?.addEventListener("click", () => refreshWorkspace());
  document.getElementById("closeMobilePanel")?.addEventListener("click", () => document.getElementById("colFigures")?.classList.remove("mobile-open"));
  for (const button of document.querySelectorAll("[data-mobile-panel]")) {
    button.addEventListener("click", () => {
      const figures = document.getElementById("colFigures");
      figures?.classList.add("mobile-open");
      switchPanel(button.dataset.mobilePanel);
    });
  }
  beginPolling();
  refreshWorkspace();
}
