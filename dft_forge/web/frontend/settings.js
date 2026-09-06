/* Vibe-coding style connection center: model provider + compute backend. */

const dlg = document.getElementById("settingsDlg");
const btnSettings = document.getElementById("btnSettings");
const elBase = document.getElementById("cfgBaseUrl");
const elModel = document.getElementById("cfgModel");
const elKey = document.getElementById("cfgApiKey");
const elHint = document.getElementById("cfgHint");
const elError = document.getElementById("cfgError");
const testResult = document.getElementById("cfgTestResult");
const providerGrid = document.getElementById("providerGrid");

let selectedVendor = "custom";
let selectedExecutor = "local";
let configState = null;

function showError(el, message) { el.textContent = message || ""; }

function providerSubtitle(provider) {
  if (provider.id === "rules") return "离线 · 零密钥";
  if (provider.id === "ollama") return "本地模型";
  return provider.needs_key ? "OpenAI API 兼容" : "自定义端点";
}

function renderProviders(providers) {
  providerGrid.replaceChildren();
  for (const provider of providers || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `provider-card${provider.id === selectedVendor ? " active" : ""}`;
    const strong = document.createElement("strong");
    strong.textContent = provider.name;
    const sub = document.createElement("span");
    sub.textContent = providerSubtitle(provider);
    button.append(strong, sub);
    button.addEventListener("click", () => selectProvider(provider));
    providerGrid.appendChild(button);
  }
}

function selectProvider(provider) {
  selectedVendor = provider.id;
  renderProviders(configState?.providers || []);
  if (provider.id !== "custom") {
    elBase.value = provider.base_url || "";
    elModel.value = provider.model || "";
  }
  const disabled = provider.id === "rules";
  elBase.disabled = disabled;
  elModel.disabled = disabled;
  elKey.disabled = disabled;
  testResult.hidden = true;
}

function paintState(cfg) {
  configState = cfg;
  selectedVendor = cfg.provider === "dummy" ? "rules" : (cfg.vendor || "custom");
  selectedExecutor = cfg.compute?.executor || "local";
  renderProviders(cfg.providers || []);
  const chosen = (cfg.providers || []).find((provider) => provider.id === selectedVendor);
  selectProvider(chosen || { id: selectedVendor, base_url: cfg.base_url, model: cfg.model });
  elBase.value = cfg.base_url || chosen?.base_url || "";
  elModel.value = cfg.model || chosen?.model || "";
  elKey.value = "";
  // Credentials are write-only in the UI: do not echo prefixes, suffixes, or
  // masked fragments into the DOM. An empty field keeps the stored key.
  elKey.placeholder = cfg.has_api_key ? "已配置 API Key · 留空保持" : "仅保存在本机";
  if (cfg.llm_ready) {
    elHint.textContent = `已接入 ${chosen?.name || selectedVendor} · ${cfg.model || "模型待填写"}`;
    elHint.classList.add("on");
  } else if (cfg.provider === "dummy") {
    elHint.textContent = "当前使用内置规则规划，不调用外部模型。";
    elHint.classList.remove("on");
  } else {
    elHint.textContent = "配置尚未完成。保存前可先测试连接。";
    elHint.classList.remove("on");
  }
  paintCompute(cfg.compute || {});
  const modelName = cfg.provider === "dummy" ? "规则" : (chosen?.name || "LLM");
  const computeName = selectedExecutor === "ssh" ? "HPC" : selectedExecutor === "fake" ? "演示" : "本地";
  btnSettings.textContent = `${modelName} · ${computeName}`;
  btnSettings.classList.toggle("ready", cfg.llm_ready || cfg.provider === "dummy");
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const cfg = await response.json();
  paintState(cfg);
  return cfg;
}

function modelBody() {
  const body = {
    provider: selectedVendor === "rules" ? "dummy" : "openai",
    vendor: selectedVendor,
    base_url: elBase.value.trim(),
    model: elModel.value.trim(),
  };
  if (elKey.value.trim()) body.api_key = elKey.value.trim();
  return body;
}

async function testConnection() {
  showError(elError, "");
  testResult.hidden = false;
  testResult.className = "connection-result";
  if (selectedVendor === "rules") {
    testResult.textContent = "内置规则模式无需联网，可以直接使用。";
    return;
  }
  testResult.textContent = "正在连接模型…";
  const button = document.getElementById("cfgTest");
  button.disabled = true;
  try {
    const response = await fetch("/api/config/test", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(modelBody()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    testResult.textContent = `连接成功 · ${data.model} · ${data.latency_ms} ms`;
  } catch (error) {
    testResult.classList.add("bad");
    testResult.textContent = `连接失败 · ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function saveModel() {
  showError(elError, "");
  const button = document.getElementById("cfgSave");
  button.disabled = true;
  try {
    const response = await fetch("/api/config", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(modelBody()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    await loadConfig();
    testResult.hidden = false;
    testResult.className = "connection-result";
    testResult.textContent = selectedVendor === "rules" ? "已切换到内置规则。" : "模型配置已保存。";
  } catch (error) {
    showError(elError, error.message);
  } finally {
    button.disabled = false;
  }
}

function selectExecutor(value) {
  selectedExecutor = value;
  for (const option of document.querySelectorAll(".compute-option")) option.classList.toggle("active", option.dataset.executor === value);
  document.getElementById("localFields").hidden = value !== "local";
  document.getElementById("sshFields").hidden = value !== "ssh";
}

function paintCompute(compute) {
  selectExecutor(compute.executor || "local");
  document.getElementById("cfgQeBin").value = compute.qe_bin || "";
  document.getElementById("cfgSshHost").value = compute.ssh_host || "";
  document.getElementById("cfgSshKey").value = "";
  document.getElementById("cfgSshKey").placeholder = compute.has_ssh_key ? `已配置 ${compute.ssh_key_name} · 留空保持` : "~/.ssh/id_ed25519";
  document.getElementById("cfgScheduler").value = compute.scheduler || "none";
  document.getElementById("cfgWalltime").value = compute.walltime || 1800;
  document.getElementById("cfgRemoteWorkdir").value = compute.remote_workdir || "/root/workspace";
  document.getElementById("cfgRemoteQeBin").value = compute.remote_qe_bin || "/opt/qe/bin";
}

async function saveCompute() {
  const errorEl = document.getElementById("computeError");
  showError(errorEl, "");
  const body = {
    executor: selectedExecutor,
    qe_bin: document.getElementById("cfgQeBin").value.trim(),
    ssh_host: document.getElementById("cfgSshHost").value.trim(),
    scheduler: document.getElementById("cfgScheduler").value,
    walltime: Number(document.getElementById("cfgWalltime").value || 1800),
    remote_workdir: document.getElementById("cfgRemoteWorkdir").value.trim(),
    remote_qe_bin: document.getElementById("cfgRemoteQeBin").value.trim(),
  };
  const sshKey = document.getElementById("cfgSshKey").value.trim();
  if (sshKey) body.ssh_key = sshKey;
  try {
    const response = await fetch("/api/config/compute", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    await loadConfig();
    errorEl.style.color = "var(--green)";
    errorEl.textContent = "算力配置已保存";
  } catch (error) {
    errorEl.style.color = "";
    showError(errorEl, error.message);
  }
}

function switchTab(tab) {
  for (const button of document.querySelectorAll(".cfg-tab")) button.classList.toggle("active", button.dataset.tab === tab);
  document.getElementById("cfgPaneModel").classList.toggle("active", tab === "model");
  document.getElementById("cfgPaneCompute").classList.toggle("active", tab === "compute");
}

export function openConnectionCenter(tab = "model") {
  switchTab(tab);
  showError(elError, "");
  loadConfig().then(() => dlg.showModal()).catch((error) => showError(elError, error.message));
}

export function initSettings() {
  btnSettings.addEventListener("click", () => openConnectionCenter("model"));
  document.getElementById("cfgClose").addEventListener("click", () => dlg.close());
  document.getElementById("cfgSave").addEventListener("click", saveModel);
  document.getElementById("cfgTest").addEventListener("click", testConnection);
  document.getElementById("computeSave").addEventListener("click", saveCompute);
  for (const button of document.querySelectorAll(".cfg-tab")) button.addEventListener("click", () => switchTab(button.dataset.tab));
  for (const option of document.querySelectorAll(".compute-option")) option.addEventListener("click", () => selectExecutor(option.dataset.executor));
  dlg.addEventListener("click", (event) => { if (event.target === dlg) dlg.close(); });
  loadConfig().catch(() => { btnSettings.textContent = "连接中心"; });

  const left = document.getElementById("colContents");
  const right = document.getElementById("colFigures");
  document.getElementById("foldLeft").addEventListener("click", () => { left.classList.toggle("folded"); setTimeout(resizeViewer, 200); });
  document.getElementById("foldRight").addEventListener("click", () => { right.classList.toggle("folded"); setTimeout(resizeViewer, 200); });
  const query = new URLSearchParams(location.search);
  for (const side of (query.get("fold") || "").split(",")) {
    if (side === "left") left.classList.add("folded");
    if (side === "right") right.classList.add("folded");
  }
  if (query.get("panel") === "settings") openConnectionCenter("model");
}

function resizeViewer() { window.dispatchEvent(new Event("resize")); }
