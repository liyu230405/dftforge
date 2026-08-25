/* Settings dialog (LLM key entry) + side-column folding. */

const dlg = document.getElementById("settingsDlg");
const btnSettings = document.getElementById("btnSettings");
const btnSave = document.getElementById("cfgSave");
const btnClose = document.getElementById("cfgClose");
const elBase = document.getElementById("cfgBaseUrl");
const elModel = document.getElementById("cfgModel");
const elKey = document.getElementById("cfgApiKey");
const elHint = document.getElementById("cfgHint");
const elError = document.getElementById("cfgError");

function paintBadge(cfg) {
  btnSettings.classList.toggle("ready", Boolean(cfg.llm_ready));
  btnSettings.title = cfg.llm_ready
    ? `LLM 已连接：${cfg.model || "default model"} @ ${cfg.base_url || "default endpoint"}`
    : "配置 LLM API（当前使用内置规则规划）";
  btnSettings.textContent = cfg.llm_ready ? "⚙ LLM 已接" : "⚙ LLM 设置";
}

function paintHint(cfg) {
  if (cfg.llm_ready) {
    elHint.textContent = `已连接 ${cfg.api_key_hint} · ${cfg.model || "默认模型"}。留空保存则保持不变。`;
    elHint.classList.add("on");
  } else {
    elHint.textContent = "未配置 — 当前使用内置规则规划（可完成计算，但理解为关键词匹配）。";
    elHint.classList.remove("on");
  }
}

async function loadConfig() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    elBase.value = cfg.base_url || "";
    elModel.value = cfg.model || "";
    elKey.value = "";
    elKey.placeholder = cfg.has_api_key ? `已保存 ${cfg.api_key_hint} — 留空则不修改` : "sk-…（仅存本地 .env）";
    paintBadge(cfg);
    paintHint(cfg);
    return cfg;
  } catch {
    return null;
  }
}

async function saveConfig() {
  elError.textContent = "";
  btnSave.disabled = true;
  const body = {
    provider: "openai",
    base_url: elBase.value.trim(),
    model: elModel.value.trim(),
  };
  if (elKey.value.trim()) body.api_key = elKey.value.trim();
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const cfg = await res.json();
    if (!cfg.llm_ready) {
      elError.textContent = "还差一步：请填写 API Key";
      paintHint(cfg);
      paintBadge(cfg);
      return;
    }
    paintBadge(cfg);
    paintHint(cfg);
    dlg.close();
  } catch (e) {
    elError.textContent = "保存失败：" + e.message;
  } finally {
    btnSave.disabled = false;
  }
}

export function initSettings() {
  btnSettings.addEventListener("click", () => {
    elError.textContent = "";
    loadConfig().then(() => dlg.showModal());
  });
  btnSave.addEventListener("click", saveConfig);
  btnClose.addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) dlg.close();
  });
  loadConfig();

  const left = document.getElementById("colContents");
  const right = document.getElementById("colFigures");
  document.getElementById("foldLeft").addEventListener("click", () => {
    left.classList.toggle("folded");
    setTimeout(resizeViewer, 200);
  });
  document.getElementById("foldRight").addEventListener("click", () => {
    right.classList.toggle("folded");
    setTimeout(resizeViewer, 200);
  });

  // deep links for screenshots/automation: ?panel=settings&fold=left,right
  const q = new URLSearchParams(location.search);
  for (const side of (q.get("fold") || "").split(",")) {
    if (side === "left") left.classList.add("folded");
    if (side === "right") right.classList.add("folded");
  }
  if (q.get("panel") === "settings") loadConfig().then(() => dlg.showModal());
}

function resizeViewer() {
  window.dispatchEvent(new Event("resize"));
}
