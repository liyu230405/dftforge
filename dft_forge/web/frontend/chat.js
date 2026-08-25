import { initViewer, renderCif, clear as clearViewer } from "./viewer.js";

const API = "/chat";
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("msg");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function setStatus(text, type = "") {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.className = `status ${type}`.trim();
}

function append(role, text, extra = "") {
  const row = document.createElement("div");
  row.className = `row ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = esc(text) + (extra ? `<div class="meta">${esc(extra)}</div>` : "");
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return row;
}

function renderToolCalls(row, commands, results) {
  if (!commands || !commands.length) return;
  const tool = document.createElement("div");
  tool.className = "tool";
  const lines = commands.map((c, i) => {
    const r = results && results[i] ? results[i] : {};
    const rc = r.returncode ?? "?";
    const err = r.error ? ` | error=${esc(r.error)}` : "";
    return `#${i + 1} ${c.command} (rc=${rc}${err})`;
  }).join("\n");
  tool.innerHTML = `<details><summary>tool calls: ${commands.length}</summary><pre>${esc(lines)}</pre></details>`;
  row.querySelector(".bubble").appendChild(tool);
}

function tryRenderStructure(data) {
  const payload = data && typeof data === "object" ? data : {};
  const candidates = [payload, payload.json, payload.data, payload.result, payload.output].filter(Boolean);
  for (const obj of candidates) {
    const cif = obj.cif || obj.cif_content || obj.structure || obj.atoms || obj.content;
    if (cif && typeof cif === "string") {
      renderCif(cif);
      return;
    }
    if (obj && typeof obj.to_dict === "function") {
      renderCif(JSON.stringify(obj.to_dict()));
      return;
    }
  }
  if (payload.command === "structure.import" && payload.ok) {
    clearViewer();
  }
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendBtn.disabled = true;
  setStatus("thinking...");
  append("user", text);
  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    const row = append("agent", data.reply || "(empty reply)");
    renderToolCalls(row, data.commands, data.results);
    tryRenderStructure(data);
    setStatus("ready", "success");
  } catch (e) {
    append("agent", "请求失败：" + e.message);
    setStatus("error", "error");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

export function initChat() {
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  setTimeout(() => initViewer(), 0);
}
