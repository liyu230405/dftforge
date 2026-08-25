const NS = "http://www.w3.org/2000/svg";
const INK = "#1F1B16", INK3 = "#8A8175", HAIR = "#DDD6C8", BLUE = "#22527A", RED = "#A03D2E";

export function renderCharts(container, chart) {
  container.innerHTML = "";
  if (!chart) {
    container.innerHTML = '<div class="fig-empty">完成一次计算后，能带、态密度与关键数值出现在这里</div>';
    return;
  }
  if (chart.metrics) renderMetrics(container, chart.metrics);
  if (chart.bands) renderBands(container, chart.bands);
  if (chart.dos) renderDos(container, chart.dos);
}

function renderMetrics(container, metrics) {
  const NAMES = {
    energy_ry: ["总能量", "Ry"],
    a_angstrom: ["晶格常数 a", "Å"],
    pressure_kbar: ["残余压力", "kbar"],
    max_force_ev_ang: ["最大原子力", "eV/Å"],
  };
  const rows = Object.entries(metrics)
    .filter(([k]) => NAMES[k])
    .map(([k, v]) => `<tr><td>${NAMES[k][0]}</td><td>${fmt(v)} ${NAMES[k][1]}</td></tr>`)
    .join("");
  if (!rows) return;
  container.insertAdjacentHTML("beforeend", `<table class="metric-table">${rows}</table>`);
}

function fmt(v) {
  return typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(1) : v.toPrecision(6)) : v;
}

function renderBands(container, bands) {
  const ev = bands.eigenvalues_ev || [];
  if (!ev.length || !ev[0]?.length) return;
  const nK = ev.length, nB = ev[0].length;
  const fermi = bands.fermi_ev ?? 0;
  const gap = bands.band_gap_ev ?? 0;

  let all = ev.flat().filter(Number.isFinite);
  let lo = Math.min(...all), hi = Math.max(...all);
  // focus window: fermi ±6 eV or data range, whichever tighter
  lo = Math.max(lo, fermi - 6);
  hi = Math.min(hi, fermi + 6);
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;

  const W = 340, H = 260, ML = 36, MR = 8, MT = 10, MB = 22;
  const axis = bands.k_axis; // cumulative-k plotting axis (unequal spacing preserved)
  const axisMax = axis ? axis.filter(Number.isFinite).at(-1) || 1 : 1;
  const x = axis
    ? (i) => ML + (axis[i] / axisMax) * (W - ML - MR)
    : (i) => ML + (i / (nK - 1)) * (W - ML - MR);
  const y = (e) => MT + ((hi - e) / (hi - lo)) * (H - MT - MB);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "能带结构图" });

  // high-symmetry ticks & verticals on the linear-k axis
  const labels = bands.k_labels || [];
  const ticksX = bands.k_ticks || [];
  const shown = new Set();
  labels.forEach((lb, i) => {
    if (!lb || !Number.isFinite(ticksX[i]) || shown.has(lb)) return;
    shown.add(lb);
    const xt = ML + (ticksX[i] / axisMax) * (W - ML - MR);
    if (i > 0 && i < labels.length - 1) svg.append(el("line", { x1: xt, y1: MT, x2: xt, y2: H - MB, stroke: HAIR, "stroke-width": 1 }));
    const t = el("text", { x: xt, y: H - MB + 14, "text-anchor": "middle", "font-size": 11, fill: INK3, "font-family": "Spline Sans Mono, monospace" });
    t.textContent = lb;
    svg.append(t);
  });

  // fermi line: midgap for semiconductors, fermi for metals
  const ef = bands.is_metal ? fermi : fermi + (gap || 0) / 2;
  svg.append(el("line", { x1: ML, y1: y(ef), x2: W - MR, y2: y(ef), stroke: RED, "stroke-width": 1, "stroke-dasharray": "4 3" }));
  const efLabel = el("text", { x: W - MR - 2, y: y(ef) - 3, "text-anchor": "end", "font-size": 9.5, fill: RED, "font-family": "Spline Sans Mono, monospace" });
  efLabel.textContent = bands.is_metal ? `E_F=${fermi.toFixed(2)} eV` : `E_F(中带隙)`;
  svg.append(efLabel);

  // band curves — split at NaN discontinuities in the k axis (path jumps)
  const splits = new Set();
  if (axis) axis.forEach((v, i) => { if (!Number.isFinite(v)) splits.add(i); });
  for (let b = 0; b < nB; b++) {
    let pts = [];
    const flush = () => {
      if (pts.length > 1) svg.append(el("polyline", { points: pts.join(" "), fill: "none", stroke: BLUE, "stroke-width": 1.4, "stroke-linejoin": "round" }));
      pts = [];
    };
    for (let k = 0; k < nK; k++) {
      if (splits.has(k)) { flush(); continue; }
      const e = ev[k][b];
      if (Number.isFinite(e) && Number.isFinite(x(k))) pts.push(`${x(k).toFixed(1)},${y(e).toFixed(1)}`);
    }
    flush();
  }

  // axes
  svg.append(el("line", { x1: ML, y1: H - MB, x2: W - MR, y2: H - MB, stroke: INK, "stroke-width": 1 }));
  svg.append(el("line", { x1: ML, y1: MT, x2: ML, y2: H - MB, stroke: INK, "stroke-width": 1 }));
  for (const e of ticks(lo, hi)) {
    const t = el("text", { x: ML - 4, y: y(e) + 3, "text-anchor": "end", "font-size": 9, fill: INK3, "font-family": "Spline Sans Mono, monospace" });
    t.textContent = e.toFixed(0);
    svg.append(t);
  }

  const head = document.createElement("div");
  head.className = "chart-block";
  head.innerHTML = `<h3>能带结构</h3><div class="chart-note">${
    bands.is_metal ? "金属（无带隙）" : `带隙 ${gap.toFixed(3)} eV · ${nB} 带 × ${nK} k点`
  }</div>`;
  head.appendChild(svg);
  container.appendChild(head);
}

function renderDos(container, dos) {
  const energies = dos.energies_ev || [], ds = dos.dos || [];
  if (!energies.length || energies.length !== ds.length) return;
  const fermi = dos.fermi_ev ?? 0;

  let lo = Math.max(Math.min(...energies), fermi - 8);
  let hi = Math.min(Math.max(...energies), fermi + 8);
  const maxD = Math.max(...ds) * 1.08 || 1;

  const W = 340, H = 170, ML = 8, MR = 36, MT = 8, MB = 22;
  const x = (e) => ML + ((e - lo) / (hi - lo)) * (W - ML - MR);
  const y = (d) => MT + (1 - d / maxD) * (H - MT - MB);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "态密度图" });
  svg.append(el("line", { x1: x(fermi), y1: MT, x2: x(fermi), y2: H - MB, stroke: RED, "stroke-width": 1, "stroke-dasharray": "4 3" }));

  const pts = [];
  for (let i = 0; i < energies.length; i++) {
    if (energies[i] < lo || energies[i] > hi) continue;
    pts.push(`${x(energies[i]).toFixed(1)},${y(ds[i]).toFixed(2)}`);
  }
  if (pts.length > 1) {
    svg.append(el("polyline", { points: `ML,${H - MB} ${pts.join(" ")} ${x(pts.length ? energies[energies.length - 1] : lo)},${H - MB}`.replace(/^ML,/, `${pts[0].split(",")[0]},`), fill: "rgba(34,82,122,0.12)", stroke: "none" }));
    svg.append(el("polyline", { points: pts.join(" "), fill: "none", stroke: BLUE, "stroke-width": 1.4 }));
  }

  svg.append(el("line", { x1: ML, y1: H - MB, x2: W - MR, y2: H - MB, stroke: INK, "stroke-width": 1 }));
  for (const e of ticks(lo, hi)) {
    const t = el("text", { x: x(e), y: H - MB + 13, "text-anchor": "middle", "font-size": 9, fill: INK3, "font-family": "Spline Sans Mono, monospace" });
    t.textContent = e.toFixed(0);
    svg.append(t);
  }
  const fl = el("text", { x: x(fermi), y: MT + 9, "text-anchor": "end", "font-size": 9.5, fill: RED, "font-family": "Spline Sans Mono, monospace" });
  fl.textContent = "E_F";
  svg.append(fl);

  const head = document.createElement("div");
  head.className = "chart-block";
  head.innerHTML = `<h3>态密度 (DOS)</h3><div class="chart-note">${energies.length} 能量点 · E − E_F (eV)</div>`;
  head.appendChild(svg);
  container.appendChild(head);
}

function el(name, attrs = {}) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function ticks(lo, hi) {
  const step = niceStep((hi - lo) / 4);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(+t.toFixed(4));
  return out;
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const r = raw / p;
  return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p;
}
