const NS = "http://www.w3.org/2000/svg";
const INK = "#1F1B16", INK3 = "#8A8175", HAIR = "#DDD6C8", BLUE = "#22527A", RED = "#A03D2E";

export function renderCharts(container, chart, provenance = chart?.provenance) {
  container.innerHTML = "";
  if (!chart) {
    container.innerHTML = '<div class="fig-empty">完成一次计算后，能带、态密度与关键数值出现在这里</div>';
    return;
  }
  renderProvenance(container, provenance || {});
  if (chart.metrics) renderMetrics(container, chart.metrics);
  if (chart.convergence) renderConvergence(container, chart.convergence);
  if (chart.chemical_potential) renderChemicalPotential(container, chart.chemical_potential);
  if (chart.bands) renderBands(container, chart.bands);
  if (chart.dos) renderDos(container, chart.dos);
  if (chart.pdos) renderPdos(container, chart.pdos);
}

function renderProvenance(container, provenance) {
  const bar = document.createElement("div");
  bar.className = "chart-provenance";
  const simulated = provenance.contains_simulated === true;
  let label = "来源未记录";
  if (provenance.execution_class === "unknown" && simulated) {
    label = "来源不完整 · 含模拟尝试 · 非科研结论";
  } else if (provenance.execution_class === "simulated" || provenance.execution_class === "mixed" || simulated) {
    label = "模拟数据 · 非科研结论";
  } else if (provenance.execution_class === "compute") {
    const backend = provenance.backend === "ssh" ? "远程" : provenance.backend === "local" ? "本地" : "计算";
    label = `${backend}计算输出 · 未经独立科研验证`;
  } else if (provenance.source_kind === "parsed") {
    label = "解析已有文件 · 原始执行来源未记录";
  } else if (provenance.source_kind === "user_input") {
    label = "基于用户输入的确定性推导";
  }
  bar.textContent = label;
  container.appendChild(bar);
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

function renderConvergence(container, curve) {
  const points = (curve.points || [])
    .filter(p => Number.isFinite(p.cost_proxy) && Number.isFinite(p.delta_mev_per_atom))
    .sort((a, b) => a.cost_proxy - b.cost_proxy);
  if (!points.length) return;
  const W = 340, H = 220, ML = 44, MR = 10, MT = 16, MB = 34;
  const minX = Math.min(...points.map(p => p.cost_proxy));
  const maxX = Math.max(...points.map(p => p.cost_proxy));
  const maxY = Math.max(curve.threshold_mev_per_atom || 0, ...points.map(p => p.delta_mev_per_atom), 1) * 1.12;
  const x = v => ML + ((v - minX) / Math.max(maxX - minX, 1)) * (W - ML - MR);
  const y = v => MT + (1 - v / maxY) * (H - MT - MB);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "参数收敛误差曲线" });
  svg.append(el("line", { x1: ML, y1: H - MB, x2: W - MR, y2: H - MB, stroke: INK, "stroke-width": 1 }));
  svg.append(el("line", { x1: ML, y1: MT, x2: ML, y2: H - MB, stroke: INK, "stroke-width": 1 }));
  const threshold = Number(curve.threshold_mev_per_atom);
  if (Number.isFinite(threshold)) {
    svg.append(el("line", { x1: ML, y1: y(threshold), x2: W - MR, y2: y(threshold), stroke: RED, "stroke-width": 1, "stroke-dasharray": "4 3" }));
  }
  svg.append(el("polyline", { points: points.map(p => `${x(p.cost_proxy).toFixed(1)},${y(p.delta_mev_per_atom).toFixed(1)}`).join(" "), fill: "none", stroke: BLUE, "stroke-width": 1.5 }));
  for (const p of points) {
    const circle = el("circle", { cx: x(p.cost_proxy), cy: y(p.delta_mev_per_atom), r: 3, fill: p.delta_mev_per_atom <= threshold ? BLUE : RED });
    const title = el("title", {});
    title.textContent = `ecut=${p.ecutwfc} Ry, k=${(p.kpoints || []).slice(0, 3).join("×")}, 误差=${p.delta_mev_per_atom.toPrecision(4)} meV/atom`;
    circle.append(title);
    svg.append(circle);
  }
  const head = document.createElement("div");
  head.className = "chart-block";
  head.innerHTML = `<h3>参数收敛曲线</h3><div class="chart-note">成本代理（横轴） vs 参考点误差（纵轴，meV/atom） · 红虚线阈值 ${Number.isFinite(threshold) ? threshold : "—"}</div>`;
  head.appendChild(svg);
  container.appendChild(head);
}

function renderChemicalPotential(container, domain) {
  const elements = domain.elements || [];
  const svg = el("svg", { viewBox: "0 0 340 190", role: "img", "aria-label": "化学势稳定域相图" });
  const head = document.createElement("div");
  head.className = "chart-block";
  const status = domain.status === "complete" ? "竞争相约束已提供" : "竞争相集合不完整：仅作部分稳定域";
  if (domain.dimension === 1) {
    const elName = elements[0] || "μ";
    const range = (domain.interval || {})[elName] || [0, 0];
    const lo = Number(range[0]), hi = Number(range[1]);
    const x = v => 36 + ((v - lo) / Math.max(hi - lo, 1e-12)) * 268;
    svg.append(el("line", { x1: 36, y1: 92, x2: 304, y2: 92, stroke: HAIR, "stroke-width": 8, "stroke-linecap": "round" }));
    svg.append(el("line", { x1: x(lo), y1: 92, x2: x(hi), y2: 92, stroke: BLUE, "stroke-width": 8, "stroke-linecap": "round" }));
    for (const [value, label] of [[lo, "贫元素端"], [hi, "富元素端"]]) {
      const xt = x(value);
      svg.append(el("circle", { cx: xt, cy: 92, r: 4, fill: BLUE }));
      const t = el("text", { x: xt, y: 72, "text-anchor": "middle", "font-size": 10, fill: INK3, "font-family": "Spline Sans Mono, monospace" });
      t.textContent = `${label} ${value.toPrecision(4)}`;
      svg.append(t);
    }
    const axis = el("text", { x: 170, y: 125, "text-anchor": "middle", "font-size": 12, fill: INK });
    axis.textContent = `Δμ${elName} (eV)`;
    svg.append(axis);
    head.innerHTML = `<h3>化学势稳定区间</h3><div class="chart-note">${status}</div>`;
  } else {
    const points = (domain.vertices || []).map(v => [Number(v[`delta_mu_${elements[0]}_ev`]), Number(v[`delta_mu_${elements[1]}_ev`])]).filter(p => p.every(Number.isFinite));
    if (!points.length) return;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const xlo = Math.min(...xs), xhi = Math.max(...xs), ylo = Math.min(...ys), yhi = Math.max(...ys);
    const x = v => 42 + ((v - xlo) / Math.max(xhi - xlo, 1e-12)) * 250;
    const y = v => 154 - ((v - ylo) / Math.max(yhi - ylo, 1e-12)) * 118;
    svg.append(el("polygon", { points: points.map(p => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" "), fill: "#22527A22", stroke: BLUE, "stroke-width": 1.5 }));
    const xlabel = el("text", { x: 170, y: 184, "text-anchor": "middle", "font-size": 11, fill: INK });
    xlabel.textContent = `Δμ${elements[0] || "1"} (eV)`;
    svg.append(xlabel);
    const ylabel = el("text", { x: 12, y: 95, "text-anchor": "middle", "font-size": 11, fill: INK, transform: "rotate(-90 12 95)" });
    ylabel.textContent = `Δμ${elements[1] || "2"} (eV)`;
    svg.append(ylabel);
    head.innerHTML = `<h3>化学势可行域</h3><div class="chart-note">${status} · ${points.length} 个顶点</div>`;
  }
  head.appendChild(svg);
  container.appendChild(head);
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

function renderPdos(container, pdos) {
  const energies = pdos.energies_ev || [];
  const elemDos = pdos.element_dos || {};
  const elems = Object.keys(elemDos);
  if (!energies.length || !elems.length) return;
  const fermi = pdos.fermi_ev ?? 0;

  let lo = Math.max(Math.min(...energies), fermi - 8);
  let hi = Math.min(Math.max(...energies), fermi + 8);
  let maxD = 0;
  for (const el_ of elems) {
    const arr = elemDos[el_] || [];
    for (let i = 0; i < energies.length && i < arr.length; i++) {
      if (energies[i] >= lo && energies[i] <= hi) maxD = Math.max(maxD, arr[i]);
    }
  }
  maxD = maxD * 1.08 || 1;

  const W = 340, H = 170, ML = 8, MR = 36, MT = 8, MB = 22;
  const x = (e) => ML + ((e - lo) / (hi - lo)) * (W - ML - MR);
  const y = (d) => MT + (1 - d / maxD) * (H - MT - MB);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "投影态密度图" });
  svg.append(el("line", { x1: x(fermi), y1: MT, x2: x(fermi), y2: H - MB, stroke: RED, "stroke-width": 1, "stroke-dasharray": "4 3" }));

  const SERIES = ["#22527A", "#A03D2E", "#5B7553", "#7A5C8E", "#B07D2B", "#3E8E9E"];
  const legend = [];
  elems.forEach((name, idx) => {
    const arr = elemDos[name] || [];
    const color = SERIES[idx % SERIES.length];
    legend.push({ name, color });
    const pts = [];
    for (let i = 0; i < energies.length && i < arr.length; i++) {
      if (energies[i] < lo || energies[i] > hi) continue;
      pts.push(`${x(energies[i]).toFixed(1)},${y(arr[i]).toFixed(2)}`);
    }
    if (pts.length > 1) {
      svg.append(el("polyline", { points: pts.join(" "), fill: "none", stroke: color, "stroke-width": 1.3 }));
    }
  });

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
  const legendHtml = legend
    .map((g) => `<span class="pdos-legend-item"><span style="display:inline-block;width:14px;height:3px;background:${g.color};vertical-align:middle;margin-right:4px"></span>${g.name}</span>`)
    .join("");
  head.innerHTML = `<h3>投影态密度 (PDOS)</h3><div class="chart-note">${energies.length} 能量点 · 分元素贡献 · E (eV)，E_F 虚线</div><div class="pdos-legend">${legendHtml}</div>`;
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
