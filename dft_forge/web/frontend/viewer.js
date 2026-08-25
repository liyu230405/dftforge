let viewer = null;
let currentCif = null;

const CPK = {
  H: "#E8E8E8", B: "#FFB5B5", C: "#909090", N: "#3050F8", O: "#FF0D0D",
  F: "#90E050", Na: "#AB5CF2", Mg: "#8AFF00", Al: "#BFA6A6", Si: "#F0C8A0",
  P: "#FF8000", S: "#FFFF30", Cl: "#1FF01F", K: "#8F40D4", Ca: "#3DFF00",
  Ti: "#BFC2C7", Cr: "#8A99C7", Mn: "#9C7AC7", Fe: "#E06633", Co: "#F090A0",
  Ni: "#50D050", Cu: "#C88033", Zn: "#7D80B0", Ga: "#C28F8F", Ge: "#668F8F",
  As: "#BD80E3", Se: "#FFA100", Br: "#A62929", Rb: "#7070B0", Sr: "#00FF00",
  Zr: "#94E0E0", Nb: "#73C2C9", Mo: "#54B5B5", Cd: "#FFD98F", In: "#A67573",
  Sn: "#668F8F", Sb: "#9E63B5", Te: "#D47A00", I: "#940094", Cs: "#57178F",
  Ba: "#00C900", La: "#70D4FF", W: "#219E94", Pt: "#D0D0E0", Au: "#FFD123",
  Hg: "#B8B8D0", Pb: "#575961", Bi: "#9E4FB5",
};

export function initViewer() {
  const el = document.getElementById("viewer3d");
  if (!el || viewer) return;
  if (typeof $3Dmol === "undefined") return;
  viewer = $3Dmol.createViewer(el, { backgroundColor: "#FFFDF9" });
  viewer.render();
}

export function renderCif(cif, meta = {}) {
  if (!viewer) initViewer();
  if (!viewer) return;
  currentCif = cif;
  const el = document.getElementById("viewer3d");
  el.dataset.replica = "off";

  viewer.removeAllModels();
  viewer.removeAllShapes();
  viewer.removeAllLabels();

  const model = viewer.addModel(cif, "cif");
  model.setStyle({}, { sphere: { scale: 0.28, colorscheme: "Jmol" }, stick: { radius: 0.12, colorscheme: "Jmol" } });

  const showCell = document.getElementById("toggleCell")?.checked ?? true;
  if (showCell) viewer.addUnitCell(model, { box: { color: "#5C544A" }, astyle: { color: "#22527A" }, bstyle: { color: "#22527A" }, cstyle: { color: "#22527A" } });

  // adsorption-site markers from structure.build2d
  const sites = meta.sites || {};
  for (const [name, xyz] of Object.entries(sites)) {
    viewer.addLabel(name, {
      position: { x: xyz[0], y: xyz[1], z: (xyz[2] || 0) + 1.2 },
      backgroundColor: "#22527A", fontColor: "#FFFDF9", fontSize: 11,
      backgroundOpacity: 0.85, padding: 2, borderRadius: 3,
    });
    viewer.addSphere({
      center: { x: xyz[0], y: xyz[1], z: (xyz[2] || 0) + 1.2 },
      radius: 0.18, color: "#22527A", alpha: 0.75,
    });
  }

  viewer.zoomTo();
  viewer.render();
  updateLegend(model);
}

export function toggleReplica(on) {
  if (!viewer || !currentCif) return;
  const el = document.getElementById("viewer3d");
  const models = viewer.getAllModels();
  const m0 = models[0];
  if (!m0) return;
  if (on) {
    viewer.replicateUnitCell(2, 2, 1, m0);
    el.dataset.replica = "on";
  } else {
    viewer.removeAllModels();
    const model = viewer.addModel(currentCif, "cif");
    model.setStyle({}, { sphere: { scale: 0.28, colorscheme: "Jmol" }, stick: { radius: 0.12, colorscheme: "Jmol" } });
    const showCell = document.getElementById("toggleCell")?.checked ?? true;
    if (showCell) viewer.addUnitCell(model, { box: { color: "#5C544A" } });
    el.dataset.replica = "off";
  }
  viewer.render();
}

export function toggleCellFrame(on) {
  if (!viewer || !currentCif) return;
  const models = viewer.getAllModels();
  if (!models.length) return;
  viewer.removeAllShapes();
  if (on) viewer.addUnitCell(models[0], { box: { color: "#5C544A" } });
  viewer.render();
}

function updateLegend(model) {
  const legend = document.getElementById("atomLegend");
  if (!legend || !model) return;
  const counts = {};
  for (const a of model.selectedAtoms({})) {
    const s = a.elem || "X";
    counts[s] = (counts[s] || 0) + 1;
  }
  legend.innerHTML = Object.entries(counts)
    .sort()
    .map(([el, n]) =>
      `<span class="key"><span class="swatch" style="background:${CPK[el] || "#999"}"></span>${el}×${n}</span>`
    )
    .join("");
}

export function clear() {
  if (!viewer) return;
  viewer.removeAllModels();
  viewer.removeAllShapes();
  viewer.removeAllLabels();
  viewer.render();
  currentCif = null;
  const legend = document.getElementById("atomLegend");
  if (legend) legend.innerHTML = "";
}
