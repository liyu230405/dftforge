let viewer = null;

export function initViewer() {
  const el = document.getElementById("viewer3d");
  if (!el || viewer) return;
  viewer = $3Dmol.createViewer(el, { backgroundColor: "#0b1220" });
  viewer.render();
}

export function renderCif(cif) {
  if (!viewer) initViewer();
  viewer.removeAllModels();
  viewer.addModel(cif, "cif");
  viewer.setStyle({}, { sphere: { colorscheme: "Jmol" } });
  viewer.zoomTo();
  viewer.render();
}

export function clear() {
  if (!viewer) return;
  viewer.removeAllModels();
  viewer.render();
}
