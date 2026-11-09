#!/usr/bin/env python3
"""Run all five ABINIT templates and export a portable evidence bundle.

No LLM, .env, user sessions or remote compute configuration is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PSEUDO_URL = "https://raw.githubusercontent.com/abinit/abipy/master/abipy/data/pseudos/14si.pspnc"
PSEUDO_SHA256 = "3fb98731545d0734058bc5fec3c2ba80447a14e83095e1b2782b62b016d5400f"
CASES = ["a0_abinit_scf", "a1_abinit_relax", "a2_abinit_bands", "a2_abinit_dos", "a3_abinit_phonon"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abinit-bin", default=os.environ.get("DFT_FORGE_ABINIT_BIN") or shutil.which("abinit"))
    parser.add_argument("--pp-dir", type=Path, default=os.environ.get("DFT_FORGE_ABINIT_PP_DIR"))
    parser.add_argument("--fetch-si-pseudo", action="store_true", help="download checksum-pinned public LDA tutorial pseudo")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--ecut", type=float, default=15, help="cutoff in Hartree; no convergence claim")
    parser.add_argument("--ngkpt", type=int, default=4)
    parser.add_argument("--ngqpt", type=int, default=2)
    args = parser.parse_args()
    if not args.abinit_bin:
        parser.error("provide --abinit-bin or put abinit on PATH")
    binary = Path(shutil.which(args.abinit_bin) or args.abinit_bin).expanduser().resolve()
    for name in [binary.name, "anaddb", "mrgddb"]:
        path = binary if name == binary.name else binary.parent / name
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"missing executable: {path}")
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error("output directory already exists; choose a new directory")
    if not args.fetch_si_pseudo and not args.pp_dir:
        parser.error("provide --pp-dir or use --fetch-si-pseudo")
    out.mkdir(parents=True)
    if args.fetch_si_pseudo:
        pp_dir = out / "pseudos"
        pp_dir.mkdir()
        payload = urllib.request.urlopen(PSEUDO_URL, timeout=30).read()
        if hashlib.sha256(payload).hexdigest() != PSEUDO_SHA256:
            raise ValueError("tutorial pseudo checksum mismatch")
        (pp_dir / "Si.pspnc").write_bytes(payload)
        pseudo_note = {"source": PSEUDO_URL, "sha256": PSEUDO_SHA256,
                       "method": "Troullier-Martins LDA tutorial pseudo (pspxc=1)",
                       "purpose": "software integration demonstration, not a converged material prediction"}
    else:
        pp_dir = args.pp_dir.expanduser().resolve()
        pseudo_note = {"source": "user-supplied", "directory": str(pp_dir)}
    os.environ.update({"DFT_FORGE_EXECUTOR": "local", "DFT_FORGE_ABINIT_BIN": str(binary),
                       "DFT_FORGE_ABINIT_PP_DIR": str(pp_dir), "DFT_FORGE_LLM_PROVIDER": "dummy"})
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from dft_forge.tools.graph_tools import _graph_run
    results = []
    for case in CASES:
        allowed = json.loads((REPO / "templates" / f"{case}.json").read_text())["inputs"]
        candidates = {"material": "Si", "ecut": args.ecut, "ngkpt": [args.ngkpt] * 3,
                      "dos_ngkpt": [max(6, args.ngkpt)] * 3, "ngqpt": [args.ngqpt] * 3,
                      "nband": 8, "ndivsm": 6, "_parameters_locked": True}
        inputs = {key: value for key, value in candidates.items() if key in allowed}
        print(f"运行 {case}", flush=True)
        started = time.monotonic()
        result = _graph_run({"template_id": case, "workdir": str(out / "runs"), "inputs": inputs})
        result.update({"case": case, "inputs": inputs, "elapsed_seconds": round(time.monotonic() - started, 3)})
        (out / f"{case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"  {result.get('state', 'error')} · {result['elapsed_seconds']} s", flush=True)
        results.append(result)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    summary = {"abinit_version": version, "pseudo": pseudo_note,
               "ecut_ha": args.ecut, "ngkpt": [args.ngkpt] * 3, "ngqpt": [args.ngqpt] * 3,
               "scientific_scope": "Si integration test; parameters and supercell size have not been converged; no LO/TO correction",
               "binary_sha256": {name: sha256(binary if name == "abinit" else binary.parent / name)
                                  for name in ["abinit", "anaddb", "mrgddb"]},
               "cases": [{k: r.get(k) for k in ["case", "state", "run_id", "inputs", "elapsed_seconds", "outputs"]} for r in results]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    export_evidence(out, results, summary)
    print(out / "report.md", flush=True)
    return 0 if all(r.get("state") == "succeeded" for r in results) else 1


def export_evidence(out, results, summary):
    """Package a completed run; also usable without rerunning the calculations."""
    out = Path(out)
    evidence = out / "evidence"
    for r in results:
        if not r.get("run_id"):
            continue
        for src in (out / "runs" / r["run_id"]).rglob("*"):
            if not src.is_file() or src.is_symlink():
                continue
            name = src.name
            keep = src.suffix in {".abi", ".abo", ".in", ".out", ".log", ".err", ".cif", ".files"}
            keep |= any(name.endswith(x) for x in ["_DDB", "_GSR.nc", "_EIG.nc", "_PHBST.nc", "_PHDOS.nc", "_EIG", "_DOS", "_DOS_TOTAL"])
            keep |= "_DOS_AT" in name
            if keep:
                target = evidence / r["case"] / src.relative_to(out / "runs" / r["run_id"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    for name in CASES:
        target = evidence / "templates" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "templates" / f"{name}.json", target)
    source_files = list((REPO / "dft_forge" / "engines" / "abinit").glob("*.py"))
    source_files += [REPO / "dft_forge" / "executor.py", REPO / "dft_forge" / "tools" / "graph_tools.py", Path(__file__)]
    for src in source_files:
        target = evidence / "source" / src.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    rows = ["# DFT-Forge ABINIT 真实验证", "",
            f"ABINIT {summary['abinit_version']}；Si；ecut={summary['ecut_ha']:g} Ha；k 网格 {summary['ngkpt']}；q 网格 {summary['ngqpt']}。", "",
            "五条模板均使用本地真实二进制。该演示验证软件链路；计算参数尚未做收敛扫描，数值不作为已收敛材料结论。", "",
            "| 模板 | 状态 | 用时（秒） |", "|---|---|---:|"]
    rows += [f"| {r['case']} | {r.get('state', 'error')} | {r['elapsed_seconds']} |" for r in results]
    phonon = next((r.get("outputs", {}) for r in results if r["case"] == "a3_abinit_phonon"), {})
    if phonon.get("frequencies_thz"):
        rows += ["", f"声子结果共 {len(phonon['frequencies_thz'])} 个频率值，最低 {min(phonon['frequencies_thz']):.6f} THz。"]
        if phonon.get("has_imaginary_modes"):
            rows += ["当前网格结果存在虚频；需做截断能、k/q 网格和几何收敛后再判断材料动力学稳定性。"]
    rows += ["", "详细输入、逐节点执行来源与结果见各模板 JSON；原始输入、输出、最终结构和曲线在 evidence/；manifest.json 可核对 SHA-256。",
             "", "本次下载的公开赝势包含在 pseudos/（使用外部赝势库时需自行提供）；来源和校验值见 summary.json。",
             "", "完整运行目录保留密度和波函数供本机回放，下载证据包省略这些可再生大文件。原始输入和 JSON 保留执行时的绝对路径；迁移后请从项目仓库运行 scripts/abinit_demo.py 生成新路径。evidence/source/ 是审阅用源码快照。"]
    (out / "report.md").write_text("\n".join(rows) + "\n")
    files = sorted(p for p in evidence.rglob("*") if p.is_file())
    files += sorted(p for p in (out / "pseudos").glob("*") if p.is_file())
    files += [out / f"{r['case']}.json" for r in results]
    files += [out / "summary.json", out / "report.md"]
    manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in files]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(out / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, out / "manifest.json"]:
            archive.write(path, path.relative_to(out))


if __name__ == "__main__":
    raise SystemExit(main())
