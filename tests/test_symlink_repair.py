"""Task: upstream .save symlink robustness (stale-link relink, resume warnings)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dft_forge.engines.qe import QECalcTool
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import ExecutionContext, GraphRun, NodeRun
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.runtime.store import SQLiteStateStore


def _ctx(base_dir: Path, upstream_ids: list) -> ExecutionContext:
    return ExecutionContext(
        base_dir=base_dir,
        run_id="r1",
        upstream={i: {} for i in upstream_ids},
    )


class TestLinkUpstreamSave:
    def _make_save(self, base_dir: Path, node_id: str, prefix: str = "si") -> Path:
        save_dir = base_dir / node_id / f"{prefix}.save"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "data-file-schema.xml").write_text("<x/>")
        return save_dir

    def test_links_absolute_and_survives_cwd_change(self, tmp_path: Path):
        base = tmp_path / "runs" / "r1"
        save = self._make_save(base, "scf")
        workdir = tmp_path / "node_work"
        workdir.mkdir()
        QECalcTool._link_upstream_save(_ctx(base, ["scf"]), workdir, "si")
        link = workdir / "si.save"
        assert link.is_symlink() and link.exists()
        # resolve()d absolute target: still valid from any cwd
        assert link.resolve() == save.resolve()

    def test_replaces_broken_stale_symlink(self, tmp_path: Path):
        base = tmp_path / "runs" / "r1"
        save = self._make_save(base, "scf")
        workdir = tmp_path / "node_work"
        workdir.mkdir()
        stale = workdir / "si.save"
        stale.symlink_to(tmp_path / "vanished" / "si.save", target_is_directory=True)
        assert stale.is_symlink() and not stale.exists()

        QECalcTool._link_upstream_save(_ctx(base, ["scf"]), workdir, "si")
        assert stale.is_symlink() and stale.exists()
        assert stale.resolve() == save.resolve()

    def test_prefers_nscf_over_scf(self, tmp_path: Path):
        base = tmp_path / "runs" / "r1"
        scf_save = self._make_save(base, "scf")
        nscf_save = self._make_save(base, "nscf")
        workdir = tmp_path / "node_work"
        workdir.mkdir()
        QECalcTool._link_upstream_save(_ctx(base, ["scf", "nscf"]), workdir, "si")
        assert (workdir / "si.save").resolve() == nscf_save.resolve() != scf_save.resolve()

    def test_missing_upstream_raises(self, tmp_path: Path):
        base = tmp_path / "runs" / "r1"
        base.mkdir(parents=True)
        workdir = tmp_path / "node_work"
        workdir.mkdir()
        from dft_forge.runtime.run import ToolError

        with pytest.raises(ToolError, match="no upstream"):
            QECalcTool._link_upstream_save(_ctx(base, ["scf"]), workdir, "si")


class TestResumeBrokenSymlinkWarning:
    def test_resume_run_warns_on_dead_link(self, tmp_path: Path, caplog):
        store = SQLiteStateStore(tmp_path / "state.db")
        template = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="a", tool="fake"), NodeSpec(id="b", tool="fake", depends_on=["a"])],
        )
        run = GraphRun(run_id="r1", template_id="t", inputs={})
        run.state = RunState.RUNNING
        run.nodes["a"] = NodeRun(node_id="a")
        run.nodes["b"] = NodeRun(node_id="b")
        store.save_run(run)

        # node b stuck Running with a broken upstream symlink in its workdir
        wd = tmp_path / "ws" / "r1" / "b"
        wd.mkdir(parents=True)
        (wd / "si.save").symlink_to(tmp_path / "gone" / "si.save", target_is_directory=True)
        run.nodes["b"].workdir = str(wd)
        run.nodes["b"].state = NodeState.RUNNING
        store.save_run(run)
        store.save_node_run(run, run.nodes["b"])

        with caplog.at_level(logging.WARNING):
            resumed = store.resume_run("r1", template)
        assert resumed is not None
        assert resumed.nodes["b"].state == NodeState.PENDING
        assert any("broken symlink" in r.message for r in caplog.records)
