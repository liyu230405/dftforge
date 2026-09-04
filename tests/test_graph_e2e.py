"""End-to-end tests: templates + QECalcTool through the graph runtime."""

from pathlib import Path

import pytest

from dft_forge.engines.qe import QECalcTool, QEParamRepairer
from dft_forge.executor import FakeExecutor
from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.run import NodeRun, ToolError, create_run, resolve_params
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.runtime.graph import load_template

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
QE_BIN = Path("/Users/liyu/Documents/Codex/2026-07-29/bang-w/work/qe-7.5/bin")
HAS_QE = QE_BIN.is_dir()


def make_engine(tmp_path, executor=None, max_concurrency=2):
    from dft_forge.runtime.engine import GraphEngine as GE

    return GE(
        base_dir=tmp_path / "ws",
        store_path=tmp_path / "ws" / "state.db",
        tools={"qe": QECalcTool(executor=executor or FakeExecutor())},
        repairers={"qe_param_bump": QEParamRepairer()},
        templates_dir=TEMPLATES_DIR,
        max_concurrency=max_concurrency,
    )


class TestTemplates:
    @pytest.mark.parametrize("tid", ["t1_vc_relax", "t2_bands", "t2_dos"])
    def test_template_valid(self, tid):
        template = load_template(TEMPLATES_DIR / f"{tid}.json")
        assert template.validate(known_tools={"qe"}) == []

    def test_engine_lists_templates(self, tmp_path):
        engine = make_engine(tmp_path)
        assert set(engine.list_templates()) >= {"t1_vc_relax", "t2_bands", "t2_dos"}

    def test_t2_bindings_resolve_after_success(self):
        template = load_template(TEMPLATES_DIR / "t2_bands.json")
        run = create_run(template, {"material": "GaAs"})
        run.nodes["scf"].state = NodeState.SUCCEEDED
        run.nodes["scf"].outputs = {"stdout_file": "/x/scf.out"}
        run.nodes["nscf"].state = NodeState.SUCCEEDED
        run.nodes["nscf"].outputs = {"stdout_file": "/x/nscf.out"}
        params = resolve_params(template.node("bands").params, run)
        assert params["scf_stdout"] == "/x/scf.out"
        assert params["nscf_stdout"] == "/x/nscf.out"
        assert params["material"] == "GaAs"


class TestTimeoutRepair:
    def _timeout_error(self):
        return ToolError("pw.x failed: QE timed out after 600s", category="execution", repairable=True)

    def test_timeout_reduces_band_path_points(self):
        repairer = QEParamRepairer()
        node = NodeRun(node_id="nscf", params={"nkpoints_bands": 100, "ecutwfc": 45.0})
        assert repairer.repair(node, self._timeout_error()) is True
        assert node.params["nkpoints_bands"] == 60
        assert node.params["ecutwfc"] == 45.0  # cutoff must NOT grow on timeout

    def test_timeout_repairs_stay_above_floor(self):
        repairer = QEParamRepairer()
        node = NodeRun(node_id="nscf", params={"nkpoints_bands": 100})
        node.repair_attempts = 0
        for expected in (60, 36, 24):
            assert repairer.repair(node, self._timeout_error()) is True
            assert node.params["nkpoints_bands"] == expected
        # at the floor there is nothing left to shrink
        assert repairer.repair(node, self._timeout_error()) is False

    def test_timeout_reduces_uniform_mesh(self):
        repairer = QEParamRepairer()
        node = NodeRun(node_id="scf", params={"kpoints": [8, 8, 1, 1, 1, 1]})
        assert repairer.repair(node, self._timeout_error()) is True
        assert node.params["kpoints"] == [4, 4, 1, 1, 1, 1]

    def test_convergence_still_bumps_cutoff(self):
        repairer = QEParamRepairer()
        err = ToolError("scf not converged", category="scf_not_converged", repairable=True)
        node = NodeRun(node_id="scf", params={"ecutwfc": 45.0})
        assert repairer.repair(node, err) is True
        assert node.params["ecutwfc"] == 55.0


class TestQECalcToolErrors:
    def _tool(self):
        return QECalcTool(executor=FakeExecutor())

    def test_missing_material(self, tmp_path):
        from dft_forge.runtime.run import ExecutionContext

        tool = self._tool()
        node = NodeRun(node_id="n", params={"calc": "scf"})
        with pytest.raises(ToolError, match="material"):
            tool.execute(node, ExecutionContext(base_dir=tmp_path, run_id="r"))

    def test_unknown_calc(self, tmp_path):
        from dft_forge.runtime.run import ExecutionContext

        tool = self._tool()
        node = NodeRun(node_id="n", params={"calc": "eph", "material": "Si"})
        with pytest.raises(ToolError, match="unknown qe calc"):
            tool.execute(node, ExecutionContext(base_dir=tmp_path, run_id="r"))

    def test_nscf_without_upstream_save(self, tmp_path):
        from dft_forge.runtime.run import ExecutionContext

        tool = self._tool()
        node = NodeRun(node_id="n", params={"calc": "nscf", "material": "Si"})
        ctx = ExecutionContext(base_dir=tmp_path, run_id="r", upstream={"scf": {}})
        with pytest.raises(ToolError, match="save"):
            tool.execute(node, ctx)

    def test_scf_compiles_input(self, tmp_path):
        from dft_forge.runtime.run import ExecutionContext

        tool = self._tool()
        node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
        with pytest.raises(ToolError):  # fake stdout cannot pass verification
            tool.execute(node, ExecutionContext(base_dir=tmp_path, run_id="r"))
        assert (tmp_path / "scf" / "si_scf.in").exists()


class TestGraphE2EFake:
    def test_t1_fails_after_repairs_offline(self, tmp_path):
        """FakeExecutor stdout fails physics verification: repair path exercises
        two ecut bumps, then the node fails deterministically."""
        engine = make_engine(tmp_path)
        run = engine.run_template("t1_vc_relax", {"material": "NaCl"})
        assert run.state == RunState.FAILED
        node = run.nodes["vc_relax"]
        assert node.state == NodeState.FAILED
        assert node.repair_attempts == 2
        assert node.params["ecutwfc"] == pytest.approx(65.0)  # 45 -> 55 -> 65
        assert "verification" in (node.error or "")

    def test_run_persisted_and_resumable(self, tmp_path):
        engine = make_engine(tmp_path)
        run = engine.run_template("t1_vc_relax", {"material": "Si"})
        status = engine.status(run.run_id)
        assert status["state"] == "failed"
        assert status["nodes"]["vc_relax"]["repair_attempts"] == 2


@pytest.mark.skipif(not HAS_QE, reason="local QE 7.5 build not available")
class TestGraphE2ERealQE:
    def test_t1_nacl_real_pw(self, tmp_path):
        from dft_forge.executor import LocalExecutor

        engine = make_engine(tmp_path, executor=LocalExecutor(qe_bin_dir=QE_BIN))
        run = engine.run_template("t1_vc_relax", {"material": "NaCl"})
        node = run.nodes["vc_relax"]
        if run.state != RunState.SUCCEEDED:
            pytest.fail(f"graph run failed: {node.error}")
        assert node.outputs["energy_ry"] == pytest.approx(-128.9, abs=0.5)
        assert node.outputs["a_angstrom"] == pytest.approx(5.6, abs=0.15)
        # artifacts live under <run_id>/<node_id>/
        assert (engine.base_dir / run.run_id / "vc_relax" / "nacl_vcrelax.in").exists()

    def test_t2_bands_si_real(self, tmp_path):
        from dft_forge.executor import LocalExecutor

        engine = make_engine(tmp_path, executor=LocalExecutor(qe_bin_dir=QE_BIN))
        run = engine.run_template("t2_bands", {"material": "Si"})
        if run.state != RunState.SUCCEEDED:
            errors = {n: r.error for n, r in run.nodes.items() if r.error}
            pytest.fail(f"t2 graph failed: {errors}")
        assert run.nodes["scf"].state == NodeState.SUCCEEDED
        assert run.nodes["bands"].outputs["n_bands"] > 0
