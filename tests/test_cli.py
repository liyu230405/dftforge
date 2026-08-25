"""Tests for the agent-friendly CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dft_forge.cli import main


class TestCLIStructure:
    def test_structure_import_poscar(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        argv = [
            "dft-forge",
            "structure",
            "import",
            str(Path(__file__).resolve().parent / "fixtures" / "POSCAR_Si"),
            "--output",
            str(tmp_path / "report.json"),
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        data = json.loads((tmp_path / "report.json").read_text())
        assert data["ok"] is True
        assert data["command"] == "structure.import"
        assert data["success"] is True
        assert data["natoms"] == 2

    def test_structure_import_cif(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        argv = [
            "dft-forge",
            "structure",
            "import",
            str(Path(__file__).resolve().parent / "fixtures" / "Si.cif"),
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["success"] is True

    def test_structure_analyze_poscar(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        argv = [
            "dft-forge",
            "structure",
            "analyze",
            str(Path(__file__).resolve().parent / "fixtures" / "POSCAR_Si"),
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["command"] == "structure.analyze"
        assert data["success"] is True
        analysis = data.get("analysis") or {}
        assert analysis["formula"] == "Si2"
        assert analysis["natoms"] == 2
        assert analysis["minimum_distance_angstrom"] is not None


class TestCLIMaterials:
    def test_materials_list(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        argv = ["dft-forge", "materials", "list"]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["command"] == "materials.list"
        assert data["count"] >= 1
        names = [m["material"] for m in data["materials"]]
        assert "Si" in names


class TestCLIInput:
    def test_input_build_material(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "Si.scf.in"
        argv = [
            "dft-forge",
            "input",
            "build",
            "--material",
            "Si",
            "--type",
            "scf",
            "--prefix",
            "si",
            "--output",
            str(out),
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["command"] == "input.build"
        assert data["material"] == "Si"
        assert data["calc_type"] == "scf"
        assert out.exists()
        text = out.read_text()
        assert "calculation = \"scf\"" in text
        assert "prefix = \"si\"" in text


class TestCLIDoctor:
    def test_doctor_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        argv = ["dft-forge", "doctor"]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["command"] == "doctor"
        assert "hostname" in data
        assert "python_packages" in data


class TestCLILedger:
    def test_ledger_record_and_query(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        db = tmp_path / "ledger.db"
        argv = [
            "dft-forge",
            "ledger",
            "record",
            "--database",
            str(db),
            "--task-id",
            "T1_Si_vcrelax",
            "--task-type",
            "T1",
            "--success",
            "false",
            "--verifier-passed",
            "false",
            "--failure-reasons",
            "SCF did not converge",
            "--failure-kind",
            "scf_not_converged",
            "--walltime-sec",
            "12.5",
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["row_id"] == 1

        argv = [
            "dft-forge",
            "ledger",
            "query",
            "--database",
            str(db),
            "--task-id",
            "T1_Si_vcrelax",
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1
        assert data["runs"][0]["failure_kind"] == "scf_not_converged"
