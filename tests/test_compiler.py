"""Tests for the QE compiler."""

import pytest
from pathlib import Path
from dft_forge.compiler import QECompiler, MATERIAL_DB, build_atoms, get_kpoints


class TestQECompiler:
    @pytest.fixture
    def pseudo_dir(self, tmp_path):
        # Create minimal pseudo directory
        pseudo_dir = tmp_path / "pseudos"
        pseudo_dir.mkdir()
        # Copy actual pseudos
        import shutil
        src = Path("/Users/liyu/Documents/Codex/dft-forge/assets/pseudos")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, pseudo_dir / f.name)
        return pseudo_dir
    
    def test_compile_si_vcrelax(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "si_vcrelax.in"
        
        content = compiler.compile_t1("Si", out_file)
        
        assert out_file.exists()
        assert 'calculation = "vc-relax"' in content
        assert 'prefix = "si"' in content
        assert "ibrav = 2" in content
        assert "Si_r.upf" in content
        assert "K_POINTS automatic" in content
    
    def test_compile_al_vcrelax(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "al_vcrelax.in"
        
        content = compiler.compile_t1("Al", out_file)
        
        assert out_file.exists()
        assert 'calculation = "vc-relax"' in content
        assert 'prefix = "al"' in content
        assert "Al.pbe-n-rrkjus_psl.1.0.2.UPF" in content
        assert "degauss = 0.02" in content  # Metal
    
    def test_compile_mgo_vcrelax(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "mgo_vcrelax.in"
        
        content = compiler.compile_t1("MgO", out_file)
        
        assert out_file.exists()
        assert "Mg.pz-n-vbc.UPF" in content
        assert "O-q6.gth" in content
        assert "ntyp = 2" in content
    
    def test_unknown_material_raises(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        with pytest.raises(ValueError, match="Unknown material"):
            compiler.compile_t1("Unknown", tmp_path / "test.in")
    
    def test_input_hash(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        content1 = compiler.compile_t1("Si", tmp_path / "test1.in")
        content2 = compiler.compile_t1("Si", tmp_path / "test2.in")
        
        hash1 = compiler.compute_input_hash(content1)
        hash2 = compiler.compute_input_hash(content2)
        
        assert hash1 == hash2  # Same material = same hash
        assert len(hash1) == 16


class TestBuildAtoms:
    def test_si_atoms(self):
        atoms = build_atoms("Si")
        assert len(atoms) == 2
        assert list(atoms.symbols) == ["Si", "Si"]
    
    def test_al_atoms(self):
        atoms = build_atoms("Al")
        assert len(atoms) == 1
        assert atoms.symbols[0] == "Al"
    
    def test_mgo_atoms(self):
        atoms = build_atoms("MgO")
        assert len(atoms) == 2
        assert set(atoms.symbols) == {"Mg", "O"}


class TestKpoints:
    def test_si_kpoints(self):
        kp = get_kpoints("Si")
        assert kp == (4, 4, 4, 1, 1, 1)
    
    def test_al_kpoints(self):
        kp = get_kpoints("Al")
        assert kp == (8, 8, 8, 1, 1, 1)
    
    def test_override_kpoints(self):
        kp = get_kpoints("Si", kpoints_override=(6, 6, 6, 0, 0, 0))
        assert kp == (6, 6, 6, 0, 0, 0)


class TestCompileSCF:
    """Tests for compile_scf used in T2 workflows."""

    @pytest.fixture
    def pseudo_dir(self, tmp_path):
        pseudo_dir = tmp_path / "pseudos"
        pseudo_dir.mkdir()
        import shutil
        src = Path("/Users/liyu/Documents/Codex/dft-forge/assets/pseudos")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, pseudo_dir / f.name)
        return pseudo_dir

    def test_compile_scf_si(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "si_scf.in"
        content = compiler.compile_scf("Si", out_file, calculation="scf")
        assert out_file.exists()
        assert 'calculation = "scf"' in content
        assert "prefix = \"si\"" in content
        assert "nbnd = 8" in content  # 2 atoms * 4

    def test_compile_nscf_al(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "al_nscf.in"
        content = compiler.compile_scf(
            "Al", out_file, calculation="nscf", nbnd=12
        )
        assert 'calculation = "nscf"' in content
        assert "nbnd = 12" in content
        assert "restart_mode = \"from_scratch\"" in content

    def test_compile_scf_mgo(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "mgo_scf.in"
        content = compiler.compile_scf("MgO", out_file, calculation="scf")
        assert "Mg.pz-n-vbc.UPF" in content
        assert "O-q6.gth" in content
        assert "ntyp = 2" in content


class TestCompileBandsInput:
    """Tests for compile_bands_input used in T2 band structure."""

    @pytest.fixture
    def pseudo_dir(self, tmp_path):
        pseudo_dir = tmp_path / "pseudos"
        pseudo_dir.mkdir()
        import shutil
        src = Path("/Users/liyu/Documents/Codex/dft-forge/assets/pseudos")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, pseudo_dir / f.name)
        return pseudo_dir

    def test_compile_bands_input_si(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "si_bands.in"
        content = compiler.compile_bands_input("Si", out_file, prefix="si")
        assert out_file.exists()
        assert 'prefix = "si"' in content
        assert "&bands" in content
        assert "outdir" in content
        # Should have high-symmetry path points
        assert "GAMMA" in content or "Gamma" in content

    def test_compile_bands_input_with_override(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "test_bands.in"
        kpath = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        klabels = ["GAMMA", "M"]
        content = compiler.compile_bands_input(
            "Si", out_file, prefix="si",
            kpath=kpath, klabels=klabels, nkpoints=50
        )
        assert "GAMMA" in content
        assert "M" in content
        assert "50" in content

    def test_compile_bands_input_mgo(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "mgo_bands.in"
        content = compiler.compile_bands_input("MgO", out_file, prefix="mgo")
        assert 'prefix = "mgo"' in content
        assert "&bands" in content


class TestCompileDOSInput:
    """Tests for compile_dos_input used in T2 DOS."""

    @pytest.fixture
    def pseudo_dir(self, tmp_path):
        pseudo_dir = tmp_path / "pseudos"
        pseudo_dir.mkdir()
        import shutil
        src = Path("/Users/liyu/Documents/Codex/dft-forge/assets/pseudos")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, pseudo_dir / f.name)
        return pseudo_dir

    def test_compile_dos_input_si(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "si_dos.in"
        content = compiler.compile_dos_input("Si", out_file, prefix="si")
        assert out_file.exists()
        assert 'prefix = "si"' in content
        assert "&dos" in content
        assert "degauss = 0.05" in content
        assert "DeltaE = 0.01" in content

    def test_compile_dos_input_with_range(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "al_dos.in"
        content = compiler.compile_dos_input(
            "Al", out_file, prefix="al",
            emin=-10.0, emax=20.0, deltae=0.02, fwhm=0.1, ngauss=0
        )
        assert "Emin = -10.0" in content
        assert "Emax = 20.0" in content
        assert "DeltaE = 0.02" in content
        assert "degauss = 0.1" in content
        assert "ngauss = 0" in content

    def test_compile_dos_input_mgo(self, tmp_path, pseudo_dir):
        compiler = QECompiler(pseudo_dir=pseudo_dir)
        out_file = tmp_path / "mgo_dos.in"
        content = compiler.compile_dos_input("MgO", out_file, prefix="mgo")
        assert 'prefix = "mgo"' in content
        assert "&dos" in content

