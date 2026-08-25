"""Tests for the 2D structure builder."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_forge.structure_builder import (
    MIN_DISTANCE_ANGSTROM,
    StructureBuildError,
    build_2d,
    parse_supercell,
    surface_sites,
)


class TestMonolayer:
    def test_graphene_primitive(self):
        r = build_2d("graphene")
        assert r.atoms.get_chemical_formula() == "C2"
        d = r.to_dict()["min_distance"]
        assert abs(d - 1.42) < 0.01  # C-C bond length

    def test_hbn(self):
        r = build_2d("bn")
        syms = set(r.atoms.get_chemical_symbols())
        assert syms == {"B", "N"}

    def test_vacuum_layer(self):
        r = build_2d("graphene", vacuum=20.0)
        c = r.atoms.cell[2, 2]
        assert 19.9 < c < 20.1

    def test_bad_kind_raises(self):
        with pytest.raises(StructureBuildError):
            build_2d("mos2")


class TestModifications:
    def test_supercell(self):
        r = build_2d("graphene", supercell="3x3")
        assert len(r.atoms) == 18

    def test_doping(self):
        r = build_2d("graphene", supercell="2x2", dopants=[{"index": 0, "element": "N"}])
        assert "N" in r.atoms.get_chemical_symbols()
        assert r.atoms.get_chemical_formula() == "C7N"

    def test_vacancy(self):
        r = build_2d("graphene", supercell="2x2", vacancy_index=1)
        assert len(r.atoms) == 7

    def test_adsorbate_all_sites(self):
        for site in ("top", "bridge", "hollow"):
            r = build_2d("graphene", supercell="2x2", adsorbate={"element": "O", "site": site})
            assert len(r.atoms) == 9
            assert r.to_dict()["min_distance"] > MIN_DISTANCE_ANGSTROM

    def test_bad_site_raises(self):
        with pytest.raises(StructureBuildError):
            build_2d("graphene", adsorbate={"element": "O", "site": "fcc4"})


class TestSiteGeometry:
    def test_sites_match_honeycomb_theory(self):
        r = build_2d("graphene")
        s = surface_sites(r.atoms)
        d = 2.46 / math.sqrt(3)  # C-C bond from lattice constant a=2.46
        assert abs(math.dist(s["top"][:2], s["bridge"][:2]) - d / 2) < 0.01
        assert abs(math.dist(s["top"][:2], s["hollow"][:2]) - d) < 0.01
        assert abs(math.dist(s["bridge"][:2], s["hollow"][:2]) - d * math.sqrt(3) / 2) < 0.01

    def test_sites_always_reported(self):
        r = build_2d("graphene")
        assert set(r.sites) == {"top", "bridge", "hollow"}

    def test_vacancy_degrades_sites_gracefully(self):
        r = build_2d("graphene", supercell="2x2", vacancy_index=1)
        assert len(r.atoms) == 7  # build still succeeds; sites are best-effort


class TestParseSupercell:
    def test_ok(self):
        assert parse_supercell("3x4") == (3, 4)
        assert parse_supercell("") == (1, 1)

    def test_bad(self):
        with pytest.raises(StructureBuildError):
            parse_supercell("axb")
        with pytest.raises(StructureBuildError):
            parse_supercell("0x2")
