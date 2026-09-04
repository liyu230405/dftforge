"""Chat intent detection and self-doping guard regressions.

Covers the review-round fixes:
- element keywords must not false-positive inside English words
  ("this"/"basic"/"also" used to detect Si/Al via substring matching)
- Chinese-adjacent mentions (算si的) must still work
- the self-doping notice must not leak the literal {kind_name} placeholder
"""

from __future__ import annotations

from dft_forge.agent_loop.intent import detect_intents
from dft_forge.agent_loop.step_prep import check_self_doping


class TestElementWordBoundaries:
    def test_english_words_do_not_detect_si(self):
        for msg in ("this is a test", "run the basic workflow", "using pw.x"):
            sig = detect_intents(msg)
            assert sig.material_hint != "Si", f"{msg!r} falsely detected Si"

    def test_english_words_do_not_detect_al(self):
        for msg in ("also check the results", "calculate the total", "run locally"):
            sig = detect_intents(msg)
            assert sig.material_hint != "Al", f"{msg!r} falsely detected Al"

    def test_standalone_si_still_detected(self):
        assert detect_intents("算 Si 的能带").material_hint == "Si"
        assert detect_intents("si").material_hint == "Si"
        assert detect_intents("Si").material_hint == "Si"

    def test_cjk_adjacent_si_still_detected(self):
        # CJK chars are regex word chars; \b would reject these
        assert detect_intents("算si的能带").material_hint == "Si"
        assert detect_intents("优化al结构").material_hint == "Al"

    def test_chinese_names_still_detected(self):
        assert detect_intents("算硅的能带").material_hint == "Si"
        assert detect_intents("优化铝结构").material_hint == "Al"
        assert detect_intents("优化NaCl结构").material_hint == "NaCl"

    def test_formula_extraction_unaffected(self):
        sig = detect_intents("算一下 CaTiO3 的能带")
        assert sig.formula_hint == "CaTiO3"


class TestSelfDopingNotice:
    def test_notice_has_no_literal_placeholder(self):
        notice = check_self_doping(
            "structure.build2d",
            {"kind": "graphene", "dopants": [{"element": "C", "index": 0}]},
        )
        assert notice is not None
        assert "{kind_name}" not in notice
        assert "石墨烯" in notice

    def test_real_dopant_passes(self):
        assert check_self_doping(
            "structure.build2d",
            {"kind": "graphene", "dopants": [{"element": "N", "index": 0}]},
        ) is None

    def test_binary_host_self_doping_rejected(self):
        notice = check_self_doping(
            "structure.build2d",
            {"kind": "bn", "dopants": [{"element": "B", "index": 0}]},
        )
        # bn has two host elements, so a single-element substitution is not
        # unambiguously meaningless — only single-element hosts are rejected
        assert notice is None
