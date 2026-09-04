"""Task: data-driven intent classification (vocab tables + traditional Chinese)."""

from __future__ import annotations

from dft_forge.agent_loop.intent import detect_intents


class TestKeywordTables:
    def test_compare_intent(self):
        sig = detect_intents("对比一下两个体系的结果")
        assert sig.wants_compare is True
        assert sig.wants_status is False  # compare suppresses status dead-end

    def test_band_target(self):
        sig = detect_intents("算一下 Si 的能带")
        assert "t2_bands" in sig.graph_targets
        assert sig.material_hint == "Si"

    def test_dos_target(self):
        sig = detect_intents("计算 NaCl 的态密度")
        assert "t2_dos" in sig.graph_targets
        assert sig.material_hint == "NaCl"

    def test_generic_calc_defaults_to_relax(self):
        sig = detect_intents("帮我计算 GaAs")
        assert sig.graph_targets == ["t1_vc_relax"]

    def test_analyze_suppresses_submit(self):
        sig = detect_intents("分析一下这个结构的键长")
        assert sig.wants_analyze is True
        assert sig.wants_submit is False
        assert sig.strong_analyze is True


class TestTraditionalChinese:
    """繁体提问经归一化后命中简体关键词表。"""

    def test_traditional_bands(self):
        sig = detect_intents("計算 Si 的能帶結構")
        assert "t2_bands" in sig.graph_targets

    def test_traditional_dos(self):
        sig = detect_intents("算一下態密度")
        assert "t2_dos" in sig.graph_targets

    def test_traditional_import(self):
        sig = detect_intents("導入這個結構檔案")
        assert sig.wants_import is True

    def test_traditional_compare(self):
        sig = detect_intents("對比一下結果")
        assert sig.wants_compare is True

    def test_traditional_monolayer(self):
        sig = detect_intents("建立單層石墨烯")
        assert sig.build2d_kind == "graphene"


class Test2DMaterialGrouping:
    def test_graphene_always_2d(self):
        assert detect_intents("构建石墨烯").build2d_kind == "graphene"

    def test_tmd_bulk_stays_bulk(self):
        # bulk MoS2 without 单层/2D keyword must NOT route to build2d
        sig = detect_intents("计算 MoS2 的能带")
        assert sig.build2d_kind is None
        assert "t2_bands" in sig.graph_targets

    def test_tmd_monolayer_keyword_triggers_2d(self):
        sig = detect_intents("构建单层 MoS2")
        assert sig.build2d_kind == "mos2"

    def test_tmd_adsorption_triggers_2d(self):
        sig = detect_intents("O 吸附在 MoS2 上")
        assert sig.build2d_kind == "mos2"
        assert sig.ads_el is not None

    def test_hbn_by_name(self):
        assert detect_intents("构建 h-BN 单层").build2d_kind == "bn"


class TestNewVocabIsOneLine:
    def test_adding_a_keyword_to_table_takes_effect(self):
        from dft_forge.agent_loop import intent as intent_mod

        original = intent_mod._INTENT_KEYWORDS["wants_eads"]
        try:
            intent_mod._INTENT_KEYWORDS["wants_eads"] = original + ("吸附能量",)
            assert detect_intents("算一下吸附能量").wants_eads is True
        finally:
            intent_mod._INTENT_KEYWORDS["wants_eads"] = original
