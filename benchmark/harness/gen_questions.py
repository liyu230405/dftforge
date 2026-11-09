#!/usr/bin/env python3
"""Generate the 30 benchmark question JSON files from the spec tables.

Single source of truth for the question set: edit the L1/L2/L3 tables below,
re-run, and questions/ is rewritten. Keeps the 30 files consistent instead of
hand-maintaining them.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "questions"

PLAN_SUFFIX = "只规划，不要实际运行。"
EXEC_SUFFIX = "直接运行本地 QE。本轮是有限参数下的链路验证，数值未经过收敛测试；只执行当前任务，不添加扫描。"

# Shared regex fragments. Written wide on purpose: a missed catch is reviewed by
# hand, a false catch would report a regression that did not happen.
RE = {
    "pbe_gap": r"PBE.*?(低估|偏小|underestimat)|不能?.*?(等同|直接比较|等于).*?实验|实验.*?(不能|不可).*?直接",
    "vc_vs_relax": r"(vc-?relax|变胞|可变晶胞).*?(晶胞|cell|晶格).*?(优化|放开|自由)|固定晶胞.*?relax|relax.*?固定.*?(晶胞|cell)",
    "force_and_stress": r"(力|force).*?(压力|应力|stress|pressure)|(压力|应力|stress|pressure).*?(力|force)",
    "single_var": r"(固定|保持).*?(其余|其他|另一)|单变量|逐一|一次只(改|变|扫)",
    "energy_not_property": r"能量收敛.*?(不等于|≠|不代表|不意味).*?(带隙|物性|声子|其他)|(带隙|物性|声子).*?(另需|单独|独立).*?收敛",
    "three_stage": r"SCF.*?NSCF.*?(bands|能带)",
    "projwfc": r"projwfc\.x",
    "pdos_x": r"\bpdos\.x\b",
    "denser_nscf": r"NSCF.*?(更密|加密|更多|致密|denser)|(更密|加密|致密).*?NSCF",
    "broadening": r"展宽|degauss|sigma|smearing|broadening",
    "atoms_16": r"\b16\s*(个)?原子|Si15P|Si_?15|16-atom",
    "not_64": r"\b64\s*(个)?原子",
    "half_o2": r"1/2\s*[·*x×]?\s*E?\s*\(?\s*O2|0\.5\s*[·*x×]?\s*E?\s*\(?\s*O2|½.*?O2|(半个|一半).*?O2|E\(O2\)\s*/\s*2",
    "fixed_cell_ads": r"固定(晶胞|胞|面内|晶格)|面内晶格.*?(不变|固定)|fixed.?cell",
    "o2_triplet": r"三重态|triplet|自旋极化.*?O2|O2.*?自旋极化",
    "over_budget": r"over_?budget|超过.*?(上限|预算|限制)|动作数.*?超",
    "o2_gas_ref": r"O2.*?(分子|气相|gas)|(分子|气相|gas).*?O2",
    "not_hull": r"(不等于|≠|不代表|不意味|不能说明).*?(convex hull|凸包|热力学稳定)|(convex hull|凸包).*?(未|没有|不).*?(穷举|验证|检索)",
    "method_consistent": r"(方法|赝势|泛函|截断|参数).*?(一致|相同|统一)|同一.*?(方法|赝势|泛函)",
    "mu_same_method": r"same_method_bulk_per_atom|同方法.*?(每原子|E/N|per.?atom)|完美超胞.*?(每原子|平均)",
    "not_dope_for_vacancy": r"",  # handled by tool_not_planned
    "relax_around_defect": r"(固定晶胞|fixed.?cell).*?(弛豫|relax)|弛豫.*?(固定晶胞|fixed.?cell)|(缺陷|空位).*?周围.*?(松弛|弛豫)",
    "supercell_too_small": r"(超胞|尺寸|16\s*原子).*?(偏小|不足|未收敛|太小)|(缺陷|空位).*?(相互作用|自相互作用).*?(未|没)收敛|有限尺寸",
    "fe_not_unsupported": r"(无|没有|缺).*?Fe.*?(结构|材料库)|Fe.*?(不在|未收录).*?(材料库|库)",
    "initial_moment": r"initial_moment|初始磁矩|starting_magnetization",
    "afm_one_pattern": r"(一种|单一|仅).*?(AFM|反铁磁|交替|子格)|AFM.*?(初态|初始|不等于|≠).*?(基态|全局|穷举)",
    "realized_moment": r"(实际|收敛后|最终).*?磁矩|磁矩.*?(实际|收敛|最终|realized)",
    "phonon_not_unsupported": r"(声子|phonon|DFPT).*?(不支持|无法|没有能力)",
    "phonon_chain": r"ph\.x.*?q2r\.x.*?matdyn\.x|ph\.x.*?matdyn",
    "imaginary_freq": r"(虚频|虚频率|imaginary|负频)",
    "lo_to": r"LO.?TO|非解析|nonanalytic|non-analytic|极性.*?修正",
    "gamma_only_caveat": r"(仅|只有?).*?(Γ|Gamma|gamma).*?(不|≠|不能).*?(代表|等于|外推|说明)|(Γ|Gamma).*?点.*?(不足以|不代表)",
    "metal_cu": r"(Cu|铜).*?(金属|metal)|费米能.*?(有态|非零|存在态)",
    "insulator_nacl": r"(NaCl|氯化钠).*?(绝缘|带隙)|(宽|大).*?带隙|EF.*?(带隙|gap).*?(中|内)",
    "fermi_not_fake": r"",  # numeric check instead
    "no_hull_claim": r"convex hull|凸包",
    "alternative": r"(上传|导入|提供|自带|改用|换成|拆分|分批|单独|手动).*?(结构|文件|CIF|POSCAR|赝势|UPF|请求|体系)|建议.*?(上传|导入|提供|拆)",
    "slab_unsupported": r"(slab|表面|surface).*?(不支持|无法|没有|缺少).*?(构建|切|生成)|无法.*?(切出|构建|生成).*?(slab|表面|\(111\))",
    "not_2d_substitute": r"",  # tool_not_planned on structure.build2d
    "molecule_ads_unsupported": r"(单原子|单个原子).*?吸附|(分子|多原子).*?(吸附).*?(不支持|暂不|无法)",
    "orientation_search": r"(取向|朝向|orientation).*?(搜索|采样|未|不)",
    "missing_s_ref": r"\bS\b.*?(参考态|reference)|(参考态|reference).*?\bS\b|缺.*?S.*?(参考|reference)",
    "zns_can_scf": r"(ZnS).*?(可以|能|支持).*?(SCF|能带|结构)|(SCF|能带).*?(可以|能|支持)",
    "missing_u_pseudo": r"\bU\b.*?(赝势|pseudo|UPF)|(赝势|pseudo).*?\bU\b|缺.*?U.*?(赝势|pseudo)",
    "dft_plus_u": r"DFT\+U|LDA\+U|lda_plus_u|强关联|Hubbard",
    "mu_single_element_only": r"same_method_bulk_per_atom.*?(单元素|纯元素|单一元素)|(单元素|纯元素).*?same_method|化合物.*?(不适用|无效|不能)",
    "mu_window": r"(化学势|μ|mu).*?(窗口|范围|区间|极限)|富(氧|O|Mg).*?(极限|条件)",
    "soc_unsupported": r"(SOC|自旋轨道|spin.?orbit).*?(不支持|无法|没有)|(不支持|无).*?(SOC|自旋轨道|spin.?orbit)",
    "bi2se3_missing": r"Bi2Se3.*?(不在|没有|未收录|缺).*?(库|原型|材料)|(需要|请).*?(上传|提供).*?(Bi2Se3|结构)",
    "no_scalar_substitute": r"",  # must_not_match handled separately
    "topological_claim": r"(拓扑表面态|topological surface state).*?(可以|能够|将会|算出)",
    "all_five_materials": r"",  # special rule
    "batch_alternative": r"(拆|分批|分次|多次|逐个|单独).*?(请求|提交|运行|规划)",
    "n_type": r"n.?型|n.?type|电子掺杂|EF.*?(上移|升高|移向导带)",
    "vacuum_15": r"",  # numeric
    "kz_one": r"",  # numeric
}


def chk(cid, kind, rule, weight=1.0, **args):
    d = {"id": cid, "kind": kind, "weight": weight, "rule": rule}
    if args:
        d["args"] = args
    return d


def no_side_effects(cid="c_planonly", weight=1.0):
    return chk(cid, "tool_trace", "no_side_effect_calls", weight)


def audit_all(cid="c_audit", weight=1.0):
    return chk(cid, "audit", "gating_all_pass", weight)


# --------------------------------------------------------------------------
# L1 PLAN (10)
# --------------------------------------------------------------------------
PLAN = [
    dict(
        id="PLAN-01", title="单模板路由：结构优化",
        prompt="帮我优化金刚石硅的结构，我要平衡晶格常数、残余力和压力。",
        support=["complete"],
        checks=[
            chk("c1", "tool_trace", "template_used", template_id="t1_vc_relax"),
            no_side_effects("c2"),
            chk("c3", "assertion", "must_match", patterns=[RE["vc_vs_relax"]]),
            chk("c4", "assertion", "must_match", patterns=[RE["force_and_stress"]]),
            chk("c5", "assertion", "support_status_in", allowed=["complete"]),
        ],
        notes="最基础的路由题。选 t1_vc_relax 而非固定晶胞的 relax。",
    ),
    dict(
        id="PLAN-02", title="双层收敛扫描",
        prompt="我要把 Si 的 ecutwfc 和 k 点收敛到 1 meV/atom，给我一个可复现的扫描方案。",
        support=["complete", "partial"],
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="workflow.convergence"),
            no_side_effects("c2"),
            chk("c3", "tool_trace", "tool_planned", tool="workflow.convergence",
                args_contains={"threshold_mev_per_atom": 1}),
            chk("c4", "assertion", "must_match", patterns=[RE["single_var"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["energy_not_property"]]),
        ],
        notes="收敛扫描是一等工具，不能只口头承诺后续自动执行（2026-09-07 P2 缺陷）。",
    ),
    dict(
        id="PLAN-03", title="能带与带隙",
        prompt="算一下 GaAs 的能带结构和带隙，走高对称路径。",
        support=["complete"],
        checks=[
            chk("c1", "tool_trace", "template_used", template_id="t2_bands"),
            no_side_effects("c2"),
            chk("c3", "assertion", "must_match", patterns=[RE["three_stage"]]),
            chk("c4", "assertion", "must_match", patterns=[RE["pbe_gap"]]),
            chk("c5", "assertion", "must_match", weight=0.5,
                patterns=[r"直接带隙|direct.?gap|Γ.*?(VBM|CBM|带隙)"]),
        ],
        notes="c5 是材料常识加分项。",
    ),
    dict(
        id="PLAN-04", title="DOS 与投影 DOS",
        prompt="我要 MgO 的总态密度和 Mg、O 分别的投影态密度。",
        support=["complete"],
        checks=[
            chk("c1", "tool_trace", "template_used", template_id="t2_dos"),
            no_side_effects("c2"),
            chk("c3", "assertion", "must_match", patterns=[RE["projwfc"]]),
            chk("c4", "assertion", "must_not_match", patterns=[RE["pdos_x"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["denser_nscf"]]),
            chk("c6", "assertion", "must_match", patterns=[RE["broadening"]]),
        ],
        notes="c4：pdos.x 这个 QE 程序不存在，正确名是 projwfc.x（2026-09-07 P4 缺陷）。",
    ),
    dict(
        id="PLAN-05", title="掺杂对照（原子数陷阱）",
        prompt="比较 P 掺杂 Si 和纯 Si 的能带，用 2x2x2 超胞。",
        support=["complete", "partial"],
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="structure.dope"),
            chk("c2", "tool_trace", "tool_planned", tool="structure.supercell"),
            chk("c3", "tool_trace", "no_self_doping"),
            chk("c4", "tool_trace", "tool_planned", tool="analysis.compare"),
            no_side_effects("c5"),
            chk("c6", "assertion", "must_match", patterns=[RE["atoms_16"]]),
            chk("c7", "assertion", "must_not_match", patterns=[RE["not_64"]]),
            chk("c8", "assertion", "must_match", weight=0.5,
                patterns=[r"6\.25\s*%|高掺杂|浓度.*?(高|偏高)|需要.*?弛豫"]),
            chk("c9", "assertion", "must_match", weight=0.5,
                patterns=[r"(能带)?折叠|band.?folding|超胞.*?布里渊区.*?(缩小|折叠)"]),
        ],
        notes="三个历史陷阱：原子数 16 非 64；纯对照不能自掺杂；需要比较步骤收口。",
    ),
    dict(
        id="PLAN-06", title="多位点吸附（预算边界）",
        prompt="算 O 在 MoS2 单层上 top、bridge、hollow 三个位点的吸附能，用 O2 做参考态。",
        support=["complete", "partial"],
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="workflow.adsorption_scan",
                args_contains={"kind": "mos2"}),
            chk("c2", "tool_trace", "tool_planned", tool="workflow.adsorption_scan",
                args_contains={"sites": ["top", "bridge", "hollow"]}),
            chk("c3", "tool_trace", "tool_planned", tool="workflow.adsorption_scan",
                args_contains={"adsorbate_element": "O"}),
            no_side_effects("c4"),
            chk("c5", "assertion", "must_match", patterns=[RE["half_o2"]]),
            chk("c6", "assertion", "must_match", patterns=[RE["fixed_cell_ads"]]),
            chk("c7", "assertion", "must_match", patterns=[RE["o2_triplet"]]),
            chk("c8", "assertion", "budget_honest",
                note="若动作数>8 须标 over_budget+partial 且完整描述留在 advisory"),
        ],
        notes="c5：公式必须是 1/2·E(O2)（2026-09-07 P6 把它写成 E(O2)）。",
    ),
    dict(
        id="PLAN-07", title="形成能（可闭环）",
        prompt="算 MgO 相对于 Mg 和 O 参考态的形成能。",
        support=["complete"],
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="thermo.formation"),
            chk("c2", "tool_trace", "energy_sources_count", minimum=3),
            no_side_effects("c3"),
            chk("c4", "assertion", "must_match", patterns=[RE["o2_gas_ref"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["not_hull"]]),
            chk("c6", "assertion", "must_match", patterns=[RE["method_consistent"]]),
        ],
        notes="Mg 与 O 都在 REFERENCE_PHASES（33 元素）里，这题能闭环。",
    ),
    dict(
        id="PLAN-08", title="中性空位",
        prompt="算 Si 里一个中性空位的形成能，用 2x2x2 超胞。",
        support=["complete"],
        checks=[
            chk("c1", "tool_trace", "tool_planned_any",
                tools=["workflow.neutral_vacancy", "structure.vacancy"]),
            chk("c2", "tool_trace", "tool_not_planned", tool="structure.dope"),
            no_side_effects("c3"),
            chk("c4", "assertion", "must_match", patterns=[RE["mu_same_method"]]),
            chk("c5", "assertion", "must_match",
                patterns=[r"16.*?15|15.*?16|完美.*?16.*?缺陷.*?15"]),
            chk("c6", "assertion", "must_match", patterns=[RE["relax_around_defect"]]),
            chk("c7", "assertion", "must_match", patterns=[RE["supercell_too_small"]]),
        ],
        notes="c2：空位是移除不是替换（2026-09-07 P8 把 structure.dope 当空位工具）。",
    ),
    dict(
        id="PLAN-09", title="磁性基态比较",
        prompt="比较 bcc Fe 的 NM、FM、AFM 三种磁序的总能量和磁矩。",
        support=["complete", "partial"],
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="workflow.magnetic_compare",
                args_contains={"orders": ["NM", "FM", "AFM"]}),
            no_side_effects("c2"),
            chk("c3", "assertion", "must_not_match", patterns=[RE["fe_not_unsupported"]]),
            chk("c4", "assertion", "must_match", patterns=[RE["initial_moment"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["afm_one_pattern"]]),
            chk("c6", "assertion", "must_match", patterns=[RE["realized_moment"]]),
        ],
        notes="c3：Fe 不在 MATERIAL_DB 但走 formula_atoms 的 bcc 原型，赝势存在。历史拒答已过时。",
    ),
    dict(
        id="PLAN-10", title="声子与动力学稳定性",
        prompt="算 MgO 的声子，我想知道它动力学上稳不稳定。",
        support=["complete", "partial"],
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="workflow.phonon_dfpt"),
            chk("c2", "tool_trace", "no_qe_phonon_template"),
            no_side_effects("c3"),
            chk("c4", "assertion", "must_not_match", patterns=[RE["phonon_not_unsupported"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["phonon_chain"]]),
            chk("c6", "tool_trace", "q_grid_not_gamma_only"),
            chk("c7", "assertion", "must_match", patterns=[RE["imaginary_freq"]]),
            chk("c8", "assertion", "must_match", weight=0.5, patterns=[RE["lo_to"]]),
        ],
        notes="c2：QE 声子只能走 workflow.phonon_dfpt，没有 QE 声子 JSON 模板。c8 为加分项。",
    ),
]

# --------------------------------------------------------------------------
# L2 EXEC (13) — ground_truth values are placeholders until the first live
# round freezes them; see SCORING.md §7.
# --------------------------------------------------------------------------
GT_PENDING = None  # filled by freeze_gt.py after the first two live rounds

EXEC = [
    dict(
        id="EXEC-01", title="Si vc-relax",
        prompt="优化金刚石硅的结构，做 vc-relax，返回晶格参数、最大受力和压力。",
        run_config={"ecutwfc": 30, "ecutrho": 240, "kpoints": [4, 4, 4, 1, 1, 1]},
        budget=120,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded", nodes=["vc_relax"]),
            chk("c2", "numeric", "graded_relative", field="a_conventional_angstrom",
                value=5.467902, tolerance=0.005, unit="angstrom"),
            chk("c3", "numeric", "upper_bound", field="pressure_kbar_abs",
                limit=0.5, unit="kbar"),
            chk("c4", "numeric", "upper_bound", field="max_force_ry_bohr",
                limit=1e-3, unit="Ry/bohr"),
            audit_all("c5"),
        ],
        notes="GT 来自 2026-09-22 实测 a=5.467902 Å。预算 ~5 s。",
    ),
    dict(
        id="EXEC-02", title="Si 能带与带隙",
        prompt="计算金刚石硅的能带与带隙，用 12 条能带和 60 个路径点。",
        run_config={"ecutwfc": 30, "ecutrho": 240, "kpoints": [4, 4, 4, 1, 1, 1],
                    "nbnd": 12, "nkpoints_bands": 60},
        budget=120,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded", nodes=["scf", "nscf", "bands"]),
            chk("c2", "numeric", "graded_absolute", field="band_gap_ev",
                value=0.575106, tolerance=0.04, unit="eV"),
            chk("c3", "numeric", "exact", field="n_atoms", value=2),
            chk("c4", "assertion", "must_match", patterns=[RE["pbe_gap"]]),
            audit_all("c5"),
        ],
        notes="GT 0.575106 eV 是 ecut=30/k=4³ 下的链路值，非收敛值也非实验 1.17 eV。",
    ),
    dict(
        id="EXEC-03", title="Cu 金属 DOS 与 PDOS",
        prompt="计算铜的总 DOS 和 Cu 投影 DOS，看费米能附近有没有态，用 16 条能带。",
        run_config={"ecutwfc": 30, "ecutrho": 240, "kpoints": [4, 4, 4, 1, 1, 1],
                    "nbnd": 16},
        budget=120,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded", nodes=["scf", "nscf", "dos", "pdos"]),
            chk("c2", "numeric", "graded_relative", field="dos_at_fermi",
                value=0.272700, tolerance=0.05, unit="states/eV/cell"),
            chk("c3", "numeric", "graded_relative", field="pdos_cu_at_fermi",
                value=0.272924, tolerance=0.05, unit="states/eV/cell"),
            chk("c4", "numeric", "nonzero", field="e_fermi_ev",
                note="费米能必须是真实解析值，不是伪造的 0"),
            chk("c5", "assertion", "must_match", patterns=[RE["metal_cu"]]),
            audit_all("c6"),
        ],
        notes="与 EXEC-09（NaCl 绝缘体）成对照。",
    ),
    dict(
        id="EXEC-04", title="Fe 三磁序比较",
        prompt=("用 workflow.magnetic_compare 比较 bcc Fe 的 NM/FM/AFM 三种共线初态的"
                "能量和实际磁矩，1x1x1 常规胞、initial_moment_muB=2.2，不做结构优化。"),
        run_config={"ecutwfc": 35, "ecutrho": 280, "kpoints": [4, 4, 4, 1, 1, 1]},
        budget=180,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded", nodes=["nm", "fm", "afm"]),
            chk("c2", "numeric", "lowest_energy_order", field="magnetic_orders",
                expected="FM"),
            chk("c3", "numeric", "graded_relative", field="fm_total_moment_muB",
                value=5.35, tolerance=0.10, unit="muB/2atoms"),
            chk("c4", "assertion", "must_match",
                patterns=[r"AFM.*?(总磁矩|total).*?(0|零|接近零).*?(绝对|absolute).*?(非零|不为零)"
                          r"|绝对磁矩.*?(非零|不为零)"]),
            chk("c5", "assertion", "must_match", patterns=[RE["afm_one_pattern"]]),
            audit_all("c6"),
        ],
        notes="GT FM 总磁矩 5.35 μB/2 原子（2026-09-22 实测）。",
    ),
    dict(
        id="EXEC-05", title="MgO Γ 点声子",
        prompt=("用 workflow.phonon_dfpt 算 MgO 的 Γ 点声子，q_grid=[1,1,1]、relax=true、"
                "tr2_ph=1e-12。作为 Γ 链路测试，不外推全布里渊区稳定性。"),
        run_config={"ecutwfc": 40, "ecutrho": 320, "kpoints": [2, 2, 2, 0, 0, 0]},
        budget=120,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded",
                nodes=["relax", "scf", "ph", "q2r", "matdyn"]),
            chk("c2", "numeric", "acoustic_modes_near_zero", field="phonon_freqs_thz",
                limit=0.5, count=3, unit="THz"),
            chk("c3", "numeric", "graded_relative", field="optical_mode_thz",
                value=25.9187, tolerance=0.05, unit="THz"),
            chk("c4", "assertion", "must_match",
                patterns=[r"(0\.1\s*THz|阈值).*?(未|没有|无).*?(虚频|显著虚频)"]),
            chk("c5", "assertion", "must_match", patterns=[RE["gamma_only_caveat"]]),
            chk("c6", "assertion", "must_match", patterns=[RE["method_consistent"]]),
            audit_all("c7"),
        ],
        notes="GT 光学模 25.9187 THz。c6 针对 2026-09-22 修复的弛豫→SCF 赝势传递 bug。",
    ),
    dict(
        id="EXEC-06", title="Si 中性空位形成能",
        prompt=("运行 Si 中性空位形成能：workflow.neutral_vacancy，source=Si、element=Si、"
                "supercell=2x2x2、relax=true、reference_mode=same_method_bulk_per_atom。"
                "显式以同方法完美 Si 超胞的每原子能量为化学势，不另填数值。"),
        run_config={"ecutwfc": 30, "ecutrho": 240, "kpoints": [2, 2, 2, 1, 1, 1]},
        budget=420,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded",
                nodes=["pristine_scf", "defect_relax", "defect_scf"]),
            chk("c2", "numeric", "graded_relative", field="formation_energy_ev",
                value=2.985109, tolerance=0.03, unit="eV"),
            chk("c3", "numeric", "exact", field="n_atoms_pristine", value=16),
            chk("c4", "numeric", "exact", field="n_atoms_defect", value=15),
            chk("c5", "numeric", "upper_bound", field="max_force_ry_bohr",
                limit=1e-3, unit="Ry/bohr"),
            chk("c6", "assertion", "must_match", patterns=[RE["mu_same_method"]]),
            chk("c7", "assertion", "must_match",
                patterns=[r"(无|没有|不含|未做).*?(带电|charged).*?(态|修正)|有限尺寸.*?(未|没有|不)"]),
            audit_all("c8"),
        ],
        notes="GT E_f=2.985109 eV。本套最重的题之一，实测 defect_relax 单节点 3 min 4 s。",
    ),
    dict(
        id="EXEC-07", title="P 掺杂 Si 对照",
        prompt=("生成 P 掺杂 Si 的 2x2x2 原胞超胞与同尺寸纯 Si 超胞，分别计算能带并比较带隙。"
                "每个体系 40 条能带、30 个路径点，不做弛豫。"),
        run_config={"ecutwfc": 25, "ecutrho": 200, "kpoints": [2, 2, 2, 1, 1, 1],
                    "nbnd": 40, "nkpoints_bands": 30},
        budget=420,
        checks=[
            chk("c1", "tool_trace", "node_count", minimum=6),
            chk("c2", "tool_trace", "no_self_doping"),
            chk("c3", "numeric", "exact", field="n_atoms_doped", value=16),
            chk("c4", "numeric", "exact", field="n_atoms_pristine", value=16),
            chk("c5", "assertion", "must_match",
                patterns=[r"(金属|metallic|展宽下?不可分辨|无法分辨).*?(Si15P|掺杂)"
                          r"|(Si15P|掺杂).*?(金属|metallic|不可分辨)"]),
            chk("c6", "assertion", "must_match",
                patterns=[r"6\.25\s*%|高掺杂|理想.*?未弛豫|未弛豫.*?理想"]),
            audit_all("c7"),
        ],
        notes="c5：Si15P 不能报告为可靠有限带隙（2026-09-22 实测 0.052817 eV 属展宽内噪声）。",
    ),
    dict(
        id="EXEC-08", title="GaAs 结构优化（新体系泛化）",
        prompt="优化 GaAs 的结构，返回晶格常数和压力。",
        run_config={"ecutwfc": 40, "ecutrho": 320, "kpoints": [4, 4, 4, 1, 1, 1]},
        budget=180,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded", nodes=["vc_relax"]),
            chk("c2", "numeric", "graded_relative", field="a_conventional_angstrom",
                value=GT_PENDING, tolerance=0.01, unit="angstrom"),
            chk("c3", "numeric", "upper_bound", field="pressure_kbar_abs",
                limit=1.0, unit="kbar"),
            chk("c4", "assertion", "must_match",
                patterns=[r"Ga.*?As.*?(赝势|pseudo)|(赝势|pseudo).*?Ga.*?As"]),
            audit_all("c5"),
        ],
        notes="GT 待首轮冻结（MP 参考 a≈5.75 Å，GBRV+PBE 会略有偏差）。测材料库泛化。",
    ),
    dict(
        id="EXEC-09", title="NaCl 离子晶体 DOS",
        prompt="算 NaCl 的总 DOS 和 Na、Cl 投影 DOS。",
        run_config={"ecutwfc": 40, "ecutrho": 320, "kpoints": [4, 4, 4, 1, 1, 1],
                    "nbnd": 16},
        budget=180,
        checks=[
            chk("c1", "tool_trace", "nodes_succeeded", nodes=["scf", "nscf", "dos", "pdos"]),
            chk("c2", "numeric", "upper_bound", field="dos_at_fermi",
                limit=0.01, unit="states/eV/cell"),
            chk("c3", "assertion", "must_match", patterns=[RE["insulator_nacl"]]),
            chk("c4", "assertion", "must_match", weight=0.5,
                patterns=[r"Cl.*?3p|Cl.*?p.*?(价带|VBM|主导)"]),
            audit_all("c5"),
        ],
        notes="与 EXEC-03 成金属/绝缘体对照。c4 加分项。",
    ),
    dict(
        id="EXEC-10", title="收敛扫描实跑",
        prompt=("用 workflow.convergence 对 Si 做 ecutwfc 收敛扫描，扫 20/30/40/50 Ry，"
                "k 点固定 4x4x4，阈值 1 meV/atom。"),
        run_config={},
        budget=180,
        checks=[
            chk("c1", "tool_trace", "node_count", minimum=4),
            chk("c2", "numeric", "selected_within_threshold",
                field="convergence_selection", threshold_mev_per_atom=1.0),
            chk("c3", "assertion", "reference_is_highest",
                note="参考点须是最高截断点，不能自己跟自己比"),
            chk("c4", "assertion", "must_match", patterns=[RE["energy_not_property"]]),
            audit_all("c5"),
        ],
        notes="收敛扫描真实执行，选点规则可核验。",
    ),
    dict(
        id="EXEC-11", title="MgO 形成能闭环",
        prompt="算 MgO 相对于 Mg 固体和 O2 分子的形成能，三个体系都用一致的方法。",
        run_config={"ecutwfc": 40, "ecutrho": 320, "kpoints": [4, 4, 4, 1, 1, 1]},
        budget=240,
        checks=[
            chk("c1", "tool_trace", "node_count", minimum=3),
            chk("c2", "tool_trace", "energy_steps_bound", tool="thermo.formation"),
            chk("c3", "numeric", "in_range", field="formation_energy_ev_per_atom",
                low=-4.5, high=-1.5, unit="eV/atom"),
            chk("c4", "assertion", "must_match", patterns=[RE["o2_triplet"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["method_consistent"]]),
            chk("c6", "assertion", "must_match", patterns=[RE["not_hull"]]),
            audit_all("c7"),
        ],
        notes="Mg hcp + O2 分子参考。GT 区间宽，首轮后收窄。",
    ),
    dict(
        id="EXEC-12", title="石墨烯单层 + N 掺杂",
        prompt="生成 3x3 石墨烯超胞，把一个 C 换成 N，算它的 DOS。",
        run_config={"ecutwfc": 30, "ecutrho": 240, "kpoints": [4, 4, 1, 0, 0, 0],
                    "nbnd": 40},
        budget=240,
        checks=[
            chk("c1", "tool_trace", "tool_planned", tool="structure.build2d",
                args_contains={"kind": "graphene", "supercell": "3x3"}),
            chk("c2", "numeric", "exact", field="n_atoms", value=18),
            chk("c3", "numeric", "exact", field="n_atoms_N", value=1),
            chk("c4", "numeric", "exact", field="kpoints_kz", value=1),
            chk("c5", "numeric", "lower_bound", field="vacuum_angstrom", limit=15.0),
            chk("c6", "assertion", "must_match", weight=0.5, patterns=[RE["n_type"]]),
            audit_all("c7"),
        ],
        notes="c4：2D 体系垂直方向不采样。3×3 石墨烯 = 18 原子。",
    ),
    dict(
        id="EXEC-13", title="O@MoS2 单位点吸附能",
        prompt=("用 workflow.adsorption_scan 算 O 在 MoS2 单层 top 位的吸附能，3x3 超胞、"
                "O2 做参考、固定晶胞弛豫。"),
        run_config={"ecutwfc": 30, "ecutrho": 240, "kpoints": [2, 2, 1, 0, 0, 0]},
        budget=600,
        checks=[
            chk("c1", "tool_trace", "node_count", minimum=4),
            chk("c2", "tool_trace", "energy_steps_bound", tool="thermo.eads"),
            chk("c3", "numeric", "finite", field="e_ads_ev", unit="eV"),
            chk("c4", "assertion", "must_match", patterns=[RE["o2_triplet"]]),
            chk("c5", "assertion", "must_match", patterns=[RE["fixed_cell_ads"]]),
            chk("c6", "assertion", "must_match",
                patterns=[r"(单|一个|仅).*?位点.*?(不等于|≠|不代表|不一定).*?(最稳|最低|全局)"]),
            audit_all("c7"),
        ],
        notes="本套最重的题（~5 min）。只做 top 一个位点控预算；E_ads 公式核对在 c3 的解析里。",
    ),
]

# --------------------------------------------------------------------------
# L3 EDGE (7)
# --------------------------------------------------------------------------
EDGE = [
    dict(
        id="EDGE-01", title="表面 slab（完全不支持）",
        prompt="算 Pt(111) 表面上 H 原子的吸附能，用 4 层 slab、上面两层放开弛豫。",
        support=["unsupported", "partial"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["unsupported", "partial"]),
            chk("c2", "assertion", "must_match", patterns=[RE["slab_unsupported"]]),
            chk("c3", "tool_trace", "tool_not_planned", tool="structure.build2d"),
            chk("c4", "assertion", "alternative_given"),
            no_side_effects("c5"),
        ],
        notes="全仓无 ase.build.surface / fcc111 / cut 调用。最干净的 unsupported 题。",
    ),
    dict(
        id="EDGE-02", title="分子吸附（明确不支持）",
        prompt="算 CO2 吸附在石墨烯上的吸附能。",
        support=["unsupported", "partial"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["unsupported", "partial"]),
            chk("c2", "assertion", "must_match", patterns=[RE["molecule_ads_unsupported"]]),
            chk("c3", "tool_trace", "no_molecule_split",
                note="不得把 CO2 拆成 C 和 O 分别做单原子吸附"),
            chk("c4", "assertion", "alternative_given"),
            chk("c5", "assertion", "must_match", weight=0.5,
                patterns=[RE["orientation_search"]]),
            no_side_effects("c6"),
        ],
        notes="adsorb.element 正则 ^[A-Z][a-z]?$ 挡多原子；rule_planner.py:168 有专门拒绝语。",
    ),
    dict(
        id="EDGE-03", title="形成能参考态缺口",
        prompt="算 ZnS 相对于 Zn 和 S 参考态的形成能。",
        support=["unsupported", "partial"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["unsupported", "partial"]),
            chk("c2", "assertion", "must_match", patterns=[RE["missing_s_ref"]]),
            chk("c3", "assertion", "no_fabricated_number",
                fields=["formation_energy", "chemical_potential"]),
            chk("c4", "assertion", "alternative_given"),
            chk("c5", "assertion", "must_match", weight=0.5, patterns=[RE["zns_can_scf"]]),
            no_side_effects("c6"),
        ],
        notes="ZnS 在 MATERIAL_DB 但 S 不在 REFERENCE_PHASES 的 33 元素里。考'库里有≠都能算'。",
    ),
    dict(
        id="EDGE-04", title="赝势缺口（原型有但赝势没有）",
        prompt="算 UO2 的结构优化和带隙。",
        support=["unsupported", "partial"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["unsupported", "partial"]),
            chk("c2", "assertion", "must_match", patterns=[RE["missing_u_pseudo"]]),
            chk("c3", "assertion", "no_fabricated_number",
                fields=["band_gap", "lattice_constant"]),
            chk("c4", "assertion", "alternative_given"),
            chk("c5", "assertion", "must_match", weight=0.5, patterns=[RE["dft_plus_u"]]),
            no_side_effects("c6"),
        ],
        notes="_FLUORITE_FORMULAS 有 UO2 原型，但 GBRV 64 元素不含 U。考'能建结构≠能算'。",
    ),
    dict(
        id="EDGE-05", title="化合物 + 单元素化学势模式（语义冲突）",
        prompt="算 MgO 里一个 O 空位的形成能，化学势用 same_method_bulk_per_atom 模式。",
        support=["unsupported", "partial"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["unsupported", "partial"]),
            chk("c2", "assertion", "must_match", patterns=[RE["mu_single_element_only"]]),
            chk("c3", "assertion", "alternative_given"),
            chk("c4", "assertion", "must_match", patterns=[RE["mu_window"]]),
            no_side_effects("c5"),
        ],
        notes="same_method_bulk_per_atom 只对单元素有效；MgO 移除 O 后 E/N 不是 O 的化学势。",
    ),
    dict(
        id="EDGE-06", title="SOC / 非共线磁性（无代码路径）",
        prompt="算 Bi2Se3 的自旋轨道耦合能带，我要看拓扑表面态。",
        support=["unsupported"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["unsupported"]),
            chk("c2", "assertion", "must_match", patterns=[RE["soc_unsupported"]]),
            chk("c3", "assertion", "must_match", patterns=[RE["bi2se3_missing"]]),
            chk("c4", "assertion", "must_not_match",
                patterns=[r"(先|可以)?算.*?标量相对论.*?(能带).*?(即可|作为|替代|满足)"]),
            chk("c5", "assertion", "must_not_match", patterns=[RE["topological_claim"]]),
            no_side_effects("c6"),
        ],
        notes="全仓无 lspinorb/noncolin/LSORBIT；Bi2Se3 也不在库和原型表。双重不支持。",
    ),
    dict(
        id="EDGE-07", title="动作预算超限（不得静默截断）",
        prompt=("我要对 Si、Ge、GaAs、AlAs、InP 五个材料，每个都做结构优化、能带和 DOS，"
                "最后汇总成一张带隙对比表。"),
        support=["partial"],
        checks=[
            chk("c1", "assertion", "support_status_in", allowed=["partial"]),
            chk("c2", "assertion", "flag_set", flag="over_budget"),
            chk("c3", "assertion", "all_entities_present",
                entities=["Si", "Ge", "GaAs", "AlAs", "InP"],
                note="五个材料都要在计划或 advisory 里出现，不得丢材料"),
            chk("c4", "assertion", "no_silent_truncation",
                note="不得只做前 8 个动作而不声明"),
            chk("c5", "assertion", "must_match", patterns=[RE["batch_alternative"]]),
            no_side_effects("c6"),
        ],
        notes="5 材料 × 3 计算 + 汇总 ≥ 16 动作，远超 MAX_PLAN_ACTIONS=8。",
    ),
]


def build(entry, layer):
    q = {
        "schema_version": "1.0",
        "id": entry["id"],
        "layer": layer,
        "title": entry["title"],
    }
    if layer == "EXEC":
        q["prompt"] = entry["prompt"] + " " + EXEC_SUFFIX
        q["run_config"] = {"mode": "research", "approved": True,
                           "accuracy": "custom", **entry.get("run_config", {})}
        q["expect"] = {"execution_mode": "execute"}
        q["budget_seconds"] = entry.get("budget", 300)
    else:
        q["prompt"] = entry["prompt"] + " " + PLAN_SUFFIX
        q["run_config"] = {"mode": "research", "approved": True}
        q["expect"] = {"execution_mode": "plan",
                       "support_status": entry["support"]}
        q["budget_seconds"] = entry.get("budget", 120)
    q["checks"] = entry["checks"]
    q["notes"] = entry.get("notes", "")
    return q


def main():
    QDIR.mkdir(parents=True, exist_ok=True)
    written = []
    for entries, layer in ((PLAN, "PLAN"), (EXEC, "EXEC"), (EDGE, "EDGE")):
        for e in entries:
            q = build(e, layer)
            path = QDIR / f"{q['id']}.json"
            path.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            written.append(q["id"])
    index = {
        "bench_version": "1.0",
        "generated_for": "dft-forge",
        "layers": {"PLAN": len(PLAN), "EXEC": len(EXEC), "EDGE": len(EDGE)},
        "total": len(written),
        "question_ids": written,
    }
    (QDIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} questions to {QDIR}")
    for layer, n in index["layers"].items():
        print(f"  {layer}: {n}")


if __name__ == "__main__":
    main()
