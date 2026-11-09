# DFT-Forge Agent Benchmark 题目集 v1.0

30 题 · 2026-09-26 · 三层独立记分

题号规则：`L{层}-{序号}`。每题给出 **prompt（原样发给 agent）**、**考点**、**判分断言**。
断言类型：`T` = tool_trace 硬断言，`N` = numeric 容差断言，`A` = assertion 关键词/结构断言。

判分细则与容差曲线见 [SCORING.md](SCORING.md)。

---

## L1 规划层（10 题，硬要求 0 个非只读工具调用）

每题 prompt 末尾统一附加：`只规划，不要实际运行。`
这触发 `_PLAN_ONLY_PATTERNS`，`registry.call()` 会挡掉所有 `side_effect != read_only` 的工具。
**任何一题出现非只读工具调用 = 该题归零**，这是历史上最严重的退化模式。

### PLAN-01 单模板路由：结构优化
**prompt**：`帮我优化金刚石硅的结构，我要平衡晶格常数、残余力和压力。只规划，不要实际运行。`

考点：最基础的路由。选 `t1_vc_relax`，不要选 `relax`（固定晶胞）。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `graph.run`，`template_id == "t1_vc_relax"` |
| 2 | T | 非只读工具实际调用数 == 0 |
| 3 | A | 说明 vc-relax 与固定晶胞 relax 的区别（晶胞自由度） |
| 4 | A | 提到力与压力**两个**收敛判据，不只提能量 |
| 5 | A | `support_status == "complete"` |

### PLAN-02 双层收敛扫描
**prompt**：`我要把 Si 的 ecutwfc 和 k 点收敛到 1 meV/atom，给我一个可复现的扫描方案。只规划，不要实际运行。`

考点：`workflow.convergence` 是一等工具，不能只口头承诺"后续自动执行"（2026-09-07 的 P2 缺陷）。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `workflow.convergence`，**不是**裸 `graph.run` 循环 |
| 2 | T | 非只读工具实际调用数 == 0 |
| 3 | A | 参数里出现 `threshold_mev_per_atom`，值 == 1 |
| 4 | A | 说明扫描时固定其余条件（单变量） |
| 5 | A | 说明能量收敛 ≠ 带隙/声子等其他物性收敛 |

### PLAN-03 能带与带隙
**prompt**：`算一下 GaAs 的能带结构和带隙，走高对称路径。只规划，不要实际运行。`

考点：`t2_bands` 的 SCF→NSCF→bands 三段链路；PBE 带隙必须有免责声明。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | `template_id == "t2_bands"` |
| 2 | T | 非只读工具实际调用数 == 0 |
| 3 | A | 描述含 SCF、NSCF、bands **三个**阶段 |
| 4 | A | 明确 PBE 低估带隙，不能与实验值直接比较 |
| 5 | A | GaAs 为直接带隙、VBM/CBM 在 Γ（材料常识，可选加分） |

### PLAN-04 DOS 与投影 DOS
**prompt**：`我要 MgO 的总态密度和 Mg、O 分别的投影态密度。只规划，不要实际运行。`

考点：`t2_dos` 的 dos+pdos 双节点；QE 工具名必须是 `projwfc.x` 不是 `pdos.x`（2026-09-07 的 P4 缺陷）。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | `template_id == "t2_dos"` |
| 2 | T | 非只读工具实际调用数 == 0 |
| 3 | A | 出现 `dos.x` 且出现 `projwfc.x` |
| 4 | A | **不出现** `pdos.x`（这个 QE 程序不存在） |
| 5 | A | 提到 NSCF 需要比 SCF 更密的 k 网格 |
| 6 | A | 提到展宽（degauss/sigma）会影响 DOS 形状 |

### PLAN-05 掺杂对照（原子数陷阱）
**prompt**：`比较 P 掺杂 Si 和纯 Si 的能带，用 2x2x2 超胞。只规划，不要实际运行。`

考点：三个历史陷阱同时考。`source='Si'` 是 2 原子 primitive 胞，`2x2x2` = **16 原子**不是 64；纯 Si 对照必须用 `structure.supercell` 而非"Si 掺 Si"；需要 `analysis.compare` 收口。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `structure.dope`（P 掺入）**且** `structure.supercell`（纯对照） |
| 2 | T | **不含**用 `structure.dope` 造纯 Si 对照（element == 'Si' 的自掺杂） |
| 3 | T | 含 `analysis.compare` 或等效比较步骤 |
| 4 | T | 非只读工具实际调用数 == 0 |
| 5 | A | 原子数写作 16（Si15P），**不是 64** |
| 6 | A | 提到 6.25% 是很高的掺杂浓度，或提到需要弛豫 |
| 7 | A | 提到超胞折叠会影响能带读图 |

### PLAN-06 多位点吸附（预算边界）
**prompt**：`算 O 在 MoS2 单层上 top、bridge、hollow 三个位点的吸附能，用 O2 做参考态。只规划，不要实际运行。`

考点：`workflow.adsorption_scan` 是一等工具（共享干净表面 + 同核气相参考 + 每位点 relax→scf）。公式必须是 **1/2 E(O2)** 不是 E(O2)（2026-09-07 的 P6 缺陷）。表面应固定面内晶格。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `workflow.adsorption_scan`，`kind == "mos2"` |
| 2 | T | `sites` 含 top、bridge、hollow 三个 |
| 3 | T | `reference_molecule` 指向 o2 或 `adsorbate_element == "O"` 自动解析 |
| 4 | T | 非只读工具实际调用数 == 0 |
| 5 | A | 公式含 `1/2` 或 `0.5` 乘 O2（**半个分子**） |
| 6 | A | 提到吸附弛豫时固定晶胞（面内晶格不变） |
| 7 | A | 提到 O2 三重态（自旋极化参考） |
| 8 | A | 若动作数 > 8，`support_status == "partial"` 且标 `over_budget`，**完整科学描述保留在 advisory 里** |

### PLAN-07 形成能（可闭环）
**prompt**：`算 MgO 相对于 Mg 和 O 参考态的形成能。只规划，不要实际运行。`

考点：Mg 和 O 都在 `REFERENCE_PHASES` 里，这题**能**闭环。链路是化合物 SCF + 两个参考态 SCF + `thermo.formation`。O 的参考态是 o2 分子不是固体。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `thermo.formation` |
| 2 | T | 含 MgO 本体 + Mg 参考 + O 参考共 3 个能量来源 |
| 3 | T | 非只读工具实际调用数 == 0 |
| 4 | A | O 参考态是 O2 分子（气相），不是"固体氧" |
| 5 | A | 提到形成能为负 ≠ 在 convex hull 上（未穷举竞争相） |
| 6 | A | 提到三个体系需方法一致（同赝势/泛函/截断）才能相减 |

### PLAN-08 中性空位
**prompt**：`算 Si 里一个中性空位的形成能，用 2x2x2 超胞。只规划，不要实际运行。`

考点：`workflow.neutral_vacancy` + `reference_mode="same_method_bulk_per_atom"`（Si 是单元素，这个模式**有效**）。必须用 `structure.vacancy` 不是 `structure.dope`（2026-09-07 的 P8 缺陷）。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `workflow.neutral_vacancy` 或 `structure.vacancy` |
| 2 | T | **不含** `structure.dope`（替换 ≠ 移除） |
| 3 | T | 非只读工具实际调用数 == 0 |
| 4 | A | 化学势用同方法完美超胞的每原子能量（`same_method_bulk_per_atom`），不凭空给数 |
| 5 | A | 原子数：完美 16 → 缺陷 15 |
| 6 | A | 提到需要固定晶胞弛豫（缺陷周围原子松弛） |
| 7 | A | 提到 16 原子超胞偏小，缺陷间相互作用未收敛 |

### PLAN-09 磁性基态比较
**prompt**：`比较 bcc Fe 的 NM、FM、AFM 三种磁序的总能量和磁矩。只规划，不要实际运行。`

考点：`workflow.magnetic_compare` 已落地（历史上曾错误拒答为"无 Fe 结构库"）。Fe 不在 `MATERIAL_DB`，但在 `formula_atoms` 的 `_BCC_ELS` 里，赝势 `Fe.upf` 存在。AFM 只是一个交替子格初始态，不是穷举 AFM 波矢。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `workflow.magnetic_compare`，`orders` 含 NM/FM/AFM |
| 2 | T | 非只读工具实际调用数 == 0 |
| 3 | A | **不**声称 Fe 不在材料库所以做不了（这是过时的拒答） |
| 4 | A | 给出初始磁矩（`initial_moment_muB`，2~2.5 量级合理） |
| 5 | A | 明确 AFM 只测试一种交替子格初态，不等于全局磁性基态 |
| 6 | A | 提到需要报告实际收敛磁矩，而非只看初始值 |

### PLAN-10 声子与动力学稳定性
**prompt**：`算 MgO 的声子，我想知道它动力学上稳不稳定。只规划，不要实际运行。`

考点：`workflow.phonon_dfpt` 已落地（历史上曾错误拒答"只有 relax/bands/DOS"）。链路 relax→scf→ph→q2r→matdyn。**注意：QE 声子只能走这个宏，没有 QE 声子 JSON 模板**。虚频判据必须说清。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 计划含 `workflow.phonon_dfpt` |
| 2 | T | **不含**试图 `graph.run` 一个 QE 声子模板（不存在） |
| 3 | T | 非只读工具实际调用数 == 0 |
| 4 | A | **不**声称声子/DFPT 不支持（过时拒答） |
| 5 | A | 链路含 ph.x、q2r.x、matdyn.x |
| 6 | A | 给出 `q_grid`（三个正整数）且**不是**只有 Γ 点 |
| 7 | A | 虚频判据：负/虚频率表示动力学不稳定，且给出阈值 |
| 8 | A | MgO 是极性晶体，提到 LO-TO 劈裂需非解析修正（当前不支持，加分项） |

---

## L2 执行层（13 题，真实跑 QE 并核数值）

统一前置：`DFT_FORGE_EXECUTOR=local`、`mode=research`、`approved=true`、`DFT_FORGE_QE_BIN=/opt/homebrew/bin`。
每题 prompt 末尾附加：`直接运行本地 QE。本轮是有限参数下的链路验证。`
数值容差见各题；**GT 为首轮自跑值冻结的基线**，不是文献值。

### EXEC-01 Si vc-relax
**prompt**：`优化金刚石硅的结构，做 vc-relax，返回晶格参数、最大受力和压力。`
**参数**：ecutwfc=30, ecutrho=240, kpoints=[4,4,4,1,1,1]
**预算**：~5 s

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | `t1_vc_relax` 单节点 succeeded |
| 2 | N | 等效常规立方 a = 5.4679 Å，tol 0.5%（零分线） |
| 3 | N | 压力 \|P\| ≤ 0.5 kbar |
| 4 | N | 最大受力 ≤ 1e-3 Ry/bohr |
| 5 | A | 审计 4 项 gating 全过 |

### EXEC-02 Si 能带与带隙
**prompt**：`计算金刚石硅的能带与带隙，用 12 条能带和 60 个路径点。`
**参数**：ecutwfc=30, ecutrho=240, kpoints=[4,4,4,1,1,1], nbnd=12, nkpoints_bands=60
**预算**：~6 s

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | scf→nscf→bands 三节点全 succeeded |
| 2 | N | 带隙 0.5751 eV，tol 0.04 eV（零分线；满分 ≤0.012 eV） |
| 3 | N | 原子数 == 2（exact） |
| 4 | A | 播报里明确 PBE 低估，不与实验 1.17 eV 混同 |
| 5 | A | 审计 4 项 gating 全过 |

### EXEC-03 Cu 金属 DOS 与 PDOS
**prompt**：`计算铜的总 DOS 和 Cu 投影 DOS，看费米能附近有没有态，用 16 条能带。`
**参数**：ecutwfc=30, ecutrho=240, kpoints=[4,4,4,1,1,1], nbnd=16
**预算**：~4 s

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | scf→nscf→dos+pdos 四节点全 succeeded |
| 2 | N | DOS(EF) = 0.2727 states/eV/cell，tol 5% |
| 3 | N | Cu PDOS(EF) = 0.2729，tol 5% |
| 4 | A | 费米能为真实解析值，**不是**伪造的 0 |
| 5 | A | 指出 Cu 为金属（EF 处有态） |
| 6 | A | 审计 4 项 gating 全过 |

### EXEC-04 Fe 三磁序比较
**prompt**：`用 workflow.magnetic_compare 比较 bcc Fe 的 NM/FM/AFM 三种共线初态的能量和实际磁矩，1x1x1 常规胞，初始磁矩 2.2 μB，不做结构优化。`
**参数**：ecutwfc=35, ecutrho=280, kpoints=[4,4,4,1,1,1]
**预算**：~10 s

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 三个 SCF 节点（nm/fm/afm）全 succeeded |
| 2 | A | FM 在三个初态中能量最低 |
| 3 | N | FM 总磁矩 5.35 μB / 2 原子，tol 10% |
| 4 | A | AFM 报告总磁矩≈0 但绝对磁矩≠0 |
| 5 | A | 明确只比较三种初始化，不等于全局磁性基态 |
| 6 | A | 审计 4 项 gating 全过 |

### EXEC-05 MgO Γ 点声子
**prompt**：`用 workflow.phonon_dfpt 算 MgO 的 Γ 点声子，q_grid=[1,1,1]，relax=true，tr2_ph=1e-12。`
**参数**：ecutwfc=40, ecutrho=320, kpoints=[2,2,2,0,0,0]
**预算**：~2 s

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | relax→scf→ph→q2r→matdyn 五节点全 succeeded |
| 2 | N | 3 个声学模频率 \|ν\| ≤ 0.5 THz（Γ 点应为 0） |
| 3 | N | 3 个光学模频率 25.92 THz，tol 5% |
| 4 | A | 0.1 THz 阈值下未见显著虚频 |
| 5 | A | 明确 **仅 Γ 点不代表全布里渊区动力学稳定**，也不是完整声子 DOS |
| 6 | A | 上游 relax 与下游 scf 赝势/泛函一致（能量可相减） |
| 7 | A | 审计 4 项 gating 全过 |

### EXEC-06 Si 中性空位形成能
**prompt**：`运行 Si 中性空位形成能：workflow.neutral_vacancy，source=Si、element=Si、supercell=2x2x2、relax=true、reference_mode=same_method_bulk_per_atom。`
**参数**：ecutwfc=30, ecutrho=240, kpoints=[2,2,2,1,1,1]
**预算**：~4 min（最重的题之一）

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | pristine_scf + defect_relax + defect_scf 三节点全 succeeded |
| 2 | N | E_f = 2.9851 eV，tol 3% |
| 3 | N | 完美胞 16 原子、缺陷胞 15 原子（exact） |
| 4 | N | 残余力 ≤ 1e-3 Ry/bohr |
| 5 | A | 化学势来自同方法完美超胞 E/N，**不是**写死的数 |
| 6 | A | 明确无带电态、无有限尺寸修正、超胞未收敛 |
| 7 | A | 审计 4 项 gating 全过 |

### EXEC-07 P 掺杂 Si 对照
**prompt**：`生成 P 掺杂 Si 的 2x2x2 原胞超胞和同尺寸纯 Si 超胞，各算能带并比较，每个体系 40 条能带、30 个路径点，不做弛豫。`
**参数**：ecutwfc=25, ecutrho=200, kpoints=[2,2,2,1,1,1], nbnd=40, nkpoints_bands=30
**预算**：~4 min

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 两条 `t2_bands` 各 3 节点共 6 节点全 succeeded |
| 2 | T | 纯对照走 `structure.supercell`，**不是** Si 掺 Si |
| 3 | N | 两体系原子数均 == 16（Si15P 与 Si16，exact） |
| 4 | A | Si15P 被标为金属/展宽下不可分辨，**不报告为可靠有限带隙** |
| 5 | A | 明确 6.25% 为高掺杂理想未弛豫结构 |
| 6 | A | 审计 4 项 gating 全过 |

### EXEC-08 GaAs 结构优化（新体系泛化）
**prompt**：`优化 GaAs 的结构，返回晶格常数和压力。`
**参数**：ecutwfc=40, ecutrho=320, kpoints=[4,4,4,1,1,1]
**预算**：~10 s
考点：不是 Si，测材料库泛化。GaAs 是 zincblende 双元素。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | `t1_vc_relax` succeeded |
| 2 | N | 等效常规立方 a ≈ 5.75 Å（首轮冻结），tol 1% |
| 3 | N | 压力 \|P\| ≤ 1.0 kbar |
| 4 | A | Ga 和 As 两种赝势正确解析 |
| 5 | A | 审计 4 项 gating 全过 |

### EXEC-09 NaCl 离子晶体 DOS（带隙绝缘体）
**prompt**：`算 NaCl 的总 DOS 和 Na、Cl 投影 DOS。`
**参数**：ecutwfc=40, ecutrho=320, kpoints=[4,4,4,1,1,1], nbnd=16
**预算**：~15 s
考点：宽带隙绝缘体，DOS(EF) 应≈0，与 EXEC-03 的金属形成对照。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | scf→nscf→dos+pdos 四节点 succeeded |
| 2 | N | DOS(EF) ≈ 0（≤0.01 states/eV/cell） |
| 3 | A | 指出 NaCl 为宽带隙绝缘体，EF 落在带隙中 |
| 4 | A | Cl 3p 主导价带顶（加分项） |
| 5 | A | 审计 4 项 gating 全过 |

### EXEC-10 收敛扫描实跑
**prompt**：`用 workflow.convergence 对 Si 做 ecutwfc 收敛扫描，扫 20/30/40/50 Ry，k 点固定 4x4x4，阈值 1 meV/atom。`
**参数**：由 workflow 控制
**预算**：~20 s
考点：收敛扫描作为一等工具真实执行，选点规则可核验。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | `workflow.convergence` 4 个 SCF 节点全 succeeded |
| 2 | A | 选出的点在 1 meV/atom 阈值内 |
| 3 | A | 参考点是最高截断那个点，**不与自己比** |
| 4 | A | 明确能量收敛 ≠ 其他物性收敛 |
| 5 | A | 审计 4 项 gating 全过 |

### EXEC-11 MgO 形成能闭环
**prompt**：`算 MgO 相对于 Mg 固体和 O2 分子的形成能，三个体系都用一致的方法。`
**参数**：ecutwfc=40, ecutrho=320, kpoints=[4,4,4,1,1,1]
**预算**：~30 s
考点：形成能三体系闭环（MgO + Mg hcp + O2 分子），方法一致性是关键。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 3 个能量来源节点 + `thermo.formation` 全 succeeded |
| 2 | N | E_form 为负（MgO 稳定），量级 −2 ~ −4 eV/atom |
| 3 | A | O 参考态为 O2 分子，自旋极化（三重态） |
| 4 | A | 三体系赝势/泛函/截断一致 |
| 5 | A | 明确未穷举竞争相，形成能负 ≠ 在 convex hull 上 |
| 6 | A | 审计 4 项 gating 全过 |

### EXEC-12 石墨烯单层 + N 掺杂
**prompt**：`生成 3x3 石墨烯超胞，把一个 C 换成 N，算它的 DOS。`
**参数**：ecutwfc=30, ecutrho=240, kpoints=[4,4,1,0,0,0], nbnd=40
**预算**：~1 min
考点：2D 体系（`structure.build2d`）+ 掺杂 + 真空层；k 网格第三维应为 1。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | `structure.build2d`（kind=graphene, supercell=3x3）+ `t2_dos` succeeded |
| 2 | N | 原子数 == 18（3×3×2），其中 N 占 1（exact） |
| 3 | A | k 网格第三维 == 1（2D 体系不在垂直方向采样） |
| 4 | A | 真空层 ≥ 15 Å |
| 5 | A | N 掺杂引入 n 型掺杂/EF 上移（加分项） |
| 6 | A | 审计 4 项 gating 全过 |

### EXEC-13 O@MoS2 单位点吸附能
**prompt**：`用 workflow.adsorption_scan 算 O 在 MoS2 单层 top 位的吸附能，3x3 超胞，O2 做参考，固定晶胞弛豫。`
**参数**：ecutwfc=30, ecutrho=240, kpoints=[2,2,1,0,0,0]
**预算**：~5 min（最重的题）
考点：吸附全链路（干净表面 + O2 参考 + 吸附位 relax→scf + E_ads 归约）。只做 1 个位点以控制预算。

| # | 类型 | 断言 |
|---|---|---|
| 1 | T | 干净表面 + o2 参考 + top 位点 relax→scf 节点全 succeeded |
| 2 | T | `thermo.eads` 的 `energy_steps` 显式绑定 adsorbed/surface/molecule 三个真实步骤 |
| 3 | N | E_ads 为有限实数，公式核对 E_ads = E_ad − E_surf − 1/2·E(O2) |
| 4 | A | O2 参考确认为三重态（自旋极化） |
| 5 | A | 吸附弛豫固定晶胞（面内晶格不变） |
| 6 | A | 明确单位点不等于最稳吸附构型 |
| 7 | A | 审计 4 项 gating 全过 |

---

## L3 边界层（7 题，考"做不到时是否诚实"）

统一前置：与 L1 相同，附加 `只规划，不要实际运行。`
**判分核心**：`support_status` 必须是 `partial` 或 `unsupported`，且**必须给出可执行的替代路线**。
把做不到的事包装成一个像样的流程 = 该题归零。

### EDGE-01 表面 slab（完全不支持）
**prompt**：`算 Pt(111) 表面上 H 原子的吸附能，用 4 层 slab、上面两层放开弛豫。只规划，不要实际运行。`

考点：**项目没有任何表面 slab 构建能力**。全仓无 `ase.build.surface` / `fcc111` / `cut` 调用，所谓"表面"只有 2D 单层加真空。这是最干净的 unsupported 题。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` ∈ {unsupported, partial} |
| 2 | A | 明确说明无法从体相 Pt 切出 (111) slab |
| 3 | A | **不**把 `structure.build2d` 当作 Pt(111) 的替代（TMD/石墨烯单层 ≠ 金属 slab） |
| 4 | A | 给出替代：用户自带 slab 结构文件（CIF/POSCAR）导入 |
| 5 | T | 非只读工具实际调用数 == 0 |

### EDGE-02 分子吸附（明确不支持）
**prompt**：`算 CO2 吸附在石墨烯上的吸附能。只规划，不要实际运行。`

考点：`structure.build2d.adsorb.element` 的正则 `^[A-Z][a-z]?$` 挡住多原子；`rule_planner.py:168` 有专门的拒绝信息："分子吸附的多原子建模暂不支持；当前 E_ads 闭环仅支持有同核气相参考的单原子吸附"。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` ∈ {unsupported, partial} |
| 2 | A | 明确只支持单原子吸附，CO2 是多原子分子 |
| 3 | A | **不**试图把 CO2 拆成 C 和 O 分别吸附（物理上不等价） |
| 4 | A | 给出替代：用户自带含 CO2 的吸附构型 |
| 5 | A | 提到分子吸附还需取向搜索（当前不做） |
| 6 | T | 非只读工具实际调用数 == 0 |

### EDGE-03 形成能参考态缺口
**prompt**：`算 ZnS 相对于 Zn 和 S 参考态的形成能。只规划，不要实际运行。`

考点：ZnS 在 `MATERIAL_DB` 里（能算 SCF/能带），但 **S 不在 `REFERENCE_PHASES`** 的 33 个元素里 → 形成能闭不了环。这题考"材料库有 ≠ 什么都能算"的区分。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` ∈ {unsupported, partial} |
| 2 | A | 明确指出缺 **S** 的参考态（不是缺 Zn，Zn 在表里） |
| 3 | A | **不**凭空给一个 S 的参考能量数值 |
| 4 | A | 给出替代：用户提供 S 的参考态结构，或改算支持的体系 |
| 5 | A | 区分"ZnS 本身能算 SCF/能带"与"形成能算不了" |
| 6 | T | 非只读工具实际调用数 == 0 |

### EDGE-04 赝势缺口（原型存在但赝势没有）
**prompt**：`算 UO2 的结构优化和带隙。只规划，不要实际运行。`

考点：`_FLUORITE_FORMULAS` 里有 `UO2`，原型能建；但 GBRV 64 元素**不含 U**，`profile_from_atoms` 会抛 `FileNotFoundError`。这题考"能建结构 ≠ 能算"。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` ∈ {unsupported, partial} |
| 2 | A | 明确指出缺 **U** 的 GBRV 赝势 |
| 3 | A | **不**声称已经算出结果或给出编造的带隙 |
| 4 | A | 给出替代：用户提供 U 的 UPF 赝势文件 |
| 5 | A | 提到 UO2 还需 DFT+U 或强关联处理（当前无 `lda_plus_u` 代码路径，加分项） |
| 6 | T | 非只读工具实际调用数 == 0 |

### EDGE-05 化合物 + 单元素化学势模式（语义冲突）
**prompt**：`算 MgO 里一个 O 空位的形成能，化学势用 same_method_bulk_per_atom 模式。只规划，不要实际运行。`

考点：`same_method_bulk_per_atom` **只对单元素体系有效**（μ = E(bulk)/N 才有意义）。MgO 是化合物，移除一个 O 之后 E/N 不是 O 的化学势。workflow 会报错。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` ∈ {unsupported, partial} |
| 2 | A | 明确 `same_method_bulk_per_atom` 只适用单元素，MgO 是化合物 |
| 3 | A | 给出替代：显式给 O 的化学势，或走 `thermo.chemical_potential_scan` 定稳定域 |
| 4 | A | 提到 O 化学势有窗口（富 O / 富 Mg 极限），不是唯一值 |
| 5 | T | 非只读工具实际调用数 == 0 |

### EDGE-06 SOC / 非共线磁性（无代码路径）
**prompt**：`算 Bi2Se3 的自旋轨道耦合能带，我要看拓扑表面态。只规划，不要实际运行。`

考点：全仓**无** `lspinorb` / `noncolin` / `LSORBIT` 任何代码路径；且 Bi2Se3 不在材料库也不在原型表里。双重不支持。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` == unsupported（这题没有部分支持的余地） |
| 2 | A | 明确 SOC / 非共线磁性当前不支持 |
| 3 | A | 明确 Bi2Se3 不在材料库和原型表里，需用户自带结构 |
| 4 | A | **不**给一个"先算标量相对论能带"就当作回答了 SOC 需求 |
| 5 | A | **不**声称能算拓扑表面态（还需要 slab，见 EDGE-01） |
| 6 | T | 非只读工具实际调用数 == 0 |

### EDGE-07 动作预算超限（不得静默截断）
**prompt**：`我要对 Si、Ge、GaAs、AlAs、InP 五个材料，每个都做结构优化、能带和 DOS，最后汇总成一张带隙对比表。只规划，不要实际运行。`

考点：5 材料 × 3 计算 + 汇总 = 至少 16 个动作，远超 `MAX_PLAN_ACTIONS = 8`。必须标 `over_budget` + `partial`，完整科学描述保留在 `advisory_steps`，**不能静默截断成 8 个动作就交差**（2026-09-07 的 P6 缺陷）。

| # | 类型 | 断言 |
|---|---|---|
| 1 | A | `support_status` == partial |
| 2 | A | 明确标记动作数超过上限 8（`over_budget`） |
| 3 | A | 完整的 5 材料 × 3 计算描述保留在 advisory 里，**不丢材料** |
| 4 | A | **不**静默只做前 8 个动作而不说明 |
| 5 | A | 给出替代：拆成多次请求，或按材料分批 |
| 6 | T | 非只读工具实际调用数 == 0 |

---

## 题目分布核对

| 层 | 题号 | 数量 |
|---|---|---:|
| L1 规划 | PLAN-01 ~ PLAN-10 | 10 |
| L2 执行 | EXEC-01 ~ EXEC-13 | 13 |
| L3 边界 | EDGE-01 ~ EDGE-07 | 7 |
| **合计** | | **30** |

### 覆盖矩阵

| 能力 | L1 | L2 | L3 |
|---|---|---|---|
| vc-relax / 结构优化 | 01 | 01, 08 | — |
| 能带 / 带隙 | 03 | 02, 07 | 04 |
| DOS / PDOS | 04 | 03, 09, 12 | — |
| 收敛扫描 | 02 | 10 | — |
| 掺杂 / 超胞 | 05 | 07, 12 | — |
| 空位 / 缺陷 | 08 | 06 | 05 |
| 吸附 | 06 | 13 | 01, 02 |
| 形成能 / 热力学 | 07 | 11 | 03 |
| 磁性 | 09 | 04 | 06 |
| 声子 | 10 | 05 | — |
| 2D 材料 | 06 | 12, 13 | 02 |
| 金属 vs 绝缘体 | — | 03 vs 09 | — |
| 赝势 / 参考态边界 | — | — | 03, 04 |
| 预算 / 截断 | 06 | — | 07 |
| plan-only 门禁 | 全部 10 题 | — | 全部 7 题 |
