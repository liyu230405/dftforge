# DFT-Forge Agent Benchmark v1.0

30 道题，给 DFT-Forge（本仓库）的每一次迭代做效果测试。以下命令都在 `benchmark/` 目录下运行。
每轮只回答一个问题：**比上一轮好了还是坏了，好在哪、坏在哪。**

| 层 | 题数 | 考什么 | 跑 QE |
|---|---:|---|---|
| `PLAN` L1 | 10 | 自然语言 → 选对工具/模板/参数；只规划不执行 | 否 |
| `EXEC` L2 | 13 | QE 真跑起来 + 数值落在容差内 + 反作弊审计 | 是 |
| `EDGE` L3 | 7 | 做不到的事明确说做不到，并给可行替代路线 | 否 |

三层**分别**记分，不合总分。另有三个红线指标必须为 0，任一不为 0 即判本轮退化，
哪怕平均分上升：`plan_only_violations`、`audit_failures`、`false_capability_claims`。

## 目录

```
docs/BENCH_DESIGN.md   为什么这么设计、体系规模上限的实测依据、已知局限
docs/QUESTIONS.md      30 道题的完整说明与能力覆盖矩阵
docs/SCORING.md        题目 schema、断言规则表、打分曲线、GT 冻结流程
harness/gen_questions.py  题目唯一来源，生成 questions/*.json
harness/run.py            跑一轮 + 判分
harness/checks.py         36 条判分规则的实现
harness/extract.py        record → 数值字段的抽取器
harness/scoring.py        分级打分曲线
harness/audit.py          反作弊 gating
harness/effects.py        工具副作用表（与产品对账，漂移会报警）
harness/trace.py          run record schema
harness/compare.py        跨轮退化对比
harness/freeze_gt.py      用两轮一致的实测值冻结 L2 基线
harness/selftest.py       判分器自检（不需要 LLM 和 QE）
questions/                30 个题目 JSON + index.json
results/                  已跑轮次的汇总 report.json（见下文「已有结果」）
runs/<UTC>_<sha>/         每轮的 record.json / score.json / report.json（含 QE 中间文件，不入库）
```

## 用法

改了判分逻辑之后先自检，确认空答案得 0、正确答案能拿满分：

```bash
python3 harness/selftest.py
```

改了题目之后重新生成（`questions/` 是产物，不要手改）：

```bash
python3 harness/gen_questions.py
```

跑规划层，同题 3 次取中位数（LLM 有温度，单次结论不稳）：

```bash
python3 harness/run.py --layer PLAN --repeat 3
```

跑执行层（真调本地 QE，全套约 25 分钟）：

```bash
python3 harness/run.py --layer EXEC
```

跑单题调试：

```bash
python3 harness/run.py --only EXEC-01 --repeat-exec 1
```

改了判分器之后，用已存的 record 重新判分，不重花 QE 时间：

```bash
python3 harness/run.py --score-only runs/2026-09-27T101500Z_a6eac1c
```

对比两轮：

```bash
python3 harness/compare.py runs/<旧轮次> runs/<新轮次>
```

## 首次使用

1. `python3 harness/selftest.py` 应输出 `OK`
2. 跑两轮 L2：`python3 harness/run.py --layer EXEC --round gt1` 和 `--round gt2`
3. 试运行冻结：`python3 harness/freeze_gt.py runs/gt1_* runs/gt2_*`
4. 确认无误后 `--apply` 写回 `questions/EXEC-*.json` 的 `ground_truth`

EXEC-08（GaAs 晶格常数）的目标值目前是 `null`，等首轮冻结填入。

## 环境

`run.py` 只对未设置的变量填默认值，已有的环境变量优先：

| 变量 | 默认 |
|---|---|
| `DFT_FORGE_REPO` | `benchmark/` 的上一级，即本仓库根目录 |
| `DFT_FORGE_LLM_PROVIDER` / `_MODEL` | `openai` / `deepseek-flash` |
| `DFT_FORGE_EXECUTOR` | `local`（`fake` 会被 gating 判为作弊） |
| `DFT_FORGE_QE_BIN` | `/opt/homebrew/bin` |
| `DFT_FORGE_WALLTIME` | `600` |

API key 从 `DFT_FORGE_LLM_API_KEY` 读，没有就用 `getpass` 交互输入，只留在内存。
落盘前所有记录都过一遍脱敏，key 不会进 `runs/`。

## 已有结果

`results/` 里是 2026-10-08 在 `a6eac1c` 上跑的三轮汇总（deepseek-flash + 本地 QE 7.5）。
完整 record 和 QE 中间文件约 1.9 GB，不入库。

| 轮次 | 范围 | L2 得分 | 红线 |
|---|---|---:|---|
| `gt1` | EXEC 13 题 | 0.7976 | 全为 0 |
| `gt2` | EXEC 13 题 | 0.8188 | 全为 0 |
| `fix1` | EXEC 2 题复测 | 0.8409 | 全为 0 |

只跑了执行层，PLAN 和 EDGE 两层还没有基线；L2 的 ground truth 尚未冻结（EXEC-08 仍为 `null`）。
`gt1` 的 `llm_model` 字段记成了 `unknown`，与 `gt2` 是同一配置。

## 比较的前提

同一个 LLM 模型、同一台机器、同一个 QE 版本和赝势。换任何一个，
`compare.py` 会提示两轮不可直接比较，需要重跑基线。

L2 的数值 GT 是**本机实测基线，不是文献值**，比的是"和基线一致"而不是"和实验一致"。
这些参数（如 `ecutwfc=30`）未经收敛测试，容差按链路可复现给，不按物理正确给。
