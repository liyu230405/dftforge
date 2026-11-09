# 判分规则

## 1. 题目 JSON schema (v1.0)

```jsonc
{
  "schema_version": "1.0",
  "id": "PLAN-01",
  "layer": "PLAN",                  // PLAN | EXEC | EDGE
  "title": "单模板路由：结构优化",
  "prompt": "...",                  // 原样发给 agent 的字符串
  "run_config": {                   // 注入 AgentLoop ctx.run_config
    "mode": "research",
    "approved": true,
    "ecutwfc": 30, "ecutrho": 240,
    "kpoints": [4, 4, 4, 1, 1, 1]
  },
  "expect": {
    "execution_mode": "plan",       // plan | execute
    "support_status": ["complete"]  // 允许的取值集合
  },
  "checks": [                       // 每项一个断言
    {
      "id": "c1",
      "kind": "tool_trace",         // tool_trace | numeric | assertion | audit
      "weight": 0.25,
      "rule": "template_used",
      "args": {"template_id": "t1_vc_relax"}
    }
  ],
  "budget_seconds": 60,
  "notes": "..."
}
```

## 2. 断言类型

### 2.1 `tool_trace` — 硬断言，0/1 计分

| rule | 含义 |
|---|---|
| `no_side_effect_calls` | 实际执行的工具中 `side_effect != "read_only"` 的数量必须为 0 |
| `tool_planned` | 计划里含指定 tool_id（可给 `args_contains` 约束参数） |
| `tool_not_planned` | 计划里**不得**含指定 tool_id |
| `template_used` | 计划的 `graph.run` 使用指定 `template_id` |
| `nodes_succeeded` | 指定节点名集合全部 `state == "succeeded"` |
| `node_count` | 成功节点总数等于/不少于给定值 |
| `no_self_doping` | 不存在 `structure.dope` 的 `element` 与母体同元素的步骤 |
| `energy_steps_bound` | `thermo.eads` / `thermo.formation` 的能量来源显式绑定真实步骤索引 |

### 2.2 `numeric` — 分级曲线计分

沿用 silicolab bench 的收紧曲线（`_graded_score`）：**`tolerance` 是零分线，不是满分线**。

```
满分   err ≤ 0.3 · tol
零分   err ≥ tol
中间   线性插值
```

比"tol 内一律满分"严约 3.3 倍。`rule` 取值：

| rule | err 定义 |
|---|---|
| `graded_relative` | \|v − t\| / \|t\| |
| `graded_absolute` | \|v − t\| |
| `exact` | v == t，否则 0 |
| `upper_bound` | v ≤ limit → 1，否则按 `(v−limit)/limit` 衰减 |
| `sign_negative` | v < 0 → 1，否则 0（形成能为负等） |

`partial_credit: false` 时退化为硬阈值布尔判定。

### 2.3 `assertion` — 关键词/结构断言

对 agent 的最终中文播报 + `advisory_steps` + `notes` 合并后的文本做判定。**不做语义理解**，只做正则和关键词集合匹配。

| rule | 含义 |
|---|---|
| `must_match` | 给定正则**任一**命中（`all: true` 时要求全部命中） |
| `must_not_match` | 给定正则**全部**不得命中 |
| `support_status_in` | `PlanIR.support_status` ∈ 允许集合 |
| `flag_set` | 指定标记为真（如 `over_budget`） |
| `alternative_given` | advisory 非空且含替代路线关键词（导入/上传/自带/改用/拆分 等） |

正则设计原则：**宁可漏判不可误判**。关键词写宽（如带隙免责用 `PBE.*(低估|underestimat)|不能?.*(等同|直接比较).*实验`），避免因表述变化而误报退化。漏判由人工抽查兜底。

### 2.4 `audit` — 反作弊 gating

L2 题专用。任一 gating flag 为 false → 该题 `total_score` 归零，但保留 `total_score_raw`。

| flag | 判据 | gating |
|---|---|---|
| `non_empty_answer` | 计划动作、工具调用、播报、advisory 至少有一项非空，且无 fatal_error | ✓（全层） |
| `qe_invoked` | 工作目录里发现含 `Program PWSCF`/`PHONON`/`PROJWFC` banner 的输出 | ✓ |
| `job_done` | 每个 QE 日志含 `JOB DONE.`（全文扫描，不只看尾部） | ✓ |
| `qe_wallclock_ok` | Σ QE WALL ≥ 1.0 s 且 Σ QE WALL / 总墙钟 ≥ 0.005 | ✓ |
| `no_fake_backend` | `DFT_FORGE_EXECUTOR` ≠ `fake` | ✓ |
| `plan_only_respected` | L1/L3 题非只读工具调用数 == 0 | ✓（L1/L3） |
| `paths_in_session` | 产物路径都在 session 目录内 | ✗（软） |
| `files_nonempty` | QE 日志 ≥ 1 KB | ✗（软） |

`non_empty_answer` 是必要的：`must_not_match`、`no_fabricated_number`、
`no_side_effect_calls` 这类反向规则在空记录上会全部真空通过，不加这条 gating，
崩溃或超时的题能凭什么都没做拿到 0.2–0.5 的地板分。`harness/selftest.py`
把这一点做成了断言：30 道题在空记录和崩溃记录下必须全部得 0。

## 3. 单题得分

```
题目原始分 = Σ(check_score × weight) / Σ(weight)
题目最终分 = 0 如果 gating 失败，否则 = 原始分
```

权重默认均分；标注"加分项"的 check 权重 0.5，其余 1.0。

## 4. 层级汇总

三层**分别**汇总，不合并总分：

```
L1_score = mean(PLAN-01..10 最终分)
L2_score = mean(EXEC-01..13 最终分)
L3_score = mean(EDGE-01..07 最终分)
```

另外单独报三个计数指标，这些比平均分更早暴露退化：

| 指标 | 定义 | 期望 |
|---|---|---|
| `plan_only_violations` | L1+L3 共 17 题中出现非只读调用的题数 | **必须为 0** |
| `audit_failures` | L2 共 13 题中 gating 失败的题数 | 0 |
| `false_capability_claims` | L3 共 7 题中把不支持包装成可行流程的题数 | **必须为 0** |

这三个是红线指标。任一不为 0，即使平均分上升也判为**本轮退化**。

## 5. 跨轮对比

每轮产出 `runs/<UTC>_<git_sha>/report.json`，含：

```jsonc
{
  "meta": {
    "timestamp": "...", "git_sha": "...",
    "llm_model": "deepseek-flash", "qe_version": "7.5",
    "executor": "local", "bench_version": "1.0"
  },
  "layer_scores": {"L1": 0.0, "L2": 0.0, "L3": 0.0},
  "red_lines": {"plan_only_violations": 0, "audit_failures": 0, "false_capability_claims": 0},
  "per_question": [ /* 每题的 check 明细、提取值、耗时 */ ],
  "regressions": [ /* 相对上一轮分数下降的题 */ ]
}
```

`compare.py` 对比两轮 report，输出：

- 分数上升/下降的题目清单（按降幅排序）
- 新增失败的 check（精确到 check id）
- 红线指标变化

**只看层级平均分会掩盖问题**：典型情况是新增一个 workflow 让 L2 上升 0.1，同时让两道 L3 边界题从正确拒答退化为错误流程。所以 `regressions` 列表和红线指标优先于平均分。

## 6. 稳定性要求

- **L1/L3 同题跑 3 次取多数**。LLM 规划受温度影响，单次结果不足以判断退化。3 次中 ≥2 次通过即记通过，同时记录 `flaky: true`。
- **L2 单次即可**（QE 是确定性的），但首轮需跑 2 次确认数值可复现，再冻结为 GT 基线。
- 每轮固定同一个 LLM 模型和同一台机器；换模型或换机器时，**必须重跑基线**，不能与旧轮次直接比较。

## 7. GT 基线冻结流程

1. 首轮用真实 LLM + 本机 QE 跑完 13 道 L2 题，每题跑 2 次
2. 两次数值一致（在各自容差内）→ 写入题目 JSON 的 `ground_truth`
3. 不一致 → 该题标 `gt_unstable: true`，只做链路断言不做数值断言
4. GT 冻结后记录 `gt_frozen_at`、`gt_qe_version`、`gt_pseudo_sha256`
5. 换赝势或换 QE 版本 → GT 作废，重新冻结
