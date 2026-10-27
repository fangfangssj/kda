---
type: Runbook
title: performance.py 无 hidden-params 通路，这类算子一律报 ascendc ERROR 且拿不到任何加速比
description: 脚本直接以用例 init/inputs 调用候选，没有 load_case_params/hidden_params 支持，因此 process_hashed_hidden_state 类算子必然在计时前抛错，性能腿没有分母。
tags:
- task.debugging
- topic.performance.analysis
status: draft
sources: []
created_at: '2026-09-25T08:59:06Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases: ["CAND-EC-PERF-HIDDEN-PARAMS"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# performance.py 无 hidden-params 通路，这类算子一律报 ascendc ERROR 且拿不到任何加速比

## 证据

`src/scripts/patch_sources/performance.py` 中 `load_case_params` 与 `hidden_params`
出现 **0 次**（在 a5_ops `main@8aada6d69998` 上复验）。脚本直接用用例的 init 参数与输入调用候选，
因此对 `process_hashed_hidden_state` 类算子，候选必然在任何计时发生**之前**抛错，
报告打印 `ascendc  ERROR`、无延迟、**无加速比**。

精度侧已经解决了同一问题：Pass-B 路径在跑用例前先喂该用例的 manifest。
所以这是**性能脚本独有**的缺口。

该 run 之所以仍拿到 median 1.99×，是因为另写了 `perf_hidden_params.py`，复刻该脚本
**自己文档化的** `_fallback` 计时法（`perf_counter` + 每次迭代 `npu.synchronize`、
延迟的算术平均、`OVERALL_SPEEDUP_AGGREGATION = arithmetic_mean`），并在计时前喂 manifest。
**那是缺口的证据，不是修复**：它是同一度量的第二份实现。

上游缺陷单：私有沉淀仓 仓 issue #32（私有仓，不可匿名访问；内容见下文转述）

## 适用范围与边界

- 适用：用例参数不在 init/inputs 里、需要按用例注入的算子（level4_cv 的
  `process_hashed_hidden_state` 类）。
- **与"性能终态未被记录"不同**：a5_ops issue #14 讲的是 producer 发出的状态没被
  `workflow_timing` 记录；本卡在它**上游**——数字根本没产生出来。
- **次生风险**：它把每个算子逼去自写计时脚本，这正是"同一度量两份实现"开始漂移的地方；
  修法应复用既有的参数装载入口，而非新增第二个（该文件已在仓库 BIG_FILE 上限）。
- 实测环境：soc=Ascend950DT_9582、cann=9.1.0，run `142408a6-72e4-45a7-adea-1c83590452ff`。

## 适用场景

- **工程形态**：脚本直接以用例 init/inputs 调用候选，没有 load_case_params/hidden_params 支。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
