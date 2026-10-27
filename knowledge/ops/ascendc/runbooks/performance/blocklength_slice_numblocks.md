---
type: "Runbook"
title: "blockLength 对齐 slice 致 numBlocks 塌缩单核"
description: "SwiGlu host tiling 把 blockLength 对齐到 slice 边界后，当 totalSlices < coreNum（极端 totalSlices=1）时 numBlocks=min(coreNum,totalSlices/minSlicesPerCore) 塌缩为 1，整 block 串行单核，大 shape case 退化约一个数量级；判别方法：对齐版大 slice case 成对耗时远高于不对齐版即命中。"
tags: ["task.debugging", "topic.tiling", "topic.performance.analysis", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T15:58:27Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# blockLength 对齐 slice 致 numBlocks 塌缩单核

## 现象

在 Ascend910B3（DAV_2201，a2，CANN 9.2.0）上为 SwiGlu 支持跨 slice 批量路径，把 host tiling 的 `blockLength` 对齐到 `slice` 边界（`blockLength = slicesPerBlock * slice`）。本意是保证每 block 起点落在 slice 边界以便批量路径正确，但引入了隐性退化：对齐版 cann-bench 评测中大 slice case 串行单核、成对耗时远高于不对齐版——case 4（fp16, shape `[8192,16384]`, dim=0）慢约 25×，case 6（bf16）/case 8（fp32）同样退化约一个数量级。

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke），host tiling 函数计算 `blockLength`/`numBlocks`/`slicesPerBlock`。
- **API 路径**：host `GetBlockNum()`/tiling 参数 `blockLength`；kernel `GetBlockIdx()` 多核切分。
- **验证平台与 CANN 版本**：实测 Ascend910B3 / DAV_2201（a2），CANN 9.2.0；frontmatter `platforms: [a2]` 严格对应此实测。
- **适用粒度**：`operator_family`——任何 host tiling 把 `blockLength` 对齐到 `slice` 的算子。
- **路线优先级**：`banned`（`blockLength` 无条件对齐到 slice 的策略在 `totalSlices < coreNum` 时 banned）；条件对齐（`totalSlices < coreNum` 时不对齐保持多核，否则对齐支持批量路径）属 `safety_fallback`。
- **不适用/未验证场景**：`totalSlices >= coreNum` 时对齐不会塌缩 numBlocks，不触发本退化。

## 根因

`numBlocks = min(coreNum, totalSlices / minSlicesPerCore)`；`slicesPerBlock = ceil(totalSlices / numBlocks)`；对齐口径下 `blockLength = slicesPerBlock * slice`。当某 case 的 `totalSlices` 很小（case 4 在对齐口径下 `totalOut = slice` → `totalSlices = 1`），`numBlocks` 塌缩到 1，整 block 只分给 1 个核，失去多核并行。不对齐版 v1 用 `numBlocks = min(coreNum, totalOut / MIN_ELEMS_PER_CORE)` 跑满 24 核，故对齐后慢约 25×（24 核 → 1 核）。case 6/8 的 slice 同样等于 totalOut（`totalSlices = 1`），同一机制退化约一个数量级。

机制链：blockLength 对齐 slice → totalSlices < coreNum 的 case 上可用 slice 数不够分给所有核 → numBlocks 塌缩为 1 → 整 block 串行单核 → 大 shape case 退化一个数量级。这不是"算力不够"，而是"对齐口径把多核并行度压没"。

## 判别方法

- **触发（tool_observed）**：host tiling 把 `blockLength` 对齐到 `slice` 边界后，cann-bench 评测中大 slice case（`totalSlices < coreNum`，典型 `totalSlices=1`）成对耗时远高于不对齐版（本 trace case 4 慢约 25×，case 6/8 退化约一个数量级）。
- **探针（executable check）**：跑 cann-bench 评测，比较大 slice case 在"对齐 blockLength"与"不对齐 blockLength"两版下的成对相对耗时：
  ```bash
  cd <cann-bench-root> && bash scripts/run_evaluation.sh \
    --source-dir <source-dir> --operator SwiGlu --device-id 0 2>&1 | \
    grep -E "case_4|case_6|case_8|综合得分"
  ```
  若大 slice case 在对齐版成对远慢于不对齐版（退化一个数量级以上），即命中。判别时须同时核验对齐口径下该 case 的 `totalSlices` 是否 `< coreNum`（host tiling 中打印 `totalSlices`/`numBlocks` 即可定位塌缩）。
- **观测（tool_observed，本 trace 实测 a2）**：对齐版 case 4 退化约 25×、case 6/8 退化约一个数量级，三者均 `totalSlices=1`、`numBlocks` 塌缩为 1 串行单核；不对齐版（v1）同样 case 多核并行、无退化。
- **解释（interpretation）**：多核并行度由 `numBlocks` 决定。`blockLength` 对齐到 slice 看似安全（保证 slice 边界），但在 `totalSlices < coreNum` 的 case 上可用 slice 数不够分给所有核，`numBlocks` 塌缩到 1，整 block 串行单核执行，大 shape case 退化一个数量级。判别要点：若对齐版大 slice case 成对远慢于不对齐版、且 `totalSlices < coreNum` → 即为 numBlocks 塌缩，优化方向是条件对齐（`totalSlices < coreNum` 时不对齐），而非加核或换 API。

## 验证信号

- blockLength 对齐 slice 版 cann-bench 评测：大 slice case（case 4，`totalSlices=1`）串行单核、耗时约 25× 慢于不对齐版 v1 多核路径（参照路线：v1 不对齐 blockLength 多核路径；来源平台：a2；证据等级：tool_observed）。
- 同次评测 case 6（bf16）/case 8（fp32）亦 `totalSlices=1`、`numBlocks` 塌缩为 1，耗时退化约一个数量级（来源平台：a2；证据等级：tool_observed）。

## 失效边界

- 本退化在 a2（910B3 / DAV_2201，24 核）实测；a3、950 平台未在本 trace 实测，**未验证，推断可迁移**（同属多核 `numBlocks` 切分模型，但核数不同，`totalSlices < coreNum` 阈值不同，需重测）。
- `totalSlices >= coreNum` 时对齐不塌缩 `numBlocks`，不触发本退化。
- 修复（条件对齐 `edit`）的 verbatim before/after 在归一化事件流中为占位（adapter 丢失），未从 trace 逐字恢复；判别与根因由对齐/不对齐版前后成对评测耗时对照支撑，修复后信号未独立隔离观测。