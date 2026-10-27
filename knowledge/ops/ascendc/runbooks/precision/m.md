---
type: "Runbook"
title: "批次行索引未对 M 回绕致整行置零判别"
description: "批次化逐元素掩码 kernel 用线性偏移 gmOffset % (M*N) 反推 (i,j)，在 chunk 循环内推进行索引时 i += newJ/N 未对 M 取模；tile 跨 batch 边界时 batch>0 的行 0 得到 i=M 而非 i=0，使 start_col=clamp(i+diagonal,0,N)=N、keep 区间为空、整行误置零。判别=只在 rank≥3 且 batch>1 的用例失败、且表现为整行置零而非逐元素错位。"
tags: ["task.debugging", "topic.precision", "topic.compute.indexing", "topic.implementation.masking"]
status: stable
sources: []
created_at: "2026-08-18T05:07:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["i_wrap"]
---

# 批次行索引未对 M 回绕致整行置零判别

## 现象

在 ascend950 平台实现 Triu 算子后，白盒 aclnn ST 中 `r4_float32_rank3_d0`（shape `{2,4,4}` fp32，diagonal=0，连续）精度失败。修复 ACLNN 561000（AutoContiguous）后，该用例从运行时错误转为纯精度错误：输出无 `ACLNN error` 行，仅 `[FAIL]`，ST 从 82/83 到 83/83 只差这一个用例。Triu 对 `(2,4,4)` diagonal=0 的期望：每个 4×4 矩阵的上三角保留（row <= col 保留，否则置零）；batch 1 的行 0 应全部保留（0<=0,1,2,3），但 kernel 错误地将 batch 1 的行 0-3 全部置零。

## 适用场景

- **工程形态**：kernel 直调（`__simd_vf__` + `asc_vf_call` MicroAPI regbase 范式）。
- **API 路径**：`TriuVF` 内 `ij0 = gmOffset % mn` → `i = ij0 / N` → `start_col = clamp(i+diagonal, 0, N)` → keepMask 构建。
- **验证过的平台与 CANN 版本**：ascend950（dav-3510），CANN 9.1.0。
- **适用粒度**：`operator_family`——所有按"线性偏移 % (M*N) 反推 (i,j)"范式实现的批次化逐元素掩码 kernel。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；修复实现（`i = (i + newJ/N) % M`）的终版代码未从 trace 逐字恢复。
- **不适用或未验证场景**：a2/a3 未验证，推断可迁移；单 2D 矩阵（无 batch 维，dim0 = M×N）不触发此 bug（i 不超过 M-1）。

## 根因

`TriuVF` 用线性偏移 `gmOffset` 反推当前 tile 起始的行列坐标：`ij0 = gmOffset % mn`（mn = M×N），`i = ij0 / N`，`j = ij0 % N`。这给出 tile 起始点的正确 (i,j)（i ∈ [0, M)）。但随后在 chunk 循环内推进索引时，`i += newJ / N` 不对 M 取模——当 tile 跨越 batch 边界（如 `(2,4,4)` 的 32 元素 tile = 2 个 4×4 矩阵），batch 1 的行 0 得到 `i=4` 而非 `i=0`。`start_col = clamp(i + diagonal, 0, N)`：对 batch 1 的 row 0，`i=4` → `start_col = clamp(4+0, 0, 4) = 4 = N` → keep 区间为空 → 整行置零。正确值应为 `i=0` → `start_col = 0` → 整行保留。

## 判别方法

**触发条件**（tool_observed 失败症状）：aclnn ST 输出 `[FAIL] r4_float32_rank3_d0`，summary 为 `82/83 cases PASSED`，仅此一例失败、无 ACLNN error 行（纯精度错误，非运行时错误）。

**可执行探针**：
1. 跑白盒 ST 并过滤失败/汇总：`cd operators/Triu_package/tests/aclnn && ./run.sh 2>&1 | grep -E 'FAIL|summary'`。
2. 确认失败用例为 rank≥3、batch>1、diagonal=0 的连续 fp32 用例（如 `(2,4,4)`）；核对失败用例的输出是否为 batch>0 的整行被置零（而非逐元素错位）。
3. 读 kernel 行索引推进逻辑，检查 `i += newJ / N`（或等价推进）是否对 M = mn/N 取模；若未取模且 tile 跨 batch，即为根因。
4. 对照实验：单 2D 矩阵用例（dim0 = M×N，无 batch 维）应 PASS——i 不超过 M-1，不跨 batch，不触发。

**原始观测**（Ascend950，CANN 9.1.0，Triu 白盒 ST）：
- `[FAIL] r4_float32_rank3_d0` / `82/83 cases PASSED`，仅此一例失败，无 ACLNN error 行。
- 失败用例为 rank=3、batch=2、diagonal=0 的 fp32 连续张量；batch 1 的行 0-3 全部置零。

**机制解释**（根因分析，属 interpretation）：kernel 用 `gmOffset % mn` 反推 (i,j) 并在 chunk 循环内推进 `i += newJ/N`，但未对 M 取模；tile 跨 batch 边界时 batch 1 的行 0 得 i=M 而非 i=0，使 `start_col = clamp(i+diagonal, 0, N)` 钳到 N、keep 区间为空、整行置零。判别核心是"只在 rank≥3 且 batch>1 的用例失败、且表现为整行置零而非逐元素错位"——指向行索引跨 batch 回绕缺失，而非计算链或掩码公式本身错误。正确算式为 `i = (i + newJ/N) % M`（M = mn/N = 矩阵行数）。

## 预防

- 任何用"线性偏移 % (M*N) 反推 (i,j)"范式实现批次化逐元素操作的 kernel，行索引 i 必须在推进时对 M 取模。
- DESIGN 伪码写 `i = ij0 / N; ... if (j >= N) { j = 0; i++; }` 未体现 `i % M` 回绕——实现时须补全。
- 测试须覆盖 rank≥3 且 batch>1 的用例（如 `(2,4,4)`），否则单 batch 测试无法暴露跨 batch 回绕 bug。

## 验证信号

- `[FAIL] r4_float32_rank3_d0`，Triu ST summary `82/83 cases PASSED`（仅此一例失败，无 ACLNN error 行）（来源平台：950；证据等级：tool_observed）

## 失效边界

- a2/a3 平台：未验证，推断可迁移（行索引回绕语义跨平台一致）。
- 单 2D 矩阵（无 batch 维，dim0 = M×N）：i 不超过 M-1，不触发此 bug。
- rank 8D `(1,1,1,1,1,1,4,4)`（dim0=16, mn=16, M=4）：i 从 0..3 不超 M-1，不跨 batch，不触发。

## 坏实践

- 假设"batch 维由 gmOffset 单调递增自然覆盖，无需显式 b"就无需 i 回绕——该描述误导，i 在跨 batch 时必须 `% M`。

## 误诊

- 初始怀疑 `r4_float32_rank3` 的 561000 与 r6 非连续用例同因（non-contiguous）。AutoContiguous 修复后 r6 PASS 但 r4 转为精度错误，才定位到 i 回绕 bug。
- 初始怀疑 UB headroom（196608→180224）是 561000 根因，修复后仍 80/83 同样 3 失败——UB headroom 非根因。
