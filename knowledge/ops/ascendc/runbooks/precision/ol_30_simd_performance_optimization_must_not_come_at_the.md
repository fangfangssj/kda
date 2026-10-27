---
type: Runbook
title: "SIMD performance optimization must not come at the cost of precision"
description: "SIMD performance optimization must not come at the cost of precision"
tags: [task.debugging, topic.precision, topic.programming_model.execution_model.simt, topic.data_movement]
status: draft
sources: []
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-30"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# SIMD performance optimization must not come at the cost of precision

> 原 OKF v1 卡号：**OL-30**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMD kernel optimization attempts, especially eliminating per-group/per-block loops

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process, trust_calibration
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: SIMD kernel optimization attempts, especially eliminating per-group/per-block loops
- **Lesson**: MXFP4 SIMD V4 "fast" replaced per-32-group exponent with a tile-wide shared exponent (1024 elements shared).
  Performance did improve (small tensors 1.08x faster than SIMT), but **precision no longer matches MXFP4 spec**.
  A3's hand-written SIMD has the same issue — BATCH=512 sharing one exponent is exactly the root cause of its precision bug.
  **Never publish a "faster but numerically wrong" kernel as production**.
  **Hard rules**:
  1. After any performance optimization, must compare against PyTorch spec precision (not only against the previous AscendC version)
  2. If an optimization requires changing algorithmic precision semantics (e.g. enlarging group_size), must be **explicitly labeled "approximate"** in docs
  3. "0 mismatch precision" is only meaningful when compared against the PyTorch spec. Comparing against our own CPU ref does not count (our ref may have the same bug)
  4. The A3 implementation's precision bug is precisely because per-group precision was abandoned for SIMD performance — this is not an acceptable trade-off
- **Evidence**: MXFP4 SIMD V4 (2026-04-07), A3 hand-written SIMD precision bug analysis

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-30（category=process, trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓（不公开，pinned commit cb9c78c85ead3088398f3455051355349298b3cb）SIMT_VS_SIMD_DECISION.md#precision-constraint-ol30 节选卡同 lesson 复核：确认失败例（MXFP4 1024 元素共享 exponent、A3 手写 SIMD BATCH=512）与规则与本卡 Lesson/Hard rules 一致；补录其独有的安全反例——SG forward 为 per-token 计算、每个 token 独立，向量化不引入精度问题（同内容亦见 `../../concepts/simt_vs_simd_decision.md` 的 OL-30 小节）。platforms 依来源节选卡 frontmatter 声明补 950（[a3]→[a3, "950"]，批次 7 先例）。
