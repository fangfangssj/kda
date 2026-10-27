---
type: Runbook
title: "CANN 9.2.0 Level-0 ReduceSum(dst,src,tmp,mask,repeat,stride) 是全归约进 dst[0]，不是逐 repeat 求和——逐 repeat 要用 WholeReduceSum 的 int-mask 重载"
description: "CANN 9.2.0 Level-0 ReduceSum(dst,src,tmp,mask,repeat,stride) 是全归约进 dst[0]，不是逐 repeat 求和——逐 repeat 要用 WholeReduceSum 的 int-mask 重载"
tags: [task.debugging, topic.precision]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-REDUCESUM-L0-TOTAL-NOT-PER-REPEAT"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-REDUCESUM-L0-TOTAL-NOT-PER-REPEAT

## 适用场景

- soc=Ascend950DT_9582 (351x)，CANN 9.2.0；任何需要逐 repeat/逐行局部求和的 AIV kernel（KDA/attention 内积归约等）。

## 结论（72_ChunkKdaFwdIntra 实测 root-cause）

CANN 9.2.0 上 Level-0 `ReduceSum(dst, src, tmp, mask, repeat, stride)` 是**全量归约**——文档原文「sum all input elements」，结果只写 dst[0]，**不是**按 repeat 段分别求和。症状极具迷惑性：输出第 s=0 行 = 所有 repeat 行的总和，其余全 0，看起来像索引错误。

**修法**：用 `WholeReduceSum` 的 int-mask 重载（逐 repeat 求和，dstRepStride=1，srcRepStride=块数），3510 上存在于 `kernel_operator_vec_reduce_intf_impl.h`（P-P47 形态）。

- **取证手法（值取证）**：Aqk 第 t=7 行只有 s=0 非零且恰好等于全行和（-0.02939 vs 全行和 -0.0294 完全一致）→ 锁定全归约语义。
- **证伪器**：修复后 Aqk 第 t=7 行逐 s 与参考一致。
- **Evidence**：72_ChunkKdaFwdIntra worker（2026-09-20，run 见 mfu56_closeloop 归档）PROGRESS.md 逻辑定位段；最终 P 50/50 + F 5.1977×。
- **Cross-ref**：P-P47（reduce 重载族）。
