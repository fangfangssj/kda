---
type: Runbook
title: "输出幅值远小于 atol 时 allclose 不可能 fail（全零输出也 PASS）——精度判定必须 MERE/matched_ratio 承重并用已知坏变异体证明比较器灵敏度"
description: "输出幅值远小于 atol 时 allclose 不可能 fail（全零输出也 PASS）——精度判定必须 MERE/matched_ratio 承重并用已知坏变异体证明比较器灵敏度"
tags: [task.testing, topic.precision]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PRECISION-ALLCLOSE-VACUOUS-SMALL-OUTPUT"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PRECISION-ALLCLOSE-VACUOUS-SMALL-OUTPUT

## 适用场景

- 参考实现故意取小值的 op（KDA/线性注意力、门控递归、归一化/衰减状态）：输出 ~1e-3 vs bf16 atol=0.1。

## 结论（39_ChunkKdaFwd 实测，row20，run 7aaa60a6）

allclose 在这种 op 上 vacuous：**全零输出都判 allclose=True**。精度报告必须 MERE/matched_ratio 承重，且先用 known-bad 变异体（o*1.05 / 全零 / 末时间步清零）证明比较器能 FAIL，再报 N/N。39 的实测：变异体全被 MERE 正确捕获（MERE 4.74e-02 / 1033 / 12.82），真候选 MERE 1.60e-07 vs 阈值 7.8e-03（约 49000× 余量）。

- **Evidence**：39_ChunkKdaFwd 归档（mfu56_closeloop）evidence/kb_*.json + README known-bad 表。
- **Cross-ref**：CAND-MULTI-OUTPUT-NONE-LEAF-DEAD-ZONE（同族另一测量坑）。
