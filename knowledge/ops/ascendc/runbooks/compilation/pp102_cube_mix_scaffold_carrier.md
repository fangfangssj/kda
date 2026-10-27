---
type: Runbook
title: "P-P102 载体：CUBE_MIX 类 A5 移植的原生 MIX 脚手架（Mmad cube + vector 收尾 + WorkspaceQueue 跨核同步，绝不纯 VEC）"
description: "P-P102 锚载体卡——正文即 cube_vector_fusion 例卡的规则面，全量 worked example 见 examples/cube_vector_fusion.md"
tags: [task.implementation, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: 'https://gitcode.com/cann/cannbot-knowledge/blob/6f0a9eb7d77b3971a8e312c08083891f0b997e8e/knowledge/ops/ascendc/examples/cube_vector_fusion.md'
  role: primary
created_at: "2026-09-21T00:00:00Z"
updated_at: "2026-09-21T00:00:00Z"
platforms: ["950"]
aliases: ["P-P102"]
---

# P-P102 — CUBE_MIX 类 A5 移植的原生 MIX 脚手架

## 适用场景

- soc=Ascend950PR/V351；op_class=CUBE_MIX（matmul/attention/conv/rnn/gmm/ffn）。can 参考是 cube 必需的 A3→A5 移植。

## 规则面（正文 = examples/cube_vector_fusion.md 的浓缩；全量 worked example 读那张例卡）

cube 类 A5 移植用原生 MIX：手写 `Mmad` cube（cmatrixInitVal=true 单 K 块）+ vector 收尾 + `WorkspaceQueue` 跨核同步（每 producer↔consumer 方向一个队列、PIPE 配对、i%K 错开调度）。**纯 VEC 顶替 = 架构性作弊（ARCHITECTURAL_HACK），finalize 门会拒**。harness 层根因（OL-235）：pybind 链接行固定不含 MatmulApi 主机 tiling → matmul 库 cube 路径在本 harness 结构性不可链 ⇒ 手写 Mmad 是唯一可构建路径。

- **Evidence**：flash_attention_score A5（decision_manifest 2026-05-29）；deformable_conv2d port_a3（2026-06-20）。
- **Cross-ref**：OL-188/OL-190/OL-200/OL-235、P-P103/P-P116、PB-35；完整正文 examples/cube_vector_fusion.md。
