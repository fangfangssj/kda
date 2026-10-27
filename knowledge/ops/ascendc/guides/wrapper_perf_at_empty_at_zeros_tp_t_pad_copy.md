---
type: Guide
title: "wrapper 数据搬移是端到端 perf 合同的一等优化目标（at::empty 替代 at::zeros + Tp==T 跳过 pad-copy）"
description: "wrapper-inclusive 计时下 host 布局变换占落后用例 19-44%；两项零风险消除 +11% 平均 ratio，50/50 逐位不变；含 at::empty 合同 falsifier"
tags: [task.optimization, topic.data_movement.layout_transform, topic.performance.analysis]
status: draft
sources: []
created_at: "2026-09-25T00:18:15Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# wrapper 数据搬移是端到端 perf 合同的一等优化目标（at::empty 替代 at::zeros + Tp==T 跳过 pad-copy）

## Evidence

- **run_id**: a37a0522-5a5f-4bc8-b9ab-fc68bdcf7443
- **host**: Ascend950DT
- **clock**: 2026-09-25T00:20Z
- **trigger**: manual-post-run-intake
- **authorization**: 2026-09-25 ruling

## 适用

wrapper-inclusive 性能合同（`model(*inputs)` 端到端计时，候选侧含 host 布局变换）下的 AscendC 算子性能优化；尤其 port_a3 类带 chunk/pad 布局的 op。

## 机制

端到端计时把候选 wrapper 的 host 侧数据搬移算进测量窗口——在 4-14 ChunkScaledDotKktFwd 上它占落后用例候选时间的 19–44%。两个零风险消除：

1. 输出用 `at::empty` 替代 `at::zeros`——前提：kernel 可证明存满每个元素（本例：每 unit 行 store 的并集覆盖 [0,T)×H，masked 值显式存 0）。
2. 当 `Tp == T`（T 已是 chunk 倍数）跳过 zeros-pad+slice-copy：padding 字节根本不存在，view 与 padded tensor 逐位相同。

收益：平均 ratio +11%，精度 50/50 逐位不变。

## 探测模式（先测后改）

动 kernel 之前，对代表性用例分别单测 `wrapper-only` / `ext-only` / `zeros(out)` / `zeros+copy(k)` 四段耗时，先确认 wrapper 占比再决定优化落点。

## 反例（falsifier）

若未来 kernel 写入少于 out 的全部元素（如跳过全 masked 行），`at::empty` 合同即破（未写元素是垃圾值）——该 kernel 必须恢复 `at::zeros` 或自行零填跳过区。

## 证据

run a37a0522-5a5f-4bc8-b9ab-fc68bdcf7443（4-14，fixed-50）：Opt1 应用后 perf mean 0.9775→1.1126（worker 复测口径），精度 50/50 PASS 且逐位不变，det 50/50；独立 O5 复测 ratio 1.0788（5×50）。
