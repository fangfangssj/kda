---
type: Guide
title: "共享 NPU 上跨 run 性能 A/B 必须以候选侧绝对时间 cand_us 为判据（ratio 差 <5% 被噪声主导）"
description: "reference 侧小用例 ref_us 分钟级摆动 40-50%；跨 run ratio A/B ~5% 以下为噪声；回归排查重测两侧、不做跨 session 纯 ratio 比较"
tags: [task.optimization, topic.performance.analysis]
status: draft
sources: []
created_at: "2026-09-25T00:18:31Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# 共享 NPU 上跨 run 性能 A/B 必须以候选侧绝对时间 cand_us 为判据（ratio 差 <5% 被噪声主导）

## Evidence

- **run_id**: a37a0522-5a5f-4bc8-b9ab-fc68bdcf7443
- **host**: Ascend950DT
- **clock**: 2026-09-25T00:20Z
- **trigger**: manual-post-run-intake
- **authorization**: 2026-09-25 ruling

## 适用

共享 NPU 上的性能 A/B 判断（优化前后对比、跨 run 对比）。

## 反模式

共享 NPU 上顺序两次 run 之间 ratio 漂移 ±3–5%（4-14 实测：小用例 reference 侧 ref_us 几分钟内摆动 40–50%，大用例稳定）。**跨 run 的 ratio A/B 在 ~5% 以下被噪声主导**——用 ratio 差判优化优劣会得出假结论。

## 正确做法

跨 run 比较一律用**候选侧绝对时间（cand_us）**逐用例对比；reference 侧是共享噪声的主要载体。同 session 内背靠背 A/B（两侧同窗口测）仍可用 ratio。

## 回归排查

op 跌回 1.0 以下时：重跑 perf runner，逐用例对比 cand_us 与上次 run 的留存值；远端 /tmp 可能已清——两侧都重测，不做跨 session 的纯 ratio 比较。

## 证据

run a37a0522-5a5f-4bc8-b9ab-fc68bdcf7443（4-14）优化期多次顺序全量 50 用例测量的 reference 侧摆动记录。
