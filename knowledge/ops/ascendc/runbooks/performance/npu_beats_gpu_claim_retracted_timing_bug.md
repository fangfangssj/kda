---
type: Runbook
title: "SG \"NPU beats GPU 0.6x\" claim retracted after timing bug fix"
description: "SG \"NPU beats GPU 0.6x\" claim retracted after timing bug fix"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-9"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-9 — OL-9: A5Ops additive extension

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；适用范围细节见正文）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

SG "NPU beats GPU 0.6x" claim retracted after timing bug fix

- Former A5Ops ID: `OL-9`
- Fusion disposition: `additive_new`

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: measurement
- **Loaded by**: QA
- **Trigger**: when an NPU kernel appears to outperform GPU on a comparable workload
- **Lesson**: Early Sparse-Gather benchmarks using chrono timing with memset inside the measurement window showed SG backward xlarge at 0.29ms -- appearing to beat GPU's 0.48ms (0.6x ratio, "NPU faster"). After switching to aclrtEvent device timing with memset outside the event, the actual NPU time was 3.87ms (8x slower than GPU, not faster). The 13x measurement error came from chrono capturing only host-side launch latency while the kernel ran asynchronously on the device. Any "NPU beats GPU" result should be treated with extreme skepticism and verified with proper device-event timing.
- **Evidence**: EXPERT_FEEDBACK.md Q4/Q6 analysis ("当前容器 SG xlarge bwd 3.87ms, 旧容器 0.29ms"), output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md

---