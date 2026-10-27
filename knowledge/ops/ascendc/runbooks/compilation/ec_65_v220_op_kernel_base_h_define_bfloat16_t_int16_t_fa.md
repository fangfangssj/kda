---
type: Runbook
title: "V220 `op_kernel/*_base.h` `#define bfloat16_t int16_t` fallback silently disables bf16 on A5 — author local A5-safe copy under `workspace/kernel/`"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=norm-quant-port_a3-V220-pure"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-65-v220-op-kernel-base-h-define-bfloat16-t-int16-t-fa.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["EC-65"]
---

# V220 `op_kernel/*_base.h` `#define bfloat16_t int16_t` fallback silently disables bf16 on A5 — author local A5-safe copy under `workspace/kernel/`

> 原 OKF v1 卡号：**EC-65**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - A5 port of V220-pure port_a3 norm/quant op compiles + builds clean but bf16 inputs silently produce wrong results (or fp16-truncated results). Surface error mes

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=norm-quant-port_a3-V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

- **Symptom**: A5 port of V220-pure port_a3 norm/quant op compiles + builds clean but bf16 inputs silently produce wrong results (or fp16-truncated results). Surface error message often absent — output values just don't match reference. Probe shows downstream casts producing int16-like values instead of bfloat16.
- **Root cause**: Upstream V220 `op_kernel/*_base.h` (norm and quant family base headers) contains a fallback `#define bfloat16_t int16_t` near the top, used when V220 toolchain bfloat16 support is absent. On A5 (Ascend950PR) with CANN 9.0.0 + bisheng that natively supports bfloat16, this fallback is wrong — bfloat16 should be the native type, not int16_t aliased. The fallback `#define` shadows the real type at compile time; downstream `Cast<bfloat16_t, ...>` becomes `Cast<int16_t, ...>`, silently producing wrong bit-patterns.
- **Affected**: port_a3 V220-pure norm/quant family ports (RMSNorm, LayerNorm, quant variants) that include `op_kernel/<op>_base.h` from upstream verbatim.
- **Workaround**: Author a local A5-safe copy of the base header under `workspace/<op>/kernel/` (e.g., `<op>_base_a5.h`) with the fallback `#define` removed, and `#include` the local copy instead of the upstream V220 header. Do NOT modify the upstream submodule.
- **Status**: OPEN (upstream-V220-side compatibility fallback that is hostile to A5)
- **Cross-ref**: KB_INDEX EC-1..EC-65 row; OL-185 V220→V351 calibration anchor.

<!-- 迁移自 porter kb/target/ascendc/（EC-65，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
