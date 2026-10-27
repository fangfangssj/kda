---
type: Runbook
title: "cube_utilization is core occupancy, not MAC activity (cycles-only; a scalar-bound core reads ~100%) — use aic_mac_ratio"
description: "cube_utilization is core occupancy, not MAC activity (cycles-only; a scalar-bound core reads ~100%) — use aic_mac_ratio"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-331"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-331 — OL-331: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (any msprof reading)。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0-beta.1。

cube_utilization is core OCCUPANCY not MAC activity (cycles-only; scalar-bound core reads ~100%); use aic_mac_ratio

- Former A5Ops ID: `OL-331`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (any msprof reading)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0-beta.1`

**Principle.** Two msprof columns look like they answer "is the cube busy?" and only one does.
`cube_utilization(%)` is derived from cycle counts alone, so **a core running nothing but scalar
instructions reads near 100%**. Use `aic_mac_ratio` for MAC-pipe activity.

```
cube_utilization(%) = 100 * aic_total_cycles / (ai_core_cnt * cube_freq * Task Duration)
aic_<pipe>_ratio    = aic_<pipe>_time(us) / aicore_time(us)
```

Fitted over 900 profile rows across two profiles, two kernels, block counts 2/28/32 and durations
7.7–710 us: max absolute error **0.0005 pp**. `ai_core_cnt` and `cube_freq` come from the SoC
`platform_config` .ini, not from the launch configuration.

**Consequence worth internalising.** When block count exceeds core count the metric reports the
*wave-quantisation efficiency*: 32 blocks on 28 cores gives 32/(2 waves x 28) = 57.1%, and the observed
value was 56.91%. Read naively that looks like "the cube is over half busy"; it means 24 of 28 cores were
**finished and idle** during the second wave. Contemporaneous `aic_mac_ratio` was **0.029**.

⚠️ **Degeneracy trap when fitting this yourself**: on a single profile where every kernel launches the same
block count, `Block Num` and the true core count are indistinguishable, and normalising by `Block Num`
yields a *perfect* fit (5 ppm) with a physically wrong clock. Only rows at a different block count break
the tie. **A tight fit on one profile is not evidence.**

- **Evidence**: FlashKDA A5 port, 2026-08-01. I had reported "cube utilisation 56.9%, so the fused part
  runs fine" and retracted it on this arithmetic.
- **Other instances (predicted)**: any occupancy-style percentage derived from cycles — treat "utilisation"
  as occupancy until you have found the formula.
