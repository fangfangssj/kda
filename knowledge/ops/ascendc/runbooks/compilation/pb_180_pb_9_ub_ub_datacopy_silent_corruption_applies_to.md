---
type: Runbook
title: "PB-9 (UB→UB `DataCopy` silent corruption) applies to arch 200 (dav-m200) TOO, not just 220x"
description: "PB-9（UB→UB DataCopy 静默损坏）在 arch20成 shape\"（`str(None0/dav-m200 同样成立，不是 220x-only——`Adds` 恒等桥在 310P 仍是强制的。"
tags: [task.debugging, topic.data_movement.data_copy]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-180", "DataCopy"]
---

# PB-9 (UB→UB `DataCopy` silent corruption) applies to arch 200 (dav-m200) TOO, not just 220x

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0/9.1; op_class=all (any UB->UB DataCopy)`
**PB-180 — PB-9 (UB→UB `DataCopy` silent corruption) applies to arch 200 (dav-m200) TOO, not just 220x** — falsified on 15_Pad ko-8 Opt6 (2026-09-01): pass_a 21/51, max_diff 3-5, single falsification build, reverted. `Adds` identity stays mandatory on 310P. PB-9's "220x-only" scope is now known-wrong; treat UB→UB DataCopy as corrupt on every arch until proven. backend=ascendc.
