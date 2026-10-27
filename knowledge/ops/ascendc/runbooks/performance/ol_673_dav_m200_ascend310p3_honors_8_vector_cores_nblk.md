---
type: Runbook
title: "dav-m200 (Ascend310P3) honors 8 vector cores; nblk=8 scales monotonically over 6/7"
description: "dav-m200 实测 8 个向量核可用，nblk=8 单调优于 6/7——7 核的旧记录被测量推翻"
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-673"]
---

# dav-m200 (Ascend310P3) honors 8 vector cores; nblk=8 scales monotonically over 6/7

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0/9.1; op_class=any dav-m200 multi-core kernel`
**OL-673 — dav-m200 (Ascend310P3) honors 8 vector cores; nblk=8 scales monotonically over 6/7** — the "7 vector cores" note (PB-28-adjacent) is superseded by measurement: 21_Scatter ko-1 Opt3b probe nblk 6/7/8 → 8 best. backend=ascendc.
