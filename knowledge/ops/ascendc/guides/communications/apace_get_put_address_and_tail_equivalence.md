---
type: Guide
title: APACE GET/PUT 地址与尾块等价条件
description: APACE GET/PUT 地址与尾块等价条件的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE GET/PUT 地址与尾块等价条件

来源GET从remoteWin[target]+winOffset+slotByteOffset+selfRank*tileMaxBytes读到localAddr+target*chunkBytes+tileByteOffset。PUT从localAddr+target*chunkBytes+currentTileIdx*tileMaxBytes写到remoteWin[target]+winOffset+selfRank*chunkBytes+tileByteOffset。GET的rank槽与PUT的source chunk是不同布局，不能只交换src/dst指针而保留其他游标。

PUT源步长固定tileMaxBytes，而目标按实际tile长度推进。head长度h、尾长q<=h且只有一个尾块时，tail起点headCnt*h恰正确；多个短尾块时第二个尾块的固定步长比连续输入多(h-q)，tail>head时甚至head阶段也因tileMax取tail而错位。等长块无需此修正。padding方案必须真的分配/生成固定pitch输入，或适配源地址按实际累计，不能只改tiling数字。

GET与PUT DoCommit都跳过self，因为self channel未建立。GET DoWait跳过self整个hook；PUT DoWait仅跳过Drain，仍执行需要的barrier。self结果因此需要本地直读或显式本地拷贝，接收窗口self槽没有远端写入者。ReduceScatter组合读取self的staging段、remote的Win段。

来源GET环形示意还需实际地址取模；基类slotByteOffset单调递增不会自动变成tid%bufferCount。只有生产与消费共用同一ring映射、槽释放在真实读取完成后、重新写入前等待旧slot完成，才能安全复用。全量buffer>=总tile无重用依赖。GET固定示例无可核验完整kernel，不能把契约图当已验证实现。
