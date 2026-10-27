---
type: Guide
title: APACE 通信游标单位与末轮 Wait
description: APACE 通信游标单位与末轮 Wait的来源契约、适用范围和失效边界。
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

# APACE 通信游标单位与末轮 Wait

来源CollectiveCommBase的CommTilingData描述splitAxis的headSize/headCnt、tailSize/tailCnt及nonSplitAxisSize。chunkSize=headSize*headCnt+tailSize*tailCnt（元素轴长度），chunkBytes=chunkSize*nonSplitAxisSize*sizeof(D)，tileMaxBytes=max(headSize,tailSize)*nonSplitAxisSize*sizeof(D)。元素长度、字节游标和tile编号不可互换；文档remaining字段有“字节”描述，但Commit减currentTileSize的示意为轴元素单位。

Init接收ctx、barrier、tiling、localAddr、UB、totalJobs、jobIndex及可选winOffset，共7个必填参数和1个可选参数。job>=totalJobs时早退，后续成员可能未初始化；全核读取轮次应来自全核已初始化的tiling。totalJobs>0，参与job映射及全部成员使用需有效。文档要求无guard全AIV Finalize与Init早退并存，必须证实该变体Finalize是空/有保护，否则不能直接复用此编排。

Commit按head/tail选择本轮有效长度，逐target发起；随后currentTileIdx++、tileByteOffset推进有效字节、slotByteOffset推进R*tileMaxBytes。Wait的bool waitLast与BarrierMode模板是两维参数。来源Wait(true)在currentTileIdx!=totalTiles-1时早退；若循环每轮Commit后Wait，T=1时永不DoWait，T=2时第一轮DoWait而末轮不DoWait，一般T仅倒数第二轮执行。不能以“true”解释为末轮必定drain。

完整实现须明确提交后计数代表下一tile还是已完成tile，最后一次提交后使用真实可保证完成的等待。单独调用Finalize未必替代Drain，GET有barrier、PUT是空hook。按T=1/2/3记录每次Commit、cursor、Wait与最后一次传输完成，是最低限度的协议验证。
