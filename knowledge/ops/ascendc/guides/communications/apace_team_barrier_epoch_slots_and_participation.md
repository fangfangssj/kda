---
type: Guide
title: APACE TeamBarrier 的 epoch 槽与参与者
description: APACE TeamBarrier 的 epoch 槽与参与者的来源契约、适用范围和失效边界。
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
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/architecture.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE TeamBarrier 的 epoch 槽与参与者

来源TeamBarrier是AIV侧UBMEM计数轮询，AIC分支为空；UB32B。每rank GM需求(1+2*totalJobs)*32B：baseflag、CrossDevice per-job槽、CrossCore per-job槽。CrossCore读取本job的epoch+1，写回后轮询其余job；CrossDevice读取本job epoch+1，先写共享baseflag，再按step=min(totalJobs,R)轮询i=jobIndex;i<R;i+=step的远端baseflag，完成后写回per-job槽。轮询无超时，缺rank/缺job/错epoch都会挂起。

CrossDevice的每job槽记录局部进度，远端轮询目标却是每rank单个baseflag，不能误认为每个producer都有独立远端完成标志。多job共写baseflag要求外部核间阶段顺序闭合；任意一个job到达不等价全部producer完成。来源前R核映射报告通过、后R核compute-first报告失败，不能由此证明totalJobs=R天然正确或天然失效，两者须结合完整CrossCore/SyncAll序列。

后R核通信映射job=blockNum-1-blockIdx，通信对象jobs=R分target；归约在前blockNum-R核。已记录替代方案是TeamBarrier jobs=1，由其有效job单核在所有本地通信完成并SyncAll之后CrossDevice，再SyncAll放行归约。物理block0不必等于job0，文档两种叫法不一致，实际guard必须与Init映射相同。不能同时保留额外内建Device barrier而改变调用次数。

各轮epoch初值与生命周期必须一致，data buffer memset不能清独立barrier区。来源builder将2MiB barrier区附在engine context后但未明确memset，零初值是未证实分配假设。新轮/新launch复用不得在peer尚访问时重置。按rank/job记录每个barrier序号和等待对象验证，不以单rank PASS证明协议。
