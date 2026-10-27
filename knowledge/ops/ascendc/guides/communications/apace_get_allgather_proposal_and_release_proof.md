---
type: Guide
title: APACE GET AllGather 设想与槽释放证明
description: APACE GET AllGather 设想与槽释放证明的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/get-all-gather-quant-matmul/design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/get-all-gather-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/index.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE GET AllGather 设想与槽释放证明

来源GET AllGather登记not_found：有AllToAll GET hook但没有AllGather GET hook或完整GET kernel证据。可提炼的数学设想是复制A[M,K]，每rank持B的N/R列，先算C_s[M,N/R]再沿N收集完整C[M,N]。这里按N分片的是通信前局部产物，收集后的每rank最终输出完整，不能同时将最终输出称N分片。

AllToAll GET按requester rank选remote槽，AllGather则多个requester需要同一个producer产物。直接复用A2A hook只有在窗口每requester槽都放了对应数据或地址适配时才等价；前者可能复制R份，增加内存和搬运。不能从逐target循环就宣布天然AllGather。源“GET要2倍搬运”也只在额外staging拷贝方案成立。

ring必须在覆盖前等旧slot被全部需要的remote reader消费。当前rank AIV完成自己的GET只证明它读过别人，不自动证明别人已读完本rank槽；释放需要全读者完成的协议或等价rendezvous。生产者本地CrossCore通知也不能单独证明远端AIC已写完。将回压Wait放在写Win之后已经太晚，源联合图的顺序需修正为复用前等待。

来源Wait(true)最后一轮不Drain，随后DoFinalize仅barrier，不能无API完成证据称“barrier完成final drain”。SyncAll是本地核同步；是否需要由参与核依赖决定，非GET名字保证可省。直接读取可寻址remote window的MTE兜底还需注册、完成、cache及生命周期证明。

文档声称4+rank不稳定，但上级fusion只登记无官方样例和一次根因未定位回退；保留为未验证风险，不能下GET在R>=4普遍错误的结论。GET测试表混入PUT单轮512KiB经验也不能作为GET硬限制。对应验证应覆盖不同rank身份、两次以上ring绕回、尾轮、self与全部reader完成。
