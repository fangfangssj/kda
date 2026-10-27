---
type: Guide
title: MoE 阶段归属与共享统计
description: MoE 阶段归属与共享统计的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 阶段归属与共享统计

发送按有效token或linear route，状态发布按expert/目标槽，等待按接收状态槽，回搬按对应有效记录，combine最终聚合按原token。它们的N和参与核集合不同，不应复用一套未改语义的begin/end。q=N//P、r=N%P，core i起点i*q+min(i,r)、数量q+(i<r)覆盖余数；overview combine伪码只取BS/num_aiv漏余数，N<P时甚至全漏。

等待核对自己状态段求c_i并写共享workspace；完成可见后按状态段顺序计算exclusive prefix，得到全局输出偏移。expandX、expandIdx及scale都按同一偏移回搬，段内slot顺序保持；按核号scan要求核号与状态段顺序相同。若修改等待与回搬分工，需要明确重映射，并非数学上必须同一核处理两阶段。

本核UB、LocalTensor和成员数组不自动对其他核可见。共享GM通常按核独占槽或使用正确原子协议，读取方经真实一致性路径取得；SyncAll不能独自修复任意DCache旧值。DataCopyPad是该sample主要路径，SetValue/GetValue是否可用取决于地址空间、cache、清理/失效和同步合同，不能把所有共享GM标量访问永久禁用。

host只传总核数、kernel再决定各阶段分核是该sample组织约定，Host也可以合法传阶段参数，关键是两侧工作域一致。最终expert/rank统计可以分层并行收口，不一定单核，但每个输出需完整贡献与唯一写者。热点expert/rank会造成尾核，均分token不保证均分通信或回搬字节。
