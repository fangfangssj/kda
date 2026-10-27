---
type: Guide
title: MoE 计数、前缀边界与专家总量
description: MoE 计数、前缀边界与专家总量的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/dispatch-dataflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/combine-dataflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/spec-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 计数、前缀边界与专家总量

裸count[e,r]、每expert内沿rank累计P[e,r]和按expert-major全展平累计F[e,r]是三种表示。裸计数总量sum(count)；P的总量sum_e P[e,lastRank]；F总量最后一个元素。取每rank原始数时P[e,r]-P[e,r-1]（r0减0），F在r0还需减上expert末值，不能都按同一公式解读。

overview例两行[1,1,2,3]、[3,8,10,12]声称总量12：只有全展平prefix语义成立；若每行各自prefix则总量3+12=15。expert1/rank1差8-3=5两种都相同，单个差分例不能证明总量口径。dispatch-dataflow又给epRecvCounts常见一维EP裸计数，而overview是expert×rank矩阵，属于不同版本/内部与外部表示，不能因名称同就互换。

状态矩阵是每expert每source原始tokenCount；按source汇总得rank计数，按expert汇总得专家计数。expertTokenNumsType=1对应数量、0对应前缀是该来源合同；取沿rank累计矩阵末行只能得到各expert总量，要输出跨expert前缀还需再scan，源文字在这两步间混用。

dispatch epRecvCounts作为对应combine epSendCounts时，双方必须共享统计轴、计数/边界编码和expandX排序。不能直接用A或某核局部总量当发送任务数，也不能用聚合后rank数替代等待所需expert×source状态。计数需非负、prefix单调，差分/总和/字节乘积须防类型溢出。

用两个expert、多个rank含零段的非对称矩阵检查裸值、prefix与最终expandX长度；再验证每个返回三元组对应原始route，避免一个全局sum相等掩盖错段。
