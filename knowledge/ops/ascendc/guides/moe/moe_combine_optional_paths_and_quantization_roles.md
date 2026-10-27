---
type: Guide
title: MoE Combine 的可选路径与量化角色
description: MoE Combine 的可选路径与量化角色的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/combine-dataflow.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/spec-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/dispatch-dataflow.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE Combine 的可选路径与量化角色

回来源window协议以expandIdx[srcRank,srcToken,srcTopK]把专家输出写到来源token的固定槽；source本地等实际需要的槽，再按expertScales加权聚合，恢复XOut[Bs,H]。通信按有效expand项发送，最终聚合按本rank原token，工作量不同。window[token,slot]中的slot通常前K为MoE、后S为shared；顺序是逻辑坐标，不是接收到多少就追加多少。

expertScales是路由权重，dynamicScales是量化解码尺度，expandScales的确切角色依接口版本，不能同名就替换。通信量化需发送编码、接收解码后再聚合，FP32累加及最后输出dtype舍入需闭合；将权重在dispatch和combine都应用会重复乘。packed FP4的数据存储宽度与expert计算后输出宽度也须按真实类型重新确认。

zero expert不贡献值但改变等待目标；copy expert用oriX乘路由权重；const expert使用oriX、alpha1/2、constV的真实函数，来源只列输入未给完整公式，不能臆造计算；shared可来自window槽或额外sharedExpertX，二者若都启用需确认不是同份重复累加。token mask/expert mask同样改变发送集合、等待目标和输出定义。

residualX、gamma或Add/RMSNorm辅助函数在签名中不代表主路径已使用。非线性RMSNorm通常需要完整聚合后的统计，不能独立对每专家先归一化再求和而称等价；残差、scale的顺序须按最终数学合同。融合可以减少中间GM回写，但只有真正消除对应搬运才有收益。

performanceInfo是可选等待统计缓冲，来源部分实现按rank做atomic max，需初始化、原子类型支持与输出域完整；它不是XOut功能输出。groupList等预留项状态依版本，不能从本页概述宣布所有公开API支持或拒绝。实际值域、参数数量以匹配版本入口为准。
