---
type: Guide
title: MoE 样例位重解释前缀计数的数值前提
description: MoE 样例位重解释前缀计数的数值前提的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/moe_dispatch.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/include/utils.h
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 样例位重解释前缀计数的数值前提

CalcCurExpertCnt不是普通数值Cast：先int32 Sub，再ReinterpretCast<float>做Abs，按int32 Mins(...,1)，再把0/1位模式当float ReduceSum，最后按int32位模式取结果。int32位值1在float中是最小正subnormal，而不是数值1.0；若设备保留这些subnormal并按所需精度累加，小计数可以由结果位模式还原。

合法小expert差中正差是subnormal位模式，负差的float视图可能是NaN，Abs后的正位模式经int32 min截成1。该技巧依赖Abs处理NaN位模式、subnormal输入/输出和Reduce的真实设备语义；若flush-to-zero把这些最小subnormal清为0，异expert计数就错误归零，dstSlot接近linearIdx，可能越过每expert BS容量。不能仅用数学公式sum(min(abs(ids-e),1))证明此实现。

INT_MIN差位模式为负零，Abs变零造成假相等；正常0..EP-1且EP<=64不触发，但无范围检查允许非法输入进入。极大累加也不能无限用浮点结果位值当整数，UB和计数域需另限定。对比独立整数prefix，覆盖同expert、正/负差、混合路由、零长度及目标设备subnormal行为；未做NPU验证，本文保留为待验证实现条件。可用支持的整数/比较计数或数值Cast+Reduce+Cast替代时须同时确认API和性能。
