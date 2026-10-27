---
type: Guide
title: Blaze 执行引擎、布局标签与特化
description: 纠正框架总览的流水方向与核数查询，保留抽象层和数据 Trait 的区别。
tags:
- task.design
- topic.programming_model
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-framework-overview.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-sync-patterns.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/platform_info/platform_ascend_c/get_core_num_aic.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/platform_info/platform_ascend_c/get_core_num_aiv.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/platform_info/platform_ascend_c/get_core_mem_size.md
  role: constraint
created_at: '2026-09-11T02:57:41Z'
updated_at: '2026-09-11T02:57:41Z'
platforms: []
---

# Blaze 执行引擎、布局标签与特化

在来源所述分离AIC/AIV模型中，常见AIC数据流为GM→MTE2→L1→MTE1→L0A/L0B→Mmad→L0C→Fixpipe→UB/GM；AIV为GM→MTE2→UB→Vector→UB→MTE3→GM。总览图把MTE1/MTE2方向写反，并把MTE3与Fixpipe混同，应按真实数据流修正。不能据总览“AIC包含Vector”推导在分离架构的AIC实例中任意Vector调用均有效。

GetCoreNumAic查询Cube核数，GetCoreNumAiv查询Vector核数；总览将后者赋给aicNum并注释AIC不正确。耦合架构的返回含义与分离架构不同。L1/L0A/L0B/L0C/UB容量应来自目标平台和实际可用资源，来源512/64/64/256/256KiB表只是特定上下文的候选值，不能跨版本硬编码；GetCoreMemSize的真实原型、存储枚举和返回方式也应核对，不能照抄GetCoreMemSize(0/1)的示意。

Blaze在选定Kernel/Block/Policy中封装tile循环、K迭代、缓冲和同步；Tensor API提供张量描述与Copy/Mmad路由；Ascend C原语执行实际指令。封装范围随组件而变，不代表所有组合均有完整同步，也不能因header-only/编译期标签便宣称整个运算零运行时开销。

Tensor API的Shape、Stride、Coord可为嵌套tuple，Layout还携带Pattern/Trait。来源NDExt/DNExt用于外层ND/转置表示，ND/DN用于相应内部表示，NZ/ZN为分形标签，ZZ/NN及ScaleA/B的ND/DN标签用于相应Scale路由。标签不是实际数据变换；源/目标位置、arch、Pattern和dtype/Trait共同决定可用Copy特化，Ext与非Ext不能只按名称互换。

32B数据载体对FP16/FP32/8bit/FP4分别为16/8/32/64逻辑元素，但Scale Trait可能使用C0=2，L0C分形边长也不能套32/sizeof规则。具体Trait决定布局，见[物理存储](matmul_logical_shape_and_physical_storage.md)。MakeFrameLayout构造描述，MakeCopy/MakeMmad组装调用对象，with注入参数；描述有效不证明真实消费者支持非连续stride或尾部。

Policy的ScheduleType可驱动模板偏特化，具体Kernel还依赖ProblemShape、BlockMmad、Epilogue、Scheduler等类型。公开定义存在不证明某组类型已经构成可执行实例；Batch、Grouped、量化、MX、融合是可能叠加的需求轴，各自组合仍须满足同一入口的真实绑定。本文不继承来源固定版本支持全集，也未进行设备验证。
