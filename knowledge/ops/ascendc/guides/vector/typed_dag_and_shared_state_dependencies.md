---
type: Guide
title: 带类型的计算 DAG 与共享状态依赖
description: 从混合示例中提炼节点依赖、输入转换复用和多输出调度约束，保留算子语义边界。
tags:
- task.design
- topic.compute.composite
- topic.optimizer
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P13_special_algorithms.md
  role: primary
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# 带类型的计算 DAG 与共享状态依赖

表达式 DAG 用节点显式描述输入、类型转换、计算和输出依赖。共同输入的转换可成为共享节点，调度器据依赖及资源策略安排计算、暂存和释放；这提供了复用和融合机会，但不能仅凭 DAGSch、Bind 或 MemOptCfg 名字断言最优寄存器分配、零总开销或自动获得更快性能。

来源代码描述一个 AdaGrad 类状态链：gradPower=grad*grad，accumOut=accum+gradPower，accumSqrt=sqrt(accumOut)，lrMulGrad=lr*grad，varT=lrMulGrad/accumSqrt，varOut=var-varT，并发布 varOut、accumOut 两个输出。共享 accumOut 既被后续除法消费又是输出，必须保留到所有消费者结束；输入输出别名时也不能提前覆写旧 var/accum/grad。是否更新 slots、是否有 epsilon、广播和 dtype 都要来自目标算子定义，本片段不证明完整 AdaGrad API 语义。

来源文字另一部分讨论 FakeQuantAffineCachemask 的基类/派生类，却没有与以上 DAG 对应的量化计算。可保留其共享 CopyIn/CopyOut、分块字段与按 dtype 选择计算逻辑的组织方式，但不能把 AdaGrad 的 DAG 当作 FakeQuant 的等价实现。模板静态分发可避免相应运行时分支，普通继承或类名本身不能证明零虚调用、所有指令均可消除或 BF16 已受支持。

节点图的真实性可通过逐输出展开依赖、确认旧状态和新状态的版本、对齐每个转换的 dtype/rounding 以及计算同时存活的字节范围来检查。比较多输出值、别名与非别名输入、不同 tile 和尾部；缓存中间结果的收益应连同新增缓冲、执行顺序及端到端时延评估。这些 Bind/ElementwiseSch 名称是来源框架片段，未在本次编译验证为通用 SDK 接口。
