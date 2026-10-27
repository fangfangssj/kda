---
type: Guide
title: APACE 入口 ABI 与 tiling 消费者
description: APACE 入口 ABI 与 tiling 消费者的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/operator-anatomy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/dev-methodology.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/design-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/plan-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE 入口 ABI 与 tiling 消费者

来源UDMA样例7参依次context指针、A、ScaleA、B、ScaleB、C和按值算子tiling；compute-first示例增加bias与workspace成9参，bias省略时8参，数量由真实入口决定。Impl持有&tilingData只能在入口局部对象有效期内使用。按值的语义不保证编译器所有版本都将数据复制到GM，地址空间与传参由实际编译ABI核验。

通信5字段、计算17字段与算子级组合是不同角色。A2A含data+scale两份CommTilingData、计算tiling和localMatmul；AG含mmTile+commTile，kernel自行推scale通信参数，缺Host scale字段不是遗漏。字段类型/顺序/宽度/pack、alignas及sizeof/offsetof需Host和Device一致；pack(8)+alignas(8)不足以独自保证布局，必须成对push/pop并验证嵌套结构及编译器。

固定A2A Impl有A/B/C类型与TransA/TransB五参数，入口固定false/true；AG实际只有三个类型参数。文档另一处宣称两者均有转置参数，与其签名表矛盾。泛化转置支持需数据/scale布局、Host tiling和入口实例同时闭合，不能仅添加bool。相同FP8字节宽度可在已证实只依大小的SWAT分支共享tiling，但不能由sizeof相同推导所有dtype策略等价。

源KERNEL_TYPE_MIX_AIC_1_1设置特定AIC/AIV参与模型。AIV分支写在AIC前面不产生执行先后，真正顺序由事件/通信依赖建立；模板CommPolicy消除动态分发不意味着实际Wait零耗时。目标类型和SDK对应的调度属性需符合参与模型，不将某次507015唯一诊断为schedmode。

FP32输入扩展不自动使L0C翻倍：该MX基线累加本就FP32。若仍用FP32累加，相同tile的L0C字节数不变，输入/scale搬运与支持策略须另算。bias扩展也要确认每个累加阶段实际加几次，不能仅因kernel存在bias参数就认定无需语义改动。
