---
type: Guide
title: ORI 归约路由、填充值与方差遍历
description: 保留分块和数值模型，限制统一对齐与性能禁令。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_reduce_patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_norm_fusion_patterns.md
  role: supporting
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# ORI 归约路由、填充值与方差遍历

对单个连续逻辑归约轴可分解(O,R,I)：O为之前维乘积，R为归约长，I为之后维乘积。I>1可把(o,iTile)作为工作单元沿R归约；I=1且整行可驻留时批量AR，多于UB时分块累计再归约。非连续输入需真实stride，多个不连续归约轴不能无说明直接套一个连续R。物理转置可能有成本，但并非总比归约更贵或永远不值得。

树形归约可缩短依赖链，浮点误差取决于输入分布和顺序，不保证总小于线性。分块Vector累加器后一次末级归约可减少Scalar回读/全屏障，但sum/max/min/prod需各自运算和中性值；不应把逐块“+=”套到所有归约。归约长度0、输出shape和mean分母先按合同处理。

padding按语义：sum为0、max为-inf、min为+inf、prod为1，均需匹配异常值与有效mask。来源先写“全dtype先补零”与后表冲突，max/min/prod不能补0通用处理。DataCopyPad的rightPadding最大255及API内部AlignUp(R,8)、FP16实际按16元素、Level2输出8B对齐、AR仅FP32是所述版本/重载约束，应与实际定义核对，不能扩展到所有归约API。R=2007等能暴露物理行距与逻辑长度不一致。

FP16/BF16升FP32常有利于求和累加；有限输入的max/min本身不会因比较树自动损失1ULP，绝对值和最大值可精确保持原可表示值，不能把求和精度理由套到max。isReuseSource只有源确实不再被其他消费者使用且重载允许时才开；2–3倍收益、icache6.5–7%均为来源案例。

sum/sumsq同遍可形成E[x²]-mean²，但小方差相对均值平方仍会灾难消减，输入小/已归一化并不是安全证明。中心化两遍或Welford需按统计合同选择。来源“均值修正”先求mean，再求residual均值mean'=mean+Σ(x-mean)/L，之后求Σ(x-mean')²并除L或L-1；伪代码未写分母不能照抄。若数据不能驻留，mean、残差修正、中心平方和、最终apply各可能需要再次读取，不能仍统称两次GM遍历。

Kahan跨chunk补偿为y=x-c;t=sum+y;c=(t-sum)-y;sum=t，可减少某些累加误差但不能恢复输入/partial已丢失的信息。来源FP32从3e-4到1e-6是有限案例。FP32不普遍精确匹配，BF16大R误差0.5–0.8%也不是可放宽合同的通用阈值。

UB预算包含输入、输出、FP32中间、scratch与buffer份数；8192–12288元素只是示例量级。int64中做ceil/乘积后收窄可避免部分int32溢出，但int64本身也需范围检查。增加核数或缩tile可能增加流量和固定成本；标量、小输出写回、物化broadcast与转置均应按实际成本及正确性选择，不能用统一禁令替代分析。
