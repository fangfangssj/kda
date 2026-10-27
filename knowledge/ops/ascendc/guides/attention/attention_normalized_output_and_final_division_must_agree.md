---
type: Guide
title: Attention 在线输出归一化与最终除法必须一致
description: 纠正设计中的二次除法，保存mask、分核和内存尺度条件。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/fused_attention_online_softmax_design.md
  role: primary
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Attention 在线输出归一化与最终除法必须一致

在线Attention可维护未归一化a：mNew=max(mOld,tileMax)，alpha=exp(mOld-mNew)，p=exp(score-mNew)，lNew=lOld*alpha+sum(p)，aNew=aOld*alpha+pV，最后a/l。也可每步维护已归一化O：ONew=OOld*(lOld*alpha/lNew)+(pV)/lNew，此时最终直接输出O。

来源更新伪代码采用第二种已归一化形式，但外层结束又ElemwiseDiv(oUb,lUb)，会二次除法。取两个score都0、V为2/4，则l2、正确O3，源码组合最后变1.5。必须选定并保持一种表示；章节称O_acc并不能代替数学合同。若输出L接口实际期望logsumexp则应m+log(l)，不能把示例写L=lFinal直接当通用Attention接口。

causal/window判断要统一含尾或开区间：以query/key含尾[q0,q1]/[k0,k1]、合法k≤q时，k0>q1整块跳过，k1≤q0整块有效，其余逐元素mask。滑窗q-W<k≤q时，k1≤q0-W可跳过过旧块，边界等号与真实接口preTokens/nextTokens定义对照。source理论近50%只适用等分方阵因果块计数，不同S实测百分比无环境证据不推广。

m/l/a可由同一核在同次launch跨S2保留，GM workspace不是在线算法必需；跨核SplitKV、跨调用或状态放不下时才需要显式暂存/归并。MTE2加载UB/L1各需对应管线，PV既可Cube也可Vector实现，不能说910B必由Vector计算PV或固定只有8核。decode也不必绝对memory-bound，取决于batch、并行度、KV布局及计算。

内存要按当前query tile Sq、KV tile Sk和D计量：score Sq*Sk、输出状态Sq*D、m/l各Sq，以及Q/K/V、float转换、workspace和buffer份数；不物化S²中间矩阵不等于所有配置都固定65KB。来源单Q行D64/Sk128表逐项合计约68224B，D128约134400B，未含未列buffer/对齐/队列，不能凭总计和192KB断言任意平台安全。全局Q/K/V/O内存仍随S*D增长。

向量Mul+Reduce和标量PV只是可表达的基线，不是生产高性能证明。收益表明确预计，单launch省调度不能自动保证1.2–5倍。全掩码首块、首尾query/key、精度、跨tile最大值上涨与真实端到端性能需单独验证。
