---
type: Guide
title: Softmax 在线重标定与有限 Mask 限制
description: 保存分块状态和压缩偏移，限定高阶API精度声明。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_05_softmax_stability.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# Softmax 在线重标定与有限 Mask 限制

标准有限输入softmax先减全行max再exp即可限制每个指数≤1；完整score矩阵的O(Sq*Sk)存储压力不等于exp本身必溢出。在线softmax节省中间矩阵并维护分块running max/sum/output，不应把空间收益说成只有在线算法才能数值稳定。

若块max为m_b、按m_b算l_b与a_b，合并m=max(m_old,m_b)，alpha=exp(m_old-m)、beta=exp(m_b-m)，l=alpha*l_old+beta*l_b，a=alpha*a_old+beta*a_b。若当前块直接按新m算exp，可省显示beta；来源伪码只写running_sum*alpha+block_sum，成立前提是block_sum已在新m域，不能把局部m_b域统计直接相加。最终输出a/l，旧output重标定应对应未归一化表示。

有限mask值-10000并不在任意score/scale下保证无效概率0，特别是有效score更小、全mask或scale作用顺序改变时；应定义全mask、空row、Inf/NaN和输出策略。Causal/Band/Prefix压缩偏移须与所选mask物理形状对应，ComputeOffsetForCausal的delta≤0取min(-delta,s1Base)、否则min(delta,s2Base)*maskS2只是该压缩模板的地址算法，宽有符号边界和布局仍需检查。

M对齐8/K对齐64及SoftmaxConfig{false}是来源组件配置，不是所有SoftMax接口的常量。高阶SoftMax<float,false,false>仍需正确scratch、tiling与有效shape；不能由API名称证明其内部必有更高精度、Kahan、额外特殊值处理或一定优于手写版本。保持输入scale/mask、padding和输出Cast合同后，才比较有限数误差及特殊值。
