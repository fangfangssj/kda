---
type: Guide
title: Softmax 模板的轴映射与Norm扫描次数
description: 保存六类组织及Norm可复用部分，纠正维度改名和阶段数混淆。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/usage_guide.md
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Softmax 模板的轴映射与Norm扫描次数

六类模板以真实存储布局区分AR[A1,R]和ARA[A1,R,A0]：SmallR把短末轴转置到外侧，FullLoad保留整段输入或指数，Recompute分chunk进行max/exp-sum/output三次GM读，ARA Online将max/sum在线融合为一次GM读再一次输出读。在线chunk内部仍先求max再求exp和，不是每元素只有一次UB访问；FullLoad也非一个Vector扫描。当前目录只有独立Softmax参考，融合Attention设计没有配套完整Kernel。

归约统计和逐元素输出骨架可用于LayerNorm/RMSNorm，但不是只换函数名。连续[B,S,H]沿H归约可展平AR[B*S,H]，不能直接改名为ARA[A1=B,R=H,A0=S]，后者真实stride是[B,H,S]；若归约S才可直接设ARA[B,S,H]。多轴归约需连续合并条件，非连续view不能只乘shape。

LayerNorm均值/方差可两遍统计或Welford单遍统计，若整行不驻留，最终归一化还需重读输入，因此统计遍数不等于完整算子GM读次数。RMSNorm一次平方和统计后同样需输出阶段。Welford合并还需count与delta修正，不能套Softmax max缩放；gamma/beta通常是逐元素向量，不应仅用标量Muls代替。eps位置、方差定义、affine广播及FP32中间精度必须保持接口语义。

SmallR的FP32≤16/FP16≤32、FullLoad的可用UB及Online默认128只是模板策略。树形合并改变浮点顺序，FP32并不保证所有长R无精度问题；空R、NaN、±Inf、全掩码和归一化分母0需定义行为，不能仅凭减max称天然稳定。模板代码中的shape窄化和buffer资源另有约束。
