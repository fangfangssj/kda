---
type: Guide
title: dtype 直传与 Wrapper 转换流量
description: 将低精度数据直接传给 Kernel，保存内部 FP32 计算与原有舍入语义。
tags:
- task.optimization
- topic.precision
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/discovered/disc_X3.md
  role: primary
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# dtype 直传与 Wrapper 转换流量

多输入elementwise可按原输入dtype编译/分发Kernel，在UB内拓宽到所需计算精度，最终按输出合同转换。这样可省去Wrapper中物化的完整FP32输入和输出转换临时Tensor。它仍需要保证外部dtype、shape、stride和设备stream顺序，reshape/contiguous本身也可能发生复制。

以三输入一输出foreach_addcdiv为例，在全部输入均为2B低精度、三个输入各转换一次、FP32 Kernel读三写一、最终输出再转换一次且不计缓存复用的模型中：输入转换3*(2+4)=18B/元素；Kernel 4*4=16B；输出转换4+2=6B，总40B。直接低精度读三写一为8B，流量模型比为5。来源称“两次cast”不足以描述这一账本；若只有部分输入转换、已有FP32输入、融合/缓存或别的准备步骤，按真实流量重算。全FP32读三写一是16B，不有同样5倍空间。

Kernel的输入存储类型、内部FP32 x1/x2/x3与tmp、输出区和每槽份数分别预算；减少GM临时Tensor不会自动减少UB峰值。可复用x1f作为tmp的前提是其旧值已消费，Div/Mul/Add等支持对应别名。32B对齐折算为32/sizeof(T)元素仅适用于需要该字节对齐的对象。

Host需完整分发FP16/BF16/FP32及不支持类型的行为，模板sizeof(T)==4不能在一般类型域证明T就是float。标量系数、除法中间精度、运算顺序和最终舍入点要与参考一致；FP16 CAST_ROUND、BF16 CAST_RINT须核对当前接口/目标，不可泛化为所有框架版本的唯一匹配方案。FMA、零值捷径及NaN/Inf传播可能改变结果，不能承诺FP32路径自动bitwise一致。

验证输入组合、非连续视图、空Tensor、尾块、除零、舍入中点与大幅值。来源tile7008、系数36→28、误差0.27和性能案例未复现；改变系数后改善不能独立证明原失败来自UB溢出，应以真实存活分配和越界证据核对。
