---
type: Guide
title: 三方精度比较与诊断假设的边界
description: 区分通过判据、参考误差比、量化容许和统计症状，补全生成器与真实 dtype 的一致性。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/ascendc-evaluation/references/precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/ascendc-evaluation/references/test-cases.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-test-develop/references/precision-and-perf.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-test-develop/references/test-framework.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-test-develop/references/precision-and-perf.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-test-develop/references/test-framework.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/references/integer_compute.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/scripts/integer_compute_check.py
  role: implementation
created_at: '2026-09-11T03:46:00Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 三方精度比较与诊断假设的边界

三方比较中Golden为较高精度独立参考，Ref为工作dtype的NPU参考实现，Ans为目标Kernel；比较前输入字节、参数权重、输出结构和最终cast语义须一致。CPU FP64不是任意问题的精确数学真值，不能表示全部uint64，遇到不支持或失败回退两方比较时证据强度已改变，应明确记录。

CAKE来源的浮点通过条件是无新增非预期NaN/Inf且torch.isclose逐元素mismatch_rate=0，误差比只作诊断；其他SHMEM来源允许basic OR reference兜底是不同策略，不能合并。CAKE另有Ship/NeedsReview/Iterate的比值分级，也不应覆盖原通过条件。每个输出独立声明指标、阈值与实测，不能用宽松atol/rtol替代要求的平均误差或其他统计。

来源CAKE的BF16/FP16/FP32 atol=rtol分别1e-2/1e-3/1e-5，ulp_tol=2，small-value阈值sv_th分别2^-8/2^-11/2^-14、sv_err分别2^-16/2^-16/2^-30；direct测试文档给同量级默认容差。这些是来源配置，未定义完整ULP/svec公式时不能自行补造。它们不构成所有算子的统一数值精度。

误差比ans_error/max(ref_error,floor)需同一metric定义、域和floor。来源max_re参考分级10/20，mean_re/rmse/svec为2/4，只表达该框架审视阈值；floor可显著改变小误差比分母，不能把比值小当精确。参考与目标同错、golden舍入或输入不一致需要单独排除。

整数/bool默认在整数域exact，绝对差需防止有符号溢出或无符号回绕；不能先转float64再宣称int64全部精确。来源quantized_output=true对所有整数输出atol=1是一种放宽合同，不是float→int必然允许±1：固定舍入可要求exact，融合输出中的index/count也不能随量化结果一起豁免。

zero_output、localized_outlier、systematic_drift、small_value_error等只是假设线索：零输出可能合法或未执行，局部误差可能地址/同步/公式，均匀偏差也不唯一对应精度。整数ae_max>128不证明溢出，ae_max=1不证明floor/ceil，均值近0/方差大不证明符号扩展错误；完整dtype、索引、输入、阶段与反例才支持归因。

参数化case必须让声明dtype/shape与生成结果一致。来源unary/binary生成函数无论case字段均创建FP32，grid的seq_len与var0_shape独立笛卡尔积还可能互相冲突。normal(0,0.1)是无界正态，不能认为只在近零；uniform(2,8)/(-8,-2)是否饱和由具体函数决定。ones、正常、近零及不同量级分布只是候选，不能代替边界和特殊值。

## 整数比较器自身的反例

来源 integer_compute_check.py 在 np.array_equal 失败后将输入转为 int64 并用绝对差判断；这会掩盖错误。NumPy 1.26.4 中，[[1,2]] 与 [1,2] 能广播出全零差值，尽管 shape 不同；uint64 最大值 2^64-1 转 int64 后变为 -1，因此能与 int64(-1) 误判相等。int64 的最小值 -2^63 取 abs 仍溢出为负数，若同一误差数组还有一个零，max 会变为零，导致 [-2^63,0] 与 [0,0] 也被放行。空数组还会在无 identity 的最大值归约处报错。

比较前必须验证 shape 和 dtype/有符号性合同；数值 exact 判断应避免依赖可能回绕的减法，误差诊断与通过判定分别实现。跨整数类型需要明确可表示范围，不先做有损转换。空输出是否合法以及其比较结果须显式定义。最小/最大整数、异 shape、跨符号、空输入应进入比较器的反例集，而不是被算子测试的高通过率掩盖。
