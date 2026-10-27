---
type: Guide
title: 软件 Sigmoid 的范围规约与精度证据边界
description: 保存多项式/Newton 候选，纠正所有硬件超越函数仅 FP16 精度的泛化。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dequant_kernel_patterns.md
  role: primary
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# 软件 Sigmoid 的范围规约与精度证据边界

存储dtype不单独保证整条链的误差，但来源把A2所有Exp/Div/Sqrt/Rsqrt/GELU等统一称为FP16尾数精度、下游低位全是物理噪声，未绑定API/模式/输入范围证据。基本Mul/Add/Sub/Cast也有浮点舍入、overflow/flush和顺序差，不能称任意链bit-exact于CPU；某个操作近似误差也不一定支配最终误差或固定锁在2^-9。

来源Tier1/2/3、CPU libm约0.5ULP、NPU双方误差互相cancel、INT8自动吸收为±1LSB都不是通用保证。阈值也与同目录动态量化文档冲突：本篇FP32 MERE/MARE为2^-13/2^-12、FP16为2^-10/2^-9，另一篇给不同MARE值。应以实际验证器版本/公式及合同为准，不能按源码是否出现Exp机械给精度等级或改用FP16门限。

软件exp候选把z=-x按k=RINT(z*log2e)、r=z-k*ln2规约，再以七阶Horner多项式逼近exp(r)，乘由(k+127)<<23构造的2^k。系数、常量精度/FMA、舍入误差和域处理必须给出，来源没有系数表，不能称完整标准实现。构造指数仅对可表示正常范围成立，超范围、subnormal和NaN/Inf需分支；规约单精度ln2误差也计入整体界。

对正正常D提取指数和尾数m∈[1,2)，可用y0_m=1.5-0.5*m并乘2^-e，再Newton y←y*(2-D*y)。理想实数下初始相对残差最大1/8，每步平方，三步为(1/8)^8≈5.96e-8；实际FP32还叠加每步舍入，不能因此无条件宣称24bit/1ULP。48/17-32/17*D用于归一化D∈[0.5,1]，直接套[1,2)时D>1.5会负，正数倒数迭代可能发散。

位提取需uint32逻辑移位，(bits<<9)>>9若用int32算术右移会保留符号位；exp_bits应是真实8位指数域，(254-exp_bits)<<23仅在可表示范围有效。把0x3f800000与尾数字段相加等价OR需字段无重叠，且是整数位运算；不能对浮点值Adds该常数来构造位模式。ReinterpretCast只改视图，数值Cast不能由它替代。

将x clamp到[-50,50]不是全域FP32 Sigmoid等价：sigmoid(-100)约3.72e-44，clamp后约1.93e-22，绝对值很小但相对误差巨大，下游放大也可显著。正端舍入饱和与负端接近0不同，需按输出容差/组合算法证明或用稳定分段公式。来源整体2^-20相对误差及50case通过未附当前可复核制品，不作新平台能力。

来源helper使用dst/src/scratchA/scratchB四份H个FP32，其中src被覆盖，可按最后消费者复用中间量。subtensor若参数为非常量引用需绑定具名LocalTensor；in-place模式须先捕获全部仍需读取的源再覆写。不同dtype同地址Cast是否允许由重载定义，不能从相同位宽推导安全；独立buffer是保守方案但仍需同步及容量。
