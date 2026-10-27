---
type: Guide
title: 升精度计算中的状态、舍入边界与 UB 成本
description: 补充混合精度链的存活缓冲计算、累加状态初始化、非线性更新顺序以及 FP16/BF16 的取舍。
tags:
- task.design
- topic.precision
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A1_fp32_intermediate.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D1_mixed_precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D2_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A1_fp32_intermediate.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D1_mixed_precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D2_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A1.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D1.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D2.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# 升精度计算中的状态、舍入边界与 UB 成本

低精度存储与较宽精度计算可以分别选择。输入/输出 FP16 或 BF16 每元素占 2 字节，FP32 占 4 字节；只有同元素数的那部分存储或传输减半，整个算子的 UB 和耗时不一定减半。BF16 有较大的指数范围但只有 7 位显式尾数，不能由其动态范围推断累加、除法或开方无须更高精度。

设每 tile 有 n 个有效元素，n_aligned 是实际分配元素数。把一个原有 FP16/BF16 中间缓冲改成 FP32，额外占用为 2*n_aligned 字节；保留原缓冲并新增一份 FP32 则增加 4*n_aligned 字节。输入、输出、累加器、临时平方/开方、每个队列槽都须独立计数，最后加对齐、API scratch 与固定预留。不能把单缓冲的增量当成整条计算链增量，也不能把来源 80% 预算余量当作硬件规则。

累加器在首次读取前必须初始化到算法规定状态。示例中 `sum += input` 的 sum 必须已有本次归约的零状态或上次合法 partial；分配 buffer 本身不建立该状态。AdaGrad 一类状态更新还包含梯度平方、历史累计、开方/除法和参数更新，是否保留 FP32 状态需按参考计算链决定，不能只提升最后一步。

同一个 TBuf 连续两次 `Get<T>()` 并不会分配两份独立存储。示例中 xCalc/yCalc 同取 calcBuf 可能别名：仅当算子允许对应原地形式、旧值已经消费且 API 支持重叠时才能复用。也不能为了凑齐 Cast 链而读取未初始化的输出 yLocal。Cast 与 ReinterpretCast 分别执行数值转换和视图重解释。

合并连续计算中的上提/下落点可减少转换和中间舍入，但必须保留算子所要求的可观察舍入边界。主块、尾块和每个 dtype 分支应遵守同一约定；纯搬运或类型相同的路径不应强行执行同类型 Cast。升精度策略对计算吞吐、寄存器/UB 压力和额外搬运的影响需单独测量。

舍入不能仅按“上提/下落”套一个跨所有类型的模式：FP16/BF16 的有限值拓宽到 FP32 可精确表示，而 INT32/INT64 转浮点可能丢有效位。选用目标重载实际支持、并与 oracle 一致的模式；来源中所有 FP16 运算必须升精度、BF16 总可保持原类型、BF16 固定容差 1e-2 的说法均不是通用合同。参见[转换链与存储约束](conversion_chain_and_storage.md)。

验证包含长累计、抵消、大均值小差值、舍入中点、各 dtype 的尾块和特殊值；同时检查存活状态和 UB 峰值。输入本来已是 FP32 时，重复上提没有价值，但仍可能存在不稳定的算法或错误的舍入点。
