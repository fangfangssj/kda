---
type: Optimization
title: Matmul 静态 Tiling、部分常量化与运行期 shape
description: 区分静态配置上界、基本块和真实矩阵跨度，保留初始化顺序、转置、bias及原子累加语义，并验证实际 Scalar 收益。
tags:
- task.optimization
- topic.tiling
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/constant_folding_design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/guide.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_tiling_constant.md
  role: supporting
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# Matmul 静态 Tiling、部分常量化与运行期 shape

Matmul 初始化和迭代包含对 Tiling 字段、循环边界及地址的 Scalar 计算。静态 Tiling 将可确定的信息放入模板或常量表达式，减少运行期字段解析、分支和偏移计算；它不改变矩阵乘的数学定义，也不自动解决 MTE2 或 Cube 的吞吐瓶颈。

官方既有[常量化指南](../../guides/operator_practice/best_practices/matmul_tuning_cases/matmul_tiling_constant.md)区分两种范围：基本块和 singleCore shape 都确定时可做全量常量化；仅基本块确定时仍可做部分常量化。因此，“任意一个参数动态就所有常量化都无效”并不成立。`constexpr` 也不等于整个 Kernel 已经折叠；要看常量是否传播到真正使用处和编译结果。

## 三组 shape 不得混用

| 信息 | 作用 | 不能替代 |
|---|---|---|
| 静态 singleCore 配置/上界 | 限定模板预留能力和合法运行区间 | 每次调用的真实有效尺寸 |
| baseM/baseN/baseK | 基本计算与搬运粒度 | 全矩阵原始跨度或最大任务 shape |
| SetOrgShape、SetSingleShape、SetTail 等实际参数 | 描述真实存储跨度、单核工作量和尾块 | 静态配置中碰巧相同的默认值 |

静态上界必须覆盖所有调用分支；基本块过大可能增加尾块和补齐开销，但“base 大于实际尺寸”本身不等于必然越界，是否安全取决于接口的合法范围和尾部处理。数据 dtype、累加 dtype、ND/NZ、转置、bias、scale和各层缓冲份数须一起决定静态资源，而不是复制某个 128/256/64 配置。

某些版本可通过 `MatmulApiStaticTiling`、`GetMMConfig`、`GetMatmulApiTiling` 将配置作为 Matmul 模板参数；字段、模板槽位与初始化参数必须核对目标 SDK。给不存在字段赋值是编译错误，随后出现的常量表达式错误不证明单纯把 const 改为 constexpr 就能修复。可在 constexpr 函数中构造并修改局部配置，只要满足该 C++ 版本和实现的常量求值规则。

## 初始化可能重设 shape

来源案例在切换到静态初始化后，`REGIST_MATMUL_OBJ(..., nullptr)` 使用静态配置重新设置内部 shape，覆盖了更早的 SetOrgShape。可复用约束是：真实 shape 应在会重置它的初始化/注册完成之后设置，并在每个动态调用前保持正确。是否允许传 nullptr 及确切调用顺序仍由实际模板合同决定，不能普遍删除所有 Host Tiling。

来源的码本内积例中，A/B 原始跨度为 `Ka=128`、`Kb=32`；若初始化把二者都改成 128，B 会按错误行距读数，即使实际归约 K 仍是 32。应分别保留 SetOrgShape 的参数个数、Ka/Kb、实际 K，以及模板与 SetTensor 的转置约定。更换静态配置也不能改变 BiasType、输出方式、`IterateAll` 原子累加参数或原有执行模式。`MatmulImpl` 与消息驱动 Matmul 的选择涉及调用和同步合同，应作为独立改动验证，不能按类名断言全部模式都会死锁。

## 控制特化数量

动态尾行、有效 KV 长度、batch 和可变 S 不能强行替换成满块常量。可只枚举有限的 D/headDim 档位，保留其他维度动态；来源 IFA 示例按 D=64/128/256/512 选择 profile，两路 BMM 分别为 `[G,D]×[D,S]` 和 `[G,S]×[S,D]`。profile 中的分块 S 与实际序列长度是不同量，不能混写。增加档位会增加模板实例、二进制体积及编译成本，应保留未命中 shape 的合法路径。

## 证据和收益边界

来源记录的 FP16、M=N=K=8192 教程对比中，A2 Scalar 时间约从 1753.463 降至 968.616 μs，Task 从 4088.36 降至 4053.44 μs；950 Scalar 从 765.125 降至 426.398 μs，Task 从 2589.888 降至 2589.090 μs。这些是来源报告，未在本卡重新复现，恰好说明约 44% 的 Scalar 降幅不能当作端到端加速比，也不能据此证明指令缓存收益。

检查编译出的字段加载、分支、Scalar 指令和真实 Task 时间，并固定其他 Tiling、数值模式及执行路径。输出全零、局部偏差或小误差只能帮助形成假设；同时核对实际写入、跨度、索引/ties及算子的原始容差，不能从误差量级唯一认定根因或接受结果。
