---
type: Guide
title: 多输出归约的存活缓冲预算
description: 按输入、转换、累加器、临时空间与输出的真实类型和同时存活数量计算 UB 及跨核 workspace。
tags:
- task.design
- topic.compute.reduction
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/multi-output-buffer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/reduction_operations/reduce_max_interface/get_reduce_max_max_min_tmp_size.md
  role: constraint
created_at: '2026-09-10T13:09:00Z'
updated_at: '2026-09-10T13:09:00Z'
platforms: []
---

# 多输出归约的存活缓冲预算

多输出归约可以在一次输入扫描中计算多个统计量，但需要额外的常驻状态。sum 与 squareSum、value 与 index 可能有不同的 dtype、临时空间和输出布局，不能仅用“输出数 K × 一个标量大小”推导全部资源。

可将预算写成同时存活的输入槽位、转换区、各累加器、算法临时区和输出槽位的字节数之和。每项都使用实际分配后的对齐长度；双缓冲按两个实体槽位计费，类型重解释不能扩大底层分配。

例如，若确实采用两个低精度输入槽位、一个 FP32 转换区、K 个等长 FP32 累加器、一个等长临时向量和两个输出槽位，则可写成：`tileElements*(2*inputBytes+4)+alignedA*4*(K+3)+其他临时空间 ≤ 可用UB`。这是特定缓冲设计的方程；不同输出类型、转换原地复用或额外 reduction scratch 会改变系数。

API 临时空间应使用与具体重载、shape 和 dtype 一致的查询或明确约束。例如 [GetReduceMaxMaxMinTmpSize](../../apis/simd_api/adv_api/reduction_operations/reduce_max_interface/get_reduce_max_max_min_tmp_size.md)给出相应接口的临时空间上下界；不能把参考示例中的固定 4 KiB 公式用于所有 Reduce API。

固定开销已经超过可用容量时，需要选择其他 tile 或算法。把负容量相减转成无符号数，或强行将不可行的 tile 数改成 1，都会隐藏资源不足。求出的最大 tile 还需满足搬运和计算接口限制，并复算对齐后的完整预算。

跨核 workspace 应记录每核、每个输出状态的物理字节范围，以及对齐、发布和复用要求。Welford 的 count/mean/M2、ArgMax 的 value/index 不能自动按同一 T_acc 套公式。最终还需核对 Host 申请的字节数与 Kernel 使用的偏移完全一致。
