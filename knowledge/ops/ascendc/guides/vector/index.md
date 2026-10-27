# vector

- [Vector repeat 分批必须同时满足计数与步长位宽](repeat_parameter_width.md)：对实际使用 uint8_t repeatTime 的重载，把大 tile 分为不超过 255 次的批次，并独立检查 repeat stride 和地址偏移。

- [Ascend C 向量广播的地址语义与标量替换边界](broadcast_addressing_and_scalar_substitution.md)：用重复步长表达跨行复用，并区分数学等价、浮点等价和 API 调用次数减少。
- [GatherMask 解交织的源跨度与输出长度](gathermask_deinterleave.md)：在 A2/A3 参考范围内区分固定模式、自定义选择掩码、repeat 源跨度和压缩后输出长度。

- [Vector 比较的位掩码、补齐范围与有效元素](comparison_masks_and_padding.md)：将 Compare 的位结果与输入元素数、缓冲容量和后续 Select/归约区分，避免把向量粒度或填充值当成所有重载的共同限制。

- [字节 Mask、Bit Mask 与数值谓词](boolean_masks_and_numeric_predicates.md)：区分布尔极性、打包格式和加性掩码，说明向量条件组合与标量广播的等价条件。

- [带类型的计算 DAG 与共享状态依赖](typed_dag_and_shared_state_dependencies.md)：从混合示例中提炼节点依赖、输入转换复用和多输出调度约束，保留算子语义边界。

- [SwiGLU 双分支语义与高阶临时空间](swiglu_two_branches_and_api_scratch.md)：区别单独 SiLU 和双输入 SwiGLU，明确 beta、布局、别名及双缓冲范围。

- [Counter Mask 的作用域与尾部计数](counter_mask_scope_and_tail_counts.md)：区分 Normal、Counter 与 Compare/Select 重载的实际覆盖，保留 A2/A3 与 950 的文档边界。
- [UB 计算链融合与队列交接](ub_chain_fusion_and_queue_handoffs.md)：保留中间值免 GM 往返、混合类型与多队列所有权的完整条件。

- [解交织模式与 Host 预计算保留](deinterleave_patterns_and_host_precompute_preservation.md)：纠正 GatherMask 参数含义，保存四分量物理跨度及迁移后的完整数据流。
- [逐行广播的成本与读后写依赖](row_broadcast_cost_and_read_after_write.md)：保留行尺度扩展、gStart 地址、scratch 与广播结果消费的 RAW 边。
- [标量参数向量链与小工作量分发](scalar_tensor_chains_and_small_work_dispatch.md)：保存归一化计算顺序及小任务路径的搬入、标量读取和写回合同。

- [Ascend C API 家族与存储位置决定的搬运](api_family_mapping_and_scope_specific_copy.md)：补充向量标量分流、scope 搬运和重载身份，限制机械映射。
- [Norm 融合的 Affine 重排与流水边界](norm_fusion_affine_rounding_and_pipeline_limits.md)：区分统计域、真实遍历、浮点重排和缓冲生命周期。

- [LeakyRelu 分解表达式与尾块边界](leaky_relu_split_expression_boundaries.md)：保存BF16升精度链，限定分段等价的特殊值范围。

- [转置、寄存器循环与 Scatter 累加](transpose_register_loops_and_scatter_accumulation.md)：保存SIMD布局变换和寄存器接口，指出冲突写入并非原子归约。
- [向量谓词、标量数学与 Axpy 修改语义](vector_predicates_scalar_math_and_axpy_mutation.md)：保留专用指令选择，区分两种Select极性及浮点运算合同。

- [Vector 效率的mask生命周期与归约接口形态](vector_efficiency_mask_lifecycle_and_reduction_api_shapes.md)：保留计数模式和融合，纠正单槽重复Alloc及伪标量归约调用。
