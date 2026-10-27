# memory

- [DMA 原子写回的状态范围与输出初始化](dma_atomic_write_scope.md)：说明随路原子累加影响后续 GM 搬运、需要正确初始化目标与关闭状态，并区分 DMA 原子和跨卡通信原子。

- [Ascend C 缓冲所有权、流水交接与槽位布局](buffer_ownership_and_slot_layout.md)：区分 TQue 深度和实际缓冲块数，说明手工内存、队列和多阶段共享槽位的布局及同步约束。

- [串行计算阶段的资源重建与预计算结果交接](phase_boundaries_and_resource_reinitialization.md)：说明阶段拆分时片上资源回收、独立 TPipe 生命周期和 GM 预计算结果的可见性。
- [常驻参数与跨迭代状态的资源预算](resident_data_and_state_budget.md)：区分循环不变量、递推状态和阶段内中间值，评估 UB/L1 常驻复用的容量、同步与数值代价。

- [L1 常驻分区的字节布局与复用坐标](l1_resident_partition_coordinates.md)：补齐 QP 四槽和 KV 三槽示例的容量、dtype、K 分块及预加载同步边界。
- [手动双缓冲、三缓冲与二维槽的游标合同](manual_slot_cursors_and_event_ownership.md)：区分任务游标、物理存储位置和事件所有权，补齐初始化、延迟阶段与释放闭环。
- [TQueBind 复用中的内存收益与中间计算](tquebind_reuse_and_intermediate_compute.md)：区分共享输入输出存储、队列深度、物理份数和额外 Vector 依赖。

- [分阶段别名与转置状态常驻](phase_aliasing_and_transposed_state_residency.md)：给出 RmsNorm/RoPE 区域范围及迭代转置的实际循环次数，区分物理分区和分时复用。
- [常驻参数的扩宽视图与复用范围](promoted_constant_views_and_reuse_domains.md)：给出 gamma 后半区暂存的字节范围，区分原始、转换和广播结果的实际生命期。
- [递推状态版本与梯度 lane 累计](recurrent_state_versions_and_gradient_lanes.md)：明确反向更新旧状态、跨 batch 的 lane 归约以及最终输出的初始化与所有权。

- [UB Bank 映射与带 Padding 的物理步长](ub_bank_mapping_and_padded_pitches.md)：把具体架构 bank 模型与通用地址分析分开，保持改 stride 后的计算语义。

- [高阶 Scratch 复用与 TPipe Reset 边界](high_level_scratch_reuse_and_tpipe_reset.md)：区分临时空间、输出和 GM 分配，按实际 API 空间需求建立阶段复用。
- [TBuf 常驻仍占 UB 且需要真实数据依赖](resident_tbuf_is_ub_and_requires_data_dependencies.md)：保留循环不变量、状态更新和搬入转换链，纠正常驻比例与队列误解。

- [优化组合的资源与依赖合同](optimization_composition_resource_contracts.md)：按同一缓冲对象、同一控制域和峰值存活量判断兼容，纠正全局硬互斥。

- [权重反量化的 UB、L1 与 Bias 联合预算](weight_dequant_ub_l1_and_bias_budget.md)：保留双输入输出槽和 metadata 的参数化容量，并在输出 padding 后重新计算。

- [量化 Scatter 的复制归属与 Stream 顺序](quantized_scatter_copy_ownership_and_stream_order.md)：保留双 Kernel 设计条件，纠正只读索引竞争及搬运对齐绝对化。

- [Cube/Vector 缓冲生命周期与输出行距](translated_cube_vector_buffers_and_output_stride.md)：修正 TBuf 预加载同步并保留 K_L1 分块和 Fixpipe 布局。

- [DSL 转换的 Cast Staging 与 DMA 所有权](dsl_cast_staging_and_dma_ownership.md)：区分参数字节数、有效区、补齐、事件和缓冲复用。

- [BF16 别名缓冲与平台采样路径](bf16_alias_buffers_and_platform_sampling_paths.md)：保存半区Cast布局和310P采样，补齐覆盖、Mask及原子条件。

- [尾部 DMA、原子补零与压紧](tail_dma_atomic_zero_padding_and_compaction.md)：保存DataCopyPad及GatherMask，补齐合法GM范围和Mask宽度。
- [UB 分区字节核算与复制别名](ub_partition_byte_accounting_and_copy_aliases.md)：保存索引缓存及分区，纠正分配数与容量混淆。

- [双缓冲物理槽与常驻参数预算](double_buffer_slot_counts_and_resident_parameters.md)：按实际队列、参数复用和分支核算容量，区分具备槽位与实现重叠。

- [对齐单位、布局行距与填充值](alignment_units_layout_stride_and_padding_values.md)：修正以元素数冒充字节数的多处说明，保存不同类型的物理布局。
- [回退搬运的别名、原子重叠与短长度](rollback_copy_aliases_atomic_overlap_and_small_counts.md)：保留老平台尾部搬运思路，明确回退并非天然安全。

- [UB 驻留生命周期与架构相关bank端口](ub_resident_lifetimes_and_architecture_specific_bank_ports.md)：保留常驻与分时复用，补齐3510两读一写冲突和地址映射限制。
