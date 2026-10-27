# layouts

- [小通道转置的补齐布局与周期偏移表](small_channel_transpose_and_offsets.md)：从 half 的 16×16 转置块提取有效通道，并利用偏移表周期结构减少逐元素标量生成。

- [MX 矩阵加载的数据与 Scale 坐标](mx_data_and_scale_coordinates.md)：将逻辑元素、打包字节、NZ 分形和 Scale 载体分别建模，避免跨 dtype 复制 LoadData 偏移公式。

- [稀疏索引寻址与连续暂存的语义边界](sparse_indices_and_contiguous_staging.md)：分离逻辑索引、字节偏移、分块迭代和稀疏 KV 预合并，保留边界与写入归属。
- [按目标布局搬运与合并写回](strided_placement_and_batched_writeback.md)：在搬运阶段完成块位置重排，并按实际 UB 布局合并小结果写回，区分 stride、padding 和有效长度。

- [索引位宽、越界策略与动态输出形状](index_width_and_out_of_range_semantics.md)：补充 Gather/Scatter 索引窄化、空维度、字节偏移和多输出梯度形状的边界。

- [偏移表、分块 Scatter 与稀疏迭代进度](offset_tables_and_sparse_chunk_progress.md)：补齐索引表生成、块内连续性、分页缓存和 L1 chunk 前进条件。

- [分组 Head 输出到 NBSD 的块放置](grouped_head_output_placement.md)：给出分组 head、batch、query 起点的完整目标地址及普通搬运参数单位。
- [持久 KV NZ 布局与消费端合同](persisted_kv_nz_and_consumer_contracts.md)：区分真实预转换、L1/L0格式及页缓存地址，保留未对齐维度和来源性能案例。

- [稀疏预合并的物理大小与 Chunk 偏移](sparse_premerge_physical_sizes_and_chunk_offsets.md)：补齐 NZ 暂存容量、聚合中逐块搬运、超大块和每轮偏移重置。

- [LocalTensor 视图与 IoU 分量寿命](local_views_and_iou_component_lifetimes.md)：区分无搬运切片与非连续重排，保存示例中被覆盖的坐标与真正输出。

- [多 Head NZ 拼接的独立数据与 Scale 步长](multi_head_nz_diagnostic_slices.md)：保留 dtype 切换和逐 stage 定位细节，避免将 NaN 图样当成唯一根因。

- [MX 分组 Scale 广播与 FP4 字节打包](mx_block_scales_and_fp4_byte_packing.md)：保留四种转置的 Scale 物理排列、低位打包顺序和实际编码输入的 Golden。

- [小通道转置的 UB 预算、索引与 repeat 上限](small_channel_transpose_tile_budget_and_offsets.md)：修正 FP32 经 FP16 的精度边界及 4080/32 对齐冲突。

- [重排的偏移、广播行段与队列生命周期](shuffle_offsets_broadcast_runs_and_queue_lifetime.md)：保存规则重排方案并修正 Gather 源必须独立的泛化。

- [索引校验、前缀计数与分片归属](index_validation_prefix_counts_and_piece_ownership.md)：保存Argmax、Embedding、Scatter及插值边界的完整前提。

- [广播 Mask 极性、步长与暂存复用](broadcast_mask_polarity_strides_and_scratch_reuse.md)：纠正BA/AB方向及Select极性，保留Softmax布局。

- [Gather 尾轴模式与有符号索引规范化](gather_last_axis_modes_and_signed_index_normalization.md)：保留多行、单行和稀疏载入，限定负索引与取模。
- [MaskedScatter SIMT 前缀、快速除法与空 Mask](masked_scatter_simt_prefixes_magic_division_and_empty_masks.md)：保存线程分发，补齐前缀基数和未修改输出语义。

- [向量索引生成的精确性与位宽保留](vector_index_generation_exactness_and_width_preservation.md)：保留等差数列、商余数和负索引转换，同时限定浮点与重新解释。

- [重排路线启发式与连续搬运证据](conversion_route_heuristics_and_contiguous_burst_evidence.md)：保存NLast和片上转置的设计选择，限定固定阈值与性能归因。
- [小通道重排模型与 repeat 对齐上限](small_channel_conversion_model_and_repeat_alignment.md)：保留专用16通道中间布局，纠正任意perm规约和末次向上对齐。

- [高维Transpose的内层DMA与外层混合进制](transpose_big_dim_inner_dma_and_outer_mixed_radix.md)：保留大于5维的坐标拆分和尾块，限定状态化helper的复用。
- [Transpose 单轴切分布局、写出循环与尾部](transpose_cut_once_layouts_copyout_loops_and_tails.md)：保存CUT_ONCE地址推导及完整描述符条件。
- [Transpose融合的部分块与变长序列偏移](transpose_fusion_partial_blocks_and_variable_sequence_offsets.md)：补齐BNSD/TND布局写回、头尾重合与驻留状态条件。
- [Transpose SIMT索引、魔数除法与合并访问证据](transpose_simt_indices_magic_division_and_coalescing_claims.md)：保存2到8维映射，纠正单线程字节比推带宽和C循环自动向量化。
- [Transpose 策略ABI与同名字段的不同单位](transpose_strategy_abi_and_overloaded_tiling_fields.md)：保留四类POD及路由关系，防止跨策略机械复用字段。
- [Transpose 退化拷贝、别名与绑定队列容量](transpose_tensor_move_alias_and_bound_queue_capacity.md)：保存按位拷贝与q/r分核，限制满核和零开销推断。

- [N_LAST 合并搬运必须同时保持两端地址顺序](transpose_nlast_burst_merge_requires_both_address_orders.md)：纠正仅按输出轴相邻就增大blockLen的方案，并保留可测量的优化路径。
- [N_LAST 转置的多层DMA循环与物理容量](transpose_nlast_loop_strides_and_physical_tile_capacity.md)：保存末轴不变策略的源目的地址、尾块及对齐条件。
- [Transpose 双轴切分的四类任务与区间契约](transpose_two_cut_task_intervals_and_four_tail_classes.md)：保留双切分完整地址机制，补充零区间和混合进制边界。

- [021 转置的位宽地址表与H分核尾部](transpose_021_bitwidth_address_tables_and_hsplit_tails.md)：补齐8位双半块、32位拆地址及HSplit的读写边界。
- [5HD 二维转置的16位分块及主尾因子](transpose_5hd_16bit_tiles_and_main_tail_factor_contracts.md)：保存R/C/RC三种切法与16×16地址表，明确逻辑和物理长度。
- [Gather 转置索引的完整向量空间与窄整数](transpose_gather_index_vector_extent_and_narrow_arithmetic.md)：补充首向量无掩码写、相位间距及位宽转换的约束。
- [Gather 转置的相位布局与缓冲区计量](transpose_gather_phase_layout_and_buffer_accounting.md)：保留预生成索引、Compact输入与两端DMA循环的完整契约。
