# attention

- [分块 Attention 的 online softmax 状态与归并](online_softmax_state_and_partition_merge.md)：保持 max、指数和与未归一化加权和一致，正确处理跨 KV 分块缩放、Split-KV 和分块 softmax 输出。

- [Attention 的逻辑 token、head 与物理缓存地址](logical_tokens_heads_and_cache_addresses.md)：分开处理 TND batch 边界、GQA head 映射、稀疏 token 选择与 page table，防止布局组合后混用坐标。
- [Attention 打分维度与 Value 输出维度](score_dimensions_and_value_dimensions.md)：在 MLA 等分段输入路径中分别管理 QK 打分维度、Value 宽度及 latent 数据消费范围。
- [Attention 的 score 可见性与额外 KV 来源](score_visibility_and_additional_kv_sources.md)：区分 causal/显式 mask 与额外 sink KV，保证有效列集合和 online softmax 的统一归一化。

- [线性 Attention 的分组 head 缓存与梯度归属](grouped_head_cache_and_gradient_ownership.md)：按真实数据依赖限定 HK 缓存，保持跨 stage 的 workspace 映射，并区分共享输入与需累加输出。

- [CV 交错流水的任务槽与数据槽](cv_pipeline_task_and_data_slots.md)：明确 MM1、Vec1、MM2、Vec2 的三期排程、索引范围和跨核生产消费及回收关系。

- [Online Softmax 状态槽的接续与最终选择](online_state_slots_and_final_selection.md)：明确 max、sum、缩放因子和输出累加器的任务身份，避免双槽递推丢失或重复计数。
- [Workspace 槽地址与延迟梯度 Scatter](workspace_slots_and_deferred_gradient_scatter.md)：区分阶段槽、梯度聚合域和最终刷新时机，保留完整 dK/dV 贡献。

- [G 轴分工与变长 KV 线段切分](g_axis_partition_and_variable_length_segments.md)：补齐核号映射、零维、贪心超额分核及独立输出和 KV 归约的区别。

- [双 AIV 行范围与 Workspace 份数](dual_aiv_row_ranges_and_workspace_multiplicity.md)：保存对齐分工公式、空消费者与物理槽容量，避免混淆核比例和核总数。
- [稀疏与稠密分发需要有效集合等价](sparse_dense_dispatch_requires_set_equivalence.md)：区分块数量预算和实际 token 覆盖，保留分发后的掩码与资源合同。

- [反向 Cube 融合需保留全部梯度路径](backward_cube_fusion_preserves_all_gradients.md)：区分 dS 与概率矩阵，补齐累加器、转置输入和输出写回的寿命。
- [Sink 任务与跳块后的流水状态](sink_tasks_and_skipped_block_state.md)：补齐 sink 与 KV 的集合关系、任务字段及 online 状态跨任务归并。
- [Softmax 反向行点积的预计算](softmax_backward_row_dot_precomputation.md)：保存梯度公式、每 query 行结果容量及 dY/O 版本一致性。

- [Decode 拼接搬运与 online tile 归约](decode_concat_dma_and_online_tiles.md)：融合 nope/rope 的物理暂存，并保存两级点积、有效域与在线 softmax 状态。
- [共享 KV head 的 Decode GEMV 合并](grouped_decode_gemv_to_gemm.md)：以同组 query heads 为矩阵 M 维，保留分组、数值稳定性、任务并行和分块累加。

- [因果窗口、GQA 与整块跳过的证明](causal_windows_gqa_and_block_skip_proofs.md)：补全闭区间边界、query 坐标和空行语义，限定 Attention 性能案例。

- [Attention 特性的可见集合与量化交互](features_sparse_visibility_and_quantization_interactions.md)：抽取 causal/window/sink、稀疏索引、RoPE 和后量化的组合边界。
- [GQA 任务几何与 Split-KV 并行条件](gqa_task_geometry_and_split_kv_capacity.md)：区分 head 分组、查询行、物理 stride 和可用核数。
- [MLA 扩展假设与 Attention 特化范围](mla_extension_hypotheses_and_specialization_scope.md)：保留扩展点而不把未验证路线写成平台能力。
- [Attention 在线 Softmax 的空行与量化阶段合同](softmax_empty_rows_and_quantized_stage_contracts.md)：修正有限 mask、初始 rescale 及 QK/PV 归约轴混淆。
- [Attention Tiling 的完整内存预算与流式边界](tiling_memory_budget_and_streaming_limits.md)：按物理布局和同时存活对象核算 UB、L1、L0 与 workspace。

- [Attention 模式组合保持逻辑 Token 身份](pattern_composition_preserves_logical_token_identity.md)：从导航提炼变长、分组、分页、稀疏及可见性的组合条件。

- [Sage 缓冲布局与容量冲突检查](sage_buffers_layouts_and_capacity_conflicts.md)：保存GM和片上地址合同，揭示源布局表中的重叠与扩容缺口。
- [Sage 编译 Helper 与 API 适用范围](sage_compiler_helpers_and_api_scope.md)：保留地址空间、寄存器和向量辅助实现的条件。
- [Sage 游标 Split-KV 与归并边界](sage_cursor_split_merge_and_mask_boundaries.md)：保存任务首步、完整状态槽与因果位置合同。
- [Sage 分层取证与证据边界](sage_diagnostics_and_evidence_limits.md)：保留拟合和中间量探针，避免余弦相似度与症状唯一归因。
- [Sage 四阶段流水与版本事件所有权](sage_pipeline_flags_and_version_ownership.md)：保留mode4槽位协议，限定Bootstrap与双重等待结论。
- [Sage 量化与 K 平滑补偿合同](sage_quantization_and_smoothing_contract.md)：保存INT8 QK与FP16 PV路径，修正精确性和类型转换推断。

- [A2 FA Stage 资源与组件适用范围](fa_a2_stage_resources_and_component_scope.md)：保存四段流水和轮转容量，修正平台与管线泛化。
- [FA 性能输入身份与覆盖分布](fa_benchmark_inputs_and_coverage_distribution.md)：保存结构张量生成、长度分桶及多batch故障探测。
- [FA 复合 Causal Varlen 与 Sink 数学](fa_compound_causal_varlen_and_sink_math.md)：修正局部列坐标、有限Mask与Sink分子缩放。
- [手写 FA 缓冲与状态故障模式](fa_handcrafted_buffer_and_state_failures.md)：保留多任务覆盖和事件问题，修正共享UB与最终Max公式。
- [FA 布局与 SeqLen 接口变体](fa_layout_and_seqlen_interface_variants.md)：区分BNSD、TND、分页格式和两类长度数组。
- [FA 分页与量化操作数合同](fa_paged_and_quantized_operand_contract.md)：补齐物理Head偏移、尾页和反量化尺度。
- [FA 性能采集范围与汇总口径](fa_performance_capture_scope_and_aggregation.md)：区分全 forward、Event 与 device task，按已声明完整段选择同源整版；防止有利窗拼接、汇总指标替换及归档指针漂移。
- [FFA TensorList 前缀与组件合同](ffa_tensorlist_prefix_and_component_contract.md)：区分拼接KV和双KV对角模型，保存FD LSE与量化入口语义。
- [Varlen L1 Carry 与 SameAB 约束](varlen_l1_carry_and_sameab_constraints.md)：保存驻留优化条件，修正跨batch无条件复用与容量常量。
- [Varlen 任务前缀与 Sparse 边界](varlen_task_prefixes_and_sparse_bounds.md)：保存四类累计量、TND偏移与尾块双重约束。

- [FA 优化归因与内存证据](fa_optimization_attribution_and_memory_evidence.md)：纠正Codegen唯一根因和大输入容量估算。
- [FA Q Head 打包与 Block 生命周期](fa_q_head_packing_and_block_lifetimes.md)：保存GQA行映射和组件内部槽位，限制硬事件照抄。
- [FAQK 交接与 Vector Count 约束](faqk_handoffs_vector_counts_and_claim_limits.md)：保存Token复用和批量布局知识，限制器件故障泛化。
- [FFA 正交 Batch 映射与流式状态](ffa_orthogonal_batch_mapping_and_streaming.md)：保存双KV对角模型的四路GEMM和完整平面偏移。
- [KFC Chunk 原子累加与尾部合同](kfc_chunk_atomic_rescale_and_tail_contract.md)：保留Prescale顺序和双缓冲状态，限定尾块性能案例。
- [KFC Launch ABI 与 Matmul 异步范围](kfc_launch_abi_and_matmul_async_scope.md)：保留系统区和用户区分离，修正参数数目和硬编码地址窗口。
- [S1 Sink Softmax API 与缓冲复用](s1_sink_softmax_api_and_buffer_reuse.md)：保存播种法和状态写回，纠正Tiling配置及行列循环混淆。
- [S1 三槽流水与排空身份](s1_three_slot_pipeline_and_drain_identity.md)：修正模3时间标签和伪任务索引，保留L1Reuse配对。
- [S1 Workspace 单位与 Tiling 候选](s1_workspace_units_and_tiling_candidates.md)：保存三段双缓冲公式，限定资源覆盖和缓存Shape条件。

- [MLA 延迟 PV 槽位与跨任务状态](mla_delayed_pv_slots_and_cross_task_state.md)：保存全局和局部PingPong、前取与事件所有权。
- [MLA FD Partial 表示与容量](mla_fd_partial_representation_and_capacity.md)：区分归一化Partial和指数和，补齐全核发布及数据类型。
- [MLA Latent 组件与特化合同](mla_latent_component_and_specialization_contract.md)：保存两段Score、V复用与模板差异，限定历史适用域。
- [MLA Split 规划与任务一一映射](mla_split_planning_and_task_bijection.md)：保存贪心切分与双解码，补齐页容量及蛇形一致性。
- [MLA Tiling Word 偏移与字段冲突](mla_tiling_words_offsets_and_field_conflicts.md)：保存400加17布局，纠正Float数组和绝对地址混淆。
- [Paged FA 组件布局与 ABI 修正](paged_fa_component_layout_and_abi_repairs.md)：保存三项关键修复，补齐GQA基址和声明宏问题。
- [Paged FA 延迟 Finalize 与 Mask 限制](paged_fa_delayed_finalize_and_mask_limits.md)：保存排空和Tail调用，纠正任意Mask套Causal及无符号边界。

- [Softmax 在线重标定与有限 Mask 限制](softmax_online_rescale_and_finite_mask_limits.md)：保存分块状态和压缩偏移，限定高阶API精度声明。

- [Attention 在线输出归一化与最终除法必须一致](attention_normalized_output_and_final_division_must_agree.md)：纠正设计中的二次除法，保存mask、分核和内存尺度条件。

- [推理 RunInfo 初始化与 Query/KV 坐标读取](inference_runinfo_initialization_and_coordinate_reads.md)：从 FA 模板的条件初始化与消费路径识别潜伏未初始化字段，按推理语义建立默认值及逐实例验证。
