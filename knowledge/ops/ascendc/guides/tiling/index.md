# tiling

- [多维广播的坐标映射与展开阶段](broadcast_coordinates_and_expansion_stage.md)：以各输入的广播模式和真实 stride 合轴，并比较 GM 搬入时展开与 UB 内展开的资源及接口边界。
- [线性切块的单位、核数与尾部范围](linear_partition_units_and_tails.md)：统一元素数与字节数，按正长度定义核内循环，并把核数选择与资源、同步及实际开销共同验证。

- [按实际代价划分不均匀工作量](cost_weighted_work_partition.md)：用可获得的稀疏度、有效长度和对齐开销估算工作量，区分行数均分、代价均分与实际核间耗时。

- [卷积的空间分核与分层资源预算](convolution_spatial_and_reduction_partition.md)：区分输出空间切分、通道归约和分组语义，协调 L1、L0、UB 与具体卷积实现的 Tiling 字段。

- [自适应容量分块与多行缓冲预算](adaptive_capacity_and_multirow_budget.md)：补齐按不同 dtype、固定预留、索引区和同时存活状态计算行数的具体方法。
- [代价前缀和分核的边界与局限](cost_prefix_partition_boundaries.md)：明确按工作代价划分连续行段时的端点、零代价行、不可拆分任务和二维核网格。
- [一维切分的完整覆盖与尾部搬运](one_dimensional_coverage_and_tail_transfers.md)：给出核间与核内余数公式，分开处理分块边界、物理对齐和实际有效长度。

- [对齐代价表与 CV 分块候选](aligned_cost_tables_and_cv_tile_candidates.md)：保存四类尾块代价和 Decode/GQA 分块启发式的具体形式，区分候选模型与实测结论。

- [对角线网格置换与边缘窗口](diagonal_grid_permutation_and_edge_windows.md)：保存 LCM 局部映射及其尾窗口越界反例，区分置换完整性和缓存收益。

- [直调 Add 的分核覆盖与 tile 地址](direct_add_tile_offsets_and_coverage.md)：从模板多 tile 重复读写反例提取完整分核、尾块、整数宽度和资源约束。

- [MX Tiling Stub 的字段关系与整数边界](mx_stub_fields_capacity_and_integer_bounds.md)：解析示例基块、Scale跨度与缓冲参数，保留 mock 未完成的硬件与整数验证。

- [Tiling 整数范围证明与守卫语义](integer_range_proofs_and_guard_semantics.md)：修正取绝对值、类型提升、ceildiv 和尾部减法示例，区分算术安全与参数合同。

- [直调单目模板的分区、UB 与类型合同](direct_unary_template_partition_and_buffer_contracts.md)：保留 Kernel/Tiling 示例可复用逻辑，纠正固定常量、尾块和同步红线的泛化。

- [形状特化的分发与共享资源边界](shape_specialization_dispatch_and_shared_resources.md)：保留多 shape 路径隔离方法，限定默认分支保护与 TilingKey 的版本约束。

- [注册 Add 的分区、空输入与队列缓冲语义](registry_add_partition_empty_and_queue_buffer_semantics.md)：抽取双输入分块算法并修正空路径除零、粒度单位与模板键假设。

- [动态 UB Broadcast 的重载与存储约束](dynamic_broadcast_overloads_and_storage_contracts.md)：区分动态 shape 广播接口和既有静态接口，保留容量、地址与空输入边界。
- [GMM 不规则分组任务与资源单位](grouped_matmul_ragged_tasks_and_resource_units.md)：保留分组、量化、缓存和调度模型，修正任务数及预算公式。
- [Matmul 分层 Tiling、Batch 广播与 Split-K 边界](matmul_hierarchy_batch_and_split_k_corrections.md)：保留缓存/核/基础块模型，修正广播、分配和转置偏移公式。

- [DSL 转换的分核、尾部与阶段 ABI](dsl_partition_and_stage_abi.md)：保存余数分配公式，修正 Host 与 Kernel 示例版本不一致。

- [多轴尾块、池化索引与阶段归属](multiaxis_tails_pool_indices_and_phase_ownership.md)：保存BatchNorm及SparseToDense切分，纠正索引单位。

- [自适应算法的退化域与 Fallback](adaptive_algorithm_degenerate_domains_and_fallbacks.md)：保存多路径选择，纠正D为1直通和未执行Tiling的路径。

- [自适应分块搜索、可行性与安全容量](adaptive_tiling_search_capability_and_safe_capacity.md)：保留多策略选择及二维搜索，明确示例谓词、启发式和失败回退。
- [Attention 稀疏块与分组矩阵窗口映射边界](attention_sparse_tiles_and_grouped_window_mapping_errors.md)：保留基本块对齐和窗口化思想，指出尾窗口反向扫描的代码反例。
- [缓冲容量公式与分支边界](buffer_capacity_formulas_and_branch_boundaries.md)：保存量化及归一化的细粒度预算，并修正单位和下溢陷阱。
- [分层分块、参数复用与 Foreach 余数](hierarchical_tiles_parameter_reuse_and_foreach_remainders.md)：保存核、UB、循环层的数据划分，纠正整块数量与尾块位置。

- [分核数量、尾部成本与阶段参与者](core_partition_counts_tail_cost_and_phase_participants.md)：区分差一份的余数分配和末核余量，保留两阶段稀疏写入合同。
- [多轴任务组、Stream-K 与尾块再分裂](multiaxis_work_groups_streamk_and_tail_split_limits.md)：保存层次化并行，并补齐分组索引与跨核合并边界。

- [Broadcast 合轴、路由与字段一致性](broadcast_collapse_routing_and_field_consistency.md)：保留广播的输入坐标和分阶段展开，指出fallback返回值缺口。
- [Broadcast fallback 尾块公式与缩块状态](broadcast_fallback_tail_equations_and_shrink_state.md)：保留从内向外切分，修正尾块变量错用和未同步返回字段。
- [Elementwise fallback 字节、核数与精度字段](elementwise_fallback_bytes_cores_and_precision_fields.md)：提炼脚本分块模型，保留空输入、容量强行回退和dtype别名反例。
- [Tiling 搜索参数角色与语义约束](tiling_search_parameter_roles_and_semantic_constraints.md)：从参数分析流程中抽取可复用约束，纠正只按常量来源放宽搜索。

- [MC2 fallback 尾块大小与候选搜索边界](mc2_fallback_tail_size_and_candidate_search_limits.md)：保存整数切分启发式，修正短块比长块大和不足16行的情况。

- [广播模板容量、分核与DMA有效长度](broadcast_template_capacity_partition_and_dma_limits.md)：保存通用切分等式，指出全shape可放时内层积与保底容量问题。
- [广播模板选路与每tile有效shape](broadcast_template_routes_and_per_tile_shapes.md)：保留外层offset广播、NDDMA与UB展开的能力区别。
- [卷积 fallback HW字段与乒乓标志编码](convolution_fallback_hw_fields_and_pingpong_encoding.md)：记录通道/K混用、innerBatch和标志位的CPU反例。
- [卷积 fallback 搜索回退与资源单位](convolution_fallback_search_rollback_and_resource_units.md)：保留L0/L1分块策略，纠正无合法最小块时继续返回参数。
- [卷积分组合并与多核代价模型范围](convolution_group_enlargement_and_multicore_cost_scope.md)：保存合组因子和分核候选，指出组数与输出通道混用。
- [OneDim 广播tile数与标量驻留](onedim_broadcast_tile_counts_and_scalar_residency.md)：纠正负尾核循环，限定扁平化与标量一次填充的前提。

- [Softmax Host容量公式与空维失效条件](softmax_host_capacity_formulas_and_empty_shape_failures.md)：保存六种分配成本，纠正强制最小tile和注释中的低估。
