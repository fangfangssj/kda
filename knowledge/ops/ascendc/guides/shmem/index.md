# shmem

- [Ascend C SHMEM 的对称内存、地址可达性与完成语义](symmetric_memory_and_completion.md)：区分 PE 对称堆、远端地址映射、RMA 提交、完成和通知，说明不同通信引擎的地址约束需要独立核对。
- [Ascend C SHMEM 的 PE、核与 Tile 分工](work_partition_and_pipeline.md)：给出对称通信工作划分的整除条件、余数分配和缓冲交接不变量，避免把示例核比例与同步编号当作通用常量。

- [SHMEM 初始化的依赖、会话与失败证据](bootstrap_dependencies_and_failure_evidence.md)：将库加载、bootstrap、PE 身份、对称堆和 Kernel 执行分层定位，避免把初始化失败误判为算子精度错误。
- [SHMEM 归约中的贡献计数与目标写权](reduction_contribution_ownership.md)：区分本地初值、远端贡献、原子累加与独占归约，明确分组并行和对称缓冲复用的条件。

- [SHMEM Bootstrap、Team 与公开接口分层](bootstrap_teams_and_public_interface_layers.md)：补充默认、MPI、UniqueID 初始化，Team 编号转换及对称分配的接口角色。
- [CoC Matmul Tile、累加器与 Workspace 交接](coc_matmul_tiles_accumulators_and_workspace_handoffs.md)：补全计算分块、SplitK、Bias/Fixpipe 及通信计算槽位轮换的前提。
- [SHMEM Dump ABI 与局部观测边界](dump_abi_and_observation_limits.md)：补充调试宏造成的入口变化、DumpTensor 有效范围和日志/周期采集限制。
- [SHMEM 拓扑并发与硬件参数的适用边界](hardware_topology_and_measured_capacity_boundaries.md)：保留链路并行度和资源查询方法，纠正跨平台硬件表与峰值带宽混用。
- [SHMEM 布局单位、尾块地址与归约复位](layout_units_tail_partition_and_reduction_reset.md)：补入对称头部单位换算、MoE 元数据索引、peer 分工和多轮原子结果初始化约束。
- [SHMEM 环境来源、多 PE 失败传播与 Torch 加载](runtime_environment_and_multi_pe_failure_propagation.md)：提取独立工程依赖、容器环境、启动进程回收及共享扩展的兼容条件。
- [SHMEM 数据面引擎与完成域](transport_engines_and_completion_domains.md)：补全 RMA 族、UB 生命周期、quiet/sync/barrier 差别及数据先于通知的条件。

- [Blaze Heavy Kernel 覆盖与性能条件](blaze_heavy_kernel_coverage_and_perf_conditions.md)：Blaze Heavy Kernel 覆盖与性能条件的来源实现、适用条件与审视结论。
- [Blaze-SHMEM Host Bootstrap 生命周期与失败传播](blaze_host_bootstrap_lifetime_and_failure.md)：Blaze-SHMEM Host Bootstrap 生命周期与失败传播的来源实现、适用条件与审视结论。
- [Blaze 本地 MX 模板的 L1 尺度窗口与事件](blaze_local_l1_scale_windows_and_events.md)：Blaze 本地 MX 模板的 L1 尺度窗口与事件的来源实现、适用条件与审视结论。
- [Blaze 本地 MX Scale 加载与 K 尾填充](blaze_local_mx_copy_and_tail_padding.md)：Blaze 本地 MX Scale 加载与 K 尾填充的来源实现、适用条件与审视结论。
- [Blaze 本地 MX 模板的 SplitK 哨兵与原子输出](blaze_local_splitk_sentinels_and_atomic_output.md)：Blaze 本地 MX 模板的 SplitK 哨兵与原子输出的来源实现、适用条件与审视结论。
- [Blaze 随附本地模板与 Toolkit 接口的区别](blaze_local_templates_are_separate_interfaces.md)：Blaze 随附本地模板与 Toolkit 接口的区别的来源实现、适用条件与审视结论。
- [Blaze 性能采样的 Rank 身份与聚合](blaze_perf_rank_identity_and_aggregation.md)：Blaze 性能采样的 Rank 身份与聚合的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 样例的构建依赖身份](blaze_sample_build_dependency_identity.md)：Blaze-SHMEM 样例的构建依赖身份的来源实现、适用条件与审视结论。
- [Blaze-SHMEM Cache Hint 与复用范围](blaze_sample_cache_hints_and_reuse_domain.md)：Blaze-SHMEM Cache Hint 与复用范围的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 编码 Golden 与精度判定](blaze_sample_encoded_golden_and_tolerance.md)：Blaze-SHMEM 编码 Golden 与精度判定的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 文件 IO、工作目录与转储身份](blaze_sample_io_paths_and_dump_identity.md)：Blaze-SHMEM 文件 IO、工作目录与转储身份的来源实现、适用条件与审视结论。
- [Blaze-SHMEM MX 分 rank 尺度填充与物理布局](blaze_sample_mx_rank_scale_padding.md)：Blaze-SHMEM MX 分 rank 尺度填充与物理布局的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 外层 M 尾部与 Tiling ABI](blaze_sample_outer_m_tail_and_tiling_abi.md)：Blaze-SHMEM 外层 M 尾部与 Tiling ABI的来源实现、适用条件与审视结论。
- [Blaze-SHMEM Rank 累加、Self 路径与 Bias](blaze_sample_rank_accumulation_and_bias.md)：Blaze-SHMEM Rank 累加、Self 路径与 Bias的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 环形槽、释放信用与 Flag 预算](blaze_sample_ring_credit_and_flag_budget.md)：Blaze-SHMEM 环形槽、释放信用与 Flag 预算的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 路线约束与集合通信语义](blaze_sample_route_constraints_and_collective_semantics.md)：Blaze-SHMEM 路线约束与集合通信语义的来源实现、适用条件与审视结论。
- [Blaze-SHMEM Scale 先行发送与完成链](blaze_sample_scale_put_completion_chain.md)：Blaze-SHMEM Scale 先行发送与完成链的来源实现、适用条件与审视结论。
- [Blaze-SHMEM 对称地址、源目标偏移与容量](blaze_sample_symmetric_offsets_and_capacity.md)：Blaze-SHMEM 对称地址、源目标偏移与容量的来源实现、适用条件与审视结论。
- [Blaze 本地 SWAT 调度坐标与尾块覆盖](blaze_swat_scheduler_coordinates_and_tail_coverage.md)：Blaze 本地 SWAT 调度坐标与尾块覆盖的来源实现、适用条件与审视结论。
- [Blaze SWAT Tiling 搜索与资源边界](blaze_swat_tiling_search_and_resource_bounds.md)：Blaze SWAT Tiling 搜索与资源边界的来源实现、适用条件与审视结论。
