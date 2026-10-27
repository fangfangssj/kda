# performance

- [通信与通算融合的计时边界和带宽口径](communication_timing_and_bandwidth.md)：对齐 SHMEM 与基准实现的数据边界，区分多卡统计、任务时长与调用跨度，并明确带宽分子的单位和含义。
- [Profiler 指标口径与瓶颈假设](profiler_metrics_and_bottleneck_hypotheses.md)：从任务身份、采集范围和原始单位解释流水指标，区分利用率信号、理论估算与可验证的瓶颈原因。
- [SHMEM Cycle 打点的采集范围与扰动](shmem_cycle_observation_scope.md)：说明 PE、Block、Frame 和 Count 的统计含义，检查异步完成边界及插桩对流水重叠的影响。

- [MC2 切分开销与通信等待归因](mc2_partition_overhead_and_wait_attribution.md)：联合考察通信粒度、Matmul 利用率与依赖等待，避免仅按切分数或 MTE2 占比推断 MC2 性能瓶颈。

- [L2 Tensor 策略与页表语义的独立性](l2_tensor_policy_and_page_table_independence.md)：保留选择性缓存的可测条件，区分框架标志、实际访存 hint 和分页地址映射。
- [Scalar 头部时间与真实访存指令](scalar_prologue_and_actual_memory_operations.md)：区分 Device 初始化、Host 启动开销及编译器寄存器分配，避免按 C++ 存储形式推断性能。

- [GM 访问错峰与真实地址对齐](gm_access_rotation_and_physical_alignment.md)：保持任务置换、输出映射及生产消费布局，区分性能对齐与合法搬运要求。
- [KV 工作集估计与 L2 策略候选](kv_cache_footprint_for_l2_candidates.md)：补齐 K/V 双份容量、布局 padding、溢出和配置到实际访存的传递。

- [算子族选型的结构与测量条件](family_selection_requires_measured_structure.md)：把族名和固定阈值变为候选线索，保留每类算子的实际性能约束。

- [批处理与缓存复用优化的语义前提](batching_reuse_and_capacity_preconditions.md)：修正逐行 Sub 展平、16KiB 下限和双缓冲检查的过度推论。

- [SHMEM 基线范围与重复测量的可比性](shmem_baseline_scope_and_repeated_measurements.md)：补充 fused/stitched/proxy 基线、inline 与 msprof 关联及状态复位成本。

- [HAP 评分定义域与真实执行的可观测性](hap_score_domain_and_execution_observability.md)：保留基线偏移评分公式，明确无效分母、执行范围和评测保护的局限。

- [VF 融合上限与资源证据](vf_fusion_limits_and_resource_evidence.md)：区分计算调用数、实际指令、寄存器压力和融合收益，保留模板中的成本因素。

- [优化经验中的硬件模型与缓冲份数](evolution_hardware_models_and_buffer_counts.md)：限定示例规格、Roofline 和 DMA 经验阈值，纠正流水份数与自动重叠解释。
- [msprof CSV 字段、单位与诊断边界](msprof_csv_fields_units_and_diagnostic_limits.md)：保存八类 CSV 的可用指标及交叉分析方式，纠正单阈值根因判定。
- [优化实验的证据与假设范围](optimization_experiment_evidence_and_hypothesis_scope.md)：从搜索工具的资料中抽取测量与推理知识，保留未知值、失败归因和收益上界。

- [Attention 的 Cube、Vector 与内存性能下界](attention_three_channel_roofline_and_traffic.md)：统一有效 FLOP、实际流量与阶段重叠的计量。

- [msprof 采集模式与结果身份](msprof_capture_modes_and_trace_identity.md)：保存真机、仿真和CSV时间线交叉分析，限制示例CLI与空字段解释。
- [优化证据与预测结论的区分](optimization_evidence_and_model_claims.md)：从工具设计中提取可复用的实验记录语义，剥离架构和流程。
- [优化标签与规则适用性的边界](optimization_tags_and_rule_applicability.md)：保存执行单元、症状和上下文的区分，防止相似度代替硬约束。
- [性能目标的方向、单位与基线](performance_targets_and_relative_units.md)：明确时延降低、加速比、百分点及复合目标的数学含义。

- [小行批处理与排序键精度](small_row_batching_and_sort_key_precision.md)：保存SingleN和多行策略，补齐索引转Float的失真边界。

- [MC2 计时、轮次身份与抖动统计](mc2_measurement_round_identity_and_jitter_statistics.md)：提炼多卡时空波动分析，纠正按CSV行序配对和固定预热剔除。
- [MC2 流水递推与错误配平剪枝](mc2_pipeline_recurrence_and_invalid_balance_pruning.md)：保存非均匀切分搜索，修正会排除全部非零尾块的约束。
- [流水瓶颈证据与向量指令说明边界](pipeline_bound_evidence_and_vector_instruction_claims.md)：保留访存、Scalar、Vector和低利用率诊断，修正计数与API推断。

- [Scalar 首尾剥离与卷积事件配对](scalar_loop_peeling_and_convolution_event_pairs.md)：指出首块加载遗漏、单次重复计算及方向不配对的模板。
- [Scalar 九条原则的编译器条件](scalar_nine_principles_as_compiler_hypotheses.md)：完整保留九类改写选择，限定别名、常量传播与spill的推断。

- [MC2 缓存状态控制与预热的区别](mc2_cache_conditioning_is_distinct_from_warmup.md)：保留flush实现思路和trace检查，限制必刷缓存与固定阈值。
- [MC2 多rank统计口径与重叠指标](mc2_multirank_aggregation_and_overlap_metrics.md)：保存来源两种统计政策，防止平均值和实际最慢轮混用。
- [MC2 tile数量、轮转槽与flag积压](mc2_tile_count_ring_slots_and_flag_backlog.md)：区分工作块数和在途容量，补齐tiling变化与正确性条件。

- [APACE Host 参数接线与实际计时](apace_host_parameters_and_observed_timing.md)：APACE Host 参数接线与实际计时的来源实例与适用边界。
- [APACE 调优假设与负收益的适用范围](apace_performance_tuning_hypotheses_and_failures.md)：APACE 调优假设与负收益的适用范围的来源实例与适用边界。

- [TilingData 紧凑布局、字段位宽与 ABI](compact_tiling_data_and_host_device_abi.md)：以字段值域、物理布局、序列化和消费者一致性约束 TilingData 压缩，验证真实加载成本而非只看字段数量。
- [无 VF 计算的 DMA 算子重叠与性能判读](dma_only_kernel_overlap_and_latency.md)：解释没有 VF 计算时 MTE/VEC 重叠指标失效的原因，并区分时延收益与双缓冲机制证据。
- [npusim summary 字段、采样范围与缺失值](npusim_summary_fields_and_capture_scope.md)：解释 npusim summary 的时间、核范围、流水活动和可选计数，避免把缺失或采样子集当作硬件事实。
- [Profiler 的 Block、dispatch 与物理核绑定](profiler_block_dispatch_and_physical_core_identity.md)：明确特定 Profiler 报告中 Block、dispatch、物理核和有向访存边的关联键，避免统计错绑。
- [SIMD VF 指标的分母、发射停顿与代码形态](simd_vf_metric_denominators_and_issue_stalls.md)：区分 SIMD VF 的调用数、工作量、发射间隙和累计时长，约束代码形态诊断所依赖的统计范围。
- [流水线空泡的区间并集、单位与因果证据](trace_interval_union_and_bubble_attribution.md)：用事件区间并集计算忙闲和重叠，以核型映射、真实单位和依赖证据约束空泡归因。

- [torch_npu.profiler 的 op_statistic 聚合与采集窗口](torch_npu_profiler_statistic_aggregation.md)：解释 Total Time(us) 求和、实际采集步数归一化、repeat 导出选择和 CSV 身份，区分 OP 累计时间与墙钟耗时。

- [asys 系统采样的参数范围与 msprof 下发身份](asys_system_profiling_scope.md)：区分系统级采样和算子级度量，保留采集类型、单设备参数、PMU 条件及周期/时长表述的核验边界。
