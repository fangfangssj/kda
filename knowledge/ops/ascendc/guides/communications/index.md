# communications

- [Ascend C HCCL 通道与 Hcomm 完成边界](hccl_channels_and_hcomm_completion.md)：连接 Host 通信域、通道和引擎上下文的所有权与 Device 提交、完成、跨核通知之间的依赖。

- [MC2 本地 Matmul 与通信依赖的重叠](local_matmul_and_communication_dependencies.md)：将不依赖远端输入的本地计算从通信块循环中独立调度，并保持贡献归属、输出位置及独立 Tiling 合同。

- [通信上下文 ABI 与窗口地址的分层](context_abi_and_window_address_layers.md)：区分上下文对象、按rank的窗口基址和业务slot偏移，避免用精简样例结构解析真实HCCL资源。

- [HCCL commit 参数与逐对 rank 计数](hccl_commit_flag_and_peerwise_counts.md)：纠正阻塞参数误读、AlltoAllV 总和检查和 MoE 路由边界的不足。

- [MC2 变长块分发、偏移与完成协议](mc2_variable_tile_dispatch_offsets_and_completion.md)：保存实现层结构，补齐零短块字段、FIX和远端完成约束。

- [APACE 先计算后通信的实现边界与失效信号](apace_compute_first_review_findings_and_limits.md)：保存Fragment映射、归约分工及事件失效经验，限定数值阈值。
- [APACE 组件版本与通信内存归属](apace_versioned_components_and_memory_ownership.md)：从工作流和审查项抽取接口及资源事实，保留版本范围。
- [集合通信不自动决定切轴、PUT/GET与求和](collectives_do_not_determine_axes_put_get_or_reduction.md)：纠正AllToAll示例中的分布混用，保存四种原语的数学边界。
- [MC2 bias、量化次序与接口合同](mc2_bias_quantization_order_and_interface_contracts.md)：从需求文档抽取影响正确性的参数语义，排除追问流程。
- [MC2 路线坐标与能力证据范围](mc2_route_coordinates_and_capability_evidence_scope.md)：区分通信资源分配、设备执行路径和样例验证范围。

- [APACE AllToAll Matmul 的全局与局部坐标](apace_alltoall_matmul_global_local_coordinates.md)：APACE AllToAll Matmul 的全局与局部坐标的来源契约、适用范围和失效边界。
- [APACE barrier 钩子与同通道完成顺序](apace_barrier_hooks_and_ordered_channel_completion.md)：APACE barrier 钩子与同通道完成顺序的来源契约、适用范围和失效边界。
- [APACE 通信游标单位与末轮 Wait](apace_comm_cursor_units_and_last_wait.md)：APACE 通信游标单位与末轮 Wait的来源契约、适用范围和失效边界。
- [APACE 先计算后通信的三级流水生命周期](apace_compute_first_three_stage_lifetimes.md)：APACE 先计算后通信的三级流水生命周期的来源契约、适用范围和失效边界。
- [APACE context 建链复用与窗口归属](apace_context_creation_reuse_and_window_ownership.md)：APACE context 建链复用与窗口归属的来源契约、适用范围和失效边界。
- [APACE GET/PUT 地址与尾块等价条件](apace_get_put_address_and_tail_equivalence.md)：APACE GET/PUT 地址与尾块等价条件的来源契约、适用范围和失效边界。
- [APACE localMatmul 累加模式与 L0C 容量](apace_local_matmul_accumulation_modes_and_l0c.md)：APACE localMatmul 累加模式与 L0C 容量的来源契约、适用范围和失效边界。
- [APACE 批量归约的 UB 与事件生命周期](apace_reduce_ub_batching_and_event_lifecycle.md)：APACE 批量归约的 UB 与事件生命周期的来源契约、适用范围和失效边界。
- [APACE 归约组合与远端累加初值](apace_reduction_composition_and_initial_values.md)：APACE 归约组合与远端累加初值的来源契约、适用范围和失效边界。
- [APACE TeamBarrier 的 epoch 槽与参与者](apace_team_barrier_epoch_slots_and_participation.md)：APACE TeamBarrier 的 epoch 槽与参与者的来源契约、适用范围和失效边界。
- [APACE 通信切分与计算 tiling 的一致性](apace_tiling_alignment_and_partition_consistency.md)：APACE 通信切分与计算 tiling 的一致性的来源契约、适用范围和失效边界。
- [APACE 等待掩码与 pipe 等待范围](apace_waited_mask_and_pipeline_wait_scope.md)：APACE 等待掩码与 pipe 等待范围的来源契约、适用范围和失效边界。

- [APACE AllGather 预触发与 scale 等待边界](apace_allgather_pretrigger_and_scale_wait_conflicts.md)：APACE AllGather 预触发与 scale 等待边界的来源实例与适用边界。
- [APACE AllGather 三区与输出归属](apace_allgather_three_regions_and_output_ownership.md)：APACE AllGather 三区与输出归属的来源实例与适用边界。
- [APACE AllToAll 变体组合边界](apace_alltoall_variant_composition_boundaries.md)：APACE AllToAll 变体组合边界的来源实例与适用边界。
- [APACE 入口 ABI 与 tiling 消费者](apace_entry_abi_and_tiling_consumers.md)：APACE 入口 ABI 与 tiling 消费者的来源实例与适用边界。
- [APACE GET AllGather 设想与槽释放证明](apace_get_allgather_proposal_and_release_proof.md)：APACE GET AllGather 设想与槽释放证明的来源实例与适用边界。
- [APACE ReduceScatter 轮次坐标与舍入点](apace_reduce_scatter_round_coordinates_and_rounding.md)：APACE ReduceScatter 轮次坐标与舍入点的来源实例与适用边界。

- [MC2 Communication (compute-communication fusion) 域模式：HcclResource / CrossRankSync / Window RDMA / MakeGlobalTensor](mc2_communication_fusion_patterns.md)：MC2 多 rank 计算-通信融合 kernel 通信参考：HCCL context 生命周期、跨 rank 软屏障、window RDMA 与 GM 对齐标量 I/O（通信路径锁定 Path B1；内嵌 P-P110~P-P113 四个模式）。
