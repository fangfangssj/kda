# moe

- [MoE Dispatch 与 Combine 的元数据对应关系](dispatch_combine_metadata.md)：将特征、来源三元组、收发计数和量化 scale 按同一有效条目顺序关联，区分最大容量与本轮实际 token 数。
- [MoE 通信窗口的数据发布、状态消费与复用](window_publication_and_reuse.md)：区分数据区与状态区的索引布局，按先完成数据、再发布状态、消费后复用的依赖管理多轮通信窗口。

- [MoE 多核分阶段工作项与全局输出偏移](stage_partition_and_global_output_offsets.md)：区分发送token、发布状态槽和回搬记录的分核域，用局部计数与前缀偏移构造无重叠的全局输出。

- [MoE 容量公式与路由假设](moe_capacity_formulas_and_routing_assumptions.md)：MoE 容量公式与路由假设的接口、实现假设及失效边界。
- [MoE Combine 的可选路径与量化角色](moe_combine_optional_paths_and_quantization_roles.md)：MoE Combine 的可选路径与量化角色的接口、实现假设及失效边界。
- [MoE 计数、前缀边界与专家总量](moe_counts_prefix_variants_and_expert_totals.md)：MoE 计数、前缀边界与专家总量的接口、实现假设及失效边界。
- [MoE 确定性前缀槽位与原子顺序](moe_deterministic_prefix_slots_and_atomic_order.md)：MoE 确定性前缀槽位与原子顺序的接口、实现假设及失效边界。
- [MoE 直调 context ABI 与构建边界](moe_direct_context_abi_and_build_boundary.md)：MoE 直调 context ABI 与构建边界的接口、实现假设及失效边界。
- [MoE 执行单元与同步证据](moe_execution_units_and_sync_evidence.md)：MoE 执行单元与同步证据的接口、实现假设及失效边界。
- [MoE 融合内部元数据与重叠条件](moe_fusion_internal_metadata_and_overlap_conditions.md)：MoE 融合内部元数据与重叠条件的接口、实现假设及失效边界。
- [MoE ready 目标、状态复位与窗口轮次](moe_ready_targets_reset_and_window_epochs.md)：MoE ready 目标、状态复位与窗口轮次的接口、实现假设及失效边界。
- [MoE 阶段归属与共享统计](moe_stage_ownership_and_shared_statistics.md)：MoE 阶段归属与共享统计的接口、实现假设及失效边界。

- [MoE 样例位重解释前缀计数的数值前提](moe_sample_bitwise_prefix_counter.md)：MoE 样例位重解释前缀计数的数值前提的代码事实、前提与失效边界。
- [MoE 样例 context 窗口寻址与容量](moe_sample_context_windows_and_capacity.md)：MoE 样例 context 窗口寻址与容量的代码事实、前提与失效边界。
- [MoE 样例 Golden 与 FP16 造数边界](moe_sample_golden_and_fp16_input_limits.md)：MoE 样例 Golden 与 FP16 造数边界的代码事实、前提与失效边界。
- [MoE 样例 Host 多卡线程失败与资源生命周期](moe_sample_host_thread_failure_lifecycle.md)：MoE 样例 Host 多卡线程失败与资源生命周期的代码事实、前提与失效边界。
- [MoE 样例输出顺序、有效尾与 UB 预算](moe_sample_output_order_and_ub_budget.md)：MoE 样例输出顺序、有效尾与 UB 预算的代码事实、前提与失效边界。
- [MoE 样例 ready 发布、UB 依赖和复用缺口](moe_sample_ready_publication_and_reuse.md)：MoE 样例 ready 发布、UB 依赖和复用缺口的代码事实、前提与失效边界。
- [MoE 直调样例的实际支持域和构建配置](moe_sample_support_domain_and_actual_build.md)：MoE 直调样例的实际支持域和构建配置的代码事实、前提与失效边界。
