# synchronization

- [Ascend C 跨核通知通道与缓冲复用](cross_core_channels_and_reuse.md)：将 flag 标识、未消费通知计数、生产消费流水和参与核集合分开建模，避免将计数容量误解为循环次数限制。

- [精度转换链中的同步作用域](precision_chain_dependency_scope.md)：区分 Cast 数值误差与缓冲依赖，明确同流水、跨流水和跨核同步不能按名称互换。

- [分块流水的启动、事件复用与排空](tiled_pipeline_event_lifecycle.md)：补齐 CopyIn/Compute/CopyOut 流水中的零任务、槽位归还、跨 Scalar 消费和事件资源生命周期。

- [向量递推中的真实依赖与别名冲突](alias_hazards_and_vector_recurrence.md)：用读写集合区分 RAW、WAR 和 WAW，说明额外缓冲与局部屏障的实际作用。

- [手动 Mmad 累加器与 UnitFlag 闭环](manual_mmad_accumulators_and_unit_flags.md)：补齐 L0 输入释放、结果发布及条件屏障示例的失效边界。

- [原子模式切换与首次覆盖顺序](atomic_mode_and_initial_write_order.md)：保持覆盖写先于原子累加，区分 FIX 局部排序与跨核初始化完成。
- [条件消费者与空任务发布](conditional_consumers_and_empty_publication.md)：覆盖可选 Vector 后处理、对端通知、模式参与者和空分支中的真实数据合同。
- [事件完成语义与实际复用者](event_completion_and_actual_reuser.md)：纠正 SetFlag 发射解释及错误回边方向，区分静态候选与真实依赖。
- [环形缓冲的地址值与索引重绑定](ring_region_value_and_rebinding.md)：以物理区域和期次验证同步，而非仅比较变量根名称。

- [Blaze 局部槽位与跨核通道的分别闭合](blaze_local_slots_and_cross_core_channels.md)：保留事件申请、计数轮换与阶段参与者，纠正固定 ID 和全局屏障的过度归纳。
- [Blaze 两阶段初始化与空闲路径](blaze_two_phase_initialization_and_idle_paths.md)：纠正成员析构安全推断、SyncAll 默认假设和 tuple 特化重名，保留全参与者初始化。

- [审查规则中的同步必须回到真实依赖](review_rules_require_actual_dependency_proofs.md)：分离 PipeBarrier、事件、核间同步、原子初始化和 SuperKernel 参与者。

- [Attention 流水 slot 与跨核交接的平台约束](attention_schedule_slots_and_platform_handoffs.md)：保存预取排空与数值状态版本，区分同步示例和已证明协议。

- [队列身份与跨核 Pipe 合同](queue_identity_and_cross_core_pipe_contracts.md)：修正同步指南中的示意释放、固定 Pipe 和事件 ID 泛化。

- [Workspace Ring 的 Credit 与整行统计发布](workspace_ring_credits_and_bulk_statistics.md)：保存逐 tile 与批量交接，修正 flag 位置信息不足以证明协议。

- [事件所有权、重叠执行与原子完成](event_ownership_overlap_and_atomic_completion.md)：保存逐算子同步点，限定Barrier数量与吞吐关系。
- [Vector 流水事件方向与读取位置](vector_pipeline_event_direction_and_read_placement.md)：纠正MTE方向、跨事件配对与Scalar读后才等待的示例。

- [Ping-Pong 初始许可与延迟 Atomic 刷新](ping_pong_event_credits_and_delayed_atomic_flush.md)：保留手工流水和局部累加设计，补齐缺少的事件与输出协议。

- [广播紧凑源同步与复用槽身份](broadcast_compact_source_sync_and_slot_identity.md)：补齐TBuf搬运到Vector与复用前的生产消费关系。

- [双缓冲模板队列占用与事件闭合](double_buffer_template_queue_occupancy_and_event_closure.md)：纠正预取必须重写循环、depth1连续入队及不完整手工事件样例。

- [常驻Softmax状态的递推槽与复用事件](resident_softmax_state_slot_recurrence_and_reuse_events.md)：补齐pingpong状态传递、正确事件方向和初始化范围。

- [APACE ReduceScatter 时序与通知计数](apace_reduce_scatter_schedule_and_credit_audit.md)：APACE ReduceScatter 时序与通知计数的来源实例与适用边界。
