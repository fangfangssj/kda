# catlass

- [CATLASS 的 Tile、Block 与 Kernel 组件契约](component_contracts.md)：说明矩阵计算、搬运、后处理与调度组件的职责，以及集成 DeviceGemm 或 Kernel 时需要核对的参数和生命周期。
- [CATLASS tile 的 L1、L0 与 UB 分项预算](tile_resource_budget.md)：按输入类型、累计类型和实际缓冲阶段分别计算片上资源，避免混淆 L1 与 UB 或遗漏双缓冲份数。

- [CATLASS FA 接口映射及参考配方冲突](attention_interface_mapping_and_conflicting_recipes.md)：保存长度、布局、mask 和 MLA 语义，限制固定尺寸与通路结论。
- [CATLASS Epilogue 槽位与自定义 Tile 边界](epilogue_slot_topology_and_custom_tile_limits.md)：保留组件接口和操作数拓扑，修正任意 Tile 列表及零开销推断。
- [CATLASS 分组融合的跨块依赖与小 M 调度](grouped_fusions_cross_tile_dependencies_and_small_m.md)：保留配对列、整行归约和跨核合并的完整数据所有权。
- [CATLASS 线性 Attention 的阶段合同与 Chunk 归属](linear_attention_stage_semantics_and_chunk_ownership.md)：补充 KDA dAv、物理块、分组 head 与跨阶段数据依赖。
- [CATLASS Matmul 模板索引与实例化合同](matmul_template_catalog_and_instantiation_contracts.md)：保存来源示例能力及 Host/Device 参数边界，限制固定最优配置。
- [CATLASS 参考版本与评测语义身份](reference_versions_and_evaluation_semantic_identity.md)：保留线性 Attention 参考定位信息及独立证据范围。
- [CATLASS 轮转 workspace 与 Callback 的收益条件](rotating_workspace_capacity_and_callback_evidence.md)：按缓存、容量、事件和数值合同评估 Matmul 融合。

- [CATLASS 编译架构与启动资源合同](compilation_architecture_and_launch_resources.md)：区分编译宏、目标架构、链接组件和实际核资源。
- [CATLASS 开发实例的装配与 Bias 所有权](development_assembly_and_bias_ownership.md)：保存实现层接口差异，限定槽位、累加类型和固定策略。
- [Grouped 融合事件 Credit 与排障证据](grouped_event_credits_and_diagnostic_limits.md)：保留尾段事件故障机制，纠正强制Barrier即可排除竞争的误判。
- [Grouped Matmul 融合的分区与 Partial Max](grouped_fusion_partition_and_partial_max.md)：补齐不均匀分组、小M二维调度和跨N统计索引。
- [CATLASS Host Device 与 Workspace 指针 ABI](host_device_workspace_and_pointer_abi.md)：保留适配器、Params、别名和系统前缀的实现知识。
- [独立 Golden 与整数 Dot 的精确性条件](independent_golden_and_integer_dot_exactness.md)：保留阶段舍入与量化验证知识，修正最终幅值即可保证FP32精确的说法。
- [CATLASS 精度条件的并集与验证器漏判](precision_criteria_union_and_validator_failures.md)：保存MERE MARE和误差比例的区别及模板反例。
- [CATLASS Shape 合法性与布局实例化](shape_legality_and_layout_instantiation.md)：保存固定计算长度、尾块、转置和测试域的边界。
