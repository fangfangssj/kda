# blaze

- [Blaze 硬件基形与运行期尾块范围的区别](base_shapes_and_runtime_extents.md)：将 Block/Copy 的合法 base shape 与 Scheduler 的当前逻辑范围分开，按实际消费者校验 tiling、workspace 和融合后处理。
- [Blaze 优化构建挂起时的控制数据与依赖检查](optimized_build_hang_diagnosis.md)：从 Device 实际解码的 tiling 和生产消费次数定位挂起，区分类型别名问题、诊断扰动与真正的同步缺陷。
- [Blaze SplitM Epilogue 的局部与全局坐标](split_m_epilogue_coordinates.md)：分别计算每个 AIV sub 的有效行、UB slot 起点及 GM 输入输出偏移，并保留空任务的同步责任。

- [Blaze 融合输出的整行量化与阶段边界](fused_output_row_quantization.md)：在 Matmul 或 GLU 后按完整逻辑行求 scale，区分 workspace 行距、分块宽度、量化语义和异步完成范围。
- [Blaze GLU 双分支的布局与元数据配对](glu_branch_layout_and_metadata.md)：保持 act/gate 的列配对、padding 和量化元数据一致，并在完整 K 累加后执行非线性融合。
- [Blaze Matmul 逻辑形状与物理存储的一致性](matmul_logical_shape_and_physical_storage.md)：协调 transpose、ND/NZ 布局、数据生成、物理容量和 slice 消费语义，避免逻辑 shape 正确但设备地址错位。
- [Tensor API 布局描述与 Copy 特化的边界](tensor_layout_and_copy_specialization.md)：将 Layout 可表示的形状和步长与具体 Copy 实现实际消费的字段分开，保持地址单位、存储跨度和同步合同一致。

- [权重反量化 Prologue 的数值链与交接](weight_dequant_producer_consumer_contract.md)：将低比特权重的逐通道反量化、物理布局和 AIV 到 AIC 的数据交接作为同一输入合同验证。

- [APACE FragmentTensor 的离散视图与搬运消费者](fragment_views_and_copy_consumers.md)：以地址表描述离散GM段的逻辑拼接，区分虚拟切片、实际gather/scatter、填充长度和异步生命周期。

- [Blaze MX 组件 Params 与直调启动合同](mx_component_params_and_launch_contract.md)：保留组件组合、实际 dtype 与转置分发，并纠正仅凭目录或宏推导功能支持。

- [Blaze 执行引擎、布局标签与特化](execution_engines_patterns_and_specializations.md)：纠正框架总览的流水方向与核数查询，保留抽象层和数据 Trait 的区别。
- [GLU 与逐元素输出接整行量化的联合约束](glu_and_elementwise_row_quant_integration.md)：保留 BMM/Grouped 行映射、paired metadata、单两遍选择与执行拓扑边界。
- [Blaze Launcher 的空输入与物理输入证据](launcher_empty_inputs_and_physical_evidence.md)：补齐逻辑输入到设备字节、零工作量、输出清单及实际执行产物的一致性。
- [Matmul Softmax 的历史指数 workspace 与行归并](softmax_history_workspace_and_row_merge.md)：保留 per-core max/sum、逐 tile 历史最大值及跨核完整归一化地址。

- [权重反量化的 Cast 顺序与 V→C 释放关系](dequant_cast_order_and_vc_release.md)：补全来源 INT8/FP8 两步转换、Bias 元信息、前后融合边界及诊断限制。
- [Blaze 逐元素 Epilogue 的广播映射与加载边界](elementwise_operand_mapping_and_load_guards.md)：补全 MemBase/RegBase 的 operand 映射、独立 slot 预算与 mask 前整向量读取。
- [Grouped Matmul 的行编码与逐组权重地址](grouped_rows_encodings_and_weight_offsets.md)：补齐 cumsum/count、空组、完整 Golden 写回及独立 group/batch 物理映射。

- [Blaze 迁移的资源语义与制品隔离](migration_resource_semantics_and_artifact_isolation.md)：保留参数最终使用点、高阶计算展开、独立加载及比较证据的失效边界。

- [Blaze 注册 Matmul 的 mock Tiling 与属性传递](registry_matmul_mock_tiling_and_attribute_propagation.md)：保存 GemmUniversal 组件连接，明确转置、形状、资源和参数初始化缺口。

- [基础APIGEMV：固定 reference 调用约束](tensor_api_gemv_reference_contract.md)：固定 reference 的调用约束与版本边界。GEMV的核心功能体现为：当矩阵A的维度M取值为1时，接口可以手动启用GEMV功能，该操作退化为1×K维度的行向量与K×N维度矩阵之间的乘法运算。
- [UnitFlag：固定 reference 调用约束](tensor_api_unit_flag_reference_contract.md)：固定 reference 的调用约束与版本边界。unitFlag的核心功能体现为：Mmad和Copy接口引入了单元标志（unit-flag）机制，通过以内存块为粒度实现精细化的数据同步，从而有效降低同步延迟，提升系统整体性能。当UnitFlag开关打开后，对于L0C Buffer中的每个内存块（512B），提供一个单元标志位，用于指示该块是否可读或可写。
