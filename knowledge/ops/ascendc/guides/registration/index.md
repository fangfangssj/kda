# registration

- [OPC 构建阶段与设备产物身份](opc_build_artifact_identity.md)：区分配置、任务生成、设备编译、打包和运行，核对 Blaze/Tensor API staging 与最终 Kernel 变体集合。
- [OpDef 类型组合与二进制配置的一致性](opdef_dtype_combinations_and_binary_config.md)：区分按列组合与自由类型组合，并核对目标 SoC 的有效原型、Kernel 输入输出和编译配置。
- [Tiling 分发、模板实例与运行数据合同](tiling_dispatch_and_data_contracts.md)：区分 TilingKey、TilingData、BlockDim 和 workspace，保持 Host 决策与已编译 Kernel 分支一致。

- [ACLNN 运行张量的逻辑视图与物理存储](runtime_tensor_view_and_storage.md)：对齐 shape、stride、offset、dtype 与存储容量，区分转置视图、实际重排和动态输出的验证对象。

- [dtype 特化、TilingKey 编码与平台 Fallback](dtype_specialization_and_platform_fallback.md)：补充存储/索引/计算类型组合、分发位域一致性和原生能力不足时的实际转换依赖。

- [Grouped POD 消费与 MIX 唯一写者](grouped_pod_consumers_and_mix_writers.md)：保留完整 Tiling 序列化、GM loader、按组任务重置和单行 sub-block ownership。

- [算子文档模板中的接口事实与占位值边界](api_contracts_extracted_from_document_templates.md)：提取两段调用、输入状态、图模式和 Tiling 的知识，避免把示例产品表与容量表当成能力证据。

- [Host 元数据、运行属性与图输出类型](host_metadata_runtime_attrs_and_graph_dtypes.md)：保留描述符查询、属性类型和实际注册绑定，避免用工程命名规则推导 API 禁令。

- [Add 注册模板的跨层 shape 与 dtype 合同](add_template_cross_layer_shape_and_dtype_contracts.md)：核对 L2、L0、OpDef、图推导与 Kernel 的实际支持域，补足广播和空输入检查。

- [注册转直调的 Tiling 保真与 launch ABI](reverse_migration_preserves_tiling_and_launch_abi.md)：提取依赖闭包、双 pass 可见性和参数布局，限定机械替换规则。

- [msOpGen 迁移的原型身份与类型映射](msopgen_contract_identity_and_dtype_mapping.md)：保存注册点、类型元组与分发对应关系，明确源码规则的适用边界。
- [msOpGen 迁移的数值保真与示例冲突](msopgen_semantic_preservation_and_conflicting_examples.md)：保存运算次序、分区和类型可见性知识，识别文本计数不能证明等价。
- [msOpGen Tiling 序列化与拷贝证据](msopgen_tiling_serialization_and_copy_evidence.md)：区分嵌套对象所有权、Host/Device视图和上下文接口，限定 double free 归因。

- [转换后 Kernel 的绑定、Tiling 与分发合同](translated_kernel_binding_tiling_and_dispatch_contracts.md)：保存 Pybind/TORCH_LIBRARY 身份及整除假设，排除文件组织强制规则。

- [DSL 多 dtype 编译宏与模板绑定](dsl_dtype_macro_and_template_binding.md)：保留按输入名特化和合法分支，限制二进制相同的误诊。

- [平台注册、数据类型与 Shape 限制](platform_registration_dtype_and_shape_limits.md)：保存具体配置差异，限定宏、动态标志与硬件能力推断。

- [索引、Pooling 与采样分发合同](index_pooling_sampling_dispatch_contracts.md)：保存合轴、索引宽度和采样分组，补齐边界及非法Fallback。
- [归一化类型矩阵与分发边界](normalization_type_matrix_and_dispatch_boundaries.md)：保存统计输出和分块Key，修正九组合、阈值及舍入时机。
- [优化器与 Foreach 分发映射](optimizer_and_foreach_dispatch_maps.md)：保存具体Key、类型组合与状态能力，限定泛化收益。
- [量化 TilingKey 字段与示例修正](quantization_tiling_key_fields_and_examples.md)：保存多格式路由与后处理要求，纠正1420和六位编码。
- [模板状态、队列与寄存器 Lane 合同](template_state_queues_and_register_lane_contracts.md)：保存CRTP和混合类型状态布局，纠正同位宽视图与自动转换推断。
- [类型化 TensorList、可选参数与输出](typed_tensor_lists_optional_parameters_and_outputs.md)：保存Foreach及双量化接口，补齐缺失参数状态和输出合同。

- [DAG 输出与 Foreach Tensor 区间合同](dag_outputs_and_foreach_tensor_range_contracts.md)：保存调度抽象和列表游标，补齐状态输出及整数边界。
