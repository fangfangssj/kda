# testing

- [算子测试的行为断点与属性交叉覆盖](behavior_boundaries_and_case_coverage.md)：从实现分支和接口合法域识别尾块、资源轮换及路由边界，避免随机样本数量冒充覆盖。

- [UT 覆盖率的统计范围与执行证据](coverage_denominators_and_execution_evidence.md)：区分文件范围、行分支计数和语义断言，纠正仅编译或汇总表查零替代测试的做法。
- [Host UT 的属性、CompileInfo 与推导上下文](host_context_attributes_tiling_and_inference.md)：明确 Tiling、InferShape 和 InferDataType 各自验证内容及上下文实例的一致性。
- [Host UT CSV 的字段合同与 Optional 状态](host_csv_schema_and_optional_tensor_states.md)：保留来源 CSV loader 的编码、实例和解析关系，区分框架约定与 Tensor 本体语义。
- [Kernel CPU 仿真与 AICPU UT 的证据边界](kernel_cpu_simulation_and_aicpu_test_boundaries.md)：提取 tikicpulib 的参数链、资源与输出验证，纠正缺库即删测试及固定 workspace 示例。
- [OP API UT 的 Tensor 描述与两段式断言](opapi_tensor_descriptors_and_two_stage_assertions.md)：保留非连续视图、Scalar 描述和 dtype 合法组合，界定 WorkspaceSize 测试的覆盖范围。
- [CSV 用例转换中的形状与预期值保真](preserving_case_semantics_during_csv_conversion.md)：防止旧测试转表格时丢失存储形状、动态逻辑、精度和独立预期。

- [SHMEM Rank 特征、文件长度与真实边界测试](shmem_rank_patterns_payload_files_and_boundary_cases.md)：补齐输入可辨识性、BF16 编码、per-PE oracle 和不整除/资源复用的实际覆盖。

- [黑盒因子与可达白盒分支的覆盖关系](blackbox_factors_and_reachable_whitebox_branches.md)：补充受约束组合、零维广播、dtype 数值域与真实分发观测。

- [测试约束图的语义与求解完整性](constraint_graph_semantics_and_solver_completeness.md)：区分推导、匹配、广播、存在性和类型兼容，保留可执行依赖模型及缺口。
- [ST 参数域与精确值生成](parameter_domains_and_exact_value_generation.md)：保存 Tensor、列表、标量的因子模型，修正特殊值与低位格式范围示例。

- [GE 与 ACLNN 调用示例的证据范围](ge_and_aclnn_examples_are_execution_evidence.md)：保存图构造与两阶段调用知识，纠正数据编码、输入索引和无比较成功消息。
- [注册 ST 比较器的失效方式与覆盖缺口](registry_st_comparison_failures_and_coverage.md)：保留 MERE/MARE 公式，纠正 BF16 编码、NaN 漏检、Mock 与跳过成功。

- [注册 UT Faker 的属性顺序与平台模拟合同](registry_context_faker_attribute_and_mock_contracts.md)：核对动态 IO、属性编码、对象寿命与模拟平台的实际适用范围。
- [注册 UT 的 Tiling 断言与 Tensor 工厂边界](registry_tiling_assertions_and_tensor_factory_limits.md)：抽取断言漏检、原始字节布局、空输入和仅参数校验的证据限制。
- [独立 Host 的 BF16 编码与比较器反例](standalone_host_bf16_and_comparator_counterexamples.md)：保留最近偶数舍入方法，识别 NaN 漏检及异常资源窗口。

- [参考污染与 Scan 舍入证据](reference_mutation_and_scan_rounding_evidence.md)：保留 torch 算子加载和 scan 差异诊断，排除改写基准来匹配候选。

- [白盒用例描述的可选、列表与输出语义](whitebox_case_descriptor_semantics.md)：保存 Mapper-v1 的数据含义，避免空值、嵌套和顺序转换丢失。
- [白盒 Host–Kernel 路径谓词](whitebox_host_kernel_path_predicates.md)：保留 key 构造与二级分支，修正笛卡尔积和优先分支的错误简化。
- [白盒接口模型的层次与关联字段](whitebox_interface_contract_layers.md)：区分注册、推导、ACLNN 和框架绑定，保存可选、动态、标量及别名语义。
- [Mapper 模板的校验能力与遗漏](whitebox_mapper_validator_limits.md)：从完整模板提取结构校验边界及变体追踪，排除生成脚本流程。
- [网络样例与数值域的可追溯性](whitebox_network_shapes_and_value_domains.md)：保留真实工作负载关系与特殊值，限制按算子名字猜合法输入。
- [白盒分析的平台与算子身份](whitebox_platform_and_operator_identity.md)：分开保存硬件身份、注册名、调用入口和可验证证据。
- [白盒可达性与联合约束域](whitebox_reachability_and_constraint_domains.md)：以可构造见证区分可达、不可达和未知，避免范围推导漏路径。
- [白盒采样与覆盖证据的边界](whitebox_sampling_and_coverage_evidence.md)：保存元组覆盖和采样复现，分清清单一致性与实际命中。
- [TTK Kernel CSV 的 TensorList 编码](whitebox_ttk_csv_tensorlist_encoding.md)：保留26列位置与嵌套范围，限制版本特有的广播和别名规则。
- [TTK 数据范围、容差与门禁解释](whitebox_ttk_ranges_and_gate_interpretation.md)：区分环境可启动、单用例链路和完整数值验收。

- [APACE 多rank启动与 context 复用范围](apace_multirank_startup_and_context_reuse_scope.md)：APACE 多rank启动与 context 复用范围的来源实例与适用边界。
- [APACE 对标路径与失败证据边界](apace_reference_baselines_and_failure_evidence.md)：APACE 对标路径与失败证据边界的来源实例与适用边界。

- [随机数分布检验的样本、参数与批量判定](random_distribution_tests_and_batch_contract.md)：区分随机数分布检验与逐位比较，明确样本、分布参数、单次显著性和批量次数的一致性。

- [用 MoE 路由反例证明内核接口的信息充分性](moe_routing_interface_information_sufficiency.md)：以固定原版 MoE 路由合同和无相等键的 CPU 反例，证明遗漏专家数据的扩展接口不能唯一决定重排输出。

- [先界定未定义占位输出的比较合同](undefined_placeholder_output_comparison_contract.md)：通过非零 shape 占位输出与扩展调用前返回的独立源码，区分内容无定义、数值失败和真实内核执行。
