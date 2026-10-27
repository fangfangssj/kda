# simt

- [SIMT 调试输出的资源与执行影响](debug_output_resource_cost.md)：将 printf 的线程并发、FIFO 和栈消耗纳入调试判断，区分 SIMT、SIMD 与 AICPU 接口支持范围。
- [SIMT 启动规模与元素覆盖的对应关系](launch_geometry_and_element_coverage.md)：将核块数量、块内线程和 grid-stride 地址公式保持一致，区分启动上界、实际线程数与编译配置。
- [SIMT 共享存储与同步作用域](shared_memory_and_sync_scope.md)：区分块内 UB 共享、跨核 GM 交换和函数串行，按真实存活资源与生产消费关系判断可见性及复用。

- [SIMT 调试上下文与缓存可见性](debug_context_and_cache_visibility.md)：区分 MainScalar、SIMT 访存缓存与跨核发布协议，保留打印入口和刷新范围。
- [SIMT 数值宏族与高低位常量](numeric_macro_families_and_split_constants.md)：保存原参考的完整常量导航，并区分最小次正规数、负零与高低位分解。
- [SIMT 运算符与显式数学函数](operators_and_explicit_math_contracts.md)：区分语言类型规则、FMA、近似除法与接口特殊值合同。
- [SIMT TensorList 描述符的 ABI 与 shape 步长](tensorlist_descriptor_abi_and_shape_stride.md)：保留动态输入 Host 元信息、设备地址解析及不同 rank 下的描述符边界。

- [SIMT API 迁移的签名、lane 与版本边界](api_migration_requires_signature_and_lane_semantics.md)：保留维度和数值语义映射，纠正 shuffle 参数及 VF 统一禁令。

- [SIMT 注册工程的限制与内存配置](registry_engineering_limits_and_memory_configuration.md)：提取调用域、共享区与缓存配置知识，限定线程数和工程路径规则。

- [SIMT 一致分支、launch上界与寄存器证据](simt_uniform_branches_launch_bounds_and_register_evidence.md)：纠正所有if都会线程分化和编译通过即无spill的推断。
