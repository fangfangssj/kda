# diagnostics

- [Ascend C 内存检测报告的判读与覆盖边界](memory_sanitizer_reports.md)：根据 msSanitizer 报告中的地址空间、访问跨度、核号和调用栈定位问题，区分非法访问、多核重叠与 Host 分配生命周期。
- [Ascend C 运行异常的二进制、路由与执行证据](runtime_failure_evidence.md)：将加载产物、TilingKey 变体、ABI、异步错误和内存同步分层核对，避免依据错误码或输出不变直接认定根因。

- [Ascend C 插桩数据的同步、定位与观测边界](instrumentation_and_observation.md)：将 DumpTensor 和打印结果与核、tile、阶段及真实生产完成点对齐，避免插桩改变时序后误判修复。

- [错误层级与非对齐搬运证据](error_layer_and_non_aligned_copy_evidence.md)：保存 ACLNN 诊断入口，并纠正以 507035 推断统一 32B 搬运长度的做法。
- [插桩产物与内存检测执行证据](sanitized_binary_and_test_execution_evidence.md)：在自动化包装中区分构建、安装、实际测试执行和检测结果，保留证据链。

- [二进制算子测试的完整输入与失败传播](binary_test_io_and_partial_failure.md)：从 Add 与麒麟样例提取文件长度、dtype、输出覆盖和 ACL 错误传播约束。

- [MX 测试的资源析构与非有限值验收](mx_test_raii_and_nonfinite_acceptance.md)：保留较完整 I/O 和 RAII 设计，同时定位异步失败与两份验证器的差异。

- [Blaze 分阶段诊断需保留正式执行条件](blaze_staged_modes_preserve_execution.md)：将 C-direct、C-through 与 V-known 的观察点绑定到真实入口和数据表示。

- [Host 字符串、格式化与复制长度](host_strings_formatting_and_buffer_bounds.md)：纠正跨平台整数格式表，保留临时字符串寿命、destMax 和异步日志的实际风险。

- [Flash 模板故障中的编译、初始化与数值合同](flash_template_failures_and_numeric_contracts.md)：提取通用故障原因，纠正 DMA 长度、类型转换和迭代误差的过度结论。

- [优化经验 API 示例的类型与生命周期核对](evolution_api_examples_require_typed_contracts.md)：集中保存队列、TBuf、DMA 字段和向量化示例中的可复用限制。

- [DumpTensor 的阶段身份与 Attention 参考语义](dump_tensor_stage_identity_and_attention_reference.md)：保存生产消费观测法，修正同步探针、尾块和局部 softmax 参考的偏差。

- [Profiler 不同观测层的数据不可替代性](profiler_memory_source_and_sampling_evidence.md)：区分分配生命周期、访存计数、源码指令映射、PC 采样和时间线，明确各类性能证据的能力边界。

- [msaicerr 的 Dump dtype 解析与 AI Core Error 分析边界](msaicerr_dump_dtype_and_analysis_scope.md)：保存 msaicerr 对 dtype 覆盖、故障材料、路径约束、首次错误和环境样例的行为，区分解析成功与数值正确。

- [asys 文件解析的输入配对、符号搜索与错误范围](asys_analysis_inputs_symbols_and_error_scope.md)：区分 trace、coredump、stackcore、coretrace、Unified Bus 和 AI Core Error 解析的依赖、参数、版本及首错边界。
- [asys 故障采集上下文、业务子进程与实时堆栈](asys_collection_context_and_live_stacktrace.md)：解释环境变量覆盖、编译与 Dump 材料前提、信号 35、堆栈导出参数和超时观测，避免采集成功却缺失关键证据。
- [asys 诊断的产品范围、超时和压测电压恢复](asys_diagnostic_side_effects_and_platform_scope.md)：说明诊断权限、并发与依赖、HBM/CPU 超时差异，以及压力检测异常退出后的电压观测和恢复。
- [asys 健康状态与硬件检测结果的不同含义](asys_health_and_diagnostic_result_semantics.md)：区分 Healthy、Warning、Pass、Warn、Fail 和 HBM 新增 ECC 计数，保留设备选择及终端显示截断边界。
- [msnpureport 的设备调试配置、核屏蔽与恢复边界](msnpureport_device_debug_configuration.md)：区分 Device 和工具日志，解释 singlecommit、核屏蔽、自动复位及 icache 检查对算子排障和性能的影响。
- [msnpureport 的日志导出范围、设备身份与材料定位](msnpureport_export_scope_and_log_identity.md)：解释单次和连续导出的范围、SMP 和容器限制、目录中的设备号及缺失日志的观测边界。
- [msnpureport 连续采集的日志保留、容量与脚本轮转](msnpureport_retention_and_auto_export.md)：保存各类日志数量和大小参数、覆盖边界及自动导出脚本的路径、清理和半容量轮转合同。

- [assert（AI CPU API）：固定 reference 调用约束](aicpu_assert_reference_contract.md)：固定 reference 的调用约束与版本边界。该接口实现AI CPU算子Kernel调试场景下的assert断言功能。 算子执行中，如果assert内部条件判断不为真，则输出assert条件、触发文件名、行号等信息。
- [printf（AI CPU API）：固定 reference 调用约束](aicpu_printf_reference_contract.md)：固定 reference 的调用约束与版本边界。该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。
- [Printf：固定 reference 调用约束](ascendc_printf_reference_contract.md)：固定 reference 的调用约束与版本边界。SIMD和SIMT编程调试场景下提供格式化输出功能，在算子Kernel侧打印日志信息。

- [参考实现的量化建模与器件不一致时，精度缺陷会记在算子头上](reference_quantisation_model_mismatch.md)：golden 用 HiFloat8 建模 P、器件用 MXFP8，造成一个随 s2 变化的假 dV 缺陷；给出四个判据：斜率而非距离、只随尾数变即看舍入网格、第三方基准裁决、改器件不读的输入。
