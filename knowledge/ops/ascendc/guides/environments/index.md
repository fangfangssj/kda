# environments

- [Ascend C 编译与运行环境的依赖边界](build_and_runtime_dependency_boundaries.md)：区分工具链、运行时、算子包和自定义库路径，按实际安装版本核对配套关系与加载结果。
- [设备身份、芯片型号与编译架构的区别](device_identity_and_architecture_selection.md)：分开核对设备映射、SoC 与 NPU 架构，并沿构建和依赖链确认实际选择的实现。

- [Ascend C 的设备 C++ 子集与标量、向量接口边界](device_cpp_and_scalar_vector_interfaces.md)：按目标编译单元和执行对象选择语言功能与数学接口，区分常量表达式、运行参数、片上缓冲分配和动态堆能力。

- [Ascend C API 家族与文档版本的兼容边界](api_family_and_release_compatibility.md)：区分接口家族、Host 与 Kernel 职责以及文档、编译和运行组件的版本，保存跨版本材料中的具体适用范围。
- [Kirin Ascend C Simulator 与 mobile-station 发行包范围](kirin_simulator_distribution_scope.md)：保存 Kirin 端侧开发材料中 Simulator 与 mobile-station CANN 的配套关系，区分服务器环境检查与目标模拟器可用性。

- [MIX 的流顺序与诊断证据边界](mix_stream_order_and_diagnostic_evidence.md)：提炼初始化覆盖、构建缓存、设备状态与中间值定位中的可复用知识。

- [直调工程的编译单元、目标名与实际加载产物](direct_build_targets_and_loaded_artifacts.md)：区分 ASC 设备编译、C++ 注册 ABI、CMake 目标和运行路径，保留模板依赖条件。
- [麒麟 CCE 架构选择与仿真运行库解析](kirin_cce_arch_and_simulator_resolution.md)：从实际 CMake 模块提取 Kirin arch 映射、编译检测缺口和运行库选择条件。

- [Blaze 设备地址空间与 VF 循环约束](blaze_device_address_spaces_and_vf_loops.md)：保留真实编译诊断中的 GM/UB 指针、宏分区、Reg mask 和向量循环宽度条件。
- [Blaze 头文件、宏与三种链接上下文](blaze_headers_macros_and_link_contexts.md)：保留同源模板依赖、宏值、头搜索及普通 Host/ASC Host/Device 的独立验证。

- [Host C++ 所有权、对象表示与声明边界](host_cpp_ownership_and_type_boundaries.md)：补齐异步捕获、智能指针、宏展开、字节复制和 ABI 扩展的适用条件。
- [Host ELF 构建选项与加载边界](host_elf_build_hardening_boundaries.md)：解释 PIC、PIE、RELRO、栈保护、符号裁剪与搜索路径，限定于真实 Host 制品。

- [注册包的架构选择与运行库身份](registry_package_architecture_and_runtime_identity.md)：抽取 ASC 构建目标、依赖布局和运行加载限制，保留脚本中的具体失败边界。

- [注册 UT 的链接 ABI 与覆盖率范围](registry_ut_linkage_abi_and_coverage_scope.md)：保留动态注册库保留、依赖身份及覆盖率统计的边界。

- [msOpGen 构建与逐字节比较的证据范围](msopgen_build_and_binary_comparison_evidence.md)：抽取生成器、实际加载包及输出身份的验证条件，保留脚本漏检。

- [API 示例清单与真实编译合同](advisor_api_examples_and_validation_limits.md)：纠正设备修饰符和命名空间的误判，保存Tensor、队列及Mask操作边界。

- [npusim 仿真目标、运行依赖与结果证据边界](npusim_target_and_result_evidence_scope.md)：区分 npusim 的目标配置、程序运行、数值结果和性能报告，限定 CLI 与单卡假设的版本范围。
