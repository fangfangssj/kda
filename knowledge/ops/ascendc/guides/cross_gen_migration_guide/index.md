# cross gen migration guide

- [3510 arch migration](3510_arch_migration/index.md)：共 4 张知识卡（如 2201到3510架构变更 等）。
- [Ascend C API兼容性策略](ascend_c_api_compatibility.md)：本兼容性说明仅适用于Ascend C算子开发的兼容性迁移指导。总体兼容性策略见表1，兼容性范围不包含编译器BuiltIn API、Ascend C内部实现接口等。文档中涉及的兼容性分为两类：一是功能兼容，包括数据类型兼容、接口原型兼容和常量兼容；二是性能兼容，指对于同等数据量，新架构上执行API耗时不高于旧架构。
- [instructions for new features](instructions_for_new_features/index.md)：本文介绍NPU架构版本3510新增特性，为开发者将算子程序从其他架构迁移至3510架构提供参考。有关3510架构的整体变更，请参见2201到3510架构变更。
- [跨代迁移指南概述](overview.md)：Ascend C的核心设计目标之一是实现跨代兼容。然而，每一代芯片在微架构上都存在独特的硬件特性，因此无法保证所有接口仅通过重编译就能达成完全的跨代兼容。基于这一前提，本手册旨在为Ascend C算子开发者提供系统化指导，帮助将算子Kernel代码从当前芯片平滑移植到目标代际芯片上运行，实现高效、可控的迁移。
- [arch22 → arch35 AscendC Migration and Backward-Generation Reference](arch22_to_arch35_porting_playbook.md)：arch22→arch35 迁移与反向生成的参考总纲：给出 L1~L5 分层迁移模型的适用条件与范围（L1 基础适配必做、L2 RegBase 重写、L3 SIMT、L4 需 IsRegbaseSocVersion 或 UB 不足时升级、L5 仅供参考），以及目录地图、被 brief 构建层消费的方式与来源可复现性说明。
- [AscendC arch22 → arch35 算子迁移与反向算子生成 (L1+L2+L3)](source_skill.md)：arch22→arch35 迁移与反向算子生成的源 skill 全文（L1+L2+L3）：证据与终验边界（禁逐字复制目标实现，golden 优先用 KernelBench 风格 task+sidecar，反向以 CPU fp64 autograd 为准）、迁移层级决策树与 L4 升级信号、工作流概览，以及强制的阶段 0 环境获取与验证。
- [L1 迁移实施详细指南](l1_implementation.md)：L1 级迁移（基础适配）实施细则：_def.cpp 用独立 OpAICoreConfig 并以 opFile 指向 _apt RegBase kernel、新建 _apt.cpp 与 arch35/ 目录、改 CMakeLists 与 config/ascend950/；另含 UB 预留规范与升级 L4 的信号。
- [L1/L2 迁移实施详细指南](l1_l2_implementation.md)：L1 与 L2 两档迁移的实施细则：L1 为零改动配置注册（触发条件是纯 INT32、无已删数据通路、UB 占用小等，只需 AddConfig(ascend950) 复用 910b 配置并建 config 目录）；L2 为同构适配，需独立配置、_apt.cpp、arch35/ 目录与 include 路径调整及 BF16 简化。
- [L2：RegBase API 重写（MicroAPI）详细指南](l2_register_based.md)：标为 L2 档的 RegBase MicroAPI 重写指南：950 的 Vector Core 由 MemBase 变 RegBase，计算路径整体改写为寄存器式。含 RegTensor/MaskReg、取代 RoundMode 的 CastTrait、SatMode 与溢出模式、量化 Pack 与非对齐 Store。
- [L3：SIMT 优化（多线程替代 Scatter/Gather）详细指南](l3_simt_optimization.md)：标为 L3 档的 SIMT 优化指南：用最多 2048 线程直接读写 GM 取代 Scatter/Gather 的 GM→UB→计算→UB→GM 四步；含 __simt_vf__ + LAUNCH_BOUND 声明、线程索引与启动、Simt::AtomicAdd 与 __local_mem__ 访问 UB。
- [L4：SIMT 优化（多线程替代 Scatter/Gather）详细指南](l4_simt_optimization.md)：标为 L4 档的 SIMT 优化指南（内容与 L3 档同）：用最多 2048 线程直接访问 GM 取代 Scatter/Gather 的 UB 中转四步；含 __simt_vf__ + LAUNCH_BOUND 声明、线程索引与启动、Simt::AtomicAdd 与 __local_mem__ 读写 UB。
- [L5：Register-based 重写（MicroAPI）详细指南](l5_register_based.md)：标为 L5 档的 Register-based MicroAPI 重写指南（内容与 L2 档同）：950 Vector Core 由 MemBase 转 RegBase，计算路径全部改写。含 MicroAPI 命名空间、寄存器向量长度、CastTrait 四要素与 SatMode 规则、溢出模式控制、量化 Pack 与非对齐 Store。
- [ops-nn arch35 Artifact Layout](ops_nn_a5_artifact_layout.md)：arch22→arch35 移植后 ops-nn 仓每个算子必须产出的文件与 schema：op_host 的 _def.cpp（ASCEND950 OpAICoreConfig）、op_kernel 的 _apt.cpp 入口体、config/ascend950/ 下 binary.json 与 simplified_key.ini。

- [Porting CUDA Benchmark Suites (BabelStream / STREAM / HPL-AI style)](porting_cuda_benchmark_suites.md)：CUDA benchmark 套件移植方法论：host C++ 按语义重写为 PyTorch harness（不逐字移植）、每个 kernel 独立 op-gen、真值来自 A100 CUDA 运行（含 nvcc 标志与 FMA 分组对齐）、case_gen 形状/分布覆盖矩阵强制。
- [A3→A5 移植先看驻留量，不是先看算法](a3_to_a5_ub_residency_is_the_port_decision.md)：A3 内核常把整块数据设为 UB 常驻；A5 UB 仅约 248 KB，逐行移植会在大 case 上直接不可表示。判据与改法。
