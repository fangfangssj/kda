---
type: "Runbook"
title: "CMake arch 子目录隔离 dav-3510 专属 API"
description: "xpu_ops v2 框架的 CMake 用 file(GLOB_RECURSE) 扫描算子根目录 .cpp（排除 arch_* 子目录）对所有架构编译，仅当前架构的 arch_<X>/*.cpp 按架构编译。把 dav-3510 专属 API（MicroAPI/RegTensor/aicpu tiling/get_aiv_num）代码放在根目录时，NPU_ARCH=dav-2201 全量编译因找不到这些符号而 exit=2；将专属代码移入 arch_3510/ 子目录后 dav-2201 单算子编译通过。判别=用 -DOPERATOR_NAME 单算子构建隔离错误来源，确认错误来自兄弟算子根目录代码而非目标算子 arch_2201 代码。"
tags: ["task.debugging", "topic.compilation", "topic.implementation"]
status: stable
sources: []
created_at: "2026-08-18T15:56:42Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# CMake arch 子目录隔离 dav-3510 专属 API

## 适用场景

- **工程形态**：构建——使用 xpu_ops v2 框架的 CMake 构建系统（`ops/backends/ascend/CMakeLists.txt` 与 `build_asc.cmake`），按 `NPU_ARCH` 派生 `ASCEND_ARCH_OPS_DIR` 隔离不同架构的算子实现。
- **API 路径**：CMake 变量 `NPU_ARCH`（`dav-3510`/`dav-2201`）→ 派生 `ASCEND_ARCH_OPS_DIR`（`arch_3510`/`arch_2201`）→ `file(GLOB_RECURSE)` 扫描算子根目录 .cpp（排除 `arch_*` 子目录，对所有架构编译）+ 当前架构 `arch_<X>/*.cpp`（仅匹配架构编译）。
- **验证平台与版本**：Ascend950PR（dav-3510），CANN 版本未从 trace 恢复。
- **适用粒度**：`operator_family`——所有在 v2 框架根目录使用 dav-3510 专属 API 的算子（dequant_swiglu_quant、moe_gating_topk、moe_init_routing_dynamic_quant）均适用。
- **不适用或未验证场景**：无 `ASCEND_ARCH_OPS_DIR` 派生与 `arch_*` 子目录扫描机制的旧版 CMake 不适用；a3/a2 平台未验证。

## 判别方法

**触发**（tool_observed 失败）：在 xpu_ops v2 框架中迁移 dequant_swiglu_quant 算子，算子根目录（`kernels/dequant_swiglu_quant/*.cpp`）使用 dav-3510 专属 API（`DequantSwigluQuantAicpuTiling` aicpu tiling kernel、`MicroAPI`、`RegTensor`、`get_aiv_num`、`get_ub_size`）。配置 `NPU_ARCH=dav-2201` 执行全量构建，`make` 返回 exit=2（compile_failure）。

**探针**（可执行检查）：

1. 用单算子构建隔离错误来源：`cmake .. -DTARGET_DEVICE=ASCEND -DNPU_ARCH=dav-2201 -DOPERATOR_NAME=dequant_swiglu_quant && make -j64`，确认目标算子的 `arch_2201/` 代码本身能否编译通过。
2. 在全量构建失败的输出中 grep 报错的 `.cpp` 文件名与未定义符号（`MicroAPI`/`RegTensor`/`DequantSwigluQuantAicpuTiling`/`get_aiv_num`），确认错误是否来自兄弟算子（moe_gating_topk、moe_init_routing_dynamic_quant）根目录代码而非目标算子 `arch_2201/` 代码。
3. 将 dav-3510 专属 API 的 kernel/layer 代码移入 `arch_3510/` 子目录，根目录只保留跨架构共享的 struct 声明头文件，重跑 dav-3510 clean 构建与 dav-2201 单算子构建，观察 exit 是否为 0。

**观测**（tool_observed）：

- `NPU_ARCH=dav-2201` 全量构建返回 exit=2（compile_failure）。
- `NPU_ARCH=dav-3510` clean rebuild 返回 exit=0，.so 含 dequant arch_3510 layer object。
- `NPU_ARCH=dav-2201` 单算子 clean 构建（`-DOPERATOR_NAME=dequant_swiglu_quant`）返回 exit=0，.so 1.29MB，link 含 `arch_2201/dequant_swiglu_quant.cpp.o`。

**判读**（根因解读，agent 对被 redaction 占位的构建错误输出做的归因）：xpu_ops v2 CMake 的 `file(GLOB_RECURSE)` 把算子根目录 .cpp（排除 `arch_*` 子目录）作为"跨架构代码"对**所有**架构编译，仅当前架构对应的 `arch_<X>/*.cpp` 按架构编译。当 dav-3510 专属 API（`MicroAPI`/`RegTensor`/`DequantSwigluQuantAicpuTiling`/`get_aiv_num`）的代码留在根目录时，`NPU_ARCH=dav-2201` 下编译器找不到这些符号的定义，全量构建 exit=2；而目标算子的 `arch_2201/` 代码本身无错误（单算子构建 exit=0），失败来自兄弟算子（moe_gating_topk、moe_init_routing_dynamic_quant）根目录代码对 dav-3510 专属 API 的引用。修复方向=将 dav-3510 专属代码移入 `arch_3510/` 子目录、dav-2201 兼容版本移入 `arch_2201/` 子目录、根目录只留跨架构共享头文件，使 CMake 仅在 `NPU_ARCH=dav-3510` 时编译含专属 API 的代码。

## 验证信号

来源：dequant_swiglu_quant 算子从 operators 直调工程迁移至 xpu_ops v2 框架的构建环节，Ascend950PR（dav-3510）硬件。

- `NPU_ARCH=dav-2201` 全量构建返回 exit=2，编译失败（来源平台：950；证据等级：tool_observed）
- `NPU_ARCH=dav-3510` clean rebuild 返回 exit=0，.so 含 dequant arch_3510 layer object（来源平台：950；证据等级：tool_observed）
- `NPU_ARCH=dav-2201` 单算子 clean 构建返回 exit=0，.so 1.29MB，link 含 `arch_2201/dequant_swiglu_quant.cpp.o`（来源平台：950；证据等级：tool_observed）