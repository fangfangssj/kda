---
type: "Runbook"
title: "C-API asc_simd.h 编译缺 include 根"
description: "在 Ascend 950（dav-3510 / npu_arch_3510, CANN 9.1.0）下用自写 CMakeLists 编译含 `#include \"c_api/asc_simd.h\"` 的 .asc 时，报 `fatal error: 'instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h' file not found`。表象像 CANN 安装损坏，实为 include 根选错：C-API 公共头内部以 `instr_impl/...` 相对路径 #include 兄弟内部实现头，仅当 `asc/`（符号链接到 x86_64-linux/asc，下含并列的 include/ 与 impl/）作为 include 根时才解析成功。官方 mmad 样例用 `include_directories(\"${ASCEND_HOME}/asc/\")` 即可编译。"
tags: ["task.debugging", "topic.implementation.api_layer.c_api", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-17T03:30:01Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["fatal error: file not found"]
---

# C-API asc_simd.h 编译缺 include 根

## 现象

在 Ascend 950（CANN 9.1.0，`CMAKE_ASC_ARCHITECTURES=dav-3510`）下开发基于 Ascend C C API 的单核 Cube Matmul 算子。源文件 `matmul.asc` 顶部含 `#include "c_api/asc_simd.h"`，自写 `CMakeLists.txt` 配置 include 路径后执行 `cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510 && make -j`，编译立即失败，完整报错（原文）：

```
[ 50%] Building ASC object CMakeFiles/demo.dir/matmul.asc.o
In file included from /workspace/matmul/matmul.asc:19:
In file included from /usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/asc/include/c_api/asc_simd.h:44:
In file included from /usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/asc/include/c_api/vector_datamove/vector_datamove.h:26:
/usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h:50:10: fatal error: 'instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h' file not found
#include "instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h"
         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1 error generated.
```

报错指向一个深层内部实现头 `asc_set_gm2ub_pad_impl.h` "file not found"，第一直觉会怀疑 CANN 安装不完整或路径配错；但该文件实际存在于安装目录中（见判别方法），真实原因是 C-API 公共头链对 include 根的隐式要求未满足。

## 适用场景

- **工程形态**：Ascend C **C API**（`.asc` + `c_api/*.h`）单核算子工程，Kernel 直调（`kernel_direct_call`，`add_executable(demo xxx.asc)`）。
- **API 路径**：源文件 `#include "c_api/asc_simd.h"`（进而拉入 `c_api/vector_datamove/vector_datamove.h` → `impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h` → `instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h`）。
- **验证过的平台与 CANN 版本**：Ascend 950（`dav-3510` / `npu_arch_3510`），CANN 9.1.0（`/usr/local/Ascend/cann-9.1.0-new/cann-9.1.0`，`asc` 符号链接到 `x86_64-linux/asc`）。仅此平台在本 trace 内实测编译/运行通过。
- **适用粒度**：`platform_wide`——任何在 Ascend 950 上从零编写 C-API 算子 CMakeLists 的工程都会命中（与具体算子无关）。
- **路线优先级**：`recommended_fastpath`——直接照官方 mmad 样例 CMakeLists 的 include 配置（见下）。
- **不适用或未验证场景**：A2（`dav-2201` / `npu_arch_2201`）与 A3 的 impl 头目录同名结构存在（`asc_copy_l12l0b_trans_impl.h` 在 `npu_arch_2201` 下亦存在），但本 trace 未在其上编译，**未验证，推断可迁移**（见 `## 失效边界`）。非 C-API（纯 C++ Tensor API）工程若不拉入 `asc_simd.h` 则不会触发本头链。

## 判别方法

本卡判别目标是把"`asc_set_gm2ub_pad_impl.h` file not found"这一表象从"安装损坏"误诊中分离出来，归因到 include 根选错。触发、探针、观测、解释四段均取自 trace 内 tool_observed / artifact_verified 事件。

**触发条件**（tool_observed，编译报错原文）：在自写 CMakeLists 下编译含 `c_api/asc_simd.h` 的 `.asc`，报 `fatal error: 'instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h' file not found`，且报错调用栈顶部为 `asc/include/c_api/asc_simd.h` → `asc/include/c_api/vector_datamove/vector_datamove.h` → `asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h:50`。

**判别探针**（区分"安装损坏"与"include 根选错"）：

1. 确认被报"not found"的文件实际存在。运行：

```bash
ls -la /usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/asc
ls /usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h
```

本 trace 实测的目录结构（`ls -la .../asc` 输出原文，tool_observed）：

```
lrwxrwxrwx 1 root root 16 Jul 30 13:52 /usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/asc -> x86_64-linux/asc
===
total 16
drwxr-xr-x  4 root root 4096 Jul 30 13:52 .
drwxr-xr-x 15 root root 4096 Jul 30 13:56 ..
drwxr-xr-x  8 root root 4096 Jul 30 13:52 impl
drwxr-xr-x 11 root root 4096 Jul 30 13:52 include
```

即 `asc/` 是指向 `x86_64-linux/asc` 的符号链接，其下 `impl/` 与 `include/` 并列。被报缺失的 `asc_set_gm2ub_pad_impl.h` 实际就在 `asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/` 下——文件并未缺失，安装完好。

**观测**（tool_observed / artifact_verified）：

- 编译报错原文（`## 现象` 代码块）的 `#include` 行 `#include "instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h"`，前缀是 `instr_impl/...`，**不是** `c_api/instr_impl/...`。
- `ls` 输出显示 `asc/` 根下 `impl/` 与 `include/` 并列，`asc/impl/c_api/instr_impl/npu_arch_3510/` 真实存在。
- 官方 mmad 样例 `CMakeLists.txt`（artifact_verified，逐字读到）使用 `include_directories("${ASCEND_HOME}/asc/")`，可编译含 `asc_simd.h` 的 `.asc`。

**解释**（include 根选错的机制，由上述 verbatim 报错路径 + verbatim 目录布局推导，非 agent 口述）：

- 报错文件 `asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h` 第 50 行写的是 `#include "instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h"`（前缀 `instr_impl/...`，**无** `c_api/` 前缀）。
- 该相对 include 由编译器按"包含此文件的目录"解析：包含文件所在目录为 `asc/impl/c_api/instr_impl/npu_arch_3510/`，从该目录拼 `instr_impl/npu_arch_3510/...` 会得到 `.../npu_arch_3510/instr_impl/npu_arch_3510/...`——`instr_impl/npu_arch_3510/` 段被加倍，故找不到。
- 唯一能让 `instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h` 正确解析的，是把 **`asc/impl/c_api/`** 作为 include 搜索根（从该根出发，`instr_impl/npu_arch_3510/...` 恰好命中 `asc/impl/c_api/instr_impl/npu_arch_3510/...`）。
- 而把 `asc/`（而非 `asc/include` 或 `asc/impl/c_api`）作为 include 根，既能让 `#include "c_api/asc_simd.h"` 命中 `asc/include/c_api/asc_simd.h`，也能让上述内部 `instr_impl/...` 相对 include 命中——因为 `asc/` 根下并列存在 `impl/`，内部头链沿 `asc/impl/c_api/...` 自洽解析。

**判别结论**：只要被报"not found"的文件 `ls` 能找到、且调用栈来自 `asc_simd.h` 公共头链，即可判定为 **include 根选错**而非安装损坏，按下方参考配置修正即可，无需重装 CANN。

## 参考配置

官方 mmad 样例 CMakeLists 的 include 配置（verbatim，取自样例工程 `CMakeLists.txt`，本 trace 通过 `cat` 读到，artifact_verified）：

```cmake
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    mmad_tensor_api.asc
)

set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})

set(ASC_INCLUDE_DIR "${ASCEND_HOME}/asc/")

include_directories(
    "${ASC_INCLUDE_DIR}"
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
```

关键差异点：`include_directories("${ASCEND_HOME}/asc/")`——include 根是 `asc/`（覆盖并列的 `include/` 与 `impl/`），**不是** `asc/include`、也**不是** `asc/impl`。`ASCEND_HOME_PATH` 由 `source ${install_path}/cann-9.1.0/set_env.sh` 注入，指向 `cann-9.1.0-new/cann-9.1.0`。

## 系统信息

- **工具链**：CANN 9.1.0（`Ascend-cann-toolkit_9.1.0_linux-x86_64.run` + weekly `9.1.0~weekly.20260708.01`），编译器 `bisheng`（`/usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/bin/bisheng`），链接器 `ld.lld`（`x86_64-linux/ccec_compiler/bin/ld.lld`）。
- **目标平台**：Ascend 950（`dav-3510` / `npu_arch_3510`）。
- **CANN 安装根**：`/usr/local/Ascend/cann-9.1.0-new/cann-9.1.0`（`asc` → 符号链接 `x86_64-linux/asc`）。
- **环境变量**：`source /usr/local/Ascend/cann-9.1.0-new/cann-9.1.0/set_env.sh` 注入 `ASCEND_HOME_PATH`。
- **相关文件**（与 trace 实际读/写一一对应）：
  - `asc/include/c_api/asc_simd.h`（公共入口头，`#include` 拉起整条内部头链）
  - `asc/include/c_api/vector_datamove/vector_datamove.h`（第 26 行拉入 impl 头）
  - `asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h`（第 50 行 `#include "instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h"`）
  - `asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h`（被报"not found"但实际存在的内部实现头）
  - 官方 mmad 样例 `CMakeLists.txt`（`include_directories("${ASCEND_HOME}/asc/")` 的 verbatim 来源）

## 设计决策

- **为何选 `asc/` 作 include 根而非 `asc/include`**：C-API 公共头 `asc_simd.h` 通过 `c_api/vector_datamove/vector_datamove.h` 跨进了 `impl/c_api/...` 实现头，而该 impl 头内部以 `instr_impl/...`（无 `c_api/` 前缀）相对 #include 兄弟实现头。只有把 `asc/` 作为根时，`impl/c_api/` 这条解析路径才自洽（`asc/` 根下并列 `impl/`，使 `instr_impl/npu_arch_3510/...` 命中 `asc/impl/c_api/instr_impl/npu_arch_3510/...`）。选窄根（如 `asc/include`）会丢失 `impl/` 解析能力，触发本 fatal error。
- **被放弃的候选**：仅加 `asc/impl/c_api/` 也能让 `instr_impl/...` 解析，但会破坏 `#include "c_api/asc_simd.h"` 的解析（需 `asc/include` 或 `asc/` 根）；同时维护两条 include 根不如直接用官方的单一 `asc/` 根。
- **当前方案属哪档路线**：`recommended_fastpath`——直接复用官方样例配置，零试错。

## 验证信号

本卡验证信号仅取 trace 内 `tool_observed`/`artifact_verified` 事件，不含 agent 口述数值。

- 编译报 `fatal error: 'instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h' file not found`，调用栈顶部为 `asc/include/c_api/asc_simd.h:44` → `vector_datamove.h:26` → `asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h:50`（来源平台：950；证据等级：tool_observed）
- CANN 目录结构 `asc -> x86_64-linux/asc`（符号链接），其下并列 `impl/` 与 `include/` 两子目录（来源平台：950；证据等级：tool_observed）
- 官方 mmad 样例 `CMakeLists.txt` 使用 `include_directories("${ASCEND_HOME}/asc/")` 可编译含 `asc_simd.h` 的 `.asc`（来源平台：950；证据等级：artifact_verified）
- 修正 include 配置后 `./demo` 运行 `EXIT CODE: 0`（编译/链接通过且可执行）（来源平台：950；证据等级：tool_observed）

## 失效边界

- **A2（`dav-2201` / `npu_arch_2201`）与 A3**：CANN 安装中 `asc/impl/c_api/instr_impl/npu_arch_2201/...` 同名结构存在（本 trace 在查找 `asc_copy_l12l0b_trans` impl 时观测到 `npu_arch_2201` 目录，tool_observed），故同型 fatal error 推断会在 A2/A3 上以 `npu_arch_2201` 路径出现，同一 `include_directories("${ASCEND_HOME}/asc/")` 修正推断可迁移——**未验证，推断可迁移**。
- **CANN 大版本切换**：impl 头的内部相对 #include 写法属 CANN 内部实现，跨大版本（如 9.0→9.1→未来 9.2）可能调整；升级后若报错文本变化，需重跑判别探针确认文件仍实际存在再下结论。
- **非 C-API 工程**：纯 C++ Tensor API（不 `#include "c_api/asc_simd.h"`）不拉入本头链，不触发本判别。

## 坏实践

- 见到 `fatal error: '...asc_set_gm2ub_pad_impl.h' file not found` 就重装 CANN——文件实际存在，重装无效且浪费时间。
- 把 include 根设为 `asc/include`（看似"正解"）会丢失 `impl/` 解析，仍报同一 fatal error。
- 手动改 `vector_datamove_impl.h` 里的 `#include` 路径——该文件是 CANN 内部实现头（`#warning ... must not be used directly`），改它会在升级时被覆盖且违反使用约束。

## 误诊

- agent 第一反应曾怀疑"impl 文件在 `x86_64-linux` 子目录、include 路径指错 arch 子目录"（reasoning 口述，`agent_reported`），后经 `ls` 确认 `asc` 是符号链接、`x86_64-linux/asc` 与 `asc` 同址，排除该猜测。真实根因是 include **根层级**选错（缺 `impl/` 解析能力），而非 arch 子目录错配。