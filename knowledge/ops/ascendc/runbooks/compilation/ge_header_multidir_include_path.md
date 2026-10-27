---
type: "Runbook"
title: GE 头文件多目录布局致 GEIR 示例编译失败
description: "CANN 9.1.0 x86_64 安装把 GE 头/库分散在 include/graph（壳文件）、x86_64-linux/include/{graph,ge}、opp/built-in/op_graph/inc 与 x86_64-linux/lib64 多目录，且 graph.h 内部按前缀自引用需父目录进 include 路径；示例模板 CMakeLists 只搜 aarch64-linux/compiler 路径，GE IR 示例报 fatal error: graph/operator.h: No such file or directory，补齐 x86_64-linux 路径与父目录后编译通过。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T04:13:22Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# GE 头文件多目录布局致 GEIR 示例编译失败

## 现象

为自定义算子（Ascend950PR 目标，vendor 包已安装）开发 GE IR（图模式）调用示例 `examples/test_geir_*.cpp`（`#include "graph/graph.h"` / `ge/ge_api.h`，链接 libgraph、libge_runner），按 add_example 参考模板写好 `CMakeLists_geir.txt` 后编译，aclnn 直调示例（同一 examples/ 目录、CMakeLists.txt）一次通过，GE IR 示例编译即报：

```text
In file included from .../examples/test_geir_<op>.cpp:21:
${ASCEND_HOME_PATH}/include/graph/graph.h:19:10: fatal error: graph/operator.h: No such file or directory
   19 | #include "graph/operator.h"
      |          ^~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [...] Error 1
```

特征：`graph.h` 本身已被找到（报错位置就在 CANN 安装目录的 `include/graph/graph.h:19`），挂掉的是它内部的 `#include "graph/operator.h"`——报错文件在 CANN 安装目录内、报错行为带 `graph/` 前缀的包含，说明不是没装 CANN，而是 include 根没进搜索路径。

部分示例源码用裸名 `#include "graph.h"`，此时还会出现两轮互相矛盾的报错（include path 指向 `include/graph` 时报 `graph/operator.h` 找不到；改成父目录 `include` 后又报 `graph.h` 找不到）——只给裸名目录或只给父目录都会漏掉另一层解析。

## 适用场景

- 工程形态：自定义算子的 GE IR 图模式调用示例 / 任何在 x86_64 主机上引用 `graph/graph.h`、`ge/ge_api.h` 并链接 libgraph/libge_runner 的宿主 C++ 工程（CMake）。
- API 路径：CANN 9.1.0 x86_64 安装布局——`${ASCEND_HOME_PATH}/include/graph/`（graph.h 壳文件）、`${ASCEND_HOME_PATH}/x86_64-linux/include/graph/`（operator.h、operator_reg.h、tensor.h、types.h）、`${ASCEND_HOME_PATH}/x86_64-linux/include/ge/`（ge_api.h、ge_ir_build.h）、`${ASCEND_HOME_PATH}/opp/built-in/op_graph/inc/`（array_ops.h；部分安装 op_proto/inc 下亦有同名）、`${ASCEND_HOME_PATH}/x86_64-linux/lib64/`（libgraph.so、libge_runner.so、libgraph_base.so、libge_compiler.so）；CMake `find_path`/`find_library`/`target_include_directories`。
- 验证过的平台与 CANN 版本：Ascend950PR 目标算子包 + CANN 9.1.0（x86_64-linux 安装布局），十条独立 trace 复现（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 x86_64 安装布局下的 GE IR 示例工程通用，与具体算子无关。
- 路线优先级：recommended_fastpath（一次性把 include/库目录配全，多架构路径并存兼容）。
- 明确不适用 / 未验证场景：aarch64 原生安装（头文件在 `aarch64-linux/include/graph/`，模板原路径即为其设计目标，本卡路径表已兼容但未在该环境验证）；仅用 aclnn 头文件（不需要 graph 头）的目标不受影响；其他 CANN 版本的目录布局未验证（推断可迁移）。

## 根因

CANN 9.1.0 在 x86_64 主机上的安装布局把 GE 头文件与库分散在多棵不嵌套的目录树，且包含指令两种写法并存：

1. **架构前缀**：完整 GE 头文件在 `${ASCEND_HOME_PATH}/x86_64-linux/include/{graph,ge}`；GE 运行时库 `libge_runner.so`/`libge_runner_v2.so`/`libge_compiler.so` 只在 `x86_64-linux/lib64/`（实测一份安装顶层 `lib64` 仅有 libgraph*.so 系列、无 libge_runner；另一份安装 libgraph 在两处均有——不同安装间有差异，搜索路径须全覆盖）。模板 CMakeLists 的 find_path/find_library 只列 `aarch64-linux`、`compiler`、顶层 `include/lib64` 三组路径，在 x86_64 主机上 miss 或命中残缺副本（顶层 `include/graph/` 只有 graph.h 等少量文件，无 operator.h 完整依赖闭包）。
2. **父目录缺失**：`graph.h` 内部以带前缀方式 `#include "graph/operator.h"`——解析它需要**父目录**（`.../include` 或 `.../x86_64-linux/include`）在 include 路径中；`find_path(GE_GRAPH_INCLUDE_DIR NAMES graph.h)` 找到的是 `.../include/graph` 目录本身，只加它只能解析源文件里的 `#include "graph.h"`，无法解析 graph.h 内部的带前缀包含。
3. **裸名包含并存**：示例源码若用裸名 `#include "graph.h"`，还需要 `.../graph` 目录本身也在 include 路径——于是出现"加子目录报 operator.h 找不到、加父目录报 graph.h 找不到"的两轮矛盾错误，两类目录都要。
4. **built-in 头落点**：`array_ops.h`（op::Data/op::Const 定义）在 `opp/built-in/op_graph/inc/`（部分安装在 `opp/built-in/op_proto/inc/` 亦有同名），模板只写其一会漏。

## 解决配方

修复 = find_library/find_path 搜索清单补 `x86_64-linux` 路径（保留 aarch64/compiler/lib64 兜底，多架构并存）+ include 目录同时加"find_path 结果、其父目录、架构 include 根、两个 built-in 头目录"。修复后的完整 `CMakeLists_geir.txt`（verbatim，替换可执行名与源文件名即可复用）：

```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

project(GE_IR_EXAMPLE)

if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
    get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

file(GLOB files CONFIGURE_DEPENDS
     test_geir_<op>.cpp
)

add_executable(test_geir_<op> ${files})

find_library(GRAPH_LIBRARY graph
    PATHS
        ${ASCEND_PATH}/aarch64-linux/lib64
        ${ASCEND_PATH}/x86_64-linux/lib64
        ${ASCEND_PATH}/compiler/lib64/stub
        ${ASCEND_PATH}/lib64
)
find_library(GE_RUNNER_LIBRARY ge_runner
    PATHS
        ${ASCEND_PATH}/aarch64-linux/lib64
        ${ASCEND_PATH}/x86_64-linux/lib64
        ${ASCEND_PATH}/compiler/lib64/stub
        ${ASCEND_PATH}/lib64
)
find_library(GRAPH_BASE_LIBRARY graph_base
    PATHS
        ${ASCEND_PATH}/aarch64-linux/lib64
        ${ASCEND_PATH}/x86_64-linux/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/lib64
)
find_library(GE_COMPILER_LIBRARY ge_compiler
    PATHS
        ${ASCEND_PATH}/aarch64-linux/lib64
        ${ASCEND_PATH}/x86_64-linux/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/lib64
)

if(NOT GRAPH_LIBRARY OR NOT GE_RUNNER_LIBRARY)
    message(FATAL_ERROR "未找到 GE IR 依赖库 (libgraph/libge_runner)，请检查 CANN 安装")
endif()

target_link_libraries(test_geir_<op> PRIVATE
     ${GRAPH_LIBRARY}
     ${GE_RUNNER_LIBRARY}
)
if(GRAPH_BASE_LIBRARY)
    target_link_libraries(test_geir_<op> PRIVATE ${GRAPH_BASE_LIBRARY})
endif()
if(GE_COMPILER_LIBRARY)
    target_link_libraries(test_geir_<op> PRIVATE ${GE_COMPILER_LIBRARY})
endif()

find_path(GE_GRAPH_INCLUDE_DIR
    NAMES graph.h
    PATHS
        ${ASCEND_PATH}/aarch64-linux/include/graph
        ${ASCEND_PATH}/x86_64-linux/include/graph
        ${ASCEND_PATH}/compiler/include/graph
        ${ASCEND_PATH}/include/graph
)
find_path(GE_API_INCLUDE_DIR
    NAMES ge_api.h
    PATHS
        ${ASCEND_PATH}/aarch64-linux/include/ge
        ${ASCEND_PATH}/x86_64-linux/include/ge
        ${ASCEND_PATH}/compiler/include/ge
        ${ASCEND_PATH}/include/ge
)

# graph.h 内部 #include "graph/operator.h"，需要父目录在 include 路径中
get_filename_component(GE_INCLUDE_PARENT ${GE_GRAPH_INCLUDE_DIR} DIRECTORY)

target_include_directories(test_geir_<op> PRIVATE
     ${GE_GRAPH_INCLUDE_DIR}
     ${GE_API_INCLUDE_DIR}
     ${GE_INCLUDE_PARENT}
     ${ASCEND_PATH}/opp/built-in/op_graph/inc/
     ${ASCEND_PATH}/opp/built-in/op_proto/inc/
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${ASCEND_PATH}/aarch64-linux/include
     ${ASCEND_PATH}/x86_64-linux/include
     ${ASCEND_PATH}/compiler/include
)
```

要点：

- `get_filename_component(... DIRECTORY)` 从 find_path 结果推导父目录，兼容 find_path 在不同布局命中的目录；同时仍显式追加 `${ASCEND_PATH}/x86_64-linux/include` 双保险。
- 示例源码用裸名 `#include "graph.h"` 时，`${GE_GRAPH_INCLUDE_DIR}`（graph/ 目录本身）与父目录都必须保留。
- 改完 CMakeLists 必须**清空 build 目录重跑 cmake**——复用旧 build 目录时 CMakeCache 仍指向旧目标（实测曾把 geir 构建跑成 aclnn 示例）。
- 运行 GEIR 二进制时 `LD_LIBRARY_PATH` 同样需要包含 `${ASCEND_HOME_PATH}/x86_64-linux/lib64`（libge_runner 在该目录）。

## 判别方法

用 find 直接定位真实布局（不要信模板的搜索路径）：

```bash
find "${ASCEND_HOME_PATH}" -name "graph.h" -path "*/graph/*" -type f 2>/dev/null | head -5
find "${ASCEND_HOME_PATH}" -name "operator.h" -path "*/graph/*" 2>/dev/null | head -5
find "${ASCEND_HOME_PATH}" -name "libge_runner.so" 2>/dev/null | head -5
ls -d "${ASCEND_HOME_PATH}/"*/include 2>/dev/null   # 确认架构前缀目录（x86_64-linux/include 是 GE 头真实落点）
```

判读：报错文件路径在 CANN 安装目录内、且报错的是头文件内部的 `graph/...` 包含 → 不是 CANN 没装，而是 include 根没进搜索路径；`find` 确认 `operator.h` 只在 `x86_64-linux/include/graph/` 下即锁定本卡布局。

## 设计决策

- **加路径而非替换路径**：`x86_64-linux` 插入 PATHS、`aarch64-linux`/`compiler`/`lib64` 保留——同一份 CMakeLists 在两种宿主布局下都能配置，谁的布局谁命中。
- **`graph_base`、`ge_compiler` 做成可选链接（if 包裹）**：部分安装不提供这两个库，硬性 FATAL_ERROR 会让示例在精简安装上无法构建；libgraph/libge_runner 是构图与执行的硬依赖，缺失即报错退出。
- **`op_graph/inc` 与 `op_proto/inc` 都加入 include**：实测 array_ops.h 在不同安装中落点漂移（op_graph/inc 或 op_proto/inc），双路径都加避免猜单点。
- **不把源码统一改成 `#include "graph/graph.h"`**：示例沿用模板裸名写法时改 CMake 目录集成本更低；两种写法对应的目录集本卡均已覆盖。
- 当前方案属 recommended_fastpath：目录补齐后无其他副作用（仅多出 array_ops.h 的 2026-12 弃用告警，不阻塞）。

## 预防

- 在 x86_64 主机写任何 CANN graph/ge 构图工程，include/link 搜索一律同时列 `aarch64-linux`、`x86_64-linux`、`compiler`、根 `include/lib64` 四组，或先跑一遍 find 探针按实际命中写 PATHS。
- `find_path` 找目录型头（graph.h 位于 `graph/` 子目录）时，把它**父目录**也加进 include——头文件内部常以 `graph/xxx.h` 形式自引用。
- 从参考示例拷 CMake 后，先 `ls` 验证模板里的 PATHS 候选在本机是否存在，不要默认布局一致。
- 算子包示例交付前分别在 aclnn / GEIR 两套 CMakeLists 下各做一次干净构建（清 build 目录）。

## 验证信号

来源说明：以下信号来自十条独立 trace 的 GE IR 示例构建会话（均为 Ascend950PR 目标算子包 + CANN 9.1.0 x86_64 安装，涉及去重/分组矩阵乘/双线性插值/GCD/Gather/GridSampler3d/foreach 标量算子/Conv2d/动态量化等算子），每条为修复前后同一构建命令的对比实测。

- 修复前（多 trace 同型）：`fatal error: graph/operator.h: No such file or directory`（出错点 `include/graph/graph.h:19`），make Error 1（来源平台：950；证据等级：tool_observed）。
- 两轮矛盾错误形态：include path=include/graph 时报 `graph/operator.h` 找不到，改为父目录 include 后报 `graph.h` 找不到——两类目录都加后才通过（来源平台：950；证据等级：tool_observed）。
- 布局探针：`find` 定位 `operator.h`/`operator_reg.h` 仅存在于 `${ASCEND_HOME_PATH}/x86_64-linux/include/graph/`，`ge_api.h`/`ge_ir_build.h` 在 `x86_64-linux/include/ge/`；`compiler/include/graph` 与 `aarch64-linux/include/graph` 不存在（来源平台：950；证据等级：tool_observed）。
- 库布局探针：顶层 `lib64` 仅 libgraph*.so 系列，`libge_runner.so`/`libge_runner_v2.so`/`libge_compiler.so` 只在 `x86_64-linux/lib64/`；`compiler/lib64/stub` 不存在（另一安装中 libgraph 在两处均有，libge_runner 仍只在 x86_64-linux/lib64）（来源平台：950；证据等级：tool_observed）。
- 修复后（多 trace）：清空 build 目录重新 cmake+make，`[100%] Built target test_geir_*` / `[100%] Linking CXX executable`，仅余 array_ops.h 的 2026-12 弃用 `#warning`（来源平台：950；证据等级：tool_observed）。
- 编译推进证明：同一示例修复 include 后编译推进到源码级下一错误（AddGraph 重载二义性，属另一问题）——证明 include 解析已通（来源平台：950；证据等级：tool_observed）。
- 端到端：修复后 `bash run.sh --graph` 在 Ascend950PR 上图编译 + NPU 执行 + 精度比对全流程 PASS（输出 dtype/shape 正确）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 目录布局以 CANN 9.1.0 x86_64 安装实测；不同版本/安装方式（完整包 vs 最小安装）头文件与库的分布可能变化（如统一迁入 compiler/include 或头文件归并），移植时用 find 探针重新定位（未验证，推断可迁移）。
- aarch64 原生环境未验证（模板原路径即其设计目标；本卡路径表已把 aarch64-linux 并列保留）。
- 本卡只解决编译期头文件/库路径与链接；链接成功后的 `RunGraph` 运行期失败（如算子缺 InferDataType 注册）是另一层问题，见对应卡。
- `array_ops.h` 按其弃用告警将在 2026-12 拆分为 ops_proto_*.h 系列头文件，届时 include 名称需迁移（当前未到期，未验证迁移路径）。

**未决项**：

- CANN 后续版本若弃用 `x86_64-linux/` 架构子目录（或提供统一 `compiler/` 布局），需更新路径清单；可在示例 run.sh 里保留 `ls -d "${ASCEND_HOME_PATH}/"*/include` 的布局自检。
- CANN 9.1.0 将 graph 头拆分多目录的原因（打包策略 vs 架构差异）未追；跨版本布局对比未做。
## 坏实践

- 找到 `${ASCEND_PATH}/include/graph/graph.h` 就以为 GE 头文件齐了——x86_64 布局下它是壳文件，真正的子头文件在 `x86_64-linux/include/graph/`。
- 只把 `find_path` 找到的 `include/graph` 目录加进 include 路径——`graph.h` 内部以 `"graph/operator.h"` 形式引用，include 根必须是 `graph/` 的父目录。
- 照抄面向 aarch64 交叉编译的模板搜索路径到 x86_64 主机，报错后去改系统目录或复制头文件补位（污染 CANN 安装目录且升级即失效）。
- 用 `ln -s` 把 `x86_64-linux/lib64` 的库软链到顶层 `lib64` 来"修"链接。
- 复用未清空的 build 目录直接改 CMakeLists 重跑——CMakeCache 指向旧目标，构建产物张冠李戴。

## 误诊

- "include path 给错了方向"单向思维：第一轮错误后直接把 include path 换成父目录，触发第二轮镜像错误（graph.h 找不到）；正确做法是裸名目录与父目录两类都加。
- 见到 fatal error 就重装 CANN 或怀疑安装损坏——报错路径明明指向安装目录内的现存文件。
- 顶层 `include/` 下存在残缺 graph 头副本时，搜索顺序不当会再次命中残缺副本——include 顺序应让 `x86_64-linux/include` 优先（一份安装实测 `include/graph` 与 `x86_64-linux/include/graph` 双分布，operator_reg.h 仅在后者）。
