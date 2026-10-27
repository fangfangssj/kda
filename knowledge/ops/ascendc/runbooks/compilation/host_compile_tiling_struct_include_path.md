---
type: "Runbook"
title: host 编译找不到 tiling_struct.h：补 include 路径
description: 自定义算子 host tiling 源码以裸文件名 include 放在 op_kernel/arch35/ 的 TilingData 结构体头时，打包构建 libascend_all_ops.so 报 fatal error No such file or directory；须给全部三个 host 编译目标（HOST_OPS_CXX_FLAGS、cust_optiling_obj、cust_proto_obj）补 op_kernel/arch35 include 路径。
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# host 编译找不到 tiling_struct.h：补 include 路径

## 现象

在为 Ascend950（arch35，CANN 9.2.0）上的自定义算子实现公共 Tiling 后运行算子打包构建 `./build.sh <Op>_package`，host 侧 `libascend_all_ops.so` 编译立即失败，报错原文：

```
-- Opbuild: compiling libascend_all_ops.so ...
CMake Error at CMakeLists.txt:143 (message):
  Failed to compile libascend_all_ops.so: In file included from
  .../op_host/arch35/<op>_tiling_arch35.cpp:30:
  .../op_host/arch35/../<op>_tiling_host.h:31:10:
  fatal error: <op>_tiling_struct.h: No such file or directory
     31 | #include "<op>_tiling_struct.h"
        |          ^~~~~~~~~~~~~~~~~~~~~~~
  compilation terminated.
```

当时的代码状态：TilingData 结构体头 `<op>_tiling_struct.h` 放在 `op_kernel/arch35/`（kernel 侧目录），而公共 tiling 纯计算层 `op_host/<op>_tiling_host.h` 用裸文件名 `#include "<op>_tiling_struct.h"` 引用它；arch35 glue（`op_host/arch35/<op>_tiling_arch35.cpp`）虽然用相对路径 `../../op_kernel/arch35/...` 能找到，但 glue 又 include 了 `../<op>_tiling_host.h`，后者在 host 编译单元里解析不到该头。算子包 CMakeLists 中 host 侧编译目标（构建 `libascend_all_ops.so` 的 `execute_process` 用 `HOST_OPS_CXX_FLAGS`，以及 `cust_optiling_obj`、`cust_proto_obj` 两个 OBJECT 库）都只带了 `op_host` 相关 include 路径。

## 适用场景

- 工程形态：算子仓（ops-registry 风格 `operators/<Op>_package`）+ `./build.sh <Op>_package` 打包构建，host tiling 与 kernel 共享同一份 TilingData 结构体头。
- API 路径：CMakeLists.txt 的 `HOST_OPS_CXX_FLAGS`、`target_include_directories(cust_optiling_obj ...)`、`target_include_directories(cust_proto_obj ...)`；头文件布局 `op_host/<op>_tiling_host.h` ↔ `op_kernel/arch35/<op>_tiling_struct.h`。
- 验证过的平台与 CANN 版本：Ascend950（arch35/DAV_3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该构建布局下的 include 路径问题与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：TilingData 头与 host 源码同目录的工程（不会触发）；其他 CANN 版本的 CMake 结构（机制相同但目标名/变量名可能变化，未验证）。

## 根因

host 侧 tiling 源码引用的 TilingData 结构体头位于 `op_kernel/arch35/`，而打包 CMake 中三个 host 编译目标（`HOST_OPS_CXX_FLAGS` 指定的 `libascend_all_ops.so` 编译、`cust_optiling_obj`、`cust_proto_obj`）的 include 路径都没有 `op_kernel/arch35`，导致裸文件名 include 解析失败。

## 解决配方

修复前（CMakeLists.txt，host 目标缺 `op_kernel/arch35`，三个目标各自为政）：

```cmake
set(HOST_OPS_CXX_FLAGS
    -g -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -O0
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    -I ${CANN_INC}
    -I ${CANN_ARCH_INC}
    -I ${CANN_ARCH_INC}/pkg_inc
    -I ${CANN_HOME}/x86_64-linux/pkg_inc/base
    -D BUILD_SOC_VERSION=Ascend950
)
```

修复后（三个 host 编译目标全部补 `-I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35`）：

```cmake
set(HOST_OPS_CXX_FLAGS
    -g -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -O0
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    -I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35
    -I ${CANN_INC}
    -I ${CANN_ARCH_INC}
    -I ${CANN_ARCH_INC}/pkg_inc
    -I ${CANN_HOME}/x86_64-linux/pkg_inc/base
    -D BUILD_SOC_VERSION=Ascend950
)
```

```cmake
add_library(cust_optiling_obj OBJECT ${host_tiling_srcs} op_graph/<op>_graph_infer.cpp)
target_include_directories(cust_optiling_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_ARCH_INC}/pkg_inc
    ${CANN_HOME}/x86_64-linux/pkg_inc/base
)
```

```cmake
add_library(cust_proto_obj OBJECT ${proto_srcs})
target_include_directories(cust_proto_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_ARCH_INC}/pkg_inc
    ${CANN_HOME}/x86_64-linux/pkg_inc/base
)
```

要点：只补一个目标不够——`libascend_all_ops.so` 的 `execute_process` 编译、`cust_optiling_obj`、`cust_proto_obj` 三处 host 编译路径都要能看到该头，否则换一个构建入口（打包/opbuild）仍会失败。

## 设计决策

- 为什么把 TilingData 头放 `op_kernel/arch35/` 并让 host 裸 include：host tiling 纯计算层与 kernel 侧共用同一份 TilingData 定义（same-source 原则，TilingUT 与上板实现测同一结构体）；代价就是 host 编译目标必须显式加 kernel 侧目录到 include 路径。
- 备选：把 tiling_struct.h 挪到 `op_host/` 或复制一份——被放弃，因为会造成 kernel/host 两份定义漂移，违背 same-source。
- 当前方案路线档位：recommended_fastpath（正确性必需的构建配置修正）。

## 系统信息

- 硬件/软件：Ascend950（arch35/DAV_3510），CANN 9.2.0（`ASCEND_HOME_PATH` 指向 cann-9.2.0 安装），GNU 13.3.0 host 编译器。
- 相关文件：`operators/<Op>_package/CMakeLists.txt`（修复点）、`op_host/<op>_tiling_host.h:31`（裸 include 触发点）、`op_host/arch35/<op>_tiling_arch35.cpp`（glue）、`op_kernel/arch35/<op>_tiling_struct.h`（被引用头）。

## 验证信号

以下信号来自该算子公共 Tiling 落地会话：补齐 include 路径后重新打包构建并逐级回归。

- 修复前：`./build.sh <Op>_package` → `CMake Error ... Failed to compile libascend_all_ops.so: fatal error: <op>_tiling_struct.h: No such file or directory`，EXIT 1（来源平台：950；证据等级：tool_observed）。
- 修复后：`./build.sh <Op>_package` → `SUCCESS`，EXIT 0（来源平台：950；证据等级：tool_observed）。
- 修复后：构建日志 grep `do not registe tiling struct` 0 命中（tiling 结构体注册门禁通过）（来源平台：950；证据等级：tool_observed）。
- 修复后：同源 TilingUT 17/17 通过；aclnn 冒烟 3/3（fp32/fp16/bf16 两段式调用 OK，回归未退化）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"host 编译目标 include 路径缺失"这一根因；若报错行号落在 kernel 侧编译（ascendc_kernels 目标）则不是本卡场景。
- CMake 目标名（`cust_optiling_obj`/`cust_proto_obj`）与 `HOST_OPS_CXX_FLAGS` 变量名是该算子包模板的命名，其他模板可能不同——迁移时按"三个 host 编译入口都要补"的机制找对应目标（未验证，推断可迁移）。
- 若未来把 tiling_struct.h 移回 `op_host/`，本修复成为多余但无害的 include 路径。

**未决项**：

- 无（构建失败已闭环修复并回归）。
## 坏实践

- 只给 `cust_optiling_obj` 补路径就收工——`libascend_all_ops.so` 的 `execute_process` 编译与 `cust_proto_obj` 还会再失败。
- 在 tiling_host.h 里写死 `../op_kernel/arch35/...` 相对路径来绕过——glue 里可以，公共头里会随 include 位置变化而漂移，不如统一 include 路径。

## 误诊

- 无（本失败一眼可见根因，无错误归因轮次）。
