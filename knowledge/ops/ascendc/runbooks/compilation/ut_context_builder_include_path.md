---
type: "Runbook"
title: UT 编译缺 context_builder include 路径
description: 自定义算子 UT 编译报 op_tiling_context_builder.h / op_infer_shape_context_builder.h 找不到；CANN 9.2.0 下这两个头位于 x86_64-linux/include/base/context_builder/ 子目录，faker 头按裸文件名引用，UT CMakeLists 须补 include/base 与 include/base/context_builder 两个目录。
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# UT 编译缺 context_builder include 路径

## 现象

在为一个自定义算子（LSTM 类，arch35/ascend950）开发 op_host UT（TilingContextFaker + InferShapeContextFaker 测试框架）时：common 基础设施（`tiling_context_faker.h/cpp`、`infershape_context_faker.h/cpp`、`tiling_case_executor.h/cpp`、`infershape_case_executor.h/cpp`，复制自参考工程）与测试文件都已写好，首次构建：

```
cd operators/<op>/tests/ut && mkdir -p build && cd build && cmake .. -DSOC_VERSION=ascend950 && make -j$(nproc)
```

CMake 配置成功，make 编译到 80%/95% 时连续 3 个编译单元失败：

```
In file included from .../tests/ut/common/tiling_case_executor.h:14,
                 from .../tests/ut/common/tiling_case_executor.cpp:11:
.../tests/ut/common/tiling_context_faker.h:14:10: fatal error: op_tiling_context_builder.h: No such file or directory
   14 | #include "op_tiling_context_builder.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
In file included from .../tests/ut/common/infershape_context_faker.cpp:11:
.../tests/ut/common/infershape_context_faker.h:14:10: fatal error: op_infer_shape_context_builder.h: No such file or directory
   14 | #include "op_infer_shape_context_builder.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [op_host/CMakeFiles/<op>_op_host_ut.dir/build.make:146: ...] Error 1
```

当时代码状态：faker 头文件从参考工程（add_example）复制而来，`#include "op_tiling_context_builder.h"` 按裸文件名包含；UT 顶层 CMakeLists 的 `UT_COMMON_INCLUDE_DIRS` 已包含 `$ENV{ASCEND_HOME_PATH}/include`、`${ASCEND_LINUX_DIR}/include`、`${ASCEND_LINUX_DIR}/asc/include` 等目录，但不含 context_builder 子目录。

## 适用场景

- 工程形态：自定义算子 op_host UT 工程（gert::TilingContextFaker / InferShapeContextFaker + GoogleTest），CMake 独立构建。
- API 路径：`base/context_builder/op_tiling_context_builder.h`、`base/context_builder/op_infer_shape_context_builder.h`（CANN x86_64-linux include 树）；UT CMakeLists `UT_COMMON_INCLUDE_DIRS`。
- 验证过的平台与 CANN 版本：ascend950（DAV_3510/arch35）+ CANN 9.2.0（x86_64-linux 布局；与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 安装布局级问题，与具体算子无关）。
- 路线优先级：recommended_fastpath（补 include 目录是唯一正解）。
- 明确不适用 / 未验证场景：其他 CANN 版本的安装布局可能不同（未验证，推断可迁移）；`build.sh -u` 官方 UT 入口自带正确 include 时不适用；同签名报错的另一根因（整树 aarch64 前缀路径写死）见库内 [编译报 dlog_pub.h 缺失：模板硬编码 aarch64 路径](template_aarch64_hardcode_x86_64_host.md)——本卡是"缺子目录"，彼卡是"错架构前缀"，判别时先看 include 列表里有没有 `include/base/context_builder`。

## 根因

CANN 9.2.0 安装中，`op_tiling_context_builder.h` 与 `op_infer_shape_context_builder.h` 位于 `${ASCEND_HOME_PATH}/x86_64-linux/include/base/context_builder/` 子目录（经 `find $ASCEND_LINUX_DIR -name` 确认），而 faker 头文件按裸文件名 `#include "op_tiling_context_builder.h"` 引用；UT CMakeLists 的 include 目录列表缺少 `${ASCEND_LINUX_DIR}/include/base` 与 `${ASCEND_LINUX_DIR}/include/base/context_builder`，导致编译器搜索不到。

## 解决配方

在 UT 顶层 CMakeLists 的 `UT_COMMON_INCLUDE_DIRS` 中补两行（`${ASCEND_LINUX_DIR}/include/base` 与 `${ASCEND_LINUX_DIR}/include/base/context_builder`），重新 cmake + make。

修复前：

```cmake
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    ${ASCEND_LINUX_DIR}/include
    ${ASCEND_LINUX_DIR}/asc/include
    ${ASCEND_LINUX_DIR}/asc/include/tiling
    ${ASCEND_LINUX_DIR}/pkg_inc
    ${ASCEND_LINUX_DIR}/pkg_inc/op_common
    ${ASCEND_LINUX_DIR}/pkg_inc/base
    ${ASCEND_LINUX_DIR}/pkg_inc/exe_graph
    ${ASCEND_LINUX_DIR}/pkg_inc/graph
)
```

修复后：

```cmake
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    ${ASCEND_LINUX_DIR}/include
    ${ASCEND_LINUX_DIR}/include/base
    ${ASCEND_LINUX_DIR}/include/base/context_builder
    ${ASCEND_LINUX_DIR}/asc/include
    ${ASCEND_LINUX_DIR}/asc/include/tiling
    ${ASCEND_LINUX_DIR}/pkg_inc
    ${ASCEND_LINUX_DIR}/pkg_inc/op_common
    ${ASCEND_LINUX_DIR}/pkg_inc/base
    ${ASCEND_LINUX_DIR}/pkg_inc/exe_graph
    ${ASCEND_LINUX_DIR}/pkg_inc/graph
)
```

## 设计决策

- 修复选择补 include 目录而非改 faker 头文件为相对路径 include（如 `#include "base/context_builder/op_tiling_context_builder.h"`）：保持 faker 基础设施与参考工程（add_example）一致，便于后续同步；include 目录收敛在 CMakeLists 一处。
- 若顺带编译 tiling.cpp，还需 `${ASCEND_LINUX_DIR}/include/kernel_tiling` 解析（本环境该头实际位于 `$ASCEND_LINUX_DIR/include/kernel_tiling/kernel_tiling.h`，已被 `${ASCEND_LINUX_DIR}/include` 覆盖）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 工具链：CANN 9.2.0（`ASCEND_HOME_PATH` 指向 cann-9.2.0 安装，x86_64-linux 布局）、gcc 13.3.0、cmake 3.20.5。
- 目标平台：ascend950（DAV_3510/arch35），`cmake .. -DSOC_VERSION=ascend950`。
- 相关文件：`operators/<op>/tests/ut/CMakeLists.txt`（本次修改）、`tests/ut/common/tiling_context_faker.h`、`tests/ut/common/infershape_context_faker.h`（报错包含点）。

## 验证信号

本验证来自该算子迭代一 UT 开发：补 include 路径后同一构建目录重新配置构建并运行测试。

- 修复前：`make -j` 三个编译单元 `fatal error: op_tiling_context_builder.h / op_infer_shape_context_builder.h: No such file or directory`，Error 1（来源平台：950；证据等级：tool_observed）。
- 修复后：`cmake .. -DSOC_VERSION=ascend950 && make -j` 构建成功，仅剩 CANN 头文件 deprecation pragma 警告（ge_error_codes.h 等，提示 2027-06 后移除，不影响编译）（来源平台：950；证据等级：tool_observed）。
- UT 全量 26/26 通过（InferShape 8 + Tiling 18），`bash run.sh` 从清理构建目录到运行全流程复跑亦 26/26（来源平台：950；证据等级：tool_observed）。

## 失效边界

- CANN 9.2.0 x86_64-linux 布局实测；其他 CANN 版本/架构子目录（aarch64-linux）布局未验证（推断可迁移）。
- 该修复只覆盖 faker 体系对 context_builder 头的解析；若 UT 还引入其他 CANN 子树头文件，需按同样方式追加对应 include 目录。

**未决项**：

- 无（本根因已闭合到构建通过 + 全量 UT 通过）。
## 坏实践

- 从参考工程复制 UT 基础设施后直接构建，不核对参考工程 CMakeLists 的 include 目录全集——参考工程能编过是因为它的 CMakeLists 已包含这些子目录，裸文件名 include 依赖的是 include 路径而非文件相邻关系。

## 误诊

- 无。
