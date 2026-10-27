---
type: "Runbook"
title: "高阶 API tiling 头传递依赖须补 include"
description: "op_host include \"topk/topk_tiling.h\"/\"softmax/softmax_tiling.h\" 后其内部裸名依赖（topk_utils_constants.h、softmax_utils.h 等）位于 asc/include/adv_api/{sort,activation} 与 tiling 子目录，不在 npu_op_code_gen 默认路径；须给 COMPILE_OPTIONS、cust_optiling、cust_op_proto 三处目标补 -I。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.api_layer.high_level_api"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# 高阶 API tiling 头传递依赖须补 include

## 现象

两个独立算子工程（均为 Ascend950PR / arch35，CANN 9.1.0）在 op_host 侧引用 SoftMax/TopK 高阶 API 的 tiling 头时撞到同一报错。`op_host/arch35/<op>_tiling.cpp` 里按官方头文件路径写：

```cpp
#include "softmax/softmax_tiling.h"
#include "topk/topk_tiling.h"
```

这两个头位于 `$ASCEND_HOME_PATH/aarch64-linux/asc/include/tiling/` 下（默认 include 路径已覆盖）。首次执行 `bash build.sh --soc=ascend950 -j8` 编译失败，CMake 在 `npu_op_code_gen` 阶段报：

```text
In file included from .../op_host/arch35/<op>_tiling.cpp:29:
<ASCEND_HOME>/aarch64-linux/asc/include/tiling/topk/topk_tiling.h:20:10:
  fatal error: topk_utils_constants.h: No such file or directory
     20 | #include "topk_utils_constants.h"
        |          ^~~~~~~~~~~~~~~~~~~~~~~~
```

即 `topk_tiling.h` 自身 `#include "topk_utils_constants.h"` 的传递依赖无法解析。实际磁盘布局（find 探针确认，两个工程观测一致）：`topk_utils_constants.h` 位于 `asc/include/adv_api/sort/`（softmax 侧的 `softmax_utils.h` 位于 `asc/include/adv_api/activation/`）；`topk_tiling.h` 还依赖 `kernel_tiling/kernel_tiling.h`，位于 `ascendc/include/highlevel_api`。另一工程观测到 `topk_tilingdata.h` 与 tiling 头同在 `asc/include/tiling/topk/` 目录。当时 op_host/CMakeLists.txt 的 include 路径只有 `asc/include`、`asc/include/tiling`、`ascendc/include/highlevel_api`、`pkg_inc` 系列等上层目录，不含这些子目录。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子仓 op_host tiling 开发，构建入口 `bash build.sh --soc=ascend950 -j8`；op_host 侧调用 SoftMax/TopK 高阶 API 的 TilingFunc。
- API 路径：`asc/include/tiling/topk/topk_tiling.h`、`asc/include/tiling/softmax/softmax_tiling.h`（及其内部依赖 `topk_utils_constants.h`、`softmax_utils.h` 等 adv_api 头、`kernel_tiling/kernel_tiling.h`）；`npu_op_code_gen` 的 `COMPILE_OPTIONS` 与 `npu_op_library` 目标的 `target_include_directories`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / arch35，交叉编译场景 `aarch64-linux` 为指向 `x86_64-linux` 的符号链接）+ CANN 9.1.0（两个独立工程复现）。
- 适用粒度：operator_family——任何在 op_host 中使用高阶 API tiling 头的自定义算子。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：op_kernel 侧（device 编译单元）的 include 集合由 kernel CMake 单独管理，本卡只覆盖 op_host 的 `npu_op_code_gen`/`cust_optiling`/`cust_op_proto` 三个目标；Matmul 高阶 API 的 kernel 侧 ini 机制见 [Matmul 高阶 API 头不可见卡](matmul_highlevel_api_include_paths_ini.md)；其他高阶 API（非 sort/activation 目录）的依赖路径未逐一验证；其他 CANN 版本的目录布局未验证。

## 根因

高阶 API 的 tiling 头以裸文件名引入自身依赖（`topk_tiling.h` 内 `#include "topk_utils_constants.h"`），而这些依赖头实际分散在多个 arch 专属 include 目录（`asc/include/adv_api/sort`、`asc/include/adv_api/activation`、`ascendc/include/highlevel_api`、`asc/include/tiling/<api>` 子目录）；`npu_op_code_gen` 的默认 COMPILE_OPTIONS 只包含 `asc/include/tiling` 等上层目录，不含这些子目录，预处理器找不到被依赖头。

## 判别方法

1. 报错特征：`fatal error: <高阶API内部依赖头>: No such file or directory`，且报错位置在 CANN 安装目录内的头文件（非算子自己的代码）。
2. 用 find 定位缺失头所在目录，确认需要追加的 include 路径（两个工程实际执行的探针）：

```bash
cd $ASCEND_HOME_PATH/aarch64-linux
find . -name "topk_utils_constants.h" 2>/dev/null
# ./asc/include/adv_api/sort/topk_utils_constants.h
ls asc/include/tiling/topk/
find $ASCEND_HOME_PATH/aarch64-linux/ascendc/include -name "kernel_tiling*"
```

3. 反面判别：同批出现 `fatal error: exe_graph/runtime/compile_shape_context.h: No such file or directory` 这类报错时先核实该头是否真实存在——有按记忆误写不存在 include 的情况，删掉该行即可，不要盲目加路径。

## 解决配方

修改 `op_host/CMakeLists.txt`，在 `npu_op_code_gen` 的 `COMPILE_OPTIONS` 以及 `cust_optiling`、`cust_op_proto` 的 `target_include_directories` 中补 adv_api 子目录（三个目标编译同一批 host 源，缺一不可）。

after（`npu_op_code_gen` 段，新增 `asc/include/adv_api/sort` 与 `asc/include/adv_api/activation` 两行；另一工程按 find 结果另补了 `asc/include/tiling/topk`）：

```cmake
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling/topk
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/sort
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/activation
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)
```

`cust_optiling` 与 `cust_op_proto` 的 `target_include_directories` 做同样追加：

```cmake
target_include_directories(cust_optiling PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling/topk
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/sort
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/activation
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)
```

## 设计决策

- 三处都要加（npu_op_code_gen 的 COMPILE_OPTIONS、cust_optiling、cust_op_proto）：op_host 源被 autogen、tiling 库、proto 库三个目标重复编译，漏掉任何一个目标都会在同一头文件上再次失败。一个工程第一轮只改了前两处，后续构建在 cust_op_proto 目标上复现同类错误后又补了第三处。
- 同时补 `sort` 与 `activation` 两个子目录：SoftMax 与 TopK 的 tiling 头分属两个 adv_api 目录，融合算子一次补齐，避免二次失败。
- 路径用 `$ENV{ASCEND_HOME_PATH}` 环境变量而非硬编码安装目录，保持与构建脚本 `--soc`/环境解耦。
- 判定依赖目录的方法是 find/ls 探针，不猜路径。

## 预防

- op_host 侧使用任一 AscendC 高阶 API（TopK/Sort/MrgSort/GatherMask/SoftMax 等）前，先用 find 确认其头文件及传递依赖所在目录，把相关目录一次性加齐三个目标（autogen COMPILE_OPTIONS + tiling 库 + proto 库）。
- 看到"报错位置在 CANN 安装目录内的头"时，优先怀疑 include 路径缺失，而不是去改 CANN 头或复制头文件。
- 只 include 确认存在的头：写 infershape/tiling 前可 `find $ASCEND_HOME_PATH -path "*<目录>*"` 核对头文件名，不要凭记忆写 include。

## 验证信号

来源说明：两个独立算子工程（Softmax+TopK 融合算子、TopK 归约类算子）首次构建失败到修复后的连续构建（均为 Ascend950PR + CANN 9.1.0）。

- 工程一修复前：`topk_tiling.h:20:10: fatal error: topk_utils_constants.h: No such file or directory`，构建在 `npu_op_code_gen` 阶段终止（来源平台：950；证据等级：tool_observed）。
- 工程一修复后同头文件解析通过：下一次构建错误变为 tiling.cpp 内 `SoftMaxTilingFunc was not declared in this scope`（命名空间问题，属另一根因），include 类 fatal error 消失（来源平台：950；证据等级：tool_observed）。
- 工程二修复前：同错 `fatal error: topk_utils_constants.h: No such file or directory`（topk_tiling.h:20）（来源平台：950 编译链；证据等级：tool_observed）。
- 工程二修复后：追加路径后重新构建，`topk_utils_constants.h` 相关 fatal error 不再出现，编译推进到 tiling 源自身的 API 用法错误（另卡处理）；三处路径全部补齐后最终构建成功生成算子包（来源平台：950 编译链；证据等级：tool_observed）。

## 失效边界

- 目录布局结论验证于 cann-9.1.0（`asc/include/adv_api/{sort,activation}`、`asc/include/tiling/topk`）；其他 CANN 版本高阶 API 依赖目录可能变化（未验证，迁移时先 find 确认）。
- 本卡只解决 host 侧 tiling 头依赖；若 op_kernel 侧也需要这些头，kernel 的 CMake 需另行补路径（未验证）。
- 两个工程对 `topk_utils_constants.h` 的实际落点观测分别为 `asc/include/adv_api/sort/`（find 直证）与 tiling 头同目录（ls 列证），可能存在多副本/符号链接；追加路径前以自己环境的 find 结果为准。

**未决项**：

- `topk_utils_constants.h` 等头为何放在 adv_api 而非 tiling 目录、是否有官方聚合头可一次性引入，未查证；如后续 CANN 版本提供统一入口可简化 CMake。
## 坏实践

- 把 CANN 安装目录里的头文件复制进算子仓库来"解决"找不到依赖。
- 只给 npu_op_code_gen 加路径、漏掉 cust_op_proto/cust_optiling 目标（一个工程实际踩过：第二轮才补齐 proto 目标）。
- 凭记忆写不存在的 include（`exe_graph/runtime/compile_shape_context.h`）：同一次构建因此多报一个 fatal error，浪费一轮定位。

## 误诊

- 无（两例 include 失败均直接由 find 定位；同批误写头行被删掉后其余报错模式不变）。
