---
type: "Runbook"
title: "aclnn bfloat16.h 内嵌 fp16_t.h 编译失败"
description: "宿主测试 #include \"aclnn/opdev/bfloat16.h\" 时，该头第 18 行内嵌无前缀的 #include \"opdev/fp16_t.h\" 解析不到而 fatal error；把 CANN 安装根下的 include/aclnn 子目录加入 include_directories 即编译通过。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# aclnn bfloat16.h 内嵌 fp16_t.h 编译失败

## 现象

为一个 ACLNN 算子仓框架生成的自定义算子（arch35/Ascend 950，CANN 9.2.0）编写 aclnn 集成测试：测试是宿主侧 C++ 程序（x86_64-linux，CMake 工程），需要用 CANN 官方宿主 bfloat16 类型 `op::bfloat16` 构造 bf16 输入，于是测试源文件包含：

```cpp
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn/opdev/bfloat16.h"
#include "aclnn_bench_harness_engram_gate_fusion.h"
```

当时 `tests/aclnn/CMakeLists.txt` 的 `include_directories` 只含 CANN 安装根的 `x86_64-linux/include`、`x86_64-linux/include/aclnnop`、`include` 和 vendor 包的 `op_api/include`。在 `tests/aclnn` 下执行 `rm -rf build && mkdir build && cd build && cmake .. && make`，编译该测试源文件时直接 fatal error：

```
In file included from /data1/.../operators/engram_gate_fusion_package/tests/aclnn/test_aclnn.cpp:34:
/home/<user>/cann/Ascend/cann-9.2.0/x86_64-linux/include/aclnn/opdev/bfloat16.h:18:10: fatal error: opdev/fp16_t.h: No such file or directory
   18 | #include "opdev/fp16_t.h"
      |          ^~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [CMakeFiles/test_aclnn.dir/build.make:76: CMakeFiles/test_aclnn.dir/test_aclnn.cpp.o] Error 1
```

注意报错点不在测试源文件自己写的 include，而在 CANN 官方头文件 `bfloat16.h` 的第 18 行。

## 适用场景

- 工程形态：aclnn/ST 宿主测试构建（CMake 工程，GNU 13 编译器，链接 CANN 9.2.0 x86_64-linux 头文件与 vendor `libcust_opapi.so`）。
- API 路径：`#include "aclnn/opdev/bfloat16.h"`（CANN 安装目录 `<CANN_HOME>/x86_64-linux/include/aclnn/opdev/bfloat16.h`，用于宿主侧 `op::bfloat16` 类型）。
- 验证过的平台与 CANN 版本：950 + CANN 9.2.0（与 frontmatter `platforms` 一致；本问题属宿主编译期，与芯片型号无关，但仅在 9.2.0 布局下实测）。
- 适用粒度：platform_wide——同一 CANN 安装布局下所有宿主侧使用 `op::bfloat16` 的测试工程。
- 路线优先级：recommended_fastpath（加 aclnn 子目录到 include 路径，一次修复、无副作用）。
- 明确不适用 / 未验证场景：kernel 侧（`op_kernel` 内 bfloat16 由 `kernel_operator.h` 框架自带，不经此头，不适用）；改用 `op_common/op_host/util/bfloat16.h`（CANN 安装内另一处 bfloat16 定义）时包含路径需求不同，本卡结论不适用；其他 CANN 版本的头文件布局未验证。

## 根因

单句归因：CANN 9.2.0 的头文件 `x86_64-linux/include/aclnn/opdev/bfloat16.h` 第 18 行以内嵌相对路径 `#include "opdev/fp16_t.h"` 引依赖（不带 `aclnn/` 前缀），该路径只有相对 `.../include/aclnn` 目录才能解析；测试工程只把 CANN 安装根 `.../include` 放进 include 路径时，这一内嵌 include 无法解析。

机制链（不跳步）：

1. 测试源 `#include "aclnn/opdev/bfloat16.h"` 经 `-I${CANN_HOME}/x86_64-linux/include` 正常命中头文件——第一层包含没问题。
2. 预处理器进入 `bfloat16.h` 后遇到 `#include "opdev/fp16_t.h"`：带引号的 include 先搜**当前文件所在目录**（`.../include/aclnn/opdev/` 下没有二级 `opdev/` 子目录，找不到），再依次搜各 `-I` 路径（`.../x86_64-linux/include`、`.../x86_64-linux/include/aclnnop`、`.../include` 下都不存在直接的 `opdev/fp16_t.h`）。
3. 两轮都落空 → `fatal error: opdev/fp16_t.h: No such file or directory`，编译终止。
4. 修复点：`.../include/aclnn` 目录下确实存在 `opdev/fp16_t.h`（会话内 `find` 输出直接确认），把它加入 include 路径后，`opdev/fp16_t.h` 相对 `-I.../include/aclnn` 可解析。

CANN 9.2.0 `aclnn/opdev/bfloat16.h` 开头原文（会话内 `head -80` 输出节选，含肇事第 18 行）：

```cpp
#ifndef OP_API_BFLOAT16_H
#define OP_API_BFLOAT16_H

#include <cmath>
#include <complex>
#include <limits>

#include "opdev/fp16_t.h"

namespace op {

// Single precision complex.
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

// see framework/bfloat16.h for description.
struct bfloat16 {
```

相邻卡片与区分条件：

- [fp16/bf16 位流转 float 零依赖转换卡](../validations/cxx_fp16_bf16_bitstream_to_float_conversion.md)——互补替代路线：那卡手写位级转换以规避 CANN 转换库链接依赖；本卡解决"要用官方 `op::bfloat16` 类型时 include 路径怎么配"，两条路线按是否需要官方类型选用。
- [aclnn 测试链接安装目录 libcust_opapi 卡](aclnn_test_links_installed_vendors_opapi.md)——同类宿主测试构建契约先例：测试链接/加载的是 OPP 安装目录产物而非 build 树；include 路径同理须对齐安装布局。

## 解决配方

修前 `tests/aclnn/CMakeLists.txt`（只含安装根，不含 aclnn 子目录）：

```cmake
include_directories(
    ${CANN_HOME}/x86_64-linux/include
    ${CANN_HOME}/x86_64-linux/include/aclnnop
    ${CANN_HOME}/include
    ${CANN_HOME}/opp/vendors/BenchHarnessEngramGateFusion/op_api/include
)
```

修后（新增两个 aclnn 子目录到 include 路径）：

```cmake
include_directories(
    ${CANN_HOME}/x86_64-linux/include
    ${CANN_HOME}/x86_64-linux/include/aclnn
    ${CANN_HOME}/x86_64-linux/include/aclnnop
    ${CANN_HOME}/include
    ${CANN_HOME}/include/aclnn
    ${CANN_HOME}/opp/vendors/BenchHarnessEngramGateFusion/op_api/include
)
```

会话内 `find`/`grep` 输出确认 `x86_64-linux/include/aclnn/opdev/fp16_t.h` 与 `include/aclnn/opdev/bfloat16.h` 两处副本均存在，故两个根的 `aclnn` 子目录都加入，覆盖宿主编译时任一生效布局。改完后 `rm -rf build && mkdir build && cd build && cmake .. && make` 重新配置构建即通过。

## 设计决策

- 选择"加 include 路径"而不是改用 `#include "opdev/fp16_t.h"`（以 aclnn 子目录为根包含官方头）：不改 CANN 官方头的包含约定，也不在同一工程里引入两套 include 写法；vendor 模板 `CMakeLists.txt` 的原始结构保持最小增量。
- 同时加 `x86_64-linux/include/aclnn` 与 `include/aclnn`：CANN 安装在两个根下都有 `aclnn/opdev` 副本，双保险覆盖两种安装布局，成本为零。
- 当前方案属 recommended_fastpath：一次 include 路径修正即恢复编译，无运行期代价。

## 预防

- 宿主测试凡包含 `aclnn/opdev/*` 系列头（`bfloat16.h`/`fp16_t.h`/`op_executor.h` 等），CMake 里除 CANN 安装根外，同步加 `<root>/x86_64-linux/include/aclnn` 与 `<root>/include/aclnn`。
- 遇到 "fatal error: xxx.h: No such file or directory" 且报错位于 CANN 官方头内部时，先 `head` 该头看内嵌 include 是否无前缀，再决定补哪个 include 根，不要怀疑自己的源文件。

## 验证信号

以下信号均为本开发会话（编写该算子 kernel ST）内的实测输出：

- 修前构建：`bfloat16.h:18:10: fatal error: opdev/fp16_t.h: No such file or directory`，`make Error 1`（来源平台：950；证据等级：tool_observed）。
- 头文件原文探针：`head -80` 输出直接展示 `bfloat16.h` 第 18 行 `#include "opdev/fp16_t.h"`（无 `aclnn/` 前缀）；`find` 输出确认 `x86_64-linux/include/aclnn/opdev/fp16_t.h` 存在（来源平台：950；证据等级：tool_observed）。
- 修后同一构建命令：`[ 50%] Building CXX object ... test_aclnn.cpp.o`、`[100%] Built target test_aclnn`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在宿主侧（x86_64-linux）测试工程编译期实测；kernel 侧包含链未涉及。
- CANN 9.2.0 头文件布局下验证；其他版本若官方头已改为带前缀 include，本卡失效（未验证，推断相近版本可迁移）。
- 未验证只加 `x86_64-linux/include/aclnn` 一个路径是否已足够（实际修复同时加了两个 aclnn 子目录，最小充分集未单独隔离）。

**未决项**：

- 最小充分 include 路径集未做单变量隔离（见失效边界）；下个会话可只保留一个 aclnn 子目录验证。
## 坏实践

- 只把 `${CANN_HOME}/x86_64-linux/include` 加进 include 路径就直接 `#include "aclnn/opdev/bfloat16.h"`：该头内嵌的无前缀 `opdev/fp16_t.h` 必然解析失败。
- 报错后去改测试源文件的 include 顺序或删除 `bfloat16.h` 包含：错误根源在 CMake include 路径集合，不在源文件。

## 误诊

- 无（本会话一次定位：先读 `bfloat16.h` 原文确认第 18 行无前缀内嵌 include，再对照 `find` 结果修 CMake，未走弯路）。
