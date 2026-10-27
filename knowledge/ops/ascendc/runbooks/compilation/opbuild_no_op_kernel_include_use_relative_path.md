---
type: "Runbook"
title: opbuild 编译 op_host 不注入 op_kernel 头路径
description: "自定义算子包 opbuild（npu_op_code_gen）编译 op_host 时只注入 CANN include、不含 op_kernel，host 侧头文件平铺 #include \"xxx_tiling.h\" 报 No such file or directory；改成 ../op_kernel/ 相对引用即可通过。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# opbuild 编译 op_host 不注入 op_kernel 头路径

## 现象

在为自定义算子 softplus_v2_grad 的算子包新增 host 侧 tiling 实现文件（`op_host/softplus_v2_grad_tiling_host.cpp` + `softplus_v2_grad_tiling_host.h`，TilingData 结构体定义在 `op_kernel/softplus_v2_grad_tiling.h`）并把 `add_subdirectory(tests/ut)` 编入主构建后，执行 `cd operators && ./build.sh softplus_v2_grad_package`，cmake 配置阶段直接失败，报错原文：

```text
CMake Error at /home/<user>/Ascend/cann-9.1.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:732 (message):
  opbuild run failed! In file included from
  /workspace/.../op_host/softplus_v2_grad_tiling_host.cpp:11:

  /workspace/.../op_host/softplus_v2_grad_tiling_host.h:17:10:
  fatal error: softplus_v2_grad_tiling.h: No such file or directory

     17 | #include "softplus_v2_grad_tiling.h"
        |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~

  compilation terminated.
  ...
  RuntimeError: compile ascend all ops failed
Call Stack (most recent call first):
  CMakeLists.txt:18 (npu_op_code_gen)
EXIT=1
```

当时 `softplus_v2_grad_tiling.h`（TilingData POD 结构体）确实存在于 `op_kernel/` 目录；同一构建里老的 `op_host/softplus_v2_grad_host.cpp` 却能编译通过。

## 适用场景

- 工程形态：AscendC 自定义算子包（op_host + op_kernel 分目录 + `npu_op_package_add` 构建），host 侧 tiling 实现需要引用 op_kernel 里的 TilingData 头。
- API 路径：`operators/build.sh <pkg>` → CMake `npu_op_code_gen`（opbuild，`func.cmake`）→ `compile_ascendc_all_ops_so.py` 编译 op_host 全部源文件。
- 验证过的平台与 CANN 版本：Ascend950（ascend950 / DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（opbuild 的 include 注入行为与具体算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 侧源文件（op_kernel/*.cpp 由 ascendc 编译器另走编译链，include 上下文不同）；ut 目录下的测试（tests/ut 的 CMake 可自行加 `-I op_kernel`，不受 opbuild 影响）；其他 CANN 版本的 opbuild 行为未验证。

## 根因

opbuild 编译 op_host 源文件时只注入 CANN 安装目录 include（编译命令为 `/usr/bin/c++ -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -I/home/<user>/Ascend/cann-9.1.0/include -c .../softplus_v2_grad_tiling_host.cpp`），**不注入算子包的 op_kernel 路径**，也不加包内相对 `-I`；`#include "..."` 对引号包含先按"包含者所在目录"解析再查 `-I`，所以 host.cpp 里写 `#include "../op_kernel/softplus_v2_grad_tiling.h"`（相对 op_host/ 回到 op_kernel/）能解析，而 tiling_host.h 里平铺 `#include "softplus_v2_grad_tiling.h"` 在 op_host/ 下找不到、`-I` 里也没有 op_kernel，于是 fatal error。

## 解决配方

修复前（`op_host/softplus_v2_grad_tiling_host.h`，平铺引用 op_kernel 内的 TilingData 头）：

```cpp
#include <cstdint>
#include <climits>
#include "graph/types.h"
#include "softplus_v2_grad_tiling.h"
```

修复后（改为相对路径，与 `op_host/softplus_v2_grad_host.cpp` 的既有写法一致）：

```cpp
#include <cstdint>
#include <climits>
#include "graph/types.h"
// 相对路径与 op_host/softplus_v2_grad_host.cpp 一致：opbuild (npu_op_code_gen)
// 仅注入 CANN include，不注入 op_kernel，故用 ../op_kernel/ 相对引用定位 TilingData。
#include "../op_kernel/softplus_v2_grad_tiling.h"
```

要点：引号 include 的相对路径按"当前头文件所在目录"解析，所以 `../op_kernel/` 从 op_host/ 出发即可命中，opbuild 上下文与 ut 编译上下文（自行加 `-I`）都能解析，不需要改 CMake。

## 设计决策

- 备选一：给 opbuild 补 `-I op_kernel`（改 func.cmake 或包级 CMake 注入）——侵入构建系统、跨包不可移植，放弃。
- 备选二：把 TilingData 头挪到 op_host/——但 TilingData 要被 kernel 侧 `GET_TILING_DATA` 消费，放 op_kernel 是模板工程惯例，挪动破坏既有布局，放弃。
- 采用：`../op_kernel/` 相对引用，零构建改动、与模板工程 host.cpp 既有写法一致；当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0，GCC 13.3.0，Ascend950。
- 相关文件：`op_host/softplus_v2_grad_tiling_host.h`（修复点）、`op_host/softplus_v2_grad_host.cpp`（既有正确写法参照）、`op_kernel/softplus_v2_grad_tiling.h`（TilingData 定义）、`operators/CMakeLists.txt` 与包级 `CMakeLists.txt`（`npu_op_code_gen` 调用点）。

## 预防

- host 侧新增文件引用 op_kernel 头时，一律照抄 host.cpp 的 `../op_kernel/xxx.h` 相对引用写法，不要平铺文件名。
- 遇到 `fatal error: xxx_tiling.h: No such file or directory` 且文件明明存在时，先看 opbuild 实际编译命令的 `-I` 列表（build 日志里有完整命令行），确认是 include 上下文问题而不是文件缺失。

## 验证信号

以下信号来自 softplus_v2_grad 公共 TilingUT 收尾阶段的构建修复会话：修改 include 后同一构建命令重跑验证。

- 修复前：`./build.sh softplus_v2_grad_package` 在 cmake 阶段报 `opbuild run failed! ... fatal error: softplus_v2_grad_tiling.h: No such file or directory`，EXIT=1；opbuild 编译命令只带 `-I/home/<user>/Ascend/cann-9.1.0/include`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一构建命令完整通过，`softplus_v2_grad_tiling_host.cpp` 编入 `cust_optiling`，UT 目标 `softplus_v2_grad_tiling_ut` 链接成功，构建退出 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论基于 CANN 9.1.0 的 opbuild include 注入行为；若后续版本 opbuild 改为自动注入包内目录，平铺写法也能过（未验证）。
- 相对引用依赖 op_host 与 op_kernel 的目录布局（同一算子包内平级）；目录被重排时路径需同步调整。

**未决项**：

- 无。
## 坏实践

- 在 op_host 头文件里平铺 `#include "xxx_tiling.h"` 指望构建系统帮你找到 op_kernel。
- 见到 "No such file or directory" 就去复制一份头文件到 op_host/（造成两份 TilingData 定义漂移）。

## 误诊

- 无（本次一次性定位到 include 上下文差异：对照同构建里 host.cpp 能过、tiling_host.cpp 不能过，差异只在引用写法）。
