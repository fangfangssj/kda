---
type: "Runbook"
title: "Matmul 头路径 lib/matmul 不存在，应为 adv_api"
description: "CANN 9.2.0 安装树与仓内 asc-devkit 头树均无 lib/ 目录；matmul_tiling.h/matmul_tilingdata.h/matmul_intf.h 位于 adv_api/matmul/（host tiling 另有 asc/include/tiling/matrix/），按旧示例路径 include 直接 fatal error。"
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-17T04:05:46Z'
updated_at: '2026-09-17T04:05:46Z'
platforms: ["950"]
---

# Matmul 头路径 lib/matmul 不存在，应为 adv_api

## 现象

开发 BenchNoHarnessGru（GRU，Cube+Vector MIX 架构）初版骨架时，op_host/arch35/bench_no_harness_gru_tiling.cpp、op_kernel/arch35/bench_no_harness_gru_tiling_data.h、op_kernel/arch35/bench_no_harness_gru.h 三个文件按 AscendC 常见示例的旧路径写：

```cpp
#include "lib/matmul/matmul_tiling.h"        // host tiling
#include "lib/matmul/matmul_tilingdata.h"    // tiling data (host/kernel 共享头)
#include "lib/matmul/matmul_intf.h"          // kernel 侧 Matmul 接口
```

`bash build.sh --soc=ascend950 -j8`（CANN 9.2.0 安装目录经 $ASCEND_HOME_PATH 指定，工程同时依赖仓内 asc-devkit 头文件树）编译报：

```text
Cleaning build directory...
fatal error: lib/matmul/matmul_tiling.h: No such file or directory
```

## 适用场景

- 工程形态：自定义算子仓开发（op_host TilingFunc 用 MatmulApiTiling、op_kernel 用 Matmul 高阶 API/REGIST_MATMUL_OBJ）。
- API 路径：Matmul 高阶 API 头文件族 `matmul_tiling.h`（host 侧 tiling 基类）、`matmul_tilingdata.h`（optiling::TCubeTiling 定义）、`matmul_intf.h`（kernel 侧接口）。
- 验证过的平台与 CANN 版本：Ascend950（build.sh --soc=ascend950 编译流水线），CANN 9.2.0 + 仓内 asc-devkit 头（与 frontmatter 一致）。
- 适用粒度：platform_wide（CANN 版本/头文件布局绑定，与算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：旧版 CANN（lib/matmul/ 真实存在的版本）不适用本卡替换方向；`asc/include/tiling/matrix/matmul_tiling.h` 与 `adv_api/matmul/matmul_tiling.h` 的取舍差异未逐项比对（本卡只验证 adv_api 路径可用）；自定义算子 kernel 编译的 include 集合若不含 adv_api 所在根目录，改路径后仍需构建层注入 include（另见 [Matmul 高阶 API 头不可见卡](matmul_highlevel_api_include_paths_ini.md)与 [kernel 侧 Matmul 头两棵树卡](kernel_matmul_intf_include_tree.md)）。

## 根因

CANN 9.2.0 安装目录与仓内 asc-devkit 的 include 树中**不存在 `lib/` 目录**。工具查证（本会话实测）：

```text
# CANN toolkit x86_64-linux/asc/include 与 asc-devkit/include 顶层目录均为：
adv_api  aicpu_api  basic_api  c_api  interface  kernel_operator.h  simt_api  tensor_api  tiling  utils
# 实际位置：
$ASCEND_HOME_PATH/x86_64-linux/asc/include/tiling/matrix/matmul_tiling.h      # host 侧 tiling
$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/matmul/matmul_tilingdata.h
<repo>/asc-devkit/include/adv_api/matmul/matmul_tiling.h
<repo>/asc-devkit/include/adv_api/matmul/matmul_tilingdata.h
```

旧示例/旧文档中的 `lib/matmul/...` 是历史头文件布局，本版本一律在 `adv_api/matmul/` 下。

## 解决配方

修复前：

```cpp
// op_kernel/arch35/bench_no_harness_gru_tiling_data.h
#include "lib/matmul/matmul_tilingdata.h"
// op_host/arch35/bench_no_harness_gru_tiling.cpp
#include "lib/matmul/matmul_tiling.h"
// op_kernel/arch35/bench_no_harness_gru.h
#include "lib/matmul/matmul_intf.h"
```

修复后（三个文件统一改 adv_api 路径）：

```cpp
// op_kernel/arch35/bench_no_harness_gru_tiling_data.h
#include "adv_api/matmul/matmul_tilingdata.h"
// op_host/arch35/bench_no_harness_gru_tiling.cpp
#include "adv_api/matmul/matmul_tiling.h"
// op_kernel/arch35/bench_no_harness_gru.h
#include "adv_api/matmul/matmul_intf.h"
```

## 设计决策

- 选 `adv_api/matmul/` 而非 `tiling/matrix/`：asc-devkit 与 CANN toolkit 两棵树都有 adv_api/matmul，工程同时消费两棵树时该路径最稳；`tiling/matrix/matmul_tiling.h` 仅在 CANN toolkit 侧确认存在。
- 注意：tiling_data.h 是 host/kernel 共享头，kernel 侧 include `adv_api/matmul/matmul_tilingdata.h` 会引入新的冲突（graph 头泄漏进 kernel TU），那是另一个独立问题，见卡内"失效边界"与 [TCubeTiling 双定义卡](tcube_tiling_host_kernel_dual_definition.md)；本卡只闭合"路径不存在"这一层。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写 include 前先查目录：`ls $ASCEND_HOME_PATH/x86_64-linux/asc/include/` 与 `ls <repo>/asc-devkit/include/`。
- 从旧示例复制工程骨架时，把 include 路径当作版本敏感项逐一核对。

## 验证信号

本卡验证信号来自 BenchNoHarnessGru（GRU，Cube+Vector MIX）初版骨架的编译循环：

- 修复前一轮构建报 `fatal error: lib/matmul/matmul_tiling.h: No such file or directory`；三个 include 改为 `adv_api/matmul/...` 后，下一轮构建该 fatal error 消失（剩余错误转为枚举名/形状 API 等独立问题）（来源平台：950 编译流水线；证据等级：tool_observed）。
- kernel 侧 `adv_api/matmul/matmul_intf.h` 在后续 kernel 编译阶段已被成功解析（错误进入其内部的枚举冲突阶段，证明头文件找到并展开了）（来源平台：950 编译流水线；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.2.0 + 本仓 asc-devkit 头布局；其他版本未验证，推断可迁移。
- `#include "adv_api/matmul/matmul_tilingdata.h"` 在 **kernel 编译单元**会引发 graph 头与 kernel 头的枚举冲突（`graph/types.h: expected identifier`），共享 TilingData 头不能直接包含它——这是独立根因，本卡不覆盖该修复，见 [TCubeTiling 双定义卡](tcube_tiling_host_kernel_dual_definition.md)。
- 本卡的 kernel 侧路径结论基于 asc-devkit 工程的 include 集合；不同构建框架的 kernel -I 集合可能只解析 ascendc 树的 `lib/matmul/matmul_intf.h` 或需 ini 注入，见 [kernel 侧 Matmul 头两棵树卡](kernel_matmul_intf_include_tree.md)。

**未决项**：

- `tiling/matrix/matmul_tiling.h` 与 `adv_api/matmul/matmul_tiling.h` 两份同名头的差异未比对。
## 坏实践

- 从旧代码/旧文档照抄 `lib/matmul/` 路径不查当前安装布局。

## 误诊

- 无（fatal error 信息直接指明文件不存在；用 find/ls 核对目录后立即定位）。
