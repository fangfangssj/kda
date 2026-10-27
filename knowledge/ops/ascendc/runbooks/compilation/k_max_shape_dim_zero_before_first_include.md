---
type: "Runbook"
title: "K_MAX_SHAPE_DIM=0须先于首个include定义"
description: "不使用 ShapeInfo 功能的 kernel 可在 TU 首个 kernel_operator.h include 前定义 K_MAX_SHAPE_DIM=0，收缩每个 Tensor 的 ShapeInfo 栈预留与标量头开销；SDK 默认值为 8（sys_macros.h 的 #ifndef 守护），定义晚于首个 include 静默无效。"
tags: ["task.optimization", "topic.implementation.api_layer.basic_api", "topic.compilation"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# K_MAX_SHAPE_DIM=0须先于首个include定义

## 现象

在 Ascend950PR（CANN 9.2.0）上优化融合算子 FusedRmsNormSigmoidMul 的设备 kernel：前期画像显示 scalar 通路共 bound（利用率 0.33），手段清单里有一项"收缩每个 Tensor 的 ShapeInfo 栈预留与标量头开销"。查 SDK 头文件确认机制：`asc/include/basic_api/kernel_tensor.h` 中多处 `ASCENDC_ASSERT((inputShapeDim <= K_MAX_SHAPE_DIM && inputOriginalShapeDim <= K_MAX_SHAPE_DIM), {...})`——每个 Tensor 按 `K_MAX_SHAPE_DIM` 槽位预留 shape 信息；`asc/impl/utils/sys_macros.h` 是该宏的默认定义点（`#ifndef` 守护，默认 8）：

```c
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif
```

该算子 kernel 全仓不使用 `SetShapeInfo/GetShapeInfo`（ShapeInfo 功能零使用），预留 8 槽纯属浪费。优化落地时把宏定义为 0，但有两个工程落点要选对：直调工程里 kernel 头自身 `#include "kernel_operator.h"`，注册工程里则是 `apt.cpp` 先包含它——定义必须先于**本 TU 首个** kernel_operator.h include 生效，放错位置等于没改。

## 适用场景

- 工程形态：AscendC device kernel（直调工程与注册算子工程的设备 TU 均适用）。
- API 路径：SDK 头文件宏 `K_MAX_SHAPE_DIM`（`asc/impl/utils/sys_macros.h` 默认定义 + `#ifndef` 守护；`kernel_tensor.h` 按 `__NPU_ARCH__` 消费）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（SDK 宏机制，与算子无关；前提是该 kernel 不使用 ShapeInfo 功能）。
- 路线优先级：recommended_fastpath（零语义风险的条件化手段；前提核验成本一行 grep）。
- 明确不适用 / 未验证场景：**使用了 `SetShapeInfo/GetShapeInfo` 或依赖 Tensor shape 查询的 kernel 禁用**（预留收缩为 0 会使相关接口不可用）；依赖框架按默认 8 槽透传 shape 的场景未验证；其他 CANN 版本的宏位置/默认值可能不同（升级需重查 sys_macros.h）。

## 根因

`K_MAX_SHAPE_DIM` 决定每个 Tensor 的 ShapeInfo 槽数（默认 8），槽位以栈预留与标量头开销形式进入每个 kernel；对不消费 ShapeInfo 的 kernel 这是纯开销。宏带 `#ifndef` 守护且默认定义位于 `sys_macros.h`（被 kernel_operator.h 包含链引入），因此用户侧定义必须出现在**本 TU 首个 kernel_operator.h include 之前**才能覆盖默认值——之后定义无效（宏已定，守护分支跳过）。

## 解决配方

直调工程：在 kernel 头之前加一个 local_deps 头（在 entry 包含链中最早被包含），并保持幂等守护：

```cpp
#pragma once
// K_MAX_SHAPE_DIM 必须先于首个 kernel_operator.h include 生效：
// 本算子不使用 ShapeInfo 功能（csrc/op_kernel 全仓无 SetShapeInfo/GetShapeInfo），
// 设 0 收缩每个 Tensor 的 ShapeInfo 栈预留与标量头开销。
// SDK 默认值为 8（impl/utils/sys_macros.h 的 #ifndef 守护），此处必须先定义。
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
```

kernel 头自身同样加守护（防直接包含时漏定义，幂等）：

```cpp
#pragma once
// K_MAX_SHAPE_DIM 必须先于本文件的 kernel_operator.h include 生效；
// local_deps.h 在 entry.h 包含链中更早定义同一宏，此处 #ifndef 守护保证幂等。
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "fused_rms_norm_sigmoid_mul_tiling_struct.h"
```

注册工程（apt.cpp 是设备 TU 首个包含 kernel_operator.h 的位置）：

```cpp
// K_MAX_SHAPE_DIM 必须先于本 TU 首个 kernel_operator.h include 生效：
// 本算子不使用 ShapeInfo 功能（op_kernel 全仓无 SetShapeInfo/GetShapeInfo），设 0
// 收缩每个 Tensor 的 ShapeInfo 栈预留与标量头开销。SDK 默认值为 8
// （asc/impl/utils/sys_macros.h 的 #ifndef 守护），此处必须先定义。
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "arch35/fused_rms_norm_sigmoid_mul_kernel.h"
```

前提核验（一行 grep，确认零 ShapeInfo 使用）：

```bash
grep -rn "SetShapeInfo\|GetShapeInfo" <op_kernel 目录>   # 应无命中
grep -rn "define K_MAX_SHAPE_DIM" $ASCEND_HOME_PATH/x86_64-linux/asc/  # 定位默认定义点
```

## 设计决策

- 为什么定义 0 而不是 1：完全不使用 ShapeInfo，最小预留即 0；`kernel_tensor.h` 的断言只约束实际传入的 shapeDim 上限，0 槽对不使用者无影响。
- 为什么两个落点都加 `#ifndef` 守护：直调工程 kernel 头可能被直接包含（不经 local_deps），守护保证任意包含顺序下幂等且语义一致；注册工程在 apt.cpp（TU 首个 include 点）落定义。
- 为什么不用编译命令行 `-DK_MAX_SHAPE_DIM=0`：可行（宏带 ifndef 守护），但注册工程构建链路（CMake/op 编译器）传参落点多、易漏 TU；头文件内定义随源码走、对包含链自证（本会话选择源码内定义；命令行方案未对比）。
- 当前方案路线档位：recommended_fastpath（与其他手段叠加交付）。

## 系统信息

- 硬件/软件：Ascend950PR（arch3510），CANN 9.2.0；宏默认定义：`$ASCEND_HOME_PATH/x86_64-linux/asc/impl/utils/sys_macros.h`；消费点：`asc/include/basic_api/kernel_tensor.h`（多处 ASCENDC_ASSERT）。
- 相关文件：直调工程 `csrc/op_kernel/kernel/fused_rms_norm_sigmoid_mul_local_deps.h`（新增加头条）、`kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h`（幂等守护）；注册工程 `op_kernel/fused_rms_norm_sigmoid_mul_apt.cpp`（TU 首个 include 前定义）。

## 验证信号

以下信号来自该融合算子性能优化候选的实施与集成会话：该手段作为终选实现 14 项手段之一落地（与其余手段捆绑验收，无孤立对照采集）。

- SDK 事实锚点：`sys_macros.h` 中 `#ifndef K_MAX_SHAPE_DIM / #define K_MAX_SHAPE_DIM 8 / #endif`；`kernel_tensor.h:61/76/89` 等处按该宏断言 shapeDim 上限（来源平台：950；证据等级：tool_observed）。
- 前提核验：op_kernel 全仓 `SetShapeInfo/GetShapeInfo` 零命中（来源平台：950；证据等级：tool_observed）。
- 正确性（捆绑验证）：含该手段的候选 18 条 L0 strict 18/18 PASS、L0+L1 全集 239/239 PASS、集成回注册工程后 18/18 PASS（aclnn 接口）——定义 0 未破坏任何功能（来源平台：950；证据等级：tool_observed）。
- 性能：该手段的孤立贡献未单独采集（bundled in 14-feature 终选实现，整体相对原始基线 9.16x（参照路线：原始注册基线；来源平台：950；证据等级：tool_observed），见 [vf_lane_accumulate_row_loop_consolidation](../performance/vf_lane_accumulate_row_loop_consolidation.md)；手段级拆分见"遗留问题"）。

## 失效边界

- **使用 ShapeInfo 的 kernel 禁用**：`SetShapeInfo/GetShapeInfo` 或依赖 Tensor shape 查询的实现会把预留收缩到 0 后失效——落地前必须 grep 核验全仓零使用。
- 定义位置晚于本 TU 首个 kernel_operator.h include 时**静默无效**（宏已被默认值占据，无编译报错）——这是本卡最大的坑。
- 性能收益量级未孤立测量：本会话只验证"可安全启用"，未测"启用后单独快多少"；scalar 开销占比小的 kernel 可能无感（未验证精度/性能影响之外的边界）。
- CANN 版本升级后宏的默认值/定义点可能变化，升级需重查 sys_macros.h（未验证其他版本）。

**未决项**：

- 孤立性能贡献未测量（优先项：同算子做一次 ±该宏的成对 A/B 采集即可量化）。
- `kernel_tensor.h` 断言在 shapeDim>K_MAX_SHAPE_DIM=0 时是否会在 debug 构建下触发（本会话 Release 形态未验证）。
## 坏实践

- 把 `#define K_MAX_SHAPE_DIM 0` 放在 kernel_operator.h **之后**（如某个 .cpp 中段）——静默无效还以为已优化。
- 只在直调工程加了定义、集成回注册工程时遗忘——注册工程设备 TU 的首个 include 点不同（apt.cpp），必须重新落点（本会话集成阶段单独处理了该落点，见 [direct_optimized_integrate_back_registered](direct_optimized_integrate_back_registered.md)）。
- 不核验 ShapeInfo 零使用就设 0——使用了 ShapeInfo 的 kernel 会直接功能损坏。
