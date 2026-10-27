---
type: "Runbook"
title: attrs 分支算子 TILING_KEY_IS 单 binary 分发
description: 无编译期特化需求的 attrs 分支型算子可用 extern "C" 入口 + REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT + TILING_KEY_IS if-else 编成单 kernel binary 多分支，tiling 端 SetTilingKey 置键；相对 ASCENDC_TPL（每组合一条 kernelList、共用 binary）是结构更简的替代路线，该宏在 ascend950 + CANN 9.1.0 实测可用（未被废弃）。
tags: [task.implementation, topic.implementation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# attrs 分支算子 TILING_KEY_IS 单 binary 分发

## 现象

在 Ascend950（CANN 9.1.0）上开发 attrs 分支型算子（同一 kernel 按 groups/dilation 组合走不同参数路径）时面临两种 TilingKey 机制选择：默认模板（ASCENDC_TPL）会为每个 TilingKey 组合在 kernel JSON 注册独立条目；改用 `TILING_KEY_IS` 运行时分发则编译为单个 `extern "C"` 入口、kernel 内 if-else 选分支。本轨迹完整落地了后一条路线：4 个 TilingKey 分支（fp16 × groups{1,>1} × dilation{1,>1}）、单 kernel binary、tiling 端 `SetTilingKey` 置键，UT 逐用例断言 TilingKey 值（0/2/4/6）全部通过，ST 5 用例连续运行全过（配 op_api 层 AbandonCache）。

## 适用场景

- 工程形态：Ascend C 自定义算子（算子仓工程，op_kernel + op_host + op_api），算子行为按 attrs（groups/dilation/stride 档位等）分派、但各分支**无编译期特化需求**（同一份模板代码即可）。
- API 路径：`TILING_KEY_IS(k)`（`utils/kernel_utils_macros.h:445`，展开为 `TILING_KEY_VAR == (k)`）；`REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT`（构建期生成）；`gert::TilingContext::SetTilingKey`。
- 验证过的平台与 CANN 版本：Ascend950（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（机制层面）；最窄可迁移档为 operator_family（attrs 分支型算子普遍适用）。
- 路线优先级：safety_fallback——ASCENDC_TPL（模板 + `ASCENDC_TPL_ARGS_DECL/SEL`）是模板生态默认路线，本路线是"无需编译期特化时更简单的替代"，两者都可用；本卡路线不解决 executor 缓存类问题（那需要 op_api 层 AbandonCache，见同域卡）。fallback 声明：选本路线的代价是失去模板特化能力；相对 ASCENDC_TPL 的结构差异是单入口单 kernelList 条目集，未测加载/选择开销差。
- 明确不适用 / 未验证场景：分支需要不同模板参数/数据类型实例化（如 half 与 bf16 不同模板实参）时必须走 ASCENDC_TPL；TILING_KEY_IS 宏参数必须为数值宏（constexpr 会致分支不注册，见同域卡）；其他 CANN 版本未验证。

## 设计决策

- 为什么选 TILING_KEY_IS：本算子 4 个分支的计算代码完全相同（dilation/groups 是 tiling 字段而非模板参数），ASCENDC_TPL 为此生成 4 条 kernelList 条目（实测 4 个 simplifiedKey 共用 1 个 binary 文件）属于冗余结构；运行时分发把结构收敛为单入口。
- 备选与放弃原因：保留 ASCENDC_TPL（默认、与部分内置算子一致）被放弃——非必要复杂度；某自定义算子模板注明"禁止使用废弃的 TILING_KEY_IS 宏"，但同版本内置算子（group_norm_grad 等 ops_nn）仍在使用 TILING_KEY_IS，且本轨迹实测该宏在 ascend950 + CANN 9.1.0 可用并全量通过——两处指引互相矛盾时以本机实测为准。
- 为"区分 attrs"给 ASCENDC_TPL 追加模板参数（如 DIL_MODE）是坏路线：若各分支代码相同，这只是膨胀 kernelList，且对本域的 executor 缓存问题无效（本轨迹实测无效）。

## 解决配方

三个文件的 verbatim 实现（取自该算子重构后的最终形态）。

（1）`op_kernel/arch35/*_tiling_key.h` —— 键值定义（必须用 `#define`，不能用 constexpr）：

```c
#ifndef __BENCH_NO_HARNESS_CONV_3D_BACKPROP_FILTER_TILING_KEY_H__
#define __BENCH_NO_HARNESS_CONV_3D_BACKPROP_FILTER_TILING_KEY_H__

#include <stdint.h>

// TilingKey 编码（运行时分发，不使用 ASCENDC_TPL 模板机制）：
//   bit0 = dtype   (0=fp16, 1=bf16 预留)
//   bit1 = groups  (0=groups=1, 1=groups>1)
//   bit2 = dilation(0=dilation=1, 1=dilation>1)
//
// 采用 TILING_KEY_IS + if-else 运行时分发（参考内置 group_norm_grad / conv3d_backprop_filter_v2），
// 编译为单个 kernel binary，避免 ASCENDC_TPL 多 kernelList 的 binary 缓存复用问题。
//
// 注意：必须使用 #define 宏而非 constexpr，因为 TILING_KEY_IS 在预编译阶段
// 要求参数为数值常量或宏（constexpr 变量无法被预编译器识别为数值）。
#define TILING_KEY_FP16_G1_DIL1  0  // fp16 + groups=1   + dilation=1
#define TILING_KEY_FP16_GN_DIL1  2  // fp16 + groups>1   + dilation=1
#define TILING_KEY_FP16_G1_DILN  4  // fp16 + groups=1   + dilation>1
#define TILING_KEY_FP16_GN_DILN  6  // fp16 + groups>1   + dilation>1

#endif
```

（2）`op_kernel/*_arch35.cpp` —— kernel 入口（extern "C" 单入口 + if-else 分发）：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#include "arch35/bench_no_harness_conv_3d_backprop_filter.h"

// Kernel 入口：使用 TILING_KEY_IS + if-else 运行时分发（非 ASCENDC_TPL 模板机制）。
// 编译为单个 kernel binary，由框架根据 context->SetTilingKey() 设置的 TilingKey
// 在运行时将 g_tilingKey 传入 kernel，kernel 内部通过 TILING_KEY_IS 选择执行分支。
extern "C" __global__ __aicore__ void bench_no_harness_conv3d_backprop_filter(
    GM_ADDR x, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv3dBackpropFilterTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv3dBackpropFilterTilingData, tilingData, tiling);
    (void)workspace;
    // 4 个分支的 kernel 计算逻辑相同（DIL_MODE 不影响计算，dilation 由 tilingData 字段决定），
    // 但通过独立的 TilingKey 区分，确保框架不会跨用例复用错误的 kernel binary。
    if (TILING_KEY_IS(TILING_KEY_FP16_G1_DIL1)) {
        NsConv3dBpFilter::Conv3dBpFilterKernel<half, 0> op;
        op.Init(x, grad, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_G1_DILN)) {
        NsConv3dBpFilter::Conv3dBpFilterKernel<half, 0> op;
        op.Init(x, grad, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_GN_DIL1)) {
        NsConv3dBpFilter::Conv3dBpFilterKernel<half, 1> op;
        op.Init(x, grad, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_GN_DILN)) {
        NsConv3dBpFilter::Conv3dBpFilterKernel<half, 1> op;
        op.Init(x, grad, y, &tilingData, &tPipe);
        op.Process();
    }
}
```

（3）`op_host/arch35/*_tiling.cpp` —— TilingFunc 置键段（SetBlockDim 之后；由两次编辑合并，dtype 枚举为 `ge::DT_BF16`）：

```cpp
    context->SetBlockDim(static_cast<uint32_t>(coreNum));

    // TilingKey 编码（与 _tiling_key.h 中常量定义一致）：
    //   bit0 = dtype   (0=fp16, 1=bf16 预留)
    //   bit1 = groups  (0=groups=1, 1=groups>1)
    //   bit2 = dilation(0=dilation=1, 1=dilation>1)
    uint32_t tilingKey = 0;
    if (dtypeX == ge::DT_BF16) {
        tilingKey |= 1U;  // bit0 = 1 (bf16，暂未支持但预留编码)
    }
    if (groups > 1) {
        tilingKey |= 2U;  // bit1 = 1 (groups>1)
    }
    if (dilationD > 1 || dilationH > 1 || dilationW > 1) {
        tilingKey |= 4U;  // bit2 = 1 (dilation>1)
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
```

## 验证信号

来源说明：该算子从 ASCENDC_TPL 到 TILING_KEY_IS 的完整重构与回归。

- 重构前（ASCENDC_TPL 版）：kernel 配置 JSON 的 binList 含 4 个 simplifiedKey 条目但仅 1 个 binary 文件（4 个 TilingKey 组合共用同一 binary）（来源平台：950；证据等级：tool_observed）
- 重构后：`#define` 键值版干净构建（无预编译诊断），kernelList 注册 4 个条目（TilingKey 后缀 0/2/4/6）（来源平台：950；证据等级：tool_observed）
- UT 12/12 通过，逐用例 TilingKey 断言：`tiling_fp16_g1_kernel1x1x1 | TilingKey=0`、`tiling_fp16_gn_groups2 | TilingKey=2`、`tiling_fp16_g1_dilation2 | TilingKey=4`、`tiling_fp16_gn_groups2_dilation2 | TilingKey=6`（来源平台：950；证据等级：tool_observed）
- ST Real 5 用例连续运行 5/5 全部 PASS（MERE=0.00e+00；配 op_api 层 AbandonCache(true)，attrs 正确性由该修复保证）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本路线不解决"同 shape 不同 attrs 的 executor tiling 缓存"问题（该问题在本轨迹中曾误归于 ASCENDC_TPL binary 结构，重构后依旧复现，最终由 AbandonCache 修复）。
- 分支需要不同模板实参（dtype 不同实现、Tile 尺寸特化）时不适用，须回 ASCENDC_TPL。
- `TILING_KEY_IS` 参数必须为数值宏/常量（constexpr 会致分支不注册，见同域卡）。
- "某模板禁止使用废弃的 TILING_KEY_IS 宏"的口径与内置算子用法矛盾，本卡仅证明 CANN 9.1.0 + ascend950 实测可用；后续版本若真废弃该宏，本路线失效（未验证，推断：同版本其他平台可用）。
- kernelList 多条目（0/2/4/6）相对单条目在算子加载/选择路径上的开销差未测量。

**未决项**：

- 未尝试 ASCENDC_TPL 与 TILING_KEY_IS 混用（模板特化 + 运行时分支）的可行性。
## 坏实践

- 为"区分 attrs"给 ASCENDC_TPL 追加模板参数（如 DIL_MODE）——若各分支代码相同，这只是膨胀 kernelList，且对本域的缓存问题无效。

## 误诊

- 重构动机中的"避免 binary 缓存复用导致的 tiling 烘焙错误"是错误归因：binary 侧无此问题，缓存问题在 executor 层（详见同域 executor 缓存卡）。本卡只保留该路线作为结构选择的价值，不继承该动机。
