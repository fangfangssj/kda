---
type: "Runbook"
title: arch35 算子扩展 fp16/bf16 支持全链路配方
description: 单 dtype Ascend C 算子扩 fp16/bf16 的五步配方：op_def 三列表配对、tiling_key 多 dtype 模板声明与选择、tiling.cpp 传 dtype、kernel fp32 暂存路径、二进制后缀校验；全 dtype NPU ST 验证通过。
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T01:58:00Z'
updated_at: '2026-09-17T01:58:00Z'
platforms: ["950"]
---

# arch35 算子扩展 fp16/bf16 支持全链路配方

## 现象

一个已实现 fp32 单 dtype 的 arch35 自定义算子（池化类，`op_host` + `op_kernel/arch35` 目录结构）要扩展 fp16/bf16 支持。完整改动横跨 4 层：算子原型注册（`op_host/*_def.cpp`）、模板变体声明（`op_kernel/arch35/*_tiling_key.h`）、Host tiling 传参（`op_host/arch35/*_tiling.cpp`）、Kernel 模板分支（`op_kernel/arch35/*.h` 的 `if constexpr` 路径）。扩展过程中踩到两个构建层坑：多 dtype 后 `Format`/`UnknownShapeFormat` 列表条目数不配对导致 opbuild 报 dtype size 0（判别与修复见 [op_def dtype 三列表长度配对卡](opbuild_dtype_lists_must_be_equal_length.md)）；bf16 的 context TilingKey 取枚举值 27 而非分支索引（编码规则与 UT 期望写法见 [TilingKey 编码为 dtype 枚举+mode×256 卡](../validations/tilingkey_dtype_enum_plus_mode_256.md)）。最终 3 个 Kernel 二进制变体全部生成，NPU 上 fp32/fp16/bf16 全量 ST 通过。

## 适用场景

- 工程形态：Ascend C 算子仓开发（ops-registry-invoke 类工程），已有单 dtype 实现需要扩展低精度 dtype。
- API 路径：`OpDef`（`.DataType/.Format/.UnknownShapeFormat`）；`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_DATATYPE_DECL`/`ASCENDC_TPL_SEL`/`ASCENDC_TPL_SEL_PARAM` 模板参数宏族；`ge::DataType` 枚举（fp32=0/fp16=1/bf16=27）；Kernel 侧 `Cast`/`ReduceSum`/`Duplicate`。
- 验证过的平台与 CANN 版本：Ascend 950PR（`--soc=ascend950`）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（数值计算类算子的 dtype 扩展通用链路）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：整型 dtype（INT8/INT32 等枚举值不同但链路同构，未验证）；需要 dtype 间不同 tiling 策略的场景（本例 tiling 与 dtype 无关）。

## 根因/解决配方

单 dtype 算子默认只在一处绑定 dtype（tiling_key.h 的 `ASCENDC_TPL_DATATYPE_DECL` 单值 + op_def 单 DataType），扩展 dtype 必须把"类型注册 → 模板变体 → Host 传参 → Kernel 分支"四层同时打开，缺任何一层都会在构建或运行期失败（opbuild 报错 / 只生成 1 个二进制 / 运行期选错变体）。五步配方：

**第 1 步：op_def 注册多 dtype（三列表配对）**——`op_host/*_def.cpp` 的 Input/Output 同时扩 `DataType` 与 `Format`/`UnknownShapeFormat`，条目数严格相等（见下方完整代码；长度不配对的报错形态与判别另见 [op_def dtype 三列表长度配对卡](opbuild_dtype_lists_must_be_equal_length.md)）。

**第 2 步：tiling_key.h 声明多 dtype 模板**——`ASCENDC_TPL_DATATYPE_DECL` 列出全部 dtype 并绑定 `ASCENDC_TPL_INPUT(0)`；`ASCENDC_TPL_SEL` 为每个 dtype 写一个选择分支，每个分支生成一个 Kernel 二进制变体。

**第 3 步：tiling.cpp 传实际 dtype**——Host 侧取输入 dtype 后 `ASCENDC_TPL_SEL_PARAM(context, dTypeX)`，框架把 `ge::DataType` 枚举值设置为 context TilingKey 并映射到对应模板变体（注意：`GetTilingKey()` 返回枚举原值，bf16=27 非索引）。

**第 4 步：Kernel 模板分支 + fp32 暂存**——kernel 以 `D_T_X` 模板参数实例化；低精度路径 `Cast` 到 float 暂存计算（累加用 `ReduceSum<float>`、输出 `Duplicate<float>` 后 `Cast` 回，`CAST_RINT` 舍入），避免 fp16/bf16 直接累加的溢出/精度损失；Cast 提升用 CAST_NONE、回退用 CAST_RINT 的方向语义见 [FP16 算子 Cast 提升-计算-回退链卡](../precision/fp16_cast_up_fp32_compute_cast_down_chain.md)。

**第 5 步：编译产物校验**——解析 `build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.json`，确认每个 dtype 一个变体且 kernel 名后缀 = 枚举值（`_0`/`_1`/`_27`），再上 NPU 跑分 dtype ST。

## 完整代码

第 1 步，`op_host/bench_no_harness_adaptive_avg_pool_3d_def.cpp`（Input/Output 同构，Output 略同）：

```cpp
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

第 2 步，`op_kernel/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling_key.h`（完整文件终版）：

```cpp
 * 模板参数：
 *   - D_T_X: 输入数据类型（float / half / bfloat16_t）
 *
 * 支持 3 种 dtype：fp32, fp16, bf16，通过 ASCENDC_TPL_DATATYPE_DECL 绑定 input[0] 类型
 * TilingKey（0/1/2 分支路径）通过 TilingData.tilingKey 字段传递（非模板参数）
 */

#ifndef __BENCH_NO_HARNESS_ADAPTIVE_AVG_POOL_3D_TILING_KEY_H__
#define __BENCH_NO_HARNESS_ADAPTIVE_AVG_POOL_3D_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 声明 D_T_X 支持 3 种 dtype，绑定 input[0] 的实际类型
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessAdaptiveAvgPool3d,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

// 为每种 dtype 生成独立的 Kernel 二进制变体
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)
    ),
);

#endif // __BENCH_NO_HARNESS_ADAPTIVE_AVG_POOL_3D_TILING_KEY_H__
```

第 3 步，`op_host/arch35/bench_no_harness_adaptive_avg_pool_3d_tiling.cpp` 的 TilingFunc 收尾（部分恢复：trace 内只读到此片段；`dtypeX` 来自上文 `GetInputDesc(0)->GetDataType()` 类调用）：

```cpp
    tiling->outH = outH;
    tiling->outW = outW;
    tiling->maxRowsPerBatch = MAX_ROWS_PER_BATCH;
    tiling->inputDType = static_cast<uint32_t>(dtypeX);

    context->SetBlockDim(usedCoreNum);

    // 8. 设置 TilingKey（模板参数选择）
    // 参数顺序与 tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义一致：
    //   D_T_X (DATATYPE)
    // TilingKey 通过 TilingData.tilingKey 字段传递（非模板参数）
    uint32_t dTypeX = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);

    return ge::GRAPH_SUCCESS;
}
```

配套 `op_kernel/arch35/*_tiling_data.h` 的 dtype 字段注释按枚举值校准：

```cpp
    uint32_t inputDType = 0;      // 输入数据类型 (ge::DT_FLOAT=0, ge::DT_FLOAT16=1, ge::DT_BF16=27)
```

第 4 步的低精度暂存路径为 kernel 内 `if constexpr` 按 `D_T_X` 分支的 Cast/ReduceSum/Duplicate 链，与既有 FP16/BF16 暂存计算卡同构，本卡不重复内联。

## 设计决策

- **为什么用模板变体而不是运行时分支**：每个 dtype 生成独立 Kernel 二进制（`_0`/`_1`/`_27`），编译期特化避免运行期类型分发开销；dtype 由 tiling 期 `ASCENDC_TPL_SEL_PARAM` 选定。
- **为什么低精度路径统一升 fp32 暂存**：池化窗口可达数百万元素，fp16/bf16 直接累加必然溢出/精度崩坏（需求文档明确要求 fp32 中间累加）；Cast 开销相对累加可忽略。
- **放弃的候选**：在 kernel 内用 `if/else` 运行时判 dtype 单一二进制——会引入类型擦除与分支开销，且 `LocalTensor<T>` 强类型不友好；`tiling_data.h` 自定义编码 dtype（0/1/2）——与 `ge::DataType` 枚举冲突易错，最终直接存枚举值并在注释里写明 27。
- **输出端舍入**：`Cast` 回低精度用 `CAST_RINT`（round-to-nearest-even），升精度方向用 `CAST_NONE`（无损）。

## 系统信息

- 芯片/架构：Ascend 950PR（DAV_3510，arch35），`bash build.sh --soc=ascend950 -j8` 编译。
- CANN：9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装目录）。
- 目录布局：`op_host/bench_no_harness_adaptive_avg_pool_3d_def.cpp`、`op_host/arch35/*_tiling.cpp`、`op_kernel/arch35/*_tiling_key.h`、`op_kernel/arch35/*_tiling_data.h`、`op_kernel/arch35/*.h`（Kernel 类）、二进制产物 `build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*.json`。
- 相关文件清单（与本配方直接相关）：上述 5 类文件 + `op_api`（已天然支持三 dtype，无需改）+ `op_graph/*_proto.h`（已支持，无需改）。

## 预防

- 动 `DataType` 必同步动 `Format`/`UnknownShapeFormat`（条目数相等）。
- 涉及 bf16 的断言/日志先查枚举表（27），不要按列表位置编号。
- 扩展后第一件事解析二进制 JSON 确认变体数与后缀，再上板。

## 验证信号

来源：一次为池化基准算子做 fp16/bf16 全 dtype 扩展并逐 dtype NPU 验证的开发会话（Ascend950PR + CANN 9.1.0）。

- 重建成功生成 3 个 Kernel 二进制变体，JSON 确认 `_0`=float32、`_1`=float16、`_27`=bfloat16（来源平台：950；证据等级：tool_observed）。
- 安装后 ST Real 模式（NPU）分 dtype 全量：fp32 37/37、58/58；fp16 51/51（最大 MARE 8.15e-04 < 阈值 9.77e-04）；bf16 52/52（MERE/MARE 全 0.00e+00）（来源平台：950；证据等级：tool_observed）。
- UT（Host 侧 tiling/infershape，SOC Ascend950）：59/59 通过，含 bf16 TilingKey=27 期望断言（来源平台：950；证据等级：tool_observed）。
- PyTorch 接入测试（同算子、golden float64 对齐后）全量 0 失败，交叉印证三 dtype 路径正确（来源平台：950；证据等级：tool_observed）。

## 失效边界

- tiling 策略与 dtype 强相关的算子（如 fp16 需要不同切分）需把 dtype 纳入 tiling 决策，本配方未覆盖。
- `ASCENDC_TPL_DATATYPE_DECL` 绑定的是 `ASCENDC_TPL_INPUT(0)`（按输入 0 的类型选变体）；输入输出 dtype 不同（如 mixed precision Cast 算子）需另配绑定方式，未验证。
- bf16 全零误差（NPU 输出与 golden 逐位一致）是本算子 fp32 暂存 + CAST_RINT 路径与 golden double 路径舍入恰好重合的结果，不应作为 bf16 精度比对的普适预期。
- 整型 dtype 与其他 SoC/CANN 版本未验证，推断可迁移。

**未决项**：

- `ASCENDC_TPL_SEL_PARAM` 值→变体映射的框架实现细节未源码级确认（由二进制后缀与 NPU 分派间接证实）。
## 坏实践

- 只改 op_def 的 `DataType` 不改 tiling_key 模板声明——opbuild 报错或只出 1 个二进制。
- tiling_data 里自造 dtype 编码（0/1/2）与 ge 枚举混用——bf16 上必然错位（本例初版注释即犯此错，后校准为枚举值注释）。
