---
type: "Runbook"
title: 审计计算逻辑是否全在 Kernel 而非 Host 侧
description: 验收自定义算子"计算是否全在 Kernel"的判别法：对 op_host/op_api 逐文件按职责白名单核对（注册/shape 推导/tiling 切分/参数校验/调度四类合法），kernel 侧列计算函数清单兜底；tiling 中的步长预计算（hwStride=H*W 等）属合法切分参数、不算数据计算。
tags: [task.testing, topic.architecture]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# 审计计算逻辑是否全在 Kernel 而非 Host 侧

## 现象

（任务场景）同一个已开发完成的自定义算子（grid_sampler_3d，ascend950/arch35，工程与编译产物齐备）的人工检查第二项："是否所有计算逻辑都在 Kernel 内完成，而非在 Host 上实现"，禁止修改代码、仅读取分析。这个检查有一个容易误判的点：op_host 的 tiling 代码里出现了 `hwStride = H * W`、`dhwStride = D * H * W`、`cdhwStride = C * D * H * W` 等一批预计算，还有 `totalPoints`/`blockFactor` 切分计算——如果按"Host 代码里有算术"来判，会误判成"Host 侧做了计算"；需要一套判据把"Host 侧合法的切分/推导/校验/调度"与"非法的 Host 侧数据计算"区分开。

## 适用场景

- 工程形态：自定义算子（op_kernel/op_host/op_api 三层工程）开发完成后的计算归属审计/验收检查；纯读取分析。
- API 路径：op_host 侧 `register/op_def_registry.h`（OpDef/OP_ADD）、`register/op_impl_registry.h`（IMPL_OP_INFERSHAPE）、`IMPL_OP_OPTILING`（TilingFunc + TilingParse）；op_api 侧 L0（`ADD_TO_LAUNCHER_LIST_AICORE`、`executor->AllocTensor`、`OP_TYPE_REGISTER`）与 L2（`CREATE_EXECUTOR`、`l0op::Contiguous`、`l0op::ViewCopy`、`CommonOpExecutorRun`）；op_kernel 侧 `__aicore__` 计算函数。
- 验证过的平台与 CANN 版本：ascend950（arch35；op_def `AICore().AddConfig("ascend950")` + `binary/ascend950/` 产物核对，tool_observed）；CANN 版本未在本检查中记录。
- 适用粒度：`platform_wide`——Host/Kernel 职责分层与各层注册/调度 API 是 CANN 算子工程的结构性约定，与算子类型无关。
- 路线优先级：`recommended_fastpath`（纯读审计，零侵入）。
- 明确不适用 / 未验证场景：入图（GE 图执行）形态的计算归属、AICPU 算子形态未验证；Host 侧通过间接手段（如 tiling 期解引用数据指针）接触数据值的形态属静态读取判据的盲区，未验证。

## 判别方法（Host 职责白名单 + Kernel 计算清单，本例全部实际执行）

**第一步：逐文件读 op_host / op_api，按白名单核对。Host 合法职责只有四类：算子注册、形状推导、切分参数计算（Tiling）、参数校验与调度；Host 代码不得接触输入数据 x/grid 的实际值（只允许指针与元信息）。**

本例核对结果：

| Host 文件 | 白名单职责 | 核对结果 |
|---|---|---|
| op_host/*_def.cpp | 算子定义注册 | 仅 Input("x")/Input("grid")/Output("y") + 3 个 Attr + `AICore().AddConfig("ascend950")` |
| op_host/*_infershape.cpp | 形状推导 | 仅 `*yShape = {N, C, Dout, Hout, Wout}` 拼接 + 维度校验 |
| op_host/arch35/*_tiling.cpp | 切分与 TilingData 填充 | 平台信息获取、维度校验、`CeilDiv` 多核切分、TilingData 填充、步长预计算、BlockDim/TilingKey 设置、workspace 置 0 |
| op_api/aclnn_*.cpp（L2） | 参数校验/Contiguous/调度 | CheckParams（null/dtype/format/shape/attr）+ 空 Tensor 快速返回 + Contiguous + 调 L0 + ViewCopy + CommonOpExecutorRun |
| op_api/*.cpp（L0） | 形状推导/AllocTensor/调度 | InferShape + AllocTensor + IsAiCoreSupport（NpuArch::DAV_3510 + dtype 白名单）+ ADD_TO_LAUNCHER_LIST_AICORE |

**第二步：kernel 侧列计算函数清单，确认全部数据计算落在 `__aicore__` 函数内。**

本例清单：`CopyGridIn`（DataCopyPad GM→UB 读 grid 坐标）、`ComputeCoordinates`（坐标映射/填充/索引/权重）、`Unnormalize`（align_corners 两种公式）、`ApplyPadding`（zeros 不改坐标只算 mask / border 用 ScalarFmin(ScalarFmax(...)) 钳位）、`ComputeReflection`（正数取模 + 越界反射 + span=0 特判）、`ScalarRint`（nearest 的 round-half-to-even）、`GatherAndInterpolate`（逐通道 DataCopyPad gather + 标量加权求和）、`CopyResultOut`（Duplicate + Cast + DataCopyPad 写回 GM）、`Process`（GetBlockIdx 多核分发）。

**第三步：边界定性——步长预计算不算数据计算。**

tiling 中的 `hwStride = H * W` 等 5 个 stride 字段是"避免 Kernel 内每个采样点重复乘法"的切分参数预填，输入只有 shape 元信息、不接触数据值，属合法 Host 职责；数据流核对兜底：Host 侧全部文件中没有任何对 x/grid 元素值的读取或计算语句，所有数据读取、坐标计算、采样、插值、结果写回均在 AICore 侧完成。

## 完整代码（判别所依据的关键 verbatim 代码，取自本检查读取的终态源码）

Host tiling 的多核切分 + TilingData 填充 + 步长预计算段（判别"Host 只做切分"的直接证据）：

```cpp
    // 5. 多核切分：按采样点总数均分
    int64_t totalPoints = N * Dout * Hout * Wout;
    int64_t blockFactor = CeilDiv(totalPoints, coreNum);
    if (blockFactor == 0) {
        blockFactor = 1;
    }
    int64_t usedCoreNum = CeilDiv(totalPoints, blockFactor);

    // 6. 填充 TilingData
    auto tiling = context->GetTilingData<BenchNoHarnessGridSampler3dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(BenchNoHarnessGridSampler3dTilingData), 0,
                 sizeof(BenchNoHarnessGridSampler3dTilingData)) != EOK,
        OP_LOGE(context, "memset_s tiling data error"), return ge::GRAPH_FAILED);

    tiling->N = N;
    tiling->C = C;
    tiling->D = D;
    tiling->H = H;
    tiling->W = W;
    tiling->Dout = Dout;
    tiling->Hout = Hout;
    tiling->Wout = Wout;
    tiling->totalPoints = totalPoints;
    tiling->blockFactor = blockFactor;
    tiling->paddingMode = padMode;
    tiling->alignCorners = alignCorners ? 1 : 0;
    tiling->interpolationMode = interpMode;
    // 预计算步长
    tiling->hwStride = H * W;
    tiling->dhwStride = D * H * W;
    tiling->cdhwStride = C * D * H * W;
    tiling->outPlaneStride = Hout * Wout;
    tiling->outVolStride = Dout * Hout * Wout;
```

Host L0 的调度段（判别"Host 只做调度"的直接证据）：

```cpp
static const aclTensor* BenchNoHarnessGridSampler3dAiCore(
    const aclTensor* x, const aclTensor* grid, const aclTensor* y,
    int64_t interpolationMode, int64_t paddingMode, bool alignCorners,
    aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessGridSampler3dAiCore, x, grid, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessGridSampler3d,
        OP_INPUT(x, grid), OP_OUTPUT(y),
        OP_ATTR(interpolationMode, paddingMode, alignCorners));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessGridSampler3dAiCore failed."),
             return nullptr);
    return y;
}
```

Kernel 侧多核分发与逐采样点计算主循环（判别"计算全在 Kernel"的直接证据）：

```cpp
template <typename T, int INTERP_MODE>
__aicore__ inline void GridSampler3D<T, INTERP_MODE>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    int64_t startIdx = blockIdx * blockFactor_;
    int64_t endIdx = startIdx + blockFactor_;
    if (endIdx > totalPoints_) {
        endIdx = totalPoints_;
    }

    for (int64_t p = startIdx; p < endIdx; p++) {
        // 1. 读取 grid 坐标
        CopyGridIn(p);

        // 2. 计算坐标映射、填充、索引、权重（标量 FP32）
        ComputeCoordinates(p);

        // 3. 逐通道 gather + 插值
        for (int64_t c = 0; c < C_; c++) {
            GatherAndInterpolate(c, p);
        }
    }
}
```

## 设计决策

- 判据取"Host 职责白名单"而非"Host 代码行数/复杂度"：tiling 里必然有切分参数计算（CeilDiv、stride 预计算），按代码量或"有无算术"判会误伤合法实现；白名单法把"切分/推导/校验/调度"与"对输入数据值的计算"区分开。
- 步长预计算定性为合法：TilingData 中预填 hwStride/dhwStride/cdhwStride/outPlaneStride/outVolStride，Kernel 直接读取（`offsets_[i] = d0c*hwStride_ + h0c*W_ + w0c`），避免每采样点重复乘法——这是切分参数，不是数据计算。
- 数据流核对作为兜底判据：从"Host 是否接触数据值"这一数据流视角封闭结论，而不只依赖函数名/文件名表面特征。

## 系统信息

- 审计对象：自定义算子完整工程（op_kernel 4 文件 / op_host 4 文件 / op_api 4 文件 + CMakeLists，共 13 个源文件，全部读取核对）。
- 目标平台：ascend950（arch35）；kernel 二进制位于 `build/op_kernel/ascendc_kernels/binary/ascend950/`。
- Host 侧文件与注册入口一一对应：def.cpp → `OP_ADD(BenchNoHarnessGridSampler3d)`；infershape.cpp → `IMPL_OP_INFERSHAPE(...)`；arch35/tiling.cpp → `IMPL_OP_OPTILING(...).Tiling(...).TilingParse<...CompileInfo>(...)`（`struct ...CompileInfo {}` 空结构体必须定义，入图场景依赖）；op_api L0 → `OP_TYPE_REGISTER` + `ADD_TO_LAUNCHER_LIST_AICORE`；op_api L2 → `extern "C"` 两段式 `aclnn...GetWorkspaceSize` + `aclnn...`。

## 预防

- 验收检查前先写下四类 Host 合法职责清单，再逐文件对照，不要边读边凭感觉判。
- 审计报告按"文件 → 白名单职责 → 核对结果"三列表格落盘，可复核。

## 验证信号

本检查结论来自同一次纯读取审计（op_host/op_api/op_kernel 全部 13 个源文件逐文件读取核对，检查报告落盘为算子工程根下的 MANUAL_CHECK.md）：

- op_host tiling 函数仅含平台信息获取、维度校验、`CeilDiv(totalPoints, coreNum)` 切分、TilingData 字段填充与步长预计算、BlockDim/TilingKey 设置、workspace 置 0；无任何对 x/grid 数据值的计算（来源平台：950；证据等级：tool_observed）。
- op_api L0 仅含 InferShape/AllocTensor/IsAiCoreSupport/ADD_TO_LAUNCHER_LIST_AICORE；L2 仅含 CheckParams/空 Tensor 快速返回/Contiguous/ViewCopy/CommonOpExecutorRun（来源平台：950；证据等级：tool_observed）。
- op_host def 仅含 Input/Output/Attr 注册与 AICore ascend950 配置；infershape 仅含 y=(N,C,Dout,Hout,Wout) 拼接（来源平台：950；证据等级：tool_observed）。
- op_kernel 实现头含全部数据计算函数（CopyGridIn/ComputeCoordinates/Unnormalize/ApplyPadding/ComputeReflection/ScalarRint/GatherAndInterpolate/CopyResultOut/Process，均为 `__aicore__ inline`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 白名单判据针对算子仓式三层工程（op_kernel/op_host/op_api）；入图（GE 图执行）形态与 AICPU 算子形态未验证。
- "Host 不得接触数据值"以源码静态读取为判；Host 侧通过间接手段（如 tiling 期解引用数据指针做数据相关分支）的形态属静态判据盲区，未验证。
- 步长预计算的定性边界：只覆盖"纯 shape 派生乘法"；若 tiling 中出现依赖输入数据值的分支/计算，即越界为 Host 计算（本例无此形态）。
- 其他 CANN 版本各层注册/调度 API 形态可能变化（未验证，推断可迁移）。

**未决项**：

- 运行时核验未做：静态读取判据的兜底信号（上板 profiling 确认计算确实发生在 AICore 侧、Host 侧无数据计算痕迹）未采集，接续时优先补齐。
## 坏实践

- 按"Host 代码里有算术表达式"判"Host 侧做了计算"——tiling 中的 CeilDiv 切分与 stride 预计算是合法切分参数，这样判会把所有正常算子都判成违规。
- 只抽查 op_host 不查 op_api——L0 层同样可能藏计算（本例 L0 的 InferShape/AllocTensor/调度均需核对）。

## 误诊

- 无（白名单判据一次通过，无误判轮次）。
