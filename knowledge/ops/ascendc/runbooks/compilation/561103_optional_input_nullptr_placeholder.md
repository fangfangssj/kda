---
type: "Runbook"
title: 561103 可选输入 nullptr 须 shape=[0] 占位
description: 可选输入为 nullptr 时 GetWorkspaceSize 返回 561103 或输出全 0；L0 API 用 executor->AllocTensor 建 shape=[0] 占位 tensor，且 [0] 而非 [1] 才能与 Tiling 的 shapeSize>0 可选性判定对齐。
tags: [task.debugging, topic.implementation.api_layer]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# 561103 可选输入 nullptr 须 shape=[0] 占位

## 现象

在自定义算子（MoE finalize routing，含 skip1/skip2/bias/scales/expert 五个可选输入）的 C++ ST 联调验证阶段，通过 `bash run.sh`（Real 模式，NPU 执行，Ascend950PR）跑 L1 全量用例时，凡可选输入含 nullptr 的用例统一失败，`aclnnXxxGetWorkspaceSize` 直接报错，日志原文：

```
[L1-016] fp16 mode0 无skip 仅bias+scales+expert
  参数: mode=0, dtype=fp16, num_rows=1024, K=8, H=512, E=16, C=0
  [FAIL] GetWorkspaceSize 失败: 561103
```

失败组合包括：skip1/skip2 为空仅 bias+scales+expert、bias 与 expert 为空、scales 为空等（凡把任一可选 aclTensor 传成 nullptr 的用例）。另有一类更隐蔽的表现：某用例（skip1=null、skip2=null、bias=null、仅 scales+expert）不报 561103，但输出全为 0（actual=0.000000，精度全错）——即 nullptr 可选输入既可能显式报错，也可能静默算错。当时代码状态：L2 aclnn 层对 nullptr 可选输入用 `(skip1 != nullptr) ? l0op::Contiguous(skip1, executor) : nullptr` 跳过连续化后，把 nullptr 原样传给 L0 API，L0 API 直接把可选 tensor 塞进 `ADD_TO_LAUNCHER_LIST_AICORE`。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke 形态），aclnn 两段式接口 + L0 API + 自定义 AscendC kernel，C++ ST Real 模式联调。
- API 路径：`aclnnXxxGetWorkspaceSize` / L2 `l0op::Contiguous` / L0 `ADD_TO_LAUNCHER_LIST_AICORE`（`OP_INPUT` 宏）/ `executor->AllocTensor` / Tiling `gert::TilingContext::GetInputShape`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（任何带可选输入的 aclnn 自定义算子，L0 层用 ADD_TO_LAUNCHER_LIST_AICORE 注册）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：GE 图模式（框架对 nullptr 可选输入自动生成 placeholder，行为未在本轨迹验证）；Tiling 侧不用 shapeSize>0 判定可选性的算子（见失效边界）。

## 根因

`ADD_TO_LAUNCHER_LIST_AICORE` 不支持 nullptr tensor：L2 层 `l0op::Contiguous` 对 nullptr 跳过返回 nullptr 后，nullptr 直传 L0 的 `ADD_TO_LAUNCHER_LIST_AICORE`，注册失败使 L0 返回 nullptr，L2 的 `CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR)` 触发 561103。完整调用链：

```
aclnnGetWorkspaceSize
  → l0op::Contiguous(skip1=nullptr) → 跳过, 返回 nullptr
  → l0op::<Op>(..., skip1Contiguous=nullptr, ...)
    → <Op>AiCore(..., skip1=nullptr, ...)
      → ADD_TO_LAUNCHER_LIST_AICORE(..., nullptr, ...) → 失败
      → 返回 nullptr
    → 返回 nullptr
  → CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR) → 561103
```

## 解决配方

在 L0 API 中为 nullptr 可选输入创建 placeholder tensor 再进 `OP_INPUT`。**关键约束：placeholder 的 shape 必须是 [0]（0 元素），不能是 [1]**——本算子 Tiling 的可选性判定契约是"shape 元素数 > 0 才算存在"：

```cpp
// Tiling 侧（op_host/arch35/*_tiling.cpp，GetInputMetaInfo 内）——可选性判定契约：
// 注意: aclnn eager 模式下可选输入为 nullptr 时 GE 会生成 placeholder tensor (shape 为空/0元素)，
//       所以不能仅用 GetInputShape(i) != nullptr 判断，还需检查 shape 元素数 > 0
auto IsOptionalPresent = [](gert::TilingContext* ctx, size_t idx) -> bool {
    auto shapePtr = ctx->GetInputShape(idx);
    if (shapePtr == nullptr) { return false; }
    auto shape = shapePtr->GetStorageShape();
    // placeholder tensor 的 shape 通常为 0 维或 0 元素
    if (shape.GetDimNum() == 0) { return false; }
    int64_t shapeSize = 1;
    for (size_t d = 0; d < shape.GetDimNum(); d++) {
        shapeSize *= shape.GetDim(d);
    }
    return (shapeSize > 0);
};
info.hasSkip1 = IsOptionalPresent(context, 2);
info.hasSkip2 = IsOptionalPresent(context, 3);
info.hasBias = IsOptionalPresent(context, 4);
info.hasScales = IsOptionalPresent(context, 5);
info.hasEfsr = IsOptionalPresent(context, 6);
```

修复前（L0 API，op_api/<op>.cpp，nullptr 直传宏）：

```cpp
static const aclTensor* BenchNoHarnessMoeFinalizeRoutingAiCore(
    const aclTensor* expandedPermutedRows,
    const aclTensor* expandedSrcToDstRow,
    const aclTensor* skip1,
    const aclTensor* skip2,
    const aclTensor* bias,
    const aclTensor* scales,
    const aclTensor* expertForSourceRow,
    int64_t dropPadMode,
    const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessMoeFinalizeRoutingAiCore, expandedPermutedRows, expandedSrcToDstRow,
           skip1, skip2, bias, scales, expertForSourceRow, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessMoeFinalizeRouting,
        OP_INPUT(expandedPermutedRows, expandedSrcToDstRow, skip1, skip2, bias, scales, expertForSourceRow),
        OP_ATTR(dropPadMode),
        OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessMoeFinalizeRoutingAiCore failed."),
             return nullptr);
    return out;
}
```

修复后（终版：nullptr 可选输入替换为 shape=[0] placeholder，dtype 按原输入约定取主 dtype 或 int32）：

```cpp
static const aclTensor* BenchNoHarnessMoeFinalizeRoutingAiCore(
    const aclTensor* expandedPermutedRows,
    const aclTensor* expandedSrcToDstRow,
    const aclTensor* skip1,
    const aclTensor* skip2,
    const aclTensor* bias,
    const aclTensor* scales,
    const aclTensor* expertForSourceRow,
    int64_t dropPadMode,
    const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessMoeFinalizeRoutingAiCore, expandedPermutedRows, expandedSrcToDstRow,
           skip1, skip2, bias, scales, expertForSourceRow, out);

    // ADD_TO_LAUNCHER_LIST_AICORE 不支持 nullptr tensor，
    // 为 nullptr 可选输入创建 placeholder tensor (shape=[0], 0 元素)
    // Tiling 代码的 IsOptionalPresent 通过 shapeSize > 0 判断输入是否存在，
    // shape=[0] 会被正确识别为"不存在"，Kernel 通过 TilingData 标志跳过相关计算
    auto createPlaceholder = [&executor](DataType dtype) -> const aclTensor* {
        Shape s;
        s.SetDimNum(1);
        s.SetDim(0, 0);  // shape=[0], 0 元素
        return executor->AllocTensor(s, dtype);
    };

    DataType mainDtype = expandedPermutedRows->GetDataType();
    const aclTensor* placeholderMain = createPlaceholder(mainDtype);
    const aclTensor* placeholderInt32 = createPlaceholder(DataType::DT_INT32);

    // nullptr 可选输入替换为 placeholder
    const aclTensor* skip1T = (skip1 != nullptr) ? skip1 : placeholderMain;
    const aclTensor* skip2T = (skip2 != nullptr) ? skip2 : placeholderMain;
    const aclTensor* biasT = (bias != nullptr) ? bias : placeholderMain;
    const aclTensor* scalesT = (scales != nullptr) ? scales : placeholderMain;
    const aclTensor* expertT = (expertForSourceRow != nullptr) ? expertForSourceRow : placeholderInt32;

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessMoeFinalizeRouting,
        OP_INPUT(expandedPermutedRows, expandedSrcToDstRow, skip1T, skip2T, biasT, scalesT, expertT),
        OP_ATTR(dropPadMode),
        OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessMoeFinalizeRoutingAiCore failed."),
             return nullptr);
    return out;
}
```

修改后重新编译算子包（`make -j` + `make package` + 安装 custom_opp run 包）并重跑全量 ST。

## 设计决策

- **placeholder 放在 L0 API 而非 L2 API**：L2 层 `l0op::Contiguous` 已经对 nullptr 做了跳过语义，替换点收口在进入 `ADD_TO_LAUNCHER_LIST_AICORE` 之前最靠近失败点的位置；L0 层一处修改覆盖所有调用路径。
- **placeholder shape=[0] 而非 [1]**：初版用 shape=[1]，自查 Tiling 代码时发现 `IsOptionalPresent` 以 shapeSize>0 判定存在性，[1] 会让 Tiling 误认为输入存在、Kernel 去读未初始化的占位数据（可能引入精度错误甚至越界），遂改为 [0]。[0] 同时满足"宏不接受 nullptr"与"Tiling 判定不存在"两个约束。
- **dtype 选择**：skip1/skip2/bias/scales 占位用主 dtype（与 expanded_permuted_rows 一致），expert 用 DT_INT32，与算子 dtype 校验约束一致；因 Kernel 按 TilingData 标志跳过读取，占位 dtype 不参与计算。
- 当前方案属 recommended_fastpath：一次替换覆盖全部可选输入，不依赖 GE 图模式行为。

## 预防

- L0 API 中所有进入 `OP_INPUT` 的可选 tensor，先做 nullptr → shape=[0] placeholder 替换，再进宏。
- placeholder 一律 0 元素，与 Tiling 可选性判定契约（shapeSize>0）保持一致。
- Tiling 侧可选性判断不能只看 `GetInputShape(i) != nullptr`，必须检查 shape 元素数 > 0（GE 在 eager 模式本来就可能给空 shape 的 placeholder）。
- ST 用例表应覆盖"每个可选输入单独为空"和"多个可选输入同时为空"的组合，否则该缺陷只在部分组合暴露（本例 561103 与输出全 0 分别在不同组合出现）。

## 验证信号

本卡信号来自该算子联调验证阶段的一次真实修复闭环（Ascend950PR Real 模式）：修改 L0 API 前后各跑一轮全量 ST，修复后重新编译并安装算子包再复跑。

- 修复前，可选输入含 nullptr 的用例报 `[FAIL] GetWorkspaceSize 失败: 561103`；某组合（skip1/skip2/bias 为空）静默输出全 0（来源平台：950；证据等级：tool_observed）。
- 修复后同批用例 PASS：skip1/skip2 为空用例 MERE=2.40e-08、MARE=9.29e-04（threshold=9.77e-04，524288 elems）；bias/expert 为空与 scales 为空用例 MERE=0.00e+00、MARE=0.00e+00（来源平台：950；证据等级：tool_observed）。
- 原输出全 0 的用例转为 PASS（MERE=0.00e+00，893 个小值域元素 max_abs_err=0.00e+00），确认静默错误与 561103 同根因（来源平台：950；证据等级：tool_observed）。
- 当轮全量汇总：通过 41、失败 0、跳过 3（3 例为当时尚未实现的混合精度路径）；随后全功能版本全量 57/57 通过，含各可选输入组合用例（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 依赖"Tiling 以 shapeSize>0 判定可选输入存在性"这一契约：若算子的 Tiling/Kernel 以其他方式（专用属性、shape nullptr 判断）判定可选性，需重新核对 placeholder 语义；若 Tiling 完全不判可选性而直接读数据，本配方不适用。
- placeholder 只消除 561103 与 nullptr 直传问题；若 Kernel 未按 TilingData 标志跳过对应读取，仍会出错——本配方必须与 Kernel 侧 hasXxx 标志分支配套。
- 仅在 Ascend950PR + CANN 9.1.0 aclnn eager 路径实测；GE 图模式下框架对 nullptr 可选输入的 placeholder 生成行为未验证（未验证，推断图模式无需本修复）。
- 本算子可选输入的 dtype 约定为主 dtype/int32；其他 dtype 组合的占位 tensor 未验证。

**未决项**：

- 输出全 0 用例在 kernel 侧的确切执行路径未单独追根因（已被 placeholder 修复一并解决）；如需机制级确认，可在旧代码上复现并单步定位（优先级低）。
- shape=[1] 误判场景的实际危害（是否真的产生错误数据/越界）未上板验证，仅静态推断。
## 坏实践

- 第一版 placeholder 用 shape=[1]：会被 Tiling `IsOptionalPresent` 误判为"输入存在"，Kernel 将读取占位数据，可能引入精度错误或越界（本次在实现中通过核对 Tiling 判定逻辑自查纠正，未上板验证其危害；不要带着 [1] 版本上板）。
- 只在 L2 层给 nullptr 做跳过（`Contiguous` 三目）而不管 L0 宏是否接受 nullptr——跳过动作本身把 nullptr 原样下传，正是本缺陷的成因。

## 误诊

- 输出全 0 的用例最初被怀疑是 Kernel 内 `has_bias = use_bias && use_expert` 组合逻辑导致 expanded 累加被跳过——该假设仅基于代码阅读，未获证实；placeholder 修复后该用例通过，证实根因是 nullptr 可选输入而非 Kernel 逻辑（该机制猜测已被修复结果替代）。
