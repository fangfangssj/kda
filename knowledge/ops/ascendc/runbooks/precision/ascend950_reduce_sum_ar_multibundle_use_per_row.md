---
type: "Runbook"
title: "Ascend950 ReduceSum AR 多行归约逐行错"
description: "Ascend950 上 ReduceSum<AR> 单次传入 aBundle>1 行时逐行结果错（总量近似但逐元素偏）；改逐行调用 ReduceSum(aBundle=1) 规避。仅当 aBundle>1 才触发。"
tags: ["task.debugging", "topic.precision", "topic.compute.reduction"]
status: stable
sources: []
created_at: "2026-08-17T03:56:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["per_row_reduce"]
---

# Ascend950 ReduceSum AR 多行归约逐行错

## 现象

在 Ascend950（3510）上开发 `euclidean_norm` 算子 Branch-0（Normal Tail-R, `Pattern::Reduce::AR`）kernel 时，ST 用例 B0-3/4/5/11 精度失败：输出值与期望接近但超出容差，例如 B0-3（bf16 `[2,3]`, axis=`[-1]`）`expected[0]=1.32812` 而 `actual[0]=1.49219`；B0-11（fp32 rank=8）`expected[0]=0.6291` 而 `actual[0]=1.4347`。编译成功无报错，TilingUT 全过、tiling 路由正确。现象不是全零或 -1，而是"看起来算了一部分但逐元素错"。

关键相关性：所有 `aLen=1`（即 `aBundle=1`）的用例全部通过；所有 `aLen>1`（`aBundle>1`）的用例全部失败。失败集合与 aBundle 是否大于 1 严格对应。

## 适用场景

- 工程形态：Ascend C kernel 直调（`op_kernel/arch35/*.h` 内 `EuclideanNormKernel` 类），regbase 路径；
- API 路径：`AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR | RA, isReuseSource=true>`，`srcShape={rows, rBundle}`，`srcInnerPad=true`；
- 验证平台与版本：Ascend950（3510），CANN 对应 asc-devkit（build 产物 `aic-ascend950-ops-info.json`）；其余平台未验证；
- 适用粒度：`operator_family`——任何在 950 上用 `ReduceSum<AR/RA>` 单次处理多行归约（aBundle>1）的算子；
- 路线优先级：`recommended_fastpath`——逐行 `ReduceSum(aBundle=1)` 是已验证可行的快路径；
- 不适用或未验证场景：a3/a2 未验证（推断可能不触发，但未实测，见 `## 失效边界`）；`aBundle=1` 用例本身不受影响，无需改写。

## 根因

`ReduceSum<float, Pattern::Reduce::AR>` 在 `srcShape={aBundle, rBundle}`（`aBundle>1`）且 `srcInnerPad=true` 时，3510 上的归约实现在多行间做了错误的分割/重排——总体量近似保留但逐行结果错位。根因链：单次 ReduceSum 调用把 `aBundle` 行作为外维一并送入 → 950 归约硬件/API 在 aBundle>1 时按错误的行边界切分 → 每个输出元素混入了相邻行的部分贡献。这不是数据搬运问题（DataCopyPad/普通 DataCopy 给出完全一致的结果），而是 ReduceSum 内部的多行处理缺陷。

## 解决配方

把单次多行 `ReduceSum` 改成对每一行 `a` 循环调用 `ReduceSum`（`aBundle=1` 每次），`dst[a]` 与 `src[a*rBundle]` 一一对应。

### before（ver​batim，修复前 `ReduceRPattern`，取自 file_read 事件）

```cpp
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::ReduceRPattern()
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto src = tmpAll;
    auto dst = tmpAll[static_cast<int32_t>(tmpSlotElems_)];

    const uint32_t aBundle = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t rBundle =
        static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_); // padded（srcInnerPad=true 要求）

    if constexpr (isTailR) {
        uint32_t srcShape[2] = {aBundle, rBundle};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
            dst, src, srcShape, /*srcInnerPad=*/true);
    } else {
        uint32_t srcShape[2] = {rBundle, aBundle};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
            dst, src, srcShape, /*srcInnerPad=*/true);
    }
}
```

### after（ver​batim，修复后 `ReduceRPattern`，取自 file_read 事件）

```cpp
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::ReduceRPattern()
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto src = tmpAll;
    auto dst = tmpAll[static_cast<int32_t>(tmpSlotElems_)];

    const uint32_t aBundle = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t rBundle =
        static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_); // padded（srcInnerPad=true 要求）

    // 逐行 ReduceSum（aBundle=1 每次），规避 3510 上 aBundle>1 时 AR pattern 的分割缺陷
    if (isTailR_) {
        for (uint32_t a = 0; a < aBundle; ++a) {
            uint32_t srcShape[2] = {1, rBundle};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                dst[static_cast<int32_t>(a)],
                src[static_cast<int32_t>(a * rBundle)],
                srcShape, /*srcInnerPad=*/true);
        }
    } else {
        for (uint32_t a = 0; a < aBundle; ++a) {
            uint32_t srcShape[2] = {rBundle, 1};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                dst[static_cast<int32_t>(a)],
                src[static_cast<int32_t>(a * rBundle)],
                srcShape, /*srcInnerPad=*/true);
        }
    }
}
```

注：修复同时把 `if constexpr (isTailR)` 改为运行时 `if (isTailR_)`（`isTailR_` 为 Init 中从模板参数镜像的运行时成员），原因是同类内同时存在 tail-R/tail-A 两个实例化时 `if constexpr` 会在 3510 上引发另一类 codegen 污染（见 `## 设计决策` 与同 trace 的 `if constexpr` 失败路线卡）。

## 设计决策

- 为何选逐行 ReduceSum 而非换 API/换 dtype：观察到 `aBundle=1` 用例全过、`aBundle>1` 全败，最强假设即"单次多行 ReduceSum 的多行分割有问题"；逐行是改动最小、语义等价（`sum` 满足行间独立性）的规避，无需引入新 API 或升精度；
- 放弃的候选：(1) 把 `DataCopyPad` 换成 per-row plain `DataCopy`——实测结果完全不变，证明搬运无错，放弃；(2) per-row scalar `GetValue/SetValue`、dense 单 block `DataCopy`、valid-only `CastSquareVf`——前序 agent 均已试且给出相同错误结果，放弃；
- 当前方案属哪档路线：`recommended_fastpath`（已实测 12/15 通过、TilingUT 83/83 无退化）。

## 验证信号

本卡验证来自 euclidean_norm 算子 Task 24 的 ST 运行（`cd operators/euclidean_norm_package/tests/aclnn && ./run.sh`，950）：

- 修复前 ST 8/15 passed，B0-3/4/5/11 输出值接近但错误（来源平台：Ascend950；证据等级: tool_observed）；
- 修复后 ST 12/15 passed（B0-1/3/4/5/8/9/10/11/12/13/14/15），B0-2/6/7 允许 FAIL（路由 Group/Branch-1）（来源平台：Ascend950；证据等级: tool_observed）；
- TilingUT 83/83 无退化（来源平台：Ascend950；证据等级: tool_observed）；
- 辅助判据：`grep -c __VEC_SCOPE__ euclidean_norm.h`=6、`grep -c ReduceSum`=6、`asc_vf_call`=1、`__simd_vf__`=2（来源平台：Ascend950；证据等级: tool_observed）。
## 性能影响

逐行调用使 ReduceSum 调用次数从 1 增至 aBundle 次（本算子 aBundle 通常为 2~16），相对单次多行路线增加若干倍 ReduceSum 调用开销，但避免了错误结果。未实测绝对延迟；相对代价以"调用次数 × aBundle"计，参照路线为单次多行 ReduceSum（来源平台：Ascend950）。

## 失效边界

- a3、a2 未实测；推断若其归约实现不共享 950 的多行分割路径则可能不触发，但"未验证，推断可迁移"——不得据此前移到 `platforms`；
- `aBundle=1` 场景本就不触发，无需应用本配方；
- 配方仅针对 `ReduceSum<AR|RA>` + `srcInnerPad=true` + `aBundle>1` 的组合；其它归约 API（如 `ReduceMax`、自定义归约）未在本 trace 验证，不外推。

## 坏实践

- 把"输出接近但错"当精度容差问题去放宽容差（rtol/atol 调松）——会掩盖 ReduceSum 多行分割缺陷；本 trace 的修复是改 ReduceSum 调用结构，不是放宽阈值；
- 仅凭"DataCopyPad 在 3510 有硬件缺陷"的先入为主就直接反复换搬运策略——本 trace 实测换搬运策略结果完全不变，应先做 aBundle 相关性诊断再动手。

## 误诊

- 编排者与早期执行者一度把根因归为"3510 `copy_gm_to_ubuf_align_v2`（DataCopyPad）硬件缺陷"（agent_reported），据此反复更换数据搬运策略均无果；实测替换后结果不变，最终判定根因在 ReduceSum 而非 DataCopyPad。

## 预防

在 950 上用 `ReduceSum<AR|RA>` 时，若发现"部分用例值接近但错、且失败集合恰为外维行数>1 的用例"，优先怀疑单次多行 ReduceSum 的多行分割；用逐行调用作为最小语义等价规避。