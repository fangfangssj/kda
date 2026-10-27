---
type: "Runbook"
title: tail-R 切分 rUbFactor 实切最内 R 才 FloorAlign
description: reduce tail-R（AR pattern）UB 切分实现配方：ComputeAUbFactor 的 atmp 带 min(cacheLine, CeilDiv(totalA, coreNum)) 多核下界；ComputeRUbFactor 对最内 R 做 CeilAlign，FloorAlign 守卫仅在「tail-R 且 rSplit==lastR 且 rUbFactor<rAxisSize（实际切 chunk）」时把 rUbFactor 落到块倍数、为 0 则 TILING_FAIL；tail-A/tail-R 共用单驱动与统一字段集。
tags: [task.implementation, topic.tiling, topic.compute.reduction, topic.alignment]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# tail-R 切分 rUbFactor 实切最内 R 才 FloorAlign

## 现象

EuclideanNorm（L2 范数 reduce）多分支开发中，tail-R 分支（AR pattern、axisNum 偶数、tilingKey=1，分支判定规则见库内 [合轴 fuse R 无尾 A → tail-R key=1](reduce_fuse_pattern_is_tail_r_branch_rule.md)）的 TilingFunc 此前是单核桩（usedCoreNum=1、全部切分字段 0），独立 oracle 的 TilingUT 以 0/9 红基线（TDD）锁定期望：多核切分、`rUbFactor` 带 FloorAlign 守卫、R 全载时扩 A。实现该分支切分时，tail-A 分支（key=0）已有的 UB 三步（ComputeAUbFactor / ComputeRUbFactor / ExpandAIfRFullyLoaded）不能直接照搬——tail-R 的最内轴是 R（burst 尾轴），对齐守卫的位置和触发条件与 tail-A 相反，写错守卫条件要么破坏对齐、要么把不该对齐的中间轴错误对齐。

## 适用场景

- 工程形态：多分支 reduce 算子的 op_host TilingFunc 实现（合轴后 A 起头严格交替 pattern 的 UB/多核切分）。
- API 路径：`gert::TilingContext` + 自研切分函数（ComputeAUbFactor / ComputeRUbFactor / ComputeRiMax / CeilAlignI64 / CeilDivI64）；平台参数来自 `platform_ascendc::PlatformAscendC`（coreNum / blockSize / UB 容量 / cacheLineSize）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0——TilingUT（host 加载 op_host .so、真实平台参数）与 aclnn ST（上板）双验证。
- 适用粒度：operator_family——采用"合轴 pattern + aUb/rUb 双因子切分"的 reduce 算子族。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：tail-A（尾轴是 A）分支的守卫条件相反（见设计决策对照）；group（A×R 二维分核）分支另有切分链，未在本配方内；同类双轴预算反解环的整体骨架见库内 [UB 预算反解切分：R 因子反解、R 驻留扩 A、钳制环](ub_budget_backsolve_dual_axis_split_clamp_ring.md)。

## 解决配方

tail-R 分支三处关键改动（与 tail-A 共用同一驱动，靠 `ctx.isTailR` 守卫分叉）：

**① ComputeAUbFactor：atmp 带多核下界**（tail-A 无此下界）——A 侧搬运粒度至少能喂满多核，防止过度细分：

```cpp
// ComputeAUbFactor — climb from the innermost A outward,
//   absorbing whole A axes into innerAProdAlign until ≥ atmp.
//   tail-A (isTailR=false): atmp = cacheLineSize / dtypeSize (no multi-core
//   lower bound); innermost A is the burst tail axis → CeilAlign to bsAElem.
//   tail-R (isTailR=true): atmp carries the multi-core lower bound
//   min(cacheLineSize/dtypeSize, CeilDiv(totalA, coreNum));
//   aSplit is never the burst tail (tail axis is R) → innerAProdAlign
//   and aUbFactorAlign take the plain (no CeilAlign) branch.
void ComputeAUbFactor(EuclideanNormCtx& ctx)
{
    const int32_t lastA = LastAAxisIdx(ctx);
    const int64_t bsAElem = ctx.blockSize / ctx.dtypeSize;
    int64_t atmp;
    if (ctx.isTailR) {
        // tail-R multi-core lower bound:
        //   atmp = min(cacheLineSize/dtypeSize, CeilDiv(totalA, coreNum)).
        int64_t totalA = 1;
        for (int32_t i = 0; i < ctx.axisNum; i += 2) {   // A axes sit at even indices
            totalA *= ctx.axisShape[static_cast<size_t>(i)];
        }
        atmp = std::min(ctx.cacheLineSize / ctx.dtypeSize,
                        CeilDivI64(totalA, static_cast<int64_t>(ctx.coreNum)));
        atmp = std::max<int64_t>(atmp, 1);
    } else {
        atmp = ctx.cacheLineSize / ctx.dtypeSize;   // tail-A: no lower bound
    }

    ctx.innerAProd = 1;
    // …（向外吸收整根 A 轴直到 ≥ atmp，tail-R 走 plain 分支、tail-A 对最内 A CeilAlign）
```

**② ComputeRUbFactor：最内 R CeilAlign + FloorAlign 守卫仅"实际切 chunk"时触发**：

```cpp
// ComputeRUbFactor — climb from the innermost R outward as long
//   as rSize×innerRProdAlign ≤ r_i_max.
//   tail-A (isTailR=false): rSplit is NOT the burst tail (tail axis is A) →
//   rUbFactorAlign == rUbFactor, no FloorAlign guard, no CeilAlign.
//   tail-R (isTailR=true): innermost R (rSplit==lastR) is the burst tail →
//   innerRProdAlign climb CeilAligns it; rUbFactorAlign = CeilAlign(rUbFactor,
//   bsElem); the tail-R-only FloorAlign guard fires when actually chunking
//   (rSplit==lastR && rUbFactor<rAxisSize).
ge::graphStatus ComputeRUbFactor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const int32_t lastR = LastRAxisIdx(ctx);
    if (lastR < 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rIMax = ComputeRiMax(ctx);
    if (rIMax < 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;

    ctx.innerRProd = 1;
    ctx.innerRProdAlign = 1;
    ctx.rSplitAxisIdx = lastR;
    while (true) {
        const int64_t rSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
        if (rSize * ctx.innerRProdAlign > rIMax) { break; }
        const int32_t prev = PrevSameTypeAxis(ctx, ctx.rSplitAxisIdx, /*wantR=*/true);
        if (prev < 0) { break; }
        if (ctx.rSplitAxisIdx == lastR && ctx.isTailR) {
            // tail-R: innermost R is the burst tail → CeilAlign.
            ctx.innerRProdAlign *= CeilAlignI64(rSize, bsElem);
        } else {
            // tail-A: rSplit is never the burst tail → plain multiply.
            ctx.innerRProdAlign *= rSize;
        }
        ctx.innerRProd *= rSize;
        ctx.rSplitAxisIdx = prev;
    }

    const int64_t rAxisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
    ctx.rUbFactor = std::min(rIMax / ctx.innerRProdAlign, rAxisSize);
    ctx.rUbFactor = std::max<int64_t>(ctx.rUbFactor, 1);
    // FloorAlign guard (tail-R + rSplit==lastR + actually chunking R):
    //   rUbFactor must be a bsElem multiple for the burst-tail R to stay aligned.
    if (ctx.isTailR && ctx.rSplitAxisIdx == lastR && ctx.rUbFactor < rAxisSize) {
        ctx.rUbFactor = ctx.rUbFactor / bsElem * bsElem;   // floor to a bsElem multiple
        if (ctx.rUbFactor == 0) {
            return ge::GRAPH_FAILED;   // TILING_FAIL
        }
    }
    // rUbFactorAlign: CeilAlign on burst-tail R (tail-R + rSplit==lastR).
    if (ctx.isTailR && ctx.rSplitAxisIdx == lastR) {
        ctx.rUbFactorAlign = CeilAlignI64(ctx.rUbFactor, bsElem);
    } else {
        // tail-A: no CeilAlign.
        ctx.rUbFactorAlign = ctx.rUbFactor;
    }
    return ge::GRAPH_SUCCESS;
}
```

**③ FillNormalTilingData 统一完整字段集 + 单驱动跑双分支**（去掉 tail-R 单核桩）：

```cpp
// NORMAL/GROUP public-path fill.
//   tail-A (isTailR=false) + tail-R (isTailR=true): the split chain has already
//   populated every ctx split field — write them all into TilingData. The two
//   branches differ only in which CeilAlign/FloorAlign guards fired inside the
//   split (resolved by isTailR), not in the field set written.
void FillNormalTilingData(EuclideanNormTilingData* td, EuclideanNormCtx& ctx)
{
    td->axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i]  = ctx.axisShape[i];     // unused positions already =1 from PreprocessPattern
        td->axisStride[i] = ctx.axisStride[static_cast<size_t>(i)];
    }
    td->rGroupCnt      = ctx.rGroupCnt;
    td->cacheBufUbSize = kCacheBufBytes;

    // The same field set applies to tail-A and tail-R; only the split values
    // differ (resolved inside the split functions via isTailR guards).
    td->aSplitAxisIdx     = ctx.aSplitAxisIdx;
    td->rSplitAxisIdx     = ctx.rSplitAxisIdx;
    td->aUbFactor         = ctx.aUbFactor;
    td->aUbFactorAlign    = ctx.aUbFactorAlign;
    td->rUbFactor         = ctx.rUbFactor;
    td->rUbFactorAlign    = ctx.rUbFactorAlign;
    td->innerAProd        = ctx.innerAProd;
    td->innerAProdAlign   = ctx.innerAProdAlign;
    td->innerRProd        = ctx.innerRProd;
    td->innerRProdAlign   = ctx.innerRProdAlign;
    td->aLoopCntTotal     = ctx.aLoopCntTotal;
    td->aSplitChunkCnt    = ctx.aSplitChunkCnt;
    td->aBigCoreLoopCnt   = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->aBigCoreCnt       = ctx.aBigCoreCnt;
    td->rLoopCntTotal     = ctx.rLoopCntTotal;
    td->usedCoreNum       = ctx.usedCoreNum;
    td->preReduceUbSize   = ctx.preReduceUbSize;
    td->postReduceUbSize  = ctx.postReduceUbSize;
    td->tmpBufUbSize      = ctx.tmpBufUbSize;
}
```

```cpp
    } else {
        // NORMAL/GROUP branch — run 合轴四步 then branch split.
        PreprocessPattern(ctx);
        ctx.isEmptyTensor = false;
        // The split functions are branch-aware (isTailR guards), so a single
        // driver serves both tail-A (key=0) and tail-R (key=1). Group split
        // (ComputeGroupSplit) + workspace 属后续独立分支；ShouldUseGroup
        // 此处 only feeds the tilingKey encoding.
        if (ComputeBranch0Split(context, ctx) != ge::GRAPH_SUCCESS) {
            context->SetBlockDim(1);
            size_t* wsFail = context->GetWorkspaceSizes(1);
            if (wsFail != nullptr) { wsFail[0] = 0; }
            return ge::GRAPH_FAILED;
        }
        ctx.isGroup = ShouldUseGroup(ctx);
        FillNormalTilingData(td, ctx);
    }
```

## 设计决策

- **FloorAlign 守卫三条件缺一不可**：`isTailR && rSplit==lastR && rUbFactor<rAxisSize`。前两条锁定"burst 尾轴是最内 R"；第三条"实际切 chunk"（rUbFactor<rAxisSize）保证 R 轴能整段装进 UB 时（rUbFactor==rAxisSize）不引入人为取整损失——整载无需对齐守卫。守卫触发时 `rUbFactor` floor 到 bsElem 倍数；若 floor 到 0（UB 装不下一个对齐块）直接 `TILING_FAIL`，不产出非法切分。
- **rUbFactorAlign 仅对 burst 尾 R CeilAlign**：尾块补齐用 CeilAlign（向上），主块粒度用 FloorAlign（向下），两个方向服务于不同目的（尾块覆盖 vs 主块对齐），不能混用。
- **tail-A/tail-R 共用一套驱动与字段集**：差异全部内聚在切分函数的 isTailR 守卫里，TilingData 字段集两个分支完全一致——避免"每个分支一套 Fill"带来的字段漂移。
- **atmp 多核下界 min(cacheLine/dtypeSize, CeilDiv(totalA, coreNum))**：A 侧既不小于 cacheLine 粒度、也不小于"每核至少一块"的均衡需求；tail-A 分支无此下界（尾轴是 A，粒度由 burst 尾轴对齐决定）。
- 备选方案"每个分支独立一套切分函数"被放弃：字段集和爬升骨架 90% 相同，复制两套会让后续 group 分支再复制一遍，守卫条件修一处漏两处。

## 系统信息

- 平台/工具链：Ascend950PR（arch35），CANN 9.1.0；平台参数（coreNum/blockSize/UB 容量/cacheLineSize）经 `platform_ascendc::PlatformAscendC` 查询。
- 相关文件：`op_host/arch35/euclidean_norm_tiling_arch35.cpp`（实现）、`tests/tiling/test_branch1_tiling.cpp`（独立 oracle UT，手算公式复刻、不调 optiling）、`tests/aclnn/test_aclnn_branch1.cpp`（上板 ST）。
- 验证命令：`tests/tiling/run.sh`（编译并跑全部 TilingUT）、`operators/build.sh euclidean_norm_package` + `tests/aclnn/run.sh`（上板 ST）。

## 验证信号

验证来自 tail-R 分支 TDD 全链（Ascend950PR，CANN 9.1.0）：

- 实现前红基线：Branch-1 TilingUT 0/9（桩字段全 0，oracle 非零，如 `aUbFactor=0 expected 4`、`rUbFactor=0 expected 4096`）（来源平台：950；证据等级：tool_observed）。
- 实现后：Branch-1 TilingUT **9/9 PASSED**（覆盖主块/尾块/非对齐/多核 FloorAlign/underfill/full reduce/partial-A-tail/fp16/int32）（来源平台：950；证据等级：tool_observed）。
- 回归：公共 TilingUT 19/19 PASSED + key-encoding sanity PASSED + Branch-0 TilingUT 9/9 PASSED，无退化（来源平台：950；证据等级：tool_observed）。
- 上板：tail-R kernel 接线后 aclnn ST key=1 精度 **18/18 PASSED**（maxRel≤8.467e-08，含 `[1024] axes=[]` full reduce、`[17] axes=[-1]` 非对齐尾块、int32 bitwise、bf16）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 守卫语义以"合轴后 A 起头严格交替 + 最内 R 是 burst 尾轴"的 pattern 为前提；tail-A（尾轴是 A）分支守卫条件相反（rUbFactorAlign==rUbFactor、无 FloorAlign），照抄本配方到 tail-A 会错误对齐中间 R 轴。
- FloorAlign 到 0 即 TILING_FAIL 的兜底意味着"UB 单块装不下一个 bsElem 对齐块"的极端 shape 会被拒绝而非降级，调用方需能处理失败。
- `CeilDiv(totalA, coreNum)` 下界假设 A 元素可均匀按 aUbFactor 分块；A 总量小于 coreNum 时 candidate=totalA、部分核空转（usedCoreNum 取实际有活核数），该退化路径已由 UT 的 partial/underfill 用例覆盖。
- 其他 NPU 架构（不同 blockSize/UB 容量）未验证，公式随平台参数自适应、机制推断可迁移。

**未决项**：

- 无（该分支切分已由独立 oracle UT + 上板 ST 双重闭环；group 分支切分链属后续任务）。
## 坏实践

- 把 FloorAlign 无条件套在所有 R 轴上（不判 `rSplit==lastR`）会让中间 R 轴的 innerRProdAlign 虚增，rIMax 预算被错误消耗、切分过度保守。
- 在 `rUbFactor==rAxisSize`（整载）时也 floor 会把整载切成"整载-尾块"两段，白白多一轮搬运。
