---
type: "Runbook"
title: "empty分支SetBlockDim(0)与aUbFactor误式判别"
description: "空 tensor 归约算子（EMPTY_A/EMPTY_R 分支）的 empty-branch TilingUT 0/14 全 FAILED；按 MISMATCH 字段分流——仅 blockDim mismatch 指向 EMPTY_A 的 SetBlockDim(0)（应为 max(usedCoreNum,1)=1），aLoopCntTotal/aUbFactor/usedCoreNum 全 mismatch 指向 EMPTY_R 的 aUbFactor=min(aTotal,maxSingleUb)（应为 min(maxSingleUb,CeilDiv(aTotal,coreNum))）；修正后 14/14。 R2（2026-09，CANN 9.1.0）：EMPTY_R 四约束完整公式 candidate=min(CeilDiv(aTotal,coreNum), 64KB/dtypeSize, aTotal)、不足 4KB 回退 4KB 粒度；字段重映射 axisNum=0/axisShape[0]=aTotal/cacheBufUbSize=0/postReduceUbSize=CeilAlign(aUbFactor×dtypeSize,blockSize)；含 FillEmptyTilingData/ComputeEmptyRSplit 完整实现与桩期单核降级坏实践。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.compute.reduction"]
status: stable
sources: []
created_at: "2026-08-18T16:11:31Z"
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# empty分支SetBlockDim(0)与aUbFactor误式判别

## 现象

在编写 EuclideanNorm（沿给定轴计算 L2 范数）算子 `op_host/euclidean_norm_host.cpp` 的 empty 分支 TilingUT 时，运行独立 empty-branch UT（`tests/tiling/build/test_branch_empty_tiling`）得到 0/14 全失败基线。oracle 按 DESIGN-BRANCH-empty.md §2.2/§2.3 计算期望值，TilingFunc 实际写出的 TilingData 字段与期望值逐字段比对，输出 `MISMATCH` 行。

EMPTY_A 路径（A 轴 size=0 → 全核早退）失败（`./run.sh` 输出原样）：

```
[empty_a_basic] MISMATCH blockDim: exp=1 act=0
Test empty_a_basic              FAILED  [§2.1/2.2: A轴=0 → EMPTY_A, usedCoreNum=0, SetBlockDim(1)]
[empty_a_keepdims] MISMATCH blockDim: exp=1 act=0
Test empty_a_keepdims           FAILED  [§2.2: keep_dims=true 不影响 tiling (EMPTY_A)]
```

EMPTY_R 路径（R 轴 size=0、A 轴非零 → 多核写 0）失败（多 A 轴 case）：

```
[empty_r_multi_a] MISMATCH aLoopCntTotal: exp=6 act=1
[empty_r_multi_a] MISMATCH aBigCoreCnt: exp=6 act=1
[empty_r_multi_a] MISMATCH usedCoreNum: exp=6 act=1
[empty_r_multi_a] MISMATCH aUbFactor: exp=1 act=6
[empty_r_multi_a] MISMATCH blockDim: exp=6 act=1
Test empty_r_multi_a            FAILED  [§2.5: 多 A 轴 aTotal=2*3=6, aUbFactor=min(16384,1)=1]
[empty_r_three_a] MISMATCH aLoopCntTotal: exp=24 act=1
```

`act` 列为 TilingFunc 实际写出的 TilingData 字段值，`exp` 为 oracle 期望值。

## 适用场景

- **工程形态**：op_host TilingFunc（算子仓 op_host 层，非 kernel 直调）。
- **API 路径**：`gert::TilingContext` → `context->SetBlockDim(...)`、`context->GetTilingData<EuclideanNormTilingData>()`、`context->SetTilingKey(...)`；涉及 `EuclideanNormCtx::kEmptyA` / `kEmptyR` 空张量分类。
- **验证平台与版本**：Ascend950，CANN 9.1.0（x86_64-linux）；host-only TilingUT（gtest，link `libeuclidean_norm_op_host_ut_lib.so`），未上板。
- **适用粒度**：`operator_family`（适用于所有"沿轴归约 + 空 tensor 早退"类算子的 empty 分支 tiling）。
- **路线优先级**：`recommended_fastpath`（DESIGN 规定的正确公式，非 fallback）。
- **不适用或未验证场景**：a3/a2 平台未验证（推断可迁移，见失效边界）；非归约类算子的 empty 处理不在本卡范围。

## 判别方法

**触发**：空 tensor 归约算子（EMPTY_A/EMPTY_R 分支）的 empty-branch TilingUT 0/14 全 FAILED；TilingData 字段比对输出 `MISMATCH` 行，`act`（TilingFunc 实际写出）与 `exp`（oracle 期望）不一致。

**可执行探针（按序）**：

1. **分层复现 + 字段分流**：跑 empty-branch TilingUT，按 `MISMATCH` 字段分流根因：
   - **仅 `blockDim` mismatch（exp=1 act=0）** → EMPTY_A 路径（A 轴 size=0，全核早退）：`HandleEmptyTensor` 的 EMPTY_A 分支调 `SetBlockDim(0)`，而框架要求 `blockDim ≥ 1`（`GetBlockDim()` 返回 0 → oracle 期望 `max(usedCoreNum,1)=1`）。
   - **`aLoopCntTotal` / `aBigCoreCnt` / `usedCoreNum` / `aUbFactor` / `blockDim` 全 mismatch（act=1 exp=aTotal）** → EMPTY_R 路径（R 轴 size=0、A 轴非零，多核写 0）：`aUbFactor = std::min(aTotal, maxSingleUb)` 漏了"按核均摊"步骤，应为 `std::min(maxSingleUb, CeilDiv(aTotal, coreNum))`。

2. **EMPTY_R 数值探针**（`aTotal = 6, coreNum = 32, maxSingleUb = 16384`）：
   - 旧公式：`aUbFactor = min(6, 16384) = 6` → `aLoopCnt = CeilDiv(6, 6) = 1` → `usedCoreNum = 1` → `blockDim = 1`。六元素挤在一核，31 核空闲。
   - DESIGN 公式：`aUbFactor = min(16384, CeilDiv(6, 32)) = min(16384, 1) = 1` → `aLoopCnt = CeilDiv(6, 1) = 6` → `usedCoreNum = aTotal = 6` → `blockDim = 6`。六元素均摊到 6 核。
   - 关键换算点：`CeilDiv(aTotal, coreNum)` 把"总 A 量按核均摊"与"单 buf 64KB 上限"取小，保证 `aLoop` 数能撑满核。

3. **对照 DESIGN 公式**：DESIGN-BRANCH-empty.md §2.2 规定 EMPTY_A `SetBlockDim(max(usedCoreNum, 1)) = SetBlockDim(1)`（usedCoreNum=0 时全核早退，但框架仍需 1 个 block 启动）；§2.3 规定 EMPTY_R `aUbFactor = min(maxSingleUb, CeilDiv(aTotal, coreNum))`。若 TilingFunc 的 empty 分支偏离这两条，即命中本判别。

**原始观测（tool_observed）**：empty-branch UT 修复前 0/14 PASS，14 case 全 FAILED（`empty_a_basic`/`empty_a_keepdims`/`empty_a_priority`/`empty_a_multi_zero`/`empty_a_fp16`/`empty_r_basic`/`empty_r_keepdims`/`empty_r_multi_a`/`empty_r_three_a`/`empty_r_big_small` 等）；EMPTY_A `blockDim` exp=1 act=0；EMPTY_R 多 A 轴 `aLoopCntTotal` exp=6/24 act=1、`aUbFactor` exp=1 act=6、`blockDim` exp=6 act=1（来源平台：950；证据等级：tool_observed）。

**解释（root-cause interpretation）**：两条独立根因同一函数 `HandleEmptyTensor` 的不同分支、不同字段：(1) EMPTY_A 分支 `SetBlockDim(0)` 与 DESIGN §2.2 `SetBlockDim(1)` 不一致——框架要求 `blockDim ≥ 1`，EMPTY_A 的 `usedCoreNum=0` 是"全核早退"语义，但 `SetBlockDim` 仍须设 `max(usedCoreNum, 1)=1` 使框架启动 1 个 block、kernel 内经 `blockIdx >= usedCoreNum_(=0)` 早退；设 0 导致 `GetBlockDim()` 返回 0。(2) EMPTY_R 分支 `aUbFactor = std::min(aTotal, maxSingleUb)` 与 DESIGN §2.3 `aUbFactor = std::min(maxSingleUb, CeilDiv(aTotal, coreNum))` 不一致——旧公式直接用 `aTotal` 与单 buf 上限取小，未按核均摊 `aTotal`，导致 `1 < aTotal < coreNum*maxSingleUb` 时 `aUbFactor` 偏大、`aLoopCntTotal=1`（单核跑完全部 A），而期望按核均摊后 `aLoopCntTotal=aTotal`。这是根因解释，明确为 interpretation——实测闭环信号是分层 mismatch + 修正后全过（见验证信号）。

## 验证信号

- empty-branch UT 修复前 0/14 PASS，14 case 全 FAILED（blockDim/aLoopCntTotal/aUbFactor/usedCoreNum mismatch）（来源平台：950；证据等级：tool_observed）
- EMPTY_A `blockDim` 从 `act=0` → 与 `exp=1` 一致（来源平台：950；证据等级：tool_observed）
- EMPTY_R 多 A 轴 `aLoopCntTotal` 从 `act=1` → 与 `exp=aTotal(6/24/...)` 一致、`blockDim` 从 `act=1` → 与 `exp=aTotal` 一致（来源平台：950；证据等级：tool_observed）
- empty-branch UT 修正 `HandleEmptyTensor` 两处公式后 14/14 PASS（`all 14 empty branch test cases passed`，全字段 mismatch 消失）（来源平台：950；证据等级：tool_observed）

- （R2 增补，Ascend950PR + CANN 9.1.0，EMPTY 分支 TDD 闭环）实现前红基线：Branch-2 TilingUT 0/13（桩：cacheBufUbSize=16384 expected 0、usedCoreNum=1 expected 8、postReduceUbSize=0 expected 4096、aUbFactor=0 expected 1250/1024/16384/2048 逐字段点名）（来源平台：950；证据等级：tool_observed）。
- （R2 增补）实现后：Branch-2 TilingUT 13/13 PASSED（含 4KB 回退、64KB 上限、部分核、fp16/bf16/int32 变体）（来源平台：950；证据等级：tool_observed）。
- （R2 增补）回归：公共 TilingUT 19/19 + Branch-0 9/9 + Branch-1 9/9 无退化；整包 build.sh 退出 0、无 tiling struct 告警（来源平台：950；证据等级：tool_observed）。
- （R2 增补）同 trace 后续 EMPTY kernel 入口接线后上板 aclnn ST 14/14 按位 0 通过（Branch-0 5/5、Branch-1 18/18 无退化）——空分类 tiling 的上板闭环（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 平台：仅 Ascend950（CANN 9.1.0）host-only TilingUT 实测；a3/a2 未验证，推断可迁移（`SetBlockDim` / `CeilDiv` 语义跨平台一致，但 `coreNum` / `ubSize` / `cacheLineSize` 平台常量需按目标平台取值）。
- 适用边界：仅"沿轴归约 + 空 tensor 早退（EMPTY_A）/ 空 R 轴多核写 0（EMPTY_R）"类算子；非归约类算子的空 tensor 处理不适用。
- 上板验证：本开发会话 的 aclnn 真机测试在 stub kernel 阶段未跑通（361001 kernel-not-found），empty 分支的上板精度未在本开发会话 验证。

- （R2 增补）4KB/64KB 阈值是设计常量（非平台硬约束），dtype 越大可用粒度档越少；其他 NPU 架构未验证，机制推断可迁移。
- （R2 增补）`empty_r_output_value=0` 是本算子（sqrt 空集）语义；其他 reduce（max 空域、带 empty_output_value 参数）填充值须按算子语义另定。

## 坏实践

- 空 tensor 归约分支的 tiling 复用 normal 路径的 `min(aTotal, maxUb)` 直觉，而不对齐 DESIGN 的 `SetBlockDim(max(usedCoreNum,1))` 与"按核均摊"公式。
- 把 empty-branch UT 的 `MISMATCH` 行误判为 oracle 错误（公共 `test_tiling` 的 empty-case oracle 曾匹配 buggy `SetBlockDim(0)`，把 stub 阶段的 buggy 输出当 red-baseline 固化）——oracle 必须按 DESIGN 公式生成期望值。

- （R2 增补）桩阶段为"先让 kernel 能跑"把 EMPTY_R 压成单核并保留 normal 的 16KB cacheBuf——真实 kernel 会按 TilingData 字段当真执行。
- （R2 增补）空 tensor 分支复用 normal 切分链再除零兜底——空轴进合轴/切分公式只会产生 0/0 与无意义字段，应在短路分类后完全走重映射路径。

## 增补：四约束完整实现与字段重映射（R2，2026-09，CANN 9.1.0）

同算子后续迭代给出 EMPTY_R 切分的完整公式与实现（本卡判别方法第 2 步公式的超集：在"按核均摊 + 64KB 单 buf 上限"外补 4KB 下界与 aTotal 上限）：

- (a) aUbFactor × dtypeSize ≥ 4KB（防过度切分，不足时回退 4KB 粒度——宁可核少用满）；
- (b) aUbFactor₀ = CeilDiv(aTotal, coreNum)（优先多核、负载均衡）；
- (c) aUbFactor ≤ 64KB / dtypeSize（单 buf 上限）；
- (d) aUbFactor ≤ aTotal（总量上限）。

合成：`candidate = min(CeilDiv(aTotal,coreNum), 64KB/dtypeSize, aTotal)`；`candidate×dtypeSize < 4KB` 时取 `4KB/dtypeSize`（may underfill cores，有意权衡）。

字段重映射（复用 normal 的 TilingData struct，不另建 struct）：`axisNum=0`（不进合轴）、EMPTY_A 全字段清零（`std::memset` 保证漏填字段确定性为 0）、EMPTY_R `axisShape[0]=aTotal`（合所有 A 轴）、`cacheBufUbSize=0`（不分配 cacheBuf——与 normal 固定 16KB 的关键差异）、`preReduceUbSize/tmpBufUbSize=0`、`postReduceUbSize=CeilAlign(aUbFactor×dtypeSize, blockSize)`（EMPTY_R 唯一 UB buffer）、`rGroupCnt=0`（empty 与 group 互斥）。

终版实现（verbatim；常量与 helper 为同文件既有定义）：

```cpp
constexpr int64_t kFp32Bytes      = 4;   // sizeof(float)
constexpr int64_t kNOut           = 1;   // single output y

// §5.5.13 / DESIGN-BRANCH-2 §2.2 EMPTY_R multi-core split constants:
constexpr int64_t kEmpty4KBLower   = 4096;          // (a) 4KB lower bound per core (avoid over-splitting)
constexpr int64_t kEmpty64KBSingle = 64 * 1024;     // (c) single-buf 64KB cap (kMaxSingleUbBytes, only outBuf)

// EMPTY short-circuit fill (§5.3.3 field remapping + DESIGN-BRANCH-2 §2 split).
//   EMPTY_A → usedCoreNum=0 (full-core early exit), every split/buf field 0.
//   EMPTY_R → Duplicate-0 fill over aTotal output elements: real multi-core
//             split (§2.2 "4 constraints"); tmpBufUbSize=0 / preReduceUbSize=0 /
//             cacheBufUbSize=0 (no preIn/tmp/cache allocation, §5.3.3).
// =============================================================================
int64_t ComputeATotal(const EuclideanNormCtx& ctx)
{
    int64_t aTotal = 1;
    for (size_t i = 0; i < ctx.rawShape.size(); ++i) {
        if (!ctx.rawIsReduce[i]) { aTotal *= ctx.rawShape[i]; }
    }
    return aTotal;
}

// §5.5.13 / DESIGN-BRANCH-2 §2.2 EMPTY_R multi-core split — "4 constraints":
//   (a) aUbFactor × sizeof(D_T) ≥ 4KB           (avoid over-splitting)
//   (b) aUbFactor₀ = CeilDiv(aTotal, coreNum)   (prefer multi-core, load balance)
//   (c) aUbFactor ≤ 64KB / sizeof(D_T)          (single-buf 64KB cap)
//   (d) aUbFactor ≤ aTotal                       (output total cap)
//   combine: candidate = min(CeilDiv(aTotal,coreNum), 64KB/sizeof, aTotal);
//            if candidate×sizeof < 4KB → fallback 4KB/sizeof (may underfill).
// Fused aLoop multi-core balance (outerAProd=1, empty has no outer A) populates
// ctx split fields; caller writes them into TilingData.
void ComputeEmptyRSplit(EuclideanNormCtx& ctx)
{
    const int64_t aTotal   = ComputeATotal(ctx);
    const int64_t cap64KB  = kEmpty64KBSingle / ctx.dtypeSize;
    const int64_t lower4KB = kEmpty4KBLower   / ctx.dtypeSize;

    int64_t candidate = std::min({CeilDivI64(aTotal, static_cast<int64_t>(ctx.coreNum)),
                                  cap64KB, aTotal});
    if (candidate * ctx.dtypeSize < kEmpty4KBLower) {
        candidate = lower4KB;   // (a) 4KB fallback — may underfill cores
    }
    ctx.aUbFactor = candidate;

    ctx.aSplitChunkCnt    = CeilDivI64(aTotal, ctx.aUbFactor);
    ctx.aLoopCntTotal     = ctx.aSplitChunkCnt;   // outerAProd=1 (empty has no outer A)
    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / static_cast<int64_t>(ctx.coreNum);   // floor
    ctx.aBigCoreCnt       = static_cast<int32_t>(ctx.aLoopCntTotal % static_cast<int64_t>(ctx.coreNum));
    ctx.aBigCoreLoopCnt   = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum       = (ctx.aSmallCoreLoopCnt > 0) ? ctx.coreNum : ctx.aBigCoreCnt;
    ctx.usedCoreNum       = std::max(ctx.usedCoreNum, 1);

    // postReduceUbSize = CeilAlign(aUbFactor × sizeof(D_T), blockSize) — the only
    // UB buffer EMPTY_R allocates (outBuf single path), §5.3.3 / §4.4.
    ctx.postReduceUbSize  = CeilAlignI64(ctx.aUbFactor * ctx.dtypeSize, ctx.blockSize);
}

void FillEmptyTilingData(EuclideanNormTilingData* td, EuclideanNormCtx& ctx)
{
    // memset already zeroed TilingData; only the EMPTY-specific non-zero fields
    // are set here. axisNum=0 → axisShape[1..8] / axisStride stay 0 from memset.
    td->axisNum = 0;   // EMPTY skips PreprocessPattern (UT asserts axisNum==0).
    if (ctx.emptyKind == EuclideanNormCtx::EmptyKind::EMPTY_A) {
        // §2.1 EMPTY_A: full-core early exit. Every split/buf field stays 0.
        td->axisShape[0]  = 0;
        td->usedCoreNum   = 0;
        ctx.usedCoreNum   = 0;
    } else {
        // §2.2 EMPTY_R: degenerate Duplicate-0 fill. Real multi-core split.
        ComputeEmptyRSplit(ctx);
        td->axisShape[0]     = ComputeATotal(ctx);   // §5.3.3: axisShape[0] = aTotal
        td->aUbFactor        = ctx.aUbFactor;
        td->aLoopCntTotal    = ctx.aLoopCntTotal;
        td->aSplitChunkCnt   = ctx.aSplitChunkCnt;
        td->aBigCoreLoopCnt  = ctx.aBigCoreLoopCnt;
        td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
        td->aBigCoreCnt      = ctx.aBigCoreCnt;
        td->usedCoreNum      = ctx.usedCoreNum;
        td->postReduceUbSize = ctx.postReduceUbSize;
    }
    // §5.3.3: EMPTY zeros cacheBufUbSize (no cacheBuf allocation; unlike
    // normal/group which set kCacheBufBytes). preReduceUbSize / tmpBufUbSize
    // stay 0 from memset (no preIn/tmp allocation, §5.3.3).
    td->cacheBufUbSize = 0;
}
```

桩期坏形态（修复前，verbatim——EMPTY_R 被强制单核 + cacheBufUbSize 沿用 normal 的 16KB 常量；多核 dispatch 与非零缓冲字段会被真实 kernel 当真，上板即错）：

```cpp
void FillEmptyTilingData(EuclideanNormTilingData* td, EuclideanNormCtx& ctx)
{
    // memset done by caller; here we only set the fields the public
    // UT asserts on (key inputs: axisShape[0], usedCoreNum).
    td->axisNum = 0;   // EMPTY skips PreprocessPattern (UT expects axisNum==0).
    if (ctx.emptyKind == EuclideanNormCtx::EmptyKind::EMPTY_A) {
        // EMPTY_A: full-core early exit.
        td->axisShape[0]  = 0;
        td->usedCoreNum   = 0;
        ctx.usedCoreNum   = 0;
    } else {
        // EMPTY_R: degenerate Duplicate-0 fill over aTotal output elements.
        const int64_t aTotal = ComputeATotal(ctx);
        td->axisShape[0]  = aTotal;
        // STUB-STAGE DOWNGRADE: real multi-core split lands in branch Task 30.
        // Until then, force single-core launch so the stub kernel can run
        // safely (the stub kernel cannot carry multi-core dispatch under a
        // real tilingKey — it would segfault at runtime, regression on the
        // aclnn smoke test). Public TilingUT only asserts usedCoreNum>=1.
        td->usedCoreNum   = 1;
        ctx.usedCoreNum   = 1;
    }
    td->cacheBufUbSize = kCacheBufBytes;
}
```
