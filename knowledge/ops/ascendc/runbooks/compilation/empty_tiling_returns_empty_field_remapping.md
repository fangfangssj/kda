---
type: "Runbook"
title: "空tensor tiling：EMPTY_A清零+EMPTY_R重映射"
description: "softmax 类 returns_empty 算子空 tensor 分支的 TilingFunc 字段重映射配方：EMPTY_A 全字段清零全核早退，EMPTY_R 按 chunk 公式切分但 kernel 空转；含 host glue 的 SetBlockDim>=1 与空 workspace 处理。"
tags: ["task.implementation", "topic.tiling", "topic.compute.normalization"]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# 空tensor tiling：EMPTY_A清零+EMPTY_R重映射

## 现象

在为 softmax 类算子 BenchHarnessSoftmax（单输入 x、attr dim、输出与输入同 shape；fp16/fp32/bf16；目标 Ascend950PR）开发"空 tensor"分支（任一维 size=0，tilingKey=2 Empty 模板）时，按测试先行流程先写了 13 条 tiling 单元测试（gtest，oracle 为独立重算的期望字段表，覆盖 EMPTY_A 5 类 shape 与 EMPTY_R 3 类 shape 的字段重映射）。当时 `ComputeBranch2Tiling` 还是返回 `INT32_MIN` 哨兵、不填任何 tiling 字段的桩，跑 UT 得到 13/13 失败的预期红基线，典型失败输出（test_branch2_tiling.cpp:286/287）：

```
Expected equality of these values:
  act.usedCoreNum
    Which is: 0
  exp.usedCoreNum
    Which is: 1
  act.aUbFactor
    Which is: 0
  exp.aUbFactor
    Which is: 6
```

随后按设计落地真实切分公式，13/13 转绿、全量 72/72（含 59 条旧分支回归）通过，上板 aclnn ST 空 tensor 用例 24/24 通过、旧分支 46/46 + 37/37 不退化。期间还把 arch35 TilingFunc glue 的空 tensor 短路从"手写少数字段"改为"统一走 ComputeBranch2Tiling"，使 glue 产物与被 UT 测的 seam 完全一致。

## 适用场景

- 工程形态：算子仓自定义算子开发（AscendC op_host TilingFunc + gtest TilingUT + aclnn ST 上板回归）。
- API 路径：op_host 侧可测 tiling seam（`ComputeBranch2Tiling(in, td)` 纯函数）+ arch 平台 glue（`HandleEmptyTensor` 短路、`context->SetTilingKey / SetBlockDim / GetWorkspaceSizes`）。
- 验证过的平台与 CANN 版本：Ascend950PR（result.csv soc=Ascend950PR_9579）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"输出与输入同 shape、空输入返回同 shape 空 tensor（returns_empty）"的 softmax 类归约算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：
  - 输出非空的归约算子（如对空 R 轴求和/均值并给输出填默认值）：EMPTY_R 不能空转，须填 `empty_r_output_value` 并 CopyOut，本配方的 `N_out=1` 且输出为空假设不成立；
  - 带 postReduceInput / post-elewise / 多输出 / 需要 workspace 的算子：`N_post_alive / N_tmp` 等参数须按实际语义代入；
  - 非空 tensor 路径（本卡只覆盖空分支短路）。

## 根因/解决配方

空 tensor 分支不能复用 normal 分支的 tiling 公式：normal 会按非空公式切分并固定申请 cacheBuf，而空分支的正确语义是"输出元素数 = ∏(A 轴) × R 轴长度"，一旦任一维为 0 输出即空，kernel 无任何数据可搬可算。配方分两支：

1. **分类**（优先级 EMPTY_A > EMPTY_R，因为 A=0 时输出必空、无论 R 是否为 0）：先 dim 折叠与越界检查（`dim<0` 加 rank，出界返回 -1）；再判"任一维 size==0"（否则返回 -2，不是本分支）；`EMPTY_A` = 任一非归约轴（`i != dimIdx`）size==0；`EMPTY_R` = 归约轴自身 size==0 且所有 A 轴 > 0。
2. **EMPTY_A**：`usedCoreNum=0`（全核早退），所有字段清零——含 `axisShape[0..2]=0`（覆盖 unused-slot 惯例的 {1,1,1}）与 `cacheBufUbSize=0`（不分配 cacheBuf，这是与 normal 分支"固定 16KB cacheBuf"的关键差异）。用 `std::memset(&td, 0, sizeof(td))` 零初始化保证漏填字段也是确定性的 0。
3. **EMPTY_R（softmax 特例）**：输出元素数 = ∏(A) × 0 = 0，输出为空 tensor，kernel 空转；但为框架一致性仍按 chunk 公式填字段：
   - `bytes_per_chunk_unit = N_post_alive*sizeofDT + N_out*sizeofDT + N_tmp*sizeof(float)`；softmax 空 R 下 `N_post_alive=0, N_out=1, N_tmp=0`，即 `bytesPerChunkUnit = sizeofDT`；
   - `perChunkElemsNatural = ubSize / bytesPerChunkUnit`；
   - 单 buffer 64KB 上限：`perChunkElemsBy64KB = 65536 / maxBufDtypeSize`，`maxUbFactor = min(perChunkElemsNatural, perChunkElemsBy64KB)`；
   - `aUbFactor = min( max(ceil(4096/sizeofDT), ceil(aTotal/coreNum)), maxUbFactor )` 再 `min(aTotal)`（4KB 下界 + 多核优先 + UB 上限 + aTotal 兜底）；
   - 大小核均衡：`aLoopCntTotal = ceil(aTotal/aUbFactor)`，`aSmallCoreLoopCnt = aLoopCntTotal/coreNum`，`aBigCoreCnt = aLoopCntTotal%coreNum`，`usedCoreNum = (aSmallCoreLoopCnt>0) ? coreNum : aBigCoreCnt`；
   - 字段重映射：`axisNum=0`（不进合轴）、`axisShape[0]=aTotal`（合所有 A 轴）、其余 stride/inner/rLoop/各 UB buffer 尺寸全 0、`rGroupCnt=0`（empty 与 group 互斥）。
4. **glue**：`SetTilingKey(EncodeTilingKey(templateType=0, isEmptyTensor=1, isTailR=0))`；`SetBlockDim(usedCoreNum>0 ? usedCoreNum : 1)`——框架要求 blockDim >= 1，EMPTY_A 的 0 必须抬到 1；`ws[0]=0`（空分支不申请 workspace）。

修复前（stub，TDD 红基线的直接原因）：

```cpp
// ─── Branch-2 (Empty, tilingKey=2) tiling ────────────────────────────────────
// Task 37 stub: returns INT32_MIN sentinel, does not fill td.
// Task 38 will replace with real §2 formula impl (HandleEmptyTensor short-circuit
// + EMPTY_A/EMPTY_R field remapping per DESIGN-BRANCH-2.md §2 / DESIGN §5.3).
int32_t ComputeBranch2Tiling(const Branch2Inputs& in,
                             BenchHarnessSoftmaxTilingData& td)
{
    (void)in;
    (void)td;
    return INT32_MIN;  // sentinel: stub not implemented (TDD red baseline)
}
```

修复后（完整实现，verbatim）：

```cpp
// ─── Branch-2 (Empty, tilingKey=2) tiling ────────────────────────────────────
// Real implementation (Task 38): DESIGN-BRANCH-2.md §2 切分计算 + DESIGN §5.3
// 空 tensor 字段重映射表.  Returns 0 on success; negative on rejection:
//   -1  dim out of range (§5.5 Step 1)
//   -2  non-empty tensor (no axis size == 0 → not branch-2)
//
// softmax EMPTY 语义 (§1.4 / §5.6): returns_empty — 返回同 shape 空 tensor,
// 不进入归约计算.  EMPTY_A 与 EMPTY_R 两子情形统一返回空 tensor, 无任何
// reduce / exp / div 计算链执行.
//
// 优先级: EMPTY_A > EMPTY_R (A=0 时输出必空, 无论 R 是否 0).
//   - EMPTY_A: 任一 A 轴 (i != dimIdx) size=0 → usedCoreNum=0, 全核早退, all fields=0.
//   - EMPTY_R: R 轴 (i == dimIdx) size=0 且所有 A 轴 > 0 → 按 chunk 分核 (§2 公式);
//     chunk-split 字段填入 TilingData 供框架一致性, 但 kernel 不据此产生输出 (§5).
//
// softmax EMPTY_R chunk-split params (§2 / §5):
//   N_post_alive = 0  (softmax 无 postReduceInput)
//   N_out        = 1  (softmax 单输出 y)
//   N_tmp        = 0  (softmax EMPTY_R 不执行 post-elewise, §5)
int32_t ComputeBranch2Tiling(const Branch2Inputs& in,
                             BenchHarnessSoftmaxTilingData& td)
{
    // ── dim fold + range check (§5.5 Step 1) ──
    int32_t dimIdx = 0;
    if (in.rank > 0) {
        dimIdx = static_cast<int32_t>((in.dim < 0) ? in.dim + in.rank : in.dim);
        if (dimIdx < 0 || dimIdx >= in.rank) return -1;
    }

    // ── empty tensor check (branch-2 requires at least one axis size == 0) ──
    bool anyZero = false;
    for (int32_t i = 0; i < in.rank; ++i) {
        if (in.shape[i] == 0) { anyZero = true; break; }
    }
    if (!anyZero) return -2;  // non-empty → not branch-2

    // ── EMPTY_A vs EMPTY_R classification (priority: EMPTY_A > EMPTY_R) ──
    // EMPTY_A: any A axis (i != dimIdx) has size == 0.
    // EMPTY_R: R axis (i == dimIdx) has size == 0 AND all A axes > 0.
    // (reduce-empty-tensor.md §一 / DESIGN-BRANCH-2.md §0)
    bool emptyA = false;
    for (int32_t i = 0; i < in.rank; ++i) {
        if (i != dimIdx && in.shape[i] == 0) { emptyA = true; break; }
    }

    // Zero-init td so any field we forget is deterministic (matches oracle).
    std::memset(&td, 0, sizeof(td));

    if (emptyA) {
        // ── EMPTY_A: usedCoreNum=0, all fields = 0 (全核早退) ──
        // (DESIGN-BRANCH-2.md §1 字段表 EMPTY_A 列 / §2 / reduce-empty-tensor.md §二)
        td.axisNum = 0;
        // axisShape default {0,0,0} after memset (override the {1,1,1} unused-slot
        // convention: EMPTY_A all axisShape = 0, §1 字段表).
        td.axisShape[0] = 0;
        td.axisShape[1] = 0;
        td.axisShape[2] = 0;
        // All other fields already 0 (memset). cacheBufUbSize = 0 (不分配, §1).
        return 0;
    }

    // ── EMPTY_R: R axis = 0, all A > 0 → 按 chunk 分核 (§2 公式) ──
    // aTotal = product of all A axes (i != dimIdx), guaranteed > 0.
    int64_t aTotal = 1;
    for (int32_t i = 0; i < in.rank; ++i) {
        if (i != dimIdx) aTotal *= in.shape[i];
    }

    // §2 chunk-split formulas (DESIGN-BRANCH-2.md §2 + reduce-empty-tensor.md §3.2/§3.3):
    //   bytes_per_chunk_unit = N_post_alive*sizeofDT + N_out*sizeofDT + N_tmp*sizeof(float)
    //                        = 0 + sizeofDT + 0 = sizeofDT  (softmax EMPTY_R)
    constexpr int64_t N_POST_ALIVE = 0;
    constexpr int64_t N_OUT = 1;
    constexpr int64_t N_TMP = 0;
    constexpr int64_t SIZEOF_FLOAT = 4;
    constexpr int64_t MAX_SINGLE_UB_BYTES = 65536;  // 64KB single-buf cap (§3.2 E)

    int64_t bytesPerChunkUnit = N_POST_ALIVE * in.sizeofDT
                              + N_OUT * in.sizeofDT
                              + N_TMP * SIZEOF_FLOAT;

    int64_t perChunkElemsNatural = in.ubSize / bytesPerChunkUnit;

    // max_ub_factor: single-buf 64KB cap (reduce-empty-tensor.md §3.2 E)
    int64_t maxBufDtypeSize = (N_TMP > 0)
        ? MaxI(in.sizeofDT, SIZEOF_FLOAT)
        : in.sizeofDT;
    int64_t perChunkElemsBy64KB = MAX_SINGLE_UB_BYTES / maxBufDtypeSize;
    int64_t maxUbFactor = MinI(perChunkElemsNatural, perChunkElemsBy64KB);

    // aUbFactor: 4KB 下界 + 优先多核 + UB 上限 + aTotal 兜底 (§3.3)
    int64_t minAPerCore = CeilDivI(4096, in.sizeofDT);
    int64_t aUbFactor = MinI(MaxI(minAPerCore, CeilDivI(aTotal, in.coreNum)), maxUbFactor);
    aUbFactor = MinI(aUbFactor, aTotal);
    if (aUbFactor < 1) aUbFactor = 1;  // defensive (aTotal > 0 guarantees ≥1)

    int64_t aLoopCntTotal = CeilDivI(aTotal, aUbFactor);

    // 大小核均衡 (同 normal 公式, §2 / ComputeFusedALoopSplit)
    int64_t aSmallCoreLoopCnt = aLoopCntTotal / in.coreNum;
    int64_t aBigCoreCnt = aLoopCntTotal % in.coreNum;
    int64_t aBigCoreLoopCnt = (aSmallCoreLoopCnt > 0) ? aSmallCoreLoopCnt + 1 : 1;
    int32_t usedCoreNum = (aSmallCoreLoopCnt > 0)
                          ? in.coreNum
                          : static_cast<int32_t>(aBigCoreCnt);

    // ── fill TilingData (DESIGN §5.3 空 tensor 字段重映射表 / DESIGN-BRANCH-2.md §1) ──
    td.axisNum = 0;  // 不进合轴
    td.axisShape[0] = aTotal;  // 合所有 A 轴 (§1 字段表 EMPTY_R 列)
    td.axisShape[1] = 0;
    td.axisShape[2] = 0;
    td.axisStride[0] = 0;
    td.axisStride[1] = 0;
    td.axisStride[2] = 0;
    td.aLoopCntTotal = aLoopCntTotal;
    td.aSplitChunkCnt = aLoopCntTotal;  // 同上 (no 合轴, §1 字段表)
    td.aBigCoreLoopCnt = aBigCoreLoopCnt;
    td.aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    td.aBigCoreCnt = static_cast<int32_t>(aBigCoreCnt);
    td.usedCoreNum = usedCoreNum;
    td.aSplitAxisIdx = 0;
    td.rSplitAxisIdx = 0;
    td.aUbFactor = aUbFactor;
    td.aUbFactorAlign = aUbFactor;  // "实际单 chunk 大小" (§1 字段表, kernel 不读)
    td.rUbFactor = 0;
    td.rUbFactorAlign = 0;
    td.innerAProd = 0;
    td.innerAProdAlign = 0;
    td.innerRProd = 0;
    td.innerRProdAlign = 0;
    td.rLoopCntTotal = 0;
    td.preReduceUbSize = 0;    // 不分配 (§1)
    td.expBufUbSize = 0;       // 不分配 (§1)
    td.postReduceUbSize = 0;   // softmax EMPTY_R 输出为空, 0 元素 (§1)
    td.tmpBufUbSize = 0;       // N_tmp=0, 无 post-elewise 执行 (§1)
    td.cacheBufUbSize = 0;     // 不分配 cacheBuf (§1, empty 与 normal 的关键差异)
    td.rGroupCnt = 0;          // empty 与 group 互斥 (§1)
    return 0;
}
```

配套：文件头补 `#include <cstring>`（`std::memset`）与 `MaxI` helper（原文件只有 `CeilDivI/CeilAlignI/MinI`）：

```cpp
#include "bench_harness_softmax_tiling_host.h"

#include <cstring>  // std::memset (Branch-2 EMPTY_A zero-init)

namespace optiling {

// ─── local arithmetic helpers (independent of the UT oracle's helpers) ──────
inline int64_t CeilDivI(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}
inline int64_t CeilAlignI(int64_t v, int64_t f)
{
    return CeilDivI(v, f) * f;
}
inline int64_t MinI(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
inline int64_t MaxI(int64_t a, int64_t b)
{
    return a > b ? a : b;
}
```

arch35 glue 的空 tensor 短路（修复前手写少数字段 → 修复后统一走被 UT 覆盖的 seam）：

修复前：

```cpp
    // ─── 3) empty tensor short-circuit (§5.4 HandleEmptyTensor) ────────────
    int32_t isRAxisEmpty = 0;
    int32_t isEmpty = HandleEmptyTensor(shapeBuf, rank, dimIdx, isRAxisEmpty);
    if (isEmpty) {
        // softmax EMPTY_A / EMPTY_R both return same-shape empty tensor (§1.4/§5.6);
        // kernel early-exits.  usedCoreNum=0 → all cores return immediately.
        td->usedCoreNum = 0;
        td->axisNum = 0;
        td->expBufUbSize = 0;
        td->postReduceUbSize = 0;
        td->tmpBufUbSize = 0;
        td->rGroupCnt = 0;
        td->rLoopCntTotal = 0;
        context->SetTilingKey(static_cast<uint64_t>(EncodeTilingKey(/*templateType=*/0,
                                                                     /*isEmptyTensor=*/1,
                                                                     /*isTailR=*/0)));
        context->SetBlockDim(1);  // framework requires ≥1
        size_t* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
```

修复后：

```cpp
    // ─── 3) empty tensor short-circuit (§5.4 HandleEmptyTensor) ────────────
    int32_t isRAxisEmpty = 0;
    int32_t isEmpty = HandleEmptyTensor(shapeBuf, rank, dimIdx, isRAxisEmpty);
    if (isEmpty) {
        // Branch-2 (Empty, tilingKey=2): route through ComputeBranch2Tiling so the
        // glue produces the same TilingData fields as the UT-tested seam.
        // softmax EMPTY_A / EMPTY_R both return same-shape empty tensor (§1.4/§5.6);
        // kernel early-exits.  ComputeBranch2Tiling fills EMPTY_A (usedCoreNum=0,
        // all fields 0) or EMPTY_R (chunk-split per §2, usedCoreNum≥1) fields.
        Branch2Inputs b2in{};
        for (int32_t i = 0; i < rank && i < MAX_RAW_RANK; ++i) b2in.shape[i] = shapeBuf[i];
        b2in.rank = rank;
        b2in.dim = dim;
        b2in.coreNum = static_cast<int32_t>(coreNum);
        b2in.ubSize = static_cast<int64_t>(ubSize);
        b2in.cacheLineSize = static_cast<int64_t>(cacheLineSize);
        b2in.sizeofDT = sizeofDT;
        // td was zero-init'd above; ComputeBranch2Tiling overwrites all fields.
        int32_t b2ret = ComputeBranch2Tiling(b2in, *td);
        if (b2ret != 0) {
            OP_LOGE("BenchHarnessSoftmax", "ComputeBranch2Tiling failed ret=%d", b2ret);
            return ge::GRAPH_FAILED;
        }
        context->SetTilingKey(static_cast<uint64_t>(EncodeTilingKey(/*templateType=*/0,
                                                                     /*isEmptyTensor=*/1,
                                                                     /*isTailR=*/0)));
        // EMPTY_A: usedCoreNum=0 → SetBlockDim(1) (framework requires ≥1).
        // EMPTY_R: usedCoreNum≥1 → SetBlockDim(usedCoreNum).
        context->SetBlockDim(static_cast<uint32_t>(td->usedCoreNum > 0 ? td->usedCoreNum : 1));
        size_t* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) ws[0] = 0;  // empty: no workspace
        return ge::GRAPH_SUCCESS;
    }
```

## 设计决策

- **为什么 EMPTY_R 仍按 chunk 公式填字段而不是像 EMPTY_A 一样全 0**：为框架一致性与可测性——TilingFunc glue 与被 UT 覆盖的 `ComputeBranch2Tiling` seam 产出同一字段集，UT 可以逐字段对 oracle 校验；kernel 在 softmax 语义下不消费这些字段（输出 0 元素）。这也是 glue 从"手写少数字段"改为"统一调 seam"的原因：两处字段集曾经不同步是潜在漂移源。
- **`N_post_alive=0 / N_out=1 / N_tmp=0` 是 softmax 特有取值**（无 postReduceInput、单输出、空 R 不执行 post-elewise）；通用归约（输出非空）三个参数不同，`bytes_per_chunk_unit` 随之改变。
- **`std::memset` 零初始化**：保证任何遗漏字段都是确定性 0（与 oracle 的期望一致），避免未初始化字段随编译器/平台漂移。
- **`SetBlockDim(max(usedCoreNum,1))`**：框架要求 blockDim >= 1；EMPTY_A 的 usedCoreNum=0 必须抬到 1，EMPTY_R 按切分核数设置。
- **放弃的方案**：复用 normal 分支 tiling 处理空 tensor——normal 按非空公式切分并固定申请 16KB cacheBuf，空分支下这些字段语义错误且浪费；全 0 方案对 EMPTY_R 会丢失框架一致性。
- 当前方案属 recommended_fastpath（空分支专用 seam + glue 统一路由）。

## 系统信息

- toolchain/平台：CANN 9.2.0，Ascend950PR（soc=Ascend950PR_9579）；算子仓 `operators/BenchHarnessSoftmax_package`，构建入口 `cd operators && ./build.sh BenchHarnessSoftmax_package`（退出码 0 判通过）。
- 相关文件（与本次读写一一对应）：
  - `op_host/bench_harness_softmax_tiling_host.cpp`：`ComputeBranch2Tiling` 实现与 `CeilDivI/CeilAlignI/MinI/MaxI` helper；
  - `op_host/bench_harness_softmax_tiling_host.h`：`Branch2Inputs`（`shape[MAX_RAW_RANK]/rank/dim/coreNum/ubSize/cacheLineSize/sizeofDT`）声明；
  - `op_host/arch35/bench_harness_softmax_tiling_arch35.cpp`：TilingFunc glue 空 tensor 短路（`HandleEmptyTensor` → `ComputeBranch2Tiling`）；
  - `tests/ut/test_branch2_tiling.cpp`：13 条 Branch2Tiling UT + 独立 oracle `OracleComputeBranch2`（不调用任何 `optiling::` 实现函数）。

## 完整代码

见上文「根因/解决配方」——`ComputeBranch2Tiling` 完整实现、helper 补充与 arch35 glue 短路 before/after 均为函数级 verbatim 代码块，取自本次开发会话的写入事件。

## 性能影响

无性能故事（空分支无可计时负载）；EMPTY_A 早退与 EMPTY_R 空转本身即性能最优形态。

## 预防

- 写空 tensor 分支前先确认算子的空输入语义：returns_empty（输出同 shape 空 tensor）还是填充值（输出非空）——两者 tiling/kernel 配方完全不同。
- TilingFunc glue 与被 UT 测试的 tiling seam 保持单一实现路径，避免两处字段集漂移。
- 空 tensor 用例（含 `[0,4]`、`[4,0]`、`[0,0]` 与高 rank 带 0 维）进 UT 与 ST 回归集。

## 验证信号

本配方在一次 softmax 类算子空 tensor 分支的测试先行开发中验证：先以桩实现跑出 13 条 tiling UT 红基线，落地公式后全量转绿，再经 aclnn ST 上板复验。

- 桩实现红基线：Branch2Tiling UT 13/13 失败（`act.usedCoreNum=0 vs exp=1`、`act.aUbFactor=0 vs exp=6`），59 条旧分支 UT 全过（来源平台：host 侧 gtest（x86_64，未上板）；证据等级：tool_observed）。
- 实现落地后：UT 72/72 全通过（13 条 branch-2 + 59 条 common/branch-0/branch-1 回归）（来源平台：host 侧 gtest（x86_64，未上板）；证据等级：tool_observed）。
- 构建退出码 0；构建日志无 "do not registe tiling struct" 告警（0 命中）（来源平台：host 构建；证据等级：tool_observed）。
- 上板 aclnn ST：Branch-2（Empty，tilingKey=2）24/24 precision PASSED、24 aclnn-OK、0 aclnn-FAIL；Branch-0 46/46、Branch-1 37/37 回归不退化；RUN_EXIT=0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 输出非空的归约算子（空 R 轴输出填 `empty_r_output_value`）：EMPTY_R 须走 Duplicate + CopyOut 填充路径，本配方 `N_out=1` 且"输出 0 元素"的假设不成立，`bytes_per_chunk_unit` 需按真实 `N_post_alive/N_tmp` 代入。
- 多输出、带 post-elewise、需要 workspace 的算子：`ws[0]=0` 与各 UB 尺寸全 0 的重映射须逐项重审。
- `SetBlockDim(0)` 的实际行为（报错 or 静默）未实测，本配方沿用 `max(usedCoreNum,1)` 防御（未验证，推断保留）。
- 64KB 单 buffer 上限与 4KB 下界常量源自 reduce-empty-tensor 参考约定，未在其他平台（950DT 等，ubSize/blockSize 不同）验证——未验证，推断可迁移。
- EMPTY_R 的 chunk 字段（aUbFactor/usedCoreNum 等）在 softmax 下 kernel 不消费，仅为框架/UT 一致性；若未来框架开始消费这些字段，须重新核对语义。

**未决项**：

- EMPTY_R chunk 字段在 softmax 下不被 kernel 消费，是否存在更简的统一空分支字段方案未评估。
- `SetBlockDim(0)` 的实际行为未做最小复现（报错 or 静默），建议下个会话以 1 分钟探针确认后回填本卡。
- 4KB 下界 / 64KB 上限常量未在本算子做参数扫描验证（当前仅按参考约定取值并通过 oracle 一致性 UT）。
## 坏实践

- 复用 normal 分支 tiling 公式处理空 tensor（按非空公式切分 + 固定 16KB cacheBuf，字段语义错误）。
- EMPTY_A 只置 `usedCoreNum=0` 而不清其余字段（残留 normal 值导致 kernel/框架读到不一致状态；正确做法是 memset 全清 + 显式覆盖关键项）。
- TilingFunc glue 手写一套与 UT seam 不同的字段集（两处漂移，UT 测不到真实 glue 产物）。

## 误诊

- oracle 独立性审计时用 `grep 'OracleComputeBranch2' -A 120` 这类固定行数窗口会把 TEST_P 测试体内合法的被测函数调用（`optiling::ComputeBranch2Tiling(in, act)`）误判为"oracle 依赖被测实现"（输出 `FOUND DEPENDENCY (BAD)` / `VIOLATION: oracle calls optiling:: function` 两次误报）；正确做法是用函数体范围限定（如 `awk '/^OracleTiling OracleComputeBranch2/,/^}/'`）只审 oracle 本体。此为本卡验证过程中的审计方法论注记（agent_reported 的误报处理过程，非 kernel 实测信号）。
