---
type: "Runbook"
title: "noTrans extract 与 NDExtLayoutPtn 方向不匹配"
description: "aclblasStrsm blocked 路径 noTrans misaligned 尝试 extract A/B 到对齐 workspace 再走 GEMM(transA=0)，extract 行主序存储与 blocked GEMM 的 NDExtLayoutPtn 读取方向不一致致精度大错（outliers 占比 48%~69%）；trans path 靠 transA=1 转置存储恰好匹配而工作；noTrans misaligned 一律回退 SIMT，extract+GEMM 路线 banned。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.layout_transform"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["StrsmExtractBSimt", "row_major", "column_major"]
---

# noTrans extract 与 NDExtLayoutPtn 方向不匹配

## 现象

为修复 noTrans 路径 `lda`/`ldb` 非 8 对齐致 Nd2Nz 读偏的潜伏 bug，尝试给 `StrsmTrailingUpdateNoTrans` 加 `misaligned` 分支：当 `(lda%8) || (ldb%8) || (bs%8)` 时，把 A 的 L21 块（`mC×bs`）与 B 的 X1 块（`bs×n`）extract 到对齐 workspace（stride `CeilAlign` 到 8），再走 GEMM。`extract_a` 用 `transA=0`（noTrans 语义），`extract_b` 同 trans path。新增 4 条 blocked+misaligned 用例（TC_L2_050~053）验证，结果 noTrans extract 路线精度大错：

- TC_L2_051（noTrans, m=200, n=256, lda=201, ldb=256, bs 尾对齐）：`FAILED (MERE < threshold && MARE < 10.000000*threshold, 31754 outliers, 0 mismatches out of 65536 elements)`。
- TC_L2_053（noTrans, m=500, n=512, lda=501, ldb=512）：`FAILED (... 181177 outliers, 0 mismatches out of 262144 elements)`。
- TC_L2_050（noTrans, m=300, ldb=256<300）：`ret=3`（后被确认为 `ldb<m` 的 `ACLBLAS_STATUS_INVALID_VALUE`，非本路线精度问题）。
- TC_L2_052（trans, m=300, ldb=257<300）：`ret=3`（同上 `ldb<m`）。

关键对照：trans path 的 `extract_a` 用 `transA=1`（转置存储），同批 TC_L2_054（trans, m=256, n=300, n%8≠0）走 trans padded path 通过（`0 outliers`）。即 trans path 工作，noTrans extract 不工作。

## 适用场景

- **工程形态**：aclblas 算子仓开发，blocked 路径 GEMM 用 Nd2Nz + `NDExtLayoutPtn` 读取 workspace。
- **API 路径**：`StrsmTrailingUpdateNoTrans`（misaligned 分支）→ `strsm_extract_a/b_kernel_do`（`transA=0`）→ `strsm_gemm_kernel`（`NDExtLayoutPtn` 读取 wsA/wsB）。
- **验证平台与版本**：Ascend950（`blas/trsm/arch35/`，`DAV_3510`→`950`）；ops-blas `aclblastrsm` 分支。失败用例 TC_L2_051/053 tool_observed，回退后 119/119 PASSED。
- **适用粒度**：`operator_family`——凡用 `NDExtLayoutPtn` 读 workspace 的 GEMM 都有此存储/读取方向约束。
- **路线优先级**：`banned`——noTrans misaligned 走 extract+GEMM（`transA=0`）方向不匹配，已回退；正确做法是 noTrans misaligned 一律回退 `StrsmLeftSimtPath`（纯标量 GM 读，无 Nd2Nz/`NDExtLayoutPtn`，无方向约束）。
- **不适用或未验证场景**：`NDExtLayoutPtn` 的实际读取方向（列主序）是 agent 从“trans path `transA=1` 工作、noTrans `transA=0` 不工作”反推的推断，未直接探针验证（见 `## 失效边界`）。

## 失败路线

**尝试**：noTrans misaligned → extract A/B 到对齐 workspace（`aWsStride=bsAligned`、`bWsStride=nAligned`，`CeilAlign(.., 8)`）→ GEMM（`transA=0`，行主序存储 `wsGm[i*aWsStride + l]`）。

**失败**：TC_L2_051/053 精度大错（outliers 占比 48%~69%，且 0 mismatches——不是个别越界而是整体方向错位）。trans path 同机制（extract + GEMM + `NDExtLayoutPtn`）工作正常，两者唯一差异是 `transA`（0 vs 1）→ 存储方向是否转置 → 是否匹配 `NDExtLayoutPtn` 读取方向。机制判定（agent 解释，未直接探针）：`StrsmExtractASimt` 把 A 的 L21 块按行主序写入 workspace `wsGm[i*aWsStride + l]`（`i` 是行、`l` 是列），blocked GEMM 的 `strsm_gemm_kernel` 通过 `NDExtLayoutPtn` 从 wsA/wsB 搬运到 L1，读取方向（推断为列主序）与 extract 的行主序存储方向不一致；trans path 用 `transA=1`（`srcIdx = aOffset + l + i*lda`，转置读取 A）存储仍 `wsGm[i*aWsStride + l]`，相当于把 A 转置后按行主序存，恰好与列主序读取匹配，故 trans path 工作。noTrans `transA=0` 不做转置，行主序存储与列主序读取方向冲突。

**回退**：删除 `StrsmTrailingUpdateNoTrans` 的 `misaligned` 分支，noTrans 一律走对齐直读；misaligned 时由 `aclblasStrsm` 路由守卫（`blockedGemmUnsafe = bsTailMisalign || (!isTrans && strideMisalign)`）回退 `StrsmLeftSimtPath`（纯标量 GM 读，无 Nd2Nz/`NDExtLayoutPtn`，无方向约束）。本路线对“noTrans misaligned 走 extract+GEMM”判 `banned`。

before（被回退的 noTrans extract 分支，verbatim 取自修复前 file_write）：

```cpp
static void StrsmTrailingUpdateNoTrans(_aclblas_handle* h, uint32_t nU32, uint32_t k, uint32_t bs,
    uint32_t mC, uint64_t aOffset, uint64_t axpyOffset, uint32_t ldaU32, uint32_t ldbU32,
    uint32_t noTransTempStride, uint32_t nAligned, uint32_t aicCoreNum, uint32_t aivCoreNum,
    const float* A, float* B, uint8_t* tempDev, uint8_t* aWsDev, uint8_t* bWsDev)
{
    bool misaligned = (ldaU32 % 8u) || (ldbU32 % 8u) || (bs % 8u);
    if (misaligned) {
        uint32_t bsAligned = CeilAlign<uint32_t>(bs, 8u);
        strsm_extract_b_kernel_do(reinterpret_cast<uint8_t*>(B), bWsDev,
            bs, nU32, nAligned, ldbU32, k, CalcAuxNumBlocks(static_cast<uint64_t>(bs) * nU32, aivCoreNum), h->stream);
        strsm_extract_a_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(A)), aWsDev,
            mC, bs, bsAligned, ldaU32, aOffset, 0,
            CalcAuxNumBlocks(static_cast<uint64_t>(mC) * bs, aivCoreNum), h->stream);
        StrsmGemmTilingData gt = CalcGemmTiling(nU32, mC, bs, nAligned, bsAligned, 0, 0,
            noTransTempStride, aicCoreNum);
        uint64_t gemmTileCount = static_cast<uint64_t>((nU32 + gt.tileM - 1) / gt.tileM)
                               * static_cast<uint64_t>((mC + gt.tileN - 1) / gt.tileN);
        uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
        if (gemmBlocks == 0) gemmBlocks = 1;
        strsm_gemm_kernel_do(bWsDev, aWsDev, tempDev, gt, gemmBlocks, h->stream);
    } else {
        StrsmGemmTilingData gt = CalcGemmTiling(nU32, mC, bs, ldbU32, ldaU32, k, aOffset,
            noTransTempStride, aicCoreNum);
        uint64_t gemmTileCount = static_cast<uint64_t>((nU32 + gt.tileM - 1) / gt.tileM)
                               * static_cast<uint64_t>((mC + gt.tileN - 1) / gt.tileN);
        uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
        if (gemmBlocks == 0) gemmBlocks = 1;
        strsm_gemm_kernel_do(reinterpret_cast<uint8_t*>(B), reinterpret_cast<uint8_t*>(const_cast<float*>(A)),
            tempDev, gt, gemmBlocks, h->stream);
    }
    StrsmAxpyTilingData at{};
    at.m = mC;
    at.n = nU32;
    at.ldb = ldbU32;
    at.tempRowStride = noTransTempStride;
    at.bOffset = axpyOffset;
    strsm_axpy_trans_kernel_do(reinterpret_cast<uint8_t*>(B), tempDev, at,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * nU32, aivCoreNum), h->stream);
}
```

after（回退为对齐直读，misaligned 由路由守卫在 `aclblasStrsm` 层回退 SIMT，verbatim 取自修复后 file_write）：

```cpp
static void StrsmTrailingUpdateNoTrans(_aclblas_handle* h, uint32_t nU32, uint32_t k, uint32_t bs,
    uint32_t mC, uint64_t aOffset, uint64_t axpyOffset, uint32_t ldaU32, uint32_t ldbU32,
    uint32_t noTransTempStride, uint32_t aicCoreNum, uint32_t aivCoreNum,
    const float* A, float* B, uint8_t* tempDev)
{
    StrsmGemmTilingData gt = CalcGemmTiling(nU32, mC, bs, ldbU32, ldaU32, k, aOffset,
        noTransTempStride, aicCoreNum);
    uint64_t gemmTileCount = static_cast<uint64_t>((nU32 + gt.tileM - 1) / gt.tileM)
                           * static_cast<uint64_t>((mC + gt.tileN - 1) / gt.tileN);
    uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
    if (gemmBlocks == 0) gemmBlocks = 1;
    strsm_gemm_kernel_do(reinterpret_cast<uint8_t*>(B), reinterpret_cast<uint8_t*>(const_cast<float*>(A)),
        tempDev, gt, gemmBlocks, h->stream);
    StrsmAxpyTilingData at{};
    at.m = mC;
    at.n = nU32;
    at.ldb = ldbU32;
    at.tempRowStride = noTransTempStride;
    at.bOffset = axpyOffset;
    strsm_axpy_trans_kernel_do(reinterpret_cast<uint8_t*>(B), tempDev, at,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * nU32, aivCoreNum), h->stream);
}
```

## 验证信号

- TC_L2_051 `FAILED (31754 outliers, 0 mismatches out of 65536 elements)`、TC_L2_053 `FAILED (181177 outliers, 0 mismatches out of 262144 elements)`——noTrans extract 路线精度大错（来源平台：950；证据等级：tool_observed）。
- 对照 TC_L2_054（trans padded, `transA=1`）`PASSED (0 outliers, 0 mismatches out of 90000 elements)`——trans path 同机制工作（来源平台：950；证据等级：tool_observed）。
- 回退 + 路由守卫 + 修正 ldb 后 TC_L2_051/053 走 SIMT 兑底 `PASSED (0 mismatches)`、最终 `[ PASSED ] 119 tests.`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 平台：`blas/trsm/arch35/`（`DAV_3510`→Ascend950→`950`）；其它架构（a2/a3）的 `NDExtLayoutPtn` 读取方向是否一致未验证，推断可迁移（方向约束是 Nd2Nz 布局语义，跨架构应一致，但未在新架构确认）。
- `NDExtLayoutPtn` 列主序读取方向是 agent 推断（从 trans/noTrans 对照反推），未直接探针验证；若实际读取方向不同，本卡机制判定需修正。
- fallback SIMT 路线对大 m 的绝对延迟未测，本卡只证 noTrans extract+GEMM 路线 banned，不证 SIMT 路线性能占优。
