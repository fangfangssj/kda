---
type: "Runbook"
title: "csr2gebsr 双指针法窄矩阵退化"
description: "csr2gebsr Convert 用双指针法替代 BinarySearchCsrRow，窄矩阵严重退化、大矩阵改善，回退禁用。"
tags: ["task.optimization", "topic.compute.indexing", "topic.data_movement.memory_access", "topic.programming_model.execution_model.simt", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-18T08:26:43Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["L2_cache"]
---

# csr2gebsr 双指针法窄矩阵退化

## 现象

在 Ascend950（arch35/DAV_3510）上为 ops-sparse 仓 PR #91 的 csr2gebsr 算子（CSR→GEBSR 格式转换，SIMT 编程模型）做性能优化时，将 Convert kernel（Kernel 3）中按块内 (inRow, inCol) 逐元素调用 `BinarySearchCsrRow` 做 O(log nnz) 随机 GM 读的 `FillBlockValues`，替换为双指针法 `FillBlockValuesDualPointer`——对每个 inRow 维护游标 `j`（初始为 `csrRowPtrA[r]-baseA`），跨所有非零 colBlock 顺序推进。编译通过、功能测试通过后跑 `Csr2GebsrPerfTest.PerfSweep`，性能数据出现明显分化：大矩阵（nb 大）显著改善，但窄矩阵（nb 小，如 10000×10）相对原 BinarySearch 路线退化约 489×。模式清晰：nb 大改善、nb 小退化。随后回退双指针法、仅保留 P4+P11，窄矩阵恢复至噪声范围，退化消除。

## 适用场景

- **工程形态**：SIMT kernel 直调（ops-sparse 仓 arch35 csr2gebsr，非 aclnn/spec 流水线）。
- **API 路径**：`__simt_vf__`/`__simt_callee__` 计算函数内 CSR 行扫描；Convert kernel Pass2 块值填充。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510/arch35），CANN 9.0.0-beta.2（x86_64-linux 模拟/上板）。
- **适用粒度**：`operator_family`（适用于 csr2gebsr 同族 CSR→Block 格式转换、且 Convert 阶段对 CSR 行做列块查找的 SIMT 算子）。
- **路线优先级**：`banned`——双指针法替代 BinarySearch 作为通用 Convert 块值填充路线被禁用。
- **不适用或未验证场景**：窄矩阵（nb 小）是本路线的失效边界，不得用于 nb 小的场景；a2/a3 未验证，推断可迁移但未实测。

## 失败路线

尝试：把 `FillBlockValues`（逐 (inRow, inCol) 调 `BinarySearchCsrRow` 做 O(log nnz) 二分查找）替换为 `FillBlockValuesDualPointer`（对每个 inRow 维护游标 `j`，跨所有非零 colBlock 顺序推进，利用 CSR 行内列索引升序 + colBlock 递增使 j 只前进不回退），期望将总读次数从每 colBlock 重新二分查找降为每 inRow 只线性扫描一次 CSR 行。

banned 路线代码（`FillBlockValuesDualPointer`，verbatim，取自 trace git diff 事件流）：

```cpp
/// P013: 块值填充独立函数（双指针法，j 游标跨 colBlock 传递）。
///
/// 对每个 inRow，j 游标从 csrRowPtrA[r] 初始化一次，跨所有非零 colBlock 顺序推进。
/// 利用 CSR 行内列索引升序 + colBlock 递增特性（targetCol 单调递增），j 只前进不回退，
/// 将总读次数从 nbz×rowBlockDim×nnz_per_row（每 colBlock 重新扫描 CSR 行）降为
/// rowBlockDim×nnz_per_row（每 inRow 只扫描一次 CSR 行）。
/// P3:  双指针法替代原 BinarySearchCsrRow，O(nnz) 顺序扫描替代 O(nnz×log) 随机读。
/// P006/G.EXP.26: blockOffset 先转型再运算，避免 (blockStart+blkIdx) int32 溢出。
/// CX-4: j<0 防御，防止 csrColIndA[j] 负索引越界读。
template <typename VAL_T>
__simt_callee__ __aicore__ inline void FillBlockValuesDualPointer(
    __gm__ const VAL_T *csrValA, __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA, __gm__ VAL_T *bsrValC,
    __gm__ const int32_t *marker, int32_t rowStart,
    int32_t m, int32_t n, int32_t rowBlockDim, int32_t colBlockDim,
    int32_t minColBlock, int32_t maxColBlock, int32_t actualBr,
    int64_t markerBase, int32_t blockStart, int32_t baseA, int32_t dir,
    int64_t blockSize)
{
    for (int32_t inRow = 0; inRow < rowBlockDim; inRow++) {
        int32_t r = rowStart + inRow;
        int32_t j = 0;
        int32_t jEnd = 0;
        if (r < m) {
            j = csrRowPtrA[r] - baseA;
            if (j < 0) { j = 0; }
            jEnd = csrRowPtrA[r + 1] - baseA;
        }
        int32_t blkIdx = 0;
        for (int32_t colBlock = minColBlock; colBlock <= maxColBlock; colBlock++) {
            if (marker[markerBase + colBlock] != actualBr) {
                continue;
            }
            int64_t blockOffset = (static_cast<int64_t>(blockStart) + static_cast<int64_t>(blkIdx)) * blockSize;
            for (int32_t inCol = 0; inCol < colBlockDim; inCol++) {
                int32_t targetCol = colBlock * colBlockDim + inCol;
                VAL_T value = static_cast<VAL_T>(0);
                if (r < m && targetCol < n) {
                    while (j < jEnd && (csrColIndA[j] - baseA) < targetCol) {
                        j++;
                    }
                    if (j < jEnd && (csrColIndA[j] - baseA) == targetCol) {
                        value = csrValA[j];
                    }
                }
                int32_t writeIdx = (dir == 0) ? inRow * colBlockDim + inCol : inCol * rowBlockDim + inRow;
                bsrValC[blockOffset + writeIdx] = value;
            }
            blkIdx++;
        }
    }
}
```

参照路线（原 `BinarySearchCsrRow` + `FillBlockValues`，verbatim）：

```cpp
/// 二分查找辅助函数：在 CSR 行 r 中查找列索引 targetCol
/// 若找到返回对应值，否则返回 0
template <typename VAL_T>
__simt_callee__ __aicore__ inline VAL_T BinarySearchCsrRow(
    __gm__ const VAL_T *csrValA,
    __gm__ const int32_t *csrColIndA,
    int32_t rowStart, int32_t rowEnd,
    int32_t targetCol, int32_t baseA)
{
    // CX-4: rowStart<0 防御，防止 csrColIndA[mid] 负索引越界读（P004 半截修复补全）
    if (rowStart < 0) { rowStart = 0; }
    int32_t lo = rowStart;
    int32_t hi = rowEnd - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        int32_t midCol = csrColIndA[mid] - baseA;
        if (midCol == targetCol) {
            return csrValA[mid];
        }
        if (midCol < targetCol) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return static_cast<VAL_T>(0);
}

/// P013: 块值填充独立函数（从 Csr2gebsrConvertSimtCompute Pass2 内层提取）。
/// 遍历块内 (inRow, inCol)，按方向 layout 写入 bsrValC[blockOffset + writeIdx]。
/// 降低原函数嵌套深度（5→3）与 NBNC（51→~36）。
/// P006: blockOffset 用 int64_t 承载，避免 (blockStart+blockIdx)*blockSize 在
///       nnzb*blockSize > INT32_MAX 时溢出为负导致 bsrValC 越界写。
///       writeIdx 保持 int32_t（块内偏移，max = rowBlockDim*colBlockDim，小值）。
template <typename VAL_T>
__simt_callee__ __aicore__ inline void FillBlockValues(
    __gm__ const VAL_T *csrValA,
    __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA,
    __gm__ VAL_T *bsrValC,
    int32_t rowStart, int32_t m, int32_t n,
    int32_t rowBlockDim, int32_t colBlockDim,
    int32_t colBlock, int32_t baseA, int32_t dir,
    int64_t blockOffset)
{
    for (int32_t inRow = 0; inRow < rowBlockDim; inRow++) {
        int32_t r = rowStart + inRow;
        for (int32_t inCol = 0; inCol < colBlockDim; inCol++) {
            int32_t col = colBlock * colBlockDim + inCol;
            VAL_T value = static_cast<VAL_T>(0);
            if (r < m && col < n) {
                int32_t pA = csrRowPtrA[r] - baseA;
                // CX-4: pA<0 防御，防止 BinarySearchCsrRow 负索引越界读
                if (pA < 0) { pA = 0; }
                int32_t qA = csrRowPtrA[r + 1] - baseA;
                value = BinarySearchCsrRow(
                    csrValA, csrColIndA, pA, qA, col, baseA);
            }
            int32_t writeIdx;
            if (dir == 0) {  // ROW
                writeIdx = inRow * colBlockDim + inCol;
            } else {  // COLUMN
                writeIdx = inCol * rowBlockDim + inRow;
            }
            bsrValC[blockOffset + writeIdx] = value;
        }
    }
}
```

客观失败（来源平台：950；证据等级：tool_observed）：

- 10000×10 窄矩阵双指针路线相对原 BinarySearch 路线退化约 489×（严重退化）。
- 1000×1000 大矩阵双指针路线相对原 BinarySearch 路线改善约 1.9×（FP32）、约 5.2×（FP16）。
- 27/27 功能测试通过（双指针路线功能正确，仅性能退化）。

边界与回退：双指针法 banned；回退到原 BinarySearch + `FillBlockValues` 参照路线，并保留 P4（缩小扫描范围至 [minColBlock,maxColBlock]）+ P11（blockOffset 先 static_cast<int64_t> 再运算）。回退后窄矩阵恢复至噪声范围，退化消除。

根因（机制链，不跳步）：双指针法在窄矩阵（nb 小）退化的机制——原 BinarySearch 路线对每个 (inRow, inCol) 做 O(log nnz_per_row) 随机 GM 读，但 CSR 行内列索引高度局部化，二分查找的随机访问被 L2 cache 高效吸收（L2 命中率引用前序分析，本 trace 未单独复测，仅作机制解释、不作验证信号）。双指针法把每个 inRow 的 j 游标跨所有非零 colBlock 顺序推进，总读次数从每 colBlock 重新二分查找变为每 inRow 线性扫描一次 CSR 行：当 nb 大（列块多）时双指针省去重复扫描、显著收益；当 nb 小（窄矩阵，如 10000×10 的 nb=5）时，双指针的线性扫描反而比已 cache 命中的二分查找读更多 GM、退化约 489×。即"用 O(n) 顺序扫描替代 O(log n) 随机读"在 cache 友好场景下是负优化。

## 验证信号

本验证信号来自 csr2gebsr 算子 `csr2gebsr_test --gtest_filter=Csr2GebsrPerfTest.PerfSweep` 在 Ascend950 上的 [PERF] stdout（同一算子内双指针路线 vs 原 BinarySearch 路线的成对相对对比；绝对单跑时延只进 manifest provenance，不写正文）。

- 10000×10 场景双指针路线相对原 BinarySearch 路线退化约 489×（来源平台：950；证据等级：tool_observed）
- 1000×1000 FP32 双指针路线相对原 BinarySearch 路线改善约 1.9×；FP16 约 5.2×（来源平台：950；证据等级：tool_observed）
- 回退 P3、仅保留 P4+P11 后 10000×10 恢复至噪声范围（与原始基线同量级），退化消除（来源平台：950；证据等级：tool_observed）
- 27/27 功能测试通过（双指针路线功能正确，仅性能退化）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 窄矩阵（nb 小，如 10000×10）为本路线严重退化边界，禁用双指针路线。
- 大矩阵（nb 大）双指针有收益，但本 trace 最终采用"回退双指针、保留 P4+P11"的安全路线，因窄矩阵退化不可接受且无 shape 自适应切换。
- a2/a3 平台未验证，推断可迁移（同 SIMT/arch 编程模型，cache 行为类似），但未实测。

## 坏实践

- 用"单组大矩阵 perf 数据"宣布双指针法为通用优化——会掩盖窄矩阵退化，正是本 trace 一度犯的错误（先看到 1000×1000 改善即采用，跑 PerfSweep 全场景后才暴露 10000×10 退化 489×）。
- 主会话最初把大矩阵退化归因于 P5 核数分配公式；后续隔离测试表明 P5 非元凶，真正根因是 P3 双指针在窄/小矩阵的退化。窄矩阵 10000×10 退化 489× 才是决定性信号。