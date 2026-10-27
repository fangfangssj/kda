---
type: "Runbook"
title: 507035 ARA-scan大A0整块UB分配超限
description: ARA-scan 标量重写版按整 A0 分配 UB，A0≈4.3万时报 507035 向量核异常；恢复 tileA0Aligned 分块循环后 2M 元素用例通过。
tags: [task.debugging, topic.memory, topic.tiling]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# 507035 ARA-scan大A0整块UB分配超限

## 现象

Cummin 算子 ARA-scan 模式（扫描维非最后一维，A0>1）从向量 Compare/Select 流水线重写为逐 A0 标量处理后，L0_008 用例（float32，shape=[48,105,23,1,2,1,1,9]，dim=-8）失败：

```
[Real] Test: L0_008_fp32_8d_small_range
  shape=[48,105,23,1,2,1,1,9], dtype=float32, dim=-8
  [FAIL] Stream sync failed: 507035
```

该 shape 归一化后 A1=1、R=48（dim 轴）、A0=105×23×1×2×1×1×9=43470。重写后的标量版 `ProcessAraScan()`（`op_kernel/arch35/cummin.h`）按整 A0 分配 UB：

```cpp
    // Initialize UB buffers for scalar processing
    // inputBuf: one row of A0 aligned elements
    pipe.InitBuffer(arInputBuf, a0Aligned * sizeof(T));
    // valuesBuf: one row of A0 aligned elements for output
    pipe.InitBuffer(arValuesBuf, a0Aligned * sizeof(T));
    // runningMinBuf: running minimum per A0 position (type T)
    pipe.InitBuffer(runningMinBuf, a0Aligned * sizeof(T));
    // runningIdxBuf: running index per A0 position (float for precision)
    pipe.InitBuffer(runningIdxBuf, a0Aligned * sizeof(float));
    // idxInt64Buf: int64 buffer for indices output
    pipe.InitBuffer(idxInt64Buf, a0Aligned * sizeof(int64_t));
```

fp32 下每元素合计 4+4+4+4+8=24 字节，A0=43470（对齐后 43472）约 1.0MB，远超单核约 248KB 的 UB 容量——与 AR-scan 整行分配同型的 507035（ACL_ERROR_RT_VECTOR_CORE_EXCEPTION）崩溃。

值得注意：重写前的向量版 `ProcessAraScan()` 本来带有 `tileA0Aligned` 分块循环（`a0TileCount = (a0Aligned + tileA0Aligned - 1) / tileA0Aligned`），标量重写时把这个资源约束处理丢掉了。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 内沿非最后一维做扫描/规约、A0（dim 后的展开内积）可能很大（10^4 以上）的算子。
- API 路径：`TPipe::InitBuffer`/`TBuf`（UB 分配）、`TilingData` 的 tileA0Aligned 字段、`DataCopyPad` 分块搬运。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR 实测）+ CANN 9.1.0。
- 适用粒度：`operator_family`。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：A0 小于 UB 预算的输入（无需分块）；A1 维多核切分与本卡正交。

## 根因/解决配方

根因：标量重写版 `ProcessAraScan()` 按整 A0（`a0Aligned`）一次性 InitBuffer 五个 UB 缓冲，A0≈4.3万、fp32 时约 1.0MB 超出约 248KB UB，运行时向量核异常（507035）。修复：把重写前已有的 `tileA0Aligned` 分块循环加回标量版本——缓冲按 tile 大小分配，外层循环遍历 A0 分块，块内逻辑不变：

```cpp
template <typename T>
__aicore__ inline void KernelCummin<T>::ProcessAraScan()
{
    uint64_t a0Size = tiling_->a0Size;
    uint64_t R = tiling_->rSize;
    uint64_t tileA0Aligned = tiling_->tileA0Aligned;
    uint64_t a0Aligned = tiling_->a0Aligned;

    // Number of A0 tiles
    uint64_t a0TileCount = (a0Aligned + tileA0Aligned - 1) / tileA0Aligned;

    // Initialize UB buffers sized for one tile (not full A0)
    pipe.InitBuffer(arInputBuf, tileA0Aligned * sizeof(T));
    pipe.InitBuffer(arValuesBuf, tileA0Aligned * sizeof(T));
    pipe.InitBuffer(runningMinBuf, tileA0Aligned * sizeof(T));
    pipe.InitBuffer(runningIdxBuf, tileA0Aligned * sizeof(float));
    pipe.InitBuffer(idxInt64Buf, tileA0Aligned * sizeof(int64_t));

    for (uint64_t a1 = a1Start_; a1 < a1End_; a1++) {
        for (uint64_t a0Tile = 0; a0Tile < a0TileCount; a0Tile++) {
            uint64_t a0Start = a0Tile * tileA0Aligned;
            if (a0Start >= a0Size) break;
            uint64_t tileCount = tileA0Aligned;
            if (a0Start + tileCount > a0Size) {
                tileCount = a0Size - a0Start;
            }

            // Allocate persistent buffers for this tile
            LocalTensor<T> runMinT = runningMinBuf.template AllocTensor<T>();
            LocalTensor<float> runIdxF = runningIdxBuf.template AllocTensor<float>();
            LocalTensor<int64_t> idxI64 = idxInt64Buf.template AllocTensor<int64_t>();

            for (uint64_t r = 0; r < R; r++) {
                // Load current tile: x[a1, r, a0Start:a0Start+tileCount]
                LocalTensor<T> inputBuf = arInputBuf.template AllocTensor<T>();
                uint64_t srcOffset = GetOffset(a1, r, a0Start);
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = tileCount * sizeof(T);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPad(inputBuf, xGM[srcOffset], copyParams, {false, 0, 0, static_cast<T>(0)});

                LocalTensor<T> valBuf = arValuesBuf.template AllocTensor<T>();

                PipeBarrier<PIPE_ALL>();

                if (r == 0) {
                    // First row: initialize running_min and running_idx
                    for (uint64_t a0 = 0; a0 < tileCount; a0++) {
                        runMinT.SetValue(a0, inputBuf.GetValue(a0));
                        runIdxF.SetValue(a0, 0.0f);
                        valBuf.SetValue(a0, inputBuf.GetValue(a0));
                        idxI64.SetValue(a0, static_cast<int64_t>(0));
                    }
                } else {
                    // Subsequent rows: scalar compare and update
                    // （int32 原生比较分支 / float 域 NaN-aware 分支）
                    ...
                }
                ...  // 分块写回 values/indices（blockLen = tileCount * sizeof(...)，字节单位）
            }
        }
    }
}
```

要点：running 状态缓冲（runMinT/runIdxF/idxI64）按 A0 分块分配，每个 tile 独立初始化（r=0 行），因为同一 a0 列的扫描序列在固定 a0Tile 内闭合，跨 tile 无需携带状态——这与 AR-scan 的 R 轴分块（状态必须跨 chunk）方向不同（R 轴整行分配的 507035 见同批《507035 AR-scan整行UB分配超248KB崩溃》卡）。同码其他根因的分型判别另见 [507035 三队列 float32 tile 超 UB 卡](507035_float32_tile_ub.md)、[Reduce work buf 额外占 UB 卡](reduce_work_buf_ub_507035.md)与 [507035 MTE DDR 卡](../runtime/mte_ddr_507035_workspace_reserved_region.md)。

## 设计决策

- 为什么 A0 分块不需要跨 tile 状态而 R 分块需要：cummin 的扫描轴是 R，A0 的每个位置是独立的扫描序列；分块沿 A0 切的是"多少条序列一起算"，序列内状态在 R 循环内自然维护。
- 教训（本卡核心）：整体重写函数时，先盘点原实现里的资源约束处理（对齐、分块、边界），重写版必须逐项继承——本例标量重写丢掉了向量版的 tileA0Aligned 分块，直接引入回归。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` + run 包安装。
- 目标平台：Ascend 950。
- 相关文件：`op_kernel/arch35/cummin.h`（ProcessAraScan）、`op_host/arch35/cummin_tiling.cpp`（tileA0Aligned 计算，本卡沿用已有字段）、`tests/st/test_aclnn_cummin.cpp`（L0_008 用例）。

## 验证信号

验证来自同一开发会话的 NPU 实测：A0 分块恢复后，L0_008（fp32 8D，dim=-8，全量 2086560 元素，未裁剪）values MERE=0.00e+00、MARE=0.00e+00，indices 全部匹配（来源平台：950；证据等级：tool_observed）；同一包全量 L0 ST 21/21 通过、无回归（来源平台：950；证据等级：tool_observed）。修复前同用例稳定复现 `[FAIL] Stream sync failed: 507035`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 验证覆盖的最大 A0 为 43470（fp32，2086560 总元素）；更大的 A0 依赖 tiling 层 tileA0Aligned 预算正确性，未实测。
- UB 预算继承 tiling 层既有 tileA0Aligned 计算；不同 soc 需重算。
- 与 507035 同码不同因：本卡是 A0 维整块分配，AR-scan 整行 R 分配见同批另一卡——同错误码须按维度/缓冲路径分型定位。

**未决项**：

- tileA0Aligned 预算的边界余量与 AR-scan tileR 的 75% 策略是否统一（当前两处预算策略独立演化），可合并为单一 UB 预算工具。
## 坏实践

- 重写函数时不盘点原实现的分块/对齐/边界处理——本例把向量版的 A0 tiling 丢掉引入回归。
- 按展开后内积 A0 全量分配 UB——A0 由 shape 决定，与 UB 容量无关。

## 误诊

- 该失败初现时曾与 OOM/进程 Killed 混淆（同一轮测试里 L0_010 因设备内存压力被系统 Killed）——Stream sync failed 507035 是 kernel 级向量核异常，进程 Killed 是主机侧 OOM，两者机制不同，需分开归因（agent_reported 的混淆在错误码查询后澄清）。
