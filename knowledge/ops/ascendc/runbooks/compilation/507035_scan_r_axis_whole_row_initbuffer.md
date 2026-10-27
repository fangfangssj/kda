---
type: "Runbook"
title: 507035 AR-scan整行UB分配超248KB崩溃
description: AR-scan 按整行 R 一次性 InitBuffer，R 大时 UB 超 248KB 报 507035 向量核异常；tiling 增加 tileR 分块并跨 chunk 维护扫描状态。
tags: [task.debugging, topic.memory, topic.tiling, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 507035 AR-scan整行UB分配超248KB崩溃

## 现象

Cummin 算子 NPU L0 测试中，L0_019 用例（bfloat16，shape=[15951628]，dim=-1，即 1D 输入沿最后一维扫描，AR-scan 模式：A0=1、R=1595万）反复失败：

```
[Real] Test: L0_019_bf16_1d_bf16_range
  [FAIL] Stream sync failed: 507035
```

当时 kernel 的 AR-scan 实现（`op_kernel/arch35/cummin.h` 的 `ProcessArScan()`）按整行一次性分配 UB 缓冲：

```cpp
// Load entire row into UB
LocalTensor<T> inputBuf = arInputBuf.template AllocTensor<T>();
DataCopyExtParams copyParams;
copyParams.blockCount = 1;
copyParams.blockLen = R * sizeof(T);
copyParams.srcStride = 0;
copyParams.dstStride = 0;
copyParams.rsv = 0;
DataCopyPad(inputBuf, xGM[baseOffset], copyParams, {false, 0, 0, static_cast<T>(0)});

LocalTensor<T> valBuf = arValuesBuf.template AllocTensor<T>();
LocalTensor<int64_t> idxBuf = arIndicesBuf.template AllocTensor<int64_t>();
```

（对应 `pipe.InitBuffer(arInputBuf, R * sizeof(T)); pipe.InitBuffer(arValuesBuf, R * sizeof(T)); pipe.InitBuffer(arIndicesBuf, R * sizeof(int64_t));` 的整行分配。）

排查期间先误判为"kernel 执行超时"，把测试的 Real 模式元素裁剪上限从 10M 逐级降到 5M、1M、100K，507035 在每个规模都复现——与规模弱相关的恒定失败推翻了超时假设。随后查 CANN 头文件：

```
$ grep -r "507035\|0x7BDAB" $ASCEND_HOME_PATH.../include/
include/acl/error_codes/rt_error_codes.h:#define  ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      507035 // vector core exception
```

507035 = 向量核异常（kernel 崩溃），不是超时。按 R=100000、bfloat16（sizeof(T)=2）计算 UB 占用：input 200KB + values 200KB + indices 800KB = 1.2MB，远超单核 UnifiedBuffer 约 248KB 的容量，`InitBuffer` 超额导致 kernel 加载/执行时向量核异常。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 内做顺序扫描/前缀类计算（cummin/cummax/scan）且扫描轴 R 可能很大（10^5 以上）的算子。
- API 路径：`TPipe::InitBuffer`/`TBuf::AllocTensor`（UB 分配）、`DataCopyPad`（GM↔UB 搬运，blockLen 单位为字节）、`TilingData` 字段下发、`rt_error_codes.h` 错误码。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR 实测）+ CANN 9.1.0。
- 适用粒度：`operator_family`（所有按行整块分配 UB 的扫描类 kernel）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：R 可控且小于 UB 预算的小规模输入（无需分块）；多核沿 R 并行的方案（顺序扫描语义上不可行，除非做跨核规约，本轨迹未尝试）。

## 根因/解决配方

根因：AR-scan 模式按整行大小 `R * (2*sizeof(T) + sizeof(int64_t))` 字节一次性 `InitBuffer`，R 大时远超约 248KB 的单核 UB 容量，运行时报 507035（ACL_ERROR_RT_VECTOR_CORE_EXCEPTION）。修复分三层：

**1. TilingData 增加分块字段**（`op_kernel/arch35/cummin_tiling_data.h`）：

```cpp
    uint64_t tileA0Aligned = 0;  // tileA0 aligned size
    uint64_t tileR = 0;          // AR-scan chunk size along R axis
    uint64_t a1PerCore = 0;      // A1 count per core
```

**2. tiling 按 UB 预算计算 tileR**（`op_host/arch35/cummin_tiling.cpp`，tileA0Aligned 计算之后、多核切分之前）：

```cpp
    uint64_t tileA0Aligned = ((tileA0 + alignElements - 1) / alignElements) * alignElements;

    // 5b. AR-scan tileR: chunk size for R axis to fit UB
    // UB budget per chunk: inputBuf(T) + valuesBuf(T) + indicesBuf(int64)
    // = tileR * (2*typeSize + sizeof(int64_t))
    uint64_t arPerElement = 2 * typeSize + sizeof(int64_t);
    uint64_t effectiveUbAr = ubSize * 3 / 4;  // reserve 25% for pipe overhead
    uint64_t maxTileR = effectiveUbAr / arPerElement;
    if (maxTileR < 1) {
        maxTileR = 1;
    }
    uint64_t tileR = (rSize < maxTileR) ? rSize : maxTileR;

    // 6. Multi-core splitting (along A1)
```

并在 tiling 赋值处加 `tiling->tileR = tileR;`。

**3. kernel 分 chunk 扫描，跨 chunk 维护运行状态**（`ProcessArScan()` 重写，要点 verbatim）：

```cpp
template <typename T>
__aicore__ inline void KernelCummin<T>::ProcessArScan()
{
    uint64_t R = tiling_->rSize;
    uint64_t tileR = tiling_->tileR;

    // Allocate UB buffers for one chunk (tileR elements, not full R)
    pipe.InitBuffer(arInputBuf, tileR * sizeof(T));
    pipe.InitBuffer(arValuesBuf, tileR * sizeof(T));
    pipe.InitBuffer(arIndicesBuf, tileR * sizeof(int64_t));

    for (uint64_t a1 = a1Start_; a1 < a1End_; a1++) {
        uint64_t baseOffset = a1 * R;  // A0=1, so offset = a1 * R

        // Running state maintained across chunks
        float runningMinF = 0.0f;
        int64_t runningIdx = 0;
        bool nanSeen = false;
        bool initialized = false;

        for (uint64_t rStart = 0; rStart < R; rStart += tileR) {
            uint64_t chunkLen = tileR;
            if (rStart + chunkLen > R) {
                chunkLen = R - rStart;
            }

            // Load chunk from GM to UB
            LocalTensor<T> inputBuf = arInputBuf.template AllocTensor<T>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = chunkLen * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPad(inputBuf, xGM[baseOffset + rStart], copyParams, {false, 0, 0, static_cast<T>(0)});

            LocalTensor<T> valBuf = arValuesBuf.template AllocTensor<T>();
            LocalTensor<int64_t> idxBuf = arIndicesBuf.template AllocTensor<int64_t>();

            PipeBarrier<PIPE_ALL>();

            // Scalar scan within this chunk (逐元素维护 runningMinF/runningIdx/nanSeen，
            // 首 chunk 首元素初始化 initialized，NaN 位级判定，
            // curF < runningMinF 更新；状态变量在 chunk 循环外声明，跨 chunk 生效)
            ...
        }
    }
}
```

关键机制：顺序扫描的最小值/索引/NaN 状态是三个标量，把它们的声明提到 chunk 循环之外，每个 chunk 只搬运 `tileR` 个元素、扫描后分块写回，语义与整行扫描等价。UB 预算按 75% 容量预留（`ubSize * 3 / 4`），给 PipeBarrier 等管道开销留余量；`DataCopyPad` 的 `blockLen` 是字节数（`chunkLen * sizeof(T)`），不是元素数。

同码异因的相邻卡（同错误码须按缓冲路径分型）：固定 tile 过大/tiling 未按物理宽度换算见 [507035 三队列 float32 tile 超 UB 卡](507035_float32_tile_ub.md)与 [UB 预算按对齐物理宽度卡](ub_budget_uses_aligned_physical_width_507035.md)；冗余 work buffer 推爆 UB 见 [Reduce work buf 额外占 UB 卡](reduce_work_buf_ub_507035.md)；workspace 保留区 MTE DDR 越界见 [507035 MTE DDR 卡](../runtime/mte_ddr_507035_workspace_reserved_region.md)；A0 轴（非扫描轴）整块分配的同型回归见同批《507035 ARA-scan大A0整块UB分配超限》卡——A0 分块的状态无需跨 tile，与本卡 R 轴方向相反。

## 设计决策

- 为什么 tileR 在 host tiling 层计算而不是 kernel 内算：UB 容量与 dtype 相关，host 侧有 `ubSize`/`typeSize` 完整信息，tileR 经 TilingData 下发后 kernel 只管分块循环，职责清晰。
- 为什么留 25% 余量：UB 同时承载队列/管道元数据，按 100% 预算分配有越界风险；75% 是保守取值（fp32: tileR ≈ 248KB×0.75/16B ≈ 11904；bf16/fp16: ≈ 15872）。
- 放弃的候选：多核沿 R 并行 + 跨核规约（前缀最小值可两段式并行，但复杂度高且索引语义要保持 first-occurrence，本轨迹未做）；继续压小测试规模（误诊产物，见误诊）。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0；构建 `bash build.sh --soc=ascend950 -j8`；安装 `./build/custom_opp_ubuntu_x86_64.run --install-path=$HOME/Ascend/opp --force`。
- 目标平台：Ascend 950（npu-smi 显示 Ascend950PR）。
- 相关文件：`op_kernel/arch35/cummin_tiling_data.h`（tileR 字段）、`op_host/arch35/cummin_tiling.cpp`（tileR 计算）、`op_kernel/arch35/cummin.h`（ProcessArScan 分块循环）、`tests/st/test_aclnn_cummin.cpp`（L0_019 用例与 Real 模式裁剪）。

## 验证信号

验证来自同一开发会话的 NPU 实测：tileR 分块落地后，此前恒败的 AR-scan 大 R 用例（bfloat16 1D、dim=-1，测试裁剪至 100000 元素）PASS，MERE=0.00e+00、全部元素匹配（来源平台：950；证据等级：tool_observed）；同一包回归全量 L0 ST 21/21 通过、无回归（来源平台：950；证据等级：tool_observed）。修复前同用例在 10M/5M/1M/100K 各裁剪规模下均报 `Stream sync failed: 507035`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 验证覆盖的最大 R 为 100000（bf16，测试侧裁剪后）；原始 15951628 全量规模未在修复后复跑，属未验证场景（分块机制上无规模上限，但未实测）。
- tileR 预算基于 248KB UB 与 75% 余量假设；不同 soc 的 UB 容量不同，需按 tiling 层 ubSize 重算。
- 顺序扫描的性能瓶颈（标量逐元素）未优化，大 R 的时延代价未测量。

**未决项**：

- 原始 15951628 全量 shape 的端到端验证（测试裁剪上限放开后复跑）。
- 大 R 顺序扫描的性能优化（向量化的前缀最小值分段方案）未尝试。
## 坏实践

- 按输入整行/整轴大小 `InitBuffer`——UB 容量与输入规模无关，必须经 tiling 换算成块大小。
- `DataCopyPad` 的 blockLen 传元素个数——该参数单位是字节，传错会静默搬运错误长度。

## 误诊

- 首诊"kernel 执行超时"（agent_reported）：依据是大 shape + 失败，把测试裁剪上限从 10M 逐级降到 100K 均无效后才查错误码定义；507035 是 ACL_ERROR_RT_VECTOR_CORE_EXCEPTION（崩溃），与规模弱相关的恒定失败应优先怀疑越界/超额而不是超时。此误诊消耗了四轮无效的测试参数调整。

## 复现增量：argmax 的两种"整行口径"失配（rAlign 口径混用 / tiling 桩未封顶）

另一独立轨迹并入同一 argmax 算子两个迭代里的同族实例——分块算法的 UB 分配/预算却按"整行（rAlign）"口径，与本卡"整行 R 一次性分配"同一根因的两种变体：

**变体一：kernel/tiling 分配口径混用（rAlign vs ubFactorR）**。kernel 明明分块处理（ubFactorR 为每块元素数），但 tiling 的 `tmpBufSize = rAlign * sizeof(float)`（R=65536 时 256KB，单独超过 ~192KB UB 总量）、kernel `InitBuffer(inQueueX/floatBuf, ..., rAlign_ * sizeof)` 同样按整行分配。判别：失败用例的 R 与通过用例的 R 二分——R 数千全过、R=65536 必崩，且与 dtype 无关（fp16 大 R 同崩），即可收敛到"随 R 线性增长的资源分配"。修复：

```cpp
            // tmpBufSize for ReduceMax(calIndex=true) sharedTmpBuffer.
            // Use fixed 8KB: ReduceMax Level2 API handles large counts internally.
            // Using rAlignBytes directly causes UB overflow when R > ~48K elements.
            tiling->tmpBufSize = static_cast<uint32_t>(CeilAlign(MIN_TMP_BUF_SIZE, 32U));
```

```cpp
    // kernel：按分块 ubFactorR_ 分配，不按 rAlign_（整行）
    if constexpr (NeedsCastToFloat<D_T_X>::value) {
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactorR_ * sizeof(D_T_X));
        pipe.InitBuffer(floatBuf, ubFactorR_ * sizeof(float));
    } else {
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactorR_ * sizeof(float));
    }
```

要点：共享临时缓冲（ReduceMax sharedTmpBuffer）优先按 API 文档最小值固定（8KB 实测有效），不要按输入规模放大；曾试 32KB 封顶仍残留超额风险。连带：tmpBufSize 固定后原先 R=32768 的 UT 用例不再触发 ColSplit，单测改用 R=65536 保持分支覆盖——改 tiling 预算口径时同步核对依赖具体数值的单测期望。

**变体二：tiling 桩未按 UB 预算封顶**。为新 dtype 路径（int64-AR）先写占位桩：`ubFactorR = min(r, rAlign)≈r`，等价于"整条 reduction 轴一次装进 UB"。R=1048576 时 kernel `InitBuffer(inQueueX, 1, 1048576*8) = 8MB`，远超 ~192KB UB，`BND_LARGE_001` 运行期报 507035；小 R 用例（R=8→64B、R=512→4KB）天然落在 UB 内，桩"看起来能用"——编译期零告警，运行期错误码离根因（tiling 参数）很远，是这类桩缺陷的隐蔽性所在。修复（与 Float/Int32-AR 同构的量纲公式：overhead 扣减 → 每元素字节数 → 封顶 → 对齐取整）：

```cpp
        } else if (isInt64Class) {
            // Int64-AR: scalar int64 comparison, only inQueueX(int64) + outBuf needed.
            uint32_t int64AlignEl = 32 / dtypeSize;  // 4 for int64
            tiling->rAlign = static_cast<uint32_t>(CeilAlign(r, static_cast<uint64_t>(int64AlignEl)));
            tiling->tmpBufSize = MIN_TMP_BUF_SIZE;

            uint64_t overhead = UB_OVERHEAD + tiling->tmpBufSize;
            uint64_t perElementBytes = bufNum * dtypeSize;  // bufNum * 8
            uint64_t availableForInput = (ubSize > overhead) ? (ubSize - overhead) : 0;
            uint64_t maxUbFactorR = availableForInput / perElementBytes;
            uint32_t ubFactorR = static_cast<uint32_t>(
                FloorAlign(std::min(maxUbFactorR, static_cast<uint64_t>(r)),
                           static_cast<uint64_t>(int64AlignEl)));
            ubFactorR = (ubFactorR == 0) ? int64AlignEl : ubFactorR;
            tiling->rAlign = ubFactorR;
            tiling->ubFactorR = ubFactorR;
        }
```

`int64AlignEl = 32/dtypeSize`（=4）：int64 元素 8 字节，DataCopyPad/UB 访问按 32B 块对齐，最小完整块含 4 个 int64；`ubFactorR == 0` 兜底为对齐粒度避免除零/死循环。507035 出现时先查 tiling 产物的内存量纲（`ubFactorR × dtypeSize × bufNum + 固定缓冲 vs UB`），再查 kernel。

**独立 trace 新增验证信号**（同 trace 两个迭代，均为 tool_observed）：

- 变体一：修复前 `shape=[65536] fp16 dim=0` 用例 `流同步失败: 507035`；tiling 推导 `tmpBufSize = rAlign(65536) * 4 = 256KB` 单项已超 ~192KB UB；tmpBufSize 固定 8KB + 输入缓冲按 ubFactorR_ 后大 R 用例（65536 一维 fp16、大 2D）全部 PASS（bitwise-equal），ST 升至 91/101，最终 101/101；UT ColSplit 用例改 R=65536 后 54/54（来源平台：950；证据等级：tool_observed）。
- 变体二：修复前 `[Real] BND_LARGE_001: Large 1D int64 (1048576) ... [FAIL] 流同步失败: 507035`（回退 DoubleBuffer 后仍单独失败，根因隔离）；补预算计算后同轮 ST 159 条通过 157（98.74%），BND_LARGE_001 恢复 `[PASS] bitwise-equal`（剩余 2 条为已知 int64 框架路由问题，与本根因无关）（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界**：8KB sharedTmpBuffer 对 ReduceMax(calIndex=true) 有效；其他 Level 2 接口的 tmpBuffer 需求换接口时按其文档重查。预算口径统一为"凡算法按块处理，分配一律按单块大小 × buffer 深度，禁止拿整行/整轴大小做参数"；DoubleBuffer（buffer 深度 2）场景下预算需按深度重新计算（bufNum 因子一致性另有专卡）。