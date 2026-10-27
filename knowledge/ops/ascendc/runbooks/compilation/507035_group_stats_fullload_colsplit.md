---
type: "Runbook"
title: "507035 全载超 UB：大 groupSize 分载三遍扫描"
description: "全载模式把整个 group 搬入 UB，单 group 元素数超过阈值（8192×fp32 缓冲数）后 UB 需求超 Ascend950 约 192KB 容量，必然 507035；tiling 按阈值自动切分载（colsplit）模式，kernel 分 chunk 三遍扫描（mean/var/normalize+affine 各一遍），UB 需求与 groupSize 解耦。"
tags: [task.implementation, topic.tiling, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# 507035 全载超 UB：大 groupSize 分载三遍扫描

## 现象

在 Ascend950PR（CANN 9.1.0）上开发 BenchNoHarnessGroupNorm（GroupNorm 类算子，fp32）。骨架阶段只实现了"全载"模式：整个 group（C/numGroups × 空间元素数）一次性搬入 UB 完成 mean/var/归一化/仿射。ST Real 测试中，5 个大 groupSize 用例稳定崩溃：

```
[L0_025] 4D N=1 C=3 ng=1 eps=1e-05 dtype=float32   (groupSize = 3*224*224 = 150528)
  [FAIL] 流同步失败: 507035
[L0_027] 4D N=1 C=6 ng=1 ... (groupSize = 6*512*512 = 1572864)
  [FAIL] 流同步失败: 507035
```

按全载缓冲账本（x/y/work/tmp 等多份 `groupSizeAlign * sizeof(float)`）估算：L0_035 需约 384KB、L0_025 需约 1764KB，而 Ascend950 的典型 UB 容量约 192KB（196608 字节；同批另一卡的会话估值为 256 KiB，两口径均为会话估算、超容结论在两值下均成立，见 [fp32 tile UB 卡](507035_float32_tile_ub.md)）——单 group 数据本身就放不下，无论怎么压缩缓冲份数都超容。小 groupSize（≤8192）用例全部正常。

## 适用场景

- 工程形态：AscendC kernel 开发，算子需要对"一组数据"做统计（mean/var）再逐元素变换，组数据可能很大。
- API 路径：op_host tiling（`FULLLOAD_THRESHOLD` 判定、TilingKey/modeKey 设置、`GetTilingData` 填充）+ op_kernel 模板分发（`if constexpr (MODE == 0)`）+ `pipe.InitBuffer` UB 分配。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"组内统计+逐元素变换"类算子（GroupNorm/LayerNorm/InstanceNorm/RMSNorm 等）。
- 路线优先级：recommended_fastpath（大 group 场景的正确性必需路径）。
- 明确不适用 / 未验证场景：group 本身可整块放入 UB 的场景（全载更简单高效）；其他平台的 UB 容量与阈值需重算。

## 根因

全载模式的 UB 需求随单 group 元素数线性增长（多份 `groupSizeAlign*4B` 缓冲），groupSize 超过阈值后超出 UB 硬件容量，`InitBuffer` 后的访问越界/失败表现为向量核异常 507035。这类算子的正确解法不是压缩缓冲，而是换数据通路：**分载（按 chunk 搬入）+ 多遍扫描**——mean、var、normalize+affine 各扫一遍，每遍只要求单个 chunk（固定 `COLSPLIT_CHUNK_SIZE`）驻留 UB，UB 需求与 groupSize 解耦。

## 解决配方

过渡期防御（tiling 显式拒绝，防止 kernel 崩溃；同时 ST 侧跳过该类用例并注明待分载支持）：

```cpp
    // 6. 判定全载/分载
    bool isFullLoad = (groupSizeAlign <= FULLLOAD_THRESHOLD);
    // 过渡期: 仅支持 fullload 模式, 大 groupSize 拒绝
    OP_CHECK_IF(!isFullLoad,
        OP_LOGE(context, "groupSizeAlign=%lu > FULLLOAD_THRESHOLD=%u, colsplit mode not supported in iter1",
                groupSizeAlign, FULLLOAD_THRESHOLD),
        return ge::GRAPH_FAILED);
```

最终实现（三处配合）：

1）tiling 自动切换 + 分载参数（op_host/arch35/bench_no_harness_group_norm_tiling.cpp）：

```cpp
    // 6. 判定全载/分载
    bool isFullLoad = (groupSizeAlign <= FULLLOAD_THRESHOLD);

    // 7. 多核切分 (按 N * numGroups 切分)
    uint32_t blockDim = std::min(static_cast<uint32_t>(totalGroups), coreNum);
    uint64_t groupsPerCore = totalGroups / blockDim;
    uint64_t tailGroups = totalGroups % blockDim;

    // 8. 计算 ReduceSum 临时 buffer 大小 + 分载参数
    uint32_t dtypeKey = 0; // 该阶段仅 fp32
    uint32_t modeKey = isFullLoad ? 0 : 1;
    uint64_t chunkSize = 0;
    uint64_t chunkSizeAlign = 0;
    uint32_t numChunks = 0;
    uint32_t lastChunkSize = 0;
    if (isFullLoad) {
        // 全载模式: chunk 参数不使用, reduceLen = groupSizeAlign
    } else {
        // 分载模式: 计算 chunk 参数
        chunkSize = std::min(static_cast<uint64_t>(groupSize),
                              static_cast<uint64_t>(COLSPLIT_CHUNK_SIZE));
        chunkSizeAlign = AlignUp(chunkSize, elemPerBlock);
        numChunks = static_cast<uint32_t>((groupSize + chunkSize - 1) / chunkSize);
        lastChunkSize = static_cast<uint32_t>(groupSize - (numChunks - 1) * chunkSize);
    }
    uint32_t reduceLen = isFullLoad ? static_cast<uint32_t>(groupSizeAlign)
                                     : static_cast<uint32_t>(chunkSizeAlign);
    uint32_t tmpBufSize = ComputeReduceTmpBufSize(reduceLen, calcTypeSize);

    // 9. 填充 TilingData（totalGroups/groupSize/spatialSize/blockDim/.../chunk 参数）
    BenchNoHarnessGroupNormTilingData* tiling = context->GetTilingData<BenchNoHarnessGroupNormTilingData>();
    ...
    tiling->chunkSize = chunkSize;
    tiling->chunkSizeAlign = chunkSizeAlign;
    tiling->numChunks = numChunks;
    tiling->lastChunkSize = lastChunkSize;
```

2）kernel 入口按 MODE 分发（op_kernel/bench_no_harness_group_norm_arch35.cpp）：

```cpp
    if constexpr (MODE == 0) {
        // 全载模式: 整个 group 一次性搬入 UB
        NsBenchNoHarnessGroupNorm::KernelFullLoad<DTYPE, MODE> op;
        op.Init(x, gamma, beta, y, workspace, &tilingData);
        op.Process();
    } else {
        // 分载模式: 分 chunk 搬入, 三遍扫描 (mean → var → normalize+affine)
        NsBenchNoHarnessGroupNorm::KernelColSplit<DTYPE, MODE> op;
        op.Init(x, gamma, beta, y, workspace, &tilingData);
        op.Process();
    }
```

3）分载 kernel：UB 分配只按 chunk 大小，三遍扫描（op_kernel/arch35/bench_no_harness_group_norm.h，核心结构）：

```cpp
// 三遍扫描策略:
//   遍历1: GM(x) → UB(chunk) → ReduceSum → 标量累加 → mean
//   遍历2: GM(x) → UB(chunk) → Adds(-mean) → Mul(平方) → ReduceSum → 标量累加 → var → rstd
//   遍历3: GM(gamma,beta) → UB; GM(x) → UB(chunk) → Adds(-mean) → Muls(rstd)
//         → 仿射(标量) → GM(y)
template <uint32_t DTYPE, uint32_t MODE>
class KernelColSplit {
    ...
    // UB Buffer 分配: 基于 chunkSizeAlign (分载模式核心区别)
    uint32_t chunkBytes = static_cast<uint32_t>((chunkSizeAlign_ + FP32_ELEM_PER_BLOCK) * sizeof(float));
    pipe.InitBuffer(xQueue, 1, chunkBytes);
    pipe.InitBuffer(yQueue, 1, chunkBytes);
    ...
    // 遍历 1: 计算 mean（逐 chunk ReduceSum + 标量累加）
    float sum = 0.0f;
    for (uint32_t chunkIdx = 0; chunkIdx < numChunks_; ++chunkIdx) {
        uint64_t chunkOffset = static_cast<uint64_t>(chunkIdx) * chunkSize_;
        uint32_t count = (chunkIdx == numChunks_ - 1) ? lastChunkSize_ : static_cast<uint32_t>(chunkSize_);
        LocalTensor<float> xLocal = xQueue.template AllocTensor<float>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[gmOffset + chunkOffset], copyParams, padParams);
        xQueue.EnQue(xLocal);
        xLocal = xQueue.template DeQue<float>();
        ReduceSum<float>(meanVarLocal, xLocal, tmpLocal, static_cast<int32_t>(count));
        sum += meanVarLocal.GetValue(0);
        xQueue.FreeTensor(xLocal);
    }
    float mean = sum * invGroupSize_;
    // 遍历 2/3 同构: var 累加得 rstd, 再逐 chunk 中心化+归一化+仿射写回
```

## 设计决策

- 为什么三遍扫描而不是"两遍"（Welford 单遍）：三遍实现最直白（mean/var 各一遍、变换写回一遍），GM 重复读两遍的带宽代价在正确性优先阶段可接受；Welford 单遍融合省带宽但数值与控制流更复杂，且 var 公式与 golden（两遍公式）不一致会引入对齐 golden 的麻烦。
- 为什么 chunk 固定上限（`COLSPLIT_CHUNK_SIZE`）而不是按 UB 动态求解：固定 chunk 让 tiling 参数（numChunks/lastChunkSize）与 ReduceSum tmpBuf 规模稳定，UT 期望值可枚举；UB 账本只须验证"chunk + gamma/beta + tmp"不超容一次。
- 为什么保留全载模式：小 groupSize 时全载单遍完成，避免三遍扫描的重复搬运——tiling 按 `FULLLOAD_THRESHOLD`（8192）自动切换，两条路线各取所长。
- 多核切分按 `totalGroups = N * numGroups` 分核（blockDim = min(totalGroups, coreNum)），组间天然无依赖，无需跨核同步。
- 当前方案路线档位：recommended_fastpath（大 groupSize）/ 全载为小 groupSize 的 fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（arch35），CANN 9.1.0；UB 约 192KB。
- 相关文件：`op_host/arch35/bench_no_harness_group_norm_tiling.cpp`（判定/参数/TilingData）、`op_kernel/bench_no_harness_group_norm_arch35.cpp`（MODE 分发）、`op_kernel/arch35/bench_no_harness_group_norm.h`（KernelColSplit 三遍扫描）、`tests/st/test_aclnn_bench_no_harness_group_norm.cpp`（跳过逻辑随实现移除）、`tests/ut/op_host/test_bench_no_harness_group_norm_tiling.cpp`（分载 tiling 用例）。

## 预防

- 设计"组内统计"类 kernel 先算最坏 UB 账本：`单 group 字节数 × 驻留缓冲份数 ≤ UB 容量`；不满足就预先规划分载路径，不要等 507035 再改架构。
- tiling 层对不支持的形态显式 `GRAPH_FAILED` + 明确日志，不要让 kernel 带着超容参数上板崩溃。

## 验证信号

以下信号来自该算子的开发/联调会话：分载实现合入并重装后，全量回归。

- 修复前：5 个大 groupSize 用例（groupSize 32768~1572864，UB 需求约 384KB~1764KB > 192KB）稳定 `流同步失败: 507035`；小 groupSize 用例正常（来源平台：950；证据等级：tool_observed）。
- 过渡防御：tiling 显式拒绝后 ST 以 `[SKIP] groupSize=... > 8192 (需 colsplit 模式)` 跳过，执行用例 35/35 通过、0 失败（来源平台：950；证据等级：tool_observed）。
- 最终实现：ST 跳过逻辑移除后 L0 全量 41/41 通过、0 跳过（原先 6 个跳过用例含 5 个大 groupSize 全部转为通过）；新增分载专项 25 用例 25/25 通过；UT 26/26 通过（来源平台：950；证据等级：tool_observed）。
- 分载大 group 精度：groupSize=150528 用例 MERE 1.20e-07 / MARE 5.34e-05（阈值 1.22e-04）通过；L0 全量 MERE 最大 1.89e-07（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `FULLLOAD_THRESHOLD=8192` 与 `COLSPLIT_CHUNK_SIZE` 按本算子 fp32 缓冲账本标定；换 dtype（fp16/bf16 走 fp32 中间计算时缓冲仍按 fp32 账）或增减驻留缓冲份数须重算阈值（机制可迁移，数值未逐一验证）。
- 三遍扫描对 GM 的重复读取在大 group 下有带宽代价；未做与"两遍/Welford"方案的成对性能对比（无实测数据，不构成性能结论）。
- 192KB UB 容量为该平台典型值；其他 NPU 架构容量不同（未验证，推断机制可迁移）。

**未决项**：

- `COLSPLIT_CHUNK_SIZE` 的最优值（UB 占用 vs chunk 次数）未做扫描实验；当前值正确性验证充分、性能未调优。
## 坏实践

- 全载模式硬扛大 group，试图靠"压缩缓冲份数"救 1764KB 的需求——单 group 数据本身就超容。
- 不做 tiling 层形态判定，让超容参数直接下发 kernel（表现为难排查的 507035）。

## 误诊

- 初期把这 5 个用例与"对齐类 507035"混为一谈：对齐根因修复后这 5 个用例仍失败，失败集收敛到"大 groupSize"特征，UB 账本计算（384KB/1764KB vs 192KB）才把两者区分开（对齐类根因见独立卡：向量算子操作数偏移/count 非 32B 对齐触发 aivec 340）。
