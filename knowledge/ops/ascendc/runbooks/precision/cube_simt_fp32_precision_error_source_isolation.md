---
type: "Runbook"
title: "稀疏反卷积精度误差源隔离配方"
description: "用 K=1/inChannels=1/单核/SIMT 线程=1 退化测试与 kIdx=0 DumpTensor 对比，逐层隔离 Cube MatMul 与 SIMT ScatterAdd 的 FP32 误差贡献。"
tags: ["task.debugging", "topic.precision", "topic.programming_model.execution_model.simt", "topic.programming_model.execution_unit.cube", "topic.implementation.debugging"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# 稀疏反卷积精度误差源隔离配方

## 现象

调试 `cloud_sparse_inverse_conv3d_with_matmul` 的 ATK 精度失败时，误差源混在一起：Cube MatMul 的 FP32 累加、SIMT `asc_atomic_add` 的跨 warp 竞争、`loopPointCount` 批切分、CPU BLAS 的 FP64 内部累加，四者都会贡献数值差异。需要一个系统化配方逐层隔离，定位每一层的贡献量级。本卡记录一套在 trace 内实测验证过的隔离配方与 dump-tag 探针方案，最终把误差量级钉死在 Cube MatMul 的约 4 ULP FP32 累加差异上。

## 适用场景

- **工程形态**：kernel 直调（稀疏卷积 CSR 解码 + Cube MatMul AIC + SIMT ScatterAdd AIV 的混合架构）。
- **API 路径**：AscendC `DumpTensor`（kernel 内 dump 到 result.csv）；`Simt::VF_CALL<ScatterAddSimt>`；Cube `MatMul` API；`loopPointCount_`/`ubMaxTaskNum_` tiling 字段。
- **验证过的平台与 CANN 版本**：Ascend950，CANN 9.2.0（arch35=ascend950）。
- **适用粒度**：`operator_family`（任何 Cube+SIMT 混合 FP32 稀疏算子的精度误差源隔离）。
- **路线优先级**：`recommended_fastpath`（精度失败时先做隔离定位根因层，再决定修复路线）。
- **不适用或未验证场景**：纯 Cube 算子（无 ScatterAdd，无需隔离 SIMT 层）；FP16/BF16（ULP 与累加路径不同）；Atlas A2/A3 未实测。

## 系统信息

- **工具链**：CANN 9.2.0，DrivingSDK（`customize_arch35=ascend950`），AscendC kernel + DumpTensor。
- **目标平台**：Ascend950（950）。
- **目录布局**：算子源在 `/mnt/workspace/gitCode/cann/cloud-ops-turbo/driving/cloud_sparse_inverse_conv3d_with_matmul/`（`op_kernel/*.h`、`op_kernel/*.cpp`、`op_host/*`）；CPU golden 与 dump 分析在 `/mnt/workspace/gitCode/cann/inverse_demo.py`、`compare_kidx0.py`、`result.csv`。
- **相关文件清单**（与事件流实际读写一一对应）：
  - `op_kernel/cloud_sparse_inverse_conv3d_with_matmul.h` — `Process` AIV 批切分循环、`OutFeaturesScatterAdd`、`CalOutFeaturesMatmul`
  - `op_kernel/cloud_sparse_inverse_conv3d_with_matmul.cpp` — kernel 入口
  - `sparse_conv_common/op_kernel/sparse_conv_kernel_common.h` — `ScatterAddSimt`（共享）
  - `inverse_demo.py` — CPU golden（`sparse_inverse_conv3d_cpu`，加 `loop_point_count` 参数）
  - `result.csv` — DumpTensor 输出（NPU tag=210/220/232/331 + CPU `k = N` 分段）

## 完整实现

（完整实现单元；逐字代码见下 `## 完整代码`。）

## 完整代码

### 隔离配方（四层退化测试 + 一层直接对比）

按隔离强度从强到弱执行，每步缩小误差源范围：

1. **K=1 退化**：`kernel_size=[1,1,1]`，无 kIdx 循环，`loopPoint` 设大到不分批、单核。整个算子退化为「一次 MatMul + 一次 ScatterAdd」，唯一差异源是 Cube vs CPU 的内积。
2. **inChannels=1 标量退化**：MatMul 退化为逐元素乘法，无内积累加。若此时 NPU vs CPU 通过 ATK → 反证问题在多通道内积累加。
3. **SIMT 线程数=1**：把 `THREAD_NUM/outFeatures_` 的 yBlockDim 设为 1（单 yId 线程），退化为严格 taskIdx 递增，消除跨 warp 仲裁。
4. **单核（usedVectorNum=1）**：消除核间 atomic_add 竞争。若仍失败 → 核间竞争非主导。
5. **kIdx=0 DumpTensor 直接对比**：在 kernel 内对 kIdx=0 dump Cube MatMul 的输入与输出，与 CPU 同 kIdx 的 partial 逐元素对比，直接量化 Cube 累加差异。

### Dump-tag 探针方案（verbatim，仅在 kIdx=0 触发）

| Tag | 位置 | 类型 | 数据 | 说明 |
|---|---|---|---|---|
| 210 | `CalMatmul` (AIC) | float32 | 2048 | 特征输入（matmul Tensor A），`featureWsp_` GM→UB |
| 220 | `CalMatmul` (AIC) | float32 | 2048 | 权重输入（matmul Tensor B），`weightGM_` GM→UB |
| 232 | `CalMatmul` (AIC) | float32 | 2048 | matmul 输出（前 8 entries × 256 inChannels） |
| 331 | `ProcessAIV` (AIV) | int32 | totalSparseNum | scatter 索引 |

tag=210 的 2048 floats = 8 entries × 256 inChannels（共 44 entries，截断到前 8 个）；tag=220 的 2048 floats = weight 矩阵前 2048 个元素（共 256×192=49152）；重新编译运行后 `result.csv` 中会多出 tag=210/220 数据，可与 CPU 侧 features 与 weight 逐一对比。

### 被诊断的系统代码（verbatim，取自 file_read 事件）

`ScatterAddSimt`（`sparse_conv_kernel_common.h:45-61`）：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterAddSimt(
    __gm__ float* outFeaturesGmPtr, __ubuf__ float* partialOutPtr, __ubuf__ int32_t* outputIdxPtr, int32_t sparseM,
    uint32_t outChannel)
{
    int32_t xId = threadIdx.x;
    int32_t yId = threadIdx.y;
    int32_t yBlockDim = blockDim.y;

    for (int32_t taskIdx = yId; taskIdx < sparseM; taskIdx += yBlockDim) {
        int32_t outputIdx = outputIdxPtr[taskIdx];

        int32_t gmOffset = outputIdx * outChannel + xId;
        int32_t ubOffset = taskIdx * outChannel + xId;

        asc_atomic_add(outFeaturesGmPtr + gmOffset, partialOutPtr[ubOffset]);
    }
}
```

`OutFeaturesScatterAdd` 内层 `ubMaxTaskNum_` 子批切分（`cloud_sparse_inverse_conv3d_with_matmul.h`）：

```cpp
__aicore__ inline void SparseInverseConv3dWithMatmulKernel::OutFeaturesScatterAdd(uint32_t m)
{
    if (m <= 0) {
        return;
    }
    SetFlag<HardEvent::V_MTE2>(1);

    for (int32_t idxM = 0; idxM < m; idxM += ubMaxTaskNum_) {
        uint32_t loopCount = min(ubMaxTaskNum_, m - idxM);

        WaitFlag<HardEvent::V_MTE2>(1);
        DataCopy(tmpOutFeatureLocal_, tempOutFeatureWsp_[idxM * outChannels_], loopCount * outChannels_);

        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);

        Simt::VF_CALL<ScatterAddSimt>(
            Simt::Dim3{outChannels_, THREAD_NUM / outChannels_}, (__gm__ float*)outFeaturesGM_.GetPhyAddr(),
            (__ubuf__ float*)tmpOutFeatureLocal_.GetPhyAddr(), (__ubuf__ int32_t*)outputIdxPtrLocal_[idxM].GetPhyAddr(),
            loopCount, outChannels_);

        SetFlag<HardEvent::V_MTE2>(1);
    }
    WaitFlag<HardEvent::V_MTE2>(1);
}
```

外层 `Process` AIV 批切分循环（同文件）：

```cpp
if ASCEND_IS_AIV {
    Simt::VF_CALL<PrepareGlobalHashMap>(
        Simt::Dim3{THREAD_NUM}, (__gm__ int32_t*)uniqueIndicesOffsetGM_.GetPhyAddr(),
        (__gm__ int32_t*)sortedIdxToFormerIndicesGM_.GetPhyAddr(), (__gm__ int32_t*)inputIdxWsp_.GetPhyAddr(),
        (__gm__ int32_t*)outputIdxWsp_.GetPhyAddr(), (__gm__ uint8_t*)kIdxWsp_.GetPhyAddr(), totalTaskNum_,
        kernelSize_, startOffset_);
    SyncAll();

    for (int32_t pointIdx = blockIdx_ * loopPointCount_; pointIdx < totalPointsCount_;
         pointIdx += loopPointCount_ * usedVectorNum_) {
        int32_t ubPointCount = min(totalPointsCount_ - pointIdx, (int32_t)(loopPointCount_));

        CopyInHashMap(pointIdx, ubPointCount);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        uint32_t maskAlign = AlignUp(ubPointCount, INT8_PER_LOOP);
        uint32_t bitLoopCount = DivCeil(ubPointCount, BITS_PER_UINT64);

        for (uint8_t kIdx = 0; kIdx < kernelSize_; ++kIdx) {
            uint16_t flagId = kIdx % CROSS_CORE_FLAG_SLOTS;

            Compares(maskLocal_, kIdxLocal_, kIdx, CMPMODE::EQ, maskAlign);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);

            int32_t totalSparseNum = 0;
            CopyInFeature(bitLoopCount, ubPointCount, totalSparseNum);

            idxInfoLocal_.SetValue(0, totalSparseNum);
            SetFlag<HardEvent::S_MTE3>(flagId);
            WaitFlag<HardEvent::S_MTE3>(flagId);
            DataCopyPad(
                sparseNumWsp_[blockIdx_ * kernelSize_ + kIdx], idxInfoLocal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(INT32_SIZE), 0, 0, 0});

            CrossCoreSetFlag<0x4, PIPE_MTE3>(flagId);

            CrossCoreWaitFlag<0x4>(flagId);

            OutFeaturesScatterAdd(totalSparseNum);
        }
    }
}
```

## 设计决策

- **为何选 kIdx=0 直接对比而非整层对比**：整层输出混有 ScatterAdd 的累加序误差，无法干净隔离 Cube 贡献；kIdx=0 是第一个 kernel 位置，其 MatMul 输出（tag=232）在 scatter 之前，直接反映 Cube 累加差异。
- **为何 dump 输入（tag=210/220）而不只 dump 输出**：先确认 NPU 与 CPU 的 MatMul 输入逐位一致，才能把输出差异归因到累加路径而非输入差异。trace 内已验证 `Indices match: True`（tag=331 vs CPU out_idx）。
- **放弃的设计**：最初尝试「匹配 CPU golden 的 `loop_point_count`」作为隔离手段，被实测推翻后放弃；用户还原了代码，改用 dump-tag 直接对比路线。
- **当前方案属哪档路线**：`recommended_fastpath`（隔离定位后再决定修复，不盲目改 tiling/累加顺序）。

## 验证信号

验证来源：本次会话 dump-tag 探针 + `compare_kidx0.py` 运行结果（kIdx=0 直接对比）。

- kIdx=0 Cube MatMul 输出（tag=232）vs CPU partial：1920 对中 max_diff 超过 ATK 阈值（约 4 ULP，数值 ~1073-1131 时 FP32 ULP ≈ 2^-13）（来源平台：950；证据等级: tool_observed）
- NPU tag=331 的 44 个 outputIdx 与 CPU out_idx 完全匹配（`Indices match: True`），索引逻辑无问题（来源平台：950；证据等级: tool_observed）
- 单核（usedVectorNum=1）仍失败 → 排除核间 atomic 竞争主导（来源平台：950；证据等级: tool_observed）

误差量级算式：数值 ~1073-1131 时 FP32 的 ULP = 2^(10-23) = 2^-13 ≈ 0.000122，实测 max_diff ≈ 4 ULP，符合 Cube FP32 累加顺序差异的预期量级。

## 性能影响

- 相对代价：dump-tag 探针只在 kIdx=0 触发，对正常运行无性能影响；全量 dump 会显著拖慢 kernel，仅用于调试。

## 预防

- 在精度失败时先做隔离定位（K=1/inChannels=1/单核/SIMT=1/kIdx 直接对比），再决定修复路线，避免盲目改 tiling 或放宽容差。
- dump 输入 + 输出双探针，先证输入一致再归因输出差异到累加路径。

## 失效边界

- 推断可迁移：Atlas A3（同 Cube+SIMT 架构，未验证，推断可迁移）；任何 Cube MatMul + SIMT ScatterAdd 混合 FP32 稀疏算子。
- 不适用：纯 Cube 算子（无 SIMT 层需隔离）；FP16/BF16/INT8（ULP 与累加路径不同，阈值与 ULP 算式需重算）。
- 未验证：K=1 与 inChannels=1 退化测试在本 trace 内由 assistant 提议但未由用户实测执行（用户最终走的是 kIdx=0 dump 对比路线）；`√B` 误差增长模型未做受控扫频。

## 坏实践

- 不做隔离就改 tiling/累加顺序/容差 —— 可能掩盖真因（`loop_point_count` 匹配失败已实测推翻）。
- 用全量 dump 跑性能评测 —— dump 会拖慢 kernel，性能数据失真。

## 误诊

- 最初假设「核间 atomic_add 竞争主导」→ 单核仍失败推翻。
- 「匹配 loop_point_count 能消除误差」→ 实测推翻。
- 「NPU 差 96 倍」→ CPU BLAS FP64 累加使分母趋零的假信号。