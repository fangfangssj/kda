---
type: "Runbook"
title: "TQue 单核稳定排序去重 unique kernel 骨架"
description: "小规模 unique（排序去重+可选 inverse 输出）算子的完整 AscendC 实现配方：TQue 三队列自动同步 + 标量插入排序保稳定 + 单遍去重同时算 inverse + 按 hasInverse 动态 UB 预算（ubCoeff 12/20 B/elem、UB_RESERVED 32KB），int32 全量 ST bit-exact 通过。"
tags: [task.implementation, topic.compute.composite, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# TQue 单核稳定排序去重 unique kernel 骨架

## 现象

对标 `torch.unique` 的去重算子（输入 x，输出 y=升序去重值 + inverse=每元素映射到唯一值索引，inverse 为 OPTIONAL 输出）在 Ascend950PR 上实现时的任务形态：小规模（测试用例 ≤ 1000 元素）、单块可整装进 UB。早期实现用裸 TPipe/TBuf + 手工 SetFlag/WaitFlag 同步 + 手写 radix sort 路径，反复触发 507035（其中可选输出为 null 时的占位 tensor 崩溃是独立根因，见 [可选输出为 null 时占位 tensor 地址无效卡](optional_output_null_placeholder_507035.md)）。最终按 registry-invoke 模板参考（add_example）重写为 TQue 模式 + 标量插入排序的单核实现，一次通过全部 int32 用例（含 return_inverse 双形态）。本卡沉淀这套可直接复用的 kernel 骨架与 tiling 预算法。

## 适用场景

- 工程形态：registry-invoke 自定义算子开发（op_kernel + op_host tiling + kernel 入口三件套），小规模数据（单块可整装进 UB）。
- API 路径：AscendC `TQue<QuePosition::VECIN/VECOUT>`（AllocTensor/EnQue/DeQue/FreeTensor）、`TBuf<TPosition::VECCALC>`、`DataCopyPad` + `DataCopyParams`、`GET_TILING_DATA_WITH_STRUCT`/`REGISTER_TILING_DEFAULT`、tiling 侧 `ASCENDC_TPL_SEL_PARAM` 双 TilingKey（dtype × hasInverse）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：operator_family——"排序去重/按值分组"类算子的小规模路径（大规模需换 radix sort + 多核，见失效边界）。
- 路线优先级：recommended_fastpath（小规模正确性优先路径；大规模场景此骨架只作 safety_fallback，须分块）。
- 明确不适用 / 未验证场景：totalNum 超过单块 ubFactor 的分块/多核路径未实现（本骨架单块截断 `n = min(totalNum, ubFactor)`，超出部分静默丢）；fp16/bf16 等浮点 dtype 的 ±0/NaN 语义未在本阶段验证。

## 解决配方

复用本骨架的步骤（三件套）：

1. **kernel 类**（`op_kernel/arch35/<op>.h`）：TQue 三队列（VECIN 输入 + 两个 VECOUT 输出）+ VECCALC 索引缓冲；Init 统一 `pipe.InitBuffer` 全部缓冲；Process 按"CopyIn → DeQue → 标量稳定排序 → 去重（+inverse）→ EnQue → DeQue CopyOut"组织，可选输出用运行时 `hasInverse_` 分支。
2. **kernel 入口**（`op_kernel/<op>_arch35.cpp`）：`REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` 读 tiling，模板参数与 TilingKey 声明（dtype × bool）对应。
3. **TilingFunc**（`op_host/arch35/<op>_tiling.cpp`）：`GetPlatformInfo` 取 ubSize/coreNum → 按 hasInverse 选 ubCoeff（12.0/20.0）→ `ubFactor = FloorAlign((ubSize - 32KB)/ubCoeff, ubBlockSize)` → `SetBlockDim(1)`、workspace=0、`ASCENDC_TPL_SEL_PARAM(context, dTypeX, hasInverse)`。

关键设计事实：

- **TQue 自动同步**：`AllocTensor/EnQue/DeQue/FreeTensor` 自动维护 MTE2↔V↔MTE3 流水线事件，不需要手工 SetFlag/WaitFlag——裸同步写法在本算子族上事件方向/配对易错，是小规模路径反复 507035 的来源。
- **标量插入排序天然稳定**：相等元素不交换、索引跟随，保证 inverse 语义与 CPU golden 的 stable_sort 对齐；排序时带平行索引缓冲（idxBuf 与值同步搬移），去重遍历时 `invLocal.SetValue(origIdx, uniqueCount-1)` 一步得到 inverse。
- **无条件 InitBuffer 全部队列、Process 里用运行时 hasInverse_ 分支**（而不是 `if constexpr` 按模板参数跳过 invQueue_ 初始化）：可选输出为 null 时框架会给占位地址，队列初始化统一做、只在运行时分支决定是否真正读写，规避"未初始化队列"类问题；模板参数 HAS_INVERSE 仍保留用于 TilingKey 选择。
- **UB 预算**：每元素 inputQueue 4B + idxBuf 4B + yQueue 4B（+ invQueue 8B）→ ubCoeff 按 hasInverse 取 12.0 或 20.0；预留 `UB_RESERVED = 32KB` 给 TQue 元数据与栈；ubFactor 落到 ubBlockSize 对齐。实测 950 的 ubSize=253952 时 ubFactor 分别为 18432（无 inverse）/ 11040（有 inverse）。
- **workspace**：本骨架纯 UB 内完成，`currentWorkspace[0] = 0`。

## 完整代码

kernel 类（`op_kernel/arch35/bench_no_harness_unique.h`，NPU 验证通过版本，verbatim）：

```cpp
/*!
 * \file bench_no_harness_unique.h
 * \brief 去重算子 Kernel 类定义（arch35 / Ascend950）
 *
 * 算法路径：展平 → 稳定排序 → 去重紧凑 → 输出 y + inverse
 */

#ifndef BENCH_NO_HARNESS_UNIQUE_H
#define BENCH_NO_HARNESS_UNIQUE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bench_no_harness_unique_tiling_data.h"
#include "bench_no_harness_unique_tiling_key.h"

namespace NsBenchNoHarnessUnique {

using namespace AscendC;

template <typename D_T_X, int HAS_INVERSE>
class BenchNoHarnessUnique {
public:
    __aicore__ inline BenchNoHarnessUnique() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR inverse, GM_ADDR workspace,
                                const BenchNoHarnessUniqueTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<D_T_X> inputGMX_;
    GlobalTensor<D_T_X> outputGMY_;
    GlobalTensor<int64_t> outputGMInv_;

    int64_t totalNum_ = 0;
    int64_t ubFactor_ = 0;
    uint32_t hasInverse_ = 0;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECOUT, 1> invQueue_;
    TBuf<TPosition::VECCALC> idxBuf_;
};

template <typename D_T_X, int HAS_INVERSE>
__aicore__ inline void BenchNoHarnessUnique<D_T_X, HAS_INVERSE>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR inverse, GM_ADDR workspace,
    const BenchNoHarnessUniqueTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    ubFactor_ = tilingData->ubFactor;
    hasInverse_ = tilingData->hasInverse;

    inputGMX_.SetGlobalBuffer((__gm__ D_T_X*)x);
    outputGMY_.SetGlobalBuffer((__gm__ D_T_X*)y);
    outputGMInv_.SetGlobalBuffer((__gm__ int64_t*)inverse);

    pipe.InitBuffer(inputQueue_, 1, ubFactor_ * sizeof(int32_t));
    pipe.InitBuffer(idxBuf_, ubFactor_ * sizeof(int32_t));
    pipe.InitBuffer(yQueue_, 1, ubFactor_ * sizeof(int32_t));
    pipe.InitBuffer(invQueue_, 1, ubFactor_ * sizeof(int64_t));
}

template <typename D_T_X, int HAS_INVERSE>
__aicore__ inline void BenchNoHarnessUnique<D_T_X, HAS_INVERSE>::Process()
{
    if (totalNum_ <= 0) {
        return;
    }

    int64_t n = (totalNum_ <= ubFactor_) ? totalNum_ : ubFactor_;

    // ===== Step 1: CopyIn (GM → UB, MTE2) =====
    auto inputLocal = inputQueue_.template AllocTensor<int32_t>();
    DataCopyParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = static_cast<uint32_t>(n * sizeof(int32_t));
    inParams.srcStride = 0;
    inParams.dstStride = 0;
    DataCopyPad(inputLocal, inputGMX_[0], inParams, {false, 0, 0, 0});
    inputQueue_.EnQue(inputLocal);

    // ===== Step 2: DeQue (等待 MTE2→V 同步) =====
    inputLocal = inputQueue_.template DeQue<int32_t>();

    // ===== Step 3: 生成原始索引 =====
    auto idxLocal = idxBuf_.template Get<int32_t>();
    for (int64_t i = 0; i < n; i++) {
        idxLocal.SetValue(i, static_cast<int32_t>(i));
    }

    // ===== Step 4: 插入排序（稳定排序） =====
    for (int64_t i = 1; i < n; i++) {
        int32_t keyVal = inputLocal.GetValue(i);
        int32_t keyIdx = idxLocal.GetValue(i);
        int64_t j = i - 1;
        while (j >= 0 && inputLocal.GetValue(j) > keyVal) {
            int32_t tmpVal = inputLocal.GetValue(j);
            int32_t tmpIdx = idxLocal.GetValue(j);
            inputLocal.SetValue(j + 1, tmpVal);
            idxLocal.SetValue(j + 1, tmpIdx);
            j--;
        }
        inputLocal.SetValue(j + 1, keyVal);
        idxLocal.SetValue(j + 1, keyIdx);
    }

    // ===== Step 5: 去重 + inverse 计算 =====
    auto yLocal = yQueue_.template AllocTensor<int32_t>();
    int64_t uniqueCount = 0;
    int32_t prevVal = 0;
    bool hasPrev = false;

    if (hasInverse_ != 0) {
        // ---- HAS_INVERSE=1: 计算 inverse 并写回 GM ----
        auto invLocal = invQueue_.template AllocTensor<int64_t>();

        for (int64_t i = 0; i < n; i++) {
            int32_t currVal = inputLocal.GetValue(i);
            int32_t origIdx = idxLocal.GetValue(i);
            if (!hasPrev || currVal != prevVal) {
                yLocal.SetValue(uniqueCount, currVal);
                uniqueCount++;
                prevVal = currVal;
                hasPrev = true;
            }
            invLocal.SetValue(origIdx, static_cast<int64_t>(uniqueCount - 1));
        }

        invQueue_.EnQue(invLocal);
        yQueue_.EnQue(yLocal);
        inputQueue_.FreeTensor(inputLocal);

        // CopyOut y
        auto yOut = yQueue_.template DeQue<int32_t>();
        DataCopyParams yOutParams;
        yOutParams.blockCount = 1;
        yOutParams.blockLen = static_cast<uint32_t>(uniqueCount * sizeof(int32_t));
        yOutParams.srcStride = 0;
        yOutParams.dstStride = 0;
        if (uniqueCount > 0) {
            DataCopyPad(outputGMY_[0], yOut, yOutParams);
        }
        yQueue_.FreeTensor(yOut);

        // CopyOut inverse
        auto invOut = invQueue_.template DeQue<int64_t>();
        DataCopyParams invOutParams;
        invOutParams.blockCount = 1;
        invOutParams.blockLen = static_cast<uint32_t>(n * sizeof(int64_t));
        invOutParams.srcStride = 0;
        invOutParams.dstStride = 0;
        DataCopyPad(outputGMInv_[0], invOut, invOutParams);
        invQueue_.FreeTensor(invOut);

    } else {
        // ---- HAS_INVERSE=0: 仅去重，不计算 inverse，不操作 invQueue_ ----
        for (int64_t i = 0; i < n; i++) {
            int32_t currVal = inputLocal.GetValue(i);
            if (!hasPrev || currVal != prevVal) {
                yLocal.SetValue(uniqueCount, currVal);
                uniqueCount++;
                prevVal = currVal;
                hasPrev = true;
            }
        }

        yQueue_.EnQue(yLocal);
        inputQueue_.FreeTensor(inputLocal);

        // CopyOut y
        auto yOut = yQueue_.template DeQue<int32_t>();
        DataCopyParams yOutParams;
        yOutParams.blockCount = 1;
        yOutParams.blockLen = static_cast<uint32_t>(uniqueCount * sizeof(int32_t));
        yOutParams.srcStride = 0;
        yOutParams.dstStride = 0;
        if (uniqueCount > 0) {
            DataCopyPad(outputGMY_[0], yOut, yOutParams);
        }
        yQueue_.FreeTensor(yOut);
    }
}

} // namespace NsBenchNoHarnessUnique
#endif // BENCH_NO_HARNESS_UNIQUE_H
```

kernel 入口（`op_kernel/bench_no_harness_unique_arch35.cpp`，终版；调试中曾因误诊改用逐字段手动读 tiling，最终确认 `GET_TILING_DATA_WITH_STRUCT` 可正常使用后回归宏写法）：

```cpp
template <typename D_T_X, int HAS_INVERSE>
__global__ __aicore__ void bench_no_harness_unique(
    GM_ADDR x, GM_ADDR y, GM_ADDR inverse, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessUniqueTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessUniqueTilingData, tilingData, tiling);

    NsBenchNoHarnessUnique::BenchNoHarnessUnique<D_T_X, HAS_INVERSE> op;
    op.Init(x, y, inverse, workspace, &tilingData);
    op.Process();
}
```

tiling 侧 UB 预算核心（`op_host/arch35/bench_no_harness_unique_tiling.cpp`，终版片段）：

```cpp
constexpr int64_t UB_RESERVED = 32 * 1024;   // 预留 32KB 给 TQue 元数据和栈

// 获取平台信息
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BenchNoHarnessUniqueTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息 / 2. 输入形状与属性 return_inverse → hasInverse
    // 3. 空 Tensor 早退（SetBlockDim(1)、workspace=0）
    // ...
    // 5. 单核执行（blockDim=1）
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t blockFactor = CeilAlign(totalNum, ubBlockSize);
    int64_t usedCoreNum = 1;

    // 6. UB 切分
    // UB 预算：
    //   HAS_INVERSE=1: inputQueue(4B) + idxBuf(4B) + yQueue(4B) + invQueue(8B) = 20B/element
    //   HAS_INVERSE=0: inputQueue(4B) + idxBuf(4B) + yQueue(4B) = 12B/element
    double ubCoeff = hasInverse ? 20.0 : 12.0;
    int64_t availableUb = static_cast<int64_t>(ubSize) - UB_RESERVED;
    int64_t ubFactor = static_cast<int64_t>(static_cast<double>(availableUb) / ubCoeff);
    ubFactor = FloorAlign(ubFactor, ubBlockSize);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor too small: ubSize=%lu", ubSize),
                return ge::GRAPH_FAILED);

    // 7. Workspace 计算（简化版不需要 workspace）
    // ...
    // 8. 填 tiling 结构 + SetBlockDim(usedCoreNum) + ASCENDC_TPL_SEL_PARAM(context, dTypeX, hasInverse)
```

## 设计决策

- **为什么用 TQue 而不是裸 TBuf + 手工事件同步**：`TQue` 的 AllocTensor/EnQue/DeQue/FreeTensor 自动维护 MTE2↔V↔MTE3 流水线事件，模板参考实现即此模式；裸同步写法在本算子上排查成本高（事件方向/配对易错）。
- **为什么标量插入排序**：小规模（≤ 1000 元素量级）下 `GetValue/SetValue` 标量插入排序 O(n²) 足够且**天然稳定**（相等元素不交换，索引跟随），保证 inverse 语义与 CPU golden 的 stable_sort 对齐；更大规模须换 radix sort（tiling 里 KEY_WORDS/SORT_PASSES 字段已预留）。
- **为什么单核（blockDim=1）**：去重紧凑写 y 需要全局有序视图；多核需归并，小规模不值得。tiling 显式 `context->SetBlockDim(1)`。
- **为什么排序时带平行索引缓冲**：排序只搬值会丢"原位置"信息，inverse 恰恰要写回原位置；idxBuf 与值同步搬移，去重遍历时一步得到 inverse。
- **为什么无条件 InitBuffer 全部队列、运行时 hasInverse_ 分支**：见解决配方要点；`if constexpr` 跳过可选输出队列初始化配合框架占位输出行为容易踩未初始化队列问题。

## 验证信号

本卡信号来自该算子重写后的 NPU 联调与回归（Ascend950PR + CANN 9.1.0，构建 `bash build.sh --soc=ascend950 -j8` + 安装 + ST/UT）：

- 重写后首次回归：`return_inverse=true`（HAS_INVERSE=1）全部用例 PASS（此时 HAS_INVERSE=0 因占位输出问题失败，随后由 L2 层 dummy tensor 修复，见可选输出占位卡）（来源平台：950；证据等级：tool_observed）。
- 最终回归：int32 全部用例 PASS（`return_inverse=true/false` 双形态，K 与 golden 一致、y/inverse bit-exact），ST 32/34（int32 32/32；float16 0/2 为本阶段范围外预期失败）（来源平台：950；证据等级：tool_observed）。
- UT：tiling 单测 13/13 通过（ubFactor 期望值按 UB_RESERVED=32KB + 动态 ubCoeff 12.0/20.0 同步更新后）（来源平台：950；证据等级：tool_observed）。
- tiling 数值自洽：ubSize=253952 时 ubFactor 无 inverse = FloorAlign((253952-32768)/12.0, 32) = 18432、有 inverse = 11040，与 plog 观测一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 规模边界：totalNum > ubFactor 时本骨架只处理前 ubFactor 个元素（截断），未做分块——大规模必须换分块/多核 + radix sort 路径（未实现验证）。
- dtype 边界：仅 int32 在 NPU 验证；fp16 的 ±0 合并/±inf 独立/NaN 语义属于后续阶段（golden 自测已在 CPU 侧定义并通过，NPU kernel 未验证），fp16 路径的可排序 key 编码另见 [unique 负零符号归一卡](../precision/torch_unique_negzero_sign_normalize.md)。
- 排序稳定性依赖插入排序实现；换排序算法时必须保持稳定，否则 inverse 语义漂移。
- 单核 blockDim=1，未利用多核；性能不是本骨架目标（无 profiling 数据，不构成性能结论）。
- 其他 SoC 未实测（未验证，推断可迁移——TQue/tiling 机制为 AscendC 通用机制）。

**未决项**：

- 大规模分块/多核 + radix sort 路径（tiling 已预留 KEY_WORDS/SORT_PASSES 字段）未实现。
- fp16 路径（含可排序 key 编码与 ±0/±inf/NaN 语义）未闭环。
## 坏实践

- 裸 TPipe/TBuf + 手工 SetFlag/WaitFlag 搭 DataCopyPad 的旧写法在本算子族上排查成本高（事件方向与配对易错），小规模路径优先 TQue。
- `if constexpr` 按模板参数跳过可选输出队列的 InitBuffer——配合框架占位输出行为容易踩未初始化队列问题（本骨架改为无条件初始化 + 运行时分支）。

## 误诊

- 调试期曾误诊 `GET_TILING_DATA_WITH_STRUCT` 不可用而改为逐字段手动读 tiling；最终确认宏可正常使用后回归宏写法（该误诊记录属同算子调试史，不构成对本卡配方的反证）。
