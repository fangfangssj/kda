---
type: "Runbook"
title: "TopK输出行步长kAlign与kPad不一致"
description: "TopK 按 tilling.kAlignFourBytes 步长逐行写输出，kernel 的 yBuf 行步长/Cast/搬出必须用同源对齐宽度（kPad 按 COMPUTE_T、kPadOutput 按 D_T_X），错用输入 dtype 对齐导致输出错位。"
tags: [task.debugging, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# TopK输出行步长kAlign与kPad不一致

## 现象

在 Ascend950PR 上联调 Softmax+TopK 组合 kernel（fp32 计算、fp16/fp32/bf16 三种输入输出 dtype）：aclnn 参数校验修好后首次跑 NPU ST，45 条用例仅 8 条通过，失败形态是**输出整体错位/垃圾**——y 的 MERE 高达 3.96e+01~9.01e+03（值本身像合法 softmax 值但位置错乱，如期望 0.094728 实际 0.174683），row_idx 全是垃圾（17378、931397879 等），部分用例流同步失败 507035。fp16 小 k（k<16）用例几乎全军覆没，fp32 对齐用例小 shape 通过——典型的"行步长错位"指纹。

## 适用场景

- 工程形态：kernel 内调用高层 API `TopK`（adv_api，arch35/DAV_3510 路径）后继续加工其 UB 输出（Cast、DataCopyPad 搬出到 GM）的算子。
- API 路径：`TopK`（`asc/include/adv_api/sort/topk.h` + `asc/impl/adv_api/detail/sort/topk/topk_3510_impl.h`）；`TopKTilingFunc`/`GetTopKMaxMinTmpSize`（tiling 侧）；`TopkTiling.kAlignFourBytes/kAlignTwoBytes` 字段（`adv_api/sort/topk_tilingdata.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR (arch35) + CANN 9.1.0。
- 适用粒度：`operator_family`（所有消费 TopK UB 输出的 kernel）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用 / 未验证场景：TopK 输出直接整块搬出且无中间 Cast 的场景（stride 天然一致）；`kAlignTwoBytes`（half 计算）路径本轨迹未用（COMPUTE_T 恒为 float），其量级推断为 CeilAlign(k, 16)（未验证）。

## 根因

TopK（arch3510 实现）**逐行**写输出，行起始偏移由 tiling 决定：

```cpp
// topk_3510_impl.h（CANN 9.1.0 实测节选）
for (int j = 0; j < topKInfo.outter; ++j) {
    int32_t dstOffsetFourBytes = j * tilling.kAlignFourBytes;
    TmpLocalSort32<T, isInitIndex>(...);
    GatherDstValAndDstIdx(dstValueLocal, dstIndexLocal, tmpLocal, tilling, dstOffsetFourBytes, j);
    ...
}
```

即值与索引缓冲的**行步长 = kAlignFourBytes**（由 `TopKTilingFunc` 按 32B/计算 dtype 对齐计算，float 下为 CeilAlign(k, 8) 量级）。而原 kernel/tiling 把 `kPad` 按 `alignElem = max(32/inputDtypeSize, 32/computeDtypeSize)` 计算——fp16 输入时 alignElem=16，`kPad=CeilAlign(k,16)`，与 TopK 实际写入步长 `CeilAlign(k,8)` 不一致（如 k=8：kPad=16 vs TopK 步长 8），第 2 行起读的全是错位数据；后续 Cast 与输出 DataCopyPad 的行步长也未按输出 dtype 单独对齐，错误层层传递。

## 解决配方

建立三个**同源**的对齐宽度，全部从 32 字节对齐第一性推导：

- `kPad = CeilAlign(k, 32/sizeof(COMPUTE_T))` —— 匹配 **TopK 输出行步长**（yBuf 分配与读取）；
- `kPadOutput = CeilAlign(k, 32/sizeof(D_T_X))` —— 匹配 **Cast 之后输出 dtype 的行步长**（使 `kPadOutput*sizeof(D_T_X)=CeilAlign(k*sizeof(D_T_X),32)`，输出 DataCopyPad 可恒用 srcStride=0）；
- `kPadIndex = CeilAlign(k, 32/sizeof(int32_t))` —— int32 索引行步长（与 TopK 索引写入步长一致）。

tiling 侧（`op_host/arch35/<op>_tiling.cpp`）before → after：

```cpp
// 修复前：kPad 混用输入 dtype 对齐宽度
    int64_t alignElem = static_cast<int64_t>(std::max(ALIGN_BYTES / inputDtypeSize, ALIGN_BYTES / computeDtypeSize));
    int64_t EAlign = CeilAlign(E, alignElem);
    int64_t kPad = CeilAlign(k, alignElem);
    int64_t kPadIndex = CeilAlign(k, static_cast<int64_t>(ALIGN_BYTES / sizeof(int32_t)));
```

```cpp
// 修复后：kPad 按 COMPUTE_T、kPadOutput 按 D_T_X 分别对齐
    int64_t alignElemE = static_cast<int64_t>(std::max(ALIGN_BYTES / inputDtypeSize, ALIGN_BYTES / computeDtypeSize));
    int64_t EAlign = CeilAlign(E, alignElemE);
    // kPad 必须匹配 TopK 输出 stride: CeilAlign(k, 32/computeDtypeSize)
    int64_t kPad = CeilAlign(k, static_cast<int64_t>(ALIGN_BYTES / computeDtypeSize));
    // kPadOutput 用于输出 Cast 后的 DataCopyPad: CeilAlign(k, 32/inputDtypeSize)
    int64_t kPadOutput = CeilAlign(k, static_cast<int64_t>(ALIGN_BYTES / inputDtypeSize));
    int64_t kPadIndex = CeilAlign(k, static_cast<int64_t>(ALIGN_BYTES / sizeof(int32_t)));  // int32 对齐 8
```

（TilingData 增加 `kPadOutput` 字段并在赋值处同步输出。）

kernel 侧 y 输出加工（Cast + 搬出）before → after：

```cpp
// 修复前：整块线性 Cast（隐含行步长=kPad，且未考虑输出 dtype 对齐）
    LocalTensor<D_T_X> yOut;
    if constexpr (std::is_same_v<D_T_X, float>) {
        yOut = yCompute.template ReinterpretCast<D_T_X>();
    } else {
        LocalTensor<D_T_X> yDtype = yBuf_.Get<D_T_X>();
        Cast<D_T_X, COMPUTE_T>(yDtype, yCompute, RoundMode::CAST_ROUND,
                               static_cast<uint32_t>(curBatch * kPad_));
        yOut = yDtype;
    }
```

```cpp
// 修复后：kPadOutput > kPad 时逐行 Cast 到正确行步长
    LocalTensor<D_T_X> yOut;
    if constexpr (std::is_same_v<D_T_X, float>) {
        yOut = yCompute.template ReinterpretCast<D_T_X>();
    } else {
        LocalTensor<D_T_X> yDtype = yBuf_.Get<D_T_X>();
        if (kPadOutput_ > kPad_) {
            // kPadOutput > kPad: 逐行 Cast, 确保输出行 32B 对齐
            for (int64_t row = 0; row < curBatch; row++) {
                Cast<D_T_X, COMPUTE_T>(
                    yDtype[row * kPadOutput_],
                    yCompute[row * kPad_],
                    RoundMode::CAST_ROUND,
                    static_cast<uint32_t>(kPad_));
            }
        } else {
            // kPadOutput == kPad: 线性 Cast (stride 匹配)
            Cast<D_T_X, COMPUTE_T>(yDtype, yCompute, RoundMode::CAST_ROUND,
                                   static_cast<uint32_t>(curBatch * kPad_));
        }
        yOut = yDtype;
    }
```

输出搬出（利用 `kPadOutput*sizeof(D_T_X)` 恒 32B 对齐的性质，srcStride 恒 0）：

```cpp
    // 6. DataCopyPad: UB → GM(y)
    // kPadOutput * sizeof(D_T_X) = CeilAlign(k * sizeof(D_T_X), 32), 始终 32B 对齐
    // srcStride = (kPadOutput * sizeof(D_T_X) - CeilAlign(k * sizeof(D_T_X), 32)) / 32 = 0
    DataCopyExtParams copyYParams;
    copyYParams.blockCount = static_cast<uint16_t>(curBatch);
    copyYParams.blockLen = static_cast<uint32_t>(k_ * sizeof(D_T_X));
    copyYParams.srcStride = static_cast<int64_t>(0);
    copyYParams.dstStride = static_cast<int64_t>(0);
    copyYParams.rsv = 0;
    DataCopyPad(gmY_[startRow * k_], yOut, copyYParams);
```

容量安全性：`batchRows * kPadOutput * sizeof(D_T_X) <= batchRows * kPad * sizeof(COMPUTE_T)`（half 占 float 一半、kPadOutput ≤ 2*kPad），yBuf 原尺寸即可容纳 Cast 后按 kPadOutput 步长排布的输出。

## 设计决策

- 为什么三个宽度分开算而不是统一一个：TopK 写入步长由**计算 dtype** 的 32B 对齐决定（kAlignFourBytes），而搬出/Cast 后的行布局由**输出 dtype** 的 32B 对齐决定，两者只在 fp32 时恰好相等；fp16/bf16 下必然分叉，混用任一侧都会错位。
- 逐行 Cast 的触发条件用运行时 `kPadOutput_ > kPad_` 判断（fp16 且 k 非 16 倍数时为真），避免无谓的逐行开销；fp32 与 16 倍数 k 走线性快路径。
- 备选（未采用）：输出搬出按实际行步长算 srcStride（`(kPadOutput*s - CeilAlign(k*s,32))/32`）——由对齐定义该值恒为 0，直接写 0 并加注释说明推导，比运行时计算更直白。
- **同批伴随修正**（无独立验证信号，为复现 29/45 一并记录）：
  - `isInitIndex` tiling 参数 false→true，与 kernel 侧 `TopK<..., true, ...>` 模板实参一致；
  - TopK 从 9 参数（内部 PopStackBuffer）换 **10 参数重载**显式传 tmpLocal（`TopK<COMPUTE_T, true, ..., TOPK_NORMAL>(yCompute, expertIdxLocal, softmaxDst, srcIndexLocal, finishLocal, sharedTmp, k_, topkTiling_, topKInfo_, true)`），避免 reserved UB 不足。

## 预防

- 消费任何高层 API 的 UB 输出前，先读该 API 的 impl 头确认**写入布局**（行步长、对齐），不要假设"输出就是紧凑 k 个元素"。
- 凡涉及行布局的宽度（kPad/kPadOutput/kPadIndex/EAlign），在 tiling 里集中定义并注释"匹配哪个 API 的哪个约束"，杜绝按单一 alignElem 复用。
- fp32 调通的组合 kernel 上 fp16 用例时，第一怀疑 dtype 相关的对齐宽度分叉。

## 验证信号

来源说明：问题与验证来自同一 NPU ST 联调会话（修复前后同机同用例重建对比）。

- 修复前：y `MERE=3.96e+01~9.01e+03`、`row_idx 位置 224 期望=14 实际=931397879`、`[FAIL] 流同步失败: 507035`（8/45）（来源平台：950；证据等级：tool_observed）。
- 机制证据：`topk_3510_impl.h:126 dstOffsetFourBytes = j * tilling.kAlignFourBytes` 逐行写输出（CANN 9.1.0 头文件）（来源平台：950；证据等级：artifact_verified）。
- 修复后：ST 8/45 → 29/45，常规 shape 全通过（含 fp16 k<16 用例），失败集合缩至非对齐 E/k=E/大规模/fp16 大E（各有独立根因卡：`softmax_srck_row_stride`、`datacopypad_input_dtype_align_507035`、`softmax_topk_large_scale_507035_unresolved`、`fp16_large_e_threshold_margin_unresolved`）（来源平台：950；证据等级：tool_observed；注：本卡与同批 4 项修正共同验证，单卡归因以"行步长契约"为主线）。

## 失效边界

- `kAlignFourBytes` 的具体数值由 `TopKTilingFunc`（闭源 so）计算，本卡按 32B/computeDtypeSize 对齐推导并在 float 计算路径实测吻合；half 计算路径（kAlignTwoBytes）未验证。
- 逐行 Cast 在 curBatch 很大时有额外开销，本轨迹未做性能对比（无性能实测信号）。
- k=E 与大规模场景的残留失败与本卡无关（见相关负结果卡）。

**未决项**：

- `TopKTilingFunc` 的 kAlignFourBytes 精确公式未在源码层验证（闭源）；如需完全确定性，可用单行 TopK + UB dump 实测步长。
- 逐行 Cast 路径的性能代价未量化。
## 坏实践

- 用一个 `alignElem` 同时充当输入搬运、TopK 输出、输出搬出的对齐宽度——三种 dtype 视角（D_T_X/COMPUTE_T/int32）在 fp16/bf16 下必不一致。
- 见输出"像合法值但顺序错"时先怀疑数值算法——本例 SoftMax 数值完全正确，错的是内存布局。

## 误诊

- 早期把 y 错位怀疑为 TopK 选错元素/排序不稳定，深入 GatherMask/Sear 排序语义多轮未果；后经 impl 头的 `dstOffsetFourBytes` 行步长证据转向布局错位（该段转向过程为推理叙述；行步长证据本身来自头文件实测）。
