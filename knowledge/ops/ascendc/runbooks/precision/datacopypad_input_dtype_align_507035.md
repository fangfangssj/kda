---
type: "Runbook"
title: "DataCopyPad padding须按输入dtype对齐"
description: "输入 DataCopyPad 的 rightPadding/dstStride 用了 compute dtype 对齐宽度 EAlign（被 k=E 规则抬高），E=1 fp16 用例行布局 96B≠128B 致 507035；分离 EAlignInput=CeilAlign(E,32/sizeof(D_T_X)) 修复。"
tags: [task.debugging, topic.alignment]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# DataCopyPad padding须按输入dtype对齐

## 现象

ST 推进到 33/45 后剩一个孤立的边界失败：E=1 的 fp16 用例（shape=[1024,1]，k=1）稳定流同步失败 507035，其他 E=32 对齐、E=33/100 非对齐用例都过了。E=1 的语义极简单（softmax 单元素=1.0），却跑挂 kernel，说明是布局/搬运层问题而非算法。

## 适用场景

- 工程形态：GM→UB 输入搬运用 `DataCopyPad` + `rightPadding` 填充到对齐宽度、且 UB 行步长由**另一个**（计算 dtype 的）对齐宽度决定的 kernel。
- API 路径：`DataCopyPadExtParams<D_T_X>(isPad, leftPadding, rightPadding, paddingValue)`；`DataCopyExtParams.dstStride`（UB 侧单位 32B）。
- 验证过的平台与 CANN 版本：Ascend950PR (arch35) + CANN 9.1.0。
- 适用粒度：`operator_family`（多 dtype 输入 + fp32 计算的搬运布局设计）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用 / 未验证场景：输入 dtype 与计算 dtype 相同（如 fp32 直通）时 EAlignInput==EAlign，本卡问题不显现。

## 根因

输入搬运的 `rightPadding` 与 `dstStride` 之前用 `EAlign`（= max(32/inputDtypeSize, 32/computeDtypeSize) 对齐，fp16 输入 + fp32 计算时按 **16** 元素对齐）计算，但 32B 块对齐的**搬运边界**只取决于输入 dtype：`CeilAlign(E*sizeof(D_T_X), 32) = EAlignInput*sizeof(D_T_X)`，其中 `EAlignInput=CeilAlign(E, 32/sizeof(D_T_X))`（fp16 即 16 元素对齐——恰与 EAlign 同为 16？否：**k=E 边界规则会额外抬高 EAlign**）。E=1、k=1 时 `k==E` 触发 `EAlign += alignElem`，EAlign 被抬高到 32（元素），而 EAlignInput 仍应为 16：

```
blockLen        = E * sizeof(fp16)  = 2 字节
CeilAlign(2,32) = 32 字节（搬运侧实际占用）
dstStride       = (topkInner*4 - EAlign*2)/32 = (128-64)/32 = 2  → 总步长 = 32 + 2*32 = 96 字节
而缓冲区行步长 = topkInner * sizeof(COMPUTE_T) = 128 字节    → 96 ≠ 128，行间重叠错位 → kernel 异常 507035
```

用 EAlign（被 k=E 规则抬高、且视角混乱）代入 dstStride/rightPadding 公式，得到的行距与真实布局不符。

## 解决配方

TilingData 增加独立字段，输入搬运只认输入 dtype 对齐：

```cpp
// op_kernel/arch35/<op>_tiling_data.h（before → after）
    int64_t EAlign = 0;         // E 对齐到 max(input, compute) 对齐
+   int64_t EAlignInput = 0;    // E 对齐到输入 dtype 对齐 (32/sizeof(D_T_X)), 用于 DataCopyPad
    int64_t topkInner = 0;      // TopK inner = CeilAlign(EAlign, 32), 满足 TopK inner%32==0 约束
```

tiling 计算：

```cpp
    int64_t alignElemE = static_cast<int64_t>(std::max(ALIGN_BYTES / inputDtypeSize, ALIGN_BYTES / computeDtypeSize));
    int64_t EAlign = CeilAlign(E, alignElemE);
    // EAlignInput: 仅基于输入 dtype 对齐, 用于 DataCopyPad rightPadding
    int64_t EAlignInput = CeilAlign(E, static_cast<int64_t>(ALIGN_BYTES / inputDtypeSize));
    // 修复: k=E 时确保 EAlign > E 且是 alignElemE 的倍数
    if (k == E) {
        EAlign = EAlign + alignElemE;
    }
    ...
    tiling->EAlign = EAlign;
    tiling->EAlignInput = EAlignInput;
    tiling->topkInner = topkInner;
```

kernel 输入搬运改用 EAlignInput：

```cpp
    // 1. DataCopyPad: GM(x) → UB
    // rightPadding 基于 EAlignInput (输入 dtype 对齐), dstStride 基于 topkInner 行步长
    LocalTensor<D_T_X> xLocal = inputBuf_.Get<D_T_X>();
    DataCopyExtParams copyXParams;
    copyXParams.blockCount = static_cast<uint16_t>(curBatch);
    copyXParams.blockLen = static_cast<uint32_t>(E_ * sizeof(D_T_X));
    copyXParams.srcStride = static_cast<int64_t>(0);
    // UB 侧 stride = (topkInner_*sizeof(COMPUTE_T) - CeilAlign(E*sizeof(D_T_X),32)) / 32
    // CeilAlign(E*sizeof(D_T_X),32) = EAlignInput_*sizeof(D_T_X)
    copyXParams.dstStride = static_cast<int64_t>(
        (topkInner_ * sizeof(COMPUTE_T) - EAlignInput_ * sizeof(D_T_X)) / ALIGN_BYTES);
    copyXParams.rsv = 0;
    DataCopyPadExtParams<D_T_X> padXParams(true, 0,
        static_cast<uint8_t>(EAlignInput_ - E_), static_cast<D_T_X>(0));
    DataCopyPad(xLocal, gmX_[startRow * E_], copyXParams, padXParams);
```

fp16/bf16 的逐行 Cast 宽度同步改为 `EAlignInput_`：

```cpp
        // fp16/bf16: 逐行 Cast, 因为 inputBuf_ 行步长 = topkInner_*sizeof(COMPUTE_T)/sizeof(D_T_X)
        LocalTensor<COMPUTE_T> fp32Buf = softmaxBuf_.Get<COMPUTE_T>();
        int64_t srcRowStride = topkInner_ * sizeof(COMPUTE_T) / sizeof(D_T_X);  // half 行步长
        for (int64_t row = 0; row < curBatch; row++) {
            Cast<COMPUTE_T, D_T_X>(fp32Buf[row * topkInner_], xLocal[row * srcRowStride],
                                   RoundMode::CAST_NONE, static_cast<uint32_t>(EAlignInput_));
        }
```

恒等式校验（修复后必成立）：`EAlignInput * sizeof(D_T_X) == CeilAlign(E * sizeof(D_T_X), 32)`，故 dstStride 公式与搬运块边界自洽；E=1 用例中 EAlignInput*2=32 字节，总步长 = 32 + (128-32)/32*32 = 128 = 行步长 ✓。

## 设计决策

- 选择"分离字段"而非"让 EAlign 也按输入 dtype 对齐"：EAlign 还承担 SoftMax/compute 侧布局与 k=E 抬高规则，单一字段多义是本次 bug 根源；EAlignInput 只服务输入搬运这一件事。
- 用恒等式（EAlignInput*sizeof(D_T_X)=CeilAlign(blockLen,32)）作为设计不变量写进注释，后续改任何对齐宽度时可即时自检。

## 预防

- 多 dtype 视角（输入 dtype / 计算 dtype / 索引 dtype）的对齐宽度必须各自命名、各自计算；"一个 alignElem 走天下"在 fp16 输入 + fp32 计算的组合里必然翻车。
- 写 dstStride 公式时附带代入一个最小 shape（E=1）做数值自检——本例 96≠128 一眼可见。

## 验证信号

来源说明：问题与验证来自同一 NPU ST 联调会话（修复前后同机同用例重建对比）。

- 修复前：E=1、fp16、k=1 用例稳定 `[FAIL] 流同步失败: 507035`（来源平台：950；证据等级：tool_observed）。
- 机制算式：总步长 96B vs 行步长 128B 的失配推导（E=1, EAlign=32, topkInner=32, fp16）（来源平台：950；证据等级：artifact_verified，事件内算式）。
- 修复后：ST 33/45 → 34/45，该用例转 PASS（该轮唯一变化的通过用例即它，单卡归因干净）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖输入搬运侧的 padding/stride；SoftMax/TopK 侧布局（topkInner/EAlign）与输出侧（kPadOutput）各有独立契约，见同批 `softmax_srck_row_stride` 与 `topk_row_stride_misalignment_507035`。
- `rightPadding` 为 uint8 字段，EAlignInput-E 超过 255 的场景（E 极端非对齐 + 1 字节 dtype）未验证。

**未决项**：

- 无（E=1 已闭环；剩余 11 失败属其他根因卡）。
## 坏实践

- 把被 k=E 规则抬高过的 EAlign 复用在输入搬运参数上——抬高属于 compute/TopK 侧语义，与输入搬运块边界无关。

## 误诊

- E=1 失败初期被归因为"SoftMax 单元素 API 行为"或"blockLen=2 过小"，比对 M=1 通过的用例排除后转向布局推导（该段为推理叙述；96/128 算式为事件内推导）。
