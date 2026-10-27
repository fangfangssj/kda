---
type: "Runbook"
title: "NDDMA多维dstStride致data搬运偏移的判别"
description: "Ascend950PR 上 NDDMA nddma_out_to_ub_b32 在 RANK=4 模板 2+ 实际维度时 loopDstStride 被硬件错误应用致 GM→UB data 搬运偏移；1 维通过、2+ 维偏移，GetValue 探针与 NddmaDispatch 对比定位。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-18T16:15:51Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["MaskedFillKernel"]
---

# NDDMA多维dstStride致data搬运偏移的判别

## 现象

Ascend950PR MaskedFill 算子（mask_predicate + Broadcast 范式，RegBase 手写 MicroAPI，NDDMA `DataCopy<DT, ND, cfg>` 做 GM→UB 搬运 + `DataCopyPad` UB→GM 回写）ST 联调中，1 维 tensor 全部通过，2+ 维 tensor 的 data 路径搬运结果出现整体偏移。以 FP32 `x.shape=[2,3]`、`mask=[1,0,1,0,1,0]`、value=-1.0 为例：output 自索引 3 起整体偏移 2——`output[3]=x[1]`（应为 `x[3]`）、`output[5]=x[3]`（应为 `x[5]`）。mask 选择路径本身正确（0/2/4→value、1/3/5→x），错的是 **x 的 GM→UB 搬运**：`gmX_.GetValue(3)` 返回 `x[1]` 而非 `x[3]`，同 case 的 1 维 `x=[1..8]` `GetValue(0..7)` 全部正确。判别信号：**1 维通过 + 2+ 维偏移 + mask 选择路径正确** → 指向"data 搬运错位"而非精度逻辑或 mask 路径。

## 适用场景

- 工程形态：kernel 直调（RegBase 手写 MicroAPI + NDDMA `DataCopy<DT, ND, cfg>` GM→UB 搬运 + `DataCopyPad` UB→GM 回写）。
- API 路径：`AscendC::DataCopy<DT, ND, NdDmaConfig>(UB, GM, MultiCopyParams)`；`MultiCopyParams<DT, ND>` 的 `loopInfo.loopDstStride` 字段；`AscendC::GlobalTensor<T>::GetValue`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 soc / arch35），CANN 9.1.0；kernel 目录 `op_kernel/arch35/`。
- 适用粒度：`operator_family`——所有走 NDDMA `nddma_out_to_ub_b32`（b32=4 字节元素）做 GM→UB 多维搬运的 Broadcast/elementwise 算子。
- 路线优先级：`safety_fallback`——本卡只给判别方法；规避此问题的 flat `DataCopyPad` 修复属 workaround，非根本修复，根本修复需驱动/工具链层解决 NDDMA 多维 dstStride 应用问题。
- 不适用或未验证场景：mask 路径 int8 b8 NDDMA 不受此问题影响（实测 mask 选择正确）；RANK>5 路径（外层 for + `DataCopy<DT, ND, cfg>`）未单独验证；a3/a2 平台未验证，推断可迁移性需独立实测。

## 判别方法

> 形态：diagnostic_rule。本卡只给触发/探针/观测/解释；修复后的 ST 验证闭环在归一化轨迹中被 redact（after/closure 未恢复），故不发布为 repair_recipe。

### 触发（trigger）

NDDMA b32 多维 GM→UB 搬运后，2+ 实际维度 case 的 data 路径结果整体偏移；1 维 case 全部正确；mask 选择路径本身正确。判别信号组合（1 维通过 + 2+ 维偏移 + mask 路径正确）将排查方向锁定到 **data 路径的 GM→UB 搬运**，而非精度逻辑或 mask 路径。

### 探针（probe）

1. **GetValue 直接读 GM 源**：在 kernel 入口对 2+ 维 case 的 `gmX_` 逐元素 `GetValue(i)` 并经 `gmOut_.SetValue` 回读，对照 1 维 case 同样探针，排除 UB 搬运后的下游环节（compute/回写）。
2. **NddmaDispatch 精确维度**：按 `nddma_dims_` 运行时 switch 调 `DataCopy<DT,1|2|3|4|5,cfg>`，消除 RANK=4 模板里的 filler（`loopSize=1`）维度，复跑 ST，验证 filler 维度假说。
3. **静态校验 loopInfo 编码**：核对 `loopInfo[0]` 为 innermost 维度、stride 单位为元素、filler 维度 `loopSize=1`（对照 CANN 头 `data_copy_interface.h` 注释 "Index [0, dim) represents lowerest dimension to highest dimension accordingly"），排除参数计算错误。

### 观测（observation）

- GetValue 探针：2 维 `[2,3]` case `gmX_.GetValue(3)=x[1]`（应为 `x[3]`），偏移 2；1 维 `[8]` case `gmX_.GetValue(0..7)` 全部正确（来源平台：950；证据等级：tool_observed）。
- NddmaDispatch 精确维度后：失败模式与 NDDMA RANK=4 模板完全一致，filler 维度假说不成立（来源平台：950；证据等级：tool_observed）。
- mask 路径 int8 b8 NDDMA 实测选择正确，不受 b32 多维问题影响（来源平台：950；证据等级：tool_observed）。

### 解释（interpretation）

NDDMA `nddma_out_to_ub_b32` 在 `RANK=4` 模板（ND=4，含 filler 维度）下、当**实际维度 ≥ 2** 时，`loopDstStride` 参数被硬件错误应用，导致 GM→UB 搬运后 UB 内 data 错位；1 维（仅一个实际维度）时 filler 不触发该路径，搬运正确。被诊断的 NDDMA 搬运机制单元（`CopyInBrcImpl`，RANK≤5 分支）：

```cpp
    template <typename DT>
    __aicore__ inline void CopyInBrcImpl(
        AscendC::MultiCopyParams<DT, ND>& params, int64_t outerIters,
        const int64_t* coord, int inputIdx, int slot, int64_t a_i_seg,
        const int64_t* strides, AscendC::GlobalTensor<DT>& gmIn,
        int64_t dstStrideScale = 1)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcOffset(coord, strides, RANK);
        const int64_t* dstShape = td_->max_bro_shape;

        auto p = params;
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd) p.loopInfo.loopSize[nd] = a_i_seg;
            p.loopInfo.loopDstStride[nd] = inner * dstStrideScale;
            inner *= p.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        if constexpr (RANK <= 5) {
            AscendC::DataCopy<DT, ND, cfg>(buf_[slot].Get<DT>(), gmIn[off], p);
        } else {
            AscendC::LocalTensor<DT> buf = buf_[slot].Get<DT>();
            for (int64_t oi = 0; oi < outerIters; oi++) {
                int64_t elem_adj = 0, tmp = oi;
                for (int64_t d = RANK - nddma_dims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? a_i_seg : dstShape[d];
                    elem_adj += (tmp % sz) * strides[d];
                    tmp /= sz;
                }
                AscendC::DataCopy<DT, ND, cfg>(buf[oi * inner], gmIn[off + elem_adj], p);
            }
        }
    }
```

机制链：(1) `CopyInBrcData` 调 `CopyInBrcImpl<T>(nddmaX_, ...)`，RANK≤5 分支执行 `DataCopy<DT, ND, cfg>`，`ND=4`，`loopDstStride` 按 `inner * dstStrideScale`（data 路径 dstStrideScale=1）逐维累乘填充；(2) case `[2,3]` split_axis=1，innermost `loopSize=3, dstStride=1`，split axis `loopSize=a_i_seg, dstStride=3`，filler `loopSize=1, dstStride=3`；(3) NPU 实测 `gmX_.GetValue(3)` 返回 `x[1]` 而非 `x[3]`，即 GM→UB 搬运后 UB 内 data 已错位；(4) NddmaDispatch 消除 filler 后失败模式不变，排除 filler 维度假说；(5) 静态校验 loopInfo 编码无误，排除参数计算错误。结论：`nddma_out_to_ub_b32` 在 2+ 实际维度时对 `loopDstStride` 的硬件应用与文档/静态语义不符，导致 data 错位。底层驱动/硬件机制层根因本 trace 未追到底。

## 验证信号

溯源：本卡证据取自 Ascend950PR MaskedFill 单迭代 ST 联调轨迹（GetValue 探针 + ST 运行 + NddmaDispatch 精确维度对比）。

- ST 修复前多维 case data 路径偏移、通过率 9/16（1 维 case 全过、2+ 维 case 偏移）（来源平台：950；证据等级：tool_observed）
- GetValue 探针：2 维 case `gmX_[3]=x[1]`（偏移 2），1 维 case `gmX_[0..7]` 全部正确（来源平台：950；证据等级：tool_observed）
- NddmaDispatch 精确维度后失败模式不变，filler 维度假说不成立（来源平台：950；证据等级：tool_observed）
- mask 路径 int8 b8 NDDMA 选择正确，不受 b32 多维问题影响（来源平台：950；证据等级：tool_observed）

## 失效边界

- Ascend950PR / arch35 已验证；a3、a2 未验证，推断可迁移（NDDMA 行为若同架构应一致，但需独立实测确认）。
- RANK>5 路径（`DataCopy<DT, ND, cfg>` ND=5 外层 for）未单独验证，未来若 mask 存储类型变更可能暴露同类问题。
- 本判别针对 b32（4 字节元素）NDDMA data 路径；int8 b8 mask 路径不受此问题影响。
- 本卡为判别方法，不包含修复闭环；驱动/工具链升级后 NDDMA b32 多维 dstStride 问题若被根本修复，可考虑回退 data 路径至 NDDMA 并重跑 ST 确认。

## 误诊

- 早期误判 mask 间距（`dstStrideScale=sizeof(T)`）导致偏移——参考文档 `int8-mask-alignment.md` 明确 sizeof(T) 间距是正确设计，mask 选择实测也正确，排除。
- 早期怀疑 `Contiguous` 创建的新 tensor GM 数据有问题——跳过 Contiguous 后通过率不变，排除。
- 早期怀疑 `CopyOutOne` 地址计算或 `DataCopyPad` 回写——简单 DataCopy 回写同样偏移，定位到搬运而非回写。
- workspace GM `SetValue` dump 调试：kernel 内 `SetValue` 写 workspace 后 ST 读回全 0，`wsAddr_` 传递链未跑通，调试手段失败，未产出可复用观测。

## 坏实践

- 用 `AscendC::printf` + `PipeBarrier<PIPE_ALL>` 在 MTE 流水中间 dump UB——触发 NPU 流同步错误 `507035`，破坏流水线时序（banned：kernel 内 printf + 全屏障 dump UB 不可用于 MTE 流水中间的数据采集）。