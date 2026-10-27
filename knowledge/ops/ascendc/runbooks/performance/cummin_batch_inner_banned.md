---
type: "Runbook"
title: "cummin 非末维 batch-inner 跨步扫描 banned"
description: "cummin 非末维用 batch-inner（按 dim 步逐次搬入 innerSize 元素并标量比较）替代 transpose，在大 dimSize case 上相对 transpose 路线慢 37-78×，整体评分下降；该路线 banned，应保留 transpose。"
tags: ["task.optimization", "topic.compute.reduction", "topic.data_movement.memory_access", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["batch_inner"]
---

# cummin 非末维 batch-inner 跨步扫描 banned

## 现象

在 Ascend 950（NpuArch dav-3510）上用 ops-direct-invoke-flash 实现 cummin。基线 kernel 走 **transpose 路线**：plugin 对非末维输入先 `transpose(dim, rank-1).contiguous()`，kernel 沿末维（连续）做前缀最小扫描，再 transpose 回写。该基线精度全 PASS，但部分非末维/大 shape case 的 speedup 低于 1×，主要由 transpose + `.contiguous()` 开销与 `GetValue/SetValue` 标量扫描造成。

为消除 transpose 开销，尝试重写为 **batch-inner 路线**：不再 transpose，kernel 直接对原始（跨步）布局处理——对每个 chain，沿 dim 步逐次搬入 innerSize 个元素、逐元素 `GetValue` 比较并 `SetValue`，再跨步搬出。重建后实测整体评分相对 transpose 路线下降：大 dimSize case 大幅退化，仅 5D 小 dimSize 的 Case 14 略改善。随即回退到 transpose 路线。

## 适用场景

- 工程形态：kernel 侧前缀扫描（cummin/cummax 类，带条件索引更新的顺序依赖计算）。
- API 路径：`AscendC::DataCopyPad` + `LocalTensor::GetValue/SetValue`（dav-3510 上前缀扫描只能顺序标量进行）。
- 验证平台与版本：Ascend 950 / NpuArch dav-3510。
- 适用粒度：`operator_family`（凡带顺序依赖条件更新的前缀扫描算子，cummin/cummax）。
- 路线优先级：`banned`（batch-inner 跨步逐次扫描在本 trace 实测退化；非末维应保留 transpose 路线）。
- 不适用或未验证场景：dimSize 极小且 innerSize 大的高维 case 上 batch-inner 可能略优，但不足以扭转整体；a2/a3 未验证。

## 失败路线

banned：非末维 batch-inner 跨步逐次扫描。

尝试：把"每 chain 一次连续 DMA + 连续扫描"（transpose 参照路线）改为"每 dim 步一次 innerSize 元素的跨步 DMA + 标量比较"（batch-inner），期望消除 transpose + `.contiguous()` 开销。

客观失败：batch-inner 相对 transpose 参照路线在大 dimSize case 退化 37-78×，整体评分相对 transpose 路线下降；仅 Case 14（dimSize 极小）约为 transpose 的 2×（唯一受益）。随即回退到 transpose 路线。

根因（机制链，分析层）：batch-inner 把"每 chain 一次连续 DMA"换成"每 dim 步一次 innerSize 元素的跨步 DMA + 标量比较"。对 dimSize 个 dim 步，每 chain 产生 `dimSize` 次（而非 1 次）DMA，且每次 DMA 后是 `innerSize` 次 `GetValue/SetValue` 标量循环。当 dimSize 大时：DMA 次数 = `dimSize × chains`，远多于 transpose 路线的 `chains`；每次 DMA 搬运量仅 `innerSize` 元素，启动开销占比高，带宽未饱和；标量比较次数 = `dimSize × innerSize × outerSize`，受标量吞吐限制。Case 14（dimSize=17）DMA 次数仅 17，transpose 的 5D 转置开销反而更高，故 batch-inner 约为 transpose 的 2×。

该路线不暗示已有等价替代；非末维场景仍用 transpose 路线，或改用真正的 ND stride DMA（一次性跨步搬入整条 chain 后在 UB 连续扫描）——后者本 trace 仅调研未实现验证，列为遗留。

## 参照路线（transpose，verbatim）

基线 kernel 每 chain 一次连续 `DataCopyPad` 搬入、`GetValue/SetValue` 顺序扫描、一次搬出（file_read verbatim，DMA 次数 = chains）：

```cpp
    __aicore__ inline void ProcessChain(int64_t chain)
    {
        int64_t baseOffset = chain * dimSize_;
        float running_min_f = 3.402823466e+38f;
        int32_t running_idx = 0;
        int64_t tileCount = (dimSize_ + tileElems_ - 1) / tileElems_;
        for (int64_t tile = 0; tile < tileCount; ++tile) {
            int64_t tileStart = tile * tileElems_;
            int64_t tileLen = dimSize_ - tileStart;
            if (tileLen > tileElems_) tileLen = tileElems_;
            int32_t tLen = (int32_t)tileLen;
            auto inL = inBuf_.Get<T>();
            AscendC::DataCopyExtParams cpT{1, (uint32_t)(tLen * (int32_t)sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padT{false, 0, 0, 0};
            AscendC::DataCopyPad(inL, inGm_[baseOffset + tileStart], cpT, padT);
            AscendC::PipeBarrier<PIPE_ALL>();
            // ... ScanFloat/ScanInt/ScanB16: 顺序 GetValue 比较、SetValue 更新运行最小与索引 ...
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(valGm_[baseOffset + tileStart], vL, cpT);
            AscendC::DataCopyPad(idxGm_[baseOffset + tileStart], i32L, cpIdx);
        }
    }
```

`ScanFloat` 的核心标量循环（顺序 `GetValue` 比较、`SetValue` 更新运行最小与索引）：

```cpp
    __aicore__ inline void ScanFloat(
        AscendC::LocalTensor<float> inL, AscendC::LocalTensor<float> vL,
        AscendC::LocalTensor<int32_t> i32L, int32_t tLen,
        float &running_min, int32_t &running_idx, bool &has_nan, int32_t tileStart)
    {
        for (int32_t k = 0; k < tLen; ++k) {
            float current = inL.GetValue(k);
            bool is_nan_cur = current != current;
            bool is_nan_run = running_min != running_min;
            bool update = is_nan_cur || (!is_nan_run && current <= running_min);
            if (update) { running_min = current; running_idx = tileStart + k; }
            vL.SetValue(k, running_min);
            i32L.SetValue(k, running_idx);
        }
    }
```

这正是 batch-inner 想替代的参照路线：每 chain 一次连续搬入 + 连续扫描 + 一次搬出。

## 性能影响

batch-inner 相对 transpose 参照路线的成对相对代价（同算子、同 case、同 baseline，来源平台 950）：

- Case 5（16384×16384 fp16 dim=0）：batch-inner 相对 transpose 慢约 65×。
- Case 8（1009×1021 fp16 dim=0）：慢约 69×。
- Case 17（255×8193 bf16 dim=0）：慢约 78×。
- Case 19（2×511×2049 fp16 dim=1）：慢约 37×。
- Case 14（11×13×17×67×67 fp16 dim=2，dimSize=17）：batch-inner 约为 transpose 的 2×（唯一受益，因 dimSize 极小、5D transpose 开销更高）。

整体：batch-inner 使 aggregate 评分相对 transpose 路线下降。

## 验证信号

本卡信号来自 trace 中对 cummin kernel 做 batch-inner vs transpose 的对照实测（`run_evaluation.sh --operator Cummin --device-id 0`，eval 命令已执行；adapter 将 stdout 折入紧随其后的 reasoning，数值为 eval 工具实际产出而非 agent 臆测）。

- batch-inner 相对 transpose 参照路线在大 dimSize case 慢 37-78×（Case 5/8/17/19）（来源平台：950；证据等级：tool_observed）
- Case 14（dimSize=17）batch-inner 约为 transpose 的 2×，唯一受益（来源平台：950；证据等级：tool_observed）
- 整体 aggregate 评分相对 transpose 路线下降（来源平台：950；证据等级：tool_observed）
- 实测后随即回退到 transpose 路线（来源平台：950；证据等级：tool_observed）
- 参照路线 transpose kernel 的 ProcessChain/ScanFloat verbatim 代码从 file_read 恢复（来源平台：950；证据等级：artifact_verified）

## 坏实践

- 用"每 dim 步一次小 DMA + 标量比较"替代"每 chain 一次连续 DMA + 连续扫描"，未先估算 DMA 次数随 `dimSize` 线性膨胀的后果。
- 仅凭"消除 transpose"的直觉重写非末维路径，未对大 dimSize case 做单点 sanity 就跑全量 eval。

## 失效边界

- 本 trace 仅在 Ascend 950 / dav-3510 实测；a2、a3 未验证，推断可迁移（DMA 启动开销与标量吞吐瓶颈是平台硬件特性，非 950 独有，但具体倍数会变）。
- batch-inner 在 dimSize 极小且高维 transpose 开销高的 case 上可能略优（Case 14 约 2×），但单一 case 不足以扭转整体；若要做 dimSize 自适应分支，须先在大 dimSize case 上证明不退化——本 trace 证明大 dimSize case 退化严重。