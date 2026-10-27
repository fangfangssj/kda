---
type: "Runbook"
title: "Reg 固定地址跨块累加需逐 VF PipeBarrier"
description: "同一 VF 循环对固定 UB 累加器反复 StoreAlign→LoadAlign 会在 B16 多块规约中漏加；逐块 asc_vf_call 并在调用间 PipeBarrier 可闭合正确性。"
tags: ["task.debugging", "topic.precision", "topic.implementation.synchronization", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Reg 固定地址跨块累加需逐 VF PipeBarrier

## 现象

在 Ascend950 `AscendC::Reg` 末轴规约 kernel 中，单个向量 chunk 的 float16/bfloat16 结果精确，但同一 VF 内循环处理多个 chunk 后，归一化和开始大于 1。定向常量输入显示：

- 4 个 chunk 时，float16 输出首元素约 `0.005207`、归一化和约 `1.3330`；若四块都被累加，和应为 1；
- 16 个 chunk 时，float16 输出首元素约 `0.001736`、归一化和约 `1.7773`；
- bfloat16 呈现同类偏差；
- float32 在相同多 chunk 路径中误差保持在 `1e-9` 量级。

完整 pytest 在中间修复后仍为 48 通过、10 失败，失败集中在 float16/bfloat16 的多 chunk 规约。失败不是容差边缘：常量输入的归一化和已经明显偏离 1。

## 适用场景

- 工程形态：Ascend950 kernel 直调，`AscendC::Reg` 末轴规约；
- API 路径：`asc_vf_call` 调用 `__simd_vf__`，VF 内通过固定 `accAddr` 做 `LoadAlign→计算→StoreAlign` 的跨 chunk 累加；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致；
- 适用粒度：`operator_family`，适用于用同一 UB 地址跨向量 chunk 累加 max/sum 的规约类 kernel；
- 路线优先级：`safety_fallback`；
- **这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。** 当前草稿尚未建立相关卡，采用前必须重新做 capability/性能验证；
- 未验证场景：单 chunk 路径、没有固定地址 UB 回写的纯寄存器实现、其他 CANN 版本、A2/A3。

## 根因

单次 `__simd_vf__` 调用内部的 chunk 循环每轮都对同一 `accAddr` 执行： ```text LoadAlign(accAddr) → 更新寄存器 → StoreAlign(accAddr) ↓ 下一轮 LoadAlign(accAddr) ``` 循环内部没有可插入的 `PipeBarrier` 调用边界。工具信号表明 B16 路径的下一轮读取没有稳定看到上一轮完整写回，导致部分 chunk 贡献丢失。以 4 个全零 chunk 为例，正确指数和应为 `4×64=256`；观测首元素 `0.005207` 对应有效分母约 `1/0.005207≈192≈3×64`，即少累加一整块。把循环移到 `__aicore__` harness，每块单独 `asc_vf_call` 并在调用间加 `PipeBarrier<PIPE_V>()` 后，所有 dtype 与特殊值测试同时闭合。

## 解决配方

把“一个 VF 内循环所有 chunk”改成“一个 VF 只处理一个 chunk”；在 `__aicore__` 的 chunk 循环中，每次调用后执行 `PipeBarrier<PIPE_V>()`，再进入下一块。max 累加与 exp-sum 累加使用同一结构。以下代码 verbatim 来自事件流。

### before

revision: 未提交 pre-fix 工作树快照
path: `operators/softmax/softmax.asc`
symbol: `softmax_la_sum_vf<T>`
line segment: 函数全体

```cpp
// pass 2: row exp-sum. Lane-wise sum accumulated in a UB buffer (same reason as
// max). Reduce<SUM> + broadcast s.
template <typename T>
__simd_vf__ inline void softmax_la_sum_vf(__ubuf__ T *inAddr, __ubuf__ float *accAddr,
                                          __ubuf__ float *broadMAddr, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t fullLane = VL;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(fullLane);
    // accAddr pre-initialised (0.0) by a separate GenInit call + PipeBarrier.
    AscendC::Reg::RegTensor<float> bMReg;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(bMReg, broadMAddr);

    AscendC::Reg::RegTensor<float> sReg, xReg, tReg, eReg;
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t off = chunk * VL;
        uint32_t rem = count - off; rem -= ((rem - VL) & ((uint32_t)0 - (uint32_t)(rem > VL)));
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sReg, accAddr);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, inAddr + off);
        } else {
            AscendC::Reg::RegTensor<T> hReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hReg, inAddr + off);
            static constexpr AscendC::Reg::CastTrait traitIn = {
                AscendC::Reg::RegLayout::ZERO,
                AscendC::Reg::SatMode::UNKNOWN,
                AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            AscendC::Reg::Cast<float, T, traitIn>(xReg, hReg, mask);
        }
        AscendC::Reg::Sub(tReg, xReg, bMReg, mask);   // x - m
        AscendC::Reg::Exp(eReg, tReg, mask);
        AscendC::Reg::Add(sReg, sReg, eReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(accAddr, sReg, mask);
    }
}
```

revision: 未提交 pre-fix 工作树快照
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmaxLastAxis<T>::Process`
line segment: 函数全体

```cpp
    __aicore__ inline void Process()
    {
        constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        for (int64_t r = 0; r < mCount_; ++r) {
            int64_t off = r * K_;
            auto inLocal = inQueue_.AllocTensor<T>();
            AscendC::DataCopyPad(inLocal, xGm_[off], copyParams_, padParams);
            inQueue_.EnQue(inLocal);
            auto in = inQueue_.DeQue<T>();

            GenInit(accMaxBuf_.Get<float>(), NEG_INF, VL);
            AscendC::PipeBarrier<PIPE_V>();
            LaMax<T>(in, accMaxBuf_.Get<float>(), static_cast<uint32_t>(K_));
            AscendC::PipeBarrier<PIPE_V>();
            LaReduceMax(accMaxBuf_.Get<float>(), broadMBuf_.Get<float>());
            AscendC::PipeBarrier<PIPE_V>();
            GenInit(accSumBuf_.Get<float>(), 0.0f, VL);
            AscendC::PipeBarrier<PIPE_V>();
            LaSum<T>(in, accSumBuf_.Get<float>(), broadMBuf_.Get<float>(), static_cast<uint32_t>(K_));
            AscendC::PipeBarrier<PIPE_V>();
            LaReduceSum(accSumBuf_.Get<float>(), broadSBuf_.Get<float>());
            AscendC::PipeBarrier<PIPE_V>();

            auto outLocal = outQueue_.AllocTensor<T>();
            LaOut<T>(outLocal, in, broadMBuf_.Get<float>(), broadSBuf_.Get<float>(), static_cast<uint32_t>(K_));
            outQueue_.EnQue(outLocal);
            inQueue_.FreeTensor(in);

            auto out = outQueue_.DeQue<T>();
            AscendC::DataCopyPad(yGm_[off], out, copyParams_);
            outQueue_.FreeTensor(out);
        }
    }
```

### after

revision: fixed 工作树快照，后并入 `09b27d1`
path: `operators/softmax/softmax.asc`
symbol: `softmax_la_sum_chunk_vf<T>`
line segment: 函数全体

```cpp
// One chunk of row exp-sum accumulation (online-style: exp(x-m) added per
// chunk). Per-chunk vf + harness PipeBarrier for reliable accumulator commit.
template <typename T>
__simd_vf__ inline void softmax_la_sum_chunk_vf(__ubuf__ T *inAddr, __ubuf__ float *accAddr,
                                                 __ubuf__ float *broadMAddr, uint32_t off, uint32_t chunkElems)
{
    AscendC::Reg::RegTensor<float> bMReg;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(bMReg, broadMAddr);
    AscendC::Reg::RegTensor<float> sReg, xReg, tReg, eReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(chunkElems);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sReg, accAddr);
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, inAddr + off);
    } else {
        AscendC::Reg::RegTensor<T> hReg;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hReg, inAddr + off);
        static constexpr AscendC::Reg::CastTrait traitIn = {
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        AscendC::Reg::Cast<float, T, traitIn>(xReg, hReg, mask);
    }
    AscendC::Reg::Sub(tReg, xReg, bMReg, mask);   // x - m
    AscendC::Reg::Exp(eReg, tReg, mask);
    AscendC::Reg::Add(sReg, sReg, eReg, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(accAddr, sReg, mask);
}
```

revision: fixed 工作树快照，后并入 `09b27d1`
path: `operators/softmax/softmax.asc`
symbol: `KernelSoftmaxLastAxis<T>::Process`
line segment: 函数全体

```cpp
    __aicore__ inline void Process()
    {
        constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t K_u = static_cast<uint32_t>(K_);
        uint32_t chunkCount = (K_u + VL - 1) / VL;
        for (int64_t r = 0; r < mCount_; ++r) {
            int64_t off = r * K_;
            auto inLocal = inQueue_.AllocTensor<T>();
            AscendC::DataCopyPad(inLocal, xGm_[off], copyParams_, padParams);
            inQueue_.EnQue(inLocal);
            auto in = inQueue_.DeQue<T>();

            GenInit(accMaxBuf_.Get<float>(), NEG_INF, VL);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < chunkCount; ++c) {
                uint32_t coff = c * VL;
                uint32_t ce = K_u - coff; if (ce > VL) ce = VL;
                LaMaxChunk<T>(in, accMaxBuf_.Get<float>(), coff, ce);
                AscendC::PipeBarrier<PIPE_V>();
            }
            LaReduceMax(accMaxBuf_.Get<float>(), broadMBuf_.Get<float>());
            AscendC::PipeBarrier<PIPE_V>();

            GenInit(accSumBuf_.Get<float>(), 0.0f, VL);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < chunkCount; ++c) {
                uint32_t coff = c * VL;
                uint32_t ce = K_u - coff; if (ce > VL) ce = VL;
                LaSumChunk<T>(in, accSumBuf_.Get<float>(), broadMBuf_.Get<float>(), coff, ce);
                AscendC::PipeBarrier<PIPE_V>();
            }
            LaReduceSum(accSumBuf_.Get<float>(), broadSBuf_.Get<float>());
            AscendC::PipeBarrier<PIPE_V>();

            auto outLocal = outQueue_.AllocTensor<T>();
            LaOut<T>(outLocal, in, broadMBuf_.Get<float>(), broadSBuf_.Get<float>(), K_u);
            outQueue_.EnQue(outLocal);
            inQueue_.FreeTensor(in);

            auto out = outQueue_.DeQue<T>();
            AscendC::DataCopyPad(yGm_[off], out, copyParams_);
            outQueue_.FreeTensor(out);
        }
    }
```

## 设计决策

- 选择逐 chunk VF + barrier，因为它提供显式可见性边界，正确性不依赖同一 VF 内固定地址 forwarding；
- max 与 exp-sum 两个累加阶段采用对称结构，避免只修分母而遗漏 max 累加；
- 先把最终 `Reduce` 拆到独立 VF 解决同一次 VF 尾部的 RAW，只能修特殊值和单块；多块 B16 仍失败，说明 chunk 间同址依赖也必须拆；
- 曾尝试把末轴重排到 batched general 路线，有限值诊断归零但特殊值全套回归出现 20 个失败，已回退；
- 当前方案属于 `safety_fallback`：正确性闭合，但多次 `asc_vf_call` 和 barrier 带来显著性能代价。

## 性能影响

## 预防

- 看到“单 chunk 正确、多 chunk 错”时，用常量输入反推有效累加块数，不要只看随机 max_diff；
- 固定 UB 地址跨迭代 `StoreAlign→LoadAlign` 时，优先建立明确 VF 调用边界与 `PipeBarrier`；
- 将 float32 与 B16 分开诊断；float32 通过不能证明带 Cast 的 B16 时序同样安全；
- 最终验证同时覆盖单块、多块、尾块、float16/bfloat16/float32 与 NaN/Inf。

## 验证信号

来源为一次 Ascend950 Flash kernel 直调工程的定向 NPU 诊断、完整 pytest、干净重建和 cann-bench 评测。

- 修正前，单 chunk 的 float16/bfloat16 误差为 0，多 chunk 随机输入出现约 `1e-2` 到 `5e-2` 的最大误差；float32 同路径保持 `1e-9` 量级（证据等级: tool_observed；来源平台: 950）；
- 修正前，4-chunk 与 16-chunk 常量输入的归一化和分别约为 `1.33` 与 `1.78`，直接证明分母累加不完整（证据等级: tool_observed；来源平台: 950）；
- 逐 chunk VF + barrier 后，覆盖有限值与特殊值的 NPU pytest 为 58/58 通过（证据等级: tool_observed；来源平台: 950）；
- 再次干净重建仍为 58/58 通过（证据等级: tool_observed；来源平台: 950）；
- 最终 cann-bench 精度评测 20/20 通过（证据等级: tool_observed；来源平台: 950）。

## 失效边界

- 本卡证明的是 CANN 9.0.1 / 950 上该固定地址累加结构；不宣称所有 VF 内 Store/Load 都必须拆；
- 如果新实现能把全部累加保留在寄存器且不跨固定 UB 地址，可能不需要此 fallback；
- A2/A3 和其他 CANN 版本未验证，推断可迁移但不得当作已验证；
- 该方案性能明显低于基线，不适合作为未经比较的默认 fast path。

## 坏实践

- 只扩大 UB buffer：新鲜构建后的定向误差仍存在，不能解释 B16 多块漏加；
- 只把最终 Reduce 拆成独立 VF：能修最后一次 RAW，却不能修循环迭代之间的同址 RAW；
- 因 float32 通过就把 B16 失败归为容差问题；常量输入的归一化和已经大于 1；
- 放宽 fp16/bfloat16 容差以通过测试：这是测试侧放松，会掩盖整块贡献丢失。

## 误诊

- 初始怀疑是非对齐 `DataCopyPad`/`LoadAlign` 越界；按向量宽度取整缓冲后，定向错误仍可复现；
- 又怀疑 BF16 Cast 数值精度；全零/常量输入不涉及复杂数值，却仍显示漏块，排除了普通舍入误差；
- card lock、rsync timeout 与远程遗留进程是环境噪声，不解释稳定的 dtype/chunk 差异。