---
type: "Runbook"
title: "AR 输出张量未初始化导致 Softmax 部分用例错"
description: "Ascend C Reg 的原地 max-sub-exp helper 误读新分配输出张量；显式拆分源、目标并让 fp32 路径读取输入张量后恢复精度。"
tags: ["task.debugging", "topic.compute.normalization", "topic.implementation.memory_management", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# AR 输出张量未初始化导致 Softmax 部分用例错

## 现象

在为 `softmax` 编写 Ascend C 直调 Reg kernel、完成编译后，运行 NPU pytest 全量精度套件时得到 `39 passed, 45 failed`。其中末轴连续（AR）fp32 的最小探针没有 NaN/Inf，输出之和也接近 1，却与有限的 golden 分布明显不符。这种“看起来已归一化、但概率对应不上输入”的信号出现在 `operators/softmax/softmax.asc` 的 AR 分支；当时代码已经从输入队列取出 `inT`，却把刚从输出队列分配的 `outT` 直接交给原地 `max_sub_exp_vf`。

## 适用场景

- 工程形态：PyTorch 扩展中的 Ascend C kernel 直调，末轴连续 softmax 使用 `__simd_vf__` Reg helper；
- API 路径：`TQue::DeQue` / `TQue::AllocTensor` → `asc_vf_call` → `Reg::LoadAlign` / `Reduce` / `Exp` / `StoreAlign`；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1；
- 适用粒度：`operator_family`，适用于“从输入队列读数据、向另一个输出 LocalTensor 写结果”的 Reg 归约/变换 helper；
- 路线优先级：`safety_fallback`，该实现以三阶段正确性为先，没有证明性能最优；
- 明确不适用 / 未验证场景：原本就要求同一 LocalTensor 原地更新、且调用前已写满有效数据的 helper；其他 NPU 平台与其他 CANN 版本未验证。

## 根因

修复前的 helper 契约只有一个 `buf`，数学上执行： `m = max(buf)`，随后 `buf[i] = exp(buf[i] - m)`。 fp32 的 AR `Compute` 却执行 `outT = AllocTensor<T>()` 后立刻调用 `CallMaxSubExp(outT, r_)`。`AllocTensor` 只取得可写 UB 张量，不会把 `inT` 复制进去；因此 helper 的两遍 `LoadAlign` 都从未初始化的 `outT` 读取。后续 `sum_inv_muls_vf` 仍会把这些任意正数再归一化，所以故障不必表现为 NaN，而可能表现成“和为 1 的错误概率”。 低精度分支先把 `inT` cast 到 `work`，再原地处理 `work`，其数据流本来成立。正确修复不是给所有路径加初始化常量，而是把 helper 的读源和写目标显式分开：fp32 用 `(outT, inT)`，低精度仍用 `(work, work)`。

## 解决配方

代码证据来自 `operators/softmax/softmax.asc`；下面均为事件流中的完整函数体。before 是修复前未提交工作树中的符号全文，after 是修复后写入并完成上板验证的符号全文。

### before

`max_sub_exp_vf`（修复前，行段为完整函数边界）：

```cpp
__simd_vf__ inline void max_sub_exp_vf(__ubuf__ float *buf, uint32_t count)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> accReg;
    uint32_t oneElem = 1u; AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::Duplicate<float>(accReg, NEG_INF, scalarMask);

    // pass 1: reduce max
    uint32_t chunkCount = (count + VL - 1) / VL;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::RegTensor<float> srcReg;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, buf + off);
        AscendC::Reg::RegTensor<float> tmpReg;
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(tmpReg, srcReg, mask);
        AscendC::Reg::Max(accReg, accReg, tmpReg, scalarMask);
    }

    // pass 2: in-place sub(max) + exp. broadcast accReg lane0 per chunk.
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::RegTensor<float> srcReg;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, buf + off);
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::Duplicate<float>(maxReg, accReg, mask);
        AscendC::Reg::Sub(srcReg, srcReg, maxReg, mask);
        AscendC::Reg::Exp(srcReg, srcReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(buf + off, srcReg, mask);
    }
}
```

`CallMaxSubExp`（修复前，行段为完整函数边界）：

```cpp
__aicore__ inline void CallMaxSubExp(const AscendC::LocalTensor<float> &buf, int64_t count)
{
    asc_vf_call<max_sub_exp_vf>((__ubuf__ float *)buf.GetPhyAddr(), (uint32_t)count);
}
```

`KernelSoftmaxAR<T>::Compute`（修复前，行段为完整函数边界）：

```cpp
    __aicore__ inline void Compute()
    {
        auto inT = inQueue_.DeQue<T>();
        auto outT = outQueue_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            CallMaxSubExp(outT, r_);                // outT = exp(inT - max)
            AscendC::PipeBarrier<PIPE_V>();
            CallSumInvMuls(outT, r_);               // outT = outT / sum
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            auto work = workBuf_.Get<float>();
            CallCastIn<T>(work, inT, r_);
            AscendC::PipeBarrier<PIPE_V>();
            CallMaxSubExp(work, r_);
            AscendC::PipeBarrier<PIPE_V>();
            CallSumInvMuls(work, r_);
            AscendC::PipeBarrier<PIPE_V>();
            CallCastOut<T>(outT, work, r_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        outQueue_.EnQue(outT);
        inQueue_.FreeTensor(inT);
    }
```

### after

`max_sub_exp_vf`（修复后，行段为完整函数边界）：

```cpp
__simd_vf__ inline void max_sub_exp_vf(__ubuf__ float *dst, __ubuf__ float *src, uint32_t count)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> accReg;
    uint32_t oneElem = 1u; AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::Duplicate<float>(accReg, NEG_INF, scalarMask);

    // pass 1: reduce max over src
    uint32_t chunkCount = (count + VL - 1) / VL;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::RegTensor<float> srcReg;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, src + off);
        AscendC::Reg::RegTensor<float> tmpReg;
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(tmpReg, srcReg, mask);
        AscendC::Reg::Max(accReg, accReg, tmpReg, scalarMask);
    }

    // pass 2: dst = exp(src - max). broadcast accReg lane0 per chunk.
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::RegTensor<float> srcReg;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, src + off);
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::Duplicate<float>(maxReg, accReg, mask);
        AscendC::Reg::Sub(srcReg, srcReg, maxReg, mask);
        AscendC::Reg::Exp(srcReg, srcReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dst + off, srcReg, mask);
    }
}
```

`CallMaxSubExp`（修复后，行段为完整函数边界）：

```cpp
__aicore__ inline void CallMaxSubExp(const AscendC::LocalTensor<float> &dst,
                                     const AscendC::LocalTensor<float> &src, int64_t count)
{
    asc_vf_call<max_sub_exp_vf>((__ubuf__ float *)dst.GetPhyAddr(),
                                (__ubuf__ float *)src.GetPhyAddr(), (uint32_t)count);
}
```

`KernelSoftmaxAR<T>::Compute`（修复后，行段为完整函数边界）：

```cpp
    __aicore__ inline void Compute()
    {
        auto inT = inQueue_.DeQue<T>();
        auto outT = outQueue_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            CallMaxSubExp(outT, inT, r_);           // outT = exp(inT - max)
            AscendC::PipeBarrier<PIPE_V>();
            CallSumInvMuls(outT, r_);               // outT = outT / sum
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            auto work = workBuf_.Get<float>();
            CallCastIn<T>(work, inT, r_);
            AscendC::PipeBarrier<PIPE_V>();
            CallMaxSubExp(work, work, r_);         // work = exp(work - max), in-place
            AscendC::PipeBarrier<PIPE_V>();
            CallSumInvMuls(work, r_);
            AscendC::PipeBarrier<PIPE_V>();
            CallCastOut<T>(outT, work, r_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        outQueue_.EnQue(outT);
        inQueue_.FreeTensor(inT);
    }
```

## 设计决策

1. 选择显式 `dst, src` 契约，因为它同时覆盖“fp32 输入到独立输出”和“低精度工作区原地更新”，调用点直接表达数据所有权；
2. 没有选择把 `inT` 先复制到 `outT` 再沿用单参数 helper，因为这会多一次 UB 内搬运且仍隐藏 helper 的读写契约；
3. 没有把错误归因于 softmax 归约公式或容差，因为输出仍归一化，且仅改正源张量后定向探针即精确恢复；
4. 当前路线为 `safety_fallback`：正确性闭环完整，但整条三阶段实现相对 benchmark reference 较慢。

## 性能影响

在 950 的同一标准评测中，修复后的整条 correctness-first kernel 相对 benchmark reference 的平均速度约为 `0.18x`。该数值是完整实现与参照路线的成对相对比较，不能归因成这一个参数拆分的独立开销。

## 预防

- 对所有 `AllocTensor` 后立即进入 Reg helper 的调用点，沿数据流检查 helper 首次操作是写还是读；
- helper 同时存在原地与非原地用法时，用 `dst, src` 参数显式表达，允许调用者传同一张量实现原地；
- 定向探针若“输出和正常但元素排列/概率错误”，优先核对源张量，而不是先放宽精度阈值；
- 代码评审时把注释中的赋值关系与实参对照：`outT = f(inT)` 不应实现为只传 `outT`。

## 验证信号

来源是一次 softmax Ascend C Reg kernel 开发会话中对 AR 数据流的修复与上板回归；以下均为同一 950（Ascend950PR_957b / dav-3510）环境的工具输出。

- 修复前完整 NPU 套件为 `39 passed, 45 failed`（950，证据等级: tool_observed）；
- 修复后最小连续 fp32 探针结果与 golden 逐元素相等（950，证据等级: tool_observed）；
- 随后的完整套件为 `59 passed, 25 failed`，剩余失败全部落在另一条非末轴 ARA 路径，证明 AR 路径已恢复（950，证据等级: tool_observed）；
- 另一独立 ARA 根因修复后，完整套件为 `84 passed`；clean rebuild 再次为 `84 passed`，标准评测为 `20/20` 通过（950，证据等级: tool_observed）。

## 失效边界

- 此配方只修正 LocalTensor 源/目标契约，不处理跨 MTE/V pipeline 的队列同步；若输出为全 NaN 且只发生在 GM 行切片路径，应另查 TQue 所有权；
- 对已经写满有效值、明确原地更新的 `buf`，单参数 helper 并非天然错误；
- A2、A3 未验证；相同未初始化读取机制推断可迁移，但必须在目标平台重新编译和上板（未验证，推断可迁移）。

## 坏实践

- 把 `AllocTensor` 当成“已包含输入”的张量：它只分配缓冲，不提供数据初始化；
- 用输出和接近 1 证明 softmax 正确：任意有限正数经错误的归一化阶段也可能满足这一弱不变量；
- 通过调宽 rtol/atol 接受错误概率：本次修复没有改测试、golden 或容差。

## 误诊

无已验证的其他根因。归约公式、指数稳定化和容差不是本闭环的根因；它们没有形成独立失败—修复—验证链。

这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。
