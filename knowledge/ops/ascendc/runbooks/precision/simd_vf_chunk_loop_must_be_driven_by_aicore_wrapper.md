---
type: "Runbook"
title: "AR 跨块循环放在 __simd_vf__ 只执行首块"
description: "Ascend950 Reg kernel 把 AR 跨向量块状态机留在单次 __simd_vf__ 中时只完成首块；将循环上移至 __aicore__ wrapper，并在同址读改写调用之间加入 Vector barrier。"
tags: ["task.debugging", "topic.programming_model.execution_model.simd", "topic.implementation.synchronization", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-28T06:56:32Z"
platforms: ["950"]
---

# AR 跨块循环放在 __simd_vf__ 只执行首块

## 现象

Ascend950 上的 AscendC Softmax 最后一维 Reg 路径在归约长度不超过一个 float 向量寄存器宽度时结果正确；长度跨过该边界后，从第二个向量块开始出现未写、按块单独归一化或显著偏差。设备探针确认运行架构为 dav-3510，故障由真实 NPU 输出观测，不是 host 仿真结论。

复现步骤是在加载 `libop_softmax.so` 后调用 `torch.ops.op_softmax.softmax(x.npu(), -1)`、同步并与 `torch.nn.functional.softmax` 比较；对于全零输入，期望每项为 `1/R`，故障输出的首块却为 `1/VL`，说明实际只完成 `c=0`，而非期望的 `c=0..ceil(R/VL)-1`。

## 适用场景

- 工程形态：AscendC 自定义算子，Softmax 折叠为 `a0 == 1` 的最后一维 AR 路径。
- 关键 API：`asc_vf_call`、`Reg::LoadAlign`、`Reg::StoreAlign`、`PipeBarrier<PIPE_V>`。
- 已验证环境：Ascend950，`__NPU_ARCH__ == 3510`，轨迹环境中的 CANN 9.0.1。
- 迁移边界：只直接适用于“多个向量块连续读改写同一 UB 累加地址”的 Reg 实现；ARA 等其他工程形态只能迁移机制，仍需独立复现和验证。
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请用另一路线（见 `## 相关`）。本轨迹没有证明无 barrier 的等价高性能路线。
- 未验证：A2、A3、其他 CANN 版本以及不同 Reg 编程模型。
- 不适用：每次 Vector 调用只访问互不重叠地址，或归约始终不跨一个向量块的实现。

## 根因

AR 把跨向量块状态机留在单次 `__simd_vf__` 中，导致该调用只完成首块且同址累加无法在块间可靠提交。

## 解决配方

1. 将每个 `__simd_vf__` 函数收缩为只处理一个向量块，显式接收 `offset` 和 `remaining`。
2. 将块循环上移到 `__aicore__` wrapper，每个块执行一次 `asc_vf_call`。
3. 对 max 与 sum 这类同址读改写，在相邻调用之间加入 `PipeBarrier<PIPE_V>`；输出 pass 写不相交地址，不额外加入该 barrier。
4. 用恰好一个向量块、刚跨边界、多个完整块、尾块和长归约共同验证，不能只测单块。

代码来源：本轨迹操作者实际修改；路径 `operators/softmax/softmax.asc`。before 是本轮故障复现后的未提交状态；after 是本轮修复状态。轨迹仅恢复终版短提交 `b62b948`，未恢复 40 位 revision。以下均为完整函数的逐字重建。

### before

#### `ar_max_vf`

```cpp
__simd_vf__ inline void ar_max_vf(__ubuf__ float *maxAccAddr, __ubuf__ T *xAddr,
                                  uint32_t count, uint32_t chunkCount)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * VL;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> acc;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(acc, maxAccAddr);
        AscendC::Reg::RegTensor<float> x;
        if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
            AscendC::Reg::RegTensor<T> lo;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(lo, xAddr + offset);
            AscendC::Reg::Cast<float, T, SoftmaxCastTraits<T>::lowToHigh>(x, lo, mask);
        } else {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + offset);
        }
        AscendC::Reg::Max(acc, acc, x, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(maxAccAddr, acc, mask);
    }
}
```

#### `ar_sum_vf`

```cpp
__simd_vf__ inline void ar_sum_vf(__ubuf__ float *sumAccAddr, __ubuf__ T *xAddr,
                                  __ubuf__ float *maxSlotAddr, uint32_t count, uint32_t chunkCount)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(VL);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * VL;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        // re-broadcast max scalar each chunk
        AscendC::Reg::RegTensor<float> maxScalar;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(maxScalar, maxSlotAddr);
        AscendC::Reg::RegTensor<float> maxBroad;
        AscendC::Reg::Duplicate<float>(maxBroad, maxScalar, fullMask);
        AscendC::Reg::RegTensor<float> x;
        if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
            AscendC::Reg::RegTensor<T> lo;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(lo, xAddr + offset);
            AscendC::Reg::Cast<float, T, SoftmaxCastTraits<T>::lowToHigh>(x, lo, mask);
        } else {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + offset);
        }
        AscendC::Reg::Sub(x, x, maxBroad, mask);
        AscendC::Reg::Exp(x, x, mask);
        AscendC::Reg::RegTensor<float> acc;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(acc, sumAccAddr);
        AscendC::Reg::Add(acc, acc, x, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sumAccAddr, acc, mask);
    }
}
```

#### `ar_out_vf`

```cpp
__simd_vf__ inline void ar_out_vf(__ubuf__ T *yAddr, __ubuf__ T *xAddr,
                                  __ubuf__ float *maxSlotAddr, __ubuf__ float *sumSlotAddr,
                                  uint32_t count, uint32_t chunkCount)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(VL);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * VL;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> maxScalar;
        AscendC::Reg::RegTensor<float> sumScalar;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(maxScalar, maxSlotAddr);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sumScalar, sumSlotAddr);
        AscendC::Reg::RegTensor<float> maxBroad;
        AscendC::Reg::RegTensor<float> sumBroad;
        AscendC::Reg::Duplicate<float>(maxBroad, maxScalar, fullMask);
        AscendC::Reg::Duplicate<float>(sumBroad, sumScalar, fullMask);
        AscendC::Reg::RegTensor<float> x;
        if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
            AscendC::Reg::RegTensor<T> lo;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(lo, xAddr + offset);
            AscendC::Reg::Cast<float, T, SoftmaxCastTraits<T>::lowToHigh>(x, lo, mask);
        } else {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + offset);
        }
        AscendC::Reg::Sub(x, x, maxBroad, mask);
        AscendC::Reg::Exp(x, x, mask);
        AscendC::Reg::Div(x, x, sumBroad, mask);
        if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
            AscendC::Reg::RegTensor<T> y;
            AscendC::Reg::Cast<T, float, SoftmaxCastTraits<T>::highToLow>(y, x, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, y, mask);
        } else {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, x, mask);
        }
    }
}
```

#### `ArMax`

```cpp
__aicore__ inline void ArMax(const AscendC::LocalTensor<float> &maxAcc, const AscendC::LocalTensor<T> &x, int64_t count)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunkCount = (cnt + VL - 1) / VL;
    asc_vf_call<ar_max_vf<T>>((__ubuf__ float *)maxAcc.GetPhyAddr(),
                              (__ubuf__ T *)x.GetPhyAddr(), cnt, chunkCount);
}
```

#### `ArSum`

```cpp
__aicore__ inline void ArSum(const AscendC::LocalTensor<float> &sumAcc, const AscendC::LocalTensor<T> &x,
                             const AscendC::LocalTensor<float> &maxSlot, int64_t count)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunkCount = (cnt + VL - 1) / VL;
    asc_vf_call<ar_sum_vf<T>>((__ubuf__ float *)sumAcc.GetPhyAddr(),
                              (__ubuf__ T *)x.GetPhyAddr(),
                              (__ubuf__ float *)maxSlot.GetPhyAddr(), cnt, chunkCount);
}
```

#### `ArOut`

```cpp
__aicore__ inline void ArOut(const AscendC::LocalTensor<T> &y, const AscendC::LocalTensor<T> &x,
                             const AscendC::LocalTensor<float> &maxSlot, const AscendC::LocalTensor<float> &sumSlot,
                             int64_t count)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunkCount = (cnt + VL - 1) / VL;
    asc_vf_call<ar_out_vf<T>>((__ubuf__ T *)y.GetPhyAddr(),
                              (__ubuf__ T *)x.GetPhyAddr(),
                              (__ubuf__ float *)maxSlot.GetPhyAddr(),
                              (__ubuf__ float *)sumSlot.GetPhyAddr(), cnt, chunkCount);
}
```

### after

#### `ar_max_vf`

```cpp
__simd_vf__ inline void ar_max_vf(__ubuf__ float *maxAccAddr, __ubuf__ T *xAddr,
                                  uint32_t offset, uint32_t remaining)
{
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
    AscendC::Reg::RegTensor<float> acc;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(acc, maxAccAddr);
    AscendC::Reg::RegTensor<float> x;
    if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
        AscendC::Reg::RegTensor<T> lo;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(lo, xAddr + offset);
        AscendC::Reg::Cast<float, T, SoftmaxCastTraits<T>::lowToHigh>(x, lo, mask);
    } else {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + offset);
    }
    AscendC::Reg::Max(acc, acc, x, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(maxAccAddr, acc, mask);
}
```

#### `ar_sum_vf`

```cpp
__simd_vf__ inline void ar_sum_vf(__ubuf__ float *sumAccAddr, __ubuf__ T *xAddr,
                                  __ubuf__ float *maxSlotAddr, uint32_t offset, uint32_t remaining)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(VL);
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
    AscendC::Reg::RegTensor<float> maxScalar;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(maxScalar, maxSlotAddr);
    AscendC::Reg::RegTensor<float> maxBroad;
    AscendC::Reg::Duplicate<float>(maxBroad, maxScalar, fullMask);
    AscendC::Reg::RegTensor<float> x;
    if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
        AscendC::Reg::RegTensor<T> lo;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(lo, xAddr + offset);
        AscendC::Reg::Cast<float, T, SoftmaxCastTraits<T>::lowToHigh>(x, lo, mask);
    } else {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + offset);
    }
    AscendC::Reg::Sub(x, x, maxBroad, mask);
    AscendC::Reg::Exp(x, x, mask);
    AscendC::Reg::RegTensor<float> acc;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(acc, sumAccAddr);
    AscendC::Reg::Add(acc, acc, x, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sumAccAddr, acc, mask);
}
```

#### `ar_out_vf`

```cpp
__simd_vf__ inline void ar_out_vf(__ubuf__ T *yAddr, __ubuf__ T *xAddr,
                                  __ubuf__ float *maxSlotAddr, __ubuf__ float *sumSlotAddr,
                                  uint32_t offset, uint32_t remaining)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(VL);
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
    AscendC::Reg::RegTensor<float> maxScalar;
    AscendC::Reg::RegTensor<float> sumScalar;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(maxScalar, maxSlotAddr);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sumScalar, sumSlotAddr);
    AscendC::Reg::RegTensor<float> maxBroad;
    AscendC::Reg::RegTensor<float> sumBroad;
    AscendC::Reg::Duplicate<float>(maxBroad, maxScalar, fullMask);
    AscendC::Reg::Duplicate<float>(sumBroad, sumScalar, fullMask);
    AscendC::Reg::RegTensor<float> x;
    if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
        AscendC::Reg::RegTensor<T> lo;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(lo, xAddr + offset);
        AscendC::Reg::Cast<float, T, SoftmaxCastTraits<T>::lowToHigh>(x, lo, mask);
    } else {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + offset);
    }
    AscendC::Reg::Sub(x, x, maxBroad, mask);
    AscendC::Reg::Exp(x, x, mask);
    AscendC::Reg::Div(x, x, sumBroad, mask);
    if constexpr (SoftmaxCastTraits<T>::kNeedCast) {
        AscendC::Reg::RegTensor<T> y;
        AscendC::Reg::Cast<T, float, SoftmaxCastTraits<T>::highToLow>(y, x, mask);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, y, mask);
    } else {
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, x, mask);
    }
}
```

#### `ArMax`

```cpp
__aicore__ inline void ArMax(const AscendC::LocalTensor<float> &maxAcc, const AscendC::LocalTensor<T> &x, int64_t count)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    __ubuf__ float *maxAccAddr = (__ubuf__ float *)maxAcc.GetPhyAddr();
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    for (uint32_t off = 0; off < cnt; off += VL) {
        uint32_t rem = cnt - off;
        asc_vf_call<ar_max_vf<T>>(maxAccAddr, xAddr, off, rem);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
```

#### `ArSum`

```cpp
__aicore__ inline void ArSum(const AscendC::LocalTensor<float> &sumAcc, const AscendC::LocalTensor<T> &x,
                             const AscendC::LocalTensor<float> &maxSlot, int64_t count)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    __ubuf__ float *sumAccAddr = (__ubuf__ float *)sumAcc.GetPhyAddr();
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ float *maxSlotAddr = (__ubuf__ float *)maxSlot.GetPhyAddr();
    for (uint32_t off = 0; off < cnt; off += VL) {
        uint32_t rem = cnt - off;
        asc_vf_call<ar_sum_vf<T>>(sumAccAddr, xAddr, maxSlotAddr, off, rem);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
```

#### `ArOut`

```cpp
__aicore__ inline void ArOut(const AscendC::LocalTensor<T> &y, const AscendC::LocalTensor<T> &x,
                             const AscendC::LocalTensor<float> &maxSlot, const AscendC::LocalTensor<float> &sumSlot,
                             int64_t count)
{
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    __ubuf__ T *yAddr = (__ubuf__ T *)y.GetPhyAddr();
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ float *maxSlotAddr = (__ubuf__ float *)maxSlot.GetPhyAddr();
    __ubuf__ float *sumSlotAddr = (__ubuf__ float *)sumSlot.GetPhyAddr();
    for (uint32_t off = 0; off < cnt; off += VL) {
        uint32_t rem = cnt - off;
        asc_vf_call<ar_out_vf<T>>(yAddr, xAddr, maxSlotAddr, sumSlotAddr, off, rem);
    }
}
```

## 设计决策

选择 wrapper 循环是为了让每个块成为独立 Vector 调用；barrier 只放在确有前写后读依赖的 max/sum wrapper 中，避免给写互不相交输出的 pass 添加无依据同步。放弃“扩大 UB”是因为它不改变首块后不执行的输出模式，也放弃在 VF 内保留循环仅补注释的方案，因为它没有新增提交边界。当前方案明确属于 `safety_fallback`，正确性已经上板闭环，性能最优性未证明。

## 系统信息

- 产品平台：Ascend950
- NPU 架构：dav-3510，宏探针观测值 3510
- CANN：9.0.1（由轨迹中的构建环境路径恢复）
- 代码终版：仅恢复短提交 `b62b948`

## 预防

- 为向量宽度边界建立成对用例：单块正确不能代表多块正确。
- Review 中逐个标记 UB 地址是同址 RMW 还是不相交写，再决定 barrier。
- 保留确定性构造输入，使“分块独立归一化”和“尾块未写”可直接从输出模式辨认。

## 验证信号

本卡信号来自一次 Ascend950 Softmax kernel 开发中修复前后的真实 NPU 命令输出和 clean build；内部事件锚点仅保存在 manifest。

- 修复前的边界扫描显示单块结果正确，跨过向量宽度后 max_diff 非零；确定性输入进一步显示首块独立归一化、后续区域异常（来源平台：950；证据等级：tool_observed）。
- 六个函数按上述方式重构后 clean build 成功（来源平台：950；证据等级：tool_observed）。
- 相同边界扫描覆盖单块、跨边界、多块、尾块和长归约，全部得到 max_diff 为零（来源平台：950；证据等级：tool_observed）。

## 性能影响

该方案为正确性优先的安全路线；max/sum 每块一次 `asc_vf_call` 并带 Vector barrier，会增加调度与串行化开销。轨迹没有同一实现的受控无 barrier A/B，因此不写性能比例，也不声称性能最优。

## 失效边界

- 若后续工具链明确保证单次 `__simd_vf__` 内同址 RMW 循环的提交语义，需要重新做对照实验后才能收回 barrier。
- 若一个块的 `remaining` 大于实际寄存器容量，必须先修正分块，不能靠 mask 掩盖。
- 该修复不处理 UB 行首未对齐导致的运行时错误；那是独立地址契约问题。

## 坏实践

- 只验证一个向量块就宣称 AR 路径正确。
- 在 `__simd_vf__` 内继续循环读写同一 UB 累加器，仅补注释或扩大 buffer。
- 对所有 pass 无差别添加 barrier，掩盖真实依赖并扩大性能损失。
- 用放宽精度容差替代对块边界错误的修复。

## 误诊

最初容易把后续块异常归因于尾部越界或 UB 容量；但单块与跨边界的受控对比，以及首块独立归一化的输出模式，更直接指向块间提交/循环语义。扩大末尾 padding 不能解释第二块起始即失败。