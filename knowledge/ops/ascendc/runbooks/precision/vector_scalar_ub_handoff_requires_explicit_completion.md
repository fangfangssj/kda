---
type: "Runbook"
title: "VF 归约落 UB 后 GetValue 读取陈旧标量"
description: "Vector 归约结果经 StoreAlign 落 UB 后立即由标量侧 GetValue 读取会出现竞态；保持归约和广播在同一 VF 内可消除错误。R8（CANN 9.2.0）：标量 SetValue 写入后原地 Rsqrt(dst,dst,1)（count=1）读回垃圾值并放大到 1e+35——改 Duplicate 广播+非原地 count=8 恢复量级、终版标量 Newton-Raphson 完全绕开 S 写→V 读。R9-R11：三条已验证修复路线——跨管道屏障全局升 PIPE_ALL（输出从 elem 0 全错恢复）、ReduceSum 聚合镜像取标量（V→S 同步）、HardEvent S_V/V_S 事件对按数据流向定点插（稀疏 FA 5 处交换点 10/10 确定性通过，事件对路线升级为已验证）；3510 上 count 模式 Reduce 的 isSetMask 被忽略，勿当同步依据。"
tags: ["task.debugging", "topic.compute.reduction", "topic.implementation.synchronization", "topic.programming_model.execution_unit.vector", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-17T04:15:00Z"
platforms: ["950"]
---

# VF 归约落 UB 后 GetValue 读取陈旧标量

## 现象

在 Ascend 950 `dav-3510` 的 Kernel 直调工程中，为绕过一次 RegTensor 广播编译问题，把末轴归约拆成三个 VF：第一个 VF 将最大值写入 32B UB 槽，`__aicore__` 包装层执行 `PipeBarrier<PIPE_V>()` 后用 `LocalTensor::GetValue(0)` 读回标量，再把它作为下一 VF 的 scalarValue。

该版本可以干净构建，但上板执行完整 pytest 时只有 21 个用例通过、25 个失败。失败集中在采用这条末轴路径的多种 dtype；部分输出为正常概率，部分行却出现千级到万级异常值，记录到的最大绝对差超过 `1e6`。使用纯 per-lane VF 的非末轴路径仍能通过，说明问题落在“VF 归约结果→UB→标量读取→下一 VF”的跨执行域交接，而不是 softmax 公式或测试容差。

## 适用场景

- 工程形态：Ascend 950 Kernel 直调，`__simd_vf__` 与 `__aicore__` 包装层混合编排。
- API 路径：`AscendC::Reg::Reduce`/`StoreAlign` → `PipeBarrier<PIPE_V>` → `LocalTensor::GetValue` → 下一次 `asc_vf_call`。
- 验证过的平台与 CANN 版本：Ascend 950（`dav-3510`）+ CANN 9.0.1，与 frontmatter `platforms` 一致。
- 适用粒度：`operator_family`；适用于“向量归约产生 lane 0 标量，再广播回整向量”的 RegBase 算子族。
- 路线优先级：`recommended_fastpath`；保持归约结果、广播和消费都在同一个 VF/寄存器域内。
- 明确不适用：已经有文档化 V→Scalar 同步原语并在目标版本实测通过的路径；纯 Vector→Vector UB 依赖不等同于本卡的 Scalar `GetValue` 读取。
- 增量适用（独立复现 R5–R7，CANN 9.2.0）：SIMT kernel 中标量循环（raw `__ubuf__` 指针或 `LocalTensor::GetValue/SetValue`）与 `asc_vf_call` 交替读写同一 UB 缓冲的 S↔V 双向交接——原始证据只覆盖 V 写→S 读方向，增量证据补齐 S 写→V 读方向。
- 路线分支：标量控制流无法收敛进单一 VF（如逐迭代读 gate 标量做决策）时，改用 `AscendC::HardEvent::S_V`/`V_S` 定点事件对——该分支经 R11（稀疏 FlashAttention online softmax，5 处交换点）10/10 确定性通过升级为已验证路线，见解决配方的增量节。

## 根因

`PipeBarrier<PIPE_V>` 只约束 Vector pipe 内部顺序，不能保证前一 VF 的 `StoreAlign` 已对 `__aicore__` 标量侧 `LocalTensor::GetValue` 可见，因而包装层可能读取陈旧 UB 标量并把错误的 max/sum 传播到整行输出。

增量（R5–R7）：AIV 上标量（S）与向量（V）是两条独立流水线，双向（S 写→V 读、V 写→S 读）访问对方刚写入的 UB 都必须显式事件同步；缺同步除 bitwise mismatch 外还可表现为间歇性 `ACL err 507035` 崩溃。`AscendC::HardEvent` 枚举（`$ASCEND_HOME_PATH/x86_64-linux/asc/impl/basic_api/kernel_event.h`）含 `S_V`/`V_S`（及 `S_MTE2`/`MTE2_S` 等 S↔MTE 组合），源码 `// src_dst` 注释表明方向语义。

增量（R9–R11，独立 trace）：`LocalTensor::GetValue` 在 `kernel_tensor.h` 标注 `__inout_pipe__(S)`（Scalar 管道），`AscendC::Duplicate`/`Exp`/`Muls`/`ReduceMax` 等向量 API 运行在 Vector 管道——V 写 → S 读（GetValue）与 S 写（SetValue）→ V 读两个方向的 UB 交接都无自动可见性保证。缺同步的形态谱系：输出从 elem 0 起全错且 0 运行时错误（R9，PIPE_V 屏障不覆盖 S 管道读）、多行用例 sum_to_one 偏离且逐次运行漂移而单行用例通过（R10，raw `asc_vf_call` 写后 GetValue）、间歇性精度失败（同用例 7 连跑 PASS=3 FAIL=4，R11，全屏蔽行纯 Scalar 写窗口踩中竞争）。判别指纹：**"从 elem 0 起全错 + 0 运行时错误 + 平凡用例（标量/长度 1/全零）也错"是同步/可见性类问题的指纹（数据从源头就是错的）；"首错位置在行内中段（elem 3/4）"才是数值精度类问题**（R9）。单行/单核用例通过不代表同步正确——跨 pipe 竞态在低压时序下会伪装成通过；多行多核用例加总量不变量（如 sum_to_one）是更灵敏的探针（R10）。

## 解决配方

不要把 Reg 归约标量经 UB 交给 AICore 标量侧读取；让归约、lane 0 广播和后续向量计算留在同一 `__simd_vf__`，使用 RegTensor→RegTensor `Duplicate<T, HighLowPart::LOWEST, MaskMergeMode::ZEROING>` 广播最低 lane。

### before

revision: 精度失败前完整函数快照；path: `operators/softmax/softmax.asc`；symbol: `softmax_last_max_vf`；lines: 106-126。

```cpp
__simd_vf__ inline void softmax_last_max_vf(__ubuf__ float *slotAddr, __ubuf__ float *xAddr, uint32_t count)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg, accMax, chunkMax;
    AscendC::Reg::MaskReg scalarMask;
    {
        uint32_t oneLen = 1;
        scalarMask = AscendC::Reg::UpdateMask<float>(oneLen);
    }
    AscendC::Reg::Duplicate<float>(accMax, -1.0e38f);  // max-neutral; see header note
    uint32_t chunkCount = (count + vectorLength - 1) / vectorLength;
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(chunkMax, xReg, mask);
        AscendC::Reg::Max(accMax, accMax, chunkMax, scalarMask);  // lane0: max(accMax, chunkMax)
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(slotAddr, accMax, scalarMask);
}
```

revision: 精度失败前完整函数快照；path: `operators/softmax/softmax.asc`；symbol: `KernelSoftmaxLast<T>::Compute`；lines: 312-353。

```cpp
__aicore__ inline void Compute(uint32_t off)
    {
        (void)off;
        auto xIn = inQueueX_.DeQue<T>();
        auto yOut = outQueueY_.AllocTensor<T>();
        auto maxSlot = maxSlotBuf_.Get<float>();
        auto sumSlot = sumSlotBuf_.Get<float>();
        __ubuf__ float *maxSlotAddr = (__ubuf__ float *)maxSlot.GetPhyAddr();
        __ubuf__ float *sumSlotAddr = (__ubuf__ float *)sumSlot.GetPhyAddr();

        if constexpr (std::is_same_v<T, float>) {
            __ubuf__ float *xAddr = (__ubuf__ float *)xIn.GetPhyAddr();
            __ubuf__ float *yAddr = (__ubuf__ float *)yOut.GetPhyAddr();
            SoftmaxLastMaxVF(maxSlotAddr, xAddr, r_);
            AscendC::PipeBarrier<PIPE_V>();
            float maxVal = maxSlot.GetValue(0);
            float negMax = -maxVal;
            SoftmaxLastSumVF(sumSlotAddr, xAddr, negMax, r_);
            AscendC::PipeBarrier<PIPE_V>();
            float sumVal = sumSlot.GetValue(0);
            float invSum = 1.0f / sumVal;
            SoftmaxLastNormVF(yAddr, xAddr, negMax, invSum, r_);
        } else {
            auto xFp32 = xFp32Buf_.Get<float>();
            __ubuf__ float *fp32Addr = (__ubuf__ float *)xFp32.GetPhyAddr();
            CastInVF<T>(fp32Addr, (__ubuf__ T *)xIn.GetPhyAddr(), r_);
            AscendC::PipeBarrier<PIPE_V>();
            SoftmaxLastMaxVF(maxSlotAddr, fp32Addr, r_);
            AscendC::PipeBarrier<PIPE_V>();
            float maxVal = maxSlot.GetValue(0);
            float negMax = -maxVal;
            SoftmaxLastSumVF(sumSlotAddr, fp32Addr, negMax, r_);
            AscendC::PipeBarrier<PIPE_V>();
            float sumVal = sumSlot.GetValue(0);
            float invSum = 1.0f / sumVal;
            SoftmaxLastNormVF(fp32Addr, fp32Addr, negMax, invSum, r_);
            AscendC::PipeBarrier<PIPE_V>();
            CastOutVF<T>((__ubuf__ T *)yOut.GetPhyAddr(), fp32Addr, r_);
        }
        outQueueY_.EnQue(yOut);
        inQueueX_.FreeTensor(xIn);
    }
```

### after

revision: 终版完整函数，由完整函数写入与后续精确 `-INFINITY` 编辑重建；path: `operators/softmax/softmax.asc`；symbol: `softmax_last_vf`；lines: 109-165。

```cpp
__simd_vf__ inline void softmax_last_vf(__ubuf__ float *yAddr, __ubuf__ float *xAddr, uint32_t count)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg, eReg, yReg, bcastMax, bcastSum;
    AscendC::Reg::RegTensor<float> accMax, accSum, chunkMax, newMax, diff, rescale, chunkSum;
    AscendC::Reg::MaskReg fullMask;
    {
        uint32_t fullLen = vectorLength;
        fullMask = AscendC::Reg::UpdateMask<float>(fullLen);
    }
    AscendC::Reg::MaskReg scalarMask;
    {
        uint32_t oneLen = 1;
        scalarMask = AscendC::Reg::UpdateMask<float>(oneLen);
    }

    // init: accMax = -inf (max-neutral; below all finite fp32 incl. extreme negatives, and
    // reproduces torch special-value behavior for +/-inf/NaN rows via natural IEEE arithmetic).
    AscendC::Reg::Duplicate<float>(accMax, -INFINITY);
    AscendC::Reg::Duplicate<float>(accSum, 0.0f);

    uint32_t chunkCount = (count + vectorLength - 1) / vectorLength;

    // pass 1: online max + sum. accMax/accSum are scalars in lane 0.
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(chunkMax, xReg, mask);
        AscendC::Reg::Max(newMax, accMax, chunkMax, scalarMask);          // newMax[0] = max(accMax, chunkMax)
        AscendC::Reg::Sub(diff, accMax, newMax, scalarMask);             // oldMax - newMax (<=0)
        AscendC::Reg::Exp(rescale, diff, scalarMask);
        AscendC::Reg::Mul(accSum, accSum, rescale, scalarMask);          // rescale running sum
        AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST,
                                 AscendC::Reg::MaskMergeMode::ZEROING>(bcastMax, newMax, fullMask);
        AscendC::Reg::Sub(xReg, xReg, bcastMax, mask);
        AscendC::Reg::Exp(eReg, xReg, mask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(chunkSum, eReg, mask);
        AscendC::Reg::Add(accSum, accSum, chunkSum, scalarMask);
        AscendC::Reg::Max(accMax, newMax, newMax, scalarMask);           // accMax = newMax (copy)
    }

    // pass 2: y = exp(x - max) / sum. Recompute exp (x persists in UB).
    AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST,
                             AscendC::Reg::MaskMergeMode::ZEROING>(bcastMax, accMax, fullMask);
    AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST,
                             AscendC::Reg::MaskMergeMode::ZEROING>(bcastSum, accSum, fullMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Sub(xReg, xReg, bcastMax, mask);
        AscendC::Reg::Exp(eReg, xReg, mask);
        AscendC::Reg::Div(yReg, eReg, bcastSum, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, yReg, mask);
    }
}
```

revision: 修复后完整函数快照；path: `operators/softmax/softmax.asc`；symbol: `KernelSoftmaxLast<T>::Compute`；lines: 294-311。

```cpp
__aicore__ inline void Compute(uint32_t off)
    {
        (void)off;
        auto xIn = inQueueX_.DeQue<T>();
        auto yOut = outQueueY_.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            SoftmaxLastVF((__ubuf__ float *)yOut.GetPhyAddr(), (__ubuf__ float *)xIn.GetPhyAddr(), r_);
        } else {
            auto xFp32 = xFp32Buf_.Get<float>();
            CastInVF<T>((__ubuf__ float *)xFp32.GetPhyAddr(), (__ubuf__ T *)xIn.GetPhyAddr(), r_);
            AscendC::PipeBarrier<PIPE_V>();
            SoftmaxLastVF((__ubuf__ float *)xFp32.GetPhyAddr(), (__ubuf__ float *)xFp32.GetPhyAddr(), r_);
            AscendC::PipeBarrier<PIPE_V>();
            CastOutVF<T>((__ubuf__ T *)yOut.GetPhyAddr(), (__ubuf__ float *)xFp32.GetPhyAddr(), r_);
        }
        outQueueY_.EnQue(yOut);
        inQueueX_.FreeTensor(xIn);
    }
```

### 增量：SIMT 标量循环无法同 VF 收敛时的 S↔V HardEvent 事件对（CANN 9.2.0，部分验证）

适用条件分支：能把归约/广播收敛进单一 VF 时，优先用上文主配方（recommended）；标量控制流必须留在 S 侧（如逐迭代读 gate 标量做决策）时，按数据流向在依赖边界成对插入定点事件。先取事件 ID，再在每个真实数据依赖边界成对插入同步（模式 verbatim）：

```cpp
int32_t evSV = (int32_t)GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
int32_t evVS = (int32_t)GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
```

标量写 → VF 读（gate 链：标量把 rawGate 写进 tmpBuf，VF 消费）：

```cpp
tmpPtr[0] = rawGate;
AscendC::SetFlag<AscendC::HardEvent::S_V>(evSV);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(evSV);
asc_vf_call<GateSafeAbsSqrtVF>(b1Fp32, tmpPtr, (uint16_t)1, gateRep);
```

VF 写 → 标量读（sigmoid 输出 gate 标量）：

```cpp
asc_vf_call<SigmoidVF>(tmpPtr, tmpPtr, (uint16_t)1, gateRep);
AscendC::SetFlag<AscendC::HardEvent::V_S>(evVS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVS);
float gateVal = tmpPtr[0];
```

要点：`SetFlag`+`WaitFlag` 必须成对、插在依赖边界处；方向按“生产者→消费者”选 S_V（S 生产 V 消费）或 V_S（V 生产 S 消费）。双向事件配对模式在 MTE 槽复用场景的同构应用（TQue + 双向 HardEvent 防陈旧 UB）见 [MTE2 跨流水未建依赖导致 UB 读到上次调用残值卡](direct_mte_slot_reuse_needs_bidirectional_hard_events.md)——区分条件：那边的事件对建在 MTE 与 V 之间，本节建在 S 与 V 之间；S 与 MTE 之间的事件方向语义（MTE2_S/MTE3_S，kernel 末尾同步不可由 stream sync 替代）见 [探针末尾 MTE2_S/MTE3_S 同步不可由 stream sync 替代卡](../compilation/mte2_s_mte3_s_stream_sync.md)。

### 增量：V→S 取值经 Reduce API 聚合镜像（R10，CANN 9.1.0，已验证）

适用条件分支：标量 carry/中间量由 raw `asc_vf_call`（`Reg::StoreAlign` 落 UB）维护、结束端需要 `GetValue` 读回标量时，把"最终标量的提取"换成经 Reduce API 的聚合读——`GetValue` 只在 ReduceMax/ReduceSum 这类 Reduce API 之后才有可见性保证（本例为 `Pattern::Reduce::AR` 聚合形态，R10 场景实测 3 次运行确定；Reduce API 的同步语义按调用形态区分，见失效边界）：

```cpp
    // 读取最终 s（归一化分母）。carryBuf[0..kCarryCount-1] 经 DuplicateVf(0)
    // 初始化且 RescaleSumVf/AddScalarVf 均匀作用，故全部元素 == s（镜像）。
    // 用 ReduceSum 聚合取出 s：Reduce API 内部完成 V→S 同步，使后续 GetValue 安全
    // （GetValue 跨 pipe 读 UB，仅 Reduce API 后有保证；raw asc_vf_call 后无同步）。
    auto sDst = sumBuf_.Get<float>();
    uint32_t sShape[2] = {1, kCarryCount};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(sDst, carry, sShape, true);
    float s = sDst.GetValue(0) / static_cast<float>(kCarryCount);
```

镜像性质的成立链（修复正确性的前提）：carry 初始化（`DuplicateVf` 写满 kCarryCount=64 个元素）+ 后续重缩放/累加（`RescaleSumVf`/`AddScalarVf`）都对全部元素施加同一标量运算 ⇒ 64 个元素恒等于 s；`ReduceSum([1,64]) = 64·s`，`GetValue` 后除以 64 还原。NaN 传播不受影响（64 个 NaN 求和再除仍是 NaN）。完整两趟在线 softmax 配方（carry 结构 + 该取值路线）见 [ChunkedR 在线两趟 softmax kernel 实现卡](../compilation/chunked_r_online_two_pass_softmax_kernel.md)。

### 增量：双向 HardEvent 事件对的完整验证（R11，CANN 9.2.0，已验证）

稀疏 FlashAttention online softmax（Scalar `SetValue/GetValue` 与 Vector `Mul/ReduceMax/ReduceSum/Duplicate/Exp/Muls` 直接交替读写同一批 UB tensor）在 5 个 Scalar↔Vector 交换点全部裸奔时出现非确定性精度失败（同用例 7 连跑 PASS=3 FAIL=4，失败态 MERE 1.95e-02~3.68e-02 / MARE 6.58e+03~3.89e+04，通过态恒 1.75e-04/4.96e-04）。修复 = 在每个交换点按数据流向成对插入 `SetFlag/WaitFlag<HardEvent::S_V 或 V_S>`（事件 ID 从 `GetTPipePtr()->FetchEventID(HardEvent::…)` 取），模式 verbatim：

```cpp
    // S→V 屏障：确保循环中 SetValue(Scalar 写 scores) 完成后，后续 Muls(Vector 读 scores) 才能执行。
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
```

```cpp
    // V→S 屏障：确保 ReduceMax(Vector 写 reduceOut) 完成后再 GetValue(Scalar 读)。
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float chunkMaxVal = reduceOut.GetValue(0);
```

（5 处交换点：S 写 scores→V 读（S_V）；ReduceMax 写→GetValue 读（V_S）；ReduceSum 写→GetValue 读（V_S）；全屏蔽跳过路径 `Duplicate(scores,0)` 清零→后续 `GetValue` 读（V_S）；标量 exp 借 8 元素 Vector `Exp` 写 expTmp→`GetValue` 读 rescale 系数（V_S）。）

两个关键平台事实（从 CANN 9.2.0 头文件 `x86_64-linux/ascendc/include/basic_api/impl/kernel_operator_vec_reduce_intf_impl.h` 核实）：

1. `ReduceMax` count 模式（Level 2）签名末参是 `calIndex`（是否计算索引），**不存在 `isSetMask` 参数**——"把 isSetMask 改 true 换取隐式屏障"的修复建议基于把末参误读为 isSetMask，按此修改无法修复缺陷；
2. `ReduceSum` count 模式虽有 `isSetMask` 模板参数，但在 `__NPU_ARCH__ == 3510` 分支里调用的是不带该参数的 `ReduceSumImpl<PrimType>`（2201 架构分支才透传）——即 **3510 上该参数被忽略，count 模式 ReduceSum 不提供 V→S 屏障**。

教训：同一函数族里的同类交换点要一次排查全——第一轮只补 3 处"最显眼"的交换点后仍 4/10 失败（漏掉跳过路径 `Duplicate`→`GetValue` 与辅助函数 `ScalarExp` 的 `Exp`→`GetValue`）；补齐全部 5 处后 10/10 且指标逐次一致。验收/回归对可疑用例做 ≥7 次单跑统计通过率，一次通过不等于稳定。

## 设计决策

选择把 online max/sum、lane 0 广播和归一化保留在一个 VF 内，避免 V→UB→Scalar→V 的未定义交接。被放弃的三 VF 方案虽然能编译，且 `PipeBarrier<PIPE_V>` 看似合理，但上板出现大面积真实错误；增加更多同类 barrier 没有可用证据能建立 Scalar 可见性。当前方案为 `recommended_fastpath`，三 VF + `GetValue` 路线为 `banned`。

增量：事件对路线选“按数据流向定点插同步”而不是全量插满——实测无差别插满反而恶化（见坏实践）；正确粒度是缓冲的生产者-消费者边界（RmsNormRowVF 写 B2→标量 dot 读 B2 需 V_S；标量写 tmpBuf→Gate VF 读需 S_V；conv 标量写 B1→SiLU VF 读需 S_V；AddRowVF 写 B0→标量 cast 读需 V_S）。

## 系统信息

- toolchain：CANN 9.0.1、毕昇编译器、pytest NPU 测试。
- SoC/NPU arch：Ascend 950、`dav-3510`。
- 相关文件：`operators/softmax/softmax.asc`、`operators/softmax/test_softmax.py`。
- 关键 API：`Reduce`、`StoreAlign`、`LocalTensor::GetValue`、`PipeBarrier<PIPE_V>`、RegTensor 广播 `Duplicate`。

## 预防

- Reg 归约标量若下一步仍是 Vector 计算，优先保留在 RegTensor 中并在同一 VF 广播。
- 不把 `PipeBarrier<PIPE_V>` 自动解释为 Vector 写 UB 对 Scalar `GetValue` 的可见性保证。
- 若只有某条“落 UB→GetValue”路线错误，而纯 VF 路线正常，先检查跨执行域同步，不要放宽精度容差。
- 测试需覆盖多个 dtype 和多行输入，陈旧标量竞态可能表现为部分行正确、部分行异常。
- 增量：写 SIMT kernel 时先画“缓冲 × 生产线”表——每个 UB 缓冲列出标量/VF 的读写点，凡跨 S↔V 的读写对都补 `S_V`/`V_S` 事件对（能收敛进同一 VF 的优先收敛）。
- 增量：code review 检查清单——`asc_vf_call` 的邻接行若是标量访问同一缓冲，必须有对应 SetFlag/WaitFlag 或同 VF 收敛。
- 增量：S pipe 与 MTE 管线之间的依赖同理须显式建立（TQue 的 V-pipe 事件不覆盖 S pipe，须 TBuf + PIPE_ALL 串行化），见 [标量 S pipe 误用 TQue 导致输出全常数卡](scalar_s_pipe_requires_explicit_mte_dependencies.md)——选用条件：标量直访 UB 的 kernel 在 TBuf/TQue 布局选型时。

## 验证信号

来源于一次 Ascend 950 RegBase softmax Kernel 开发会话：三 VF + UB 标量槽方案先成功构建，随后真实 NPU pytest 暴露 25 个失败；替换为同一 VF 内的 RegTensor 广播后重新执行构建、完整 pytest 和标准评测。

- 三 VF + `StoreAlign`/`PipeBarrier<PIPE_V>`/`GetValue` 版本能够构建，但上板 pytest 为 25 failed、21 passed，并观测到最大绝对差超过 `1e6`（来源平台：950；证据等级：tool_observed）。
- 错误集中在末轴路径，使用另一条纯 per-lane VF 路径的用例仍通过（来源平台：950；证据等级：tool_observed）。
- 改为同一 VF 内 RegTensor 广播后成功链接，完整 pytest 46/46 通过，标准评测 20/20 正确性通过（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：去除 vector 写后立即 Scalar GetValue 的跨管线读取后，四组定向 shape/dim 的 values_ok 与 indices_ok 均为 True（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：把跨 VF 的 UB槽→GetValue 标量交接改为同一 VF 内 RegTensor 广播后，修后 NPU pytest 46/46 通过（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3：消除 Vector→UB→Scalar 的未完成交接后，cann-bench 最终汇总为 20/20 精度通过（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4：把 upcast 后的 PIPE_V 改为 PIPE_ALL，并在 Scalar SetValue 后增加 PIPE_ALL 后，half ties/Inf/NaN、bf16 ties 与 fp32 ties 五组定向探针的 values/indices 均为 True（来源平台：950；证据等级：tool_observed）。

R5–R7 来自 BenchHarnessEngramGateFusion SIMT kernel（CANN 9.2.0，与上文 softmax 会话不同算子、不同版本、不同会话）的独立复现：标量循环与 `asc_vf_call` 交替读写同一 UB 的 S↔V 双向交接，mismatch 数为同一测试 `output X/64 elems differ` 形式的工具输出。

- 独立复现 R5（BenchHarnessEngramGateFusion SIMT kernel，CANN 9.2.0）：标量循环（raw `__ubuf__` 指针与 `GetValue/SetValue` 混用）与 `asc_vf_call` 交替读写同一 UB（tmpBuf/B0/B1/B2）、全程只有 MTE2_V/V_MTE2/V_MTE3 搬运同步而无任何 S↔V 事件时，E3_AllZero 12/64 elems differ、E1/E2/E4–E8 均 64/64 differ，最小调用通路间歇 `ACL err 507035`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R6：按数据流向定点插入事件对（标量写→VF 读边界 `SetFlag/WaitFlag<HardEvent::S_V>`、VF 写→标量读边界 `V_S`）后，E3_AllZero 从 12/64 降至 8/64 elems differ（56/64 正确），E2_Bf16Overflow60000/E7_RMSNormZero/E8_InfInject 与 F1_StandardPrefill 从 64/64 降至 63/64 differ——部分验证：mismatch 数下降但未全绿（来源平台：950；证据等级：tool_observed）。
- 独立复现 R7（负信号）：定点事件对后最小通路仍间歇 `ACL err 507035`，崩溃未在该会话根除，说明还存在未定位的 S↔V（或 S↔MTE）边界；无差别在所有 VF 调用前后插满 S_V/V_S 则把全零用例从 47/64 恶化到 64/64 elems differ，回退同步后恢复 37/64（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论针对 CANN 9.0.1、Ascend 950 `dav-3510` 上实测的 `StoreAlign`→`GetValue` 交接；其他版本未验证，推断可迁移但需独立 probe。
- 本卡不禁止一般 `GetValue`；只有当值由前一 VF/Vector pipe 刚写入且缺少明确 V→Scalar 同步契约时禁止采用。
- 同一 V pipe 内按程序顺序执行的 VF→VF 依赖不是本卡直接证明的范围。
- 增量：HardEvent 事件对路线为部分验证——仅证明 mismatch 数下降（E3 12/64→8/64 等），未全绿；该会话 22 个数值用例的剩余 mismatch 有其他独立根因（NaN/Inf 传播、conv_state_out 更新逻辑）。
- 增量：间歇性 507035 未在事件对路线下消除，本卡的 S↔V 边界清单不完整；事件对证据绑定 SIMT kernel + CANN 9.2.0。
- 增量：TQue/TPipe 队列路径的隐式同步语义未验证，推断可迁移但未验证。

## 坏实践

- 把 `PipeBarrier<PIPE_V>` 当成任意执行域之间的通用内存栅栏，然后立即在 AICore 标量侧 `GetValue`。
- 为让失败用例通过而放宽 rtol/atol，会掩盖千级异常输出，不能修复同步根因。
- 增量：无差别在所有 VF 调用前后插满 S_V/V_S（不看数据流向）——实测全零用例从 47/64 恶化到 64/64 elems differ，回退同步后恢复 37/64；盲目加同步既伤性能又可能引入错误顺序。
- 增量：把病因误判为“raw `__ubuf__` 指针在 NPU 上不能工作”并整体改写为 `GetValue/SetValue`——单独改写只把 E3 从 37/64 提到 12/64 differ，507035 依旧；访问方式不是病因，缺事件才是。

## 误诊

- 该现象不是普通浮点累积误差：错误量级超过概率范围，且同行中正常值和巨大异常值并存。
- 该现象也不是测试 harness 问题：替换 Kernel 内跨域交接后，同一测试和参考实现全部通过。
- 增量：独立复现会话中先后两个错误假设——①“raw 指针访问不工作”（agent 层推断，无工具证据，改 GetValue/SetValue 后问题未解）；②“同步本身有害”（回退同步后短暂变好，实为无差别插同步引入的错误顺序）。两者最终以“按数据流向定点插同步”的实测改善证伪。


### 独立复现 R8（MlaProlog RMSNorm，CANN 9.2.0）：标量 SetValue → 原地 Rsqrt(count=1) 读回垃圾值

纯 AIV kernel（RMSNorm 归一化系数 1/sqrt(meanEps)）把标量写入与 V 流水线原地读组合使用，输出爆炸到 1e+35 量级：

```cpp
// 失败写法（修复前，verbatim）：
        reduceDst.SetValue(0, meanEps);       // 标量写
        Rsqrt<float>(reduceDst, reduceDst, 1);  // V 流水线原地读+写, count=1
        PipeBarrier<PIPE_V>();
        rmsInv = reduceDst.GetValue(0);
```

两个隐患叠加：标量 `SetValue` 写入 LocalTensor 后立即用 V 流水线 API 原地读取（S 写→V 读无可见性保证，`PipeBarrier<PIPE_V>` 只管 V 流水线内部顺序），且 count=1 的原地操作在该平台上不产生正确结果——rmsInv 读回垃圾值并被逐元素放大（c_kv actual=-5.61e+35、k_rope 7.84e+35，MERE=1.1e+37）。与 R5-R7 的 S↔V 双向交接同族；未解实例（TopK 非末维 S 写 V 读不一致）见 `arch35_s_pipe_write_v_pipe_read_ub` 卡。

修复演进（两个可用替代，函数级 verbatim）：

```cpp
        // 中间修复：Duplicate 向量写 + 非原地 Rsqrt, count=8（量级恢复正常）
        float mean = sum / static_cast<float>(N);
        float meanEps = mean + epsilon;
        Duplicate<float>(xSq, meanEps, 8);  // xSq (tmpBuf) 在 ReduceSum 后不再需要
        PipeBarrier<PIPE_V>();
        Rsqrt<float>(reduceDst, xSq, 8);
        PipeBarrier<PIPE_V>();
        float rmsInv = reduceDst.GetValue(0);
```

```cpp
        // 终版：标量 Newton-Raphson 快速反平方根, 完全避免 V-pipeline API
        float mean = sum / static_cast<float>(N);
        float meanEps = mean + epsilon;
        union { float f; int32_t i; } u;
        u.f = meanEps;
        u.i = 0x5f3759df - (u.i >> 1);
        float rmsInv = u.f;
        rmsInv = rmsInv * (1.5f - 0.5f * meanEps * rmsInv * rmsInv);
        rmsInv = rmsInv * (1.5f - 0.5f * meanEps * rmsInv * rmsInv);
```

选型：终版选标量 Newton-Raphson——完全在标量 ALU 执行，从根上消除"标量写→V API 读"与"原地 count=1"两类隐患；精度核算：初值误差 ~3.4%，2 次迭代 ~1.4e-6，低于 bf16 输出量化步长（2^-8≈3.9e-3），对最终 bf16 输出无感知差异。Duplicate 版保留 V API 依赖且当时精度问题未闭环，无法区分残余误差来源。

- 标量 SetValue+原地 Rsqrt(count=1)：c_kv actual=-5.61e+35、k_rope 7.84e+35（MERE=1.1e+37）（来源平台：950；证据等级：tool_observed）。
- 改 Duplicate+Rsqrt(count=8)：c_kv actual=41.75、k_rope=-0.977——从 1e+35 垃圾恢复到合理量级（数值仍错，同步根因未修）（来源平台：950；证据等级：tool_observed）。
- 终版标量 Newton-Raphson（2 次迭代，叠加同步与 fp32 workspace 修复后）：c_kv/k_rope MERE=0.00e+00，总结果 100% PASS（来源平台：950；证据等级：tool_observed）。

边界增补："标量写→V API"失效的确切机制（可见性/发射顺序/in-place 语义）未隔离（优先项：隔离验证失效点）；Rsqrt 在非原地、count=8、无标量写前置时的正确性未单独回归；NR 初值对 denormal/极小 meanEps 的行为未验证（本算例 meanEps≥epsilon=1e-5 正常范围）；其他 NPU 架构未验证（推断可迁移）。


### 独立复现 R9（softmax 类算子 ST，CANN 9.2.0）：PIPE_V 屏障全局升 PIPE_ALL

多阶段混合管道 kernel（`StoreReduceToCache` 用 UB 级 `AscendC::Duplicate`（V 管道）把每行 m/s 标量写入 cacheBuf → `DoSubExp` 用 `LocalTensor::GetValue`（S 管道）读成标量传给逐行 VF）阶段间只插 `PipeBarrier<PIPE_V>`：kernel 正常跑完（0 ACLNN_FAIL）但 ST 46 用例中 45 个从 elem 0 就 PRECISION_FAIL，连标量输入、reduce 轴长度 1、全零输入这类平凡用例也全错。修复 = 跨管道读写点之间的屏障升级为全管道屏障（全局替换 27 处）：

```bash
sed -i 's/AscendC::PipeBarrier<PIPE_V>/AscendC::PipeBarrier<PIPE_ALL>/g' \
  operators/<Pkg>_package/op_kernel/arch35/<kernel>.h
```

关键位置 before/after：

```cpp
// before（PIPE_V：Scalar 管道 GetValue 可能读到 Vector 管道写入前的旧值）
StoreReduceToCache(validARows, 0);          // Duplicate 写 cacheBuf（V 管道）
AscendC::PipeBarrier<PIPE_V>();             // 只序 V 管道，不覆盖 S 管道读
DoSubExp(validARows);                       // GetValue 读 cacheBuf（S 管道）

// after（PIPE_ALL：跨管道可见性成立）
StoreReduceToCache(validARows, 0);          // Duplicate 写 cacheBuf（V 管道）
AscendC::PipeBarrier<PIPE_ALL>();           // 全管道序，S 管道读必然看到新值
DoSubExp(validARows);                       // GetValue 读 cacheBuf（S 管道）
```

边界：纯单管道数据流（写与读都在 V 管道，如 Duplicate 后紧接 VF LoadAlign）用 `PIPE_V` 屏障即可——修复前的历史版本正是这种单管道流且工作正常；问题是"屏障类型与数据流管道组合不匹配"，不是 `GetValue`/`Duplicate` 本身。PRINTF 打印中间标量与 golden 对照可直接看到 Scalar 管道读到的值不对。

### 独立复现 R10（在线两趟 softmax，CANN 9.1.0）：raw asc_vf_call 写 carryBuf 后 GetValue 读陈旧 s

多行多核用例 `[4,2048,50272]` 三个 dtype 全部 `sum_to_one` 不变量失败（sum=1.273149 fp16 / 1.273214 bf16 / 1.014345 fp32，且逐次运行漂移），单行长 R 用例（226 次 carry 迭代）却精确通过——"用例越简单越像是对的"是该坑最迷惑的地方。根因：`GetValue(0)` 是 S pipe 读，它前面最后一次对 `carryBuf_` 的写来自 raw `asc_vf_call<AddScalarVf>`（V pipe 的 `Reg::StoreAlign`），无同步的跨 pipe 读；多核/多行时序下 S pipe 抢跑读到陈旧 s（约真值的 78.5%），Pass2 除以错误分母。修复 = 取值改 ReduceSum 聚合镜像（见解决配方增量节），9/9 + 全量回归 32/32 + 连续 3 次运行确定。

### 独立复现 R11（稀疏 FlashAttention online softmax，CANN 9.2.0）：5 处交换点 HardEvent 全补齐

- 修复前：目标用例（B=1、S1=S2=512 因果方阵、fp16，含"全部 sparse indices 越过 causal 阈值"的全屏蔽行）单用例 7 连跑 PASS=3 FAIL=4；失败态 `MERE=1.95e-02 MARE=6.58e+03`（另观测 2.09e-02/1.51e+04、3.68e-02/3.89e+04），通过态恒 `MERE=1.75e-04 MARE=4.96e-04`；输入固定 seed 确定性生成，排除数据随机性；对照组（S1<S2 无全屏蔽行等）全部稳定通过（来源平台：950；证据等级：tool_observed）。
- 第一轮修复（只补 3 处交换点屏障）：10 跑 PASS=6 FAIL=4，失败 MARE 6.58e+03~3.89e+04 依旧——漏掉跳过路径与辅助函数里的同类冒险（来源平台：950；证据等级：tool_observed）。
- 第二轮修复（补齐全部 5 处 `SetFlag/WaitFlag<HardEvent::S_V/V_S>`）：10 跑 PASS=10 FAIL=0，且逐次 `MERE=1.75e-04 MARE=4.96e-04` 完全一致（确定性输出）（来源平台：950；证据等级：tool_observed）。
- 回归：全量 L0 24/24、L1 55/55 通过；后续全量 183/183 通过（来源平台：950；证据等级：tool_observed）。
- 代码差异：`git diff` 确认单文件 30 insertions / 2 deletions，5 处屏障（来源平台：950；证据等级：artifact_verified）。

R9 增量信号（softmax 类算子 ST，CANN 9.2.0）：

- 修复前：45/46 用例 `PRECISION_FAIL (first fail @ elem 0)`（含标量、reduce_axis_size_1、all_zero 等平凡用例），0 ACLNN_FAIL（kernel 正常跑完而非崩溃）（来源平台：950；证据等级：tool_observed）。
- 修复后（27 处 PIPE_V→PIPE_ALL）：`29/46 precision PASSED, 46 aclnn-OK, 0 aclnn-FAIL`——全部平凡用例与特殊值切片转 PASS，剩余失败收敛为独立精度问题（来源平台：950；证据等级：tool_observed）。

R10 增量信号（在线两趟 softmax，CANN 9.1.0）：

- 修复前：三 dtype `sum_to_one` 失败（sum=1.273149/1.273214/1.014345，逐次漂移），单行与特殊值用例通过，6/9（来源平台：950；证据等级：tool_observed）。
- 修复后（ReduceSum 聚合取 s）：9/9；全量回归 32/32 + 9/9，run.sh 退出码 0；连续 3 次独立运行确定（来源平台：950；证据等级：tool_observed）。

### R9-R11 边界与坏实践增补

- Reduce API 的隐式 V→S 同步语义按调用形态区分：R10 的修复依赖 `Pattern::Reduce::AR` 聚合形态（其场景实测 3 次运行确定）；count 模式的 `isSetMask` 在 3510 分支被忽略（头文件核实，见 R11 平台事实），不能作为同步依据。两形态结论都以当次实测/头文件为准，跨版本迁移须重新核对。
- R9 的"全局替换 PIPE_ALL"是正确性优先的粗粒度修复：对纯 V 管道点是过度同步，性能优化（降回单管道屏障）留待正确性达标后；哪些点可安全降回未逐一判定。
- R9 未做最小复现实验单独验证"PIPE_V 下 S 管道读到旧值"（机制推断基于 API 管道标注 + 现象学证据）。
- 坏实践增补：在注释里声称"ReduceSum(isSetMask=true 含隐式屏障)，Scalar 写安全"/"Vector 写后读，有隐式同步"却不核对当前架构的实现分支——R11 中两条注释都是错的，直接掩盖了缺屏障事实；修复竞争类缺陷只补"最显眼"的交换点（第一轮 3 处后仍 4/10 失败）；用单次运行通过判定非确定性缺陷已修复（应 ≥7~10 连跑且指标逐次一致）。
- 同 trace 相邻形态：LocalTensor 上先 `Duplicate` 清零再 `SetValue` 写入、再向量 `Exp` 的混用写法使 exp 输入恒 0（改纯 Vector 路径修复），见 [LocalTensor 混用 Vector/Scalar 写致输出全错卡](localtensor_vector_scalar_mixed_write.md)。
