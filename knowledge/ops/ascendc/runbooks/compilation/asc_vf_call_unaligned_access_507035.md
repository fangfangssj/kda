---
type: "Runbook"
title: "asc_vf_call VF内非对齐访存致507035"
description: "__simd_vf__ 函数内用逐元素偏移指针调 Reg::LoadAlign（非 32B 对齐）触发 AIV 异常 507035（SIGSEGV 139）；改为 kernel 侧预广播 + VF 扁平对齐循环，或标量传参 + VF 内标量广播后崩溃全消。"
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# asc_vf_call VF内非对齐访存致507035

## 现象

在 ascend950（arch35/dav-3510，CANN 9.2.0）regbase 自定义算子的 kernel 调试中：kernel 已能编译通过（build.sh 退出码 0），首次在 `tests/aclnn/` 下运行 `./run.sh` 做上板 ST 时，全部 46 个用例报 `[ACLNN_FAIL]`，每个用例前都输出 `ACL error 507035 at test_aclnn.cpp:...: aclrtSynchronizeStream(stream)`，run.sh 整体退出码 139（SIGSEGV）。测试覆盖 fp16/fp32/bf16 三个 dtype 的小 shape、大 shape 与特殊值切片，全部失败，无任何一例进入精度比对。kernel 的计算链是 CopyIn → Cast b16→fp32 → ReduceMax（AR pattern）→ SubExp → ReduceSum → DivCast → CopyOut，其中 Sub+Exp 与 Div+Cast 按要求实现为 `__simd_vf__` 函数、经 `asc_vf_call` 调用。当时 VF 函数采用"每行加载一个 m 标量再寄存器广播"的写法：`AscendC::Reg::LoadAlign(mReg, mAddr + i)` 直接对 `__ubuf__ T*` 做逐元素指针算术，随后 `AscendC::Reg::Duplicate(mReg, mReg, maskFull)` 广播，内层数据块用 `CreateAddrReg<T>(elemIdx, 1)`（stride=1 的绝对元素下标）。用 PRINTF 在 Process 各阶段打点后定位：日志能打出 `after CopyIn / after Cast / after ReduceMax / after StoreMax`，但永远到不了 `after SubExp`——崩溃点在第一个 `asc_vf_call`（SubExpVf）内部。

## 适用场景

- 工程形态：算子仓式（`operators/<Pkg>_package`）自定义算子 kernel 开发与上板 ST（aclnn run.sh）；涉及 `__simd_vf__` + `asc_vf_call` 的 regbase VF 编程。
- API 路径：`AscendC::Reg::LoadAlign/StoreAlign/Sub/Exp/Duplicate/CreateAddrReg/UpdateMask`；`asc_vf_call<Func>(...)`；`AscendC::Duplicate`（UB 级，kernel 侧）；`LocalTensor::GetValue`。
- 验证过的平台与 CANN 版本：Ascend 950（ascend950 / dav-3510）+ CANN 9.2.0。与 frontmatter `platforms` 一致。
- 适用粒度：platform_wide（dav-3510 regbase VF 编程通用）。
- 路线优先级：recommended_fastpath（对齐访问是对 VF 编程的硬约束，不是可选优化）。
- 明确不适用/未验证场景：直接 `__aicore__` 函数内的 UB API（非 VF）未观察到同类崩溃；其他 arch（非 3510/950）未验证（未验证，推断机制相同——对齐要求来自向量访存硬件语义）。

## 根因/解决配方

根因：`asc_vf_call` 调用的 `__simd_vf__` 函数体内，用**非对齐的逐元素偏移指针**访问 UB——`LoadAlign(mReg, mAddr + i)` 的字节偏移是 `i * sizeof(T)`（如 fp32 下 4 字节粒度），不满足 32 字节数据块对齐；叠加 `CreateAddrReg<T>(elemIdx, 1)` 这种 stride=1 的地址模式——在 dav-3510 上触发 AIV 核异常，表现为 `aclrtSynchronizeStream` 返回 507035、宿主进程 SIGSEGV(139)。消除法证据：保留 `mAddr + i` 加载的所有变体（自动分配 tmpBuf、按块 mask、AddrReg 地址、去显式模板参数）全部崩溃；用 UB API `AscendC::Sub/Exp` 替换 VF 可正常执行（证明崩溃确在 VF 函数体内）。

解决配方（两条均已验证）：

配方 A（kernel 侧预广播 + VF 扁平对齐循环）——把 m/s 标量在 VF **外**用 UB 级 `AscendC::Duplicate(tensor, scalar, count)` 广播成整 tile 大小的缓冲，VF 内只做一维对齐循环：

```cpp
// 调用侧（kernel 内，__aicore__）：预广播 m 到整 tile 缓冲 tmpBuf1
__aicore__ inline void DoSubExp(int64_t validARows)
{
    auto dataTensor = preInBuf_.Get<float>();
    auto mTensor = cacheBuf_.Get<float>();
    auto broadBuf = tmpBuf1_.Get<float>();
    // Pre-broadcast m per-row into tmpBuf1 (full tile size).
    for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
        float mVal = mTensor.GetValue(i);
        AscendC::Duplicate(broadBuf[i * td_->rUbFactorAlign], mVal,
                           static_cast<int32_t>(td_->rUbFactorAlign));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    // VF: data = exp(data - m_broadcast)
    uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactorAlign * td_->rUbFactorAlign);
    uint16_t repTimes = static_cast<uint16_t>(CeilDivU32(totalElems, VL_F32));
    asc_vf_call<BenchHarnessSoftmaxSubExpVf<float>>(
        (__ubuf__ float*)dataTensor.GetPhyAddr(),
        (__ubuf__ float*)broadBuf.GetPhyAddr(),
        totalElems, VL_F32, repTimes);
}

// VF：扁平 1-D 对齐循环（对齐 scale_custom 参考实现）
template <typename T>
__simd_vf__ inline void BenchHarnessSoftmaxSubExpVf(
    __ubuf__ T* dataAddr, __ubuf__ T* mBroadAddr,
    uint32_t totalCount, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> dataReg;
    AscendC::Reg::RegTensor<T> mReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        uint32_t remain = totalCount - (uint32_t)i * oneRepeatSize;
        mask = AscendC::Reg::UpdateMask<T>(remain);

        AscendC::Reg::LoadAlign(dataReg, dataAddr, aReg);
        AscendC::Reg::LoadAlign(mReg, mBroadAddr, aReg);
        AscendC::Reg::Sub(dataReg, dataReg, mReg, mask);
        AscendC::Reg::Exp(dataReg, dataReg, mask);
        AscendC::Reg::StoreAlign(dataAddr, dataReg, aReg, mask);
    }
}
```

配方 B（标量传参 + VF 内标量广播）——kernel 侧 `GetValue` 读出标量直接作为 VF 参数，VF 内用 `Reg::Duplicate<float>(reg, scalar, maskFull)` 标量广播（完整结构见同批参考结构卡 `softmax_regbase_vf_compute_chain_reference`），同样无崩溃。

崩溃版 before（VF 内逐元素偏移加载 + 寄存器间广播 + stride-1 AddrReg）：

```cpp
template <typename T>
__simd_vf__ inline void BenchHarnessSoftmaxSubExpVf(
    __ubuf__ T* dataAddr, __ubuf__ T* mAddr,
    uint32_t srcM, uint32_t srcK, uint32_t oneRepeatSize, uint16_t repeatTimesPerRow)
{
    AscendC::Reg::RegTensor<T> dataReg;
    AscendC::Reg::RegTensor<T> mReg;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskCnt;
    AscendC::Reg::AddrReg aReg;

    for (uint32_t i = 0; i < srcM; ++i) {
        AscendC::Reg::LoadAlign(mReg, mAddr + i);          // 非对齐：字节偏移 i*sizeof(T)
        AscendC::Reg::Duplicate(mReg, mReg, maskFull);     // 寄存器间广播

        uint32_t remain = srcK;
        for (uint16_t j = 0; j < repeatTimesPerRow; ++j) {
            uint32_t elemIdx = i * srcK + j * oneRepeatSize;
            aReg = AscendC::Reg::CreateAddrReg<T>(elemIdx, 1);   // stride=1 地址模式
            maskCnt = AscendC::Reg::UpdateMask<T>(remain);
            AscendC::Reg::LoadAlign(dataReg, dataAddr, aReg);
            AscendC::Reg::Sub(dataReg, dataReg, mReg, maskCnt);
            AscendC::Reg::Exp(dataReg, dataReg, maskCnt);
            AscendC::Reg::StoreAlign(dataAddr, dataReg, aReg, maskCnt);
            remain -= oneRepeatSize;
        }
    }
}
```

判别方法：PRINTF 打点逐阶段定位崩溃发生在第一个 `asc_vf_call` 之后；把 VF 调用临时替换为等价 UB API（`AscendC::Sub` + `AscendC::Exp`）若能跑通，即可确认崩溃在 VF 函数体内而非调用约定或缓冲区管理。

## 设计决策

- 为何选预广播而非在 VF 内加载广播：SDK 的 `SimpleSoftMaxGenericNDImpl` 在**直接 `__simd_vf__` 调用**里用 `LoadAlign(maxVreg, maxUb + i * blockStride)` + `Reg::Duplicate` 广播（blockStride 保证 32B 对齐），证明寄存器广播本身可用；但经 `asc_vf_call` 的用户 VF 更贴近 scale_custom 参考实现（扁平 1-D + `CreateAddrReg<T>(i, oneRepeatSize)`），且预广播把对齐责任收敛到 kernel 侧、VF 保持最简形态，排查面最小。
- `oneRepeatSize = VL_F32`（64 个 fp32 元素 = 256B）：向量寄存器宽度 `VECTOR_REG_WIDTH = 256` bit，一次 repeat 处理一个寄存器宽度的数据，地址按 `i * oneRepeatSize` 元素推进，天然对齐。
- 放弃的候选：按块 mask 修复、AddrReg 化、去显式模板参数——均保留非对齐 m 加载，全部无效；自动分配 tmpBuf 同样无效。这些失败尝试共同把根因收敛到"非对齐访存模式"而非"缓冲区大小/mask/模板参数"。
- 当前方案属 recommended_fastpath：对齐访问是 VF 编程的正确性前提。

## 系统信息

- 工具链：CANN 9.2.0，bisheng compiler (clang 15.0.5)；算子仓 `operators/build.sh <Pkg>_package`。
- 目标平台：ascend950（kernel 二进制 `op_impl/ai_core/tbe/kernel/ascend950/<snake>/`，arch35/dav-3510）。
- 相关文件：`op_kernel/arch35/bench_harness_softmax_kernel.h`（VF 函数与 Do* 调用点）、`op_kernel/bench_harness_softmax_apt.cpp`（tilingKey 分发）、`tests/aclnn/run.sh`（ST 入口，含 `aclrtSynchronizeStream` 报错点）。

## 验证信号

问题与验证来自同一次 softmax 类算子分支 kernel 的上板 ST 调试（PRINTF 逐阶段打点 + 每轮改码后全量重跑 run.sh）：

- 崩溃态：全部 46 用例 `[ACLNN_FAIL]` + `ACL error 507035 at aclrtSynchronizeStream`、run.sh 退出码 139（来源平台：950；证据等级：tool_observed）。
- PRINTF 定位：日志止步于 `after StoreMax`、无 `after SubExp`，崩溃在第一个 `asc_vf_call` 内（来源平台：950；证据等级：tool_observed）。
- 交叉验证：用 UB API `AscendC::Sub/Exp` 替换 VF 调用后 kernel 可跑过 SubExp/ReduceSum/StoreSum 阶段（来源平台：950；证据等级：tool_observed）。
- 修复后（配方 A）：507035 全部消失，17/46 precision PASS（fp32 全过、fp16/bf16 转为 elem-1 精度失败，属另一独立问题）（来源平台：950；证据等级：tool_observed）。
- 修复后（配方 B，另一轮重写）：无任何 AIV 异常（0 ACLNN_FAIL），fp32 大 shape 用例转 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定 dav-3510/ascend950 + `asc_vf_call` 路径；其他 arch 未验证（未验证，推断可迁移——对齐约束来自向量访存硬件语义）。
- 配方 A 额外占用一块整 tile 大小的广播缓冲（tmpBuf1），tile 很大时会挤压 UB 预算——UB 紧张时优先配方 B（标量传参，无额外缓冲）。
- 配方 A/B 都要求 m/s 在 kernel 侧先经 `GetValue`（Scalar 管道）读出——跨管道可见性需 `PipeBarrier<PIPE_ALL>`（Scalar 管道读 V 管道写需全管道屏障），本卡示例代码中的屏障已按该结论写为 PIPE_ALL。

**未决项**：

- 优先项：507035 AIV 异常的硬件级精确机制未验证——非对齐 VF 访存为何表现为运行时异常而非编译期拒绝；建议做最小复现（单条非对齐 LoadAlign 的 VF）定位异常来源。
- 触发点未单独隔离：`mAddr + i` 非对齐加载与 stride-1 AddrReg 在修复中同时移除，二者是否独立均可触发未验证。
- `Reg::Duplicate` 寄存器间广播（SDK 模式）在 `asc_vf_call` 场景 + 块对齐地址下已验证可用，但与标量广播的性能差异未测量。
## 坏实践

- 在 `__simd_vf__` 函数内对 `__ubuf__` 指针做逐元素偏移算术（`mAddr + i`）后直接传给 `Reg::LoadAlign`——"Align" 命名即对齐语义，逐元素偏移在 i≥1 时立即破坏 32B 对齐。
- `CreateAddrReg<T>(absoluteElemIdx, 1)`：把绝对元素下标当 repeat 序号、stride 设 1，绕开了 API 的"repeat 序号 × 向量长度"设计意图。
- 崩溃后盲目加大缓冲/加 mask/换模板参数写法——三类尝试在本轨迹中全部无效，根因在访存对齐。

## 误诊

- 曾怀疑 tmpBuf 不足（自动分配 sharedTmpBuffer）、mask 未按块截断、显式模板参数导致推导异常、`asc_vf_call` 调用约定不匹配——逐项实验排除（均为 tool_observed 的失败尝试）；这些假设都不能解释"UB API 同位替换即可跑通"。
- 曾假设"寄存器间 `Duplicate` 在 `asc_vf_call` 下不可用"（对比 scale_custom 无 Duplicate）——后被第二轮重写证伪：块对齐地址 + `Reg::Duplicate` 正常工作，真正的差异是对齐性（该中间假设已由后续 tool_observed 修正）。
