---
type: "Runbook"
title: "Vector Sub 源 UB 非 32B 对齐触发错误 340"
description: "Ascend950/arch35 上用 AscendC::Sub Level 2 接口对 LocalTensor 做相邻元素差分时，若某个源是 LocalTensor::operator[] 的非零偏移切片（如 rpLocal[1]，偏移 1×sizeof(int32_t)=4 字节），起始地址不满足 Vector 指令 32B 对齐要求，运行时触发错误码 507035、plog 报 errcode:(340) The instruction access UB address is not aligned。判别=grep plog 取 errcode + 审查 Sub/Subs/Add 调用源是否为 operator[] 非零偏移切片 + 临时注释可疑 Sub 验证错误消失。"
tags: ["task.debugging", "topic.alignment", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-18T08:44:04Z"
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# Vector Sub 源 UB 非 32B 对齐触发错误 340

## 现象

在 Ascend950 (arch35) 上为稀疏算子（aclsparseCsr2coo）做 Kernel 侧向量化重构（用 `AscendC::Sub`/`Subs` Level 2 接口替代循环内 `GetValue` 逐元素读）时，第一版向量化方案编译通过，但运行单测 `L0_4x4_z` 立即失败：gtest 报错误码 `507035`（AICore 错误）。查 plog 日志（`~/ascend/log/debug/plog/plog-*.log`）得到精确报错：

```
errcode:(340) errorStr: The instruction access UB address is not aligned. subErrType: 0x4.
```

即「指令访问 UB 地址未对齐」。此时 Kernel 处于第一版修复状态：用 `Subs` + `Sub` Level 2（calCount）接口做差分预计算，`Sub` 的一个源是 `rpLocal[1]`（`LocalTensor::operator[](1)`，偏移 1 个元素）。

## 适用场景

- **工程形态**：kernel 直调（稀疏算子 Kernel 侧 `__aicore__` 函数内向量化重构）。
- **API 路径**：`AscendC::Sub` / `AscendC::Subs` Level 2 接口（`Sub(dst, src0, src1, count)` / `Subs(dst, src0, scalar, count)`）对 `LocalTensor<int32_t>` 的差分计算；`LocalTensor::operator[]` 做偏移切片。
- **验证过的平台与 CANN 版本**：Ascend950（arch35，`--soc=ascend950`），CANN 9.0.0-beta.2，bisheng clang 15.0.5；a2/a3 未验证。
- **适用粒度**：`operator_family`——所有在 Kernel 内对 `LocalTensor` 做相邻元素差分、需用 `Sub`/`Add` 向量化的稀疏/索引类算子。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；两次 DataCopyPad 对齐方案的逐字 after 代码与修后测试通过的闭环信号未从 trace 完整恢复。
- **不适用/未验证场景**：a2/a3 平台未验证（推断可迁移，Vector 32B 对齐是 Ascend 系列共性约束，但具体错误码可能不同）；Cube 管线不在本卡范围。

## 判别方法

**触发条件**（tool_observed 失败症状）：Kernel 编译通过但运行单测立即报 `507035`（AICore 错误），plog 出现 `errcode:(340) errorStr: The instruction access UB address is not aligned. subErrType: 0x4`。失败发生在引入 `Sub`/`Subs`/`Add` 等 Vector 二元/标量算术 Level 2 接口对 `LocalTensor` 做差分之后。

**可执行探针**：

1. 定位错误：grep plog 日志（`grep -iE "csr2coo|ERROR|aic.*error|507035|address" ~/ascend/log/debug/plog/plog-*.log`）取 `errcode` 与 `errorStr`，确认是否为 `340 The instruction access UB address is not aligned`。
2. 审查算术调用源：grep Kernel 代码中所有 `Sub(` / `Subs(` / `Add(` / `Adds(` / `Mul(` / `Muls(` 等 Vector 二元/标量算术 Level 2 调用，检查每个调用的 `src0` / `src1` / `dst` 是否为 `LocalTensor::operator[]` 的**非零偏移切片**（如 `rpLocal[1]`、`buf[2]`）。
3. 算对齐：对每个非零偏移切片，计算 `offset × sizeof(T) mod 32`（int32_t 时 sizeof=4）；若结果 ≠ 0，则该切片起始地址 = TBuf 基址 + 非零偏移，不满足 32B 对齐，即为根因。确认 `TBuf`/`TQue` 的 `InitBuffer` 分配的基址本身 32B 对齐（`Get` / `GetWithOffset(…, 0)` 返回的起始地址对齐），问题只在 `operator[]` 偏移。
4. 夹逼验证：把可疑 `Sub` 调用临时注释掉只留 `Subs`（或反向），重跑单测；若错误消失则确认是 `Sub` 的非对齐源触发。

**原始观测**（Ascend950，arch35，CANN 9.0.0-beta.2）：

- 运行 `L0_4x4_z` 单测：第一版 `Sub(nnzLocal, rpLocal[1], rpLocal[0], count)` 方案失败，错误码 `507035`（AICore 错误）。
- plog grep 命中 `errcode:(340) errorStr: The instruction access UB address is not aligned. subErrType: 0x4.`，定位到「UB 地址未对齐」。
- `Subs(startLocal, rpLocal, idxBase_, count)` 不触发（`rpLocal` offset=0 与 `startLocal` 起始均 32B 对齐），错误定位到 `Sub` 的 `rpLocal[1]` 源。
- 改用两次 `DataCopyPad` 从 GM 分别搬到两个对齐 buffer 后，全部 55 个测试通过（`[  PASSED  ] 55 tests.`，61159 ms total）。

**机制解释**（根因分析，属 interpretation）：`AscendC::Sub` Level 2 接口（`Sub(dst, src0, src1, count)`）底层 Vector 指令要求 `src0`/`src1`/`dst` 三个 LocalTensor 的起始地址均 32B 对齐（每 block 32 字节，即 8 个 int32）。`rpLocal` 通过 `rowPtrBuf_.GetWithOffset<int32_t>(rowPtrCountAligned, 0)` 获取，其起始地址由 `TBuf` 分配保证 32B 对齐。但 `rpLocal[1]`（`LocalTensor::operator[](1)`）返回的切片起始地址 = `rpLocal` 起始地址 + 4 字节（1 个 `int32_t`），4 mod 32 = 4 ≠ 0，不满足 32B 对齐。因此 `Sub(nnzLocal, rpLocal[1], rpLocal[0], count)` 中 `rpLocal[1]` 这个源触发 `340` 对齐错误。`Subs` 不触发是因为其源 `rpLocal`（offset=0）与 `startLocal` 均对齐。机制链不跳步：`rpLocal` 起始 = TBuf 基址（32B 对齐）→ `rpLocal[1]` 地址 = 基址 + 1×sizeof(int32_t) = 基址 + 4 字节 → 4 mod 32 ≠ 0 → 非 32B 对齐 → Vector Sub 读源地址时硬件报 `instruction access UB address is not aligned`（错误码 340）。calCount 只控制有效元素数（用 mask），不放宽地址对齐。

## 预防

- 在 Kernel 中对 `LocalTensor` 做任何 Vector 二元/标量算术（`Sub`/`Subs`/`Add`/`Adds`/`Mul`/`Muls` 等）时，**禁止用 `operator[]` 非零偏移切片作源**；若需相邻元素差分，改用两次 `DataCopyPad` 从 GM 分别搬到两个由 `TBuf` 分配的对齐 buffer，再做 `Sub`。
- 消除 `GetValue`（API-1/R1 合规）向量化时，务必同步检查源地址对齐约束，不要在消除逐元素串行时引入对齐错误。
- 编码规范层面，`LocalTensor::operator[]` 的非零偏移切片仅可用于标量逐元素读（`GetValue`，本身也被禁），不可作 Vector 算术 API 的源。

## 坏实践

- 用 `LocalTensor::operator[](非零偏移)` 作 Vector 算术 API 的源，假定 Level 2 calCount 接口会处理非对齐——实际底层 Vector 指令仍要求 32B 对齐起始地址，calCount 只控制有效元素数（用 mask），不放宽地址对齐。
- 消除 `GetValue` 时只想着「换成 Sub」而忽略源地址对齐，导致修掉 API 违规却引入运行时 340 错误。

## 验证信号

- 运行 `L0_4x4_z` 单测：第一版 `Sub(rpLocal[1])` 方案失败，错误码 `507035`（AICore 错误）（来源平台：950；证据等级：tool_observed）
- plog grep 命中 `errcode:(340) errorStr: The instruction access UB address is not aligned. subErrType: 0x4.`（来源平台：950；证据等级：tool_observed）
- 改用两次 DataCopyPad 对齐方案后全部测试：`[==========] 55 tests from 2 test suites ran. (61159 ms total) [  PASSED  ] 55 tests.`（来源平台：950；证据等级：tool_observed）

## 失效边界

- **a2 / a3 平台**：未验证，推断可迁移（Vector 指令 32B 对齐是 Ascend 系列共性约束，但具体错误码可能不同）；迁移时需重新跑全部测试确认。
- **非 int32_t dtype**：对齐要求仍为 32B，偏移切片的非对齐条件变为 `offset × sizeof(T) mod 32 ≠ 0`；int16_t / half 偏移 1 个元素 = 2 字节，2 mod 32 ≠ 0 同样触发。
- **Cube 管线**：本卡只覆盖 Vector 管线（Sub/Subs），Cube 管线对齐约束不在本卡范围。
## 复现增量：三例同约束扩展（行内半区 Muls / 树归约 data[half] / 逐元素 Cast dst）

本卡原判别方法覆盖"Sub/Subs 差分的 operator[] 非零偏移源"。另一独立轨迹并入三个不同 trace 的同根因实例，把约束面扩展到"任何向量 API 的非 32B 对齐源/目的子张量"：

**实例一（RoPE rotate_half 行内半区偏移，dav-3510/CANN 9.2.0）**：interleaved 布局下逐行 `Muls(rhTensor[i*D], inTensor[i*D+H], -1, H)` 与 `Muls(rhTensor[i*D+H], inTensor[i*D], 1, H)`——后半起点偏移 `i*D+H` 元素在 fp32 H=2 时 ≡ 2 (mod 4)，永非 8 的倍数，逐行每次 Muls 都在非对齐地址上执行。通过/崩溃按 `H×sizeof(T)` 是否 32B 倍数精确分界（D=128 fp16 H=64→128B 通过；D=4/2/6 全崩）。修复为 planar 半区分解（利用 cos/sin tile 性质把 rotate_half 代数消解为两个连续平面缓冲的线性组合，向量 API 全部从 TBuf 基址出发）：

```cpp
    // 修复后（planar 半区分解 + flat 向量链，节选）
    AscendC::Muls(ns, s, ComputeT(-1.0), static_cast<int32_t>(totalElems));   // ns = -sinH
    AscendC::Mul(o1, q1, c, static_cast<int32_t>(totalElems));               // out1 = q1*c
    uint16_t rep = static_cast<uint16_t>(CeilDivU32(totalElems, kVecLen));
    asc_vf_call<MulAddVF<ComputeT>>(
        (__ubuf__ ComputeT*)o1.GetPhyAddr(), (__ubuf__ ComputeT*)q2.GetPhyAddr(),
        (__ubuf__ ComputeT*)ns.GetPhyAddr(), totalElems, kVecLen, rep);      // + q2*(-s)
```

该实例同时澄清：**Level 0（mask/repeatParams）跨步形式同样不豁免对齐约束**——约束针对访问地址本身；行 pad 到 32B 也只解决行起点不解决半区起点。

**实例二（Max 树归约 data[half]，arch35/CANN 9.1.0）**：int32 AR 路径用 `Max<int32_t>(data, data, data[half], pairs)` 两两归并，`half = (n+1)/2` 逐轮折半——n=16→half=8（32B 对齐安全）→n=8→half=4（16B）→…→n=2→half=1（4B 偏移必错）。plog 566 处同码同 PC、多核同时报（系统性指令级问题，非数据相关）。修复为标量 GetValue 循环归约（逐元素读无对齐要求；每行归约结果只消费 1 次、chunk 长度千级，标量代价可忽略）：

```cpp
static __aicore__ inline int32_t ReduceMaxTreeInt32(
    LocalTensor<int32_t>& data, uint32_t count)
{
    // Scalar GetValue-based reduction: avoids non-32B-aligned data[half] access
    // in vectorized Max-tree (UB alignment error 340 on arch35).
    int32_t maxVal = data.GetValue(0);
    for (uint32_t i = 1; i < count; i++) {
        int32_t val = data.GetValue(i);
        if (val > maxVal) {
            maxVal = val;
        }
    }
    return maxVal;
}
```

**实例三（逐元素 Cast 目的地址，arch35/CANN 9.1.0）**：FP16/BF16 none 模式逐元素 `Cast(outLocal[outBufIdx], floatSrc, RoundMode::CAST_ROUND, 1)`——目的地址随 `outBufIdx++` 递增，FP16 每元素 2 字节，写到奇数下标时目的地址仅 2 字节对齐，大 shape 必跨到非对齐下标触发 507035（错误组合画像：仅"窄 dtype + 逐元素输出 + 大 N"崩，同 shape 的 FP32/mean/sum 都过——FP32 SetValue 直写与单标量 OutputScalar 天然对齐）。修复为 FP32 累积缓冲 + 满批对齐 Cast（`outBatchSize = 32/sizeof(T)`，FP16 为 16，整批恰 32B 对齐，Cast 的 src/dst 都落在各自缓冲位置 0）：

```cpp
        // FP16/BF16: accumulate in float buffer, batch Cast when full
        auto floatAccum = floatOutBuf.Get<float>();
        floatAccum.SetValue(outBufIdx, lossVal);
        ...
        if (outBufIdx == outBatchSize_) {
            // Cast float buffer → output buffer (both at position 0, 32B aligned)
            Cast(outLocal, floatAccum, RoundMode::CAST_ROUND,
                 static_cast<int32_t>(outBatchSize_));
            // V → MTE3 sync (TBuf doesn't auto-sync)
            event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
            outCopyParams.blockLen = outBatchSize_ * sizeof(T);
            DataCopyPad(outputGM[outOffset], outLocal, outCopyParams);
            outOffset += outBatchSize_;
            outBufIdx = 0;
        }
```

**独立 trace 新增验证信号**（三个不同 trace，均为 tool_observed）：

- RoPE 实例：修复前 ST 10/11 数值用例 `ACL error 507035`，plog `errcode:(340) The address for VEC to access UB is not aligned. subErrType: 0x4`、`Vector Core kernel execution failed, retCode=0x31`；唯一通过用例 D=128 fp16（H×sizeof=128B 对齐）；planar 分解后同套件不再出现 340（该轮剩余失败转为 MTE3 对齐问题，另卡；再修后 12/12 PASS）（来源平台：950；证据等级：tool_observed）。
- 树归约实例：int32 AR 用例 100% 崩、float 用例不崩（走 ReduceMax Level 2 不经 Max 树）；plog 566 处 `error code = 340` 多核同一 PC；标量归约替换后 int32 AR 全部 bitwise-equal，ST 由 22/101 升至 75/101，最终 101/101（来源平台：950；证据等级：tool_observed）。
- Cast 实例：仅"FP16/BF16 + none + 大 N"组合 507035（如 (1024,2048) FP16 none），同 shape FP32/mean/sum 全过；改批量 Cast 后同用例 PASS（MERE=1.79e-04 < 阈值 9.77e-04），全量 105/105 通过（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界**：树归约这类"half 逐轮变化"的模式天然不安全（尾轮 half=1 必然非 8 倍数），直接标量收尾或按对齐块设计；dtype 对齐阈值按 `32/sizeof(T)` 换算（fp32=8、fp16/bf16=16、int32=8、int64=4）。逐元素 Cast 的"小 N 不崩"是概率性暴露（N 不足时写位置可能恰好全在对齐下标内），review 层面一律禁止 `Cast(可变下标目的地址, …, 1)` 形态。
