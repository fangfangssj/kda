---
type: "Runbook"
title: argmax NaN输出0或2^32：calIndex索引无效
description: 含 NaN 输入做 argmax 时，AscendC ReduceMax(calIndex=true) 返回 max=NaN 但 index 位模式为 0xFFFFFFFF（无效），且 IEEE 754 下 NaN>x 恒 false 使标量合并丢弃 NaN；需位判 NaN + 标量扫描恢复首个 NaN 索引 + NaN 优先合并。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T00:49:13Z'
platforms: ["950"]
---

# argmax NaN输出0或2^32：calIndex索引无效

## 现象

自定义 argmax 算子（BenchNoHarnessArgMax，Ascend950PR/arch35，CANN 9.1.0）迭代一 ST Real 模式（NPU 执行，fp32 + 尾轴归约 AR 路径）出现一个精度失败用例：

```
[Real] L0_031: 1D fp32 contains NaN
  shape=[1000], dim=0, keepdim=false, special=nan
  不匹配 [0]: 期望=333, 实际=0
  [FAIL] 不匹配 1/1 (mismatch-rate=1.000000)
```

输入是 [1000] 的 fp32，位置 333 起放了 NaN（torch.argmax 语义：NaN 视为最大值，返回首个 NaN 的索引 333）。当时 kernel 状态：`ProcessFloatAR()` 用 `ReduceMax(dstLocal, inData, tmpLocal, chunkLen, true)`（calIndex=true）逐 chunk 求 max+索引，chunk 结果以 `if (chunkMax > globalMax)` 标量合并，最后 `float(globalIdx) → Cast<int64_t>` 输出。除 NaN 用例外，全 same/inf/普通值用例全部 bitwise-equal 通过。

修复过程出现两次不同的错误输出，构成完整判别链：

1. 只修标量合并（NaN 优先）后：`期望=333, 实际=4294967296`（= 2^32）。
2. 位级反推：4294967296 = float(0xFFFFFFFF) 就近舍入为 4294967296.0 后 Cast int64 的结果——说明 ReduceMax 在 chunk 含 NaN 时返回的 index 位模式是 0xFFFFFFFF（无效哨兵），不是首个 NaN 索引。

## 适用场景

- 工程形态：AscendC kernel（`__aicore__`）实现 argmax/带索引归约，要求对齐 torch.argmax 的 NaN 语义（NaN > +inf > 有限值，首个 NaN 胜出）。
- API 路径：`AscendC::ReduceMax<float>(dst, src, sharedTmp, count, true)`（calIndex 通道，dst[0]=max、dst[1]=index 位模式）；标量 `LocalTensor::GetValue`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——ReduceMax 的 NaN 行为是 3510 向量归约指令语义，与本算子无关；同平台所有用 calIndex 的 kernel 适用。
- 路线优先级：recommended_fastpath（NaN chunk 走标量扫描属非热路径的正确性兜底）。
- 明确不适用 / 未验证场景：fp16/bf16 输入的 NaN 扫描（本卡修复只覆盖 Cast→float 之后的 fp32 通道，fp16/bf16 的 NaN 在设计上列为已知限制）；NaN 与 +inf 共存时"首个 NaN 优先于首个 +inf"的语义未构造用例验证；其他 NPU_ARCH（非 3510/950）上 ReduceMax 的 NaN index 行为未验证。

## 根因

两个叠加缺陷，缺一即错：

1. **ReduceMax(calIndex=true) 的 NaN index 无效**：chunk 含 NaN 时 dst[0]=NaN（NaN 传播正确），但 dst[1] 的位模式是 0xFFFFFFFF，不是首个 NaN 的索引。证据链：仅修标量合并后输出 4294967296 = 2^32，恰为 `static_cast<float>(0xFFFFFFFFu)` 就近舍入到 2^32 再 Cast<int64_t> 的值（float 在 2^32 邻域 ULP=512，4294967295 必舍入为 4294967296.0）。
2. **IEEE 754 NaN 比较恒 false**：标量合并 `if (chunkMax > globalMax)` 中 `NaN > -inf` 为 false，NaN 永远不更新 globalMax/globalIdx——首版 kernel 因此输出初始值 0。

## 解决配方

在 kernel 头文件（`op_kernel/arch35/<op>.h`）加位判 NaN helper（不依赖编译器 fast-math 假设，比 `v != v` 更稳）：

```cpp
// Bitwise NaN test for fp32 (IEEE 754: exponent all-1s && mantissa non-zero).
// Robust against compiler fast-math assumptions.
static __aicore__ inline bool IsNanFp32(float v) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&v);
    uint32_t bits = *p;
    return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
}
```

**before**（chunk 结果提取与合并，输出 0 的版本）：

```cpp
            // Extract results
            float chunkMax = dstLocal.GetValue(0);
            float idxFloat = dstLocal.GetValue(1);
            uint32_t chunkArgIdx;
            // Reinterpret float bits as uint32_t (index stored as float bit pattern)
            const float* idxPtr = &idxFloat;
            chunkArgIdx = *reinterpret_cast<const uint32_t*>(idxPtr);

            // Scalar merge: first-max semantics (only update if strictly greater)
            if (chunkMax > globalMax) {
                globalMax = chunkMax;
                globalIdx = static_cast<uint32_t>(chunkStart) + chunkArgIdx;
            }
```

**after**（最终版：NaN chunk 标量扫描恢复首个 NaN 索引 + torch.argmax 合并语义）：

```cpp
            // Extract results
            float chunkMax = dstLocal.GetValue(0);
            float idxFloat = dstLocal.GetValue(1);
            uint32_t chunkArgIdx;
            // Reinterpret float bits as uint32_t (index stored as float bit pattern)
            const float* idxPtr = &idxFloat;
            chunkArgIdx = *reinterpret_cast<const uint32_t*>(idxPtr);

            bool chunkIsNan = IsNanFp32(chunkMax);

            // ReduceMax(calIndex=true) returns an INVALID index (bit pattern 0xFFFFFFFF)
            // when the chunk contains NaN (verified on DAV_3510: dst[1] reinterprets to
            // 0xFFFFFFFF, producing the 4294967296 seen before this fix). Recover the
            // first NaN index by scanning the chunk input. NaN is a rare, non-hot path,
            // so a scalar GetValue scan is acceptable here; NaN-free chunks are unaffected.
            if (chunkIsNan) {
                uint32_t firstNan = 0;
                bool found = false;
                for (uint32_t i = 0; i < chunkLen; i++) {
                    if (IsNanFp32(inData.GetValue(i))) {
                        firstNan = i;
                        found = true;
                        break;
                    }
                }
                chunkArgIdx = found ? firstNan : 0;
            }

            // Scalar merge: torch.argmax semantics
            //   - NaN is the largest (NaN > inf > finite > -inf)
            //   - first-max wins: once globalMax is NaN, never update (keep first NaN)
            //   - otherwise update only on strictly greater (first-max wins)
            bool globalIsNan = IsNanFp32(globalMax);
            bool update = false;
            if (globalIsNan) {
                update = false;                      // already hold first NaN
            } else if (chunkIsNan) {
                update = true;                       // first NaN encountered
            } else if (chunkMax > globalMax) {
                update = true;                       // strictly greater
            }
            if (update) {
                globalMax = chunkMax;
                globalIdx = static_cast<uint32_t>(chunkStart) + chunkArgIdx;
            }
```

多 chunk 正确性：NaN 在后面 chunk 时，前面无 NaN chunk 已设有限最大值，`chunkIsNan && !globalIsNan` 仍会更新为 NaN 索引；globalMax 一旦为 NaN 不再更新，保住"首个 NaN"。NaN-free chunk 完全不走扫描，性能不受影响。

输出侧无需改动（`idxLocal.SetValue(0, static_cast<float>(globalIdx)); Cast<int64_t, float>(... CAST_ROUND, 1)`），因为修复后 globalIdx 是真实索引。

## 设计决策

- 否决的候选方案：
  1. **NaN→+inf 预处理 + ReduceMax**（Compare(EQ,x,x) 生成非 NaN mask + Select 标量模式替换）：Select 模式1/2 需预留 8K UB 临时空间且签名不带 sharedTmpBuffer，UB 预算与隐式分配不可控；且 NaN 与 +inf 共存时语义错（首个 +inf 可能抢先），弃用。
  2. **两阶段 ReduceMax**（NaN 位置构造 ±FLT_MAX 再归约）：需要构造索引数组 + 第二次归约，复杂度高于标量扫描，收益只在 NaN 密集的热路径，而 NaN 输入本身是罕见路径，弃用。
- 采用"标量 GetValue 扫描"的理由：只在 chunkMax 为 NaN 时触发（非热路径），不改正常路径的数据流与 UB 布局；正确性直接、可逐位验证。
- IsNanFp32 用位判而非 `v != v`：不受编译器 fast-math/符号化假设影响，与 kernel 内既有的 `reinterpret_cast` 位模式读取风格一致。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510/arch35），CANN 9.1.0；TTK 不可用，ST 为自建 C++ aclnn 直调（tests/st，Real 模式 NPU 执行，CPU golden 比对 bitwise-equal）。
- 相关文件：`op_kernel/arch35/bench_no_harness_arg_max.h`（ProcessFloatAR 的 chunk 循环与标量合并）、`tests/st/test_aclnn_bench_no_harness_arg_max.cpp`（L0_031 用例与 golden）、`issues/issue_20260717_reducemax_nan_index_1.md`（问题记录）。

## 预防

- 用 ReduceMax(calIndex) 做 argmax 类算子时，先确认目标框架的 NaN 语义；要求 torch.argmax 语义就必须单独处理 NaN（硬件 calIndex 不给有效 NaN 索引）。
- 标量/向量合并处凡出现 `a > b` 的浮点比较，检查 NaN 路径：IEEE 754 下 NaN 参与的比较恒 false，"最大值"会被静默丢弃。
- 输出值出现 4294967296（2^32）这类"哨兵值舍入"特征时，优先怀疑某 uint32 通道被 0xFFFFFFFF 污染，而不是算子逻辑错。

## 验证信号

以下信号来自该算子迭代一联调会话（ST Real 模式 NPU 实测 + UT 回归）：

- 修复前：L0_031 `期望=333, 实际=0`，mismatch 1/1（来源平台：950；证据等级：tool_observed）。
- 第一轮（只修合并）后：L0_031 实际输出 4294967296（=2^32，tool_observed）；"dst[1] 位模式为 0xFFFFFFFF"是由该输出值经 float 舍入链（0xFFFFFFFF→2^32→int64）反推的机制结论（来源平台：950；证据等级：tool_observed，位模式结论为输出值推导）。
- 最终修复后：L0_031 `[PASS] bitwise-equal, 1 个元素完全一致`（输出 333）；ST 执行 10/10 全通过（跳过 4 条 ARA 属迭代二范围）（来源平台：950；证据等级：tool_observed）。
- 无回归：修复后重跑 UT，op_host 27/27 + op_api 10/10 = 37/37 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- ReduceMax NaN index=0xFFFFFFFF 的行为在 DAV_3510/CANN 9.1.0 实测（经输出值反推）；其他 arch/版本未验证（未验证，推断同族指令行为一致）。
- 标量扫描只在 chunk 级触发；若输入大面积是 NaN（每个 chunk 都扫满），扫描开销线性放大——热路径大量 NaN 属罕见，未做性能评估。
- NaN 与 +inf 共存场景（L1 用例，迭代三范围）未验证。
- fp16/bf16 原生 NaN 通道：设计文档已列为已知限制，本卡不覆盖。

**未决项**：

- 优先项：ReduceMax(calIndex) 对 NaN chunk 返回 0xFFFFFFFF 的确切硬件机制（归约树对 NaN 的 index 通道行为）未做位级 dump 直接验证，复现时可对 dst[1] 做逐位打印确认。
- NaN 与 +inf 共存用例、fp16/bf16 NaN 语义待后续迭代补齐。
## 坏实践

- 只修标量合并就收工——输出会从 0 变成 4294967296（2^32），看起来"更接近了"但仍是错值；必须同时处理 calIndex 的无效索引。
- 用 `v != v` 判 NaN 且不确认编译选项——fast-math 类优化可能破坏该假设。

## 误诊

- "ReduceMax 会忽略 NaN、返回最大有限值"：若如此，输出应是有限最大值的索引（非 0），与实测 0 矛盾——实测 NaN 传播进了 max、丢在合并比较上。
- "输出 4294967296 是 Cast 精度 bug"：Cast 链（uint32→float→int64）本身正确，错在喂进来的 globalIdx 已是 0xFFFFFFFF。
