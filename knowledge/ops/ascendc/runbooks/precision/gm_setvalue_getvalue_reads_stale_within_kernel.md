---
type: "Runbook"
title: "GM SetValue 写回后 GetValue 读到旧值"
description: "同一 Kernel 内用 SetValue 把中间结果写回 GM，后续 GetValue 读回的不是刚写入的新值而是更新前旧值，导致依赖该中间值的下游计算出现系统性小偏差。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# GM SetValue 写回后 GetValue 读到旧值

## 现象

在 Ascend950 上用 ops-registry-invoke-glacier 工作流开发 ApplyCamePart2（CAME 优化器 Part2）算子的 Branch-0 fp32 Kernel。该 Kernel 分 `ProcessR → ProcessC → ProcessU` 三阶段：ProcessR 计算并写出 r_out、ProcessC 计算并写出 c_out、ProcessU 读回 r_out/c_out 计算 `u = grad / sqrt(r*c/denominator)` 与 `sum_square_u`。编译通过、r_out/c_out 输出正确，但 `u` 与 `sum_square_u` 在所有小 shape 用例上系统性偏差约 2e-4，远超 DESIGN §4.3.1 的 fp32 阈值 rtol/atol=1e-5。运行 `operators/ApplyCamePart2_package/tests/aclnn && ./run.sh` 的典型输出（B0-S0/S1/S2）：

```
[B0-S0] u[0]: got -3.21718693e-01, expected -3.21867615e-01
[B0-S0] sum_square_u: got 8.47996831e-01, expected 8.48850012e-01
[B0-S0] FAILED (N=4, M=6)
[B0-S1] u[0]: got -3.00000042e-01, expected -3.00195217e-01
[B0-S1] sum_square_u: got 1.68506533e-01, expected 1.68705508e-01
[B0-S1] FAILED (N=1, M=6)
[B0-S2] u[0]: got -3.21741462e-01, expected -3.21926653e-01
[B0-S2] sum_square_u: got 1.54920787e-01, expected 1.55099168e-01
[B0-S2] FAILED (N=4, M=1)
```

注意 B0-S1 `got -3.00000042e-01`：这恰是 `grad[0] / sqrt(1.0 * 1.0 / denom)`（ri、ci 取原值 1.0）的结果，而非用更新后 `r_out=0.999`、`c_out=0.999` 应得到的 `-0.300195`。偏差量级与 `beta2=0.999`、`(1-0.999)` 的修正项一致，强烈暗示 ProcessU 读到的 r_out/c_out 是更新前的值。

出问题时的 ProcessU 实现（修复前，verbatim，取自 apply_came_part2_kernel.h）：

```cpp
    __aicore__ inline void ProcessU()
    {
        float denom = beta2_ * sumR_ / N_ + (1.0f - beta2_) * sumGradRc_ / (M_ * N_);
        float rcCoeff = 1.0f / denom;

        LocalTensor<float> cOutLocal = buf1_.Get<float>();
        CopyInArray(cOutLocal, cOutGm_, m_);
        PipeBarrier<PIPE_ALL>();

        float sumSquareU = 0.0f;

        for (int64_t i = 0; i < n_; i++) {
            float ri = rOutGm_.GetValue(i);
            float coeff = ri * rcCoeff;

            LocalTensor<float> gradRow = buf3_.Get<float>();
            CopyInArray(gradRow, gradGm_, m_, i * m_);
            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> rcRow = buf4_.Get<float>();
            Muls(rcRow, cOutLocal, coeff, m_);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> sqrtRow = buf5_.Get<float>();
            Sqrt(sqrtRow, rcRow, m_);
            PipeBarrier<PIPE_V>();

            Div(gradRow, gradRow, sqrtRow, m_);
            PipeBarrier<PIPE_ALL>();

            CopyOutArray(gradRow, uGm_, m_, i * m_);

            float rowSum = 0.0f;
            for (int64_t j = 0; j < m_; j++) {
                float uVal = gradRow.GetValue(j);
                rowSum += uVal * uVal;
            }
            sumSquareU += rowSum;
        }

        sumSquareUGm_.SetValue(0, sumSquareU);
    }
```

其中 `CopyInArray`/`CopyOutArray` 为：

```cpp
    __aicore__ inline void CopyInArray(LocalTensor<float>& dst, GlobalTensor<float>& src,
                                        int64_t count, int64_t offset = 0)
    {
        int64_t alignedCount = (count + NUM_PER_BLK - 1) / NUM_PER_BLK * NUM_PER_BLK;
        DataCopy(dst, src[offset], alignedCount);
    }

    __aicore__ inline void CopyOutArray(LocalTensor<float>& src, GlobalTensor<float>& dst,
                                         int64_t count, int64_t offset = 0)
    {
        for (int64_t i = 0; i < count; i++) {
            dst.SetValue(offset + i, src.GetValue(i));
        }
    }
```

关键点：ProcessR 用 `CopyOutArray(rOutLocal, rOutGm_, n_)`（即 `rOutGm_.SetValue(...)`）把 r_out 写回 GM，ProcessC 同样把 c_out 写回 GM；ProcessU 随后用 `rOutGm_.GetValue(i)` 与 `CopyInArray(cOutLocal, cOutGm_, m_)` 再从 GM 读回。读回的值并未反映本 Kernel 刚写入的更新。

## 适用场景

- **工程形态**：Kernel 直调（ops-registry-invoke，Ascend C device kernel），多阶段 Kernel 内前一阶段写 GM、后一阶段读同一 GM 缓冲的场景。
- **API 路径**：`GlobalTensor::SetValue` / `GlobalTensor::GetValue` / `DataCopy`（GM↔UB）；`LocalTensor::SetValue`/`GetValue`（UB 内）。
- **验证平台与版本**：Ascend950（CANN 1.0.0, TimeStamp 202602），fp32 计算链。trace 内仅在此平台实测。
- **适用粒度**：`platform_wide`（凡 Ascend950 Kernel 内存在「写 GM 后同 Kernel 内再读 GM」依赖链的算子均可能命中）。
- **路线优先级**：`recommended_fastpath`——优先把跨阶段复用的中间结果留在 UB 局部缓冲（`TBuf<VECCALC>`）传递，仅在 Kernel 末尾一次性写回 GM；不从 GM 回读自己刚写入的值。
- **不适用或未验证场景**：跨 Kernel（不同核函数实例）的 GM 传递走 `SyncAll`/`aclrtSynchronizeStream` 保证可见性，不在本结论范围；非 950 平台（a2/a3）未验证，推断可迁移但未实测。

## 判别方法

**触发条件**：Kernel 编译通过、单测输出 r_out/c_out 正确，但下游 u/sum_square_u 等依赖「回读中间结果」的输出出现系统性小偏差（与 `(1-beta2)` 修正项量级一致），且偏差在多 shape 上一致、非随机噪声。

**可执行探针**：在写 GM 之后、回读处之前，用 `PRINTF` 打印写入值与回读值做配对对比；并打印由回读值参与计算的下游量（如本例 `ri`、`coeff`）。PRINTF 必须配合「验证是否新 binary 生效」使用——本平台存在 kernel binary 缓存，单纯的源码改动可能不生效（见相关卡 [kernel binary 缓存](../compilation/kernel_binary_cache_printf_hash_toggle.md)）；PRINTF 改变 device 侧 binary hash，可同时兼作「强制加载新 binary」的开关。

```cpp
// ProcessR 内 r_out 写回后：
//   CopyOutArray(rOutLocal, rOutGm_, n_);
// ProcessU 内回读处加探针：
    for (int64_t i = 0; i < n_; i++) {
        float ri = rOutGm_.GetValue(i);
        float coeff = ri * rcCoeff;
        PRINTF("i=%lld ri=%f coeff=%f\n", (long long)i, ri, coeff);
        // ...
    }
```

**原始观测**（Ascend950，fp32，beta2=0.999，B0-S0/S1/S2）：
- `sumR_`（在 ProcessR 内由局部 `rLocal.GetValue` 累加得到，未走 GM 回读）经 PRINTF 确认正确：B0-S0=4.600000（=1.0+1.1+1.2+1.3）、B0-S1=1.000000、B0-S2=4.600000。
- 但 `ri=rOutGm_.GetValue(i)` 返回值与更新后 r_out 不符——计算期望 `r_out[0] = 0.999*1.0 + 0.001*(-0.3/6) = 0.99895`，而回读路径参与计算的 `u[0]` 恰为 `grad[0]/sqrt(1.0*1.0/denom) = -0.3` 量级（B0-S1 got -0.300000042），即 ri、ci 均取原值 1.0。
- 把 r_out 改为缓存到 UB 局部缓冲 `workBuf`、c_out 缓存到 `buf2`，ProcessU 改从局部缓冲读 `ri`/`c_out` 后，PRINTF 显示 `ri=0.998950`（B0-S0/B0-S1）、`ri=0.998700`（B0-S2/S3），与期望一致；B0-S0/S1/S2 精度达 1e-5 阈值并通过。

**机制解释**：在 Ascend950 的 Kernel 执行模型下，`SetValue`/`CopyOutArray` 把数据写回 GM 与 `GetValue`/`DataCopy` 从 GM 回读，不保证在**同一 Kernel 实例内**的先后顺序间建立可访问性——回读路径取到的是 GM 中该地址更新前的旧内容（或经 DMA 缓存未刷新的视图）。跨阶段依赖的中间结果若先落 GM 再回读，会读到旧值，造成系统性而非随机偏差。正确做法是：凡本 Kernel 后续阶段还要用的中间结果，一律保留在 UB 局部缓冲（`TBuf<VECCALC>`）并在阶段间通过局部 Tensor 引用传递，仅在不再需要回读的最终输出点一次性写回 GM。

## 解决配方

修复方向（已在 trace 内对 r_out/c_out 验证有效）：为需要跨阶段复用的中间输出各分配一块 UB 局部缓冲，ProcessR/ProcessC 计算完 `r_out`/`c_out` 后同时写入对应局部缓冲（除写回 GM 外），ProcessU 改从局部缓冲读 `ri`/`c_out`，不再从 GM 回读。

注意：本卡的「解决配方」给出经 PRINTF 探针 + 单测通过验证的修复方向与作用点；完整的 after 函数级代码因 markdown_transcript 转写截断了 edit 工具参数（`newString` 被记为 `_see below_`），未能在事件流中完整恢复，故不在此内联 verbatim after 代码块。

## 验证信号

本卡验证信号来自 ApplyCamePart2 Branch-0 fp32 Kernel 在 Ascend950 上的 ST 运行（`operators/ApplyCamePart2_package/tests/aclnn && ./run.sh`），同一 trace 内的修复迭代实测。

- 修复前 B0-S0/S1/S2 u[0] 与 sum_square_u 系统性偏差 ~1.5e-4~2e-4，全部 FAILED（来源平台：Ascend950；证据等级: tool_observed）。
- PRINTF 探针确认 `sumR_` 正确（4.6/1.0/4.6），`ri=rOutGm_.GetValue(i)` 取到更新前旧值 → 单点归因到 GM 回读路径（来源平台：Ascend950；证据等级: tool_observed）。
- 改从 UB 局部缓冲读 `ri`/`c_out` 后，PRINTF 显示 `ri=0.998950`（B0-S0/S1）、`ri=0.998700`（B0-S2/S3），与期望一致；B0-S0/S1/S2 精度达 rtol/atol=1e-5 阈值并 PASSED（来源平台：Ascend950；证据等级: tool_observed）。

> B0-S3 [1,1] 及更大 shape 的最终通过状态：trace 在 Task 30 仍未结案时截断，未验证。
## 失效边界

- 本结论仅在 Ascend950（CANN 1.0.0, 202602）fp32 计算链实测。
- a2/a3 平台：未验证，推断可迁移（GM↔UB 一致性模型在 AscendC 跨平台理论上同源），但未实测，采用前需先在小 shape 单测复现 PRINTF 探针确认。
- 跨 Kernel（不同核函数实例）间的 GM 传递依赖 `SyncAll`/`aclrtSynchronizeStream` 保证可见性，不在本卡结论范围。

## 坏实践

- 在 ProcessU 里对 `rOutGm_`/`cOutGm_` 直接 `GetValue` 回读本 Kernel 刚 `SetValue` 写入的值——这是触发本偏差的根因写法。
- 试图用 `PipeBarrier<PIPE_ALL>()` 在「写 GM」与「读 GM」之间加屏障来强制可见性——trace 内未观察到该写法能修复偏差，屏障不能替代「不在同 Kernel 内回读自己刚写的 GM」这一结构改写。
- 用放宽容差（如把 rtol/atol 从 1e-5 放到 1e-3）来让 `u` 通过——这是测试侧放松捷径，掩盖了真实的 GM 一致性缺陷，不得作为解决配方（见反投机取巧纪律）。

## 误诊

- 初判曾怀疑 `ComputeSumR` 用 `DataCopy` 读 r 局部缓冲的方式有误，把 `ComputeSumR` 改为 `GetValue` 直读 GM——未消除偏差。PRINTF 探针随后表明 `sumR_` 本就正确，偏差源在 ProcessU 对 `r_out`/`c_out` 的 GM 回读（证据等级: agent_reported；非实测结论，仅记录调试路径）。
- 初判曾怀疑 NPU 运行旧 kernel binary（见相关卡 [kernel binary 缓存](../compilation/kernel_binary_cache_printf_hash_toggle.md)）。该怀疑在「源码改动但输出不变」阶段成立且已用 PRINTF 验证并解决；但在 binary 已确认更新（sumR_ 正确）后偏差仍存，说明 binary 缓存只是第一层干扰，真正根因是 GM 回读可见性——两层问题叠加，需逐变量探针分离。

## 预防

- 设计多阶段 Kernel 时，默认原则：跨阶段复用的中间量一律走 UB 局部缓冲传递，仅在最终输出点写 GM；不在同 Kernel 内回读自己刚写入的 GM 地址。
- 若必须跨 GM 传递中间结果，应拆分为两个 Kernel 实例并用 `SyncAll`/`aclrtSynchronizeStream` 在 Kernel 边界保证可见性，而非在同一 Kernel 内回读。
- 评审 Kernel 实现时，对 `GlobalTensor.SetValue` 后同函数内 `GlobalTensor.GetValue` 读同一地址的模式直接打回。