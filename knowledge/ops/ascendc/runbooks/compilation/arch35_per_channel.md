---
type: "Runbook"
title: "arch35 小 per-channel 输出读旧值判别与规避"
description: "arch35 上 DCCI 缓存不一致使 DataCopyPad 对小 per-channel [C] 输出不可靠、TQue GetValue(MTE2→scalar) 读到旧值；判别方法为先 V-launder(Duplicate 0.0f+Add) 到 TBuf 再 GetValue，并用 GlobalTensor::SetValue 直写 GM per-channel 结果。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-18T04:31:14Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# arch35 小 per-channel 输出读旧值判别与规避

## 现象

BNTrainingUpdate 在 Ascend950PR/arch35 上做汇合验证时反复出现一类 per-channel 小输出（[C] 维，C 通常 ≤2048，如 mean_out/variance_out/batch_mean/batch_variance）行为异常：用 `DataCopyPad` 把这些小 [C] 结果从 UB 写回 GM 后，host 侧（或后续 MTE2 读取）读到旧值/错值；同时从 TQue buffer 直接 `GetValue(c)` 取标量也拿到过期数据。开发当时在 stage 1（core 0 预计算 multiplier/addend/batch 统计量）与 stage 2（所有核读 multiplier/addend 算 y）之间存在 GM 数据传递，per-channel 中间量（multiplier/addend）最初用 SetValue 写到**输入 tensor 的 GM 地址**（`gm_mul_` 复用 `inputs[5]`=mean），stage 2 经 MTE2 DataCopyPad 读回时受 DCCI 缓存一致性问题影响。

## 适用场景

- **工程形态**：kernel 直调 / aclnn regbase，多核分阶段 kernel 中 per-channel 小输出（[C] 维，C≤2048 量级）的 GM 写出与读回。
- **API 路径**：`GlobalTensor::SetValue` / `LocalTensor::GetValue` / `DataCopyPad`（MTE2/MTE3）/ `Duplicate`+`Add`（V-launder）。
- **验证平台与版本**：Ascend950PR（arch35 / DAV_3510），CANN 9.0.0-beta.2。
- **适用粒度**：`platform_wide`（arch35 平台级机制；推断适用所有 arch35 算子的小 per-channel 输出）。
- **路线优先级**：`recommended_fastpath`——per-channel 小输出用 SetValue 直写 GM + V-launder 读标量；多核间 per-channel 中间量用独立 workspace + MTE3 DataCopyPad 写、MTE2 DataCopyPad 读（32B 对齐）。
- **不适用或未验证场景**：大块连续输出（DataCopyPad 仍首选，32B 对齐时无问题）；a2/a3 平台未验证（见 `## 失效边界`）。

## 判别方法

### 触发

- per-channel [C] 小输出经 `DataCopyPad` 写 GM 后，host/下游读到旧值或错值；
- 从 TQue（MTE2 装入的 buffer）直接 `GetValue(c)` 取标量参与标量运算（如 `factor_is_zero` 分支判断、`gm_var_out_f_.SetValue(c, s2.GetValue(c))`），结果与预期不符但 V 计算本身无错；
- 多核间经 GM 传递的 per-channel 中间量（multiplier/addend）在 stage 2 MTE2 读回时偶发错误，且复现与 DCCI 缓存状态相关。

### 可执行探针

kernel `BNKernel::PrecomputeChannel`（`op_kernel/arch35/bn_training_update_kernel.h` 第 215-313 行，verbatim，取自 trace 的 file_read tool_result）——展示 V-launder + SetValue 的判别与修复写法：

```cpp
template <typename T, int64_t RANK>
__aicore__ inline void BNKernel<T, RANK>::PrecomputeChannel()
{
    int64_t C = td_->c_dim;
    float inv_num = 1.0f / (float)td_->num;
    // factor/epsilon read from TilingData (DynamicCompileStaticFlag(false) ensures fresh)
    float factor = factor_;
    float epsilon = epsilon_;
    uint32_t C32 = static_cast<uint32_t>(C);
    uint32_t chan_bytes = static_cast<uint32_t>(C * sizeof(float));
    uint32_t chan_bytes_padded = (chan_bytes + 31u) & ~31u;
    AscendC::DataCopyPadExtParams<T> padParams{};

    auto bmean = buf_bmean_.template Get<float>();
    auto bvar   = buf_bvar_.template Get<float>();
    auto mul    = buf_mul_.template Get<float>();
    auto add    = buf_add_.template Get<float>();
    auto s1     = buf_s1_.template Get<float>();
    auto s2     = buf_s2_.template Get<float>();

    // --- Load sum, compute batch_mean -> bmean ---
    {
        auto t = q_chan_.template AllocTensor<T>();
        AscendC::DataCopyExtParams ep{1, chan_bytes_padded, 0, 0, 0};
        AscendC::DataCopyPad(t, gm_sum_, ep, padParams);
        q_chan_.EnQue(t);
        auto sum = q_chan_.template DeQue<T>();
        auto sumF = sum.template ReinterpretCast<float>();
        AscendC::Muls<float>(bmean, sumF, inv_num, C32);
        AscendC::PipeBarrier<PIPE_V>();
        q_chan_.FreeTensor(sum);
    }

    // --- Load square_sum, compute batch_var -> bvar ---
    {
        auto t = q_chan_.template AllocTensor<T>();
        AscendC::DataCopyExtParams ep{1, chan_bytes_padded, 0, 0, 0};
        AscendC::DataCopyPad(t, gm_sq_, ep, padParams);
        q_chan_.EnQue(t);
        auto sqsum = q_chan_.template DeQue<T>();
        auto sqsumF = sqsum.template ReinterpretCast<float>();
        AscendC::Muls<float>(bvar, sqsumF, inv_num, C32);
        AscendC::PipeBarrier<PIPE_V>();
        q_chan_.FreeTensor(sqsum);
    }
    AscendC::Mul<float>(s1, bmean, bmean, C32);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(bvar, bvar, s1, C32);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs<float>(bvar, bvar, 0.0f, C32);
    AscendC::PipeBarrier<PIPE_V>();

    // --- batch_var_unbiased in s1 ---
    if (td_->num > 1) {
        float bessel = (float)td_->num / (float)(td_->num - 1);
        AscendC::Muls<float>(s1, bvar, bessel, C32);
    } else {
        AscendC::Muls<float>(s1, bvar, 0.0f, C32);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // --- Save batch_var and batch_mean via SetValue ---
    // arch35 API-1 note: GlobalTensor::SetValue used for per-channel [C] outputs because
    // arch35 DCCI cache inconsistency makes DataCopyPad unreliable for small per-channel
    // outputs. SetValue writes directly to GM (host-visible, no MTE2 read-back needed).
    // Performance impact is limited: only core 0 in stage 1, C typically <=2048.
    for (int64_t c = 0; c < C; c++) gm_bvar_f_.SetValue(c, bvar.GetValue(c));
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < C; c++) gm_bmean_f_.SetValue(c, bmean.GetValue(c));
    AscendC::PipeBarrier<PIPE_V>();

    // --- variance_out = factor*batch_var_unbiased + (1-factor)*variance ---
    // arch35 fix: TQue buffer GetValue (MTE2->scalar) is unreliable.
    // Solution: copy TQue data to TBuf via V operation (Add with zero), then read from TBuf.
    // Also avoids Muls(dst, src, 1.0f, C) copy (issue #6 class) for factor=0/1.
    // Use tolerance comparison instead of exact == for factor edge cases.
    // factor/epsilon are correctly propagated via TilingData every call
    // (DynamicCompileStaticFlag(false) ensures fresh tiling).
    bool factor_is_zero = (factor < 1e-6f && factor > -1e-6f);
    bool factor_is_one  = (factor > 1.0f - 1e-6f && factor < 1.0f + 1e-6f);
    {
        auto t = q_chan_.template AllocTensor<T>();
        AscendC::DataCopyExtParams ep{1, chan_bytes_padded, 0, 0, 0};
        AscendC::DataCopyPad(t, gm_var_, ep, padParams);
        q_chan_.EnQue(t);
        auto var = q_chan_.template DeQue<T>();
        auto varF = var.template ReinterpretCast<float>();
        // V-launder: copy varF to s2 via element-wise Add with zero (reliable on arch35)
        AscendC::Duplicate<float>(s2, 0.0f, C32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(s2, s2, varF, C32);  // s2 = 0 + varF = variance
        AscendC::PipeBarrier<PIPE_V>();
        q_chan_.FreeTensor(var);
        // Now s2 has variance (V-written TBuf, GetValue reliable)
        // arch35 API-1 note: SetValue for per-channel output (DCCI constraint, see above)
        if (factor_is_zero) {
            // variance_out = variance
            for (int64_t c = 0; c < C; c++) gm_var_out_f_.SetValue(c, s2.GetValue(c));
        }
        // --- factor_is_one / else 分支与 mean_out 计算（第 314 行起）未在 trace 中恢复 ---
    }
}
```

> 注：`PrecomputeChannel` 第 314 行起（`factor_is_one`/else 分支与 `mean_out` 计算）未在本 trace 的 file_read 中恢复（read 在第 314 行被截断）。V-launder 与 SetValue 模式在 215-313 行已完整呈现，足以表达判别方法—。

### 原始观测

- per-channel [C] 输出经 DataCopyPad 写 GM 后 host 侧读到旧值；改用 `GlobalTensor::SetValue` 逐元素直写 GM 后正确（来源平台：950；证据等级：tool_observed）
- TQue buffer 直接 `GetValue(c)` 取标量偶发错值；经 `Duplicate 0.0f` + `Add(s2, s2, varF)` V-launder 到 TBuf 后 `GetValue` 可靠（来源平台：950；证据等级：tool_observed）

### 解释

arch35 上 V-pipeline 写 UB 与 MTE2/MTE3 搬运之间存在 DCCI（entire data cache invalidate）一致性问题：小 per-channel 块（<32B 对齐粒度，或 C 很小）经 DataCopyPad 写/读时缓存行为不可靠；`GetValue` 从 TQue（MTE2 装入路径）取标量走的是另一条缓存路径，同样不可靠。V-launder（先用 `Duplicate` 填零、再用 `Add` 把 TQue 数据加到 TBuf）强制走 V-pipeline 写 TBuf，使后续 `GetValue` 从 V-written TBuf 读取，绕过 MTE2 标量路径的缓存问题。`SetValue` 直写 GM（host-visible，无需 MTE2 read-back），规避小块 DataCopyPad 的 DCCI 问题；性能代价有限：仅 core 0 在 stage 1、C 通常 ≤2048。

> 以上解释为开发轨迹中的根因分析（机制层主张），DCCI 触发的确切缓存行边界未在本 trace 系统化测量。

## 验证信号

- 用 SetValue 替代 DataCopyPad 写 per-channel [C] 输出后，mean_out/variance_out/batch_mean/batch_variance 在 factor=0.1 的常规用例上 PASS（来源平台：950；证据等级：tool_observed）
- V-launder 后 `s2.GetValue(c)` 返回正确 variance，`factor_is_zero` 分支可正确取到值（来源平台：950；证据等级：tool_observed）

## 性能影响

- SetValue 逐元素写 [C] 相对 DataCopyPad 块搬运约慢，但仅 core 0 在 stage 1 执行、C≤2048，相对整体 y 计算（stage 2 全核、大块 tile）的搬运量可忽略；未给出绝对时延，只表征"代价有限、非瓶颈"（参照路线：DataCopyPad 块搬运）。

## 预防

- arch35 上 per-channel 小输出优先用 `GlobalTensor::SetValue` 直写 GM；多核间 per-channel 中间量用独立 workspace（32B 对齐）+ MTE3 DataCopyPad 写、MTE2 DataCopyPad 读，避免复用输入 tensor 地址与依赖 DCCI。
- 从 TQue 取标量参与标量运算前，先 V-launder 到 TBuf 再 `GetValue`，不要直接对 MTE2 装入的 TQue buffer `GetValue`。

## 失效边界

- 已验证：Ascend950PR/arch35，CANN 9.0.0-beta.2，FP32 per-channel [C] 输出。
- 未验证，推断可迁移：a2、a3 平台——DCCI/缓存一致性是 arch 级机制，是否同款问题需在各 arch 上实测；标注"未验证，推断可迁移"。
- 不适用：大块连续 32B 对齐输出（DataCopyPad 仍首选且可靠）。