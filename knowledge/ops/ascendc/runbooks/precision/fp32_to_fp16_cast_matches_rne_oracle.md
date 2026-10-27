---
type: "Runbook"
title: "fp32 窄化舍入模式不一致触发稀疏 MARE 越线"
description: "fp32 计算结果窄化到 fp16/bf16 时用 CAST_RINT 对齐 round-to-nearest-even，可消除边界值的稀疏相对误差尖峰。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# fp32 窄化舍入模式不一致触发稀疏 MARE 越线

## 现象

在 Gather 寻址问题修好后，运行 ApplyRotaryPosEmb 的正式 NPU 用例，20 项中仍有 4 个 fp16 用例失败。失败样例的 MERE 只有约 `6.78e-7`，但 MARE 达到约 `0.103`，刚超过 `0.1` 门槛；这说明绝大多数元素正确，只在很小的参考值附近出现稀疏相对误差尖峰。当时 fp32 中间结果通过 `RoundMode::CAST_ROUND` 窄化回输入 dtype。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`Cast`、`CAST_ROUND`、`CAST_RINT`、`round_to_nearest_even` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

输出从 fp32 窄化到 fp16 时，kernel 的 `CAST_ROUND` 与评测 oracle 的 round-to-nearest-even 没有对齐。在接近 fp16 舍入边界的值上，两种模式可能落到相邻可表示数；绝对差仍很小，所以 MERE 接近零，但相对误差按 `|y-ref|/(|ref|+1e-7)` 计算，小 `|ref|` 会把一个量化步长放大到约 0.1。把窄化模式换为 `CAST_RINT` 后，这些边界值与 oracle 对齐。

## 解决配方

### before

- path: `operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`
- symbol: `KernelApplyRotaryPosEmb<T>::ComputeRow`
- claim: fp32 输出用 CAST_ROUND 窄化，和评测的 round-to-nearest-even 不一致

```cpp
    __aicore__ inline void ComputeRow(uint32_t r, GlobalTensor<T>& xGm, GlobalTensor<T>& yGm) {
        uint32_t s = RowToSeqIdx(r);

        // --- CopyIn x ---
        LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyX{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padX{false, 0, 0, 0};
        DataCopyPad(xUb, xGm[(uint64_t)r * D_], copyX, padX);
        inQueueX.EnQue(xUb);
        LocalTensor<T> xUbDeq = inQueueX.DeQue<T>();

        // --- CopyIn cos / sin (half-D rows) ---
        LocalTensor<T> cosUb = inQueueCos.AllocTensor<T>();
        DataCopyExtParams copyC{1, static_cast<uint32_t>(halfD_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padC{false, 0, 0, 0};
        DataCopyPad(cosUb, cosGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueCos.EnQue(cosUb);
        LocalTensor<T> cosUbDeq = inQueueCos.DeQue<T>();

        LocalTensor<T> sinUb = inQueueSin.AllocTensor<T>();
        DataCopyPad(sinUb, sinGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueSin.EnQue(sinUb);
        LocalTensor<T> sinUbDeq = inQueueSin.DeQue<T>();

        // --- Cast x -> fp32 (fp32 input: DataCopy; fp16/bf16: Cast CAST_NONE) ---
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(xFp32, xUbDeq, DAlignF_);
        } else {
            Cast(xFp32, xUbDeq, RoundMode::CAST_NONE, D_);
        }
        PipeBarrier<PIPE_V>();

        // --- Cast cos/sin -> fp32 ---
        LocalTensor<float> cosHalfF = cosHalfFBuf.Get<float>();
        LocalTensor<float> sinHalfF = sinHalfFBuf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(cosHalfF, cosUbDeq, halfDAlignF_);
            DataCopy(sinHalfF, sinUbDeq, halfDAlignF_);
        } else {
            Cast(cosHalfF, cosUbDeq, RoundMode::CAST_NONE, halfD_);
            Cast(sinHalfF, sinUbDeq, RoundMode::CAST_NONE, halfD_);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cosFullF = cosFullFBuf.Get<float>();
        LocalTensor<float> sinFullF = sinFullFBuf.Get<float>();
        LocalTensor<float> rotF = rotFBuf.Get<float>();

        if (mode_ == 0) {
            // half: cos_full = [cos_half | cos_half] (repeat-2x)
            DataCopy(cosFullF, cosHalfF, halfD_);
            DataCopy(cosFullF[halfD_], cosHalfF, halfD_);
            DataCopy(sinFullF, sinHalfF, halfD_);
            DataCopy(sinFullF[halfD_], sinHalfF, halfD_);
            PipeBarrier<PIPE_V>();
            // rotate_half = concat(-x2, x1): rot[0..h) <- x[h..D) then negate; rot[h..D) <- x[0..h)
            DataCopy(rotF, xFp32[halfD_], halfD_);
            DataCopy(rotF[halfD_], xFp32, halfD_);
            PipeBarrier<PIPE_V>();
            Muls(rotF, rotF, -1.0f, halfD_);   // negate first half -> [-x2 | x1]
            PipeBarrier<PIPE_V>();
        } else {
            // interleaved: cos_full[2k]=cos_full[2k+1]=cos_half[k] via Gather(offExpand[i]=i/2 bytes)
            Gather(cosFullF, cosHalfF, offExpandBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            Gather(sinFullF, sinHalfF, offExpandBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            // rotate_half interleaved: rot[i]=x[i^1], then *signFull[-1,1,-1,1..]
            Gather(rotF, xFp32, offRotBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            Mul(rotF, rotF, signFullBuf.Get<float>(), D_);
            PipeBarrier<PIPE_V>();
        }

        // --- y = x*cos + rot*sin ---
        LocalTensor<float> mulF = mulFBuf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
        Mul(mulF, xFp32, cosFullF, D_);
        PipeBarrier<PIPE_V>();
        Mul(yFp32, rotF, sinFullF, D_);
        PipeBarrier<PIPE_V>();
        Add(yFp32, yFp32, mulF, D_);
        PipeBarrier<PIPE_V>();

        // --- Cast y -> input dtype (fp32: DataCopy; fp16/bf16: Cast CAST_ROUND) ---
        LocalTensor<T> yUb = outQueueY.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(yUb, yFp32, DAlignF_);
        } else {
            Cast(yUb, yFp32, RoundMode::CAST_ROUND, D_);
        }
        outQueueY.EnQue<T>(yUb);
        LocalTensor<T> yUbDeq = outQueueY.DeQue<T>();

        DataCopyExtParams copyY{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[(uint64_t)r * D_], yUbDeq, copyY);

        inQueueX.FreeTensor(xUbDeq);
        inQueueCos.FreeTensor(cosUbDeq);
        inQueueSin.FreeTensor(sinUbDeq);
        outQueueY.FreeTensor(yUbDeq);
    }
```

### after

- path: `operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`
- symbol: `KernelApplyRotaryPosEmb<T>::ComputeRow`
- claim: fp32→fp16/bf16 的最终 Cast 改为 CAST_RINT，其他计算顺序不变

```cpp
    __aicore__ inline void ComputeRow(uint32_t r, GlobalTensor<T>& xGm, GlobalTensor<T>& yGm) {
        uint32_t s = RowToSeqIdx(r);

        // --- CopyIn x ---
        LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyX{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padX{false, 0, 0, 0};
        DataCopyPad(xUb, xGm[(uint64_t)r * D_], copyX, padX);
        inQueueX.EnQue(xUb);
        LocalTensor<T> xUbDeq = inQueueX.DeQue<T>();

        // --- CopyIn cos / sin (half-D rows) ---
        LocalTensor<T> cosUb = inQueueCos.AllocTensor<T>();
        DataCopyExtParams copyC{1, static_cast<uint32_t>(halfD_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padC{false, 0, 0, 0};
        DataCopyPad(cosUb, cosGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueCos.EnQue(cosUb);
        LocalTensor<T> cosUbDeq = inQueueCos.DeQue<T>();

        LocalTensor<T> sinUb = inQueueSin.AllocTensor<T>();
        DataCopyPad(sinUb, sinGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueSin.EnQue(sinUb);
        LocalTensor<T> sinUbDeq = inQueueSin.DeQue<T>();

        // --- Cast x -> fp32 (fp32 input: DataCopy; fp16/bf16: Cast CAST_NONE) ---
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(xFp32, xUbDeq, DAlignF_);
        } else {
            Cast(xFp32, xUbDeq, RoundMode::CAST_NONE, D_);
        }
        PipeBarrier<PIPE_V>();

        // --- Cast cos/sin -> fp32 ---
        LocalTensor<float> cosHalfF = cosHalfFBuf.Get<float>();
        LocalTensor<float> sinHalfF = sinHalfFBuf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(cosHalfF, cosUbDeq, halfDAlignF_);
            DataCopy(sinHalfF, sinUbDeq, halfDAlignF_);
        } else {
            Cast(cosHalfF, cosUbDeq, RoundMode::CAST_NONE, halfD_);
            Cast(sinHalfF, sinUbDeq, RoundMode::CAST_NONE, halfD_);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cosFullF = cosFullFBuf.Get<float>();
        LocalTensor<float> sinFullF = sinFullFBuf.Get<float>();
        LocalTensor<float> rotF = rotFBuf.Get<float>();

        if (mode_ == 0) {
            // half: cos_full = [cos_half | cos_half] (repeat-2x)
            DataCopy(cosFullF, cosHalfF, halfD_);
            DataCopy(cosFullF[halfD_], cosHalfF, halfD_);
            DataCopy(sinFullF, sinHalfF, halfD_);
            DataCopy(sinFullF[halfD_], sinHalfF, halfD_);
            PipeBarrier<PIPE_V>();
            // rotate_half = concat(-x2, x1): rot[0..h) <- x[h..D) then negate; rot[h..D) <- x[0..h)
            DataCopy(rotF, xFp32[halfD_], halfD_);
            DataCopy(rotF[halfD_], xFp32, halfD_);
            PipeBarrier<PIPE_V>();
            Muls(rotF, rotF, -1.0f, halfD_);   // negate first half -> [-x2 | x1]
            PipeBarrier<PIPE_V>();
        } else {
            // interleaved: cos_full[2k]=cos_full[2k+1]=cos_half[k] via Gather(offExpand[i]=i/2 bytes)
            Gather(cosFullF, cosHalfF, offExpandBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            Gather(sinFullF, sinHalfF, offExpandBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            // rotate_half interleaved: rot[i]=x[i^1], then *signFull[-1,1,-1,1..]
            Gather(rotF, xFp32, offRotBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            Mul(rotF, rotF, signFullBuf.Get<float>(), D_);
            PipeBarrier<PIPE_V>();
        }

        // --- y = x*cos + rot*sin ---
        LocalTensor<float> mulF = mulFBuf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
        Mul(mulF, xFp32, cosFullF, D_);
        PipeBarrier<PIPE_V>();
        Mul(yFp32, rotF, sinFullF, D_);
        PipeBarrier<PIPE_V>();
        Add(yFp32, yFp32, mulF, D_);
        PipeBarrier<PIPE_V>();

        // --- Cast y -> input dtype (fp32: DataCopy; fp16/bf16: Cast CAST_RINT = round-to-nearest-even) ---
        LocalTensor<T> yUb = outQueueY.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(yUb, yFp32, DAlignF_);
        } else {
            Cast(yUb, yFp32, RoundMode::CAST_RINT, D_);
        }
        outQueueY.EnQue<T>(yUb);
        LocalTensor<T> yUbDeq = outQueueY.DeQue<T>();

        DataCopyExtParams copyY{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[(uint64_t)r * D_], yUbDeq, copyY);

        inQueueX.FreeTensor(xUbDeq);
        inQueueCos.FreeTensor(cosUbDeq);
        inQueueSin.FreeTensor(sinUbDeq);
        outQueueY.FreeTensor(yUbDeq);
    }
```

验证命令（trace 中原样执行）：

## 设计决策

- 只改最终窄化舍入模式，保持 fp32 乘加、旋转和搬运路径不变，使因果归因集中在 Cast。
- 没有放宽 MARE/阈值，也没有删除失败用例；测试侧绕过不作为修复。
- 当前方案属于 `recommended_fastpath`，因为它与 oracle 的数值契约一致且通过完整回归。

## 预防

- 在实现“低精度输入、fp32 中间计算、低精度输出”时，先从 oracle 确认最终窄化的舍入规则。
- 同时查看 MERE 与 MARE：MERE 极小而 MARE 单点越线，优先排查小参考值与舍入边界。
- 用完整失败集合验证 Cast 修改，避免只凭一个随机种子下的探针下结论。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：正式 fp16 回归汇总为 4 failed、16 passed。（来源平台：a3；证据等级: tool_observed）。
- 修复后：CAST_RINT 后定向失败集合全部通过。（来源平台：a3；证据等级: tool_observed）。
- 修复后：clean rebuild 成功，完整测试 30 passed。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 此配方只解决最终窄化的舍入契约不一致，不解决 fp32 计算顺序、FMA 与非 FMA、溢出或 NaN/Inf 传播差异。
- 当 oracle 不是 round-to-nearest-even 时，`CAST_RINT` 可能反而错误。
- 其他平台未验证；推断可迁移前应以目标平台的 Cast 契约和上板结果为准。

## 坏实践

- 为让 MARE 通过而放宽阈值、固定随机种子或删掉边界用例；这些做法隐藏了 kernel 与 oracle 的舍入差异。
- 只看平均绝对误差很小就忽略相对误差尖峰。

## 误诊

单独复现的随机输入曾得到 MARE 约 0.021，但它没有覆盖正式失败数据；不能据此否定正式用例的真实边界问题。
