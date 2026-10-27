---
type: "Runbook"
title: "BF16 gamma Cast 分阶段编排导致随机 NaN"
description: "BF16 gamma Cast 与输入 Cast 分处 LoadGamma/Compute 时出现随机坏行；在同一 MTE2 屏障后顺序发射两次 Cast 可消除竞态。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion", "topic.compute.normalization"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# BF16 gamma Cast 分阶段编排导致随机 NaN

## 现象

在 Ascend C 直调的高阶 RMSNorm kernel 中，FP32、FP16 和非对齐 FP32 冒烟结果正常，而 BF16 路径直接打印 `bf16 max diff: nan`。继续重复执行同一 BF16 输入时，某次可能 `max diff=0`，下一次又出现约 `1e37` 到 `1e38` 的异常值，坏行位置也会变化；增加行数后同时出现 `max diff=nan` 和多行错误。代码把 gamma 的 BF16→FP32 Cast 放在 `LoadGamma`，把 x 的 BF16→FP32 Cast 放在后续 `Compute`，两个 Vector 操作分属不同阶段。

## 适用场景

- 工程形态：Ascend C kernel 直调，BF16 输入包装 FP32 高阶 RMSNorm；
- API 路径：`DataCopyPad` + `Cast` + `PipeBarrier<PIPE_MTE2/PIPE_V>` + `AscendC::RmsNorm<float, false>`；
- 验证过的平台与版本：Ascend 950PR / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于多个 BF16→FP32 Cast 共享后续 Vector 消费、且输入由 MTE2 分阶段搬入的 kernel；
- 路线优先级：`recommended_fastpath`，先等待相关 MTE2 完成，再在同一 Compute 阶段顺序发射 gamma/x Cast；
- 明确不适用：不含 BF16 Cast 的 FP16/FP32 直通路径，以及由高阶 API 完全拥有搬运与类型转换的封装。

## 根因

before 将 gamma 的搬入与 Cast 封装在 `LoadGamma`，随后 `CopyIn` 又发起 x 的 MTE2 搬入，`Compute` 再单独 Cast x。实际重复运行显示，这种跨阶段编排没有为后续 RmsNorm 建立一个同时覆盖 gamma 和 x 的完成点：gammaFp32 或 srcFp32 偶发包含未就绪数据，RmsNorm 读取后产生随机超大值或 NaN。 after 把 gamma Cast 延后到 `Compute`：先用一个 `PIPE_MTE2` 屏障等待 gamma 和 x 的搬运，再在同一 Vector 序列中依次 Cast gamma 与 x，最后用 `PIPE_V` 确认两个 FP32 输入都完成。这样形成明确的 `MTE2(gamma,x) → V(Cast gamma, Cast x) → V(RmsNorm)` 依赖链。

## 解决配方

`LoadGamma` 只负责搬入 gamma 并等待 MTE2；BF16 的两次 Cast 都移到 `Compute` 的同一屏障之后。

### before

`op_kernel/rms_norm_kernel.cpp`，`KernelRmsNorm::LoadGamma`：

```cpp
    __aicore__ inline void LoadGamma()
    {
        constexpr uint32_t tsz = sizeof(T);
        auto gammaLocal = gammaBuf_.Get<T>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(D_ * tsz),
            0, 0, 0
        };
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(hLength_ - D_);
        padParams.paddingValue = (T)0;
        AscendC::DataCopyPad(gammaLocal, gammaGm_, copyParams, padParams);

        // For bf16: Cast gamma to fp32
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            AscendC::PipeBarrier<PIPE_MTE2>();
            auto gammaFp32 = gammaFp32Buf_.Get<float>();
            AscendC::Cast(gammaFp32, gammaLocal, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(hLength_));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }
```

`op_kernel/rms_norm_kernel.cpp`，`KernelRmsNorm::Compute`：

```cpp
    __aicore__ inline void Compute()
    {
        auto xLocal = xQueue_.DeQue<T>();
        auto yLocal = yQueue_.AllocTensor<T>();
        auto tmpLocal = tmpBuf_.Get<uint8_t>();

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            // ---- bf16 path: Cast in -> RmsNorm<float> -> Cast out ----
            auto srcFp32 = srcFp32Buf_.Get<float>();
            auto dstFp32 = dstFp32Buf_.Get<float>();
            auto gammaFp32 = gammaFp32Buf_.Get<float>();

            // Cast bf16 -> fp32
            AscendC::Cast(srcFp32, xLocal, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(coreRows_ * hLength_));
            AscendC::PipeBarrier<PIPE_V>();

            // Set ShapeInfo on fp32 tensors
            AscendC::ShapeInfo bshInfo{3, inputShape_, 3, originShape_, AscendC::DataFormat::ND};
            AscendC::ShapeInfo gammaInfo{1, gammaShape_, 1, originGammaShape_, AscendC::DataFormat::ND};
            srcFp32.SetShapeInfo(bshInfo);
            dstFp32.SetShapeInfo(bshInfo);
            gammaFp32.SetShapeInfo(gammaInfo);

            // RmsNorm<float> (epsilon as float) — no MTE2 barrier needed,
            // src data comes from Cast (PIPE_V), already synced above
            AscendC::RmsNorm<float, false>(dstFp32, srcFp32, gammaFp32, tmpLocal,
                                           epsilon_, rmsNormTiling_);
            AscendC::PipeBarrier<PIPE_V>();

            // Cast fp32 -> bf16
            AscendC::Cast(yLocal, dstFp32, AscendC::RoundMode::CAST_RINT,
                          static_cast<int32_t>(coreRows_ * hLength_));
        } else {
            // ---- direct path: RmsNorm<T> (T = float or half) ----
            auto gammaLocal = gammaBuf_.Get<T>();

            AscendC::ShapeInfo bshInfo{3, inputShape_, 3, originShape_, AscendC::DataFormat::ND};
            AscendC::ShapeInfo gammaInfo{1, gammaShape_, 1, originGammaShape_, AscendC::DataFormat::ND};
            xLocal.SetShapeInfo(bshInfo);
            yLocal.SetShapeInfo(bshInfo);
            gammaLocal.SetShapeInfo(gammaInfo);

            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::RmsNorm<T, false>(yLocal, xLocal, gammaLocal, tmpLocal,
                                       static_cast<T>(epsilon_), rmsNormTiling_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }
```

### after

`op_kernel/rms_norm_kernel.cpp`，`KernelRmsNorm::LoadGamma`：

```cpp
    __aicore__ inline void LoadGamma()
    {
        constexpr uint32_t tsz = sizeof(T);
        auto gammaLocal = gammaBuf_.Get<T>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(D_ * tsz),
            0, 0, 0
        };
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(hLength_ - D_);
        padParams.paddingValue = (T)0;
        AscendC::DataCopyPad(gammaLocal, gammaGm_, copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
```

`op_kernel/rms_norm_kernel.cpp`，`KernelRmsNorm::Compute`：

```cpp
    __aicore__ inline void Compute()
    {
        auto xLocal = xQueue_.DeQue<T>();
        auto yLocal = yQueue_.AllocTensor<T>();
        auto tmpLocal = tmpBuf_.Get<uint8_t>();

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            // ---- bf16 path: Cast gamma + Cast x -> RmsNorm<float> -> Cast out ----
            auto srcFp32 = srcFp32Buf_.Get<float>();
            auto dstFp32 = dstFp32Buf_.Get<float>();
            auto gammaFp32 = gammaFp32Buf_.Get<float>();
            auto gammaLocal = gammaBuf_.Get<T>();

            // Ensure all MTE2 (gamma + x DataCopyPad) complete before any Cast
            AscendC::PipeBarrier<PIPE_MTE2>();

            // Cast gamma bf16 -> fp32
            AscendC::Cast(gammaFp32, gammaLocal, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(hLength_));
            // Cast x bf16 -> fp32
            AscendC::Cast(srcFp32, xLocal, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(coreRows_ * hLength_));
            AscendC::PipeBarrier<PIPE_V>();

            // Set ShapeInfo on fp32 tensors
            AscendC::ShapeInfo bshInfo{3, inputShape_, 3, originShape_, AscendC::DataFormat::ND};
            AscendC::ShapeInfo gammaInfo{1, gammaShape_, 1, originGammaShape_, AscendC::DataFormat::ND};
            srcFp32.SetShapeInfo(bshInfo);
            dstFp32.SetShapeInfo(bshInfo);
            gammaFp32.SetShapeInfo(gammaInfo);

            // RmsNorm<float> (epsilon as float) — no MTE2 barrier needed,
            // src data comes from Cast (PIPE_V), already synced above
            AscendC::RmsNorm<float, false>(dstFp32, srcFp32, gammaFp32, tmpLocal,
                                           epsilon_, rmsNormTiling_);
            AscendC::PipeBarrier<PIPE_V>();

            // Cast fp32 -> bf16
            AscendC::Cast(yLocal, dstFp32, AscendC::RoundMode::CAST_RINT,
                          static_cast<int32_t>(coreRows_ * hLength_));
        } else {
            // ---- direct path: RmsNorm<T> (T = float or half) ----
            auto gammaLocal = gammaBuf_.Get<T>();

            AscendC::ShapeInfo bshInfo{3, inputShape_, 3, originShape_, AscendC::DataFormat::ND};
            AscendC::ShapeInfo gammaInfo{1, gammaShape_, 1, originGammaShape_, AscendC::DataFormat::ND};
            xLocal.SetShapeInfo(bshInfo);
            yLocal.SetShapeInfo(bshInfo);
            gammaLocal.SetShapeInfo(gammaInfo);

            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::RmsNorm<T, false>(yLocal, xLocal, gammaLocal, tmpLocal,
                                       static_cast<T>(epsilon_), rmsNormTiling_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }
```

## 设计决策

- 选择移动 gamma Cast，而不是在原 `LoadGamma` 末尾继续堆屏障：后者仍把两个相关 Cast 分散在不同阶段；
- 两次 Cast 在同一 MTE2 完成点后按 Vector 顺序发射，减少跨函数隐式时序；
- 保留 BF16 输出 `CAST_RINT`，因为本次根因是执行编排而不是舍入模式；
- 当前方案为 `recommended_fastpath`。

## 系统信息

- 工具链：CANN 9.0.1，bisheng `-xasc`；
- 目标：Ascend 950PR；
- 文件：`op_kernel/rms_norm_kernel.cpp`；
- 相关符号：`KernelRmsNorm::LoadGamma`、`KernelRmsNorm::Compute`；
- dtype 路径：BF16→FP32 Cast，高阶 `RmsNorm<float, false>`，FP32→BF16 `CAST_RINT`。

## 预防

- 对同一个高阶计算消费的多个输入，画出共同的“全部搬入完成”屏障，不要只逐函数验证局部顺序；
- 间歇性坏行、坏行位置漂移、同输入时好时坏优先指向流水或未初始化数据，不要先调容差；
- BF16 路径应单独重复运行，不要只看一次 smoke 的最终 PASS 文案。

## 验证信号

来源为一次 BF16 高阶 RMSNorm 的间歇性精度调试：定向重复复现后只重排 gamma/x Cast，再运行相同探针和后续全任务。

- 修复前一次运行显示 `bf16 max diff: nan`，另一次重复探针出现约 `1e37` 到 `1e38` 的异常差值和随机坏行（Ascend 950PR，证据等级: tool_observed）；
- 修复后五次重复探针与一次扩大行数探针全部 `max diff=0`、`bad rows=[]`（Ascend 950PR，证据等级: tool_observed）；
- 保留该修复的最终全任务评测为 20/20（Ascend 950PR，证据等级: tool_observed）。

## 失效边界

- 只验证 Ascend 950PR / CANN 9.0.1 和 BF16→FP32 高阶 RMSNorm 路径；
- 若多个输入来自不同队列或使用不同事件同步，不能机械复制一个 `PIPE_MTE2`，应按真实生产者建立依赖；
- A2/A3 未验证，推断可迁移但需重新做重复运行探针。

## 坏实践

- 一次 smoke 末尾打印 `PASSED` 就忽略前面已出现的 `max diff=nan`；
- 用固定随机种子或只跑一次来规避间歇性坏行；
- 把随机 NaN 当成 BF16 精度阈值问题并放宽容差。

## 误诊

曾尝试把高阶 `RmsNorm` 临时替换为 identity copy 来判断 Cast 是否工作；这只能缩小范围，不能证明跨阶段 Cast 编排正确。