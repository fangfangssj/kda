---
type: "Runbook"
title: ReduceSum复用源改写computeBuf致脏读
description: "AscendC::ReduceSum(Pattern::Reduce::AR) 以复用源语义改写源 computeBuf；之后从该 buffer 读 fp32 exp 值做除法会读到脏数据，mare 从 0.0375 恶化到 0.2138（约 5.7 倍）；归约后读源 buffer 的优化路线 banned，fp16/bf16 须从 preInBuf 重算。"
tags: [task.debugging, topic.precision, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# ReduceSum复用源改写computeBuf致脏读

## 现象

在 Ascend950PR（CANN 9.2.0）上调 BenchHarnessSoftmax 的 fp16 精度时，为对齐 torch 参考实现做过一次"消除二次 exp 计算"的优化尝试：`ProcessFastPath` 原流程是 `DoSubExp`（把 fp32 的 exp(x-m) 写入 computeBuf）→ `DoReduceSum`（从 computeBuf 归约求和）→ `DoFinalDivCast`（fp16/bf16 分支**从 preInBuf 重读原始 x、重算一遍 Sub+Exp+Div+Cast**）。优化思路是：既然 computeBuf 里已经是 fp32 exp 值，`DoFinalDivCast` 直接从 computeBuf 读出来做 Div+Cast，省掉重算 exp，还保证"求和与除法用同一份 exp"。

改动只动了 `DoFinalDivCast` 的 fp16/bf16 分支（把 `BHSoftmaxSubExpDivCastFromSrcVf` 换成读 fp32 源的 `BHSoftmaxDivCastRowVf`），求和仍是 `DoReduceSum`。结果 TTK 重跑同一用例，精度不降反升：mare 从 0.0375 恶化到 **0.2138**（mere 也从 1.73e-8 升到 4.82e-8），失败元素集合大幅变化；把 `DoFinalDivCast` 单独回退（求和不动）后 mare 回落到 0.0600 水平。

## 适用场景

- 工程形态：AscendC kernel 数值路径开发/优化（复用中间 buffer 省二次计算）+ TTK 白盒精度评测。
- API 路径：`AscendC::ReduceSum<Dtype, Pattern::Reduce::AR, ...>` 的 `isReuseSource` 复用源语义；`asc_vf_call` VF 读 UB computeBuf。
- 验证过的平台与 CANN 版本：Ascend950PR_9579，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"归约后还要读归约源 buffer"的多阶段 kernel（softmax 三段式、LayerNorm、reduce+scale 等）。
- 路线优先级：banned（归约后读源 buffer 这条优化路线被成对实验否定；如需复用 exp 值必须先关闭源复用或换缓冲区，未验证）。
- 明确不适用 / 未验证场景：
  - 归约后**不再**读源 buffer 的 kernel（源被改写无影响，原始代码正是靠"重算 exp"避开）；
  - `isReuseSource=false`（或换临时缓冲区）能否既保源数据又复用值——本轨迹未实测，tmpBuf 空间是否足够未知。

## 根因/解决配方

**根因**：`DoReduceSum` 调用 `AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(dstTensor, srcTensor, tmpTensor, srcShape, true)`，其复用源语义会把归约中间结果写回源 tensor（computeBuf）；`DoFinalDivCast` 随后从 computeBuf 读"exp 值"时读到的是被归约覆写后的数据，除法输入错误，精度大幅劣化。

改动前的原始 `DoFinalDivCast`（before，verbatim，函数级完整——fp16/bf16 分支从 preInBuf 重读原始 x 融合重算）：

```cpp
    // Fast path final step: for fp16/bf16, re-read original x from preInBuf
    // and do fused Sub+Exp+Div+Cast. For fp32, data is already exp'd in
    // preInBuf (in-place compute buffer), so just Div+Cast.
    __aicore__ inline void DoFinalDivCast(int64_t validARows)
    {
        auto cache = cacheBuf_.Get<float>();
        int64_t sOffset = td_->aUbFactorAlign;
        uint16_t srcK = static_cast<uint16_t>(td_->rUbFactorAlign);
        if constexpr (std::is_same_v<T, float>) {
            // fp32: data is exp(x-m) in preInBuf (compute buffer), just Div.
            __ubuf__ float* srcPtr = GetComputePtr();
            __ubuf__ T* dstPtr = (__ubuf__ T*)preInBuf_.Get<T>().GetPhyAddr();
            for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
                float sVal = cache.GetValue(static_cast<uint32_t>(sOffset + i));
                AscendC::PipeBarrier<PIPE_ALL>();
                asc_vf_call<BHSoftmaxDivCastRowVf<T, VL_F32_U16>>(
                    dstPtr + i * td_->rUbFactorAlign,
                    srcPtr + i * td_->rUbFactorAlign, sVal, srcK);
            }
        } else {
            // fp16/bf16: re-read original x from preInBuf, fused Sub+Exp+Div+Cast.
            __ubuf__ T* srcPtr = (__ubuf__ T*)preInBuf_.Get<T>().GetPhyAddr();
            for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
                float mVal = cache.GetValue(static_cast<uint32_t>(i));
                float sVal = cache.GetValue(static_cast<uint32_t>(sOffset + i));
                AscendC::PipeBarrier<PIPE_ALL>();
                asc_vf_call<BHSoftmaxSubExpDivCastFromSrcVf<T, VL_F32_U16>>(
                    srcPtr + i * td_->rUbFactorAlign,
                    srcPtr + i * td_->rUbFactorAlign, mVal, sVal, srcK);
            }
        }
    }
```

被调用的归约（verbatim，函数级完整）：

```cpp
    __aicore__ inline void DoReduceSum(bool canReuseSrc = false)
    {
        auto dstTensor = tmpBuf0_.Get<float>();
        auto tmpTensor = tmpBuf1_.Get<uint8_t>();
        uint32_t srcShape[2] = {
            static_cast<uint32_t>(td_->aUbFactorAlign),
            static_cast<uint32_t>(td_->rUbFactorAlign)
        };
        if constexpr (std::is_same_v<T, float>) {
            auto srcTensor = preInBuf_.Get<float>();
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
                dstTensor, srcTensor, tmpTensor, srcShape, true);
        } else {
            auto srcTensor = computeBuf_.Get<float>();
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
                dstTensor, srcTensor, tmpTensor, srcShape, true);
        }
    }
```

尝试的 `DoFinalDivCast`（改动后，fp16/bf16 分支从 computeBuf 读 fp32 exp；verbatim，函数级完整）：

```cpp
    // Fast path final step: divide exp values (in computeBuf) by per-row sum
    // and cast to output dtype. Uses the SAME fp32 exp values that were used
    // to compute the sum (DoReduceSum reads computeBuf), ensuring exp/sum
    // consistency and matching torch's pairwise-sum + e/s computation more
    // closely than recomputing exp from source.
    __aicore__ inline void DoFinalDivCast(int64_t validARows)
    {
        auto cache = cacheBuf_.Get<float>();
        int64_t sOffset = td_->aUbFactorAlign;
        uint16_t srcK = static_cast<uint16_t>(td_->rUbFactorAlign);
        if constexpr (std::is_same_v<T, float>) {
            // fp32: data is exp(x-m) in preInBuf (compute buffer), just Div.
            __ubuf__ float* srcPtr = GetComputePtr();
            __ubuf__ T* dstPtr = (__ubuf__ T*)preInBuf_.Get<T>().GetPhyAddr();
            for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
                float sVal = cache.GetValue(static_cast<uint32_t>(sOffset + i));
                AscendC::PipeBarrier<PIPE_ALL>();
                asc_vf_call<BHSoftmaxDivCastRowVf<T, VL_F32_U16>>(
                    dstPtr + i * td_->rUbFactorAlign,
                    srcPtr + i * td_->rUbFactorAlign, sVal, srcK);
            }
        } else {
            // fp16/bf16: exp(x-m) is already in computeBuf (fp32) from DoSubExp.
            // Use those fp32 exp values directly for Div+Cast, ensuring the
            // same exp used for sum and division (no recomputation from source).
            __ubuf__ float* srcPtr = GetComputePtr();
            __ubuf__ T* dstPtr = (__ubuf__ T*)preInBuf_.Get<T>().GetPhyAddr();
            for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
                float sVal = cache.GetValue(static_cast<uint32_t>(sOffset + i));
                AscendC::PipeBarrier<PIPE_ALL>();
                asc_vf_call<BHSoftmaxDivCastRowVf<T, VL_F32_U16>>(
                    dstPtr + i * td_->rUbFactorAlign,
                    srcPtr + i * td_->rUbFactorAlign, sVal, srcK);
            }
        }
    }
```

**解决（回退）**：fp16/bf16 分支恢复为从 preInBuf 重读原始 x、用 `BHSoftmaxSubExpDivCastFromSrcVf` 融合重算 Sub+Exp+Div+Cast（重算 exp 不依赖 computeBuf，源被改写无影响）。终态代码在函数头注释中固化了这条约束（verbatim，取自最终 diff）：

```cpp
    // Fast path final step: for fp16/bf16, re-read original x from preInBuf
    // and do fused Sub+Exp+Div+Cast. For fp32, data is already exp'd in
    // preInBuf (in-place compute buffer), so just Div+Cast.
    // Note: DoReduceSum with isReuseSource=true may modify computeBuf, so
    // fp16/bf16 MUST recompute exp from preInBuf (not read corrupted computeBuf).
```

注意：fp32 分支读同一 computeBuf 却不受影响——fp32 路径的 `DoReduceSum` 源是 preInBuf 而非 computeBuf（见上面 `if constexpr` 分支），且其归约发生在读取之后。**是否踩坑取决于"归约源"与"后续读取源"是否同一 buffer**。

## 设计决策

- 为什么最初认为读 computeBuf 更好：省一次 Sub+Exp 重算，且"求和与除法用同一份 exp 值"在直觉上更贴近 torch 的单次计算链；没有先核对 `ReduceSum` 的源复用语义。
- 为什么回退而不是关掉复用：`isReuseSource=false` 可能需要额外临时缓冲（tmpBuf 容量未验证），而重算 exp 的代价在 UB 内可接受且是已验证正确的原始形态；优化收益（省一次 exp）远小于引入的验证成本。
- 当前方案路线档位：banned（"归约后读源 buffer"路线）；正确替代 = 归约后重算，或（未验证）关闭源复用/换缓冲。

## 预防

- 用 `AscendC::ReduceSum`（及其他带 `isReuseSource` 语义的 reduce API）前，先确认归约之后是否还有任何路径读源 tensor；有则要么换读取源、要么显式关闭源复用并评估缓冲占用。
- 对"复用中间 buffer 省计算"的优化，上线前用同用例成对 A/B（只动这一处）看 mare/mere 变化——本例正是靠 0.0375→0.2138→回退→0.0600 的成对序列定位到这一处改动。

## 验证信号

以下信号来自该算子 fp16 失败用例修复会话：同一用例（fp16 8 维大 shape、dim=-1）在"只改 `DoFinalDivCast` 读取源"前后的 TTK 白盒实测（Ascend950PR，CANN 9.2.0）。

- 读 computeBuf（改动后）：mare=0.2137793004512787、mere=4.8197183133424915e-08，`threshold exceeded: mare(2.14e-01>=0.009765625)`（来源平台：950；证据等级：tool_observed）。
- 回退为重算 exp（同一 `DoReduceSum` 求和不变）：mare=0.05996023118495941——精度恢复到该求和策略的基线水平（来源平台：950；证据等级：tool_observed）。
- 对照基线（改动前同用例）：mare=0.037484459578990936（顺序累加策略版）；恶化幅度 0.2138/0.0375≈5.7 倍（来源平台：950；证据等级：tool_observed）。
- 终态代码检查：`git diff` 相对基线仅剩"求和回退 + 上述 isReuseSource 注释"，`Reg::Div` 计数 5、`Reg::Mul(recip)` 计数 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 成对证据证明"该 kernel 的 AR 模式 `ReduceSum(..., true)` 之后读 computeBuf 会读到脏数据"；`isReuseSource` 的精确参数位（模板第三参 vs 函数末参）与改写内容的逐字节行为未做内存 dump 验证，机制解释以 API 语义 + 成对实验为准。
- 恶化后的 mare（0.2138）同时叠加了底层 1-ULP 次正规差异（同批另有判别卡），"纯脏读贡献量"无法从单一数字剥离。
- 仅在 Ascend950PR + CANN 9.2.0 的 AR pattern 实测；其他 pattern（RA/ARF 等）与其他 NPU 世代的源复用行为未验证（未验证，推断可迁移——用前按本卡预防节先核对）。

**未决项**：

- `isReuseSource=false` 路径的缓冲需求与正确性未验证；若未来确需复用 exp 值（如性能优化），应先补这组实验。
- `ReduceSum` 对源 buffer 的具体覆写模式（哪些偏移、写入什么）未做 dump 级确认。
## 坏实践

- 归约后直接复用"归约前写入、归约时作为源"的 buffer，不核对 reduce API 的源复用语义。
- 看到"省一次重算"的优化就合并，不做只动这一处的成对精度对照。

## 误诊

无（本卡范围内部署后即被成对实验定位；最初的"复用 exp 更接近 torch"是设计误判而非观测误诊，已记入设计决策）。
