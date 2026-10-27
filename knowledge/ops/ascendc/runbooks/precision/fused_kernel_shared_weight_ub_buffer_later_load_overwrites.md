---
type: "Runbook"
title: "共享权重buffer被覆写致l≥1误用wc"
description: "融合kernel多权重共用一个UB buffer，循环内后加载的权重覆写先加载的，l≥1迭代用错权重致近100% bitwise不一致；判别信号是首迭代对、后续全错且逐元素比值恒等于两权重之比；修复为专用w1Buf_循环外一次性预加载。"
tags: [task.debugging, topic.compute.normalization, topic.memory]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# 共享权重buffer被覆写致l≥1误用wc

## 现象

在一个 7 步融合算子（RMSNorm → 点积门控 → 广播乘 → 因果空洞深度卷积 → SiLU 残差，bf16 输入、fp32 内部计算、bitwise_equal 精度标准）的 Ascend C kernel（arch35，asc_vf_call/__simd_vf__ 行函数风格）联调中：ACLNN ST 23 用例只通过 7 个，其中大 shape 用例 M1/M3/M5/D3 近 100% 元素 bitwise 不一致（8388608/8388608 全错），F1 小用例 38/64 元素错。kernel 可正常运行、无任何报错——静默算错。

用 kernel 侧 PRINTF（DEBUG_KERNEL 宏）逐步 dump 中间量、与测试侧 C++ golden 参考实现的 DEBUG_GOLDEN 打印逐行对比，二分定界后发现特征性形态：

- F1 用例 l=0（首个 L 迭代）：statK/invRmsK/nk/gate/silu/out 全部中间值与 golden 一致；
- 从 l=1 起 normed_keys 开始错：kernel `nk[0]=0.018705` vs golden `nk[0]=0.020696`，而同一行的 `statK=0.020881` 完全一致——说明 keys 数据和统计量都对，是"归一化乘的权重"错了；
- 逐元素比值恒定：`0.018705/0.020696 = 0.9038`，恰好等于 `wc[0]/w1[0] = 0.06299888/0.06970517 = 0.9038`——S5 步读到的不是 norm1_weight(w1) 而是 conv_norm_weight(wc)。

## 适用场景

- 工程形态：aclnn 算子仓（operators/<pkg>/op_kernel/arch35）Kernel 直调 ST 联调，多权重融合 kernel。
- API 路径：AscendC `TBuf<TPosition::VECCALC>` + `DataCopyPad`（GM→UB 权重加载）+ `asc_vf_call<VF>`（__simd_vf__ 行函数）。
- 验证过的平台与 CANN 版本：950（arch35/DAV）+ CANN 9.2.0。
- 适用粒度：operator_family——任何"L 循环内多步复用同一块共享 UB 权重 buffer"的融合 kernel。
- 路线优先级：recommended_fastpath（专用 buffer 循环外预加载）。
- 明确不适用/未验证场景：权重只在循环外使用一次的 kernel；循环内每迭代确实要换权重的场景（需另行设计加载-使用同步，本卡方案不覆盖）；非 arch35 平台未验证。

## 根因

norm1_weight(w1) 只在 ProcessOneBhc 开头加载一次到共享的 `wBuf_`，但 L 循环体内的后续步骤（S7 加载 w2、S12 加载 wc）复用同一 `wBuf_` 并覆写其内容，导致 l≥1 迭代里 S5 读到的是 wc 而不是 w1。

单句归因：多个权重张量共用一个 UB buffer，循环内后加载者覆写先加载者。

与库内『写入破坏后续读取』族卡的区分：[FP16 half 就地覆写 x1 破坏 y_second 读依赖](fp16_half_x1_y_second.md) 与 [Abs 原地取绝对值后量化丢符号致 int8 坍缩](abs_inplace_overwrites_signed_source_before_quant.md) 是同一条指令 dst==src 的逐元素别名；本卡没有任何单条指令读写同址——是多个权重张量共用一个加载 buffer、循环内后加载者覆写先加载者的跨迭代生命周期错配。

## 判别方法

1. 现象形态："l=0 全对、l≥1 起全错" → 优先怀疑循环体内 buffer 被覆写/复用，而不是公式实现。
2. 逐元素比值探针：对首个出错迭代的出错中间量计算 `kernel值/golden值`，若比值逐元素恒定且等于两个候选权重张量对应元素之比，即可锁定"用错权重"及具体是哪两个权重。
3. 统计量对而归一化输出错（statK 一致、nk 不一致）→ 错在"乘权重"这一步，不在数据搬运和 reduce。

## 解决配方

Before（问题代码形态，S5 使用共享 wBuf_，ProcessOneBhc 开头一次性加载后被循环内 S7/S12 覆写）：

```cpp
// 成员：所有权重共用 wBuf_
AscendC::TBuf<AscendC::TPosition::VECCALC> bufB0_, bufB1_, bufB2_, tmpBuf_, wBuf_, convWBuf_, ...;

// ProcessOneBhc 开头（循环外）加载 w1 一次
LoadNormWeight(3, hc);   // 载入 wBuf_
...
for(int64_t l=0;l<L_;l++) {
    ...
    // S5: normed_k→B2 —— 此处 wBuf_ 已被上一迭代的 S7(w2)/S12(wc) 覆写
    asc_vf_call<RmsNormRowVF>((__ubuf__ float*)bufB2_.Get<float>().GetPhyAddr(),
        (__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),invRmsK,
        (__ubuf__ float*)wBuf_.Get<float>().GetPhyAddr(),(uint16_t)dAlignElems_,dRep_);
    ...
}
```

After（专用 w1Buf_，循环外一次性加载，永不被覆写）：

```cpp
// 1) 成员声明：新增专用 w1Buf_
AscendC::TBuf<AscendC::TPosition::VECCALC> bufB0_, bufB1_, bufB2_, tmpBuf_, wBuf_, w1Buf_, convWBuf_, convWTBuf_, lookbackBuf_, convStateBuf_, saveRowBuf_, csOutBuf_;

// 2) Init 中分配
pipe_->InitBuffer(tmpBuf_,256); pipe_->InitBuffer(wBuf_,rb);
pipe_->InitBuffer(w1Buf_,rb);  // dedicated norm1_weight buffer (never overwritten)

// 3) ProcessOneBhc 开头一次性加载 w1（所有 l 迭代共用）
template <typename T, int isEmpty>
__aicore__ inline void BenchHarnessEngramGateFusionKernel<T, isEmpty>::ProcessOneBhc(int64_t b, int64_t hc) {
    // Load norm1_weight into dedicated w1Buf_ (never overwritten, used by all l iterations)
    { AscendC::DataCopyExtParams e; e.blockCount=1; e.blockLen=(uint32_t)(D_*4); e.srcStride=0; e.dstStride=0; e.rsv=0;
      AscendC::DataCopyPadExtParams<float> p; p.isPad=(D_%8!=0); p.leftPadding=0; p.rightPadding=0; p.paddingValue=0;
      AscendC::DataCopyPad(w1Buf_.Get<float>(), gmIn_[3].template ReinterpretCast<float>()[hc*D_], e, p); }
    ...
}

// 4) S5 改用 w1Buf_
// S5: normed_k→B2 (w1 in dedicated w1Buf_, never overwritten by w2/wc)
asc_vf_call<RmsNormRowVF>((__ubuf__ float*)bufB2_.Get<float>().GetPhyAddr(),
    (__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),invRmsK,
    (__ubuf__ float*)w1Buf_.Get<float>().GetPhyAddr(),(uint16_t)dAlignElems_,dRep_);
```

## 设计决策

- 为何选专用 buffer：一次 MTE2 加载、循环内零额外加载与同步开销，且从结构上消除"覆写"这一类缺陷。
- 放弃的候选：循环内每次迭代 S5 前重载 w1（`LoadNormWeight(3, hc)` + `PipeBarrier<PIPE_ALL>`）——实测在非调试构建下引入新的时序敏感失败（E5/E6/F5/F8 反而被打破），详见坏实践。
- 相对代价：多占一份 `dAlignElems_*4` 字节（32 对齐）的 UB。本算子 UB 预算内可接受。

## 预防

- 多权重融合 kernel 设计期就为"每个跨迭代使用的权重"分配独立 TBuf，禁止共享复用。
- 评审检查清单：grep 循环体内所有 `LoadNormWeight`/`DataCopyPad(... wBuf_ ...)` 写入点，确认每个读取点的数据在上一次写入后未被覆写。
- 判别口诀：首迭代对、后续迭代错 → 查循环内 buffer 覆写；逐元素比值恒定 → 查权重/系数用错。

## 验证信号

本次验证来自该融合算子 Branch-0 kernel 精度修复链中的一轮修复会话（位于环形 lookback 修复与乘法原位覆盖修复之后）：用 kernel PRINTF 与 golden 双通道 dump 二分定位，代码修复后重跑同一 ACLNN ST 测试集。

- 修复前基线：7/23 用例通过，F1 38/64 元素错，M1/M3/M5/D3 近 100% 不一致（来源平台：950；证据等级：tool_observed）。
- 定界证据：l=0 全部中间值匹配 golden；l=1 起 `nk` 比值 0.9038 = `wc[0]/w1[0]`（来源平台：950；证据等级：tool_observed）。
- 修复后：12/23 用例通过，F1_StandardPrefill / F2_ReduceD1 / F5_K1StateLen0 / F6_DecodeL1ConvState / F8_Dilation1 / E5_RawGatePos / E6_RawGateNeg 转 PASSED，E1/E2 失败元素数从 42/64、38/64 收敛到 16/64（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只闭合"权重用错"这一根因；修复后仍失败的用例属其他根因（大 D 输入 cast 截断、NaN/Inf 注入等），不在本卡范围。
- UB 紧张、权重种类继续增多时，专用 buffer 方案需重新评估容量预算。
- 仅验证 bf16 输入 + fp32 计算通路、arch35（950）+ CANN 9.2.0；其他平台未验证（推断可迁移）。

**未决项**：

- 本根因已闭环。大 D 用例的剩余失败由 __simd_vf__ 多 rep 截断根因解释，见 [单次 asc_vf_call 大 rep 覆盖截断](simd_vf_multi_rep_truncation_large_d.md)（区分：rep 循环只执行前若干轮是向量调用内部覆盖截断，与本卡跨迭代 buffer 覆写无关）。
- 链式衔接：本卡修前基线（7/23、F1 38/64）与 [VF 乘法原位覆盖源 buffer 致 bitwise 失配](vf_mul_inplace_dst_overwrites_source_bitwise_mismatch.md) 修后状态逐字吻合——该卡修掉乘法原位覆盖后暴露出本卡的权重错用；更早一环是 [卷积环形 lookback 写早于读覆盖历史状态](conv_ring_lookback_write_before_read_overwrites_history.md)。三张卡对应三个不同代码位置、不同机制的独立根因闭环，不合并。
## 坏实践

- 把"多个权重张量轮流加载进同一个 UB buffer"当作省内存的手段，同时循环体内多次读取——覆写顺序与读取顺序稍不匹配即静默算错。
- 第一次修复尝试：把权重加载挪进循环内每次迭代重载（即使加了 `PipeBarrier<PIPE_ALL>`），非调试构建下反而把原本通过的 E5/E6/F5/F8 打挂（6/23）——循环内新增 MTE2 加载会引入时序敏感问题。

## 误诊

- 最初看到 kernel `statK=0.028437` vs golden `0.020096`，误判为"测试 PRNG 差异导致 golden 与 kernel 输入不同"。实际原因是多个用例连续运行时 kernel PRINTF 输出经 host 缓冲后交错乱序，把 minimal_call_path 的输出当成了 F1 的——按数值内容（而非输出顺序）匹配各用例的 DBG 行后，F1 的 statK 与 golden 完全一致。kernel 侧 PRINTF 是缓冲乱序输出的，多用例联跑时必须按内容值匹配。
- 排除输入搬运问题的过程：keys 逐元素值与 golden 一致、statK 一致 → 排除 GM offset/DataCopy 错误，聚焦权重乘法。
