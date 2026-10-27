---
type: "Runbook"
title: "卷积环形 lookback 写早于读覆盖历史状态"
description: "因果空洞卷积的环形 lookback 缓冲在卷积读取前写入当前行，覆盖 k=0 抽头尚未消费的历史/state_prefix；须先读后写且 k=K-1 直接读当前值缓冲。"
tags: [task.debugging, topic.precision, topic.compute.neighborhood.convolution, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# 卷积环形 lookback 写早于读覆盖历史状态

## 现象

为融合算子（dual RMSNorm → gate → 因果空洞 depthwise ShortConv → SiLU → residual）实现 Branch-0 kernel 时，因果空洞卷积用 `stateLen × D` 的环形 lookback UB 缓冲保存历史 normed_vg（state_prefix 初始化 + 滚动更新）。初版把"当前行写入 lookback"放在卷积读取之前。上板跑 ST：22 个数值用例全部 bitwise mismatch，例如 `F7_D769Prime: output 6140/6152 elems differ`、`M1_DeepSeekPrefill: output 8388596/8388608 elems differ`、全零用例 `E3_AllZero: output 22/64 elems differ`。

当时代码状态（修复前，verbatim）：

```cpp
// Save normed_vg to lookback + conv → B1
int64_t lbRow = l % stateLen_;
for (int64_t d = 0; d < D_; d++) {
    float vgVal = b0Fp32[d];
    lookbackPtr[lbRow * dAlignElems_ + d] = vgVal;
    float convSum = 0.0f;
    for (int64_t k = 0; k < K_; k++) {
        int64_t n = l + k * dil_;
        float xVal = lookbackPtr[(n % stateLen_) * dAlignElems_ + d];
        convSum += convWPtr[d * K_ + k] * xVal;
    }
    b1Fp32[d] = convSum;
}
```

## 适用场景

- 工程形态：算子仓 aclnn 包 `op_kernel` 的 kernel 实现与精度联调（ST bitwise 对齐 golden）。
- API 路径：UB 环形缓冲（`TBuf` + `GetPhyAddr` 标量访问）上的因果/空洞卷积状态管理。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.2.0。
- 适用粒度：operator_family——任何带 rolling window 状态（lookback/ring buffer）的序列卷积 kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证：stateLen=0（无状态）或 K=1（单抽头）的卷积无环别名问题；跨核分片的环形状态未验证。

## 根因

在环形 lookback 上，行 l 的写入槽 `l % stateLen` 在写入那一刻恰好保存着本次卷积 k=0 抽头要读的 `x_cat[l]`（l < stateLen 时是 state_prefix[l]，l ≥ stateLen 时是 normed_vg[l − stateLen]，即 stateLen 轮前存入该槽的值）；save-before-read 把它覆盖成当前行 normed_vg[l]，k=0 抽头系统性读错。推导：x_cat = [state_prefix(stateLen) | normed_vg(L)]，输出行 l 的抽头 k 读 `x_cat[l + k·dil]`，槽映射为 `(l + k·dil) % stateLen`；别名条件 `k·dil ≡ 0 (mod stateLen)` 在默认配置 `stateLen = (K−1)·dil` 下对 k=0 与 k=K−1 同时成立——k=K−1 语义上就该读当前值（`x_cat[l + stateLen] = normed_vg[l]`），预写"侥幸正确"；k=0 需要的是历史值，被预写破坏。

与库内『写入破坏后续读取』族卡的区分：[FP16 half 就地覆写 x1 破坏 y_second 读依赖](fp16_half_x1_y_second.md) 与 [Abs 原地取绝对值后量化丢符号致 int8 坍缩](abs_inplace_overwrites_signed_source_before_quant.md) 都是同一条指令 dst==src 的逐元素别名；本卡没有任何指令读写同址——是环形缓冲槽位在被消费前被新值覆写的时序问题（写早于读），机制不同、各自独立闭环。

## 解决配方

两处改动（修复后，verbatim）：

1. k=K−1 抽头不再经过环，直接读当前值缓冲 B0；
2. 当前行保存推迟到卷积读取之后。

```cpp
// Conv: for k in [0,K), x_cat[l+k*dil] from lookback (k<K-1) or B0 (k=K-1)
// Then save normed_vg[l] to lookback[l % stateLen] AFTER conv
int64_t lbRow = l % stateLen_;
for (int64_t d = 0; d < D_; d++) {
    float convSum = 0.0f;
    for (int64_t k = 0; k < K_; k++) {
        int64_t n = l + k * dil_;
        float xVal;
        if (k == K_ - 1) {
            // x_cat[l + stateLen] = normed_vg[l], read from B0 directly
            xVal = b0Fp32[d];
        } else {
            xVal = lookbackPtr[(n % stateLen_) * dAlignElems_ + d];
        }
        convSum += convWPtr[d * K_ + k] * xVal;
    }
    b1Fp32[d] = convSum;
}
// Save normed_vg[l] to lookback AFTER conv (for future positions)
for (int64_t d = 0; d < D_; d++)
    lookbackPtr[lbRow * dAlignElems_ + d] = b0Fp32[d];
```

一般化配方：环形缓冲上的"先读后写"——任何写入槽 `i % N` 的操作必须晚于本轮所有对该槽的读取；与写入槽别名且语义为"当前值"的抽头改读源缓冲，不经过环。

## 设计决策

- k=K−1 直读 B0 而不是也走环后再读：直读消除该抽头对环的依赖，写入时机即可整体推迟到卷积后，一处改动同时修复别名与时机。
- 保存循环独立于卷积循环（两段而不是合并）：卷积内同时写 lookback 会重新引入读-写交错，拆开保证"读全部完成 → 再写"。

## 预防

- 实现滚动状态缓冲时先列抽头-槽映射表：每个抽头 k 的槽 `(l + k·dil) % stateLen`，标出与写入槽 `l % stateLen` 别名的抽头，检查其语义是"当前值"还是"历史值"。
- code review 检查清单：环形缓冲的写语句是否出现在本轮最后一个读语句之前。

## 验证信号

来源说明：以下信号来自同一次 Branch-0 kernel 开发会话修复前后的两次 ST 运行（Ascend 950，CANN 9.2.0；数值为 `output X/N elems differ` 形式的工具输出，X 下降表示改善）。

- 修复前（写早于读）：F7_D769Prime 6140/6152、M1_DeepSeekPrefill 8388596/8388608、M2_DeepSeekDecode 4081/4096、M3_LargeBatch 8385515/8388608 differ（来源平台：950；证据等级：tool_observed）。
- 改为先读后写后：F7 6101/6152、M1 8388540/8388608、M2 4071/4096、M3 8384702/8388608 differ——依赖历史/前缀的用例全部改善，证实写早于读是历史状态被破坏的根因（来源平台：950；证据等级：tool_observed）。
- 如实记录：全零用例 E3 在本次修复后从 22/64 变为 47/64 differ（恶化）——全零输入下本修复两侧卷积值理论相同（全零），该变化更可能来自并存的标量/向量流水线竞争，不能归因于本修复；22 个数值用例在本次会话内未达全绿（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 验证于本算子的因果空洞 depthwise 卷积环形状态（`stateLen = (K−1)·dil` 默认配置）；别名条件 `k·dil ≡ 0 (mod stateLen)` 随 K/dil/stateLen 组合变化，换配置需重列抽头-槽映射表。
- "先读后写"是一般纪律，适用于任何 rolling window；但"当前值抽头直读源缓冲"这一步依赖 `stateLen = (K−1)·dil` 的索引代数，其他配置需按 `x_cat[l + k·dil]` 重新推导哪个抽头落在当前行。
- K=1、stateLen=0 场景无环别名，不适用。

**未决项**：

- 优先项：E3（全零输入）在本次修复后的恶化未追根因——需复现并确认其驱动因素（倾向标量/向量流水线竞争；S/V 跨流水线 UB 交接的同步配方参见 [S/V 跨流水线 UB 交接需显式完成同步](vector_scalar_ub_handoff_requires_explicit_completion.md)，选用条件：标量控制流无法同 VF 收敛时用 S_V/V_S HardEvent 事件对，E3 是否同根因未验证）。
- 22 个数值用例未全绿；剩余 mismatch（NaN/Inf 传播、conv_state_out 更新）由后续修复接续，本卡只闭合"历史状态被覆盖"这一根因。本 trace 修复链：507035 崩溃修复（[多余核无守卫越界分发](../runtime/507035_unguarded_extra_cores_oob_dispatch.md)、[环形索引负模回绕](../runtime/507035_negative_modulo_ring_index_ub_offset.md)）→ 本卡 → [VF 乘法原位覆盖源 buffer 致 bitwise 失配](vf_mul_inplace_dst_overwrites_source_bitwise_mismatch.md)（区分：乘法目的地址覆盖后续要读的源，是别名而非写早于读）→ [多权重共用加载 buffer 后加载覆写](fused_kernel_shared_weight_ub_buffer_later_load_overwrites.md)（区分：跨迭代 load-target 生命周期错配，非环形缓冲）；NaN/Inf 位模式问题另由 [bf16 输出 NaN 载荷归一化](vf_cast_bf16_nan_payload_0x7fff_vs_golden_7fc0_normalize.md) 闭合。三张覆写卡是三个不同代码位置的独立根因闭环，不合并。
## 坏实践

- 为省一个临时缓冲，把"保存当前行"内联进卷积循环里与读取交错——重新引入写读竞争。
- 只修时机（推迟写）但让所有抽头仍走环：k=K−1 语义上读当前值，走环会在别名槽读到旧值，必须同时改直读 B0。
