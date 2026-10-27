---
type: "Runbook"
title: int32经fp32中转丢精度，超2^24须原生比较
description: 标量扫描统一走 ToFloat/FromFloat 后 int32 值失配（golden=-538846106 actual=-538846080）；|v|>2^24 整数不能经 fp32 往返，int32 须原生比较。
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# int32经fp32中转丢精度，超2^24须原生比较

## 现象

Cummin 算子 ARA-scan 标量重写后，此前通过的 L0_003 用例（int32，shape=[6603,10,6]，dim=-3，A0=60）出现值回归：

```
[Real] Test: L0_003_int32_3d_full_range
  shape=[6603,10,6], dtype=int32, dim=-3
    mismatch [60]: golden=-538846106, actual=-538846080
    mismatch [61]: golden=774414982, actual=774414976
    mismatch [62]: golden=-1415088108, actual=-1415088128
    mismatch [63]: golden=-1359637234, actual=-1359637248
    mismatch [64]: golden=-1868089178, actual=-1868089216
  [FAIL] values: 392067 mismatches out of 396180
  [PASS] indices: all 396180 elements match
```

特征：索引全对、值系统性偏移小量（26、6、20、14、38…）。标量重写版当时对所有 dtype 统一走 `ToFloat<T>`/`FromFloat<T>`（fp32 中转）做比较与回存：

```cpp
// Subsequent rows: scalar compare and update
for (uint64_t a0 = 0; a0 < a0Size; a0++) {
    float curF = ToFloat<T>(inputBuf.GetValue(a0));      // T=int32_t 时 static_cast<float>
    float minF = ToFloat<T>(runMinT.GetValue(a0));
    ...
    if (curF < minF) { minF = curF; ... }
    runMinT.SetValue(a0, FromFloat<T>(minF));            // static_cast<int32_t>(f)
    ...
}
```

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 内对 int32/int64 数据做比较/最值维护，且实现为统一类型模板（fp32 中转）。
- API 路径：`LocalTensor::GetValue/SetValue` + 标量类型转换（`static_cast<float>`/`static_cast<int32_t>`）。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：`platform_wide`（fp32 表示能力是平台无关的数值事实；触发条件是 |v|>2^24 的 int 数据）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：|v| ≤ 2^24 的 int32 值（fp32 精确表示，中转无损）；int64 数据（问题同型且更严重，本轨迹无 int64 输入用例）。

## 根因/解决配方

根因（机制链）：fp32 尾数 24 位（含隐含位），只能精确表示 |v| ≤ 2^24 = 16777216 的整数；超出后相邻可表示数间隔为 2^(⌊log2|v|⌋-23)。以观测值演算：

- -538846106：|v| ≈ 5.39e8 ∈ [2^29, 2^30)，fp32 间隔 2^(29-23) = 64；538846106 = 64×8419470 + 26，最近可表示点为 64×8419470 = 538846080（候选 538846144 距离 38 > 26）→ round-to-nearest 落在 -538846080，与观测 actual 精确一致；
- 774414982 ∈ [2^29, 2^30)，间隔 64；774414982 = 64×12037734 + 6 → 最近点 774414976，同样与观测一致。

`ToFloat→比较→FromFloat` 往返把 int32 值舍入到最近 fp32 可表示点，392067/396180 个 full_range 随机值中 |v|>2^24 的全部失真；索引不受影响（比较结果仅在舍入改变大小关系时才错，本例数据未触发）。

解决配方：整数类型不走 float 中转，用 `if constexpr` 分流出原生 int32 比较路径（verbatim）：

```cpp
if (r == 0) {
    // First row: initialize running_min and running_idx
    for (uint64_t a0 = 0; a0 < a0Size; a0++) {
        runMinT.SetValue(a0, inputBuf.GetValue(a0));
        runIdxF.SetValue(a0, 0.0f);
        valBuf.SetValue(a0, inputBuf.GetValue(a0));
        idxI64.SetValue(a0, static_cast<int64_t>(0));
    }
} else {
    // Subsequent rows: scalar compare and update
    if constexpr (std::is_same_v<T, int32_t>) {
        // Integer path: direct int32 comparison (no float conversion)
        // This avoids precision loss for values outside float32 exact range (>2^24)
        for (uint64_t a0 = 0; a0 < a0Size; a0++) {
            int32_t cur = inputBuf.GetValue(a0);
            int32_t minVal = runMinT.GetValue(a0);
            float idxF = runIdxF.GetValue(a0);

            if (cur < minVal) {
                minVal = cur;
                idxF = static_cast<float>(r);
            }

            runMinT.SetValue(a0, minVal);
            runIdxF.SetValue(a0, idxF);
            valBuf.SetValue(a0, minVal);
            idxI64.SetValue(a0, static_cast<int64_t>(idxF));
        }
    } else {
        // Float path: NaN-aware comparison（fp32 域，见相关卡）
        ...
    }
}
```

运行索引仍可用 float 维护（索引值 ≤ 扫描维长度，远小于 2^24），只有数据值必须原生比较。

## 设计决策

- 为什么按类型分流而不是全程 int64/int32：浮点类型仍需 float 域（bf16 标量比较限制，见 [bf16 标量位级转换卡](../compilation/bf16_scalar_to_fp32_bit_extension.md)）与 NaN 语义；整数与浮点的正确比较域不同，单一通路无法同时满足。
- 索引保留 float 表示：向量 Select/Cast 路线（float 索引）已验证，且索引量级无精度风险。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` + run 包安装 + C++ ST（NPU 实测）。
- 相关文件：`op_kernel/arch35/cummin.h`（ProcessAraScan/ProcessArScan 标量循环的 int32 分支）、`tests/st/test_aclnn_cummin.cpp`（L0_003 full_range int32 用例）。

## 验证信号

验证来自同一开发会话的 NPU 前后对照：修复前 L0_003 `values: 392067 mismatches out of 396180`（golden=-538846106 vs actual=-538846080 等，索引全对）（来源平台：950；证据等级：tool_observed）；int32 原生比较分支落地后 L0_003 `[PASS] values: all 396180 elements match / [PASS] indices: all 396180 elements match`（来源平台：950；证据等级：tool_observed）；该用例在后续全量 L0 ST 21/21 中保持通过（来源平台：950；证据等级：tool_observed）。观测差值与 fp32 舍入间隔演算（64 间隔、最近点偏移 26/6）逐值吻合（机制推导，非实测）。

## 失效边界

- int64 输入同理不可经 fp32 中转（|v|>2^24 即失真，且 int64→fp32 还有表示范围问题）；本轨迹无 int64 用例，未实测。
- 该修复只覆盖比较与回存路径；若索引/计数等整型中间量超过 2^24（约 1677 万），同样不能存进 float。
- 浮点路径的 NaN 语义分支与整数分支互斥，混合 dtype 算子需各自分流。
- 同一"fp32 精确整数范围 2^24"数值事实在 golden 容差侧的另一失效形态（int32 fp32 累加 reduce 与不同 reduce 树的 bitwise 不可达）见 [int32 fp32累加reduce binary_equal不可达卡](int32_fp32_reduce_binary_equal.md)——那卡属 spec 容差设计问题，本卡是 kernel 数据路径修复，适用条件不同。

**未决项**：

- 无（机制与修复均已闭环；int64 场景为推断边界，标注未验证）。
## 坏实践

- 为"统一模板简洁"让所有 dtype 走同一条 fp32 中转通路——int 数据的精确表示边界是硬约束，模板分流是必须的。
- 见到"值偏移小量、索引全对"就怀疑搬运/对齐——偏移量恰为 2 的幂的倍数附近（26/6/14/38 对 64 间隔）是浮点舍入的指纹。

## 误诊

- 无。
