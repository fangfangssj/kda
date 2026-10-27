---
type: "Runbook"
title: "手写 fp32→fp16/bf16 舍入未实现 RNE 偏 1 ulp"
description: "手写位级 fp32→fp16/bf16 转换的 round-to-nearest-even 实现错误——FloatToHalf 用截断尾数 LSB 当进位、FloatToBf16 的 tie 偏置方向接反，约半数转换值偏 1 ulp；按 guard 位/tie-even 修正后 4 个 fp16 用例转 PASS。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# 手写 fp32→fp16/bf16 舍入未实现 RNE 偏 1 ulp

## 现象

在 Ascend950PR（CANN 9.2.0）上修 BenchHarnessScatter 的黑盒失败用例时，因 AscendC 的 `bfloat16_t`/`half` 不支持标量 `static_cast`（见 [bisheng 不支持 bf16 标量强转卡](../compilation/bf16_scalar_to_fp32_bit_extension.md)），kernel 里手写了位级 fp32↔fp16/bf16 转换。修完拷贝竞争与多核竞争后 TTK 剩 11 个失败用例，其中 4 个 fp16 用例（含 data=-65504（fp16 最大负数）边界值用例）在单值 dump 里表现为 1~2 ulp 量级的小偏差（如期望 0.004001617、实际 0.004005432，diff≈0.00095，恰好超 fp16 阈值 2^-10）。当时的 `FloatToHalf` normal 分支写着（roundBit 取的是截断结果的 LSB，不是 guard 位）：

```cpp
        uint32_t roundBit = (mant >> 13) & 1u;
        uint32_t hm = (mant >> 13) + roundBit;
```

## 适用场景

- 工程形态：AscendC kernel 内手写位级浮点转换（因 half/bf16 无标量算术/转换运算符而不得不自实现）+ TTK stat_rel_err 精度评测。
- API 路径：`__builtin_memcpy` 位视图 + IEEE 754 binary16/bfloat16 布局换算；golden 用 numpy `astype`（round-to-nearest-even）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何在 kernel 里手写 fp32→fp16/bf16 舍入的场合，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：使用 `AscendC::Cast` 向量 API 或硬件转换指令的路径（其舍入语义由 API 保证，不在本卡）；其他 CANN 版本未验证（推断可迁移）。

## 根因

手写的舍入没有实现 round-to-nearest-even：`FloatToHalf` 把"截断后 10 位尾数的最低位"（mant 的 bit 13）当成进位位——截断结果为奇数时无条件 +1（哪怕被丢弃的 13 位全是 0），为偶数时永不进位（哪怕丢弃部分已超过一半）；`FloatToBf16` 的加偏置法把 tie（丢弃位恰为一半）时的偏置方向接反（lsb 为 0 时用 0x8000，tie 会进位到奇数）。

机制链：fp32 尾数 23 位 → fp16 尾数 10 位，丢弃 13 位；正确 RNE 为 `dropped > halfway || (dropped == halfway && (tm & 1))` 才进位（halfway = 1<<12 = 0x1000）。错误实现下，凡截断尾数为奇数的值都被抬高 1 ulp（约占一半编码），凡为偶数的值即使超出 halfway 也被截断——系统性偏离 numpy `astype`。

## 解决配方

修复前（verbatim，含 FloatToBf16 错误偏置与 FloatToHalf 错误进位）：

```cpp
__aicore__ inline bfloat16_t FloatToBf16(float f)
{
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even
    uint32_t lsb = (u >> 16) & 1;
    uint32_t bias = lsb ? 0x7FFFu : 0x8000u;
    uint16_t h = static_cast<uint16_t>((u + bias) >> 16);
    bfloat16_t r;
    __builtin_memcpy(&r, &h, sizeof(r));
    return r;
}
```

```cpp
    } else if (exp >= 113) {                           // normal half
        uint32_t he = exp - 127 + 15;
        uint32_t roundBit = (mant >> 13) & 1u;
        uint32_t hm = (mant >> 13) + roundBit;
        if (hm > 0x3FFu) { hm = 0; he++; }
        if (he >= 31) { h = static_cast<uint16_t>((sign << 15) | 0x7C00u); }
        else { h = static_cast<uint16_t>((sign << 15) | (he << 10) | hm); }
    }
```

修复后（标准 RNE，verbatim 取自会话中修正版）：

```cpp
__aicore__ inline bfloat16_t FloatToBf16(float f)
{
    // bf16 = high 16 bits of fp32, round-to-nearest-even on the low 16 bits.
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    uint32_t lsb      = (u >> 16) & 1u;        // LSB of the bf16 mantissa
    uint32_t dropped  = u & 0xFFFFu;            // 16 bits to discard
    uint32_t halfway  = 0x8000u;                // 1 << 15
    // carry (round up) iff dropped > halfway, or (tie and lsb==1)
    uint32_t bias = lsb ? halfway : (halfway - 1u);
    uint16_t h = static_cast<uint16_t>((u + bias) >> 16);
    bfloat16_t r;
    __builtin_memcpy(&r, &h, sizeof(r));
    return r;
}
```

```cpp
    } else if (exp >= 113) {                           // normal half
        uint32_t he = exp - 127 + 15;
        uint32_t tm = mant >> 13;                      // truncated 10-bit mantissa
        uint32_t dropped = mant & 0x1FFFu;             // 13 discarded bits
        uint32_t halfway = 0x1000u;                    // 1 << 12
        if (dropped > halfway || (dropped == halfway && (tm & 1u))) { tm++; }
        if (tm > 0x3FFu) { tm = 0; he++; }             // mantissa carry
        if (he >= 31) {                                // overflow → inf
            h = static_cast<uint16_t>((sign << 15) | 0x7C00u);
        } else {
            h = static_cast<uint16_t>((sign << 15) | (he << 10) | tm);
        }
    }
```

验证偏置法正确性的算式（bf16，lsb=偶数、tie 场景）：u 低 16 位 = 0x8000，bias = 0x7FFF，和 = 0xFFFF 无进位 → 保持偶数 ✓；lsb=奇数、tie：0x8000 + 0x8000 = 0x10000 有进位 → 进到偶数 ✓；丢弃位 0x8001、lsb=偶：0x8001 + 0x7FFF = 0x10000 进位 ✓。

## 设计决策

- 为什么自己写转换而不是用 `AscendC::Cast` 向量 API：reduce 是单步标量运算（每元素读-改-写），向量 Cast 需要整块缓冲且当时 UB 预算紧张；scalar 上下文无可用转换 API（SIMT 的 `__cvt_float/__cvt_half` 在常规 kernel 不可用）。
- 为什么坚持对齐 numpy `astype`：golden（`_compute_dtype_for` 后 `y.astype(原dtype)`）是 RNE；kernel 端任何其他舍入模式都会在 stat_rel_err 阈值（fp16 2^-10、bf16 2^-7）下暴露。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 手写位级舍入先写 4 个用例自测：丢弃位全 0（不进位）、丢弃位刚过一半（进位）、恰为一半且尾数偶（不进位）、恰为一半且尾数奇（进位）。
- 对照实现直接抄 numpy/IEEE 754 RNE 语义，不要"加个 roundBit"式即兴写法。

## 验证信号

以下信号来自该算子的一次黑盒失败用例修复会话：TTK 单用例 dump 定位偏差量级，修正舍入后全量回归。

- 修复前：4 个 fp16 用例（含 -65504 边界值用例）单值偏差 ~0.00095（≈1 ulp@0.004），超 fp16 阈值 2^-10（来源平台：950；证据等级：tool_observed）。
- 修复后：TTK 正向 388→392/399（11→7 失败，4 个边界值用例转 PASS）；既有 3 个复现用例保持全 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖 normal 区间的舍入进位与 bf16 偏置方向；fp16 **次正规区间**另有独立缺陷（阈值与移位公式，见 [FloatToHalf 次正规阈值卡](floattohalf_subnormal_threshold_exp102.md)），两卡需同时修正才能全绿。
- 溢出上界（[65504, 65520) 应舍到 65504、≥65520 舍到 Inf）在本会话按 RNE 推导核对过，未单独构造探针实测。
- 仅 Ascend950PR + CANN 9.2.0 实测（未验证，推断可迁移）。

**未决项**：

- 无（本根因已机制级闭环；次正规区间缺陷见上卡）。
## 坏实践

- `roundBit = (mant >> 13) & 1` 式"看保留位最低位决定进位"——它不是 guard 位，丢弃位信息完全没参与判断。
- 偏置法里不区分 lsb 奇偶的固定 bias——tie 时方向错。
