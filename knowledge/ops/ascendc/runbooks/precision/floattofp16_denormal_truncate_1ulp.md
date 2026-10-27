---
type: "Runbook"
title: FloatToFp16 denormal 路径截断致 1-ULP 偏差
description: 测试侧软浮点 FloatToFp16 的 denormal 分支只移位截断不做最近偶舍入，与硬件 Cast(CAST_RINT) 差 1-ULP，denormal 区相对误差可达 33%/0.596 而超阈；denormal 分支补齐 round-to-nearest-even（含 denormal→normal 进位链）即修复。
tags: [task.testing, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T00:54:40Z'
platforms: ["950"]
---

# FloatToFp16 denormal 路径截断致 1-ULP 偏差

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子 bench_no_harness_sigmoid 的 fp16 用例做精度排查：golden 已量化到输出 dtype（[golden 未量化到输出 dtype](golden_quantize_to_output_dtype.md) 卡修复，mismatches 从 459 降到 152）后，7777 元素 fp16 用例仍有 152 个 mismatch，且 MARE 恶化到 5.96e-01：

```
[FAIL] MERE=3.67e-03 (>= 9.77e-04), MARE=5.96e-01 (>= 9.77e-03), 7777 elems, 152 mismatches
    mismatch [36]: golden=1.192093e-06, actual=1.251698e-06
    mismatch [103]: golden=4.327297e-05, actual=4.333258e-05
    mismatch [139]: golden=1.788139e-07, actual=2.384186e-07
    mismatch [194]: golden=4.768372e-07, actual=5.364418e-07
    mismatch [950]: golden=0.000000e+00, actual=5.960464e-08
```

当时代码状态：kernel 走"fp32 计算 + 末尾 Cast(B32→B16, RoundMode=CAST_RINT)"，CAST_RINT 是最近偶舍入（round-to-nearest-even）；测试侧用自写的软浮点 `FloatToFp16`（位级 fp32→fp16）做 golden 量化与输入转换。

## 适用场景

- 工程形态：aclnn 自定义算子 ST 脚手架里的软浮点 dtype 模拟（fp32↔fp16 位级转换）+ 精度比对。
- API 路径：自建 `FloatToFp16`/`SimulateDtype` 转换；对照硬件侧 `AscendC::Reg::Cast` + `CastTrait{RoundMode::CAST_RINT}`（DAV_3510 vconv）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有在 host 侧用位级软浮点模拟 NPU 低精度行为的测试代码。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：bf16 转换（本卡只实测 fp16 分支；bf16 denormal 语义不同需单独核对）；硬件 Cast 的 CAST_RINT 以外舍入模式。

## 根因

测试侧 `FloatToFp16` 的 **denormal 分支只做移位截断、没有舍入**，而硬件 `Cast(B32→B16, CAST_RINT)` 对 denormal 做最近偶舍入——两侧在 fp16 denormal 区系统性相差 1 ULP。

机制链（逐项算式）：

1. 失配值全部落在 fp16 denormal 区（数值 = mantissa × 2^-24，最小步长 2^-24≈5.96e-8）：
   - golden=1.192093e-06 = 20×2^-24，actual=1.251698e-06 = 21×2^-24 → 差 1 ULP；
   - golden=1.788139e-07 = 3×2^-24，actual=2.384186e-07 = 4×2^-24 → 差 1 ULP；
   - golden=0，actual=5.960464e-08 = 1×2^-24 → 零边界差 1 ULP。
2. denormal 区 1-ULP 的相对误差可以远超 2^-10 阈值：3×2^-24 → 4×2^-24 相对差 |4-3|/3 = **33.3%**；golden=0 → 1×2^-24 经比对公式 1e-7 稳定项放大成 relErr = 5.96e-8/(0+1e-7) = **0.596**（即实测 MARE=5.96e-01 的来源）。
3. `FloatToFp16` 的 normal 分支有完整的最近偶舍入（`remainder > 0x1000 || (remainder == 0x1000 && (mant & 1))` 时进位），denormal 分支却只有 `mant >>= shift`—— halfway 及以上的尾数被直接丢掉，量化后的 golden 系统性偏小 1 ULP，与硬件 CAST_RINT 不一致。

## 判别方法

1. 把每个失配的 golden/actual 除以 2^-24：商都是相邻整数 → 全部是 denormal 区 1-ULP 差（非随机错误）。
2. 读 `FloatToFp16`：normal 分支有 RNE 进位逻辑、denormal 分支（`exp <= 0`）只有 `mant |= 0x800000; mant >>= shift;` 无舍入 → 分支不对称即根因。
3. 修复后重跑：fp16 mismatches 归零。

## 解决配方

修复前（`tests/st/test_aclnn_bench_no_harness_sigmoid.cpp` 的 `FloatToFp16`，denormal 分支无舍入）：

```cpp
uint16_t FloatToFp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return sign;  // too small → zero
        mant |= 0x800000;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        mant >>= shift;
        return sign | static_cast<uint16_t>(mant & 0x3FF);
    } else if (exp >= 31) {
        if (mant == 0) return sign | 0x7C00;  // Inf
        return sign | 0x7C00 | 0x200;          // NaN
    }
    mant >>= 13;
    uint32_t remainder = bits & 0x1FFF;
    if (remainder > 0x1000 || (remainder == 0x1000 && (mant & 1))) {
        mant++;
        if (mant > 0x3FF) { mant = 0; exp++; }
        if (exp >= 31) return sign | 0x7C00;  // overflow → Inf
    }
    return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant);
}
```

修复后（denormal 分支补齐 round-to-nearest-even，与硬件 CAST_RINT 行为一致）：

```cpp
uint16_t FloatToFp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return sign;  // too small → zero
        mant |= 0x800000;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        // round-to-nearest-even for denormal values (匹配硬件 CAST_RINT 行为)
        uint32_t remainder = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        mant >>= shift;
        if (remainder > halfway || (remainder == halfway && (mant & 1))) {
            mant++;
            if (mant > 0x3FF) { mant = 0; exp = 1; }
        }
        if (exp >= 1) return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant);
        return sign | static_cast<uint16_t>(mant & 0x3FF);
    } else if (exp >= 31) {
        if (mant == 0) return sign | 0x7C00;  // Inf
        return sign | 0x7C00 | 0x200;          // NaN
    }
    mant >>= 13;
    uint32_t remainder = bits & 0x1FFF;
    if (remainder > 0x1000 || (remainder == 0x1000 && (mant & 1))) {
        mant++;
        if (mant > 0x3FF) { mant = 0; exp++; }
        if (exp >= 31) return sign | 0x7C00;  // overflow → Inf
    }
    return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant);
}
```

要点：denormal 的 RNE 进位可能把尾数推满（mant > 0x3FF），此时结果从 denormal 溢出为最小 normal（mant=0, exp=1），分支里必须处理该进位链。

## 设计决策

- 修软浮点转换而不是给 denormal 区加绝对误差豁免：豁免会同时掩盖 kernel 侧真正的 denormal 舍入错误；对齐舍入模式后 1-ULP 差异从根上消失，判据保持逐元素相对误差。
- halfway 判定用 `remainder > halfway || (remainder == halfway && (mant & 1))`（标准 RNE：大于一半进位，等于一半向偶数进位），与 normal 分支的 `0x1000` halfway 逻辑同构，也与 numpy `astype`/硬件 CAST_RINT 的 RNE 一致。
- 当前方案路线档位：recommended_fastpath。

## 预防

- host 侧写位级 fp32→fp16 软浮点时，normal 与 denormal 两条分支都要实现 RNE（denormal 的 halfway 是 `1u << (shift-1)`，进位还要处理 denormal→normal 溢出）；写完先用已知 halfway 值（如 3.5×2^-24、20.5×2^-24）自测。
- 凡是与硬件 Cast(CAST_RINT) 对拍的软浮点转换，优先用编译器/标准库现成路径（如 `_cvtss_sh`、`_Float16`/`__bf16` 原生转换，见 [CPU golden 对齐 torch：RNE 舍入与 rint](../validations/cpu_golden_align_torch_rne_rint.md)）或至少交叉验证一个 denormal halfway 样本。
- kernel 侧手写转换的同类缺陷（次正规阈值/移位/normal 舍入变体）见 [FloatToHalf 次正规阈值漏 exp102](floattohalf_subnormal_threshold_exp102.md) 与 [手写 fp32→fp16/bf16 舍入未实现 RNE](handwritten_float_cast_rne_roundbit_bias.md)——本卡是测试侧软浮点的 denormal 分支舍入不对称，与上述 kernel 侧卡互为姊妹。

## 验证信号

以下信号来自该算子 fp16/bf16 扩展轮的 kernel 联调会话：golden 量化修复后的残余失配分析 → denormal RNE 修复 → 复跑。

- 修复前：7777 元素 fp16 用例 152 mismatches，全部为 denormal 区 1-ULP（20→21、3→4、8→9、0→1 倍 2^-24），MARE=5.96e-01 由 golden=0 vs actual=5.96e-8（1e-7 稳定项放大）造成（来源平台：950；证据等级：tool_observed）。
- 修复后：fp16 用例 mismatches 152→0；ST Real 72/75（剩余 3 个为空 Tensor 脚手架问题，同会话另卡），阈值未放宽（来源平台：950；证据等级：tool_observed）。
- 最终回归：空 Tensor 修复后 ST Real 75/75，fp16 全部 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只实测 fp16 分支；`FloatToBf16` 的 denormal/舍入路径未在同会话深挖（bf16 阈值宽，未暴露失配），移植前需单独核对。
- 硬件侧行为以 CANN 9.1.0 DAV_3510 的 `Cast + CAST_RINT` 为准；其他 RoundMode（CAST_ROUND/CAST_FLOOR 等）与本卡不一致。
- 极小值（|f| < 2^-25）两侧都舍到 0，不在本卡讨论范围。

**未决项**：

- 无（本根因已闭环）。
## 坏实践

- 软浮点转换只写 normal 路径的舍入、denormal 路径裸截断（两条分支舍入不对称是系统性偏差源）。
- 发现 denormal 区失配就把该区间从统计里剔除（用例删减式"修复"）。

## 误诊

- 上一轮（golden 未量化）排查时曾把 MARE≈0.22~0.6 归因为 kernel 尾块处理/LDIST_UNPACK_B16 搬运语义错误（7777=121×64+33 等尾块规律引发怀疑）：失配样本逐个量化分析后证明全部是 1-ULP 表示差，与搬运/尾块无关（该假设为 agent 归纳，被失配值分布证伪）。
