---
type: "Runbook"
title: "FloatToHalf 次正规阈值漏 exp102 致 mare=0.373"
description: "位级 fp32→fp16 转换的次正规分支阈值写成 exp>=103 会把 [2^-25,2^-24) 的值 flush 为 0，偏离 numpy astype 的舍入，产生特征值 mare≈0.373；阈值应为 exp>=102。R2（CANN 9.2.0 独立复现）：同分支移位公式写成 113−exp（应为 126−exp）时最小次正规 0x0001 往返变 16.0，全编码探针精确复现单点失配。R3（宿主侧独立复现，平台无关 IEEE 事实）：读方向 Float16ToFloat 非规格化分支基指数写成 127-15-e（应为 127-14-e，即 112-e 应为 113-e），round-trip 值恒减半（0x0060→0x0030）、MARE≈0.499 系统性误报；10 万随机位型 round-trip 探针修复前 bad=3146/96889、修复后 bad=0。R4（host UT 模拟 kernel Cast，CANN 9.1.0）：写方向正规数路径误加隐含位（mantissa |= 0x800000 后 >>13）使半精度尾数多出 bit10、进位后指数+1——所有可精确表示值精确翻倍（0.5→1、1→2、0.25→0.5）；正规数只移 23 位 fraction，隐含位只在次正规路径补。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-14T02:36:53Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# FloatToHalf 次正规阈值漏 exp102 致 mare=0.373

## 现象

在 Ascend950PR（CANN 9.2.0）上修 BenchHarnessScatter 的黑盒失败用例时，fp16 multiply 大用例（(22186,36,61)、dim=-1、48.7M 元素、data 范围 (-10,-2)、updates (-0.01,-0.001)）在实施了 fp32 workspace 累加路径（精度模型已对齐 golden）之后仍然失败，且指标呈现两个特征：

- mare=0.373451828956604 与修复前**逐位相同**，mere 在 0.0008048~0.0008096 之间微小波动，报错原文 `reason: 'threshold exceeded: mare(3.73e-01>=0.009765625)'`；
- 自建 aclnn 诊断程序在**同 shape 全规模**（48,720,456 元素）下对自建 host golden 却是 mism=0/48720456、maxRel=0——kernel 与诊断 golden 完全一致，却与 TTK 的 numpy golden 不一致。

kernel 中自带的位级 fp32→fp16 转换 `FloatToHalf` 当时把次正规（subnormal）分支的阈值写成 `exp >= 103`（fp32 biased exponent），`exp < 103` 一律按 underflow 处理为 0。

## 适用场景

- 工程形态：aclnn 自定义算子 kernel 内手写位级浮点转换（不经编译器 builtin），用 numpy/torch 作精度 golden 的黑盒评测。
- API 路径：自实现 `FloatToHalf`（fp32→fp16 位级转换，round-to-nearest-even）；对照物为 `numpy.ndarray.astype(np.float16)` / `ml_dtypes`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何在该平台手写 fp32→fp16 转换的 kernel，与具体算子无关（机制是 IEEE-754 半精度次正规舍入边界，非算子语义）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：bf16 转换（bf16 指数域与 fp32 相同，无次正规边界问题，本轨迹未涉及 bf16 阈值 bug）；使用编译器/硬件内置转换指令的路径（语义由实现保证，未验证）；非 round-to-nearest-even 舍入模式。

## 根因

`FloatToHalf` 的次正规分支阈值 `exp >= 103` 把 fp32 biased exponent 为 102（值域 [2^-25, 2^-24)）的数错误地 flush 为 0，而 numpy `astype(fp16)` 对该区间按 round-to-nearest-even 舍入到最小次正规数 2^-24，两者在 [2^-25, 2^-24) 区间分歧。

机制链与算式：

- fp32 biased exponent e 的值域：v = 2^(e-127)×(1+mant/2^23)。e=113 ⇒ v∈[2^-14, 2^-13)（fp16 最小正规数 2^-14 对应 e=113）；e=103 ⇒ v∈[2^-24, 2^-23)；e=102 ⇒ v∈[2^-25, 2^-24)。
- numpy 行为（实测探针）：`fp16(5.96e-8) = 6e-08`（= 2^-24，最小次正规），`fp16(2.98e-8) = 0.0`（2^-25 恰为 halfway，round-to-even 落 0）。即 v∈(2^-25, 2^-24) → 2^-24；v = 2^-25 → 0；v < 2^-25 → 0。
- 阈值 103 的 kernel：v∈[2^-25, 2^-24)（e=102）→ 0；numpy → 2^-24。分歧区间即 (2^-25, 2^-24)。
- mare 特征值推导：golden=2^-24、kernel=0 时，rel = |0−2^-24|/(2^-24+1e-7) = 5.96e-8/(5.96e-8+1e-7) = 5.96/15.96 ≈ 0.3734——与实测 mare=0.373451828956604 吻合。高碰撞 fp16 multiply（data∈(-10,-2)、updates∈(-0.01,-0.001)，多 update 连乘）大量产生落入该区间的中间积，故 mere 极小（绝大多数位置无关）而 mare 稳定在 0.373。
- 次正规舍入的位级算式（修复后分支）：shift = 126−e；对 e=102，shift=24，m = mant|0x800000（24 位）；tm = m>>24（此时恒 0），dropped = m，halfway = 1<<23 = 0x800000；m > 0x800000（即 v > 2^-25）⇒ tm 进位为 1 ⇒ h = 2^-24；m == 0x800000（v == 2^-25）⇒ round-to-even，tm=0 保持 0。与 numpy 逐点一致。

## 解决配方

修复前（`bench_harness_scatter_kernel.h` 的 `FloatToHalf` 次正规分支）：

```cpp
    } else if (exp >= 103) {                           // subnormal half (exp 103..112)
        // fp16 subnormal mantissa = m >> (126 - exp), where m = mant | implicit 1.
        // shift = 23 for exp=103 (smallest subnormal) down to 14 for exp=112.
        int shift = static_cast<int>(126 - exp);
        uint32_t m = mant | 0x800000u;                 // implicit 1
        uint32_t tm = m >> shift;                      // subnormal mantissa (may carry to normal)
        uint32_t dropped = m & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (dropped > halfway || (dropped == halfway && (tm & 1u))) { tm++; }
        h = static_cast<uint16_t>((sign << 15) | tm);
    } else {                                           // underflow → 0
        h = static_cast<uint16_t>(sign << 15);
    }
```

修复后（阈值 103→102，注释写明边界算式）：

```cpp
    } else if (exp >= 102) {                          // subnormal half (exp 102..112)
        // fp16 subnormal mantissa = m >> (126 - exp), where m = mant | implicit 1.
        // exp=103..112 → v in [2^-24, 2^-14); exp=102 → v in [2^-25, 2^-24) which
        // numpy rounds to the smallest subnormal 2^-24 (v > 2^-25) or 0 (v == 2^-25,
        // round-to-even).  Threshold must include exp=102 (NOT 103) else values in
        // (2^-25, 2^-24) are wrongly flushed to 0, diverging from numpy astype and
        // producing mare = 2^-24/(2^-24+1e-7) ≈ 0.373 on high-collision fp16 multiply.
        int shift = static_cast<int>(126 - exp);
        uint32_t m = mant | 0x800000u;                 // implicit 1
        uint32_t tm = m >> shift;                      // subnormal mantissa (may carry to normal)
        uint32_t dropped = m & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (dropped > halfway || (dropped == halfway && (tm & 1u))) { tm++; }
        h = static_cast<uint16_t>((sign << 15) | tm);
```

配套判别探针（numpy 行为锚定，可在任何有 numpy/ml_dtypes 的环境复跑）：

```bash
python3 - <<'PY'
import numpy as np, ml_dtypes
# numpy 不 FTZ
a = np.float32(2.0**-127)
print("2^-127 * 4 =", a * np.float32(4.0), "(expect 2^-125=5.88e-38 if no-FTZ, 0.0 if FTZ)")
# fp16 次正规舍入边界
print("fp16(5.96e-8) =", np.float32(5.96e-8).astype(np.float16),
      "fp16(2.98e-8) =", np.float32(2.98e-8).astype(np.float16))
PY
# 期望输出：fp16(5.96e-8) = 6e-08（=2^-24），fp16(2.98e-8) = 0.0
```

### 独立复现 R2：同分支移位公式 113−exp 致 0x0001 往返变 16.0

同一算子家族的姊妹会话（同为手写 `FloatToHalf`，同为次正规分支）踩的是**移位公式**而非阈值：`shift = 113 - exp`（取值 1..10，隐含 1 在 bit 10）应为 `shift = 126 - exp`（取值 23..14，隐含 1 在 bit 23）。失败签名与阈值 bug 不同——不是 flush 为 0，而是出现 **2 的幂垃圾值**：期望 0.0000231862（fp16 次正规，编码 0x0017 一类）、实际 32.0；期望 0.0000045、实际 -2.0。全编码探针（把全部 1023 个正/负 fp16 次正规编码 0x0001..0x03FF、0x8001..0x83FF 加最小正规 0x0400/0x8400 和 ±0 共 2050 个值作为 data，走 amax + 全 -Inf updates——数学上 no-op，输出应逐位等于输入）精确复现单点失配：

```
  [0x0001] got=0x2000 exp=0x0001
subnormal round-trip test: FAIL (mism=1/2050)
```

最小正次正规 0x0001（2^-24）经 kernel 往返后变成 0x2000（16.0 = 2^4），差 2^28 倍——量级恰好是移位差 13 位（`0x800000 >> 10 = 0x2000`，远超 10 位尾数上限 0x3FF，直接把尾数写成了指数位）。

修复（次正规分支，verbatim，仅移位公式与注释变化，舍入逻辑不动）：

```cpp
    } else if (exp >= 103) {                           // subnormal half (exp 103..112)
        // fp16 subnormal mantissa = m >> (126 - exp), where m = mant | implicit 1.
        // shift = 23 for exp=103 (smallest subnormal) down to 14 for exp=112.
        int shift = static_cast<int>(126 - exp);
        uint32_t m = mant | 0x800000u;                 // implicit 1
        uint32_t tm = m >> shift;                      // subnormal mantissa (may carry to normal)
        uint32_t dropped = m & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (dropped > halfway || (dropped == halfway && (tm & 1u))) { tm++; }
        h = static_cast<uint16_t>((sign << 15) | tm);
    } else {                                           // underflow → 0
        h = static_cast<uint16_t>(sign << 15);
    }
```

注：`tm` 进位到 0x400 时 `(sign<<15)|tm` 的编码自然落成最小正规（指数位=1、尾数=0），无需特判。读方向 HalfToFloat 的次正规正规化基指数（113 而非 112）为同会话配套修正，经同一探针验证。

R2 的探针设计要点（可复用）：amax + 全 -Inf updates 是 no-op reduce，把"kernel 里的 dtype 转换"从 scatter 语义中完全剥离——任何输出失配都只能是转换本身造成的；次正规区间只有 2048 个编码（含符号），**全枚举零死角**且失败输出直接给出出错编码（受控随机数据恰好没命中 0x0001，若停在随机复现就会误判 kernel 正确）。探针骨架与直连 aclnn 复现程序的编译运行命令见 [aclnn 直连复现骨架卡](../compilation/aclnn_direct_repro_skeleton_noop_probe.md)。

## 设计决策

- 为什么逐位手写转换而不是依赖内置转换：该算子 reduce 路径在 S-pipe 标量域做 RMW，half/bf16 无原生标量算术，需要显式 `HalfToFloat/FloatToHalf` 位级往返；既然手写，就必须把 IEEE-754 次正规舍入边界（halfway=2^-25、最小次正规=2^-24、round-to-nearest-even）全部实现对齐 numpy。
- 判别依据选择"mare 与 golden/kernel 值的特征比"而非盲试：mare=0.373 恰等于 2^-24/(2^-24+1e-7)，直接锁定"golden=2^-24、kernel=0"这一唯一分歧形态，再由 numpy 探针确认 2^-25 舍入边界，一步定位到 exp=102 区间。
- 当前方案路线档位：recommended_fastpath（一处阈值/移位修正，无代价）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；numpy + ml_dtypes 作 golden。
- 相关文件：`op_kernel/arch35/bench_harness_scatter_kernel.h`（`FloatToHalf`）；诊断程序 `tests/aclnn/repro_diag.cpp`（其 host golden 的 `f32_to_f16` 也曾缺次正规分支，属同类坑，见误诊）。
- R2 探针：`tests/aclnn/repro_subnormal.cpp`（全编码次正规往返探针）、`tests/aclnn/repro_fp16.cpp`（受控数据对照，未命中 0x0001）。

## 预防

- 手写任何低位宽浮点转换前，先用 numpy 探针固定三个锚点：最小正规数、最小次正规数、halfway 值的 round-to-even 行为，再写分支阈值。
- fp32→fp16 的次正规区间换算口诀：fp16 次正规覆盖 fp32 biased exponent 102..112（v∈[2^-25, 2^-14)），边界值 2^-25 按 round-to-even 落 0；阈值写 103 即漏掉 [2^-25, 2^-24)。
- 次正规移位公式凭直觉写不推导：隐含 1 在 bit 23，目标位宽差必须显式算出（126−exp）。
- 看到"mare 稳定等于某特征比值、且跨运行逐位不变"的精度失败，优先怀疑确定性的舍入边界分歧，而不是随机性丢写（后者 mare 会随数据波动）。

- 与硬件/固有指令的 FTZ 行为区分：本卡根因是 kernel 自写转换代码的阈值 bug，不是向量指令 FTZ（后者见 [Ascend950 向量管 FTZ 刷新次正规 exp 为零卡](ascend950_ftz_exp.md)、[Exp 默认 INTRINSIC FTZ 致 subnormal 残留失配卡](exp_intrinsic_ftz_subnormal.md)）；两条判据：本例 mare 特征值 0.373 恰为舍入边界比值、且探针证明 numpy 不 FTZ。
- normal 区间舍进位（roundBit 取截断 LSB）与 bf16 tie 偏置方向是同函数另一独立缺陷，见 [手写 fp32→fp16/bf16 舍入未实现 RNE 卡](handwritten_float_cast_rne_roundbit_bias.md)。

## 验证信号

以下信号来自该算子 blackbox-fix 会话：fp32 workspace 路径落地后用例仍失败，经 numpy 探针锚定舍入边界、修正 `FloatToHalf` 阈值后单用例与全量回归。

- 修复前：TTK 下该 fp16 multiply 用例 mere=0.0008056、mare=0.373451828956604、threshold=0.0009765625 超限 FAIL，且 mare 与修复前逐位相同（来源平台：950；证据等级：tool_observed）。
- 修复前：numpy 探针实测 `fp16(5.96e-8) = 6e-08`、`fp16(2.98e-8) = 0.0`、numpy 不 FTZ（2^-127×4 = 2^-125 保留）（来源平台：950；证据等级：tool_observed）。
- 修复后：该用例 mere=0.0、mare=0.0、PASS（与 numpy golden 完全一致）（来源平台：950；证据等级：tool_observed）。
- 修复后：14 用例子集 14/14 PASS；正向全量 399/399 PASS（ttk aclnn --pc=1 --proc-timeout 300）（来源平台：950；证据等级：tool_observed）。

- 独立复现 R2（姊妹会话，移位公式变体）：修复前 2050 个边界编码探针恰 1 个失配 `[0x0001] got=0x2000`（2^-24 → 16.0）；修正 shift 113→126 后同探针 0/2050 失配（全编码往返恒等）；TTK 正向 395→397/399，两个 fp16 用例（amax -Inf 次正规 data、amin 阈值摆动用例）转 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修复针对 round-to-nearest-even 语义；若目标 golden 使用截断（RTZ）或其他舍入模式，边界与阈值须重新推导（未验证）。
- 仅在 950 + CANN 9.2.0 + numpy/ml_dtypes golden 实测；机制是 IEEE-754 语义层面的，推断可迁移至任何手写 fp32→fp16 的 AscendC kernel（未验证，推断可迁移）。
- 值恰好等于 2^-25（halfway）时 round-to-even 落 0，golden 与修复后 kernel 一致；若业务期望"恰好一半也进位"，则与 numpy 语义冲突，属 golden 定义问题而非 kernel bug。
- 溢出上界（[65504, 65520) 应舍到 65504、≥65520 舍到 Inf）在 R2 会话按 RNE 推导核对过，未单独构造探针实测。

**未决项**：

- 无（根因已追到 IEEE-754 舍入边界机制层并闭环验证；R2 移位变体亦经全编码探针闭环）。


### 独立复现 R3：读方向 Float16ToFloat 非规格化基指数 127-15-e（宿主侧，IEEE 位型事实）

稀疏注意力算子 C++ ST 工程（Mock 模式，CPU golden 自测，actual = golden 经 fp16/bf16 写入读回的 round-trip）的手写 `Float16ToFloat` 在非规格化分支踩了与 R1/R2 同族的第三处坑——**读方向**的规格化基指数：

```cpp
// 修复前（非规格化分支，verbatim）：
            mant &= 0x3FFU;
            f = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
// 修复后：
            mant &= 0x3FFU;
            // fp16 非规格化基数是 2^(-14)（非 2^(-15)），归一化后 exp = 127 - 14 - e
            f = (sign << 31) | ((uint32_t)(127 - 14 - e) << 23) | (mant << 13);
```

机制：fp16 非规格化值 = (-1)^sign × **2^(-14)** × (mant/1024)，基数是 2^-14 而非 2^-15（偏置 15 是指数域偏置，不是非规格化基数）。手写实现把 mant 规格化（左移至隐含位、计 e 次移位）后，float 指数应为 `127 - 14 - e`；写成 `127 - 15 - e` 使结果**小了一倍**（指数偏 1），round-trip 时舍回最近可表示值恰好落在原值一半的位置（0x0060→0x0030），相对误差≈0.5。R2 会话配套修正的"读方向 HalfToFloat 次正规正规化基指数（113 而非 112）"即同一处——`127-14-e = 113-e` 与 `127-15-e = 112-e` 之差，本轨迹为独立复现。

判别签名：Mock 比对中 MARE **恒 ≈0.499**（"某个元素值差了一倍"的确定性指纹，非随机舍入）且失败集中在 exp==0 位型（h 高 6 位为 0x000~0x03F 区间）；`MARE 离散 0.03~0.23 且集中在近零行` 则是近零指标伪影（另见近零处置卡），两者同貌不同因——恒定≈0.5 先查转换实现位级正确性。探针：10 万随机 fp16 位型（跳过 inf/nan）round-trip，`bad>0` 且失配集中在 exp==0 即命中；正确实现 bad=0。

- 修复前探针：`bad=3146/96889, maxRel=4.988503e-01`，失配样本如 h=0x0060（f≈5.72e-6）→h2=0x0030（恰为一半）（来源平台：宿主机 g++ 探针（x86_64，IEEE 754 位型事实，平台无关，未上板）；证据等级：tool_observed）。
- 修复后探针：`bad=0/96889, maxRel=0.000000e+00`（同种子同规模）（来源平台：宿主机 g++ 探针；证据等级：tool_observed）。
- 主测试程序修复后 Mock 重跑：MARE≈0.499 的系统性误报消失（残留个别 MARE 偏大属近零指标伪影，另行处置）（来源平台：宿主机 Mock 比对；证据等级：tool_observed）。

边界增补：探针跳过 inf/nan 位型（NaN payload 语义未验证）；bf16 手写转换分支未做同规模 round-trip 审计（工程内同文件实现，建议同型探针覆盖）；"漂亮值"（1.0/0.5/0.25）全部命中规格化区间，测不出非规格化分支 bug——测试数据必须含随机位型或显式小值。


### 独立复现 R4：写方向正规数路径误加隐含位——往返值精确翻倍（host UT 的 SimulateFloatToHalf）

R4（grid_sampler_3d 类算子 nearest 插值 UT，Ascend950PR/CANN 9.1.0 目标语义的 host 模拟，x86_64 主机执行）：kernel 的 `__aicore__` 标量函数无法在 host 直接调用，UT 里复刻 host 端算法时 `SimulateFloatToHalf`（模拟 `Cast<half, float>(dst, src, CAST_ROUND, 1)`）在**正规数路径也执行 `mantissa |= 0x800000`（加隐含位）再 >>13**。fp32 的 23 位 fraction 加隐含位后 24 位，右移 13 位得到 11 位值——bit10 恰好是 fp16 的隐含位位置，落进本应只有 10 位的尾数字段：以 0.5 为例，mantissa=0 → 加隐含位 0x800000 → >>13 = 0x400 = 1024 > 0x3FF，触发进位分支把 halfExp 从 14 抬到 15，输出 0x3C00 = 1.0（正确应为 0x3800 = 0.5）。

症状指纹：**凡是 fp16 可精确表示的值全部 ×2**（0.5→1、-0.5→-1、1→2、0.25→0.5），nearest 的 index/mask 断言连锁失败（如 index 2 vs 3、mask 1 vs 0）——"往返值全部翻倍"是确定性的位级 bug 指纹，不是随机精度噪声，禁止据此放宽容差。

正确机制：两个格式的隐含位都由指数承载，**正规数转换只需把 23 位 fraction 右移 13 位（23-10），不加隐含位**；只有次正规路径（fp32 指数 < -14，fp16 尾数没有隐含位可用）才补隐含位并按 `shift = -1 - exp` 移位（推导：同为正规数且指数对齐时 `mantissa/2^23 = halfMantissa/2^10 ⇒ halfMantissa = mantissa >> 13`；次正规时 `halfM = (mantissa|0x800000) × 2^(exp+1)`）。次正规进位（halfM=0x400）显式映射到最小正规数 `(1u<<10)`。

修复要点（函数级）：

```cpp
    if (exp < -14) {
        // Subnormal in fp16: 需要补 fp32 隐含位，shift = -1 - exp，RNE 舍入，
        // 进位到 0x400 时直接返回最小正规数 (sign<<15)|(1u<<10)
    }
    // Normal in fp16: 不加隐含位，直接移位舍入
    int shift = 13;                        // 23 - 10
    uint16_t halfExp = static_cast<uint16_t>(exp + 15);
    uint32_t roundBit = (mantissa >> 12) & 1u;
    uint32_t stickyBit = (mantissa & 0xFFFu) ? 1u : 0u;
    uint16_t halfMantissa = static_cast<uint16_t>(mantissa >> shift);
    // RNE 进位、溢出转 inf 同常规
```

R4 验证信号：

- 首跑失败：4 个 fp16_cast 用例失败，往返值精确翻倍（0.5→1、-0.5→-1、1→2、-1→-2、0.25→0.5）（来源平台：950 目标 kernel 语义的 host 模拟，x86_64 主机执行；证据等级：tool_observed）。
- 修复后重跑：72/72 通过（fp16_cast 系列用例全过）（来源平台：950；证据等级：tool_observed）。
- 被模拟目标旁证：kernel 真实 Cast 路径在 ST Real 上板 40 个 fp16 用例全部达标（最大 MARE=8.850e-04 < 9.766e-04）（来源平台：950；证据等级：tool_observed）。

R4 边界增补：本卡只修 host 模拟的位级正确性；模拟（nearest-even）与 kernel 实际 RoundMode（CAST_ROUND，away-from-zero）在精确中点值上不同，逐位断言时需注意；移位常数只适用 binary32→binary16，bf16 需重推导；NaN 载荷只保留 1 位标记。预防：写 fp32→fp16 位级转换的铁律——正规数不加隐含位、次正规才补；写完用 0.5/1.0/0.25 三个可精确表示值做往返自测（翻倍即中招）。
## 坏实践

- 把次正规分支阈值写成"最小次正规数的指数"（103）而不是"会舍入进最小次正规的最低指数"（102）——漏掉整个 [2^-25, 2^-24) 舍入区间。
- 次正规移位公式不推导隐含 1 的位位置就写（113−exp 把 bit 23 的隐含 1 当 bit 10 移）。
- 用受控"看起来像"的数据代替全编码枚举测位级转换——随机/均匀值几乎不会命中 0x0001 这类编码角落。
- 用自建 host golden 验证手写转换时，host 参考实现自身缺次正规分支（同样 FTZ），kernel 与参考同错从而假阴性（本例诊断程序的 `f32_to_f16` 就犯过，修正前 TEST A 残留 20349 个"舍入"失配，修正后归零）。

## 误诊

- "TTK 结果与修复前逐位相同 ⇒ 新代码没生效 / 内核缓存过期"：曾据此清理 `/tmp/ops-test-kit/kernel_meta` 缓存并核对 vendor 预编译 .o 时间戳，均未解决；最终通过 plog 中的 kernel PRINTF 确认新路径确实在运行，失败是真实的舍入分歧。教训：先确认代码路径（见 [AscendC PRINTF 路由卡](../runtime/ascendc_printf_plog_routing.md)），再怀疑缓存。
- "自建全规模 repro mism=0 ⇒ kernel 正确、golden 有错"：repro 的 host golden 与 kernel 犯了同样的次正规 FTZ 错误，两边一致地错；对齐 numpy 后才暴露分歧。
- R2 会话曾把 fp16 amax 用例的 2 的幂垃圾值归因为 GM 写竞争 / DMA burst 越界写，先后加了拷贝后屏障、逐元素 PipeBarrier、DataCopyPad 改写均无效——失配位置随每轮 TTK 随机数据变化（因为只有落在次正规区间的位置才错），最终全编码探针一击定位。教训：**受控复现 PASS 而 TTK FAIL 时，优先怀疑数据分布差异（编码角落），而不是同步机制**。
