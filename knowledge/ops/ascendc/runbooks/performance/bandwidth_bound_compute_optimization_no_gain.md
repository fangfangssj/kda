---
type: "Runbook"
title: 带宽受限 kernel 计算链优化无带宽收益
description: "对已达带宽上限的 2R+1W elementwise kernel，把两分支 sigmoid（2 Exp+2 Div）重写为单 Exp 稳定链后精度全过但带宽无变化（优化后/优化前仅 1.003x，噪声级）——计算被 MTE 搬运完全隐藏；先证带宽受限再投计算优化。"
tags: [task.optimization, topic.performance.optimization.optimization_overview, topic.compute.elementwise.activation]
status: stable
sources: []
created_at: '2026-09-17T04:39:31Z'
updated_at: '2026-09-17T04:39:31Z'
platforms: ["950"]
---

# 带宽受限 kernel 计算链优化无带宽收益

## 现象

softplus_v2_grad（2R+1W elementwise 梯度算子，AscendC dav-3510 VF 链，56 核，双缓冲 tile-loop）在带宽验收时实测 L1 [4096,4096] fp32 device-only 带宽未达目标。当时的假设之一是计算链太重：原实现用两分支稳定 sigmoid——z≥0 走 `1/(1+exp(-z))`、z<0 走 `exp(z)/(1+exp(z))`，共 2 次 Exp + 2 次 Div + 多次 Mins/Adds。于是把 VF 函数重写为单 Exp 稳定链：`e = exp(-|z|)`，`s = 1/(1+e)`，`σ(z) = Select(z≥0, s, 1-s)`——1 Exp + 1 Div + 1 Sub + 1 Min，计算量约减半。

重写后：编译通过（途中修复一处 `Reg::Mins`→`Reg::Min` 误用）、全量 ST 25/25 精度全过、grep 验收项（`__simd_vf__`=2、`asc_vf_call`=2）满足——但带宽复测**与优化前实质不变（优化后/优化前 ≈1.003x，噪声级；参照路线=优化前两分支链，来源平台：950）**。约减半的计算量没有换来任何带宽提升。

随后 verifier 用控制变量实验定论：把该 VF 函数体替换为**纯拷贝**（同 tiling、56 核、2R+1W、零计算）也只比计算版快约 7.4%——计算版已达纯拷贝上界的 92.6%（参照路线=纯拷贝变体，来源平台：950），计算时间被 MTE 搬运完全隐藏，瓶颈在搬运不在计算。

## 适用场景

- 工程形态：AscendC elementwise/访存主导 kernel 的性能优化决策（何时投计算链优化、何时收手）。
- API 路径：`AscendC::Reg` 寄存器计算链（Exp/Div/Min/Sub/Select/Muls/Compares）经 `__simd_vf__` + `asc_vf_call`；带宽测量用 aclrtEvent kernel-only。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR，56 AIV）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（优化决策判据；具体数字仅对本算子本平台有效）。
- 路线优先级：`banned`——对已证实带宽受限的 kernel，继续投计算链精简属于无效功（不带来带宽收益）；应转向搬运路径（tile 大小、队列深度、核数、访存模式）。
- 明确不适用 / 未验证场景：计算受限 kernel（超越函数密集、读写比低、UB 重用高）不适用——那里计算链精简有真实收益；Cube/MIX 路线未验证。

## 根因（尝试与失败机制）

该 kernel 每 tile 的总时间 ≈ max(MTE 搬运时间, V 计算时间)（双缓冲下两流水重叠）。2R+1W 的搬运量固定（201326592 字节），MTE 时间是下界；原计算链虽比单 Exp 链重约一倍，但仍在 MTE 时间之内（V < MTE），总时间由 MTE 决定。把 V 从"MTE 的 x%"压到"MTE 的 x/2"，max() 不变，带宽不变。纯拷贝实验给出上界印证：零计算也只快约 7.4%（计算版为纯拷贝上界的 92.6%），即剩余差距来自计算与搬运的少量串行段，而非计算量本身。

## 尝试的实现（优化后计算链，verbatim）

```cpp
        // z = beta * x
        AscendC::Reg::Muls(z, xF, beta, mask);
        // maskGT = (z > threshold)  — strict >, z==threshold -> smooth branch
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(mGT, z, threshold, mask);

        // ---- stable sigmoid via single Exp: e = exp(-|z|), s = 1/(1+e) ----
        // -|z| = Min(-z, z) ≤ 0  → exp arg always ≤ 0, e ∈ (0,1], no overflow
        AscendC::Reg::Muls(t, z, -1.0f, mask);    // -z
        AscendC::Reg::Min(t, t, z, mask);          // Min(-z, z) = -|z|
        AscendC::Reg::Exp(t, t, mask);              // e = exp(-|z|) ∈ (0,1]
        AscendC::Reg::Adds(t, t, 1.0f, mask);       // 1 + e
        AscendC::Reg::Div(sig, one, t, mask);       // s = 1/(1+e) = sig_pos
        AscendC::Reg::Sub(sn, one, sig, mask);      // 1-s = sig_neg

        // sig = (z >= 0) ? sig_pos : sig_neg
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(mGE, z, 0.0f, mask);
        AscendC::Reg::Select(sig, sig, sn, mGE);  // dst=src0 (sig_pos) in-place

        // smooth = dy * sig ; dx = where(z > threshold, dy, smooth)
        AscendC::Reg::Mul(smooth, dyF, sig, mask);
        AscendC::Reg::Select(dxF, dyF, smooth, mGT);
```

（被替换的两分支链为：`sig_pos = 1/(1+exp(Mins(-z,0)))` 与 `sig_neg = exp(Mins(z,0))/(1+exp(Mins(z,0)))` 各自独立计算后 Select，共 2 Exp + 2 Div。）

## 设计决策

- 单 Exp 链本身的数值设计是健全的（exp 参数恒 ≤0 无溢出、精度 ST 全过），被保留在交付物中；被判无效的是"用它换带宽"的预期。
- 事后复盘的正确决策顺序：先做 5 分钟的纯拷贝 A/B 确认瓶颈侧，再决定投计算还是投搬运；本例顺序反了，先写了优化再发现无收益。
- 常量寄存器 `one` 在循环外 `Duplicate` 一次、循环内复用（Div/Sub 共用），是链内合理的寄存器 Economy，但对总时长无影响（同因）。

## 系统信息

- toolchain：CANN 9.1.0；设备 Ascend950PR（56 AIV、UB 253952B）；算子包 `softplus_v2_grad_package`。
- 相关文件：`op_kernel/softplus_v2_grad_kernel.cpp`（`SoftplusV2GradVF` 计算链）、`tests/aclnn/bench_aclnn.cpp`（带宽基准，验收后移除）。

## 预防

- 优化前先判瓶颈侧：同 tiling 纯拷贝变体 A/B（见 [无 HBM 峰值时用 copy-variant 对照判带宽达标](copy_variant_bandwidth_acceptance_without_peak.md)），比值 >90% 即带宽受限，跳过计算链优化。
- 报告优化收益时必须给同口径 before/after 数字，避免把噪声当收益（本例优化前后仅差 1.003x，在噪声范围内）。

## 验证信号

来源说明：问题与验证来自一次 softplus_v2_grad 带宽优化尝试及其判定。实测信号（L1 [4096,4096] fp32，同设备）：

- 优化前后 device-only 带宽成对对比（L1 [4096,4096] fp32，iters=30，同设备；参照路线=优化前两分支链，来源平台：950）：优化后/优化前 ≈1.003x，无实质变化（来源平台：950；证据等级：tool_observed）。
- 优化后全量 ST：`total=25  PASSED=25  FAILED=0  ERROR=0`，`RUN_EXIT=0`（来源平台：950；证据等级：tool_observed）。
- 瓶颈判定：计算版 kernel-only 为同 tiling 纯拷贝变体的 92.6%（纯拷贝仅快约 7.4%；参照路线=纯拷贝变体，来源平台：950），计算被搬运隐藏（来源平台：950；证据等级：tool_observed）。

## 性能影响

- 同算子内成对相对对比（来源平台：950）：单 Exp 链相对两分支链带宽约 1.003x（噪声级）；计算版相对纯拷贝上界约 0.926x。无绝对通用预期收益。

## 失效边界

- "无收益"仅对带宽受限成立；若后续把 tile/队列/核数改到搬运更快（纯拷贝上界抬升），计算链可能重新成为瓶颈，届时本卡结论失效需重判。
- 本例计算链为 1 Exp+1 Div 级别；更重的计算链（多级超越函数、fp32 长链）在同样 tiling 下可能本来就 V>MTE（计算受限），不适用本卡。
- 未验证，推断可迁移：fp16 路径（含 cast）同样被搬运隐藏，推断可迁移但未实测。

**未决项**：

- 无（瓶颈侧已由控制变量实验定论；剩余 7.4% 与纯拷贝的差距未逐项分解，对结论非必需）。
## 坏实践

- 不判瓶颈侧就投计算链精简——本例完整走完"重写→编译修错→精度回归→带宽复测" cycle 后才发现零收益。
- 把 ±0.3% 的测量波动报成优化收益。

## 误诊

- "带宽不达标是因为计算链太重（两分支 sigmoid 2 Exp+2 Div）"（agent_reported 假设，驱动了本次优化；被纯拷贝实验证伪——真正瓶颈是 MTE 搬运）。
