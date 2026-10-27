---
type: "Runbook"
title: "给已有算子新增 dtype：缺陷聚集在准入门与被静默烤进代码的不变量，不在算术里"
description: "为已有 FA 类算子增加一个低精度 dtype 时，缺陷不出现在新写的算术里，而出现在既有代码沿途的准入点：dtype 枚举门、lane 掩码、维度 extent、对齐假设。共同形状是「函数体已支持该 dtype，门不支持」，且门的报错信息常只提到另一个 dtype，说明该限制是从别的 dtype 继承来的。QFAG MXFP8 一次移植中同一形状出现三次：enablePreSfmg 门漏列 DT_FLOAT8_E4M3FN 而 DoPreSfmgTiling 早有 FP8 分支；对 [TS, MAX_D_SIZE] tile 做全 lane 规约，在 MXFP8 强制 D=64 时把 64 条未初始化 lane 计入；vf.interleave 丢弃第二个寄存器使 D=128 时 64..127 列未缩放。另有 GQA 门与 D 门同形。且多个缺陷互相掩盖：修好其一端到端指标不动，不能据此判定该修复无效。"
tags: [task.debugging, topic.precision, topic.compute.quantization]
status: draft
sources: []
created_at: "2026-09-23T19:45:39Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["Ascend950DT_9582", "DT_FLOAT8_E4M3FN"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。

# 新增 dtype：缺陷聚集在准入门，不在算术里

## 适用场景

- 工程形态：为已有算子增加一个此前不支持的低精度 dtype（FP8 族、HiFloat8、MXFP8 等），而非从零生成算子。
- API 路径：host 侧 tiling 的 dtype 判定分支；kernel 侧掩码构造（`vf.create_mask` / `vf.update_mask`）、寄存器交织（`vf.interleave`）、按 `MAX_*` 常量声明的 tile。
- 验证过的平台与 CANN 版本：Ascend950DT_9582 / CANN 9.2.0。
- 适用粒度：`operator_class`，FA 类反向已实证；同类「多 dtype 共享一套 tiling 与 kernel」的算子适用。
- 路线优先级：`recommended_fastpath`。
- 明确不适用或未验证场景：从零经 op-gen 生成的新算子（契约自洽，不存在继承的不变量）。

## 结论

新 dtype 打破的是既有代码里**没有写下来**的不变量。这些假设不在注释、文档或测试里，只存在于「一直以来都成立」。因此**先查准入点，不要先查新写的算术**。

## 三个同形实例

| 缺陷 | 位置 | 被打破的隐含不变量 |
|---|---|---|
| `enablePreSfmg` 门漏列 `DT_FLOAT8_E4M3FN`，而 `DoPreSfmgTiling` 内已有完整 FP8 分支 | `op_host/arch35/quant_flash_attention_score_grad_tiling_normal_regbase.cpp:502` | 「到达此函数的每个 dtype 都在上面的门里列出了」 |
| 对 `[TS, MAX_D_SIZE]` tile 做全 lane 规约 | `op_kernel/quant_flash_attn_grad.py:213` | 「D 恒等于 `MAX_D_SIZE`」，在 MXFP8 强制 D=64 后失效，把未初始化 lane 计入 |
| `vf.interleave` 丢弃第二个返回寄存器 | `op_kernel/quant_flash_attn_grad.py:2198` | 「D 可装进一个向量寄存器」，代码写于 D<=64 时期，D=128 时 64..127 列未被缩放 |

第三条有单变量对照：把 V 的指数强制为单位后，D=128 以 D=64 的精度通过（dq `22.41 -> 0.00023`），而非单位 V scale 只在 D=128 失败。

同一移植中 GQA 门与 D 门也是同形。GQA 门在
`op_host/arch35/quant_flash_attention_score_grad_tiling_common_regbase.cpp:278`，其报错原文是
*"When the dType of query is HIFLOAT8, n of query must be equal to n of keyIn"*，而它所在分支同时覆盖
MXFP8。**报错信息里出现的 dtype 与分支实际覆盖的 dtype 不一致，是继承限制的强信号。**

（GQA 门放开后算子可接受 `n1 != n2` 并输出正确形状，但数值仍然错误——放开门是必要而不充分的，
这一条不属于本卡结论，记在此仅为界定本卡不主张「放开门即支持 GQA」。）

## 互相掩盖，以及由此产生的误判

新算子是增量长出来的，任何时刻都有可用基线，一次只有一个缺陷。新增 dtype 会**一次点亮多条从未执行过的路径**，于是得到多个互相掩盖的缺陷，且**没有基线可二分**。

由此产生一个危险的误读：**修好其中一个，端到端指标不动，看起来像「这不是那个 bug」**，从而把正确的修复回退掉。

判据：在互相掩盖的场景下，端到端指标无改善**不是**该修复无效的证据。应为每个缺陷准备**单独的对照**（能证明该机制本身已正确的探针），而不是依赖端到端数值。

## 工件身份

- SoC `Ascend950DT_9582` / CANN 9.2.0；算子 `quant_flash_attn_grad`（MXFP8 反向）。
- 上表行号对应随证据发布的源码副本（车队内部存档，暂无可公开复核出处）。
- 三处修复后的封存 DSO：kernel `.o` `8168de3e3e0b749999dda1b14550a741e4bb3b5cec551960159622da12195ed0`，
  `liboptiling.so` `99a2f85939705b4ac0f75788c4b8d555859049eae4f3150548523d67f243821c`。
