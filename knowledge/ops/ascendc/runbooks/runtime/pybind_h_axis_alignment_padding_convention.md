---
type: Runbook
title: "Pybind H-axis alignment padding convention (kernel DataCopy datablock alignment)"
description: "Pybind H-axis alignment padding convention (kernel DataCopy datablock alignment)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P66"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P66

## 适用场景

- 同号异题：optimizations/ 区既有 v1 卡 [p_p66_tquebind_vecin_vecout_1_for_in_place_ub_buffer_r.md](../../optimizations/p_p66_tquebind_vecin_vecout_1_for_in_place_ub_buffer_r.md) 与本卡同号 P-P66 但主题不同（TQueBind 原地 UB 复用）；引用 P-P66 时需按主题区分。
- 适用平台与范围：卡内无 applies_to 行；适用触发条件见正文模式索引行（trigger 列）。
- 验证来源：卡内无 verified_on/confirmed_on 行；本卡为 PATTERN_INDEX 模式行沉淀，未附验证记录。

| P-P66 | **Pybind H-axis alignment padding convention (kernel DataCopy datablock alignment)** | memory_access | Inside the kernel, `DataCopy(dst, src, count)` requires count aligned to the dtype datablock (fp32: 8, fp16/bf16: 16). If H may be non-aligned, do NOT add large DataCopyPad branches inside the kernel — instead, in the **pybind layer** use `torch::zeros((*, H_padded, *)).copy_(..., src)` to pre-pad input H to a multiple of `lcm(8,16)=16`. The kernel continues computing with H_padded; the output uses `out.narrow(dim, 0, H_orig)` to slice back to the original dim. The padded region is naturally zero and does not affect reduce/scatter semantics. **Applies to**: fused ops with multi-dtype inputs (mixed bf16+fp32) / scatter+reduce combinations / any multi-kernel pipeline that needs a uniform alignment assumption. First used in op#19 FusedResidualRmsNormBackward; inherited by op#17 EmbeddingWithInitialLayernormBackward (probe CONVENTION classification). **Extension — small per-channel affine params (group_norm_grad, 2026-06-03, port_a3_to_a5 V220)**: the same convention generalizes from the H-axis to tiny per-channel affine vectors. `weight[C]` with C=8 fp16 = 16 bytes is not 32B-aligned and V220 has no usable `DataCopyPad` UB→GM (EC-23); pad `weight` to `Caln=ceil(C/16)*16` via `torch::empty({Caln}).narrow(0,0,C).copy_(weight)`, and the multi-output grads `dweight`/`dbias` use a Caln-padded reduce buffer narrowed back in pybind. **Legitimacy distinction vs the OL-167 cheat**: this is NOT host-side data-path repair — the kernel reads only the valid channel indices via `GetValue` (padding lanes are never consumed) and the reduce writes correct values to the valid prefix, so the narrowed output is genuinely kernel-computed. EC-23 sanctions pybind padding on V220 (where DataCopyPad UB→GM crashes); on V351 the kernel MUST instead use DataCopyPad per P-P98 / OL-167. | **HIGH** |
