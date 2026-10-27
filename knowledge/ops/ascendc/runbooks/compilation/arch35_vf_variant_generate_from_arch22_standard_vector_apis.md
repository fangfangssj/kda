---
type: Runbook
title: "when upstream arch35/ exists only as a VF-micro-API variant, generate from the arch22 algorithm with standard AscendC vector APIs — do NOT copy the VF micro-helpers"
description: "when upstream arch35/ exists only as a VF-micro-API variant, generate from the arch22 algorithm with standard AscendC vector APIs — do NOT copy the VF micro-helpers"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-ARCH35-VF-VARIANT-GENERATE-FROM-SOURCE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-ARCH35-VF-VARIANT-GENERATE-FROM-SOURCE — when upstream arch35/ exists only as a VF-micro-API variant, generate from the arch22 algorithm with standard AscendC vector APIs — do NOT copy the VF micro-helpers

## 适用场景

- 适用平台与范围：soc=Ascend950PR；cann=9.1.T500；op_class=port_a3_to_a5（recurrent / linear-attention / SSM 族）；scope=generation-strategy。
- 验证来源：recurrent_gated_delta_rule，A5 Ascend950PR_957b cann-9.1.T500，30/30 T1 PASS + 1.36× perf + deterministic（2026-06-18）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=port_a3_to_a5 (recurrent / linear-attention / SSM family); scope=generation-strategy`
`verified_on: recurrent_gated_delta_rule, a5 Ascend950PR_957b cann-9.1.T500, 30/30 T1 PASS + 1.36× perf + deterministic (2026-06-18)`

**Thesis** (reinforces the port_a3 default-OFF arch35 prestage rule): when CANN ships an `op_kernel/arch35/` reference for an op but that reference is ONLY a VF-micro-helper variant (e.g. `VecMulMatVF` / `OuterAddVF` — VF micro-API building blocks), do NOT copy those VF micro-helpers into the kernel TU. Copying `#include "arch35/..."` VF helpers is a V351-wrap, not a port (ARCH35_WRAP_CHEAT red line). Instead **generate** the A5 kernel from the arch22 algorithm source using standard AscendC vector APIs (`ReduceSum` / `Broadcast` / `MulAddDst` / `Muls` / `Exp` / `Cast`).

**Key empirical finding**: the standard-vector generation is correct, portable, self-contained AND already clears the perf floor (>0.6×) WITHOUT the VF rewrite. The VF micro-API variant is therefore an OPTIONAL L2 perf upgrade the optimizer may revisit later — NOT a correctness requirement and NOT a floor requirement. This removes the temptation to copy the VF variant "for perf".

**Concrete anchor**: recurrent_gated_delta_rule (Gated DeltaNet recurrent decode, qwen3-next family) — single AIV_ONLY vec-only kernel, L1 mechanical port. arch35/ had `VecMulMatVF`/`OuterAddVF`; kernel generated from the arch22 delta-rule algorithm (`state += beta·(v − state·k)⊗k`, `out = state·q`, per-head exp-decay gate) with standard vector APIs → 30/30 T1 PASS, 1.36× perf vs A3 baseline, deterministic.

**Anti-pattern avoided**: `#include "arch35/<op>.h"` of the VF-helper variant into the build TU (would be V351-wrap, not a port — and customer without upstream arch35 could not reproduce it).

**Promote when**: a second recurrent / linear-attention / SSM-family port whose arch35/ reference is a VF-only variant reproduces "standard-vector generation from arch22 clears the floor without the VF rewrite". Sibling: CAND-GDN-CHUNK-RECURRENT-COMPOSE (chunk variant via catlass — different decomposition, same don't-copy-upstream-template thesis).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-ARCH35-VF-VARIANT-GENERATE-FROM-SOURCE，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
