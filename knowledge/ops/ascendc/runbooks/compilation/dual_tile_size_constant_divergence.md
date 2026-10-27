---
type: Runbook
title: "Duplicated tile-size constants across host launch glue and device tiling header diverge silently"
description: "Duplicated tile-size constants across host launch glue and device tiling header diverge silently"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-DUAL-TILE-SIZE-CONSTANT-DIVERGENCE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-DUAL-TILE-SIZE-CONSTANT-DIVERGENCE — Duplicated tile-size constants across host launch glue and device tiling header diverge silently

## 适用场景

- 适用平台与范围：soc/cann 不限；任何 raw-launch pybind + host 侧 plan 计算与 device tiling header 配对的算子。
- 验证来源：soc=Ascend950DT_9582、CANN 9.1.0（13_Cat，2026-09-16）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**CAND-DUAL-TILE-SIZE-CONSTANT-DIVERGENCE** — when a raw-launch pybind TU hardcodes its own copy of a tile-size constant (e.g. `TILE_BYTES`) instead of including the device-side tiling header, the host plan (totalTiles, per-core split, k clamps) and the device plan diverge SILENTLY on any tile-size edit: the kernel is internally consistent, the launch geometry is not, and the resulting garbage output masquerades as a kernel bug. Any tile-size change must edit BOTH copies (or better, single-source the constant via a shared header) — treat duplicated plan constants across the host/device boundary as a divergence trap by construction.

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (any raw-launch pybind + host-side plan math pairing with a device tiling header)`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (13_Cat, 2026-09-16)`

- **Concrete anchor**: `pybind11.cpp` computes `totalTiles = ceil(numel / TILE_BYTES)` with its own literal `TILE_BYTES` while `cat_tiling.h` defines the value the kernel actually uses. Editing only the header leaves host launch geometry on the old tile count.
- **Symptom signature**: bit-mismatch / garbage after a tile-size-only edit, on a kernel whose device code is provably unchanged in logic — looks like a tiling-logic bug, is actually host/device plan skew.
- **Evidence**: 13_Cat ko-2 probe B-invalid (2026-09-16, 内部路径（不公开）workspace/13_Cat/ — workspace scratch, gitignored): smoke FAIL from editing the device header alone; fixed-by-consistency PASS after both copies were edited and MUST-match comments added.
- **Other instances (predicted)**: any op whose pybind layer recomputes tile counts, block counts, UB budgets, or alignment quanta that the device side also derives — split/copy engines, reduction tilings, multi-kernel pipelines with host-computed launch geometry.
- **Cross-ref**: harness-side analogue noted in 13_Cat failures_ledger (rsync exclude ordering) — same "two copies, one truth" class, different surface. backend=ascendc.
