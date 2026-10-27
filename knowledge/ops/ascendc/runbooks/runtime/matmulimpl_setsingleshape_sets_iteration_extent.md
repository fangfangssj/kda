---
type: Runbook
title: "MatmulImpl.SetSingleShape sets the block's iteration extent, not shape metadata — split axes take tile extents, unsplit axes take full extents"
description: "MatmulImpl.SetSingleShape sets the block's iteration extent, not shape metadata — split axes take tile extents, unsplit axes take full extents"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-481"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-481 — OL-481: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579/Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=any MatmulImpl (LAYER 1) kernel with multi-block M/N decomposition。
- 验证来源：1_BatchMatmul tile-split arm (2026-09-07, Ascend950DT_9582, CANN 9.1.0, matmul_client.h::SetTail source read + on-card discriminator probes)。

**OL-481** — `MatmulImpl.SetSingleShape` sets the block's ITERATION EXTENT, not shape metadata. On the LAYER-1 path (impl-side `MatmulImpl`, 内部路径（不公开）matmul_shape_info.h:120) it overwrites `MatmulShapeInfo.singleCoreM/N/K` — the field the scheduler's M/N loops (内部路径（不公开）scheduler_base.h:88 `MLoop->Init(singleCoreM)`) and the fixpipe copy-out extents both derive from. (The KFC-client class shows the same semantics as `SetTail` → `mIter=ceil(tailM/baseM)` consumed by `++cntIter_ >= mnIter_`, 内部路径（不公开）matmul_client.h:238-260 — same trap, different class.) On a kernel that splits one matrix across blocks, passing the FULL matrix shape makes every block iterate the whole MxN base-tile grid from its own offset base: cross-block stomping plus out-of-bounds writes past the C buffer (EC-41 class). Passing a base-CAPPED tile size on an UNSPLIT axis instead truncates shapes larger than the cap to one base tile (silent uncomputed rows/columns). Rule: TILE extents on split axes (edge-clamped), FULL extents on unsplit axes — or nothing, since `Init` already seeds the same field from `cubeTiling.singleCoreM/N`. Anchor: 1_BatchMatmul tile-split, random-A maxdiff ~1e2 on every nTiles>1 config while diagonal-A probes stayed clean; review-round-1 repro shapes b1/m256, b1/n1024, b16/n512.

- Former A5Ops ID: `OL-481`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_9579/Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=any MatmulImpl (LAYER 1) kernel with multi-block M/N decomposition`
`verified_on: 1_BatchMatmul tile-split arm (2026-09-07, Ascend950DT_9582, CANN 9.1.0, matmul_client.h::SetTail source read + on-card discriminator probes)`

**Principle**: AscendC "shape" setters may encode loop trip counts rather than describe tensors. Before calling any setter on a MatmulImpl instance, read its implementation (`adv_api/matmul/matmul_client.h`) — the `Init` defaults derived from `cubeTiling` are usually already correct, and an "obviously redundant" call can silently multiply every block's iteration space. Over-iterated blocks do not crash: they write structurally plausible values into other blocks' regions, and blocks whose row offset exceeds zero also write past the allocation.

**Concrete anchor**: 1_BatchMatmul tile-split (2026-09-07): `SetSingleShape(m, n, k)` on a (batch, mTile, nTile) decomposition gave `mIter=8, nIter=2..4` per block instead of 1x1. Symptom: with random inputs every nt=0 tile wrong except row 0, nt=1 tiles correct; with `A=[I|0]` and `A=[0|I]` probes everything "correct" — diagonal A masks over-iteration because over-iterated writes carry structurally correct values. Fix: `SetSingleShape(singleM, singleN, k)` with the edge-clamped per-block extents → all 7 split configurations bit-exact vs torch.bmm, canary clean.

**Evidence**: source read of impl-side `MatmulShapeInfo::SetSingleShape` (内部路径（不公开）matmul_shape_info.h:120) and the KFC-client `SetTail` (内部路径（不公开）matmul_client.h:238-260) + on-card random-A/diagonal-A discriminator probes + full 51-case bit-exact grade after the fix (fixed51_grade.tilev4.json in the 1_BatchMatmul delivery closure).

**Other instances (predicted)**: any MatmulImpl submatrix/split kernel (batched GEMM, grouped GEMM, MoE expert GEMM) that calls SetSingleShape/SetTail; the same read-the-impl rule applies to SetOrgShape (proven redundant where `tiling.M/N/Ka` already hold the full dims, and measured ~1% device-time cost on small cases).
