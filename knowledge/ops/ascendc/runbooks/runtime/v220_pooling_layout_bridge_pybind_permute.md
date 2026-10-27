---
type: Runbook
title: "V220 pooling kernel layout bridge — channels-last (NDHWC) algorithm vs PyTorch channels-first (NCDHW) via pybind permute"
description: "V220 pooling kernel layout bridge — channels-last (NDHWC) algorithm vs PyTorch channels-first (NCDHW) via pybind permute"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-POOL-LAYOUT-BRIDGE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-POOL-LAYOUT-BRIDGE — V220 pooling kernel layout bridge — channels-last (NDHWC) algorithm vs PyTorch channels-first (NCDHW) via pybind permute

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=pooling（卡内 applies_to 行）。
- 验证来源：adaptive_avg_pool3d (V220→A5 L1 port, 2026-06-16)（卡内 verified_on 行）；未验证范围：other V220 pooling ops (adaptive_max_pool3d, avg_pool2d, max_pool2d); non-pooling V220 ops with channels-last native layout（卡内 unverified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=pooling`
`verified_on: adaptive_avg_pool3d (V220→A5 L1 port, 2026-06-16)`
`unverified_on: other V220 pooling ops (adaptive_max_pool3d, avg_pool2d, max_pool2d); non-pooling V220 ops with channels-last native layout`

**Trigger**: A V220 pooling algorithm (e.g., adaptive_avg_pool3d SplitC/SplitW/MultiW) assumes channels-last memory layout (D, H, W, C — NDHWC), but PyTorch's tensor convention is channels-first (C, D, H, W — NCDHW). A direct port of the V220 kernel reads memory in the wrong order, producing wrong outputs.

**Recommendation**: Do NOT rewrite the kernel algorithm to use channels-first. Instead, bridge the layout mismatch in the pybind layer:
1. `input_ncdhw.permute(0, 2, 3, 4, 1).contiguous()` → NDHWC before kernel launch
2. Launch the (unchanged) V220-algorithm kernel on NDHWC input
3. `output_ndhwc.permute(0, 4, 1, 2, 3).contiguous()` → NCDHW after kernel returns

**Evidence**: adaptive_avg_pool3d L1 V220→A5 port (2026-06-16): initial direct port produced wrong outputs on all 30 cases. Root cause identified as NCDHW→NDHWC layout mismatch. Fix in pybind (permute before kernel + after kernel) resolved all 30/30 cases to bit-exact vs CPU truth. The V220 kernel algorithm was preserved unchanged — only the pybind interface adapted.

**Hard do-not-apply**:
- Do NOT use this pattern when the V220 algorithm HAS a channels-first code path — prefer the native path over the pybind bridge.
- Do NOT permute inside the kernel (wastes UB on layout transform) — keep it in host/pybind where it is a one-time cost per launch.
- Do NOT apply blindly to non-pooling V220 ops — verify the algorithm's assumed layout first by reading the op_host/kernel source.

**Cross-reference**: OL-141 (target arch35 is advisory; generate from selected arch22 source rather
than wrapping target code); selected-source pre-stage policy.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-POOL-LAYOUT-BRIDGE，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
