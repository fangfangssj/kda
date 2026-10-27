---
type: Runbook
title: "GEIR attr element indexing must be diffed across ALL host entry points — infershape, tiling, and the aclnn wrapper can silently diverge on attr conventions, and a host-computed-shape wrapper never exercises infershape so the suite stays green"
description: "host 各入口点按各自约定读 attr 且互不相通：infershape 按 2 元素约定读 strides/pads，tiling 和 aclnn 按 GEIR 4 元素约定——同名字段两套语义。"
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-663"]
---

# GEIR attr element indexing must be diffed across ALL host entry points — infershape, tiling, and the aclnn wrapper can silently diverge on attr conventions, and a host-computed-shape wrapper never exercises infershape so the suite stays green

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=vendor ops with op_host infershape + tiling + aclnn wrapper (GEIR attr list convention)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (conv2d_backprop_input 310p ops-math flow, 2026-08-26 — found by Phase A read-through, fixed + regression-UT'd)`
`unverified_on: other SoCs — the GEIR attr convention is host-side CANN, expected SoC-independent, but only the 310P flow is measured`
- **Category**: vendor-op host chain (infershape / tiling / aclnn) consistency
- **Trigger**: an op whose aclnn wrapper computes the output shape itself (so the aclnn path never calls the op's infershape), AND whose attrs are list-typed (strides/pads/dilations) with more than one element-count convention in play.
- **Principle**: each host entry point reads attrs through its own convention, and nothing links them: the staged infershape read `strides[0..1]`/`pads[0..1]` (2-elem convention) while tiling + the aclnn wrapper used GEIR 4-elem (`strides[2..3]`, `pads[0]`/`pads[2]`). Because the aclnn path never exercises infershape, the divergence was a LATENT bug invisible to the full 270-case suite — infershape only runs on the GE graph path. Rule: for ops with a host-computed-shape aclnn wrapper, diff the attr element indexing of infershape vs tiling vs wrapper EXPLICITLY at author/review time, and add a regression UT that drives infershape directly.
- **Concrete anchor**: `conv2d_backprop_input_infershape.cpp` — strides/pads read as 2-elem; corrected to GEIR 4-elem with a dedicated infershape UT.
- **Evidence**: conv2d_backprop_input 310p (2026-08-26): latent bug found by Phase A read-through (NOT by any test), fixed + regression-UT'd; ophost UT 11/11 (`conv2d_backprop_input__opsmath/evidence/ut_result.json`). First instance.
- **Other instances (predicted)**: conv3d_backprop_input and any conv/pool-class op with GEIR 4-elem attr lists; any op where the aclnn wrapper bypasses its own infershape.
- **Cross-ref**: [`OL-662`](ol_662_encode_the_documented_aclnn_error_code_contract.md) (the UT-first methodology that would have caught this class), (the sibling "default-config batteries never exercise it" lesson). backend=ascendc.
