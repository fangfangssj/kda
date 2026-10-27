---
type: Runbook
title: "Encode the documented aclnn error-code contract in UT FIRST and let the red UT drive wrapper structure — semantic param checks belong in `CheckParams` (161002/161001), not inside the graph builder (561000)"
description: "aclnn 错误码契约（161001/161002/561000…）就是 wrapper 的公开 API——检查放在哪一层决定调用方看到什么码；语义检查放错层会掩盖真因。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-662", "CheckParams"]
---

# Encode the documented aclnn error-code contract in UT FIRST and let the red UT drive wrapper structure — semantic param checks belong in `CheckParams` (161002/161001), not inside the graph builder (561000)

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=aclnn wrapper / op_api authoring (ops-math / ops-nn vendor-op flow)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (conv2d_backprop_input 310p ops-math flow, 2026-08-26 — opapi UT 14/14)`
`unverified_on: other CANN versions — the error-code VALUES are aclnn-contract-general, but only the 9.0.0 ops-math flow is measured`
- **Category**: vendor-op aclnn wrapper authoring / UT methodology
- **Trigger**: writing or staging an `aclnn<Op>` wrapper (`op_api/<op>.cpp`) with a documented error-code table in the API docs.
- **Principle**: the aclnn error-code contract (161001 NULLPTR, 161002 PARAM_INVALID, 561000 INNER ...) IS the wrapper's public API surface, and *where* a check lives determines which code the caller sees. Semantic checks (out-shape overflow, outputPadding range, groups divisibility) placed INSIDE the graph builder surface as 561000 (INNER), not the documented 161002 (PARAM_INVALID); a missing explicit nullptr intercept turns 161001 into 161002. The fix is structural: all semantic checks in `CheckParams`, plus an explicit nullptr intercept BEFORE it. Methodology that catches this on first run: transcribe the docs' error-code table into UT cases FIRST, watch them go red against the naive wrapper, and let the red UT drive the wrapper's check placement — instead of writing the wrapper and discovering the contract drift in review.
- **Concrete anchor (conv2d_backprop_input)**: staged `aclnn_convtranspose2d.cpp` computed out-shape/outputPadding/groups checks inside the graph builder → param errors surfaced 561000, nullptr surfaced 161002; moving checks into `CheckParams` + nullptr intercept → contract-correct codes, caught by the UT encoding the documented table.
- **Evidence**: conv2d_backprop_input 310p (2026-08-26, ops-math commercial flow): ophost UT 11/11 + opapi UT 14/14 (`conv2d_backprop_input__opsmath/evidence/ut_result.json`). First instance.
- **Other instances (predicted)**: every staged aclnn wrapper on the vendor-op flow (conv_transpose2d/3d sibling uses the same pattern); any op whose docs carry an error-code table.
- **Cross-ref**: (the 561000 family of misplaced-symptom failures), [`EC-60`](ec_60_aclrt_launch_kernel_with_blockdim_0_causes_acl_err.md) (param-invalid at a different layer). backend=ascendc.
