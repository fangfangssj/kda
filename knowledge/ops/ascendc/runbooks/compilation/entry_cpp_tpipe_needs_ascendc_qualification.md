---
type: Runbook
title: "220x-family entry-point `*_kernels.cpp`: bare `TPipe` does not resolve — qualify `AscendC::TPipe`"
description: "220x-family entry-point `*_kernels.cpp`: bare `TPipe` does not resolve — qualify `AscendC::TPipe`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p", "a3"]
aliases: ["AscendC::TPipe", "TPipe"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-198 — EC-198: A5Ops additive extension

> 关联：同一 EC-198 主题已由既有卡 [ec_198_220x_family_entry_point_kernels_cpp_bare_tpipe_d.md](ec_198_220x_family_entry_point_kernels_cpp_bare_tpipe_d.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）、Ascend910_220x；CANN 9.0.0；op_class=all（pybind / ACLRT_LAUNCH_KERNEL 验证路径入口）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（ge_glu_grad_v2 iter-1 build，2026-08-08；修复后 9/9 T1 PASS）。

**EC-198** — 220x-family entry-point `*_kernels.cpp`: bare `TPipe` does not resolve; qualify `AscendC::TPipe`. The entry `.cpp` is at global scope and does not inherit the kernel header's `using namespace AscendC`, so `TPipe pipe;` fails while it compiles inside the `XxxNs` namespace in the `.h`. C++ namespace-visibility gotcha, not an AscendC API issue. Do NOT add `using namespace AscendC` at global scope in the `.cpp`. Companion entry-form facts: scalar args (no `GET_TILING_DATA`; cf. OL-160/P140), no `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` on 220x/dav-m200 (PB-28), dav-m200 has 7 vector cores. Anchor: ge_glu_grad_v2 310p iter-1 build (内部路径（不公开）workspace/ge_glu_grad_v2/kernel/ge_glu_grad_v2_kernels.cpp:14,24), 9/9 T1 after fix. Cross-ref [OL-160](ol_160_safety_net_file_name_coupling_is_an_architectural.md), [PB-28](pb_28_kernel_task_type_default_kernel_type_aiv_only_macr.md), [OL-36](../precision/ol_36_npukernelbench_pytorch_wrapper_is_cann_delegation.md). backend=ascendc.

- Former A5Ops ID: `EC-198`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200), Ascend910_220x; cann=9.0.0; bisheng=n/a; op_class=all (pybind / ACLRT_LAUNCH_KERNEL verify-path entry)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (ge_glu_grad_v2 iter-1 build, 2026-08-08)`
`unverified_on: soc=Ascend950PR (A5 — the same two-file layout is used there too; the resolution rule is C++-namespace-level and SoC-agnostic, but the compile failure was observed on dav-m200)`

- **Category**: build / compile
- **Loaded by**: Builder (when authoring the entry-point `*_kernels.cpp`)
- **Symptom**: compile error — `TPipe` undeclared / "not declared in this scope" in the entry-point `*_kernels.cpp` that declares the `extern "C" __global__ __aicore__` kernel.
- **Mechanism**: the canonical multi-file layout puts the kernel class inside `namespace XxxNs { using namespace AscendC; ... }` in the header, so `TPipe` resolves unqualified *inside that namespace*. The entry `.cpp`, however, is at global scope (it only `#include`s the header and emits the extern-C entry) and does **not** inherit `using namespace AscendC`, so a bare `TPipe pipe;` fails to resolve. This is a C++ namespace-visibility gotcha, not an AscendC API issue.
- **Fix**: write `AscendC::TPipe pipe;` in the entry `.cpp`. Do **not** add `using namespace AscendC;` at global scope in the `.cpp` — keeping the global namespace clean is the point of the two-file split.
- **Correct pattern** (anchor, 内部路径（不公开）workspace/ge_glu_grad_v2/kernel/ge_glu_grad_v2_kernels.cpp:9-17):
  ```cpp
  extern "C" __global__ __aicore__ void my_kernel_fp16(
      GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
      int64_t totalElements, int64_t halfWidth, int32_t activateLeft)
  {
      XxxNs::XxxKernel<half> op;
      AscendC::TPipe pipe;          // qualified — entry .cpp is at global scope
      op.Init(dy, x, gelu, dx, totalElements, halfWidth, activateLeft, &pipe);
      op.Process();
  }
  ```
- **Companion entry-form facts (dav-m200 / 220x-family verify path)**: the entry takes **scalar args** (no `GET_TILING_DATA` / tiling struct — cf. OL-160 / P140 entry-point contract) and does **not** use `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` — that macro is rejected on 220x/dav-m200 (PB-28: `RegisterAscendBinary 107000`). dav-m200 exposes 7 vector cores (`kMaxBlockDim=7`, confirmed via the celu 310p peer-op precedent).
- **Evidence**: ge_glu_grad_v2 iter-1 build failure (2026-08-08, Ascend310P3, 内部路径（不公开）workspace/ge_glu_grad_v2/kernel/ge_glu_grad_v2_kernels.cpp:14,24) — bare `TPipe` failed to compile; qualifying as `AscendC::TPipe` resolved it; final binary 9/9 T1 PASS (6 fp16, 3 fp32).
- **Cross-ref**: OL-160 (verify-path canonical entry-point contract), PB-28 (AIV_ONLY macro rejected on 220x/dav-m200 — the sibling entry-form gotcha), OL-36 (pybind purity — this entry form is the legitimate non-delegating launch). backend=ascendc.
