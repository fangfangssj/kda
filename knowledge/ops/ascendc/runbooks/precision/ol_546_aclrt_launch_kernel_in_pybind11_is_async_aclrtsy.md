---
type: Runbook
title: "ACLRT_LAUNCH_KERNEL in pybind11 is ASYNC — aclrtSynchronizeStream(stream) before any local at::Tensor goes out"
description: "pybind 里 ACLRT_LAUNCH_KERNEL 是异步的：局部临时 tensor 返回前必须同步 stream，否则 kernel 读已释放内存；这是 host 生命周期 bug 不是 kernel bug"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-546"]
---

# ACLRT_LAUNCH_KERNEL in pybind11 is ASYNC — aclrtSynchronizeStream(stream) before any local at::Tensor goes out

**OL-546** — **`ACLRT_LAUNCH_KERNEL` in pybind11 is ASYNC — `aclrtSynchronizeStream(stream)` before any local `at::Tensor` goes out of scope, or the kernel reads freed GM (garbage/zeros, no error).** Host-side C++ lifetime bug (arch-independent), NOT a kernel bug; invisible to msprof, looks like intermittent/first-call precision fail. Decision rule: local/temporary input (`.contiguous()`/`.to()`/`.narrow()`/pad) → sync after launch before return; caller-held input → no sync needed for it, but NAME any staging tensor so it outlives the launch. Perf: the sync sets the launch-overhead floor on tiny kernels (~110µs dav-m200) — don't tune the kernel then. backend=ascendc. moe_gating_top_k_softmax 2026-08-11 (100/100 PASS). Cross-ref [`OL-179`](ol_179_tpipe_must_be_a_class_member_local_destructor_free.md) (kernel-side analog), [`OL-66`](../compilation/ol_66_torch_zeros_on_npu_is_not_stream_ordered_with_cust.md), [`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md).

`applies_to: soc=all (host-side C++ tensor-lifetime rule — arch-independent, NOT a hardware-empirical claim; applies to ANY SoC using the ACLRT_LAUNCH_KERNEL + pybind11 launch pattern); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — moe_gating_top_k_softmax, single op`
`unverified_on: the mechanism is host-side (async device launch vs C++ stack-frame lifetime) and is expected to hold on every raw-launch SoC (a3/220x, a5/arch35); empirically confirmed only on dav-m200 — re-confirm if cited elsewhere`

- **Category**: correctness
- **Loaded by**: Generator (pybind11.cpp author), QA
- **Trigger**: a pybind11 op function that builds a kernel input from a local / temporary `at::Tensor` (e.g. `.contiguous()`, `.to(dtype)`, `.narrow()`, a padded buffer) and returns immediately after `ACLRT_LAUNCH_KERNEL(...)` without an explicit stream synchronize.
- **Principle**: `ACLRT_LAUNCH_KERNEL` (and `aclrtLaunch*`) is **asynchronous** — the host call returns as soon as the launch is enqueued, BEFORE the device kernel executes. Any `at::Tensor` whose storage is owned by a local variable in the pybind function is freed when that variable goes out of scope at function exit (C++ destructor → device-memory release). If function exit races ahead of the async kernel's read of that tensor, the kernel reads freed/reused GM → garbage or all-zeros, with no error raised. This is a **host-side lifetime bug, not a kernel bug**; it is invisible to `msprof` and presents as an intermittent or first-call precision failure.
- **Mitigation (decision rule)** — pick by where the tensor's lifetime is managed:
  - **Local / temporary input** (built inside the pybind fn): call `aclrtSynchronizeStream(stream)` AFTER the launch and BEFORE the function returns, so the kernel finishes reading before the destructor runs. Minimal anchor:
    ```cpp
    at::Tensor keepAlive = input.contiguous().to(dtype);   // local temp
    ACLRT_LAUNCH_KERNEL(op)(blockDim, stream, keepAlive.data_ptr(), ...);
    aclrtSynchronizeStream(stream);   // block host until kernel completes — keepAlive's GM stays valid
    ```
  - **Caller-held input** (the caller's tensor outlives the call): no sync needed for THAT input — but if you create ANY local staging tensor (pad / contiguous / cast), either sync, or name it so it outlives the launch. Naming the staging tensor is the cheapest correctness fix; `aclrtSynchronizeStream` is the belt-and-suspenders fix that also covers determinism.
  - **Perf note**: the sync serializes host↔device and costs the host round-trip (~110µs on dav-m200 aarch64 — the same launch-overhead floor documented for the group_norm_grad / moe_init_routing_v2 single-launch counter-instances). For tiny kernels that floor dominates, so do NOT then "tune the kernel" — the floor is the sync, not the compute. Prefer caller-held, already-correctly-typed/contiguous inputs + no sync when throughput matters.
- **Evidence**: moe_gating_top_k_softmax (2026-08-11, Ascend310P3 / dav-m200, CANN 9.0.0) — the `finished` boolean mask was built as a local `at::Tensor finishedKeepAlive = finished->contiguous().to(at::kBool)` and passed to the async launch by raw pointer; without a sync the kernel read freed GM (precision failures-ledger `iter 1 | small/idx_mismatch | CAND-ASYNC-LAUNCH-TENSOR-LIFETIME`). Fixed by (a) naming the keep-alive tensor so it outlives the launch AND (b) `aclrtSynchronizeStream(stream)` after launch (`kernel/pybind11.cpp`). Final: Precision 100/100 PASS, Determinism 99/100.
- **Other instances (predicted)**: ANY pybind11 raw-launch op that pads / contiguous-izes / casts a kernel input into a local tensor and returns right after launch — the smaller the kernel, the more likely the race bites (kernel compute shorter than the scope-exit free). Also applies to local output-staging tensors the host reads back before any sync.
- **Cross-ref**: [`OL-179`](ol_179_tpipe_must_be_a_class_member_local_destructor_free.md) (kernel-side analog — a local `TPipe` destructor frees `TQue` buffers before async MTE3 writes complete; same lifetime-vs-async-write class, device-side), [`OL-66`](../compilation/ol_66_torch_zeros_on_npu_is_not_stream_ordered_with_cust.md) (`torch::zeros` fill is NOT stream-ordered vs a raw launch — sibling host/device ordering race), [`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (torch_npu dispatcher ops vs a raw launch are NOT stream-ordered — the cross-op sibling). backend=ascendc.
- OL-101 / OL-406 / OL-418 (orthogonal "which reference" axis — CPU-truth vs A3-capture vs fp64-disguised native; this entry is the "how to compare" axis they presuppose).
- OL-103 (the reciprocal situation — a *real* Tier-2 transcendental floor must not be confused with this *artefactual* representational floor; the dtype-cast fix must be applied BEFORE attributing a Pass-B failure to OL-103).

## 补充证据（2026-10-10 合并自 `aclrt_launch_kernel_async_sync_before_tensor_scope_end.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-546 — OL-546: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=all (host-side C++ tensor-lifetime rule — arch-independent, NOT a hardware-empirical claim; applies to ANY SoC using the ACLRT_LAUNCH_KERNEL + pybind11 launch pattern); cann=9.0.0; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0 — moe_gating_top_k_softmax, single op。

**OL-546** — **`ACLRT_LAUNCH_KERNEL` in pybind11 is ASYNC — `aclrtSynchronizeStream(stream)` before any local `at::Tensor` goes out of scope, or the kernel reads freed GM (garbage/zeros, no error).** Host-side C++ lifetime bug (arch-independent), NOT a kernel bug; invisible to msprof, looks like intermittent/first-call precision fail. Decision rule: local/temporary input (`.contiguous()`/`.to()`/`.narrow()`/pad) → sync after launch before return; caller-held input → no sync needed for it, but NAME any staging tensor so it outlives the launch. Perf: the sync sets the launch-overhead floor on tiny kernels (~110µs dav-m200) — don't tune the kernel then. backend=ascendc. moe_gating_top_k_softmax 2026-08-11 (100/100 PASS). Cross-ref [OL-179](ol_179_tpipe_must_be_a_class_member_local_destructor_free.md) (kernel-side analog), OL-66, PB-106.

- Former A5Ops ID: `OL-546`
- Fusion disposition: `additive_new`

`applies_to: soc=all (host-side C++ tensor-lifetime rule — arch-independent, NOT a hardware-empirical claim; applies to ANY SoC using the ACLRT_LAUNCH_KERNEL + pybind11 launch pattern); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — moe_gating_top_k_softmax, single op`
`unverified_on: the mechanism is host-side (async device launch vs C++ stack-frame lifetime) and is expected to hold on every raw-launch SoC (a3/220x, a5/arch35); empirically confirmed only on dav-m200 — re-confirm if cited elsewhere`

- **Category**: correctness
- **Loaded by**: Generator (pybind11.cpp author), QA
- **Trigger**: a pybind11 op function that builds a kernel input from a local / temporary `at::Tensor` (e.g. `.contiguous()`, `.to(dtype)`, `.narrow()`, a padded buffer) and returns immediately after `ACLRT_LAUNCH_KERNEL(...)` without an explicit stream synchronize.
- **Principle**: `ACLRT_LAUNCH_KERNEL` (and `aclrtLaunch*`) is **asynchronous** — the host call returns as soon as the launch is enqueued, BEFORE the device kernel executes. Any `at::Tensor` whose storage is owned by a local variable in the pybind function is freed when that variable goes out of scope at function exit (C++ destructor → device-memory release). If function exit races ahead of the async kernel's read of that tensor, the kernel reads freed/reused GM → garbage or all-zeros, with no error raised. This is a **host-side lifetime bug, not a kernel bug**; it is invisible to `msprof` and presents as an intermittent or first-call precision failure.
- **Mitigation (decision rule)** — pick by where the tensor's lifetime is managed:
  - **Local / temporary input** (built inside the pybind fn): call `aclrtSynchronizeStream(stream)` AFTER the launch and BEFORE the function returns, so the kernel finishes reading before the destructor runs. Minimal anchor:
    ```cpp
    at::Tensor keepAlive = input.contiguous().to(dtype);   // local temp
    ACLRT_LAUNCH_KERNEL(op)(blockDim, stream, keepAlive.data_ptr(), ...);
    aclrtSynchronizeStream(stream);   // block host until kernel completes — keepAlive's GM stays valid
    ```
  - **Caller-held input** (the caller's tensor outlives the call): no sync needed for THAT input — but if you create ANY local staging tensor (pad / contiguous / cast), either sync, or name it so it outlives the launch. Naming the staging tensor is the cheapest correctness fix; `aclrtSynchronizeStream` is the belt-and-suspenders fix that also covers determinism.
  - **Perf note**: the sync serializes host↔device and costs the host round-trip (~110µs on dav-m200 aarch64 — the same launch-overhead floor documented for the group_norm_grad / moe_init_routing_v2 single-launch counter-instances). For tiny kernels that floor dominates, so do NOT then "tune the kernel" — the floor is the sync, not the compute. Prefer caller-held, already-correctly-typed/contiguous inputs + no sync when throughput matters.
- **Evidence**: moe_gating_top_k_softmax (2026-08-11, Ascend310P3 / dav-m200, CANN 9.0.0) — the `finished` boolean mask was built as a local `at::Tensor finishedKeepAlive = finished->contiguous().to(at::kBool)` and passed to the async launch by raw pointer; without a sync the kernel read freed GM (precision failures-ledger `iter 1 | small/idx_mismatch | CAND-ASYNC-LAUNCH-TENSOR-LIFETIME`). Fixed by (a) naming the keep-alive tensor so it outlives the launch AND (b) `aclrtSynchronizeStream(stream)` after launch (`kernel/pybind11.cpp`). Final: Precision 100/100 PASS, Determinism 99/100.
- **Other instances (predicted)**: ANY pybind11 raw-launch op that pads / contiguous-izes / casts a kernel input into a local tensor and returns right after launch — the smaller the kernel, the more likely the race bites (kernel compute shorter than the scope-exit free). Also applies to local output-staging tensors the host reads back before any sync.
- **Cross-ref**: OL-179 (kernel-side analog — a local `TPipe` destructor frees `TQue` buffers before async MTE3 writes complete; same lifetime-vs-async-write class, device-side), OL-66 (`torch::zeros` fill is NOT stream-ordered vs a raw launch — sibling host/device ordering race), PB-106 (torch_npu dispatcher ops vs a raw launch are NOT stream-ordered — the cross-op sibling). backend=ascendc.
- OL-101 / OL-406 / OL-418 (orthogonal "which reference" axis — CPU-truth vs A3-capture vs fp64-disguised native; this entry is the "how to compare" axis they presuppose).
- OL-103 (the reciprocal situation — a *real* Tier-2 transcendental floor must not be confused with this *artefactual* representational floor; the dtype-cast fix must be applied BEFORE attributing a Pass-B failure to OL-103).
