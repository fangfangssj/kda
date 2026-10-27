---
type: Runbook
title: "host-path-first perf on slow-host raw-launch kernels"
description: "aarch64 慢 host 上小 shape 逐元素算子的开销大头在 host 侧（pybind/alloc/launch 约 34µs vs kernel 5µs）——先分解再瘦身四板斧，写 kernel 第一版就该用上"
tags: [task.optimization, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P202"]
---

# host-path-first perf on slow-host raw-launch kernels

## 优化点

**P-P202** — **host-path-first perf on slow-host raw-launch kernels** (promoted from CAND-DAV-M200-HOST-PATH-SLIM, 2 independent ops): on aarch64-host SoCs the per-call cost is host-dominated (pybind ~9µs + at::empty ~8µs + aclrt ~7µs + macro arg-pack ~10µs vs kernel ~5µs) — decompose first, then slim: `__call__=forward` (−8µs), is_contiguous/empty (−2µs), RegisterAscendBinary-once + rtKernelLaunchWithHandle (+7.3%), empty_with_format direct (+20.0%). Guard rails: stub fallback for msprof ([`OL-163`](ol_163_cross_arch_perf_timing_methodology.md)), bit-exact re-grade as positive control (OL-321), no dispatcher staging ([`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md)). sin 310p 0.39→0.78 median; add 310p 0.661→1.198 median wall. Cross-ref [`OL-163`](ol_163_cross_arch_perf_timing_methodology.md)/[`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md)/OL-321/[`P-P201`](p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md).

(promoted from CAND-DAV-M200-HOST-PATH-SLIM, 2026-08-08 — 2 independent ops)

`applies_to: soc=Ascend310P3 (dav-m200, aarch64 host); cann=9.0.0; torch=2.9 (aarch64 build); bisheng=n/a; op_class=elementwise raw-launch (ACLRT_LAUNCH_KERNEL pybind)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (sin 310p 2026-08-06; add 310p 2026-08-08)`
`unverified_on: soc=Ascend950PR, Ascend910 (x86_64 host — host-round-trip cost differs; the slim techniques are torch/CANN-API-level so the direction should transfer, the numbers will not)`

**Principle**: on a slow-host SoC (dav-m200, aarch64 host), a raw-launch elementwise kernel's
per-call cost is **host-dominated** — measured decomposition at n=1024 fp32: pybind boundary
~9µs, `at::empty` ~8µs, aclrt launch ~7µs, `ACLRT_LAUNCH_KERNEL` macro arg-pack/checks ~10µs,
python wrapper ~2µs (post-slim), kernel exec ~5µs. Optimizing the KERNEL (the 5µs part) is
noise until the host path is slimmed. Decompose first (per-component timing), then apply the
cheap wins.

**Slim menu** (each item independently KEEP-verified on sin 310p; a 2nd op applied items 1–4
verbatim with measured deltas):
1. **`__call__ = forward` nn.Module hook-machinery bypass** — for hook-free leaf modules in
   perf-critical wrapper paths: −8µs/call (torch 2.9).
2. **`contiguous()` → `is_contiguous` TORCH_CHECK guard + `empty_like` → `empty`** — avoid
   unconditional copy/alloc fast-path misses: −2µs/call.
3. **Lean launch: `RegisterAscendBinary` ONCE (magic-static) + per-call
   `rtKernelLaunchWithHandle` with a stub-identical args struct** — hoists the generated
   stub's per-call `aclrtMalloc`/`aclrtFree`: sin Δ+7.3%.
4. **`at_npu::native::custom_ops::empty_with_format` direct, skipping the dispatcher** —
   the same native impl the dispatcher eventually reaches (NPUStorageImpl/format metadata
   preserved; plain `empty_generic` is REJECTED — breaks torch_npu op assumptions): sin Δ+20.0%.

**Guard rails**: (a) the slims must NOT reintroduce dispatcher staging that races the raw
launch (PB-106); (b) keep a fallback to the generated `ACLRT_LAUNCH_KERNEL` stub when
registration fails or the profiler is active — raw launches are invisible to
`operator_details.csv`, so profiler-based perf silently degrades to host-stopwatch asymmetry
(OL-163); (c) re-grade precision after any launch-mapping change — identical per-case MAREs
across macro/lean launch is the positive control that the key-0/args-layout mapping is
correct (OL-321).

**Concrete anchor**: sin 310p `optimization_log.md`（a5_ops 路径，未迁移）
(opt1 lean launch / opt2 empty_with_format); add 310p
`代码片段 add/pybind11.cpp.snippet.cpp（见文末）` (workspace scratch, gitignored).

**Evidence**:
- sin 310p (2026-08-06, first instance): ratio 0.39 → 0.7822 median across the menu sequence
  (independent re-measure A/B 0.758/0.772). Kernel-bound 1M-element case stays 0.271× — the
  host slims do NOT close a device-side algorithmic gap (handed to ko).
- add 310p (2026-08-08, promoting instance): items 1–4 reapplied verbatim with a
  stub-identical args struct; ours_wall 46→30µs on 1K-element cases, median wall ratio
  0.661 → 1.198 vs on-device aclnnAdd; bit-exact pass_a re-grade 39/39 with identical
  per-case MAREs (the OL-321 positive control).
- is_finite 310p (2026-08-09): the same launch-overhead levers converged on at ko —
  5 ko spawns, 0.68 → 0.8238× — the discover-at-ko-cap shape.
- is_inf 310p (2026-08-09): the levers applied AT KW-1 generation time (Opt1 launch-bypass
  + slim menu planned pre-spawn) — 1.2936×/1.2743× device-event across two independent full
  runs, above parity with ZERO ko spawns. Structural lesson: for dav-m200 raw-launch
  elementwise ops the host-path levers are now KNOWN — plan them in the kw-1 template, don't
  rediscover them at ko-cap. (Part of the is_inf vs is_finite delta may be vendor-kernel
  speed differences; the kw-1-application point stands independent of it.)
- pows 310p (2026-08-09): lean launch (item 3) applied verbatim at kw-1 with a
  stub-identical args layout (trailing overflow pointer) — ours_wall 37µs → 25.4µs on
  small cases, ratio 0.75 → 1.21 wall / 1.09× device-event vs on-device CANN torch.pow.
  Bit-exact pass_a re-grade (34/34, identical per-case MAREs across macro/lean launch)
  guards the key=0 mapping — the OL-321 positive control holds on a 3rd op. Host-side-only
  ~12µs/launch win, math untouched.
- histogram_v2 310p (2026-08-09): raw-launch + launch-bypass planned at kw-1 → 2.20×/2.28×
  device-event median vs on-device torch.histc (two back-to-back runs) — the vendor pays
  aclnn host prep per call, ours doesn't; our scalar-loop kernel is algorithmically the
  same shape as upstream's. Tiny-shape (numel ≤ 17) cases run 0.61–0.72× — the launch-bound
  floor regime: per-call workspace alloc + aclrtMemset overhead dominates; above the 0.6
  floor, hoistable via a persistent workspace if those shapes matter (ko lever, not a
  kernel bug).
- logical_not 310p (2026-08-09): Opt1 launch-bypass + slim menu applied at kw-1 as planned
  (the is_inf shape — levers known pre-spawn, zero ko spawns). Decomposition with the OL-321
  no-launch positive control: a NO-LAUNCH call (pybind + at::empty only) already measures
  102.5µs on this aarch64 host vs 129µs full call (launch slice ≈ 27µs) — ours is flat
  across numel 1→1M, vendor aclnnLogicalNot totals ~103µs; the 0.78× device-event / 0.96×
  same-host wall ratio is host-launch-bound, NOT kernel-bound (kernel exec is noise at all
  sizes), so the ratio is reported honestly against the structural ceiling rather than
  ko-churned. 34/34 bit-exact, det 34/34.
- lin_space_d 310p (2026-08-10): new menu item — never D2H-sync a scalar the host already
  has. A ValueDepend `num` tensor's `.item()` costs ~100µs/call on this aarch64 host;
  keeping `num` on CPU (kernel never reads it from GM — `totalLength` is a scalar launch
  arg) took the event window 165µs → 66–88µs. Audit raw-launch wrappers for hidden
  `.item()`/`.cpu()` syncs on shape/num scalars before blaming the launch slice.

**Other instances (predicted)**: any small-shape elementwise raw-launch op on an
aarch64-host SoC where the perf comparison is against a vendor integrated dispatcher; the
decomposition table itself (which host component costs what) transfers as the diagnostic
starting point even where individual menu items don't.

**Cross-ref**: [`OL-163`](ol_163_cross_arch_perf_timing_methodology.md) (device-event timing methodology; launch-bound-regime evidence), [`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md)
(raw-launch stream-ordering), OL-321 (positive-control re-grade), P-P201 (kernel-side
counterpart). backend=ascendc.

## 内嵌代码片段

### add/pybind11.cpp.snippet.cpp

```cpp
    bool ok = false;
    LeanLaunchState()
    {
        uint32_t rc = RegisterAscendBinary(
            reinterpret_cast<const char *>(__ascend_kernel_ascend310p3_kernels.aic_buf),
            __ascend_kernel_ascend310p3_kernels.aic_file_len, 2, &handle);
        if (rc != 0 || handle == nullptr) {
{
    // Inputs are always contiguous in the edge contract; fail loudly otherwise
    // (no hidden dispatcher copy — PB-60 race surface + pybind purity).
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous(), "run_add: inputs must be contiguous");
    TORCH_CHECK(x1.scalar_type() == at::kFloat && x2.scalar_type() == at::kFloat,
                "run_add: fp32 only on this verify path (310p GE kernel covers fp16/int32)");
    TORCH_CHECK(x1.dim() <= MAX_DIM && x2.dim() <= MAX_DIM, "run_add: rank > 8 unsupported");
    // call the exported torch_npu native allocator entry directly (sin 310p KO
    // precedent). acl_format=2 == ACL_FORMAT_ND. Same allocator, same storage
    // type — only the dispatch hops are removed. Guarded by bit-exact pass_a.
    at::Tensor y = at_npu::native::custom_ops::empty_with_format(
        outSizes, x1.scalar_type(), ::std::nullopt, x1.device(),
        ::std::nullopt, /*acl_format=*/2, ::std::nullopt);
    const int64_t numel = y.numel();
```

## 补充证据（2026-10-10 合并自 `host_path_first_perf_slow_host_raw_launch.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## P-P202 — P-P202: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200, aarch64 host); cann=9.0.0; torch=2.9 (aarch64 build); bisheng=n/a; op_class=elementwise raw-launch (ACLRT_LAUNCH_KERNEL pybind)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (sin 310p 2026-08-06; add 310p 2026-08-08)。

**P-P202** — **host-path-first perf on slow-host raw-launch kernels** (promoted from CAND-DAV-M200-HOST-PATH-SLIM, 2 independent ops): on aarch64-host SoCs the per-call cost is host-dominated (pybind ~9µs + at::empty ~8µs + aclrt ~7µs + macro arg-pack ~10µs vs kernel ~5µs) — decompose first, then slim: `__call__=forward` (−8µs), is_contiguous/empty (−2µs), RegisterAscendBinary-once + rtKernelLaunchWithHandle (+7.3%), empty_with_format direct (+20.0%). Guard rails: stub fallback for msprof (OL-163), bit-exact re-grade as positive control (OL-321), no dispatcher staging (PB-106). sin 310p 0.39→0.78 median; add 310p 0.661→1.198 median wall. Cross-ref [OL-163](ol_163_cross_arch_perf_timing_methodology.md)/[PB-106](../compilation/torch_npu_dispatcher_raw_launch_not_stream_ordered.md)/[OL-321](../validations/green_empty_zero_must_prove_scope_covers_claim.md)/[P-P201](../runtime/in_kernel_ragged_tail_datacopypad_less_socs.md).

- Former A5Ops ID: `P-P202`
- Fusion disposition: `additive_new`

(promoted from CAND-DAV-M200-HOST-PATH-SLIM, 2026-08-08 — 2 independent ops)

`applies_to: soc=Ascend310P3 (dav-m200, aarch64 host); cann=9.0.0; torch=2.9 (aarch64 build); bisheng=n/a; op_class=elementwise raw-launch (ACLRT_LAUNCH_KERNEL pybind)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (sin 310p 2026-08-06; add 310p 2026-08-08)`
`unverified_on: soc=Ascend950PR, Ascend910 (x86_64 host — host-round-trip cost differs; the slim techniques are torch/CANN-API-level so the direction should transfer, the numbers will not)`

**Principle**: on a slow-host SoC (dav-m200, aarch64 host), a raw-launch elementwise kernel's
per-call cost is **host-dominated** — measured decomposition at n=1024 fp32: pybind boundary
~9µs, `at::empty` ~8µs, aclrt launch ~7µs, `ACLRT_LAUNCH_KERNEL` macro arg-pack/checks ~10µs,
python wrapper ~2µs (post-slim), kernel exec ~5µs. Optimizing the KERNEL (the 5µs part) is
noise until the host path is slimmed. Decompose first (per-component timing), then apply the
cheap wins.

**Slim menu** (each item independently KEEP-verified on sin 310p; a 2nd op applied items 1–4
verbatim with measured deltas):
1. **`__call__ = forward` nn.Module hook-machinery bypass** — for hook-free leaf modules in
   perf-critical wrapper paths: −8µs/call (torch 2.9).
2. **`contiguous()` → `is_contiguous` TORCH_CHECK guard + `empty_like` → `empty`** — avoid
   unconditional copy/alloc fast-path misses: −2µs/call.
3. **Lean launch: `RegisterAscendBinary` ONCE (magic-static) + per-call
   `rtKernelLaunchWithHandle` with a stub-identical args struct** — hoists the generated
   stub's per-call `aclrtMalloc`/`aclrtFree`: sin Δ+7.3%.
4. **`at_npu::native::custom_ops::empty_with_format` direct, skipping the dispatcher** —
   the same native impl the dispatcher eventually reaches (NPUStorageImpl/format metadata
   preserved; plain `empty_generic` is REJECTED — breaks torch_npu op assumptions): sin Δ+20.0%.

**Guard rails**: (a) the slims must NOT reintroduce dispatcher staging that races the raw
launch (PB-106); (b) keep a fallback to the generated `ACLRT_LAUNCH_KERNEL` stub when
registration fails or the profiler is active — raw launches are invisible to
`operator_details.csv`, so profiler-based perf silently degrades to host-stopwatch asymmetry
(OL-163); (c) re-grade precision after any launch-mapping change — identical per-case MAREs
across macro/lean launch is the positive control that the key-0/args-layout mapping is
correct (OL-321).

**Concrete anchor**: sin 310p `output/npukernelbench-310p/src/kernels/sin/optimization_log.md`
(opt1 lean launch / opt2 empty_with_format); add 310p
内部路径（不公开）workspace/add/kernel/pybind11.cpp:66,160,171 (workspace scratch, gitignored).

**Evidence**:
- sin 310p (2026-08-06, first instance): ratio 0.39 → 0.7822 median across the menu sequence
  (independent re-measure A/B 0.758/0.772). Kernel-bound 1M-element case stays 0.271× — the
  host slims do NOT close a device-side algorithmic gap (handed to ko).
- add 310p (2026-08-08, promoting instance): items 1–4 reapplied verbatim with a
  stub-identical args struct; ours_wall 46→30µs on 1K-element cases, median wall ratio
  0.661 → 1.198 vs on-device aclnnAdd; bit-exact pass_a re-grade 39/39 with identical
  per-case MAREs (the OL-321 positive control).
- is_finite 310p (2026-08-09): the same launch-overhead levers converged on at ko —
  5 ko spawns, 0.68 → 0.8238× — the discover-at-ko-cap shape.
- is_inf 310p (2026-08-09): the levers applied AT KW-1 generation time (Opt1 launch-bypass
  + slim menu planned pre-spawn) — 1.2936×/1.2743× device-event across two independent full
  runs, above parity with ZERO ko spawns. Structural lesson: for dav-m200 raw-launch
  elementwise ops the host-path levers are now KNOWN — plan them in the kw-1 template, don't
  rediscover them at ko-cap. (Part of the is_inf vs is_finite delta may be vendor-kernel
  speed differences; the kw-1-application point stands independent of it.)
- pows 310p (2026-08-09): lean launch (item 3) applied verbatim at kw-1 with a
  stub-identical args layout (trailing overflow pointer) — ours_wall 37µs → 25.4µs on
  small cases, ratio 0.75 → 1.21 wall / 1.09× device-event vs on-device CANN torch.pow.
  Bit-exact pass_a re-grade (34/34, identical per-case MAREs across macro/lean launch)
  guards the key=0 mapping — the OL-321 positive control holds on a 3rd op. Host-side-only
  ~12µs/launch win, math untouched.
- histogram_v2 310p (2026-08-09): raw-launch + launch-bypass planned at kw-1 → 2.20×/2.28×
  device-event median vs on-device torch.histc (two back-to-back runs) — the vendor pays
  aclnn host prep per call, ours doesn't; our scalar-loop kernel is algorithmically the
  same shape as upstream's. Tiny-shape (numel ≤ 17) cases run 0.61–0.72× — the launch-bound
  floor regime: per-call workspace alloc + aclrtMemset overhead dominates; above the 0.6
  floor, hoistable via a persistent workspace if those shapes matter (ko lever, not a
  kernel bug).
- logical_not 310p (2026-08-09): Opt1 launch-bypass + slim menu applied at kw-1 as planned
  (the is_inf shape — levers known pre-spawn, zero ko spawns). Decomposition with the OL-321
  no-launch positive control: a NO-LAUNCH call (pybind + at::empty only) already measures
  102.5µs on this aarch64 host vs 129µs full call (launch slice ≈ 27µs) — ours is flat
  across numel 1→1M, vendor aclnnLogicalNot totals ~103µs; the 0.78× device-event / 0.96×
  same-host wall ratio is host-launch-bound, NOT kernel-bound (kernel exec is noise at all
  sizes), so the ratio is reported honestly against the structural ceiling rather than
  ko-churned. 34/34 bit-exact, det 34/34.
- lin_space_d 310p (2026-08-10): new menu item — never D2H-sync a scalar the host already
  has. A ValueDepend `num` tensor's `.item()` costs ~100µs/call on this aarch64 host;
  keeping `num` on CPU (kernel never reads it from GM — `totalLength` is a scalar launch
  arg) took the event window 165µs → 66–88µs. Audit raw-launch wrappers for hidden
  `.item()`/`.cpu()` syncs on shape/num scalars before blaming the launch slice.

**Other instances (predicted)**: any small-shape elementwise raw-launch op on an
aarch64-host SoC where the perf comparison is against a vendor integrated dispatcher; the
decomposition table itself (which host component costs what) transfers as the diagnostic
starting point even where individual menu items don't.

**Cross-ref**: OL-163 (device-event timing methodology; launch-bound-regime evidence), PB-106
(raw-launch stream-ordering), OL-321 (positive-control re-grade), P-P201 (kernel-side
counterpart). backend=ascendc.

---
