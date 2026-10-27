---
type: Runbook
title: "An inter-stage dtype quantize between two raw kernel launches belongs IN-KERNEL — host-staged `.to(at::kCPU)` materialization is a P149-rejected CPU-offload shape and independently pays a D2H+cast+H2D+double-sync round-trip"
description: "两级 raw launch 之间的 dtype 量化必须在 kernel 内做——host 侧 .to(CPU) 中转既是 P149 拒绝的 CPU-offload 形态，又白付 D2H+cast+H2D。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-564"]
---

# An inter-stage dtype quantize between two raw kernel launches belongs IN-KERNEL — host-staged `.to(at::kCPU)` materialization is a P149-rejected CPU-offload shape and independently pays a D2H+cast+H2D+double-sync round-trip

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=any multi-launch raw-ACLRT pipeline whose reference chain quantizes per stage (bf16/fp16 per-op rounding); the P149-compliance half is harness-wide (soc=all), the transport-reliability half is torch_npu-build-specific`
`verified_on: soc=Ascend310P3; cann=9.0.0 (24_KvCacheUpdateWithRopeBackward kw-1→kw-2, 2026-09-06 — gate rejection + in-kernel rewrite, DEVICE-RUN)`
`unverified_on: soc=Ascend950PR, Ascend910 — the P149 rejection transfers (harness gate), but the perf delta and the cast↔launch unreliability were measured on one torch_npu build`
**Principle**: when a fused pipeline's reference rounds to a low-precision dtype BETWEEN stages (per-op bf16 chains — OL-102), the kernel side must reproduce that quantize, and "where does the conversion RUN?" has exactly one compliant answer: inside the kernel that produces the intermediate. A host-side round-trip of the intermediate (`.to(at::kCPU)` → aten cast → back to device) fails on two independent axes: (1) the P149 pybind-host-logic gate (`_check_pybind_host_logic`, `finalize_checks_structural.py:684`（a5_ops 路径，未迁移）) classifies it as CPU offload even when the payload is pure dtype bits — "it's just memory movement + the exact aten cast" is not an exemption; (2) it converts an on-device producer→consumer edge into D2H+cast+H2D with two sync points — measured here as the op's worst-case perf limiter.
**Transport decision rule (applies to ANY raw-launch pipeline on this torch_npu build)**: launch→launch on the SAME stream is reliable; a cast↔launch boundary is unreliable in BOTH directions (dispatcher-cast→launch AND launch→dispatcher-cast each measured reading zeros despite post-sync host views showing nonzero content — PB-106 evidence). So keep every inter-stage edge launch→launch, and put any dtype change on the PRODUCING kernel's write side.
**Concrete anchor**: kw-1 shape (REJECTED): pybind materializes the fp32 intermediate `.to(at::kCPU)`, casts to bf16, returns it to device, launches stage 2. kw-2 shape (SHIPPED): the stage-1 kernel emits bf16 directly via an in-kernel RNE int16-lane chain (`24_KvCacheUpdateWithRopeBackward/kernel/kvropebwd_kernel.h:254-268 [snippet-unavailable: source workspace not on this box]`); pybind is metadata+alloc+device-side input widen only (`代码片段 24_KvCacheUpdateWithRopeBackward/pybind11.cpp.snippet.cpp（见文末）`).
**Evidence**: 24_KvCacheUpdateWithRopeBackward kw-2 310p (2026-09-06, Ascend310P3/dav-m200, benchmark): P149 rejected the host-staged shape; the in-kernel rewrite was bit-exact (Pass A 50/50 strict vs the CANN-NPU reference incl. all bf16 cases) AND raised mean symmetric ratio 2.9482×→3.4098× (+15.6%) — the round-trip-bound worst case (27, bf16, 1.2869×) left the bottom-5 entirely. No `bfloat16_t` on the arch → the RNE itself is CAND-INKERNEL-BF16-RNE; on archs WITH `bfloat16_t`, a native Cast(CAST_RINT) in the same position is the compliant spelling.
**Other instances (predicted)**: any multi-launch pipeline with per-op low-precision reference chains (kv-cache/RoPE fwd+bwd, bf16 checkpoint boundaries, quantize-aware-training grads); fp16 per-op chains (same rule, cheaper cast); NOT applicable where the reference keeps inter-stage fp32 (OL-303 — rounding once at the epilogue is the correct design there and no inter-stage quantize exists).
**Cross-ref**: [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (Fix-3 supersession + the both-direction cast↔launch evidence), [`OL-167`](ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md) (the P149 retraction catalog this instance joins), [`OL-102`](../precision/ol_102_fp16_bf16_per_op_rounding_chain.md) (the per-op rounding-chain demand that creates the boundary), OL-303 (the complementary no-inter-stage-quantize case), CAND-INKERNEL-BF16-RNE (the arch-without-bfloat16_t implementation). backend=ascendc.

## 内嵌代码片段

### 24_KvCacheUpdateWithRopeBackward/pybind11.cpp.snippet.cpp

```cpp
    TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape mismatch");
    TORCH_CHECK(cache_position.dim() == 1, "cache_position must be 1D");
    TORCH_CHECK(cache_position.dtype() == torch::kInt64, "cache_position must be int64");

    auto gkc_c = grad_key_cache.contiguous();
    auto gvc_c = grad_value_cache.contiguous();
    auto ks_c  = key_states.contiguous();
    auto cs_c  = cos.contiguous();
    auto si_c  = sin.contiguous();
    auto cp_c  = cache_position.contiguous();
```
