---
type: Runbook
title: "a kernel whose output is uint64 (packed quant params / hash / packed-index) is bridged through the pybind veri"
description: "输出是 uint64（打包量化参数/哈希）的 kernel 在 pybind 验证路径走 int64 桥接——PyTorch 没有 uint64，按位型比对而非数值比对"
tags: [task.optimization, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-413"]
---

# a kernel whose output is uint64 (packed quant params / hash / packed-index) is bridged through the pybind veri

## 优化点

**OL-413** — **a kernel whose output is `uint64` (packed quant params / hash / packed-index) is bridged through the pybind verify path as `int64`: PyTorch has no `uint64` dtype, so allocate `at::kLong` and let the kernel write its `uint64` bit pattern into the same memory (valid when packed-width ≤ 63 bits); CPU ref reinterprets `np.uint64`→`int64` for a bit-pattern (not magnitude) compare.** Anchor `代码片段 trans_quant_param_v2/pybind11.cpp.snippet.cpp（见文末）` (FP19+INT9 packer, 9/9 bit-exact). Device provenance UNVERIFIED (dav-m200 NOT_RUN flag). Cross-ref OL-136 (host-side uint64 packing), [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md), [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md). backend=ascendc.

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=bit-packing/quant-param/hash/packed-index`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0`
`unverified_on: soc=Ascend950PR, Ascend910 — the bridge is host-side / PyTorch-API (PyTorch lacks `uint64` on every backend) so it transfers; only the device kernel's `uint64` write is single-SoC-observed.`

**Principle**: PyTorch exposes no `torch.uint64` dtype — its unsigned ladder stops at `uint32`, and the next wider integer is `int64`. A kernel that computes a `uint64` result (packed quant params such as FP19+INT9 bitfields, a high marker bit + packed fields, 64-bit hashes, packed index/mask words) therefore cannot return it through a `torch.uint64` tensor. The bridge: allocate the output as `int64` — `at::empty({n}, at::dtype(at::kLong))` — and have the kernel's `GlobalTensor<uint64_t>` alias the SAME memory (`SetGlobalBuffer((__gm__ uint64_t*)y)`). The bit pattern is preserved verbatim: `int64` and `uint64` share one 64-bit representation, only the host-side *interpretation* differs. This is valid whenever the packed values fit in the `int64` non-negative range (`< 2^63`); bit-fields packed into the low bits always satisfy this (e.g. a 19-bit field + a 9-bit field + one fixed high bit occupy ≤ 46 bits ≪ 63). The CPU reference computes in `np.uint64` then reinterprets to `int64` for the comparison (`result.astype(np.int64)`), so the bit-exact check compares identical bit patterns, not signed magnitudes — there is no rounding and no tolerance.

**Concrete anchor** (pybind shim):
```cpp
// Output: uint64 stored as int64 (PyTorch has no uint64 dtype)
at::Tensor y = at::empty({outLength}, at::dtype(at::kLong).device(scaleC.device()));
ACLRT_LAUNCH_KERNEL(<op>)(1, stream, ..., y.data_ptr(), ...);
// inside the kernel:  yGm_.SetGlobalBuffer((__gm__ uint64_t*)y);   // aliases the int64 memory
```
CPU side: `return torch.from_numpy(result.astype(np.int64))` (numpy `uint64` → same bits as `int64`).

**Evidence**: trans_quant_param_v2 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — `[first evidence]`. The op packs an FP19 scale (QUANT_SCALE marker at bit 46) + an INT9 offset (bits 37–45) into one `uint64` per element; output allocated `at::kLong`, kernel writes `uint64_t`. 9/9 cases bit-exact (0 diffs, incl. a 1 048 576-element case) vs a CPU `np.uint64`→`int64` reference; determinism 9/9 bit-identical. Device provenance UNVERIFIED (`back/310p-hw-spec-pages` declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN`; `verification.json` `method` claims an Ascend310P3 run the branch hardware doc contests) — the bit-pattern-preservation fact is host/API-level and holds regardless; the device-side `uint64` write and the 9/9 PASS carry no confirmed device weight.

**Other instances (predicted)**: any kernel that emits `uint64` — quantization-parameter packers (FP→FP19/INT9, MX/FP8 scale-word pack), 64-bit hash/fingerprint kernels, packed-bitmask / packed-index emitters, RNG-state words. Each should reuse the `at::kLong` bridge; if a future field could push packed-width past bit 63, assert `packed_width ≤ 63` in the op source rather than silently relying on the positive-`int64` fit.

**Cross-ref**: OL-136 (TILING_KEY bit-encoding — a complementary HOST-side use of `uint64` packing, not an output-dtype question), [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md) (canonical entry-point contract the pybind shim satisfies), [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md) (dav-m200 `DataCopy` for the `uint64` UB→GM tail this op's kernel uses). backend=ascendc.

## 内嵌代码片段

### trans_quant_param_v2/pybind11.cpp.snippet.cpp

```cpp
    int32_t q_tile_start, int32_t kv_tile_start);

extern "C" uint32_t aclrtlaunch_fa_softmax_fp16(
    uint32_t blockDim, void* stream,
    void* scores_fp16, void* attn_fp16,
    int32_t total_tiles, int32_t S, float scale);

extern "C" uint32_t aclrtlaunch_fa_matmul2_cube_fp16(
    uint32_t blockDim, void* stream,
```
