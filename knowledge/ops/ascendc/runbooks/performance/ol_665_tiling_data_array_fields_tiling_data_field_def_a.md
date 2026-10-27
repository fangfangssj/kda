---
type: Runbook
title: "Tiling-data array fields (`TILING_DATA_FIELD_DEF_ARR`) surface kernel-side as a plain C++ array read via `tiling->field[i]` — there is NO generated `get_field()`; `set_field(arr)` copies the FULL declared length, so zero-init the host array before filling a partial prefix"
description: "TILING_DATA_FIELD_DEF_ARR 数组字段在 kernel 侧生成的是数组成员直下标（不是 get/set 访问器）；host set 会拷满声明长度，部分填充前必须先清零"
tags: [task.optimization, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-665", "TILING_DATA_FIELD_DEF_ARR", get_field, set_field]
---

# Tiling-data array fields (`TILING_DATA_FIELD_DEF_ARR`) surface kernel-side as a plain C++ array read via `tiling->field[i]` — there is NO generated `get_field()`; `set_field(arr)` copies the FULL declared length, so zero-init the host array before filling a partial prefix

## 优化点

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a (opc codegen behavior — the getter-vs-array distinction is compiler-independent); op_class=all (any kernel carrying a host-precomputed array in tiling data)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (fp32_sinusoidal_embedding_1d 310p, 2026-08-27)`
- **Principle**: `TILING_DATA_FIELD_DEF_ARR(T, N, name)` in an `optiling` struct is codegen'd by the opc compiler into a kernel-side C++ array member `T name[N] = {}` — NOT the `get_name()`/`set_name()` scalar accessors that `TILING_DATA_FIELD_DEF` emits. The kernel therefore reads it as `tiling->name[i]` (direct array subscript), while the host tiling func writes it via the generated `tilingData.set_name(arr)` setter. Two consequences: (1) the scalar-field `get_name()` pattern does NOT exist for ARR fields — a kernel written as `tiling->get_freq()` will not compile; (2) `set_name(arr)` copies the FULL declared `N` entries (256 for a 256-slot array), so if only a prefix is meaningful the host array must be zero-initialized before the partial fill, or the kernel reads garbage in `[prefix, N)`.
- **Concrete anchor**:
  // host tiling func (op_host/<op>_tiling.cpp)
  float freqArr[256] = {0};                                  // zero-init: only [0..half) are valid
  for (uint64_t k = 0; k < half; ++k) freqArr[k] = static_cast<float>(std::pow(10000.0, -double(k)/half));
  tilingData.set_freq(freqArr);                              // copies all 256 entries
  // kernel (op_kernel/<op>.h)
  for (int64_t k = 0; k < half; ++k) freq.SetValue(k, tiling->freq[k]);   // plain array read, NOT get_freq()
  Anchor: `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md` (declaration) + `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md` (fill + `set_freq`), `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md` (kernel read). The codegen that emits the C++ array field is the opc compiler's 内部路径（不公开）get_op_tiling.py:445.
- **Evidence**: fp32_sinusoidal_embedding_1d 310p (2026-08-27): first instance — the kernel-side read as `tiling->freq[k]` (array) rather than a `get_freq()` accessor was the load-bearing detail for carrying the host-precomputed fp64→fp32 freq table into the kernel.
- **Other instances (predicted)**: any kernel carrying a host-precomputed table (freq/scale/bias/offset/lookup tables, per-dim constants) through `TILING_DATA_FIELD_DEF_ARR` — the getter-vs-array distinction and the full-length-copy zero-init rule apply to every ARR field regardless of dtype or content.
- **Cross-ref**: P-P128 (the technique this mechanism enables — host-precompute dim-only fp64 tables and carry them via the tiling array), OL-657 (the numerics rule that motivates precomputing on the host in double). backend=ascendc.

## 适用场景

- **工程形态**：TILING_DATA_FIELD_DEF_ARR 数组字段在 kernel 侧生成的是数组成员直下标（不是 get/set 访问器）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
