---
type: Runbook
title: "DataCopy count is an ELEMENT count on every arch; byte-count expressions carried from byte-typed upstream sour"
description: "DataCopy 的 count 参数在任何架构上都是元素个数而非字节数——按字节理解会少搬 4/8 倍"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-407"]
---

# DataCopy count is an ELEMENT count on every arch; byte-count expressions carried from byte-typed upstream sour

**OL-407** — **DataCopy `count` is an ELEMENT count on every arch; byte-count expressions carried from byte-typed upstream sources (int8/uint8) over-copy sizeof(T)× when the element type is widened to fp16/fp32 → silent buffer overflow, no error.** Drop the `sizeof` factor on port; symptom = large-shape fails, tiny shapes pass, structured adjacent-memory diffs. Element-count FACT is multi-SoC (351x+220x via OL-167, + dav-m200 here); the int8-port TRAP is dav-m200-anchored. Mirror of OL-167 (under-copy/truncation vs this over-copy). Anchor `代码片段 scatter_elements_v2/scatter_elements_v2_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`. Evidence: scatter_elements_v2 310p iter-1, 9/9 T1 after fix. backend=ascendc.

`applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=data-movement/port-from-byte-typed-source`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (scatter_elements_v2 310p 2026-08-08) — element-count semantics additionally confirmed on Ascend950PR (351x) + Ascend910 (220x) via OL-167, i.e. 3 SoC families`
`unverified_on: none for the element-count FACT (typed-tensor API contract, arch-uniform). The int8-byte-count PORT-TRAP is empirically anchored on dav-m200 only.`

### Principle

The plain `DataCopy(LocalTensor<T>, GlobalTensor<T>, count)` form takes `count` as an **element count**, not a byte count, for any typed `LocalTensor<T>` on every AscendC arch (dav-m200 / 220x / 351x). This is typed-tensor API contract, **not** a per-arch quirk — do not assume "newer archs differ" (a worker on the source op speculated dav-m200 is element-count while "220+ may differ"; both halves are wrong — it is element-count uniformly).

The over-copy bug enters at **port time**, not at API time. An upstream source authored for a **byte-typed** tensor (`int8_t` / `uint8_t`) frequently computes the DataCopy length as a **byte count** — which equals the element count *only* because `sizeof(int8)==1`. Widening the element type to fp16/fp32 while copying the byte-count expression verbatim passes `count*sizeof(T)` as the count, copying `sizeof(T)×` too many elements. Symptom: **silent** data corruption — the over-copy reads adjacent GM into UB and writes past the UB/GM tile bounds — with no error, no warning, no crash (until it happens to clobber a sentinel). For float32 this is a 4× over-copy.

### Evidence

- scatter_elements_v2 310p (2026-08-08, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): worker carried an upstream byte-count DataCopy expression into the float32 port; `count*sizeof(float)` over-copied 4× → silent data corruption. Fix = pass element count directly (`代码片段 scatter_elements_v2/scatter_elements_v2_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`). After fix: 9/9 T1 PASS, all `max_abs_diff_vs_cpu_fp64 = 0.0` (`scatter_elements_v2/verification.json`). Logged as a novel precision failure (failures_ledger iter 1). Cross-checked against the celu 310p peer op, which uses the same dav-m200 element-count convention.
- Multi-arch basis for the element-count FACT: OL-167 already exercises element-count `DataCopy(dst, src, count)` semantics on both Ascend950PR (351x) and Ascend910/220x (A3) — its `count*sizeof(T)%32==0` alignment constraint only makes sense if `count` is an element count. This entry adds dav-m200 as the 3rd SoC family.

## 内嵌代码片段

### scatter_elements_v2/scatter_elements_v2_kernel.h.snippet.h

```cpp
        if (alignedLen > TILE_ELEMS) alignedLen = TILE_ELEMS;

        // 1. Load var tile from GM (include_self: output starts as copy of var)
        // DataCopy takes ELEMENT COUNT (not bytes) for typed LocalTensor<float>
        DataCopy(varLocal, varGm[tileStart], alignedLen);
        WaitMTE2();

        // 2. Scan all indices/updates in batches
```

**孪生卡 ol-634 合并的独有内容**（ol-634-datacopy-count-is-an-element-count-on-every-arch.md 已删除）：

> // WRONG — byte-count expression carried from an int8_t upstream source into a float32 port:

> DataCopy(varLocal, varGm[tileStart], count * sizeof(T)); // copies 4× too many fp32 elements → silent overflow

> // CORRECT — count is an element count for typed LocalTensor<float>:

> DataCopy(varLocal, varGm[tileStart], alignedLen); // alignedLen is in ELEMENTS

> Anchor: `target/ascendc/examples/scatter_elements_v2/snippets.md` (shipped fix, comment "DataCopy takes ELEMENT COUNT (not bytes) for typed LocalTensor<float>").

> ### Detection rule (applies to ANY port from a byte-typed source)

> When porting an upstream kernel whose element type is `int8_t`/`uint8_t`/`char` (or any `sizeof(T)==1` type) to a wider type (fp16/fp32/int32):

> 1. Grep the upstream DataCopy calls for any `* sizeof(...)` factor on the count argument.

> 2. If present, it was a byte count that happened to equal the element count. **Drop the `sizeof` factor** for the wider-type port — pass the element count directly.

> 3. Symptom fingerprint if you forget: per-case precision fails on the *larger* shapes (over-copy corrupts more of the buffer) while tiny shapes may pass; `max_abs_diff` large and structured (adjacent-memory values bleeding in), not 1-ULP noise.

> - scatter_elements_v2 310p (2026-08-08, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): worker carried an upstream byte-count DataCopy expression into the float32 port; `count*sizeof(float)` over-copied 4× → silent data corruption. Fix = pass element count directly (`target/ascendc/examples/scatter_elements_v2/snippets.md`). After fix: 9/9 T1 PASS, all `max_abs_diff_vs_cpu_fp64 = 0.0` (`scatter_elements_v2/verification.json`). Logged as a novel precision failure (failures_ledger iter 1). Cross-checked against the celu 310p peer op, which uses the same dav-m200 element-count convention.

> - Any port from an int8/uint8/byte-typed CANN source (quantization kernels, char/byte-mask ops, byte-wise reductions) to fp16/fp32.

> - Ops where the upstream DataCopy length was authored as `byteLen` / `count * sizeof(uint8_t)` and the port widens the dtype.

> - Note this is the **mirror failure mode** of OL-167: OL-167 = under-copy (silent truncation when `count*sizeof(T)` is unaligned); this entry = over-copy (passing a byte count as an element count). Both are bugs in the same `count` parameter; check both when debugging a silent DataCopy-data-corruption signature.

> **Cross-ref**: [`OL-167`](ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md) (the alignment-truncation sibling of this `count` parameter — read both when triaging silent DataCopy corruption), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) / [`OL-130`](ol_130_v220_libascendcl_soname_conflict.md) (API-existence + lookup-chain: verify DataCopy semantics against a peer op on the target arch before trusting an upstream expression).

## 补充证据（2026-10-10 合并自 `datacopy_count_is_element_count_byte_expressions_overcopy.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-407 — OL-407: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=all; cann=9.0.0; bisheng=n/a; op_class=data-movement/port-from-byte-typed-source。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0 (scatter_elements_v2 310p 2026-08-08) — element-count semantics additionally confirmed on Ascend950PR (351x) + Ascend910 (220x) via OL-167, i.e. 3 SoC families。

**OL-407** — **DataCopy `count` is an ELEMENT count on every arch; byte-count expressions carried from byte-typed upstream sources (int8/uint8) over-copy sizeof(T)× when the element type is widened to fp16/fp32 → silent buffer overflow, no error.** Drop the `sizeof` factor on port; symptom = large-shape fails, tiny shapes pass, structured adjacent-memory diffs. Element-count FACT is multi-SoC (351x+220x via OL-167, + dav-m200 here); the int8-port TRAP is dav-m200-anchored. Mirror of OL-167 (under-copy/truncation vs this over-copy). Anchor 内部路径（不公开）workspace/scatter_elements_v2/kernel/scatter_elements_v2_kernel.h:114-115. Evidence: scatter_elements_v2 310p iter-1, 9/9 T1 after fix. backend=ascendc.

- Former A5Ops ID: `OL-407`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=data-movement/port-from-byte-typed-source`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (scatter_elements_v2 310p 2026-08-08) — element-count semantics additionally confirmed on Ascend950PR (351x) + Ascend910 (220x) via OL-167, i.e. 3 SoC families`
`unverified_on: none for the element-count FACT (typed-tensor API contract, arch-uniform). The int8-byte-count PORT-TRAP is empirically anchored on dav-m200 only.`

### Principle

The plain `DataCopy(LocalTensor<T>, GlobalTensor<T>, count)` form takes `count` as an **element count**, not a byte count, for any typed `LocalTensor<T>` on every AscendC arch (dav-m200 / 220x / 351x). This is typed-tensor API contract, **not** a per-arch quirk — do not assume "newer archs differ" (a worker on the source op speculated dav-m200 is element-count while "220+ may differ"; both halves are wrong — it is element-count uniformly).

The over-copy bug enters at **port time**, not at API time. An upstream source authored for a **byte-typed** tensor (`int8_t` / `uint8_t`) frequently computes the DataCopy length as a **byte count** — which equals the element count *only* because `sizeof(int8)==1`. Widening the element type to fp16/fp32 while copying the byte-count expression verbatim passes `count*sizeof(T)` as the count, copying `sizeof(T)×` too many elements. Symptom: **silent** data corruption — the over-copy reads adjacent GM into UB and writes past the UB/GM tile bounds — with no error, no warning, no crash (until it happens to clobber a sentinel). For float32 this is a 4× over-copy.

### Concrete anchor

```cpp
// WRONG — byte-count expression carried from an int8_t upstream source into a float32 port:
DataCopy(varLocal, varGm[tileStart], count * sizeof(T));   // copies 4× too many fp32 elements → silent overflow

// CORRECT — count is an element count for typed LocalTensor<float>:
DataCopy(varLocal, varGm[tileStart], alignedLen);          // alignedLen is in ELEMENTS
```
Anchor: 内部路径（不公开）workspace/scatter_elements_v2/kernel/scatter_elements_v2_kernel.h:114-115 (shipped fix, comment "DataCopy takes ELEMENT COUNT (not bytes) for typed LocalTensor<float>").

### Detection rule (applies to ANY port from a byte-typed source)

When porting an upstream kernel whose element type is `int8_t`/`uint8_t`/`char` (or any `sizeof(T)==1` type) to a wider type (fp16/fp32/int32):
1. Grep the upstream DataCopy calls for any `* sizeof(...)` factor on the count argument.
2. If present, it was a byte count that happened to equal the element count. **Drop the `sizeof` factor** for the wider-type port — pass the element count directly.
3. Symptom fingerprint if you forget: per-case precision fails on the *larger* shapes (over-copy corrupts more of the buffer) while tiny shapes may pass; `max_abs_diff` large and structured (adjacent-memory values bleeding in), not 1-ULP noise.

### Evidence

- scatter_elements_v2 310p (2026-08-08, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): worker carried an upstream byte-count DataCopy expression into the float32 port; `count*sizeof(float)` over-copied 4× → silent data corruption. Fix = pass element count directly (内部路径（不公开）workspace/scatter_elements_v2/kernel/scatter_elements_v2_kernel.h:115). After fix: 9/9 T1 PASS, all `max_abs_diff_vs_cpu_fp64 = 0.0` (`workspace/scatter_elements_v2/verification.json`). Logged as a novel precision failure (failures_ledger iter 1). Cross-checked against the celu 310p peer op, which uses the same dav-m200 element-count convention.
- Multi-arch basis for the element-count FACT: OL-167 already exercises element-count `DataCopy(dst, src, count)` semantics on both Ascend950PR (351x) and Ascend910/220x (A3) — its `count*sizeof(T)%32==0` alignment constraint only makes sense if `count` is an element count. This entry adds dav-m200 as the 3rd SoC family.

### Other instances (predicted)

- Any port from an int8/uint8/byte-typed CANN source (quantization kernels, char/byte-mask ops, byte-wise reductions) to fp16/fp32.
- Ops where the upstream DataCopy length was authored as `byteLen` / `count * sizeof(uint8_t)` and the port widens the dtype.
- Note this is the **mirror failure mode** of OL-167: OL-167 = under-copy (silent truncation when `count*sizeof(T)` is unaligned); this entry = over-copy (passing a byte count as an element count). Both are bugs in the same `count` parameter; check both when debugging a silent DataCopy-data-corruption signature.

**Cross-ref**: OL-167 (the alignment-truncation sibling of this `count` parameter — read both when triaging silent DataCopy corruption), OL-80 / OL-130 (API-existence + lookup-chain: verify DataCopy semantics against a peer op on the target arch before trusting an upstream expression).
