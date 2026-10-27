---
type: Runbook
title: "An offline numerics simulation validates only the arithmetic layer it re-implements — storage/transfer-layer kernel defects are structurally invisible to it, so a green offline proof is zero evidence about the engineering layer and the first device run is that layer's only witness"
description: "离线数值仿真（torch 模拟 VEC 序列）只证明算术层正确，覆盖不了 TBuf/对齐/队列生命周期等存储传输层缺陷——仿真全绿但设备挂时先怀疑存储传输层"
tags: [task.optimization, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-563"]
---

# An offline numerics simulation validates only the arithmetic layer it re-implements — storage/transfer-layer kernel defects are structurally invisible to it, so a green offline proof is zero evidence about the engineering layer and the first device run is that layer's only witness

## 优化点

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (any op-gen flow that pairs a torch-emulated numerics proof with an AscendC kernel); workflow=verification`
`verified_on: soc=Ascend310P3; cann=9.0.0 (23_HyenaFftSizePaddingRfft, 2026-09-05)`

**Principle**: an offline numerics proof (a torch emulation of the kernel's VEC op sequence — cast widths, add/mul order, twiddle arithmetic, compensation scheme) re-implements the ARITHMETIC semantics only. It does not model the C++ storage/transfer layer — TBuf backing of `LocalTensor` members, `DataCopy` count/address alignment, queue/buffer lifetimes, event fences. A defect in that layer therefore leaves the offline proof green while device output is corrupted. Consequences: (1) a green offline proof must never be cited as evidence about the storage/transfer layer — the layers are disjoint, and the proof transfers nothing between them; (2) the first device execution is the earliest witness for the engineering layer, so reading a green proof as "numerics done, kernel done" is a category error; (3) when a kernel fails on device while its offline proof is green, suspect the storage/transfer layer FIRST (member backing, copy alignment, fence ordering) and leave the numerics alone — do not re-derive formulas against a green proof.

**Concrete anchor**: 23_HyenaFftSizePaddingRfft 310p (2026-09-05): the torch-level proof of the DFT VEC sequence stayed green across BOTH engineering defects — an unbacked `LocalTensor` member used as Cast dst / Gather index source (EC-129: real channel 9.75e-2 flaky f32 / NaN f16 while imag sat at 1.5e-8) and unaligned MTE2 byte counts leaving stale tails (PB-127: 1.75e+2 at 400B rows). After the two engineering fixes (`23_HyenaFftSizePaddingRfft/kernel/hyenarfft_kernel.h:224 [snippet-unavailable: source workspace not on this box]` backing assignment, `:583-596` aligned-prefix + scalar tail), the UNTOUCHED numerics passed 49/49 PASS_T1 strict + 78/78 edge vs recomputed fp64 on the first correct execution (`23_HyenaFftSizePaddingRfft/verification.json`) — the arithmetic had never been the defect carrier.

- **Evidence**: 23_HyenaFftSizePaddingRfft 310p (2026-09-05, Ascend310P3/dav-m200): two independent engineering-layer defects, zero numerics defects; binary discrimination matrices over 9 S-values × 3 dtypes isolated each to the storage/transfer layer while the offline proof held green throughout (`23_HyenaFftSizePaddingRfft/knowledge_update.md` Finding 3).
- **Other instances (predicted)**: any verification flow pairing an analysis.md numerics derivation + torch op-sequence emulation with an AscendC kernel (OL-188 direct-DFT class, split-stream TwoSum/Kahan emulations); queue-fence defects (CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS class) and TBuf lifecycle defects (EC-62/PB-24 class) — the same argument puts them outside an arithmetic emulation's reach.
- **Cross-ref**: [`OL-565`](../precision/ol_565_a_prose_simulator_invariant_divergence_is_itself.md) (sibling blind spot — divergence BETWEEN prose and sim inside the numerics layer; this entry is the sim-vs-engineering-layer gap), [`EC-129`](../compilation/ec_129_an_unbacked_localtensor_class_member_used_as_a_v.md) / [`PB-127`](../compilation/pb_127_on_dav_m200_an_mte2_datacopy_gm_ub_whose_byte_co.md) (the two witness defects), [`PB-128`](../compilation/pb_128_torch_npu_fft_is_dead_on_ascend310p3_torch_fft_r.md) (why the cpu-truth lane was mandatory for this op), [`OL-89`](../precision/ol_89_when_the_active_reference_is_non_deterministic_loo.md) (decomposed CPU reference as truth), [`OL-214`](../precision/ol_214_single_core_first_testing_strategy_halve_the_searc.md) (single-core-first device isolation once device runs exist). backend=ascendc.
