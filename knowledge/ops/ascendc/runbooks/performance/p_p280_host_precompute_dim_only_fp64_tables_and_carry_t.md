---
type: Runbook
title: "Host-precompute dim-only fp64 tables and carry them via a tiling data array — never compute dim-only pow/exp tables on-device in fp32 when a Ratio-vs-fp32-competitor gate applies"
description: "参考实现里 fp64 精度的逐维常数表（如 sinusoidal 频率表）在设备上用 fp32 现算会差 1 ulp 并被大角度放大——解法是 host 侧 fp64 算好单次转 fp32 经 tiling 数组带进 kernel"
tags: [task.optimization, topic.implementation.tiling, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P280"]
---

# Host-precompute dim-only fp64 tables and carry them via a tiling data array — never compute dim-only pow/exp tables on-device in fp32 when a Ratio-vs-fp32-competitor gate applies

## 优化点

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=any op whose per-dim constants are a pure function of the dim (sinusoidal PE freq, RoPE inv-freq, scale/bias tables) and graded Ratio-vs-an-fp32-competitor`
- **Principle**: when a reference computes a per-dim constant table in a way that is fp64-accurate (torch's `pow(10000.0, -k/half)` — the wrapped-scalar float64→fp32 single-cast), any ON-DEVICE fp32 chain that re-derives the same table (`Exp(idx * -ln(10000)/half)`, `exp2`, or `exp(ln)` variants) is fp32-rounded and lands ~1 ulp off. For small angles that drift is invisible, but a normal-distribution / large-angle sweep amplifies the 1-ulp freq drift into ~1 extra ulp of angle → `cos`/`sin` error ≈ 2× the competitor, tripping a tight Ratio gate (2.1). The fix is NOT a more careful on-device chain — it is to compute the dim-only table ON THE HOST in fp64, round to fp32 ONCE, and carry it into the kernel as tiling data; the kernel then scalar-copies it into UB, so the angle bit-matches the competitor.
- **Concrete anchor**:
  // host tiling func — fp64 precompute, single fp32 round, carried via tiling array
  float freqArr[256] = {0};
  for (uint64_t k = 0; k < half; ++k)
      freqArr[k] = static_cast<float>(std::pow(10000.0, -static_cast<double>(k) / half));
  tilingData.set_freq(freqArr);            // TILING_DATA_FIELD_DEF_ARR(float, 256, freq)
  // kernel — scalar-copy into UB once at Init, then VEC Muls/Cos/Sin on the bit-matched angle
  freq.SetValue(k, tiling->freq[k]);
  Anchor: `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md` + `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md`.
- **Evidence**: fp32_sinusoidal_embedding_1d 310p (2026-08-27): 8 normal-distribution sweep cases failed the 2.1 Ratio (2.1..6.3 > 2.0) despite max abs error ≈ the competitor's — bit-diff 66% of elements off by 1 ulp from the on-device `Exp` chain. Host fp64 precompute + tiling-array carry → angle bit-matches the competitor, Ratio ≈ 1 (`verification.json` cases_v21 228/228 PASS).
- **Other instances (predicted)**: any op whose per-dim constants are a closed-form function of the dim — RoPE `inv_freq = 1/(base^(2k/d))`, sinusoidal PE, ALiBi slopes, per-channel scale/bias tables — whenever the grading is Ratio-vs-an-fp32-competitor rather than a loose abs tolerance. If the table is data-dependent (not dim-only), this pattern does not apply; see OL-590 for the host-derivation rule on scalars.
- **Cross-ref**: OL-598 (the `TILING_DATA_FIELD_DEF_ARR` access mechanism the carry depends on), OL-590 (the scalar-coefficient sibling — derive host scalars in double, cast once), PB-145 (the sibling VEC-trig defect on the same op). backend=ascendc.
