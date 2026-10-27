---
type: Optimization
title: "Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling"
description: "### Trigger Level-3 cube op (matmul / batch_matmul / gemm / Linear / Conv) where the gemm fits a single AIC (M/N ≤ ~512, fp32/fp16/bf16, ND inputs+outputs). When operands need logical transpose, layer"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.implementation.atomic, topic.data_movement.data_copy, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p68-single-aic-gemm-with-constexpr-static-tiling-on.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-27T16:55:00Z"
platforms: ["950"]
---

# Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling

> 原 OKF v1 卡号：**P-P68**。

## 优化点

### Trigger
Level-3 cube op (matmul / batch_matmul / gemm / Linear / Conv) where the gemm fits a single AIC (M/N ≤ ~512, fp32/fp16/bf16, ND inputs+outputs). When operands need logical transpose, layer P-P69 on top.

### Pattern

```cpp
template <typename T>
__aicore__ inline constexpr MatmulApiStaticTiling make_static_cfg() {
    MatmulApiStaticTiling t{};                      // every shape field defaults to -1
    t.cfg = CFG_NORM;
    t.usedCoreNum = 1;
    t.baseM = 128; t.baseN = 128;
    t.baseK = std::is_same_v<T, float> ? 64 : 128;  // L0A budget per dtype
    t.depthA1 = 8; t.depthB1 = 8;
    t.stepM = 1; t.stepN = 1; t.stepKa = 1; t.stepKb = 1;
    t.dbL0A = 1; t.dbL0B = 1; t.dbL0C = 1;
    t.iterateOrder = 0;
    t.isBias = 0; t.transLength = 0;
    return t;
}

template <typename T>
__aicore__ void op_one_impl(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                            int32_t M_, int32_t N_, int32_t K_) {
    static constexpr auto MM_CFG = make_static_cfg<T>();
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using BT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;

    GlobalTensor<T> aT, bT, cT;
    aT.SetGlobalBuffer((__gm__ T*)a, M_ * K_);
    bT.SetGlobalBuffer((__gm__ T*)b, K_ * N_);
    cT.SetGlobalBuffer((__gm__ T*)c, M_ * N_);

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubScratch;
    pipe.InitBuffer(ubScratch, 1024);              // static-check workaround

    TCubeTiling tiling{};                           // on-stack, runtime fields only
    tiling.M = M_; tiling.N = N_; tiling.Ka = K_; tiling.Kb = K_;
    tiling.singleCoreM = M_; tiling.singleCoreN = N_; tiling.singleCoreK = K_;

    MatmulImpl<AT, BT, CT, /*BIAS=*/CT, MM_CFG> mm;
    mm.Init(&tiling, &pipe);                        // non-__gm__ overload — no host H2D
    mm.SetTensorA(aT, /*isTransposeA=*/false);      // see P-P69 for trans variants
    mm.SetTensorB(bT, /*isTransposeB=*/false);
    mm.SetSingleShape(M_, N_, K_);
    mm.IterateAll<true>(cT, 0, false, false, false);
    mm.End();
}
```

### Performance unlock — why constexpr beats runtime tiling

`asc/impl/adv_api/detail/matmul/utils/matmul_utils.h::CopyTiling`:
```cpp
if constexpr (MM_CFG.<field> == -1) cubeTiling.<field> = gmCubeTiling-><field>;
```
**Each non-(-1) field eliminates one GM read in the hot path.** With ~25 shape-independent fields constexpr, only `M/N/Ka/Kb/singleCore*` remain runtime — and those fit on the stack, eliminating the ~5–10 µs `torch::empty(200B) + .copy_()` H2D too.

### Determinism (by-construction)
- blockDim ≤ batch_n with 1 AIC per output tile — no cross-core comm
- No `SetAtomicAdd` — fp32-accum mmad order fixed by constexpr tiling
- `IterateAll<sync=true>` + `End()` barrier per launch
- Satisfies `DET_POLICY=required` without algorithm contortion

### Performance trajectory (op#1 BatchMatmul)

| Variant | Ratio | asc median ms | kernel task_dur | scalar_ratio |
|---------|-------|---------------|-----------------|--------------|
| Opt0 (runtime tiling 64×64) | 0.515× | 0.033 | 4.34 µs | 0.92 |
| Opt1 (128×128 block, runtime tiling) | 0.543× | 0.035 | 3.23 µs | 0.64 |
| **Opt2 (constexpr + on-stack)** | **1.267×** | **0.015** | **2.67 µs** | **0.51** |

Validated on:
- 1_BatchMatmul (Opt2): 1.27× median, 51/51 + 14/14 PASS, fp32 bit-exact
- 4_MatmulTransA: 1.36× median, 50/50 + 16/16 PASS (P-P69 layered)
- 5_MatmulTransB: 1.29× median, 50/50 + 16/16 PASS (P-P69 layered)
- 3_MatmulBothTrans: 1.45× median, 50/50 + 16/16 PASS (P-P69 layered, both bools true)
  — **regime-bound, see the 2026-09-26 update: this headline was measured in the single-AIC /
  small-shape regime and does not transfer to a large-shape population.**

### Build invariants (CRITICAL — see EC-39 / EC-40)
- `MM_CFG` MUST be `MatmulApiStaticTiling` (NOT `CFG_NORM` directly) → EC-39
- Host-side POD mirroring `TCubeTiling` is 50 int32 = 200 B → EC-40
- `make_static_cfg<T>()` must be `__aicore__ inline constexpr`
- Local `constexpr auto X = factory<T>()` inside templated function must be `static constexpr`

### Static-check workaround
`ascendc_static_check.py kernel_has_computation` requires ≥3 of {TQue/TBuf, DataCopy, VEC_op, GlobalTensor, LocalTensor}. Cube kernels using `MatmulImpl::IterateAll` legitimately have only `GlobalTensor`. Use a 1-KB unused `TBuf<VECCALC>` scratch as workaround until the marker set is extended.

### L1 budget — the tiling above reserves up to 100 % of L1 by static footprint

The `depthA1 = depthB1 = 8` above is **not a safe default with slack**: its static footprint is
exactly the whole L1. Measured against the target's own platform config, not estimated:

- `l1_size = 524288` (512 KiB) — `platform_config/Ascend950DT_950x.ini`, CANN 9.1.0.
- L1 footprint per operand — note the two operands use **different** base dimensions, which
  happen to coincide numerically here only because `baseM == baseN == 128`:
  - A: `baseM × baseK × depthA1 × sizeof(T)`
  - B: `baseK × baseN × depthB1 × sizeof(T)`

| dtype | `baseK` | per operand | A + B | share of L1 |
|---|---|---|---|---|
| fp16 / bf16 | 128 | `128×128×8×2` = 262,144 B | **524,288 B** | **100.0 %** |
| fp32 | 64 | `128×64×8×4` = 262,144 B | **524,288 B** | **100.0 %** |

Both dtype branches reserve exactly `l1_size`. The configuration therefore **works only because
`isBias = 0` and nothing else in the kernel claims L1.** Enabling bias, raising `baseM`, deepening
`stepKa`/`stepKb`, or adding any other L1-resident buffer overflows it. Budget L1 explicitly before
changing any of those fields; do not treat depth 8 as headroom.

**RESOLVED 2026-09-27 by reading the allocation path — and the question was binary when the
code is not.** Neither branch above is universally true: the depth term enters the L1 allocation
**only for some `CubeInBuffer` variants**, and the variant is a compile-time selection.

Read at CANN 安装树内 asc/impl/adv_api/detail/matmul/ under CANN 9.1.0 (`aarch64-linux`). The call chain is
`CopyCubeIn::Init()` -> `CubeInBuffer->Init(CopyCubeInParams->GetBufferSize(), CopyCubeInParams->GetDepth())`
(`stage/copy_cube_in/base/copy_cube_in_base.h`), where `GetDepth()` returns `tiling.GetDepthA1()`
raw for A and `GetDepthB1()` for B (`stage/copy_cube_in/base/copy_cube_in_params_base.h`), and
`GetBufferSize()` returns `GetBaseSize()` (or `GetOrgSizeAlign()` on the UB->L1 single-shape path).

Closed enumeration of `resource/cube_in_buffer/` — 9 variants declare
`Init(int32_t baseBlockSize, int32_t cacheNum)`, and they split three ways on whether `cacheNum`
(the depth) sizes the allocation:

| variant | does depth size the buffer? | allocation |
|---|---|---|
| `cube_in_buffer_double_global_buffer.h` | **yes, linearly** | `InitBuffer(matrixByteSize * cacheNum)` |
| `cube_in_buffer_single_global_buffer.h` | **yes, linearly** | `InitBuffer(matrixByteSize * cacheNum)` |
| `cube_in_buffer_normal.h` | yes, via `cacheSize_` | `InitBuffer(qidCache_, SINGLE_QUE, cacheSize_ * matrixByteSize)` |
| `cube_in_buffer_n_buffer.h` | **no — `cacheNum` is accepted and unused in the allocation** | `InitBuffer(qid_, Ceil(singleCoreM/baseM), matrixByteSize * Ceil(singleCoreK/baseK))` |
| `cube_in_buffer_double_buffer_sparse.h` | only to pick queue depth | `cacheFactor_ = (cacheNum/stepSize - 1) & 1`; `InitBuffer(qid_, queDepth, (matrixByteSize*stepSize) >> 1)` |
| `cube_in_buffer_bmm_db.h` | only to pick queue depth | `InitBuffer(qid_, queDepth, matrixByteSize / queDepth)` |
| `cube_in_buffer_double_buffer.h` | only to pick queue depth | `InitBuffer(qid_, addr0)` |
| `cube_in_buffer_single_buffer.h` | no | `InitBuffer(qid_, SINGLE_QUE, matrixByteSize)` |
| `cube_in_buffer_intf.h` | — | interface only |

So:

- On the **`*_global_buffer`** variants the depth term is allocated **statically, independent of the
  actual `K`**. The 100 % table above is then exactly right, and a hard cap is warranted.
- On **`n_buffer`** the allocation is driven by `Ceil(singleCoreK / baseK)` and **the depth never
  enters the size at all**. The 100 % table does not model this variant: it is trimmed by the real
  `K`, which is why `depth = 8` can go unnoticed at `K <= 512`.
- On the double-buffer family the depth only selects single vs double queue, not the footprint.

**Still `UNVERIFIED`, and now it is one specific step rather than the whole path:** which variant
this card's own `MM_CFG` selects. That is a compile-time dispatch through `MatmulFeatureTrait<MM_CFG>`
and the module-selection headers, and until it is traced, the 100 % figure above should be read as
**the worst case across variants**, not as this configuration's measured footprint.

**Related, found in the same read:** `MatmulShapeTiling::DepthCheck()`
(`param/matmul_shape_tiling.h`) asserts `depthA1 % (stepM*stepKa) == 0` and
**`depthA1 / (stepM*stepKa) <= 2`** (and the `B`/`stepN`/`stepKb` mirror). It is compiled only for
`__NPU_ARCH__` in {2201, 3002, 3003, 3113, **3510**} — 3510 is this target — and guarded by
`if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG))`, so it does **not** apply to a
`CFG_NORM` configuration. For an MDL config with `stepM = stepKa = 1`, that assert caps `depthA1` at
**2**, which the `depthA1 = 8` above would violate; whether it fires in practice also depends on
`ASCENDC_ASSERT` being enabled in the build type.

### When NOT to apply
- batch > 1 with non-uniform shapes per batch — use IterateBatch (TODO)
- Very large M/N/P (> ~1024) — needs multi-AIC partitioning
- MX-FP8 / quantized matmul — has its own scale-tile path

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P68，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `platform_compat`；trigger: Level-3 cube ops (matmul / batch_matmul / gemm / Linear / Conv) where the gemm fits one AIC. `MatmulImpl<AT,BT,CT,BIAS,MM_CFG>` + `MM_CFG = constexpr MatmulApiStaticTiling` with ~25 shape-independent fields lifted to compile-time + on-stack `TCubeTiling tiling{}` filled at kernel entry + non-`__gm__` `Init(const TCubeTiling*, TPipe*)`. Eliminates ~25 GM tiling-reads per call AND the host-side `torch::empty(200B) + .copy_()` H2D (~5–10 µs). Determinism by-construction (single-AIC, no atomicAdd, IterateAll<sync=true>+End()). Validated on op#1 BatchMatmul (Opt0 0.515× → Opt2 1.27×). Carries cleanly to TransA/TransB/BothTrans. See OL-91 / EC-39 / EC-40 for the build invariants.；优先级: HIGH。

> 索引行原文：`| P-P68 | **Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling (Opt2 baseline)** | platform_compat | Level-3 cube ops (matmul / batch_matmul / gemm / Linear / Conv) where the gemm fits one AIC. `MatmulImpl<AT,BT,CT,BIAS,MM_CFG>` + `MM_CFG = constexpr MatmulApiStaticTiling` with ~25 shape-independent fields lifted to compile-time + on-stack `TCubeTiling tiling{}` filled at kernel entry + non-`__gm__` `Init(const TCubeTiling*, TPipe*)`. Eliminates ~25 GM tiling-reads per call AND the host-side `torch::empty(200B) + .copy_()` H2D (~5–10 µs). Determinism by-construction (single-AIC, no atomicAdd, IterateAll<sync=true>+End()). Validated on op#1 BatchMatmul (Opt0 0.515× → Opt2 1.27×). Carries cleanly to TransA/TransB/BothTrans. See OL-91 / EC-39 / EC-40 for the build invariants. | **HIGH** |`

## 更新 (2026-09-26) — perf headline scoped, L1 budget recorded

Revision from an independent `port_a3` re-generation of the same operator this card cites,
run `3a2afc29-151c-4b91-90df-5f3ce7313dc3`, op `3_MatmulBothTrans`, `Ascend950DT_9582` / CANN 9.1.0,
frozen 50-case denominator, `allow_output_reference = false` (the historical archive was closed for
the whole run, so this card was the only route by which the prior result reached the implementer).

**The pattern held on correctness.** Precision 50/50 on both tiers with 48/50 bit-exact, and
determinism 50/50 bit-identical with 0 drift — including the card's by-construction determinism
rationale, which survived an explicit 3-run check. The approach was adopted directly from this card
(constexpr `MatmulApiStaticTiling`, on-stack `TCubeTiling`, non-`__gm__` `Init`) and its EC-39 /
EC-40 build invariants were honoured.

**The `1.45×` headline did not transfer, and the card's own stated scope is why.** Over the 50-case
population the same approach measured **0.5264×** (arithmetic mean of per-case speedups; geometric
0.4888×, case-sum 0.4836×; all three agree in sign, `basis_failures = 0`, all 50 cases with valid
ratios). Partitioning those 50 cases by this card's own `M/N ≤ ~512` boundary:

- **inside** the scope: 3 cases — 1.2222×, 1.2941×, and 0.4559× — median 1.2222×;
- **outside** the scope: 47 cases, median **0.4391×**, worst **0.1676×** (fp16, shape
  `(10616832, 6)`).

Every case that beat 1.0× fell inside the stated scope, and the two best run at `gridDim = 1` where
launch overhead dominates. So **this card's boundary is corroborated by measurement**; what is not
established is the `1.45×` itself — `n = 3` inside the scope is too thin to confirm or refute it, and
`spread: UNMEASURED` for a single run. The 1.2222× is explicitly **not** claimed as a reproduction.

Structurally, the deficit sits where the card says it does not apply: fp32 held 0.8308× median while
fp16 / bf16 fell to 0.4260× / 0.4375×, and the loss grew monotonically with size
(`<1M` 0.6860 → `1–50M` 0.4541 → `≥50M` 0.4056). Reaching that population at all required supplying
the multi-AIC output-tile partition this card lists under *When NOT to apply*.

**How to read the headline from here:** treat a pattern's headline speedup as bound to the
population it was measured on. Before inheriting the number, check the pattern's stated scope
against your own denominator; if the mass of your cases is outside it, the number does not carry
even when the technique does.
