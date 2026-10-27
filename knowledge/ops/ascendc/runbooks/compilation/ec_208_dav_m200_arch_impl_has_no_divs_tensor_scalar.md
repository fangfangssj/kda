---
type: Runbook
title: "dav-m200 arch impl has NO Divs (tensor÷scalar)"
description: "dav-m200 没有 `Divs`（张量÷标量）；位忠实路径用 Duplicate 常数 + 张量 `Div`，bit-exact 路径别用倒数+`Muls`。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-208"]
---

# dav-m200 arch impl has NO Divs (tensor÷scalar)

**EC-208** — **dav-m200 arch impl has NO `Divs` (tensor÷scalar)** (build error; Muls/Adds present in the same impl header): substitute `Duplicate(const)` + tensor `Div` — TRUE IEEE division, bit-faithful to torch `/ scalar`; do NOT reach for reciprocal+`Muls` on bit-exact paths (OL-256 is the perf-tolerant option on Divs-bearing arches — different scope). Detection: grep `Divs(` before first dav-m200 build. 17_AdamW 310p 2026-08-15 (6 sites, fp32 bit-exact); prior art: celu 310p (reciprocal, concedes 0–1 ULP), op#17 AdamW 220x 2026-05-02. Cross-ref [`OL-127`](ol_127_cann_api_surface_shrinks_across_versions_check_api.md), OL-256, [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all`
`unverified_on: soc=Ascend950PR (a5 — Divs exists there, OL-80 catalog); soc=Ascend910_220x (Divs also absent per OL-127's checklist — different workaround era)`

- **Error**: build-time — `Divs` has zero instantiations in the dav-m200 arch impl (CANN 安装树内 asc/impl/basic_api/dav_m200/kernel_operator_vec_binary_scalar_impl.h; `Muls`/`Adds` present, `Divs` absent), so any `Divs(dst, src, scalar, n)` fails to compile.
- **Fix**: broadcast the divisor into a UB tensor once, then tensor `Div`:
  ```cpp
  Duplicate(b2, dM1_, n);   // divisor constant → UB tensor
  Div(t1, t3, b2, n);       // TRUE IEEE division, bit-faithful to torch `/ scalar`
  ```
  This preserves per-element IEEE division — bit-faithful to torch's tensor÷scalar — and is the preferred substitute whenever bit-exactness is the target (ALWAYS_LOADED iron law §5: no reciprocal strength-reduction). Anchor: `代码片段 17_AdamW/adamw_kernel.h.snippet.h（见文末）` (6 sites).
- **Detection**: grep the kernel for `Divs(` before first build on a dav-m200 target; every hit is this EC.
- **Evidence**: 17_AdamW 310p build iter-1 (2026-08-15): 6× Divs → Duplicate+Div; fp32 path bit-exact vs the CPU chain (`17_AdamW/failures_ledger.md:9`). Prior art, same arch-gap class: celu 310p (reciprocal workaround — concedes 0–1 ULP; superseded by this fix pattern where bit-exactness matters); op#17 AdamW 220x port 2026-05-02 (OL-127's Divs row — reciprocal-in-pybind+`Muls` era).
- **Tension note (deliberate, not a conflict)**: OL-256 recommends `Divs→Muls(1/scalar)` as a PRECISION-TOLERANT fp32 perf optimization on arches where Divs exists. That rule targets throughput under tolerance; THIS EC targets the Divs-absent BUILD ERROR on dav-m200 with bit-exactness preferred. Do not cite OL-256 to justify reciprocal-multiply on a bit-exact dav-m200 path.
- **Other instances (predicted)**: any dav-m200 kernel with tensor÷scalar sites — optimizer bias-correction divisions, normalization eps-divisions, LR-scaled updates.
- Cross-ref [`OL-127`](ol_127_cann_api_surface_shrinks_across_versions_check_api.md) (API surface shrinks across versions — the 220x Divs row this extends to arch-200x with a better workaround), OL-256 (reciprocal perf option, different scope), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (API-catalog lookup discipline), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) (dav-m200 signature-surface gap class). backend=ascendc.

## 内嵌代码片段

### 17_AdamW/adamw_kernel.h.snippet.h

```cpp
        int32_t use_nesterov,
        int64_t total_elems,
        int64_t blk_idx, int64_t blk_cnt)
    {
        beta1_ = beta1;
        beta2_ = beta2;
        epsilon_ = epsilon;
        use_nesterov_ = use_nesterov;
        one_minus_beta1_ = one_minus_beta1;
        one_minus_beta2_ = one_minus_beta2;
        lr_corrected_ = lr_corrected;
        inv_one_minus_b2p_ = inv_one_minus_b2p;

        constexpr int32_t ALIGN = 16;
        int64_t elems_per_block = (total_elems + blk_cnt - 1) / blk_cnt;
        elems_per_block = ((elems_per_block + ALIGN - 1) / ALIGN) * ALIGN;

        int64_t start = blk_idx * elems_per_block;
        int64_t end = start + elems_per_block;
        if (end > total_elems) end = total_elems;
        if (start >= total_elems) { my_elems_ = 0; return; }
        my_elems_ = static_cast<int32_t>(end - start);

        gmVar_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(var_gm) + start, my_elems_);
        gmM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m_gm) + start, my_elems_);
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v_gm) + start, my_elems_);
        gmGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(grad_gm) + start, my_elems_);

        pipe_.InitBuffer(varHInQ_, 1, ADAM_TILE * sizeof(half));
        pipe_.InitBuffer(gradHInQ_, 1, ADAM_TILE * sizeof(half));
        pipe_.InitBuffer(mInQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(vInQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(varHOutQ_, 1, ADAM_TILE * sizeof(half));
        pipe_.InitBuffer(mOutQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(vOutQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(varFBuf_, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(gradFBuf_, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(tmp1Buf_, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(tmp2Buf_, ADAM_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (my_elems_ <= 0) return;
        int32_t loops = (my_elems_ + ADAM_TILE - 1) / ADAM_TILE;
        for (int32_t i = 0; i < loops; i++) {
            int32_t offset = i * ADAM_TILE;
            int32_t cur = ADAM_TILE;
            if (offset + cur > my_elems_) cur = my_elems_ - offset;
            int32_t aligned = ((cur + 15) / 16) * 16;
            CopyIn(offset, aligned);
            Compute(aligned);
            CopyOut(offset, aligned);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t count)
    {
        LocalTensor<half> varH = varHInQ_.AllocTensor<half>();
        LocalTensor<half> gradH = gradHInQ_.AllocTensor<half>();
        LocalTensor<float> mIn = mInQ_.AllocTensor<float>();
        LocalTensor<float> vIn = vInQ_.AllocTensor<float>();
        DataCopy(varH, gmVar_[offset], count);
        DataCopy(gradH, gmGrad_[offset], count);
        DataCopy(mIn, gmM_[offset], count);
        DataCopy(vIn, gmV_[offset], count);
        varHInQ_.EnQue(varH);
        gradHInQ_.EnQue(gradH);
        mInQ_.EnQue(mIn);
        vInQ_.EnQue(vIn);
    }

    __aicore__ inline void Compute(int32_t count)
    {
        LocalTensor<half> varH = varHInQ_.DeQue<half>();
        LocalTensor<half> gradH = gradHInQ_.DeQue<half>();
        LocalTensor<float> m = mInQ_.DeQue<float>();
        LocalTensor<float> v = vInQ_.DeQue<float>();

        LocalTensor<half> varHOut = varHOutQ_.AllocTensor<half>();
        LocalTensor<float> mOut = mOutQ_.AllocTensor<float>();
        LocalTensor<float> vOut = vOutQ_.AllocTensor<float>();
        LocalTensor<float> varF = varFBuf_.Get<float>();
        LocalTensor<float> gradF = gradFBuf_.Get<float>();
        LocalTensor<float> t1 = tmp1Buf_.Get<float>();
        LocalTensor<float> t2 = tmp2Buf_.Get<float>();

        Cast(varF, varH, RoundMode::CAST_NONE, count);
        Cast(gradF, gradH, RoundMode::CAST_NONE, count);

        // m_new
```

**孪生卡 ec-276 合并的独有内容**（ec-276-dav-m200-arch-impl-has-no-divs-tensor-scalar-sub.md 已删除）：

> Div(t1, t3, b2, n); // TRUE IEEE division, bit-faithful to torch `/ scalar`

> - **Evidence**: 17_AdamW 310p build iter-1 (2026-08-15): 6× Divs → Duplicate+Div; fp32 path bit-exact vs the CPU chain (`17_AdamW/failures_ledger.md:9`). Prior art, same arch-gap class: celu 310p (reciprocal workaround — concedes 0–1 ULP; superseded by this fix pattern where bit-exactness matters); op#17 AdamW V220 port 2026-05-02 (OL-127's Divs row — reciprocal-in-pybind+`Muls` era).

> - Cross-ref [`OL-127`](ol_127_cann_api_surface_shrinks_across_versions_check_api.md) (API surface shrinks across versions — the V220 Divs row this extends to arch-200x with a better workaround), OL-256 (reciprocal perf option, different scope), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (API-catalog lookup discipline), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) (dav-m200 signature-surface gap class). backend=ascendc.

## 适用场景

- **工程形态**：dav-m200 没有 `Divs`（张量÷标量）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
