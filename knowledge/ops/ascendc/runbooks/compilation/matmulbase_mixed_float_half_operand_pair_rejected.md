---
type: Runbook
title: "CONFIRMED — on `Ascend950DT_9582` / CANN 9.1.0 B150, the regbase `fa_base_matmul::MatmulBase` rejects a MIXED `float`/`half` operand pair"
description: "CONFIRMED — on `Ascend950DT_9582` / CANN 9.1.0 B150, the regbase `fa_base_matmul::MatmulBase` rejects a MIXED `float`/`half` operand pair"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-75"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-75 — PB-75: CONFIRMED — on `Ascend950DT_9582` / CANN 9.1.0 B150, the regbase `fa_base_matmul::MatmulBase` rejects a MIXED `fp

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582；CANN 9.1.0 (B150)；op_class=flash_attention_style_mix_aic_aiv；api=fa_base_matmul::MatmulBase（regbase close-port of matmul.h L1003-1018）。
- 验证来源：a5_ops 75_BatchDecodeMlaPaged — fp32-P 臂复现失败构建；候选 DSO dfa36395aa822e1112dbf209b7a56e8c29fcd2bbca36eb2724ebcc9d85a8f328 验证 workaround（precision 50/50 OVERALL_T1，profile npukernelbench_v1@b7f91f8bc067bd9a，冻结 50-case 分母）。

`applies_to: soc=Ascend950DT_9582; cann=9.1.0 (B150); op_class=flash_attention_style_mix_aic_aiv; api=fa_base_matmul::MatmulBase (regbase close-port of matmul.h L1003-1018)`
`verified_on: a5_ops:75_BatchDecodeMlaPaged — failing build reproduced on the fp32-P arm; workaround verified by candidate DSO dfa36395aa822e1112dbf209b7a56e8c29fcd2bbca36eb2724ebcc9d85a8f328 (precision 50/50 OVERALL_T1, profile npukernelbench_v1@b7f91f8bc067bd9a, frozen 50-case denominator)`
`last_verified: 2026-09-05`
`status: CONFIRMED`

**The constraint.** `MatmulBase<A, B, C, baseM, baseN, baseK, AL, BL>` declares **A and B as
independent template parameters**, so a mixed pair is *syntactically* expressible. It does not
compile. Instantiating `MatmulBase<float, half, float, …>` fails inside the CANN header:

```
CANN 安装树内 /usr/local/Ascend/cann-9.1.0/aarch64-linux/asc/impl/basic_api/dav_3510/kernel_operator_mm_impl.h:475:23
error: the 3rd parameter maybe need a type '__cb__ float *'
```

(`ext:` — CANN toolkit header, outside this repo and therefore not auditable from it; the line
number is pinned to CANN 9.1.0 B150 as installed at that path.)

Both `MatmulBase<half,half,float,…>` and `MatmulBase<float,float,float,…>` instantiate fine on the
same build — the failure is specific to the **mixed** pair, not to either dtype.

⇒ **In THIS configuration you cannot raise the precision of one operand alone.** This is a
statement about the tested combination — `Ascend950DT_9582`, CANN 9.1.0 B150, the regbase
close-port `fa_base_matmul::MatmulBase` with these template arguments. It is **not** a claim
that single-operand precision is unavailable on other SoCs, other CANN versions, other cube
APIs (`matmul::MatmulImpl`, KFC paths), or other `baseM/baseN/baseK`/layout combinations.
Raising both means carrying the B-side
tile at fp32 in L1 as well: for a `BLOCK_N × MAX_DC` V tile that is 64·512·4 = 128 KB per buffer
(256 KB double-buffered), plus a UB round-trip to cast, since `CopyToL1Nd2NzCap` copies and does not
convert.

**Workaround — split the A operand into hi + lo, two accumulating matmuls.** Where A is a value
whose precision matters (softmax probabilities entering P@V), represent it as two `fp16` parts and
issue two matmuls into the same L0C:

```cpp
Cast(pHi, sCompute, RoundMode::CAST_RINT, cnt);          // hi
Cast(resid, pHi, RoundMode::CAST_NONE, cnt);
Sub(resid, sCompute, resid, cnt);                        // residual = P - fp16(P)
Cast(pLo, resid, RoundMode::CAST_RINT, cnt);             // lo
// ... both staged to the GM ring, lo at a +BLOCK_M*BLOCK_N half offset ...
MatmulBase<QType, QType, float, …>(pHiL1, vL1, …, cL0, pm);      // pm.isOutKFisrt = true
MMParam pmLo = MakeMMParam(…);
pmLo.isOutKFisrt = false;                                        // ACCUMULATE
MatmulBase<QType, QType, float, …>(pLoL1, vL1, …, cL0, pmLo);
```

Both calls use an **already-instantiated** `MatmulBase<QType,QType,float>`, and `isOutKFisrt=false`
as the accumulate flag is the same idiom the kernel's own QK stage uses for `S += Qp @ Kpe^T`.
The second matmul reuses the B tile already resident in L1.

**Measured effect** (75_BatchDecodeMlaPaged, frozen 50, bound profile):

| | precision | per-case geomean |
|---|---|---|
| single fp16 A | 49/50 (one case failing MERE only) | 4.056362 |
| split hi+lo   | **50/50 OVERALL_T1** | 3.778260 |

⚠ **Evidence boundary on these two numbers.** Both are complete and were independently
re-derived (50 cases / 58 outputs profile-accepted; performance GM re-derived 100/100 legs
from preserved raw with exact 8-call splits, fallback 0). However **neither graded run
recorded the loaded-DSO binding from inside its own measurement process**, so the association
between these numbers and that specific binary is *pending* a fresh acceptance that captures
it per-leg. **The boundary covers BOTH figures, not only the delta**: the 49/50 and 50/50
precision numbers come from the same unbound collections, so the qualitative contrast carries
it too. Cite neither the 49→50 comparison nor **-6.86%** without stating that the
collection-association is pending. This grade is updated only after a fresh bound acceptance.

The failing case's MERE went 1.223e-3 → 1.5298e-05 (threshold 9.766e-04) with **zero regression**
in the other 49. Cost: **-6.86%** of the performance geomean — well below the 2× the extra matmul
suggests, because BMM2 was only part of the work and the second call reuses the resident B tile.

**Check before reaching for this.** The staging ring must have headroom for the low part. In this
op it was free: the workspace was already allocated `torch::kFloat` host-side while the kernel
wrote `half`, so each slot held 4096 halves with only 2048 used — no host-side change was needed.
An exactly-sized ring would need one.

**What is NOT claimed.**
- No claim that mixed operands fail on other SoCs, other CANN versions, or through
  `matmul::MatmulImpl` / KFC paths. This is the regbase close-port `MatmulBase` on
  `Ascend950DT_9582` / CANN 9.1.0 (B150) only.
- No claim that operand precision is the general cause of attention precision failures. In this op
  it was established by a single-variable change on real hardware; elsewhere it must be established
  the same way, not assumed from this entry.
- No claim that ANY number here is acceptance-grade. Both the precision contrast (49/50 → 50/50)
  and the -6.86% delta rest on collections that did not record their loaded-DSO binding, so the
  association is pending for both. The -6.86% additionally does not transfer: it is one op's ratio
  between two specific candidates, and where the split matmul dominates runtime the cost will be
  closer to 2×.
- This entry prescribes nothing about `bf16` or `fp8` operand pairs — untested here.

**Separately bound result for the same candidate (added 2026-09-05, own provenance).** The two
arms above stay as they are — the figures in them are NOT restated or replaced here, and their
pending loading association stands. What follows is a different, independently bound measurement
and is recorded only under its own identities:

    run_id        ds-op75splitp-boundv2-20260905T1020Z   (F leg re-collected; P not rerun)
    candidate     dfa36395aa822e1112dbf209b7a56e8c29fcd2bbca36eb2724ebcc9d85a8f328
    F             per-case geomean 3.7822326743114263, full 50, 42 faster cases
    binding       in-process hook v2-same-fd (32ca34bb), rewritten to follow the central P path's
                  `candidate_binary_identity` same-FD method after a TOCTOU window was found in v1
    association   candidate same-FD dev / inode / SHA matches the valid original central P50 on
                  the same DSO; all 100 raw PID/case/role/wrapper associations checked
                  independently; zero parser mismatch; raw 100 = exact 8 / skip 3 / mean 5
    F JSON sha256 7f68183da612e4d2e0cc2df9b13ee425e164c188f3a49235751bfe744029a2cb

⚠️ Scope of that line: **P was not rerun alongside this F.** It is a same-candidate full P/F pass
**in its stated white-box consumer, not an authoritative O5 result**, and it does not upgrade
anything in the two arms above.

**Cross-reference**: PB-100 (same op family, `MIX_AIC_1_2` + manual `CrossCore*` on this SoC).

<!-- 2026-09-05: removed a pointer to `ascend950dt.md` cube operand notes — no such file exists
     anywhere in this repo, so the reference resolved to nothing. Re-add a real path if and when
     a hardware note for this SoC lands. -->
