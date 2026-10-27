---
type: Runbook
title: "Dual-axis attention over one physical feature matrix must keep the two Gram axes distinct: position attention is softmax(X^T X / sqrt(C)) while channel attention is softmax(X X^T / sqrt(N))"
description: "Dual-axis attention over one physical feature matrix must keep the two Gram axes distinct: position attention is softmax(X^T X / sqrt(C)) while channel attention is softmax(X X^T / sqrt(N))"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-519"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-519 — OL-519: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=dual-axis position+channel self-attention over a flattened feature X[C,N]。
- 验证来源：host algebra discriminator C=2,N=3 plus 4-46 DAModule retained AscendC candidate on Ascend950DT_9582 / CANN 9.1.0, 2026-09-06 (186/186 precision and 186/186 determinism under its bound non-O5 receipt)。

**OL-519** — **dual-axis attention over one physical feature matrix must keep the two Gram axes distinct: for `X[C,N]`, position attention is `softmax(X^T X / sqrt(C))` while channel attention is `softmax(X X^T / sqrt(N))`.** The position branch's explicit `permute` pair can be absorbed into matmul transpose/output-layout choices without materialising `X^T`: return `X @ softmax(X^T X / sqrt(C))^T`. Channel attention remains `softmax(X X^T / sqrt(N)) @ X`. Use a rectangular `C != N` case as the discriminator; a square case can hide both an axis swap and a wrong shared scale.

- Former A5Ops ID: `OL-519`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=dual-axis position+channel self-attention over a flattened feature X[C,N]`
`verified_on: host algebra discriminator C=2,N=3 plus 4-46 DAModule retained AscendC candidate on Ascend950DT_9582 / CANN 9.1.0, 2026-09-06 (186/186 precision and 186/186 determinism under its bound non-O5 receipt)`
`unverified_on: genuine A3 execution and every other operator implementation; Ascend950PR and all other SoCs; broader portability is predicted from the algebra but not device-verified; performance benefit; formal O5 or product closure; the retained candidate changed more than layout, so its 186/186 result is compatibility evidence, not a single-variable layout A/B`

- **Category**: operator semantics / attention layout planning
- **Trigger**: one flattened feature map feeds both a position/spatial attention branch and a channel attention branch, and the port wants to avoid explicit transpose kernels by using matmul transpose flags or column-major output.
- **Principle**: write the two branches from the same physical matrix before choosing any storage transform. With `X` laid out as `[C,N]` and row-wise softmax:
  - position: `P = softmax(X^T X / sqrt(C))`, shape `[N,N]`; the reference's logical output `P X^T` has shape `[N,C]`, so the physical `[C,N]` result is exactly `X P^T`;
  - channel: `A = softmax(X X^T / sqrt(N))`, shape `[C,C]`; the physical result is `A X`.
  The normaliser follows the contraction width of the corresponding Gram matrix (`C` for position, `N` for channel), not a shared attention constant. Runtime transpose flags may remove data movement, but they do not license swapping these axes.
- **Concrete AscendC mapping**: the retained 4-46 candidate keeps the convolution output in `[C,N]`, and `DaCvPlan(position, probability, ...)` selects the four contraction shapes, offsets and transpose flags before each `SetOrgShape` / `SetSingleShape` / `SetTensorA` / `SetTensorB` call. This is an application of the identity above, not a general prescription for one `MatmulImpl` tiling; OL-508 governs that API's shape-field mapping.
- **Falsifier / negative control**: use `C != N` and a non-symmetric `X`. For `X=[[0.2,-0.7,1.3],[1.1,0.4,-0.2]]` (`C=2,N=3`), a dependency-free host calculation gives `max_abs(position_reference, X@P^T)=0` and `max_abs(channel_reference, A@X)=0`; mutating only the channel scale from `1/sqrt(N)` to `1/sqrt(C)` gives `max_abs=0.067721056299282711`. The score shapes are `[3,3]` and `[2,2]`, so exchanging the branches fails structurally. A square or symmetric fixture is not an adequate control.
- **Bound evidence and limit**: reference source blob `3389fa5895505b3abb4a605423cd4bb2f83394aa` (`model.py` sha256 `d5241ff24d9a3fb3aacfb27db9edf80bcb801298f27dfa53a81c0c37856f7799`) contains the two explicit PyTorch branches. Retained candidate `output/npukernelbench/src/kernels/46_DAModule/candidate_fuse4a_1ac00aeb/` binds kernel-header sha256 `7b81bf66752cb7e1f545774e9826516b11d52a6ed8c24932e6b6e80c41e07abc`, DSO `1ac00aeb8efc4352207d9d425f12f2388f4c62859c181e8fbc1b385598b6984e`, and the 186-case non-O5 receipt. The sibling root archive explicitly says `CODE ONLY`; no old verdict is carried into this entry. No selection, Read/adopt, source-applied consumer, A3, performance, or product claim is established here.
- **Blind-first disposition**: no equivalent dual-axis identity was found in current canonical bodies/index or any remote ref. P-P101/P-P102/P-P103 describe generic attention pipelines but do not bind the position/channel Gram axes; OL-508 describes `MatmulImpl` shape fields, not this operator algebra. OL-516/OL-517 are build/instrumentation findings from the same archive and do not substitute for the mechanism.
- **Other instances (predicted)**: DANet-style position/channel attention, criss-cross or polarized variants only when their source actually reduces to these two Gram forms. The name “dual attention” alone is not enough; bind the forward equations first.
- **Cross-ref**: P-P101/P-P102/P-P103 (generic attention execution templates), OL-508 (`MatmulImpl` shape-field mapping), OL-516/OL-517 (same archive, orthogonal build/measurement axes). backend=ascendc.
