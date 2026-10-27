---
type: Runbook
title: "`DataCopyPad` `rightPadding` is counted in elements but each side must occupy at most 32 bytes — derive the alignment as `32 / sizeof(T)`"
description: "`DataCopyPad` `rightPadding` is counted in elements but each side must occupy at most 32 bytes — derive the alignment as `32 / sizeof(T)`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-172"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-172 — EC-172: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579；CANN 9.1.0.B060；op_class=all（带显式 rightPadding 的 DataCopyPad GM→UB 行加载）。
- 验证来源：soc=Ascend950PR_9579；CANN 9.1.0.B060（op57 终端绑定证据；op55 仅为 lineage 受限佐证；A3 未验证）。

**EC-172** — `DataCopyPad` `rightPadding` is counted in elements but each side must occupy at most 32 bytes. A dtype-naive `AlignUp16` is legal for fp16/bf16 and can request 60 bytes for fp32, producing errcode 78; derive the alignment as `32 / sizeof(T)` (fp32 `AlignUp8`, fp16/bf16 `AlignUp16`). Bound evidence: op57 on Ascend950PR_9579/CANN 9.1.0.B060; op55 is lineage-limited corroboration only; A3 unverified.

- Former A5Ops ID: `EC-172`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=all (DataCopyPad GM-to-UB row loads with explicit rightPadding)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060`
`unverified_on: soc=Ascend910_9382 (A3 220x — not probed); cann=9.1.T500+9.0.0`

- **Admission boundary**: narrow EC-level correction backed by one terminal-bound A5 op and the in-repo CANN API contract. It is not an OL/P-pattern promotion, and the A3 behavior is unverified.
- **Error pattern**: an otherwise identical kernel raises errcode 78, "MTE instruction abnormal", on fp32 rows while fp16/bf16 rows are clean. Audit any dtype-naive element alignment used to construct `DataCopyPadExtParams<T>::rightPadding` before changing the kernel algorithm.
- **Contract and root cause**: `rightPadding` is expressed in **elements**, but the API constrains each padding side in **bytes**: `leftPadding * sizeof(T) <= 32` and `rightPadding * sizeof(T) <= 32`. The vendored CANN reference states the 32-byte limit at `docs/references/ascendc_cann_910beta/pages/atlasascendc_api_07_0265.md:155`. A fixed `AlignUp16` gives fp16/bf16 at most 15 padding elements (30 bytes), but gives fp32 at most 15 elements (60 bytes), which violates the instruction bound.
- **Anti-pattern**: do not encode one alignment in element units and reuse it across dtypes. Derive the one-datablock element count as `32 / sizeof(T)`, then align the logical row to that count. For the demonstrated dtypes this is fp32 `AlignUp8` and fp16/bf16 `AlignUp16`.
- **Known-bad control**: `57_ParallelPolarizedSelfAttention` Phase-D iter 2 (2026-08-18, Ascend950PR_9579 / CANN 9.1.0.B060) used fp32 `AlignUp16`; its worst `rightPadding` was 15 elements = 60 bytes, and **every fp32 case** faulted with errcode 78. The fp16/bf16 arms, whose byte extent stayed within the bound, did not show this fault. Evidence: 内部路径（不公开）/home/npu_user/workspace/a3-the executor/私有沉淀仓_op62/workspace/57_ParallelPolarizedSelfAttention/failures_ledger.md:9` (sha256 `9876c96ce5298708ad6ee4fc7794282bbdb74161c3a91c71dadc28606ba9e6d6`).
- **Positive control**: changing only the row-alignment rule to dtype-aware fp32 `AlignUp8` removed this errcode-78 class while leaving fp16/bf16 unchanged. The retained terminal later completed Pass A 49/49 with one structural empty-batch skip and Pass B 50/50; those totals show the repair remained present, not that this EC alone caused every final pass. Terminal receipt: 内部路径（不公开）/home/npu_user/workspace/a3-the executor/私有沉淀仓_op62/workspace/57_ParallelPolarizedSelfAttention/verification.json:7` (sha256 `3f5efda9a036a9d2bd424e18008f0211c868b85af5f4d1be6670dd2a6ceb5867`).
- **Corroborating, lineage-limited witness only**: the current `55_OutlookAttention` workspace independently records `rightPadding=13` fp32 elements = 52 bytes, the same errcode-78 class, and a repair that caps in-instruction padding at `32 / sizeof(T)` while pre-zeroing any remaining lanes. Its verification belongs to an older run and does not bind the current intake, so it is supporting implementation evidence, **not** a second terminal-bound admission witness. Evidence: 内部路径（不公开）/home/npu_user/workspace/scan_agent/私有沉淀仓/workspace/55_OutlookAttention/failures_ledger.md:8 (sha256 `64303ce1d9ee8754c5cbb2486f8173bdbc7a0cef55ade9291bd754e04d3e143b`).
- **Distinct from**: EC-23's older UB-to-GM direction claim and PB-47's cross-iteration UB reuse race. EC-172 is a GM-to-UB instruction-parameter bound with a dtype-dependent byte conversion; fences do not legalize an oversized padding field. backend=ascendc.
