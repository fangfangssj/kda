---
type: Runbook
title: "DataCopyPad with byte-count extents and unaligned GM addresses is device-legal in both directions on Ascend950DT_9582 / CANN 9.1.0 — verified for one exact call-site set; does not relax EC-23 on 950PR"
description: "DataCopyPad with byte-count extents and unaligned GM addresses is device-legal in both directions on Ascend950DT_9582 / CANN 9.1.0 — verified for one exact call-site set; does not relax EC-23 on 950PR"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-452"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-452 — OL-452: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0 (image 33dd99717caf); bisheng=n/a; scope=the exact call-site set of ct2d_ko_kernel.cpp sha256 06f16658328afb36... (10_ConvTranspose2d KO v1)。
- 验证来源：the op's frozen fixed-50 case set, Pass-A under the npukernelbench_v1 comparator, 2026-09-11。

**OL-452** — **`DataCopyPad` with byte-count extents AND unaligned GM addresses is device-legal in BOTH directions on Ascend950DT_9582 / CANN 9.1.0 — verified for ONE exact source's joint call-site set, and this does NOT relax EC-23 on 950PR.**

- Former A5Ops ID: `OL-452`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0 (image 33dd99717caf); bisheng=n/a; scope=the exact call-site set of ct2d_ko_kernel.cpp sha256 06f16658328afb36... (10_ConvTranspose2d KO v1)`
`verified_on: the op's frozen fixed-50 case set, Pass-A under the npukernelbench_v1 comparator, 2026-09-11`
`unverified_on: any other source; Ascend950PR / 910-family (A3 = UNKNOWN); other dtypes; the general claim "unaligned access is safe"`

- **Category**: platform_bug (VERIFIED-NOT-A-BUG, narrow joint instance)
- **Loaded by**: Builder, QA
- **Scope — hard boundary, do NOT extrapolate**: three GM->UB loads (whole-tensor x, per-co weight slab, bias vector) plus one UB->GM output-row store, ALL with byte-count extents and the store with generally unaligned GM row starts. This is a claim about that joint set on that chain, not about either direction in general.
- **Trigger**: considering `DataCopyPad` with byte lengths or unaligned GM addresses, in either direction, on the 950DT / CANN 9.1.0 chain, and wondering whether compile-clean implies device-legal.
- **Lesson**: on this chain both directions are device-legal, and the evidence is **joint end-to-end**, not per-direction isolation — the pinned fixed-50 Pass-A passed 50/50 with zero failing cases, with every call site exercised under genuinely unaligned parameters. Compile-clean preceded device-legal here; the reverse reading is explicitly NOT supported.
- **Measured coverage — calls that ARRIVE at the DataCopyPad, early-return rows excluded**:

  | call site | direction | arriving calls | byte%32!=0 | start%32!=0 |
  |---|---|---|---|---|
  | xL load | GM->UB | 14,802 | 801 | 0 |
  | wL slab | GM->UB | 14,802 | 1,190 | 675 |
  | biasL | GM->UB | 7,540 | 1,204 | 0 |
  | y row store | UB->GM | **14,802** | **7,874** | **6,348** |

  These are ARRIVAL counts. The retained coverage table
  (`docs/evidence/kb_op310_20260911/datacopypad_coverage_table.md`, retained on `main` by !1115) reports ISSUED
  counts that include rows the kernel early-returns before reaching the copy: xL/wL 14,927 with 829
  and 1,219; biasL 7,617 with 1,233. Only the `y` row is identical in both readings, because the
  store sits after the early-return path. **Provenance of the arrival figures**: supplied by the
  non-author exact-head review of this entry. They are NOT independently recomputed here — the
  retained table is a per-call-site aggregate with no per-case detail, so the early-return filter
  cannot be reproduced from the artifact alone. Anyone re-deriving them needs the frozen 50-case
  parameter table together with the kernel's early-return condition.

- **The 950PR boundary is UNRESOLVED, and this entry does not resolve it.** EC-23 states that
  `DataCopyPad` UB->GM crashes on **950PR** (507035). That statement is **already in conflict with
  other canonical entries in this same tree**, independently of anything measured here: OL-167 /
  P-P98 treat V351 `DataCopyPad` as usable in both directions, and ERROR_CORRECTIONS itself records
  three cross-op positives — `group_norm_silu` kw-1 (2026-05-24, both directions clean on V351),
  `top_k_top_p_sample` kw-4 (2026-06-24, int64 UB->GM on **Ascend950PR_9579**, no 507035), and the
  2026-05-18 hardware probe over `blockLen` in {31,33,47,63} — alongside the line concluding the
  upstream V220 guard is "stale for V351".
  **This entry deliberately takes no side.** Its evidence is Ascend950DT_9582 and says nothing about
  950PR either way, so it can neither uphold EC-23 nor retire it. Recorded as
  `BOUNDARY_UNRESOLVED: EC-23 vs OL-167/P-P98 on 950PR`, to be adjudicated by whoever holds 950PR
  evidence. A reader must not take this entry as licence to ignore EC-23, and must not take it as
  confirmation of EC-23 either. Adjacent boundary likewise not crossed: CAND-PP83 (GM->UB last tile,
  950PR).
- **Scope extension or independent mechanism? UNKNOWN.** Deciding that needs 950DT's architecture
  token: 950PR is `__NPU_ARCH__` 3510 (`DAV_3510`), but `hardware/target/ascend950dt.md` records no
  `__NPU_ARCH__` at all. If 950DT is also 3510 this is a second-SoC scope extension of the V351
  evidence; if not, it is an independent mechanism. The hardware doc gap is the blocker, and it is
  noted here rather than guessed.
- **Evidence** — retained in this exact from `main@20da0bff3baf0152f22ea47b469c64580ecf9bfb` / !1115: `docs/evidence/kb_op310_20260911/ko_device_p_pass_a_default_b7.json` (`b7_pass: 50`, `b7_failing_cases: []`, candidate binary sha256 aecc4de8..., dataset ea82472f..., comparator binding recorded), `docs/evidence/kb_op310_20260911/ko_device_p_release_readback.txt`, `docs/evidence/kb_op310_20260911/ko_device_p_MANIFEST.sha256`; bound source `ct2d_ko_kernel.cpp` sha256 06f16658328afb36...
- **Dedup receipt (blind-first; candidates read before the incumbent KB)**: the incumbent set searched
  was (a) the open evidence PR's `OL-484` / `OL-489` on `kimi/kb-op310-cann91-preprocess-20260911@94f6955ab241`, (b) canonical `OL-167` and `P-P98`,
  (c) `EC-23`, (d) `CAND-PP83`. Disposition: `OL-489` is the same semantic claim as this entry and is
  **superseded by OL-452** — it must not be landed separately under the the executor number; `OL-484` is a
  different subject (the build chain) and was dispositioned separately as an in-place correction to
  `PB-125` rather than a new id; `OL-167`/`P-P98` overlap in conclusion but are V351/950PR-scoped, and
  whether this 950DT evidence extends or sits beside them is UNKNOWN per the note above; `EC-23` is
  recorded as an unresolved conflict rather than resolved in either direction.
- **Provenance**: promoted from an immutable LOCAL_ONLY candidate carried as `OL-489` on `kimi/kb-op310-cann91-preprocess-20260911@94f6955ab241`. That number is the kimi lane's candidate identity (block OL-480..499) and is **not** a Scan allocation; this entry is allocated in the Scan block (440-459) under current ownership. Semantics, scope and denominators are preserved unchanged from the candidate. `OL-444`/`OL-445` were skipped: the block allocator sees only `main`, and both are already claimed by an unmerged Scan branch.
- **Cross-ref**: EC-23 (the 950PR UB->GM prohibition this does not relax), OL-167 / P-P98 (V351 DataCopyPad guidance), PB-125 (the same op's build chain, corrected in the same commit).
