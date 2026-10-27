---
type: Runbook
title: "on the CANN 9.1.0 tikcfw (dav-m200 builds) `FetchEventID` returns `TEventID`, not `event_t` — drop the `static_cast<event_t>` idiom"
description: "on the CANN 9.1.0 tikcfw (dav-m200 builds) `FetchEventID` returns `TEventID`, not `event_t` — drop the `static_cast<event_t>` idiom"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["FetchEventID", "TEventID"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-213 — EC-213: A5Ops additive extension

> 关联：同一 EC-213 主题已由既有卡 [ec_213_on_the_cann_9_1_0_tikcfw_dav_m200_builds_fetchev.md](ec_213_on_the_cann_9_1_0_tikcfw_dav_m200_builds_fetchev.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.1.0；op_class=all（任何使用手工 SetFlag/WaitFlag 跨 pipe 标记的 kernel）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（reduce_sum 310p，2026-08-17；sort 310p 2026-08-18 第二实例）。

**EC-213** — **on the CANN 9.1.0 tikcfw (dav-m200 builds) `FetchEventID` returns `TEventID`, not `event_t`** — the `static_cast<event_t>` idiom carried by EC-13's fix snippets (correct on 220x/950PR header sets) fails to COMPILE here. Fix: declare as `TEventID`/`auto` and pass straight to `SetFlag/WaitFlag`; never force the cast idiom across SoC/CANN versions — read the toolchain header first (OL-80 habit applies to tikcfw headers). reduce_sum 310p 2026-08-17 (event-id scaffolding failed compile → TEventID → 37/37 tier-1). Cross-ref [EC-13](ec_13_ascendc_syncfunc_does_not_exist.md), P-P201（A5Ops 绑定外部引用；本语料未收录目标）, [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md).

- Former A5Ops ID: `EC-213`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a; op_class=all (any kernel using manual SetFlag/WaitFlag cross-pipe flags)`
`unverified_on: soc=Ascend950PR, Ascend910 on cann=9.1.0 (the return type is a per-toolchain header property — other SoC header sets on the same CANN may differ)`

- **Symptom**: a kernel that stores the fetched event as `event_t ev = ...FetchEventID(...)` — the idiom carried by EC-13's fix snippets, correct on the 220x/950PR header sets — fails to COMPILE against the CANN 9.1.0 dav-m200 tikcfw: `FetchEventID` is declared to return `TEventID` (an int8-backed handle), not `event_t`.
- **Fix**: declare the variable as `TEventID` (or `auto`) and pass it straight to `SetFlag/WaitFlag<HardEvent::...>(ev)` — both accept `TEventID` on this header set. Do NOT force the `static_cast<event_t>` cast idiom across SoC/CANN versions.
- **Detection**: a compile error on the `FetchEventID` assignment line in code that built clean on another target → per-toolchain return-type drift, not a logic bug. Read the toolchain header for the actual return type before casting (the OL-80 per-SoC header-check habit applies to tikcfw headers too).
- **Evidence**: reduce_sum 310p (2026-08-17, first instance): P-P201 S↔V interlock scaffolding written with the `event_t` idiom failed to compile on the CANN 9.1.0 tikcfw; switching the declaration to `TEventID` built clean; kernel shipped 37/37 tier-1 PASS.
- **Evidence (2nd instance)**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 1): `sort_kernel.h` event-fetch code written with the `event_t` idiom failed compile on the same tikcfw; `TEventID` declaration built clean; kernel shipped 26/26 bit-exact.
- **Cross-ref**: EC-13 (the SyncFunc/FetchEventID entry whose snippets carry the `static_cast<event_t>` idiom — correct on its own toolchain), P-P201 (the FetchEventID interlock pattern), OL-80. backend=ascendc.
