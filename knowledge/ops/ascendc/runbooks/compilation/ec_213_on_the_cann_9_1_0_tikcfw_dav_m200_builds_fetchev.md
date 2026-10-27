---
type: Runbook
title: "on the CANN 9.1.0 tikcfw (dav-m200 builds) FetchEventID returns TEventID, not event_t"
description: "CANN 9.1.0 tikcfw 里 `FetchEventID` 返回 `TEventID` 而非 `event_t`——EC-13 的 static_cast 惯用法在这里编译失败，用 `auto`/`TEventID` 直接传。"
tags: [task.debugging, topic.implementation.synchronization, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-213"]
---

# on the CANN 9.1.0 tikcfw (dav-m200 builds) FetchEventID returns TEventID, not event_t

**EC-213** — **on the CANN 9.1.0 tikcfw (dav-m200 builds) `FetchEventID` returns `TEventID`, not `event_t`** — the `static_cast<event_t>` idiom carried by [`EC-13`](ec_13_ascendc_syncfunc_does_not_exist.md)'s fix snippets (correct on 220x/950PR header sets) fails to COMPILE here. Fix: declare as `TEventID`/`auto` and pass straight to `SetFlag/WaitFlag`; never force the cast idiom across SoC/CANN versions — read the toolchain header first ([`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) habit applies to tikcfw headers). reduce_sum 310p 2026-08-17 (event-id scaffolding failed compile → TEventID → 37/37 tier-1). Cross-ref [`EC-13`](ec_13_ascendc_syncfunc_does_not_exist.md), [`P-P201`](../performance/p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md).

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a; op_class=all (any kernel using manual SetFlag/WaitFlag cross-pipe flags)`
`unverified_on: soc=Ascend950PR, Ascend910 on cann=9.1.0 (the return type is a per-toolchain header property — other SoC header sets on the same CANN may differ)`

- **Symptom**: a kernel that stores the fetched event as `event_t ev = ...FetchEventID(...)` — the idiom carried by EC-13's fix snippets, correct on the 220x/950PR header sets — fails to COMPILE against the CANN 9.1.0 dav-m200 tikcfw: `FetchEventID` is declared to return `TEventID` (an int8-backed handle), not `event_t`.
- **Fix**: declare the variable as `TEventID` (or `auto`) and pass it straight to `SetFlag/WaitFlag<HardEvent::...>(ev)` — both accept `TEventID` on this header set. Do NOT force the `static_cast<event_t>` cast idiom across SoC/CANN versions.
- **Detection**: a compile error on the `FetchEventID` assignment line in code that built clean on another target → per-toolchain return-type drift, not a logic bug. Read the toolchain header for the actual return type before casting (the OL-80 per-SoC header-check habit applies to tikcfw headers too).
- **Evidence**: reduce_sum 310p (2026-08-17, first instance): P-P201 S↔V interlock scaffolding written with the `event_t` idiom failed to compile on the CANN 9.1.0 tikcfw; switching the declaration to `TEventID` built clean; kernel shipped 37/37 tier-1 PASS.
- **Evidence (2nd instance)**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 1): `sort_kernel.h` event-fetch code written with the `event_t` idiom failed compile on the same tikcfw; `TEventID` declaration built clean; kernel shipped 26/26 bit-exact.
- **Cross-ref**: [`EC-13`](ec_13_ascendc_syncfunc_does_not_exist.md) (the SyncFunc/FetchEventID entry whose snippets carry the `static_cast<event_t>` idiom — correct on its own toolchain), [`P-P201`](../performance/p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md) (the FetchEventID interlock pattern), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

**孪生卡 ec-285 合并的独有内容**（ec-285-on-the-cann-9-1-0-tikcfw-dav-m200-builds-fetchev.md 已删除）：

> - **Evidence**: reduce_sum 310p (2026-08-17, first instance): P-P123 S↔V interlock scaffolding written with the `event_t` idiom failed to compile on the CANN 9.1.0 tikcfw; switching the declaration to `TEventID` built clean; kernel shipped 37/37 tier-1 PASS.

> - **Cross-ref**: [`EC-13`](ec_13_ascendc_syncfunc_does_not_exist.md) (the SyncFunc/FetchEventID entry whose snippets carry the `static_cast<event_t>` idiom — correct on its own toolchain), P-P123 (the FetchEventID interlock pattern), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.
