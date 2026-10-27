---
type: Runbook
title: "on dav-m200, scalar GM stores sharing a 64B GM sector with an in-flight MTE3 DataCopy (UB→GM) silently DROP th"
description: "dav-m200 上标量 GM 写与在途 MTE3 DataCopy 共享 64B 扇区时会静默丢弃拷贝的最后 32B 块"
tags: [task.debugging, topic.data_movement.data_copy, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-123"]
---

# on dav-m200, scalar GM stores sharing a 64B GM sector with an in-flight MTE3 DataCopy (UB→GM) silently DROP th

**PB-123** — **on dav-m200, scalar GM stores sharing a 64B GM sector with an in-flight MTE3 `DataCopy` (UB→GM) silently DROP the copy's last 32B block** — output keeps pre-launch bytes (y=0.0 holes, run-to-run non-determinism; 6/40 cases on abs, sector arithmetic confirmed on all failing + both tail-bearing passing cases). Same-core ORDERING hazard, distinct from the cross-core tear CAND-M200-CROSS-PIPE-32B-BLOCK-RACE. Fix: `SetFlag/WaitFlag<HardEvent::MTE3_S>` (pool-fetched event id) after the vector body, before any scalar GM store sharing a sector (+ defensive `S_MTE3` after a scalar head). Post-fence 40/40 bit-exact + det. abs 310p 2026-08-17 (DEVICE-RUN). Cross-ref CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS, [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md), CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL. [dav-m200/Ascend310P3, pipe-ordering, silent-data-loss]

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (kernels mixing vector DataCopy UB→GM output with scalar-pipe GM stores, e.g. a sub-32B tail handled element-wise)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (abs 310p, 2026-08-17 — DEVICE-RUN, benchmark-acceptance lane)`
`unverified_on: soc=Ascend910, Ascend950PR (the sector write-combine behavior is a dav-m200 observation; re-probe before generalizing)`

- **Severity**: HIGH — silent data loss + run-to-run non-determinism; the kernel launches clean, syncs clean, and the output region keeps its pre-launch bytes (fresh `empty_like` allocations make the holes show up as `y=0.0` or garbage, varying per run).
- **Symptom**: when a kernel's vector body ends with a `DataCopy` (UB→GM, MTE3) whose LAST 32B block shares a 64B GM sector with a subsequent SCALAR GM store sequence (e.g. an unaligned tail written element-wise via `GlobalTensor.SetValue`), the MTE3 block is silently dropped. abs 310p: 6/40 benchmark cases failed with max_abs_diff up to 2.7e13 + `y=0.0` holes; the failing set varied across reruns.
- **Mechanism**: sector-granularity write-combine clobber — unordered scalar-pipe GM stores vs an in-flight MTE3 copy on dav-m200. Sector arithmetic confirmed on all 6 failing AND both tail-bearing passing cases: the passing tails started exactly on a new 64B sector (no sharing → no clobber). This is an ORDERING hazard (same core, in-flight copy vs later scalar stores), distinct from the API-absence stubs PB-105/PB-112/PB-113 and from the CROSS-CORE write-write tear CAND-M200-CROSS-PIPE-32B-BLOCK-RACE (32B block, two cores, S-vs-MTE3).
- **Fix**: after the vector body, BEFORE any scalar GM store that may share a sector: `SetFlag/WaitFlag<HardEvent::MTE3_S>` (scalar pipe stalls until the copy retires — the drain CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS catalogs as the only reliable MTE3 fence on dav-m200); plus a defensive `S_MTE3` after a non-zero scalar HEAD sequence. Use pool-fetched event ids (`FetchEventID(HardEvent::MTE3_S)`), never literals (CAND-DAV-M200-HARDCODED-EVENT-ID-HANG). Post-fix: the 6 previously-failing shapes bit-exact + deterministic.
- **Detection rule for future ops**: any dav-m200 kernel mixing vector `DataCopy` output with scalar GM tail/head writes needs the MTE3→S fence at the copy↔scalar boundary. Symptom signature: large-diff holes localized to the last 32B block before a scalar-written region, flaky across reruns.
- **Concrete anchor**: `代码片段 abs/abs_kernels.cpp.snippet.cpp（见文末）` (MTE3_S fence before scalar tail; `:74-76` defensive S_MTE3 after scalar head — workspace scratch, gitignored).
- **Evidence**: abs 310p port (2026-08-17, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): `abs/knowledge_update.md` finding 1 + `abs/failures_ledger.md` precision iter 1. 6/40 failing cases → post-fence 40/40 bit-exact vs `310p_vendor_native` golden, det 40/40 (`abs/verification.json`).
- **Other instances (predicted)**: any m200 kernel with scalar tail writes adjacent to vector `DataCopy` output — scatter tails, unaligned elementwise (PB-105 forces scalar tails on this SoC, so the class is broad), per-core variable-length writes, RMW fallback paths (diag_v2-style).
- **Cross-ref**: CAND-M200-CROSS-PIPE-32B-BLOCK-RACE (the cross-core S-vs-MTE3 32B-block sibling), CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS (MTE3_S is the reliable drain; this is its 4th witness), CAND-DAV-M200-HARDCODED-EVENT-ID-HANG (event ids from the pool), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (DataCopyPad absent on m200 — why scalar tails exist here at all), CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL (the tail recipe this fence completes), [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (32B copy quantum). backend=ascendc.

## 内嵌代码片段

### abs/abs_kernels.cpp.snippet.cpp

```cpp
            // setup — pure fixed-cost reduction, identical data flow
            // (CopyIn -> Abs -> CopyOut on the same 32B-aligned body).
            pipe_.InitBuffer(inBuf_, TILE_BYTES);
            pipe_.InitBuffer(outBuf_, TILE_BYTES);
            LocalTensor<T> in = inBuf_.Get<T>();
            LocalTensor<T> out = outBuf_.Get<T>();
            uint32_t n = static_cast<uint32_t>(bodyLen);
            DataCopy(in, xGm_[head], n);
            event_t evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
```
