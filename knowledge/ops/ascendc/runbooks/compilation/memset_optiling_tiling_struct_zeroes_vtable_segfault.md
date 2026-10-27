---
type: Runbook
title: "`memset(0)` on an `optiling` tiling struct zeroes the vtable pointer — segfault on first virtual dispatch"
description: "`memset(0)` on an `optiling` tiling struct zeroes the vtable pointer — segfault on first virtual dispatch"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["MatmulApiTiling", "SoftMaxTiling", "TilingDef"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-202 — EC-202: A5Ops additive extension

> 关联：同一 EC-202 主题已由既有卡 [ec_202_memset_0_on_an_optiling_tiling_struct_tcubetilin.md](ec_202_memset_0_on_an_optiling_tiling_struct_tcubetilin.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=CUBE_MIX——任何持有 optiling tiling struct 的自定义 launch 宿主路径。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（swin_attention_score_quant iter 1，2026-08-11）。

**EC-202** — `memset(0)` on an `optiling` tiling struct (`TCubeTiling`/`SoftMaxTiling`, or any host struct embedding them) zeroes the `TilingDef` **vtable pointer → segfault on first virtual dispatch** (`GetTiling`/`SaveToBuffer`/getters); these look like POD but are polymorphic (`optiling::TilingDef` base). Fix: standalone `optiling::TCubeTiling v;` (constructor sets the vtable) — never `memset` the whole struct, only scalar fields. Evidence: swin_attention_score_quant iter 1 2026-08-11 Ascend310P3/dav-m200 CANN 9.0.0. Cross-ref [OL-268](ol_268_tilingdata_two_registration_styles_mismatch.md) (sibling), EC-40. backend=ascendc.

- Former A5Ops ID: `EC-202`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3/dav-m200; cann=9.0.0; bisheng=n/a; op_class=CUBE_MIX — any custom-launch host path that holds an optiling tiling struct`
`verified_on: soc=Ascend310P3; cann=9.0.0`
`unverified_on: soc=Ascend950PR/V351 — the polymorphism is a CANN host tiling-lib structural property (optiling structs derive from optiling::TilingDef across chips), so the trap is expected to be chip-independent, but it is cross-verified only on dav-m200 here`

- **Symptom (runtime, no error code)**: the host process segfaults the first time a virtual method is called on a tiling struct that was zero-initialized with `memset(&td, 0, sizeof(td))` — typically inside `MatmulApiTiling::GetTiling(...)` or a later `SaveToBuffer(...)` / `get_*()` getter. A standalone test calling the same `MatmulApiTiling` sequence on a non-memset'd struct works fine, so it looks data-dependent rather than API-dependent.
- **Root cause**: `optiling::TCubeTiling` and `optiling::SoftMaxTiling` derive from `optiling::TilingDef`, which has virtual functions → the object begins with a vtable pointer. `memset(0)` overwrites that pointer, so the next virtual dispatch jumps through a null/garbage vtable → SIGSEGV. This is the standard C++ "do not memset a polymorphic object" trap, easy to hit here because these structs look like plain POD tiling blobs.
- **Fix**: never `memset` the whole tiling struct. Either (1) declare standalone `optiling::TCubeTiling qkTiling;` / `optiling::SoftMaxTiling smTiling;` variables — their constructors run and set up the vtable, and `GetTiling` fills the scalar fields; or (2) if you hold an aggregate struct, zero only its scalar/POD fields manually and leave the embedded `optiling` members default-constructed.
- **Concrete anchor** (`kernel/pybind11.cpp`):
  ```
  // WRONG (original) — HostSwinTilingData embeds TCubeTiling/SoftMaxTiling;
  //                    memset zeroes their vtable ptr → segfault in GetTiling.
  HostSwinTilingData td; std::memset(&td, 0, sizeof(td));   // ❌
  // RIGHT — standalone optiling vars; constructors set up the vtable.
  optiling::TCubeTiling   qkTiling;   // 内部路径（不公开）pybind11.cpp:89 (workspace kernel/pybind11.cpp)
  optiling::TCubeTiling   pvTiling;   // :90
  optiling::SoftMaxTiling smTiling;   // :91
  qkBmm.GetTiling(qkTiling);          // :109 — virtual dispatch OK
  ```
- **Detection**: stderr shows the pybind's pre-tiling log line (e.g. "[swin_pybind] QK tiling start") with NO corresponding "done" line, then a SIGSEGV — and the identical `MatmulApiTiling` sequence runs clean in a standalone host test.
- **Evidence**: swin_attention_score_quant iter 1 (2026-08-11, Ascend310P3 / dav-m200, CANN 9.0.0) — failures-ledger `iter 1 | runtime (not compile) | novel | pybind11.cpp:memset on optiling struct → segfault`. Fix: removed the aggregate `memset`, used standalone `optiling::` variables → resolved. Pairs with sibling OL-268 (serializing these same structs to device via `SaveToBuffer`).
- **Cross-ref**: OL-268 (sibling — `SaveToBuffer` serialization of these same host structs, same op/host path), EC-40 (device POD `TCubeTiling` = 50 int32 = 200 B). backend=ascendc.

---
