---
type: Runbook
title: "`AscendC::GetTPipePtr()` does not compile — `GetTPipePtr()` is declared at global scope, not in `namespace AscendC`"
description: "`AscendC::GetTPipePtr()` does not compile — `GetTPipePtr()` is declared at global scope, not in `namespace AscendC`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-171"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-171 — EC-171: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579；CANN 9.1.0.B060；bisheng=CANN-9.1.0 ccec；op_class=all。
- 验证来源：soc=Ascend950PR_9579；CANN 9.1.0.B060（SpatialGroupEnhance kw iter 1，2026-08-21；MUSEAttention 2026-08-23/24 反向过度修正证据见正文）。

**EC-171** — **`AscendC::GetTPipePtr()` does not compile**: `GetTPipePtr()` is declared at GLOBAL scope by `basic_api/kernel_common.h` even though it returns an `AscendC::TPipe*`, while every neighbour on the same line (`AscendC::HardEvent`, `SetFlag`, `WaitFlag`, `TPipe`) IS in `namespace AscendC` — so qualifying it is the natural generalization from its own neighbours, and wrong. Symptom: `no member named 'GetTPipePtr' in namespace 'AscendC'`. Fix: un-qualified callee, qualified enum — `static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S))`. Every existing occurrence in this KB is un-qualified, so a qualified one is generated, not copied. Cross-ref [EC-13](ec_13_ascendc_syncfunc_does_not_exist.md) (`AscendC::SyncFunc<>` — same call site, invented API rather than mis-scoped real one).

- Former A5Ops ID: `EC-171`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=CANN-9.1.0 ccec; op_class=all`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060`

- **Symptom**: compile error at a `FetchEventID` call site — `no member named 'GetTPipePtr' in
  namespace 'AscendC'`.
- **Root cause**: `GetTPipePtr()` is declared in the **global** namespace by the CANN kernel headers
  (`basic_api/kernel_common.h`), even though it RETURNS an `AscendC::TPipe*`. Every symbol beside it
  on the same source line — `AscendC::HardEvent`, `AscendC::SetFlag`, `AscendC::WaitFlag`,
  `AscendC::TPipe` — IS inside `namespace AscendC`. So qualifying `GetTPipePtr` is the natural
  generalization from its own immediate neighbours, and it is wrong. This is a generation trap
  rather than a knowledge gap: the model does not need to know the API, it needs to not smooth an
  irregularity.
- **Fix**: drop the qualifier on the callee; keep it on the enum argument.
  ```cpp
  // WRONG
  event_t e = static_cast<event_t>(AscendC::GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
  // RIGHT — un-qualified callee, qualified enum
  event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
  ```
- **Corroboration inside this KB** (every existing occurrence is un-qualified, so a qualified one is
  a generalization rather than a copy): `src/skills/references/target/ascendc/ERROR_CORRECTIONS.md:436`, `src/skills/references/target/ascendc/PLATFORM_BUGS.md:393`, and the
  fa_class template at
  `src/skills/references/target/ascendc/patterns/domains/fa_class/templates/op_kernel/kernel_common.h:11`.
- **Evidence**: SpatialGroupEnhance kw iter 1 (2026-08-21, Ascend950PR_9579 / CANN 9.1.0.B060) — two
  call sites (the `V_S` and `S_V` scalar/vector fences) failed the build; dropping the qualifier
  fixed both, green on the next iteration.
- **Avoid the reverse overcorrection** (MUSEAttention, 2026-08-23/24,
  CANN 9.1.0.B060 / `dav_3510`): removing `AscendC::` from every related
  helper caused a second compile failure. `GetSysWorkSpacePtr()` was global,
  but `SetSysWorkspaceForce()` and `GetUserWorkspace()` were declared in
  `AscendC`. Qualify each symbol from its own declaration; a neighboring API's
  namespace is not evidence. The retained compiler diagnostics name both
  declarations in `kernel_operator_common_impl.h` (lines 35 and 51).
  Source: MUSE intake evidence（A5Ops 绑定外部引用；本语料未收录目标 docs/evidence/kb_intake/muse_20260824.md）.
- **Cross-ref**: EC-13 (`AscendC::SyncFunc<>` does not exist — the SAME `FetchEventID` call site, the
  other way to get it wrong: EC-13 is an invented API, this is a real API mis-scoped). backend=ascendc.
