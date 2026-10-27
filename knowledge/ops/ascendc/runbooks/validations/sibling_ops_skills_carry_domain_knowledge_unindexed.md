---
type: Runbook
title: "Domain knowledge we re-derive already ships in the sibling ops/ cannbot skills, and our KB indexes none of it — the RegBase shelf's coverage boundary is VEC-side only"
description: "Domain knowledge we re-derive already ships in the sibling ops/ cannbot skills, and our KB indexes none of it — the RegBase shelf's coverage boundary is VEC-side only"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-322"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-322 — OL-322: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR (DAV_3510 / A5) + all; cann=9.0.0/9.1.0; bisheng=n/a; op_class=all (where to look before deriving)。
- 验证来源：measured against the shipped skill tree and against this KB, 2026-08-01。

**OL-322** — **Domain knowledge we re-derive already ships in the SIBLING `ops/` cannbot skills, and our KB indexes none of it** — so each agent pays the discovery cost privately and nothing in a run reports the redundancy. Measured 2026-08-01: this KB discusses `RegBase` in **74 files** and points at `ops/ascendc-regbase-best-practice/` **zero** times. That skill carries the API reference/whitelist/sync cards, a common-traps table, DAV_3510 (A5) working-vs-failed case signals, RegBase performance practices, and — highest leverage — a **704-row operator table mapping op → real CANN source path** (`ops-nn/foreach/…`, `ops-transformer/mc2/…`, incl. the `op_kernel/arch35/*_regbase.h` route), i.e. the index to the "existing CANN implementations" agents are told to consult and cannot find. ⚠️ **Measured coverage boundary, recorded so the next reader does not repeat the search**: it is the VEC/RegBase shelf — `IterateAll` / `MatmulImpl` / `Mmad` / `cube_util` / `occupancy` all hit **0 files**; `scalar` 7, `GetValue` 1. So it answers "my scalar GetValue loop dominates" and NOT "MatmulImpl::IterateAll dominates my AIC time" (→ OL-317, OL-206/OL-220). The boundary matters more than the pointer: without it a first miss reads as "the knowledge does not exist" rather than "wrong shelf" (OL-311).

- Former A5Ops ID: `OL-322`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR (DAV_3510 / A5) + all; cann=9.0.0/9.1.0; bisheng=n/a; op_class=all (where to look before deriving)`
`verified_on: measured against the shipped skill tree and against this KB, 2026-08-01`

**Principle.** A resource that exists but is *not indexed where people look* is, in outcome,
identical to a resource that does not exist. The failure is invisible from inside a session: an
agent that never knew of it produces perfectly reasonable work that silently re-derives a
solved problem, and nothing in the run reports a redundancy.

**Measured (2026-08-01).** This KB discusses `RegBase` across **74 files**. Its pointers to the
sibling `ops/` skills that carry a dedicated RegBase skill: **zero.**

```
ops/ascendc-regbase-best-practice/          <- a whole skill, referenced nowhere in our KB
  references/api/                            regbase_api_reference · whitelist · sync ·
                                             pipeline_and_buffer · datacopy_best_practices ·
                                             compute_api_membase_vs_regbase
  references/pitfalls/common_traps.md
  references/dev-experience/
      working_vs_failed_dav_3510_cases.md    A5 success/failure signals + fix priority
      regbase_performance_practices.md       §1 what to check first, §2 problem->direction table
  references/reference-ops/
      open_source_operator_table.md          ** 704 rows: op -> real CANN source path **
```

⭐ **The 704-row operator table is the highest-leverage item**, because it is the index to the
"existing CANN implementations" agents are repeatedly told to consult and repeatedly cannot
locate: it maps an op to its actual tree (`ops-nn/foreach/foreach_add_scalar/`,
`ops-transformer/mc2/all_gather_matmul/`, …) including the `op_kernel/arch35/*_regbase.h`
route. Before hand-deriving a kernel shape, look up the op class there and read the production
implementation.

⚠️ **Measured COVERAGE BOUNDARY — record it, so the next reader does not repeat the search.**
This skill is the VEC/RegBase side. It does **not** cover the cube path:

| probe | files hit |
|---|---|
| `IterateAll` / `MatmulImpl` / `Mmad` / `cube_util` / `occupancy` | **0** |
| `scalar` | 7 |
| `GetValue` | 1 |
| `blockDim` | 2 |

⇒ So it answers *"my scalar `GetValue`/`SetValue` loop dominates"* and does **not** answer
*"`MatmulImpl::IterateAll` dominates my AIC time"*. For the latter see OL-317 (occupancy must be
measured before a ceiling is declared) and OL-206 / OL-220 (hand-rolled cube on A5 and the
managed alternative).

**Why the boundary matters more than the pointer.** "This skill exists" invites a reader to
search it for everything; the first miss then reads as *"the knowledge does not exist"* rather
than *"wrong shelf"* — a negative existence result scoped to the probe (OL-311).

**Evidence / how this surfaced.** A peer found the skill while debugging RegBase compile errors
by hand and reported that it covers *exactly* the ground they had spent hours re-deriving,
having never once looked there. Independently, a convolution-backward worker brief was measured
at 28,709 chars containing **zero** occurrences of `P-P78`, `CAND-FA-CV-10` or `im2col` — the
same failure one layer in: the knowledge is present and indexed in *this* KB, and still is not
delivered. ⇒ Two distinct undelivered-knowledge layers, one across repos and one inside ours.

**Other instances (predicted).** Any sibling skill under `ops/` — the same argument applies to
every one our KB does not reference. Before deriving a domain technique, check whether a
sibling skill owns that domain, and record its coverage boundary here when you do.

**Cross-ref**: OL-311 (a negative existence result is scoped to the probe), OL-317 + DEBT-314
(the cube-side question this skill does not answer), DEBT-258 (`COMPENSATED-BY-CRITIC` — the
in-repo version: knowledge that reaches the worker only when a critic happens to inject it).
backend=all.
