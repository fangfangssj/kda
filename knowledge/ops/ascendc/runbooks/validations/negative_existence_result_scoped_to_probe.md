---
type: Runbook
title: "A negative existence result is scoped to the probe — a probe keyed on location expires when things move; re-probe by identity, compare counts not existence"
description: "A negative existence result is scoped to the probe — a probe keyed on location expires when things move; re-probe by identity, compare counts not existence"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-311"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-311 — OL-311: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all。
- 验证来源：repo-tooling (not chip-specific)。

**OL-311** — **A negative existence result is scoped to the PROBE, and a probe keyed on LOCATION expires when things move.** "Not found" means *this probe did not find it*, never *it does not exist* — the part that keeps biting is the TIME dimension: a probe keyed on a path / directory / index entry / catalog is only as valid as the layout it was written against, and after any reorganisation it silently reports **"moved" as "deleted"** while looking entirely correct, because it *was* correct when written. ⛔ **The failure manufactures PHANTOM WORK**: a false "this knowledge was lost" sends someone to re-derive something that exists (and the re-derived copy then competes with the original); a false "this API doesn't exist" sends a worker to hand-roll a decomposition that drifts from the reference. **Decision rule before reporting ANY absence**: (1) re-probe by **identity not location** (`git log --all --diff-filter=A -- "**/<name>"`, `find -name`, grep the index) — (2) compare **counts, not existence** ("old path 2 files, new tree 45" identifies supersession instantly; "old path missing" does not) — (3) only then is it absent, and say which probes ran. **Anchor (measured 2026-07-31)**: `src/skills/references/PATTERN_INDEX.md` + `patterns/` returned MISSING against the live tree, reading as knowledge loss; they had been reorganised to `target/{ascendc,triton}/patterns/` with **45** live pattern files against the stale copy's **2** — the true statement ("the stale copy is superseded") is the INVERSE of what the path check implied. The identity probe also worked in the other direction in the same sweep: two documents returned nothing from `--diff-filter=A`, which is what genuinely-never-committed looks like, and were preserved. **Sibling**: EC-38 is this failure's API-catalog instance (catalog miss read as "API not available" → manual `Exp+Reciprocal+Mul` → Pass A drift); OL-311 is the general form. ⭐ **Corollary — do not sort by SIZE when deciding what to discard**: the same sweep found a 38 GB backup holding 37 GB of gitignored regenerable scratch and ~20 MB of genuinely unique content, i.e. volume and irreproducibility **inversely correlated** (a directory is large *because* it is full of regenerable intermediates), so "clean the biggest first" sorts by the inverse of what matters. The criterion is **irreproducibility**, uncorrelated with size and often opposed to it.

- Former A5Ops ID: `OL-311`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all`
`verified_on: repo-tooling (not chip-specific)`
**Principle.** "Not found" means *this probe did not find it*, never *it does not exist*. That
much is standard. The part that keeps biting is the **time dimension**: a probe keyed on a
LOCATION — a file path, a directory, an index entry, a catalog — is only as valid as the layout
it was written against. After any reorganisation the same probe silently starts reporting
**"moved" as "deleted"**, and it was *correct when written*, so nothing about it looks stale.
⇒ The failure is directional and expensive in one specific way: it manufactures **phantom
work**. A false "this knowledge was lost" sends someone to re-derive something that already
exists, and the re-derived copy then competes with the original. A false "this API doesn't
exist" sends a worker to hand-roll a decomposition that drifts from the reference.
**Decision rule — applies to ANY absence claim before it is reported:**
1. Re-probe by **identity**, not by location: search for the content/symbol/entry ID across the
   whole tree (`git log --all --diff-filter=A -- "**/<name>"`, `find -name`, grep the index),
   not at the path you expect.
2. Compare **counts**, not existence. "The old path has 2 files, the new tree has 45" identifies
   a supersession instantly; "the old path is missing" does not.
3. Only after both fail is the thing absent — and say which probes were run.
**Concrete anchor (measured 2026-07-31, 私有沉淀仓).** A stale checkout held
`src/skills/references/PATTERN_INDEX.md` + `patterns/`. Checking those paths against the live
tree returned MISSING, which reads as knowledge loss. They had been **reorganised** to
`src/skills/references/target/{ascendc,triton}/patterns/` — live carried **45** pattern files
against the stale copy's **2**. The correct statement was "the stale copy is superseded", the
inverse of the one the path check implied. In the same sweep the identity probe did its job in
the other direction: two documents returned nothing from `git log --all --diff-filter=A`, which
is what genuinely-never-committed looks like, and they were preserved.
**Sibling already in the KB:** EC-38（A5Ops 绑定外部引用；本语料未收录目标） — an `ASCENDC_API_CATALOG.md` miss
read as "API not available on this chip", worker fell back to a manual `Exp + Reciprocal + Mul`
decomposition, Pass A drifted from the CANN reference. Same shape, different index: one catalog
answered "no" and was treated as the world answering "no". OL-311 is the general form; EC-38 is
its API-catalog instance.
**Corollary — do not sort by size when deciding what to discard.** In the same sweep a 38 GB
backup held **37 GB of gitignored op-gen scratch** and ~20 MB of genuinely unique content (an
uncommitted findings document, an FA oracle, a vendor baseline). Volume and irreproducibility
were **inversely correlated**, because a directory is large precisely when it is full of
regenerable intermediates. "Clean the biggest first" is therefore sorted by the *inverse* of the
thing that matters. The criterion is **irreproducibility**, which is uncorrelated with size and
often opposed to it.
