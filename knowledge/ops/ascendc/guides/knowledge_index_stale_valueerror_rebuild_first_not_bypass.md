---
type: Guide
title: "knowledge-query 报 ValueError『card set changed after index build』= 派生索引陈旧：build+verify 分钟级原地重建即可，语料没坏；verify 门全拒是设计不是故障"
description: "knowledge-query 每次查询（正控负控同错）返回 ValueError『knowledge card set changed after index build; rebuild the index』或『cannot prove its corpus tip』时，坏的是 gitignored 派生索引（artifacts/indexes/knowledge.sqlite3），不是语料：pull/切分支/合新卡/搬迁 local root 后索引 meta（commit、卡数、卡哈希）与磁盘不再一致，verify 门（retrieval/index.py）按设计 typed-stop 全拒，宁可零投递也不静默给陈旧答案。修复是对自有克隆或本地库原地跑 knowledge_index.py build + verify（验收 meta.repository_state.commit==HEAD、provenance 合法、card_count 与该 tip 应有数一致；非 git 的 local-kb 用同一工具，no_git_directory 是合法状态），不得绕过门、从已知陈旧索引降级消费，也不需要重新克隆——重建不动卡片字节，本地卡住在语料克隆之外，天然免疫。"
tags:
- task.lookup
- topic.implementation.debugging
status: draft
sources: []
created_at: '2026-09-26T00:00:00Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- a2
- a3
- "950"
aliases:
- OL-610
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# knowledge-query ValueError "card set changed after index build" = stale derived index: rebuild (build+verify), the corpus is not broken

**Symptom (total refusal, never degraded recall):** every `knowledge_query.py search` call — positive control AND negative control returning the same error — fails with `{"error": {"type": "ValueError", "message": "knowledge card set changed after index build; rebuild the index"}}` (or the sibling provenance stop "knowledge index cannot prove its corpus tip …"). The discriminating signature is that it hits **every** query: the instrument is dead, the card set is not empty.

**Root cause:** the retrieval index `artifacts/indexes/knowledge.sqlite3` is a **gitignored, per-clone derived artifact**. `git pull`, a branch switch, or any merged/new card advances the checkout past the index's recorded meta (commit, card count). The verify gate (`retrieval/index.py`) then typed-stops **by design**: an index that cannot prove its corpus tip must refuse rather than silently serve stale or zero-hit answers. The error is the gate working, not the knowledge base being broken.

**Fix (minutes, per-clone, never inside another party's active worktree):**

```bash
cd <your own corpus clone>
git fetch <remote> && git checkout -B main <remote>/main   # substitute your own remote name (e.g. origin); clean consumer checkout only: if dirty, stop and report, do not reset
python3 .agents/skills/knowledge-query/scripts/knowledge_index.py --knowledge-root . build
python3 .agents/skills/knowledge-query/scripts/knowledge_index.py --knowledge-root . verify --json   # --json prints the index meta block incl. repository_state
```

Acceptance: `meta.repository_state.commit == HEAD` with `provenance: "git"` (read it from the `verify --json` output — plain `verify` prints only `ok cards=N`), and `card_count` equals the expected count for that tip (以你的 tip 实际卡数为准). After the rebuild, newly merged cards are recallable **immediately** — there is no other review or wait step. During the rebuild, consumers stay typed-stopped: no degradation mode, no bypass.

**Discrimination traps (each one fleet-verified, same incident family):**

1. **Disk `.md` count ≠ `card_count`.** Non-card Markdown (README/templates/docs) is not indexed. Comparing `find -name '*.md' | wc -l` against the index's `card_count` manufactures phantom drift (observed: 8101 vs 7439 on a healthy checkout). Compare index meta against the checkout HEAD instead.
2. **`find -maxdepth` false-zero.** The query script lives at `.agents/skills/knowledge-query/scripts/knowledge_query.py` and the index under `artifacts/indexes/` — shallow `find -maxdepth 3/4` reports "no tooling / no index" that is not true.
3. **`git ls-files` cannot see the index** — it is gitignored; absence from `ls-files` proves nothing.
4. **Single-source vs dual-source is a tooling difference, not a write failure — and "the dual-source tool" is two different scripts with opposite behavior.** The corpus's own `knowledge_query.py` searches only the governed corpus; user-local cards never appear in its output, so local cards "missing" there proves nothing about whether a write happened. The AOG wrapper `a5ops_knowledge_query.py` (a5_ops repo) adds the `[local]` section and turns any local-layer failure into a **visible typed stop** — it never rebuilds an index and never swallows an error. The *other* dual-source path, `okf_local_layer.py query` (`merged_query`), is the opposite: it silently rebuilds the local index before querying (an unchecked `subprocess.run`) and silently drops an underlying `{"error": …}` payload into `hits: []`. Know which script your evidence came from: a typed stop is only possible from the wrapper; an empty result from `merged_query` may be a hidden error, not an honest no-match.
5. **A persistent local-layer "card changed after index build" stop means the silent rebuild is failing — surface it, don't bypass it.** The governed local `write` and `okf_local_layer.py query` both rebuild the local index automatically; if the typed stop persists, the rebuild (`subprocess.run` without a return-code check) is swallowing its own failure. Run the build manually with visible output — `python3 <corpus>/.agents/skills/knowledge-query/scripts/knowledge_index.py --knowledge-root <local store> build` — and read the exact failing line (usually a card schema/lint issue). Note what `okf_local_layer.py stale` does NOT prove: it compares **corpus** card hashes against local `supersedes:` markers (has the corpus target moved upstream), and performs **no** journal↔disk byte-drift check on local cards — a hand-edited local card passes `stale` silently. The gate error itself names the drifting card (`… changed after index build: <relative path>`); journal byte-drift is checked only at `retire` time (`check_card_version`), and no CLI audit subcommand exists. Never hand-edit bytes to appease the gate.
6. **Rebuild ownership is per-clone.** Each agent rebuilds only their own clone's index; never `checkout -B` or rebuild inside another party's active worktree — if a shared clone is on someone else's feature branch, create your own clone instead.
7. **Positive/negative controls must be capable of disagreeing.** Before reading anything into a control pair, confirm the two commands can produce different results: a positive and a negative control that both fail with the same `platform is required` ValueError (wrong scope for the flags) are non-discriminating — "the negative control errored too" then proves nothing about the index. Pick a scope/flag combination whose failure modes differ (e.g. a `--technology platforms --type concepts` query vs a gibberish query), and re-run the pair whenever the invocation changes.
8. **Moving a local root invalidates its index — rebuild after any relocation.** Writing cards to the XDG default local root and then `tar`-moving the store to `<launch>/.cannbot/local-kb` leaves the index recording the pre-move identity: the card bytes themselves are unchanged by the move, but index↔disk identity no longer matches, and every local query typed-stops. The repair is the same one-shot build against the new root; the write/query-side auto-rebuild does NOT reliably recover an existing-but-stale index (its failure is swallowed by an unchecked `subprocess.run` — see trap 5), so run the build manually once and read its output.

**Boundary:** never bypass the verify gate, never serve from a known-stale index, and never treat a total-refusal ValueError as "zero hits". The only sanctioned recovery is the fresh build+verify above.

**Team Wiki cross-reference**: `projects/a5ops-okf-v4/KB_QUERY_INDEX_STALE.md` (agent-team-wiki) — the a5ops-consumer-side repair procedure and the development-practice clauses that cite this card.
