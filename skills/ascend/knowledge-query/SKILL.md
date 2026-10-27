---
name: knowledge-query
description: Search and read the in-repository Ascend C knowledge base for hardware constraints, memory movement, alignment, tiling, buffers, pipelines, synchronization, operator examples, and known issues. Use when Ascend C technical facts are needed during design, implementation, debugging, or optimization.
---

# Ascend C Knowledge Query

Use the local Markdown knowledge under `knowledge/ops/ascendc/` as the first reference for Ascend C engineering questions. The corpus is already part of this repository and is available offline.

## Search and read

1. State the concrete question and target Ascend SoC/NpuArch and CANN version when known. Do not infer the target from the machine running the agent.
2. Start from `knowledge/ops/ascendc/index.md`, then choose the relevant section: `guides/`, `concepts/`, `examples/`, `operators/`, `optimizations/`, or `runbooks/`. For API definitions and overloads, use `ascendc-api-knowledge-query`.
3. Use the agent's native repository search and file-reading tools. Search filenames and exact technical terms first; then refine with constraints such as alignment, UB, DMA, event, pipeline, SoC, or error code. With `rg`, for example:

   ```bash
   rg --files knowledge/ops/ascendc
   rg -n -i -g '*.md' 'tiling|buffer|pipeline|synchronization' knowledge/ops/ascendc
   ```

4. Read the complete relevant Markdown cards, including frontmatter, tables, code examples, notes, and applicability conditions. Follow links within the repository only when they add evidence needed for the question. Some upstream cards retain links to source snapshots excluded from this integration; do not fetch or require those snapshots. An index entry or search hit is navigation, not technical evidence.
5. Check SoC/NpuArch and CANN applicability in the card. Keep platform- or version-specific statements scoped to the stated target. Separate documented facts from engineering inference and from anything actually tested.
6. Answer with links or repository paths to the cards read and state material gaps or version limits. A missing card does not prove an API or behavior is unsupported.

## Search by topic

| Question | Likely starting point |
|---|---|
| Hardware limits, core model, memory hierarchy | `concepts/`, `guides/programming_guide/`, `guides/memory/` |
| Data movement, alignment, padding, layouts | `guides/data_movement/`, `guides/layouts/`, `apis/` |
| Tiling, local buffers, double buffering | `guides/tiling/`, `guides/memory/`, `guides/performance/` |
| Pipelines and synchronization | `guides/synchronization/`, `guides/programming_guide/`, `apis/` |
| Operator implementation patterns | `examples/`, `operators/`, `optimizations/` |
| Compile, precision, runtime, or performance issues | `runbooks/` and the matching `guides/` section |

## Boundaries

- Use only the in-repository Markdown corpus for this workflow; do not require another checkout, network access, an index build, a Python query script, a query service, or new runtime dependencies.
- If a conclusion depends on an excluded source snapshot or another unavailable file, say so and distinguish the knowledge card's embedded example from source code verified in this task.
- Do not create or modify knowledge cards as part of answering a query.
- Do not turn retrieved guidance into an automatic optimization choice. Explain options and evidence; let the task contract and validation determine the implementation decision.
- When the corpus is silent or conflicting, report that limitation instead of filling it with an unsupported claim.
