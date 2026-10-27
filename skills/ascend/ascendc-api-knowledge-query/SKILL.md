---
name: ascendc-api-knowledge-query
description: Find and verify Ascend C API definitions, overloads, parameters, types, alignment and SoC/CANN applicability in the local knowledge base. Use for API-specific questions; use knowledge-query for broader implementation or troubleshooting questions.
---

# Ascend C API Knowledge Query

Use this skill for questions about Ascend C APIs: declaration, purpose, parameters, overloads, data types, alignment constraints, SoC/CANN availability, examples, and caveats. The knowledge corpus is local at `knowledge/ops/ascendc/`; no external knowledge checkout is required.

## Workflow

1. Confirm the exact API name and the target Ascend SoC/NpuArch and CANN version when available. If a name is ambiguous, identify the API family before drawing a conclusion.
2. Begin at `knowledge/ops/ascendc/apis/index.md`. Search the API tree for the exact symbol, aliases, and relevant parameter names using native repository search. For example:

   ```bash
   rg --files knowledge/ops/ascendc/apis
   rg -n -i -g '*.md' 'DataCopyPad|srcStride|blockLen' knowledge/ops/ascendc/apis
   ```

3. Read the full matching API card. Check its declaration, overload/template parameters, parameter table, return value, restrictions, examples, and frontmatter or body notes for platform and version applicability. Do not treat an API name mention or index summary as its definition.
4. If the card links to a guide, example, operator, or runbook relevant to the question, open that local file and read the needed content. Some source-code links target snapshots excluded from this integration; do not fetch or require another checkout. Keep API contract, usage guidance, and observed troubleshooting experience distinct.
5. Compare overloads and SoC/CANN constraints explicitly. Do not merge signatures, infer availability from a similar API, or generalize one version's behavior to all Ascend products.
6. Answer with the API card path and any supporting local cards read. If no matching definition is present, say the local corpus did not contain one; do not claim the API does not exist.

## Boundaries

- Use the agent's native file search, text search, and file-reading tools. Do not run or add Python query scripts, build a vector index, start a query service, install dependencies, or access another checkout.
- Do not scan local CANN SDK headers unless the user explicitly asks for verification against an installed SDK. The knowledge-card answer must remain clear about what the corpus itself supports.
- If a conclusion requires an excluded source file or SDK header, report that evidence gap; do not present the card's embedded snippet as a source checkout verified in this task.
- This skill is for Ascend C APIs, not ACLNN API lookup or a complete operator implementation review. Use `knowledge-query` for broader hardware, tiling, pipeline, operator-example, or troubleshooting questions.
