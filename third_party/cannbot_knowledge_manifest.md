# cann-bot knowledge integration manifest

## Source

- Repository: [cann/cannbot-knowledge](https://gitcode.com/cann/cannbot-knowledge)
- Pinned commit: `a76850a6457c3ac67bee4a897ed4902f1e80f925`
- Commit URL: <https://gitcode.com/cann/cannbot-knowledge/commit/a76850a6457c3ac67bee4a897ed4902f1e80f925>
- The supplied local snapshot was compared with this commit before import: all 7,280 files under `knowledge/ops/ascendc/` matched, with no missing, extra, or changed files. The source `LICENSE` also matches the included license copy.

## Imported content

- `knowledge/ops/ascendc/`: copied from the pinned commit with its source directory structure and Markdown indexes intact. It includes the API, guide, concept, example, operator, optimization, and runbook content in that subtree.
- `.agents/skills/knowledge-query/SKILL.md` and `.agents/skills/ascendc-api-knowledge-query/SKILL.md` informed the two KDA Skills under `skills/ascend/`. Their query workflow was adapted to native repository search and file reading; upstream query scripts, resource resolver scripts, and their runtime requirements were not imported.
- `LICENSE`: retained as `third_party/licenses/cannbot-knowledge-LICENSE`.
- The two adapted Query Skills retain the source Skills' CANN knowledge-query workflow under the same license notice.

## Excluded content

No other upstream knowledge domains, DSLs, model-optimization material, knowledge-production tooling, Python query/runtime code, vector indexes, query services, governance systems, or build components are included.

Operator cards may retain relative links to upstream fixed-source snapshots such as `cann-ops-raw`; those source snapshots are outside the requested import scope and are not required to search or read the embedded Markdown knowledge. The Skills instruct agents to treat them as unavailable and not to fetch or require another checkout.

## License

The imported upstream knowledge and license terms are covered by the CANN Open Software License Agreement Version 2.0. Keep the included license and this source record with the imported content when redistributing it.
