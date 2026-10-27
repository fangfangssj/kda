# skills/ascend — 昇腾技能层

本目录包含 19 个来自 [cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills) Skill 层的昇腾技能，以及 2 个 KDA 本地知识查询 Skill。未引入上游 Plugin 编排层（plugins-official 的多角色工作流）与 pypto / graph / model / runtime 家族。

- 来源快照：原有技能来自 `cannbot-skills @ c1ad585`（2026-09-12）；[`ascendc-code-review`](https://gitcode.com/cann/cannbot-skills/tree/591e5fd668decdf3b26545914ce98928c5fcb93d/ops/ascendc-code-review) 后补自上游 `591e5fd`（2026-09-29）。
- 格式：Agent Skills 开放标准（SKILL.md frontmatter），与 Claude Code / OpenCode / Cursor 等兼容
- 许可：CANN Open Software License v2.0（个人使用；再分发时请附带其 LICENSE）

## 上游 cannbot 技能（19 个）

| 类别 | 技能 |
|---|---|
| 架构知识 | npu-arch |
| AscendC 知识 | ascendc-tiling-design, ascendc-api-best-practices, ascendc-performance-best-practices, ascendc-perf-optimize |
| 代码检视 | ascendc-code-review |
| 工程模板 | ascendc-direct-invoke-template |
| 测试 | ascendc-st-design, ascendc-ut-develop, ops-precision-standard |
| 调试 | ascendc-precision-debug, ascendc-crash-debug, ascendc-runtime-debug, ascendc-sync-audit |
| 环境 | cann-env-setup, ascendc-env-check |
| 性能工具 | ops-profiling, msnpureport-toolkit, asys-toolkit |

上游还有更多技能（TileLang 家族、CATLASS、pypto、graph/model 推理优化等）未纳入；需要时从 cannbot-skills 仓库按同样方式复制整个技能目录到这里即可。

## 安装（symlink 到 Claude Code 技能目录）

```bash
for d in "$(pwd)"/skills/ascend/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$d" ~/.claude/skills/"$(basename "$d")"
done
```

技能只按需触发（SKILL.md 的 description 决定触发时机），全量 symlink 不会污染无关任务的会话。

## KDA 本地知识查询技能

| 技能 | 用途 |
|---|---|
| `knowledge-query` | 查询硬件、数据搬运、tiling、buffer、流水线、同步、算子案例与 Runbook。 |
| `ascendc-api-knowledge-query` | 查询 Ascend C API 定义、参数、重载、对齐与 SoC/CANN 适用条件。 |

两者读取仓内 `knowledge/ops/ascendc/` Markdown，只使用 Agent 原生文件搜索和阅读工具；不需要 Python 查询脚本、向量索引或服务。
