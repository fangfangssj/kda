# skills/ascend — 昇腾技能层

本目录的技能**只取自 [cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills) 的 Skill 层**（知识、模板、调试、测试、工具类技能），未引入其 Plugin 编排层（plugins-official 的多角色工作流）与 pypto / graph / model / runtime 家族——编排层与 KDA 的最小循环哲学冲突，后者由 KDA 的任务契约 + 证据机制承担。

- 来源快照：原有技能来自 `cannbot-skills @ c1ad585`（2026-09-12）；[`ascendc-code-review`](https://gitcode.com/cann/cannbot-skills/tree/591e5fd668decdf3b26545914ce98928c5fcb93d/ops/ascendc-code-review) 后补自上游 `591e5fd`（2026-09-29）。
- 格式：Agent Skills 开放标准（SKILL.md frontmatter），与 Claude Code / OpenCode / Cursor 等兼容
- 许可：CANN Open Software License v2.0（个人使用；再分发时请附带其 LICENSE）

## 已接入清单（25 个）

| 类别 | 技能 |
|---|---|
| 架构知识 | npu-arch |
| AscendC 知识 | ascendc-tiling-design, ascendc-api-best-practices, ascendc-performance-best-practices, ascendc-perf-optimize |
| 代码检视 | ascendc-code-review |
| Triton (triton_ascend) | triton-op-designer, triton-op-coding, triton-op-verifier, triton-precision-debug, triton-latency-optimizer, triton-simulator-optimizer |
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
