---
type: Runbook
title: "Deploy stage authority root must pre-exist on the target host"
description: "Deploy stage authority root must pre-exist on the target host"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-DEPLOY-STAGE-AUTHORITY-ROOT-PRECREATE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-DEPLOY-STAGE-AUTHORITY-ROOT-PRECREATE — Deploy stage authority root must pre-exist on the target host

## 适用场景

- 适用平台与范围：所有使用 deploy_to_npu.sh stage guard 的目标。
- 验证来源：13_Cat M-I run-1（.188 容器 deploy）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: all targets using deploy_to_npu.sh stage guards`
`verified_on: 13_Cat M-I run-1 (.188 container deploy)`

`deploy_to_npu.sh`'s stage object guard refuses to create the configured `A5_DEPLOY_STAGE_HOST` root (by design — it is the safety anchor). A run-scoped root that no setup step provisioned fails every deploy with `stage authority/object guard failed` before any upload. One `mkdir -p` of the exact declared root unblocks; the guard then validates it normally.

- **Runbook**: provision the exact declared stage root on the target host BEFORE the first deploy; the guard then passes. Do NOT bypass the guard (P9). backend=ascendc.

## 增量验证（2026-09-24, 4-34 Mamba2Bwd r2/r3 lanes @ .122）

- 第二次独立复现：两条独立 lane 在同一 env 形状下 mkdir 前 exit 10（infra 类，两次），`mkdir -p <root>` 后 deploy 正常进入 build。
- guard 形态细化为 `test -d $authority && ! -L $authority`：只校验 root 本身存在且非 symlink，per-run 子目录由 guard 自建。
- 边界对照：benchmark_root 类 truth 路径由相应流程自建，**只有 deploy stage root 需要预建**——两类路径不要混为一谈。
- 处置顺序：先 `ls -ld $A5_DEPLOY_STAGE_HOST` 探测再 mkdir，不要先升级为 harness/网络诊断。
