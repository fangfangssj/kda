---
type: Runbook
title: "AscendC documentation is JS-rendered — use dev-browser plugin to access"
description: "AscendC documentation is JS-rendered — use dev-browser plugin to access"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-22-ascendc-documentation-is-js-rendered-use-dev-brows.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-22"]
---

# AscendC documentation is JS-rendered — use dev-browser plugin to access

> 原 OKF v1 卡号：**OL-22**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - need to look up AscendC API, SIMD/SIMT programming guide, type support

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: environment
- **Loaded by**: all skills
- **Trigger**: need to look up AscendC API, SIMD/SIMT programming guide, type support
- **Lesson**: AscendC official docs at hiascend.com are JS SPA — direct URL fetch returns empty or wrong page. Use dev-browser plugin with Playwright. Key documentation URLs:
  - API reference (CANN 9.0 beta2): `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlasascendc_api_07_10293.html`
  - Type conversion API (reg_convert.h): see CANN source `/usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/include/c_api/reg_compute/reg_convert.h`
  - CANN source code: `~/workspace/cann` (local copy)
- **When to check docs**: Before assuming a type cast or API doesn't exist, ALWAYS check reg_convert.h and the CANN source for alternative APIs.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-22（category=environment，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
