---
type: Runbook
title: "Missing `config/ascend950/<op>_binary.json` → 950 build SILENTLY SKIPPED [V351, build-system]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2"
tags: [task.debugging, topic.implementation.pipeline, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-31-missing-config-ascend950-op-binary-json-950-build.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["PB-31"]
---

# Missing `config/ascend950/<op>_binary.json` → 950 build SILENTLY SKIPPED [V351, build-system]

> 原 OKF v1 卡号：**PB-31**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Build appears to succeed (no error / warning), but the resulting op package does NOT include the A5 (ascend950) kernel binary. At runtime, aclnn or torch_npu re

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §369-372`

**Symptom**: Build appears to succeed (no error / warning), but the resulting op package does NOT include the A5 (ascend950) kernel binary. At runtime, aclnn or torch_npu reports "operator not supported on this device" or similar generic missing-op error.

**Root cause**: The CANN build system gates per-SoC kernel compilation on the presence of `op_host/config/<compute_unit>/<op_name>_binary.json`. The check is "if file exists, compile for this SoC; else skip without warning". For port_a3_to_a5, if the dev added `OpAICoreConfig` for ascend950 in `_def.cpp` AND modified CMakeLists.txt AND created `arch35/` source — but FORGOT to copy `config/ascend950/<op>_binary.json` — the 950 path is silently skipped and the bug surfaces only at runtime on the target device.

**Anti-pattern**:
```bash
ls op_host/config/
# ascend910b/  ascend910_93/    ← ✘ no ascend950/
```

**Fix**:
```bash
mkdir -p op_host/config/ascend950
cp op_host/config/ascend910b/<op>_binary.json       op_host/config/ascend950/
cp op_host/config/ascend910b/<op>_simplified_key.ini op_host/config/ascend950/
# Adjust _binary.json contents if A5 supports different dtypes (FP8/HiFloat8/etc.)
```

**Detection signature**:
```bash
# After kw declares done, verify:
test -f op_host/config/ascend950/${op}_binary.json &&
test -f op_host/config/ascend950/${op}_simplified_key.ini ||
  echo "BUG: config/ascend950/ incomplete — 950 build will be silently skipped"
```

**Evidence**:
- PR 103 codifies as Trap #3; explicitly described as "缺失则 950 编译被静默跳过（无报错），极易遗漏"
- Direct hardware reproduction: when build/ pipeline doesn't even reach compile-arch35 step

**Mitigation gate**: `aog-self-critic` post-worker pass MUST verify both files exist in `config/ascend950/`; reject finalize if missing. Additionally, `aog-prior-art-verify` Phase 3 (build candidate) should fail-fast if these files don't exist.

**Other instances (predicted)**: any future A3→A5 port. The 14 no-upstream ops in our scan list (cohort 1+2) will all need this check at finalize time.

<!-- 迁移自 porter kb/target/ascendc/（PB-31，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
