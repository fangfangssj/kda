---
type: Runbook
title: "`add_modules_sources` called twice for same op → silent target conflict in `generate_bin_scripts` [V351, build-system]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2"
tags: [task.debugging, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-29-add-modules-sources-called-twice-for-same-op-silen.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["PB-29", "add_modules_sources", "generate_bin_scripts"]
---

# `add_modules_sources` called twice for same op → silent target conflict in `generate_bin_scripts` [V351, build-system]

> 原 OKF v1 卡号：**PB-29**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - At link/binary-generation time, generate_bin_scripts reports duplicate target name or duplicate symbol for the same op. Build log may also show repeated entries

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §339-352`

**Symptom**: At link/binary-generation time, `generate_bin_scripts` reports duplicate target name or duplicate symbol for the same op. Build log may also show repeated entries in `COMPILED_OPS` / `COMPILED_OP_DIRS`.

**Root cause**: `add_modules_sources` appends the op name to global cache variables (`COMPILED_OPS`, `COMPILED_OP_DIRS`) via `set(... CACHE FORCE)`. Two calls = two appends = duplicate listing. Downstream `generate_bin_scripts` then tries to generate the same kernel binary target twice.

**Anti-pattern** (BAD):
```cmake
# A3 baseline already registers:
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude DEPENDENCIES ...)

# Then A5 port ADDS a second call (WRONG):
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT "ascend950" TILING_DIR "arch35")
```

**Fix** (consolidate to ONE call):
```cmake
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
    TILING_DIR   "default"    "default"      "arch35"
    DEPENDENCIES ...)
```

**Detection signature**:
```bash
# In CMakeLists.txt of an op being ported, count add_modules_sources calls for that op
grep -c "add_modules_sources.*OPTYPE\s*${op_name}\b" op_host/CMakeLists.txt
# > 1 → BUG
```

**Evidence**:
- PR 103 codifies this as Trap #1 in CMakeLists.txt 经验教训
- Reasoning is `CACHE FORCE` semantics in CMake — verifiable in build/CMakeCache.txt

**Mitigation gate**: `aog-self-critic` post-worker pass should grep CMakeLists.txt for duplicate `add_modules_sources` registrations of the same op name; reject if found.

<!-- 迁移自 porter kb/target/ascendc/（PB-29，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
