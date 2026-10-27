---
type: Runbook
title: "CMake `COMPUTE_UNIT` ≠ `TILING_DIR` list length → `find_value_by_key` FATAL_ERROR [V351, build-system]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2"
tags: [task.debugging, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-30-cmake-compute-unit-tiling-dir-list-length-find-val.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["PB-30", "COMPUTE_UNIT", "TILING_DIR", "find_value_by_key"]
---

# CMake `COMPUTE_UNIT` ≠ `TILING_DIR` list length → `find_value_by_key` FATAL_ERROR [V351, build-system]

> 原 OKF v1 卡号：**PB-30**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - At CMake configure time, find_value_by_key() (internal CANN cmake helper) calls message(FATAL_ERROR ...) complaining about COMPUTE_UNIT and TILING_DIR list-leng

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §354-367`

**Symptom**: At CMake configure time, `find_value_by_key()` (internal CANN cmake helper) calls `message(FATAL_ERROR ...)` complaining about COMPUTE_UNIT and TILING_DIR list-length mismatch.

**Root cause**: CMake silently drops empty strings (`""`) from list arguments. When the author intends `TILING_DIR = ["", "", "arch35"]` to mean "A3/910b/910_93 use root, A5 uses arch35/", CMake collapses it to `["arch35"]` (1 element) while `COMPUTE_UNIT` stays at 3 elements → length mismatch.

**Anti-pattern** (BAD — empty string eaten):
```cmake
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   ""           ""              "arch35"
# Actual after parse: COMPUTE_UNIT=3 items, TILING_DIR=1 item → FATAL_ERROR
```

**Fix** (use named subdirs even when not strictly needed):
```cmake
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   "default"    "default"      "arch35"
# Then create op_host/default/ alongside op_host/arch35/
```

**Detection signature**:
```bash
# Count list items in each — must match
awk '/COMPUTE_UNIT/{n=0; for(i=2;i<=NF;i++) if($i!~/^$/) n++; print "cu",n}
     /TILING_DIR/{n=0; for(i=2;i<=NF;i++) if($i!~/^$/) n++; print "td",n}' \
     op_host/CMakeLists.txt
```

**Evidence**:
- PR 103 codifies as Trap #2; ties directly to CMake list-arg semantics
- Reproducible: `cmake -DLIST_VAR="" ""` produces 0-length list, not 2-length

**Mitigation gate**: `aog-self-critic` post-worker — when adding an `ascend950` element to COMPUTE_UNIT, REQUIRE that TILING_DIR has a corresponding non-empty entry; if mismatched, suggest the named-subdir fix.

<!-- 迁移自 porter kb/target/ascendc/（PB-30，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
