---
type: Runbook
title: "Manual ascend950 kernel registry install — 3-file merge into running CANN runtime"
description: "Manual ascend950 kernel registry install — 3-file merge into running CANN runtime"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-A3A5-15"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-A3A5-15 — Manual ascend950 kernel registry install — 3-file merge into running CANN runtime

## 适用场景

- 适用平台与范围：soc=Ascend950PR；cann=9.0.0；op_class=port_a3_to_a5；phase=Phase C install。
- 验证来源：卡内无 verified_on 行；来源为 workspace/top_k_top_p_sample knowledge_update.md F3（kw-1，2026-05-14），实测锚点见正文 Concrete anchor 段。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: workspace/top_k_top_p_sample/knowledge_update.md F3 (kw-1, 2026-05-14)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5; phase=Phase C install`

**Principle**: after `build.sh --pkg --ops=<op> --soc=ascend950` produces `.o`+`.json` artifacts in `build/binary/ascend950/bin/ascend950/<op>/`, registering them into the running CANN runtime requires THREE separate filesystem edits, NOT one. Missing any of the three leaves the kernel "built but not callable" — verify reports EZ1001 even though `find` shows the binary on disk. Many port_a3_to_a5 sessions misdiagnose "Phase C done" because step 1 alone (binary install) was performed without steps 2 and 3 (registry merge + per-op JSON copy).

**3-step install procedure**:

1. **Install per-binary `.o` + `.json` into kernel directory**:
   ```bash
   cp build/binary/ascend950/bin/ascend950/<op>/*.o   $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/<op>/
   cp build/binary/ascend950/bin/ascend950/<op>/*.json $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/<op>/
   ```

2. **Merge op entry into runtime `binary_info_config.json`** (single top-level key per op_type):
   ```python
   import json, shutil
   live = "$CANN/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_nn/binary_info_config.json"
   built = "build/binary/ascend950/bin/config/ascend950/binary_info_config.json"
   shutil.copy(live, f"{live}.bak")
   merged = {**json.load(open(live)), **json.load(open(built))}  # op_type unique → no schema conflict
   json.dump(merged, open(live, "w"), indent=2)
   ```

3. **Copy per-op registration JSON**:
   ```bash
   cp build/binary/ascend950/bin/config/ascend950/<op>.json \
      $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_nn/<op>.json
   ```
   (Follows the existing pattern for `apply_top_k_top_p_with_sorted.json` and other pre-shipped registry entries.)

**Concrete anchor (top_k_top_p_sample, 2026-05-14)**: Step 2 merged 251 → 252 keys; new key `TopKTopPSample` added cleanly (no schema conflict — op_type is unique). After all three steps the V1 kernel is callable from any aclnn entry that targets `TopKTopPSample`.

**Anti-pattern**: doing step 1 only and assuming the runtime auto-discovers binaries by directory scan. CANN's runtime indexes by `binary_info_config.json` keys — a binary on disk WITHOUT a registry entry is invisible at op-resolve time.

**Promotion gate**: needs ≥ 1 additional op confirmation that the same 3-step procedure works for another op_type without further filesystem edits. If a second op surfaces a 4th required step (e.g., `simplified_key.ini` re-generation, `opp_kernel_list.txt` append), this candidate should be revised before promotion.

**Cross-ref**:
- OL-158 (Phase C activation criteria — when to attempt build+install at all)
- CAND-A3A5-14 (variant-aliasing trap — install can succeed yet verify still FAILS due to routing)
- OL-132 (port strategy `regbaseCfg` vs flat `AddConfig` — determines the `op_def.cpp` shape that produced these artifacts)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-15，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
