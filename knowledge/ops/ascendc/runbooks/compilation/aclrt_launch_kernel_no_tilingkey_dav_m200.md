---
type: Runbook
title: "`ACLRT_LAUNCH_KERNEL` arg count is arch-dependent — on dav-m200 (`__CCE_AICORE__=200`) the launcher takes no tilingKey param"
description: "`ACLRT_LAUNCH_KERNEL` arg count is arch-dependent — on dav-m200 (`__CCE_AICORE__=200`) the launcher takes no tilingKey param"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["ACLRT_LAUNCH_KERNEL"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-203 — EC-203: A5Ops additive extension

> 关联：同一 EC-203 主题已由既有卡 [ec_203_aclrt_launch_kernel_arg_count_is_arch_dependent.md](ec_203_aclrt_launch_kernel_arg_count_is_arch_dependent.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=CUBE_MIX——任何 pybind 自定义 launch kernel。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（prompt_flash_attention iter 2，2026-08-11）。

**EC-203** — **`ACLRT_LAUNCH_KERNEL` arg count is arch-dependent: on dav-m200 (`__CCE_AICORE__=200`) the launcher takes NO tilingKey param — passing one is a "too many arguments" compile error** (the arch35/V351 form emits a 9-param launcher WITH tilingKey; dav-m200 is 8-param). Tiling is NOT passed via the macro on this arch — it ships in the tiling-GM pointer arg + `GET_TILING_DATA` (OL-268). Fix: omit tilingKey on dav-m200. [Worker mislabeled the arch "arch22"; it is `==200`/dav-m200 per OL-173.] Evidence: prompt_flash_attention iter 2 2026-08-11 Ascend310P3/dav-m200 CANN 9.0.0. Cross-ref [OL-268](ol_268_tilingdata_two_registration_styles_mismatch.md), OL-546（A5Ops 绑定外部引用；本语料未收录目标）, [OL-173](ol_173_cce_aicore_macro_values_per_ascend_arch_when_readi.md). backend=ascendc.

- Former A5Ops ID: `EC-203`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=CUBE_MIX — any pybind custom-launch kernel`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0`
`unverified_on: soc=Ascend950PR/V351 — the 9-param (with-tilingKey) launcher form is reported for arch35/V351 but not independently re-verified here; only the 8-param form on dav-m200 is confirmed`

- **Symptom (compile)**: `ACLRT_LAUNCH_KERNEL(op)(...)` with a trailing tilingKey argument fails to compile — "too many arguments to function/macro" — on dav-m200 (`__CCE_AICORE__ == 200`). The same call shape reportedly compiles on arch35/V351, where the macro expansion emits a 9-param launcher (blockDim, stream, tilingKey, …pointers) vs the 8-param launcher here.
- **Root cause / fix**: the `ACLRT_LAUNCH_KERNEL` macro expands to a per-arch launcher whose parameter list differs. On dav-m200 OMIT the tilingKey argument. Tiling data is NOT passed through the launch macro on this arch — it is shipped via a separate tiling-GM pointer argument and unpacked by the kernel with `GET_TILING_DATA` (see OL-268 for the host→device tiling serialization).
- **Concrete anchor** (内部路径（不公开）pybind11.cpp:260):
  ```
  // dav-m200: 8-param — NO tilingKey. Tiling travels in the tiling-GM pointer arg.
  ACLRT_LAUNCH_KERNEL(prompt_flash_attention)(blockDim, stream, xq_ptr, xkv_ptr, ...);
  // do NOT append a tilingKey arg (the arch35/V351 form) — compile error on ==200.
  ```
- **Detection**: hard compile failure naming the macro expansion with an arg-count mismatch; no runtime symptom.
- **Evidence**: prompt_flash_attention iter 2 (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0) — failures-ledger `iter 2 | compile | novel | 内部路径（不公开）pybind11.cpp:263 ACLRT_LAUNCH_KERNEL tilingKey arg not supported on 310p → removed (arch35-only)`. [Worker labeled the arch "arch22"; per OL-173 the chip is `__CCE_AICORE__=200`/dav-m200 — "arch22" is a mislabel.] Source candidate: `CAND-310P-ACLRT-NO-TILINGKEY`.
- **Cross-ref**: OL-268 (how tiling IS passed on this launch path — via tiling-GM + `SaveToBuffer`, not the macro), OL-546 (the async-lifetime rule for this same macro on dav-m200), OL-173 (`__CCE_AICORE__` value table — 200=dav-m200, correcting the worker's "arch22" label). backend=ascendc.
