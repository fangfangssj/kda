---
type: Runbook
title: "算子开发三条铁律——不接受单核 / 不接受标量搬运与标量计算 / CV 融合算子不接受纯 AIV（所有 op 类、所有 target，最高优先级）"
description: "CV/AIV 算子三条铁律（权威源在 shared/ALWAYS_LOADED_RULES.md §0）：不接受单核、主数据通路必须 block DataCopy+VEC、cube 类必须真走 cube（dav-m200 raw-launch 走 bare-Mmad 配方），精度 PASS 不是豁免理由"
tags: [task.optimization, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-626"]
---

# 算子开发三条铁律——不接受单核 / 不接受标量搬运与标量计算 / CV 融合算子不接受纯 AIV（所有 op 类、所有 target，最高优先级）

## 优化点

`applies_to: all backends, all targets, all op_class; benchmark / port_a3 / vendor lanes alike`
`verified_on: 2026-09-08 — level3 310p 批跑 6 个 matmul/conv kernel 全数纯 AIV 交付（fp16 大 shape 最差 0.011×）触发 owner 指令；执行链＝ALWAYS_LOADED_RULES §0 + benchmark-lane ARCHITECTURAL_HACK gate + VECTOR_REGIME lane 扩展（commit f1e665cbe/221808971/559621841）`
`unverified_on: 无（规则层，非平台层）`
**规则全文见 `shared/ALWAYS_LOADED_RULES.md` §0（单一权威源，本条为 KB_INDEX 可导航的 canonical 锚）**。三条都不接受"精度 PASS"作为豁免理由；违反 = `ARCHITECTURAL_HACK`，与 OL-36 委托作弊同级。①单核仅限无可分形 / A/B 实测更快（regime 表声明，OL-397 同款）；②主数据通路必须 block DataCopy + VEC，标量仅限 sub-64 residue tail 与索引类 load；③cube 类（reference 主计算是矩阵乘）主计算必须走 cube——dav-m200 上按 launch 方式分流：raw-launch lane 走 PB-71 bare-Mmad 配方（highlevel `matmul::Matmul<>` 的 TU 在 raw ACLRT launch 下必段错误，5 轮探针 2026-09-08；`MatmulImpl<>` 在 2002 被 arch gate）；op-host executor lane 走 highlevel API + TSCM/NZ L1 staging（joint_attention / vision_attention / conv_transpose_310p_deliverable 三范例）。豁免通道：`.workflow_exception_arch_class` waiver 或 escalate owner，禁止静默降级。
**Cross-ref**: [`OL-127`](../compilation/ol_127_cann_api_surface_shrinks_across_versions_check_api.md)（单核禁终态 precursor）、OL-397（VECTOR 域 regime 表）、[`OL-188`](../compilation/ol_188_op_family_architecture_requirement_classification.md)（port_a3 域 ARCHITECTURAL_HACK）、`PB-71`（310B 卡，本批未迁移）（bare-Mmad 配方 + 支持矩阵）、（RESOLVED 2026-09-08——lane-conditional 路由）、CAND-DAV-M200-MATMULIMPL-MULTIBLOCK-BROKEN（GM-ND 多块损坏 + enVecND2NZ）。backend=ascendc。

`applies_to: any port/analysis that reads upstream 310P support state (op selection, portability judgement, provenance claims)`
`verified_on: upstream-documented — Ascend-SACT op-ascend-migration skill (gitcode, commit bf37a53c, Apache-2.0), from ops-transformer PR87 + multi-op 310P samples`
`unverified_on: soc=n/a (structure knowledge, not device behavior)`

**Principle — Host registration layers (pick the minimal sufficient level, do NOT default to the full PR87 config)**:
- **L1**: `AddConfig("ascend310p")` alone — same dtype/format/kernel, no narrowing. First choice for simple ops (e.g. swin_attention_score_quant).
- **L2**: `OpAICoreConfig` + target-specific flags only (`enableVectorCore.flag`, `DynamicCompileStaticFlag`, `ExtendCfgInfo`, `aclnnSupport.value`...) — no dtype relisting. Add flags ONLY with repo precedent or verification evidence, never "to look more complete".
- **L3**: full config — dtype/format narrowing + `opFile.value` routing; `Format`/`UnknownShapeFormat`/`DataType` list lengths strictly consistent; ND format; keep dynamic/aclnn flags; 310P default: NO BF16/quant/arch35-only capabilities unless kernel implements and verifies them. `opFile.value` must point to a 310P-compilable kernel, never arch35/APT/910B-only source.

**Structure facts**:
- **Config-only ops exist**: `op_host/config/ascend310p` present without a local explicit `AddConfig("ascend310p")` (e.g. `mc2/3rd/quant_batch_matmul_v3`) — registration comes from a shared/3rd OpDef; trace it, do NOT judge "missing registration".
- **`*_binary.json` / `*_simplified_key.ini` / `*runtime_kb.json` can NOT be generated from Host dtype lists** — inherit from same-op same-SoC precedent or tuning evidence; report provenance (同算子同SoC先例 / 同族可验证配置 / 调优产物 / "缺少配置").
- **Build side**: `SUPPORT_COMPUTE_UNIT`/`SUPPORT_COMPUTE_UNIT_SHORT`/`ASCEND_SOC_UNITS`/`SOC_ARRAY` allowlists, `cmake/scripts/get_soc_version.py` exclusion routes (arch22/arch35/910B dirs), `tests/test_soc_config.yaml` grouping — all must include `ascend310p`; absence of evidence = no build-support claim.
- **BF16 is NOT blanket-removed on 310P**: MatMul/GMM 310P may keep BF16 — decide per-op from Host 310P config + kernel branch, not from the MoE/PR87 default narrowing.
- **Kernel guard semantics over spelling**: `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200` is the clean form, but some code uses bare `#if __CCE_AICORE__ == 200` or routes 220/`__NPU_ARCH__` into the native branch and 310P into `#else` — judge "does 310P land in the right branch", not macro spelling.

**Scan pattern catalog** (from `scan_ascendc_migration.py`, commit bf37a53c — the pattern families to rg when auditing a port): `AddConfig("ascend310p")`/`AddConfig`-any, `OpAICoreConfig`, target-config flags (`DynamicCompileStaticFlag|DynamicFormatFlag|DynamicRankSupportFlag|DynamicShapeSupportFlag|NeedCheckSupportFlag|PrecisionReduceFlag|ExtendCfgInfo|enableVectorCore.flag|jitCompile.flag|aclnnSupport.value|prebuildPattern.value|coreType.value`), `opFile.value` (4 extraction syntaxes incl. JSON/YAML), `SUPPORT_COMPUTE_UNIT|ASCEND_SOC_UNITS|SOC_ARRAY`, `test_soc_config|GetSocModules|get_related_soc|UTExcludes`, `_binary.json|_simplified_key.ini|runtime_kb.json|binary_mode_flag`, `simplified_key_mode`, `tilingKey`, `__CCE_AICORE__ == 200/220`, `__NPU_ARCH__`, `DataCopyPad(|DataCopyPadExtParams|DataCopyPadCustom|DataCopyPadExtCustom|CopyGmToUbPad(Ext)?Compat|DataCopyCustom|CopyUbToGmTailSafe`, `AtomicAdd`, plain `DataCopy(`, `GetValue|SetValue`, `HardEvent|SetFlag|WaitFlag|PipeBarrier`, dtype literals (DT_BF16/FLOAT16/FLOAT/INT8), FORMAT_FRACTAL_NZ, arch22/arch35 literals. High-value FINDINGS the scanner emits: config-only ops; runtime_kb ops (cannot synthesize); DataCopyPad-without-aicore_200-branch; 310P-branch-without-pad-compat-wrappers; UB-scalar-without-HardEvent; opFile.value-may-misroute; SoC-exclusion-routes.

**Other instances (predicted)**: port_a3 op-selection briefs reading upstream `op_host/config/ascend310p` presence to gauge 310P support; any provenance claim about "upstream supports 310P" should cite which of L1/L2/L3 + which config artifacts exist.
**Cross-ref**: [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md) (entry-point net — the host-side analog: check the SOURCE side the same way), OL-312 (LAYER-3 taxonomy), EC-96 (installed-toolkit surface), OL-399 (the kernel-side tree this host structure feeds). backend=ascendc.
