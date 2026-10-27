---
type: Runbook
title: "knowledge_update — 73_ChunkKdaBwdIntra (DRAFT, updated incrementally)"
description: "knowledge_update — 73_ChunkKdaBwdIntra (DRAFT, updated incrementally)"
tags:
- task.debugging
- topic.implementation.debugging
status: draft
sources: []
created_at: "2026-09-21T05:32:39Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-SPAWN-WALLCLOCK-9000S"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# knowledge_update — 73_ChunkKdaBwdIntra (DRAFT, updated incrementally)

> 来源：a5_ops 运行后 knowledge_update 合入（OKF cutover 写路径）。status=draft，按卡内声明范围消费。

## 现象 / 教训

## Summary of findings this spawn

### F1 (CAND, multi-core correctness): 4-byte-column fp32 outputs shared across cores need region-ownership, and CANN 9.2.0 has NO DCache-bypass scalar store
- **Symptom**: output tensor `db_out [B,T,HV]` fp32 corrupted when each hv column is written by a different AIV core via `GlobalTensor::SetValue` — wrong values even when every accumulator is provably zero (i.e. the corruption is in the STORE path, not the math). On a probe: `db` max_abs_diff 0.6–61 (same magnitude as the data); dq/dk/dg (written by MTE3 DataCopy) exact.
- **Root cause**: `GlobalTensor::SetValue` writes through DCache at 64B-line granularity (OKF `simd_api/basic_api/atomic_operations/key_features.md` documents exactly this multi-core clobber). Adjacent hv columns of one row share a line → random overwrite.
- **The documented workaround does NOT exist on CANN 9.2.0**: `WriteGmBypassDCache` (OKF `simd_api/basic_api/scalar_compute/write_gm_bypass_d_cache_isasi.md`, platforms: 950 newer SDK) is absent from the 9.2.0 `aarch64-linux/asc/include` tree — verified by grepping the remote SDK headers (OL-130 chain). `DataSyncBarrier` exists; the bypass store does not.
- **Sound fix (structural, not workaround)**: choose the parallel unit so that each core owns a contiguous, 32B-aligned span of the narrow output and writes it with ONE plain DataCopy. For this op: unit = chunk (all HV heads inside) → core owns `[L*HV]` fp32 db region; values staged in UB via `LocalTensor::SetValue` (single-core, sound), `S→V` fence, `Add` with widened db input tile, single MTE3 store. Host-side `CheckShape` asserts `(BT*HV)%16==0 && (T*HV)%16==0` so region start+length are always 32B multiples in the dense layout.
- **Generalization**: ANY kernel emitting a sub-32B-granular output tensor from multiple cores (scalar columns, per-row scalars like `db`/`mean`/`var` side-outputs) must either (a) give each core a line-aligned contiguous region, or (b) stage per-core values in UB and have ONE core own each output line. Per-column scalar GM stores from multiple cores are unsound on this toolchain.
- **Determinism note**: the structural fix also preserves by-construction determinism (each output element written by exactly one core via MTE3).

### F2 (process): varlen rejected as an explicit capability boundary
- cu_seqlens chunk starts make the db region alignment unprovable; pybind raises TORCH_CHECK with an explicit message. No varlen case exists in the 50-case population. NOT a silent skip — recorded in analysis.md and pybind error text.

### F3 (probe methodology): isolation probes beat diff-staring for store-vs-math attribution
- Zeroing dAqk/dAkk separately on a tiny case (T=4, HV=2) attributed the failure to the db STORE path in one step (db wrong with all accumulators zero). Also: a probe's golden must replicate the reference's masking exactly — the first probe iteration used an unmasked random E and produced golden-side false failures on dq/dk/dg (self-caught before any kernel edit based on them).

### F4 (deploy ergonomics, for orchestrator/knowledge-maintain, not a worker workaround)
- `deploy_to_npu_lane.sh --build` re-tars and re-uploads the full workspace including the multi-GB `pass_a_dataset.*.pt` on EVERY build (~45 min at ~0.7 MB/s for a 4 GB workspace on this link). A content-addressed skip for unchanged heavyweight artifacts would cut the precision-fix iteration cost ~10×. Filed as observation only; no worker-side bypass was used (P6).

## Failure-causality fields (for non-PASS handoffs; update at handoff time)
- failing_layer: TBD
- first_failing_iter: TBD
- root_cause_class: TBD
- minimal_repro: TBD
- attempted_fixes: TBD

## 故障处置（2026-09-21 05:0x，spawn wall-clock timeout @ await_worker）

**失败**：worker spawn 撞 9000s 墙钟帽（receipt ff-ecb4fdbcc19754ec9df728ca）。worker 最后活动=沙箱旁路部署舞（copy transitive modules/verify closure），之后 2h 无 trace 增长——进程死在半路且无 result 事件（zombie 类），不是内核缺陷。

**本轮已知新知**：① 73 候选 kernel 四文件在 kernel/（pre-build 静态门 11/11 全过，UB 预算 229.7KB<248KB）；② db 标量收尾需全 VEC 化（BlockReduceSum/PairReduceSum，禁 VEC 写后标量读）；③ CANN 9.2.0 ReduceSum L0 是全归约（用 WholeReduceSum int-mask 重载）；④ 部署 staging 在沙箱里写不了兄弟目录是环境事实，不是内核问题。

**重试差异**：续跑从现有 workspace 工件继续（不重新生成），worker 从 db 收尾+构建验证接着走；无 harness 变更。

→ orchestrator: fail — spawn wall-clock timeout (worker process died mid-deploy without result event); resume continues from retained workspace artifacts

## 故障处置（2026-09-20 22:0x，spawn wall-clock timeout @ await_worker，cold-start 后首轮）

**失败**：worker spawn 撞 9000s 墙钟帽（receipt ff-ecb4…728ca 系）；worker 最后活动=沙箱旁路部署（模块闭包拷贝），随后 ~2h 无 trace 增长、无 result 事件——进程半路死/挂起，非内核缺陷。

**处置知识**：① 候选 kernel 四文件在 kernel/ 且静态门 11/11 过；② 部署 staging 在沙箱内写兄弟目录会吃 EROFS，worker 的旁路舞是环境事实；③ 墙钟帽超时时进程已死=静默死亡，监控要看 trace 增长而不是进程在不在。

**重试差异**：从保留 workspace 工件续跑（不重新生成），worker 从 db 收尾+构建验证继续；无 harness 变更。

→ orchestrator: fail — spawn wall-clock timeout (worker stalled in sandbox-bypass deploy, died silently mid-flight); resume continues from retained workspace artifacts

别名: CAND-SPAWN-WALLCLOCK-9000S（本卡别名：spawn 墙钟帽超时= worker 静默死亡类，监控看 trace 增长而非进程存在）

## 适用场景

- **工程形态**：knowledge_update — 73_ChunkKdaBwdIntra (DRAFT, updated incrementally)。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
