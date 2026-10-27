---
type: Runbook
title: "Profiler COLLECT and EXPORT/PARSE are separable stages — a missing kernel_details.csv indicts the chain, not the host"
description: "Profiler COLLECT and EXPORT/PARSE are separable stages — a missing kernel_details.csv indicts the chain, not the host"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-453"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-453 — OL-453: profiler COLLECT and EXPORT/PARSE are separable stages — a missing `kernel_details.csv` indicts the chain, not the host

## 适用场景

- 适用平台与范围：scope=torch_npu.profiler / msprof device-time capture wherever kernel_details.csv is the measurement basis; soc=Ascend950DT_9582; cann=9.1.0 (parse) and 9.2.0 (collect); bisheng=n/a。
- 验证来源：soc=Ascend950DT_9582 — DS 4-55 archive current_p_control paired device F, Ascend950DT host card 2, 2026-09-11; evidence IS in main at commit 581cd303b8eac5744d18c20f01d5c5fddd2e4733 (!943), archive tree 5225e02dc420bd7ee2a83c502a2fd14725d24942。

**OL-453** — **the profiler's COLLECT stage and its EXPORT/PARSE stage are separable across container images, so a missing `kernel_details.csv` is evidence about the chain you ran, not about the host.** Raw profiler data collected under one CANN/torch_npu can be parsed by a different image — and the image that fails to parse may be the NEWER one, so "upgrade the image" is the wrong reflex.

<!-- applies_to_backend: all -->

`applies_to: scope=torch_npu.profiler / msprof device-time capture wherever kernel_details.csv is the measurement basis; soc=Ascend950DT_9582; cann=9.1.0 (parse) and 9.2.0 (collect); bisheng=n/a`
`verified_on: soc=Ascend950DT_9582 — DS 4-55 archive current_p_control paired device F, Ascend950DT host card 2, 2026-09-11; evidence IS in main at commit 581cd303b8eac5744d18c20f01d5c5fddd2e4733 (!943), archive tree 5225e02dc420bd7ee2a83c502a2fd14725d24942`
`unverified_on: Ascend950PR (any revision) — UNVERIFIED for this entry; and every other SoC, host and image pair. The sealed archive carries NO SoC field of its own; the SoC above is the host→SoC binding for Ascend950DT host from the fleet registry, not a token read out of the archive. Do not restate it as archive-sourced.`

- **Category**: process, instrument
- **Trigger**: `kernel_details.csv` is absent or empty after a profiler run, or `analyse()` / `CANNTimelineParser` errors — and the next step being considered is "profiling is unavailable on this host", a fall back to Event/wall-clock timing, or an image upgrade.
- **Mechanism**: `torch_npu.profiler` writes raw profiler data in a **collect** stage; a separate **export/parse** stage (`torch_npu.profiler.profiler.analyse`, backed by `CANNTimelineParser`) turns that raw into `kernel_details.csv`. The two stages do not have to run in the same image. Observed pair: image **e268** (CANN 9.2.0, torch_npu `2.9.0.post7.dev20260824`) collects correctly but **its own parse produces no `kernel_details.csv`** — `CANNTimelineParser` fails and nine downstream tasks cascade. Image **33dd** (CANN 9.1.0, torch_npu `2.10.0.post5.dev20260817`) parses that same raw correctly. **The failing image is the newer CANN**, so version order does not predict which side works.
- **Positive control**: 33dd re-parsing e268-collected raw matched 33dd's own **native** matmul control at **2154.02 µs vs 2153.47 µs (ratio 1.00025)** — i.e. the cross-image re-parse and a same-image parse of the same workload agree to 2.5e-4. In the production F run the in-band positive control read **2153.639 µs (1 CSV)**.
- **Negative control**: a host-side loop with no device work produced **0 traces / 0 CSV**. The chain therefore reports absence as absence; the positive number is not an artefact of a runner that always emits something.
- **Counter-example 1 — the profiler schedule is NOT the cause**: the same parser failure occurred in e268 **both with and without** the `torch_npu.profiler` schedule (`wait=0,warmup=5,active=5,repeat=1`), run as an explicit A/B. Changing the schedule or the capture schema does not recover the CSV, so do not spend a window tuning it.
- **Counter-example 2 — the alternative instrument is not a substitute**: `torch.npu.Event` + synchronized wall-clock does run in e268, but its reading is contaminated by **any host work enclosed by the markers, and that contamination scales with the host work** — it is NOT a fixed platform floor. Measured on this host as three control arms (`E_RESULT.json`, medians): **null call `14.705 µs`** (nothing between the markers), **deliberately inserted host loop `816.58 µs` Event / `873.71 µs` wall**. The inserted loop inflated the Event delta ~55× over the null arm, which is the direct demonstration of PB-51's admission condition (1). Separately, Event and the parsed `kernel_details` landed ~24% apart (Event GM `0.00487` vs parsed GM `0.003916`) — but in **different windows**, so that figure is **MOTIVATING EVIDENCE ONLY** and is not a cross-validation. See PB-51.
- **Limit — the re-parse is NOT cardless**: `msprof --export` needs driver/device init, so the parse step still consumes a device window. Budget a **second** window; a host-only re-parse is not available.
- **Rules**:
  1. When `kernel_details.csv` is missing, **test the parse stage separately from the collect stage** before declaring profiling unavailable on the host, falling back to Event/wall-clock, or upgrading an image.
  2. Any split-chain run carries **both** controls: a **positive** (a known workload re-parsed cross-image vs parsed natively in the parsing image) and a **negative** (host-only loop → 0 traces). Without the negative, "we got a CSV" does not distinguish a real parse from a runner that emits regardless.
  3. Record **which image parses, by identity** (image tag + CANN + torch_npu build). Do not infer the direction from version order — here the newer CANN is the failing side.
  4. Budget a device window for the parse; it is not free.
- **Evidence**: DS, 2026-09-11, 4-55 `OutlookAttention` archive `current_p_control` paired device F on `Ascend950DT host` card 2. Archive commit `00b0a08ab0b1183ffaa3c31f89bc78d76c955592`, `私有沉淀仓` branch `ds/op55-p-v2-window-20260911`, directory `output/op55_jx1_20260910/p_v2/device_windows/20260911T0245Z_A5DT-122_archive_control_F/` (70 files). The quoted numbers are read from that commit's `RESULT.md` — controls at line 18, the split rationale and the 1.00025 positive control at line 19, the Event control arms at line 42 (whose prose phrase “host-loop floor” is what an earlier revision of this entry MISREAD as a platform floor — the JSON shows it is the inserted-host-loop control arm), and the per-window table (parser failure both schedule arms; 33dd validation PASS; Event INTERIM; collection; parse+aggregation) at lines 48-53. In-archive receipts: `win_parseval_20260911T0312Z/PARSE_VAL.json`, `win_agg_20260911T0330Z/F_RESULT.json`, `win_collect_20260911T0325Z/COLLECT_MANIFEST.json`. Wiki locator: 内部路径（不公开）agent-team-wiki/projects/priority24/RESULT_20260911T0400Z_DS_OP55_ARCHIVE_CONTROL_F.md. The Event control medians are read from `win_event_20260911T0320Z/E_RESULT.json` (`controls.null_call.event_us_median`, `controls.negative_host_loop.event_us_median`, `controls.negative_host_loop.wall_us_median`) — not from prose.
- **Provenance — the evidence IS in `main`**: commit `581cd303b8eac5744d18c20f01d5c5fddd2e4733` (`!943`, the merge of `ds/evidence-op55-archive-control-f-20260911`) is an ancestor of 私有沉淀仓 `main`（内部留存，上游不可见）, and its copy of the archive directory is the byte-identical tree `5225e02dc420bd7ee2a83c502a2fd14725d24942`. An earlier revision of this entry claimed the evidence was "not in main"; that was **wrong** — it bound the ds-branch *commit*, which is indeed not an ancestor, and never checked whether the *content* had reached main by another route. A commit not being an ancestor says nothing about whether its bytes are in main.
- **Not checked**: which of the three differences between the two images (CANN version, torch_npu version, container parser dependencies) causes the split was **not isolated** — all three differ. This entry claims the **procedure**, not the root cause. Not measured on any other host or image pair; the direction (9.2.0 fails / 9.1.0 parses) is a single observed pair, not a version rule.
- **Disposition**: transferred into canonical KB from DS's measured artifact; **not** an independent rediscovery by the author of this entry.
- **Cross-reference**: **PB-51** — a DIFFERENT reason the same CSV goes missing (an EPERM from a container/securityfs profiling-syscall restriction, not a parser failure), and the entry that carries the Event admissibility rule this one's counter-example 2 feeds. Also distinct from the `operator_details.csv` invisibility rule in this file (kernels launched via raw `aclrtlaunch_*` never appear as rows, producing a bogus speedup) — that is rows missing **within** a working chain. Three distinct ways a profiler number goes wrong; do not conflate them.
