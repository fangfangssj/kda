---
type: "Runbook"
title: "删 smalle slim TU 致 e≤32 decode icache 税"
description: "对 moe_topk_softmax 去重重构尝试删 smalle slim TU（base 并入 full 大 TU，4→2），msprof N=9 interleaved 18 档实测 full 入口相对 smalle 在 e≤32 decode 全 FAIL（+2.53%~+15.38%），门未过→保 smalle（4→3）；记录 banned 删除路线与 icache 取指税边界。"
tags: ["task.optimization", "topic.compilation", "topic.performance.analysis", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T15:56:49Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 删 smalle slim TU 致 e≤32 decode icache 税

## 现象

在为 `moe_topk_softmax` 算子做"去重重构"收口时，coordinator 亲跑"删 smalle icache-税消失门"。该算子去重后 op_kernel 目录有 4 个文件：`moe_topk_softmax_kernel.asc`（full TU，含 base + V4 两大类，device `.text` 约 72.5KB）、`moe_topk_softmax_smalle_kernel.asc`（smalle slim TU，仅 base 路径，约 34KB，12 行薄壳）、tiling 头文件、以及一个已被无条件删除的 mide TU。host 侧通过 `segCount==1` 派发 smalle、其它派发 full。

门的目标：用同一份 `build_v51r1/moe_topk_softmax` exe，通过环境变量 `MTS_FORCE_SMALLE=0`（强制 full 入口）vs `=1`（强制 smalle 入口）切换，跑 msprof 采集 e{8,16,32} × dtype{fp32,fp16,bf16} × bs{1(decode),4096(prefill)} 共 18 档 Task Duration，N=9 interleaved 取 median。判据：full 入口相对 smalle ≤ +2%（无 icache 悬崖税）→ 删 smalle（文件 4→2、host 单入口）；否则保 smalle（4→3）。

实测结果 `GATE_RESULT: SOME_FAIL`：decode（bs=1）9/9 全 FAIL，full 入口相对 smalle +2.53%~+15.38%（中位 +8.13%），远超 ≤+2% 阈值；prefill（bs=4096）6/9 PASS，但 bf16 三档仍 FAIL（+8.66%~+10.33%）。门未通过 → 保留 smalle slim TU（4→3）。这是对设计先验"full-72.5KB 可能对 e8-base 免税"的否证：e8-base-on-72.5KB 仍明显吃 icache 取指税。

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke），算子工程重构/精简阶段；非新算子开发、非精度修复。
- **API 路径**：`MTS_FORCE_SMALLE` 环境变量切 full/smalle 入口；msprof timeline `--ai-core=on --aic-metrics=PipeUtilization --task-time=on --ascendcl=on`；cannsim `record -s Ascend950` + `report` 复验 VLOOPv2。
- **验证过的平台与 CANN 版本**：Ascend950PR / DAV_3510 / arch35 / CANN 9.1.0 / coreNum=56 / UB=216KB（仅 950 实测，与 frontmatter `platforms: [950]` 严格一致）。
- **适用粒度**：`operator_family`——适用于"launch-bound、compute 极小、icache 敏感的 topk/softmax 类 kernel 在多 TU 入口间的选择决策"，不限于 moe_topk_softmax 单算子。
- **路线优先级**：`banned`——**禁止在未跑 icache-税门禁前删除 slim TU**；删除 slim TU（把 base 路径并入 full 大 TU）对 e≤32 decode 形状实测造成 +2.53%~+15.38% 性能回退，属 banned 路线。安全路线是保留 slim TU（4→3）。
- **明确不适用/未验证场景**：e≥64（V4 路径，本就只走 full TU，不涉及 slim 选择）；compute-bound 大 bs 场景（取指税被摊薄，门可能 PASS 但不构成删除许可——仍须实测）。其它平台（a2/a3）未验证，推断可迁移性见 `## 失效边界`。

## 根因

e≤32 的 decode（bs=1）是 **launch-bound** 场景：单次 launch 的 compute 极小、取指（icache）成本占比高。full TU（约 72.5KB，含 base + V4 两大类全部代码）相对 smalle slim TU（约 34KB，仅 base 路径）在冷取指/launch 敏感场景暴露 icache 取指税——full 入口的取指 footprint 近似为 smalle 的 2.13×（72.5KB / 34KB），在 icache 容量边界上更易触发缺失。这正是引入 smalle slim TU 的原始动机：把 e≤32 base 路径从 full 大 TU 中拆出单独薄壳，降低取指 footprint。

prefill（bs=4096）时 compute 时间主导，取指税占比降低 → fp32/fp16 多落噪声带内（≤±2%）；但 bf16 路径（批 Cast）对取指更敏感，即使 prefill 仍 FAIL（+8.66%~+10.33%），说明 bf16 的取指敏感度非噪声能解释。

分离度稳健非噪声：以最坏 e8 fp32 decode 为例，full 全部 9 个样本的最小值几乎完全高于 smalle 的中位值（两分布近乎不重叠），属真实回归。

## 失败路线

**banned 路线：删除 smalle slim TU，把 e≤32 base 路径并入 full 大 TU（op_kernel 4→2、host 单入口）。**

实测否决（Δ 为 full 相对 smalle 的成对代价，正 = full 更慢 = 吃税；参照路线 = smalle slim TU 入口）：

| e | dtype | decode(bs=1) Δ | 判定 | prefill(bs=4096) Δ | 判定 |
|---|---|---:|:---:|---:|:---:|
| 8 | fp32 | +12.35% | FAIL | −3.36% | PASS |
| 8 | fp16 | +7.52% | FAIL | +0.10% | PASS |
| 8 | bf16 | +15.38% | FAIL | +10.33% | FAIL |
| 16 | fp32 | +6.75% | FAIL | −2.42% | PASS |
| 16 | fp16 | +8.13% | FAIL | +0.31% | PASS |
| 16 | bf16 | +13.22% | FAIL | +8.66% | FAIL |
| 32 | fp32 | +5.89% | FAIL | +1.36% | PASS |
| 32 | fp16 | +2.53% | FAIL | −1.51% | PASS |
| 32 | bf16 | +11.96% | FAIL | +8.85% | FAIL |

- decode（bs=1）：0/9 PASS，中位 +8.13%，最坏 e8 bf16 +15.38%。
- prefill（bs=4096）：6/9 PASS，但 bf16 三档 FAIL（+8.66%~+10.33%）。
- 综合 9 FAIL / 18 → 门未通过 → **保 smalle（4→3）**。

VLOOPv2 辅助复验（cannsim, full 入口 base 路径）确认 full 入口 e≤32 base 路径 VF 区健康（VLOOP≥2、SCBZI=0、SJUMP=0，无软循环退化），即"删死码 + helper 下沉"本身未致 VF 退化——性能回退纯粹来自 icache 取指 footprint 增大，非硬件循环退化。VLOOPv2 不影响删/保结论：性能门已明确否决删除。

## 验证信号

本验证信号来自本 trace 内 coordinator 亲跑的 perf 门（msprof N=9 interleaved, 18 档）与 cannsim 复验，均为上板/模拟器实测，非开发者自报。

- decode（bs=1）9/9 FAIL：full 入口相对 smalle slim TU +2.53%~+15.38%，中位 +8.13%，超 ≤+2% 判据（来源平台：950；证据等级：tool_observed）。
- prefill（bs=4096）6/9 PASS；bf16 三档 FAIL（+8.66%~+10.33%）（来源平台：950；证据等级：tool_observed）。
- 综合门禁结果 `GATE_RESULT: SOME_FAIL`（9 FAIL / 18）→ 保留 smalle slim TU（4→3）（来源平台：950；证据等级：artifact_verified）。
- cannsim full 入口 e≤32 base 路径 VF 区 VLOOPv2 HEALTHY：SelectK1ReduceMax=2、InitIndices=4、ComputeValues=4、OutputTopKImpl=2，SCBZI=0、SJUMP=0（来源平台：950；证据等级：tool_observed）。
- 前序去重阶段已证 full 与 smalle 输出 528/528 byte-exact（本门仅测性能入口选择，不引入数值风险）（来源平台：950；证据等级：artifact_verified）。

## 失效边界

- **a2 / a3 平台未验证，推断可迁移**：icache 取指税是 Ascend 系列通用机制，但 a2/a3 的 icache 容量、launch 开销与 950 不同，full TU（72.5KB）vs smalle（34KB）的取指 footprint 比在 a2/a3 上的绝对敏感度未实测。迁移到此二平台前须重跑本门。
- **compute-bound 大 bs 场景**：prefill（bs=4096）上税被摊薄，门可能 PASS，但 PASS 不构成删除许可——bf16 prefill 仍 FAIL 说明 bf16 路径对取指更敏感，即使 compute-bound 也不能假定免税。
- **e≥64（V4 路径）**：本就只走 full TU，不涉及 slim 选择，本卡 banned 路线不适用。
- **非 topk/softmax 类、非 launch-bound kernel**：若算子本身 compute-heavy（matmul/conv），icache 税占比低，删 slim TU 可能无回退——但仍须实测门禁，不得假定。

## 坏实践

- 在未跑 icache-税门禁前为"精简文件数"直接删 slim TU，把 base 路径并入 full 大 TU——对 launch-bound decode 形状实测回退 +2.53%~+15.38%。
- 用"前序阶段 byte-exact 528/528"作为删 slim TU 的许可——byte-exact 只证数值等价，不证性能不退化；性能入口选择必须独立上板门禁。
- 用 cannsim VLOOPv2 HEALTHY 作为删 slim TU 的许可——VLOOPv2 只证 VF 未退化，不证 icache 取指税；取指 footprint 必须上板 msprof 实测。

## 预防

- 任何"把 slim/specialized TU 合并进 full 大 TU 以精简文件数"的重构，在合并前必须跑 icache-税门禁：同一 exe、`MTS_FORCE_SMALLE` 切入口、msprof N≥9 interleaved、覆盖 launch-bound（bs=1）× 全 dtype × 多 e 档，判据 Δ ≤ +2% 方可合并。
- 判据须含 bf16 档：bf16 路径（批 Cast）对取指更敏感，prefill 也可能 FAIL，不能只测 fp32。
- slim TU 的存在价值在取指 footprint（34KB vs 72.5KB，近似 2.13×），而非 VF 健康——cannsim VLOOPv2 只能排除断裂，不能证明免税。

## 复现

```bash
source /home/developer/Ascend/cann-9.1.0/set_env.sh
cd /workspace/liulinxiang/e128_cmp/moe_topk_softmax
PERF_N=9 python3 scripts/perf_stage4b_smalle_gate.py   # 性能门（约 40 分钟，18 档 Δ≤+2% 判据）
bash scripts/_cannsim_stage4b.sh                          # VLOOPv2 复验（full 入口 base 路径 VF 健康）
```

产物归档于 `docs/perf/round_050_stage4b/`（`summary.md` + `verdict.csv` + `samples_*.csv` + `run_full.log` + `cannsim/` + `raw/` + `syms/`）。
