---
type: Runbook
title: "V220 hand-written bare-mmad cube paths silently miscalculate on A5 (L0A transpose misplacement at blockM>=2) — route-guard them off and keep MatmulImpl (PB-43 cross-gen instance)"
description: "Cross-gen (A3→A5) instance of PB-43: 910B-era hand-written cube pipelines (*_manual.h, 16x16-fractal LoadData(ifTranspose=true) + bare Mmad hardcoding DAV_2201 L0A/L0B/L0C fractal layout) compute silently wrong on A5 when a tile spans multiple M fractals (blockM>=2, M>16) — max_abs_diff 32-148, output std ~13, no error. Same-op MatmulImpl (reserved==0) is bit-exact (diff=0.0) for fp32/fp16/bf16."
tags: [task.debugging, topic.precision, topic.implementation.pipeline, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/cube-manual-mmad-crossgen-001.md
    role: primary
created_at: "2026-08-26T12:44:00Z"
updated_at: "2026-09-26T19:10:00Z"
platforms: [a2, a3, "950"]
aliases: ["cube-manual-mmad-crossgen-001"]
---

# V220 hand-written bare-mmad cube paths silently miscalculate on A5 (L0A transpose misplacement at blockM>=2) — route-guard them off and keep MatmulImpl (PB-43 cross-gen instance)

> 原 OKF v1 卡号：**cube-manual-mmad-crossgen-001**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - diff magnitude >> rounding error (multiples of output std) and strongly correlated with blockM/tile size — tile misplacement, not accumulation precision
> - hand-written *_manual.h cube path comments claim reliability only on DAV_2201 / 910B
> - large-shape cases route to manual mmad paths despite the intuition that they use the library path

## 现象 / 触发

MatmulTransA（Ascend950DT + CANN 9.2.0 实测）：

- A3 时代手写裸 cube 流水线（`*_manual.h`：逐 16x16 分形 `LoadData(ifTranspose=true)` + 裸 `Mmad`，硬编码 DAV_2201 的 L0A/L0B/L0C 分形布局）在 A5 上**当 tile 跨多个 M 分形（blockM≥2，M 维>16）时 L0A 转置错位、静默算错**（max_abs_diff 32~148，输出 std ~13）；blockM==1 时布局退化一致、碰巧正确。
- 同一算子里的高层库 `MatmulImpl`（reserved==0，MDL scheduler）在 A5 上对 fp32/fp16/bf16 **全部 bit 级正确（diff=0.0）**。

### 源码侧触发信号（generated bare-cube，不只是手写 `*_manual.h`）

本卡原有的触发描述以**手写** `*_manual.h` 为主。实测还有一个同机制但源码长相完全不同的家族：
**TileLang/catlass 生成的 bare-cube kernel**。它同样硬编码 DAV_2201 分形布局，但文件名里没有
`manual`，所以按 `*_manual.h` 找不到。

port_a3 worker 从 A3 源码入手时实际会撞到的 token（下列均在一份真实 A3 源里逐条核过）：

| token | 出现位置（实例） |
|---|---|
| `CATLASS_ARCH` / `-DCATLASS_ARCH=2201` | `CMakeLists.txt`（实例 `:85`）、`op_kernel/*_common.h` 的 `#ifndef`/`#define`（实例 `mbt_common.h:19-20`） |
| `CATLASS_DEVICE` | 生成的 kernel 头文件函数修饰（实例 `mbt_gen_kernel.h:43`） |
| `tl::ascend::gemm_v0` | 生成 kernel 的裸 cube 调用点（实例 `mbt_gen_kernel.h:125`） |
| `tl::ascend::copy_gm_to_l1` | 生成 kernel 的 L1 搬运点（实例 `mbt_gen_kernel.h:111`、`:116`） |
| `BACKEND_HYBM` | 编译定义 |
| vendored `tilelang-ascend` / `3rdparty/catlass` include 路径 | `CMakeLists.txt` |

**为什么补这一段（可复核的检索测量，不是推测）**：在 k=14（消费者真实 k）下对本语料索引实测——

- query `CATLASS_ARCH 2201` ⇒ `cube_manual_mmad_crossgen_001` 与 `OL-197` **两张都不在前 14**；
- query `V220 manual per-fractal Mmad silently wrong on A5 arch35` ⇒
  `cube_manual_mmad_crossgen_001` **rank 1**、`OL-197` **rank 6~7**，**两者均在前 10**。
  （名次随语料状态漂移，所以单报一个名次是欠定的：本次测于语料 `7a2c3f31` 为 6；非作者评审在
  `94f788bb` / `122f066c` 各配置下复现为 7，`ol_240` 占据第 6。前 10 这个结论在三个状态下都成立。）

即这两张卡**按症状词可召回、按源码自带的宏 token 不可召回**。而 port_a3 的第一步恰恰是读
`op_host/`/`op_kernel/` 源码，最自然的检索词就是源码里那个显眼的宏。这不是"没人找得到"——本次
移植的 worker 确实用症状词命中了本卡并据此改了设计——而是**最自然的入口会漏**。补 token 只是
让入口对齐，不改本卡任何结论、范围或动作规则。

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A2 系列、Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 动作规则（A3→A5 cube 类移植第一步）

1. **先做路由探针**：给 host tiling 加调试输出，确认每种 dtype/shape 实际走哪条路径（reserved/分支），不要凭"大 shape 走库路径"的直觉下结论——MatmulTransA 的 fp16/bf16 大 shape 实测全走手工 mmad。
2. **禁用手工裸 cube 路径**（路由守卫 `kDisableManualCubePaths=true` 或删路由分支），全部落 MatmulImpl/高层库；tiny shape 若依赖手工路径破 ~13us 固定开销，需为 arch35 重写，正确性优先先禁用。
3. 修复前先看源码自证：手工路径头文件常有 "only ... reliable on DAV_2201" 类注释，即高危标记。

## 方法普适性验证（2026-08-22，deepseek-v4-flash × 2 agent 并行）

- **3_MatmulBothTrans_evo**：无手写裸 mmad，运行级零回归；阻塞在编译期——纯 AIC `*_cube.cpp` 在 CANN 9.2.0 arch35 报 set_padding 错误，最小修复 = `if ASCEND_IS_AIC` 设备侧守卫（~6 行/3 文件）。修复后 55/55 PASS diff=0.0。
- **8_ConvStandard2d_evo**：35/59 FAIL 全为 AIC 裸 cube 变体（裸 Mmad + 手工 L0 分形 load），NaN 输出；AIV Vector 路径全对。conv **无 MatmulImpl 可退** → `kDisableCubePaths` 全走 AIV → 59/59 PASS（1.2e-7~5.5e-6）。代价：大 K 慢 30-60×。
- **教训汇总**：① 路由探针是通用第一步（3/3 算子都靠它纠正前提）；② 手工裸 cube 路径跨代必炸（matmul/conv 两例实证）；③ 有库路径退库路径（零代价），无库路径退 AIV 兜底（正确但大性能回退）；④ 纯 AIC 编译在 arch35 需要 `ASCEND_IS_AIC` 守卫；⑤ 共享卡可能被外部会话占满——运行中也要盯 npu-smi。

## 证据

- `docs/mmta_fp16_rootcause_report.md`、`docs/conv2d_a5_rootcause_report.md`、`docs/mbt_a5_rootcause_report.md`。
- 关联：PB-43（cube 手工 fractal load 静默算错，build-success≠validation）——本条即其跨代移植实例。
