---
type: "Runbook"
title: "大规模M/E用例507035未解负结果"
description: "M≥65536 或大 E+大 k 用例稳定流同步失败 507035；UB 余量 32KB、TopK 显式 tmpLocal、topkInner 均无效；row_idx SetValue 标量循环疑点未在重编 kernel 上验证。"
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# 大规模M/E用例507035未解负结果

## 现象

Softmax+TopK 组合 kernel（Ascend950PR / arch35，CANN 9.1.0）在 ST 收尾阶段剩 6 个"大规模"用例稳定流同步失败（error 507035，kernel 执行超时或硬件异常），贯穿多轮修复始终不变：

```
fp16 [65536,64],   k=8    （大 M）
fp16 [131072,64],  k=8    （超大 M）
fp16 [262144,64],  k=8    （最大 M）
fp32 [2,4096,256], k=32   （3D 大 shape）
fp16 [131072,1024],k=128  （大 M + 大 E + 大 k）
fp32 [131072,1024],k=128  （大 M + 大 E + 大 k）
[FAIL] 流同步失败: 507035
```

同一批修复让其余 34 条用例通过，唯独这 6 条纹丝不动。

## 适用场景

- 工程形态：Softmax+TopK 组合 kernel 的大规模（大 M / 大 E / 大 k）输入场景；多核按 blockFactor 切行、核内按 batchRows 分批循环的执行结构。
- API 路径：`aclrtSynchronizeStream`（507035 返回）；kernel 内 TopK/SoftMax/DataCopyPad/SetValue 循环。
- 验证过的平台与 CANN 版本：Ascend950PR (arch35) + CANN 9.1.0。
- 适用粒度：`operator_family`。
- 路线优先级：banned（对"靠加大 UB 安全余量硬扛大规模 507035"这一路线——实测无效）。
- 明确不适用/未验证场景：小中规模用例（M≤~4096）全部通过，不受此问题影响；507035 的其他成因（如非法地址）不适用本卡判别。

## 已尝试路线（均未解决）

1. **UB 安全余量 16KB → 32KB**（`UB_SAFETY_MARGIN`，从 availableUb 中扣除）：6 条用例全部依旧 507035。
2. **TopK 9 参数（PopStackBuffer）→ 10 参数显式 tmpLocal**（sharedTmpBuf 尺寸按 `GetTopKMaxMinTmpSize` 实算）：无效。
3. **topkInner 方案**（inner 满足 %32==0，缓冲区重排）：无效（该方案修复了非对齐 E 精度，见同批 `softmax_srck_row_stride`，但大规模 507035 不动）。
4. **向量化 GenerateRowIdx**（把 `row_idx` 生成的 SetValue 双重循环 O(curBatch*k) 标量写替换为 Arange/Adds 向量化 + 批量 DataCopyPad）：**未获真实验证**——重建时 kernel .o 未重编（增量构建沿用旧二进制，见库内 `incremental_build_stale_kernel_binary`），ST 对比无效。疑点本身有估算支撑：大 M+大 k 用例每核约 600K 次 SetValue 标量调用，超时嫌疑最大。

## 根因分析（未定位）

三个候选机制均未证实：UB 溢出（32KB 余量后按 tiling 公式手算 3D 大 shape 用例总 UB ~62KB « 248KB，存疑）、TopK 大 k 排序超时、大 M 迭代次数过多 × 每迭代同步开销。plog 中无 kernel 侧明确错误信号。此为**未追到根因**的问题，issue 文件已记录。

## 预防

- 大规模用例尽早纳入联调（本轨迹到收尾阶段才发现 6 条硬骨头，修复窗口不足）。
- kernel 内避免 O(行数×k) 的 SetValue 标量循环（本轨迹的 row_idx 生成即此模式），改向量化生成。

## 验证信号

来源说明：以下均为本轨迹 NPU ST 实测（同一联调会话多轮重建后）：

- 现象信号：6 用例在各轮修复版本均为 `[FAIL] 流同步失败: 507035`，失败集合逐字不变（来源平台：950；证据等级：tool_observed）。
- 尝试 1 无效：UB_SAFETY_MARGIN 32KB 重建后该 6 条不变（来源平台：950；证据等级：tool_observed）。
- 尝试 2/3 无效：同上，多轮 34/45 时该 6 条仍失败（来源平台：950；证据等级：tool_observed）。
- 尝试 4 无结论：修复后 ST 34/45 与改前一致，但 `find -newer` 证实 kernel .o 未重编，结果无效（来源平台：950；证据等级：tool_observed）。
- UB 手算反证：3D 大 shape（fp32, M=8192, E=256）按 tiling 公式总 UB ≈ 62KB，远小于 248KB 上限——"简单 UB 溢出"解释存疑（来源平台：950；证据等级：artifact_verified，事件内手算）。

## 失效边界

- 本卡不给出修复承诺；结论边界是"上述 4 条路线中前 3 条已被同轨迹证伪，第 4 条未验证"。
- 507035 若伴随明确越界日志则属其他根因，本卡判别不适用。

**未决项**：

- **优先项：复现并追根因到机制层**——先在确认重编的环境下验证向量化 row_idx 的效果（最小成本、疑点最集中）；仍失败则用 msprof/任务耗时分解区分"单迭代慢"与"迭代数多"。
- 候选方向（本轨迹仅提案）：进一步压 batchRows（余量 48KB+）、DoubleBuffer 流水、大 k 拆分 TopK、核查硬件超时配置。
## 坏实践

- 在 kernel 未确认重编时用 ST 对比否定修复（增量构建旧二进制陷阱）。
- 仅凭"加大安全余量"处理超时类失败而不做时间/迭代分解——本轨迹 16K→32K 一轮即证伪。

## 误诊

- 曾按"UB 溢出"主线索连续两轮加余量/换 tmpLocal，均无效后转向超时假设；未做 profiling 级验证（该段归因为推理叙述）。
