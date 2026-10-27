---
type: "Runbook"
title: Select张量-标量模式索引不更新，须TT模式
description: ARA-scan 用 VSEL_TENSOR_SCALAR_MODE 更新 float 索引并以张量充当标量参数，NPU 上索引恒 0；改 Duplicate 广播 + VSEL_TENSOR_TENSOR_MODE 后恢复。
tags: [task.debugging, topic.precision, topic.compute.elementwise.comparison_logic]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# Select张量-标量模式索引不更新，须TT模式

## 现象

Cummin 算子首次上 NPU 跑 L0 ST（C++ 测试、真实 NPU 执行、CPU golden 比对），ARA-scan 模式（扫描维非最后一维）出现"值对、索引错"的失败：

- L0_003（int32，shape=[6603,10,6]，dim=-3，归一化 A1=1、R=6603、A0=60）：values 全部匹配，但 `indices: 394944 mismatches out of 396180`，失配模式是 `mismatch [61]: golden=1, actual=0`——输出索引几乎全停在初始值 0；
- L0_004（bfloat16，shape=[983,5,1,5]，dim=-4）：values 通过，indices 大量垃圾值 `mismatch [0]: golden=0, actual=-4611756388245323777`；
- L0_005（float16，shape=[13,235,1,1,4]，dim=1）：values MERE=2.19e-02 超阈 + indices 1481/12220 mismatch（此用例另含向量管线根因，见同批《向量Compare/Select小A0结果错，改标量扫描》卡）。

当时 ARA-scan 的向量流水线（`op_kernel/arch35/cummin.h`）中，r=1..R-1 迭代用 Compare 生成掩码后，以张量-标量模式 Select 更新运行索引：

```cpp
// Update running_idx
LocalTensor<float> runIdx = runningIdxBuf.template AllocTensor<float>();
LocalTensor<float> rowIdxFloat = rowIdxFloatBuf.template AllocTensor<float>();
Duplicate(rowIdxFloat, static_cast<float>(r), tileA0Aligned);

// Select(TS): runningIdx = cmp ? runningIdx : r_float
Select(runIdx, cmpMask, runIdx, rowIdxFloat, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileA0Aligned);
```

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 向量流水线中用 `Select` 做条件更新（min/max 索引跟踪、条件搬运选择等）。
- API 路径：`AscendC::Select` 的 `SELMODE::VSEL_TENSOR_SCALAR_MODE` / `VSEL_TENSOR_TENSOR_MODE`、`Duplicate`（标量广播）。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：`operator_family`（向量 Select 条件更新路径）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：值路径的 Select（`Select(runMin, cmpMask, runMin, currentBuf, VSEL_TENSOR_TENSOR_MODE, ...)`，张量-张量模式，本例正常）；TS 模式配真实标量（非张量）参数的行为未验证。

## 根因/解决配方

根因：索引更新调用使用 `SELMODE::VSEL_TENSOR_SCALAR_MODE`，但第 4 个操作数传入的是 `LocalTensor<float>`（rowIdxFloat）而非标量值——该模式下索引缓冲不被更新，输出停留在 Duplicate 初始化的 0；L0_004 的垃圾索引则来自未初始化的 int64 输出缓冲被原样写出。修复：先 `Duplicate` 把行号广播成张量，再用张量-张量模式 Select：

修复前：

```cpp
// Update running_idx
LocalTensor<float> runIdx = runningIdxBuf.template AllocTensor<float>();
LocalTensor<float> rowIdxFloat = rowIdxFloatBuf.template AllocTensor<float>();
Duplicate(rowIdxFloat, static_cast<float>(r), tileA0Aligned);

// Select(TS): runningIdx = cmp ? runningIdx : r_float
Select(runIdx, cmpMask, runIdx, rowIdxFloat, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileA0Aligned);
```

修复后：

```cpp
// Update running_idx
LocalTensor<float> runIdx = runningIdxBuf.template AllocTensor<float>();
LocalTensor<float> rowIdxFloat = rowIdxFloatBuf.template AllocTensor<float>();
Duplicate(rowIdxFloat, static_cast<float>(r), tileA0Aligned);
PipeBarrier<PIPE_V>();

// Select(TT): runningIdx = cmp ? runningIdx : r_float
Select(runIdx, cmpMask, runIdx, rowIdxFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileA0Aligned);
```

## 设计决策

- 选择"广播 + TT 模式"而不是给 TS 模式传真正的标量：TT 模式与值路径的 Select 一致、行为已验证；TS 模式在本硬件上的确切签名（标量参数类型、掩码语义）当时未查证，贸然再试成本高。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` + run 包安装 + C++ ST（NPU 实测）。
- 相关文件：`op_kernel/arch35/cummin.h`（ProcessAraScan 向量流水线的索引更新调用点）、`tests/st/test_aclnn_cummin.cpp`（L0_003/L0_004 用例与比对输出）。

## 验证信号

验证来自同一开发会话的 NPU 前后对照：修复前 L0_003 `indices: 394944 mismatches out of 396180`（golden=1 actual=0 模式）（来源平台：950；证据等级：tool_observed）；修复后 L0_003 `[PASS] values: all 396180 elements match / [PASS] indices: all 396180 elements match`（来源平台：950；证据等级：tool_observed）；L0_004 索引失配数从 24542 降至 26（残余失配由另一根因覆盖，见同批《向量Compare/Select小A0结果错，改标量扫描》卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只证明"TS 模式 + 张量实参"不产生更新；TS 模式配标量实参是否可用未验证（推断可用，未实测）。
- TT 模式修复要求额外一个广播缓冲（rowIdxFloatBuf）与一次 Duplicate，占用 UB 与向量指令带宽。
- 其他 CANN 版本的 Select 实现可能不同，未验证，推断可迁移。
- 相邻卡片区分：Select 目标与源张量别名（dst==src）先写后读破坏源值是另一根因，见 [Select dst==src 别名破坏张量输入值卡](select_dst_src.md)；count>64 的掩码复用错位见 [arch35 Select VSEL_CMPMASK_SPR 每 64 元素复用卡](arch35_select_vsel_cmpmask_spr_mask_reuse_count_over_64.md)；向量流水线中 value 与 index 共用同一 NaN 合成掩码的完整模式见 [累计极值 NaN 语义须用 self-NE 合成选择掩码卡](ordered_compare_alone_cannot_propagate_nan_prefix_state.md)。

**未决项**：

- **优先项**：查证本硬件/CANN 版本 Select 张量-标量模式的官方签名（标量参数是否必须为标量值、掩码方向语义），把"TS+张量实参不更新"从实测结论升级为 API 语义结论。
## 坏实践

- 见到"值对、索引全 0"先怀疑索引算术（Cast/偏移）——本例根因在条件更新指令的模式选择；判别方法是看失配索引是否恰好等于初始化值。
- 按记忆拼 Select 参数而不核对 SELMODE 语义与参数形态。

## 误诊

- 无。
