---
type: "Runbook"
title: 绝对max_diff误比相对阈值致fp16假FAIL
description: 自验脚本把绝对最大偏差直接对比 fp16 阈值 2^-10=0.000977 误判精度 FAIL；2^-10 是 MERE/MARE 相对误差标准的 Threshold 而非绝对误差容限，输出绝对值达 ~1 量级时 fp16 绝对舍入 ~2^-11 完全正常；按社区标准重算 MERE/MARE 后 10 种子全过。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# 绝对max_diff误比相对阈值致fp16假FAIL

## 现象

自定义算子（depthwise conv fp16，Ascend950PR，CANN 9.2.0）经 torch 扩展接入后做交付前自验：验证脚本取 NPU 输出与 CPU fp32 golden 的**绝对**最大偏差对比 fp16 阈值 2^-10：

```
max diff:  0.0039052963256835938
mean diff: 0.00041311283712275326
fp16 threshold (2^-10): 0.0009765625
PASS: False
```

判成 FAIL，且看起来"超阈 4 倍"，像是算子精度有问题。当时算子此前已通过 53/53 ST 与 43/43 PyTorch ST（均按 MERE/MARE 相对误差标准），自验脚本的判据与已验收标准不一致。

## 适用场景

- 工程形态：算子交付/接入后的自验脚本、快速冒烟验证（PyTorch/torch_npu 调用 + CPU golden 比对）。
- API 路径：浮点计算类算子精度社区标准——fp16：MERE < 2^-10 ≈ 9.77e-04 且 MARE < 10×2^-10 ≈ 9.77e-03；小值域（|golden| < 小值域阈值）改绝对误差判定；MERE=mean(|a-g|/(|g|+eps))、MARE=max 同式。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0，torch 2.7.1+cpu + torch_npu 2.7.1.post4（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——精度判据选择，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非浮点计算类算子（随机数生成/非计算类有各自标准）；要求 bit-exact 的场景（比较口径不同，本卡不涉及）。

## 根因

自验脚本把"绝对最大偏差"直接对比 fp16 相对误差阈值 2^-10——该阈值是 MERE/MARE 相对误差标准的 Threshold，不是绝对误差容限；输出元素绝对值可达 ~1 量级时，fp16 表示本身就有 ~2^-11 量级的绝对舍入，绝对 max_diff 到几个 2^-10 完全正常，误用判据必然虚报 FAIL。自验判据必须与验收判据（社区 MERE/MARE 标准）同源，否则自验结果与验收结果不可比。

## 判别方法（解决配方）

正确判据与复算步骤（可直接复用的探针脚本核心）：

```python
import numpy as np

# npu: 算子输出转 fp64；golden: CPU fp32 参考（fp16 算子用 fp32 golden）
eps = 1e-7
abs_diff = np.abs(npu - golden)
rel_diff = abs_diff / (np.abs(golden) + eps)

mere = np.mean(rel_diff)          # MERE = 平均相对误差
mare = np.max(rel_diff)           # MARE = 最大相对误差
threshold = 2 ** (-10)            # fp16 Threshold ≈ 9.77e-04
mare_threshold = 10 * threshold   # ≈ 9.77e-03

# 小值域分析：|golden| < 2^-11 的元素单独看
small_val_threshold = 2 ** (-11)
small_mask = np.abs(golden) < small_val_threshold
normal_mask = ~small_mask
mere_normal = np.mean(rel_diff[normal_mask]) if normal_mask.sum() > 0 else 0.0
mare_normal = np.max(rel_diff[normal_mask]) if normal_mask.sum() > 0 else 0.0

overall_pass = (mere < threshold) and (mare < mare_threshold)
```

判定口径：`MERE < 2^-10` 且 `MARE < 10×2^-10`（小值域元素按标准另行豁免）；绝对 max_diff 只作参考读数，不作为门禁。阈值语义与 TTK stat_rel_err 默认阈值表同源（fp16 2^-10 为相对误差口径），见 [TTK stat_rel_err 默认阈值 fp32 2^-13](ttk_stat_rel_err_default_threshold_table.md)。

## 设计决策

- 判据来源对齐：该算子的 ST 工程、cann-bench 评测器用的是同一 MERE/MARE 社区标准（ops-precision-standard 体系：浮点计算类→社区标准；随机数生成类/非计算类另有标准）——自验脚本判据必须与验收判据同源，否则自验结果与验收结果不可比。
- 绝对误差读数保留在输出里（不删）：排障时有用，但明确标注为参考值。

## 系统信息

- 工具链：CANN 9.2.0；torch 2.7.1+cpu + torch_npu 2.7.1.post4。
- 目标平台：Ascend950PR（DAV_3510），device 0。
- 相关文件：自验脚本（inline python，经 bash 执行）、精度标准参考 `ops-precision-standard/reference/float_compute_community.md` 与 `special_cases.md`（判据事实源）。

## 验证信号

来源说明：问题与验证来自该算子 bench 接入后的自验会话——首版判据误报 FAIL，按社区标准重算并多种子复验，全部为 950 真机实测。

- 误报现场：`max diff: 0.0039052963256835938 / fp16 threshold (2^-10): 0.0009765625 / PASS: False`（来源平台：950；证据等级：tool_observed）。
- 按标准重算（seed=42，fp16 [1,64,56,56]×[64,3,3]）：`MERE=0.000176 < 0.000977 PASS`、`MARE=0.000488 < 0.009766 PASS`；小值元素 30/200704（0.01%），正常值域子集 MERE/MARE 同样 PASS（来源平台：950；证据等级：tool_observed）。
- 多种子稳定性（seed 0~9 共 10 次）：`10/10 seeds passed`，MERE 范围 [0.000176, 0.000177]，MARE 范围 [0.000486, 0.003407]（全部 < 0.009766）（来源平台：950；证据等级：tool_observed）。
- 同算子此前 ST 53/53、PyTorch ST 43/43（MERE/MARE 标准）通过——旁证算子本身精度正常（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判据数值绑定 dtype：fp16 Threshold=2^-10；fp32/bf16 用各自 Threshold（同标准体系换数值），不能拿 fp16 阈值判 fp32 输出。
- 相对误差标准对"golden 接近 0 的元素"必须配合小值域豁免，否则 MARE 侧会反向误报，见 [近零 golden 相对误差 — 分母改用 max(|g|, th)](stat_rel_err_near_zero_floor_denominator.md)。
- 大 value_range 下的抵消点问题：相对误差标准也判不住 fp32 顺序累加的固有 ULP 误差，见 [fp32算子bit-exact仍败于fp64 golden比对](fp64_golden_fp32_cancellation_dtype_limit.md)——三张卡分别覆盖三种典型误判方向。

**未决项**：

- 无。
## 坏实践

- 拿绝对 max_diff 对比相对误差阈值当门禁——判据错位，好算子被误判。
- 误报后第一反应去"修算子精度"而不先核对判据口径与验收标准是否一致。

## 误诊

- 首版脚本附带用 `atol=1e-3, rtol=1e-3` 的混合判据补救判 PASS（`Note: checking atol=1e-3, rtol=1e-3 -> PASS: True`）——碰巧通过但口径仍非标准，属待修正的中间版本；最终以 MERE/MARE 标准重算定论。
