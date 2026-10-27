---
type: "Runbook"
title: "L2 异常用例 PARAM_COUNT_MISMATCH 为预期失败"
description: "TTK aclnn 黑盒跑批中 dtype/format/dim 异常用例（L2）的 attributes 为空，而 aclnn 头文件把 dim 声明为必填 int64_t 位置参数，TTK 组不出参数即报 PARAM_COUNT_MISMATCH——属异常用例预期失败，须与真实精度失败分开统计。"
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:21:08Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# L2 异常用例 PARAM_COUNT_MISMATCH 为预期失败

## 现象

softmax 基准算子（Ascend 950，CANN 9.1.0）TTK aclnn 模式黑盒验收 520 条用例（含 3 条 L2 异常用例：`aclnnBenchHarnessSoftmax_float64_random_x_dtype_exception`（x dtype=float64 不支持）、`aclnnBenchHarnessSoftmax_float16_random_x_NCHW_format_exception`（format=NCHW 不支持）、`aclnnBenchHarnessSoftmax_float16_random_x_dim_exception`（dim 异常））。终局 26 条失败中，这 3 条 L2 用例的失败形态与其余 23 条 bf16 精度失败**不同**——result.csv 里 precision 与 status 均为：

```
precision=PARAM_COUNT_MISMATCH, status=PARAM_COUNT_MISMATCH
```

而非 `FAIL`；且三条用例在源 CSV 里的 `attributes` 列均为空（`attrs=''`）。首次看到容易误判为 TTK harness 或 golden 插件把参数搞坏了。

## 适用场景

- 工程形态：TTK aclnn 模式黑盒精度验收的结果判读（用例集含异常/负向用例）。
- API 路径：`ttk aclnn` result.csv 的 `precision`/`precision_status` 列；用例 CSV 的 `attributes` 列；算子 aclnn 头文件签名。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR_9579），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow 级（TTK aclnn 异常用例的结果语义）。
- 路线优先级：recommended_fastpath（结果分流的第一步）。
- 明确不适用 / 未验证场景：正常用例（attributes 齐全）不会出现该形态；TTK kernel 模式的异常用例语义未验证。

## 判别方法

1. **触发**：黑盒跑批部分失败，失败行 `precision=PARAM_COUNT_MISMATCH`（非 FAIL/GOLDEN_FAILURE 字面量）。
2. **探针**：对 result.csv 按 `precision` 值分类计数（python csv 模块，勿用 awk 切列——引号内逗号会错位）；回源用例 CSV 核对失败行的 `attributes` 列是否为空。
3. **观测**（本会话实测值）：3 条 L2 异常用例全部 `PARAM_COUNT_MISMATCH`、`attributes` 均为空；L2 层通过率 0/3。
4. **解释**：异常用例**设计上就是负向测试**（无效 dtype/format/dim），CSV 里不带 attributes；而本算子 aclnn 头文件把 `dim` 声明为必填的裸 `int64_t` 位置参数（非 Optional 指针），TTK `build_args` 按 `attributes.get(name, default)` 取不到值、组不出完整参数表，遂报 PARAM_COUNT_MISMATCH。这是异常用例的预期失败路径——它测的就是"非法输入应被拒绝"，与真实精度失败（`precision=FAIL`）性质完全不同，统计时必须分流。

## 设计决策

- 无设计取舍（结果判读规则）。

## 预防

- 黑盒结果汇报先按 `precision` 值分流（PARAM_COUNT_MISMATCH / GOLDEN_FAILURE / FAIL 各自计数），再按 dtype/层统计通过率；异常用例单列，不混入精度通过率。
- 设计 L2 异常用例时明知 attributes 为空会触发 PARAM_COUNT_MISMATCH 即可，不必为其补属性——失败本身是断言。

## 验证信号

以下信号来自该算子黑盒验收会话。

- 终局：total=520 pass=494 fail=26 skip=0；26 条 = 3 条 L2 `PARAM_COUNT_MISMATCH` + 23 条 bf16 `precision=FAIL`（来源平台：950；证据等级：tool_observed）。
- 三条 L2 用例明细：dtype_exception / NCHW_format_exception / dim_exception，源 CSV `attributes` 均为空，result.csv 三行均 `precision=PARAM_COUNT_MISMATCH, status=PARAM_COUNT_MISMATCH`（来源平台：950；证据等级：tool_observed）。
- L2 层通过率 0/3（0.0%，预期失败）；分层统计排除 L2 后 fp16 192/192、fp32 172/172、bf16 130/153（来源平台：950；证据等级：tool_observed）。
- 头文件证据：`aclnnBenchHarnessSoftmaxGetWorkspaceSize(const aclTensor *x, int64_t dim, const aclTensor *out, ...)`，`dim` 为裸 int64_t 必填位（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "空 attributes + dim 必填 → TTK 组不出参数"的因果链是解释性推断（头文件签名 + build_args 源码 + 空属性行为三方一致），TTK 源码中 PARAM_COUNT_MISMATCH 的确切抛出点未逐行核实。
- 若算子的异常维度属性在头文件里是 Optional 指针（可为 null），空 attributes 的表现可能不同（未验证）。

**未决项**：

- TTK 源码中 PARAM_COUNT_MISMATCH 的确切产生点（参数计数校验的哪一层）未核实；不影响判读结论。
## 坏实践

- 把 PARAM_COUNT_MISMATCH 与真实精度 FAIL 混在一个失败率里（26/520 的笼统失败率会同时高估精度问题、掩盖异常用例语义）。
- 看到 PARAM_COUNT_MISMATCH 就去改 TTK 或给异常用例补 attributes——异常用例就该失败。

## 误诊

- 无（本会话首轮汇报即正确分流）。
