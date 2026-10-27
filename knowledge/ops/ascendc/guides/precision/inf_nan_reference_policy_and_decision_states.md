---
type: Guide
title: NaN/Inf 比较的参考策略与通过、跳过、待查状态
description: 解释旧版 INF_NAN_MODE_ENABLE 映射和比较器真实分支，区分无特殊值、跳过 Inf、符号不一致及第三方同错。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/engine/src/scripts/orchestrator/precision/cannbot_standard/reference/special_cases.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/engine/src/scripts/orchestrator/precision/cannbot_standard/scripts/inf_nan_check.py
  role: implementation
created_at: '2026-09-14T07:15:54Z'
updated_at: '2026-09-14T07:15:54Z'
platforms: []
aliases:
- INF_NAN_MODE_ENABLE
- inf_nan_check
---

# NaN/Inf 比较的参考策略与通过、跳过、待查状态

固定 AOG 比较器 `check_inf_nan_consistency` 以函数参数 `is_before_910a` 选择策略，默认 True。来源把 True 关联到 Ascend910A 及之前、或 910B+ 且 `INF_NAN_MODE_ENABLE=0`，把 False 关联到 910B+ 且该变量为 1；函数本身不读取设备或环境变量。这是来源的策略映射，不是本卡确认的跨所有 CANN 版本硬件行为。调用方必须单独确认实际模式，不能只设环境变量便假定比较器参数随之改变。

比较应区分有限值、NaN、+Inf、-Inf。实现用数值相等或“双方均 NaN”判断匹配，因此 +Inf 和 -Inf 不匹配；这不是 NaN payload 或所有结果位模式的一致性检查。

## 两种策略的实际分支

True 策略只检查 NaN 一致性：两方均 NaN 计 pass，仅一方 NaN 计 fail，其余涉及 Inf 的位置计 skipped。即使 golden=+Inf、actual=1，也可能因 fail=0、need_check=0 返回 `is_pass=True`，同时 `skipped_count=1`。跳过不代表证明数值正确，必须保留状态和被排除范围。

False 策略对含 actual/golden 特殊值的位置按下表处理；benchmark 为可选第三方结果：

| 条件 | 固定实现的结果 |
|---|---|
| golden 为特殊值，actual 与 golden 匹配 | pass，不要求 benchmark 也匹配 |
| golden 为特殊值，actual 不匹配，但 benchmark 匹配 golden | fail |
| golden 为特殊值，actual 不匹配且 benchmark 也不匹配 | 有 benchmark 则 need_check；没有则 fail |
| golden 有限、actual 特殊，actual 与 benchmark 匹配 | pass |
| golden 有限、actual 特殊，benchmark 也是特殊值但与 actual 不匹配 | need_check |
| golden 有限、actual 特殊，benchmark 缺失或不是对应特殊值 | fail |

这比文档或函数说明中的简化规则更具体。例如 golden=+Inf，actual 和 benchmark 都为 NaN，函数仍归 need_check；不能以“actual 与 benchmark 相等”覆盖 golden 为特殊值时的分支。反过来 golden=1，actual 和 benchmark 都为 +Inf，函数可以通过；该结果只符合来源兜底策略，不能证明独立参考意义下的数值正确，更不能用来放行不允许新增 Inf 的算子。

## 状态汇总与观测边界

没有 actual/golden 特殊值时，返回 `has_special_values=False, is_pass=None`，应继续正常数值比较，不能把 None 当作失败或整体通过。benchmark 独自出现特殊值不会扩大该函数的扫描范围。存在特殊值时，聚合条件为 fail_count=0 且 need_check_count=0；need_check 会阻断通过，skipped 则不会。

调用前应核对三组 shape、有效元素与 dtype。实现按 `.flat` 遍历，却没有完整的结构校验；诊断样本仅用 `np.where(...)[0]` 的前十个首维索引，多维数据可能打印整行并丢失完整坐标，不能把它当准确的失败元素定位。实际诊断需保留全部维度坐标、类别及对应输出身份。

验证至少包括无特殊值、双方 NaN、+Inf/-Inf 符号、Inf 对有限值、第三方缺失、三方同错/不同错，以及 True/False 两种参数。特殊值检查和 [小值域计数](small_value_subset_error_counts.md)、正常有限值误差属于不同覆盖范围；只有显式定义的组合合同才能决定整例结果。本卡的分支反例在 CPU 上验证，未验证来源的硬件模式映射。
