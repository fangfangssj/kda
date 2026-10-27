---
type: Runbook
title: "重复采样时间数组零离散 ≠ 稳定性证据"
description: "重复采样的时间数组只有在样本之间可能不同时才能证明稳定性；逐元素合法 + 样本计数不构成离散度。N 份同一数字的中位数是穿着离散度外衣的点估计——诚实状态是 N/A MEASUREMENT_FAILED，不是 PASS/BELOW_THRESHOLD。原 legacy 锚 OL-518。"
tags: [task.testing, topic.performance.analysis]
status: draft
sources: []
created_at: "2026-09-22T05:30:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: [a2, a3, "950"]
aliases: [OL-518]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# 重复采样数组零离散 ≠ 稳定性证据

## 现象

`device_timing_sample_error` 只逐元素校验（正/有限/在界），分母只报 `len(...)`——N 份同一数字两个都过，于是数组只有 ONE 个观测却装成 N 个，中位数是穿着离散度外衣的点估计。案例：4-76 的 `ratio=6.7661`[evidence unverified——原始 verification.json 未归档] 曾被当作"不可回退硬底线"写进 respawn brief，底层数组是 `[432.5849 ×5]` / `[63.9344 ×5]`。

## 处置

- 零离散数组的诚实状态 = N/A `MEASUREMENT_FAILED`，绝不编码成 PASS 或 BELOW_THRESHOLD（不确定的测量不能编码成判决）。
- 离散度检查（`len(set(values)) == 1 and len(values) > 1`）应挂在现有的样本有效性缝上——那个缝本来就有这个判决权。

## 边界

- 相同值是**离散度信号，不是伪造证明**：粗计时器合法地会重复，样本持久化前还会 round 到 4 位小数。读作"无法证明稳定"，引用前重测。
- 交叉：OL-163（对称窗口 + 逃逸到 N/A + operator_details/kernel_details 列的误引警告）、OL-510。

## 适用场景

- **工程形态**：重复采样的时间数组只有在样本之间可能不同时才能证明稳定性；逐元素合法 + 样本计数不构成离散度。N 份同一数字的中位数是。
- **验证过的平台**：a2、a3、Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
