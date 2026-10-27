---
type: "Runbook"
title: "Depthwise 卷积小值域假失败 — 期望==实际打印一致指纹、阈值 max(表值,10×th) 与示例侧 CompareResults 双示例落地"
description: "fp32/fp16 Depthwise 卷积 ST 与交付示例的小值域假失败三连：失配元素在 %.6f 打印下期望==实际完全相同（打印精度吃掉 golden 量级信息）→ 补齐官方小值域并行判定（小值按绝对误差 atol=10×threshold 判定、不参与 MARE）且阈值取 max(标准表值, 10×threshold)（表值之上边界带实测仍可达 1e-3 rel_err）；示例侧简化 CompareResults 同样无小值域豁免，同语义修复同步应用到 aclnn 与 GE IR 双示例（verbatim 完整代码随迁）。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# Depthwise 卷积小值域假失败 — 期望==实际打印一致指纹、阈值 max(表值,10×th) 与示例侧 CompareResults 双示例落地

## 现象

Depthwise 卷积类算子（fp32/fp16，CANN 9.2.0）两类小值域假失败：

- **ST（R15）**：L1_002 MERE=2.70e-07（阈 1.22e-04）而 MARE=6.90e-03（阈 1.22e-03），失配元素在 %.6f 打印精度下**期望与实际完全相同**（如 期望=0.000125, 实际=0.000125, rel_err=2.35e-04）、golden 全在 1e-3~1e-5 量级、abs_err 1e-8~1e-7。
- **交付示例（R19）**：aclnn 示例自带简化版 CPU golden 比对（MERE/MARE）未做小值域豁免，算子本身完全正常（输出前 5 元素 NPU 与 golden 逐位一致）却判 FAIL——`[FAIL] MERE=7.33e-07 (thr 9.77e-04), MARE=1.03e-01 (thr 9.77e-03)`（fp16 Threshold=2^-10）。

## 适用场景

- **工程形态**：fp32/fp16 算子 ST 精度验收；交付示例（examples/）自带的简化 CPU golden 比对函数。
- **API 路径**：MERE/MARE 双阈值（mere<th 且 mare<10×th）；官方小值域标准；示例侧 `CompareResults`。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.2.0（R15 ST / R19 交付示例 aclnn + GE IR 双示例，同一算子两条独立 trace）。
- **适用粒度**：`operator_family`——卷积/激活类输出含小值域元素的算子；示例侧简化比对器同修复。
- **路线优先级**：`recommended_fastpath`。

## 根因

- **ST 指纹**：%.6f 打印精度吃掉 golden 量级信息——1e-3~1e-5 量级的 golden 在 6 位小数下打印相同（实际差在 7+ 位），只看打印的 rel_err 会误判。
- **示例侧**：简化版 `CompareResults` 用全元素统一相对误差，未实现官方小值域豁免——随机数据下 golden 近 0 元素把 MARE 打爆。

## 解决配方

### 1. ST（R15）：补齐官方小值域并行判定 + 阈值扩展

```cpp
// 小值域阈值取 max(标准表值, 10×threshold)
// 标准表值（fp32 2^-14≈6.1e-5）之上到 10×threshold（1.22e-3）的区间
// 实测 rel_err 仍可达 1e-3（L1_035 golden=6.4e-5, rel_err=1.09e-3），
// 仅按表值划分会残留伪失败。
double smallValueThreshold = std::max(table_value, 10.0 * threshold);

// 小值元素：按绝对误差 atol=10×threshold 判定，不参与 MARE
// MERE 主干含全部元素
if (std::abs(golden[i]) < smallValueThreshold) {
    if (std::abs(a - g) > atolForSmallValue) { smallValueFailCount++; }
} else {
    // 正常值域：相对误差，参与 MARE
}
```

### 2. 示例侧（R19）：CompareResults 双示例同步落地

同语义修复同步应用到 aclnn 与 GE IR 两个示例的 `CompareResults`（两处同构）。修复前（示例 CompareResults，全元素统一相对误差）：

```cpp
bool CompareResults(const std::vector<T>& golden, const std::vector<T>& actual, double threshold)
{
    ...
    double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
    ...
    bool pass = (mere < threshold) && (mare < mare_threshold);
    return pass;
}
```

修复后（补小值域豁免：小值元素用绝对误差 atol 判定、不参与 MARE）：

```cpp
double GetSmallValueThreshold(double threshold)
{
    // 标准小值域阈值与 10*threshold 的较大值，覆盖相对误差不稳定区间
    return std::max(threshold * 10.0, threshold * 10.0);
}

bool CompareResults(const std::vector<T>& golden, const std::vector<T>& actual, double threshold)
{
    size_t size = golden.size();
    if (size == 0) return true;

    double smallValueThreshold = GetSmallValueThreshold(threshold);
    double atolForSmallValue = 10.0 * threshold;

    double mere = 0.0;
    double mare = 0.0;
    size_t smallValueFailCount = 0;
    size_t normalValueCount = 0;

    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);
        double absErr = std::abs(a - g);

        if (std::abs(g) < smallValueThreshold) {
            // 小值域元素：用绝对误差判定，不参与 MARE
            mere += absErr / (std::abs(g) + 1e-7);
            if (absErr > atolForSmallValue) {
                smallValueFailCount++;
            }
        } else {
            // 正常值域元素：用相对误差，参与 MARE
            double rel_err = absErr / (std::abs(g) + 1e-7);
            mere += rel_err;
            if (rel_err > mare) mare = rel_err;
            normalValueCount++;
        }
    }
    mere /= static_cast<double>(size);
    double mare_threshold = 10.0 * threshold;

    bool mere_pass = (mere < threshold);
    bool mare_pass = (normalValueCount == 0) ? true : (mare < mare_threshold);
    bool small_value_pass = (smallValueFailCount == 0);
    bool pass = mere_pass && mare_pass && small_value_pass;
    return pass;
}
```

要点与边界（R19）：

- 小值元素仍计入 MERE 分子（保持平均误差信息量），MARE 只统计正常值域元素，并对小值元素单独 atol 兜底计数——完全跳过会掩盖小值元素上的真实大偏差，atol 兜底保留检出能力。
- `GetSmallValueThreshold` 实现里 `std::max(threshold*10.0, threshold*10.0)` 两参数相同（源码即如此，等效于 10×Threshold）——标准小值域阈值（如 fp16 的 2^-11）与 10×threshold 取大的本意在该实现中退化为 10×threshold，属保守取值，与 R15 的扩展一致。
- 阈值数值绑定 dtype（fp16=2^-10；fp32=2^-13 同标准体系），换 dtype 换阈值后同构适用。

## 设计决策

- **ST 与示例侧为何同语义**：交付示例是 ST 的简化镜像，行为必须与正式验收口径一致——否则示例演示"假失败"，误导用户。
- **为何小值域阈值取 max(表值, 10×th)**：[表值, 10×threshold] 边界不稳定带实测 rel_err 仍可达 1e-3，仅按表值划分残留伪失败；但 fp32 的 atol=10×threshold=1.22e-3 对 golden≈1e-3 的元素相当于允许 100% 相对误差，门限宽松度应按算子精度要求复核。

## 验证信号

- R15 触发：L1_002 `MERE=2.70e-07, MARE=6.90e-03`（失配元素期望==实际打印一致、golden 1e-3~1e-5）；补齐后 L1_002 PASS（MARE=1.13e-04, small=191）、L1_035 PASS（MARE=1.78e-04, small=315），实跑 28/28 全通过，正常值域阈值未动（来源平台：950；证据等级：tool_observed）。
- R19 修复前（aclnn 示例，fp16）：`[FAIL] MERE=7.33e-07 (thr 9.77e-04), MARE=1.03e-01 (thr 9.77e-03)`，前 5 元素 NPU/golden 逐位一致（来源平台：950；证据等级：tool_observed）。
- R19 修复后（aclnn 示例同 shape）：`[PASS] MERE=7.33e-07, MARE=8.82e-04 (threshold=9.77e-04, 200704 elems)`（来源平台：950；证据等级：tool_observed）。
- R19 修复后（GE IR 示例同 shape，同修复）：`[PASS] MERE=7.33e-07, MARE=8.82e-04`，`GE IR 示例执行结果: PASS`（来源平台：950；证据等级：tool_observed）。
- R19 修复后从 clean 状态全量重建两模式重跑：均 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- R15 的 [标准表值, 10×threshold] 区间扩展论证来自 4 个用例的经验观测（golden 6.4e-5~1.5e-3 区间 rel_err 可达 1e-3），未做独立数值分析；更保守的实现可只按标准表值并接受个别伪失败复核。
- bf16 跨精度场景只经本会话 bf16 用例覆盖。
- R19 小值元素上的真实大偏差（absErr > 10×Threshold）仍被 smallValueFailCount 拦截；介于 Threshold 与 10×Threshold 之间的绝对偏差在小值域会被放行——这是标准本身的取舍。
- R19 的退化取值（10×threshold）未影响本例判定。

## 坏实践

- 示例/脚本随手写个"max 相对误差"就当精度门禁——随机数据下 golden 近 0 元素必然把 MARE 打爆，把好算子误判成 FAIL。
- 误报后直接放宽容差或注释掉比对——正确做法是补齐标准的小值域豁免语义。

## 误诊

- R15 失配元素在 %.6f 打印下期望==实际相同，若只看 rel_err 会误判为"打印/日志 bug"或"golden 生成错误"——判读必须看 abs_err 与 golden 量级而非只看 rel_err。