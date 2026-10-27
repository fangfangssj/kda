---
type: "Runbook"
title: "GroupNorm 归一化输出近零元素 — MARE/MERE 分离判别、混合分母与 torch 工程完整比对器"
description: "归一化输出大量 golden≈0：MARE 超阈而 MERE 低于阈 2~3 数量级 + 绝对误差 ~1e-7（fp32 舍入噪声）+ 爆炸机制推导（分母塌缩）+ 反例自检四步判别链；校准取法=混合分母 max(|g|+1e-7, 10×threshold)。torch 工程完整比对器实现（FLOAT_THRESHOLDS 阈值表 + 混合分母 + 整数匹配 + 失败诊断打印）+ 双阈值防误杀实证（fp32 maxMARE=1.430e-04 超过 MERE 阈但低于 10×MARE 阈——单一阈值会误杀）。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# GroupNorm 归一化输出近零元素 — MARE/MERE 分离判别、混合分母与 torch 工程完整比对器

## 现象

GroupNorm 类算子（Ascend950PR + CANN 9.1.0）归一化输出大量 golden≈0 的元素。3 个失败用例 MARE 1.38e-03~3.48e-02 超阈（阈值 1.22e-03）而同批 MERE 2.01e-07~2.39e-07（低于阈值约 3 个数量级），对照组大 shape 用例 MARE 4.64e-04 正常。

## 适用场景

- **工程形态**：归一化输出类算子的精度验收（C++ ST + PyTorch 接入 ST 双工程），MERE/MARE 双阈值。
- **API 路径**：混合分母 `max(|golden| + 1e-7, 10×threshold)`；MERE/MARE 分离判别；torch 完整比对器。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0（同轨迹终态）。
- **适用粒度**：`operator_family`——归一化/零中心化输出类算子；推断 BN/LN 等其他归一化层适用。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用 / 未验证场景**：非近零输出算子混合分母不生效（退化为常规相对误差）；整数输出走精确匹配分支；bool 等 dtype 未验证。

## 根因

归一化输出大量 golden≈0 元素，纯相对误差 `|a-g|/(|g|+1e-7)` 在 |g|→0 时分母塌缩到 1e-7，绝对差（~1e-7）被放大成 rel≈1——但该绝对差本身在 fp32 舍入噪声量级，不是 kernel 缺陷。

## 解决配方

### 四步判别链（验证后进入度量校准）

1. **MARE 超阈而 MERE 低于阈值 2~3 个数量级**（如 MARE 1.38e-03~3.48e-02 超阈 1.22e-03，同批 MERE 2.01e-07~2.39e-07）→ 失败集中在极少数近零元素而非整体精度劣化；
2. **绝对误差量级核对**：全体元素平均绝对差 ~1e-7，与 fp32 舍入噪声同量级；
3. **爆炸机制推导**：分母 |g|+1e-7 → 1e-7，绝对差 1e-6 即 rel≈10；
4. **反例自检**：小 shape/输出远离 0 的用例全过。

### 校准取法

```python
denom = max(|golden| + 1e-7, 10 * threshold)
# |g| ≥ absScale 的元素仍按相对误差，近零元素按 abs_err/(10*threshold) 计
# 绝对差超过 10×threshold 的近零元素依然 FAIL（防松绑）
```

矫正后 MARE 类失败清零（31→36）。

### torch 工程完整比对器

完整 verbatim 代码（`tests/st/torch/compare.py`）：

```python
FLOAT_THRESHOLDS = {
    torch.float16: 2 ** (-10),       # ~0.000977
    torch.bfloat16: 2 ** (-7),       # ~0.00781
    torch.float32: 2 ** (-13),       # ~0.000122
    torch.float64: 2 ** (-13),       # ~0.000122
}

def _compare_float(golden, actual):
    threshold = _get_threshold(golden.dtype)
    abs_scale = 10.0 * threshold
    abs_err = (actual_f - golden_f).abs()
    denom = torch.max(golden_f.abs() + 1e-7, torch.tensor(abs_scale, dtype=torch.double))
    rel_err = abs_err / denom
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    passed = (mere < threshold) and (mare < 10.0 * threshold)
    if not passed:
        print诊断信息（MERE/MARE/最大差异位置/前5不匹配元素）
    return passed
```

### 双阈值防误杀实证

fp32 全量实测 maxMARE=1.430e-04，已超过 MERE 阈值 th=2^-13≈1.22e-04、但低于 MARE 阈值 10×th≈1.22e-03——即"只设单一 MERE 阈值（或把 MARE 上限也设为 th）"会把该批次误杀；`MERE < th 且 MARE < 10×th` 的双阈值标准正确放行。

## 设计决策

- **为何选择混合分母而非 max(|g|, th)**：max(|g|, th) 在 |g|<th 区按绝对差 atol=th 判定，混合分母 |g|+1e-7 的绝对值项（1e-7）保留极小 g 上的纯相对误差信息——GroupNorm 的均值池化使大多数元素恰好在 th 附近，混合分母的双重保护（+1e-7 防极小区、absScale 保判定阈值语义）通过实测。
- **为何做四步判别链才进入度量校准**：防止将真缺陷的 MARE 超标（如 kernel 选错节点导致部分元素全 0）误判为指标伪影。

## 验证信号（同一轨迹 C++ 与 PyTorch 双工程全量实跑）

- C++ Real 模式 ST 全量 716/716 通过；L1 600 例各 dtype fp32 maxMERE=2.940e-07 / maxMARE=1.430e-04、fp16 maxMERE=2.480e-04 / maxMARE=4.900e-04、bf16 maxMERE=1.490e-03 / maxMARE=3.890e-03（来源平台：950；证据等级：tool_observed）。
- fp32 maxMARE=1.430e-04 已超 MERE 阈值 1.22e-04、但低于 10×MARE 阈值 1.22e-03——反证"只设单一 MERE 阈值"会误杀（来源平台：950；证据等级：tool_observed）。
- PyTorch 接入全量 641/641（fp32 239 / fp16 190 / bf16 212）用同一比对器全部通过（来源平台：950；证据等级：tool_observed）。
- 判别输入：3 个失败用例 MARE 1.38e-03~3.48e-02 超阈（阈值 1.22e-03）而同批 MERE 2.01e-07~2.39e-07；混合分母校准后 MARE 类失败清零、通过数 31→36（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 混合分母弱化了"近零元素绝对偏差很大"的可检性（被钳位吸收）——对近零区域绝对精度敏感的算子应补充独立 atol 判定通道（未验证）。
- 整数输出走精确匹配分支；bool 等既非浮点又非整数的 dtype 路径未验证。
- 非近零输出算子混合分母不生效（退化为常规相对误差），阈值需按算子精度要求另定。
- 钳位尺度取 `10×threshold` 与当前 MARE 阈值体系绑定，换阈值体系须同步复核。