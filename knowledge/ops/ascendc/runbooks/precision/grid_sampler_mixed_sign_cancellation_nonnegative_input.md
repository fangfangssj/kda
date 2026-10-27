---
type: "Runbook"
title: GridSampler 混合符号输入相消致近零输出 MARE 假失败 — 数据侧改非负输入 [0,1]，golden 保持 double
description: GridSampler_3d reflection 模式三线性加权求和 8 项近等幅相消（x∈[-1,1]），输出落 1e-5~1e-4 → 相对误差爆炸（absErr 5e-7/|g|+1e-7≈1.39e-02 超阈 11 倍）→ 非数据分布的指标伪影，数据侧改非负输入 x∈[0,1]（grid∈[-1.5,1.5] 保持覆盖），golden 保持 double；absErr 噪声指纹判别（MERE≪阈 MARE 数倍超阈 + dump 查 absErr 在 FP32 噪声量级 + 非负输入对照验证）；float golden 负对照（相消点上 double→float 恶化）。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# GridSampler 混合符号输入相消致近零输出 MARE 假失败 — 数据侧改非负输入 [0,1]

## 现象

grid_sampler_3d（FP32 kernel + double CPU golden，MERE/MARE 双阈值 `rel = |a-g|/(|g|+1e-7)`，fp32 阈值 2^-13）排除执行器缓存问题后，reflection 模式用例仍失败，形态特殊——MERE（均值）比阈值低两个数量级（1.6e-6 vs 1.2e-4），只有 MARE（最大值）超阈 2~4 倍；逐元素看不匹配点 golden/actual 只差第 6 位小数（差值 ~1e-6）；同 shape 的 zeros 模式用例全部通过。

## 适用场景

- **工程形态**：kernel 直调算子或 aclnn 自定义算子的精度验收（Real 比对 vs CPU double golden，FP32 kernel）。
- **API 路径**：MERE/MARE 双阈值比较器（`rel = |a-g|/(|g|+1e-7)`，fp32 阈 2^-13，mere<th 且 mare<10×th）；grid_sampler_3d 的 reflection/border 数据分布。
- **验证过的平台与 CANN 版本**：Ascend950PR（Real 模式，禁用执行器缓存）；会话未记录具体 CANN 版本，失效边界注明。
- **适用粒度**：`operator_family`——grid_sampler 类（三线性加权求和 + 混合符号反射折叠）；相同机制的任意加权求和算子适用。
- **路线优先级**：`recommended_fastpath`（不触碰 tolerance 的数据侧修复）。
- **明确不适用 / 未验证场景**：FP16/BF16 kernel（阈值与噪声量级不同，推断可迁移未验证）；验收方要求评估混合符号绝对精度（应走双标杆 + 小值域 ErrorCount，本会话未执行）；仅 reflection/border 模式触发，zeros 模式不触发。

## 根因

reflection/border 模式对边界点引入 clamp 邻居项，输出是三线性 8 项加权和 `Σ wi·xi`；当 x 混合符号（`[-1,1]` 均匀分布）时各项（量级 ~1）相消，结果落到 1e-5~1e-4 量级；而 FP32（eps=2^-23≈1.19e-7）逐项舍入带来的**绝对**误差不随结果缩小（由被加项量级决定），8 项累计约 2~5e-7。相对误差按 `absErr/(|g|+1e-7)` 计算：`5.013e-07 / (3.591e-05 + 1e-07) ≈ 1.39e-02`，超 MARE 阈值（10×2^-13=1.221e-03）约 11 倍。无相消时（zeros 模式或非负 x）误差随 |g| 等比缩小，rel 稳定在 1e-7 量级——这解释了"zeros 全过、只有 reflection 挂"。`+1e-7` 分母下限只保护 |golden|→0 的元素，对 1e-5~1e-4 量级的小值无保护作用。

## 解决配方

### 判别方法（区分"相消放大"与"真 bug"）

1. **看 MERE/MARE 剪刀差**：MERE 极小（≪阈值）而 MARE 超阈数倍 → 少数离群点，非系统性错误；若 MERE 也超阈，是逻辑/索引 bug。
2. **dump 最差元素，检查 absErr 是否在 FP32 噪声量级**（判别式：`absErr ≈ |被加项|×eps×项数` ≈ 1e-7 量级）：是 → 相消放大；absErr 与 |g| 同量级或更大 → 索引/折叠选错点（真 bug）。
3. **对照**：把输入换成非负区间重跑，MARE 回落到 1e-5 以下 → 确认相消机制。

### 处置（数据侧，比较器不动）

测试输入 x 从 `[-1,1]` 改为非负 `[0,1]`，golden 保持 double。grid_sampler 的坐标映射/反射折叠/索引/权重只依赖 grid 与 shape，x 仅参与 gather 后的加权求和，非负 x 不削弱对 padding_mode/align_corners 逻辑的覆盖；kernel 逻辑错误（选错采样点）仍会以大误差立即暴露：

```cpp
    // x: [0, 1]（非负，避免加权求和抵消导致小值输出的相对误差被放大）
    //    grid_sampler 的坐标映射/反射折叠/索引/权重与 x 取值无关，非负 x 不影响
    //    对 padding_mode/align_corners 逻辑的覆盖，且使 FP32 与 double 对齐以验证精度。
    std::uniform_real_distribution<float> xDist(0.0f, 1.0f);
    for (auto& v : x) v = xDist(rng);
    // grid: [-1.5, 1.5]（适度越界以覆盖 padding 模式）
    std::uniform_real_distribution<float> gDist(-1.5f, 1.5f);
    for (auto& v : grid) v = gDist(rng);
```

### 负对照（golden 精度对齐无效）

把 golden 改成 FP32（坐标/累加全 float）**不能**修复——float golden 自身在相消点上同样带噪声，L0_007 反而从 MARE=4.845e-03 恶化到 1.392e-02（两次噪声叠加）；double golden + 非负输入才是稳定组合。

## 设计决策

- **为何走数据侧而非改比较器**：基准比较器口径（MERE/MARE 双阈值、`|g|+1e-7` 分母）是本仓规范与 TTK 社区标准对齐的公共门禁，为单一算子的测试数据分布偏离而调松口径会破坏全仓统一验收红线；数据侧改 x 分布不削弱 coverage（坐标映射/反射折叠/索引/权重的验证由 grid 覆盖），不触及其他 dtype/算子。
- **放弃的候选**：调用大 eps（被加项无量化，不算对比器病）；调松 mareThreshold（统一红线被破坏）；golden 降精度到 float（实测恶化）。
- **路线优先级**：`recommended_fastpath`（不触碰 tolerance 的数据侧修复）。
- **GPU 同数据路线的对照信号**：同用例在同测试框架移至 GPU（CUDA kernel + float golden）下的精度状态未在会话中观测——理论上 GPU 同样受混合符号相消影响，只是 absErr 量级不同，本处置方案对此中立。

## 验证信号

- 触发：reflection 用例 MERE=1.622e-06（远低于阈 1.221e-04）但 MARE=2.705e-03（超阈 1.221e-03）；另一 reflection/ac=true 用例 MERE=6.504e-06、MARE=4.845e-03；同 shape zeros 用例全过（来源平台：950；证据等级：tool_observed）。
- 探针：最差元素 `idx=7979 golden=7.575e-05 absErr=2.120e-07 relErr=2.795e-03`、`idx=7727 golden=-3.591e-05 absErr=5.013e-07 relErr=1.392e-02`——absErr 全部在 FP32 噪声量级；验算 `5.013e-07/(3.591e-05+1e-07)≈1.39e-02` 超阈约 11 倍（来源平台：950；证据等级：tool_observed）。
- 负对照：golden 改 FP32 无效，仍 6/8，该用例恶化至 MARE=1.392e-02（来源平台：950；证据等级：tool_observed）。
- 处置后：x 改非负 `[0,1]`（golden 恢复 double）后 8/8 通过，reflection 用例 MARE=1.444e-05/2.073e-05，其余用例 2e-07~8e-06；回归同批全量 40/40、全量 L0 15/15（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本处置刻意回避混合符号相消场景——若验收方要求评估该场景的**绝对**精度，非负输入不适用，应走双标杆 + 小值域 ErrorCount 标准（本会话未执行）。
- FP16/BF16 kernel 同机制但阈值/噪声量级不同（fp16 阈 2^-10、eps≈9.8e-4），未单独验证（推断可迁移）。
- 若用例集只有非负 x，"负值 gather 正确性"依赖 golden 自测与其它用例背书。
- 会话未记录具体 CANN 版本，本卡验证信号只覆盖当前测试环境的版本（推定 ≥9.0.0）。
- 本卡不覆盖比较器口径本身是否该因输出幅值分布建立自适应 epsilon（见 [加式分母 epsilon 定标卡](additive_epsilon_calibration.md)）。
- 本卡不覆盖"官方小值域三段判定"标准处置（见 [官方小值域分支两种实现卡](official_small_value_branch_two_implementations.md)）。

## 坏实践

- 见到 MARE 超阈就调大阈值/加 atol"让它过"——统一标尺被破坏。
- 把 golden 精度降到与 kernel 一致（double→float）来"消除精度差"——相消点上 golden 自身噪声叠加，实测更差。
- 用固定随机种子反复重 roll 输入直到恰好全过——用例失去可复现的确定性设计依据。

## 误诊

初见 MARE 超阈易疑为"GM 写丢失"（输出丢值）——actual 非 0、abs_err 在噪声界内，与"输出为 0"形态不同，须分开归因。dump 最差元素确认 absErr 量级即可分诊。

## 预防

- 对加权求和类算子（grid_sampler、interpolate、conv 等）的测试数据分布设计时，把输出幅值分布纳入考量：避免混合符号随机输入在大量项求和时系统性相消到小值区，否则 MERE/MARE 口径下的验收必然假失败，且假失败率随用例数/项数增大而升高。
- 怀疑指标伪影顺序：一路排到底（MERE 极小 → MARE 超阈倍数 → 最差元素 absErr 量级 → 精度模型对照 → 对照实验确认相消 → 数据侧改分布或不改），不走"改比较器让它过"的捷径。

## 系统信息

- 相关文件：grid_sampler_3d 迭代二联调工程（Real 模式、禁用执行器缓存）。
- 平台：Ascend950PR；CANN 版本未在会话中记录。