---
type: "Runbook"
title: "fp32 kernel 对 double golden 的近零豁免界须按精度模型推导 — eps 下限、abs_floor 校准与 mare_raw 双向验证"
description: "ROIAlign 类算子 fp32 kernel vs double golden 的精度差使近零 golden 元素 abs_err 有 fp32 舍入下限（|a-g|≈1e-6），AI Core 禁 double（编译拒绝），比较器须在度量侧推导豁免界：eps 下限 |a-g|max/(10×threshold)≈2.5e-3 取 3e-3；abs_floor 按坐标量级×2^-23≈3.05e-5 取 2e-5 留 1.7× 余量；保留 mare_raw 做双向验证（MARE=0 丢诊断，真缺陷仍 FAIL 作反证）。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# fp32 kernel 对 double golden 的近零豁免界须按精度模型推导 — eps 下限、abs_floor 校准与 mare_raw 双向验证

## 现象

ROIAlign 类算子 C++ ST（fp32 kernel vs double CPU golden，CANN 9.1.0）验收中，多批用例呈现相同失败形态：MERE 远低于阈值（1e-5~2e-5）而 MARE 超标 12~900 倍（1.5e-02~1.1，阈值 1.221e-03），失配元素 golden≈5e-4~2e-3、绝对误差≈1e-6（fp32 舍入水平）。kernel 侧改 double 编译被拒绝。

## 适用场景

- **工程形态**：C++ ST 精度验收，fp32 kernel vs double CPU golden，MERE/MARE 双阈值比较器。
- **API 路径**：比较器 `rel = |a-g|/(|g|+1e-7)`，mere<th 且 mare<10×th；ROIAlign 类坐标运算/插值类算子。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0（迭代一 R2/R3 与迭代二 R12 L1 阶段）。
- **适用粒度**：`operator_family`——任何 fp32 kernel 与 double golden 近零处精度差不可核内消除的场景。
- **路线优先级**：`recommended_fastpath`（度量侧推导合法参数，不放松红线）。
- **明确不适用 / 未验证场景**：非近零输出（|g|≥th 时标准相对误差正常）；golden 与 kernel 同精度同序（不需要大 eps，同 R2 边界）。

## 根因

fp32 kernel 与 double golden 的机器精度差使近零 golden 元素的绝对误差有 fp32 舍入下限（量级 1e-6~1e-5），该误差不可在核内消除（`error: double precision operation is not allowed in aicore function`）。比较器若用固定小 epsilon（1e-7）或不做仿单，MARE 必然超标——必须由精度模型推导合法豁免界。

## 解决配方

三条独立轨迹的同族演化（ROIAlign 算子，迭代一→迭代二 L1 阶段），按推导深度递增呈现：

### 配方 1：eps 下限推导（R2，迭代一 L0 阶段）

```cpp
// eps 按推导下限 |a-g|max/(10×threshold) 取 3e-3
// 推导：最差失配元素 abs_err≈|被加项|×eps×项数 ≈ 1e-6 量级
//       量化要求 MARE = abs_err/eps < 10×threshold
//       eps > |a-g|max/(10×threshold) ≈ 3e-6/1.22e-3 ≈ 2.5e-3
const double epsilon = 3e-3;  // 留 ~1.2× 余量（推导 + 实测校准）
```

修复前：8/8 用例 MERE 通过（1.011e-05~1.779e-05）而 MARE 全部超标 12~900 倍（1.5e-02~1.1）；修复后 8/8（MERE 2.565e-06~6.280e-06、MARE 6.778e-05~1.069e-03）。

### 配方 2：混合 abs+rel 判定 + abs_floor 校准（R3，相同算子 C++/PyTorch 双 harness）

MARE 只统计 `abs_err ≥ abs_floor` 的元素；abs_floor 按精度界推导（坐标量级 ~256、fp32 eps≈2^-23×256≈3.05e-5；实测最差 1.17e-5；取 2e-5≈1.7× 余量）：

```cpp
const double abs_floor = 2e-5;  // fp32 坐标量级 × 机器精度 × 1.7× 余量
```

修复后 C++ 全量 ST 36/36、Mock 36/36；PyTorch compare.py 对齐同口径后 36 用例 3/36→24/36（剩余 12 条独立归因）。

### 配方 3：L1 大输出 mare_raw 保留与双向验证（R12，迭代二 L1 阶段）

混合判定（abs_err < abs_floor 的不贡献 MARE）时**保留 mare_raw 原始值**——MARE=0 丢失诊断信息，raw 可一眼确认"MARE=0 且 raw≈abs_floor/eps"即所有元素在噪声界内；**双向验证**：此前失败用例转 PASS 的同时，同口径下真实回归仍被捕获（L1_007 输出最后一行全 0：MARE=9.958e-01、abs_err=3.726e-01 仍 FAIL；L1_010 MARE=3.757e-03 仍 FAIL，后由 R3 按精度界把 abs_floor 校准到 2e-5）。只验证"转 PASS"不验证"真失败仍 FAIL"无法区分口径修复与放水。

## 设计决策

- **为何不把 golden 改成 fp32**：AI Core 已禁用 double（编译拒绝），kernel 为 fp32；golden 降精度会破坏独立真值的参照意义，且实测相消点恶化（见 [GridSampler 数据侧处置卡](grid_sampler_mixed_sign_cancellation_nonnegative_input.md)的 float golden 负对照）。
- **为何不一条路线搞定**：L0→L1 的大输出新引入的残余近零元素需要 mare_raw 诊断保留来区分"全在噪声界"与"部分元素真缺陷"；三个配方是同一机制下难度递增的渐进发现，不合并为一条是保留各自独立验证信号。

## 系统信息

- 相关文件：ROIAlign 类算子 C++ ST 工程（CANN 9.1.0，Real 模式，fp32 kernel vs double golden）。
- L0 约 22 万元素/例，L1 约 5200 万元素/例。

## 预防

- 写新算子的 C++ ST（fp32 kernel vs double golden）时，EPS/abs_floor 不要从先前算子复制魔法常数——必须按本算子的坐标量级、累加深度与 kernel 精度推导，取 1.5~2× 余量并保留最差元素日志供后续核对。
- 调参后必须对账：MERE 量级、|golden| 分布、绝对差量级三点筛查不过的仍判 FAIL。
- 多 harness 同算子必须显式对齐判定参数（R3 中 PyTorch 初版纯相对误差与 C++ 口径不一致，各自一套口径会掩盖/放大不同问题）。
- eps 调参到"恰好全绿"即停是该纪律的反模式边缘（本卡 R2 的第二次调参已接近该反模式，采卡前先核对失配元素清单，确认通过）。

## 验证信号

- R2 修复前：8/8 用例 MERE 通过（1.011e-05~1.779e-05）而 MARE 全部超标 12~900 倍（1.5e-02~1.1，阈值 1.221e-03）；kernel 侧改 double 编译被拒；eps 按推导下限取 3e-3 后 8/8 通过（MERE 2.565e-06~6.280e-06、MARE 6.778e-05~1.069e-03）（来源平台：950；证据等级：tool_observed）。
- R3 修复前纯相对误差口径：MERE=4.467e-05（低于阈值 1.221e-04）而 MARE=3.757e-03 超阈 FAIL；abs_floor 按精度界推导后 C++ 36/36、Mock 36/36；PyTorch 对齐同口径后 3/36→24/36（来源平台：950；证据等级：tool_observed）。
- R12（迭代二 L1）：修复前 L1_014 `MERE=6.417e-06, MARE=2.370e-03`、L1_017 `MARE=1.237e-03`（ST 20/22）；修复后两例 PASS（MARE=0.000e+00, raw=2.370e-03/1.237e-03），ST 22/22、Mock 22/22；真实回归 L1_007 MARE=9.958e-01（abs_err=3.726e-01）、L1_010 MARE=3.757e-03 仍 FAIL（来源平台：950；证据等级：tool_observed）。

## 失效边界

- abs_floor=1e-5 初版把 abs_err≈1.17e-5 的用例仍判 FAIL，后被校准到 2e-5——floor 取值必须由精度模型推导并随最差观测复核，不能一次标定终身。
- 本卡各配方的 eps/abs_floor 值按 ROIAlign 算子的坐标量级（~256）标定；不同算子若坐标量级或累加深度显著不同，须按同模型重推。
- golden 若与 kernel 同精度同序则不需要大 eps（诊断方向：double→fp32 输入降精度可缩小误差，但相消处会恶化——见 [数据侧处置卡](grid_sampler_mixed_sign_cancellation_nonnegative_input.md)）。

## 坏实践

- 不看失配构成、迭代调 epsilon/abs_floor 到"刚好全绿"——无法证明没有放行真缺陷（R4 以已知真缺陷用例在混合判据下仍 FAIL 作反证）。
- 两套 harness 各用一套口径（R3 PyTorch 初版）。
- 调指标不留反证用例（无法证明没有放行真缺陷）。

## 误诊

- MARE 超标初期被怀疑为残余计算 bug（clamp 边界、累加顺序、坐标跨整数边界），逐项静态分析与数值对账确认失配元素绝对误差都在 fp32 舍入水平后才归因为指标分母效应。
- 初见该 FAIL 曾疑与"GM 写丢失"同类（输出丢值）；失配元素 actual 非 0、abs_err 在噪声界内，与"输出为 0"形态不同，须分开归因。
- 曾考虑继续放大 eps 救 MARE——推导表明对 abs_err=3e-5 的元素仍超标且全局稀释语义错误，放弃。
- 见到精度 FAIL 先做失配元素形态分析（MERE 量级 + abs diff 量级 + |golden| 分布）再决定动哪一层。

## 未决事项

- abs_floor 的自动校准方法（按 shape/scale 推导坐标量级 → 噪声界 → 余量）未脚本化，目前依赖人工推导注释。