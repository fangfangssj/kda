---
type: "Runbook"
title: "MERE 过而 MARE 爆炸：golden 近零与精度模型不一致"
description: "fp16 算子 PyTorch ST 中 MERE 全 PASS 而 MARE 巨大（8~32）、最大绝对差位置相对误差却正常（≈1e-3）——三联征判别为 golden 近零处分母过小 + golden 与 kernel 精度模型不一致（torch fp16 输入路径 ≠ fp32 计算路径）；golden 统计探针锚定，处置=同精度模型 golden + fp32 比对空间 + 按输出 dtype 取阈。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# MERE 过而 MARE 爆炸：golden 近零与精度模型不一致

## 现象

为一个 AscendC 自定义算子（Conv3D filter 梯度反向，fp16 输入输出，kernel 内 float acc 累加后单次舍回 fp16，Ascend950PR）开发 PyTorch ST：golden 用 CPU 的 `F.grad.conv3d_weight`（fp16 输入），比对用社区标准 MERE/MARE（分母 |golden|+eps，eps=finfo(fp16).tiny=6.1e-5，阈值 2^-10，MARE 阈值 10×）。首轮全量运行出现一致的反常模式：所有用例 MERE 全部 PASS（5.1e-4~8.9e-4 < 9.77e-4）而 MARE 全部 FAIL，且数值大得离谱——首个用例 MARE=8.43、另两例 12.5/31.9（阈值仅 9.77e-3）。但打印的最大差异位置却是大数值且差异极小：`[20670] Golden=-259.000000, Actual=-259.250000`，该处相对误差 = 0.25/259 ≈ 9.7e-4，完全正常——MARE 的大值必然来自别的位置（golden 接近 0 的地方）。

## 适用场景

- 工程形态：算子 PyTorch ST / 精度比对（低精度算子输出 vs PyTorch CPU golden）。
- API 路径：`F.grad.conv3d_weight`（aten::convolution_backward）作 golden；自研 MERE/MARE 比对器。
- 验证过的平台与 CANN 版本：Ascend 950PR（ascend950），CANN 9.1.0，torch 2.7.1+cpu。
- 适用粒度：operator_family（收缩/累加类 fp16 算子 + 跨精度 golden 比对普遍适用）。
- 路线优先级：recommended_fastpath（先探针定性再动比对口径，避免盲目放阈）。
- 明确不适用 / 未验证场景：golden 与 kernel 精度模型完全一致（如自写同实现 golden，MERE/MARE 应为 0，此判别模式不会出现）；MERE 也超阈的场景（先按真实精度问题排查，不要套用本规则）。

## 判别方法

触发信号（三联征）：① MERE 全部 PASS 且离阈值不远；② MARE 全部 FAIL 且大 2~3 个数量级（8~32 vs 9.77e-3）；③ 打印的最大绝对差位置上相对误差正常（≈1e-3）。三者同现 ⇒ MARE 大值来自近零 golden 位置，不是真实精度错误。

可执行探针（对本算子失败用例参数，实录命令要点）：

```python
torch.manual_seed(42)
x_cpu = torch.randn(N, Cin, D, H, W, dtype=torch.float16)
grad_cpu = torch.randn(N, Cout, D_out, H_out, W_out, dtype=torch.float16)
golden = conv_3d_backprop_filter(x_cpu, grad_cpu, ...)          # fp16 输入 golden
print('golden zeros count:', (golden == 0).sum().item(), '/', golden.numel())
print('golden near-zero (<1e-3):', (golden.abs() < 1e-3).sum().item())
print('golden abs min:', golden.abs().min().item(), 'abs max:', golden.abs().max().item())
x_f32 = x_cpu.float(); grad_f32 = grad_cpu.float()
golden_f32 = conv_3d_backprop_filter(x_f32, grad_f32, ...)       # fp32 输入 golden
diff = (golden - golden_f32.half()).abs()
print('fp16 golden vs f32->fp16 golden max diff:', diff.max().item())
print('mismatch positions:', (diff > 0).sum().item())
```

原始观测（fp16 输入、filter 形状 [128,64,3,3,3]、共 221184 元素）：

```
golden zeros count: 0 / 221184
golden near-zero (<1e-3): 2
golden abs min: 0.0003757476806640625 abs max: 274.0
fp16 golden vs f32->fp16 golden max diff: 0.125
fp16 golden vs f32->fp16 golden mismatch positions: 59686
```

解释：golden 存在近零值（abs min=3.76e-4），MARE 分母 |golden|+6.1e-5 在这些位置过小，微小绝对偏差被放大成 8~32 的"相对误差"（量级演算：|diff|≈5e-4、分母≈6.1e-5 时 5e-4/6.1e-5≈8.2，与 MARE=8.43 吻合）；同时 fp16-input golden 与 fp32-input golden 在 59686/221184 个位置不同、max diff 0.125——PyTorch 的 fp16 输入路径与 fp32 计算路径并非同一实现，golden 自身就带有与其精度模型相关的离散性。因此 MARE 的爆炸是比对度量在近零处的病态 + golden/kernel 精度模型差异的叠加，不代表算子算错。

处置方向（按验证程度排序）：

1. **golden 与算子同精度模型（同 dtype 输入生成 golden），在 float32 空间比对，阈值按算子输出 dtype 取**（本算子最终采用，L0 由全 FAIL 转 5/5 PASS；阈值绑定细节见[比对阈值随 golden dtype 漂移卡](compare_threshold_bound_to_output_dtype.md)）。
2. **混合误差度量**：分母取 max(|golden|, 1.0)，即 |golden|≥1 用相对误差、|golden|<1 用绝对误差/1.0（本轨迹实测把 MARE 从 8.43 压到 3.03e-2，仍超 10×Threshold 参考阈）：

```python
def _compute_relative_errors(golden_f: torch.Tensor, actual_f: torch.Tensor) -> torch.Tensor:
    """
    计算逐元素相对误差（混合误差度量）

    当 |golden| >= 1.0 时：相对误差 = |actual - golden| / |golden|
    当 |golden| <  1.0 时：相对误差 = |actual - golden| / 1.0  （以 1.0 为基准）

    混合度量避免 golden≈0 时相对误差爆炸（CANN 社区标准 MARE 的已知局限）。
    """
    abs_diff = (golden_f - actual_f).abs()
    abs_golden = golden_f.abs()
    # 分母：|golden| 与 1.0 取大值，避免 golden≈0 时分母过小
    denom = torch.clamp(abs_golden, min=1.0)
    return abs_diff / denom
```

## 设计决策

- 判别顺序上先跑 golden 统计探针再改度量：探针把"近零值数量/最小值/两种 golden 的差异"一次定性，避免在真实精度问题与度量病态之间误判。
- golden 精度模型选择"与算子一致（fp16 输入）+ float32 比对空间"而非"纯 fp32 golden"：后者会把 golden 与 kernel 的模型差异全算到算子头上（fp32 golden 路线下 MERE 升到 9.9e-4 逼近阈值）。
- 当前判别流程属 recommended_fastpath。

## 验证信号

来源说明：以下信号来自 PyTorch ST 在 NPU 上的首轮全量运行、诊断探针与处置后的复测（Ascend950PR，CANN 9.1.0）。

- 触发信号（首轮）：`MERE: 6.836086e-04 (阈值: 9.765625e-04) PASS` + `MARE: 8.428572e+00 (阈值: 9.765625e-03) FAIL` + `最大差异位置: [20670] Golden=-259.000000, Actual=-259.250000`（同模式用例 MARE=1.25e+01、3.19e+01）（来源平台：950；证据等级：tool_observed）
- 探针观测：golden 零值 0/221184、近零(<1e-3) 2 个、abs min=3.76e-4；fp16 golden vs f32→fp16 golden max diff=0.125、59686 位置不同（来源平台：950；证据等级：tool_observed）
- 处置 2 效果：混合度量后 MARE 8.43 → 3.027344e-02（仍超 9.77e-3 参考阈）（来源平台：950；证据等级：tool_observed）
- 处置 1 效果：同精度模型 golden + fp32 比对空间 + fp16 阈值后，首个用例 PASS `MERE=3.125183e-04`，全量 L0 5/5 PASS（来源平台：950；证据等级：tool_observed）

## 失效边界

- 判别三联征只在"MERE 本身通过"时成立；若 MERE 也超阈，先按真实精度问题排查，不要套用本规则。
- 混合度量（clamp min=1.0）改变了 MERE/MARE 的语义（|golden|<1 区间退化为绝对误差判定），与社区标准公式不再逐字一致；本轨迹实测它单独不足以让 MARE 过 10×Threshold 参考阈。
- golden 近零值密度高的算子（如大量零填充/掩码输出）病态更重，本卡的数值边界不直接迁移。
- 其他平台未实测，未验证，推断可迁移（度量病态与 golden 模型差异与平台无关）。

**未决项**：

- MARE 在近零 golden 场景的规范度量（社区标准层面的修订或官方混合度量口径）待跟踪。
- 本算子混合度量下 MARE 仍超参考阈（3.03e-2 vs 9.77e-3），其可接受性最终靠同实现 golden 的 C++ ST MERE=0.00 背书；若无该背书需另立判据。
## 坏实践

- 看到 MARE 巨大就直接放宽容差或删 MARE 判定——先确认 MERE 与最大绝对差位置。本轨迹中 MARE 判定被改为"参考(超标)"不参与 pass/fail，属放宽判定口径（社区标准为 MERE+MARE 双判），当时在代码检视报告中被登记为偏离项。采用此口径必须显式声明并给出 MERE 侧的独立佐证（如同实现 golden 的 C++ ST MERE=0.00）。
- 用 `finfo(dtype).tiny` 当 MARE 分母的 eps 就以为解决了近零问题——量级演算表明它只能压到 ~1e2 级，压不到阈值内。

## 误诊

- 首轮看到全量 MARE FAIL 时一度怀疑"算子精度不达标"；探针显示 MERE 通过、最大绝对差位置相对误差正常、且 golden 自身两种精度路径就差 0.125，才改判为度量病态 + golden 模型差异。

## 预防

- 比对器默认附带 golden 统计探针（零值/近零计数、abs min）与"最大绝对差位置 + 该位置相对误差"的成对打印，让三联征一眼可见。
- 低精度算子的 ST 设计中，golden 的输入 dtype 与算子一致、比对在 float32 空间、阈值按算子输出 dtype，三件套写成模板。
