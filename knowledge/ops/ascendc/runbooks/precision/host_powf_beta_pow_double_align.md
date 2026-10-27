---
type: "Runbook"
title: "偏置校正 β^t 用 powf 单精度与 golden double 漂移"
description: "AdamW 类算子 host tiling 以 C 库单精度 powf 预计算 β^t，golden（torch/Python）按 double 计算 beta**step；大 step 下误差被偏置校正分母（~10×）与输出相消点两级放大，mere 极小而 mare 超阈；host 预计算改 double pow 对齐。"
tags: [task.debugging, topic.precision, topic.optimizer, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# 偏置校正 β^t 用 powf 单精度与 golden double 漂移

## 现象

在 Ascend950PR（CANN 9.2.0）上为 BenchHarnessApplyAdamW（AdamW 单步更新）修白盒精度失败。两个大 step 用例（`step=1000`、`beta1=0.99`、`beta2=0.9999`、`lr=0.1`、`weight_decay=0.5`、`epsilon=0.0001`）精度 FAIL 而 perf PASS：

- fp32 用例（shape (4607,)，数据范围 (-10,10)）：TTK stat_rel_err 报 `mere=1.04e-05, mare=7.76e-03, threshold=0.0001220703125, reason: 'threshold exceeded: mare(7.76e-03>=0.001220703125)'`——均值误差极小、个别元素相对误差大。
- bf16 用例（5D shape，数据范围 (-1,1)）：同轮 `mare=0.153` 超阈值 0.078125（bf16 阈值 2^-7），且失败与否随随机种子波动（换种子可变 mare=3.75 或通过）。

当时代码状态：host 侧 TilingFunc 把偏置校正所需的 β^t 预计算进 TilingData，用的是 C 库单精度 `powf`；golden（torch/Python）按 `beta ** step` 在 double 域计算标量后再作用于 fp32 张量。kernel 侧全程 fp32 寄存器运算。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发 + TTK kernel 白盒精度评测（stat_rel_err / isclose 两种比对标准均适用本现象判读）。
- API 路径：`op_host/*_tiling_host.cpp` 的 host 标量预计算（`powf`/`pow` + TilingData 标量字段），kernel 侧经 TilingData 消费 β^t。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0。
- 适用粒度：operator_family——一切「host 预计算标量 + kernel fp32 消费 + golden 以 Python double 算标量」的优化器/归一化类算子（AdamW/Adam/LAMB 等含 β^t 偏置校正、指数衰减类）。
- 路线优先级：recommended_fastpath（host 侧一行精度对齐，无额外开销）。
- 明确不适用 / 未验证场景：golden 本身定义为 fp32 逐步累加语义的算子（对齐目标不同）；step 为小值（如 1）时 β^t=β 本身、pow 精度无影响（本轨迹 step=1 用例本就通过）。

## 根因

host 预计算 β^t 的浮点精度与 golden 不一致：golden 的偏置校正分母是 `fp32(1.0 − double(β)**double(t))`（Python/torch 标量在 double 域算完再作用到 fp32 张量），kernel 拿到的却是 `1.0f − powf(fp32(β), fp32(t))`（全 fp32）。误差经两级放大：

1. **偏置校正分母放大**：设 β^t 的相对误差为 ε，则 v̂ = v_new/(1−β^t·(1+ε)) 的相对误差 ≈ β^t·ε/(1−β^t)。step=1000、β2=0.9999 时 β2^1000 ≈ 0.905、1−β2^1000 ≈ 0.095，放大系数 ≈ 0.905/0.095 ≈ 9.5；fp32 powf 的 ε（数个 ulp，~1e-7 量级）被放大到 v̂ 的 ~1e-6 量级。
2. **输出相消点放大**：y = var − s·lr·update，当某元素 var ≈ s·lr·update（输出接近 0）时，~1e-6 量级的输入端差异在输出端按 |var|/|golden| 放大（stat_rel_err 分母为 |golden|+1e-7），可放大到远超阈值——这正是「mere 极小（1.04e-5）而 mare 大（7.76e-3）」的判别特征：只有个别相消点元素失配。

注：直估第 1 级只有 ~1e-6，单看它解释不了 mare 7.76e-3；第 2 级相消放大才是把误差推过阈值的关键（同种子修复前后 mare 实测减半佐证了该误差源的真实贡献，见验证信号）。

## 解决配方

思路：host 预计算标量的浮点域与 golden 对齐——`powf`（float）改 `pow`（double），结果 cast 回 float 交给 kernel。

修复前（`op_host/bench_harness_apply_adam_w_tiling_host.cpp`，host attr writing 段，verbatim）：

```cpp
    // step → beta^t via powf (fp32, aligns golden.py beta ** step); step=1 → betaPow=beta
    float step_f  = static_cast<float>(static_cast<double>(in.step));
    float beta1Pow_f = powf(beta1_f, step_f);
    float beta2Pow_f = powf(beta2_f, step_f);
```

修复后（verbatim）：

```cpp
    // step → beta^t via pow (double precision, aligns golden.py beta ** step which
    // computes in Python double); cast result to float32 for kernel consumption.
    // Using powf (float32) caused precision divergence for large step (e.g. 1000).
    double beta1_d = static_cast<double>(in.beta1);
    double beta2_d = static_cast<double>(in.beta2);
    double step_d  = static_cast<double>(in.step);
    float beta1Pow_f = static_cast<float>(pow(beta1_d, step_d));
    float beta2Pow_f = static_cast<float>(pow(beta2_d, step_d));
```

（`beta1Pow_f`/`beta2Pow_f` 随后照旧写入 TilingData 的 float 字段，kernel 侧不变。）

失配定位探针（TTK `--dump-on-fail` 落 npy 后逐元素对照 mere/mare 分布，本轨迹用于确认「个别元素大偏差、均值极小」的相消特征）：

```bash
cd "$TTK" && python3 -c "
import numpy as np
out = np.load('caseNNNN_dyn_output_0.npy'); gold = np.load('caseNNNN_golden_0.npy')
rel = np.abs(out - gold) / (np.abs(gold) + 1e-7)
print('mere', rel.mean(), 'mare', rel.max(), 'worst idx', rel.argmax(), 'gold[worst]', gold[rel.argmax()])
"
```

## 设计决策

- 为什么 β^t 在 host tiling 预计算而不是 kernel 逐 tile 计算：偏置校正是全局标量，只依赖属性（β、step），与元素/tile 无关；host 算一次写 TilingData，kernel 每 tile 直接消费，避免 device 侧重复超越函数。
- 为什么对齐 double 而不是让 golden 降精度：golden 是精度验收的唯一语义锚点（本算子 DESIGN 规定 golden 与 `torch.optim.AdamW` 严格一致），kernel/host 向 golden 对齐是唯一方向。
- 备选与放弃原因：kernel 内用 fp32 迭代乘（t 次乘法）算 β^t——迭代舍入误差只大不小，且大 step 下次数高，放弃。

## 验证信号

以下信号来自该算子白盒修复会话（Ascend950PR，CANN 9.2.0）：定位 powf 精度差异后改 double pow、重建重跑，同种子前后对照；再经换种子与更换比对标准复核。

- 修复前（powf fp32，同种子）：bf16 大 step 用例 mare=0.153 ≥ 阈值 0.078125 FAIL；fp32 大 step 用例 mare=7.48e-3 ≥ 上限 1.22e-3 FAIL、mere=9.91e-6（来源平台：950；证据等级：tool_observed）
- 修复后（double pow，同种子复跑）：bf16 大 step 用例转 PASS；fp32 大 step 用例 mare 降至 3.55e-3（约减半，stat_rel_err 下仍超 1.22e-3 上限）（来源平台：950；证据等级：tool_observed）
- 换种子复核：bf16/fp16 大 step 用例在 stat_rel_err 下仍随种子波动（曾出现 mare=3.75 的失配），证明 β^t 只是误差源之一、fp32 kernel 全链与 double golden 的残余差异仍在相消点暴露（来源平台：950；证据等级：tool_observed）
- 按 DESIGN 的 rtol=0.05（fp32）/0.1（fp16/bf16）走 `--compare close`（np.isclose，rtol/atol 取自 CSV precision_tolerances）并固定种子 42 复跑：全量 33/34 通过（剩 1 例为 bf16 极值 NaN/Inf 用例，另卡）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本修复只消除 β^t 这一个误差源：kernel 其余链路（Sqrt/Div/乘加顺序）仍为 fp32，与 double golden 在输出相消点上残余 ~1e-3 量级 mare，在 stat_rel_err 的 dtype 硬件阈值（fp32 2^-13=1.22e-4、bf16 2^-7）下仍可能失配且种子相关；验收口径须按算子 DESIGN 的 rtol 定比对标准。
- β^t 写入 TilingData 时仍有一次 double→float 舍入——与 golden「double 标量作用 fp32 张量」的舍入点一致，故对齐后残余差异不来自此处。
- 仅在 Ascend950PR + CANN 9.2.0 实测；`powf`/`pow` 的具体精度属 libc 实现，其他平台的 ulp 行为未验证（机制相同，未验证，推断可迁移）。

**未决项**：

- fp32 用例残余 mare 3.55e-3 的逐元素归因（相消点元素占比、|golden| 分布）未做——可复用上面的 npy 探针补齐，用于确认残余误差全部来自 fp32 链路而非其他源。
- stat_rel_err 的 dtype 硬件阈值与算子 DESIGN rtol 的口径差异属评测配置选择，本卡不判定哪种更合理。
## 坏实践

- host 预计算与 golden 语义对齐的标量时随手用单精度内建（`powf`/`expf` 等）——误差虽小，但会被偏置校正分母（β^t/(1−β^t)，大 step 下 ~10×）和输出相消点（可达 |var|/|golden| 倍）两级放大。
- 只看 mere 不看 mare 分布定位精度问题——mere 极小会让人误判「整体没问题」，漏掉相消点个例。

## 误诊

- 初步数量级估算认为 powf 误差（~1e-6）不足以解释 mare 7.76e-3，转而怀疑 AscendC Sqrt/Div 精度与乘加顺序；实测对齐 double 后同种子 mare 减半、bf16 用例转 PASS，证明 β^t 精度是真实主要误差源之一——估算时漏算了输出相消点放大这一级。
- 同会话曾把种子相关的失败集合变化当作「修复引入回归」；实际是随机输入换了相消点分布，需固定种子做前后对照。
