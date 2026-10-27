---
type: "Runbook"
title: "MERE 阈值 AND（MARE 阈值 OR 逐元素兜底）混合判定"
description: "浮点 ST 比对采用 PASS = MERE 达标 AND（MARE 达标 OR 全部元素通过逐元素 atol/rtol 兜底）的混合判定：MARE 只对 |golden|>atol 元素计算，PASS 行的 MARE 可超 MARE 阈值（fp16 最大 2.88e-02、fp32 125 条靠兜底通过）；读日志勿据单行 MARE 判失败，改回纯双阈值会误杀合法用例。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# MERE 阈值 AND（MARE 阈值 OR 逐元素兜底）混合判定

## 现象

解读自定义分组矩阵乘算子迭代三全量 ST 日志（687 条，Real NPU vs CPU golden，Ascend950PR）时发现：**PASS 行的 MARE 可以超过 MARE 阈值**。fp16 的阈值打印为 `thresh=9.77e-04`（MERE 阈值 2^-10，MARE 上限为其 10 倍 = 9.77e-03），而仅统计 PASS 行得到的各 dtype 最大 MARE：

```text
=== 正常通过用例的最大 MARE（仅 PASS 行）===
fp16 max MARE (PASS only): 1.80e-02 / 2.76e-02 / 2.88e-02
bf16 max MARE (PASS only): 7.81e-03 (×3)
fp32 max MARE (PASS only): 3.52e-02 / 4.33e-02 / 4.95e-02

=== fp32 超 MARE 阈值但通过（atol+rtol 兜底）的用例数 ===
125
```

即 fp16 有 PASS 用例 MARE=2.88e-02 > 9.77e-03（MARE 上限），fp32 有 125 条用例靠兜底判定通过。若按"MERE/MARE 双阈值必须同时达标"的朴素理解读日志，会误判这些 PASS 为判定逻辑 bug 或数据错误。

## 适用场景

- 工程形态：自定义算子 ST 精度比对的设计与实现（C++/PyTorch 测试脚手架），以及 ST 日志的结果解读。
- API 路径：自建比对逻辑（MERE/MARE 社区标准 + 逐元素混合判定）；浮点 dtype 阈值表（fp16=2^-10、bf16=2^-7、fp32=2^-13）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（C++ ST Real 全量 + PyTorch golden 自测）。
- 适用粒度：platform_wide（浮点比对语义与平台无关；该判定是本项目 harness 的设计选择而非 CANN 内置行为——其他项目须自带实现）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：整型输出（走精确比对，无兜底）；纯双阈值（TTK stat_rel_err 原生口径）的评测链不适用本解读规则。

## 根因

判定逻辑不是纯双阈值：PASS = `MERE 达标 AND (MARE 达标 OR 全部元素通过逐元素混合判定)`。MARE 只对 `|golden| > atol` 的元素取最大值，长 K 链累加（fp16 大 K）等场景下单点最大相对误差天然偏大，但若每个元素都满足 `abs(a-g) <= atol` 或 `rel_err <= rtol`（atol=Threshold、rtol=10×Threshold），用例仍应判过——纯双阈值会把这类合法用例误杀。

## 解决配方

完整判定实现（PyTorch 版，与同项目 C++ ST 判定逻辑一致；阈值表 fp16=2^-10、bf16=2^-7、fp32=2^-13）：

```python
FLOAT_THRESHOLDS = {
    torch.float16: 2 ** (-10),       # ~0.000977
    torch.bfloat16: 2 ** (-7),       # ~0.00781
    torch.float32: 2 ** (-13),       # ~0.000122
    torch.float64: 2 ** (-13),       # ~0.000122
}


def _get_threshold(dtype: torch.dtype) -> float:
    """根据 dtype 获取 MERE 阈值"""
    return FLOAT_THRESHOLDS.get(dtype, 2 ** (-13))


def _compare_float(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """
    浮点精度比对，按 MERE/MARE 阈值 + 逐元素混合判定

    与 C++ ST 测试完全一致的判定逻辑:
      1. MERE < threshold (平均相对误差达标)
      2. MARE < 10*threshold OR 所有元素通过逐元素混合判定
         逐元素混合判定: abs(a-g) <= atol OR rel_err <= rtol
         atol = threshold, rtol = 10*threshold
    """
    if golden.numel() == 0:
        return True  # 空 tensor 直接通过

    golden_f = golden.flatten().double()
    actual_f = actual.flatten().double()

    threshold = _get_threshold(golden.dtype)
    mare_threshold = 10.0 * threshold
    atol = threshold
    rtol = mare_threshold
    eps = 1e-7

    abs_diff = (actual_f - golden_f).abs()
    abs_golden = golden_f.abs()
    rel_err = abs_diff / (abs_golden + eps)

    # MERE: 平均相对误差
    mere = rel_err.mean().item()

    # MARE: 最大相对误差（仅对 |golden| > atol 的元素计算）
    mask_significant = abs_golden > atol
    if mask_significant.any():
        mare = rel_err[mask_significant].max().item()
    else:
        mare = 0.0

    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold

    if mere_pass and mare_pass:
        return True

    # 逐元素混合判定: 每个元素 abs(a-g) <= atol OR rel_err <= rtol
    elem_pass = (abs_diff <= atol) | (rel_err <= rtol)
    all_elem_pass = elem_pass.all().item()

    passed = mere_pass and (mare_pass or all_elem_pass)

    if not passed:
        print(f"    MERE: {mere:.6e} (阈值: {threshold:.6e}) "
              f"{'PASS' if mere_pass else 'FAIL'}")
        print(f"    MARE: {mare:.6e} (阈值: {mare_threshold:.6e}) "
              f"{'PASS' if mare_pass else 'FAIL'}")
        print(f"    逐元素混合判定: {'PASS' if all_elem_pass else 'FAIL'}")
        # 打印最大差异位置
        diff = abs_diff
        max_idx = diff.argmax().item()
        print(f"    最大差异位置: [{max_idx}] Golden={golden_f[max_idx].item():.6f}, "
              f"Actual={actual_f[max_idx].item():.6f}, "
              f"diff={diff[max_idx].item():.6e}")

    return passed
```

读日志口径：`(thresh=` 后打印的是 MERE 阈值，MARE 上限是其 10 倍；PASS 行的 MARE 超限不代表判定异常，须回看该用例是否走了逐元素兜底。

## 设计决策

- 为何选混合判定而不是放宽容差：MERE（平均）必须严格达标控制整体精度；MARE（单点最大）允许被逐元素兜底替代，避免长 K 累加等场景下单点相对误差误杀整条用例——放宽容差（改 atol/rtol 数值）会同时削弱两层约束，属测试侧放松，不采用。
- MARE 只对 `|golden| > atol` 的元素计算：接近 0 的 golden 会让相对误差分母失真（加 eps=1e-7 缓解），先行排除。
- PyTorch 版与 C++ ST 版保持逐语义一致：两套 harness（C++ Real / PyTorch 接入）结果须可比，避免"同一算子在两套测试里判定不同"。
- 当前方案属 recommended_fastpath（该算子族的验收既定方案）。

## 验证信号

来源说明：以下信号来自迭代三全量 ST 日志统计与 PyTorch ST 工程的 golden 自测记录（Ascend950PR，CANN 9.1.0）。

- C++ ST Real 全量：L0 37/37、L1 647/650 首跑 + 3 条偶发重跑通过（有效 687/687）；其中 fp32 经 atol+rtol 兜底通过的用例 125 条、fp16 PASS 行最大 MARE=2.88e-02（来源平台：950；证据等级：tool_observed）
- PyTorch 版同逻辑实现后 CPU golden 自测 7/7 通过（含 fp16/bf16×fp32 bias/空组/split_item=0 不均匀分组）（来源平台：950；证据等级：tool_observed）
- 失败路径诊断输出（MERE/MARE/逐元素三级 FAIL 打印 + 最大差异位置）在用例失败时给出可定位信息（来源平台：950；证据等级：tool_observed）

## 失效边界

- MERE 无兜底：平均相对误差超阈值即 FAIL，本判定不救。
- 判定语义依赖阈值表与 10× 系数，迁移到其他精度标准（如商用标准）须整体替换而非只改一处。
- 整型输出走 `torch.equal` 精确比对，不适用本卡。
- 混合判定的引入动机（fp16 大 K 精度问题）出自更早开发片段的 issue 分析，本卡未重新复核该分析。
- 其他项目/平台未验证，推断可迁移（比对语义自包含）。

**未决项**：

- 无。
## 坏实践

- 把 MARE 超阈的 PASS 用例当判定 bug 去"修"比对器（改回纯双阈值）：会把 125+ 条合法用例误杀为失败。
- 为让 MARE 达标而调大全局 atol/rtol：属测试侧放松，削弱 MERE 层约束，掩盖真实回归。
- 读日志时把 `(thresh=9.77e-04` 当成 MARE 阈值（实际是 MERE 阈值，MARE 上限为 10×），从而误判大量 PASS 行"超阈"。

## 误诊

- 见现象节：朴素双阈值理解会把 PASS 行的 MARE 超限误读为判定逻辑 bug 或数据错误。
