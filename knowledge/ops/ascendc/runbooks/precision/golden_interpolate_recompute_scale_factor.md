---
type: "Runbook"
title: golden interpolate需显式recompute=True
description: torch interpolate 对 list scale_factor 默认 recompute_scale_factor=None 等效 False（直接按 sf 映射坐标），与"先 floor(in*sf) 再按 output_size 映射"的算子语义不符致 ST 巨错（floor 边界 MERE=4.93）；golden 显式 recompute_scale_factor=True 修复。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: "2026-10-10T02:49:37Z"
platforms: ["950"]
---

# golden interpolate需显式recompute=True

## 冻结原合同的适用边界

下文 `recompute_scale_factor=True` 修复只适用于**权威需求明确要求先 floor 输出大小、再按 size 映射**的原案例。它不是所有 Torch interpolate 的默认修复。若当前验收合同是冻结原 Model.forward 的默认 scalar/list scale_factor 语义，保留该 forward、dtype、保存输入和 grader；不能改 recompute、转 size 或调整分布来获得通过。

固定 scalar 反例：H=1399、scale_factor=1.5、Hout=2098，默认 non-align 比例2/3与 recompute/size 比例1399/2098不同。原 scalar合同必须使用其原比例；本卡历史 list实验不自动授权替换 scalar合同。独立 Torch v2.10.0 CPU固定50的后续验证支持这个合同区分，未给本卡旧配方补做上板资格。[Gather 历史配方的范围与缺陷](../compilation/gather_resize_bilinear_dav3510_recipe.md)。

## 现象

为 Ascend C 自定义算子（resize 双线性，arch35/ascend950，CANN 9.1.0）开发 PyTorch ST（适配层把 aclnn 两段式接口注册为 torch 自定义 op，golden 用 `torch.nn.functional.interpolate`）。39 条用例首跑 36 通过，**3 条失败全部集中在 scale_factor 模式**，其中 floor 边界用例误差巨大（不是精度级别而是语义级别）：

```text
测试 [L1]: L1 sf floor边界 fp32 (99x99, sf=[0.5,0.5]->49x49) ac=false
    MERE: 4.933686e+00 (阈值: 1.220703e-04) FAIL
    MARE: 2.007277e+03 (阈值: 1.220703e-03) FAIL
    最大差异点: Golden=0.328398, Actual=-0.756651   ← 符号都不同
```

另两条（bf16 sf 上采样、fp32 sf 多 batch）为 MARE 临界超阈。当时代码状态：golden.py 的 `resize_bilinear()` 直接把 `scale_factor` 透传给 `torch.nn.functional.interpolate(..., mode='bilinear', align_corners=...)`，未指定 `recompute_scale_factor`；而算子需求语义是"scale_factor 先转换为 output_size = floor(in_size * sf)，再以 output_size 为准做坐标映射"。output_size 模式的全部用例（含相同 shape）都通过。

## 适用场景

- 工程形态：以 PyTorch 算子（`torch.nn.functional.interpolate` 等）为 golden 的 NPU 自定义算子 ST/精度验收。
- API 路径：`torch.nn.functional.interpolate(x, scale_factor=..., mode='bilinear', align_corners=..., recompute_scale_factor=...)`。
- 验证过的平台与 CANN 版本：Ascend 950PR（NPU 实测）+ CANN 9.1.0 + torch 2.7.1（CPU golden）。
- 适用粒度：platform_wide（golden 语义问题，与 NPU 平台无关；torch 2.7.1 实测）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：output_size 模式（size= 参数）不受影响；标量 float scale_factor 的默认行为未单独验证（本例均为 list）；非 interpolate 类算子不适用。

## 根因

PyTorch 2.7.1 中 `interpolate` 的 `recompute_scale_factor=None` 对 **list 型 scale_factor 等效 False**：直接用 `h = (i + 0.5) / scale_factor - 0.5`（align_corners=false 时）做坐标映射；而 `recompute_scale_factor=True` 先算 `output_size = floor(in_size * sf)` 再按 output_size 映射 `h = (i + 0.5) * H_in / H_out - 0.5`。两者在 `in_size * sf` 不是整数时给出**不同的坐标映射**（99x99、sf=0.5 时：True 走 99/49，False 走 1/0.5=2 倍步长），插值结果自然大相径庭。算子语义（按需求文档）是 recompute=True 模型，golden 默认行为却是 False 模型——golden 错了，不是 NPU 算子错了。

## 判别方法

探针脚本（纯 CPU，直接区分三种口径）：

```python
import torch
x = torch.randn(1, 1, 99, 99)
y_default    = torch.nn.functional.interpolate(x, scale_factor=[0.5, 0.5], mode='bilinear', align_corners=False)
y_recompute  = torch.nn.functional.interpolate(x, scale_factor=[0.5, 0.5], mode='bilinear', align_corners=False, recompute_scale_factor=True)
y_outputsize = torch.nn.functional.interpolate(x, size=[49, 49], mode='bilinear', align_corners=False)
print("Default == Recompute:", torch.equal(y_default, y_recompute))    # False
print("Recompute == Output_size:", torch.equal(y_recompute, y_outputsize))  # True
print("Max diff (default vs recompute):", (y_default - y_recompute).abs().max())  # 2.362259864807129
print("Max diff (recompute vs output_size):", (y_recompute - y_outputsize).abs().max())  # 0.0
```

原始观测：Default != Recompute（最大差 2.36）；Recompute 与 output_size 模式逐元素相等（max diff = 0.0）。即 `recompute_scale_factor=True` 才与"先 floor 再按 output_size 映射"的算子语义一致。

## 解决配方

修复前（golden.py——scale_factor 直接透传，语义与算子不符）：

```python
def resize_bilinear(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None,
) -> torch.Tensor:
    """..."""
    if x.dim() != 4:
        raise ValueError(f"ResizeBilinear requires 4D input (N, C, H, W), got {x.dim()}")
    return torch.nn.functional.interpolate(
        x,
        size=output_size,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=align_corners,
    )
```

修复后（scale_factor 分支显式 `recompute_scale_factor=True`，并写明理由）：

```python
def resize_bilinear(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    使用双线性插值调整图像大小（仅 4D 输入）。

    ...

    Note:
        当提供 scale_factor 时，显式设置 recompute_scale_factor=True，与算子行为一致
        （REQUIREMENTS.md §4.1：scale_factor 先转换为 output_size = floor(in_size * sf)，
        再以 output_size 为准进行坐标映射）。
        PyTorch 默认 recompute_scale_factor=None 对 list 类型 scale_factor 等效于 False，
        与算子行为不符，会导致 scale_factor 模式精度比对失败。
    """
    if x.dim() != 4:
        raise ValueError(f"ResizeBilinear requires 4D input (N, C, H, W), got {x.dim()}")
    if scale_factor is not None:
        return torch.nn.functional.interpolate(
            x,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=align_corners,
            recompute_scale_factor=True,
        )
    else:
        return torch.nn.functional.interpolate(
            x,
            size=output_size,
            mode='bilinear',
            align_corners=align_corners,
        )
```

## 预防

- 用 torch 算子做 golden 前，先跑一次三口径对照探针（default / recompute=True / 等价 output_size），确认 golden 语义与算子需求文档一致再批量跑 NPU。
- 涉及 scale_factor 的对齐基线，把 `recompute_scale_factor` 的口径写进 golden 函数 docstring，防止后来者"简化"掉。

## 验证信号

来源说明：来自该算子 PyTorch ST 当轮修复的实测（NPU 上板 + CPU 探针）。

- 误报证据：3 条失败全部在 scale_factor 模式，floor 边界用例 MERE=4.93（语义级巨错）；同 shape 的 output_size 模式用例全过（来源平台：950（NPU 实测）；探针为 CPU torch 2.7.1；证据等级：tool_observed）。
- 探针观测：default vs recompute 最大差 2.362259864807129；recompute vs output_size 最大差 0.0（来源平台：950；证据等级：tool_observed）。
- 修复生效：golden 加 `recompute_scale_factor=True` 后（未改算子、未改容差），floor 边界用例同一位置 Golden=0.281140 / Actual=0.281152（此前 0.328398 vs -0.756651），残余差异降到近零值指标放大级别；配合数据分布调整后 39/39 全通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 实测基于 torch 2.7.1（`recompute_scale_factor=None` 对 list sf 等效 False）；其他 torch 版本默认行为可能变化（未验证，推断需逐版本跑探针确认）。
- 本卡只覆盖 golden 语义对齐；残余的近零值 MARE 放大是另一独立问题（见 [近零 golden 相对误差 — 分母改用 max(|g|, th)](stat_rel_err_near_zero_floor_denominator.md)）。
- 标量 float scale_factor 的默认行为与 list 是否一致未验证。

**未决项**：

- 无（该缺陷已完整闭环）。
## 误诊

- 首见 3 条失败时曾怀疑 NPU 算子 scale_factor 处理路径有 bug（拟对照 output_size 模式排查算子内部）——探针证实算子与 recompute=True 口径一致、错在 golden 默认口径，避免了无谓的算子侧排查。
