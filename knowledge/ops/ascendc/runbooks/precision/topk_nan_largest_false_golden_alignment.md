---
type: "Runbook"
title: torch.topk 最小值 NaN 不排前：golden 需对齐
description: PyTorch torch.topk 在 largest=False 时不把 NaN 排前，与"NaN 恒排最前"的算子语义/C++ golden 不一致，含 NaN 最小值用例报 MERE=nan；PyTorch golden 需按算子语义做 NaN 填 -inf 后处理。
tags: [task.testing, topic.precision, topic.compute.ordering]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-16T11:11:42Z'
platforms: ["950"]
---

# torch.topk 最小值 NaN 不排前：golden 需对齐

## 现象

为自定义 TopK 算子（语义：NaN 无论升降序恒排最前）搭建 PyTorch ST 测试工程（torch_npu 接入，CPU golden + NPU 实测比对）后，L0 全量 40 用例中唯一失败的是「含NaN/Inf fp16 topk 最小值」用例（fp16 [64,1024]，k=10，dim=-1，largest=False，数据注入约 5% NaN）：

```
    dtype=float16, shape=[64, 1024], k=10, dim=-1, largest=False, pattern=nan
    [FAIL] y 比对(MERE/MARE): MERE=nan, MARE=nan (th=9.77e-04, 640/640 elems)
```

对照组「含 NaN，largest=True」用例通过（MERE=0.00e+00）。写出最小复现探针（同 seed 生成数据，分别跑 CPU torch.topk 与 NPU 算子）后得到决定性观测：

```
NaN count in x: 3276 / 65536
golden y[0,:5]: [-0.998046875, -0.9951171875, -0.994140625, -0.994140625, -0.9921875]
golden NaN count in y: 0 / 640
npu y[0,:5]: [nan, nan, nan, nan, nan]
npu NaN count in y: 640 / 640
golden row0 has NaN: False
npu row0 has NaN: True
```

即 golden（torch.topk）的 y 完全不含 NaN，NPU 算子的 y 全是 NaN。当时 C++ ST 的同语义用例（含 NaN largest=False）已 110/110 通过——C++ golden 与 NPU 算子行为一致，只有 PyTorch golden 不一致。

## 适用场景

- 工程形态：自定义 aclnn 算子的 PyTorch ST（torch_npu + torch.ops 注册适配层 + CPU golden 比对）。
- API 路径：`torch.topk`（golden 参考）、`torch.nan_to_num`/`Tensor.gather`（对齐处理）；算子侧为自定义 TopK（语义：NaN 恒排最前，与 largest 无关）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，PyTorch 2.7.1+cpu / torch_npu 2.7.1.post5 / Python 3.13。
- 适用粒度：operator_family——所有"NaN 恒排前"语义的 TopK/排序类算子用 torch.topk 做 golden 时。
- 路线优先级：recommended_fastpath（golden 对齐是唯一正确做法；跳过用例或放宽比对均为禁用项）。
- 明确不适用 / 未验证场景：largest=True 含 NaN（torch.topk 已是 NaN 排前，直接用，对照组通过为证）；其他 PyTorch 版本的 torch.topk NaN 行为未验证（本卡观测基于 2.7.1）；输入同时含 NaN 与 -inf 时 NaN→-inf 填充会与原生 -inf 混淆（本算子测试模式将 NaN 与 -inf 分开注入，未触发）。

## 根因

PyTorch 2.7.1 的 `torch.topk` 对 NaN 的排序是不对称的：`largest=True` 时 NaN 排最前，`largest=False` 时 NaN **不**排前（被排到末尾，k 截断后 golden y 不含 NaN）。而算子需求（REQUIREMENTS.md §8.1）与 C++ golden 的语义是"NaN 无论升降序均排最前"（C++ golden 的比较器对 NaN 显式 `return true` 排前）。NPU 算子实现了需求语义（NaN 排前，k=10 全取到 NaN），PyTorch golden 直接调 `torch.topk` 则体现了 torch 的不对称语义。两边 y 一个全 NaN、一个无 NaN，逐元素相对误差出现 `|正常值 - NaN| / ... = NaN`，MERE/MARE 均为 nan。golden 与被测件语义不一致，属 ST 脚手架缺陷，不是算子缺陷。

## 解决配方

修复 golden.py 的 `compute_golden`：无 NaN 或 largest=True 时直接用 torch.topk；largest=False 含 NaN 时把 NaN 填充为 -inf（使其在升序中排最前），用 torch.topk 取 idx，再从原始 x gather 取值（保留 NaN）。

修复前（golden.py 初版，直接透传 torch.topk）：

```python
def compute_golden(x: torch.Tensor, k: int, dim: int, largest: bool):
    """计算 topk 的 golden 结果。

    Args:
        x: 输入张量（任意 dtype，CPU 上计算）
        k: topk 数量
        dim: 排序维度（支持负索引）
        largest: True 返回最大 k 个（降序），False 返回最小 k 个（升序）

    Returns:
        (y, idx): y.dtype = x.dtype，idx.dtype = int64
    """
    y, idx = torch.topk(x, k=k, dim=dim, largest=largest)
    return y, idx
```

修复后（verbatim 取自修复 edit 事件）：

```python
def compute_golden(x: torch.Tensor, k: int, dim: int, largest: bool):
    """计算 topk 的 golden 结果。

    NaN 排前语义：NaN 在 topk 中无论 largest true/false 均排最前（与 NPU 算子、
    C++ golden 和需求文档 REQUIREMENTS.md §8.1 对齐）。
    PyTorch torch.topk 在 largest=True 时已把 NaN 排前，但在 largest=False 时
    **不**把 NaN 排前，因此对 largest=False 含 NaN 的输入需要特殊处理：
    将 NaN 填充为 -inf（使 NaN 排最前），用 torch.topk 取 idx，再从原始 x gather
    取值（保留 NaN）。

    Args:
        x: 输入张量（任意 dtype，CPU 上计算）
        k: topk 数量
        dim: 排序维度（支持负索引）
        largest: True 返回最大 k 个（降序），False 返回最小 k 个（升序）

    Returns:
        (y, idx): y.dtype = x.dtype，idx.dtype = int64
    """
    # 无 NaN 或 largest=True 时直接用 torch.topk（torch.topk largest=True 已 NaN 排前）
    has_nan = x.is_floating_point() and torch.isnan(x).any()
    if not has_nan or largest:
        return torch.topk(x, k=k, dim=dim, largest=largest)

    # largest=False 含 NaN：把 NaN 填充为 -inf 使其排最前，取 idx 后从原始 x gather 取值
    x_filled = x.nan_to_num(float('-inf'))
    _, idx = torch.topk(x_filled, k=k, dim=dim, largest=False)
    y = x.gather(dim, idx)
    return y, idx
```

配套要点（同一 ST 工程内已被证明必要）：

- y 比对须做 NaN/Inf 掩码：`both_nan = torch.isnan(g) & torch.isnan(a)`（同号 Inf 同理）排除后再算 MERE/MARE，否则任一侧含 NaN 即 MERE=nan；全部元素被掩码排除时（两边同 NaN）按通过处理。
- gather 校验（`x.gather(dim, idx)` 与 y 比对）同样用 NaN 感知比较，避免 NaN!=NaN 恒 False 的误报。

## 设计决策

- 为什么修 golden 而不是改算子：算子语义来自需求文档且 C++ ST 已按该语义验收（110/110），torch.topk 的 largest=False 行为才与需求描述不符；golden 的职责是复刻被测件规格。
- 为什么用"NaN 填 -inf + gather 原值"而不是手写排序：复用 torch.topk 的正确排序与索引语义，仅纠正 NaN 的排序位置；gather 从原始 x 取值保证 y 中 NaN 保留、idx 指向真实位置。
- 已知歧义与取舍：若输入同时含 NaN 与 -inf，填充后二者不可区分（idx 可能指向原生 -inf 而非 NaN）；本算子的测试模式（nan 与 -inf 分开注入）不会触发，卡内如实标注。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0，PyTorch 2.7.1+cpu，torch_npu 2.7.1.post5（conda Python 3.13）。
- 相关文件：
  - `tests/st/torch/golden.py`——修复主体（compute_golden）。
  - `tests/st/torch/compare.py`——NaN/Inf 掩码比对（both_nan/both_pos_inf/both_neg_inf）。
  - `tests/st/torch/test.py`——用例与执行器（gen_data 的 nan 模式注入 ~5% NaN）。
  - `tests/st/test_aclnn_bench_no_harness_top_k.cpp`——C++ golden（NaN 排前语义对照，CompareYFp16 的 NaN 判等）。

## 预防

- 用 torch 原生算子做自定义算子 golden 前，先对每个"特殊值 + 方向/极性"组合做语义对照小样（NaN/±Inf × largest/sorted 等），本例正是 largest=False × NaN 这一组合失配。
- golden 语义必须锚定需求文档或被测件规格，而不是"torch 怎么做就怎么对"。

## 验证信号

以下信号来自该算子 PyTorch ST 工程的联调环节：失败用例定位 → 探针分离 golden/NPU 行为 → 修 golden → 单用例与全量回归。

- 修复前：「含NaN/Inf fp16 topk 最小值」用例 `MERE=nan, MARE=nan (640/640 elems)`，L0 全量 35 执行 34 通过 1 失败（来源平台：950；证据等级：tool_observed）。
- 探针观测：x 含 NaN 3276/65536；golden（torch.topk largest=False）y NaN 计数 0/640、y[0,:5] 为正常最小值；NPU 算子 y NaN 计数 640/640、y[0,:5] 全 NaN（来源平台：950；证据等级：tool_observed）。
- 修复后：该用例 `MERE=0.00e+00, MARE=0.00e+00`（640 元素两边同 NaN 被掩码，有效比对点 0，按通过），单用例 1/1 通过（来源平台：950；证据等级：tool_observed）。
- 回归：golden.py CPU 自测 11/11 PASS；L0 全量 35/35 通过（跳过 5 个 dim!=last）；最终 L0+L1 全量 736 用例执行 577 全通过（跳过 159 个 dim!=last）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 输入同时含 NaN 与 -inf 时填充法有索引歧义（未解决，测试模式未触发）；如需支持，须用掩码显式先取 NaN 的索引。
- torch.topk 的 NaN 行为是版本相关的（本卡基于 PyTorch 2.7.1 实测），升级 torch 后需重跑语义对照小样（未验证，推断其他版本可能不同）。
- 算子侧语义若改为"跟随 torch"（NaN 在 largest=False 排后），本卡的对齐方向需反转——以需求文档为准。

**未决项**：

- NaN 与 -inf 共存时的填充歧义处理（掩码先取 NaN 索引）未实现。
- torch.topk 各版本 NaN 行为差异未系统梳理（仅 2.7.1 实测）。
## 坏实践

- 含 NaN 用例失败时直接跳过或对 NaN 位置免比对（不查两侧 NaN 分布是否一致）——会掩盖真实的排序语义缺陷。
- golden 与被测件语义不一致时先怀疑算子——本例 C++ ST 已通过的事实即证明算子符合规格，差异在 golden。

## 误诊

- 初步分析曾怀疑 NPU 算子在 largest=False 时"不把 NaN 排前"（与结论相反）；探针（golden NaN 0/640 vs npu NaN 640/640）直接推翻——不排前的是 torch.topk，排前的是算子。特殊值语义问题必须用双向探针定方向，不能凭记忆推断 torch 行为。
