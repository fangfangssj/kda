---
type: "Runbook"
title: "conv3d_weight 不支持非对称 pads，golden 报维度错"
description: "CANN 卷积类算子的 pads 是 6 元素（每个空间维前后独立），而 PyTorch F.grad.conv3d_weight 的 padding 只接受 3 元素对称 padding；golden 映射只取前/上/左时非对称 pads 用例在 golden 阶段直接 RuntimeError 维度校验错。非对称 pads 用例须显式 skip 登记或自写逐元素 golden。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# conv3d_weight 不支持非对称 pads，golden 报维度错

## 现象

为一个 AscendC 自定义算子（Conv3D filter 梯度反向，kernel 支持 6 元素 pads：[D_front, D_back, H_top, H_bottom, W_left, W_right]，即前后/上下/左右填充可以不同）开发 PyTorch ST 测试，golden 用 CPU 上的 `torch.nn.functional.grad.conv3d_weight`。把 CANN 语义的 pads 映射成 PyTorch 的对称三元组后跑非对称 pads 用例（如 pads=[1,2,1,2,1,2]，对应 grad 的 D_out=9/H_out=17/W_out=17），在 golden 计算一行直接抛异常，整轮测试中断：

```
  File ".../tests/st/torch/golden.py", line 62, in conv_3d_backprop_filter
    y = F.grad.conv3d_weight(
        x, tuple(filter_size), grad,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
  File ".../torch/nn/grad.py", line 286, in conv3d_weight
    return torch.ops.aten.convolution_backward(
        ...
RuntimeError: Expected a tensor of dimension 5 and tensor.size[2] == 8 but got: dimension 5 and tensor.size[2] = 9
```

即 PyTorch 按"对称 padding=pads[0],pads[2],pads[4]"推导期望的 grad spatial 维度（8），与用例真实传入的 9 不符。对称 pads 的全部用例正常。

## 适用场景

- 工程形态：算子 ST 测试 golden 选型（PyTorch CPU golden 对标 CANN 卷积类算子）。
- API 路径：`torch.nn.functional.grad.conv3d_weight` → `torch.ops.aten.convolution_backward`（PyTorch 2.7.1，CPU）。
- 验证过的平台与 CANN 版本：测试工程运行于 Ascend 950PR 环境（CANN 9.1.0，torch 2.7.1+cpu / torch_npu 2.7.1.post4）；本限制本身是 CPU 侧 PyTorch golden 行为，与设备无关。
- 适用粒度：workflow 级 ST 测试设计知识（何种 pads 语义不能用 conv3d_weight 做 golden；operator_family——一切 6 元素 pads 的 CANN 卷积类算子）。
- 路线优先级：safety_fallback（非对称 pads 用例当前标记 skip；自写逐元素 golden 是根本替代，未在本轮实现）。fallback 声明：跳过损失了非对称 pads 的自动化覆盖，需在验收报告中显式登记。
- 明确不适用 / 未验证场景：2D 的 conv2d_weight、其他 PyTorch 版本未验证（推断同样仅支持对称 padding，未验证）。

## 根因

CANN 卷积类算子的 pads 是 6 元素（每个空间维前后独立），而 PyTorch `F.grad.conv3d_weight` 的 padding 参数只接受 3 元素对称 padding；golden.py 的映射 `padding = (pads[0], pads[2], pads[4])` 只取了前/上/左。当 pads 非对称（front≠back 等）时，PyTorch 按对称 padding 推导的输出/梯度空间维度与用例按 6 元素 pads 推导的维度不一致，aten::convolution_backward 形状校验直接报维度错。这是 golden 实现能力边界，不是算子 kernel 的问题（kernel 侧 6 元素 pads 语义完整实现）。

## 解决配方

已验证的处置：非对称 pads 用例在 PyTorch ST 用例表中显式标记 skip 并注明原因，不再走 PyTorch golden：

```python
    {"id": "L1-08", "level": "L1", "dtype": torch.float16, "skip": True,
     "desc": "非对称pads pads=[1,2,1,2,1,2]（PyTorch conv3d 不支持非对称 padding，跳过）"},
```

同时测试主循环加 try-except（防止单用例异常中断整轮，与大 shape 资源异常的同类保护一致）。根本替代（未实施）：为非对称 pads 自写逐元素 CPU golden（按 6 元素 pads 逐位置判边界累加），或用支持非对称 padding 的参考实现。

## 验证信号

来源说明：以下信号来自 PyTorch ST 全量运行（NPU 环境，golden 在 CPU 计算失败）的实测输出（Ascend950PR，CANN 9.1.0）。

- 崩溃原文（非对称 pads 用例 golden 阶段）：`RuntimeError: Expected a tensor of dimension 5 and tensor.size[2] == 8 but got: dimension 5 and tensor.size[2] = 9`，进程以 `ERR99999 UNKNOWN applicaiton exception` 退出（来源平台：950；证据等级：tool_observed）
- 标记 skip 后该用例输出 `[SKIP] 条件不满足，跳过`，后续用例继续执行、全量跑完（来源平台：950；证据等级：tool_observed）
- golden.py 的 pads 映射代码（触发前提）：`padding = (pads[0], pads[2], pads[4])`、`y = F.grad.conv3d_weight(x, tuple(filter_size), grad, stride=stride, padding=padding, dilation=dilation, groups=groups)`（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅非对称 pads（任一维 front≠back）触发；对称 pads 用例不受影响。
- conv3d_weight 之外（自写 golden、支持非对称 padding 的参考实现）不受此限。
- 2D/其他 PyTorch 版本未验证（推断可迁移）。
- 其他平台同类测试未实测，未验证，推断可迁移（限制属 PyTorch CPU 接口语义）。

**未决项**：

- 为非对称 pads 补自写逐元素 CPU golden，恢复该类用例的自动化覆盖。
## 坏实践

- 把 6 元素 pads 直接塞给只收 3 元素对称 padding 的 PyTorch 接口并指望"取一半也能对"——非对称时维度直接对不上，且即使不报错语义也是错的（只等效 front/top/left 一侧填充）。
- 非对称 pads 用例静默从用例表删除而不是显式 skip + 原因——覆盖缺口不可见。

## 误诊

- 无。

## 预防

- 卷积类算子 ST 设计阶段先盘 pads 语义：只要算子声明 6 元素 pads，PyTorch golden 方案就必须同时给出非对称用例的处置（自写 golden 或显式 skip 登记）。
