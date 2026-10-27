---
type: "Runbook"
title: F.group_norm单元素组报错，golden改手工双遍
description: torch F.group_norm 在 N*C/num_groups==1 时无条件抛 ValueError 且 no_grad 无效；算子 CPU golden 须手工双遍 double 实现，并在可用域交叉验证。
tags: [task.testing, topic.compute.normalization]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# F.group_norm单元素组报错，golden改手工双遍

## 现象

为一个 GroupNorm 类自定义算子开发 PyTorch 接入 ST 测试工程（`tests/st/torch/golden.py`：CPU golden 计算 + NPU 执行结果比对）。golden 初版直接调用 `torch.nn.functional.group_norm` 作为参考实现，运行 golden 自测时第 2 例即失败——该例是 `x=[[1.0, 3.0]]`（shape (1,2)）、`num_groups=2` 的最小边界（每组恰好 1 个元素，InstanceNorm 等价场景），抛出（原文）：

```text
ValueError: Expected more than 1 value per channel when training, got input size [1, 2]
```

traceback 指向 `torch/nn/functional.py` 中 `group_norm` 内的 `_verify_batch_size([input.size(0) * input.size(1) // num_groups, num_groups] + list(input.size()[2:]))`。第一次修复尝试按"训练模式检查由梯度模式触发"的直觉，把调用包进 `torch.no_grad()`，重跑仍报**一模一样**的错误。

当时代码状态：golden 的 `compute_golden` 直接 `return torch.nn.functional.group_norm(input=x, num_groups=num_groups, weight=gamma, bias=beta, eps=epsilon)`；而 NPU 算子本身支持该边界输入（C++ ST 的 1 元素用例早已通过），golden 反而覆盖不了算子的合法输入域。

## 适用场景

- 工程形态：自定义算子 ST 测试工程中的 CPU golden 参考实现开发（PyTorch 接入测试 / 任何用 torch 写 norm 类 golden 的测试脚手架）。
- API 路径：`torch.nn.functional.group_norm`（torch 2.7.1）的 `_verify_batch_size` 检查；替代方案为手工双遍实现（见解决配方）。
- 验证过的平台与 CANN 版本：CPU golden 在 x86 python3/torch 2.7.1 环境验证；端到端（golden 参与 NPU 比对）在 Ascend 950PR + CANN 9.1.0 验证（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only 偏 operator_family（所有 norm/归一化类算子的 torch golden 开发；触发条件是 PyTorch 侧行为，与 NPU 平台无关）。
- 路线优先级：recommended_fastpath（手工 golden 是覆盖小 shape 边界用例的正解；`F.group_norm` 只能用于其可用域内的交叉验证）。
- 明确不适用/未验证场景：`N*C//num_groups > 1` 的常规 shape 下 `F.group_norm` 可正常当 golden；其他 torch 版本的检查实现未逐一验证。

## 根因/解决配方

根因（单句）：`torch.nn.functional.group_norm` 无条件调用 `_verify_batch_size([N*C//num_groups, num_groups] + spatial)`，当 `N*C//num_groups == 1`（如 N=1 且 C==num_groups 且无空间维）时该尺寸列表前缀乘积为 1 即抛 ValueError，且该检查与 `torch.no_grad()`/梯度模式无关——"when training"只是报错文案，不是开关条件。

机制链：`_verify_batch_size` 逐维累乘、任一前缀乘积等于 1 就抛错 → `N*C//num_groups == 1` 时第一个元素即 1 → 必抛。而 `N=1, C=num_groups`（InstanceNorm 等价）、单元素组是 NPU 算子的合法边界输入，golden 必须能算。

解决配方：把 golden 从 `F.group_norm` 直调改为手工双遍 double 精度实现（完整代码见下节），并保留 `F.group_norm` 作为其可用域内的交叉验证 witness。

## 完整代码

```python
def compute_golden(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    计算 GroupNorm 的 golden 结果。

    Args:
        x: 输入张量，shape (N, C, ...) 或 (N, C)，维度 2~5
        gamma: 缩放参数，shape (C,)
        beta: 偏置参数，shape (C,)
        num_groups: 分组数，C 必须能被 num_groups 整除
        epsilon: 数值稳定性参数，防止除零

    Returns:
        分组归一化后的张量，shape 和 dtype 与 x 相同

    实现细节:
        - 中间计算使用 double 精度（与 C++ golden 一致）
        - 方差采用 biased 估计（除以 N，非 N-1）
        - 两遍法：先算均值，再算方差
        - 处理所有边界场景，包括 N*C/num_groups == 1
    """
    N = x.shape[0]
    C = x.shape[1]
    spatial_shape = x.shape[2:]

    channels_per_group = C // num_groups

    # 转换为 double 精度进行计算
    x_d = x.contiguous().double()
    gamma_d = gamma.double()
    beta_d = beta.double()

    # Reshape: (N, C, *spatial) -> (N, num_groups, channels_per_group, spatial_size)
    # 按 num_groups 分组，每组包含 channels_per_group 个通道的所有空间元素
    x_reshaped = x_d.reshape(N, num_groups, channels_per_group, -1)

    # Flatten 通道和空间维度用于 reduce: (N, num_groups, group_elem_count)
    x_flat = x_reshaped.reshape(N, num_groups, -1)

    # 两遍法计算均值和方差（biased，除以 N）
    mean = x_flat.mean(dim=2, keepdim=True)                          # (N, num_groups, 1)
    var = ((x_flat - mean) ** 2).mean(dim=2, keepdim=True)           # (N, num_groups, 1)

    # 归一化: norm = (x - mean) / sqrt(var + eps)
    inv_std = 1.0 / torch.sqrt(var + epsilon)
    norm = (x_flat - mean) * inv_std                                  # (N, num_groups, group_elem_count)

    # Reshape 回原始 shape
    norm = norm.reshape(x.shape)

    # 仿射变换: y = norm * gamma + beta
    # gamma/beta shape: (C,) -> broadcast to (1, C, 1, 1, ...)
    gamma_shape = [1, C] + [1] * len(spatial_shape)
    y = norm * gamma_d.reshape(gamma_shape) + beta_d.reshape(gamma_shape)

    # 转回输入 dtype
    return y.to(x.dtype)
```

配套自测保留 `F.group_norm` 作为**交叉验证**（只在其可用域 `N*C//num_groups > 1` 内对比手工实现与官方实现一致）：

```python
    # 测试 12: 与 torch.nn.functional.group_norm 交叉验证
    # 在 group_norm 可用的场景下（N*C/ng > 1）验证手动实现与官方实现一致
    total_count += 1
    x = torch.randn(4, 8, 4, 4, dtype=torch.float32)
    gamma = torch.randn(8, dtype=torch.float32)
    beta = torch.randn(8, dtype=torch.float32)
    manual = compute_golden(x, gamma, beta, num_groups=4, epsilon=1e-5)
    with torch.no_grad():
        official = torch.nn.functional.group_norm(
            input=x, num_groups=4, weight=gamma, bias=beta, eps=1e-5)
    match = torch.allclose(manual, official, atol=1e-5)
    print(f"  测试 12: 与 torch.nn.functional.group_norm 交叉验证 -> {'PASS' if match else 'FAIL'}")
    if match:
        passed_count += 1
    else:
        all_passed = False
```

## 设计决策

- 为何手工实现而非绕过检查：`_verify_batch_size` 无条件执行，`no_grad`/`inference_mode` 均不改变其行为（实测重跑同错）；"用更大 batch 算完再切首样本"之类绕法会改变语义；直接手工双遍是与 C++ golden（同为 double、两遍、biased 方差）对齐的正解。
- 中间计算用 double：与 C++ ST 的 `ComputeGoldenGroupNorm` 保持一致，避免 golden 自身引入 fp32 累加误差污染精度判定。
- 交叉验证策略：官方 `F.group_norm` 降级为"可用域内的等价性 witness"（测试 12），边界域由手工用例覆盖（测试 2/6 等单元素组）。

## 验证信号

来源说明：问题与验证来源：PyTorch 接入 ST 工程开发会话中 golden 自测的三次实跑（失败 → no_grad 尝试仍失败 → 手工实现通过），以及此后 NPU 全量回归；命令输出均完整保留。

- `torch.no_grad()` 尝试后自测仍在第 2 例抛同一 `ValueError: Expected more than 1 value per channel when training, got input size [1, 2]`（来源平台：CPU（golden 自测，非 NPU 信号）；证据等级：tool_observed）。
- 手工双遍 golden 落地后自测 12/12 通过，含第 2 例（InstanceNorm 等价）与第 12 例（与 `F.group_norm` 交叉验证）（来源平台：CPU（golden 自测，非 NPU 信号）；证据等级：tool_observed）。
- 该 golden 参与比对的 NPU 测试：L0 41/41 + 非连续 tensor 测试 PASS、L1 抽样 30/30、全量 641/641（L0 41 + L1 600，fp32/fp16/bf16）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 手工实现要求 `C % num_groups == 0`（`channels_per_group = C // num_groups`）；不整除输入未定义。
- 2D 输入（无空间维）与 2~5D 均已验证；>5D 未验证。
- 该结论针对 torch 2.7.1 的 `F.group_norm` 检查行为；其他版本未逐一验证（未验证，推断行为类似）。

**未决项**：

- 无（根因已定位到 torch 源码层检查逻辑；如需更早版本行为可补充多版本探针）。
## 坏实践

- 用 `torch.no_grad()` 包裹 `F.group_norm` 期望跳过 batch size 检查——实测无效，浪费一轮排查。
- 为了绕开报错而把 1 元素组/InstanceNorm 等价边界用例从测试集删掉——那是删用例掩盖 golden 缺陷（测试侧放松），NPU 算子本身支持这些输入。

## 误诊

- "报错文案带 when training，所以 eval/no_grad 能关掉"——`_verify_batch_size` 的调用不受梯度模式控制（第一次修复即踩此误诊，重跑同错证伪）。
