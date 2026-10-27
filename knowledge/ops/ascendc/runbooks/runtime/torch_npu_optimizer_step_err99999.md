---
type: "Runbook"
title: "torch_npu AdamW.step 触发 ERR99999"
description: "torch_npu 给 torch 优化器打的补丁在 step() 选 foreach/fused 路径时无条件初始化 NPU（CPU 张量也挡不住），soc 不支持时直接 RuntimeError: Unsupported soc version + ERR99999；构造器传 foreach=False 走纯 CPU 路径绕过，交叉验证 max_abs_diff≈3e-8。"
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# torch_npu AdamW.step 触发 ERR99999

## 现象

在装了 torch_npu 的主机上为算子 golden 参考实现做 CPU 侧交叉验证：手写 AdamW 单步公式，用 `torch.optim.AdamW` 单步结果作对照。所有张量都在 CPU 上，`opt.step()` 却直接崩溃：

```text
[ERROR] 2026-07-25-23:45:28 (PID:..., Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

完整 traceback（关键帧）：

```text
File ".../torch/optim/adam.py", line 901, in adam
    _, foreach = _default_to_fused_or_foreach(
File ".../torch_npu/utils/_optim.py", line 17, in patch_supported_devices
    else torch_npu.npu.get_device_name(torch_npu.npu.current_device()))
File ".../torch_npu/npu/utils.py", line 95, in current_device
    torch_npu.npu._lazy_init()
File ".../torch_npu/npu/__init__.py", line 255, in _lazy_init
    torch_npu._C._npu_init()
RuntimeError: Unsupported soc version: Ascend950PR 9579
```

崩溃点不在计算，而在 torch_npu 给 torch 优化器打的补丁：`step()` 在选择 foreach/fused 执行路径前要查询 NPU 设备名（`patch_supported_devices`），强制触发 NPU 运行时 lazy init；在该 python 环境里 soc（Ascend950PR）不被支持/驱动不可用，直接 `RuntimeError`。CPU 张量、`Device:-1` 都挡不住这次设备探测。

## 适用场景

- 工程形态：golden/参考实现的 CPU 侧交叉验证（TTK golden_invariant_check、oracle 等价性验证）；任何在 torch_npu 主机上用 torch.optim.* 做 CPU 对照的场景。
- API 路径：`torch.optim.AdamW(...)`（及同类被 torch_npu patch 的优化器）的 `step()`；`torch_npu/utils/_optim.py patch_supported_devices`。
- 验证过的平台与 CANN 版本：Ascend 950 环境（soc Ascend950PR；python 3.12 + torch 2.7.1+cpu + torch_npu）。
- 适用粒度：platform_wide（torch_npu 宿主环境的通用行为）。
- 路线优先级：safety_fallback（fallback 声明：让 CPU 交叉验证在设备不可用/SoC 不支持的环境跑通的兜底；不改变单步数值语义；相对代价：仅覆盖非 foreach 单张量路径，foreach 批量路径与 NPU fused 路径未被交叉验证——参照路线为在支持该 SoC 的 NPU 环境直接运行 torch.optim，本轨迹未验证）。
- 明确不适用 / 未验证场景：无 torch_npu 的纯 torch 环境不触发；`fused=True` 场景未验证；其他 torch_npu 版本的触发条件集合未系统验证。

## 根因/解决配方

根因（单句）：torch_npu 的 `patch_supported_devices` 钩子在 `optimizer.step()` 的 foreach/fused 路径选择中**无条件探测 NPU 设备**（`npu.current_device()` → `_lazy_init()`），CPU-only 上下文也不例外，soc 不支持/驱动不可用时初始化直接抛 `RuntimeError`。

解决配方：构造优化器时显式 `foreach=False`——跳过 `_default_to_fused_or_foreach` 的设备探测，`step()` 走纯 CPU 单张量路径。

before（崩溃，verbatim 探针核心）：

```python
import torch
var = torch.randn(4,4) * 0.3
grad = torch.randn(4,4) * 0.1
lr, b1, b2, wd, eps = 1e-3, 0.9, 0.999, 0.01, 1e-8
# 手写 AdamW 单步公式（m=v=0, step=1）……
ours = var - lr * update
p = var.clone(); p.requires_grad_(True); p.grad = grad.clone()
opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps,
                        weight_decay=wd, maximize=False)   # ← 默认路径：step() 内探测 NPU → RuntimeError
opt.step()                                               # ← ERR99999 崩溃点
```

after（绕过，verbatim）：

```python
try:
    opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps,
                            weight_decay=wd, maximize=False,
                            foreach=False)                # ← 显式关闭 foreach：跳过设备探测
    opt.step()
    torch_res = p.detach().clone()
    print('max_abs_diff', (ours - torch_res).abs().max().item())
    print('allclose rtol1e-5', torch.allclose(ours, torch_res, rtol=1e-5, atol=1e-7))
except Exception as e:
    print('FOREACH_FALSE_FAILED:', type(e).__name__, str(e)[:200])
```

## 预防

- 在 torch_npu 宿主上写 golden 交叉验证时，凡用 `torch.optim.*` 对照，构造器默认带 `foreach=False`，不要依赖"张量在 CPU 就不会碰 NPU"的直觉。
- 报错特征速记：`ERR99999 UNKNOWN applicaiton exception`（注意 applicaiton 为原始拼写）+ traceback 落在 `torch_npu/utils/_optim.py` → 就是这个设备探测钩子，不是你的公式错。

## 验证信号

以下信号来自该算子 golden 交叉验证崩溃修复闭环（CPU 侧对照实验，950 环境宿主）：

- 修复前：`opt.step()` 两次运行均崩（`ERR99999` + `RuntimeError: Unsupported soc version: Ascend950PR 9579`，traceback 经 `_default_to_fused_or_foreach` → `patch_supported_devices` → `_lazy_init`）（来源平台：950；证据等级：tool_observed）
- 修复后：`foreach=False` 下 step 正常完成，手写 AdamW 单步公式 vs `torch.optim.AdamW`：`max_abs_diff 2.9802322387695312e-08`、`allclose rtol1e-5 True`（来源平台：950；证据等级：tool_observed）

## 失效边界

- torch_npu 版本若不再在 step 路径中探测设备（或支持该 SoC 初始化），本卡失效。
- `foreach=False` 只保证单张量 CPU 路径的语义等价；多参数组/foreach 批量路径、NPU fused 路径未被此交叉验证覆盖。
- 若宿主 NPU 运行时可用且 SoC 受支持，直接上 NPU 运行对照更优（本轨迹未验证）。
- 其他 NPU 世代宿主未验证，推断可迁移（patch 行为属 torch_npu 版本而非 SoC）。

**未决项**：

- torch_npu 各版本/各优化器中该探测钩子的触发条件集合未系统验证（AdamW 已实证，Adam/SGD 等同类 patch 推断可触发、未验证）。
## 误诊

- 容易误判为"golden 公式写错导致崩溃"：报错发生在 `step()` 内部的路径选择阶段，先看 traceback 是否落在 `torch_npu/utils/_optim.py` 再怀疑公式。
