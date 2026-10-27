---
type: "Runbook"
title: aclnn golden插件bf16需视图转换防崩
description: TTK aclnn 模式把输入以 torch.Tensor 传入 golden 插件；插件模板 _to_np 对 bfloat16 张量无差别调 .numpy() 报 "Got unsupported ScalarType BFloat16"，bf16 用例 golden 生成即崩全 GOLDEN_FAILURE——须委托 int16/uint16 视图转换（ml_dtypes）而非直通。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# aclnn golden插件bf16需视图转换防崩

## 现象

softmax 基准算子（Ascend 950，CANN 9.1.0）TTK aclnn 模式黑盒验收，golden 插件首跑全量失败。除 fp16/fp32 用例的 `softmax() received an invalid combination of arguments`（输出位下标错位，独立缺陷，另见 [TTK aclnn golden插件输出位须按C签名序换算](ttk_aclnn_golden_output_index_c_signature.md)）外，**bfloat16 用例报的是另一路错误**，run.log 原文（输入 dtype bfloat16 的用例）：

```
[ERROR] [OnGenGolden] [golden_generation.py]:
Golden generation failure: Got unsupported ScalarType BFloat16
GOLD: GOLDEN_FAILURE
PRECISION_STATUS: FAIL
```

即：只要输入是 bf16，golden 生成阶段直接崩，与 dim 传参无关——输出位下标修正后这条错误路径依然存在（修正后插件仍会对 bf16 输入张量调 `.numpy()`）。

## 适用场景

- 工程形态：TTK aclnn 模式黑盒精度验收的 golden 插件（外部 golden 按 numpy 约定编写，插件负责 torch↔numpy 桥接）；算子支持 bfloat16 输入。
- API 路径：插件 `_to_np`/`_to_torch`（torch.Tensor ↔ numpy.ndarray）；golden 侧 `ml_dtypes.bfloat16` + uint16/int16 视图转换。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR_9579），CANN 9.1.0，TTK（ops-test-kit 检出）+ Python 3.13 + torch + ml_dtypes。
- 适用粒度：workflow 级（TTK aclnn golden 插件的 bf16 桥接契约），适用于任何 bf16 用例的 aclnn golden 插件。
- 路线优先级：recommended_fastpath（bf16 算子插件必做）。
- 明确不适用 / 未验证场景：fp16/fp32 用例（`.numpy()` 直通即可）；TTK kernel 模式 golden（收 numpy、以 ml_dtypes.bfloat16 传递，转换方向不同——kernel 模式的桥接见 [TTK kernel模式torch golden的numpy桥接插件](ttk_kernel_golden_plugin_numpy_torch_bridge.md)）；golden 本身是 torch 实现、无需 numpy 桥接的场景（模板注明可去掉 `_to_np` 直传）。

## 根因

TTK aclnn 模式把输入以 **torch.Tensor**（而非 numpy）传给插件 `_wrap`，而 `torch.bfloat16` 的 `.numpy()` 原生不支持（numpy 无 bfloat16 类型），直接调用抛 `Got unsupported ScalarType BFloat16`。模板的 `_to_np` 只做了无差别 `x.detach().cpu().numpy()`，没有 bf16 分支。返回方向同理：golden 以 ml_dtypes.bfloat16 的 numpy 数组返回，`torch.from_numpy` 不认该 dtype。bf16 在 numpy↔torch 间必须走 16-bit 视图（uint16/int16）无损转换。

## 解决配方

修复前（模板 `_to_np`/`_to_torch`，无 bf16 分支）：

```python
def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()   # bf16 → RuntimeError: Got unsupported ScalarType BFloat16
    except ImportError:
        pass
    return x

def _to_torch(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(_np.asarray(x))   # ml_dtypes.bfloat16 → TypeError: can't convert np.ndarray of type ml_dtypes.bfloat16
```

修复后（插件终版 `_to_np`/`_to_torch`，bf16 委托 golden 的视图转换；verbatim）：

```python
def _to_np(x):
    """torch.Tensor → numpy.ndarray，bfloat16 经 golden._torch_to_np（int16 视图）无损转换。"""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16 and hasattr(_g, "_torch_to_np"):
            return _g._torch_to_np(t)
        return t.numpy()
    return x


def _to_torch(x):
    """numpy.ndarray → torch.Tensor，bfloat16 经 golden._np_to_torch（uint16 视图）无损转换。"""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, _np.ndarray):
        try:
            from ml_dtypes import bfloat16 as _NP_BF16
            if x.dtype == _NP_BF16 and hasattr(_g, "_np_to_torch"):
                return _g._np_to_torch(_np.ascontiguousarray(x))
        except ModuleNotFoundError:
            pass
        return torch.from_numpy(_np.ascontiguousarray(x))
    return torch.as_tensor(x)
```

被委托的 golden 侧转换函数（`docs/<Op>/develop/golden.py`，verbatim）：

```python
def _np_to_torch(arr: np.ndarray) -> torch.Tensor:
    """numpy → torch，保留 dtype。bfloat16 走 uint16 视图（torch.from_numpy 不支持 ml_dtypes.bf16）。"""
    if arr.dtype == _NP_BF16:
        # bfloat16 与 uint16 同为 16-bit；view 成 uint16 拷贝进 torch，再 view 回 bfloat16
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_to_np(t: torch.Tensor) -> np.ndarray:
    """torch → numpy，保留 dtype。bfloat16 走 int16 视图（torch.bfloat16.numpy() 不支持）。"""
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().view(_NP_BF16).copy()
    return t.numpy()
```

判别方法：golden 生成失败的 traceback 若为 `Got unsupported ScalarType BFloat16`（torch→numpy 方向）或 `can't convert np.ndarray of type ml_dtypes.bfloat16`（numpy→torch 方向），且失败用例的输入 dtype 全部是 bfloat16，即命中本坑；fp16/fp32 用例不受影响。

## 设计决策

- 为什么委托 golden 的 `_torch_to_np`/`_np_to_torch` 而不是在插件里重写视图转换：golden 模块已加载（`_g`）且其转换函数已按"numpy(bf16)↔torch(bf16) 无损"验证过，插件复用保证两侧口径一致、避免两份实现漂移。
- 为什么 `_to_torch` 里用 `try: from ml_dtypes import bfloat16` 包裹：ml_dtypes 是运行环境依赖（golden 已声明必装），插件侧探测失败时退回 `torch.from_numpy`，保持对无 bf16 算子的兼容。
- 放弃的候选：在插件里把 bf16 统一提升为 fp32 再比较（改变 golden 口径，bf16 原生精度语义丢失，可能掩盖/制造精度差异）——禁止。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

来源说明：以下信号来自该算子黑盒验收会话（与输出位下标修复合并落地后验证），全部为 950 真机实测。

- 修复前：bf16 用例 golden 生成即崩，run.log 报 `Golden generation failure: Got unsupported ScalarType BFloat16`，`GOLD: GOLDEN_FAILURE`（来源平台：950；证据等级：tool_observed）。
- 修复后冒烟（bf16 张量经 uint16 视图构造，`_wrap(xt, 1, yt)`）：输出 dtype=torch.bfloat16、行和 `[1.0, 1.0]`（来源平台：950；证据等级：tool_observed）。
- 修复后全量复跑：153 条 bf16 用例 golden 全部正常生成（无一条 GOLDEN_FAILURE），其中 130 条精度 PASS、23 条 FAIL 为精度比较失败（bf16 精度问题另案）；fp16 192/192、fp32 172/172 全过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 视图转换仅做位级搬运（无损）；golden 在原生 bf16 下计算的精度口径问题不在本卡范围（本算子 23 条 bf16 精度失败即属彼类，未解决）。
- 无 ml_dtypes 的环境（`ModuleNotFoundError`）下 `_to_torch` 退回 `torch.from_numpy`，bf16 返回值会抛 TypeError——TTK 运行环境必装 ml_dtypes，其他环境未验证。
- `hasattr(_g, "_torch_to_np")` 探测不到转换函数的 golden（未提供视图转换）时，插件退回直通 `.numpy()`，bf16 仍会崩——要求 golden 侧提供转换函数。

**未决项**：

- 无。
## 坏实践

- 插件对 bf16 输入直接 `.numpy()`（或返回方向直接 `torch.from_numpy`）。
- 为绕过转换把 bf16 用例提升为 fp32 比较（改变口径，可能掩盖真实精度差异）。

## 误诊

- 曾怀疑错误来自 TTK `_golden_mode()` 的 promote 上下文对 bf16 做了特殊处理；核对 traceback 后确认崩点在插件自身 `_to_np` 的 `.numpy()` 调用，与 TTK 上下文无关。
