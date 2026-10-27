---
type: "Runbook"
title: TTK kernel模式torch golden的numpy桥接插件
description: TTK kernel 模式的 golden 插件按位置收 numpy 真实输入、返回 numpy；golden.py 为 torch 实现时须写 numpy↔torch 桥接插件（ttk.utilities 转换或 bf16 走 uint16/int16 位视图）+ 属性归一化 + __golden__["kernel"][op_snake] 契约，插件输出须先与直接 torch golden 逐位核对再上全量。
tags: [task.testing, topic.implementation, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# TTK kernel模式torch golden的numpy桥接插件

## 现象（任务配方入口）

为含 bfloat16 输入/输出的自定义算子（softmax 基准算子、RoPE 算子，两例均为 torch 实现的独立 golden.py + TTK kernel 模式白盒执行）接 golden 插件时，TTK kernel 模式的插件契约是**按位置传入 numpy 真实输入（不含输出位）、期望返回 numpy 数组**，而 numpy 没有原生 bfloat16——TTK 用 `ml_dtypes.bfloat16` 传递 bf16（`ttk/utilities/dtypes.py` 的 `numpy_bfloat16()` 即返回 `ml_dtypes.bfloat16`）。torch 实现的 golden 无法直接消费这些输入，实测撞两堵墙（报错原文）：

```
torch.from_numpy(bf16_np_array)
→ TypeError: can't convert np.ndarray of type ml_dtypes.bfloat16. The only supported types are:
  float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.

torch_bf16_tensor.numpy()
→ TypeError: Got unsupported ScalarType BFloat16
```

即 numpy(bf16) 进不去 torch、torch(bf16) 出不来 numpy；对普通 dtype（fp16/fp32/int）直接 `torch.from_numpy`/`.numpy()` 透传即可，但属性（attributes）可能以 0 维 numpy array 传入，须归一化后喂 golden。需要且仅需要一个桥接插件，golden 数学逻辑保持在外部 golden.py 内、独立于被测 kernel。

## 适用场景

- 工程形态：TTK（ops-test-kit）kernel 模式白盒/精度执行（`ttk kernel -i <csv> --plugin <golden_plugin.py>`），被测对象为 vendor 自定义算子二进制，golden 为 torch 实现（或任何需要 numpy↔torch 桥接的 golden）。
- API 路径：TTK 插件契约 `__golden__ = {"kernel": {"<op_name_snake>": "_wrap"}}`；`ttk.utilities.numpy_to_torch_tensor/torch_to_numpy_tensor`（含 bf16 的位级处理）；`ttk.utilities.dtypes.resolve_custom_numpy_dtypes`（bf16→ml_dtypes.bfloat16）；`ml_dtypes.bfloat16`；`torch.from_numpy`/`Tensor.numpy()`/`Tensor.view`。
- 验证过的平台与 CANN 版本：Ascend950PR（两例分别为 CANN 9.1.0 与 9.2.0，TTK 检出于 ops-test-kit）+ torch 2.7.1+cpu + ml_dtypes（与 frontmatter `platforms` 一致；golden 在 CPU 侧计算，被测 kernel 在 NPU 侧执行）。
- 适用粒度：platform_wide——TTK kernel 模式对任意算子的 torch-golden 桥接，与算子无关。
- 路线优先级：recommended_fastpath（ttk.utilities 首选；16-bit 位视图为等价自实现路线）。
- 明确不适用 / 未验证场景：TTK aclnn 模式（输入以 torch.Tensor 传入、桥接方向相反，见 [TTK aclnn golden插件输出位须按C签名序换算](ttk_aclnn_golden_output_index_c_signature.md) 与 [aclnn golden bf16 需视图转换防 ScalarType 崩](ttk_aclnn_golden_plugin_bf16_view_bridge.md)）；golden 本身就是 numpy 实现时无需桥接。

## 配方（插件完整代码）

插件放在 `$TTK/kernel_meta/golden_plugin.py`（TTK plugin_loader 扫描位置），实例化自 whitebox-tester 的 `golden_plugin.py.templ` 并加 numpy↔torch 桥（verbatim 取自会话写入；占位符：{GOLDEN_PATH}=外部 golden.py 绝对路径、{GOLDEN_FN}=入口函数名、{OP_SNAKE}=算子蛇形名）：

```python
"""TTK kernel golden 插件。

实例化自 whitebox-tester/references/golden_plugin.py.templ，并因外部 golden.py 为
**torch 实现**做必要扩展：TTK kernel 模式只把 numpy 真实输入按位置传给 golden，
而 golden.py 用 torch 张量算 (`.to()/.unsqueeze()/.repeat()` 等)。故本插件在
numpy<->torch 之间做接口桥接（与 TTK 自身 `__call_torch_api` 同款 `numpy_to_torch_tensor`/
`torch_to_numpy_tensor`）。**golden 数学逻辑完全在外部 golden.py 内、独立于被测 kernel**，
本插件不臆造也不引用 kernel 输出。

kernel 模式：inputs 只含真实输入张量(numpy，不含输出位)；`__golden__` key 为 "kernel"、
内层 key 为 op_name 蛇形名。
"""
import importlib.util as _ilu
import inspect as _inspect
import numpy as _np

_spec = _ilu.spec_from_file_location("_op_golden", "<repo>/docs/<Op>/develop/golden.py")
_g = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_g)
_golden_fn = getattr(_g, "<golden_fn>")
_sig = _inspect.signature(_golden_fn)
_accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in _sig.parameters.values())


def _norm_attr(v):
    # 属性归一化：TTK 可能以 numpy array 传属性；转 python 标量/list，外部 golden 无需自己处理
    if isinstance(v, _np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    return v


def _np_to_torch(x):
    if x is None:
        return None
    from ttk.utilities import numpy_to_torch_tensor
    return numpy_to_torch_tensor(x)


def _to_numpy(r):
    if r is None:
        return None
    try:
        import torch as _torch
    except Exception:
        return r
    if isinstance(r, _torch.Tensor):
        from ttk.utilities import torch_to_numpy_tensor
        return torch_to_numpy_tensor(r)
    return r  # 非 torch 张量（标量/numpy）原样返回


def _wrap(*inputs, **kwargs):
    # kernel 模式：inputs 只含真实输入张量(numpy，不含输出位)；桥接为 torch 喂外部 golden
    call_kwargs = kwargs if _accepts_kwargs else {k: v for k, v in kwargs.items() if k in _sig.parameters}
    call_kwargs = {k: _norm_attr(v) for k, v in call_kwargs.items()}
    t_inputs = [_np_to_torch(x) for x in inputs]
    results = _golden_fn(*t_inputs, **call_kwargs)
    if isinstance(results, (tuple, list)):
        return [_to_numpy(r) for r in results]
    return _to_numpy(results)


__golden__ = {"kernel": {"<op_snake>": "_wrap"}}
```

关键机制（从 TTK 源码核实的契约）：

- 插件函数是普通 Python 函数，`framework_of` 按 `__module__` 判定走 **custom golden** 路径（`__call_custom_golden_func`），输入即 TTK 生成的 numpy 数组（含 `ml_dtypes.bfloat16` dtype 的 bf16 数组）。golden 模块顶层 `import torch` 且入口直接调 torch API 即被识别为 torch 框架 golden——不要把入口包进 functools.partial 等丢失模块归属的包装。
- `numpy_to_torch_tensor` 把 `ml_dtypes.bfloat16` numpy 正确转成 `torch.bfloat16`（无 ml_dtypes 时 bf16 数组根本生成不出来，需 `pip install ml_dtypes`）。
- `__golden__` 内层 key 必须等于 CSV 的 op_name（蛇形），否则插件不被调用。

### bf16 位视图自实现变体（不依赖 ttk.utilities 时的等价路线）

若桥接不能复用 `ttk.utilities`（如 golden 模块需自持转换函数），bf16 的无损往返用 16-bit 整型视图（980 条全量验证过的实现，verbatim）：

```python
try:
    from ml_dtypes import bfloat16 as _NP_BF16
except ModuleNotFoundError as _e:  # pragma: no cover - TTK 运行环境必装 ml_dtypes
    raise RuntimeError("ml-dtypes is required for bfloat16 golden; pip3 install ml-dtypes") from _e


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

要点：`arr.view(np.uint16).copy()` 的 `.copy()` 必须有（`from_numpy` 要求持有连续内存、且后续 `view(torch.bfloat16)` 重释类型）；回程 `t.contiguous()` 同理。fp16/fp32/int 走 `torch.from_numpy`/`.numpy()` 直通分支。**golden 的 dtype 口径须按算子 spec 保持**：若 spec 规定"输入按 case 实际 dtype 原样送入、不额外提升"，则桥接必须位级无损（视图转换），不能用 `t.float().numpy()` 中转——那会把 bf16 期望值静默提升为 fp32，精度口径被改，bf16 用例可能假失败/假通过。

## 设计决策

- 为什么桥接而不是改写 golden 为 numpy：golden 的独立性与数学语义以"同一份 golden.py 同时服务 ST 生成与 TTK 校验"为前提；在插件层桥接保持 golden 零改动、无双实现漂移风险。
- 为什么首选 `ttk.utilities` 的转换：bf16 的 numpy↔torch 转换有专门位级处理（与 TTK 自身 `__call_torch_api` 同款），dtype 覆盖全；自写 16-bit 视图为无该工具路径时的等价替代（位级无损，已独立验证）。
- 前置依赖：TTK 环境 bf16 用例需 `pip install ml_dtypes`（python 3.13 用 cp313 轮子；无网络/慢网时 pip 会卡在下载，需设超时重试）。
- **插件单例核对纪律（上全量前的必检项）**：以一组真实输入直接调 `_wrap`，输出与直接调用 torch golden 的结果逐位比对一致后才上全量——插件桥接 bug 会被误读为 kernel 精度缺陷。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

来源说明：以下信号来自两次独立开发会话（softmax 基准算子 980 条白盒、RoPE 算子 236 条白盒，均为 TTK kernel 模式 + torch golden + 本桥接插件），全部为 950 真机实测。

- 通路探测：`from_numpy(bf16) FAIL`、`bf16.numpy() FAIL`、`view(int16)->numpy->view(bf16) OK`；`float().numpy()` 可行但把 bf16 降成 fp32（口径被改，不可用）（来源平台：950；证据等级：tool_observed）。
- softmax 例：合成用例逐 dtype 验证转换与语义——fp32 `sum_axis1=[1. 1.]`、fp16 `[1.0001831...]`、bf16 `[1.0007324...]`（原生 dtype 舍入量级正确）；2D dim=0 逐列和 `[1. 1. 1. 1.]`；特殊值 `[nan nan nan nan]`（±inf/NaN IEEE754 传播）；bf16 往返 `back to np: bfloat16 ... sum=1.0`（来源平台：950；证据等级：tool_observed）。
- softmax 例全量跑批（980 条，该 golden 为唯一期望值来源）：修复 kernel 前一轮 `total=980 pass=955 fail=25`（25 条失败均定位为 kernel 侧写出缺陷、与 golden 无关），kernel 修复后 `total=980 pass=980 fail=0 skip=0`（来源平台：950；证据等级：tool_observed）。
- RoPE 例 preflight：TTK kernel 前置检查 PASS（退出码 0）（来源平台：950；证据等级：tool_observed）。
- RoPE 例插件单例核对：以 (8,57,431,64) bf16 真实输入调用 `_wrap`，输出 dtype=bfloat16、形状正确，且与直接调用 torch golden 的结果逐位一致（`direct golden match q: True k: True`）（来源平台：950；证据等级：tool_observed）。
- RoPE 例全量执行：236/236 条全部产出结果（`total=236 pass=172 fail=64 skip=0`，无 skip、无插件崩溃），失败分布按 dtype×原因可完整归类——证明桥接插件在三种 dtype × 236 条上稳定工作（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 视图转换前提是 bf16 与 uint16/int16 同为 16-bit 小端表示——平台字节序不同时需先确认；x86_64 CPU 上验证。
- golden 只覆盖 fp16/fp32/bf16 三 dtype；fp8/int 等其他无 numpy 原生类型的 dtype 未验证（思路同：按等宽整型视图转换）。
- 入口签名绑定 TTK 插件契约（numpy 入/出、属性作关键字参数）；TTK 版本变更契约后需复核。
- torch 原生支持 bf16 `from_numpy/.numpy()` 后，视图转换成为多余但无害。
- golden 返回嵌套结构/自定义对象时 `_to_numpy` 原样透传，须 golden 返回 Tensor 或 Tensor 列表/元组。
- 插件以 `{GOLDEN_PATH}/{GOLDEN_FN}/{OP_SNAKE}` 三个占位实例化，换算子须替换；从模板重新实例化插件会覆盖手工桥接版——防复发与恢复流程见 [TTK golden插件被模板重实例化覆盖致GOLDEN_FAILURE](ttk_golden_plugin_template_overwrite.md)。
- 极端大 tensor（GB 级）下 `.copy()` 双份内存的峰值占用未测。

**未决项**：

- 未尝试：为 TTK `--golden-mode Promote` 场景单独验证本 golden 的行为（TTK 会在比较前把 golden 规范化到输出 dtype，默认模式已够用）。
## 坏实践

- 把 torch golden 改写成 numpy 版供 TTK 用——双实现漂移风险，ST 与 TTK 的 oracle 不再同源。
- `t.float().numpy()` 或 `np.asarray(x, dtype=np.float32)` 中转——静默把 bf16 期望值提升为 fp32，精度口径被改。
- golden 内调用被测算子的 aclnn 接口或复用其 tiling 产物——破坏 oracle 独立性，期望值与实现同源则永远"通过"。
- 不做插件单例核对直接上全量——插件桥接 bug 会被误读为 kernel 精度缺陷。
- 忽略 `ml_dtypes` 依赖缺失直接 `from ml_dtypes import bfloat16`——TTK 环境必装但显式报错信息（位视图变体写法）比裸 ModuleNotFoundError 更可诊断。

## 误诊

- 一度考虑"CPU torch 不支持 bf16 就用 torch_npu 在 NPU 侧算 golden"——会破坏"golden 独立于被测 NPU 路径"的隔离意图且引入设备竞争，未采用。
