---
type: "Runbook"
title: TTK golden插件被模板重实例化覆盖致崩
description: TTK kernel 模式跑 torch 实现 golden.py 的白盒全量时 0 pass、dyn_precision 近全量 GOLDEN_FAILURE——whitebox-tester 的 golden_plugin.py.templ 模板 _wrap 原生透传 numpy，运行前按模板重新实例化插件会覆盖此前手工加好的 numpy↔torch 桥接版；先单例调 _wrap 复现 AttributeError，再从分支归档恢复桥接插件。
tags: [task.debugging, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# TTK golden插件被模板重实例化覆盖致崩

## 现象

用 TTK（ops-test-kit）kernel 模式全量执行一个 RoPE 自定义算子（BenchHarnessApplyRotaryPosEmb）branch-3 分支的白盒用例（`ttk_design_doc_cases.csv` 共 238 条，interleaved-BNSD-cos2d，vendor 安装源、Ascend950PR_9579）：

```text
expect-total=238
total=238 pass=0 fail=238 skip=0
GATE: FAIL
```

逐例看 `result.csv`，`dyn_precision` 分布是：

```text
precision_status: Counter({'FAIL': 238})
dyn_precision: Counter({'GOLDEN_FAILURE': 230, 'INVALID_TILING': 7, 'DYN_COMPILE_FAILURE': 1})
```

终端输出里 `Golden Shape: ('GOLDEN_FAILURE',)`、`DYN_GOLD: GOLDEN_FAILURE`——golden 插件本身崩溃，不是 kernel 精度失配。此前 branch-2 分支用同一份 golden.py 跑通过（`branch-2_archive/` 有完整产物），本次运行前按 whitebox-tester 技能要求从 `golden_plugin.py.templ` 模板重新实例化了 `tests/whitebox/ttk/golden_plugin.py`。

## 适用场景

- 工程形态：TTK（ops-test-kit）kernel 模式白盒执行（`"$PY" -m ttk kernel -i <csv> --pc 1 --plugin golden_plugin.py`），被测对象为 vendor 自定义算子二进制；每个分支重新实例化 golden 插件的工作流。
- API 路径：TTK golden 插件契约 `__golden__ = {"kernel": {"<op_snake>": "_wrap"}}`；whitebox-tester 的 `references/golden_plugin.py.templ` 模板；`ttk.utilities.numpy_to_torch_tensor / torch_to_numpy_tensor` 桥接。
- 验证过的平台与 CANN 版本：Ascend950PR_9579 + CANN 9.2.0（TTK 执行环境与 frontmatter `platforms` 一致；桥接模式本身与平台无关）。
- 适用粒度：workflow_only（TTK 评测工具链插件工作流；不宣称 kernel 行为结论）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：golden 本身就是 numpy 实现时模板可直接用；TTK aclnn 模式 golden 另路处理。numpy↔torch 桥接插件本身的完整写法见 [TTK kernel模式torch golden的numpy桥接插件](ttk_kernel_golden_plugin_numpy_torch_bridge.md)，本卡只覆盖"模板覆盖事故"的判别与恢复。

## 根因

`golden_plugin.py.templ` 模板的 `_wrap` **原生不做 numpy→torch 转换**，直接把 TTK 传入的 numpy 数组透传给 golden 函数（模板 verbatim）：

```python
def _wrap(*inputs, **kwargs):
    # kernel 模式：inputs 只含真实输入张量（不含输出位）；直接喂外部 golden
    call_kwargs = kwargs if _accepts_kwargs else {k: v for k, v in kwargs.items() if k in _sig.parameters}
    call_kwargs = {k: _norm_attr(v) for k, v in call_kwargs.items()}
    return _golden_fn(*inputs, **call_kwargs)   # 返回 numpy
```

而 golden.py 是 torch 实现（内部 `query.to(compute_dtype)` 等张量方法）。此前分支的白盒运行靠**手工在插件里加 numpy↔torch 桥**（`_np_to_torch` / `_to_numpy`）才能工作；本次运行前按技能流程"从模板实例化"插件，把带桥接的工作版**覆盖回了无桥接的模板版**——numpy 数组进 torch golden，第一处 `.to()` 即抛异常，TTK 把每个用例记为 GOLDEN_FAILURE。

## 判别方法

1. **近乎全量（230/238）GOLDEN_FAILURE = 插件崩溃，不是 kernel 精度问题**。精度缺陷只会让 mere/mare 超阈（dyn_precision 仍是 FAIL,FAIL 之类），不会成批 GOLDEN_FAILURE。
2. 二分定位 golden 链路：直接用 torch 张量调 golden.py（正常，`OK torch.Size([7, 55, 411, 96])`），再模拟 TTK 以 numpy 输入调插件 `_wrap`：

```python
import numpy as np, golden_plugin
q  = np.random.randn(7,55,411,96).astype(np.float32)
k  = np.random.randn(7,55,411,96).astype(np.float32)
cos = np.random.randn(1,411,1,48).astype(np.float32)
sin = np.random.randn(1,411,1,48).astype(np.float32)
out = golden_plugin._wrap(q, k, cos, sin, layout=1, rotaryMode='interleaved')
```

   崩溃点即根因证据（报错指回 golden.py 的 torch 调用）：

```text
AttributeError: 'numpy.ndarray' object has no attribute 'to'
  File ".../ttk/golden_plugin.py", line 26, in _wrap
    return _golden_fn(*inputs, **call_kwargs)   # 返回 numpy
  File ".../develop/golden.py", line 71, in apply_rotary_pos_emb
    query_compute = query.to(compute_dtype)
```

3. 对照模板确认：`grep -n "_wrap\|_np_to_torch\|_to_numpy\|__golden__" golden_plugin.py`——只有模板的透传版 `_wrap`、没有 `_np_to_torch`/`_to_numpy` 即被覆盖。

## 解决配方

从上一分支的归档恢复带桥接的插件（桥接只做 numpy↔torch 类型转换，golden 数学仍在外部 golden.py 内、独立于被测 kernel），单例验证后重跑全量：

```bash
PKG=<operator_package>; OUTDIR="$PKG/tests/whitebox/ttk"
# 恢复此前分支验证过的桥接版插件（numpy<->torch 桥，golden 数学仍在独立 golden.py）
cp "$OUTDIR/branch-2_archive/golden_plugin.py" "$OUTDIR/golden_plugin.py"
grep -n "_wrap\|_np_to_torch\|_to_numpy\|__golden__" "$OUTDIR/golden_plugin.py"
# 单例 probe：_wrap(q,k,cos,sin,layout=1,rotaryMode='interleaved') 应返回 list of 2 numpy float32
# 然后重跑 ttk kernel 全量（可复用上次已编译的 kernel_meta，上次只是 golden 崩、kernel 已编译）
```

恢复后的插件 `_wrap` 形态（关键差异：`t_inputs = [_np_to_torch(x) for x in inputs]` 喂 golden、结果逐个 `_to_numpy` 返回；完整实现见桥接配方卡）：

```python
def _np_to_torch(x):
    if x is None: return None
    from ttk.utilities import numpy_to_torch_tensor
    return numpy_to_torch_tensor(x)

def _to_numpy(r):
    if r is None: return None
    try: import torch as _torch
    except Exception: return r
    if isinstance(r, _torch.Tensor):
        from ttk.utilities import torch_to_numpy_tensor
        return torch_to_numpy_tensor(r)
    return r

def _wrap(*inputs, **kwargs):
    call_kwargs = kwargs if _accepts_kwargs else {k: v for k, v in kwargs.items() if k in _sig.parameters}
    call_kwargs = {k: _norm_attr(v) for k, v in call_kwargs.items()}
    t_inputs = [_np_to_torch(x) for x in inputs]
    results = _golden_fn(*t_inputs, **call_kwargs)
    if isinstance(results, (tuple, list)):
        return [_to_numpy(r) for r in results]
    return _to_numpy(results)

__golden__ = {"kernel": {"bench_harness_apply_rotary_pos_emb": "_wrap"}}
```

注意：golden 恢复正常后，失败数字会从 238 变成真实的精度/边界失败（本例 136 pass / 102 fail），这才是可归因的基线——不要把 0 pass 误当 kernel 全坏，也不要把恢复后的 fail 又误当插件问题。

## 设计决策

- 为什么恢复归档版而不是改模板：golden 数学必须保持在外部 golden.py（独立 oracle 红线），插件层只做类型桥接；模板缺陷（不含桥接）应反馈工具链维护者，运行现场以恢复已验证桥接版为最短路径。
- 为什么重跑可复用 kernel_meta：上次失败只崩在 golden，kernel 已全部编译执行过，缓存二进制有效，重跑只重算 golden 比对。
- 当前方案路线档位：recommended_fastpath（每分支重新实例化插件的工作流里，"实例化后必查桥接是否在"是防复发关键）。

## 预防

- 按模板实例化 golden 插件后，**必跑单例 `_wrap` probe**（numpy 进、numpy 出、形状 dtype 正确）再上全量；probe 崩溃即桥接缺失。
- 分支切换时把带桥接的插件随产物一起归档（`branch-N_archive/golden_plugin.py`），下次可直接恢复。
- 全量结果里出现成批 GOLDEN_FAILURE 时，先怀疑插件/桥接，再怀疑 kernel。

## 验证信号

来源说明：问题与修复来源：一次 RoPE 算子 branch-3（interleaved-BNSD）白盒 TTK kernel 全量执行会话，同会话内完成"0/238 → 定位 → 恢复 → 136/238"闭环，全部为 950 真机实测。

- 首跑全量：`total=238 pass=0 fail=238 skip=0`，`GOLDEN_FAILURE 230 + INVALID_TILING 7 + DYN_COMPILE_FAILURE 1`（来源平台：950；证据等级：tool_observed）。
- torch 直调 golden 正常：`OK torch.Size([7, 55, 411, 96]) torch.bfloat16`（来源平台：950；证据等级：tool_observed）。
- numpy 进 `_wrap` 崩溃锚点：`AttributeError: 'numpy.ndarray' object has no attribute 'to'`（golden_plugin.py line 26 → golden.py line 71）（来源平台：950；证据等级：tool_observed）。
- 恢复桥接版后单例 probe：`OK type= list, n= 2, shapes= [(7, 55, 411, 96), (7, 55, 411, 96)], dtypes= [float32, float32]`（来源平台：950；证据等级：tool_observed）。
- 重跑全量：`total=238 pass=136 fail=102 skip=0`，golden 全部正常产出（失败转为 FAIL,FAIL/PASS,FAIL/INVALID_TILING/DYN_COMPILE_FAILURE/PROFILE_CRASH 可归因类别）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"模板实例化覆盖桥接插件"这一事故形态；桥接插件完整契约（含 bf16 依赖 ml_dtypes）另见桥接配方卡。
- 模板不含桥接是观测到的事实（该版 templ 第 26 行透传 numpy）；工具链后续版本若修复模板，"实例化后必跑单例 probe"的预防仍然无害且必要。
- GOLDEN_FAILURE 判别只适用于"近乎全量崩溃"；个别用例 GOLDEN_FAILURE 可能是单例 shape/dtype 边界触发 golden 自身问题，需单独复现。

**未决项**：

- `golden_plugin.py.templ` 模板为何不含桥接（设计如此假设 numpy-golden，还是缺陷）未与工具链维护方核对；建议向 whitebox-tester 技能所有者反馈"模板实例化会静默丢弃桥接"。
- 本次事故的"按模板实例化"动作发生在上一段执行流程，本卡只闭合其后果与恢复；桥接版插件与模板版的 diff 未逐行归档（恢复自 branch-2_archive）。
## 坏实践

- 看到 0 pass 就开始改 kernel/放宽容差——根因在评测侧插件，改 kernel 无效且掩盖问题。
- 每分支从模板重实例化插件后不做单例 probe 直接上全量——238 条跑完才发现全崩，浪费一轮全量执行。

## 误诊

- 首见 GOLDEN_FAILURE 容易怀疑"branch-3 的 interleaved+BNSD 组合让 golden.py 自身崩溃"；实际 golden.py 直调 torch 张量完全正常（工具验证），崩溃点在插件把 numpy 透传给 torch 代码——是桥接缺失，不是 golden 数学问题。
