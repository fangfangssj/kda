---
type: "Runbook"
title: TTK空attrs用例golden插件须补默认超参
description: TTK kernel 模式 golden 插件只把 CSV attributes 透传给 oracle 入口；默认值覆盖用例（attributes={}）时 oracle 必填 keyword-only 超参缺失，golden 生成阶段报 missing required keyword-only arguments 而 GOLDEN_FAILURE；插件须镜像 spec/OpDef 默认值补齐并做属性归一。
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# TTK空attrs用例golden插件须补默认超参

## 现象

在 Ascend950PR（CANN 9.2.0）上用 TTK kernel 模式给自定义算子 BenchHarnessApplyAdamW 跑白盒用例（34 条）时，22/34 条用例 `dyn_precision=GOLDEN_FAILURE`，但这些用例的 `perf_status=PASS`、OOB/内存检查全部正常——kernel 本身编译、运行都健康，只是 golden 参考输出没有生成出来。full_debug.log 中每条失败用例在 OnGenGolden 阶段的报错原文：

```
Golden generation failure: bench_harness_apply_adam_w() missing 4 required keyword-only arguments: 'lr', 'beta1', 'beta2', 'weight_decay'
```

携带部分属性的用例报缺 3 个，例如只带 `{'weight_decay': 0.0}` 的用例报 `missing 3 required keyword-only arguments: 'lr', 'beta1', and 'beta2'`。

当时代码状态：oracle 参考实现 golden.py 的入口签名是 `def bench_harness_apply_adam_w(var, grad, m, v, *, lr, beta1, beta2, weight_decay, epsilon=1e-8, step=1, maximize=False, **kwargs)`——lr/beta1/beta2/weight_decay 是**无默认值的必填 keyword-only 参数**；而 golden 插件（由 whitebox-tester 模板实例化）的 `_wrap` 只把 TTK 传入的 kwargs 按签名过滤后透传，不补任何默认值。用例侧：这 22 条用例的 attributes 为 `{}` 或仅含个别键（默认值覆盖用例，验证算子默认超参路径），TTK 对 attributes={} 的用例只传 `{"kernel_name": ...}`，lr/beta1/beta2/weight_decay 无人提供 → golden 生成直接 TypeError。

## 适用场景

- 工程形态：TTK kernel 模式白盒/精度测试（`python3 -m ttk kernel --plugin golden_plugin.py`），golden 插件由外部 oracle 函数（golden.py）包装而成；测试集包含默认值覆盖用例（attributes={}）。
- API 路径：TTK golden 插件 `__golden__ = {"kernel": {"<op_snake>": "_wrap"}}` → `output_generation` 的 golden 调用 `golden_func(*input_arrays, **kwargs)`，kwargs 来自用例 attributes（`__collect_dynamic_golden_kwargs`）。
- 验证过的平台与 CANN 版本：Ascend950PR（`--plat Ascend950PR_9579`），CANN 9.2.0（与 frontmatter 一致）。
- 适用粒度：platform_wide——TTK kernel 模式 golden 插件装配机制与平台无关，任何带可选属性+默认值的算子在做白盒默认值覆盖测试时都会遇到。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：TTK aclnn（e2e）golden 路径（TestSpec / 现代 `__spec__` 插件系统走不同装配逻辑，未实测）；oracle 入口自身给全部超参提供默认值时不触发本问题。

## 根因

TTK kernel 模式的 golden 插件只把 CSV attributes 透传给 oracle 入口函数，而白盒默认值覆盖用例（attributes={}）依赖算子侧默认超参，此时 oracle 的必填 keyword-only 参数无人补齐，golden 生成阶段以 TypeError 失败并被 TTK 记为 GOLDEN_FAILURE。

机制链：

1. TTK 对每条用例调用 `golden_func(*input_arrays, **kwargs)`，kwargs 来自 `__collect_dynamic_golden_kwargs(context)`——即 CSV attributes 原样（可能带 numpy 0-d array 包装）。
2. attributes={} 的用例 kwargs 里没有任何超参；attributes={'weight_decay': 0.0} 的用例只有 weight_decay。
3. oracle 入口按 keyword-only 必填参数要求 lr/beta1/beta2/weight_decay → Python 层 `TypeError: missing N required keyword-only arguments`。
4. TTK 把该异常记为 `dyn_precision=GOLDEN_FAILURE`，与真实精度失配共用同一状态值——必须读 OnGenGolden 日志区分"golden 没算出来"与"算出来对不上"。

## 解决配方

修复前（golden_plugin.py 的 `_wrap`：只过滤签名，不补默认值）：

```python
def _norm_attr(v):
    # 属性归一化：TTK 可能以 numpy array 传属性；转 python 标量/list，外部 golden 无需自己处理
    if isinstance(v, _np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    return v

def _wrap(*inputs, **kwargs):
    # kernel 模式：inputs 只含真实输入张量（不含输出位）；直接喂外部 golden
    call_kwargs = kwargs if _accepts_kwargs else {k: v for k, v in kwargs.items() if k in _sig.parameters}
    call_kwargs = {k: _norm_attr(v) for k, v in call_kwargs.items()}
    return _golden_fn(*inputs, **call_kwargs)   # 返回 numpy
```

修复后（镜像 spec/OpDef 默认值补齐 + 属性归一 + step 整数化）：

```python
def _norm_attr(v):
    # 属性归一化：TTK 可能以 numpy array 传属性；转 python 标量/list，外部 golden 无需自己处理
    if isinstance(v, _np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    return v

# spec.yaml / OpDef 声明的 attr 默认值；CSV attributes 为 {} 或仅含部分键时补齐，
# 使依赖默认超参的用例能进入真实精度比对。
_SPEC_DEFAULTS = {
    "lr": 1.0e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "epsilon": 1.0e-8,
    "step": 1,
    "maximize": False,
}

def _wrap(*inputs, **kwargs):
    # kernel 模式：inputs 只含真实输入张量（不含输出位）；直接喂外部 golden
    call_kwargs = kwargs if _accepts_kwargs else {k: v for k, v in kwargs.items() if k in _sig.parameters}
    call_kwargs = {k: _norm_attr(v) for k, v in call_kwargs.items()}
    # 补齐省略 attr 的默认值（CSV attributes={} 用例依赖默认超参）
    for _k, _v in _SPEC_DEFAULTS.items():
        if _k not in call_kwargs:
            call_kwargs[_k] = _v
    # step 在 OpDef/spec 中为 int；CSV 可能以 float 采样（1.0/10.0/1000.0），cast 回 int
    if "step" in call_kwargs:
        call_kwargs["step"] = int(call_kwargs["step"])
    return _golden_fn(*inputs, **call_kwargs)   # 返回 numpy
```

注意：`_SPEC_DEFAULTS` 必须逐项镜像 spec.yaml / OpDef 的默认值（本例 lr=1.0e-3、beta1=0.9、beta2=0.999、weight_decay=0.01、epsilon=1e-8、step=1、maximize=False）；两者漂移会让默认值用例的 golden 与 kernel 按不同超参计算，造成假精度失败。

判别方法（可复用）：

1. result.csv 里 `dyn_precision=GOLDEN_FAILURE` 且 `perf_status=PASS` 的用例，先在日志搜 `OnGenGolden` / `Golden generation failure`——有 missing-args 字样即为本卡问题，与真实精度失配区分。
2. 统计失败用例的 attributes 列：全为 `{}` 或仅含个别键 → 默认值覆盖用例，检查 golden 插件是否补默认值。

## 设计决策

- 为什么在插件层补默认值而不是给 golden.py 的必填参数加默认值：golden.py 是独立 oracle，显式必填参数是刻意的调用契约（强迫调用方声明超参，双重身份：TestSpec 插件 + 独立入口）；TTK kernel 模式的"默认值覆盖用例"语义来自算子 spec，归插件层翻译最合适。
- 为什么同时做 `int(step)` cast：CSV 侧 step 以 float 采样（配合 OpDef Float 声明规避属性类型校验失败），而 oracle 数学上按整数 step 计算 β^t，插件层归一保证 oracle 语义不变。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 实例化 golden 插件时，只要算子有 OPTIONAL 属性且测试集包含 attributes={} 用例，就在 `_wrap` 里加 `_SPEC_DEFAULTS` 补齐，并加注释要求与 spec.yaml 同步维护。
- 白盒用例集设计默认值覆盖用例（attributes={}）时，验收清单加一条"golden 插件已补默认超参"。

## 验证信号

以下信号来自该算子白盒失败修复会话：golden 插件补默认超参（与 OpDef 属性类型修复同批）后，用 TTK 复跑同一 34 条用例。

- 修复前：22/34 用例 GOLDEN_FAILURE，perf_status 全部 PASS；日志逐条报 `bench_harness_apply_adam_w() missing 4 required keyword-only arguments: 'lr', 'beta1', 'beta2', 'weight_decay'`（部分属性用例报 missing 3）（来源平台：950；证据等级：tool_observed）。
- 修复后：total=34 pass=29 fail=5；原 22 条 GOLDEN_FAILURE 用例中 20 条 PASS——attributes={} 与部分属性用例均进入真实精度比对并通过（来源平台：950；证据等级：tool_observed）。
- 残留：一条 attributes={} 用例仍 GOLDEN_FAILURE；一条 `{'step': 1}` int 属性用例转为属性类型校验失败（属另一问题，未在本会话闭环）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖 TTK kernel 模式的 golden 插件路径；golden.py 直连 / TestSpec `__spec__` 系统未验证。
- `_SPEC_DEFAULTS` 与 spec.yaml 漂移会导致默认值用例假失败——本卡不提供自动同步机制（需人工或生成器保证）。
- oracle 若把必填参数改成自带默认值即可规避本问题，但会弱化"调用方必须显式声明超参"的契约（设计取舍见设计决策）。

**未决项**：

- 一条 attributes={} 用例修复后仍 GOLDEN_FAILURE，原因未查（会话被编译环境问题中断）；需单独复跑该用例并读 OnGenGolden 日志，确认是新缺参还是真实精度失配。
- `_SPEC_DEFAULTS` 与 spec.yaml 的一致性无自动校验（可补一个对拍检查脚本）。
## 坏实践

- 看到 GOLDEN_FAILURE 直接当精度 bug 去查 kernel 数值路径——先看 perf_status 与 OnGenGolden 日志："golden 没算出来"和"算出来对不上"是两回事。
- 在 golden.py 里给必填超参随手加默认值来"消报错"——掩盖调用方缺参事实，且随手写的默认值可能与 spec 不一致。

## 误诊

无（本卡问题由日志一次定位；注意与属性类型校验失败的现象区分：GOLDEN_FAILURE 且 perf PASS vs 编译/属性失败且 perf FAIL）。
