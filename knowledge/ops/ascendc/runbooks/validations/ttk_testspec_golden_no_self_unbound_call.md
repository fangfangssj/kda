---
type: "Runbook"
title: "TTK TestSpec golden 函数不得带 self 形参"
description: "TTK 框架以未绑定函数方式调用 TestSpec 类属性 golden（cls.golden(*args)）；写成 def golden(self, ...) 会使 self 吃掉第一个输入张量、全部形参左移一位。"
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# TTK TestSpec golden 函数不得带 self 形参

## 现象

为一个 aclnn 自定义算子编写 TTK（ops-test-kit）TestSpec——`docs/<op>/develop/golden.py`，格式为模块级 `__spec__ = {"aclnn<Op>": "<Op>Spec"}` 注册 + 一个 TestSpec 类，`golden` / `third_party` 等作为类属性函数。初版按"类方法"习惯把参考实现写成 `def golden(self, keys, ...)`（self 在第一个形参位）。TTK 框架并不会实例化这个类：`ttk/test_spec/manager.py` 用 `getattr` 从**类**上取属性并按普通函数调用（Python 3 未绑定函数语义），即 `cls.golden(*positional_args)`，位置参数按 aclnn C 头顺序传入。因此 `self` 形参实际接收到的是第一个输入张量，其余形参整体左移一位，最后一个输出张量形参落入 `**kwargs`/缺参——golden 输出全错或直接 TypeError。该缺陷在编写时通过阅读框架源码发现，运行前修复。

## 适用场景

- 工程形态：TTK（ops-test-kit）aclnn 模式 TestSpec 插件编写（`golden` / `third_party` 等类属性函数）。
- API 路径：`ttk.test_spec`（`TestSpecManager` / `SpecLoader` / `get_spec_attr`）、`__spec__` 注册协议。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.2.0（TTK 仓 PR#57 状态）。
- 适用粒度：`platform_wide`——所有 TTK TestSpec golden/third_party 插件，与具体算子无关。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证：自行实例化 spec 类再调用方法的自定义 harness（那属于实例方法语义，另当别论）；TTK 未来若改为实例化调用则本卡失效。

## 根因

框架调用契约（`ttk/test_spec/manager.py`，逐字摘录）：

```python
cls = mgr.load("softmax_v2")
if cls and mgr.has(cls, "golden"):
    golden_fn = mgr.get(cls, "golden")
```

```python
def get(self, cls: type, name: str, default: Any = None) -> Any:
    """Get attribute value. Returns default if missing or None."""
    value = getattr(cls, name, None)
    return value if value is not None else default
```

`mgr.get(cls, "golden")` 取到的是**类属性上的未绑定函数**；框架随后按 `golden_fn(*args)` 直调（位置参数 = aclnn C 头顺序的全部输入/输出/标量）。类体内定义的函数一旦带 `self`，第一个位置实参（第一个输入张量）就被 `self` 接收，后续形参全部错位。

## 解决配方

修复即删除 `self`，并留下契约注释。before（缺陷形态，逐字）：

```python
    # -- golden (numpy, CPU) ------------------------------------------------
    def golden(self, keys, hiddenStates, value, norm1Weight, norm2Weight,
               convNormWeight, convWeight, convStateOptional,
               hcMult, hiddenSize, kernelSize, dilation, normEps,
               outputOut, convStateOutOut, **kwargs):
```

after（修复后，逐字）：

```python
    # -- golden (numpy, CPU) ------------------------------------------------
    # NOTE: no ``self`` — the framework calls ``cls.golden(*args)`` as a
    # plain function (Python-3 unbound), so the first parameter receives
    # the first input tensor, not the class instance.
    def golden(keys, hiddenStates, value, norm1Weight, norm2Weight,
               convNormWeight, convWeight, convStateOptional,
               hcMult, hiddenSize, kernelSize, dilation, normEps,
               outputOut, convStateOutOut, **kwargs):
```

编写规则：

1. `golden` / `third_party` 及一切被框架按名取用的类属性函数，**一律不写 `self`**；
2. 形参按目标算子 aclnn C 头的参数顺序逐个命名（含可选 tensor 的 `xxxOptional` 槽位），尾部加 `**kwargs` 兜底框架附加参数；
3. 冒烟验证时用与框架一致的调用方式：`<Op>Spec.golden(<全部 C 头顺序位置参数>)`（不实例化），例如：

```python
result3 = BenchHarnessEngramGateFusionSpec.golden(
    keys_empty, hs_empty, val_empty, norm1_weight, norm2_weight,
    conv_norm_weight, conv_weight, None,
    HC, D, K, dil, 1e-5,
    np.zeros((B, 0, HC, D), dtype=bf16), np.zeros((B, CD, SL), dtype=bf16),
)
```

## 预防

- 写完 TestSpec 先跑 TTK 官方 validator（`from ttk.test_spec.validator import validate`），再跑一次"未绑定调用"冒烟（如上），两者都过再接入 CSV 用例。
- 检查 golden 的 import 隔离（不模块级 import torch、不 import 被测 kernel 的 aclnn 产物）时，用 AST 解析模块级 `Import/ImportFrom` 节点，不要用字符串包含匹配。
- 同框架契约的邻接卡：golden 的 bf16 数值产出与 CAST_RINT 舍入对齐见 [golden bf16 输出用 ml_dtypes 而非 torch](golden_bf16_ml_dtypes_astype_rne_aligns_cast_rint.md)（区分：本卡管函数签名/调用契约，彼卡管数值桥接与舍入语义）；从另一独立 trace 印证本契约被反复踩中的镜像面（实例化 spec 再访问 golden 报 TypeError）见 [白盒生成输出形状全错：golden 探测静默回退 shape_rule](whitebox_gen_shape_rule_fallback_wrong_shapes.md)；`__spec__` 注册键与 CSV op_name 蛇形/驼峰不一致是另一 nugget，见 [TTK 白盒 CSV op_name 蛇形与 golden 注册键不一致](ttk_csv_op_name_snake_case_golden_registration_key.md)。

## 验证信号

以下信号来自同一次开发会话中该算子 TTK TestSpec（golden.py）的编写与验收过程：

- 修复后以框架同款未绑定方式调用 `golden`（15 个 C 头顺序位置参数）冒烟：返回 2 个输出、shape 分别为 (2,8,2,16) 与 (2,32,9)、dtype bfloat16；Prefill（conv_state=None）、Empty(L=0)、K=1(stateLen=0) 三场景全部 PASS（来源平台：950；证据等级：tool_observed）
- TTK 官方 validator：`TTK validator: PASS`（来源平台：950；证据等级：tool_observed）
- SPEC-ORACLE-1 值级不变量检查（RMSNorm 保范性，4 组 shape × 3 个归一化张量）全部 PASS，脚本退出码 0（来源平台：950；证据等级：tool_observed）
- 黑盒 L0 CSV 用例逐例绑定（L0_001/L0_002/L0_007/L0_008：输入/输出 shape、dtype、attrs 名值与 golden 调用一致）全部 PASS（来源平台：950；证据等级：tool_observed）
- 缺陷形态（`def golden(self, ...)`）与修复后签名、以及 `manager.py` 调用契约，均按当次源码/框架源逐字核对（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 仅覆盖 TTK `test_spec` 类属性函数的调用契约；golden 内部数值正确性（公式、dtype 桥、舍入）是另一层问题。
- 该缺陷在本次会话中是运行前修复，未积累"带 self 实际跑到 NPU"的失败现场；错位调用的具体表现（TypeError 还是静默错位）取决于 `**kwargs` 兜底情况，未逐形态验证。
- TTK 版本变更（如改为实例化调用）会使本卡失效；使用前确认 `manager.py` 仍是 `getattr(cls, name)` 直调语义。

**未决项**：

- 无（本缺陷已闭环）。未验证不同 TTK 版本的行为差异。
## 坏实践

- 用字符串包含检查做 import 隔离门禁：`assert 'import torch' not in src` 会误报方法体内缩进的 `import torch`（合法），`assert 'aclnn' not in src` 会误报 `__spec__` 注册键与 docstring 里的算子名（合法）。本次会话两次误报后改为 AST 模块级 import 扫描才干净通过。

## 误诊

- 无。
