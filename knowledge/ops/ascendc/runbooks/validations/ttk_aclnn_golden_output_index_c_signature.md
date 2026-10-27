---
type: "Runbook"
title: TTK aclnn golden插件输出位须按C签名序换算
description: 黑盒 CSV 的 output_tensor_indexes 是张量位下标，golden 插件收到的是按 aclnn C 头文件签名顺序交错的全部参数位；输出位须加"排在输出前的属性个数"换算，错位会把输出张量喂进 golden 标量形参报 Boolean value of Tensor is ambiguous。 复现增量：softmax 形态复现（CANN 9.1.0，520 例）——错位另一报错形态为 softmax() received an invalid combination of arguments；插件冒烟必须模拟 build_args 全位置调用形态，"两位置参数 + 关键字"形态的冒烟会恰好掩盖错位。
tags: [task.testing, topic.implementation.api_layer.c_api, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# TTK aclnn golden插件输出位须按C签名序换算

## 现象

在 Ascend950PR（CANN 9.2.0）上对自定义算子 BenchHarnessApplyRotaryPosEmb 跑 TTK aclnn 模式黑盒全量评测（528 条，golden 用独立 torch 实现 `golden.py` 的 `apply_rotary_pos_emb`，经 `golden_plugin.py` 桥接）。此时 CSV 属性键已修正（rotaryMode→rotaryModeOptional）、preflight 已 PASS，但全量执行（expect-total=528）出现 golden 生成失败，可见用例（如 L1_319，fp16 interleaved）报：

```
GOLD: GOLDEN_FAILURE
PRECISION_STATUS: FAIL
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
  File ".../golden.py", line 153, in apply_rotary_pos_emb
    query_out = apply_rotary(query_compute, cos_compute, sin_compute, rotaryMode)
  File ".../golden.py", line 127, in apply_rotary
    if layout == 1:
```

当时插件按黑盒 CSV 的 `output_tensor_indexes=(4,5)`（proto/CSV 的张量位下标：query0,key1,cos2,sin3,queryOut4,keyOut5）实例化了 `_OUTPUT_IDX = frozenset((4, 5))`，在 `_wrap(*positions)` 里按这些下标剔除后再喂 golden。

## 适用场景

- 工程形态：自定义算子 aclnn 黑盒评测 + 自定义 golden 插件桥接外部参考实现（TTK aclnn 模式）。
- API 路径：`ttk aclnn -i <csv> --plugin golden_plugin.py`；TTK `TestcaseAclnn.build_args`（位置参数构造）；插件 `__golden__ = {"aclnn": {"aclnn<Op>": "_wrap"}}` 约定。
- 验证过的平台与 CANN 版本：Ascend950PR（SOC Ascend950PR_9579），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有"输入张量与属性参数在 C 签名里交错、输出张量排在属性之后"的 aclnn 算子 golden 插件。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：golden 形参序与 aclnn C 签名输入序不一致的算子（须在插件里显式重排，不能只按位过滤）；其它 TTK 版本（build_args 行为可能不同，需重验）。

## 根因

TTK aclnn 的 `TestcaseAclnn.build_args` 按 aclnn C 头文件签名顺序构造位置参数，tensors 与 attributes 交错排列——本算子为 `query(0), key(1), cos(2), sin(3), layout(4), rotaryModeOptional(5), queryOutOut(6), keyOutOut(7)`。而 CSV 的 `output_tensor_indexes=(4,5)` 是"张量位"下标（只数张量、不含属性位）。插件直接拿它当 C 签名位过滤时，剔除的恰是 `layout`/`rotaryModeOptional` 两个标量位，而把 `queryOutOut`/`keyOutOut` 两个输出张量留在输入里喂给 golden——golden 的标量形参落到大张量上，`if layout == 1:` 触发 `Boolean value of Tensor with more than one value is ambiguous`。两个下标体系（张量位 vs C 签名位）相差"排在输出张量之前的属性/标量参数个数"（本算子 = 2）。

TTK 源码（`ttk/core_modules/testcase_manager/testcase_aclnn.py`）佐证（verbatim）：

```python
def build_args(self, tensors, scalars, attributes):
    """Build positional args list for custom plugin calls.

    Iterates param_layout in C header order, consuming tensors and scalars
    from their respective queues, and looking up attribute values.

    Args:
        tensors: nested tensor list (from context.tensors)
        scalars: nested scalar list (from context.scalars)
        attributes: dict of API parameters

    Returns:
        list of args in C header function signature order
    """
    args = []
    tensor_queue = list(tensors)
    scalar_queue = list(scalars)

    for kind, name, acl_type, default in self.param_layout:
        if kind == self.TENSOR:
            args.append(tensor_queue.pop(0))
        elif kind == self.SCALAR:
            args.append(scalar_queue.pop(0))
        else:
            args.append(attributes.get(name, default))

    return args
```

## 解决配方

修复前（`_OUTPUT_IDX` 沿用 CSV 张量位下标，错位）：

```python
_OUTPUT_IDX = frozenset((4, 5,))   # 全部张量位里属于"输出位"的下标（Pitfall 3）
```

修复后（换成 C 签名位，并写明换算规则，verbatim）：

```python
_OUTPUT_IDX = frozenset((6, 7,))   # C-header 位置（build_args 把 tensors+attrs 按 C 签名顺序交错：query0,key1,cos2,sin3,layout4,rotaryModeOptional5,queryOut6,keyOut7）—— CSV output_tensor_indexes=(4,5) 是张量位下标，需加前置属性数 2 映射到 C 位 6,7
```

配套 `_wrap` 终版（verbatim）：属性按位置传入（build_args 决定），kwargs 只含 TTK context；torch golden 直传 Tensor 不转 numpy；numpy 标量归一化：

```python
def _wrap(*positions, **kwargs):
    # Pitfall 3：positions 含 C-header 全部参数（tensors+attrs 交错，含输出位）；剔除输出位，只把真实输入喂外部 golden
    inputs = [p for i, p in enumerate(positions) if i not in _OUTPUT_IDX]
    # golden.py 是 torch 实现 → torch.Tensor 直传（不转 numpy）；numpy 属性归一化（Pitfall 7）
    inputs = [_norm_attr(p) if isinstance(p, _np.ndarray) else p for p in inputs]
    # 属性以位置传入（build_args），kwargs 仅含 TTK context（short_soc_version 等）；按 golden 形参过滤
    call_kwargs = kwargs if _accepts_kwargs else {k: v for k, v in kwargs.items() if k in _sig.parameters}
    call_kwargs = {k: _norm_attr(v) for k, v in call_kwargs.items()}   # Pitfall 7：属性归一化
    out = _golden_fn(*inputs, **call_kwargs)
    if isinstance(out, (tuple, list)):
        return tuple(_to_torch(o) for o in out)
    return _to_torch(out)
```

属性按位置传入时，只要 golden 形参序与 C 签名的输入序一致（本例 golden `(query, key, cos, sin, layout, rotaryMode)` vs C 输入序 `query, key, cos, sin, layout, rotaryModeOptional`），值即对位，名字不同无妨。

流程：改 `_OUTPUT_IDX` → 先用 2 条 mini CSV 验证 `GOLD: PASS,PASS` → 再全量重跑。

## 设计决策

- 选"按位过滤 + 手工换算下标"而非"按参数名过滤"：positions 是裸值列表没有名字，只能按位过滤；C 签名序可从头文件 param_layout 静态确定，前置属性个数固定，换算确定性强，mini 跑 2 例即可验证。
- 曾在 kwargs 里做 `rotaryModeOptional→rotaryMode` 键名回映（当时误以为属性走 kwargs）；读 build_args 源码确认属性在位置参数里、kwargs 只有 TTK context 后，删除该回映——留着无害但误导后人。
- torch golden 直传 Tensor（不走模板默认 `_to_np` 转 numpy 再转回），模板注释即如此建议，省两次全量拷贝。

## 系统信息

- toolchain：CANN 9.2.0（`$ASCEND_TOOLKIT_HOME` 指向 CANN 9.2.0 安装根），TTK ops-test-kit（`$PY -m ttk aclnn`），执行机 python 为 CPU torch 2.7.1+cpu（无 torch_npu，不影响 aclnn 模式：golden 在 host 算，设备调用走 acl C API）。
- 目标平台：Ascend950PR（SOC Ascend950PR_9579，Device 0）。
- 相关文件：`tests/blackbox/ttk/golden_plugin.py`（插件，本卡修复对象）、`tests/blackbox/ttk/blackbox_all.csv`（528 条合并 CSV）、`docs/.../develop/golden.py`（独立 torch golden）、`$ASCEND_OPP_PATH/vendors/<Op>/op_api/include/aclnn_*.h`（C 头文件，形参序事实源）、TTK `ttk/core_modules/testcase_manager/testcase_aclnn.py`（build_args）。

## 验证信号

来源说明：以下信号来自该算子黑盒 golden 插件的失败—修复—复跑过程（先 mini 2 例再全量 528 例，TTK aclnn 模式）。

- 修复前：全量执行出现 `GOLD: GOLDEN_FAILURE / PRECISION_STATUS: FAIL`，traceback 止于 golden.py `if layout == 1:` 的 `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`（来源平台：950；证据等级：tool_observed）。
- 修复后 mini 验证：2 条用例 `GOLD: PASS,PASS / PRECISION_STATUS: PASS`（来源平台：950；证据等级：tool_observed）。
- 修复后全量：total=528 pass=521 fail=7（1 例 L0 真精度失败 + 6 例 L2 异常用例参数绑定失败），无 GOLDEN_FAILURE（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "加 2"只对本算子成立（输出张量前恰好 2 个属性参数）；一般规则是 `C 签名位 = CSV 张量位 + 排在输出张量前的属性/标量参数个数`，须逐算子按 `aclnn_*.h` 数出来。
- 属性走位置还是 kwargs 取决于 TTK 版本的 build_args 实现（本卡依据的版本把属性放位置参数）；换 TTK 版本需重验。
- golden 形参序与 C 签名输入序不一致时，本配方前提不成立，需显式重排。

**未决项**：

- 无（本闭环完整：下标换算 → mini PASS → 全量出结果）。
## 坏实践

- 直接把 CSV `output_tensor_indexes` 抄进 `_OUTPUT_IDX` 不做口径换算（本卡根因）。
- torch 实现的 golden 走模板默认 `_to_np` 转 numpy 再转回 torch（模板注释明确提示 torch golden 应直传）。

## 误诊

- 最初以为属性经 kwargs 传入、在 kwargs 里做键名回映——读 TTK build_args 源码后确认属性在位置参数里，kwargs 只含 TTK context（读源码修正归因，无实测影响）。

## 后续增量（2026-09-17，并行策展批：softmax 形态复现与"冒烟形态必须同构"教训）

另一 trace（softmax 基准算子 BenchHarnessSoftmax，Ascend 950 + CANN 9.1.0，520 条 = L0 17 + L1 500 + L2 3）复现同一下标错位机制，补充三点：

**1. 错位的另一报错形态**：golden 内部 `torch.nn.functional.softmax(t, dim=dim)` 的 dim 位收到输出张量的 numpy 数组时报 `Golden generation failure: softmax() received an invalid combination of arguments - got (numpy.ndarray), but expected one of: (int dim, ...)`——traceback 落在 golden 内部第一行 API 调用、报"参数组合无效/类型不对"时，先怀疑**传参错位**而不是 golden 实现本身。

**2. 冒烟形态必须与 TTK 调用形态同构（本增量的独立教训）**：首版插件冒烟用"两个位置参数 + dim 关键字"的形态（`gp._wrap(x, y_out, dim=np.int64(-1))`）通过、行和=1.0 正常，恰好掩盖错位——冒烟必须模拟 `build_args` 的全位置调用形态（含属性位与输出位，如 `gp._wrap(x_tensor, dim_value, out_tensor)`）才与 TTK 真实调用方式一致。本开发会话 该冒烟假阴性导致 520 条全量白跑一轮。

**3. 位置参数里的非张量须逐位归一化**：属性位从 build_args 进入位置参数后绕过模板里只作用于 kwargs 的归一化——`_wrap` 对位置参数逐位分派（torch.Tensor→numpy / 非张量→`_norm_attr` 归一化为 python 标量），避免 `torch.softmax(dim=np.array(...))` 崩溃；bfloat16 经 golden 的 int16/uint16 视图桥接转换。

修正后（`_OUTPUT_IDX=2`，头文件 `(x, dim, out, ...)` 全位置序，CSV `output_tensor_indexes=(1,)` 为张量位下标）：合成张量冒烟按真实位置序调用通过；全量复跑 520 条无一条 GOLDEN_FAILURE（终局 494 pass / 26 fail = 23 条 bf16 真精度失败 + 3 条 L2 异常用例预期失败，分流判读见 ttk_aclnn_l2_exception_param_count_mismatch 卡与黑盒 dtype 分层判别卡）。

### 复现增量验证信号

来源说明：该 softmax 算子黑盒验收会话的失败—修正—冒烟—全量复跑链路。

- 修复前：全量跑批 30 秒内已完成用例全部 `GOLD: GOLDEN_FAILURE / PRECISION_STATUS: FAIL`，traceback 落在 `torch.nn.functional.softmax(t, dim=dim)`（来源平台：950；证据等级：tool_observed）。
- 修复后冒烟（按真实位置序 `_wrap(x, 1, out)`）：`OUTPUT_IDX: frozenset({2})`；fp16 含 inf 行输出 [nan,nan,nan]（IEEE754 原生传播）、bf16/fp32 行和 [1.0,1.0]、numpy 标量 dim 归一化通过（来源平台：950；证据等级：tool_observed）。
- 修复后全量：首批用例即 `GOLD: PASS / PRECISION_STATUS: PASS`；终局 total=520 pass=494 fail=26 skip=0，无一条 GOLDEN_FAILURE（来源平台：950；证据等级：tool_observed）。
- 根因锚点：头文件 `aclnnBenchHarnessSoftmaxGetWorkspaceSize(const aclTensor *x, int64_t dim, const aclTensor *out, ...)`；`AclnnParamPlan.build_args` 按 `param_layout` 逐位组装并注释 "list of args in C header function signature order"；TTK 内部 `pure_output_indexes`（张量扁平下标）与插件 `_OUTPUT_IDX`（全位置下标）是两套语义（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 本增量为 CANN 9.1.0 + softmax 单属性交错签名（输出前置属性数=1，`C 签名位 = CSV 张量位 + 1`）；多属性算子按"数头文件全位置序"同一规则换算（与原卡"加前置属性个数"规则一致）。
- TTK kernel 模式 golden 只收真实输入的 numpy、不含输出位，不适用本卡（两增量一致）；其他 TTK 版本 build_args 行为变化需重验。