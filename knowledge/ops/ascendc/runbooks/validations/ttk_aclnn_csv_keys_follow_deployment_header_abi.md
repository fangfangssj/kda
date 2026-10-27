---
type: "Runbook"
title: aclnn CSV attributes 键用 ABI 名 hasBias
description: TTK/aclnn 用例 CSV 的 attributes 键必须用部署头 ABI 参数名（小驼峰 hasBias），不是 IR/OpDef 的 snake_case 属性名 has_bias；optional attr 显式传值不生成缺省态。复现增量：TTK 按头文件参数名精确匹配绑定且要求"张量数+标量数+属性数 == 头文件参数个数"，键名不符或空属性触发 PARAM_COUNT_MISMATCH（用例根本不执行）；空属性可按 shape 语义推导补齐。复现增量：--validate 结构校验层同签名复现（weightDecay 双精度属性、Attribute 6 vs 头文件 7），修正参数定义并清理旧约束文件后 L0/L1 全 valid。复现增量：L2 反例层同签名复现——生成器 L2 路径不落基本标量 attributes（_process_scalar_types 缺口），L2 反例 6/6 PARAM_COUNT_MISMATCH；补全脚本保留非法维度，反例也须参数齐全。
tags: [task.testing, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T04:12:25Z'
platforms: ["agnostic"]
---

# aclnn CSV attributes 键用 ABI 名 hasBias

## 现象

在验收一个对标 `aten::_thnn_fused_gru_cell_backward` 的算子的黑盒 ACLNN ST 用例（L0 44 行 / L1 515 行 / L2 49 行三份 CSV）时，逐参数比对 CSV 与 `develop/proto.md` 发现一个容易误判的两层命名差：proto.md OpDef 里的属性名是 snake_case 的 `has_bias`（`this->Attr("has_bias").AttrType(OPTIONAL).Bool(false)`），而 CSV `attributes` 列里的键是小驼峰的 `hasBias`（`{"hasBias":false}`，CSV 转义后写作 `{""hasBias"":false}`）。这不是拼写错误，而是有意的分层命名映射：aclnn 部署头 `aclnn_thnn_fused_gru_cell_grad.h` 的 ABI 参数名就是 `hasBias`。此外该 optional attr 在全部 608 行 CSV 里都显式传了 true/false，没有任何"缺省态"行。

## 适用场景

- 工程形态：黑盒 ACLNN ST 用例生成（ascendc-st-design 公共链 / TTK aclnn 模式 CSV）与验收；aclnn host 层调用代码编写。
- API 路径：OpDef `.Attr()`（IR 层）、部署头 `aclnn_xxx.h`（ABI 层）、`aclnnXxxGetWorkspaceSize(..., hasBias, ...)`、TTK aclnn CSV 的 `attributes` 列与 `scalar_dtypes` 列。
- 验证过的平台与 CANN 版本：无上板/编译实测（host 侧 CSV 静态校验）。结论属 aclnn ABI 命名约定，按定义与芯片平台无关（`platforms: [agnostic]`）；算子目标产品 Ascend 950PR / 950DT 为设计声明。CANN 版本未探测到。复现增量在 Ascend950PR + CANN 9.1.0 上板执行复现（见验证信号）。
- 适用粒度：`platform_wide`——凡经 TTK/aclnn CSV 表达属性、且 IR 属性名与部署头 ABI 参数名大小写风格不一致的算子（aclnn ABI 常见小驼峰参数名）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：IR 属性名与 ABI 参数名恰好一致的算子（无映射差）；Kernel 直调流程（无 aclnn ABI 层）；CSV 之外的属性表达（如 host 侧 C++ 调用直接用 ABI 名，不存在两层差）。

## 根因

aclnn 的部署头 ABI 参数名与 IR/OpDef 属性名是两层命名：IR 属性 `has_bias`（snake_case）映射到 aclnn ABI 参数 `hasBias`（小驼峰，取自部署头 `aclnn_xxx.h` 的函数签名）；TTK/aclnn 调用层按 ABI 参数名绑定 CSV attributes 键，写 IR 属性名（`has_bias`）会导致调用层无法把属性映射进 aclnn 调用。配套规则：TTK 必须显式传入的 optional attr 不生成缺省态（每行都写键值，值域须包含 proto 默认值 false），但不得因此删掉 optional tensor 本身的空指针负向场景。

## 设计决策

- 为何 CSV 用 ABI 名而不是 IR 名：CSV 是 TTK/aclnn 调用的直接输入，绑定发生在 ABI 层；设计会话把该决策连同出处写进 `00_interface_type.yaml` notes（"CSV attributes 键使用 aclnn 头文件 ABI 参数名 hasBias（部署头 aclnn_thnn_fused_gru_cell_grad.h）"）与 `01_parameter_description.md` 的 has_bias 行（"ACLNN ABI 参数名 hasBias，TTK/aclnn 调用显式传值，不生成缺省态"）。
- 备选与放弃原因：无"CSV 键沿用 IR 名"的备选讨论记录——公共 Skill 的建模规则（blackbox-designer 约定：optional attr 使用 ABI 参数名且值域包含 proto 默认值，不生成缺省态）已定死。
- 当前方案属哪档路线：recommended_fastpath（照部署头抄，无歧义）。

## 系统信息

- 工程形态：黑盒 ACLNN ST 用例验收（ascendc-st-design 产物链）。
- 相关文件清单（与事件流实际读取一一对应）：
  - `docs/<Op>/develop/proto.md`（IR 属性名 has_bias 的出处）
  - `operators/<Op>_package/tests/blackbox/design/00_interface_type.yaml`（CSV 键用 ABI 名的决策记录）
  - `operators/<Op>_package/tests/blackbox/design/01_parameter_description.md`（has_bias 参数行 + aclnn 签名）
  - `operators/<Op>_package/tests/blackbox/testcases/aclnn<Op>_l0/l1/l2_*.csv`（attributes 列实际形态）

## 完整代码

两层命名的对照（verbatim）：

proto.md OpDef——IR 属性名（snake_case）：

```cpp
this->Attr("has_bias").AttrType(OPTIONAL).Bool(false);
```

`01_parameter_description.md`——has_bias 参数行（ABI 名与显式传值规则）：

```markdown
| has_bias | bool | 输入（attr） | 可选（默认 false；ACLNN ABI 参数名 hasBias，TTK/aclnn 调用显式传值，不生成缺省态） | - | - | bool | true / false | - | - | 前向 input_bias 是否定义（aten 语义：has_bias = input_bias.defined()）；控制两路 bias 梯度输出 shape：(3H,)（true）或 (0,)（false，aten 未定义张量的实体表达，proto.md §4 定案） |
```

CSV 行——attributes 键的实际上形态（L0 用例行 verbatim，注意 CSV 内双引号转义 `{""hasBias"":false}`）：

```csv
aclnnThnnFusedGruCellGrad_L0_001,aclnnThnnFusedGruCellGrad,"((512,1077,),(512,5385,),(512,3231,),(512,3231,),(512,1077,),(0,),(0,),)","('float32','float32','float32','float32','float32','float32','float32',)","('ND','ND','ND','ND','ND','ND','ND',)",,"{""hasBias"":false}","(2,3,4,5,6,)",,,"((-1000,-10,),(nan,nan,),)",,
aclnnThnnFusedGruCellGrad_L0_002,aclnnThnnFusedGruCellGrad,"((144,2018,),(144,10090,),(144,6054,),(144,6054,),(144,2018,),(6054,),(6054,),)","('float16','float16','float16','float16','float16','float16','float16',)","('ND','ND','ND','ND','ND','ND','ND',)",,"{""hasBias"":true}","(2,3,4,5,6,)",,,"((0.01,1,),(-0.01,0.01,),)",,
```

验收脚本中解析该键的方式（按 ABI 名正则，全量行命中；verbatim）：

```python
attrs = row["attributes"]
m = re.search(r'"hasBias"\s*:\s*(true|false)', attrs)
has_bias = (m.group(1) == "true") if m else None
```

## 预防

- 生成 CSV 前先打开部署头 `aclnn_<op>.h` 抄 ABI 参数名（不是从 proto.md OpDef 抄 IR 属性名）。
- 自检脚本对 attributes 键做存在性断言：本例脚本在缺 hasBias 键时记 `hasBias attr missing` 错误——把"键用错名字"变成可机检的失败而不是静默漏配。
- optional attr 的值域必须包含 proto 默认值（false），且不生成缺省态行；同时保留 optional tensor 的空指针负向场景（EX-S6 类）。

## 验证信号

来源：同一次黑盒 ACLNN ST 用例验收会话——独立校验脚本对全部 608 行 CSV 逐行解析 attributes 并统计 hasBias 取值分布，退出码 0、零错误。

- L0 44 行 hasBias=true 20 / false 24，L1 515 行 true 262 / false 253——两文件合计 559 行全部命中 hasBias 键、零缺键（来源平台：agnostic——host 侧 CSV 静态校验；证据等级：tool_observed）。
- L2 49 行负向同样全部携带 hasBias 键，且 dtype×hasBias 在 L0 层 3×2 全交叉非零（脚本显式检查 `L0 missing dtype x hasBias combo` 无报错）（来源平台：agnostic；证据等级：tool_observed）。
- 脚本运行输出 `NO CONTRACT ERRORS`、EXIT=0（来源平台：agnostic；证据等级：tool_observed）。

## 失效边界

- 本卡断言"CSV attributes 键 = 部署头 ABI 参数名"；不同算子的部署头参数名大小写风格须逐算子核对（本例 has_bias→hasBias，不代表所有属性都机械转小驼峰）。
- 未验证场景：TTK/harness 对错误键名（如写成 has_bias）的实际报错形态——本开发会话 无执行事件，未上板验证。（独立 trace 更新：错误键名与空属性的实际报错形态已实测补齐，见下方增量。）
- 推断可迁移：凡 aclnn 部署头两段式签名的算子，该两层命名机制均成立（未验证，推断可迁移——结论按定义与芯片平台无关）。

## 坏实践

- 从 proto.md OpDef 抄 snake_case 属性名直接写进 CSV attributes（`{"has_bias":false}`），调用层无法映射。
- optional attr 只在有非默认值时才写键、生成"缺省态"行——与"TTK 必须显式传入 optional attr"的约定冲突，也会让 dtype×attr 交叉覆盖统计失真。
- 为省事删掉 optional tensor 的空指针负向场景（EX-S6 类）——显式传值不等于该 tensor 永远存在。

## 误诊

- 验收时看到 `hasBias` ≠ `has_bias` 误判为"CSV 键拼写错误/与 proto 不一致"——实际是 IR↔ABI 两层命名的有意映射（本次验收若不读设计文件 notes 很容易误打回）。

## 增量（执行层失败签名：PARAM_COUNT_MISMATCH 机制与空属性推导补齐）

另一独立轨迹并入执行层实测（euclidean_norm 算子，三层合并 636 条 CSV，Ascend950PR + CANN 9.1.0），把"键名不符/属性缺失"的实际报错形态与机制补齐：

**1. 执行前预检签名**：TTK preflight 直接点名键名问题——

```
[preflight] FAIL: 属性键 'keep_dims' 不是 aclnnEuclideanNorm 的接口参数名（Pitfall 6）；头文件参数：['axes', 'executor', 'keepDims', 'out', 'workspaceSize', 'x']
EXIT=1
```

**2. 执行层失败签名（PARAM_COUNT_MISMATCH，用例根本不执行）**：

```
[ERROR] [testcase_api.py:1041]: OpApi [aclnnEuclideanNorm] L2 interface has 4 parameters: odict_keys(['x', 'axes', 'keepDims', 'out']). But testcase configured 3: Tensor: 3, Scalar: 0, Attribute: 0
GOLD: PARAM_COUNT_MISMATCH
PRECISION_STATUS: PARAM_COUNT_MISMATCH
```

**3. 机制（TTK `param_plan.py` `build_args` 源码级）**：① `elif param.name in attrs:` 按头文件参数名**精确匹配**，`'keep_dims' ≠ 'keepDims'` 不命中 → 该参数 `append(None)`；② 过载匹配 `key_score = len(attrs & non_tensor_names)` 同样按精确名，TTK 不做 snake/camel 互转；③ 参数计数检查 `tensor_count + scalar_count + attr_count` 必须等于 `len(op_api_info.params)`——空属性行 3+0+0=3 ≠ 4 → PARAM_COUNT_MISMATCH。**TTK 没有"缺省属性"概念，空即计数不匹配。**

**4. 空属性按 shape 语义推导补齐（合并脚本核心，verbatim）**：

```python
def derive_keepdims(in_shape, axes_shape, out_shape):
    in_rank = len(in_shape); out_rank = len(out_shape)
    num_axes = 1
    for d in axes_shape: num_axes *= d
    if out_rank == in_rank:
        return True
    if out_rank == in_rank - num_axes:
        return False
    return False  # ambiguous (exception cases w/ invalid shapes) — fill False; will fail on real reason

for row in rows:
    attr = (row.get('attributes') or '').strip()
    tvs = parse_shapes(row['tensor_view_shapes'])
    in_shape, axes_shape, out_shape = tvs[0], tvs[1], tvs[2]
    if attr and attr != '{}':
        d = ast.literal_eval(attr)
        if 'keep_dims' in d:
            d['keepDims'] = d.pop('keep_dims')
            renamed += 1
            row['attributes'] = str(d)
    else:
        # empty: derive keepDims from shapes
        in_rank = len(in_shape); out_rank = len(out_shape)
        num_axes = 1
        for dx in axes_shape: num_axes *= dx
        if out_rank == in_rank:
            kd = True; filled_true += 1
        elif out_rank == in_rank - num_axes:
            kd = False; filled_false += 1
        else:
            kd = False; filled_ambig += 1
        row['attributes'] = str({'keepDims': kd})
```

推导规则：`out_rank == in_rank` → True（keep_dims 保留被归约维）；`out_rank == in_rank − axes 元素数` → False；异常 shape（非法输入用例）保守填 False——其失败预期来自非法输入被拒，属性值不改变预期结果。先在 317 个显式属性用例上验证推导与显式值 100% 一致（0 mismatches）、319 个空属性行中 314 可推导，才落地脚本。改用例配置而不动 TTK、不放宽比较器（精度阈值原样保留）——是用例生成约定的缺陷修复，不是测试放松。

### 复现增量验证信号

- 修正前预检：`属性键 'keep_dims' 不是 aclnnEuclideanNorm 的接口参数名（Pitfall 6）`，EXIT=1（来源平台：950；证据等级：tool_observed）。
- 修正前 smoke：`PARAM_COUNT_MISMATCH ... has 4 parameters ... But testcase configured 3: Tensor: 3, Scalar: 0, Attribute: 0`（来源平台：950；证据等级：tool_observed）。
- CSV 审计：636 行 = 319 空 + 171 `{'keep_dims': False}` + 146 `{'keep_dims': True}`；rank 推导规则在 317 个显式用例上 100% 一致（0 mismatches），319 空行中 314 可推导、5 异常（来源平台：950；证据等级：tool_observed）。
- 修正后：`[preflight] PASS` PREFLIGHT_EXIT=0；合并 CSV 636 行属性齐备，smoke 用例进入正常执行（后续失败均为精度/符号层，无参数绑定层失败）（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 推导规则依赖"CSV 的 `tensor_view_shapes` 第三元是期望输出 shape"这一生成约定；换用例生成器需先核对该约定。
- 只实测了单 bool 属性算子；多属性（多标量）算子须逐属性按头文件参数名核对，且计数检查要求全部属性齐备。
- snake→camel 的坑只在"OpDef 属性名 ≠ aclnn 参数名"时出现；两侧同名的算子无此问题（但空属性→计数不匹配仍然适用）。

## 增量（weightDecay 双精度属性形态与 --validate 修复闭环）

另一独立轨迹并入第三条独立 trace（同形逐元素 AdamW 优化器算子 aclnnBenchHarnessApplyAdamW，Ascend950PR + CANN 9.2.0 vendors 安装），补齐 `--validate` 结构校验层的同签名复现与修复闭环：

**1. 触发形态（--validate 全量 PARAM_COUNT_MISMATCH，用例不执行）**：`python -m ttk aclnn -i <cases.csv> --validate` 后 L0 全部 21 个用例报 `PARAM_COUNT_MISMATCH: 21`（Validate Complete: 21 total, 0 valid）：

```
[ERROR] [TestcaseManager aclnnBenchHarnessApplyAdamW_L0_021] [testcase_aclnn.py:1018]:
OpApi [aclnnBenchHarnessApplyAdamW] L2 interface has 12 parameters:
odict_keys(['var', 'grad', 'm', 'v', 'lr', 'beta1', 'beta2', 'weightDecay', 'epsilon', 'step', 'maximize', 'out']).
But testcase configured 11: Tensor: 5, Scalar: 0, Attribute: 6
```

参数定义里 7 个属性键都写了（lr/beta1/beta2/weight_decay/epsilon/step/maximize），但 TTK 只数到 6 个 Attribute——`weight_decay` 键匹配不上头文件形参 `weightDecay`（部署头签名节选）：

```c
aclnnStatus aclnnBenchHarnessApplyAdamWGetWorkspaceSize(
    const aclTensor *var, const aclTensor *grad, const aclTensor *m, const aclTensor *v,
    double lr, double beta1, double beta2,
    double weightDecay, double epsilon,
    int64_t step, bool maximize,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
```

命名证据：部署头文件形参 `double weightDecay`（camelCase）与算子 binary_info_config.json 属性 `weight_decay`（snake_case）并存，证实部署头文件命名与文档/注册信息不同源。

**2. 修复（参数定义改 ABI 名，before → after）**：

```yaml
  - name: "weight_decay"        # before：文档命名，匹配不上头文件
    type: "double"
    ...
  - name: "weightDecay"         # after：与部署头文件形参名一致
    type: "double"
    ...
```

**3. 重生成链 gotcha（合并/追加而不覆盖）**：改参数名后重跑 生成因子 → 隐式约束 → 显式约束 → 求解配置 → 因子值 → L0/L1/L2 时，`generate_implicit_constraints.py` 对已存在的 05_约束定义.yaml 是**合并/追加而不覆盖**——直接重跑后旧文件里仍残留 13 处 `weight_decay`（手写显式约束与 factors 块），必须先 `rm -f 05_约束定义.yaml` 再重新生成，然后重写显式跨张量约束块；求解器对同一 target 取最后一个成功约束，残留旧名约束会静默生效/失效。

**4. 判别要点**：`--validate` 的 PARAM_COUNT_MISMATCH 报错自带部署接口完整形参名单，拿它与 attributes 键逐一对比即可定位不匹配键；tensor 输出用 `output_tensor_indexes` 按位置匹配（头文件输出名叫 `out`、参数定义叫 `y` 也不影响），只有 attributes 按名匹配。

### 复现增量验证信号

- 修复前：L0 `Validate Complete: 21 total, 0 valid, 21 invalid`，`PARAM_COUNT_MISMATCH: 21`，报错列出部署接口 12 形参（含 `weightDecay`）vs 用例 Attribute: 6（来源平台：950；证据等级：tool_observed）。
- 命名证据：部署头文件形参 `double weightDecay`（camelCase）与算子 binary_info_config.json 属性 `weight_decay`（snake_case）并存（来源平台：950；证据等级：tool_observed）。
- 修复后：L0 20/20 valid、L1 586/586 valid，`All testcases passed validation`（来源平台：950；证据等级：tool_observed）。
- 重生成链完整性：07_因子值.csv 10000 行跨张量 shape/dtype/dimensions 0 mismatch（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- OpDef snake_case → 头文件 camelCase 的转换规则由两处观测对比推断（binary_info_config.json vs 部署头文件），未读代码生成器源码验证；不排除某些属性名（无下划线、缩写）不发生转换——始终以部署头文件实读为准。
- `--validate` 是 host 侧结构校验，与芯片无关；行为绑定当前 TTK 版本的参数匹配实现。
- 重生成"删旧 05 再生成"的 gotcha 绑定当前生成链脚本行为，脚本升级后需复核。

## 增量（L2 反例层的参数完整性：生成器 L2 路径不落基本标量 attributes）

另一独立轨迹并入同 trace 的 L2 反例层闭环（同一 aclnnBenchHarnessApplyAdamW 会话：先前增量 修复 weightDecay 键名后 L0 20/20、L1 586/586 valid，但 L2 反例仍 6/6 全部 `PARAM_COUNT_MISMATCH`——反例在 TTK 参数匹配层被拒，负例意图根本没到达算子）：

**1. 触发形态（L2 反例全量参数计数不匹配）**：分层校验 L0（基础覆盖）20/20 valid、L1（两两组合）586/586 valid 都通过，L2（异常/预期失败用例：不支持的 int32 dtype、9 维越界、NCHW 错 format、张量间 shape 不匹配）6/6 全部 `PARAM_COUNT_MISMATCH`。检查 L2 CSV 发现所有反例行的 `attributes` 与 `output_tensor_indexes` 列**全为空**——5 tensors + 0 attributes = 5 个配置参数，而部署头文件期望 12 个（5 tensor + 7 attribute）。

**2. 根因（生成器 L2 路径的类型缺口）**：生成器的 L2 反例构建路径 `_process_scalar_types` 只处理 `aclScalar`/`aclIntArray`/`aclFloatArray`/`aclBoolArray` 类型参数、不处理基本标量类型（`double`/`int64_t`/`bool`），导致 L2 行缺全部 attributes 与 output_tensor_indexes → 参数计数不匹配。

**3. 修复（L2 反例补全脚本，verbatim）**：

```python
import csv
T = "<用例目录>"
path = f"{T}/<op>_l2_test_cases.csv"
with open(path, newline='') as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]

# proto/OpDef 声明的属性默认值（Python 字面量格式，键名与部署头文件一致）
ATTRS = "{'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'weightDecay': 0.01, 'epsilon': 1e-08, 'step': 1, 'maximize': False}"
# 追加 tensor_formats 列（TTK 按表头名解析，列位置无关）
if 'tensor_formats' not in header:
    header = header + ['tensor_formats']
ai = header.index('attributes')
oi = header.index('output_tensor_indexes')
tfi = header.index('tensor_formats')

seen = set(); out = []
for row in data:
    while len(row) < len(header): row.append('')   # 补齐行宽
    name = row[0]
    if name in seen:            # 去重（生成器可能产出重复 testcase_name）
        continue
    seen.add(name)
    row[ai] = ATTRS
    row[oi] = '(4,)'
    # format 反例：var=NCHW（算子仅支持 ND）→ 真负例
    if 'format_exception' in name:
        row[tfi] = "('NCHW','ND','ND','ND','ND')"
    else:
        row[tfi] = ''
    out.append(row)

with open(path, 'w', newline='') as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerow(header)
    w.writerows(out)
print(f"L2 fixed: {len(out)} unique cases")
```

**4. 设计原则**：反例的"错"必须落在被测维度（dtype/format/dimensions/shape）上，参数完整性是所有层级用例的结构性前提——缺参数制造出的"失败"测的是 TTK 参数匹配层，不是算子输入校验。补全后非法维度原样保留（var_dtype=int32、9 维 var、NCHW format、var(2,3)≠grad(2,3,4)）；format 反例单独给 `tensor_formats` 列（format 是独立于 dtype/shape 的被测维度）；按 testcase_name 去重（生成器对个别反例产出重复名）。每次重跑生成会重新引入未补全的反例行，须重放补全。

### 复现增量验证信号

- 修复前：L2 6/6 全部 `PARAM_COUNT_MISMATCH`（同批 L1 586/586 valid 作对照，非环境问题）（来源平台：950；证据等级：tool_observed）。
- 根因观测：生成器 `_process_scalar_types` 仅处理 aclScalar/aclIntArray/aclFloatArray/aclBoolArray，基本标量类型不落 attributes（源码读取）（来源平台：950；证据等级：tool_observed）。
- 修复后：L2 `Validate Complete: 5 total, 5 valid, 0 invalid`，`All testcases passed validation`（来源平台：950；证据等级：tool_observed）。
- 负例意图保留：补全后 5 个反例的非法维度逐一核对——var_dtype=int32、9 维 var、NCHW format、var(2,3)≠grad(2,3,4)（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 反例执行期行为（算子是否拒绝、返回什么错误码）未在该闭环验证，属黑盒执行阶段；若测试意图仅为端到端报错覆盖，参数层拒绝也可按预期失败分流统计——但那样测的是 TTK 参数匹配层而非算子输入校验，负向覆盖数为零。
- 属性默认值取自该算子 proto 声明；换算子时须替换为其自身默认值。
- 生成器补齐基本标量类型支持后，"生成侧缺口"失效，但"反例也须参数齐全"的 TTK 契约仍适用。
