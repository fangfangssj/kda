---
type: "Runbook"
title: 用例 shape 全 nan：约束表达式名不匹配被静默跳过
description: ascendc-st-design 求解链里手写约束表达式拼写与求解器识别前缀不一致（derive_shape_from_dimension 少个 s）时求解器静默跳过，因子值列为空，用例 tensor_view_shapes 全变 nan。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T04:11:06Z'
updated_at: '2026-09-17T04:11:06Z'
platforms: ["950"]
---

# 用例 shape 全 nan：约束表达式名不匹配被静默跳过

## 现象

在为 aclnn 算子生成黑盒 TTK 用例的过程中，因属性参数更名（weight_decay → weightDecay）整链重跑生成器后，检查产物发现**全部用例的 tensor_view_shapes 变成了 NaN**：L1 主表 620/620 行 shape 含 nan，L0 首行原样为：

```text
<L0 首条用例>:
  tensor_view_shapes: (nan,nan,nan,nan,nan,)
```

生成过程没有任何报错——求解、用例生成全部正常退出，坏数据静默落盘。进一步的线索：用 TTK 的 `shapelike_stc_nested` 解析该字段时抛 `ast` 相关异常（nan 不是字面量）；检查上游 07_因子值.csv 发现 `var.shape` 列全部是**空字符串**，而 `var.dimensions`（1/2/3/4…）与 `var.dtype` 都正常。

## 适用场景

- 工程形态：黑盒 ACLNN ST 用例生成（ascendc-st-design 技能求解链：05_约束定义.yaml → 06_求解配置.yaml → 07_因子值.csv → 用例 CSV）
- API 路径：`generate_solver_config.py` / `generate_factor_values.py` / `generate_test_cases.py`
- 验证过的平台与 CANN 版本：Ascend 950 算子工程的用例生成环境（host 侧 Python 链）
- 适用粒度：workflow_only（ascendc-st-design 技能求解链脚本的特有静默失败模式）
- 路线优先级：recommended_fastpath
- 明确不适用 / 未验证场景：其他版本求解器若对未知表达式报错则本卡失效；非该技能链生成的用例不适用

## 根因/解决配方

根因（单句）：手写显式约束时把求解器识别的表达式名 `derive_shape_from_dimensions` 误写成单数 `derive_shape_from_dimension`，而求解器 `generate_factor_values.py` 用 `expression.startswith('derive_shape_from_dimensions')` **精确前缀匹配**——不认识的表达式被静默跳过（不报错），`var.shape` 解析结果为空，用例生成阶段把空 shape 落成 NaN。

诊断规则（trigger → probe → observation → interpretation → repair）：

1. **触发信号**：用例 CSV `tensor_view_shapes` 全为 `(nan,nan,...)`；或 TTK shapelike 解析器对 shape 字段报非字面量错误。
2. **探针 A（定位空列）**：检查 07_因子值.csv 中对应因子列是否为空字符串：

```python
import csv
with open("07_因子值.csv") as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= 4: break
        print(f"row{i}: var.dimensions={row.get('var.dimensions')!r} "
              f"var.shape={row.get('var.shape')!r}")
# 病态输出：var.dimensions='1'  var.shape=''   ← dimensions 有值、shape 空
```

3. **探针 B（表达式核对）**：grep 求解器源码实际识别的表达式前缀，与 05_约束定义.yaml 中（尤其手写追加的显式约束块）逐字对比：

```bash
grep -n "derive_shape" <skills>/ascendc-st-design/scripts/generate_implicit_constraints.py   # 生成器输出什么
grep -n "derive_shape" <skills>/ascendc-st-design/scripts/generate_factor_values.py          # 求解器认什么
grep -n "derive_shape" 05_约束定义.yaml                                                       # 你实际写了什么
```

观测：隐式约束生成器输出 `derive_shape_from_dimensions(sources[0])`（复数，141 行）；求解器 351 行检查的也是复数前缀；而手写块里是单数 → 不匹配。

4. **解释**：求解器对无法识别的表达式不报错不告警，直接不产出该因子值 → 07 里该列空 → 用例生成把空 shape 写成 nan。
5. **修复**：把手写表达式改成求解器识别的精确拼写（before → after）：

```yaml
    expression: "derive_shape_from_dimension(sources[0])"    # before：单数，被静默跳过
    expression: "derive_shape_from_dimensions(sources[0])"   # after：复数，与求解器前缀一致
```

6. **验证**：重跑 求解配置 → 因子值 → 用例生成，确认 `var.shape` 非空（如 `[1869]`）、空列数 0、跨张量 shape/dtype/dimensions mismatch 数 0。

相邻契约（同一求解链）：求解器对同一 target 取**最后一个成功约束**——显式跨张量约束必须排在隐式约束之后才能生效（05 文件"隐式在前、显式在后"的结构约定）。

## 预防

- 手写/修改 05_约束定义.yaml 的表达式后，固定跑探针 A（因子值空列检查）再进用例生成；空列是静默跳过的唯一上游痕迹。
- 复制隐式生成器的输出作为手写约束模板，不要凭记忆重打表达式名。

## 验证信号

以下信号来自该算子用例重生成后 NaN 诊断闭环：

- 触发：L1 620/620 行 tensor_view_shapes 含 nan，L0 首行 `(nan,nan,nan,nan,nan,)`（来源平台：950；证据等级：tool_observed）
- 探针观测：07_因子值.csv `var.shape` 列空字符串、`var.dimensions`/`var.dtype` 正常（来源平台：950；证据等级：tool_observed）
- 根因观测：隐式生成器输出与求解器识别均为复数 `derive_shape_from_dimensions`，手写块为单数（来源平台：950；证据等级：tool_observed）
- 修复后：`var.shape` 非空（样本 `[1869]`）、empty_var_shape=0、跨张量 shape_mismatch=0，重生成用例经 TTK --validate 全部通过（来源平台：950；证据等级：tool_observed）

## 失效边界

- 绑定 ascendc-st-design 技能当前求解器实现（startswith 精确前缀 + 静默跳过）；求解器改为报错或模糊匹配后本卡失效。
- 只覆盖"表达式名不匹配 → 因子值空列 → shape NaN"这一条静默链；因子值列为空还有其他可能来源（约束互相覆盖等），探针 B 核对拼写无果时应继续查约束排序与 target 覆盖。

**未决项**：

- 求解器为何不报错（设计取舍 or 缺陷）未追；给求解链补"未知表达式告警"是上游改进方向。
## 坏实践

- 手写约束表达式名凭记忆输入（单复数/下划线极易拼错），生成链不报错就当成功。
- 只检查用例 CSV 行数不检查字段值，让 NaN 坏数据静默流入测试执行。
