---
type: "Runbook"
title: 白盒生成器 int 属性自动补 range(5) 致 step=0 用例
description: ascendc-whitebox-design 生成器对 int64 声明的属性忽略显式 values 与 min/max、无条件叠加 range(5)={0..4} 从 0 轮转采样，会产出非法 step=0 用例并挤掉 fp16/bf16 覆盖；改声明 float32+整数字面量可按显式值采样且 CSV 保持 int。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# 白盒生成器 int 属性自动补 range(5) 致 step=0 用例

## 现象

用 `ascendc-whitebox-design` 的统一入口 `whitebox_designer.py design-to-ttk <design_facts.json> <opspec.json> <cases.json> <ttk.csv>` 为 AdamW 类优化器算子（属性含 `step`，spec 约束 `step >= 1`）生成 TTK kernel 模式白盒用例时，`opspec.json` 里把 `step` 按真实接口声明为 int64 并给出期望取值：

```json
{"name": "step", "type": "int64", "default": 1, "values": [1, 10, 1000]}
```

生成结果与预期完全不符（探针实测）：

- 15 例结构用例里 `step` 分布 `{0: 12, (default): 3}`——**显式 `values: [1, 10, 1000]` 被整体忽略**，所有结构用例 step=0；
- 给属性加 `"min": 1, "max": 1000` 约束再生成，22 例里 step 分布 `{(none): 2, 0: 12, 1: 2, 2: 2, 3: 2, 4: 2}`——**min/max 同样被忽略**，step=0 依旧大批出现。

step=0 对该算子是非法输入（bias correction 中 `1 - β^0 = 0`，除零；spec `machine_constraint lower_inclusive=1`）。若用 `whitebox_constraints.yaml` 的 `exclude_combo` 丢掉 step=0 用例，fp16/bf16 的结构用例会因为唯一 step 值就是 0 被**整体丢弃**，dtype 覆盖塌缩到只剩 float32。

## 适用场景

- 工程形态：用 ascendc-whitebox-design（`whitebox_designer.py`）生成 TTK kernel 模式白盒用例的流程；算子具有"语义上必须为正整数/非零"的 int 属性（step、迭代数、窗口长度等）。
- API 路径：`whitebox_designer.py` 的 `design-to-ttk` / `validate-inputs` / `coverage-audit`；输入文件 `opspec.json`（`interface.attrs`）与 `whitebox_constraints.yaml`（`exclude_combo`）。
- 验证过的平台与 CANN 版本：生成行为与平台无关（host 工具）；下游用例在 Ascend950PR_9579（注册平台 950）+ CANN 9.2.0 上执行验证。
- 适用粒度：workflow_only（白盒用例生成工具的行为配方，适用于所有走该生成器的算子）。
- 路线优先级：recommended_fastpath（需要整数 CSV 字面量且避开 0 值时的唯一可行声明方式）。
- 明确不适用/未验证场景：属性本身允许 0（range(5) 含 0 无害）时无需此技巧；生成器版本升级后行为可能变化（以 `print-schema` 与探针实测为准）。

## 根因

生成器对 attr 的取值策略按**声明类型**分类：int64 类型属性走 `enum_int`（模式驱动）分支，该分支**无条件自动叠加 `range(5) = {0,1,2,3,4}` 作为候选值并从 0 开始轮转**，显式 `values` 与 `min/max` 都不参与该分支的取值；float32 类型属性走 `scale`（值采样）分支，**按显式 `values` 逐字采样、无自动 range**。因此只要属性被声明为 int64，就无法通过 opspec 控制其采样值集合。

## 解决配方

把属性在 `opspec.json` 中声明为 `float32` 类型、但 `values` 用**整数字面量**（无小数点）：

before：

```json
{"name": "step", "type": "int64", "default": 1, "values": [1, 10, 1000]}
```

after：

```json
{"name": "step", "type": "float32", "default": 1, "values": [1, 10, 1000]}
```

效果（实测）：生成器按 scale 分支用显式值 `[1, 10, 1000]` 采样，无 step=0；同时 JSON 整数字面量被原样保留，导出的 CSV `attributes` 列里 step 仍是 int 类型：

```
step types found: {'int'}
step values found: [1, 10, 1000]
```

配套保留防御性 `whitebox_constraints.yaml`（当前声明下永不命中，作为安全网）：

```yaml
# step=0 is invalid per spec (step >= 1) and would cause 1-beta^0=0 divide-by-zero in bias correction.
- type: exclude_combo
  params:
    match: {attr: {step: 0}}
```

验证命令（生成后自检字面量类型与取值集合）：

```bash
python3 -c "
import csv, ast
vals=set(); types=set()
for r in csv.DictReader(open('<out>/ttk_design_doc_cases.csv')):
    a=ast.literal_eval(r['attributes'] or '{}')
    if 'step' in a: vals.add(a['step']); types.add(type(a['step']).__name__)
print('step types:', types, 'values:', sorted(vals))"
```

## 设计决策

评估过的三条路线：

- ①int64 + `exclude_combo` 丢 step=0：被否——实测丢完后 fp16/bf16 结构用例全灭（它们的 step 轮转值恰好只有 0），dtype 覆盖塌缩；即使加大 dtype/rank 多样性让部分非零值出现，覆盖也不可控。
- ②int64 + 生成后后处理把 step=0 改成 1：被否——违反"用例必须由 gen_cases 物化生成、不得手改具体用例"的流程约束。
- ③float32 + 整数字面量（采用）：既走 scale 分支拿到显式值，又借 JSON 字面量保型让 CSV 输出 int；代价是 opspec 中的声明类型与真实接口类型（int）不一致，需在 gaps/约束文件里注明这是分类技巧（"float32 声明仅为触发 scale 采样，值为整数"），供下游知晓。

## 系统信息

- 工具：`skills/ascendc-whitebox-design/references/whitebox_designer.py`；输入 `opspec.json`/`design_facts.json` 为临时文件（放 /tmp，不是交付物），交付物为 `design_doc_cases.json` + `ttk_design_doc_cases.csv` + `whitebox_constraints.yaml` + `WORKFLOW_PROVENANCE.json`。
- 下游执行：TTK kernel 模式，Ascend950PR_9579 + CANN 9.2.0。

## 预防

- 用该生成器前，对每个 int 属性先跑一次小探针（单 dtype/rank、`mode_struct_k` 调小），打印生成用例的属性值分布，确认采样行为符合预期再写正式 design_facts。
- 凡属性有"非零/非负"语义约束（对应 spec `machine_constraint`），一律在 `whitebox_constraints.yaml` 放 `exclude_combo` 安全网，即使当前声明理论上不会产出非法值。

## 验证信号

问题与修复来自同一次白盒用例生成会话（下游执行于 950 + CANN 9.2.0）：

- 修复前（int64 声明）：探针 15 例 step 分布 `{0: 12, '(default)': 3}`，显式 values 全部被忽略（来源平台：950；证据等级：tool_observed）。
- 加 `min:1/max:1000` 无效：22 例分布 `{(none): 2, 0: 12, 1: 2, 2: 2, 3: 2, 4: 2}`，仍含 12 例 step=0（来源平台：950；证据等级：tool_observed）。
- 修复后（float32+整数字面量）：CSV `step types: {'int'}`、`step values: [1, 10, 1000]`（来源平台：950；证据等级：tool_observed）。
- 最终交付：主表 32 例通过 validate-ttk（26 列/编码/唯一性）、coverage-audit 退出码 0（size L/M/S、dtype fp32/fp16/bf16、align、rank 1-8 全覆盖）、validate-shapes 0 violations（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该配方依赖生成器"按声明类型分类取值策略"的实现细节；生成器升级后须用探针重测（`print-schema` 输出不含此行为说明）。
- float32 声明会使生成器把属性按"值采样"对待：若同时希望该属性参与"模式驱动"的组合展开（enum 语义），此法不适用。
- 整数字面量保型依赖 JSON 值原样透传到 CSV；若下游（TTK 侧）解析链路变化，仍需用自检命令复核 CSV 字面量类型。
- 本卡只解决"取值集合与字面量类型"；CSV 整型字面量与 OpDef Float 声明不匹配会触发 para_check E80003，须保证 OpDef 侧类型与 CSV 字面量一致（配套问题，另行处理）。

**未决项**：

- 生成器对 int64 属性忽略显式 `values`/`min`/`max` 是否为预期设计（enum_int 分支的文档化行为）未见说明；若属缺陷可向上游反馈让 int 属性支持显式值集合。
- 同一会话还发现生成器的两个相关行为（未单独成卡，使用时留意）：`shape_boundaries` 条目的 `dtype` 字段被忽略（恒生成 float32，但 `value_range` 与 `attrs` 被尊重）；`extreme_inputs` 条目的 `attrs` 字段被忽略（生成用例 attrs 恒为空）。
## 坏实践

- 声明 int64 后指望 `values`/`min`/`max` 生效——实测全部被忽略，白等一轮生成结果。
- 用 `exclude_combo` 硬丢非法值而不核对 dtype 覆盖——会静默丢光某些 dtype 的全部结构用例（coverage-audit 才会暴露）。

## 误诊

- 无。
