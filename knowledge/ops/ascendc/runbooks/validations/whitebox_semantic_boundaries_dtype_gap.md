---
type: "Runbook"
title: 白盒 semantic_boundaries 无法物化 fp16 边界用例
description: ascendc-whitebox-design 的 semantic_boundaries（如 fp16 [60000,65504] dtype 边界）只记录不物化，float16 结构用例只分到 zero/near-zero/inf 值类；shape_boundary 的 dtype 又被忽略，fp16 边界值用例无生成路径，只能登记 gap 并由其他测试层补足。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# 白盒 semantic_boundaries 无法物化 fp16 边界用例

## 现象

为 AdamW 类优化器算子（fp16 输入需覆盖大值边界：60000 在 fp32 中 `grad^2 = 3.6e9` 不溢出、65504 为 fp16 max）生成 split-load 分支白盒用例时，在 `design_facts.json` 的 `semantic_extract.semantic_boundaries` 里登记了 16 条语义边界，其中包含目标用例：

```json
{"note": "fp16 large value 60000 upcast fp32 safe (grad^2=3.6e9 in fp32). value_range [60000, 65504] maps to dtype_boundary (65504=fp16 max), includes spec value 60000", "value_range": [60000.0, 65504.0], "dtype": "float16"}
```

`design-to-ttk` 运行后该边界**只被记录、没有生成任何 fp16 大值用例**：产物里 float16 用例的 `_data_range` 只有 `[0, 0]`、`[-0.01, 0.01]`、`[-inf, inf]` 三种（zero/near-zero/inf 值类轮转），始终轮不到 `dtype_boundary`。多次调整生成参数试图强制物化：

- `special_valclasses: ["dtype_boundary"]`——无效，float16 用例值域不变；
- `mode_struct_k: 48`（加大结构扫描宽度）——无效；
- `max_shape_elems: 50000` / `20000`（缩小形状规模改变结构扫描组合）——均无效。

同时测试替代路径：把边界写成 `shape_boundaries` 条目（带 `dtype: {"var": "float16", ...}` 与 `value_range: [60000.0, 65504.0]`）——`value_range` 被尊重，但 **`dtype` 字段被忽略，生成的仍是 float32 用例**。即 fp16 边界值用例在该生成器中无路可走，最终作为 gap 记录在 WORKFLOW_PROVENANCE 的 gaps 里，测试点由既有 fp16 白盒用例与 C++ ST 框架测试补足。

## 适用场景

- 工程形态：ascendc-whitebox-design 白盒用例生成；需要为特定 dtype 注入边界值域（dtype 边界、饱和区、极值）用例的算子。
- API 路径：`design_facts.json` 的 `semantic_extract.semantic_boundaries`（`value_range`+`dtype`）、`special_valclasses`、`shape_boundaries`；`whitebox_designer.py design-to-ttk`。
- 验证过的平台与 CANN 版本：生成行为与平台无关（host 工具）；相关用例下游在 Ascend950PR_9579（注册平台 950）+ CANN 9.2.0 执行。
- 适用粒度：workflow_only（生成工具行为边界，适用于所有走该生成器的算子）。
- 路线优先级：safety_fallback（承认 gap 并用其他测试层补足是当前唯一可行路线；不要在 design_facts 参数上反复尝试）。fallback 声明：此为生成器能力受限下的兜底做法，代价是白盒层 dtype 边界覆盖依赖外部用例（ST 框架/其他分支 CSV），覆盖一致性需人工对账。
- 明确不适用/未验证场景：float32 边界（结构扫描本身产 float32 用例，值类轮转可达）不受此限制；生成器升级后须重测。

## 根因

生成器的值类分配按"结构扫描产出的 dtype × 组合"轮转：float16 在结构扫描中只生成约 3 个用例，值类轮转在 zero → near-zero → inf 后即耗尽，**轮不到 `dtype_boundary`**；`semantic_boundaries` 的条目只进 provenance 记录，不反向强制生成用例。而能显式指定值域的 `shape_boundaries` 又**忽略 `dtype` 字段**（恒 float32），两条路径都封死。未追到生成器源码层的确切分配算法（是轮转顺序还是白名单），属机制层未完全闭合的负结果。

## 尝试与失败（可重建）

1. `special_valclasses: ["dtype_boundary"]`（design_facts 改参后重跑 design-to-ttk）：产物 32 例，float16 用例值域仍为 zero/near-zero/inf，无 [60000, 65504]。
2. `mode_struct_k: 48`：同样 32 例，float16 值域不变。
3. `max_shape_elems: 50000`：coverage-audit 通过（size 分布 M:7 S:10 L:16），但 fp16 60000 仍未物化。
4. `max_shape_elems: 20000`（早期窄配置探针中曾见过 fp16 [60000, 65504] 用例）：加上 rank 1-8 全量需求后仍不物化。
5. `shape_boundaries` + `dtype` dict + `value_range: [60000.0, 65504.0]`：vrange 被尊重但 dtype 落地为 float32。

接受的替代覆盖：该测试点由 full-load 分支白盒 CSV 中的 fp16 [60000, 65504] 用例与 C++ ST 框架测试（`fp16_large_60000` 等显式用例）覆盖，并在 gaps 中登记（注明"helper valclass 循环限制"）。

## 设计决策

- 选择"登记 gap + 外部层补足"而不是继续调参或手写用例：调参四次全部无效（见上），手写具体用例违反"用例必须由 gen_cases 物化"的流程约束；把 dtype 边界责任交给已有 ST 框架显式用例是当时唯一合规且可交付的路线。
- 未选择把 fp16 用例全部转成 shape_boundary 表达——dtype 被忽略，表达不出来。

## 系统信息

- 工具：`skills/ascendc-whitebox-design/references/whitebox_designer.py`；输入 `design_facts.json`（/tmp 临时），交付 `design_doc_cases.json` / `ttk_design_doc_cases.csv` / `WORKFLOW_PROVENANCE.json`（gaps 含该条目）。
- 算子与需求：fp16 60000 注入到大 shape 的某 tile 内（split-load 路径），DESIGN 分支章节要求。

## 预防

- 生成用例前先盘点"哪些测试点必须落白盒层"；若含特定 dtype 的边界值域，先用最小探针（单条 semantic_boundary / 单条 shape_boundary）验证能否物化，不能物化及早转 ST 框架显式用例，不要在正式 design_facts 上反复试。
- 交付前在 WORKFLOW_PROVENANCE 的 gaps 里如实登记未覆盖点及其外部补足位置，供验收与回归对账。

## 验证信号

负结果与兜底来自同一次白盒用例生成会话（下游执行于 950 + CANN 9.2.0）：

- `semantic_boundaries recorded: 16`（含 fp16 [60000.0, 65504.0]）但产物中 float16 用例 vrange 仅 `[0,0]/[-0.01,0.01]/[-inf,inf]`，`special_valclasses=["dtype_boundary"]`、`mode_struct_k=48`、`max_shape_elems=50000/20000` 四轮尝试均未物化（来源平台：950；证据等级：tool_observed）。
- `shape_boundaries` 探针：声明 `dtype: float16` 与 `value_range [60000,65504]`，生成结果 vrange 被尊重但 dtype=float32（来源平台：950；证据等级：tool_observed）。
- 最终交付 32 例通过全部校验（validate-ttk/coverage-audit/validate-shapes 退出码 0），gap（fp16 60000）登记于 provenance gaps，外部由 full-load 分支白盒 fp16 用例与 C++ ST 框架 `fp16_large_60000` 用例覆盖（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本负结果针对"特定 dtype（float16）× 特定值域"的物化；float32 的 shape_boundary + value_range 路径有效（实测 vrange 被尊重），不要误伤。
- 未追到生成器源码层的值类分配机制（轮转顺序/白名单位置不明），生成器升级后行为可能变化，须以最小探针重测。
- "外部补足"依赖 ST 框架用例真实存在且覆盖同一测试点（本例已核对存在）；换算子复用此兜底时必须重新对账，不能默认已覆盖。

**未决项**：

- 【优先】到生成器源码层确认值类分配算法（semantic_boundaries 是否有任何物化路径、float16 值类轮转的终止条件），把"无法物化"从经验事实升级为机制结论。
- 为 `shape_boundaries` 的 `dtype` 字段被忽略向上游提需求（shape_boundary 支持显式 dtype），打通 fp16 边界用例的显式生成路径。
## 坏实践

- 在 design_facts 参数上无限试错（本例四轮全败，每轮都是完整生成+解析的成本）。
- 因为生成器物化不了就悄悄删掉该测试点不登记——gap 必须显式记录并被外部用例承接。

## 误诊

- 早期探针曾在 `max_shape_elems=20000, ranks [1,2,3]` 的窄配置下见过 fp16 [60000, 65504] 用例，据此误以为"缩小规模就能触发"；加回 rank 1-8 全量需求后不物化——说明触发条件与结构扫描组合空间相关，非单纯规模参数（该相关性为推断，机制未验证）。
