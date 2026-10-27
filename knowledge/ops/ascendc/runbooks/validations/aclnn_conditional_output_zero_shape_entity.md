---
type: "Runbook"
title: aclnn 条件输出用 (0,) 空张量非零填充
description: aten 未定义张量输出（如 has_bias=false 的 bias 梯度）在 aclnn 固定输出槽位下用 (0,) 空张量实体表达，不省略、不零填充；ST 场景须双态枚举，B=0 退化场景锁无 bias 态。
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["agnostic"]
---
# aclnn 条件输出用 (0,) 空张量非零填充


## 现象

在验收对标 `aten::_thnn_fused_gru_cell_backward` 的算子的黑盒 ACLNN ST 用例时，按 proto.md §4 Shape 契约逐行核对 CSV：该算子有条件化输出——`grad_input_bias` / `grad_hidden_bias` 两个 bias 梯度输出的 shape 由 `has_bias` 条件化：hasBias=true 时为 `(3H,)`，hasBias=false 时为 `(0,)` 空张量。CSV 里的实际形态（L0 用例行，hasBias=false 时第 6/7 个张量是 `(0,)`）：

```csv
…,"((512,1077,),(512,5385,),(512,3231,),(512,3231,),(512,1077,),(0,),(0,),)",…,"{""hasBias"":false}",…
```

关键点：条件不成立时输出张量**仍然占槽位**（7 元组恒为 7 个），只是 shape 是 `(0,)`——不是省略张量、不是零填充。设计文档明确该 `(0,)` 是"aten 未定义张量的实体表达，proto.md §4 定案"，且 REQUIREMENTS 里" `(0,)` 或零填充 spec 期定案"的歧义被 proto.md §4 定案为 `(0,)`。

## 适用场景

- 工程形态：aclnn 算子仓开发中的 proto/§4 Shape 契约定义、黑盒 ACLNN ST 用例设计（ascendc-st-design 公共链）与验收；涉及"若 A 则 X 否则 Y"型条件化输出约束的算子。
- API 路径：OpDef `Output(...)` 固定输出槽位、aclnn 部署头两段式签名（输出指针槽位恒在）、TTK aclnn CSV 的 `tensor_view_shapes` 列、场景枚举的 C6 类"若 A 则 X 否则 Y"约束。
- 验证过的平台与 CANN 版本：无上板/编译实测（host 侧 CSV 静态校验）。结论属 aclnn ABI/ST 约定层，按定义与芯片平台无关（`platforms: [agnostic]`）；算子目标产品 Ascend 950PR / 950DT 为设计声明。CANN 版本未探测到。
- 适用粒度：`operator_family`——凡对标 aten 且有"未定义张量输出"语义（`xxx.defined()` 控制输出是否存在）的条件化输出算子族。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：无条件化输出的算子；`(0,)` 输出在 kernel/运行时的实际内存行为（本卡只覆盖 ST/ABI 层的 shape 表达约定，kernel 空张量分支的交付是算子侧契约，本 trace 未上板验证）。

## 根因

aclnn ABI 的输出参数槽位固定（两段式签名里 5 个输出指针槽恒在，不能按条件省略张量）；aten 用"未定义张量"（`input_bias.defined() == false`）表达的"无输出"，在 aclnn 侧需要一个实体张量形态。零填充不可取：`(3H,)` 全 0 会把"无 bias 梯度"伪装成"梯度恰为全 0"，既违反 §4 契约的 shape 断言又改变语义；省略张量不可取：破坏槽位固定。标准表达是空张量 `(0,)`。机制算例（CSV 实测行）：H=2018 时 3H = 3×2018 = 6054，hasBias=true 行 bias=(6054,)；hasBias=false 行 bias=(0,)。退化轴闭合：H=0 时 3H = 3×0 = 0，两态 bias 都坍缩为 (0,)，故 ET 场景 H=0 两态均闭合；B=0 时 bias 沿 batch 归约无输入行，aten 侧 B=0 且带 bias 时 bias 输出值未定义，故 B=0 场景锁定 hasBias=false 以规避未定值比较。

## 设计决策

- 为何 `(0,)` 而非零填充：proto.md §4 定案（REQUIREMENTS §8.2.2 曾留" `(0,)` 或零填充 spec 期定案"的口子）；`(0,)` 保持"输出不存在"的语义并与 shape 断言自洽，零填充则同时破坏 shape 契约与数值语义。
- 条件化场景必须双态枚举：设计文档把该约束标为"需要枚举=是"（C6，关键词"若 A 则 X 否则 Y"），子场景 C6-S1（true→(3H,)）/ C6-S2（false→(0,)）都必须生成，双态到达由 L0 离散因子 hasBias ∈ {true,false} 保证；只生成一个分支会留下覆盖缺口。
- B=0 退化场景锁 hasBias=false：规避 aten 侧"B=0 且带 bias 时 bias 输出值未定义"的未定值比较（设计文档 ET 场景注明）。
- 当前方案属哪档路线：recommended_fastpath。

## 系统信息

- 工程形态：黑盒 ACLNN ST 用例验收（ascendc-st-design 产物链：design/00-06 + L0/L1/L2 CSV + qa_auto_report）。
- 相关文件清单（与事件流实际读取一一对应）：
  - `docs/<Op>/develop/proto.md`（§4 输出 shape 契约，(3H,)/(0,) 定案出处）
  - `operators/<Op>_package/tests/blackbox/design/03_scenario_enumeration.md`（C6 双态 + ET 空张量场景表）
  - `operators/<Op>_package/tests/blackbox/design/01_parameter_description.md`（R6 条件化约束 + (0,) 实体表达说明）
  - `operators/<Op>_package/tests/blackbox/testcases/aclnn<Op>_l0/l1_*.csv`（(0,)/(3H,) 实际行）

## 完整代码

proto.md §4 输出 shape 契约（定案原文，verbatim）：

```markdown
- 输出 shape / 派生量：grad_input_gates.shape = grad_hidden_gates.shape = (grad_hy.shape[0], 3 * grad_hy.shape[1])（行内 [grg, gig, gin] / [grg, gig, ghn] 三段）；grad_hx.shape = grad_hy.shape = (B, H)；grad_input_bias.shape = grad_hidden_bias.shape = (3 * H,)（has_bias=true）或 (0,)（has_bias=false 空张量）；派生量 hsz = storage.shape[1] // 5
```

场景枚举 C6 双态 + 空张量维度场景（verbatim）：

```markdown
| C6 | bias 输出 shape 由 hasBias 条件化：(3H,) / (0,) | hasBias.value, grad_input_bias.shape, grad_hidden_bias.shape | 若 A 则 X 否则 Y | **是** | grad_input_bias.shape, grad_hidden_bias.shape |

### C6: bias 输出 shape 条件化（hasBias）
| 子场景 ID | 描述 | @solves 应生成的值示例（H=64） |
|----------|------|---------------------|
| C6-S1 | hasBias=true → grad_input_bias.shape = grad_hidden_bias.shape = (3H,) | [192] |
| C6-S2 | hasBias=false → grad_input_bias.shape = grad_hidden_bias.shape = (0,) | [0] |

> tag 绑定：`solve_grad_input_bias_shape` / `solve_grad_hidden_bias_shape` 内按 hasBias 分支调用 `tag('C6', 'C6-S1'/'C6-S2')`；两分支均由 L0 离散因子覆盖（hasBias.value ∈ {true, false}）保证到达。

## 空tensor维度场景
> 依据：kernel 交付件含空张量分支（REQUIREMENTS §8.1），proto.md §4 shape 公式对 B=0 / H=0 闭合定义（输出 (0,·)/(·,0)，bias=(3H,) 或 (0,) 由 hasBias 决定）。B=0 场景锁定 hasBias=false：aten 侧 B=0 且带 bias 时 bias 输出值未定义（REQUIREMENTS §6 空张量语义行），规避未定值比较；H=0 时 3H=0，两 hasBias 取值均闭合。
```

CSV 实测行——双态对照与退化场景（verbatim）：

```csv
aclnnThnnFusedGruCellGrad_L0_001,…,"((512,1077,),(512,5385,),(512,3231,),(512,3231,),(512,1077,),(0,),(0,),)",…,"{""hasBias"":false}",…
aclnnThnnFusedGruCellGrad_L0_002,…,"((144,2018,),(144,10090,),(144,6054,),(144,6054,),(144,2018,),(6054,),(6054,),)",…,"{""hasBias"":true}",…
aclnnThnnFusedGruCellGrad_L0_empty_032,…,"((0,64,),(0,320,),(0,192,),(0,192,),(0,64,),(0,),(0,),)",…,"{""hasBias"":false}",…
aclnnThnnFusedGruCellGrad_L0_empty_033,…,"((2,0,),(2,0,),(2,0,),(2,0,),(2,0,),(0,),(0,),)",…,"{""hasBias"":true}",…
```

（L0_002：H=2018，3H=6054，hasBias=true → bias=(6054,)；L0_empty_033：H=0，3H=0，hasBias=true → bias=(0,) 自然坍缩；L0_empty_032：B=0 锁 hasBias=false，全输出空。）

验收脚本中的条件化断言（verbatim）：

```python
exp_bias = (3 * H,) if has_bias else (0,)
chk(gib, exp_bias, "grad_input_bias")
chk(ghb, exp_bias, "grad_hidden_bias")
```

## 预防

- 定 proto §4 时显式写死条件化输出的两态 shape（`(3H,)`（true）或 `(0,)`（false 空张量）），不留"或零填充"口子。
- 场景枚举把"若 A 则 X 否则 Y"型约束标 `需要枚举=是`，双态各生成子场景并绑定 tag，到达性由离散因子双值保证。
- 退化空输入（B=0/H=0）按正向 ET 场景设计（不是负向）；B=0 且 bias 输出值在参考实现未定义时，锁无 bias 态规避未定值比较。
- 验收自检对条件化输出做逐行断言（如上 `exp_bias` 检查），并交叉核对 tagged 用例的 (B,H,hasBias) 与设计锁定值。

## 验证信号

来源：同一次黑盒 ACLNN ST 用例验收会话——独立校验脚本对全部 559 个正向用例逐行重算 §4 契约（含 bias 条件化断言），并单独交叉核对 ET/BD tagged 用例与设计锁定值。

- 559 正向行（L0 44 + L1 515）全部满足 `bias == (3H,) ⇔ hasBias=true`，脚本输出 `NO CONTRACT ERRORS`、EXIT=0（来源平台：agnostic——host 侧 CSV 静态校验；证据等级：tool_observed）。
- hasBias 取值覆盖：L0 true 20/false 24、L1 true 262/false 253，两态均有大量用例（C6-S1/S2 双态到达）（来源平台：agnostic；证据等级：tool_observed）。
- ET/BD tagged 用例与设计锁定值逐一吻合：L0_empty_032=(0,64) hasBias=false、L0_empty_033=(2,0) hasBias=true、L0_bound_034~041 对应 BD-S1~S8（含 BD-S8 (1,64) hasBias=true → bias=(192,)）（来源平台：agnostic；证据等级：tool_observed）。

## 失效边界

- `(0,)` 表达的前提是输出槽位在 ABI 里固定且算子交付空张量分支；若某 aclnn 接口允许输出指针传 null（本算子不允许，R9 只约束输入），表达方式须按该接口的 ABI 另行确认——未验证。
- B=0 锁 hasBias=false 的规避依据是 aten 参考实现"B=0 且带 bias 时 bias 输出值未定义"；若参考实现语义不同（B=0 带 bias 有定义），该锁定不应照抄。
- 推断可迁移：条件化输出用 `(0,)` 空张量的约定适用于整个 aclnn 生态（未验证，推断可迁移——结论按定义与芯片平台无关）。

## 坏实践

- 条件不成立时省略输出张量（7 元组变 5 元组）——破坏 ABI 槽位，解析/调用层直接崩。
- 用零填充（`(3H,)` 全 0）表达"无 bias 梯度"——shape 违反 §4 断言、数值语义错误（把"不存在"伪装成"恰为 0"）。
- 只生成 hasBias=true 一个分支的用例——C6 双态覆盖缺口。
- 把 B=0/H=0 输入当负向异常用例——它们是正向空张量场景（kernel 交付空张量分支）。

## 误诊

- 无（本卡结论在 trace 内有完整契约原文 + 608 行实测佐证；" `(0,)` vs 零填充"曾在 spec 期是真实分歧点，已由 proto.md §4 定案）。


