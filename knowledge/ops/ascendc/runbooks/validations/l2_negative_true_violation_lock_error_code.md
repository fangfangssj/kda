---
type: "Runbook"
title: aclnn L2 负向须真实违约并锁错误码
description: 黑盒 ACLNN ST 的 L2 负向用例每行必须真实违反至少一条 proto 契约、remark 锁定期望 ACL 错误码；缺必选参数用 None 形状锁原始输入，不造派生量。
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["agnostic"]
---
# aclnn L2 负向须真实违约并锁错误码


## 现象

在验收对标 `aten::_thnn_fused_gru_cell_backward` 的算子的黑盒 ACLNN ST 用例时，对 L2 负向 CSV（`aclnn<Op>_l2_exception.csv`，49 行）做覆盖度审计：每行必须在 `remark` 列锁定期望错误码（如 `dtype不在合法域; expect ACL_ERROR_DTYPES_NOT_MATCH`），且每行必须**真实**违反至少一条 proto 契约——不能"看起来负向、实则合法"。49 行的违约类型分布：dtype 违约 14（用 `bool` / `complex128` 等非法 dtype）、format 违约 7（单张量 `FRACTAL_NZ`）、shape 违约 5（把输出张量写成 `(999,999)`）、维度类 14、约束场景 EX-S1~S8 各 1（共 8）、缺必选参数 1（`grad_hy` 形状写 `None`，`必选参数缺失; expect ACL_ERROR_INVALID_PARAM`）。

负向行的实际形态（L2 首行 verbatim，第一张量 dtype 为 `bool` 触发 dtype 违约）：

```csv
aclnnThnnFusedGruCellGrad_L2_exc_dtype_001,aclnnThnnFusedGruCellGrad,"((42,64,),(42,320,),(42,192,),(42,192,),(42,64,),(192,),(192,),)","('bool','float32','float32','float16','bfloat16','float16','bfloat16',)","('ND','ND','ND','ND','ND','ND','ND',)",,"{""hasBias"":true}","(2,3,4,5,6,)",,,"((inf,inf,),(0,0,),)",,dtype不在合法域; expect ACL_ERROR_DTYPES_NOT_MATCH
```

## 适用场景

- 工程形态：黑盒 ACLNN ST 用例设计（L2 负向层）与验收；aclnn host 层（op_host GetWorkspaceSize 参数校验）实现错误码口径时同样适用（负向用例与校验实现共用同一错误语义表）。
- API 路径：TTK aclnn CSV（`tensor_view_shapes`/`tensor_dtypes`/`tensor_formats`/`attributes`/`remark` 列）、aclnn 参数校验错误码（`ACL_ERROR_*`）、约束编号 R1~R9 → 错误码映射。
- 验证过的平台与 CANN 版本：无上板/TTK 执行实测（host 侧 CSV 静态校验 + 程序化违约判定）；期望错误码为设计期锁定值，运行期实际返回码未验证。结论属 aclnn 校验语义/ST 约定层，按定义与芯片平台无关（`platforms: [agnostic]`）；算子目标产品 Ascend 950PR / 950DT 为设计声明。CANN 版本未探测到。
- 适用粒度：`platform_wide`——凡走 TTK/aclnn CSV 的 L2 负向设计。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：错误码的具体枚举值是否随 CANN 版本变化（未验证，使用前应以一次真实负向执行校准）；白盒 ST 的负向表达（产物链不同）。

## 根因

负向用例的判定权在执行期：aclnn 层参数校验失败时返回错误码，"负向用例通过" = "执行返回了锁定的期望错误码"。因此：① 行内容必须真实违反至少一条契约——否则执行时它会当正向用例跑通，负向断言落空；② 每行必须在 remark 锁定期望错误码——否则执行期无法区分"预期失败"与"意外失败"；③ 缺必选参数的场景必须锁定**原始输入**（把该张量形状写 `None`），不得通过篡改派生量制造异常。错误码与约束的映射口径（本例实测锁定的枚举）：形状强绑定违约（R1）→ `ACL_ERROR_SHAPE_NOT_MATCH`；dtype 不一致/非法（R2/R3）→ `ACL_ERROR_DTYPES_NOT_MATCH`；格式不支持 → `ACL_ERROR_FORMAT_NOT_SUPPORT`；输入维度违例（R7，非 2D）→ `ACL_ERROR_INVALID_DIMENSION`；必选输入空指针（R9，null_input）→ `ACL_ERROR_INVALID_PARAM`。

## 设计决策

- 违约载体选择：dtype 族用 `bool`/`complex128` 等明确非法 dtype（而非仅两合法 dtype 互混——两者都构成违约，非法 dtype 更不可争辩）；format 族只改单张量格式为 `FRACTAL_NZ`（保持其他张量合法，让违约点唯一可定位）；shape 族改输出张量形状为 `(999,999)`（违约在输出侧，与输入侧 shape_mismatch 场景区分）；维度族按 EX-S7/S8 物化为 3 维/1 维输入（真实 rank 违约）。
- 缺参表达：必选输入缺失用 `None` 占位于 `tensor_view_shapes` 元组（保留 7 槽结构），锁定原始输入而非派生量（公共 Skill 约定：负向场景必须锁定原始输入，不得锁定派生量制造异常）。
- EX-S1~S8 每个异常子场景物化恰一行 + 随机负向填充到规模（49 = 8 EX + 41 随机分布各违约族）。
- 当前方案属哪档路线：recommended_fastpath。

## 系统信息

- 工程形态：黑盒 ACLNN ST 用例验收（ascendc-st-design 产物链）。
- 相关文件清单（与事件流实际读取一一对应）：
  - `operators/<Op>_package/tests/blackbox/testcases/aclnn<Op>_l2_exception.csv`（49 行负向）
  - `operators/<Op>_package/tests/blackbox/design/03_scenario_enumeration.md`（异常场景 EX-S1~S8 表 + 期望错误码）
  - `operators/<Op>_package/tests/blackbox/design/01_parameter_description.md`（R1~R9 约束与错误语义）
  - `docs/<Op>/develop/proto.md`（§4 契约——违约判定的基准）

## 完整代码

EX-S1~S8 异常场景表（期望错误码锁定的锚点，`03_scenario_enumeration.md` verbatim）：

```markdown
| 子场景 ID | locked因子 | 预期错误码 | 测试重点 |
|----------|-----------|-----------|---------|
| EX-S1 | grad_hy.shape=[2,64], grad_hy.dimensions=2, storage.shape=[2,321], storage.dimensions=2 | ACL_ERROR_SHAPE_NOT_MATCH | 违反 R1：storage 第二维 321 非 5 的倍数 |
| EX-S2 | grad_hy.shape=[2,64], grad_hy.dimensions=2, storage.shape=[3,320], storage.dimensions=2 | ACL_ERROR_SHAPE_NOT_MATCH | 违反 R1：B 维不一致（2≠3） |
| EX-S3 | grad_hy.shape=[2,64], grad_hy.dimensions=2, storage.shape=[2,640], storage.dimensions=2 | ACL_ERROR_SHAPE_NOT_MATCH | 违反 R1：第二维 640=5×128 ≠ 5×64（5 的倍数但 H 不匹配） |
| EX-S4 | grad_hy.dtype=float32, storage.dtype=float16 | ACL_ERROR_DTYPES_NOT_MATCH | 违反 R2：两输入 dtype 不一致 |
| EX-S5 | grad_hy.dtype=float32, grad_input_gates.dtype=float16 | ACL_ERROR_DTYPES_NOT_MATCH | 违反 R3：输出 dtype 与输入不一致 |
| EX-S6 | storage.exist=False | ACL_ERROR_INVALID_PARAM | 违反 R9：必选输入 storage 空指针（null_input） |
| EX-S7 | _b.value=2, _h.value=64, grad_hy.shape=[2,64,1], grad_hy.dimensions=3 | ACL_ERROR_INVALID_DIMENSION | 违反 R7：grad_hy 3 维（非 2D） |
| EX-S8 | _b.value=2, _h.value=64, storage.shape=[320], storage.dimensions=1 | ACL_ERROR_INVALID_DIMENSION | 违反 R7：storage 1 维（非 2D） |
```

各类违约的 CSV 实测行（verbatim）：

```csv
aclnnThnnFusedGruCellGrad_L2_exc_missing_027,…,"(None,(186,10240,),(186,6144,),(186,6144,),(186,2048,),(6144,),(6144,),)",…,"{""hasBias"":false}",…,必选参数缺失; expect ACL_ERROR_INVALID_PARAM
aclnnThnnFusedGruCellGrad_L2_exc_constraint_047,…,"((1,936,),None,(1,2808,),(1,2808,),(1,936,),(2808,),(2808,),)",…,"{""hasBias"":false}",…,EX-S6; 约束违反; expect ACL_ERROR_INVALID_PARAM
aclnnThnnFusedGruCellGrad_L2_exc_constraint_048,…,"((2,64,1,),(12,1025,),(12,615,),(12,615,),(12,205,),(0,),(0,),)",…,EX-S7; 约束违反; expect ACL_ERROR_INVALID_DIMENSION
aclnnThnnFusedGruCellGrad_L2_exc_shape_022,…,"((5,1176,),(5,5880,),(999,999,),(5,3528,),(5,1176,),(0,),(0,),)",…,输出shape与期望不匹配; expect ACL_ERROR_SHAPE_NOT_MATCH
aclnnThnnFusedGruCellGrad_L2_exc_format_003,…,"('FRACTAL_NZ','ND','ND','ND','ND','ND','ND',)",…,format不支持; expect ACL_ERROR_FORMAT_NOT_SUPPORT
```

（`None` 占位保留 7 槽结构；EX-S6 的 `None` 落在 storage 槽位 = 必选输入空指针。）

## 预防

- L2 生成后必须机检两项：① 每行至少违反一条契约（逐行重算 dtype 合法性/一致性、shape 契约、格式域、张量数）；② 每行 remark 含 `expect <错误码>`。任一不满足即生成缺陷。
- 期望错误码从"约束编号 → 错误码"映射表推导（R1→SHAPE_NOT_MATCH、R2/R3→DTYPES_NOT_MATCH、R7→INVALID_DIMENSION、R9→INVALID_PARAM、格式→FORMAT_NOT_SUPPORT），不要凭印象写。
- 缺参场景锁原始输入（`None` 占位），不改派生张量造异常。
- 上板校准：首轮真实执行后核对 ACL 实际返回码与锁定值的一致性（本 trace 未做，见失效边界）。

## 验证信号

来源：同一次黑盒 ACLNN ST 用例验收会话——独立校验脚本对 49 行 L2 逐行做"真实违约"判定（重算 dtype 合法/一致、§4 shape 契约、ND 格式、7 张量结构，任一违约才算真负向）与 remark expect 检查。

- 49/49 行判定为真实负向（无一行"看似负向实合法"），违约类型分布 dtype_mismatch 14 / format 7 / shape 5 / dimension 14 / EX-scenario 8 / missing_param 1，脚本输出 `NO CONTRACT ERRORS`、EXIT=0（来源平台：agnostic——host 侧 CSV 静态校验；证据等级：tool_observed）。
- 缺参行 `L2_exc_missing_027` 与 EX-S6 行 `L2_exc_constraint_047` 均以 `None` 占位表达空指针，remark 锁 `ACL_ERROR_INVALID_PARAM`（来源平台：agnostic；证据等级：tool_observed）。

（期望错误码的具体值为设计期锁定、运行期实际返回码未上板验证——该未验证声明见失效边界与遗留问题，不作为验证信号。）

## 失效边界

- 期望错误码枚举值未运行期验证：不同 CANN 版本/校验实现顺序可能返回不同码（尤其一行含多重违约时，先命中哪个校验由实现决定）；采用前应以一次真实负向执行校准。
- "每行恰一类违约"并非总能保证（随机负向填充的行可能同时违约多项）——此时期望码应取校验实现最先命中的那条；单违约载体（如只改单张量格式）是让违约点可定位的工程手段而非硬约束。
- 推断可迁移：负向"真实违约 + 锁码 + None 缺参"的构造规则适用于整个 TTK/aclnn CSV 生态（未验证，推断可迁移——结论按定义与芯片平台无关）。

## 坏实践

- 生成"看起来负向"的行（如只改 remark 不改行内容、或行内容其实满足全部契约）——执行期当正向跑通，负向层形同虚设。
- 不锁期望错误码——执行期"预期失败"与"意外失败"无法区分，负向断言没有牙齿。
- 用篡改派生量（如硬编输出 shape 违约来冒充"缺输入"）制造异常——应锁原始输入。
- 把 `ACL_ERROR_*` 具体枚举值当成跨版本恒定——未校准就断言。

## 误诊

- 把"负向行执行失败"一概当算子缺陷——若行内容违约类型与锁定错误码不匹配（如行内只有 dtype 违约却锁定维度错误码），失败是用例设计问题而非算子问题（详见姊妹话题：L2 维度类负向期望码与实际违约不符）。


