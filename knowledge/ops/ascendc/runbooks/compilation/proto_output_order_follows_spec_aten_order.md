---
type: "Runbook"
title: proto 输出序以 spec/aten 序为准，勿照抄 vendor 基线
description: 前代 vendor/旧工作区 IR 输出序 grad_hx 在首（校验器不变量挂首输出），新 spec 按 aten 返回序；OpDef 与 kernel 签名输出序必须同源于权威 spec。
tags: [task.migration, topic.implementation.operator_registration, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# proto 输出序以 spec/aten 序为准，勿照抄 vendor 基线


## 现象

重开 ThnnFusedGruCellGrad（GRU 单步融合反向，5 路输出）开发时，手头有三份口径不一致的"接口真值"：

1. **前代安装基线**（`$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_proto/inc/op_proto.h` 的 REG_OP）与前代完整开发产物（旧工作区 proto.md / op_host def.cpp / kernel 入口）：输出序 **grad_hx 在首**（grad_hx, grad_input_gates, grad_hidden_gates, grad_input_bias, grad_hidden_bias）。
2. **旧工作区 spec.yaml**：`outputs[0] = grad_hx`，注释写明"值级不变量（零吸收）由校验器 stage 11 挂在首个输出上机器执行"——前代是**故意**把 grad_hx 放首位。
3. **当前工作区（本次重做）spec.yaml**：按 **aten 返回序**（grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias）。

生成 proto.md（OpDef + kernel 入口签名）时必须裁定用哪个序——OpDef 的 `.Output()` 声明序、GEIR `.OUTPUT` 序、kernel 入口的 `GM_ADDR` 输出参数位序、后续 aclnn 适配层的位置映射四处联动，序错了后面全错位。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发——proto/OpDef 生成、kernel 入口签名推导、后续 aclnn/op-plugin 位置映射。
- API 路径：OpDef `this->Output("<name>")` 声明序、GEIR `REG_OP .OUTPUT(<name>, ...)` 序、kernel 入口 `GM_ADDR` 输出参数位序。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR/Ascend950DT 工具链）+ CANN 9.0.0-beta.2（host 侧 artifact 核验；未上板）。
- 适用粒度（transfer_scope）：operator_family——多输出融合反向算子族（凡 torch 返回 tuple 且存在前代基线/spec 双口径的场景）。
- 路线优先级：recommended_fastpath——无内置同名时以当前工程 spec.yaml 为接口事实源，输出序对齐 aten 返回序。
- 明确不适用 / 未验证场景：内置注册表**有**同名 REG_OP 的算子（内置 IR 为权威，须一致或写「内置同名 proto 裁定」块，本卡 fastpath 不适用）；spec 与 aten 语义冲突时的裁决不属本卡。

## 判别方法

**根因（单句）**：前代 IR 把 grad_hx 放输出首位是"把 spec 校验器 stage-11 值级不变量挂在首个输出上"的设计选择，而当前 spec 为简化 op-plugin 位置映射改按 aten 返回序——两种序各自合法，但 OpDef/GEIR/kernel 参数位序/aclnn 位置映射必须**同源一致**，照抄前代基线序会与新 spec 错位（aclnn 按位置取输出时取到错误的梯度）。

**裁定链（本会话实际执行）**：

1. 内置检查：`grep -rln "ThnnFusedGruCellGrad" $ASCEND_OPP_PATH/built-in/op_proto/inc/` → `grep-exit:1`（无内置同名）→ 无内置 IR 权威约束。
2. 任务规定：以本工作区 REQUIREMENTS.md / spec.yaml 为准独立裁定；任务输入无 origin_sources 参考源（旧工作区产物仅为参考）→ spec.yaml 是接口事实源。
3. spec.yaml outputs 逐字读出权威序：grad_input_gates → grad_hidden_gates → grad_hx → grad_input_bias → grad_hidden_bias（aten 返回序）。
4. OpDef `.Output()` 与 kernel 签名输出参数按同一序生成（见完整代码）。

## 解决配方

**判别命令（新旧口径输出序对比，生成 proto 前必跑）**：

```bash
# 当前工作区 spec 输出序（权威）
sed -n '/^outputs:/,/^reduction:/p' docs/ThnnFusedGruCellGrad/spec.yaml | grep -E "^  - name:"
## → grad_input_gates / grad_hidden_gates / grad_hx / grad_input_bias / grad_hidden_bias

## 前代/旧工作区口径（对照，勿照抄）
sed -n '/^outputs:/,/^reduction:/p' <旧工作区>/docs/ThnnFusedGruCellGrad/spec.yaml | grep -E "^  - name:"
## → grad_hx（首位，注释：stage-11 不变量挂首输出）/ grad_input_gates / ...
```

**生成后自检（dtype 表与数组长度一致性）**：

```bash
F=docs/ThnnFusedGruCellGrad/develop/proto.md
sed -n '/## 2/,/## 3/p' $F | grep "^|"                          # 列名 = IO 名按 OpDef 序；数据行数 = dtype 组合数
grep -o "DataType({[^}]*})" $F | awk -F',' '{print NF}' | sort -u  # 每个 DataType 数组长度 == 组合数（本例 3）
grep -o "Format({[^}]*})" $F   | awk -F',' '{print NF}' | sort -u  # Format 数组同长
```

（注意 `grep -c "^| float"` 会因 bfloat16 行不以 `| float` 开头而少计——用上表的数组长度法或目测表内容。）

## 设计决策

- **输出序对齐 aten 返回序**：op-plugin 适配层按位置映射 torch 返回 tuple ↔ 算子输出，同序可省掉位置重排；前代"grad_hx 首位"是为 spec 校验器 stage-11 的首输出不变量挂载——新 spec 已不采用该挂载口径，两代口径差异是设计演进而非对错。
- **kernel 入口签名由 OpDef 推导**：输入按 OpDef Input 序、输出按 OpDef Output 序，尾接 workspace/tiling 两个系统参数；本例尾参命名沿用前代 `gruCellGradWorkspace / gruCellGradTiling`（样例惯例是 `workspace/tiling`，LSTM 孪生为 `workspace/lstmCellGradTiling`——命名选择不改变位序语义）。
- **proto 签名不带实现分支形参**：前代 kernel 的 `template <bool isGroup, bool isEmptyTensor>` 是实现期分支决策，proto 阶段先于它，签名不写模板形参（前代 proto.md 先例同）。
- **supported_chips 映射**：spec 的 [Ascend950PR, Ascend950DT] → 单条 `AddConfig("ascend950")`（样例/前代 proto 同款，两个 chip 变体同属 ascend950 arch）。

## 系统信息

- toolchain：CANN 9.0.0-beta.2（x86_64-linux），目标 Ascend950PR/Ascend950DT。
- 相关文件（本会话实际读/写）：
  - 当前工作区 `docs/ThnnFusedGruCellGrad/spec.yaml`（权威 outputs 序）
  - 旧工作区 `docs/ThnnFusedGruCellGrad/spec.yaml`、`develop/proto.md`（前代口径对照）
  - 旧工作区 `operators/ThnnFusedGruCellGrad_package/op_host/thnn_fused_gru_cell_grad_def.cpp`、`op_kernel/thnn_fused_gru_cell_grad_apt.cpp`（前代 OpDef/kernel 实况）
  - `$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_proto/inc/op_proto.h`（前代安装基线 IR）
  - 当前工作区 `operators/add_custom_package/`（样例 OpDef/kernel 惯例）
  - `docs/ThnnFusedGruCellGrad/develop/proto.md`（产出）

## 完整代码

产出的 OpDef（输出按当前 spec = aten 返回序，verbatim）：

```cpp
namespace ops {
class ThnnFusedGruCellGrad : public OpDef {
public:
    explicit ThnnFusedGruCellGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_hy").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("storage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_input_gates").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_hidden_gates").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_hx").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_input_bias").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_hidden_bias").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("has_bias").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend950");
    }
};
OP_ADD(ThnnFusedGruCellGrad);
} // namespace ops
```

产出的 kernel 入口签名（输出参数按 OpDef 序）：

```cpp
__global__ __aicore__ void thnn_fused_gru_cell_grad(GM_ADDR grad_hy, GM_ADDR storage,
    GM_ADDR grad_input_gates, GM_ADDR grad_hidden_gates, GM_ADDR grad_hx, GM_ADDR grad_input_bias,
    GM_ADDR grad_hidden_bias, GM_ADDR gruCellGradWorkspace, GM_ADDR gruCellGradTiling)
```

负例对照——前代基线序（**勿照抄**，vendor REG_OP 输出段逐字）：

```cpp
REG_OP(ThnnFusedGruCellGrad)
    .INPUT(grad_hy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(storage, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_hx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_input_gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_hidden_gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_input_bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_hidden_bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(has_bias, Bool, false)
    .OP_END_FACTORY_REG(ThnnFusedGruCellGrad)
```

dtype 组合列表（产出原文，列名 = IO 名按 OpDef 序，3 组合 × 7 列）：

```
| grad_hy | storage | grad_input_gates | grad_hidden_gates | grad_hx | grad_input_bias | grad_hidden_bias |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| float16 | float16 | float16 | float16 | float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
```

## 预防

- 生成任何多输出算子的 proto 前，先逐字读当前 spec.yaml 的 outputs 段拿权威序；存在前代/vendor 基线时显式 diff 两者，不一致必须裁定并记录。
- OpDef Output 序、GEIR `.OUTPUT` 序、kernel 签名输出参数序、aclnn 适配层位置映射四处必须同一序——改任何一处都要联动核查其余三处。
- dtype 表列名 = IO 名按 OpDef 序；DataType/Format 数组长度必须等于 dtype 组合数（自检命令见解决配方）。

## 验证信号

本卡信号来自一次 GRU Cell 融合反向算子原型提取会话（host 侧命令实测与 artifact 核验）：

- 产出 proto.md 的 OpDef/kernel 签名输出序 = 当前 spec.yaml outputs 序（grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias），验收 5 项 PASS：test -f / OP_ADD( / public OpDef / __global__ __aicore__ / 无内置同名（来源平台：950——CANN 9.0.0-beta.2 工具链 host 侧；证据等级：tool_observed）
- dtype 表 3 数据行（fp32/fp16/bf16）× 7 列（2 入 5 出）核对通过，全部 DataType/Format 数组长度均=3（来源平台：950——host 侧；证据等级：tool_observed）
- 新旧 spec 输出序差异（旧 outputs[0]=grad_hx + stage-11 注释 vs 新 aten 序）与 vendor 基线 IR（grad_hx 首位）均为工作区/安装包 artifact 逐字读证（来源平台：950——CANN 9.0.0-beta.2 安装包与工作区文件；证据等级：tool_observed）

## 失效边界

- 裁定规则前提是"无内置同名 REG_OP"（本例 grep exit 1 实证）；内置存在时内置 IR 为权威（须一致或写裁定块），不适用本卡 fastpath。
- 输出序对齐 aten 的收益绑定"op-plugin/aclnn 按位置映射"的适配模式；若适配层按名映射，序的收益减弱（但四处同源一致的要求不变）。
- 未验证：kernel 尾参命名（gruCellGradWorkspace vs workspace）是否存在编译器侧约束（两者均为合法形参名、位序语义不变；本会话未编译验证）。

## 坏实践

- 照抄 vendor/前代基线的 IR 输出序生成新 OpDef——与当前 spec 错位，aclnn 按位置取输出会取错梯度路。
- OpDef 与 kernel 签名用不同序——kernel 签名必须由 OpDef 推导，不能一个按 spec、一个按前代。
- proto 阶段把实现期分支形参（`template <bool ...>`）写进 kernel 签名。
- 用 `grep -c "^| float"` 数 dtype 表行数（bfloat16 行不匹配，少计）。

## 误诊

- 无（本会话无错误归因）。注意：前代 grad_hx 首位是合法设计选择（服务 spec 校验器首输出不变量挂载），勿误判为"前代写错了"——差异是两代 spec 的挂载口径不同。

## 未决事项

- 未验证：aten 返回序在后续 aclnn 适配层的实际位置映射收益（须等本工程 aclnn 适配实现后核对，下个会话可接续）。
- 未做：新旧两种输出序在 GEIR/op-plugin 双路径下的回归对比（本会话为 proto 阶段，无运行产物）。
