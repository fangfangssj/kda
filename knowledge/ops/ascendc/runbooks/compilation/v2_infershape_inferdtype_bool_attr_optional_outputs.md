---
type: "Runbook"
title: InferShape/InferDataType 模板：attr 驱动可选输出
description: V2 推导回调完整模板：GetBool(index) 读属性、has_bias 决定 bias 输出 (3H,)/(0,)、UNKNOWN_DIM 传播、非法输入 GRAPH_FAILED 拒收、OpDef+IMPL_OP 双注册。
tags: [task.implementation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# InferShape/InferDataType 模板：attr 驱动可选输出


## 现象

给一个自定义算子（GRU Cell 反传类：2 输入 `grad_hy(B,H)`/`storage(B,5H)`、5 输出、1 个布尔属性 `has_bias`，dtype 组合 {fp32, fp16, bf16}）打通 GE 图模式时，其宿主侧推导回调原本是桩：InferShape 只看 `grad_hy` 的 rank，把两路 bias 输出**固定**写成 `(3H,)`（注释自认"`has_bias=false → (0,)` 空张量分支留待真实实现"），不做 storage 形状约束、dtype、format 校验；InferDataType 无条件把首输入 dtype 抄给 5 个输出。桩状态下图执行直接找不到回调（注册问题，见"坏实践"交叉引用）；补上注册后，若不实现完整推导与校验，负向用例（非法 dtype/rank/shape/format）也无法在 GEIR 执行阶段被真实拒绝——工作流门禁要求负向用例必须以 `EXEC_FAILURE` 且 `precision_status` 以 `ERROR:` 开头收场，`INVALID`/SKIP/普通精度 FAIL 都不算拒绝。本卡是替换桩之后的**完整实现模板**及其在图模式下的实测行为。

## 适用场景

- 工程形态：AscendC 算子仓开发（ops-registry-invoke），为 GEIR/GE 图模式实现宿主侧 V2 推导回调（`gert::InferShapeContext` / `gert::InferDataTypeContext`）。
- API 路径：`exe_graph/runtime/infer_shape_context.h`、`infer_datatype_context.h`、`extended_kernel_context.h`（`GetAttrs()`）、`runtime_attrs.h`（`GetBool(index)`）、`register/op_impl_registry.h`（IMPL_OP 注册）；`UNKNOWN_DIM=-1`、`UNKNOWN_DIM_NUM=-2`。
- 验证过的平台与 CANN 版本：Ascend950 + CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——属性驱动可选输出、需要图阶段拒收非法输入的自定义算子。
- 路线优先级：**recommended_fastpath**。
- 明确不适用/未验证：负向用例的 GRAPH_FAILED 拒绝路径**未在本轨迹内验证**（负向套件未跑完）；动态 shape 套件（`-d=true`）与二进制通路（`-b=release`）未在本轨迹内完成；非 GE 图模式（aclnn 直调）不经图推理阶段，校验时机不同。

## 根因

（本卡为已验证实现模式，非缺陷修复）实现要点：图模式下输出 desc 在 InferShape 之前已按测试 CSV 预置，InferShape 推导结果必须与之自洽；dtype 推导与 shape 推导是两个独立回调，各自承担一类校验；属性只能按**声明序索引**读取（`GetAttrs()->GetBool(0)`），RuntimeAttrs 无按名读取接口。

## 设计决策

- **校验前移到推导回调**：TilingFunc 在编译期已做 dtype/rank/shape 校验，但 GE 图模式下 InferShape/InferDataType 在图推理阶段**先于** tiling 执行；要让负向用例在 GEIR 执行阶段"真实拒绝"（门禁原文：`全部在 GEIR 执行阶段真实拒绝（不接受 PASS/SKIP/INVALID/普通精度 FAIL）`），校验必须写在推导回调里并以 `GRAPH_FAILED` 收场，使图编译失败 → TTK 捕获异常 → `precision_status="ERROR: ..."`、`precision="EXEC_FAILURE"`。
- **属性按索引读**：`context->GetAttrs()` 返回 `const RuntimeAttrs*`，`GetBool(0)` 返回 `const bool*`（可能为 null，null 时回落默认值）——索引 = OpDef/REG_OP 的 Attr 声明序；头文件中无按名读取接口。
- **可选输出的空张量约定**：`has_bias=false` 时 bias 输出写 `(0,)` 空张量（而非不设置 shape）；空输出在 TTK 比较器里有专门语义（output/golden 双空 → PASS），写错维度会直接撞预置输出 desc。
- **UNKNOWN_DIM 传播**：`UNKNOWN_DIM=-1` 表示该维未知。校验侧：任一侧为 UNKNOWN_DIM 就跳过该维约束（支持动态 shape 用例）；推导侧：`H` 未知时 `3H` 也置 UNKNOWN_DIM（`threeH = (hiddenSize == UNKNOWN_DIM) ? UNKNOWN_DIM : 3 * hiddenSize`），不能对 -1 做乘法得出无意义的 -3。
- **format 校验的容忍值**：只接受 `FORMAT_ND` 与 `FORMAT_RESERVED`（后者视为未指定，不拒收）。
- **InferDataType 的备选**：`OpImplRegisterV2::InferOutDataTypeSameWithFirstInput()` 内建规则可让全部输出随首输入 dtype，免去自写函数；本实现保留显式函数，因为要顺带做 dtype 合法域 + 双输入一致校验（负向用例契约），内建规则不带校验。
- **双注册**：OpDef `SetInferShape/SetInferDataType` 绑定 + `IMPL_OP` V2 注册并存（缺后者的后果见坏实践第 1 条）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`x86_64-linux/include/exe_graph/runtime/*`、`register/*`）；构建 `cmake -S operators/<OP>_package -B <pkg>/build && cmake --build ... --target binary package -j`。
- 目标平台：Ascend950。
- 相关文件（与事件流实际读写一一对应）：
  - `operators/<OP>_package/op_host/thnn_fused_gru_cell_grad_infershape.cpp`——InferShape 实现（本文完整代码 A）
  - `operators/<OP>_package/op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp`——InferDataType + IMPL_OP 注册（本文完整代码 B）
  - `operators/<OP>_package/op_host/thnn_fused_gru_cell_grad_def.cpp`——OpDef（Input/Output/Attr 声明序决定属性索引）
  - `operators/<OP>_package/op_graph/thnn_fused_gru_cell_grad_proto.h`——REG_OP 镜像声明
- 契约参照（本算子）：`grad_hy(B,H)`、`storage(B,5H)`；输出 `grad_input_gates(B,3H)`、`grad_hidden_gates(B,3H)`、`grad_hx(B,H)`、`grad_input_bias(3H,)`/`(0,)`、`grad_hidden_bias(3H,)`/`(0,)`；dtype 全 IO 同型且 ∈ {DT_FLOAT, DT_FLOAT16, DT_BF16}；format 仅 ND。

## 完整代码

**A. `op_host/thnn_fused_gru_cell_grad_infershape.cpp`（终版全文，verbatim）**：

```cpp
/**
 * =============================================================================
 * <OP>_package/op_host/thnn_fused_gru_cell_grad_infershape.cpp
 * =============================================================================
 * Role: InferShape (V2) for ThnnFusedGruCellGrad（经 OpDef SetInferShape 绑定，
 *       非 IMPL_OP_INFERSHAPE 注册）。按 proto.md §4 Shape 契约由 grad_hy (B, H)
 *       派生五个输出的 shape，并对输入 rank、storage=(B,5H) 约束、输入/输出
 *       format（仅 ND）进行校验，非法输入返回 GRAPH_FAILED。
 * =============================================================================
 */

#include "register/op_def_registry.h"

namespace ge {

// [REF_SAMPLE] 原样例工程的等形推导（x/y rank 相等 + UNKNOWN_DIM 互补
// 解析 + 逐维相等校验 → z 随 x）整段注释保留，全文见 git 历史。本算子输出
// shape 不随输入等形传播，改按 GRU 反传契约派生。

graphStatus InferShape(gert::InferShapeContext* context)
{
    // ── 输入存在性校验（缺失必选输入 → GRAPH_FAILED）──
    const gert::Shape* gradHyShape = context->GetInputShape(0);
    if (gradHyShape == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* storageShape = context->GetInputShape(1);
    if (storageShape == nullptr) {
        return GRAPH_FAILED;
    }

    // ── rank==2 强校验（proto.md §4：2D ND 张量）──
    if (gradHyShape->GetDimNum() != 2 || storageShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    const int64_t batchSize = gradHyShape->GetDim(0);
    const int64_t hiddenSize = gradHyShape->GetDim(1);
    const int64_t storageB = storageShape->GetDim(0);
    const int64_t storageH = storageShape->GetDim(1);

    // ── shape 约束：storage.shape == (B, 5H)（proto.md §4）
    //    未知维（UNKNOWN_DIM=-1）跳过对应维的校验，支持动态 shape。──
    if (batchSize != UNKNOWN_DIM && storageB != UNKNOWN_DIM && storageB != batchSize) {
        return GRAPH_FAILED;
    }
    if (hiddenSize != UNKNOWN_DIM && storageH != UNKNOWN_DIM && storageH != 5 * hiddenSize) {
        return GRAPH_FAILED;
    }

    // ── has_bias 属性（OpDef attr index=0，默认 false）──
    bool hasBias = false;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* hasBiasPtr = attrs->GetBool(0);
        if (hasBiasPtr != nullptr) {
            hasBias = *hasBiasPtr;
        }
    }

    // ── format 校验：输入/输出仅支持 ND（proto.md §1 OpDef Format 声明）──
    for (size_t i = 0; i < 2; ++i) {
        const auto* desc = context->GetInputDesc(i);
        if (desc != nullptr) {
            const ge::Format fmt = desc->GetStorageFormat();
            if (fmt != ge::FORMAT_ND && fmt != ge::FORMAT_RESERVED) {
                return GRAPH_FAILED;
            }
        }
    }
    for (size_t i = 0; i < 5; ++i) {
        const auto* desc = context->GetOutputDesc(i);
        if (desc != nullptr) {
            const ge::Format fmt = desc->GetStorageFormat();
            if (fmt != ge::FORMAT_ND && fmt != ge::FORMAT_RESERVED) {
                return GRAPH_FAILED;
            }
        }
    }

    // ── 输出 shape 派生（proto.md §4）──
    gert::Shape* gradInputGatesShape = context->GetOutputShape(0);
    gert::Shape* gradHiddenGatesShape = context->GetOutputShape(1);
    gert::Shape* gradHxShape = context->GetOutputShape(2);
    gert::Shape* gradInputBiasShape = context->GetOutputShape(3);
    gert::Shape* gradHiddenBiasShape = context->GetOutputShape(4);
    if (gradInputGatesShape == nullptr || gradHiddenGatesShape == nullptr ||
        gradHxShape == nullptr || gradInputBiasShape == nullptr ||
        gradHiddenBiasShape == nullptr) {
        return GRAPH_FAILED;
    }

    // 未知维传播：hiddenSize==UNKNOWN_DIM → 3H 也为 UNKNOWN_DIM
    const int64_t threeH = (hiddenSize == UNKNOWN_DIM) ? UNKNOWN_DIM : 3 * hiddenSize;

    // grad_input_gates / grad_hidden_gates: (B, 3H)，行内三段门梯度。
    gradInputGatesShape->SetDimNum(2);
    gradInputGatesShape->SetDim(0, batchSize);
    gradInputGatesShape->SetDim(1, threeH);

    gradHiddenGatesShape->SetDimNum(2);
    gradHiddenGatesShape->SetDim(0, batchSize);
    gradHiddenGatesShape->SetDim(1, threeH);

    // grad_hx: (B, H)，随 grad_hy。
    gradHxShape->SetDimNum(2);
    gradHxShape->SetDim(0, batchSize);
    gradHxShape->SetDim(1, hiddenSize);

    // grad_input_bias / grad_hidden_bias: (3H,) if has_bias, (0,) if not
    const int64_t biasDim = hasBias ? threeH : 0;
    gradInputBiasShape->SetDimNum(1);
    gradInputBiasShape->SetDim(0, biasDim);
    gradHiddenBiasShape->SetDimNum(1);
    gradHiddenBiasShape->SetDim(0, biasDim);

    return GRAPH_SUCCESS;
}

} // namespace ge
```

**B. `op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp`（终版全文，verbatim）**：InferDataType 实现 + IMPL_OP V2 注册（与本仓库另一张卡「EZ9999 找不到 infer 回调 — 补 IMPL_OP 注册」共用同一文件，此处完整内联保证本卡自包含）：

```cpp
/**
 * =============================================================================
 * <OP>_package/op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp
 * =============================================================================
 * Role: InferShape + InferDataType (V2) for ThnnFusedGruCellGrad。
 *       同时经 OpDef SetInferShape/SetInferDataType 绑定和 IMPL_OP V2 注册
 *       （OpImplRegisterV2），确保 runtime 能通过 RegisterOpImplToRegistry
 *       发现回调。五个输出 dtype 一律随 grad_hy（proto.md §2 三组合内逐 IO 同
 *       dtype）。校验输入 dtype 在合法域 {DT_FLOAT, DT_FLOAT16, DT_BF16} 内
 *       且双输入一致，非法输入返回 GRAPH_FAILED。
 * =============================================================================
 */

#include "register/op_impl_registry.h"

namespace ge {

// [REF_SAMPLE] 原样例工程 InferDataType（x/y dtype 相等校验后写单输出 z）
// 整段注释保留，全文见 git 历史。本实现按 GRU 反传契约改为五输出随首输入，
// 并增加 dtype 合法域 + 双输入一致校验。

graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType gradHyDtype = context->GetInputDataType(0);
    const ge::DataType storageDtype = context->GetInputDataType(1);

    // ── dtype 合法域校验：仅 {DT_FLOAT, DT_FLOAT16, DT_BF16}（proto.md §2）──
    const bool gradHyOk =
        (gradHyDtype == ge::DT_FLOAT) || (gradHyDtype == ge::DT_FLOAT16) || (gradHyDtype == ge::DT_BF16);
    const bool storageOk =
        (storageDtype == ge::DT_FLOAT) || (storageDtype == ge::DT_FLOAT16) || (storageDtype == ge::DT_BF16);
    if (!gradHyOk || !storageOk) {
        return GRAPH_FAILED;
    }

    // ── 双输入 dtype 一致校验（proto.md §2：三组合内逐 IO 同 dtype）──
    if (gradHyDtype != storageDtype) {
        return GRAPH_FAILED;
    }

    // ── 五输出 dtype 随 grad_hy ──
    for (size_t i = 0; i < 5; ++i) {
        if (context->SetOutputDataType(i, gradHyDtype) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

// 前置声明（定义在 op_host/thnn_fused_gru_cell_grad_infershape.cpp）
graphStatus InferShape(gert::InferShapeContext* context);

} // namespace ge

// ── V2 注册（OpImplRegisterV2）：使 InferShape/InferDataType 进入
//    RegisterOpImplToRegistry 通路，供 runtime MergeTypesToImpl 发现。
//    与 OpDef SetInferShape/SetInferDataType 并行，两者注册同一函数指针。──
IMPL_OP(ThnnFusedGruCellGrad)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
```

## 预防

- 写推导回调前先列负向用例清单（dtype 越界/dtype 不一致/rank 错/shape 违约/format 非 ND），逐条确认由哪个回调以 `GRAPH_FAILED` 拒收；TilingFunc 校验不能替代图推理阶段校验（执行时机不同）。
- 属性读取只用索引，写完 OpDef 后立即在注释里标注每个属性的 index，防止后续插入属性导致索引错位。
- 含可选输出（属性开关）的算子，把"开关关闭 → 输出 `(0,)`"写进推导，并与测试 CSV 的期望输出 shape 对齐。

## 验证信号

以下信号来自该算子 GEIR 图模式联调：TTK GEIR 正向源码静态套件（隔离 staging、Ascend950 上板执行）的结果 CSV 与 verify 输出；验证发生在推导回调补齐 IMPL_OP 注册之后的整轮运行。

- 整轮 44 个正向用例全部完成执行并进入精度比对（无 `Can not find Node`/EXEC_FAILURE 级失败）（来源平台：Ascend950；证据等级：tool_observed）
- 按序校验通过前 19 个用例，覆盖 float32 / float16 / bfloat16 × `hasBias` true/false 全组合（hasBias=true 8 例、false 11 例），`cst_precision` 严格 `PASS`（来源平台：Ascend950；证据等级：tool_observed）
- `hasBias=true` 用例 bias 输出 `(3H,)` 与 `hasBias=false` 用例 bias 输出 `(0,)` 均通过图准备与精度比对——输出 desc 由 CSV 预置，InferShape 推导结果与其一致才不冲突（来源平台：Ascend950；证据等级：tool_observed）
- 12 个常规数值域用例（输入不含 Inf/NaN）全部 PASS；另 7 个含 Inf/NaN 极端输入用例的 PASS 发生在比对钩子已放宽 NaN/Inf 判定的运行中（见失效边界），不单独作为形状/dtype 推导正确性的依据（来源平台：Ascend950；证据等级：tool_observed）
- 负向 49 用例的 GRAPH_FAILED 拒绝路径（EXEC_FAILURE+ERROR:）：**未验证**——负向套件未在本轨迹内执行完成（来源平台：Ascend950；证据等级：tool_observed，"套件未完成"本身为命令输出观测）

## 性能影响

无性能故事（推导回调为宿主侧一次性逻辑，本轨迹无相关 profiling 数据）。

## 失效边界

- 模板中的形状契约（`storage=(B,5H)`、`3H` 派生、bias `(0,)/(3H,)`）是本算子的语义，跨算子复用时替换为各自契约；可迁移的是结构：存在性校验 → rank → 逐维约束（UNKNOWN_DIM 跳过）→ 属性索引读取 → format 校验 → 输出派生（UNKNOWN_DIM 传播）。
- 负向拒绝路径、动态 shape 套件、二进制通路未验证；其他 CANN 版本未验证（推断可迁移，未证实）。
- 含 Inf/NaN 输入用例的 PASS 信号混入了比对钩子的 NaN/Inf 判定放宽（测试侧放松，非本卡内容），以其证明模板正确性需排除该因素——常规数值域用例不受影响。
- 属性读取按索引与 OpDef/REG_OP 声明序绑定：声明序变更（增删/移动 Attr）会静默改变索引语义。

## 坏实践

- **只绑定不注册**：函数经 OpDef `SetInferShape/SetInferDataType` 绑定、无 `IMPL_OP` V2 注册——图执行阶段直接 `EZ9999 Can not find Node ... custom infer_datatype func`，全量用例 EXEC_FAILURE（详见本仓库另卡「EZ9999 找不到 infer 回调 — 补 IMPL_OP 注册」）。
- **桩式固定输出**：把 bias 输出固定 `(3H,)` 不看属性——`has_bias=false` 用例的输出 shape 直接错误（预置 desc 与推导冲突）。
- **对 UNKNOWN_DIM 做算术**：`3 * (-1) = -3` 会产生非法维；必须显式传播 `UNKNOWN_DIM`。
- **把校验全部推给 TilingFunc**：图模式下推导回调先于 tiling 执行，负向用例会在图推理阶段被放过，无法满足"GEIR 执行阶段真实拒绝"的门禁。

## 误诊

无（本卡为实现模式卡；联调期的注册缺失与两处误诊见本仓库另卡「EZ9999 找不到 infer 回调 — 补 IMPL_OP 注册」误诊段）。

## 未决事项

- 负向用例在本模板下的真实拒绝表现（EXEC_FAILURE+ERROR: 是否稳定达成、`InferShape` 拒收与 `InferDataType` 拒收的报错可辨识度）：负向套件未完成，待下轮验证补齐。
- 动态 shape（`-d=true`，desc 全 -1）与二进制通路（`-b=release`）下本模板的行为：未验证。
