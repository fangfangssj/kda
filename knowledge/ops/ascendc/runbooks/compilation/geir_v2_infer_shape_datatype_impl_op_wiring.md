---
type: "Runbook"
title: GEIR 图模式 V2 推理注册与构建接线配方
description: 不改 REG_OP/OP_DEF，在 V2 文件实现 InferShape（属性按索引读）+ InferDataType + IMPL_OP 双注册，配 CMake 双目标编译；GEIR 用例输出 shape/dtype 全对。CANN 9.2.0 框架符号不可用时用 IMPL_OP_INFERSHAPE/IMPL_OP 注册自有回调的三件套变体同样验证通过。
tags: [task.implementation, topic.implementation.operator_registration, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---
# GEIR 图模式 V2 推理注册与构建接线配方


## 现象

为一个五输出（两输入、五个输出、一个 Bool 属性）的 GRU Cell 反向融合算子接通 TTK GEIR 图模式：任务约束只允许添加/修改 V2 InferShape、InferDataType、参数校验、IMPL_OP 注册（`op_host/*_infershape.cpp`、`op_graph/*_graph_infer.cpp` 等 V2 文件），**禁止改动 REG_OP/OP_DEF 既有注册语句、Kernel、Tiling、golden**。接线完成后跑 TTK GEIR 验收：全部 44 个正向用例的输出 shape/dtype 派生全部正确——含 hasBias=False 时两路 bias 输出为 (0,) 空张量、GE 正常返回空输出、TTK 按日志 `output[3] empty from GE; expect_shape=[0]; skipping reshape` 跳过 reshape 的边界路径；42/44 精度通过（2 个 bf16 宽范围用例失败与推理接线无关，见 precision 卡）。执行期 slog 可见 `RegisterInferShapeV2Func` / `RegisterInferDataTypeFunc` 初始化；`nm -D` 确认安装产物 libcust_opsproto_rt2.0.so 内含 `ge::InferShape(gert::InferShapeContext*)`、`ge::InferDataType(gert::InferDataTypeContext*)` 与 `gert::OpImplRegisterV2` 相关符号。

后续在 CANN 9.2.0 + Ascend950PR 上为另一个自定义 scatter 类算子（三输入 data/indices/updates + 属性 dim/reduce）接 GEIR 图模式时遇到变体场景：规格文档的 OpDef 写法是直接绑定框架推导符号 `this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);`，但该构建中 `ge::InferShape` / `ge::InferDataType` 两个框架符号不可用（源码注释记载的已文档化偏差），规格文档写法编译不过。改用"REG_OP IR 节点（含 .cc 翻译单元）+ `IMPL_OP_INFERSHAPE(<Op>).InferShape(<自有回调>)` + `IMPL_OP(<Op>).InferDataType(<自有回调>)`"三件套运行时注册后，独立验收通过：route 图日志 `CallInferFunc:Op n_route_target[<Op>] Call InferShapeFuncV2`（V2 路径唯一命中），infer 套件 10/10 PASS（3 InferShape + 7 InferDataType），reject 1/1 PASS（非法 dtype 输入被拒且无 kernel 下发）。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：aclnn 算子仓（ops-registry-invoke 流程）新算子/存量算子补 GEIR 图模式支持（V2 推理注册），保持 REG_OP/OP_DEF 冻结。
- API 路径：`gert::InferShapeContext` / `gert::InferDataTypeContext`；`IMPL_OP(<Op>).InferShape(...).InferDataType(...)`（register/op_impl_registry.h，OpImplRegisterV2 通路）；OpDef `SetInferShape/SetInferDataType` 绑定；`graph/operator_reg.h` 的 REG_OP / TensorType / OP_END_FACTORY_REG；CMake `npu_op_library(cust_optiling TILING)` / `npu_op_library(cust_op_proto GRAPH)` / `npu_op_package_add`。
- 验证过的平台与 CANN 版本：Ascend950（AddConfig("ascend950")）+ cann-9.0.0-beta.2；变体注册路径另在 Ascend950PR + CANN 9.2.0 验证（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度（transfer_scope）：operator_family——多输出、输出 dtype 随首输入、含 Bool 属性控制可选输出维度的算子族（RNN/门控类融合反向最贴）；三件套变体适用于任何需要 V2 推理注册的自定义算子（换算子名即可复用）。
- 路线优先级：recommended_fastpath（GEIR V2 接线的标准做法：双注册 + 按索引读属性 + 双目标编译；框架符号不可用时用自有回调三件套变体）。
- 明确不适用/未验证场景：负向用例通路（本 trace 未跑到负向套件——首轮精度失败即退出，`expect_error` 类 tiling 拒收路径未验证；变体验收含 reject 1/1 但仅覆盖非法 dtype 一类）；动态 shape 套件（-d=true）与 binary 套件未完整执行——未验证场景；单输入单输出简单算子用本配方会冗余但无害；V1（IMPLEMT_COMMON_INFERFUNC）路径未启用；复杂 dtype 推导（输出 dtype 不等于某输入 dtype）需扩展回调，本配方未覆盖。

## 解决配方

三个动作，全部落在 V2 文件与构建接线，不碰 REG_OP/OP_DEF：

1. `op_host/<op>_infershape.cpp`：实现 `ge::InferShape(gert::InferShapeContext*)`——输入存在性校验、rank==2 强校验、storage==(B,5H) 约束（UNKNOWN_DIM 跳过对应维校验以支持动态 shape）、**has_bias 按 attr index 0 读**（`attrs->GetBool(0)`，缺省回 OpDef 默认）、输入/输出 format 仅放行 ND/RESERVED、五输出 shape 派生（bias 维 = hasBias ? 3H : 0；hiddenSize 未知时 3H 同置 UNKNOWN_DIM）。
2. `op_graph/<op>_graph_infer.cpp`：实现 `ge::InferDataType(gert::InferDataTypeContext*)`（dtype 合法域 {DT_FLOAT, DT_FLOAT16, DT_BF16} + 双输入一致校验 + 五输出 SetOutputDataType 随首输入），并在文件尾部 `IMPL_OP(<Op>).InferShape(ge::InferShape).InferDataType(ge::InferDataType)` 完成 V2 注册（与 OpDef SetInferShape/SetInferDataType 并行、注册同一函数指针）。
3. CMake：graph_infer.cpp 同时编入 `cust_optiling_obj`（→ liboptiling，图编译期 tiling 库）与 `cust_proto_obj`（→ libcust_opsproto_rt2.0.so，graph IR 注册库）；`npu_op_library(cust_optiling TILING)` / `npu_op_library(cust_op_proto GRAPH)` / `npu_op_package_add(... LIBRARY cust_optiling cust_opapi cust_op_proto)`。

**CANN 9.2.0 变体（框架符号 ge::InferShape/ge::InferDataType 不可用时）**——shape 与 dtype 推导拆成独立翻译单元、用宏注册自有静态回调：

1. `op_graph/<op>_proto.h`（+ `<op>_proto.cc` 翻译单元）：`REG_OP` 注册 IR 节点——输入/输出的 TensorType dtype 集合、`REQUIRED_ATTR`/`ATTR` 属性。**REG_OP 宏在头文件的 namespace scope 展开只完成声明，必须有一个 .cc 把头文件 include 进 `namespace ge {}` 才会生成目标代码注册进 GE 工厂**。dtype 的列对齐配对约束（哪些输入输出必须同 dtype）由 op_host 的 OpDef `DataType({...})` 列表强制，REG_OP 只声明每端口允许的 dtype 集。
2. `op_host/<op>_infershape.cpp`：`IMPL_OP_INFERSHAPE(<Op>).InferShape(<回调>)` 注册 shape 推导（同形拷贝 + UNKNOWN_RANK 传播，完整代码见下）。
3. `op_graph/<op>_graph_infer.cpp`：`IMPL_OP(<Op>).InferDataType(<回调>)` 注册 dtype 推导（"输出 dtype = 输入 0 的 dtype"）。

判定注册生效的信号（验收命令与期望输出）：

```bash
# route 图日志应出现 V2 推导调用（且无 V1、无 DESCRIPTOR_ERROR）
grep -E "CallInferFunc:Op .*<OpType>" "$RAW_ROUTE"
#   期望：CallInferFunc:Op n_route_target[<OpType>] Call InferShapeFuncV2
# infer 套件应全 PASS
cd tests/geir && ./run.sh infer V2   # 期望 INFER-SUMMARY route=V2 total=10 pass=10 fail=0
```

## 设计决策

- **双注册并行**：OpDef `SetInferShape/SetInferDataType`（经 libcust_opsproto_rt2.0.so 的 OpDef 静态注册）与 `IMPL_OP` V2 注册（RegisterOpImplToRegistry 通路）注册同一函数指针——两条发现通路并行，runtime 无论从哪条找都能拿到回调。注意：slog 里 `RegisterOpImplToRegistry` 行只列内建算子，自定义算子的 V2 注册不产生该行，属正常现象（不代表注册失败；以 `RegisterInferShapeV2Func` 初始化行 + 输出 shape 正确为准）。
- **属性按索引读而非按名读**：InferShape `GetBool(0)` / TilingFunc `GetAttrPointer<bool>(0)`——IR 层属性名（snake_case）与 aclnn ABI 参数名（camelCase）是不同层级，按索引读避免命名层传导（前序会话曾误改 IR 属性名后回退；本 trace 的 config 键 camelCase/snake_case 并存观测进一步说明按名读的脆弱性，见 runtime 卡）。
- **五输出 dtype 一律随首输入**：契约三 dtype 组合内逐 IO 同 dtype；InferDataType 做合法域 + 双输入一致校验后统一 `SetOutputDataType(i, gradHyDtype)`。
- **UNKNOWN_DIM 传播**：hiddenSize==UNKNOWN_DIM 时 3H 也置 UNKNOWN_DIM，静态/动态两套 shape 模式共用一份推导。
- **空输出 (0,) 走通**：hasBias=False 时 bias 输出维置 0，GE 返回空张量、TTK 跳过 reshape——推理侧无须特殊分支。
- **9.2.0 变体为何不用 OpDef 内框架符号绑定**（规格文档原始写法 `SetInferShape(ge::InferShape)`）：该 CANN 构建中 `ge::InferShape`/`ge::InferDataType` 框架符号不可用（源码注释记载的偏差，该轮验收未重新编译验证符号缺失本身），运行时注册是标准替代；改用宏注册后 V2 路由实测命中。
- **变体为何 shape 与 dtype 回调拆成两个翻译单元**（op_host infershape.cpp / op_graph graph_infer.cpp）：与 CANN 运行时注册机制的分类一致（IMPL_OP_INFERSHAPE 与 IMPL_OP 分属不同注册面），也便于按文件对应验收清单。
- **变体为何 V1 legacy 示例保留但不链接**：该算子冻结 route 为 V2（InferShapeFuncV2）；V1（`IMPLEMT_COMMON_INFERFUNC` + `COMMON_INFER_FUNC_REG`）示例单独放 `legacy_v1_infer_example.cpp` 只做 compile-check，避免同一名双注册冲突，若未来冻结 route 变为 V1 可适配后加入编译。

## 系统信息

- 实测设备：NPU:0 Ascend950PR；CANN cann-9.0.0-beta.2；TTK 3.0.0。变体验证：Ascend950PR（Ascend950PR_9579）+ CANN 9.2.0，GE `ge::Session` 图模式（AddGraph/RunGraph），fresh staging 隔离安装包验证（GEIR_VENDOR_DIR）。
- 验证代码状态：op 源码全部与 HEAD 一致（git status 仅 tests/geir 的 log/CSV 结果文件有未暂存变更；`git diff HEAD` 对 proto.h/def.cpp 无输出）——代码绑定提交 7f4b6507b41bd226a4897efb4280168f3fc7ff58。
- staging 安装布局：`<staging>/vendors/<Op>/{op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so, op_impl/ai_core/tbe/{kernel/ascend950/<op>/, op_tiling/}, op_api/, framework/, scripts/, version.info}`；测试通过 `ASCEND_CUSTOM_OPP_PATH="$STAGING"` 隔离加载。
- 相关文件清单：`op_host/<op>_infershape.cpp`（InferShape V2）、`op_graph/<op>_graph_infer.cpp`（InferDataType V2 + IMPL_OP）、`op_host/<op>_def.cpp`（OpDef 绑定，冻结）、`op_graph/<op>_proto.h`（REG_OP，冻结）、`CMakeLists.txt`（对象库与打包接线）、`op_host/arch35/<op>_tiling_*.cpp`（TilingFunc，attr index 读取处）。变体追加：`op_graph/<op>_proto.cc`（REG_OP 翻译单元）、`legacy_v1_infer_example.cpp`（compile-check 用）；安装产物 `libcust_opsproto_rt2.0.so`（op_proto）+ `libcust_opmaster_rt2.0.so`（op_tiling/op_host），从 staging 注册（AddSoToRegistry 日志可见两个 SO 均来自 staging 路径）。

## 完整代码

InferShape（V2，`op_host/<op>_infershape.cpp` 全函数 verbatim）：

```cpp
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
```

InferDataType + IMPL_OP V2 注册（`op_graph/<op>_graph_infer.cpp` 全文 verbatim）：

```cpp
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

OpDef 绑定（`op_host/<op>_def.cpp` 构造函数内，冻结不动，接线参照）：

```cpp
        this->Attr("has_bias").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend950");
```

CMake 关键接线（`CMakeLists.txt` verbatim 摘录）：

```cmake
# cust_optiling_obj: Tiling object library.
## Contains host tiling sources + graph inference (used at graph compile time).
add_library(cust_optiling_obj OBJECT ${host_tiling_srcs} op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp)
target_compile_definitions(cust_optiling_obj PRIVATE
    BUILD_SOC_VERSION=Ascend950
    OP_TILING_LIB
    _GLIBCXX_USE_CXX11_ABI=0
)

## cust_proto_obj: Graph proto object library.
## Contains ALL host sources: tiling, infershape, graph inference, proto.
## Used by the graph engine for IR-level operator definition.
set(proto_srcs ${host_tiling_srcs} ${host_graph_srcs})
add_library(cust_proto_obj OBJECT ${proto_srcs})

## npu_op_library: Declares a library for the operator package.
##   TILING type: Used for tiling computation (graph compilation phase).
npu_op_library(cust_optiling TILING)
target_sources(cust_optiling PUBLIC $<TARGET_OBJECTS:cust_optiling_obj>)

## npu_op_library GRAPH type: Used for graph-engine proto registration.
npu_op_library(cust_op_proto GRAPH)
target_sources(cust_op_proto PUBLIC $<TARGET_OBJECTS:cust_proto_obj>)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)
```

CANN 9.2.0 变体注册（自有回调三件套，`op_graph/bench_harness_scatter_proto.h` / `op_graph/bench_harness_scatter_proto.cc` / `op_host/bench_harness_scatter_infershape.cpp` / `op_graph/bench_harness_scatter_graph_infer.cpp`，verbatim，替换算子名与 dtype 集合即可复用）：

```cpp
// op_graph/bench_harness_scatter_proto.h —— REG_OP IR 节点
#ifndef BENCH_HARNESS_SCATTER_PROTO_H
#define BENCH_HARNESS_SCATTER_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

REG_OP(BenchHarnessScatter)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(dim, Int)
    .ATTR(reduce, String, "update")
    .OP_END_FACTORY_REG(BenchHarnessScatter)
}

#endif
```

```cpp
// op_graph/bench_harness_scatter_proto.cc —— REG_OP 翻译单元（必须有，否则注册不生效）
// Includes the REG_OP(BenchHarnessScatter) declaration.  The REG_OP macro expansion at
// namespace scope registers the operator type with the GE operator factory.
#include "bench_harness_scatter_proto.h"

// The namespace must wrap the include because REG_OP is inside `namespace ge`.
// The actual registration code emitted by the macro lives in bench_harness_scatter_proto.h.
namespace ge {
}
```

```cpp
// op_graph/bench_harness_scatter_graph_infer.cpp —— dtype 推导注册（完整注册单元）
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferDataTypeForBenchHarnessScatter(gert::InferDataTypeContext *context)
{
    // Output dtype = input 0 (data) dtype.
    const ge::DataType dataDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, dataDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BenchHarnessScatter).InferDataType(InferDataTypeForBenchHarnessScatter);

} // namespace ops
```

```cpp
// op_host/bench_harness_scatter_infershape.cpp —— shape 推导注册（含 UNKNOWN_RANK 传播）
#include "register/op_impl_registry.h"

#include <cstdint>
#include <vector>

#include "graph/types.h"

using namespace ge;

namespace ops {

namespace {

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 &&
           shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

} // namespace

static ge::graphStatus InferShapeForBenchHarnessScatter(gert::InferShapeContext* context)
{
    const gert::Shape* dataShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (dataShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsUnknownRank(dataShape)) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    const size_t rank = dataShape->GetDimNum();
    yShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        yShape->SetDim(i, dataShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BenchHarnessScatter).InferShape(InferShapeForBenchHarnessScatter);

} // namespace ops
```

## 预防

- 接 GEIR 前先冻结 REG_OP/OP_DEF（git diff 确认 `.ATTR/.INPUT/.OUTPUT` 与 `this->Attr/Input/Output` 语句零变更），再动 V2 文件——属性名误改是本类任务的高频事故（IR 名与 ABI 名两层易混）。
- InferShape/TilingFunc 一律按 attr index 读属性，不按名读。
- 验收顺序：先 projection/frozen 校验，再跑正向 static 套件看 shape/dtype，最后看精度——shape/dtype 全对而精度失败时，问题不在 V2 接线（见 precision 卡判别法）。
- 写 GE 图模式推导前先确认目标 CANN 版本是否提供 `ge::InferShape`/`ge::InferDataType` 框架符号；不可用则直接采用自有回调三件套变体（IMPL_OP_INFERSHAPE + IMPL_OP），不要在 OpDef 构造函数里绑定不存在的符号。
- REG_OP 只在头文件展开、没有 .cc 翻译单元时注册不会生效——检查链接产物里是否真的有对应目标文件。

## 验证信号

以下信号来自一次 Ascend950 上的 GRU Cell 反向算子 GEIR 图模式验收会话（对已提交的 V2 接线做全量用例验证），均为命令实测输出；末三条来自 CANN 9.2.0 + Ascend950PR 上 scatter 类算子三件套变体的独立验收会话。

- projection-check：`PASS projection-check positive=44 negative=49 not_applicable=0`；冻结校验：`PASS geir-frozen files=5`（来源平台：950；证据等级：tool_observed）。
- GEIR 正向 static 套件 44 用例输出 shape/dtype 全部正确（例：((10, 1809), (10, 1809), (10, 603), (0,), (0,))、全 bfloat16；含 hasBias=False 的 (0,) 空输出路径，TTK 日志 `output[3] empty from GE; expect_shape=[0]; skipping reshape`）（来源平台：950；证据等级：tool_observed）。
- 首轮 42/44 精度通过（2 个失败为 bf16 宽范围用例，其 shape/dtype 亦正确——失败与推理接线无关）（来源平台：950；证据等级：tool_observed）。
- 执行期 slog 观测 `RegisterInferShapeV2Func` / `RegisterInferDataTypeFunc` 初始化加载（来源平台：950；证据等级：tool_observed）。
- `nm -D` 确认安装产物 libcust_opsproto_rt2.0.so 含 `ge::InferShape(gert::InferShapeContext*)`、`ge::InferDataType(gert::InferDataTypeContext*)` 及 `gert::OpImplRegisterV2` 相关符号（来源平台：950；证据等级：tool_observed）。
- 变体验收（CANN 9.2.0）：推导路由命中——route 日志 `CallInferFunc:Op n_route_target[BenchHarnessScatter] Call InferShapeFuncV2`（V2 唯一路径，无 V1、无 DESCRIPTOR_ERROR）（来源平台：950；证据等级：tool_observed）。
- 变体验收（CANN 9.2.0）：InferShape 3/3 PASS（known-shape-2d（[4,3],[2,3],[2,3]→[4,3]）、known-shape-1d（[8],[4],[4]→[8]）、unknown-rank（[-2]→[-2]））；InferDataType 7/7 PASS（f16/f32/bf16/i32/i64（data/updates/y）× i32/i64（indices）组合全部推出期望 dtype），`INFER-SUMMARY route=V2 total=10 pass=10 fail=0`（来源平台：950；证据等级：tool_observed）。
- 变体验收（CANN 9.2.0）：Reject 1/1 PASS——indices 传 DT_FLOAT16 的非法输入在 RunGraph 被拒（`status=4294967295`）且无 kernel 下发（`GRAPH-REJECT ... value=PASS`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 负向用例通路（49 条）与动态 shape（-d=true）、binary 套件未在本 trace 完整执行——未验证场景；`expect_error` 类 tiling 拒收的判读路径未验证。
- 代码状态绑定提交 7f4b6507b41bd226a4897efb4280168f3fc7ff58（工作树对 op 源码零变更，git 实测）；其它提交状态未验证。
- Ascend950 之外的 SOC（BUILD_SOC_VERSION/AddConfig 其它值）未验证，接线模式推断可迁移（未验证）。
- 变体路径的"`ge::InferShape`/`ge::InferDataType` 符号不可用"来自源码注释记载（该轮验收未重新编译验证符号缺失本身），其他 CANN 9.2.0 安装需自行确认后再选路线。
- 变体的 InferDataType 回调只做"输出=输入 0"的传播；输出 dtype 需要推导（如混合 dtype 提升）的算子要扩展回调逻辑，未验证；输出 shape 与输入不同形（如 reduce 后降维）的算子需改写 InferShape 回调，变体只验证了同形拷贝 + UNKNOWN_RANK 传播。

## 坏实践

- 为接 GEIR 直接改 REG_OP/OP_DEF 的 `.ATTR`/`Input/Output` 注册语句（含"顺手统一"属性名大小写）——破坏 IR 层契约，前序会话已发生过并回退（转写记载，本 trace 未复现）。
- 只做 OpDef SetInferShape 绑定而不做 IMPL_OP V2 注册（或反之）——依赖单一发现通路，runtime 从另一通路找不到回调时行为不确定。
- 把 graph_infer.cpp 只编进一个库——tiling 库（图编译期）与 proto 库（graph IR）都需要它。
- 在 OpDef 构造函数里绑定当前构建不存在的框架符号（`ge::InferShape`/`ge::InferDataType`）——编译不过；应改用自有回调宏注册变体。
- REG_OP 宏只在头文件展开、不提供 .cc 翻译单元——注册不会生效，且无编译报错，只能靠路由日志/链接产物发现。

## 误诊

- "slog 中看不到自定义算子的 `RegisterOpImplToRegistry` 行 = V2 注册失败"——误读：该日志行只覆盖内建算子；自定义算子以 `RegisterInferShapeV2Func` 初始化 + 输出 shape 正确为准（本 trace 一度据此怀疑注册未生效，经符号与运行结果排除）。

## 未决事项

1. 负向用例（49 条）在 GEIR 执行阶段的真实拒绝路径（reject node 不生成目标 Kernel）未验证——首轮精度失败即退出，未跑到 negative 套件。
2. 动态 shape 套件（-d=true）下 UNKNOWN_DIM 传播路径未实测。
3. V2 实现的编写/迭代过程发生在前序会话（本 trace 仅恢复终版并验证），版本演化细节见 git 历史而非本卡。
