---
type: "Runbook"
title: GE 图模式 1343242270：缺 InferDataType 注册
description: "自定义算子 op_host 只注册 InferShape 时，aclnn 直调正常但 GE 图模式跑图失败（ErrorNo 1343242270，CallInferFuncV2 Failed to infer data_type，输出 origin_dtype=DT_UNDEFINED）；补 InferDataType 并在 IMPL_OP_INFERSHAPE 链式注册后图模式通过。复现增量：逐元素算子可用内置规则 InferOutDataTypeSameWithFirstInput() 一行注册替代自定义函数（与自定义 InferDataType 同时注册时最后注册的生效）；plog 中 infer_shape/tiling_parse 等已注册而独缺 infer_datatype 的差分可排除 .so 未加载疑云。复现增量：十一条独立 trace（GRU/MoE/depthwise conv/scatter/resize/gather/argmax/sigmoid 等，CANN 9.1.0/9.2.0）批量复现；静默失败三取错入口（GEGetErrorMsgV2/plog grep [ERROR]/ge_check_op.json——后者 error_type=infer_shape_error 字面误导）；IGNORE_INFER_ERROR=1 旁路-修复-移除完整纪律；nm 符号级前后对照；OP_ADD 展开注册 nullptr 机制；autogen op_proto.h TensorType::ALL 无 T-推导依据且仅修 proto 不充分（A/B 实证）。"
tags: [task.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# GE 图模式 1343242270：缺 InferDataType 注册

## 现象

给一个自定义 RoPE 类算子（aclnn 双段式 + AscendC kernel，Ascend950PR）补交付物时，开发 GE IR 图模式调用示例：示例程序用安装包 `op_proto.h` 里的 `REG_OP` 定义算子节点、构造双输入双输出计算图，经 xir session 跑图。此时 aclnn 直调（eager）示例已经 PASS；图模式示例编译成功，但跑图直接失败：

```
2026-07-15 04:09:39,000 - INFO - [XIR]: Add graph success
2026-07-15 04:09:39,000 - INFO - [XIR]: Start to run graph
2026-07-15 04:09:39,000 - ERROR - [XIR]: Run graph failed
```

翻 GE 的 plog 才看到真正原因——图编译的 InferShapePass 阶段 dtype 推导失败：

```
[WARNING] GE(...):[op_desc_utils_ex.cc:71] CallInferFuncV2:Assert failed: [Call][InferFuncV2]Failed to infer data_type of node rope1[BenchNoHarnessApplyRotaryPosEmb].
[ERROR] GE(...):[infershape_pass.cc:122] Infer: ErrorNo: 1343242270(Prepare Graph infershape failed)[Call][InferShapeAndType] for node:rope1(BenchNoHarnessApplyRotaryPosEmb) failed
```

plog 里节点张量信息显示：四个输入 dtype 正常（`dtype:DT_FLOAT, origin_dtype:DT_FLOAT`），两个输出 shape 已推出（`shape:[2,4,2,8]`）但 `dtype:DT_FLOAT, origin_dtype:DT_UNDEFINED`（origin_shape 为空）。当时算子的 `op_host` infershape 实现只注册了 InferShape（把 query/key 的 shape 拷给两个输出），没有 InferDataType；`def.cpp` 的 OUTPUT 用 `TensorType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})` 声明了候选 dtype（安装包 ops-info.json 中 output0 的 dtype 声明 "float32,float16,bfloat16" 也正确），但这对 GE 的 InferFuncV2 路径不够。

同一根因在另一个 GroupNorm 类自定义算子上独立复现：该算子 ACLNN 两段式 C++ ST 全量 716/716 通过（修复前），补 GE 图模式示例时同样因 op_host infershape 只注册 InferShape 而跑不通，补 InferDataType 回调并重编重装后 GE 图模式示例与 ACLNN 示例均恢复 PASS。

## 适用场景

- 工程形态：自定义算子（op_host + op_graph + aclnn）增加 GE IR 图模式调用示例或接入图执行；aclnn 直调与图模式行为差异排查。
- API 路径：`IMPL_OP_INFERSHAPE(OpName).InferShape(...)` 注册、`gert::InferDataTypeContext`（`exe_graph/runtime/infer_datatype_context.h`）、GE InferShapePass/InferFuncV2、`SetOutputDataType`/`GetInputDataType`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/dav_3510），CANN 9.1.0 与 9.2.0（多条独立 trace 复现，见验证信号）；与 frontmatter `platforms` 一致。
- 适用粒度：platform_wide（GE 图编译的 dtype 推导机制与芯片型号无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：单算子 aclnn 直调路径（实测不需要 InferDataType 也能正常执行）；输出 dtype 与输入无关（固定 dtype 输出）的算子注册方式未验证；其他 CANN 版本的接口形态未验证。

## 根因

GE 图编译的 InferShapePass 走 InferFuncV2 路径，要求算子同时提供 shape 与 dtype 的推导。只注册 InferShape 时，输出 shape 能推出来（日志中输出 shape 已填充），但输出 `origin_dtype` 停留在 DT_UNDEFINED，InferFuncV2 断言失败 → ErrorNo 1343242270 → 图编译立即终止，对外只表现为 "Run graph failed"。`def.cpp` OUTPUT 的 TensorType 候选列表与 ops-info.json 的 dtype 声明描述的是"允许的 dtype 集合"，不提供"输出 dtype 等于哪个输入 dtype"的推导规则，不能替代 InferDataType 注册。aclnn 直调不走 GE 图编译的 InferFuncV2，所以同算子 eager 路径完全正常——这也是该缺陷容易漏到交付末期才暴露的原因。

## 解决配方

修复前（`op_host/bench_no_harness_apply_rotary_pos_emb_infershape.cpp` 全文，只有 InferShape）：

```cpp
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4BenchNoHarnessApplyRotaryPosEmb(gert::InferShapeContext* context)
{
    // queryOut shape = query shape
    const gert::Shape* queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape* queryOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryOutShape);
    *queryOutShape = *queryShape;

    // keyOut shape = key shape
    const gert::Shape* keyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape* keyOutShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyOutShape);
    *keyOutShape = *keyShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BenchNoHarnessApplyRotaryPosEmb).InferShape(InferShape4BenchNoHarnessApplyRotaryPosEmb);

} // namespace ops
```

修复后（新增 infer_datatype_context 头、InferDataType 函数，并链式注册；全文 verbatim）：

```cpp
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4BenchNoHarnessApplyRotaryPosEmb(gert::InferShapeContext* context)
{
    // queryOut shape = query shape
    const gert::Shape* queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape* queryOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryOutShape);
    *queryOutShape = *queryShape;

    // keyOut shape = key shape
    const gert::Shape* keyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape* keyOutShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyOutShape);
    *keyOutShape = *keyShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BenchNoHarnessApplyRotaryPosEmb(gert::InferDataTypeContext* context)
{
    // queryOut dtype = query dtype
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // keyOut dtype = key dtype
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BenchNoHarnessApplyRotaryPosEmb)
    .InferShape(InferShape4BenchNoHarnessApplyRotaryPosEmb)
    .InferDataType(InferDataType4BenchNoHarnessApplyRotaryPosEmb);

} // namespace ops
```

随后 `bash build.sh --soc=ascend950 --make_clean` 重编并重装算子包，图模式示例跑通。GroupNorm 算子侧的等价修复（输出 dtype = 输入 dtype 的单输出直传，`IMPL_OP_INFERSHAPE(<Op>).InferShape(...).InferDataType(...)` 链式注册）同样验证通过。

### 增量（独立 trace：内置规则一行注册 + plog 已注册项差分）

**内置规则路线**：对"输出 dtype = 第一个输入 dtype"的逐元素算子，`register/op_impl_registry.h` 提供内置规则 `InferOutDataTypeSameWithFirstInput()`，一行注册即可，无需写自定义 `InferDataType` 函数（头文件注释明确：注册此规则可不用再注册自定义推导规则；若同时注册 `InferDataType` 与该规则，使能最后注册的）：

```cpp
// 修复前（只有 shape 推导）
IMPL_OP_INFERSHAPE(<Op>).InferShape(InferShape4<Op>);

// 修复后（注册链追加输出 dtype 推导规则，逐元素算子用内置规则即可）
IMPL_OP_INFERSHAPE(<Op>)
    .InferShape(InferShape4<Op>)
    .InferOutDataTypeSameWithFirstInput();
```

改动在 op_host，必须重打包安装算子包才生效（重装后删掉示例的 build-geir 目录重跑图模式示例）。

**plog 已注册项差分**：同一份 plog 里 `infer_shape func has been registered`、`tiling_parse func has been registered`、`compile_info_creator/compile_info_deleter/tiling/max_tiling_data_size has been registered` 等 WARNING 行证明 op_impl 库已被加载且其余注册项可见、缺的只是 infer_datatype——不需要怀疑 op_proto `.so` 没加载（报错文案 "Please confirm whether the op_proto shared library (.so) has been loaded successfully" 的第一种可能性可据此排除）。该形态的报错原文（shape_inference.cc:607）：`InferDtypeByRegisteredFuncOrRule: ErrorNo: 4294967295(failed)Cannot find Node <node>[<Op>] custom infer_datatype func. ...`，级联 `InferDataTypeOnCompile ... ret[-1]` → `Prepare Graph infershape failed (1343242270)` 直到 `Run graph failed`。

## 设计决策

- InferDataType 直接取对应输入的 dtype（queryOut ← query，keyOut ← key），与本算子"输出与输入同 dtype"的语义一致；TensorType 三候选只约束允许集合，具体值由 InferDataType 按 input 推导。
- 参考模板 add_example 的 infershape 同样只注册 InferShape（其图模式示例在本轨迹中未验证能否跑通）；双输出 + 多 dtype 候选场景实测必须显式注册 InferDataType，不依赖模板默认。
- 当前方案路线档位：recommended_fastpath（图模式交付必需，改动一行注册 + 一个直传函数）。

## 系统信息

- 环境：Ascend950PR（arch35/dav_3510），CANN 9.1.0；算子包安装于 `$ASCEND_HOME_PATH/opp/vendors/<op>_custom`。
- 相关文件：`op_host/bench_no_harness_apply_rotary_pos_emb_infershape.cpp`（本修复对象）、`op_host/..._def.cpp`（OUTPUT TensorType 声明，未改动）、`examples/test_geir_*.cpp` + `CMakeLists_geir.txt`（图模式示例）、安装包 `ops-info.json`（用于核对输出 dtype 声明）、GE plog（`$ASCEND_HOME_PATH/log/plog/` 或环境对应 plog 目录）。
- 复现方式：跑图模式示例失败后，在 plog 中 grep `CallInferFuncV2\|1343242270\|DT_UNDEFINED` 定位。

## 预防

- 自定义算子交付验收前至少跑一次 GE IR 图模式调用（不只 aclnn 直调），可在交付前暴露 InferDataType 缺失。
- 写 op_host infershape 时默认成对实现 InferShape + InferDataType——"输出 dtype 随输入"的算子只是逐输出一行 `SetOutputDataType(i, GetInputDataType(j))`。

## 验证信号

来源说明：以下信号来自两个独立算子（RoPE 类、GroupNorm 类）的 GE 图模式示例开发会话（均为 Ascend950PR、CANN 9.1.0）修复前后同一示例的对比。

- RoPE 算子修复前：图模式 `Run graph failed`，plog ErrorNo 1343242270、`CallInferFuncV2: Failed to infer data_type of node`、输出 `origin_dtype:DT_UNDEFINED`；同算子 aclnn 直调示例 PASS（不受影响）（来源平台：950；证据等级：tool_observed）。
- RoPE 算子修复后（重编重装）：图模式示例 `Run graph success`，queryOut/keyOut verification 均 PASS；aclnn 直达示例回归 PASS（来源平台：950；证据等级：tool_observed）。
- GroupNorm 算子修复前：ACLNN 两段式 C++ ST 全量 716/716 通过，GE 图模式示例跑不通（缺 InferDataType 的结论来自该示例子任务排查）（来源平台：950；证据等级：tool_observed，ST 结果为程序实跑输出）。
- GroupNorm 算子修复后：GE 图模式示例 PASS（MERE=1.06e-07、MARE=3.73e-06，fp32，192 elems）；同一算子包上 ACLNN 两段式示例同参数 PASS（MERE=1.06e-07）（来源平台：950；证据等级：tool_observed）。
- GroupNorm 算子修复后 `op_host/<op>_infershape.cpp` 同时注册 `InferShape` 与 `InferDataType`（来源平台：950；证据等级：artifact_verified——git diff 工具输出完整恢复修复后文件内容）。

增量信号（独立 trace，Elementwise 指数算子 GE IR 示例联调，Ascend950PR + CANN 9.1.0）：

- 修复前：XIR `Run graph failed`（应用层仅一行，无算子级信息）；plog `Cannot find Node <node>[<Op>] custom infer_datatype func`（shape_inference.cc:607，ErrorNo 4294967295）级联 1343242270；同 plog 显示 infer_shape/tiling_parse/compile_info_creator/tiling 等均已注册（排除 op_proto 未加载）（来源平台：950；证据等级：tool_observed）。
- 修复后（注册链追加 `.InferOutDataTypeSameWithFirstInput()` 并重打包安装）：`[XIR]: Session run ir compute graph success`、`[XIR]: Precision is ok`，输入输出 dtype/shape 正确回读，示例执行结果 PASS（来源平台：950；证据等级：tool_observed）。
- 同环境 aclnn（eager）示例修复前后均 PASS，证明改动未影响 eager 路径（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对"输出 dtype 可从输入推导"的算子；输出 dtype 固定或需属性参与的注册方式未验证。
- InferFuncV2 / InferDataType 为 CANN 9.1.0 的接口形态，其他版本接口可能变化（未验证）。
- 双输出与单输出是否影响触发未隔离验证（本例为双输出算子）。
- 其他 CANN 版本接口形态未验证（推断可迁移）。
- （独立 trace 补充）`InferOutDataTypeSameWithFirstInput` 只覆盖"输出 dtype = 第一个输入 dtype"的算子；多输出各随不同输入、dtype 提升或广播类算子需要自定义 `InferDataType` 函数（本增量未覆盖）。GE 图编译对"IR 上标 T-derivation 规则"替代路径的写法未验证；动态 shape（-1 维）下该内置规则是否充分未验证（本例为静态 shape）。

**未决项**：

- add_example 模板只注册 InferShape 时其图模式示例能否跑通（单输出是否豁免）未验证。
- InferFuncV2 对 origin_dtype/origin_shape 的完整语义（为何输出 dtype 字段已填 DT_FLOAT 仍判失败）未深入，本例只修到图模式跑通。
## 坏实践

- 只跑 aclnn 直调就认为算子交付完成——图执行路径的 dtype 推导要求不同，缺失会静默漏到示例/图接入阶段才爆。
- 看到 "Run graph failed" 就去查构图代码——先翻 GE plog 拿 ErrorNo 与 CallInferFuncV2 断言信息，能直接指向 infershape 注册问题。
- 试图用 def.cpp OUTPUT 的 TensorType 候选列表或 ops-info.json 的 dtype 声明"代替"InferDataType 注册——它们只声明允许集合，不提供推导规则。

## 误诊

- 最初怀疑算子实现未注册或注册失败，plog 显示 infershape/tiling 注册均已正常加载，排除。
- 注意到安装包 `op_proto.h` 的属性类型（Int/REQUIRED_ATTR）与源码（Int64 带默认值）存在生成差异，一度怀疑 proto 问题；核对 ops-info.json 输出 dtype 声明正确后，才转向 InferDataType 缺失这一真因。

## 增量（2026-09-17，并行策展批：十一条独立 trace 批量复现与机制/诊断增补）

十一条独立 trace（GRU、MoE token 重排、depthwise conv、scatter、resize 双线性、gather、argmax、sigmoid、gcd、apply_adam_w、foreach addcdiv 标量等自定义算子，Ascend950PR + CANN 9.1.0/9.2.0）以同一失败签名（`InferDtypeByRegisteredFuncOrRule ... Cannot find Node ... custom infer_datatype func` → `InferDataTypeOnCompile ... ret[-1]` → `Prepare Graph infershape failed (1343242270)` → `Run graph failed`）批量复现本卡根因，并补齐以下机制与诊断面。

### 1. 静默失败的三个取错入口

GE 图模式失败时 stdout 常只有一行 `Run graph failed`（退出码 255），部分形态 plog 也无 ERROR 细节（PreRun 3~4ms 内即销毁 session，`ASCEND_GLOBAL_LOG_LEVEL=1` 全量日志仍只有十几行）。按序取错：

```cpp
    // 入口一：RunGraph 失败分支补 GEGetErrorMsgV2（模板默认只在成功路径打印）
    ret = session->RunGraph(graphId, inputTensors, outputTensors);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed, ret=%d\n", GetTime().c_str(), static_cast<int>(ret));
        ge::AscendString errMsg = ge::GEGetErrorMsgV2();
        ge::AscendString warnMsg = ge::GEGetWarningMsgV2();
        printf("GE error message: %s\n", errMsg.GetString());
        printf("GE warning message: %s\n", warnMsg.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }
```

- 入口二：`grep "\[ERROR\]" ~/ascend/log/debug/plog/plog-<pid>_*.log | head -20` 取首因——粗 grep（error|fail|not found）会命中大量收尾期 CloseHandle/Finalize 与其他 vendor 的 op_master/op_tiling WARNING 噪音，必须用 `"[ERROR]"` 精确过滤。
- 入口三：构建目录下自动生成的 `ge_check_op.json` 结构化给出 `error_type: "infer_shape_error"` + `reason: "InferShapeFailed!"`——**字面上像 shape 推导问题，实际失败点是同 pass 内的 InferDataType**（plog 才给出准确指向）；单看 json 会误修 InferShape 函数。

### 2. IGNORE_INFER_ERROR=1 的完整纪律（旁路 → 修复 → 移除）

错误信息自带官方逃生门 `You can ignore this validation by exporting IGNORE_INFER_ERROR=1 if necessary, but it is highly recommended to fix this problem`。实测纪律：

1. **中间态旁路验证**：`export IGNORE_INFER_ERROR=1` 重跑——图执行成功且精度正常即证明"除 dtype 推导外图执行链路本身正常"，锁定校验层阻塞（EZ9999 文本仍打印但降级为非致命，不要据其误判失败）。
2. **正式修复**：补 InferDataType 注册，重编重装算子包。
3. **移除旁路**：从 run.sh/交付物中删除 `IGNORE_INFER_ERROR=1` 再复跑——该变量全局生效，会跳过整图的推导校验，把单算子缺陷放大成全局放松；固化进交付物会误导后续使用者并掩盖其他算子的真实推导缺陷。

边界：该旁路仅当调用方显式给定全部输入输出 shape/dtype（示例场景）时可安全使用；图中存在依赖推导结果的节点（未显式指定 desc、级联推导）时禁用——跳过会导致下游拿到未推导的 desc（未验证）。

### 3. 符号级前后对照（排除"so 未加载"）

`nm -C ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so | grep -i "InferDataType\|InferOut\|SameWithFirst"`：

- 修复前：可见 `ops::op_impl_register_infershape_<Op>` 与 `ops::InferShape4<Op>` 符号（注册变量与形状推导函数都在库里），但**无任何 InferDataType 符号**——库已加载、是注册内容缺一项，不是库没加载。
- 修复后（`.InferOutDataTypeSameWithFirstInput()` 路线）：出现 `gert::(anonymous namespace)::InferOutDataTypeSameWithFirstInputFunc(gert::InferDataTypeContext*)` 与 `gert::OpImplRegisterV2::InferOutDataTypeSameWithFirstInput()` 符号。

### 4. 机制增补：OP_ADD 展开注册 nullptr

新式 OpDef（`OP_ADD`）宏展开含 `impl.InferShape(op.GetInferShape()).InferShapeRange(op.GetInferShapeRange()).InferDataType(op.GetInferDataType())`——未 `SetInferDataType` 时 `GetInferDataType()` 返回 nullptr，注册的是空指针；vendor 包内 `libcust_opsproto_rt2.0.so` 的符号表只有 `OpDef::GetInferShape/GetInferShapeRange` 未定义引用（U）、构建产物 `build/autogen/op_proto.cc` 是空壳——老式 proto 注册路径不会代劳 dtype 推导。正解仍是 `IMPL_OP_INFERSHAPE(<Op>).InferShape(...).InferDataType(...)` 链式补注册，无需改 def.cpp。

### 5. autogen op_proto.h 的 TensorType::ALL 边界与 A/B 实验

新式 OpDef 工程的构建系统从 `_def.cpp` 自动生成的 `build/autogen/op_proto.h` 把 INPUT/OUTPUT 都写成 `ge::TensorType::ALL()`（dtype 映射没有被翻译成 REG_OP 的显式类型列表）、`keepdim` 被生成为 `REQUIRED_ATTR`——GE 图编译既无法从 IR 规则推导输出 dtype（输出固定 int64、与输入 dtype 不同，本来也无法 1:1 推导），又找不到 `infer_datatype` 注册项。A/B 实验证伪"只修 proto"：仅把 autogen `op_proto.h` 的 `TensorType::ALL()` 换成显式类型列表并重装，geir 仍报同一 `infer_datatype` 错误；补 InferDataType 注册并重建拷贝 `libcust_opmaster_rt2.0.so`/`liboptiling.so` 后才通过——**InferDataType 注册是必要修复，proto 显式类型只是配套修正**。手改 autogen 头在全量构建后会被重新生成覆盖（本轨迹用增量 make + 手动拷贝规避，长效方案见遗留问题）。

### 6. 修复形态与重装要求

- 链式自定义函数：输出 dtype 透传/跟随输入（`SetOutputDataType(i, GetInputDataType(j))`）；多 dtype 输出算子逐输出给映射（如 MoE 重排：输出 0 跟随输入 0、输出 1 固定 DT_FLOAT、输出 2 固定 DT_INT32、输出 3 跟随输入 1）。
- 内置规则：输出 dtype == 首输入 dtype 的算子用 `.InferOutDataTypeSameWithFirstInput()` 一行注册。
- 固定输出 dtype：输出与输入 dtype 解耦的算子（如 argmax 索引恒 int64）在函数内 `SetOutputDataType(0, ge::DT_INT64)`。
- 修复在 op_host，**必须重编 + 重装算子包**（注册信息编译进 `libcust_opsproto_rt2.0.so`/`libcust_opmaster_rt2.0.so`），只重跑示例不生效。

### 7. 验证信号（复现增量，十一条 trace 汇总，均为修复前后同一示例对比）

- 修复前（多 trace 同型）：GE IR 示例 `Run graph failed`（stdout 一行、退出码 255，部分 trace plog 亦无细节）；取错后 plog 报 `InferDtypeByRegisteredFuncOrRule ... Cannot find Node <node>[<Op>] custom infer_datatype func`，级联 `InferDataTypeOnCompile ... ret[-1]`、`Prepare Graph infershape failed (ErrorNo 1343242270)`（GRU/depthwise conv/scatter/resize/gather/argmax/sigmoid/gcd/apply_adam_w/foreach addcdiv，CANN 9.1.0/9.2.0）（来源平台：950；证据等级：tool_observed）。
- 静默失败取错：加 `GEGetErrorMsgV2` 后取得 EZ9999 全文与 TraceBack（含 IGNORE_INFER_ERROR=1 官方提示）——stdout 与 plog 均无细节的形态下这是唯一入口（来源平台：950；证据等级：tool_observed）。
- ge_check_op.json：`error_type: "infer_shape_error"` + `reason: "InferShapeFailed!"`，输入 shape/dtype 均正确——实际失败点为同 pass 的 InferDataType（来源平台：950；证据等级：tool_observed）。
- 注册表差分：DEBUG plog `RegisterOpImplToRegistry ... register OP_IMPL : [InferShape]` 与 `[Tiling][TilingParse]` 而无 `[InferDataType]` 条目（来源平台：950；证据等级：tool_observed）。
- 符号探针：修复前 `nm -C libcust_opmaster_rt2.0.so` 有注册变量与 InferShape 符号、无任何 InferDataType 符号；修复后出现 `InferOutDataTypeSameWithFirstInputFunc` 符号（来源平台：950；证据等级：tool_observed）。
- 旁路中间态：`IGNORE_INFER_ERROR=1` 下 RunGraph success 且精度校验 PASS（证明图执行链路本身正常）（来源平台：950；证据等级：tool_observed）。
- 修复后（重编 + 重装 + 移除旁路）：GE IR 示例 `RunGraph success` / `Session run ir compute graph success`，输出 shape/dtype 正确（含 argmax 输出 dtype=9(DT_INT64)、MoE 四输出 dtype 1/0/3/3 正确推导）、`Precision is ok`、EXIT_CODE=0；aclnn eager 示例均无回归（来源平台：950；证据等级：tool_observed）。
- 无回归（scatter trace）：修复后 UT 17/17、ST Real 876/876、Mock ST 876/876、PyTorch 795/795（来源平台：950；证据等级：tool_observed）。
- A/B 反证（argmax trace）：仅修 autogen `op_proto.h`（TensorType::ALL() 换显式类型）重装后仍报同一 `infer_datatype` 错误；补注册 + 重建拷贝库后才通过（来源平台：950；证据等级：tool_observed）。

### 8. 边界补充（复现增量）

- "eager 全绿 / 图模式一挂"的分裂是缺 InferDataType 注册的指纹：aclnn 调用前已显式设置输出 tensor 的 dtype，完全绕过图模式推导——排查时应把"ACLNN 能跑"排除出"算子注册完备"的证据。
- 多 dtype 输出（TensorType 多候选、输出 dtype 跟随输入）与固定输出 dtype 两类算子均实测必须注册；输出 dtype 完全固定的算子是否可豁免未单独验证。
- `ge.opsProtoLibPath` 等加载选项对本案无效：加完后报错逐字不变（库本就经 opp/vendors/config.ini 被 PluginManager 加载），不要在加载面反复投入。
- CANN 9.2.0 头文件事实：`register/op_impl_registry.h` 只有 `IMPL_OP_INFERSHAPE`/`IMPL_OP_OPTILING`/`IMPL_OP` 三个注册宏（**无 `IMPL_OP_INFERDTYPE`**）；自定义函数签名 `InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *)`；`InferDataTypeContext` 方法名为 `GetInputDataType/SetOutputDataType`（无 `SetOutputType/GetInputType`）。

### 9. 坏实践增补

- 把 `IGNORE_INFER_ERROR=1` 固化进 run.sh/README 当"修复"交付；或全局 export 静默掩盖所有图的推导错误。
- 见到 "Please confirm whether the op_proto shared library (.so) has been loaded successfully" 就只查 so 加载（多 trace 误诊实录：补全 `LD_LIBRARY_PATH`、`LD_PRELOAD` proto 库、`ge.opsProtoLibPath` 三种手段均不改变报错；nm 证明注册变量在库内）。
- 只改 autogen `op_proto.h` 的 `TensorType::ALL()` 就认为修好了（A/B 实证不充分）。
- 凭直觉写 `IMPL_OP_INFERDTYPE` 宏和 `SetOutputType/GetInputType` 方法名，不先检索头文件确认 API 是否存在。
- 修完 op_host 不重编重装算子包就重跑示例（注册信息在 so 里，不重装则 GE 仍读旧注册表）。

### 10. 误诊增补

- 误诊"op_proto 库未被 GE 加载"：错误文本第一句的字面引导——多 trace 均以"补路径/预载后报错逐字不变 + nm/DEBUG plog 证明库在且已注册其余项"排除；错误文案覆盖"未加载"与"未开发函数"两种成因，先核对注册面再怀疑加载面。
- 误诊"ge_check_op.json 的 infer_shape_error = shape 推导问题"：实际失败点是同 pass 内的 InferDataType，plog 才给出准确指向。
- 误诊"第一版修复编译错即走弯路"：写了不存在的 `IMPL_OP_INFERDTYPE` 宏与 `SetOutputType/GetInputType` 方法名，编译报 `'class gert::InferDataTypeContext' has no member named 'SetOutputType'` 与 `expected constructor, destructor, or type conversion before '('`——最终被 `InferOutDataTypeSameWithFirstInput()` 一行方案替换。