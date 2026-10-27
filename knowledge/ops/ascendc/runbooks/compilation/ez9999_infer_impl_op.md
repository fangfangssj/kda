---
type: "Runbook"
title: EZ9999 找不到 infer 回调 — 补 IMPL_OP 注册
description: GEIR 图模式全量 EXEC_FAILURE：OpDef SetInferDataType 绑定不进运行时注册表；在注册单元补 IMPL_OP V2 注册后恢复。后续增量（）：IMPL_OP_INFERSHAPE 只注册 InferShape、OpDef(OP_ADD)/REG_OP-only 无推导注册三种同签名形态（ErrorNo 1343242270）；GEGetErrorMsgV2 失败分支探针；IGNORE_INFER_ERROR=1 示例级旁路（须构图 desc 自足，EZ9999 文本仍打印但非致命）与 vendor op_proto/op_tiling 库路径；输出 dtype 随首输入可用 .InferOutDataTypeSameWithFirstInput() 便捷修复。
tags: [task.debugging, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T01:58:00Z'
platforms: ["950"]
---

# EZ9999 找不到 infer 回调 — 补 IMPL_OP 注册

## 现象

在 AscendC 算子仓开发流程中给一个自定义算子（2 输入 5 输出、带 `has_bias` 布尔属性、宿主侧自写 InferShape/InferDataType）打通 GEIR 图模式验证：InferShape/InferDataType 从桩替换为完整实现后，`cmake` 构建算子包成功（`libcust_opsproto_rt2.0.so` 正常产出，`nm` 可见 `ge::InferShape` / `ge::InferDataType` 符号），kernel 二进制与 ops-info 均安装进本轮隔离 staging，随后跑 TTK GEIR 验证（`bash skills/geir-route-tester/scripts/run_ttk_geir.sh "$PWD" <OP> verify`）。结果正向套件 44 个用例**全部 EXEC_FAILURE、0 个 PASS**，结果 CSV 中每行报错原文：

```
ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!
EZ9999[PID: 2660368] 2026-09-07-14:08:19.613.486 (EZ9999):  Can not find Node op1[ThnnFusedGruCellGrad] cust...
```

极具迷惑性的是：编译、链接、安装全绿，staging 加载日志（`AddSoToRegistry:Save so symbol and handle in path[.../libcust_opsproto_rt2.0.so] successfully!`）明确显示 .so 已加载，而报错文案却提示"确认 op_proto .so 是否加载成功"——当时代码状态是：InferShape/InferDataType 函数本体已完整实现并编入 .so，但**只经 OpDef 的 `SetInferShape`/`SetInferDataType` 绑定**（include `register/op_def_registry.h`），没有任何 `IMPL_OP` 注册。

## 适用场景

- 工程形态：AscendC 算子仓开发（ops-registry-invoke 流程）下的 GEIR/GE 图模式验证（TTK `geir` 套件）；aclnn 直调路径不经图准备阶段的注册表查找，不适用本卡。
- API 路径：宿主侧推导回调注册——`register/op_def_registry.h` 的 `OpDef::SetInferShape`/`SetInferDataType` 与 `register/op_impl_registry.h` 的 `IMPL_OP` / `gert::OpImplRegisterV2`。
- 验证过的平台与 CANN 版本：Ascend950（`AddConfig("ascend950")`、`kernel/ascend950` 产物、上板 GE 执行）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——任何走 GE 图模式、含宿主推导回调的自定义算子。
- 路线优先级：**recommended_fastpath**。
- 明确不适用/未验证：CANN 9.0.0-beta.2 之外版本未验证；只缺 InferShape 注册（不缺 InferDataType）是否同样失败未单独验证（本卡修复同时注册两者）；下述机制链中"两条查找通路不对称"的 CANN 内部实现细节为日志级证据支撑的推断。

## 根因

宿主侧 InferDataType 回调只经 OpDef `SetInferDataType` 绑定、未进入 `OpImplRegisterV2` 运行时注册表，GE 图准备阶段按注册表查找 infer_datatype 回调失败。

机制链（DEBUG 日志逐条对应，均在同轮 staging 加载区段内）：

1. .so 加载成功：`AddSoToRegistry:Save so symbol and handle in path[.../op_proto/lib/linux/x86_64//libcust_opsproto_rt2.0.so] successfully!`
2. 但运行时注册表只登记了 tiling 系回调：
   ```
   RegisterOpImplToRegistry:LocalRegistry[21998484978668984] Op type[ThnnFusedGruCellGrad] register OP_IMPL : [Tiling][TilingParse]
   ```
   ——没有 `[InferShape]`、没有 `[InferDataType]`。
3. `MergeFunctions` 对该算子只出现 `tiling_parse / compile_info_creator / compile_info_deleter / tiling / max_tiling_data_size func has been registered`，无 infer_shape/infer_datatype 注册行。
4. 图准备阶段 InferShape 以 V2 被调起（`CallInferFunc:Op op1[ThnnFusedGruCellGrad] Call InferShapeFuncV2`），随后 dtype 推导按注册表查找失败——这才是被截断的 `Can not find Node ... cust` 的全文：
   ```
   InferDtypeByRegisteredFuncOrRule: ErrorNo: 4294967295(failed)Can not find Node op1[ThnnFusedGruCellGrad] custom infer_datatype func. Please confirm whether the op_proto shared library (.so) has been loaded successfully, and that you have already developed the infer_datatype func or marked the T-derivation rules on the IR.
   InferDataTypeOnCompile: ErrorNo: 4294967295(failed)[Check][InferDataType] result failed, op_desc[op1], ret[-1]
   CallInferFuncV2:Assert failed: [Call][InferFuncV2]Failed to infer data_type of node op1[ThnnFusedGruCellGrad].
   ```
   → InferShapePass 失败 → `RunGraph failed: -1` → EZ9999。
5. `nm` 证实"函数在 .so 里、注册对象不在"：修复前 `T _ZN2ge10InferShapeEPN4gert17InferShapeContextE`、`T _ZN2ge13InferDataTypeEPN4gert20InferDataTypeContextE`（函数已定义）、`U _ZN3ops5OpDef16SetInferDataTypeEPFj...`（OpDef 绑定为外部引用），但没有 `op_impl_register_<OpType>` 静态对象。

一句话：**符号存在 ≠ 运行时注册**。`IMPL_OP` 宏在静态初始化期构造 `gert::OpImplRegisterV2(#op_type)` 对象并链式登记回调，运行时的 `RegisterOpImplToRegistry`/`MergeTypesToImpl` 只认这条通路；只做 OpDef 绑定，推导回调就不在注册表里。

## 判别方法

对 EXEC_FAILURE 的 GEIR 结果日志（`tests/geir/logs/positive-source-static.log`，需 DEBUG 级）执行：

```bash
grep -a "Can not find Node\|RegisterOpImplToRegistry.*<OpType>\|MergeTypesToImpl.*<OpType>\|CallInferFunc.*<OpType>" \
  operators/<OP>_package/tests/geir/logs/positive-source-static.log | head -10
```

- 命中 `register OP_IMPL : [Tiling][TilingParse]` 而**无** `[InferShape]`/`[InferDataType]`，且随后出现 `Can not find Node op1[<OpType>] custom infer_datatype func` → 即本卡根因。
- 再用 `nm` 双确认（函数符号在、注册对象缺）：

```bash
nm operators/<OP>_package/build/libcust_opsproto_rt2.0.so | grep -i "op_impl_register\|InferDataType\|InferShape"
```

修复前：有 `T _ZN2ge...InferShape/InferDataType...` 但无 `op_impl_register_<OpType>`；修复后应出现 `_ZL38op_impl_register_<OpType>0`（bss 静态对象）。

## 设计决策

- **双注册并存**：`IMPL_OP(<OpType>).InferShape(...).InferDataType(...)` 与 OpDef 的 `SetInferShape`/`SetInferDataType` 绑定同时保留，两者指向同一函数指针。图模式验证工作流的路由门禁本来就要求 `RegisterOpImplToRegistry` 出现 `[InferShape]`/`[InferDataType]` 且 `MergeTypesToImpl:op type <OpType> infer_shape/infer_datatype func registered first.`——只靠 OpDef 绑定既过不了运行时也过不了路由取证。
- **备选方案（未采用）**：`OpImplRegisterV2::InferOutDataTypeSameWithFirstInput()` 内建规则（头文件注释：将第一个输入的 datatype 作为所有输出的 datatype；若与自定义 InferDataType 同时注册，最后注册的规则生效）。若算子输出 dtype 全随首输入且无需在推导阶段拒收非法 dtype，可用该规则替代自定义 InferDataType；本算子需要在图推理阶段拒收 dtype 越界/不一致输入（负向用例契约），故保留显式函数。
- **注册位置**：放在 op_graph 侧注册翻译单元（graph_infer.cpp）文件尾、`namespace ge` 之外（宏展开为全局作用域静态对象），include 从 `register/op_def_registry.h` 换成 `register/op_impl_registry.h`；InferShape 函数本体仍在 op_host 文件中，用前置声明衔接（两文件链入同一个 .so）。
- **为何不退回 V1**：仓库自带的 legacy V1 示例（`COMMON_INFER_FUNC_REG`）是编译检查用途、默认不链接；工作流明确把命中 `InferShapeFuncV1` 视为硬停止条件，必须走 V2（`IMPL_OP`）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`，x86_64-linux 头文件）；TTK GEIR runner（隔离 staging + `ASCEND_CUSTOM_OPP_PATH`）。
- 目标平台：Ascend950（`AICore().SetTiling(...).AddConfig("ascend950")`）。
- 相关文件（与事件流实际读写一一对应）：
  - `operators/<OP>_package/op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp`——本卡修改点（include + IMPL_OP 注册）
  - `operators/<OP>_package/op_host/thnn_fused_gru_cell_grad_infershape.cpp`——InferShape 本体（未改）
  - `operators/<OP>_package/op_host/thnn_fused_gru_cell_grad_def.cpp`——OpDef 绑定（保留）
  - `operators/<OP>_package/build/libcust_opsproto_rt2.0.so`——nm 检查对象
  - CANN 头：`register/op_impl_registry.h`、`register/op_def.h`、`register/op_def_registry.h`
- 构建/验证命令：
  ```bash
  source docs/<OP>/develop/env.sh
  cmake -S operators/<OP>_package -B operators/<OP>_package/build
  cmake --build operators/<OP>_package/build --target binary package -j
  bash skills/geir-route-tester/scripts/run_ttk_geir.sh "$PWD" <OP> verify
  ```

## 完整代码

**before**（`op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp` 修复前全文——只有函数定义与 OpDef 绑定，无注册）：

```cpp
/**
 * =============================================================================
 * <OP>_package/op_graph/thnn_fused_gru_cell_grad_graph_infer.cpp
 * =============================================================================
 * Role: InferDataType (V2) for ThnnFusedGruCellGrad（经 OpDef
 *       SetInferDataType 绑定，非 IMPL_OP 注册）。五个输出 dtype 一律随
 *       grad_hy（proto.md §2 三组合内逐 IO 同 dtype）。校验输入 dtype 在合法域
 *       {DT_FLOAT, DT_FLOAT16, DT_BF16} 内且双输入一致，非法输入返回
 *       GRAPH_FAILED。
 * =============================================================================
 */

#include "register/op_def_registry.h"

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

} // namespace ge
```

**after**（同一文件修复后全文——include 换 `op_impl_registry.h`，文件尾追加 IMPL_OP V2 注册）：

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

（OpDef 侧 `this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);` 保持不变。）

## 预防

给走 GE 图模式的算子仓上 GEIR 验证前过一遍检查清单：

1. 宿主推导回调一律补 `IMPL_OP(<OpType>)` V2 注册（`register/op_impl_registry.h`），不要只依赖 OpDef 的 `SetInferShape`/`SetInferDataType`——后者在图准备阶段不够用。
2. 构建后先 `nm` 自检：`nm <build>/libcust_opsproto_rt2.0.so | grep op_impl_register` 必须看到 `op_impl_register_<OpType><N>` 静态对象，再进六套件（单套件一轮十几分钟起步，别把注册问题留到图执行阶段才发现）。
3. 日志门禁三条：`RegisterOpImplToRegistry ... [<OpType>] ... [InferShape]`、`[InferDataType]` 与 `MergeTypesToImpl:op type <OpType> infer_shape/infer_datatype func registered first.`——必须落在**本轮 staging** 的 `AddSoToRegistry` 区段内（不能拿全局安装包的区段拼接证据）。
4. 报错文案让查".so 是否加载"，先用 `AddSoToRegistry` 日志排除加载问题，再查注册表——别被文案带偏去重装。

## 验证信号

以下信号来自该算子 GEIR 图模式联调：同一轮隔离 staging 验证运行（TTK GEIR 正向源码静态套件）的命令输出、CANN DEBUG 日志与结果 CSV；修复（IMPL_OP 注册）为该轮唯一代码改动，前后各跑完整一轮。

- 修复前：正向源码静态套件 44 用例全部 EXEC_FAILURE、0 PASS，报错原文 `ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error! ... Can not find Node op1[ThnnFusedGruCellGrad] cust...`（来源平台：Ascend950；证据等级：tool_observed）
- 注册表证据：`RegisterOpImplToRegistry:LocalRegistry[...] Op type[ThnnFusedGruCellGrad] register OP_IMPL : [Tiling][TilingParse]`——仅 Tiling/TilingParse，无推导回调（来源平台：Ascend950；证据等级：tool_observed）
- 精确定位：`InferDtypeByRegisteredFuncOrRule: ErrorNo: 4294967295(failed)Can not find Node op1[ThnnFusedGruCellGrad] custom infer_datatype func. ...` + `[Check][InferDataType] result failed, op_desc[op1], ret[-1]`（来源平台：Ascend950；证据等级：tool_observed）
- 修复后构建产物：`nm` 出现 `_ZL38op_impl_register_ThnnFusedGruCellGrad0` 静态注册对象（来源平台：Ascend950；证据等级：tool_observed）
- 修后首轮：首个用例从 EXEC_FAILURE 变为真实执行+比对通过（结果 CSV：`precision="PASS,PASS,PASS,100%,100%"`、`precision_status=PASS`）（来源平台：Ascend950；证据等级：tool_observed）
- 修后整轮：44/44 用例全部完成执行并进入精度比对，`Can not find Node` 报错消失（后续验证转入精度判读问题）（来源平台：Ascend950；证据等级：tool_observed）

## 失效边界

- 仅在 CANN 9.0.0-beta.2 + Ascend950 实测（单算子、单平台、单轮验证）；其他 CANN 版本的注册行为未验证，推断可迁移但未证实。
- aclnn 直调路径不经图准备阶段注册表查找，本卡不适用。
- "InferShape 经 OpDef 绑定即可被调起、InferDataType 必须走注册表"的两通路不对称性来自日志与头文件推断，CANN 内部实现源码未核；只缺 InferShape 注册的独立后果未验证。
- 单跑结果（内部 single_run），跨算子/跨轨迹复现待 curator 判定。

**未决项**：

- 只缺 InferShape 注册（InferDataType 注册在）是否同样失败：未单独验证——可在下个算子上做对照实验（分别只注册其一）。
- 该版本 CANN 中 OpDef `SetInferDataType` 存储的函数指针究竟被哪条通路消费：未追到 CANN 实现源码层；建议复现时用 `ltrace`/断点或读 `op_impl_space_registry_v2_impl.cc` 对应版本源码确认两通路边界。
## 坏实践

- 只写函数 + OpDef 绑定就认为注册完成——`nm` 里有 `T ge::Infer*` 符号不代表运行时注册表有条目。
- 见到"Please confirm whether the op_proto shared library (.so) has been loaded successfully"就去重装/排查 .so 路径——该文案在本场景是误导，加载证据（AddSoToRegistry）与失败点（注册表查找）是两回事。
- 把注册问题留到六套件整轮跑完才发现：GEIR 单用例约 15-20s、单套件 15-20 分钟，先 `nm`+路由日志三条自检再上套件。

## 误诊

- **误诊 1（环境残留）**：先发现 CANN OPP 全局目录 `.../opp/vendors/<OP>` 有旧版安装包，怀疑与 staging 冲突，执行 `uninstall.sh --quiet` 卸载后重跑——仍 0 PASS / 43 行 EXEC_FAILURE。残留安装不是根因；卸载这一步对问题无效。
- **误诊 2（符号缺失）**：一度怀疑 InferDataType 没编进 .so——`nm` 推翻：函数符号是 `T`（已定义），缺的是 `op_impl_register_<OpType>` 静态注册对象。

## 增量（IMPL_OP_INFERSHAPE 只注册 InferShape 的变体与 IGNORE_INFER_ERROR 旁路）

两个独立算子会话补齐"同一错误签名、另一批注册形态"：

**变体 1（IMPL_OP_INFERSHAPE 链只注册 InferShape）**：StridedSlice 类算子（Ascend950PR，CANN 9.1.0）的 `op_host` infershape 用 `IMPL_OP_INFERSHAPE(<Op>)` 注册链但只挂了 `InferShape`，GE IR 示例同样 `Run graph failed` 无细节；开日志后 GE 报 `shape_inference.cc:608 InferDtypeByRegisteredFuncOrRule ... Cannot find Node ... custom infer_datatype func`、`InferDataTypeOnCompile ... ret[-1]`、`infershape_pass.cc:122 Infer: ErrorNo: 1343242270`。符号探针：`nm -D libcust_opsproto_rt2.0.so` 只有 `GetInferShape`/`GetInferShapeRange` 未解析引用、没有 `GetInferDataType`——注册链上确实没挂 dtype 推导。修复（链尾补官方便捷注册，输出 dtype 恒等于第一输入的算子适用）：

```cpp
IMPL_OP_INFERSHAPE(BenchNoHarnessStridedSlice)
    .InferShape(InferShape4BenchNoHarnessStridedSlice)
    .InferOutDataTypeSameWithFirstInput();
```

重编重装后 GE IR 示例 `Run graph success`、输出与期望一致。边界：`InferOutDataTypeSameWithFirstInput()` 只适用于"输出 dtype == 第一输入 dtype"的算子；多输出或 dtype 变换算子（如 Cast）须手写 `InferDataType` 函数（`op_impl_registry.h` 的 `OpImplRegisterV2::InferDataType(InferDataTypeKernelFunc)` 通用入口）。

**变体 2（OpDef(OP_ADD) 注册模式 + 示例级 IGNORE_INFER_ERROR 旁路）**：去重类算子（OpDef/`OP_ADD` + `IMPL_OP_INFERSHAPE` 注册）GE IR 示例报同一 EZ9999（`shape_inference.cc:602`）。可运行规避 = 把 vendor 的 op_proto/op_tiling 库目录加入 LD_LIBRARY_PATH 并设 `IGNORE_INFER_ERROR=1`，固化到 `examples/run.sh`（verbatim）：

```bash
# 图模式需要加载 op_proto 和 op_impl 库
if [ "$EXAMPLE_MODE" = "graph" ]; then
    for lib_dir in \
        "${VENDOR_DIR}/op_proto/lib/linux/x86_64" \
        "${VENDOR_DIR}/op_proto/lib/linux/aarch64" \
        "${VENDOR_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64" \
        "${VENDOR_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"; do
        if [ -d "${lib_dir}" ]; then
            export LD_LIBRARY_PATH=${lib_dir}:${LD_LIBRARY_PATH}
        fi
    done
    # 算子使用 OpDef 注册，GE 运行时 infertype pass 需要 IGNORE_INFER_ERROR=1 跳过显式 infertype 检查
    export IGNORE_INFER_ERROR=1
fi
```

注意：加了库路径后 EZ9999 错误信息仍会打印（注册模式不提供该注册），但 `IGNORE_INFER_ERROR=1` 使其变为非致命，图照常执行且结果正确；示例侧应显式给输出分配上限 shape，不依赖运行期推导。该旁路对全图生效，可能掩盖其他算子的真实推导缺陷，只适合示例/测试环境。

**验证信号（复现增量）**：

- 修复前（StridedSlice）：`Run graph failed` 无细节；开日志后 `shape_inference.cc:608 ... Cannot find Node stridedSlice1[...] custom infer_datatype func` + `ErrorNo: 1343242270`（来源平台：950；证据等级：tool_observed）。
- 符号探针：修复前 `nm -D libcust_opsproto_rt2.0.so` 无 `GetInferDataType` 引用（来源平台：950；证据等级：tool_observed）。
- 修复后（补 `.InferOutDataTypeSameWithFirstInput()` 并重编重装）：GE IR 示例 `Run graph success`，输出 shape/数值与期望一致（来源平台：950；证据等级：tool_observed）。
- 规避后（unique，库路径 + IGNORE_INFER_ERROR=1）：图执行成功，输出 y（DT_INT32）/inverse（DT_INT64）正确、`Precision is ok`；EZ9999 仍打印但非致命（来源平台：950；证据等级：tool_observed）。
- run.sh 固化后清理重建：aclnn 与图模式两示例均 PASS（来源平台：950；证据等级：tool_observed）。

## 增量（GEGetErrorMsgV2 失败分支探针与构图 desc 自足前提）

registry-invoke 工程（InferShape 注册在 op_host 侧、编译进 `libcust_opmaster_rt2.0.so`，op_proto 只有 REG_OP 骨架）的逐元素最大值算子 GE IR 示例：`RunGraph` 失败 `ret = -1` 且失败分支默认只打印一行 "Run graph failed"。**RunGraph 失败分支必须先补 `GEGetErrorMsgV2()`（可加 `GEGetWarningMsgV2()`）打印真实错误**（verbatim 探针）：

```cpp
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed, ret = %d\n", GetTime().c_str(), ret);
        ge::AscendString emsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: Error message: %s\n", GetTime().c_str(), emsg.GetString());
        ge::AscendString wmsg = ge::GEGetWarningMsgV2();
        printf("%s - ERROR - [XIR]: Warning message: %s\n", GetTime().c_str(), wmsg.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }
```

补打后取得完整 TraceBack（`Cannot find Node ... custom infer_datatype func ... You can ignore this validation by exporting IGNORE_INFER_ERROR=1 ...`）。排除"so 未加载"用 `ldd` 确认 vendor `libcust_opsproto_rt2.0.so` 依赖全可解析、`opp/vendors/config.ini` 的 `load_priority` 含本 vendor。处置 = 图模式运行脚本 `export IGNORE_INFER_ERROR=1`；**fallback 声明：仅当构图侧显式提供全部输入/输出 desc 时采用**（本示例即如此）——跳过校验后若构图 desc 有误，错误将延迟到执行期才暴露。

**验证信号（复现增量）**：

- 绕过前：`Run graph failed, ret = -1`，完整错误含 `Cannot find Node max1[...] custom infer_datatype func`（来源平台：950；证据等级：tool_observed）。
- `IGNORE_INFER_ERROR=1` 下：`Session run ir compute graph success`，输出 max(1.0, 2.0)=2.0 正确，同时日志仍打印同一 EZ9999 信息但不再阻断（来源平台：950；证据等级：tool_observed）。
- 写入 run.sh 后清理重建复跑图模式 PASS（来源平台：950；证据等级：tool_observed）。

## 增量（REG_OP-only 模板算子形态与两分支判别）

fp16/bf16→int8 量化类算子（op_graph 目录只有 `*_proto.h`（REG_OP）+ CMakeLists，无 graph_infer 实现）GE IR 示例同型失败。判别用两分支探针（verbatim）：

```bash
# 分支一：op_proto so 是否已安装且可加载
find "${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_proto/" -type f
# 输出含 .../op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so  ← so 存在，排除缺库

# 分支二：算子工程是否有图模式推导实现
ls operators/<op>/op_graph/
# 输出仅 *_proto.h  CMakeLists.txt  ← 只有 REG_OP 原型，无 graph_infer
```

so 存在 + 无 graph_infer 文件 → 定位为「缺少 infer_datatype 注册」而非「库没加载」。可运行配方 = run.sh 图模式分支加 op_proto 库路径 + `IGNORE_INFER_ERROR=1`（verbatim）：

```bash
# 图模式需要加载自定义算子 op_proto 库，并忽略 infer_dtype 检查（自定义算子无 graph_infer 注册）
if [ "$EXAMPLE_MODE" = "graph" ]; then
    VENDOR_PROTO_DIR="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}/op_proto/lib/linux/x86_64"
    if [ ! -d "${VENDOR_PROTO_DIR}" ]; then
        VENDOR_PROTO_DIR="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}/op_proto/lib/linux/aarch64"
    fi
    if [ -d "${VENDOR_PROTO_DIR}" ]; then
        export LD_LIBRARY_PATH=${VENDOR_PROTO_DIR}:${LD_LIBRARY_PATH}
    fi
    export IGNORE_INFER_ERROR=1
fi
```

单独加载 op_proto 库不解决（注册事实不变，须与 IGNORE_INFER_ERROR 联用）。根治路线（未实施，供接续）：在 op_graph 增加 `*_graph_infer.cpp` 注册 `INFER_SHAPE_AND_TYPE_FUNC_REG`（或等价 infer_datatype 函数 / IR T-derivation 标注）并进 CMakeLists 一起编译安装。参照模板 add_example 的 op_graph 同样只有 proto.h——该模板 lineage 的算子默认都没有图模式推导注册，进 GE 图模式都会触发同一错误。

**验证信号（复现增量）**：

- 修复前：`Run graph failed` 无详情；补 GEGetErrorMsgV2 后取得 `EZ9999 ... Cannot find Node dynamic_quant1[...] custom infer_datatype func ... [FUNC:InferDtypeByRegisteredFuncOrRule][FILE:shape_inference.cc][LINE:602]`（来源平台：950；证据等级：tool_observed）。
- 排查观测：vendor op_proto so 已安装；op_graph 仅 proto.h + CMakeLists；模板 add_example 的 op_graph 同样无 graph_infer（来源平台：950；证据等级：tool_observed）。
- 绕过后：`Session run ir compute graph success`，输出 dtype/shape 正确、`Precision is ok`；run.sh 固化后 `--graph` 全流程 PASS（来源平台：950；证据等级：tool_observed）。

**增量合并后的失效边界补充（并行轨迹）**：

- IGNORE_INFER_ERROR=1 旁路的前提是构图侧已显式设置全部输入/输出 desc（示例显式上限分配/显式 TensorDesc）；完全依赖 GE 运行期 shape 推导的调用方式未验证，多算子图上其他节点的真实推导错误也会被吞掉。
- IGNORE_INFER_ERROR 的官方语义（仅 dtype 推断 or 全部 infer 失败）未查证文档；全局导出会掩盖所有算子的推导注册缺陷，不应写进产品构建脚本。
- OpDef/REG_OP-only 注册与 GE infertype 注册联动的机制层解释未做源码级验证；实测确证的是"错误出现 + IGNORE_INFER_ERROR=1 后图执行正确"这一输入输出关系。
- aarch64 安装未实测（脚本已按目录探测兼容）。