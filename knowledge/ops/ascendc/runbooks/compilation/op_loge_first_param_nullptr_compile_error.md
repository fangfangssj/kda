---
type: "Runbook"
title: "OP_LOGE 首参传 nullptr 触发编译错"
description: "op_common/log/log.h 的 OP_LOGE(opName, ...) 首参为 nullptr 时，宏内部 GetOpInfo 模板仍实例化 context->GetNodeType()，报 base operand of '->' is not a pointer（CANN 9.1.0 与 9.2.0 双版本复现）；首参应传 gert::TilingContext* 等真实上下文指针，无 context 的 helper 须把 context 作为形参一路传入。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# OP_LOGE 首参传 nullptr 触发编译错

## 现象

自定义 GRU 类算子的 host Tiling（`op_host/arch35/<op>_tiling.cpp`）里，Matmul tiling 辅助函数不持有 context，错误日志首参图省事传了 nullptr：

```cpp
int64_t ret = tiling.GetTiling(mmTiling);
OP_CHECK_IF(ret != 0, OP_LOGE(nullptr, "Gi MatmulApiTiling failed"), return ge::GRAPH_FAILED);
```

`bash build.sh --soc=ascend950 -j8`（CANN 9.2.0）编译报 4 处同错，错误点落在 CANN 头文件内部而非业务代码（路径泛化后）：

```text
op_host/arch35/<op>_tiling.cpp:126:5: required from here
$ASCEND_HOME_PATH/x86_64-linux/include/op_common/log/log.h:100:26:
error: base operand of '->' is not a pointer
  100 |     std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
op_common/log/log.h:102:15: error: base operand of '->' is not a pointer
  102 |     opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
```

报错在第三方头里，业务行号只出现在 "required from here"，定位有一定迷惑性。

## 适用场景

- 工程形态：自定义算子仓开发，op_host 侧 TilingFunc/辅助函数的 `OP_LOGE`/`OP_CHECK_IF` 错误日志。
- API 路径：`op_common/log/log.h` 的 `OP_LOGE(opName, ...)`、`GetOpInfo<T>()` 模板、`OP_CHECK_NULL_WITH_CONTEXT` 宏。
- 验证过的平台与 CANN 版本：Ascend950（build.sh --soc=ascend950 编译流水线），CANN 9.2.0（与 frontmatter 一致）。
- 适用粒度：platform_wide（CANN 版本绑定，与算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：opdev 侧 `OP_LOGE(errCode, fmt, ...)`（op_api 层，首参是错误码，另一族宏）不在本卡；`GetOpInfo` 对字符串 opName 的重载路径未在本会话验证。

## 根因

CANN 9.2.0 `op_common/log/log.h` 的 `OP_LOGE(opName, ...)`（1077 行）内部把首参送进模板函数：

```cpp
typename std::enable_if<IsContextType<T>(), std::string>::type GetOpInfo(T context)
{
    if (context == nullptr) {
        return "nil:nil";
    }
    std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
    opInfo += ":";
    opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
    return opInfo;
}
```

传 `nullptr` 时 `T` 推导为 `std::nullptr_t`，该实例化路径仍会编译函数体里的 `context->GetNodeType()`——`std::nullptr_t` 不是指针，`->` 作用于它直接编译错（`if (context == nullptr)` 的提前返回救不了编译期）。传真实的 `gert::TilingContext*`（满足 IsContextType，有 GetNodeType/GetNodeName）则正常。

## 解决配方

修复前（辅助函数无 context，首参传 nullptr，verbatim）：

```cpp
// 计算 Matmul Tiling (gi: X @ W_ih^T, per-timestep)
static ge::graphStatus CalcGiMatmulTiling(
    const platform_ascendc::PlatformAscendC& ascendcPlatform,
    int64_t batchPerCore, int64_t inputDim, int64_t hiddenSize,
    bool hasBias, ge::DataType dataType,
    optiling::TCubeTiling& mmTiling)
{
    ...
    int64_t ret = tiling.GetTiling(mmTiling);
    OP_CHECK_IF(ret != 0, OP_LOGE(nullptr, "Gi MatmulApiTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

修复后（把 gert::TilingContext* 一路传入辅助函数，OP_LOGE 首参用 context）：

```cpp
// 计算 Matmul Tiling (gi: X @ W_ih^T, per-timestep)
static ge::graphStatus CalcGiMatmulTiling(
    gert::TilingContext* context,
    const platform_ascendc::PlatformAscendC& ascendcPlatform,
    int64_t batchPerCore, int64_t inputDim, int64_t hiddenSize,
    bool hasBias, ge::DataType dataType,
    optiling::TCubeTiling& mmTiling)
{
    ...
    int64_t ret = tiling.GetTiling(mmTiling);
    OP_CHECK_IF(ret != 0, OP_LOGE(context, "Gi MatmulApiTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 调用侧同步传 context：
OP_CHECK_IF(CalcGiMatmulTiling(context, ascendcPlatform, batchPerCore, inputDim, hiddenSize, ...),
            OP_LOGE(context, "CalcGiMatmulTiling failed"), return ge::GRAPH_FAILED);
```

（gh 方向的 CalcGhMatmulTiling 同样处理。）

## 设计决策

- 选择"把 context 传进辅助函数"而非"删掉日志只留返回码"：保留错误日志的 opInfo（节点名/类型）上下文，排障时能定位到算子节点；代价是多一个函数参数。
- 也可选字符串字面量当 opName 的路径（本会话未验证），context 方案与 add_example 模板一致，风险最低。
- 当前方案路线档位：recommended_fastpath。

## 预防

- host 侧写 `OP_LOGE` 首参永远给真实 context 指针；辅助函数从签名上就接收 `gert::TilingContext*`。
- 见到 `op_common/log/log.h: xxx error: base operand of '->' is not a pointer` 且 "required from here" 指向业务文件时，先查业务代码里 `OP_LOGE(nullptr, ...)` 一类调用。

## 验证信号

来源说明：该算子迭代一编译循环，修复前后连续两轮完整构建对照（Ascend950PR 构建链，CANN 9.2.0）。

- 修复前一轮构建 4 处 `base operand of '->' is not a pointer`，错误栈指向 log.h:100/102，触发点 tiling.cpp:126（来源平台：950 编译流水线；证据等级：tool_observed）。
- OP_LOGE 首参改为 context 并给 CalcGi/CalcGhMatmulTiling 增加 context 参数后，下一轮构建该 4 处错误全部消失（剩余错误转为 op_api 侧独立问题），host 侧随后编译通过（来源平台：950 编译流水线；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.2.0 的 log.h 实现（GetOpInfo 模板 + enable_if 行为）；其他版本未验证，推断可迁移（未验证，推断可迁移）。
- 只验证了 `gert::TilingContext*` 作为首参；其他满足 IsContextType 的上下文类型（如 InferShapeContext）推断可用、未验证。

**未决项**：

- `IsContextType<std::nullptr_t>()` 为真导致函数体被实例化的精确 SFINAE 行为未深究（不影响结论：传 nullptr 必炸）。
## 坏实践

- 给 OP_LOGE 首参传 nullptr（"反正只是日志"）——它不是普通格式串前缀，是模板分发用的 opName 槽位。
- 报错在 CANN 头内部就怀疑工具链，不看 "required from here" 的业务触发点。

## 误诊

- 第一反应以为是 `OP_CHECK_NULL_WITH_CONTEXT` 宏的 context 用法问题（该宏在本会话中其实是正确使用的），逐层 grep 宏定义后才定位到 `OP_LOGE(nullptr, ...)` 是触发源（该中间假设为会话内推断，被头文件展开的 tool_observed 证据纠正）。

## 增量（CANN 9.1.0 独立复现与 helper 签名透传模式）

另一独立轨迹并入第二条独立 trace（MoE 分组矩阵乘算子 GenerateMatmulTiling helper，Ascend950PR + CANN 9.1.0），把版本覆盖从 9.2.0 扩到 9.1.0，并补齐"helper 无 context"这一高发场景的修复模式：

**1. 触发形态**：TilingFunc 的 Matmul tiling helper 不持有 context，错误日志首参图省事传了 nullptr：

```cpp
    int64_t ret = tilingApi.GetTiling(mmTiling);
    OP_CHECK_IF(ret != 0,
                OP_LOGE(nullptr, "MatmulApiTiling GetTiling failed, ret=%ld", ret),
                return ge::GRAPH_FAILED);
```

`bash build.sh --soc=ascend950 -j1` 报 4 处同错，报错位置落在 CANN 头文件内而非业务代码：

```text
op_common/log/log.h:100:26: error: base operand of '->' is not a pointer
  100 |     std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
op_common/log/log.h:102:15: error: base operand of '->' is not a pointer
  102 |     opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
```

**2. 修复模式（helper 增加 context 形参，日志传真实 context；调用点同步透传）**：

```cpp
// before（helper 无 context，日志首参传 nullptr）
static ge::graphStatus GenerateMatmulTiling(platform_ascendc::PlatformAscendC& ascendcPlatform,
                                             int64_t maxMi, int64_t N, int64_t K,
                                             ge::DataType dtype, bool hasBias,
                                             optiling::TCubeTiling& mmTiling)
{
    ...
    int64_t ret = tilingApi.GetTiling(mmTiling);
    OP_CHECK_IF(ret != 0,
                OP_LOGE(nullptr, "MatmulApiTiling GetTiling failed, ret=%ld", ret),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

```cpp
// after（helper 增加 gert::TilingContext* context 形参，日志传真实 context；调用点同步透传）
static ge::graphStatus GenerateMatmulTiling(gert::TilingContext* context,
                                             platform_ascendc::PlatformAscendC& ascendcPlatform,
                                             int64_t maxMi, int64_t N, int64_t K,
                                             ge::DataType dtype, bool hasBias,
                                             optiling::TCubeTiling& mmTiling)
{
    ...
    int64_t ret = tilingApi.GetTiling(mmTiling);
    OP_CHECK_IF(ret != 0,
                OP_LOGE(context, "MatmulApiTiling GetTiling failed, ret=%ld", ret),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 调用点（TilingFunc 入口）：
    OP_CHECK_IF(GenerateMatmulTiling(context, ascendcPlatform, maxMi, N, K, dtype, hasBias, mmTiling)
                != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GenerateMatmulTiling error"),
                return ge::GRAPH_FAILED);
```

**3. 边界增补**：op_api（aclnn）层的 OP_LOGE 首参是错误码（如 `OP_LOGE(ACLNN_ERR_PARAM_INVALID, ...)`），传 nullptr 的问题是 op_host 层特有的；kernel 侧用 AscendC::PRINTF，不走此宏。

### 复现增量验证信号

- 修复前：`op_common/log/log.h:100:26: error: base operand of '->' is not a pointer`（共 4 处，均在 log.h 的宏展开行）（来源平台：950；证据等级：tool_observed）。
- 修复后：context 形参补齐后重新编译，log.h 内的 `->` 报错消失，host 侧最终 Built target 通过；全量回归 `build.sh --soc=ascend950 -j8` 生成 custom_opp_ubuntu_x86_64.run 与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。

### 复现增量坏实践

- 在无 context 的函数里写 `OP_LOGE(nullptr, ...)`——不是运行期空指针问题，而是编译期 `std::nullptr_t` 不能用 `->` 的硬错误。
- 报错行号全部指向 `op_common/log/log.h` 而非自己的源文件，第一眼容易当成 CANN 头文件损坏/版本不匹配；实际是宏实参类型错误。