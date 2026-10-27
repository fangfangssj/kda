---
type: "Runbook"
title: "CheckNotNull 占位 nullptr 致 161001 两形态"
description: 复用参数更多的判空辅助函数时把没有的形参（workspaceSize/executor）硬编码 nullptr 传入，OP_CHECK_NULL 对字面量 nullptr 恒触发：阻断形态让 GetWorkspaceSize 恒返 161001；噪声形态只打错误日志不中断调用，形成高误导性 plog。修法是校验范围与实参一一对应，出参在入口自查。
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# CheckNotNull 占位 nullptr 致 161001 两形态

## 现象

两条独立轨迹、同一反模式的两种表现（均为 aclnn L2 适配层 `op_api/aclnn_<op>.cpp` 的 CheckParams/CheckNotNull 辅助函数，Ascend950PR，CANN 9.1.0）：

**形态一（阻断——全部用例恒失败）**：MoE 分组矩阵乘算子修完 OP_CHECK 条件反转（161002）后，ST 用例继续在 `GetWorkspaceSize` 阶段失败，plog 报错原文：

```
[ERROR] OP: [op_error_check.h:34] errno[161001] Expected a value of type [unsigned long] for argument [workspaceSize] but instead found nullptr.
[ERROR] OP: [aclnn_bench_no_harness_grouped_matmul.cpp:188] errno[161001] CheckNotNull failed
```

161001 是空指针参数错误，但测试侧明明传了真实的 `&workspaceSize` 和 `&executor`——报错却坚持说 `workspaceSize` 是 nullptr。当时 `CheckParams` 只接收 7 个业务参数（不含 workspaceSize/executor），却调用同时校验 6 个指针的 `CheckNotNull(x, weight, group_list, y, nullptr, nullptr)`，为凑签名把后两个硬编码为 `nullptr`。

**形态二（噪声——功能照常但日志吓人）**：双线性插值算子运行期间，plog 稳定出现同样的两条 161001 日志（`[op_error_check.h:34][IsNullptr] ... argument [workspaceSize] ... found nullptr` + `[aclnn_..._resize_bilinear.cpp:79][CheckParams] CheckNotNull failed`），但同一运行里调用链并未中断：GetWorkspaceSize 正常返回，kernel 下发、同步全部继续，ST 全量 23/23 通过、输出数值正确。排查其它内核故障时，这两条 161001 日志具有强误导性——看起来像 GetWorkspaceSize 因空指针失败，实际不是。

## 适用场景

- 工程形态：自定义 aclnn 算子 op_api（L2 API）层的参数校验函数拆分/复用；依赖 plog 排查 aclnn 调用问题的联调。
- API 路径：`OP_CHECK_NULL(param, retExpr)`（`include/aclnn_kernels/common/op_error_check.h:103`，param 为 nullptr 时触发）；`CHECK_COND` / `OP_CHECK_NOTNULL` 宏族；`aclnnXxxGetWorkspaceSize(..., uint64_t* workspaceSize, aclOpExecutor** executor)` 出参约定。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（两条轨迹均与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——纯 C++ 函数复用 + CANN 宏语义问题，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：校验函数与被校验参数一一对应、无占位传参的写法不会触发本问题；`CHECK_COND` 在 bool 返回路径上"只记日志不中断"的精确宏展开语义未做源码级验证，形态二的"不中断"是运行观测结论。

## 根因

调用方只持有部分参数，却复用了参数更多的判空辅助函数，把字面量 `nullptr` 当实参传入：`OP_CHECK_NULL(workspaceSize, return false)` 检查的是"传入值是否为 nullptr"——占位 nullptr 让该检查必然触发。两种形态的差异在错误如何浮出：

- **阻断形态**：`CheckNotNull` 返回 false 经 `CHECK_COND(CheckNotNull(...), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed")` 直接把 161001 返回给调用方——`GetWorkspaceSize` 永远失败，无论调用方传什么。
- **噪声形态**：错误经 bool 返回路径未把 aclnn 入口中断（观测上同一运行内调用链继续执行），表现为"每次调用必打 161001 错误日志、但功能照常"。真实参数的判空早在入口 `OP_CHECK_NOTNULL` 完成，此处属重复且错误的二次校验。

## 解决配方

把"输出参数空指针校验"从复用的 CheckNotNull 中拆出，由 `GetWorkspaceSize` 入口对真实出参自查；判空辅助函数的形参收缩为实际持有的张量。

形态一修复前（占位 nullptr 必触发）：

```cpp
// ===== GetWorkspaceSize 参数校验 =====
static bool CheckNotNull(const aclTensor* x, const aclTensor* weight,
                          const aclIntArray* group_list, aclTensorList* y,
                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(group_list, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                const aclIntArray* group_list, int64_t split_item,
                                bool transpose_weight, aclTensorList* y)
{
    CHECK_COND(CheckNotNull(x, weight, group_list, y, nullptr, nullptr),
               ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    ...
}
```

修复后（校验范围与实际持有的参数对齐；出参在 GetWorkspaceSize 入口自查）：

```cpp
// ===== GetWorkspaceSize 参数校验 =====
static bool CheckNotNull(const aclTensor* x, const aclTensor* weight,
                          const aclIntArray* group_list, aclTensorList* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(group_list, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                const aclIntArray* group_list, int64_t split_item,
                                bool transpose_weight, aclTensorList* y)
{
    CHECK_COND(CheckNotNull(x, weight, group_list, y),
               ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    ...
}
```

```cpp
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输出参数空指针校验
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    // 参数校验
    auto ret = CheckParams(x, weight, bias, group_list, split_item, transpose_weight, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
```

形态二修复（两处编辑，形参收缩 + 调用点去掉 nullptr 实参）：

```cpp
static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}
```

```cpp
    CHECK_COND(CheckNotNull(x, y), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
```

判别 / 定位方法：

1. plog 中 `[op_error_check.h:34][IsNullptr] errno[161001] ... for argument [workspaceSize/executor] ... found nullptr` 且报错的参数是**函数出参**——调用方根本不会传 nullptr 的参数被报空，第一怀疑就是校验函数调用点有字面量 nullptr 实参。
2. 读 plog 时区分"记了错误日志"与"调用失败返回"：核对返回码与后续执行痕迹再定性，161001 日志不必然意味着 GetWorkspaceSize 失败。
3. `grep -n "CheckNotNull(" op_api/*.cpp` 核对所有调用点实参不是 nullptr 字面量。

## 设计决策

- 选"拆分校验范围 + 入口自查"而不是"把 workspaceSize/executor 一路传进 CheckParams"：出参校验语义上属于 `GetWorkspaceSize` 自身的契约（拿到出参即可校验），不必穿透两层；也避免校验函数签名随接口演进出更多占位参数。
- 形态二选"移除冗余形参"而不是"把真实 workspaceSize/executor 传进去"：入口处 `OP_CHECK_NOTNULL` 已对真实形参判空，CheckParams 内再查一遍属重复；收缩为只查张量与检查函数的职责对齐。所有套用该 L2 模板的算子都应检查自己的 CheckParams 是否同样误传。

## 预防

- 复用校验函数时参数必须与被校验对象一一对应；没有的参数宁可拆成两个校验函数，也不要用 `nullptr` 占位凑签名。
- `OP_CHECK_NULL` 检查的是"传入值是否为空"，不是"调用方是否传了"——占位 nullptr 等于主动喂给它一个空指针。
- 评审口径：校验函数调用处出现字面量 `nullptr` 实参即打回；grep 所有 `CheckNotNull(` 调用点核对实参。

## 验证信号

以下信号来自两条独立轨迹的 ST 联调会话（形态一：分组矩阵乘首轮 ST；形态二：双线性插值示例运行 plog 观测与修复后回归），均为 NPU 实测。

- 形态一修复前：全部用例 `[FAIL] GetWorkspaceSize 失败: 161001`；plog `[op_error_check.h:34] errno[161001] Expected a value of type [unsigned long] for argument [workspaceSize] but instead found nullptr` + `[aclnn_..._grouped_matmul.cpp:188] CheckNotNull failed`（来源平台：950；证据等级：tool_observed）。
- 形态一修复后：重编译安装并复跑 ST，161001 消失，流程推进到 kernel 启动阶段（后续 timeout 挂起为另一独立根因），证明 161001 闭环（来源平台：950；证据等级：tool_observed）。
- 形态二噪声观测：plog 出现同型 161001 两条，同一运行内 kernel 任务继续下发执行（调用未中断）；ST 全量 23/23 通过（来源平台：950；证据等级：tool_observed）。
- 形态二修复后回归：CheckNotNull 收缩为只校验 x/y 并重编译安装后，ST 全量 23/23 通过、UT 通过，无功能回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 形态二"161001 日志修复后消失"未在修复后单独复查 plog（回归只跑了功能测试）；如需确认噪声清零，装后重跑任一 aclnn 调用并 grep plog 即可，一分钟成本。
- `CHECK_COND`/`OP_CHECK_NULL` 为何在 bool 返回路径上"只记日志不中断"未做头文件源码级验证，形态二对该行为的表述止于运行观测与代码路径推断。
- 本卡只解决"占位 nullptr 导致校验必失败/假日志"；同轮出现的其它失败（段错误、内核越界等）另有独立根因，不要把 161001 修完就默认一切正常。

**未决项**：

- 形态二修复后未复查 plog 确认 161001 噪声清零（低成本补验项）。
- `OP_CHECK_NULL`/`CHECK_COND` 宏在 bool 函数内的权威语义（含返回错误码的转换行为）未从头文件源码确认，本卡对该机制的表述标为推断。
## 坏实践

- 用 `nullptr` 占位去复用参数更多的校验函数。
- 把出参校验塞进只拿得到入参的中间函数里。
- 排查时把 plog 里的 161001/CheckNotNull failed 当作调用失败的铁证，围绕"为什么传了空指针"展开调查（实际调用方传的是合法地址）。

## 误诊

- 形态二的代码检视报告曾将其判为 CRITICAL："GetWorkspaceSize 恒定失败、算子完全不可用"——与运行证据直接矛盾（同期 ST 23/23 通过、示例正常）。检视只做了静态推演（OP_CHECK_NULL 恒 return false → CHECK_COND 返回错误码），未核对宏在 bool 返回路径上的实际控制流，也未用运行证据校准严重级。教训：静态"恒失败"结论要用"功能明明在跑"的运行事实反问一遍。
