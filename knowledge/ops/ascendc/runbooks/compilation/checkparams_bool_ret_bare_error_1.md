---
type: "Runbook"
title: "aclnn 返回裸 1：校验函数 bool 被当状态码"
description: 自定义算子 op_api 层的 CheckParams 返回 bool，调用方用 CHECK_RET(ret == ACLNN_SUCCESS, ret) 包裹，成功时 true(1) 被当错误码返回，aclnnGetWorkspaceSize 恒返回 1（撞上基础 ACL 层 ACL_ERROR_INVALID_PARAM）；校验函数应返回 aclnnStatus，失败路径保留真实错误码。
tags: [task.debugging, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# aclnn 返回裸 1：校验函数 bool 被当状态码

## 现象

在 Ascend950PR（CANN 9.1.0）上联调自定义算子 BenchNoHarnessGroupNorm 的 C++ ST 测试（Real 模式，NPU 执行 + CPU golden 比对）时，41 个 L0 用例全部在两段式接口的第一段就失败：

```
[L0_001] 2D N=1 C=1 ng=1 eps=1e-05 dtype=float32
  [FAIL] GetWorkspaceSize 失败: 1
```

返回码是裸的 `1`。查 CANN 头文件 `opdev/op_errno.h` 确认 ACLNN 段错误码是 161001（PARAM_NULLPTR）/161002（PARAM_INVALID）/561101（CREATE_EXECUTOR）等，`1` 不属于 ACLNN 段，而是基础 ACL 层的 `ACL_ERROR_INVALID_PARAM`——但 op_api 代码里没有任何地方显式返回 1。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发，自定义算子的 op_api（aclnn L2 适配层）`GetWorkspaceSize` 接口。
- API 路径：CANN aclnn opdev 宏族 `CHECK_RET` / `CHECK_COND` / `OP_CHECK`（`include/aclnn/opdev/op_log.h`、`op_errno.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何使用 opdev 检查宏编写 aclnn 参数校验的自定义算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不使用 `CHECK_RET(cond, ACLNN_SUCCESS 比较)` 模式的代码；kernel 侧（op_kernel）代码；其他 CANN 版本宏语义未验证。

## 根因

参数校验函数 `CheckParams` 返回 `bool`，而调用方按状态码语义比较：

```cpp
auto ret = CheckParams(x, gamma, beta, numGroups, epsilon, y);
CHECK_RET(ret == ACLNN_SUCCESS, ret);
```

机制链：`CheckParams` 全部通过时 `return true` → `ret` 为 `true` → `ret == ACLNN_SUCCESS` 即 `1 == 0`，恒为 `false` → `CHECK_RET`（`if (!(condition)) return ret_value;`）把 `ret`（`true`）返回 → `bool` 提升为 `int` 的 `1`，正好撞上基础 ACL 的 `ACL_ERROR_INVALID_PARAM`。同时 `CHECK_COND` 在校验失败时从 `CheckParams` 内部 `return 161001/161002`，也会被 `bool` 返回类型截断成 `true`，错误码语义全部丢失——所以无论校验通过与否，接口都返回 1。

## 解决配方

修复前（op_api/aclnn_bench_no_harness_group_norm.cpp）：

```cpp
static bool CheckParams(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                        int64_t numGroups, float epsilon, const aclTensor* y)
{
    CHECK_COND(CheckNotNull(x, gamma, beta, y), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x, gamma, beta, y), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed");
    CHECK_COND(CheckShape(x, gamma, beta, y, numGroups), ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    OP_CHECK(epsilon <= 0 || epsilon > 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epsilon=%f not in (0, 1]", epsilon),
             return false);
    return true;
}
```

修复后（返回类型改为 `aclnnStatus`，成功返回 `ACLNN_SUCCESS`，失败路径返回真实错误码）：

```cpp
static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                               int64_t numGroups, float epsilon, const aclTensor* y)
{
    CHECK_COND(CheckNotNull(x, gamma, beta, y), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x, gamma, beta, y), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed");
    CHECK_COND(CheckShape(x, gamma, beta, y, numGroups), ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    OP_CHECK(epsilon <= 0 || epsilon > 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epsilon=%f not in (0, 1]", epsilon),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
```

调用方 `CHECK_RET(ret == ACLNN_SUCCESS, ret)` 的写法保留——它对 `aclnnStatus` 返回值是正确的。

定位命令（可复用）：

```bash
grep -n "ACLNN_ERR" $ASCEND_HOME_PATH/include/aclnn/opdev/op_errno.h
export ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=0
# 重跑 ST 二进制，观察 OP 日志中 CheckParams/GetWorkspaceSize 的 errno
```

## 设计决策

- 为什么改成 `aclnnStatus` 而不是让调用方判断 `if (!ret)`：`CHECK_COND` 宏本身就是"失败即 `return <错误码>`"的语义，校验函数返回 `aclnnStatus` 后这些宏可以原样工作并保留具体错误码（161001/161002），上层日志与调用方都能拿到真实失败原因；`bool` 方案会把所有失败压成同一个值。
- 当前方案路线档位：recommended_fastpath（正确性必需）。

## 预防

- op_api 层参数校验函数一律返回 `aclnnStatus`，成功 `return ACLNN_SUCCESS`；不要返回 `bool` 再与 `ACLNN_SUCCESS` 比较。
- 看到 aclnn 接口返回裸 `1`（而非 161xxx/561xxx 段）时，先怀疑"非 ACLNN 段的值被误当状态码返回"，直接检查调用链上所有 `return` 的类型。
- 在 `bool` 函数里用 `CHECK_COND`（它 `return` 的是 int 错误码），错误码会被隐式截断。

## 验证信号

以下信号来自该算子的 ST 联调会话：修复 CheckParams 返回类型并重装算子包后重跑同一组 L0 用例。

- 修复前：41/41 用例 `[FAIL] GetWorkspaceSize 失败: 1`（来源平台：950；证据等级：tool_observed）。
- 修复后：错误码迁移为 `161002`（`[FAIL] GetWorkspaceSize 失败: 161002`），证明参数校验逻辑真实生效、真实失败原因（当时的 epsilon 校验问题）开始暴露（来源平台：950；证据等级：tool_observed）。
- 错误码事实锚点：`op_errno.h:15-24` 定义 `ACLNN_ERR_PARAM_NULLPTR 161001`、`ACLNN_ERR_PARAM_INVALID 161002`、`ACLNN_ERR_INNER_CREATE_EXECUTOR 561101` 等，均不等于 1（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"校验函数返回 bool 被当状态码"这一模式；修复后暴露的 161002（epsilon 参数、条件写反）是独立根因，见 [OP_CHECK 断言条件写反卡](161002_op_check_condition_inverted.md)。
- `CHECK_RET`/`CHECK_COND` 宏语义以 CANN 9.1.0 的 opdev 头为准，其他版本未验证（未验证，推断机制可迁移）。

**未决项**：

- 无（该根因已闭环并经复测验证）。
## 坏实践

- 校验函数签名 `static bool CheckParams(...)` + 调用方 `CHECK_RET(ret == ACLNN_SUCCESS, ret)` 组合——成功路径必然返回 1。

## 误诊

- 算子包安装期曾报 `libcust_opapi.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor`，一度怀疑库符号问题导致返回 1——该符号由 libopapi.so 运行时解析、与本错误无关；真因在 CheckParams 的返回类型。
