---
type: "Runbook"
title: aclScalar 无 Get()，读标量用 ToInt64()
description: aclnn 层校验函数里写 dim->Get(dimValue) 编译报 'const aclScalar' has no member named 'Get'；CANN aclnn/opdev 的 aclScalar 取值方法是 ToInt64()（common_types.h），返回值直接接收即可。
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-16T10:23:52Z'
platforms: ["950"]
---

# aclScalar 无 Get()，读标量用 ToInt64()

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子 BenchNoHarnessCummin 编写 aclnn 两段式接口（`op_api/aclnn_bench_no_harness_cummin.cpp`）的入参校验：算子签名 `aclnnBenchNoHarnessCumminGetWorkspaceSize(const aclTensor* input, const aclScalar* dim, ...)`，需要把 `dim` 标量读出来做范围校验。按"输出参数回填"的直觉写成：

```cpp
int64_t dimValue = 0;
dim->Get(dimValue);
```

第二次执行 `bash build.sh --soc=ascend950 -j8`（host 侧 tiling/proto 已过）编译到 `cust_opapi` 目标时失败，报错原文：

```text
/workspace/.../op_api/aclnn_bench_no_harness_cummin.cpp: In function 'bool CheckDim(const aclTensor*, const aclScalar*)':
.../aclnn_bench_no_harness_cummin.cpp:94:10: error: 'const aclScalar' {aka 'const class aclScalar'} has no member named 'Get'
     94 |     dim->Get(dimValue);
        |          ^~~

/workspace/.../op_api/aclnn_bench_no_harness_cummin.cpp:148:10: error: 'const aclScalar' {aka 'const class aclScalar'} has no member named 'Get'
   148 |     dim->Get(dimValue);
```

同一份日志里还并列出现 OP_CHECK 宏参数错误（独立根因，另见相关知识卡）。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子开发，op_api/aclnn 层（L2 两段式接口 + L0 launcher）。
- API 路径：`aclnn/opdev/common_types.h` 的 `aclScalar` 类；场景为 aclnn 入口把 `const aclScalar*` 标量属性读成整型。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（验证方式为编译通过）。
- 适用粒度：platform_wide——所有 aclnn 接口中读取 aclScalar 标量属性的自定义算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非 int64 标量（ToInt32/ToFloat 等同族方法以 common_types.h 为准，本开发会话 未逐一验证）；graph 侧（gert）的属性读取是另一套 API（TilingContext 须经 GetAttrs()）。

## 根因

CANN `aclnn/opdev/common_types.h` 中 `aclScalar` 的取值方法是 `ToInt64()`（common_types.h:312 一带），不存在输出参数风格的 `Get(...)`；返回值直接接收。

机制链：`Get(dimValue)` 找不到任何成员 → 编译失败；`ToInt64()` 以值返回 int64，校验逻辑改为 `int64_t dimValue = dim->ToInt64();` 后无需预声明变量，也避免未初始化路径。

## 解决配方

修复前（`op_api/aclnn_bench_no_harness_cummin.cpp` 的 CheckDim 函数与 GetWorkspaceSize 中的读取段，verbatim）：

```cpp
static bool CheckDim(const aclTensor* input, const aclScalar* dim)
{
    int64_t dimValue = 0;
    dim->Get(dimValue);
    int64_t rank = static_cast<int64_t>(input->GetViewShape().GetDimNum());
    OP_CHECK(dimValue >= -rank && dimValue < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "dim out of range: dim=%ld, rank=%ld, valid range=[%ld, %ld]",
                     dimValue, rank, -rank, rank - 1),
             return false);
    return true;
}
```

修复后（CheckDim 全函数，GetWorkspaceSize 中的 `dim->Get` 同步改为 `ToInt64()`）：

```cpp
static bool CheckDim(const aclTensor* input, const aclScalar* dim)
{
    int64_t dimValue = dim->ToInt64();
    int64_t rank = static_cast<int64_t>(input->GetViewShape().GetDimNum());
    OP_CHECK(dimValue >= -rank && dimValue < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "dim out of range: dim=%ld, rank=%ld, valid range=[%ld, %ld]",
                     dimValue, rank, -rank, rank - 1),
             return false);
    return true;
}
```

定位探针（在 CANN 头文件里确认可用方法，可复用）：

```bash
grep -rn "GetScalar\|GetValue\|aclScalar" "$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/" | grep -i scalar | head
# 再读 aclnn/opdev/common_types.h 的 class aclScalar 段，可见 ToInt64()（约 :312）
```

## 预防

- 写 aclnn 层标量读取前，先读 `aclnn/opdev/common_types.h` 的 `aclScalar` 类定义，确认本版本的方法名。
- 记住分层口径：op_api 层标量 = `aclScalar::ToInt64()`；op_host/tiling 层属性 = `GetAttrs()->GetAttrPointer<int64_t>`；op_def 层声明 = `.Attr("dim").Int()`。三层三套 API，不要互相套用。

## 验证信号

以下信号来自该算子骨架搭建会话的编译修复循环。

- 第二次编译：`error: 'const aclScalar' {aka 'const class aclScalar'} has no member named 'Get'`（CheckDim 与 GetWorkspaceSize 各一处）（来源平台：950；证据等级：tool_observed）。
- 头文件核查：`common_types.h` 的 aclScalar 提供 `ToInt64()`（来源平台：950；证据等级：tool_observed）。
- 修复后第三次编译：`cust_opapi` 目标编译链接通过（`Linking CXX shared library libcust_opapi.so` / `Built target cust_opapi`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅核对 CANN 9.1.0 的 aclScalar 方法集；其他版本以本机头文件为准（未验证，推断可迁移）。
- 未验证 bool/float 等其他类型标量的同族方法名。

**未决项**：

无（该错误已闭环到编译通过）。
## 坏实践

- 用输出参数风格臆造 `Get(value)` 方法。
- 先声明 `int64_t dimValue = 0;` 再赋值——`ToInt64()` 值返回可直接初始化，也消除"读取失败保留 0"的隐式路径。

## 误诊

无。
