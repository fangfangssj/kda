---
type: "Runbook"
title: UT ExecuteTestCase {0} 重载歧义需显式 vector
description: "算子 UT 框架 ExecuteTestCase 第 4 参用 {0} 触发 string/vector 重载歧义编译失败；显式 std::vector<size_t>{0} 修复。"
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:15:37Z'
updated_at: '2026-09-16T11:15:37Z'
platforms: ["950"]
---

# UT ExecuteTestCase {0} 重载歧义需显式 vector

## 现象

为算子 `op_host` tiling 实现编写 gtest UT（`tests/ut/op_host/test_<op>_tiling.cpp`，使用 CANN UT 框架的 `tiling_context_faker.h` / `tiling_case_executor.h`）。首次编译 UT（`bash tests/ut/run.sh`，g++ 13.3.0）在三处报同族错误：

```text
test_<op>_tiling.cpp:101:20: error: call of overloaded 'ExecuteTestCase(const gert::TilingContextPara&, const ge::graphStatus&, uint64_t&, <brace-enclosed initializer list>)' is ambiguous
  101 |     ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, {0});
      |     ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
note: candidate: 'void ExecuteTestCase(const gert::TilingContextPara&, ge::graphStatus, uint64_t, const std::string&, const std::vector<long unsigned int>&)'
note: candidate: 'void ExecuteTestCase(const gert::TilingContextPara&, ge::graphStatus, uint64_t, const std::vector<long unsigned int>&)'
```

同样地，`{0}`（成功/空 Tensor 路径）与 `{}`（失败路径 `ExecuteTestCase(para, ge::GRAPH_FAILED, 0, {})`）都触发歧义。

## 适用场景

- 工程形态：算子 op_host UT（gtest + CANN tiling UT 框架 `tiling_case_executor.h`）
- API 路径：`ExecuteTestCase(para, expectStatus, expectTilingKey, expectWorkspaces[, expectTilingData])` 的两个重载；`ExecuteTiling(para, tilingInfo)`
- 验证过的平台与 CANN 版本：Ascend 950（ascend950，DAV_3510 平台参数）+ CANN 9.1.0，x86_64 主机（host 侧 UT，被测为 ascend950 包 tiling 源）
- 适用粒度：`operator_family`（所有用该 UT 框架写 tiling 用例的算子）
- 路线优先级：recommended_fastpath
- 明确不适用 / 未验证场景：不走 tiling_case_executor 的 UT（如 infershape 专用 executor）不涉及此 API

## 根因

UT 框架的 `ExecuteTestCase` 提供两个重载，第 4 参分别为 `const std::string&`（附带 expectTilingData 断言）与 `const std::vector<size_t>&`（只断言 workspace）；花括号初始化列表 `{0}` / `{}` 既能构造单元素 vector 也能构造含 `\0` 的 string，重载决议歧义。

## 解决配方

**1) 成功路径显式类型（before → after，成功与空 Tensor 两处同改）：**

before：

```cpp
    // ---- 第一阶段：使用 ExecuteTestCase 验证 status + tilingKey + workspace ----
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, {0});
```

after：

```cpp
    // ---- 第一阶段：使用 ExecuteTestCase 验证 status + tilingKey + workspace ----
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{0});
```

**2) 失败路径改用 `ExecuteTiling` 断言返回值（before → after）：**

before：

```cpp
static void VerifyTilingFailure(
    initializer_list<int64_t> inputShape,
    DataType dtype)
{
    auto para = CreateTilingContext(inputShape, dtype);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, {});
}
```

after：

```cpp
static void VerifyTilingFailure(
    initializer_list<int64_t> inputShape,
    DataType dtype)
{
    auto para = CreateTilingContext(inputShape, dtype);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(para, tilingInfo);
    EXPECT_FALSE(ret) << "Tiling should fail for unsupported dtype";
}
```

## 设计决策

- 失败路径弃用 `ExecuteTestCase(para, ge::GRAPH_FAILED, 0, {})` 改走 `ExecuteTiling` + `EXPECT_FALSE`：`{}` 歧义只是表因，失败路径本来就不需要 workspace/tilingData 断言，直接断言 tiling 布尔返回更贴近被测行为（同族算子 swi_glu 的 UT 即此写法）。
- 成功路径保留 `ExecuteTestCase`（其内部同时断言 status/tilingKey/workspace），只显式化类型。

## 预防

- 在该 UT 框架下调用 `ExecuteTestCase` 时，第 4 参一律写全 `std::vector<size_t>{...}`，不用裸花括号。
- 传 workspace 期望值时从同族算子 UT（如同期开发的 gelu/swi_glu 工程）抄调用形态，不要凭直觉写初始化列表。

## 验证信号

以下信号来自 x86_64 主机上的 UT 编译运行（被测为 ascend950 包 tiling 源）：

- 修复前：UT 编译三处 `call of overloaded 'ExecuteTestCase(...)' is ambiguous`，编译失败（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 编译通过并执行（该轮暴露的 TilingKey 预期值问题另行修正后最终 25/25 PASS，覆盖 TilingData 字段、TilingKey、空 Tensor、失败路径、多核切分、InferShape）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该重载布局基于 CANN 9.1.0 的 UT 框架 common 文件；若后续版本给 `ExecuteTestCase` 合并签名或补 explicit 构造，歧义可能消失。
- `EXPECT_FALSE(ret)` 只断言 tiling 整体失败，不区分失败原因（如 dtype 非法 vs shape 越界）。

**未决项**：

- 无。
## 坏实践

- 失败路径继续用 `ExecuteTestCase(para, ge::GRAPH_FAILED, 0, {})`——除歧义外，还绕过了对"tiling 确实返回失败"的直接断言。
