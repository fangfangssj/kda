---
type: "Runbook"
title: GetBool 返回指针，隐式转 bool 致 hasInverse 恒 1
description: op_host TilingFunc 里 bool returnInverse = attrs->GetBool(0) 把 const bool* 赋给 bool，指针非空恒为 true，return_inverse=false 时 hasInverse 仍为 1、tilingKey 错选 HAS_INVERSE=1 模板（259 而非 3）；改为判空解引用后 UT 13/13 通过。
tags: [task.debugging, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# GetBool 返回指针，隐式转 bool 致 hasInverse 恒 1

## 现象

BenchNoHarnessUnique 算子（去重，支持 return_inverse 属性）的 op_host TilingFunc 里，读取布尔属性 `return_inverse` 的写法是：

```cpp
bool returnInverse = attrs->GetBool(0);
uint32_t hasInverse = returnInverse ? 1 : 0;
```

为 TilingFunc 编写 gtest 单测（UT faker 伪造 TilingContext 与属性表）后运行，所有 `return_inverse=false` 的用例断言失败，报错原文：

```
[TEST_CASE] tiling_int32_noinv_100
tests/ut/op_host/test_bench_no_harness_unique_tiling.cpp:231: Failure
Expected equality of these values:
  tiling->hasInverse
    Which is: 1
  param.expectHasInverse
    Which is: 0
hasInverse mismatch for case: tiling_int32_noinv_100

  -> totalNum=100, blockFactor=128, ubFactor=19776, sortPasses=8, keyWords=2, keySize=4,
     hasInverse=1, wsSize=804, blockDim=1, tilingKey=259
```

三个 noinv 用例（100/1000/33 元素）全部失败，inv 用例通过——`hasInverse` 与属性真值无关、恒为 1。连带后果：`tilingKey` 恒为 259（`HAS_INVERSE=1` 的模板实例）而不是 3（`HAS_INVERSE=0`），即算子在用户未请求 inverse 时也会按带 inverse 的模板编译与实例化。

## 适用场景

- 工程形态：AscendC 算子 op_host 侧（TilingFunc）读取算子布尔属性；任何用 `gert::RuntimeAttrs` 取属性的 host 代码。
- API 路径：`gert::RuntimeAttrs::GetBool(size_t index)`（返回 `const bool*`，见 `$ASCEND_HOME_PATH/include/exe_graph/runtime/runtime_attrs.h`）；tiling 内的 `ASCENDC_TPL_SEL_PARAM`（tilingKey 模板参数选择）。
- 验证过的平台与 CANN 版本：ascend950 算子工程、CANN 9.1.0（与 frontmatter `platforms` 一致；验证方式为 x86_64 宿主机上的 op_host gtest UT，运行的是真实 tiling 源码）。
- 适用粒度：platform_wide——所有 CANN 算子 host 侧属性读取，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 侧（device 代码）无此 API；非布尔属性（GetInt64 等同类指针返回型 API 未逐一验证，机制相同需同样小心）。

## 根因

`gert::RuntimeAttrs::GetBool(size_t index)` 的返回类型是 `const bool*`（属性数据指针），把它直接赋给 `bool` 变量触发 C++ 指针→bool 隐式转换——只要属性存在、指针非空就恒为 `true`，与属性真值无关。

机制链：`attrs->GetBool(0)` 返回 `const bool*` → 赋值 `bool returnInverse` 时非空指针隐式转 `true` → `hasInverse = 1` 恒成立 → `ASCENDC_TPL_SEL_PARAM` 按错的 hasInverse 选模板参数 → `tilingKey=259`（HAS_INVERSE=1）而非 `3`（HAS_INVERSE=0）。编译期不报错（合法隐式转换），属于静默语义错误，只有断言 tiling 字段的 UT 能抓住。

## 解决配方

修复前（tiling 读属性 + 空 Tensor 路径硬编码 0）：

```cpp
    // 获取属性 return_inverse
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    bool returnInverse = attrs->GetBool(0);
    uint32_t hasInverse = returnInverse ? 1 : 0;

    // 3. 计算 key 参数（int32 固定）
    uint32_t keyWords = KEY_WORDS_INT32;
    uint32_t keySize = KEY_SIZE_INT32;
    uint32_t sortPasses = SORT_PASSES_INT32;

    // 4. 空 Tensor 检查
    if (totalNum == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_INT32), 0);
        size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
```

修复后（判空 + 解引用；空 Tensor 路径复用同一 hasInverse 变量）：

```cpp
    // 获取属性 return_inverse
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* returnInversePtr = attrs->GetBool(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnInversePtr);
    bool returnInverse = *returnInversePtr;
    uint32_t hasInverse = returnInverse ? 1 : 0;

    // 3. 计算 key 参数（int32 固定）
    uint32_t keyWords = KEY_WORDS_INT32;
    uint32_t keySize = KEY_SIZE_INT32;
    uint32_t sortPasses = SORT_PASSES_INT32;

    // 4. 空 Tensor 检查
    if (totalNum == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_INT32), hasInverse);
        size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
```

判别方法：UT 中对 `return_inverse=false` 的用例断言 `tiling->hasInverse == 0` 与 `tilingKey` 期望值——本类指针误用会在"属性存在但为 false"的用例上暴露（属性缺失时另一个失败模式是解引用前未判空）。

## 设计决策

- 修复采用"判空 + 解引用"而不是把 `GetBool` 包一层返回值封装：`OP_CHECK_NULL_WITH_CONTEXT` 是工程既有判空宏，保持风格一致且失败时带 context 报错位置；封装函数会掩盖"属性可能缺失"这一语义。
- 空 Tensor 路径同步改用 `hasInverse` 变量：原硬编码 `0` 与非空路径行为不一致（空输入 + return_inverse=true 会错选无 inverse 模板），同一次修复闭合。

## 系统信息

- 硬件/软件：ascend950 算子工程、CANN 9.1.0（`include/exe_graph/runtime/runtime_attrs.h`：`GetAttrPointer<T>`/`GetBool` 返回 `const T*`）、x86_64 宿主机 gtest UT。
- 相关文件：`op_host/arch35/bench_no_harness_unique_tiling.cpp`（修复点）、`tests/ut/op_host/test_bench_no_harness_unique_tiling.cpp`（暴露问题的断言）、`issues/issue_20260722_getbool_pointer_implicit_conversion_1.md`（问题记录）。

## 验证信号

以下信号来自该算子 op_host UT 的两轮运行：修复前 noinv 用例断言失败，修复后全部通过。

- 修复前：`tiling_int32_noinv_100 / noinv_1000 / noinv_33` 三个用例断言失败，`tiling->hasInverse` 实际 1、期望 0，tilingKey=259（来源平台：950 算子工程的 op_host 代码，UT 于 x86_64 宿主机执行；证据等级：tool_observed）。
- 根因核对：`runtime_attrs.h` 中 `GetBool` 经 `GetAttrPointer<T>` 返回 `const T*`（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 13/13 通过——`return_inverse=false` 用例 `hasInverse=0`、`tilingKey=3`；`return_inverse=true` 用例 `hasInverse=1`、`tilingKey=259`（来源平台：950 算子工程的 op_host 代码，UT 于 x86_64 宿主机执行；证据等级：tool_observed）。

## 失效边界

- 本卡针对 `GetBool`；`RuntimeAttrs` 的其他取值接口（如 GetInt64）同样是 `const T*` 返回，未逐一实测但机制相同（未验证，机制推断可迁移）。
- 属性确实缺失时（GetBool 返回 nullptr），修复后的代码走 `OP_CHECK_NULL_WITH_CONTEXT` 报错返回——"缺失时默认 false"的策略需要另行显式实现，不在本卡范围。
- UT faker 的属性注入行为与真实框架的属性序列化若有差异，本卡未覆盖。

**未决项**：

- 无（修复闭环，UT 全绿）。
## 坏实践

- `bool v = attrs->GetBool(i);`——一行就埋下"恒真"陷阱，编译器不告警。
- 空 Tensor / 退化路径硬编码模板参数字面量（`ASCENDC_TPL_SEL_PARAM(context, dt, 0)`），与非空路径的真实变量脱节。

## 误诊

- 最初怀疑 UT faker 没把 `return_inverse` 属性设置进 context（"the attribute is not being set correctly in the faker"）——检查 faker 的 `VT_BOOL` 注入路径后排除：属性已正确写入，真因在 tiling 侧把属性指针当布尔值用。这类误诊方向（怪测试脚手架）在 faker 框架里很常见，先用最小用例区分"数据没进去"与"读法错了"再动手。
