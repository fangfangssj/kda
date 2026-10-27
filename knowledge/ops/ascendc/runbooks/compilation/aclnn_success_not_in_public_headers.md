---
type: "Runbook"
title: "ACLNN_SUCCESS 公开头未导出须自定义宏"
description: "CANN 公开导出头（aclnn/*.h）只有 constexpr aclnnStatus OK = 0 与 ACL_SUCCESS，不导出 ACLNN_SUCCESS 宏；算子仓外测试代码判返回值须自行 #define ACLNN_SUCCESS 0 兜底。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api, topic.implementation]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# ACLNN_SUCCESS 公开头未导出须自定义宏

## 现象

同一个 ST 测试工程首次以 Real 模式（`cmake -DUSE_MOCK=OFF`）编译。测试代码包含了 `aclnn/aclnn_base.h`、`aclnn/acl_meta.h`、`acl/acl_rt.h`、`acl/acl_base.h` 与算子头文件，用 `if (status != ACLNN_SUCCESS)` 判断两段式接口返回值，编译失败，三个调用点同错：

```
tests/st/test_aclnn_<op>.cpp:785:19: error: 'ACLNN_SUCCESS' was not declared in this scope; did you mean 'ACL_SUCCESS'?
  785 |     if (status != ACLNN_SUCCESS) {
      |                   ^~~~~~~~~~~~~
      |                   ACL_SUCCESS
```

而算子仓内部的 op_api 代码里明明到处在用 `return ACLNN_SUCCESS;`——同样的宏，在算子工程里能编译，在外部测试工程里不行。

## 适用场景

- 工程形态：算子仓之外的 C++ 代码（独立 ST/示例/bench）调用 aclnn 两段式接口并判返回值。
- API 路径：`aclnn/aclnn_base.h`、`aclnn/acl_meta.h`（`constexpr aclnnStatus OK = 0;`）；算子内部经 opdev 头链拿到 `ACLNN_SUCCESS`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0；编译实测。
- 适用粒度：platform_wide（CANN 头文件导出面）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：算子仓内部代码（可包含 opdev 内部头）不受影响；直接用字面量 0 判等也可行，本卡给宏兜底方案。

## 根因

`ACLNN_SUCCESS` 定义在 CANN 内部（opdev）头文件里，公开导出头（`aclnn/*.h`）只有 `constexpr aclnnStatus OK = 0;` 与 `ACL_SUCCESS = 0`（acl 层），不导出 `ACLNN_SUCCESS` 宏——外部代码 include 公开头拿不到该符号。

## 判别方法

grep 公开头文件确认无导出（本会话实际执行）：

```bash
grep -rn "ACLNN_SUCCESS\|constexpr.*OK\|define ACLNN" \
  $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/aclnn_base.h \
  $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/acl_meta.h
# 仅命中：aclnn/acl_meta.h:34:constexpr aclnnStatus OK = 0;
```

## 解决配方

在 Real 模式 include 块后补充宏定义（修复后，verbatim）：

```cpp
#ifndef USE_MOCK
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "aclnn_<op>.h"

// aclnn 成功码（CANN 公开头文件未导出此宏，算子内部通过 opdev 获得，此处补充定义）
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif
#endif
```

修复前：include 块止于算子头文件，无宏定义，`status != ACLNN_SUCCESS` 三处编译报错。

## 预防

- 外部工程判 aclnn 返回值：要么 `#define ACLNN_SUCCESS 0` 兜底，要么直接 `status == 0` / `constexpr aclnnStatus OK`。
- 不要照抄算子仓内部源码的宏用法到仓外——两边的可见头文件面不同。

## 验证信号

问题与验证来自同一次外部 ST 工程首次 Real 模式编译闭环（Ascend950PR + CANN 9.2.0）：

- 修复前：三处 `error: 'ACLNN_SUCCESS' was not declared in this scope; did you mean 'ACL_SUCCESS'?`（来源平台：950 编译链；证据等级：tool_observed）。
- 修复后重编：编译错误清零，构建推进到链接阶段（来源平台：950 编译链；证据等级：tool_observed）。
- 最终 Real 可执行文件生成并 NPU 跑通两段式调用（GetWorkspaceSize status=0 / execute status=0）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 若未来 CANN 公开头导出该宏，`#ifndef` 兜底不冲突；直接硬编码 0 的写法在错误码语义变化时需复核（aclnn 惯例成功=0，稳定）。
- 公开头导出面基于 CANN 9.2.0 实测；其他版本未验证（未验证，推断同机制）。

**未决项**：

- 无。
## 坏实践

- 引入算子仓内部头（opdev/...）只为拿一个宏——污染外部工程依赖面且可能不可分发。

## 误诊

编译器提示 "did you mean 'ACL_SUCCESS'?"，直接换成 ACL_SUCCESS 也能工作（同为 0），但语义上 aclnnStatus 与 aclError 是两套类型；显式定义 ACLNN_SUCCESS 保持类型语义清晰。
