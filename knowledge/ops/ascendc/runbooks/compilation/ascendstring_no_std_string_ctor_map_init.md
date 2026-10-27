---
type: "Runbook"
title: "AscendString 不能从 std::string 隐式构造"
description: "GE 初始化选项 std::map<ge::AscendString, ge::AscendString> 的初始化列表里放 std::to_string(deviceId) 编译报 could not convert brace-enclosed initializer list；AscendString 只接受 const char*，先落局部变量再 .c_str() 构造。"
tags: [task.implementation, topic.graph_execution, topic.implementation.api_layer]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# AscendString 不能从 std::string 隐式构造

## 现象

给自定义算子写 GE IR 图模式调用示例（Ascend950PR，CANN 9.2.0），GE 初始化需要动态传入设备号，直接在 options map 初始化列表里拼接：

```cpp
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", std::to_string(deviceId)},
        {"ge.graphRunMode", "1"}
    };
```

编译报错（报错指向整个 brace-init-list，容易看偏）：

```
test_geir_bench_no_harness_depthwise_conv_2d.cpp:349:5: error: could not convert '{{"ge.exec.deviceId", std::__cxx11::to_string(int)()}, {"ge.graphRunMode", "1"}}' from '<brace-enclosed initializer list>' to 'std::map<ge::AscendString, ge::AscendString>'
  349 |     };
      |     ^
```

同一编译轮还伴随 `aclInit` 未声明类错误（示例用了 ACL 接口但没包含 `<acl/acl.h>`）：

```
test_geir_bench_no_harness_depthwise_conv_2d.cpp:358:19: error: 'aclInit' was not declared in this scope
  358 |     auto aclRet = aclInit(nullptr);
```

## 适用场景

- 工程形态：GE IR 图模式程序/示例开发中构造 `GEInitialize` 的 `std::map<ge::AscendString, ge::AscendString>` 全局选项。
- API 路径：`graph/ascend_string.h` 的 `ge::AscendString`（仅 `const char*` 构造）；GE 选项键 `ge.exec.deviceId`/`ge.graphRunMode`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（与 frontmatter `platforms` 一致；验证等级为编译级）。
- 适用粒度：platform_wide——GE API 类型契约。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：字面量键值（`{"ge.graphRunMode", "1"}`）不受影响，可正常隐式转换。

## 根因

`ge::AscendString` 只有从 `const char*` 的构造，不能从 `std::to_string(...)` 返回的 `std::string` 临时对象隐式转换，放在 map 初始化列表里整组 brace-init-list 转换失败。

## 解决配方

修复前：

```cpp
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", std::to_string(deviceId)},
        {"ge.graphRunMode", "1"}
    };
```

修复后（先落局部 `std::string`，再用 `.c_str()` 显式构造；同轮补上 `#include "acl/acl.h"`）：

```cpp
    LOG_PRINT("%s - INFO: Start to initialize ge\n", GetTime().c_str());
    std::string deviceStr = std::to_string(deviceId);
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", AscendString(deviceStr.c_str())},
        {"ge.graphRunMode", "1"}
    };
```

```cpp
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "acl/acl.h"
```

## 设计决策

- 没有选"deviceId 固定为字符串字面量 "0""（add_example 的写法）：示例需要支持命令行传设备号，动态值必须经局部变量中转；局部 `std::string` 生命周期覆盖 map 使用期，`c_str()` 指针安全。

## 系统信息

- 工具链：CANN 9.2.0，GCC 13.3.0，cmake。
- 目标平台：Ascend950PR（arch35）。
- 相关文件：`examples/test_geir_<op>.cpp`（GE 初始化段与 include 列表）。

## 预防

- 给 GE 选项 map 传动态拼出的值时，一律先落 `std::string` 局部变量再 `AscendString(str.c_str())`，不要把 `std::to_string(...)` 直接放进初始化列表。

## 验证信号

问题与验证来自该算子 GE IR 示例编译会话。

- 修复前编译输出：`error: could not convert '{{"ge.exec.deviceId", std::__cxx11::to_string(int)()}, ...}' from '<brace-enclosed initializer list>' to 'std::map<ge::AscendString, ge::AscendString>'`（来源平台：950；证据等级：tool_observed）。
- 修复后：编译通过，进入运行阶段（AddGraph 运行期失败属另一独立问题）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- C++ 层类型转换契约，CANN 版本间推断稳定（未跨版本验证）。
- 若 map 值改用字面量则不触发；只有动态拼出的 `std::string` 值需要中转。

**未决项**：

- 无。
## 坏实践

- 看到 "could not convert brace-enclosed initializer list" 就去改 map 的模板参数或整体换 `std::map<std::string, std::string>`——GE API 只收 `AscendString` map，正确做法是让每个值可构造。

## 误诊

- 无。
