---
type: "Runbook"
title: aclCreateTensorList 是两参签名，旧多参写法编译失败
description: 旧测试代码用 aclCreateTensorList(ACL_FLOAT, len, ...) 多参签名编译报 cannot convert 'aclDataType'；CANN 9.1.0 实际为 (const aclTensor* const*, uint64_t) 两参，按安装头文件修正。
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# aclCreateTensorList 是两参签名，旧多参写法编译失败

## 现象

为一个自定义算子恢复旧版 C++ ST 测试代码后重新编译（`cmake .. && make`，链接 CANN 9.1.0 自定义算子包），在创建 TensorList 的两处报同一种模板推导错误：

```text
test_aclnn_bench_no_harness_foreach_norm.cpp:399:53: error: cannot convert ‘aclDataType’ to ‘const aclTensor* const*’
test_aclnn_bench_no_harness_foreach_norm.cpp:451:53: error: cannot convert ‘aclDataType’ to ‘const aclTensor* const*’
```

对应代码是旧写法（第一个参数传 dtype 枚举、第二个传长度，后续还有更多参数）：

```cpp
    aclTensorList* x_acl_list = aclCreateTensorList(ACL_FLOAT, static_cast<int32_t>(list_len),
                                                    ...);
    aclTensorList* y_acl_list = aclCreateTensorList(ACL_FLOAT, static_cast<int32_t>(list_len),
                                                    ...);
```

## 适用场景

- 工程形态：aclnn C++ 原生测试 / op_api 侧代码，需要构造 `aclTensorList*` 输入。
- API 路径：`aclCreateTensorList`（声明在 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/acl_meta.h` 与 `opdev/common_types.h`）。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：platform_wide（同一 CANN 头文件契约下的通用 API 用法）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：其他 CANN 版本的签名以各自安装头文件为准；本卡只锚定 9.1.0。

## 根因

测试代码沿用了过时的 `aclCreateTensorList` 多参签名（首参传 `aclDataType`），而 CANN 9.1.0 的实际签名只有两个参数（张量指针数组 + 个数），首参不收 `aclDataType`，模板推导报 "cannot convert 'aclDataType' to 'const aclTensor* const*'"。

## 解决配方

先在安装目录里核对真实签名（这一步是通用判别方法）：

```bash
grep -rn "aclCreateTensorList" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/ | head
# .../cann-9.1.0/x86_64-linux/include/aclnn/opdev/common_types.h:271:    friend aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size);
# .../cann-9.1.0/x86_64-linux/include/aclnn/acl_meta.h:45:ACL_FUNC_VISIBILITY aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size);
```

修复前（过时多参签名）：

```cpp
    aclTensorList* x_acl_list = aclCreateTensorList(ACL_FLOAT, static_cast<int32_t>(list_len),
                                                    /* 旧签名多余参数 */);
```

修复后（两参签名：先 `aclCreateTensor` 建好每个张量，再收进 list）：

```cpp
    aclTensorList* x_acl_list = aclCreateTensorList(x_tensor_list.data(),
                                                    static_cast<uint64_t>(list_len));
    aclTensorList* y_acl_list = aclCreateTensorList(y_tensor_list.data(),
                                                    static_cast<uint64_t>(list_len));
```

## 设计决策

- 修复方式选"对照安装头文件改代码"而不是"找兼容宏"：ACLNN C++ 接口以安装包头文件为唯一事实源，`opdev/common_types.h` 的 friend 声明与 `acl_meta.h` 的 `ACL_FUNC_VISIBILITY` 声明可直接确认签名。

## 预防

- 写/恢复 aclnn 测试代码遇到 `cannot convert ... to 'const aclTensor* const*'` 一类模板推导错误时，第一步先 `grep` 安装头文件确认当前签名，不要按记忆或旧模板补参数。
- 从旧版本迁移测试工程时，把 `aclCreate*` 系列签名全部对照 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/` 过一遍。

## 验证信号

来自该算子 ST 工程修复会话的实测：

- 修复前：make 报两处 `error: cannot convert ‘aclDataType’ to ‘const aclTensor* const*’`（399:53 / 451:53），链接失败（来源平台：950；证据等级：tool_observed）。
- 核对头文件：`common_types.h:271` 与 `acl_meta.h:45` 均声明两参签名 `aclCreateTensorList(const aclTensor* const* value, uint64_t size)`（来源平台：950；证据等级：tool_observed）。
- 修复后：`[100%] Linking CXX executable test_aclnn_bench_no_harness_foreach_norm` 编译链接通过；随后 Real 模式 10/10 用例在 NPU 上运行通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅锚定 CANN 9.1.0 签名；后续版本签名若再演化，以安装头文件为准（未验证，推断可迁移）。
- 本卡只覆盖签名修正；同文件里头文件名大小写问题（`aclnn<Op>.h` PascalCase vs 安装的 `aclnn_<op>.h` snake_case）是同场景另一处独立小修，报错信息会直接给出缺失文件名，按安装目录实际文件名改 include 即可，不在此展开。

**未决项**：

- 未追溯旧签名出自哪个 CANN 版本（对跨版本迁移的完整对照表无需求，未尝试）。
## 坏实践

- 按旧模板/旧仓库记忆直接写 `aclCreateTensorList(ACL_FLOAT, ...)`，不查安装头文件。
- 用 `reinterpret_cast` 硬转参数绕过编译错误（会掩盖真实签名不匹配）。

## 误诊

- 无。
