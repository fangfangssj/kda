---
type: "Runbook"
title: "gert::Shape 无 ToString，日志用 GetDimNum"
description: "op_api 的参数校验日志对 aclTensor::GetViewShape() 返回值调 ToString() 编译报 'gert::Shape has no member named ToString'；CANN 9.1.0 的 Shape 只有 GetDimNum()/GetDim() 等访问器，日志需自行拼接。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# gert::Shape 无 ToString，日志用 GetDimNum

## 现象

在编写 Mish 激活算子（Ascend950PR，CANN 9.1.0）的 aclnn L2 接口实现 `op_api/aclnn_*.cpp` 时，参数校验函数 CheckShape 里对 shape 不一致的报错日志写成 `x->GetViewShape().ToString().c_str()`（想直接打印两个张量的形状）。host 三个编译目标中 cust_optiling、cust_op_proto 通过，cust_opapi 编译失败：

```
op_api/aclnn_bench_no_harness_mish.cpp:85:78: error: 'const op::Shape' {aka 'const struct gert::Shape'} has no member named 'ToString'
   85 |                      x->GetViewShape().ToString().c_str(), y->GetViewShape().ToString().c_str()),
      |                                                                              ^~~~~~~~
```

（同函数对 x 的调用在第 83-85 行先报，报错沿 OP_LOGE → REPORT_ERROR_MESSAGE → OP_CHECK 宏展开链给出。）

## 适用场景

- 工程形态：算子仓工程 op_api（aclnn L2 接口）参数校验与日志。
- API 路径：`aclTensor::GetViewShape()`（返回 `gert::Shape`/`op::Shape`）、OP_CHECK/OP_LOGE 宏。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 9.x op_api 层）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他提供 ToString 的类型（如 std::string）不适用；后续 CANN 版本若补齐该方法则本卡过期（未验证）。

## 根因

CANN 9.1.0 中 `GetViewShape()` 返回的 `gert::Shape`（别名 `op::Shape`）没有 `ToString()` 成员，只能用 `GetDimNum()`、`GetDim(i)` 等访问器自行拼日志。这是"以为容器类自带字符串化方法"的常见假设在 aclnn host 层落空。

## 解决配方

修复前（CheckShape 的 shape 一致性校验，OP_CHECK 第一个参数为真时打日志并执行返回动作）：

```cpp
    // 输出 shape 必须与输入一致（不支持 broadcast）
    OP_CHECK(x->GetViewShape() != y->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Shape mismatch: x=%s, y=%s. Output shape must equal input shape.",
                     x->GetViewShape().ToString().c_str(), y->GetViewShape().ToString().c_str()),
             return false);
```

修复后（用 GetDimNum() 打维数，`%zu` 格式）：

```cpp
    // 输出 shape 必须与输入一致（不支持 broadcast）
    OP_CHECK(x->GetViewShape() != y->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Shape mismatch: x_dim=%zu, y_dim=%zu. Output shape must equal input shape.",
                     x->GetViewShape().GetDimNum(), y->GetViewShape().GetDimNum()),
             return false);
```

如需完整形状，可逐维拼接：`for (size_t i = 0; i < shape.GetDimNum(); i++)` 取 `shape.GetDim(i)`（本例未采用，维数已足够定位）。

## 设计决策

- 选择只打 GetDimNum 而非逐维拼接：报错场景（shape 不一致）下维数差异通常就是第一信号，逐维拼接会让日志宏参数列表更冗长；需要时可扩展。
- 当前方案路线档位：recommended_fastpath。

## 预防

- op_api 侧写日志前先确认类型可用成员；Shape 一律按 GetDimNum/GetDim 访问器使用。
- 这类错误发生在 OP_LOGE 宏展开内部，报错行号指向宏参数行，定位时直接看 error 行的源码片段即可。

## 验证信号

以下信号来自该 Mish 算子工程的编译联调会话：cust_opapi 编译失败 → 换 GetDimNum → 重编。

- 修复前：`aclnn_*.cpp:85:78: error: 'const op::Shape' {aka 'const struct gert::Shape'} has no member named 'ToString'`，cust_opapi 目标失败（来源平台：950；证据等级：tool_observed）。
- 修复后：cust_opapi 编译通过，host 三库（cust_optiling/cust_opapi/cust_op_proto）全部 Built target，编译进入 kernel 阶段（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 绑定 CANN 9.1.0 的 gert::Shape 定义；若后续版本新增字符串化接口（未验证），按新接口为准（未验证，推断可迁移）。

**未决项**：

- 无。
## 坏实践

- 对不确定成员的框架类型直接链式 `.ToString().c_str()`；先编译最小片段或查头文件确认。

## 误诊

- 无（报错即根因）。
