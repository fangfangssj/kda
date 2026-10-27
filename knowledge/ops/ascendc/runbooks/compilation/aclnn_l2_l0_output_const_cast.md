---
type: "Runbook"
title: "L2 传 L0 输出张量需 const_cast"
description: "aclnn L2 API 中经 Contiguous 变换得到的输出张量是 const aclTensor*，而自定义 L0 API 的输出参数声明为非 const aclTensor*，跨层传输出须 const_cast，否则 invalid conversion 编译错；输入保持 const 不 cast。"
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# L2 传 L0 输出张量需 const_cast

## 现象

BenchNoHarnessGru 的 aclnn 两段式 L2 API（op_api/aclnn_bench_no_harness_gru.cpp）在 GetWorkspaceSize 阶段对输出 y/hn 做非连续 Tensor 转换（Contiguous），随后把变换结果传给自定义 L0 API：

```cpp
auto l0Ret = l0op::BenchNoHarnessGru(
    xContiguous, wIhContiguous, wHhContiguous,
    inputSize, hiddenSize, numLayers,
    bias, batchFirst, dropout, bidirectional,
    biasIhTensor, biasHhTensor, h0Contiguous,
    yContiguous, hnContiguous, uniqueExecutor.get());
```

`bash build.sh --soc=ascend950 -j8`（CANN 9.2.0）编译报 2 处：

```text
op_api/aclnn_bench_no_harness_gru.cpp:223:9: error: invalid conversion from 'const aclTensor*' to 'aclTensor*' [-fpermissive]
op_api/aclnn_bench_no_harness_gru.cpp:223:22: error: invalid conversion from 'const aclTensor*' to 'aclTensor*' [-fpermissive]
```

## 适用场景

- 工程形态：自定义算子仓开发，op_api 层两段式 aclnn 接口（L2 GetWorkspaceSize/执行 → 自定义 L0 算子实现），输出张量走 Contiguous 归一化。
- API 路径：`aclnn_kernels/contiguous.h` 的 Contiguous 变换（返回 const aclTensor*）、自定义 L0 API（`l0op::<Op>` 声明，输出为 aclTensor*）。
- 验证过的平台与 CANN 版本：Ascend950（build.sh --soc=ascend950 编译流水线），CANN 9.2.0（与 frontmatter 一致）。
- 适用粒度：platform_wide（aclnn 分层约定，与算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：输入张量（L0 声明为 const aclTensor*，无需 cast）；L0 输出参数若声明为 const（个别官方算子形态）则不适用；运行期所有权语义（cast 后 Tensor 生命周期管理）未在本卡验证范围。

## 根因

aclnn 分层的 const 约定不对齐：L2 侧 Contiguous 变换与入参链路统一 `const aclTensor*`（只读视角）；而自定义 L0 API 的**输出**参数（y/hn 是被写入的出参）声明为非 const `aclTensor*`。跨层传输出张量时 const 属性需要显式卸掉，编译器不做隐式 const 剥离。

## 解决配方

修复前（直接传 const 输出）：

```cpp
// op_api/aclnn_bench_no_harness_gru.cpp（GetWorkspaceSize 内）
// 调用 L0 API
auto l0Ret = l0op::BenchNoHarnessGru(
    xContiguous, wIhContiguous, wHhContiguous,
    inputSize, hiddenSize, numLayers,
    bias, batchFirst, dropout, bidirectional,
    biasIhTensor, biasHhTensor, h0Contiguous,
    yContiguous, hnContiguous, uniqueExecutor.get());
```

修复后（输出张量显式 const_cast，输入保持 const）：

```cpp
// op_api/aclnn_bench_no_harness_gru.cpp（GetWorkspaceSize 内）
// 调用 L0 API
auto l0Ret = l0op::BenchNoHarnessGru(
    xContiguous, wIhContiguous, wHhContiguous,
    inputSize, hiddenSize, numLayers,
    bias, batchFirst, dropout, bidirectional,
    biasIhTensor, biasHhTensor, h0Contiguous,
    const_cast<aclTensor*>(yContiguous), const_cast<aclTensor*>(hnContiguous),
    uniqueExecutor.get());
```

## 设计决策

- cast 只加在输出（y/hn）：它们在 L0 内被写入，非 const 是语义正确；输入若也 cast 反而丢失只读保护。
- 放在调用点而非改变 L0 签名：L0 出参非 const 是 opdev 惯例（与官方 level2 算子 L0 实现一致），改签名偏离惯例。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写 L2→L0 调用时按参数角色分三类处理：输入原样传 const、输出 const_cast、executor/属性按 L0 签名。

## 验证信号

本次验证来自 BenchNoHarnessGru 编译循环（来源说明：同一算子迭代的多轮构建输出）：

- 修复前一轮构建 2 处 `invalid conversion from 'const aclTensor*' to 'aclTensor*'`（223 行）（来源平台：950 编译流水线；证据等级：tool_observed）。
- 输出参数加 const_cast 后，下一轮构建该 2 处错误消失（剩余错误转为 OP_CHECK 宏的独立问题）；再下一轮 op_api 层全部编译通过（构建错误进入 kernel 阶段）（来源平台：950 编译流水线；证据等级：tool_observed）。

## 失效边界

- 结论绑定 CANN 9.2.0 opdev/aclnn_kernels 的 const 约定；其他版本未验证，推断可迁移。
- const_cast 后 Tensor 的生命周期/所有权（Contiguous 新建 Tensor 的释放责任在 executor）本卡未验证，属 aclnn 框架通用语义。

**未决项**：

- 无。
## 坏实践

- 为了过编译把 L0 输出参数声明改成 const aclTensor*（丢失出参语义，且与 opdev 惯例冲突）。
- 对输入张量也无脑 const_cast。

## 误诊

- 无（错误信息直接指向 223 行的两个实参）。
