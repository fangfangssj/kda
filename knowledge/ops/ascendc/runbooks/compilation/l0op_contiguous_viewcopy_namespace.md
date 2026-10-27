---
type: "Runbook"
title: ViewCopy/Contiguous 未声明：在 l0op 命名空间
description: "op_api 的 aclnn 实现已 include aclnn_kernels/contiguous.h 且 using namespace op，直接调用 ViewCopy/Contiguous 仍报 \"was not declared in this scope\"；这两个工具函数声明在 l0op 命名空间，必须写 l0op::ViewCopy / l0op::Contiguous。"
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:22:16Z'
updated_at: '2026-09-16T10:22:16Z'
platforms: ["950"]
---

# ViewCopy/Contiguous 未声明：在 l0op 命名空间

## 现象

给自定义 aclnn 算子 bench_no_harness_cummin 改 op_api 实现时，两次独立改动里调用 `aclnn_kernels/contiguous.h` 提供的张量工具函数，编译（`bash build.sh --soc=ascend950 -j8`）均报同名错误：

第一次，在 GetWorkspaceSize 里加"executor 内部输出张量拷回用户输出张量"时：

```
op_api/aclnn_bench_no_harness_cummin.cpp:150:15: error: 'ViewCopy' was not declared in this scope; did you mean 'l0op::ViewCopy'?
  150 |     CHECK_RET(ViewCopy(valuesOut, values, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
      |               ^~~~~~~~
op_api/aclnn_bench_no_harness_cummin.cpp:151:15: error: 'ViewCopy' was not declared in this scope; did you mean 'l0op::ViewCopy'?
```

第二次（另一处独立改动），在 GetWorkspaceSize 里加"非连续输入显式连续化"时：

```
op_api/aclnn_bench_no_harness_cummin.cpp:143:40: error: 'Contiguous' was not declared in this scope; did you mean 'l0op::Contiguous'?
  143 |     const aclTensor* inputContiguous = Contiguous(input, uniqueExecutor.get());
      |                                        ^~~~~~~~~~
      |                                        l0op::Contiguous
In file included from op_api/aclnn_bench_no_harness_cummin.cpp:18:
x86_64-linux/include/aclnn_kernels/contiguous.h:59:18: note: 'l0op::Contiguous' declared here
   59 | const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
```

两次的头文件都已 `#include "aclnn_kernels/contiguous.h"`，且文件顶部有 `using namespace op;`——直觉上"头文件已包含、又是自由函数"就会裸调函数名，但这两个函数不在 `op` 命名空间里。

## 适用场景

- 工程形态：aclnn 自定义算子 op_api 层开发/修改（L2 GetWorkspaceSize 中做输入连续化或内部张量到用户张量的拷回）。
- API 路径：`aclnn_kernels/contiguous.h` 的 `const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor)` 与 `const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)`（CANN 9.1.0 `x86_64-linux/include/aclnn_kernels/contiguous.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（编译目标 `--soc=ascend950`，编译产物安装后上板回归通过）。
- 适用粒度：platform_wide——所有用该头文件的 aclnn op_api 代码，与算子无关。
- 路线优先级：recommended_fastpath（一行命名空间限定的直接修法）。
- 明确不适用 / 未验证场景：其他 aclnn_kernels 头文件中的符号未逐一核对命名空间（同文件 `ContiguousParam` 等类型同理需注意）；C++ ADL（参数依赖查找）不适用于此——参数类型 `aclTensor*/aclOpExecutor*` 不在 l0op 内，ADL 找不到。

## 根因

（单句归因）`aclnn_kernels/contiguous.h` 把 `Contiguous`/`ViewCopy` 声明在 `l0op` 命名空间，而 aclnn 实现文件常见的 `using namespace op;` 只引入 `op` 命名空间，裸调函数名自然解析失败。

## 解决配方

修复前（裸调，编译失败）：

```cpp
    // GetWorkspaceSize 内（两处独立出现）
    const aclTensor* inputContiguous = Contiguous(input, uniqueExecutor.get());
    ...
    CHECK_RET(ViewCopy(valuesOut, values, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopy(indicesOut, indices, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

修复后（加 l0op:: 限定）：

```cpp
    const aclTensor* inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    ...
    CHECK_RET(l0op::ViewCopy(valuesOut, values, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(indicesOut, indices, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

只改命名空间限定，无其他改动；重新 `bash build.sh --soc=ascend950 -j8` 编译通过并成功打包安装（`custom_opp_ubuntu_x86_64.run`）。

## 设计决策

- 选 `l0op::` 显式限定而不是 `using namespace l0op;`：限定写法作用域最小、意图明确，避免把整个 l0op 命名空间拉进翻译单元引入新的解析歧义。
- gcc 的 "did you mean" 提示在这两次都直接给出了正确答案——修法本身不曲折，价值在于把这个命名空间事实沉淀下来，写 aclnn 代码时不再依赖编译器纠错。

## 系统信息

- Ascend950PR，CANN 9.1.0；头文件 `x86_64-linux/include/aclnn_kernels/contiguous.h`（`ContiguousParam` 结构体、`Contiguous`、`ViewCopy` 等声明，均位于 `l0op` 命名空间）。
- 相关文件：`op_api/aclnn_bench_no_harness_cummin.cpp`（两次调用点）。

## 预防

- 在 aclnn op_api 代码里调用 `aclnn_kernels/` 头文件的自由函数时，默认带 `l0op::` 前缀；review 时把"裸调 Contiguous/ViewCopy"当作已知坑直接指出来。
- 若编译报 `was not declared in this scope` 且头文件确已 include，先看错误提示的 `note: ... declared here` 指向的命名空间，再决定是限定还是 using。

## 验证信号

以下信号来自同一次算子联调会话中两次独立的编译失败-修复闭环。

- 失败：`error: 'ViewCopy' was not declared in this scope; did you mean 'l0op::ViewCopy'?`（op_api 编译，gmake Error 2）（来源平台：950；证据等级：tool_observed）。
- 失败：`error: 'Contiguous' was not declared in this scope; did you mean 'l0op::Contiguous'?`，note 指向 `aclnn_kernels/contiguous.h:59` 的 `l0op::Contiguous` 声明（来源平台：950；证据等级：tool_observed）。
- 修复后：`bash build.sh --soc=ascend950 -j8` 编译通过、CPack 生成 `custom_opp_ubuntu_x86_64.run` 并成功安装（ViewCopy 修复后安装成功；Contiguous 修复后编译+打包成功）（来源平台：950；证据等级：tool_observed）。
- 下游回归：Contiguous 修复后的算子包经 NPU 边界与全量 ST 验证通过（非连续输入 MERE=0，分级回归 L0 级 30/30、L1 级 553 PASS）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 命名空间归属以所用 CANN 版本的头文件为准（本卡实测 9.1.0）；若未来版本把这两个函数迁入 `op` 命名空间，`l0op::` 限定会编译失败——届时报错同样直白，反向修一行即可。
- 本卡只覆盖 `contiguous.h` 的 `Contiguous`/`ViewCopy`；`aclnn_kernels` 其他头文件的符号命名空间未逐一核对（未验证）。

**未决项**：

- 无（编译类闭环已完整）。可顺带核对 `aclnn_kernels` 目录其他工具头文件的命名空间归属，形成一张完整的 aclnn 工具函数命名空间清单（未尝试）。
## 坏实践

- 在已 `using namespace op;` 的文件里凭直觉裸调 `aclnn_kernels` 头文件的自由函数，然后靠编译器 "did you mean" 纠错——能过但浪费一轮编译（自定义算子全量编译一次数分钟）。
- 用 `using namespace l0op;` 全局引入来"一劳永逸"——增加名字冲突面。

## 误诊

- 无明显误诊轮次：两次都是编译错误直出、note 指向声明、一行限定修复。此卡价值是高频小坑的事实沉淀，而非曲折定位过程。
