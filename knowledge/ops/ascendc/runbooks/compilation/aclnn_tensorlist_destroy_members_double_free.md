---
type: "Runbook"
title: "aclDestroyTensorList 连带销毁成员致 double free"
description: "CANN 的 aclDestroyTensorList 销毁列表容器时连带销毁列表内成员张量（容器接管成员所有权）；UT cleanup 先 DestroyTensorList 再逐个 DestroyTensor 会让同一批 aclTensor 被释放两次，用例 PASSED 后收尾报 free(): double free detected in tcache 2。清理只 DestroyTensor、跳过 List（测试态容忍容器泄漏）。"
tags: [task.testing, topic.memory, topic.implementation.memory_management, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:32:55Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# aclDestroyTensorList 连带销毁成员致 double free

## 现象

在 GRU 类自定义算子（Ascend950PR，CANN 9.2.0）的 op_api UT 里，fixture 用 `MakeStandardParams` 造参：每个 `aclTensor*` 记入 `p.ownedTensors`，再把 weight/bias 组成 `aclTensorList*` 记入 `p.ownedLists`。清理函数先销毁 List 再销毁 Tensor：

```cpp
void CleanupParams(StandardParams& p) {
    for (auto list : p.ownedLists) { TestTensorFactory::DestroyTensorList(list); }
    for (auto tensor : p.ownedTensors) { TestTensorFactory::DestroyTensor(tensor); }
}
```

跑单个用例时诡异的一幕：用例本身 PASSED，进程收尾时崩溃：

```
[ RUN      ] <gtest 用例名>
[       OK ] <gtest 用例名> (6 ms)
...
[  PASSED  ] 1 test.
free(): double free detected in tcache 2
/bin/bash: line 1: 664827 Aborted (core dumped) ./<op>_op_api_ut --gtest_filter=...
```

## 适用场景

- 工程形态：op_api UT（宿主侧直接调 aclnn GetWorkspaceSize 的参数校验用例）；也适用于任何"Tensor + TensorList 混合持有"的清理路径。
- API 路径：`aclCreateTensorList` / `aclDestroyTensorList` / `aclCreateTensor` / `aclDestroyTensor`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（编译目标 ascend950，宿主执行）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有 UT/工具代码里"Tensor + TensorList 混合持有"的清理路径。
- 路线优先级：recommended_fastpath（测试态）。
- 明确不适用 / 未验证场景：长生命周期服务/生产代码不可照抄"跳过 DestroyTensorList"（那是用受控泄漏换稳定）；aclCreateTensorList 是否在所有版本都接管成员所有权未逐一验证（本例 CANN 9.2.0 实测会）。

## 根因

CANN 的 `aclDestroyTensorList` 销毁列表容器时**连带销毁列表内的成员张量**（容器接管成员所有权）。清理顺序"先 List 后 Tensor"会让同一批 `aclTensor*` 被释放两次：List 析构一次、`DestroyTensor` 再一次 → glibc 检出 `double free detected in tcache 2`，进程 Aborted。用例逻辑本身（断言、API 调用）完全正常，崩溃发生在 gtest 收尾之后，极易被误判为被测代码问题。

## 解决配方

before（双重释放）：

```cpp
void CleanupParams(StandardParams& p) {
    for (auto list : p.ownedLists) { TestTensorFactory::DestroyTensorList(list); }
    for (auto tensor : p.ownedTensors) { TestTensorFactory::DestroyTensor(tensor); }
}
```

after（只销毁成员张量，跳过 List 容器；测试进程短生命周期，容器泄漏可接受）：

```cpp
void CleanupParams(StandardParams& p) {
    // Note: aclDestroyTensorList may internally destroy tensors in the list,
    // causing double-free if we also call DestroyTensor on them.
    // Solution: only destroy individual tensors, skip list destruction.
    // The TensorList containers leak (acceptable for tests).
    for (auto tensor : p.ownedTensors) { TestTensorFactory::DestroyTensor(tensor); }
}
```

生产/长驻代码的对照写法（单一所有权方，二选一销毁；PyTorch 适配层的 RAII 版本见 [PyTorch 接入 TensorList+scalar 算子 ST 工程卡](../validations/torch_tensorlist_scalar_st_adapter.md)）：

```cpp
    ~OpWorkspace() {
        // aclDestroyTensorList 会销毁其包含的 aclTensor，因此：
        // - 列表已创建时: 仅销毁列表 (列表负责销毁内部 tensor)
        // - 列表未创建时 (创建失败): 逐个销毁 tensor
        if (y_list)  { aclDestroyTensorList(y_list);  y_list = nullptr; }
        else { for (auto* t : y_tensors) { if (t) aclDestroyTensor(t); } }
    }
```

## 设计决策

- 选"只 DestroyTensor"而不是"只 DestroyTensorList"：成员张量并非全部入 List（x/y/hn 等单张量参数），统一走 ownedTensors 一个所有权列表，规则简单不易漏。
- 容器泄漏在 UT 场景（进程跑完即退）用 valgrind 抑制或忽略；生产代码的正确做法是理清单一所有权方，二选一销毁。

## 预防

- "TensorList 会吃掉成员"写进 UT 工具类的注释头；新 fixture 一律单一所有权列表。
- 看到 `double free detected in tcache 2` 且发生在 `[ PASSED ]` 之后，先查测试自身的清理路径，不要先怀疑被测 aclnn。

## 验证信号

问题与验证来自同一次 UT 会话（编译目标 ascend950，CANN 9.2.0，宿主执行）：

- 修改前：单测 `[ OK ]` 后进程 `free(): double free detected in tcache 2` + `Aborted (core dumped)`（来源平台：950；证据等级：tool_observed）。
- 修改后：全量 op_api UT 17/17 PASSED、进程正常退出，double free 消失（来源平台：950；证据等级：tool_observed）。
- 连同 op_host UT 的 run.sh 端到端汇总 PASS（31/31）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "跳过 DestroyTensorList"以受控泄漏为代价，仅适用于短生命周期测试进程；不可迁移到服务端长驻代码。
- 依赖 CANN 9.2.0 的容器所有权行为；版本升级后若语义变化（不再接管成员），该清理方式会转为泄漏成员，需回归。
- 同进程多次 aclnn 调用场景下"完全不销毁 tensor"是另一种清理契约（框架缓存 tensor 指针，销毁即悬垂，见 [同进程连跑 Segfault：勿销毁框架已缓存 tensor 卡](aclnn_cached_tensor_pointer_destroy_segfault.md)）；本卡是 UT 单进程内"销毁谁"的所有权问题，两者适用场景不同。

**未决项**：

- 无（本根因已闭环：17/17 通过且无收尾崩溃）。
## 坏实践

- 同一批张量既交给 TensorList 持有又留在自有清理列表里销毁。
- 把收尾 double free 当作被测算子的缺陷去排查（浪费在错误层面）。

## 误诊

- 无（本次定位直击清理路径；但同会话另一处段错误确实来自测试自身的悬垂指针清理问题，属另一根因，不并入本卡）。
