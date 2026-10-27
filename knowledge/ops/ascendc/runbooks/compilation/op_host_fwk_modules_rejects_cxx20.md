---
type: "Runbook"
title: "op_host fwk 编译链拒 C++20 写法"
description: "op_host TilingFunc 源码经 CANN fwk_modules 编译链以低于 C++20 的语言标准编译：std::sort lambda 形参 const auto& 报 use of 'auto' in lambda parameter declaration only available，for(auto& [a,b]) 结构化绑定连锁 no match for call；须改显式 std::pair 类型与下标访问。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# op_host fwk 编译链拒 C++20 写法

## 现象

在为 MoE 分组矩阵乘算子（Ascend950PR / arch35 / CANN 9.1.0）的 op_host/arch35 TilingFunc 里实现"按组行数贪心均衡分配到多核"的调度函数时，用了现代 C++ 写法：

```cpp
    // 按 m_i 降序排序
    std::sort(nonEmpty.begin(), nonEmpty.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    ...
    // 贪心分配：每组分配给累计负载最小的核
    for (const auto& [mi, g] : nonEmpty) {
```

`bash build.sh --soc=ascend950 -j1` 时该文件被 CANN 的 `npu_op_code_gen` 编译链（`compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake`）编译，报错原文：

```text
error: use of 'auto' in lambda parameter declaration only available
error: request for member 'first' in 'a', which is of non-class
error: request for member 'first' in 'b', which is of non-class
/usr/include/c++/13/bits/predefined_ops.h:158:30: error: no match for call
CMake Error at .../fwk_modules/func.cmake:732 (message):
```

宿主机 g++ 本身是 13.3.0（支持 C++20），但 op_host 的这条编译链没有开到对应语言标准，`auto` 形参与结构化绑定都无法通过。

## 适用场景

- 工程形态：registry-invoke 自定义算子 op_host 层 TilingFunc（经 `npu_op_code_gen` / fwk_modules 编译）。
- API 路径：`compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake` 驱动的 host 源码编译。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 9.1.0 fwk 编译链的语言标准限制，与具体算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：op_api / tests 等走普通 g++ + CMAKE_CXX_STANDARD 的目标不受此限（本工程 ST 测试代码同风格写法编译通过）；能否通过给 npu_op_code_gen 追加 `-std=` 选项放开未验证。

## 根因

op_host TilingFunc 源码由 CANN fwk_modules 编译链以低于 C++20 的语言标准编译：lambda 形参 `const auto&`（generic lambda，C++14 起在部分实现受限、此链下直接拒绝）与结构化绑定 `const auto& [mi, g]`（C++17 特性）都不可用；随后 `a.first` 在 auto 推导失败后变成非类类型访问、`std::sort` 内部 `predefined_ops.h` 报 no match for call，均是同一根因的连锁。

## 解决配方

修复前（GreedySchedule 内的排序与遍历）：

```cpp
    // 按 m_i 降序排序
    std::sort(nonEmpty.begin(), nonEmpty.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int64_t usedCoreNum = std::min(coreNum, static_cast<int64_t>(nonEmpty.size()));
    std::vector<int64_t> coreLoad(usedCoreNum, 0);  // 各核累计负载

    // 贪心分配：每组分配给累计负载最小的核
    for (const auto& [mi, g] : nonEmpty) {
        // 找到负载最小的核
        int64_t minCore = 0;
        for (int64_t c = 1; c < usedCoreNum; c++) {
            if (coreLoad[c] < coreLoad[minCore]) {
                minCore = c;
            }
        }
        coreAssign[g] = minCore;
        coreLoad[minCore] += mi;
    }
```

修复后（显式 std::pair 类型 + 下标访问）：

```cpp
    // 按 m_i 降序排序
    std::sort(nonEmpty.begin(), nonEmpty.end(),
              [](const std::pair<int64_t, int64_t>& a,
                  const std::pair<int64_t, int64_t>& b) { return a.first > b.first; });

    int64_t usedCoreNum = std::min(coreNum, static_cast<int64_t>(nonEmpty.size()));
    std::vector<int64_t> coreLoad(usedCoreNum, 0);  // 各核累计负载

    // 贪心分配：每组分配给累计负载最小的核
    for (size_t i = 0; i < nonEmpty.size(); i++) {
        int64_t mi = nonEmpty[i].first;
        int64_t g = nonEmpty[i].second;
        // 找到负载最小的核
        int64_t minCore = 0;
        for (int64_t c = 1; c < usedCoreNum; c++) {
            if (coreLoad[c] < coreLoad[minCore]) {
                minCore = c;
            }
        }
        coreAssign[g] = minCore;
        coreLoad[minCore] += mi;
    }
```

## 设计决策

- 选择"降级写法"而不是给编译链加 `-std=c++20`：fwk_modules 是 CANN 内部编译设施，向 `npu_op_code_gen` 私传语言标准选项的可行性未知且有升级兼容风险；显式类型写法零成本、可移植。
- 保留 `std::pair<int64_t,int64_t>`（first=m_i，second=组号）的降序排序 + 贪心最小负载核分配：这是多组任务到多核的经典 LPT（最长处理时间优先）近似，写法调整不影响调度语义（后续 UT 验证变长分组时 coreAssign 的均衡结果一致）。

## 预防

- op_host TilingFunc 源码一律按 C++14 以前的保守子集写：lambda 形参写显式类型、循环变量用下标或迭代器、不用结构化绑定。
- 提交前可在本地用 `g++ -std=c++14 -fsyntax-only` 快速自检 tiling 源码的语言标准兼容性。

## 验证信号

来源说明：该算子骨架的编译闭环（Ascend950PR + CANN 9.1.0）；写法替换后重新编译，auto/结构化绑定相关报错消失。

- 修复前：`error: use of 'auto' in lambda parameter declaration only available`、`request for member 'first' in 'a'`、`predefined_ops.h: error: no match for call`（来源平台：950；证据等级：tool_observed）。
- 修复后：上述报错串不再出现，host 编译推进到后续类型解析阶段（来源平台：950；证据等级：tool_observed）。
- 全量回归：连同后续修复，`build.sh --soc=ascend950 -j8` 生成 custom_opp_ubuntu_x86_64.run 与 3 个 kernel .o（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 语言标准限制随 CANN 版本可能放开；升级后可回验 `-std` 与新写法（未验证）。
- fwk_modules 编译链的确切 `-std` 值未在轨迹中探明（仅从报错推断低于 C++20）。
- 本卡只覆盖 op_host 的 fwk 编译链；kernel 侧（ccec）与 op_api/tests 侧（普通 g++）另有标准。

**未决项**：

- fwk_modules 编译链的确切语言标准值未探明（不影响修复结论）。
## 坏实践

- 在 op_host tiling 代码里顺手写 `[](const auto& a, ...)` / `for (auto& [k, v] : ...)`——本地 g++ 能过，进 fwk 编译链即挂。

## 误诊

- 无。
