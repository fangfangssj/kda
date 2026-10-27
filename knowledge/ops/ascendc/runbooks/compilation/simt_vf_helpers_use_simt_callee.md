---
type: "Runbook"
title: "SIMT helper 误标 __simt_vf__ 导致编译失败"
description: "SIMT VF 入口调用的辅助函数也标成 __simt_vf__ 时，CANN 编译器拒绝调用；辅助函数须改用 __simt_callee__。"
tags: ["task.debugging", "topic.programming_model.execution_model.simt", "topic.implementation.api_layer.simt_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# SIMT helper 误标 __simt_vf__ 导致编译失败

## 现象

在 Ascend950 的 direct-launch Ascend C 工程中实现 `.asc` 核函数时，顶层 `__simt_vf__` 扫描入口需要调用一个用于 dtype 分支判断的函数模板。代码完整写入后，通过 CANN 9.0.1 的 CMake/make 构建入口编译，四种 dtype 的模板实例都在调用处失败，核心报错为：

```text
error: no matching function for call to 'is_nan_t'
note: candidate function not viable: simt_vf function can only call simt_callee function
9 errors generated.
```

当时代码并非缺少模板实参或函数声明；辅助函数与启动入口一样标成了 `__simt_vf__`。

## 适用场景

- 工程形态：Ascend C direct-launch 单文件 `.asc` 工程，CMake 编译并由 Host 侧启动 Kernel。
- API 路径：`asc_vf_call` 启动 `__simt_vf__` 入口，入口内部再调用设备辅助函数。
- 验证过的平台与版本：Ascend950PR_957b（平台注册值 `950`），CANN 9.0.1。
- 适用粒度：`platform_wide`；约束来自 SIMT VF 的设备调用约定，不依赖具体算子语义。
- 路线优先级：`recommended_fastpath`；被 `asc_vf_call` 启动的函数保留 `__simt_vf__`，被它调用的辅助函数标成 `__simt_callee__`。
- 不适用场景：Host 普通函数、`__aicore__` Kernel 壳函数，以及没有发生函数调用的内联表达式；其他 CANN 版本未验证。

## 根因

CANN SIMT 编译器把 `__simt_vf__` 识别为 VF 启动入口，而 VF 调用图中的普通设备辅助函数必须声明为 `__simt_callee__`，因此把被调用的 helper 也标成 `__simt_vf__` 会形成非法调用边。

## 解决配方

保留被 `asc_vf_call` 直接启动的顶层函数为 `__simt_vf__`，只把它调用的辅助函数改为 `__simt_callee__`；不要改函数模板参数、返回值或业务函数体。

### before

revision：轨迹内完整 `file_write` 源码快照（失败构建前）
path：`operators/<op>/<op>.asc`
symbol：`is_nan_t<T>`
line span：48–67
支撑结论：被调用 helper 使用 `__simt_vf__` 时触发 SIMT 调用约定编译错误。

```cpp
template <typename T>
__simt_vf__ inline bool is_nan_t(T x)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        return false; // int32 has no NaN
    } else if constexpr (std::is_same_v<T, half>) {
        uint16_t bits;
        __builtin_memcpy(&bits, &x, sizeof(T));
        return ((bits & 0x7C00u) == 0x7C00u) && ((bits & 0x03FFu) != 0u);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        uint16_t bits;
        __builtin_memcpy(&bits, &x, sizeof(T));
        return ((bits & 0x7F80u) == 0x7F80u) && ((bits & 0x007Fu) != 0u);
    } else { // float (fp32)
        uint32_t bits;
        __builtin_memcpy(&bits, &x, sizeof(T));
        return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
    }
}
```

### after

revision：轨迹内 `edit` 后源码快照（成功重建前）
path：`operators/<op>/<op>.asc`
symbol：`is_nan_t<T>`
line span：48–67
支撑结论：仅把 helper 属性改为 `__simt_callee__` 即恢复合法调用，函数体保持原样。

```cpp
template <typename T>
__simt_callee__ inline bool is_nan_t(T x)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        return false; // int32 has no NaN
    } else if constexpr (std::is_same_v<T, half>) {
        uint16_t bits;
        __builtin_memcpy(&bits, &x, sizeof(T));
        return ((bits & 0x7C00u) == 0x7C00u) && ((bits & 0x03FFu) != 0u);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        uint16_t bits;
        __builtin_memcpy(&bits, &x, sizeof(T));
        return ((bits & 0x7F80u) == 0x7F80u) && ((bits & 0x007Fu) != 0u);
    } else { // float (fp32)
        uint32_t bits;
        __builtin_memcpy(&bits, &x, sizeof(T));
        return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
    }
}
```

## 设计决策

选择把 `__simt_vf__` 限定为由 `asc_vf_call` 启动的入口，把复用逻辑声明为 `__simt_callee__`，因为这与编译器明确给出的调用约定一致，且不复制 helper 业务逻辑。轨迹没有尝试把函数体复制进每个调用点，也没有通过删掉 helper 调用规避问题。当前方案属于 `recommended_fastpath`。

## 系统信息

- Toolchain：CANN 9.0.1 的 ASC 编译链。
- 目标：Ascend950PR_957b / dav-3510。
- 工程入口：CMake/make 构建单个 `.asc` 翻译单元。
- 相关文件：`operators/<op>/<op>.asc`。

## 预防

- 代码审查时区分“由 `asc_vf_call` 启动的 VF 入口”和“入口调用的设备 helper”，不要仅因两者都在 SIMT 侧执行就使用相同属性。
- 搜索每个 `__simt_vf__` 函数体里的自定义函数调用，逐个确认被调用符号带 `__simt_callee__`。
- 在增加 dtype 模板实例后立即编译全部分支；该错误会在每个实例化点重复出现，首条 `simt_callee` note 已足够定位根因。

## 验证信号

本知识来自一次 Ascend950 direct-launch Kernel 开发中的真实编译失败与单行属性修复；修复后继续完成动态库链接、真机 pytest 和干净重建复验。

- 修复前构建输出包含 `simt_vf function can only call simt_callee function`、`9 errors generated`，目标未生成（来源平台：950；证据等级：tool_observed）。
- 修复后构建输出为 `Linking ASC shared library` 与 `Built target`（来源平台：950；证据等级：tool_observed）。
- 修复后的真机测试集 42/42 通过，随后干净重建后的同一测试集再次 42/42 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论只覆盖 CANN 9.0.1 的 SIMT VF 调用约定；后续编译器版本是否放宽或改名未验证。
- A2、A3 未验证，不能从本次 Ascend950 结果直接推断其可用属性组合。
- 本配方只修复调用约定，不证明 helper 函数体的数值语义正确；数值正确性仍须独立测试。

## 坏实践

把所有 SIMT 侧函数统一标成 `__simt_vf__` 会混淆“启动入口”和“可调用设备函数”；即使符号可见、模板实例化正常，编译器仍会拒绝 VF 到 VF 的普通函数调用。

## 误诊

错误在多个 dtype 实例化点重复出现，容易被误判为模板重载或 NaN 位模式实现问题；编译器 note 已明确指出候选不可调用的原因是函数属性，不应先改模板参数或函数体。