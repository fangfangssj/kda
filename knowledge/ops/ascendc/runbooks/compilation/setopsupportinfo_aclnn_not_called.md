---
type: "Runbook"
title: "SetOpSupportInfo 回调在 aclnn 直调不被调用"
description: "_def.cpp 注册 SetOpSupportInfo（返回 int64Mode=true）编译链接安装成功且符号入 .so，但 aclnn 单算子执行路径从不调用该回调（文件副作用探针证实）；kernel JSON 的 int64Mode 由 OPC 编译期独立生成，不能用该回调改变 aclnn 直调行为（banned）。"
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# SetOpSupportInfo 回调在 aclnn 直调不被调用

## 现象

在给一个自定义 argmax 算子（OpDef/`_def.cpp` 新式注册，ascend950）排查 int64 用例失败时，尝试用 `SetOpSupportInfo` 回调让框架按 `int64Mode=true` 处理 int64 输入：在 `op_host/*_def.cpp` 中定义回调并注册，`bash build.sh --soc=ascend950 -j8` 编译通过、`custom_opp_ubuntu_x86_64.run --force` 安装成功，`strings` 也能在安装后的 `libcust_opsproto_rt2.0.so` 里找到 `int64Mode` 符号串（说明注册确实编译进了 proto 库）。但重跑 aclnn 直调 ST 测试后，目标失败用例的输出一个字节都没变；为了确认回调是否被触发，在回调里加了写 `/tmp/getsupportinfo_debug.log` 的副作用语句，跑完全量 ST 后该日志文件根本不存在——回调在 aclnn 执行路径中从未被调用。

## 适用场景

- 工程形态：算子仓自定义算子（`register/op_def.h` 的 `OpDef` + `OpAICoreDef::SetOpSupportInfo`）+ aclnn 单算子直调（`aclnnXxxGetWorkspaceSize`/`aclnnXxx` 两段式）测试或推理路径。
- API 路径：`optiling::OP_CHECK_FUNC`（`ge::graphStatus (*)(const ge::Operator &op, ge::AscendString &result)`）、`OpAICoreDef::SetOpSupportInfo`、`register/op_def_registry.h` 头文件链（其内含 `register/op_def.h` → `graph/operator_reg.h`，`ge::Operator`/`ge::AscendString` 由传递包含提供，无需额外 include）。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：platform_wide（结论是 CANN 框架 aclnn 路径的调用时机问题，与具体算子无关）。
- 路线优先级：banned（作为"用 SetOpSupportInfo 修 aclnn 直调行为"的路线已被本切片证伪；图模式场景未验证，不作推荐）。
- 明确不适用/未验证场景：GE 图编译路径（是否调用该回调未验证）；本卡只证明"aclnn 直调不触发"，不涉及回调返回值语义本身的对错。

## 根因

`SetOpSupportInfo` 注册的是框架侧的 support info 查询回调，编译期 kernel JSON 的 `int64Mode` 由 OPC 工具独立生成（对动态 shape 算子恒为 false），aclnn 单算子执行路径不下发该查询——回调注册成功与回调被调用是两件事。本切片以文件副作用探针证实"不被调用"，但框架内部不查询的确切代码路径未追到（见遗留问题）。

## 解决配方

**尝试的注册代码（可编译、可链接、可安装，但 aclnn 路径不生效）：**

```cpp
// Register OpSupportInfo to enable int64Mode=true in kernel JSON.
// Without this, the framework defaults int64Mode=false and skips calling
// the custom tiling function for int64 dtype inputs, causing incorrect
// kernel binary selection for int64 AR small-shape cases.
static ge::graphStatus GetOpSupportInfo(const ge::Operator &op, ge::AscendString &result) {
    result = "{\"int64Mode\": true}";
    return ge::GRAPH_SUCCESS;
}
```

注册处（`_def.cpp` 构造函数内）：

```cpp
        // Enable int64Mode=true so the framework calls custom tiling for int64 dtype.
        this->AICore().SetOpSupportInfo(GetOpSupportInfo);
```

**验证回调是否被调用的副作用探针**（在回调内写标志文件，跑完全量后检查文件是否存在）：

```cpp
static ge::graphStatus GetOpSupportInfo(const ge::Operator &op, ge::AscendString &result) {
    FILE* dbg = std::fopen("/tmp/getsupportinfo_debug.log", "a");
    if (dbg) { std::fprintf(dbg, "GetOpSupportInfo called\n"); std::fclose(dbg); }
    result = "{\"int64Mode\": true}";
    return ge::GRAPH_SUCCESS;
}
```

结论性做法：不要依赖该回调改变 aclnn 直调行为；若目标问题确实是框架选择逻辑（如 int64 跳过 tiling），转向"三步探针确认归属 + 已知限制记录 + 升级 CANN"的处置。

## 设计决策

- 选文件副作用而不是日志框架来验证调用：回调发生在 host 侧进程内，`fprintf` 到 `/tmp` 的标志文件最直接、无日志级别/路由依赖；文件不存在即为"未调用"的一票证据。
- 先 `strings` 确认符号进入 `libcust_opsproto_rt2.0.so` 再做运行时验证：把"注册未生效（编译/安装问题）"与"注册成功但不被调用（框架行为）"两类假设分开，避免误判。

## 系统信息

- 芯片 Ascend950PR + CANN 9.1.0；头文件 `include/register/op_def.h`（`OP_CHECK_FUNC` 定义于第 37 行附近，`SetOpSupportInfo` 声明于 `OpAICoreDef` 第 444 行附近）。
- 相关文件：`op_host/bench_no_harness_arg_max_def.cpp`（回调 + 注册）、安装目录 `opp/vendors/<vendor>/op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so`（strings 验证对象）。
- 验证命令形态：`bash build.sh --soc=ascend950 -j8`、`bash build/custom_opp_ubuntu_x86_64.run --force`、`rm -f /tmp/getsupportinfo_debug.log && rm -rf ~/atc_data/kernel_cache/`、跑全量 ST 后 `ls /tmp/getsupportinfo_debug.log`。

## 验证信号

以下信号来自该 argmax 算子 int64 用例排查会话（Ascend950PR + CANN 9.1.0）：

- 注册后编译、打包、安装均成功（build completed successfully）（来源平台：950；证据等级：tool_observed）。
- `strings` 在 `libcust_opsproto_rt2.0.so`、`liboptiling.so`、`libcust_opmaster_rt2.0.so` 中命中 `int64Mode`（注册编译进库）（来源平台：950；证据等级：tool_observed）。
- 跑完全量 aclnn ST 后 `/tmp/getsupportinfo_debug.log` 未生成，测试输出明确给出 `GetOpSupportInfo NOT called!`（来源平台：950；证据等级：tool_observed）。
- 安装后 kernel JSON 5 个 binList 条目仍全部 `"int64Mode": false`（编译期字段未被回调覆盖）（来源平台：950；证据等级：tool_observed）。
- 目标失败用例（int64 AR 小 shape）结果不变：`期望=7, 实际=6`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡边界是"aclnn 单算子直调路径 + CANN 9.1.0"：该路径下回调不被调用；GE 图编译路径未验证（未验证，推断可迁移到其他 CANN 9.x aclnn 路径）。
- 回调本身 API 用法（签名、注册点）是正确的，可作为图模式场景的参考实现；被证伪的只是"它能影响 aclnn 直调行为"这一假设。

**未决项**：

1. 【优先】复现并追根因——在 GE 图编译路径上复现本探针，确认 `GetOpSupportInfo` 是否被调用及其返回值的实际消费方式。
2. 定位 CANN 框架中消费 `SetOpSupportInfo` 注册的调用点（哪些组件、什么时机），给出官方适用场景说明。
## 坏实践

- 把 `SetOpSupportInfo` 注册当成 aclnn 路径的 runtime 开关来用：编译通过不等于会被调用，必须先做调用时机验证。
- 只对比"改前改后 ST 结果"就下结论：本切片中即使回调被注册，ST 输出也无差异，若不做副作用探针，无法区分"没被调用"和"被调用但返回值被忽略"两种机制。

## 误诊

- "SetOpSupportInfo 已通过源码分析确认可开启 int64Mode"（任务输入中的既有结论）——本切片实测证伪其 aclnn 路径有效性（来源平台：950；证据等级：tool_observed，作为误诊记录）。
- "回调在图模式下会生效"——仅会话内推断，无图模式调用证据（未验证）。
