---
type: "Runbook"
title: ExtendCfgInfo 设 jitCompile：enable/0 非法
description: 试图用 OpAICoreConfig::ExtendCfgInfo 给自定义算子 ops-info 加 jitCompile 字段以启用预编译二进制匹配：.value 子键框架不读，.flag 子键的 enable 与 0 均被 op_kernel_info_constructor 判 invalid，尝试整条路线失败回退。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# ExtendCfgInfo 设 jitCompile：enable/0 非法

## 现象

在 Ascend950PR（CANN 9.2.0）上排查自定义算子 BenchHarnessScatter 的 GE 图路径恒走 TBE 动态编译（kernel_name 为 `te_benchharnessscatter_...`、blockDim=6）问题时，route 日志里有一条关键线索：

```
[DEBUG] FE(...): [op_kernel_info_constructor.cc:1335] GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!
```

即框架初始化 op kernel info 时会查 ops-info 的 `jitCompile` 字段，而该算子的 `aic-ascend950-ops-info.json` 里没有这个字段（dump 证实只有 `dynamicCompileStatic: {flag: "true"}`、`dynamicRankSupport`、`opFile` 等）。当时推断：补上 `jitCompile` 字段可能让框架优先走预编译二进制匹配。于是依次尝试了四种写法（详见尝试段），全部未达到目的：

- `.value` 子键：打包成功，但框架日志仍是 `not found jitCompile in OpContent!` + `Operation BenchHarnessScatter not found in jitCompile.flag of OpContent!`；
- `.flag` 子键 + 值 `enable`：打包成功、字段被读到，但 `The parameter value [enable] for [jitCompile] ... is invalid`；
- `.flag` 子键 + 值 `0`：同样 `The parameter value [0] for [jitCompile] ... is invalid`；
- 最终把该行 `ExtendCfgInfo` 整体删除回退，kernel 选择行为与尝试前完全一致。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：AscendC 自定义算子 op_host 注册层（OpDef/OpAICoreConfig）+ GE 图路径加载。
- API 路径：`OpAICoreConfig::ExtendCfgInfo("jitCompile.flag", ...)` → ops-info.json → GE `op_kernel_info_constructor.cc` 的 `GetStrFromOpContent`/`InitOpItemValueByMaping`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`——CANN 9.2.0 上所有想通过 op 注册层控制编译模式的自定义算子。
- 路线优先级：`banned`——"经 `ExtendCfgInfo` 设置 `jitCompile`"这条路线在本版本被实测证伪（四种形态全部失败），不要再原样重试；可行替代路线未找到（见遗留问题）。
- 明确不适用 / 未验证场景：内置算子（其 ops-info 由 ATC 侧生成，不走本打包链）；其他 CANN 版本的合法值集；`ge.jit_compile` 会话级选项（本组会话只试过 `GE_OPTIONS`/`GE_JIT_COMPILE` 环境变量，无效且原因未验证）。

## 尝试与观测（负结果）

四次尝试的全部代码形态（`op_host/bench_harness_scatter_def.cpp`，基线为无 jitCompile 配置）：

1. `.value` 子键（打包成功、框架不读）：

```cpp
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt")
            .ExtendCfgInfo("jitCompile.value", "enable");
```

route 日志（verbatim）：

```
[DEBUG] FE(3609679,geir_route):2026-09-05-04:43:52.214.739 [op_kernel_info_constructor.cc:1335]3609679 GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!
[DEBUG] FE(3609679,geir_route):2026-09-05-04:43:52.215.252 [op_kernel_info_constructor.cc:1343]3609679 GetStrFromOpContent:Operation BenchHarnessScatter not found in jitCompile.flag of OpContent!
```

2. `.flag` 子键 + `enable`（字段被读到、值被拒）：

```cpp
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt")
            .ExtendCfgInfo("jitCompile.flag", "enable");
```

route 日志（verbatim）：

```
[DEBUG] FE(3612442,geir_route):2026-09-05-04:44:45.022.869 [op_kernel_info_constructor.cc:1335]3612442 GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!
[ERROR] FE(3612442,geir_route):2026-09-05-04:44:45.023.591 [op_kernel_info_constructor.cc:1394]3612442 InitOpItemValueByMaping:[InitOpInfoLib][InitOpKernel] The parameter value [enable] for [jitCompile] in operation [BenchHarnessScatter] is invalid.
```

3. `.flag` 子键 + `0`（同样被拒）：

```cpp
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt")
            .ExtendCfgInfo("jitCompile.flag", "0");
```

route 日志（verbatim）：

```
[DEBUG] FE(3615550,geir_route):2026-09-05-04:45:38.972.750 [op_kernel_info_constructor.cc:1335]3615550 GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!
[ERROR] FE(3615550,geir_route):2026-09-05-04:45:38.973.248 [op_kernel_info_constructor.cc:1394]3615550 InitOpItemValueByMaping:[InitOpInfoLib][InitOpKernel] The parameter value [0] for [jitCompile] in operation [BenchHarnessScatter] is invalid.
```

4. 回退（删除整行，恢复基线）：

```cpp
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt");
```

注意第 2/3 步的 ERROR 级日志出现在 `[InitOpInfoLib][InitOpKernel]` 上下文——即字段值非法时 op 的 kernel info 初始化本身报错，比"字段缺失走默认"更糟。

## 根因（未追到底，如实记录）

`jitCompile` 的合法值集未确定：`op_kernel_info_constructor.cc` 对 `enable` 与 `0` 均判 invalid，但本会话没有找到该字段的合法取值（CANN 安装内 grep `JIT_COMPILE_MODE`/`jit_compile_mode` 只命中编译器 Python 侧的 `jit_compile_mode`（默认 2），`strings` libopskernel.so 仅确认 `ge.jit_compile` 与 `jitCompile` 两个符号存在，没有值表）。"补 jitCompile 字段能否启用 binary matching"这一上层假设也因此未获验证。

## 设计决策

- 为什么试 `.value` 又改 `.flag`：框架日志显式提到查找 `jitCompile.flag`（`not found in jitCompile.flag of OpContent!`），说明读取路径是"顶层 `jitCompile` 字符串 → `jitCompile.flag` 子键"两级，`.value` 不在其列。
- 为什么试值 `0`：C++ 侧枚举 `JitCompileMode { REUSE_BINARY = 0, COMPILE_ONLINE, AUTO }` 的直觉映射——被值校验拒绝，说明 ops-info 侧的值域与该枚举字面值不同（agent 推断，未证实）。
- 为什么回退而不是继续穷举值：ERROR 出现在 op 初始化路径，风险高于收益；且任务约束为"只修改 GEIR 实现"，继续盲试值集属于高风险低把握探索。
- 当前方案路线档位：banned（本路线），恢复基线。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0。
- 相关文件：`op_host/bench_harness_scatter_def.cpp`（尝试点）、`build/autogen/aic-ascend950-ops-info.ini`（键形态）、route 日志（值校验报错源）；框架侧 `libopskernel.so`（闭源，仅 strings 探测）。

## 预防

- 不要假设 ops-info 字段值域与 C++ 头文件枚举字面值一致——本例 `enable`/`0` 双双被拒。
- 给 op 配置加字段前先确认读取方的两级键路径（顶层/子键）与值域；框架日志 `GetStrFromOpContent` 的两个 not found 行就是读取路径的直接证据。

## 验证信号

来源说明：以下信号来自该算子一次 GEIR 验收会话：围绕"让框架优先匹配预编译二进制"的目标，对 jitCompile 字段做的四种形态尝试与回退。

- `jitCompile.value=enable` 打包成功，route 日志两条 not found（顶层 jitCompile 与 jitCompile.flag 均未命中）——框架不读 `.value` 子键（来源平台：950；证据等级：tool_observed）。
- `jitCompile.flag=enable`：`[ERROR] InitOpItemValueByMaping ... The parameter value [enable] for [jitCompile] in operation [BenchHarnessScatter] is invalid.`（来源平台：950；证据等级：tool_observed）。
- `jitCompile.flag=0`：`[ERROR] ... The parameter value [0] for [jitCompile] ... is invalid.`（来源平台：950；证据等级：tool_observed）。
- 回退后重建重跑：route/infer/reject 合同检查恢复 PASS（`INFER-SUMMARY route=V2 total=10 pass=10`、`GEIR-SUMMARY total=1 pass=1`），kernel 选择行为与尝试前一致（仍 te_ + blockdim 6）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定"CANN 9.2.0 + 自定义算子打包链 + ExtendCfgInfo 途径"：`enable` 与 `0` 非法是实测；**合法值是否存在、是什么，未确定**（未穷举，如 `disable`/`1`/`2`/`auto` 等未试）。
- `ge.jit_compile` 会话/图选项途径未试（route_main.cpp 以空 options 建 Session，本组会话仅试过 `GE_OPTIONS`/`GE_JIT_COMPILE` 环境变量且无效）。
- 其他 CANN 版本的值域可能不同（未验证，推断不可直接迁移）。

**未决项**：

- **优先项：复现并追根因到机制层**——确定 CANN 9.2.0 `jitCompile` 字段的合法值集（可行入口：反汇编 `libopskernel.so` 中 `op_kernel_info_constructor.cc:1394` 对应的值校验函数；或用内置算子 ops-info 中已存在的 jitCompile 字段值做对照样本）。
- 上一层假设"补上合法 jitCompile 后 binary matching 是否被启用"仍未验证。
- `ge.jit_compile` 经 ge::Session options 传入的途径未试。
## 坏实践

- 值被拒后继续盲猜枚举字面值并反复重装全局 OPP——每次尝试都触发 op 初始化 ERROR，且无值表依据；应先定位合法值集（见遗留问题）再动手。
- 把 `InitOpItemValueByMaping` 的 ERROR 当 warning 忽略——它出现在 `[InitOpInfoLib][InitOpKernel]` 路径，意味着 op kernel info 初始化失败，比字段缺失更严重。

## 误诊

- "枚举 `JitCompileMode { REUSE_BINARY=0, ... }` 的 `0` 应该对应复用二进制"——值校验直接拒绝 `0`，该映射在 ops-info 值域内不成立（agent_reported 假设，被实测证伪）。
