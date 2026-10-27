---
type: "Runbook"
title: "DynamicCompileStaticFlag(false) 致 E40021"
description: "期望用 OpAICoreConfig.DynamicCompileStaticFlag(false) 强制 GE 图路径只用预编译二进制不可行：两次独立排障会话实测两种失败形态——TBE 编译进程 E40021（Compilation_Error / Python 模块导入失败）与 stale 清理后无任何 kernel 下发的静默 RunGraph 失败；回退 true 恢复，该路线 banned。"
tags: [task.debugging, topic.compilation, topic.graph_execution, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# DynamicCompileStaticFlag(false) 致 E40021

## 现象

在 Ascend950PR（CANN 9.2.0）上为 BenchHarnessScatter 做 GEIR 图模式验收/排障时，GE 图路径持续落到 TBE 动态编译（blockDim 固化平台核数，float 用例 EZ9999）。为了"强制 GE 只用包内预编译二进制（blockDim=-1）"，在 `op_host/bench_harness_scatter_def.cpp` 的 OpAICoreConfig 里把标志翻为 false：

```cpp
        // --- AICore configuration ---
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "bench_harness_scatter_apt");
```

重新 stage + build 后（ops-info JSON 中 `dynamicCompileStatic: {"flag": "false"}` 确认生效），两次独立会话共观测到两类都算失败的形态：

- **形态一（TBE 编译进程内失败，报 E40021）**：route 的 GRAPH-RUN 阶段 `rungraph_rc=4294967295`（-1）、`outputs=0`，日志中不再有 `FillTaskDefAfterGenTask` / `StreamLaunchKernelV2`（kernel 未下发，图编译期失败）；TTK geir static 全部用例 EXEC_FAILURE，报错原文：

```
ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error:
[PID: ...] Compilation_Error(E40021): Failed to compile Op op1.
oppath is /tmp/geir_opp.*/ve...
```

  另一次实验（CANN OPP 里还残留旧的同名算子安装时）route 仍发起 TBE 动态编译，且编译进程内 Python 导入失败，route 日志报 `error_code:E40021`，编译异常原文：

```
[INFO] GE(...): ReportErrMessage:report error_message, error_code:E40021, work_stream_id:..., error_mode:0.
[WARNING] TEFUSION(...): ReportBuildErrMessage Compile exception message is:[File "/home/<user>/miniconda3/envs/.../lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  ...
```

  完整 ModuleNotFoundError 为 `No module named 'BenchHarnessScatter_impl.bench_harness_scatter_apt'`（staging 包里明明有 `op_impl/ai_core/tbe/BenchHarnessScatter_impl/dynamic/bench_harness_scatter_apt.py`）。

- **形态二（stale 安装清理后，静默无下发）**：route 里完全没有 kernel 下发——无 `StreamLaunchKernelV2`、无 blockDim、无 GenSimplifiedKey 调用、无 binary match、也没有任何报错行，RunGraph 直接 `rungraph_rc=4294967295 value=FAIL`（静默失败）。

伴随信号（flag=false 的 route 日志）：`Env[OP_DYNAMIC_COMPILE_STATIC] is not found`；`LoadOpMasterSo:Get path with op_master path [.../op_impl/ai_core/tbe/op_master/lib/linux/x86_64/] failed, reason:No such file or directory`（staging 与全局 vendor 均无此目录）；`Failed to dlopen .../op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so! ... cannot open shared object file`。

两种形态都没拿到预编译二进制，最终都把标志回退为 `DynamicCompileStaticFlag(true)`；回退重建后 route / infer V2 10/10 / reject 全部恢复 PASS，生产源码 git diff 为空。

## 适用场景

- 工程形态：自定义算子 GEIR/GE 图模式（RunGraph / TTK geir）开发与联调，试图控制图编译路径走预编译二进制、绕开 TBE 动态编译问题。
- API 路径：OpDef `OpAICoreConfig::DynamicCompileStaticFlag`（`aicoreConfig.DynamicCompileStaticFlag(...)`，经 op_build 生成 ops-info 的 `dynamicCompileStatic` 条目）；GE 图编译的 staticKey/预编译二进制选择链路。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有走 op_build + GE 图编译的自定义算子（host 侧标志语义，与具体算子无关）。
- 路线优先级：banned（作为"强制预编译二进制"的手段）；排查 stale 安装本身见同批 opp/vendors stale 安装混读 flag 知识。
- 明确不适用 / 未验证场景：本卡不断言 `dynamicCompileStatic` 的完整权威语义（true/false 与 TBE 回退的关系在两次会话内只有行为对照，无官方定义核对）；仅记录该尝试的客观失败形态。若算子的 TilingParse 正常注册、op_master 目录完整，flag=false 是否可用未验证。

## 根因

尝试的出发点基于一个未证实的语义假设——"false = 只用预编译二进制、不命中即失败"。两次会话的行为对照：

- false + 存在 stale vendor（true 混读）：仍发起 TBE 动态编译，并因该路径下 TBE 编译进程无法导入 `BenchHarnessScatter_impl.bench_harness_scatter_apt` Python 入口而报 E40021（导入失败的确切环境原因未定位，见遗留问题）。
- false + 环境干净：不发起 TBE 编译，但预编译二进制仍未命中（binary matching 本身未生效，见同批 GE 图路径 TBE 动态编译判别知识），于是 RunGraph 无 kernel 可下发、无错误信息直接失败。

即：**flag 翻 false 改变不了"预编译二进制不命中"这一上游事实**，只会把失败形态从"TBE 回退编译（可能 E40021）"换成"静默无下发"。E40021 的确切失败层（staticKey 匹配不命中 vs op_master/liboptiling SO 缺失 vs TilingParse 禁用致 CompileInfo 缺失）未逐层定位。

## 设计决策

- 为什么尝试 false：当时假设 dynamicCompileStatic=false 能让 GE 只查预编译二进制（blockDim=-1 + SetBlockDim 生效），从根上绕开 TBE 动态编译的 blockDim 固化问题。
- 为什么回退 true：false 的两种形态都未恢复正向执行，而 true 至少保持 TBE 编译可用（ACLNN 直调回归不受影响）；且日志混读排查成本高。
- 放弃的第三条路：直接改 kernel 加核守卫——被任务边界（不修改 Kernel）排除。
- 当前方案档位：banned——不要把"翻 false"当作让图模式走预编译二进制的手段。

## 预防

- 动 OpDef 标志前，先用 route 日志确认当前实际走哪条编译路径（kernel JSON 来源/te_ 前缀/blockDim 值），并确认 CANN OPP 下没有同名算子的 stale 安装（否则标志读数混叠，任何实验结论都不可信）。
- 不要依据标志名望文生义；CANN 侧标志语义以官方文档/头文件注释为准（两次会话均未核对成功，见遗留问题）。

## 验证信号

以下信号来自该算子两次独立 GEIR 排障会话中对 DynamicCompileStaticFlag 的双向实验（改 false → 失败 → 回退 true → 恢复）的 route 日志与 TTK 用例实测。

会话一（TilingParse 注册被注释、以隔离 561002 问题的代码状态）：

- flag=false 生效证据：重建后 `aic-ascend950-ops-info.json` 中 `"dynamicCompileStatic":{"flag":"false"}`（来源平台：950；证据等级：tool_observed）。
- flag=false 失败证据：route `GRAPH-RUN ... rungraph_rc=4294967295 outputs=0`、无 FillTaskDef/StreamLaunchKernelV2；TTK static 用例（f16/f32/bf16/i32/i64/f32_i64 等 8 例）全部 `EXEC_FAILURE ... Compilation_Error(E40021): Failed to compile Op op1`（来源平台：950；证据等级：tool_observed）。
- 伴随信号：`Env[OP_DYNAMIC_COMPILE_STATIC] is not found`；`LoadOpMasterSo ... op_master ... No such file or directory`；`Failed to dlopen .../liboptiling.so`（来源平台：950；证据等级：tool_observed）。
- 回退恢复证据：`DynamicCompileStaticFlag(true)` 重建后 `check-st-frozen PASS(16 files)`、`check-route --expect pass PASS`、`infer V2 10/10 PASS（RUN-EXIT status=0）`、`reject 1/1 PASS`；生产源码 `git diff --stat -- op_host op_graph op_kernel` 为空（来源平台：950；证据等级：tool_observed）。

会话二（含 stale vendor 清理前后对照）：

- flag=false（stale 未清理）：route `EXIT=1`、`probe_exit=1`，日志 `ReportErrMessage:report error_message, error_code:E40021`，`ReportBuildErrMessage Compile exception message` 为 Python `importlib/_bootstrap._gcd_import` 栈，`ModuleNotFoundError: No module named 'BenchHarnessScatter_impl.bench_harness_scatter_apt'`；`InitOpItemValueByMaping` 出现 false/true/true 三次读数（来源平台：950；证据等级：tool_observed）。
- flag=false（stale 清理后）：route `EXIT=1`，无 StreamLaunchKernelV2、无 blockDim、无 binary match、无错误行，`rungraph_rc=4294967295 value=FAIL`（来源平台：950；证据等级：tool_observed）。
- 回退 true 后：TBE 动态编译恢复（coreDim=6 下发、EZ9999 复现）——即 true/false 都无法让预编译二进制命中（来源平台：950；证据等级：tool_observed）。

## 失效边界

- E40021 的 Python 导入失败形态依赖"stale vendor 混读 true"这一环境条件；本卡未在无 stale 的环境里复现该形态（干净环境里 false 表现为静默失败）。
- "false 时不回退 TBE"仅对清理后的环境观测成立；因果归因（混读的 true 触发 TBE）是对照推断，未做定向实验（agent_reported，见误诊标注）。
- 本卡结论是"该 flag=false 路线在此算子 + CANN 9.2.0 + 本 staging 布局下让图编译失败"，不是对 flag 语义的一般性结论；其他 CANN 版本 / TilingParse 正常注册的算子未验证。
- 其他算子（opFile 入口名不同）的 E40021 文本会不同，但"编译进程失败 + E40021"形态可对照。

**未决项**：

- **优先项：核对 dynamicCompileStatic 的官方语义**（CANN 文档/头文件），确定 false 下预编译二进制不命中时的预期行为（直接失败 vs 回退 TBE）。
- E40021 中 TBE 编译进程导入 `BenchHarnessScatter_impl.bench_harness_scatter_apt` 失败的确切原因（PYTHONPATH、模块布局、staging 挂载方式）未定位。
- E40021 的确切失败层定位：在 staticKey 匹配链路打开 debug 日志，区分"匹配不命中"与"op_master/liboptiling 加载失败"，并验证 TilingParse 恢复注册后 flag=false 是否可用。
- `Cannot find aicore config for op[...]` 警告（两种 flag 配置下均出现 4 次）与本次失败的关系未评估。
## 坏实践

- 把 `DynamicCompileStaticFlag(false)` 当成"强制预编译二进制"开关写进算子包并指望它修复图模式执行——两种实测形态都失败。
- 在 CANN OPP 存在同名 stale 安装时做任何标志实验并采信结论（读数混叠）。
- 初次改 false 后 route 仍见 kernel_meta 引用就断定"flag 未生效"并反复重装——先确认 ops-info 是否重建、staging 是否混用旧包。

## 误诊

- "false 时系统仍回退 TBE 动态编译"最初被当作 flag 语义的反证；清理 stale 后 false 不再触发 TBE，更可能的解释是混读的 true 触发了 TBE——该因果链未验证（agent_reported）。
- 事后归因"staticKey 匹配不命中（TilingParse 被禁用致 CompileInfo 缺失）"写入验收文档，属于 agent 假设（agent_reported）：会话内只验证了 E40021 现象与 SO 缺失伴随信号，未验证该机制链。
