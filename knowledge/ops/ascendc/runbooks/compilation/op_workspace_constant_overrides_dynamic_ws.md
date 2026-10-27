---
type: "Runbook"
title: OP_WORKSPACE 常量覆盖 tiling 动态 workspace
description: ADD_TO_LAUNCHER_LIST_AICORE 里加 OP_WORKSPACE(常量 0) 后，tiling GetWorkspaceSizes 动态声明的 29184 被静默覆盖为 0，触发 workspaceSize=0 fallback；移除宏后恢复——对已由 tiling 动态声明的算子禁止再用常量 OP_WORKSPACE。
tags: [task.debugging, topic.memory, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T03:26:59Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# OP_WORKSPACE 常量覆盖 tiling 动态 workspace

## 现象

排查 GQA 算子 bench_no_harness_gqa 的真机 507035 时，一度怀疑 aclnn 适配层没有把 workspace 声明进 launcher。该算子 op_api 的适配代码原本是：

```cpp
ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessGqa,
    OP_INPUT(query, key, value), OP_ATTR(scaleValue, is_causal), OP_OUTPUT(y));
```

tiling 函数则通过 `context->GetWorkspaceSizes(WORKSPACE_NUM)` 动态计算并写入 `ws[0]`（对本用例为 29184 字节）。在 launcher 宏里追加了 `OP_WORKSPACE(static_cast<uint64_t>(0))`（意图是"显式声明 workspace 槽位"）并重装算子包后，ST 输出从：

```
  workspaceSize=29184
```

变为：

```
  workspaceSize=0
  [WARN] workspaceSize=0, allocated 4096 bytes fallback
  [FAIL] 流同步失败: 507035
```

移除该宏、重装后 workspaceSize 恢复 29184。当时 kernel 对 workspace 的访问仍触发 507035（真正根因是 workspace 声明须含保留区，见 507035 MTE DDR 越界卡），本卡记录的是这次尝试中发现的 workspace 声明通道互斥行为。

## 适用场景

- 工程形态：aclnn 自定义算子的 op_api 适配层（`aclnn/opdev/make_op_executor.h` 的 `ADD_TO_LAUNCHER_LIST_AICORE` / `OP_WORKSPACE` 宏）。
- API 路径：`ADD_TO_LAUNCHER_LIST_AICORE` + `OP_WORKSPACE(常量)` vs TilingFunc `context->GetWorkspaceSizes()` 动态声明。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（aclnn 框架宏行为）。
- 路线优先级：banned——对已由 tiling 动态声明 workspace 的算子，禁止再用常量 `OP_WORKSPACE` 声明。
- 明确不适用 / 未验证场景：确实需要固定大小 workspace、tiling 不做动态声明的算子（`OP_WORKSPACE` 常量是其正常用法）；宏与动态声明的合并语义（覆盖 vs 相加）未展开宏验证。

## 根因

`OP_WORKSPACE` 宏以固定常量声明 workspace 大小，会覆盖 Tiling 函数通过 `context->GetWorkspaceSizes()` 写入的动态值——本例中常量 0 把 29184 覆盖为 0（机制层解读：单次前后对照 + 宏参数语义；宏展开未验证，见遗留问题）。

## 解决配方

无正向修复（该尝试本身即坏实践）；正确做法是移除常量 `OP_WORKSPACE`，保持 tiling 的 `GetWorkspaceSizes` 动态声明为唯一 workspace 大小来源：

```cpp
ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessGqa,
    OP_INPUT(query, key, value), OP_ATTR(scaleValue, is_causal), OP_OUTPUT(y));
```

```cpp
    // op_host tiling：动态 workspace 的唯一声明点
    int64_t scoresSize = S * S_kv * FP32_TYPE_SIZE;
    tiling->wsPerCore = CeilAlign(scoresSize, static_cast<int64_t>(512));
    int64_t totalWs = tiling->wsPerCore * aivNum;
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    ws[0] = CeilAlign(static_cast<uint64_t>(totalWs), 512UL);
```

## 预防

- workspace 大小声明只保留一个来源：tiling `GetWorkspaceSizes`（动态）或 `OP_WORKSPACE`（固定常量），二者不同时使用。
- ST 验证 workspace 声明链路时，打印实际 `workspaceSize`（两段式第一段返回值）与 tiling 预期对照，覆盖为 0 时立刻可见。

## 验证信号

以下信号来自该算子 507035 排查会话中的一次受控前后对照：

- 加宏前：ST 打印 `workspaceSize=29184`（来源平台：950；证据等级：tool_observed）。
- 加 `OP_WORKSPACE(static_cast<uint64_t>(0))` 后：ST 打印 `workspaceSize=0` + `[WARN] workspaceSize=0, allocated 4096 bytes fallback`（来源平台：950；证据等级：tool_observed）。
- 移除宏后：恢复 `workspaceSize=29184`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 单次前后对照（加/删宏各一次）；未展开宏验证覆盖语义是"赋值覆盖"还是"取 max/min"。
- 常量非 0 时的行为未测试（本例常量为 0，表现为清零）。

**未决项**：

- 展开 `OP_WORKSPACE` 宏（`aclnn/opdev/op_arg_def.h`）验证与 `GetWorkspaceSizes` 的合并语义，确认是否所有常量值都覆盖动态声明。
## 坏实践

- 在 tiling 已动态声明 workspace 的算子上又加常量 `OP_WORKSPACE`（即使值是 0 也会静默清零动态值）。
- ST 侧对 `workspaceSize==0` 做 fallback 分配（4096 字节），会让"声明被覆盖"这类问题以更隐蔽的方式存活。

## 误诊

- 加宏的动机本身是一次误诊（以为 507035 是 launcher 未声明 workspace 导致）——加宏后 507035 依旧，且引入了 workspaceSize=0 的新干扰。
