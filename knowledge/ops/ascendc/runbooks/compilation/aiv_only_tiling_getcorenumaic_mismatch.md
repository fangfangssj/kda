---
type: "Runbook"
title: "纯向量算子tiling误用GetCoreNumAic"
description: 从 Matmul MIX 版重写为纯向量版后 host tiling 仍取 AIC 核数，blockDim 与 AIV 实际核数不符；改为 GetCoreNumAiv 并对齐参考算子。
tags: [task.debugging, topic.tiling, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# 纯向量算子tiling误用GetCoreNumAic

## 现象

GQA 注意力算子 bench_no_harness_gqa 最初用 Matmul 高阶 API 实现，kernel 入口带 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0)`，当时 host tiling 按 MIX 模式规范用 AIC 核数切分（注释写明"MIX 模式下用 AIC 核数（每个 AIC 关联一个 AIV）"）。后来 kernel 因 NPU 反复卡死整体重写为**纯向量 API 版**（Mul+ReduceSum，入口不再声明 MIX 任务类型），但 `op_host/arch35/bench_no_harness_gqa_tiling.cpp` 的 `GetPlatformInfo` 里原样保留了：

```cpp
    // MIX 模式下用 AIC 核数（每个 AIC 关联一个 AIV）
    *aivNum = plat.GetCoreNumAic();
    OP_CHECK_IF(*aivNum == 0, OP_LOGE(context, "AicNum is 0"), return ge::GRAPH_FAILED);
```

该核数直接进入 `tiling->headsPerCore = CeilDiv(totalQHeads, aivNum)` 与 blockDim 设置。真机 ST 表现为流同步失败 507035 与多次 NPU 驱动级锁死（npu-smi 超时、D 状态进程 `kill -9` 无效）。对照同仓参考算子 add_example（arch35）的 tiling 实现：

```
references/add_example/op_host/arch35/add_example_tiling.cpp:
  Line 55:     *coreNum = ascendcPlatform.GetCoreNumAiv();
```

参考算子（纯向量、无 MIX 宏）统一使用 `GetCoreNumAiv()`。

## 适用场景

- 工程形态：算子仓开发（op_host TilingFunc + AscendC kernel）。
- API 路径：`platform_ascendc::PlatformAscendC::GetCoreNumAiv()/GetCoreNumAic()` + tiling 核数字段 + blockDim。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有 AIV-only 纯向量算子的 tiling。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：MIX（AIC+AIV）或 AIC-only 算子应按其任务类型规范取核数，不适用本卡；其他架构未验证。

## 根因

纯向量（AIV-only）算子的多核切分核数必须取 `GetCoreNumAiv()`，沿用 Matmul/MIX 版的 `GetCoreNumAic()` 会让 blockDim 按立方核数设置，与向量核实际可调度核数不匹配。

## 解决配方

修复前（op_host/arch35/bench_no_harness_gqa_tiling.cpp，`GetPlatformInfo`）：

```cpp
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto plat = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // MIX 模式下用 AIC 核数（每个 AIC 关联一个 AIV）
    *aivNum = plat.GetCoreNumAic();
    OP_CHECK_IF(*aivNum == 0, OP_LOGE(context, "AicNum is 0"), return ge::GRAPH_FAILED);
```

修复后：

```cpp
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto plat = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // 纯向量算子用 AIV 核数（对齐 add_example 参考实现）
    *aivNum = plat.GetCoreNumAiv();
    OP_CHECK_IF(*aivNum == 0, OP_LOGE(context, "AivNum is 0"), return ge::GRAPH_FAILED);
```

## 预防

- 从 Matmul/MIX 版本重写或回退为纯向量版时，用 `grep -rn "GetCoreNum"` 同时检查 op_host tiling 与 kernel 侧所有核数消费点，任务类型宏与核数获取必须成对切换。
- 新写纯向量算子直接对照参考算子 add_example 的 tiling 模板。

## 验证信号

来源说明：以下信号来自该算子真机联调会话（修复时 NPU 处于驱动锁死状态，无法立即单独回测；修复保留在终版中随全量回归组合验证）：

- 参考算子证据：add_example arch35 tiling 使用 `GetCoreNumAiv()`（同仓参考源码 grep，工具输出）（来源平台：950；证据等级：tool_observed）。
- 修复前症状：真机 ST 507035 反复出现并伴随 NPU 驱动级锁死（来源平台：950；证据等级：tool_observed）。
- 组合验证：含本修复的终版（两遍算法 + PIPE_ALL 同步 + GetCoreNumAiv）全部 27 条用例 PASS（来源平台：950；证据等级：tool_observed；单变量隔离未做，见失效边界）。

## 失效边界

- 本修复未做单变量隔离复现（与同步修复、workspace 规避同批落地后统一回归）；"GetCoreNumAic 单独导致锁死"的因果强度弱于同批 workspace 根因（见同批 `kernel_workspace_param_direct_bind_507035`）。
- 仅 Ascend950PR 实测；AIC/AIV 配比不同的架构上错核数的具体表现（锁死 vs 越界 vs 静默错误）未验证。
- 相关背景：运行时 `GetCoreNumAiv()` 实测可能低于平台 INI 物理核数（见库内 `getcorenumaiv_runtime_below_ini_core_count`），含 coreNum 阈值的分支谓词须按运行时值标定。

**未决项**：

- 做一次单变量隔离复现：仅回退 GetCoreNumAiv，验证是否独立复现锁死/507035。
## 坏实践

- kernel 侧删掉 MIX 任务类型宏但 tiling 侧核数获取不动——两处必须成对改。
- 只看算子自身代码不看同仓参考算子的规范写法，凭注释（"MIX 模式下用 AIC 核数"）延续旧语义。

## 误诊

无独立误诊记录；本缺陷是在排查 507035 期间对照参考算子发现的伴生缺陷（当时曾把 NPU 锁死主因先后误归于任务类型宏与标量对齐，见同批 workspace 卡的误诊节）。
