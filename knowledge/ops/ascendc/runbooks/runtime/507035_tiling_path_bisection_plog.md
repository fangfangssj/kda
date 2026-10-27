---
type: "Runbook"
title: "507035 超时排障法：强制 tiling 路径隔离 + plog 探针"
description: "自定义算子全量 507035 时的三步定位法：临时硬编码 tilingPath 单路径二分隔离 → TilingFunc 加 OP_LOGI 探针 → 到运行用户 ~/ascend/log/debug/plog 按 PID 取日志。"
tags: [task.debugging, topic.implementation.debugging, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# 507035 超时排障法：强制 tiling 路径隔离 + plog 探针

## 现象

在 Ascend950PR（CANN 9.1.0）上联调 BenchNoHarnessSwiGlu 迭代三时，Real（NPU）模式 115 条用例**从第一条起全部** `[FAIL] 流同步失败: 507035`（kernel 执行超时），CPU golden 自测全过、Mock 模式此前全过——故障只在"真实 tiling + kernel"路径上。此时有两个干扰：测试二进制 stdout 缓冲导致前几轮看不到任何输出（误似挂死）；用例含大 shape，超时加长到 600s 依然 Exit 124，易误判为"跑得慢"。需要一套能快速把"全量超时"收敛到具体机制的方法。

## 适用场景

- 工程形态：aclnn 自定义算子开发的双 TilingKey 路径（或多模板路径）联调；任何"Real 全崩、Mock/golden 全过"的 507035 类超时。
- API 路径：Host TilingFunc（gert::TilingContext）；OP_LOGI 日志；plog 日志体系。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——tiling 探针与 plog 取证的方法（日志路径按运行用户变化）。
- 路线优先级：recommended_fastpath（排障方法，与具体算子无关）。
- 明确不适用 / 未验证场景：编译期错误（不走运行时日志）；host 侧崩溃（看 stderr/coredump 更直接）。

## 判别方法

三步（本轨迹完整走通；根因实例见 [507035_getinputstride_unfilled_misjudge](../compilation/507035_getinputstride_unfilled_misjudge.md)）：

**第 1 步：强制单路径二分隔离。** 在 TilingFunc 里临时硬编码走最简路径，重编安装重跑（verbatim 探针代码）：

```cpp
    // 4. 迭代三: 检测输入连续性
    NonContiguousInfo ncInfo = DetectContiguity(context, totalInput);
    uint32_t isContiguous = ncInfo.isContiguous ? 1 : 0;
    uint32_t tilingPath = isContiguous ? 1 : 2;
    // DEBUG: 临时强制 path=1 隔离问题
    tilingPath = 1;
    isContiguous = 1;
```

全量转 PASS → 问题在被禁用的路径（TilingKey=2）而非公共计算；仍崩 → 问题在公共路径或环境。一轮重编（约 2 分钟）换掉一半搜索空间。

**第 2 步：TilingFunc 加 OP_LOGI 探针。** 在可疑分支的关键变量处打点（verbatim）：

```cpp
    OP_LOGI(context, "DEBUG DetectContiguity: isView=%d, isContiguous=%d, tilingPath=%d, totalInput=%lu",
            context->InputIsView(0) ? 1 : 0, isContiguous, tilingPath, totalInput);
```

```cpp
    for (int64_t i = rank - 1; i >= 0; i--) {
        int64_t actualStride = strides->GetStride(i);
        if (actualStride != expectedStride) {
            cOrderContiguous = false;
            OP_LOGI(context, "DEBUG stride mismatch: dim=%ld, actual=%ld, expected=%ld, dimSize=%ld",
                    i, actualStride, expectedStride, viewShape.GetDim(i));
            break;
        }
        expectedStride *= viewShape.GetDim(i);
    }
    OP_LOGI(context, "DEBUG cOrderContiguous=%d, rank=%ld", cOrderContiguous ? 1 : 0, rank);
```

**第 3 步：到运行用户的 plog 目录按 PID 取日志。** host 侧 OP_LOGI 不进 stdout，落在**运行用户**家目录：

```bash
ls -lt ~<运行用户>/ascend/log/debug/plog/ | head -5
grep -i "DEBUG DetectContiguity" ~<运行用户>/ascend/log/debug/plog/plog-<pid>_*.log
```

本轨迹实测命中输出：

```
DEBUG stride mismatch: dim=1, actual=0, expected=1, dimSize=8
DEBUG cOrderContiguous=0, rank=2
DEBUG DetectContiguity: isView=1, isContiguous=0, tilingPath=2, totalInput=32
```

拿到 actual=0（且对每个输入都如此）即锁定 `GetInputStride` 未填充 → 误判 → 错误路径触发 → 超时，根因闭环。

**辅助口径**：跑测试二进制加 `stdbuf -o0` 关 stdout 缓冲（否则看不到进行到哪条用例）；超时判据看逐条 FAIL 文本而不是整体 exit code（124 既可能是慢也可能是逐条超时，日志里每条都有 507035 即是逐条超时）。

## 根因

（方法卡——本卡沉淀的是定位该类问题的可执行探针组合；根因实例见同批独立卡 [507035_getinputstride_unfilled_misjudge](../compilation/507035_getinputstride_unfilled_misjudge.md)：stride 未填充致 TilingKey 误判 + SyncAll 死锁。）

## 预防

- 双路径/多模板算子联调前，先把"强制单路径"开关写进 tiling 代码（临时注释块），需要时一轮重编即可二分。
- plog 找不到日志时先确认**运行用户**（ps 看属主），再按 PID 定文件；/var/log/npu 与其他用户目录大概率扑空。
- OP_LOGI 探针定位完成后记得移除（本轨迹在修复轮同步删除）。

## 坏实践

- 只看整体 exit code/总时长猜"慢 or 挂"，不逐条看 FAIL 文本。
- 在错误用户的 plog 目录里 grep 不到就断定"日志没开"。

## 验证信号

来源：迭代三联调验证会话的完整排障链。

- 第 1 步结果：强制 `tilingPath=1` 后，原全崩的 Real 用例首条即 `builtin_fp16_2d_dim-1_[4,8] [PASS] MERE=2.22e-05`（来源平台：950；证据等级：tool_observed）。
- 第 3 步结果：运行用户 plog 命中 `DEBUG stride mismatch: dim=1, actual=0, expected=1` 与 `DetectContiguity: isView=1, isContiguous=0, tilingPath=2`（多输入复现）（来源平台：950；证据等级：tool_observed）。
- 反例（方法修正）：先查 /var/log/npu（空）、另一用户目录 plog（全是旧日期文件）均未命中，按运行用户切换后才命中（来源平台：950；证据等级：tool_observed）。
- 探针有效性闭环：依据探针结论修复（stride 有效性回退）后 507035 全量消失（来源平台：950；证据等级：tool_observed）。

## 失效边界

- OP_LOGI 需要 plog 级别开启（本环境默认 debug 级可用）；若日志级别被调高，探针不落盘（未验证其他级别）。
- 强制单路径法要求路径间可独立编译执行；若故障在两路径公共段，二分不收敛（此情形第 1 步即暴露：仍崩）。
- plog 目录按运行用户隔离，容器/多用户混跑环境 PID 会复用，取**最新时间戳**文件再 grep。
- 本卡 OP_LOGI 落 plog 的结论限 CANN 9.1.0 实测环境；CANN 9.0.0-beta.2 的 opp/vendors 安装算子曾观测到 host OP_LOGI 不进 stdout 也无测试进程 plog（见 [host_op_logi_stdout_plog](../compilation/host_op_logi_stdout_plog.md)），跨版本使用前先确认日志可达性。

**未决项**：

- 无（方法在本轨迹完整闭环验证）。
