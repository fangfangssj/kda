---
type: "Runbook"
title: "507035：多核 reduce SyncAll 回退单核"
description: "DAV_3510（ascend950/arch35）上 reduce 模式多核 + workspace 部分和 + SyncAll 跨核合并触发 aclrtSynchronizeStream 507035 向量核异常；硬件 SyncAll、软件 SyncAll<true>(gm,ub,usedCores)、SetBlockDim(usedCoreNum)、GM 标志轮询四种方案均失败，回退单核 reduce（usedCoreNum=1）功能正确。"
tags: [task.debugging, topic.architecture, topic.parallelism]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 507035：多核 reduce SyncAll 回退单核

## 现象

在 Ascend950PR（arch35 / DAV_3510，CANN 9.1.0）上给 cross entropy loss 类算子做优化迭代时，把 reduce 模式（sum/mean）从单核改为多核：每核算 `partialSum/partialCount` 写入 workspace GM，`SyncAll` 跨核同步，Core 0 读所有核 partial 合并输出。编译安装都正常，但一上板就崩——ST 单用例执行输出：

```
[Real] L0_002 - 基础shape(fp32+int64硬标签) reduction=mean
  [FAIL] 流同步失败: 507035
[Real] L0_003 - 基础shape(fp32+int64硬标签) reduction=sum
  [FAIL] 流同步失败: 507035
```

即 aclnn 下发成功、`aclrtSynchronizeStream` 返回 507035（ACL_ERROR_RT_VECTOR_CORE_EXCEPTION，向量核异常；`rt_error_codes.h` 确认错误码定义）。同日同版本 kernel 的 `none` 模式（多核切分、无跨核同步）完全正常（L0_001 PASS，MERE=4.54e-08），单核 reduce 也正常。当时 tiling 状态：reduce 模式 `syncCoreNum` 覆盖全部启动核，kernel 内 `syncCoreNum_ > 1` 时走 workspace + SyncAll + MergeAndOutput 路径。

排查采用二分跳段定位：注释掉逐行计算循环（`partialSum=0; partialCount=rowsThisCore`），保留 workspace 写 + SyncAll + MergeAndOutput → **仍然 507035**；再跳过 workspace 写/SyncAll/MergeAndOutput，让 core 0 直接写一个固定值到输出 → **不再崩溃**（退化为 MERE=1.00e+00 的纯精度错）。崩溃点由此锁定在"workspace 写 / SyncAll / core0 合并"路径，与计算逻辑和标签读取无关。

## 适用场景

- 工程形态：aclnn 自定义算子（AscendC kernel），reduce/聚合类输出需要跨 AIV 核合并（多核 partial + 核间同步 + 主核归并）。
- API 路径：`SyncAll()`（无参硬件版）、`SyncAll<true>(const GlobalTensor<int32_t>&, const LocalTensor<int32_t>&, int32_t usedCores)`（软件版）、`SetBlockDim`/`GetBlockNum`/`GetBlockIdx`；CANN 头文件 `kernel_operator_block_sync_intf_impl.h`（SyncAll 实现）、`kernel_utils_mode.h`（`SyncAllConfig`/`DEFAULT_SYNC_ALL_CONFIG`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510，28 AIC + 56 AIV），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该平台上"AIV 多核 + SyncAll 归并"模式的行为，与具体算子无关。
- 路线优先级：safety_fallback（单核 reduce 为已验证可用回退；多核 SyncAll 路线在本轨迹标记为不可用，直到根因澄清）。
- fallback 声明：单核 reduce 只保证功能正确，牺牲大 batch 吞吐（相对多核的代价未测量，本轨迹无 profiling 数据；仅逻辑判断"大 batch 时 reduce 性能不如多核"）。采用前接受该相对代价。
- 明确不适用 / 未验证场景：`none` 模式多核切分（无跨核同步）不受影响；AIC 侧同步、其他 NPU 架构（910/310P/A2/A3 等）未验证；CANN 其他版本未验证。

## 根因

未追到机制层：DAV_3510 上"多核 reduce + 跨核同步"组合稳定触发向量核异常（507035），四种同步方案（见下）同轨迹逐一排除，异常均复现；确切硬件错误源未定位（未采集 plog 机制层报错）。

已排除/已核实的机制线索（同步侧）：

1. CANN 9.1.0 头文件实证：DAV_3510 上 `SyncAll()` 走 `template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>`，`DEFAULT_SYNC_ALL_CONFIG = {PIPE_ALL, PIPE_ALL}`；`isAIVOnly=true` 时用 `ffts_cross_core_sync(SYNC_AIV_ONLY_ALL)` 硬件指令同步 AIV 核。
2. 软件 `SyncAll<true>(gmWorkspace, ubWorkspace, usedCores)` 实现（`SoftSyncAllImpl`）为 GM 计数器轮询（`copy_gm_to_ubuf_align_v2`/`copy_ubuf_to_gm_align_v2`），`totalBlocks = usedCores != 0 ? usedCores : get_block_num()`——传 `usedCores=syncCoreNum_` 可只等指定核数。
3. 排除项：workspace GM 地址对齐（per-core 偏移 `blockIdx*32` 字节，均 32B 对齐）、workspace 越界（大小按 syncCoreNum*64B 声明，写偏移在界内）、idle 核缓冲未初始化（idle 核走 WritePartialToWorkspace(0,0) 参与同步）逐一核对无异常。
4. 已排除的因素：逐行计算与标签读取（跳段实验）、`outBuf` 未初始化（伴生缺陷，见误诊）、DataCopyPad 参数结构（`DataCopyParams` → `DataCopyExtParams`）、V→MTE3 / MTE2→V / MTE3→V 事件同步缺失。候选方向（均未验证）：`SyncAll` 在 DAV_3510 MIX 架构上的语义/约束、是否需用 CrossBlockSetFlag/CrossBlockWaitFlag 类跨核事件替代、tiling 是否需配 SetScheduleMode 批调度（库内另有"SyncAll 须配 SetScheduleMode(BATCH_MODE_SCHEDULE)"的配对规则卡，本轨迹未尝试该方向）。

## 解决配方

四种多核同步尝试（全部失败，负知识，勿直接照抄）：

1. 硬件 `SyncAll()`，`SetBlockDim(56)` 全核启动 + idle 核写零 partial 参与同步 → 仍 507035。
2. 软件 `SyncAll<true>(syncGM, syncUbLocal, syncCoreNum_)`，workspace 扩为 `syncCoreNum * 64` 字节（前半 partial、后半 sync 标志）+ 32B syncUB → 仍 507035。
3. `SetBlockDim(usedCoreNum)` 只启动有任务的核（无 idle 核）+ 软件 SyncAll → 仍 507035。
4. 手写 `ManualSyncAll`（GM 标志位 + DataCopyPad 轮询）→ 仍 507035；期间还引入一个编译错误：`DataCopyPad(flagLocal, syncGM[i * 8], flagReadParams, {false,0,0,0})` 的 `flagLocal` 误用 `LocalTensor<float>` 而 `syncGM` 是 `GlobalTensor<int32_t>`，类型不匹配编译失败，改 `LocalTensor<int32_t>` 后通过、运行仍 507035。

可用回退（本轨迹验证通过的最终状态）——reduce 模式回退单核，tiling 侧：

```cpp
// op_host/arch35/bench_no_harness_cross_entropy_loss_tiling.cpp（最终状态）
    // 8. Multi-core split by N_total
    // - none mode: multi-core split (each core independently processes rows)
    // - sum/mean mode: single core (SyncAll has compatibility issues on DAV_3510;
    //   multi-core reduce deferred - single core ensures correctness)
    uint32_t usedCoreNum;
    uint32_t syncCoreNum;
    if (nTotal == 0) {
        usedCoreNum = 1;
        syncCoreNum = 1;
    } else if (reductionMode != 0) {
        // Reduce mode: single core for correctness
        usedCoreNum = 1;
        syncCoreNum = 1;
    } else {
        usedCoreNum = std::min(static_cast<uint32_t>(coreNum), static_cast<uint32_t>(nTotal));
        if (usedCoreNum == 0) {
            usedCoreNum = 1;
        }
        syncCoreNum = usedCoreNum;
    }
```

```cpp
    // 15. Set Workspace size (for sum/mean cross-core merge)
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    if (reductionMode != 0) {
        // Single-core reduce: workspace for partial sum + sync flags (64 bytes)
        currentWorkspace[0] = static_cast<size_t>(syncCoreNum) * 64;
    } else {
        currentWorkspace[0] = 0;
    }
```

kernel 侧保留单核快路径（`syncCoreNum_ == 1` 时不走 workspace/SyncAll，直接 `OutputScalar`）：

```cpp
    // Single-core fast path: skip workspace and SyncAll
    if (syncCoreNum_ == 1) {
        float result;
        if (reduction_ == 1) {
            result = partialSum;
        } else {
            if constexpr (IS_HARD_LABEL) {
                result = (partialCount > 0) ? partialSum / static_cast<float>(partialCount) : 0.0f;
            } else {
                result = (nTotal_ > 0) ? partialSum / static_cast<float>(nTotal_) : 0.0f;
            }
        }
        OutputScalar(result);
        return;
    }
```

判别方法：

1. 错误形态：aclnn 返回 0、`aclrtSynchronizeStream` 返回 507035，且只有"多核 + 跨核同步"的执行路径触发、同 kernel 的无同步多核路径（none）与单核路径正常 → 锁定跨核同步路径。
2. 快速二分：把 tiling 的 reduce 分支 `usedCoreNum/syncCoreNum` 强制为 1（如上），其余不动——507035 消失即为同步路径问题。
3. 若要继续追机制：采集 plog（`~/ascend/log/debug/plog/`）看向量核异常的错误码与出错指令，本轨迹未做（遗留）。

## 设计决策

- 为什么回退单核而不是继续换同步原语：四种方案（硬件/软件/半核/手写轮询）同轨迹全部失败，继续投入的边际收益低，正确性优先；任务边界明确允许"多核修不了就保单核功能正确"。
- 为什么 none 模式保留多核：none 无跨核依赖，多核切分实测正常，没必要陪葬。
- 单核快路径放在 kernel（`syncCoreNum_ == 1` 分支）而不是删掉多核代码：保留多核合并代码便于后续在根因澄清后复活。
- tiling 里保留 `usedCoreNum/syncCoreNum` 可一键切单核的开关（如本卡配方），作为同步类异常的应急回退通道。
- 当前方案路线档位：safety_fallback（见 fallback 声明）。

## 预防

- 在 DAV_3510 上做"多核 partial + SyncAll 归并"设计前，先用最小用例验证 SyncAll 路径可用，再铺开实现——四种方案各写一轮、每轮都完整编译安装，排障成本高。
- 507035 出现时先做路径二分（有无跨核同步 / 单核多核），不要先怀疑数据 layout，也不要在未定位机制的 vector core 异常上逐个叠加事件同步"碰运气"（V→MTE3、MTE2→V、MTE3→V 依次试——本轨迹中全部无效）。
- 507035 出现后先反复调整 workspace 大小/对齐/启动核数而不采集 plog 定位错误指令，是本轨迹第 2~4 轮的教训。

## 验证信号

来源说明：以下为该算子迭代二/三排障与联调验证中的实测（C++ ST，Real 模式，单用例进程）。

- 修复前：mean/sum 反复 `[FAIL] 流同步失败: 507035`（含 outBuf 修复、DataCopyExtParams、MTE2→V 同步等中间修补之后依旧）（来源平台：950；证据等级：tool_observed）。
- 二分定位：跳过计算保留"workspace 写+SyncAll+合并"仍 507035；跳过该路径（core0 直写固定值）不再崩溃、退化为 MERE=1.00e+00 精度错（来源平台：950；证据等级：tool_observed）。
- 硬件 SyncAll 多核方案：L0_002 `[FAIL] 流同步失败: 507035`（来源平台：950；证据等级：tool_observed）。
- 软件 `SyncAll<true>(gm,ub,usedCores)` 方案：同一用例仍 `[FAIL] 流同步失败: 507035`（来源平台：950；证据等级：tool_observed）。
- `SetBlockDim(usedCoreNum)` + 软件 SyncAll：仍 507035（来源平台：950；证据等级：tool_observed）。
- ManualSyncAll GM 轮询：仍 507035（来源平台：950；证据等级：tool_observed）。
- 单核化后：L0_001（none）、L0_002（mean）、L0_003（sum）单用例全 PASS；26/26 用例通过（`=== Results: PASS=26 FAIL=0 ===`）；mean PASS MERE=8.96e-08（来源平台：950；证据等级：tool_observed）。
- 迭代三重新启用多核 reduce（tiling 去掉单核限制）→ L0_002 立即复现 507035；在其 WritePartialToWorkspace 后追加 MTE3→V 同步仍无效；回退单核后恢复 PASS——同根因第二次独立复现（来源平台：950；证据等级：tool_observed）。
- 最终全量：105/105 ST + 74/74 UT 通过，mean/sum 语义正确（含全 ignore 分母为 0 返回 0 等边界）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定"DAV_3510 / ascend950 / CANN 9.1.0 上的本 kernel 模式"：是同轨迹四种方案的反例结论，不是"该平台 SyncAll 不可用"的普适结论——其他算子的 SyncAll 用法（如不同 pipe 配置、AIC 侧参与）未测试。
- 单核回退的性能代价未量化（无多核 vs 单核对比数据）；大 batch reduce 的吞吐损失需按具体 shape 评估。
- 未验证场景：`CrossBlockSetFlag/CrossBlockWaitFlag` 替代方案（本轨迹只读了 API 名未实现）；tiling 侧 `SetScheduleMode(BATCH_MODE_SCHEDULE)` 配对（库内另有该配对规则卡，本轨迹未尝试）；CANN 升级后行为变化未测。

**未决项**：

- **优先项**：构造最小多核 reduce 用例，采集 plog 错误码/出错指令，把 507035 追到机制层（SyncAll 在 DAV_3510 MIX 架构的语义/约束、workspace GM 对齐要求、CrossBlockSetFlag/CrossBlockWaitFlag 与 tiling SetScheduleMode(BATCH_MODE_SCHEDULE) 两个替代/配对方向的可行性），再恢复多核 reduce。
- 单核 vs 多核 reduce 的相对性能代价未测量（无 profiling 数据）；若根因澄清并修复多核路径，需成对给出多核 vs 单核的相对性能（同算子内相对对比，标注参照路线）。
## 坏实践

- 在未验证 SyncAll 可用性前，把多核 reduce 的 workspace 布局、idle 核参与、软件同步全部实现完才上板。
- 在未定位机制的 vector core 异常上逐个叠加事件同步"碰运气"。
- 把 507035 当"偶发硬件问题"重跑放行——本轨迹中该错误完全确定性地随缓冲深度/同步配置出现/消失。

## 误诊

- 曾判断"硬件 SyncAll 等待全部物理 AIV 核（含未启动核）导致挂起超时报 507035"：按此假设做了"全核启动 + idle 核参与"与"只启有任务核 + 软件 SyncAll"两种针对性方案，均失败，假设未获支持（假设为 agent 归因，反证为实测）。
- 一度把崩溃归因于 `outBuf` 未初始化（`outBuf` 只在 `reduction_==0` 时 `pipe.InitBuffer`，而 reduce 模式的 MergeAndOutput 与空输入分支都用了 `outBuf.Get<T>()`）——该缺陷真实存在并已修复（改为无条件 `pipe.InitBuffer(outBuf, outBatchSize_ * sizeof(T))`），但修完 507035 依旧：它是伴生缺陷，不是崩溃根因。
