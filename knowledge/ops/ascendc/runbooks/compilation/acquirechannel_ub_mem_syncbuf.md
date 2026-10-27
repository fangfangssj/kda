---
type: "Runbook"
title: "AcquireChannel 协议精确匹配失败的 syncBuf 判别"
description: "CommChannelBuilder::AcquireChannel 只接受 linkProtocol==protocol 的链路；当探针确认返回链路不含请求协议时，通道构建在 kernel 启动前失败并上报 syncBuf failed。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.programming_model.cooperative_groups"]
status: stable
sources: []
created_at: "2026-08-18T05:08:36Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# AcquireChannel 无 UB_MEM 链路致 syncBuf 失败

## 现象

在多 rank kernel 直调 ST 测试（qrun 在 Ascend950PR 4 卡环境跑 apace 融合通信-计算算子）的 NPU 运行阶段，4 个 rank 全部失败，退出码 255，错误为 `AllocRegAndBuildChannels syncBuf failed`。失败发生在 kernel 执行之前的 HCCL channel 建链阶段，output 目录下无 `npu_out.bin` 产出。算子以 UBMEM 模式（`ChannelMode::UBMEM`）建链，环境为 CANN 9.2.0。

## 适用场景

- **工程形态**：kernel 直调 ST 测试（host 侧 `CommChannelBuilder` 先于 kernel 建 HCCL channel，再 `<<<>>>` 直调）。
- **API 路径**：`block/aiv_comm/comm_channel_builder.h` 的 `CommChannelBuilder::AcquireChannel` / `AllocRegAndBuildChannels`；底层 HCCL `HcclRankGraphGetLinks` / `HcclChannelAcquire`。
- **验证过的平台与 CANN 版本**：Ascend950PR（DAV_3510，4 卡），CANN 9.2.0。仅在本 trace 实测。
- **适用粒度**：`operator_family`（所有经 apace `CommChannelBuilder` 建 UBMEM channel 的融合通信-计算算子通用）。
- **路线优先级**：`diagnostic_only`（本卡只描述如何把 `syncBuf failed` 定位到协议精确匹配分支，不提供协议降级路线）。
- **不适用或未验证场景**：硬件确实提供 UB_MEM 物理链路的环境不触发本问题（链路可精确匹配）；URMA（`COMM_PROTOCOL_UBC_CTP`）模式本身不要求 UB_MEM，不受影响。

## 判别方法

**触发**：qrun 跑多 rank kernel 直调，NPU 输出 `AllocRegAndBuildChannels syncBuf failed`，4 rank 全失败，退出码 255，无 `npu_out.bin` 产出。

**可执行探针**：
1. 在 `comm_channel_builder.h` 的 `AcquireChannel`（约第 204-246 行）打印 `HcclRankGraphGetLinks` 返回的 `linkNum` 与每条 `links[i].linkAttr.linkProtocol` 值。
2. 对照算子请求的 `protocol`：`ChannelMode::URMA` → `COMM_PROTOCOL_UBC_CTP`，`ChannelMode::UBMEM` → `COMM_PROTOCOL_UB_MEM`。
3. 若返回链路集合中不含与 `protocol` 相等的 `linkProtocol`，即可确认 `AcquireChannel` 会在该分支返回 false；仍需保留原始链路枚举日志，不能仅凭平台名称推断协议集合。

**原始观测**：`AcquireChannel` 的匹配逻辑（verbatim 取自 trace 内 file_read，第 226-237 行）：

```cpp
        bool linkFound = false;
        for (uint32_t i = 0; i < linkNum; i++) {
            if (links[i].linkAttr.linkProtocol == protocol) {
                channelDesc.localEndpoint = links[i].srcEndpointDesc;
                channelDesc.remoteEndpoint = links[i].dstEndpointDesc;
                linkFound = true;
                break;
            }
        }
        if (!linkFound) {
            return false;
        }
```

即遍历 `HcclRankGraphGetLinks` 返回的全部链路，要求至少一条 `linkProtocol == protocol` 精确命中，否则直接 `return false`，`HcclChannelAcquire` 不再执行。在 Ascend950PR 上，rank graph 返回的链路 `linkProtocol` 均为 `COMM_PROTOCOL_UBC_CTP`（DMA），不含 `COMM_PROTOCOL_UB_MEM`，故 UBMEM 模式下 `linkFound` 恒为 false、`AcquireChannel` 返回 false、`BuildChannels` 经 `ASC_CHECK` 短路失败、`AllocRegAndBuildChannels` 整体失败、上层报 `syncBuf failed`。

**判读**：代码可直接证明“不精确命中即返回 false”。本次探针中返回链路为 `COMM_PROTOCOL_UBC_CTP`，而请求为 `COMM_PROTOCOL_UB_MEM`，因此该次 UBMEM channel 在 host 建链阶段失败，kernel 未进入执行。该结论只覆盖实际打印出的链路集合，不能泛化为所有 Ascend950 环境均不提供 UB_MEM。

## 验证信号

- 修复前：qrun NPU 运行报 `AllocRegAndBuildChannels syncBuf failed`，4 rank 全失败，退出码 255，output 目录无 `npu_out.bin`（来源平台：950；证据等级：tool_observed）。
- `AcquireChannel` 第 226-237 行 verbatim 代码确认 `linkProtocol == protocol` 精确匹配、不命中即 `return false`、无 fallback 分支（来源平台：950；证据等级：artifact_verified）。

## 预防

遇到 `syncBuf failed` 时，在 `return false` 前打印请求协议、`linkNum` 和每条 `linkProtocol`，先确认失败发生在 host 建链层。是否允许协议替代取决于上层同步语义，本卡没有验证任何降级方案。

## 坏实践

- 在 `AcquireChannel` 不命中时静默 `return false`、不打印 `linkNum`/各 `linkProtocol`，使 `syncBuf failed` 这一高层错误完全无法定位到链路协议不匹配。
- 未打印实际链路枚举，就根据芯片型号假定某协议存在或不存在。

## 失效边界

- 本卡只覆盖 host 侧 channel 建链失败；kernel 内 `TeamBarrier` 在 UBC_CTP 下的非确定性是独立根因。
- 其他机器、拓扑和 HCCL 配置返回的协议集合未验证，必须重新执行链路枚举探针。
