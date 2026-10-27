---
type: "Runbook"
title: "507015—Execute预写workspace对kernel不可见"
description: ACLNN Execute 里先 aclrtMemcpy 预写 device workspace 再 CommonOpExecutorRun，memcpy 返回 0（数据已入显存）但 kernel 读不到该地址数组，流同步失败 507015；host workspace 在 Execute 阶段同样为 null。绕开框架私自预写框架管理的缓冲没有可见性保证。
tags: [task.debugging, topic.implementation.api_layer, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# 507015—Execute预写workspace对kernel不可见

## 现象

在 Ascend950PR（CANN 9.1.0）上继续 BenchNoHarnessGroupedMatmul 的"多输出地址传 kernel"改造：既然 GetWorkspaceSize 阶段 host workspace 是 null（561103，见同批卡），第二方案改为两步——GetWorkspaceSize 里把 E 个输出 tensor 设备地址存进**全局变量**，Execute 里先 `aclrtMemcpy` 写进 device workspace 再调 `CommonOpExecutorRun`：

```cpp
// ---- GetWorkspaceSize 内（L0 API 调用后）----
    // 多输出路径（split_item=0/1）：保存 E 个输出 tensor 的设备地址到全局变量
    // 在 Execute 阶段通过 aclrtMemcpy 写入 device workspace
    // （GetWorkspaceSize 阶段 workspace host buffer 可能尚未分配）
    if (split_item <= 1 && y != nullptr && y->Size() > 0) {
        g_outputAddrs.clear();
        g_outputAddrs.reserve(y->Size());
        for (uint64_t i = 0; i < y->Size(); i++) {
            const aclTensor* yTensor = (*y)[i];
            OP_CHECK_NULL(yTensor, return ACLNN_ERR_PARAM_NULLPTR);
            g_outputAddrs.push_back(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(yTensor->GetStorageAddr())));
        }
        g_needWriteOutputAddrs = true;
    } else {
        g_needWriteOutputAddrs = false;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
```

```cpp
// ---- Execute 内（含 host workspace 优先 + device memcpy 回退调试版）----
extern "C" aclnnStatus aclnnBenchNoHarnessGroupedMatmul(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
{
    L2_DFX_PHASE_2(aclnnBenchNoHarnessGroupedMatmul);

    // 多输出路径：将输出 tensor 设备地址写入 workspace
    if (g_needWriteOutputAddrs && !g_outputAddrs.empty() && executor != nullptr) {
        void* hostWs = executor->GetWorkspaceAddr();
        uint64_t hostWsSize = executor->GetWorkspaceSize();
        fprintf(stderr, "[L2-Execute] addrs=%zu, hostWs=%p, hostWsSize=%lu, devWs=%p, devWsSize=%lu\n",
                g_outputAddrs.size(), hostWs, hostWsSize, workspace, workspaceSize);
        if (hostWs != nullptr && hostWsSize >= g_outputAddrs.size() * sizeof(uint64_t)) {
            size_t bytes = g_outputAddrs.size() * sizeof(uint64_t);
            std::memcpy(hostWs, g_outputAddrs.data(), bytes);
            g_needWriteOutputAddrs = false;
        } else if (workspace != nullptr &&
                   workspaceSize >= g_outputAddrs.size() * sizeof(uint64_t)) {
            size_t bytes = g_outputAddrs.size() * sizeof(uint64_t);
            auto ret = aclrtMemcpy(workspace, bytes, g_outputAddrs.data(), bytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE);
            fprintf(stderr, "[L2-Execute] aclrtMemcpy to device, ret=%d, bytes=%zu\n", ret, bytes);
            if (ret != ACL_SUCCESS) {
                OP_LOGE(ACLNN_ERR_INNER, "aclrtMemcpy output addrs failed: %d", ret);
                return ACLNN_ERR_INNER;
            }
            g_needWriteOutputAddrs = false;
        } else {
            OP_LOGE(ACLNN_ERR_INNER, "Execute: no valid workspace for output addrs");
            return ACLNN_ERR_INNER;
        }
    }

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
```

配套地，TilingFunc 把 workspace 声明成 `E*8 字节（32 对齐）+ sysWsSize`，TilingData 带 `outputAddrOffset=0`，kernel 从 `workspace + outputAddrOffset` 读地址数组。重装后在 NPU 上跑 ST：split_item=0/1 用例不再报 561103，但子进程输出里出现 `[FAIL] 流同步失败: 507015`（`aclrtSynchronizeStream` 失败，即 kernel 已下发但执行失败），14 条 split_item=0/1 用例仍然全挂。

## 适用场景

- 工程形态：aclnn 自定义算子开发（两段式 GetWorkspaceSize/Execute），想借 device workspace 给 kernel 捎带 host 侧数据。
- API 路径：`aclrtMemcpy(ACL_MEMCPY_HOST_TO_DEVICE)`、`aclOpExecutor::GetWorkspaceAddr()`、`CommonOpExecutorRun`、`aclrtSynchronizeStream`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.x ACLNN 执行框架的 workspace 生命周期行为，与算子无关。
- 路线优先级：**banned**（Execute 里预写 device workspace 的路线客观失败；正确替代见 [多输出 TensorList 地址经 op 属性传 kernel 卡](../precision/multi_output_tensorlist_addr_op_attr.md)）。
- 明确不适用 / 未验证场景：kernel 确实需要大块初始化数据、且走框架认可通道写入 workspace 的场景不属于本卡（本卡针对"绕过框架私自预写"）。

## 判别方法

触发条件：Execute 里先 `aclrtMemcpy(workspace, ...)` 再 `CommonOpExecutorRun(...)`，kernel 从 workspace 头部读自定义数据。

可执行探针（区分 host/device workspace 状态）：

```cpp
void* hostWs = executor->GetWorkspaceAddr();
uint64_t hostWsSize = executor->GetWorkspaceSize();
fprintf(stderr, "[L2-Execute] addrs=%zu, hostWs=%p, hostWsSize=%lu, devWs=%p, devWsSize=%lu\n",
        g_outputAddrs.size(), hostWs, hostWsSize, workspace, workspaceSize);
```

原始观测（ST 子进程 stderr）：

```text
[L2-Execute] addrs=4, hostWs=(nil), hostWsSize=16777728, devWs=0x12004c600000, devWsSize=16777728
[L2-Execute] aclrtMemcpy to device, ret=0, bytes=32
...
[FAIL] 流同步失败: 507015
```

解释：Execute 阶段 host workspace 仍为 `(nil)`（`GetWorkspaceAddr()` 在 Execute 也拿不到可用缓冲）；device workspace 指针有效且 `aclrtMemcpy` 返回 0（32 字节 = 4 个地址确实写进了 device 显存），但随后 kernel 读该区域得不到写入的地址数组，按无效地址输出，流同步以 507015 失败。合理机制（推断）：`CommonOpExecutorRun` 在下发 kernel 前会按 executor 自身状态整理/覆盖 workspace 内容，Execute 入口处的手工预写不保证存活到 kernel 执行。

## 根因

Execute 阶段对 device workspace 的手工预写不能到达 kernel 可见状态——`CommonOpExecutorRun` 接管 workspace 后 kernel 读到的不是预写数据（覆盖机制未读到框架源码，失败信号客观）。单句归因：**device workspace 的内容所有权属于 CommonOpExecutorRun 之后的框架搬运，Execute 入口预写无效**。

## 设计决策

- 该方案是在 561103（host workspace 不可用）之后的第二尝试，动机是"device workspace 指针在 Execute 参数里明摆着，直接 memcpy 进去"；放弃原因：memcpy 成功但 kernel 读不到，507015。
- 全局变量暂存（`g_outputAddrs`）只是让数据跨过两个阶段的手段，本身不是失败点；最终方案里随属性通道一并删除。
- 最终采用 op 属性通道（`OP_ATTR` + `GetListInt(3)` + TilingData），不再触碰 workspace 生命周期。

## 预防

- 不要在 Execute 里"抢跑"写 workspace：workspace 从分配、内容填充到回收都属于框架（`CommonOpExecutorRun`）管辖，绕过它写内容没有可见性保证。
- kernel 需要的少量 host 侧数据走 op 属性 / TilingData；大块数据走独立 device buffer 参数（如有）。
- 排查此类问题时用 `fprintf(stderr, "hostWs=%p ...")` 直接打印两段 workspace 状态，比猜快得多。

## 验证信号

来源说明：以下信号来自该算子迭代二开发会话，方案二落地后由 ST Real 模式（NPU）实测得到。

- 14 条 split_item=0/1 用例报 `[FAIL] 流同步失败: 507015`（561103 消失，错误形态改变）（来源平台：950；证据等级：tool_observed）。
- Execute 谋试探针输出 `hostWs=(nil), hostWsSize=16777728, devWs=0x12004c600000, aclrtMemcpy ret=0 bytes=32`——host workspace 在 Execute 仍不可用，memcpy 本身成功（来源平台：950；证据等级：tool_observed）。
- 该路线废弃、改属性通道后同批用例全过（见修复卡的 37/37 等信号），形成成对回退信号（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "CommonOpExecutorRun 覆盖/丢弃预写数据"的机制是按失败信号做的推断（未读框架实现），边界结论不依赖该推断成立：客观事实是"预写 + CommonOpExecutorRun 组合下 kernel 读不到"。
- 若未来版本框架提供官方的 workspace 预填 API，本卡 banned 结论相应失效（未验证）。
- 本卡不涉及 workspace 大小声明错误类问题（那是另一类根因）。

**未决项**：

- 复现并追根因到机制层：读 CANN 侧 `CommonOpExecutorRun` 实现，确认它对 device workspace 的写入/覆盖行为——本卡机制解释目前是推断，列为下个 agent 优先项。
## 坏实践

- 在框架管理的缓冲上私自预写并指望它存活到 kernel——两次尝试（host 阶段、device 阶段）都失败，不要第三次。
- 用 `aclrtMemcpy` 返回 0 当作"kernel 能读到"的证据——它只证明写进了显存，不证明框架不会再动这块内存。

## 误诊

- 无（507015 曾被怀疑是"输出地址本身无效"，但测试里输出 tensor 均为 `aclrtMalloc` 正常分配且 `GetStorageAddr()` 与创建参数一致，排除后定位到 workspace 通道）。
