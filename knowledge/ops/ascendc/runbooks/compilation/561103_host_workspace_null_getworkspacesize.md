---
type: "Runbook"
title: "561103—GetWorkspaceSize阶段host缓冲为空"
description: ACLNN 两段式执行中，GetWorkspaceSize 阶段调 executor->GetWorkspaceAddr() 取 host workspace 写数据返回 null，L2 API 报 561103（ACLNN_ERR_INNER_NULLPTR）；想把 host 侧数据经 workspace 捎给 kernel 的方案在入口即不成立，走 op 属性通道。
tags: [task.debugging, topic.implementation.api_layer, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# 561103—GetWorkspaceSize阶段host缓冲为空

## 现象

在 Ascend950PR（CANN 9.1.0）上给 BenchNoHarnessGroupedMatmul 做"多输出地址传给 kernel"改造时，第一个方案是走 workspace：在 L2 `aclnnBenchNoHarnessGroupedMatmulGetWorkspaceSize` 里、调用完 L0 API（`ADD_TO_LAUNCHER_LIST_AICORE` 已触发 TilingFunc、workspace size 已定）之后，直接取 executor 的 host workspace 把 E 个输出 tensor 设备地址写进去：

```cpp
    // 调用 L0 API：Kernel 调度（输出 TensorList 由调用方预分配，直接传递）
    auto opResult = l0op::BenchNoHarnessGroupedMatmul(
        xContiguous, weightContiguous, biasContiguous,
        group_list, split_item, transpose_weight, y, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 多输出路径（split_item=0/1）：将 E 个输出 tensor 的设备地址写入 workspace
    // Kernel 从 workspace 读取这些地址，为每组创建独立的 GlobalTensor 输出
    // split_item=2/3 不需要此步骤（单 tensor 连续输出）
    if (split_item <= 1 && y != nullptr && y->Size() > 0) {
        void* wsAddr = uniqueExecutor->GetWorkspaceAddr();
        OP_CHECK(wsAddr != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GetWorkspaceAddr returned null after L0 API"),
                 return ACLNN_ERR_INNER_NULLPTR);
        uint64_t* outputAddrs = static_cast<uint64_t*>(wsAddr);
        for (uint64_t i = 0; i < y->Size(); i++) {
            const aclTensor* yTensor = (*y)[i];
            OP_CHECK_NULL(yTensor, return ACLNN_ERR_PARAM_NULLPTR);
            outputAddrs[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(yTensor->GetStorageAddr()));
        }
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
```

重新编译安装算子包后在 NPU 上跑 37 条 L0 ST 用例：23 通过、14 失败，全部是 split_item=0/1 用例，报错原文 `[FAIL] GetWorkspaceSize 失败: 561103`。注意失败集合里多了两条此前能过的用例——它们也是 split_item=0，被这个新加的 null 检查拦下。

## 适用场景

- 工程形态：aclnn 自定义算子开发（L2 GetWorkspaceSize/Execute 两段式接口），想把 host 侧已知数据（输出地址、小配置）经 workspace 传给 kernel。
- API 路径：`aclOpExecutor::GetWorkspaceAddr()/GetWorkspaceSize()`、`OP_CHECK`/`ACLNN_ERR_INNER_NULLPTR`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.x ACLNN 两段式执行模型下的通用行为，与具体算子无关。
- 路线优先级：**banned**（在 GetWorkspaceSize 阶段依赖 host workspace 可写，本轨迹客观失败；正确替代见 [多输出 TensorList 地址经 op 属性传 kernel 卡](../precision/multi_output_tensorlist_addr_op_attr.md)）。
- 明确不适用 / 未验证场景：Execute 阶段的 `GetWorkspaceAddr()` 同样观测到返回 null（见同批 507015 负结果卡），但其他 CANN 版本/补丁行为未验证。

## 判别方法

触发条件：L2 GetWorkspaceSize 内（CREATE_EXECUTOR / L0 API 调用之后）调用 `uniqueExecutor->GetWorkspaceAddr()` 并解引用。

可执行探针（对报错码定位语义）：

```bash
grep -rn "ACLNN_ERR_INNER_NULLPTR\|56110" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_errno.h
```

原始观测（节选）：

```text
# ST Real 模式输出：
[L0-037] split_item=1 with transpose and bias, fp16
  M=128 K=128 N=128 E=4 dtype=float16 split_item=1 tw=true bias=yes(same)
  [FAIL] GetWorkspaceSize 失败: 561103
...
总计: 37
通过: 23
失败: 14
# op_errno.h grep 结果（节选）：
#define ACLNN_ERR_INNER 561000
#define ACLNN_ERR_INNER_TILING_ERROR 561002
#define ACLNN_ERR_INNER_CREATE_EXECUTOR 561101
#define ACLNN_ERR_INNER_NULLPTR 561103
```

解释：561103 = `ACLNN_ERR_INNER_NULLPTR`，正是 L2 代码里 `OP_CHECK(wsAddr != nullptr, ...)` 自己抛的——即 `GetWorkspaceAddr()` 在 GetWorkspaceSize 阶段返回了 null。ACLNN 是两段式执行：`GetWorkspaceSize`（创建 executor、跑 tiling、返回 workspaceSize）→ 调用方分配 device workspace → `Execute`。executor 的 host workspace 缓冲在这个阶段尚未分配（`GetWorkspaceAddr()` 返回的缓冲在框架内部后续才就绪），不能在此阶段当可写通道用。

## 根因

`aclOpExecutor::GetWorkspaceAddr()` 在 GetWorkspaceSize 阶段（L0 API/tiling 刚结束、调用方还没分配 workspace、Execute 未开始）返回 null，此时写 host workspace 的方案从入口就不成立。单句归因：**host workspace 缓冲在 GetWorkspaceSize 阶段不可用**。

## 设计决策

- 这条路线的动机是"workspace 由框架搬到 device，正好捎带输出地址"；放弃原因即上述客观失败。后续改走 Execute 阶段 `aclrtMemcpy` 预写 device workspace（同样失败，另立卡），最终采用 op 属性通道。
- 保留的教训：给"框架内部何时就绪"类假设先加 null 检查再解引用，本卡正是因为有 `OP_CHECK` 才拿到干净的 561103 而不是段错误。

## 预防

- 需要把 host 侧数据传给 kernel 时，优先走 op 属性（`OP_ATTR` + TilingFunc 读取 + TilingData）或 kernel 参数，不要依赖 executor host workspace 的可用时机。
- 使用 `GetWorkspaceAddr()` 前一律判空；按 561xxx 前缀查 `op_errno.h` 定位语义。

## 验证信号

来源说明：以下信号来自该算子迭代二开发会话，方案一试跑后由 ST Real 模式（NPU）实测得到。

- 37 条 L0 用例 23 过 14 失败，全部失败为 split_item=0/1 且报 `[FAIL] GetWorkspaceSize 失败: 561103`（来源平台：950；证据等级：tool_observed）。
- `op_errno.h` grep 确认 561103 = `ACLNN_ERR_INNER_NULLPTR`，且为 L2 自身 `OP_CHECK(wsAddr != nullptr, ...)` 抛出路径（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定"GetWorkspaceSize 阶段"。Execute 阶段的 host workspace 可用性另行观测：同轨迹 Execute 里 `GetWorkspaceAddr()` 也返回 null，见 [507015 device workspace 预写负结果卡](507015_device_workspace_prewrite_invisible.md)。
- CANN 9.1.0 实测；其他版本未验证，推断行为一致（两段式模型未变），未验证标注。

**未决项**：

- executor host workspace 在框架内部的确切分配/就绪时机（哪个阶段之后非空）未追到 CANN 源码层；如需 workspace 通道，先做小探针实证再设计。
## 坏实践

- 不判空直接解引用 `GetWorkspaceAddr()` 返回值——会把问题变成难以定位的段错误。
- 因为"接口存在"就假设"阶段可用"——框架接口的可用时机属于隐式契约，必须实测。

## 误诊

无。
