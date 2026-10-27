---
type: "Runbook"
title: "eager直调GetUserWorkspace恒返回保留区常量"
description: "Ascend950PR 实测 <<<>>> eager 直调不设置 KFC 系统工作区寄存器，kernel 内 GetUserWorkspace 恒返回 0x1000000（16MiB 保留区常量）与 launch 传入内容无关，且 <<<>>> 只有 (blockDim, dynUbuf, stream) 三个配置槽无 workspace 通道；直调 kernel 的 workspace 必须作显式 GM_ADDR 入参，或入口 SetSysWorkspaceForce 缓解。"
tags: ["task.debugging", "topic.memory", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# eager直调GetUserWorkspace恒返回保留区常量

## 现象

把注册算子转 `<<<>>>` kernel 直调时，需要确认 kernel 内 `AscendC::GetUserWorkspace`（Group 路径取用户 workspace 的官方入口）在直调 launch 下是否仍可用。写了一个上板探针：kernel 把 `GetUserWorkspace(nullptr)` 的返回值写到 GM 输出，host 侧分配 4 MiB workspace 并作为 `<<<>>>` 第二个配置参数传入，同步后把输出拷回打印：

```cpp
__global__ __aicore__ void kfc_probe2(GM_ADDR out)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint64_t> o;
    o.SetGlobalBuffer((__gm__ uint64_t*)out);
    GM_ADDR uw = AscendC::GetUserWorkspace(nullptr);
    o.SetValue(0, (uint64_t)(reinterpret_cast<uintptr_t>(uw)));
}

extern "C" void launch_probe2(GM_ADDR out, uint32_t blockDim, void* stream, GM_ADDR ws)
{
    kfc_probe2<<<blockDim, ws, stream>>>(out);
}
```

host 侧（aclInit → SetDevice → CreateStream → Malloc ws 4MiB → launch → sync → D2H）实际输出：

```
ws host ptr = 0x12004c600000
GetUserWorkspace on device = 0x1000000 (delta vs ws = -19792473882624)
```

设备侧读数 `0x1000000` 与传入 workspace 完全无关。随后尝试给 `<<<>>>` 增加第四个配置项专门传 workspace（`kfc_probe3<<<blockDim, nullptr, stream, ws>>>(out)`），直接编译失败：

```
t3.cpp:15:51: error: too many arguments to function call, expected 4, have 5
    kfc_probe3<<<blockDim, nullptr, stream, ws>>>(out);
```

即 `<<<>>>` 语法只有 `(blockDim, dynUbufSize, stream)` 三个配置槽，第二个槽是动态 UB 大小（uint32 口径，另有探针把 dynUbuf 作第二参编译通过的对照），**没有 workspace 通道**。

## 适用场景

- **工程形态**：kernel 直调（`<<<>>>` eager launch）；注册算子转直调时评估 kernel 内 `GetUserWorkspace` 依赖。
- **API 路径**：`AscendC::GetUserWorkspace` / `GetSysWorkSpacePtr` / `RESERVED_WORKSPACE`（`kernel_operator_common_impl.h`、`kernel_utils_constants.h`）；`<<<>>>` 启动语法。
- **验证过的平台与 CANN 版本**：Ascend950PR（dav_3510 类），CANN 9.2.0。与 frontmatter `platforms` 一致。
- **适用粒度**：`platform_wide`（950 系 eager 直调语义；其他平台保留区大小档位不同但机制同构——未验证，推断可迁移）。
- **路线优先级**：`recommended_fastpath`（workspace 显式入参为直调主路径；`SetSysWorkspaceForce` 为恢复框架语义的缓解手段）。
- **明确不适用 / 未验证场景**：注册/aclnn 框架 launch（runtime 会设置系统工作区）不适用本结论；`SetSysWorkspaceForce` 缓解的实际效果在本轨迹片段内未独立上板验证（见失效边界）。

## 判别方法

1. 用上面的探针 kernel + host 程序实测：若 device 读数为固定值（本机 `0x1000000`）且与 host 传入的 ws 指针差值巨大（delta 为负的 TB 级），即可判定 eager launch 未把 workspace 送达 `GetUserWorkspace` 读取的通道。
2. 静态核对机制链：`GetUserWorkspace(ws)` 返回 `GetSysWorkSpacePtr() + RESERVED_WORKSPACE`（`asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h:58`）；`RESERVED_WORKSPACE` 在 `kernel_utils_constants.h` 按 arch 分档，dav_3510 类为 `16 * 1024 * 1024`（另有 2MiB 档）。系统工作区指针由框架 launch 设置（KFC workspace 寄存器路径，`WORKSPACE_PARAM_OFFSET == 0xffffffff` 时走 `g_sysWorkspaceReserved`）；eager `<<<>>>` launch 不设置 → `GetSysWorkSpacePtr()=0` → 恒返回 `0 + 16MiB = 0x1000000`。
3. 若 kernel 需要 workspace：改为显式 `GM_ADDR workspace` 入参（Base 路径），或 kernel 入口首行 `AscendC::SetSysWorkspaceForce(workspace)` 并由 host 按 `16MiB RESERVED_WORKSPACE + workspaceSize` 分配（框架语义：`GetUserWorkspace = kfc 基址 + 16MiB`）。

## 设计决策

- Base 路径（workspace 作显式 GM_ADDR 入参）完全可用，定为直调主路径；Group 路径（kernel 内 `GetUserWorkspace`）受 eager launch 不设 KFC 寄存器限制，配 `SetSysWorkspaceForce` 缓解保留。
- host 侧分配大小取 `kReservedWorkspaceBytes(16MiB) + workspaceSize`，与框架 `workspace[0] = systemWorkspaceSize + workspaceSize` 语义对齐（真实工程 `EnsureSysWorkspace` 实现此分配）。
- 注册/框架路径下"声明大小必须含保留区、kernel 侧经 GetUserWorkspace 取用户区"的约定见 [mte_ddr_507035_workspace_reserved_region](../runtime/mte_ddr_507035_workspace_reserved_region.md)；本卡是该约定在 eager 直调形态下的对偶结论。

## 验证信号

（来自同一次直调工程搭建会话的上板探针与头文件核对，Ascend950PR / CANN 9.2.0）

- 探针实测：`ws host ptr = 0x12004c600000`，`GetUserWorkspace on device = 0x1000000 (delta vs ws = -19792473882624)`（来源平台：950；证据等级：tool_observed）。
- `<<<blockDim, nullptr, stream, ws>>>` 四配置项编译报 `error: too many arguments to function call, expected 4, have 5`（来源平台：950；证据等级：tool_observed）。
- CANN 头文件机制锚点：`kernel_operator_common_impl.h:58`（`return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;`）与 `kernel_utils_constants.h`（dav_3510 类 `RESERVED_WORKSPACE = 16 * 1024 * 1024`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `0x1000000` 是 dav_3510 类 16MiB 保留区档位下的观测值；其他 arch 档位（2MiB）数值不同但机制同构（未验证，推断可迁移）。
- `SetSysWorkspaceForce` 缓解路径在本轨迹片段内未独立上板验证（实现声明：入口首行调用 + host 16MiB+workspaceSize 分配已写入真实直调工程，效果待验，agent_reported 实现声明不计入实测）；验证前不应假设 Group 语义已恢复。
- 注册工程 / aclnn 框架 launch 下 runtime 会设置系统工作区，本卡结论不适用。

**未决项**：

- **优先项**：在 TPipe 507015 问题解决后（该问题已由 [tpipe_eager_launch_507015_aivec_magic_patch](tpipe_eager_launch_507015_aivec_magic_patch.md) 闭合），上板复现并验证 `SetSysWorkspaceForce + 16MiB+workspaceSize` 分配是否完整恢复框架 `GetUserWorkspace`/DataCopy sys-workspace 语义，把机制追到寄存器层。
- `WORKSPACE_PARAM_OFFSET` 非 0xffffffff 时（框架 launch）`get_para_base()[offset]` 路径与 eager 路径的差异未展开验证。
## 坏实践

- 指望 `<<<>>>` 第二配置参数传 workspace——该槽是 dynUbufSize，传指针会被按 uint32 截断解释。
- 直调 kernel 直接沿用注册版 kernel 的 `GetUserWorkspace` 调用而不加 `SetSysWorkspaceForce`——会拿到固定的 0x1000000 附近地址，轻则数据写飞重则越界。

## 误诊

- 初看容易怀疑"workspace 传错了/没对齐"；探针的 delta 数值（-19792473882624，TB 级负偏移）说明读数根本不来自传入指针，应直接判定通道缺失而非对齐问题。
