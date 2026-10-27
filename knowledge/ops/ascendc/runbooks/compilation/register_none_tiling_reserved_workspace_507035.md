---
type: "Runbook"
title: "507035 host tiling workspace 缺 RESERVED"
description: "Ascend950 kernel 侧 507035 向量核异常，根因为 host tiling 把 workspace 大小设成 2MB+用户量而框架生成的 wrapper 把 usrWorkspace 偏移 16MB RESERVED_WORKSPACE，写穿分配缓冲区。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.implementation.memory_management"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 507035 host tiling workspace 缺 RESERVED

## 现象

在 Ascend950（NpuArch dav-3510，arch 3510，CANN 9.0.0-beta.2）上为 `bn_training_update_custom` 算子实现 Branch-4 kernel（`op_kernel/arch35/bn_training_update_custom_kernel.h`，使用 `REGISTER_NONE_TILING`）。`./build.sh bn_training_update_custom_package` 编译通过（退出码 0），但 `tests/aclnn && ./run.sh` 跑 18 个 ST 用例全部失败：`aclrtSynchronizeStream` 返回错误码 **507035（向量核异常，VEC_ERROR）**，0 passed。崩溃对所有 18 个用例都发生（不分 shape/dtype），说明是系统性问题而非边界问题。

## 适用场景

- **工程形态**：aclnn 算子开发，`REGISTER_NONE_TILING` 模式（无 TilingFunc 注册、host 仅在 `GetWorkspaceSize` 设 `workspaces[0]`）。
- **API 路径**：host `GetWorkspaceSize` → `workspaces[0]` → 框架生成的 kernel wrapper → `SetSysWorkspaceForce(workspace)` + `usrWorkspace = workspace + AscendC::RESERVED_WORKSPACE` → kernel `workspace` 形参。
- **验证过的平台与 CANN 版本**：Ascend950 / dav-3510 / CANN 9.0.0-beta.2（`RESERVED_WORKSPACE = 16MB`）。
- **适用粒度**：`operator_family`——所有在 dav-3510 上用 `REGISTER_NONE_TILING` 且 kernel 写 workspace GM 的 broadcast/stateful 算子。
- **路线优先级**：`recommended_fastpath`——host `GetWorkspaceSize` 必须让 `workspaces[0] = RESERVED_WORKSPACE(平台常量) + 用户数据量`。
- **不适用或未验证场景**：用 `REGISTER_TILING` 且 TilingFunc 已显式处理 workspace 的算子未验证；其他 arch（dav_c100=2MB、dav_c220、dav_m200 等）的 `RESERVED_WORKSPACE` 值不同，须按目标 arch 查 `kernel_utils_constants.h` 取值，不得套用 16MB。

## 根因

`REGISTER_NONE_TILING` 模式下，框架生成的 kernel wrapper 调用 `SetSysWorkspaceForce(workspace)` 把整个 workspace 起始段留作系统保留区，再计算 `usrWorkspace = workspace + AscendC::RESERVED_WORKSPACE` 并把 `usrWorkspace` 作为 `workspace` 形参传给 kernel。kernel 把 multiplier/addend 写入 `workspace[0..2C)`（即 `usrWorkspace` 偏移处）。但 host tiling 把 `workspaces[0]` 设成 `2*1024*1024 + 2*C*sizeof(float)`（约 2MB），而 arch 3510 的 `RESERVED_WORKSPACE = 16*1024*1024`（16MB）。`usrWorkspace = workspace + 16MB` 落在仅有 ~2MB 的 buffer 末尾之外 ~14MB，kernel 写 multiplier/addend 时 GM 越界 → 507035 向量核异常。kernel 代码本身正确（写到 `workspace` 形参即用户段），bug 在 host tiling 用了 2MB 而非 16MB。

机制链（逐字 CANN 头文件，dav_c310 对应 arch 3510）：

```cpp
// /home/developer/Ascend/cann-9.0.0-beta.2/x86_64-linux/asc/impl/basic_api/dav_c310/kernel_operator_common_impl.h
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr),
        { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
    return workspace;
#else
    (void)(workspace);
    return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;
#endif
}

__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr),
        { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
#if (WORKSPACE_PARAM_OFFSET == 0xffffffff)
#if defined(__NPU_DEVICE__)
    __set_kfc_workspace_addr(workspace);
#else
    g_sysWorkspaceReserved = workspace;
#endif
#endif
#endif
}
```

`RESERVED_WORKSPACE` 常量（`kernel_utils_constants.h`）：

```cpp
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
```

## 解决配方

修复点在 host tiling 的 `GetWorkspaceSize`：把 `kReservedWorkspace` 从 2MB 改成 16MB（arch 3510 的 `RESERVED_WORKSPACE`）。

before（host tiling，错误——用了 2MB，应是 16MB）：

```cpp
kReservedWorkspace = 2 * 1024 * 1024;  // WRONG: should be 16MB for dav-3510
ws[0] = kReservedWorkspace + 2 * C * sizeof(float);
```

after（修复）：

```cpp
kReservedWorkspace = 16 * 1024 * 1024;  // RESERVED_WORKSPACE for dav-3510 (arch 3510)
ws[0] = kReservedWorkspace + static_cast<size_t>(2) * static_cast<size_t>(derived.C) * sizeof(float);
```

`ws[0]` = 16MB（系统保留）+ 2·C·4 字节（kernel 实际写入的 multiplier[0..C) + addend[0..C)）。框架 wrapper 计算 `usrWorkspace = workspace + 16MB` 后落在 buffer 内，kernel 写 `usrWorkspace[0..2C)` 不再越界。

## 验证信号

- CANN dav_c310 `GetUserWorkspace` 返回 `GetSysWorkSpacePtr() + RESERVED_WORKSPACE`、`SetSysWorkspaceForce` 设置 `g_sysWorkspaceReserved`——逐字 CANN 头文件，证明 wrapper 把 user workspace 偏移 RESERVED_WORKSPACE（来源平台：Ascend950；证据等级: tool_observed）
- `RESERVED_WORKSPACE = 16 * 1024 * 1024`（arch 3510 / 5102）——逐字 CANN `kernel_utils_constants.h`（来源平台：Ascend950；证据等级: tool_observed）
- host tiling 错误值 `kReservedWorkspace = 2 * 1024 * 1024`——逐字 host tiling 源（来源平台：Ascend950；证据等级: tool_observed）

## 预防

- 在 host `GetWorkspaceSize` 中显式 `#include` 并引用 `AscendC::RESERVED_WORKSPACE`（或按 `__NPU_ARCH__` 取平台常量），不要硬编码 2MB/16MB。
- 凡用 `REGISTER_NONE_TILING` 且 kernel 写 workspace GM 的算子，`workspaces[0]` 必须含 `RESERVED_WORKSPACE + 用户数据量`；review 时单独查这条。
- 跨 arch 迁移时先查目标 arch 的 `RESERVED_WORKSPACE`（dav_c100=2MB、dav_c310=16MB），不可照搬。

## 失效边界

- 仅 Ascend950（arch 3510）实测；其他 dav 平台未验证，推断可迁移但须替换为该 arch 的 `RESERVED_WORKSPACE` 常量值——未验证，推断可迁移。
- 用 `REGISTER_TILING`（TilingFunc 模式）时 workspace 管理路径不同，本卡不适用。

## 坏实践

- 把 `RESERVED_WORKSPACE` 当作可硬编码的 2MB 常量跨平台复用（dav_c100=2MB，dav_c310=16MB，值不同）。
- kernel 侧绕开 `workspace` 形参、直接 `GetUserWorkspace(workspace)` 取地址写——本卡根因不在 kernel，kernel 写 `workspace` 形参即用户段是正确的；修 kernel 不修 host 无效。

## 误诊

前序开发者曾把 `kReservedWorkspace` 设为 `2MB + 2*C*4`（仍是 2MB 量级），仍未解决崩溃——因为 2MB 远小于 16MB。正确做法是改成 16MB，不是在 2MB 的基础上加用户量。