---
type: "Runbook"
title: "507035 MTE DDR 越界：workspace 须含 16MB 保留区"
description: "tiling 声明的 workspace 大小若不含平台保留区（Ascend950 为 16MB），kernel 对用户工作区的 MTE 写会落在分配范围外，报 507035 向量核异常（plog 错误码 95 MTE DDR 越界）。复现增量：另一 trace 复现同机制——MIX→纯 AIV 重写时把 GetWorkspaceSizes 总量精简为仅用户 workspace 后写 workspace 即 507035 errcode 95，最小 256 字节写也复现；可移植修复=总量加回 GetLibApiWorkSpaceSize() 系统工作区（勿手写 16MB 常量）。 复现增量：第三 trace（CANN 9.1.0）形态——tiling 自定 2MB 保留常量 vs 框架 16MB，全核 aivec 95 + 宿主段错误 139；附逐架构 RESERVED_WORKSPACE 常量表与 kernel 二进制文件名哈希不变判别。"
tags: [task.debugging, topic.memory, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-14T02:36:53Z'
updated_at: '2026-09-17T03:21:08Z'
platforms: ["950"]
---

# 507035 MTE DDR 越界：workspace 须含 16MB 保留区

## 现象

在 Ascend950PR（CANN 9.2.0）上给自定义算子 BenchHarnessScatter 新增"fp32 workspace 累加路径"后，kernel 一启动就崩：自建 aclnn 诊断程序里 `aclnnBenchHarnessScatter(...)` 本身返回 0（下发成功），但 `aclrtSynchronizeStream` 返回 507035，输出全 0。用 8 元素最小用例（wsSize=32 字节）和 1000 元素用例（wsSize=4000 字节）复现，全部同样崩溃；同版本 kernel 的原生路径（不碰 workspace）sync ret=0 正常。错误码查证：`rt_error_codes.h:126: #define ACL_ERROR_RT_VECTOR_CORE_EXCEPTION 507035 // vector core exception`。plog（`~/ascend/log/debug/plog/`）中的关键报错原文：

```
error code = 95, ... errcode:(95) errorStr: The DDR address of the MTE instruction is out of range.
subErrType: 0x4.  mte error info: 0x201a4, ... mte error: 0x80000000
```

当时 tiling 里按用户区大小声明工作区（`workspaces[0] = dataNumel * sizeof(float)`），kernel Init 里直接 `gmAcc_.SetGlobalBuffer((__gm__ float*)workspace)` 用原始 `workspace` 参数作基址。

## 适用场景

- 工程形态：aclnn 自定义算子（AscendC kernel + host TilingFunc），kernel 需要用户 workspace（中间累加缓冲等）并在其中做 MTE（DataCopyPad）读写。
- API 路径：host 侧 `context->GetWorkspaceSizes(1)`（TilingFunc 内声明工作区大小）→ 框架分配 → kernel 入口第 5 参 `workspace`；kernel 侧 `AscendC::GetUserWorkspace(workspace)`、`kernel_utils_constants.h` 的 `RESERVED_WORKSPACE` 常量。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510 / dav_3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该平台上任何使用用户 workspace 的 aclnn 算子，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不使用 workspace 的算子（`workspaces[0]=0`，无此问题）；其他 NPU 架构（`RESERVED_WORKSPACE` 数值不同，如 arch 1001 为 2MB——机制相同但须按目标架构常量声明，未验证）；以 `SetSysWorkSpacePtr`/`GetSysWorkSpacePtr` 显式管理系统工作区的特殊用法（本轨迹未涉及）。

## 根因

tiling 通过 `GetWorkspaceSizes(1)` 声明的工作区大小必须覆盖"系统保留区 + 用户区"，而 kernel 侧可写的用户区基址 = 工作区基址 + `RESERVED_WORKSPACE`（Ascend950/arch3510 上为 16MB）；只声明用户大小会让 kernel 的 workspace 写落在调用方分配范围之外，触发 MTE 指令 DDR 地址越界。

机制链（全部为本会话实测证据）：

1. 框架布局：plog 中 host 传入的工作区基址 args[4]=0x12000001b000；kernel 内打印（按 32 位高低半字）workspace 参数 = 0x1200_0101_b000，两者相差 0x1000000 = 16MB。即 kernel 拿到的指针已经偏移到"基址+16MB"的用户区位置，而调用方缓冲区只按声明大小分配。
2. `kernel_utils_constants.h`（CANN 9.2.0，按 `__NPU_ARCH__` 分档）：`// workspace system reserve 16MB`，arch3510 为 `constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;`（arch 1001 为 2MB，其余架构各有常量）。
3. `GetUserWorkspace(workspace)` 的实现即 `GetSysWorkSpacePtr() + RESERVED_WORKSPACE`——用户区永远在保留区之后。
4. 声明 `workspaces[0] = dataNumel*4`（只含用户大小）时，对 `gmAcc_[0]` 写哪怕 1 个 float（4 字节）都触发 507035；把声明改为 `16MB + dataNumel*4` 后同一写操作正常、输出非零。写入越界是异步暴露的：WaitFlag 正常返回，崩溃浮现在其后的 `SyncAll()` 或 `PipeBarrier<PIPE_ALL>`（任何排空流水的屏障都会检测到 MTE 异常），plog 的错误码 95 指明是 MTE DDR 越界而非同步原语本身的问题。

## 解决配方

修复前（tiling，只声明用户大小；kernel，直接用原始 workspace 参数）：

```cpp
    // tiling: bench_harness_scatter_tiling_arch35.cpp
    // 7. Set workspace: fp32 accumulation buffer for fp16/bf16 add/multiply.
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = useFp32Acc ? (static_cast<size_t>(dataNumel) * sizeof(float)) : 0;
    }
```

```cpp
    // kernel Init: bench_harness_scatter_kernel.h
    if (td_->useFp32Acc) {
        gmAcc_.SetGlobalBuffer((__gm__ float*)workspace);   // 错：直接用系统工作区基址
    }
```

修复后（声明含 16MB 保留区；kernel 用 GetUserWorkspace 取用户区）：

```cpp
    // tiling: bench_harness_scatter_tiling_arch35.cpp
    // 7. Set workspace: fp32 accumulation buffer for fp16/bf16 add/multiply.
    // The kernel accesses the USER workspace via GetUserWorkspace(), which is at
    // (sys_workspace_base + RESERVED_WORKSPACE).  RESERVED_WORKSPACE = 16MB on
    // Ascend950 (arch 3510), so the declared size must cover 16MB reserved +
    // the user buffer, else the user-portion write lands beyond the allocation
    // → vector-core exception (507035 / "MTE DDR out of range").
    constexpr int64_t kReservedWorkspace = 16 * 1024 * 1024;  // RESERVED_WORKSPACE, arch 3510
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = useFp32Acc
            ? (static_cast<size_t>(kReservedWorkspace) + static_cast<size_t>(dataNumel) * sizeof(float))
            : 0;
    }
```

```cpp
    // kernel Init: bench_harness_scatter_kernel.h
    if (td_->useFp32Acc) {
        // The `workspace` param is the system-workspace base; the user workspace
        // (the bytes the host declared via GetWorkspaceSizes) lives at an offset
        // and must be obtained via GetUserWorkspace().  Using the raw `workspace`
        // pointer writes into reserved system memory → vector-core exception.
        GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
        gmAcc_.SetGlobalBuffer((__gm__ float*)userWs);
    }
```

判别 / 定位方法（本会话走通的完整链路，可复用）：

1. 最小化复现：写只含一个极小用例的独立 repro（如 8 元素 bf16 add），确认 `aclnn` 返回 0 而 `aclrtSynchronizeStream` 返回 507035——错误在 kernel 执行期而非下发期。
2. 查错误码语义：`grep -rn "507035" $ASCEND_HOME_PATH/include/acl/error_codes/rt_error_codes.h` → `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`。
3. 读 plog 拿机制层报错：plog 目录为 `~/ascend/log/debug/plog/`（`/home/<user>/ascend/log/debug/plog`），取最新 `plog-*_<pid>_*.log`，找 `error code =` 行——本例为 95 "The DDR address of the MTE instruction is out of range"，直接指向 MTE 写越界。
4. kernel 内打印指针定基址：`AscendC::PRINTF` 的 `%lx` 不可靠，把 64 位地址拆成高低 32 位打印，与 plog args 里的基址对比，确认偏移量（本例 +0x1000000=16MB）。PRINTF 在 TTK 下的输出路由与收尾移除纪律见 [AscendC PRINTF 路由卡](ascendc_printf_plog_routing.md)。
5. 二分写入点：临时把 MTE3 写长度改成 1 个元素仍崩 ⇒ 与长度无关，是基址本身越界；对照原生路径（不写 workspace）正常 ⇒ 锁定 workspace。
6. 查平台常量：`grep -n "RESERVED_WORKSPACE" $ASCEND_HOME_PATH/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h`，按 `__NPU_ARCH__` 取值；再对照 `GetUserWorkspace` 实现（`kernel_operator_swap_mem_intf.h`：`return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;`）。

## 设计决策

- 为什么声明大小里手动加 16MB 而不是指望框架自动补：实测只声明用户大小时写越界崩溃、补上 16MB 后立即恢复，证明该 CANN 版本（9.2.0）的分配就是按声明值来的；16MB 是 arch3510 的 `RESERVED_WORKSPACE` 常量，写死时必须带架构注释，跨架构移植时按常量重取。
- 为什么 kernel 侧改用 `GetUserWorkspace(workspace)`：语义正确（用户区 = 系统区 + 保留量），且与本会话观察一致（`GetUserWorkspace` 返回值与原始 workspace 参数数值相同，因为后者已偏移 16MB；但显式调用使意图明确并兼容框架行为变化）。
- 排障顺序上先怀疑 nullptr 工作区（wsSize=0）再怀疑指针错位：本会话先实测 `GetWorkspaceSize` 返回的 wsSize 非零且调用方已 malloc，排除空指针后才进入地址错位假设。
- 当前方案路线档位：recommended_fastpath（正确性必需，无替代路线；"不使用 workspace"只是绕开而非解决）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1.6），CANN 9.2.0；头文件 `$ASCEND_HOME_PATH/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h`、`.../kernel_operator_swap_mem_intf.h`、`$ASCEND_HOME_PATH/include/acl/error_codes/rt_error_codes.h`。
- 相关文件：`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`（声明）、`op_kernel/arch35/bench_harness_scatter_kernel.h`（Init 取基址）、`tests/aclnn/repro_min.cpp`（8 元素最小复现）、`tests/aclnn/repro_min1k.cpp`（1000 元素复现）。
- 独立 repro 编译命令（直连 aclnn）：

```bash
cd operators/BenchHarnessScatter_package/tests/aclnn && source "$ASCEND_HOME_PATH/set_env.sh" && \
g++ -std=c++17 -O2 -fpermissive repro_min.cpp \
  -I"$ASCEND_HOME_PATH/x86_64-linux/include" -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/opp/vendors/<Vendor>/op_api/include" \
  -L"$ASCEND_HOME_PATH/lib64" -L"$ASCEND_HOME_PATH/opp/vendors/<Vendor>/op_api/lib" \
  -lascendcl -lnnopbase -lcust_opapi -o repro_min
```

## 预防

- 任何 kernel 要用用户 workspace 时，tiling 声明一律写 `RESERVED_WORKSPACE(按架构) + 用户字节数`，并在注释里写明架构常量来源；kernel 侧一律经 `GetUserWorkspace(workspace)` 取基址，不要直接解引用 `workspace` 参数。
- 遇到 507035 先做三件事：最小用例复现、查 `rt_error_codes.h`、读 `~/ascend/log/debug/plog/` 最新日志的 `error code =` 行——错误码 95（MTE DDR 越界）直接把范围缩到 MTE 访存越界。
- 在 kernel 里打印 64 位地址用高低 32 位分开打印，不要信 `%lx` 单次输出。

## 验证信号

以下信号来自该算子 fp32 workspace 路径落地会话：先在最小用例上稳定复现 507035 并经 plog 定位，再修正声明与取址后逐级放大验证。

- 修复前：8 元素用例（wsSize=32）与 1000 元素用例（wsSize=4000）均 `aclnnBenchHarnessScatter ret=0` 后 `aclrtSynchronizeStream` 返回 507035；对工作区写 1 个 float（4 字节）同样崩溃；不碰 workspace 的原生路径 sync ret=0（来源平台：950；证据等级：tool_observed）。
- 修复前：plog 报 `error code = 95, errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4`；kernel 内实测 workspace 指针 0x12000101b000，与 plog args[4]=0x12000001b000 相差 0x1000000=16MB（来源平台：950；证据等级：tool_observed）。
- 修复后：1000 元素 bf16 add 输出非零（out[0]=0x42C8、out[1]=0x3DCD 等）、无崩溃（来源平台：950；证据等级：tool_observed）。
- 修复后：全规模 48,720,456 元素用例 wsSize=211659040（=16MB 保留 + 48.7M×4 ≈ 201.9MB）、sync ret=0、精度 mism=0/48720456（来源平台：950；证据等级：tool_observed）。
- API 事实锚点：`rt_error_codes.h:126` 定义 507035=`ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`；`kernel_utils_constants.h` 注释 `workspace system reserve 16MB` 并按 `__NPU_ARCH__` 给出常量（arch3510=16MB、arch1001=2MB）；`GetUserWorkspace` 实现为 `GetSysWorkSpacePtr() + RESERVED_WORKSPACE`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `RESERVED_WORKSPACE` 是按 `__NPU_ARCH__` 分档的常量：本卡数值（16MB）仅对 Ascend950/arch3510 实测；其他架构须重新查 `kernel_utils_constants.h`（未验证，机制推断可迁移）。
- 本卡只覆盖"声明大小 + GetUserWorkspace"约定；若框架未来版本改为自动附加保留区，声明侧的 16MB 会造成多分配（无害但浪费），届时以框架文档为准（未验证）。
- 崩溃浮现在 SyncAll/PipeBarrier 而非写入指令处，是异步 MTE 异常的暴露方式；若 kernel 在写后没有任何屏障/同步就退出，可能表现为静默数据丢失而非 507035（未验证）。

**未决项**：

- 框架内部把 `workspace` 参数偏移 16MB 再传给 kernel 的确切传参路径（GE/runtime 哪一层做的偏移）未逐层核实；不影响结论（实测 + 修复生效），但跨版本升级 CANN 时建议重新验证该约定。
## 坏实践

- tiling 只声明用户字节数（`workspaces[0] = 用户大小`）就以为框架会补保留区——实测不会。
- kernel 直接把入口 `workspace` 参数当用户缓冲区基址做 MTE 写。
- 看到 507035 只当成"向量核异常"去查 UB 溢出/mask/对齐，而不读 plog 的错误码 95 明示的 MTE DDR 越界方向。

## 误诊

- 曾假设 wsSize=0 导致空指针工作区：实测 `GetWorkspaceSize` 返回的 wsSize 与调用方 malloc 均正常（32/4000/4000000 字节随用例正确），排除。
- 曾怀疑"SyncAll 单核会挂 / Cast 向量指令 UB 竞争"：把 SyncAll 换成 PipeBarrier<PIPE_ALL> 仍崩、把 Cast 换成逐元素转换仍崩——屏障与转换都不是因，越界的 MTE3 写才是；异步异常"在哪个屏障爆"与"哪条指令写错"是两回事。
- 曾按"NPU PRINTF %lx 输出不可信"得出地址打印错误的结论：实际数值正确，改用高低 32 位分开打印只是让证据可靠，结论（+16MB 偏移）未变。

## 后续增量（2026-09-17，并行策展批：GetLibApiWorkSpaceSize 可移植声明与 MIX→AIV 重写触发形态）

另一条 trace（GRU 算子 bench_no_harness_gru，Ascend950PR + CANN 9.2.0）复现同一机制，触发形态值得单独警惕：把 kernel 从 MIX 模式（Matmul+Vector）重写为**纯 AIV 模式**（coreType: VectorCore）时，顺手把 host tiling 里 GetWorkspaceSizes 的总量从"用户 workspace + 系统 workspace"改成**仅用户 workspace**（注释写"纯 AIV 模式不需要系统 workspace (Matmul 才需要)"）。随后所有用例一致 `[FAIL] 流同步失败: 507035`，plog `errcode:(95) The DDR address of the MTE instruction is out of range. subErrType: 0x4`，只有 AIVec 异常 + MTE 越界（cube/l1 error info 全 0），且 kernel 参数里 workspace 指针非空（plog `args(8) = 0x12000005b000`）。

消融定位（每轮 rebuild+install+run 单用例）：Process 置空 → 正常；只写普通输出 → 正常；写 workspace（连基地址无偏移的最小 `Duplicate`+`DataCopyPad`，仅 256 字节）→ 必崩；把该中间量移入 UB → 恢复正常。结论：workspace 指针本身不可写——与本卡主体机制一致（声明缺系统/保留区，偏移后的用户区指针落在分配范围外）。

可移植修复（before → after，verbatim）：

```cpp
    // before：总量只含用户 workspace（MIX→AIV 重写时错误精简）
    int64_t giAllSize = seqLen * batchPerCore * gateSize * TYPE_SIZE_FLOAT;
    int64_t perCoreWsSize = giAllSize;
    int64_t totalUserWs = usedCores * perCoreWsSize;
    int64_t totalWsSize = totalUserWs;
```

```cpp
    // after：总量加回系统 workspace（GetLibApiWorkSpaceSize 为平台 API，勿手写 16MB）
    int64_t giAllSize = seqLen * batchPerCore * gateSize * TYPE_SIZE_FLOAT;
    int64_t perCoreWsSize = giAllSize;
    int64_t totalUserWs = usedCores * perCoreWsSize;
    // 纯 AIV 也需要系统 workspace (ReduceSum 等内部使用)
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    int64_t totalWsSize = totalUserWs + static_cast<int64_t>(sysWorkspaceSize);
```

（`currentWorkspace[0] = static_cast<size_t>(totalWsSize);` 保持不变；kernel 侧取址方式不变：`wsBase = workspace + blockIdx_ * perCoreWsSize`，用户数据从 `wsBase` 开始。）

### 复现增量验证信号

来源说明：以下信号来自该 GRU 算子纯 AIV 重写会话的消融-修复-回归链路（上板实测）：

- 修复前：5 个用例全部 `[FAIL] 流同步失败: 507035`；plog `errcode:(95) The DDR address of the MTE instruction is out of range`，最小 256 字节写 workspace 基地址也复现（来源平台：950；证据等级：tool_observed）。
- 修复后（仅加系统 workspace，Process 仍为消融版）：同用例 507035 消失，仅剩输出全 0 的精度失败（无计算，符合预期）（来源平台：950；证据等级：tool_observed）。
- 恢复完整计算后回归：5/5 用例 PASS，y/hn MERE 均在 1e-06~1e-07 量级（阈值 5e-02）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践

- 本增量的 `GetLibApiWorkSpaceSize()` 是平台 API 取系统工作区大小，比手写 16MB 常量可移植；该 API 在本平台返回的具体字节数未单独打印（未验证是否恰为 16MB）。
- "ReduceSum 等库 API 内部使用系统 workspace"来自修复时的源码注释，属推断（未验证）。
- 系统/用户 workspace 的地址布局机制（用户区是否从 kernel 收到的指针基址开始）未做指令级验证——本例用户数据写在 `workspace + 0` 且精度通过，与本卡主体"kernel 收到的指针已偏移到用户区"的观测一致。
- 坏实践：重写 kernel（MIX→AIV、去 Matmul 化）时顺手"精简" tiling 的 workspace 声明，把系统 workspace 当成 Matmul 专属删掉——是否用 Matmul 与系统 workspace 声明无关；拿到 507035 + plog error 95 只看 host 返回码不翻 plog，长时间在 kernel 偏移算术里打转。
- 相关：同批多 trace 的"kernel 直绑 workspace 入参即崩"负结果卡（`kernel_workspace_param_direct_bind_507035`）与本卡机制同源，可交叉验证。

## 后续增量（2026-09-17，并行策展批：逐架构保留区常量表与 aivec95+段错误形态）

第三条 trace（BenchNoHarnessForeachNorm，Ascend950PR/arch35 + CANN 9.1.0，多核"各核算部分和 → 写 workspace 用户区 partialResults → core 0 合并"结构）以第三种形态复现同机制：tiling 侧**自定了 `RESERVED_WORKSPACE_SIZE = 2MB` 常量**并按 `2MB + CeilAlign(用户区, 256)` 声明——比"完全不声明保留区"更隐蔽。运行 ST 进程直接 `Segmentation fault (core dumped)`（退出码 139），plog 显示 kernel 已被 launch、全部参与核（core id 0/1/2/3/6/7/8…）同时报 `aivec error exception, error code = 95, pc current: 0x120041000f18, mte error: 0x80000000`。

逐架构常量表（CANN 9.1.0 `x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h`，verbatim；GetUserWorkspace 的返回值与 TilingFunc 自报常量完全无关）：

```cpp
// workspace system reserve 16MB
#if (__NPU_ARCH__ == 1001)
constexpr size_t RESERVED_WORKSPACE = 2 * 1024 * 1024;
#elif (__NPU_ARCH__ == 2002)
constexpr size_t RESERVED_WORKSPACE = 2 * 1024 * 1024;
#elif (__NPU_ARCH__ == 2201)
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#elif (__NPU_ARCH__ == 3002)
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#elif (__NPU_ARCH__ == 3102)
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
constexpr size_t RESERVED_WORKSPACE = 0;
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3113))
constexpr size_t RESERVED_WORKSPACE = 0;
#endif
```

修复：tiling 侧常量对齐框架值（arch35 上 2MB→16MB，workspace 总量 = 16MB + 用户区），与复现增量的 `GetLibApiWorkSpaceSize()` 平台 API 互为替代（后者免手工同步常量更可移植；前者须逐架构对齐 `kernel_utils_constants.h` 的条件值）；kernel 侧取用户区不变（`userWorkspace_ = GetUserWorkspace(workspace)`，即系统 workspace 基址 + 16MB）：

```cpp
// tiling 修复（before → after，verbatim）
constexpr size_t RESERVED_WORKSPACE_SIZE = 2 * 1024 * 1024;   // before：自定 2MB（照抄其他架构保留值）
constexpr size_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;  // after：对齐 arch35 框架常量
```

```cpp
// workspace 总量声明保持"保留区 + 用户区"结构
size_t partialResultsSize = static_cast<size_t>(tensorNum * usedCoreNum * sizeof(float));
partialResultsSize = CeilAlign(partialResultsSize, 256);
return RESERVED_WORKSPACE_SIZE + partialResultsSize;   // 16MB + 用户区
```

### 复现增量验证信号

来源说明：该算子迭代一 ST 联调会话的 error 95 复现 → 头文件溯源 RESERVED_WORKSPACE → 修 tiling 常量 → 重编译重装 → 重跑链路。

- 修复前：ST 段错误（退出码 139）；plog 全核 `aivec error exception, core id 0/1/2/…, error code = 95, pc current: 0x120041000f18, mte error: 0x80000000`（来源平台：950；证据等级：tool_observed）。
- 框架证据：`GetUserWorkspace` 返回 `GetSysWorkSpacePtr() + RESERVED_WORKSPACE`（dav_3510 `kernel_operator_common_impl.h`），返回值与 tiling 自报常量无关（来源平台：950；证据等级：tool_observed）。
- 修复后：error 95 与 `Task run failed` 消失，kernel 正常执行并进入结果比对阶段（后续数值失败为独立根因，另卡覆盖）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践

- 保留区数值逐架构变化：1001/2002=2MB，2201/3002/3102/3510/5102=16MB，3003/3113=0；跨架构移植必须重查常量表（未验证，推断可迁移）。
- 坏实践：TilingFunc 里自定义保留常量且不与目标架构框架值对齐（照抄其他算子/其他架构的 2MB）。
- 坏实践：用 kernel 二进制文件名里的哈希判断"是否重新编译过"——该哈希由算子名/配置派生、不随代码内容变化；核对重编须看时间戳或 `md5sum` 对比 build 产物与安装目录。
- 误诊记录：曾假设 `ReduceSum(dst, src, work, n)` 的 src/work 同缓冲是 error 95 根因，改独立 workBuf 重编后错误原样复现（同 PC、同错误码）——无害卫生改动但非根因。