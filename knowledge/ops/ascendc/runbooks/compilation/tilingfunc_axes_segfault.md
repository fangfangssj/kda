---
type: "Runbook"
title: "TilingFunc 未声明依赖读 axes 致 segfault"
description: "vendor TilingFunc 内对 axes 输入(index 1) 调 GetInputTensor(1)->GetData<int32_t>() 而 未声明 TilingInputsDataDependency({1}) 时，C++ ST aclnn segfault；回退该读取即恢复。该读法在未声明依赖时禁用。 R2（2026-09，CANN 9.1.0）：崩因定位到张量 placement——aclnn 路径 axes 为 kOnDeviceHbm，GetData 返回设备地址，host 解引用即 139，UT 用 kOnHost 构造故全绿；IsOnHost 守卫可止崩但退空 axes 会静默改语义（错 key→桩 kernel→全 0 输出），已验证正解有二：placement-agnostic 读取（device 分支 dlsym 解析 aclrtMemcpy 拷到 host）与组合注册（TilingInputsDataDependency+HostInputs 并存触发框架 D2H，单声明均不物化）。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
aliases: ["EuclideanNorm"]
---

# TilingFunc 未声明依赖读 axes 致 segfault

## 现象

在 Ascend950PR（soc=ascend950，CANN 9.0.0-beta.2）调试 `euclidean_norm` 归约算子时，为了让 vendor `TilingFunc` 能按 axes 的**实际值**（而非 axes 的 COUNT）划分 reduce 轴，agent 在 `op_host/arch35/euclidean_norm_tiling_arch35.cpp` 的 `LoadAxesValues` 路径里对 axes 输入（`GetInputTensor(1)`）取 `GetData<int32_t>()` 读取轴值，且**未**在 `IMPL_OP_OPTILING(...).Tiling(TilingFunc)` 注册链上声明 `.TilingInputsDataDependency({1})`（因同名 built-in `InputsDependency` 冲突，声明它会在 TTK 重现 attr 越界，本开发会话 另行记录该 built-in 冲突）。

改完跑 C++ ST（`tests/aclnn/run.sh`）：进程 **segfault**（段错误）。把 `InferShape` 改回安全版本后仍 segfault——定位到 segfault 源在 `TilingFunc` 的 `LoadAxesValues`：未声明 data dependency 时 `GetInputTensor(1)` 返回的 Tensor 不持有 host 端可读内存，`GetData<int32_t>()` 解引用即崩。回退 `op_host/arch35/euclidean_norm_tiling_arch35.cpp`（`git checkout`）后 C++ ST 恢复至 baseline 通过率（82 PASS / 15 FAIL，与改动前一致）。

## 适用场景

- **工程形态**：ops-registry-invoke vendor 算子仓，Host Tiling 阶段需读 axes 等数据相关输入的值。
- **API 路径**：`TilingContext::GetInputTensor(index)` → `Tensor::GetData<T>()`；`.TilingInputsDataDependency({idx,...})` 注册。
- **验证平台与版本**：Ascend950PR（soc=ascend950），CANN 9.0.0-beta.2。仅 950 实测。
- **适用粒度**：`operator_family`（任何 vendor TilingFunc 试图在 tiling 期读 axes/动态输入数据的归约/动态 shape 算子）。
- **路线优先级**：`banned`——在未声明 `TilingInputsDataDependency` 的情况下直接 `GetInputTensor(i)->GetData<T>()` 是禁用路线；要么先声明依赖、要么不读数据改走 shape/count 路径。
- **不适用/未验证场景**：a2/a3 未验证。本卡只覆盖 "未声明依赖 → 解引用即崩" 这一负路径；声明依赖后能否正常取数据在本开发会话 因 built-in 冲突未能验证。

## 失败路线

**尝试**：在 vendor `TilingFunc` 的 `LoadAxesValues` 路径里对 axes 输入（index 1）调用 `GetInputTensor(1)->GetData<int32_t>()` 读取轴值，期望在 tiling 期按 axes 实际值划分 reduce 轴，而**不**在算子 tiling 注册上声明 `.TilingInputsDataDependency({1})`。

**失败 / 回退**：

1. 改完跑 C++ ST（`tests/aclnn/run.sh`）→ 进程 segfault（段错误，无输出）。
2. 先把 `InferShape` 改回安全版本隔离变量 → 仍 segfault，排除 `InferShape` 因素，segfault 源在 `TilingFunc` 的 `LoadAxesValues`。
3. `git checkout -- op_host/arch35/euclidean_norm_tiling_arch35.cpp` 回退该 TilingFunc 改动 → C++ ST 恢复至 baseline 通过率（82 PASS / 15 FAIL，与改动前一致），构成"该读取导致崩溃、移除即恢复"的闭环。

**边界（banned）**：在未声明 `TilingInputsDataDependency` 的情况下直接 `GetInputTensor(i)->GetData<T>()` 解引用数据是禁用路线。本开发会话 的 950 / CANN 9.0.0-beta.2 环境下稳定 segfault。替代正路径有二：先声明 `.TilingInputsDataDependency({i})` 再读（本开发会话 因 built-in 同名冲突未能验证），或不读数据改走 shape/count 路径。

## 根因

`TilingContext::GetInputTensor(index)` 的契约：仅当算子通过 `TilingInputsDataDependency({...})` 声明了对应输入为 tiling 数据依赖时，返回的 Tensor 才持有 host 端可读内存；否则该内存不可解引用。本开发会话 实测在未声明依赖时 `GetInputTensor(1)` 的返回不持有可读内存，`GetData<int32_t>()` 解引用即 segfault。

agent 据 segfault 行为推断 `GetInputTensor(1)` 返回"非空但无效"指针（与 `tiling_context.h` 文档"反之返回 nullptr"的描述不一致），但该推断仅来自崩溃行为反推，未直接观测到指针值，属解释而非实测。

为何本算子不能声明依赖：vendor `euclidean_norm` 与 built-in `libophost_legacy.so` 同名，built-in 已注册 `InputsDependency`，vendor 的 `.TilingInputsDataDependency({1})` 被 "has been registered" 拒绝、且会触发 built-in data-dependency 路径访问 attr[1] 越界（本开发会话 另行记录该 built-in 冲突）。于是在本环境下"声明依赖"与"不声明直接读"两条路都被堵死——前者触发框架越界，后者 segfault。

## 验证信号

本开发会话 实测信号（均在 950 平台、CANN 9.0.0-beta.2）：

- C++ ST 在 `TilingFunc` 的 `LoadAxesValues` 路径（`GetInputTensor(1)->GetData<int32_t>()`、未声明 `TilingInputsDataDependency`）下 segfault；先把 `InferShape` 改安全后仍 segfault，定位 segfault 源在 `TilingFunc` 而非 `InferShape`。（来源平台：950；证据等级：tool_observed）
- 回退 `op_host/arch35/euclidean_norm_tiling_arch35.cpp`（`git checkout`）后，C++ ST 恢复至 baseline 通过率（82 PASS / 15 FAIL，与改动前一致），构成"该读取导致崩溃、移除即恢复"的闭环。（来源平台：950；证据等级：tool_observed）

- （R2 增补，Ascend950PR（arch35）+ CANN 9.1.0，EuclideanNorm aclnn 冒烟/ST 两轮修复）直读崩溃：aclnn 冒烟 8/8 全部 Segmentation fault 退出码 139、stdout 空；同机 add_custom 对照 28/28 PASSED；git stash 回退后恢复通过（来源平台：950；证据等级：tool_observed）。
- （R2 增补）最小化二分：纯桩（SetBlockDim(1)+ws=0）→ 8/8 PASSED；桩+写 TilingData → 8/8 PASSED；完整实现（含 axes 读值）→ 139；强制 tilingKey=0 仍崩、usedCoreNum=1 单核仍崩（来源平台：950；证据等级：tool_observed）。
- （R2 增补）IsOnHost 守卫落地：冒烟 8/8 PASSED、公共 TilingUT 19/19 PASSED、冒烟与 UT 串行互跑均退出 0（来源平台：950；证据等级：tool_observed）。
- （R2 增补）守卫陷阱暴露：守卫期 aclnn ST key=0 用例输出全 0（elem0 dev=0.000000 golden=25.495098/18932.458984 等），key=1 桩用例同签名全 0（来源平台：950；证据等级：tool_observed）。
- （R2 增补）placement-agnostic 终态：修复后 T1 maxRel=9.929e-08、T3 maxRel=0.000e+00 PASSED；Branch-0 (key=0) 精度 5/5 PASSED、run.sh 退出 0、TilingUT 19/19+9/9 无回归；aic-ascend950-ops-info.ini 无 valueDepend（TilingInputsDataDependency 已回退）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 平台：仅 950 实测。a2/a3 未验证、未定位——不得据 950 结果推断其他平台 `GetInputTensor` 的返回契约。
- CANN 版本：9.0.0-beta.2。其他版本 `GetInputTensor` 在未声明依赖时的返回值（nullptr vs 非空无效）可能不同，需重测。
- 触发条件：仅当 vendor 算子名与 built-in 重名、无法声明 `TilingInputsDataDependency` 时，本"未声明依赖直接读"的负路径才会被尝试。非同名算子应走"声明依赖后读"的正路径（本开发会话 未验证该正路径）。

- （R2 增补）修复只覆盖"host 可安全读"与"device HBM"两种 placement；其他 placement 走 GRAPH_FAILED 拒启，不猜测。GE 图执行路径、`kFollowing`、`kOnDeviceP2p` 等 placement 组合未验证。
- （R2 增补）"退空 axes 回退全规约"只对语义等价的用例无害；调用方 device 侧 axes 不等价于全规约时结果会错——退空兜底是本卡证伪的路线。
- （R2 增补）`aclrtMemcpy` 以 int32_t 返回值 + 硬编码 kAclMemcpyDeviceToHost=2 复刻头文件枚举，枚举值变化需同步；axes 元素数极大时整缓冲拷贝开销未评估（本场景 ≤ pattern 秩上界，几十字节）。

## 增补：placement 机制、守卫陷阱与 placement-agnostic 正解（R2，2026-09，CANN 9.1.0）

EuclideanNorm 同一问题域的两步演进（同 trace 先后两轮修复），补齐机制层与正解：

**机制**：aclnn 路径的 axes 张量由 `aclrtMalloc + aclCreateTensor` 创建，`TensorPlacement = kOnDeviceHbm`，`GetData<void>()` 返回**设备地址**；TilingFunc 在 host 进程运行，直接解引用即段错误（退出码 139）。host 侧 TilingUT 构造 axes 用 host 缓冲（`kOnHost`）→ 同一份 ParseInputs 代码读值正常、UT 全绿——**同一 TilingFunc 两条执行路径的 placement 不同**，这是"UT 绿、上板崩"的形态根源。

**第一步修复（IsOnHost 守卫）**：读值前 `gert::TensorPlacementUtils::IsOnHost(tensor->GetPlacement())` 守卫，非 host 不读——止崩；但当时把非 host 一律退空 axes（按 full-reduce 语义兜底），埋下第二步问题。

**守卫陷阱（退空改变语义）**：aclnn 路径下守卫恰好跳过 device 侧 axes → 空按 full-reduce 合轴 → pattern 变成 AR（axisNum=2）→ tilingKey 误判（key=0 用例被路由到未实现的 key=1 桩 kernel）→ 桩不计算 → **输出全 0**（ST FAILED: elem0 dev=0.000000 golden=25.49…）。"UT 全绿 + ST 全 0"时先怀疑 tiling 拿到的输入与 UT 不是同一 placement，而不是先怀疑 kernel。

**最终正解（placement-agnostic 读取）**：host 张量直接解引用；device 张量用运行时 `dlsym(RTLD_DEFAULT, "aclrtMemcpy")` 把小缓冲（元素数 ≤ pattern 秩上界）D2H 拷到 host 再读——dlsym 惰性解析保证 tiling .so 在不链接 libascendcl 的上下文（如 TilingUT）也能干净加载，解析失败返回 GRAPH_FAILED 干净拒启。verbatim（取自终版 `ParseInputs` 的 axes 读取块）：

```cpp
        const bool axesOnHost = gert::TensorPlacementUtils::IsOnHost(axesTensor->GetPlacement());

        // Host-side staging buffer used only for the device-resident case.
        std::vector<uint8_t> axesHostBuf;
        const void* axesReadPtr = axesData;   // default: dereference in place (host)
        if (!axesOnHost) {
            // Device tensor: copy bytes device→host via aclrtMemcpy (lazy dlsym).
            using AclrtMemcpyFn = int32_t (*)(void*, size_t, const void*, size_t, int32_t);
            AclrtMemcpyFn pAclrtMemcpy = reinterpret_cast<AclrtMemcpyFn>(dlsym(RTLD_DEFAULT, "aclrtMemcpy"));
            if (pAclrtMemcpy == nullptr) {
                // No ACL runtime available (should not happen in the aclnn path);
                // bail out so the framework rejects the launch cleanly.
                return ge::GRAPH_FAILED;
            }
            const size_t axesBytes = static_cast<size_t>(axesCount) *
                static_cast<size_t>(ge::GetSizeByDataType(axesDtype));
            axesHostBuf.resize(axesBytes);
            constexpr int32_t kAclMemcpyDeviceToHost = 2;   // ACL_MEMCPY_DEVICE_TO_HOST
            const int32_t cpRet = pAclrtMemcpy(axesHostBuf.data(), axesBytes, axesData, axesBytes,
                                               kAclMemcpyDeviceToHost);
            if (cpRet != 0) {
                return ge::GRAPH_FAILED;
            }
            axesReadPtr = axesHostBuf.data();
        }
```

配套头文件：`_GNU_SOURCE` 前置（glibc 下 `RTLD_DEFAULT`）+ `#include <dlfcn.h>`。

**TilingInputsDataDependency 单声明在本路径不生效**：曾在注册链声明 `.TilingInputsDataDependency({1})` 期望框架把 device axes 拷到 host（依 GetInputTensor 文档"若算子被配置为 tiling 数据依赖则返回的 Tensor 对象中保存了 Host 内存地址"）——CANN 9.1.0 aclnn 自定义算子路径实测 TilingFunc 仍拿到 device 指针，已整体回退，终态构建产物 ini 无 valueDepend 残留。单声明不物化已在三条独立 trace 一致复现（本开发会话与另一独立 trace）；`TilingInputsDataDependency({1})` 与 `HostInputs({1})` **并存注册**时框架完成 D2H（60/60 通过）——框架注册路径正解见下节。与库内《ValueDepend(REQUIRED) 让 axes 走 host 端》卡的 OpDef ValueDepend 机制是不同注册点，后者未在本路径验证。

**隔离定位方法（可复用链路）**：同机已知正常算子 aclnn 对照（排除设备/环境）→ `git stash` 回退可疑变更复跑（锁定回归源）→ TilingFunc 最小化二分（纯桩 / 桩+写 TilingData / 桩+SetTilingKey / 完整实现四版本，把崩溃收敛到"读张量值"）→ grep TilingFunc 的每个 `GetData<` 解引用点核对 placement 守卫。附加排除项：强制 tilingKey=0 仍崩（排除 key）、usedCoreNum=1 单核仍崩（排除多核 blockDim）。

**增补坏实践**：

- 在 TilingFunc（host 回调）里直接 `tensor->GetData<T>()` 并解引用，不做 placement 检查（即本卡 banned 路线的机制层表述）。
- 只跑 host 侧 UT 就认为 TilingFunc 安全——UT 的张量都在 host，掩盖 placement 缺陷；至少配一个 device 张量的 aclnn 冒烟用例做双路径验证。
- 写"非 host 就退空"的兜底——退空会静默改变语义（错 key → 桩 kernel → 全 0 输出），是本卡第二步证伪的路线。

## 增补：框架注册路径——单声明不物化，组合注册生效（R2·另一独立 trace 增补）

并行批次两条独立 trace 补齐"框架级搬运"路径的边界（与上节 dlsym 手工 D2H 互补）：

**单声明不物化（负结果，另一独立 trace）**：`IMPL_OP_OPTILING(...).HostInputs({1})` 或 `.TilingInputsDataDependency({1}, {gert::TilingPlacement::TILING_ON_HOST})` 单独注册时，plog 可见 vendor 注册行（`[Tiling][HostInputs]` / `[Tiling][TilingDependency][TilingDependencyPlacement]`）——注册成功，但 aclnn 单算子执行器不消费该声明，`GetInputTensor(1)` 仍返回 `kOnDeviceHbm` 的 device 地址（placement 探针：`axesOnHost=0 placement=0 dataPtr=0x120000018000`），ST 仍全 0。`GetInputTensor` 头文件文档"未配置数据依赖时地址为 nullptr"与实测不符（未物化时返回真实 device 地址而非 nullptr）——判别以 placement 探针为准：

```cpp
    const auto* axesTensor = context->GetInputTensor(1);
    fprintf(stderr, "[TILING-DIAG] placement=%d isOnHost=%d dataPtr=%p\n",
            axesTensor ? (int)axesTensor->GetPlacement() : -1,
            (int)(axesTensor && gert::TensorPlacementUtils::IsOnHost(axesTensor->GetPlacement())),
            axesTensor ? axesTensor->GetData<void>() : nullptr);
```

**组合注册正解**：`TilingInputsDataDependency({1})` 表达"tiling 读该输入"的语义声明，`HostInputs({1})` 触发框架 tiling 前 D2H——两者**并存**注册后 placement 变为 host 可读，60/60 用例通过（foreach 类标量张量输入算子，CANN 9.1.0/Ascend950PR）：

```cpp
IMPL_OP_OPTILING(BenchNoHarnessForeachNorm)
    .Tiling(BenchNoHarnessForeachNormTilingFunc)
    .TilingParse<BenchNoHarnessForeachNormCompileInfo>(TilingParseForBenchNoHarnessForeachNorm)
    .TilingInputsDataDependency({1})  // scalar (IR index 1) 需要 tiling 阶段读取数据
    .HostInputs({1});  // scalar (IR index 1) 标记为 host 输入，框架在 tiling 前完成 D2H
```

**dlsym 兜底的分歧证据**：先前增量 会话中 dlsym `aclrtMemcpy`/`rtMemcpy` D2H 变体曾 60s 挂起（机制未定位，勿作主路径）；上节 EuclideanNorm trace 的 dlsym `aclrtMemcpy` 路径则实测通过——同符号不同会话结果分歧未定位，dlsym 方案仅作注册路径不可用时的 fallback 并保留 GRAPH_FAILED 拒启。

验证信号（R2·并行轨迹 增补）：

- （另一独立 trace，BenchNoHarnessForeachNorm，CANN 9.1.0）修复前 ST Real 60/60 `[CRASH] signal=11`，15 步探针定位崩溃在 GetWorkspaceSize 内；插桩观测 scalarTensor 非空、scalarAddr=0x120000018000（设备地址段），GetInputTensor(1) 同地址；注册 `.HostInputs({1})` 后探针全过、result=32 正确；全量回归 60/60 PASS、UT 24/24 PASS（来源平台：950；证据等级：tool_observed）。
- （独立 trace，EuclideanNorm，CANN 9.1.0）`HostInputs({1})` 单独注册后 plog 出现 vendor 注册行但 ST 仍 0/16、DoTiling 仍全 tilingKey:1；`TilingInputsDataDependency({1},{TILING_ON_HOST})` 单独注册同象；placement 探针 `axesOnHost=0 placement=0 dataPtr=0x120000018000`——声明注册成功 ≠ 执行期物化（来源平台：950；证据等级：tool_observed）。

失效边界（R2·并行轨迹 增补）：

- 只否定 CANN 9.1.0 的 aclnn 单算子路径；`TilingInputsDataDependency` 在 GE 图模式下是否兑现 host 物化未验证，不要外推到所有执行路径。
- "HostInputs 使 placement 变 host"的机制解释是会话内推断（探针通过 + 直接读取分支生效间接支持），修复后 placement 实际取值未直接观测。
- 组合注册正结果覆盖单输入 `{1}` 组合；多输入 host 化（HostInputs({0,1})）未验证。

## 增补：第三条路线——TilingData 标量化 + Device 侧读 GM 计算（并行策展批次增补）

独立复现 R3（BenchNoHarnessMoeReRouting，MoE token 重排算子，CANN 9.1.0 / Ascend950PR）：该算子的 `expert_token_num_per_rank` 输入原已声明 `ValueDepend(REQUIRED, DependScope::TILING)`，因 opbuild dtype 组合冲突被移除（见同批《ValueDepend 与 Follow 输出 dtype 组合冲突》卡）后，Host tiling 为预计算 block schedule 直接 `GetInputTensor(1)->GetData<int32_t>()` 读输入——首个 Real 用例进程段错误（`run.sh: line 210: ... Segmentation fault (core dumped)`，无任何框架报错，不走 `OP_CHECK_NULL_WITH_CONTEXT` 的优雅失败路径）。

与前两条路线（dlsym 手工 D2H、组合注册框架 D2H）互补的**第三条已验证路线：把数据依赖计算搬到 Device 侧**——Host tiling 只传标量，kernel 从 GM 读数据自行计算 schedule：

1. TilingData 从整段 schedule 数组（~131KB 上限）精简为 ~64B 标量（A/H/N/E/blockDim/tileTokens/numDtype/hAligned）——顺带消除"框架对 TilingData 大小限制"风险项；
2. Host tiling 只读 shape/dtype（`GetInputShape`/`GetInputDesc`），不读数据；多核切分与 tileTokens 按常规预算，且必须为 Device 侧 matrix(int64)/srcOffset(int32)/expertNum(int64) 三个 buffer 同步预留 UB 开销（N*E 上限守卫 `BENCH_MOE_DEVICE_MAX_NE` 一并入 tiling 检查）；
3. kernel 侧 `ReadMatrixAndComputeSchedule()`：`DataCopyPad` 把 expert_token_num_perRank 矩阵从 GM 搬入 UB（int32/int64 双 dtype 分支），再逐元素算行前缀和 srcOffset（全局绝对前缀和），Process 先算 schedule 再逐块 gather。

修复前（Host tiling 读输入数据，崩溃点，verbatim 节选）：

```cpp
    // 3. 读取 expert_token_num_per_rank 数据（ValueDepend 已被移除 → 段错误点）
    auto numTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, numTensor);

    std::vector<int64_t> matrix(N * E);
    if (numDtype == ge::DT_INT32) {
        const int32_t* data = numTensor->GetData<int32_t>();   // ← 无 host 数据，崩溃
        OP_CHECK_NULL_WITH_CONTEXT(context, data);
        for (uint32_t i = 0; i < N * E; i++) {
            matrix[i] = static_cast<int64_t>(data[i]);
        }
    } else if (numDtype == ge::DT_INT64) {
        ...
```

修复后（TilingData 标量化，verbatim 节选）：

```cpp
// op_kernel/arch35/<op>_tiling_data.h（TilingData 精简为 ~64B 标量）
struct BenchNoHarnessMoeReRoutingTilingData {
    // === 标量参数 ===
    uint64_t A = 0;             // token 总数
    uint32_t H = 0;             // token 长度 (hidden_dim)
    uint32_t N = 0;             // rank 数
    uint32_t E = 0;             // 专家数
    uint32_t blockDim = 0;      // 使用的核数
    uint32_t tileTokens = 0;    // 单次 UB 处理的 token 数
    uint32_t numDtype = 0;      // 0=int32, 1=int64
    uint32_t hAligned = 0;      // H*sizeof(T) 是否 32 字节对齐
    // block schedule 和 expert_token_num 在 Device 侧从 GM 读取自行计算
};
```

```cpp
// op_kernel/arch35/<op>.h（Device 侧计算 schedule，verbatim 节选）
template <typename T_TOK, bool HAS_SCALES>
__aicore__ inline void BenchNoHarnessMoeReRouting<T_TOK, HAS_SCALES>::ReadMatrixAndComputeSchedule()
{
    uint32_t ne = N_ * E_;

    // 1. Read matrix from GM to UB（int32/int64 输入均可正常搬运）
    DataCopyParams matCopyParams;
    matCopyParams.blockCount = 1;
    matCopyParams.srcStride = 0;
    matCopyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};

    if (numDtype_ == 0) {
        auto matLocal = matrixBuf.Get<int32_t>();
        matCopyParams.blockLen = static_cast<uint16_t>(ne * sizeof(int32_t));
        DataCopyPad(matLocal, expertTokenNumPerRankGmInt32, matCopyParams, padParams);
    } else {
        auto matLocal = matrixBuf.Get<int64_t>();
        matCopyParams.blockLen = static_cast<uint16_t>(ne * sizeof(int64_t));
        DataCopyPad(matLocal, expertTokenNumPerRankGmInt64, matCopyParams, padParams);
    }

    // 2. Compute srcOffset (row-major prefix sum) -> srcOffsetBuf (int32)
    //    srcOffset[i][j] = sum of matrix[i][0..j-1]
    auto srcOffsetLocal = srcOffsetBuf.Get<int32_t>();
    for (uint32_t i = 0; i < N_; i++) {
        int32_t rowPrefix = 0;
        for (uint32_t j = 0; j < E_; j++) {
            uint32_t idx = i * E_ + j;
            srcOffsetLocal.SetValue(idx, rowPrefix);
            rowPrefix += GetMatrixValue(idx);
        }
    }
}
```

适用边界：ValueDepend 因框架约束不可声明（本例 opbuild dtype 组合冲突）、输入数据量在 Device UB 预算内（N*E 上限守卫拒绝超限）；O(N×E) 标量整数加法的开销可忽略。设计文档已把「Host 仅传标量，Device 侧读 GM 自行计算」登记为备选方案。

坏实践（R3 增补）：

- 以为 `OP_CHECK_NULL_WITH_CONTEXT(data)` 能兜住 GetData 的失败——实际拿到的是无效指针/无效数据，直接段错误，没有优雅返回的机会。
- 移除 ValueDepend 后保留 Host 侧 schedule 计算「试试看」——首个 Real 用例必崩。

验证信号（R3 增补，BenchNoHarnessMoeReRouting 迭代一联调会话，Ascend950PR / CANN 9.1.0）：

- 修复前：`run.sh: line 210: 769173 Segmentation fault (core dumped)`，首个 Real 用例 tiling 阶段崩溃（来源平台：950；证据等级：tool_observed）。
- 修复后：同命令推进至首用例 kernel 执行与输出比对（CPU Golden 自测 ALL PASS、GetWorkspaceSize ret=0、aclnn ret=0），段错误消失（来源平台：950；证据等级：tool_observed）。
- 最终：全部独立缺陷修复后 ST 12/12、UT 30/30 通过（来源平台：950；证据等级：tool_observed）。

失效边界（R3 增补）：

- N×E 超过 device 侧 UB 预算上限时 tiling 显式 GRAPH_FAILED 拒绝——更大矩阵需回到 Host 计算（ValueDepend/组合注册路线）或分批搬运（未实现）。
- int64 输入时 matrixBuf 按 int64 展开，UB 消耗翻倍，N×E 上限相应减半（按公式推导，未单独实测）。
- 其他 CANN 版本的 ValueDepend/GetData 行为未验证（推断可迁移）。

## 增量（守卫受控配对复现与"静默构建失败 + 旧 .so 回滚假象"调试纪律）

同一 EuclideanNorm 问题域的又一条独立 trace（Ascend950，CANN 9.1.0）补两层证据：

**1）崩溃-恢复的受控配对（守卫存在性单变量）**：把宿主侧 TilingFunc `ParseInputs` 的 placement 守卫去掉（保留注册链声明）→ `build exit=0` → `run exit=139`（`Segmentation fault (core dumped)`、plog 里 DoTiling 都没写全、一条用例输出都没有——tiling 崩溃带崩整个 aclnn 执行器进程）；同改动第二次独立复现同象；`git checkout` 恢复带守卫版本 → `build exit=0` → `run exit=0`（基线 16 failed、无崩溃）。此前 placement 探针已证明该指针是 device HBM 地址（`placement=0 dataPtr=0x120000018000`）。守卫形态（终版稳定版，verbatim）：

```cpp
    std::vector<int32_t> axesRaw;
    const bool axesOnHost = (axesTensor != nullptr) &&
        gert::TensorPlacementUtils::IsOnHost(axesTensor->GetPlacement());
    if (axesCount > 0 && axesTensor != nullptr && axesOnHost) {
        const ge::DataType axesDtype = axesTensor->GetDataType();
        const void* axesData = axesTensor->GetData<void>();
        if (axesData == nullptr) {
            return ge::GRAPH_FAILED;
        }
        axesRaw.reserve(static_cast<size_t>(axesCount));
        if (axesDtype == ge::DT_INT32) {
            const auto* p = static_cast<const int32_t*>(axesData);
            for (int64_t i = 0; i < axesCount; ++i) { axesRaw.push_back(p[i]); }
        } else if (axesDtype == ge::DT_INT64) {
            const auto* p = static_cast<const int64_t*>(axesData);
            for (int64_t i = 0; i < axesCount; ++i) { axesRaw.push_back(static_cast<int32_t>(p[i])); }
        } else {
            return ge::GRAPH_FAILED;
        }
    }
```

**2）调试陷阱（本层新知识）**：第一次崩溃后把守卫加回去，回滚 edit 意外删掉 `if (...) {` 块的函数体、只留悬空开括号，产生 `error: a function-definition is not allowed here before '{' token`（×13）语法错误；而调试命令把构建输出重定向 `/dev/null` 且用 `;` 串联后续步骤，构建连续失败无人察觉——之后几次"回滚后仍 139"其实跑的一直是**旧的无守卫 .so**，一度误判"回滚无效、另有根因"。直到发现 `build exit=1`、定位语法错误、修复后 `build exit=0` 基线才恢复。这是"回滚未生效被当成回滚无效"的典型形态。

坏实践（独立 trace 增补）：

- 调试循环的构建命令把输出重定向 `/dev/null` 且用 `;` 串联后续步骤（`./build.sh ... > /dev/null 2>&1 && echo "built"; cd ... && ./test_aclnn`）——build 失败时后续照跑旧产物；调试期构建必须看 exit code（`echo "build exit=$?"`）或至少保留输出。
- "回滚后现象仍在"时先怀疑"另有根因"，而不是先核对回滚是否真正生效（build 失败？装的是旧 .so？产物时间戳？）。

验证信号（独立 trace 增补，EuclideanNorm aclnn ST 联调会话，Ascend950 / CANN 9.1.0）：

- 无守卫版本构建成功后运行 `./test_aclnn` → `Segmentation fault (core dumped)`、exit=139、无任何用例输出；同改动第二次独立复现 build exit=0 → run exit=139（来源平台：950；证据等级：tool_observed）。
- `git checkout` 恢复带守卫 tiling 源码 → build exit=0 → run exit=0（基线 16 failed、无崩溃）（来源平台：950；证据等级：tool_observed）。
- 语法错误期间构建失败实证：build exit=1 + `a function-definition is not allowed here before '{' token` ×13（来源平台：950；证据等级：tool_observed）。
- placement 探针佐证：`placement=0 dataPtr=0x120000018000`（kOnDeviceHbm device 地址）（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：

- 崩溃归因以"守卫在→不崩、守卫去→崩"的两次受控配对为证据，未采集 core dump 栈；若需栈级证据须开 core 并分析执行器进程。
- "静默构建失败 + 旧 .so"陷阱依赖 build.sh 安装到 OPP vendors 目录、运行期从该目录加载的链路；纯就地加载的工程形态不适用。
- host 侧解引用不崩的唯一安全前提是 placement 判定为 host（或框架物化）；`dataPtr != nullptr` 不是安全判据。
