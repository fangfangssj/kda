---
type: "Runbook"
title: "aclnn L2不传导tiling workspace致507035"
description: 手写/朴素 aclnn L2 GetWorkspaceSize 的返回值不含 TilingFunc 声明的 workspace（只回 uniqueExecutor 固定小值或 0），测试侧按返回值分配 → workspace 缺失/过小，kernel 写 workspace 即 507035（nullptr 或 errcode 264 越界）；判别=两侧 workspaceSize 对照 + DevMalloc 对账；规避=输出别名/禁用大 workspace；手动调 NnopbaseRunForWorkspace 补救 segfault banned。
tags: ["task.debugging", "topic.memory", "topic.implementation.api_layer.c_api", "topic.data_movement"]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# aclnn L2不传导tiling workspace致507035

## 现象

两条独立 trace、两类形态，同一传导缺口：

**形态 A（返回 0 → nullptr）**：Ascend950PR + CANN 9.2.0，softmax 类算子的 Group 分核分支（tilingKey=3 GroupTailR）。Phase 1 由 R 核做部分双归约，把 `m_part`/`s_part` 两个 fp32 数组用 `DataCopyPad` 写进 aclnn 框架传入的 workspace，`SyncAll` 后 Phase 2 读回合并。tiling 侧按 `workspaceSize = rGroupCnt × aTotal × sizeof(float) × 2` 计算用户 workspace（示例 shape [1,131072]：rGroupCnt=110、aTotal=1 → 880 B），但 `aclnnXxxGetWorkspaceSize` 返回给测试的 `workspaceSize=0` → 测试脚本 `if (workspaceSize > 0) aclrtMalloc(...)` 不执行 → `workspace=nullptr` 传入 → kernel `Init` 里 `gmWs_.SetGlobalBuffer((__gm__ float*)workspace)` 绑定空指针 → 任何对该 GM 的读或写（包括 32B 对齐、offset 0、仅 32B 长度）都触发向量核异常。plog `PreCheckTaskErr retCode=0x31 [vector core exception]`、`rtStreamSynchronize:ErrCode=507035`，DevMalloc 对账只有 `size=64`（tiling data），无 880B 量级分配。

**形态 B（返回固定小值 → 越界）**：Ascend 950PR + CANN 9.1.0，TopK 类算子（nohar 自定义算子包，手写 op_api/aclnn L2 API）。为绕开非末维 UB 同步问题实现"GM workspace transpose"：TilingFunc 在 dim!=last 时计算 workspaceSize（输入区 + 输出区 + 索引区，32 字节对齐，示例用例 626688 B）并通过 `context->GetWorkspaceSizes(1)` 写 `currentWorkspace[0]`。但手写 L2 API 的 GetWorkspaceSize 阶段只做 `*workspaceSize = uniqueExecutor->GetWorkspaceSize();`——该值是 executor 固定的 512 字节，不含 TilingFunc 写入的 `currentWorkspace[0]`。测试按 512 字节分配，kernel 按 626688 字节布局写 GM workspace → 越界写 → plog 所有参与核 `errcode:(264) errorStr: The address for scalar to access GM is invalid` → Host `rtStreamSynchronize:ErrCode=507035`。注意：引入 workspace 方案**之前**这些用例是"静默 mismatch"，引入之后变成硬性 507035——形态升级本身即是 workspace 越界的信号。

## 适用场景

- 工程形态：自定义算子 aclnn L2 API（手写 `aclnnXxxGetWorkspaceSize` + `aclnnXxx`，或对生成层取值路径未核实的 L2），kernel 需要大 GM workspace（transpose/中转/多阶段数据/跨核部分和）。
- API 路径：`aclnnXxxGetWorkspaceSize` 中 `uniqueExecutor->GetWorkspaceSize()`；TilingFunc 中 `context->GetWorkspaceSizes(WORKSPACE_NUM)` 与 `currentWorkspace[0]`；autogen 内部函数 `NnopbaseRunForWorkspace`；kernel `Init(GM_ADDR ..., GM_ADDR workspace, ...)`；测试侧 `if (workspaceSize > 0) aclrtMalloc(...)` 惯用写法。
- 验证过的平台与 CANN 版本：Ascend 950PR（arch35/DAV_3510）+ CANN 9.1.0 与 9.2.0（与 frontmatter 一致）。
- 适用粒度：operator_family——手写 L2 API 且 TilingFunc 声明了非零 workspace 的自定义算子都会踩到该传导缺口。
- 路线优先级：safety_fallback。**fallback 声明**：本卡的正解（让 L2 正确传导 tiling workspace）在手写 L2 下未闭合——`NnopbaseRunForWorkspace` 手动调用 segfault、手动复算公式有双份维护风险；两条已验证规避（输出别名、禁用大 workspace）都是绕开独立 workspace 的次优路线。
- 明确不适用 / 未验证场景：autogen 生成的 L2 API（其内部已正确调用 `NnopbaseRunForWorkspace`，本卡问题是手写 L2 没有等价调用）；A2/A3 等其他平台的 executor workspace 行为未验证。

## 根因/判别方法

根因链（形态 B 已定位到实现层；形态 A 的取值断点未追明但现象同构）：

1. TilingFunc 侧正确计算并设置 workspace（`currentWorkspace[0] = 626688`，TILING_DEBUG 打印可证）。
2. 手写 L2 API 的 GetWorkspaceSize 阶段只回 `uniqueExecutor->GetWorkspaceSize()`——该值是 executor 固定的 512 字节，**不包含** TilingFunc 写入的 `currentWorkspace[0]`。autogen 生成的 L2 API 用内部函数 `NnopbaseRunForWorkspace(*executor, &wsLen)` 执行 TilingFunc 并把 workspace 大小写回 executor，手写 L2 缺这一步。
3. 测试按 L2 返回值分配（0 → 不分配；512 → 过小），kernel 按 tiling 布局写 GM workspace → nullptr 解引用或越界写 → 507035（形态 A 无 plog 错误标记、DevMalloc 无对应分配；形态 B errcode 264）。

判别探针（可复用）：

- **两侧 workspaceSize 对照**：TilingFunc 里 `fprintf`/日志打印 `currentWorkspace[0]`；L2 API GetWorkspaceSize 阶段打印 `uniqueExecutor->GetWorkspaceSize()` 与最终 `*workspaceSize`。两者不一致（626688 vs 512；或 tiling 非零 vs L2 返回 0）即命中本缺口。
- **DevMalloc 对账**：开 plog 重跑后 `grep -i DevMalloc`，核对是否存在期望尺寸的设备分配；只有 `size=64`（tiling data）⇒ workspace 从未分配（L2 返回 0）。
- **形态反转信号**：同一用例从"静默 mismatch"变成"硬性 507035"，通常意味着新引入的写入路径（workspace）越界，而不是原精度问题恶化。
- **plog errcode 确认**：`errcode:(264) The address for scalar to access GM is invalid` + `rtStreamSynchronize:ErrCode=507035` 即 kernel 侧标量 GM 访问越界（本例为 workspace 越界写）。

## 解决配方

本缺口在手写 L2 下没有已验证的正向修复；可用规避与失败尝试如下。

**规避 1（输出别名，已验证）**：Group 分核 kernel 把中间 m/s 部分和缓冲改绑**输出 GM**（y）——Phase 1 写到输出区开头，Phase 2 在 CopyOut 覆盖输出之前读回，`SyncAll` 保证跨核写读顺序。before/after（kernel `Init`，verbatim）：

```cpp
        // before：绑定入参 workspace（GetWorkspaceSize 返回 0 时为 nullptr）
        if (td_->rGroupCnt > 0) {
            int64_t wsElems = td_->rGroupCnt * td_->axisShape[0] * 2;
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace,
                                  static_cast<uint64_t>(wsElems));
        } else {
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
        }

        // after：Group 模式改绑输出 y（框架可能不分配独立 workspace buffer）
        if (td_->rGroupCnt > 0) {
            gmWs_.SetGlobalBuffer((__gm__ float*)y);
        } else {
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
        }
```

**规避 2（禁用大 workspace，已验证）**：TilingFunc `needTranspose=false`（`currentWorkspace[0]=0`），kernel 保留 transpose 代码但不再触发，非末维用例标记跳过划入后续迭代；回归后交付。

**失败尝试 1（banned）：dlsym / 直接声明调用 `NnopbaseRunForWorkspace`**——模仿 autogen 在手写 L2 里补传导：

```cpp
    static auto nnopbaseRunForWorkspaceFunc = reinterpret_cast<aclnnStatus (*)(void*, uint64_t*)>(
        dlsym(RTLD_DEFAULT, "NnopbaseRunForWorkspace"));
    if (nnopbaseRunForWorkspaceFunc != nullptr) {
        uint64_t tilingWorkspaceSize = 0;
        auto wsRet = nnopbaseRunForWorkspaceFunc(uniqueExecutor.get(), &tilingWorkspaceSize);
        if (wsRet != ACLNN_SUCCESS) {
            *workspaceSize = 0;
            return wsRet;
        }
        *workspaceSize = tilingWorkspaceSize;
    } else {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    }
```

结果：测试程序在第一个用例即 `Segmentation fault (core dumped)`（dlsym 版、extern "C" 文件顶声明版、直接调用版三种写法均 segfault），segfault 根因未查明（未验证是否 TilingFunc 重复执行）。

**失败尝试 2（已回退）：L2 API 手动复算 workspace 大小**——按与 TilingFunc 相同的公式在 L2 里手算 dim!=last 的 workspace 再 `*workspaceSize = uniqueExecutor->GetWorkspaceSize() + calcWorkspaceSize;`。能让 ST 分配到足量 workspace，但 tiling 与 L2 各维护一份相同布局公式，shape/dtype 规则一变即失同步，且与并发修复线冲突，收尾时被回退为 `*workspaceSize = uniqueExecutor->GetWorkspaceSize();` 原状。

## 设计决策

- 选规避而不是修传导链：会话目标是恢复绿基线；`NnopbaseRunForWorkspace` 是公开头文件之外的 ACLNN 内部符号（无签名与 ABI 保证），三种写法实测 segfault；手动复算有双份维护风险——即使未来重新启用也应优先找框架侧传导机制，而不是复制公式。
- 输出别名的前提：输出元素数 ≥ `rGroupCnt × aTotal × 2` 个 fp32（本例 [1,131072] 输出远大于 110×1×2）；"Phase 2 读回先于 CopyOut 覆盖"由代码顺序 + `SyncAll` 保证；fp16/bf16 输出时 m/s 以 fp32 位模式写进输出区，读回方必须按 `float` 解释。
- 需要大 workspace 的自定义算子，优先评估 autogen 生成 L2 API（内部已处理传导），手写 L2 要把 workspace 传导当作独立验证项。

## 预防

- 手写 L2 API 且 TilingFunc 声明非零 workspace 时，**先做两侧 workspaceSize 对照探针**再写 kernel 侧布局代码；两者不一致时 kernel 一旦按 tiling 侧大小使用即越界。
- 测试侧 `if (workspaceSize > 0)` 才分配的惯用写法会把 L2 返回 0 静默化为 nullptr 传入——两端都没有把"workspace 应该非 0"显式断言出来。
- 看到 `errcode 264 "The address for scalar to access GM is invalid"` + `rtStreamSynchronize 507035`，优先核对所有 GM_ADDR 参数的实际分配大小（workspace 是最常见缺口），而不是先怀疑 kernel 指针算术。
- 调试期在 TilingFunc/L2 API 加的 `fprintf`/`[DEBUG]`/`[TILING_DEBUG]` 打印，交付前统一移除。

## 系统信息

- 硬件/软件：Ascend 950PR（arch35/DAV_3510），CANN 9.1.0 与 9.2.0 两条 trace；`bash build.sh --soc=ascend950` + custom_opp 安装；plog 位于 `~/ascend/log/debug/plog/`（`/tmp` 拒写时用 `~/plog` 兜底）。
- 相关文件（形态 B）：`op_api/aclnn_bench_no_harness_top_k.cpp`（L2 GetWorkspaceSize 阶段）、`op_host/arch35/bench_no_harness_top_k_tiling.cpp`（TilingFunc workspace 计算）、`op_kernel/arch35/bench_no_harness_top_k.h`（Init workspace 重定向、TransposeInput/TransposeOutput）、`build/autogen/`（autogen L2 参照，含 `NnopbaseRunForWorkspace` 用法）。
- 相关文件（形态 A）：`op_kernel/arch35/bench_harness_softmax_kernel.h`（Init/ProcessGroup/WriteWorkspaceMs）、`op_host/bench_harness_softmax_tiling_host.cpp`（workspaceSize 公式）、`tests/aclnn/test_aclnn.cpp`（ST harness，含 `if (workspaceSize > 0)` 分配逻辑）。

## 验证信号

来源说明：以下信号来自两条独立 trace 的上板实测（TopK 手写 L2 联调会话与 softmax Group 分支定位会话）：

- 形态 A 修复前：group 分支 34 例全部 `[ACLNN_FAIL] ... ACL error 507035 ... aclrtSynchronizeStream(stream)`；运行时路由 group 的大 R 用例同报 507035（来源平台：950；证据等级：tool_observed）。
- 形态 A 指针证据：plog `PreCheckTaskErr retCode=0x31, [vector core exception]`、`rtStreamSynchronize:ErrCode=507035 ... InnerCode=0x715005e`、aivmap 多核非零；DevMalloc 仅 `size=64`，无 880B workspace 分配（来源平台：950；证据等级：tool_observed）。
- 形态 A 规避 1 后：group 分支（tilingKey=3）34/34 precision PASSED；其余分支 46/46、37/37、24/24 无回归；`run.sh` 退出码 0；独立验收方重跑 build/run 均退出 0（来源平台：950；证据等级：tool_observed）。
- 形态 B 引入 workspace transpose 后：5 个非末维用例全部 `[FAIL] 流同步失败: 507035`（此前 inner=16 用例是 PASS，形态从静默 mismatch 升级为硬失败）（来源平台：950；证据等级：tool_observed）。
- 形态 B plog：所有核 `errcode:(264) errorStr: The address for scalar to access GM is invalid subErrType: 0x4`，`rtStreamSynchronize:ErrCode=507035, desc=[vector core exception]`（来源平台：950；证据等级：tool_observed）。
- 形态 B 对照探针：TilingFunc `[TILING_DEBUG] needTranspose=1, ... workspaceSize=626688` vs L2 API `[DEBUG] workspaceSize=512`——传导缺口实锤（来源平台：950；证据等级：tool_observed）。
- `NnopbaseRunForWorkspace` 三种写法（dlsym / 函数内 extern "C" / 文件顶 extern "C" 直接调用）均使测试程序第一个用例 `Segmentation fault (core dumped)`（来源平台：950；证据等级：tool_observed）。
- 形态 B 规避 2 后：ST 50/50 通过（跳过后续迭代范围用例）、退出码 0；UT 53/53 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `GetWorkspaceSize()` 返回 512 是 CANN 9.1.0 该环境实测值（推测为 executor 固定最小 workspace），其他 CANN 版本数值可能不同，但"手写 L2 API 不自动传导 currentWorkspace[0]"这一缺口形态值得在任何版本上先探针再信任。
- 形态 A 中 tiling 侧 `ws[0]` 为何没有传播到 L2 返回值的断点未追明（TilingFunc 是否实际写 `GetWorkspaceSizes(1)`、op_api 生成层如何取 workspace size）。
- 输出别名要求输出元素数 ≥ 中间结果元素数；输出紧张时本方案越界写坏输出，必须回退到真 workspace。
- `NnopbaseRunForWorkspace` segfault 根因未查明，不排除特定调用顺序下可用——但本开发会话 三种写法均失败，按 banned 处理。
- 手动复算公式与 TilingFunc 的等价性只在 dim!=last + 5 种 dtype 组合上对齐过，泛化到其他布局未验证。

**未决项**：

- **优先项：查明手写 L2 API 中 workspace 的正确传导机制**——读透 autogen `NnopbaseRunForWorkspace`/`NnopbaseRunWithWorkspace` 的调用契约（执行时机、executor 状态），给出可复用的手写 L2 API 大 workspace 用法或明确"不支持"结论。
- 形态 A 的取值断点（tiling `ws[0]` 为何未进 L2 返回值）未追明，确认后在框架/tiling 层根治并撤掉输出 GM 复用。
- `NnopbaseRunForWorkspace` segfault 根因（是否 TilingFunc 重复执行）未查明。
- `GetWorkspaceSize()` 固定返回 512 的构成（是否含保留区、与平台保留区的关系）未确认。
## 坏实践

- kernel 按 TilingFunc 声明的大小使用 workspace，而 L2 GetWorkspaceSize 阶段只回 `uniqueExecutor->GetWorkspaceSize()`——分配与使用脱节，必越界。
- 在公开头文件之外的 ACLNN 内部符号（`NnopbaseRunForWorkspace`）上做 dlsym/extern 声明直接调用——无签名与 ABI 保证，本开发会话 实测 segfault。
- 测试侧 `if (workspaceSize > 0)` 才分配，L2 返回 0 时静默把 nullptr 传给 kernel，kernel 侧无任何防护即写。
- plog 里看到 264/507035 就重装驱动/换板子——本例是纯 workspace 分配缺口，环境无故障。

## 误诊

- 形态 B 曾被误诊为"507035 是 GM SetValue/GetValue（S pipe）访问 workspace 本身有问题"——两侧 workspaceSize 对照（626688 vs 512）证明是分配大小缺口，不是访问语义问题。
- 形态 A 曾依次假设：Phase 2 merge 读回越界 → 两次 DataCopyPad 流水线 hazard → Scalar→MTE3 不同步 → GM 地址非 32B 对齐 → `SetGlobalBuffer` 缺 bufferSize——全部被对照实验否定（跳层二分 + 读试验 + DevMalloc 对账；跳层二分定位"仅写 workspace 即崩"，读 32B 试验证明指针无效）。
- "segfault 是 `extern "C"` 声明位置不对"——把声明移到文件顶部后依旧 segfault，声明位置不是根因。
