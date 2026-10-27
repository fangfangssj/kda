---
type: "Runbook"
title: aclnn 手动 Run 须 AbandonCache 防堆损坏
description: 自定义 aclnn 绕过 CommonOpExecutorRun 直接 SetStream/Run 时，executor 仍留在执行缓存中；ST 全量（600 用例）连续创建/销毁下缓存残留累积，SetStream 写失效指针段错误（139）；执行前 AbandonCache(true) 禁用缓存后全量通过。
tags: [task.debugging, topic.memory, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-16T10:23:52Z'
platforms: ["950"]
---

# aclnn 手动 Run 须 AbandonCache 防堆损坏

## 现象

在 Ascend950PR（CANN 9.1.0，`ASCEND_HOME_PATH` 指向 cann-9.1.0）上开发自定义 aclnn 算子 bench_no_harness_cummin 的 ST 全量回归：L0（30 条用例）Real NPU 模式 28 PASS / 0 FAIL / 2 SKIP 全部正常；升级跑 L1 全量（600 条用例，CSV 驱动、每条用例独立走 `aclnnBenchNoHarnessCumminGetWorkspaceSize` → `aclnnBenchNoHarnessCummin` → `aclrtSynchronizeStream` → `aclDestroyAclOpExecutor` 循环）时，运行到 `[15/600]` 打印用例名处直接：

```
run.sh: line 204: 55451 Segmentation fault      (core dumped) ./${TEST_BIN} "${CASE_FILE}" --level "${LEVEL_UPPER}" ${EXTRA_ARGS}
```

EXIT_CODE=139，重跑稳定复现。当时代码状态：op_api 的 `aclnnBenchNoHarnessCummin` 是此前为规避 CANN 9.1.0 下 `CommonOpExecutorRun` + `aclDestroyAclOpExecutor` 的 double free 而写的"绕过"版本——不走 `CommonOpExecutorRun`，直接 `executor->SetStream(stream)` → `UpdateTensorAddr` → `Run()`（该 double free 现象见同场景配对知识卡）。

关键判别信号（隔离实验 + ASan，均为实测）：

1. 崩溃用例（`dtype=float32 shape=(26, 29) dim=1 range=[+0,+0] elems=754`，一个 754 元素的小用例）单独放进 CSV 运行：1/1 PASS，EXIT_CODE=0。
2. 前 15 条用例一起运行：13 PASS / 2 SKIP（2 条 >100M 元素被上限跳过），第 15 条 PASS，EXIT_CODE=0。
3. 给测试程序临时加 `ST_LIMIT` 环境变量，**加载全部 600 条但只执行前 15 条**：第 15 条 PASS，EXIT_CODE=0——证明与"加载条数导致的堆布局差异"无关，也排除"前 14 条执行造成累积损坏"。
4. GDB 因容器 ptrace 权限被拒不可用；改用 ASan（`-fsanitize=address` 编译；直接运行报 `ASan runtime does not come first in initial library list`，需 `LD_PRELOAD=$(gcc -print-file-name=libasan.so.8)` 预加载）后，在完整 600 条加载下于第 15 条捕获：

```
==57513==ERROR: AddressSanitizer: SEGV on unknown address 0x03e90000e0a9 (pc 0x7f3851914401 ...)
==57513==The signal is caused by a WRITE memory access.
    #0 0x7f3851914401 in op::OpExecutorImpl::SetStream(void*) (.../cann-9.1.0/lib64/libnnopbase.so+0x1e9401)
    #1 0x7f385170e7d2 in aclnnBenchNoHarnessCummin (.../opp/vendors/bench_no_harness_cummin_custom/op_api/lib/libcust_opapi.so+0x87d2)
    #2 0x55b5413b18eb in RunCaseReal .../tests/st/test_aclnn_bench_no_harness_cummin.cpp:1042
```

即崩溃点在算子自己的 aclnn 执行体里对 `executor->SetStream()` 的调用，向一个明显损坏的指针（0x03e90000e0a9，非正常堆地址形态）写入；ASan 版只跑前 15 条则无任何错误。host 侧测试代码（std::vector 管理、goto cleanup）经 ASan 全程无 heap 报错，损坏源头在 CANN/op_api 侧的 executor 状态。

## 适用场景

- 工程形态：aclnn 自定义算子（op_api L2 实现层）+ C++ ST 全量回归（每用例独立 Create executor → Run → Destroy 的长循环，数百条以上用例）。
- API 路径：`aclOpExecutor` / `CommonOpExecutorRun`（`opdev/op_executor.h`）、`UniqueExecutor::ReleaseTo`、`executor->AbandonCache / SetStream / UpdateTensorAddr / Run`、`aclDestroyAclOpExecutor`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：platform_wide——同一 CANN 版本下任何手动管理 executor 执行路径的 aclnn 实现，与具体算子无关。
- 路线优先级：recommended_fastpath（有条件：先确认所用 CANN 版本的 `CommonOpExecutorRun` 是否存在 double free；若标准路线无问题，应优先标准路线，本卡是标准路线损坏时的规避配方）。
- 明确不适用 / 未验证场景：使用标准 `CommonOpExecutorRun` 且无 double free 问题的 CANN 版本；单次/少量调用场景（本轨迹 L0 30 条以内未复现）；其他 CANN 版本与 NPU 架构未验证。

## 根因

（单句归因）绕过 `CommonOpExecutorRun` 的手动 Run 路径没有清理 executor 的执行缓存状态——`GetWorkspaceSize` 里的 `UniqueExecutor::ReleaseTo(executor)` 会调用 `(*executor)->AddCache()` 把 executor 登记进执行缓存，而手动路径不经过 `CommonOpExecutorRun` 的缓存收尾，全量循环中缓存里的残留/失效状态逐步累积，最终 `SetStream` 通过失效内部指针写入引发 host 堆损坏段错误。

机制链（均有轨迹内证据）：

1. **缓存登记是自动发生的**：`opdev/op_executor.h`（CANN 9.1.0）中 `UniqueExecutor::ReleaseTo` 的实现：`*executor = uniqueExecutor_.release(); (*executor)->AddCache();`——只要 GetWorkspaceSize 正常返回，executor 就进了缓存。
2. **缓存 API 存在且需成对使用**：同一头文件 `aclOpExecutor` 公开 `AddCache / DeleteCache / FinalizeCache / AbandonCache(bool) / SetStream / UpdateTensorAddr / Run`；`~UniqueExecutor()` 析构未 release 时会调 `DeleteCache()`。
3. **手动路径让登记与收尾脱节**：手动 `SetStream/Run` 不触碰 cache，`aclDestroyAclOpExecutor` 销毁后缓存侧状态与实际堆状态不一致。
4. **规模相关性**：L0 30 条不崩、L1 600 条在第 15 条崩、单条/前 15 条/"加载 600 只执行 15"都不崩——损坏只在"完整 600 条逐条执行"下累积到临界（ST_LIMIT 对照实验把"加载条数/堆布局"与"执行累积"两个变量解耦后确认）。

## 解决配方

修复前（op_api/aclnn_bench_no_harness_cummin.cpp，绕过 CommonOpExecutorRun 的旧版执行体——此前为规避 double free 写的，问题代码）：

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessCummin(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBenchNoHarnessCummin);
    // 绕过 CommonOpExecutorRun 以避免 CANN 9.1.0 的 double free 问题
    // (issue_20260715_executor_double_free_0.md)
    // CommonOpExecutorRun 内部会修改 executor 的 cache 状态,
    // 导致后续 aclDestroyAclOpExecutor 重复释放内部资源。
    // 直接调用 executor->Run() 不触碰 cache, 让 aclDestroyAclOpExecutor 正常清理。
    executor->SetStream(stream);
    if (workspaceSize > 0 && workspace != nullptr) {
        executor->UpdateTensorAddr(workspace, static_cast<size_t>(workspaceSize));
    }
    return executor->Run();
}
```

修复后（在手动路径上先显式弃用缓存；GetWorkspaceSize 保持用户输出张量重载不变）：

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessCummin(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBenchNoHarnessCummin);
    // 绕过 CommonOpExecutorRun 以避免 CANN 9.1.0 的 double-free 问题
    // (issue_20260715_executor_double_free_0.md)
    // CommonOpExecutorRun 内部调用 FinalizeCache 修改 executor cache 状态,
    // 导致后续 aclDestroyAclOpExecutor 重复释放内部资源。
    // 直接调用 executor->Run() 并在执行前 AbandonCache 禁用 cache 机制,
    // 使 aclDestroyAclOpExecutor 不走 cache 释放路径, 避免 heap 损坏。
    executor->AbandonCache(true);
    executor->SetStream(stream);
    if (workspaceSize > 0 && workspace != nullptr) {
        executor->UpdateTensorAddr(workspace, static_cast<size_t>(workspaceSize));
    }
    return executor->Run();
}
```

唯一差异是执行前先 `executor->AbandonCache(true);`。重新编译（`bash build.sh --soc=ascend950 -j8`）并安装（`cd build && bash custom_opp_ubuntu_x86_64.run --force --quiet`）后全量回归通过（见验证信号）。

判别 / 定位方法（本会话走通的三步，可复用于任何 ST 规模相关段错误）：

1. **用例隔离三连**：崩溃用例单独跑（排除用例本身）→ 前 N 条一起跑（排除邻居用例）→ 给测试程序加"加载全量、只执行前 N 条"的限流开关（解耦"加载条数/堆布局"与"执行累积"两个变量）。第三步是定位规模相关堆损坏的关键，本例正是它把假设收敛到"完整执行的累积效应"。
2. **GDB 不可用时用 ASan**：`-fsanitize=address` 重编译；CANN 动态库场景下直接运行会报 `ASan runtime does not come first in initial library list`，需 `LD_PRELOAD=$(gcc -print-file-name=libasan.so.8)` 预加载（CANN 库非 ASan 插桩，抓不到其内部 overflow，但能给出符号化崩溃栈与 SEGV 读写属性）。
3. **读头文件核对生命周期 API**：`$ASCEND_HOME_PATH/include/aclnn/opdev/op_executor.h` 里 `UniqueExecutor::ReleaseTo`（AddCache）与 `aclOpExecutor` 的 cache 方法族对照，找到"谁登记、谁收尾"的缺口。

## 设计决策

- 为什么不直接恢复标准 `CommonOpExecutorRun`：实测恢复后 double free 立即回归（L0 就崩，`free(): double free detected in tcache 2`，EXIT_CODE=134）——标准路线在该 CANN 版本 + 每用例销毁的 ST 模式下不可用（详见坏实践与配对知识卡）。
- 为什么不换标准 AllocTensor + ViewCopy 张量管理再走 CommonOpExecutorRun：实测同样 double free——证明问题与 L0 算子的张量管理方式无关，锁定在 executor cache 生命周期。
- 为什么选 `AbandonCache(true)`：它同时保住两头——保留手动 Run 规避 CommonOpExecutorRun 引发的 double free，又显式弃用缓存让 `aclDestroyAclOpExecutor` 不走 cache 释放路径，消除堆损坏累积源。
- 当前方案路线档位：recommended_fastpath（本场景内的正确解；升级 CANN 版本后应重测标准路线，见失效边界）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` 构建，`custom_opp_ubuntu_x86_64.run --force --quiet` 安装到 `opp/vendors/bench_no_harness_cummin_custom/`。
- 相关文件：`op_api/aclnn_bench_no_harness_cummin.cpp`（执行体与 GetWorkspaceSize）、`tests/st/test_aclnn_bench_no_harness_cummin.cpp`（RunCaseReal 每用例 Create/Run/Destroy 循环）、CANN `$ASCEND_HOME_PATH/include/aclnn/opdev/op_executor.h`（executor/cache API 事实源）、`issues/issue_20260715_executor_double_free_0.md`（double free 历史记录）。
- ST 运行入口：`tests/st/run.sh --real --level L1`（自动 cmake + make + 运行，LD_LIBRARY_PATH 指向 vendor op_api/lib）。

## 预防

- 写 aclnn 执行体时若绕过 `CommonOpExecutorRun`，必须先 `executor->AbandonCache(true)` 再手动 `SetStream/UpdateTensorAddr/Run`；绕过而不弃缓存 = 登记进缓存却无人收尾。
- 任何"小规模全过、全量段错误、崩溃点在框架库内"的 ST 现象，优先做"加载全量 + 限制执行条数"的解耦实验，再上 ASan（LD_PRELOAD 方式）拿崩溃栈，不要先猜用例本身。
- 升级 CANN 版本后回归两条路线（标准 CommonOpExecutorRun / 手动 + AbandonCache），确认 double free 是否已被上游修复。

## 验证信号

以下信号来自 bench_no_harness_cummin 的 ST L1 段错误调试会话：先在 600 用例全量运行下稳定复现第 15 用例段错误，经用例隔离、ST_LIMIT 解耦实验与 ASan 崩溃栈定位后落 AbandonCache 修复，逐级放大复验。

- 修复前：L1 全量 600 用例运行至 `[15/600]` Segmentation fault (core dumped)，EXIT_CODE=139，重跑稳定复现；L0 30 用例正常（来源平台：950；证据等级：tool_observed）。
- 修复前：ASan 捕获 `SEGV on unknown address 0x03e90000e0a9`（WRITE），栈为 `op::OpExecutorImpl::SetStream(void*)` (libnnopbase.so+0x1e9401) ← `aclnnBenchNoHarnessCummin` (libcust_opapi.so) ← `RunCaseReal:1042`（来源平台：950；证据等级：tool_observed）。
- 定位对照：崩溃用例单独运行 1/1 PASS；前 15 条运行 13 PASS/2 SKIP；加载 600 条仅执行前 15 条（ST_LIMIT=15）第 15 条 PASS、EXIT_CODE=0；ASan 版前 15 条无任何报告（来源平台：950；证据等级：tool_observed）。
- 修复后：ST L0 28 PASS / 0 FAIL / 2 SKIP，EXIT_CODE=0（来源平台：950；证据等级：tool_observed）。
- 修复后：L1 全量 600 用例 553 PASS / 0 FAIL / 47 SKIP（SKIP 为 >100M 元素上限），EXIT_CODE=0（来源平台：950；证据等级：tool_observed）。
- 修复后：移除调试用 ST_LIMIT 改动、经标准 `run.sh --real --level L1` 复跑，测试结果 PASS ✓（553/553，0 FAIL）；UT 29 用例全过、L0 复验通过，无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论仅对 CANN 9.1.0 + Ascend950PR 实测；其他 CANN 版本未验证——若新版已修复 `CommonOpExecutorRun` 的 double free，应直接用标准路线，`AbandonCache` 绕过反而会失去缓存复用收益（该收益损失未量化）。
- 损坏触发与"每用例独立 Create→Run→Destroy 的长循环"强相关：单次调用或少量用例（本轨迹 ≤30 条）未复现；更短循环里的表现未验证。
- `AbandonCache(true)` 参数（disableRepeat）的精确语义未查证（仅从头文件签名确认存在）；本卡只验证"调用后问题消失、全部测试通过"。
- kernel 侧行为不受本卡影响（本修复全部在 host 侧 op_api 层）；若段错误根因在 kernel/NPU 侧越界，本卡配方不适用。

**未决项**：

- （优先项）`CommonOpExecutorRun` 内部对 cache 的确切调用序列（是否/何时调用 `FinalizeCache`）未读 CANN 源码核实，当前机制描述部分来自项目内 issue 文档转述；建议在新会话中复现并追根因到机制层（读 libnnopbase.so 对应实现或向 CANN 框架团队确认）。
- `AbandonCache(true)` 的参数语义与作用域（单 executor 或全局缓存池）未查证。
- 弃用缓存对多次同 shape 调用场景的性能代价未测量（本算子 ST 每用例形状不同，影响不可见）。
- 跨 CANN 版本（9.2.x 等）两条路线的行为对照未执行。
## 坏实践

- 绕过 `CommonOpExecutorRun` 时当缓存不存在：不调 `AbandonCache` 就手动 `SetStream/Run`——小规模测试全绿，全量回归才爆段错误，且崩溃点出现在框架库（libnnopbase.so）里，极易误判为框架 bug 或测试 harness bug。
- 恢复 `CommonOpExecutorRun` 前不验证 double free 是否回归：本轨迹两次恢复尝试（用户张量直传、标准 AllocTensor+ViewCopy）都在 L0 立即 double free（EXIT_CODE=134）。
- 见到"600 条第 15 条崩"就直接二分用例找"肇事用例"：本例崩溃用例单独跑完全正常，肇事的不是任何一条用例，而是跨用例累积的 host 侧状态。

## 误诊

- 一度怀疑"第 15 用例自身触发 kernel 内存损坏"：该用例单独运行 PASS，排除。
- 一度怀疑"前 14 用例（含 dim=-7 的 7 维 int32 用例、66M 元素 fp16 用例）累积损坏"：前 15 条整体运行通过，排除。
- 一度怀疑"加载 600 条 CSV 改变堆布局导致"：ST_LIMIT=15（加载 600 只执行 15）通过，排除——只有"完整执行"才触发，最终收敛到 executor 缓存状态累积。
- 两次"换回标准路线"的失败尝试（CommonOpExecutorRun 恢复、AllocTensor+ViewCopy 标准模式）均 double free——反而成为锁定 cache 生命周期根因的关键反证。
