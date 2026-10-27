---
type: "Runbook"
title: CommonOpExecutorRun 加销毁循环 double free
description: 'CANN 9.1.0 下 ST 每用例 Create→CommonOpExecutorRun→aclDestroyAclOpExecutor 循环触发 "free(): double free detected in tcache 2"（SIGABRT/134）；与 L0 算子张量管理方式无关，标准 AllocTensor+ViewCopy 模式同样复现；规避路线见 AbandonCache 配对卡。'
tags: [task.debugging, topic.memory, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# CommonOpExecutorRun 加销毁循环 double free

## 现象

调试 bench_no_harness_cummin 的 ST 全量段错误（见同场景配对知识卡：aclnn 手动 Run 须 AbandonCache 防堆损坏）时，为验证根因归属，把 op_api 的 `aclnnBenchNoHarnessCummin` 从"绕过版"恢复为标准执行路线：

```cpp
return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
```

重新编译（`bash build.sh --soc=ascend950 -j8`）并安装（`custom_opp_ubuntu_x86_64.run --force --quiet`）后，ST **L0（仅 30 条用例）** 第一条用例执行完进入清理阶段就崩：

```
free(): double free detected in tcache 2
EXIT_CODE=134
```

（134 = SIGABRT，glibc tcache 检测到重复释放。）随后又把 GetWorkspaceSize 改成**标准 AllocTensor + ViewCopy 模式**（L0 算子只收 executor、内部 AllocTensor 分配输出张量，再 `l0op::ViewCopy` 拷回用户输出张量，消除"L0 直接写用户张量"这一自定义因素），配合 `CommonOpExecutorRun` 重跑 L0：**仍然同样 double free，EXIT_CODE=134**。即该现象与 L0 算子侧张量管理方式无关，两次独立复现。对照：改回"手动 Run + `AbandonCache(true)`"路线后，同一套 L0/L1 全量测试全部通过。

背景：这一 double free 最早在早期联调时发现（当时临时注释掉 ST 里的 `aclDestroyAclOpExecutor` 绕过，存在内存泄漏）；随后用"绕过 CommonOpExecutorRun"规避；本会话的两次恢复实验把"标准路线在该场景不可用"钉死为可复现事实。

## 适用场景

- 工程形态：aclnn 自定义算子（op_api L2 实现层）+ C++ ST 测试，**每用例真正调用 `aclDestroyAclOpExecutor` 销毁 executor** 的循环模式。
- API 路径：`CommonOpExecutorRun`（`opdev/op_executor.h:210` 声明）、`aclDestroyAclOpExecutor`、`UniqueExecutor::ReleaseTo`（内部 `AddCache`）、`aclOpExecutor::AddCache/DeleteCache/FinalizeCache/AbandonCache`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0。
- 适用粒度：platform_wide——同一 CANN 版本下任何走标准 CommonOpExecutorRun 且逐用例销毁 executor 的 aclnn ST 工程。
- 路线优先级：**banned**（本场景内：CANN 9.1.0 + 逐用例销毁循环下，标准 CommonOpExecutorRun 路线不可用；规避路线 = 手动 Run + AbandonCache，见配对卡）。
- 明确不适用 / 未验证场景：不销毁 executor（容忍泄漏）的调用方；复用同一 executor 的重复下发模式；其他 CANN 版本（9.2.x 等未测试，可能已修复）。

## 根因

（单句归因）`CommonOpExecutorRun` 与后续 `aclDestroyAclOpExecutor` 的 executor 缓存生命周期不匹配，销毁时走 cache 释放路径重复释放内部资源。

机制说明与证据边界：`opdev/op_executor.h`（CANN 9.1.0，轨迹内读到的头文件原文）确认 `UniqueExecutor::ReleaseTo` 会 `AddCache()` 登记、`aclOpExecutor` 暴露 `AddCache/DeleteCache/FinalizeCache/AbandonCache`——缓存机制真实存在且参与生命周期；但"`CommonOpExecutorRun` 内部调用 `FinalizeCache` 等修改 cache 状态"这一步来自项目内 issue 文档（`issues/issue_20260715_executor_double_free_0.md`）的记载，**本轨迹未读 CommonOpExecutorRun 实现源码验证**，根因停在"生命周期不匹配"层，未追到内部调用序列（见遗留问题）。

## 判别方法

1. 触发条件三要素：CANN 9.1.0 + `CommonOpExecutorRun` 执行 + 测试侧每用例 `aclDestroyAclOpExecutor`。
2. 复现信号：`free(): double free detected in tcache 2`、EXIT_CODE=134（SIGABRT），L0 小规模即可复现（无需全量）。
3. 排除法归因：换 L0 张量管理方式（用户张量直传 ↔ 标准 AllocTensor+ViewCopy）double free 不消失 → 与算子侧张量管理无关，锁定 executor 生命周期。
4. 反证：同套测试在"手动 Run + `AbandonCache(true)`"路线下 L0/L1 全量通过 → 差异变量只剩 CommonOpExecutorRun 的 cache 收尾路径。

## 设计决策

- 记录为 negative_result 而非 repair_recipe：本卡的闭环是"两条标准姿势都被证伪 + 规避路线被验证"，修复配方（AbandonCache）在同场景配对知识卡里单独成卡，避免一张卡混两个根因。
- 路线判 banned 的依据：两次独立 tool_observed 复现（不同张量管理模式），且崩溃发生在算子包安装后的任何 ST 入口，无法通过算子侧代码规避。

## 系统信息

- Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` + `custom_opp_ubuntu_x86_64.run --force --quiet` 安装。
- 相关文件：`op_api/aclnn_bench_no_harness_cummin.cpp`（aclnn 执行体，两次改动的对象）、`tests/st/test_aclnn_bench_no_harness_cummin.cpp`（cleanup 中 `aclDestroyAclOpExecutor` 调用点）、`issues/issue_20260715_executor_double_free_0.md`（历史记录）、CANN `include/aclnn/opdev/op_executor.h`。

## 预防

- CANN 9.1.0 上写 aclnn ST 时，若测试需要逐用例销毁 executor，先在 L0 级别（几十条）验证 `CommonOpExecutorRun` 路线是否 double free，再铺全量用例。
- 看到 `free(): double free detected in tcache 2` + EXIT_CODE=134 且调用链含 aclnn 算子，先查 executor 生命周期（谁创建、谁运行、谁销毁、缓存谁收尾），不要先怀疑算子张量管理或测试 harness。
- 不要用"注释掉 aclDestroyAclOpExecutor"当修复——那是用内存泄漏换通过（早期临时方案），全量长跑会耗尽内存。

## 验证信号

以下信号来自同一段 ST 段错误调试会话中两次"恢复标准路线"的对照实验。

- 恢复 `CommonOpExecutorRun`（L0 算子直接使用用户输出张量）：ST L0 运行即 `free(): double free detected in tcache 2`，EXIT_CODE=134（来源平台：950；证据等级：tool_observed）。
- 标准 AllocTensor + `l0op::ViewCopy` 模式 + `CommonOpExecutorRun`：ST L0 仍 `free(): double free detected in tcache 2`，EXIT_CODE=134——排除张量管理方式归因（来源平台：950；证据等级：tool_observed）。
- 对照组（手动 Run + `AbandonCache(true)`）：同套 ST L0 28 PASS / 0 FAIL / 2 SKIP、L1 600 用例 553 PASS / 0 FAIL / 47 SKIP、EXIT_CODE=0（来源平台：950；证据等级：tool_observed）。
- API 事实锚点：`op_executor.h` 中 `UniqueExecutor::ReleaseTo` 实现含 `(*executor)->AddCache()`；`aclOpExecutor` 公开 `AddCache/DeleteCache/FinalizeCache/AbandonCache(bool)`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论仅对 CANN 9.1.0（Ascend950PR）实测；未验证，推断不可迁移到已修复该问题的新版本——升级 CANN 后应先重测标准路线。
- 触发依赖"测试侧真正调用 `aclDestroyAclOpExecutor`"：若调用方从不销毁 executor（只泄漏），double free 不会出现（但内存会涨）。
- L0 即可复现意味着触发与用例规模无关；但"多少条以内一定不崩"未测（本轨迹 30 条内必崩于首条清理）。

**未决项**：

- （优先项）根因未追到机制层：`CommonOpExecutorRun` 内部对 cache 的确切调用序列（`FinalizeCache`/`AddCache`/`DeleteCache` 的调用点与顺序）未读 CANN 源码核实，需在新会话复现并追到 libnnopbase.so 实现层。
- 是否为 CANN 9.1.0 已知 bug / 是否有官方 patch 未查证（issue 中"联系 CANN 框架团队"未执行）。
- 其他 CANN 版本与平台上的行为未验证。
## 坏实践

- 早期临时方案：注释掉 ST cleanup 里的 `aclDestroyAclOpExecutor` 让测试跑完——用泄漏掩盖生命周期 bug，全量长跑有耗尽内存风险，且掩盖了真正的 cache 生命周期问题。
- 把本现象当"测试 harness bug"处理（改测试期望/跳过用例）：崩溃在算子库调用链内，与测试断言无关。

## 误诊

- 曾按"L0 实现直接写用户输出张量导致重复释放"归因：换成标准 AllocTensor + ViewCopy 后仍 double free，排除。
- issue 历史中曾逐一排除 `Contiguous`/`ViewCopy` 函数、清理顺序（先销毁 executor 还是先释放 workspace）、`MALLOC_CHECK_=0`——均与 double free 无关（来自项目内 issue 文档记载，非本轨迹实测）。

## 增量（plog 析构时序证据 + 不销毁形态 + fork 每用例隔离兜底）

另一独立轨迹并入 BenchNoHarnessForeachNorm（Ascend950PR，CANN 9.1.0）ST 的独立复现，把本卡遗留问题中"`CommonOpExecutorRun` 内部是否已析构 executor"推进到 plog 实证，并补齐"不显式销毁"路径的失败模式与测试侧兜底：

**1. plog 析构时序证据（本卡遗留问题的机制层信号）**：plog 中 executor 析构发生在执行函数返回**之前**、由 `CommonOpExecutorRun` 触发（verbatim 节选，行号为 plog 行号）：

```text
64226:[INFO] OP(...): ... [op_executor.cpp:969][NNOP][CommonOpExecutorRun] ... Workspace addr: 0x12004c600000, size: 16777728, executor: 0x55d751597b98
64394:[INFO] OP(...): ... [op_executor.cpp:346][NNOP][~aclOpExecutor] ... ~aclOpExecutor, addr: 0x55d751597b98
64399:[INFO] OP(...): ... [aclnn_bench_no_harness_foreach_norm.cpp:197] ... Leaving function aclnnBenchNoHarnessForeachNorm
```

`~aclOpExecutor`（64394）先于 `Leaving function`（64399）出现且 executor 地址一致——第二段调用完成后 executor 已被框架析构，测试侧再调 `aclDestroyAclOpExecutor` 即第二次释放；报错落在其后的下一个 `free()`（恰为局部 vector 析构），故"清理全打印完成、随后 double free"。判别命令（plog 位于 `~/ascend/log/debug/plog/`）：

```bash
grep -nE "~aclOpExecutor|aclDestroyAclOpExecutor|CommonOpExecutorRun|ReleaseTo|Leaving function" \
    ~/ascend/log/debug/plog/plog-*.log | grep <OpName> | tail -40
```

结论规则：执行段日志已见 `~aclOpExecutor` ⇒ 不要再调 `aclDestroyAclOpExecutor`。

**2. 不显式销毁路径的失败模式（双路径两难）**：不调 `aclDestroyAclOpExecutor` 时首用例通过，但第二用例 `GetWorkspaceSize` 报 `errno[161002] dtype 43 not supported`（新张量 `tensor_ address is (nil)`）、第三用例起 `std::bad_alloc`——plog 显示 executor 其实已被框架析构（并非泄漏），但析构路径留下影响后续调用的残留（机制未验证，见遗留问题）。同进程下一个用例出现 `tensor_ address is (nil)` / dtype 43 / bad_alloc ⇒ 上一用例的 ACLNN 调用污染了进程堆。`aclrtResetDevice + aclrtSetDevice` 逐用例重置实测无效（与本卡 先前增量 轨迹"aclrtResetDevice/aclFinalize 无效"独立复证）。

**3. 测试侧兜底：fork 每用例隔离（safety_fallback）**：显式销毁（double free）与不销毁（跨用例污染）两路径在进程内无解时，测试框架层每用例 fork 子进程执行、父进程只收集结果——单用例内的堆污染被进程边界隔离。fallback 声明：以多进程换正确性，代价是每用例一次进程复制与 ACL 上下文重建，仅适用于测试场景，不建议用于在线服务：

```cpp
for (const auto& tc : test_cases) {
    pid_t pid = fork();
    if (pid == 0) {                    // 子进程：完整走一遍 ACL 初始化 + 单用例 + 清理
        ...
        _exit(RunTestReal(tc, stream) ? 0 : 1);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    bool passed = WIFEXITED(status) && (WEXITSTATUS(status) == 0);
    ...
}
```

单用例内的清理顺序（进程内路径配合使用）：先 `aclrtFree(workspace)`/`aclDestroyTensorList`/`aclDestroyScalar`/`aclDestroyTensor`，executor 不显式销毁（CANN 9.1.0 上会 double free）。

**4. 误诊增量**：曾把 double free 归因于 `l0op::Contiguous` 返回原张量导致 executor 与测试双释放——移除 Contiguous/AllocTensorList 后仍 double free，排除；曾把不销毁路径的 dtype 43 完全归因于 executor 泄漏——plog 证明 executor 已被析构，污染另有根因（同 trace 的 scalar 类型混淆，另卡）。

验证信号（复现增量，foreach 范数 ST 内存污染修复会话）：

- 显式销毁路径：清理日志全部完成后 `free(): double free detected in tcache 2`，SIGABRT 退出码 134（来源平台：950；证据等级：tool_observed）。
- plog 判别：`~aclOpExecutor`（executor 0x55d751597b98）由 CommonOpExecutorRun 触发，先于 `Leaving function aclnnBenchNoHarnessForeachNorm`（来源平台：950；证据等级：tool_observed）。
- 不显式销毁路径：下一用例 `GetWorkspaceSize` 报 161002 dtype 43、plog `tensor_ address is (nil)`，后续 `std::bad_alloc`（来源平台：950；证据等级：tool_observed）。
- fork 隔离后（叠加 scalar 类型修复）：ST 通过率 100%（10/10）（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- "执行段已析构 ⇒ 显式销毁 double free"基于 CANN 9.1.0 观测；其他版本 executor 所有权模型可能不同，迁移前先跑 plog 判别（未验证，可迁移的是判别方法）。
- fork 隔离不修复污染本身：单用例进程内如还有越界写，崩溃只是被限制在该用例。
- fork 子进程继承父进程 ACL 状态的语义（SetDevice/Context 是否重做）依赖实现细节。
- `aclDestroyAclOpExecutor` 的正确使用前提（何种 executor 归调用方销毁）仍未从文档/头文件注释确认。

坏实践（复现增量）：

- 在执行段日志已见 `~aclOpExecutor` 后仍调 `aclDestroyAclOpExecutor`。
- 认为"不崩溃就不泄漏"：不显式销毁路径不崩溃首用例，但把失败推迟到第二用例，更难归因。
- 多用例同进程是危险默认：测试框架从第一天就按"每用例一进程"组织最稳。

## 增量（原始发现轨迹的第一手探针信号）

另一独立轨迹并入该 double free 的原始发现会话（bench_no_harness_cummin 迭代一联调 + 迭代二直调改造，同算子同环境 Ascend950PR/CANN 9.1.0），把本卡「误诊」节中引自项目 issue 文档的排除实验补齐为同轨迹实测（tool_observed），并补齐迭代二直调改造后的单轮复测信号：

**1. 发现时序（迭代一）**：标准 `CommonOpExecutorRun` 路线下，每用例执行成功（aclnnXxx ret=0、`aclrtSynchronizeStream` ret=0）后，cleanup 的 `aclDestroyAclOpExecutor(exec)` 处稳定 `free(): double free detected in tcache 2` → core dump；当时代码状态：GetWorkspaceSize 内部 `CREATE_EXECUTOR()` + `uniqueExecutor.ReleaseTo(executor)`（ReleaseTo 会 `AddCache()`），ST cleanup 顺序为 exec → ws → tensors → device buffers。

**2. 排除实验（全部同轨迹实测，升级原 issue 文档转述的排除证据）**：

- 跳过 aclnn 执行（仅 GetWorkspaceSize + cleanup）→ 无崩溃；
- 跳过 `aclDestroyAclOpExecutor` → 30 用例全部跑完（fp16 5 例 PASS）——确认冲突环节在「执行 + 销毁」的组合，与配对卡 plog 证据（执行段内 `~aclOpExecutor` 已析构）吻合；
- 移除 Contiguous、移除 ViewCopy、调换清理顺序（先销毁 exec 或先 free ws）、`MALLOC_CHECK_=0` → 均不能消除崩溃。

**3. 迭代二直调改造复测**：L2 API 改为 `executor->SetStream(stream)` + `UpdateTensorAddr` + `Run()`（绕过 CommonOpExecutorRun，即配对卡「aclnn 手动 Run 须 AbandonCache 防堆损坏」的修复前形态），ST 侧恢复 `aclDestroyAclOpExecutor` 调用，重编译重装后单轮 L0 Real 模式不再出现 double free。

**4. 演化链澄清**：仅直调而未禁用 cache 时，长序列（数百用例连跑）会演变为 heap 损坏延迟崩溃，最终形态为执行前 `AbandonCache(true)`——该演化的完整判别链（ASan 栈、ST_LIMIT 解耦实验）与全量回归证据已在配对卡闭合，本增量不重复；原始会话对最终形态全量回归的数字摘要与配对卡略有出入，以配对卡逐用例输出留存的实测数据为准。

### 复现增量验证信号

- 迭代一失败复现：执行后 cleanup 于 `aclDestroyAclOpExecutor` 处 `free(): double free detected in tcache 2` → core dump（来源平台：950；证据等级：tool_observed）。
- 探针（同轨迹实测）：跳过 aclnn 执行（仅 GetWorkspaceSize + cleanup）无崩溃；跳过 `aclDestroyAclOpExecutor` 则 30 用例全部跑完（fp16 5 例 PASS）；移除 Contiguous、移除 ViewCopy、调换清理顺序、`MALLOC_CHECK_=0` 均不能消除崩溃（来源平台：950；证据等级：tool_observed）。
- 迭代二直调改造后：ST 恢复 `aclDestroyAclOpExecutor` 调用重编译重装，单轮 L0 Real 模式不再出现 double free（来源平台：950；证据等级：tool_observed）。

## 增量（六条独立 trace 复现：ASAN 栈、第一段失败路径边界、官方宏对照与两类坑）

另一独立轨迹并入六条独立 trace 的同根因复现（Ascend950PR/950，CANN 9.1.0 ×5 + CANN 9.2.0 ×1，均为 ST/测试工程清理段手动 `aclDestroyAclOpExecutor` 触发），把定位手段与边界补齐：

**1. ASAN 栈定位（比 plog 更直接的一次定性手段）**：`-fsanitize=address` 重编测试（链接报 "ASan runtime does not come first in initial library list" 时调整链接方式重试），完整栈：

```text
==102478==ERROR: AddressSanitizer: attempting double-free on 0x5020002a4110 in thread T0:
    #0 operator delete[](void*)
    #1 aclTensorList::~aclTensorList()               libnnopbase.so
    #2 aclOpExecutor::~aclOpExecutor()               libnnopbase.so
    #3 aclOpExecutor::~aclOpExecutor()               libnnopbase.so
    #4 aclDestroyAclOpExecutor                       libnnopbase.so
    #5 operator() .../tests/st/test_aclnn_<op>.cpp:587   ← 测试 cleanup lambda
```

栈含 `aclDestroyAclOpExecutor → aclOpExecutor::~aclOpExecutor → aclTensorList::~aclTensorList → operator delete[]` 即本坑。排除库加载干扰的对照：用同一组 `.so`（libascendcl/libnnopbase/libcust_opapi）链接最小 hello-world 正常退出，即排除静态初始化问题。低成本替代（ASAN/plog 都不可用时）：在两段式每步与每个清理动作前后插桩日志重跑一次，可把崩溃精确到语句级。

**2. 第一段失败路径边界（何时暴露）**：`GetWorkspaceSize` 内 `CheckParams` 失败时 `CHECK_RET` 提前返回，`UniqueExecutor` 析构函数自行销毁 executor 且从不执行 `ReleaseTo`，输出参数 `*executor` 保持 nullptr（`op_executor.h` 中 `~UniqueExecutor` 会打 `errno[561102] "When %s do success, ReleaseTo(executor) should be called before return."`）——因此 cleanup 的 `if (executor)` 不触发，double free **只在第一段成功、第二段已执行的场景暴露**（本增量的一条 trace 在前置的 161002 参数校验修复后才首次撞上）。

**3. 修复形态两变体（均实测有效）**：a) 统一删除 aclnn 调用后的全部手动销毁（正常/错误分支都删，留注释防"补全"）；b) 第二段调用后立即 `executor = nullptr;`（cleanup 不动）——b 同时覆盖"第一段成功但第二段未执行（如 workspace 分配失败提前返回）"时 executor 仍有效、需显式销毁的路径。

**4. 官方对照证据**：`op_api_common.h` 的 `EXEC_NPU_CMD_V1/V2` 宏全程只 `ReleaseConvertTypes`（转换参数），无任何 executor 销毁调用；`ascendc-registry-invoke-template/references/add_example` 的调用方代码 `aclDestroyAclOpExecutor` 出现 0 次（grep 计数）——executor 只创建、只使用、从不销毁。

**5. 命名坑**：不存在 `aclopDestroyExecutor` 符号（编译报 undeclared），销毁接口实名 `aclDestroyAclOpExecutor`（`aclnn/acl_meta.h:74`）——不要按 aclop* 旧接口家族类推新 API 名。

**6. 动机坑（banned）**：为"刷新 tiling/executor 缓存"在清理段补销毁——实测既不能刷新缓存（缓存不随 executor 销毁失效），又直接引入 double free；框架缓存的控制点是 op_api 层 `uniqueExecutor->AbandonCache(true)`（见 [aclnn TilingData 缓存致 runtime attr 陈旧卡](../precision/aclnn_tilingdata_runtime_attr.md)）。

**7. 版本扩展**：CANN 9.2.0 同样复现并修复（插桩定位崩溃在 destroyExecutor 行、移除后回归 PASS），本卡结论跨 9.1.0/9.2.0 成立。

### 复现增量验证信号

- ASAN 栈指向 cleanup 中 `aclDestroyAclOpExecutor`（`aclOpExecutor::~aclOpExecutor → aclTensorList::~aclTensorList → operator delete[]`），8/8 用例无法执行；置空修复后 ST Real 全流程运行到精度比对无崩溃（来源平台：950；证据等级：tool_observed）。
- 添加 `aclDestroyAclOpExecutor` 后 `free(): double free detected in tcache 2` + Aborted，3/3 复现；移除（executor 仅置空）后崩溃消失、恢复正常执行（来源平台：950；证据等级：tool_observed）。
- 首个用例 aclnn ret=0、流同步完成后的清理段 double free；`add_example` 参考代码 `aclDestroyAclOpExecutor` 出现 0 次；移除后用例正常走完输出比对（来源平台：950；证据等级：tool_observed）。
- 修复前 ST 启动即 double free、exit 134，plog 显示 aclnn 执行与流同步已完成（崩溃在 host 清理路径）；同机参考算子 ST（不销毁 executor）55/55 PASS；移除后 10 用例 8 过（余 2 条为独立的 ColSplit 路径问题）（来源平台：950；证据等级：tool_observed）。
- 为刷新 tiling 缓存补销毁：运行即 `free(): double free detected in tcache 2` + Aborted；回退该行后恢复，最终以 op_api 层 `AbandonCache(true)` 完成修复、全量 5/5 通过（来源平台：950；证据等级：tool_observed）。
- CANN 9.2.0：最小用例在 destroyExecutor 处 100% 崩溃（插桩定位：GetWorkspaceSize status=0 / execute status=0 / sync ok / D2H ok 之后）；移除手动销毁后同用例 PASS（MERE=0），回归 2 用例 PASS=2 FAIL=0（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- "第二段后置空"变体假设第二段内部已析构（与本卡先前增量 plog 证据一致）；第二段返回错误时是否也已销毁未逐路径验证。
- ASAN 栈与插桩定位均要求可重编测试工程；闭源调用方只能靠 plog/输出判别。


## 增量（四个新独立 trace：调用方侧销毁形态全景 + aclIntArray 边界 + 执行后堆污染的全泄漏兜底）

另一独立轨迹并入四个不同算子/工程（AdamW 类标量属性算子、conv2d 类 ListInt 属性算子、TensorList 算子、foreach 范数类算子的 PyTorch 适配层）的独立复现，把"调用方不得销毁 executor"从单一形态扩展为全景，并补两条新边界：

**1. API 面事实与官方示范锚点**：CANN 9.1.0 头文件中销毁函数名为 `aclDestroyAclOpExecutor`（`aclnn/acl_meta.h:74` 声明），不存在 `aclDestroyExecutor`——写调用代码前先 grep 头文件确认符号名。官方 add_example 的 ST 清理代码只含 workspace/tensor/device 内存三类，**不调用** `aclDestroyAclOpExecutor`。

**2. 算子侧所有权判别法（写清理代码前先做）**：查 `op_api/aclnn_<op>.cpp` 是否出现 `CREATE_EXECUTOR()`（展开为 `UniqueExecutor(__func__)` RAII 对象）与 `uniqueExecutor.ReleaseTo(executor)`——是则所有权在框架，调用方不得 destroy。基于 opdev 框架（CREATE_EXECUTOR）实现的自定义 aclnn 入口，测试侧手动 `aclDestroyAclOpExecutor` 必然二次释放。

**3. 调用方侧销毁的三种崩溃形态（均为首用例即崩或启动即崩）**：

- 5~6 处撒网式销毁（成功路径 + 全部错误路径）：首用例输出任何结果之前 `free(): double free detected in tcache 2` 直接崩；
- `corrupted size vs. prev_size in fastbins` + Aborted（另一 glibc 堆元数据破坏报错形态）；
- `MALLOC_CHECK_=0` 下仍崩（真堆破坏，非检测误报；现代 glibc 的 tcache double-free 检测不受该开关抑制）。

**4. aclIntArray 属性边界（单次销毁即触发，与调用组合无关）**：对挂有 `aclIntArray` 属性（经 `NnopbaseAddIntArrayAttr` 挂入 executor）的算子，**只调一次** `aclDestroyAclOpExecutor`（不再调 `aclDestroyIntArray`）同样 double free——冲突不在用户侧调用组合，而在销毁路径内部（机制未追到 libnnopbase 实现层）。配套符号事实：生成代码把 `NnopbaseMatchArgs` 声明为 weak 符号，但 `nm -D libnnopbase.so` 显示其为强符号（`T`）——试图用弱符号覆盖改变销毁/匹配行为不可行。

**5. 执行后堆污染的全泄漏兜底（PyTorch 适配层形态）**：aclnn 两段式执行完成后进程堆已处于污染状态时，其后**任何一次**堆释放（`aclDestroyTensor`/`aclDestroyTensorList`/`aclDestroyAclOpExecutor`/`delete`）都可能命中被破坏的堆元数据——lambda 内提前销毁、同步流后按创建逆序销毁、仅 executor 泄漏三种组合均实测崩溃；唯一稳定形态是**全泄漏**（不调任何 `aclDestroy*`、不 `delete ws`，acl 句柄交进程退出回收，`torch::Tensor` 照常 RAII、流同步后析构安全），配合每用例独立进程控制累积后全量 74/74 干净退出。重新引入显式销毁 double free 立即复现，回到全泄漏再干净退出——干预-复现闭环成立。

验证信号（复现增量，四个独立 trace，均为 tool_observed）：

- AdamW 类算子 ST：加入 6 处 `aclDestroyAclOpExecutor` 后首用例即 `corrupted size vs. prev_size in fastbins` + Aborted (core dumped)；`sed` 全部删除后全量测试恢复运行（原 5 例精度失败不受影响，与本问题独立）；算子侧入口源码含 `auto uniqueExecutor = CREATE_EXECUTOR();` 与 `uniqueExecutor.ReleaseTo(executor);`（来源平台：950；证据等级：tool_observed）。
- TensorList 算子 ST：首用例 `[PASS] MERE=1.76e-09` 后 `free(): double free detected in tcache 2`；`MALLOC_CHECK_=0` 下仍崩溃；移除 4 处显式销毁后不再 double free，进程进入第二用例（随后暴露的段错误为 TensorList 成员销毁这一独立缺陷，另卡）（来源平台：950；证据等级：tool_observed）。
- conv2d 类算子 ST：5 处销毁（成功 + 4 错误路径）程序启动即 `free(): double free detected in tcache 2`；对照 add_example 官方 ST 清理代码不含 `aclDestroyAclOpExecutor`；移除后 43 用例全部运行到数值比对（来源平台：950；证据等级：tool_observed）。
- API 事实锚点：`aclnn/acl_meta.h:74 ACL_FUNC_VISIBILITY aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor);`，公开头文件无 `aclDestroyExecutor`（来源平台：950；证据等级：tool_observed）。
- aclIntArray 边界：只调 `aclDestroyAclOpExecutor` 不调 `aclDestroyIntArray` 仍 `free(): double free detected in tcache 2` + Aborted；`MALLOC_CHECK_=0` 无效；回退不销毁后全量 ST 43/43 PASS；`nm -D libnnopbase.so` 输出 `0000000000287be0 T NnopbaseMatchArgs`（强符号）而生成代码中为 weak 声明（来源平台：950；证据等级：tool_observed）。
- PyTorch 适配层全泄漏：v1（OpCommand + 析构销毁）单用例精度 PASS 但退出 Aborted；lambda 内提前销毁仍 double free；绕过 OpCommand 直调 + executor 泄漏变 `double free or corruption (fasttop)`；同步后逆序销毁仍崩；全泄漏后同用例干净退出 EXIT_CODE=0；重新引入显式销毁复现、再回全泄漏又干净退出；全泄漏版 + 每用例子进程最终全量 74/74（含确定性 3/3）（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- "调用方不销毁 executor"按 CANN 9.1.0 + 两段式自定义算子 aclnn（opdev/CREATE_EXECUTOR 框架）实测；若使用 `aclSetAclOpExecutorRepeatable` 等执行器复用高级模式，所有权约定不同（未验证）。若某 aclnn 接口文档明确要求调用方释放 executor，则必须按文档释放——判别法见上。
- 全泄漏方案不适合长驻进程（推理服务、训练循环内反复调用）：acl 句柄持续累积会耗尽内存；长驻场景需按调用轮次重启进程或验证新版 CANN 的显式释放。泄漏只覆盖 acl 句柄；`torch::Tensor` 的 NPU 内存仍由 PyTorch 管理，流同步后析构安全（实测干净退出）。
- aclIntArray 边界的 double free 与具体算子/属性个数的关系未隔离（本例 3 个 ListInt 属性；单属性或无属性场景未验证）。
- "执行后堆已污染、任何释放都可能崩"的机制层（哪一次分配/释放破坏了堆元数据）未追到源码级，客观闭环是"全泄漏干净退出 + 引入销毁即复现"。

坏实践（复现增量）：

- 依据"executor 未释放导致状态泄漏"的推理链给所有错误路径撒网式补 `aclDestroyAclOpExecutor`"防止泄漏"——泄漏猜想未成立，销毁反而引入确定性崩溃。
- 在 aclnn 执行后的进程里"按创建逆序仔细释放"——顺序不是问题（三种顺序方案全部失败），花时间调整销毁顺序是无效功。
- 精度 PASS 但退出 Aborted 就当作测试通过——崩溃进程可能已把后续用例的状态破坏，退出码必须纳入判定。
- 试图在测试程序里弱符号覆盖 `NnopbaseMatchArgs` 改变 executor 匹配/销毁行为——libnnopbase.so 中为强符号，覆盖函数根本不被调用。
- 从内置算子示例照抄 `aclDestroyAclOpExecutor` 模板：内置算子与自定义算子（opdev 框架）的 executor 所有权语义可能不同。

误诊（复现增量）：

- "executor 泄漏导致后续用例状态污染"——加释放前后失败用例集合完全一致，方向错误（真正的用例间失败是 tiling 缓存键不含属性，另卡）。
- PyTorch 适配层首轮曾把退出崩溃归因为"OpWorkspace 析构与 OpCommand 框架对 executor 的清理冲突，提前销毁/绕开框架即可"——三种局部修复均失败证伪；实际是执行路径本身留下污染堆，与宿主侧销毁位置无关。
- 曾怀疑 `l0op::Contiguous` 返回原张量导致 executor 与测试双释放——对照官方示例确认参考实现同样走 Contiguous 但不销毁 executor，排除。