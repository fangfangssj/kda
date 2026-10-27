---
type: "Runbook"
title: "fork 隔离 35 例后驱动耗尽：分批父进程"
description: CANN 9.1.0 单进程顺序执行第二条 ACLNN 用例即堆污染崩溃（销毁 executor 报 double free、不销毁则 SIGSEGV），每用例 fork 子进程完整 ACL 生命周期可隔离；但 fork+aclInit/aclFinalize 循环约 35 次后驱动拒绝新分配并挂起，全量回归须分批（每批 ≤25 用例独立父进程）执行。
tags: [task.testing, topic.architecture]
status: stable
sources: []
created_at: '2026-09-17T02:30:52Z'
updated_at: '2026-09-17T02:30:52Z'
platforms: ["950"]
---

# fork 隔离 35 例后驱动耗尽：分批父进程

## 现象

在 Ascend950PR（CANN 9.1.0）上为一个自定义算子（foreach p-范数类，TensorList 输入 + 标量阶数）编写 C++ 原生 ST 测试（直接链接 aclnn 接口、NPU 执行 + CPU golden 比对）时遇到两个互相挤压的限制：

**限制一：单进程顺序执行多用例即崩。** 第一条 aclnn 用例正常 PASS（MERE=0.00e+00），第二条用例开始就崩溃：

- 若用例结束时显式调用 `aclDestroyAclOpExecutor(executor)`，第一条用例 PASS 后立刻 `free(): double free detected in tcache 2`，进程 Aborted (core dumped)，EXIT 134；
- 若不调用 `aclDestroyAclOpExecutor`（executor 交给 `CommonOpExecutorRun` 内部析构），第一条 PASS，第二条用例（相同代码路径、不同 shape）直接 `Segmentation fault (core dumped)`，EXIT 139。

`MALLOC_CHECK_=0` 与 `MALLOC_CONF_=tcache_count=0` 均无法抑制——不是检测误报，而是实际堆损坏（机制归因见 CommonOpExecutorRun double free 卡：executor 析构污染堆内存；堆污染本体未做 valgrind/ASAN 级定位，见遗留问题）。

**限制二：fork 隔离有驱动累积上限。** 改用每用例 `fork()` 子进程 + 子进程内完整 ACL 生命周期后，小规模全过（10/10）；但全量回归（74 用例）跑到约第 35 个用例（BF16、p=1、1024 元素小张量）时报 `[FAIL] 创建输入 tensor[0] 失败`（`aclrtMalloc` 返回非 0，仅申请 2KB），随后子进程 `[CRASH] signal=11`（SIGSEGV），下一个用例挂起、整轮 600 秒超时。关键对照：同 shape 同 dtype 的更早用例通过；迭代二 60 用例、同样 fork 机制曾 60/60 全过；失败用例抽出为独立单用例程序在全新进程复跑，全步骤返回 0、actual=golden=808、PASS——失败与用例内容无关，与顺序执行的用例数量相关。

## 适用场景

- 工程形态：aclnn 两段式接口的 C++ ST/回归测试工程，NPU 上一个进程内循环执行多条用例（触发限制一）或 fork 进程隔离跑全量（触发限制二）。
- API 路径：`aclnn<Op>GetWorkspaceSize` + `aclnn<Op>` 两段式调用、`CommonOpExecutorRun`、`aclDestroyAclOpExecutor`、`aclInit/aclFinalize`、`aclrtMalloc/aclrtSetDevice`、`fork/waitpid/_exit`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0。
- 适用粒度：platform_wide——CANN 9.1.0 上所有多用例 aclnn ST/回归程序。
- 路线优先级：recommended_fastpath（fork 每用例隔离 + 子进程完整 ACL 生命周期）；全量大规模用例的分批父进程执行为 safety_fallback。fallback 声明：分批以多次进程启动换稳定，代价是总耗时增加与结果聚合需拼接，仅适用于测试场景。
- 明确不适用/未验证场景：单条用例的进程不受影响；用例间需共享设备状态（stream 复用、持久 workspace）时不适用 fork 模式；其他 CANN 版本未验证（未验证，不推断可迁移）；Python/PyACL 侧是否存在同类问题未验证。

## 根因

两个限制同源于 CANN 9.1.0 的 aclnn 执行框架：

1. 同进程第二条用例崩溃：`CommonOpExecutorRun` 内部析构 executor 的堆污染（机制归因为开发会话内推断，未做 valgrind/ASAN 级定位）——无论调用方是否显式销毁 executor，第二条用例都无法在同进程存活；
2. fork 循环约 35 次后失败：跨用例累积的 ACL 驱动/运行时状态——每用例 fork + `aclInit/aclFinalize` 循环执行约 35 次后，驱动对新进程的内存分配请求开始拒绝（2KB 的 `aclrtMalloc` 都失败），失败后的清理路径（`aclrtResetDevice/aclFinalize`）在受损状态下段错误，并把设备留在坏状态使后续用例挂起（驱动到底累积了什么资源、阈值是否精确为 ~35 次，机制层未验证）。

## 解决配方

**第一层：每用例 fork 子进程 + 完整 ACL 生命周期（主进程不预初始化 ACL）。**

修复前（单进程顺序执行，父进程提前 `aclInit`/`aclrtSetDevice`/`aclrtCreateStream`）：

```cpp
    for (const auto& tc : test_cases) {
        if (RunTestReal(tc, stream)) {
            passed++;
        } else {
            failed++;
        }
    }
```

修复后（fork 子进程 + 完整 ACL 生命周期）：

```cpp
    // Real 模式: 每个 test case 在独立子进程中执行，避免 executor 析构导致的堆污染
    int32_t deviceId = 0;

    for (const auto& tc : test_cases) {
        LOG_PRINT("\n[Real] %s - L=%zu, scalar=%.1f, %s",
                  tc.id.c_str(), tc.shapes.size(), tc.scalar, tc.description.c_str());
        fflush(stdout);

        pid_t pid = fork();
        if (pid == 0) {
            // 子进程: 完整 ACL 初始化
            aclInit(nullptr);
            aclrtSetDevice(deviceId);
            aclrtStream childStream;
            aclrtCreateStream(&childStream);

            bool result = RunTestReal(tc, childStream);
            fflush(stdout);

            aclrtDestroyStream(childStream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            _exit(result ? 0 : 1);
        } else if (pid > 0) {
            int status = 0;
            waitpid(pid, &status, 0);
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                passed++;
            } else {
                failed++;
                if (!WIFEXITED(status)) {
                    LOG_PRINT("  [CRASH] signal=%d", WTERMSIG(status));
                }
            }
        } else {
            LOG_PRINT("  [FAIL] fork() 失败");
            failed++;
        }
    }
```

关键点：① 子进程退出必须用 `_exit()`（不走 atexit/flush 双份）；② 父进程**不要**提前 `aclInit`——子进程继承父进程 ACL 状态会得到不可用的运行时；③ 子进程必须从 `aclInit` 到 `aclFinalize` 全套自建，只补 `aclrtSetDevice` 不够（见坏实践）。

**第二层：全量大规模用例分批执行（规避 fork 累积上限）。** 用例数超过 fork 累积安全余量时，按批切分、每批一个独立父进程：

```bash
    # 每批 ≤25 用例独立父进程（测试器支持 --start/--count 切片）
    ./run_st.sh --start 1  --count 25   # 25/25
    ./run_st.sh --start 26 --count 25   # 25/25
    ./run_st.sh --start 51 --count 24   # 24/24 → 拼接 74/74
```

**判别探针（区分"用例问题"与"累积问题"）**：把失败用例的 aclnn 调用序列抽成独立单用例程序，在全新进程中复跑（同 seed 同数据），逐步打印每个 ACL 调用返回码。本例探针输出 `aclInit: 0`、`aclrtMalloc(x): 0`、`GetWorkspaceSize: 0`、执行 `0`、`actual: 808`、`golden: 808`、`diff: 0`、`Result: PASS`——单用例完全正常，锁定套件级累积。

## 设计决策

- 为什么 fork 而不是修 executor 生命周期：`CommonOpExecutorRun` 是 aclnn 标准执行路径，无法在测试侧替换其内部析构行为；"不调用 aclDestroyAclOpExecutor"只能把 double free 换成第二条用例的段错误，说明污染源在执行/析构内部而非显式销毁。
- 为什么子进程要完整 ACL 生命周期：fork 只继承内存副本，NPU 设备上下文不可继承；实测"子进程仅补 aclrtSetDevice"全部用例创建输入张量失败。
- 为什么分批而不是换隔离方案：单进程 no-fork 被限制一封死；fork 被限制二封顶；分批是两个约束下的可行解（后续验收中单父进程 76 次 fork 一次性 74/74 通过过一次——耗尽具间歇性，分批是不依赖运气的稳定形态）。
- 用 `WIFEXITED`/`WTERMSIG` 打印 `[CRASH] signal=N`，把用例崩溃和断言失败区分开。

## 预防

- 写多用例 aclnn C++ ST 时，第一天就按"每用例一个子进程"搭框架，不要先写单进程循环再排查崩溃。
- 全量套件规模超过几十条时直接规划分批执行入口（--start/--count 切片），不要等第 35 例挂起超时才发现上限。
- 见到套件中段 `aclrtMalloc` 失败（哪怕只申请 KB 级内存）+ 子进程 SIGSEGV + 后续用例挂起，且失败用例单独跑完全正常——先跑单用例隔离探针区分"用例问题"与"累积问题"，不要查用例的 shape/dtype。

## 验证信号

来源说明：以下信号来自该算子三轮 ST 联调会话的实测（单进程崩溃复现、fork 隔离闭环、全量分批交付）。

- 显式调用 `aclDestroyAclOpExecutor` 时：首用例 PASS 后 `free(): double free detected in tcache 2`，EXIT 134（来源平台：950；证据等级：tool_observed）。
- 移除显式销毁后：首用例 PASS，第二用例 `Segmentation fault (core dumped)`，EXIT 139；`MALLOC_CONF_=tcache_count=0` 下同样段错误（来源平台：950；证据等级：tool_observed）。
- 子进程仅补 `aclrtSetDevice`（父进程预 `aclInit`）时：10/10 用例全部 `[FAIL] 创建输入 tensor[0] 失败`（来源平台：950；证据等级：tool_observed）。
- 采用 fork + 子进程完整 ACL 生命周期后：10/10 用例 PASS（含多张量、大尺寸、4D 等），EXIT 0（来源平台：950；证据等级：tool_observed）。
- fork 隔离下的全量回归：74 用例跑到约第 35 个用例（BF16 小张量）`aclrtMalloc` 失败 + `[CRASH] signal=11` + 后续用例 600s 挂起；此前 34 个用例全 PASS；迭代二 60 用例同机制曾 60/60（来源平台：950；证据等级：tool_observed）。
- 独立单用例进程（同用例同数据）：`aclInit: 0`、`aclrtMalloc: 0`、`GetWorkspaceSize: 0`、执行 `0`、actual=808、golden=808、diff=0、PASS（来源平台：950；证据等级：tool_observed）。
- 分批交付：每批 ≤25 用例独立父进程，25/25 + 25/25 + 24/24 = 74/74；后续验收单父进程 76 次 fork 一次性 74/74（耗尽具间歇性）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只证明 CANN 9.1.0 + Ascend950PR 环境下该隔离模式有效；其他 CANN 版本未验证（更新版本若修复 executor 析构问题则无需此模式）。
- "~35 次循环"是单次观测的失败位置，不是精确阈值；用例规模（是否含大张量）可能影响累积速度。
- fork 模式假设每条用例独立申请/释放全部设备资源；若用例间需要共享设备状态（stream 复用、持久 workspace），该模式不适用。
- 子进程内不能再 `fork` 嵌套执行（未验证）。
- 探针只证明"单用例无问题 + 套件内有问题"，不定位驱动累积的具体资源。

**未决项**：

- （优先项）复现并追根因到机制层：单进程第二条用例崩溃的准确定位（哪一次 free 破坏 tcache，valgrind/ASAN 级），以及 fork+aclInit/aclFinalize 循环的驱动资源累积点（fd/context/mapping 级别）与精确阈值。
- 该行为是否在 CANN 后续版本修复未跟踪。
## 坏实践

- 子进程只补 `aclrtSetDevice`（继承父进程 ACL 状态）：全部用例 `aclCreateTensor`/`aclrtMalloc` 失败——设备上下文不能跨 fork 继承，必须子进程内从 `aclInit` 开始全套初始化。
- 用 `MALLOC_CHECK_=0` / `MALLOC_CONF_=tcache_count=0` 压掉 tcache 检测：现代 glibc 的 tcache double-free 检测不受 `MALLOC_CHECK_` 控制；且这里是真实堆损坏，抑制检测只会把 abort 换成更隐蔽的段错误。
- 子进程 SIGSEGV 后不清理直接继续跑下一个用例：设备可能留在坏状态，把一个失败放大成后续全部挂起。
- 只因"重跑一次过了"就放行：fork 耗尽具间歇性，稳定交付应分批。

## 误诊

- 最初怀疑崩溃由 `aclScalar*`/`aclTensor*` 标量类型不匹配引起：改对类型后第二条用例仍段错误，证明类型不是崩溃根因。
- 全量第 35 例失败曾疑为该 BF16 kernel 回归（"迭代二该用例通过"）：单用例隔离复跑 PASS 后排除用例/算子回归，定位为框架级累积。
