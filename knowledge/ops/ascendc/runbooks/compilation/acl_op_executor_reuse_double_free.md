---
type: "Runbook"
title: 复用 aclOpExecutor 致 double free 崩溃
description: "同一 aclOpExecutor 跨多次 aclnnXxx 调用复用触发堆损坏（free(): double free detected in tcache 2 / corrupted size vs. prev_size）；性能循环须每次迭代新建 aclTensor 并重调 GetWorkspaceSize 取新 executor。"
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-17T02:14:14Z'
updated_at: '2026-09-17T02:14:14Z'
platforms: ["950"]
---

# 复用 aclOpExecutor 致 double free 崩溃

## 现象

在 Ascend950PR（CANN 9.1.0）上跑 BenchNoHarnessSwiGlu 的 C++ 性能测试（tests/perf/perf_test.cpp）：对每个 shape×dtype 用例先 `aclnnBenchNoHarnessSwiGluGetWorkspaceSize(...)` 一次拿到 workspaceSize + `aclOpExecutor*`，再在 warmup 5 次 + 测量 50 次的循环里**用同一个 executor 反复调 `aclnnBenchNoHarnessSwiGlu(workspace, size, executor, stream)`**。程序启动后第一轮即崩溃（原文）：

```
free(): double free detected in tcache 2
/bin/bash: line 1: 3592860 Aborted                 (core dumped) ./build/perf_test --output perf_results.json
```

第二版重写（仍复用 executor、仅改 CPU 计时）后换个位置再崩：

```
TestCase                    Op           Input(MB)  Output(MB)     Avg(ms) ...
small_[1024,2048]_FP16      Custom     FAILED: Warmup run failed
corrupted size vs. prev_size in fastbins
/bin/bash: line 1: 3595318 Aborted                 (core dumped) ...
```

两次都是 glibc 堆完整性检查直接 abort，发生在同一 executor 的第二次及以后的 aclnnXxx 调用路径上。

## 适用场景

- 工程形态：aclnn 两段式接口（GetWorkspaceSize + aclnnXxx）的 host 侧批量/循环调用——性能压测、批量推理、反复 launch 的任何场景。
- API 路径：`aclnnXxxGetWorkspaceSize(aclTensor*..., uint64_t* wsSize, aclOpExecutor** executor)` + `aclnnXxx(void* ws, uint64_t wsSize, aclOpExecutor* executor, aclrtStream)`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 aclnn executor 的生命周期语义。
- 路线优先级：recommended_fastpath（每次迭代新建是当前唯一被验证稳定的循环模式）。
- 明确不适用 / 未验证场景：单次调用（建一次 executor 调一次后随框架回收，ST 测试即此模式，从未崩）；executor 跨调用复用是否在更高 CANN 版本受支持（未验证）。

## 根因

CANN 9.1.0 公共头文件没有 `aclDestroyExecutor` 一类显式释放接口（grep `include/aclnn/` 仅见 aclDestroyTensor/TensorList/ScalarList 与 opdev 内部 `ReleaseTo(executor)`），executor 生命周期由框架内部管理；观测事实是：**同一 executor 连同其创建时绑定的 aclTensor 描述，在第二次进入 aclnnXxx 执行路径后触发 double free / fastbins 损坏**——框架内部对 executor（或其持有的参数上下文）的释放与用户侧再次使用发生竞争/重复释放。内部确切机制（哪块内存被谁释放两次）未读到实现层，属黑盒对照结论：复用 → 必崩；每次新建 → 稳定。

## 解决配方

修复思路：性能循环里**每次迭代完整重建调用上下文**——新建 `aclTensor`（输入/输出各一）→ 重调 `GetWorkspaceSize` 取新 executor → 调 `aclnnXxx` → 销毁本次 aclTensor。workspace 只按首次探测的大小分配一次、跨迭代复用（workspace 是纯内存，不带生命周期语义）。

修复后核心结构（tests/perf/perf_test.cpp RunPerfCase，verbatim）：

```cpp
    auto outputShape = ComputeOutputShape(tc.inputShape, tc.dim);

    // 性能测试策略: 每次迭代创建新的 executor (GetWorkspaceSize + aclnnXxx)
    // 对大 shape, kernel 时间远大于 GetWorkspaceSize 开销
    // 对小 shape, 通过批量执行 amortize 开销

    // 先确定 workspace 大小
    uint64_t wsSize = 0;
    {
        aclTensor* inTensor = CreateAclTensor(inputDev, tc.inputShape, tc.dtype);
        aclTensor* outTensor = CreateAclTensor(outputDev, outputShape, tc.dtype);
        aclOpExecutor* exec = nullptr;
        if (opType == OpType::CUSTOM) {
            ret = aclnnBenchNoHarnessSwiGluGetWorkspaceSize(inTensor, tc.dim, outTensor, &wsSize, &exec);
        } else {
            ret = aclnnSwiGluGetWorkspaceSize(inTensor, tc.dim, outTensor, &wsSize, &exec);
        }
        aclDestroyTensor(inTensor);
        aclDestroyTensor(outTensor);
        if (ret != 0) {
            result.errorMsg = "GetWorkspaceSize (probe) failed: " + std::to_string(ret);
            aclrtFree(inputDev);
            aclrtFree(outputDev);
            return result;
        }
    }

    // 分配 workspace
    void* workspace = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            result.errorMsg = "workspace malloc failed: " + std::to_string(ret);
            aclrtFree(inputDev);
            aclrtFree(outputDev);
            return result;
        }
    }

    // 每次迭代的完整调用: GetWorkspaceSize + aclnnXxx
    auto runFull = [&]() -> bool {
        aclTensor* inTensor = CreateAclTensor(inputDev, tc.inputShape, tc.dtype);
        aclTensor* outTensor = CreateAclTensor(outputDev, outputShape, tc.dtype);
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnStatus r1;
        if (opType == OpType::CUSTOM) {
            r1 = aclnnBenchNoHarne...  // (逐字截断于事件流，下接 aclnnSwiGlu 分支同构)
```

修复前（崩溃版）核心差异：`GetWorkspaceSize` 只在循环外调用一次，`aclTensor` 与 `executor` 都是循环外一份，循环体内仅反复 `aclnnXxx(workspace, wsSize, executor, stream)`。

## 设计决策

- 选"每次迭代重建全部调用上下文"而非只重建 executor：GetWorkspaceSize 的签名同时产出 executor 与绑定的参数描述，只重建一半无法保证配对；且实测 GetWorkspaceSize 单次开销远小于大 shape 用例的 kernel 执行时间（来源平台 950 上的同轨迹对照），对小 shape 由统计多次取均值摊薄。
- workspace 跨迭代复用：workspace 是按 wsSize 分配的裸 device 内存，无 executor 语义，复用安全且省 aclrtMalloc/aclrtFree 抖动。
- 计时口径改为包含 GetWorkspaceSize 的 CPU 侧 chrono + 流同步：避免事件 API 与崩溃变量纠缠；报告中注明该口径对小 shape 会略偏高。

## 预防

- 任何"循环里反复调 aclnnXxx"的代码（压测、服务化推理）一律按"一次调用 = 一次 GetWorkspaceSize + 一次 aclnnXxx"组织，不要缓存 executor 复用。
- 诊断此类堆崩溃先看崩溃点是否总在第二次及以后的调用——是则优先怀疑跨调用复用的句柄（executor/tensor 描述），而非业务逻辑。

## 坏实践

- 假设"executor 只是计算图描述，可像 CUDA Graph handle 一样反复 launch"——CANN 9.1.0 不成立。
- cleanup lambda 里对 executor 写空操作分支（`if (executor) { /* not freeing */ }`）却继续复用它，掩盖生命周期责任不清的问题。

## 验证信号

来源：性能验收会话中 perf harness 两轮崩溃与重写后的全量运行。

- 复用版两次崩溃原文：`free(): double free detected in tcache 2`（首轮）与 `FAILED: Warmup run failed` + `corrupted size vs. prev_size in fastbins`（第二版 warmup 阶段）（来源平台：950；证据等级：tool_observed）。
- 头文件 grep 确认公共接口无 aclDestroyExecutor/Release 类显式释放（来源平台：950；证据等级：tool_observed）。
- 改为每次迭代新建后全量运行成功：9 个用例（小/中/大 shape × FP16/FP32/BF16）× 2 算子（自定义 + 基线 aclnnSwiGlu）全部出数、无任何崩溃（含大 shape 用例）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "复用必崩"是 9.1.0 黑盒对照结论，内部机制（谁释放了什么两次）未定位；不能推广为"CANN 所有版本 executor 不可复用"。
- 每次迭代重建引入 GetWorkspaceSize 开销进计时窗口；对 kernel 时间与 GetWorkspaceSize 开销同量级的小 shape，该口径会高估时延（本轨迹 small 用例数据已含此偏差，使用小 shape 绝对值时须注意）。
- 未验证场景：多 stream 并发下同模式；同进程混合多算子循环。

**未决项**：

- 优先项：读 CANN 内部实现或用 ASan 定位 double free 的确切内存块与释放路径，把"不可复用"从观测升级为机制结论。
- 机制层候选解释（跨卡推断，本轨迹未采集 plog）：同批已 staged 的配对卡（CommonOpExecutorRun 加销毁循环 double free）有 plog 实证——CANN 9.1.0 的 CommonOpExecutorRun 在执行段内即触发 `~aclOpExecutor` 析构；若本算子执行体同样走该入口，第二次调用复用的即是已析构的悬空指针，与本处"第二次及以后调用必崩"的观测吻合，接续时可用该卡的 plog 判别命令核实。
- 评估是否存在官方支持的批量 launch 接口（如算子队列/Graph 方式）替代逐次重建。
