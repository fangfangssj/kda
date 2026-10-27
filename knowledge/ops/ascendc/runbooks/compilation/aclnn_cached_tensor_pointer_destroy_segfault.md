---
type: "Runbook"
title: "同进程连跑 Segfault：勿销毁框架已缓存 tensor"
description: ACLNN 框架在 L0 调度（GetOpArgContext/op_cache）中跨调用缓存 tensor 指针；同进程多用例 cleanup 里 aclDestroyTensor/aclDestroyTensorList/aclDestroyIntArray 销毁 host 对象后缓存悬垂，下一个用例进入 ACLNN 调度时段错误（exit 139）。cleanup 只释放 device memory、host tensor 对象留给进程退出回收即恢复。
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T02:32:55Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# 同进程连跑 Segfault：勿销毁框架已缓存 tensor

## 现象

在 Ascend950PR（CANN 9.1.0）上给自定义分组矩阵乘算子跑 C++ ST：CSV 驱动的多个测试用例在**同一进程**里循环执行，每个用例结束调用 cleanup 释放资源。现象是：**任意单用例独跑都通过；一旦连跑，第 2 个用例必在 ACLNN 调用阶段 Segmentation fault（exit 139，core dumped）**。测试输出停在用例头信息之后；加 fprintf 调试后定位到崩溃点在 `GetWorkspaceSize` 内 L0 API 调用（`l0op::<Op>(...)` → `ADD_TO_LAUNCHER_LIST_AICORE`）内部，且调试打印显示第 2 个用例 L0 层收到的输出 tensor 指针（yOut=0x5561ed6697d8）与第 1 个用例**完全相同**——框架在跨用例复用缓存的对象。当时的 cleanup 是"教科书式"的全量销毁：`aclDestroyTensorList` + 逐个 `aclDestroyTensor` + `aclDestroyIntArray` + `aclrtFree`。

## 适用场景

- 工程形态：自定义 aclnn 算子的 C++ ST/上层调用方，在同一进程内连续调用 `aclnnXxxGetWorkspaceSize → aclnnXxx` 多次（不同 shape/参数的用例循环）。
- API 路径：`aclCreateTensor`/`aclCreateTensorList`/`aclCreateIntArray` 与对应的 `aclDestroy*`；ACLNN 框架的 `GetOpArgContext`/op_cache；`aclDestroyAclOpExecutor`（`aclnn/acl_meta.h`，由 `libnnopbase.so` 导出）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0。
- 适用粒度：platform_wide——由 ACLNN 框架缓存行为决定，与具体算子无关。
- 路线优先级：recommended_fastpath（同进程循环调用场景的清理策略）；用例数极大或要求零泄漏时退到每用例独立进程（safety_fallback，代价是进程/ACL 初始化开销）。
- 明确不适用 / 未验证场景：每用例独立进程（无跨调用缓存共享）不受影响；长生命周期服务在同一组 tensor 上反复调用（不销毁不重建）也不触发本问题；其他 CANN 版本的缓存策略未验证。

## 根因

ACLNN 框架在 L0 调度（`ADD_TO_LAUNCHER_LIST_AICORE` → `GetOpArgContext`/op_cache）中跨调用缓存 tensor 指针。用例 cleanup 调 `aclDestroyTensor`/`aclDestroyTensorList`/`aclDestroyIntArray` 销毁 host 侧对象后，缓存里仍留着这些地址；下一个用例再次进入 ACLNN 调度时解引用已释放内存 → 段错误。另一个相关约束：`aclDestroyAclOpExecutor` 不能手动调用——CANN 9.1.0 的 `CommonOpExecutorRun` 内部管理 executor 生命周期，手动释放直接 double free（见 [CommonOpExecutorRun 加销毁循环 double free 卡](common_op_executor_run_double_free.md)）。

## 解决配方

用例间 cleanup **只释放 device memory，不销毁框架可能已缓存的 tensor 类 host 对象**（tensor 对象随进程退出回收，属于可接受的小泄漏）；executor 只置空不手动释放。

修复前（全量销毁——第 2 个用例必崩）：

```cpp
    auto cleanup = [&]() {
        executor = nullptr;  // executor 由 ACL 框架管理，不主动释放
        if (y_list) aclDestroyTensorList(y_list);
        for (auto t : y_tensors) { if (t) aclDestroyTensor(t); }
        for (auto d : y_devs) { if (d) aclrtFree(d); }
        if (gl_array) aclDestroyIntArray(gl_array);
        if (bias_tensor) aclDestroyTensor(bias_tensor);
        if (w_tensor) aclDestroyTensor(w_tensor);
        if (x_tensor) aclDestroyTensor(x_tensor);
        if (bias_dev) aclrtFree(bias_dev);
        if (w_dev) aclrtFree(w_dev);
        if (x_dev) aclrtFree(x_dev);
    };
```

修复后（只释放 device memory，指针清零防 double free；host tensor 对象留给进程退出回收）：

```cpp
    auto cleanup = [&]() {
        // 多用例兼容：只释放 device memory，不销毁 tensor 对象
        // 原因：ACL 框架的 GetOpArgContext 会缓存 tensor 指针，
        //       销毁 tensor 后缓存引用悬空指针，导致下一个用例 segfault
        // tensor 对象会随进程退出自动释放（可接受的小内存泄漏）
        executor = nullptr;
        for (auto d : y_devs) { if (d) aclrtFree(d); }
        if (bias_dev) { aclrtFree(bias_dev); bias_dev = nullptr; }
        if (w_dev) { aclrtFree(w_dev); w_dev = nullptr; }
        if (x_dev) { aclrtFree(x_dev); x_dev = nullptr; }
        y_devs.clear();
        y_tensors.clear();
        y_list = nullptr;
        gl_array = nullptr;
        x_tensor = nullptr;
        w_tensor = nullptr;
        bias_tensor = nullptr;
    };
```

executor 侧（`GetWorkspaceSize` 成功、`aclnnXxx` 执行并 `aclrtSynchronizeStream` 之后）**不要**手动释放：

```cpp
    if (workspace) aclrtFree(workspace);

    // 注意：不调用 aclDestroyAclOpExecutor — CANN 9.1.0 中 CommonOpExecutorRun
    // 已内部管理 executor 资源，手动释放会导致 double free
```

## 设计决策

- 为什么选择"泄漏 host 对象"而不是"每用例独立进程"：进程级隔离最彻底，但本会话先尝试 fork 失败——父进程已初始化 ACL/stream 时子进程设备上下文失效（`aclrtSetDevice`/`aclrtMalloc` 报错）；"只放显存、留对象"在 13 用例规模下泄漏可忽略（每个 tensor host 头部几百字节）。同轨迹后续把执行循环改为**父进程不初始化 ACL、每用例 fork 子进程完整走 aclInit→SetDevice→CreateStream→_Exit** 后隔离成功（687/687 验证，见 [aclnn TilingData 缓存致 runtime attr 陈旧卡](../precision/aclnn_tilingdata_runtime_attr.md) 的 fork 兜底段）——fork 路线要求父进程不持有 ACL 状态。
- 为什么不留 `aclDestroyAclOpExecutor`：实测手动调用即 double free（`aclDestroyAclOpExecutor` 虽由 `libnnopbase.so` 导出且链接可解析，但 CANN 9.1.0 的 `CommonOpExecutorRun` 已接管生命周期）。
- 当前方案路线档位：recommended_fastpath（对本场景）；若用例数极大或要求零泄漏，退到每用例独立进程（safety_fallback，代价是进程/ACL 初始化开销，未在本轨迹量化）。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0；`aclDestroyAclOpExecutor` 声明于 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/acl_meta.h`，符号在 `libnnopbase.so`（`nm` T 导出）。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（RunRealTest 的 cleanup 与执行循环）。
- 复现：同进程 CSV 双用例（第 2 个用例 shape/参数与第 1 个不同）→ 第 2 用例 `ADD_TO_LAUNCHER_LIST_AICORE` 阶段 exit 139。

## 预防

- 同进程循环调用 ACLNN 的测试/服务：资源释放分两层——device memory（`aclrtFree`）随用例释放，`aclTensor*`/`aclTensorList*`/`aclIntArray*` host 对象不销毁或进程级统一回收。
- 不要手动调用 `aclDestroyAclOpExecutor`（CANN 9.1.0 实测 double free）。
- 排障段错误时在 ACLNN 入口与 L0 API 前后插 `fprintf(stderr, ...); fflush(stderr);`（stdout 缓冲在崩溃时会丢），把崩溃点钉到具体调用。

## 验证信号

来源说明：以下信号来自该算子 ST 联调与修复会话：从"第 2 用例必崩"到"同进程 13 用例全过"。

- 修复前：单用例独跑 PASS；同进程连跑第 2 用例 Segmentation fault（exit 139，core dumped），崩溃点定位在 `GetWorkspaceSize` → L0 API（`ADD_TO_LAUNCHER_LIST_AICORE`）内；调试打印显示第 2 用例 yOut 指针与第 1 用例相同（框架缓存复用）（来源平台：950；证据等级：tool_observed）。
- 中间反证：cleanup 中先 `aclDestroyAclOpExecutor(executor)` 的版本出现 double free 崩溃；用例间 `aclrtResetDevice` + 重建 stream 无效；fork 每用例子进程中子进程 `aclrtMalloc` 失败（设备上下文不能跨 fork 继承）（来源平台：950；证据等级：tool_observed）。
- 修复后：cleanup 改为只释放 device memory、不销毁 tensor 对象、不手动释放 executor，同进程连跑 13 用例（用例子集 CSV）无段错误，13/13 PASS（EXIT CODE: 0）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡解决"销毁对象导致的段错误"；同进程内**相同 shape、不同参数（如转置/非转置）**交替执行时仍会出现结果错乱（框架缓存复用旧执行上下文），那是另一个独立问题（见 [aclnn TilingData 缓存致 runtime attr 陈旧卡](../precision/aclnn_tilingdata_runtime_attr.md)），本卡的"不销毁"方案并不能避免它。
- host tensor 对象泄漏在用例数极大（数千以上）或长驻进程中需要进程级回收策略（未验证）。
- "executor 由框架管理"是 CANN 9.1.0 的实测行为，其他版本须重验（未验证，推断可迁移）。

**未决项**：

- 框架缓存（GetOpArgContext/op_cache）的确切缓存键与失效时机未追到源码层；本卡处置是行为级规避。
- 大规模用例下的 host 对象累计泄漏量未量化。
## 坏实践

- 用例 cleanup 里对 `aclTensor*`/`aclTensorList*`/`aclIntArray*` 逐个 `aclDestroy*`（同进程多调用场景）。
- 手动 `aclDestroyAclOpExecutor`（CANN 9.1.0 下 double free）。
- 崩溃排障只看 stdout——缓冲丢失会误判崩溃点位置。

## 误诊

- 曾假设 executor 未释放是根因，先做"cleanup 中先销毁 executor 再销毁 tensor"——引发 double free，说明 executor 生命周期归框架；真正的悬垂源是 tensor 对象销毁。
- 曾怀疑并尝试跳过 `l0op::Contiguous`（以为它把 tensor 注册进 executor 导致泄漏）——跳过后段错误依旧，证实与 Contiguous 无关，随后恢复了 Contiguous 调用。
