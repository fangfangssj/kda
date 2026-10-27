---
type: "Runbook"
title: "RunOpApiV2 后再同步同 stream 段错误"
description: "在 OpCommand::RunOpApiV2 返回后对同一 getCurrentNPUStream 调 aclrtSynchronizeStream（为延长 aclIntArray/aclTensor 生命周期）导致进程段错误（3/3 复现）；回退官方 acl_call 模式（参数在 acl_call 内随入队释放、外部不加同步）即恢复。该外部强同步路线 banned。"
tags: [task.implementation, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# RunOpApiV2 后再同步同 stream 段错误

## 现象

在 torch_npu 自定义算子适配层（torch_adapter.cpp 的 forward_npu，经 `OpCommand::RunOpApiV2` 调度 aclnn 两段式）中，出于"确保 aclIntArray/aclTensor 在算子真正执行完之前有效"的动机，尝试把参数销毁延迟到 `RunOpApiV2` 返回之后、并在此之前显式同步 stream（attempt 代码 verbatim）：

```cpp
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    // 注意：ws 不在 acl_call 中销毁，而是在 RunOpApiV2 返回后销毁。
    // RunOpApiV2 内部会同步等待 acl_call 完成（算子入队），但 NPU 上的实际执行
    // 可能异步。aclTensor 持有的是 PyTorch tensor 的 data_ptr（在 y/x 释放前有效），
    // aclIntArray 的数据在 GetWorkspaceSize 时已被 executor 引用。
    // 为确保安全，在 RunOpApiV2 后、delete ws 前，同步 stream 确保算子完成。
    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        return status == ACL_SUCCESS ? 0 : 1;
    };
    at_npu::native::OpCommand::RunOpApiV2("bench_no_harness_strided_slice", acl_call);

    // 同步确保 NPU 上的算子执行完成，再销毁 aclIntArray/aclTensor
    aclrtSynchronizeStream(acl_stream);
    delete ws;

    return y;
```

重新编译 `libtorch_adapter.so` 后，测试脚本在算子调用处直接崩溃（报错原文，连续 3 次一致）：

```text
/bin/bash: line 17: 401528 Segmentation fault      (core dumped) python3 -c "..."
```

（脚本无任何 Python 侧输出即崩，进程在首次算子调用路径上段错误退出。）

## 适用场景

- 工程形态：torch_npu 自定义算子适配层（op-plugin 风格 forward_npu / torch library 扩展）。
- API 路径：`at_npu::native::OpCommand::RunOpApiV2(const string&, const PROC_FUNC&, bool sync=false)` + `aclrtSynchronizeStream` + `c10_npu::getCurrentNPUStream().stream(false)`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0 + torch 2.7.1(cpu) + torch_npu（其头文件 `op_api_common.h` 为官方调用模式参考）。
- 适用粒度：platform_wide（RunOpApiV2 与外部同步的冲突是框架层行为）。
- 路线优先级：banned（在 RunOpApiV2 返回后对同一 stream 外加 aclrtSynchronizeStream 的做法禁止使用；参数生命周期问题不应靠外部强同步解决）。**fallback 声明**：正确路线是回退官方 EXEC_NPU_CMD 宏形态（见解决配方），无性能代价。
- 明确不适用/未验证场景：自建 stream（非 getCurrentNPUStream 获取的当前 stream）上同步未验证；另一变体——把 aclrtSynchronizeStream 放进 acl_call 内部（OpExecute 之后、资源释放之前）不崩，但同样不解决连续执行正确性，也不推荐。

## 根因

RunOpApiV2 对传入 acl_call 的调度与同步有自己的内部机制，宿主线程在其返回后再对同一 stream 调 `aclrtSynchronizeStream` 与该机制冲突导致段错误。（归因依据：官方 `op_api_common.h` 的 EXEC_NPU_CMD_V1/V2 宏均不在外部再同步；本轮未做崩溃栈分析，机制细节未完全证实。）

## 解决配方

本卡为负结果卡：正确"配方"是回退到官方 acl_call 模式——资源（aclTensor/aclIntArray 等转换产物）在 acl_call 内、紧跟算子入队之后释放，宿主侧不在 RunOpApiV2 外部对同一 stream 做任何显式同步。官方模式（`op_api_common.h` 的 EXEC_NPU_CMD_V1/V2 宏内联形态，verbatim）：

```cpp
auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {
    auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");
    ReleaseConvertTypes(converted_params);  // 在 acl_call 中释放转换后的参数
    return api_ret;
};
at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);
```

即：参数转换产物的释放在 acl_call 内、紧跟算子入队之后完成；`RunOpApiV2` 返回即认为提交完成，由框架的内部机制管理执行与可见性。

## 预防

- 封装 aclnn 算子的 PyTorch 适配层严格跟随官方 EXEC_NPU_CMD 宏形态：资源在 acl_call 内随算子入队后释放；不要在 RunOpApiV2 外部对 getCurrentNPUStream 再调 aclrtSynchronizeStream。
- 在未读官方 `op_api_common.h` 宏实现前，不自行设计适配层的同步点。
- "算子执行完前参数必须存活"的需求应通过 executor 持有引用（GetWorkspaceSize 阶段已拷贝/引用所需数据）理解，而不是外部强同步。

## 验证信号

来源说明：本负结果来自该适配层调试过程中的尝试-崩溃-回退对照（Ascend950PR + CANN 9.1.0）：

- 尝试版本（RunOpApiV2 后 aclrtSynchronizeStream）：`Segmentation fault (core dumped)`，连续 3 次复现，发生在算子调用路径上（来源平台：950；证据等级：tool_observed）。
- 回退官方模式后：无崩溃，单条/成对用例通过（来源平台：950；证据等级：tool_observed）。
- 参考事实：官方 op_api_common.h 的 EXEC_NPU_CMD_V1/V2 宏中 acl_call 只调 opApiFunc + ReleaseConvertTypes，外部无 stream 同步（来源平台：CANN 9.1.0 头文件；证据等级：tool_observed）。

## 失效边界

- 仅验证"RunOpApiV2 返回后对 getCurrentNPUStream 同步"这一形态；自建 stream 场景未验证。
- 段错误的底层机制（死锁/野指针/UAF）未定位（无 core 栈分析）。
- 其它 CANN/torch_npu 版本未验证（未验证，推断可迁移——官方宏形态跨版本一致）。

**未决项**：

- 【优先项】复现并追根因到机制层：对 core 做 gdb 栈分析，定位 RunOpApiV2 内部调度与外部 stream 同步冲突的具体形态（死锁/双重释放/UAF）。
- RunOpApiV2 内部 `sync` 形参的准确语义（OpCommand.h 仅见声明 `bool sync = false`）未从实现源确认。
## 坏实践

- 试图用外部 `aclrtSynchronizeStream` "补" aclnn 参数的生命周期；在未读官方 op_api_common.h 宏实现前自行设计同步点。

## 误诊

- 无（崩溃形态单一、回退对照直接）。
