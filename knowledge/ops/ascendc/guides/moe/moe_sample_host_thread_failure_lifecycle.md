---
type: Guide
title: MoE 样例 Host 多卡线程失败与资源生命周期
description: MoE 样例 Host 多卡线程失败与资源生命周期的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/test/test_moe_dispatch.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/run_multi_data_test.sh
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 样例 Host 多卡线程失败与资源生命周期

单进程先每device创建context/stream、HcclCommInitAll建域，再每rank线程SetCurrentContext、构造tiling和通信资源、分配拷贝并launch。WaitForLaunchBarrier用atomic计数与condition_variable：最后到达线程看到计数齐备后notify_all；源码声明g_hcclAllocMutex却没有加锁使用，不能把声明当串行资源分配事实。

某线程在barrier前失败并goto cleanup，其余线程仍等initDoneCount==rankSize，既没有失败状态广播也没有超时，会阻塞join；stream的10000ms同步超时在launch后，不能救这个Host barrier。计数/g_canLaunch也未重置，函数不能原样当多轮屏障。可共享失败/取消状态并唤醒等待者，配合每rank资源状态处理部分初始化。

InitializeRuntime仅完成stream后才增加initializedRanks，若当前rank已建context而建stream失败，cleanup循环不覆盖当前部分资源。LaunchOneThread清理HCCL和自有buffer，context由HCCL管理不aclrtFree；main最终释放stream/context/device/runtime。分配到device tiling后的函数早退、线程早退和某些HCCL创建失败路径应逐一核对ownership，不能只凭正常逆序清理证明无泄漏。

WriteFile检查一次write是否完整，但WriteLaunchArtifacts忽略返回值，输出目录未创建/短写可能仍报Host成功；需让文件错误传播给最终校验并避免旧文件掩盖。CLI atoi/atoll没有rank/BS/H正值或设备范围检查，负数转size_t可能巨额分配；HCCL初始化成功不验证算子shape容量。调试指针打印不是资源可访问性证明。
