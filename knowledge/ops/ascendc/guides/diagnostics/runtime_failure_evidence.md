---
type: Guide
title: Ascend C 运行异常的二进制、路由与执行证据
description: 将加载产物、TilingKey 变体、ABI、异步错误和内存同步分层核对，避免依据错误码或输出不变直接认定根因。
tags:
- task.debugging
- topic.compilation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-runtime-debug/references/kernel_binary_debug.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-runtime-debug/references/error_codes.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-crash-debug/references/crash_workflow.md
  role: supporting
created_at: '2026-09-10T12:38:25Z'
updated_at: '2026-09-10T12:38:25Z'
platforms: []
---

# Ascend C 运行异常的二进制、路由与执行证据

## 先确认执行对象

修改 Kernel 后输出不变，应先确认对应源文件参与了构建，再核对构建产物、安装产物和运行时实际选择的二进制及配置。哈希可以证明两个文件的字节是否相同，但哈希未变化不单独证明缓存错误：修改可能没有进入该目标，或者不影响最终代码。

多个 vendor 或安装位置可能造成选择与预期不一致。应以当前环境、实际加载路径和运行日志核对，而不是假设各版本都采用同一个固定搜索顺序。清理应针对已确认的构建或算子条目，不能将删除整个安装树当作诊断前提。

## 路由和 ABI

Host 下发的 dtype、TilingKey 或模板参数组合，必须能在实际构建的 Kernel 变体中找到匹配入口。声明列表、SEL 组合、Host 编码、Kernel 解码与部署配置需要一致；某个 dtype 单独失败时，这条映射链值得重点核对。

二进制后缀或 funcEntry 的编号规则取决于编译方案，不能固定理解为 dtype 值或列表序号。应检查实际生成配置与对应宏的定义。已有 [TilingKey/funcEntry 不匹配案例](../../runbooks/compilation/aclnn_361001_tilingkey_funcentry_sel_position.md)给出一种具体实现的证据与边界。

`opParaSize` 与 `sizeof(TilingData)` 不同，需要核对真实参数布局、序列化与工具链 ABI。不能仅凭数值差异通过 pragma pack 强行消除 padding，也不能把“小幅差异”自动判为安全。

## 错误上报与故障定位

ACLNN 参数、内部 Tiling/Kernel 查找、Runtime 执行属于不同层级。调用成功后，异步执行错误仍可能在 stream 同步时出现。返回码用于确定排查入口，还需结合详细错误信息、plog、实际 Kernel 和失败阶段。

例如 361001 不能单独证明某个 dtype 变体缺失，507035 也不能单独证明 UB 容量不足。卡死时应核对循环终止、队列交接、缓冲释放以及每个等待是否存在匹配的生产者；崩溃则可结合 Host core 栈与 Device 异常信息区分位置。

验证应先复现原失败组合，再覆盖其他 dtype、路由和尾块；记录实际产物哈希、入口选择和执行结果。错误码不变也不意味着所有底层原因相同。
