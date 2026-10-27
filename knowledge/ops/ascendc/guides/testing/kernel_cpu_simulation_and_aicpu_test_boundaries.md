---
type: Guide
title: Kernel CPU 仿真与 AICPU UT 的证据边界
description: 提取 tikicpulib 的参数链、资源与输出验证，纠正缺库即删测试及固定 workspace 示例。
tags:
- task.testing
- topic.memory
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ut-guide/op-kernel-ut-guide.md
  role: primary
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# Kernel CPU 仿真与 AICPU UT 的证据边界

来源 Ascend C Kernel UT 使用 tikicpulib 的 GmAlloc/GmFree、ICPU_SET_TILING_KEY 和 ICPU_RUN_KF 执行 CPU 仿真；AICPU UT 则经 NodeDefBuilder 构建节点并 RUN_KERNEL。它们覆盖不同执行环境，CPU 仿真通过不证明目标 NPU 的流水时序、资源限制或性能。

手工 TilingData 适合有明确字段合同的情况，需初始化所有消费字段，并使 key、blockNum、workspace 和分区一致。ExecuteTiling 路线可从实际 Host Tiling 得到这些数据，调用成功、workspaceSizes 是否为空、tilingDataSize 与目标缓冲容量仍需检查，不能无条件索引 workspaceSizes[0]。两种方法各自检测的链路不同。

GmAlloc 的容量按真实 dtype、storage 和 padding 需求计算；输入需初始化，输出需与预期逐元素或按合同比较，最后释放所有资源。来源手工示例只分配/运行/释放，没有输出比较，不能称其已经测试正确性；16MiB 或 16781184 字节 workspace 只是示例。32 字节对齐不修复错误索引、容量不足或未初始化访问。

目标模式与编译特化要匹配实际 Kernel：来源某些 Vector UT 需要 AIV_MODE，AddOpTestCase 的 SoC 与 -DDTYPE_X 选择也影响实际被测实例。不能推广为所有 Kernel 都设同一模式。找不到 -lpem_davinci 应检查依赖、链接路径和该版本仿真支持，缺库本身不证明算子不支持 op_kernel，更不是删除测试的理由。

AICPU NodeDef 的所有输入输出、形状、类型、数据指针及属性必须符合真实签名。来源 ADD 示例的 CREATE_NODEDEF 宏只写一个输入、一个输出，却传入两个输入与输出的数组，不能原样视为有效 Add UT。应从节点实际创建内容确认第二输入和输出下标。

op_kernel 与 op_kernel_aicpu 是两类目录角色；来源将其写成绝对互斥，但不同后端/分发可以共存，不能从这一模板推导所有仓只能有其一。本卡仅提取机制，未执行仿真或设备测试。
