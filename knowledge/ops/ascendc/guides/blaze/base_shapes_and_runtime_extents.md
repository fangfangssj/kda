---
type: Guide
title: Blaze 硬件基形与运行期尾块范围的区别
description: 将 Block/Copy 的合法 base shape 与 Scheduler 的当前逻辑范围分开，按实际消费者校验 tiling、workspace
  和融合后处理。
tags:
- task.design
- topic.tiling
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/tiling-selection.md
  role: primary
created_at: '2026-09-10T12:37:22Z'
updated_at: '2026-09-10T12:37:22Z'
platforms: []
---

# Blaze 硬件基形与运行期尾块范围的区别

Blaze tiling 字段的含义来自消费它的 Scheduler、Policy、Block 和 Copy 特化。同名字段或相似样例不足以证明两个方案可共用 tiling。

用于片上资源或搬运原语实例化的 baseM/baseN/baseK 是硬件基形，需要满足该特化的粒度、对齐、容量和交叉约束；curM/curN/curK 或等价当前块范围表达本轮有效逻辑数据。将基形设置成 `min(逻辑尺寸, 首选基形)` 可能产生不合法的小块。

一个可执行尾块方案需要同时具备合法 base shape、当前有效范围以及所选 Block/Copy 对该范围的处理能力。若该特化不支持某种尾块，应选择具有对应支持的实现，不能仅靠缩小 base 字段宣称兼容。

## Tiling 与融合的联合约束

Host 生成的字段应逐项对应 Device 消费者，明确类型、单位、合法范围、核数、workspace 与输出偏移。Grouped 输入还要核对组元数据，量化路径还要核对 scale 和 packing。

在 MatMul 后融合 Elementwise/Broadcast Epilogue 时，额外输入、广播映射、UB 暂存、slot 和输出应纳入同一资源及生命周期预算。SplitK/StreamK 产生的 partial 必须先按算法完成合并，再交给要求完整结果的后处理；普通逐 tile Epilogue 不能自动获得 final 语义。

## 验证

对有粒度约束的轴分别覆盖小于粒度、等于粒度、base+1 和多 tile，Grouped 还需覆盖空组。不同失败用例若共享同一个小 M，不能仅依据用例名叫 K-tail 或 N-tail 就定位到对应轴。

本卡提供字段与边界的推导原则，没有给出某个 Blaze 版本的固定合法参数组；具体支持域依赖选定源码特化，故不声明统一平台和固定默认值。
