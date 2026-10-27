---
type: Guide
title: Blaze 融合输出的整行量化与阶段边界
description: 在 Matmul 或 GLU 后按完整逻辑行求 scale，区分 workspace 行距、分块宽度、量化语义和异步完成范围。
tags:
- task.design
- topic.compute.quantization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/per-token-quant-development.md
  role: primary
created_at: '2026-09-10T14:12:04Z'
updated_at: '2026-09-10T14:12:04Z'
platforms: []
---

# Blaze 融合输出的整行量化与阶段边界

设融合计算产生逻辑 workspace `W[R,Q]`，每个输出 token 对应一整行。per-token scale 的统计域是该行全部 Q 个有效元素；单个 Matmul N tile 或一个量化 chunk 不能独立计算 scale 后冒充整行量化。物理行距可以大于 Q，但 padding 不属于逻辑统计域。

## 数值合同

对约定 `a=max_j(abs(W[r,j]))`、`s=max(a,scaleMin)/quantMax` 的对称量化，不能改为 `s=max(a/quantMax,scaleMin)`。这两种 clamp 的位置和单位不同。量化整数范围、舍入规则、零行以及非有限数的处理需要来自实际算子定义；不能从 INT8 dtype 推出统一的 `[-127,127]` 或 `[-128,127]` 合同。

整行装入 UB 时，应同时计入原值、绝对值/归约临时区、转换结果、输出和异步在途槽位。整行不能常驻时，可第一遍按 chunk 求全行 absmax，得到最终 scale 后第二遍重读原值，所有 chunk 使用同一 scale。两遍读取要求输入内容在该期间保持不变。已有 [DynamicQuant A2/A3 实现](../../operators/aclnn/nn/quant/dynamic_quant.md)记录了完整行与两遍扫描的具体实现，本卡补充融合生产者的完成范围。

## 融合与完成范围

Matmul/Elementwise/GLU 的各输出 tile 可能由不同任务生成。量化某行之前，必须保证该行所有有效列均已完成最终写入；进入下一阶段不能只等待当前 tile。若存在 SplitK，完整 K 部分和还必须先合并。

scale 写出完成与整行量化结果写出完成是不同事件。UB 输出槽位要等搬出结束后才能复用；workspace 要等所有读取者完成后才能覆写。只有证明两个阶段不重叠并完成必要资源重初始化，才可按两个阶段预算的最大值分配，而非把声明在不同函数当作内存已经释放。

## 验证与边界

覆盖 Q 跨越多个 N tile/chunk、物理行距大于 Q、零值行、尾行、正负数及舍入半整数附近。scale 相同只说明行最大值相同，不能排除其他列的 workspace 值出错。应分别比较失败列的设备中间值、设备 scale 和相同舍入链；浮点 ULP 差异在量化阈值附近可能改变整数结果。

本卡来自固定版本技术说明，未执行设备验证，不给出统一的 Blaze 模板参数、UB 容量或误差放宽阈值。
