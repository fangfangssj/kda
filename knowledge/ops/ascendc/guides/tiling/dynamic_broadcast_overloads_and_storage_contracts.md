---
type: Guide
title: 动态 UB Broadcast 的重载与存储约束
description: 区分动态 shape 广播接口和既有静态接口，保留容量、地址与空输入边界。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/broadcast/dynamic-ub-broadcast.md
  role: primary
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# 动态 UB Broadcast 的重载与存储约束

来源描述DAV3510路径的动态UB Broadcast：运行期uint32 shape、rank 1–9、srcInnerPad=false、源维等于目标维或为1、源和目标不能重叠，列出11种宽度/类型组合。例如[2,1,4]扩为[2,3,4]。这些是该路径声称的条件，需要与实际版本重载核对；不能据此扩展到任意芯片、dtype、rank或stride。

另一类Broadcast使用编译期维数、源内轴32B补齐或half/float及sharedTmpBuffer等签名。动态shape不要求特定shape对齐，不等于任意起始地址均可非对齐，更不等于取消其他重载的限制。rank、shape元素类型、源内轴是否padding和临时区参数必须作为接口身份一起记录。

Tiling描述不负责分配LocalTensor。仍需按物理padding、数据类型、同时存活的源/目标、scratch及buffer份数计算UB容量，不能只预算广播后的逻辑元素数。扩展坐标在源为1的轴映射为0，其他轴保持坐标；非连续输入还需实际stride，不能直接沿用连续布局公式。

应在乘积和地址计算前处理零维/空张量、rank范围、维度是否可广播及uint32/总元素数溢出；“每维相等或1”也不能替代实际接口对零维和rank0的合同。不能从一例连续三维广播推导所有shape组合已验证。
