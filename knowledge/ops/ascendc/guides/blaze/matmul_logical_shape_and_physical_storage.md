---
type: Guide
title: Blaze Matmul 逻辑形状与物理存储的一致性
description: 协调 transpose、ND/NZ 布局、数据生成、物理容量和 slice 消费语义，避免逻辑 shape 正确但设备地址错位。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-matmul-layout.md
  role: primary
created_at: '2026-09-10T14:27:23Z'
updated_at: '2026-09-10T14:27:23Z'
platforms: []
---

# Blaze Matmul 逻辑形状与物理存储的一致性

Matmul 的数学对象 A[M,K]、B[K,N] 与设备输入字节可以采用不同表示。transpose 标志、GM 布局 Pattern、数据生成顺序、Host 分配容量和 Kernel 的取址必须共同描述同一变换，不能只修改其中一个开关。

对来源采用的特定 NZ 排列，二维物理轴 (dim0,dim1) 被组织为 `(ceil(dim1/C0), ceil(dim0/16), 16, C0)`；dim0 按 16、dim1 按该 dtype/布局的 C0 补齐。16 bit 输入常用 C0=16，8 bit 输入常用 C0=32；这不推导所有张量位置与打包类型都采用同一个 C0。实际容量来自补齐后的物理元素数与载体字节宽度。

如果先对数学矩阵转置再做 NZ 排列，补齐和容量计算应使用转置后的两个物理轴。A、B 的类型可能不同，各自应按自己的布局粒度计算；不能直接复用仅取 `sizeof(AType)` 的示例代码。

子 Tensor 可能保留父 Tensor 的 stride，而具体搬运实现又可能从子 shape 推导内部块跨度。连续多次 slice 是否正确，应检查两者如何组合消费；“先切 M/N 再切 K”是来源给出的特定 Copy 路径适配方式，不能替代任意版本的地址推导。

普通浮点输入的 Golden 应来自约定的逻辑输入，并将设备格式转换限制在独立副本；当接口的真实输入合同是已编码的量化字节时，Golden 则应解码实际交付设备的字节。不能用量化前值评判量化后输入的算子，也不能用错误物理排列反推一个随之变化的 Golden。

验证应跨越首个分形块和首个 tile，分别改变 transpose、格式、M/N/K 尾块及 dtype。单 tile 或偏移为零的样例通过，不能证明父 stride 与子 shape 在多块路径中一致。具体 Copy 路由边界参见[布局与 Copy 特化](tensor_layout_and_copy_specialization.md)。
