---
type: Guide
title: MX 分组 Scale 广播与 FP4 字节打包
description: 保留四种转置的 Scale 物理排列、低位打包顺序和实际编码输入的 Golden。
tags:
- task.implementation
- topic.compute.quantization
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/scripts/gen_data_mxfp4.py
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/scripts/gen_data_mxfp8.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze_torch.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze.cpp
  role: supporting
created_at: '2026-09-11T02:53:42Z'
updated_at: '2026-09-11T02:53:42Z'
platforms: []
---

# MX 分组 Scale 广播与 FP4 字节打包

来源MX格式每32个K元素共用一个Scale，以两个Scale组成一个64元素K段。令G=ceil(K/64)：A未转置存[M,K]、Scale[M,G,2]，A转置存[K,M]、Scale[G,M,2]；B转置存[N,K]、Scale[N,G,2]，B未转置存[K,N]、Scale[G,N,2]。这描述本样例GM编码，不能直接套到任意NZ或L1格式。

末轴K的矩阵将每个Scale重复32次，再把[G,64]合并为K并裁到有效K；首轴K的矩阵从[G,其他轴,2]扩到[G,其他轴,64]，再转[G,64,其他轴]、合并并裁K。不能先把两个Scale平均，也不能在转置时只交换数据而不改变Scale排列。Scale实际编码为E8M0，uint8是字节载体，不是数值scale。

FP4以两个逻辑值放入一字节，来源第一个值占低4位、第二个占高4位，按物理末轴连续配对。因此A未转置要求K为偶数、A转置要求M为偶数；B转置要求K为偶数、B未转置要求N为偶数。来源pack函数虽有全局补一个元素的分支，其预计算行shape没有同步扩展，不能用于任意奇数行宽；若支持奇数，需明确每行padding、物理stride与消费者合同。

当前FP8 Host分配A=M*K字节、B=K*N字节，Scale分别为M*G*2、N*G*2字节，BF16输出为M*N*2字节。切换FP4时不能只换生成器，还要修改Kernel类型、每行打包容量及所有搬运/布局粒度。C++载体sizeof不是FP4逻辑位宽，见[逻辑与物理存储](../blaze/matmul_logical_shape_and_physical_storage.md)。

Golden先把实际量化输入与实际编码Scale解码到FP32，逐K元素反量化，再按transpose恢复数学A[M,K]/B[K,N]做乘法，最后转BF16。BF16文件以uint16位视图保存，读取应恢复位模式，不能数值cast uint16到BF16。参考FP32 Matmul的累加次序仍可能不同于设备，需采用实际验收合同。

来源随机输入仅正数且Scale取有限正范围，不能覆盖零块、负数、指数极限或特殊编码。增加每32/64段可区分Scale、K=31/32/33/63/64/65、四种transpose和低高nibble不同值，独立核对原始字节与Golden。生成器还同时输出当前目录及scripts目录，测试必须绑定同一批输入，不能混用两处遗留文件。本批未运行依赖这些低比特包的生成器。
