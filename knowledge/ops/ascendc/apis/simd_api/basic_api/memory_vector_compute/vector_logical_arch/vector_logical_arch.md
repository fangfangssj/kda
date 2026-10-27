---
type: API
title: Vector逻辑架构
description: Vector计算单元专用于执行向量计算。如下图所示，高亮部分展示了Vector计算单元及其关联的存储单元。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/Vector_logical_arch/Vector_logical_arch.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Vector逻辑架构<a name="ZH-CN_TOPIC_0000002573902865"></a>

## Vector逻辑架构说明

Vector计算单元专用于执行向量计算。如下图所示，高亮部分展示了Vector计算单元及其关联的存储单元。


- 针对Atlas A3系列产品、Atlas A2系列产品，Vector逻辑架构如下图1所示。

  **图1** 向量计算单元架构图

  ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/a2a3_npu_arch.png)

  **表1** 向量计算单元关联的存储单元

  | 存储单元 | 说明 | 
  | --- | --- |
  | Unified Buffer（UB） | AI Core内部物理存储单元，通常用于存储向量计算的输入/输出数据。 |




- 针对Ascend 950PR&950DT系列产品，Vector逻辑架构如下图2所示。

  **图2** 向量计算单元架构图

  ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/npu_3510_hw_arch_vector.png)

  **表2** 向量计算单元关联的存储单元

  | 存储单元 | 说明 | 
  | --- | --- |
  | UB | AI Core内部物理存储单元，通常用于存储向量计算的输入/输出数据。 |
  | SIMD Register File | AI Core内部物理存储单元，在SIMD程序中，数据从UB搬运到Register进行计算，产生的中间结果可以不用传回UB，直接在寄存器计算。 |



## UB介绍<a name="ZH-CN_TOPIC_0000002574022839"></a>

### UB的通用约束说明

- UB地址对齐约束，请参考[通用地址对齐约束](../../../general_description_and_constraints.md#通用地址对齐约束)。
- UB地址重叠约束，请参考[通用地址重叠约束](../../../general_description_and_constraints.md#通用地址重叠约束)。

### UB的内存结构与bank冲突

为了提高数据访问的效率和吞吐量，UB采用了大小相等的内存模块（bank）结构设计。当多条读写指令同时访问UB时，由于硬件资源的限制，这些指令不能同时执行，从而引发bank冲突。在这种情况下，指令需要排队等待资源，无法在一个指令周期内完成。


- 针对Atlas A3系列产品、Atlas A2系列产品
  - UB的内存结构

    **图3** UB的内存结构图

    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/a2a3_ub_memory_structure.png)

    如图3所示，UB总大小为192KB，包含16个bank group（BG0 ~ BG15），每个bank group包含3个bank。每个bank大小为4KB，由128行组成，每行长度为32B。
    - **读写冲突**：读操作和写操作同时尝试访问同一个bank。
    - **写写冲突**：多个写操作同时尝试访问同一个bank group。
    - **读读冲突**：多个读操作同时尝试访问同一个bank group。

  - bank冲突优化

    可参考[避免bank冲突（NPU架构版本2201）](../../../../../guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_2201.md)。



- 针对Ascend 950PR&950DT系列产品
  - UB的内存结构

    **图4** UB的内存结构图

    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/950_ub_memory_structure.png)

    如图4所示，UB总大小为256KB，包含8个bank group（BG0 ~ BG7），每个bank group包含2个bank。每个bank大小为16KB，由512行组成，每行长度为32B。
    - **读写冲突**：读操作和写操作同时尝试访问同一个bank。
    - **写写冲突**：多个写操作同时尝试访问同一个bank group。
    - **读读冲突**：两个读操作同时尝试访问同一个bank，或者两个以上读操作同时尝试访问同一个bank group。
  
  - bank冲突优化

    可参考[避免bank冲突（NPU架构版本3510）](../../../../../guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md)。

