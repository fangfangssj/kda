---
type: API
title: 原子操作概述
description: 数据搬运随路原子操作接口用于对后续目的地址为GM的数据搬运开启原子操作，涉及的接口请参见表1。如下图1的左侧子图所示，未开启原子操作时，写入GM的数据搬运完成后，GM中原始数据将被新搬运数据完全覆盖。如图1右侧子图所示，当数据搬运随路原子操作接口被调用后，系统将为后续写入GM的数据搬运开启原子操作。此时，数据搬运完成后，GM中的最终数据由原始GM数据与新搬运数据共同决定。
tags:
- task.lookup
- topic.implementation.atomic
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/atomic_operations_overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 原子操作概述<a name="ZH-CN_TOPIC_0000002555621180"></a>

数据搬运随路原子操作接口用于对后续目的地址为GM的数据搬运开启原子操作，涉及的接口请参见[表1](#table1)。如下图1的左侧子图所示，未开启原子操作时，写入GM的数据搬运完成后，GM中原始数据将被新搬运数据完全覆盖。如图1右侧子图所示，当数据搬运随路原子操作接口被调用后，系统将为后续写入GM的数据搬运开启原子操作。此时，数据搬运完成后，GM中的最终数据由原始GM数据与新搬运数据共同决定。

**图1**  数据搬运随路原子累加效果  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/atomic_operation_pipelined_data_movement_effect.png "数据搬运随路原子累加效果")

**表1**  数据搬运随路原子操作接口<a name="table1"></a>

| 对应接口 | 接口功能描述 |
| --- | --- |
| [SetAtomicAdd](set_atomic_add.md) | 对后续目的地址为GM的数据搬运开启原子累加。原子累加过程：将待拷贝的内容和GM已有内容进行求和，然后将求和结果写入GM。 |
| [SetAtomicType](set_atomic_type.md) | 通过设置模板参数来设定原子操作不同的数据类型。 |
| [DisableDmaAtomic](disable_dma_atomic.md) | 关闭数据搬运随路原子操作功能。 |
| [SetAtomicMax(ISASI)](set_atomic_max_isasi.md) | 设置后续搬运到GM的数据是否执行原子比较：将待拷贝的内容和GM已有内容进行比较，然后将最大值写入GM。 |
| [SetAtomicMin(ISASI)](set_atomic_min_isasi.md) | 设置后续搬运到GM的数据是否执行原子比较：将待拷贝的内容和GM已有内容进行比较，然后将最小值写入GM。 |

Scalar原子操作接口，能够在指定GM地址上进行单点原子计算操作，涉及的接口请参见[表2](#table2)。对比数据搬运随路原子操作接口，Scalar原子操作接口不会影响后续向GM搬运数据的指令。

如下图2左侧子图所示，不使用AtomicAdd接口时，多个AI Core同时对同一GM地址执行累加操作会相互覆盖，操作不具备原子性，最终结果不可预期。如右侧子图所示，使用AtomicAdd接口后，各AI Core的累加操作串行化执行，确保每次累加操作的原子性，最终结果符合预期。

**图2**  标量原子累加效果  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/scalar_atomic_operation_diagram.png "标量原子累加效果")

**表2**  Scalar原子操作接口<a name="table2"></a>

| 对应接口 | 接口功能描述 |
| --- | --- |
| [AtomicAdd](atomic_add.md) | 用于在指定GM地址上进行原子加操作，将`address`指向的GM地址上的旧值（`old_value`）与输入标量值（`value`）求和，将和结果（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [AtomicMin](atomic_min.md) | 用于在指定GM地址上进行原子取最小值操作，将`address`指向的GM地址上的旧值（`old_value`）与输入标量值（`value`）做比较，将较小值（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [AtomicMax](atomic_max.md) | 该接口用于在指定GM地址上进行原子取最大值操作，将`address`指向的GM地址上的旧值（`old_value`）与输入的标量值（`value`）进行比较，将较大值（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [AtomicCas](atomic_cas.md) | 在指定GM地址上进行原子比较操作，读取`address`指向的GM地址上的旧值（`old_value`）与输入标量值`value1`进行比较：如果相等，则将输入标量值`value2`写入GM地址；如果不相等，则GM地址上的值保持不变。返回该地址修改前的值（`old_value`）。 |
| [AtomicExch](atomic_exch.md) | 用于在GM内存中执行原子交换操作，读取`address`指向的GM地址上的旧值（`old_value`），并将输入的标量值（`value`）替换旧值存储回同一地址，返回该地址修改前的值（`old_value`）。 |
| [SetStoreAtomicConfig(ISASI)](set_store_atomic_config_isasi.md) | 设置标量原子操作配置：操作类型（仅支持求和操作）和数据类型。 |
| [GetStoreAtomicConfig(ISASI)](get_store_atomic_config_isasi.md) | 获取SetStoreAtomicConfig设置的标量原子操作配置。 |
