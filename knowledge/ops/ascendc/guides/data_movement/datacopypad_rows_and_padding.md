---
type: Guide
title: DataCopyPad 多行搬运的有效长度、padding 与物理行距
description: 分别推导 GM 与 UB 地址推进、显式 padding 和隐式补齐，检查参数范围、尾行及批量搬运的完整往返。
tags:
- task.implementation
- topic.data_movement.data_copy
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-datacopy.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_gm_to_ub.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_ub_to_gm.md
  role: supporting
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.md
  role: constraint
created_at: '2026-09-11T01:01:58Z'
updated_at: '2026-10-17T19:12:09Z'
platforms: []
---

# DataCopyPad 多行搬运的有效长度、padding 与物理行距

多行搬运应同时维护三个量：每行有效元素数 `n`、有效字节数 `L=n*sizeof(T)`、内存中相邻行起点的字节跨度 `P`。API 的 `blockLen`、计算的 `count`、UB 分配大小和行偏移不是同一个量。对 Normal padding 的普通 GM↔UB 路径，UB 行按该接口要求补齐；Compact 等模式必须按对应重载重新推导。

## 地址和参数单位

在来源所述扩展 GM↔UB 重载中，`blockLen` 是有效字节数，GM 侧 stride 是字节间隙，UB 侧 stride 是 32 字节 dataBlock 间隙。设 UB 一行经本次搬运自身的 padding/对齐后已占 `U` 字节，期望物理行距为 `P_ub`，则额外间隙是 `P_ub-U`，对应参数为 `(P_ub-U)/32`，前提是差值非负且能整除。不能不加区分地写 `(P_ub-L)/32`，也不能把完整行距写成额外间隙。

GM 侧在连续有效块之间的额外间隙为 `P_gm-L`。GM 与 UB 两端分别推导，不能交换参数单位。FP32 每行有效 13 元素时 `L=52B`，Normal 模式的基础 UB 行距可为 64B；若实际 UB 行距就是 64B，额外 UB stride 为零，第二行从第 16 个 FP32 元素开始。若选择 96B 行距，额外 stride 才是 `(96-64)/32=1`。

`GlobalTensor<T>[i]` 与 `LocalTensor<T>[i]` 的下标按元素解释；把字节 offset 直接填成该下标会再次乘 `sizeof(T)`。UB 切片起点还须满足对应接口的地址对齐。窄行结果可先放到合法行距的 staging 区，再批量写回；staging 的元素行距是字节行距除以元素字节数，不应对所有 dtype 都固定写成 32 个元素。

## padding 与计算范围

GM→UB 的显式左/右 padding 和隐式对齐 dummy 要分开核算。`isPad=false` 不保证所有被计算读取的补齐元素都是零；填充值、自动填充规则与参数组合依赖具体重载。需要把补齐区纳入归约时，应显式设置该运算的中性值，或在计算中屏蔽无效元素：求和可用零，最大值归约一般不能用零代替任意负数输入的无效项。

若计算只覆盖有效 `n` 个元素，未读的 dummy 值不应影响结果；但仍须检查后续广播、向量化或归约是否实际扩大了读取范围。FP32 的 13→16 元素补齐例中，逐行归约 count 可以是 13，而行起点和缓冲容量按 16 计算。不能因 CopyIn 支持非对齐就认为 CopyOut 的普通 DataCopy 也能按相同 count 正确写出。

UB→GM 的非对齐写回丢弃的是接口规定的 dummy，而不是任意自定义的逻辑元素。GM→UB 带填充参数的重载不能机械套到 UB→GM；普通 DataCopy 与 DataCopyPad 的参数单位也不能因结构体字段同名而互换。

## 参数范围与批量成本

Host 应在窄化转换之前检查 `blockCount`、`blockLen`、stride、padding 字段与完整地址计算的范围。来源参数表记录了 4095 个 block、普通参数 65535 和扩展 blockLen 2097151 等上限；这些必须对应到实际 SDK、接口和平台核对，不能由 C++ 字段是 uint16/uint32 推断全部位模式都有效。超范围时按行或列拆批，并重新推导每一批的两端起点；padding 的 uint8 字段也不能一概解释为 255 字节。

固定版本官方 GM→UB `DataCopyPad` 文档的 `DataCopyPadExtParams<T>` 表5规定：`leftPadding`、`rightPadding` 的单位均为元素个数，每一侧占用的字节数均不能超过32B。对该普通重载，需分别检查 `leftPadding*sizeof(T)<=32` 与 `rightPadding*sizeof(T)<=32`；FP32 每侧最多8元素，FP16/BF16每侧最多16元素。能转换为字段类型不能代替这项范围检查，也不能用一个很大的 `rightPadding` 一次初始化整个目标槽。仍需同时核对UB容量、实际目的地址及计算的有效范围。

这一范围来自固定官方参数合同；没有恢复本地旧会话的硬件失败探针，因此不把它表述为旧会话的实测结论。官方同页对 Ascend 950PR/950DT 的 `PaddingMode::Compact` 另有约束：`dstStride`、`leftPadding`、`rightPadding` 无效；不能把普通模式的逐侧字段检查机械套作该模式的配置方法。

将合法多行合成一次搬运可能减少下发和事件开销；扩大行距到 64B 等值则是需要测量的候选，会增加 UB 占用，可能挤压主 tile。不能为了对齐而向上移动真实 GM 数据起点，也不能把逐行搬运一律禁用：复杂 stride、接口限制或诊断阶段可能需要显式逐行地址。

来源提到 strided 搬运约 29–32 行后出现周期性零值，但没有闭合的原始轨迹和版本证据；这不构成通用硬件上限。对此应比较 1D 与 2D 的相同逻辑地址、字段截断、实际 UB 行距、同步和分配范围，而不是直接把所有 batch 钳到 32 行。

验证可先做不带计算的 CopyIn→CopyOut 往返，再加入逐行计算；输入编码行号和列号，分别检查第一行、第二行、满批末行、尾批及有效区外哨兵。批量事件只可覆盖已经证明存在相同生产消费关系的数据；API 调用数量本身不能证明硬件指令数或性能收益。

基本接口见 [GM→UB](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_gm_to_ub.md) 与 [UB→GM](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_ub_to_gm.md)；跨块写回的语义见[按目标布局搬运](../layouts/strided_placement_and_batched_writeback.md)。
