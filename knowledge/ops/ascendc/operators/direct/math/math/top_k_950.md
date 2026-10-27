---
type: Operator
title: "TopK 950 Direct Invoke 算子知识"
description: "说明 cann_bench::top_k 在 950 上的接口边界、Host 安全 selector、streaming/sharded heap 闭包、保留的 TopKV2 快路、POD、workspace 与多 Kernel 启动图。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/README.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/arch35/radix_sort_top_k.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/arch35/top_k_merge_sort.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/arch35/radix_sort_top_k_single_block.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/arch35/radix_sort_top_k_single_core.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/arch35/radix_sort_top_k_inter_core_template_optimization.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/arch35/top_k_non_last_small_non_transpose.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_host/arch35/top_k_v2_tiling_arch35.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_kernel/top_k_v2_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/top_k_v2/op_host/top_k_v2_def.cpp"
    role: implementation
created_at: "2026-08-24T19:33:22Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# TopK 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::top_k` → `top_k_npu` → `calc_top_k_tiling` → `launch_top_k_kernel`，Direct Invoke，不经 ACLNN |
| 语义 | 沿 `dim` 返回有序的最大或最小 `k` 项；`values` 保持输入 dtype，`indices` 为轴内相对 `int64` 索引 |
| 支持域 | 连续 ND，rank 1–8；`int8/uint8/int32/int64/float16/float32/bfloat16`；`0<=k<=axis`；单轴不超过 `2^23`，总元素不超过 `2^26`，`k<=2048` |
| 关键设计 | 先尝试有完整资源证明的 TopKV2 单块/独立行、small-merge、非末轴融合和 small-k 快路；任何 tiling 缺口、跨核归约、不一致字段或保守边界都回到纯设备 streaming heap |
| 安全闭包 | 普通形态一行一线程在输出 GM 上维护有界 heap；少行长轴拆为至多 64 个 `(row, shard)` heap，再以第二个 Kernel 按行归并 |
| 不可达旧路径 | 生产 Host 不保留 `MULT_CORE_MODE/MULT_CORE_OPTIM_MODE`；大非末轴把 NDDMA selector 强制改成 64 位地址的 SIMT 转置；output enum 1/3/4/6/7/8 只有遗留 router/body，不是当前 Host Route |
| 内存 | 安全单段 heap 不用 workspace/显式 UB；分片 heap 只用有界候选 GM；快路使用动态 UB、16 MiB system workspace 前缀及按 512 B 对齐的组合切片 |

## 接口、shape 与排序契约

令归一化轴为 `a = dim < 0 ? dim + rank : dim`，并记：

```text
outer = product(d[0:a])
axis  = d[a]
inner = product(d[a+1:rank])
rows  = outer * inner

input(o,p,i)  = x[(o*axis + p)*inner + i]
output(o,r,i) = y[(o*k    + r)*inner + i]
```

对每个 `(o,i)` 逻辑行，输出 `k` 个最优 `(value, axis_index)`。`largest=true` 时 best-to-worst 为降序，`largest=false` 时为升序。Direct SIMT 路线统一使用全序：

1. `largest` 时 NaN 优于非 NaN，`smallest` 时 NaN 劣于非 NaN；
2. 非 NaN 按数值与方向比较；
3. 值相等时原轴索引较小者优先。

因此 safety heap、sharded heap 与 small-k 都确定性保留较小索引。相关 TopKV2 快路以 `isSort=1` 输出有序结果；若迁移到不同 CANN body，NaN 与相等值的次序仍需按上面的 Direct 契约单独复核，不能只验证 value 集合。

| 参数 | shape/layout | dtype | Host 契约 |
|---|---|---|---|
| `x` | 连续 ND，rank 1–8 | 七种已列 dtype | 非连续输入拒绝；空维允许走空输出短路 |
| `k` | Host `int64` | 标量 | `0<=k<=axis` 且 `k<=2048`；`k=0` 直接返回空结果 |
| `dim` | Host `int64` | 标量 | 支持负轴；归一化后位于 `[0,rank)` |
| `largest` | Host bool | 标量 | 决定比较器模板与 TopKV2 order |
| `values` | 原 shape 的轴 `a` 改为 `k` | 同 `x` | 总是有序输出 |
| `indices` | 同 `values` | `int64` | 值域 `[0,axis)`，是轴内索引而非扁平地址 |

### 空张量与 checked arithmetic

`top_k_meta` 先分配输出。只要 `values.numel()==0`，包括 `k=0` 或非选择维为空，`top_k_npu` 就在 tiling、workspace 和 launch 之前返回。非空路径再由 `calc_top_k_tiling` 要求所有维正。

Host 先检查每维 `<=2^23`，再用除法形式保护总乘积：

```text
if totalElements > (2^26 / extent): reject
totalElements *= extent
```

所以后续 `outer`、`inner`、`rows` 和 GM 的 `uint64` 地址乘法均受总元素上界约束；轴索引暂存为 `uint32` 也由 `axis<=2^23` 证明安全。不得把这些乘法改回无上界的 `int32`，也不得只在最后检查已溢出的乘积。

## Host selector 与 fallback 图

### 选择顺序

```text
Meta/contiguous/empty-output
  -> query AIV coreNum and UB; usableUb = ubSize - 32768
  -> build outer/axis/inner
  -> initial optimized tiling
       last axis: BF16 or FP16 with 4*k>=axis, axis<=1024
                    -> small merge, failure -> CalcLastAxisTiling
                  otherwise -> CalcLastAxisTiling
       non-last axis<=2048:
                  int32 k>64 -> SIMT axis-to-last + CalcLastAxisTiling
                  otherwise -> fused non-last; failure -> SIMT axis-to-last
       non-last axis>2048:
                  CalcLastAxisTiling, then NDDMA intent is rewritten to SIMT
  -> safety override
       tiler failed / int64 inter-core / wide-k boundary
       / invalid fast geometry / any inter-core mode
       / k==1 / nontrivial full selection / unclosed large-k
          -> streaming heap; few-row long-axis subset -> sharded heap
  -> surviving fast plan
       choose small-k or short-strided post-sort when applicable
       plan transpose/int64-dynamic slices
  -> allocate workspace and launch one same-stream graph
```

“optimized tiler 成功”不是充分条件。Fast plan 还必须同时满足：

- `0<blockDim<=coreNum`，`platformCoreNum==coreNum`；
- `lastAxisNum==axis`、`topKRealValue==outputLastDimValue==k`；
- logical rows 与 `unsortedDimNum` 一致；mode 8 还要求 `oneCoreRowNum==outer`、`unsortedDimNum==inner`；
- `numTileDataSize` 非零且不越过本 Route 的轴/对齐上界；
- API tmp 不超过 `usableUb`；
- TopKV2 workspace 在 `[16 MiB, 16 MiB + declared-domain cap]` 内。

任一不变量失败都重新构造 safety POD，而不是继续消费部分写入的 tiling。

### Route 重要性与退化关系

| 重要性 | Route | 可达性 | selector/key | 适用形态 | 价值、边界与 fallback |
|---:|---|---|---|---|---|
| P0 | R-001 通用 streaming heap | active safety | output 9 | 声明域内任意 dtype、末轴或 strided 非末轴；优化 tiling 失败或被 safety guard 淘汰 | 纯 NPU 语义闭包；无 workspace、无跨核同步、无转置。吞吐受一线程一行和 `O(axis log k)` 限制；少行长轴且可拆 shard 时升级 R-002 |
| P1 | R-002 sharded streaming/merge | active promotion | output 10 | R-001 且 `shards>=2`；少量逻辑行、长轴 | 用最多 64 个独立 task 恢复核并行；两个同流 Kernel，无全局 barrier。shard 不足时退回 R-001 |
| P2 | R-003 small-k SIMT | active override | output 5 | `axis<=2048`，且满足：末轴 int64 `k<=32`；或非末轴 `axis<=32,k<=128,dtype∈{f16,bf16,f32,i32}`；或非末轴 int32 `k<=32`。`k=1` 已被 safety guard 提前送 R-001 | 私有有序候选表，直接读写原 layout，避免 UB API 与全量转置；超过模板 `MAX_K` 或未命中即保留初始快路/安全路 |
| P3 | R-004 whole-axis single-block TopKV2 | active fast | common key，mode 0 | 一批完整行连同 tmp 可驻 `usableUb`，且 safety geometry 全部成立 | 单 Kernel、整轴选择与有序输出；UB 不闭合继续 CalcLastAxis 的 mode 1/2/4，后两者最终由 R-001 接管 |
| P4 | R-005 row-parallel single-core TopKV2 | active fast | common key，mode 1 | 长轴多 tile，通常 `rows>=coreNum`；少行时只作为最后的 exact tiler 候选；大 k 快路还要求 axis≤8192 和独立行几何闭合 | 每核独立处理行，无跨核归约；末轴 int64 叠加 R-010，转置中的 int64 mode 1 使用 v9.0.1 body。资源或 safety guard 不满足转 R-001/R-002 |
| P5 | R-006 非末轴 UB 融合 | active fast | common key，mode 8 | 非末轴 `axis<=2048`，至少一个 dtype 相关 `innerChunk` 满足完整 peak UB；int32 `k>64` 不走此路 | `[axis,chunk]` 搬入后在 UB 中换成 TopK-major，避免三块全量 transpose GM；int32 `33<=k<=64` 追加 R-009；失败转 R-008，再不安全转 R-001 |
| P6 | R-007 FP16/BF16 small-merge | active fast | key 13002/14002 | 末轴、axis≤1024；BF16，或 FP16 且 `4*k>=axis`；Concat/tmp/行批资源闭合 | 全轴 merge-sort 后截取 k，BF16 中间为 FP32；tiling 失败先退 common last-axis，随后仍受 safety guard |
| P7 | R-008 64 位 SIMT transpose composition | active composition | `needsTranspose=2` | 非末轴 int32 `k>64`、融合失败，或大非末轴的保留快路 | forward `[o,a,i]→[o,i,a]`、last-axis TopK、values/indices reverse；任一 TopKV2 安全条件失败时直接改走 R-001，不依赖 NDDMA |
| P8 | R-009 非末轴 int32 短步有序化 | active post-node | mode 8 + output 2 | R-006 产出且 `33<=k<=64` | 第二个 SIMT Kernel 原地对 `[outer,k,inner]` 的每行插入排序；`k<=32` 由 R-003 覆盖，`k>64` 由 R-008 组合处理 |
| P9 | R-010 int64 动态 int32 窄化 | active graph，Device conditional | mode 1 + dynamic flag | 非转置、末轴、int64 single-core；但若 output 5/9/10 已短路则不执行 | 全域证明可精确窄化时走 int32 TopK，否则同流执行 int64 body；任一越界不是 Host 失败，而是 Device fallback |

### safety override 的关键谓词

| Guard | 命中结果 | 原因 |
|---|---|---|
| `modeType` 为 2 或 4 | R-001/R-002 | 不让含 `SyncAll` 与紧凑位置状态的跨核 TopKV2 进入生产 launch |
| `k==1` | R-001/R-002 | 统一走可解释闭包，不进入边界敏感优化 |
| `k==axis && axis>32` | R-001/R-002 | 非平凡全选交给稳定全序 heap |
| `k>=1024` 且非 bounded fast | R-001/R-002 | 仅 axis≤8192 的 mode 0/独立行 mode 1 可保留 |
| int64 tiling 失败或得到 mode 2/4 | R-001/R-002 | 避免宽整数跨核值/索引对应风险 |
| `k==2048` 的长轴 16/32-bit 边界，或特定非末轴 32-bit single-core 组合 | R-001/R-002 | 避免紧凑位置字段越界及值/索引错配 |
| fast POD 任一字段不一致 | R-001/R-002 | 不让 workspace、tile 或 row 几何从错误状态派生 |

## Host、POD 与 Launch 契约

### Host/Launcher 生命周期

`top_k_npu` 用 device guard 固定输入设备；Meta 分配两个输出；非空时构造 `TopKLaunchParams`，随后以 `torch::empty(uint8)` 分配精确 workspace。输入、输出和 workspace 都由 lambda 捕获，并在当前 NPU stream 上调用 `launch_top_k_kernel`。多节点图没有跨 stream event，producer-consumer 可见性只依赖同流提交顺序。

Launcher 返回类型为 `void`；Host 能传播的是参数、tiling 与分配阶段错误，源码没有逐 Kernel 返回码。不要把“成功提交”解释成设备执行状态已同步确认。

### 活跃字段与 producer/consumer

`TopKLaunchParams` 是 Host-only 总结构；真正传入官方 Device entry 的是 64 B `TopKDirectTiling` 或 56 B `TopKMergeDirectTiling`。Safety heap/sharded heap 不 pack 官方 POD，而是从 Host 总结构抽取标量作为 kernel args。

| 字段 | Host producer | Device/launcher consumer | 约束 |
|---|---|---|---|
| `tiling.isLargest` | `largest` | 全部 order 模板 | 0/1 |
| `tiling.modeType` | route tiler | compact pack/router | 0/1/8；2/4 不可生产 launch |
| `tiling.lastAxisNum` | `axis` | TopKV2/safety scalar | 必须等于 `transposeRows` |
| `tiling.unsortedDimNum` | rows 或 mode 8 的 inner | fast body/geometry audit | 与 Route 行定义严格一致 |
| `tiling.unsortedDimParallel` | core/shard planner | fast body或 sharded phase | R-002 中表示 `shards` |
| `blockDim` | tiler或 safety planner | 主 TopK / sharded phase 1 | safety 单段为 `min(coreNum,rows)`；sharded 为 `rows*shards` |
| `transposeBlockDim` | AIV core count | small-k/transpose/cast | 大于 0 |
| `localMemorySize` | `ubSize-32768` | TopKV2 dynamic UB launch | safety heap不消费 |
| `tilingKey` | dtype/merge selector | launcher 首级 router | common `1001/2001/1003/1004/3002/3003/4002` 或 merge `13002/14002` |
| `needsTranspose` | non-last selector | launch graph | 当前生产值 0/2；值 1 会在返回前改为 2 |
| `needsOutputSort` | safety/fast selector | launcher 首级短路 | 活跃值 0/2/5/9/10 |
| `transposeOuter/Rows/Cols` | shape product | 所有 ND 地址与组合图 | 分别为 outer/axis/inner，使用 u64 |
| `sortValueOffset/sortIndexOffset` | sharded planner | 两阶段候选 heap | R-002 的 value 从 0 开始，index 起点 512 B 对齐 |
| `int32*Offset/Count` | int64 overlay | cast + gated TopK | 各切片 512 B 起始对齐 |
| `workspaceSize/topkWorkspaceSize` | workspace planner | Torch allocation/TopKV2 | R-001 两者均为 0；R-002 仅 `workspaceSize>0` |

### `TopKDirectTiling`：64 B 两套互斥 slot

`TopKDirectTiling` 是 `uint32_t data[16]`，slot `i` 的 byte offset 为 `4*i`，并由 `static_assert(sizeof(TopKDirectTiling)==64)` 固化。`PackDirectTiling` 先以 `{}` 清零，再从 Host `TopKV2TilingDataSimd` 写 slot；`top_k_direct_kernel` 按 `data[0]` 重建 active 字段，强制补 `isSort=1`、`isInInt32Range=1`，最终调用 `generateOpObject<T,UnsignedT,NumPass,int64_t>`。

mode 8 只用于 R-006，布局如下：

| slot/offset | Host source（producer） | Device reconstructed field（consumer） | 单位与边界 |
|---|---|---|---|
| `data[0]` / 0 | `CalcNonLastTiling: modeType=8` | `modeType`，选择 non-last body | enum，必须为 8 |
| `data[1]` / 4 | `largest` | `isLargest` | bool 0/1 |
| `data[2]` / 8 | `keyParams0` | `keyParams0` | 0=radix-select，1=merge-sort |
| `data[3]` / 12 | `lastAxisNum=axis` | `lastAxisNum` | element；`axis<=2048` |
| `data[4]` / 16 | `topKRealValue=k` | `topKRealValue` 且复制到 `outputLastDimValue` | element；`k<=2048` |
| `data[5]` / 20 | `oneCoreRowNum=outer` | `oneCoreRowNum` | outer group；`<=2^26` |
| `data[6]` / 24 | `unsortedDimNum=inner` | `unsortedDimNum` | logical inner rows；`<=2^26` |
| `data[7]` / 28 | `numTileDataSize=AlignUp(axis,32)` | `numTileDataSize` | element；mode 8 geometry audit 限界 |
| `data[8]` / 32 | `topkAcApiTmpBufferSize` | 同名字段 | byte；radix tmp，`<=usableUb` |
| `data[9]` / 36 | `tmpUbSize` | 同名字段 | byte；merge Sort tmp，radix 时为 0 |
| `data[10]` / 40 | `keyParams1=bestChunk` | `keyParams1` | inner element；候选最大 32 |
| `data[11]` / 44 | `keyParams2=bestLoops` | `keyParams2` | `ceil(inner/bestChunk)` |
| `data[12]` / 48 | `keyParams3=bestInputRow` | `keyParams3` | byte；aligned input row |
| `data[13]` / 52 | `keyParams4=bestAxisRow` | `keyParams4` | byte；aligned TopK-major row |
| `data[14]` / 56 | `keyParams5=bestValueRow` | `keyParams5` | byte；aligned value row |
| `data[15]` / 60 | zero-init，不写 | 不读取 | 必须保持 0，不能借给末轴字段 |

R-004/R-005、R-008 中间 TopK 与 R-010 使用末轴布局；生产 `modeType` 只有 0/1：

| slot/offset | Host source（producer） | Device reconstructed field（consumer） | 单位与边界 |
|---|---|---|---|
| `data[0]` / 0 | `FillSmallMode` 保持 0；`FillSingleCoreMode` 写 1 | `modeType` | 0=single-block，1=single-core |
| `data[1]` / 4 | `largest` | `isLargest` | bool 0/1 |
| `data[2]` / 8 | `sortLoopTimes`，u64→u32 | `sortLoopTimes` | loop；由 `rows<=2^26` 保证窄化 |
| `data[3]` / 12 | `unsortedDimParallel` | 同名字段 | active core/parallel rows，`<=coreNum` |
| `data[4]` / 16 | `unsortedDimNum=rows`，u64→u32 | 同名字段 | row；`<=2^26` |
| `data[5]` / 20 | `lastDimNeedCore` | 同名字段 | mode 0 为 1；mode 1 为 0 |
| `data[6]` / 24 | `numTileDataSize` | 同名字段 | element；完整 axis 或 tile size |
| `data[7]` / 28 | `platformCoreNum` | 同名字段，并参与派生 `lastDimTileNumTimes` | core；非零 |
| `data[8]` / 32 | `topkAcApiTmpBufferSize` | 同名字段 | byte；`<=usableUb` |
| `data[9]` / 36 | `batchNumInUb` | 同名字段 | mode 0 的 UB row batch；mode 1 为 0 |
| `data[10]` / 40 | `tailLoopBatchNum` | 同名字段 | mode 0 末轮每核行数 |
| `data[11]` / 44 | `tailBatchNum` | 同名字段 | mode 0 的额外 tail cores；mode 1 的 tail rows |
| `data[12]` / 48 | `tailTileNum` | 同名字段 | mode 1 轴尾元素；mode 0 为 0 |
| `data[13]` / 52 | `lastAxisNum=axis`，i64→u32 | 同名字段 | element；`axis<=2^23` |
| `data[14]` / 56 | `topKRealValue=k`，i64→u32 | `topKRealValue` 且复制到 `outputLastDimValue` | element；`k<=2048` |
| `data[15]` / 60 | `lastDimTileNum`，i64→u32 | 同名字段 | tile count；`<=axis` |

Device 还从 slot 派生：

```text
lastDimTileNumTimes = ceil(data[15] / data[7])
```

因此 `data[7]` 既是平台核数又是除数，Host 的 `coreNum>0` 检查属于 ABI 正确性条件。所有显式 u64/i64→u32 窄化都由 `axis<=2^23`、`numel/rows<=2^26`、`k<=2048` 先证明；新增更大声明域时必须先扩 POD，不能继续静默 cast。

### `TopKMergeDirectTiling`：56 B 顺序 ABI

R-007 不复用 `data[16]`。`PackMergeDirectTiling` 从 `CalcSmallMergeTiling` 产生的完整 tiling 复制九个 active 字段，value-init 使 `reserved=0`；`top_k_merge_direct_kernel<T,ConvertT>` 再逐字段重建 `TopKV2TilingDataSimd`，调用 `generateMergeTopKObject<T,ConvertT,int64_t>`。结构按 8 B 对齐，并由 `static_assert(sizeof(TopKMergeDirectTiling)==56)` 固化：

| offset | 字段/类型 | Host producer | Device consumer | 单位、边界与重建 |
|---:|---|---|---|---|
| 0 | `sortLoopTimes:u64` | `ceil(virtualCores/coreNum)` | merge `ProcessSort` 外层 loop | loop，非零 |
| 8 | `unsortedDimNum:u64` | logical `rows` | 末轮有效 row/GM 行界 | row，`<=2^26` |
| 16 | `oneCoreRowNum:u64` | `rowsPerCore` | 每次 `ProcessSingleBlockSort` 行批 | row/core，非零 |
| 24 | `outputLastDimValue:i64` | `k` | copy-out 截取长度 | element，`1..axis` |
| 32 | `unsortedDimParallel:u32` | `coreNeed` | block/core 有效域 | core，`1..coreNum` |
| 36 | `numTileDataSize:u32` | `axis` | 输入行 tile 与对齐 copy | element，`1..1024` |
| 40 | `mergSortAcApiNeedBufferSize:u32` | `GetConcatTmpSize` | `MergeSortInitBuffer`/VBS tmp | byte，`<usableUb` |
| 44 | `keyParams4:u32` | `bufferNum` | queue/buffer 初始化 | 当前 `axis<=1024` 的 R-007 为 2；结构容许 1/2 |
| 48 | `isLargest:i32` | `largest` | `MergeSort<...,true/false,...>` router | bool 0/1 |
| 52 | `reserved:u32` | value-init | 不重建、不读取 | 必须保持 0 |

这两个 compact POD 只服务 active/conditional fast route。R-001/R-002/R-003/R-009 直接传 scalar kernel args，不消费它们；不得为了复用字段而让 safety Route 伪造 official tiling。

### 实际启动图

```text
R-001: streaming_heap_top_k_kernel

R-002: sharded_heap_phase1_kernel
       -> sharded_heap_phase2_kernel

R-003: small_k_top_k_kernel

R-004/R-005: top_k_direct_kernel
R-007:       top_k_merge_direct_kernel

R-006: top_k_direct_kernel(mode 8)
       -> [R-009] sort_short_strided_top_k_kernel

R-008: transpose_axis_topk_kernel<AxisToLast>
       -> TopKV2 fast node
       -> transpose_axis_topk_kernel<LastToAxis>(values)
       -> transpose_axis_topk_kernel<LastToAxis>(indices)

R-010: range_init -> int64_cast_input(atomic_or)
       -> int32 TopK or gated int32-compatible body
       -> gated int64 fallback
       -> int64_cast_output when flag==0
```

### Route → entry → body/template →关键 API

| Route | launcher 与 Device entry | body/template 实例 | 关键 API/阶段 |
|---|---|---|---|
| R-001 | `LaunchStreamingHeapTopKTyped<T,order>` → `streaming_heap_top_k_kernel<T,order>` | `StreamingHeapTopKRowsSimt<T,order>` | `asc_vf_call<...,2048 lanes>`；输出 GM 上 build/replace/sift/heap-sort |
| R-002 | `LaunchShardedStreamingHeapTopKTyped<T,order>` → `sharded_heap_phase1_kernel` → `sharded_heap_phase2_kernel` | `ShardedHeapPhase1Simt<T,order>`、`ShardedHeapPhase2Simt<T,order>` | 两次 `asc_vf_call<...,1 lane>`；local heap→candidate GM→row merge heap |
| R-003 | `LaunchSmallKTyped<T,order,MAX_K,THREADS>` → `small_k_top_k_kernel<T,order,MAX_K,THREADS>` | `SmallKRowsSimt<T,order,MAX_K,THREADS>` | `asc_vf_call`；thread-private insertion list；Host 保证 `k<=MAX_K` |
| R-004 | `LaunchTypedTopK<T,U,passes>` → `top_k_direct_kernel<T,U,passes>` | `generateOpObject` → `RadixSortTopKSingleBlock<T,U,order,true,int32_t,int64_t>` | `Init/Process`；`DataCopyPad`；`AscendC::TopK<...,RADIX_SELECT>` |
| R-005 | 普通 dtype 同 R-004 entry；int64 mode 1 由 `launch_top_k_v901_int64_single_core` → `top_k_v901_int64_single_core_kernel<order>` | `RadixSortTopKSingleCore<T,U,passes,order,true,int32_t,int64_t>`；int64 为 `top_k_v901::` 同模板 | per-tile radix histogram/cumsum；`AscendC::TopK` 归约 candidates；需要时 `AscendC::Sort`；event + `DataCopyPad` |
| R-006 | `LaunchTypedTopK` → `top_k_direct_kernel`，compact POD 的 mode 8 先于其它 body 分支 | `TopKNonLastSmallAxisNonTranspose<T,int64_t,order,true,UseMergeSort>` | `LoadTile/TransposeToTopKMajor`；`UseMergeSort=true` 调 `AscendC::Sort`，否则 `AscendC::TopK`；V→S 后 `StoreTile` |
| R-007 | `LaunchMergeTopK<half,half>` 或 `<bfloat16_t,float>` → `top_k_merge_direct_kernel<T,ConvertT>` | `generateMergeTopKObject` → `topkV2::MergeSort<T,ConvertT,TopKV2TilingDataSimd,order,int64_t>` | `Init/ProcessSort`；`KernelVbsMergeSort::VbsMergeSort/Bf16`；Queue/DataCopyPad |
| R-008 | `LaunchTransposeBySize<true>` → `transpose_axis_topk_kernel<T,true>` → `LaunchOfficialTopK` → 两个 `<false>` reverse entry | `AxisToLastSimt<T>` / R-004 或 R-005 body / `LastToAxisSimt<T>` | 三次 `asc_vf_call<...,2048 lanes>` 包围 compact-POD TopKV2；同流 GM producer→consumer |
| R-009 | `LaunchShortStridedSortTyped<int32_t,order>` → `sort_short_strided_top_k_kernel<int32_t,order>` | `SortShortStridedRowsSimt<int32_t,order>` | `asc_vf_call<...,2048 lanes>`；原地 scalar insertion sort，不用 Sort API/UB |
| R-010 | `LaunchInt64DynamicInt32Typed<order>` → 三个 cast entry + one/two gated TopK entry | `CastInputVf/CastOutputVf`；`top_k_direct_kernel<int32_t,uint32_t,4>` 或 `top_k_v901::RadixSortTopKSingleCore<int32_t\|int64_t,...>` | `asc_atomic_or` 范围证明；`CalcLastAxisTiling` retile；phase flag gate；int32 cast-back |

R-004/R-005/R-006/R-008 的普通 `LaunchTypedTopK` dtype 模板映射是固定 ABI 的一部分：

```text
int8     -> <int8_t,    uint8_t,  1>
uint8    -> <uint8_t,   uint8_t,  1>
int32    -> <int32_t,   uint32_t, 4>
int64    -> <int64_t,   uint64_t, 8>   // mode 0；mode 1 走 v9.0.1 closure
float16  -> <half,      uint16_t, 2>
float32  -> <float,     uint32_t, 4>
bfloat16 -> <bfloat16_t,uint16_t, 2>
```

`launch_top_k_kernel` 先检查 merge key，再按 output selector 短路，最后才进入 TopKV2/transpose 图。因此 output 5/9/10 的优先级高于 `useInt64DynamicInt32`；即使 Host 已预留 dynamic slice，只要先命中这些 output，窄化图就不会执行。

## Kernel 阶段、归约与排序

### R-001：单段 streaming heap

每个 AIV block 取得连续 row 区间，每个 SIMT lane 以 `local=threadIdx.x; local<count; local+=blockDim.x` 独占一行：

```text
seed first k pairs into strided output GM
build worst-at-root heap
for pos in [k, axis):
    if input_pair(pos) is better than heap_root:
        replace root and sift down
heap-sort output in place -> best-to-worst
```

选择阶段只扫描一次剩余 axis，复杂度 `O(axis log k)`；最后 heap-sort 为 `O(k log k)`。输出本身兼作 heap，因此无额外 GM/UB。线程只写自己的 `(outer,inner)` 行，既无 block 间共享状态，也无需 atomic/barrier。

### R-002：分片选择与行归并

Host 计算：

```text
taskBudget   = min(coreNum, 64)
maxShards    = floor(taskBudget / rows)
desired      = ceil(axis / 4096)
shards       = min(maxShards, desired)
taskCount    = rows * shards
candidateCnt = taskCount * k
```

仅 `shards>=2` 才启用。phase 1 的每个 block/单 lane 独占一个 `(row,shard)`，边界为：

```text
axisBegin = floor(axis*shard/shards)
axisEnd   = floor(axis*(shard+1)/shards)
localK    = min(axisEnd-axisBegin, k)
```

它在专属连续 GM slice 上完成 local heap；phase 2 每个 block/单 lane 独占一行，读取各 shard 的 `localK` 个有效候选，维护最终 k-entry heap 并原地排序。两个阶段通过同一 stream 的 Kernel 边建立可见性，不使用 `SyncAll`。候选上界为 `64*2048` 项；最宽 dtype 的 value 与 int64 index 合计约 2 MiB，再做 512 B 对齐。

### R-003：small-k 私有有序表

每线程声明编译期 `bestValues[MAX_K]` 与 `bestIndices[MAX_K]`，顺序扫描 axis，以插入移动维护 best-to-worst。模板按 dtype/k 选择 `MAX_K` 和线程数；例如 int64 为 10/32，FP16/INT32 可到 128。扫描按轴索引递增，严格优于才前移，因此相等值自然保留较小索引在前。

```text
for pos in [0,axis):
    insert = pos                    if pos<k
           = k-1                   if pair(pos) better than best[k-1]
           = skip                  otherwise
    while insert>0 and pair(pos) strictly better than best[insert-1]: shift right
    best[insert] = (value,pos)
write best[0:k] to strided [outer,k,inner]
```

该 Route 不分配显式 UB queue/workspace；`MAX_K` 是编译期容量契约，Host selector 必须保证 `k<=MAX_K`。

| dtype 与 k 子域 | `SmallKRowsSimt<T,order,MAX_K,THREADS>` |
|---|---|
| FP16 `k<=20 / <=32 / <=64 / <=128` | `<half,...,20,2048>` / `<half,...,32,256>` / `<half,...,64,128>` / `<half,...,128,128>` |
| BF16 `k<=8 / <=32 / <=64 / <=128` | `<bfloat16_t,...,8,1024>` / `<bfloat16_t,...,32,256>` / `<bfloat16_t,...,64,128>` / `<bfloat16_t,...,128,128>` |
| FP32 `k<=64 / <=128` | `<float,...,64,128>` / `<float,...,128,128>` |
| INT32 `k<=20 / <=32 / <=64 / <=128` | `<int32_t,...,20,2048>` / `<int32_t,...,32,256>` / `<int32_t,...,64,128>` / `<int32_t,...,128,128>` |
| INT64 `k<=10 / <=32` | `<int64_t,...,10,1024>` / `<int64_t,...,32,256>` |

### R-004/R-005/R-007：保留的 TopKV2 fast body

| Route | 选择/排序阶段 | UB 与 workspace | 尾块 |
|---|---|---|---|
| R-004 mode 0 | 整轴进入 `RadixSortTopKSingleBlock`，`isSort=1` | `TopKTmp + SingleBlockRuntimeSpace <= usableUb`；通常含 16 MiB system 前缀 | `tailLoopBatchNum/tailBatchNum` 控制末轮行批 |
| R-005 mode 1 | 每核独立流式处理一行的多个 tile，再归约 tile candidates | `TopKTmp + SingleCoreRuntimeSpace <= usableUb`；user workspace 为 `tileCount*256*parallel*4` | `tailTileNum` 描述轴末 tile，row/core 由 loop 与 tail 字段控制 |
| R-007 merge | 对 aligned axis 全排序并截 k；BF16 先升 FP32 | queue/bufferNum + Concat tmp；workspace 含 histogram/tile metadata | `sortLoopTimes/oneCoreRowNum/unsortedDimNum` 限定末轮有效行 |

```text
R-004 for loop/batch: GM row batch -> UB -> AscendC::TopK(k,sorted) -> GM values/int64 indices
R-005 for each owned row:
          for axis tile: radix preprocess/histogram/cumsum -> tile candidates
          AscendC::TopK(all tile candidates,k) -> [optional AscendC::Sort] -> GM
R-007 for each valid core-loop row batch:
          DataCopyPad(aligned axis) -> [BF16 cast FP32]
          VbsMergeSort -> take first k -> DataCopyPad(values,indices)
```

`CalcLastAxisTiling` 虽会计算 mode 4/2，但最终 Host safety override 不允许它们启动；因此知识迁移时不能把“tiler 中存在”误写成“生产可达”。

### R-006/R-009：非末轴融合与有序化

融合 Route 为每个 `[outer, innerChunk]` tile 分配：原 `[axis,chunk]` 输入、TopK-major `[chunk,alignedAxis]`、value/index、可选 BF16 cast 与 API tmp。Host 在 dtype 相关候选 `{1,2,4,...,32}` 中，以 active core 优先、chunk 次优选择满足：

```text
peak = axis*inputRowBytes
     + chunk*(axisRowBytes + valueRowBytes + indexRowBytes + castRowBytes)
     + tmpBytes
peak <= usableUb
```

Device 先 GM→UB，UB 内换序后按行执行 merge-sort 或 radix-select，再以 V→S event 交给 SIMT store，按 `[outer,k,inner]` 散写。末 `innerChunk` 使用真实 `curInnerChunk`，不得把对齐 padding 写入 GM。

int32 的 `33<=k<=64` 结果再由 R-009 一线程一逻辑行原地插入排序；节点间依赖 stream order。`k<=32` 由 R-003 直接产生有序结果；`k>64` 在 Host 初选阶段就改走 R-008，不会落入遗留的 int64-index radix post-sort。

```text
R-009 row=(outerId,innerId), base=outerId*k*inner+innerId
for pos in [1,k):
    key=(values[base+pos*inner], indices[base+pos*inner])
    向前移动严格劣于 key 的 strided pair，再写回 key
```

### R-008：SIMT layout composition

forward 和 reverse 都按 `total=outer*axis*inner` 做连续 block 分区，再由 2048 个线程 `local<count` 保护尾部。地址关系为：

```text
forward: [outer,axis,inner] -> [outer,inner,axis]
reverse: [outer,inner,k]    -> [outer,k,inner]
```

所有除模、乘法和 GM offset 使用 u64。三块 workspace 分别保存 transposed input、values、indices，起点 512 B 对齐且生命周期覆盖整个图。生产 selector 强制 `needsTranspose=2`，所以 Host 中三套 NDDMA POD 与 Device NDDMA router 只是保留源码，不是当前可达路径。

### R-010：int64 范围证明

cast-input 遍历完整输入，写 int32 mirror，并在任一往返不等时 `atomic_or(rangeFlag,1)`。随后：

- `flag==0`：尝试按真正 int32 几何重新 tiling；失败时使用兼容的 int32 single-core body；最后无损扩回 int64 values；
- `flag!=0`：int32 body 与 cast-output 都 gate 掉，完整 int64 single-core body 写最终结果。

flag、int32 input/value 和 radix user workspace 的可见性依赖同流顺序。范围判断必须覆盖全输入；抽样、非原子 flag 或在 cast 完成前读取 flag 都会破坏 fallback 互斥。

## 内存、对齐与尾块总表

| Route | GM/workspace | 显式 UB | 同步/生命周期 | tail/mask |
|---|---|---|---|---|
| R-001 | 输出 GM 兼作 heap；workspace=0 | 无 | 单线程行私有 | block 的 `count` 与 lane 的 `local<count` |
| R-002 | `values[candidateCnt]` + 512 B aligned `indices[candidateCnt]` | 无 | phase 1→2 同流；每 task/row 独占 | floor 分片边界、`localK/valid` |
| R-003 | 原输入/输出 GM | 私有定长数组，不是 TQue | 单线程行私有 | `k<=MAX_K`、`local<count` |
| R-004 | TopKV2 system/user workspace | whole-axis queues/tmp | TPipe queue | batch 与 core tail 字段 |
| R-005 | histogram user workspace | tile queues/hist/tmp | 核内流水，无跨核行归约 | `tailTileNum` |
| R-006 | 通常仅 TopKV2 system 前缀 | input/transpose/value/index/cast/tmp | V→S event | `curInnerChunk`、对齐行只写有效项 |
| R-007 | histogram/tile metadata | merge queues/Concat tmp | Queue EnQue/DeQue/Free | 末轮有效 row |
| R-008 | 三个 512 B aligned 全量 slice | transpose 节点无显式 UB | 四节点同流 | 64 位 total 与 `local<count` |
| R-009 | 原地 output GM | 无 | R-006→sort 同流 | 一线程一 row，`k<=64` |
| R-010 | flag 512 B + aligned int32 input/value + TopKV2 user workspace | cast 无；TopK 用 dynamic UB | atomic flag + 同流 gate | cast block/thread count 保护 |

TopKV2 launch 前把 workspace 指针跳过 16 MiB system 前缀；安全 heap 不做该偏移。不要给 R-001 强行附加 system workspace，也不要让 R-002 的 offset 以 16 MiB 为起点，否则会丢失安全闭包的有界内存性质。

## 不变量、fallback 与实现禁区

### Route 间必须保持的不变量

- 所有 Route 使用同一 `(outer,axis,inner)` 行定义，输出始终是 `[outer,k,inner]`；组合 Route 只能改变中间 layout。
- `indices` 始终为原轴相对位置并以 int64 输出；shard index 必须加 `axisBegin`，不能返回 shard-local index。
- Safety selector 必须在 workspace slice 派生之前运行；fallback 要清零旧 tiling、transpose 与 dynamic-int32 状态。
- `largest`、NaN 位置、相等值的较小索引优先和 best-to-worst 顺序必须在 selection、merge、heap-sort 三个阶段一致。
- 只有可证明独立行的 mode 0/1/8 与 merge route 可进入 TopKV2 body；mode 2/4 不得因局部 UB 账成立而绕过 safety override。
- 多 Kernel graph 的所有中间 Tensor/Workspace 必须活到同流最后一个 consumer 提交完成。

### Fallback 边界

- 声明域内非空输入不因优化 tiler、Sort tmp 或 transpose tiler 缺失而 Host 失败；应进入 R-001/R-002。
- R-002 只是 R-001 的并行优化。`rows` 太多、axis 不够长或 task budget 不足时退回单段 heap，语义不变。
- R-006 资源不闭合先尝试 R-008；若其 TopKV2 fast plan 不满足 safety geometry，则绕过 transpose，直接以原 strided layout 进入 R-001/R-002。
- R-010 的范围越界是设备内正常 fallback，不是异常；它必须执行原 int64 路线。
- 无 CPU、native、ACLNN 或 Host 排序 fallback。

### Dormant source 分离

| 源码路径 | 为什么生产不可达 | 若要重新启用必须补齐 |
|---|---|---|
| common mode 2/4 的 `RadixSortTopK` / `RadixSortTopKMultiCoreOptimization` | `CalcLastAxisTiling` 可暂时产生，但 fast geometry 后的 `interCoreRoute` 无条件重建 R-001/R-002 plan | 删除 safety override 前，证明跨核同步、位置位宽、最终 tmp/UB 与值-索引对应在完整声明域闭合 |
| `needsTranspose=1` 的 NDDMA/Gather/VCONV graph | 所有 surviving plan 在返回前执行 `needsTranspose=1 -> 2`；safety plan 则清为 0 | 恢复 selector，并闭合 forward/value/index 三套 tiling、block/stride 编码和 64 位边界 |
| output 1 的 radix post-Sort | 初始判断最终分别被 mode 0/transpose 的 none、int32 short-sort 或 small-k 覆盖，没有最终 Host plan 返回 1 | 增加可达 producer，并重新证明 int64-index Sort tmp、UB 五段与 sort transpose slices |
| output 3/4/6/7/8 | 仅保留常量、router/body；当前 `configureStreamingHeap` 总是产生 9，再按 shard 提升到 10 | 增加明确 Host selector、资源计划与完整合法域 safety 证明 |

dormant 代码不参与 active POD/Route 验收；其字段不能占用 mode 8 的 `data[15]` 或 merge `reserved`。

### 明确限制

- 不支持 `int16/uint16/uint32/uint64`、`sorted=false`、Tensor `k` 或 int32 indices；不能用相关官方 TopKV2 的更宽接口替代 Direct 契约。
- `axis>2^23`、`numel>2^26`、`k>2048`、`k>axis`、rank 超界或非连续输入在 Host 拒绝。
- output selector 1/3/4/6/7/8 与 NDDMA value 1 有 launcher/body，但当前 Host 不生产；它们只能视为 dormant source，不能列为可达优化 Route。
- 当前源码对 launch API 的设备错误没有逐节点返回传播；需要运行时错误定位时必须在外层 stream/runtime 诊断，不能伪造本算子内部状态码。

## 最小落地骨架

```cpp
auto [outer, axis, inner] = checked_geometry(shape, dim);
if (output_numel == 0) return outputs;

Plan p = try_fast_tiling(dtype, outer, axis, inner, k, largest, usable_ub);
if (!fast_geometry_closed(p) || unsafe_inter_core(p) || conservative_boundary(p)) {
    p = make_streaming_heap_plan(outer, axis, inner, k, largest);
    maybe_promote_to_sharded_heap(p, core_num, /*max_tasks=*/64, /*grain=*/4096);
} else {
    force_64bit_simt_transpose_for_non_last(p);
    select_small_k_or_short_sort(p);
    plan_aligned_workspace_and_int64_overlay(p);
}

launch_same_stream_graph(x, values, indices, workspace, p, stream);
```

该骨架表达的是控制不变量，不是可直接复制的完整实现：TopKV2 compact POD、UB 查询、dtype 模板实例、heap 比较器和各 tail 字段仍必须与实际 producer/consumer 一一对应。

## 源码闭包索引

| 链路 | 关键符号 | 支撑事实 |
|---|---|---|
| Schema/Meta/NPU | `top_k_meta`、`top_k_npu` | dtype/shape/空输出、workspace 生命周期、同流 direct call |
| Host selector | `calc_top_k_tiling` | checked product、优化 tiler、安全审计、streaming/sharded 配置、workspace offsets |
| Last-axis tiler | `CalcSmallMergeTiling`、`CalcLastAxisTiling` | mode 0/1/2/4、UB 公式与 tail；最终可达性由上层 safety selector裁决 |
| Non-last tiler | `CalcNonLastTiling` | innerChunk 搜索、mode 8 peak UB、fallback |
| Safety kernels | `StreamingHeapTopKRowsSimt`、`ShardedHeapPhase1Simt`、`ShardedHeapPhase2Simt` | heap 选择、稳定比较、两阶段归并、64 位地址 |
| Fast SIMT | `SmallKRowsSimt`、`SortShortStridedRowsSimt`、`AxisToLastSimt/LastToAxisSimt` | 小 k、有序化、非末轴 layout 与 tail |
| Compact ABI | `PackDirectTiling`、`PackMergeDirectTiling` | 64 B/56 B Device POD 与互斥字段布局 |
| Launcher | `launch_top_k_kernel` | key/output selector 优先级与实际多 Kernel 图 |
| int64 overlay | `CastInputVf`、`LaunchInt64DynamicInt32Typed`、v9.0.1 single-core body | 全域范围证明、atomic flag、互斥 fallback |
| 构建接线 | `register_direct_launch_op` | 主 kernel、cast kernel 与 plugin 共同编译注册 |

## 证据边界

本卡记录 Direct 交付实现的 Host selector、Device body、launch graph 与构建接线，并以固定官方 TopKV2/950 源码解释被复用的 arch35 body。streaming/sharded heap、64 位 SIMT 转置、安全 override 与动态窄化属于 Direct 实现扩展，不应归因于官方 TopKV2。Frontmatter 固定的是已恢复并核对的官方 Commit 文件；Direct 交付文件不在这些官方 sources 中，其内容哈希仍未记录，不能用官方文件身份核对替代 Direct 实现复核。复用前需取得相应交付快照；若更新 CANN 版本，需重新核验 compact POD、tmp 查询与 official body 的 NaN/稳定索引行为。

- [asc_atomic_or](../../../../apis/simt_api/atomic_operations/asc_atomic_or.md)
