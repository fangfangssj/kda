---
type: Operator
title: "UnsortedSegmentSum 950 Direct Invoke 算子知识"
description: "说明 launch_unsorted_segment_sum 在 950 上的接口语义、Launcher 参数生成、15 条 Route、启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_host/unsorted_segment_sum_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simt_tiling.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simt_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simd_non_sort_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/unsorted_segment_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/unsorted_segment_sum.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/uss_simd_non_sort.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/uss_simd_split_col.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/unsorted_segment_sum_apt.cpp
    role: implementation
created_at: "2026-08-25T19:30:50Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UnsortedSegmentSum 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | <code>cann_bench::unsorted_segment_sum</code> → <code>unsorted_segment_sum_npu</code> → <code>launch_unsorted_segment_sum</code>；Direct Invoke |
| 计算语义 | data[N,*tail] 按一维 segment_ids[N] 分段求和，空段为 0 |
| 输入输出 | Direct 子域：data 为 FP16/BF16/FP32/INT32/INT64，ids 为 INT32/INT64，rank 1..8，输出 [num_segments,*tail] |
| 执行类型 | AIV_VECTOR 或 AIV-only entry 内 <code>asc_vf_call</code> 的 SIMT；无 AIC/Cube、无 MIX_CV |
| Kernel 路线 | R-001～R-015 均有生产 Host producer、launcher 与 Device body，均为 `active`；R-006～R-009 是同一 low wide Host 叶内的 Device 条件特化，不是额外 Route；无 Route 级 `conditional`；1、2 或 3 Kernel |
| 核心资源 | GM data/ids/output；low FP32 accumulator；FP32 partials；显式 TBuf 峰值 192 KiB |
| 来源闭包 | direct_implementation_closed：plugin、POD、blockDim、全部可达分支、entry/body、stream 边和构建接线闭合 |
| 复核边界 | 已完成源码闭包核对；不据此宣称已建立编译、运行、数值或性能结论。 |

Frontmatter 只列固定 Commit 的官方 950/arch35 相关源码，证明公开语义、平台接线及 SIMT/SIMD 转换基线。后文“Direct 转换件”的 Host、POD、launcher 和派生 Device 事实来自已闭合的转换源码；它们不是官方 TilingKey，也不被表述为官方原生实现。

### 来源与转换关系

| 层次 | 本卡保留的事实 | 证据边界 |
|---|---|---|
| 官方原生 | `UnsortedSegmentSum` 的输入/输出语义、950/arch35 接线、`SimtTilingData` 四个 `uint64_t` 字段、zero→guarded atomic 的 generic SIMT 关系，以及 non-sort/split-column 的 2D MTE、vector atomic 和列所有权原则 | 由 Frontmatter 的固定 HTTP 源码直接支撑；它们不直接证明下述 Direct selector 阈值或精度扩展 |
| Direct 调用适配 | `TORCH_LIBRARY_FRAGMENT`/`PrivateUse1`、`OptionalDeviceGuard`、contiguous 物化、Tensor workspace、`OpCommand::RunOpApi`、current stream、按值 POD、`blockDim` 和多 Kernel 有序 launch | 属于 Direct 转换件；不是官方图算子 Host/Tiling 的逐字搬运 |
| Device 派生补充 | low 的 FP32 accumulator+一次 `CAST_RINT`、small-owned、INT64 column-owned、single-segment partial/finish、1024-bin histogram、magic-div 与 wide row-vector 的具体阈值及拓扑 | 是为 Direct 声明域补全的派生路线；应以本卡的 Host→Device 闭包和失效边界复用，不应外推为其它平台或官方 key |

## 接口与语义

### 计算语义

令 <code>N=data.shape[0]</code>、<code>A=Π data.shape[1:]</code>、<code>S=num_segments</code>：

~~~text
y[s,a] = Σ data[n,a],  n∈[0,N) and segment_ids[n]=s
~~~

每个可达 body 在形成地址前检查 <code>id&lt;0 || id&gt;=S</code>，非法 id 跳过。FP16/BF16 在 FP32 中累加后一次 CAST_RINT；其余保持本 dtype。官方 README 还公开 UINT32/UINT64，但 Direct 不支持。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| data | [N,*tail]，转 contiguous | FP16/BF16/FP32/INT32/INT64 | rank 1..8 |
| segment_ids | [N]，转 contiguous | INT32/INT64 | 一维且长度 N；值域 Device 防护 |
| num_segments | Host 标量 | INT64 | S&gt;0；不同于官方 tensor ABI |
| y | [S,*tail]，ND contiguous | 同 data | 分段和，空段 0 |
| accumulator | 内部 GM | FP32 | low 为 [S,*tail]；R-004 为 [blockDim] |

### 入口与版本

官方 README、OpDef 和 CMake 把算子接到 <code>ascend950</code>/<code>arch35</code>。本卡覆盖 CANN 9.1.0、950/950 Direct：Torch PrivateUse1 取 current NPU stream，经 <code>OpCommand::RunOpApi</code> 调 launcher。官方图 kernel 按 1000/5000/6000 等 key 路由；Direct 不消费这些 key，而按 dtype/shape 自定义分发。

## Host/Launcher 执行流程

### 参数校验与前处理

依次检查 data rank、ids 一维且首维与 N 相等、S 与 dtype；产生 `dataContiguous/idsContiguous`；计算 N、A、S；创建 y；计算 tiling；所有 low 输入都无条件分配 FP32 [S,*tail]（包括 R-001），R-004 分配 FP32 [blockDim]；取 current stream 后 launch。无显式 same-device、shape 乘积溢出或 UB capacity 检查。

### 参数、Workspace 与生命周期

令 <code>C=max(GetCoreNumAiv(),1)</code>、<code>I=N*A</code>、<code>O=S*A</code>、<code>Binit=4</code>（low）或 data bytes：

~~~text
computeCores=min(ceil(I/2048),C)
perCoreInit=max(ceil(O/C),2048/Binit)
initCores=min(ceil(O/perCoreInit),C)
blockDim=max(1,max(computeCores,initCores))
~~~

需区分 Host 分配量与算法必需 workspace：所有 low 输入的 Host 分配量都是 <code>4*O</code> B；R-005～R-010 的三节点算法确实消费这块 accumulator，而 R-001 在顶层短路中不把 `accumulator` 传给任何 Device entry，因而其算法必需 workspace 为 0 B，`4*O` B 只是无效 Host 分配。R-004 的算法必需 partials 为 <code>4*blockDim</code> B。allocator 物理对齐不可见；POD 按值传 Device，外层 launch struct 只在 Host bridge 使用。

同一 stream 只建立已提交 Kernel 之间的执行顺序，不自动证明 GM owner 生命期。转换源码的 Host lambda 只按值捕获 `dataPtr/idsPtr/outputPtr/accumulatorPtr` 等裸地址，没有捕获 `dataContiguous`、`idsContiguous` 或 `accumulator` Tensor，也未显式展示 `record_stream` 或 completion owner。因此，所有 Route 的 contiguous 临时输入、R-004 partials 和 R-005～R-010 accumulator 能否保持到最后一个消费节点，都属于 `RunOpApi`/缓存 allocator 的外部契约，不由本源码单独证明。复用实现必须选择一个显式闭包：让异步 callback/lambda 捕获 Tensor owner，对所有临时分配正确 `record_stream`，或使用覆盖该 stream 完成点的 completion owner。R-001 的无效 accumulator 没有 Device consumer，不需跨 Kernel 保活；可在保持 launcher 分支互斥的前提下删除该分配。

### 启动图与失败传播

~~~text
R001/002/003: compute
R004: partial -> partials -> finish
R005/006..010: init FP32 -> compute -> cast
R011/012/013/014: init output -> compute
R015: SIMT init -> SIMT compute
~~~

节点均在同一 stream，无显式 event。launcher 为 void、Host lambda 固定返回 0；异步 fault 传播与上述 owner 保活均是框架/缓存 allocator 的外部契约。

## Route 与参数设计

### 分核与资源计算

各 body 用 <code>ceil(extent/blockDim)</code> 分行、列或 flattened 元素，空核返回。Vector 列域按 32 B 对齐但 GM 尾写只用有效 elements；SIMT 固定 2048 lanes。常量：wide=256 B，vector tile=8192，cast tile=16384，low small N/S=4，FP32 small N/S=16，hist N≥65536/S≤4096/window=1024，FP32 vector A≥256，INT64 column A≥512。

### Route 选择

#### 重要性顺序

下表的 P0→P2 只定义知识卡的阅读与复用优先级，不改变 launcher 的运行时选择顺序。分类口径是：R-001～R-015 都能从 `unsorted_segment_sum_npu` 的合法 dtype/shape 输入到达，因而都是 `active`；R-010/R-013/R-014/R-015 是 active safety fallback，不是仅编译未接线的备用 entry。本实现没有 Route 级 `conditional`；R-006～R-009 是 low wide entry 内由 index 宽度和 `innerDim` 确定的 active 子特化。

| 优先级 | Route | 分类 | Route 特征 | 复用时首先锁定 | Host/Device 选择位置 |
|---|---|---|---|---|---|
| P0 | R-011 | active fast | FP32/INT32 宽尾主干；2D MTE + native vector atomic | FP32 的 256 elements / INT32 的 256 B 门槛、batch4、128 KiB dataBuf | dtype 主分支 |
| P0 | R-006 | active fast | low+INT32 中宽尾；batch8×tile4096，FP32 workspace | 192 KiB allocation、升精度与一次降精度 | Host low-wide→Device `shortI32` |
| P0 | R-001 | active owned | low 小 N/S 宽尾；列独占、UB FP32 累加 | N/S guard、176 KiB UB、无 GM atomic | 顶层 if-1 |
| P0 | R-005 | active specialized | low 标量长行域；1024-bin window histogram | window owner、UB/GM 两级 atomic、cast 边 | low histogram 分支 |
| P1 | R-003 | active owned | INT64 宽尾列独占；SIMT 直接 GM 累加 | A≥512、列域互斥、无 init kernel | 顶层 if-3 |
| P1 | R-002 | active owned | FP32 小 N/S 宽尾；列独占 UB 累加 | N/S≤16、64 KiB UB、一次写回 | 顶层 if-2 |
| P1 | R-004 | active specialized | FP32 单段标量；partial→finish 两阶段归约 | 32 B blockSum、partials lifetime、barrier | 顶层 if-4 |
| P1 | R-007 | active fast | low+INT32 长尾；batch4×tile8192 | 2D stride、192 KiB allocation | Host low-wide→Device `!shortI32` |
| P1 | R-008 | active fast | low+INT64 中长尾；batch2×tile8192 | INT64 id guard、active/allocated UB 区分 | Host low-wide→Device `A<8192` |
| P1 | R-009 | active fast | low+INT64 超长尾；batch1×tile8192 | 单行复用、exact-byte tail | Host low-wide→Device `A>=8192` |
| P2 | R-010 | active safety | low 短尾通用 SIMT；FP32 GM atomic | flattened row 映射、three-kernel ordering | low fallback |
| P2 | R-012 | active fast | FP32 短尾 u32 specialization | magic/shift producer、u32 guard | FP32 branch |
| P2 | R-014 | active safety | INT32 短尾通用 SIMT | native integer atomic、zero edge | INT32 fallback |
| P2 | R-015 | active safety | INT64 短尾通用 SIMT | SIMT zero、INT64 atomic、stream edge | INT64 fallback |
| P2 | R-013 | active safety | FP32 超 u32 flattened 域 fallback | power-of-two/u32/u64 地址映射 | FP32 final fallback |

#### Selector 真实选择顺序

R-006～R-009 在 Host 上共享同一 `wideTail` entry 叶，之后由 `LowRowVectorScatter` 的 C++ Device body 按 `sizeof(Index)` 与 `innerDim` 选择 batch/tile；表中将这两层展开，不把 Device 子选择伪装成 Host key。

| 顺序 | Route | 选择层 / Key | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 low small-owned | Direct if-1 | low,N≤4,S≤4,A≥128 | 2 |
| 2 | R-002 FP32 small-owned | if-2 | FP32,N≤16,S≤16,A≥64 | 3 |
| 3 | R-003 INT64 column | if-3 | INT64,A≥512 | 4 |
| 4 | R-004 single reduce | if-4 | FP32,S=A=1 | dtype 分支 |
| 5 | R-005 low histogram | low-1 | low,A=1,S≤4096,N≥65536 | wide |
| 6 | R-006 low vec i32-short | Host low-wide → Device `shortI32` | low+i32,128≤A≤4096,非R001 | Host wide 未命中去 R010；Device 谓词假则 R007 |
| 7 | R-007 low vec i32-long | Host low-wide → Device `!shortI32` | low+i32,A&gt;4096,非R001 | Host wide 未命中去 R010 |
| 8 | R-008 low vec i64-mid | Host low-wide → Device `A<8192` | low+i64,128≤A&lt;8192,非R001 | Host wide 未命中去 R010 |
| 9 | R-009 low vec i64-long | Host low-wide → Device `A>=8192` | low+i64,A≥8192,非R001 | Host wide 未命中去 R010 |
| 10 | R-010 low SIMT | low fallback | low,非R001/R005,A&lt;128 | 终点 |
| 11 | R-011 native vector | FP32/INT32 | FP32 A≥256非R002；INT32 A≥64 | SIMT |
| 12 | R-012 FP32 magic | FP32 | 非前序,A&lt;256,I≤UINT32_MAX | R013 |
| 13 | R-013 FP32 generic | FP32 | 非前序,A&lt;256,I&gt;UINT32_MAX | 终点 |
| 14 | R-014 INT32 generic | INT32 | A&lt;64 | 终点 |
| 15 | R-015 INT64 generic | INT64 | A&lt;512 | 终点 |

~~~text
if R001; else if R002; else if R003; else if R004;
else low: histogram -> four wide subroutes -> SIMT
else FP32: vector -> magic32 -> generic64
else INT32: vector -> generic
else INT64: generic
~~~

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-011 | FP32/INT32 vector | FP32 A≥256且非R-002；INT32 A≥64；950 | init→vector | NativeRowVectorScatter | init 32 KiB；scatter 128 KiB；stream+GM atomic | 8192-element exact tail |
| R-006 | low i32 short | low+i32，128≤A≤4096，非R-001；950 | init→vec→cast | LowRowVectorScatter batch8,tile4096 | 192 KiB；4*O GM；PIPE+stream | Host wide guard 失败才去 R-010；exact bytes |
| R-001 | top if-1 | low，N/S≤4，A≥128；950 | single Vector | SmallRowsVectorOwned | 176 KiB；列独占；Host 仍分配但 body 不消费 `4*O` B acc | 8192 tail / 后续 Host 判断 |
| R-005 | low histogram | low，A=1，S≤4096，N≥65536；950 | init→hist→cast | HistogramScatter/Chunk | peak 96 KiB；UB/GM atomic；stream | 1024-bin window / low fallback |
| R-003 | top if-3 | INT64，A≥512；950 | single SIMT | NativeColumnOwnedVf | GM 直访；列独占 | thread-stride / INT64 fallback |
| R-002 | top if-2 | FP32，N/S≤16，A≥64；950 | single Vector | SmallNativeVectorOwned | 64 KiB；列独占 | 8192 tail / 后续 Host 判断 |
| R-004 | top if-4 | FP32，S=A=1；950 | partial→finish | two reduce VF | 32 B UB + 4*blockDim GM；barrier+stream | empty block partial=0 |
| R-007 | low i32 long | low+i32，A>4096，非R-001；950 | init→vec→cast | batch4,tile8192 | 192 KiB；4*O GM | Host wide guard 失败才去 R-010；exact bytes |
| R-008 | low i64 mid | low+i64，128≤A<8192，非R-001；950 | init→vec→cast | batch2,tile8192 | 192 KiB allocated；4*O GM | Host wide guard 失败才去 R-010；exact bytes |
| R-009 | low i64 long | low+i64，A≥8192，非R-001；950 | init→vec→cast | batch1,tile8192 | 192 KiB allocated；4*O GM | Host wide guard 失败才去 R-010；exact bytes |
| R-010 | low fallback | low，非R-001/R-005，A<128；950 | init→SIMT→cast | Fp32ScatterVf | peak 96 KiB；4*O GM；atomic+stream | flattened tail |
| R-012 | FP32 magic | FP32，前序未命中，A<256，I≤UINT32_MAX；950 | init→SIMT | NativeScatterMagicF32Vf | init 32 KiB；GM atomic | u32 domain / R-013 |
| R-014 | INT32 fallback | INT32，A<64；950 | init→SIMT | NativeScatterVf<int32,Index> | init 32 KiB；GM atomic | flattened tail |
| R-015 | INT64 fallback | INT64，A<512；950 | SIMT init→SIMT | ZeroOutputVf/NativeScatterVf | no explicit TBuf；stream+atomic | flattened tail |
| R-013 | FP32 final fallback | FP32，前序未命中，A<256，I>UINT32_MAX；950 | init→SIMT | NativeScatterVf<float,Index> | init 32 KiB；GM atomic | u64-capable tail |

### 分层存储与生命期总账

全部 active Route 都是 AIV-only，不调用 Cube/AIC、L1 或 L0A/L0B/L0C API，因而每条 Route 的 active L1/L0 均为 0 B。表中 UB 是单个节点的显式 `TBuf` 峰值；多 Kernel Route 的不同节点不同时占用 UB。每条 Route 的 GM data/ids 都可能指向 `contiguous()` 临时存储，其 owner 生命期统一受 `RunOpApi`/缓存 allocator 外部契约约束；表中另列跨节点 workspace 的额外生命期。

| Route | GM producer→consumer | 显式 UB 峰值 | L1/L0 | workspace 与生命期 |
|---|---|---:|---:|---|
| R-001 | caller data/ids→owned UB sums→y；Host 另分配但不传入 entry 的 FP32 acc | 176 KiB | 0/0 B | Host 分配 `4*O` B，算法必需 0 B；无效 acc 无 Device consumer，无需跨 Kernel 保活 |
| R-002 | caller data/ids→FP32 UB sum→y | 64 KiB | 0/0 B | 0；单 Kernel |
| R-003 | caller data/ids→owner-thread GM y | 0 B | 0/0 B | 0；单 Kernel |
| R-004 | data/ids→UB `blockSum`→GM partials→y | 32 B | 0/0 B | `4*blockDim` B；需外部 owner 覆盖 partial→finish，源码不单独证明保活 |
| R-005 | data/ids→UB histogram→GM FP32 acc→y | 96 KiB（cast 节点；hist 节点 4 KiB） | 0/0 B | `4*O` B；需外部 owner 覆盖 init→hist→cast |
| R-006 | low data/ids→low/FP32 UB→GM FP32 acc→y | 192 KiB | 0/0 B | `4*O` B；需外部 owner 覆盖三节点 |
| R-007 | low data/ids→low/FP32 UB→GM FP32 acc→y；batch4 | 192 KiB | 0/0 B | `4*O` B；需外部 owner 覆盖三节点 |
| R-008 | low data/ids→low/FP32 UB→GM FP32 acc→y；active batch2 | 192 KiB allocation | 0/0 B | `4*O` B；需外部 owner 覆盖三节点 |
| R-009 | low data/ids→low/FP32 UB→GM FP32 acc→y；active batch1 | 192 KiB allocation | 0/0 B | `4*O` B；需外部 owner 覆盖三节点 |
| R-010 | data/ids→GM FP32 acc→y | 96 KiB（cast 节点） | 0/0 B | `4*O` B；需外部 owner 覆盖 init→scatter→cast |
| R-011 | caller data/ids→128 KiB row UB→GM y | 128 KiB | 0/0 B | 0；y 跨 zero/vector 节点 |
| R-012 | caller data/ids→SIMT registers→GM y | 32 KiB（zero 节点） | 0/0 B | 0；y 跨 zero/magic 节点 |
| R-013 | caller data/ids→SIMT registers→GM y | 32 KiB（zero 节点） | 0/0 B | 0；y 跨 zero/native 节点 |
| R-014 | caller data/ids→SIMT registers→GM y | 32 KiB（zero 节点） | 0/0 B | 0；y 跨 zero/native 节点 |
| R-015 | caller data/ids→SIMT registers→GM y | 0 B | 0/0 B | 0；y 跨 SIMT zero/native 节点 |

### 实现安全与性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| wide | low 2*A≥256；INT32 4*A≥256 | 2D MTE/vector atomic | R001/006..011 | SIMT | wideTail |
| column | small N/S 或 INT64 A≥512 | 消除atomic | R001..003 | 后序 | if-1..3 |
| hist | A=1,N≥65536,S≤4096 | 每核1024 bins | R005 | 后序 | localHistogram |
| hist coverage | 隐含 blockDim≥ceil(S/1024) | window必须有owner | R005 | 无fallback | HistogramScatter |
| low UB | 固定 32768 low elements + 32768 FP32 elements | 192 KiB | R006..009 | 无IsCapable | LowRowVectorScatter |
| magic | I≤u32 | u32 index | R012 | R013 | magic entry |
| id | id&lt;0 or id≥S | 防越界 | 全部 | skip | bodies |
| ordering | same stream | 节点GM可见性 | multi-kernel | 无event fallback | launcher |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| SimtTilingData | inputOuterDim | u64/64 | sizeof32,align8,off0 | rows | =N | loops | N*A unchecked |
| 同上 | outputOuterDim | u64/64 | off8 | segments | =S | guard/address | S&gt;0 |
| 同上 | innerDim | u64/64 | off16 | elements | =A | route/address | unchecked |
| 同上 | maxThread | u64/64 | off24 | lanes | =2048 | VF launch | some entries constant |
| LaunchTiling | data | struct/256 | sizeof40,align8,off0 | POD | calc | launcher | Host-only |
| 同上 | blockDim | u32/32 | off32+4B tail padding | blocks | formula | grid/partials | C=0→1 |

### Route 源码闭包

`op_plugin/unsorted_segment_sum_plugin.cpp::unsorted_segment_sum_npu` 是所有 Route 的 Host producer：它产生 `dataTag/indexTag/N/S/A`，调用 `calc_unsorted_segment_sum_tiling`，并只为 low 或 FP32 单段标量分配 accumulator。`op_kernel/unsorted_segment_sum_kernel.cpp::launch_unsorted_segment_sum` 是唯一 router。下表按知识重要性排列，不改变 Host 真实短路顺序。

| 重要性 | Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|
| P0 | R-011 | FP32 `A>=256` 或 INT32 `4*A>=256` | `unsorted_segment_sum_npu`→`calc_unsorted_segment_sum_tiling`→dtype branch | `td`, y | `uss_init_f32/i32`→`uss_vec_f32/i32_{i32,i64}` | `VectorZeroOutput<T>`→`NativeRowVectorScatter<T,Index>` | `unsorted_segment_sum/CMakeLists.txt::register_direct_launch_op` |
| P0 | R-006 | low+i32 wide，`A<=4096` | 同上；launcher 选 low-wide，body 产生 `shortI32` | `td`, FP32 acc | `uss_init_f32`→`uss_vec_f16/bf16_i32`→`uss_cast_f16/bf16` | `VectorZeroOutput<float>`→`LowRowVectorScatter` batch8/tile4096→`VectorCastFromFp32` | 同上 |
| P0 | R-001 | 顶层 if-1 | 同上；low, `N<=4,S<=4,2*A>=256`；plugin 已分配未消费 acc | `td`, data/index tags；launcher 接收但本分支忽略 acc | `uss_small_f16/bf16_{i32,i64}` | `SmallRowsVectorOwned<LowT,Index>` | 同上 |
| P0 | R-005 | low histogram | 同上；`A=1,S<=4096,N>=65536` | `td`, FP32 acc | `uss_init_f32`→`uss_hist_f16/bf16_{i32,i64}`→cast | `LocalHistogramScatter`→`LocalHistogramChunkVf`→vector GM merge | 同上 |
| P1 | R-003 | 顶层 if-3 | 同上；INT64, `A>=512` | `td` | `uss_col_i64_{i32,i64}` | entry 列分核→`NativeColumnOwnedVf<int64_t,Index>` | 同上 |
| P1 | R-002 | 顶层 if-2 | 同上；FP32, `N<=16,S<=16,4*A>=256` | `td` | `uss_owned_f32_{i32,i64}` | `SmallNativeVectorOwned<Index>` | 同上 |
| P1 | R-004 | 顶层 if-4 | plugin 分配 `[blockDim]` partials；launcher 判 `FP32,S=A=1` | `N`, `blockDim`, partials | `uss_single_partial_f32_{i32,i64}`→`uss_single_finish_f32<<<1>>>` | `SingleSegmentPartialF32Vf`→`FinishSingleSegmentF32Vf` | 同上 |
| P1 | R-007 | low+i32 wide，`A>4096` | low-wide entry 内 `shortI32=false` | `td`, FP32 acc | 与 R-006 相同 entry 家族 | `LowRowVectorScatter` batch4/tile8192 | 同上 |
| P1 | R-008 | low+i64 wide，`A<8192` | low-wide entry 内 `sizeof(Index)==8 && A<8192` | `td`, FP32 acc | `uss_init_f32`→`uss_vec_f16/bf16_i64`→cast | `LowRowVectorScatter` batch2/tile8192 | 同上 |
| P1 | R-009 | low+i64 wide，`A>=8192` | low-wide entry 内 `sizeof(Index)==8 && A>=8192` | `td`, FP32 acc | 与 R-008 相同 entry 家族 | `LowRowVectorScatter` batch1/tile8192 | 同上 |
| P2 | R-010 | low safety fallback | low 分支的 histogram/wide 均未命中 | `td`, FP32 acc | `uss_init_f32`→`uss_acc_f16/bf16_{i32,i64}`→cast | `LaunchFp32Scatter`→`Fp32ScatterVf`→`VectorCastFromFp32` | 同上 |
| P2 | R-012 | FP32 u32 magic | FP32，前序未命中，`N*A<=UINT32_MAX` | `td` | `uss_init_f32`→`uss_magic_f32_{i32,i64}` | `GetUintDivMagicAndShift`→`NativeScatterMagicF32Vf` | 同上 |
| P2 | R-014 | INT32 safety fallback | INT32，wide 未命中 | `td` | `uss_init_i32`→`uss_native_i32_{i32,i64}` | `LaunchNativeScatter`→`NativeScatterVf<int32_t,Index>` | 同上 |
| P2 | R-015 | INT64 safety fallback | INT64，column-owned 未命中 | `td` | `uss_init_i64`→`uss_native_i64_{i32,i64}` | `InitOutputPartition<int64_t>`→`NativeScatterVf<int64_t,Index>` | 同上 |
| P2 | R-013 | FP32 u64 safety fallback | FP32，前序未命中，`N*A>UINT32_MAX` | `td` | `uss_init_f32`→`uss_native_f32_{i32,i64}` | `LaunchNativeScatter`→`NativeScatterVf<float,Index>` | 同上 |

### Dormant 源码与 Route 内条件子路径

| 类别 | 已存在符号 | 生产可达性 | 处理原则 / 重新启用条件 |
|---|---|---|---|
| dormant low native | `uss_native_f16/bf16_*`、`uss_init_f16/bf16`、`CastFromFp32Vf` | launcher 的 low 分支只调 FP32 init、`uss_acc/vec/hist_*` 与 vector cast；这些 entry/helper 没有生产 caller | 不计入 low 路线的节点、UB 或精度账；重启须先明确是否放弃 FP32 累加契约，再补 selector、zero/cast 顺序和数值边界 |
| dormant 单段 direct atomic | `uss_single_f32_{i32,i64}`→`SingleSegmentF32Vf` | R-004 只调 partial/finish entry，不调 direct atomic entry | 不可当作 R-004 fallback；重启需新增 y 清零节点、证明跨 AIV 归约顺序，并与 partials 路线做互斥 selector |
| dormant column entry | `uss_col_f32_*`、`uss_col_i32_*`、`uss_col_f16/bf16_*`、`LowColumnOwnedVf` | launcher 只对 INT64 `A>=512` 调 `uss_col_i64_*`；`kLowColumnOwnedMinInner` 没有 active consumer | 不构造额外列路线；重启须增 Host 优先级与 shape guard，为 low 闭合 accumulator/output 生命期，并证明列所有权与累加顺序 |
| dormant helper | `SmallRowsOwnedVf`、`LocalHistogramVf`、`kGmAlignBytes` | 没有 active entry/body 调用；histogram 生产路线只调单缓冲 `LocalHistogramChunkVf` | 不将四缓冲 histogram 或 scalar-owned 算入 active UB/节点；重启须新增 caller、重算 UB 峰值与 window 覆盖，并闭合 local atomic ABI |
| active Route 内 conditional | R-006～R-009 的 `shortI32/batchRows/tileElements` | Host 先命中 low wide entry，Device C++ body 再依 `sizeof(Index)` 与 `innerDim` 选 batch8/4/2/1 | 这些是四个 active 实现特化，共享 entry 家族但 UB/tail 账分别闭合；不把它们误认为 Host key |

## Kernel 实现

### 公共执行流程

全部 entry 为 AIV_ONLY。Vector body 使用 TPipe/TBuf、DataCopyPad、Duplicate/Cast/Add 与 PIPE_ALL；SIMT 由 entry 以 2048 lanes 启动。地址统一为 <code>data[n*A+a]</code> 和 <code>output[id*A+a]</code>。atomic Route 先零 GM；列所有权 Route 自行写零并独占列；low 多节点只写 FP32 accumulator，最终 cast 写 y。

### Route R-011：Native Row-Vector

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | FP32/INT32 branches | FP32 A≥256非R002；INT32 A≥64 | uss_vec_* |

#### Device 数据流
##### 函数级调用链

~~~text
VectorZeroOutput<T> -> NativeRowVectorScatter<T,Index>(batch4,tile8192)
~~~

##### 源码等价伪码

~~~text
Host: zero y on blockDim AIVs; launch vec entry on the same stream
Device block b:
  own rows [ceil(N/blockDim)*b, ...); allocate 4*32KiB dataBuf
  for rows<=4, for off in [0,A) by 8192 elements:
    DataCopyPad 2D rows into padded UB stride; wait PIPE_ALL
    SetAtomicAdd<T>
    for each row: id=ids[row]; if 0<=id<S, copy exactly valid bytes UB->y[id,off:]
    SetAtomicNone; wait PIPE_ALL
~~~

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| init/dataBuf/y | 32/128 KiB / id*A+off | FP32/INT32 | zero→Copy→atomic | two nodes |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  init["init 32 KiB"]
  scatter["data 128 KiB"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=128 KiB; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| dataBuf/y | Copy/zero+atomic | MTE3/caller | batch/call | PIPE_ALL/stream | AtomicNone |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| vector atomic | SetAtomicAdd<T> | repeated ids | native dtype；exact bytes |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | vector branch / NativeVector entries / RowVectorScatter | two kernels |

### Route R-006：Low Row-Vector INT32 Short

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | LowRowVectorScatter | i32,128≤A≤4096 | batch8,tile4096 |

#### Device 数据流

##### 函数级调用链

~~~text
init FP32 -> uss_vec_low_i32 -> LowRowVectorScatter -> cast
~~~

##### 源码等价伪码

~~~text
8-row 2D CopyIn; cast aligned rows FP32; valid id => vector atomic acc; final RINT
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| low/float/acc | row*align:64+128 KiB / id*A+off:4*O | low/FP32 | MTE→Cast→atomic→cast | UB 按 batch 复用；acc 需外部 owner 覆盖到 cast |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  low["low 64 KiB"]
  fp["fp32 128 KiB"]
~~~

##### 字节账与复用顺序

~~~text
scatter UB=192 KiB; init=32 KiB; cast=96 KiB; workspace=4*O B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| batch buffers | Copy/Cast | atomic MTE3 | next batch | PIPE_ALL/AtomicNone | write complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| 2D MTE/atomic | exact blockLen, padded UB stride | row tile | padding not written to GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | low wide / vec_i32 / LowRowVectorScatter | batch8/tile4096 |

### Route R-001：Low 小 N/S 列所有权

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | launcher if-1 | low,N/S≤4,A≥128 | uss_small_* |

#### Device 数据流

##### 函数级调用链

~~~text
uss_small_* -> SmallRowsVectorOwned -> Duplicate sums -> CopyIn/Cast/Add -> Cast/CopyOut
~~~

##### 源码等价伪码

~~~text
owned column tile: zero sum[S]; scan N in order; valid id => sum[id]+=float(row); cast_rint each S once
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| low/float/sum TBuf | 0:16 KiB / 0:32 KiB / s*8192:128 KiB | low/FP32 | MTE→Cast→Add→Cast→MTE3 | row/tile |
| Host 无效 accumulator | `4*O` B GM allocation | FP32 | plugin 分配→launcher 本分支忽略 | 无 Device consumer；不需跨 Kernel 保活，不计入算法 workspace |

##### 分层内存布局

~~~mermaid
block-beta
  columns 3
  low["low 16 KiB"]
  fp["fp32 32 KiB"]
  sum["sum 128 KiB"]
~~~

##### 字节账与复用顺序

~~~text
UB=176 KiB; Host allocation=4*O B; algorithm-required workspace=0 B
PIPE_ALL 后覆盖 low/float，tile 完成后覆盖 sum
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| low/float/sum | Copy/Cast/Add | next stage | next row/tile | PIPE_ALL | 消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| pipeline | PIPE_ALL+列独占 | UB/y | low→FP32，输入顺序加，一次RINT；有效elements写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | if-1 / uss_small_* / SmallRowsVectorOwned | 完整单节点 |

### Route R-005：Low 局部 Histogram

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | localHistogram | low,A=1,S≤4096,N≥65536 | init→hist→cast |

#### Device 数据流

##### 函数级调用链

~~~text
VectorZeroOutput -> LocalHistogramScatter -> LocalHistogramChunkVf -> vector atomic merge -> VectorCastFromFp32
~~~

##### 源码等价伪码

~~~text
Host: same-stream zero(acc[0:S]) -> histogram(blockDim blocks) -> cast(acc to y)
W = ceil(S / 1024)
Device block b:
  window = b % W
  groupRank = b / W
  groupCount = (blockDim + W - 1 - window) / W
  rowsPerCore = ceil(N / groupCount)
  begin = rowsPerCore * groupRank
  count = begin >= N ? 0 : min(rowsPerCore, N - begin)
  if count == 0: return
  segmentBegin = window * 1024
  elements = min(1024, S - segmentBegin)       # last window tail
  Duplicate(histogram[0:elements], 0)
  SIMT threads scan rows [begin, begin+count):
    id = ids[row]
    if segmentBegin <= id < segmentBegin+elements:
      AtomicAdd_UB(histogram[id-segmentBegin], float(data[row]))
  SetAtomicAdd<float>; copy exactly elements from UB to acc[segmentBegin:]; SetAtomicNone
Final cast node: acc[0:S] -> CAST_RINT -> y[0:S]
~~~

`LocalHistogramScatter` 没有断言 `blockDim>=W`。若该前提不成立，`b%W` 无法为每个 window 产生 owner，后续 cast 会读到未被 histogram 节点覆盖的窗口；生产 selector 未提供 Device fallback。

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| init/hist/cast UB | 0:32/4/96 KiB | FP32/low | three nodes | per-node |
| accumulator | window*1024 / 4*S B | FP32 | init/hist→cast | 需 `RunOpApi`/allocator 外部 owner 覆盖三个节点 |

##### 分层内存布局

~~~mermaid
block-beta
  columns 3
  init["init 32 KiB"]
  hist["hist 4 KiB"]
  cast["cast 96 KiB"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=96 KiB; workspace=4*S B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| histogram/acc | SIMT/GM atomic | merge/cast | node/call end | PIPE_ALL/stream | window owner；acc 需外部 owner |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| hist | UB atomic+GM vector atomic | FP32 acc | window tail≤1024；one RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | predicate / uss_hist_* / HistogramScatter/Chunk | three kernels |

### Route R-003：INT64 列所有权

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | if-3 | INT64,A≥512 | uss_col_i64_* |

#### Device 数据流

##### 函数级调用链

~~~text
uss_col_i64_* -> AIV column partition -> NativeColumnOwnedVf(2048)
~~~

##### 源码等价伪码

~~~text
each thread owns a: zero y[:,a]; scan N; valid id => y[id,a]+=data[n,a]
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| GM data/y | n*A+a / s*A+a，scalar | INT64 | caller→thread→caller | column unique |

##### 分层内存布局

N/A：SIMT registers + GM，无 InitBuffer。

##### 字节账与复用顺序

~~~text
explicit UB=0; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| y[:,a] | owner thread | same thread | next row | ownership | columns disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| zero/add | thread order | column | INT64；no atomic；invalid id skip |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | if-3 / uss_col_i64_* / NativeColumnOwnedVf | single SIMT |

### Route R-002：FP32 小 N/S 列所有权

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | if-2 | FP32,N/S≤16,A≥64 | uss_owned_f32_* |

#### Device 数据流

##### 函数级调用链

~~~text
uss_owned_f32_* -> SmallNativeVectorOwned -> per output row Duplicate/scan/Add/CopyOut
~~~

##### 源码等价伪码

~~~text
for owned tile, for s: sum=0; scan rows with valid id==s; sum+=data; write y[s] once
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| dataBuf/sumBuf | 0/32 KiB each | FP32 | MTE→Add→MTE3 | row/S reuse |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  data["data 32 KiB"]
  sum["sum 32 KiB"]
~~~

##### 字节账与复用顺序

~~~text
UB=64 KiB; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| data/sum | Copy/Duplicate/Add | Add/CopyOut | next row/S | PIPE_ALL | stage complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| accumulate | PIPE_ALL+列独占 | sum/y | FP32 input order；8192 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | if-2 / uss_owned_* / SmallNativeVectorOwned | 无 init kernel/atomic |

### Route R-004：FP32 单段两阶段归约

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | 交付 plugin/kernel.cpp | accumulator / if-4 | FP32,S=A=1 | partial→finish |

#### Device 数据流

##### 函数级调用链

~~~text
uss_single_partial_* -> SingleSegmentPartialF32Vf -> partials[block]
same stream -> uss_single_finish_f32 -> FinishSingleSegmentF32Vf -> y[0]
~~~

##### 源码等价伪码

~~~text
threads sum id==0; asc_reduce_add; lane0 atomic UB blockSum; barrier; thread0 writes partial; one thread finishes
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| blockSum/partials | 0:32 B / blockIdx:4*blockDim B | FP32 UB/GM | lanes→thread0→finish | UB 仅 partial 节点；partials 需外部 owner 覆盖到 finish |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  ub["blockSum 32 B"]
  gm["partials 4*blockDim B"]
~~~

##### 字节账与复用顺序

~~~text
UB(partial)=32 B; UB(finish)=0; workspace=4*blockDim B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| blockSum/partials | lane0/partial | thread0/finish | finish/call end | syncthreads/stream | 生产者完成；partials 需外部 owner |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| reduce | asc_reduce_add+AtomicAdd+barrier | UB/partials | FP32；id!=0 skip；空核写0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 plugin/kernel.cpp | partial alloc / two entries / two VF | two-kernel closure |

### Route R-007：Low Row-Vector INT32 Long

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | LowRowVectorScatter | i32,A&gt;4096 | batch4,tile8192 |

#### Device 数据流

##### 函数级调用链

~~~text
init -> low vector i32(batch4) -> cast
~~~

##### 源码等价伪码

~~~text
Host low+i32 wide -> same-stream zero(acc), vec_i32, cast(y)
Device block b:
  own a contiguous row slice; batchRows=4, tileElements=8192
  2D CopyIn up to 4 low rows -> CAST_NONE to FP32 -> valid-id vector atomic to acc
  restore AtomicNone after each tile; exact-byte GM tail
Final node: acc[0:O] -> CAST_RINT -> y[0:O]
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| scatter/acc | 192 KiB / 4*O B | low/FP32 | Copy/Cast→atomic→cast | UB 按 batch 复用；acc 需外部 owner 覆盖到 cast |

##### 分层内存布局

~~~mermaid
block-beta
  columns 1
  scatter["low 64 KiB + fp32 128 KiB"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=192 KiB; workspace=4*O B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| batch | Copy/Cast | atomic | next | PIPE_ALL | rows≤4 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| pipeline | 2D MTE+CAST_NONE+atomic | acc | exact tail；final RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | A&gt;4096 / vec_i32 / LowRowVectorScatter | batch4 |

### Route R-008：Low Row-Vector INT64 Mid

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | LowRowVectorScatter | i64,128≤A&lt;8192 | batch2,tile8192 |

#### Device 数据流

##### 函数级调用链

~~~text
init -> low vector i64(batch2) -> cast
~~~

##### 源码等价伪码

~~~text
Host low+i64 wide -> same-stream zero(acc), vec_i64, cast(y)
Device block b:
  own a contiguous row slice; because 128<=A<8192, batchRows=2 and tileElements=8192
  2D CopyIn two low rows -> CAST_NONE to FP32
  read int64 id before destination formation; valid rows vector-atomic exactly A-tail bytes to acc
Final node: one CAST_RINT from FP32 acc to low y
~~~

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| allocated/active UB | 192/96 KiB | low/FP32 | MTE→Cast→atomic | only first2 slots |
| accumulator | `4*O` B GM | FP32 | init→vector atomic→cast | 需外部 owner 覆盖三节点 |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  alloc["allocated 192 KiB"]
  active["active 96 KiB"]
~~~

##### 字节账与复用顺序

~~~text
allocated UB=192 KiB; workspace=4*O B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| first2 slots | Copy/Cast | atomic | next | PIPE_ALL | inactive slots unread |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| id/data | i64 guard+vector atomic | address/acc | guard before address |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | i64 mid / vec_i64 / LowRowVectorScatter | batch2 |

### Route R-009：Low Row-Vector INT64 Long

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | LowRowVectorScatter | i64,A≥8192 | batch1,tile8192 |

#### Device 数据流
##### 函数级调用链

~~~text
init -> low vector i64(batch1) -> cast
~~~

##### 源码等价伪码

一行一 tile 搬运、升 FP32、vector atomic，最后统一降精度。

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| allocated/active UB | 192/48 KiB | low/FP32 | Copy→Cast→atomic | slot0 reuse |
| accumulator | `4*O` B GM | FP32 | init→vector atomic→cast | 需外部 owner 覆盖三节点 |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  alloc["allocated 192 KiB"]
  active["active 48 KiB"]
~~~

##### 字节账与复用顺序

~~~text
allocated UB=192 KiB; workspace=4*O B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| slot0 | Copy/Cast | atomic | next row | PIPE_ALL/AtomicNone | write complete |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| tail | exact DataCopyPad bytes | acc | no padded GM write；final RINT |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | i64 long / vec_i64 / LowRowVectorScatter | batch1 |

### Route R-010：Low 通用 SIMT

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | low fallback | A&lt;128，非hist | uss_acc_* |

#### Device 数据流
##### 函数级调用链

~~~text
VectorZeroOutput(acc) -> LaunchFp32Scatter -> Fp32ScatterVf -> VectorCastFromFp32
~~~

##### 源码等价伪码

flatten I；shift/u32/u64 映射 row；guard id；AtomicAdd FP32 accumulator；final RINT。

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| init/scatter/cast | 32 KiB / id*A+a:4*O B / 96 KiB | FP32/low | init→atomic→cast | 同 stream 定序；acc 需外部 owner 覆盖到 cast |

##### 分层内存布局

~~~mermaid
block-beta
  columns 3
  init["32 KiB"]
  simt["registers+GM"]
  cast["96 KiB"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=96 KiB; workspace=4*O B
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| accumulator | init/atomic | cast | call end | same stream | producers complete |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| atomic/cast | Simt::AtomicAdd+stream+RINT | FP32 acc | local&lt;count；one cast |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | low fallback / DEFINE_LOW_ENTRY / Fp32ScatterVf | three kernels |

### Route R-012：FP32 Magic-Div SIMT

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | FP32 branch | A&lt;256,I≤u32,非前序 | uss_magic_f32_* |

#### Device 数据流
##### 函数级调用链

~~~text
VectorZeroOutput -> GetUintDivMagicAndShift -> NativeScatterMagicF32Vf
~~~

##### 源码等价伪码

row=UintDiv(u32 index,magic,shift)；a=index-row*A；guard id；FP32 atomic。

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| init/scatter | 32 KiB / scalar GM | FP32 | Duplicate/lane→y | u32 domain |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  init["32 KiB"]
  simt["registers+GM"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=32 KiB; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| y | init/atomic | caller | call end | same stream | I≤u32 |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| map/update | magic UintDiv+AtomicAdd | y | count clamp |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | I guard / magic entry / MagicF32Vf | u32 specialization |

### Route R-014：INT32 通用 SIMT

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | INT32 fallback | A&lt;64 | uss_native_i32_* |

#### Device 数据流
##### 函数级调用链

~~~text
VectorZeroOutput<int32> -> NativeScatterVf<int32,Index>
~~~

##### 源码等价伪码

~~~text
Host: VectorZeroOutput<int32>(y,O); then launch native_i32 on the same stream
Device block b:
  begin/count = contiguous slice of I=N*A
  for inputIndex in thread-stride slice:
    row = shift when A is power-of-two, else guarded u32/u64 division
    a = inputIndex-row*A; id=ids[row]
    if 0<=id<S: AtomicAdd<int32>(y[id*A+a], data[inputIndex])
~~~

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| init/y | 32 KiB / 4*O B | INT32 | zero→atomic | call |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  init["32 KiB"]
  simt["no TBuf"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=32 KiB; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| y | zero/atomic | caller | call end | stream | native atomic |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| update | AtomicAdd<int32> | y | Device INT32 overflow semantics |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | INT32 else / native entry / NativeScatterVf | two kernels |

### Route R-015：INT64 通用 SIMT

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | INT64 fallback | A&lt;512 | simt init→native |

#### Device 数据流
##### 函数级调用链

~~~text
uss_init_i64 -> ZeroOutputVf; same stream -> uss_native_i64_* -> NativeScatterVf<int64,Index>
~~~

##### 源码等价伪码

SIMT zero O；SIMT flatten I；guard id；INT64 atomic。

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| y | begin+local / id*A+a; 8*O B | INT64 GM | zero→atomic→caller | call |

##### 分层内存布局

N/A：两节点均 SIMT registers + GM，无 InitBuffer/dynamic UB。

##### 字节账与复用顺序

~~~text
explicit UB=0; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| y | zero | atomic | caller | same stream | zero complete |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| nodes/update | stream+AtomicAdd<int64> | y | count clamp；INT64 semantics |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | INT64 branch / init+native / Zero+Native VF | two SIMT kernels |

### Route R-013：FP32 通用 SIMT

#### 适用条件
##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct | 交付 kernel.cpp | FP32 fallback | A&lt;256,I&gt;u32 | uss_native_f32_* |

#### Device 数据流
##### 函数级调用链

~~~text
VectorZeroOutput -> LaunchNativeScatter -> NativeScatterVf<float,Index>
~~~

##### 源码等价伪码

power-of-two shift，否则按当前 AIV 区间选择 u32/u64 divide；guard；atomic。

#### 内存与数据搬运
##### 资源与搬运表

| 缓冲区 | offset/容量 | dtype | producer→consumer | lifetime/复用 |
|---|---|---|---|---|
| init/y | 32 KiB / id*A+a | FP32 | zero→atomic | call |

##### 分层内存布局

~~~mermaid
block-beta
  columns 2
  init["32 KiB"]
  simt["no TBuf"]
~~~

##### 字节账与复用顺序

~~~text
peak UB=32 KiB; workspace=0
~~~

| 复用子区 | 写入者 | 读取者 | 下一覆盖 | 依据 | 条件 |
|---|---|---|---|---|---|
| y | zero/atomic | caller | call end | stream | guard first |

#### 同步、精度与尾块
##### 同步与数值边界表

| 阶段 | 机制 | 保护对象 | 精度/尾块 |
|---|---|---|---|
| update | FP32 atomic | y | repeated-id order nondeterministic |

#### 关键实现
##### 符号证据表

| 阶段 | 文件 | 符号 | 实现事实 |
|---|---|---|---|
| Host/entry/body | 交付 kernel.cpp | FP32 else / native entry / NativeScatterVf | u64 fallback |

## 算子特定优化

### Trick T-001：同 Stream Kernel 边替代图模式全局屏障

#### 触发与源码依据

- 适用 Route：所有 init→compute Route。
- 触发条件：Direct scaffold 不复刻官方 <code>SetScheduleMode(1)</code>。
- 固定官方源码/相关映射：[<code>op_kernel/arch35/unsorted_segment_base.h:109-124</code>](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/unsorted_segment_base.h)

**官方源码（固定 Commit 连续逐字摘录）**：

~~~cpp
for (COM_T inputIndex = block_idx * blockDim.x + threadIdx.x; inputIndex < inputLength;
     inputIndex = inputIndex + blockNums * blockDim.x) {
    COM_T inputSegmentIndex = Simt::UintDiv(inputIndex, magic, shift);
    COM_T segmentOffset = inputIndex - inputSegmentIndex * innerDimSize;
    const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
    if (outputSegmentIndex < 0 || outputSegmentIndex >= outputOuterDimSize) {
        continue;
    }
    const uint64_t outputIndex = outputSegmentIndex * innerDimSize + segmentOffset;
    asc_atomic_add(outputGm + outputIndex, xGm[inputIndex]);
}
~~~

#### 机制与设计原因

官方 generic 在 <code>InitGm</code> 后 <code>SyncAll</code>，Host 设置 schedule mode。Direct 保留 flatten、guard、atomic，把 zero/scatter 拆为同 stream kernels，以 kernel 边提供跨 AIV ordering；low 再加 cast 边。

#### 迁移边界

- 适用：Direct 全局初始化→跨核 atomic。
- 前提：同一有序 stream；GM owner 须由 Tensor 捕获、`record_stream` 或 completion owner 独立覆盖后继节点。
- 失效：跨 stream 必须 event。

#### 错误做法

不同 stream 无 event 会使 zero/scatter 竞争；官方 <code>SyncAll</code> 是可验证反例边界。

### Trick T-002：二维整行搬运加 Vector Atomic

#### 触发与源码依据

- 适用 Route：R-006～R-009、R-011。
- 触发条件：wide tail 且前序列所有权未命中。
- 固定官方源码/相关映射：[<code>op_kernel/arch35/uss_simd_non_sort.h:91-109</code>](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/uss_simd_non_sort.h)

**官方源码（固定 Commit 连续逐字摘录）**：

~~~cpp
LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
uint64_t offset = blockOffsetX + sLoop * td_->baseS * td_->innerDim + aLoop * td_->baseA;
CopyIn(xLocal, xGm_, offset, rows, cols, td_->innerDim - cols);
xQue_.EnQue<X_T>(xLocal);
xLocal = xQue_.DeQue<X_T>();
event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
SetAtomicAdd<X_T>();
~~~

#### 机制与设计原因

官方 non-sort 用 2D MTE 后 vector atomic。Direct native 固定 batch4×8192；low 先把最多 32768 low elements 升 FP32 再写 accumulator，消除逐元素 row division/标量 atomic。192 KiB 是 allocation，不是 active bytes。

#### 迁移边界

- 适用：连续尾维、同一行目标段相同。
- 前提：保持 blockLen/srcStride/dstStride 单位；low 需 FP32 GM 和 final cast。
- 失效：短 tail 回 SIMT；INT64 大 tail回列所有权。

#### 错误做法

把 aligned elements 当有效长度会写坏下一行；Direct 只写 <code>elements*sizeof(T)</code>。

### Trick T-003：列所有权消除初始化 Kernel 与 GM Atomic

#### 触发与源码依据

- 适用 Route：R-001～R-003。
- 触发条件：small N/S wide tail，或 INT64 A≥512。
- 固定官方源码/相关映射：[<code>op_kernel/arch35/uss_simd_split_col.h:98-113</code>](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_kernel/arch35/uss_simd_split_col.h)

**官方源码（固定 Commit 连续逐字摘录）**：

~~~cpp
for (uint64_t i = 0; i < rows; i++) {
    uint64_t dstIdx = idsLocal.GetValue(i);
    if (dstIdx < 0 || dstIdx >= td_->outputOuterDim) {
        continue;
    }
    uint64_t dstOffset = dstIdx * colsAlign;
    Add(yLocal[dstOffset], yLocal[dstOffset], xLocal[i * colsAlign], cols);
}
~~~

#### 机制与设计原因

官方 split-column 让 core 独占列、UB 累加后一次写回。Direct R-001/002 复用该原则；R-003 在 SIMT 直接串行更新 GM INT64。没有两个 worker 写同一列，所以无需 GM atomic 和独立 zero kernel。

#### 迁移边界

- 适用：列数足够，扫描 N×S 受 guard 控制。
- 前提：列域不重叠；owner 写出全部 S 行。
- 失效：shape guard 失败转 row-vector/SIMT。

#### 错误做法

只写命中段会使空段未初始化；R-001/002 先 Duplicate，R-003 逐段写零。

### Trick T-004：复用官方 Compute/Init Core 取最大公式

#### 触发与源码依据

- 适用 Route：全部。
- 触发条件：同 launcher 同时覆盖 input compute 与 output init。
- 固定官方源码/相关映射：[<code>op_host/arch35/unsorted_segment_sum_simt_tiling.cpp:45-54</code>](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simt_tiling.cpp)

**官方源码（固定 Commit 连续逐字摘录）**：

~~~cpp
usedCoreNum_ = Ops::Base::CeilDiv(dataShapeSize_, static_cast<uint64_t>(maxThread_));
usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
uint64_t outSize = outputOuterDim_ * innerDim_;
uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
normBlockData = std::max(normBlockData, UB_MIN_FACTOR / valueTypeBytes_);
uint64_t initUsedCore = Ops::Base::CeilDiv(outSize, normBlockData);
usedCoreNum_ = std::max(usedCoreNum_, initUsedCore);
~~~

#### 机制与设计原因

Direct 保留“input/2048 lanes”和“至少 2 KiB init block”两侧取最大并限于 AIV cores。low 初始化 FP32 accumulator，故 Binit=4。统一 blockDim 也匹配 R-004 partial 数。

#### 迁移边界

- 适用：节点共享 grid 且空核受保护。
- 前提：乘法/ceil-div 不溢出。
- 失效：每节点不同 grid 或 hist windows&gt;blockDim 时重算。

#### 错误做法

low 按 2 B 算 init 粒度会偏离实际 FP32 accumulator；Direct 明确用 4 B。

## 模板与实现对比

| 对比维度 | R001..003 | R004/005 | R006..011 | R012..015 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中/优先级 | column最高 | specialized reduce | wide tail | short/fallback | 先去冲突再放大搬运 |
| 模板/dtype | low/FP32/INT64 | FP32或low | low/FP32/INT32 | native dtype | Direct子域 |
| 循环/分核 | columns | partial/windows | rows+column tiles | flattened | shape |
| TBuf | 176/64/0 KiB | 32 B/4/96 KiB | 192/128 KiB | scatter 0 B | InitBuffer |
| 算法必需 workspace | R001=0（Host 额外无效分配4*O）；R002/3=0 | R004=4*blockDim；R005=4*O | low 4*O | low 4*O/其它0 | FP32 contract |
| API | UB/GM column | reduce/hist | 2D MTE+vector atomic | scalar atomic | tail width |
| 同步/精度 | ownership | barriers+stream | PIPE+stream | stream+atomic | one RINT |
| fallback | next | shape next | SIMT | terminal | source priority |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方 README 不支持 | 官方 README 不支持 | 官方支持 950PR/950DT |
| Host/Launcher | 不适用 | 不适用 | 官方 ascend950/arch35；Direct GetCoreNumAiv/current stream |
| Kernel/资源 | 不适用 | 不适用 | 2048 SIMT lanes，AIV-only，max TBuf 192 KiB |

## 限制与失败边界

- Direct 不支持 UINT32/UINT64、多维 ids 或 tensor num_segments。
- Host 无显式 same-device、乘积溢出、输出容量和 UB IsCapable 检查。
- R-005 隐含 <code>blockDim≥ceil(S/1024)</code>，无断言。
- R-006..009 固定 192 KiB，不得无条件迁移到非 950。
- 多 Kernel 只依赖同 stream；跨 stream 必须 event。
- 同 stream 不保活 GM owner；contiguous 临时输入、R-004 partials 和 R-005～R-010 accumulator 依赖 `RunOpApi`/allocator 外部契约，复用时须显式捕获 owner、`record_stream` 或持有 completion owner。
- launcher void、lambda 返回 0；异步 fault 在框架边界。
- low FP32 atomic 顺序可能不确定；只保证最后一次降精度。
- integer overflow 遵循 Device dtype，不饱和。
- dormant entry 与官方更多 deterministic/sort key 均不属于 Direct dispatch。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 语义/平台 | index/unsorted_segment_sum/README.md | 产品支持/公式 | 950与语义 |
| 构建 | index/unsorted_segment_sum/CMakeLists.txt | SUPPORT_COMPUTE_UNIT/TILING_DIR | ascend950/arch35 |
| OpDef | index/unsorted_segment_sum/op_host/unsorted_segment_sum_def.cpp | UnsortedSegmentSum | dtype/ND/950 |
| 公共 Tiling | index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_tiling_base.cpp | GetPlatformInfo/FlatInput | core/thread/flatten |
| SIMT POD | index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simt_tiling.h | SimtTilingData | ABI |
| SIMT producer | index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simt_tiling.cpp | DoOpTiling/PostTiling | blockDim/schedule |
| non-sort selector | index/unsorted_segment_sum/op_host/arch35/unsorted_segment_sum_simd_non_sort_tiling.cpp | IsCapable | 256 B/UB |
| SIMT helper | index/unsorted_segment_sum/op_kernel/arch35/unsorted_segment_base.h | SimtComputeSegment/InitGm | zero/guard/atomic |
| generic body | index/unsorted_segment_sum/op_kernel/arch35/unsorted_segment_sum.h | KernelUnsortedSegmentSum | POD consumer |
| non-sort body | index/unsorted_segment_sum/op_kernel/arch35/uss_simd_non_sort.h | USSKernelSimdNonSort | 2D MTE/atomic |
| split-col body | index/unsorted_segment_sum/op_kernel/arch35/uss_simd_split_col.h | USSKernelSimdSplitCol | column owner |
| official router | index/unsorted_segment_sum/op_kernel/unsorted_segment_sum_apt.cpp | unsorted_segment_sum | key→body，区别Direct |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 4×u64 arch35 fields | order/width/by-value | SimtTilingData |
| 调用顺序 | zero→scatter、acc→cast、partial→finish | same stream/event | launcher |
| workspace/lifetime | Host 对所有 low 分配4*S*A；算法消费者仅 R005～010；R001 必需0；R004 partials=4*blockDim | same stream 只定序；临时输入/partials/active acc 需外部 owner；192 KiB为 allocation | plugin/InitBuffer/RunOpApi boundary |
| 同步 | PIPE/barrier/atomic/stream | AtomicNone；producer先完成 | bodies |
| 精度/tail | low→FP32→one RINT；exact bytes | no low atomic/padding write | LowRowVector/Cast |
| 失败边界 | invalid id skip、empty core return | guard before address；dormant unreachable | NativeScatterVf/DataTag |
