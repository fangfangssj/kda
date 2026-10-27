---
type: Operator
title: "Unique 950 Direct Invoke 算子知识"
description: "说明 cann_bench::unique、unique_npu 与 launch_unique 在 950 上的接口语义、Host/POD/selector、Route 选择、GM/UB/workspace 交接、排序去重与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_constant.h"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_kernel.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h"
    role: implementation
created_at: "2026-08-25T19:40:24Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Unique 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench.unique` → `torch.ops.cann_bench.unique` → `cann_bench::unique_npu` → `calc_unique_tiling` → `launch_unique`；Direct Invoke。 |
| 计算语义 | 把 contiguous ND 输入按存储顺序展平；在各 Route 的完整语义域内返回按比较序排列的唯一值 `y`，可选 `inverse` 满足 `y[inverse.reshape(-1)] == x.reshape(-1)`。FP32 official 的完整 unique 保证限定到无 NaN 输入。 |
| 输入输出 | 输入 rank 1–8、元素数 `[1,UINT32_MAX]`；dtype 为 uint8/int8/int32/int64/fp16/fp32/bf16；`y` 与输入同 dtype，`inverse` 为 int64 且 shape 与输入相同。 |
| 执行类型 | 固定域与稠密整数：多 Kernel AIV carrier + SIMT VF，稠密 extrema 节点含 AIV_VECTOR；官方 fallback：单 AIV carrier 内 SIMT sort/scan/gather 与 AIV UB/MTE。无可达 Cube/AIC 或 `MIX_CV`。 |
| Kernel 路线 | 重要性顺序为 P0 R-003、P1 R-008、P2 R-002、P3 R-006、P4 R-005、P5 R-004、P6 R-007、P7 R-001。P 级只组织知识；运行时仍按 Host dtype 顺序分派，并在 dense Route 内按 value-domain 短路。 |
| 核心资源 | 固定域 `first/rank` 两张 uint32 表；稠密整数为官方 workspace + 128 B 对齐的 `first/rank/partial/meta` 尾区；官方路线为 indices/sort/flags/scan GM 与双半区 sort UB、shape UB。 |
| 来源闭包 | `direct_implementation_closed`：Direct binding、plain POD、selector、全部 Host 可达 entry/body/helper、多 Kernel GM 边、同步、生命周期与构建接线闭合；固定官方 Commit 只锚定字节一致的 UniqueDim body，不伪称 Direct wrapper 为官方入口。 |
| 证据边界 | 正文结论来自固定官方源码与 Direct 转换源码的静态闭包；不把构建产物、运行记录或转换交付状态写成算子机制。 |

## 接口与语义

### 计算语义

令 `v=x.reshape(-1)`、`N=v.numel()`。在各 Route 的完整语义域内，实现产生按路线比较序排列的不同值序列 `y`；浮点输入无 NaN 时严格升序：

```text
y = sorted(distinct(v))
return_inverse=false: result=[y]
return_inverse=true : result=[y,inverse], inverse.shape=x.shape,
                      对任意线性位置 i，y[inverse[i]]=v[i]
```

Direct 任务语义与官方 `UniqueDim` 变体不同：这里没有 `dim`、`sorted` 或 counts 输出，永远按整个 contiguous 存储展平，且只暴露可选 inverse。适配器把官方 POD 固定为 `rowLen=1, dim=0, outerSize=innerSize=1, returnCounts=0`，所以官方 body 被复用为一维标量去重。固定域与稠密整数路线是 Direct 派生补充：它们用数值域表代替比较排序，但保持升序输出与 inverse 语义。

浮点边界按源码分两类。FP16/BF16 固定域把 `+0/-0` 合成一个 key，并在输出保留首个零的符号；其映射假定任务域不含 NaN。FP32 official 的排序比较只用 `<`/`>`，所以 NaN 与任意值都得到 compare-equal；dedup 比较却把任一含 NaN 的相邻对判为不等。于是 NaN 会保留且还可能隔断本应相邻的普通等值类，例如 `[1, NaN, 1]` 的稳定排序次序可能不变，compact 可产生 `[1, NaN, 1]`，两个 1 还会取得不同 inverse rank。Host 没有 NaN guard 或下一 fallback，故 R-007/R-008 的完整 unique 语义只在无 NaN 域成立；在该域内 `+0/-0` compare-equal，稳定 merge 与 gather 保留首个零的符号。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | contiguous ND，rank 1–8 | uint8/int8/int32/int64/fp16/fp32/bf16 | 必须 defined、strided、contiguous；`1<=numel<=UINT32_MAX`。不做自动 Contiguous。 |
| `return_inverse` | Host bool | bool | 默认 false；决定结果 arity、inverse 分配与编译期/节点级裁剪。 |
| `y` | 初始分配 `[N]`，完成后原地 resize 为 `[numOut]` | 同 x | 严格按路线定义的数值顺序 compact；`1<=numOut<=N`。 |
| `inverse` | 与 x shape 相同；仅属性 true 时返回 | int64 | `inverse[i]` 是 `x.flatten()[i]` 在 y 中的 rank。false 时仅分配 1 元素 scratch，所有可达 body 均不写正式 inverse。 |
| `countsScratch` | `[1]` | int64 | 为官方模板 ABI 提供占位；Direct 固定 `returnCounts=false`，不可作为输出读取。 |
| `shapeScratch` | `[27]` | int64/按 uint64 位模式使用 | Kernel 在槽 1 写 `numOut`；Host 只 D2H 该 8 B 后同步 stream。稠密预检还暂用槽 2 传 Device route flag。 |

### 入口与版本

Python 包装层 `cann_bench.unique` 在 `return_inverse=false` 时从 dispatcher 的 `Tensor[]` 解包为单 Tensor，属性为 true 时转为二元 tuple。`TORCH_LIBRARY_FRAGMENT` 声明 `unique(Tensor x, bool return_inverse=False) -> Tensor[]`；Meta 与 PrivateUse1 分别绑定 `unique_meta/unique_npu`。Meta 只能返回最大 y 与完整 inverse shape，真实 y 长度由 Device 数据决定。`unique_npu` 直接取得 current NPU stream 并调用 `launch_unique`，随后 `aclrtMemcpyAsync` 读取 `shapeScratch+8` 的输出个数并同步，再收缩 `y`；inverse 不 resize。它不是 ACLNN、不是 runtime registry lookup，也没有外部算子二进制或 CPU 实现。

本卡固定 CANN 9.1.0、950/Ascend 950。source mapping 证明候选中的 `unique_dim_{constant,kernel,simt,simt_sort}.h` 与固定 Commit `955f8b33985b6d72989ebff341bb9c2c564a51bc` 字节一致；Direct wrapper、plain POD、固定域/稠密整数 selector 与多 Kernel 图由转换候选派生，不能外推为官方 `UniqueDim` 的公开接口或官方变体。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序为：绑定 `x.device()`；`CheckInput` 检查 defined、strided+contiguous、rank、numel 与七种 dtype；按最大长度分配 y、可选 inverse 及两个 scratch；`calc_unique_tiling` 再检查元素数/元素宽度并查询 AIV core 数；按 dtype 冻结 Route 与 workspace；取当前 stream 后同步执行 direct closure。未知 torch dtype 在进入 tiler 前失败；AIV core 数为 0 或元素宽度不在 `{1,2,4,8}` 时抛异常。

入口不把非连续输入转连续，也不接受空输入。所有 shape 被压成逻辑一维；输出 inverse 仍按原输入 shape 分配，线性内容不变。

### 参数、Workspace 与生命周期

设 `C=GetCoreNumAiv()`、`N=numel`、`A(z)=ceil(z/128)*128`：

```text
official:
  indicesOff=0
  sortOff=A(4N)
  flagsOff=A(sortOff+4N)
  partialOff=A(flagsOff+4N)
  prefixOff=A(partialOff+128C)
  shapeOff=A(prefixOff+4(C+1))
  transposeOff=A(shapeOff+27*8)
  officialBytes=transposeOff                  # dim=0，不分配 transpose payload

fixed D∈{256,65536}:
  first=[0,4D), rank=[4D,8D), workspaceBytes=8D

dense integer:
  denseOff=A(officialBytes)
  first=[denseOff,denseOff+4*2^20)
  rank =[denseOff+4*2^20,denseOff+8*2^20)
  partial=后续 4C 个 int64（每核 32 B）
  meta=再后续 2 个 int64
  denseBytes=8*2^20+(4C+2)*8
  workspaceBytes=denseOff+denseBytes
```

`unique_npu` 实际分配 `max(workspaceBytes,officialWorkspaceBytes,128)` 字节。稠密尾区与 official 区不别名：前置 reduce/finalize/dense table 的缓存状态不能覆盖官方 fallback 所需的 indices/sort/flags。workspace、shape scratch、inverse/count scratch 与最大 y 都由当前调用持有；同 stream Kernel 图及最后 D2H 完成后，Host 才 resize/返回。

### 启动图与失败传播

```text
以下按真实 Host 顺序 H1→H4，不按 P 级：
H1 u8/i8:
  P2/R-002 inv : clear -> mark8 -> compact -> inverse8
  P7/R-001 noinv: clear -> mark8 -> compact
H2 fp16/bf16:
  P5/R-004 inv : clear -> mark16 -> compact -> inverse16
  P0/R-003 noinv: clear -> mark16 -> compact
H3 int32/int64:
  P3/R-006 inv / P4/R-005 noinv:
    reduce -> finalize(routeFlag) -> clear -> mark -> compact -> [inverse]
    -> pending_official
       V1 routeFlag==0: return
       V2 routeFlag==1: TPipe::Reset -> UniqueDimKernel<T,RETURN_INVERSE,false>
H4 fp32:
  P1/R-008 inv / P6/R-007 noinv:
    launch_unique_official -> UniqueDimKernel<float,RETURN_INVERSE,false>

all routes -> D2H shape[1] -> synchronize stream -> validate 1<=numOut<=N -> resize y
```

所有 Host Kernel 以同一个 current stream 顺序提交，GM/workspace 是节点间交接介质；因此 consumer 等待其前序 producer。wrapper 为 `void` 且没有逐个 `<<<>>>` 的 Host 返回检查；异步 Device 错误最终在 D2H/stream synchronize 边界暴露。D2H 或同步失败立即抛错，不执行 resize。稠密宽域仍会提交 clear/mark/compact/optional inverse，但这些 body 读 `meta` 后提前返回；最后 pending 节点才运行官方实现。

### 节点级 launch、读写集与完成关系

`config.blockDim=C`，但 compact/finalize/prefix 这类全域串行汇总节点固定只发 1 个 AIV block。下表同时给出 grid、线程主体与跨节点 GM 边；这也是多 Kernel 图不能任意重排的最小闭包。

| 节点 | Host launch grid | Device 线程/执行单元 | 主要读取 | 主要写入 | 后继与完成关系 |
|---|---:|---|---|---|---|
| `unique_clear_u32` | C | AIV carrier → 512-thread VF | `total,value` | fixed `first[0:D)` | same-stream 后 `mark8/16`；各 block 清自己的 domain 分片 |
| `unique_fixed_mark8/16` | C | AIV carrier → 512-thread VF | x、first sentinel | first presence/first-zero index | same-stream 后单 block compact；跨 block 冲突由幂等 0 或 `AtomicMin` 收敛 |
| `unique_fixed_compact` | 1 | AIV carrier + 256-thread VF + 1 KiB TBuf | first，zero 类还读 x | y、rank、`shape[1]` | inverse 可选消费 rank；Host 最终消费 shape |
| `unique_fixed_inverse8/16` | C | AIV carrier → 512-thread VF | x、rank | inverse | Host 最终 stream 同步 |
| `unique_dense_reduce_i32/i64` | C | AIV Vector/MTE，非 VF 主体 | x tiles | `partial[4*b:4*b+4)` | same-stream 后 finalize MTE2 搬入全部 C 条记录 |
| `unique_dense_finalize` | 1 | MTE2 → 512-thread VF → MTE3 | partial | `meta={min,max}`、`shape[0:4]={0,UINT64_MAX,routeFlag,0}` | clear/mark/compact 读 meta；pending 以 MTE2 读完整 route block |
| `unique_dense_clear` | C | AIV carrier → 512-thread VF | meta | `first[0:range)` | valid range 才写；same-stream 后 mark |
| `unique_dense_mark_i32/i64` | C | AIV carrier → 512-thread VF | x、meta、first | first | valid 且 range>1 时 `AtomicMin`；range=1 只由首分片发布 `first[0]=0` |
| `unique_dense_compact_i32/i64` | 1 | AIV carrier + 256-thread VF + 1 KiB TBuf | meta、first | y、rank、`shape[1]` | valid 时完成 dense y；inverse 可选消费 rank |
| `unique_dense_inverse_i32/i64` | C | AIV carrier → 512-thread VF | x、meta、rank | inverse | 仅 R-006/V1 有效；V2 立即返回 |
| `unique_pending_i{32,64}_{noinv,inv}` | C | AIV entry；先 MTE2 route，必要时 official phases | `shape[0:4]`；V2 再读 x/official workspace | V1 无写；V2 写 y/[inverse]/shape | 每个 core 取得同一 flag；V2 在 `TPipe::Reset` 后进入含 `SyncAll` 的 official body |
| `unique_f32_{noinv,inv}` | C | AIV entry → official SIMT/UB phases | x、plain POD | y/[inverse]、official workspace、27 槽 shape | 单 entry 内以 `SyncAll` 闭合；Host D2H 等待 entry |

`unique_dense_finalize` 对 `meta[0:2]` 使用 SIMT 标量 GM 写，对 route block 使用 32 B MTE3；因此二者不能混称为同一发布方式。fixed/dense compact 只保证 `shape[1]` 有效，official Phase8 才会覆盖完整 27 槽描述符。Host 对所有 Route 始终只复制 `shape[1]` 的 8 B。

## Route 与参数设计

### 分核与资源计算

共同分核为 `blockDim=C`，`per=ceil(N/C)`，block `b` 处理 `[b*per,min(N,(b+1)*per))`；固定域 clear 则对 D 用同一公式。SIMT 主节点用 512 threads，compact 用 256 threads；官方 body 的普通 SIMT 为 256 threads，gather 为 2048 threads，tileSize=512。

稠密 extrema 的每核 AIV Vector tile 为 `192KiB/sizeof(T)` 元素，DMA 有效字节为 `current*sizeof(T)`，右侧 pad 到 32 B；`ReduceMin/ReduceMax` 的逻辑长度仍是 `current`，padding 不参与归约。每核产生 32 B `{min,max,0,0}`。finalize 单核把 `C*32 B` 搬入 UB，判断 `uint64(max)-uint64(min)<2^20`，并通过一整块 32 B MTE3 写出 route flag。

### Route 选择

| 优先级 | Route | Route 特征 | 适用域 | 实现前提 | Host 顺序 | 短路/fallback |
|---:|---|---|---|---|---|---|
| P0 | R-003 16-bit fixed/noinv | 65536 key 固定域表、三 Kernel、无 inverse | fp16/bf16，`returnInverse=false` | 无 NaN 的 Direct 输入域；两张 256 KiB 表；AIV core 可用 | H2a：H1 未命中后，`else if dtype∈{fp16,bf16}`，再按属性选 noinv | dtype 分支一旦命中不再尝试后续 Route；无 Device fallback |
| P1 | R-008 fp32 official/inv | 官方比较排序单 Kernel，编译期启用 inverse | fp32，`returnInverse=true`；完整 unique 需无 NaN | official workspace、plain POD、跨核 `SyncAll`；Host 不筛 NaN | H4b：H1/H2/H3 均未命中；合法 dtype 只剩 fp32，再选 inv | `launch_unique` 的 dormant radix 分支不可达；含 NaN 也直接进入 official，无语义 fallback |
| P2 | R-002 8-bit fixed/inv | 256 key 固定域表、四 Kernel、rank lookup | uint8/int8，`returnInverse=true` | 两张 1 KiB 表；first/rank 的 same-stream 可见性 | H1b：首个 `if dtype∈{u8,i8}` 命中，再按属性选 inv | 首个 dtype 分支短路；无 fallback |
| P3 | R-006 dense integer/inv | extrema 预检、稠密表、可选 official body、启用 inverse | int32/int64，`returnInverse=true` | 213024 B reduce UB；official/dense 区不别名；32 B route publish | H3b：H1/H2 未命中后，`else if dtype∈{i32,i64}`，再按属性选 inv | V1：`max-min<2^20` 完成 dense；V2：否则前序 body 早退，pending official fallback |
| P4 | R-005 dense integer/noinv | 与 R-006 相同的 value-domain 分流，裁掉 inverse 节点 | int32/int64，`returnInverse=false` | 同 R-006，但不分配正式 inverse 输出 | H3a：同一 H3 dtype 分支，再按属性选 noinv | V1/V2 与 R-006 相同；wide 进入 noinv official fallback |
| P5 | R-004 16-bit fixed/inv | 65536 key 固定域表、四 Kernel、canonical rank | fp16/bf16，`returnInverse=true` | R-003 的固定域前提 + inverse GM | H2b：H2 dtype 分支命中后选 inv | H2 短路；无 fallback |
| P6 | R-007 fp32 official/noinv | 官方比较排序单 Kernel，裁掉 inverse | fp32，`returnInverse=false`；完整 unique 需无 NaN | official workspace、plain POD、跨核同步；Host 不筛 NaN | H4a：前三个 dtype 分支均未命中，再选 noinv | dormant radix 不可达；含 NaN 也直接 official，无语义 fallback |
| P7 | R-001 8-bit fixed/noinv | 256 key 固定域表、三 Kernel、裁掉 inverse | uint8/int8，`returnInverse=false` | R-002 的固定域前提；rank 表仍由 compact 产生 | H1a：首个 dtype 分支命中后选 noinv | H1 短路；无 fallback |

```text
H1 if dtype in {u8,i8}:
     returnInverse ? P2/R-002 : P7/R-001
H2 else if dtype in {f16,bf16}:
     returnInverse ? P5/R-004 : P0/R-003
H3 else if dtype in {i32,i64}:
     route = returnInverse ? P3/R-006 : P4/R-005
     V1 if uint64(max)-uint64(min) < 2^20: dense body, pending returns
     V2 else: dense helpers short-return, pending runs official fallback
H4 else:  # CheckInput 后只剩 fp32
     returnInverse ? P1/R-008 : P6/R-007
other: CheckInput failure before selector
```

源码还定义 `UNIQUE_ROUTE_RADIX_FLOAT32`、`ConfigureFloat32Radix` 与四个 radix 节点，但 `calc_unique_tiling` 没有调用该配置函数，fp32 落到 `UNIQUE_ROUTE_OFFICIAL_SORT`；故 radix 是 dormant，不是正式 Route。

### 模板/实现执行矩阵

| 优先级/Route | Host 顺序 | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|---|
| P0/R-003 | H2a | fixed D65536/noinv | 950、N合法、fp16/bf16、无NaN域、noinv | 3 Kernel | `ClearU32→MarkFixed<uint16_t>→CompactFixed` | first/rank各256KiB；零key atomic；same stream | domain/线程span tail；无fallback |
| P1/R-008 | H4b | official/inv | fp32+inverse；完整语义需无NaN | 单 Kernel | `UniqueDimKernel<float,true,false>` | official GM workspace；8416B UB；`SyncAll` | core/tile/merge tail；含NaN无fallback；radix dormant |
| P2/R-002 | H1b | fixed D256/inv | 950、N合法、u8/i8+inverse | 4 Kernel | `ClearU32→MarkFixed<uint8_t>→CompactFixed→InverseFixed<uint8_t>` | first/rank各1KiB；mark atomic；same stream | core/domain tail；无fallback |
| P3/R-006 | H3b+V1/V2 | dense/inv | i32/i64+inverse | 6 前置节点 + 1 pending节点 | `DenseRangeVectorReducer<T>`、dense SIMT、`InverseDense<T>`、conditional official | 8MiB表+32C+16B；official独立区；route block | wide 前序早退，pending official |
| P4/R-005 | H3a+V1/V2 | dense/noinv | i32/i64+noinv | 5 前置节点 + 1 pending节点 | 同上但裁掉 `InverseDense<T>` | 同 R-006 | wide 进入 noinv official |
| P5/R-004 | H2b | fixed D65536/inv | fp16/bf16、无NaN域+inverse | 4 Kernel | R-003 主体 + `InverseFixed<uint16_t>` | R-003 资源 + inverse GM | core/domain tail；无fallback |
| P6/R-007 | H4a | official/noinv | fp32+noinv；完整语义需无NaN | 单 Kernel | `UniqueDimKernel<float,false,false>` | official GM workspace；8416B UB；`SyncAll` | core/tile/merge tail；含NaN无fallback；radix dormant |
| P7/R-001 | H1a | fixed D256/noinv | 950、N合法、u8/i8+noinv | 3 Kernel | `ClearU32→MarkFixed<uint8_t>→CompactFixed` | first/rank各1KiB；same stream | core/domain tail；无fallback |

### 隐式资源 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 输入域 | `1<=N<=UINT32_MAX` | 索引、first/rank 与 flag 为 uint32 | 全部 | Host failure | `CheckInput/calc_unique_tiling` |
| 固定域 dtype | u8/i8→256；f16/bf16→65536 | 可穷举 key 空间，compact 按 numeric order 扫描 | R-001..4 | 其它 dtype 去后续 Route | `calc_unique_tiling/FixedKindForDType` |
| 稠密范围 | `max>=min && uint64(max)-uint64(min)<2^20` | 两张固定 2^20 uint32 表可覆盖全部值 | R-005/6 dense branch | Device flag=1，pending official | `FinalizeDenseRange/DenseRangeValid` |
| 稠密区不别名 | `denseOff=A(officialBytes)` | 避免预检缓存状态污染官方跨核 workspace | R-005/6 | 无替代布局 | `calc_unique_tiling/LaunchDenseInteger` |
| extrema UB | 固定 192KiB inputQ +16KiB work+32B partial | Reduce API 工作区与 MTE3 publish | R-005/6 reduce | 无 Host capability fallback | `DenseRangeVectorReducer::Init` |
| official tile | `tileSize=512, blockDimX=256` | 两半各512×int64供 SIMT merge | official 分支、R-007/8 | 无替代 key | `UniqueDimKernel::Init` |
| counts 禁用 | `returnCounts=0,positionsOffset=0` | Direct 无 counts；模板编译期裁掉 positions/count body | 全部 official body | 不得读取 countsScratch | `calc_unique_tiling/DEFINE_UNIQUE_ENTRY` |
| route publish | finalize 以 32B MTE3 写 shape[0..3]；pending 以 32B MTE2 读 | 避免短 Kernel 标量 cache race | R-005/6 | flag0直接return；flag1 Reset后official | `unique_dense_finalize/DEFINE_PENDING_UNIQUE_ENTRY` |
| dormant radix | 无 selector producer | body/enum 存在不等于 Host 可达 | 无正式 Route | fp32始终 R-007/8 | `ConfigureFloat32Radix/calc_unique_tiling` |

### POD/ABI

Direct `UniqueDimTilingData` 是 32 个连续 `int64_t`：`sizeof=256`、`alignof=8`、字段 `i` 的 offset 为 `8*i`；`static_assert(sizeof==32*sizeof(int64_t))` 锁定总长。下表中的 producer 均为 `calc_unique_tiling`，consumer 为字节一致的官方 `UniqueDimKernel` 或 direct entry。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `UniqueDimTilingData` | `numInp` | int64/64 | off0 | element | `N` | `Init/Process` | `[1,UINT32_MAX]` |
| 同上 | `rowLen` | int64/64 | off8 | element/row | 1 | compare/gather | Direct 固定1 |
| 同上 | `returnInverse` | int64/64 | off16 | bool | 属性 | Host只镜像；模板由launcher选 | 0/1 |
| 同上 | `returnCounts` | int64/64 | off24 | bool | 0 | Host只镜像 | Direct固定0 |
| 同上 | `usedCoreNum` | int64/64 | off32 | core | C | launch grid | >0 |
| 同上 | `elementsPerCore` | int64/64 | off40 | element/core | `ceil(N/C)` | official coreStart/len | 可能有空核 |
| 同上 | `tailCoreElements` | int64/64 | off48 | element | `max(N-(C-1)per,0)` | official末核 | 非负 |
| 同上 | `blockDimX` | int64/64 | off56 | thread | 256 | official vf_call | gather另用2048常量 |
| 同上 | `inputDtypeSize` | int64/64 | off64 | B/element | elementBytes | Device不读 | `{1,2,4,8}` |
| 同上 | `tileSize` | int64/64 | off72 | element | 512 | official sort UB | 固定 |
| 同上 | `numTiles` | int64/64 | off80 | tile | `ceil(N/512)` | Device不读 | 诊断镜像 |
| 同上 | `indicesOffset` | int64/64 | off88 | byte | 0 | official bind | uint32[N] |
| 同上 | `sortBufOffset` | int64/64 | off96 | byte | `A(4N)` | official bind | uint32[N] |
| 同上 | `flagsOffset` | int64/64 | off104 | byte | `A(sortOff+4N)` | official bind | uint32[N] |
| 同上 | `positionsOffset` | int64/64 | off112 | byte | 0 | counts=false不读 | 与indices别名但inactive |
| 同上 | `partialSumOffset` | int64/64 | off120 | byte | `A(flagsOff+4N)` | official scan | C×128B |
| 同上 | `globalPrefixOffset` | int64/64 | off128 | byte | `A(partialOff+128C)` | official prefix | uint32[C+1] |
| 同上 | `shapeOutOffset` | int64/64 | off136 | byte | `A(prefixOff+4(C+1))` | Device不绑定 | 预留27×8B |
| 同上 | `workspaceSize` | int64/64 | off144 | byte | `transposeOff` | Device不读 | official总量 |
| 同上 | `dim` | int64/64 | off152 | axis | 0 | transpose/shape | Direct固定0 |
| 同上 | `outerSize` | int64/64 | off160 | element | 1 | transpose | inactive |
| 同上 | `innerSize` | int64/64 | off168 | element | 1 | transpose/gather | 固定1 |
| 同上 | `transposeDstOffset` | int64/64 | off176 | byte | `A(shapeOff+216)` | dim0不读 | 等于officialBytes |
| 同上 | `inputDimNum` | int64/64 | off184 | dim | 1 | shape writer | 固定1 |
| 同上 | `inputDim0` | int64/64 | off192 | element | N | shape writer | y最大长度 |
| 同上 | `inputDim1` | int64/64 | off200 | element | 0 | 不读 | inactive |
| 同上 | `inputDim2` | int64/64 | off208 | element | 0 | 不读 | inactive |
| 同上 | `inputDim3` | int64/64 | off216 | element | 0 | 不读 | inactive |
| 同上 | `inputDim4` | int64/64 | off224 | element | 0 | 不读 | inactive |
| 同上 | `inputDim5` | int64/64 | off232 | element | 0 | 不读 | inactive |
| 同上 | `inputDim6` | int64/64 | off240 | element | 0 | 不读 | inactive |
| 同上 | `inputDim7` | int64/64 | off248 | element | 0 | 不读 | inactive |

`UniqueLaunchConfig` 是 Host-only 聚合：`tiling`、dormant `sortTiling`、`blockDim`、总/dense/official/radix workspace 字节及 offset、`route`、`fixedDomain`。它按 C++ 引用传给 launcher，不序列化给某一个 Device entry；可达 consumer 只有 `tiling/blockDim/workspaceBytes/denseWorkspaceBytes/denseWorkspaceOffset/officialWorkspaceBytes/route/fixedDomain`，全部 radix 字段在正式 Route inactive。

### shape scratch 协议

`shapeScratch` 是独立的 27×uint64 GM 缓冲，不是 `tiling.shapeOutOffset` 指向的 official workspace 预留区；Direct entry 收到的是独立 Tensor 指针。27 槽沿用三组 `1+8` 的动态 shape 描述，但 Host 只消费 `shape[1]`：

| 槽 | official Phase8 写值 | fixed/dense 写值 | Direct Host 是否读取 |
|---:|---|---|---|
| 0 | `0x80000000 \| inputDimNum`，Direct 恒为 `0x80000001` | finalize 暂写 0；fixed 不写 | 否 |
| 1 | y 的 `numOut` | compact 写 `numOut`；dense finalize 先写 `UINT64_MAX` 防止旧值冒充结果 | 是，D2H 8 B |
| 2..8 | y 的其余维，Direct rank1 因而保持初始化值 | dense 槽2暂作 `routeFlag`；其余不保证 | 否 |
| 9/10 | inverse 描述；槽10为 `N` 或 noinv 时 1 | 不保证 | 否 |
| 18/19 | counts 描述；Direct counts=false，槽19为1 | 不保证 | 否 |
| 20..26 | 其余维初始化值 | 不保证 | 否 |

这解释了两个必要边界：一是 dense pending 必须在 official fallback 可能覆盖 shape 前先读取槽2；二是 Host 不能把 fixed/dense 路线未初始化的其余 26 槽当作公开 shape 元数据。`numOut` 用 uint64 写、int64 读，但由 `1<=numOut<=N<=UINT32_MAX` 保证位模式和值域一致。

### Route 源码闭包

| 优先级/Route | Host 顺序 | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|
| P0/R-003 | H2a | fixed D65536 | `calc_unique_tiling` | scalar args + config | clear/mark16/compact | `ClearU32/MarkFixed<uint16_t>/CompactFixed` | direct op CMake |
| P1/R-008 | H4b | official/inv | 同上 | plain `UniqueDimTilingData` | `unique_f32_inv` | 官方 `UniqueDimKernel<float,true,false>` 与 simt/sort helpers | 同上 |
| P2/R-002 | H1b | fixed D256 | 同上 | scalar args + config | clear/mark8/compact/inverse8 | `ClearU32/MarkFixed<uint8_t>/CompactFixed/InverseFixed<uint8_t>` | 同上 |
| P3/R-006 | H3b+V1/V2 | dense/inv | Host dtype + Device range flag | config + official POD | reduce/finalize/clear/mark/compact/inverse/pending | Vector reducer、dense SIMT；宽域到官方四文件 | 同上 |
| P4/R-005 | H3a+V1/V2 | dense/noinv | 同上 | 同上 | reduce/finalize/clear/mark/compact/pending | 同上但裁掉 inverse body | 同上 |
| P5/R-004 | H2b | fixed D65536 | `calc_unique_tiling` | scalar args + config | clear/mark16/compact/inverse16 | `ClearU32/MarkFixed<uint16_t>/CompactFixed/InverseFixed<uint16_t>` | 同上 |
| P6/R-007 | H4a | official/noinv | 同上 | plain `UniqueDimTilingData` | `unique_f32_noinv` | 官方 `UniqueDimKernel<float,false,false>` 与 simt/sort helpers | 同上 |
| P7/R-001 | H1a | fixed D256 | 同上 | scalar args + config | clear/mark8/compact | `ClearU32/MarkFixed<uint8_t>/CompactFixed` | 同上 |

### 编译存在性不等于 Route 可达性

从 `calc_unique_tiling → launch_unique → Launch*` 反向核对全部 enum、entry 与 helper 后，可达叶子严格只有 R-001..R-008；以下符号不产生额外 Route：

| 编译驻留符号 | 为什么不可达 | 不得据此推出的能力 |
|---|---|---|
| `ConfigureFloat32Radix`、`UNIQUE_ROUTE_RADIX_FLOAT32`、`LaunchFloat32Radix`、四个 `unique_radix_*` entry | 配置函数无调用点，selector 从不产生 route=3 | fp32 radix 不是 fallback，也不是 R-007/R-008 的子分支 |
| `unique_dense_meta → InitDenseMeta` | `LaunchDenseInteger` 直接从 Vector reducer 开始，无 launch 调用该 entry | 运行图没有独立 meta-init 节点 |
| `ReduceDenseRange<T>` | 没有任何 entry 或 helper 调用它；可达 extrema 是 `DenseRangeVectorReducer<T>` | 不能把 warp-shuffle/每 warp 两次 atomic 写进 R-005/R-006 的真实 reduce 路径 |
| `unique_{u8,i8,f16,bf16}_{noinv,inv}` official entries | 对应 dtype 在 Host 先命中 fixed route，且 fixed 无 Device fallback | fixed 域没有隐含 official 兜底 |
| `unique_{i32,i64}_{noinv,inv}` official entries | int dtype 先命中 dense；宽域由四个 `unique_pending_*` entry 内嵌 official body，不调用这些普通 entry | V2 fallback 的 route-reader/`TPipe::Reset` 不能省略 |
| official `Phase_transpose/Phase_transpose_back` | Direct POD 固定 `dim=0` | Direct 只实现 flatten，不公开任意 dim |
| official `Phase7_ComputeCounts`、`positions` 写入 | 模板第三参数固定 false，POD `returnCounts=0` | countsScratch 只是 ABI 占位，不是隐藏输出 |
| official `OutputEmptyShape` | Host 在 tiling 前拒绝 `N=0` | 不支持空输入 shortcut |

`launch_unique_official` 的 switch 具备七 dtype × 两种 inverse 的语法实例，不代表十四条运行 Route；在当前 selector 中只有 `unique_f32_noinv/unique_f32_inv` 从该 switch 可达。dense 宽域走的是另外四个 pending 实例。

## Kernel 实现

### 公共执行流程

Direct selector 先按 dtype 决定 launcher 图，再按 `returnInverse` 选择 noinv/inv 实例。每个 supplemental entry 标为 `KERNEL_TYPE_AIV_ONLY`，实际线程主体由 `asc_vf_call` 启动 SIMT；compact 节点另持有 1KiB VECCALC prefix TBuf。官方 entry 直接实例化 `UniqueDimKernel<T,RETURN_INVERSE,false>`，其 AIV carrier 在一个 Kernel 内完成多次 SIMT phase、UB sort、GM merge、scan/gather 与 shape MTE3。

跨 Host Kernel 的依赖由 same-stream 顺序保证；单 entry 内的 SIMT VF 用 `__syncthreads` 与前后 D-cache clean/invalidate，跨 AIV core phase 用 `SyncAll`。所有输出值均直接写 GM，不做浮点升降精度；排序比较与 key 变换决定数值边界。

### Route R-003（P0）：16-bit fixed/noinv

Route 特征为 D65536 固定域三节点图；适用 fp16/bf16、`returnInverse=false` 的无 NaN 输入域；实现前提是两张 256 KiB GM 表、1 KiB compact UB 与 AIV core 可用。Host 顺序为 H2a：H1 未命中后进入 fp16/bf16 分支并选 noinv；该 dtype 分支短路且无 Device fallback。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `unique_kernel.cpp` | `calc_unique_tiling` | fp16/bf16且noinv | D65536、R-003 |
| Launcher | 同上 | `LaunchFixedDomain` | D!=256 | mark16→compact |

#### Device 数据流

##### 函数级调用链

```text
clear(D65536) -> MarkFixed<uint16_t> -> CompactFixed<256>
```

##### 源码等价伪码

```text
key = (raw是+0/-0) ? 0 : raw
zero key: AtomicMin(first[0],i)以保留首个零的符号
其它key: 若sentinel则幂等写0（presence）
compact按 sign-aware total-order 扫65536编码；-0槽跳过，+0槽从x[first[0]]回取原符号
写y与rank并在shape[1]写唯一数
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| first/rank | 0/262144 | 各65536×4=256KiB | uint32 | clear/mark/compact | compact | workspace | 不别名 |
| prefix UB | 0 | 1024B | uint32 | compact | compact | Kernel期 | 同R-001 |
| x/y | i/outputIndex | N×2/≤65536×2 | uint16位模式 | caller/compact | mark/caller | 调用期 | zero回取first raw |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  first["GM first · 256 KiB"]:1
  rank["GM rank · 256 KiB"]:1
  prefix["compact UB · 1 KiB"]:1
```

##### 字节账与复用顺序

`UB_total(compact)=1024 B；route_used_workspace=524288 B；actual_allocation=max(524288,officialBytes,128)`。clear→mark→compact 同流；first[0] 生命周期覆盖首次零发布到 compact 回取。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| zero canonicalization | atomic/presence store | mark | compact | first | +0/-0相等并保首符号；任务域要求无NaN |
| total order | bit transform | compact | y/rank | 编码序 | 负数取反、非负异或sign bit |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | D65536/512KiB |
| Kernel entry/router | 同上 | `unique_fixed_mark16/compact` | 三节点 |
| Device body/helper | 同上 | `CanonicalFixedKey/RawFromOrder/MarkFixed` | zero与数值序 |

### Route R-008（P1）：fp32 official/inv

Route 特征为官方比较排序单 Kernel 并启用 inverse phase；Host 适用域是全部 fp32、`returnInverse=true`，但完整 unique/inverse 语义域须进一步限定为无 NaN。实现前提是完整 official workspace、256 B plain POD 与跨核同步协议。Host 顺序为 H4b：H1/H2/H3 均未命中后由合法 dtype 余项 fp32 进入 official；它不检查 NaN，radix body 也没有 selector producer，因此含 NaN 时没有语义 fallback。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Entry | `unique_kernel.cpp` | `launch_unique_official/unique_f32_inv` | fp32且inverse=true | `UniqueDimKernel<float,true,false>` |

#### Device 数据流

##### 函数级调用链

```text
R-007 phase0..4 -> Phase5_ScatterInverse -> SyncAll -> Phase6 gather -> Phase8
```

##### 源码等价伪码

```text
for sorted position i:
  original=indices[i]; group=flagsInclusive[i]-1; inverse[original]=group
随后执行R-007 compact/gather/shape
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| R-007资源 | 同R-007 | 同R-007 | 同左 | 同左 | 同左 | 同左 | 同左 |
| inverse | original index | N×8B | int64 | Phase5 | caller | 输出期 | 每original唯一写者 |

##### 分层内存布局

与 R-007 相同；inverse 在 GM，不增加 UB/workspace。

##### 字节账与复用顺序

`UB_total=8416 B；workspace_total=officialBytes`。Phase5 完成并 `SyncAll` 后 Phase6 才允许复用 flags/indices 作为 compact 输入。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| inverse | SIMT+SyncAll | prefix scan | Phase6 | inverse/flags | group uint32转int64；coreLen裁尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | inverse=true |
| Kernel entry/router | 同上 | `unique_f32_inv` | true,false模板 |
| Device body/helper | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `Phase5_ScatterInverse` | inverse phase |

### Route R-002（P2）：8-bit fixed/inv

Route 特征为 D256 固定域四节点图与 rank lookup；适用 uint8/int8、`returnInverse=true`；实现前提是两张 1 KiB GM 表及 same-stream producer→consumer 可见性。Host 顺序为 H1b：首个 dtype 分支命中后选 inv，立即短路且无 fallback。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin/Host | `unique_plugin.cpp`/`unique_kernel.cpp` | `unique_npu/calc_unique_tiling` | u8/i8且inverse=true | R-002 |
| Launcher | `unique_kernel.cpp` | `LaunchFixedDomain` | true | R-001三节点后加inverse8 |

#### Device 数据流

##### 函数级调用链

```text
R-001 chain -> unique_fixed_inverse8 -> InverseFixed<uint8_t><512>
```

##### 源码等价伪码

```text
执行R-001；随后 for i in block/thread interval: inverse[i]=int64(rank[key(x[i])])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| R-001资源 | 同R-001 | 同R-001 | 同左 | 同左 | 同左 | 同左 | 同左 |
| GM inverse | `i` | N×8B | int64 | inverse线程 | caller | 输出期 | 每i唯一写者 |

##### 分层内存布局

与 R-001 相同；inverse 节点无 UB，`route_used_workspace=2048 B`，实际分配仍为 `max(2048,officialBytes,128)`。

##### 字节账与复用顺序

rank 在 compact 写完后由 inverse 只读，same-stream 后释放；无物理复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| inverse | same-stream + thread loop | compact rank | inverse threads | rank/inverse | uint32 rank无损扩为int64；末核裁尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | returnInverse镜像 |
| Kernel entry/router | 同上 | `unique_fixed_inverse8` | 第四节点 |
| Device body/helper | 同上 | `InverseFixed<uint8_t>` | rank lookup |

### Route R-006（P3）：dense integer/inv

Route 特征为 extrema 预检、dense 表、inverse 与 pending official；适用 int32/int64、`returnInverse=true`；实现前提是 213024 B reduce UB、互不别名的 official/dense workspace 与 32 B route publish。Host 顺序为 H3b；Device 顺序为 V1 `max-min<2^20` 完成 dense 并令 pending 返回，V2 宽域使 dense helpers 短路后由 pending official 产生结果。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Device | `unique_kernel.cpp` | `calc_unique_tiling/FinalizeDenseRange` | i32/i64、inverse=true | dense或official分支 |
| Launcher | 同上 | `LaunchDenseInteger` | true | compact后加inverse，再pending |

#### Device 数据流

##### 函数级调用链

```text
R-005前五节点 -> InverseDense<T> -> pending_unique<T,true>
 -> [UniqueDimKernel<T,true,false> when wide]
```

##### 源码等价伪码

```text
valid range: for i inverse[i]=int64(rank[uint64(x[i])-uint64(min)])
wide range: dense inverse早退；pending official执行Phase5_ScatterInverse
```

这里的 key 与 R-005 相同，`uint64(value)-uint64(min)` 以无符号模运算表达有符号差；range 先经 `<2^20` 约束，所以 lookup 严格落在 rank 有效前缀。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| R-005全部资源 | 同R-005 | 同R-005 | 同左 | 同左 | 同左 | 同左 | 同左 |
| inverse GM | i | N×8B | int64 | dense或official inverse | caller | 输出期 | branch只一方有效写 |

##### 分层内存布局

同 R-005；inverse 节点不增 UB/workspace。

##### 字节账与复用顺序

`workspace_total` 与 R-005 相同。dense inverse 在 pending 之前；宽域时其 `DenseRangeValid` 失败即返回，pending official 后写正式 inverse，不存在双写。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| inverse branch | meta guard + same stream | compact/meta | dense/pending | inverse | rank uint32无损扩int64；core tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | inverse flag |
| Kernel entry/router | 同上 | `unique_dense_inverse_* / pending_*_inv` | branch专用节点 |
| Device body/helper | 同上及官方四文件 | `InverseDense/Phase5_ScatterInverse` | 两分支inverse |

### Route R-005（P4）：dense integer/noinv

Route 特征为 R-006 的 noinv 变体，裁掉 inverse 节点；适用 int32/int64、`returnInverse=false`；实现前提与 workspace 协议同 dense/inv，但正式 inverse 资源 inactive。Host 顺序为 H3a；V1/V2 value-domain 短路不变，V2 进入 noinv official fallback。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `unique_kernel.cpp` | `calc_unique_tiling` | i32/i64、noinv | dense graph + official capacity |
| Device selector | 同上 | `FinalizeDenseRange` | `uint64(max)-uint64(min)<2^20` | flag0 dense；否则flag1 official |
| Pending entry | 同上 | `DEFINE_PENDING_UNIQUE_ENTRY` | routeFlag | 0 return；1 Reset+official |

#### Device 数据流

##### 函数级调用链

```text
reduce<T>::Init/Process(DataCopyPad,ReduceMin,ReduceMax,MTE3 partial)
 -> finalize(MTE2 partial -> FinalizeDenseRange -> MTE3 route block)
 -> ClearDense -> MarkDense<T> -> CompactDense<T>
 -> pending_unique<T,false> -> [UniqueDimKernel<T,false,false> when wide]
```

##### 源码等价伪码

```text
每核分tile归约min/max，写32B partial
单核归并C条partial；valid=(max>=min && unsigned(max)-unsigned(min)<2^20)
routeFlag=!valid
if valid:
  clear first[0:range]=sentinel
  range==1只写first[0]，否则AtomicMin(first[x-min],i)
  compact按key升序写 y=T(uint64(min)+key)、rank[key]、shape[1]
pending读取routeFlag：valid直接return；wide则Reset pipe并执行官方比较排序
```

`uint64(value)-uint64(min)` 与 `uint64(min)+key` 都依赖无符号模运算表达有符号差/和，随后 cast 回 T；这样跨过 0 的小范围不触发有符号溢出。range 判定先要求 `max>=min`，再要求同一无符号差 `<2^20`，因此所有实际 key 都落在 `[0,range)`。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| official GM区 | base+POD offsets | officialBytes | uint32 slices | pending official | official phases | workspace | 与dense尾区隔离 |
| first/rank | denseOff / +4MiB | 各4MiB | uint32 | clear/mark/compact | compact | workspace | 仅前range有效 |
| partial GM | denseOff+8MiB | C×32B | 4×int64/core | reduce MTE3 | finalize MTE2 | 两节点 | 32B/core |
| meta GM | partial+C×32 | 16B | `{min,max}` | finalize VF | dense helpers | dense图 | routeFlag在shape槽2 |
| reduce inputQ | UB | 192KiB depth1 | T | DataCopyPad | Reduce | reduce Kernel | i32 49152/i64 24576元素 |
| reduce work/partial | UB | 16KiB +32B | T/int64 | Reduce/scalar | Reduce/MTE3 | reduce Kernel | events隔离 |
| finalize queues | UB | C×32B +32B | int64/u64 | MTE2/VF | VF/MTE3 | finalize Kernel | depth1 |
| compact prefix | UB | 1KiB | uint32 | threads | scan/emit | compact Kernel | depth1 TBuf |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  official["official GM region · officialBytes"]:1
  first["dense first · 4 MiB"]:1
  rank["dense rank · 4 MiB"]:1
  pm["partial 32C B + meta 16 B"]:1
  inq["reduce inputQ · 192 KiB"]:2
  work["reduce work · 16 KiB"]:1
  pl["partial UB · 32 B"]:1
```

##### 字节账与复用顺序

```text
reduce UB_total=192KiB+16KiB+32B=213024 B
finalize UB_total=32C+32 B
compact UB_total=1024 B
workspace_total=A(officialBytes)+8MiB+(4C+2)*8
dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| reduce workLocal | ReduceMin | scalar read→ReduceMax | 同tile max | V_S/S_V event + PipeBarrier | current逻辑长度不含pad |
| partial GM | reduce cores | finalize | 调用结束 | same stream + MTE3/MTE2 | 每核独占32B |
| shape route block | finalize | pending entry/Host最终D2H | official可能覆盖shape | same stream；pending先读 | 先完整32B publish |
| official/dense区 | 各自节点 | 各自consumer | 调用结束 | 物理不别名 | denseOff按128B对齐 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| reduce | TQue+V_S/S_V+S_MTE3 | DMA/Vector/scalar | next stage | UB/partial | int32/int64精确；tail DataCopyPad但Reduce长度=current |
| route | MTE2→SIMT→MTE3 | partial/finalize | pending MTE2 | 32B block | unsigned差避免有符号减溢出 |
| dense mark | AtomicMin | input threads | compact | first | range1跳过全量重读 |
| pending | TQue MTE2 + `TPipe::Reset` | route read | official InitBuffer | event/UB | 仅flag1 Reset后启动official |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | disjoint workspace公式 |
| Kernel entry/router | 同上 | `LaunchDenseInteger/DEFINE_PENDING_UNIQUE_ENTRY` | 七节点、Device branch |
| Device body/helper | 同上 | `DenseRangeVectorReducer/FinalizeDenseRange/ClearDense/MarkDense/CompactDense` | extrema、表与fallback |
| Official fallback | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `UniqueDimKernel<T,false,false>` | 宽域比较排序 |

### Route R-004（P5）：16-bit fixed/inv

Route 特征为 D65536 固定域四节点图与 canonical rank lookup；适用 fp16/bf16、`returnInverse=true` 的无 NaN 输入域；实现前提为 R-003 的固定域资源加 inverse GM。Host 顺序为 H2b：fp16/bf16 分支命中后选 inv，dtype 分支短路且无 fallback。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Launcher | `unique_kernel.cpp` | `calc_unique_tiling/LaunchFixedDomain` | fp16/bf16且inverse | R-003后加inverse16 |

#### Device 数据流

##### 函数级调用链

```text
R-003 chain -> unique_fixed_inverse16 -> InverseFixed<uint16_t>
```

##### 源码等价伪码

```text
for i: inverse[i]=int64(rank[CanonicalFixedKey(raw16(x[i]))])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| R-003资源 | 同R-003 | 同R-003 | 同左 | 同左 | 同左 | 同左 | 同左 |
| inverse | i | N×8B | int64 | inverse16 | caller | 输出期 | 每i唯一写者 |

##### 分层内存布局

同 R-003；inverse 节点无 UB。

##### 字节账与复用顺序

`route_used_workspace=524288 B；actual_allocation=max(524288,officialBytes,128)`；rank 从 compact 到 inverse 只读，same-stream 保证可见。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| inverse | canonical key lookup | compact | inverse | rank | +0/-0映到同rank；core tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | inverse=true |
| Kernel entry/router | 同上 | `unique_fixed_inverse16` | 第四节点 |
| Device body/helper | 同上 | `InverseFixed<uint16_t>` | canonical rank |

### Route R-007（P6）：fp32 official/noinv

Route 特征为官方比较排序单 Kernel并编译期裁掉 inverse；Host 适用域是全部 fp32、`returnInverse=false`，但完整 unique 语义域须进一步限定为无 NaN。实现前提为 official workspace、plain POD 与跨核同步。Host 顺序为 H4a：前三个 dtype 分支均未命中后进入 official；它不检查 NaN，radix 不可达且无下一 Route。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `unique_kernel.cpp` | `calc_unique_tiling` | dtype fp32落入else | OFFICIAL_SORT |
| Launcher/Entry | 同上 | `launch_unique_official/unique_f32_noinv` | inverse=false | `UniqueDimKernel<float,false,false>` |

#### Device 数据流

##### 函数级调用链

```text
unique_f32_noinv -> UniqueDimKernel<float,false,false>::Init/Process
 -> Phase0 InitIndices -> Phase1 UB tile sort + per-core GM merge
 -> Phase2 all-core merge-path -> Phase3 AdjacentDiff
 -> Phase4 local/global prefix -> Phase6 detect+gather -> Phase8 shape
```

##### 源码等价伪码

```text
每核初始化indices；每512项在UB双半区归并排序，尾lane以-1 sentinel排末
每核多tile在indices/sortBuf间ping-pong归并
全核以width=perCore逐轮merge-path，奇偶轮交换src/dst并处理未配对tail
相邻已排序值比较：不同或任一NaN则flag=1
局部inclusive scan；core0扫每核和；各核加global prefix
每个新组写首个原索引，按输出元素域gather x到y；末核MTE3写shape
```

排序稳定性来自三处同向约束：两元素网络只在 `cmp>0` 时交换；UB merge 与 GM merge 在比较相等时取左段；初始 key 是递增原索引。因此普通相等类的 `compactKeys[group]` 是最早输入位置，gather 保留首个值的位模式。Phase0 的显式 identity 初始化仍不能删除：`coreLen<=1` 时 Phase1 直接返回，只有 Phase0 为该分片提供有效索引；`coreLen>1` 时 Phase1 会重新构造 identity，避免 tile sort 依赖跨 VF_CALL 的旧 GM cache。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices/sortBuf/flags | POD offsets | 各N×4B，起点128B对齐 | uint32 | phases0/1/3 | sort/scan/gather | official区 | sort两表ping-pong |
| partial/globalPrefix | POD offsets | C×128B / (C+1)×4B | uint32 | local/global scan | add/gather | official区 | partial按页stride32 |
| ubBuf | 0与4096 | `2*A(512*8,32)=8192B` | int64 index | tile sort | tile merge | entry期 | 双半区交替读写 |
| shapeBuf | 独立TBuf | `A(27*8,32)=224B` | uint64 | Phase8 | MTE3 | entry期 | 不与sort UB别名 |
| x/y/shape | value index/输出/槽 | N×4/numOut×4/216B写 | fp32/u64 | caller/gather | compare/caller | 调用期 | gather 2048 threads |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  ubA["ubBuf A · 4096 B"]:1
  ubB["ubBuf B · 4096 B"]:1
  shape["shapeBuf · 224 B"]:1
  idx["GM indices · 4N"]:1
  sort["GM sortBuf · 4N"]:1
  scan["GM flags/partial/prefix"]:1
```

##### 字节账与复用顺序

```text
UB_total=8192+224=8416 B
workspace_total=officialBytes=transposeOff
dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ubBuf halves | tile items/merge round | next merge round | 下一round/tile | `__syncthreads` | readFirst翻转后才覆盖 |
| indices/sortBuf | 当前merge round | 下一round | 交替dst | `PipeBarrier+SyncAll` | 所有核完成pair |
| sortBuf compactKeys | Phase6 detect | gather | Kernel结束 | `SyncAll` | merge已结束且indices固定 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile sort | `__syncthreads`+dcci | UB rounds | next round/GM | UB/GM | sentinel -1排末，tileLen裁尾 |
| cross merge | `PipeBarrier<PIPE_ALL>+SyncAll` | pair merge | next pair/round | indices/sort | 未配对tail显式copy |
| scan/gather | `SyncAll` | flags/prefix/keys | next phase | GM workspace | uint32 group≤N；无NaN时完整；NaN可能隔断普通等值类 |
| shape | V_S + S_MTE3 | scalar/vector shape | Host D2H | shapeScratch | 只槽1被Host读取 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | plain POD、official区 |
| Kernel entry/router | 同上 | `unique_f32_noinv` | 模板false,false |
| Device body/helper | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `UniqueDimKernel::Process` | phase顺序/UB/同步 |
| Device body/helper | `.../unique_dim_simt_sort.h`、`.../unique_dim_simt.h` | sort/scan/gather helpers | SIMT循环/tail/NaN |

### Route R-001（P7）：8-bit fixed/noinv

Route 特征为 D256 固定域三节点图并裁掉 inverse；适用 uint8/int8、`returnInverse=false`；实现前提为 fixed8 两表与 same-stream 顺序，compact 仍产生 rank 表。Host 顺序为 H1a：首个 dtype 分支命中后选 noinv，立即短路且无 fallback。

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | `unique_plugin.cpp` | `CheckInput/DTypeCode` | u8或i8，inverse=false | Direct selector |
| Host | `unique_kernel.cpp` | `calc_unique_tiling` | dtype code 1/2 | D=256、fixed |
| Launcher | 同上 | `LaunchFixedDomain` | `returnInverse=false` | clear→mark8→compact |

#### Device 数据流

##### 函数级调用链

```text
launch_unique -> LaunchFixedDomain
 -> unique_clear_u32 -> ClearU32<512>
 -> unique_fixed_mark8 -> MarkFixed<uint8_t><512>
 -> unique_fixed_compact -> CompactFixed<256>
```

##### 源码等价伪码

```text
parallel clear first[0:256]=UINT32_MAX
for i in block interval: raw = uint8 bit pattern of x[i]
  markKey=raw                              # u8/i8 都不做 ^0x80
  if i < first[markKey]: AtomicMin(first[markKey],i)
single block scans numeric order:
  raw = order                              # uint8
  raw = order ^ 0x80                       # int8，仅 compact 顺序变换
  present = first[raw] != UINT32_MAX
thread0 exclusive-scans 256 local counts and writes shape[1]
each thread emits present raw bit pattern and writes rank[raw]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `first` | base | 256×4=1024B | uint32 key table | clear/mark | compact | workspace，图全程 | sentinel→first index |
| GM `rank` | base+1024 | 1024B | uint32 key table | compact | noinv不读 | workspace | 仍写，为图统一 |
| UB `prefix` | 0 | 256×4=1024B | uint32 local | compact threads | thread0/threads | compact Kernel | 无alias |
| GM x/y/shape | i/key/槽1 | N×1、≤256×1、8B有效 | byte/u64 | caller/compact | mark/caller | 调用期 | 位模式 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  first["GM first · 1024 B"]:1
  rank["GM rank · 1024 B"]:1
  prefix["compact UB prefix · 1024 B"]:1
```

##### 字节账与复用顺序

`UB_total(clear/mark)=0；UB_total(compact)=1024 B；dynamic_ub=N/A；route_used_workspace=2048 B；actual_allocation=max(2048,officialBytes,128)`。same-stream 使 clear 完成后 mark 覆盖 sentinel、compact 再读取；rank 不与 first 别名。

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| first | clear | mark/compact | 下次调用 | same stream | 不跨调用复用 |
| prefix | each compact thread | thread0 scan/each emit | Kernel结束 | `__syncthreads` | 256 threads恰好一槽/线程 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mark | `AtomicMin` | 并发输入线程 | compact下个Kernel | first[key] | 保存首次索引；block/thread loop裁尾 |
| compact | two `__syncthreads` | count/scan | emit | prefix/shape | i8 用 order→raw 异或0x80；不做算术cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `unique_kernel.cpp` | `calc_unique_tiling` | D256、2KiB、C核 |
| Kernel entry/router | 同上 | `LaunchFixedDomain/unique_*8` | 三节点顺序 |
| Device body/helper | 同上 | `MarkFixed<uint8_t>/CompactFixed` | first/rank表与数值序 |

## 算子特定优化

### Trick T-001：512 项 UB 双半区排序后再做 GM merge-path

#### 触发与源码依据

- 适用 Route：P1/R-008、P3/R-006 与 P4/R-005 的宽域 official 分支、P6/R-007。
- 触发条件：进入字节一致的 `UniqueDimKernel<T,RETURN_INVERSE,false>`；`tileSize=512`。
- 固定源码：[`index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h:188-214`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h#L188-L214)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t items[ITEMS_PER_THREAD];
for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
    int64_t idx = tid * ITEMS_PER_THREAD + j;
    items[j] = (idx < tileLen) ? (tileGlobalStart + idx) : -1;
}
```

#### 机制与设计原因

256 个 SIMT 线程各持两个索引，先在寄存器比较，再把 512 项写入 UB；后续归并轮在两个 4096 B 半区间 ping-pong。只有完成 tile 局部排序后才进入 `indices/sortBuf` GM 归并，避免每个比较步骤都物化到 GM。`-1` sentinel 由行比较器排到真实索引之后，tail 无须另建短模板。

#### 迁移边界

- 适用：比较 key 可由原索引回查，且 512×int64×2 加 shape buffer 可放入 UB。
- 前提：256 threads、每线程2项、双半区与每轮 `__syncthreads` 不变。
- 失效：UB 不足或不允许跨核 `SyncAll` 时需重新 tiling/拆 Kernel；fixed/dense有效分支不使用此 trick。

#### 错误做法

固定源码未附独立反例；但实现明确依赖两半区交替与 barrier，不能把它误改成单缓冲原地 merge。

### Trick T-002：跨核归并以 indices/sortBuf 轮换并显式处理奇尾

#### 触发与源码依据

- 适用 Route：同 T-001。
- 触发条件：`elementsPerCore < N`，存在跨核 merge round。
- 固定源码：[`index/unique_dim/op_kernel/arch35/unique_dim_kernel.h:340-368`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_kernel.h#L340-L368)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__gm__ uint32_t *src = (round % 2 == 0) ?
    (__gm__ uint32_t *)indicesGm_.GetPhyAddr() :
    (__gm__ uint32_t *)sortBufGm_.GetPhyAddr();
__gm__ uint32_t *dst = (round % 2 == 0) ?
    (__gm__ uint32_t *)sortBufGm_.GetPhyAddr() :
    (__gm__ uint32_t *)indicesGm_.GetPhyAddr();
```

#### 机制与设计原因

每轮宽度翻倍，所有 core×256 threads 用 merge-path 分割同一 pair 的输出区间；奇偶轮交换两张等长 GM 索引表，省去每轮第三张中间表。未配对尾段显式从 src copy 到 dst，最终奇数轮再 copy back，保证 Phase3 总从 indices 读取。`PipeBarrier<PIPE_ALL>` 与 `SyncAll` 是下一 pair/round 可见性的必要部分。

#### 迁移边界

- 适用：全局稳定排序、两张等长索引表可用、平台支持跨核同步。
- 前提：src/dst 不别名；每个 pair 后 barrier；tail copy 与最终奇轮 copy-back 保留。
- 失效：无 gang-style 全核同步时应拆成多个 Host Kernel，不得只删除 `SyncAll`。

#### 错误做法

固定仓未提供独立错误版本；源码中的 tail copy 和 odd-round copy-back 本身证明“只合并完整 pair”会丢失或读取旧尾段。

### Trick T-003：排序比较与去重比较分离 NaN 语义

#### 触发与源码依据

- 适用 Route：P1/R-008、P6/R-007 的 fp32，及 dense 宽域 official fallback 的通用模板语义。
- 触发条件：相邻已排序行执行 `SimtRowCompareDedup`。
- 固定源码：[`index/unique_dim/op_kernel/arch35/unique_dim_simt.h:110-135`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt.h#L110-L135)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (cmpResult != 0) {
    return cmpResult;
}
if (SimtIsNan<T>(valA) || SimtIsNan<T>(valB)) {
    return 1;
}
```

#### 机制与设计原因

排序阶段的 FP32 普通比较把 NaN 与任意值都视为 compare-equal，稳定 merge 因而可以保留原相对位置；相邻去重阶段则在普通比较未分出大小后检测位模式，只要一方是 NaN 就判为不同。这个分离确实令 NaN 本身不被合并，但不能保证所有普通等值元素在排序后相邻：`[1, NaN, 1]` 可能保持原序，两个 1 被 NaN 隔开后都成为新组。故该机制是含 NaN 时的语义缺口，不是完整 unique 的可复用技巧；FP32 的完整保证须限定到无 NaN 域。

#### 迁移边界

- 适用：仅用于审计 official comparator/dedup 的差异；不能作为含 NaN 的完整 unique 方案。
- 前提：R-007/R-008 若要保持完整 unique，输入域必须无 NaN。
- 失效：若 NaN 属合法输入域，应建立能把普通等值类排到连续区间的 total-order/bucket 规则，再按目标 NaN 合并语义调整 dedup；只改 dedup 不足以修复被 NaN 隔开的普通重复值。

#### 错误做法

不能把“排序 compare-equal、dedup 判不等”当成完整 NaN 处理；该组合会破坏“所有相等普通值排序后连续”这一相邻去重前提。

## 模板与实现对比

| 优先级/Route | Host 顺序 | Route 特征 | 适用域 | 实现前提与资源 | 搬运/计算/同步 | tail 与 fallback |
|---|---|---|---|---|---|---|
| P0/R-003 | H2a | D65536、noinv、三 Kernel | fp16/bf16，无NaN域 | fixed16；compact 1KiB UB；使用512KiB，实际分配取 officialBytes 最大值 | SIMT GM、零类 atomic、presence publish、same-stream | core/domain/span tail；无fallback |
| P1/R-008 | H4b | official compare、inv、单 Kernel | fp32；完整语义需无NaN | `UniqueDimKernel<float,true,false>`；8416B UB；officialBytes | 512 tile、GM merge-path、scan、inverse、gather、`SyncAll` | 含NaN可能重复普通值且无fallback；radix不可达 |
| P2/R-002 | H1b | D256、inv、四 Kernel | uint8/int8 | fixed8；compact 1KiB UB；使用2KiB，实际分配取 officialBytes 最大值 | SIMT GM、`AtomicMin`、rank lookup、same-stream | core/domain tail；无fallback |
| P3/R-006 | H3b+V1/V2 | extrema+稠密表+inv+pending | int32/int64 | 213024B reduce UB；official区+8MiB+32C+16B | `DataCopyPad/ReduceMin/ReduceMax`、MTE3/MTE2、dense SIMT、pipe Reset | value-domain宽时 pending official；两分支均有core/tile tail |
| P4/R-005 | H3a+V1/V2 | 与R-006同图但裁掉inverse | int32/int64 | 同R-006，正式inverse资源inactive | 同R-006但无inverse节点/phase | value-domain宽时 noinv official |
| P5/R-004 | H2b | D65536、inv、四 Kernel | fp16/bf16，无NaN域 | R-003资源 + inverse GM | R-003主体 + canonical rank lookup | core/domain tail；无fallback |
| P6/R-007 | H4a | official compare、noinv、单 Kernel | fp32；完整语义需无NaN | `UniqueDimKernel<float,false,false>`；8416B UB；officialBytes | 与R-008共用sort/scan/gather，编译期裁掉inverse | 含NaN可能重复普通值且无fallback；radix不可达 |
| P7/R-001 | H1a | D256、noinv、三 Kernel | uint8/int8 | R-002 fixed8资源，rank仍生成 | clear/mark/compact，裁掉inverse节点 | core/domain tail；无fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本卡不覆盖，未提取。 | 本卡不覆盖，未提取。 | Direct 交付固定 Ascend 950/CANN 9.1.0；官方 arch35 四文件只作字节一致实现锚点。 |
| Host/Launcher | 未提取 | 未提取 | Platform manager 查询 AIV core；当前 NPU stream 直调多/单 Kernel。 |
| Kernel/资源 | 未提取 | 未提取 | AIV carrier、SIMT VF、AIV Vector reduce；无 Cube/AIC/MIX_CV；workspace 公式见各 Route。 |

## 限制与失败边界

- Direct 语义是全量 flatten unique，不是官方带 dim/counts 的 `UniqueDim`；不得把接口、输出 arity 或 shape 协议互换。
- 输入必须 contiguous、rank 1–8 且非空；不提供 framework Contiguous 或空输入 shortcut。
- 仅七种 dtype；官方源中其它 dtype 能力不自动成为 Direct 能力。
- `N<=UINT32_MAX` 是硬约束，所有索引/rank/scan count 都以 uint32 表示；输出 `inverse` 才扩为 int64。
- FP16/BF16 fixed 派生路线假定无 NaN 的任务域；NaN bit patterns 会按编码 presence 处理，不具备官方 dedup comparator 的“NaN各自唯一”语义。
- FP32 R-007/R-008 的 Host selector 接受 NaN，但 sort comparator 把 NaN 与任意值视为相等，dedup comparator 又把含 NaN 对视为不等；`[1,NaN,1]` 可能产生重复 1，故含 NaN 是 candidate 语义缺口而非受保证域。
- `first[key]` 的非零 16-bit key 用 guarded普通写作为 presence publish；该协议只依赖幂等值0，不能改成保存首次索引后再据此读输入。
- dense extrema 固定分配约 208KiB UB，但 Host 只查询 core 数，未查询/检查 UB 上限；迁移到不同 950 配置时需重新闭合。
- dense 宽域仍提交若干会早退的节点，真正结果由最后 pending official entry 产生；不得把前序 compact 的旧 shape 当最终 shape。
- pending entry 在 route TQue 后必须 `TPipe::Reset()` 才能初始化官方 queues；删除该边界会复用事件/UB 状态。
- fixed Route 逻辑上只消费 2 KiB/512 KiB，但插件统一按 `max(config.workspaceBytes,officialWorkspaceBytes,128)` 分配；大 N 时实际分配由 official fallback 公式主导，不能用表大小估算 allocator 压力。
- `ConfigureFloat32Radix` 与 radix entry/launcher 存在但 Host 不可达；不能依据 enum 或 body 存在发布 radix Route。
- wrapper 没有逐 Kernel launch 返回检查；错误在最终 D2H/stream synchronize 边界传播。
- 临时 Tensor 的异步生命周期依赖 torch_npu allocator；源码没有独立 event/recordStream 证明。
- fixed/dense selector 与 wrapper 是 Direct 派生补充；固定官方 Commit 只证明四个 UniqueDim 文件的相关 body，不能证明派生 Route 为官方实现。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| Python 包装入口 | `cann_bench/__init__.py` | `unique` | false 时返回单 Tensor，true 时把 dispatcher `Tensor[]` 转成二元 tuple；派生事实源。 |
| 入口声明/约束 | `csrc/ops/unique/op_plugin/unique_plugin.cpp` | schema/`CheckInput/unique_npu` | Direct 参数、输出、D2H、stream；派生事实源。 |
| Host/Tiling/Launcher | `csrc/ops/unique/op_kernel/unique_launch.h` | `UniqueLaunchConfig/launch_unique` | Host config 与入口；派生事实源。 |
| POD/ABI | `csrc/ops/unique/op_kernel/unique_tilingdata.h` | `UniqueDimTilingData` | 256B plain POD；派生自官方字段像。 |
| Host/Tiling/Launcher | `csrc/ops/unique/op_kernel/unique_kernel.cpp` | `calc_unique_tiling/Launch*` | 全 Route、workspace、图与派生 body。 |
| Kernel body | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `UniqueDimKernel` | 官方 phase、UB、workspace consumer、同步。 |
| Kernel helper | `index/unique_dim/op_kernel/arch35/unique_dim_constant.h` | shape/queue常量 | 27槽 shape 与固定常量。 |
| Kernel helper | `index/unique_dim/op_kernel/arch35/unique_dim_simt.h` | compare/scan/gather | SIMT 比较、NaN、prefix、inverse、gather。 |
| Kernel helper | `index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h` | tile sort/merge-path | 双半区排序与 GM merge。 |
| 构建接线 | `csrc/ops/unique/CMakeLists.txt` | `register_direct_launch_op` | kernel/plugin Direct 编译注册；派生事实源。 |

Frontmatter `sources` 只列固定 Commit 下转换实际参考且可公开定位的四个官方文件。表中 `csrc/ops/unique/*` 是 source mapping 所指的 Direct 派生候选相对路径，不对应官方 URL，也不把它们伪装为官方来源。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 32×int64、256B、align8 | 字段顺序/offset；counts/dim inactive值；by-value传参 | `UniqueDimTilingData/static_assert` |
| 调用顺序 | fixed noinv/inv 为 3/4 节点；dense noinv/inv 为 6/7 节点；official 为单节点 | same-stream；dense pending 永远最后；最终 D2H 之后才 resize | `LaunchFixedDomain/LaunchDenseInteger/unique_npu` |
| workspace/内存 | fixed两表；dense official+独立尾区；official三索引表与scan | 128B offset、depth、C×32 partial、两区不alias | `calc_unique_tiling/InitBuffer` |
| 同步 | stream、SIMT barrier/dcci、Vector events、MTE3/MTE2、SyncAll/pipe Reset | route block完整32B；pending Reset；merge每pair barrier | `FinalizeDenseRange/DEFINE_PENDING/UniqueDimKernel` |
| 精度与尾块 | signed key顺序、zero canonical、unsigned dense delta、official comparator/dedup 差异 | fp16/bf16 与 fp32 完整语义均需无NaN；current而非aligned参与Reduce；所有core/tile tail | `CanonicalFixedKey/DenseRangeValid/SimtRowCompareDedup` |
| 失败边界 | 非法输入Host失败；dense宽域Device fallback；radix dormant | 不扩dtype/rank/空域；不把body存在当可达 | `CheckInput/calc_unique_tiling` |

- [AtomicMin](../../../../apis/simd_api/basic_api/atomic_operations/atomic_min.md)
