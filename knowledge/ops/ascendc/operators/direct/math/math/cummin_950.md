---
type: Operator
title: "Cummin 950 Direct Invoke 算子知识"
description: "说明 cann_bench::cummin 在 950 上的语义、Host checked arithmetic、Direct Launcher、可达 Route、Kernel 数据流及源码边界。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cummin/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cummin/op_host/cummin_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cummin/op_host/arch35/cummin_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cummin/op_kernel/arch35/cummin.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cummin/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T19:31:28Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Cummin 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 契约 |
|---|---|
| Direct 入口 | `cann_bench::cummin(Tensor input, int dim) -> (values, indices)`，由 `PrivateUse1` 实现直接取得当前 NPU stream 并调用 dtype launcher |
| 语义 | 把连续 ND 输入折叠为 `[M,R,N]`，固定 `(m,n)` 沿 `R` 做 inclusive cumulative minimum；`indices` 保存扫描轴内索引 |
| shape/layout | `rank>=1`，`dim∈[-rank,rank-1]`；当前 Direct 实现只接受 contiguous ND，不隐式复制非连续输入 |
| dtype | 输入与 `values`：`float16/float32/bfloat16/int32`；`indices`：`int64` |
| 可达 key | Host 可生成 `0/1/2/3/5/6/8`；key `4/7` 只有 entry/body，没有可达 selector，不能作为当前 Route |
| 执行单元 | 所有 entry 都声明 `KERNEL_TYPE_AIV_ONLY`；key `0/1/2/8` 在 AIV 上使用 GM↔UB 与 MicroAPI，key `3/5/6` 从 AIV wrapper 进入 SIMT；无 Cube、L1、L0 路径 |
| workspace | 独立 workspace 为 0；key 6 的三个节点用最终输出 GM 作为中间交接 |
| 权威边界 | Direct 构建只编译 `op_kernel/cummin_kernel.cpp`。源码树中携带的 `cummin_tiling_official.cpp` 未接入该 CMake target，不能代替 Direct `HostTiling` 判断运行时 Route |

## 语义、shape 与数值规则

### `[M,R,N]` 归约视图

标准化 `axis=(dim+rank)%rank` 后：

```text
M = product(shape[0:axis])
R = shape[axis]
N = product(shape[axis+1:rank])
p(m,r,n) = (m*R+r)*N+n
```

对每个 `(m,n)`，Device 写出：

```text
state = (input[p(m,0,n)], 0)
for r in 0..R-1:
    state = combine(state, (input[p(m,r,n)], r))
    values[p(m,r,n)]  = state.value
    indices[p(m,r,n)] = state.index
```

当前各 body 的普通数比较都以严格 `<` 判断左状态是否保留，因此非 NaN 的相等值选择右状态，即相等最小值取最新索引。浮点 NaN 规则不是全 Route 不变量：key `5/6/8` 显式维护 sticky NaN 且后出现的 NaN 更新索引，key `0/1/2/3` 没有这层状态，不能把两组路线的 NaN 行为视为等价。

### Host 接口契约

| 对象 | shape/dtype | Host 行为 |
|---|---|---|
| `input` | contiguous ND；四种已列 dtype | 非连续或其它 dtype 直接报错 |
| `dim` | `int64` 属性 | Meta 与 Tiling 均检查范围，负轴取模归一化 |
| `values` | 同 shape、dtype、device | `empty_like(input)` |
| `indices` | 同 shape、device，`int64` | `empty(input.sizes(), dtype=int64)` |
| stream | 当前 NPU stream | 三节点 Route 也在同一个 stream 依次提交 |

### checked arithmetic 与空维边界

Direct 源码使用有符号 `int64_t` 直接做 shape 乘积和地址乘法，且没有空 Tensor 快路径。复用这一设计时，Host 必须在 Tiling 前补齐下列证明；这些是实现契约，不是当前代码已经完成的检查：

```text
checked_product(prefix_dims) -> M
checked_product(suffix_dims) -> N
checked_mul(R, N)            -> RN
checked_mul(M, RN)           -> numel
checked_ceil_align(N, ceil(32/d)) -> alignedN
checked_mul(alignedN, R)
checked_mul(M, ceil_div(R,1024))  # segmented taskCount
```

- 每个 shape 维度应为正；若框架允许零维长度，先在 Host 返回两个空输出，不能进入当前 selector 的除法/取模。
- `d∈{2,4}`、`argminDSize=8`、`coreNum>0`、原始 UB 字节数大于 4096。
- 在构造 `DataCopyExtParams` 前验证 `copyRows` 可表示为 `uint16_t`，`dataNum*d`、GM/UB stride 字节数可表示为 `uint32_t`。
- 在写 `blockDim` 前验证核数可表示为 `uint32_t`；地址 offset 的所有乘加都应 checked。
- `ceil_div(x,y)` 与 `ceil_align(x,a)` 必须先证明 `y>0`、`a>0` 且 `x+a-1` 不溢出。

## Host Tiling 与实际 selector

### 资源量

令 `C` 为 AIV 核数，`UBraw` 为平台查询的 UB 字节数，`d=sizeof(input dtype)`：

```text
P = ceil(32/d)                         # 每个 32B block 的元素数
A = ceil_align(N,P)
U = floor_to_32((UBraw-4096)/(1+8/d)) # Host 中的 ubSize，单位为字节
q = (U/d-A)/(A*R)
rowsPerCore = M/C
reservedRows = M%C
```

`U` 是把 value 与 `int64` index 比例折算后的选择预算，不是某个单独 Queue 的容量。Kernel 的真实 UB 分配仍须按各 Queue 字节和对齐独立核账。

### 基础 key 与覆盖顺序

```text
if q >= 1:
    base = key0                       # RN_FULLY_LOAD
    computeM = q
else:
    L = ceil_align(N*d,32)
    ubLines = (U-L)/L
    if ubLines >= 2:
        base = key1                   # N_FULLY_LOAD, R 分批
        computeR = ubLines
    else:
        base = key2                   # N_NOT_FULLY_LOAD, N 分片
        computeN = floor_to_256(U/2)/d

key = key3 if (N<128 && M>12800 && R<12800) else base

if N>=4096 && R<=32:
    key = key8
else if N==1 && M>=256:
    key = key5
else if N==1 && R>=8192:
    key = key6
```

key 8、5、6 是覆盖，不是与基础 key 并列的独立初筛。特别地，`N==1,M>=256,R>=8192` 仍走 key 5；key 6 的实际可达域因此是 `N==1,M<256,R>=8192`。`localMemorySize=U` 在基础/key3 选择之后、特殊覆盖之前写入 config，但 launch 语句的动态内存参数是 `nullptr`。

### Route 重要性与退化关系

重要性按“可复用覆盖面、实现闭合度、前置证明强度”排列；它不改变上述 selector 的源码顺序。

| 重要性 | Route | key | 适用特征与价值 | 退化/边界 |
|---:|---|---:|---|---|
| P0 | R-001 自适应 AIV 主干 | 0/1/2 | 以 UB 容量在 M 批、R 批、N 片之间切换，覆盖普通连续 `[M,R,N]` | key0 容不下则 key1/key2；必须修复无剩余 M 行时的 POD 消费边界 |
| P1 | R-002 大 M、小 N lane-SIMT | 3/default | 每线程独占 `(m,n)`，无需 UB，R 串行；结构简单且分核直接 | 特殊覆盖命中时让位；普通 body 没有 sticky NaN |
| P2 | R-003 `N==1` warp-row | 5 | 一个 warp 扫一行，以 shuffle 降低每元素控制开销；包含显式 NaN 状态 | `M<256` 时让位给 key6 或基础/key3；R 尾块需 active-lane mask |
| P3 | R-004 少行超长 R segmented-row | 6 | 把单行拆成 1024 元素 segment，以三节点合并跨段 carry | 仅 `N==1,M<256,R>=8192` 可达；依赖同 stream 和输出 GM 生命周期 |
| P4 | R-005 NaN-aware AIV | 8 | 在 AIV R-batch 主干上增加 prior/current NaN mask | 源码谓词不足以保证 `ComputeR` POD 有效；只有额外证明基础路径本来是 key1 时才可采用 |

### 完整合法域可达性闭包

接口检查后的输入还必须满足前述 `M,R,N>0` 与 checked arithmetic，当前 selector 才是全定义函数；零长度 shape 虽未被 Meta 拒绝，却会在对齐、除法或 `blockDim` 生成阶段失去有效 launch，这是 Host 合法域缺口，不对应另一条 Route。对正长度且算术可表示的输入，基础判定必在 key `0/1/2` 中三选一；`S3=(N<128 && M>12800 && R<12800)` 可覆盖为 key3，随后 `S8=(N>=4096 && R<=32)`、`S5=(!S8 && N==1 && M>=256)`、`S6=(!S8 && !S5 && N==1 && R>=8192)` 依次覆盖或保留前值。因此生产 selector 的完整可达集合严格为 `{0,1,2,3,5,6,8}`。

| Route/key | 精确保留条件 | active POD | entry → body | block/core 映射 | selector fallback/覆盖 |
|---|---|---|---|---|---|
| R-001a / 0 | `q>=1 && !S3 && !S8 && !S5 && !S6` | `M/R/N,mRowsPerCore,splitM/computeM/ReservedM`；有剩余行时再用分组字段 | `cummin_k0_<dtype>` → `Init; ComputeM; ComputeReservedM` | `blockDim=(rowsPerCore>0?C:Cr)`；主段一核一段连续 M 行，余行按 G 核切 N | `q<1` 退到 key1/2；S3 或特殊 Route 可覆盖 |
| R-001b / 1 | `q<1 && ubLines>=2 && !S3 && !S8 && !S5 && !S6` | `M/R/N,mRowsPerCore` 与 `generalProcessInfo` 的 R-split 字段；有剩余行时再用 former/tail info | `cummin_k1_<dtype>` → `Init; ComputeR; ComputeReservedM` | 同 key0 | `ubLines<2` 退到 key2；S3 或特殊 Route 可覆盖 |
| R-001c / 2 | `q<1 && ubLines<2 && !S3 && !S8 && !S5 && !S6` | `M/R/N,mRowsPerCore` 与 `generalProcessInfo` 的 N-split 字段；有剩余行时再用 former/tail info | `cummin_k2_<dtype>` → `Init; ComputeN; ComputeReservedM` | 同 key0 | S3 或特殊 Route 可覆盖；无更低基础 key |
| R-002 / 3 | `S3 && !S8 && !S5 && !S6` | `M/R/N`；基础 split 字段虽已生产但 body 不读 | `cummin_k3_<dtype>` → `InitForSimT; ComputeSimT` → `SimtCompute` | `blockDim=min(M,C)`；每 block 连续 M 片，2048 线程 grid-stride `(m,n)` | S8/S5/S6 可覆盖；router 的 unknown-key default 也进入本 body |
| R-003 / 5 | `S5` | `M/R`（selector 已证明 `N=1`） | `cummin_k5_<dtype>` → `RunWarpRowScan` → `WarpRowScan/WarpScanChunk` | `blockDim=C`；每核 `min(ceil(M/C),64)` 个 warp，warp grid-stride row | S8 优先但与 `N=1` 互斥；未命中 S5 才检查 S6 |
| R-004 / 6 | `S6`，等价于 `N=1,M<256,R>=8192` | `M/R` | `cummin_k6_scan/summary/apply_<dtype>` → 三个 `RunSegment*` body | scan/apply 为 C blocks；summary 为 `min(M,C)` blocks；scan/summary 每核至多 64 warp，apply 每核 2048 线程 | S8/S5 均未命中才可达；条件 false 保留 key0/1/2/3 |
| R-005 / 8 | `S8` | Device 读取 `M/R/N,mRowsPerCore,generalProcessInfo` 与余行字段；只有基础 key1 时 R-split 字段有效 | `cummin_k8_<dtype>` → `Cummin<X,int64,true>::Init; ComputeR; ComputeReservedM` | `blockDim=C`，主段/余行映射同 key1 | 最高覆盖、无后续 Route；若基础不是 key1，不存在安全 Device fallback |

每条 active Route 的内存、tail 与精度闭包如下；详细循环伪码仍以各 Route 章节为准：

| key | GM/UB/寄存器 | tail | 精度与索引 |
|---:|---|---|---|
| 0 | 四 Queue，`B=A,Q=computeM*R`，`UB=B*d+ceil_align(B,256/d)*4+B*Q*d+B*Q*8` | `ReservedM` 收尾 M batch；`reservedRows` 走余行 N 分组 | X 原 dtype MicroAPI；UB `int32` index Cast 到 GM `int64`；strict-LT、latest tie、无 sticky NaN |
| 1 | 同一四 Queue 公式，`B=A,Q=computeR`；value/index carry 跨 R batch 存活 | `reservedR` 改最后 batch 行数；另有 `reservedRows` | 同 key0 |
| 2 | 同一四 Queue 公式，`B=computeN,Q=1`；carry 跨 R 循环存活 | `reservedN` 更新 mask/DMA 有效长度；另有 `reservedRows` | 同 key0 |
| 3 | 无 TQue/TBuf；GM + 每线程 value/index 寄存器 | block 内 M 余量由 `M%blockDim` 分配；线程以物理地址上界结束 | X 标量原 dtype，直接写 GM `int64`；latest tie、无 sticky NaN |
| 5 | 无 UB；GM + warp shuffle/carry 寄存器 | `R%32` 用 active lane，行数用 warp grid-stride | BF16 升 FP32 比较，其余用自身类型；以 `int32` 保存索引状态（浮点占最高位存 NaN）、GM 写 `int64`，sticky/latest NaN |
| 6 | 无 UB/workspace；三节点以 `values/indices` GM 原位交接，另用 warp/thread 寄存器 | 最后一段 `min(begin+1024,R)`；summary 以 32 段为 chunk；apply 以 `M*R` 上界结束 | 与 key5 相同；同 stream 顺序和 endpoint fixed-point 是精度/生命周期不变量 |
| 8 | 仅基础 key1 时可按 key1 的 `B=A,Q=computeR` 与四 Queue 公式核账；否则 active split 为 0 或 N-split，内存契约不成立 | 有效时用 `reservedR`；余行仍按 former/tail info 的 R/N 子 Route 收尾 | X 原 dtype MicroAPI，`int32→int64`；sticky/latest NaN；int32 的 NaN mask 恒假 |

### Dormant entry（与可达 Route 分离）

| key | 已存在 entry/body | dormant 证明 | 处理原则 |
|---:|---|---|---|
| 4 `PARALLEL_LANE_SIMT` | `cummin_k4_<dtype>` → `RunLaneScan/LaneScan` | `HostTiling::Run` 没有任何赋值为 4 的分支 | 不进入生产 launch 图；若未来接线，必须新增 selector、blockDim 与语义验证 |
| 7 `UB_SKLANSKY_REG` | `cummin_k7_<dtype>` → `CumminUbSklansky::Kernel::Process` | `HostTiling::Run` 没有任何赋值为 7 的分支 | 不把现有 UB body 当作当前 fallback 或 active Route |

未知 key 会落入 key3 body，而不是落入 key4/7 或报错；这是 router 的 ABI fallback，不改变 key4/7 的 dormant 状态。

## POD、Launcher 与生命周期

### ABI

`CumminSplitInfo` 与 `CumminRegbaseTilingData` 处于 `#pragma pack(push,8)`，字段全为 `int64_t`：

| POD | 大小/对齐 | 字段组 | consumer |
|---|---|---|---|
| `CumminSplitInfo` | 72B / 8B | `isNFullyLoad`; `splitR,reservedR,computeR`; `splitN,reservedN,computeN`; `computeLength,allocLength` | key1/2/8 及剩余 M 行子路径 |
| `CumminRegbaseTilingData` | 328B / 8B | `M,R,N`; 主核与剩余行分组；`splitM,computeM,ReservedM`; 三个 `CumminSplitInfo` | 所有 Device entry |
| `CumminLaunchConfig` | 352B / 8B | tiling@0，`blockDim`@328，padding，`tilingKey`@336，`localMemorySize`@344 | Host launcher；Device 仅按值接收 328B tiling |

字段生产契约：

| 字段组 | producer | 有效条件 |
|---|---|---|
| `M/R/N` | shape 合轴 | 所有 key |
| `mRowsPerCore` | `M/C` | key0/1/2/8 主段 |
| `splitM/computeM/ReservedM` | `DoUBSliceForM` | 基础 key0，且未被覆盖后仍只由 key0 body 消费 |
| `generalProcessInfo.R` 字段 | `DoUBSliceForR` | 基础 key1 |
| `generalProcessInfo.N` 字段 | `DoUBSliceForN` | 基础 key2 |
| `coreNum/perGroupCoreNum/former/tail` 及两个 split info | reserved-row 分支 | `reservedRows>0` |

config 使用值初始化，因此未生产字段为 0。Device 不能仅凭“字段存在于 POD”就消费它；key8 正是需要额外检查这一点的分支。

### 分核与剩余 M 行

R-001/R-005 的 AIV 主段先令每个 block 处理 `rowsPerCore` 个连续 M 行。若 `reservedRows>0`，Host 再计算：

```text
Cr = reservedRows                         if A*d <= 256
     floor(C/reservedRows)*reservedRows   otherwise
G  = Cr/reservedRows
formerCore = N%G
formerLen  = N/G+1
tailLen    = N/G
```

每个剩余 M 行由一组 `G` 个核切 N；前 `formerCore` 个核各处理 `formerLen`，其余处理 `tailLen`。Kernel 完成主段后 `TPipe::Reset()`，按 `isNFullyLoad` 在该 N 子片内再次选择 R 分批或 N 分片。

源码边界：`Cummin::Init` 在 `ComputeReservedM` 的 `reservedRows` guard 之前就按 `perGroupCoreNum` 计算 `gmOffset`。当 `reservedRows==0` 时，该字段保持 0，else 分支包含除 0。可靠实现应把这段地址计算移入 `reservedRows>0` 分支，或为 inactive POD 设置并证明安全的非零哨兵；不能把“没有剩余工作”当作现有代码自然安全的证明。

### launch 图

```text
cann_bench::cummin
  -> cummin_meta + contiguous/dtype checks
  -> calc_cummin_tiling
  -> launch_cummin_<f16|f32|bf16|i32>(x,y,indices,config,stream)
       key0 -> cummin_k0_*                       # one AIV
       key1 -> cummin_k1_*                       # one AIV
       key2 -> cummin_k2_*                       # one AIV
       key3/default -> cummin_k3_*               # AIV wrapper -> SIMT
       key5 -> cummin_k5_*                       # AIV wrapper -> warp SIMT
       key6 -> scan_* -> summary_* -> apply_*    # three AIV wrappers/SIMT
       key8 -> cummin_k8_*                       # one AIV
```

四个 dtype launcher 只更换 entry 后缀。所有 launcher 返回 `void`，外层 `acl_call` 固定返回 0；源码没有逐节点返回码、异步错误或中途清理分支。输入和两个输出由 Tensor 持有，lambda 捕获指针、config 与 stream。没有独立 workspace；key6 从第一次 launch 到第三次 launch 始终复用并要求保活 `values/indices`。

## Kernel Route 实现

### P0 · R-001：自适应 AIV 主干（key 0/1/2）

#### key 0：RN 全载，沿 M 分批

命中 `q>=1` 时，`computeM=q`，每核主段按 `splitM=ceil(rowsPerCore/computeM)` 分批。一次把 `[rows,R,N]` 搬入 UB，UB stride 为 `A=ceil_align(N,P)`：

```text
for each M batch:
    CopyIn rows*R records, each record has N valid values and A-N padding
    for each local row:
        carryValue = r0
        carryIndex = 0
        for r=1..R-1:
            retain prior iff prior < current
            write value and int32 index into UB
    cast indices int32 -> int64 in the output queue
    CopyOut only the N valid elements of each record
```

key0 的四块 UB（`B=A,Q=computeM*R`）：

```text
xTmpQue      B*d
argminTmpQue ceil_align(B,256/d)*4
yQue         B*Q*d
argminQue    B*Q*8
```

`argminQue` 的后半区先承载 `int32` 结果，随后将同一物理区解释为 `int64` 目的区做 Cast。覆盖顺序受 V→MTE3 硬事件约束。

#### key 1：N 全载，沿 R 分批

基础 key0 容不下且 `ubLines>=2` 时：

```text
L = ceil_align(N*d,32)
computeR = (U-L)/L
splitR = ceil(R/computeR)
reservedR = R%computeR
B = ceil_align(N,P), Q = computeR
```

每个 M 行按 R batch 搬运。第一个 batch 从 r0 初始化 value/index carry；后续 batch 复用 `xTmpQue/argminTmpQue`，`duplicate=j*computeR` 恢复全局 R 索引。最后一个 batch 以 `reservedR` 改写 `copyRows/computeTimes`，但 UB 仍按完整 `computeR` 分配。

#### key 2：单行 N 也不能全载，沿 N 分片

当 `ubLines<2`：

```text
computeN = floor_to_256(U/2)/d
splitN = ceil(N/computeN)
reservedN = N%computeN
B = computeN, Q = 1
```

每个 M 行先遍历 N tile，再在 tile 内按 r 串行。`r=0` 直接初始化并输出 carry；`r>=1` 每次只搬一行 N tile，更新 carry 后输出。尾 N 片按 `reservedN` 更新 MicroAPI mask 和 DMA 有效长度，不写 padded lane。

#### 数据流与同步不变量

| 阶段 | 内存/API | 必须保持的顺序 |
|---|---|---|
| GM→UB | `DataCopyPad`，`MTE3_MTE2` 与 `MTE2_V` | 前一输出不再占用复用区后才载入 |
| prefix compare | MicroAPI RegTensor + strict-LT mask | 同一 `(m,n)` 的 R 状态不可跨 batch/tile 重排 |
| index 扩宽 | UB 内 `int32 -> int64` Cast | Cast 完成后才允许 MTE3 读取 |
| UB→GM | `DataCopyPad`，V→MTE3/MTE3→V 事件 | 仅传有效 N，UB padding 不落 GM |

Host 的 `U` 公式只是近似预算。落地时须用上述真实 Queue 总字节复核 `<=` 实际可用 UB，尤其要计入 `argminTmpQue` 的 256B/vReg 对齐和每个 Queue 的分配约束。

### P1 · R-002：大 M、小 N lane-SIMT（key 3/default）

Host 预选条件为 `N<128 && M>12800 && R<12800`，随后仍可能被 key8/5/6 覆盖。`blockDim=min(M,C)`。每个 AIV block 用 `InitForSimT` 取得一段连续 M 行，再调用固定 2048 线程的 `SimtCompute`：

```text
logical b -> (m=b/N, n=b%N)
idx = m*R*N+n
for r in 0..R-1:
    scalar load x[idx+r*N]
    retain prior iff prior < current
    scalar store value and int64 index
```

线程以 2048 为 stride 继续处理本 block 的其它 `(m,n)`。它没有 TQue/TBuf、shared memory、跨线程同步或 atomic；所有依赖都在单线程寄存器中。R 尾部由循环条件自然覆盖。

`switch default` 与显式 key3 进入同一 body。这个行为只是源码 fallback，不是推荐的 ABI 容错：未知 key 应在 Host 报错，因为错误的 `blockDim` 或 POD 仍可能使默认 body 的分核与预期不一致。

### P2 · R-003：`N==1` warp-row（key 5）

key8 未命中后，`N==1 && M>=256` 覆盖所有基础 key 和 key3。`blockDim=C`，每核启动：

```text
rowsPerCore = ceil(M/C)
warpsPerCore = min(rowsPerCore,64)
threads = warpsPerCore*32
row = coreId + warpId*C
rowStride = C*warpsPerCore
```

一个 warp 顺序处理一行的 32 元素 chunk。每个 chunk 通过 `WarpShflUpSync` 的 `delta=1,2,4,8,16` 构造前缀；上一 chunk 的最后 active lane 广播为下一 chunk carry。末尾不足 32 元素时，只有 `r<R` 的 lane 读写，最后 active lane 产生 carry。

浮点 index 的最高位临时编码 NaN 状态，输出前以 `0x7fffffff` 取回索引；int32 输入不使用状态位。BF16 先转 FP32 寄存器比较，再转回 BF16 存储；half 与 float 使用自身计算类型。相等非 NaN 值保留右状态，sticky NaN 中后出现的 NaN 同样保留右状态。

该 Route 完全走 GM 与 warp 寄存器，不分配 UB。R 索引压入 31 位状态，因此复用时还需证明 `R-1<=INT32_MAX`；当前 Host 没有这一检查。

### P3 · R-004：少行超长 R segmented-row（key 6）

实际可达条件为 `N==1,M<256,R>=8192`。Launcher 在同一 stream 顺序提交：

```text
node 1: SegmentScan       blockDim=C
node 2: SegmentSummary    blockDim=min(M,C)
node 3: SegmentApplyCarry blockDim=C
```

数据生命周期：

1. node 1 将每行切为 `SEGMENT_SIZE=1024`，每个 warp 对一个 `(row,segment)` 做 32-lane chunk scan，向 `values/indices` 写 segment-local prefix。
2. node 2 每个 warp 扫一行的所有 segment endpoint，把 endpoint 原位改写为跨 segment 的全局 prefix 摘要。
3. node 3 以 2048 线程 grid-stride 扫 `M*R`；对 `segment>0` 的元素，从 `segment*1024-1` 读取前一段摘要并与本地 prefix 合并。

节点间没有额外 event 或 workspace，正确性依赖同 stream 的 Kernel 完成顺序和 GM 可见性。node 3 也会访问、重写后续 segment 的 endpoint；它依赖 node 2 结果对同一 `combine` 已是固定点，因此不得更换 tie/NaN combine 规则后仍假设该原位流程成立。

最后一个 segment 的 `end=min(begin+1024,R)`，warp 最后 active lane产生 carry。数值状态与 key5 相同：BF16 用 FP32 比较，浮点以 index 高位传递 NaN，输出为原 dtype value 与 `int64` index。

### P4 · R-005：NaN-aware AIV（key 8，受限）

Host 以 `N>=4096 && R<=32` 无条件覆盖前一 key；Device 却固定调用：

```text
Cummin<X,int64_t,true>::Init
ComputeR(generalProcessInfo)
ComputeReservedM
```

`DoCompare<true>` 的保留 mask 为：

```text
retainPrior = (isnan(prior) || prior < current) && !isnan(current)
```

所以 current NaN 总会成为新状态，prior NaN 会压过后续非 NaN，两个 NaN 相遇时取 current。int32 实例也可编译，`x!=x` mask 恒假，退化为严格 `<`。

这条 Route 的关键前置条件不是只有 `N>=4096,R<=32`：`generalProcessInfo.computeR/splitR/allocLength` 仅在基础 selector 原本选择 key1 时才由 `DoUBSliceForR` 生产。若基础路径是 key0，整个结构保持 0；若基础路径是 key2，只有 N-split 字段有效。两种情况下 key8 仍调用 `ComputeR`，参数契约不闭合。

因此可靠落地只能二选一：

- 把 key8 guard 收紧为“特殊谓词成立且基础 key 为 1”；或
- 为 key8 独立生产一套经 UB 核账的 R-split info，并在 `computeR>0` 后才启动。

在完成这一 Host 修复前，不能把源码中的整个 key8 谓词域当作可安全复用范围。它还继承 R-001 的剩余 M 行 `perGroupCoreNum` 边界。

## Route 间不变量与精度路径

| 维度 | 不变量 | Route 差异/边界 |
|---|---|---|
| layout | 所有地址都假设 contiguous `[M,R,N]`；扫描 stride 为 N | key5/6 只在 `N==1` 可达，因此行内 R 连续 |
| values dtype | 输出 dtype 与输入相同 | key5/6 的 BF16 比较升到 FP32；AIV 与 key3 不升精度 |
| indices dtype | GM 始终为 `int64` | AIV 在 UB 用 `int32` 再 Cast；SIMT 直接写 `int64`；必须证明 `R-1<=INT32_MAX` |
| tie | 非 NaN 相等时取右/最新索引 | 修改 compare 方向会同时破坏 segmented fixed-point 假设 |
| NaN | 不可声明为全 Route 相同 | key5/6/8 sticky 且 latest-NaN；key0/1/2/3 无显式 sticky 状态 |
| workspace | 0 | key6 使用输出 GM 作为三节点中间态 |
| 跨核通信 | 单节点 Route 无 atomic/跨核归并 | key6 的归并发生在 Kernel 边界，不是同 Kernel barrier |
| 尾块 | 只写有效元素 | AIV 用 `reservedM/R/N`、mask 与 Pad DMA；warp 用 active lane；SIMT 用循环上界 |

## 失败、fallback 与不可外推边界

- 空维、shape/offset 溢出、DMA 参数窄化、SIMT 31 位状态索引都缺少完整 Host 检查；这些必须在调用前闭合。
- key0/1/2/8 在 `reservedRows==0` 时存在 inactive 剩余行 POD 被 `Init` 提前消费的问题；可靠实现需先修 Host 默认值或 Device 控制流。
- key8 的 selector 与 `ComputeR` 所需 POD 不匹配；只有基础 key1 或独立 R-split producer 能闭合。
- `config.localMemorySize` 已计算，却没有出现在 `<<<blockDim,nullptr,stream>>>` 的 launch 参数中。Kernel 实际 Local Memory 上限与 Host 预算的一致性不在这段 Direct launcher 内证明。
- key4/key7 不可达；不能因 entry 存在就宣称 selector 会使用它们。
- unknown key 静默走 key3，launcher 为 `void`，外层固定返回 0；实现者应把未知 key 和 launch 错误显式上抛，而不是依赖这一 fallback。
- key6 三节点不可跨 stream、不可重排，也不能在节点之间释放或别名覆盖输出 Tensor。
- 官方固定提交可以支撑 Cummin 的基础 AIV/SIMT 设计与 950 映射；Direct 新增的 key5/6/8、PyTorch PrivateUse1 ABI 与多 Kernel 图只由当前 Direct 源码闭合，不能外推成官方公共接口事实。

## 源码闭包索引

| 阶段 | 文件/符号 | 支撑事实 |
|---|---|---|
| schema/Meta/PrivateUse1 | `op_plugin/cummin_plugin.cpp` | rank/dim、contiguous、dtype、输出分配、stream、dtype launcher |
| Direct Tiling | `op_kernel/cummin_kernel.cpp::HostTiling/calc_cummin_tiling` | M/R/N、UB/core 查询、key 生产、blockDim、覆盖顺序 |
| POD/ABI | `cummin_struct.h`, `cummin_launch.h` | 328B tiling 与 launcher config |
| AIV 主干与 key3 | `cummin.h::Cummin/SimtCompute` | key0/1/2/3 的 GM/UB、MicroAPI、剩余行与 SIMT body |
| warp/segmented | `cummin_parallel_scan.h` | key5/6 的 warp mapping、NaN 状态、segment 三阶段 |
| 未接线路径 | `cummin_parallel_scan.h::RunLaneScan`, `cummin_ub_sklansky.h::Kernel` | key4/7 body 存在但 Host 不生产 key |
| Router/launch graph | `cummin_kernel.cpp::LAUNCH_BY_KEY/launch_segmented_*` | 单/三 Kernel 分发、default、同 stream |
| 构建接线 | `cummin/CMakeLists.txt::register_direct_launch_op` | 只编译 `cummin_kernel.cpp` 与 plugin；旁置 Tiling 文件不参与 Direct target |

## 可迁移设计

“先按 UB 层级选择 M/R/N 切分，再由窄域 SIMT 路线覆盖”的结构适用于连续二维 lane 集合上的 prefix scan，但迁移时必须同时复制以下契约，而不能只复制 key：

1. value、carry index 与最终 `int64` index 的联合 UB 字节账；
2. `[M,R,N]` 地址关系和剩余 M 行的 N 分组 POD；
3. R/N/M 三类 tail 的有效长度、mask 与 DMA stride；
4. compare 的 tie/NaN 结合律，以及多阶段 segmented scan 的 fixed-point 条件；
5. 每个 override 对基础 POD producer 的依赖证明。

其中第 5 点是这份 Direct 实现最重要的边界：覆盖 key 不等于覆盖数据生产。任何特殊 Route 在改写 `tilingKey` 前，都应验证它要消费的 split 字段已经由相应 producer 填充。