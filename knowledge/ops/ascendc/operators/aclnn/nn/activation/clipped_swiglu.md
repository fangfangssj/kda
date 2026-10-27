---
type: Operator
title: "ClippedSwiglu A2/A3 ACLNN 算子知识"
description: "说明 aclnnClippedSwiglu 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/docs/aclnnClippedSwiglu.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/clipped_swiglu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/clipped_swiglu_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/clipped_swiglu_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/clipped_swiglu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/clipped_swiglu.hpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/config/ascend910b/clipped_swiglu_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/config/ascend910_93/clipped_swiglu_binary.json
    role: implementation
created_at: "2026-08-25T14:58:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ClippedSwiglu A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnClippedSwigluGetWorkspaceSize` + `aclnnClippedSwiglu`，公开两段式 ACLNN；固定仓只公开文档原型，inner bridge/executor 为框架生成边界。 |
| 计算语义 | 将 `x` 沿 `dim` 合并为 `[B,2H]`；可选取 `B'=min(sum(group_index),B)`；按奇偶或前后拆为 `A,Bgate`，计算 `min(A,L)/(1+exp(-alpha*min(A,L))) * (clamp(Bgate,-L,L)+bias)`。 |
| 输入输出 | `x/y` 为 ND、FP16/BF16/FP32，`y` 的 `dim` 长度为 `x/2`；`group_index` 可选、ND、INT64、1D。 |
| 执行类型 | 单 Kernel：`clipped_swiglu`，`AIV_VECTOR`；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001/R-002 奇偶无组/有组；R-003/R-004 前后短 H 无组/有组；R-005/R-006 前后长 H 无组/有组。每条均含 FP16、BF16、FP32 编译特化。 |
| 核心资源 | GM；`xQueue_`/`groupQueue_`/`yQueue_`；`tmpBuf1_`/`tmpBuf2_`；16 MiB user workspace 只作为入口非空条件，body 不读写。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→OpDef/自动 ACLNN 框架边界→Tiling 注册/完整 POD→key 1 router→`ClippedSwigluBase<T>`→binary config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设规范化后的 `dim` 为 `d`，`B=prod(x.shape[:d])`，`2H=prod(x.shape[d:])`。无 `group_index` 时 `B'=B`；有组时 Device 将 INT64 转 FP32、求和并取 `B'=min(int64(sum),B)`。`interleaved=true` 时 `A=x[:B',0::2]`、`Bgate=x[:B',1::2]`；否则 `A=x[:B',:H]`、`Bgate=x[:B',H:]`。令 `Ac=min(A,limit)`、`Bc=clamp(Bgate,-limit,limit)`，输出为 `Ac/(1+exp(-alpha*Ac))*(Bc+bias)`，仅覆盖前 `B'` 个 batch 对应输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，1–8 维 | FP16/BF16/FP32 | 非空；`dim` 轴为偶数；公开文档要求不含 Inf/NaN。 |
| `groupIndexOptional` | 可选 ND、1D，长度不超过 1024 | INT64 | 元素公开约束为非负；和限制实际 batch；固定 Kernel 对和取 `min(...,B)`。 |
| `dim` | 标量 | INT64 | `[-rank,rank-1]`，默认 `-1`。 |
| `alpha` | 标量 | double→Host float | 默认/建议 1.702。 |
| `limit` | 标量 | double→Host float | 公开要求 `>0`，默认/建议 7.0；本代 Tiling 未再次检查正值。 |
| `bias` | 标量 | double→Host float | 默认/建议 1.0。 |
| `interleaved` | 标量 | bool | 默认 `true`；决定奇偶或前后切分。 |
| `out` | 与 `x` 同 rank/ND，`dim` 减半 | 与 `x` 相同 | 非空；未进入 `B'` 的 batch 不由 Kernel 写出。 |

### 入口与版本

固定 Commit `955f8b33985b6d72989ebff341bb9c2c564a51bc` 的公开文档给出唯一 `aclnnClippedSwiglu` 两段式版本。README 与接口文档同时明确支持 Atlas A2、A3；OpDef 注册 `ascend910b`、`ascend910_93`，两份 binary config 均接入三种 dtype。自动生成的 ACLNN inner bridge、连续化与 executor 壳不在本目录源码中，按框架边界处理；算子拥有的 shape inference、Tiling、POD、Kernel 与构建配置均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口检查必需指针、dtype/rank/dim/output shape；OpDef 声明 ND 与三种同构 dtype。`InferShapeForClippedSwiglu` 将负 `dim` 加 rank，拒绝越界与奇数轴，并将该输出维减半。Tiling 按 `CheckAndGetXAndAttrs`→`CheckAndGetGroupIndex`→`CheckY`：读取属性，折叠 `B` 与 `2H`，检查 `dim`、偶数、x dtype、可选输入 1D/INT64、输出 rank/切分维。注意 `CheckY` 源码以 `GetInputDesc(Y_INDEX)` 读取 index 0 的输入描述，因此其 `yDtype` 比较并不构成独立输出 dtype 校验；输出 dtype 契约依赖公开 ACLNN/OpDef 框架边界。

### 参数、Workspace 与生命周期

`GetPlatformInfo` 取 `GetCoreNumAiv()` 与 UB 字节；`CountMaxPair` 生成 `ubMaxPair/isLongH`；`PostTiling` 固定 key=1、`blockDim=coreNumAll`，保存 POD，并申请 `16*1024*1024` bytes workspace。Kernel 要求 workspace 及 `GetUserWorkspace(workspace)` 非空，但随后不读写 `userspace`；workspace 由 ACLNN executor 管理并在该次 stream 执行期间存活。

### 启动图与失败传播

单节点图：executor 在给定 stream 启动 `clipped_swiglu(x,groupIndex,y,workspace,tiling)`，grid=`coreNumAll`，POD=`ClippedSwigluTilingData`。节点读 `x` 与可选 `groupIndex`，直接写 `y`，没有 Kernel 间 workspace 边。Host 校验、UB 容量或 Tiling 失败会阻断 launch；Kernel 收到空 workspace/userspace 时直接返回且不写输出；未知 key/dtype 编译分支没有 body。

## Route 与参数设计

### 分核与资源计算

`B=prod(shape[0:d])`，`2H=prod(shape[d:])`，`H=2H/2`。令 `G=align32(groupNum*8)`（无组为 0）、`N=ubSize-1024-2G`。Host 以每 pair `8+2+4+4=18` bytes 的队列/TBuf 模型，计算 `ubMaxPair=(floor(floor(N/18)/32)*32-31)/2`，要求 `N>0 && ubMaxPair>0`；`isLongH=(2*ubMaxPair<2H)`。Device 有组时先得到 `B'`。奇偶路线按总 pair `B'*H` 均分 AIV core；前后路线按 batch 均分 core。短 H 在单 loop 中合并多个 batch，长 H 对每 batch 再沿 H 切 `ubMaxPair`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 奇偶无组 | key 1；`ClippedSwigluBase<T>` | `isInterleaved=1 && isGroup=0` | R-002/R-003 |
| 2 | R-002 奇偶有组 | 同上 | `isInterleaved=1 && isGroup=1` | R-003 |
| 3 | R-003 前后短 H 无组 | 同上 | `isInterleaved=0 && isLongH=0 && isGroup=0` | R-004/R-005 |
| 4 | R-004 前后短 H 有组 | 同上 | `isInterleaved=0 && isLongH=0 && isGroup=1` | R-005 |
| 5 | R-005 前后长 H 无组 | 同上 | `isInterleaved=0 && isLongH=1 && isGroup=0` | R-006 |
| 6 | R-006 前后长 H 有组 | 同上 | `isInterleaved=0 && isLongH=1 && isGroup=1` | 失败 |

```text
TilingForClippedSwiglu: non-regbase -> priority 20000 ClippedSwigluTiling -> key 1
  interleaved ? (group ? R-002 : R-001)
              : (isLongH ? (group ? R-006 : R-005)
                         : (group ? R-004 : R-003))
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001/R-002 | 1 + flags | A2/A3，奇偶；可选组分别缺失/存在；FP16/BF16/FP32 | 单 AIV Kernel | `ClippedSwigluBase<half/bfloat16_t/float>`；Gather 拆偶奇 | 有组多一次 GM→groupQueue→ReduceSum；Queue/PIPE_V | core 与 UB loop 双尾 |
| R-003/R-004 | 1 + flags | A2/A3，前后，`2H<=2*ubMaxPair`；可选组 | 单 AIV Kernel | 同模板；strided 两次 CopyIn，批合并 | 32B batch padding；Queue/PIPE_V | 尾 batch loop |
| R-005/R-006 | 1 + flags | A2/A3，前后，`2H>2*ubMaxPair`；可选组 | 单 AIV Kernel | 同模板；每 batch 沿 H 两次连续 CopyIn | Queue/PIPE_V | 每 batch 尾 H loop |

三种编译特化共享控制流；FP32 无 cast，FP16 输入升 FP32/输出 `CAST_NONE`，BF16 输入升 FP32/输出 `CAST_RINT`，因此实现复用时必须保留 dtype 精度分支。

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| SoC | `ASCEND910B \|\| ASCEND910_93` | 本模板只注册这两种非 regbase SoC | 全部 | `IsCapable=false` | `ClippedSwigluTiling::IsCapable` |
| UB | `ubSize-1024-2*align32(groupNum*8)>0 && ubMaxPair>0` | 覆盖 double-buffer Queue 与两个 FP32 TBuf | 全部 | Tiling 失败 | `CountMaxPair` |
| 长 H | `2*ubMaxPair < 2H` | 决定按 batch 合并或 H 分段 | R-003～R-006 | 另一前后 Route | `CountMaxPair` |
| 32B 对齐 | `AlignBytes(pair*2)` | DMA block 与 FP32 物理复用要求 | 全部 | Host 减小 tile | `AlignBytes`/`CountMaxPair` |
| 活跃核 | `blockIdx < usedCoreNum` | group sum 可能缩短真实 batch | 全部 | Kernel 直接跳过 | `Process` |
| key/dtype | `TILING_KEY_IS(1)` 且编译 dtype 宏命中 | router 仅实例化三种 dtype/key 1 | 全部 | 无输出 body | `clipped_swiglu` |

### POD/ABI

宏生成结构的精确 `sizeof/alignof/offsetof` 在固定目录没有展开或 `static_assert`，下表保留声明顺序、位宽与端到端 producer/consumer；复刻 ABI 时必须使用同一 Tiling 宏生成器，不能手工猜偏移。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ClippedSwigluTilingData` | `coreNumAll` | int64/64 | 第1字段；精确 offset 固定仓不可见 | core | `set_coreNumAll(coreNumAll_)` | `CalTilingParam` 分核 | >0 |
| 同上 | `dimBatchSize` | int64/64 | 第2字段 | batch | 折叠 `dim<d` | `Process/CalTilingParam` | 乘法溢出未显式检查 |
| 同上 | `dim2H` | int64/64 | 第3字段 | element | 折叠 `dim>=d` | `Init/CalTilingParam` | 偶数 |
| 同上 | `isLongH` | int64/64 | 第4字段 | bool flag | `ubMaxPair*2<dim2H` | `Process/CopyIn/CopyOut` | 仅前后有效 |
| 同上 | `isGroup` | int64/64 | 第5字段 | bool flag | optional input 是否存在 | `Init/Process` | 0/1 |
| 同上 | `isInterleaved` | int64/64 | 第6字段 | bool flag | attr | 全控制流 | 0/1 |
| 同上 | `gluAlpha` | float/32 | 第7字段 | scalar | attr/default | `Compute` | ACLNN double 窄化为 float |
| 同上 | `gluLimit` | float/32 | 第8字段 | scalar | attr/default | `Compute` | 公开要求 >0 |
| 同上 | `gluBias` | float/32 | 第9字段 | scalar | attr/default | `Compute` | 有效 |
| 同上 | `ubMaxPair` | int64/64 | 第10字段 | pair | `CountMaxPair` | UB/loop 切分 | >0 |
| 同上 | `groupNum` | int64/64 | 第11字段 | element | group shape[0] | group DMA/reduce | 无组不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-006 | key 1 + 三个 POD flag | `clipped_swiglu_tiling.cpp`: `TilingForClippedSwiglu`/`DoOpTiling` | `ClippedSwigluTilingData` | `clipped_swiglu.cpp`: `clipped_swiglu` | `clipped_swiglu.hpp`: `ClippedSwigluBase<T>` 全方法 | `ascend910b`、`ascend910_93` binary JSON + `op_host/CMakeLists.txt` |

## Kernel 实现

### 公共执行流程

router 验证 workspace/userspace，按 `ORIG_DTYPE_X` 与 key 1 实例化 `ClippedSwigluBase<T>`。`Init` 绑定 GM，按组条件分配 `groupQueue_`，并分配 double-buffer `xQueue_`、single-buffer `yQueue_`、两个 TBuf。`Process` 可选在每个核独立复制并归约 group；再计算本核与本 loop 地址，执行 `CopyIn→(16-bit cast FP32)→GetAB→clamp/exp/div/mul→降精度→CopyOut`。队列提供 GM↔V 同步，连续 Vector 依赖由 `PipeBarrier<PIPE_V>` 隔开。

### Route R-001：奇偶切分无 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/clipped_swiglu_tiling.cpp` | `DoOpTiling` | `isInterleaved=1,isGroup=0,key=1` | R-001 |
| Router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | key 1 + dtype macro | `ClippedSwigluBase<T>` |

#### Device 数据流

##### 函数级调用链

```text
clipped_swiglu -> Init -> Process -> CalTilingParam(interleaved)
  -> ProcessSingleLoop -> CopyInInterLeaved -> Cast(16-bit only)
  -> GetAB(Gather even/odd) -> Compute -> CopyOut
```

##### 源码等价伪码

```text
pairTotal=B*H; distribute contiguous pair ranges across used cores
for each UB loop: p=front or tail; copy x[2*pairOffset:2*(pairOffset+p)]
  cast to FP32 if needed; Gather A from byte offset 0, Bgate from byte offset 4
  Ac=min(A,L); Bc=max(min(Bgate,L),-L)+bias
  y=Ac/(1+exp(-alpha*Ac))*Bc; cast back; copy p outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `2*pairOffset` / `pairOffset` | `2p` / `p` | T, interleaved | caller/Kernel | CopyIn/caller | executor | tail exact bytes |
| `xQueue_` | slot base | depth2 × `4A` | physical FP32 workspace | CopyIn/cast | Gather/Compute | pipe; Free after Compute | `A=align32(2*ubMaxPair)` |
| `yQueue_` | slot base | depth1 × `2A` | FP32 then T alias | Compute/cast | CopyOut | pipe | exact valid `p` |
| `tmpBuf1_/2_` | each base | each `4A` | FP32 | Gather/progression | Compute/Gather | whole Kernel | tmp1 halves A/B |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  xq["xQueue · depth=2 · 8A"]:2
  yq["yQueue · depth=1 · 2A"]:1
  t1["tmpBuf1 · 4A"]:1
  t2["tmpBuf2 · 4A"]:1
```

##### 字节账与复用顺序

```text
A=align32(2*ubMaxPair); UB_total=8A+2A+4A+4A=18A
UB_total <= ubSize-1024; dynamic_ub=N/A; workspace_total=16 MiB (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xQueue first FP32 area | 16-bit Cast/FP32 DMA | Gather、sigmoid temp | next AllocTensor | DeQue→FreeTensor | Compute 完成后才 Free |
| tmpBuf1 halves | Gather A/B | Compute | next loop Gather | PIPE_V barriers | 当前 Mul 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→V | xQueue EnQue/DeQue | CopyIn | cast/Gather | x slot | `p` tail |
| Vector chain | PIPE_V barrier | Gather/clamp/mul/exp/add/div | 下一 Vector op | tmp aliases | FP16/BF16 升 FP32；BF16 RINT、FP16 CAST_NONE 输出 |
| V→DMA | yQueue EnQue/DeQue | Compute/cast | CopyOut | y slot | 写 `p` elements |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling.cpp` | `CountMaxPair`/`DoOpTiling` | UB、flags、key、blockDim |
| Kernel entry/router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | dtype/key→模板 |
| Device body/helper | `op_kernel/clipped_swiglu.hpp` | `CalTilingParam`/`CopyInInterLeaved`/`GetAB`/`Compute` | pair 分核、Gather、FP32 数学、tail |

### Route R-002：奇偶切分有 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/clipped_swiglu_tiling.cpp` | `CheckAndGetGroupIndex` | optional shape 非空→`isGroup=1` | R-002 |
| Device | `op_kernel/clipped_swiglu.hpp` | `Process` | `tl_->isGroup==1` | group prepass 后复用 R-001 主体 |

#### Device 数据流

##### 函数级调用链

```text
clipped_swiglu -> Init(groupQueue depth2) -> Process
  -> DataCopyPad(group) -> Cast<int64,float> -> ReduceSum -> min(sum,B)
  -> R-001 奇偶主链
```

##### 源码等价伪码

```text
on every launched core: copy groupNum int64; cast RINT to float; ReduceSum; scalar read; B'=min(int64(sum),B)
then execute R-001 over pairTotal=B'*H; blocks outside usedCoreNum return without x/y work
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| group GM→Queue | 0 | `groupNum*8` valid；depth2 × G allocated | INT64 1D | caller/DMA | Cast/ReduceSum | pipe | G=align32(groupNum*8) |
| 其余 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  gq["groupQueue · depth=2 · 2G"]:2
  xq["xQueue · 8A"]:2
  yq["yQueue · 2A"]:1
  tmp["tmpBuf1+2 · 8A"]:1
```

##### 字节账与复用顺序

```text
UB_total=2G+18A <= ubSize-1024; workspace_total=16 MiB (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf1 first scalar | ReduceSum | GetValue(0) | R-001 Gather | scalar read precedes CalTilingParam | group prepass complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group | groupQueue + ReduceSum | DMA/Cast | scalar `GetValue` | group/tmp | INT64 先转 FP32；和再转 INT64，取 min B |
| main | 同 R-001 | 同 R-001 | 同 R-001 | x/y | B' 缩短产生核尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling.cpp` | `CheckAndGetGroupIndex`/`CountMaxPair` | group flag/容量 |
| Kernel entry/router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | 单 Kernel |
| Device body/helper | `op_kernel/clipped_swiglu.hpp` | `Init`/`Process` | 每核 group 归约与 B' 限制 |

### Route R-003：前后切分短 H 无 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/clipped_swiglu_tiling.cpp` | `CountMaxPair` | `isInterleaved=0,isLongH=0,isGroup=0` | R-003 |

#### Device 数据流

##### 函数级调用链

```text
Process -> CalTilingParam(short H batches) -> CopyInHalfShortH(two strided DMA)
  -> GetAB(contiguous halves) -> Compute -> CopyOut(batch blocks)
```

##### 源码等价伪码

```text
distribute B batches; batchSpace=4*align32(2H bytes); merge ubMaxBatch=xQueSpace/batchSpace
for loop batches: copy each A row and B row with srcStride=H*sizeof(T)
pad each row to 32B physical FP32 layout; compute calPairNum over padded rows; copy valid H per batch
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `batch*2H`、第二半 `+H` | blockCount=batch，blockLen=`H*sizeof(T)` | T rows | caller | DMA | caller | srcStride=H*sizeof(T) |
| UB A/B | x slot halves | `batch*align32(H*2)/2` FP32 elements each | padded FP32 | DMA/cast | Compute | xQueue | per-row 32B |
| y GM | `batch*H` | blockCount=batch，blockLen=`H*sizeof(T)` | T rows | CopyOut | caller | caller | padding not written |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xQueue · depth=2 · 8A"]:2
  yq["yQueue · 2A"]:1
  tmp["tmpBuf1+2 · 8A"]:1
```

##### 字节账与复用顺序

```text
UB_total=18A; calPair includes 32B row padding while CopyOut uses pairNum valid elements
workspace_total=16 MiB (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| padded lanes | DMA/cast | Vector ops | next loop | Queue lifetime | CopyOut excludes padding |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| batch DMA | Queue | two CopyIn calls | Compute | A/B halves | last loop batch count |
| output | CopyOut blockCount | yQueue | GM | valid H rows | skips UB padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling.cpp` | `CountMaxPair` | short/long flag |
| Kernel entry/router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | key 1 |
| Device body/helper | `op_kernel/clipped_swiglu.hpp` | `CalTilingParam`/`CopyInHalfShortH`/`CopyOut` | batch 合并与 padding |

### Route R-004：前后切分短 H 有 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| POD/Device | `op_kernel/clipped_swiglu.hpp` | `Process` | R-003 条件且 `isGroup=1` | group prepass→R-003 |

#### Device 数据流

##### 函数级调用链

```text
R-002 group prepass -> R-003 short-H front/back chain with B'
```

##### 源码等价伪码

```text
B'=min(sum(float(groupIndex)),B); recompute used cores and batch tails from B'; then execute R-003
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| group + main | 同 R-002 group 与 R-003 main | `2G+18A` | INT64 + T/FP32 | DMA/Vector | scalar/main | pipe | tmpBuf1 在 group 后复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  gq["groupQueue · 2G"]:1
  xq["xQueue · 8A"]:2
  yq["yQueue · 2A"]:1
  tmp["tmpBuf1+2 · 8A"]:1
```

##### 字节账与复用顺序

```text
UB_total=2G+18A; workspace_total=16 MiB (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf1 | group ReduceSum | scalar read | GetAB | program order/Queue | scalar read complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group/main | 同 R-002 + R-003 | group prepass | main | B' and buffers | B' batch tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling.cpp` | `CheckAndGetGroupIndex`/`CountMaxPair` | flags/UB |
| Kernel entry/router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | single node |
| Device body/helper | `op_kernel/clipped_swiglu.hpp` | `Process`/`CopyInHalfShortH` | group + short H |

### Route R-005：前后切分长 H 无 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/clipped_swiglu_tiling.cpp` | `CountMaxPair` | `isInterleaved=0,isLongH=1,isGroup=0` | R-005 |

#### Device 数据流

##### 函数级调用链

```text
Process(batch loop -> H loop) -> CopyInHalfLongH(two contiguous DMA)
  -> GetAB -> Compute -> CopyOut(single block)
```

##### 源码等价伪码

```text
distribute B batches; for each batch, for h0 in steps ubMaxPair:
  p=min(ubMaxPair,H-h0); copy A[h0:h0+p] and Bgate[h0:h0+p]
  compute p values and write y[h0:h0+p]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `batch*2H+h0`, second `+H` | `p*sizeof(T)` each | T contiguous | caller | DMA | caller | tail DataCopyPad |
| y GM | `batch*H+h0` | `p*sizeof(T)` | T | Kernel | caller | caller | exact tail |
| UB | same 18A layout | `p<=ubMaxPair` | FP32 compute | DMA/Vector | Vector/DMA | pipe | no row padding in calPair |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xQueue · 8A"]:2
  yq["yQueue · 2A"]:1
  tmp["tmpBuf1+2 · 8A"]:1
```

##### 字节账与复用顺序

```text
UB_total=18A; workspace_total=16 MiB (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| all UB slots | current H tile | Compute/CopyOut | next H tile | FreeTensor + PIPE_V | output enqueued before overwrite |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| H tiling | loop condition | CalTilingParam | CopyIn/Out | GM slice | final `pairLastLoop` |
| numeric | same public chain | FP32 Vector | cast | output | dtype-specific cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling.cpp` | `CountMaxPair` | long H guard |
| Kernel entry/router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | dtype route |
| Device body/helper | `op_kernel/clipped_swiglu.hpp` | `Process`/`CopyInHalfLongH` | nested batch/H loops |

### Route R-006：前后切分长 H 有 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| POD/Device | `op_kernel/clipped_swiglu.hpp` | `Process` | R-005 条件且 `isGroup=1` | R-006 |

#### Device 数据流

##### 函数级调用链

```text
R-002 group prepass -> R-005 batch/H nested main chain with B'
```

##### 源码等价伪码

```text
derive B' from group; distribute B' batches; execute H tiles exactly as R-005; inactive launched cores do no x/y work
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| group + main | R-002 + R-005 | `2G+18A` | INT64/T/FP32 | DMA/Vector | scalar/main | pipe | sequential reuse |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  gq["groupQueue · 2G"]:1
  xq["xQueue · 8A"]:2
  yq["yQueue · 2A"]:1
  tmp["tmpBuf1+2 · 8A"]:1
```

##### 字节账与复用顺序

```text
UB_total=2G+18A; workspace_total=16 MiB (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf1/xQueue/yQueue | group or current tile | scalar/Compute/CopyOut | following stage | Queue + program order | group scalar consumed first |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all | R-002 then R-005 mechanisms | group/DMA/Vector | scalar/main/CopyOut | B' and tiles | batch + H tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling.cpp` | all check/count methods | group and long H |
| Kernel entry/router | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | single AIV node |
| Device body/helper | `op_kernel/clipped_swiglu.hpp` | `Process`/`CalTilingParam`/`Compute` | full route |

## 算子特定优化

### Trick T-001：同一 xQueue 物理区完成 16-bit 输入到 FP32 的原地升精度

#### 触发与源码依据

- 适用 Route：R-001～R-006 的 FP16/BF16 特化。
- 触发条件：`ORIG_DTYPE_X != DT_FLOAT`。
- Golden 来源：[`op_kernel/clipped_swiglu.hpp:245-265`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/clipped_swiglu.hpp#L245-L265)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
#if (ORIG_DTYPE_X != DT_FLOAT) // 16位的数据DataCopy的时候放在buffer的后面一半，防止后续Cast出错
    xLocalOffset1_ = xQueSpace_ / SWI_FACTOR / static_cast<int64_t>(sizeof(bfloat16_t));
    xLocalOffset2_ = xLocalOffset1_ / SWI_FACTOR;
#endif
```

#### 机制与设计原因

每个 xQueue slot 按 FP32 计算空间预留；16-bit DMA 写入后半区，随后 Cast 写前半区，避免另配输入转换 buffer，也防止 Cast 的目标覆盖尚未读取的 16-bit 源。前后切分时 `xLocalOffset2_` 再定位 B 半区；奇偶切分一次搬入 2p 元素。该布局与 `xQueSpace_=4*align32(2*ubMaxPair bytes)`、double buffer 一起决定 Host 的 UB 上界。

#### 迁移边界

- 适用：输入需升到 FP32 且可把窄类型源放在同一较大物理 slot 尾部的逐元素 Vector 算子。
- 前提：A2/A3、FP16/BF16、目标区与源区无破坏性重叠、DMA/Vector 间由 Queue 保序。
- 失效：FP32 直接输入不需要该偏移；改变 tile/slot 尺寸必须重新证明覆盖顺序。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码注释明确指出若不把 16-bit 数据放到 buffer 后半区，后续 Cast 可能破坏源数据；固定仓未给出独立 fallback。

### Trick T-002：短 H 合并 batch、长 H 沿 H 切片

#### 触发与源码依据

- 适用 Route：R-003～R-006。
- 触发条件：`isLongH = (2*ubMaxPair < dim2H)`。
- Golden 来源：[`op_kernel/clipped_swiglu.hpp:165-193`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/clipped_swiglu.hpp#L165-L193)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tl_->isLongH == 0) {
    int64_t batchSpace = SWI_FACTOR * DTYPE_FACTOR * AlignBytes(dimH_ * static_cast<int64_t>(sizeof(bfloat16_t)));
    int64_t ubMaxBatch = xQueSpace_ / batchSpace;
    loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
} else {
    loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
}
```

#### 机制与设计原因

短 H 时把多个 batch 合成一次 Vector 工作量并保留每行 32B padding；长 H 时固定一个 batch，沿 H 连续搬运，避免跨行 stride 与 padding 占满 UB。二者共享计算 body，但 DMA 参数、地址递增与 tail 不同。

#### 迁移边界

- 适用：二维 `[B,2H]` 前后切分且 H 尺寸跨度大的逐元素算子。
- 前提：Host 与 Device 使用同一 `ubMaxPair/isLongH`，短 H CopyOut 必须排除 padding。
- 失效：奇偶布局走连续 2p 搬运；UB 公式变化必须重算分界。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码通过两个显式分支证明统一采用跨 batch 或统一采用单 batch H 切片会改变 UB 与 DMA 约束；未提供第三条 fallback。

## 模板与实现对比

| 对比维度 | R-001/R-002 奇偶 | R-003/R-004 前后短 H | R-005/R-006 前后长 H | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | interleaved | !interleaved && !long | !interleaved && long | 属性 + UB/H |
| 模板与 dtype | 同一 `Base<T>`，Gather | 同一 `Base<T>` | 同一 `Base<T>` | dtype 编译宏保留 cast 差异 |
| 循环与分核 | pair 均分 | batch 均分并合批 | batch 均分、每批 H loop | 连续性和 UB |
| Queue/TBuf | 无组 18A；有组 +2G | 同 | 同 | group 条件分配 |
| 内存总占用与复用 | tmp2 生成 Gather offset | row padding | contiguous tiles | 保持同一上界 |
| 搬运/Vector API | 单 DMA + Gather | 两个 strided DMA + Copy | 两个连续 DMA + Copy | 输入布局 |
| 同步、精度与 tail | pair tail | batch/padding tail | batch+H tail | Queue + PIPE_V |
| fallback/失败边界 | UB/key/workspace 失败 | 同 | 同 | 无其他 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持（`ascend910b`） | 支持（`ascend910_93`） | 支持但使用独立 `_950` 卡 |
| Host/Launcher | 共用非 regbase Tiling，blockDim=全部 AIV core | 同 A2 | regbase 自定义 Tiling、真实核数、4 template keys |
| Kernel/资源 | `ClippedSwigluBase<T>` + Vector API | 同 A2 | MicroAPI body、7×tile UB 模型、V_S event |

## 限制与失败边界

- 公开 ACLNN 要求 rank 1–8、非空、`dim` 合法、切分维为偶数、三种 dtype、group 1D/INT64/长度≤1024/元素非负、`limit>0`。本代算子 Tiling 源码未重验 rank 上限、空 Tensor、group 长度/元素值、Inf/NaN、limit 正值或独立 y dtype；这些属于公开 ACLNN/框架校验边界，不能从 Kernel 补推。
- shape 连乘、`groupNum*sizeof(int64_t)` 与若干 `uint16_t` DMA 参数窄化没有本目录显式溢出保护。
- group 求和路径先 `INT64→FP32` 再 ReduceSum，精确整数范围受 FP32 表示约束；源码未提供另一精确归约路线。
- workspace 固定 16 MiB 且必须可解析 user workspace，虽然 body 不访问；不得传空指针省略。
- 有 group 时所有已 launch core 都独立搬运并求和，随后只有 `blockIdx<usedCoreNum` 的核处理 x/y。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnClippedSwiglu.md` | 两段式原型/产品表 | 公开 API、约束、A2/A3 支持 |
| 入口实现 | `op_host/clipped_swiglu_def.cpp` | `ClippedSwiglu`/`OP_ADD` | dtype/layout/平台配置与 ACLNN 生成接线 |
| Host/Tiling/Launcher | `op_host/clipped_swiglu_tiling.h` | `ClippedSwigluTilingData` | POD 声明 |
| Host/Tiling/Launcher | `op_host/clipped_swiglu_tiling.cpp` | `TilingForClippedSwiglu` 等 | 校验、UB、key、blockDim、workspace |
| Kernel entry/body | `op_kernel/clipped_swiglu.cpp` | `clipped_swiglu` | router |
| Kernel entry/body | `op_kernel/clipped_swiglu.hpp` | `ClippedSwigluBase<T>` | 完整 Device 实现 |
| 构建接线 | `op_host/config/ascend910b/clipped_swiglu_binary.json` | `op_list` | A2 三 dtype binary |
| 构建接线 | `op_host/config/ascend910_93/clipped_swiglu_binary.json` | `op_list` | A3 三 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 11 字段驱动单 key body | 字段顺序/位宽、float 窄化、flag 语义 | `ClippedSwigluTilingData` |
| 调用顺序 | optional group→CalTiling→tile pipeline | group scalar必须先读完 | `Process` |
| workspace/内存 | 18A(+2G) UB 模型 | depth、32B alignment、1024B reserve | `CountMaxPair`/`Init` |
| 同步 | Queue + PIPE_V | Cast/Gather/数学链依赖 | `ProcessSingleLoop`/`Compute` |
| 精度与尾块 | FP32 中间数学、dtype-specific output cast | BF16 RINT；有效长度与 padding 分离 | `ProcessSingleLoop` |
| 失败边界 | 非法 shape/dtype/UB 阻断 launch | key 1、workspace 非空、SoC capable | `IsCapable`/entry |