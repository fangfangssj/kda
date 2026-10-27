---
type: Operator
title: "ClippedSwiglu 950 ACLNN 算子知识"
description: "说明 aclnnClippedSwiglu 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/docs/aclnnClippedSwiglu.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/clipped_swiglu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/clipped_swiglu_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/clipped_swiglu_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/arch35/clipped_swiglu_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/arch35/clipped_swiglu_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/clipped_swiglu_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/arch35/clipped_swiglu_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_host/config/ascend950/clipped_swiglu_binary.json
    role: implementation
created_at: "2026-08-25T14:58:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ClippedSwiglu 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnClippedSwigluGetWorkspaceSize` + `aclnnClippedSwiglu`，公开两段式 ACLNN；inner bridge/executor 为框架生成边界。 |
| 计算语义 | 将 `x` 合并为 `[B,2H]`，可选按 `min(sum(group_index),B)` 截短 B；奇偶或前后拆 A/G，输出 `min(A,L)/(1+exp(-alpha*min(A,L)))*(clamp(G,-L,L)+bias)`。 |
| 输入输出 | ND FP16/BF16/FP32 `x/y`；可选 1D INT64 `group_index`；`y` 的 `dim` 轴减半。 |
| 执行类型 | 单 Kernel `clipped_swiglu`，显式 `KERNEL_TYPE_AIV_ONLY`，AIV MicroAPI；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 奇偶无组、R-002 奇偶有组、R-003 前后无组、R-004 前后有组；四个 template key，各含三种 dtype binary。 |
| 核心资源 | GM；double-buffer `inQueX_`/`outQueY_`；`vectorBuf_`；有组时 `reduceSumBuf_`；寄存器级解交织/FP32 数学；lib-api workspace 仅取 user 地址但 body 不使用。 |
| 来源闭包 | `source_closed`：公开 ACLNN→OpDef/框架 bridge→regbase selector→Arch35 POD/key→AIV-only entry→完整 MicroAPI body→ascend950 binary config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

规范化 `d=dim>=0?dim:dim+rank`，`B=prod(shape[:d])`，`2H=prod(shape[d:])`。无组时 `B'=B`；有组时每个核用 INT64 MicroAPI ReduceSum 得到 `B'=min(sum(group_index),B)`。奇偶路线取 `A=x[:B',0::2]`、`G=x[:B',1::2]`；前后路线取 `A=x[:B',:H]`、`G=x[:B',H:]`。MicroAPI 在 FP32 中计算 `Ac=min(A,L)`、`Gbc=clamp(G,-L,L)+bias`、`Ac/(1+exp(-alpha*Ac))*Gbc`；16-bit 输出以 `CAST_RINT` 降精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，1–8 维 | FP16/BF16/FP32 | 非空，`dim` 轴偶数，不支持 Inf/NaN。950 OpDef 配置 `AutoContiguous`。 |
| `groupIndexOptional` | 可选 ND、1D、长度≤1024 | INT64 | 元素公开要求非负；950 `AutoContiguous`。 |
| `dim` | scalar | INT64 | `[-rank,rank-1]`，默认 -1。 |
| `alpha` | scalar | double→float POD | 默认/建议 1.702。 |
| `limit` | scalar | double→float POD | 必须 >0，默认/建议 7.0；950 Tiling 显式检查。 |
| `bias` | scalar | double→float POD | 默认/建议 1.0。 |
| `interleaved` | scalar | bool | true=奇偶，false=前后，默认 true。 |
| `out` | 与 x 同 rank，`dim` 减半 | 同 x | ND；未进入 B' 的 batch 不由 Kernel 写出。 |

### 入口与版本

固定 Commit `955f8b33985b6d72989ebff341bb9c2c564a51bc` 仅有一个公开 ACLNN 版本。产品表明确支持 Ascend 950PR/950DT，OpDef `ascend950` regbase 配置接到 `clipped_swiglu_apt`，binary config 接入三种 dtype。`TilingForClippedSwiglu` 先用 `IsRegbaseSocVersion` 选择 `Tiling4ClippedSwigluArch35`；因此 950 使用该架构的最新实际实现，而不复用 A2/A3 body。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段 ACLNN/框架校验必需指针、rank、dtype、连续性与输出契约。950 Tiling 按 `CheckAndGetXAndAttrs`→`CheckAndGetGroupIndex`→`CheckY` 执行：归一化负 dim，拒绝 rank/总元素非正、dim 越界、切分维奇数、非三种 x dtype、`limit<=0`、group 非 1D/INT64，以及输出 rank/shape 不一致；`CheckY` 逐轴比较除切分维外的 x/y shape。该函数以 `GetInputDesc(Y_INDEX)` 读取 index 0 的输入描述，所以其中 `yDtype` 比较并不构成独立输出 dtype 校验；该契约由公开 ACLNN/OpDef 框架边界保证。

### 参数、Workspace 与生命周期

`GetPlatformInfo` 读取 AIV 核数、UB 字节、平台 UB block size 与 `GetLibApiWorkSpaceSize()`。`CountUbFactor` 计算 `hUbFactor/bUbFactor`；`ComputeCoreSplit` 计算 `realCoreNum`；`SetTilingKey` 用 `interleaved×group` 生成四个 tuple key；Host 写 64-byte POD（普通 ABI 推导）并设置 `blockDim=realCoreNum`、workspace=`libApiWorkSpaceSize`。entry 调用 `GetUserWorkspace(workspace)`，但 `usrWorkspace` 后续不被读取；owner/lifetime 仍由 executor 和 stream 管理。

### 启动图与失败传播

单节点图：`clipped_swiglu<isInterleaved,isGroup>(x,groupIndex,y,workspace,tiling)`，AIV-only grid=`realCoreNum`，POD=`ClippedSwigluArch35TilingData`。四个 key 编译期消除 group/布局分支，entry 再实例化 `ClippedSwigluKernel<DTYPE_X,bool,bool>`。Host 任一检查、UB 计算或 workspace 获取失败阻断 launch；Device `ComputeTiling` 在 group 截短后可能把实际活跃核数减小，`blockIdx>=realCoreNum_` 直接返回。

## Route 与参数设计

### 分核与资源计算

`B=prod(shape[:d])`、`H=prod(shape[d:])/2`、`s=sizeof(T)`、`blk=GetUbBlockSize()`、`q=blk/s`。有组时 Host 先预留 `blk` bytes；`ubFactor=floor_to_q((ubSize-(group?blk:0))/(7*s))`。默认 `hUbFactor=ubFactor,bUbFactor=1`；仅前后路线且 `H<hUbFactor` 时，`hUbFactor=max(floor_to_q(H),q)`，`bUbFactor=ubFactor/hUbFactor`，把剩余 UB 转为 batch tile。奇偶路线按 `B*H` 一维均分核；前后路线先按 B 分核，再用剩余核切 H；有组的 Host launch 固定全部 AIV core，因为真实 B' 只能在 Device 求得，Device 重算活跃核。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 奇偶无组 | `(TPL_INTERLEAVED_TRUE,TPL_NO_GROUP_INDEX)` | `interleaved && optional group absent` | R-002/R-003 |
| 2 | R-002 奇偶有组 | `(TRUE,GROUP_INDEX)` | `interleaved && group present` | R-003 |
| 3 | R-003 前后无组 | `(FALSE,NO_GROUP_INDEX)` | `!interleaved && group absent` | R-004 |
| 4 | R-004 前后有组 | `(FALSE,GROUP_INDEX)` | `!interleaved && group present` | 失败 |

```text
IsRegbaseSocVersion == true -> Tiling4ClippedSwigluArch35
  key = GET_TPL_TILING_KEY(interleaved ? 1 : 0, group ? 1 : 0)
  -> clipped_swiglu<isInterleaved,isGroup>
  -> ClippedSwigluKernel<DTYPE_X,isInterleaved,isGroup>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `(1,0)` | 950；三 dtype；奇偶；无组 | 单 AIV-only | `Kernel<T,true,false>` | 连续 DMA、寄存器 DeInterleave、7U UB | core/H/mask tail |
| R-002 | `(1,1)` | 950；三 dtype；奇偶；有组 | 单 AIV-only | `Kernel<T,true,true>` | 先 INT64 reduce、V_S event，再同 R-001；+32B | B' 核尾 + H/mask tail |
| R-003 | `(0,0)` | 950；三 dtype；前后；无组 | 单 AIV-only | `Kernel<T,false,false>` | 双 strided DMA、B×H 二维分核、7U UB | B/H/mask tail |
| R-004 | `(0,1)` | 950；三 dtype；前后；有组 | 单 AIV-only | `Kernel<T,false,true>` | group reduce + R-003；+32B | B'、B/H/mask tail |

FP16/BF16 路由使用 `DIST_UNPACK_B16`→FP32 和 `CAST_RINT` 回 T；FP32 直接以 FP32 RegTensor 运算。四条控制路线各保留这两类精度特化。

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| regbase | `IsRegbaseSocVersion(context)` | 决定 950 自定义 Tiling/body | 全部 | 非 regbase 模板 | `TilingForClippedSwiglu` |
| UB factor | `ubFactor>0` | 4U input Queue +2U output Queue +1U vectorBuf | 全部 | Tiling 失败 | `CountUbFactor` |
| group fits input slot | `groupUb<=2*ubFactor` | group 临时复用 `inQueX_` 一个 `2U` slot | R-002/R-004 | Tiling 失败 | `CountUbFactor` |
| H/B tile | `H<hUbFactor && !interleaved` | 前后布局可把剩余容量换为 batch 合并 | R-003/R-004 | `bUbFactor=1` | `CountUbFactor` |
| group launch cores | `isGroup ? realCoreNum=coreNumAll : computed` | Host 未知真实 B' | R-002/R-004 | Device 重算后闲核返回 | `ComputeCoreSplit`/`ComputeTiling` |
| vector tail | `tail=dim1OnceSize%VF_LEN_FP32` | MicroAPI 完整向量与 mask tail 分离 | 全部 | `tailTimes=0/1` | `ComputeVfSwiglu` |

### POD/ABI

`ClippedSwigluArch35TilingData` 是显式普通结构：六个 int64 后接三个 float；按成员自然对齐可得 offsets 0/8/16/24/32/40/48/52/56，alignof 8、sizeof 64（末尾 4-byte padding）。固定源码没有 `static_assert`，跨编译器复刻应补 ABI 检查。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ClippedSwigluArch35TilingData` | `realCoreNum` | int64/64 | offset 0 | core | `ComputeCoreSplit` | `Init/ComputeTiling` | >0 |
| 同上 | `dimBatchSize` | int64/64 | offset 8 | batch | shape 前缀积 | `ComputeTiling` | 乘法溢出未显式检查 |
| 同上 | `dimH` | int64/64 | offset 16 | element | 后缀积/2 | `Init/CopyIn/Out` | >0 |
| 同上 | `groupNum` | int64/64 | offset 24 | element | group shape[0] | group DMA/reduce | 无组模板不读 |
| 同上 | `bUbFactor` | int64/64 | offset 32 | batch/tile | `CountUbFactor` | B loop/UB | >=1 |
| 同上 | `hUbFactor` | int64/64 | offset 40 | element/tile | `CountUbFactor` | H loop/UB | block 对齐且 >0 |
| 同上 | `gluAlpha` | float/32 | offset 48 | scalar | attr/default | `ComputeVfSwiglu` | double→float |
| 同上 | `gluLimit` | float/32 | offset 52 | scalar | attr/default | `ComputeVfSwiglu` | >0 |
| 同上 | `gluBias` | float/32 | offset 56 | scalar | attr/default | `ComputeVfSwiglu` | 有效 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-004 | 四个 tuple key | `clipped_swiglu_tiling.cpp` selector + `clipped_swiglu_tiling_arch35.cpp` | 显式 Arch35 POD | `clipped_swiglu_apt.cpp`: template entry | `arch35/clipped_swiglu_kernel.h`: 全部 helper | `clipped_swiglu_def.cpp` + `ascend950` binary JSON |

## Kernel 实现

### 公共执行流程

entry 标注 AIV-only，解析 POD，按两个 key 模板参数实例化 body。`Init` 绑定 GM 并分配 UB；`ComputeTiling` 可选归约 group，再还原 core/B/H tile。每个 tile 执行 `DataCopyPad GM→inQueX`，MicroAPI 从 UB 加载并在寄存器中升 FP32、拆 A/G、clamp/exp/div/mul、降精度，写 `vectorBuf_`；`UbDataCopy` 把结果搬入 `outQueY_`，`CopyOut` 以有效 block 参数写 GM。Queue 维护 DMA/Vector 生命周期；group 的向量结果到 scalar `GetValue` 用显式 `V_S` event。

### Route R-001：奇偶切分无 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/clipped_swiglu_tiling_arch35.cpp` | `SetTilingKey` | interleaved true, group false | tuple `(1,0)` |
| Router | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<1,0>` | `if constexpr` | `Kernel<T,true,false>` |

#### Device 数据流

##### 函数级调用链

```text
clipped_swiglu<1,0> -> Init -> Process -> ComputeTiling
  -> CopyIn(contiguous 2p) -> ComputeVfSwiglu(DeInterleave)
  -> UbDataCopy -> CopyOut(contiguous p)
```

##### 源码等价伪码

```text
pairTotal=B*H; distribute contiguous pair range per core
for p in hUbFactor tiles: input offset=2*output offset; DMA 2p elements
  register-load alternating values; DeInterleave to A/G
  FP32 Ac=min(A,L); s=Ac/(1+exp(-alpha*Ac)); Gc=clamp(G,-L,L)+bias; out=s*Gc
  masked tail, cast/store to vectorBuf, UB copy to outQueue, DMA p outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `2*outOffset`/`outOffset` | `2p`/`p` | T interleaved/T | caller/Kernel | DMA/caller | executor | exact tail bytes |
| `inQueX_` | slot base | depth2 × `2U` =4U | T alternating | DMA | MicroAPI | pipe until FreeTensor | `U=hUb*bUb*s` |
| `outQueY_` | slot base | depth2 × U =2U | T | UbDataCopy | DMA | pipe | valid p |
| `vectorBuf_` | base | U | T | MicroAPI | UbDataCopy | Kernel | reused each tile |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inq["inQueX · depth=2 · 4U"]:4
  outq["outQueY · depth=2 · 2U"]:2
  vec["vectorBuf · U"]:1
```

##### 字节账与复用顺序

```text
U=hUbFactor*bUbFactor*sizeof(T); UB_total=7U <= ubSize
dynamic_ub=N/A; workspace_total=GetLibApiWorkSpaceSize() (Kernel body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| vectorBuf | ComputeVfSwiglu | UbDataCopy | next tile | function order before next iteration | UB copy completes before overwrite |
| Queue slots | DMA/UbDataCopy | MicroAPI/DMA | next AllocTensor | DeQue/FreeTensor | depth2 protocol |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V | Queue | DataCopyPad | DeQue | input slot | DMA p tail |
| compute | MaskReg | register loads | math/store | lanes | full vectors + `maskT`; 16-bit升FP32/RINT回写 |
| V→GM | Queue | UbDataCopy | CopyOut | output slot | p exact |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling_arch35.cpp` | `ComputeCoreSplit`/`SetTilingKey` | pair 分核、key |
| Kernel entry/router | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<1,0>` | AIV-only specialization |
| Device body/helper | `op_kernel/arch35/clipped_swiglu_kernel.h` | `Process`/`ComputeVfSwiglu` | continuous DMA、DeInterleave、masked math |

### Route R-002：奇偶切分有 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/clipped_swiglu_tiling_arch35.cpp` | `SetTilingKey` | interleaved true, group present | `(1,1)` |
| Device | `op_kernel/arch35/clipped_swiglu_kernel.h` | `ComputeTiling` | `if constexpr(isGroup)` | group prepass→R-001 main |

#### Device 数据流

##### 函数级调用链

```text
clipped_swiglu<1,1> -> Init(+reduceSumBuf) -> ComputeTiling
  -> group DataCopyPad -> ReduceAllVf -> V_S event -> scalar B'
  -> R-001 main chain
```

##### 源码等价伪码

```text
each launched core copies groupNum int64 into an inQueue slot; vector reduce with masked tail
store one int64 to reduceSumBuf; V_S wait; B'=min(value,B); recompute active pair cores
inactive blocks return; active blocks execute R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| group GM | 0 | `groupNum*8` | INT64 1D | caller | DMA | caller | DataCopyPad |
| inQueue slot | base | one 2U slot; condition `align(groupNum)*8<=2*ubFactor` | INT64 then T x | group DMA/main DMA | Reduce/main | sequential | same slot reused after Free |
| `reduceSumBuf_` | base | 32B | INT64 scalar | ReduceAllVf | scalar GetValue | Kernel | group-only allocation |
| main | same R-001 | 7U | T | same | same | pipe | after group prepass |

##### 分层内存布局

```mermaid
block-beta
  columns 8
  inq["inQueX · 4U"]:4
  outq["outQueY · 2U"]:2
  vec["vectorBuf · U"]:1
  red["reduceSum · 32B"]:1
```

##### 字节账与复用顺序

```text
UB_total=7U+32B <= ubSize; group temporary aliases an inQueue slot
workspace_total=lib-api size (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | group DMA | ReduceAllVf | x CopyIn | group FreeTensor | reduce finished |
| reduceSumBuf | Vector reduce | scalar GetValue | no further write | V_S SetFlag/WaitFlag | scalar waits for Vector |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group | Queue + `HardEvent::V_S` | DMA/Vector | scalar GetValue | group/scalar | exact INT64 reduce，mask tail |
| main | same R-001 | main pipeline | GM | x/y | B' core tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling_arch35.cpp` | `CheckAndGetGroupIndex`/`ComputeCoreSplit` | group flag/full-core launch |
| Kernel entry/router | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<1,1>` | compile-time group route |
| Device body/helper | `op_kernel/arch35/clipped_swiglu_kernel.h` | `ReduceAllVf`/`ComputeTiling` | exact group reduce/event/B' |

### Route R-003：前后切分无 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/clipped_swiglu_tiling_arch35.cpp` | `SetTilingKey` | interleaved false, group absent | `(0,0)` |
| Router | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<0,0>` | compile-time tuple | `Kernel<T,false,false>` |

#### Device 数据流

##### 函数级调用链

```text
clipped_swiglu<0,0> -> Init -> ComputeTiling(B×H core map)
  -> CopyIn(two strided A/G DMA) -> ComputeVfSwiglu
  -> UbDataCopy -> CopyOut(strided rows)
```

##### 源码等价伪码

```text
split cores into bCore*hCore; each core gets B and H ranges
for each B tile and H tile: DMA A rows from base and G rows from base+H,
  blockCount=onceB, blockLen=onceH*s, srcStride=(2H-onceH)*s
  compute each row with alignDim1 stride; output DMA blockCount=onceB, dstStride=(H-onceH)*s
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `(b0+bi)*2H+h0`; G at `+H` | `onceB×onceH` twice | T rows | caller | two DMA | caller | srcStride `(2H-onceH)*s` |
| inQueue A/G | A base；G at `bUb*hUb` | each U, depth2 slots | T padded rows | DMA | MicroAPI | pipe | `alignDim1In` |
| vector/outQueue | base | U / depth2×U | T row-major padded | MicroAPI/UbCopy | UbCopy/DMA | pipe | output stride excludes padding |
| y GM | `(b0+bi)*H+h0` | `onceB×onceH` | T | Kernel | caller | caller | dstStride `(H-onceH)*s` |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inq["inQue(A|G) · depth=2 · 4U"]:4
  outq["outQue · depth=2 · 2U"]:2
  vec["vectorBuf · U"]:1
```

##### 字节账与复用顺序

```text
UB_total=7U; U=hUbFactor*bUbFactor*s; workspace_total=lib-api size (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| vector padded rows | ComputeVfSwiglu | UbDataCopy | next B/H tile | sequential call | CopyOut consumes Queue copy, not vectorBuf |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| B/H split | former/tail formulas | ComputeTiling | loops | GM range | independent B and H tails |
| register compute | full mask + maskT | input Queue | vectorBuf | lanes | per-row tail, FP32 intermediate |
| output | Queue/DataCopyPad | UbDataCopy | GM | valid rows | strided exact onceH |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling_arch35.cpp` | `CountUbFactor`/`ComputeCoreSplit` | B/H tile/core |
| Kernel entry/router | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<0,0>` | template mapping |
| Device body/helper | `op_kernel/arch35/clipped_swiglu_kernel.h` | `ComputeTiling`/`CopyIn`/`CopyOut` | two-dimensional strided path |

### Route R-004：前后切分有 group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/clipped_swiglu_tiling_arch35.cpp` | `SetTilingKey` | interleaved false, group present | `(0,1)` |
| Device | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<0,1>` | constexpr tuple | group prepass→front/back |

#### Device 数据流

##### 函数级调用链

```text
clipped_swiglu<0,1> -> group DataCopy/ReduceAllVf/V_S -> recompute B'×H core map
  -> R-003 CopyIn/ComputeVfSwiglu/UbDataCopy/CopyOut
```

##### 源码等价伪码

```text
B'=min(reduce_sum_int64(group),B); recompute bCore/hCore from launched realCoreNum
blocks outside recomputed realCoreNum return; active blocks execute B'×H tiles exactly as R-003
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| group stage | same R-002 | inQueue slot + 32B | INT64 | DMA/Vector | scalar | pipe | released before main |
| main stage | same R-003 but B→B' | 7U | T | DMA/MicroAPI | MicroAPI/DMA | pipe | B/H tails |

##### 分层内存布局

```mermaid
block-beta
  columns 8
  inq["inQue · 4U"]:4
  outq["outQue · 2U"]:2
  vec["vectorBuf · U"]:1
  red["reduceSum · 32B"]:1
```

##### 字节账与复用顺序

```text
UB_total=7U+32B <= ubSize; workspace_total=lib-api size (body unused)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue | group DMA | ReduceAllVf | A/G DMA | FreeTensor | group read complete |
| remaining UB | main tiles | MicroAPI/DMA | next tile | Queue/sequential order | tail extents respected |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group | Queue + V_S event | ReduceAllVf | scalar | B' | INT64 masked reduction |
| main | Queue + masks | strided DMA | register/GM | B'×H | B/H/register tails |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/clipped_swiglu_tiling_arch35.cpp` | `CheckAndGetGroupIndex`/`ComputeCoreSplit` | full-core launch/key |
| Kernel entry/router | `op_kernel/clipped_swiglu_apt.cpp` | `clipped_swiglu<0,1>` | AIV-only route |
| Device body/helper | `op_kernel/arch35/clipped_swiglu_kernel.h` | `ReduceAllVf`/`ComputeTiling`/`Process` | group + B/H route |

## 算子特定优化

### Trick T-001：template key 在编译期消除 group 与布局分支

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：Host 由两个 bool 产生 tuple key。
- Golden 来源：[`op_kernel/clipped_swiglu_apt.cpp:35-54`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/clipped_swiglu_apt.cpp#L35-L54)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (isInterleaved == 1) {
    if constexpr (isGroup == 1) {
        ClippedSwigluKernel<DTYPE_X, true, true> op(&tilingData, &pipe);
        op.Init(x, groupIndex, y);
        op.Process();
    } else {
        ClippedSwigluKernel<DTYPE_X, true, false> op(&tilingData, &pipe);
```

#### 机制与设计原因

Host key 把 `interleaved` 与 optional group 存在性变成两个编译期模板参数。无组实例完全消除 group GM 绑定、32B buffer、归约与 V_S event；前后实例不生成 DeInterleave 路线，奇偶实例不生成双 strided DMA 路线。优化改变了分支、内存和搬运，而不只是 C++ 封装。

#### 迁移边界

- 适用：小而稳定的离散属性组合，且每种组合有显著资源/搬运差异。
- 前提：Host `GET_TPL_TILING_KEY` 与 entry 模板参数枚举完全一致；binary config 覆盖 dtype。
- 失效：新增属性值必须同步扩展 key 声明、Host selector 与 entry，否则无匹配实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓没有运行时通用 fallback；key 与实例不一致会缺少正确 body。

### Trick T-002：奇偶输入直接在寄存器 DeInterleave

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：`isInterleaved=true`。
- Golden 来源：[`op_kernel/arch35/clipped_swiglu_kernel.h:371-384`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/clipped_swiglu/op_kernel/arch35/clipped_swiglu_kernel.h#L371-L384)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (isInterleaved) {
    if constexpr (sizeof(T) == sizeof(half)) {
        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
            vregX1, x1UbAddr, srcIdxOffset);
        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
            vregX2, x2UbAddr, srcIdxOffset);
    }
    AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
```

#### 机制与设计原因

GM→UB 保持连续的 `[A0,G0,A1,G1,...]`，MicroAPI 在寄存器加载时对 16-bit 解包并用 `DeInterleave` 得到 FP32 A/G；无需在 UB 物化两个拆分张量，也避免两次带 stride 的 GM DMA。FP32 也走两寄存器加载后解交织。

#### 迁移边界

- 适用：二路固定交错、逐元素配对计算且 VReg 可表达解交织的场景。
- 前提：`gateAddr=inputUb[VF_LEN_FP32]` 与 `vfLen=2*VF_LEN_FP32` 必须保持；tail mask按输出 pair 而非输入元素。
- 失效：前后布局应使用双 DMA；多路交错或不同 element size 需重做地址与 LoadDist。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的前后路线是可验证反例：若对前后布局强行 DeInterleave，会把两个连续半区错误配对；必须走 `Kernel<T,false,...>`。

## 模板与实现对比

| 对比维度 | R-001 奇偶无组 | R-002 奇偶有组 | R-003 前后无组 | R-004 前后有组 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `(1,0)` | `(1,1)` | `(0,0)` | `(0,1)` | 两个 bool tuple |
| 模板与 dtype | `T,true,false` | `T,true,true` | `T,false,false` | `T,false,true` | dtype再分 FP32/16-bit精度 |
| 循环与分核 | pair 1D | B'×H pair 1D | B×H 2D | B'×H 2D | 布局与 group |
| Queue/TBuf/寄存器 | 7U | 7U+32B | 7U | 7U+32B | group条件分配 |
| 内存总占用与复用 | in/out/vector | inQueue先给group | A/G同slot halves | group后复用slot | Host 7倍模型 |
| 搬运/MicroAPI | 连续2p+DeInterleave | 同+ReduceAllVf | 双strided DMA | 同+ReduceAllVf | GM连续性 |
| 同步、精度与 tail | Queue/mask | +V_S/B'核尾 | Queue/B/H/mask | +V_S/B' | 编译期消枝 |
| fallback/失败边界 | 无通用key fallback | group容量失败 | UB失败 | group容量失败 | Host阻断 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持（Ascend 950PR/950DT） |
| Host/Launcher | priority Tiling 模板，key 1，全 AIV core | 同 A2 | regbase selector、自定义 POD、4 tuple key、真实 blockDim |
| Kernel/资源 | 高阶 Vector API/Gather，18A(+2G) | 同 A2 | MicroAPI/DeInterleave，7U(+32B)，INT64 reduce + V_S |

## 限制与失败边界

- 公开限制为 rank 1–8、非空、三种 dtype、dim 合法且偶数、group 1D/INT64/长度≤1024/元素非负、limit>0、不含 Inf/NaN。950 Tiling 显式覆盖 x shape/x dtype/dim/limit/输出 shape，但不独立校验 y dtype，也不读取 Device 数据检查 group 元素非负或 Inf/NaN；这些仍依赖 ACLNN/调用方契约。
- shape 乘积、DMA `blockLen/srcStride/dstStride` 的窄化和 MicroAPI `uint16_t` loop 次数没有固定目录内的显式全范围溢出证明。
- group route 在每个 launch core 重复搬运和归约；Host 因未知 B' 启动全部 AIV core，Device 再使多余 core 返回。
- `CopyOut` 源码在 `DeQue` 后再次 `EnQue(outputUb)` 再执行 DMA/Free；复刻必须保持该固定仓顺序，不能仅凭常见 Queue 模式删去。
- entry 获取 `usrWorkspace` 但不使用；workspace 大小仍由平台 lib-api producer 给出，不能推断为可传空。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnClippedSwiglu.md` | ACLNN 原型/产品表 | 950 公开支持与接口契约 |
| 入口实现 | `op_host/clipped_swiglu_def.cpp` | `regbaseConfig`/`AddConfig("ascend950")` | AutoContiguous、apt 接线 |
| Host/Tiling/Launcher | `op_host/clipped_swiglu_tiling.cpp` | `TilingForClippedSwiglu` | regbase 分流 |
| Host/Tiling/Launcher | `op_host/clipped_swiglu_tiling_arch35.cpp` | `ClippedSwigluArch35Tiling` | 校验、资源、key、POD、blockDim |
| Host/Tiling/Launcher | `op_kernel/arch35/clipped_swiglu_tiling_data.h` | POD struct | Device ABI |
| Host/Tiling/Launcher | `op_kernel/arch35/clipped_swiglu_tiling_key.h` | TPL args/selector | 四 key |
| Kernel entry/body | `op_kernel/clipped_swiglu_apt.cpp` | template `clipped_swiglu` | AIV-only router |
| Kernel entry/body | `op_kernel/arch35/clipped_swiglu_kernel.h` | `ClippedSwigluKernel`/helpers | 完整 body |
| 构建接线 | `op_host/config/ascend950/clipped_swiglu_binary.json` | `op_list` | 950 三 dtype binaries |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 64B、6×int64+3×float | 字段顺序/offset、float 窄化 | `ClippedSwigluArch35TilingData` |
| 调用顺序 | optional group→重算分核→tile pipeline | V_S 必须先于 scalar B' | `ComputeTiling` |
| workspace/内存 | 7U(+32B)，inQueue group复用 | Queue depth、U定义、group fit guard | `Init`/`CountUbFactor` |
| 同步 | Queue + group V_S | output Queue 的固定 DeQue/EnQue/DMA/Free 顺序 | `ComputeTiling`/`CopyOut` |
| 精度与尾块 | FP32寄存器数学、16-bit RINT、mask tail | interleaved input stride与输出 pair mask | `ComputeVfSwiglu` |
| 失败边界 | Host检查/UB/key阻断 | 四 tuple key与AIV-only执行类型 | `SetTilingKey`/entry |