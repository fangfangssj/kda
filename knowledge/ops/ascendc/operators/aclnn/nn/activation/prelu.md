---
type: Operator
title: "Prelu A2 ACLNN 算子知识"
description: "说明 aclnnPrelu 在 A2 上的接口语义、Host/Launcher 参数生成、七条 Route、单 Kernel 启动图与 AIV 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/docs/aclnnPrelu.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_host/prelu_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_host/prelu_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_kernel/prelu_tiling_data.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_kernel/prelu_tiling_key.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_kernel/prelu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_kernel/prelu.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_host/CMakeLists.txt"
    role: implementation
created_at: "2026-08-25T18:52:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# Prelu A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnPreluGetWorkspaceSize` + `aclnnPrelu`；公开两段式 ACLNN，inner bridge/executor 由 `ACLNNTYPE aclnn` 生成。 |
| 计算语义 | `y = max(x, 0) + weight[c] * min(x, 0)`；`weight` 为单元素或第 1 维通道权重。 |
| 输入输出 | `self/weight/out` 为 ND，FLOAT/FLOAT16/BFLOAT16，rank 0–8；此实现要求三者 dtype 一致。 |
| 执行类型 | 七条路线均为单 AIV Kernel：`AIV_VECTOR`，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 标量；R-002 NC 常驻权重；R-003 NC 权重复用；R-004 NC split-C；R-005 split-L 跨核；R-006 full-L；R-007 split-L。 |
| 核心资源 | GM；depth=2 的 VECIN/VECOUT；按 dtype 条件分配的 VECCALC TBuf；workspace 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→自动 ACLNN bridge→A2 OpDef→Tiling/POD/key→entry/router→七个 AIV body→活动 experimental CMake。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对展平位置 `i`，若是通道权重，令原 shape 为 `[N,C,D2,...]`、`L=ΠDk`、`c=(i/L)%C`；输出为 `x_i>0 ? x_i : x_i*weight_c`。源码的 `Maxs/Mins/Mul/Add` 在 `x==0` 时同样得到 0。单元素权重直接广播至全部元素；Backward 不属于本卡。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，rank 0–8 | FLOAT/FLOAT16/BFLOAT16 | 可为空；shape 必须等于 `out`。 |
| `weight` | 文档允许 rank 0–8；Kernel Tiling 要求 1-D | 同 `self` | 元素数为 1 或 `self.shape[1]`；通道模式要求 `N,C` 及后续维均正。 |
| `out` | ND，与 `self` 相同 | 同 `self` | 只写真实元素；32B 对齐 padding 不写回。 |

### 入口与版本

固定 Commit 中候选目录公开 `aclnnPrelu`，`op_host/CMakeLists.txt:add_modules_sources(... OPTYPE prelu ACLNNTYPE aclnn)` 将自动 ACLNN bridge 作为框架边界接入；`Prelu` OpDef 只配置 `ascend910b`。这是 A2 实际支持的最新可达实现。A3 没有对应活动 OpDef/config/Host/Kernel 链路，故本卡仅标 A2。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口检查空指针、ND/FLOAT/FLOAT16/BFLOAT16、三者 dtype 一致、`self/out` shape 一致及权重元素数。自动 bridge 的逐行实现固定仓不可见；算子 Host 再校验 x/y storage shape、三 dtype、weight 必须 1-D 且正。通道模式将 `C=x[1]`、`L=Πx[2:]`、`rowNum=N*C`，每次乘法均先做 `int64/uint64` 溢出检查。

### 参数、Workspace 与生命周期

`GetPlatformInfo` 读取 AIV 核数和 UB，核数截断到 40；`usableUbSize = ubSize>1024 ? ubSize-1024 : ubSize`。`GetWorkspaceSize` 写 `workspace[0]=0`。executor、stream 与生成 bridge 的中间对象生命周期属于框架边界；唯一算子 Kernel 直接消费 x/weight、写 y。

### 启动图与失败传播

```text
aclnnPreluGetWorkspaceSize (生成 bridge)
  -> PreluTilingFunc -> SetBlockDim(usedCoreNum), SetTilingKey(key)
aclnnPrelu(workspace=0, executor, stream)
  -> prelu<schMode>(x, weight, y, workspace, tiling)
     -> Init* -> Process*
```

单节点 K0 的 POD 为 `PreluTilingData`，workspace slice 为 0 字节。Tiling 任一空指针、shape/dtype、UB 或溢出检查失败即不启动 K0；Kernel 失败由 executor/stream 返回。无跨 Kernel 边。

## Route 与参数设计

### 分核与资源计算

令 `S=sizeof(T)`、`A=32/S`。基础每元素 UB 字节为 FP32 24、FP16 12、BF16 20；`ubFactor=floor(usableUb/baseBytes/A)*A`。标量路线按 `512/S` 元素对齐分核。通道路线先计算 `LAligned=ceil(L/A)*A`；NC 路线按 batch 行分核，split-C 和 split-L-parallel 按 task 分核，普通 full/split-L 按 `N*C` 行分核。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 标量 | mode 0 | `weightSize==1` | 通道决策 |
| 2 | R-002 NC 常驻权重 | mode 6 | `weightMode==1 && preferReuse && C>1 && N>0 && L==1 && weightCache有效且整行 tile 可容纳` | R-004/后续 |
| 3 | R-003 NC 权重复用 | mode 4 | 同上但 `L>1` | R-004/后续 |
| 4 | R-004 NC split-C | mode 5 | reuse 优先但整行放不下；`L==1` 可容纳至少一个 32B C tile，或 `L>1` 可容纳且最终核数达到 `min(coreLimit,10)` | R-005 |
| 5 | R-005 split-L 跨核 | mode 3 | `rowNum*2<=coreLimit && L>=2*(512/S)`，按目标核数求 tile 后 `finalCore>=min(coreLimit,rowNum*2)` | R-006/R-007 |
| 6 | R-006 full-L | mode 1 | 通道模式且上述均未命中，`LAligned<=ubFactor` | R-007 |
| 7 | R-007 split-L | mode 2 | 通道模式最终 fallback，`LAligned>ubFactor` | Tiling 失败仅来自前置 guard |

```text
weightSize == 1 -> R-001
else try whole-NC reuse -> R-002(L==1) / R-003(L>1)
else try split-C reuse -> R-004
else try split-L tasks across cores -> R-005
else LAligned <= ubFactor -> R-006
else -> R-007
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 0 | 单权重，A2，三 dtype | K0 | `Prelu<T>::ProcessScalar` | 双队列+2/3 TBuf | 32B pad，真实长度写回 |
| R-002 | 6 | 通道，`L=1`，整行 NC tile 可容纳 | K0 | `ProcessChannelNcResidentWeightReuse` | 权重常驻并倍增复制 | 最后一批 N 行 |
| R-003 | 4 | 通道，`L>1`，整行 NC tile 可容纳 | K0 | `ProcessChannelNcWeightReuse` | 权重一次搬入、展开成 NCL 向量 | 最后一批 N 行 |
| R-004 | 5 | reuse 优先，整行不容纳但 C tile 可容纳 | K0 | `ProcessChannelNcSplitCWeightReuse` | 按 C tile 缓存权重 | 最后 C tile |
| R-005 | 3 | 少行、大 L，跨核 tile 有益 | K0 | `ProcessChannelSplitLParallel` | task 均分，标量 weight 按 channel 缓存 | 最后 L tile |
| R-006 | 1 | 每行完整 L 可容纳 | K0 | `ProcessChannelFullL` | 每行一次搬入/写回 | L pad |
| R-007 | 2 | 每行 L 超 UB | K0 | `ProcessChannelSplitL` | 核内遍历 L tiles | 最后 L tile |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| UB 最小 tile | `floor(usableUb/baseBytes)>=A` | 至少一个 32B 对齐 tile | 全部 | `GRAPH_FAILED` | `CalcTiling` |
| 核上限 | `coreLimit=min(GetCoreNumAiv(),40)` | 实现显式限制 | 全部 | 截断 | `GetPlatformInfo` |
| reuse 启发式 | `L==1 \|\| (L<=64&&C>=32) \|\| (L<=1024&&32<=C<=1024)` | 限制权重展开范围 | R-002–004 | R-005 | `preferWeightReuseByRow` |
| split-C 并行度 | `finalCore>=min(coreLimit,10)`（L>1） | 避免少 task 的 split-C | R-004 | R-005 | `CalcTiling` |
| 32B tail | `computeLen=AlignUp(realLen,32/S)` | Vector/搬运对齐，padding 清零 | 全部 | 真实长度写回 | `CopyGmToLocalPad` |
| BF16 升精度 | `T=bfloat16_t` | FP32 中间与 RINT 降精度 | 全部 | 编译期 FP16/FP32 分支 | `Compute/ComputeNc` |

### POD/ABI

`PreluTilingData` 由 14 个连续 `int64_t` 组成，固定定义下 `sizeof=112`、`alignof=8`，字段 offset 按表递增 8 字节。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `PreluTilingData` | `totalLength` | int64/64 | off 0 | 元素 | `totalNum` | GM extent、batch 推导 | shape size |
| 同上 | `usedCoreNum` | int64/64 | off 8 | core | `finalCoreNum` | Init 的有效核 guard | 1..40 |
| 同上 | `formerNum` | int64/64 | off 16 | core | 标量分核 | `InitScalar` | 非标量为 0 |
| 同上 | `formerLength` | int64/64 | off 24 | 元素/core | `blockFactor` | `InitScalar` offset/length | 512B 对齐 |
| 同上 | `tailLength` | int64/64 | off 32 | 元素 | 余量 | `InitScalar` | 可为 0（空 x） |
| 同上 | `tileLength` | int64/64 | off 40 | 元素或 C | `ubFactor/ncMax/splitC/parallelTile` | 各 Init | Route 相关 |
| 同上 | `channelSize` | int64/64 | off 48 | channel | `x[1]` | channel index | 正数 |
| 同上 | `innerSize` | int64/64 | off 56 | 元素/channel | `Πx[2:]` | L 循环/地址 | 通道模式正数 |
| 同上 | `innerSizeAligned` | int64/64 | off 64 | 元素 | `AlignUp(L,A)` 或 aligned row/C | 各 NC/full Init | Route 相关 |
| 同上 | `baseRows` | int64/64 | off 72 | row/core | `batch/core` 或 `rowNum/core` | 行分核 | task 路线 inactive |
| 同上 | `extraRows` | int64/64 | off 80 | core | 余数 | 行分核 | task 路线 inactive |
| 同上 | `tilesPerRow` | int64/64 | off 88 | tile/row | split route | task→row/tile | 普通路线 0 |
| 同上 | `baseTasks` | int64/64 | off 96 | task/core | totalTask/core | task 分核 | 行路线 0 |
| 同上 | `extraTasks` | int64/64 | off 104 | core | task 余数 | task 分核 | 行路线 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | mode 0 | `prelu_tiling.cpp:CalcTiling` | `PreluTilingData` | `prelu.cpp:prelu<0>` | `prelu.h:InitScalar/ProcessScalar` | `op_host/CMakeLists.txt:add_modules_sources` |
| R-002 | mode 6 | 同上 | 同上 | `prelu<6>` | `InitChannelNcResidentWeightReuse/Process...` | 同上 |
| R-003 | mode 4 | 同上 | 同上 | `prelu<4>` | `InitChannelNcWeightReuse/Process...` | 同上 |
| R-004 | mode 5 | 同上 | 同上 | `prelu<5>` | `InitChannelNcSplitCWeightReuse/Process...` | 同上 |
| R-005 | mode 3 | 同上 | 同上 | `prelu<3>` | `InitChannelSplitLParallel/Process...` | 同上 |
| R-006 | mode 1 | 同上 | 同上 | `prelu<1>` | `InitChannel/ProcessChannelFullL` | 同上 |
| R-007 | mode 2 | 同上 | 同上 | `prelu<2>` | `InitChannel/ProcessChannelSplitL` | 同上 |

## Kernel 实现

### 公共执行流程

`prelu<schMode>` 解析 POD，创建单核私有 `TPipe`，编译期分派 Init/Process。CopyIn 用单 block、`blockLen=dataNum*S` 的 `DataCopyPad`，右侧补零；Compute 为 `Maxs(x,0)`, `Mins(x,0)`, `Mul[s]`, `Add`；CopyOut 仅提交 `realLen*S`。队列的 `Alloc/EnQue/DeQue/Free` 建立 MTE2→V→MTE3 生命周期。BF16 先 `CAST_NONE` 到 FP32，最后 `CAST_RINT`。

### Route R-001：标量权重

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `CalcTiling` | `weightMode==0` | mode 0 |
| Router | `op_kernel/prelu.cpp` | `prelu<schMode>` | 最终 `else` | `InitScalar/ProcessScalar` |

#### Device 数据流

##### 函数级调用链

```text
prelu<0> -> InitScalar -> InitBuffers -> ProcessScalar
  -> CopyInByOffset -> Compute(Muls scalar) -> CopyOut
```

##### 源码等价伪码

```text
coreOffset=core*formerLength; len=(core<formerNum?formerLength:tailLength)
weightScalar=load(weight[0]); for tile: real=min(ubLength,remain); n=align32(real)
copy x[coreOffset+tile] with zero pad; y=max(x,0)+min(x,0)*weightScalar; copy real
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `core*formerLength+i*ubLength` | `real` | T/ND | API/Compute | CopyIn/user | API/K0 | 真实长度 |
| input/output Q | 0 | 各 `2*ubLength*S` | T | MTE2/V | V/MTE3 | TPipe | depth 2 |
| pos/neg/xFp32 | 0 | 非BF16各 `ubLength*S`；BF16各 `ubLength*4` | T/FP32 | V | V | tile | 无 alias 证据 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["VECIN depth=2 · 2U·S"]:1
  outQ["VECOUT depth=2 · 2U·S"]:1
  x32["BF16 xFP32 · U·4"]:1
  pos["pos · U·calcS"]:1
  neg["neg · U·calcS"]:1
```

##### 字节账与复用顺序

```text
UB_total = U*{24(FP32),12(FP16),20(BF16)} <= usableUb
dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputQueue slot | CopyIn | Compute | 下一 tile | DeQue/FreeTensor | depth 2 |
| outputQueue slot | Compute | CopyOut | 下一 tile | EnQue/DeQue/FreeTensor | 只写 real |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile pipeline | Queue | MTE2/V | V/MTE3 | Q slot | 32B pad；真实长度写回 |
| BF16 | Cast | BF16 x | FP32 vector | xFp32 | `CAST_NONE`→FP32→`CAST_RINT` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `CalcTiling` | 512B 分核与 tile |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<0>` | 默认标量分支 |
| Device body/helper | `op_kernel/prelu.h` | `InitScalar/ProcessScalar/Compute` | 标量缓存、pad、Vector 管线 |

### Route R-002：NC 常驻权重

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `CalcTiling` | reuse 条件、`L==1`、`ncMaxTile>=alignedRow` | mode 6 |
| Router | `op_kernel/prelu.cpp` | `prelu<6>` | constexpr mode 6 | resident body |

#### Device 数据流

##### 函数级调用链

```text
InitChannelNcResidentWeightReuse -> copy weight once -> BuildNcResidentWeightVec
-> ProcessChannelNcResidentWeightReuse -> CopyInNcByRows -> ComputeNc -> CopyOutNcByRows
```

##### 源码等价伪码

```text
distribute N rows; load C weights once; duplicate 1 row to rowsPerTile by doubling copies
for N-tile: copy [rows,C] (pad C); vector Mul with resident weightVec; write real C per row
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weightVec | 0 | `ubLength*calcS` | T/FP32 | Init+doubling | all N tiles | core lifetime | 常驻 |
| x/y queues | `n*C` | each `2*ubLength*S` | T/NC | GM/V | V/GM | tile | padded C |
| BF16 weight caches | 0 | `alignedC*(2+4)` | BF16+FP32 | MTE2/Cast | build vec | init | 非BF16不存在 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["inQ 2U·S"]:1
  outQ["outQ 2U·S"]:1
  calc["calc 2U·calcS"]:2
  wvec["weightVec U·calcS"]:1
  wcache["BF16 cache alignedC·6"]:1
```

##### 字节账与复用顺序

```text
UB_total = U*{28 FP32,14 FP16,24 BF16} + (BF16?alignedC*6:0) <= usableUb
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| weightVec | Init | every ComputeNc | Kernel end | PipeBarrier | 构建完成后只读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 权重构建 | MTE2_V/V barrier | GM/Cast | Duplicate/Mul | weightVec | 最后 N/C pad 不写回 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `CalcTiling` | NC tile 字节预算 |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<6>` | resident route |
| Device body/helper | `op_kernel/prelu.h` | `InitChannelNcResidentWeightReuse/BuildNcResidentWeightVec` | 常驻与倍增复制 |

### Route R-003：NC 权重复用

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `CalcTiling` | reuse 条件、`L>1`、整行 NC tile 可容纳 | mode 4 |

#### Device 数据流

##### 函数级调用链

```text
InitChannelNcWeightReuse -> weight GM->UB -> BuildNcWeightVec
-> ProcessChannelNcWeightReuse -> CopyInNcByRows -> ComputeNc -> CopyOutNcByRows
```

##### 源码等价伪码

```text
for c: scalar-load cached weight[c]; FillWeightSpan(weightVec,c*L,L,value)
copy first NCL weight row to remaining tile rows; reuse vector across all assigned N rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weightBuf(+FP32) | 0 | `alignedC*(S + BF16?4:0)` | T/FP32 | Init | builder | core | cached |
| weightVec | `c*L` | `ubLength*calcS` | NCL | builder | ComputeNc | core | across N tiles |
| queues/calc | 0 | base + vector | T/FP32 | Copy/Vector | Vector/Copy | tile | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  io["queues 4U·S"]:2
  calc["calc BF16:12U else:2U·S"]:2
  wvec["weightVec U·calcS"]:1
  wc["weight cache alignedC·(S+BF16?4:0)"]:1
```

##### 字节账与复用顺序

```text
UB_total=U*{28,14,24}+alignedC*{4,2,6} <= usableUb
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| weightVec | BuildNcWeightVec | all ComputeNc | end | V/S + PIPE_V barriers | 完成展开后只读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 展开 | `MTE2_S`/`V_S` + `PIPE_V` | weight cache | scalar read/Duplicate | weightVec | span 首尾用 SetValue |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `GetNcWeightReuseBufferBytesPerElement/CalcTiling` | 预算/选择 |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<4>` | mode 4 |
| Device body/helper | `op_kernel/prelu.h` | `BuildNcWeightVec/FillWeightSpan/ProcessChannelNcWeightReuse` | 展开和复用 |

### Route R-004：NC split-C 权重复用

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `CalcTiling` | 整行 NC 不容纳且 split-C tile/并行 guard 成立 | mode 5 |

#### Device 数据流

##### 函数级调用链

```text
InitChannelNcSplitCWeightReuse -> ProcessChannelNcSplitCWeightReuse
-> CopyWeightTile -> BuildNcWeightVec(1) -> CopyInByOffset -> ComputeNc -> CopyOutByOffset
```

##### 源码等价伪码

```text
task=(cTile,n); if cTile changed: load realC weights and expand each over L
gm=n*C*L+cOffset*L; copy realC*L, compute aligned length, write real length
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weight tile | `cOffset` | `realC`, capacity alignedC | T | GM | builder | cTile | reused across adjacent N tasks |
| x/y | `n*C*L+cOffset*L` | `realC*L` | T | GM/V | V/GM | task | tail pad |
| UB | 0 | same formula as R-003 with tile-local alignedC/U | mixed | Init/Copy/V | V/Copy | core | conditional BF16 buffers |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  io["queues 4U·S"]:1
  calc["calc buffers"]:1
  wt["weight tile"]:1
  wv["expanded weight U·calcS"]:1
  wf["BF16 fp32 weight"]:1
```

##### 字节账与复用顺序

```text
UB_total=U*{28,14,24}+alignedWeightSize*{4,2,6} <= usableUb
workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| weight tile/vector | CopyWeightTile/Build | all adjacent N tasks | cTile changes | task order + barriers | task order groups same cTile |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| weight tile | MTE2_V/V_S/PIPE_V | GM/Cast/build | Mul | cache/vector | last C tile uses realC |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `CalcTiling` lines 339–405 | split-C size/task guards |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<5>` | split-C dispatch |
| Device body/helper | `op_kernel/prelu.h` | `CopyWeightTile/ProcessChannelNcSplitCWeightReuse` | cTile cache |

### Route R-005：split-L 跨核

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `CalcTiling` lines 409–445 | 少行且大 L，task 核数 guard | mode 3 |

#### Device 数据流

##### 函数级调用链

```text
InitChannelSplitLParallel -> ProcessChannelSplitLParallel
-> LoadChannelWeight(on change) -> CopyInByOffset -> Compute -> CopyOutByOffset
```

##### 源码等价伪码

```text
taskIdx -> row=task/tilesPerRow,tile=task%tilesPerRow
load weight[row%C] only when channel changes; process L tile with aligned tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM tile | `row*L+tile*ubLength` | `min(ubLength,L-off)` | T | GM/V | V/GM | task | 32B pad |
| UB | 0 | base bytes `ubLength` | T/FP32 | pipeline | pipeline | tile | same as R-001 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQ 2U·S"]:1
  outQ["outQ 2U·S"]:1
  pos["pos"]:1
  neg["neg + BF16 xFP32"]:1
```

##### 字节账与复用顺序

```text
UB_total=U*{24,12,20}<=usableUb; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| scalar register | LoadChannelWeight | Compute | channel changes | sequential task loop | same channel only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| task | Queue | CopyIn/Compute | Compute/CopyOut | Q slots | final L tile guarded |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `CalcTiling` | task sizing |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<3>` | split-L parallel |
| Device body/helper | `op_kernel/prelu.h` | `ProcessChannelSplitLParallel` | task mapping/cache |

### Route R-006：full-L

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `PreluTilingFunc` | prior flags false且 `innerSizeAligned<=tileLength` | mode 1 |

#### Device 数据流

##### 函数级调用链

```text
InitChannel -> ProcessChannelFullL -> LoadChannelWeight -> CopyInByOffset -> Compute -> CopyOutByOffset
```

##### 源码等价伪码

```text
for assigned row: c=row%C; gm=row*L; load weight[c]; copy L/pad LAligned; compute; write L
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM row | `row*L` | L | T | GM/V | V/GM | row | LAligned compute |
| UB | 0 | base `ubFactor` | T/FP32 | pipeline | pipeline | tile | unused tail capacity allowed |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQ 2U·S"]:1
  outQ["outQ 2U·S"]:1
  pos["pos"]:1
  neg["neg + BF16 xFP32"]:1
```

##### 字节账与复用顺序

```text
UB_total=ubFactor*{24,12,20}<=usableUb; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queues | row Copy/Compute | next stage | next row | Queue lifecycle | row write complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| row | Queue | MTE2/V | V/MTE3 | Q | pad only compute, not output |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `PreluTilingFunc` | full-L key |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<1>` | full-L dispatch |
| Device body/helper | `op_kernel/prelu.h` | `ProcessChannelFullL` | row mapping/tail |

### Route R-007：split-L

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/prelu_tiling.cpp` | `PreluTilingFunc` | prior flags false且 `innerSizeAligned>tileLength` | mode 2 |

#### Device 数据流

##### 函数级调用链

```text
InitChannel -> ProcessChannelSplitL -> per row LoadChannelWeight
-> per L tile CopyInByOffset -> Compute -> CopyOutByOffset
```

##### 源码等价伪码

```text
for row: load weight[row%C]; for off=0;off<L;off+=U: real=min(U,L-off); computeLen=align32(real); process
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM tile | `row*L+off` | real | T | GM/V | V/GM | tile | tail padded |
| UB | 0 | base `ubFactor` | T/FP32 | pipeline | pipeline | tile | reused each L tile |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQ 2U·S"]:1
  outQ["outQ 2U·S"]:1
  pos["pos"]:1
  neg["neg + BF16 xFP32"]:1
```

##### 字节账与复用顺序

```text
UB_total=ubFactor*{24,12,20}<=usableUb; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| all UB buffers | current L tile | CopyOut | next L tile | Queue FreeTensor | current output issued |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| L tail | DataCopyPad + Queue | GM | Vector/GM | valid lanes | padded lanes never written |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/prelu_tiling.cpp` | `PreluTilingFunc` | fallback key |
| Kernel entry/router | `op_kernel/prelu.cpp` | `prelu<2>` | split-L dispatch |
| Device body/helper | `op_kernel/prelu.h` | `ProcessChannelSplitL` | nested row/tile loop |

## 算子特定优化

### Trick T-001：NC 权重向量一次展开、多批复用

#### 触发与源码依据

- 适用 Route：R-002、R-003。
- 触发条件：`preferWeightReuseByRow && C>1 && N>0 && ncMaxTileElements>=alignedRowElements`。
- Golden 来源：[`experimental/activation/prelu/op_kernel/prelu.h:803-819`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_kernel/prelu.h#L803-L819)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
BuildNcWeightVec(rowsPerTile);
for (int64_t rowProgress = 0; rowProgress < blockRowNum; rowProgress += rowsPerTile) {
    int64_t tileRows = blockRowNum - rowProgress;
    tileRows = tileRows > rowsPerTile ? rowsPerTile : tileRows;
    uint32_t computeLen = static_cast<uint32_t>(tileRows * alignedChannelSize);
    int64_t nOffset = rowOffset + rowProgress;
    CopyInNcByRows(nOffset, tileRows);
    ComputeNc(computeLen);
    CopyOutNcByRows(nOffset, tileRows);
}
```

#### 机制与设计原因

`BuildNcWeightVec` 在 batch 循环前把 C 权重按 L 展开，并复制到 `rowsPerTile` 行；后续每个 N tile 直接逐元素 `Mul`，避免逐 row 重新从 GM 装载/展开通道权重。Host 字节预算把 weight cache 和 weightVec 纳入 UB 上界。

#### 迁移边界

- 适用：通道权重、多个 batch 行、展开向量可放入 UB。
- 前提：A2、同 dtype、ND、上述 reuse/UB guard。
- 失效：整行放不下转 R-004；启发式不成立转 R-005–R-007。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：BF16 全计算链升至 FP32

#### 触发与源码依据

- 适用 Route：全部。
- 触发条件：`T=bfloat16_t`。
- Golden 来源：[`experimental/activation/prelu/op_kernel/prelu.h:535-545`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/prelu/op_kernel/prelu.h#L535-L545)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Cast(xFp32, xLocal, RoundMode::CAST_NONE, currentNum);
Maxs(pos, xFp32, 0.0f, currentNum);
Mins(neg, xFp32, 0.0f, currentNum);
Muls(neg, neg, weightValFp32, currentNum);
Add(pos, pos, neg, currentNum);
Cast(yLocal, pos, RoundMode::CAST_RINT, currentNum);
```

#### 机制与设计原因

输入与权重先转/解释为 FP32，正负分支、乘法和加法均以 FP32 完成，最终一次 RINT 降回 BF16；这固定了精度顺序，并由 BF16 的 20/24 bytes-per-element UB 预算支撑。

#### 迁移边界

- 适用：需要复用同类 BF16 piecewise elementwise 计算。
- 前提：预留 FP32 x/pos/neg（NC 路线还含 FP32 weightVec/cache）。
- 失效：FP16/FP32 使用编译期原 dtype 路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 | R-002/003/004 | R-005/006/007 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | scalar | NC reuse 优先 | fallback/并行 L | weight 模式、UB、并行度 |
| 模板与 dtype | mode 0，三 dtype | modes 6/4/5 | modes 3/1/2 | BF16 条件 buffer |
| 循环与分核 | 元素/core | N 或 `(Ctile,N)` | `(row,Ltile)` 或 row | 提高权重复用/核利用 |
| Queue/TBuf | base | base+weight cache/vector | base | Host bytes/element 匹配 |
| 内存总占用与复用 | 12/20/24 B/elem | 14/24/28 B/elem+cache | 同标量 | NC 常驻换 GM 装载 |
| 搬运/Vector API | pad + scalar Muls | pad + Mul vector | pad + scalar Muls | 权重表示不同 |
| 同步、精度与 tail | Queue；BF16 FP32 | 额外 hard event/barrier | Queue；BF16 FP32 | 构建 weightVec 需同步 |
| fallback/失败边界 | UB 太小失败 | 不满足落后续 | R-007 最终 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本卡源码闭合 | 公开稳定 README 声称支持，但固定 Commit 无活动 A3 实现，未发布卡 | 见独立 `_950` 卡 |
| Host/Launcher | experimental `PreluTilingFunc`，7 modes | 固定仓不可见 | stable arch35 Broadcast Tiling |
| Kernel/资源 | 手写 AIV，0 workspace | 固定仓不可见 | atvoss Broadcast AIV/NDDMA |

## 限制与失败边界

- 文档允许空 Tensor，但 Host Tiling 的通道模式要求 weight、N/C/L 正数；单权重空 x 可形成 `usedCoreNum=1, tailLength=0`，Device 循环不执行。
- 文档写 weight rank 0–8，而此 A2 Tiling 明确要求 weight storage rank 为 1；公开 bridge 如何把标量规范为 1-D 在固定仓不可见，调用复用时应保留此边界。
- 所有元素/字节、对齐、task 乘法的显式溢出 guard 不得删；`uint32_t` 搬运长度要求所选 tile 已由 UB 约束。
- A3 缺少 source-closed 实现；未推测复用 A2 或 950。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `experimental/activation/prelu/docs/aclnnPrelu.md` | `aclnnPrelu*` | A2 公开 API、dtype/shape |
| 入口实现 | `experimental/activation/prelu/op_host/CMakeLists.txt` | `ACLNNTYPE aclnn` | 自动 bridge 活动接线 |
| Host/Tiling/Launcher | `experimental/activation/prelu/op_host/prelu_def.cpp` | `Prelu` | A2 OpDef |
| Host/Tiling/Launcher | `experimental/activation/prelu/op_host/prelu_tiling.cpp` | `PreluTilingFunc/CalcTiling` | 七路线 selector/资源 |
| Host/Tiling/Launcher | `experimental/activation/prelu/op_kernel/prelu_tiling_data.h` | `PreluTilingData` | POD ABI |
| Host/Tiling/Launcher | `experimental/activation/prelu/op_kernel/prelu_tiling_key.h` | `PRELU_TPL_*` | key 实例集 |
| Kernel entry/body | `experimental/activation/prelu/op_kernel/prelu.cpp` | `prelu<schMode>` | router |
| Kernel entry/body | `experimental/activation/prelu/op_kernel/prelu.h` | `Prelu<T>` | body、UB、同步、精度、tail |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 14×int64 route-neutral POD | 字段顺序/Route inactive 语义 | `PreluTilingData` |
| 调用顺序 | 单 Kernel Init→Process→Copy/Compute/Copy | constexpr router 顺序 | `prelu<schMode>` |
| workspace/内存 | workspace 0；UB bytes/element | depth=2、1024B reserve、条件 buffers | `CalcTiling/Init*Buffers` |
| 同步 | Queue + NC hard events/barriers | weightVec 生成后才可 Mul | `Sync*/BuildNc*` |
| 精度与尾块 | BF16 FP32→RINT；32B zero pad | 只写 realLen | `Compute/Copy*Pad` |
| 失败边界 | dtype/shape/UB/overflow guards | 不得静默 fallback 越界 | `GetShapeAndDtypeInfo/CalcTiling` |