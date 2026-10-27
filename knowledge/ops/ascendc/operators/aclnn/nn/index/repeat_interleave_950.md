---
type: Operator
title: "RepeatInterleave 950 ACLNN 算子知识"
description: "说明 aclnnRepeatInterleave 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/docs/aclnnRepeatInterleave.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_api/aclnn_repeat_interleave.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_api/aclnn_repeat_interleave.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_api/repeat_interleave.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_normal.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_normal.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_kernel/repeat_interleave_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_kernel/arch35/repeat_interleave.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_kernel/arch35/repeat_interleave_scalar.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_kernel/arch35/repeat_interleave_gather.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_kernel/arch35/repeat_interleave_small.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/repeat_interleave_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/config/ascend950/repeat_interleave_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T16:56:35Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# RepeatInterleave 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnRepeatInterleaveGetWorkspaceSize` + `aclnnRepeatInterleave`，公开的无 dim、Tensor repeats ACLNN；不包含 Int/Tensor/WithDim 命名变体 |
| 计算语义 | 先把 `self` flatten 为 `x[0:N]`；若 repeats 为单元素 `r`，每个 `x[i]` 连续输出 `r` 次；否则每个 `x[i]` 连续输出 `repeats[i]` 次 |
| 输入输出 | `self/out` ND，公开支持 UINT8/INT8/INT16/INT32/INT64/BOOL/FP16/BF16/FP32；`repeats` 为 INT32/INT64 的 0D/1D；`out` 为长度 `outputSize` 的 1D |
| 执行类型 | 非空非恒等路径只有一个 `repeat_interleave` Device 节点，所有 101/102/201/202 路线均为 `AIV_VECTOR`；无 Cube/AIC、无 `MIX_CV` |
| Kernel 路线 | R-001 标量/单元素 repeats（key 101）；R-002 未拆 repeats 的逐元素路线（102）；R-003/R-004 拆 repeats 并行路线（201/202，偏移累加分别用 INT32/INT64） |
| 核心资源 | GM 输入/输出；AIV `TQue`/MicroAPI 寄存器；201/202 使用 user workspace 的每核 128B 记录槽和 `SyncAll`，全部路线另含 16MiB 系统 workspace |
| 来源闭包 | source_closed：公开 ACLNN→连续化/flatten/BOOL cast→`l0op::RepeatInterleave`→950 arch35 Tiling/POD→101/102/201/202 router→全部可达 AIV body/helper→950 OpDef/config/CMake 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `N=numel(self)`，flatten 后记为 `x[0:N]`。若 `repeats` 是 0D 或 1D 单元素，令 `r=repeats[0]`，则 `y` 是每个 `x[i]` 连续复制 `r` 次的串接，`outputSize=N*r`。若 `repeats` 是长度 N 的向量，令前缀和 `p[0]=0, p[i+1]=p[i]+repeats[i]`，则对 `p[i] <= j < p[i+1]` 有 `y[j]=x[i]`，且 `outputSize=p[N]`。公开约束要求 repeats 值为自然数；Kernel 对 `self` 只做同 dtype 数据搬运，不改变数值位模式。

该 ACLNN 固定 flatten 并以 `axis=0` 启动核心算子，因此核心三维折叠只会是单元素 repeats 的 `[N,1,1]` 或逐元素 repeats 的 `[1,N,1]`；arch35 body 中一般 axis/copy-plane 大于 1 的分支对本入口不可达，不作为本卡 Route。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 0–8D、ND，可非连续、可为空 | UINT8、INT8、INT16、INT32、INT64、BOOL、FP16、BF16、FP32 | 先连续化；0D 先广播为 `[1]`，随后 flatten 成 `[N]` |
| `repeats` | 0D 或 1D、ND，可非连续、可为空 | INT32、INT64 | 1D 长度只能为 1 或 N；固定源码不在 Device 侧检查负值，公开契约要求自然数 |
| `outputSize` | Host 标量 | INT64 | 单元素 repeats 时应为 `N*repeats[0]`，向量时应为 repeats 总和；实现用它分配核心临时输出并和 `out.shape` 比较，不读取 Device repeats 验证和值 |
| `out` | 1D、ND，可非连续 | 与 `self` 相同 | 长度必须与 `outputSize` 及核心临时结果一致；最终经 `ViewCopy` 提交 |
| `workspaceSize` | Host 输出 | UINT64 指针 | executor 中全部框架节点与核心节点所需 workspace 总量 |
| `executor` | Host 输出 | `aclOpExecutor**` | 持有前处理、核心 launch 和输出提交节点；第二段在给定 stream 上运行 |

### 入口与版本

固定仓公开文档和头文件都提供两段式 `aclnnRepeatInterleave`；本卡只覆盖其 `(self, repeats Tensor, outputSize, out)` 语义。相邻的 `aclnnRepeatInterleaveInt`、`aclnnRepeatInterleaveTensor`、`aclnnRepeatInterleaveWithDim`、`aclnnRepeatInterleaveIntWithDim` 是不同公开语义，未并入版本族。公开产品表声明 A2/A3/950 均支持；950 的 `RepeatInterleave` OpDef、binary config、CMake 与 arch35 Host/Kernel 同时可见并闭合。A2/A3 的核心实现不在该固定仓公开源码中，故不能用 950 body 外推。

`op_api/aclnn_repeat_interleave.h` 的注释仍把 repeats 写为 INT64，而同 Commit 的公开文档、ACLNN `CheckParams`、OpDef 与 950 binary config均明确接入 INT32/INT64；本卡记录这一固定源码内的声明注释差异，以公开文档和实际校验/接线为支持范围依据。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先检查 `workspaceSize/executor`，创建 executor，然后按顺序执行：`self/repeats/out` 非空；按 950 `GetDtypeSupportList()` 校验 self；repeats 只允许 INT32/INT64；out/self dtype 相同；repeats 最大 1D；其 1D 长度为 1 或 N；self 最大 8D；self 为空时 repeats 只能为空、0D 单元素或 1D 单元素。`self.IsEmpty() || out.IsEmpty()` 直接返回 workspace 0；非空 self 配空 repeats 返回参数错误。

普通路径对 self/repeats 调 `Contiguous`；0D 由 `BroadcastTo([1])` 变成 1D。若 `outputSize==N` 且 repeats 只有一个元素，只把 self reshape 为 `[N]` 并 `ViewCopy`，不读取 repeats 值、不启动核心 Kernel。否则 BOOL self 先 Cast 为 INT8，self reshape 为 `[N]`，调用 `l0op::RepeatInterleave(flattenSelf,repeats,axis=0,outputSize)`；核心输出 shape 与 out 不同则失败；BOOL 结果 Cast 回 BOOL，最后 `ViewCopy`。

### 参数、Workspace 与生命周期

`l0op::RepeatInterleave` 按 `flattenSelf` dtype/format 分配长度 `outputSize` 的临时 y，并把 `(x,repeats)->y, axis=0` 加入 AICore launcher。Tiling 从平台取得 AIV core 数 C 和原始 UB 大小 U。全部核心路线设置 workspace 为 `16MiB + 128*C`；entry 先 `SetSysWorkspace(workspace)`，再通过 `GetUserWorkspace` 获得用户段，任一指针为空即返回。101/102 不读用户段，201/202 把其起点视为 `recordSumGm`，每个 block 使用以 128B cache line 隔开的一个标量槽。固定源码未公开系统段与用户段的运行时绝对 offset，只闭合了总量和 user pointer 边界。

executor 持有连续化/广播/Cast/Reshape 临时 Tensor、核心 y、ViewCopy 以及 workspace 到第二段执行结束；第二段 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)` 在同一 stream 依序运行。框架生成节点各自的 workspace slice、内部 Kernel 和事件属于允许的框架边界，固定仓不可见。

### 启动图与失败传播

```text
aclnnRepeatInterleaveGetWorkspaceSize
  -> CheckParams / empty-or-error terminal
  -> Contiguous(self,repeats); optional BroadcastTo; Reshape(self,[N]); optional Cast(BOOL->INT8)
  -> l0op::RepeatInterleave
     -> ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleave, x, repeats, y, axis=0)
     -> one Device node: repeat_interleave(x,repeats,y,workspace,tiling)
        blockDim=usedCoreNum, localMemorySize=Tiling-selected UB, POD=Norm or Small
  -> optional Cast(INT8->BOOL) -> ViewCopy(y,out)
aclnnRepeatInterleave -> CommonOpExecutorRun(all nodes, same stream)
```

核心是单 Kernel launch graph，没有 Kernel 间 workspace 边。201/202 的 `recordSumGm` 是同一 Kernel 内跨核边：各核写 `recordSumGm[blockIdx*(128/sizeof(U))]`，DCCI clean/invalidate 后 `SyncAll`，再由每核搬入全部槽并归约本 batch 之前的 slice 和；owner 为调用者 workspace/executor，生命周期覆盖该 Kernel。launcher 添加失败使 l0op 返回空并终止第一段；任一前处理、shape 检查或 ViewCopy 返回空也立即返回错误；第二段返回执行器的运行状态。

## Route 与参数设计

### 分核与资源计算

对本 ACLNN，950 Tiling 得到 `[B,R,CP]`：单元素 repeats 为 `[N,1,1]`，逐元素 repeats 为 `[1,N,1]`。先令 `eachBatch=ceil(B/C)`、`used=ceil(B/eachBatch)`、`tailBatch=B-eachBatch*(used-1)`。通用代码只在 `used<C/2 && CP*sizeof(T)>128` 时拆 CP；本入口 `CP=1`，该分支恒不命中。随后若 `used<C/2 && (R>64 || (CP==1 && R!=1)) && outputSize>1024`，把 repeats 轴拆为 `slice=ceil(R/ceil(R/(C/used)))` 份，`used=used_before*slice`，主片 `normalR=ceil(R/(C/used_before))`，尾片 `tailR=R-normalR*(slice-1)`。

`repeatsCount=outputSize/inputShape[0]` 仅对 0D/长度 1 repeats 产生，选择 101；向量不拆选 102；向量拆分且 repeats 为 INT32 且总输出元素不超过 INT32_MAX 选 201，否则 202。`averageRepeatTime=outputSize/inputShape[0]` 是整数除法。普通路线 `F=floor32(U/2/(2*sizeof(T)+sizeof(U)))`；拆分路线先令 `U'=U-2*C*32`，再取 `F=floor32((U'-32*sizeof(V))/(2*sizeof(T)+sizeof(U)))`，其中 V 为 201 的 INT32 或 202 的 INT64。`PostTiling` 设置 `blockDim=used`、schedule mode 1 和 local memory size（普通为 U，拆分为 U'）。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 单元素 repeats | 101 / `RepeatInterleaveScalarImpl<T,U>` | 非空、非 ACLNN 恒等快路，且 repeats 为 0D 或 1D 长度 1；`repeatsCount=outputSize/N>-1` | repeats 长度 N 时检查拆分 |
| 2 | R-002 普通逐元素 | 102 / `RepeatInterleaveImpl<T,U>` | 向量 repeats 且拆分谓词不成立 | 拆分谓词成立时按位宽去 R-003/R-004 |
| 3 | R-003 拆 repeats INT32 累加 | 201 / `RepeatInterleaveSmall<T,U,int32_t>` | 向量 repeats；拆分谓词成立；`repeats.dtype==INT32 && outputSize<=INT32_MAX` | INT64/大输出去 R-004 |
| 4 | R-004 拆 repeats INT64 累加 | 202 / `RepeatInterleaveSmall<T,U,int64_t>` | 向量 repeats；拆分谓词成立；且上一 INT32 安全谓词不成立 | 无；有效拆分输入命中 |

```text
if repeats is scalar or length 1:
    key = 101
else if used<C/2 && (R>64 || (CP==1 && R!=1)) && outputSize>1024:
    key = 201 if repeats is INT32 and numel(y)<=INT32_MAX else 202
else:
    key = 102
```

空 self/out 和单元素 repeats 的 `outputSize==N` 在 ACLNN 层提前返回，不进入 key 决策。表按逻辑优先级表达；源码 `GetTilingKey` 先判断拆分 201/202，再判断 `repeatsCount`，两者在 `MergDim/GetUbFactor` 产生条件上互斥。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 单元素 repeats | 101 | 950；非空、非恒等；0D/长度1 repeats；ACLNN 下 CP=1 | 单 AIV Kernel；每核处理 N 的 batch 段 | `RepeatInterleaveScalarImpl<DTYPE_X,DTYPE_REPEATS>`；tiny-CP gather 或 unaligned replication | 三个 depth1 Queue；MTE3→V event；无 user workspace 消费 | `r*sizeof(T)<=256` 走 gather，否则逐 cp 分块；最后核 `tailBatch` |
| R-002 普通逐元素 | 102 | 950；repeats 长 N；拆分谓词为 false；CP=1 | 单 AIV Kernel；ACLNN 下通常 1 核依次读 repeats 并输出 | `RepeatInterleaveImpl<T,U>` | repeats depth1、x/y depth2；MTE2→S 读标量 | repeats 按 F 分段重载；单个 repeat 超过 UB 输出容量时循环提交；0 次跳过 |
| R-003 拆 repeats INT32 累加 | 201 | 950；向量；拆分谓词 true；U=INT32 且 outputSize<=INT32_MAX | 单 AIV Kernel；同 batch 的多个核各处理 repeats slice，Kernel 内前缀定位 | `RepeatInterleaveSmall<T,U,int32_t>` | 5 个 depth1 `TQue`；128B/core workspace；DCCI + SyncAll；V→S/MTE2→V | CP=1；按当前片 repeatCount、repeat sum 和 F 分组/拆大值 |
| R-004 拆 repeats INT64 累加 | 202 | 同 R-003，但 repeats 为 INT64 或 y 元素数>INT32_MAX | 同 R-003 | `RepeatInterleaveSmall<T,U,int64_t>` | 本地 offset/reduction RegTensor 和 outputOffset buffer 升为 INT64；跨核 record 仍为 U | 同 R-003；record 位宽边界见限制 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| ACLNN 恒等快路 | `outputSize==N && numel(repeats)==1` | 直接 Reshape+ViewCopy，省去核心复制 | 无 Kernel | 正常核心路线 | `aclnnRepeatInterleaveGetWorkspaceSize` |
| repeats 轴并行 | `used<C/2 && (R>64 \|\| (CP==1 && R!=1)) && outputSize>1024` | batch/CP 不足以占核时，增加 repeats slice；需跨核前缀和 | R-003/R-004 | R-002 | `GetUbFactor`, `SplitRepeatsShape` |
| CP 拆分 | `used<C/2 && CP*sizeof(T)>128` | 大连续 copy plane 可按 CP 分核 | 通用实现；本 ACLNN 不可达 | 保持 batch/repeats 分配 | `GetUbFactor`, `SplitCP` |
| offset 位宽 | `repeats.dtype==INT64 \|\| numel(y)>INT32_MAX` | 归约和输出 offset 不能窄化为 INT32 | 201/202 | 选 201 或 202 | `UseInt64`, `GetTilingKey` |
| scalar gather | `CP*sizeof(T)<32 && CP*r*sizeof(T)<=256`；本入口 CP=1 | 用寄存器 gather 一批生成多个重复值，8bit 另用宽载入+DeInterleave | R-001 | unaligned copy loop | `ProcessWholeCp`, `CopyCpToRepeatOutGatherAicore` |
| 普通 UB factor | `F=floor32(U/2/(2sT+sU))` | 覆盖 Queue depth 后为输入、输出、repeats 留上界 | R-001/R-002 | 无 fallback；F 为 Device 除数 | `GetUbFactor` |
| 拆分 UB/同步预留 | `U'=U-64C`，F 再扣 `32*sizeof(V)` | 为每核记录槽、归约输出以及 input/output/repeats buffer 留空间 | R-003/R-004 | 不满足资源时固定源码无显式 fallback | `GetUbFactor`, `RepeatInterleaveSmall::Init` |
| 大 repeat 值 | `repeatTimes>repeatsFactorSum` | 单个元素的全部复制结果放不进 output Queue | R-003/R-004 | `IndexRepeatLarge` 多次提交 | `CopyOutSplitImpl` |

### POD/ABI

两种结构均由 `TILING_DATA_FIELD_DEF` 按声明顺序序列化，Host `SaveToBuffer` 与 Device `GET_TILING_DATA_WITH_STRUCT` 使用同名结构。所有字段是 int64_t；固定仓没有展开生成宏后的 `sizeof/alignof/offsetof` 或 static_assert，故 ABI 布局以声明顺序为准，精确 byte offset 标为固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `RepeatInterleaveTilingKernelDataNorm` | `totalCoreNum` | int64/64 | 声明序 0；offset 不可见 | core | 平台 AIV C | 仅 POD 保留 | >0 |
| 同上 | `usedCoreNum` | int64/64 | 序 1 | core | batch/CP 分核结果 | launch blockDim 间接一致 | <=C |
| 同上 | `eachCoreBatchCount` | int64/64 | 序 2 | batch/core | `ceil(B/C)` | `ProcessWholeCp`, start offset | >=1 |
| 同上 | `tailCoreBatchCount` | int64/64 | 序 3 | batch | B 的余量 | 最后一核 | 1..each |
| 同上 | `isSplitCP` | int64/64 | 序 4 | bool | `SplitCP` | `Process` 分支 | ACLNN 恒为0 |
| 同上 | `normalCP` | int64/64 | 序 5 | element | `ceil(CP,ratio)` | split-CP 主核 | inactive for ACLNN |
| 同上 | `tailCP` | int64/64 | 序 6 | element | CP 余量 | split-CP 尾核 | inactive for ACLNN |
| 同上 | `cpSlice` | int64/64 | 序 7 | slice/batch | `ceil(CP/normalCP)` | block→batch/cp 映射 | inactive for ACLNN |
| 同上 | `ubFactor` | int64/64 | 序 8 | 混合 buffer 元素容量 | `GetUbFactor` 向下 32 对齐 | Queue 容量、tile/repeats chunk | 必须非零；Host 未显式检查 |
| 同上 | `repeatsCount` | int64/64 | 序 9 | repeat/element | 单元素时 `outputSize/inputShape[axis]`，否则 -1 | scalar body 的 r；normal body scalar flag | key101 时 >=0 |
| 同上 | `totalRepeatSum` | int64/64 | 序 10 | output axis element | `yShape[axis]` | 向量 batch 输出起点 | =outputSize |
| 同上 | `mergedDims[0]` | int64/64 | 序 11 数组元素0 | batch | `MergDim` | 分核/输入输出 offset | ACLNN：N 或1 |
| 同上 | `mergedDims[1]` | int64/64 | 序 11 数组元素1 | repeat-axis element | `MergDim` | repeats modulo/循环 | ACLNN：1或N |
| 同上 | `mergedDims[2]` | int64/64 | 序 11 数组元素2 | CP element | `MergDim` | copy length/stride | ACLNN恒为1 |
| `RepeatInterleaveTilingKernelDataSmall` | `totalCoreNum` | int64/64 | 声明序 0；offset 不可见 | core | 平台 C | UB/workspace公式间接来源 | >0 |
| 同上 | `usedCoreNum` | int64/64 | 序 1 | core | `used_before*repeatsSlice` | workspace搬运、active guard | <=C |
| 同上 | `eachCoreBatchCount` | int64/64 | 序 2 | batch/core | 初始 batch 分核 | POD 保留 | ACLNN batch=1 |
| 同上 | `tailCoreBatchCount` | int64/64 | 序 3 | batch | 初始余量 | POD 保留 | ACLNN为1 |
| 同上 | `normalRepeatsCount` | int64/64 | 序 4 | repeats/slice | `ceil(R,ratio)` | x/repeats GM 起点、主片长度 | >0 |
| 同上 | `tailRepeatsCount` | int64/64 | 序 5 | repeats/slice | R 余量 | 最后 slice | 1..normal |
| 同上 | `repeatsSlice` | int64/64 | 序 6 | slice/batch | `ceil(R/normalR)` | block→batch/slice | >1 on split route |
| 同上 | `ubFactor` | int64/64 | 序 7 | buffer 元素容量 | 拆分 UB 公式 | Queue、chunk、group | 32 对齐 |
| 同上 | `totalRepeatSum` | int64/64 | 序 8 | output axis element | y axis | batch 输出基址 | =outputSize |
| 同上 | `averageRepeatTime` | int64/64 | 序 9 | repeat/element | 整除 `outputSize/N` | group mainLen | N>0；可能为0 |
| 同上 | `mergedDims[0]` | int64/64 | 序 10 数组元素0 | batch | `MergDimForTensor` | batchNO 输出基址 | ACLNN为1 |
| 同上 | `mergedDims[1]` | int64/64 | 序 10 数组元素1 | repeats | 同上 | split threshold/offset | ACLNN为N |
| 同上 | `mergedDims[2]` | int64/64 | 序 10 数组元素2 | CP element | 同上 | copy stride | ACLNN为1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 101 | `RepeatInterleaveTilingKernelNorm::{GetUbFactor,GetTilingKey,PostTiling}` | Norm POD | `repeat_interleave_apt.cpp::repeat_interleave` 101 分支 | `RepeatInterleaveScalarImpl` + gather helper | 950 OpDef、binary JSON、CMake、ascendc config |
| R-002 | 102 | 同上，向量未拆 | Norm POD | 同 entry 102 分支 | `RepeatInterleaveImpl::{Init,ProcessWholeCp,...}` | 同上 |
| R-003 | 201 | `SplitRepeatsShape`, `UseInt64`, `GetTilingKey`, Small `PostTiling` | Small POD，V=INT32 | 同 entry 201 分支 | `RepeatInterleaveSmall<T,U,int32_t>` | 同上 |
| R-004 | 202 | 同上，`isUseInt64=1` | Small POD，V=INT64 | 同 entry 202 分支 | `RepeatInterleaveSmall<T,U,int64_t>` | 同上 |

## Kernel 实现

### 公共执行流程

唯一 entry 检查 workspace/system/user workspace，构造 `TPipe`，标记 `KERNEL_TYPE_MIX_AIV_1_0`，再按 201→202→101→102 的互斥 key 解析 POD 并实例化 body。该标记的可达代码只有 AIV `DataCopyPad`、LoopMode 和 Vector/MicroAPI，没有 AIC/Cube 或 `MIX_CV` 节点。数据 T/U 直接来自 binary config 的 x/repeats dtype；201/202 的 V 只控制 repeats 总和与 offset 的计算位宽。未知 key 不写 y，固定 entry 没有显式错误返回。

### Route R-001：单元素 repeats

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_repeat_interleave.cpp` | `aclnnRepeatInterleaveGetWorkspaceSize` | repeats 0D 经 BroadcastTo 成 `[1]`；1D 长度1保持 | 核心 x 为 `[N]`, axis=0 |
| Tiling | `repeat_interleave_tiling_base.cpp` | `MergDimForScalar` | repeats 0D 或 dim0==1 | merged=`[N,1,1]` |
| Tiling | `repeat_interleave_tiling_normal.cpp` | `GetUbFactor/GetTilingKey` | `repeatsCount=outputSize/N > -1` | key101 |
| Router | `repeat_interleave_apt.cpp` | `repeat_interleave` | `TILING_KEY_IS(101)` | `RepeatInterleaveScalarImpl<T,U>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::RepeatInterleave -> repeat_interleave<101>
  -> RepeatInterleaveScalarImpl::Init -> ComputeStartOffset
  -> Process -> ProcessWholeCp                    # ACLNN isSplitCP=0
     -> tiny cp: ProcessCpMatchToUbGather
        -> CopyInX -> CopyXToMatchOutGather
           -> CopyCpToRepeatOutGatherAicore -> MicroAPI gather/deinterleave -> CopyMatchOutToY
     -> otherwise: ProcessCpMatchToUb -> CopyInX -> CopyXToMatchOut
        -> CopyOneCpToRepeatOut/CopyOneCpToRepeatOutLarge -> CopyMatchOutToY
```

##### 源码等价伪码

```text
# ACLNN specialization: B=N,R=1,CP=1,r=outputSize/N
core owns [start,end) from eachCoreBatchCount/tailBatch
for chunks of at most floor(F/CP) input elements:
  DMA x[start:start+count] -> xIn
  if sizeof(T)<32 and r*sizeof(T)<=256:
    derive gather src index floor(idx/(r*CP))*CP + idx%CP
    gather several cp groups per register; 8-bit path deinterleaves widened lanes
  else:
    for each input element, load its CP elements to RegTensor and emit r copies
    if r*CP>F, split r into floor(F/CP)-sized output chunks
  after MTE3->V reuse wait, DMA valid generated elements to y at start*r
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `startCpIdx`，ACLNN CP=1 | core batch count elements | T/[N] | caller | `CopyInX` | caller/executor | DataCopyPad tail |
| y GM | `batchStart*r + localRepeatOffset` | valid repeats*CP | T/[outputSize] | MicroAPI+MTE3 | out/ViewCopy | executor temp | only valid length written |
| `xInQueue_` | UB allocator | `F*sizeof(T)` | T | GM→UB | gather/unaligned copy | depth1, per chunk | Free after chunk |
| `xOutQueue_` | UB allocator | `F*sizeof(T)` | T | MicroAPI | UB→GM | depth1, repeatedly DeQue/EnQue | MTE3→V before overwrite |
| `repeatsQueue_` | UB allocator | `F*sizeof(U)` | U | 未写 | 未读 | depth1 reserved for shared layout | scalar value comes from POD |
| RegTensor/UnalignReg | registers | one vector register group | T/index | xIn | xOut | VF call | no GM lifetime |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  repeatsQ["repeatsQueue · depth=1 · F*sizeof(U) · inactive payload"]:1
  xInQ["xInQueue · depth=1 · F*sizeof(T)"]:1
  xOutQ["xOutQueue · depth=1 · F*sizeof(T)"]:1
```

##### 字节账与复用顺序

```text
F = floor_to_32(U / 2 / (2*sizeof(T)+sizeof(U)))
UB_total = F*(sizeof(U)+2*sizeof(T))
UB_total <= U/2 <= U
dynamic_ub = N/A（AIV localMemorySize=U）
workspace_total(core node) = 16MiB + 128*C；本 Route 不消费 user slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xIn | `CopyInX` | gather/CopyOneCp | 下一 chunk | `DeQue` 后计算，末尾 `FreeTensor` | 当前 chunk 全部复制完 |
| xOut | Vector/MicroAPI | `CopyMatchOutToY` | 下一 output chunk | `MTE3_V` event + Queue 状态 | 前次 UB→GM 已提交完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→xIn | Queue EnQue/DeQue | MTE2 | Vector | xIn chunk | DataCopyPad 用精确 byte length |
| xOut 复用 | `SetFlag/WaitFlag<MTE3_V>` | 上次 UB→GM | 下一次 Vector 写 | xOut | 防止提前覆盖 |
| gather | `DataCopyGather`, mask, unaligned store | xIn | xOut | register lanes | 8bit 宽载后 DeInterleave；tail mask=`tailOutElems` |
| 普通复制 | `DataCopyUnAlign` | xIn RegTensor | xOut | CP=1 值 | 无算术 cast；r 以 uint16_t 传入，超范围风险见限制 |
| 核尾 | `tailCoreBatchCount` | Tiling | 最后 block | 输入范围 | 仅有效 count 输出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/repeat_interleave_tiling_normal.cpp` | `GetUbFactor/GetTilingKey/PostTiling` | scalar 值、F、核数与 key101 |
| Kernel entry/router | `op_kernel/repeat_interleave_apt.cpp` | `repeat_interleave`, key101 | Norm POD→Scalar body |
| Device body/helper | `op_kernel/arch35/repeat_interleave_scalar.h` | `RepeatInterleaveScalarImpl::ProcessWholeCp` | gather/普通复制分流、循环、Queue/Event |
| Device body/helper | `op_kernel/arch35/repeat_interleave_gather.h` | `CopyCpToRepeatOutGatherAicore` | gather index、8bit 去交织和动态 tail |

### Route R-002：普通逐元素 repeats

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `aclnn_repeat_interleave.cpp` | `CheckRepeatsSize` | repeats 1D 长 N | flatten x, axis0 |
| Tiling | `repeat_interleave_tiling_base.cpp` | `MergDimForTensor` | repeats length !=1 | merged=`[1,N,1]` |
| Tiling | `repeat_interleave_tiling_normal.cpp` | `GetUbFactor/GetTilingKey` | repeats split guard false，`repeatsCount==-1` | key102 |
| Router | `repeat_interleave_apt.cpp` | `repeat_interleave` | key102 | `RepeatInterleaveImpl<T,U>` |

#### Device 数据流

##### 函数级调用链

```text
repeat_interleave<102> -> RepeatInterleaveImpl::Init -> ComputeStartOffset
  -> Process -> ProcessWholeCp -> ProcessCpMatchToUb
     -> CopyInRepeats as needed
     -> CopyInX(chunk) -> CopyXToMatchOut
        -> ComputeRepeatsOnCurDim (MTE2->S GetValue)
        -> CopyOneCpToRepeatOutVf -> CopyMatchOutToY
```

##### 源码等价伪码

```text
# ACLNN specialization: B=1,R=N,CP=1, normally one block
load up to F repeats beginning at current i
for x chunk of up to F elements:
  DMA x chunk -> xIn
  for each i in chunk:
    reload repeats window if i leaves [head,tail)
    wait MTE2->S; r = repeatsLocal[i-head]
    if r==0: advance x source only
    else for ceil(r/F) chunks:
      load x[i] once into RegTensor; emit min(F,r-left) copies into xOut
      DMA valid xOut sequence to y[outStart+written]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| repeats GM | `i%N` | `min(F,N-offset)` | U/[N] | caller | CopyInRepeats | caller | windowed |
| x GM | `i` | <=F elements | T/[N] | caller | CopyInX | caller | exact bytes |
| y GM | `copyToGmNum` | sum of valid r | T/[outputSize] | CopyMatchOutToY | ViewCopy | executor | compact |
| repeats Queue | UB | `F*sizeof(U)` depth1 | U | MTE2 | scalar GetValue | per window | MTE2→S before scalar |
| x input Queue | UB | `2*F*sizeof(T)` | T | MTE2 | Vector | depth2 | Free per x chunk |
| x output Queue | UB | `2*F*sizeof(T)` | T | Vector | MTE3 | depth2 | alloc/free per result chunk |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  repeatsQ["repeatsQueue · depth=1 · F*sizeof(U)"]:1
  xInQ["xInQueue · depth=2 · 2F*sizeof(T)"]:2
  xOutQ["xOutQueue · depth=2 · 2F*sizeof(T)"]:2
```

##### 字节账与复用顺序

```text
F = floor_to_32(U / 2 / (2*sizeof(T)+sizeof(U)))
UB_total = F*sizeof(U) + 2F*sizeof(T) + 2F*sizeof(T)
         = F*(sizeof(U)+4*sizeof(T)) <= U
dynamic_ub = N/A（AIV localMemorySize=U）
workspace_total = 16MiB + 128*C；user slice inactive
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| repeats Queue slot | DataCopyPad | `GetValue` | 下一 window | MTE2→S + FreeTensor | 目标 i 不在 window 才重载 |
| x input depth slot | CopyInX | CopyXToMatchOut | 后续 chunk | Queue FreeTensor | 所有 r 次复制完成 |
| x output depth slot | VF | CopyMatchOutToY | 后续 r chunk | DeQue/FreeTensor | valid count 已 DMA |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| repeats 标量读取 | `MTE2_S` flag | DataCopyPad | `GetValue` | 当前 U 元素 | INT32/INT64 原样读取 |
| x/y pipeline | Queue | MTE2/Vector/MTE3 | DeQue/Free | depth2 slots | DataCopyPad 精确 byte tail |
| r=0 | branch | scalar read | x source advance | output offset | 不生成输出 |
| r 大于 F | loop | one x RegTensor | 多个 GM chunk | output capacity | 每次 uint16 repeatTimes，见限制 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `repeat_interleave_tiling_normal.cpp` | `GetUbFactor/GetTilingKey` | 不拆时 key102、F 和 totalRepeatSum |
| Kernel entry/router | `repeat_interleave_apt.cpp` | key102 branch | Norm POD→`RepeatInterleaveImpl` |
| Device body/helper | `repeat_interleave.h` | `Init/ProcessWholeCp/ProcessCpMatchToUb` | repeats window、逐 r 复制、GM offset/tail |

### Route R-003：拆 repeats，INT32 offset

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `repeat_interleave_tiling_normal.cpp` | `GetUbFactor` | 低占核、R/CP条件且 outputSize>1024 | `SplitRepeatsShape` |
| Tiling | 同上 | `UseInt64/GetTilingKey` | repeats INT32 且 y numel<=INT32_MAX | key201 |
| Router | `repeat_interleave_apt.cpp` | key201 branch | Small POD | `RepeatInterleaveSmall<T,U,int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
repeat_interleave<201> -> RepeatInterleaveSmall::Init
  -> Process -> CalcRepeatsPartStart -> ComputeOutputOffset -> CustomReduceSum<int32>
  -> CalcOutputStartOfset: recordSumGm write -> DCCI -> SyncAll -> gather records -> prefix ReduceSum
  -> CopyIn/CopyInAlign -> CopyOut/CopyOutSplit/CopyOutY
     -> IndexRepeatGroupVf or IndexRepeatVf/IndexRepeatLarge -> CopyOutData
```

##### 源码等价伪码

```text
slice = block % repeatsSlice; count = normalR or tailR
sum_this = reduce(repeats[sliceStart:sliceStart+count]) in INT32
recordSumGm[block*128B] = sum_this; clean cache; SyncAll
load one scalar per active core with 128B srcStride
prefix = reduce(record sums from this batch's first slice to block-1)
outputBase = batch*outputSize + prefix
if CPbytes>256: aligned CopyIn then per element LoopMode copy   # ACLNN CP=1: unreachable
else group repeats while input count and output repeat sum fit F:
  copy x and repeats to UB
  if group sum fits output capacity: one IndexRepeatGroupVf and DMA
  else split groups; a single r>capacity goes through IndexRepeatLarge
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| user workspace records | `blockIdx*128B` | `usedCoreNum*128B` active，预留 C*128B | U，标量/slot | each block + DCCI | all blocks prefix | caller workspace, Kernel lifetime | cacheline isolated |
| repeats Queue | UB | `F*sizeof(U)` depth1 | U | GM DMA | reduce/IndexRepeat | per chunk | exact valid count |
| input Queue | UB | `F*sizeof(T)` depth1 | T | GM DMA | IndexRepeat | per chunk | CP=1 compact |
| output Queue | UB | `F*sizeof(T)` depth1 | T | MicroAPI | GM DMA | per output group | large r splits |
| outputOffset buffer | UB | `32*sizeof(int32)` depth1 | INT32 | ReduceSum | scalar GetValue | per reduction | V→S |
| recordOffset buffer | UB | `GetBlockNum()*32B` depth1 | packed U records | workspace DMA | prefix ReduceSum | prefix stage | each record padded to 32B |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  repeatsQ["repeatsQueue · depth=1 · F*sizeof(U)"]:1
  inputQ["inputQueue · depth=1 · F*sizeof(T)"]:1
  outputQ["outputQueue · depth=1 · F*sizeof(T)"]:1
  sumBuf["outputOffsetBuf · 32*sizeof(int32)"]:1
  records["recordOffsetBuf · usedCoreNum*32B"]:1
```

##### 字节账与复用顺序

```text
U' = U - 64*C
F = floor_to_32((U' - 32*sizeof(int32))/(2*sizeof(T)+sizeof(U)))
UB_total = F*(sizeof(U)+2*sizeof(T)) + 32*sizeof(int32) + usedCoreNum*32
UB_total <= U' + usedCoreNum*32 <= U - 32*C <= U
dynamic_ub = N/A（AIV localMemorySize=U'）
workspace_total = 16MiB + 128*C
user_workspace_used = 128*usedCoreNum bytes, one scalar per cache line
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| workspace record slot | local block | all blocks | Kernel end | DCCI + `SyncAll` | one writer/slot；128B isolation |
| outputOffset buffer | CustomReduceSum | scalar GetValue | next reduction | V→S + FreeTensor | previous scalar consumed |
| input/repeats Queue | CopyIn | IndexRepeat | next chunk | Queue FreeTensor | output materialized before free |
| output Queue | IndexRepeat | CopyOutData | next group | DeQue/FreeTensor | only valid repeat sum copied |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| local reduce | MicroAPI `ReduceSum/Add` | repeats UB | outputOffset | V RegTensor | INT32；tail mask由 dataLen |
| cross-core record | DCCI + `SyncAll` | every block | record DMA | user workspace | active core count；128B stride |
| DMA→Vector | `MTE2_V` | record DataCopyPad | prefix reduce | recordOffset UB | pad to 32B per record |
| Vector→scalar | `V_S` | reduction | GetValue | sum scalar | prefix length may be zero |
| output tail | group/single-r loops | MicroAPI | CopyOutData | output queue | tailLoopCount/repeat sum exact |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `repeat_interleave_tiling_normal.cpp` | `SplitRepeatsShape/UseInt64/PostTiling` | slice、V位宽、F、Small POD |
| Kernel entry/router | `repeat_interleave_apt.cpp` | key201 | V=int32 |
| Device body/helper | `repeat_interleave_small.h` | `Init/CalcOutputStartOfset/Process` | workspace prefix、group/split、大值与 tail |

### Route R-004：拆 repeats，INT64 offset

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `repeat_interleave_tiling_normal.cpp` | `UseInt64` | repeats dtype INT64 或 y numel>INT32_MAX | `isUseInt64=1` |
| Tiling | 同上 | `GetTilingKey` | isSplitShape 且 isUseInt64 | key202 |
| Router | `repeat_interleave_apt.cpp` | key202 branch | Small POD | `RepeatInterleaveSmall<T,U,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
repeat_interleave<202> -> RepeatInterleaveSmall<T,U,int64_t>::Init
  -> Process -> CalcRepeatsPartStart / CalcOutputStartOfset
  -> CopyIn -> CopyOut or CopyOutSplit -> IndexRepeat* -> CopyOutData
```

##### 源码等价伪码

```text
与 R-003 相同的 block→slice、workspace prefix 和分组复制；
所有 CustomReduceSum 的累加寄存器、outputOffsetBuf 元素和 scalar prefix 使用 int64_t，
避免本地归约与输出 offset 的 32-bit 窄化；但跨核 `recordSumGm_` 仍以 U 存储，见限制。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace records | `blockIdx*128B` | `used*128B` active | U scalar/slot | each block | prefix | Kernel | 同 R-003 |
| input/output/repeats Queue | UB allocator | F*(2sT+sU) | T/U | DMA/Vector | Vector/DMA | per chunk | depth1 |
| outputOffset buffer | UB allocator | `32*sizeof(int64)` | INT64 | ReduceSum | GetValue | per reduction | V→S |
| recordOffset buffer | UB allocator | `used*32B` | padded U | workspace DMA | prefix | prefix stage | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  repeatsQ["repeatsQueue · depth=1 · F*sizeof(U)"]:1
  inputQ["inputQueue · depth=1 · F*sizeof(T)"]:1
  outputQ["outputQueue · depth=1 · F*sizeof(T)"]:1
  sumBuf["outputOffsetBuf · 32*sizeof(int64)"]:1
  records["recordOffsetBuf · usedCoreNum*32B"]:1
```

##### 字节账与复用顺序

```text
U' = U - 64*C
F = floor_to_32((U' - 32*sizeof(int64))/(2*sizeof(T)+sizeof(U)))
UB_total = F*(sizeof(U)+2*sizeof(T)) + 32*sizeof(int64) + usedCoreNum*32 <= U
dynamic_ub = N/A（AIV localMemorySize=U'）
workspace_total = 16MiB + 128*C；active records=128*usedCoreNum
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| workspace/UB queues | 同 R-003 | 同 R-003 | 同 R-003 | DCCI/SyncAll/Queue/Event | 同 R-003；V 位宽改为64 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| repeats/prefix reduce | Cast INT32→INT64 if U differs, `ReduceSum/Add` | repeats/records | scalar offset | V registers | INT64 accumulation，tail mask |
| cross-core | DCCI + SyncAll | all blocks | record load | workspace | 与 R-003 相同 |
| output | grouped/split MicroAPI + DataCopyPad | x/repeats | y GM | valid region | T 位模式不变；本地 offset 64-bit，workspace record 为 U |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `repeat_interleave_tiling_normal.cpp` | `UseInt64/GetTilingKey/GetUbFactor` | 202 guard和64-bit buffer预算 |
| Kernel entry/router | `repeat_interleave_apt.cpp` | key202 | V=int64 |
| Device body/helper | `repeat_interleave_small.h` | `LoadData/CustomReduceSumVf/Process` | U→V cast、64-bit sum/offset、tail |

## 算子特定优化

### Trick T-001：低占核时拆 repeats 轴并用 Kernel 内前缀和定位输出

#### 触发与源码依据

- 适用 Route：R-003、R-004
- 触发条件：初始 `usedCoreNum<C/2`，且 `(R>64 || (CP==1 && R!=1))`，并且 `totalRepeatSum>1024`
- Golden 来源：[`repeat_interleave_tiling_normal.cpp:76-87`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_normal.cpp#L76-L87)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// 核数不满一半，继续切分CP轴
int64_t cpAxis = 2;
if ((usedCoreNum_ < totalCoreNum_ / DOUBLE) &&
    (mergedDim_[cpAxis] * ge::GetSizeByDataType(inputDtype_) > MIN_CP_THRESHOLD)) {
    SplitCP();
}
// CP轴切分后，核数还是不满，则切分repeats shape
if ((usedCoreNum_ < totalCoreNum_ / DOUBLE) &&
    (repeatShape_.GetDim(0) > MIN_SHAPE_THRESHOLD || (mergedDim_[cpAxis] == 1 && repeatShape_.GetDim(0) != 1)) &&
    (totalRepeatSum_ > MIN_TOTAL_REPEATS_SUM)) {
    SplitRepeatsShape();
}
```

#### 机制与设计原因

ACLNN flatten 使 CP 恒为1，不能靠 CP 分核；长 repeats 且输出足够大时把 R 切成多片，使多个 AIV block 同时生成输出。每片的输出长度取决于运行时 repeats 值，不能仅从 shape 得到绝对起点，因此 Device 先在隔离 cache line 中发布本片和，通过 DCCI+`SyncAll` 建立全核可见性，再归约前序片得到无重叠输出 offset。这改变了调度并行度，同时只物化每核一个和，而非完整前缀数组。

#### 迁移边界

- 适用：变长扩展、每片输出长度由 Device 输入决定、batch/CP 不能充分占核的 AIV 算子。
- 前提：950、同一 Kernel 内全核同步可用、workspace 至少每核128B、调度模式1、总和位宽按输入/输出上界选择。
- 失效：输出不超过1024或初始核数已过半时使用 R-002；不能提供全核同步与 cache 可见性时不可复用此协议。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能只按 slice 索引乘平均 repeat 次数计算输出起点；非均匀 repeats 会导致输出段重叠或空洞。固定仓以实际片和及跨核前缀规避该错误。

### Trick T-002：单元素 repeats 的 tiny-CP 寄存器 gather 路线

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：`CP*sizeof(T)<32 && CP*repeatsCount*sizeof(T)<=256`；本 ACLNN 的 CP=1
- Golden 来源：[`repeat_interleave_scalar.h:421-425`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_kernel/arch35/repeat_interleave_scalar.h#L421-L425)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr uint64_t VREG_SIZE = 256;
if (tilingData_.mergedDims[2] * sizeof(T) < GATHER_CP_THRESHOLD &&
    tilingData_.mergedDims[2] * tilingData_.repeatsCount * sizeof(T) <= VREG_SIZE) {
    ProcessCpMatchToUbGather(startCpIdx, curCoreCpCount);
    return;
}
```

#### 机制与设计原因

当一个 CP 很窄且其 r 份结果能落入 256B 寄存器跨度时，helper 直接构造每个输出 lane 对应的源索引并 `DataCopyGather`，一批处理多个输入元素；8bit 数据借助 16bit gather 后 `DeInterleave` 取有效字节。相比逐输入值嵌套 r 次 unaligned store，此路线把复制映射为规则 gather，且 tail 通过动态 mask 限制。

#### 迁移边界

- 适用：固定重复次数、窄 contiguous record、源索引能在 gather index 位宽和单寄存器跨度内表示的复制/扩展。
- 前提：950 MicroAPI gather；`r*CP*sizeof(T)<=256`；8bit 路线必须保留宽载入和 DeInterleave。
- 失效：阈值不满足时回退 R-001 的 `ProcessCpMatchToUb`/大 repeat 分块，不得继续使用单寄存器 gather。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若对 8bit 数据直接套用非 8bit gather 模板，会忽略硬件宽 lane 布局；固定源码明确以宽类型 gather 后去交织。固定仓未提供除此之外的可验证反例。

### Trick T-003：只凭 shape/outputSize 判定的恒等快捷提交

#### 触发与源码依据

- 适用 Route：核心 Kernel 前的无 Kernel 路径
- 触发条件：`outputSize==numel(selfContiguous) && numel(repeatsContiguous)==1`
- Golden 来源：[`aclnn_repeat_interleave.cpp:426-435`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/repeat_interleave/op_api/aclnn_repeat_interleave.cpp#L426-L435)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// repeats等于1，直接返回原tensor
if (outputSize == GetTensorElementsNum(selfContiguous) && GetTensorElementsNum(repeatsContiguous) == 1) {
    auto flattenShape = GetFlattenShape(selfContiguous, uniqueExecutor.get());
    auto flattenSelf = l0op::Reshape(selfContiguous, flattenShape, uniqueExecutor.get());
    CHECK_RET(flattenSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewcopyResult = l0op::ViewCopy(flattenSelf, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

#### 机制与设计原因

合法调用契约下，单元素 repeats 且 outputSize=N 意味着 repeats[0]=1，因此 Host 只保留 flatten view 与最终提交，省去 BOOL cast 和 RepeatInterleave launch。该机制改变 launch graph 和数据物化；判断没有把 repeats 从 Device 搬回 Host。

#### 迁移边界

- 适用：调用者已保证 outputSize 与 Device 参数一致、且 shape 足以证明恒等的两段式 API。
- 前提：必须维持公开契约“outputSize=N*repeats[0]”；不得把实现的快捷判断误写成实际读取/验证 repeats 值。
- 失效：长度不为1或 outputSize!=N 时进入 R-001/R-002/R-003/R-004。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若调用者提供与 repeats 值不一致的 outputSize，这个实现条件仍可能走快捷路；固定源码没有 Device 值校验兜底。因此复用时不能把该条件当作对 repeats 值的独立验证。

## 模板与实现对比

| 对比维度 | R-001 单元素 | R-002 普通逐元素 | R-003 拆分 INT32 | R-004 拆分 INT64 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | repeats 0D/len1 | 向量且不拆 | 向量拆分、32bit安全 | 向量拆分、需64bit | 先区分标量，再按占核/输出量和位宽 |
| 模板与 dtype | Scalar `<T,U>` | Normal `<T,U>` | Small `<T,U,int32>` | Small `<T,U,int64>` | T/U 是数据/重复输入，V 是 offset 累加 |
| 循环与分核 | N batch 分核 | ACLNN 常为单核遍历 N | repeats slice 多核 | 同左 | 可预测 r 用 batch；运行时 r 向量用 slice prefix |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3×depth1 + gather regs | repeats d1，x/y d2 | 5×depth1 + reduce regs | 同左，sum buffer更宽 | 均无 dynamic UB/通信 |
| 内存总占用与复用 | F(sU+2sT) | F(sU+4sT) | F(sU+2sT)+32sV+32used | 同左，sV=8 | Tiling按真实 depth 预算 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Gather/Unalign/DataCopyPad | Unalign/DataCopyPad/LoopMode | ReduceSum/Cast/Unalign/DataCopyPad | 同左 | 全部 AIV，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | MTE3→V，gather mask | MTE2→S，Queue | DCCI+SyncAll、V→S、MTE2→V | 同左，INT64 sum | 运行时 offset 路线才需跨核同步 |
| fallback/失败边界 | gather→普通/大r | 无更低 Route | 大r/group 内部分块 | 同左 | 未知 key 无输出；F=0 无显式 Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档支持；固定仓核心源码不可见，未生成卡 | 公开文档支持；固定仓核心源码不可见，未生成卡 | 公开文档、OpDef、config、CMake 与完整 arch35 实现一致支持 |
| Host/Launcher | ACLNN/l0op 前端可见；实际内置 RepeatInterleave Tiling/POD/Kernel 不可见 | 同 A2 | `ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleave)` 接 arch35 101/102/201/202 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | AIV_VECTOR；平台 C/U；16MiB+128C workspace；可选跨核前缀 |

## 限制与失败边界

- 本卡只覆盖 `aclnnRepeatInterleave`；Int、Tensor-only、WithDim 和 IntWithDim 公开变体未提取，也不得据本卡合并语义。
- 公开约束要求 repeats 值为自然数且 outputSize 与实际值一致；固定 ACLNN/Host/Tiling/Kernel 链没有读取全部 Device repeats 来验证。负值会进入循环/offset 计算，行为不能由固定源码安全闭合为合法输出。
- ACLNN 恒等快路只比较 outputSize、self 元素数和 repeats 元素数，没有验证唯一 repeats 值确为1。
- shape 乘积、`outputSize/N`、`averageRepeatTime` 和若干 offset 乘法没有统一的显式溢出检查；R-003 仅在特定条件下选 INT32 V，R-004 只保护已识别的累加/offset 路径。
- Device 多处把 repeat 次数或 count 窄化为 `uint16_t`（例如 gather、`IndexRepeatVf`、LoopMode），固定 Host 未给出 repeats 单值 `<=65535` 的公开校验或 fallback；超界语义固定仓无法证明。
- R-004 在 repeats 为 INT32 但 y 元素数超过 INT32_MAX 时把本地归约 V 升为 INT64；然而跨核 `recordSumGm_` 的元素类型仍是 U=INT32。固定源码没有证明每片和恒不超过 INT32_MAX，故这一组合的跨核前缀安全边界不可外推。
- `ubFactor` 被向下对齐到32并用于除法/Queue容量；固定 Tiling 未显式拒绝 F=0，实际 950 UB 对所有 binary dtype 是否恒保证 F>0 由平台资源隐含，固定源码没有独立断言。
- 201/202 `averageRepeatTime=outputSize/N` 可为0，而 `CopyOutSplitGroup` 用它做 `ceil(repeatCount,averageRepeatTime)`；是否总由其他分支避免除零在固定源码不能对所有合法 repeats 分布闭合，复用时需保留输入域审计。
- entry 对未知 tiling key 静默返回且不写 y；正常 Host 只产生 101/102/201/202。
- 框架 `Contiguous/BroadcastTo/Cast/Reshape/ViewCopy` 的生成 Kernel、逐节点 workspace slice 和内部事件在固定仓不可见，作为明确的框架边界；算子拥有的 RepeatInterleave 核心链已闭合。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/repeat_interleave/docs/aclnnRepeatInterleave.md` | 产品表、函数原型、约束 | A2/A3/950公开支持、语义/dtype/shape |
| 入口声明/约束 | `index/repeat_interleave/op_api/aclnn_repeat_interleave.h` | `aclnnRepeatInterleave*` | C ABI；注释差异 |
| 入口实现 | `index/repeat_interleave/op_api/aclnn_repeat_interleave.cpp` | `CheckParams`, `InitializeTensor`, `aclnnRepeatInterleaveGetWorkspaceSize` | 校验、快路、flatten/cast、核心调用、ViewCopy |
| Launcher | `index/repeat_interleave/op_api/repeat_interleave.cpp` | `l0op::RepeatInterleave` | y shape/分配与 AICore launcher |
| Host/Tiling | `index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_arch35.cpp` | prepare/Tiling注册 | AIV core、UB producer |
| Host/Tiling | `index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_base.cpp` | shape/dtype/merge/workspace | 三维折叠、校验、16MiB+128C |
| Host/Tiling/POD | `index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_normal.h` | Norm/Small POD | 逐字段 ABI 声明 |
| Host/Tiling | `index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_normal.cpp` | split/F/key/PostTiling | 101/102/201/202 producer |
| Kernel entry/body | `index/repeat_interleave/op_kernel/repeat_interleave_apt.cpp` | `repeat_interleave` | key router、workspace gate、模板实例 |
| Kernel entry/body | `index/repeat_interleave/op_kernel/arch35/repeat_interleave.h` | `RepeatInterleaveImpl` | key102 window/复制/Queue/tail |
| Kernel entry/body | `index/repeat_interleave/op_kernel/arch35/repeat_interleave_scalar.h` | `RepeatInterleaveScalarImpl` | key101 gather/普通/大r路线 |
| Kernel helper | `index/repeat_interleave/op_kernel/arch35/repeat_interleave_gather.h` | gather helpers | index公式、8bit处理和tail mask |
| Kernel entry/body | `index/repeat_interleave/op_kernel/arch35/repeat_interleave_small.h` | `RepeatInterleaveSmall` | key201/202跨核前缀、归约、复制、内存 |
| 构建接线 | `index/repeat_interleave/op_host/repeat_interleave_def.cpp` | `RepeatInterleave` OpDef | ascend950 AICore config |
| 构建接线 | `index/repeat_interleave/op_host/config/ascend950/repeat_interleave_binary.json` | `op_type=RepeatInterleave` | 950 x/repeats dtype实例 |
| 构建接线 | `index/repeat_interleave/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950 + arch35 活动接线 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `name=RepeatInterleave` | ascend950 Kernel 编译及 no-dcache-flush 选项 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Norm/Small 两套 int64 字段布局与四 key | 字段顺序、Small V位宽选择、inactive CP字段仍序列化 | `repeat_interleave_tiling_normal.h` |
| 调用顺序 | 连续化→flatten/BOOL cast→核心→回 cast/ViewCopy | 空/恒等快路与错误短路；axis固定0 | `aclnnRepeatInterleaveGetWorkspaceSize` |
| workspace/内存 | 16MiB系统段+128C用户段；Small 每核128B记录 | user pointer获取、cacheline隔离、Queue depth和F公式 | `GetWorkspaceSize`, `RepeatInterleaveSmall::Init` |
| 同步 | Queue events；Small DCCI→SyncAll→MTE2_V→V_S | 跨核记录发布顺序和schedule mode1 | `CalcOutputStartOfset` |
| 精度与尾块 | T位模式复制；201/202按上界选V；动态 mask/tail | INT64触发、workspace record 仍为U、有效 byte count、uint16窄化风险 | `UseInt64`, `CustomReduceSumVf`, gather helpers |
| 失败边界 | 参数/shape/dtype失败、空路、unknown key | 不推测负 repeats、outputSize不一致、F=0与超大repeat行为 | `CheckParams`, `CheckShape`, `repeat_interleave` |
