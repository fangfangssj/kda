---
type: Operator
title: "GroupNormSwish A2/A3 ACLNN 算子知识"
description: "说明 aclnnGroupNormSwish 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐路线 Vector 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/docs/aclnnGroupNormSwish.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/group_norm_swish_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/group_norm_swish_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/group_norm_swish_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_hw1_b16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_hw1_b32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_small_b16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_small_b32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_norm_b16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_norm_b32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_large_b16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_large_b32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/config/ascend910b/group_norm_swish_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/config/ascend910_93/group_norm_swish_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:56:05Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# GroupNormSwish A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnGroupNormSwishGetWorkspaceSize` / `aclnnGroupNormSwish`；固定仓未出现手写 ACLNN bridge，`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 接入生成边界。 |
| 计算语义 | 对 NCHW 的每个 `(n,g)` 在 `D=(C/numGroups)·∏shape[2:]` 个元素上求 mean/variance，做 affine；`activateSwish=true` 时再做 `z/(1+exp(-swishScale·z))`。 |
| 输入输出 | `x/y`: FP16/BF16/FP32、ND、2–8D；`gamma/beta`: 长度 C，同 dtype 或在 x 为 16-bit 时为 FP32；`mean/rstd`: `[N,numGroups]`、gamma dtype。 |
| 执行类型 | 单 Kernel、单节点 `AIV_VECTOR`；Host 明确以 `GetCoreNumAiv()` 取核数，无 SIMT/NDDMA/通信/Cube 节点。 |
| Kernel 路线 | R-001…R-012：四个 shape family（HW1/Small/Norm/Large）× FP16/BF16 同型、16-bit+FP32 affine、FP32。 |
| 核心资源 | GM + `TQue<VECIN/VECOUT>` + FP32 `TBuf`；workspace 仅设置系统保留区，Kernel 参数 `workspace` 未读取。 |
| 来源闭包 | `source_closed`：公开 API 文档→ACLNN OpDef/生成接线→`Tiling4GroupNormSwish`/`GroupNormSwishTilingData`→`group_norm_swish` router→12 个 Host 可达模板实例与直接 helper→A2/A3 binary config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `H=∏_{i=2}^{rank-1} shape[i]`、`D=C/numGroups`、`M=D·H`。对每个 batch `n` 和 group `g`：`μ=Σx/M`，`σ²=Σ(x-μ)²/M`，`rstd=1/sqrt(σ²+eps)`，`z=(x-μ)·rstd·gamma[c]+beta[c]`。输出 `mean[n,g]=μ`、`rstd[n,g]=rstd`；激活关闭时 `y=z`，开启时 `y=z/(1+exp(-swishScale·z))`。Grad、Silu、Quant 和原地语义不属于本卡。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，2–8D，逻辑 NCHW | FP16/BF16/FP32 | 非空；N、C 大于 0；C 可被 `numGroups` 整除。 |
| `gamma` | `[C]`/ND | 与 x 相同；x 为 16-bit 时也可 FP32 | 非空；与 beta dtype 相同。 |
| `beta` | `[C]`/ND | 同 gamma | 非空；逐通道偏置。 |
| `numGroups` | 标量属性 | INT64 | `>0`；整除 C。 |
| `dataFormatOptional` | Host 字符串 | `char*` | OpDef 默认 `NCHW`；Tiling 不读取该属性，公开 README 声明当前仅 NCHW。 |
| `eps` | 标量属性 | API double→OpDef/Tiling float | OpDef 默认 `1e-5`；POD 为 FP32。 |
| `activateSwish` | 标量属性 | bool→POD int64 | OpDef 默认 true。 |
| `swishScale` | 标量属性 | API double→OpDef/Tiling float | OpDef 默认 1.0；仅激活开启时参与计算。 |
| `yOut` | 与 x 相同 | 与 x 相同 | 完整有效域。 |
| `meanOut`,`rstdOut` | `[N,numGroups]`/ND | 与 gamma 相同 | 每组各一个标量。 |

### 入口与版本

固定 Commit 的公开文档只声明无版本后缀的最新入口 `aclnnGroupNormSwishGetWorkspaceSize`/`aclnnGroupNormSwish`。第一段完成校验、连续化/执行器准备并返回 workspace；第二段在传入 stream 上执行。手写 inner bridge 不在固定仓中，按生成 ACLNN 框架边界处理；算子拥有的 OpDef、Tiling、POD、router/body 和 binary config 均可见。README/API 产品表同时明确支持 Atlas A2 与 A3，OpDef 分别接线 `ascend910b`、`ascend910_93`，不存在公开“不支持”冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

生成 ACLNN 层先检查必需 tensor 非空和公开 dtype 范围，并由 `AutoContiguous()` 处理 x/gamma/beta 非连续输入。进入 `Tiling4GroupNormSwish` 后，`CheckInputParams` 依次检查 x dtype、rank≥2、gamma/beta 为 1D 且长度 C、二者 dtype 一致，以及 x=FP32 时 affine 必须 FP32；`CheckAttrParams` 检查 `numGroups>0 && C%numGroups==0`。任一失败返回 `GRAPH_FAILED`，不启动 Kernel。

### 参数、Workspace 与生命周期

`SetAttrParams` 把四个参与计算的属性写 POD；`SetInputParams` 生成 `hwNum`、`shapeD=C/G`、`numPerGroup=shapeD·hwNum`。`SetBlockTiling` 以 `totalGroup=N·G` 和 `totalCoreNum=GetCoreNumAiv()` 计算分核；`SetUbTiling` 固定 `numPerLoop=8192`。A2/A3 的 `GetSocVersion()!=ASCEND310P`，workspace size 为 16 MiB 系统保留区；Kernel 签名虽接收 `workspace`，router/body 没有读取，生命周期仍由 ACLNN executor 持有至 stream 上任务完成。

### 启动图与失败传播

```text
aclnnGroupNormSwishGetWorkspaceSize
  -> generated ACLNN executor / AutoContiguous
  -> GroupNormSwish OpDef
  -> Tiling4GroupNormSwish (validate, POD, key, blockDim, workspace=16 MiB)
aclnnGroupNormSwish(workspace, workspaceSize, executor, stream)
  -> group_norm_swish<<<usedCoreNum, stream>>>(x,gamma,beta,y,mean,rstd,workspace,tiling)
  -> exactly one AIV_VECTOR body selected by key
```

唯一 Device 节点的 `blockDim=usedCoreNum`，POD 为 `GroupNormSwishTilingData`，GM 输出直接写 `y/mean/rstd`，无节点间 workspace edge。executor/stream 提供启动顺序和完成可见性；Tiling/launch 失败会阻断节点，router 对未知 key 没有 fallback 分支，也不会写输出。

## Route 与参数设计

### 分核与资源计算

`totalGroup=N·G`；`groupPerCore=ceil(totalGroup/totalCoreNum)`；`groupPerCoreAlign=ceil(groupPerCore/8)·8`；`usedCoreNum=ceil(totalGroup/groupPerCore)`；`groupLastCore=totalGroup-(usedCoreNum-1)·groupPerCore`。每核 x/y 基址为 `blockIdx·groupPerCore·numPerGroup`，普通核处理 `groupPerCore` 组，末核处理 `groupLastCore` 组。`shapeCAlign=ceil(C/(32/sizeof(gamma)))·(32/sizeof(gamma))`。`loopTimes=ceil(M/8192)`、`loopTimesAlign=ceil(loopTimes/8)·8`、`numTailLoop=(M%8192==0?8192:M%8192)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 HW1 B16 same | 111 / `GroupNormSwishHW1B16<T1,T1>` | `sizeof(x)=2 && sizeof(gamma)=2 && hwNum==1` | R-002/R-003 或下一 shape |
| 2 | R-002 HW1 B16 mixed | 121 / `GroupNormSwishHW1B16<T1,float>` | `sizeof(x)=2 && sizeof(gamma)=4 && hwNum==1` | 下一 dtype/shape |
| 3 | R-003 HW1 B32 | 211 / `GroupNormSwishHW1B32<float,float>` | `sizeof(x)=4 && sizeof(gamma)=4 && hwNum==1` | 下一 shape |
| 4 | R-004 Small B16 same | 112 / `GroupNormSwishSmallB16<T1,T1>` | 16-bit same；`hwNum!=1 && M<=8192 && shapeCAlign+groupPerCoreAlign<=16384/2-8` | R-010 |
| 5 | R-005 Small B16 mixed | 122 / `GroupNormSwishSmallB16<T1,float>` | 16-bit+FP32 affine；同 Small 谓词（limit 4088） | R-011 |
| 6 | R-006 Small B32 | 212 / `GroupNormSwishSmallB32<float,float>` | FP32；同 Small 谓词（limit 4088） | R-012 |
| 7 | R-007 Norm B16 same | 113 / `GroupNormSwishNormB16<T1,T1>` | 16-bit same；`hwNum!=1 && M>8192 && shapeCAlign+groupPerCoreAlign<=16384/2-2·loopTimesAlign` | R-010 |
| 8 | R-008 Norm B16 mixed | 123 / `GroupNormSwishNormB16<T1,float>` | 16-bit+FP32 affine；同 Norm 谓词 | R-011 |
| 9 | R-009 Norm B32 | 213 / `GroupNormSwishNormB32<float,float>` | FP32；`shapeCAlign+groupPerCoreAlign<=16384/4-2·loopTimesAlign` | R-012 |
| 10 | R-010 Large B16 same | 114 / `GroupNormSwishLargeB16<T1,T1>` | 16-bit same；非 HW1 且 Small/Norm 资源式失败 | 无下一 Route |
| 11 | R-011 Large B16 mixed | 124 / `GroupNormSwishLargeB16<T1,float>` | 16-bit+FP32 affine；同 Large fallback | 无下一 Route |
| 12 | R-012 Large B32 | 214 / `GroupNormSwishLargeB32<float,float>` | FP32；同 Large fallback | 无下一 Route |

```text
key = (xBytes==2 ? 100 : xBytes==4 ? 200 : 0)
    + (xBytes==gammaBytes ? 10 : 20)
if hwNum==1: key += 1
else if M<=8192:
  key += (shapeCAlign+groupPerCoreAlign <= 16384/xBytes-8 ? 2 : 4)
else:
  key += (shapeCAlign+groupPerCoreAlign <= 16384/xBytes-2*loopTimesAlign ? 3 : 4)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 111 | FP16/BF16 x 与 affine 同型；H=1 | 单 AIV | `GroupNormSwishHW1B16<T1,T1>` | equal/one-loop/multi-loop Device 子分支；FP32 中间 | 8192 元素尾搬运 |
| R-002 | 121 | FP16/BF16 x，FP32 affine；H=1 | 单 AIV | `GroupNormSwishHW1B16<T1,float>` | 多 `CastGammaAndBeta` 分支被编译消除；equal 路线 mean 转 FP32 | 同上 |
| R-003 | 211 | 全 FP32；H=1 | 单 AIV | `GroupNormSwishHW1B32<float,float>` | x 可在 inQueue 上原地 affine；无 x1Buf32 | 同上 |
| R-004/R-005 | 112/122 | 16-bit；Small 谓词 | 单 AIV | `GroupNormSwishSmallB16<T1,T2>` | M 一次装入；H%8 决定整组或逐通道处理 | Large fallback 已由 Host 选择 |
| R-006 | 212 | FP32；Small 谓词 | 单 AIV | `GroupNormSwishSmallB32<float,float>` | 激活开启时 xUB 原地保存 z，yUB 保存分母/结果 | 同上 |
| R-007/R-008 | 113/123 | 16-bit；Norm 谓词 | 单 AIV | `GroupNormSwishNormB16<T1,T2>` | 8192 分片 two-pass；mean/rstd 分片统计缓存 | Large fallback |
| R-009 | 213 | FP32；Norm 谓词 | 单 AIV | `GroupNormSwishNormB32<float,float>` | 同上，无 B16 cast buffer | Large fallback |
| R-010/R-011 | 114/124 | 16-bit；Large fallback | 单 AIV | `GroupNormSwishLargeB16<T1,T2>` | 固定 256 affine/output chunk 与 1792 个统计片 chunk | 无 fallback |
| R-012 | 214 | FP32；Large fallback | 单 AIV | `GroupNormSwishLargeB32<float,float>` | 固定 192 KiB UB；x queue 原地 | 无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV 核数 | `GetCoreNumAiv()>0` | 执行族和 blockDim producer | 全部 | `GRAPH_FAILED` | `Tiling4GroupNormSwish` |
| 32B affine 对齐 | `shapeCAlign=ceil(C/(32/gammaBytes))·...` | gamma/beta DMA 和 FP32 local 容量 | Small/Norm | 计入 limit；超限 Large | `SetTilingKey` |
| 组统计对齐 | `groupPerCoreAlign=align_up(groupPerCore,8)` | mean/rstd 32B 块 | Small/Norm | 计入 limit | `SetBlockTiling` |
| Small UB guard | `CAlign+GAlign<=16384/xBytes-8` | `InitLocal*` 的 affine、输出统计与 8 个 reduce slot | R-004…006 | Large | `SetTilingKey` |
| Norm UB guard | `CAlign+GAlign<=16384/xBytes-2*loopTimesAlign` | 每片 mean/rstd 各需 `2*loopTimesAlign` FP32 slot | R-007…009 | Large | `SetTilingKey` |
| 16-bit DMA length | `copyNum*sizeof(T)<=uint16_t` on `DataCopyPad` branch | `blockLen` 为 uint16_t byte | 所有 8192 tile | 固定 tile 使 16-bit 最大 16384 B；尾同界 | `CopyInX` |
| 非 Pad 尾写 | `copyNum % elementsPerBlock != 0` | 避免多写 GM | 各 Route tail | 前段普通拷贝+最后 32B AtomicAdd | `CopyOutWithOutPadT1/T2` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 的序列化宏定义不在固定仓，因此整体 `sizeof/alignof` 与每字段 `offsetof` 固定仓不可见；字段顺序、源码类型和 Host/Device producer-consumer 如下。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `GroupNormSwishTilingData` | `numGroups` | int64/64 | offset 不可见 | group | `SetAttrParams(*numGroups)` | group→channel 映射 | >0，整除 C |
| 同上 | `epsilon` | float/32 | offset 不可见 | 标量 | `set_epsilon(*epsilon)` | `sqrt(var+epsilon)` | API double 已窄化 |
| 同上 | `activateSwish` | int64/64 | offset 不可见 | bool 值 | `set_activateSwish(*bool)` | 激活分支 | 0/1 |
| 同上 | `swishScale` | float/32 | offset 不可见 | 标量 | `set_swishScale(*float)` | `Muls(...,-scale)` | 激活关闭时不读计算值 |
| 同上 | `hwNum` | int64/64 | offset 不可见 | element/channel | shape[2:] 乘积 | key 与 loop | 乘法溢出未显式检查 |
| 同上 | `shapeC` | int64/64 | offset 不可见 | channel | `shape[1]` | affine 范围/equal guard | >0 由公开 API |
| 同上 | `shapeCAlign` | int64/64 | offset 不可见 | affine element | 32B 对齐 | buffer size | ≥C |
| 同上 | `shapeD` | int64/64 | offset 不可见 | channel/group | `C/numGroups` | channel loop | ≥1 |
| 同上 | `numPerGroup` | int64/64 | offset 不可见 | x element/group | `shapeD*hwNum` | reduce、address | >0 |
| 同上 | `groupPerCore` | int64/64 | offset 不可见 | group/core | `ceil(totalGroup/core)` | base/loop | ≥1 |
| 同上 | `groupLastCore` | int64/64 | offset 不可见 | group | remainder | last core loop | 1…groupPerCore |
| 同上 | `groupPerCoreAlign` | int64/64 | offset 不可见 | group | align 8 | output queue size | ≥groupPerCore |
| 同上 | `numPerLoop` | int64/64 | offset 不可见 | element | 固定 8192 | Queue/tile | 固定值 |
| 同上 | `loopTimes` | int64/64 | offset 不可见 | loop | `ceil(M/8192)` | Norm/Large | ≥1 |
| 同上 | `loopTimesAlign` | int64/64 | offset 不可见 | loop slot | align 8 | statistics TBuf | ≥loopTimes |
| 同上 | `numTailLoop` | int64/64 | offset 不可见 | element | M modulo 8192（0→8192） | tail DMA/reduce | 1…8192 |
| 同上 | `totalCoreNum` | int32/32 | offset 不可见 | AIV core | `GetCoreNumAiv()` | Host-only block tiling；Device 不读 | >0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-002 | 111/121 | `group_norm_swish_tiling.cpp::SetTilingKey` | 公共 POD | `group_norm_swish.cpp::group_norm_swish` | `group_norm_swish_hw1_b16.h` + base | ascend910b/910_93 JSON + CMake |
| R-003 | 211 | 同上 | 同上 | 同上 | `group_norm_swish_hw1_b32.h` + base | 同上 |
| R-004/R-005 | 112/122 | 同上 | 同上 | 同上 | `group_norm_swish_small_b16.h` + base | 同上 |
| R-006 | 212 | 同上 | 同上 | 同上 | `group_norm_swish_small_b32.h` + base | 同上 |
| R-007/R-008 | 113/123 | 同上 | 同上 | 同上 | `group_norm_swish_norm_b16.h` + base | 同上 |
| R-009 | 213 | 同上 | 同上 | 同上 | `group_norm_swish_norm_b32.h` + base | 同上 |
| R-010/R-011 | 114/124 | 同上 | 同上 | 同上 | `group_norm_swish_large_b16.h` + base | 同上 |
| R-012 | 214 | 同上 | 同上 | 同上 | `group_norm_swish_large_b32.h` + base | 同上 |

## Kernel 实现

### 公共执行流程

router 解 POD 后创建 `TPipe`，按 key 实例化一个类并依次 `Init`、`Process`。普通基类把当前核 x/y 基址偏移到 `blockIdx·groupPerCore·M`，gamma/beta 与 mean/rstd 保持全局基址；各类在 `Process` 中选择末核 group 数。所有非 equal 路线按“加载 affine→逐组统计→FP32 mean/rstd→affine→可选 Swish→降精度→写 y→写 mean/rstd”执行。`TQue` 的 EnQue/DeQue/FreeTensor 建立 MTE↔Vector 生命周期，Vector→Scalar 读 reduce 结果前用 `HardEvent::V_S`，连续 Vector primitive 间用 `PipeBarrier<PIPE_V>`。

### Route R-001：HW1 B16 同型

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_swish_tiling.cpp` | `SetTilingKey` | xBytes=2、gammaBytes=2、`hwNum==1` | key 111 |
| Router | `op_kernel/group_norm_swish.cpp` | `group_norm_swish` | `TILING_KEY_IS(111)` | `GroupNormSwishHW1B16<DTYPE_X,DTYPE_X>` |

#### Device 数据流

##### 函数级调用链

```text
group_norm_swish -> GroupNormSwishHW1B16::Init -> Process -> ProcessPerCore
  -> shapeD==1: ComputeEqual(mean=x, rstd=1/sqrt(eps), y=Swish(beta) or beta)
  -> shapeD<=8192 && groupNum<=2048: ComputeOneLoop -> reduce -> affine/Swish -> CopyOut
  -> otherwise: ComputeMultipleLoop -> 8192-slice two-pass -> affine/Swish
```

##### 源码等价伪码

```text
for localGroup in coreGroups:
  if D==1: bulk-copy x->mean; fill rstd; map beta cyclically to y and optional Swish
  else: μ,var = two_pass_reduce(x[group], tile=8192); save μ,rstd
        for d/tile: z=(x-μ)*rstd*gamma+beta; y=activate?z/(1+exp(-s*z)):z
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | coreBase+group·D | D/group | T1/ND | GM/Kernel | reduce/output | caller | 8192 tile + tail |
| gamma/beta GM | globalGroup·D | D/group | T1 | caller | affine | caller | tile |
| equal `inQueueX` | 0 | depth2·8192·4=65536 B | raw/FP32-capacity | DMA/Vector | direct route | pipe | queue reused across mean/rstd/y |
| equal x1/x2 TBuf | 0 | 2·8192·4=65536 B | FP32 | cast/Swish | Vector | pipe | total equal UB=131072 B |
| regular queues/TBuf | local | 188416 B total | mixed/FP32 | Init | route | pipe | one-deep x/y, FP32 stats |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gin["gammaQ+betaQ · 65536 B"]:1
  xy["xQ+yQ · 32768 B"]:1
  stat["meanQ+rstdQ+statBuf · 24576 B"]:1
  calc["x1+x2 · 65536 B"]:1
```

##### 字节账与复用顺序

```text
UB_equal = 2*8192*4 + 2*8192*4 = 131072 B
UB_regular = 2*(8192*4) + 2*(8192*2) + 2*(2048*4) + 2*(8192*4) + 2*(1024*4) = 188416 B
dynamic_ub = N/A
workspace_total = 16 MiB system reservation; operator body consumes 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `x1Ub32` | Cast/affine | reduce/Swish/Cast-out | next tile | `PIPE_V` + queue lifetime | current y 已 EnQue |
| `inQueueX` equal | x DMA / beta DMA / rstd fill | direct GM write/Swish | next phase | `PIPE_ALL`, FreeTensor | phase completes before reuse |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| reduce→scalar | `HardEvent::V_S` | `ReduceSumCustom` | scalar `GetValue` | mean/rstd slots | FP32 two-pass |
| vector chain | `PipeBarrier<PIPE_V>` | Cast/Add/Mul/Exp | next Vector op | x1/x2 | T1→FP32；`GetRoundMode()` 回 T1 |
| tail | DataCopyPad or atomic tail helper | UB | GM | final <32B block | only valid elements committed |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_swish_tiling.cpp` | `SetTilingKey` | key 111 |
| Kernel entry/router | `op_kernel/group_norm_swish.cpp` | key 111 branch | 实例化同型 B16 |
| Device body/helper | `op_kernel/group_norm_swish_hw1_b16.h` | `ComputeEqual/ComputeOneLoop/ComputeMultipleLoop` | 三个 HW1 子路径完整实现 |

### Route R-002：HW1 B16 + FP32 affine

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `SetTilingKey` / key 121 | xBytes=2、gammaBytes=4、H=1 | `GroupNormSwishHW1B16<T1,float>` |

#### Device 数据流

##### 函数级调用链

```text
key121 -> GroupNormSwishHW1B16<T1,float>::Init/Process -> same three HW1 branches
  -> equal: x T1 cast to FP32 mean; rstd FP32; beta FP32 -> y T1
  -> regular: CopyInGammaAndBeta already FP32; FP32 reduce/affine/Swish -> Cast y
```

##### 源码等价伪码

```text
same as R-001, except affine/stat outputs remain FP32 and only y is cast back to T1
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| affine/stat GM | channel/group offsets | D and groupNum | FP32 | caller/Kernel | affine/caller | caller | 32B tail helper |
| regular UB | same allocation sizes as R-001 | 188416 B | T1 + FP32 | Init | Process | pipe | no B16 affine cast |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  affine["gammaQ+betaQ · 65536 B"]:1
  xy["xQ+yQ · 32768 B"]:1
  stats["mean/rstd Q+Buf · 24576 B"]:1
  calc["x1+x2 · 65536 B"]:1
```

##### 字节账与复用顺序

```text
UB_equal = 131072 B; UB_regular = 188416 B; workspace body use = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1/x2 | cast/reduce/Swish | next Vector stage | next tile | barriers | output committed |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| affine | FP32 queue | FP32 Vector | barrier | gamma/beta | avoids affine down-cast |
| y tail | Pad/atomic helper | yUB | yGM | last block | Cast RINT/NONE per arch macro |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `SetTilingKey` | dtype digit 120 + HW1 |
| Kernel entry/router | kernel cpp | key 121 | `GroupNormSwishHW1B16<DTYPE_X,float>` |
| Device body/helper | hw1_b16.h/base.h | `ComputeEqualMeanDiffType`, `ComputeGroupNormSwish` | FP32 affine/stat 路线 |

### Route R-003：HW1 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 211 | x/gamma 4B、H=1 | `GroupNormSwishHW1B32<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
key211 -> GroupNormSwishHW1B32::Init -> ProcessPerCore -> equal / one-loop / multi-loop
```

##### 源码等价伪码

```text
FP32 x remains in inQueueX; reduce uses x directly; affine may overwrite xUB; yUB holds denominator/result
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| equal xQ+x2 | local | 65536+32768=98304 B | FP32 | DMA/Vector | direct route | pipe | xQ phase reuse |
| regular UB | local | 188416 B | FP32 | Init | all stages | pipe | x1 buffer eliminated |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  q["xQ/affine queues"]:1
  stat["mean/rstd queues+buffers"]:1
  x2["x2 scratch"]:1
```

##### 字节账与复用顺序

```text
UB_equal=98304 B; UB_regular=188416 B; workspace body use=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xUB | DMA | reduce→affine | after y EnQue | queue FreeTensor | activation result no longer needs x |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all compute | FP32 Vector + barriers | prior op | next op | x/x2/y | no cast loss; same tail helper |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 211 | FP32/HW1 |
| Kernel entry/router | kernel cpp | key 211 | B32 body |
| Device body/helper | hw1_b32.h | `ComputeOneLoopInner`, `ComputeEqual*` | FP32 in-place route |

### Route R-004：Small B16 同型

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 112 | 16-bit 同型、H≠1、M≤8192、Small UB guard | `GroupNormSwishSmallB16<T1,T1>` |

#### Device 数据流

##### 函数级调用链

```text
GroupNormSwishSmallB16::ProcessPerCore -> CopyInGammaAndBeta -> Compute
  -> H%8==0: whole-group CopyIn -> ComputeMeanAndRstd -> per-channel affine -> Swish -> CopyOut
  -> H%8!=0: whole-group statistics, then per-channel CopyIn/affine/CopyOut
```

##### 源码等价伪码

```text
load all gamma/beta; for each group compute FP32 mean/variance over M
if H aligned to 8: retain group in x1 and apply each channel segment
else: reload each H segment to avoid cross-channel padded lanes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| affine queues | 0 | `8·shapeCAlign` B | FP32 capacity | DMA/cast | affine | pipe | one-deep |
| x/y queues | group/channel offsets | `4·8192·sizeof(T1)` B | T1 | DMA/Vector | compute/GM | pipe | depth2 each |
| x1/x2 | 0 | 65536 B | FP32 | Cast/reduce | affine/Swish | pipe | x2 reduction scratch |
| mean/rstd Q+Buf | group index | `8·groupPerCoreAlign+64` B | FP32 | reduce | output | pipe | 8 stat slots |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  affine["affineQ · 8*CAlign"]:1
  xy["x/yQ · 4*8192*sizeof(T1)"]:1
  calc["x1+x2 · 65536"]:1
  stat["stats · 8*GAlign+64"]:1
```

##### 字节账与复用顺序

```text
UB_total = 131072 + 8*(shapeCAlign + groupPerCoreAlign + 8) B
selector guard: shapeCAlign+groupPerCoreAlign <= 8184
workspace body use = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x2Buf | reduce work / centered-square | second reduce | next group | V_S + PIPE_V | scalar read complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast/reduce | Queue + V_S | x DMA/Vector | scalar mean/rstd | group data | FP32 accumulate |
| H unaligned | per-channel DMA | GM | y GM | padded lane isolation | only H valid elements |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `SetTilingKey` | key 112 guard |
| Kernel entry/router | kernel cpp | key 112 | GroupNormSwishSmallB16 same |
| Device body/helper | small_b16.h/base.h | `ComputeOneLoop/ComputeMulLoop/ReduceSumCustom` | aligned/unaligned H 实现 |

### Route R-005：Small B16 + FP32 affine

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 122 | R-004 shape guard，gammaBytes=4 | `GroupNormSwishSmallB16<T1,float>` |

#### Device 数据流

##### 函数级调用链

```text
key122 -> GroupNormSwishSmallB16<T1,float> -> aligned-H or per-channel-H branch -> FP32 affine/Swish -> y cast
```

##### 源码等价伪码

```text
R-004 pipeline; gamma/beta already FP32, so constexpr CastGammaAndBeta branch is absent
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| queues/TBuf | same as R-004 | `131072+8*(CAlign+GAlign+8)` B | T1/FP32 | Init | Process | pipe | affine queues directly FP32 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  affine["FP32 affineQ"]:1
  xy["T1 x/y double queue"]:1
  calc["FP32 x1/x2/stats"]:1
```

##### 字节账与复用顺序

```text
UB_total = 131072+8*(shapeCAlign+groupPerCoreAlign+8) B; workspace body use=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| affineQ | DMA | all groups on core | end core | FreeTensor | queue retained until Compute returns |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| output | Cast + Pad/atomic tail | FP32 y | GM | last H/group | x T1 output semantics |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 122 | mixed dtype |
| Kernel entry/router | kernel cpp | key 122 | GroupNormSwishSmallB16<T1,float> |
| Device body/helper | small_b16.h | constexpr dtype branch | FP32 affine retained |

### Route R-006：Small FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 212 | FP32 + Small shape guard | `GroupNormSwishSmallB32<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
GroupNormSwishSmallB32::ProcessPerCore -> ComputeOneLoop(H%8==0) or ComputeMulLoop -> ReduceSumCustom -> affine/Swish
```

##### 源码等价伪码

```text
reduce directly from xUB; if activation on, affine result stays in xUB and yUB builds exp denominator/result;
if activation off, affine writes yUB directly
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| all UB | local | `163840+8*(CAlign+GAlign+8)` B | FP32 | InitLocalB32 | Process | pipe | no x1Buf |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  q["affine/x/y queues"]:1
  x2["x2 reduce scratch · 32768"]:1
  stat["mean/rstd queues+buf"]:1
```

##### 字节账与复用顺序

```text
UB_total=163840+8*(shapeCAlign+groupPerCoreAlign+8) B; selector limit CAlign+GAlign<=4088
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xUB | DMA/affine | y Swish | FreeTensor | EnQue y + queue discipline | activation branch controls dst alias |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| affine/Swish | PIPE_V | xUB | yUB | in-place source | FP32 throughout |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 212 | FP32 Small |
| Kernel entry/router | kernel cpp | key 212 | GroupNormSwishSmallB32 |
| Device body/helper | small_b32.h | `ComputeOneLoopInner/CalcGroupNormSwish` | activation-dependent alias |

### Route R-007：Norm B16 同型

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 113 | 16-bit same、M>8192、Norm guard | `GroupNormSwishNormB16<T1,T1>` |

#### Device 数据流

##### 函数级调用链

```text
GroupNormSwishNormB16::Compute -> ComputeMeanAndRstd (8192 slices, TwoPassSumOneLoop)
  -> H<=8192 && H%8==0: ComputeSmallHW packs multiple channels/tile
  -> else: ComputeNormHW splits every channel spatial extent
```

##### 源码等价伪码

```text
for each group: collect per-tile mean and squared residual sums; merge tiles with weighted variance formula;
then traverse D channels in packed-channel or per-channel tiles and apply affine/Swish
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| stat buffers | `2*tileIndex` | each `2*loopTimesAlign*4` B | FP32 | per-tile reduce | merge | pipe | mean/rstd separate |
| total UB | local | `131072+8*(CAlign+GAlign+2*LAlign)` B | mixed | InitLocal | Process | pipe | Host guard bounds it |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  affine["affineQ"]:1
  xy["x/y double Q"]:1
  calc["x1+x2"]:1
  stats["mean/rstd · 16*LAlign + output stats"]:1
```

##### 字节账与复用顺序

```text
UB_total=131072+8*(shapeCAlign+groupPerCoreAlign+2*loopTimesAlign) B
workspace body use=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mean/rstd slice slots | `ComputeSumTwoPass` | `TwoPassSumOneLoop` | next group | V_S wait | all loopTimes slots merged |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile merge | weighted two-pass + compensated sum | per-tile stats | final μ/var | large M precision | last tile uses numTailLoop |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 113 | Norm resource guard |
| Kernel entry/router | kernel cpp | key 113 | GroupNormSwishNormB16 same |
| Device body/helper | norm_b16.h/base.h | `ComputeMeanAndRstd`, `TwoPassSumOneLoop` | 分片统计/合并 |

### Route R-008：Norm B16 + FP32 affine

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 123 | R-007 shape guard，gammaBytes=4 | `GroupNormSwishNormB16<T1,float>` |

#### Device 数据流

##### 函数级调用链

```text
key123 -> GroupNormSwishNormB16<T1,float> -> tiled two-pass stats -> packed/per-channel affine -> optional Swish
```

##### 源码等价伪码

```text
R-007 pipeline with FP32 gamma/beta and FP32 mean/rstd outputs; only x/y use T1
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB | local | `131072+8*(CAlign+GAlign+2*LAlign)` B | T1/FP32 | Init | route | pipe | constexpr cast absent |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  affine["FP32 affineQ"]:1
  tile["T1 tile Q + FP32 calc"]:1
  stats["per-tile FP32 stats"]:1
```

##### 字节账与复用顺序

```text
same allocation formula as R-007; workspace body use=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1/x2 | tile cast/reduce | output compute | next tile | barriers/queue | merged stats already scalarized |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dtype | FP32 affine/stat | Vector | output cast | numerical path | T1 y rounding only |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 123 | mixed Norm |
| Kernel entry/router | kernel cpp | key 123 | GroupNormSwishNormB16<T1,float> |
| Device body/helper | norm_b16.h | `ComputeSmallHWInner/ComputeNormHWInner` | mixed precision affine |

### Route R-009：Norm FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 213 | FP32、M>8192、Norm guard | `GroupNormSwishNormB32<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
GroupNormSwishNormB32::Compute -> tiled FP32 statistics -> packed-channel/per-channel affine and Swish
```

##### 源码等价伪码

```text
same tile graph as R-007; reduce reads FP32 xUB directly and output computation reuses xUB
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| total UB | local | `163840+8*(CAlign+GAlign+2*LAlign)` B | FP32 | InitLocalB32 | route | pipe | Host limit 4096-2LAlign |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  q["FP32 queues"]:1
  x2["reduce scratch"]:1
  stats["slice/output stats"]:1
```

##### 字节账与复用顺序

```text
UB_total=163840+8*(shapeCAlign+groupPerCoreAlign+2*loopTimesAlign) B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xUB | DMA/reduce | affine/Swish | next tile | FreeTensor after EnQue y | no remaining stat consumer |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| merge/tail | V_S + weighted merge | reduce | scalar formula | last partial tile | FP32 throughout |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 213 | FP32 Norm |
| Kernel entry/router | kernel cpp | key 213 | GroupNormSwishNormB32 |
| Device body/helper | norm_b32.h/base.h | `ComputeSumTwoPass/TwoPassSumOneLoop` | FP32 tile stats |

### Route R-010：Large B16 同型

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 114 | 16-bit same、非 H=1，Small/Norm guard 失败 | `GroupNormSwishLargeB16<T1,T1>` |

#### Device 数据流

##### 函数级调用链

```text
GroupNormSwishLargeB16::ProcessPerCore -> group chunks of 256
  -> CalcMeanAndRstd -> statistics chunks of 1792 tiles -> corrected two-pass merge
  -> CalcGroupNormSwish -> affine chunks of 256 channels -> spatial tiles -> Swish/y
```

##### 源码等价伪码

```text
for groups in chunks(256):
  for each group, merge 8192-element statistics in chunks(1792 tile records)
  load gamma/beta in chunks(256), precompute scale=rstd*gamma and bias=beta-mean*scale
  stream each channel's H elements in 8192 tiles to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| affine chunk | current channel | 256 entries each gamma/beta | FP32 capacity | DMA/cast | scale/bias | pipe | reload every 256 channels |
| stat chunk | tile index modulo 1792 | 3584 FP32 slots each mean/rstd | FP32 | reduce | merge | pipe | merge before overwrite |
| total UB | local | 163840 B | mixed | InitLocal | route | pipe | fixed-size fallback |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  affine["gamma+beta · 2048 B"]:1
  xy["x/y queues · 65536 B"]:1
  calc["x1+x2 · 65536 B"]:1
  stats["mean/rstd Q+Buf · 30720 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=131072+8*(256+256+3584)=163840 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| meanBuf | tile stats then `-mean*gamma` | merge / bias Add | next group/affine chunk | V_S waits | phase ordering in `Compute` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| corrected mean | second reduce of residual | first mean | variance | cancellation control | final tile weighted by actual length |
| chunk output | queue + Pad/atomic tail | tile y | GM | H tail | valid elements only |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 114 | Large fallback |
| Kernel entry/router | kernel cpp | key 114 | GroupNormSwishLargeB16 same |
| Device body/helper | large_b16.h/base.h | `CalcMeanAndRstdInner/CalcGroupNormSwish` | fixed chunks and corrected mean |

### Route R-011：Large B16 + FP32 affine

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 124 | R-010 shape fallback、gammaBytes=4 | `GroupNormSwishLargeB16<T1,float>` |

#### Device 数据流

##### 函数级调用链

```text
key124 -> GroupNormSwishLargeB16<T1,float> -> 256-group/1792-stat/256-affine chunk network -> y
```

##### 源码等价伪码

```text
R-010 with FP32 affine input/output statistics; x tile cast to FP32, final y cast to T1
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| total UB | local | 163840 B | T1+FP32 | Init | route | pipe | same fixed chunks |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  affine["FP32 affine chunks"]:1
  tile["T1 queues + FP32 calc"]:1
  stats["FP32 chunked stats"]:1
```

##### 字节账与复用顺序

```text
UB_total=163840 B; workspace body use=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| gamma/beta Q | FP32 DMA | precompute scale/bias | next 256 channels | FreeTensor | current channels done |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| affine | FP32 Muls/Add | stats | x tile | scale/bias | output cast only |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 124 | mixed Large |
| Kernel entry/router | kernel cpp | key 124 | GroupNormSwishLargeB16<T1,float> |
| Device body/helper | large_b16.h | constexpr affine cast branch | FP32 affine chunk |

### Route R-012：Large FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | key 214 | FP32、非 H=1，Small/Norm guard 失败 | `GroupNormSwishLargeB32<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
GroupNormSwishLargeB32::ProcessPerCore -> 256-group chunks -> 1792-stat chunks -> 256-affine chunks -> tiled y
```

##### 源码等价伪码

```text
same chunk graph as R-010; xUB is directly reduced/modified, yUB stores output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| total UB | local | 196608 B | FP32 | InitLocalB32 | route | pipe | fixed fallback allocation |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  affine["gamma+beta · 2048 B"]:1
  xy["x/y queues · 131072 B"]:1
  x2["x2 · 32768 B"]:1
  stats["stats · 30720 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=163840+8*(256+256+3584)=196608 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xUB | DMA/reduce/affine | y calculation | next tile | queue lifecycle | output EnQue completed |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all stages | FP32 + PIPE_V/V_S | tile reducer | scalar/Vector | shared scratch | tail uses actual num |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | key 214 | FP32 Large fallback |
| Kernel entry/router | kernel cpp | key 214 | GroupNormSwishLargeB32 |
| Device body/helper | large_b32.h/base.h | `ComputeSum/CalcGroupNormSwishInner` | direct FP32 chunk implementation |

## 算子特定优化

### Trick T-001：shape family 与固定 UB 方程联合分流

#### 触发与源码依据

- 适用 Route：R-004…R-012
- 触发条件：非 `hwNum==1`，并按 `M<=8192` 与 `shapeCAlign+groupPerCoreAlign` 的资源式选择 Small/Norm/Large。
- Golden 来源：[`norm/group_norm_swish/op_host/group_norm_swish_tiling.cpp:238-255`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_host/group_norm_swish_tiling.cpp#L238-L255)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tilingData.get_hwNum() == 1) {
    tilingKey += SHAPE_HW1_TILING_KEY;
} else {
    if (tilingData.get_numPerGroup() <= ProcessSize) {
        int64_t limit = ProcessSize * 2 / xDtypeSize - 8;
        if (tilingData.get_shapeCAlign() + tilingData.get_groupPerCoreAlign() <= limit) {
            tilingKey += SHAPE_SMALL_TILING_KEY;
        } else {
            tilingKey += SHAPE_LARGE_TILING_KEY;
        }
    } else {
        int64_t limit = ProcessSize * 2 / xDtypeSize - tilingData.get_loopTimesAlign() * 2;
```

#### 机制与设计原因

Small 把整组留在 UB，Norm 以 8192 元素切片但保留全部切片统计，二者所需可变区分别对应常数 8 和 `2*loopTimesAlign`；Host 在 launch 前计算上界，资源不足时切到固定 chunk 的 Large，避免 Device 侧分配越界。该选择改变的是 UB 物化范围和 DMA 粒度，不是数学语义。

#### 迁移边界

- 适用：同样具有“整域/分片统计/固定 chunk fallback”的纯 Vector 归一化。
- 前提：tile=8192、32B affine 对齐、POD 与 `InitLocal*` 字节式保持一致。
- 失效：改变 Queue depth、统计 slot 或 dtype 字节数时必须同步重推 limit；否则应进入 Large/failure。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓给出的真实 fallback 是 Large；忽略 limit 仍强制 Small/Norm 会使源码分配式失去 Host 上界保证。

### Trick T-002：Large 路线预计算 256 通道 scale/bias

#### 触发与源码依据

- 适用 Route：R-010…R-012
- 触发条件：Large fallback；每批 affine chunk 为 256 通道。
- Golden 来源：[`norm/group_norm_swish/op_kernel/group_norm_swish_large_b32.h:183-209`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_large_b32.h#L183-L209)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
this->CopyInGammaAndBeta(channelIdGlobal, inQueSize, inQueSize);
LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
Muls(gammaLocal, gammaLocal, rstd, inQueSize);
PipeBarrier<PIPE_V>();
Muls(this->meanUb, gammaLocal, -mean, inQueSize);
PipeBarrier<PIPE_V>();
Add(betaLocal, this->meanUb, betaLocal, inQueSize);
```

#### 机制与设计原因

`gammaLocal` 原地变为 `rstd*gamma`，`betaLocal` 原地变为 `beta-mean*scale`；随后同一通道的所有 H tile 只执行一次 Muls+Adds，而不重复标量组合。`meanUb` 在统计阶段消费完后复用为 `-mean*scale` scratch，减少额外 TBuf。

#### 迁移边界

- 适用：归一化统计固定后、同一通道跨多个空间 tile 重用 affine 系数。
- 前提：统计阶段已完成；gamma/beta chunk 在所有对应 H tile 完成前不释放。
- 失效：group/channel 映射或 affine 参数逐元素变化时不可复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的边界是每跨过 256 通道即 Free/重载 affine queue；跨 chunk 继续读取旧 LocalTensor 会违反生命周期。

### Trick T-003：非 32B 对齐输出的最后一块原子覆盖协议

#### 触发与源码依据

- 适用 Route：全部 Route 在非 `DataCopyPad` 编译分支且 `copyNum%elementsPerBlock!=0`。
- 触发条件：输出元素数不是 32B 块整数倍。
- Golden 来源：[`norm/group_norm_swish/op_kernel/group_norm_swish_base.h:223-237`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish/op_kernel/group_norm_swish_base.h#L223-L237)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (copyNum % elementsPerBlockT1 == 0) {
    DataCopy(dstGM, srcUB, copyNum);
} else {
    int64_t copyNumAlign = CeilDiv(copyNum, elementsPerBlockT1) * elementsPerBlockT1;
    int64_t copyNumTail = copyNumAlign - elementsPerBlockT1;
    for (int64_t i = copyNum; i < copyNumAlign; i++) {
        srcUB(i) = 0;
    }
    if (copyNum > elementsPerBlockT1) {
        DataCopy(dstGM, srcUB, copyNum);
    }
    SetAtomicAdd<T1>();
    DataCopy(dstGM[copyNumTail], srcUB[copyNumTail], elementsPerBlockT1);
    SetAtomicNone();
```

#### 机制与设计原因

源码把 UB padding 置零，先提交完整前段，再对最后一个 32B block 开 `SetAtomicAdd` 搬运，零 padding 不改变邻接 GM 元素。这样在无 Pad API 的编译分支中避免尾块越界值覆盖。

#### 迁移边界

- 适用：GM 输出尾块不足 32B，且目标初值/前段写入协议与原子加一致。
- 前提：无效 lane 必须清零，Atomic 必须在尾块后立刻 `SetAtomicNone()`。
- 失效：不能保证 padding 为零或目标不满足原子语义时应使用受支持的 Pad 搬运。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码反例边界是直接按向上对齐长度普通写出：它会覆盖有效域后的邻接元素，因此实现显式改用零填充+Atomic tail。

## 模板与实现对比

| 对比维度 | HW1（R-001…003） | Small（R-004…006） | Norm（R-007…009） | Large（R-010…012） | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | H=1，最高 | H≠1、M≤8192、资源式通过 | M>8192、资源式通过 | 资源式失败 fallback | Host 决策顺序固定 |
| 模板与 dtype | B16 same/mixed、B32 | B16 same/mixed、B32 | B16 same/mixed、B32 | B16 same/mixed、B32 | key 百/十位编码 dtype |
| 循环与分核 | equal/one/multi Device 子分支 | 整组或逐通道 | 8192 stats + packed/per-channel | 256 group、1792 stat、256 affine chunk | shape/UB |
| Queue/TBuf | equal 特化；regular 188416 B | 可变 C/G | 可变 C/G/L | 固定 160/192 KiB | 避免超 UB |
| 内存总占用与复用 | 98–188 KiB | 公式上界 | 公式上界 | 固定 | 见字节账 |
| 搬运/Vector API | direct copy、Reduce、Muls/Add/Exp/Div | whole-group/channel DMA | tiled DMA + weighted merge | chunked DMA + precomputed scale/bias | 搬运粒度 |
| 同步、精度与 tail | PIPE_ALL/V_S；FP32 stats | PIPE_V/V_S | 分片 V_S merge | phase/chunk barriers | valid tail 长度 |
| fallback/失败边界 | 无内部 Host fallback | Large | Large | unknown key no-op | Host key 唯一 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` config | 公开支持；`ascend910_93` config | 公开支持；见独立 `_950` 卡 |
| Host/Launcher | 共用 Tiling/OpDef，16 MiB system workspace | 同 A2 | 共用 Host，OpDef 指向 `_apt` entry |
| Kernel/资源 | 非 arch35 body | 同一非 arch35 body | arch35 body；算法/分配相同，编译 guard 文本不同 |

## 限制与失败边界

- 公开 API 约束 rank 2–8、非空、N/C>0；Tiling 自身只显式检查 rank≥2、dtype/affine shape 与 group 整除，调用者不能绕过 ACLNN 上层约束。
- 公开 ACLNN 文档的 beta 表格把“维度(shape)”写为 2，但同表语义只要求元素数 C，`CheckInputParams` 明确要求 beta `GetDimNum()==1`；实现复用应以固定 Host 的 1D 检查为准，并保留这一文档冲突，不能推测为 2D 支持。
- `hwNum`、`N*numGroups`、`shapeD*hwNum` 使用整数乘法且源码未见显式溢出检查；不得扩展到超范围 shape。
- `ubSizePlatForm` 被查询但没有进入选择式；资源安全依赖固定 tile/limit 与目标平台实际 UB，不应把该实现直接迁移到 UB 更小平台。
- `data_format` 在 Tiling 未读取，语义按公开 NCHW；不能据此推导 NHWC 支持。
- workspace 为 16 MiB 系统保留区，算子 body 不读；不能将其当作可复用的算子中间结果。
- router 未知 key 没有错误/fallback 输出；正常 ACLNN 路径由 Host 生成列出的 12 个 key。
- 固定仓未包含生成 ACLNN inner bridge 和 tiling-data 宏布局实现；前者按框架边界，后者导致精确 POD offsets/sizeof 不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/group_norm_swish/docs/aclnnGroupNormSwish.md` | 两段式原型、产品表 | 公开 API、A2/A3 支持、参数约束 |
| 入口实现 | `norm/group_norm_swish/op_host/group_norm_swish_def.cpp` | `ops::GroupNormSwish` | AutoContiguous、dtype、平台 config |
| Host/Tiling | `norm/group_norm_swish/op_host/group_norm_swish_tiling.cpp` | `Tiling4GroupNormSwish` | 校验、分核、key、workspace |
| POD | `norm/group_norm_swish/op_host/group_norm_swish_tiling.h` | `GroupNormSwishTilingData` | 字段顺序/类型 |
| Kernel entry | `norm/group_norm_swish/op_kernel/group_norm_swish.cpp` | `group_norm_swish` | 12 key router |
| 公共 body | `norm/group_norm_swish/op_kernel/group_norm_swish_base.h` | `GroupNormSwishBase` | Queue、DMA、reduce、Swish、tail |
| HW1 B16 body | `norm/group_norm_swish/op_kernel/group_norm_swish_hw1_b16.h` | `GroupNormSwishHW1B16` | keys 111/121 |
| HW1 B32 body | `norm/group_norm_swish/op_kernel/group_norm_swish_hw1_b32.h` | `GroupNormSwishHW1B32` | key 211 |
| Small B16 body | `norm/group_norm_swish/op_kernel/group_norm_swish_small_b16.h` | `GroupNormSwishSmallB16` | keys 112/122 |
| Small B32 body | `norm/group_norm_swish/op_kernel/group_norm_swish_small_b32.h` | `GroupNormSwishSmallB32` | key 212 |
| Norm B16 body | `norm/group_norm_swish/op_kernel/group_norm_swish_norm_b16.h` | `GroupNormSwishNormB16` | keys 113/123 |
| Norm B32 body | `norm/group_norm_swish/op_kernel/group_norm_swish_norm_b32.h` | `GroupNormSwishNormB32` | key 213 |
| Large B16 body | `norm/group_norm_swish/op_kernel/group_norm_swish_large_b16.h` | `GroupNormSwishLargeB16` | keys 114/124 |
| Large B32 body | `norm/group_norm_swish/op_kernel/group_norm_swish_large_b32.h` | `GroupNormSwishLargeB32` | key 214 |
| A2 binary config | `norm/group_norm_swish/op_host/config/ascend910b/group_norm_swish_binary.json` | `op_list` | 五组 dtype binary |
| A3 binary config | `norm/group_norm_swish/op_host/config/ascend910_93/group_norm_swish_binary.json` | `op_list` | 五组 dtype binary |
| 构建接线 | `norm/group_norm_swish/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | ACLNN Host 模块 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 17 字段 producer/consumer | 字段顺序、int64/int32/float 位宽、8192 单位 | `GroupNormSwishTilingData` |
| 调用顺序 | validate→POD/key→单 launch→Init/Process | router key 和模板实例对应 | `Tiling4GroupNormSwish`, `group_norm_swish` |
| workspace/内存 | body 不读 workspace；UB 公式可复用 | 16 MiB reservation、Queue depth、chunk 常量 | Tiling/`InitLocal*` |
| 同步 | Queue、PIPE_V/ALL、V_S | scalar 读取前等待；复用前 FreeTensor | base 与各 body |
| 精度与尾块 | FP32 stats/affine、T1 输出 cast、Pad/atomic tail | RoundMode 编译 guard、实际 tail 长度 | `GetRoundMode`, `CopyOutWithOutPad*` |
| 失败边界 | invalid input `GRAPH_FAILED`；Large 是资源 fallback | 未知 key 不写；不可绕过 ACLNN 约束 | `Check*`, `SetTilingKey`, router |