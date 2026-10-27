---
type: Operator
title: "GroupNormBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnGroupNormBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/docs/aclnnGroupNormBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/op_api/aclnn_group_norm_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/op_api/group_norm_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/group_norm_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/group_norm_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_kernel/group_norm_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_kernel/group_norm_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/config/ascend910b/group_norm_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/config/ascend910_93/group_norm_grad_binary.json
    role: implementation
created_at: "2026-08-25T17:56:20Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# GroupNormBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnGroupNormBackwardGetWorkspaceSize` + `aclnnGroupNormBackward`，公开 ACLNN 两段式接口；固定仓只有这一版本族。 |
| 计算语义 | 对 `[N,C,HxW]` 按 `G=group` 分组；`xHat=(x-mean)*rstd`，输出 `dbeta=sum(dy)`、`dgamma=sum(dy*xHat)` 与 `dx=rstd*gamma*(dy-(dbeta+xHat*dgamma)/(C/G*HxW))`。 |
| 输入输出 | `dy/x/dx` 同 dtype 与 shape；A2/A3 上 `mean/rstd/gamma/dgamma/dbeta` 与 `dy` 同 dtype；dtype 为 FP32/FP16/BF16，ND。 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；不存在 Cube/AIC、SIMT、NDDMA、通信或 MIX_CV 节点。 |
| Kernel 路线 | R-000 空输入无 launch；R-001 `HXW=1`；R-002 整组全载；R-003 单通道全载；R-004 双通道流水；R-005 `HXW` 分片重算。 |
| 核心资源 | GM 输入/输出；AIV UB `TQue/TBuf`；FP32 中间量；两片 dgamma/dbeta workspace；跨核 `SyncAll` 与 AtomicAdd。 |
| 来源闭包 | `source_closed`：公开 ACLNN→参数检查/Contiguous/Reshape→`l0op::GroupNormGrad`→Tiling/POD→`group_norm_grad` router→`GroupNormGrad<T,isDeterministic>` body→binary config 均在固定仓闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `D=C/G`、`M=D*HXW`，`n,g,c,h` 分别索引 batch、group、组内 channel 与空间元素：

```text
xhat[n,g,c,h] = (x[n,g,c,h] - mean[n,g]) * rstd[n,g]
dbeta[g,c]    = sum_n,h dy[n,g,c,h]
dgamma[g,c]   = sum_n,h dy[n,g,c,h] * xhat[n,g,c,h]
dx            = rstd * gamma[c] * (dy - (sum_c,h(dy*gamma)
                 + xhat*sum_c,h(dy*gamma*xhat)) / M)
```

Device 实现把 x 的 UB 副本原地归一化为 `xhat`，再求每 channel 的 `db=sum_h(dy)` 与 `dg=sum_h(dy*xhat)`；组内对 `db*gamma`、`dg*gamma` 归约后形成 dx 的两个标量项，不在 GM 物化 `xhat`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOut`/`input` | 任意 0–8 维视图，经 API reshape 为 `[N,C,HxW]`/ND | FP32/FP16/BF16，同 dtype | numel 都为 `N*C*HxW`；允许非连续。 |
| `mean`/`rstd` | API reshape 为 `[N,G]`/ND | 与 `gradOut` 相同 | numel=`N*G`，两者同 shape/dtype。 |
| `gamma` | `[C]`/ND | 与 `gradOut` 相同 | 可为 nullptr；此时用同 dtype 的全 1 Tensor；非空时 numel=`C`。 |
| `N/C/HxW/G` | 标量 | int64 | `G>0`，`C%G=0`，`C/G<=8000`，`C<=240000`；numel 关系同时约束零域。 |
| `outputMask` | 长度 3 | bool | 依次控制 `dx/dgamma/dbeta`；为 true 的输出指针必须非空。 |
| `gradInput` | 与 `input` view shape 相同 | 与 `gradOut` 相同 | Kernel 输出先为 `[N,C,HxW]`，再 reshape/ViewCopy。 |
| `gradGammaOut`/`gradBetaOut` | 与 `gamma` shape 相同 | 与 `gradOut` 相同 | Kernel 输出 `[C]`，ViewCopy 支持非连续目的 Tensor。 |

### 入口与版本

公开文档的产品表明确 A2、A3 支持；`GroupNormGrad` OpDef 同时注册 `ascend910b` 与 `ascend910_93`，两个 binary config 都接线 FP32/FP16/BF16。API 在 `DAV_2201` 分支直接添加 `GroupNormGrad`，A2/A3 共用非 arch35 Tiling 和 Kernel body。框架 `Contiguous/Reshape/ViewCopy/CommonOpExecutorRun` 是明确的框架边界；算子拥有的 selector、POD、router 和 body 已继续闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 依次执行：非空检查→mask 长度和按位输出指针→dtype→`G/C` 属性→numel/输出 shape。非空路径随后对五个输入 `Contiguous`；`gamma==nullptr` 时 `FillValue(...,1)`；再 reshape 为 `[N,C,HxW]`、`[N,G]`、`[C]`。A2/A3 若 `gradOut` 或 `input` 为空，在这些转换前直接返回 workspace 0，不添加 launch。否则分配三个内部输出，`l0op::GroupNormGrad` 通过 `ADD_TO_LAUNCHER_LIST_AICORE` 添加一个节点，按 mask 对结果做 reshape/ViewCopy。任一中间 Tensor 或 launcher 添加失败立即返回，后续节点不执行。

### 参数、Workspace 与生命周期

`CREATE_EXECUTOR` 拥有 Contiguous/Fill、内部输出、launcher 和 ViewCopy；第一段用 `GetWorkspaceSize()` 返回总空间并把 owner 移交给调用方 executor，第二段在传入 stream 上 `CommonOpExecutorRun`。

令 `W` 为每片 FP32 workspace 元素数，系统库空间为 `S`：

| runtime key | 条件 | `W` | 用户 workspace 字节 | 用途 |
|---|---|---:|---:|---|
| 0 | FP32 且 `N<=2` | 0 | 0 | 直接 AtomicAdd 到 FP32 输出。 |
| 1/2 | FP16/BF16 且 `N<=2` | `C`（`N=1` 时不读） | `2*C*4` | `N=2` 先累加 FP32，再 cast；`N=1` 直接 cast 输出。 |
| 10/11/12 | 任意 dtype 且 `N>2` | `ceil(N/2)*C` | `2*W*4` | 前后两片分别为 dgamma/dbeta；每片按 batch-half×C 排布。 |

总 workspace=`S+用户 workspace`；Kernel 以 `GetUserWorkspace` 跳过系统段。workspace 在单 Kernel 完成前有效，`SyncAll` 后才由同一 Kernel 的 stage2 读取。

### 启动图与失败传播

```text
aclnnGroupNormBackwardGetWorkspaceSize
  -> CheckParams -> Contiguous/Fill -> Reshape
  -> l0op::GroupNormGrad / ADD_TO_LAUNCHER_LIST_AICORE
     -> [single AIV node] group_norm_grad(..., usrWorkspace, tilingData)
        -> stage1: each core handles contiguous (n,g) tasks
        -> SyncAll
        -> optional stage2: workspace N-reduce and/or FP32->T cast
  -> Reshape/ViewCopy(masked outputs)
aclnnGroupNormBackward
  -> CommonOpExecutorRun(workspace, executor, stream)
```

节点 blockDim=`coreNumUsed=ceil(N*G/taskNumPerCore)`；POD 为 `GroupNormGradTilingData`。唯一跨阶段边是 workspace：stage1 producer `Mode_Selection/fp32_dgamma_dbeta2GM`，stage2 consumer `cast_dgamma_dbeta_WSP2GM` 或 `reduce[_cast]_dgamma_dbeta_WSP2GM`；offset 见上表，dtype FP32，ordering 为同 Kernel 的 `SyncAll`，owner 为 ACLNN workspace。空输入 R-000 不创建节点。unknown runtime key 没有 body 分支，Host 只产生 0/1/2/10/11/12。

## Route 与参数设计

### 分核与资源计算

记 `Q=N*G`、`D=C/G`、`H=HXW`、`K=totalCoreNum`、`B=32`，`s=sizeof(T)`。Tiling 计算：

```text
taskNumPerCore = align_up(Q,K)/K
coreNumUsed = ceil(Q/taskNumPerCore)
tailCore/taskNumPerTailCore = Q % coreNumUsed ?
  (Q % coreNumUsed, taskNumPerCore-1) : (coreNumUsed, taskNumPerCore)
dtypeBytes = 4(FP32), 6(FP16/BF16，含 T+FP32 中间量)
elePerBlock = 8(FP32), 16(FP16/BF16)
reserve = 1024 + align_up(D,elePerBlock)*dtypeBytes*3
mode0Cap = (UB-reserve-unalignedExtra)/(align_up(D*H,elePerBlock)*dtypeBytes*3)
mode1Cap = (UB-reserve)/(align_up(H,elePerBlock)*dtypeBytes*3)
```

`unalignedExtra=0` 当 `H%8==0 || D==1`，否则为 `2*align_up(H,elePerBlock)*4`。所有量先在 Host `uint32_t` 中计算；API 的 int64 numel 乘法及 Tiling 向 uint32 窄化没有独立溢出保护，是固定实现边界。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 0 | R-000 空输入 | 无 Kernel | `gradOut->IsEmpty() \|\| input->IsEmpty()` | 非空进入 R-001。 |
| 1 | R-001 H=1 | POD `Tiling_key=5` | `H==1`，优先于所有 UB capacity 判断 | 否则 R-002。 |
| 2 | R-002 整组全载 | POD mode 0 | `H!=1 && mode0Cap>0` | 否则 R-003。 |
| 3 | R-003 单通道全载 | POD mode 1 | `mode0Cap<=0 && mode1Cap==1` | 否则 R-004。 |
| 4 | R-004 双通道流水 | POD mode 2 | `mode1Cap>1` | 否则 R-005。 |
| 5 | R-005 H 分片 | POD mode 3 | `mode1Cap<=0` 且 `floor((UB-reserve)/(dtypeBytes*3),elePerBlock*8)>0` | 否则 Tiling 失败。 |

```text
empty input -> no launch
else H==1 -> mode5
else mode0Cap>0 -> mode0
else mode1Cap==1 -> mode1
else mode1Cap>1 -> mode2
else -> mode3; tile=align_down((UB-reserve)/(dtypeBytes*3), elePerBlock*8)
runtime key = dtype(0/1/2) + (N>2 ? 10 : 0)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-000 | API branch | A2/A3 且 x/dy 空 | 无 Device 节点 | API early return | workspace 0 | 输出未由算子写入。 |
| R-001 | mode5 + runtime 0/1/2/10/11/12 | `H=1` | 单 AIV，stage1+可选 stage2 | `Compute_Mode_5` | channel 向量直接处理 | `D` mask tail。 |
| R-002 | mode0 + runtime keys | 整个 `D*H` 可入 UB | 单 AIV，stage1+可选 stage2 | `CopyIn_Mode_0/Compute_Mode_0` | 一次载入整组；非对齐 channel 重载 | `H` DMA/mask tail。 |
| R-003 | mode1 + runtime keys | 单 channel 可入 UB，capacity=1 | 单 AIV，stage1+可选 stage2 | `Compute_Mode_1` | channel 串行 | 每 channel 有效 H。 |
| R-004 | mode2 + runtime keys | capacity>1 | 单 AIV，stage1+可选 stage2 | `Compute_Mode_2` | depth=2 的 x/dy/temp 流水 | 奇数 D 单独处理。 |
| R-005 | mode3 + runtime keys | 单 channel 也放不下 | 单 AIV，stage1+可选 stage2 | `Compute_Mode_3` | H 切为 `q`，补偿求和 | 最后一 tile=`mode2UbTailNum`。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 组承载 | `align_up(D,elePerBlock)<=8000` | POD/UB 与逐 channel 循环上界 | 全部 | Tiling 失败 | `GroupNormGradTiling::Init` |
| 非对齐重载 | `H%8!=0 && D!=1` | mode0 中 channel 起点可能不在 FP32 block 边界 | R-002 | 预留两个 H-sized TBuf | `CalStage1TilingInfo`、`Compute_Mode_0` |
| deterministic key | `N>2` | 避免跨超过两批次的直接 Atomic 累加顺序 | 全部非空 | runtime key +10、启用 stage2 | `GroupNormGradTiling::Init` |
| mode3 tile | `q=floor(...,elePerBlock*8)>0` | 三份 tile UB 且 Vector repeat 对齐 | R-005 | Tiling 失败 | `CalStage1TilingInfo` |
| stage2 UB | `coreBatchParts>0` | 一次装入若干 N×channel workspace 行 | deterministic | Tiling 失败 | `ComputeAllocUBStage2` |
| blockLen 窄化 | stage2 `count*sizeof(float)` 写入 `uint16_t` | `DataCopyExtParams` 位宽 | stage2 | 固定仓无额外 fallback | `reduce_axis_n_WSP2UB` |

### POD/ABI

`GroupNormGradTilingData` 全部字段由 tiling 宏生成；固定仓可确认字段顺序、源码类型和 producer/consumer，但宏展开后的 `sizeof/alignof/offsetof` 不在仓内，故各行 ABI 布局均为“固定仓不可见”。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| GroupNormGradTilingData | `Tiling_key` | uint64/64 | 固定仓不可见 | mode | `CalStage1TilingInfo` | `Process/Init_Unified_Buffer` | 0/1/2/3/5；4 不可达。 |
| 同上 | `N` | uint64/64 | 同上 | element-axis | dy dim0 | ctor/stage2 | API 非空路线上 >0。 |
| 同上 | `C` | uint64/64 | 同上 | element-axis | dy dim1 | 地址、stage2 stride | <=240000。 |
| 同上 | `HXW` | uint64/64 | 同上 | element | dim[2:] product | ctor | >0。 |
| 同上 | `G` | uint64/64 | 同上 | group | attr0 | channel offset | >0。 |
| 同上 | `NXG` | uint64/64 | 同上 | task | `N*G` | GM extent | >0。 |
| 同上 | `C_G` | uint64/64 | 同上 | channel/group | `C/G` | all route loops | 1..8000。 |
| 同上 | `task_num_per_core` | uint64/64 | 同上 | task/core | `PlanStepCoreUsage` | ctor | former core。 |
| 同上 | `task_num_per_tail_core` | uint64/64 | 同上 | task/core | 同上 | ctor | tail cores。 |
| 同上 | `tail_core` | uint64/64 | 同上 | core index | 同上 | ctor | 分界。 |
| 同上 | `mode1_ub_cap_C_num` | uint64/64 | 同上 | channel/tile | `CalStage1TilingInfo` | mode2 loops indirectly by selected route | mode0/3 inactive。 |
| 同上 | `mode1_ub_iter_C_num` | uint64/64 | 同上 | loop | `ceil(D/cap)` | 固定 body 未读取 | 保留 ABI。 |
| 同上 | `mode1_ub_tail_C_num` | uint64/64 | 同上 | channel | Host tail | 固定 body 未读取 | 保留 ABI。 |
| 同上 | `mode2_ub_capacity_ele` | uint64/64 | 同上 | element | mode3 UB floor | `Compute_Mode_3` | 仅 R-005。 |
| 同上 | `mode2_ub_iteration_num` | uint64/64 | 同上 | loop/channel | `ceil(H/q)` | R-005 | 仅 R-005。 |
| 同上 | `mode2_ub_tail_num` | uint64/64 | 同上 | element | `H-(loops-1)q` | R-005 | 仅 R-005。 |
| 同上 | `workSpaceSize` | uint64/64 | 同上 | FP32 element/slice | stage2 tiling | workspace GM binding | 0/C/ceil(N/2)C。 |
| 同上 | `stage2CoreUsed` | uint64/64 | 同上 | core | stage2 tiling | `Process` | inactive route may be 0。 |
| 同上 | `castEleNum` | uint64/64 | 同上 | channel/core | aligned C split | stage2 | 64-element aligned。 |
| 同上 | `tailCastNum` | uint64/64 | 同上 | channel | C tail | stage2 | last core。 |
| 同上 | `coreBatchParts` | uint64/64 | 同上 | N rows/chunk | `ComputeAllocUBStage2` | stage2 reduce | deterministic only。 |
| 同上 | `coreBatchPartsTailRepeat` | uint64/64 | 同上 | N rows | Host | 固定 body 未读取 | 保留 ABI。 |
| 同上 | `repeatTime4Stage2` | uint64/64 | 同上 | loop | Host | 固定 body未读取 | 保留 ABI。 |
| 同上 | `dx_is_require` | uint64/64 | 同上 | bool-as-u64 | attr2 | all Compute | 0/1。 |
| 同上 | `dgamma_is_require` | uint64/64 | 同上 | bool-as-u64 | attr3 | store/stage2 | 0/1。 |
| 同上 | `dbeta_is_require` | uint64/64 | 同上 | bool-as-u64 | attr4 | store/stage2 | 0/1。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-000 | empty early return | `aclnnGroupNormBackwardGetWorkspaceSize` | N/A | N/A | N/A | API source |
| R-001 | mode5 | `CalStage1TilingInfo` | `GroupNormGradTilingData` | `group_norm_grad` runtime keys | `Compute_Mode_5` + helpers | ascend910b/ascend910_93 config |
| R-002 | mode0 | 同上 | 同上 | 同上 | `CopyIn_Mode_0/Compute_Mode_0` | 同上 |
| R-003 | mode1 | 同上 | 同上 | 同上 | `Compute_Mode_1` | 同上 |
| R-004 | mode2 | 同上 | 同上 | 同上 | `Compute_Mode_2` | 同上 |
| R-005 | mode3 | 同上 | 同上 | 同上 | `CopyIn_x/compute_dgamma/Compute_Mode_3` | 同上 |

## Kernel 实现

### 公共执行流程

router 把 runtime key `0/1/2` 映射到 `GroupNormGrad<float/half/bfloat16,false>`，把 `10/11/12` 映射到相同 dtype 的 `<...,true>`。构造函数解析 POD、把 block 映射到连续 `(n,g)` task，绑定 GM 与两片 workspace，core0 初始化目的 GM/workspace，随后 `SyncAll`。每个 route 都在 FP32 中求 `dbeta` 和 `dgamma=sum(dy*xhat)`，通过 `Mode_Selection` 写输出或 workspace；若需要 dx，再读取 gamma、求两个 group 标量并完成公式。stage1 后 `pipe.Reset()+SyncAll`；非 FP32 非确定路线执行 cast，确定路线执行 workspace N 轴归约（FP32 输出用 Atomic 合并两半，16 位输出归约后 cast）。

### Route R-000：空输入无 launch

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_group_norm_backward.cpp` | `aclnnGroupNormBackwardGetWorkspaceSize` | `(gradOut->IsEmpty() \|\| input->IsEmpty()) && !IsRegbase` | workspace=0，释放 executor owner，无 launch。 |

#### Device 数据流

##### 函数级调用链

```text
GetWorkspaceSize -> empty predicate -> ReleaseTo(executor) -> return ACLNN_SUCCESS
```

##### 源码等价伪码

```text
if x.empty || dy.empty: workspaceSize=0; return
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Device | N/A | 0 | N/A | N/A | N/A | 无 Device 生命周期 | 无搬运。 |

##### 分层内存布局

N/A：无 AIV 节点和 UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | API 返回 | 输入满足空域。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| API | early return | 参数检查 | caller | executor | 不写输出。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | `op_host/op_api/aclnn_group_norm_backward.cpp` | `aclnnGroupNormBackwardGetWorkspaceSize` | A2/A3 空 Tensor 无 launch。 |
| Kernel entry/router | N/A | N/A | 无节点。 |
| Device body/helper | N/A | N/A | 无节点。 |

### Route R-001：HXW=1 专用向量路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | `hxw==1` | POD mode 5。 |
| Router | `op_kernel/group_norm_grad.cpp` | `group_norm_grad` | runtime key by dtype/N | `GroupNormGrad<T,det>::Process`。 |

#### Device 数据流

##### 函数级调用链

```text
group_norm_grad -> GroupNormGrad<T,det>::Process
  -> CopyIn_Mode_0 -> Compute_Mode_5 -> Mode_Selection
  -> [SyncAll] -> optional workspace reduce/cast stage2
```

##### 源码等价伪码

```text
for each (n,g) task: load D x and dy; x=(x-mean)*rstd
dbeta=dy; ds=dy*xhat
store partial dgamma/dbeta
if dx: sum1=sum(ds*gamma)/D; sum2=sum(dbeta*gamma)/D
       dx=rstd*gamma*dy-(rstd*sum1+rstd*sum2*xhat)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/dy/dx | `task*D` | D | T/contiguous | DMA | mode5 | ACLNN tensors | effective D。 |
| UB queues | 0 | FP32: `6*A_D*4`; 16-bit: `34*A_D` bytes（`A_D=align_up(D,32/s)`） | mixed | InitBuffer | mode5 | Kernel | depth 1。 |
| workspace | common formula | common formula | FP32 | stage1 | stage2 | executor | two slices。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  dy["dyQ · 1 · A_D*4"] x["xQ · 1 · A_D*4"] gamma["gammaQ · 1 · A_D*4"]
  db["dbetaQ · 1 · A_D*4"] ds["dgammaQ · 1 · A_D*4"] tmp["tempQ · 1 · A_D*4"]
```

16 位另有 dy/x/gamma 与两个输出 cast TBuf，逐项见 `Init_Unified_Buffer_FP16_BF16_MODE_5`。

##### 字节账与复用顺序

```text
UB_total_fp32 = 6*A_D*4
UB_total_16 = 34*A_D
workspace_total = S + 2*W*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tempQ | `Mul/Sub` | dx CopyOut | 下一 task | `FreeTensor` | 当前输出已提交。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| vector | mask/queue | CopyIn | Compute | UB | 16 位先 cast FP32，末端 `CAST_ROUND`；mask=D。 |
| cross-core | `SyncAll` | stage1 | stage2 | workspace | 空闲 lanes 不提交。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | H=1 最高优先级。 |
| Kernel entry/router | `op_kernel/group_norm_grad.cpp` | `group_norm_grad` | dtype/deterministic 实例。 |
| Device body/helper | `op_kernel/group_norm_grad.h` | `Compute_Mode_5` | channel 向量化、FP32 中间计算。 |

### Route R-002：整组全载

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | `H!=1 && mode0UbCapGNum>0` | mode0。 |
| Device | `op_kernel/group_norm_grad.h` | `Process` | POD mode0 | `CopyIn_Mode_0/Compute_Mode_0`。 |

#### Device 数据流

##### 函数级调用链

```text
group_norm_grad -> Process -> CopyIn_Mode_0 -> Compute_Mode_0
  -> ReduceSum_template / Mul_ReduceSum_template -> Mode_Selection -> optional stage2
```

##### 源码等价伪码

```text
load whole group [D,H]; xhat=(x-mean)*rstd
for c: db[c]=reduce_h(dy); ds[c]=reduce_h(dy*xhat)
if channel GM start unaligned: reload that channel into dedicated H TBuf before reduce/copyout
store partial gradients; if dx, derive c1/c2 and write each channel
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM group | `task*D*H` | D*H | T | CopyIn | reduce/dx | tensor | contiguous group。 |
| main UB | 0 | FP32 `3*A_E*4+3*A_D*4`; 16-bit `3*A_E*4+2*A_E*s+3*A_D*4+3*A_D*s` | mixed | InitBuffer | mode0 | Kernel | `A_E=align_up(DH,32/s)`。 |
| unaligned TBuf | 0 | optional `2*A_H*4` | FP32 | reload | per-channel path | Kernel | only `H%8!=0 && D!=1`。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  dy["dyQ · 1 · A_E*4"] x["xQ · 1 · A_E*4"] tmp["tempQ · 1 · A_E*4"]
  gamma["gammaQ · 1 · A_D*4"] db["dbetaQ · 1 · A_D*4"] ds["dgammaQ · 1 · A_D*4"]
```

##### 字节账与复用顺序

```text
UB_total_fp32 = 3*A_E*4 + 3*A_D*4 + optional 2*A_H*4
UB_total_16 = 3*A_E*4 + 2*A_E*s + 3*A_D*4 + 3*A_D*s + optional 2*A_H*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xQ | CopyIn，随后原地 xhat/C2*x | reduction/dx | next task | queue lifetime + pipe events | 所有读完成。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→V | `MTE2_V` | DataCopyPad | vector | group UB | H mask；非对齐 channel重载。 |
| V→DMA | `V_MTE3` | dx vector | CopyOut | tempQ | 只写有效 H。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | group fit 选择及额外 UB。 |
| Kernel entry/router | `op_kernel/group_norm_grad.cpp` | `group_norm_grad` | runtime key。 |
| Device body/helper | `op_kernel/group_norm_grad.h` | `CopyIn_Mode_0/Compute_Mode_0` | 全载、重载 tail、reduce/dx。 |

### Route R-003：单通道全载

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | `mode0Cap<=0 && mode1Cap==1` | mode1。 |

#### Device 数据流

##### 函数级调用链

```text
Process -> Get_Mean_Rstd -> Compute_Mode_1 -> per-channel CopyIn/reduce
  -> Mode_Selection -> per-channel reload/dx -> optional stage2
```

##### 源码等价伪码

```text
for c in D: load H x/dy; reduce db,ds; free
store partials
if dx: load gamma[D]; compute group scalars; reload each channel and write H dx
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| channel GM | `task*DH+c*H` | H | T | CopyIn | reduce/dx | tensor | valid H。 |
| UB | 0 | FP32 `3*A_H*4+3*A_D*4`; 16-bit `3*A_H*4+2*A_H*s+3*A_D*4+3*A_D*s` | mixed | InitBuffer | mode1 | Kernel | depth1。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  dy["dyQ · 1 · A_H*4"] x["xQ · 1 · A_H*4"] tmp["tempQ · 1 · A_H*4"]
  gamma["gammaQ · 1 · A_D*4"] db["dbetaQ · 1 · A_D*4"] ds["dgammaQ · 1 · A_D*4"]
```

##### 字节账与复用顺序

```text
UB_total_fp32=3*A_H*4+3*A_D*4
UB_total_16=3*A_H*4+2*A_H*s+3*A_D*4+3*A_D*s
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x/dy queues | reduction pass | freed then dx pass reload | next channel | `FreeTensor` | 两阶段不别名。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| channel | MTE2/V/MTE3 events | DMA/vector | next stage | H buffer | ReduceSum 使用有效 H。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | capacity==1。 |
| Kernel entry/router | `op_kernel/group_norm_grad.cpp` | `group_norm_grad` | runtime key。 |
| Device body/helper | `op_kernel/group_norm_grad.h` | `Compute_Mode_1` | channel 串行两遍。 |

### Route R-004：双通道流水

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | `mode1Cap>1` | mode2。 |

#### Device 数据流

##### 函数级调用链

```text
Process -> Compute_Mode_2
  -> pairwise depth-2 CopyIn/reduce -> odd-channel tail
  -> Mode_Selection -> pairwise reload/dx -> optional stage2
```

##### 源码等价伪码

```text
for c by 2: allocate two x/dy/temp slots; reduce both channels
if D odd: process last channel in slot0
store partials; if dx, repeat by 2 and write outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| depth-2 queues | slot0/slot1 | each `A_H` | FP32 or T | DMA | vector | Kernel | true depth=2。 |
| UB | 0 | FP32 `6*A_H*4+3*A_D*4`; 16-bit `6*A_H*4+2*A_H*s+3*A_D*4+3*A_D*s` | mixed | InitBuffer | mode2 | Kernel | odd D uses slot0。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  dy["dyQ · depth=2 · 2*A_H*4"] x["xQ · depth=2 · 2*A_H*4"] tmp["tempQ · depth=2 · 2*A_H*4"]
  gamma["gammaQ · 1 · A_D*4"] db["dbetaQ · 1 · A_D*4"] ds["dgammaQ · 1 · A_D*4"]
```

##### 字节账与复用顺序

```text
UB_total_fp32=6*A_H*4+3*A_D*4
UB_total_16=6*A_H*4+2*A_H*s+3*A_D*4+3*A_D*s
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| depth slots | current channel pair | reduction/CopyOut | next pair | queue FreeTensor | both slot consumers complete。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pair | queue depth2 + events | DMA | vector | pair buffers | odd D branch explicit。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | capacity>1。 |
| Kernel entry/router | `op_kernel/group_norm_grad.cpp` | `group_norm_grad` | runtime key。 |
| Device body/helper | `op_kernel/group_norm_grad.h` | `Compute_Mode_2` | depth2 与 odd-channel tail。 |

### Route R-005：HXW 分片重算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | `mode1Cap<=0`、`q>0` | mode3，H 分片。 |

#### Device 数据流

##### 函数级调用链

```text
Process -> Get_Mean_Rstd -> Compute_Mode_3
  -> CopyIn_x -> compute_dgamma -> Copyout_dgamma (per tile)
  -> Mode_Selection -> tiled dx pass -> optional stage2
```

##### 源码等价伪码

```text
for channel c:
  for tile t in H: load q/tail x,dy; Kahan-like corrected sums db and ds
  write channel partials; reset corrections
if dx: reload every tile and write q/tail valid elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tile GM | `task*DH+c*H+t*q` | q or tail | T | CopyIn | reduce/dx | tensor | last tail explicit。 |
| UB | 0 | FP32 `3*A_q*4+3*A_D*4`; 16-bit `3*A_q*4+2*A_q*s+3*A_D*4+3*A_D*s` | mixed | InitBuffer | mode3 | Kernel | q aligned `elePerBlock*8`。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  dy["dyQ · 1 · A_q*4"] x["xQ · 1 · A_q*4"] tmp["tempQ · 1 · A_q*4"]
  gamma["gammaQ · 1 · A_D*4"] db["dbetaQ · 1 · A_D*4"] ds["dgammaQ · 1 · A_D*4"]
```

##### 字节账与复用顺序

```text
UB_total_fp32=3*A_q*4+3*A_D*4
UB_total_16=3*A_q*4+2*A_q*s+3*A_D*4+3*A_D*s
q=floor((UB-reserve)/(dtypeBytes*3),elePerBlock*8)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x/dy/temp tile | current tile | corrected reduction or dx | next tile | queue FreeTensor/events | partial scalar已保存。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile reduction | compensation `corr1/corr2` | tile sum | accumulated scalar | FP32 scalar | 减少跨 tile 加法误差；last=`mode2_ub_tail_num`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_grad_tiling.cpp` | `CalStage1TilingInfo` | q/loops/tail producer。 |
| Kernel entry/router | `op_kernel/group_norm_grad.cpp` | `group_norm_grad` | runtime key。 |
| Device body/helper | `op_kernel/group_norm_grad.h` | `Compute_Mode_3` | tile 地址、补偿求和、tail。 |

## 算子特定优化

### Trick T-001：UB 容量驱动的五级实现选择

#### 触发与源码依据

- 适用 Route：R-001–R-005
- 触发条件：按 `H`、整组/单通道 UB capacity 的有序判断。
- Golden 来源：[`group_norm_grad_tiling.cpp:366-389`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_host/group_norm_grad_tiling.cpp#L366-L389)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tilingParams->hxw == 1) {
    tilingParams->tilingKey = MODE_5;
} else if (tilingParams->mode0UbCapGNum > 0) {
    tilingParams->tilingKey = MODE_0;
} else if (tilingParams->mode0UbCapGNum <= 0 && tilingParams->mode1UbCapCNum == 1) {
    tilingParams->tilingKey = MODE_1;
} else if (tilingParams->mode1UbCapCNum > 1) {
    tilingParams->tilingKey = MODE_2;
} else if (tilingParams->mode1UbCapCNum <= 0) {
    tilingParams->tilingKey = MODE_3;
```

#### 机制与设计原因

优先用 H=1 消除空间归约，再尽量整组全载；放不下时依次退化为单/双通道和 H 分片。选择直接改变 GM 重载次数、depth=2 pipeline 与 UB 大小，而 runtime dtype/deterministic key 保持独立轴。

#### 迁移边界

- 适用：同样有“整对象→子对象→分片”容量梯度的 Vector backward。
- 前提：Host 与 Device 对 `dtypeBytes/elePerBlock/reserve` 的字节账一致。
- 失效：q 为 0 时必须失败，不能继续启动 mode3。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：N>2 的两片 workspace 归约

#### 触发与源码依据

- 适用 Route：所有非空 Route 的 runtime key 10/11/12
- 触发条件：`N>2`。
- Golden 来源：[`group_norm_grad.h:1225-1244`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_kernel/group_norm_grad.h#L1225-L1244)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
} else if constexpr (isDeterministic) {
    outChannelIdx = ((task_idx / this->G) / SPLIT_COUNT * this->G + (task_idx % this->G)) * this->C_G;
    fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_workspace, dgamma_ub, dbeta_workspace, dbeta_ub);
}
```

#### 机制与设计原因

Host 用 `ceil(N/2)*C` 为每种参数梯度分片，Device 把相邻 batch 对映射到同一 FP32 行，stage1 后全核同步，再按固定 workspace 行序做 N 轴归约；16 位只在归约完成后降精度。

#### 迁移边界

- 适用：参数梯度需要跨 batch 归约的 AIV Kernel。
- 前提：两片 workspace offset、`SyncAll` 与 stage2 grid 同时复用。
- 失效：N<=2 走直接/单次 cast 路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-003：mode3 的跨 tile 补偿求和

#### 触发与源码依据

- 适用 Route：R-005
- 触发条件：`mode1UbCapCNum<=0`，每 channel 必须切 H。
- Golden 来源：[`group_norm_grad.h:637-676`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_grad/op_kernel/group_norm_grad.h#L637-L676)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
float adjustedSum = ReduceSum_template(temp_hxw_local, dy_local, temp_hxw_local, process_num, true) - corr1;
float tempSum = adjustedSum + temp1I;
corr1 = (tempSum - temp1I) - adjustedSum;
temp1I = tempSum;
```

#### 机制与设计原因

每 tile 仍在 FP32 归约，跨 tile 累加用误差补偿变量，而不是简单串行相加；这改变长 H 路线的精度行为。`process_num` 来自 q 或尾 tile，避免 padding lane 进入结果。

#### 迁移边界

- 适用：单次 UB 放不下的长轴 reduce。
- 前提：补偿变量按 channel 清零，tile 顺序固定。
- 失效：其他路线单次持有完整 channel，不需要跨 tile 补偿。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 H=1 | R-002 整组 | R-003 单通道 | R-004 depth2 | R-005 H 分片 | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | H=1 | group fit | cap=1 | cap>1 | cap<=0 | 有序 selector。 |
| 模板与 dtype | 同 `GroupNormGrad<T,det>` | 同 | 同 | 同 | 同 | mode 是 POD 内部轴。 |
| 循环与分核 | task→D | task→D | task→D | task→D/2 | task→D→tile | block 均映射 (n,g)。 |
| Queue/TBuf | D-sized | DH-sized | H-sized | depth2 H | q-sized | 容量梯度。 |
| 内存总占用与复用 | 6 vectors | 3 group+3 D | 3 H+3 D | 6 H+3 D | 3 q+3 D | 16 位另加 T cast buffers。 |
| 搬运/Vector API | Copy/Mul/Sub | 全组 DMA/reduce | channel DMA | pair DMA | tile DMA+compensation | 都为 AIV Vector。 |
| 同步、精度与 tail | D mask | unaligned reload | H tail | odd D | q tail | 公共 FP32 compute/stage2。 |
| fallback/失败边界 | N/A | →R-003 | →R-004 | →R-005 | q=0失败 | 无 Device fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 见独立 `_950` 卡 |
| Host/Launcher | `DAV_2201` 单 `GroupNormGrad`；ascend910b config | 同 body；ascend910_93 config | arch35 selector/POD 不同 |
| Kernel/资源 | 非 arch35 `GroupNormGrad<T,det>` | 同 | 四个 MicroAPI 模板族+empty key |

## 限制与失败边界

- API 要求 `G>0`、`C%G==0`、`C/G<=8000`、`C<=240000`，五个输入 shape/numel 与 mask 输出 shape/dtype 匹配；Tiling 对非空路径进一步要求 `N*G>0`、`H>0`。
- `GroupNormGradTilingData` 宏展开 ABI 的 `sizeof/alignof/offsetof` 固定仓不可见；只能复用字段顺序、类型和 producer/consumer，不能猜测偏移。
- `mode1_ub_iter_C_num`、`mode1_ub_tail_C_num`、`coreBatchPartsTailRepeat`、`repeatTime4Stage2` 被 Host 写入但固定 Device body 不读取；不能赋予新语义。
- 公开文档声明可通过系统参数开启确定性，但固定 A2/A3 Tiling 未读取该参数，而是仅以 `N>2` 选择名为 `isDeterministic` 的模板；二者关系固定仓不可见，不得推测。
- Host 中部分 shape 乘法和 uint32 窄化未见独立溢出 guard；stage2 DMA 参数存在 uint16 转换，固定仓无额外 fallback。
- unknown runtime key 和 POD mode4 都无 body；Host selector 不产生它们。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/group_norm_grad/docs/aclnnGroupNormBackward.md` | `aclnnGroupNormBackward*` | 公开平台、签名、dtype、公式、错误语义。 |
| 入口实现 | `norm/group_norm_grad/op_host/op_api/aclnn_group_norm_backward.cpp` | `CheckParams`、`aclnnGroupNormBackwardGetWorkspaceSize` | API 前处理、平台分支、executor。 |
| Host launcher | `norm/group_norm_grad/op_host/op_api/group_norm_grad.cpp` | `l0op::GroupNormGrad` | 单 AICore launcher 节点。 |
| Host/Tiling | `norm/group_norm_grad/op_host/group_norm_grad_tiling.cpp` | `TilingGroupNormGrad`、`GroupNormGradTiling::*` | selector、workspace、blockDim。 |
| POD | `norm/group_norm_grad/op_host/group_norm_grad_tiling.h` | `GroupNormGradTilingData` | ABI 字段。 |
| Kernel entry | `norm/group_norm_grad/op_kernel/group_norm_grad.cpp` | `group_norm_grad` | runtime key→模板。 |
| Kernel body | `norm/group_norm_grad/op_kernel/group_norm_grad.h` | `GroupNormGrad<T,det>` | 五 route、UB、workspace、同步。 |
| 构建接线 | `norm/group_norm_grad/op_host/config/ascend910b/group_norm_grad_binary.json` | `GroupNormGrad_*` | A2 dtype binaries。 |
| 构建接线 | `norm/group_norm_grad/op_host/config/ascend910_93/group_norm_grad_binary.json` | `GroupNormGrad_*` | A3 dtype binaries。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 26 个 uint64 字段与顺序 | 宏 ABI 偏移不可猜；inactive 字段仍保留 | `GroupNormGradTilingData` |
| 调用顺序 | init→SyncAll→stage1→Reset/SyncAll→stage2 | stage2 不得早于所有 partial 写完 | `GroupNormGrad::Process` |
| workspace/内存 | 两片 FP32 slice | `W`、offset、dtype、owner 不得改变 | `CalStage2TilingInfo` |
| 同步 | queue events、Atomic、`SyncAll` | Atomic 模式必须及时恢复 `SetAtomicNone` | store helpers |
| 精度与尾块 | FP32 中间；16 位末端 cast；mode3补偿 | mask/q/tail 与 cast 顺序不可改变 | `Compute_Mode_3`、stage2 helpers |
| 失败边界 | q=0、shape/dtype/UB 检查失败 | 不得把 unknown key 当 fallback | Tiling/router |