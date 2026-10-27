---
type: Operator
title: "GroupNormSwishGrad A2/A3 ACLNN 算子知识"
description: "说明 aclnnGroupNormSwishGrad 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐阶段实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/docs/aclnnGroupNormSwishGrad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_host/group_norm_swish_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_host/group_norm_swish_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_host/group_norm_swish_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_kernel/group_norm_swish_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_kernel/group_norm_swish_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_host/config/ascend910b/group_norm_swish_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_host/config/ascend910_93/group_norm_swish_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T17:55:27Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# GroupNormSwishGrad A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnGroupNormSwishGradGetWorkspaceSize` + `aclnnGroupNormSwishGrad`，公开 ACLNN 两段式接口 |
| 计算语义 | GroupNorm+Swish 的反向：计算 `dx` 以及可选有效的 `dgamma/dbeta` |
| 输入输出 | ND；FP32/FP16/BF16；`dy/x:[N,C,...]`，`mean/rstd:[N,G]`，`gamma/beta:[C]` |
| 执行类型 | 唯一 Device 节点为 `AIV_VECTOR`；无 Cube/AIC/MIX_CV、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | R-001 整组驻 UB；R-002 单通道驻 UB；R-003 通道 HxW 切片 |
| 核心资源 | GM、AIV UB `TQue/TBuf`、Vector API、FP32 workspace、AtomicAdd、`SyncAll` |
| 来源闭包 | `source_closed`：公开 ACLNN 约束→生成桥边界→OpDef/Host Tiling/POD→router→legacy AIV body→A2/A3 binary/config 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `g=0..G-1` 的组，令 `xhat=(x-mean[n,g])*rstd[n,g]`、`y=xhat*gamma+beta`、`s=swishScale`、`den=exp(-s*y)+1`，则 `dSwish=(s*y-s*y/den+1)/den`、`dyNew=dy*dSwish`。通道梯度为 `dbeta[c]=sum(n,spatial) dyNew`、`dgamma[c]=sum(n,spatial) dyNew*xhat`；组内对 `gamma*dbeta` 和 `gamma*dgamma` 归约并除以 `C/G*HxW`，最后 `dx=rstd*(dyNew*gamma-c1-xhat*c2)`。`dgammaIsRequire/dbetaIsRequire` 只剪枝对应写回，不改变 `dx` 语义。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `dy`, `x` | ND，2–8D，相同 `[N,C,...]` | FP32/FP16/BF16，相同 | 非空；非连续由 `AutoContiguous`/生成 ACLNN 桥处理 |
| `mean`, `rstd` | ND `[N,G]` | 同 `dy` | 必选、非空，shape 必须相同 |
| `gamma`, `beta` | ND `[C]` | 同 `dy` | 必选、非空 |
| `numGroups` | 标量 | int64 API→Host int32 attr | `>0`，`C%G==0`，`C/G<=4000` |
| `dataFormatOptional` | 字符串 | char* | 默认/建议 `NCHW`；Device 按 ND 线性布局访问 |
| `swishScale` | 标量 | double API→float POD | 默认/建议 1.0 |
| `dgammaIsRequire`, `dbetaIsRequire` | 标量 | bool | 默认 true；输出指针在 ACLNN 文档中仍为必选 |
| `dxOut` | 同 `x` | 同 `x` | 必选 |
| `dgammaOut`, `dbetaOut` | 同 `gamma` | 同 `gamma` | 必选指针，require flag 控制有效写回 |

### 入口与版本

固定仓只公开该无版本后缀 API，未发现更新版本。文档明确支持 Atlas A2/A3 训练/推理；`GroupNormSwishGrad` OpDef 注册 `ascend910b` 和 `ascend910_93`，两份 binary config 均提供 FP32/FP16/BF16。ACLNN 生成的 inner bridge 源文件在固定仓不可见；其可见契约由 API 文档、`AutoContiguous` OpDef 和 `ACLNNTYPE aclnn` 构建接线锁定，算子拥有的 Tiling/POD/Kernel 均已继续追踪。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 第一段按文档检查九个 tensor 指针、dtype 一致性、shape 和 group 约束，并将非连续输入转为连续 ND。Tiling `CheckInputDtype` 再验证六输入 dtype 相同，`CheckInputShape` 验证 `dy==x`、`mean==rstd==[N,G]`、`gamma==beta==[C]`、`N*G>0`、`HxW>0`、`C%G==0`。注意 `TilingGroupNormSwishGrad` 源码调用 `Init()` 后未检查其返回值，而直接返回 `SetKernelTiling()`；不能将 Tiling 内部失败当成已被该 wrapper 完整传播。

### 参数、Workspace 与生命周期

`blockDim=coreNumUsed`，`taskNumPerCore=ceil(N*G,aivCores)/aivCores`，再以 `tailCore=N*G%coreNumUsed` 划分大/小核。系统 workspace 为 `GetLibApiWorkSpaceSize()`；用户区紧随其后，Device 用 `GetUserWorkspace` 定位。非确定性 FP32 用户区为 0；非确定性低精度为 `2*C*4` B；确定性为 `2*ceil(N/2)*C*4` B。两个 slice 依次是 `dgammaWorkspace[0:workSpaceSize]` 和 `dbetaWorkspace[0:workSpaceSize]`，owner 为 executor/调用者，有效期至第二段 stream 完成。

### 启动图与失败传播

```text
aclnnGroupNormSwishGradGetWorkspaceSize
  -> 生成 ACLNN executor/Contiguous 边界
  -> GroupNormSwishGrad OpDef
  -> TilingGroupNormSwishGrad: Init -> SetKernelTiling
aclnnGroupNormSwishGrad(workspace, executor, stream)
  -> group_norm_swish_grad<<<coreNumUsed, stream>>>(..., usrWorkspace, POD)
     -> stage-1 R-001/R-002/R-003
     -> SyncAll (stage-2 需要时)
     -> cast/reduce/reduce+cast stage-2
```

启动图只有一个 AIV Kernel 节点；stage-1→stage-2 是该 Kernel 内部跨核阶段，不是第二个 launch。constructor 中由 block 0 清零 GM/workspace，然后所有核 `SyncAll`；stage-1 写 partial 后再 `SyncAll` 保证 stage-2 可见。router 未知 compile key 无 fallback、不写输出。

## Route 与参数设计

### 分核与资源计算

Host 将 `dtypeBytes` 设为 FP32=4，FP16/BF16=2+4（同时预留原 dtype 与 FP32 cast buffer），`elePerBlock` 为 8/16。保留区 `reserve=1024+8*ceil(C/G,elePerBlock)*dtypeBytes`。`mode0Cap=(UB-reserve-unalignedExtra)/(8*ceil((C/G)*HxW,elePerBlock)*dtypeBytes)`；若不可载整组，`mode1Cap=(UB-reserve)/(8*ceil(HxW,elePerBlock)*dtypeBytes)`；最后 `mode2Capacity=floor((UB-reserve)/(8*dtypeBytes),64*elePerBlock)`，`iteration=ceil(HxW,capacity)/capacity`，tail 为末片元素数。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 整组驻 UB | POD `Tiling_key=0` | `mode0UbCapGNum>0` | R-002 |
| 2 | R-002 单通道驻 UB | POD `Tiling_key=1` | R-001 未命中且 `mode1UbCapCNum==1` | R-003 |
| 3 | R-003 通道 HxW 切片 | POD `Tiling_key=3` | 其它情况，且 `mode2UbCapacityEle>0` | Tiling 失败 |

```text
mode0Cap > 0 -> R-001
else if mode1Cap == 1 -> R-002
else if mode2Capacity > 0 -> R-003
else -> GRAPH_FAILED
compile key = dtypeBase(FP32=0,FP16=1,BF16=2) + 10*isDeterministic
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | POD 0；compile 0/1/2/10/11/12 | 整组 UB 容量式成立 | 单 AIV，stage-1+可选 stage-2 | `GroupNormSwishGrad<T,det>::CopyIn_Mode_0/Compute_Mode_0` | 整组 x/dy 一次载入；非对齐通道另行 GM recopy | HxW 用 `DataCopyPad`；stage-2 末 C tile |
| R-002 | 同上，POD 1 | 整组不可载、一通道可载 | 单 AIV | `Compute_Mode_1` | 通道两遍读 x/dy，中间保留 C/G 归约 | 通道天然 tail |
| R-003 | 同上，POD 3 | R-001/2 失败 | 单 AIV | `CopyIn_x/compute_dgamma/Compute_Mode_3` | HxW tile 两遍，tail=`mode2_ub_tail_num` | capacity 为 0 则 Host 失败 |

compile key 模板尾部：`!det,FP32` 直接 AtomicAdd 到 FP32 GM；`!det,lowp,N=1` 直接 cast 写 GM；`!det,lowp,N>1` 先 AtomicAdd 到长度 C 的两个 FP32 workspace，再按 C tile cast；`det,FP32` 写 `ceil(N/2)*C` partial 并两批 stage-2 reduce+AtomicAdd；`det,lowp` 写同样 workspace 后 reduce+cast。

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 整组 UB | `mode0Cap>0`，含 unalignedExtra | 8 份主 buffer，非对齐再留 4 份 FP32 HxW | R-001 | R-002 | `CalStage1TilingInfo` |
| 通道承载 | `mode1Cap==1` | mode-1 只为一通道容量实例化 | R-002 | R-003 | `CalStage1TilingInfo` |
| DMA blockCount | `coreBatchParts<=65535` | stage-2 `DataCopyPad.blockCount` 是有限字段 | 所有 det stage-2 | 缩小 `coreBatchParts` | `ComputeAllocUBStage2` |
| 确定性 | `N>2 && context.deterministic==1` | N 轴 partial 需固定 workspace 分组 | 模板 10/11/12 | key 0/1/2 | `Init` |
| 输出剪枝 | require flag | 避免对应 Atomic/cast/DataCopy | 全部 | 跳过该输出 | `Mode_Selection` 及 stage-2 helpers |

### POD/ABI

`GroupNormSwishGradTilingData` 由 `BEGIN/TILING_DATA_FIELD_DEF` 序列化宏定义；固定仓未包含宏实现，因而 `sizeof/alignof/offsetof` 不可见，但 Host/Device 共用同一定义且字段顺序可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `GroupNormSwishGradTilingData` | `Tiling_key` | uint64/64 | 顺序 0，精确 offset 不可见 | enum | `CalStage1TilingInfo` | `Process` stage-1 dispatch | 0/1/3 |
| 同上 | `N` | uint64/64 | 顺序 1 | 元素维 | `dyShape[0]` | stage-2/N 轴 | >0 |
| 同上 | `C` | uint64/64 | 顺序 2 | 元素维 | `dyShape[1]` | GM extent/stride | `C%G=0` |
| 同上 | `HXW` | uint64/64 | 顺序 3 | 元素 | `product(shape[2:])` | channel extent | >0 |
| 同上 | `G` | uint64/64 | 顺序 4 | 组 | attr `num_groups` | task→channel | >0 |
| 同上 | `NXG` | uint64/64 | 顺序 5 | task | `N*G` | mean/rstd extent | >0 |
| 同上 | `C_G` | uint64/64 | 顺序 6 | channel | `C/G` | group loops | 1..4000 |
| 同上 | `task_num_per_core` | uint64/64 | 顺序 7 | task/core | `PlanStepCoreUsage` | constructor | 大核 |
| 同上 | `task_num_per_tail_core` | uint64/64 | 顺序 8 | task/core | 同上 | constructor | 小核 |
| 同上 | `tail_core` | uint64/64 | 顺序 9 | core | 同上 | constructor | 大/小核分界 |
| 同上 | `mode1_ub_cap_C_num` | uint64/64 | 顺序 10 | channel | `CalStage1TilingInfo` | legacy body 不读 | inactive |
| 同上 | `mode1_ub_iter_C_num` | uint64/64 | 顺序 11 | iteration | 同上 | legacy body 不读 | inactive |
| 同上 | `mode1_ub_tail_C_num` | uint64/64 | 顺序 12 | channel | 同上 | legacy body 不读 | inactive |
| 同上 | `mode2_ub_capacity_ele` | uint64/64 | 顺序 13 | 元素/tile | `CalStage1TilingInfo` | mode-3 init/loop | 仅 R-003 |
| 同上 | `mode2_ub_iteration_num` | uint64/64 | 顺序 14 | tile/channel | 同上 | mode-3 loop | 仅 R-003 |
| 同上 | `mode2_ub_tail_num` | uint64/64 | 顺序 15 | 元素 | 同上 | mode-3 tail | 仅 R-003 |
| 同上 | `workSpaceSize` | uint64/64 | 顺序 16 | FP32 元素/slice | `CalStage2TilingInfo` | workspace binding | 0/C/ceil(N/2)*C |
| 同上 | `stage2CoreUsed` | uint64/64 | 顺序 17 | core | `ceil(C/castEleNum)` | stage-2 guard | stage-2 active |
| 同上 | `castEleNum` | uint64/64 | 顺序 18 | channel/core | `SetStage2CastInfo` | stage-2 UB/DMA | 64 对齐偏好 |
| 同上 | `tailCastNum` | uint64/64 | 顺序 19 | channel | 同上 | 末 stage-2 core | 1..castEleNum |
| 同上 | `coreBatchParts` | uint64/64 | 顺序 20 | workspace row | `ComputeAllocUBStage2` | stage-2 DMA/reduce loop | <=65535 |
| 同上 | `coreBatchPartsTailRepeat` | uint64/64 | 顺序 21 | row | 同上 | legacy body 不读 | inactive |
| 同上 | `repeatTime4Stage2` | uint64/64 | 顺序 22 | loop | 同上 | legacy body重算，不读 | inactive |
| 同上 | `dgamma_is_require` | uint64/64 | 顺序 23 | bool | attr | 写回剪枝 | 0/1 |
| 同上 | `dbeta_is_require` | uint64/64 | 顺序 24 | bool | attr | 写回剪枝 | 0/1 |
| 同上 | `swish_scale` | float/32 | 顺序 25 | 标量 | attr double→float | Swish Vector 管线 | 精度窄化 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | POD 0，compile 0/1/2/10/11/12 | `group_norm_swish_grad_tiling.cpp::CalStage1TilingInfo/SetTilingKeyMode` | `GroupNormSwishGradTilingData` | `group_norm_swish_grad.cpp::group_norm_swish_grad` | `group_norm_swish_grad.h::CopyIn_Mode_0/Compute_Mode_0` | OpDef 910B/910_93 + 两 binary JSON + `ascendc_config.json` |
| R-002 | POD 1，compile 同上 | 同上 | 同上 | 同上 | `Compute_Mode_1` | 同上 |
| R-003 | POD 3，compile 同上 | 同上 | 同上 | 同上 | `CopyIn_x/compute_dgamma/Compute_Mode_3` | 同上 |

## Kernel 实现

### 公共执行流程

router 以 compile key 0/1/2/10/11/12 实例化 `GroupNormSwishGrad<float|half|bfloat16_t,false|true>`。constructor 映射 GM，以 `blockIdx` 选择 task 区间，清零目标 GM/workspace，分配 stage-1 UB 并全核同步。三条 Route 都执行两遍数学管线：第一遍得到通道 `dbeta/dgamma` partial，`Mul_ReduceSum_template` 得到组内校正项，第二遍重算 `dyNew` 并写 `dx`。低精度输入先 `CAST_NONE`→FP32，所有计算/归约用 FP32，输出 `CAST_ROUND`。

### Route R-001：整组驻 UB

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_swish_grad_tiling.cpp` | `CalStage1TilingInfo` | `mode0UbCapGNum>0` | POD mode 0 |
| Router | `op_kernel/group_norm_swish_grad.cpp` | `group_norm_swish_grad` | compile key 0/1/2/10/11/12 | `GroupNormSwishGrad<T,det>` |

#### Device 数据流

##### 函数级调用链

```text
group_norm_swish_grad -> constructor -> Init_Output_GM -> Init_Unified_Buffer
  -> Process -> CopyIn_Mode_0 -> Get_Mean_Rstd -> Compute_Mode_0
  -> Mode_Selection -> Mul_ReduceSum_template -> dx CopyOut
  -> SyncAll -> optional stage-2 cast/reduce
```

##### 源码等价伪码

```text
for task in block-owned [n,g]:
  load whole dy,x group; x=(x-mean)*rstd; load gamma,beta
  for cInGroup: compute swish derivative; reduce dyNew and xhat*dyNew
  publish channel partial by mode; reduce gamma-weighted partial / groupElements
  for cInGroup: recompute dyNew; dx=rstd*(dyNew*gamma-c1-xhat*c2); tail-safe copy
barrier; execute compile-key stage-2 when required
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/dy/dx | `task*(C/G*HxW)+c*HxW` | `N*C*HxW` | T/ND | API/Kernel | stage-1/ caller | caller | 32 B pad tail |
| in x/dy queues | group base | `A=align(C/G*HxW,Tblock)` | FP32 | CopyIn | 两遍 compute | pipe/task | depth 1 |
| gamma/beta + grad queues | channel group base | `B=align(C/G,Tblock)` | FP32 | CopyIn/compute | compute/ModeSelection | pipe/task | depth 1 |
| scratch | 0 | `2*align(HxW,Tblock)+align(C/G*HxW,Tblock)` FP32 elements | FP32 | Vector | Vector/dx out | pipe/task | TBuf/queue 顺序复用 |
| workspace | 0 and `workSpaceSize` | 见 Host 公式 | FP32 row-major | stage-1 AtomicAdd | stage-2 | executor | `SyncAll` 后可读 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xy["x/dy queues · depth=1 · 2*A*4B"]:2
  gb["gamma/beta · 2*B*4B"]:1
  grad["dgamma/dbeta · 2*B*4B"]:1
  calc["dyNew+dswish · 2*H*4B"]:2
  temp["tempHxw · A*4B"]:2
```

##### 字节账与复用顺序

```text
FP32 UB_total = (3*A + 4*B + 2*H)*4 + conditional(2*H*4)
lowp UB_total = (5*A+4*B)*4 + (2*A+4*B)*sizeof(T) + conditional(2*H*4)
UB_total <= platform UB 由 Host mode0Cap 的 8-copy 保守式保证
dynamic_ub = N/A
workspace_total = 0 / 2*C*4 / 2*ceil(N/2)*C*4 B
stage2 UB = non-det-lowp:2*castEleNum*(4+sizeof(T)); det-FP32:2*castEleNum*4+2*coreBatchParts*castEleNum*4; det-lowp 再加 2*castEleNum*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `x_local` | CopyIn+normalize | 两遍 compute | FreeTensor 后下一 task | queue FreeTensor | 两遍完成前不释放 |
| `temp_hxw` | Swish/reduce/dx | ReduceSum/CopyOut | 下一 channel | PipeBarrier+MTE event | MTE3 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init | `SyncAll` | block0 zero | 所有 stage-1 | GM/workspace | FP32 zero |
| MTE2→V | event | DataCopyPad | Vector | UB input | count 保护 tail |
| V→MTE3 | event | Vector | DataCopyPad | dx/grad | lowp `CAST_ROUND` |
| stage boundary | `SyncAll` | all partials | stage-2 | workspace | det 固定分组；末 C tile=`tailCastNum` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_swish_grad_tiling.cpp` | `CalStage1TilingInfo` | mode0 容量和 guard |
| Kernel entry/router | `op_kernel/group_norm_swish_grad.cpp` | `group_norm_swish_grad` | compile key→T/det |
| Device body/helper | `op_kernel/group_norm_swish_grad.h` | `CopyIn_Mode_0/Compute_Mode_0` | 整组驻 UB、Atomic/workspace、tail |

### Route R-002：单通道驻 UB

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_swish_grad_tiling.cpp` | `CalStage1TilingInfo` | `mode0Cap<=0 && mode1Cap==1` | POD 1 |
| Device | `op_kernel/group_norm_swish_grad.h` | `Process` | `Tiling_key==MODE_1` | `Compute_Mode_1` |

#### Device 数据流

##### 函数级调用链

```text
Process -> Get_Mean_Rstd -> Compute_Mode_1
  -> per-channel CopyIn/normalize/Swish/reduce -> Mode_Selection
  -> group correction reduce -> per-channel recopy/recompute -> dx CopyOut -> common stage-2
```

##### 源码等价伪码

```text
for task: load mean/rstd,gamma,beta
  for c: load one HxW x/dy; normalize; accumulate dgamma/dbeta
  publish partial; reduce group correction
  for c: reload one HxW x/dy; recompute and write dx
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/dy channel | `task*groupEle+c*HxW` | `H=align(HxW,Tblock)` | FP32 UB | CopyIn | Vector | per channel | depth1，两遍 reload |
| gamma/beta/grad | group channel | `B=align(C/G,Tblock)` | FP32 | CopyIn/Vector | Vector/GM | per task | depth1 |
| calc/temp | 0 | `3*H*4` B | FP32 | Vector | Vector/MTE3 | per channel | TBuf/queue |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xy["x/dy · 2*H*4B"]:2
  calc["dyNew/dswish/temp · 3*H*4B"]:2
  params["gamma/beta/grads · 4*B*4B"]:4
```

##### 字节账与复用顺序

```text
FP32 UB_total=(5*H+4*B)*4; lowp UB_total=(5*H+4*B)*4+(2*H+4*B)*sizeof(T)
workspace_total=公共公式; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| channel queues | CopyIn | reduce/dx | FreeTensor 后下一 channel | FreeTensor+events | 每遍消费完 |
| grad queues | first pass | ModeSelection/correction | task end | FreeTensor | correction 完成前保留 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| channel load | MTE2_V | GM | Vector | HxW UB | DataCopyPad count |
| output | V_MTE3 | dx Vector | MTE3 | dx GM | FP32→T round |
| stage-2 | SyncAll | stage-1 | reduce/cast | workspace | 同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_swish_grad_tiling.cpp` | `CalStage1TilingInfo` | mode1 guard |
| Kernel entry/router | `op_kernel/group_norm_swish_grad.cpp` | `group_norm_swish_grad` | 六 compile keys |
| Device body/helper | `op_kernel/group_norm_swish_grad.h` | `Compute_Mode_1` | 通道两遍流程 |

### Route R-003：通道 HxW 切片

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/group_norm_swish_grad_tiling.cpp` | `CalStage1TilingInfo` | 前两项失败，`mode2Capacity>0` | POD 3 |
| Device | `op_kernel/group_norm_swish_grad.h` | `Process` | `Tiling_key==MODE_3` | `Compute_Mode_3` |

#### Device 数据流

##### 函数级调用链

```text
Process -> Get_Mean_Rstd -> Compute_Mode_3
  -> CopyIn_x -> compute_dgamma (all HxW tiles)
  -> Mode_Selection -> Mul_ReduceSum_template
  -> CopyIn_x/recompute/copy dx (all HxW tiles) -> common stage-2
```

##### 源码等价伪码

```text
for task:
  for c, tile: offset=task*groupEle+c*HxW+tile*capacity; count=capacity or tail
    load/normalize x, load dy, accumulate channel partial
  publish partial and form correction
  repeat same c,tile address network; compute/write dx[count]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/dy tile | `task*groupEle+c*HxW+t*capacity` | `K=align(capacity,Tblock)` | FP32 UB | CopyIn_x | compute | tile | tail count |
| channel partial | channel index | `B=align(C/G,Tblock)` | FP32 | tile reduction | correction | task | 跨 tile 累加 |
| dx tile | 同 GM offset | count | T ND | Vector | caller | output | V_MTE3 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xy["x/dy tile · 2*K*4B"]:2
  calc["dyNew/dswish/temp · 3*K*4B"]:2
  params["gamma/beta/grads · 4*B*4B"]:4
```

##### 字节账与复用顺序

```text
FP32 UB_total=(5*K+4*B)*4; lowp UB_total=(5*K+4*B)*4+(2*K+4*B)*sizeof(T)
K=align(mode2_ub_capacity_ele,Tblock)，Host 以 8-copy 公式保守限制
workspace_total=公共公式; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tile queues | CopyIn_x | compute_dgamma/dx | next tile | EnQue/DeQue/FreeTensor | consumer 完成 |
| partial scalar slots | each tile reduce | final correction | next task | PIPE_V barrier | 按 channel 累加 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile loop | queue + MTE2_V | DataCopy | Vector | tile UB | 末 tile=`mode2_ub_tail_num` |
| partial | PIPE_V | Vector reduction | next tile/final | scalar slots | FP32 accumulate |
| dx | V_MTE3 | Vector | GM | exact count | lowp round |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/group_norm_swish_grad_tiling.cpp` | `CalStage1TilingInfo` | capacity/iteration/tail |
| Kernel entry/router | `op_kernel/group_norm_swish_grad.cpp` | `group_norm_swish_grad` | compile key |
| Device body/helper | `op_kernel/group_norm_swish_grad.h` | `CopyIn_x/compute_dgamma/Compute_Mode_3` | tile 地址、两遍计算、tail |

## 算子特定优化

### Trick T-001：整组驻 UB 与非对齐通道纠正

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：`mode0UbCapGNum>0`；通道起点非 32 B 对齐时进入 GM recopy 分支。
- Golden 来源：[`group_norm_swish_grad.h:233-238`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_swish_grad/op_kernel/group_norm_swish_grad.h#L233-L238)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
} else {
    uint64_t gm_offset = static_cast<uint64_t>(task_idx) * this->ele_num_per_group + static_cast<uint64_t>(C_G_idx) * this->ele_num_per_channel;
    LocalTensor<float> unalign_channel_x = in_temp_x.Get<float>(this->ele_num_per_channel);
    LocalTensor<float> unalign_channel_dy = in_temp_dy.Get<float>(this->ele_num_per_channel);
    Custom_DataCopy_In(unalign_channel_x, in_tbuf_x_T, x_gm, gm_offset, this->ele_num_per_channel);
    Custom_DataCopy_In(unalign_channel_dy, in_tbuf_dy_T, dy_gm, gm_offset, this->ele_num_per_channel);
```

#### 机制与设计原因

整组驻 UB 让对齐通道在两遍计算间复用同一 x/dy，避免 mode-1/3 的整体 reload。但 legacy Vector API 不直接对任意通道 UB 起点读取，因此仅对非对齐通道从正确 GM 起点重拷贝到独立 HxW buffer，以正确性换取对齐通道的驻留复用。Host 将该额外 4 份 FP32 HxW 空间纳入 `unalignedExtraSpace`。

#### 迁移边界

- 适用：整组可载入 UB 且组内通道起点可能不对齐的归一化反向。
- 前提：必须保留 Host 的 unaligned extra 容量计算与 GM offset 公式。
- 失效：整组不可载时切换 R-002/R-003。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在非 32 B 对齐的通道起点上直接复用 aligned Vector 路径；固定源码以独立 GM recopy 分支作为可验证反例。

## 模板与实现对比

| 对比维度 | R-001 整组驻 UB | R-002 单通道驻 UB | R-003 通道切片 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | mode0Cap>0 | mode0=0, mode1Cap=1 | 其它 | UB 承载能力 |
| 模板与 dtype | 六 compile keys | 同 | 同 | dtype/determinism 正交于 stage mode |
| 循环与分核 | task→channel | task→channel，两遍 reload | task→channel→tile | task 映射 `[N,G]` |
| Queue/TBuf/寄存器/dynamic UB/通信 window | group buffers | HxW buffers | capacity buffers | AIV UB，dynamic UB/window N/A |
| 内存总占用与复用 | 最大，组常驻 | 中等 | 有界 | Host 8-copy 保守式 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+Vector | 同，两遍 GM | 同，tile | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | events/FP32/tail pad | 同 | 显式 tile tail | stage-2 SyncAll |
| fallback/失败边界 | R-002 | R-003 | capacity=0 失败 | unknown key 无输出 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持（ascend910b） | 支持（ascend910_93） | 支持，见 `_950` 卡 |
| Host/Launcher | 共用 Tiling；N>2 时由 context deterministic 决定 | 同 A2 | N>2 默认 deterministic |
| Kernel/资源 | legacy Vector body | 同 A2 | arch35 VF/MicroAPI，Kahan stage-2 |

## 限制与失败边界

- 文档约束 `N,C,HxW,G>0`、`C%G=0`、`C/G<=4000`；Host 形状乘法使用 uint64，未见额外溢出 guard。
- `TilingGroupNormSwishGrad` 忽略 `Init()` 返回值，是源码可见的失败传播边界。
- router 对未知 compile key 无 fallback；POD mode 非 0/1/3 时 stage-1 不写 `dx`。
- `swishScale` 从 API double 窄化为 float；FP16/BF16 上源输入升到 FP32 计算，最后 `CAST_ROUND`。
- 生成 ACLNN inner bridge 和宏定义 ABI 实现在固定仓不可见；未据此推测 offset。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/group_norm_swish_grad/docs/aclnnGroupNormSwishGrad.md` | ACLNN 两段原型 | API、shape/dtype/平台/确定性 |
| 入口实现 | `norm/group_norm_swish_grad/op_host/group_norm_swish_grad_def.cpp` | `GroupNormSwishGrad` OpDef | AutoContiguous、attrs、A2/A3 注册 |
| Host/Tiling/Launcher | `norm/group_norm_swish_grad/op_host/group_norm_swish_grad_tiling.cpp` | `GroupNormSwishGradTiling` | selector、workspace、blockDim、POD producer |
| POD | `norm/group_norm_swish_grad/op_host/group_norm_swish_grad_tiling.h` | `GroupNormSwishGradTilingData` | 26 字段契约 |
| Kernel entry | `norm/group_norm_swish_grad/op_kernel/group_norm_swish_grad.cpp` | `group_norm_swish_grad` | compile key router |
| Kernel body | `norm/group_norm_swish_grad/op_kernel/group_norm_swish_grad.h` | `GroupNormSwishGrad<T,det>` | 三 Route、UB、stage-2 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `GroupNormSwishGrad` | 910B/910_93/950 compile units |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Host/Device 共用 26 字段 | 顺序、uint64/float 位宽、元素/字节单位 | `GroupNormSwishGradTilingData` |
| 调用顺序 | init→stage1→barrier→stage2 | 清零和两个 `SyncAll` 不可省 | constructor/`Process` |
| workspace/内存 | 两个连续 FP32 slice | `dbeta` offset=`workSpaceSize`，owner 至 stream 结束 | constructor |
| 同步 | event+pipe+Atomic | Atomic 前后 flag 与 `SetAtomicNone` | `fp32_dgamma_dbeta2GM` |
| 精度与尾块 | lowp→FP32→lowp | `CAST_NONE/CAST_ROUND`，count/tailCastNum | copy helpers/stage-2 |
| 失败边界 | 三 mode 有序 fallback | mode2 capacity=0/unknown key 不得推测补齐 | `CalStage1TilingInfo`/router |