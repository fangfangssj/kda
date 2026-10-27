---
type: Operator
title: "UpsampleLinear1dBackward 950 ACLNN 算子知识"
description: "说明 aclnnUpsampleLinear1dBackwardGetWorkspaceSize/aclnnUpsampleLinear1dBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bilinear2d_grad/op_host/op_api/aclnn_upsample_linear_1d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bilinear2d_grad/op_host/op_api/aclnn_upsample_linear_1d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/common/inc/op_api/aclnn_check.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_api/resize_linear_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_host/resize_linear_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_determine.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_no_determine.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_out1.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_out1_half.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_host/config/ascend950/resize_linear_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T20:53:30Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UpsampleLinear1dBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnUpsampleLinear1dBackwardGetWorkspaceSize` / `aclnnUpsampleLinear1dBackward`；固定仓仅此公开版本 |
| 计算语义 | 线性插值反向：每个 `gradOut[n,c,l]` 按正向插值权重贡献到相邻的 `out[n,c,left/right]`；同长时退化为复制 |
| 输入输出 | `gradOut:[N,C,L_des]`、`out:[N,C,L_src]`，ND/NCL；FP16/FP32/BF16；`outputSize=[L_des]`，`inputSize=[N,C,L_src]` |
| 执行类型 | R-001 为 ACLNN 框架 `Contiguous→ViewCopy`；R-002～R-007 为一个 `SIMT` Device 节点。entry 的 `KERNEL_TYPE_MIX_AIV_1_0` 仅配置 AIV，不含 Cube/AIC 或 MIX_CV |
| Kernel 路线 | R-001 同长复制；R-002 角对齐单点；R-003 半像素单点；R-004 `L_src=1` 归约；R-005 确定性 gather；R-006 非确定性单点 atomic；R-007 非确定性线性 atomic |
| 核心资源 | GM 直读写、SIMT 标量/寄存器、`InitOutput` 与 `SyncAll`；无显式 SIMT dynamic UB/shared buffer/TQue/TBuf；异长 Kernel 固定声明 16 MiB workspace 但 body 不访问 |
| 来源闭包 | `source_closed`：公开 ACLNN、950 `IsRegBase` 分支、L0、ascend950 OpDef/build/binary、arch35 Tiling/POD/key、全部可达 SIMT entry/body/helper 闭合；Contiguous/ViewCopy/executor 为框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `L_des=gradOut.shape[2]=outputSize[0]`，`L_src=out.shape[2]=inputSize[2]`。Host/Tiling 得到从目标位置映回源位置的 `scaleL`：非角对齐时，若用户 `scales>0` 则 `scaleL=1/scales`，否则为 `L_src/L_des`；角对齐时 `L_des>1` 为 `(L_src-1)/(L_des-1)`。对目标位置 `l`：

```text
orig(l) = max(0, (l+0.5)*scaleL-0.5)  # alignCorners=false
orig(l) = l*scaleL                     # alignCorners=true
left = min(floor(orig), L_src-1)
right = left + (left < L_src-1)
delta = orig-left
out[n,c,left]  += gradOut[n,c,l]*(1-delta)
out[n,c,right] += gradOut[n,c,l]*delta
```

确定性 Route 反向遍历：每个线程独占一个 `out[n,c,L]`，枚举所有可能贡献的 `gradOut` 并以 FP32 累加，最后一次写出。非确定性 Route 每线程独占 `gradOut` 元素并对一个或两个输出位置执行目标 dtype 的 `asc_atomic_add`。`L_src=L_des` 时公开入口不启动 `ResizeLinearGrad`，直接复制。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| gradOut | `[N,C,L_des]`，ND/NCL；允许非连续 | FP16/FP32/BF16 | rank 3，非空；shape 必须等于 `[inputSize[0],inputSize[1],outputSize[0]]` |
| outputSize | `aclIntArray`，长度 1 | INT64 | `outputSize[0]=L_des>0` |
| inputSize | `aclIntArray`，长度 3 | INT64 | `[N,C,L_src]`，各维与 `out` 一致且正；N/C 与 `gradOut` 一致 |
| alignCorners | 标量 | bool | false 为 half-pixel 映射，true 为角点中心对齐 |
| scales | 标量 | double；送入 Kernel 前窄化为 float | 文档要求 `<=500` 且 `outputSize=floor(inputSize_L*scales)`；950 API 分支未调用非 RegBase 的 `CheckScales/ComputeCheckScale`，固定源码没有额外 950 值域检查 |
| out | `[N,C,L_src]`，NCL；允许非连续 | 与 gradOut 相同 | rank 3、非空；是梯度输出 |
| workspaceSize/executor/stream | 两段式句柄 | uint64/指针 | 第一段构造执行图和 workspace；第二段在指定 stream 调用 `CommonOpExecutorRun` |

文档另约束各维 `<=2^20`、`N*C<2^31`、总内存小于 60 GiB；这些约束未全部在 950 API/Tiling 源码逐项重验。

### 入口与版本

固定 Commit 仅公开无版本后缀的两段式 ACLNN。公开文档明确支持 Ascend 950PR/950DT；`ResizeLinearGrad` README、`OpDef::AICore().AddConfig("ascend950")`、`SUPPORT_COMPUTE_UNIT "ascend950"`、arch35 Tiling 与 ascend950 三 dtype binary 共同闭合 950。

950 由 `IsRegBase(curArch)` 进入专用路径；当 `L_des!=L_src` 时调用 `ResizeLinearGrad`。同一 ACLNN 的 A2/A3 路线可达 `ResizeBilinearV2Grad5Hd`，固定仓只有 L0 wrapper、没有其 Host/Tiling/Kernel 实体，因此不在本卡外推。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段依次创建 executor，并由 `CheckParams` 检查非空指针、950 rank/shape、平台与 dtype、`gradOut=[N,C,L_des]`、N/C 一致和输出 shape。950 的 `CheckShapeValid` 要求 `L_src,L_des>0`，所以随后 `gradOut->IsEmpty()` 的零 launcher 分支在合法 950 输入上不可达。

`gradOut` 先经框架 `Contiguous`。若 `CheckIOSizesIsSame(gradOut,inputSize)` 为 true，直接 `ViewCopy` 到用户 `out`。否则 executor 分配一个仅提供 `[N,C,L_src]` shape/dtype/format 的 `originalImage`，L0 再按用户 `out` 属性分配中间 `y`，加入 `ResizeLinearGrad(grads,originalImage)->y` launcher，最后 `ViewCopy(y,out)`。第二段原样传播 `CommonOpExecutorRun` 返回值。

### 参数、Workspace 与生命周期

同长路线的 workspace 完全由 `Contiguous/ViewCopy` 框架节点汇总，公式在固定仓不可见。异长路线的 Tiling 无条件写 `workspace[0]=16*1024*1024` 字节；`resize_linear_grad` entry 接收该指针但从不解引用，算子自身没有 slice/offset，固定 16 MiB 的实际框架用途不可见。

`originalImage` 和中间 `y` 均由 executor 持有到第二段 stream 执行完成。`originalImage` 的 GM 指针传入 entry，但所有可达 body 都不绑定、不读取它，只以 Host shape 参与 Tiling。`blockDim=max(realCoreNum,initCoreNum)`；`SetScheduleMode(1)`，显式 event/多 stream 不存在。

### 启动图与失败传播

```text
aclnnUpsampleLinear1dBackwardGetWorkspaceSize
  -> CheckParams -> Contiguous(gradOut)
  -> L_des == L_src: ViewCopy(contiguous,out)                         # R-001，无自有 Kernel
  -> L_des != L_src: Alloc originalImage/y -> ADD ResizeLinearGrad   # R-002..R-007
       -> arch35 Tiling -> K0 resize_linear_grad<tuple>
       -> ViewCopy(y,out)
aclnnUpsampleLinear1dBackward
  -> CommonOpExecutorRun(workspace,workspaceSize,executor,stream)
```

K0 输入边为连续 `gradOut` GM 与 shape-only `originalImage`；输出为 executor 拥有的 `y` GM。K0 后的 `ViewCopy` 在同一 executor/stream 中按 launcher 顺序消费 `y`；固定仓没有显式 event。API 校验、tensor 分配或 launcher 添加任一步失败都阻断后续节点；Tiling 检查失败返回 `GRAPH_FAILED`。

## Route 与参数设计

### 分核与资源计算

记 `X=N*C*L_src`、`Y=N*C*L_des`、`Cplat=GetCoreNumAiv()`。公共均衡函数为：

```text
Cuse = min(S,Cplat)
tail = floor(S/Cuse)
front_count = S-tail*Cuse
front_chunk = tail + (S%Cuse != 0)
```

确定性 gather 以 `S=X` 分核，每线程生成一个或多个唯一输出；非确定性 scatter 以 `S=Y` 计算 `realCoreNum/blkProcessNum`，另以 `S=X` 计算清零所需 `initCoreNum/ubFactor`，然后 `blockDim=max(realCoreNum,initCoreNum)`。`isInt32=1` 当 `X,Y<=UINT32_MAX` 且两个 L 均 `<=INT32_MAX`，对应 1024 threads/block；否则用 uint64 地址和 512 threads/block。

同长 dormant DataCopy 才使用 `UB/2` 双缓冲；公开 950 在同长时已被 ACLNN 截获，因此该 Tiling key 对本逻辑算子不可达。所有可达 SIMT Route 都无显式 dynamic UB/shared allocation。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 同长框架复制 | 无 Resize key | 950 且 `gradOut.shape==inputSize`，结合已校验 N/C 即 `L_des==L_src` | 异长进入 Resize selector |
| 2 | R-002 角对齐单点 | `(3,isInt32,0,0)` | 异长且 `L_des==1 && alignCorners` | 下一判断 |
| 3 | R-003 半像素单点 | `(4,isInt32,0,0)` | `L_des==1 && !alignCorners && scaleL` 是正奇整数 | 下一判断 |
| 4 | R-004 单源归约 | `(2,isInt32,0,1)` | `L_src==1` | `ComputeOther` |
| 5 | R-005 确定性 gather | `(0或5,isInt32,isCenter,1)` | context deterministic，或 `scaleL<0.8`，或非角对齐的 `int64(L_des*scaleL)-L_src>=100`；sch5 表示点拷贝谓词但确定性 body 与 sch0 相同 | 无 fallback |
| 6 | R-006 非确定性单点 atomic | `(5,isInt32,isCenter,0)` | 未强制确定性，且 `scaleL` 为正奇整数（非角对齐）或 `alignCorners && scaleL` 为整数 | R-007 |
| 7 | R-007 非确定性线性 atomic | `(0,isInt32,isCenter,0)` | 未强制确定性且不满足 sch5 | 无 fallback |

```text
L_des == L_src -> R-001
else ResizeLinearGrad:
  L_des==1 && alignCorners -> R-002
  else L_des==1 && !alignCorners && integer(scaleL) && odd(scaleL) -> R-003
  else L_src==1 -> R-004
  else determine = context.det || scaleL<0.8 || repeatNum>=100
       point = (!alignCorners && odd integer scaleL) || (alignCorners && integer scaleL)
       determine -> R-005 (schId=point?5:0)
       !determine && point -> R-006
       !determine && !point -> R-007
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | Host 同长判断 | 950、合法 FP16/FP32/BF16、`L_src=L_des` | 框架节点，无自有 K0 | Contiguous/ViewCopy | 框架 workspace | 框架失败终止 |
| R-002 | sch3 | 950、异长、`L_des=1`、角对齐 | K0 SIMT | `ResizeLinearGradOut1<T,T2>` | GM 清零→`SyncAll`→直接写 | core/chunk tail guard |
| R-003 | sch4 | 950、异长、`L_des=1`、half-pixel、奇整数 `scaleL` | K0 SIMT | `ResizeLinearGradOut1Half<T,T2>` | 同上，写中心 floor 位置 | 同上 |
| R-004 | sch2 | 950、异长、`L_src=1` | K0 SIMT | `ResizeLinearGradDetermine<T,T2,T3,*,2>` | 每线程 FP32 reduce，无 atomic/清零 | 线程循环尾 |
| R-005 | sch0/5,det1 | 950、异长、确定性条件 | K0 SIMT | `ResizeLinearGradDetermine<T,T2,T3,isCenter,mode>` | 每输出 FP32 gather，一次写 | 候选区间与末点分支 |
| R-006 | sch5,det0 | 950、异长、非确定性、point | K0 SIMT | `ResizeLinearGradNoDetermine<T,T2,isCenter,5>` | GM 清零→全核同步→1 atomic/input | 地址 clamp |
| R-007 | sch0,det0 | 950、异长、非确定性、linear | K0 SIMT | `ResizeLinearGradNoDetermine<T,T2,isCenter,0>` | GM 清零→全核同步→2 atomic/input | right 边界合并 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 专用 | `IsRegBase(curArch)`；OpDef/build 仅 ascend950 | 选择 arch35 SIMT 实现 | 全部 | 非 RegBase 为另一闭包 | ACLNN GetWorkspaceSize；`resize_linear_grad_def.cpp` |
| 同长截获 | `CheckIOSizesIsSame` | 避免启动插值 Kernel | R-001 | 异长 Resize | ACLNN GetWorkspaceSize |
| 32/64 位地址 | `X,Y<=UINT32_MAX && L_src,L_des<=INT32_MAX` | 控制除法/address 类型和 1024/512 线程 | R-002～007 | 自动切 uint64 模板 | `LinearComputeGrad`; entry router |
| 确定性强制 | `context.det \|\| scaleL<0.8 \|\| repeatNum>=100` | 避免高冲突 atomic；按输出独占累加 | R-005 | R-006/007 | `ComputeOther` |
| 特化单点 | `L_des==1` 与对齐/奇整数谓词 | 只搬一个贡献，省去权重与 atomic | R-002/003 | 后续 Route | `ComputeKey` |
| 单源特化 | `L_src==1` | 所有输入贡献归约到唯一源点 | R-004 | `ComputeOther` | `ComputeLenSrcL1` |
| 输出初始化 | 非确定性及单点 direct Route | atomic/稀疏写前必须把未写位置清零 | R-002/003/006/007 | N/A | 各 `Process::InitOutput→SyncAll` |
| core tail | `blockIdx` 与 front-count 比较 | 处理 `S%Cuse` 的前长后短分块 | R-002～007 | 超额 block return/不计算 | 各 `Process` |
| 平台资源 | AIV核数、UB、blockSize 均 `>0` | 分核与 dormant DataCopy 参数需要 | R-002～007 | Tiling失败 | `TilingPrepare4ResizeLinearGrad` |

### POD/ABI

`ResizeLinearGradTilingData` 由 Tiling 宏生成；固定仓可见字段声明顺序、类型和 Host/Device 同名访问，但宏展开后的 `sizeof/alignof/offsetof` 不可见，故不推测 ABI offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| ResizeLinearGradTilingData | realCoreNum | int64/64 | 固定仓不可见 | core | `CalculateCoreNum(Y或X)` | compute core guard | `1..Cplat` |
| 同上 | initCoreNum | int64/64 | 固定仓不可见 | core | 非确定性/单点 `CalculateCoreNum(X)` | 清零 core guard | 未用路线保持0 |
| 同上 | blkProcessNum | int64/64 | 固定仓不可见 | 元素/core | `CalculateCoreNum` 的 front chunk | SIMT compute chunk | 正数 |
| 同上 | ubLoopSizeB | int64/64 | 固定仓不可见 | core count | `CalculateCoreNum` front_count | compute 前长核边界 | 名称不是 UB loop（SIMT Route） |
| 同上 | ubLoopSizeT | int64/64 | 固定仓不可见 | 元素/core | `floor(S/Cuse)` | compute 后短核长度 | 正数 |
| 同上 | ubFactor | int64/64 | 固定仓不可见 | 元素/core | `CalculateCoreNum(X)` front chunk | 清零 chunk | R-004/005 inactive |
| 同上 | ubFactorTailB | int64/64 | 固定仓不可见 | core count | `CalculateCoreNum(X)` front_count | 清零前长核边界 | 同上 |
| 同上 | ubFactorTailT | int64/64 | 固定仓不可见 | 元素/core | `floor(X/initCoreNum)` | 清零后短核长度 | 同上 |
| 同上 | lenSrcLOrUb | int64/64 | 固定仓不可见 | 元素 | `L_src`（仅 dormant 同长 DataCopy 改写为 UB bytes） | 地址、边界、除法 | 公开异长 Route 恒为 `L_src` |
| 同上 | lenDesL | int64/64 | 固定仓不可见 | 元素 | `L_des` | 地址、循环 | `>0` |
| 同上 | scaleL | float/32 | 固定仓不可见 | 源元素/目标元素 | `ComputeScale` 或角对齐公式 | 坐标/权重 | ACLNN double 先窄化 float attr |
| 同上 | inverseScaleL | float/32 | 固定仓不可见 | 目标元素/源元素 | `ComputeInverseScale` 或角对齐公式 | R-005 候选起点 | 其它 Route 可不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | API 同长 | ACLNN GetWorkspaceSize | 框架参数 | 无自有 entry | Contiguous/ViewCopy 边界 | ACLNN op_host CMake |
| R-002 | sch3 | `ComputeDesL1` | 公共 POD | `resize_linear_grad` sch3 | `resize_linear_grad_out1.h` | ascend950 OpDef/CMake/binary |
| R-003 | sch4 | `ComputeDesL1AndIntScale` | 公共 POD | entry sch4 | `resize_linear_grad_out1_half.h` | 同上 |
| R-004 | sch2 | `ComputeLenSrcL1` | 公共 POD | entry det1/sch2 | `resize_linear_grad_simt_determine.h::Mode2` | 同上 |
| R-005 | sch0/5,det1 | `ComputeOther` | 公共 POD | entry determine | determine + simt_base | 同上 |
| R-006 | sch5,det0 | `ComputeOther` | 公共 POD | entry no-determine | no_determine mode5 + simt_base | 同上 |
| R-007 | sch0,det0 | `ComputeOther` | 公共 POD | entry no-determine | no_determine mode0 + simt_base | 同上 |

## Kernel 实现

### 公共执行流程

异长路线均为单 K0：模板 entry `(grads,originalImage,y,workspace,tiling)` 解析 POD，声明 `KERNEL_TYPE_MIX_AIV_1_0`，再按 `schId/isInt32/isCenter/isDetermine` 编译期实例化 body。dtype 由 ascend950 binary 的 FP16/FP32/BF16 三实例提供。`originalImage` 与 `workspace` 在 body 中均不访问。

SIMT 路线用 `blockIdx` 得到连续 chunk，再以 `threadIdx.x` 为起点、`blockDim.x` 为步长覆盖 chunk。uint32 路线 `asc_vf_call(dim3(1024))`，uint64 路线为 512。非确定性和单点路线先由 `initCoreNum` 个 block 分片 `InitOutput(y,0)`，全体 `SyncAll` 后再读 grad/写 y；确定性路线每个输出只由一个线程写，无需 atomic 或全局清零。

### Route R-001：同长框架复制

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | image/upsample_bilinear2d_grad/op_host/op_api/aclnn_upsample_linear_1d_backward.cpp | `CheckIOSizesIsSame`; GetWorkspaceSize | 三维 shape 逐维等于 inputSize | `Contiguous→ViewCopy`，不调用 Resize |

#### Device 数据流

##### 函数级调用链

```text
aclnn...GetWorkspaceSize -> Contiguous(gradOut) -> ViewCopy(contiguous,out)
aclnn... -> CommonOpExecutorRun
```

##### 源码等价伪码

```text
if valid && L_des==L_src:
  tmp = framework_contiguous(gradOut)
  framework_view_copy(tmp,out)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut GM | 0 | `N*C*L` | API dtype/NCL | caller | Contiguous | caller | 非连续允许 |
| contiguous temp | 固定仓不可见 | 固定仓不可见 | 同 dtype | framework | ViewCopy | executor | framework 边界 |
| out GM | 0 | `N*C*L` | 同 dtype/NCL | ViewCopy | caller | caller | 固定仓不可见 |

##### 分层内存布局

N/A：此 Route 无算子自有 AIV/SIMT Kernel；Contiguous/ViewCopy 的内部 UB 布局在固定仓不可见。

##### 字节账与复用顺序

```text
UB_total = N/A（无自有 Kernel）
dynamic_ub = 固定仓不可见（框架节点）
workspace_total = executor 汇总，公式固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| contiguous temp | Contiguous | ViewCopy | executor 完成后 | 同 stream executor 顺序 | executor 生命周期覆盖第二段 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| copy | executor stream 顺序 | Contiguous | ViewCopy | temp | 无算术；tail 固定仓不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/upsample_bilinear2d_grad/op_host/op_api/aclnn_upsample_linear_1d_backward.cpp | GetWorkspaceSize | 同长分支只加入 ViewCopy |
| Kernel entry/router | 框架边界 | Contiguous/ViewCopy | 固定仓无算子自有 entry |
| Device body/helper | 框架边界 | N/A | 内部实现固定仓不可见 |

### Route R-002：角对齐单点直写

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | resize_linear_grad_tiling_arch35.cpp | `ComputeKey→ComputeDesL1` | `L_des==1 && alignCorners==1` | sch3；重置 center/det key 为0 |
| Router | resize_linear_grad_apt.cpp | sch3 branch | isInt32 选择 uint32/uint64 | `ResizeLinearGradOut1` |

#### Device 数据流

##### 函数级调用链

```text
ResizeLinearGrad launcher -> resize_linear_grad<3,...>
  -> ResizeLinearGradOut1::Init -> Process
  -> InitOutput(y,0) -> SyncAll -> SimtComputeMode3
```

##### 源码等价伪码

```text
parallel_zero(y[0:X]); SyncAll()
for nc in assigned_Y with SIMT stride:
  y[nc*L_src] = gradOut[nc]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut GM | `nc` | `Y=N*C` | T/NCL | caller | SIMT | caller | direct scalar load |
| y GM | `nc*L_src` | X | T/NCL | zero then SIMT | ViewCopy | executor | sparse unique write |

##### 分层内存布局

N/A：SIMT 直接访问 GM，无显式 dynamic UB/shared/TQue/TBuf。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = 16 MiB（body 不访问）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y 全域 | InitOutput | SIMT 稀疏直写/ViewCopy | 无 | SyncAll 后计算 | 所有清零 block 参与同步 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init→write | `SyncAll` | InitOutput | SIMT call | y 全域 | 原 dtype 搬运；chunk tail 由 block/thread 循环保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeDesL1` | sch3 与两组分核参数 |
| Kernel entry/router | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | sch3 | uint32/64 specialization |
| Device body/helper | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_out1.h | `Process/SimtComputeMode3` | 清零后只写每个 NC 的 L=0 |

### Route R-003：半像素单点直写

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | resize_linear_grad_tiling_arch35.cpp | `ComputeKey→ComputeDesL1AndIntScale` | `L_des==1 && !alignCorners && integer(scaleL) && odd(scaleL)` | sch4 |
| Router | resize_linear_grad_apt.cpp | sch4 | isInt32 | `ResizeLinearGradOut1Half` |

#### Device 数据流

##### 函数级调用链

```text
resize_linear_grad<4,...> -> ResizeLinearGradOut1Half::Process
  -> InitOutput -> SyncAll -> SimtComputeMode4
```

##### 源码等价伪码

```text
parallel_zero(y); SyncAll()
center = min(floor(scaleL*0.5-0.5),L_src-1)
for nc: y[nc*L_src+center] = gradOut[nc]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut/y GM | `nc` / `nc*L_src+center` | Y/X | T/NCL | caller/zero | SIMT/ViewCopy | caller/executor | direct scalar |

##### 分层内存布局

N/A：SIMT GM 直访，无显式 local/shared buffer。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=16 MiB（未访问）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y | InitOutput | sparse writer/ViewCopy | 无 | SyncAll | center 已 clamp |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init→write | SyncAll | zero | direct write | y | `scaleL` float；floor/clamp；线程 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeDesL1AndIntScale` | sch4 |
| Kernel entry/router | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | sch4 | Out1Half实例 |
| Device body/helper | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_out1_half.h | `SimtComputeMode4` | 中心位置直写 |

### Route R-004：单源长度归约

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | resize_linear_grad_tiling_arch35.cpp | `ComputeLenSrcL1` | `L_src==1` | sch2,det1 |
| Router | resize_linear_grad_apt.cpp | determine branch | sch2 | Determine mode2 |

#### Device 数据流

##### 函数级调用链

```text
resize_linear_grad<2,...,1> -> ResizeLinearGradDetermine::Process
  -> SimtComputeDetermineMode2
```

##### 源码等价伪码

```text
for nc in assigned N*C:
  acc_fp32=0
  for l in [0,L_des): acc += float(gradOut[nc*L_des+l])
  y[nc] = T(acc)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut/y GM | `nc*L_des+l` / `nc` | Y/X=N*C | T/NCL | caller | SIMT/ViewCopy | caller/executor | scalar coalescing取决于nc/l |
| acc | thread register | 1 float | FP32 | SIMT | cast/store | thread | 生命周期为一个nc |

##### 分层内存布局

N/A：仅 SIMT 寄存器标量与 GM。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=16 MiB（未访问）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| acc register | 当前 l loop | final cast | 下一nc | 单线程顺序 | 每nc重新置0 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| reduce | 单线程顺序 | grad loads | unique y store | 每nc | FP32累加后转T；无atomic；thread loop tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeLenSrcL1` | 以X分核，强制确定性 |
| Kernel entry/router | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | determine/sch2 | 32/64位实例 |
| Device body/helper | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_determine.h | `SimtComputeDetermineMode2` | 每NC顺序FP32归约 |

### Route R-005：确定性输出中心 gather

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | resize_linear_grad_tiling_arch35.cpp | `ComputeOther` | context det 或 `scaleL<0.8` 或 `repeatNum>=100` | det1，sch0/5 |
| Router | resize_linear_grad_apt.cpp | determine branch | mode非2 | `ResizeLinearGradDetermine` generic |

#### Device 数据流

##### 函数级调用链

```text
resize_linear_grad<0/5,T2,isCenter,1>
  -> ResizeLinearGradDetermine::Process
  -> SimtComputeDetermine -> optional ComputeLimt
```

##### 源码等价伪码

```text
for output y[nc,L] assigned to this thread:
  inStart = ceil inverse-map of L-1, clamp >=0
  acc_fp32=0
  while mapped grad position can contribute and inStart<L_des:
    weight = 1-abs(max(0,map(inStart))-L)
    acc += float(gradOut[nc,inStart])*weight
    inStart++
  last-source-L uses ComputeLimt to preserve right-edge merge
  y[nc,L]=T(acc)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut GM | `NC*L_des+inStart` | Y | T/NCL | caller | gather thread | caller | 候选区间读取 |
| y GM | `NC*L_src+L` | X | T/NCL | unique thread | ViewCopy | executor | 无冲突 |
| acc/lep | registers | float标量/2元素局部数组 | FP32 | thread | final store | thread | 每输出复用 |

##### 分层内存布局

N/A：无显式 dynamic UB/shared/TQue/TBuf，仅线程局部标量/小数组。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=16 MiB（未访问）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| acc register | gather loop | y store | 下一输出 | 单线程顺序 | 输出地址线程独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| gather | 无跨线程同步 | GM loads | unique store | y元素 | 输入转float、FP32固定顺序累加、转T；边界clamp与末点专用循环 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeOther` | 冲突启发式强制det1 |
| Kernel entry/router | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | determine branch | sch0/5共用generic body |
| Device body/helper | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_determine.h | `SimtComputeDetermine/ComputeLimt` | 每输出FP32 gather |

### Route R-006：非确定性单点 Atomic

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | resize_linear_grad_tiling_arch35.cpp | `ComputeOther` | 非det且point谓词 | sch5,det0 |
| Router | resize_linear_grad_apt.cpp | no-determine | mode5 | `ResizeLinearGradNoDetermine` |

#### Device 数据流

##### 函数级调用链

```text
resize_linear_grad<5,T2,isCenter,0> -> NoDetermine::Process
  -> InitOutput -> SyncAll -> SimtComputeNoDeter<mode5>
```

##### 源码等价伪码

```text
parallel_zero(y); SyncAll()
for each assigned grad index:
  (NC,L)=divmod(index,L_des); left=clamp(floor(map(L)))
  atomic_add(y[NC*L_src+left],gradOut[index])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut/y GM | yGmIdx / `NC*L_src+left` | Y/X | T/NCL | caller/zero | SIMT atomic/ViewCopy | caller/executor | 多线程冲突域为同NC/left |

##### 分层内存布局

N/A：GM atomic 与线程寄存器，无显式 local/shared buffer。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=16 MiB（未访问）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y | InitOutput | atomic scatter/ViewCopy | 无 | SyncAll；atomic序列化冲突 | 清零必须先全局可见 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scatter | SyncAll + `asc_atomic_add` | zero/grad load | y/ViewCopy | y冲突地址 | 原dtype atomic；累加顺序非确定；floor/clamp与thread tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeOther` | point+det0→sch5 |
| Kernel entry/router | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | no-determine | mode5实例 |
| Device body/helper | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_no_determine.h | `SimtComputeNoDeter<...,5>` | 单atomic贡献 |

### Route R-007：非确定性线性 Atomic

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | resize_linear_grad_tiling_arch35.cpp | `ComputeOther` | det0且非point | sch0,det0 |
| Router | resize_linear_grad_apt.cpp | no-determine | mode0 | `ResizeLinearGradNoDetermine` |

#### Device 数据流

##### 函数级调用链

```text
resize_linear_grad<0,T2,isCenter,0> -> NoDetermine::Process
  -> InitOutput -> SyncAll -> SimtComputeNoDeter<mode0>
```

##### 源码等价伪码

```text
parallel_zero(y); SyncAll()
for each grad g at (NC,L):
  orig=map(L); left=clamp(floor(orig)); right=left+(left<L_src-1)
  delta=orig-left; v=float(g); v2=v*delta; v1=v-v2
  atomic_add(y[NC*L_src+left],T(v1))
  atomic_add(y[NC*L_src+right],T(v2))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOut/y GM | yGmIdx / two neighbor addresses | Y/X | T/NCL | caller/zero | two atomics/ViewCopy | caller/executor | right可与left相同 |
| v/v1/v2 | thread registers | 3 float | FP32 | load/weight | cast+atomic | thread | 每input复用 |

##### 分层内存布局

N/A：GM atomic 与线程寄存器，无显式 local/shared buffer。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=16 MiB（未访问）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y | InitOutput | two atomics/ViewCopy | 无 | SyncAll + atomic | 清零先完成；right边界合并安全 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scatter | SyncAll + atomic | zero/grad | y | 相邻冲突地址 | 权重和值先FP32，贡献转T后atomic；顺序非确定；right clamp |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeOther` | normal+det0→sch0 |
| Kernel entry/router | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | no-determine | mode0实例 |
| Device body/helper | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_no_determine.h | `SimtComputeNoDeter<...,0>` | FP32拆权重后双atomic |

## 算子特定优化

### Trick T-001：按冲突强度切换 scatter 与 gather

#### 触发与源码依据

- 适用 Route：R-005、R-006、R-007
- 触发条件：用户确定性开关，或 `scaleL<0.8`，或 half-pixel `repeatNum>=100`
- Golden 来源：[`image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp:150-174`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp#L150-L174)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (scaleL_ < LOWER_DETERMINE || repeatNum >= HIGH_DETERMINE) {
    isDetermine_ = static_cast<uint64_t>(1);
}
if (isDetermine_ == static_cast<uint64_t>(1)) {
    CalculateCoreNum(xSize_, realCoreNum_, ubLoopSizeT_, ubLoopSizeB_, blkProcessNum_);
```

#### 机制与设计原因

低 `scaleL` 或高重复意味着多个 grad 元素更容易竞争相同输出。Host 把此域改为每线程独占输出的 gather，消除 atomic 和结果顺序不确定性；低冲突域保留按 grad 元素 scatter，避免每个输出反查贡献区间。`xSize`/`ySize` 决定分核域，key 的 `isDetermine` 决定真实 body。

#### 迁移边界

- 适用：同类插值反向或多源贡献累加。
- 前提：950 SIMT、可从比例估算冲突，并有 output-centered 等价实现。
- 失效：阈值或确定性语义变化时必须重新选择 gather/scatter，不能只改 key。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码明确的失败边界是忽略 `context->GetDeterministic()` 会违反确定性要求。

### Trick T-002：显式 FMA 固定坐标精度路径

#### 触发与源码依据

- 适用 Route：R-005、R-006、R-007
- 触发条件：SIMT 计算正向映射坐标
- Golden 来源：[`image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_base.h:25-36`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_base.h#L25-L36)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (isCenter == 0) {
    float origWidth = fmaf((static_cast<float>(L) + 0.5f), scaleL, -0.5f);
    float origWidthNew = fmaxf(static_cast<float>(0.0f), origWidth);
    return origWidthNew;
} else {
    float origWidth = fmaf(static_cast<float>(L), scaleL, 0.0f);
```

#### 机制与设计原因

源码注释说明显式 `fmaf` 用于避免编译器只把后续权重表达式融合而坐标表达式不融合，造成两条相关精度路径不一致。half-pixel 分支同时以 `fmaxf(0,orig)` 实现边界 clamp；`isCenter` 是 tiling-key 编译期参数，`scaleL` 来自 POD。

#### 迁移边界

- 适用：坐标与权重对同一乘加舍入一致性敏感的 SIMT 插值。
- 前提：FP32 坐标、编译器支持 `fmaf` 且公式语义允许融合舍入。
- 失效：改成 double、整数定点或不同坐标约定时需重证误差顺序。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码给出的真实错误边界是让编译器只自动融合权重而不融合坐标，可能导致特定 case 计算错误；固定仓未提供数值 case。

## 模板与实现对比

| 对比维度 | R-001 同长 | R-002/003 单点 | R-004/005 确定性 | R-006/007 非确定性 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | L相等最高优先 | L_des=1特化 | L_src=1或det强制 | det0后point/linear | Host决策树 |
| 模板与 dtype | 框架 | Out1/Out1Half，T+T2 | Determine，T+T2+T3 | NoDetermine，T+T2 | binary给T，key给整数/布尔 |
| 循环与分核 | 固定仓不可见 | 清零X、计算Y | 以X分核，逐输出 | 清零X、以Y分核 | 唯一写 vs atomic scatter |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 框架不可见 | 标量寄存器，无dynUB | FP32 acc，无dynUB | FP32权重，无dynUB | 可达SIMT均GM直访 |
| 内存总占用与复用 | executor汇总 | 固定16MiB未访问 | 同左 | 同左 | 无自有workspace slice |
| 搬运/Vector/SIMT/NDDMA/通信 API | framework copy | InitOutput+direct SIMT | gather/reduce SIMT | atomic SIMT | 无NDDMA/通信/Cube |
| 同步、精度与 tail | framework | SyncAll；原dtype | unique write；FP32累加 | SyncAll+atomic；贡献转T | 512/1024线程tail |
| fallback/失败边界 | 框架错误 | 无fallback | 无fallback | 无fallback | Tiling/API失败终止 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档支持，但本固定仓核心 fallback 不闭合 | 同左 | 支持且本卡闭合 |
| Host/Launcher | 非 RegBase，部分域调用仓内无 Kernel/Tiling 的 `ResizeBilinearV2Grad5Hd` | 同左 | `IsRegBase→ResizeLinearGrad` 或同长复制 |
| Kernel/资源 | 不在本卡外推 | 不在本卡外推 | arch35 SIMT；ascend950 binary；固定16MiB workspace |

## 限制与失败边界

- 950 API 支持 FP16/FP32/BF16 且要求输入输出 dtype 一致、rank 3、N/C/长度与数组一致；非法指针、shape、dtype 或平台返回参数错误。
- API 将 `double scales` 窄化为 `float` 传给 Resize；Tiling 的整数/奇数判断与坐标均基于该 float。950 源码未执行文档的 `scales<=500` 和 `floor(inputL*scales)==outputL` 检查。
- Tiling 的总元素乘法使用 shape size 的 int64 路径，但固定仓未见显式乘法溢出检查；32/64 位模板只在计算结果后选择。
- Tiling 固定申请 16 MiB workspace，而 Device body 不访问；该保留区的运行时用途固定仓不可见。
- `ResizeLinearGradTilingData` 宏展开后的 `sizeof/alignof/offsetof` 固定仓不可见；只可复用字段顺序、类型、producer/consumer，不应硬编码推测 offset。
- 同长的内部 sch1 AIV DataCopy 对公开 ACLNN 不可达，因为 API 已先改走框架 ViewCopy；本卡没有把 dormant key 写成正式 Route。
- A2/A3 可达的 `ResizeBilinearV2Grad5Hd` 在固定仓缺 Host/Tiling/Kernel，因此无后缀卡不满足 `source_closed`。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | image/upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md | 产品表/函数原型/约束 | 公开性、950支持、接口语义 |
| 入口声明/约束 | image/upsample_bilinear2d_grad/op_host/op_api/aclnn_upsample_linear_1d_backward.h | 两段式声明 | ABI入口 |
| 入口实现 | image/upsample_bilinear2d_grad/op_host/op_api/aclnn_upsample_linear_1d_backward.cpp | `CheckParams`; GetWorkspaceSize | 950分支、同长/异长图 |
| 平台选择 | common/inc/op_api/aclnn_check.h | `IsRegBase` | `DAV_3510` 映射到 950 RegBase 分支 |
| L0 | image/resize_linear_grad/op_api/resize_linear_grad.cpp | `ResizeLinearGradAICORE` | launcher与中间y |
| 平台说明 | image/resize_linear_grad/README.md | 产品支持表 | Resize节点仅950公开支持 |
| Host注册 | image/resize_linear_grad/op_host/resize_linear_grad_def.cpp | `ResizeLinearGrad` OpDef | dtype、attrs、ascend950 |
| Host/Tiling | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.h | POD/CompileInfo | 字段类型与顺序 |
| Host/Tiling | image/resize_linear_grad/op_host/arch35/resize_linear_grad_tiling_arch35.cpp | `ComputeKey/SetTilingData` | route、分核、key/workspace |
| Kernel entry/body | image/resize_linear_grad/op_kernel/resize_linear_grad_apt.cpp | `resize_linear_grad` | tuple router与执行族 |
| Kernel entry/body | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_tiling_key.h | template key macros | key域/可编译实例 |
| Kernel entry/body | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_base.h | `ComputeOriL/ComputeOutL` | 坐标公式/FMA |
| Kernel entry/body | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_determine.h | Determine bodies | gather/reduce |
| Kernel entry/body | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_simt_no_determine.h | NoDetermine body | atomic scatter |
| Kernel entry/body | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_out1.h | Out1 | 角对齐单点 |
| Kernel entry/body | image/resize_linear_grad/op_kernel/arch35/resize_linear_grad_out1_half.h | Out1Half | half-pixel单点 |
| 构建接线 | image/resize_linear_grad/op_host/config/ascend950/resize_linear_grad_binary.json | 三个 binary | FP16/FP32/BF16实例 |
| 构建接线 | image/resize_linear_grad/CMakeLists.txt | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950/arch35接线 |

每行文件均存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 12字段 producer/consumer 与 tuple key | 字段顺序/位宽；不得臆测offset；double→float窄化 | TilingData；`SetTilingData` |
| 调用顺序 | Contiguous→可选Resize→ViewCopy | 同长必须绕过Resize；异长K0先于ViewCopy | ACLNN GetWorkspaceSize |
| workspace/内存 | 异长固定16MiB；GM直访 | body不依赖workspace内容；originalImage仅shape | `SetTilingData`; entry/helpers |
| 同步 | 非det/单点清零→SyncAll→计算 | atomic前不得省略全核清零可见性 | NoDetermine/Out1 Process |
| 精度与尾块 | det用FP32累加；non-det贡献转T后atomic；FMA坐标 | 512/1024线程与T2匹配；边界clamp/right merge | simt_base/det/no_det |
| 失败边界 | API/Tiling任一检查失败终止 | 不得把A2/A3缺失核心或dormant sch1当闭合 | CheckParams/Tiling4ResizeLinearGrad |
