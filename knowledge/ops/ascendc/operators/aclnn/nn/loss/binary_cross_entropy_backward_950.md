---
type: Operator
title: "BinaryCrossEntropyBackward 950 ACLNN 算子知识"
description: "说明 aclnnBinaryCrossEntropyBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/docs/aclnnBinaryCrossEntropyBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_api/aclnn_binary_cross_entropy_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_api/binary_cross_entropy_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_host/arch35/binary_cross_entropy_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_host/binary_cross_entropy_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_kernel/binary_cross_entropy_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_host/config/ascend950/binary_cross_entropy_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_kernel/CMakeLists.txt
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
created_at: "2026-08-25T17:16:09Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# BinaryCrossEntropyBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnBinaryCrossEntropyBackwardGetWorkspaceSize` + `aclnnBinaryCrossEntropyBackward`，公开 ACLNN，两段式接口 |
| 计算语义 | `out = gradOutput * weight * (self-target) / max((1-self)*self, 1e-12)`；`reduction=mean` 再乘 `1/numel(self)`，`none/sum` 在反向公式中共用不缩放实现 |
| 输入输出 | `gradOutput/self/target/weightOptional -> out`；ND，rank 0–8；950 支持 FP16/BF16/FP32 promotion，`out.shape=self.shape` |
| 执行类型 | 所有 ACLNN 可达 Device 节点均为 `AIV_VECTOR`；Broadcast 搬入子路径可使用 NDDMA，但没有 Cube/AIC 或 MIX_CV |
| Kernel 路线 | 单 Kernel；R-001M/N（一维 201）、R-002M/N（NDDMA 1）、R-003M/N（NDDMA 2）、R-004M/N～R-008M/N（UB Broadcast 101/102/103/104/109） |
| 核心资源 | GM 输入/输出、单个 `TBuf<VECCALC>` tensor pool、DAG 编译期 buffer ID、`TPipe`/position mutex；通用 Broadcast Tiling 固定登记 16 MiB workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 `BinaryCrossEntropyGrad`→950 Tiling/key→单 AIV Kernel→atvoss Broadcast scheduler/DAG/Vector wrapper→950 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=self`、`y=target`、`g=gradOutput`、`w=weightOptional`（为空时取全 1），`N=numel(self)`。Device DAG 按元素计算：

```text
d = max((1 - x) * x, 1e-12)
r = g * (x - y) / d
out = cast_U(r * w * (reduction == mean ? 1/N : 1))
```

`U` 是 promotion 后 Kernel dtype，FP16/BF16 输入先 `Cast<float, U, 0>`，全部中间算术在 FP32，最后 `Cast<U, float, 1>`（源码常量 `CAST_MODE_RINT=1`）。FP32 路线模板仍以 `float` 为中间类型。`sum` 与 `none` 对输入梯度都不需要额外缩放，因此 Tiling 把两者都编码为 `REDUCTION_SUM=1` 并进入同一 `BCEGSumHasWeight` DAG。源码实际分母有 `1e-12` 下界；这比公开公式多一个稳定性保护。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | ND，rank 0–8，可 broadcast 到 `self` | FP16/BF16/FP32 | 上游梯度；`self` 为标量时不能传非标量 `gradOutput` 的限制由 broadcast 校验落实 |
| `self` | ND，rank 0–8 | FP16/BF16/FP32 | 预测值；公开原型约定值域 `[0,1]`，API 实现未做逐值检查 |
| `target` | 与 `self` storage format 相同，可 broadcast 到 `self` | FP16/BF16/FP32 | 标签；公开原型约定 0/1，API 实现未做逐值检查 |
| `weightOptional` | 可空；非空时 format 与 `self` 相同且可 broadcast | FP16/BF16/FP32 | 为空时 API 创建 `OnesLike(selfContiguous)`，不是把空指针交给底层 Kernel |
| `reduction` | 标量属性 | INT64 | 仅 0=`none`、1=`mean`、2=`sum`；API 转为字符串 |
| `out` | shape 与 `self` 相同，可非连续 | FP16/BF16/FP32 | promotion 结果须可 cast 到该 dtype，末尾经 `ReFormat` + `ViewCopy` 写回 |

### 入口与版本

固定版本只有公开符号 `aclnnBinaryCrossEntropyBackward`，无更高版本后缀。公开文档在 950 产品表中声明支持；`BinaryCrossEntropyGrad` OpDef 的唯一 `AICore().AddConfig`、模块 `SUPPORT_COMPUTE_UNIT`、Kernel `COMPUTE_UNITS` 和 binary JSON 又同时锁定 `ascend950`，所以 950 发布证据一致。Grad 是独立数学语义，不与前向 `BinaryCrossEntropy` 合并。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段接口依次执行：创建 executor；检查 `gradOutput/self/target/out` 非空；所有 tensor rank≤8；按 950 dtype 支持表校验 FP16/FP32/BF16；校验 `self` 与 `target/weight` storage format；校验 `out.shape==self.shape` 以及其余输入可广播到 `self`；校验 reduction∈[0,2]；做链式 promotion 并检查可 cast 到输出。`self.IsEmpty()` 时直接返回 workspace 0，不添加 Device 节点。

非空路径依次加入 `Contiguous(gradOutput,self,target)`；空 weight 用 `OnesLike(selfContiguous)`，否则 `Contiguous(weight)`；四路均 Cast 到 promotion dtype；调用 L0 `BinaryCrossEntropyGrad`；结果 Cast 到 `out.dtype`、`ReFormat(out.viewFormat)`、`ViewCopy`。这些连续化、Cast、OnesLike、ViewCopy 是 executor/framework 节点边界；算子核心节点的 Host/Tiling/Kernel 在固定仓中闭合。

### 参数、Workspace 与生命周期

L0 按 `self` dtype/ND 分配临时输出，`INFER_SHAPE` 复用 elementwise 输出 shape，然后以输入顺序 `(self,target,gradOutput,weight)` 和字符串属性 `reduction` 添加一个 AICore 节点。Broadcast Tiling 从平台取得 `coreNumAiv` 与 `ubSize`，在 `context->GetWorkspaceSizes(1)[0]` 固定登记 `16*1024*1024` 字节。核心 Kernel entry 接收 workspace 参数但不读取；这 16 MiB 属于框架 tiling workspace 约定，不是 BCEG 节点间中间结果。executor 拥有全部临时 tensor 与 workspace 规划，第二段在传入 stream 上用 `CommonOpExecutorRun` 顺序提交，任一第一段节点创建失败都立即返回且不形成可执行 executor。

### 启动图与失败传播

```text
ACLNN framework preprocessing (Contiguous / OnesLike / Cast)
  -> L0 BinaryCrossEntropyGrad::ADD_TO_LAUNCHER_LIST_AICORE
    -> [K0] binary_cross_entropy_grad<schMode, Reduction, HasWeight=1>
       inputs: self, target, gradOutput, materialized weight
       output: promotion-dtype temporary
       tiling: BroadcastOneDimTilingData or BroadcastBaseTilingData<OpDag>
       blockDim: Broadcast Tiling produced blockNum
  -> framework Cast / ReFormat / ViewCopy to user out
```

K0 是唯一算子核心 Device 节点，没有跨 Kernel edge。同一 executor/stream 保证前处理→K0→后处理的顺序和临时 tensor lifetime；K0 launcher 添加失败返回 `nullptr`，ACLNN 第一段转换为 `ACLNN_ERR_INNER_NULLPTR`，后续 Cast/ViewCopy 不再加入。

## Route 与参数设计

### 分核与资源计算

先将标量 shape 规范为 `{1}`，再按四路输入相对输出的 broadcast flag 合并相邻等价轴；broadcast 轴 stride 写 0。合轴 rank=1 走 201：Tiling 用 `B_tile=OpDag::GetBufferNum<true,true>()` 得到 `ubFormer=floor((ubSize/B_tile)/128)*128/4`，再算 `ubOuter=ceil(dimLen/ubFormer)`、`blockFormer=ceil(ubOuter/coreNum)`、`blockNum=ceil(ubOuter/blockFormer)`，最后核使用 `blockTail/ubTail`。Device 实际 pool slot 数是 `B_201=OpDag::BufferNum=GetBufferNum<true,false>()`；两者不能混写。`MaxDtypeBytes=4`。

合轴 rank>1 时，`maxElemNum=floor(((ubSize*8)/(B*32))/alignFactor)*alignFactor`，`alignFactor=256*8/minDtypeBits`，`B` 按 NDDMA 或 UB Broadcast 选择编译期 DAG buffer 数。自尾轴向前累乘决定 `ubSplitAxis/ubFormer/ubTail`；外层 tile 总数 `fusedProduct=ubOuter*prod(outputDims[0:ubSplitAxis])`，再生成 `blockFormer/blockTail/blockNum`。`runningRank=shapeLen-ubSplitAxis` 决定 1/2 或 101–104/109。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001M/N 一维 | `(201,2\|1,1)` | 合轴 rank=1；M 为 mean，N 为 none/sum | rank>1 进入 2 |
| 2 | R-002M/N NDDMA≤5D | `(1,2\|1,1)` | rank>1，`IsUbBroadcast=false`，`runningRank<=5` | runningRank>5 进入 R-003；UB=true 进入 4 |
| 3 | R-003M/N NDDMA循环 | `(2,2\|1,1)` | rank>1，`IsUbBroadcast=false`，`runningRank>5` | 无；Tiling 失败即 `GRAPH_FAILED` |
| 4 | R-004M/N UB rank1 | `(101,2\|1,1)` | rank>1，`IsUbBroadcast=true`，`runningRank=1` | 按 rank 进入后续 |
| 5 | R-005M/N UB rank2 | `(102,2\|1,1)` | 同上，`runningRank=2` | 按 rank 进入后续 |
| 6 | R-006M/N UB rank3 | `(103,2\|1,1)` | 同上，`runningRank=3` | 按 rank 进入后续 |
| 7 | R-007M/N UB rank4 | `(104,2\|1,1)` | 同上，`runningRank=4` | 按 rank 进入后续 |
| 8 | R-008M/N UB动态 rank | `(109,2\|1,1)` | 同上，`runningRank>4` | 无 |

```text
hasWeight = (optional input shape != nullptr) = 1 for ACLNN, because null weight was materialized
reductionKey = (reduction == "mean") ? 2 : 1
collapse four input shapes against output
if collapsedRank == 1: schMode = 201
else:
  anyBroadcast = exists input/axis: inputStride==0 && outputStride!=0
  bigNLast = CopyInBrc input has non-last broadcast, nonbroadcast last axis,
             and lastAxisBytes >= nddmaDcacheSize/2
  ub = anyBroadcast && (bigNLast ||
       (dtype in {FP16,BF16} && lastDim % (32/sizeof(dtype)) == 0))
  if !ub: schMode = (runningRank > 5 ? 2 : 1)
  else:   schMode = (runningRank <= 4 ? 100+runningRank : 109)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001M | 201/2/1 | 950，mean，collapsed rank1 | 单 K0 | `BroadcastOneDimSch<BCEGMeanHasWeight<U,float>::OpDag>` | pool=`ubFormer*4*B_201`，Tiling上界使用 `B_tile`，ping/pong buffer IDs | last core `blockTail`，last tile `ubTail` |
| R-001N | 201/1/1 | 950，none/sum，collapsed rank1 | 单 K0 | `BroadcastOneDimSch<BCEGSumHasWeight<U,float>::OpDag>` | 同上，无 mean scalar Muls | 同上 |
| R-002M/N | 1/2\|1/1 | rank>1，UB=false，runningRank≤5 | 单 K0 | `BroadcastNddmaSch<Mean\|SumDag,false>` | NDDMA CopyInBrc；pool=`elemNum*4*B_N` | `ubTail`；失败无 fallback |
| R-003M/N | 2/2\|1/1 | rank>1，UB=false，runningRank>5 | 单 K0 | `BroadcastNddmaSch<Mean\|SumDag,true>` | NDDMA 五维参数外加高维循环 | `ubTail`；失败无 fallback |
| R-004M/N | 101/2\|1/1 | UB=true，runningRank1 | 单 K0 | `BroadcastUbSch<Mean\|SumDag,1>` | compact CopyIn + UB `Broadcast<T,1>`；pool=`elemNum*4*B_U` | shape-aware tail |
| R-005M/N | 102/2\|1/1 | UB=true，runningRank2 | 单 K0 | `BroadcastUbSch<Mean\|SumDag,2>` | `Broadcast<T,2>` | 同上 |
| R-006M/N | 103/2\|1/1 | UB=true，runningRank3 | 单 K0 | `BroadcastUbSch<Mean\|SumDag,3>` | `Broadcast<T,3>` | 同上 |
| R-007M/N | 104/2\|1/1 | UB=true，runningRank4 | 单 K0 | `BroadcastUbSch<Mean\|SumDag,4>` | `Broadcast<T,4>` | 同上 |
| R-008M/N | 109/2\|1/1 | UB=true，runningRank>4 | 单 K0 | `BroadcastUbSch<Mean\|SumDag,-1>` | runtime-rank `Broadcast` | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 dtype | x/y/grad/out 为 FP16/BF16/FP32且 out==x dtype | DAG以 U 输入、FP32中间计算 | 全部 | `GRAPH_FAILED` | `CalcInputDtype/CalcOutputDtype` |
| Host DAG dtype代理 | mean仅FP32 Tiling实例化 `<float,float>`；其余Tiling实例化 `<half,float>`，Kernel始终按 binary 的 `DTYPE_X`实例化 | 这些DAG拓扑与 `MaxDtypeBytes=4`相同，Tiling只消费资源/shape元信息 | 全部 | 不得据Host half模板把BF16/FP32 Kernel误写成half | `RunFp16BroadcastTiling/RunFp32BroadcastTiling`, entry |
| 合轴非空 | collapse 后 output dims 非空且每维>0 | 分核、stride 与 tail 不接受 0 | 全部 | `GRAPH_FAILED` | `BroadcastBaseTiling::DoTiling`, `DimensionCollapse` |
| UB 下界 | `coreNum>0, ubSize>0, maxElemNum>0` | pool 至少容纳一组 DAG buffers | 全部 | `GRAPH_FAILED` | `DoBrodcastTiling` |
| UB Broadcast dtype/alignment | BF16/FP16 且 `lastDim%(32/dtypeBytes)==0`，或 BigNLast | 对齐尾轴适合 UB Broadcast；大 NLast 避免 NDDMA dcache压力 | R-004～008 | NDDMA R-002/003 | `IsUbBroadcast/IsBigNLastBrc` |
| NDDMA 维数 | `runningRank<=5` | 单次 NDDMA参数最多 5D | R-002 | R-003 显式外循环 | `SetSchMode`, `BroadcastNddmaSch<...,WithLoop>` |
| 201 而非 202 | `InputSize+OutputSize=5>4` | advance scheduler只接收≤4 GM参数且无 Var | R-001 | 固定走201 | `BroadcastBaseTiling::DoTiling` |
| 公开 weight 可达性 | ACLNN null weight先 `OnesLike`，L0输入4始终非空 | selector观察到 optional shape，恒写 HasWeight=1 | 全部 | HasWeight=0模板 dormant | ACLNN GetWorkspaceSize + `DoOpTiling` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| Tiling key | `schMode` | uint/16 bit | key首字段 | 枚举 | `GetSchMode()`，201/1/2/101..104/109 | Kernel模板 `BroadcastSch<schMode,...>` | 301..305 对 ACLNN连续输入不可达 |
| Tiling key | `Reduction` | uint/2 bit | key第二字段 | 枚举 | mean→2，none/sum→1 | entry选择 Mean/else Sum DAG | key允许0但 producer不写0 |
| Tiling key | `HasWeight` | uint/1 bit | key第三字段 | bool | optional input shape存在→1 | entry选择 HasWeight DAG | ACLNN恒为1 |
| `BroadcastOneDimTilingData` | `scalarFlag` | int32/32 | offset 0 | bitset | `GetOneDimScalarFlag` | `CopyIn`判断每路 scalar并 Duplicate | 每输入一 bit |
| 同上 | `ubSplitAxis` | int32/32 | offset 4 | axis | 固定仓未赋值 | 仅日志读取 | inactive |
| 同上 | `ubFormer` | int32/32 | offset 8 | element | `GetOneDimUBFormer/DoOneDimFormerTiling` | pool容量、整 tile长度 | 128B对齐 |
| 同上 | `ubTail` | int32/32 | offset 12 | element | `DoOneDimFormerTiling` | last tile有效长度 | 整除时等于 ubFormer |
| 同上 | `blockNum` | int32/32 | offset 16 | AIV core | `DoOneDimFormerTiling` | last-core判断 | 同时写 `SetBlockDim` |
| 同上 | `blockFormer` | int64/64 | offset 24 | tile/core | `DoOneDimFormerTiling` | core GM base与循环数 | ceil分配 |
| 同上 | `blockTail` | int64/64 | offset 32 | tile | `DoOneDimFormerTiling` | last-core循环数 | 整除时等于 former |
| 同上 | `scalarData` | char[16] | offset 40，sizeof=56，alignof=8 | raw bytes | mean `SetScalar(meanCof)` | `SetScalar<0>`读 float；N路线不读 | only M有效 |
| `BroadcastBaseTilingData<OpDag>` | `scheMode` | int32/32 | struct首字段 | enum | 固定仓未赋值 | contiguous scheduler不读 | inactive |
| 同上 | `shapeLen` | int32/32 | 声明顺序 | axis count | `AdaptBroadcastBaseTilingData` | scheduler/runningRank | ≤8 |
| 同上 | `ubSplitAxis` | int32/32 | 声明顺序 | axis | 同上 | GM offset、tile shape | `[0,shapeLen)` |
| 同上 | `ubFormer` | int32/32 | 声明顺序 | element on split axis | `GetBlockSplitFactor` | full tile | >0 |
| 同上 | `ubTail` | int32/32 | 声明顺序 | element | 同上 | final tile | >0 |
| 同上 | `ubOuter` | int64/64 | 声明顺序 | tile count | 同上 | realIdx/tail判定 | ceil值 |
| 同上 | `blockFormer` | int64/64 | 声明顺序 | tile/core | `DoBrodcastTiling` | per-core循环 | ceil值 |
| 同上 | `blockTail` | int64/64 | 声明顺序 | tile | 同上 | last-core循环 | >0 |
| 同上 | `dimProductBeforeUbInner` | int64/64 | 声明顺序 | fused outer tile | 同上 | 初始 axesIndices | fusedProduct |
| 同上 | `elemNum` | int64/64 | 声明顺序 | elements/buffer | `BroadcastGetMaxElemNum` | `InitBuffer` slot容量 | repeat对齐 |
| 同上 | `blockNum` | int64/64 | 声明顺序 | AIV core | `DoBrodcastTiling` | last-core判断 | `SetBlockDim`同值 |
| 同上 | `outputDims[8]` | int64×8 | 声明顺序 | element/axis | collapse结果 | axes更新、Broadcast shape | shapeLen内有效 |
| 同上 | `outputStrides[8]` | int64×8 | 声明顺序 | element stride | collapse结果 | GM offset/tileLength | shapeLen内有效 |
| 同上 | `inputDims[4][2]` | int64×8 | 声明顺序 | element | `SetInputParams` | 普通 CopyIn full/tail | CopyInBrc路不读 |
| 同上 | `inputBrcDims[4][8]` | int64×32 | 声明顺序 | element/axis | `SetInputNddmaParams` | NDDMA/UB Broadcast shape | 4路均为 CopyInBrc |
| 同上 | `inputVecBrcDims[0][8]` | 零长度实例 | 声明顺序 | N/A | 无 | 无 | DAG无 `Vec::Brc`节点 |
| 同上 | `inputStrides[4][8]` | int64×32 | 声明顺序 | element stride | `SetInputParams` | 普通 CopyIn | BCEG四路均 CopyInBrc，inactive |
| 同上 | `inputBrcStrides[4][8]` | int64×32 | 声明顺序 | element stride | `SetInputNddmaParams` | CopyInBrc offset/NDDMA | broadcast轴为0 |
| 同上 | `inputVecBrcStrides[0]` | 零长度实例 | 声明顺序 | N/A | 无 | 无 | inactive |
| 同上 | `scalarData[16]` | char×16 | 尾字段 | raw bytes | mean `SetScalar(meanCof)` | M DAG Var0；N不读 | compiled `sizeof/offsetof`未由仓内 static_assert锁定 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001M/N | 201/2\|1/1 | BCEG Tiling + `BroadcastBaseTiling::DoOneDimOpTiling` | `BroadcastOneDimTilingData` | `binary_cross_entropy_grad`→`BroadcastSch<201>` | Mean/Sum DAG + `BroadcastOneDimSch` + `vec.h` | ascend950 module/kernel/json |
| R-002M/N | 1/2\|1/1 | `IsUbBroadcast=false`, runningRank≤5 | `BroadcastBaseTilingData` | entry→`BroadcastSch<1>` | DAG + `BroadcastNddmaSch<...,false>` + broadcast utils | 同上 |
| R-003M/N | 2/2\|1/1 | `IsUbBroadcast=false`, runningRank>5 | 同上 | entry→`BroadcastSch<2>` | DAG + `BroadcastNddmaSch<...,true>` | 同上 |
| R-004～008M/N | 101..104/109,2\|1,1 | `IsUbBroadcast=true` + runningRank | 同上 | entry→对应 `BroadcastSch` | DAG + `BroadcastUbSch<R>` + AscendC Broadcast | 同上 |

## Kernel 实现

### 公共执行流程

entry 的 GM 参数顺序是 `x,y,grad_output,weight,output,workspace,tiling`；workspace 不读。编译期先按 `Reduction==BCE_MEAN`，再按 `HasWeight` 选择 DAG。ACLNN可达分支固定 HasWeight：M DAG依次为四路 `CopyInBrc`→Cast FP32→`x-y`、`1-x`、乘法、`Maxs(1e-12,denom)`、`grad*(x-y)`、Div、乘 weight、乘 meanCof、RINT cast→CopyOut；N DAG相同但没有 meanCof Muls。scheduler用 DAG `MemOptCfg<LEVEL_2>` 生成 buffer IDs，把一个 `TBuf<VECCALC>` 切为等长 slots；`GetTensor/ReleaseTensor` 在 MTE2、V、MTE3 之间建立生产消费与复用顺序。

### Route R-001M/R-001N：一维 201

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `DoTiling` | collapsed rank==1，且 `InputSize+OutputSize=5>4` | `DoOneDimOpTiling`, schMode201 |
| BCEG | ops-nn tiling | `RunFp16/32BroadcastTiling` | mean/else none或sum，hasWeight=1 | Reduction 2/1，Mean/SumHasWeight DAG |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<201>` | key schMode=201 | `BroadcastOneDimSch` |

#### Device 数据流

##### 函数级调用链

```text
binary_cross_entropy_grad<201,2|1,1>
  -> BroadcastSch<201>::Process
    -> BroadcastOneDimSch<Mean|SumDag>::Init
    -> per-core Process -> Run DAG
      -> CopyInBrc/CopyIn -> Cast/Sub/Subs/Mul/Maxs/Mul/Div/Mul[/Muls] -> Cast -> CopyOut
```

##### 源码等价伪码

```text
base = blockIdx * blockFormer * ubFormer
loops = lastCore ? blockTail : blockFormer
for tile in [0, loops):
  len = final tile of last core ? ubTail : ubFormer
  for each input: scalarFlag bit ? load GM[0] once and Duplicate : CopyIn(GM[base+tile*ubFormer], len)
  fp32 math = grad * (x-y) / max((1-x)*x, 1e-12) * weight
  if M: math *= meanCof
  CopyOut(out[base+tile*ubFormer], CastRint<U>(math), len)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | scalar:0；tensor:`base+tile*ubFormer` | `len` | U/ND | ACLNN前处理 | CopyIn | caller/executor | scalar缓存跨tile |
| tensor pool slot k | `k*ubFormer*4` bytes | `ubFormer*4` | byte view→U/FP32 | DAG node | dependent node | K0 `TPipe` | compile-time LEVEL_2 alias，ping/pong |
| output GM | `base+tile*ubFormer` | `len` | U/ND | CopyOut | framework后处理 | executor | DataCopyPad tail |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  pool0["tensorPool slot 0 · ubFormer×4B"]
  poolk["... DAG LEVEL_2 aliased slots ..."]
  poolN["slot B_201-1 · ubFormer×4B"]
```

##### 字节账与复用顺序

```text
UB_total = B_201 * ubFormer * MaxDtypeBytes = B_201 * ubFormer * 4
ubFormer = floor((ubSize / B_tile) / 128B) * 128B / 4
B_201 = OpDag::GetBufferNum<true,false>(); B_tile = OpDag::GetBufferNum<true,true>()
dynamic_ub = N/A
workspace_total(core node contract) = 16 MiB; Kernel未读该workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG slot | CopyIn/Vector node | 下一依赖 Vector/CopyOut | buffer ID下一节点或下一 ping/pong tile | `GetTensor/ReleaseTensor` position mutex | 只有最后消费者释放后可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→V→MTE3 | tensor position mutex | CopyIn/Vector | dependent node/CopyOut | pool slot | `len`限制无效lane；FP32中间；RINT回U |
| tile流水 | `i&1` buffer IDs | tile i | tile i+2覆盖 | compile-time ping/pong | final tile使用 ubTail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `DoOneDimOpTiling/DoOneDimFormerTiling` | 128B tile、分核与tail |
| Kernel entry/router | ops-nn `binary_cross_entropy_grad.cpp` + opbase `broadcast_sch.h` | entry/`BroadcastSch<201>` | key→Mean/Sum DAG→201 scheduler |
| Device body/helper | opbase `broadcast_sch_one_dim.h`, `vec.h` | `Process/CopyIn/CopyOut/RunNormalOp` | GM offset、scalar Duplicate、Vector链、tail |

### Route R-002M/R-002N：NDDMA 五维内

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `IsUbBroadcast/SetSchMode` | rank>1，UB=false，runningRank≤5 | schMode1 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<1>` | schMode1 | `BroadcastNddmaSch<Dag,false>` |

#### Device 数据流

##### 函数级调用链

```text
entry<1,2|1,1> -> BroadcastNddmaSch<Dag,false>::Init/Process
  -> CopyInBrc -> BroadcastNddmaWithoutLoop -> common FP32 DAG -> CopyOut
```

##### 源码等价伪码

```text
axes = decode(blockIdx*blockFormer)
for ubLoopIdx: lenAxis = tail ? ubTail : ubFormer
  NDDMA-copy/broadcast each source with <=5D outputDims/inputBrcStrides
  run common BCEG DAG for lenAxis*outputStride[ubSplitAxis]
  CopyOut at BroadcastGetGmOffset(axes, outputStrides,...)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| NDDMA source GM | `BroadcastGetGmOffset(axes,inputBrcStrides,...)` | POD 5D shape/stride决定 | U/ND | executor | NDDMA | caller | broadcast stride 0 |
| pool | `slot*elemNum*4` | `B_N*elemNum*4` | U/FP32 views | NDDMA/Vector | DAG | TPipe | LEVEL_2 alias |
| output GM | output stride offset | effective tile elements | U/ND | CopyOut | framework | executor | tail有效长度 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  nddma["NDDMA destination slots · CopyInBrc"]
  calc["LEVEL_2 aliased FP32 calc/output slots"]
```

##### 字节账与复用顺序

```text
UB_total = B_N * elemNum * 4 <= ubSize
dynamic_ub = N/A
workspace_total = 16 MiB contract; no K0 read
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| pool slot | NDDMA/Vector | downstream Vector/CopyOut | next DAG lifetime | position mutex | release after last consumer |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA→V | `GetTensor/ReleaseTensor<VECIN/VECCALC>` | NDDMA CopyIn | first Vector node | source slot | ≤5D；ubTail controls final extent |
| V→MTE3 | position mutex | RINT Cast | CopyOut | output slot | common FP32 order |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `SetSchMode` | runningRank≤5→1 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<1>` | false specialization |
| Device body/helper | opbase `broadcast_sch_nddma.h` | `CopyInBrc/Run` | ≤5D NDDMA、offset、tail |

### Route R-003M/R-003N：NDDMA 高维循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `SetSchMode` | UB=false且 runningRank>5 | schMode2 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<2>` | key2 | `BroadcastNddmaSch<Dag,true>` |

#### Device 数据流

##### 函数级调用链

```text
entry<2,...> -> BroadcastNddmaSch<Dag,true>::Process
  -> BroadcastNddmaWithLoop(outer high-rank loop -> <=5D DataCopy)
  -> common BCEG DAG -> CopyOut
```

##### 源码等价伪码

```text
for core tile:
  for collapsed dimensions outside the NDDMA 5D window:
    issue NDDMA for the inner <=5 dimensions using stride-0 broadcast axes
  compute BCEG on effective tile; write only ubFormer/ubTail valid region
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM→pool | outer-loop offset + 5D NDDMA params | effective tile | U/ND | NDDMA loop | FP32 DAG | K0 | stride0广播 |
| pool→GM | output stride offset | tail-valid elements | U/ND | CopyOut | executor | K0 | same pool accounting as R-002 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  nddma["looped NDDMA input slots"]
  calc["B_N-slot aliased FP32 DAG pool"]
```

##### 字节账与复用顺序

```text
UB_total = B_N * elemNum * 4 <= ubSize
workspace_total = 16 MiB contract
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| pool slot | each high-rank NDDMA slice | matching Vector slice | next slice | position mutex | slice consumed before overwrite |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| outer-loop NDDMA | loop ordering + mutex | GM load | Vector | inner slice | >5 axes explicitly looped；final ubTail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `SetSchMode` | >5→2 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<2>` | WithLoop=true |
| Device body/helper | opbase `broadcast_sch_nddma.h` | `BroadcastNddmaWithLoop`调用 | 高维拆外循环 |

### Route R-004M/R-004N：UB Broadcast rank 1

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `SetSchMode` | UB=true，runningRank=1 | 101 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<101>` | 101 | `BroadcastUbSch<Dag,1>` |

#### Device 数据流

##### 函数级调用链

```text
CopyIn compact source -> AscendC::Broadcast<U,1> -> common BCEG FP32 DAG -> CopyOut
```

##### 源码等价伪码

```text
for tile: copy compact broadcast inputs; form dst/src shapes from POD; Broadcast rank1;
run BCEG for ubSplitSize*outputStride[axis]; copy valid result
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| compact slot | input broadcast offset | `copyInBrcInputLength` | U | CopyIn | Broadcast | pool | separate src/dst IDs encoded together |
| expanded/calc slots | slot×elemNum×4 | `B_U*elemNum*4` total | U/FP32 | Broadcast/DAG | CopyOut | pool | LEVEL_2 alias |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  compact["compact CopyInBrc source slot"]
  expanded["expanded + FP32 DAG aliased slots"]
```

##### 字节账与复用顺序

```text
UB_total = B_U * elemNum * 4 <= ubSize
workspace_total = 16 MiB contract
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| compact/expanded pair | CopyIn/Broadcast | downstream DAG | next tile | VECIN/VECCALC mutex | both pair IDs released |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Broadcast | position mutex | compact copy | Broadcast rank1 | two pool IDs | tail shape rewrites ubFormer→ubTail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `IsUbBroadcast/SetSchMode` | UB谓词+rank1 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<101>` | R=1 |
| Device body/helper | opbase `broadcast_sch_ub_broadcast.h` | `CopyInAndVecBrc/CvecBrc` | compact load、Broadcast、tail shape |

### Route R-005M/R-005N：UB Broadcast rank 2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `SetSchMode` | UB=true，runningRank=2 | 102 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<102>` | 102 | `BroadcastUbSch<Dag,2>` |

#### Device 数据流

##### 函数级调用链

```text
compact CopyIn -> Broadcast<U,2> -> common FP32 BCEG DAG -> CopyOut
```

##### 源码等价伪码

```text
same as R-004, but GetBroadcastTilingInfo/Broadcast compile with const rank 2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB pool | slot×elemNum×4 | `B_U*elemNum*4` | U/FP32 | CopyIn/Broadcast/DAG | CopyOut | K0 | rank2 shape参数 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  compact["compact source"]
  expanded["rank2 expanded/DAG slots"]
```

##### 字节账与复用顺序

```text
UB_total = B_U * elemNum * 4 <= ubSize; workspace_total = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG slots | current node | dependent node | next lifetime | mutex | last consumer released |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| UB BRC | Broadcast rank2 + mutex | compact source | common DAG | expanded slot | tail shape uses ubTail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `SetSchMode` | 100+2 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<102>` | R=2 |
| Device body/helper | opbase `broadcast_sch_ub_broadcast.h` | `CvecBrc<T>` | const-rank2 Broadcast |

### Route R-006M/R-006N：UB Broadcast rank 3

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `SetSchMode` | UB=true，runningRank=3 | 103 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<103>` | 103 | `BroadcastUbSch<Dag,3>` |

#### Device 数据流

##### 函数级调用链

```text
compact CopyIn -> Broadcast<U,3> -> common FP32 BCEG DAG -> CopyOut
```

##### 源码等价伪码

```text
R-004 pipeline with compile-time running rank 3 and three-axis shape/stride consumption
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB pool | slot×elemNum×4 | `B_U*elemNum*4` | U/FP32 | scheduler/DAG | CopyOut | K0 | rank3 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  compact["compact source"]
  expanded["rank3 expanded/DAG slots"]
```

##### 字节账与复用顺序

```text
UB_total = B_U * elemNum * 4 <= ubSize; workspace_total = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| pool slots | current node | dependent node | next tile | mutex | tail shape updated first |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| UB BRC | Broadcast rank3 | CopyIn | DAG | expanded slot | common FP32/RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `SetSchMode` | 100+3 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<103>` | R=3 |
| Device body/helper | opbase `broadcast_sch_ub_broadcast.h` | `CvecBrc` | rank3 Broadcast |

### Route R-007M/R-007N：UB Broadcast rank 4

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `SetSchMode` | UB=true，runningRank=4 | 104 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<104>` | 104 | `BroadcastUbSch<Dag,4>` |

#### Device 数据流

##### 函数级调用链

```text
compact CopyIn -> Broadcast<U,4> -> common FP32 BCEG DAG -> CopyOut
```

##### 源码等价伪码

```text
R-004 pipeline with compile-time running rank 4
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB pool | slot×elemNum×4 | `B_U*elemNum*4` | U/FP32 | scheduler/DAG | CopyOut | K0 | rank4 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  compact["compact source"]
  expanded["rank4 expanded/DAG slots"]
```

##### 字节账与复用顺序

```text
UB_total = B_U * elemNum * 4 <= ubSize; workspace_total = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| pool slots | current node | dependent node | next tile | mutex | no live alias overwritten |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| UB BRC | Broadcast rank4 | CopyIn | DAG | expanded slot | ubTail shape |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `SetSchMode` | 100+4 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<104>` | R=4 |
| Device body/helper | opbase `broadcast_sch_ub_broadcast.h` | `CvecBrc` | rank4 Broadcast |

### Route R-008M/R-008N：UB Broadcast 动态 rank

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `broadcast_tiling.h` | `SetSchMode` | UB=true，runningRank>4 | 109 |
| Router | opbase `broadcast_sch.h` | `BroadcastSch<109>` | 109 | `BroadcastUbSch<Dag,-1>` |

#### Device 数据流

##### 函数级调用链

```text
compact CopyIn -> GetBroadcastTilingInfo(runtime runningRank) -> Broadcast(runtime rank)
  -> common FP32 BCEG DAG -> CopyOut
```

##### 源码等价伪码

```text
for tile: construct runtime-rank dst/src shapes from POD; Broadcast without const-rank specialization;
then execute the same BCEG DAG and valid-tail CopyOut
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB pool | slot×elemNum×4 | `B_U*elemNum*4` | U/FP32 | runtime Broadcast/DAG | CopyOut | K0 | rank stored in POD-derived state |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  compact["compact source"]
  expanded["dynamic-rank expanded/DAG slots"]
```

##### 字节账与复用顺序

```text
UB_total = B_U * elemNum * 4 <= ubSize; workspace_total = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| pool slots | runtime Broadcast/Vector | dependent DAG | next tile | mutex | runtime rank≤8，shapeLen有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| UB BRC | runtime Broadcast tiling + mutex | compact source | DAG | expanded slot | tail shape，FP32/RINT同公共链 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.h` | `SetSchMode` | >4→109 |
| Kernel entry/router | opbase `broadcast_sch.h` | `BroadcastSch<109>` | R=-1 |
| Device body/helper | opbase `broadcast_sch_ub_broadcast.h` | `CvecBrc` runtime分支 | runtime-rank Broadcast |

## 算子特定优化

### Trick T-001：API 物化默认权重，消除公开路径的无权重 Kernel 分支

#### 触发与源码依据

- 适用 Route：全部 ACLNN Route
- 触发条件：`weightOptional==nullptr`
- Golden 来源：[`loss/binary_cross_entropy_grad/op_api/aclnn_binary_cross_entropy_backward.cpp:240-264`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_api/aclnn_binary_cross_entropy_backward.cpp#L240-L264)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
const aclTensor *weightTensor;
if (!weightOptional) {
  weightTensor = l0op::OnesLike(selfContiguous, uniqueExecutor.get());
} else {
  weightTensor = l0op::Contiguous(weightOptional, uniqueExecutor.get());
}
CHECK_RET(weightTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

#### 机制与设计原因

公开入口把“无权重”转换为同 shape 全1 tensor，然后四路统一 Cast 并传给 L0。这样 Host optional-shape selector恒取 `hasWeight=1`，公开执行只复用 HasWeight DAG；代价是空 weight 路径多一个 OnesLike/weight搬运，但避免在核心公式中维护缺省值特例，且保持 broadcast/promotion统一。

#### 迁移边界

- 适用：可用乘1表示缺省权重、并由 executor管理临时 tensor 的逐元素梯度算子。
- 前提：OnesLike dtype/shape与 self 一致；下层 optional检测发生在物化之后。
- 失效：直接调用 L0 且传空 optional 时会命中 HasWeight=0 DAG，不能套用本 ACLNN Route结论。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要把 Kernel 中 HasWeight=0模板写成 ACLNN可达 Route；固定源码的 API 顺序证明其在该入口下 dormant。

### Trick T-002：FP32计算并对 BCE 分母做 `1e-12` 下界

#### 触发与源码依据

- 适用 Route：全部
- 触发条件：任意非空输入；FP16/BF16升到FP32，FP32保持FP32
- Golden 来源：[`loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_dag.h:43-55`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_dag.h#L43-L55)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using OpOneSubXMulX = Bind<Vec::Mul<T>, OpOneSubX, OpCopyXCast>;
using OpOneE = MAKE_CONST(T, 1e-12);
using OpOneSubXMulXMax = Bind<Vec::Maxs<T>, OpOneE, OpOneSubXMulX>;
using OpResGrad = Bind<Vec::Mul<T>, OpCopyGradOutputCast, OpXSubY>;
using OpXSubYDivOneSubXMulX = Bind<Vec::Div<T>, OpResGrad, OpOneSubXMulXMax>;

using OpResWeightCast = Bind<Vec::Mul<T>, OpXSubYDivOneSubXMulX, OpCopyWeightCast>;
using OpResCast = Bind<Vec::Muls<T>, OpResWeightCast, Placeholder::Var<T, 0>>;
using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;
```

#### 机制与设计原因

分母先以 FP32形成，再由 `Maxs` 限制最小正值，避免 `x` 接近0或1时直接除0；最终才RINT降精度。注意这是 `max(1e-12,d)` 而非 `max(abs(d),1e-12)`，所以必须保留公开值域前提，不能外推到任意 x。

#### 迁移边界

- 适用：概率域 BCE 梯度。
- 前提：`self` 应在 `[0,1]`；算子不做运行时逐值校验。
- 失效：若数学定义要求保留负分母或不同 epsilon，必须改 DAG。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把公开公式直接实现成低精度除法会丢失源码中的 FP32顺序与 epsilon 保护；这会改变数值语义。

### Trick T-003：按 dtype、尾轴对齐和 NLast 大小选择 NDDMA 或 UB Broadcast

#### 触发与源码依据

- 适用 Route：R-002～R-008M/N
- 触发条件：collapsed rank>1且存在 broadcast
- Golden 来源：[`pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h:483-521`](https://gitcode.com/cann/opbase/blob/d574f5571390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h#L483-L521)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (broadcastIdx == -1) { // 纯elewise场景走NDDMA
    return false;
}
if (IsBigNLastBrc()) { // NLast场景，尾轴大于4096B时走UBBRC
    return true;
}
auto inputDesc = context_->GetInputDesc(broadcastIdx);
OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
auto inputDtype = inputDesc->GetDataType();
int64_t dTypeSize = ge::GetSizeByDataType(inputDtype);
if (dTypeSize != 0 && lastDim % (BLOCK_LENGTH / dTypeSize) == 0 && ubBroadcastDtypes.count(inputDtype)) {
    return true;
}
return false;
```

#### 机制与设计原因

FP16/BF16尾轴满足32B对齐时，先紧凑搬入再在UB调用 Broadcast；FP32通常走NDDMA直接按stride0展开。大NLast输入即使不满足普通dtype分支也切UB Broadcast，避免其长连续尾轴压迫NDDMA dcache。runningRank再决定静态rank或动态rank specialization。

#### 迁移边界

- 适用：多输入广播DAG且有 CopyInBrc节点。
- 前提：使用同一 NDDMA dcache查询、32B block、rank≤8和相同pool字节模型。
- 失效：非连续输入会进入301–305另一族；ACLNN本路径已连续化，因此那些route不在本卡。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的反例是“所有输入纯elementwise时仍强制UB Broadcast”：selector明确回退 NDDMA；强制改变会偏离官方资源分流。

## 模板与实现对比

| 对比维度 | R-001M/N 201 | R-002/003M/N NDDMA | R-004～008M/N UB BRC | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | collapsed rank1 | rank>1，UB谓词false | rank>1，UB谓词true | rank先分流，再资源谓词 |
| 模板与 dtype | `OneDimSch<Mean\|SumDag>` | `NddmaSch<...,false\|true>` | `UbSch<...,1..4\|-1>` | U可FP16/BF16/FP32，T固定float |
| 循环与分核 | 线性tile/core | axesIndices + NDDMA，>5D外循环 | axesIndices + compact load + UB Broadcast | shape复杂度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单TBuf pool，B_201 slots；Tiling用B_tile | 单TBuf pool，B_N slots | 单TBuf pool，B_U slots且CopyInBrc需要src/dst pair | 无通信/dynamic UB |
| 内存总占用与复用 | `B_201*ubFormer*4` | `B_N*elemNum*4` | `B_U*elemNum*4` | DAG LEVEL_2 buffer IDs |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Duplicate | NDDMA + Vector | DataCopyPad + Broadcast + Vector | 无SIMT/通信 |
| 同步、精度与 tail | position mutex；ubTail | 同步同左；NDDMA tail | 同步同左；tail shape重写 | 全部FP32中间/RINT输出 |
| fallback/失败边界 | UB不足失败 | 1↔2仅按rank，不是失败fallback | 101..109按rank | selector失败均 `GRAPH_FAILED` |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档称支持，但固定源码无 Host/Tiling/Kernel接线，未达到 source_closed | 同左 | 支持且 source_closed |
| Host/Launcher | 固定仓只见共用ACLNN包装，底层实现缺失 | 同左 | `arch35` Tiling + atvoss Broadcast |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | ascend950 AIV单Kernel；NDDMA/UB Broadcast/201 scheduler |

## 限制与失败边界

- API rank≤8、reduction仅0/1/2；target/grad/weight必须广播到self，out shape必须等于self。空self不启动核心Kernel。
- 核心实现不做 x/y逐值校验；调用方应遵守概率/标签值域。分母保护是 `max(d,1e-12)`，不能解释为绝对值clamp。
- API把weight空值物化，所以公开路径无 HasWeight=0 Route；直接L0调用的可达性不属于本卡。
- Host Tiling 的模板 dtype 是资源代理：FP16/BF16共用 half DAG，且 none/sum 的 FP32 Host 分支也实例化 half DAG；实际 Kernel 输入类型由 binary `DTYPE_X`决定。复用时必须同时保持DAG拓扑和 `MaxDtypeBytes=4`，不能把Host代理类型当成Device dtype。
- `none` 与 `sum`共享同一反向Kernel公式；不要据此推断前向 reduction 的输出语义相同。
- Broadcast Tiling登记16 MiB workspace，但K0 entry不读取；executor内前后处理节点的细分workspace属于框架边界。
- A2/A3虽被公开支持表列出，但该固定commit的实现接线仅ascend950，缺少 A2/A3 Host/Tiling/POD/Kernel/build闭包，因此未生成无后缀卡。
- 固定仓未提供 `sizeof/offsetof` static_assert；除一维非模板POD可按声明推导的ABI外，不臆造模板POD最终编译偏移。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/binary_cross_entropy_grad/docs/aclnnBinaryCrossEntropyBackward.md` | ACLNN原型/产品表 | 950公开性、参数和错误语义 |
| 入口实现 | `loss/binary_cross_entropy_grad/op_api/aclnn_binary_cross_entropy_backward.cpp` | `GetWorkspaceSize/aclnnBinaryCrossEntropyBackward` | 校验、promotion、executor图 |
| L0 Launcher | `loss/binary_cross_entropy_grad/op_api/binary_cross_entropy_backward.cpp` | `l0op::BinaryCrossEntropyGrad` | 输入顺序、AICore节点 |
| Host/Tiling | `loss/binary_cross_entropy_grad/op_host/arch35/binary_cross_entropy_grad_tiling_arch35.cpp` | `BinaryCrossEntropyGradTiling::*` | reduction/weight/dtype与key |
| 平台注册 | `loss/binary_cross_entropy_grad/op_host/binary_cross_entropy_grad_def.cpp` | `BinaryCrossEntropyGrad` OpDef | ascend950配置 |
| Kernel entry/body | `loss/binary_cross_entropy_grad/op_kernel/binary_cross_entropy_grad.cpp` | `binary_cross_entropy_grad` | entry router |
| Kernel DAG | `loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_dag.h` | 四个BCEG DAG | FP32公式、epsilon、cast |
| Kernel key | `loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | key字段与值域 |
| Binary config | `loss/binary_cross_entropy_grad/op_host/config/ascend950/binary_cross_entropy_grad_binary.json` | `op_list` | 三dtype×三reduction |
| 构建接线 | `loss/binary_cross_entropy_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | 仅ascend950/arch35 |
| Kernel构建 | `loss/binary_cross_entropy_grad/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950、AIV Kernel编译选项 |
| POD/key公共依赖 | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | Broadcast POD/key macros | ABI字段、schMode集合 |
| Tiling selector | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling` | collapse、route、POD、workspace |
| Tiling公式 | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse/GetBlockSplitFactor/DoBrodcastTiling` | stride、tile/core/tail |
| UB公式 | `src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp` | `BroadcastGetMaxElemNum` | pool元素上限 |
| Scheduler router | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch` | 1/2/101..109/201映射 |
| 一维scheduler | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim.h` | `BroadcastOneDimSch` | 线性地址、pool、tail |
| NDDMA scheduler | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h` | `BroadcastNddmaSch` | 5D/高维循环搬入 |
| UB Broadcast scheduler | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h` | `BroadcastUbSch` | compact load、Broadcast、tail |
| DAG内存 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch/GetBufferNum/GetBufferIds` | LEVEL_2 buffer规划 |
| Vector wrapper | `pkg_inc/op_common/atvoss/util/vec.h` | `CopyInBrc/Cast/Sub/Mul/Maxs/Div/Muls/CopyOut` | AscendC primitive实参 |

每行文件均列于 Frontmatter `sources`；直接公共依赖使用独立固定 `opbase@d574f5571391390f8b619c6a134b65c819842318`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 201 POD与Base POD字段生产关系 | key三字段位宽、四路输入顺序、scalarData仅mean有效 | Tiling key + Broadcast POD |
| 调用顺序 | Contiguous/OnesLike/Cast→BCEG→Cast/ReFormat/ViewCopy | 空weight必须在L0前物化；L0顺序self,target,grad,weight | ACLNN + L0实现 |
| workspace/内存 | 16MiB contract；单TBuf DAG pool | pool按MaxDtypeBytes=4和B_N/B_U计算，不能按U字节少算 | Broadcast Tiling/schedulers |
| 同步 | DAG position mutex与ping/pong | last consumer释放前不得复用slot | `GetTensor/ReleaseTensor` |
| 精度与尾块 | FP32中间、epsilon、RINT、ubTail | 运算顺序与 `max(1e-12,d)` 不得改写 | BCEG DAG |
| 失败边界 | selector/UB/dtype失败均fail-close | 不存在自动跨route资源fallback；A2/A3实现不可外推 | Tiling/OpDef/CMake |
