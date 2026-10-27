---
type: Operator
title: "LtTensor 950 ACLNN 算子知识"
description: "说明 aclnnLtTensor 在 950 上从公开 API、Host/Launcher、条件 Cast 到 AIV/NDDMA/UB Broadcast 或 AICPU Less 的完整实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/docs/aclnnLtTensor&aclnnInplaceLtTensor.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_api/aclnn_lt_tensor.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_api/less.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_host/less_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_host/arch35/less_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel/less_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel/arch35/less_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel_aicpu/less_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/cast_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel_aicpu/cast_aicpu.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/aicpu_common/context/utils/bcast.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/aicpu_common/context/utils/bcast.cc
    role: implementation
created_at: "2026-08-25T11:12:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# LtTensor 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnLtTensorGetWorkspaceSize` + `aclnnLtTensor`；只覆盖新建输出的 Tensor 入口，不含 Scalar 与 Inplace 变体。 |
| 计算语义 | 广播后逐元素比较：`boolOut[i] = cast_P(self[i]) < cast_P(other[i])`，再把 BOOL 结果转换到 `out.dtype`。BOOL 输入的比较域被 API 显式改为 UINT8。 |
| 输入输出 | `self/other` 为 ND、0–8 维、可非连续，支持 FLOAT/FLOAT16/BFLOAT16、各类有/无符号整数、DOUBLE、BOOL；`out` 另支持 COMPLEX64/128，shape 必须等于广播结果。 |
| 执行类型 | 核心 950 路线为 AIV_VECTOR，搬运可使用 NDDMA 或 UB Broadcast；UINT16/UINT32 promote 结果走 AICPU Less。条件 Cast 为 AIV_VECTOR、SIMT 辅助的 AIV Cast 或 AICPU。无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 一维 advance；R-002 多维 NDDMA；R-003 多维 UB Broadcast；R-004 AICPU Less。空 Tensor 在 Host 成功短路。 |
| 来源闭包 | `source_closed`：公开 API→校验/Promote/Contiguous/Cast→Less AICore/AICPU selector→950 Tiling/POD/router/body或 AICPU body→Cast/ViewCopy→构建接线闭合；Contiguous/ViewCopy/executor 聚合为明确框架边界。 |
| 验证状态 | 仅做固定源码闭包提取；未执行测试、validator、编译或 review。 |

## 接口与语义

### 计算语义

令 `B=broadcast_shape(self.shape,other.shape)`，950 使用 `BinaryOpTypePromote` 得到 `P`。非空输入先连续化并转换到相同的 `P`；若 `P=BOOL`，源码将实际核心 dtype 改为 UINT8。对 `i∈B`，按输入广播 stride 取 `x_i,y_i`，计算 `b_i=(x_i<y_i)`。核心输出是 BOOL 字节，随后条件 Cast 到用户给定的 `out.dtype` 并由 `ViewCopy` 写入可能非连续的输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，0–8 维，可非连续 | FLOAT、FLOAT16、BFLOAT16、INT8/16/32/64、UINT8/16/32/64、DOUBLE、BOOL | 必选；与 `other` 可 Promote 且可广播。 |
| `other` | 同 `self` | 同 `self` | 必选；广播轴可为1。 |
| `out` | ND，0–8 维，可非连续 | 上述类型及 COMPLEX64/COMPLEX128 | shape 精确等于广播 shape；BOOL 必须可转换到该 dtype。 |
| `workspaceSize` | Host 标量输出 | `uint64_t*` | executor 内实际节点、中间张量和对齐后的 Device workspace 总量。 |
| `executor` | Host 对象输出 | `aclOpExecutor**` | 保存节点图和中间张量，第二段执行前保持有效。 |

### 入口与版本

本卡只覆盖 `aclnnLtTensor`。固定仓公开文档列出 Ascend 950PR/950DT，`less/CMakeLists.txt` 将 `ascend950` 接到 `arch35`，OpDef 与 `less_binary.json` 也有对应 950 接线。相同目录中的 `aclnnInplaceLtTensor` 复用入口但输出生命周期不同，不并入。文档虽然还声明 A2/A3 支持，固定 Commit 的 `math/less/` 未提供可追溯到这些代际的 Host/Tiling/Kernel 实现，因此不能据此建立无后缀卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 依次检查三指针非空、输入/输出 dtype 白名单、950 `BinaryOpTypePromote` 有定义且 promote dtype 仍在输入白名单、BOOL 可转换到 `out.dtype`、两输入 rank≤8且可广播、`out.viewShape` 等于广播 shape。任一输入为空时写 `workspaceSize=0` 并释放 executor，不添加 Device 节点。

非空时严格顺序是 `Contiguous(self)`→`Cast(self,P)`→`Contiguous(other)`→`Cast(other,P)`→`Less`→`Cast(BOOL,out.dtype)`→`ViewCopy(out)`；`P=BOOL` 在第一个 Cast 前改成 UINT8。`l0op::Cast` 同 dtype 时直接返回输入；其他转换先匹配 950 AICore pair，无法匹配但两端在 AICPU 支持集时提交 AICPU Cast。`INT64→DOUBLE` 的 AIV Cast tiling 预留 32 KiB local memory 给 SIMT 辅助，仍属于 VV 范围。

`l0op::Less` 分配广播 shape 的 BOOL tensor。950 core dtype 属 `{INT64,UINT64,INT32,INT8,UINT8,FLOAT,FLOAT16,BFLOAT16,DOUBLE}` 时提交 AICore；`UINT16/UINT32` 不在 RegBase AICore 表而进入 AICPU。API 已把 BOOL 改成 UINT8，故该入口不会把 BOOL 直接送入 Less AICore。

### 参数、Workspace 与生命周期

Broadcast Tiling 从平台读取 AIV 核数与 UB 大小，并给核心节点写固定 `16*1024*1024` workspace；`less` entry 带 workspace 参数但不读取。Cast 节点有自己的 `MINIMAL_WORKSPACE` 或 AICPU task space。最终 `workspaceSize` 由 `executor->GetWorkspaceSize()` 汇总，固定源码未暴露聚合公式。executor 持有所有临时 tensor，直至第二段 `CommonOpExecutorRun` 在给定 stream 上完成。

### 启动图与失败传播

```text
empty(self) || empty(other) -> success(workspace=0), no launch
otherwise:
  Contiguous(self) -> Cast(self,P')
  Contiguous(other) -> Cast(other,P')
  P' in {U16,U32} ? Less(AICPU) : Less(AIV, schMode)
  -> Cast(BOOL,out.dtype) -> ViewCopy(out)
where P' = (BinaryOpTypePromote(...) == BOOL ? UINT8 : promoted dtype)
```

每次节点创建后都由 `CHECK_RET` 检查空指针；launcher 添加失败会立即终止后续建图。AIV Less 是单核心 Kernel launch；AICPU Less 是单 AICPU task，内部才可能按数据量 `ParallelFor`。前后 Cast 是独立节点，不与 Less 共用 UB、POD 或同步对象。

## Route 与参数设计

### 分核与资源计算

AIV 路线把 scalar 规范为 `{1}`，右对齐两输入 shape，并合并相邻且广播状态相同的轴；广播 stride 为0。合轴 rank=1 时使用 advance。令 `N1=OpDag::GetBufferNum<true,true>()`，`M=max(sizeof(P'),1)`，`m=1`（BOOL 输出和 uint8 中间节点），则 `tileNum=floor((UB/N1)/512)*512/M`，再根据 `coreNum*(512/m)` 与总元素数减少核数或 tile；`ubOuter=ceil(dimLen/tileNum)`，若 `ceil(ubOuter/coreNum)==1` 则每核改处理2个 tile，以开启 ping/pong。

多维令 `Nn=GetBufferNum<true,true>()`（NDDMA）、`Nu=GetBufferNum<false,true>()`（UB Broadcast）。通用上限为 `maxElem=floor((((UB-extra)*8)/(N*8M))/alignFactor)*alignFactor`，其中 `alignFactor=256*8/(8m)=256` 个元素。从末轴向前累计得 `ubSplitAxis/ubFormer/ubTail`；`fused=ubOuter*product(outputDims[0:ubSplitAxis])`，`blockFormer=ceil(fused/coreNum)`、`blockNum=ceil(fused/blockFormer)`、`blockTail=fused-(blockNum-1)*blockFormer`。使用符号 `N1/Nn/Nu` 是因为它们由 Less DAG 的编译期存活分析产生，不在算子源码写死。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-004 AICPU Less | AICPU task | `P'∈{UINT16,UINT32}` | 其他 950 core dtype→AIV 路线 |
| 2 | R-001 AIV 1D advance | `schMode=202` | AICore dtype，合轴 rank=1、输入连续、DAG参数≤4且无 Var | rank>1→R-002/R-003 |
| 3 | R-003 AIV UB Broadcast | `101/102/103/104/109` | rank>1且 `IsBigNLastBrc`，或窄 dtype 广播满足尾轴32B对齐等 UB 条件；running rank 1..4 / >4 | 谓词假→R-002 |
| 4 | R-002 AIV NDDMA | `1/2` | rank>1且 UB selector 为假；running rank≤5 / >5 | 无其他 fallback |

```text
if P' is UINT16 or UINT32: AICPU
else if collapsed_rank == 1: 202
else if IsUbBroadcast(): 100 + (running_rank <= 4 ? running_rank : 9)
else: running_rank <= 5 ? 1 : 2
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 202 | 950 AICore dtype、合轴一维 | 条件 Cast→单 `less`→条件 Cast | `BroadcastOneDimAdvanceSch<LessCompute<T>::OpDag>` | AIV local tensor pool，MTE2/V/MTE3 mutex，ping/pong | 最后一核/最后 tile 由 `dimLen` 推导 |
| R-002 | 1/2 | 950 AICore dtype、多维、UB selector=false | 同上 | `BroadcastNddmaSch<...,false/true>` | 五维 NDDMA MultiCopy；mode2循环高维前缀 | `ubTail`；无数值 fallback |
| R-003 | 101–104/109 | 950 AICore dtype、多维、UB selector=true | 同上 | `BroadcastUbSch<...,R=1..4/-1>` | compact CopyIn→向量 Broadcast；广播结果可跨 loop 缓存 | 109处理动态高 rank |
| R-004 | AICPU | `P'` 为 U16/U32 | 条件 Cast→单 AICPU Less→条件 Cast | `LessCpuKernel::LessCompute<T>` | Host CPU tensor地址；`ParallelFor` 分片 | 同shape/单元素快速体，否则通用 Bcast |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| core dtype一致 | x1/x2 dtype 相等 | Less DAG只有一个输入模板 `T` | R-001～R-003 | Tiling失败 | `LessTiling::DoOpTiling` |
| 512B tile可分 | `tileNum>0` | 每个存活子区都需可容纳对齐 tile | R-001 | Tiling失败 | `GetOneDimUBFormerAdvance` |
| UB容量 | `maxElem>0` | 全部编译期存活 buffer 不越 UB | R-002/R-003 | Tiling失败 | `BroadcastGetMaxElemNum` |
| NDDMA rank | running rank≤5选1，否则2 | MultiCopy参数一次描述5维 | R-002 | mode2，不是失败 | `SetSchMode` |
| UB rank | 1..4专门化，>4选109 | 静态 rank Broadcast 降低动态控制 | R-003 | 109 | `BroadcastSch` |
| AICPU分片 | 普通广播≥2K；同shape/单元素≥7K | 小任务避免线程调度开销 | R-004 | 串行循环 | `BcastCompute`,`NoBcastCompute` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LessCompileInfo` | `coreNum`,`ubSize` | uint64×2 | sizeof=16, alignof=8, offsets 0/8 | 核、字节 | `TilingPrepareForLess` | Broadcast Tiling | AIV核数、UB容量 |
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | int64/64 | sizeof=16, alignof=8, offset=0 | 元素 | 合轴shape | advance Init | >0 |
| 同上 | `tileNum` | int32/32 | offset=8 | 元素/tile | 512B容量公式 | blockLen/循环步长 | >0 |
| 同上 | `blockNum` | int16/16 | offset=12 | AIV核 | 分核公式 | block guard | 固定源码无显式窄化溢出检查 |
| 同上 | `scalarFlag` | int16/16 | offset=14 | bitset | 一维输入长度1 | scalar Duplicate | bit0/bit1对应两输入 |
| `BroadcastBaseTilingData<OpDag>` | `scheMode,shapeLen,ubSplitAxis,ubFormer,ubTail` | int32×5 | sizeof=632, alignof=8, offsets 0/4/8/12/16 | key/轴/元素 | Broadcast Tiling | scheduler | `scheMode`当前不读；padding至24 |
| 同上 | `ubOuter,blockFormer,blockTail,dimProductBeforeUbInner,elemNum,blockNum` | int64×6 | offsets 24/32/40/48/56/64 | tile、元素、核 | 容量/分核 | scheduler | 均应>0 |
| 同上 | `outputDims[8],outputStrides[8]` | int64数组 | offsets 72/136 | 元素/轴、元素stride | 合轴结果 | GM offset/MultiCopy | 仅前shapeLen活跃 |
| 同上 | `inputDims[2][2]` | int64[4] | offset=200 | 元素 | 普通 CopyIn | 本 DAG 两输入均 CopyInBrc，不读 | inactive |
| 同上 | `inputBrcDims[2][8]` | int64[16] | offset=232 | 元素/轴 | 广播合轴 | UB Broadcast | 两个输入 |
| 同上 | `inputStrides[2][8]` | int64[16] | offset=360 | 元素stride | 普通 CopyIn | 本 DAG不读 | inactive |
| 同上 | `inputBrcStrides[2][8]` | int64[16] | offset=488 | 元素stride | 广播合轴 | NDDMA/UB | 广播轴为0 |
| 同上 | `scalarData[16]` | char[16] | offset=616 | 原始字节 | 默认0 | Less DAG不读 | inactive |
| `CastTilingData` | 12个顺序字段 | int64×12 | sizeof=96, alignof=8，offset依次0..88 | 核/元素/loop | Cast Tiling | Cast policy body | 字段为 blockNum、ubFormer、blockFormer、两类ubLoop、两类ubTail、两reg step、三reg loop |
| AICPU Less | 无算子私有 Tiling POD | — | launcher传入 tensor/shape 元数据 | — | AICPU runtime | `LessCpuKernel` | 不使用上述Broadcast POD |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 202 | `LessTiling::DoOpTiling`→`DoOneDimOpTilingAdvance` | `BroadcastOneDimTilingDataAdvance` | `less<202>`→`BroadcastSch<202>` | `BroadcastOneDimAdvanceSch`; Less DAG；`Vec::Compare/Select` | `less/CMakeLists.txt` ascend950/arch35 |
| R-002 | 1/2 | `DoBroadcastOpTiling/SetSchMode` | `BroadcastBaseTilingData<OpDag>` | `less<1/2>` | `BroadcastNddmaSch`; `BroadcastNddmaWith/WithoutLoop` | 同上 |
| R-003 | 101–104/109 | `IsUbBroadcast/SetSchMode` | 同上 | `less<101..109>` | `BroadcastUbSch`; AscendC Broadcast；Less DAG | 同上 |
| R-004 | AICPU selector | `less.cpp::IsAiCoreSupport` | tensor shape/dtype metadata | `ADD_TO_LAUNCHER_LIST_AICPU(Less)` | `LessCpuKernel`; opbase `Bcast` | `add_all_modules_sources`含AICPU模块 |

## Kernel 实现

### 公共执行流程

AIV entry 固定为 `(x1,x2,y,workspace,tiling)`，workspace 不读；`schMode` 由 TilingKey 实例化。除 BOOL 特化外，entry 用 `LessCompute<DTYPE_X1>`；BOOL会改为 `int8_t`，但本 Tensor API 已预先改 UINT8。DAG 顺序是两路 `CopyInBrc<T>`→`Compare<uint8_t,T,mode0>`→分别 Duplicate `uint8_t(1)`/`uint8_t(0)`→`Select<uint8_t,uint8_t,mode2>`→`CopyOut<uint8_t>`。Compare 产生 mask，Select 把 mask显式物化成每元素1/0的 BOOL 字节；全程无近似数学。

### Route R-001：AIV 1D advance

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0 | `less.cpp` | `IsAiCoreSupport` | `P'`在950 RegBase表 | AICore Less |
| Tiling | `broadcast_tiling.h` | `DoTiling` | 合轴rank1、连续、参数≤4、无Var | mode202 |
| Router | `broadcast_sch.h` | `BroadcastSch::Process` | `schMode==202` | advance body |

#### Device 数据流

##### 函数级调用链

```text
less<T,202> -> BroadcastSch<202>::Process
  -> BroadcastOneDimAdvanceSch::Init/Process
  -> CopyInBrc(x1,x2) -> CompareLT -> Duplicate(1,0) -> Select -> CopyOut
```

##### 源码等价伪码

```text
base = blockFormer * tileNum * blockIdx
for i in this_core_loops:
  valid = min(tileNum, dimLen - (base+i*tileNum))
  a = scalarFlag.bit0 ? Duplicate(x1[0],valid) : CopyIn(x1[base+i*tileNum],valid)
  b = scalarFlag.bit1 ? Duplicate(x2[0],valid) : CopyIn(x2[base+i*tileNum],valid)
  mask = CompareLT(a,b,valid)
  out = Select(mask, Duplicate(1), Duplicate(0), valid)
  CopyOut(y[base+i*tileNum],out,valid)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 GM | `base+i*tileNum`或0 | valid | P'/连续ND | executor | CopyIn/Duplicate | executor | scalar只读元素0 |
| tensor pool | `bufferId*tileNum*M` | `N1*tileNum*M` | byte slices | MTE2/V | V/MTE3 | Kernel | 编译期LEVEL_2复用、ping/pong |
| y GM | `base+i*tileNum` | valid bytes | BOOL | Select/CopyOut | 后置Cast | executor | 只写有效元素 |

##### 分层内存布局

```text
GM(P') -> [CopyIn x2] -> [Compare mask + one/zero + Select byte] -> [CopyOut BOOL] -> GM
             tensorPool按OpDag bufferId切片，MTE2/MTE3节点ping-pong
```

##### 字节账与复用顺序

```text
UB_total = N1 * tileNum * M
tileNum = floor((UB/N1)/512)*512/M
workspace(core) = 16 MiB declared, entry unread
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 输入slice | MTE2/Duplicate | Compare | 下个同奇偶tile | `GetTensor/ReleaseTensor` | Compare消费完成 |
| mask/常量/Select slice | V | Select/CopyOut | 编译期下一合法节点 | LEVEL_2存活分析+mutex | 依赖最后一次读取完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→V | tensor mutex | CopyIn | Compare前GetTensor | 两输入slice | valid限制尾块 |
| V→MTE3 | tensor mutex | Select | CopyOut | BOOL slice | `<`按P'精确语义，NaN比较为false |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.h` | `DoOneDimFormerTilingAdvance` | 512B tile、核数调整 |
| Kernel entry/router | `less_apt.cpp` | `less<202>` | AIV entry实例化Less DAG |
| Device body/helper | `broadcast_sch_one_dim_advance.h` | `Process/Run` | scalar duplicate、双缓冲与tail |

### Route R-002：AIV NDDMA broadcast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_tiling.h` | `IsUbBroadcast` | 多维且谓词false | NDDMA |
| Tiling | 同上 | `SetSchMode` | running rank≤5 / >5 | 1 / 2 |
| Router | `broadcast_sch.h` | `BroadcastSch<1/2>` | key匹配 | NDDMA body |

#### Device 数据流

##### 函数级调用链

```text
less<T,1|2> -> BroadcastNddmaSch::Init/Process
  -> CopyInBrc(x1,x2) by 5D MultiCopy
  -> Less DAG vector nodes -> CopyOut BOOL
```

##### 源码等价伪码

```text
axes = decode(blockFormer*blockIdx)
for tile in blockFormer_or_tail:
  split = final_split ? ubTail : ubFormer
  for input in x1,x2:
    gmOffset = sum(axes[j]*inputBrcStride[j])
    build <=5D MultiCopy; mode2 loops dimensions above five
  mask = CompareLT(expandedX1,expandedX2)
  CopyOut(Select(mask,1,0), split*outputStride[splitAxis])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `sum(axis*stride)` | MultiCopy loop product | P'/broadcast | executor | NDDMA | tile | stride0重复源 |
| TBuf pool | `bufferId*elemNum*M` | `Nn*elemNum*M` | local slices | NDDMA/V | V/MTE3 | Kernel | 256元素对齐上限 |
| y GM | output stride offset | valid output elements | BOOL | CopyOut | Cast | executor | ubTail精确 |

##### 分层内存布局

```text
GM strided/broadcast -> 5D NDDMA expanded local -> Compare/Select local -> contiguous BOOL GM
```

##### 字节账与复用顺序

```text
UB_total = Nn * elemNum * M <= UB
workspace(core) = 16 MiB
dynamic_ub = N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| NDDMA输入 | MTE2 MultiCopy | Compare | 后续tile | MTE2/V mutex | current MultiCopy完成 |
| BOOL输出 | Select | MTE3 CopyOut | 后续tile | V/MTE3 mutex | CopyOut释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA→V | Get/ReleaseTensor | MultiCopy | Compare | expanded inputs | loopSize含split tail |
| V→GM | mutex | Select | CopyOut | BOOL bytes | ubTail仅最后split |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.cpp` | `GetBlockSplitFactor/DoBrodcastTiling` | 多维容量与分核 |
| Kernel entry/router | `broadcast_sch.h` | `BroadcastSch<1/2>` | mode到body |
| Device body/helper | `broadcast_sch_nddma.h` | `BroadcastNddmaWithoutLoop/WithLoop` | 五维描述及高rank外循环 |

### Route R-003：AIV UB Broadcast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_tiling.h` | `IsBigNLastBrc/IsUbBroadcast` | 大NLast或窄dtype+尾轴对齐 | UB Broadcast |
| Tiling | 同上 | `SetSchMode` | rank1..4 / >4 | 101..104 /109 |
| Router | `broadcast_sch.h` | `BroadcastSch<101..109>` | key匹配 | UB body |

#### Device 数据流

##### 函数级调用链

```text
less<T,101..104|109> -> BroadcastUbSch::Init/Process
  -> CopyInAndVecBrc(x1,x2) -> Compare -> Duplicate/Select -> CopyOut
```

##### 源码等价伪码

```text
for each tile and CopyInBrc input:
  if split-axis stride==0 and same source block is cached: reuse expanded local tensor
  else CopyIn compact GM span; AscendC::Broadcast to output tile shape
CompareLT -> Select(1,0) -> CopyOut(valid BOOL bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| compact input | broadcast GM offset | source inputLength | P'/compact | CopyIn | Broadcast | tile | stride0可缓存 |
| expanded input | combined buffer id | output tile elements | P'/output shape | Broadcast | Compare | tile/cache interval | rank专门化 |
| tensor pool | `bufferId*elemNum*M` | `Nu*elemNum*M` | local | MTE2/V | V/MTE3 | Kernel | LEVEL_2复用 |

##### 分层内存布局

```text
GM compact -> local compact -> local expanded -> Compare mask -> Select BOOL -> GM
                                  ^ stride0时跨ubLoop复用
```

##### 字节账与复用顺序

```text
UB_total = Nu * elemNum * M <= UB
workspace(core) = 16 MiB
dynamic_ub = N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| compact/expanded pair | MTE2/Broadcast | Compare | source block变化后 | MTE2/V mutex | 缓存guard为真时不覆盖 |
| Select输出 | V | MTE3 | 后续tile | V/MTE3 mutex | valid范围完成搬出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy→Broadcast | MTE2/V mutex | compact CopyIn | Broadcast | compact slice | tail shape改为ubTail |
| Broadcast→Compare | V mutex | Broadcast | Compare | expanded slice | 仅重排/复制，不改值 |
| Select→GM | V/MTE3 mutex | Select | CopyOut | BOOL slice | valid精确 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.h` | `IsUbBroadcast/SetSchMode` | route/rank选择 |
| Kernel entry/router | `broadcast_sch.h` | `BroadcastSch<101..109>` | rank模板路由 |
| Device body/helper | `broadcast_sch_ub_broadcast.h` | `CopyInAndVecBrc/CvecBrc` | compact copy、Broadcast、缓存 |

### Route R-004：AICPU Less

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lt_tensor.cpp` | promote/Cast | 950可得到U16/U32 `P'` | 同dtype核心输入 |
| L0 | `less.cpp` | `IsAiCoreSupport` | U16/U32不在RegBase表 | `LessAiCpu` |
| AICPU | `less_aicpu.cpp` | dtype switch | DT_UINT16/DT_UINT32 case | `LessCompute<T>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICPU(Less)
  -> LessCpuKernel::Compute -> LessParamCheck -> LessCompute<T>
     -> NoBcastCompute -> SpecialCompute
     or BcastCompute -> opbase::Bcast index mapping
```

##### 源码等价伪码

```text
if shapes_equal or numel(x1)==1 or numel(x2)==1:
  choose SAME_SHAPE/X_ONE_ELEMENT/Y_ONE_ELEMENT
  parallel if output_numel>=7K; <=35K cap workers at4
  out[i] = selected_x < selected_y
else:
  construct valid Bcast(right-aligned shapes, collapsed states, row-major strides)
  parallel if output_numel>=2K; <=16K cap workers at4
  out[i] = x[BcastXIndex(i)] < y[BcastYIndex(i)]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 task tensor | direct `i`、0或Bcast index | input numel | U16/U32 | executor Cast | CPU loop | AICPU task | 无UB |
| output tensor | `i` | output numel | bool | CPU loop | 后置Cast | executor | 每分片不重叠 |
| Bcast metadata | collapsed dims/strides | shape rank | int64 vectors | `Bcast`构造 | index mapper | task stack/object | 右对齐、broadcast stride语义 |

##### 分层内存布局

```text
executor tensor memory -> AICPU worker shard(s) -> BOOL executor tensor
```

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace = AicpuTaskSpace/runtime managed; fixed source无算子私有字节公式
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| output shard | one CPU worker | executor后置节点 | task完成后 | ParallelFor join | `[start,end)`不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU shards | `ParallelFor` | workers | task return/join | output区间 | 整数比较精确；末分片自然tail |
| task→Cast | executor顺序 | AICPU Less | 后置Cast | BOOL tensor | 无并发越界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host selector | `less.cpp` | `Less/IsAiCoreSupport/LessAiCpu` | U16/U32进入AICPU |
| Kernel entry/router | `less_aicpu.cpp` | `Compute` dtype switch | U16/U32实例 |
| Device body/helper | `less_aicpu.cpp`; opbase `bcast.cc` | `NoBcastCompute/BcastCompute/Bcast` | 快速路径、阈值分片、广播索引 |

## 算子特定优化

### Trick T-001：合轴一维用 512B advance 并在单 tile/核时主动减核

#### 触发与源码依据

- 适用 Route：R-001。
- 触发：合轴为一维且满足 advance guard。
- Golden 来源：[`broadcast_tiling.h:1082-1089`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h#L1082-L1089)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// 重新计算分核参数
uint64_t  ubOuter = (dimLength + tileNum - 1) / tileNum;
uint64_t  blockFormer = (ubOuter + coreNum - 1) / coreNum;
// 将核数再减少一半，减少核启动开销，并开启核内doublebuffer流水
if (blockFormer == 1) {
    blockFormer = DB_LOOP;
}
```

#### 机制与设计原因

初始 tile 按每个存活 buffer 的 512B 整数倍分配。当均分结果每核只有一个 tile 时，将 `blockFormer` 改为2并重算 `blockNum`，以减少实际启动核数，同时让每个工作核拥有可交替的 ping/pong tile。该技巧同时考虑了启动开销和搬运/计算流水，而不改变元素归属。

#### 迁移边界

可迁移到参数少、无 Var、连续一维且 DAG buffer 存活数可静态计算的逐元素 AIV 算子。若输出宽度与输入不同，必须像 Less 一样用 `MaxDtypeBytes/MinDtypeBytes` 分别约束容量与最小工作粒度。

#### 错误做法

不能把 `blockFormer=2` 理解成固定每核两块；它只在初始值为1时触发。也不能对多维广播强制使用该 POD，因为多维地址依赖完整 strides。

### Trick T-002：split 轴广播时缓存已展开输入

#### 触发与源码依据

- 适用 Route：R-003。
- 触发：输入在 `ubSplitAxis` 的广播 stride 为0，且后续 loop 仍处在同一非首广播块。
- Golden 来源：[`broadcast_sch_ub_broadcast.h:181-186`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h#L181-L186)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t realIdx = (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter;

if ((tilingData->inputBrcStrides[cBrcIndex_][tilingData->ubSplitAxis] == 0) && ubLoopIdx > 0 && realIdx > 0) {
    cBrcIndex_++;
    return;
}
```

#### 机制与设计原因

stride为0表示多个输出 tile 读取同一源块。首次把 compact 输入搬入 UB 并展开后，后续 loop 直接复用 expanded tensor，跳过 GM CopyIn 和向量 Broadcast；DAG 的 cache-aware buffer 分配保证该节点不被提前覆盖。

#### 迁移边界

适合 broadcast 输入在切分轴上保持同一源地址、且 scheduler 能让 expanded buffer 跨 loop 存活的逐元素 DAG。源地址随外层轴变化、tail shape变化或 buffer planner未启用 cache-aware 分配时必须重新展开。

#### 错误做法

不能只凭轴长度为1就无条件缓存；必须同时满足 stride0、loop位置和 `realIdx` guard。也不能在 NDDMA Route 套用该局部 tensor 生命周期。

## 模板与实现对比

| 维度 | Less 实现 | 通用 Broadcast 模板 | 结论 |
|---|---|---|---|
| shape/分核 | 无自定义公式 | 合轴、UB容量、block/tail全由 atvoss 产生 | Route选择和POD主要来自模板 |
| 计算 DAG | Compare LT→Duplicate 1/0→Select→CopyOut BOOL | scheduler按DAG依赖运行 | 算子特异点是mask物化为BOOL字节 |
| dtype | 950 AICore九种实际 `P'`；U16/U32 AICPU | 模板以 `T`、最大/最小dtype宽度计容量 | BF16 Tiling用同宽half DAG，Device仍按编译dtype实例化 |
| 广播 | NDDMA或UB Broadcast | selector、rank key、缓存均通用 | 不应把某一key写死为所有广播 |
| fallback | L0 dtype selector转AICPU | AIV模板本身无dtype fallback | fallback发生在核心launch前 |
| Cast | 条件独立节点 | 不在Less DAG内 | 其workspace、POD、SIMT/AICPU边不能并入Less核心 |

## 平台差异

| 平台/代际 | 固定源码状态 | 公开入口 | 核心实现 | 本卡处理 |
|---|---|---|---|---|
| 950/Ascend 950 | CMake `ascend950`→`arch35`，OpDef与binary config一致 | `aclnnLtTensor` | AIV Broadcast Less + U16/U32 AICPU | 完整覆盖 |
| A2/A3 | 文档声明支持，但 `math/less/` 固定源码未给出可归属这些代际的 Host/Tiling/Kernel闭包 | 文档有同一入口 | 缺失/冲突 | 不创建 `lt_tensor.md`，不推测复用950 |

## 限制与失败边界

- 只覆盖 `aclnnLtTensor`；Scalar、Inplace、Grad或其他比较变体不并入。
- 输入 rank 上限8；输入不可广播、输出 shape 不等于广播结果、Promote失败或 dtype越界均在第一段报错。
- 空输入成功返回0 workspace且无launch；不能据此推导核心支持零元素 Tiling。
- AIV Less要求两核心输入 dtype相同；该条件由前置Cast保证，独立调用核心时仍是硬约束。
- 950 AICore不承接 U16/U32 Less，由 AICPU闭合；这不是AIV Tiling失败后的重试。
- BOOL输入在本API比较前转换为UINT8，因此BOOL排序语义是0/1的无符号比较。
- FLOAT/FLOAT16/BFLOAT16/DOUBLE遵循硬件 `<`；NaN比较结果为false，源码没有额外NaN分支。
- Broadcast POD中若干窄字段/数组上限由模板固定；源码未见对 `blockNum` 写入int16的显式溢出检查。
- Contiguous、ViewCopy和 executor workspace聚合的内部实现不在固定仓直接依赖中，已作为框架边界标识，未猜测内部route。
- 本闭包未发现 Cube/AIC/MIX_CV；SIMT只出现在条件Cast local-memory配置，按VV处理。

## 源码索引

### 链路索引

| 链路阶段 | 文件/符号 | 固定事实 |
|---|---|---|
| 公开语义 | `docs/aclnnLtTensor&aclnnInplaceLtTensor.md` | dtype、shape、平台、两段式原型 |
| ACLNN入口 | `op_api/aclnn_lt_tensor.cpp::aclnnLtTensorGetWorkspaceSize/aclnnLtTensor` | 校验、空短路、完整节点顺序 |
| 核心selector | `op_api/less.cpp::Less/IsAiCoreSupport` | 950 AICore/AICPU分流、BOOL输出分配 |
| OpDef/构建 | `op_host/less_def.cpp`; `less/CMakeLists.txt` | 950 dtype与ascend950/arch35接线 |
| Tiling | `op_host/arch35/less_tiling_arch35.cpp::LessTiling` | dtype模板、key、AIV核数/UB |
| 核心DAG/entry | `op_kernel/arch35/less_dag.h`; `op_kernel/less_apt.cpp` | Compare/Select/CopyOut与router |
| AICPU核心 | `op_kernel_aicpu/less_aicpu.cpp::LessCpuKernel` | dtype switch、广播、分片阈值 |
| AICPU广播依赖 | opbase `bcast.h/bcast.cc::Bcast` | 右对齐、合并广播状态、索引映射 |
| AIV广播依赖 | opbase `broadcast_tiling*`,`broadcast_sch*` | 合轴、POD、202/1/2/101..109、NDDMA/UB body |
| 条件Cast | `math/cast/op_api/cast.cpp`, tiling/entry/body/AICPU | no-op/AIV/SIMT辅助/AICPU各边 |

### 实现复用检查

同仓 `math/less/` 只有一套 arch35 Less DAG/entry，950 与 `mc62` 构建复用该目录；本卡只把 `ascend950` 归入950。A2/A3公开声明没有固定源码中的代际实现可核对，未建立卡。核心广播不是 Less 私有 scheduler，而是固定 opbase Commit 的 atvoss 通用实现；本卡已追到其 Tiling、POD、router及三组Device body。AICPU通用广播同样追到 opbase `Bcast`。条件 Cast 是直接依赖的独立算子节点，闭合到其 950 AIV/SIMT辅助和AICPU实现，但没有把 Cast 技巧误写成 Less DAG内部行为。
