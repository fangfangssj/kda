---
type: Operator
title: "InplaceDiv 950 ACLNN 算子知识"
description: "说明 aclnnInplaceDiv 在 950 上从公开原地接口到 Host、RealDiv Tiling、AIV/AICPU Kernel 的完整实现链路。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/div/docs/aclnnDiv&aclnnInplaceDiv.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/div/op_api/aclnn_div.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_api/realdiv.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_host/real_div_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_host/arch35/real_div_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_host/config/ascend950/real_div_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_kernel/real_div_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_kernel/arch35/real_div_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_kernel/arch35/real_div_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/op_kernel_aicpu/real_div_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/real_div/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.h
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
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling_base.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling_noncontiguous.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_nlast_transpose_nddma.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/noncontiguous/nlast_transpose/broadcast_sch_nlast_transpose_ub_broadcast.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/noncontiguous/last_transpose/broadcast_sch_last_transpose.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/noncontiguous/transpose/broadcast_sch_nddma_transpose.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
created_at: "2026-08-25T10:35:41Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceDiv 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceDivGetWorkspaceSize` + `aclnnInplaceDiv`；仅原地 Tensor/Tensor 真除，不包含非原地 Div、Divs、DivMod、DivMods。 |
| 计算语义 | 广播后逐元素计算 `selfRef = selfRef / other`。950 先按互推导规则得到 `P`，整数/BOOL 推导结果再升为 FLOAT；结果最终转换回 `selfRef.dtype` 并写回原 view。 |
| 输入输出 | `selfRef` 为输入/输出，950 仅 FLOAT16、BFLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128；`other` 还可为 BOOL 与有符号/UINT8 整数。二者≤8维、可广播，且广播结果必须恰等于 `selfRef.shape`。 |
| 执行类型 | `P∈{FLOAT16,BFLOAT16,FLOAT}` 时核心是单 AIV Kernel；可连续、NDDMA、UB Broadcast 或非连续 transpose 调度。`P∈{DOUBLE,COMPLEX64,COMPLEX128}` 时核心是 AICPU。条件 Cast 为 AIV、SIMT primitive 或 AICPU。未触达 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 一维 advance；R-002 连续多维 NDDMA；R-003 连续多维 UB Broadcast；R-004/R-005 非连续 NLast NDDMA/UB；R-006 LastTranspose；R-007 非连续回退 NDDMA；R-008 AICPU。空 Tensor 由 Host 成功短路。 |
| 核心资源 | AIV 路线使用 GM、AIV local tensor pool、MTE2/V/MTE3 mutex及可选 NDDMA；核心节点声明16 MiB workspace但 entry不读。AICPU 按元素或广播 stride 读取。 |
| 来源闭包 | `source_closed`：公开 API→原地校验→复用 `aclnnDivGetWorkspaceSize` 的真除图→RealDiv 950 Host/Tiling/POD→全部 AIV scheduler/body与 AICPU body→构建接线闭合；executor、Contiguous、ViewCopy 是明确框架边界。 |
| 验证状态 | 仅完成固定提交源码闭包；按任务约束未运行 validator、编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `B=broadcast_shape(selfRef.shape, other.shape)`。接口先要求 `B==selfRef.shape`，所以 `other` 只能向原地目标广播，不能扩大目标。950 用 `P=InferDivModeDtype(selfRef.dtype,other.dtype,MODE_REAL_DIV)`；若普通 Promote 结果是整数且不是 INT32/BOOL，则 `PromoteIntegerInputsToFloat` 将其变为 FLOAT，而 RealDiv 对 INT32/BOOL 又会令输出为 FLOAT。由于 950 公开约束禁止整数/BOOL `selfRef`，实际核心 `P` 只会是 FLOAT16、BFLOAT16、FLOAT、DOUBLE、COMPLEX64 或 COMPLEX128。

对输出坐标 `i∈B`，`s_i` 与 `o_i` 按各自 view stride/broadcast stride 取数，计算 `q_i=cast_P(s_i)/cast_P(o_i)`，再转换成 `selfRef.dtype`，由 `ViewCopy` 写回原 storage/view。FLOAT16/BFLOAT16 AIV DAG 内部先升为 FLOAT，使用高精度除法后以 `RINT` 转回；FLOAT 直接用高精度除法。DOUBLE/复数由 AICPU 的 C++ `/` 实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束与语义 |
|---|---|---|---|
| `selfRef` | ND，0–8维，可非连续；既输入又输出 | FLOAT16、BFLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128 | 950 文档明确禁止 INT32/64/16/8、UINT8、BOOL；广播结果必须保持自身 shape。 |
| `other` | ND，0–8维，可非连续 | FLOAT16、BFLOAT16、FLOAT、DOUBLE、INT64/32/16/8、UINT8、BOOL、COMPLEX64、COMPLEX128 | 必须与 `selfRef` 可 Promote 且可广播；可经 Contiguous/Cast 变为 `P`。 |
| `workspaceSize` | Host 标量输出 | `uint64_t*` | executor 中节点 workspace、临时 tensor 与框架对齐后的总量。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 保存预处理、核心、后处理节点；交给第二段执行。 |

不接受私有 storage format；源码错误信息列出的公开格式为 ND/NCHW/NHWC/HWCN/NDHWC/NCDHW，而公开接口表以 ND 描述。除数为零没有 Host Guard：浮点/复数遵循对应 Device 除法行为；只有当前公开 950 原地接口不可达的 AICPU 整数模板会主动检查零。

### 入口与版本

本卡只覆盖 `aclnnInplaceDiv`。950 上 `isMixDtypeTensorSupport` 只对 ASCEND910B/ASCEND910_93 返回真，因此原地入口总是进入 `aclnnDivGetWorkspaceSize(selfRef,other,selfRef,...)`，复用非原地函数的实现图，但输出别名仍是原 `selfRef`。`real_div` 的 OpDef、binary config 与 CMake 均接线 `ascend950`/`arch35`。

公开文档同时声称 A2/A3 支持，但固定 `ops-math` 提交的 `real_div/CMakeLists.txt` 将这些平台指向 `arch32`，仓内却没有 `math/real_div/op_host/arch32` 和对应 arch32 Kernel。该代链路无法在固定提交闭合，因此未创建无后缀 `inplace_div.md`；本卡不以 arch35 反推旧代实现。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckInplace` 先检查 `selfRef/other` 非空，再推导广播 shape，并要求其等于 `selfRef.viewShape`。随后复用 `aclnnDivGetWorkspaceSize`：创建 executor，按 shape≤8且广播/输出 shape 相等→输入 dtype 白名单→Promote 与结果可回写→非私有格式检查。空输入直接返回 `workspaceSize=0`，没有 Device 节点。

非空时 950 一定走 `HandleNotMixDataTypeDiv`。每个输入独立处理：若 `input.dtype==P` 且 `IsRegBase && IsAiCoreSupport(input)`，`CreateView` 保留 view/storage shape、stride 与 offset，允许 RealDiv 直接选择非连续 Route；否则先 `Contiguous`，再 `Cast(...,P)`。Cast 在 dtype 相同即 no-op；其余按 regbase pair 走 AIV，若 pair 不支持则走 AICPU。特殊 `INT64→DOUBLE` AIV Cast 的 Tiling 为 SIMT primitive 预留32 KiB local memory，仍属于 VV 范围。

### 参数、Workspace 与生命周期

RealDiv Broadcast Tiling 总把自身 `workspace[0]` 写为 `16*1024*1024` 字节；AIV entry 的 `workspace` 参数未被读取。AIV Cast 也声明16 MiB最小 workspace。API 返回的是 `uniqueExecutor->GetWorkspaceSize()` 汇总值，不等于简单相加：中间 tensor 复用、框架节点、对齐和 workspace offset 的算法在固定算子仓不可见。调用方必须让 workspace 与 executor 存活到 `aclnnInplaceDiv` 完成。

RealDiv 输出由 executor 临时分配；随后 `Cast(divOpOut,selfRef.dtype)`（相同时 no-op），最后 `ViewCopy(castOut,selfRef)` 才覆盖原 view。因此核心读 `selfRef` 与最终写回之间有明确临时结果，不在 RealDiv Kernel 内直接原地覆盖。Contiguous/ViewCopy 的内部 Kernel/POD 与 executor 的事件编排属于框架边界，固定仓不可见。

### 启动图与失败传播

```text
CheckInplace
  -> aclnnDivGetWorkspaceSize(selfRef, other, out=selfRef)
     -> CheckParams
     -> empty ? success(workspace=0)
     -> per input:
          dtype==P && AIV-supported ? CreateView(preserve stride)
                                    : Contiguous -> Cast(to P)
     -> RealDiv(P,P) [AIV or AICPU]
     -> Cast(to selfRef.dtype)
     -> ViewCopy(to aliased selfRef)
  -> CommonOpExecutorRun(workspace, executor, stream)
```

每一步空指针/launch-list 失败都由 `CHECK_RET` 立即返回，后续节点不再加入。第二段只调用 `CommonOpExecutorRun` 并原样传播状态；节点次序由同一 executor 保证，固定仓没有暴露更细粒度 event ID。

## Route 与参数设计

### 分核与资源计算

AIV Tiling 先要求 x1/x2/y dtype 一致，再用 `BroadcastBaseTiling<OpDag>::DoTiling`。它读取 view shape/stride，合并 broadcast 标志和 stride 兼容的相邻轴。连续一维 Route 以 `B_N=OpDag::GetBufferNum<true,true>` 切分：`tileNum=floor((UB/B_N)/512)*512/maxDtypeBytes`，再按元素量选择单512B cache line或按核均分；`ubOuter=ceil(dimLen/tileNum)`，`blockFormer=ceil(ubOuter/aivCoreNum)`（为1时改2），`blockNum=ceil(ubOuter/blockFormer)`。

其余 Route 以 DAG buffer 数和 dtype bit 数求 `maxElemNum`，由末轴向前选择 `ubSplitAxis/ubFormer/ubOuter/ubTail`。令 `F=ubOuter*Π(outputDims[0:ubSplitAxis])`，则 `blockFormer=ceil(F/coreNum)`、`blockNum=ceil(F/blockFormer)`、`blockTail=F-(blockNum-1)*blockFormer`，最终 `blockDim=blockNum`。非连续 optimized POD 限4个合并轴并添加32B对齐/padded stride；fallback 保留最多8轴。

### Route 选择

| 优先级 | Route | Key | 精确条件 | 去向 |
|---:|---|---|---|---|
| 1 | R-001 | 202 | AIV dtype；合轴 rank=1；DAG 参数数3≤4且无 Var | rank>1→后续 |
| 2 | R-004/R-005/R-006 | 301/302/303 | 存在非连续 view，且无隐式 stride-0 broadcast-to；合轴≤4；满足 NLast 或末两轴 transpose 模板 | 不满足→R-007 |
| 3 | R-007 | 304/305 | 非连续但 optimized 模板不支持；UB内轴≤5为304，否则305 | — |
| 4 | R-003 | 101–104/109 | 连续、多维且 `IsUbBroadcast`：大 NLast，或广播输入 dtype∈{INT8,UINT8,F16,BF16,INT16,UINT16} 且尾轴按32B对齐 | 条件假→R-002 |
| 5 | R-002 | 1/2 | 连续、多维 NDDMA；UB内轴≤5为1，否则2 | — |
| 独立 | R-008 | AICPU | `P` 不在 950 AICore 表，即公开路线中的 DOUBLE/COMPLEX64/COMPLEX128 | 不回退 AIV |

非连续 optimized 的细分是：最后轴连续且非最后轴 transpose 数据满足 NLast 条件时，NDDMA 为301、UB Broadcast 为302；只交换最后两轴且 rank≤3时为303。若末轴 stride>1、存在真实 broadcast-to、rank>4或小包/尺寸 guard 不满足，则转304/305。`broadcast_sch.h` 入口固定 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，故所有 AIV Route 都不包含 Cube/AIC/MIX_CV。

### 模板/实现执行矩阵

| Route | dtype/shape | 核心模板/body | 搬运 | tail/fallback |
|---|---|---|---|---|
| R-001 | P=F16/BF16/F32，连续合轴1D | `BroadcastOneDimAdvanceSch<OpDag>` | `DataCopyPad`，标量输入可 `Duplicate` | 最后一核blockTail、最后tile ubTail |
| R-002 | 同dtype，连续多维、NDDMA selector | `BroadcastNddmaSch<1/2,OpDag>` | 五维 `DataCopy`；mode2加外层轴循环 | 广播 slice 可跨loop缓存 |
| R-003 | F16/BF16，或大NLast触发 | `BroadcastUbSch<101..104/109,OpDag>` | compact CopyIn后 `AscendC::Broadcast` | 109为动态rank |
| R-004 | optimized非连续 NLast、NDDMA | `BroadcastNlastTransposeNddmaSch` | padded UB layout、NDDMA/MovAlign | padded stride恢复GM布局 |
| R-005 | optimized非连续 NLast、UB | `BroadcastNlastTransposeUbBroadcastSch` | compact搬入、4D Broadcast、MoveAlign输出 | 最后tile使用padded尺寸 |
| R-006 | rank≤3且最后两轴 transpose | `BroadcastLastTransposeSch` | NDDMA或MovAlign分段搬运 | last-axis独立循环/tail |
| R-007 | 其余非连续 | `BroadcastNddmaTransposeSch<304/305>` | 最多五维NDDMA；305外循环 | 保留最多8轴stride |
| R-008 | P=DOUBLE/COMPLEX64/COMPLEX128 | `RealDivCompute`→same/scalar或general broadcast body | AICPU直接访存；大任务并行 | same阈值7168，broadcast阈值2048 |

F16 DAG 是 `CopyInBrc<half>→Cast<float,NONE>→DivHighPrecision<float>→Cast<half,RINT>→CopyOut`；BF16 使用 `bfloat16_t` 同构；FLOAT 是 `CopyInBrc<float>→DivHighPrecision<float>→CopyOut`。R-001～R-007 只改变 scheduler/搬运，不改变计算 DAG。

### 隐藏性能 Guard

| Guard | 条件 | 原因/影响 | 不满足时 |
|---|---|---|---|
| AICore dtype | 两输入与输出同dtype，且P∈F16/BF16/F32 | RealDiv 950 binary与单dtype DAG约束 | DOUBLE/复数→R-008；不一致Tiling失败 |
| 一维 advance | 合轴rank=1且参数数≤4 | 以512B tile获得轻量调度 | 多维selector |
| UB Broadcast dtype | 普通对齐触发只接受INT8/UINT8/F16/BF16/INT16/UINT16 | Broadcast primitive支持集合 | FLOAT通常R-002；大NLast仍可R-003 |
| NDDMA维数 | UB内轴≤5 | `MultiCopyParams`最多5维 | 超过选mode2/305循环 |
| optimized非连续 | rank≤4、无真实stride-0 broadcast-to；末轴/transpose/128B与8KiB×核数条件命中 | 专用padded POD与搬运器边界 | R-007 |
| LastTranspose | 只有最后两轴形成transpose且rank≤3 | 专用last-axis嵌套循环 | R-004/R-005或R-007 |
| AICPU并行 | same/scalar元素数≥7168；general broadcast≥2048 | 降低小任务并行开销 | 小任务单核；中等任务最多4核 |
| INT64→DOUBLE Cast | Direct Cast policy | SIMT primitive需保留32KiB local memory | 不改变RealDiv核心Route |

### POD/ABI

`RealDivFloatWithCast/WithoutCast` 的 DAG 常量为 `InputSize=2, CopyBrcSize=2, VecBrcSize=0`。以下 offset/sizeof 由固定字段声明与 C++ 自然对齐得到；元素 stride 均以元素为单位，不是字节。

| POD | 关键字段与 offset | sizeof/align | Producer | Consumer |
|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen@0:i64,tileNum@8:i32,blockNum@12:i16,scalarFlag@14:i16` | 16/8 | `DoOneDimOpTilingAdvance` | R-001 `Init/Process` |
| `BroadcastBaseTilingData<OpDag>` | `scheMode@0,shapeLen@4,ubSplitAxis@8,ubFormer@12,ubTail@16`; `ubOuter@24,blockFormer@32,blockTail@40,dimProduct@48,elemNum@56,blockNum@64` | 632/8 | 普通/回退 Broadcast Tiling | R-002/R-003/R-007 |
| 同上数组 | `outputDims@72,outputStrides@136,inputDims@200,inputBrcDims@232,inputStrides@360,inputBrcStrides@488,scalarData@616` | 同上 | 合轴shape/stride生成 | scheduler地址与广播参数 |
| `BroadcastNlastTransposeTilingData<OpDag>` | 五个i32@0..16；六个i64 `ubOuter..blockNum`@24..64；`minDtypeBlockAlignSize@72`；数组 `outputDims@80,outputStrides@112,padded@144,inputDims@176,inputBrcDims@240,inputStrides@304,inputBrcStrides@368,scalar@432` | 448/8 | NonContiguous NLast Tiling | R-004/R-005 |
| `BroadcastLastTransposeTilingData<OpDag>` | 八个i32@0..28；七个i64@32..80；数组 `outputDims@88,outputStrides@120,padded@152,inputDims@184,inputBrcDims@248,inputStrides@312,inputBrcStrides@376,scalar@440` | 456/8 | LastTranspose Tiling | R-006 |
| `CastTilingData` | 12个i64，依次为blockNum、ubFormer、blockFormer、两类loop/tail、reg copy step、三类reg loop | 96/8 | 950 Cast Tiling | 条件Cast AIV body |

`scheMode` 同时编码 TilingKey。RealDiv binary 声明 1、2、101–104、109、202、301–305；201/999 虽在公共宏声明，但本 DAG 的选择器不可达。核心 Broadcast workspace 是16 MiB；其在 executor 总 workspace 中的 offset 固定仓不可见。

### Route 源码闭包

| Route | Host/Tiling producer | POD | entry/router | body/直接依赖 | 构建 |
|---|---|---|---|---|---|
| R-001 | `real_div_tiling_arch35.cpp`→`DoTiling`→`DoOneDimOpTilingAdvance` | OneDimAdvance | `real_div_apt.cpp::real_div<202>`→`BroadcastSch<202>` | `broadcast_sch_one_dim_advance.h`; `real_div_dag.h`; `vec.h` | `real_div/CMakeLists.txt` ascend950/arch35 |
| R-002 | `SetSchMode`选1/2 | Base POD | `BroadcastSch<1/2>` | `broadcast_sch_nddma.h` | 同上 |
| R-003 | `IsUbBroadcast`选101–104/109 | Base POD | `BroadcastSch<101..109>` | `broadcast_sch_ub_broadcast.h` | 同上 |
| R-004 | NonContiguous selector选301 | NLast POD | `BroadcastSch<301>` | `broadcast_sch_nlast_transpose_nddma.h` | 同上 |
| R-005 | NonContiguous selector选302 | NLast POD | `BroadcastSch<302>` | `broadcast_sch_nlast_transpose_ub_broadcast.h` | 同上 |
| R-006 | LastTranspose selector选303 | Last POD | `BroadcastSch<303>` | `broadcast_sch_last_transpose.h` | 同上 |
| R-007 | fallback选304/305 | Base POD | `BroadcastSch<304/305>` | `broadcast_sch_nddma_transpose.h` | 同上 |
| R-008 | `realdiv.cpp::IsAiCoreSupport==false` | AICPU Tensor metadata，无AIV POD | `ADD_TO_LAUNCHER_LIST_AICPU(RealDiv)` | `real_div_aicpu.cpp::RealDivCompute/BcastDivInfo` | AICPU `REGISTER_CPU_KERNEL(RealDiv)` |

## Kernel 实现

### 公共执行流程

AIV entry 只解析 `schMode` 并实例化 dtype DAG，然后调用 `BroadcastSch<schMode,OpDag>.Process(x1,x2,y,tiling)`。公共 router 首行指定 AIV-only；scheduler 建 GM tensor、读取 POD、按 blockIdx 计算 GM 起点，循环执行 CopyIn/Compute/CopyOut。DAG local memory配置为 `MemCfg<LEVEL_2>`，临时 tensor 通过编译期 buffer ID 复用；CopyIn、V计算、CopyOut 由 `MTE2/V/MTE3` mutex约束生产消费顺序。

AICPU entry 校验 tensor数量、空 tensor和两输入dtype相同后分派。`BcastDivInfo` 将shape补齐到最多8维，把广播轴stride置0，并合并相邻兼容轴。same-shape/单scalar走直线循环；其余按坐标和有效stride推进。这里的 AICPU 输入在 Host 上会先 Contiguous/Cast，故公开 R-008 通常是连续 storage，但广播 stride仍由输出shape生成。

### Route R-001：一维 Advance AIV

#### 适用条件

合轴后rank=1，P为F16/BF16/FLOAT。

##### 选择证据

`BroadcastBaseTiling::DoTiling` 在输入全连续、DAG参数数≤4且rank1时调用 advance Tiling，key为202。

#### Device 数据流

##### 函数级调用链

`real_div<202>`→`BroadcastSch<202>`→`BroadcastOneDimAdvanceSch::Process`→`CopyInBrc`→RealDiv DAG→`CopyOut`。

##### 源码等价伪码

```text
for tile assigned to blockIdx (ping = tile & 1):
  load x1,x2; scalarFlag input uses Duplicate
  q = high_precision_div(cast_float_if_needed(x1), cast_float_if_needed(x2))
  store cast_RINT_if_needed(q)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层 | 内容 | 大小/行为 |
|---|---|---|
| GM | 两输入与临时输出 | 连续base offset按核/tile递增 |
| AIV local | DAG tensor pool | `B_N*tileNum*maxDtypeBytes`，编译期子区 |
| 搬运 | `DataCopyPad`/`Duplicate` | 最后tile用真实tail长度 |

##### 分层内存布局

GM→ping/pong local输入→FLOAT中间（F16/BF16）→local输出→GM。

##### 字节账与复用顺序

Host 用 `B_N` 参与tile上限；Device以相同DAG buffer ID复用，互斥顺序是 MTE2 完成→V读取/写结果→MTE3回写。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | ping/pong与tensor mutex；无跨核归约 |
| 精度 | F16/BF16升FLOAT，FLOAT直接 `DIFF_COMPENSATION` Div |
| tail | 最后一核用blockTail，最后tile用ubTail；DataCopyPad处理非32B整块 |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastOneDimAdvanceSch::Process` | 核/tile/ping-pong循环 |
| `Vec::DivHighPrecision` | `DivConfig{DIFF_COMPENSATION}` |
| `RealDivFloatWithCast` | 半精度升FLOAT并RINT回写 |

### Route R-002：连续多维 NDDMA AIV

#### 适用条件

连续、多维且 UB Broadcast guard不成立；key1覆盖UB内≤5轴，key2覆盖更多轴。

##### 选择证据

`SetSchMode` 在 `IsUbBroadcast()==false` 后按 `shapeLen-ubSplitAxis` 与5比较选择1/2。

#### Device 数据流

##### 函数级调用链

`real_div<1/2>`→`BroadcastNddmaSch<withLoop>`→`CopyInBrc`五维 NDDMA→DAG→CopyOut。

##### 源码等价伪码

```text
for fused tile of this core:
  derive axesIndices
  NDDMA x1/x2 into local (cache slice if split stride is 0)
  run RealDiv DAG for ubSplitSize * outputStride[split]
  copy result to contiguous GM output
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| local pool | `InitBuffer(elemNum*maxBytes*B_N)` |
| CopyIn | `DataCopy<T,5>`；key2由外循环分解高rank |
| CopyOut | 按output stride算GM offset，连续tile回写 |

##### 分层内存布局

非连续来源不存在；NDDMA只负责广播shape组合到连续local布局。

##### 字节账与复用顺序

每个活跃tensor子区为 `elemNum*maxBytes`；广播输入在相邻tile复用时跳过重复CopyIn。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | CopyInBrc/V/CopyOut mutex |
| 精度 | 与R-001相同 |
| tail | ubTail缩短split轴；mode2外循环不改变数值顺序 |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastNddmaSch` | 五维CopyIn与缓存broadcast slice |
| `BroadcastNddmaWithLoop` | 高rank外循环 |

### Route R-003：连续多维 UB Broadcast AIV

#### 适用条件

连续多维，`IsUbBroadcast` 命中；普通dtype集合使公开路线主要为F16/BF16，大NLast特判可覆盖FLOAT。

##### 选择证据

Tiling 对大NLast优先返回true，否则要求广播输入dtype在白名单且输出尾轴按32B对齐；rank1..4映射101..104，更高映射109。

#### Device 数据流

##### 函数级调用链

`real_div<101..104/109>`→`BroadcastUbSch`→compact CopyIn→`AscendC::Broadcast`→DAG→CopyOut。

##### 源码等价伪码

```text
load compact source slice
expand broadcast dimensions in local tensor (fixed rank or dynamic 109)
high_precision_div(expanded_x1, expanded_x2)
copy contiguous result to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| local pool | `B_U*elemNum*maxBytes`，含compact源与expanded结果 |
| CopyIn | `DataCopyPad` compact slice |
| Broadcast | AIV向量Broadcast，不用Cube |

##### 分层内存布局

GM compact广播源→local compact→local expanded→RealDiv local→GM。

##### 字节账与复用顺序

`GetBufferNum<false,true>` 将UB Broadcast额外节点计入容量；同一split广播片可跨loop缓存。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | MTE2完成后Broadcast/V；MTE3消费结果 |
| 精度 | 与R-001相同 |
| tail | 动态rank109仍使用同一ubTail与blockTail |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastUbSch::CvecBrc` | 固定/动态rank展开 |
| `ubBroadcastDtypes` | 普通selector的dtype白名单 |

### Route R-004：非连续 NLast NDDMA AIV

#### 适用条件

保留stride的输入命中optimized NLast模板且选择NDDMA，key301。

##### 选择证据

NonContiguous selector要求合轴≤4、无真实stride-0 broadcast-to，并满足末轴连续/小包/总尺寸guard；NDDMA分支写301。

#### Device 数据流

##### 函数级调用链

`real_div<301>`→`BroadcastNlastTransposeNddmaSch`→NLast NDDMA/MovAlign→DAG→padded CopyOut。

##### 源码等价伪码

```text
derive source offset from original strides
gather noncontiguous NLast tile into padded local layout
run RealDiv DAG
move aligned packed result to contiguous temporary output
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| POD | NLast结构，4轴shape/原stride/padded stride |
| local | `B_N*elemNum*maxBytes` |
| 搬运 | 五维NDDMA；可退化MovAlign |

##### 分层内存布局

strided GM→padded local→DAG local→padded output slice→contiguous GM临时结果。

##### 字节账与复用顺序

32B最小dtype对齐决定 padded stride；Host容量公式已把padding/双缓冲计入。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | NDDMA/MTE2、V、MTE3 mutex |
| 精度 | 与连续路线一致 |
| tail | ubTail与padded stride共同限定有效元素；padding不写入最终逻辑shape |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastNlastTransposeNddmaSch` | 301专用调度 |
| `outputStridesWithPad` | local padded布局 |

### Route R-005：非连续 NLast UB Broadcast AIV

#### 适用条件

同R-004的optimized NLast边界，但广播selector选择UB，key302。

##### 选择证据

NonContiguous selector在NLast可用且 `IsUbBroadcast` 为真时写302；真实stride-0 broadcast-to不允许进入该optimized模板。

#### Device 数据流

##### 函数级调用链

`real_div<302>`→`BroadcastNlastTransposeUbBroadcastSch`→CopyIn→4D Broadcast→DAG→MoveAlign CopyOut。

##### 源码等价伪码

```text
load strided compact source into padded local
UB-broadcast to 4D logical tile
run RealDiv DAG
strip padding and move aligned result to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| local | `B_U*elemNum*maxBytes` |
| CopyIn | loop/padded stride搬入 |
| Expand/Out | `Broadcast<T,4>`；MoveAlign恢复packed GM |

##### 分层内存布局

strided compact GM→padded compact local→expanded local→DAG→packed GM。

##### 字节账与复用顺序

UB容量同时覆盖compact、expanded和DAG中间；只对有效逻辑元素计算，pad仅服务搬运对齐。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | CopyIn/Broadcast/Div/CopyOut依赖由mutex串联 |
| 精度 | 与R-001一致 |
| tail | 最后tile按padded output stride搬运并裁剪 |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastNlastTransposeUbBroadcastSch` | 302专用body |
| `MoveAlign` | padded local到packed GM |

### Route R-006：非连续 LastTranspose AIV

#### 适用条件

合轴rank≤3且只有最后两轴呈transpose stride模式，key303。

##### 选择证据

`isLastTranspose` 在非连续模板分析中优先识别末两轴交换，并选择专用 LastTranspose POD/body。

#### Device 数据流

##### 函数级调用链

`real_div<303>`→`BroadcastLastTransposeSch`→last-axis分段NDDMA/MovAlign→DAG→CopyOut。

##### 源码等价伪码

```text
for ubOuterLastAxis segment:
  transpose-aware gather x1/x2
  run RealDiv DAG on valid segment
  transpose-aware aligned output move
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| POD | 独立ubFormerLastAxis/ubTailLastAxis/ubOuterLastAxis |
| CopyIn | 4维NDDMA或MovAlign |
| local | 按minDtypeBlockAlignSize填充倒数第二轴 |

##### 分层内存布局

transposed-view GM→按最后轴分段的padded local→DAG→连续GM临时结果。

##### 字节账与复用顺序

外层tile与last-axis segment构成嵌套循环；同一输入广播segment按stride guard复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | segment内MTE2/V/MTE3；核间独立 |
| 精度 | 与R-001一致 |
| tail | 同时处理ubTail和ubTailLastAxis |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastLastTransposeSch` | 303嵌套segment调度 |
| `BroadcastSetNddmaConfigWithoutLoop` | transpose stride搬运参数 |

### Route R-007：非连续回退 NDDMA AIV

#### 适用条件

非连续输入不满足optimized模板；key304覆盖UB内≤5轴，305覆盖更多轴。

##### 选择证据

selector先运行普通 Broadcast Tiling生成最多8轴 Base POD，再覆盖key为304/305，从而保留原view stride。

#### Device 数据流

##### 函数级调用链

`real_div<304/305>`→`BroadcastNddmaTransposeSch`→非连续NDDMA→DAG→CopyOut。

##### 源码等价伪码

```text
for fused output tile:
  calculate coordinates and each input GM offset from original strides
  NDDMA gather (305 loops axes beyond five)
  run RealDiv DAG
  copy contiguous result tile to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| POD | Base结构，最多8轴shape/stride |
| CopyIn | 非连续五维NDDMA，305外循环 |
| local | `B_N*elemNum*maxBytes` |

##### 分层内存布局

任意支持stride GM→连续local→DAG→连续GM临时结果。

##### 字节账与复用顺序

fallback避免Host强制Contiguous，但通常比optimized路径增加地址计算/NDDMA循环；容量仍由普通Broadcast公式约束。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | NDDMA与V/MTE3 mutex |
| 精度 | 与R-001一致 |
| tail | ubTail；305额外处理高rank坐标carry |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BroadcastNddmaTransposeSch` | 304/305共用非连续body |
| `NonContiguousDimensionCollapse` | stride兼容合轴 |

### Route R-008：DOUBLE/复数 AICPU

#### 适用条件

公开950 `P` 为DOUBLE、COMPLEX64或COMPLEX128；RealDiv AICore白名单不包含这些dtype。

##### 选择证据

`realdiv.cpp::IsAiCoreSupport` 的regbase集合只有FLOAT/F16/BF16/BOOL/INT32/INT64；否则明确提交AICPU RealDiv。

#### Device 数据流

##### 函数级调用链

`ADD_TO_LAUNCHER_LIST_AICPU(RealDiv)`→`RealDivKernel::Compute`→`RealDivCompute`→`RealDivSameTypeCompute<T>`→`SpecialComputeImpl`或`BcastComputeImpl`。

##### 源码等价伪码

```text
collapse compatible broadcast axes; effective broadcast stride = 0
if same shape or either input scalar:
  for i: y[i] = scalar_or_x[i] / scalar_or_y[i]
else:
  for output coordinate:
    y[pos] = x[x_offset] / other[other_offset]
    advance coordinates with carry
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| tensor memory | AICPU直接读取连续预处理输入、写临时输出 |
| metadata | 最多8轴dims与有效stride；相邻兼容轴折叠 |
| 并行 | same/scalar≥7168，general≥2048；中等规模最多4核并预留2 CPU |

##### 分层内存布局

GM tensor→AICPU cache/register标量或逐元素值→GM临时结果；无AIV UB/POD。

##### 字节账与复用顺序

标量输入只读取一次；inner stride为1走连续strip，stride0值在内层hoist。线程处理互不重叠的输出区间。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项 | 行为 |
|---|---|
| 同步 | AICPU parallel shard完成后节点结束；executor再启动后续Cast/ViewCopy |
| 精度 | C++ double/`std::complex`除法；`verify_zero=false` |
| tail | parallel shard以[start,end)覆盖余数；广播坐标从start恢复 |

#### 关键实现

##### 符号证据表

| 符号 | 结论 |
|---|---|
| `BcastDivInfo` | shape补齐、stride0、合轴 |
| `SpecialComputeImpl` | same/scalar三种快路 |
| `BcastComputeImpl` | 通用广播与四类inner strip |

## 算子特定优化

### Trick 1：保留可执行的非连续 View，避免前置物化

当输入dtype已经等于P且950 AIV支持时，Host不是无条件 `Contiguous`，而是 `CreateView` 原样携带 storage shape、view stride和offset。随后 RealDiv 的301–305专用scheduler直接按stride搬入。这把一次完整GM读写的Contiguous节点换成核心tile内的NDDMA/MovAlign；只要另一输入需要Cast，它仍可独立连续化，不妨碍本输入保留view。

```cpp
if (input->GetDataType() == promoteType && IsRealDivSupportNonContiguous(input)) {
    processed = executor->CreateView(input, viewShape, storageShape, viewStrides, viewOffset);
} else {
    processed = Cast(Contiguous(input), promoteType);
}
```

### Trick 2：半精度存储、FLOAT高精度计算

F16/BF16 DAG 把两个输入都以 `NONE` 模式升到FLOAT，使用 `DivHighPrecision<float>` 的 `DIFF_COMPENSATION` 配置，再以 `RINT` 写回半精度。这样保持公开输出dtype和带宽，同时把核心除法放到FLOAT精度；FLOAT路线省去两次Cast节点。

### Trick 3：AICPU按广播形态拆快路并折叠stride

AICPU先把广播轴stride置0并合并相邻兼容轴。same-shape、x scalar、y scalar三种情形不维护多维坐标；通用广播又分别优化双stride1、y stride0、x stride0和一般stride。该优化直接减少DOUBLE/复数路线的地址计算与重复标量加载。

## 模板与实现对比

| 对比项 | 通用 Broadcast 模板 | InplaceDiv 实际特化 |
|---|---|---|
| dtype | 公共模板可承载多类DAG | 公开950 AIV只达F16/BF16/FLOAT；DOUBLE/复数切AICPU |
| key空间 | 宏声明1/2、101–109、201/202、301–305、999 | 实际binary/selector达1/2、101–104/109、202、301–305；201/999不可达 |
| 计算节点 | 任意OpDag | 固定两次CopyInBrc→高精度Div→CopyOut，半精度带两次Cast |
| 非连续 | 模板含optimized/fallback族 | 由Host保留view触发；dtype需已经是P且AIV支持 |
| workspace | 模板固定16MiB | entry未读；executor汇总值与offset不可见 |
| 原地语义 | 模板输出独立tensor | API由临时RealDiv输出+Cast+ViewCopy实现最终原地覆盖 |

## 平台差异

| 平台 | 公开API | 固定仓实现 | 结论 |
|---|---|---|---|
| 950 / Ascend 950PR、950DT | 支持；selfRef禁整数/BOOL | `ascend950`→arch35 RealDiv AIV/AICPU闭合 | 本卡覆盖 |
| A2/A3 | 文档写支持（A3不含BF16） | CMake指向arch32，但固定提交缺失RealDiv arch32 Host/Kernel | 证据不足，未建 `inplace_div.md` |

不要把 `math/div/op_kernel/div_apt.cpp` 当作本接口核心：该目录的原生 Div Kernel是另一条实现。`aclnnInplaceDiv` 在950实际经L0 `RealDiv`，因此本卡只追到 `math/real_div` 及其直接 Cast/atvoss依赖。

## 限制与失败边界

| 边界 | 行为 |
|---|---|
| null | `selfRef`/`other`为空返回 `ACLNN_ERR_PARAM_NULLPTR`；workspace/executor指针由复用入口检查/使用。 |
| shape | 任一输入>8维、不可广播，或广播结果扩大selfRef，返回参数错误。 |
| dtype | 不在白名单、不可Promote或结果不可转换回selfRef.dtype时失败；950 selfRef整数/BOOL属于公开接口禁用。 |
| format | 私有storage format失败。 |
| empty | workspace=0且无Device launch。 |
| 除零 | 公开可达浮点/复数路线不做零除Host检查，结果按底层除法语义。 |
| deterministic | 公开文档声明 `aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC)` 不影响本算子确定性表现。 |
| workspace | 调用者提供不足或executor生命周期错误由运行框架报错；总量内部拆分不可从固定仓重建。 |
| 非连续 | optimized guard失败不会越界猜测，转304/305；若Tiling容量/POD检查失败则节点失败，不切AICPU。 |
| 固定仓不可见 | executor汇总/复用offset、Contiguous/ViewCopy内部POD与事件、AscendC primitive微架构实现均标为框架边界，不推测。 |

## 源码索引

### 链路索引

| 层 | 仓库@Commit | 路径/符号 | 用途 |
|---|---|---|---|
| API文档 | ops-math@165d41f | `math/div/docs/aclnnDiv&aclnnInplaceDiv.md` | 原地签名、dtype/平台/错误/确定性 |
| ACLNN Host | ops-math@165d41f | `math/div/op_api/aclnn_div.cpp::{CheckInplace,aclnnInplaceDivGetWorkspaceSize,HandleNotMixDataTypeDiv,aclnnDivGetWorkspaceSize,aclnnInplaceDiv}` | 校验、复用图、生命周期 |
| L0 router | ops-math@165d41f | `math/real_div/op_api/realdiv.cpp::{IsRealDivSupportNonContiguous,RealDiv}` | AIV/AICPU与非连续能力选择 |
| OpDef/Tiling | ops-math@165d41f | `real_div_def.cpp`; `arch35/real_div_tiling_arch35.cpp`; `real_div_binary.json` | 950 dtype、DAG、key、binary |
| AIV Kernel | ops-math@165d41f | `real_div_apt.cpp`; `arch35/real_div_dag.h`; `real_div_struct.h` | entry、计算DAG、key声明 |
| AICPU Kernel | ops-math@165d41f | `op_kernel_aicpu/real_div_aicpu.cpp` | DOUBLE/复数完整body |
| Cast直接依赖 | ops-math@165d41f | `math/cast/op_api/cast.cpp`; `op_host/arch35/cast_tiling.{h,cpp}`; `op_kernel/cast_apt.cpp`; `cast_impl.h`; `op_kernel_aicpu/cast_aicpu.cpp` | 条件dtype转换、POD、SIMT guard、AIV/AICPU body |
| Broadcast POD/Tiling | opbase@d574f55 | `broadcast_base_struct.h`; `broadcast_tiling_base.h`; `broadcast_tiling.h`; `broadcast_tiling*.cpp` | POD、分核、key与非连续selector |
| Broadcast scheduler | opbase@d574f55 | `broadcast_sch*.h`; `noncontiguous/**/broadcast_sch*.h` | 202、1/2、101–109、301–305完整body |
| Vector primitive封装 | opbase@d574f55 | `util/vec.h` | Copy、Cast、DivHighPrecision |
| 构建 | ops-math@165d41f | `math/real_div/CMakeLists.txt` | ascend950/arch35接线及arch32缺口证据 |

### 实现复用检查

| 候选实现 | 是否属于本卡 | 证据与处理 |
|---|---|---|
| `math/real_div` | 是，核心实现 | ACLNN Host明确调用 `l0op::RealDiv(...,MODE_REAL_DIV)`。 |
| `math/cast` | 是，条件直接依赖 | 输入dtype≠P或核心输出dtype≠selfRef时实际创建Cast节点。 |
| opbase atvoss Broadcast | 是，直接模板依赖 | RealDiv Tiling/Kernel直接实例化其POD、selector和scheduler。 |
| Contiguous/ViewCopy/executor | 明确框架边界 | 调用点与顺序可见，内部实现不在固定算子仓；不伪造闭包。 |
| `math/div` 原生 Div Kernel | 否 | ACLNN原地路径不提交该OpType；不与RealDiv合并。 |
| Divs/DivMod/DivMods/非原地Div语义 | 否 | 入口、属性或原地生命周期不同；仅复用非原地函数体不改变本卡逻辑算子边界。 |
