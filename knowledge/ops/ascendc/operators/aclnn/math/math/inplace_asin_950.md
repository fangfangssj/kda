---
type: Operator
title: "InplaceAsin 950 ACLNN 算子知识"
description: "说明 aclnnInplaceAsin 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/docs/aclnnAsin&aclnnInplaceAsin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/aclnn_asin.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/aclnn_asin.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/asin.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/contiguous/op_host/op_api/contiguous.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/view_copy/op_api/view_copy.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_host/asin_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_host/arch35/asin_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/arch35/asin_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/arch35/asin_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/asin_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel_aicpu/asin_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel_aicpu/asin_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_host/config/ascend950/asin_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/aux1.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/node.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
created_at: "2026-08-25T10:08:42Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceAsin 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceAsinGetWorkspaceSize` + `aclnnInplaceAsin`，无版本后缀的最新公开 ACLNN 两段式接口 |
| 计算语义 | 对 `selfRef` 的每个逻辑元素计算 `asin`，并把结果写回同一用户 Tensor 的逻辑视图 |
| 输入输出 | 单个输入/输出别名 `selfRef`；ND、0～8 维，支持空 Tensor 与非连续 Tensor；950 公开 dtype 为 FLOAT、FLOAT16、DOUBLE |
| 执行类型 | R-001/R-002 的核心 `Asin` 节点为 AIV 上的 GM↔UB 调度加 SIMT VF 计算；R-003 为 AICPU；非连续前后处理是 `Contiguous/ViewCopy` 框架边界 |
| Kernel 路线 | R-001 FLOAT 直接 FP32；R-002 FLOAT16 核内 FP16→FP32→FP16；R-003 DOUBLE AICPU fallback |
| 核心资源 | R-001/R-002 使用 GM、一个 `TBuf<VEC_CALC>` 分片池、Mutex pipe 同步和 `asc_vf_call` 预留区；R-003 使用 AICPU 输入/输出 GM 与 CPU 并行任务；Asin Tiling workspace 为 0 |
| 来源闭包 | `source_closed`：公开 Inplace 入口、dtype/空域分发、外层同 dtype Cast 消除、Asin AiCore/AICPU 选择、950 Tiling/POD、AIV+SIMT body、AICPU body及构建接线均已闭合；Contiguous/ViewCopy 与 AscendC primitive 内部按框架边界标注 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `N` 为 `selfRef` 逻辑 shape 的元素数。对每个逻辑索引 `i∈[0,N)`，接口计算 `selfRef[i] = asin(selfRef[i])`。公开文档把 `selfRef` 同时定义为输入和输出；实现并不要求核心 Kernel 的输入、输出 GM 指针物理别名，而是先构造连续输入和内部结果，再由 `ViewCopy` 把结果提交回原用户视图。因此，非连续 Tensor 的“原地”是用户可见 Tensor 对象与存储视图上的原地语义，不等价于 Device body 必须在同一地址覆盖输入。

输入超出 `[-1,1]` 时的 NaN/异常细节由 `asinf` 或 C++ `std::asin` primitive 决定；固定仓没有重写其数值域行为。文档声明该接口默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；逻辑维数不超过 8；可为空、可非连续 | 950：FLOAT、FLOAT16、DOUBLE | 非空指针；同时作为输入与输出。950 不公开 BFLOAT16 Inplace 支持；文档仅把 BFLOAT16 增量支持列给 A3。 |
| `workspaceSize` | 标量 Host 输出 | `uint64_t*` | 由 executor 汇总 Contiguous、内部结果与 ViewCopy 所需空间；空 Tensor 固定返回 0。Asin 自身 Tiling 把节点 workspace 设为 0。 |
| `executor` | Host 输出句柄 | `aclOpExecutor**` | 第一段接口创建并移交；持有已排好的任务、内部 Tensor 关系与 workspace 生命周期。 |
| `workspace` | Device 地址 | 字节缓冲区 | 第二段接口使用，大小必须来自同一 executor 的第一段调用。 |
| `stream` | ACL runtime stream | `aclrtStream` | `CommonOpExecutorRun` 在该 stream 上执行任务。 |

### 入口与版本

公开声明和文档均提供无版本后缀的 `aclnnInplaceAsinGetWorkspaceSize/aclnnInplaceAsin`；固定仓没有同语义更高版本入口。第一段入口先执行 Inplace 专属校验，再复用 `ExecAsinGetWorkspaceSize(selfRef, selfRef, ...)`，但本卡不合并非原地 `aclnnAsin` 的整数输入和独立输出语义。

950 产品支持由公开文档的 Ascend 950PR/950DT 支持表、`Asin` OpDef 的 `ascend950` AICore config、`asin_binary.json` 的 FLOAT16/FLOAT32 binary 以及 `CMakeLists.txt` 的 `SUPPORT_COMPUTE_UNIT "ascend950"` 共同支撑。950 Inplace 的公开 dtype 是 FLOAT、FLOAT16、DOUBLE；`asin_binary.json` 中 BFLOAT16 实例属于底层 Asin 能力，但被 Inplace API 的公开约束和 `GetSelfRefDtypeList` 阻断，不是本卡可达 Route。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序如下：

1. `aclnnInplaceAsinGetWorkspaceSize` 记录 DFX，把 `selfRef` 同时作为输入和输出，调用 `CheckInplaceParams` 检查空指针与 Inplace dtype。
2. `ExecAsinGetWorkspaceSize` 再执行 1-in/1-out dtype 和同 shape 校验；非 ND storage format 只告警。由于 input/out 是同一 Tensor，shape 校验天然一致；公开的 8 维限制由 ACLNN 层约束，固定实现文件中未出现独立 rank 分支。
3. 空 Tensor 在任何 `Contiguous/Asin/ViewCopy` 入队前返回：`workspaceSize=0`，executor 仍移交给调用者。
4. 非空输入调用 `l0op::Contiguous`。连续输入只生成 reshape/view；非连续输入由 Contiguous 框架边界选择优化转换或 `AsStridedToContiguous`，失败返回 `ACLNN_ERR_INNER_NULLPTR`。
5. 950 FLOAT/FLOAT16 被 `isSupportedKernel` 识别为原生 Asin Kernel dtype，`selfCast=selfContiguous`；DOUBLE 保持 DOUBLE，也不产生输入 Cast。
6. `l0op::Asin` 分配同 shape、同 dtype 的内部输出：FLOAT/FLOAT16 入队 AiCore `Asin`，DOUBLE 入队 AICPU `Asin`。
7. `l0op::Cast(asinOutRet, selfRef.dtype)` 的源、目标 dtype 相同，`Cast` 在 Host 直接返回输入 Tensor，不分配 Tensor、不入队 Cast Kernel。
8. `l0op::ViewCopy` 把内部结果提交到 `selfRef`。连续布局可通过 Tensor relation/地址重定向或 TensorMove 完成；非连续布局由 ViewCopy 框架边界恢复原视图。
9. 第一段返回 `uniqueExecutor->GetWorkspaceSize()` 并移交 executor；第二段 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)` 执行。

### 参数、Workspace 与生命周期

`selfRef` 的用户存储由调用者持有。`CREATE_EXECUTOR` 持有连续化结果、`l0op::Asin` 的内部输出以及 ViewCopy 所需元数据/任务；`ReleaseTo(executor)` 后，其生命周期转交给返回的 `aclOpExecutor`。`workspaceSize` 是 executor 对这些内部对象的最终汇总，固定算子源码没有给出一个只依赖 `N` 的闭式公式，因此不得把 Asin Tiling 的 `0` 误写成整个 ACLNN 调用恒为 0。

AiCore Asin 节点的 `TilingForAsin` 把 `workspace[0]` 设为 0；其 GM 输入是连续 Tensor，GM 输出是 executor 内部结果或被 ViewCopy 关系重定向后的用户存储。AICPU 路线也没有算子自有显式 workspace 切片。所有节点由同一 executor/stream 排序；任何 `Contiguous`、Asin、Cast 或 ViewCopy 构造失败都会立即返回，不再移交一个可执行的完整任务图。

### 启动图与失败传播

```text
aclnnInplaceAsinGetWorkspaceSize(selfRef)
  -> [N == 0] 返回 workspaceSize=0，无 Device 节点
  -> Contiguous(selfRef)                         # 框架边界；仅非连续时可能有 Device 节点
  -> Asin(selfContiguous)
       -> R-001/R-002: ADD_TO_LAUNCHER_LIST_AICORE(Asin)
          -> asin(x, y, workspace, tiling)       # 单个 AIV Kernel，内部 asc_vf_call SIMT
       -> R-003: ADD_TO_LAUNCHER_LIST_AICPU(Asin)
          -> AsinCpuKernel::Compute              # 单个 AICPU 节点
  -> Cast(result, selfRef.dtype)                 # 同 dtype Host no-op，无节点
  -> ViewCopy(result, selfRef)                   # 框架边界；提交到原用户视图
  -> executor.GetWorkspaceSize()

aclnnInplaceAsin(workspace, executor, stream)
  -> CommonOpExecutorRun(..., stream)
```

核心 Asin 节点只有一个 producer→consumer 边：连续输入 Tensor→Asin→内部结果 Tensor。之后结果由 ViewCopy 消费；交接介质是 executor 管理的 GM Tensor，shape/dtype 与 `selfRef` 相同。框架以同一 executor 的任务顺序和 `stream` 保证可见性，producer 入队失败会使第一段返回空指针错误并阻断 ViewCopy。固定仓允许把 Contiguous/ViewCopy 的多实现网络作为框架边界；其具体 workspace offset、内部 Kernel 数与地址重定向选择依输入 stride 和平台能力而变，不伪造固定 slice。

## Route 与参数设计

### 分核与资源计算

R-001/R-002 先把输出 storage shape 各维相乘为 `N=dim0`；API 已在空 Tensor 处提前返回，Tiling 遇到任一 0 维会失败。令 `Cplat=GetCoreNumAiv()`、`U` 为平台 UB 字节数、`E=155648=122880+32768` 字节预留：

- R-001 FLOAT：`minBits=maxBits=32`，DAG Level-2 buffer 数 `B=4`；`coreNum=min(ceil(N×32/32768), Cplat)=min(ceil(N/1024),Cplat)`；`ubFormer=align_down(floor((U-E)/16),64)` 元素。
- R-002 FLOAT16：DAG 中同时存在 16/32 位数据，`minBits=16,maxBits=32`，`B=6`；`coreNum=min(ceil(N×16/32768),Cplat)=min(ceil(N/2048),Cplat)`；`ubFormer=align_down(floor((U-E)/24),128)` 元素。

两者继续计算 `blockFormer=align_up(ceil(N/coreNum),512)` 元素、`blockNum=ceil(N/blockFormer)`、`blockTail=N-(blockNum-1)×blockFormer`；对 former/tail block 分别求 `ceil(blockElements/ubFormer)` 和最后 tile 有效元素数。Host 设置 `blockDim=blockNum`，并把 Kernel local memory 设为 `uint32_t(U-32768)`。若 `Cplat=0`、`U<E`、`ubFormer=0`、shape/dtype 不合法或 `DoTiling` 失败，则没有 fallback 到另一 AiCore Route。

R-003 DOUBLE 不进入 AscendC Tiling。AICPU 取 `cores=CpuKernelUtils::GetCPUNum(ctx)`，对正元素数计算 `workers=min(max(1,cores-2),N)` 和 `per_unit_size=N/workers`，再由 `ParallelFor` 切分 `[0,N)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FLOAT 直接 FP32 | 950 FLOAT32 binary；`AsinOpDirect<float>::OpDag` | Inplace dtype 校验通过且 `selfRef.dtype==DT_FLOAT`；`isSupportedKernel` 为真，`IsAiCoreSupport` 在 RegBase 上为真 | dtype 为 FLOAT16 时 R-002；DOUBLE 时 R-003；其他 dtype API 失败 |
| 2 | R-002 FLOAT16 核内升降精度 | 950 FLOAT16 binary；`AsinOpWithCast<half>::OpDag` | `selfRef.dtype==DT_FLOAT16`；950 原生 Kernel 与 AiCore 两层谓词均为真 | DOUBLE 时 R-003；其他 dtype API 失败 |
| 3 | R-003 DOUBLE AICPU | `AsinCpuKernel::Compute` | `selfRef.dtype==DT_DOUBLE`；950 `isSupportedKernel(DOUBLE)` 为假但输入 Cast 目标仍为 DOUBLE，随后 RegBase `IsAiCoreSupport(DOUBLE)` 为假 | 无下一 Route；AICPU dispatch/校验失败即接口执行失败 |

```text
if selfRef == nullptr or dtype not in A5 Inplace list {FLOAT, FLOAT16, DOUBLE}:
    fail ACLNN parameter check
if N == 0:
    return empty executor, workspaceSize = 0
contiguous = Contiguous(selfRef)
if dtype == FLOAT:
    enqueue AiCore AsinOpDirect<float>                       # R-001
else if dtype == FLOAT16:
    enqueue AiCore AsinOpWithCast<half>                      # R-002
else:  # DOUBLE
    keep DOUBLE; enqueue AICPU AsinCpuKernel::Compute        # R-003
Cast(asin_result, selfRef.dtype) returns asin_result          # same dtype
ViewCopy(asin_result, selfRef)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FLOAT 直接 FP32 | FLOAT32 950 binary，simplified key mode 0 | 950 RegBase；非空；FLOAT；输入/输出同 shape | 单核心语义节点：AIV `asin`，节点内 CopyIn→SIMT VF→CopyOut | `ElementwiseSch<0, AsinOpDirect<float>::OpDag>`；`AsinCustom<float>` | 4 个 `ubFormer×4` B pool slice；MTE2/V/MTE3 Mutex；SIMT 1024 threads | former/tail block 和尾 tile；Tiling 失败无 fallback |
| R-002 FLOAT16 核内升降精度 | FLOAT16 950 binary，simplified key mode 0 | 950 RegBase；非空；FLOAT16；输入/输出同 shape | 单核心语义节点：AIV CopyIn/Cast + SIMT VF + Cast/CopyOut | `ElementwiseSch<0, AsinOpWithCast<half>::OpDag>`；`Cast<float,half,0>`→`AsinCustom<float>`→`Cast<half,float,1>` | 6 个 `ubFormer×4` B pool slice；两个 FP32 temp；MTE2/V/MTE3 Mutex | 有效 `count=tailNum` 传给每个节点；Tiling 失败无 fallback |
| R-003 DOUBLE AICPU | Host dtype selector，无 TilingKey | 950；非空；DOUBLE；AiCore support 谓词为假 | 单 AICPU `Asin` 节点 | `AsinCpuKernel::Compute`→`ComputeAsinDispatch`→`ComputeAsin<double>` | 无 AIV UB/POD；GM 输入/输出与 CPU worker 分片 | `ParallelFor` 的 `[begin,end)` 天然尾分片；不支持 dtype/校验失败即错误 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域提前返回 | `self->IsEmpty()` | Elewise Tiling 明确拒绝 0 维；避免生成 `loopNum-1` 的无效循环 | 全部 | 成功返回 0 workspace，无 Device 节点 | `math/asin/op_api/aclnn_asin.cpp::ExecAsinGetWorkspaceSize` |
| 950 原生 dtype | `DAV_3510 && dtype∈{BF16,F16,F32}` | 只有这些 dtype 有 `asin_apt` binary/模板实例 | R-001/R-002；BF16 被更早 Inplace API guard 阻断 | DOUBLE 保持 dtype 并进入 R-003 | `math/asin/op_api/aclnn_asin.cpp::isSupportedKernel` |
| AiCore/AICPU 分发 | `IsRegBase() && dtype∈{F32,F16,BF16}` | 950 Asin OpDef/binary 不含 DOUBLE | R-001/R-002/R-003 | 不支持 AiCore 时 AICPU | `math/asin/op_api/asin.cpp::IsAiCoreSupport/Asin` |
| UB 预留 | `U>=E` 且 `ubFormer>0` | `122880` B 供 Ascend API/VF，`32768` B 为 DCache；余量再按 DAG buffer 数切 tile | R-001/R-002 | `DoTiling` 失败 | `math/asin/op_host/arch35/asin_tiling_arch35.cpp::RunTiling`；`ElewiseTiling` |
| repeat 对齐 | `ubFormer` 向下对齐 `256×8/minBits` 元素 | 保证通用 Elewise tile 满足 256 B repeat 粒度 | R-001/R-002 | 对齐后为 0 则 Tiling 失败 | `opbase/.../elewise_tiling.cpp::GetMaxElemNum` |
| 同 dtype外层 Cast | `asinOutRet.dtype==selfRef.dtype` | 避免额外 Cast Tensor 和 Kernel | 全部 | 本算子 Inplace 路线恒成立；若复用代码改变输出 dtype 则进入 Cast AiCore/AICPU | `math/cast/op_api/cast.cpp::Cast` |
| AICPU 空域安全 | ACLNN 在入队前已处理 `N==0` | `per_unit_size` 的分母以 `N` 为上界；必须保证正数 | R-003 | 空 Tensor 走 Host 成功终点 | `math/asin/op_kernel_aicpu/asin_aicpu.cpp::ComputeAsinKernel` |

### POD/ABI

`AsinTilingData` 只包含一个自然布局的 `EleBaseTilingData baseTiling`。依据字段类型与声明顺序，其 `sizeof=80`、`alignof=8`；下表 offset 为标准 C++ 自然对齐结果。R-003 不消费该 POD。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AsinTilingData.baseTiling` | `dim0` | `int64_t`/64 | offset 0，size 8 | 元素 | `ElewiseTiling`: shape 各维乘积 | Scheduler 未读取 | 非空；乘积溢出检查固定仓不可见 |
| `AsinTilingData.baseTiling` | `coreNum` | `int32_t`/32 | offset 8，size 4 | AIV core | `AdaptEleBaseTilingData`: `static_cast<int32_t>(coreNum)` | Scheduler 未读取；launch 使用独立 `blockNum` | 平台 core 数先限幅；窄化后只存 POD |
| `AsinTilingData.baseTiling` | `ubFormer` | `int32_t`/32 | offset 12，size 4 | 元素/tile | `AdaptEleBaseTilingData`: `maxElemNum` 窄化写入 | `ElementwiseSch::Init/Process`：buffer、`blockLen`、整 tile count | 必须非 0；Host 的 int64→int32 上界检查固定仓不可见 |
| `AsinTilingData.baseTiling` | `blockFormer` | `int64_t`/64 | offset 16，size 8 | 元素/core block | `ElewiseTiling`: `align_up(ceil(N/coreNum),512)` | `CalcBlockOffset`: `blockFormer×blockIdx×sizeof(T)` | 最后一核有效量可小于该值 |
| `AsinTilingData.baseTiling` | `blockNum` | `int64_t`/64 | offset 24，size 8 | block/AIV core | `ElewiseTiling`: `ceil(N/blockFormer)`；`SetBlockDim` 同值 | `Process` 判断 `blockIdx==blockNum-1` | launch blockDim 与字段必须一致 |
| `AsinTilingData.baseTiling` | `ubLoopOfFormerBlock` | `int64_t`/64 | offset 32，size 8 | tile loop | `ceil(blockFormer/ubFormer)` | 非尾 block 的 `loopNum` | 至少 1 |
| `AsinTilingData.baseTiling` | `ubLoopOfTailBlock` | `int64_t`/64 | offset 40，size 8 | tile loop | `ceil(blockTail/ubFormer)` | 最后 block 的 `loopNum` | 至少 1 |
| `AsinTilingData.baseTiling` | `ubTailOfFormerBlock` | `int64_t`/64 | offset 48，size 8 | 有效元素 | `blockFormer-(loops-1)×ubFormer` | 非尾 block 最后一次 `Run` 的 `tileLength` | `1..ubFormer` |
| `AsinTilingData.baseTiling` | `ubTailOfTailBlock` | `int64_t`/64 | offset 56，size 8 | 有效元素 | `blockTail-(loops-1)×ubFormer` | 最后 block 最后一次 `Run` 的 `tileLength` | `1..ubFormer` |
| `AsinTilingData.baseTiling` | `elemNum` | `int64_t`/64 | offset 64，size 8 | 元素 | `ElewiseTiling`: `maxElemNum` | 当前 Scheduler 未读取 | 等于 `ubFormer` 的 int64 producer 值 |
| `AsinTilingData.baseTiling` | `scheMode` | `uint64_t`/64 | offset 72，size 8 | 枚举值 | `GetScheMode()` 返回 1 | entry 实例固定 `ElementwiseSch<0,...>`，当前 body 未读取字段 | inactive；不得据此臆造第二 Route |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FLOAT 直接 FP32 | API/AiCore dtype selector；FLOAT32 binary | `aclnn_asin.cpp::isSupportedKernel`；`asin.cpp::Asin`；`asin_tiling_arch35.cpp::RunTiling` | `AsinTilingData`/`EleBaseTilingData` | `asin_apt.cpp::asin` FLOAT constexpr branch | `AsinOpDirect<float>`；`ElementwiseSch`；`AsinCustom<float>/AsinSimtCompute<float>` | `asin_def.cpp`；`asin_binary.json`；`math/asin/CMakeLists.txt` |
| R-002 FLOAT16 核内升降精度 | API/AiCore dtype selector；FLOAT16 binary | 同上，Tiling 选择 `AsinOpWithCast<half>` | 同一 POD，buffer 数与 minBits 不同 | `asin_apt.cpp::asin` half constexpr branch | `AsinOpWithCast<half>`；`Vec::Cast`；`AsinCustom<float>`；`ElementwiseSch` | 同上 |
| R-003 DOUBLE AICPU | `IsAiCoreSupport(DOUBLE)==false` | `asin.cpp::Asin/AsinAiCpu` | AICPU task context；无 Tiling POD | `ADD_TO_LAUNCHER_LIST_AICPU(Asin)`→`AsinCpuKernel::Compute` | `CheckAsin`；`ComputeAsinDispatch`；`ComputeAsin<double>`；`ComputeAsinKernel<double>` | `asin_aicpu_def.cpp`；通用 CMake `op_kernel_aicpu` glob/link 接线 |

## Kernel 实现

### 公共执行流程

非空路线共享 `Contiguous→Asin→同 dtype Cast no-op→ViewCopy`。核心 Asin 的输出 dtype 始终等于输入 dtype。R-001/R-002 的 entry 解析同一 POD、构造 `TPipe`，按 `DTYPE_X` 编译期分支实例化唯一 DAG scheduler；每个 AIV block 绑定自己的 `blockFormer×blockIdx` GM 基址，循环处理 `ubFormer` tile 并只提交尾 tile 的有效 `tailNum`。R-003 不启动 AIV，直接在 AICPU 上对 GM 元素区间并行调用 `std::asin`。

Contiguous/ViewCopy 的内部实现依 stride/offset 选择多种框架节点，属于 Skill 允许的框架边界；它们与核心 Asin 节点通过 executor Tensor 关系交接，并由同一 stream 顺序执行。

### Route R-001：FLOAT 直接 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/asin/op_api/aclnn_asin.cpp` | `GetSelfRefDtypeList/CheckInplaceParams` | 950 公开可达 FLOAT | 进入复用执行流 |
| API | 同上 | `isSupportedKernel` | `DAV_3510 && castDtype==DT_FLOAT` | 不调用输入 Cast |
| Host selector | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport/Asin` | RegBase 且 FLOAT 在 AiCore list | `ADD_TO_LAUNCHER_LIST_AICORE(Asin)` |
| Tiling | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `RunTiling` | `outputDtype==DT_FLOAT` | `DoTiling<AsinOpDirect<float>::OpDag>` |
| Router | `math/asin/op_kernel/asin_apt.cpp` | `asin` | `std::is_same<DTYPE_X,float>` | `ElementwiseSch<0,AsinOpDirect<float>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(Asin, x, y)
  -> TilingForAsin -> AsinTiling::RunTiling -> ElewiseBaseTiling::DoTiling<AsinOpDirect<float>::OpDag>
  -> asin(x, y, workspace, tiling)
     -> ElementwiseSch<0, AsinOpDirect<float>::OpDag>::Init(x, y)
     -> ElementwiseSch::Process
        -> CopyIn<float>
        -> RunNormalOp<AsinCustom<float>>
           -> asc_vf_call<AsinSimtCompute<float>>(dim3(1024), srcUB, dstUB, count)
              -> thread i: dstUB[i] = asinf(srcUB[i])
        -> CopyOut<float>
  -> executor GM result edge
  -> ViewCopy(result, selfRef)  # framework ordering/commit
```

##### 源码等价伪码

```text
# 源码等价伪码
coreBase = blockFormer * blockIdx
loopNum = (blockIdx == blockNum-1) ? ubLoopOfTailBlock : ubLoopOfFormerBlock
tailNum = (blockIdx == blockNum-1) ? ubTailOfTailBlock : ubTailOfFormerBlock
offset = 0
for tile = 0 .. loopNum-1:
    count = (tile == loopNum-1) ? tailNum : ubFormer
    pingPong = tile & 1
    MTE2-lock(inputBuffer[pingPong])
    DataCopyPad(inputBuffer, x[coreBase+offset], blockCount=1,
                blockLen=count*sizeof(float), srcStride=0, dstStride=0, pad=false)
    MTE2-unlock(inputBuffer)
    V-lock(inputBuffer); V-lock(outputBuffer)
    asc_vf_call<AsinSimtCompute<float>>(1024 threads, inputBuffer, outputBuffer, count)
      for i = threadIdx.x; i < count; i += blockDim.x: outputBuffer[i] = asinf(inputBuffer[i])
    V-unlock consumed/produced buffers
    MTE3-lock(outputBuffer)
    DataCopyPad(y[coreBase+offset], outputBuffer, blockCount=1,
                blockLen=count*sizeof(float), srcStride=0, dstStride=0)
    MTE3-unlock(outputBuffer)
    offset += count
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `(blockFormer×blockIdx+tileOffset)×4` B | `count×4` B 有效 | FLOAT/flat ND | Contiguous 结果 | `Vec::CopyIn<float>` | executor；覆盖整个 Asin 节点 | 按有效 count 搬运，无 padding 写入 |
| UB input ping/pong | pool 的 MTE2 两个 slice | 每 slice `ubFormer×4` B | FLOAT | `DataCopyPad` | `AsinSimtCompute<float>` | 当前偶/奇 tile | 两片交替；Mutex MTE2→V |
| UB output ping/pong | pool 的 MTE3 两个 slice | 每 slice `ubFormer×4` B | FLOAT | SIMT VF | `DataCopyPad` to GM | 当前偶/奇 tile | 结果直接进入输出 slice，无 temp materialization |
| SIMT registers/primitive area | VF thread 私有；具体 offset 固定仓不可见 | Host 额外预留 `122880` B；寄存器字节不可见 | FLOAT | `asinf` | UB output | `asc_vf_call` 调用期 | 1024 threads；primitive 内布局不可见 |
| GM result | 同 input 逻辑 flatten offset | `count×4` B | FLOAT/flat ND | CopyOut | ViewCopy | executor 内部结果，直到提交到 `selfRef` | 连续用户 Tensor 可由框架地址关系消除物理 copy |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inPing["MTE2 input ping · ubFormer×4 B"]:1
  outPing["MTE3 output ping · ubFormer×4 B"]:1
  inPong["MTE2 input pong · ubFormer×4 B"]:1
  outPong["MTE3 output pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
OpDag::BufferNum = 0 temp + 1 input×2 + 1 output×2 = 4
UB_pool = 4 × ubFormer × maxDtypeBytes(4) = 16×ubFormer B
reserved_for_ascend_api_and_dcache = 122880 + 32768 = 155648 B
UB_pool <= U - 155648
localMemorySize = U - 32768
dynamic_ub = N/A；VF primitive 使用上述 122880 B 显式预留，内部切片固定仓不可见
workspace_total = Asin 节点 0 B；ACLNN 总 workspace 由 executor 的框架前后处理决定
=> ubFormer = align_down(floor((U-155648)/16), 64)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input ping/pong | 当前 tile CopyIn | 当前 tile SIMT VF | 同奇偶性的下下个 tile | `Get/ReleaseTensor<VECIN>` 与 `<VECCALC>` 映射为 MTE2/V Mutex | `pingPong=tile&1` 且 V 消费后才可重写 |
| output ping/pong | 当前 tile SIMT VF | 当前 tile CopyOut | 同奇偶性的下下个 tile | V/MTE3 Mutex | MTE3 完成 GM 提交后才可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + MTE2 Mutex | GM input | SIMT 前 V lock | input slice | `blockLen=count×4` B；pad disabled |
| UB compute | `asc_vf_call` + V Mutex | input slice | output CopyOut | input/output slice | 全程 FLOAT；1024 threads，`i<count` 保护任意尾长 |
| UB→GM | `DataCopyPad` + MTE3 Mutex | SIMT output | GM result/ViewCopy | output slice | 只写 `count×4` B，不提交无效 lane |
| 跨节点 | executor/stream ordering | Asin | ViewCopy | GM result Tensor | framework 保持 dtype/shape；失败阻断后续节点 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `AsinTiling::RunTiling/SetTilingData` | FLOAT 选择 direct DAG；写 POD、blockDim、0 workspace、local memory |
| Kernel entry/router | `math/asin/op_kernel/asin_apt.cpp` | `asin` FLOAT branch | 固定 AIV_ONLY，构造 `ElementwiseSch<0,AsinOpDirect<float>::OpDag>` |
| Device body/helper | `math/asin/op_kernel/arch35/asin_dag.h` | `AsinOpDirect<float>/AsinCustom/AsinSimtCompute` | CopyIn→1024-thread `asinf`→CopyOut |
| Scheduler dependency | `opbase/.../elewise_sch.h` | `ElementwiseSch::Init/Process/Run` | pool 字节数、分核地址、ping/pong、逐 DAG 节点同步与 tail |

### Route R-002：FLOAT16 核内升降精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/asin/op_api/aclnn_asin.cpp` | `CheckInplaceParams/isSupportedKernel` | FLOAT16 且 DAV_3510 | 不产生输入 Cast，进入核心 Asin |
| Host selector | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport/Asin` | RegBase list 包含 FLOAT16 | AiCore `Asin` |
| Tiling | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `RunTiling` | `outputDtype==DT_FLOAT16` | `DoTiling<AsinOpWithCast<half>::OpDag>` |
| Router | `math/asin/op_kernel/asin_apt.cpp` | `asin` | `DTYPE_X==half` | half cast DAG |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(Asin, x, y)
  -> TilingForAsin -> DoTiling<AsinOpWithCast<half>::OpDag>
  -> asin(x, y, workspace, tiling)
     -> ElementwiseSch<0, AsinOpWithCast<half>::OpDag>::Init/Process
        -> CopyIn<half>
        -> Vec::Cast<float, half, 0>
        -> AsinCustom<float>
           -> asc_vf_call<AsinSimtCompute<float>>(1024 threads, srcFP32, dstFP32, count)
        -> Vec::Cast<half, float, 1>
        -> CopyOut<half>
  -> ViewCopy(result, selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码；分核/loop/tail 与 R-001 相同
for each tile(count, pingPong):
    DataCopyPad(halfInputSlice, x[coreBase+offset], 1, count*2 B, 0, 0, pad=false)
    Cast(fp32Temp0, halfInputSlice, static_cast<RoundMode>(0), count)
    asc_vf_call<AsinSimtCompute<float>>(1024 threads, fp32Temp0, fp32Temp1, count)
      for i = threadIdx.x; i < count; i += blockDim.x: fp32Temp1[i] = asinf(fp32Temp0[i])
    Cast(halfOutputSlice, fp32Temp1, static_cast<RoundMode>(1), count)
    DataCopyPad(y[coreBase+offset], halfOutputSlice, 1, count*2 B, 0, 0)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `(blockFormer×blockIdx+tileOffset)×2` B | `count×2` B | FLOAT16/flat ND | Contiguous / CopyOut | CopyIn / ViewCopy | executor | 有效 count 搬运 |
| UB input ping/pong | MTE2 两 slice | 每 slice `ubFormer×4` B 物理容量，逻辑 `count` half | half | CopyIn | Cast0 | 当前 tile | pool 按最大 dtype 4 B/元素统一切片 |
| FP32 temp0 | temp slice | `ubFormer×4` B | FLOAT | Cast0 | SIMT | Cast0 后至 SIMT 消费 | Level-2 DAG 临时区 |
| FP32 temp1 | temp slice | `ubFormer×4` B | FLOAT | SIMT | Cast1 | SIMT 后至 Cast1 消费 | 与 temp0 分离，允许 src/dst 不别名 |
| UB output ping/pong | MTE3 两 slice | 每 slice `ubFormer×4` B 物理容量，逻辑 `count` half | half | Cast1 | CopyOut | 当前 tile | 结果直接写 output buffer |
| SIMT primitive area | 固定仓不可见 | 显式预留 122880 B | FLOAT | `asinf` | temp1 | VF 调用期 | 1024 threads |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inPing["MTE2 in ping · ubFormer×4 B"]:1
  outPing["MTE3 out ping · ubFormer×4 B"]:1
  fp32a["FP32 temp0 · ubFormer×4 B"]:1
  fp32b["FP32 temp1 · ubFormer×4 B"]:1
  inPong["MTE2 in pong · ubFormer×4 B"]:1
  outPong["MTE3 out pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
OpDag::BufferNum = 2 FP32 temp + 1 input×2 + 1 output×2 = 6
UB_pool = 6 × ubFormer × maxDtypeBytes(4) = 24×ubFormer B
UB_pool <= U - 155648
localMemorySize = U - 32768
dynamic_ub = N/A；asc_vf_call 内部使用方式固定仓不可见
workspace_total = Asin 节点 0 B；ACLNN 总量由 executor 汇总
=> ubFormer = align_down(floor((U-155648)/24), 128)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output ping-pong | CopyIn / Cast1 | Cast0 / CopyOut | 同奇偶性的下下个 tile | MTE2/V/MTE3 Mutex | 各 pipe 消费完成后覆盖 |
| FP32 temp0 | Cast0 | SIMT VF | 下一 tile 的 Cast0 | `GetTensor/ReleaseTensor<VECCALC>` | SIMT 已读完 |
| FP32 temp1 | SIMT VF | Cast1 | 下一 tile 的 SIMT | V Mutex | Cast1 已读完 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | `DataCopyPad` + MTE2 Mutex | GM half | Cast0 | input slice | `blockLen=count×2` B |
| 升精度 | `AscendC::Cast` + V Mutex | input half | SIMT | temp0 | 模板 mode 常量 0；枚举名称在固定依赖源码不可见 |
| 反正弦 | `asc_vf_call` | temp0 | Cast1 | temp0/temp1 | FLOAT `asinf`；线程条件 `i<count` |
| 降精度 | `AscendC::Cast` + V Mutex | temp1 | CopyOut | output half | 模板 mode 常量 1；只处理 count 个有效元素 |
| CopyOut | `DataCopyPad` + MTE3 Mutex | output half | GM/ViewCopy | output slice | `blockLen=count×2` B，不写无效 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `RunTiling<AsinOpWithCast<half>::OpDag>` | 以 mixed 16/32-bit DAG 生产 tile/core/POD |
| Kernel entry/router | `math/asin/op_kernel/asin_apt.cpp` | `asin` half branch | 单 AIV Kernel 选择 cast DAG |
| Device body/helper | `math/asin/op_kernel/arch35/asin_dag.h` | `AsinOpWithCast<half>` | CopyIn→Cast0→FLOAT SIMT Asin→Cast1→CopyOut |
| Vector dependency | `opbase/.../util/vec.h` | `Vec::Cast/CopyIn/CopyOut` | count、round mode 与 DataCopyPad 参数直达 primitive |

### Route R-003：DOUBLE AICPU

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/asin/op_api/aclnn_asin.cpp` | `GetSelfRefDtypeList/CheckInplaceParams` | 950 基础公开 list 包含 DOUBLE | 通过 Inplace 校验 |
| API | 同上 | `isSupportedKernel` | DAV_3510 但 DOUBLE 不在原生 list | 进入非原生分支；target dtype 仍为 DOUBLE，无输入 Cast |
| Host selector | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport` | RegBase AiCore list 不含 DOUBLE | `AsinAiCpu` |
| AICPU router | `math/asin/op_kernel_aicpu/asin_aicpu.cpp` | `ComputeAsinDispatch` | `input_type==DT_DOUBLE` | `ComputeAsin<double>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Asin(double input, executor)
  -> AsinAiCpu
     -> ADD_TO_LAUNCHER_LIST_AICPU(Asin, input, output)
        -> AsinCpuKernel::Compute
           -> CheckAsin -> NormalCheck + ExtraCheckAsin
           -> ComputeAsinDispatch(DT_DOUBLE)
           -> ComputeAsin<double>
           -> ComputeAsinKernel<double>
           -> CpuKernelUtils::ParallelFor
              -> std::transform(..., ScalarAsin<double>)
                 -> std::asin(x)
  -> GM result
  -> ViewCopy(result, selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
require one input, one output, non-null data
require input.dtype == output.dtype == DOUBLE
require input.dataSize == output.dataSize
if dataSize == 0: return OK  # normally intercepted earlier by ACLNN
N = input.NumElements()
cores = GetCPUNum(ctx)
workers = min(max(1, int64(cores)-2), N)
perUnit = N / workers
ParallelFor(total=N, per_unit_size=perUnit):
    for i in [begin,end): output[i] = std::asin(input[i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| AICPU GM input | `input + begin×8` B | `(end-begin)×8` B | DOUBLE/contiguous | Contiguous | CPU worker | executor/任务期 | worker 区间不重叠 |
| AICPU GM output | `output + begin×8` B | `(end-begin)×8` B | DOUBLE/contiguous | CPU worker | ViewCopy | executor，直至提交 | 每 worker 独占区间，无 atomic |
| CPU local/register | worker 私有 | 固定仓不可见 | DOUBLE | `std::asin` | `std::transform` output | 单元素/worker 调用期 | 无共享归约 |

##### 分层内存布局

N/A：本 Route 不使用 AIV UB、`TQue` 或 `TBuf`；实际替代路径是 AICPU 对连续 GM 输入/输出做互不重叠的 `[begin,end)` 分片。

##### 字节账与复用顺序

```text
UB_total = N/A（AICPU）
dynamic_ub = N/A
workspace_total = Asin AICPU body 无显式 workspace；ACLNN executor 仍可能为 Contiguous/ViewCopy/内部结果分配 workspace
CPU partition bytes per worker = (end-begin)×sizeof(double)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM output worker slice | 当前 CPU worker | ParallelFor 完成后的 ViewCopy | executor 结束后用户可覆盖 | `ParallelFor` 返回即所有分片完成；executor 顺序启动 ViewCopy | worker `[begin,end)` 不重叠且 producer 成功 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU 分片 | `CpuKernelUtils::ParallelFor` | Host AICPU task | 函数返回 | GM output 全区间 | DOUBLE `std::asin`；最后 worker 自然覆盖余数 |
| 元素计算 | `std::transform` | GM input slice | GM output slice | 每 worker 私有区间 | 不做升/降精度；空域在 dispatch 返回 OK |
| 跨节点 | executor/stream | AICPU Asin | ViewCopy | GM result Tensor | AICPU 非 OK 状态阻断成功完成 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport/AsinAiCpu` | DOUBLE 在 RegBase 上回退 AICPU，无 AscendC Tiling |
| Kernel entry/router | `math/asin/op_kernel_aicpu/asin_aicpu.cpp` | `AsinCpuKernel::Compute/ComputeAsinDispatch` | 校验后按 DT_DOUBLE 选择模板 |
| Device body/helper | 同上 | `ComputeAsinKernel<double>/ScalarAsin<double>` | CPU 核数留 2、区间并行、`std::asin` 逐元素写出 |

## 算子特定优化

### Trick T-001：950 原生 dtype 跳过 ACLNN 外层输入 Cast

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：`GetCurNpuArch()==DAV_3510` 且连续化后 dtype 为 FLOAT 或 FLOAT16。
- Golden 来源：[`math/asin/op_api/aclnn_asin.cpp:139-160`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_api/aclnn_asin.cpp#L139-L160)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // 调用cast算子将不支持的类型转化为float / 将bfloat16类型转化为float
    auto castDtype = selfContiguous->GetDataType();
    const aclTensor* selfCast = nullptr;

    // 判断条件：昇腾950版本且数据类型为Kernel原生支持的类型，则跳过Cast
    if (isSupportedKernel(castDtype)) {
        // 950新增逻辑：kernel支持的数据类型零拷贝复用输入Tensor，避免冗余Cast开销
        selfCast = selfContiguous;
    } else {
        // 原有逻辑优化，目标类型与原类型相同时，跳过Cast算子调用
        DataType targetDtype = castDtype;
```

#### 机制与设计原因

FLOAT/FLOAT16 不再在 ACLNN 任务图中生成独立 Cast Tensor/Kernel；`selfCast` 直接别名连续输入。FLOAT16 所需的 FP32 精度提升被移入单个 Asin AIV Kernel 的 UB DAG，因而中间 FP32 数据不落 GM。该机制改变了任务数和 GM 中间物化，不代表未经真机验证的性能百分比。

#### 迁移边界

- 适用：950 上底层 Kernel 自身能接收 API dtype 的逐元素超越函数。
- 前提：Host 原生 dtype 谓词、OpDef/binary 和 Kernel 模板三者同时存在；Inplace API 还必须公开该 dtype。
- 失效：DOUBLE 不满足原生谓词，进入 R-003 AICPU；BFLOAT16 虽有底层 binary，但本 950 Inplace API guard 不公开，不能迁移为可达 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能仅看到 950 BFLOAT16 binary 就放宽 Inplace API dtype；公开文档和 `GetSelfRefDtypeList` 没有给 950 开放 BFLOAT16。固定仓未提供其它可验证反例。

### Trick T-002：FP16 升精度、SIMT 计算与降精度在单 Kernel UB DAG 内融合

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：950 FLOAT16 命中 `AsinOpWithCast<half>::OpDag`。
- Golden 来源：[`math/asin/op_kernel/arch35/asin_dag.h:75-85`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/asin/op_kernel/arch35/asin_dag.h#L75-L85)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
struct AsinOpWithCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, AsinDag::CastModeToFp32>, OpCopyIn>;
    using OpAsin = Bind<AsinDag::AsinCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, AsinDag::CastModeToBf16>, OpAsin>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
```

#### 机制与设计原因

输入只搬入一次；FP16→FP32 Cast、FP32 `asinf` 和 FP32→FP16 Cast 按 DAG 顺序在同一 UB pool 中完成，只有最终 FP16 回写 GM。`MemLevel::LEVEL_2` 为 input/output 配 ping/pong，并只保留两个同时存活的 FP32 temp；最终 Cast 直接写 MTE3 output slice，避免额外结果 buffer。每个 primitive 的 `count` 都来自当前 tile 的 `ubFormer` 或尾 `tailNum`。

#### 迁移边界

- 适用：输入/输出同为 16 位、核心超越函数要求 FP32 的逐元素链。
- 前提：UB 至少容纳 `6×ubFormer×4 + 155648` B，Cast mode 常量与目标 dtype 保持原 ABI，SIMT thread guard 使用同一 `count`。
- 失效：FLOAT 使用 R-001 的 direct DAG；DOUBLE 使用 R-003，不能套用 AIV UB 设计。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把两次 Cast 拆成 ACLNN 外部 Kernel 会物化 FP32 GM 中间量并改变 executor workspace/节点图；固定仓没有提供该拆分实现的数值或性能对照。

## 模板与实现对比

| 对比维度 | R-001 FLOAT 直接 FP32 | R-002 FLOAT16 核内升降精度 | R-003 DOUBLE AICPU | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 950 FLOAT，原生+AiCore | 950 FLOAT16，原生+AiCore | 950 DOUBLE，AiCore 不支持 | 两层 Host dtype 谓词 |
| 模板与 dtype | `AsinOpDirect<float>` | `AsinOpWithCast<half>`，计算为 FLOAT | `ComputeAsinKernel<double>` | binary/OpDef 与 AICPU dispatch |
| 循环与分核 | AIV block + tile；SIMT 1024 threads | 同左，core 粒度按 16-bit min dtype | CPU `ParallelFor` | AIV/CPU 资源模型不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单 TBuf pool 4 slices；VF 寄存器不可见 | 单 TBuf pool 6 slices；2 FP32 temp | 无 AIV UB；CPU local 不可见 | DAG 最大存活节点与 dtype 宽度 |
| 内存总占用与复用 | `16×ubFormer` B + 155648 B 预留 | `24×ubFormer` B + 155648 B 预留 | worker 直接读写 GM 区间 | direct 输出 vs mixed-precision temp |
| 搬运/Vector/SIMT/NDDMA/通信 API | 2 次 DataCopyPad + `asc_vf_call` | 2 次 DataCopyPad + 2 Cast + `asc_vf_call` | `std::transform/std::asin` | dtype 支持能力 |
| 同步、精度与 tail | MTE2/V/MTE3 Mutex；FP32；`i<count` | 同步相同；FP16→FP32→FP16；`i<count` | ParallelFor 完成；DOUBLE；区间尾分片 | scheduler/CPU runtime |
| fallback/失败边界 | Tiling/launch 失败 | UB/Tiling/launch 失败 | AICPU 校验/dispatch/ParallelFor 失败 | 无跨 Route runtime fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档支持，但固定仓缺少对应 AiCore Tiling/Kernel body | 公开文档支持且 Inplace 额外支持 BFLOAT16，但固定仓缺少对应 AiCore Tiling/Kernel body | 公开支持；本卡闭合 FLOAT/FLOAT16 AiCore 与 DOUBLE AICPU |
| Host/Launcher | ACLNN Host 分发可见；对应平台 AiCore 实现不可见 | ACLNN Host 分发可见；FLOAT/FLOAT16/BFLOAT16 AiCore body 不可见 | DAV_3510 原生 bypass；RegBase AiCore/AICPU 分发完整 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | `asin_apt` AIV+SIMT、arch35 Elewise Tiling；DOUBLE AICPU |

## 限制与失败边界

- 950 Inplace 只发布 FLOAT、FLOAT16、DOUBLE；不得把底层 `asin_binary.json` 的 BFLOAT16 实例外推为公开 Inplace 支持。
- 空 Tensor 由 ACLNN Host 成功短路；直接绕过 ACLNN 调用 Tiling 时，Elewise Tiling 会拒绝含 0 的 shape。
- `dim0` 的多维乘法、`ubFormer` 的 int64→int32 写入和 `SetLocalMemorySize` 的 uint32 窄化没有算子级显式溢出检查；复用时必须保留框架可表示范围。
- Asin AiCore 节点 workspace 明确为 0，但 ACLNN 总 workspace 还包含 Contiguous、内部结果与 ViewCopy 的 executor 需求，不能固定写成 0。
- Contiguous/ViewCopy 的具体内部 Kernel 网络、workspace offset 和地址重定向由框架按 stride/offset 选择，本卡按允许边界不将其伪装成固定单 Kernel；`asc_vf_call`/`asinf` 的 primitive 内部寄存器、预留区切片和数学近似固定仓不可见。
- A3 的公开 AiCore Route 在固定 `ops-math@165d41f02a43a62324b434cb6bda7c533b14da3e` 中没有对应 Tiling/POD/Kernel body与构建接线，因此未生成无后缀正式卡。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/asin/docs/aclnnAsin&aclnnInplaceAsin.md` | 产品表、`aclnnInplaceAsin*` | 950 公开支持、Inplace dtype/shape/空域/非连续约束 |
| 入口声明/约束 | `math/asin/op_api/aclnn_asin.h` | `aclnnInplaceAsinGetWorkspaceSize/aclnnInplaceAsin` | 两段式 C ABI |
| 入口实现 | `math/asin/op_api/aclnn_asin.cpp` | `CheckInplaceParams/ExecAsinGetWorkspaceSize` | 校验、空域、Contiguous、dtype bypass、Asin、Cast、ViewCopy、workspace |
| 入口实现 | `math/asin/op_api/asin.cpp` | `IsAiCoreSupport/AsinAiCore/AsinAiCpu/Asin` | AiCore/AICPU selector 与任务入队 |
| 入口实现 | `math/cast/op_api/cast.cpp` | `l0op::Cast` | 同 dtype Host no-op 证据 |
| 入口实现 | `conversion/contiguous/op_host/op_api/contiguous.cpp` | `Contiguous/ViewCopy` | 连续 view、非连续转换、结果提交与 Tensor relation 框架边界 |
| 入口实现 | `conversion/view_copy/op_api/view_copy.cpp` | `ViewCopyAiCore/ViewCopyAiCpu` | 非连续 ViewCopy 底层分发边界 |
| Host/Tiling/Launcher | `math/asin/op_host/asin_def.cpp` | `ops::Asin` | 950 OpDef、dtype 和 `asin_apt` 接线 |
| Host/Tiling/Launcher | `math/asin/op_host/arch35/asin_tiling_arch35.cpp` | `AsinTiling::RunTiling/SetTilingData` | dtype DAG、0 workspace、blockDim、local memory |
| Host/Tiling/Launcher | `opbase/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData` | POD 字段与 ABI 顺序 |
| Host/Tiling/Launcher | `opbase/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h` | `ElewiseBaseTiling::DoTiling` | DAG buffer/min-max dtype 转换为通用 Tiling 参数 |
| Host/Tiling/Launcher | `opbase/src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling` | core/tile/loop/tail 公式和 Guard |
| Kernel entry/body | `math/asin/op_kernel/arch35/asin_tilingdata.h` | `AsinTilingData` | Asin POD 仅包装 base Tiling |
| Kernel entry/body | `math/asin/op_kernel/asin_apt.cpp` | `asin` | AIV_ONLY entry 与 dtype constexpr router |
| Kernel entry/body | `math/asin/op_kernel/arch35/asin_dag.h` | `AsinOpDirect/AsinOpWithCast/AsinCustom/AsinSimtCompute` | DAG、1024-thread SIMT、cast 顺序 |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/elewise/elewise_sch.h` | `ElementwiseSch` | pool、地址、循环、DAG 执行、tail |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/vec.h` | `Vec::Cast/CopyIn/CopyOut` | Cast/DataCopyPad primitive 参数 |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | Level-2 buffer 数与 buffer id |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/aux1.h` | `MaxAliveNode` | DAG temp 最大存活数的编译期推导 |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo::GetBufferNumLevel2` | input/output ping-pong与 temp 字节账 |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 Mutex 同步 |
| Kernel entry/body | `math/asin/op_kernel_aicpu/asin_aicpu.cpp` | `AsinCpuKernel::Compute/ComputeAsinKernel` | DOUBLE AICPU 校验、分片与 `std::asin` |
| 构建接线 | `math/asin/op_host/config/ascend950/asin_binary.json` | FLOAT16/FLOAT32 entries | 950 原生 Asin binary dtype |
| 构建接线 | `math/asin/op_kernel_aicpu/asin_aicpu_def.cpp` | AICPU `Asin` OpDef | FLOAT16/FLOAT/DOUBLE AICPU 注册 |
| 构建接线 | `math/asin/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35、ACLNN exclude 接线 |
| 构建接线 | `cmake/func.cmake` | `add_all_modules_sources` 的 AICPU glob 分支 | `op_kernel_aicpu/*_aicpu.cpp` 与 OpDef 接入构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 通用 80 B `EleBaseTilingData` 可复用 | 字段顺序、位宽、offset、元素单位；R-003 不得读取该 POD | `EleBaseTilingData/AsinTilingData` |
| 调用顺序 | `Contiguous→Asin→same-dtype Cast no-op→ViewCopy` | Inplace 结果必须最终提交回原 `selfRef` 视图；失败立即阻断 | `ExecAsinGetWorkspaceSize` |
| workspace/内存 | AiCore Asin 自有 workspace 0；Level-2 pool 统一按 max dtype 4 B/元素切片 | ACLNN 总 workspace 不能等同节点 workspace；R-001 B=4、R-002 B=6、155648 B 预留 | `SetTilingData/ElementwiseSch::Init/DAGSch` |
| 同步 | pool slice 通过 MTE2/V/MTE3 Mutex 交接；AICPU 通过 ParallelFor 完成 | ping/pong id、pipe producer→consumer 顺序和 executor stream ordering | `GetTensor/ReleaseTensor/ComputeAsinKernel` |
| 精度与尾块 | FLOAT 直接 `asinf`；FLOAT16 核内 FP32；DOUBLE `std::asin` | Cast mode 0/1、1024 threads、`i<count`、former/tail POD 选择不得改变 | `asin_dag.h/ElementwiseSch::Process` |
| 失败边界 | 空域 Host 成功短路；dtype/shape/UB/Tiling/AICPU 错误逐层返回 | 不得给不公开 BF16、缺失 A3 body 或 Tiling 失败臆造 fallback | `CheckInplaceParams/RunTiling/CheckAsin` |