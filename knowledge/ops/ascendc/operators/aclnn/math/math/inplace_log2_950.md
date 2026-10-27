---
type: Operator
title: "InplaceLog2 950 ACLNN 算子知识"
description: "说明 aclnnInplaceLog2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/docs/aclnnLog2&aclnnInplaceLog2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/aclnn_log2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/aclnn_log2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/log.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/contiguous/op_host/op_api/contiguous.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/view_copy/op_api/view_copy.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/log_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/arch35/log_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/arch35/log_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/arch35/log_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/log_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel_aicpu/log_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel_aicpu/log_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/config/ascend950/log_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/config/ascend950/log_simplified_key.ini
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/op_api/level2_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/aicpu/aicpu_op_def.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch_with_scalar.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/aicpu_common/context/utils/kernel_util.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/aicpu_common/context/utils/kernel_util.cc
    role: implementation
created_at: "2026-08-25T11:17:53Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceLog2 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceLog2GetWorkspaceSize` + `aclnnInplaceLog2`，无版本后缀的最新公开 ACLNN 两段式接口 |
| 计算语义 | 对 `selfRef` 的每个逻辑元素计算以 2 为底的对数，并把结果提交回同一用户 Tensor 视图 |
| 输入输出 | 一个输入/输出别名 `selfRef`；ND、1～8 维，可非连续；950 dtype 为 FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128；空 Tensor 由 Host 短路 |
| 执行类型 | R-001/R-002 核心 `Log` 节点为 `AIV_VECTOR`；R-003/R-004 为 AICPU；`Contiguous/ViewCopy` 是框架边界。所有已闭合核心节点均不含 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 FP32 AIV；R-002 FP16/BF16 核内升降精度 AIV；R-003 DOUBLE AICPU；R-004 COMPLEX64/COMPLEX128 AICPU |
| 核心资源 | AIV：GM、ATVOSS 四分片 UB pool、MicroAPI 寄存器、Mutex pipe、16 MiB 节点 workspace（body 未读）；AICPU：GM、CPU 分片和默认 100 B workspace |
| 来源闭包 | `source_closed`：公开 Inplace API、前后处理、同 dtype Cast 消除、Log AiCore/AICPU selector、950 Tiling/POD、AIV body、AICPU body与构建接线均闭合；框架生成 launcher 和 AscendC primitive 内部为明确边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `N` 为 `selfRef` 逻辑 shape 的元素数。对每个逻辑索引 `i`，接口实现：

```text
selfRef[i] <- log_2(selfRef[i])
            = ln(selfRef[i]) * (1 / ln(2))
```

公开 API 固定向底层 `Log` 传 `base=2.0f, scale=1.0f, shift=0.0f`。AIV Route 按 `ln(scale*x+shift)*invLnBase` 执行；AICPU 实数 Route 命中专门的 `std::log2` 分支，复数 Route 计算 `std::log(x)/std::log(2)`。实数负值、零、NaN/Inf 以及复数 branch cut 的具体结果由对应 primitive 决定，固定算子源码没有另加域裁剪。

“原地”是 ACLNN 用户可见语义：`selfRef` 同时是输入和输出。实现先获得连续输入并分配核心 `Log` 结果，最后由 `ViewCopy` 提交回原视图；核心 Kernel 的 `x/y` GM 指针不要求物理别名。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；1～8 维；可非连续；可为空 | 950：FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128 | 非空指针；输入和输出是同一个 Tensor 对象。BF16 由 `GetDtypeSupportListV2` 的 RegBase 分支开放。 |
| `workspaceSize` | Host 标量输出 | `uint64_t*` | 第一段接口返回 executor 汇总后的 Device workspace；空 Tensor 固定为 0。不能把核心 AIV 节点的 16 MiB 与 ACLNN 总量简单等同。 |
| `executor` | Host 输出句柄 | `aclOpExecutor**` | 第一段创建并移交；持有任务图、临时 Tensor、workspace 切片和关系。 |
| `workspace` | Device 地址 | 字节缓冲区 | 第二段输入；容量必须匹配同一 executor 返回的 `workspaceSize`。 |
| `stream` | runtime stream | `aclrtStream` | `CommonOpExecutorRun` 在该 stream 上执行已构造任务图。 |

底层属性不暴露为本 API 参数：`base=2.0f`、`scale=1.0f`、`shift=0.0f` 都由 `aclnnLog2Common` 固定生成。

### 入口与版本

公开声明与文档只提供无版本后缀的 `aclnnInplaceLog2GetWorkspaceSize/aclnnInplaceLog2`；固定仓没有同语义更高版本。Inplace 第一段把 `selfRef` 经 `const_cast` 同时作为 `out`，先做 Inplace dtype 校验，再复用 `aclnnLog2Common`；本卡不合并非原地 `aclnnLog2`，也不合并自然对数 `aclnnInplaceLog` 或常用对数 `aclnnInplaceLog10`。

950 由公开产品表、`Log` 对 `ascend950` 的 OpDef、950 三个浮点 binary、`ascend950 -> arch35` CMake 接线共同证明。公开文档也写 A2/A3 支持，但固定提交没有 `ascend910b/ascend910_93` 的 Log OpDef、binary config 与对应 Host/Kernel 全链；`mc62` 构建名和 `arch35` 文件不能单独证明 A2/A3 实现身份，故不生成无后缀卡。

## Host/Launcher 执行流程

### 参数校验与前处理

真实调用顺序如下：

1. `aclnnInplaceLog2GetWorkspaceSize` 记录 DFX，把 `selfRef` 同时标记为输入和输出。
2. `CheckInplaceParams` 检查 `selfRef != nullptr`，再按当前架构选择 dtype 白名单；950/RegBase 使用含 BF16 的 `{FLOAT,FLOAT16,BF16,DOUBLE,COMPLEX64,COMPLEX128}`。
3. `aclnnLog2Common(selfRef,out=selfRef,...)` 再执行 `CheckParams`：非空、输入/输出 dtype 白名单、同 dtype可转换、同 shape且 rank 不超过公共 1-in/1-out 上限。因为 input/out 是同一 Tensor，dtype/shape关系天然一致，但公共校验仍真实执行。
4. `CheckFormat` 对 `FRACTAL_NZ` 仅告警；公开接口格式是 ND。
5. `selfRef->IsEmpty()` 时在任何核心节点入队前返回 `workspaceSize=0` 并移交 executor。
6. `l0op::Contiguous(selfRef)`：连续输入返回 reshape/view；非连续输入由框架选择优化组合或 `AsStridedToContiguous`，失败返回 `ACLNN_ERR_INNER_NULLPTR`。
7. `l0op::Log(selfContiguous,2,1,0,executor)` 分配同 view shape 的核心输出。FLOAT/FLOAT16/BF16 进入 AICore；DOUBLE/COMPLEX64/COMPLEX128 进入 AICPU。
8. `l0op::Cast(logOut,selfRef.dtype)` 因 `Log` 对这六类 dtype 都保持 dtype 不变，命中 `self.dtype==dstDtype` 的 Host no-op，直接返回 `logOut`，不分配 Tensor、不入队 Cast Kernel。
9. `l0op::ViewCopy(logOut,selfRef)` 把连续结果提交回用户视图；连续结果可由地址关系、TensorMove 或 NPU copy 完成，非连续结果由 ViewCopy 框架边界恢复。
10. 第一段写 `uniqueExecutor->GetWorkspaceSize()` 并 `ReleaseTo(executor)`；第二段只调用 `CommonOpExecutorRun`。

### 参数、Workspace 与生命周期

用户持有 `selfRef` 存储；executor 持有连续化结果、核心 `Log` 输出、ViewCopy 元数据/任务和各节点 workspace。核心 AIV Tiling 固定写 `workspace[0]=16,777,216` B，但 `log(x,y,workspace,tiling)` 仅把 workspace 透传至 `LogKernel`，body 从未读取该指针，因此这是节点级框架预留。AICPU OpDef 的默认配置写 `opInfo.workspaceSize="100"`；AICPU body 同样只从 context 读取输入、输出和三个属性。

ACLNN 总 workspace 还受非连续 `Contiguous/ViewCopy`、核心输出临时 Tensor、框架对齐与 slice 安排影响，固定算子源码没有给出只依赖 `N` 的闭式总公式。`ReleaseTo` 后 executor 拥有这些对象，直到第二段执行完成；调用者须保持 `selfRef/workspace/executor/stream` 有效。

### 启动图与失败传播

```text
aclnnInplaceLog2GetWorkspaceSize(selfRef)
  -> [N==0] success, workspaceSize=0, no Device node
  -> Contiguous(selfRef)                              # framework boundary
  -> Log(continuous, base=2, scale=1, shift=0)
       -> R-001/R-002: ADD_TO_LAUNCHER_LIST_AICORE(Log)
          -> log<schMode=1,attrWork=2>(x,y,workspace,tiling)  # one AIV kernel
       -> R-003/R-004: ADD_TO_LAUNCHER_LIST_AICPU(Log)
          -> LogCpuKernel::Compute                              # one AICPU node
  -> Cast(logOut,selfRef.dtype)                       # Host no-op, no node
  -> ViewCopy(logOut,selfRef)                         # framework boundary
  -> executor.GetWorkspaceSize()

aclnnInplaceLog2(workspace,workspaceSize,executor,stream)
  -> CommonOpExecutorRun(...,stream)
```

| 节点/边 | entry 与参数 | 交接介质 | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|
| `selfRef -> Contiguous` | `l0op::Contiguous(selfRef)` | 用户 GM 视图 -> executor 连续 Tensor；具体 workspace slice 随 stride 而变 | executor 持有连续 Tensor，任务在同一 stream 中先于 Log | 返回空指针即停止构图 |
| AIV Log | `log(x,y,workspace,tiling)`；`blockDim=ElewiseBaseTiling::GetBlockDim()`；POD=`EleBaseTilingData32B` | 连续 GM input -> 连续 GM temp output；16 MiB workspace 参数未读 | 单 AIV Kernel；输出由 executor 持有到 ViewCopy 消费 | Tiling/launcher失败阻断 ViewCopy |
| AICPU Log | `LogCpuKernel::Compute`；context 携带 `base/scale/shift` | 连续 GM input -> 连续 GM temp output；默认 100 B workspace 不被 body 读取 | 串行 loop 或 `ParallelFor` 完成后节点返回 | 参数、dispatch、ParallelFor错误向上传播 |
| `Log -> ViewCopy` | producer 为核心 Log，consumer 为 `l0op::ViewCopy` | 同 shape、同 dtype的 executor Tensor；offset 由 executor 管理 | 同一 executor/stream 建立 producer→consumer 顺序；结果在第二段结束前不可读取 | 核心失败则无有效结果，后续不能成功完成 |

## Route 与参数设计

### 分核与资源计算

以下只适用于 R-001/R-002。令 `N=product(storage_shape)`、`P=GetCoreNumAiv()`、`U=平台 UB 字节数`。本 API 的 `base=2` 使 `attrWork=2`，因此三种 AIV dtype 都选择 generic DAG；LEVEL-2 liveness 将该单消费者线性链的中间值映射回输入/输出分片，不增加独立 temp，最终是输入 ping/pong与输出 ping/pong共 `B=4` 个物理分片，`MaxDtypeBytes=4`。

```text
minBits = 32 (FLOAT) or 16 (FLOAT16/BFLOAT16)
C = min(ceil(N * minBits / 32768), P)                  # 目标 core 数；32768 bit=4 KiB/core
blockFormer = ceil(ceil(N / C) / 512) * 512            # 元素；通用源码直接按 512 元素对齐
blockNum = ceil(N / blockFormer)                        # 实际 launch blockDim
blockTail = N - (blockNum - 1) * blockFormer

alignFactor = 256*8/minBits                             # 元素；FLOAT 64，F16/BF16 128
maxElemNums = floor(U*8 / (B*32)) = floor(U/16)         # 每分片按 max dtype 4 B
ubFormer = floor(maxElemNums/alignFactor)*alignFactor

ubLoopFormer = ceil(blockFormer/ubFormer)
ubLoopTail = ceil(blockTail/ubFormer)
ubTailFormer = blockFormer-(ubLoopFormer-1)*ubFormer
ubTailTail = blockTail-(ubLoopTail-1)*ubFormer
```

Host 的 32 B POD 只写 `N/C/ubFormer` 与三个 scalar；Kernel scheduler 用相同公式重算 block/tile尾部。`blockNum` 留在 Host `ElewiseBaseTiling::numBlocks` 并传给 `SetBlockDim`。shape 含 0、`P==0`、`U<extraSize`、`ubFormer==0`、dtype/shape校验失败都会返回 `GRAPH_FAILED`，没有另一个 AIV fallback。

R-003/R-004 不进入 AscendC Tiling。AICPU 先按 dtype/元素数判断是否并行：DOUBLE 阈值 `60*1024` 元素，COMPLEX64/COMPLEX128 阈值 `18*1024` 元素；超过阈值时 `workers=max(1,GetCPUNum(ctx)-2)`，`perUnit=ceil(N/workers)`，交给 `ParallelFor`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 AIV | `GET_TPL_TILING_KEY(1,2)`；`LogScaleNotOneShiftNotZeroLnbaseNotOne<float>::OpDag` | 950/RegBase；非空；`selfRef.dtype==FLOAT`；L0 `IsAiCoreSupport` 为真；固定属性令 `attrWork=2` | FLOAT16/BF16 到 R-002；DOUBLE 到 R-003；复数到 R-004 |
| 2 | R-002 FP16/BF16 AIV | 同一 key表达式；`...<half/bfloat16_t>::OpDag` | 950；非空；dtype 为 FLOAT16 或 BFLOAT16；其余 API/shape guard通过 | DOUBLE/复数到 AICPU；其它 dtype API失败 |
| 3 | R-003 DOUBLE AICPU | Host dtype selector；`LogCompute<double,double>` | 950；非空；dtype DOUBLE；API白名单接受但 AiCore list不含 DOUBLE | 无下一 Route；AICPU错误终止 |
| 4 | R-004 COMPLEX AICPU | Host dtype selector；`LogCompute<complex<T>,complex<T>>` | 950；非空；dtype COMPLEX64 或 COMPLEX128；AiCore list不含复数 | 无下一 Route；AICPU错误终止 |

```text
if selfRef is null or dtype not in A5 Inplace list or rank > 8:
    fail ACLNN parameter check
if N == 0:
    return empty executor, workspaceSize=0
x = Contiguous(selfRef)
if dtype == FLOAT:                 R-001 AIV generic-log DAG
else if dtype in {F16,BF16}:       R-002 AIV generic-log DAG with FP32 intermediates
else if dtype == DOUBLE:           R-003 AICPU std::log2
else:                              R-004 AICPU complex log/log(2)
Cast(coreResult,selfRef.dtype) returns coreResult
ViewCopy(coreResult,selfRef)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 AIV | `GET_TPL_TILING_KEY(1,2)` | 950；FLOAT；同 shape；rank<=8；非空；属性 `(2,1,0)` | 单 AIV Kernel | `ElementwiseSchWithScalar<EleBaseTilingData32B,1,Generic<float>::OpDag>`；`Muls->Adds->LogCustom<float>->Muls` | 四个 `ubFormer*4` B分片；中间值复用I/O pool；MTE2/V/MTE3 Mutex；MicroAPI mask | former/tail block与tile；Tiling失败无 fallback |
| R-002 FP16/BF16 AIV | 同 key；dtype binary各自实例化 | 950；F16或BF16；其余同 R-001 | 单 AIV Kernel | `Cast<float,U>->Muls->Adds->LogCustom<float>->Muls->Cast<U,float>` | 四个 `ubFormer*4` B物理分片；输入/输出有效2 B/元素；FP32中间值在I/O pool内按liveness复用 | tail count逐节点传递；失败无 fallback |
| R-003 DOUBLE AICPU | `IsAiCoreSupport=false` | 950；DOUBLE；同 shape；rank<=8；非空 | 单 AICPU节点 | `LogCpuKernel::Compute -> LogCompute<double,double> -> Log2Compute<double>` | GM + CPU worker；`ParallelFor`返回为同步点 | 串行或 `[start,end)`分片；dispatch失败终止 |
| R-004 COMPLEX AICPU | `IsAiCoreSupport=false` | 950；COMPLEX64或COMPLEX128；其余同 R-003 | 单 AICPU节点 | `LogCompute<complex<float/double>> -> Log2Compute`复数 specialization | GM + CPU worker；18K 元素并行阈值 | worker尾分片；不支持类型/属性失败终止 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Inplace dtype | RegBase 取 `ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST` | API只允许保持 dtype 的输出类型 | 全部 | `ACLNN_ERR_PARAM_INVALID` | `aclnn_log2.cpp::CheckInplaceDtypeValid`; `level2_base.h::GetDtypeSupportListV2` |
| 空域 Host 短路 | `self->IsEmpty()` | `ElewiseTiling`拒绝 0 维并会产生除零/空循环边界 | 全部 | 成功返回0 workspace，无核心节点 | `aclnnLog2Common`; `ElewiseTiling` |
| AiCore dtype | `dtype in {FLOAT,FLOAT16,BF16}` | 950 OpDef/binary与模板只覆盖三类 | R-001/R-002 | DOUBLE/复数进入 AICPU | `log.cpp::IsAiCoreSupport/Log` |
| 固定 Log2属性 | `base=2,scale=1,shift=0`; `invLnBase=1/log(2)`，不满足 optimized natural-log谓词 | 只有 `invLnBase==1` 才能省去 scalar链；Log2必须保留换底乘法 | R-001/R-002 | 恒进入 generic Route；无自然对数 Route | `aclnnLog2Common`; `LogTiling::SetAttr` |
| UB 非零 | `ubFormer=align_down(floor(U/16),64或128)>0` | LEVEL-2 generic DAG需要4个按FP32宽度切分的pool分片 | R-001/R-002 | Tiling失败 | `ElewiseTiling::GetMaxElemNum/ElewiseTiling` |
| AIV核数 | `GetCoreNumAiv()>0` | 分核公式对 core 数除法 | R-001/R-002 | Tiling失败 | `ElewiseBaseTiling::GetPlatformInfo`; `TilingPrepareForLog` |
| tile参数窄化 | scheduler `tileLength:uint64_t` 传给 `LogCustom(...,uint32_t count)`，内部 `loopNum:uint16_t` | MicroAPI VL循环和mask有显式窄类型 | R-001/R-002 | 固定仓无独立 fallback；平台 UB导出的 tile上界是必要边界 | `ElementwiseSchWithScalar::RunOp`; `LogCustom` |
| AICPU并行阈值 | DOUBLE `N>61440`；复数 `N>18432` | 小数据避免 ParallelFor；大数据保留2个CPU并按ceil分片 | R-003/R-004 | 阈值以下串行 `shard_log(0,N)` | `NeedParallel`; `LogCompute`; `kResvCpuNum` |
| 同 dtype外层 Cast | `logOut.dtype==selfRef.dtype` | 避免额外 Cast Tensor/Kernel | 全部 | 本 Inplace路线恒成立；改变底层输出 dtype 时必须重审 | `cast.cpp::Cast` |

### POD/ABI

R-001/R-002 使用 `EleBaseTilingData32B`，自然布局 `sizeof=32` B、`alignof=8` B。它没有 block/tail字段，Device从前三个字段重算；`scalarData`依调用顺序保存三个 FP32值。R-003/R-004没有 Log 私有 Tiling POD，属性由 AICPU context提供。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `EleBaseTilingData32B` | `dim0` | `int64_t`/64 | offset 0，size 8 | 元素 | `AdaptEleBaseTilingData32B: = elewiseTilingData.dim0` | constructor写 `dim0_`并重算block/tail | shape乘积溢出检查固定仓不可见；ACLNN空域先短路 |
| 同上 | `coreNum` | `int32_t`/32 | offset 8，size 4 | AIV core | `= elewiseTilingData.coreNum`隐式窄化 | constructor写 `coreNum_`，参与 `ceil(dim0/coreNum)` | Host先按平台核数限幅；int64→int32范围检查不可见 |
| 同上 | `ubFormer` | `int32_t`/32 | offset 12，size 4 | 元素/tile | `= elewiseTilingData.ubFormer`隐式窄化 | `blockLen=ubFormer*MaxDtypeBytes`、tile循环与tail | 必须非0；F32按64元素、F16/BF16按128元素对齐 |
| `scalarData` | `scale` | `float`/32 | struct offset 16，scalar offset 0 | 无量纲 | 第一遍 `SetScalar<float>(attrScale=1.0f)` | `Placeholder::Var<float,0>` -> 首个 `Muls` | generic Route active |
| `scalarData` | `shift` | `float`/32 | struct offset 20，scalar offset 4 | 与输入值同量纲 | 第二遍 `SetScalar<float>(attrShift=0.0f)` | `Placeholder::Var<float,1>` -> `Adds` | generic Route active |
| `scalarData` | `invLnBase` | `float`/32 | struct offset 24，scalar offset 8 | 无量纲 | 第三遍 `SetScalar<float>(1.0f/log(2.0f))` | `Placeholder::Var<float,2>` -> 最后 `Muls` | 决定换底；Host以float计算 |
| `scalarData` | 保留尾部 | `char[4]` | struct offset 28～31 | N/A | 未写 | 当前 DAG不读取 | 值不可依赖 |
| AICPU context | `base` | `float`/32 | context内部ABI固定仓不可见 | 无量纲 | launcher attr `2.0f` | `LogOpCheck -> base_`; `Log2Condition` | 必须为-1或正且非1；本API恒合法 |
| AICPU context | `scale` | `float`/32 | context内部ABI固定仓不可见 | 无量纲 | launcher attr `1.0f` | `LogOpCheck -> scale_` | 本API恒为1 |
| AICPU context | `shift` | `float`/32 | context内部ABI固定仓不可见 | 输入量纲 | launcher attr `0.0f` | `LogOpCheck -> shift_` | 本API恒为0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 AIV | `GET_TPL_TILING_KEY(1,2)` | `log.cpp::IsAiCoreSupport`; `LogTiling::RunTiling`; `ElewiseTiling` | `EleBaseTilingData32B` + 3 scalars | `log<1,2>` -> `LogKernel<1,2,float>` | `ElementwiseSchWithScalar`; generic `LogDag`; `LogCustom<float>`; Vec/Sync | `log_def.cpp`; `log_binary.json`; `log_simplified_key.ini`; `CMakeLists.txt`; `func.cmake` |
| R-002 FP16/BF16 AIV | 同 key，binary dtype specialization | 同上，dtype分支选择 `half/bfloat16_t` DAG | 同一32 B POD | `LogKernel<1,2,DTYPE_X>` | generic DAG的Cast/Muls/Adds/Log/Muls/Cast；ATVOSS pool | 同上 |
| R-003 DOUBLE AICPU | L0 dtype fallback | `log.cpp::Log/LogAiCpu` | AICPU context + `(2,1,0)` attrs | `REGISTER_CPU_KERNEL("Log",LogCpuKernel)` | `LogOpCheck`; `LogCompute<double,double>`; `Log2Compute<double>` | `log_aicpu_def.cpp`; `aicpu_op_def.h`; `CMakeLists.txt`; `func.cmake` |
| R-004 COMPLEX AICPU | L0 dtype fallback | 同 R-003 | 同上 | 同 R-003 | `LogCompute<complex<T>>`; complex `Log2Compute` specialization | 同 R-003 |

## Kernel 实现

### 公共执行流程

R-001/R-002 共用单 Kernel AIV launch graph。entry解析 `EleBaseTilingData32B`，声明 `KERNEL_TYPE_AIV_ONLY`，按编译期 `attrWork=2` 构造 generic DAG scheduler。scheduler以 `blockFormer*blockIdx`绑定每核GM基址，按 `ubFormer`逐tile执行 DAG，`tile&1`选择输入/输出 ping/pong；三个 scalar从POD加载一次到 `VarType`。

本 API 永远不会到达 `LogScaleOneShiftZeroLnbaseOne`：尽管 `scale=1,shift=0`，`base=2`产生的 `invLnBase!=1`。generic DAG中的同dtype Cast由 `DAGSch::GetRealArgs/IsCastNoOp`编译期删除；F16/BF16 Cast保留。R-003/R-004绕过 AIV entry/Tiling，由 AICPU直接处理连续GM。

### Route R-001：FP32 AIV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/log/op_api/aclnn_log2.cpp` | `CheckInplaceParams/aclnnLog2Common` | RegBase list含FLOAT；固定 `(base,scale,shift)=(2,1,0)` | 连续化后调用底层 Log |
| L0 | `math/log/op_api/log.cpp` | `IsAiCoreSupport/Log` | FLOAT在 `{FLOAT,FLOAT16,BF16}` | `LogAiCore` |
| Tiling | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `SetAttr/RunTiling` | `outputDtype==DT_FLOAT`; `invLnBase!=1` | `DoTiling32B<Generic<float>::OpDag>`，key表达式 `(1,2)` |
| Router | `math/log/op_kernel/log_apt.cpp` | `log/LogKernel` | `attrWork==TPL_...NOT_ONE(2)` | `ElementwiseSchWithScalar<...,Generic<float>>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(Log, x, y, attrs=2,1,0)
  -> Tiling4Log -> LogTiling::RunTiling
     -> ElewiseBaseTiling::DoTiling32B<Generic<float>::OpDag>
  -> log<1,2>(x,y,workspace,tiling)
     -> LogKernel<1,2,float>
        -> ElementwiseSchWithScalar::Init(x,y)
        -> ElementwiseSchWithScalar::Process
           -> Vec::CopyIn<float>
           -> Vec::Muls<float>(scale)
           -> Vec::Adds<float>(shift)
           -> LogCustom<float> -> MicroAPI::Log
           -> Vec::Muls<float>(invLnBase)
           -> Vec::CopyOut<float>
  -> executor GM edge -> ViewCopy(result,selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
blockFormer = align_up(ceil(N/coreNum), 512 elements)
blockNum = ceil(N/blockFormer)
gmBase = blockFormer * blockIdx
loopNum = isLastBlock ? ubLoopTail : ubLoopFormer
tailNum = isLastBlock ? ubTailTail : ubTailFormer
offset = 0
for tile in [0,loopNum):
    count = (tile == loopNum-1) ? tailNum : ubFormer
    pp = tile & 1
    DataCopyPad(inPP, x[gmBase+offset], blockCount=1,
                blockLen=count*4 B, srcStride=0, dstStride=0, pad=false)
    tmp0 = inPP * scale            # scale=1.0f, still a runtime Vector op
    tmp1 = tmp0 + shift            # shift=0.0f, still a runtime Vector op
    for regLoop in [0,ceil(count/(VECTOR_REG_WIDTH/4))):
        mask = UpdateMask<float>(remaining count)
        vin = DataCopy<DIST_NORM>(tmp1 + regLoop*VL)
        vlog = MicroAPI::Log(vin, mask)
        DataCopy<DIST_NORM_B32>(tmp0 + regLoop*VL, vlog, mask)
    outPP = tmp0 * invLnBase       # invLnBase=1/ln(2), output pool may reuse a dead temp id
    DataCopyPad(y[gmBase+offset], outPP, 1, count*4 B, 0, 0)
    offset += count
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `(blockFormer*blockIdx+tileOffset)*4` B | `count*4` B有效 | FLOAT/连续ND | Contiguous | CopyIn | 用户/executor，核心期有效 | 只搬有效count |
| input ping/pong | pool ID 0/2，`id*blockLen` | 每区 `ubFormer*4` B | FLOAT | MTE2 DataCopyPad或已死亡中间值后的Vector op | first Muls/后续DAG op | 当前偶/奇tile | 双缓冲；LEVEL-2允许在CopyIn值死亡后作为VECCALC复用 |
| output ping/pong | pool ID 1/3 | 每区 `ubFormer*4` B | FLOAT | 中间Vector op或final Muls | 下一DAG op/MTE3 CopyOut | 当前偶/奇tile | 与input分片交替承载线性中间值；最终结果落MTE3角色 |
| MicroAPI寄存器 | `regLoop*VL` | `vregInput/vregOutput<float>` + mask；具体字节不可见 | FLOAT lanes | UB load/Log | UB store | 单reg loop | `UpdateMask`保护最后VL |
| GM result | 同逻辑元素offset | `count*4` B | FLOAT/连续ND | CopyOut | ViewCopy | executor持有至提交 | 不写无效tail |
| Log workspace | 节点slice | 16,777,216 B | opaque | Host Tiling | entry参数未读 | executor，节点期 | 算法不复用该slice |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inP["input/VECCALC ping · ubFormer×4 B"]:1
  outP["output/VECCALC ping · ubFormer×4 B"]:1
  inQ["input/VECCALC pong · ubFormer×4 B"]:1
  outQ["output/VECCALC pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
blockLen = ubFormer * MaxDtypeBytes = ubFormer*4
UB_total = (2 input + 2 output) * blockLen
         = 16*ubFormer B <= U
dynamic_ub = N/A（AIV固定UB pool）
workspace_total(core Log) = 16,777,216 B；body读取0 B
=> ubFormer = align_down(floor(U/16),64 elements)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input ping/pong | CopyIn | first Muls | 同parity下一tile | `Get/ReleaseTensor<VECIN>`与`VECCALC` Mutex | V读完后MTE2才覆盖 |
| ping/pong内的VECCALC别名 | 当前Vector/MicroAPI op | DAG下一op | liveness判定前值死亡后的后续op | `ConvertArgs/TryReleaseArgs`的V Mutex与生成的buffer ID | 单消费者完成才复用；不能跨活跃值别名 |
| output ping/pong | final Muls | CopyOut | 同parity下一tile | V/MTE3 Mutex | GM提交后才覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM->UB | `DataCopyPad` + PIPE_MTE2 Mutex | GM input | first Vector op | input ping/pong | `blockLen=count*4` B，pad=false |
| scalar链 | `Muls/Adds` + PIPE_V Mutex | 当前pool值 | 下一DAG op | input/output pool的VECCALC别名 | 全FP32；三个scalar来自POD；count逐op传递 |
| 对数 | MicroAPI masked load/`Log`/store | pre-log temp | final Muls | vreg与temp | `VL=VECTOR_REG_WIDTH/4`，最后VL由mask保护 |
| UB->GM | `DataCopyPad` + PIPE_MTE3 Mutex | output ping/pong | GM/ViewCopy | output分片 | 仅写count个元素 |
| 跨节点 | executor/stream ordering | Log | ViewCopy | GM temp结果 | 核心失败阻断最终提交 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `LogTiling::SetAttr/RunTiling` | 固定属性转scalar、选择generic DAG、写32 B POD/blockDim/16 MiB workspace |
| Kernel entry/router | `math/log/op_kernel/log_apt.cpp` | `log<1,2>/LogKernel<1,2,float>` | AIV-only，attrWork编译期分支 |
| Device body/helper | `math/log/op_kernel/arch35/log_dag.h` | `LogScaleNot...<float>/LogCustom<float>` | Muls/Adds/FP32 Log/换底Muls完整链 |
| Scheduler dependency | `opbase/.../elewise_sch_with_scalar.h`; `util/dag.h` | `Init/Process/Run`; LEVEL_2 buffer映射 | 分核/tile、scalar装载、四分片liveness复用与Mutex |

### Route R-002：FP16/BF16 核内升降精度 AIV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/log/op_api/aclnn_log2.cpp` | `CheckInplaceDtypeValid` | RegBase list含FLOAT16/BFLOAT16 | 通过Inplace校验 |
| L0 | `math/log/op_api/log.cpp` | `IsAiCoreSupport/Log` | F16/BF16在AiCore list | `LogAiCore` |
| Tiling | `log_tiling_arch35.cpp` | `RunTiling` dtype分支 | dtype为F16或BF16且`attrWork=2` | `DoTiling32B<Generic<half/bfloat16_t>>` |
| Router | `log_apt.cpp` | `LogKernel<1,2,DTYPE_X>` | binary的`DTYPE_X`为half或bfloat16_t | generic mixed-width DAG |

#### Device 数据流

##### 函数级调用链

```text
log<1,2,DTYPE_X=half or bfloat16_t>
  -> ElementwiseSchWithScalar<...,Generic<U>::OpDag>::Init/Process
     -> Vec::CopyIn<U>
     -> Vec::Cast<float,U,0>
     -> Vec::Muls<float>(scale)
     -> Vec::Adds<float>(shift)
     -> LogCustom<float> -> MicroAPI::Log
     -> Vec::Muls<float>(invLnBase)
     -> Vec::Cast<U,float,1>
     -> Vec::CopyOut<U>
  -> ViewCopy(result,selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码；block/tile地址同R-001，GM有效字节改为count*2
for each tile(count,pp):
    DataCopyPad(inputU, x[offset], 1, count*2 B, 0, 0, pad=false)
    fp32A = Cast<float,U,roundMode=0>(inputU,count)
    fp32B = Muls(fp32A,1.0f,count)
    fp32A = Adds(fp32B,0.0f,count)                 # 死值后复用temp物理区
    fp32B = masked MicroAPI::Log(fp32A,count)
    fp32A = Muls(fp32B,1.0f/ln(2),count)
    outputU = Cast<U,float,roundMode=1>(fp32A,count)
    DataCopyPad(y[offset],outputU,1,count*2 B,0,0)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `(blockFormer*blockIdx+tileOffset)*2` B | 每侧`count*2` B | F16或BF16/连续ND | Contiguous/CopyOut | CopyIn/ViewCopy | 用户或executor | 有效count搬运 |
| input ping/pong | pool ID0/2 | 每区物理`ubFormer*4` B，逻辑CopyIn为count×2 B | U或复用期FLOAT | CopyIn/中间FP32 op | Cast-up/后续DAG op | 当前parity tile | pool按MaxDtypeBytes=4统一分片，值死亡后转VECCALC角色 |
| output ping/pong | pool ID1/3 | 每区物理`ubFormer*4` B，逻辑CopyOut为count×2 B | FLOAT中间值或U最终值 | Cast/Muls/Adds/Log/Cast-down | 下一DAG op/CopyOut | 当前parity tile | 与input分片交替承载FP32链，最终降精度结果落MTE3角色 |
| MicroAPI寄存器 | regLoop×VL | FP32 input/output reg + mask；字节不可见 | FLOAT lanes | FP32 UB load | FP32 UB store | 当前reg loop | Log不走`LogCustom<U>`的16-bit内部支路 |
| Log workspace | 节点slice | 16,777,216 B | opaque | Tiling | entry未读 | executor | 不作为FP32中间GM |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inP["input/FP32 calc ping · ubFormer×4 B"]:1
  outP["output/FP32 calc ping · ubFormer×4 B"]:1
  inQ["input/FP32 calc pong · ubFormer×4 B"]:1
  outQ["output/FP32 calc pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
MaxDtypeBytes=4, MinDtypeBytes=2, BufferNum=4
UB_total = 4*ubFormer*4 = 16*ubFormer B <= U
dynamic_ub = N/A
workspace_total(core Log)=16,777,216 B；FP32中间值全部留在UB
=> ubFormer=align_down(floor(U/16),128 elements)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output ping-pong | CopyIn/Cast-down | Cast-up/CopyOut | 同parity下一tile | MTE2/V/MTE3 Mutex | 消费或搬出完成后覆盖 |
| pool内FP32别名 | 当前cast/vector/log | 下一DAG op | 前值死亡后的后续op | `TryReleaseArgs`释放VECCALC buffer并由生成buffer ID复用 | 不把仍被后续引用的源与dst复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | MTE2 Mutex + DataCopyPad | GM U | Cast-up | input分片 | `count*2` B，pad=false |
| 升精度 | PIPE_V + `AscendC::Cast` | input U | scalar链 | FP32 temp | 模板mode常量0；枚举语义固定依赖未命名，不能外推round名称 |
| Log2计算 | PIPE_V + Muls/Adds/MicroAPI Log/Muls | FP32 temp | Cast-down | 两个FP32 temp | 全FP32；mask只激活count；换底系数为Host float |
| 降精度 | PIPE_V + `AscendC::Cast` | FP32结果 | CopyOut | output U分片 | 模板mode常量1；对F16/BF16只处理有效count |
| CopyOut | MTE3 Mutex + DataCopyPad | output U | GM/ViewCopy | output分片 | `count*2` B，不提交无效lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `log_tiling_arch35.cpp` | F16/BF16 generic `DoTiling32B`分支 | mixed 16/32-bit DAG生产core/tile/POD |
| Kernel entry/router | `log_apt.cpp` | `LogKernel<1,2,DTYPE_X>` | binary dtype实例化同一AIV-only entry |
| Device body/helper | `log_dag.h` | `LogScaleNot...<U,float>` | Cast-up→scalar链→FP32 Log→Cast-down |
| Vector dependency | `opbase/.../util/vec.h` | `Cast/Muls/Adds/CopyIn/CopyOut` | count、scalar与DataCopyPad参数直达primitive |

### Route R-003：DOUBLE AICPU

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_log2.cpp` | `CheckInplaceDtypeValid` | RegBase Inplace list含DOUBLE | 通过API校验 |
| L0 | `log.cpp` | `IsAiCoreSupport/Log` | DOUBLE不在AiCore list | `LogAiCpu` |
| AICPU check | `log_aicpu.cpp` | `LogOpCheck` | input/output均DOUBLE；`base=2,scale=1,shift=0` | dispatch到map |
| AICPU router | 同上 | `log_calls.at(DT_DOUBLE).at(DT_DOUBLE)` | map精确命中 | `LogCompute<double,double>` |

#### Device 数据流

##### 函数级调用链

```text
LogAiCpu(double x)
  -> ADD_TO_LAUNCHER_LIST_AICPU(Log, attrs base/scale/shift)
  -> LogCpuKernel::Compute
     -> NormalCheck -> LogOpCheck
     -> log_calls[DT_DOUBLE][DT_DOUBLE]
     -> LogCompute<double,double>
        -> NeedParallel(N,DT_DOUBLE)
        -> shard_log -> Log2Condition -> Log2Compute<double>
           -> output[i]=std::log2(input[i])
  -> ViewCopy(result,selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
require 1 input, 1 output, same DOUBLE dtype
require base==-1 or (base>0 and base!=1)       # base=2 passes
N = input.NumElements()
if N==0: return OK                             # normally intercepted by ACLNN
shard(start,end):
    for i in [start,end): output[i]=std::log2(input[i])
if N > 60*1024:
    workers=max(1,GetCPUNum(ctx)-2)
    ParallelFor(N,ceil(N/workers),shard)
else:
    shard(0,N)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| AICPU GM input | `input+start*8` B | `(end-start)*8` B | DOUBLE连续 | Contiguous | CPU shard | executor/节点期 | worker区间不重叠 |
| AICPU GM output | `output+start*8` B | `(end-start)*8` B | DOUBLE连续 | CPU shard | ViewCopy | executor至提交 | 无atomic，每worker独占 |
| CPU local | 当前元素 | `std::log2`内部/寄存器字节不可见 | DOUBLE | input load | output store | 单迭代/worker | 无共享归约 |
| AICPU workspace | 框架节点slice | 默认100 B | opaque | `ApplyMathAicpuDefaultCfg` | body未读 | executor | 不计为算法中间数组 |

##### 分层内存布局

N/A：本 Route 不使用 AIV UB、`TQue/TBuf` 或 dynamic UB；实际路径是 AICPU 直接对连续 GM 做串行或互不重叠的区间分片。

##### 字节账与复用顺序

```text
UB_total=N/A（AICPU）
dynamic_ub=N/A
workspace_total(core Log AICPU)=100 B配置值，body显式消费0 B
worker_extent_bytes=(end-start)*8
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM output worker slice | 当前worker | ParallelFor结束后的ViewCopy | executor结束后用户可覆盖 | `ParallelFor`返回表示全部shard完成；executor再运行ViewCopy | `[start,end)`不重叠且producer成功 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU计算 | 串行loop或`ParallelFor` | GM input | 函数返回 | GM output全区 | DOUBLE `std::log2`，每个shard用半开区间自然覆盖尾部 |
| 跨节点 | executor/stream | AICPU Log | ViewCopy | GM结果Tensor | 非OK返回阻断成功提交 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_api/log.cpp` | `IsAiCoreSupport/LogAiCpu` | DOUBLE走AICPU，无AscendC POD |
| Kernel entry/router | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogCpuKernel::Compute/log_calls` | 校验并映射DOUBLE pair |
| Device body/helper | 同上 | `LogCompute<double,double>/Log2Compute<double>` | 60K阈值、留2核、ceil分片与逐元素std::log2 |

### Route R-004：COMPLEX64/COMPLEX128 AICPU

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_log2.cpp` | `CheckInplaceDtypeValid` | RegBase list含COMPLEX64/128 | 通过API校验 |
| L0 | `log.cpp` | `IsAiCoreSupport` | 复数不在AiCore list | `LogAiCpu` |
| AICPU check | `log_aicpu.cpp` | `LogOpCheck` | 非整数分支要求input/output dtype相同 | 进入复数map项 |
| AICPU router | 同上 | `log_calls` | C64/C128 pair各自存在 | `LogCompute<std::complex<float/double>,same>` |

#### Device 数据流

##### 函数级调用链

```text
LogCpuKernel::Compute
  -> LogOpCheck(base=2,scale=1,shift=0)
  -> log_calls[COMPLEX64/COMPLEX128]
  -> LogCompute<T,T>
     -> NeedParallel(N,complex dtype)
     -> Log2Compute<T> complex specialization
        -> log_2=T(std::log(2.0f))
        -> output[i]=std::log(input[i])/log_2
  -> executor edge -> ViewCopy
```

##### 源码等价伪码

```text
# 源码等价伪码；T=complex<float>或complex<double>
require input.dtype==output.dtype==selected complex dtype
denom = T(std::log(2.0f))
shard(start,end):
    for i in [start,end): output[i]=std::log(input[i])/denom
if N > 18*1024:
    workers=max(1,GetCPUNum(ctx)-2)
    ParallelFor(N,ceil(N/workers),shard)
else:
    shard(0,N)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| AICPU GM input | `input+start*sizeof(T)` | `(end-start)*sizeof(T)` | COMPLEX64(8 B)或COMPLEX128(16 B) | Contiguous | CPU shard | executor/节点期 | shard不重叠 |
| AICPU GM output | 同元素offset | 同上 | 同输入复数dtype | CPU shard | ViewCopy | executor至提交 | 无atomic |
| CPU local | 当前复数与`denom` | 字节/库内部固定仓不可见 | complex<T> | std::log/division | output store | worker调用期 | denom每次shard lambda调用路径内可复用 |
| AICPU workspace | 节点slice | 默认100 B | opaque | OpDef | body未读 | executor | 非算法数组 |

##### 分层内存布局

N/A：本 Route 不使用 AIV UB；实际替代路径为 AICPU 对 COMPLEX64/128 GM 区间并行计算。

##### 字节账与复用顺序

```text
UB_total=N/A
dynamic_ub=N/A
workspace_total(core Log AICPU)=100 B配置值，body显式消费0 B
worker_extent_bytes=(end-start)*sizeof(T), sizeof(T)=8 or 16
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM output shard | worker | ViewCopy | executor完成后 | ParallelFor完成 + executor顺序 | shard半开区间不相交 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 复数Log2 | `std::log` + complex division | GM input | GM output | worker私有元素 | C64用complex<float>，C128用complex<double>；denom由float常量2计算自然对数后转T |
| 分片 | `ParallelFor`或串行 | shard | 函数返回 | 全输出 | 18K阈值；ceil分片覆盖最后余数 |
| 跨节点 | executor/stream | AICPU Log | ViewCopy | GM结果Tensor | 错误阻断提交 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_api/log.cpp` | `Log/LogAiCpu` | 复数不走AiCore |
| Kernel entry/router | `log_aicpu.cpp` | `log_calls` C64/C128项 | 两个同dtype模板实例 |
| Device body/helper | 同上 | complex `Log2Compute`; `NeedParallel` | log/log(2)、18K阈值、CPU分片 |

## 算子特定优化

### Trick T-001：把换底、升精度和 Log 融合进一个 AIV DAG

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：950 dtype为FLOAT/FLOAT16/BFLOAT16，且固定Log2属性使`attrWork=TPL_SCALE_NOT_ONE_SHIFT_NOT_ZERO_LNBASE_NOT_ONE`。
- Golden 来源：[`math/log/op_kernel/arch35/log_dag.h:106-119`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/arch35/log_dag.h#L106-L119)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename U, typename T = float>
struct LogScaleNotOneShiftNotZeroLnbaseNotOne {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;

    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn0Cast1 = Bind<Vec::Muls<T>, OpCopyIn0Cast, Placeholder::Var<T, PLACEHOLDER_INDEX_0>>;
    using OpCopyIn0Cast2 = Bind<Vec::Adds<T>, OpCopyIn0Cast1, Placeholder::Var<T, PLACEHOLDER_INDEX_1>>;
    using OpLogResult = Bind<LogCustom<T>, OpCopyIn0Cast2>;
    using OpCopyIn0Cast3 = Bind<Vec::Muls<T>, OpLogResult, Placeholder::Var<T, PLACEHOLDER_INDEX_2>>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpCopyIn0Cast3>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;
```

#### 机制与设计原因

Log2没有单独启动scale、shift、自然对数、换底或Cast Kernel；它们被DAG串在同一tile的UB生命周期中。对F16/BF16，输入先转FP32，`Muls/Adds/Log/Muls`都在FP32 temp完成，最后才降回原dtype，FP32中间值不落GM。对FLOAT，前后同dtypeCast由DAG编译期消除。该机制减少中间GM物化并固定精度顺序；未声明未经真机验证的收益幅度。

#### 迁移边界

- 适用：可表达为逐元素scalar链且中间FP32能够容纳在ATVOSS tile pool的950 unary算子。
- 前提：Host scalar写入顺序与`Placeholder::Var<0/1/2>`一致；`BufferNum/MaxDtypeBytes`按新DAG重新计算；dtype必须有binary与selector。
- 失效：DOUBLE/复数转入AICPU；自然对数`base=-1,scale=1,shift=0`应选另一optimized DAG，不能沿用本Log2 generic链说明。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把`scale=1`和`shift=0`自行从本Route删除：Host只有在`invLnBase`也等于1时才选择optimized DAG；Log2的`1/ln(2)`使该条件恒不成立。固定仓未提供其它可验证反例。

### Trick T-002：LEVEL-2 liveness把线性中间值复用进四个I/O分片

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：generic Log DAG使用`MemOptCfg<MemLevel::LEVEL_2>`，线性中间值均为单消费者。
- Golden 来源：[`pkg_inc/op_common/atvoss/util/dag.h:369-393`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h#L369-L393)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
  template <typename NodeInfo,
            bool use_nddma = true, bool cache_brc = false>
  __aicore__ constexpr static uint32_t GetBufferNumImpl() {
    constexpr MemLevel bufferLvl = ChooseBufferLevelImpl<NodeInfo, use_nddma, cache_brc>();
    if constexpr (bufferLvl == MemLevel::LEVEL_0) {
      return NodeInfo::template GetBufferNumLevel0<use_nddma, cache_brc>();
    } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
      return NodeInfo::template GetBufferNumLevel1<use_nddma, cache_brc>();
    } else {// bufferLvl == 2
      return NodeInfo::template GetBufferNumLevel2<use_nddma, cache_brc>();
    }
  }
```

#### 机制与设计原因

LEVEL-2依据DAG liveness分配buffer ID，而不是为Cast、每个Muls、Adds和Log各永久保留一整tile。generic线性链没有独立temp计数：中间值在输入/输出ping-pong四分片中按死亡点交替复用；scheduler在消费者完成后通过V Mutex释放对应ID。这样UB上界为`16*ubFormer`，而不是按所有节点数线性增长。

#### 迁移边界

- 适用：单输出、线性或可证明liveness的逐元素DAG。
- 前提：保持`GetBufferIds`生成映射和`TryReleaseArgs`释放协议；新增多消费者、持久节点或多输出后重新计算。
- 失效：若中间值跨多个后继仍活跃，temp数会增加，原`floor(U/16)`公式失效；AICPU Route不使用该机制。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但源码的静态断言明确要求总buffer不超过32，不能绕过DAG统计手工让活跃值别名。

### Trick T-003：外层同 dtype Cast 在 Host 消除

#### 触发与源码依据

- 适用 Route：R-001～R-004
- 触发条件：底层`Log`对Inplace允许的六类dtype都分配同dtype输出，目标又是`selfRef.dtype`。
- Golden 来源：[`math/cast/op_api/cast.cpp:380-384`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp#L380-L384)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor)
{
    if (self->GetDataType() == dstDtype) {
        return self;
    }
```

#### 机制与设计原因

`aclnnLog2Common`保留统一的`Cast(logOut,out.dtype)`结构，但Inplace路线实际不产生Cast Tensor或Kernel。它避免核心结果在提交回原视图前发生一次冗余GM读写；F16/BF16所需精度转换已经在AIV UB DAG内部完成，不与这个Host no-op混淆。

#### 迁移边界

- 适用：底层输出dtype严格等于用户目标dtype的ACLNN组合流程。
- 前提：确认L0分配规则和AICPU dispatch都保持dtype；不能只凭API形参相同推断。
- 失效：若底层返回提升dtype，Cast会真实分配并进入AiCore/AICPU selector，任务图和workspace必须重做。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把AIV DAG内部的F16/BF16降精度Cast也删掉，它决定最终输出dtype与舍入路径。

## 模板与实现对比

| 对比维度 | R-001 FP32 AIV | R-002 FP16/BF16 AIV | R-003 DOUBLE AICPU | R-004 COMPLEX AICPU | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | FLOAT且AiCore可用 | F16/BF16且AiCore可用 | DOUBLE且AiCore不支持 | C64/C128且AiCore不支持 | L0 dtype白名单唯一分发 |
| 模板与 dtype | generic DAG `<float,float>`，Cast no-op消除 | generic DAG `<U,float>` | `LogCompute<double,double>` | `LogCompute<complex<T>,same>` | binary/OpDef与AICPU map |
| 循环与分核 | AIV block + tile + reg VL | 同左，GM 2 B/元素 | 阈值60K，CPU shard | 阈值18K，CPU shard | dtype最小位宽与CPU成本 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 四分片pool + FP32 regs | 四分片pool + FP32 regs | N/A，CPU local | N/A，CPU complex local | 无通信/NDDMA/SIMT |
| 内存总占用与复用 | `16*ubFormer` B | 同左；输入输出逻辑半宽 | worker slice×8 B | worker slice×8/16 B | AIV统一按max dtype切片 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、Muls/Adds、MicroAPI Log | 再加Cast-up/down | std::log2 | std::log/complex divide | 所有节点均非Cube |
| 同步、精度与 tail | MTE2/V/MTE3 Mutex；全FP32 | 核内FP32后降精度 | ParallelFor；DOUBLE | ParallelFor；complex精度 | AIV mask和CPU半开区间分别处理tail |
| fallback/失败边界 | Tiling失败终止 | Tiling失败终止 | AICPU错误终止 | AICPU错误终止 | 不存在跨Route运行时重试 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持；固定提交缺少可归属A2的Host/Tiling/Kernel/build闭包，未生成卡 | 公开文档写支持；固定提交缺少`ascend910b/910_93`完整闭包，未生成卡 | 支持；公开API、AIV三dtype、AICPU三dtype和构建均闭合 |
| Host/Launcher | 固定仓不可见对应平台真实实现 | 固定仓不可见对应平台真实实现 | 复用`aclnnLog2Common`；AIV/AICPU按dtype分发 |
| Kernel/资源 | 不以950 arch35外推 | `mc62`名字不能单独证明A3；不以其外推 | `ascend950` arch35 AIV四分片pool；AICPU fallback |

## 限制与失败边界

- 只覆盖原地以2为底语义；不把非原地Log2、自然Log、Log10合入同一版本族。
- `selfRef`空指针返回`ACLNN_ERR_PARAM_NULLPTR`；dtype、rank或公共shape/dtype关系不合法返回`ACLNN_ERR_PARAM_INVALID`；中间L0对象/launcher创建失败返回内部错误。
- 空Tensor由ACLNN成功短路；绕过ACLNN直接调用底层Tiling时，shape任一维为0会失败。
- AIV Tiling对shape乘积、`int64 -> int32` POD窄化与`uint64 -> uint32` tile count没有可见的独立溢出检查；复用者必须受平台UB与接口shape边界约束，不得扩大。
- AIV节点声明16 MiB workspace但body未读；AICPU配置100 B也未被body读取。ACLNN总workspace仍由executor汇总，不能写死为任一节点值。
- `Contiguous/ViewCopy`可能按stride选择不同框架节点；其精确内部workspace offset与Kernel数随输入而变，作为允许的框架边界，不伪造固定图。
- AscendC `Log/Cast/Muls/Adds/DataCopyPad` primitive内部实现、MicroAPI寄存器实际字节和runtime executor序列化细节在固定仓不可见。
- A2/A3只有公开支持声明而没有当前固定源码的source closure，故未创建`inplace_log2.md`。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/log/docs/aclnnLog2&aclnnInplaceLog2.md` | 产品表、Inplace参数表 | 950公开支持、dtype/rank/非连续/确定性 |
| 入口声明/约束 | `math/log/op_api/aclnn_log2.h` | `aclnnInplaceLog2*` | 两段式声明与原地语义 |
| 入口实现 | `math/log/op_api/aclnn_log2.cpp` | `CheckInplaceParams/aclnnLog2Common/aclnnInplaceLog2*` | 校验、空域、固定属性、Contiguous/Log/Cast/ViewCopy/executor |
| Host selector | `math/log/op_api/log.cpp` | `IsAiCoreSupport/LogAiCore/LogAiCpu/Log` | dtype到AIV/AICPU分发和输出dtype |
| 框架前处理 | `conversion/contiguous/op_host/op_api/contiguous.cpp` | `Contiguous` | 连续no-op与非连续框架边界 |
| 框架后处理 | `conversion/view_copy/op_api/view_copy.cpp` | `ViewCopy` | 地址关系/TensorMove/非连续提交边界 |
| Host优化 | `math/cast/op_api/cast.cpp` | `Cast` | 同dtype直接返回，无Cast节点 |
| OpDef | `math/log/op_host/log_def.cpp` | `ops::Log` | 950三dtype和`ascend950/log_apt`注册 |
| Host/Tiling | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `LogTiling::*`; `Tiling4Log` | dtype/shape/属性、DAG、scalar、workspace、key、blockDim |
| POD模板 | `math/log/op_kernel/arch35/log_struct.h` | `Log`模板声明/选择 | schMode与attrWork模板集合 |
| Device DAG | `math/log/op_kernel/arch35/log_dag.h` | `LogCustom`; 两个DAG | MicroAPI Log、generic换底和Cast精度链 |
| Kernel entry | `math/log/op_kernel/log_apt.cpp` | `log/LogKernel` | AIV-only router与scheduler实例 |
| AICPU body | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogCpuKernel/log_calls/LogCompute/Log2Compute` | DOUBLE/复数计算、阈值、分片和错误 |
| AICPU OpDef | `math/log/op_kernel_aicpu/log_aicpu_def.cpp` | `ops::Log` | AICPU dtype注册与默认配置调用 |
| AICPU公共配置 | `common/inc/aicpu/aicpu_op_def.h` | `ApplyMathAicpuDefaultCfg` | kernel so与100 B workspace |
| 950 binary | `math/log/op_host/config/ascend950/log_binary.json` | `Log` op_list | BF16/F16/F32 950实例 |
| simplified key | `math/log/op_host/config/ascend950/log_simplified_key.ini` | `[Log] default=0` | binary key模式 |
| 算子构建 | `math/log/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950使用arch35，源码模块接线 |
| 公共构建 | `cmake/func.cmake` | `add_all_modules_sources` | L0/L2/Tiling/OpDef/AICPU源码收集 |
| API架构helper | `common/inc/op_api/level2_base.h` | `GetDtypeSupportListV2` | RegBase选择含BF16白名单 |
| Tiling POD | `pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData32B` | 32 B字段布局 |
| Tiling模板 | `pkg_inc/op_common/atvoss/elewise/elewise_tiling.h` | `ElewiseBaseTiling::DoTiling32B/SetScalar` | DAG统计、POD写入、scalar offset |
| Tiling公式 | `src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling/GetMaxElemNum` | core/block/tile/UB公式与失败guard |
| Scheduler | `pkg_inc/op_common/atvoss/elewise/elewise_sch_with_scalar.h` | `ElementwiseSchWithScalar` | GM地址、tile循环、scalar读取、DAG调度 |
| DAG内存 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch/GetBufferNum/GetBufferIds` | Cast no-op、LEVEL-2 liveness与buffer ID |
| Vector helper | `pkg_inc/op_common/atvoss/util/vec.h` | `Cast/Muls/Adds/CopyIn/CopyOut/IsCastNoOp` | primitive参数、搬运字节和Cast消除 |
| 同步helper | `pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 Mutex协议 |
| AICPU helper | `pkg_inc/op_common/aicpu_common/context/utils/kernel_util.h` | `kResvCpuNum` | 保留2个CPU |
| AICPU helper | `aicpu_common/context/utils/kernel_util.cc` | `CeilMultiple` | ceil分片公式 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 32 B POD只传N/core/tile和3个float scalar | 字段offset、scalar顺序`scale/shift/invLnBase`、保留4 B不可读 | `EleBaseTilingData32B`; `SetScalar`; `SetScalar<idx>` |
| 调用顺序 | `Contiguous -> Log -> Cast(no-op) -> ViewCopy` | 原地语义靠最终ViewCopy；不能让核心临时结果提前覆盖用户非连续视图 | `aclnnLog2Common` |
| workspace/内存 | AIV 16 MiB节点预留且body不读；UB四分片；AICPU默认100 B | ACLNN总量由executor汇总；新增DAG节点要重算BufferNum | `RunTiling`; `LogKernel`; `DAGSch` |
| 同步 | AIV按MTE2/V/MTE3 Mutex，AICPU以ParallelFor返回为完成点 | ping/pong同parity覆盖必须等消费者释放；ViewCopy只在核心完成后消费 | `GetTensor/ReleaseTensor`; `LogCompute` |
| 精度与尾块 | F16/BF16核内FP32链后降精度；F32全FP32；AICPU保持dtype | Cast mode常量、换底顺序、MicroAPI mask与CPU半开区间不可改 | `LogScaleNot...`; `LogCustom`; `Log2Compute` |
| 失败边界 | 空域Host短路；dtype selector；Tiling/AICPU错误终止 | 不把自然Log optimized DAG、A2/A3缺失实现或未知dtype当fallback | `CheckInplaceParams`; `SetAttr`; `LogOpCheck` |