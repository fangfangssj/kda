---
type: Operator
title: "ClampMin 950 ACLNN 算子知识"
description: "说明 aclnnClampMin 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/docs/aclnnClampMin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/aclnn_clamp.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/clip_by_value.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/clip_by_value_v2_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_single_dim_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/op_host/math_tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/config/ascend950/clip_by_value_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:52:19Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ClampMin 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnClampMinGetWorkspaceSize` + `aclnnClampMin`，公开 ACLNN 两段式接口；固定仓未声明更高版本入口。 |
| 计算语义 | 对每个元素计算 `out[i] = max(self[i], clipValueMin)`；内部复用三输入 `ClipByValueV2`，把缺省上界正规化为当前计算 dtype 的正无穷或整数最大值。 |
| 输入输出 | `self/out` 为 ND、rank 1–8、shape 相同；950 文档 dtype 为 FLOAT16/FLOAT/DOUBLE/INT8/UINT8/INT16/INT32/INT64/BFLOAT16，`clipValueMin` 还接受 BOOL；非连续 `self/out` 由框架节点处理。 |
| 执行类型 | R-001：单 AIV_VECTOR Kernel；R-002：单 AICPU task。AIV entry 显式拒绝 `AscendC::AIC` 并声明 `KERNEL_TYPE_AIV_ONLY`，无 Cube/AIC/MIX_CV 可达节点。 |
| Kernel 路线 | R-001 `promoteType` 属于 `{FLOAT16, FLOAT, INT32, INT64, BFLOAT16}` 时走 `ClipByValueV2Ass<T>`；R-002 其余 ACLNN 可达 dtype 走 `ClipByValueV2CpuKernel`。空 `self` 在 Host 直接成功返回，不启动 Kernel。 |
| 核心资源 | R-001：GM 标量直读、VECIN/VECOUT 双缓冲 Queue、无 TBuf，算子 Tiling 请求 16 MiB workspace 但 body 不读；R-002：AICPU 输入/输出 GM 指针、CPU 标量寄存器和 `ParallelFor`，body 无 UB/POD/workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→参数正规化/Executor→AICore/AICPU selector→950 Tiling/POD→AIV entry/body，以及 AICPU task/body与 950 构建接线均已闭合；通用 ACLNN 框架节点按明确边界记录。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开语义是

```text
for i in [0, numel(self)):
    out[i] = max(self[i], clipValueMin)
```

实现复用 `ClipByValueV2(x, min, max)`。`aclnnClampMinGetWorkspaceSize` 将 `clipValueMax=nullptr` 传给 `aclnnClampCommon`；后者按 `promoteType` 生成正无穷或整数最大值作为 `max`，所以 Device 实际执行 `min(max(x, min), dtype_max)`。对本接口的合法计算 dtype，第二次上界裁剪不改变非 NaN 数值；浮点 NaN 的 Vector primitive 细节在固定仓外，不从公式外推。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，rank 1–8；可非连续 | FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16 | 必填；与 `out` shape 相同。空 Tensor 在 Host 返回 `workspaceSize=0` 且不创建计算节点。 |
| `clipValueMin` | `aclScalar`，无 Tensor layout | FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16、BOOL | 必填；在 950 先与 `self` 做 `ClampPromoteType`，再转换成单元素 Tensor。 |
| `out` | ND，rank 1–8；可非连续；shape 与 `self` 相同 | 与 `self` 相同的九种非 BOOL dtype | 必填；`promoteType` 必须可 cast 到 `out`，950 在核心结果后显式 `Cast` 再 `ViewCopy`。 |
| `workspaceSize` | Host 标量 | `uint64_t*` | 第一段接口输出完整 Executor 所需字节数。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段接口创建并移交；封装前处理、核心计算和输出写回。 |
| `workspace/workspaceSize/stream` | Device workspace 与 ACL stream | 指针、`uint64_t`、`aclrtStream` | 第二段接口原样交给 `CommonOpExecutorRun`。 |

### 入口与版本

固定仓公开 `aclnnClampMinGetWorkspaceSize` 与 `aclnnClampMin`，没有 ClampMin 的版本后缀入口。第一段仅处理 Scalar 下界；`aclnnClampMinTensor`、`aclnnInplaceClampMinTensor` 是不同逻辑算子，不在本卡范围。

公开文档明确列出 Ascend 950PR/950DT 支持。950 实现还由 `ClipByValueV2::AICore().AddConfig("ascend950", ...)`、`SUPPORT_COMPUTE_UNIT "ascend950"` 和 `arch35` Tiling/Kernel 接线共同确认。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `aclnnClampMinGetWorkspaceSize` 记录 phase-1 DFX，并单独拒绝空 `clipValueMin`。
2. `aclnnClampCommon` 创建 Executor；`CheckParams` 检查 `self/out` 非空、至少一个边界非空、平台 dtype 支持，以及 `self/out` shape 相同且 rank 不超过框架上限。
3. `self->IsEmpty()` 时直接令 `workspaceSize=0`、移交空 Executor 并成功返回。
4. 950 满足 `IsRegBase(curArch)`：以 `self.dtype` 为初值，经 `GetScalarDefaultDtype` 与 `CombineCategoriesWithComplex` 生成 `promoteType`，随后检查它可 cast 到 `out.dtype` 且属于 950 支持表。
5. `NormalizeMaxScalar(nullptr, promoteType, executor)` 生成正无穷或对应整数最大值；实际 `clipValueMin` 保持原 Scalar 对象。两者经 `ConvertToTensor(..., promoteType)` 成为单元素边界 Tensor。
6. `self` 依次经过 `l0op::Contiguous` 与 `l0op::Cast(..., promoteType)`。这些通用节点可按输入状态成为实际 Kernel 或框架空操作，固定仓只暴露调用边界。
7. `l0op::ClipByValueV2` 按计算 dtype 选择 AICore 或 AICPU；950 随后总是调用 `Cast(coreResult, out.dtype)` 和 `ViewCopy(..., out)`。
8. 任一节点创建返回空指针即中止第一段并返回相应 ACLNN 错误；成功时读取 Executor 聚合 workspace 并移交给调用者。

### 参数、Workspace 与生命周期

调用者拥有 `self`、`clipValueMin`、`out`、第二段传入的 workspace 和 stream。第一段创建的临时 Scalar/Tensor、连续化/转换结果与核心结果由 `uniqueExecutor` 持有，`ReleaseTo(executor)` 后其生命周期转交公开 Executor。

核心 AIV Tiling 的 `GetWorkspaceSize` 固定请求 `16 * 1024 * 1024` 字节；`ClipByValueV2Ass::Init/Process` 不读取 `workspace` 参数，因此该切片不承载核心节点的中间数据。AICPU body 也没有 workspace 参数。完整 `workspaceSize = uniqueExecutor->GetWorkspaceSize()` 还可能包含通用 `Contiguous/Cast/ViewCopy` 节点；这些框架节点的聚合公式、切片 offset 与别名规则在固定仓不可见。

### 启动图与失败传播

非空输入的逻辑 Executor 图如下；带 `?` 的通用节点是否物化由框架实现决定。

```text
aclnnClampMinGetWorkspaceSize
  -> Contiguous(self)?
  -> Cast(selfContiguous, promoteType)?
  -> ConvertToTensor(minScalar/maxScalar)
  -> selector IsAiCoreSupport(selfCasted)
       -> R-001 ADD_TO_LAUNCHER_LIST_AICORE(ClipByValueV2)
          -> clip_by_value_v2<<<blockNum, stream>>>(x, min, max, y, workspaceSlice, POD)
       -> R-002 ADD_TO_LAUNCHER_LIST_AICPU(ClipByValueV2)
          -> ClipByValueV2CpuKernel::Compute(ctx)
  -> Cast(coreResult, out.dtype)?
  -> ViewCopy(result, out)?
aclnnClampMin(workspace, size, executor, stream)
  -> CommonOpExecutorRun(...)
```

核心分支只有一个 Device 节点，没有节点间 workspace edge。所有已物化通用节点与核心节点由同一 Executor 在调用者提供的 stream 上提交；精确 event/任务描述符、各通用节点 workspace slice、失败后的 cleanup 实现属于固定仓不可见的 ACLNN 框架边界。第一段的每次 `CHECK_RET` 阻止后续节点加入；第二段执行错误由 `CommonOpExecutorRun` 返回。

## Route 与参数设计

### 分核与资源计算

R-001 使用 950 AIV 核数和 UB 大小。Scalar 边界在 `DimensionCollapse` 后形成单轴广播模式，优先级 1 的 `ClipByValueTilingSingleDim` 令 `isAss=(minDim==1 && maxDim==1)`，选择 key `2001100`。

令 `s=sizeof(T)`、`U=平台每核 UB 字节数`、`C=GetCoreNumAiv()`、`N=yDim`：

```text
dbUbSize       = floor(U / 2)                                      # 双缓冲预留
ubFormerBytes  = floor(floor(dbUbSize / 2) / 128) * 128            # ASS 有 2 个 alive node
ubFormer       = ubFormerBytes / s                                  # 元素
ubOuter        = ceil(N / ubFormer)                                 # tile 数
ubTail         = (N % ubFormer == 0) ? ubFormer : N % ubFormer     # 元素
blockFormer    = ceil(ubOuter / C)                                  # 普通核 tile 数
blockTail      = (ubOuter % blockFormer == 0) ? blockFormer
                                                    : ubOuter % blockFormer
blockNum       = ceil(ubOuter / blockFormer)
```

若 `blockNum < C/2` 且 `ubFormer*s*2 > 8192`，Host 以 `ceil((2*N/C)*s,128)/s` 缩小 tile，同时不低于 `floor((8192/2)/128)*128/s`，再重算核与尾块参数。`ubFormer==0`、`C==0` 或 `N==0` 均使 Tiling 失败。

R-002 不使用 AIV 分核/POD。单元素输出直接串行计算；其余数据若 `numel*sizeof(T) < 200 KiB` 也串行，否则保留 2 个 CPU，令 `max_core_num=max(1,max(cpu_num,2)-2)`，并把 `CeilMultiple(total,max_core_num)*sizeof(T)` 作为 `ParallelFor` 的 grain 参数。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 AIV Scalar-Scalar | l0 selector + Tiling key `2001100`；`ClipByValueV2Ass<DTYPE_X>` | `self` 非空；950 `promoteType ∈ {FLOAT16,FLOAT,INT32,INT64,BFLOAT16}`；`IsAiCoreSupport` 先判 true。Tiling registry 按升序先试 priority 1；合轴后四个 Tensor 均为单轴且 `minDim=maxDim=1`。 | dtype 不在集合时转 R-002；Tiling null/失败直接失败，不从 AICore 回退 AICPU。 |
| 2 | R-002 AICPU Scalar | `ClipByValueV2CpuKernel::Compute`，无 TilingKey | `self` 非空且 950 `promoteType` 不在 AICore 五类型集合；对公开 ClampMin 主要覆盖 DOUBLE/INT8/UINT8/INT16 及 promotion 后仍落入这些类型的组合。 | dtype dispatch 不支持、dtype/shape 不一致或 `ParallelFor` 失败时返回 AICPU 错误。 |

```text
if self.IsEmpty():
    workspaceSize = 0; return success
promoteType = ClampPromoteType(self.dtype, clipValueMin.dtype)
if promoteType in {F16, F32, I32, I64, BF16}:
    if collapsed_rank(x,min,max,y)==1 and minDim==1 and maxDim==1:
        key=2001100 -> R-001
    else:
        # 对当前 Scalar/Scalar ACLNN 图不活跃；registry 才继续 priority 2/3
else:
    R-002
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 AIV Scalar-Scalar | `IsAiCoreSupport` + `2001100` | 950；非空；计算 dtype 为 F16/F32/I32/I64/BF16；x/y 同 shape；min/max 为单元素，合轴单维 | 单 Kernel：`clip_by_value_v2` AIV entry | `ClipByValueV2Ass<DTYPE_X>::Init/Process` | VECIN/VECOUT 各 2 个 `ubFormer*s` buffer；Queue EnQue/DeQue；scalar GM 每核直读；核心 workspace 未使用 | 最后一核取 `blockTail`，最后 tile 取 `ubTail`；未知 key entry 静默 return。 |
| R-002 AICPU Scalar | `!IsAiCoreSupport(selfCasted)` | 950；非空；公开支持且 promotion 后不是五种 AICore dtype；三个输入和输出 dtype 相同，x/y 元素数相同，min/max 均单元素 | 单 AICPU task | `ClipByValueV2CpuKernel::Compute -> DispatchByDtype -> DoCompute<T>` | 直接 GM/CPU pointer；标量边界预装局部变量；小数据串行，大数据 `ParallelFor` | `numel==1` 单点；其余无向量 padding；dtype/shape/parallel 失败返回错误。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AICore dtype | `CheckType(selfCasted.dtype, {F16,F32,I32,I64,BF16})` | 950 binary config 和 AIV body只覆盖五种同 dtype 输入输出 | R-001/R-002 | false 走 AICPU；AICore 启动后失败不再 fallback | `op_api/clip_by_value.cpp::IsAiCoreSupport` |
| SingleDim priority | 四个合轴 dims 的 vector 长度均为 1 | ASS body仅支持线性 x/y 与两个标量边界 | R-001 | false 时 registry 尝试 priority 2；当前 ACLNN Scalar 图不活跃 | `ClipByValueTilingSingleDim::DetermineSigDimMode/IsCapable` |
| ASS UB 下限 | `ubFormer != 0`，由 `floor((U/2/2)/128)*128/s` 产生 | 两个 alive Queue、各自双缓冲，tile 字节需 128B 对齐 | R-001 | Tiling `GRAPH_FAILED` | `CalcInitialUbParams` |
| 低核利用率重切 | `blockNum < C/2 && ubFormer*s*2 > 8192` | 原 tile 过大导致活跃核少；缩小 tile 但保持每核至少 8 KiB alive 数据 | R-001 | 条件 false 保留初始 tile | `AdjustForLowUtilization` |
| AIV-only | `g_coreType == AscendC::AIC` 时立即 return；`KERNEL_TYPE_AIV_ONLY` | 防止 AIC/Cube 核执行 Vector body | R-001 | AIC 不产生输出；正常 launch 必须指向 AIV | `clip_by_value_v2` entry |
| AICPU 串并行阈值 | `total*sizeof(T) < 200*1024` | 小数据避免 `ParallelFor` 调度；大数据按 CPU 数分片 | R-002 | 小数据串行，大数据并行；并行失败返回 rc | `DispatchParallel` |
| 空域 | `self->IsEmpty()` | 避免 Tiling 的 `yDim==0` 失败与无意义 launch | 两 Route 之前 | Host 成功返回，无 Kernel | `aclnnClampCommon` |

### POD/ABI

`ClipByValueSigDimTilingData` 由 `TILING_DATA_FIELD_DEF` 生成。固定仓明确每个字段为 `int64_t`，但宏展开所在运行时头不在本仓，故结构总 `sizeof/alignof` 与逐字段 `offsetof` 固定仓不可见，不以自然布局猜测。R-002 的 AICPU task 不消费 Tiling POD。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ClipByValueSigDimTilingData` | `blockNum` | `int64_t`/64 | `sizeof/alignof/offsetof` 固定仓不可见 | AIV core | `DoOpTiling: set_blockNum(ceil(ubOuter/blockFormer))`；亦写入 launch blockDim | `Init` 保存并用于越界核 guard | 必须 >0；最后核索引为 `blockNum-1`。 |
| 同上 | `ubFormer` | `int64_t`/64 | 同上 | 元素/tile | `CalcInitialUbParams`，可能被 `AdjustForLowUtilization` 缩小 | `InitBuffer` 容量、整 tile `blockLen`、GM 步长 | 必须非零；字节容量为 `ubFormer*sizeof(T)`。 |
| 同上 | `ubTail` | `int64_t`/64 | 同上 | 元素 | `N % ubFormer`，整除时为 `ubFormer` | 最后一核最后一 tile 的 copy/compute 长度 | 范围 `1..ubFormer`。 |
| 同上 | `blockFormer` | `int64_t`/64 | 同上 | tile/core | `ceil(ubOuter/coreNum)` | 普通核循环数；GM 基址 `blockIdx*blockFormer*ubFormer` | 必须非零。 |
| 同上 | `blockTail` | `int64_t`/64 | 同上 | tile | `ubOuter % blockFormer`，整除时为 `blockFormer` | 最后一核 `blockLoopNum` | 范围 `1..blockFormer`。 |
| 同上 | `xDim` | `int64_t`/64 | 同上 | 元素 | 合轴后 `dims[x][0]` | `Init` 读入成员 `xDim`；后续未读取 | 对 ASS 为 y 的线性长度，Device 当前实现中 inactive。 |
| 同上 | `minDim` | `int64_t`/64 | 同上 | 元素 | 合轴后 `dims[min][0]`，ClampMin 为 1 | ASS body 不读取；Host 用它决定 `isAss` | key `2001100` 必须为 1。 |
| 同上 | `maxDim` | `int64_t`/64 | 同上 | 元素 | 合轴后 `dims[max][0]`，缺省上界 Tensor 为 1 | ASS body 不读取；Host 用它决定 `isAss` | key `2001100` 必须为 1。 |
| 同上 | `yDim` | `int64_t`/64 | 同上 | 元素 | 合轴后 `dims[y][0]`；用于 Host tile 公式 | ASS body 不读取 | Host 要求非零；输出覆盖由循环参数间接保证。 |
| AICPU framework task | `Input(0)` | Tensor descriptor/pointer | `ADD_TO_LAUNCHER_LIST_AICPU` task ABI 固定仓不可见 | x 元素 | `ClipByValueV2AiCpu` 的 `OP_INPUT(self, min, max)` | `Compute/DoCompute: ctx.Input(0)->GetData()` | dtype 与 output/bounds 相同；元素数等于 y。 |
| 同上 | `Input(1)` | Tensor descriptor/pointer | 同上 | min 元素 | `minTensor=ConvertToTensor(..., promoteType)` | `DoCompute: mn_ptr`，Scalar route只读 `[0]` | 本接口元素数为 1。 |
| 同上 | `Input(2)` | Tensor descriptor/pointer | 同上 | max 元素 | 缺省上界经 `ConvertToTensor` | `DoCompute: mx_ptr`，Scalar route只读 `[0]` | 本接口元素数为 1。 |
| 同上 | `Output(0)` | Tensor descriptor/pointer | 同上 | y 元素 | `executor->AllocTensor(self view shape, dtype)` | `DoCompute: y_ptr` | 元素数必须等于 x。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 AIV Scalar-Scalar | `IsAiCoreSupport`; priority 1; key `2001100` | `clip_by_value.cpp::ClipByValueV2`; `clip_by_value_v2_tiling.cpp::TilingFor/Prepare`; `clip_by_value_v2_single_dim_tiling.cpp` 注册 priority 1；共享 `clip_by_value_single_dim_tiling.cpp::DoOpTiling/PostTiling` | `ClipByValueSigDimTilingData` 九字段 | `clip_by_value_v2_apt.cpp::clip_by_value_v2` -> `ClipByValueV2Ass<DTYPE_X>` | `clip_by_value_v2_ass.h::Init/Process/CopyIn/Compute/CopyOut` | `op_host/clip_by_value_v2_def.cpp`、ascend950 binary config 与 `CMakeLists.txt` 的 arch35 接线 |
| R-002 AICPU Scalar | `!IsAiCoreSupport` | `clip_by_value.cpp::ClipByValueV2AiCpu`；无 Tiling | Executor Tensor task 参数；无算子 POD | `ADD_TO_LAUNCHER_LIST_AICPU(ClipByValueV2)` -> `REGISTER_CPU_KERNEL` | `ClipByValueV2CpuKernel::Compute/DispatchByDtype/DoCompute/DispatchParallel` | `op_kernel_aicpu_def.cpp::ApplyMathAicpuDefaultCfg/AICPU` 与目录模块接线 |

## Kernel 实现

### 公共执行流程

两个 Route 都接收同 dtype 的 `x/min/max/y`，其中 min/max 是单元素 Tensor；shape infer 使用三输入 broadcast infer，输出 dtype取 x dtype。l0 selector 在加入 launcher list 之前完成，因此一次核心调用只启动一个 AIV Kernel 或一个 AICPU task，不存在核心节点间 workspace edge。两 Route 完成后都回到 950 的结果 `Cast` 与 `ViewCopy` 框架边界。

### Route R-001：AIV Scalar-Scalar

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp` | `aclnnClampMinGetWorkspaceSize/aclnnClampCommon` | 非空；950 `ClampPromoteType`；边界转单元素 Tensor | 调用 l0 `ClipByValueV2`。 |
| l0 selector | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `IsAiCoreSupport/ClipByValueV2` | 计算 dtype 属于 F16/F32/I32/I64/BF16 | `ClipByValueV2AiCore` 加入 AICore launcher。 |
| Tiling registry | `common/inc/op_host/math_tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | `std::map` 升序尝试 priority；非 `GRAPH_PARAM_INVALID` 即终止 | 先试 priority 1。 |
| Tiling | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `DetermineSigDimMode/GetTilingKey` | 四个 collapsed dims 都为单轴；`minDim==1 && maxDim==1` | key `2001100`、POD、`blockDim=blockNum`。 |
| Router | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | AIC core直接 return；`TILING_KEY_IS(2001100)` | 实例化 `ClipByValueV2Ass<DTYPE_X>`。 |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(ClipByValueV2, x, minTensor, maxTensor -> y)
  -> clip_by_value_v2(x, min, max, y, workspace, tiling)
    -> GET_TILING_DATA_WITH_STRUCT(ClipByValueSigDimTilingData)
    -> ClipByValueV2Ass<DTYPE_X>::Init
       -> minGm.GetValue(0), maxGm.GetValue(0)
       -> InitBuffer(inQueueX, 2, ubFormer*sizeof(T))
       -> InitBuffer(outQueue, 2, ubFormer*sizeof(T))
    -> ClipByValueV2Ass<DTYPE_X>::Process
       -> CopyIn -> Compute(Maxs -> in-place Mins) -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；长度单位除 blockLen 外均为元素
b = GetBlockIdx()
if b >= blockNum: return
loops = (b == blockNum-1) ? blockTail : blockFormer
lastLen = (b == blockNum-1) ? ubTail : ubFormer
base = b * blockFormer * ubFormer
minValue = min[0]; maxValue = max[0]
bind xGm=x+base, yGm=y+base

offset = 0
for i in 0 .. loops-2:
    DataCopyPad(inQ.alloc, xGm[offset], blockLen=ubFormer*sizeof(T), no padding)
    xLocal = inQ.dequeue; yLocal = outQ.alloc
    Maxs(yLocal, xLocal, minValue, ubFormer)
    free xLocal
    Mins(yLocal, yLocal, maxValue, ubFormer)
    outQ.enqueue(yLocal)
    DataCopyPad(yGm[offset], outQ.dequeue, blockLen=ubFormer*sizeof(T))
    free yLocal
    offset += ubFormer

# 每核最后一次；只有全局最后核可能为短 tile
DataCopyPad(..., blockLen=lastLen*sizeof(T))
Maxs(..., len=lastLen); Mins(in_place, ..., len=lastLen)
DataCopyPad(..., blockLen=lastLen*sizeof(T))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `b*blockFormer*ubFormer + tile*ubFormer` | 普通 tile `ubFormer`；全局尾 tile `ubTail` | T/线性 ND | API self 经框架前处理 | `CopyIn` | 调用者/Executor，核心 launch 全程有效 | former tile字节为 128B 整数倍；tail 用 DataCopyPad。 |
| GM `min/max` | 0 | 各 1 元素 | T/scalar Tensor | Executor `ConvertToTensor` | 每核 `Init::GetValue(0)` | Executor 持有至核心结束 | 每核只读一次到标量成员。 |
| VECIN `inQueueX` | Queue slot | `2 * ubFormer * s` 字节 | T/线性 | `CopyIn` | `Compute` | `AllocTensor` 至 `FreeTensor` | depth 2；former 容量 128B 对齐。 |
| VECOUT `outQueue` | Queue slot | `2 * ubFormer * s` 字节 | T/线性 | `Compute` | `CopyOut` | `AllocTensor` 至 `FreeTensor` | depth 2；`Mins` 原地覆盖 `Maxs` 结果。 |
| GM `y` | 与 x 相同 base/offset | 有效输出为 N 元素 | T/线性 ND | `CopyOut` | 后续 Cast/ViewCopy | Executor 至后续节点完成 | 只提交有效 `blockLen` 字节。 |
| workspace | entry 参数 | Tiling 请求 16 MiB | 未定义 | Executor | body 不消费 | Executor | 核心 body 中无读写，不能视为 UB backing。 |
| 标量成员 | 每核私有 | `minValue/maxValue` 各 1 个 T | T/register归属固定仓不可见 | `GetValue(0)` | `Maxs/Mins` | Kernel 实例生命周期 | 不分配 UB Queue。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueueX · VECIN · depth=2 · 2×ubFormer×sizeof(T)"]:2
  outQ["outQueue · VECOUT · depth=2 · 2×ubFormer×sizeof(T)"]:2
```

##### 字节账与复用顺序

```text
inQueueX = 2 * ubFormer * sizeof(T)
outQueue = 2 * ubFormer * sizeof(T)
UB_total = 4 * ubFormer * sizeof(T)
ubFormerBytes = floor((UB_size / 2 / 2) / 128) * 128
UB_total = 4 * ubFormerBytes <= UB_size
dynamic_ub = N/A（AIV 静态 Queue）
workspace_total(core tiling) = 16 * 1024 * 1024 bytes；body 使用 0 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueueX` 当前 slot | `CopyIn/DataCopyPad` | `Compute/Maxs` | `FreeTensor` 后的后续 tile | `EnQue -> DeQue -> FreeTensor` | 未消费前不得复用；2 slots由 Queue 管理。 |
| `outQueue` 当前 slot | `Compute/Maxs` | `Compute/Mins` 与 `CopyOut` | `FreeTensor` 后的后续 tile | 同一 LocalTensor 顺序执行；随后 `EnQue -> DeQue -> FreeTensor` | `Mins(dst=src=yLocal)` 必须支持原地；CopyOut 前不得释放。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VECIN | `DataCopyPad` + Queue | `CopyIn` | `Compute::DeQue` | 当前 x slot | `blockCount=1`、stride=0、无左右 padding；`blockLen` 是实际字节数。 |
| 下界裁剪 | `Maxs(yLocal,xLocal,minValue,len)` | x Queue + 每核 scalar | 后续 `Mins` | yLocal 有效前 `len` 元素 | dtype T 原精度；无显式 cast/round；tail 传 `ubTail`。 |
| 上界恒等裁剪 | `Mins(yLocal,yLocal,maxValue,len)` | `Maxs` | out Queue | 同一 yLocal | 原地；max 是 dtype 正无穷/整数最大值。浮点 NaN primitive 语义固定仓不可见。 |
| VECOUT→GM | Queue + `DataCopyPad` | `Compute::EnQue` | `CopyOut::DeQue` | 当前 y slot | 只搬 `len*sizeof(T)`；最后核最后 tile 使用 `ubTail`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `DetermineSigDimMode/CalcInitialUbParams/CalcBlockParams/AdjustForLowUtilization/PostTiling` | Scalar-Scalar key、双缓冲 UB 公式、分核、低利用率重切与 16 MiB workspace。 |
| Kernel entry/router | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` key `2001100` | AIV-only guard、POD解析和 `ClipByValueV2Ass<DTYPE_X>` 实例。 |
| Device body/helper | `conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h` | `Init/Process/CopyIn/Compute/CopyOut` | GM offset、双 Queue、scalar直读、Maxs/Mins顺序、tail copy。 |

### Route R-002：AICPU Scalar

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp` | `ClampPromoteType/aclnnClampCommon` | 非空；promotion/cast检查成功；生成两个 scalar Tensor | 调用 l0。 |
| l0 selector | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `IsAiCoreSupport/ClipByValueV2AiCpu` | 计算 dtype 不在 AICore 五类型集合 | 加入 `ClipByValueV2` AICPU task。 |
| AICPU check | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `Compute/GetInputAndCheck` | 3 input/1 output；dtype一致；x/y numel一致；min/max同时为 scalar或均与x同shape | 本接口命中 scalar branch。 |
| dtype router | 同上 | `DispatchByDtype` | switch 到 `DoCompute<T>`；公开路线主要 DOUBLE/INT8/UINT8/INT16 | 不支持则 `KERNEL_STATUS_PARAM_INVALID`。 |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICPU(ClipByValueV2, x, minTensor, maxTensor -> y)
  -> REGISTER_CPU_KERNEL("ClipByValueV2", ClipByValueV2CpuKernel)
    -> ClipByValueV2CpuKernel::Compute(ctx)
      -> NormalCheck -> GetInputAndCheck
      -> DispatchByDtype(dtype)
        -> DoCompute<T>
          -> data_num==1 ? ClampScalar<T>
          -> scalar bounds: preload mn_ptr[0]/mx_ptr[0]
             -> DispatchParallel
                -> serial KernelScalarBound<T>
                -> or CpuKernelUtils::ParallelFor -> KernelScalarBound<T>
```

##### 源码等价伪码

```text
# 伪码
check input_count=3, output_count=1
check dtype(x)==dtype(min)==dtype(max)==dtype(y)
check numel(x)==numel(y)
check (numel(min)==1 && numel(max)==1) or shapes all equal
if numel(x)==0: return OK

T* x; T* y; T minValue=min[0]; T maxValue=max[0]
if numel(x)==1:
    y[0] = ClampScalar(x[0], minValue, maxValue); return OK

body(b,e):
    for i in [b,e): y[i] = ClampScalar(x[i], minValue, maxValue)
if numel(x)*sizeof(T) < 200KiB:
    body(0,numel)
else:
    cores=max(1,max(GetCPUNum(),2)-2)
    grain=CeilMultiple(numel,cores)*sizeof(T)
    ParallelFor(ctx,numel,grain,body)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM/host-visible task buffer | `x_ptr+i` | `numel(x)` | T/线性 | Executor task input | `KernelScalarBound` | Executor/task lifetime | 无显式对齐要求。 |
| min/max task buffer | element 0 | 各 1 T | T/scalar | Executor Tensor | `DoCompute` 预装局部值 | Executor/task lifetime | 后续循环不再读边界 GM。 |
| y task buffer | `y_ptr+i` | `numel(x)` | T/线性 | `ClampScalar` | 后续 Cast/ViewCopy | Executor/task lifetime | 各并行 shard 写不相交 `[b,e)`。 |
| CPU 局部状态 | shard 的 `b/e/minValue/maxValue` | 常数级 | 标量 | `DoCompute/DispatchParallel` | shard lambda | 调用栈/lambda lifetime | 无共享可写中间缓冲。 |
| workspace/UB | N/A | 0（body可见） | N/A | N/A | N/A | N/A | AICPU framework task 描述符内存固定仓不可见。 |

##### 分层内存布局

N/A：该 Route 是 AICPU task，不分配 AIV UB/TQue/TBuf；实际替代路径是 task 输入/输出指针加 CPU 栈上标量与 shard 范围。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total(core body) = 0 visible bytes
task descriptor / framework staging = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 无共享中间子区 | N/A | N/A | N/A | `ParallelFor` 完成后 `DoCompute` 返回 | 各 shard 的 `[b,e)` 必须互不重叠；由 `ParallelFor` 划分。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 串行 | 普通 for loop | `DoCompute` | 函数返回 | 整个 y | 逐元素，无 padding/tail lane。 |
| 并行 | `CpuKernelUtils::ParallelFor` | 多 shard lambda | `ParallelFor` 返回 | 不相交 y ranges | shard 最后一段由 `[b,e)` 自然保护。 |
| 浮点 clamp | `ClampScalar` | x/min/max | y store | 单元素 | 标准 float/double 检查任一 NaN 后返回 quiet NaN。 |
| half/bfloat clamp | float 临时转换 | Eigen half/bfloat输入 | cast 回 T | 单元素 | 在 float 中检查 NaN并完成 min/max，再降回 T。 |
| 整数 clamp | `std::min(std::max(...))` | 整数输入 | y store | 单元素 | 原 dtype，无溢出算术。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `ClipByValueV2AiCpu` | 分配输出并把三输入/一输出加入 AICPU launcher；无 Tiling。 |
| Kernel entry/router | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `REGISTER_CPU_KERNEL/Compute/DispatchByDtype` | task 注册、参数检查与 dtype switch。 |
| Device body/helper | 同上 | `DoCompute/DispatchParallel/KernelScalarBound/ClampScalar` | scalar预装、串并行阈值、逐元素语义与 NaN处理。 |

## 算子特定优化

### Trick T-001：Scalar 边界每核只读一次并使用 Scalar Vector API

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：priority 1 合轴单维，`minDim==1 && maxDim==1`，key `2001100`。
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h:84-94`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h#L84-L94)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
minGm.SetGlobalBuffer((__gm__ T*)min);
maxGm.SetGlobalBuffer((__gm__ T*)max);
minValue = minGm.GetValue(0);
maxValue = maxGm.GetValue(0);
pipe->InitBuffer(inQueueX, bufferNum, tilingData->ubFormer * sizeof(T));
pipe->InitBuffer(outQueue, bufferNum, tilingData->ubFormer * sizeof(T));
```

#### 机制与设计原因

边界 Tensor 在 API 层已确定为单元素；ASS body 因而不为 min/max 分配 UB Queue，也不在 tile 循环中重复 DMA。每个 AIV 核只通过 `GetValue(0)` 取得 `minValue/maxValue`，随后 `Maxs/Mins` 把标量广播到当前 `len` 个 lane。相较通用 Tensor broadcast，这直接消除两路边界 tile 的 GM→UB 搬运与 UB 容量。

#### 迁移边界

- 适用：输入/输出线性域相同、上下界均为 scalar 的逐元素 clamp。
- 前提：950 AIV；计算 dtype有 Scalar Vector primitive；边界在整个 Kernel 生命周期不变。
- 失效：任一边界为多元素 Tensor 时不能复用 ASS，须转 single-dim非ASS、UB broadcast或通用 broadcast实现；这些 Tensor 变体不属于本卡。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的真实反例边界是 `ClipByValueV2SigDim`/通用 broadcast 路径：把多元素边界误当 scalar 只读 `[0]` 会错误广播，不能套用 ASS。

### Trick T-002：输出 Queue 内原地完成第二次裁剪

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：ASS `Compute(len)` 的所有 tile，包括尾 tile。
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h:114-123`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h#L114-L123)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
auto xLocal = inQueueX.template DeQue<T>();
auto yLocal = outQueue.template AllocTensor<T>();
Maxs(yLocal, xLocal, minValue, len);
inQueueX.FreeTensor(xLocal);
Mins(yLocal, yLocal, maxValue, len);
outQueue.EnQue(yLocal);
```

#### 机制与设计原因

`Maxs` 直接把下界裁剪结果写入最终 output Queue slot；`Mins` 以同一 `yLocal` 同时作为 src/dst 原地覆盖，没有单独的中间 TBuf。于是每个 tile 的 UB 只包含双缓冲 input/output Queue，总账为 `4*ubFormer*sizeof(T)`。

#### 迁移边界

- 适用：第二个逐元素 Vector primitive支持 dst/src alias，且第一个结果不再被其他消费者使用。
- 前提：严格保持 `Maxs -> Mins -> EnQue` 顺序；`len` 在两次调用中一致。
- 失效：若后续阶段仍需未裁上界的 Maxs 结果，必须分配独立 buffer或在消费完成后再覆盖。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

提前 `EnQue` 或在 `Mins` 前释放/复用 `yLocal` 会破坏同一 slot 的顺序；源码以同一局部 Tensor 的连续调用规避该问题。

### Trick T-003：AICPU scalar 分支预装边界并按字节阈值切换并行

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：`min/max` 均为单元素；`data_num>1`；是否并行由 `data_num*sizeof(T) >= 200 KiB` 决定。
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp:163-178`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp#L163-L178)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (is_scalar) {
    const T mn_v = mn_ptr[0];
    const T mx_v = mx_ptr[0];
    auto body = [x_ptr, y_ptr, mn_v, mx_v](int64_t b, int64_t e) {
        KernelScalarBound<T>(x_ptr, y_ptr, b, e, mn_v, mx_v);
    };
    return DispatchParallel(ctx, "real-scalar", data_num, elem_bytes, kParallelBytesThresh, body);
}
```

#### 机制与设计原因

边界先复制到 `mn_v/mx_v` 并按值捕获到 shard lambda，逐元素循环只流式读取 x、写 y。`DispatchParallel` 对小于 200 KiB 的实数数据直接串行，较大数据才调用 `ParallelFor`，从源码层面避免小工作量的并行调度并保留大工作量分片。

#### 迁移边界

- 适用：AICPU逐元素标量边界计算。
- 前提：边界在任务期间不变；每个 shard 写不相交输出范围；阈值按真实 `sizeof(T)` 计算。
- 失效：Tensor 边界需要逐元素读取，源码转 `KernelElemBound`；复杂类型使用另一 40 KiB 阈值，但公开 ClampMin dtype不含 complex。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的真实反例是非 scalar bounds：若仍捕获 `[0]`，会把第一个边界错误广播；`DoCompute` 在该条件下改走 `KernelElemBound`。

## 模板与实现对比

| 对比维度 | R-001 AIV Scalar-Scalar | R-002 AICPU Scalar | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | l0 五 dtype集合；Tiling priority 1、key `2001100` | l0 dtype集合之外 | selector 先按计算 dtype 决定执行引擎；AICore 内再按 shape模板。 |
| 模板与 dtype | `ClipByValueV2Ass<DTYPE_X>`；F16/F32/BF16/I32/I64 | `DoCompute<T>`；公开可达主要 DOUBLE/I8/U8/I16 | 950 AICore OpDef/binary只列五种；其余交给 AICPU。 |
| 循环与分核 | Host按 UB tile和AIV核数分块；每核多 tile | CPU单元素快路；串行或 ParallelFor shard | AIV围绕 UB 容量，AICPU围绕总字节阈值。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | VECIN/VECOUT各depth 2；无TBuf；两个scalar成员 | 无UB/Queue；CPU栈与lambda scalar | 两者都避免边界 Tensor tile物化。 |
| 内存总占用与复用 | `4*ubFormer*sizeof(T)<=UB`；yLocal原地 Mins | body无中间数组；直接x→y | 元素独立，无需 workspace中间结果。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + Maxs/Mins | 普通CPU load/store + std::min/max | 无 SIMT/NDDMA/通信/Cube。 |
| 同步、精度与 tail | Queue协议；T原精度；ubTail | ParallelFor返回同步；half/bfloat升到float；范围端点自然保护 | 执行引擎决定同步与数值实现。 |
| fallback/失败边界 | Tiling/launch失败即失败，不转AICPU；未知key return | 参数/dtype/ParallelFor失败返回rc | AICPU只由预先 dtype selector选中，不是AICore运行时 fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档标支持，但固定仓无 A2 `ClipByValueV2` AICore config/binary/build，核心 dtype仍会被 l0 selector送往 AICore；平台链路冲突，未生成正式卡。 | 公开文档标支持，但固定仓无 A3 `ClipByValueV2` AICore config/binary/build，核心 dtype仍会被 l0 selector送往 AICore；平台链路冲突，未生成正式卡。 | 文档支持；OpDef和CMake均显式 `ascend950`，正式支持。 |
| Host/Launcher | 非 RegBase 分支把 `promoteType=out.dtype`；后续实际 A2 launcher闭包固定仓缺失。 | 非 RegBase 分支把 `promoteType=out.dtype`；后续实际 A3 launcher闭包固定仓缺失。 | RegBase 分支执行 Scalar promotion，核心结果再 Cast到out。 |
| Kernel/资源 | 固定仓不可见。 | 固定仓不可见。 | arch35 AIV ASS + AICPU dtype fallback；AIV每核平台UB和AIV核数驱动tiling。 |

## 限制与失败边界

- `self/out/clipValueMin` 空指针、dtype不支持、shape不一致、rank超限、promotion失败或不能 cast 到 `out` 时，第一段返回错误；空 `self` 是成功短路。
- AICore selector只看计算 dtype。选中 AICore 后，Tiling失败、launcher加入失败或运行失败不会再转 AICPU。
- ASS key依赖外部 `Ops::Base::DimensionCollapse` 的结果；固定仓显示其输入、输出与消费谓词，但 `atvoss/broadcast/broadcast_tiling.h` 的实现不在该仓。Scalar→单元素 Tensor与通用合轴按框架边界处理，不伪造其内部 ABI。
- `TILING_DATA_FIELD_DEF` 宏实现不在固定仓，因此 POD 的字段类型/producer/consumer已闭合，但 `sizeof/alignof/offsetof` 固定仓不可见。
- `Contiguous`、`Cast`、`ConvertToTensor`、`ViewCopy`、`CommonOpExecutorRun` 与 AICPU task封装是明确的 ACLNN 框架边界；其实际 Kernel、workspace切片、event与清理实现固定仓不可见。
- AIV浮点 `Maxs/Mins` 对 NaN 的硬件细节固定仓不可见；AICPU源码明确传播 NaN。不得据此假设两 Route 在所有 NaN payload 上位级一致。
- 公开文档声明 A2/A3 支持，但固定 commit 的 OpDef与CMake只接 `ascend950`，因此本次不创建无后缀 A2/A3 卡。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/clip_by_value_v2/docs/aclnnClampMin.md` | 产品支持表、函数原型、参数表 | 950公开支持、接口语义、dtype/shape/layout与两段式调用。 |
| 入口声明/约束 | `conversion/clip_by_value_v2/op_api/aclnn_clamp.h` | `aclnnClampMinGetWorkspaceSize/aclnnClampMin` | C ABI声明。 |
| 入口实现 | `conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp` | `ClampPromoteType/aclnnClampCommon/aclnnClampMin*` | 校验、空域、promotion、Scalar正规化、Executor图、workspace与执行。 |
| Host selector | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `IsAiCoreSupport/ClipByValueV2AiCore/ClipByValueV2AiCpu/ClipByValueV2` | dtype→AICore/AICPU分发与launcher节点。 |
| shape infer | `conversion/clip_by_value_v2/op_host/clip_by_value_v2_infershape.cpp` | `InferShape4ClipByValueV2` | 三输入 broadcast shape infer。 |
| 950 OpDef | `conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp` | `ClipByValueV2` | 五种 AICore dtype、ND与ascend950配置。 |
| Tiling入口 | `conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_tiling.cpp` | `TilingFor/PrepareForClipByValueV2` | AIV核数、UB大小和模板registry。 |
| Tiling注册 | `conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_single_dim_tiling.cpp` | `REGISTER_OPS_TILING_TEMPLATE` | 把共享 SingleDim 模板以 priority 1 注册给 `ClipByValueV2`。 |
| POD定义 | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.h` | `ClipByValueSigDimTilingData` | 九字段POD与key注册。 |
| Tiling实现 | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `ClipByValueTilingSingleDim` | ASS谓词、分核/UB/workspace公式和POD生产。 |
| Tiling分发 | `common/inc/op_host/math_tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority按map升序尝试与失败终点。 |
| Kernel entry | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | AIV-only guard、key router与模板实例。 |
| Kernel body | `conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h` | `ClipByValueV2Ass` | GM/UB、Queue、Vector、同步和tail闭包。 |
| AICPU注册 | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu_def.cpp` | `ClipByValueV2/AICPU` | AICPU dtype定义与默认配置。 |
| AICPU body | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `ClipByValueV2CpuKernel`及helper | 参数检查、dtype dispatch、串并行、精度与tail。 |
| binary配置 | `conversion/clip_by_value_v2/op_host/config/ascend950/clip_by_value_v2_binary.json` | 五个 `op_list` 条目 | 950 AICore F16/BF16/F32/I32/I64 二进制实例。 |
| 构建接线 | `conversion/clip_by_value_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR/add_all_modules_sources` | ascend950、arch35与clip_by_value依赖。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | ASS九个int64字段；R-002无Tiling POD | 字段顺序、类型、单位与inactive字段保持；精确offset不可臆测 | `ClipByValueSigDimTilingData` |
| 调用顺序 | Contiguous→Cast→边界Tensor→核心→Cast→ViewCopy | 950结果Cast不可移到核心前；任一节点失败阻断后续 | `aclnnClampCommon` |
| workspace/内存 | ASS双Queue总账、核心workspace未读；AICPU直接x→y | Queue depth 2必须计入；16 MiB请求不能误作body中间数据 | `CalcInitialUbParams/GetWorkspaceSize/ClipByValueV2Ass::Init` |
| 同步 | AIV Queue生命周期；AICPU ParallelFor完成后返回 | EnQue/DeQue/Free顺序和不相交shard范围不得改变 | `CopyIn/Compute/CopyOut/DispatchParallel` |
| 精度与尾块 | AIV按T原精度且显式ubTail；AICPU half/bfloat升float、NaN传播 | 不得外推AIV NaN细节；copy和compute的len必须一致 | `Process/ClampScalar/KernelScalarBound` |
| 失败边界 | Host空域成功；AICore/AICPU预分流；各自失败直接返回 | 不得实现成AICore运行失败后AICPU fallback；未知key当前无输出 | `aclnnClampCommon/ClipByValueV2/clip_by_value_v2/Compute` |
