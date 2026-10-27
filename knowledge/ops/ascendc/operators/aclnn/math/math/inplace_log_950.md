---
type: Operator
title: "InplaceLog 950 ACLNN 算子知识"
description: "说明 aclnnInplaceLog 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/docs/aclnnLog&aclnnInplaceLog.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/docs/zh/install/compile.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/aclnn_log.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/aclnn_log.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/log.cpp
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/aicpu/aicpu_op_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/config/ascend950/log_binary.json
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch_with_scalar.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
created_at: "2026-08-25T11:18:34Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceLog 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceLogGetWorkspaceSize` + `aclnnInplaceLog`，公开两段式 ACLNN API；固定仓未提供更高版本入口。 |
| 计算语义 | 对 `selfRef` 的每个元素计算自然对数，并把结果写回同一逻辑张量：`selfRef[i] = ln(selfRef[i])`。核心 `Log` 先写 executor 临时输出，最后由 `ViewCopy` 回写，因此不是 Device body 对输入 GM 的直接覆盖。 |
| 输入输出 | 一个输入/输出别名参数 `selfRef`，ND、1～8 维，支持 FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128；允许非连续 Tensor。 |
| 执行类型 | R-001/R-002：单个算子拥有的 `AIV_VECTOR` Log 节点；R-003/R-004：单个 AICPU 标量循环/分片节点。外围 `Contiguous`、同 dtype `Cast`、`ViewCopy` 属 ACLNN 框架节点。所有可达节点均无 Cube/AIC 或 `MIX_CV`。 |
| Kernel 路线 | R-001 FP32 AIV 直接 Log；R-002 FP16/BF16 AIV 升至 FP32 后 Log；R-003 DOUBLE/COMPLEX AICPU 串行；R-004 DOUBLE/COMPLEX AICPU 并行分片。空 Tensor 在 Host 提前成功返回。 |
| 核心资源 | AIV：GM、4 个按 tile 容量计的 ping/pong UB 子区、MicroAPI RegTensor、固定 16 MiB Log 节点 workspace；AICPU：输入/输出 GM、CPU 寄存器/线程分片、OpDef 声明的 100 字节节点 workspace。executor 还持有外围节点与临时 Tensor 的 workspace。 |
| 来源闭包 | `source_closed`：公开入口、L0 dtype 分发、950 Host/Tiling/POD、AIV entry/DAG/MicroAPI、AICPU body、直接 atvoss 调度依赖和构建接线均已闭合；ACLNN executor、`Contiguous/Cast/ViewCopy` 与 AICPU `ParallelFor` 为明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令进入核心 Log 节点的连续张量为 `x`，元素数为 `N=prod(storage_shape)`。公开入口固定传入 `base=-1.0f`、`scale=1.0f`、`shift=0.0f`，故通用公式 `log(shift + scale*x)/log(base)` 化为 `y[i]=ln(x[i])`。实数输入未做正域检查，异常值行为由 AIV `MicroAPI::Log` 或 AICPU `std::log` 决定；复数由 AICPU `std::log(std::complex<T>)` 计算。最后 `ViewCopy(y,selfRef)` 恢复 `selfRef` 的原 view/stride 并形成可观察的原地语义。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；1～8 维；可非连续；输入与输出是同一个 `aclTensor` | FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128 | 不得为空指针；dtype 必须命中目标平台 output support list。空 Tensor 返回 workspace 0 且不建 Device 任务。 |
| `workspaceSize` | 标量 Host 输出 | `uint64_t` | 由 executor 汇总计算图所需 Device workspace；调用者必须把该值原样传给第二段接口。 |
| `executor` | Host 对象输出 | `aclOpExecutor*` | 持有 `Contiguous -> Log -> Cast -> ViewCopy` 流程；第一段转移所有权，第二段消费。 |
| `workspace` | Device 地址 | byte buffer | 由调用者按第一段返回大小分配；Log AIV tiling 自身声明 16 MiB，AICPU OpDef 声明 100 字节，外围节点和临时输出由 executor 规划。 |
| `stream` | runtime stream | `aclrtStream` | 第二段在该 stream 上执行 executor 图。 |

### 入口与版本

公开声明和文档只给出 `aclnnInplaceLogGetWorkspaceSize`/`aclnnInplaceLog` 这一版本族。第一段先执行 `CheckInplace`，随后复用 `aclnnLogGetWorkspaceSize(selfRef,selfRef,...)`；这只是流程复用，不改变 InplaceLog 的逻辑身份。950 由公开产品表的 Ascend 950PR/950DT 支持项、`Log` 的 `ascend950` AICore config、`SUPPORT_COMPUTE_UNIT "ascend950"` 和 ascend950 binary config 共同证明。Log2、Log10、非原地 `aclnnLog` 不属于本卡。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckInplace(selfRef)` 依次检查非空、dtype support list 和 `rank<=8`；失败返回 `ACLNN_ERR_PARAM_NULLPTR` 或 `ACLNN_ERR_PARAM_INVALID`。
2. 复用 `aclnnLogGetWorkspaceSize(selfRef,selfRef,...)` 后，`CheckParams` 再检查非空、1-in/1-out dtype、可转换性和 shape 相等；别名参数使 shape 相等，但检查仍真实执行。
3. 创建 executor；非 ND storage format 只记录 warning，不拒绝。
4. `selfRef->IsEmpty()` 时令 `workspaceSize=0`、释放 executor 所有权并成功返回，无 Log、Cast 或 ViewCopy 节点。
5. 非空时调用框架 `l0op::Contiguous` 生成连续输入；固定属性调用 `l0op::Log`；随后请求转为 `selfRef` dtype 的 `l0op::Cast`，最后 `l0op::ViewCopy` 回写原 view。任一返回空指针立即以内部错误终止，后续节点不加入 executor。

### 参数、Workspace 与生命周期

`l0op::Log` 先按输入 view shape 分配独立输出。FLOAT/FLOAT16/BFLOAT16 命中 `LogAiCore`；DOUBLE/COMPLEX64/COMPLEX128 命中 `LogAiCpu`。AIV tiling 将 `currentWorkspace[0]` 固定为 `16*1024*1024` 字节，Kernel entry 虽接收 `workspace` 却不读取；AICPU OpDef 的默认配置声明 `opInfo.workspaceSize="100"`。公开第一段不把这些值直接返回，而是调用 `uniqueExecutor->GetWorkspaceSize()`，因此 executor 临时 Tensor、外围框架节点及对齐后的精确总和属于 executor 框架边界，固定算子源码未展开其切片 offset。

连续输入、Log 临时输出、可能的 Cast 结果只在 executor 运行期间有效；`ViewCopy` 完成前不得覆盖或释放。第一段把 executor 从 `uniqueExecutor` 转移给调用者；第二段 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)` 负责实际启动及失败传播。

### 启动图与失败传播

```text
aclnnInplaceLogGetWorkspaceSize(selfRef)
  -> [empty] workspaceSize=0, no Device node
  -> [non-empty] F-001 Contiguous(selfRef)              # 框架节点，可按连续性消除
      -> K-001 Log(FLOAT/FLOAT16/BFLOAT16): AIV_VECTOR  # R-001/R-002
         or K-002 Log(DOUBLE/COMPLEX): AICPU            # R-003/R-004
      -> F-002 Cast(logOut, selfRef.dtype)               # 同 dtype，是否消除由框架决定
      -> F-003 ViewCopy(castOut, selfRef)                # 形成原地可见结果
aclnnInplaceLog(workspace, executor, stream)
  -> CommonOpExecutorRun 按 executor 依赖在同一指定 stream 上提交
```

K-001 的 launcher 是 `ADD_TO_LAUNCHER_LIST_AICORE(Log, self, out, base,scale,shift)`，entry 为 `log<schMode=1,attrWork=1>`，`blockDim=blockNum`，POD 为 `EleBaseTilingData32B`。K-002 由 `ADD_TO_LAUNCHER_LIST_AICPU` 传递同三个属性到 `LogCpuKernel::Compute`；AICPU grid/thread ABI 由框架隐藏。F-001/F-002/F-003 的具体 kernel、workspace slice、事件实现是固定仓明确允许的框架边界。节点添加失败返回 `nullptr` 并阻断后续图构造；运行期失败由 `CommonOpExecutorRun` 返回。

## Route 与参数设计

### 分核与资源计算

AIV 路线将 shape 合轴为 `N`。设平台 AIV 数为 `P`、UB 字节数为 `U`、元素字节数为 `s`，自然 Log DAG 的 `MaxDtypeBytes=MinDtypeBytes=s`、`BufferNum=4`、`extraSize=0`：

```text
candidateCore = ceil(N * (8s) / 32768)          # 每核至少约 4 KiB 有效数据
coreNum       = min(candidateCore, P)
blockFormer   = align_up(ceil(N/coreNum), 512)  # 512 的单位是元素
blockNum      = ceil(N/blockFormer)
blockTail     = N - (blockNum-1)*blockFormer
alignFactor   = 256/s                           # 元素；一个 256-byte repeat
ubFormer      = align_down(floor(U/(4s)), alignFactor)
loopFormer    = ceil(blockFormer/ubFormer)
loopTail      = ceil(blockTail/ubFormer)
```

非空域保证 `coreNum>0`；`ubFormer==0`、平台 core 数为 0、shape 含 0 或 UB 信息缺失均使 tiling 失败。Device 从紧凑 POD 的 `N/coreNum/ubFormer` 重算 block 和 loop 参数。AICPU 不用该 POD：DOUBLE 在 `N>60*1024` 时并行，COMPLEX64/128 在 `N>18*1024` 时并行；否则单 shard。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 AIV | `GET_TPL_TILING_KEY(1,1)`；`LogScaleOneShiftZeroLnbaseOne<float>` | 非空且 dtype=FLOAT；`IsAiCoreSupport` 命中；固定属性使 `attrWork=1` | dtype 为 FP16/BF16 到 R-002；否则到 AICPU |
| 2 | R-002 FP16/BF16 AIV | 同 key；模板 `half` 或 `bfloat16_t` | 非空且 dtype∈{FLOAT16,BFLOAT16} | DOUBLE/COMPLEX 到 AICPU |
| 3 | R-003 AICPU 串行 | `LogCompute<T,T>` | dtype=DOUBLE 且 `N<=61440`，或 dtype∈{COMPLEX64,COMPLEX128} 且 `N<=18432` | 超阈值到 R-004 |
| 4 | R-004 AICPU 并行 | `ParallelFor(..., shard_log)` + `LogCompute<T,T>` | dtype=DOUBLE 且 `N>61440`，或 complex 且 `N>18432` | 无；框架/计算错误返回失败 |

```text
if selfRef is empty: Host success terminal
else if dtype in {FLOAT,FLOAT16,BFLOAT16}:
    attrWork = (scale≈1 && shift≈0 && invLnBase==1) ? 1 : 2
    # aclnnInplaceLog 固定属性，故只可达 attrWork=1
    if dtype == FLOAT: R-001
    else: R-002
else: # CheckInplace 已把集合限制为 DOUBLE/COMPLEX64/COMPLEX128
    if NeedParallel(N,dtype): R-004
    else: R-003
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | key `(schMode=1,attrWork=1)` | 950、非空、FLOAT、同 shape/dtype | 单 Log AIV 节点，外围框架节点前后包围 | `ElementwiseSchWithScalar<EleBaseTilingData32B,1,LogScaleOneShiftZeroLnbaseOne<float>::OpDag>`；`LogCustom<float>` | 2×input + 2×output UB；MTE2/V/MTE3 mutex；FP32 RegTensor | `UpdateMask` + 有效字节 CopyOut；tiling 失败无 Device launch |
| R-002 | 同 key | 950、非空、FLOAT16 或 BFLOAT16 | 同上 | 相同 scheduler；`LogCustom<half/bfloat16_t>` | 同四区 ping/pong；B16 unpack→FP32 Log→RINT cast→pack | mask 保护末向量；CopyOut 只写 tile 有效字节 |
| R-003 | `IsAiCoreSupport=false`、`NeedParallel=false` | 950、非空、DOUBLE/COMPLEX 且各自阈值内 | 单 AICPU 节点 | `LogCompute<double,double>` 或 complex 实例；`ComputeNaturalLog` | 无 AIV UB；单线程范围 `[0,N)`；输入输出独立 | `for index<end` 精确覆盖；异常由标准库语义承接 |
| R-004 | 同 selector、`NeedParallel=true` | 950、非空、DOUBLE/COMPLEX 且超过阈值 | 单 AICPU 节点，内部多 shard | 同模板；`CpuKernelUtils::ParallelFor` | 独立不重叠 `[start,end)`；框架完成同步 | 每 shard 以 `index<end` 守尾；ParallelFor 错误返回 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域 | `self->IsEmpty()` | elewise tiling 明确拒绝 0 维元素，API 提前终止 | 全部 | workspace 0 成功终点 | `aclnn_log.cpp::aclnnLogGetWorkspaceSize` |
| AICore dtype | `CheckType(dtype,{FLOAT,FLOAT16,BF16})` | 950 binary config 仅实例化三种 AIV dtype | R-001/R-002 | R-003/R-004 AICPU | `log.cpp::IsAiCoreSupport` |
| fast attr | `scale≈1 && shift≈0 && abs(invLnBase-1)<epsilon` | 省去 Cast/Muls/Adds/末乘 DAG；InplaceLog 固定满足 | R-001/R-002 | 通用 `attrWork=2` 对本 API 不可达 | `LogTiling::SetAttr` |
| 每核数据下限 | `candidateCore=ceil(N*minBits/32768)` 并 cap 到平台 AIV 数 | 避免用过多核处理小于约 4 KiB 的元素域 | R-001/R-002 | 无替代；得到较小 `blockDim` | `ElewiseTiling` |
| UB 上界 | `ubFormer=align_down((U*8)/(4*maxBits),256*8/minBits)>0` | 四个 ping/pong 子区且 tile 对齐到 repeat | R-001/R-002 | tiling 失败 | `GetMaxElemNum`、`ElewiseTiling` |
| AICPU 并行阈值 | DOUBLE `N>61440`；COMPLEX `N>18432` | 大元素域才支付 ParallelFor 分片开销 | R-003/R-004 | 阈值内串行，阈值外并行 | `NeedParallel` |
| rank | `rank<=8` | 公开 Inplace API 约束 | 全部 | `ACLNN_ERR_PARAM_INVALID` | `CheckInplace` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `EleBaseTilingData32B` | `dim0` | `int64_t`/64 | `sizeof=32, alignof=8, offsetof=0` | 元素 | `AdaptEleBaseTilingData32B`: `dim0=N` | scheduler ctor 读取后重算 block/loop | 非空，乘积使用 `int64_t`，源码未加溢出检查 |
| `EleBaseTilingData32B` | `coreNum` | `int32_t`/32 | `offsetof=8` | AIV core | `coreNum=static_cast<int32_t>(min(candidate,P))` | scheduler ctor；参与 `ceil(N/coreNum)` | 平台 core 数为 0 时 tiling 失败 |
| `EleBaseTilingData32B` | `ubFormer` | `int32_t`/32 | `offsetof=12` | 元素/tile | 从 UB、4 buffers、dtype 位宽与 repeat 对齐推导后窄化 | `blockLen=ubFormer*MaxDtypeBytes`、tile loop | `ubFormer==0` 失败；窄化无额外 guard |
| `EleBaseTilingData32B` | `scalarData` | `char[16]`/128 | `offsetof=16` | byte payload | `SetScalar<float>` 依次写 scale=1、shift=0、invLnBase=1 到 offset 0/4/8 | fast DAG `VarSize=0`，R-001/R-002 不读取 | 最后 4 byte 固定仓未证明初始化；本 Route inactive |
| AICPU `CpuKernelContext` attr | `base` | float/32 | 框架序列化 ABI/offset 固定仓不可见 | 标量 | launcher `OP_ATTR(-1,1,0)`；`LogOpCheck` 默认 -1 | `NaturalLogCondition` | 必须为 -1 或正且非 1；本 API 固定 -1 |
| AICPU `CpuKernelContext` attr | `scale` | float/32 | 同上 | 标量 | launcher 传 1；缺省也为 1 | `NaturalLogCondition` | 本 Route 固定 1 |
| AICPU `CpuKernelContext` attr | `shift` | float/32 | 同上 | 标量 | launcher 传 0；缺省也为 0 | `NaturalLogCondition` | 本 Route 固定 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `(1,1)` + FLOAT | `LogTiling::RunTiling`、`ElewiseTiling` | `EleBaseTilingData32B` | `log -> LogKernel<1,1,float>` | `LogScaleOneShiftZeroLnbaseOne<float>`、`LogCustom<float>`、atvoss scheduler/Vec/sync | `Log` ascend950 AICore config、CMake、FLOAT binary |
| R-002 | `(1,1)` + FLOAT16/BF16 | 同上 | 同上 | `LogKernel<1,1,half/bfloat16_t>` | B16 branch of `LogCustom` | FLOAT16/BFLOAT16 binary |
| R-003 | AICPU selector + `!NeedParallel` | `l0op::Log`、`LogOpCheck` | context attrs | `REGISTER_CPU_KERNEL("Log",LogCpuKernel)` | `LogCompute`、`ComputeNaturalLog` | `log_aicpu_def.cpp` + CMake AICPU source collection |
| R-004 | AICPU selector + `NeedParallel` | 同上 | context attrs + `N` | 同上 | `CpuKernelUtils::ParallelFor` + shard body | 同上 |

## Kernel 实现

### 公共执行流程

四条 Route 都接收 `Contiguous` 结果并写 executor 分配的独立 Log 输出，随后由框架 Cast/ViewCopy 回写 `selfRef`。AIV entry 参数顺序固定为 `(x,y,workspace,tiling)`，显式声明 `KERNEL_TYPE_AIV_ONLY`；workspace 参数不被 body 使用。AICPU entry 从 `CpuKernelContext` 取一输入一输出和三个属性，`NormalCheck`、`LogOpCheck` 任一失败即不计算。核心节点间没有算子自有 workspace 边；跨外围节点的 tensor offset/owner/order 由 executor 统一管理。

### Route R-001：FP32 AIV 直接自然对数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/log/op_api/aclnn_log.cpp` | `CheckInplace` | dtype FLOAT、非空、rank≤8 | 构造 Log 流程 |
| Launcher | `math/log/op_api/log.cpp` | `IsAiCoreSupport` | FLOAT 命中 AICORE list | `LogAiCore` |
| Tiling | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `SetAttr/RunTiling` | 固定属性→`attrWork=1`；output FLOAT | fast float DAG、key `(1,1)` |
| Router | `math/log/op_kernel/log_apt.cpp` | `LogKernel<1,1,float>` | `attrWork==1` | `ElementwiseSchWithScalar` fast body |

#### Device 数据流

##### 函数级调用链

```text
LogTiling::RunTiling -> DoTiling32B<LogScaleOneShiftZeroLnbaseOne<float>::OpDag>
Host Log AICore launch -> log<1,1>(x,y,workspace,tiling)
  -> LogKernel<1,1,float>
    -> ElementwiseSchWithScalar::Init(x,y)
    -> ElementwiseSchWithScalar::Process
      -> Run: Vec::CopyIn<float>
      -> RunNormalOp: LogCustom<float>
      -> Vec::CopyOut<float>
framework Cast/ViewCopy -> selfRef
```

##### 源码等价伪码

```text
# 伪码；每个 block 独占 [blockFormer*blockIdx, ...)
base = blockFormer * blockIdx
loops, tail = (last block ? tail-block values : former-block values)
for tile i in [0, loops):
    count = (i == loops-1 ? tail : ubFormer)
    pingpong = i & 1
    MTE2_lock(inBuf[pingpong]); DataCopyPad(inUB, x+base+i*ubFormer, count*4); MTE2_unlock
    V_lock(inBuf); V_lock(outBuf)
    for reg chunk of VL=VECTOR_REG_WIDTH/4 elements:
        mask = UpdateMask<float>(remaining count)
        load FP32 -> vregInput
        MicroAPI::Log(vregOutput,vregInput,mask)
        masked store FP32 -> outUB
    V_unlock(inBuf); V_unlock(outBuf)
    MTE3_lock(outBuf); DataCopyPad(y+base+i*ubFormer,outUB,count*4); MTE3_unlock
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `(blockFormer*blockIdx + tileOffset)*4` | `count*4` bytes | FP32/flat ND | Contiguous | CopyIn | executor，至 Log 完成 | blockFormer 512 元素对齐 |
| input UB ping/pong | id 0/2，`id*blockLen` | 各 `ubFormer*4` bytes | FP32 | MTE2 DataCopyPad | V load | DAG scheduler，交替 tile | depth 2；有效尾部仅 count |
| output UB ping/pong | id 1/3 | 各 `ubFormer*4` bytes | FP32 | masked RegTensor store | MTE3 CopyOut | DAG scheduler，交替 tile | Log 结果直连 output buffer，无临时物化 |
| RegTensor | 寄存器，无 UB offset | 每次一个 VL 的 input/output + mask | FP32 | MicroAPI load/Log | masked store | 单次 reg 循环 | tail mask zeroing/store guard |
| GM y | 与 x 相同元素 offset | `count*4` bytes | FP32 | CopyOut | ViewCopy | executor 临时 Tensor | 只提交有效 count |
| Log workspace | launcher slice offset 固定仓不可见 | 16 MiB | byte | executor | Kernel 参数未读取 | Log 节点期 | 不可据此与临时 Tensor alias |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inPing["input ping · ubFormer×4 B"]:1
  outPing["output ping · ubFormer×4 B"]:1
  inPong["input pong · ubFormer×4 B"]:1
  outPong["output pong · ubFormer×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*(ubFormer*4) + 2*(ubFormer*4) = 16*ubFormer bytes
ubFormer = align_down(floor(UB_platform/16), 64 elements)
UB_total <= UB_platform
dynamic_ub = N/A
workspace_total = executor_total；其中 Log AIV 节点声明 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input ping/pong | tile i CopyIn | tile i LogCustom | tile i+2 CopyIn | MTE2 unlock→V lock；V unlock 后同 id 可再取 | 只按 `i&1` 复用，V 消费结束前不覆盖 |
| output ping/pong | tile i LogCustom | tile i CopyOut | tile i+2 LogCustom | V unlock→MTE3 lock；MTE3 unlock 后复用 | CopyOut 完成前不覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `Mutex PIPE_MTE2` | DataCopyPad | V lock | input buffer id | blockCount=1、blockLen=`count*4`、stride=0、无 padding |
| UB compute | `Mutex PIPE_V` | input ready / Log store | input release、MTE3 | input/output ids | 全程 FP32；mask 只激活 `count` 内 lane |
| UB→GM | `Mutex PIPE_MTE3` | Log result | buffer reuse | output id | DataCopyPad 只写 `count*4` bytes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `LogTiling::RunTiling` | fast DAG、三标量、16 MiB、key、blockDim |
| Kernel entry/router | `math/log/op_kernel/log_apt.cpp` | `LogKernel<1,1,float>` | AIV-only、POD 解析、fast body |
| Device body/helper | `math/log/op_kernel/arch35/log_dag.h` | `LogCustom<float>` | FP32 load→Log→masked store |
| Direct dependency | `pkg_inc/op_common/atvoss/elewise/elewise_sch_with_scalar.h`、`pkg_inc/op_common/atvoss/util/vec.h`、`pkg_inc/op_common/atvoss/util/sync.h`（opbase） | `Process/Run/CopyIn/CopyOut` | 分核循环、ping/pong、DMA 与 mutex |

### Route R-002：FP16/BF16 升精度 AIV 自然对数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Launcher | `math/log/op_api/log.cpp` | `IsAiCoreSupport` | dtype FLOAT16 或 BFLOAT16 | AICore |
| Tiling | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `RunTiling` | output dtype 分支 + `attrWork=1` | half/bfloat16 fast DAG |
| Router | `math/log/op_kernel/log_apt.cpp` | `LogKernel<1,1,DTYPE_X>` | 编译期 dtype、attrWork=1 | R-002 |

#### Device 数据流

##### 函数级调用链

```text
log<1,1> -> LogKernel<1,1,half|bfloat16_t>
  -> ElementwiseSchWithScalar fast DAG
    -> Vec::CopyIn<U>
    -> LogCustom<U>: unpack B16 -> Cast<U→float> -> Log<float> -> Cast<float→U,RINT> -> pack
    -> Vec::CopyOut<U>
```

##### 源码等价伪码

```text
# 分核/tile/pingpong 与 R-001 相同，s=2
for each tile(count):
    DataCopyPad count*2 bytes into input UB
    for VL=VECTOR_REG_WIDTH/4 logical elements:
        mask = UpdateMask<float>(remaining count)
        DIST_UNPACK_B16 load U
        Cast<float,U, ZEROING>(vregInput, vregInput16, mask)
        Log(vregOutput, vregInput, mask)
        Cast<U,float, NO_SAT + CAST_RINT + ZEROING>(vregOutput16,vregOutput,mask)
        DIST_PACK_B32 masked store U to output UB
    DataCopyPad count*2 bytes to GM y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `(blockFormer*blockIdx+tileOffset)*2` | `count*2` bytes | FP16/BF16 flat | framework/CopyOut | CopyIn/ViewCopy | executor | 512-element block split |
| input ping/pong | id 0/2 | 各 `ubFormer*2` bytes | B16 | MTE2 | unpack load | scheduler | depth 2 |
| output ping/pong | id 1/3 | 各 `ubFormer*2` bytes | B16 | pack store | MTE3 | scheduler | depth 2，直连 CopyOut |
| RegTensor | register | FP32 input/output、B16 input/output、mask | mixed | unpack/casts/Log | pack | 单 reg 循环 | 无 UB 临时 tensor |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inPing["input ping · ubFormer×2 B"]:1
  outPing["output ping · ubFormer×2 B"]:1
  inPong["input pong · ubFormer×2 B"]:1
  outPong["output pong · ubFormer×2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*(ubFormer*2) + 2*(ubFormer*2) = 8*ubFormer bytes
ubFormer = align_down(floor(UB_platform/8), 128 elements)
UB_total <= UB_platform
dynamic_ub = N/A
workspace_total = executor_total；Log AIV 节点声明 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input ping/pong | CopyIn | unpack+cast+Log | tile i+2 | MTE2/V mutex | 与 R-001 相同 |
| output ping/pong | RINT cast+pack | CopyOut | tile i+2 | V/MTE3 mutex | masked store 与有效字节 CopyOut 一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入 | PIPE_MTE2 | B16 GM | Vector | input id | `count*2` bytes，无 padding |
| 计算 | PIPE_V + MicroAPI mask | unpack/cast | pack | input/output id | Log 在 FP32；降精度 `NO_SAT + CAST_RINT`；ZEROING mask |
| 搬出 | PIPE_MTE3 | packed B16 | GM | output id | 只写有效 count，尾 lane 不落 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_host/arch35/log_tiling_arch35.cpp` | half/bfloat16 `DoTiling32B` | dtype 模板与资源公式 |
| Kernel entry/router | `math/log/op_kernel/log_apt.cpp` | `LogKernel<1,1,DtypeX>` | fast DAG 实例 |
| Device body/helper | `math/log/op_kernel/arch35/log_dag.h` | `LogCustom<U>` B16 branch | unpack、FP32 Log、RINT cast、pack、mask |

### Route R-003：DOUBLE/COMPLEX AICPU 串行

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Launcher | `math/log/op_api/log.cpp` | `IsAiCoreSupport` | dtype 不在 FLOAT/FLOAT16/BF16 | `LogAiCpu` |
| AICPU | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `NeedParallel` | DOUBLE `N<=61440`；COMPLEX `N<=18432` | `shard_log(0,N)` |
| AICPU attr | 同文件 | `NaturalLogCondition` | base=-1、shift=0、scale=1 | `ComputeNaturalLog<T,T>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICPU(Log)
  -> LogCpuKernel::Compute
    -> NormalCheck -> LogOpCheck
    -> log_calls[input_dtype][output_dtype]
      -> LogCompute<T,T>
        -> NeedParallel=false
        -> shard_log(0,N)
          -> ComputeNaturalLog<T,T>
```

##### 源码等价伪码

```text
input = ctx.Input(0).data; output = ctx.Output(0).data
validate dtype equality and fixed attributes
if N == 0: return OK
for index in [0,N):
    output[index] = static_cast<T>(std::log(input[index]))
return OK
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| AICPU input | `input+index` | N elements | DOUBLE/COMPLEX contiguous | Contiguous | std::log loop | executor | read-only |
| AICPU output | `output+index` | N elements | same dtype | loop | ViewCopy | executor | 独立于 input，无原地覆盖 |
| local values | CPU register/stack | 一个元素及 std::log 临时 | T | load | store | 单迭代 | 固定仓未公开 libstdc++ 内部 scratch |
| node workspace | offset 固定仓不可见 | 100 bytes OpDef metadata | byte | executor | AICPU framework | 节点期 | body 未显式索引该 workspace |

##### 分层内存布局

N/A：该 Route 使用 AICPU 的连续输入/输出 GM 和 CPU 寄存器/栈，不使用 AIV UB、TQue 或 TBuf。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = executor_total；AICPU OpDef 节点声明 100 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 无算子自有复用区 | N/A | N/A | N/A | 单线程顺序 | input/output 独立 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元素循环 | C++ 顺序执行 | load/std::log | store | 当前 index | DOUBLE 保持 double；complex 保持 complex 元素类型；`index<end` 守尾 |
| 节点完成 | AICPU task completion | LogCpuKernel | executor 后续 ViewCopy | 整个 output | 具体 event/stream ABI 固定仓不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Launcher producer | `math/log/op_api/log.cpp` | `LogAiCpu` | attrs 与 input/output 加入 AICPU launcher |
| Kernel entry/router | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogCpuKernel::Compute`、`log_calls` | dtype map、校验、空域和错误路径 |
| Device body/helper | 同文件 | `LogCompute`、`ComputeNaturalLog` | 串行元素循环与阈值 |

### Route R-004：DOUBLE/COMPLEX AICPU 并行分片

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Launcher | `math/log/op_api/log.cpp` | `Log` | 非 AICore dtype | AICPU |
| AICPU | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `NeedParallel` | DOUBLE `N>61440` 或 COMPLEX `N>18432` | ParallelFor |
| AICPU | 同文件 | `LogCompute` | `max(1,GetCPUNum(ctx)-kResvCpuNum)`、`CeilMultiple(N,maxCore)` | 多 shard natural log |

#### Device 数据流

##### 函数级调用链

```text
LogCpuKernel::Compute -> log_calls -> LogCompute<T,T>
  -> CpuKernelUtils::GetCPUNum
  -> CpuKernelUtils::ParallelFor(ctx,N,CeilMultiple(N,maxCore),shard_log)
    -> each shard [start,end): ComputeNaturalLog<T,T>
```

##### 源码等价伪码

```text
maxCore = max(1, GetCPUNum(ctx) - kResvCpuNum)
perUnit = CeilMultiple(N, maxCore)  # helper 的精确定义属于 AICPU 框架边界
ParallelFor(N, perUnit, lambda(start,end):
    for index in [start,end): output[index]=static_cast<T>(std::log(input[index])))
if ParallelFor failed: propagate kernel error
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input shard | `[start,end)` | `end-start` elements | DOUBLE/COMPLEX contiguous | framework split | shard loop | executor/read-only | shards 由 ParallelFor 生成 |
| output shard | `[start,end)` | `end-start` elements | same dtype | shard loop | ViewCopy after task | executor | 不同 shard 写不重叠 index |
| framework scheduler | 固定仓不可见 | CpuKernelUtils 管理 | CPU task | ParallelFor | shard callbacks | AICPU runtime | barrier/队列实现不可见 |

##### 分层内存布局

N/A：使用 AICPU GM 分片和 runtime 线程调度，不使用 AIV UB；每个 shard 仅持有循环 index、边界和元素临时值。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = executor_total；AICPU OpDef 节点声明 100 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| output 各 shard 区间 | 对应 callback | 后续 ViewCopy | executor 释放时 | ParallelFor 返回后节点完成 | runtime 必须分配不重叠 `[start,end)`；body 按 index 独占写 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分片 | `ParallelFor` | runtime | callback | `[start,end)` | `index<end` 保护最后 shard |
| 汇合 | ParallelFor 返回值 | callbacks | `LogCompute` 返回 | 完整 output | runtime barrier 细节固定仓不可见；失败由 `KERNEL_HANDLE_ERROR` 传播 |
| 数值 | `std::log` | input element | output element | 单 index | dtype 与 R-003 相同，无额外 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Launcher producer | `math/log/op_api/log.cpp` | `LogAiCpu` | AICPU 节点选择 |
| Kernel entry/router | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogCpuKernel::Compute` | dtype map 与错误传播 |
| Device body/helper | 同文件 | `NeedParallel`、`LogCompute` | dtype 阈值、CPU 数、分片和逐元素 Log |

## 算子特定优化

### Trick T-001：自然对数专用 DAG 消除属性算术与 UB 中间物化

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：公开 InplaceLog 固定 `base=-1,scale=1,shift=0`，Host 因此设置 `attrWork=1`。
- Golden 来源：[`math/log/op_kernel/arch35/log_dag.h:95-104`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/arch35/log_dag.h#L95-L104)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename U, typename T = float>
struct LogScaleOneShiftZeroLnbaseOne {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpLogResult = Bind<LogCustom<U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpLogResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
```

#### 机制与设计原因

通用 Log DAG 还会执行输入 cast、`Muls(scale)`、`Adds(shift)`、Log、`Muls(invLnBase)` 和输出 cast；自然对数专用 DAG 只保留 CopyIn→Log→CopyOut。`LEVEL_2` 的 buffer allocator 为输入和直连输出配置双缓冲，Log 结果直接写 MTE3 可消费的 output ping/pong 区，不生成属性算术的中间 UB tensor。R-002 所需的 FP32 升/降精度发生在 `LogCustom` 的寄存器中，也不增加 UB 子区。

#### 迁移边界

- 适用：固定自然对数属性、逐元素同 shape 输出的 AIV Log。
- 前提：950，dtype 为 FLOAT/FLOAT16/BFLOAT16，属性判等命中 `attrWork=1`，四 buffer UB 上界成立。
- 失效：任一 `scale/shift/base` 条件不成立必须使用 `attrWork=2` 通用 DAG；该路线对 `aclnnInplaceLog` 不可达，但不能在通用 Log 中错误复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的真实反例是 `LogScaleNotOneShiftNotZeroLnbaseNotOne`：属性不满足时若仍走 fast DAG，会漏掉 scale、shift 或换底运算并产生错误语义。

## 模板与实现对比

| 对比维度 | R-001 FP32 AIV | R-002 FP16/BF16 AIV | R-003 AICPU 串行 | R-004 AICPU 并行 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | FLOAT | FLOAT16/BFLOAT16 | DOUBLE/COMPLEX，小域 | DOUBLE/COMPLEX，大域 | 先按 AICore dtype，再按 AICPU 阈值 |
| 模板与 dtype | fast DAG `<float>` | fast DAG `<half/bfloat16_t>` | `LogCompute<T,T>` | 同模板+ParallelFor | 950 AIV binary 只含三 dtype |
| 循环与分核 | AIV block/tile/reg | 同左 | `[0,N)` | runtime shards | 小 CPU 域避免分片 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4 UB 区+FP32 reg | 4 UB 区+B16/FP32 reg | CPU register/stack | 每 shard register/stack | 无通信/dynamic UB |
| 内存总占用与复用 | `16*ubFormer` B | `8*ubFormer` B | 无 AIV UB | 无 AIV UB | dtype 与 ping/pong 深度决定 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + MicroAPI Log | unpack/cast/Log/cast/pack | C++ load/std::log/store | 同左+ParallelFor | 全部为非 Cube 路线 |
| 同步、精度与 tail | MTE2/V/MTE3 mutex；FP32 mask | 同步相同；FP32 Log、RINT | 顺序；index guard | runtime barrier；shard guard | 有效元素不读写 padding |
| fallback/失败边界 | tiling/launch 失败 | 同左 | map/attr/compute 失败 | ParallelFor 失败 | 无 Cube fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定提交 CMake 未列 `ascend910b`、AICore OpDef 也无该 config，发布证据冲突；本卡不声明支持 | 公开文档写支持，但 CMake 未列 `ascend910_93`、OpDef 无该 config，发布证据冲突；本卡不声明支持 | 支持；文档、`ascend950` OpDef、CMake 与 binary config 一致 |
| Host/Launcher | 冲突下不推测 | 冲突下不推测 | arch35 `LogTiling` + AICPU fallback |
| Kernel/资源 | 不生成无后缀卡 | 不生成无后缀卡 | AIV fast DAG 与 AICPU body 如本文 |

## 限制与失败边界

- `selfRef` 为空、dtype 不支持或 rank>8 分别返回公开错误；空 Tensor 是 workspace 0 的成功终点。
- AIV Host 合轴乘积、`ubFormer/coreNum` 向 32 位字段窄化未见额外溢出 guard；超大 shape 还受公开文档所述超时风险影响。
- 实数输入不验证 `x>0`；NaN、Inf、负数和零的数值结果由对应 Log primitive 决定。
- AICPU `CeilMultiple`、`ParallelFor` 的精确调度/栅栏实现，ACLNN executor 总 workspace 切片，以及 Contiguous/Cast/ViewCopy 的 kernel/POD 是固定仓外框架边界。
- A2/A3 公开表与固定实现接线存在冲突，故未创建/覆盖 `inplace_log.md`，不得从本 950 卡外推。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/log/docs/aclnnLog&aclnnInplaceLog.md` | 产品表、Inplace 原型 | 950 公开支持、dtype/rank/非连续语义 |
| 入口声明/约束 | `docs/zh/install/compile.md` | `--soc` 产品映射 | A2=`ascend910b`、A3=`ascend910_93`、950=`ascend950` |
| 入口声明/约束 | `math/log/op_api/aclnn_log.h` | `aclnnInplaceLog*` | C ABI 与自然对数语义 |
| 入口实现 | `math/log/op_api/aclnn_log.cpp` | `CheckInplace`、两段接口 | 校验、空域、executor 图、workspace |
| 入口实现 | `math/log/op_api/log.cpp` | `Log/LogAiCore/LogAiCpu` | dtype selector 与 launcher |
| Host/Tiling/Launcher | `math/log/op_host/log_def.cpp` | `ops::Log` | ascend950 AICore 注册与 dtype |
| Host/Tiling/Launcher | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `LogTiling`、`Tiling4Log` | attr、DAG、POD、key、blockDim、16 MiB |
| Kernel entry/body | `math/log/op_kernel/arch35/log_struct.h` | template args | schMode/attrWork key 空间 |
| Kernel entry/body | `math/log/op_kernel/arch35/log_dag.h` | fast DAG、`LogCustom` | MicroAPI、精度和 mask |
| Kernel entry/body | `math/log/op_kernel/log_apt.cpp` | `log/LogKernel` | AIV-only entry/router |
| Kernel entry/body | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogCpuKernel` | AICPU route、阈值、循环、错误 |
| 构建接线 | `math/log/op_kernel_aicpu/log_aicpu_def.cpp` | AICPU `Log` OpDef | dtype、SO 与 100-byte workspace 配置 |
| 构建接线 | `common/inc/aicpu/aicpu_op_def.h` | `ApplyMathAicpuDefaultCfg` | AICPU SO、kernel lib 与 100-byte workspace 值 |
| 构建接线 | `math/log/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950 与 arch35 接线 |
| 构建接线 | `cmake/func.cmake` | `add_all_modules_sources` | L0/L2、OpDef、Tiling 与 AICPU source 收集 |
| 构建接线 | `math/log/op_host/config/ascend950/log_binary.json` | `op_list` | 950 三种 AIV dtype binary |
| Host/Tiling 直接依赖 | `pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h`（opbase） | `EleBaseTilingData32B` | POD ABI |
| Host/Tiling 直接依赖 | `pkg_inc/op_common/atvoss/elewise/elewise_tiling.h`（opbase） | `DoTiling32B/SetScalar` | shape、buffer divisor、标量 producer |
| Host/Tiling 直接依赖 | `src/op_common/atvoss/elewise/elewise_tiling.cpp`（opbase） | `ElewiseTiling` | 分核、UB、tile 公式 |
| Kernel 直接依赖 | `pkg_inc/op_common/atvoss/elewise/elewise_sch_with_scalar.h`（opbase） | scheduler `Init/Process/Run` | 地址、循环、buffer id 与生命周期 |
| Kernel 直接依赖 | `pkg_inc/op_common/atvoss/util/vec.h`（opbase） | `CopyIn/CopyOut` | DataCopyPad 参数 |
| Kernel 直接依赖 | `pkg_inc/op_common/atvoss/util/sync.h`（opbase） | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 mutex |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 32-byte POD 足以重建自然 Log tiling | `dim0/coreNum/ubFormer` offset 与 32 位窄化；fast DAG不读 scalarData | `EleBaseTilingData32B` |
| 调用顺序 | Contiguous→Log→Cast→ViewCopy | ViewCopy 前 Log 临时输出必须有效；空域不建节点 | `aclnnLogGetWorkspaceSize` |
| workspace/内存 | AIV 四区 ping/pong；Log 节点 16 MiB | `BufferNum=4`、tile 对齐、owner/lifetime；不能把未使用 workspace 当可自由复用 | `ElewiseTiling`、fast DAG |
| 同步 | MTE2→V→MTE3 mutex 链 | buffer id 与 `i&1` 配对，消费结束前不覆盖 | `sync.h`、scheduler `Run` |
| 精度与尾块 | FP32 direct；B16 升 FP32 Log 后 RINT | B16 unpack/pack、ZEROING mask、只写有效 count | `LogCustom` |
| 失败边界 | dtype selector、tiling/AICPU error 均失败关闭 | 不得把 DOUBLE/COMPLEX 送入 AIV，也不得跨平台外推 A2/A3 | `IsAiCoreSupport`、`LogCpuKernel::Compute` |
