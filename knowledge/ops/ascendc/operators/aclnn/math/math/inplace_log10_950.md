---
type: Operator
title: "InplaceLog10 950 ACLNN 算子知识"
description: "说明 aclnnInplaceLog10 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/docs/aclnnLog10&aclnnInplaceLog10.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/aclnn_log10.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_api/log.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/op_api/level2_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/log_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_host/arch35/log_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/arch35/log_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/log_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel/arch35/log_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel_aicpu/log_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel_aicpu/log_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake
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
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/node.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/buffer.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/aicpu_common/context/utils/kernel_util.cc
    role: implementation
created_at: "2026-08-25T11:18:20Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceLog10 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceLog10GetWorkspaceSize` + `aclnnInplaceLog10`，公开 ACLNN 两段式接口；固定仓未出现更高版本入口。 |
| 计算语义 | 对同一 `selfRef` 逐元素提交 `selfRef_i = log10(selfRef_i)`；实现先计算到独立临时 `logOut`，再由框架 `ViewCopy` 写回原张量。 |
| 输入输出 | 一个输入/输出别名 `selfRef`；ND，0～8 维，支持非连续 Tensor；950 支持 `FLOAT16/BFLOAT16/FLOAT/DOUBLE/COMPLEX64/COMPLEX128`。 |
| 执行类型 | R-001 无 Device 节点；R-002 单个算子自有 `AIV_VECTOR` 节点，外围 `Contiguous/ViewCopy` 是框架边界；R-003 单个算子自有 AICPU 节点，外围同样是框架边界。 |
| Kernel 路线 | R-001 空 Tensor 终止；R-002 FP16/BF16/FP32 AIV DAG；R-003 DOUBLE/COMPLEX64/COMPLEX128 AICPU `std::log10`。 |
| 核心资源 | R-002 使用 GM、6 个等长静态 UB 子区（输入 ping/pong、输出 ping/pong、2 个计算临时区）、寄存器与 16 MiB Log 节点 workspace；R-003 使用 AICPU 输入/输出 GM 与 CPU 线程分片，不使用 AIV UB。 |
| 来源闭包 | `source_closed`：公开入口、原地包装、dtype 分流、950 Host/Tiling、POD、AIV entry/DAG/调度器、AICPU body、同步与构建接线均闭合；`Contiguous/ViewCopy/CommonOpExecutorRun` 的生成实现按框架边界标注。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开语义是对每个逻辑元素执行

```text
selfRef[i] <- log10(selfRef[i])
```

R-002 不调用专用 `Log10` Vector 指令，而计算 `ln(1*x+0) * (1/ln(10))`。R-003 在 `Log10Condition(base=10, shift=0, scale=1)` 命中后调用 C++ `std::log10`。API 与 Host 均未逐元素检查实数输入的正值域，非正值、NaN、Inf 的最终行为由固定实现调用的 `MicroAPI::Log` 或 `std::log10` 决定。

原地语义只存在于 API 边界：`aclnnInplaceLog10GetWorkspaceSize` 把 `selfRef` 同时传作非原地包装的 `self` 和 `out`；`l0op::Log` 通过 `executor->AllocTensor` 创建独立 `logOut`，最后 `ViewCopy(logOut, selfRef)` 才覆盖原存储。因此算子自有 Log Kernel 的输入、输出 GM 不别名。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；0～8 维；支持非连续 Tensor | `FLOAT16/BFLOAT16/FLOAT/DOUBLE/COMPLEX64/COMPLEX128` | 非空指针；既是输入也是最终输出。950 属于 `IsRegBase`，`GetDtypeSupportListV2` 选择包含 BF16 的 `UNCHANGED_DTYPE_LIST`。 |
| `workspaceSize` | 标量 Host 输出 | `uint64_t*` | 空 Tensor 明确写 0；非空时由 `uniqueExecutor->GetWorkspaceSize()` 返回执行图所需总字节数。 |
| `executor` | Host 输出对象 | `aclOpExecutor**` | 第一段接口创建并转移所有权；第二段接口消费其中的 `Contiguous -> Log -> ViewCopy` 顺序图。 |
| `workspace` | Device 地址 | 字节区 | 由调用方按第一段返回值分配；R-002 的 Log Tiling 明确请求 16 MiB，框架节点及 executor 的合并/对齐细节固定仓不可见。 |
| `stream` | ACL runtime stream | `aclrtStream` | 第二段交给 `CommonOpExecutorRun`；算子自有代码未另建 stream/event。 |
| 内部 `base/scale/shift` | 无公开属性 | 三个 `float` 常量 | `10.0/1.0/0.0`，由 ACLNN 包装写入内部 `Log` 节点。 |

### 入口与版本

固定 Commit `165d41f02a43a62324b434cb6bda7c533b14da3e` 的公开文档同时列出 A2、A3、950，且只声明无版本后缀的 `aclnnInplaceLog10GetWorkspaceSize/aclnnInplaceLog10`，故该无后缀入口是 950 实际可见的最新版本。本文只描述原地 `InplaceLog10`；不合并 `aclnnLog10`、自然底 `Log` 或 `Log2` 的逻辑语义。

公开文档为原地接口列出的 dtype 与实现一致。头文件注释仍写整数输入并展示过时的 Cast 链，但真正的 `CheckInplaceDtypeValid` 只允许不改变 dtype 的六种浮点/复数类型；本文以可执行实现和公开接口文档为准。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `aclnnInplaceLog10GetWorkspaceSize` 创建 DFX 记录，把 `out` 设为 `const_cast<aclTensor*>(selfRef)`。
2. `CheckInplaceParams` 先拒绝空指针，再通过 `GetDtypeSupportListV2` 选择 950 列表并拒绝列表外 dtype。
3. 调用 `aclnnLog10GetWorkspaceSize(selfRef, out, ...)`；该函数创建 executor，再次检查两指针、输入/输出 dtype 一致、shape 一致且维数不超过 8。`FORMAT_FRACTAL_NZ` 只记录 warning，不在此处返回错误；公开接口约束仍是 ND。
4. 空 Tensor 直接返回 `workspaceSize=0` 和空计算 executor，不加入 Log 节点。
5. 非空 Tensor 依次记录 `l0op::Contiguous(selfRef)`、`l0op::Log(selfContiguous, 10, 1, 0)`、`l0op::ViewCopy(logOut, selfRef)`。`Contiguous/ViewCopy` 的自动生成 body 是明确框架边界。
6. 第二段 `aclnnInplaceLog10` 将 workspace、executor、stream 原样交给 `CommonOpExecutorRun`；任一第一段节点创建失败返回 `ACLNN_ERR_INNER_NULLPTR`，launcher 执行错误由 `CommonOpExecutorRun` 返回。

### 参数、Workspace 与生命周期

`uniqueExecutor` 拥有连续化结果、`logOut` 和 ViewCopy 任务；第一段结束时 `ReleaseTo(executor)` 把所有权交给调用方。`logOut` 由 `l0op::Log` 按 `selfRef` 的 view shape 与原 dtype 分配，在 Log 节点产出后保持有效，直到同一 executor 中的 `ViewCopy` 消费并写回 `selfRef`。

R-002 的 `LogTiling::RunTiling` 将该节点 workspace 写为固定 `16,777,216` 字节；Kernel 形参含 `workspace` 但 body 不读取它。ACLNN 总 workspace 还可能包含非连续输入的 `Contiguous` 临时区、`logOut` 和 `ViewCopy` 所需区，具体 slice offset、对齐和复用由 executor 框架生成，固定仓不可见，不能把 16 MiB 误写成整个 ACLNN 调用的总大小。R-003 没有算子自有 Tiling workspace 公式，其 AICPU task 元数据与框架临时区同样由 executor 汇总。

### 启动图与失败传播

```text
R-001: CheckInplaceParams -> delegated CheckParams -> selfRef.IsEmpty
       -> workspaceSize=0 -> success（无 Device 节点）

R-002/R-003:
  framework Contiguous(selfRef) [可能直接复用或物化，固定仓不可见]
    -> edge C: selfContiguous，shape=selfRef.viewShape，dtype=selfRef.dtype，executor-owned
  Log node
    R-002 -> AIV entry log<schMode=1, attrWork=2, DTYPE_X>
    R-003 -> AICPU LogCpuKernel::Compute
    -> edge L: logOut，shape=selfRef.viewShape，dtype=selfRef.dtype，executor-owned
  framework ViewCopy(logOut, selfRef)
    -> 覆盖调用方拥有的 selfRef storage
```

三节点按加入同一 `aclOpExecutor` 的顺序由同一传入 stream 执行；算子代码没有额外 event。前一节点创建失败会使第一段立即返回，执行阶段失败由 executor 返回并阻断成功提交。框架节点是否实际发射 Kernel、C/L 边在总 workspace 中的精确 offset/extent 与可覆盖时刻固定仓不可见；可确认的逻辑 extent 都是 `N * sizeof(dtype)` 个有效字节，layout 由 Contiguous 输出转为连续、ViewCopy 再按 `selfRef` view 写回。

## Route 与参数设计

### 分核与资源计算

设非空元素数 `N = product(storageShape)`；标量由 shape 逻辑得到 `N=1`。R-002 的 `ElewiseBaseTiling::DoTiling32B` 从 950 平台读取 AIV core 数 `Cmax` 和 UB 字节数 `UB`，对当前泛化 DAG 得到 `BufferNum=6`、`MaxDtypeBytes=4`；`MinDtypeBytes` 对 FP32 为 4，对 FP16/BF16 为 2。

```text
coreNum = min(ceil(N * MinDtypeBits / 32768), Cmax)
blockFormer = ceil(ceil(N / coreNum) / 512) * 512      # 单位：元素
blockNum = ceil(N / blockFormer)                        # 实际 blockDim
blockTail = N - (blockNum - 1) * blockFormer

rawTile = floor(UB / (BufferNum * MaxDtypeBytes)) = floor(UB / 24)
alignFactor = 256 / MinDtypeBytes                       # FP32=64 元素，FP16/BF16=128 元素
ubFormer = floor(rawTile / alignFactor) * alignFactor

loopFormer = ceil(blockFormer / ubFormer)
loopTail = ceil(blockTail / ubFormer)
tailFormer = blockFormer - (loopFormer - 1) * ubFormer
tailTail = blockTail - (loopTail - 1) * ubFormer
```

Host 仅把 `dim0=N/coreNum/ubFormer` 写入 32B POD；Device 用相同公式重建 `blockFormer/blockNum/loop/tail`。`blockDim=blockNum`，而不是平台 `Cmax`。若 `coreNum==0`、`UB` 小于额外空间或对齐后 `ubFormer==0`，Tiling 失败。维度乘积和这些中间值使用 `int64_t`，固定源码没有额外乘法溢出检查。

R-003 不使用 AIV 分核。`data_num=N`；DOUBLE 在 `N>60*1024` 时并行，COMPLEX64/COMPLEX128 在 `N>18*1024` 时并行。并行时 `max_core_num=max(1, GetCPUNum(ctx)-kResvCpuNum)`，`per_unit_size=ceil(N/max_core_num)`，`ParallelFor` 生成不重叠 `[start,end)` 分片；否则单线程处理 `[0,N)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 空 Tensor 终止 | 无 key/无 Kernel | 参数校验成功后 `selfRef->IsEmpty()` | 非空进入 dtype 分流 |
| 2 | R-002 950 AIV 泛化 Log DAG | `GET_TPL_TILING_KEY(1, 2)`；`schMode=1, attrWork=2` | 非空且 dtype 属于 `{FLOAT16,BFLOAT16,FLOAT}`；`IsAiCoreSupport` 先于 AICPU 判断 | dtype 不命中进入 R-003；Tiling 失败则调用失败 |
| 3 | R-003 AICPU Log10 | `log_calls[input][output]` | 非空且 dtype 属于 `{DOUBLE,COMPLEX64,COMPLEX128}` | 映射缺失或检查失败返回参数错误 |

```text
CheckInplaceParams(selfRef)
  -> fail: ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID
  -> delegate aclnnLog10GetWorkspaceSize(selfRef, selfRef)
     -> empty: R-001
     -> Contiguous
     -> l0op::Log allocates logOut
        -> dtype in {FLOAT, FLOAT16, BF16}: R-002 LogAiCore
           -> base=10 => invLnBase=1/ln(10), attrWork=2
           -> output dtype selects U=float/half/bfloat16_t
           -> key GET_TPL_TILING_KEY(1,2)
        -> otherwise: R-003 LogAiCpu
           -> base=10, shift=0, scale=1 => Log10Condition
     -> ViewCopy(logOut, selfRef)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 空 Tensor 终止 | `IsEmpty()` | 950；合法 dtype；元素数 0 | 无 Device 节点 | API early return | `workspaceSize=0` | 无输出写；原空 Tensor 保持空 |
| R-002 950 AIV 泛化 Log DAG | symbolic key `(1,2)` | 950；非空；FP16/BF16/FP32；shape≤8D | 框架 Contiguous -> 单个 Log AIV -> 框架 ViewCopy | `LogKernel<1,2,U>` -> `ElementwiseSchWithScalar<EleBaseTilingData32B,1,LogScaleNotOneShiftNotZeroLnbaseNotOne<U>::OpDag>` | 6 个 `blockLen` UB 子区；Mutex lock/unlock 连接 MTE2/V/MTE3；Log 节点 workspace 16 MiB | block 尾与 tile 尾均按有效元素数搬运；Tiling 失败无 fallback |
| R-003 AICPU Log10 | dtype map + `Log10Condition` | 950；非空；DOUBLE/COMPLEX64/COMPLEX128 | 框架 Contiguous -> 单个 AICPU Log -> 框架 ViewCopy | `LogCpuKernel::Compute` -> `LogCompute<T,T>` -> `Log10Compute<T>` | 无 AIV UB；按阈值选择串行或 `ParallelFor` | `[start,end)` 保护尾分片；map 缺失返回参数错误 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 dtype 分流 | `CheckType(dtype,{FLOAT,FLOAT16,BF16})` | 只有这三类进入公开 950 AIV Log OpDef；其余由 AICPU 实现 | R-002/R-003 | R-003 或 API dtype 失败 | `math/log/op_api/log.cpp::IsAiCoreSupport/Log` |
| 固定 Log10 属性 | `base=10, scale=1, shift=0` | `invLnBase!=1`，故自然对数专用 DAG 永不可达，必须走泛化 DAG | R-002 | 无其它 AIV Route | `LogTiling::SetAttr` |
| 非零 shape | `ElewiseTiling` 拒绝任一维为 0 | 空 Tensor 已在 ACLNN 层截断；Tiling 不接受 0 维长 | R-002 | R-001 或失败 | `ElewiseTiling` |
| 最少每核工作量 | `coreNum=ceil(N*MinDtypeBits/32768)` 后封顶 | 避免为小张量启用过多 AIV block | R-002 | 仍在 R-002，只改变 blockNum | `Ops::Base::ElewiseTiling` |
| 512 元素 block 对齐 | `blockFormer=align_up(ceil(N/coreNum),512)` | Device 用相同公式映射 block，最后 block 承担余数 | R-002 | 无 fallback | `ElewiseTiling`、`ElementwiseSchWithScalar` 构造函数 |
| UB 六分区上界 | `ubFormer=floor((UB/24)/alignFactor)*alignFactor` 且非 0 | 泛化 DAG 的 2 个 MTE2/MTE3 ping-pong 对、2 个临时区均按 FP32 最大宽度分配 | R-002 | Tiling 失败 | `DoTiling`、`GetBufferNumLevel2`、`GetMaxElemNum` |
| MicroAPI 尾 mask | `loopNum=ceil(tileLength/(VECTOR_REG_WIDTH/4))` | 最后寄存器轮只提交有效 lane | R-002 | 无 fallback | `LogDag::LogCustom<float>` |
| AICPU 并行阈值 | DOUBLE `N>61440`；复数 `N>18432` | 大张量才支付 CPU 分片开销 | R-003 | 串行 `shard_log(0,N)` | `NeedParallel`、`LogCompute` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `EleBaseTilingData32B` | `dim0` | `int64_t`/64 | `offsetof=0` | 元素 | `AdaptEleBaseTilingData32B: dim0=N` | 构造函数赋给 `dim0_`，推导 block/tail | 非空，乘积溢出未额外检查 |
| `EleBaseTilingData32B` | `coreNum` | `int32_t`/32 | `offsetof=8` | AIV core | `AdaptEleBaseTilingData32B: coreNum=elewiseTilingData.coreNum` | 推导 `blockFormer_` | 必须大于 0；由平台 AIV core 数封顶 |
| `EleBaseTilingData32B` | `ubFormer` | `int32_t`/32 | `offsetof=12` | 元素/tile | `AdaptEleBaseTilingData32B: ubFormer=elewiseTilingData.ubFormer`，发生 `int64_t -> int32_t` 赋值 | `Process` 的整 tile 长度与 `blockLen` | Host 未显式检查窄化；对齐后不得为 0 |
| `EleBaseTilingData32B` | `scalarData[0..3]` | `float`/32 | `offsetof=16` | 无量纲 | 第一次 `SetScalar<float>(attrScale)`；本 API 为 `1.0f` | `SetScalar<0>` -> DAG Var 0 -> `Vec::Muls` | R-002 active |
| `EleBaseTilingData32B` | `scalarData[4..7]` | `float`/32 | `offsetof=20` | 与输入相同的数值单位 | 第二次 `SetScalar<float>(attrShift)`；本 API 为 `0.0f` | DAG Var 1 -> `Vec::Adds` | R-002 active |
| `EleBaseTilingData32B` | `scalarData[8..11]` | `float`/32 | `offsetof=24` | 无量纲 | 第三次 `SetScalar<float>(attrInvLnBase)`；本 API 为 `1/log(10)` | DAG Var 2 -> 最后 `Vec::Muls` | R-002 active |
| `EleBaseTilingData32B` | `scalarData[12..15]` | `char[4]`/32 | `offsetof=28` | N/A | 零初始化后未写 | 无 consumer | inactive padding/reserved |

`EleBaseTilingData32B` 的可见定义为 8+4+4+16 字节，因此 `sizeof=32`、自然最大成员对齐为 8、`alignof=8`；Host 与 Device 都使用 `REGISTER_TILING_DEFAULT/GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData32B)`。R-003 不使用 AscendC Tiling POD；AICPU launcher 序列化名字为 `base/scale/shift` 的三个 float 属性，AICPU body按名读取。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 空 Tensor | `self->IsEmpty()` | `aclnnLog10GetWorkspaceSize` | `workspaceSize=0` | 无 | 无 | ACLNN 源由 `add_all_modules_sources` 收集 |
| R-002 AIV | dtype selector + `(1,2)` | `l0op::Log`、`LogTiling::RunTiling`、`ElewiseBaseTiling` | `EleBaseTilingData32B` + 3 scalars | `log` -> `LogKernel<schMode,attrWork,DTYPE_X>` | `LogScaleNotOneShiftNotZeroLnbaseNotOne`、`ElementwiseSchWithScalar`、`Vec`/MicroAPI helpers | `Log::AICore().AddConfig("ascend950")`；CMake `ascend950/arch35`；binary config `Log/ascend950` |
| R-003 AICPU | `!IsAiCoreSupport(dtype)` + `log_calls` | `l0op::LogAiCpu` | attrs `base=10,scale=1,shift=0`；无 AscendC POD | `REGISTER_CPU_KERNEL("Log",LogCpuKernel)` | `Compute` -> `LogCompute` -> `Log10Compute`；`ParallelFor`/`CeilMultiple` | `log_aicpu_def.cpp` + `add_all_modules_sources` 的 AICPU glob/target |

## Kernel 实现

### 公共执行流程

所有非空 Route 共享 API 级 `Contiguous -> Log -> ViewCopy`。`l0op::Log` 总是分配独立输出，因此 R-002/R-003 的核心节点只需顺序读连续输入、写连续 `logOut`，不承担对非连续原存储的 scatter，也不需解决输入输出别名；最终写回由框架 ViewCopy 处理。核心节点失败时 executor 不应被视为已完成原地提交。

R-002 的算子自有启动图只有一个 AIV 节点；R-003 只有一个 AICPU 节点。`Contiguous` 与 `ViewCopy` 可能各自产生零个或多个框架 Kernel，但其 selector、POD、workspace 切片和 body 不在固定算子源码中，属于允许的框架边界，不能据此构造虚假节点。

### Route R-001：空 Tensor 终止

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/log/op_api/aclnn_log10.cpp` | `CheckInplaceParams` | 非空指针且 dtype 合法 | 委托非原地 workspace 包装 |
| API | 同上 | `aclnnLog10GetWorkspaceSize` | `self->IsEmpty()` | `workspaceSize=0`，释放 executor，返回成功 |

#### Device 数据流

##### 函数级调用链

```text
aclnnInplaceLog10GetWorkspaceSize(selfRef)
  -> CheckInplaceParams
  -> aclnnLog10GetWorkspaceSize(selfRef, selfRef)
    -> CheckParams
    -> selfRef.IsEmpty() == true
    -> return workspaceSize=0, executor
```

##### 源码等价伪码

```text
# 伪码
if selfRef == null or dtype unsupported: return error
if selfRef.shape differs from selfRef.shape or rank > 8: return error
if selfRef.is_empty:
    workspaceSize = 0
    return success with executor containing no Log/ViewCopy work
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `selfRef` | 调用方地址 | 0 个元素 | 合法 dtype/ND | 调用方 | 无 Device consumer | 调用方全程拥有 | 无搬运 |
| workspace | N/A | 0 字节 | byte | API | 无 | 调用方无需分配 | N/A |

##### 分层内存布局

N/A：该 Route 不启动 AIV、SIMT、NDDMA、AICPU 或通信节点，也不分配 UB。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | API early return | 元素数为 0 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| API 终止 | C++ 条件分支 | 参数检查 | 第一段返回 | executor/workspace | 无元素、无数值计算、无尾块 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_api/aclnn_log10.cpp` | `aclnnLog10GetWorkspaceSize` | 空 Tensor 将 workspace 置 0 并立即成功返回 |
| Kernel entry/router | N/A | N/A | 无 Kernel |
| Device body/helper | N/A | N/A | 无 Device body |

### Route R-002：950 AIV 泛化 Log DAG

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/log/op_api/log.cpp` | `IsAiCoreSupport` | dtype 为 FP32/FP16/BF16 | `LogAiCore` |
| Host | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `SetAttr` | base=10、scale=1、shift=0，故 `invLnBase=1/log(10)` 且专用 ln 条件为 false | `attrWork=2` |
| Tiling | 同上 | `RunTiling` | output dtype 分别为 FP16/BF16/FP32 | `DoTiling32B<...NotOne...<U>::OpDag>`，key `(1,2)` |
| Router | `math/log/op_kernel/log_apt.cpp` | `LogKernel` | 编译期 `attrWork==2` | 构造泛化 DAG scheduler；其它 attrWork 不执行本分支 |

#### Device 数据流

##### 函数级调用链

```text
LogAiCore -> ADD_TO_LAUNCHER_LIST_AICORE(Log, selfContiguous, logOut, attrs)
  -> Tiling4Log -> LogTiling::RunTiling
    -> ElewiseBaseTiling::DoTiling32B<LogScaleNotOneShiftNotZeroLnbaseNotOne<U>::OpDag>
  -> log<schMode=1, attrWork=2>(x, y, workspace, tiling)
    -> LogKernel<1,2,U>
      -> ElementwiseSchWithScalar<EleBaseTilingData32B,1,GenericDag>::Init(x,y)
      -> ElementwiseSchWithScalar::Process
        -> per tile Run
          -> Vec::CopyIn<U>
          -> Vec::Cast<float,U,CAST_MODE_NONE>        # U=float 时 DAG 编译期消除
          -> Vec::Muls<float>(scale=1)
          -> Vec::Adds<float>(shift=0)
          -> LogCustom<float> -> MicroAPI::Log
          -> Vec::Muls<float>(invLnBase=1/ln(10))
          -> Vec::Cast<U,float,CAST_MODE_RINT>        # U=float 时 DAG 编译期消除
          -> Vec::CopyOut<U>
  -> framework edge logOut -> ViewCopy -> selfRef
```

##### 源码等价伪码

```text
# 伪码；所有 offset 在 scheduler 内以元素计，绑定 GM 时乘 sizeof(U)
block_start = blockIdx * blockFormer
block_len = (blockIdx == blockNum-1) ? blockTail : blockFormer
loop_num = ceil(block_len / ubFormer)

for tile_idx in [0, loop_num):
    tile_offset = tile_idx * ubFormer
    valid = (tile_idx == loop_num-1) ? block_len-(loop_num-1)*ubFormer : ubFormer
    pp = tile_idx & 1

    lock MTE2(input_buf[pp])
    DataCopyPad(input_buf[pp], x[block_start+tile_offset],
                blockCount=1, blockLenBytes=valid*sizeof(U), srcStride=0, dstStride=0,
                padding=false)
    unlock MTE2(input_buf[pp])

    t0 = cast_fp32(input_buf[pp])          # FP16/BF16；FP32 路线消除 no-op cast
    t1 = t0 * 1.0f
    t0 = t1 + 0.0f                         # temp0/temp1 交替复用
    for reg_offset in steps of VL=VECTOR_REG_WIDTH/sizeof(float):
        mask = UpdateMask<float>(valid remaining)
        reg_in = DataCopy DIST_NORM from t0
        reg_out = MicroAPI::Log(reg_in, mask)
        DataCopy DIST_NORM_B32 reg_out to t1 under mask
    t0 = t1 * (1.0f / ln(10.0f))
    out_buf[pp] = cast_U(t0, CAST_MODE_RINT) # FP32 路线消除 no-op cast

    lock MTE3(out_buf[pp])
    DataCopyPad(y[block_start+tile_offset], out_buf[pp],
                blockCount=1, blockLenBytes=valid*sizeof(U), srcStride=0, dstStride=0)
    unlock MTE3(out_buf[pp])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `(blockIdx*blockFormer + tileOffset)*sizeof(U)` | `valid*sizeof(U)` per tile；总 N 元素 | U/连续 | framework Contiguous | `Vec::CopyIn` | executor 到 Log 完成 | block 起点按 512 元素对齐，尾只读 valid |
| UB input ping | `0*blockLen` | `blockLen=ubFormer*4` 字节 | LocalTensor<U> | MTE2 CopyIn | 首个 Cast/Muls | 当前偶数 tile | id 0；有效数据仅 `valid*sizeof(U)` |
| UB input pong | `4*blockLen` | `blockLen` | LocalTensor<U> | MTE2 CopyIn | 首个 Cast/Muls | 当前奇数 tile | id 4；与 ping 交替 |
| UB output ping | `1*blockLen` | `blockLen` | LocalTensor<U> | 最终 Cast/Muls | MTE3 CopyOut | 当前偶数 tile | id 1 |
| UB output pong | `5*blockLen` | `blockLen` | LocalTensor<U> | 最终 Cast/Muls | MTE3 CopyOut | 当前奇数 tile | id 5 |
| UB temp0 | `2*blockLen` | `blockLen` | FP32 | DAG 当前计算节点 | 下一计算节点 | 单 tile 内，消费后可覆盖 | id 2；跨 ping/pong 共用 |
| UB temp1 | `3*blockLen` | `blockLen` | FP32 | DAG 当前计算节点 | 下一计算节点 | 单 tile 内，消费后可覆盖 | id 3；与 temp0 交替 |
| 寄存器 | 固定寄存器文件，地址不可见 | 两个 FP32 RegTensor + MaskReg；具体字节数固定仓不可见 | FP32 lanes | `MicroAPI::DataCopy` | `MicroAPI::Log`/store | 每次寄存器循环 | mask 约束尾 lane |
| GM `y=logOut` | 与 x 相同的元素 offset | `valid*sizeof(U)`；总 N 元素 | U/连续 | `Vec::CopyOut` | framework ViewCopy | Log 后到 ViewCopy 完成 | 尾只写 valid |
| Log workspace | Kernel 形参 `workspace` | 16,777,216 字节 | byte | Host Tiling | Kernel body未读取 | executor 管理 | 与 ACLNN 总 workspace slice 的 offset 固定仓不可见 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inPing["input ping · id0 · ubFormer*4 B"]:1
  outPing["output ping · id1 · ubFormer*4 B"]:1
  temp0["temp0 · id2 · ubFormer*4 B"]:1
  temp1["temp1 · id3 · ubFormer*4 B"]:1
  inPong["input pong · id4 · ubFormer*4 B"]:1
  outPong["output pong · id5 · ubFormer*4 B"]:1
```

##### 字节账与复用顺序

```text
MTE2_pingpong = 2 * ubFormer * 4
MTE3_pingpong = 2 * ubFormer * 4
temp_calc = 2 * ubFormer * 4
UB_total = 6 * ubFormer * 4 = 24 * ubFormer <= platform UB
dynamic_ub = N/A
Log_node_workspace = 16,777,216 B
ACLNN_workspace_total = executor aggregation(Contiguous temp, logOut, Log 16 MiB, ViewCopy temp)
                         # 精确 slice 对齐/复用固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input id0/id4 | `Vec::CopyIn` | 首个 DAG 计算节点 | 下一个同奇偶 tile | `Lock/Unlock<PIPE_MTE2>` 与 V 侧 `GetTensor/ReleaseTensor` | 当前 tile 的输入已被消费 |
| output id1/id5 | 最后 DAG 计算节点 | `Vec::CopyOut` | 下一个同奇偶 tile | V 与 `Lock/Unlock<PIPE_MTE3>` | MTE3 已完成当前有效字节搬出 |
| temp id2/id3 | Cast/Muls/Adds/Log 节点 | 紧随其后的 DAG 节点 | 输入节点最后一次消费后由后续节点取得 | `GetTensor/ReleaseTensor<VECCALC>`；编译期 `ReleaseAndUpdateEs` 证明不再被后续引用 | DAG 是线性链；不可在 consumer 完成前覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM->UB | `Mutex::Lock/Unlock<PIPE_MTE2>` | `DataCopyPad` | V 侧对相同 buf id 的 `GetTensor` | input ping/pong | `blockLenBytes=valid*sizeof(U)`，padding disabled |
| DAG 计算 | `GetTensor/ReleaseTensor<VECCALC>` | 上一节点 | 下一节点 | temp0/temp1 与最终 output buffer | FP16/BF16 先升为 FP32；FP32 no-op Cast 编译期消除 |
| 对数 | `MicroAPI::UpdateMask` + `Log` | FP32 UB | FP32 UB | 最后寄存器轮有效 lane | `VL=VECTOR_REG_WIDTH/4`；不写无效 lane |
| 降精度 | `Vec::Cast<U,float,CAST_MODE_RINT>` | FP32 缩放结果 | output UB | FP16/BF16 output | FP32 计算后一次降精度；FP32 路线消除 Cast |
| UB->GM | `Mutex::Lock/Unlock<PIPE_MTE3>` | output UB | GM `logOut` | output ping/pong | 只提交 `valid*sizeof(U)` |
| 跨节点 | executor stream 顺序 | Log `logOut` | ViewCopy | `logOut` 与 `selfRef` | ViewCopy 后原地结果可见；具体框架同步 primitive 不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `LogTiling::SetAttr/RunTiling` | 生产 attrWork=2、三 scalar、key、blockDim、16 MiB workspace |
| Host/Tiling helper | `opbase/.../elewise_tiling.h/.cpp` | `DoTiling32B/ElewiseTiling` | 生产 core、block、tile、loop 与 32B POD |
| Kernel entry/router | `math/log/op_kernel/log_apt.cpp` | `log`、`LogKernel<1,2,U>` | AIV-only；key 2 映射泛化 DAG |
| Device body/helper | `math/log/op_kernel/arch35/log_dag.h` | `LogScaleNotOneShiftNotZeroLnbaseNotOne<U>`、`LogCustom<float>` | FP32 scale/add/log/scale 与末端 cast |
| Device scheduler | `opbase/.../elewise_sch_with_scalar.h` | `Init/Process/Run/CopyIn/RunNormalOp/CopyOut` | block/tile 地址、ping-pong、LocalTensor 与 DAG 遍历 |
| 搬运/同步 | `opbase/.../vec.h`、`sync.h` | `CopyIn/CopyOut/GetTensor/ReleaseTensor` | `DataCopyPad` 字节参数和 MTE2/V/MTE3 mutex |

### Route R-003：AICPU Log10

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/log/op_api/log.cpp` | `Log` | `IsAiCoreSupport(self)==false` | `LogAiCpu` |
| Launcher | 同上 | `LogAiCpu` | attrs `base,scale,shift`，名字顺序 `{base,scale,shift}` | AICPU task `Log` |
| AICPU | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogOpCheck` | dtype input=output；base 合法 | 进入 dtype map |
| AICPU | 同上 | `LogCompute` | `Log10Condition(10,0,1)` | `Log10Compute<T>` |

#### Device 数据流

##### 函数级调用链

```text
LogAiCpu -> ADD_TO_LAUNCHER_LIST_AICPU(Log, attrs, selfContiguous, logOut)
  -> REGISTER_CPU_KERNEL("Log", LogCpuKernel)::Compute
    -> NormalCheck
    -> LogOpCheck(base=10, shift=0, scale=1)
    -> log_calls.at(inputType).at(outputType)
      -> LogCompute<double,double> or LogCompute<complex<T>,complex<T>>
        -> NeedParallel
        -> shard_log(start,end)
          -> Log10Condition
          -> Log10Compute<T>: output[i] = std::log10(input[i])
  -> framework edge logOut -> ViewCopy -> selfRef
```

##### 源码等价伪码

```text
# 伪码
check one input and one output
check input_dtype == output_dtype
read attrs; require base == -1 or (base > 0 and base != 1)
if output_elements == 0: return OK

fn shard(start,end):
    # base=10, shift=0, scale=1, so this branch is first matching Log10 route
    for i in [start,end):
        output[i] = std::log10(input[i])

threshold = 61440 if dtype == DOUBLE else 18432  # 此 Route 的复数类型
if N > threshold:
    cores = max(1, GetCPUNum(ctx)-kResvCpuNum)
    ParallelFor(ctx, N, ceil(N/cores), shard)
else:
    shard(0,N)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| AICPU input GM | `input+i` | N 元素 | DOUBLE/COMPLEX64/COMPLEX128 连续 | framework Contiguous | `Log10Compute` | executor 到 AICPU task 完成 | 每线程只读自己的 `[start,end)` |
| AICPU output GM | `output+i` | N 元素 | 与输入相同 | `std::log10` assignment | framework ViewCopy | executor 到 ViewCopy 完成 | 分片互不重叠，无 atomic |
| AICPU task metadata | framework管理 | attrs + shape dependency | named attrs | `LogAiCpu` launcher | `LogOpCheck` | executor/task 生命周期 | 精确序列化 ABI 固定仓不可见 |

##### 分层内存布局

N/A：该 Route 在 AICPU 上逐元素读写，不使用 AIV UB、TQue/TBuf、dynamic UB、NDDMA buffer 或通信 window；CPU cache/栈布局不在固定算子源码中。

##### 字节账与复用顺序

```text
UB_total = N/A（AICPU）
dynamic_ub = N/A
operator_owned_workspace = 0 个显式 AICPU slice
ACLNN_workspace_total = executor aggregation（精确值固定仓不可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `logOut[start:end)` | 对应 `shard_log` | 后续 ViewCopy | ViewCopy 完成后 executor 可回收 | `ParallelFor` 返回后 AICPU Compute 返回；跨节点由 executor stream 排序 | 各分片区间不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU 分片 | `CpuKernelUtils::ParallelFor` | worker `shard_log` | `ParallelFor` 返回 | output 不重叠区间 | `end` 为排他边界，最后分片自然截断 |
| 单线程 | 直接 `shard_log(0,N)` | 当前 AICPU 线程 | Compute 返回 | 全输出 | DOUBLE/complex 保持原 dtype 调用 `std::log10` |
| 跨节点 | executor stream/task 顺序 | AICPU `logOut` | ViewCopy | 临时输出 | 框架同步细节固定仓不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/log/op_api/log.cpp` | `LogAiCpu` | 以 shape dependency 与三 attrs 加入 AICPU launcher |
| Kernel entry/router | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `REGISTER_CPU_KERNEL`、`LogCpuKernel::Compute`、`log_calls` | dtype 到模板 body 的完整映射与失败终点 |
| Device body/helper | 同上；`opbase/aicpu_common/context/utils/kernel_util.cc` | `LogCompute/Log10Compute/NeedParallel/CeilMultiple` | 逐元素 `std::log10`、阈值、核数和分片长度 |

## 算子特定优化

### Trick T-001：泛化 DAG 全程 FP32 计算并以 LEVEL_2 复用两块临时 UB

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：公开 InplaceLog10 固定 `base=10,scale=1,shift=0`，Host 选择 `attrWork=2`；U 为 FP16/BF16/FP32。
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
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
```

#### 机制与设计原因

FP16/BF16 只在 GM 边界保持 16 位：CopyIn 后一次升为 FP32，scale、shift、自然对数和 `1/ln(10)` 缩放全部在 FP32 完成，最后以 `CAST_MODE_RINT` 一次降回 U，避免每个中间算子反复降精度。FP32 的两个 Cast 由 `DAGSch::GetRealArgs` 检测为 no-op 并在编译期消除。

`MemLevel::LEVEL_2` 让线性链的已消费中间节点归还临时 buffer；存活分析证明最多同时需要两块 FP32 临时区，而非为六个数学节点逐一物化 UB。结合输入/输出 ping-pong，总数固定为 6，Host 因而用 `UB/(6*4)` 计算 tile 上界。该机制减少 UB 物化，同时保留 MTE2/MTE3 双缓冲。

#### 迁移边界

- 适用：单输入单输出、可表示为线性 elementwise DAG，且中间计算适合 FP32 的 950 AIV 算子。
- 前提：950 AIV；dtype 为 FP16/BF16/FP32；DAG 生命周期分析与 `bufferIds` 分配保持一致；UB 足以容纳六个 `ubFormer*4` 子区。
- 失效：DOUBLE/复数改走 R-003；若添加分叉或多引用节点，`TempCalcNode/BufferNum` 必须重新由 DAG 分析，不能继续假设两块临时区。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；工程迁移时不得跳过最终 `CAST_MODE_RINT`，也不得在不重算 DAG 存活节点的情况下硬编码 `BufferNum=6`。

### Trick T-002：AICPU 按 dtype 阈值延迟启用 ParallelFor

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：DOUBLE 元素数大于 61,440，或 COMPLEX64/COMPLEX128 元素数大于 18,432。
- Golden 来源：[`math/log/op_kernel_aicpu/log_aicpu.cpp:142-151`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log/op_kernel_aicpu/log_aicpu.cpp#L142-L151)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
  int64_t data_num = ctx.Input(0)->NumElements();
  DataType dtype = ctx.Input(0)->GetDataType();
  if (NeedParallel(data_num, dtype)) {
    uint32_t max_core_num = std::max(1U, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num,
                        CeilMultiple(data_num, max_core_num), shard_log),
                        "Log Compute failed.")
  } else {
      shard_log(0, data_num);
  }
```

#### 机制与设计原因

小张量直接串行遍历，超过按 dtype 区分的阈值后才按可用 CPU 数切分。复数计算使用较低阈值，DOUBLE 使用通用 60 Ki 元素阈值；每个 worker 只写自己的 `[start,end)`，不需要 atomic 或共享归约。`CeilMultiple(N,cores)` 的固定实现实际返回 `ceil(N/cores)`，作为每个分片的目标元素数。

#### 迁移边界

- 适用：每个元素完全独立、输出区间可静态分割的 AICPU unary 算子。
- 前提：`ParallelFor` 返回前完成所有分片；输出不与输入别名；保留 `kResvCpuNum` 预留策略。
- 失效：有跨元素依赖、归约、重叠写或真实原地读写时不能直接复用该分片模式。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把 `CeilMultiple` 误解为“向上对齐到 core 数的倍数”，其实现返回的是除法向上取整。

## 模板与实现对比

| 对比维度 | R-001 空 Tensor | R-002 AIV 泛化 DAG | R-003 AICPU Log10 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 合法且 `IsEmpty` | 非空 FP16/BF16/FP32 | 非空 DOUBLE/复数 | 先空域，再由 `IsAiCoreSupport` 分流 |
| 模板与 dtype | 无 | `LogKernel<1,2,U>`；中间 T=float | `LogCompute<T,T>` | AIV OpDef只闭合三种实数 dtype，其余保留 AICPU 原 dtype |
| 循环与分核 | 无 | AIV block -> tile -> 寄存器 VL | CPU serial/ParallelFor -> 元素 | 平台资源和计算成本不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 | 无 TQue/TBuf；6 个静态 LocalTensor 子区 + FP32 RegTensor/MaskReg | 无 AIV 资源 | ATVOSS 使用静态 buf id 与 Mutex，不使用 TPipe queue |
| 内存总占用与复用 | 0 | `24*ubFormer` 字节；2 个 temp 跨节点复用 | 无显式 workspace/UB | DAG LEVEL_2 存活分析控制 AIV 上界 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 无 | `DataCopyPad`、Cast/Muls/Adds、MicroAPI Log | `std::log10` | 无 SIMT/NDDMA/通信节点 |
| 同步、精度与 tail | 无 | MTE2/V/MTE3 Mutex；FP16/BF16 升 FP32；mask + valid bytes | ParallelFor join；原 dtype | 各自按执行族处理尾部 |
| fallback/失败边界 | 参数错误则不进入 | Tiling/launch 失败，无 Device fallback | dtype map/参数/ParallelFor 失败 | L0 dtype 分流是唯一 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档标为支持，但固定源码缺少该平台 FP16/BF16/FP32 AICore Host/Tiling/Kernel 构建闭包，未生成 A2/A3 卡 | 同 A2 | 支持；公开文档、`ascend950` OpDef、arch35 Tiling/Kernel 与 binary config 闭合 |
| Host/Launcher | ACLNN/L0 包装可见；核心 AICore 平台实现缺失 | 同 A2 | 本文所述 dtype 分流、arch35 Tiling 与 AICPU fallback |
| Kernel/资源 | 固定仓只可见 AICPU body，不能覆盖公开全部 dtype | 同 A2 | FP16/BF16/FP32 为 AIV LEVEL_2 DAG；DOUBLE/复数为 AICPU |

## 限制与失败边界

- `selfRef==nullptr` 返回 `ACLNN_ERR_PARAM_NULLPTR`；dtype 不在六种列表、shape/rank 校验失败返回 `ACLNN_ERR_PARAM_INVALID`。源码未对 `workspaceSize/executor` 输出指针显式校验，相关宏/框架行为固定仓不可见。
- 非空 AIV Tiling 拒绝零维长、`coreNum==0`、不足以产生一个对齐 tile 的 UB；shape 乘积、`ubFormer int64->int32` 的溢出/窄化没有显式防护。
- AIV Kernel 的 `workspace` 形参未读取，但 Host 固定申请 16 MiB；不能擅自删去该请求，因为 executor/运行时可能依赖其 ABI。
- API 允许非连续 Tensor，但 Contiguous/ViewCopy 的具体 route、POD、Kernel、workspace slice 和同步 primitive 是固定仓不可见的框架边界。
- A2/A3 公开支持与固定仓实现范围不矛盾，但缺少对 FP16/BF16/FP32 的平台构建闭包；依据 `source_closed` gate 不生成 `inplace_log10.md`。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/log/docs/aclnnLog10&aclnnInplaceLog10.md` | 产品支持表、原地原型 | 950 公开性、dtype/shape/layout 与两段式语义 |
| 入口实现 | `math/log/op_api/aclnn_log10.cpp` | `CheckInplaceParams`、两段入口 | 原地包装、空域、Contiguous/Log/ViewCopy、executor |
| 入口直接依赖 | `common/inc/op_api/level2_base.h` | `GetDtypeSupportListV2` | 950 `IsRegBase` dtype 列表选择 |
| L0 selector | `math/log/op_api/log.cpp` | `Log/LogAiCore/LogAiCpu` | 临时输出分配、dtype 分流与 launcher |
| 950 OpDef | `math/log/op_host/log_def.cpp` | `ops::Log` | 三种 AICore dtype 与 `ascend950` 注册 |
| Host/Tiling | `math/log/op_host/arch35/log_tiling_arch35.cpp` | `LogTiling`、`Tiling4Log` | 属性、DAG选择、POD scalar、workspace、key、blockDim |
| POD/模板参数 | `math/log/op_kernel/arch35/log_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | `schMode/attrWork` 可实例化域 |
| Kernel entry | `math/log/op_kernel/log_apt.cpp` | `log/LogKernel` | AIV-only 与 key到 DAG 的映射 |
| Kernel DAG | `math/log/op_kernel/arch35/log_dag.h` | `LogCustom`、两种 DAG | 泛化 FP32 管线、MicroAPI Log、LEVEL_2 |
| AICPU body | `math/log/op_kernel_aicpu/log_aicpu.cpp` | `LogCpuKernel`、`LogCompute/Log10Compute` | dtype map、串并行循环、错误终点 |
| AICPU OpDef | `math/log/op_kernel_aicpu/log_aicpu_def.cpp` | `ops::Log` | AICPU dtype 注册 |
| 构建接线 | `math/log/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | 950 arch35 活动接线与 A2/A3 缺口 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `Log` 条目 | ascend950 AIV binary 与编译选项 |
| 构建接线 | `cmake/func.cmake` | `add_all_modules_sources` | ACLNN/OpDef/Tiling/AICPU 源收集规则 |
| POD 定义 | `opbase: pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData32B` | 32B POD 字段与布局 |
| Tiling helper | `opbase: pkg_inc/op_common/atvoss/elewise/elewise_tiling.h` | `DoTiling32B/SetScalar` | DAG资源参数与 scalar 序列化 |
| Tiling helper | `opbase: src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling/AdaptEleBaseTilingData32B` | core/block/tile 公式与 POD 写入 |
| Device scheduler | `opbase: pkg_inc/op_common/atvoss/elewise/elewise_sch_with_scalar.h` | `ElementwiseSchWithScalar` | Device 重建公式、循环、地址、DAG执行 |
| DAG 元编程 | `opbase: pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | no-op Cast 消除、BufferNum 与 bufferIds |
| 存活分析 | `opbase: pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo/GetBufferNumLevel2` | 六分区计数 |
| Buffer 分配 | `opbase: pkg_inc/op_common/atvoss/util/buffer.h` | `GenerateBufferIdOrder` | ping/pong id、temp复用与释放 |
| Vector primitive | `opbase: pkg_inc/op_common/atvoss/util/vec.h` | `Cast/Muls/Adds/CopyIn/CopyOut` | 计算 API 与 DataCopyPad 实参 |
| 同步 | `opbase: pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 Mutex 协议 |
| AICPU helper | `opbase: aicpu_common/context/utils/kernel_util.cc` | `CeilMultiple` | `ceil(N/core)` 分片长度语义 |

每行文件均已列入 Frontmatter `sources`；`opbase` 是 `ops-math` 当前 include 命中的直接公共依赖，引用时保留其独立 40 位 Commit `d574f5571391390f8b619c6a134b65c819842318`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 32B POD 可复用；三 scalar 连续占 offset 16/20/24 | 字段顺序、`ubFormer` 32 位、scalar 写入顺序不能变 | `EleBaseTilingData32B`、`SetScalar` |
| 调用顺序 | `Contiguous -> Log(temp) -> ViewCopy(selfRef)` | 不得让核心 Log Kernel 直接覆盖可能非连续/别名的 selfRef | `aclnnLog10GetWorkspaceSize`、`l0op::Log` |
| workspace/内存 | AIV 六等长 UB 子区与 16 MiB节点 workspace | 变更 DAG 后必须重算 BufferNum；不能把节点 workspace 当 ACLNN 总 workspace | `DAGSch/GetBufferNumLevel2`、`RunTiling` |
| 同步 | buf id 对应的 MTE2/V/MTE3 Mutex 可复用 | buffer 只有在最后 consumer 释放后才能归还；跨节点保持 executor stream 顺序 | `GetTensor/ReleaseTensor`、`GenerateBufferIdOrder` |
| 精度与尾块 | FP16/BF16 升 FP32后计算、末端 RINT；DMA与寄存器均按有效长度保护 | 不得让 padding lane 参与结果或跳过末端 cast | `GenericDag`、`LogCustom<float>`、`CopyIn/CopyOut` |
| 失败边界 | API、Tiling、AICPU map/ParallelFor 错误均必须向上返回 | 不得用 AICPU 自动掩盖 AIV Tiling/launch 失败；源码只有 dtype 分流 | `CHECK_RET/OP_CHECK_IF/KERNEL_HANDLE_ERROR` |
