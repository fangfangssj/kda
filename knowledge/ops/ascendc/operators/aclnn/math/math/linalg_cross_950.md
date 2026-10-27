---
type: Operator
title: "LinalgCross 950 ACLNN 算子知识"
description: "说明 aclnnLinalgCross 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/docs/aclnnLinalgCross.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_api/aclnn_linalg_cross.h"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_api/aclnn_linalg_cross.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_api/cross.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_host/cross_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_host/arch35/cross_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_kernel/arch35/cross_struct.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_kernel/arch35/cross.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_kernel/cross_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_host/config/ascend950/cross_binary.json"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_kernel_aicpu/cross_aicpu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_kernel_aicpu/cross_aicpu_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/common/inc/aicpu/aicpu_op_def.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/include/op_common/op_host/tiling_base_class.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/aicpu_common/context/utils/bcast.h"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/aicpu_common/context/utils/bcast.cc"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/aicpu_common/context/common/cpu_kernel_utils.cc"
    role: implementation
  - resource: "https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/aicpu_common/context/common/device_sharder.cc"
    role: implementation
created_at: "2026-08-25T10:29:34Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# LinalgCross 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnLinalgCrossGetWorkspaceSize` + `aclnnLinalgCross`；内部核心节点名为 `Cross`。 |
| 计算语义 | 沿 `dim` 上长度为 3 的向量计算右手叉积，并对其它维按 broadcast 结果逐向量执行。 |
| 输入输出 | `self/other/out` 为 ND、同 dtype、公开 rank 0～8、可非连续；实际非空计算要求存在合法 `dim`，故 rank 0 无可用轴。950 API dtype 为 INT8/INT16/INT32/INT64/UINT8/FLOAT16/BFLOAT16/FLOAT/FLOAT64/COMPLEX64/COMPLEX128。 |
| 执行类型 | FLOAT16/BFLOAT16/FLOAT 核心为 AIV-only entry 内嵌 SIMT VF；其余公开 dtype 为 AICPU。`Contiguous/ViewCopy` 是框架节点。无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 1024-thread SIMT；R-002 512-thread/int64-base SIMT；R-003 AICPU 串行；R-004 AICPU `ParallelFor`。空输出在 ACLNN Host 短路，不启动核心节点。 |
| 核心资源 | SIMT 路线直接读写 GM，无 TQue/TBuf/dynamic UB、无原子、Cross 节点 workspace 为 0；AICPU 使用两张 `int64` broadcast 索引表、CPU shard 和配置的 100 B workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN、950 executor 组图、dtype selector、Host/Tiling/POD、AIV entry、SIMT body、AICPU broadcast/body 与构建接线均闭合；通用 `Contiguous/ViewCopy` 和 executor 调度是明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `S = broadcast(self.shape, other.shape)`，`d` 为归一化后的叉积轴，且 `S[d]=3`。固定其它坐标 `q`，记 `a_k=self[q,d=k]`、`b_k=other[q,d=k]`，广播维坐标按输入维长 1 回落到 0，则：

```text
out[q,0] = a1*b2 - a2*b1
out[q,1] = a2*b0 - a0*b2
out[q,2] = a0*b1 - a1*b0
```

950 SIMT 对 FLOAT16/BFLOAT16/FLOAT 都先把每个标量显式转换为 `float`，以 FP32 完成两次乘法和一次减法，再转换回输入 `T`；源码没有显式 FMA、舍入模式、NaN/Inf 修正或饱和。AICPU 保持模板 `T` 做 C++ `*`、`-` 和赋值：DOUBLE/COMPLEX 保持自身类型，整数遵循该模板表达式的语言级算术和目标类型转换；固定源码未另行定义溢出饱和策略。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；公开 rank 0～8；可非连续 | INT8、INT16、INT32、INT64、UINT8、FLOAT16、BFLOAT16、FLOAT、FLOAT64、COMPLEX64、COMPLEX128 | 非空指针；与 `other` 可广播；与 `other/out` dtype 相同；`dim` 先按 `self.rank` 校验。 |
| `other` | ND；公开 rank 0～8；可非连续 | 与 `self` 相同 | 非空指针；与 `self` 可广播。950 SIMT Tiling 进一步要求两输入 rank 相同且两者在叉积轴均为 3；AICPU 可左补 1 处理不同 rank。 |
| `dim` | Host 标量 | `int64_t` | 文档默认 `-1`，C API 原型仍要求显式传入；范围 `[-self.rank,self.rank-1]`。负值在 ACLNN 层加实际核心输入 rank 转正。 |
| `out` | ND；shape 等于广播 shape；可非连续 | 与两输入相同 | 非空指针；空输出在第一段成功短路。 |
| `workspaceSize` | Host 输出 | `uint64_t` | executor 汇总值；SIMT Cross 节点自身为 0，AICPU OpDef 默认配置为 100 B，框架前后处理可能增加总量。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并转移任务图及临时 Tensor 所有权。 |
| `workspace` | Device 地址 | `void*` | 第二段使用第一段返回的大小；Cross SIMT entry 保留参数但不读取。 |
| `stream` | runtime stream | `aclrtStream` | 第二段交给 `CommonOpExecutorRun`。 |

### 入口与版本

公开入口为未带版本后缀的 `aclnnLinalgCrossGetWorkspaceSize/aclnnLinalgCross`，固定仓没有更高版本的同语义 ACLNN。950 公共支持由文档产品表中的 Ascend 950PR/950DT、`Cross` OpDef 的 `ascend950` AICore config、950 binary 中的 BF16/F16/F32 三个实例，以及算子 CMake 的 `SUPPORT_COMPUTE_UNIT "ascend950"`/`TILING_DIR "arch35"` 共同证明。

公开表同时列出 A2/A3；L0 对 DAV_2201 有 `CrossV2` selector，公共 binary config 也列出 `CrossV2` 的 910B/910_93 信号，但本固定主目录与活动 CMake 没有这些平台的 Host/Tiling/POD/Kernel body。因而不生成无后缀 A2/A3 卡，不能把 `experimental/math/cross` 的未指定实现冒充当前主线依赖。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecLinalgCrossGetWorkspaceSize` 的真实顺序是：

1. `CREATE_EXECUTOR`，创建失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`；`CheckNotNull` 检查三 tensor。
2. 950/DAV_3510 使用含 BF16 的白名单；随后要求 `self/other/out` dtype 完全一致。
3. `CheckShape` 对三 tensor 执行最大 8 维检查；`CheckBroadcastShape` 校验两输入可广播且广播 shape 等于 `out.viewShape`；`CheckDim` 用 `self.rank` 校验 `dim`。
4. 若 `out->IsEmpty()`，直接取得当前 executor workspace、转移 executor 并成功返回，不调用 Contiguous、Cross 或 ViewCopy。
5. 分别调用 `l0op::Contiguous(self/other)`。这两个框架节点把后续核心输入变为连续 ND tensor，失败立即返回。
6. 950 `IsRegBase()==true`，即使两输入 shape 不同也跳过 ACLNN 层 `BroadcastTo`；广播留给核心实现。非 950 才会将两侧都物化到 `out.shape`，该分支不属于本卡。
7. `dimInner=dim`。只有非 950 显式广播分支才会在 `self.rank<other.rank && dim>=0` 时增加左补维数；950 不做这次正轴平移。若 `dimInner<0`，加 `selfBroadCast.rank` 转正。
8. 检查 `selfBroadCast.shape[dimInner]==3`。950 此对象仍是原 `self` 形状；`other` 叉积轴是否为 3 留到 SIMT Tiling，AICPU 则只检查广播输出该轴。
9. 950 不走 BF16→FLOAT 的前后 Cast。`l0op::Cross` 分配广播 shape 的临时 `crossOut`，再按 dtype 选择 AICore 或 AICPU。
10. `l0op::ViewCopy(crossOut,out)` 写用户输出；取得 executor 总 workspace 并转移所有权。第二段仅调用 `CommonOpExecutorRun`。

### 参数、Workspace 与生命周期

用户持有 `self/other/out/workspace`。`uniqueExecutor` 持有 Contiguous 结果、`crossOut`、ViewCopy 参数与全部任务；`ReleaseTo(executor)` 后所有权转入返回句柄，直至第二段 stream 上的消费者完成。

SIMT Tiling 在 `PostTiling` 写 `workspaces[0]=0`，entry 的 `workspace` 参数未读取。AICPU `ApplyMathAicpuDefaultCfg` 配置 `libmath_aicpu_kernels.so`、`CUSTAICPUKernel` 与 `opInfo.workspaceSize="100"`，body 未显式读取该空间。ACLNN 对外总 `workspaceSize=uniqueExecutor->GetWorkspaceSize()`，其包含框架节点和临时对象的汇总算法在固定算子源码中不可见，不能简化为 0 或 100。

SIMT Host 的实际模板顺序由 `TilingBaseClass::DoTiling` 固定为：`GetShapeAttrsInfo`（依次执行基础 shape/attr、broadcast/merged shape、stride/vector 计算）→`GetPlatformInfo`→恒真的 `IsCapable`→`DoOpTiling`→空实现 `DoLibApiTiling`→空实现 `GetWorkspaceSize`→`PostTiling`（写 0 B workspace、blockDim 和 320 B POD）→`SetTilingKey(0)`。任一步非成功都立即返回，不启动 Kernel。

### 启动图与失败传播

```text
self  -> [framework Contiguous] --\
                                      -> l0op::Cross selector
other -> [framework Contiguous] --/       -> R-001/R-002: AIV entry `cross`
                                           -> R-003/R-004: AICPU `CrossCpuKernel`
                                      -> crossOut (executor-owned GM)
                                      -> [framework ViewCopy] -> user out
```

| 节点/边 | 启动与参数 | 介质、ordering 与 lifetime | 失败传播 |
|---|---|---|---|
| AIV `Cross` | `CrossAiCore` 经 `ADD_TO_LAUNCHER_LIST_AICORE`；`blockDim=min(totalVectors,AIV核数)`；POD=`CrossRegbaseTilingData`；每 AIV block 内 `asc_vf_call` 启动 1024 或 512 threads | 两个连续输入 GM→输出临时 GM；无节点私有 workspace；同一 executor/stream 数据依赖 | Tiling、launcher 或 kernel 失败使任务失败，后续 ViewCopy 不得到有效输入 |
| AICPU `Cross` | `CrossAiCpu` 经 `ADD_TO_LAUNCHER_LIST_AICPU`；AICPU context 传两输入、一输出和 `dim` | 连续输入 GM；AICPU heap 索引表；临时输出 GM；executor 保持对象到 ViewCopy 完成 | 参数、broadcast 或 `ParallelFor` 返回非 OK 时向上传播 |
| `Cross -> ViewCopy` | producer 为上述二选一核心，consumer 为框架节点 | `crossOut` 的 shape/dtype 与广播输出一致；物理 workspace offset 和框架 event 细节固定仓不可见 | 核心失败阻断消费；ViewCopy 失败由 executor 返回 |

## Route 与参数设计

### 分核与资源计算

令 `r=self.rank=other.rank`（SIMT Tiling 的成功前提）、`d=normalizedDim`。对 `i!=d`，`M[i]=max(X1[i],X2[i])`；对 `i==d`，`M[d]=1`。Host 反向计算 row-major stride：输入维长为 1 时对应 `xStride[i]=0`，否则是原输入 stride；`mergedStride` 对 `M` 计算；`yStride` 对输出形状（叉积轴为 3）计算。

```text
totalVectors = product(M[i], i != d)
ySize        = 3 * totalVectors
C            = compileInfo.coreNum = GetCoreNumAiv()
vectorsPerCore = floor(totalVectors / C)
formerCore     = totalVectors % C
blockDim        = min(totalVectors, C)

block b start = b<formerCore ? b*(vectorsPerCore+1)
                              : formerCore+b*vectorsPerCore
block b count = vectorsPerCore + (b<formerCore ? 1 : 0)
```

`usedInt64=(ySize>2147483647)` 仅改变 SIMT body 与线程数：false 走 1024 threads/int32 base，true 走 512 threads/int64 base。它不分配 UB，也不改变 blockDim。注意 POD 的 stride 是 int64，但 `Cross::Init` 无条件收窄到 `CrossStridePara::int32_t[8]`；`totalVectors/vectorsPerCore` 等也收窄为 int32 成员。因此 R-002 只把最终 base 累加改为 int64，并没有形成一般意义上的全 64 位 shape/stride 闭包。

AICPU 令 `N=out.NumElements()`、`V=N/3`。`BCastIndexes` 为每个输出元素生成 `x_indexes/y_indexes`；`stride_` 是输出 row-major stride，`data_stride_=stride_[dim]`。`V<=2048` 直接单 shard；否则 `max_core=max(1,GetCPUNum(ctx)-2)`、`grain=V/min(V,max_core)`，调用 `ParallelFor(ctx,V,grain,work)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 1024-thread SIMT | TilingKey `0`；`Cross<T>`；`SimtCrossCompute<T>` | 950、非空、dtype∈{FLOAT16,BFLOAT16,FLOAT}；950 SIMT Tiling 成功；`ySize<=INT_MAX` | `ySize>INT_MAX` 到 R-002；shape/Tiling 不满足则失败，无 AICPU shape fallback |
| 2 | R-002 512-thread/int64-base SIMT | TilingKey `0`；`Cross<T>`；`SimtCrossComputeInt64<T>` | 同 R-001，但 `ySize>INT_MAX` | 无其它 AICore fallback；窄化 guard 不充分，见限制 |
| 3 | R-003 AICPU 串行 | 无 TilingKey；`CrossBcastCompute<T>` | 950、非空、dtype∈{INT8,INT16,INT32,INT64,UINT8,DOUBLE,COMPLEX64,COMPLEX128}；broadcast helper 成功；`out.NumElements()/3<=2048` | 大于阈值到 R-004；dtype/broadcast/dim 无效则失败 |
| 4 | R-004 AICPU 并行 | 同 R-003 + `ParallelFor` | 同 R-003，但 `out.NumElements()/3>2048` | sharder 不可用时 opbase DeviceSharder 回落单 `work(0,total)`；参数错误则失败 |

```text
if out.empty: success without core Route
if dtype in {F16,BF16,F32}:                  # IsRegBase AICore list
    require SIMT Tiling shape guards
    if 3*totalVectors <= INT_MAX: R-001
    else:                            R-002
else:                                        # public dtype subset supported by AICPU
    GenerateBcastInfo; BCastIndexes; GetBcastVec
    if totalVectors <= 2048:          R-003
    else:                             R-004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 1024-thread SIMT | key 0 + `usedInt64=false` | 950；F16/BF16/F32；同 rank 1～8；两输入叉积轴均 3；其它轴逐维相等或一侧为 1；非空 | 单 AIV `cross`，每 block 一次 1024-thread VF | `Cross<T>::Process -> SimtCrossCompute<T>` | 直接 GM；线程/核输出域不重叠，无 barrier/atomic/workspace | 每线程 `i=tid;i<count;i+=1024`；blockDim 已去除空核 |
| R-002 512-thread/int64-base SIMT | key 0 + `usedInt64=true` | 与 R-001 相同，且 Host `ySize>INT_MAX` | 单 AIV `cross`，每 block 一次 512-thread VF | `Cross<T>::Process -> SimtCrossComputeInt64<T>` | GM base 为 int64；POD→Device stride 与 vector index 仍有 int32 窄化 | 每线程步长 512；无 fallback；超 int32 中间量不受完整保护 |
| R-003 AICPU 串行 | `IsAiCoreSupport=false` + `V<=2048` | 950；八种 fallback dtype；两输入/输出可广播；输出叉积轴为 3 | 单 AICPU kernel | `CrossCpuKernel::Compute -> CrossBcastCompute<T> -> CrossCompute<T> -> ParallelCompute<T>` | GM + 两张 `N` 项 int64 索引表；调用栈内单 shard | `[0,V)` 精确覆盖；broadcast/dim 错误返回 PARAM_INVALID |
| R-004 AICPU 并行 | 同上 + `V>2048` | 同 R-003 | 单 AICPU kernel 内 CPU shards | 同 R-003 + `CpuKernelUtils::ParallelFor` | shard 写互斥向量；`ParallelFor` 返回后节点完成 | grain 由可用 CPU 数决定；runtime sharder 缺失时串行回落 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 AICore dtype | `IsRegBase && dtype∈{F32,F16,BF16}` | OpDef/binary 只实例化三种 `Cross<T>` | R-001/R-002 | AICPU | `math/cross/op_api/cross.cpp::IsAiCoreSupport` |
| 950 同 rank | `dimNum1==dimNum2` | SIMT POD 用同一维号和 8 项 stride 数组同时解释两输入 | R-001/R-002 | Tiling `GRAPH_FAILED`；没有 AICPU fallback | `CrossTiling::CheckBaseShapeAndAttrs` |
| 叉积轴长度 | `x1[d]==3 && x2[d]==3` | 每线程固定读取轴上的 0/1/2 三个分量 | R-001/R-002 | Tiling 失败 | 同上 |
| 非叉积轴广播 | `x1[i]==x2[i] \|\| x1[i]==1 \|\| x2[i]==1` | 长度 1 的输入 stride 被置 0，重复读同一 GM 元素 | R-001/R-002 | Tiling 失败 | `CheckBroadcastAndMergeShape`; `CalcStrideAndVectors` |
| int64-base 切换 | `ySize>INT_MAX` | 输出 flat base 可能超过 int32，切到 int64 累加并降为 512 threads | R-002 | 否则 R-001 | `CrossTiling::CheckBroadcastAndMergeShape`; `Cross::Process` |
| Device 窄化 | 源码没有 `totalVectors/stride<=INT_MAX` guard | POD int64 值写入 Device int32 成员/数组，R-002 仍可能截断 | R-001/R-002 | 无 fallback；属于失败边界 | `Cross::Init`; `CrossStridePara` |
| AICPU 并行阈值 | `V>2048` | 大工作量才支付 CPU 分片开销 | R-003/R-004 | 小于等于阈值直接本线程执行 | `CrossCpuKernel::CrossCompute` |
| AICPU broadcast 表 | `shape_out[i]==max(leftPad(x)[i],leftPad(y)[i])` 且不等维只能一侧为 1 | `BCastIndexes` 才能为每个输出 flat index生成合法源 offset | R-003/R-004 | PARAM_INVALID | `Bcast::GenerateBcastInfo` |
| 空输出 | `out->IsEmpty()` | 核心均不支持/无需处理空域 | 全部 | ACLNN 第一段成功，无核心节点 | `ExecLinalgCrossGetWorkspaceSize` |

### POD/ABI

`CrossRegbaseTilingData` 使用 `#pragma pack(push,8)`，全部字段为 64 位，`alignof=8`、`sizeof=320`。Host 原样 `memcpy_s` 到 raw tiling，entry 以同一声明解析；下表 offset 由字段声明顺序确定。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CrossRegbaseTilingData` | `totalVectors` | `int64_t`/64 | offset 0 | 向量 | `product(mergedShape[i],i!=d)` | `Cross::Init -> int32 totalVectors_`，后续未直接使用 | 显式/隐式窄化保护缺失 |
| 同上 | `vectorsPerCore` | `int64_t`/64 | offset 8 | 向量/core | `totalVectors/coreNum` | `Process` 计算 start/count，先收窄 int32 | 必须适配 int32 才可靠 |
| 同上 | `coreNum` | `int64_t`/64 | offset 16 | AIV core | `GetCoreNumAiv()` | `Cross::Init -> int32 coreNum_`；body 不再读取 | blockDim 使用 `min(totalVectors,coreNum)` |
| 同上 | `dim` | `int64_t`/64 | offset 24 | 轴索引 | `(dim+rank)%rank` | 两个 SIMT body 取 `para.*[dim]` | 0～7 |
| 同上 | `dimNum` | `int64_t`/64 | offset 32 | 维数 | 输入 rank | SIMT 坐标分解循环上界 | 成功路径 1～8 |
| 同上 | `mergedStride[8]` | `int64_t[8]`/512 | offset 40 | 向量线性 stride | `CalcStrideAndVectors`; inactive 项填 1 | `para_.ms[i]=...` 后用于 `v/ms[d]` | Device 收窄为 int32；值不可为 0 |
| 同上 | `x1Stride[8]` | `int64_t[8]`/512 | offset 104 | 输入元素 | 原 shape row-major stride；广播维填 0；inactive 填 0 | 计算 `x1b` 与轴步长 | Device 收窄为 int32 |
| 同上 | `x2Stride[8]` | `int64_t[8]`/512 | offset 168 | 输入元素 | 同上，针对 x2 | 计算 `x2b` 与轴步长 | Device 收窄为 int32 |
| 同上 | `yStride[8]` | `int64_t[8]`/512 | offset 232 | 输出元素 | 输出形状 row-major stride；inactive 填 0 | 计算 `yb` 与三分量地址 | Device 收窄为 int32 |
| 同上 | `dimStride` | `int64_t`/64 | offset 296 | 元素 | 叉积轴之后 `mergedShape` 的乘积 | `Cross::Init` 收窄并传给 SIMT body | 两个 body 参数均未读取，inactive ABI 字段 |
| 同上 | `formerCore` | `int64_t`/64 | offset 304 | core 数 | `totalVectors%coreNum` | 决定前多少 block 多处理一个向量 | 小于硬件 AIV 核数 |
| 同上 | `usedInt64` | `int64_t`/64 | offset 312 | bool 标志 | `ySize>INT_MAX` | `Cross::Init -> bool`，选择 1024/512-thread body | 不验证其它被窄化字段 |
| AICPU context | `x1/x2/y/dim` | runtime Tensor/attr | 固定仓公开 context API，任务包 offset 不可见 | tensor/轴 | `ADD_TO_LAUNCHER_LIST_AICPU` | `CpuKernelContext::Input/Output/GetAttr` | 私有序列化 ABI 固定仓不可见 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 0/`usedInt64=false` | `ExecLinalgCrossGetWorkspaceSize`; `l0op::Cross`; `CrossTiling::DoTiling` | `CrossRegbaseTilingData` | `cross` -> `Cross<DTYPE_X1>` | `Cross::Process`; `SimtCrossCompute` | `cross_def.cpp`; 950 binary；CMake `ascend950/arch35` |
| R-002 | key 0/`usedInt64=true` | 同 R-001 | 同一 POD | 同一 entry/router | `SimtCrossComputeInt64` | 同 R-001 |
| R-003 | AICPU selector/`V<=2048` | `l0op::Cross/CrossAiCpu` | AICPU context + `BCalcInfo` | `REGISTER_CPU_KERNEL("Cross",CrossCpuKernel)` | `Bcast`; `CrossCpuKernel` 串行 body | `cross_aicpu_def.cpp`; `add_all_modules_sources` |
| R-004 | AICPU selector/`V>2048` | 同 R-003 | 同上 | 同上 | 同上 + `CpuKernelUtils::ParallelFor` | 同 R-003 |

## Kernel 实现

### 公共执行流程

R-001/R-002 共用单 AIV Kernel：entry 解析 320 B POD，TilingKey 仅接受 0，实例化 `Cross<DTYPE_X1>`；每个 AIV block 获得一段连续的“逻辑三维向量编号”，再在该 block 内以 `asc_vf_call` 启动 SIMT threads。每个 thread 将 vector id 按 `mergedStride` 还原为多维坐标，通过输入 stride 0 表达广播，直接从 GM 读六个标量并向 GM 写三个结果。不存在 CopyIn/CopyOut、UB queue 或跨线程归约。

R-003/R-004 共用 AICPU body：先生成 broadcast 元数据和两张 flat-index 映射表，再把输出空间除叉积轴后的每个三分量向量交给 `ParallelCompute`。串行/并行只改变 shard 调度，不改变 `SetResult<T>` 的地址和算术。

### Route R-001：1024-thread SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/L0 | `math/cross/op_api/cross.cpp` | `IsAiCoreSupport`; `Cross` | 950 RegBase 且 dtype∈{F32,F16,BF16} | `CrossAiCore` |
| Tiling | `math/cross/op_host/arch35/cross_tiling.cpp` | `CheckBaseShapeAndAttrs`; `CheckBroadcastAndMergeShape` | 同 rank、两侧轴长3、其它维可广播；`ySize<=INT_MAX` | key 0，`usedInt64=0` |
| Router | `math/cross/op_kernel/cross_apt.cpp` | `cross` | `TILING_KEY_IS(CROSS_DEFAULT)` | `Cross<DTYPE_X1>::Process` |
| Device | `math/cross/op_kernel/arch35/cross.h` | `Cross::Process` | `!usedInt64_` | `asc_vf_call<SimtCrossCompute<T>>(dim3(1024),...)` |

#### Device 数据流

##### 函数级调用链

```text
CrossAiCore -> AIV entry cross(x1,x2,y,workspace,tiling)
  -> GET_TILING_DATA(CrossRegbaseTilingData)
  -> Cross<T>::Init
  -> Cross<T>::Process
     -> asc_vf_call<SimtCrossCompute<T>>(1024 threads)
        -> direct GM loads -> FP32 cross product -> direct GM stores
```

##### 源码等价伪码

```text
# 伪码
b = GetBlockIdx()
start = b<formerCore ? b*(vectorsPerCore+1) : formerCore+b*vectorsPerCore
count = vectorsPerCore + (b<formerCore)
for i = threadIdx.x; i < count; i += 1024:
    v = start+i; x1b=x2b=yb=0
    for d in [0,dimNum):
        c = v / mergedStride[d]; v -= c*mergedStride[d]
        x1b += c*x1Stride[d]; x2b += c*x2Stride[d]; yb += c*yStride[d]
    a0,a1,a2 = float(x1[x1b + {0,1,2}*x1Stride[dim]])
    b0,b1,b2 = float(x2[x2b + {0,1,2}*x2Stride[dim]])
    y[yb + 0*yStride[dim]] = T(a1*b2-a2*b1)
    y[yb + 1*yStride[dim]] = T(a2*b0-a0*b2)
    y[yb + 2*yStride[dim]] = T(a0*b1-a1*b0)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 GM | `x1b+k*x1Stride[d]` | `product(x1.shape)*sizeof(T)` | T/连续 ND | Contiguous/用户 | SIMT thread，k=0..2 | executor/用户至核心完成 | broadcast 维 stride 0，可被多 thread重复读 |
| x2 GM | `x2b+k*x2Stride[d]` | `product(x2.shape)*sizeof(T)` | T/连续 ND | Contiguous/用户 | SIMT thread | 同上 | 同上 |
| y GM | `yb+k*yStride[d]` | `3*totalVectors*sizeof(T)` | T/广播输出 ND | 当前 SIMT thread | ViewCopy | executor 临时至 ViewCopy 完成 | 每 vector 唯一三地址，无冲突 |
| thread local | 六个 FP32 分量、三个 base、坐标临时 | 固定仓不公开寄存器字节数 | FP32/int32 | 当前 thread | 当前 thread | 一次 loop 迭代 | 无跨线程共享 |
| workspace | 参数存在但未读取 | 0 B | N/A | Tiling | N/A | runtime | 不可当算法中间存储 |

##### 分层内存布局

N/A：该 Route 不使用 AIV UB/TQue/TBuf，也不请求 SIMT dynamic UB；`asc_vf_call` 的每个 thread 直接在 GM 与线程私有标量之间搬运。

##### 字节账与复用顺序

```text
UB_total = 0 B
dynamic_ub = N/A
workspace_total(Cross node) = 0 B
GM reads/vector  = 6*sizeof(T)
GM writes/vector = 3*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private `a0..b2/x1b..yb` | 当前 loop 迭代 | 同迭代算术/地址 | `i+=1024` 后下一迭代 | 单 thread 程序顺序 | 不跨 thread共享 |
| x1/x2 broadcast 元素 | 上游 producer | 多个 SIMT thread只读 | 核心完成后才可回收 | executor 数据依赖 | 核心期间不可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| block 分区 | Host POD + `GetBlockIdx` | Tiling | 每个 AIV block | vector id 区间 | former block 多1个；不启动空 block |
| thread 分区 | `threadIdx.x/blockDim.x` | SIMT launch | thread loop | block 内 vector ids | `i<count` 保护尾部；步长1024 |
| 计算 | 无 barrier/atomic | thread GM loads | 同 thread GM stores | 唯一输出三元组 | T→FP32→T；降精度舍入细节固定 primitive/compiler不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cross/op_host/arch35/cross_tiling.cpp` | `CalcStrideAndVectors`; `DoOpTiling` | 零 stride广播、核区间、POD与blockDim |
| Kernel entry/router | `math/cross/op_kernel/cross_apt.cpp` | `cross`; `Cross<DTYPE_X1>` | key 0映射到唯一模板 |
| Device body/helper | `math/cross/op_kernel/arch35/cross.h` | `Cross::Process`; `SimtCrossCompute<T>` | 1024 threads、坐标分解、FP32算术与GM地址 |

### Route R-002：512-thread/int64-base SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `math/cross/op_host/arch35/cross_tiling.cpp` | `CheckBroadcastAndMergeShape` | `ySize_>2147483647` | `tilingData.usedInt64=1` |
| Tiling | 同上 | `DoOpTiling` | 与 R-001 相同的 shape/dtype条件 | key仍为0，POD其余公式相同 |
| Device | `math/cross/op_kernel/arch35/cross.h` | `Cross::Process` | `usedInt64_==true` | `asc_vf_call<SimtCrossComputeInt64<T>>(dim3(512),...)` |

#### Device 数据流

##### 函数级调用链

```text
CrossAiCore -> cross -> Cross<T>::Init -> Cross<T>::Process
  -> asc_vf_call<SimtCrossComputeInt64<T>>(512 threads)
     -> int64 x1b/x2b/yb accumulation -> FP32 cross product -> GM stores
```

##### 源码等价伪码

```text
# 伪码；与 R-001 相同，差异如下
for i = threadIdx.x; i < count; i += 512:
    int32 v = start+i
    int64 x1b=0,x2b=0,yb=0
    for d in [0,dimNum):
        int64 m=para.ms[d], s1=para.x1s[d], s2=para.x2s[d], sy=para.ys[d]
        int64 c=int64(v)/m
        v=int32(int64(v)-c*m)
        x1b+=c*s1; x2b+=c*s2; yb+=c*sy
    load/compute/store as R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | 同 R-001，但 base 累加变量为 int64 | 各 tensor 逻辑字节数 | T/连续 ND | 上游/当前 thread | 当前 thread/ViewCopy | executor | stride 来源已先收窄为 int32 |
| thread local | int32 `v` + int64 base/stride/c + FP32分量 | 寄存器字节数固定仓不可见 | scalar | 当前 thread | 当前 thread | 一次迭代 | 512 threads 降低每 block 并发；源码未给资源定量原因 |
| workspace | N/A | 0 B | N/A | Tiling | N/A | runtime | 无复用 |

##### 分层内存布局

N/A：与 R-001 一样无 AIV UB/TQue/TBuf/dynamic UB，差异只在线程规模和 thread-private 整数位宽。

##### 字节账与复用顺序

```text
UB_total = 0 B
dynamic_ub = N/A
workspace_total(Cross node) = 0 B
GM traffic/vector = 9*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private bases | 当前迭代 | 当前迭代 GM 指令 | `i+=512` | 单 thread 顺序 | POD stride 与 vector index 必须未发生有害截断 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread 分区 | 512-thread `asc_vf_call` | AIV block | SIMT body | vector ids | `i<count` 尾保护 |
| 地址 | int64 base 累加 | 坐标分解 | GM load/store | 大于2Gi元素的flat output | stride/vector id仍源自int32，不能据此宣称全64位安全 |
| 算术 | 同 R-001 | GM loads | GM stores | 三分量 | T→FP32→T |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cross/op_host/arch35/cross_tiling.cpp` | `CheckBroadcastAndMergeShape` | `ySize>INT_MAX` 唯一切换谓词 |
| Kernel entry/router | `math/cross/op_kernel/cross_apt.cpp` | `cross` | 与 R-001共用key 0 |
| Device body/helper | `math/cross/op_kernel/arch35/cross.h` | `SimtCrossComputeInt64<T>` | 512 threads与int64 base；保留int32窄化事实 |

### Route R-003：AICPU 串行

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0 | `math/cross/op_api/cross.cpp` | `IsAiCoreSupport`; `Cross` | 950 dtype不在{F32,F16,BF16} | `CrossAiCpu` |
| AICPU router | `math/cross/op_kernel_aicpu/cross_aicpu.cpp` | `CrossCpuKernel::Compute` | 公开可达八种 dtype switch case | `CrossBcastCompute<T>` |
| Broadcast | `opbase/.../bcast.cc` | `GenerateBcastInfo`; `BCastIndexes` | 左补1后逐维兼容，输出shape等于max | 两张源索引表 |
| Dispatch | `CrossCpuKernel::CrossCompute` | `total<=2048` | 不调用 `ParallelFor` | `cross_shard(0,total)` |

#### Device 数据流

##### 函数级调用链

```text
CrossAiCpu -> ADD_TO_LAUNCHER_LIST_AICPU(Cross)
  -> CrossCpuKernel::Compute -> CrossBcastCompute<T>
     -> Bcast::GenerateBcastInfo -> Bcast::BCastIndexes -> Bcast::GetBcastVec
     -> CrossCompute<T> -> GetDimAndCheck -> InitStride
     -> ParallelCompute<T>(0,total) -> SetResult<T>
```

##### 源码等价伪码

```text
# 伪码
left-pad x/y shapes with 1; validate output=max(x,y)
for each output flat index p:
    x_indexes[p] = source-x flat index (broadcast dims contribute 0)
    y_indexes[p] = source-y flat index
dim = attr; if dim<0: dim += output.rank; require output.shape[dim]==3
stride = row_major_stride(output); ds=stride[dim]; total=NumElements(out)/3
for vector ordinal s in [0,total):
    data_start = mixed-radix coordinate over all axes except dim
    x0,x1,x2 = x[x_indexes[data_start + {0,1,2}*ds]]
    y0,y1,y2 = y[y_indexes[data_start + {0,1,2}*ds]]
    out[data_start + {0,1,2}*ds] = cross(x,y)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `x_indexes[p]`/`y_indexes[p]` | 输入各自逻辑字节数 | T/连续 ND | Contiguous | `SetResult<T>` | executor/用户 | broadcast 通过索引重复读 |
| output GM | `data_start+k*data_stride` | `N*sizeof(T)` | T/广播输出 ND | `SetResult<T>` | ViewCopy | executor临时 | 每 vector 唯一三地址 |
| `x_indexes` | vector offset 0 | 至少 `N*8` B（不含capacity开销） | int64 flat index | `BCastIndexes` | `SetResult` | AICPU函数作用域 | 完整物化，非workspace |
| `y_indexes` | vector offset 0 | 至少 `N*8` B | int64 flat index | `BCastIndexes` | `SetResult` | 同上 | 同上 |
| shape/stride vectors | N/A | O(rank)*8 B，多张vector | int64 | Bcast/InitStride | 地址计算 | AICPU对象/函数作用域 | rank<=8 |
| AICPU workspace | runtime slice | 100 B配置 | opaque | OpDef | body未读取 | AICPU runtime | 与索引表heap不可混同 |

##### 分层内存布局

N/A：该 Route 不使用 AIV UB/TQue/TBuf 或 SIMT dynamic UB。实际局部资源为 AICPU heap 上的两张广播索引表、shape/stride vector 和 CPU 标量状态。

##### 字节账与复用顺序

```text
UB_total = N/A (AICPU)
broadcast_index_payload >= 2*N*sizeof(int64_t) = 16N B
workspace_total(core AICPU config) = 100 B
GM traffic/vector = 9*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `x_indexes/y_indexes` | `BCastIndexes` | 全部 `SetResult` 调用 | kernel函数返回后释放 | 串行函数顺序 | 表必须覆盖N个输出元素 |
| `position_in_dims/data_start` | `ParallelCompute` | 当前vector地址 | 下一vector递增/进位 | 单shard顺序 | 跳过dim轴且三分量由`ds`寻址 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| broadcast metadata | 函数调用顺序 | Bcast helper | CrossCompute | 索引表 | 任一shape错误立即返回 |
| 串行计算 | `cross_shard(0,total)` | 当前CPU线程 | kernel返回 | 全输出 | `[0,total)`无尾遗漏 |
| 数值 | 模板 `T` 的 `*`/`-` | GM标量 | T赋值 | 三分量 | 无升精度/饱和；整数溢出策略未显式定义 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cross/op_api/cross.cpp` | `Cross/CrossAiCpu` | 950非三种浮点dtype进入AICPU |
| Kernel entry/router | `math/cross/op_kernel_aicpu/cross_aicpu.cpp` | `Compute`; `CROSS_COMPUTE_CASE` | 八种公开fallback dtype映射到模板T |
| Device body/helper | 同上；`opbase/.../bcast.cc` | `ParallelCompute`; `SetResult`; `BCastIndexes` | 广播映射、vector枚举和三分量算术 |

### Route R-004：AICPU 并行

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0/AICPU | 与 R-003 相同 | 同上 | dtype/broadcast/dim校验同 R-003 | `CrossCompute<T>` |
| Dispatch | `math/cross/op_kernel_aicpu/cross_aicpu.cpp` | `CrossCompute` | `total>2048` | 计算 `max_core/per_unit_size`，调用 `ParallelFor` |
| Runtime | `opbase/.../cpu_kernel_utils.cc` | `CpuKernelUtils::ParallelFor` | device与sharder非空 | `sharder->ParallelFor(total,grain,work)` |

#### Device 数据流

##### 函数级调用链

```text
CrossCpuKernel::CrossCompute<T>
  -> max_core=max(1,GetCPUNum(ctx)-2)
  -> grain=total/min(total,max_core)
  -> CpuKernelUtils::ParallelFor(total,grain,cross_shard)
     -> each shard ParallelCompute<T>(start,end) -> SetResult<T>
```

##### 源码等价伪码

```text
# 伪码；broadcast表与单vector body同 R-003
if total > 2048:
    workers = min(total,max(1,cpu_num-2))
    grain = total/workers
    ParallelFor(total,grain, lambda(start,end):
        reconstruct data_start for start
        while start<end: SetResult(data_start); advance mixed-radix coordinate)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| shared broadcast tables | 同 R-003 | 至少16N B合计 | int64 | 主AICPU调用 | 所有shard只读 | kernel作用域 | 并发只读，无锁 |
| shard output | 各自vector ordinal `[start,end)` 对应三地址 | `3*(end-start)*sizeof(T)`有效写入 | T/ND | shard | ViewCopy | executor | shard区间不重叠 |
| shard local vector | 每shard一张 `position_in_dims` | `rank*8` B + allocator开销 | int64 | `ParallelCompute` | 当前shard | shard调用期 | 不共享 |

##### 分层内存布局

N/A：没有 AIV UB；并行层级是 AICPU runtime 的 CPU shard，每个 shard共享只读索引表、持有私有坐标状态并写不重叠的 GM 三元组。

##### 字节账与复用顺序

```text
UB_total = N/A (AICPU)
shared_index_payload >= 16N B
per_shard_coordinate_payload >= rank*8 B
workspace_total(core config) = 100 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| shared index tables | 主调用 | 全部shard | 所有ParallelFor work返回后释放 | `ParallelFor`调用完成 | runtime必须在返回前完成work；接口以同步分片函数使用 |
| shard coordinate | 当前shard | 当前shard | 下一个vector | shard程序顺序 | `[start,end)`不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU分片 | `ParallelFor` | CrossCompute | 调用返回 | shard生命周期和输出完成 | runtime决定实际区间；lambda收到精确`start/end` |
| 输出互斥 | 不相交vector ordinal | shard | ViewCopy | GM三元组 | 无atomic/barrier需求 |
| runtime回落 | DeviceSharder函数指针为空时 `work(0,total)` | opbase | CrossCompute | 全输出 | 退化串行但语义相同 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cross/op_kernel_aicpu/cross_aicpu.cpp` | `CrossCompute<T>` | 2048阈值、worker与grain公式 |
| Kernel entry/router | 同上 | `CrossCpuKernel::Compute` | dtype模板分发 |
| Device body/helper | 同上；`opbase/.../cpu_kernel_utils.cc` | `ParallelCompute<T>`; `CpuKernelUtils::ParallelFor` | shard映射、同步调用和错误边界 |

## 算子特定优化

### Trick T-001：零 stride 在核心内吸收同 rank 广播

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：950、两输入 rank 相同；非叉积轴上维度相等或一侧为 1；叉积轴两侧均为 3。
- Golden 来源：[`cross_tiling.cpp:108-133`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_host/arch35/cross_tiling.cpp#L108-L133)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
x1Stride_[i] = (x1Dims_[i] == 1) ? 0 : stride[0];
x2Stride_[i] = (x2Dims_[i] == 1) ? 0 : stride[1];
mergedStride_[i] = stride[2];
yStride_[i] = stride[3];
```

#### 机制与设计原因

ACLNN 在 950 用 `!IsRegBase()` 阻止 shape 不同输入先做 `BroadcastTo` 物化。Tiling 把长度 1 的源维 stride 设为 0；SIMT 坐标仍按广播后的 `mergedStride` 枚举，但 GM base 在该维不前进。这样直接从源 tensor复用一个标量，避免生成两个完整广播临时 Tensor及其额外 GM 读写。

#### 迁移边界

- 适用：逐输出向量独立、可用 stride-0 表达广播的 SIMT pointwise/小向量算子。
- 前提：同 rank、row-major连续核心输入、每个输出坐标到源地址的映射可纯算术获得。
- 失效：不同 rank 在当前 950 SIMT Tiling 直接失败；不能只把缺失前导维当成数组外的隐式 stride 0。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 950 分支重新改成无条件 `BroadcastTo` 会物化广播结果并改变 executor workspace/GM 流量；把不同 rank直接交给当前 `CrossTiling` 又会命中 `dimNum1_ != dimNum2_` 失败。

### Trick T-002：每个 SIMT thread 一次完成一个三分量向量

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：Host 已按 vector ordinal 分核；thread id 落在当前 block 的 `count` 内。
- Golden 来源：[`cross.h:105-129`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cross/op_kernel/arch35/cross.h#L105-L129)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
while (i < count) {
    int32_t v = startIdx + i;
    int32_t x1b = 0, x2b = 0, yb = 0;
```

#### 机制与设计原因

同一个 thread 计算该向量的三个输出，六个输入分量只在 thread-private 标量中保留；不需要把三个分量拆给线程后再做 shuffle/barrier，也不需要 UB 中间物化。`i+=blockDim.x` 允许一个 thread处理多个向量，`i<count` 原生封闭尾部。

#### 迁移边界

- 适用：每个短向量内部依赖紧密、向量之间完全独立的小维计算。
- 前提：单向量寄存器状态可控，输出地址无重叠，广播只读地址允许重复访问。
- 失效：需要跨向量归约/共享数据时必须增加同步或重新分工；大 shape 必须先解决本实现的 int32 窄化边界。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证的其它线程分工反例；不能据源码声称具体性能百分比。

## 模板与实现对比

| 对比维度 | R-001 1024-thread SIMT | R-002 512-thread/int64-base SIMT | R-003 AICPU 串行 | R-004 AICPU 并行 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | 三浮点dtype且`ySize<=INT_MAX` | 三浮点dtype且`ySize>INT_MAX` | fallback dtype且`V<=2048` | fallback dtype且`V>2048` | L0 dtype→Host size/CPU阈值 |
| 模板与 dtype | `Cross<F16/BF16/F32>` | 同模板 | `CrossBcastCompute<T>`八种实例 | 同 R-003 | binary与AICPU switch |
| 循环与分核 | AIV block+1024-thread grid-stride | AIV block+512-thread grid-stride | 单CPU shard | `ParallelFor` shards | 大flat base改int64；大CPU工作量分片 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅thread-private scalar；无UB | 更多int64 scalar；无UB | CPU vector/heap | 共享索引表+shard私有坐标 | 两执行族均无AIV queue |
| 内存总占用与复用 | GM traffic 9s/vector；workspace 0 | 同左 | 索引payload至少16N B；workspace配置100 B | 同左+每shard O(rank) | SIMT算址换物化；AICPU预建表 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `asc_vf_call(1024)`直接GM | `asc_vf_call(512)`直接GM | CPU scalar load/store | `ParallelFor`+CPU scalar | 无NDDMA/通信 |
| 同步、精度与 tail | 无barrier；FP32算术；`i<count` | 同左；int64 base | 调用顺序；T算术；`[0,V)` | ParallelFor返回；不交叠shard | 由独占输出免同步 |
| fallback/失败边界 | shape/Tiling失败，无AICPU fallback | 窄化未完整保护 | broadcast/dim/dtype失败 | 同左；sharder可串行回落 | selector与错误返回 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持；当前固定主线缺 Host/Tiling/POD/Kernel body，未生成卡 | 同 A2 | 支持；公开 API、SIMT三浮点dtype与AICPU fallback闭合 |
| Host/Launcher | DAV_2201 `CrossV2` selector/binary信号可见，但实现闭包缺失 | 同左 | DAV_3510 `Cross`；shape差异不预物化BroadcastTo |
| Kernel/资源 | 固定主线不可见 | 固定主线不可见 | arch35 AIV+SIMT直接GM，或AICPU broadcast表/CPU shard |

## 限制与失败边界

- 空输出在完成 dtype、rank、broadcast 与 `dim` 范围检查后短路；rank 0 没有任何合法 `dim`，因此公开“0～8维”不等于标量可成功计算。
- 950 SIMT 路线要求两输入 rank 相同。公开 ACLNN 允许一般 broadcast，API 的 950 分支又跳过 `BroadcastTo`；不同 rank 的 F16/BF16/F32 会在 `CrossTiling::CheckBaseShapeAndAttrs` 失败，没有自动 AICPU fallback。
- 当 `self.rank<other.rank` 时，950 不执行非 RegBase 分支中的左补维平移；正 `dim` 原样保留，负 `dim` 也先按未广播的 `self.rank` 转成正数。AICPU 随后用广播输出 rank/shape 解释这个正轴，因此两种符号都可能指向错误输出轴或因该轴不为3而失败；SIMT则更早因不同rank失败。
- API只在 ACLNN 层检查 `self` 对应轴为3；SIMT Tiling要求两输入该轴均为3，AICPU只要求广播输出该轴为3并通过索引表允许源轴长度1。故“叉积轴可否广播”在SIMT与AICPU路线不一致。
- Host 所有 shape/stride/POD 算术为 int64，但没有乘法溢出检查；Device 又把 `totalVectors/vectorsPerCore/dim/stride` 收窄为 int32。`usedInt64` 只切换base累加和线程数，不能覆盖所有超 int32 shape，复用时必须补全上界证明。
- AICPU `BCastIndexes` 为输出每个元素各生成两项 int64 索引，payload至少16N B且来自AICPU heap，不计入配置的100 B workspace；源码注释也提示大shape可能导致device侧内存申请过大。
- AICPU整数模板没有显式升精度、饱和或溢出处理；不要把结果宣传为饱和整数叉积。SIMT三浮点dtype统一FP32计算后回写，降精度的具体舍入模式固定源码未显式指定。
- `Contiguous/ViewCopy`、executor总workspace聚合、AICPU任务私有序列化与 runtime stream/event 细节属于固定算子源码外的框架边界；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/cross/docs/aclnnLinalgCross.md` | 产品表、参数表、错误表 | 950公开性、dtype/rank/broadcast/dim语义 |
| 入口声明/约束 | `math/cross/op_api/aclnn_linalg_cross.h` | 两段式API声明 | C ABI与参数顺序 |
| 入口实现 | `math/cross/op_api/aclnn_linalg_cross.cpp` | `CheckParams`; `ExecLinalgCrossGetWorkspaceSize`; `aclnnLinalgCross` | 校验、空域、950广播策略、组图、workspace |
| L0 selector | `math/cross/op_api/cross.cpp` | `IsAiCoreSupport`; `Cross`; `CrossAiCore`; `CrossAiCpu` | 950 dtype路由与任务添加 |
| OpDef | `math/cross/op_host/cross_def.cpp` | `ops::Cross` | 950三种AICore dtype与`cross_apt`接线 |
| Host/Tiling | `math/cross/op_host/arch35/cross_tiling.cpp` | `CrossTiling`; `Tiling4Cross`; `TilingPrepare4Cross` | shape/broadcast、stride、分核、workspace、注册 |
| POD | `math/cross/op_kernel/arch35/cross_struct.h` | `CrossRegbaseTilingData` | 320 B Host/Device ABI |
| Kernel entry/body | `math/cross/op_kernel/cross_apt.cpp` | `cross` | key 0 router与entry参数 |
| Kernel entry/body | `math/cross/op_kernel/arch35/cross.h` | `Cross`; 两个 `SimtCrossCompute` | SIMT线程、GM地址、精度、tail |
| binary config | `math/cross/op_host/config/ascend950/cross_binary.json` | 三个binary项 | 950 BF16/F16/F32实例 |
| AICPU body | `math/cross/op_kernel_aicpu/cross_aicpu.cpp` | `CrossCpuKernel` | dtype dispatch、broadcast消费、串/并行body |
| AICPU config | `math/cross/op_kernel_aicpu/cross_aicpu_def.cpp` | AICPU `ops::Cross` | fallback dtype与默认配置调用 |
| AICPU config | `common/inc/aicpu/aicpu_op_def.h` | `ApplyMathAicpuDefaultCfg` | kernel so、100 B workspace与通用flags |
| 构建接线 | `math/cross/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`; `SUPPORT_TILING_DIR` | 主线仅ascend950/arch35 |
| 构建接线 | `cmake/func.cmake` | `add_all_modules_sources` | L0/L2/Tiling/AICPU/OpDef收集规则 |
| Tiling依赖 | `include/op_common/op_host/tiling_base_class.h` | `TilingBaseClass::DoTiling` | Host阶段真实调用顺序和SetTilingKey |
| AICPU依赖 | `pkg_inc/op_common/aicpu_common/context/utils/bcast.h` | `BCalcInfo`; `Bcast` | broadcast对象与索引表ABI |
| AICPU依赖 | `aicpu_common/context/utils/bcast.cc` | `GenerateBcastInfo`; `BCastIndexes` | 左补1、逐维校验和flat索引生成 |
| AICPU依赖 | `aicpu_common/context/common/cpu_kernel_utils.cc` | `ParallelFor`; `GetCPUNum` | CPU shard转发和失败边界 |
| AICPU依赖 | `aicpu_common/context/common/device_sharder.cc` | `DeviceSharder::ParallelFor` | runtime符号缺失时的串行回落 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 8维stride+分核+位宽标志的320 B POD | 字段顺序/64位/pack(8)；Device当前会收窄 | `CrossRegbaseTilingData`; `Cross::Init` |
| 调用顺序 | 参数校验→空域→双Contiguous→Cross→ViewCopy | 950不能插入非950的BroadcastTo/Cast分支；失败必须短路 | `ExecLinalgCrossGetWorkspaceSize` |
| workspace/内存 | SIMT核心workspace 0；AICPU配置100 B；AICPU索引表至少16N B | 索引heap不是workspace；总workspace不可写死 | `PostTiling`; `ApplyMathAicpuDefaultCfg`; `BCastIndexes` |
| 同步 | SIMT按唯一vector写出免barrier；AICPU shard区间不重叠 | 上游GM在核心完成前有效；ParallelFor返回后才能释放共享表 | `SimtCrossCompute`; `CrossCompute` |
| 精度与尾块 | SIMT T→FP32→T；AICPU保持T；grid-stride和`[start,end)`闭尾 | 不添加FMA/饱和/舍入保证；R-002不等于全64位安全 | 两个SIMT body；`SetResult<T>` |
| 失败边界 | 同rank/轴长/广播、dtype、空域与大shape边界均可复用 | 不把不同rank SIMT、A2/A3缺失实现或无guard大shape宣称为支持 | `CheckBaseShapeAndAttrs`; selector；CMake |
