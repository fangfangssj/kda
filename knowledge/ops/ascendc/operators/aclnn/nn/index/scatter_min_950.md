---
type: Operator
title: "ScatterMin 950 ACLNN 算子知识"
description: "说明 aclnnScatterMin 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/docs/aclnnScatterMin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/op_host/scatter_min_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/op_host/scatter_min_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/op_host/arch35/scatter_min_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/op_kernel/arch35/scatter_min.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_min/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:59:35Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ScatterMin 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnScatterMinGetWorkspaceSize` + `aclnnScatterMin`，公开 ACLNN 两段式接口 |
| 计算语义 | 对每个 `indices[m]`，将 `varRef[indices[m], ...]` 与 `updates[m, ...]` 逐元素取最小值并原地写回；越界 index 在 Kernel 中跳过 |
| 输入输出 | `varRef/updates`：ND，`FLOAT16/FLOAT32/INT32/INT8/UINT8`；`indices`：ND，`INT32/INT64`；输出与 `varRef` 共用内存 |
| 执行类型 | 单节点 `K1 scatter_min`：SIMT（AIV-only Kernel 内 `asc_vf_call`，每个 AIV core 启动 512 个 VF thread） |
| Kernel 路线 | R-001 32-bit 直接 CAS；R-002 FP16 32-bit word CAS；R-003 INT8/UINT8 32-bit word CAS |
| 核心资源 | `var/indices/updates` GM、寄存器局部量、16 MiB 系统 workspace 声明；无 AIV UB/TQue/TBuf，无 dynamic UB |
| 来源闭包 | `source_closed`：公开 ACLNN 文档与 `ACLNNTYPE aclnn` 注册闭合到 `ScatterMin` OpDef、arch35 Tiling/POD、`scatter_min` entry、共享 SIMT body 与 ascend950 构建接线；生成式 ACLNN executor 壳为框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `S = product(varRef.shape[1:])`，`M = numel(indices)`。接口约束为 `updates.shape = indices.shape + varRef.shape[1:]`，因此 `numel(updates) = M * S`。对 `m in [0, M)`、`s in [0, S)`，当 `0 <= indices[m] < varRef.shape[0]` 时执行：

```text
dst = indices[m] * S + s
varRef_flat[dst] = min(varRef_flat[dst], updates_flat[m * S + s])
```

Kernel 对同一目标元素的并发更新使用 CAS 重试；越界 `indices[m]` 直接 `continue`。输出有效域是原 `varRef` 的完整 shape，未命中的位置保持原值。`varRef` 是原地目标；entry 中形式输出参数 `y` 未被 Device body 读取或写入。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | ND，公开 rank 1–8；允许非连续 Tensor | `FLOAT16/FLOAT32/INT32/INT8/UINT8` | 必填、非空；输入/输出原地目标，第一维是 index 上界 |
| `indices` | ND，公开 rank 1–8；允许非连续 Tensor | `INT32/INT64` | 必填、非空；Kernel 将负值或 `>= varFirstDim` 的值跳过 |
| `updates` | ND，公开 rank 1–8；允许非连续 Tensor | 与 `varRef` 相同 | 必填、非空；`updates.shape = indices.shape + varRef.shape[1:]` |
| `useLocking` | 标量属性 | `BOOL` | 默认 `false`；OpDef 注册该属性，但 Tiling/POD/Kernel 均不读取它，950 body 对所有更新始终使用原子 CAS |
| `varRef`（输出） | 与输入 `varRef` 相同 | 与输入 `varRef` 相同 | 与输入共享内存；InferShape/InferDataType 复制输入 0 的 shape/dtype |

### 入口与版本

固定仓公开 `aclnnScatterMinGetWorkspaceSize(varRef, indices, updates, useLocking, workspaceSize, executor)` 与 `aclnnScatterMin(workspace, workspaceSize, executor, stream)`。第一段完成参数校验并生成执行器，第二段在给定 stream 上执行。仓内没有手写 `op_api/aclnn_scatter_min.cpp`；`index/scatter_min/CMakeLists.txt` 以 `OPTYPE scatter_min ACLNNTYPE aclnn` 将 OpDef 接入生成式 ACLNN 框架，此生成式 inner bridge、非连续 Tensor 的 `Contiguous/ViewCopy` 细节和 executor 壳属于固定仓不可见的框架边界。

公开产品表只支持 Ascend 950PR/950DT，并明确不支持 Atlas A2/A3。OpDef 只添加 `ascend950` AICore config，Host/Tiling 与 Kernel 只接线 `arch35/ascend950`，不存在同平台更高版本入口或旧 body 复用分支。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口依次要求 `varRef/indices/updates` 非空，dtype 属于支持集，`varRef.dtype == updates.dtype`，且 shape 满足拼接关系；失败分别返回 `ACLNN_ERR_PARAM_NULLPTR` 或 `ACLNN_ERR_PARAM_INVALID`。非连续 Tensor 被公开声明为支持，但其连续化/回写源码位于生成框架边界，固定仓不能给出具体临时 Tensor 次序和字节数。算子自有 Host 层将输出 shape/dtype 复制自 `var`，注册 `use_locking=false`，随后调用 `Tiling4ScatterMin -> ScatterReduceCommonTiling`。

### 参数、Workspace 与生命周期

`TilingPrepareForScatterReduce` 从 `PlatformAscendC::GetCoreNumAiv()` 生成 `ScatterReduceCompileInfo.coreNum`；ACLNN runtime 若不携带 compileInfo，`ResolveCoreNum` 直接从 platform 读取，读到 0 时回退为 1。

Tiling 以 storage shape 生成五字段 POD，设置 `blockDim=blockNum`、`tilingKey=0`，并固定声明 `workspaces[0] = 16 * 1024 * 1024` 字节系统 workspace。`scatter_min` entry 接收 `workspace`，但 body 不读取它；生成式 ACLNN executor 是否另加非连续转换 workspace 固定仓不可见，因此公开 `workspaceSize` 应以第一段实际返回值为准。workspace 由调用者在第一段之后分配，至少存活到第二段在指定 stream 上执行完成；executor 同样跨两段存活。

### 启动图与失败传播

```text
aclnnScatterMinGetWorkspaceSize
  -> 生成式 ACLNN executor（框架边界）
    -> ScatterMin InferShape/InferDataType
    -> Tiling4ScatterMin
      -> ScatterReduceCommonTiling
         生成 ScatterReduceSimtTilingData，key=0，blockDim=blockNum，workspace[0]=16 MiB

aclnnScatterMin(workspace, workspaceSize, executor, stream)
  -> K1 scatter_min<<<blockNum AIV cores, stream>>>
     input: var, indices, updates
     output: var 原地；形式参数 y 不使用；workspace 不使用
```

这是单 Kernel 图，没有 Kernel 间 workspace 边。生成 executor 的失败会阻止第二段获得有效 executor；Tiling 的 shape/platform/POD/workspace 空指针检查失败返回 `GRAPH_FAILED`，阻止 K1 启动；固定仓未展开 ACLNN 状态码到 graphStatus/launch 错误的框架映射。K1 在同一 stream 上提交；调用者若要在 Host 读取结果，必须等待该 stream 完成。

## Route 与参数设计

### 分核与资源计算

定义：

```text
F = var.shape[0]，标量 storage shape 时取 1
V = numel(var)
S = (F == 0) ? 0 : V / F                  # 每个 index 对应的连续元素数
M = numel(indices)
C = max(1, GetCoreNumAiv())
B0 = (M == 0) ? 1 : min(M, C)
P = (B0 == 0) ? 0 : ceil_div(M, B0)        # 每个前块的 index 条目数
B = (P == 0) ? 1 : ceil_div(M, P)          # 实际 blockDim
T = M - (B - 1) * P                        # 尾核 index 条目数
blockTilingSize = P * S                     # 前核 update 元素数
tailBlockTilingSize = T * S                 # 尾核 update 元素数
```

核 `b` 的 update 元素起点是 `b * blockTilingSize`；前核处理 `blockTilingSize`，最后核处理 `tailBlockTilingSize`。`P*S` 使每个 index 的整个 slice 落在同一核内。Device 不做 UB tile：512 个线程以 `threadIdx.x + k*512` 遍历当前核的全部 update 元素，因此没有 slice/tile UB 上限。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 32-bit 直接 CAS | key `0`；`ScatterReduceSimt<float/int32_t, int32_t/int64_t, int64_t, MODE_MIN>` | OpDef dtype 对中 `var/updates` 为 `FLOAT32` 或 `INT32`；body 编译期满足 `sizeof(PARAMS_T)==4` | 由 dtype specialization 进入 R-002/R-003；非法 dtype 在 ACLNN 校验失败 |
| 2 | R-002 FP16 word CAS | key `0`；`ScatterReduceSimt<half, int32_t/int64_t, int64_t, MODE_MIN>` | `var/updates` 为 `FLOAT16`；`sizeof(PARAMS_T)!=4` 且 `IsSameType<T, half>` | 非 FP16 进入 R-001/R-003 |
| 3 | R-003 INT8/UINT8 word CAS | key `0`；`ScatterReduceSimt<int8_t/uint8_t, int32_t/int64_t, int64_t, MODE_MIN>` | `var/updates` 为 `INT8` 或 `UINT8`；`sizeof(T)==1`，再由类型选择有/无符号解码 | 非 8-bit 进入 R-001/R-002 |

```text
平台不是 ascend950 -> 无公开实现/构建接线
平台是 ascend950 -> key = 0
  var dtype in {FLOAT32, INT32} -> R-001
  var dtype == FLOAT16          -> R-002
  var dtype in {INT8, UINT8}    -> R-003
  otherwise                     -> ACLNN 参数无效
indices dtype in {INT32, INT64} 只改变 INDICES_T，不改变算法拓扑
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 32-bit 直接 CAS | `0` + dtype specialization | 950；`PARAMS_T=float/int32_t`；`INDICES_T=int32_t/int64_t`；公开 shape 合法 | 单 K1 SIMT，`blockNum` AIV cores × 512 threads | `ScatterReduceSimt<PARAMS_T,INDICES_T,int64_t,MODE_MIN>::Process -> SimtComputeReduce -> AtomicReduce32` | GM direct；目标元素级 CAS；无 UB；16 MiB workspace 未消费 | 尾核用 `tailBlockTilingSize`；越界 index 跳过；空域无写 |
| R-002 FP16 word CAS | `0` + dtype specialization | 950；`PARAMS_T=half`；indices 合法 | 同上 | `... -> AtomicReduceSubWord<half,MODE_MIN>`，`ACC=float`，16-bit 子域 | 对包含目标 half 的 32-bit word CAS，`keepMask` 保留邻 half | 同上；cast 回 half 后提交 |
| R-003 INT8/UINT8 word CAS | `0` + dtype specialization | 950；`PARAMS_T=int8_t/uint8_t`；indices 合法 | 同上 | `... -> AtomicReduceSubWord<int8_t/uint8_t,MODE_MIN>`，`ACC=int32_t`，8-bit 子域 | 对 32-bit word CAS，保留另外 3 byte；有符号解码仅 INT8 | 同上；cast 回原 8-bit 类型 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 guard | `COMPUTE_UNIT=ascend950`、`TILING_DIR=arch35` | SIMT VF 与 CAS body 只为 950 接线 | 全部 | A2/A3 无正式 Route | `index/scatter_min/CMakeLists.txt`；`op_kernel/CMakeLists.txt` |
| 核数回退 | `coreNum==0 ? 1 : coreNum` | 防止除 0，并为 runtime 无 compileInfo 提供平台读取路径 | 全部 | 1 core | `ResolveCoreNum` |
| 空 index | `M==0 -> B=1,P=0,T=0` | 保持合法 launch，同时 thread 循环零次 | 全部 | K1 启动但无写 | `ScatterReduceCommonTiling` |
| 空 slice | `F==0 -> S=0` 或 Device `tiling.sliceSize==0` | 避免除 0和 GM 绑定/访问 | 全部 | `Init/Process` 直接返回 | `ScatterReduceCommonTiling`；`ScatterReduceSimt::Init/Process` |
| index 边界 | `idxVal < 0 \|\| idxVal >= varFirstDim` | 避免目标 GM 越界 | 全部 | 当前 update 元素跳过 | `SimtComputeReduce` |
| CAS 宽度 | `sizeof(PARAMS_T)==4` | 原生 CAS 仅覆盖 float/int32；窄类型需 word CAS | R-001 vs R-002/R-003 | 编译期另一分支 | `SimtComputeReduce` |
| 子字 mask | `sizeof(T)==1 ? 0xFF : 0xFFFF` | word CAS 必须仅替换目标 byte/half | R-002/R-003 | 编译期选择对应 mask | `AtomicReduceSubWord` |
| 512-thread 步长 | `THREAD_NUM=512`；`index += blockDim.x` | VF 内并行覆盖任意长度，不依赖 UB tile | 全部 | 无运行时 fallback | `THREAD_NUM`；`SimtComputeReduce` |

### POD/ABI

`ScatterReduceSimtTilingData` 是未打包的五个连续 `uint64_t` 字段；按源码类型的标准 C++ 布局，其 `sizeof=40`、`alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScatterReduceSimtTilingData` | `blockNum` | `uint64_t`/64 | offset 0 | AIV core 数 | `td->blockNum = blockNum` | `Init` 判断尾核；`Process` 检查 block 上界 | 空 index 时为 1 |
| 同上 | `blockTilingSize` | `uint64_t`/64 | offset 8 | update 元素 | `perCoreIndices * sliceSize` | `Init` 生成 `xBlockOffSet_` 与前核长度 | 乘法无显式溢出检查 |
| 同上 | `tailBlockTilingSize` | `uint64_t`/64 | offset 16 | update 元素 | `tailCoreIndices * sliceSize` | `Init` 生成最后核长度 | 空 index 时为 0 |
| 同上 | `sliceSize` | `uint64_t`/64 | offset 24 | 元素/index | `varTotal / varFirstDim`，首维 0 时为 0 | `Init/Process` 空域 guard；VF 计算 `m/sliceOff/dstIndex` | 0 时 Device 直接返回 |
| 同上 | `varFirstDim` | `uint64_t`/64 | offset 32 | index 个数 | `var.shape[0]`，0-D storage shape 取 1 | VF index 上界检查 | 公开 ACLNN rank 为 1–8 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 0 + 32-bit dtype | `scatter_min_tiling.cpp::Tiling4ScatterMin`；`scatter_reduce_common_tiling.cpp::ScatterReduceCommonTiling` | `ScatterReduceSimtTilingData` | `scatter_min.cpp::scatter_min`；模板 `float/int32_t` | `ScatterReduceSimt::Init/Process`；`SimtComputeReduce`；`AtomicReduce32` | `scatter_min/CMakeLists.txt`；`op_kernel/CMakeLists.txt` |
| R-002 | key 0 + FP16 | 同上 | 同上 | `scatter_min`；模板 `half` | `ScatterReduceSimt::Init/Process`；`SimtComputeReduce`；`AtomicReduceSubWord<half>` | 同上 |
| R-003 | key 0 + INT8/UINT8 | 同上 | 同上 | `scatter_min`；模板 `int8_t/uint8_t` | `ScatterReduceSimt::Init/Process`；`SimtComputeReduce`；`AtomicReduceSubWord<int8_t/uint8_t>` | 同上 |

## Kernel 实现

### 公共执行流程

`scatter_min` 将任务类型设为 `KERNEL_TYPE_AIV_ONLY`，解析 direct-struct POD，实例化 `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MIN>`。`Init` 绑定 indices、updates、var 三路 GM，并按 `GetBlockIdx()` 计算当前核的连续 update 元素域；`Process` 用 `asc_vf_call` 启动 512-thread SIMT VF。每个线程做 grid-stride loop，把 `globalIdx` 分解成 index 条目 `m` 和 slice 内偏移，再对目标 GM 做最小值 CAS。Kernel 不使用 `y`、`workspace`、`TPipe`、UB queue 或 dynamic UB。

### Route R-001：32-bit 直接 CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnScatterMin.md` | `aclnnScatterMinGetWorkspaceSize` | `varRef/updates` 支持 FLOAT32/INT32，indices 支持 INT32/INT64 | 生成 ScatterMin executor |
| OpDef | `op_host/scatter_min_def.cpp` | `ScatterMin::ScatterMin` | 成对 dtype 列表将同 dtype var/updates 与两种 indices 类型注册 | 编译期 `DTYPE_VAR/DTYPE_INDICES` specialization |
| Tiling | `scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 所有合法 dtype 统一 `SetTilingKey(0)` | key 0 |
| Router | `op_kernel/arch35/scatter_min.cpp` | `scatter_min` | `ScatterReduceSimt<DTYPE_VAR,DTYPE_INDICES,int64_t,MODE_MIN>` | `PARAMS_T` 为 float/int32_t |
| Device | `scatter_reduce_common_simt.h` | `SimtComputeReduce` | `if constexpr (sizeof(PARAMS_T) == 4)` | `AtomicReduce32` |

#### Device 数据流

##### 函数级调用链

```text
Tiling4ScatterMin -> ScatterReduceCommonTiling(key=0, blockDim=B, POD)
Host launch K1 scatter_min(var, indices, updates, y, workspace, tiling)
  -> ScatterReduceSimt<float/int32_t, int32_t/int64_t, int64_t, MODE_MIN>::Init
  -> ScatterReduceSimt::Process
    -> asc_vf_call<SimtComputeReduce>(dim3(512), GM pointers, blockBegin, blockElems, S, F)
      -> AtomicReduce32(target, update)
        -> ReduceAcc<dtype, MODE_MIN>
```

##### 源码等价伪码

```text
# 伪码；每个 AIV core b
blockBegin = POD.blockTilingSize * b
blockElems = (b == POD.blockNum - 1) ? POD.tailBlockTilingSize : POD.blockTilingSize
for local = threadIdx.x; local < blockElems; local += 512:
    global = blockBegin + local
    m = global / POD.sliceSize
    s = global - m * POD.sliceSize
    idx = indices[m]
    if idx < 0 or idx >= POD.varFirstDim: continue
    dst = idx * POD.sliceSize + s
    cur = var[dst]
    do:
        prev = cur
        wanted = (prev < updates[global]) ? prev : updates[global]
        cur = asc_atomic_cas(&var[dst], prev, wanted)
    while cur != prev
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `indices` | `m` | `M` 元素 | INT32/INT64，ND flatten | ACLNN caller | VF | caller；K1 期间只读 | 每个 update 元素重读所属 index |
| GM `updates` | `globalIdx` | `M*S` 元素 | FLOAT32/INT32，ND flatten | caller/连续化框架 | VF | executor/caller；K1 期间只读 | 直接 GM load，无 staging |
| GM `var` | `idx*S+s` | `V` 元素 | 同 updates | caller | VF CAS，亦是最终输出 | caller；跨 K1 持续有效 | 目标元素 32-bit CAS |
| 寄存器局部量 | thread-local | 固定仓未给出物理寄存器字节数 | 地址、index、cur/prev/value | VF | VF | 当前线程/循环迭代 | 下一迭代覆盖 |
| workspace[0] | 0 | 16 MiB | opaque system workspace | Host tiling/caller | Kernel 不消费 | caller/executor | 无 Kernel 内复用证据 |

##### 分层内存布局

N/A：该 SIMT Route 直接读取 `indices/updates` GM 并原子更新 `var` GM；没有 `InitBuffer`、TQue/TBuf、shared/dynamic UB。thread-local 状态位于寄存器，固定仓不公开其物理布局。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = N/A（asc_vf_call 未传 dynamic UB）
workspace_total = 16 * 1024 * 1024 bytes（Tiling 声明，K1 未读取）
GM logical bytes = M*sizeof(INDICES_T) + M*S*4 + V*4
=> Device 处理长度不受 UB tile 限制；uint64_t 地址乘法仍需不溢出
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `var[dst]` | 成功的 `asc_atomic_cas` | 后续冲突线程 CAS 或最终输出 | 下一次成功 CAS | CAS 返回旧值驱动 retry | 所有竞争写必须沿同一 CAS 协议 |
| thread-local `cur/prev` | 当前线程 | CAS retry 条件 | 下一轮 retry/下一元素 | 单线程程序顺序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 核内遍历 | 512-thread grid-stride loop | Tiling 的 `blockElems` | 每个 VF thread | 当前核连续 update 元素域 | `local < blockElems` 精确保护尾部 |
| 跨线程/跨核归约 | `asc_atomic_cas` retry | 任一冲突线程 | CAS 成功者/失败者重读 | 单个 32-bit `var` 元素 | FLOAT32 在 float、INT32 在 int32_t 中比较；无 cast |
| 尾核 | `blockIdx == blockNum-1` | Tiling | `Init` | 最后一段 update 域 | 使用 `tailBlockTilingSize` |
| 空域 | `sliceSize==0` 或 `blockIdx>=blockNum` | Host POD/entry | `Init/Process` return | 全部 GM | 无输出写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 生成 core/slice/前尾核公式、key 0、blockDim、16 MiB workspace |
| Kernel entry/router | `index/scatter_min/op_kernel/arch35/scatter_min.cpp` | `scatter_min` | AIV-only entry 实例化 `MODE_MIN` SIMT 模板 |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt::Init/Process`、`SimtComputeReduce`、`AtomicReduce32` | GM direct loop、index guard、32-bit CAS retry |

### Route R-002：FP16 32-bit word CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `docs/aclnnScatterMin.md`；`op_host/scatter_min_def.cpp` | ACLNN 原型；`ScatterMin` | var/updates 为 FLOAT16，indices 为 INT32/INT64 | `DTYPE_VAR=half` |
| Tiling | `scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 固定 key 0 | 共用 POD/launch |
| Router | `op_kernel/arch35/scatter_min.cpp` | `scatter_min` | `MODE_MIN`、`ADDR_T=int64_t` | 共享 SIMT body |
| Device | `scatter_reduce_common_simt.h` | `AtomicReduceSubWord` | `sizeof(T)==2`，`IsSameType<T,half>` | 16-bit submask、float ACC |

#### Device 数据流

##### 函数级调用链

```text
scatter_min
  -> ScatterReduceSimt<half, INDICES_T, int64_t, MODE_MIN>::Init/Process
    -> SimtComputeReduce
      -> AtomicReduceSubWord<half, MODE_MIN>
        -> unpack target half -> float
        -> ReduceAcc<float, MODE_MIN>
        -> cast float -> half -> merge into uint32 word -> CAS retry
```

##### 源码等价伪码

```text
# 伪码；core/thread/global/m/s/dst 计算与 R-001 相同
byteOff = dst * 2
wordAddr = reinterpret_cast<uint32*>(var) + (byteOff >> 2)
shift = (byteOff & 3) * 8                 # 0 或 16
keepMask = ~(0xFFFF << shift)
valAcc = float(updates[global])
do:
    prevWord = curWord
    curHalf = bitcast_half((prevWord >> shift) & 0xFFFF)
    newHalf = half(min(float(curHalf), valAcc))
    newWord = (prevWord & keepMask) | (bits(newHalf) << shift)
    curWord = asc_atomic_cas(wordAddr, prevWord, newWord)
while curWord != prevWord
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `indices` | `m` | `M*sizeof(INDICES_T)` | INT32/INT64 | caller | VF | caller | 直接 GM |
| GM `updates` | `globalIdx*2` | `2*M*S` bytes | FP16 flatten | caller/框架 | VF | caller/executor | 直接 GM |
| GM `var` 32-bit word | `(dst*2)>>2` | 逻辑 `2*V` bytes；CAS 访问 enclosing word | 两个 FP16/word | caller + CAS | VF/final output | caller | `keepMask` 保留邻 half；GM allocation 按源码注释至少 32-byte 对齐 |
| 寄存器局部量 | thread-local | 固定仓不可见 | float ACC、uint32 word、half bits | VF | VF | 当前线程 | retry 覆盖 |
| workspace[0] | 0 | 16 MiB | opaque | Host/caller | K1 不消费 | executor | N/A |

##### 分层内存布局

N/A：无 AIV UB/dynamic UB。实际原子介质是 `var` GM 中包含目标 FP16 的 32-bit word，word 内另一 FP16 由 `keepMask` 原样合并。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = N/A
workspace_total = 16 MiB（声明但不消费）
GM logical bytes = M*sizeof(INDICES_T) + 2*M*S + 2*V
CAS granularity = 4 bytes; semantic update granularity = 2 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 目标 32-bit GM word | `AtomicReduceSubWord` CAS | 冲突 byte/half 更新与最终输出 | 下次成功 word CAS | CAS 比较完整旧 word；失败即 retry | 每次从最新成功 word 重建 keepMask 合并值 |
| 同 word 邻 half | 其他线程或原值 | 当前 CAS 的 `prev & keepMask` | 仅其自身目标更新 | 完整 word CAS | 禁止用非原子 read-modify-write 替代 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 解码/计算 | half bitcast + float ACC | GM word、update half | `ReduceAcc` | 目标 half | half→float 比较→half cast；源码未指定额外 round mode |
| word 提交 | 32-bit CAS retry | 当前线程 | 冲突线程 | 两个 half 共用 word | `keepMask` 保留非目标 half |
| 核/线程尾部 | 同 R-001 | Host Tiling | VF | update 域 | 精确 `< blockElems`；尾核字段 |
| index/空域 | 同 R-001 | POD/GM index | VF guard | GM 安全 | 越界跳过；空 slice 返回 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 与 R-001 相同的元素域分核 |
| Kernel entry/router | `index/scatter_min/op_kernel/arch35/scatter_min.cpp` | `ScatterReduceSimt<half,...,MODE_MIN>` | dtype 宏实例化窄类型路线 |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `AtomicReduceSubWord<half,MODE_MIN>` | 16-bit 提取、float ACC、邻 half 保留、word CAS retry |

### Route R-003：INT8/UINT8 32-bit word CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `docs/aclnnScatterMin.md`；`op_host/scatter_min_def.cpp` | ACLNN 原型；`ScatterMin` | var/updates 为 INT8 或 UINT8，indices 为 INT32/INT64 | `DTYPE_VAR=int8_t/uint8_t` |
| Tiling | `scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 固定 key 0 | 共用 POD/launch |
| Router | `op_kernel/arch35/scatter_min.cpp` | `scatter_min` | `MODE_MIN` | 共享 SIMT body |
| Device | `scatter_reduce_common_simt.h` | `AtomicReduceSubWord` | `sizeof(T)==1`；INT8 命中 signed 分支，UINT8 命中 else | 8-bit submask、int32 ACC |

#### Device 数据流

##### 函数级调用链

```text
scatter_min
  -> ScatterReduceSimt<int8_t/uint8_t, INDICES_T, int64_t, MODE_MIN>::Init/Process
    -> SimtComputeReduce
      -> AtomicReduceSubWord<int8_t/uint8_t, MODE_MIN>
        -> unpack target byte（INT8 符号扩展或 UINT8 零扩展）
        -> ReduceAcc<int32_t, MODE_MIN>
        -> cast back -> merge byte into uint32 word -> CAS retry
```

##### 源码等价伪码

```text
# 伪码；core/thread/global/m/s/dst 同 R-001
byteOff = dst
wordAddr = reinterpret_cast<uint32*>(var) + (byteOff >> 2)
shift = (byteOff & 3) * 8
keepMask = ~(0xFF << shift)
updateAcc = int32(updates[global])
do:
    prevWord = curWord
    bits8 = (prevWord >> shift) & 0xFF
    curVal = signed_int8(bits8) if T is int8_t else uint8(bits8)
    newVal = T(min(int32(curVal), updateAcc))
    newWord = (prevWord & keepMask) | (uint8_bits(newVal) << shift)
    curWord = asc_atomic_cas(wordAddr, prevWord, newWord)
while curWord != prevWord
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `indices` | `m` | `M*sizeof(INDICES_T)` | INT32/INT64 | caller | VF | caller | 直接 GM |
| GM `updates` | `globalIdx` | `M*S` bytes | INT8/UINT8 flatten | caller/框架 | VF | caller/executor | 直接 GM |
| GM `var` word | `dst>>2` | 逻辑 `V` bytes；CAS 访问 enclosing word | 4 byte/word | caller + CAS | VF/final output | caller | `keepMask` 保留另 3 byte；GM 至少 32-byte 对齐 |
| 寄存器局部量 | thread-local | 固定仓不可见 | int32 ACC、uint32 word/bitmask | VF | VF | 当前线程 | retry 覆盖 |
| workspace[0] | 0 | 16 MiB | opaque | Host/caller | K1 不消费 | executor | N/A |

##### 分层内存布局

N/A：无 AIV UB/dynamic UB。实际共享原子单元是 `var` GM 的 32-bit word；每次提交只改变一个目标 byte。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = N/A
workspace_total = 16 MiB（声明但不消费）
GM logical bytes = M*sizeof(INDICES_T) + M*S + V
CAS granularity = 4 bytes; semantic update granularity = 1 byte
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 目标 32-bit GM word | word CAS | 冲突 byte 更新/最终输出 | 下次成功 CAS | 完整 word compare-and-swap | 每次 retry 基于返回的最新 word |
| 同 word 其余 3 byte | 原值或其他线程 | `prev & keepMask` | 各自目标更新 | CAS 冲突会失败重试 | signed/unsigned 仅影响目标 byte 解码，不得改变其他 byte |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 解码 | INT8 符号扩展 / UINT8 零扩展 | GM word | int32 ACC | 目标 byte | 保持输入有/无符号语义 |
| 计算/提交 | int32 min + cast + word CAS | 当前线程 | 冲突线程 | 4-byte word | min 结果仍在原 8-bit 有效域；cast 不损失有效值 |
| 核/线程尾部 | grid-stride + tail block | Host/POD | VF | update 域 | `< blockElems` 保护尾部 |
| index/空域 | guard/return | index/POD | VF | GM 安全 | 越界跳过；空 slice 无写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 分核和 tail 元素数 |
| Kernel entry/router | `index/scatter_min/op_kernel/arch35/scatter_min.cpp` | `ScatterReduceSimt<int8_t/uint8_t,...,MODE_MIN>` | 8-bit dtype specialization |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `AtomicReduceSubWord<int8_t/uint8_t,MODE_MIN>` | 8-bit mask、有符号分流、int32 ACC、word CAS retry |

## 算子特定优化

### Trick T-001：GM-direct 512-thread grid-stride 消除 UB tile 上限

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：所有 950 合法 dtype/shape；每个已启动 core 的 `blockElems` 可为任意非负长度
- Golden 来源：[`index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h:98-123`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L98-L123)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (ADDR_T index = static_cast<ADDR_T>(threadIdx.x); index < blockElems;
     index += static_cast<ADDR_T>(blockDim.x)) {
    ADDR_T globalIdx = blockBegin + index;
    ADDR_T m = globalIdx / sliceSize;
    ADDR_T sliceOff = globalIdx - m * sliceSize;
```

#### 机制与设计原因

Host 只按完整 slice 划分连续 update 元素域；Device 直接从 GM 读 indices/updates，512 个 VF thread 用 grid-stride loop 覆盖整个域。`blockElems` 来自 POD 的前/尾核元素数，VF 的 `blockDim.x` 来自 `asc_vf_call(dim3(THREAD_NUM))`，其中 `THREAD_NUM=512`。这省去 GM→UB staging、tile loop 和大 slice 的 UB 容量 guard，使单核处理长度不受局部 buffer 容量约束；代价是相同 `m` 的 `indices[m]` 会被 slice 内多个线程重复读取。

#### 迁移边界

- 适用：可由每个输出元素独立原子合并、且 GM direct load 可接受的 scatter-reduce。
- 前提：950 SIMT VF 可用；目标 dtype 有直接 CAS 或可安全封装到 32-bit word CAS；地址计算使用足够宽的 `int64_t`。
- 失效：目标更新需要跨元素协同、局部归约或不具备安全原子协议时，必须设计 UB/多阶段 Route；固定仓没有此 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：子字类型用完整 word CAS 保留邻元素

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：`sizeof(PARAMS_T)!=4`，即 FP16/INT8/UINT8
- Golden 来源：[`index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h:62-95`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L62-L95)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t newWord = (prev & keepMask) | ((newBits & SUBMASK) << shift);
cur = asc_atomic_cas(wordAddr, prev, newWord);
} while (cur != prev);
```

#### 机制与设计原因

950 CAS 没有 half/int8/uint8 overload。实现把目标元素映射到 4-byte 对齐的 enclosing word：`byteOff>>2` 选 word，`(byteOff&3)*8` 选子域，`SUBMASK` 来自元素宽度；每次以 CAS 返回的最新 word 重建 `keepMask` 合并值。这样即使不同线程同时更新同一 word 中不同子元素，失败者也会重试并保留成功者的邻元素修改。FP16 在 float 中比较再 cast 回 half；INT8/UINT8 在 int32 中比较再 cast 回原类型。

#### 迁移边界

- 适用：原子 API 不支持的 8/16-bit 标量，且目标 allocation 具备至少 4-byte word 可访问性。
- 前提：GM 基址满足 4-byte 对齐；所有竞争写使用完整 word CAS 协议；子域 mask/符号扩展与 dtype 完全匹配。
- 失效：未对齐或 allocation 尾部不能安全访问 enclosing word 时不能直接迁移；固定仓依赖 GM Tensor 至少 32-byte 对齐的前提。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能用普通“读 word→改 byte/half→写 word”替代 CAS；那会丢失同 word 邻元素的并发更新。源码以完整 word CAS retry 明确规避此问题。

## 模板与实现对比

| 对比维度 | R-001 32-bit | R-002 FP16 | R-003 INT8/UINT8 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | FLOAT32/INT32 | FLOAT16 | INT8/UINT8 | OpDef dtype specialization；共同 key 0 |
| 模板与 dtype | `PARAMS_T=float/int32_t` | `PARAMS_T=half` | `PARAMS_T=int8_t/uint8_t` | entry 的 `DTYPE_VAR` 宏 |
| 循环与分核 | 共同：按完整 slice 分核、512-thread grid-stride | 同左 | 同左 | 与 dtype 无关的 POD |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB；32-bit cur/prev | 无 UB；float ACC + word bits | 无 UB；int32 ACC + word bits | GM-direct SIMT |
| 内存总占用与复用 | 目标元素 CAS | 两个 half 共用 word，mask 保邻 half | 四个 byte 共用 word，mask 保邻 byte | CAS overload 宽度 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `asc_vf_call` + `asc_atomic_cas(T*)` | `asc_vf_call` + `asc_atomic_cas(uint32_t*)` | 同 R-002 | 无 Vector/NDDMA/通信节点 |
| 同步、精度与 tail | dtype 内直接比较；元素级 CAS | half→float→half；word CAS | byte→int32→byte；word CAS | 原子粒度与计算类型 |
| fallback/失败边界 | 非支持 dtype ACLNN 失败 | 无非 CAS fallback | 无非 CAS fallback | 固定仓只有一个 key/body 族 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持 | 公开 README/API 明确不支持 | Ascend 950PR/950DT 公开支持 |
| Host/Launcher | 无正式 Route | 无正式 Route | arch35 `ScatterReduceCommonTiling`，key 0，AIV core 分核 |
| Kernel/资源 | 无构建接线 | 无构建接线 | ascend950 单 SIMT Kernel；GM direct CAS；无 UB；16 MiB system workspace 声明 |

## 限制与失败边界

- 公开 dtype 仅为 var/updates 的 `FLOAT16/FLOAT32/INT32/INT8/UINT8` 与 indices 的 `INT32/INT64`；var 与 updates dtype 必须一致，shape 必须满足拼接关系。
- 公开 ACLNN rank 是 1–8。Host Tiling 对 storage shape 为 0-D 的分支属于防御实现，不扩张公开有效域。
- Kernel 将负 index 和 `>=varFirstDim` 的 index 静默跳过；公开文档没有单独给出越界 index 的错误语义，复用时必须保留 Kernel 事实。
- `useLocking` 不影响 950 Tiling/POD/Kernel；固定 body 始终通过 CAS 并发保护目标。生成式 ACLNN 壳如何承载该属性固定仓不可见。
- 多个重复 index 的 min 通过 CAS 串行化到成功写序列；源码没有给出 NaN、signed zero 等浮点特殊值的额外确定性约束。
- `indicesNum + blockNum - 1`、`perCoreIndices * sliceSize`、`idxVal * sliceSize + sliceOff` 使用 64-bit 整数但没有显式溢出检查；极大 shape 的上界固定仓不可见。
- 16 MiB workspace 由 Tiling 固定声明但 Device body 不使用；第一段接口可能还包含生成框架的非连续转换开销，调用者不得自行用 16 MiB 替代实际返回的 `workspaceSize`。
- A2/A3 公开明确不支持，因此未生成 `scatter_min.md`。本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter_min/docs/aclnnScatterMin.md` | `aclnnScatterMinGetWorkspaceSize`、`aclnnScatterMin` | 公开原型、平台、dtype/rank/shape、错误码、两段式生命周期 |
| 入口实现 | `index/scatter_min/op_host/scatter_min_def.cpp` | `ScatterMin`、`OP_ADD` | dtype/layout/属性、ascend950 AICore 注册；生成式 ACLNN 的算子定义 |
| 入口实现 | `index/scatter_min/op_host/scatter_min_infershape.cpp` | `InferShape4ScatterMin`、`InferDataType4ScatterMin` | 输出复制输入 var 的 shape/dtype |
| Host/Tiling/Launcher | `index/scatter_min/op_host/arch35/scatter_min_tiling.cpp` | `Tiling4ScatterMin`、`IMPL_OP_OPTILING` | ScatterMin 到共享 Tiling/parse 的接线 |
| Host/Tiling/Launcher | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ResolveCoreNum`、`ScatterReduceCommonTiling` | 分核公式、POD producer、key/blockDim/workspace |
| POD | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h` | `ScatterReduceSimtTilingData`、`MODE_MIN` | direct-struct ABI 与模式常量 |
| Kernel entry/body | `index/scatter_min/op_kernel/arch35/scatter_min.cpp` | `scatter_min` | AIV-only entry、POD parse、MODE_MIN 模板实例 |
| Kernel entry/body | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt`、`SimtComputeReduce`、`AtomicReduce32/SubWord` | SIMT loop、地址、原子、精度、tail 与空域 |
| 构建接线 | `index/scatter_min/CMakeLists.txt` | `add_modules_sources` | `ACLNNTYPE aclnn`、ascend950/arch35、共享依赖 |
| 构建接线 | `index/scatter_min/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 Kernel source 与 compile option |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 五个 uint64 字段可直接复用 | 字段顺序、元素单位、40-byte 布局；`sliceSize==0` 语义 | `ScatterReduceSimtTilingData` |
| 调用顺序 | 单 K1：Tiling 后在 executor stream 启动 | `Init` 必须先于 `Process`；POD 与 GM 指针在 K1 全程有效 | `scatter_min` |
| workspace/内存 | GM-direct，无 UB tile；workspace 固定声明 | 目标是输入 `var`，不是未使用的 `y`；不能假定 workspaceSize 仅 16 MiB | `ScatterReduceCommonTiling`；`ScatterReduceSimt::Init` |
| 同步 | 元素/word CAS retry 支持跨 thread/core 冲突 | 子字更新必须保留邻域并对完整 word CAS；失败必须重试 | `AtomicReduce32`；`AtomicReduceSubWord` |
| 精度与尾块 | FP16 用 float ACC，8-bit 用 int32 ACC；前尾核长度独立 | signed INT8 解码、half bitcast、`local<blockElems`、越界跳过 | `ReduceAcc`；`SimtComputeReduce` |
| 失败边界 | coreNum 0 回退 1，空 index/slice 无写 | 非 950、非法 dtype/shape 不得伪造 fallback；算术溢出无源码保护 | `ResolveCoreNum`；ACLNN 文档错误表 |

- [asc_atomic_cas](../../../../apis/simt_api/atomic_operations/asc_atomic_cas.md)
