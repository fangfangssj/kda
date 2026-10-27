---
type: Operator
title: "ScatterMax 950 ACLNN 算子知识"
description: "说明 aclnnScatterMax 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/docs/aclnnScatterMax.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/op_host/scatter_max_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/op_host/scatter_max_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/op_host/arch35/scatter_max_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/op_kernel/arch35/scatter_max.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_max/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:59:40Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ScatterMax 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnScatterMaxGetWorkspaceSize` + `aclnnScatterMax`；固定仓只公开这一版本族。 |
| 计算语义 | 对线性化后的每个索引项 `m` 与切片内位置 `j`，原地执行 `var[indices[m], j] = max(var[indices[m], j], updates[m, j])`；重复索引通过 CAS 原子归并。 |
| 输入输出 | `varRef/updates`：ND，`FLOAT16/FLOAT32/INT32/INT8/UINT8` 且 dtype 相同；`indices`：ND，`INT32/INT64`；三者公开秩为 1–8；输出与 `varRef` 同 shape、同 dtype、共享内存。 |
| 执行类型 | 单 Device 节点：`SIMT`（`KERNEL_TYPE_AIV_ONLY` entry 内通过 `asc_vf_call` 启动 512-thread VF）；无 Cube/AIC 计算节点、无 `MIX_CV`。 |
| Kernel 路线 | R-001：32 位元素直接 CAS（`FLOAT32/INT32`）；R-002：子字元素包围字 CAS（`FLOAT16/INT8/UINT8`）。两条 Route 的 TilingKey 均为 0。 |
| 核心资源 | `var/indices/updates` 直接 GM 访问；无 `TQue/TBuf`、无 AIV UB staging、无 dynamic UB；CAS 临时值为线程局部；Tiling 声明 16 MiB system workspace，但 Kernel 的 `workspace` 参数未读取。 |
| 来源闭包 | `source_closed`：公开 ACLNN 契约 → `ACLNNTYPE aclnn` OpDef/build 接线 → `Tiling4ScatterMax` → 公共 `ScatterReduceCommonTiling` 与 POD → `scatter_max` entry → `ScatterReduceSimt<..., int64_t, MODE_MAX>` → SIMT/CAS helper 已闭合；自动生成 ACLNN bridge 与非连续 Tensor 框架处理作为明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `V = varRef`，`I = indices`，`U = updates`，`S = product(V.shape[1:])`，`M = numel(I)`。公开约束是 `U.shape = I.shape + V.shape[1:]`。实现把 `U` 线性化为 `M*S` 个元素；对 `g ∈ [0, M*S)`：

```text
m = g / S
j = g - m * S
i = I[m]
若 0 <= i < V.shape[0]：V[i * S + j] = max(V[i * S + j], U[g])
否则：跳过该更新
```

输出参数 `var` 的推导 shape/dtype 均复制输入 `var`，但 Device entry 不写独立的 `y` 指针，而是直接以输入 `var` GM 地址为原子更新目标，和公开的原地语义一致。重复 `indices` 或并发命中同一元素时，CAS retry loop 把每次最大值更新串行化到该 GM 地址；固定源码没有给出重复索引的确定执行次序保证。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | ND；公开秩 1–8 | `FLOAT16/FLOAT32/INT32/INT8/UINT8` | 输入/输出原地目标；`shape[0]` 是索引上界，尾维乘积形成一条 slice；公开接口允许非连续 Tensor，具体连续化/回写由固定仓不可见的自动生成 ACLNN 框架层负责。 |
| `indices` | ND；公开秩 1–8，Device 按线性数组读取 | `INT32/INT64` | `numel(indices)=M`；Device 对负值或 `>=varFirstDim` 的值直接 `continue`，固定公开文档未为越界索引声明错误码。 |
| `updates` | `indices.shape + varRef.shape[1:]`；ND；公开秩 1–8 | 与 `varRef` 相同 | 线性元素 `g=m*S+j` 对应索引项 `m` 的切片内元素 `j`。 |
| `useLocking` | 标量属性 | `BOOL` | 默认 `false`。OpDef 声明该属性，但 operator-owned Tiling/POD/Kernel 均不读取它；两种取值在可见 Device 实现中都使用 CAS。自动生成 bridge 如何保存该属性固定仓不可见。 |
| `workspaceSize` | Host 输出 | `uint64_t` | 第一段接口返回；本实现 Tiling 的唯一 workspace slice 固定为 `16*1024*1024` 字节。 |
| `executor` | Host 输出/第二段输入 | `aclOpExecutor*` | 第一段生成并持有计算流程，第二段消费。其内部结构与 owner 实现固定仓不可见。 |
| `workspace` | Device 地址 | 字节缓冲区 | 调用者按 `workspaceSize` 分配并保持到第二段执行完成；`scatter_max` entry 接收该地址但不读取。 |
| `stream` | runtime stream | `aclrtStream` | 第二段在指定 stream 提交执行；生成 launcher 的精确代码固定仓不可见。 |

### 入口与版本

公开文档只给出 `aclnnScatterMaxGetWorkspaceSize` 和 `aclnnScatterMax` 两段式入口：第一段完成契约校验、计算 workspace 并产生 executor，第二段接收 workspace/executor/stream 执行。`index/scatter_max/CMakeLists.txt` 将 `scatter_max` 注册为 `ACLNNTYPE aclnn`，因此公开入口到 OpDef 的自动生成 bridge 属于框架边界；仓内没有手写 `op_api/*.cpp`，不能补造其内部函数名或非连续 Tensor 的具体节点图。

公开产品表明确 Ascend 950PR/950DT 支持，而 Atlas A2/A3 均不支持；OpDef 的唯一 `AICore().AddConfig`、Host tiling 目录与 Kernel build 也分别锁定 `ascend950/arch35`。因此本卡仅覆盖 950，且不创建 A2/A3 无后缀卡。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口按契约拒绝 `varRef/indices/updates` 空指针（`161001`），以及不支持 dtype、`varRef` 与 `updates` dtype 不同、`updates.shape` 不满足拼接约束（`161002`）。文档还声明三 Tensor 均支持非连续输入。固定仓未包含自动生成 ACLNN bridge，所以这些校验、连续化与原地回写的具体调用序列是框架边界；operator-owned 源码从 `ScatterMax` OpDef 的 dtype/layout/attr 注册、输出 shape/dtype 推导开始闭合。

Tiling 的真实顺序为：

1. `Tiling4ScatterMax` 无分支委托 `ScatterReduceCommonTiling`；`TilingParse` 注册 `TilingPrepareForScatterReduce`。
2. parse 阶段从 `PlatformAscendC::GetCoreNumAiv()` 写 `ScatterReduceCompileInfo::coreNum`。ACLNN runtime 若没有 compileInfo，`ResolveCoreNum` 重新从 platform 读取 AIV 核数；0 核信号归一为 1。
3. 从 storage shape 计算 `varFirstDim`、`varTotal`、`sliceSize`、`indicesNum`；随后分配 index 项到 AIV core，写 5 字段 POD、`blockDim`、key 0 与 16 MiB workspace。
4. 单个 `scatter_max` entry 解析同一 POD，固定实例化 `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MAX>`，绑定三路 GM 后启动 SIMT VF。

### 参数、Workspace 与生命周期

记平台 AIV 核数为 `C`、`M=numel(indices)`、`S=varTotal/varFirstDim`（`varFirstDim==0` 时 `S=0`）：

```text
B0 = (M == 0) ? 1 : min(M, C)
P  = (B0 == 0) ? 0 : ceil_div(M, B0)        # 每个前核的 index 项数
B  = (P == 0) ? 1 : ceil_div(M, P)          # 最终 blockNum
T  = M - (B - 1) * P                         # 尾核 index 项数
blockTilingSize     = P * S                   # 前核 update 元素数
tailBlockTilingSize = T * S                   # 尾核 update 元素数
```

所有中间量与 POD 字段均为 `uint64_t`；固定源码没有乘法/加法溢出检查。`context->SetBlockDim(B)` 生产 top-level AIV block 数。workspace 只有 slice 0，`offset=0`、`extent=16*1024*1024` 字节；由第一段 executor 告知调用者并由调用者持有到第二段结束。Kernel 形参仍包含 `workspace`，但 entry/body 不取其地址，故它不承载节点间数据。

### 启动图与失败传播

```text
aclnnScatterMaxGetWorkspaceSize
  -> [自动生成 ACLNN bridge：校验/非连续 Tensor 框架边界]
  -> ScatterMax OpDef / InferShape4ScatterMax / InferDataType4ScatterMax
  -> Tiling4ScatterMax
     -> ScatterReduceCommonTiling
        -> POD + blockDim=B + key=0 + workspace[0]=16 MiB

aclnnScatterMax(workspace, workspaceSize, executor, stream)
  -> [自动生成 launcher，固定仓不可见]
  -> node K0: scatter_max<<<B AIV blocks>>>(var, indices, updates, y, workspace, tiling)
     -> 每个 AIV block: asc_vf_call(..., dim3(512), ...)
     -> 原地更新 var GM
```

| 节点/边 | producer | consumer | 介质与范围 | ordering、owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| K0 | 自动生成 ACLNN launcher（精确符号不可见） | `scatter_max` | `var/indices/updates/y/workspace/tiling` GM 地址；top-level `blockDim=B` | 提交到调用者传入 stream；Tensor 与 workspace 由调用者保持到执行完成 | ACLNN 第二段返回 `aclnnStatus`；生成 launcher 如何映射 launch 错误固定仓不可见。 |
| K0 内 atomic edge | `SimtComputeReduce` 的线程 | 同地址后续 CAS retry | `var[dstIndex]` 或其 32 位包围字 | `asc_atomic_cas` 提供冲突更新顺序；无跨线程 barrier | CAS 未成功则重读返回值并重试；源码没有显式错误返回。 |
| workspace slice 0 | Tiling 声明 | Kernel 不消费 | offset 0，16 MiB，字节布局无 operator-owned 数据 | 调用者分配；生命周期覆盖第二段调用 | workspace 获取为空时 Tiling 失败；Kernel 不依赖其中内容。 |

Tiling 中 input shape、platform、POD、workspace-size 指针为空会由 `OP_CHECK_NULL_WITH_CONTEXT` 返回失败；`ResolveCoreNum` 失败也显式返回 `GRAPH_FAILED`，从而阻断 K0。单 Kernel 无后继节点，也没有 operator-owned event/pipe 交接边。

## Route 与参数设计

### 分核与资源计算

Host 按 index 项而不是按单元素先分核：前 `B-1` 核各覆盖 `P` 个完整 slice，即 `P*S` 个 update 元素；最后一核覆盖 `T*S` 个元素。对 top-level block `b`：

```text
blockBegin = b * (P*S)
blockElems = (b == B-1) ? T*S : P*S
```

该范围再由 512 个 SIMT thread 做 grid-stride 循环，线程 `t` 访问 `local=t, t+512, ... < blockElems`。地址统一以 `ADDR_T=int64_t` 计算。实现没有 UB tile、former/tail tile 或 repeat/mask 上限；tail 只表现为最后一核的 `tailBlockTilingSize` 和每线程循环条件。

空域有两个闭合分支：`M==0` 时 Host 仍设置 `B=1`、两个 block size 均为 0，VF 循环零次；`S==0` 时 `Init` 与 `Process` 直接返回。`blockIdx>=blockNum` 也在 `Process` 防御性返回。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 32 位直接 CAS | key 0；`ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MAX>`；`sizeof(DTYPE_VAR)==4` | 950；`DTYPE_VAR∈{float,int32_t}`，`DTYPE_INDICES∈{int32_t,int64_t}`；在 `SimtComputeReduce` 的 compile-time `if constexpr` 首分支命中 | `sizeof(DTYPE_VAR)!=4` 进入 R-002；不在 OpDef dtype 组合则 ACLNN 参数失败。 |
| 2 | R-002 子字包围字 CAS | key 0；同一主模板；`sizeof(DTYPE_VAR)!=4` | 950；公开/OpDef 可达组合限定 `DTYPE_VAR∈{half,int8_t,uint8_t}`，`DTYPE_INDICES∈{int32_t,int64_t}` | 不支持 dtype 在 ACLNN 前端失败；无第三个 Device fallback。 |

```text
公开 A5 + 支持的 var/updates dtype + indices dtype
  -> Tiling4ScatterMax -> key 0
  -> scatter_max<DTYPE_VAR, DTYPE_INDICES>
     -> sizeof(DTYPE_VAR) == 4 ? R-001 : R-002
```

`useLocking`、shape 大小、对齐和 index 值不改变 key/Route；shape 只改变 block/POD，越界 index 在 VF 内跳过。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 32 位直接 CAS | key 0；compile-time `sizeof(PARAMS_T)==4` | 950；`var/updates=float32或int32`；`indices=int32或int64`；公开 shape 契约成立 | 单 K0：AIV-only `scatter_max` → 512-thread `SimtComputeReduce` | `ScatterReduceSimt<DTYPE_VAR,DTYPE_INDICES,int64_t,MODE_MAX>` → `AtomicReduce32<PARAMS_T,MODE_MAX>` → `ReduceAcc<T,MODE_MAX>` | GM-direct；每个目标元素直接 `asc_atomic_cas(T*)`；无 UB/barrier | 尾核用 `tailBlockTilingSize`；线程循环 `<blockElems`；越界 index 跳过；无 Device fallback。 |
| R-002 子字包围字 CAS | key 0；compile-time else | 950；`var/updates=float16/int8/uint8`；`indices=int32或int64`；公开 shape 契约成立 | 单 K0，同上 | 主模板 → `AtomicReduceSubWord<PARAMS_T,MODE_MAX>` → `ReduceAcc<ACC,MODE_MAX>`；`ACC=float` for half，else `int32_t` | GM-direct；以 32 位 word CAS，`keepMask` 保留同字其他 lane；无 UB/barrier | 同 R-001；字内并发也依赖 CAS retry；无 Device fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950-only build | 公开支持表为 950；`SUPPORT_COMPUTE_UNIT=ascend950`、`SUPPORT_TILING_DIR=arch35` | 只有 arch35 Host/Kernel 接线 | R-001/R-002 | A2/A3 明确不发布 | `index/scatter_max/{README.md,CMakeLists.txt}` |
| AIV 核数归一 | compileInfo/platform `coreNum==0` 时取 1 | 防止分核除零，并保证空输入也有合法 blockDim | R-001/R-002 | 无 fallback；platform 指针为空返回失败 | `ResolveCoreNum` |
| 完整 slice 分核 | `P=ceil_div(M,B0)`，`blockTilingSize=P*S` | 让 top-level core 范围以完整索引 slice 为单位；重复目标仍由 atomic 处理 | R-001/R-002 | 无其它分核 Route | `ScatterReduceCommonTiling` |
| 32 位直接原子 | `sizeof(PARAMS_T)==4` | `asc_atomic_cas` 对 float/int32 有直接地址重载 | R-001 | 编译期转 R-002 | `SimtComputeReduce`、`AtomicReduce32` |
| 子字 4-byte 包围字 | `sizeof(PARAMS_T)∈{1,2}`；GM tensor 注释声明至少 32-byte 对齐 | 子字没有 CAS 重载，必须更新包含该元素的 32 位 word | R-002 | 非公开 dtype 在 ACLNN 失败 | `AtomicReduceSubWord` |
| index 有效域 | `idxVal>=0 && uint64(idxVal)<varFirstDim` | 防止目标 GM 越界 | R-001/R-002 | 当前 update 元素 `continue`，不是 Host 错误 | `SimtComputeReduce` |
| 空 slice | `sliceSize==0` | 避免 `globalIdx/sliceSize` 除零且无需写输出 | R-001/R-002 | `Init/Process` 返回 | `ScatterReduceSimt::Init/Process` |
| 线程尾保护 | `index<blockElems`，步长 `blockDim.x=512` | 覆盖任意 block element 数且不越过尾部 | R-001/R-002 | 循环结束 | `SimtComputeReduce`、`THREAD_NUM` |

### POD/ABI

`ScatterReduceSimtTilingData` 是 direct struct，由 5 个连续 `uint64_t` 组成。固定定义可确定 `sizeof=40`、`alignof=8`，字段 offsets 依次为 0/8/16/24/32；Host 通过 `GetTilingData<...>()` 写入，Device 用 `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` 按同一结构读取。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScatterReduceSimtTilingData` | `blockNum` | `uint64_t`/64 | struct size 40、align 8、offset 0 | AIV core/block 数 | `td->blockNum=B` | `Init` 判断尾核；`Process` 检查 `blockIdx_>=blockNum` | `M==0` 时为 1；正常时不超过可用 AIV 核数。 |
| `ScatterReduceSimtTilingData` | `blockTilingSize` | `uint64_t`/64 | offset 8 | update 元素/前核 | `td->blockTilingSize=P*S` | `Init` 计算 `xBlockOffSet_` 与非尾核 `currBlockTilingSize_` | 乘法无溢出检查；`M==0` 或 `S==0` 时为 0。 |
| `ScatterReduceSimtTilingData` | `tailBlockTilingSize` | `uint64_t`/64 | offset 16 | update 元素/尾核 | `td->tailBlockTilingSize=T*S` | `Init` 在 `blockIdx_==blockNum-1` 读取 | 乘法无溢出检查；空输入为 0。 |
| `ScatterReduceSimtTilingData` | `sliceSize` | `uint64_t`/64 | offset 24 | 元素/index slice | `td->sliceSize=(varFirstDim==0?0:varTotal/varFirstDim)` | `Init/Process` 空域 guard；VF 的除法、余数与目标地址 | 0 时整个 Kernel 无写；固定源码不检查 shape 乘积/除法一致性。 |
| `ScatterReduceSimtTilingData` | `varFirstDim` | `uint64_t`/64 | offset 32 | 元素（dim0 长度/index 上界） | `td->varFirstDim=(rank==0?1:varShape[0])` | VF 的 index 上界判断与 `dstIndex` | 公开 API rank 为 1–8；rank-0 分支仍存在于内部 Tiling。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 32 位直接 CAS | key 0；`sizeof(PARAMS_T)==4` | `index/scatter_max/op_host/arch35/scatter_max_tiling.cpp::Tiling4ScatterMax`；`index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp::{TilingPrepareForScatterReduce,ResolveCoreNum,ScatterReduceCommonTiling}` | `ScatterReduceSimtTilingData`，5×`uint64_t` | `index/scatter_max/op_kernel/arch35/scatter_max.cpp::scatter_max`，固定 `int64_t/MODE_MAX` | `ScatterReduceSimt::{Init,Process}` → `SimtComputeReduce` → `AtomicReduce32` → `ReduceAcc` | `index/scatter_max/CMakeLists.txt` 的 `ascend950/arch35/ACLNNTYPE aclnn`；`op_kernel/CMakeLists.txt` 的 ascend950 kernel source |
| R-002 子字包围字 CAS | key 0；compile-time else | 同 R-001 | 同 R-001 | 同 R-001 | `ScatterReduceSimt::{Init,Process}` → `SimtComputeReduce` → `AtomicReduceSubWord` → `ReduceAcc` | 同 R-001 |

## Kernel 实现

### 公共执行流程

两条 Route 共享一个单 Kernel launch graph。`scatter_max` entry 把任务声明为 `KERNEL_TYPE_AIV_ONLY`，解析 direct-struct tiling，实例化 `ADDR_T=int64_t`、`Mode=MODE_MAX` 的公共类；`y` 与 `workspace` 形参不参与后续调用。`Init` 在非空 slice 下取得 AIV `blockIdx`，计算本核元素区间并把 `indices/updates/var` 分别绑定到 `GlobalTensor`。`Process` 对有效 block 调用一次 `asc_vf_call`，线程块大小固定 512；VF 中每个线程以 512 为步长直接读 GM，计算目标地址并走 dtype 对应 CAS。

没有 CopyIn/CopyOut、Queue、UB tile、shared/dynamic UB 或 barrier。输出提交就是 CAS 对 `var` GM 的成功原子替换；同一地址的冲突由 CAS retry 串行化，不同地址不需要节点内同步。

### Route R-001：32 位元素直接 CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `index/scatter_max/docs/aclnnScatterMax.md`；`op_host/scatter_max_def.cpp` | 两段式 API；`ScatterMax::ScatterMax` | 950，`var/updates` 为 FLOAT32 或 INT32，`indices` 为 INT32/INT64 | 进入编译出的 dtype specialization；非法组合报参数错误。 |
| Tiling | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 无 dtype key 分流；恒 `SetTilingKey(0)` | key 0，POD + `blockDim=B`。 |
| Router/entry | `index/scatter_max/op_kernel/arch35/scatter_max.cpp` | `scatter_max` | `ScatterReduceSimt<DTYPE_VAR,DTYPE_INDICES,int64_t,MODE_MAX>` | 启动公共 SIMT body。 |
| Device | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `SimtComputeReduce` | `if constexpr(sizeof(PARAMS_T)==4)` | 调用 `AtomicReduce32`；否则 R-002。 |

#### Device 数据流

##### 函数级调用链

```text
Host K0 launch: scatter_max(var, indices, updates, y, workspace, tiling)
  -> GET_TILING_DATA_WITH_STRUCT(ScatterReduceSimtTilingData)
  -> ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MAX>::Init
     -> bind indices/updates/var GlobalTensor directly to GM
  -> ScatterReduceSimt<...>::Process
     -> asc_vf_call<SimtComputeReduce<DTYPE_INDICES,DTYPE_VAR,int64_t,MODE_MAX>>(dim3(512), ...)
        -> per thread grid-stride loop
        -> AtomicReduce32<DTYPE_VAR,MODE_MAX>
           -> ReduceAcc<DTYPE_VAR,MODE_MAX>
           -> asc_atomic_cas(var + dstIndex, previous, max(previous, update)) until success
```

##### 源码等价伪码

```text
# 伪码；所有地址索引使用 int64_t ADDR_T
b = GetBlockIdx()
if S == 0 or b >= B: return
begin = blockTilingSize * b
count = (b == B-1) ? tailBlockTilingSize : blockTilingSize

SIMT threads = 512
for local = threadIdx.x; local < count; local += 512:
    g = begin + local
    m = g / S
    j = g - m*S
    i = indices[m]
    if i < 0 or i >= varFirstDim: continue
    dst = i*S + j
    value = updates[g]
    current = var[dst]
    do:
        previous = current
        candidate = (previous > value) ? previous : value
        current = atomic_cas(&var[dst], previous, candidate)
    while current != previous
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `indices` | `m=(begin+local)/S` | `M` elements | INT32/INT64，线性 ND | 调用者/框架 | VF load `idxGmAddr[m]` | 调用者持有至 stream 完成 | 直接 GM；每个 update 元素都会读取其所属 index。 |
| GM `updates` | `g=begin+local` | `M*S` elements | FLOAT32/INT32，线性 ND | 调用者/框架 | `AtomicReduce32` 的 `val` | 调用者持有至 stream 完成 | 直接 GM，无 staging/复用。 |
| GM `varRef`/输出 | `dst=indices[m]*S+j` | `varTotal` elements | 与 updates 相同，ND，原地 | 初始 var + 成功 CAS | 后续 CAS/最终调用者 | 输入输出共享，stream 完成前不可回收 | 4-byte element direct CAS；同地址冲突 retry。 |
| 线程局部值 | `current/previous/candidate` | 每活动线程各若干 `PARAMS_T` | register placement 由编译器决定 | VF/CAS helper | 同线程 CAS loop | 单次元素更新 | 固定仓不能证明实际寄存器字节数。 |
| workspace slice 0 | byte offset 0 | `16*1024*1024` bytes | opaque system workspace | ACLNN caller | operator Kernel 不读取 | 第二段执行期间 | 无 operator-owned layout 或复用。 |

##### 分层内存布局

N/A：该 SIMT 节点不分配 AIV `TQue/TBuf/LocalTensor` 或 dynamic/shared UB。实际路径为 `indices GM + updates GM -> 线程局部标量/CAS -> var GM`。

##### 字节账与复用顺序

```text
UB_total = 0 bytes                         # 无 InitBuffer/等价 UB 分配
dynamic_ub = N/A                          # asc_vf_call 未传 dynamic UB
workspace_total = 16 * 1024 * 1024 bytes  # 单一 system slice，Kernel 不消费
thread-local registers = 固定仓不可见      # 仅可证明变量生命周期，不能给出字节数
=> 本 Route 不存在由 UB 容量产生的 tile 上界；每线程靠 512-step loop 覆盖任意本核 count
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `var[dst]` GM | 成功 `asc_atomic_cas` | 冲突线程下一轮 CAS/最终输出 | 下一次命中同一 dst 的 CAS | CAS 返回旧值；仅当返回值等于 expected 才退出 | 所有对同一 dst 的更新都必须保留 retry loop。 |
| 线程局部 `current/prev` | 当前线程 CAS loop | 当前线程下一次比较/候选计算 | 下一 retry 或下一元素 | SIMT 线程顺序语义 | 不得跨线程共享非原子状态。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分线程 | 512-thread grid-stride | Host/POD 的 `begin/count` | VF loop | 本核 `[begin,begin+count)` | `local<count` 精确保护尾部，无 padding lane。 |
| index guard | 条件 `continue` | `indices[m]` load | 地址计算/CAS | var GM 边界 | 负 index 或 `>=varFirstDim` 被静默跳过。 |
| max 计算 | `ReduceAcc<T,MODE_MAX>` | previous var、update | CAS candidate | 单一目标元素 | FLOAT32 保持 float，INT32 保持 int32；表达式为 `a>b?a:b`，固定公开契约未规定 NaN 行为。 |
| 冲突归并 | `asc_atomic_cas(T*)` retry | 当前线程 candidate | 同线程 loop condition | 4-byte 目标元素 | CAS 失败用返回的当前值重算，直到 `current==previous`。 |
| Kernel/stream | 无 operator-owned barrier/event | K0 | 调用者 stream 后继工作 | 整个原地输出 | 生成 runtime 的完成/错误细节固定仓不可见。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 计算 B/P/T/S，写 POD、blockDim、key 0、16 MiB workspace。 |
| Kernel entry/router | `index/scatter_max/op_kernel/arch35/scatter_max.cpp` | `scatter_max`；`ScatterReduceSimt<DTYPE_VAR,DTYPE_INDICES,int64_t,MODE_MAX>` | AIV-only entry，固定 MODE_MAX/64-bit 地址，单一公共 body。 |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt::{Init,Process}`、`SimtComputeReduce`、`AtomicReduce32`、`ReduceAcc` | GM-direct 512-thread 循环、地址公式、越界跳过、4-byte CAS retry。 |

### Route R-002：子字元素包围字 CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `index/scatter_max/docs/aclnnScatterMax.md`；`op_host/scatter_max_def.cpp` | 两段式 API；`ScatterMax::ScatterMax` | 950，`var/updates` 为 FLOAT16/INT8/UINT8，`indices` 为 INT32/INT64 | 进入编译出的 dtype specialization。 |
| Tiling | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 恒 key 0，dtype 不改变 POD | key 0，POD + `blockDim=B`。 |
| Router/entry | `index/scatter_max/op_kernel/arch35/scatter_max.cpp` | `scatter_max` | 同一 `ScatterReduceSimt<...,int64_t,MODE_MAX>` | 启动公共 SIMT body。 |
| Device | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `SimtComputeReduce` | `sizeof(PARAMS_T)!=4`；公开组合只剩 1/2-byte dtype | 调用 `AtomicReduceSubWord`。 |

#### Device 数据流

##### 函数级调用链

```text
Host K0 launch: scatter_max(var, indices, updates, y, workspace, tiling)
  -> ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MAX>::Init/Process
  -> asc_vf_call<SimtComputeReduce<...>>(dim3(512), ...)
     -> per thread grid-stride loop + index guard
     -> AtomicReduceSubWord<DTYPE_VAR,MODE_MAX>(uint32_t* varBase, dstIndex, updates[g])
        -> locate enclosing 32-bit word + shift + keepMask
        -> unpack selected half/int8/uint8 lane
        -> promote to float/int32 -> ReduceAcc<ACC,MODE_MAX> -> cast back
        -> merge selected bits with untouched neighbors
        -> asc_atomic_cas(wordAddr, previousWord, newWord) until success
```

##### 源码等价伪码

```text
# 伪码；分核、线程循环、m/j/index guard 与 R-001 相同
dstByte = dstIndex * sizeof(T)
wordAddr = uint32_ptr(var) + (dstByte >> 2)
shift = (dstByte & 3) * 8
subMask = (sizeof(T)==1) ? 0xff : 0xffff
keepMask = ~(subMask << shift)
valueAcc = (T==half) ? float(update) : int32(update)

currentWord = *wordAddr
do:
    previousWord = currentWord
    currentBits = (previousWord >> shift) & subMask
    currentT = unpack_T(currentBits)
    currentAcc = (T==half) ? float(currentT) : int32(currentT)
    newT = T((currentAcc > valueAcc) ? currentAcc : valueAcc)
    newBits = bits(newT) & subMask
    mergedWord = (previousWord & keepMask) | (newBits << shift)
    currentWord = atomic_cas(wordAddr, previousWord, mergedWord)
while currentWord != previousWord
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `indices` | `m=(begin+local)/S` | `M` elements | INT32/INT64，线性 ND | 调用者/框架 | VF | 调用者持有至 stream 完成 | 直接 GM。 |
| GM `updates` | `g=begin+local` | `M*S` elements | FLOAT16/INT8/UINT8 | 调用者/框架 | 子字 helper | 调用者持有至 stream 完成 | 直接 GM。 |
| GM `varRef` 包围字 | `word=(dstIndex*sizeof(T))>>2`；lane shift=`(byteOff&3)*8` | 4 bytes/CAS word；逻辑 var 为 `varTotal` elements | 32-bit physical CAS word，内部 lane 保持原 dtype | 初始 var + CAS | 后续 CAS/最终输出 | 输入输出共享 | 源码注释要求 base 4-byte 对齐并声明 GM Tensor 至少 32-byte 对齐；同字其他 lane 用 keepMask 保留。 |
| 线程局部标量 | `subBits/currentT/currentAcc/newT/newWord` | 每线程局部，实际寄存器字节不可见 | half 路线 ACC=float；int8/uint8 路线 ACC=int32 | helper | 同线程 CAS retry | 单个更新元素 | retry 时从最新 word 重新解包，不能复用旧 neighbor bits。 |
| workspace slice 0 | byte offset 0 | 16 MiB | opaque system workspace | ACLNN caller | Kernel 不读取 | 第二段执行期间 | 无 operator-owned layout。 |

##### 分层内存布局

N/A：该 SIMT 节点没有 AIV UB 或 dynamic/shared UB。真实子字布局是一个 GM 32-bit CAS word：`[保留 lane | 目标 lane | 保留 lane ...]`，目标 lane 由 `shift/subMask` 定位，其他位由 `keepMask` 从 CAS expected word 合并。

##### 字节账与复用顺序

```text
UB_total = 0 bytes
dynamic_ub = N/A
workspace_total = 16 * 1024 * 1024 bytes
CAS physical granularity = 4 bytes
logical lane bytes = sizeof(T) = 2 (half) or 1 (int8/uint8)
=> 无 UB tile 上界；字内不同逻辑元素仍共享同一 CAS 冲突域
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 32-bit `wordAddr` | `AtomicReduceSubWord` 成功 CAS | 同字任意 lane 的后续 CAS/最终输出 | 下一次命中该 word | CAS retry；失败后用返回的最新 word 重建 `keepMask` 合并结果 | 必须从本轮 `previousWord` 保留邻居位，不能对单 lane 做非原子 load/store。 |
| 线程局部 unpack/merge 值 | 当前 retry | 本轮 CAS | CAS 失败后全部重算 | do/while 顺序 | 邻居位只在 expected word 未变化时提交。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分线程/index guard | 同 R-001 | POD + GM index | 地址计算 | update 范围与 var 边界 | `local<count`；越界 index 跳过；无 padding。 |
| unpack/promote | 位移/mask + `Conditional` | previous word、update T | `ReduceAcc` | 目标子字 lane | half 当前值与 update 升为 float；int8/uint8 升为 int32。 |
| max/cast-back | `ReduceAcc<ACC,MODE_MAX>` + `static_cast<T>` | 两个 ACC | bit pack | 单 lane | max 后降回原 dtype；对这三种 dtype 的有限可表示输入，max 结果仍为原输入之一。固定源码未定义 half NaN 的额外规则。 |
| 字内归并 | `keepMask` | previous word + new lane bits | CAS | 目标 lane及邻居 lane | 非目标位逐位保留。 |
| 冲突同步 | `asc_atomic_cas(uint32_t*)` retry | merged word | loop condition | 整个 4-byte word | 同目标或相邻 lane 的竞争均重试；无需 barrier。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 与 dtype 无关的分核、POD、key 与 workspace。 |
| Kernel entry/router | `index/scatter_max/op_kernel/arch35/scatter_max.cpp` | `scatter_max`；公共主模板 | AIV-only entry；固定 MODE_MAX/ADDR_T=int64_t。 |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `SimtComputeReduce`、`AtomicReduceSubWord`、`ReduceAcc` | 1/2-byte dtype 的 word 定位、升精度、mask merge、CAS retry。 |

## 算子特定优化

### Trick T-001：GM-direct 512-thread grid-stride 消除 UB tile 上界

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：非空 slice；每个有效 AIV block 的 `Process`；与 dtype 无关。
- Golden 来源：[`index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h:163-174`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L163-L174)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Process()
{
    if (tiling_.sliceSize == 0 || blockIdx_ >= tiling_.blockNum) {
        return;
    }
    // One SIMT launch over this core's whole element range; the grid-stride loop and direct GM reads
    // handle any size without a UB tile, so there is no per-tile element cap.
    asc_vf_call<SimtComputeReduce<INDICES_T, PARAMS_T, ADDR_T, Mode>>(
        dim3(THREAD_NUM), (__gm__ INDICES_T*)idxGm_.GetPhyAddr(), (__gm__ PARAMS_T*)xGm_.GetPhyAddr(),
        (__gm__ PARAMS_T*)outputGm_.GetPhyAddr(), xBlockOffSet_, currBlockTilingSize_,
        static_cast<ADDR_T>(tiling_.sliceSize), static_cast<ADDR_T>(tiling_.varFirstDim));
}
```

#### 机制与设计原因

Host 只按 index 项把 `M*S` 元素范围划给 AIV core；Device 不为 tile 申请 `TQue/TBuf`，而是把 GM 指针、`begin/count/S/varFirstDim` 直接传给一次 VF launch。`dim3(THREAD_NUM)` 中 `THREAD_NUM=512`，VF 再以 `blockDim.x` 为步长迭代整个 `count`。这消除了 UB 容量、DMA 对齐和 tile tail 对单核元素数的约束，也避免了 updates 的 GM→UB→GM 中间物化；代价边界是每个 update 元素都直接读取 GM index/update，源码没有做 slice index 缓存。

#### 迁移边界

- 适用：输出更新可由线程安全 GM atomic 完成、每个元素地址可独立计算的 scatter-reduce。
- 前提：950 SIMT VF；512-thread launch；目标 dtype 有直接 CAS 或可安全映射到包围字 CAS；地址范围适合 `int64_t`。
- 失效：若计算必须依赖 UB tile 内复用、跨元素 barrier/reduction 或目标 dtype 无可证明的 atomic 映射，需要另一实现族；固定源码无此 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时若仍按 UB 容量裁剪 `count` 却不增加外层 tile 循环，会遗漏本核元素。

### Trick T-002：子字 lane 的包围字 CAS 与邻位保留

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：`PARAMS_T` 为 half/int8/uint8，即 `sizeof(T)` 为 2 或 1；`SimtComputeReduce` 的 compile-time else 分支。
- Golden 来源：[`index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h:63-95`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L63-L95)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T, uint8_t Mode>
__simt_callee__ inline void AtomicReduceSubWord(__gm__ uint32_t* base, uint64_t elemIdx, T val)
{
    using ACC = typename Conditional<AscendC::IsSameType<T, half>::value, float, int32_t>::type;
    constexpr uint32_t SUBMASK = (sizeof(T) == 1) ? 0xFFU : 0xFFFFU;
    uint64_t byteOff = elemIdx * sizeof(T);
    __gm__ uint32_t* wordAddr = base + (byteOff >> 2);
    uint32_t shift = (static_cast<uint32_t>(byteOff) & 3U) * 8U;
    uint32_t keepMask = ~(SUBMASK << shift);
    ACC valAcc = static_cast<ACC>(val);
    uint32_t cur = *wordAddr;
    uint32_t prev;
    do {
        prev = cur;
        uint32_t subBits = (prev >> shift) & SUBMASK;
        T curVal;
        if constexpr (AscendC::IsSameType<T, half>::value) {
            uint16_t hb = static_cast<uint16_t>(subBits);
            curVal = *reinterpret_cast<half*>(&hb);
        } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            curVal = static_cast<int8_t>(subBits);
        } else {
            curVal = static_cast<uint8_t>(subBits);
        }
        T newVal = static_cast<T>(ReduceAcc<ACC, Mode>(static_cast<ACC>(curVal), valAcc));
        uint32_t newBits;
        if constexpr (AscendC::IsSameType<T, half>::value) {
            newBits = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&newVal));
        } else {
            newBits = static_cast<uint32_t>(static_cast<uint8_t>(newVal));
        }
        uint32_t newWord = (prev & keepMask) | ((newBits & SUBMASK) << shift);
        cur = asc_atomic_cas(wordAddr, prev, newWord);
    } while (cur != prev);
}
```

#### 机制与设计原因

950 CAS 没有 half/int8/uint8 单元素重载，helper 因此把逻辑元素映射到 4-byte 对齐 word。`byteOff>>2` 选择 word，`(byteOff&3)*8` 选择 lane；`SUBMASK` 由 `sizeof(T)` 决定。每轮从最新 expected word 解包目标 lane，在 half→float 或 int8/uint8→int32 的 ACC 中求 max，再用 `keepMask` 把邻居位合回。若同字另一 lane 并发改变，CAS 失败并用返回的新 word 重算，从而避免非目标元素的 lost update。该机制用一个 32-bit atomic 覆盖子字 dtype，无需额外锁表或 workspace。

#### 迁移边界

- 适用：1-byte/2-byte 元素的逐元素原子归并，目标 GM base 满足 4-byte CAS 对齐且同字邻位必须保留。
- 前提：固定实现声明 GM tensor 至少 32-byte 对齐；元素字节序与位移/pack 规则一致；reduce 在所选 ACC 与 cast-back 下保持目标语义。
- 失效：元素跨越 32-bit word、base 不满足对齐、需要同时更新多个 lane 为一个不可分割事务，或 dtype 不是该 helper 覆盖的 half/int8/uint8 时不能复用；本算子没有 Device fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能以普通子字 load/store 替代 word CAS，也不能在 CAS 失败后沿用旧 `previousWord` 的邻居位；前者不能保护重复目标，后者会覆盖同字其他线程已经提交的 lane。该后果直接由源码的 `keepMask + retry` 数据依赖证明。

## 模板与实现对比

| 对比维度 | R-001 32 位直接 CAS | R-002 子字包围字 CAS | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `sizeof(PARAMS_T)==4`；FLOAT32/INT32 | compile-time else；FLOAT16/INT8/UINT8 | `SimtComputeReduce` 的 `if constexpr`；Host key 均为 0。 |
| 模板与 dtype | `AtomicReduce32<T,MODE_MAX>`，ACC=T | `AtomicReduceSubWord<T,MODE_MAX>`；half ACC=float，整数 ACC=int32 | CAS 可用粒度与 dtype 宽度。 |
| 循环与分核 | B 个 AIV core，每核一次 512-thread VF grid-stride | 相同 | 共同 POD/body 外壳。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB；线程局部 current/prev | 无 Queue/TBuf/dynamic UB；额外 unpack/mask/merge 临时值 | 两者 GM-direct。 |
| 内存总占用与复用 | UB 0；4-byte 逻辑元素即 CAS 粒度 | UB 0；多个逻辑 lane 共享 4-byte CAS word | 子字原子能力限制。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM load + `asc_atomic_cas(T*)` | GM load + `asc_atomic_cas(uint32_t*)` | 不涉及 Vector DMA、NDDMA 或通信。 |
| 同步、精度与 tail | 元素级 CAS；float/int32 原类型 max；尾循环 guard | word 级 CAS；升 ACC 后降回；keepMask 保邻位；同样尾 guard | CAS 重载与子字位操作差异。 |
| fallback/失败边界 | 非 4-byte 编译期进入 R-002 | 无第三 Route；不支持 dtype 前端失败 | 单 key、二分 body。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持 | 公开 README/API 明确不支持 | Ascend 950PR/950DT 公开支持 |
| Host/Launcher | 不生成正式卡 | 不生成正式卡 | `arch35` Tiling；AIV 核数分核；key 0；16 MiB system workspace |
| Kernel/资源 | 固定仓中的注释提到旧 TBE 语境不构成公开支持或活动接线证据 | 同左 | `ascend950` build；AIV-only entry + SIMT；GM-direct；两种 CAS body |

## 限制与失败边界

- 公开约束：三 Tensor 秩为 1–8；`varRef/updates` dtype 一致且只允许 FLOAT16/FLOAT32/INT32/INT8/UINT8；`indices` 只允许 INT32/INT64；`updates.shape = indices.shape + varRef.shape[1:]`。
- operator-owned Tiling 只取 `var` 与 `indices` storage shape，并不自行复核 updates shape/dtype；这些错误必须由 ACLNN/OpDef 框架在 launch 前阻断。生成校验代码固定仓不可见。
- Device 对负 index 或 `index>=varFirstDim` 静默跳过；公开 ACLNN 文档没有声明该输入的错误语义，不能推断为 Host 报错。
- `useLocking` 在可见 Tiling/POD/Kernel 中未消费；Device 始终使用 CAS。不能据此推断自动生成 bridge 会忽略属性，只能确认它不改变 operator-owned Device Route。
- `uint64_t` shape 乘积、`P*S/T*S` 和 `int64_t` Device 地址转换均无显式溢出 guard；可支持的最大张量还受 ACL/runtime 地址空间约束，固定仓没有给出额外上限。
- R-002 依赖 GM base 至少 4-byte 对齐；源码注释声明 GM tensor 至少 32-byte 对齐。不得把 helper 迁移到未满足该对齐契约的裸地址。
- FLOAT32/half 的 `ReduceAcc` 使用 `a>b?a:b`；公开文档未规定 NaN、signed zero 或重复索引的确定提交次序，固定源码不足以补齐这些数值边界。
- `workspace[0]=16 MiB` 是 Tiling 固定声明，Kernel 不读取 `workspace`；该 system workspace 的 runtime 内部用途固定仓不可见。
- 自动生成的 ACLNN bridge、非连续 Tensor 的 Contiguous/ViewCopy 细节、executor 内部 owner、launcher 的精确符号与 stream 错误映射固定仓不可见，作为框架边界；算子拥有的 Host/Tiling/POD/entry/body/build 已闭合。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter_max/docs/aclnnScatterMax.md` | `aclnnScatterMaxGetWorkspaceSize`、`aclnnScatterMax` | 两段式原型、公开 dtype/shape/非连续支持、错误码和平台表。 |
| 入口声明/约束 | `index/scatter_max/README.md` | `ScatterMax` 功能与产品表 | 原地 max 公式、参数范围、A2/A3 不支持与 950 支持。 |
| 入口实现 | `index/scatter_max/op_host/scatter_max_def.cpp` | `ops::ScatterMax`、`OP_ADD` | dtype/layout/attr、输出、ascend950 OpDef 配置与 `opFile=scatter_max`。 |
| 入口实现 | `index/scatter_max/op_host/scatter_max_infershape.cpp` | `InferShape4ScatterMax`、`InferDataType4ScatterMax` | 输出复制 var shape/dtype。 |
| Host/Tiling/Launcher | `index/scatter_max/op_host/arch35/scatter_max_tiling.cpp` | `Tiling4ScatterMax`、`IMPL_OP_OPTILING` | ScatterMax 到公共 Tiling/parse 的接线。 |
| Host/Tiling/Launcher | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `TilingPrepareForScatterReduce`、`ResolveCoreNum`、`ScatterReduceCommonTiling` | AIV 核数、分核公式、POD、key、blockDim、workspace 和失败路径。 |
| POD/ABI | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h` | `ScatterReduceSimtTilingData`、`MODE_MAX` | 5 字段 direct struct 与 max 模式常量。 |
| Kernel entry/body | `index/scatter_max/op_kernel/arch35/scatter_max.cpp` | `scatter_max` | AIV-only entry、POD 解析、模板实例、单 Kernel 调用链。 |
| Kernel entry/body | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt`、`SimtComputeReduce`、`AtomicReduce32`、`AtomicReduceSubWord`、`ReduceAcc` | 512-thread GM-direct 循环、地址、CAS、精度、tail 与空域。 |
| 构建接线 | `index/scatter_max/CMakeLists.txt` | `add_modules_sources` | `ascend950/arch35`、`ACLNNTYPE aclnn`、公共依赖。 |
| 构建接线 | `index/scatter_max/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch35 entry 仅为 ascend950 编译，带 `--cce-no-dcache-flush`。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 5×`uint64_t` direct struct 可复用到相同分核语义 | 字段顺序、40-byte size、8-byte alignment、元素单位、Host/Device 同结构必须保持 | `ScatterReduceSimtTilingData`、`GET_TILING_DATA_WITH_STRUCT` |
| 调用顺序 | Tiling → 单 `scatter_max` K0 → 每 AIV block 一次 VF | `MODE_MAX`、`ADDR_T=int64_t`、512 threads、先 Init 后 Process 不得改变 | `Tiling4ScatterMax`、`scatter_max`、`ScatterReduceSimt::Process` |
| workspace/内存 | GM-direct 可避免 UB tile；workspace 固定 16 MiB | Kernel 不得依赖未定义 workspace layout；var 必须为原地目标；子字 base 满足 4-byte 对齐 | `ScatterReduceCommonTiling`、`ScatterReduceSimt::Init`、`AtomicReduceSubWord` |
| 同步 | CAS retry 可处理重复 index 和跨核冲突 | 直接元素 CAS/包围字 CAS 的粒度、失败重算、keepMask 邻位保护不可删 | `AtomicReduce32`、`AtomicReduceSubWord` |
| 精度与尾块 | 32 位原类型 max；half→float、int8/uint8→int32 后 cast-back；grid-stride tail | `a>b?a:b` 运算顺序、ACC 选择、`local<count`、index guard、`sliceSize==0` guard 不得改变 | `ReduceAcc`、`SimtComputeReduce`、`ScatterReduceSimt::{Init,Process}` |
| 失败边界 | Host null/platform/POD/workspace 失败关闭；Device 越界 index 跳过 | 不得把越界跳过改写成已由 Host 报错；不支持 dtype/platform 无 Device fallback | `OP_CHECK_NULL_WITH_CONTEXT` sites、`SimtComputeReduce` |

- [asc_atomic_cas](../../../../apis/simt_api/atomic_operations/asc_atomic_cas.md)
