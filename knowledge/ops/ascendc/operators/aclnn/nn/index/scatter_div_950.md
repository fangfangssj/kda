---
type: Operator
title: "ScatterDiv 950 ACLNN 算子知识"
description: "说明 aclnnScatterDiv 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/docs/aclnnScatterDiv.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/op_host/scatter_div_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/op_host/scatter_div_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/op_host/arch35/scatter_div_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/op_kernel/arch35/scatter_div.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_div/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/opbuild.cmake
    role: implementation
created_at: "2026-08-25T16:57:15Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ScatterDiv 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnScatterDivGetWorkspaceSize` + `aclnnScatterDiv`，公开、原地 ACLNN；固定仓仅有无版本后缀入口 |
| 计算语义 | 对展平后的每个索引项 `m` 和 slice 内元素 `q`，原子执行 `var[indices[m],q] /= updates[m,q]`；非法索引跳过 |
| 输入输出 | `varRef/updates` 同 dtype：FP16/FP32/INT32/INT8/UINT8；`indices` INT32/INT64；均为 ND，公开接口允许非连续 Tensor |
| 执行类型 | 单个 AIV-only `scatter_div` entry，entry 内每个 AIV core 通过 `asc_vf_call` 启动 512-thread SIMT VF；无 Cube/AIC、无 MIX_CV |
| Kernel 路线 | R-001 FP32 直接 32-bit CAS；R-002 INT32 直接 32-bit CAS；R-003 FP16 enclosing-word CAS；R-004 INT8 enclosing-word CAS；R-005 UINT8 enclosing-word CAS |
| 核心资源 | `var/indices/updates` 直接访问 GM；线程寄存器；无 Queue/TBuf/dynamic UB；Tiling 声明 16 MiB 系统 workspace，但 Kernel 不读 `workspace` |
| 来源闭包 | source_closed：公开 ACLNN 声明→OpDef 生成边界→ScatterDiv Host/Tiling→共享 direct-struct POD→`scatter_div` entry→SIMT body/CAS helper→Ascend 950 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `V0=varRef.shape[0]`，`S=prod(varRef.shape[1:])`，`I=prod(indices.shape)`，并按 ND 存储顺序把 `indices` 展平。公开约束为 `updates.shape = indices.shape + varRef.shape[1:]`。对每个 `m in [0,I)`、`q in [0,S)`，若 `0 <= indices[m] < V0`，Kernel 以 CAS 重试把

```text
dst = indices[m] * S + q
varRef[dst] = varRef[dst] / updates[m*S + q]
```

原地提交；越界索引直接 `continue`。重复索引由同一 GM 目的元素上的 CAS 串行化，但哪个线程先成功由运行时并发决定；因此 FP 舍入和整数逐次截断的实际先后不由固定源码规定。输出 `var` 的 shape/dtype 复制输入 `var`，且公开接口说明输入输出共享内存。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | 1–8D、ND，可非连续 | FP16、FP32、INT32、INT8、UINT8 | 必填输入/输出，原地更新；输出 shape/dtype 与其一致 |
| `indices` | 1–8D、ND，可非连续 | INT32、INT64 | 必填；负数或 `>=varRef.shape[0]` 的值在 Device 端跳过 |
| `updates` | 1–8D、ND，可非连续 | 与 `varRef` 相同 | 必填；公开约束 `updates.shape=indices.shape+varRef.shape[1:]` |
| `useLocking` | Host 标量 | BOOL | 默认 `false`；OpDef 注册属性，但 Tiling/POD/Kernel 均不消费该字段；实际 Device 并发保护始终来自 CAS |
| `workspaceSize` | Host 输出标量 | UINT64 | 第一段接口返回；核心 Tiling 的 workspace slice 0 固定为 16 MiB |
| `executor` | Host 句柄 | — | 第一段接口生成并持有计算流程，第二段接口在指定 stream 运行 |
| `workspace` | Device 地址 | bytes | 由调用者按 `workspaceSize` 分配；当前核心 Kernel 参数保留但不读取 |

### 入口与版本

固定仓公开文档只声明无版本后缀的 `aclnnScatterDivGetWorkspaceSize/aclnnScatterDiv`，因此它就是 950 实际支持的最新公开版本。算子根 CMake 以 `ACLNNTYPE aclnn` 收集 `scatter_div_def.cpp`；仓库通用 `opbuild` 再从 OpDef 生成 ACLNN 源/头文件。生成产物不在固定 Commit 中，故参数连续化、executor 内部节点封装和第二段通用运行函数是明确的框架生成边界，不臆造具体符号。

公开 README 与 API 文档只对 Ascend 950PR/950DT 标记支持，对 Atlas A2、A3 明确标记不支持；OpDef 的 `AddConfig("ascend950")`、根 CMake 的 `SUPPORT_COMPUTE_UNIT "ascend950"` 与 Kernel CMake 的 `COMPUTE_UNITS ascend950` 同向。因此本卡只覆盖 950，不外推 A2/A3。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口契约依次覆盖空指针、三输入 dtype 支持范围、`varRef/updates` dtype 一致性与 updates shape 关系，失败分别返回 `ACLNN_ERR_PARAM_NULLPTR` 或 `ACLNN_ERR_PARAM_INVALID`。OpDef 把合法 dtype 组合显式注册为 `var/updates` 五种同类型与 `indices` 两种类型，输出复用 `var`；InferShape/InferDataType 直接复制输入 0 的 shape/dtype。

非连续 Tensor 的物化方式以及 `useLocking` 如何写入自动生成 executor 在固定仓不可见；这部分只作为 ACLNN 生成框架边界。可见的算子 Host 路径是 `Tiling4ScatterDiv -> ScatterReduceCommonTiling`，没有额外 selector、Cast、Sort 或多 Kernel 前后处理。

### 参数、Workspace 与生命周期

`TilingPrepareForScatterReduce` 从 `PlatformAscendC::GetCoreNumAiv()` 生产编译期 `coreNum`。运行 Tiling 优先读取 `context->GetCompileInfo()`；ACLNN runtime 不带 compileInfo 时重新从 platform 读取 AIV 核数，0 核回退为 1。Tiling 写入五个 `uint64_t` 字段、`blockDim`、恒定 key 0，并把 workspace slice 0 设为 `16*1024*1024` bytes。

executor、workspace 和 stream 的拥有者/释放时点遵循公开两段式 ACLNN：第一段返回 size/executor，调用者分配 workspace；第二段接收同一 executor 与 stream。单个核心 Kernel 没有 producer→consumer workspace 边，`workspace` 与独立输出参数 `y` 都未读取；原地结果直接落在 `varRef` GM。生成 executor 内可能存在的非连续 Tensor 框架节点及其 workspace slice/offset 在固定仓不可见。

### 启动图与失败传播

```text
aclnnScatterDivGetWorkspaceSize (generated ACLNN boundary)
  -> ScatterDiv OpDef / InferShape / InferDataType
  -> Tiling4ScatterDiv
     -> ScatterReduceCommonTiling
        -> blockDim=B, tilingKey=0, workspace[0]=16 MiB
  -> one AIV kernel: scatter_div(var,indices,updates,y,workspace,tiling)
       -> ScatterReduceSimt<T,Index,int64_t,MODE_DIV>::Init
       -> ScatterReduceSimt::Process
          -> asc_vf_call<SimtComputeReduce>(dim3(512), ...)
aclnnScatterDiv(workspace,size,executor,stream) (generated run boundary)
```

唯一 Device 节点的 Host `blockDim=B`；每个 AIV core 内 VF 的 `threadsPerBlock=512`，`dynUBufSize=N/A`，因为数据直接从 GM 读取且没有 local-memory 参数。节点输入为 `var/indices/updates/tiling`，输出仍是 `var`；`y/workspace` 为未使用 ABI 参数。Tiling 的空指针或平台解析失败返回 `GRAPH_FAILED`，阻断 launch；第二段运行失败通过 `aclnnStatus` 返回。单 Kernel 无跨节点 event/GM workspace 交接；不同线程/核对同一元素的顺序由 `asc_atomic_cas` 完成。

## Route 与参数设计

### 分核与资源计算

记 `C=max(platformAivCoreNum,1)`、`I=indices.GetShapeSize()`、`V0=(rank(var)==0?1:var.dim0)`、`N=var.GetShapeSize()`、`S=(V0==0?0:N/V0)`：

```text
B0 = (I==0 ? 1 : min(I,C))
P  = (B0==0 ? 0 : ceil_div(I,B0))            # 每个普通核的 index 项数
B  = (P==0 ? 1 : ceil_div(I,P))               # 实际 blockDim
Pt = I - (B-1)*P                              # 尾核 index 项数
blockTilingSize     = P*S                      # 元素/core
tailBlockTilingSize = Pt*S                     # 元素/尾 core
```

`blockIdx<B-1` 覆盖 `[blockIdx*P*S,(blockIdx+1)*P*S)`，尾核覆盖 `Pt*S` 元素，所以单个 index slice 不跨核。Device 每个线程从 `threadIdx.x` 起，以 `blockDim.x=512` 为步长遍历当前核元素域。Host 不按 UB 切 tile，且 `P*S`、`Pt*S` 与 shape 乘积均用 `uint64_t`，固定源码没有乘法溢出检查。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 direct CAS | key 0；`ScatterReduceSimt<float,INT32/INT64,int64_t,MODE_DIV>` | OpDef 实例 `var/updates=DT_FLOAT`；`sizeof(T)==4` | dtype 不同则对应下一编译实例 |
| 2 | R-002 INT32 direct CAS | key 0；`ScatterReduceSimt<int32_t,INT32/INT64,int64_t,MODE_DIV>` | OpDef 实例 `var/updates=DT_INT32`；`sizeof(T)==4` | dtype 不同则下一实例 |
| 3 | R-003 FP16 subword CAS | key 0；`ScatterReduceSimt<half,INT32/INT64,int64_t,MODE_DIV>` | OpDef 实例 `var/updates=DT_FLOAT16`；`sizeof(T)!=4 && IsSameType<T,half>` | dtype 不同则下一实例 |
| 4 | R-004 INT8 subword CAS | key 0；`ScatterReduceSimt<int8_t,INT32/INT64,int64_t,MODE_DIV>` | OpDef 实例 `var/updates=DT_INT8`；subword signed 分支 | dtype 不同则下一实例 |
| 5 | R-005 UINT8 subword CAS | key 0；`ScatterReduceSimt<uint8_t,INT32/INT64,int64_t,MODE_DIV>` | OpDef 实例 `var/updates=DT_UINT8`；subword unsigned 分支 | 其他 dtype 由 ACLNN/OpDef 拒绝 |

```text
ACLNN/OpDef chooses compiled dtype signature
Tiling always sets key=0
entry instantiates ScatterReduceSimt<DTYPE_VAR,DTYPE_INDICES,int64_t,MODE_DIV>
if sizeof(DTYPE_VAR)==4:
    AtomicReduce32                         # FP32 or INT32
else:
    AtomicReduceSubWord                   # FP16, INT8 or UINT8
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 / dtype compile instance | 950；FP32；indices INT32/INT64；公开 shape 约束 | 单 `scatter_div`；每 AIV core 一个 512-thread VF | `ReduceAcc<float,MODE_DIV>` + `AtomicReduce32<float>` | GM-direct；目标 FP32 CAS | grid-stride 尾；非法 index skip |
| R-002 INT32 | 同上 | 950；INT32；indices INT32/INT64 | 同上 | `ReduceAcc<int32_t,MODE_DIV>` + `AtomicReduce32<int32_t>` | GM-direct；目标 INT32 CAS | 同上；除零/`INT_MIN/-1` 未 guard |
| R-003 FP16 | 同上 | 950；FP16；indices INT32/INT64 | 同上 | `AtomicReduceSubWord<half>`，ACC=float | 32-bit enclosing-word CAS；保留相邻 half | 同上；每次更新降回 FP16 |
| R-004 INT8 | 同上 | 950；INT8；indices INT32/INT64 | 同上 | subword signed decode，ACC=int32 | 32-bit enclosing-word CAS；保留其余三 byte | 同上；每次更新窄化回 INT8 |
| R-005 UINT8 | 同上 | 950；UINT8；indices INT32/INT64 | 同上 | subword unsigned decode，ACC=int32 | 32-bit enclosing-word CAS；保留其余三 byte | 同上；每次更新窄化回 UINT8 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 有效 AIV 核数 | compileInfo/platform `coreNum==0` 时置 1 | 避免 `blockDim=0` 和除零 | 全 Route | 单核 |
| 整 slice 分核 | `blockTilingSize=ceil_div(I,B)*S` | `m=globalIdx/S` 要保持每核起点为 slice 边界 | 全 Route | 无其他 Route |
| 空 indices | `I==0 -> B=1,P=0,Pt=0` | 保持合法 launch，VF 循环零次 | 全 Route | no-op |
| 空 slice | `S==0` | 避免 Device 的 `globalIdx/S` 除零 | 全 Route | `Init/Process` 直接返回 |
| index 边界 | `idxVal<0 \|\| idxVal>=V0` | 防止 var GM 越界 | 全 Route | 当前元素 `continue` |
| CAS 位宽 | `sizeof(T)==4` | primitive 对 FP32/INT32 有直接 overload，subword 无 overload | R-001/R-002 | R-003～R-005 enclosing-word helper |
| GM 对齐 | subword tensor base 至少 4B 对齐；源码注释给出 GM tensor 至少 32B 对齐 | 以 `uint32_t*` 访问 enclosing word | R-003～R-005 | 固定源码无 fallback |
| 线程数 | `THREAD_NUM=512` 与 `LAUNCH_BOUND(512)` | VF grid-stride 的 `blockDim.x` 固定 | 全 Route | 无动态线程 fallback |

### POD/ABI

`ScatterReduceSimtTilingData` 是 direct struct：五个连续 `uint64_t`，声明顺序下自然 ABI 为 `sizeof=40`、`alignof=8`、offset 依次 0/8/16/24/32。固定仓没有 `static_assert(sizeof/offsetof)`；工程复用必须共享该头文件和编译 ABI，不能另行重排。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScatterReduceSimtTilingData` | `blockNum` | `uint64_t`/64 | offset 0；struct 40B/align 8（自然布局） | AIV core/block | `ScatterReduceCommonTiling`: `B` | `Init`: 判断尾核；`Process`: active core guard | 至少 1 |
| 同上 | `blockTilingSize` | `uint64_t`/64 | offset 8 | update 元素/core | `P*S` | `Init`: `xBlockOffSet_` 与普通核元素数 | 乘法无显式溢出检查 |
| 同上 | `tailBlockTilingSize` | `uint64_t`/64 | offset 16 | update 元素 | `Pt*S` | `Init`: 最后一核元素数 | 空 indices 时 0 |
| 同上 | `sliceSize` | `uint64_t`/64 | offset 24 | 元素/index slice | `N/V0`，`V0==0` 时 0 | `Init/Process` 空域 guard；VF 计算 `m/sliceOff/dst` | 0 时 no-op |
| 同上 | `varFirstDim` | `uint64_t`/64 | offset 32 | index 上界 | `var.dim0`，rank 0 时 1 | VF index 合法性判断 | 仅合法 index 写入 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | key 0 + FP32 OpDef instance | `Tiling4ScatterDiv -> ScatterReduceCommonTiling` | `ScatterReduceSimtTilingData` | `scatter_div`: `T=float` | `SimtComputeReduce -> AtomicReduce32<float> -> ReduceAcc<float,MODE_DIV>` | ScatterDiv/common CMake，ascend950 |
| R-002 INT32 | key 0 + INT32 instance | 同上 | 同上 | `scatter_div`: `T=int32_t` | `AtomicReduce32<int32_t>` | 同上 |
| R-003 FP16 | key 0 + FP16 instance | 同上 | 同上 | `scatter_div`: `T=half` | `AtomicReduceSubWord<half>` + FP32 ACC | 同上 |
| R-004 INT8 | key 0 + INT8 instance | 同上 | 同上 | `scatter_div`: `T=int8_t` | `AtomicReduceSubWord<int8_t>` + INT32 ACC | 同上 |
| R-005 UINT8 | key 0 + UINT8 instance | 同上 | 同上 | `scatter_div`: `T=uint8_t` | `AtomicReduceSubWord<uint8_t>` + INT32 ACC | 同上 |

## Kernel 实现

### 公共执行流程

`scatter_div` 声明为 `__aicore__` 且标记 `KERNEL_TYPE_AIV_ONLY`；读取 direct-struct Tiling 后构造 `ScatterReduceSimt<DTYPE_VAR,DTYPE_INDICES,int64_t,MODE_DIV>`。`TPipe` 只满足统一构造 ABI，类内明确 `(void)pipe`，没有 `InitBuffer`。`Init` 绑定三个 GM tensor，计算当前核的 update 元素起点和长度；`Process` 仅对 `S>0 && blockIdx<B` 发起一次 512-thread `asc_vf_call`。

VF 中每个线程以 512 为步长访问本核的线性 update 元素。每次从 GM 读一次 `indices[m]` 与 `updates[globalIdx]`，计算目标地址并走直接或 enclosing-word CAS。`var` 是唯一输出；`y`、workspace、UB、Queue、TBuf、barrier 均不参与。CAS 只同步同一目的 word 的并发更新，不建立不同目的元素间的顺序。

### Route R-001：FP32 直接 CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnScatterDiv.md` | `aclnnScatterDivGetWorkspaceSize` | varRef/updates FP32，indices INT32/INT64 | 公开签名合法 |
| OpDef | `op_host/scatter_div_def.cpp` | `ScatterDiv::ScatterDiv` | `DT_FLOAT` 配对实例 | 生成 FP32 ACLNN/binary instance |
| Tiling | `scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 无 dtype selector；恒 `SetTilingKey(0)` | key 0 |
| Router | `op_kernel/arch35/scatter_div.cpp` | `scatter_div` | `DTYPE_VAR=float`，`MODE_DIV` | `ScatterReduceSimt<float,...>` |
| Body | `scatter_reduce_common_simt.h` | `SimtComputeReduce` | `sizeof(PARAMS_T)==4` | `AtomicReduce32<float>` |

#### Device 数据流

##### 函数级调用链

```text
scatter_div<float,Index>(var,indices,updates,...,tiling)
  -> ScatterReduceSimt<float,Index,int64_t,MODE_DIV>::Init
  -> ScatterReduceSimt::Process
    -> asc_vf_call<SimtComputeReduce>(512 threads)
      -> AtomicReduce32<float,MODE_DIV>
        -> ReduceAcc<float,MODE_DIV>(old,update)
        -> asc_atomic_cas(var+dst,old,old/update) until success
```

##### 源码等价伪码

```text
# 伪码；每 AIV core 内 512 threads
begin = blockIdx * blockTilingSize
elems = (blockIdx==blockNum-1 ? tailBlockTilingSize : blockTilingSize)
for local = threadIdx.x; local < elems; local += 512:
  g = begin + local; m = g / S; q = g - m*S; idx = indices[m]
  if idx<0 || idx>=V0: continue
  addr = var + idx*S + q
  do: old=cur; cur=atomicCAS(addr,old,float(old/updates[g])) while cur!=old
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices GM | `m` | `I` elements | INT32/INT64 flat ND | ACLNN input | VF bounds/address | caller/executor through Kernel | direct scalar load |
| updates GM | `g` | `I*S` elements | FP32 flat ND | ACLNN input | `ReduceAcc` | caller/executor through Kernel | direct scalar load |
| var GM | `idx*S+q` | `N` elements | FP32 flat ND | input + CAS | output varRef | caller owns in-place tensor | 32-bit atomic word |
| thread registers | per thread | 固定仓不可见字节数 | counters/FP32 temporaries | VF | CAS loop | thread invocation | no shared alias |
| workspace slice 0 | framework-chosen base | 16 MiB | bytes | Host Tiling | Kernel 不读 | caller/executor | 不参与计算 |

##### 分层内存布局

N/A：该 SIMT 节点直接读写 GM；`TPipe` 未使用，没有 AIV UB、Queue、TBuf 或 dynamic UB。实际数据层级是 `indices/updates GM -> thread registers -> atomic CAS -> var GM`。

##### 字节账与复用顺序

```text
UB_total = 0 B (no InitBuffer/Queue/TBuf)
dynamic_ub = N/A (asc_vf_call has no dynUB argument)
workspace_total = 16*1024*1024 B system slice; Kernel consumption = 0 B
register_bytes = fixed source invisible
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| var GM element | successful CAS thread | subsequent CAS/output reader | next successful duplicate-index CAS | CAS compare-and-retry | 所有竞争写必须保留 retry loop |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| VF traversal | grid-stride loop | Tiling core range | current thread | `[0,elems)` | `local<elems` 保护尾部 |
| bounds | signed compare + uint64 upper compare | indices load | address calculation | var allocation | 非法 index skip |
| divide/update | `asc_atomic_cas(float*)` retry | competing threads | successful CAS | one FP32 element | FP32 division；重复项顺序未规定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 分核、POD、key 0、16 MiB workspace |
| Kernel entry/router | `index/scatter_div/op_kernel/arch35/scatter_div.cpp` | `scatter_div` | AIV-only，绑定 `MODE_DIV` 与 `int64_t` 地址类型 |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `SimtComputeReduce<float>`, `AtomicReduce32<float>` | 512-thread GM-direct FP32 CAS |

### Route R-002：INT32 直接 CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnScatterDiv.md` | 第一段接口 | varRef/updates INT32 | 公开签名合法 |
| OpDef | `op_host/scatter_div_def.cpp` | `ScatterDiv::ScatterDiv` | `DT_INT32` 配对实例 | INT32 instance |
| Tiling | `scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | 恒 key 0 | key 0 |
| Router | `op_kernel/arch35/scatter_div.cpp` | `scatter_div` | `DTYPE_VAR=int32_t` | `ScatterReduceSimt<int32_t,...>` |
| Body | `scatter_reduce_common_simt.h` | `SimtComputeReduce` | `sizeof(T)==4` | `AtomicReduce32<int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
scatter_div<int32_t,Index> -> Init -> Process
  -> asc_vf_call<SimtComputeReduce>(512)
    -> AtomicReduce32<int32_t,MODE_DIV>
      -> ReduceAcc<int32_t,MODE_DIV> -> asc_atomic_cas retry
```

##### 源码等价伪码

```text
# 分核/地址与 R-001 相同
for g in thisCoreRange by 512-thread grid stride:
  derive m,q,idx; if invalid idx: continue
  addr=&var[idx*S+q]; val=updates[g]
  CAS retry addr from old to int32(old/val)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices GM | `m` | `I` | INT32/INT64 | input | VF | caller | direct load |
| updates GM | `g` | `I*S` | INT32 | input | integer divide | caller | direct load |
| var GM | `idx*S+q` | `N` | INT32 | input/CAS | output | caller | 32-bit atomic word |
| registers | per thread | 固定仓不可见 | int64 address + int32 CAS temps | VF | VF | thread | retry-local |
| workspace | logical slice 0 | 16 MiB | bytes | Tiling | unused | caller/executor | no reuse |

##### 分层内存布局

N/A：GM-direct SIMT，未分配 AIV UB/dynamic UB。路径为 `GM -> registers -> INT32 CAS -> GM`。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A
workspace_total=16 MiB; Kernel consumption=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| var INT32 element | CAS winner | next contender/output | next CAS | compare-and-retry | updates divisor 必须满足 Device 除法有效域；源码无 guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| traversal | grid-stride | Host range | thread | tail elements | `local<elems` |
| update | INT32 CAS | contenders | CAS winner | one INT32 element | INT32 逐次除法；除零及 `INT_MIN/-1` 未处理 |
| duplicate index | retry loop | concurrent threads | later retry | same dst | 成功顺序未规定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | whole-slice core range |
| Kernel entry/router | `index/scatter_div/op_kernel/arch35/scatter_div.cpp` | `ScatterReduceSimt<DTYPE_VAR,...,MODE_DIV>` | INT32 编译实例 |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `AtomicReduce32<int32_t>` | INT32 divide + CAS retry |

### Route R-003：FP16 enclosing-word CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnScatterDiv.md` | 第一段接口 | varRef/updates FP16 | 公开签名合法 |
| OpDef | `op_host/scatter_div_def.cpp` | `ScatterDiv::ScatterDiv` | `DT_FLOAT16` | FP16 instance |
| Tiling | common Tiling | `ScatterReduceCommonTiling` | key 0 | key 0 |
| Router | `scatter_div.cpp` | `scatter_div` | `DTYPE_VAR=half` | common SIMT template |
| Body | common SIMT header | `AtomicReduceSubWord` | `sizeof(T)!=4 && IsSameType<T,half>` | FP32 ACC + 16-bit field CAS |

#### Device 数据流

##### 函数级调用链

```text
scatter_div<half,Index> -> Init -> Process -> SimtComputeReduce
  -> AtomicReduceSubWord<half,MODE_DIV>
    -> locate enclosing uint32 word / decode half bits
    -> ReduceAcc<float,MODE_DIV> / cast back half
    -> merge neighbor bits / asc_atomic_cas(uint32_t*) retry
```

##### 源码等价伪码

```text
byteOff=dst*sizeof(half); word=&reinterpret<uint32*>(var)[byteOff>>2]
shift=(byteOff&3)*8; keepMask=~(0xffff<<shift)
retry:
  oldWord=*word; oldHalf=bits_to_half((oldWord>>shift)&0xffff)
  newHalf=half(float(oldHalf)/float(update))
  newWord=(oldWord&keepMask)|(half_bits(newHalf)<<shift)
until atomicCAS(word,oldWord,newWord) succeeds
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices/updates GM | `m` / `g` | `I` / `I*S` | Index / FP16 | input | VF | caller | direct load |
| var enclosing word | `(dst*2)>>2` words, shift 0或16 | 4B transaction | 2×FP16 packed in uint32 | input/CAS | output | caller | base 至少 4B 对齐 |
| registers | per thread | 固定仓不可见 | uint32 masks/words + FP32 ACC | helper | helper | thread | no shared buffer |
| workspace | logical slice 0 | 16 MiB | bytes | Tiling | unused | caller | — |

##### 分层内存布局

N/A：无 UB。每次原子事务覆盖一个 32-bit GM word，但 `keepMask` 保留未被更新的相邻 FP16 字段。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; workspace_total=16 MiB (unused by Kernel)
atomic transaction=4 B; logical updated field=2 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| enclosing uint32 word | FP16 CAS winner | next half-field contender/output | next CAS | full-word compare-and-retry | merge 必须基于本轮 `prev`，不可复用陈旧 neighbor bits |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| decode/compute | half→float→divide→half | old word/update | CAS merge | selected 16 bits | 每次成功更新后降回 FP16 |
| neighbor preservation | keepMask | old word | new word | other half in word | 竞争修改导致 CAS retry |
| tail/index | loop/bounds guards | Tiling/indices | VF | final elements | 空 slice return，非法 index skip |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | common Tiling cpp | `ScatterReduceCommonTiling` | dtype-independent range/POD |
| Kernel entry/router | ScatterDiv Kernel cpp | `scatter_div<half>` | MODE_DIV route |
| Device body/helper | common SIMT header | `AtomicReduceSubWord<half>` | FP32 internal divide、half cast、word merge |

### Route R-004：INT8 enclosing-word CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + `scatter_div_def.cpp` | ACLNN / `ScatterDiv` | `DT_INT8` 配对 | INT8 instance |
| Tiling | common Tiling | `SetTilingKey(0)` | unconditional | key 0 |
| Router | `scatter_div.cpp` | `ScatterReduceSimt` | `T=int8_t` | subword route |
| Body | common SIMT header | `AtomicReduceSubWord` | `sizeof(T)==1 && IsSameType<T,int8_t>` | signed decode + INT32 ACC |

#### Device 数据流

##### 函数级调用链

```text
scatter_div<int8_t,Index> -> SimtComputeReduce
  -> AtomicReduceSubWord<int8_t,MODE_DIV>
    -> signed int8 decode -> INT32 divide -> int8 narrow
    -> preserve other 3 bytes -> uint32 CAS retry
```

##### 源码等价伪码

```text
word=&uint32_var[(dst*1)>>2]; shift=(dst&3)*8; keep=~(0xff<<shift)
old=int8((oldWord>>shift)&0xff)
new=int8(int32(old)/int32(update))
CAS word from oldWord to (oldWord&keep)|(uint8(new)<<shift); retry on conflict
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputs GM | `m`,`g` | `I`,`I*S` | Index/INT8 | input | VF | caller | scalar load |
| var enclosing word | `dst>>2`, byte lane `dst&3` | 4B | 4×INT8 | input/CAS | output | caller | full-word atomic |
| registers | thread-local | 固定仓不可见 | uint32 + INT32 ACC | helper | helper | thread | — |
| workspace | slice 0 | 16 MiB | bytes | Tiling | unused | caller | — |

##### 分层内存布局

N/A：GM-direct；一个 4B CAS transaction 更新其中一个 INT8 byte，另外三个 byte 由 `keepMask` 保留。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; workspace_total=16 MiB
atomic transaction=4 B; logical field=1 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| uint32 enclosing word | byte-lane updater | competitors/output | next CAS | CAS retry + keepMask | 相邻 byte 竞争也必须重新 merge |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| arithmetic | INT8→INT32 divide→INT8 | GM values | CAS | selected byte | 每次更新窄化；除零未 guard |
| atomicity | uint32 CAS | contenders | retry loop | four-byte word | preserves neighbor lanes |
| tail | grid stride | block range | thread | remaining elements | explicit loop bound |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | common Tiling cpp | `ScatterReduceCommonTiling` | range formulas |
| Kernel entry/router | ScatterDiv Kernel cpp | `ScatterReduceSimt<int8_t,...,MODE_DIV>` | dtype specialization |
| Device body/helper | common SIMT header | `AtomicReduceSubWord<int8_t>` | signed decode, INT32 divide, byte merge |

### Route R-005：UINT8 enclosing-word CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + `scatter_div_def.cpp` | ACLNN / `ScatterDiv` | `DT_UINT8` 配对 | UINT8 instance |
| Tiling | common Tiling | `SetTilingKey(0)` | unconditional | key 0 |
| Router | `scatter_div.cpp` | `ScatterReduceSimt` | `T=uint8_t` | subword route |
| Body | common SIMT header | `AtomicReduceSubWord` | final unsigned decode branch | INT32 ACC + UINT8 merge |

#### Device 数据流

##### 函数级调用链

```text
scatter_div<uint8_t,Index> -> SimtComputeReduce
  -> AtomicReduceSubWord<uint8_t,MODE_DIV>
    -> uint8 decode -> INT32 divide -> uint8 narrow
    -> preserve other bytes -> uint32 CAS retry
```

##### 源码等价伪码

```text
locate uint32 word and byte shift from dst
old=uint8(selected bits); new=uint8(int32(old)/int32(update))
merge new byte with three bytes from prev; atomicCAS; retry if prev changed
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputs GM | `m`,`g` | `I`,`I*S` | Index/UINT8 | input | VF | caller | direct load |
| var enclosing word | `dst>>2`, lane `dst&3` | 4B | 4×UINT8 | input/CAS | output | caller | full-word atomic |
| registers | thread-local | 固定仓不可见 | uint32 + INT32 ACC | helper | helper | thread | — |
| workspace | slice 0 | 16 MiB | bytes | Tiling | unused | caller | — |

##### 分层内存布局

N/A：无 UB/dynamic UB；通过 4B GM CAS 更新一个 UINT8 lane。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; workspace_total=16 MiB
atomic transaction=4 B; logical field=1 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| enclosing word | CAS winner | next contender/output | next CAS | compare-and-retry | merge from current `prev` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| arithmetic | UINT8→INT32 divide→UINT8 | input values | CAS | selected byte | divide-by-zero no guard |
| atomicity | uint32 CAS | threads | retry | enclosing word | neighbor lanes preserved |
| bounds | index/tail guards | Tiling + indices | VF | GM extents | invalid index skipped |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | common Tiling cpp | `ScatterReduceCommonTiling` | key/POD/core split |
| Kernel entry/router | ScatterDiv Kernel cpp | `ScatterReduceSimt<uint8_t,...,MODE_DIV>` | UINT8 instance |
| Device body/helper | common SIMT header | `AtomicReduceSubWord<uint8_t>` | unsigned decode + INT32 divide + CAS |

## 算子特定优化

### Trick T-001：GM-direct 512-thread grid-stride 消除 UB tile 上限

#### 触发与源码依据

- 适用 Route：R-001～R-005。
- 触发条件：`sliceSize>0 && blockIdx<blockNum`；当前核整个 `currBlockTilingSize_` 一次交给 VF。
- Golden 来源：[`scatter_reduce_common_simt.h:163-174`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L163-L174)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
asc_vf_call<SimtComputeReduce<INDICES_T, PARAMS_T, ADDR_T, Mode>>(
    dim3(THREAD_NUM),
```

#### 机制与设计原因

Host 以 index slice 为粒度切核，但不再按 UB 容量切 tile；VF 的 512 个线程用 `index += blockDim.x` 遍历任意长的本核元素域，indices、updates、var 都直接从 GM 访问。这样源码上消除了 Queue/TBuf、DMA tail 和 per-tile 上限，也避免了为冲突更新在 UB 与 GM 间往返；代价是每个 update 元素都直接发生 GM 读取与 CAS，且同一 slice 内会重复读取 `indices[m]`。

#### 迁移边界

- 适用：输出地址由数据决定、必须对 GM 竞争目标做原子更新的 elementwise scatter-reduce。
- 前提：950 SIMT VF、合法 GM 标量访问、目标 dtype 有直接 CAS 或可安全封装到 32-bit word。
- 失效：若计算需要邻域复用、Vector 批处理或目标类型无法构造安全 CAS，必须另设 UB/Vector Route；固定仓没有这类 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：subword enclosing-word CAS 保留相邻元素

#### 触发与源码依据

- 适用 Route：R-003～R-005。
- 触发条件：`sizeof(PARAMS_T)!=4`，即 FP16/INT8/UINT8；GM base 满足源码注释要求的至少 4B 对齐。
- Golden 来源：[`scatter_reduce_common_simt.h:66-70`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L66-L70)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr uint32_t SUBMASK = (sizeof(T) == 1) ? 0xFFU : 0xFFFFU;
uint64_t byteOff = elemIdx * sizeof(T);
```

#### 机制与设计原因

硬件 helper 没有 sub-32-bit CAS overload，因此先定位目标所在 32-bit word 与 byte shift，提取目标 field，升到 FP32（half）或 INT32（int8/uint8）计算，再用 `keepMask` 合回同一 word。CAS 失败时以新读到的整个 word 重新解码和合并，因而相邻 field 被其他线程更新也不会被陈旧快照覆盖。关键常量 `0xFF/0xFFFF` 由 `sizeof(T)` 决定，shift 来自 `elemIdx*sizeof(T)` 的低两位。

#### 迁移边界

- 适用：1B/2B 元素原子更新，且能用 4B 对齐 word 包裹并定义升/降精度语义。
- 前提：allocation 首地址和尾部允许 enclosing-word 4B 访问；转换、溢出、除零语义满足目标算子契约。
- 失效：若相邻 bits 不属于同一可原子访问 allocation，或不能接受每次更新后的降精度，不能复用；固定仓没有 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只写目标 byte/half 而不把相邻 field 纳入同一个 CAS，会在相邻元素并发更新时发生 lost update；源码用 full-word `prev`、`keepMask` 和 retry 避免该问题。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 INT32 | R-003 FP16 | R-004 INT8 | R-005 UINT8 | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | FP32 signature | INT32 | FP16 | INT8 | UINT8 | 由 OpDef 编译实例选定；Tiling key 都是 0 |
| 模板与 dtype | direct float | direct int32 | half + float ACC | int8 + int32 ACC | uint8 + int32 ACC | `sizeof(T)` 与 `IsSameType` 编译分支 |
| 循环与分核 | 512-thread grid-stride | 同 | 同 | 同 | 同 | 全 Route 共用 Tiling/body |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅寄存器 | 同 | word/mask/FP32 regs | word/mask/INT32 regs | 同 | 无 UB/dynamic UB/通信 |
| 内存总占用与复用 | 4B target CAS | 4B target CAS | 4B word/2B field | 4B word/1B field | 同 | subword 必须保留邻接 lanes |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM load + float CAS | GM load + int CAS | uint32 CAS | uint32 CAS | uint32 CAS | 无 Vector DMA/NDDMA/通信 |
| 同步、精度与 tail | FP32，CAS order | INT32，逐次截断 | FP32 算后每次降 half | INT32 算后窄化 int8 | INT32 算后窄化 uint8 | tail/index guard 相同 |
| fallback/失败边界 | 无 fallback | 无除零 guard | 无 alignment fallback | 无除零 fallback | 同 | 不支持 dtype 由 ACLNN 拒绝 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持 | 公开 README/API 明确不支持 | Ascend 950PR/950DT 支持 |
| Host/Launcher | 不发布正式卡 | 不发布正式卡 | arch35 Tiling + generated ACLNN boundary；blockDim 按 indices 与 AIV cores 计算 |
| Kernel/资源 | 不适用 | 不适用 | ascend950 单 AIV-only entry；内部 SIMT、GM-direct CAS、无 UB |

## 限制与失败边界

- 固定仓未签入 `aclnnScatterDiv*` 的生成 C++/头文件；非连续 Tensor 的具体 Contiguous/View 节点、executor 内部 workspace offset 与运行包装符号属于 `opbuild` 生成框架边界，不可见。
- 公开契约要求 `updates.shape=indices.shape+varRef.shape[1:]`；可见 Tiling 只读取 var/indices shape，不再次校验 updates shape，依赖 ACLNN/OpDef 生成层守约。
- `useLocking` 未进入 Tiling/POD/Kernel；无论属性取值，Device 更新都使用 CAS。固定源码不证明该属性会改变行为。
- 重复索引的 CAS 成功顺序未规定；FP32/FP16 舍入及整数逐次截断可能随并发顺序变化。
- 整数 Route 没有 divisor 为 0 或 `INT_MIN/-1` 的 guard；固定仓没有给出这些输入的 Device 结果语义。FP32 CAS retry 用 `cur != prev` 作值比较，源码也没有 NaN 特判，故固定仓不建立 NaN 竞争下的终止保证。
- `varTotal`、`P*S`、`Pt*S` 及地址乘法没有显式 `uint64_t` 溢出检查；Kernel 的地址类型固定 `int64_t`，从 POD `uint64_t` 转入时也没有范围 guard。
- subword Route 依赖 GM base 的 4B 对齐及 enclosing word 位于 allocation 内；源码注释称 GM tensor 至少 32B 对齐，但未提供运行时检查或 fallback。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter_div/docs/aclnnScatterDiv.md` | `aclnnScatterDivGetWorkspaceSize/aclnnScatterDiv` | 两段式原型、dtype/shape、错误码、非连续支持 |
| 平台/语义 | `index/scatter_div/README.md` | 产品支持表、功能说明 | 950 支持，A2/A3 不支持，原地语义 |
| 入口生成定义 | `index/scatter_div/op_host/scatter_div_def.cpp` | `ScatterDiv`, `OP_ADD` | dtype signature、属性、ascend950 OpDef |
| shape/dtype | `index/scatter_div/op_host/scatter_div_infershape.cpp` | `InferShape4ScatterDiv`, `InferDataType4ScatterDiv` | 输出复制 var |
| Host wrapper | `index/scatter_div/op_host/arch35/scatter_div_tiling.cpp` | `Tiling4ScatterDiv` | 委托共享 Tiling 与 parse producer |
| Tiling 声明 | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.h` | `ScatterReduceCompileInfo`, `TilingPrepareForScatterReduce` | compileInfo ABI 与共享入口声明 |
| Host/Tiling | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ResolveCoreNum`, `ScatterReduceCommonTiling` | 分核、POD、key、workspace |
| POD | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h` | `ScatterReduceSimtTilingData`, `MODE_DIV` | direct struct ABI 与 mode |
| Kernel entry/body | `index/scatter_div/op_kernel/arch35/scatter_div.cpp` | `scatter_div` | AIV-only entry 与模板绑定 |
| Kernel body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt`, `SimtComputeReduce`, atomic helpers | GM-direct SIMT、CAS、精度、tail |
| 构建接线 | `index/scatter_div/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `add_modules_sources` | ascend950、arch35、ACLNN 与 common 依赖 |
| Kernel 构建 | `index/scatter_div/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch35 entry 仅接 ascend950 |
| 公共依赖构建 | `index/scatter_reduce_common/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `add_modules_sources` | 共享 Host/Tiling 仅接 ascend950/arch35 |
| ACLNN 收集 | `cmake/func.cmake` | `add_modules_sources` | `ACLNNTYPE aclnn` 收集 OpDef |
| ACLNN 生成 | `cmake/opbuild.cmake` | `gen_aclnn_classify`, `gen_aclnn_with_opdef` | 从 OpDef 生成 ACLNN 源/头的框架边界 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 五字段 direct struct | 字段顺序、全为 uint64_t、Host/Device 共用头；key 固定 0 | `ScatterReduceSimtTilingData`, `REGISTER_TILING_DEFAULT` |
| 调用顺序 | Tiling→单 AIV entry→Init→Process→一次 VF call | `sliceSize==0` 必须先退出，避免 VF 除零 | `Tiling4ScatterDiv`, `ScatterReduceSimt::Process` |
| workspace/内存 | 16 MiB 系统 slice；核心 GM-direct | `var` 原地目标；`y/workspace` 不得误当核心数据边 | `ScatterReduceCommonTiling`, `scatter_div` |
| 同步 | 同目标 word 用 CAS retry | subword merge 必须基于每轮 `prev`；不能拆成非原子 load/store | `AtomicReduce32`, `AtomicReduceSubWord` |
| 精度与尾块 | FP16 升 FP32、INT8/UINT8 升 INT32；grid-stride tail | 每次 CAS 更新后按目标 dtype 降精度；非法 index skip | `ReduceAcc`, `SimtComputeReduce` |
| 失败边界 | Host null/platform/Tiling failure 阻断；unsupported dtype 由 ACLNN 拒绝 | 不补造除零、NaN、overflow 或 alignment fallback | ACLNN 文档，`ResolveCoreNum`, atomic helpers |
