---
type: Operator
title: "ForeachAddcmulScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachAddcmulScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/docs/aclnnForeachAddcmulScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/foreach_addcmul_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/arch35/foreach_addcmul_scalar_list_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/arch35/foreach_addcmul_scalar_list_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:30:28Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddcmulScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddcmulScalarListGetWorkspaceSize` + `aclnnForeachAddcmulScalarList`，公开 ACLNN，两段式，无更高版本符号。 |
| 计算语义 | 对列表下标 `i` 和张量内元素 `j`：`y_i[j] = x1_i[j] + scalars[i] * x2_i[j] * x3_i[j]`。 |
| 输入输出 | `x1/x2/x3/y` 为等长 tensor list；对应张量 shape 相同；`scalars.numel == tensorCount`；ND；FP16/FP32/INT32/BF16。 |
| 执行类型 | 单 Kernel `foreach_addcmul_scalar_list`；Kernel 内按 tensor 顺序发起 SIMT `VF_CALL`，无 AIC/Cube/MIX_CV。 |
| Kernel 路线 | R-001 FP16/BF16→FP32 升精度；R-002 FP32 原生；R-003 INT32 原生。 |
| 核心资源 | tensor-list 描述符与数据均在 GM；512 threads/block；dynamic UB=0；算子 workspace=0；无 TQue/TBuf/atomic/barrier。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→生成式 ACLNN/AutoContiguous 框架边界→OpDef→arch35 Tiling/POD→模板 key→SIMT entry/body→ascend950 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令四个 tensor list 的有效长度为 `n`，则 `scalars` 有 `n` 个元素。每个 `i∈[0,n)` 上，三路输入张量对应元素执行左结合源码表达式 `cx1 + scalarVal * cx2 * cx3`，写入 `y_i`。FP16/BF16 路线先将四个操作数逐元素转为 FP32，计算后转回原 dtype；FP32 与 INT32 使用原 dtype 计算。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 动态列表，各 tensor ND、0–8 维 | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；允许零元素 tensor；对应 `x2/x3` shape 相同。 |
| `x2` | 与 `x1` 等长，逐位置同 shape | 同 `x1` | 乘法输入。 |
| `x3` | 与 `x1` 等长，逐位置同 shape | 同 `x1` | 乘法输入。 |
| `scalars` | ND、0–8 维，元素数等于列表长度 | 同 `x1` | 第 `i` 个值作用于第 `i` 个 tensor。 |
| `out/y` | 动态列表、ND；每个输出容量不少于对应 `x1` 元素数 | 同 `x1` | 非连续输出不支持；Kernel 只写 `x1_i.numel` 个元素。 |

### 入口与版本

公开文档声明唯一两段式版本：第一段返回 workspace 大小与 executor，第二段在给定 stream 上执行。OpDef 的五路参数均配置 `AutoContiguous()`；自动生成的 ACLNN inner bridge、连续化节点和 executor 壳未以手写文件出现在固定仓，属于明确框架边界。公开产品表明确支持 Ascend 950PR/950DT，OpDef `AddConfig("ascend950")`、Host `arch35`、Kernel `ascend950` 构建接线一致，无平台冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段检查空指针、四路输入/输出 dtype 范围与一致性；生成框架按 OpDef 对非连续输入做连续化。
2. arch35 Tiling 从动态输入 0 (`x1`) 取得实例数，拒绝 `tensorCount > 128`，并扫描每个 `x1_i` 的 storage shape 得到 `maxTensorElements`；只从 `x1_0` 取得 dtype。
3. Tiling 按 dtype 生成 key 0/1/2/3，设置 `blockDim`、POD、local memory=0、operator workspace=0。对应列表长度、shape/dtype 一致性由公开 ACLNN/OpDef 合约约束；arch35 Tiling 本体不重复这些逐列表检查。
4. 第二段让 executor 在用户 stream 上启动单个 `foreach_addcmul_scalar_list<schMode>`。生成 bridge 的错误传播细节固定仓不可见；Tiling 中任一空指针、平台信息或 local-memory 设置失败均返回 `GRAPH_FAILED`，阻断 Kernel。

### 参数、Workspace 与生命周期

设平台 AIV core 数为 `C`，最大单 tensor 元素数为 `M`：

```text
perCoreElements = 0, needCoreNum = 1                         (M <= 0)
perCoreElements = align_up(max(1024, ceil(M / C)), 32)       (M > 0)
needCoreNum = min(C, max(1, ceil(M / perCoreElements)))
blockDim = needCoreNum
localMemorySize = 0
operatorWorkspace[0] = 0
```

POD 与 tiling buffer 由 executor 持有至 launch 参数消费完成；GM tensor-list 描述符、scalars 和输出由调用方维持到 stream 完成。公开 ACLNN 总 workspace 还可能包含生成式连续化节点的空间，其公式在固定仓不可见，不能用 operator workspace=0 替代总 API workspace。

### 启动图与失败传播

```text
aclnn...GetWorkspaceSize
  -> generated contiguous/executor boundary
  -> ForeachAddcmulScalarListTilingFunc
     -> blockDim=needCoreNum, key=dtype, POD, localMemory=0, op-workspace=0
aclnnForeachAddcmulScalarList(stream)
  -> foreach_addcmul_scalar_list<schMode>(x1,x2,x3,scalars,y,workspace,tiling)
     -> for tensor i: VF_CALL<Op...<T,CalcT>>(Dim3(512), ...)
```

这是一个 Device Kernel 节点；Kernel entry 的 `workspace` 未读取。各 tensor 的 `VF_CALL` 在同一 entry 的源码循环中依次发起，没有跨 Kernel workspace 边。框架负责 stream 上 launch 与失败返回；固定仓未公开更细的 executor owner/cleanup 实现。

## Route 与参数设计

### 分核与资源计算

Host 用最大 tensor 的元素数计算一次 `blockDim`，每个 tensor 都复用这些 blocks。SIMT 中线程全局线性索引为 `blockIdx*512+threadIdx`，步长为 `512*blockNum`；因此每个 tensor 的每个有效元素恰由一个线程处理，尾部由 `index < tensorElemCount` 保护。`perCoreElements` 和 POD `needCoreNum` 在 Device body 中不读取；实际网格来自 launch `blockDim`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 升精度 SIMT | key 0 `Process<half,float>`；key 3 `Process<bfloat16_t,float>` | `x1_0.dtype` 为 FP16 或 BF16 | 继续 dtype switch；不支持 dtype 在 OpDef/API 校验失败。 |
| 2 | R-002 FP32 SIMT | key 1 `Process<float,float>` | `x1_0.dtype == DT_FLOAT` | 继续 dtype switch。 |
| 3 | R-003 INT32 SIMT | key 2 `Process<int32_t,int32_t>` | `x1_0.dtype == DT_INT32` | 不支持 dtype 失败；`GetTilingKeyByDtype` 的 default=0 仅是函数默认分支，不是公开可达 fallback。 |

```text
dtype == FP16  -> key 0 -> schMode 0 -> Process<half,float>
dtype == FP32  -> key 1 -> schMode 1 -> Process<float,float>
dtype == INT32 -> key 2 -> schMode 2 -> Process<int32_t,int32_t>
dtype == BF16  -> key 3 -> schMode 3 -> Process<bfloat16_t,float>
else           -> ACLNN/OpDef unsupported failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 升精度 SIMT | 0/3 | FP16/BF16；`tensorCount<=128`；950 | 单 Kernel；每 tensor 一个顺序 `VF_CALL` | `Process<T,float>`→`Op...<T,float>` | GM 直读直写，0 dynamic UB，无 barrier | 空 tensor 跳过；线程循环谓词保护尾部。 |
| R-002 FP32 SIMT | 1 | FP32；同上 | 同上 | `Process<float,float>` | 同上 | 同上。 |
| R-003 INT32 SIMT | 2 | INT32；同上 | 同上 | `Process<int32_t,int32_t>` | 同上 | 同上；整数溢出行为固定仓未另行定义。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| tensor 数上限 | `tensorNum <= 128` | POD `tensorCount` 为 int32，Kernel 顺序遍历；Host 明确限制 | 全部 | `GRAPH_FAILED` | `ValidateAndGetTensorCount` |
| 最少每核工作量 | `perCoreElements >= 1024` 且 32 元素对齐 | 限制 blockDim，避免为短 tensor 启动过多 blocks | 全部 | 仍走同 Route，只减少 blocks | `ComputeCoreSplit` |
| 零元素 tensor | `elemCount == 0` | 不发起无效 VF | 全部 | `continue` | `Process` |
| 固定线程数 | `THREAD_NUM == 512` | `LAUNCH_BOUND(512)` 与 `VF_CALL Dim3(512)` 必须一致 | 全部 | 无运行时 fallback | `THREAD_NUM`, `Op...`, `Process` |
| dtype 模板闭包 | key 0..3 | entry 仅实例化四个 `schMode` | 全部 | 不支持 dtype 应在 Host/API 失败 | `GetTilingKeyByDtype`, `ForeachAddcmulScalarListTilingKey` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAddcmulScalarListTilingData` | `tensorCount` | `int32_t`/32 | offset 0；结构 size 16、align 8 | tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`: 循环上界 | Host 拒绝 >128。 |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | block 数 | `static_cast<int32_t>(needCoreNum)` | 未读取；实际值由 `SetBlockDim` 进入 launch | inactive Device 字段。 |
| 同上 | `perCoreElements` | `int64_t`/64 | offset 8 | 元素/core | `ComputeCoreSplit` | 未读取 | inactive Device 字段。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 0/3 | `op_host/arch35/..._tiling.cpp::ForeachAddcmulScalarListTilingFunc` | `..._tiling.h::ForeachAddcmulScalarListTilingData` | `op_kernel/arch35/...cpp::foreach_addcmul_scalar_list<0/3>` | `..._simt.h::Process<T,float>`, `Op...<T,float>` | `op_kernel/CMakeLists.txt::add_kernel_sources(ascend950)` |
| R-002 | key 1 | 同上 | 同上 | `foreach_addcmul_scalar_list<1>` | `Process<float,float>` | 同上 |
| R-003 | key 2 | 同上 | 同上 | `foreach_addcmul_scalar_list<2>` | `Process<int32_t,int32_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 16-byte POD 后，以编译期 `schMode` 选择唯一 `Process<T,CalcT>`。每个 AIV block 均遍历 `i=0..tensorCount-1`，从 tensor-list 描述符解析四路 GM 指针和 `x1_i` 元素数；非空 tensor 用 512 线程 VF 并行直接计算、写 GM。没有显式 UB 分配、workspace、atomic、queue 或 barrier。

### Route R-001：FP16/BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/foreach_addcmul_scalar_list_tiling.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16`→0；`DT_BF16`→3 | 编译期 schMode 0/3 |
| Router | `op_kernel/arch35/foreach_addcmul_scalar_list.cpp` | `foreach_addcmul_scalar_list<schMode>` | `if constexpr schMode==0/3` | `Process<half,float>` / `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_addcmul_scalar_list<0/3>
  -> Process<T,float>
    -> SimtGetTensorAddr<T> x1/x2/x3/y
    -> GetTensorElemCount(x1,i)
    -> VF_CALL<OpForeachAddcmulScalarListSimt<T,float>>(Dim3(512), ...)
```

##### 源码等价伪码

```text
for i in [0, tensorCount):
  resolve x1_i, x2_i, x3_i, y_i from list descriptors
  N = product(shape(x1_i)); if N == 0: continue
  s = scalars[i]
  parallel tid = blockIdx*512 + threadIdx; tid < N; tid += 512*blockNum:
    c1,c2,c3,cs = fp32(x1_i[tid]),fp32(x2_i[tid]),fp32(x3_i[tid]),fp32(s)
    y_i[tid] = T(c1 + cs*c2*c3)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tensor-list descriptor | `base + offset/8`, pointer slot `+i` | 固定仓框架 ABI | uint64 descriptor | ACLNN/framework | `SimtGetTensorAddr` | executor launch 生命周期 | 指针 offset 右移 3。 |
| x1/x2/x3/y GM | resolved pointer + `index` | `N*sizeof(T)` | FP16/BF16 ND-linear | caller/framework | SIMT thread | 至 stream 完成 | 每线程标量访问。 |
| scalar GM | `scalars+i` | `tensorCount*sizeof(T)` | T | caller | `Process` | 同上 | 每 tensor 读取一次/raw value。 |
| registers | per thread `scalarVal,cx1,cx2,cx3,result` | 编译器决定 | FP32 | thread loads/casts | expression/store | 单循环迭代 | 无跨线程共享。 |

##### 分层内存布局

N/A：该 SIMT Route 不调用 `InitBuffer`，`SetLocalMemorySize(0)`；数据从 GM 进入线程标量/寄存器后直接写 GM。

##### 字节账与复用顺序

```text
AIV_UB_total = 0 (源码无 TQue/TBuf/LocalTensor)
dynamic_ub = 0
operator_workspace_total = 0
register_bytes = 固定仓不可见（编译器分配）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程寄存器 | 当前 thread load/cast | 当前表达式与 store | 下一 loop index | 单线程程序顺序 | 不跨线程共享。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tensor 调度 | entry 中顺序 `VF_CALL` | tensor i | 下一 tensor | 每次 VF 调用参数 | 空 tensor `continue`。 |
| 元素计算 | SIMT grid-stride loop | GM loads | GM store | 唯一 `index` | 全部操作数升 FP32，结果 cast 回 T；`index<N` 保护尾部。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/..._tiling.cpp` | `ComputeCoreSplit`, `SetTilingContext` | 生产 blockDim/key/local memory/workspace。 |
| Kernel entry/router | `op_kernel/arch35/...cpp` | `foreach_addcmul_scalar_list<0/3>` | key 到 `Process<T,float>`。 |
| Device body/helper | `op_kernel/arch35/..._simt.h` | `Process`, `Op...<T,float>` | descriptor、512-thread grid-stride、FP32 升精度、tail。 |

### Route R-002：FP32 原生 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/..._tiling.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT` | key 1 |
| Router | `op_kernel/arch35/...cpp` | `foreach_addcmul_scalar_list<1>` | schMode 1 | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_addcmul_scalar_list<1> -> Process<float,float>
  -> descriptor helpers -> VF_CALL<Op...<float,float>>(Dim3(512), ...)
```

##### 源码等价伪码

```text
for each nonempty tensor i, grid-stride index:
  y_i[index] = x1_i[index] + scalars[i] * x2_i[index] * x3_i[index]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs/output | tensor pointer + `index` | `N*4` bytes each | FP32 linear ND | caller | thread | stream lifetime | 直接标量访问。 |
| registers | current scalar and 4 temporaries | compiler-defined | FP32 | thread | expression | iteration | 无共享。 |

##### 分层内存布局

N/A：无 AIV UB；实际替代路径为 GM↔线程寄存器。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; operator_workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | load | expression/store | 下一 index | 程序顺序 | 每 index 独占。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | grid-stride | loads | store | index | FP32 原生；`index<N`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/..._tiling.cpp` | `GetTilingKeyByDtype` | FP32→1。 |
| Kernel entry/router | `op_kernel/arch35/...cpp` | `foreach_addcmul_scalar_list<1>` | `Process<float,float>`。 |
| Device body/helper | `op_kernel/arch35/..._simt.h` | `Op...<float,float>` | FP32 expression 与尾保护。 |

### Route R-003：INT32 原生 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/..._tiling.cpp` | `GetTilingKeyByDtype` | `DT_INT32` | key 2 |
| Router | `op_kernel/arch35/...cpp` | `foreach_addcmul_scalar_list<2>` | schMode 2 | `Process<int32_t,int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_addcmul_scalar_list<2> -> Process<int32_t,int32_t>
  -> descriptor helpers -> VF_CALL<Op...<int32_t,int32_t>>(Dim3(512), ...)
```

##### 源码等价伪码

```text
for each nonempty tensor i, grid-stride index:
  int32 result = x1_i[index] + scalars[i] * x2_i[index] * x3_i[index]
  y_i[index] = result
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs/output | tensor pointer + `index` | `N*4` bytes each | INT32 linear ND | caller | thread | stream lifetime | 直接标量访问。 |
| registers | current scalar and temporaries | compiler-defined | INT32 | thread | expression | iteration | 无共享。 |

##### 分层内存布局

N/A：无 AIV UB；实际替代路径为 GM↔线程寄存器。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; operator_workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | load | expression/store | 下一 index | 程序顺序 | 每 index 独占。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | grid-stride | loads | store | index | INT32 原生；溢出/舍入规则未在固定仓补充；`index<N`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/..._tiling.cpp` | `GetTilingKeyByDtype` | INT32→2。 |
| Kernel entry/router | `op_kernel/arch35/...cpp` | `foreach_addcmul_scalar_list<2>` | `Process<int32_t,int32_t>`。 |
| Device body/helper | `op_kernel/arch35/..._simt.h` | `Op...<int32_t,int32_t>` | INT32 expression 与尾保护。 |

## 算子特定优化

### Trick T-001：按最大 tensor 限制 blockDim、逐 tensor 复用网格

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003。
- 触发条件：`maxTensorElements > 0`；`perCoreElements` 先取 `max(1024,ceil(M/C))` 再按 32 元素向上对齐。
- Golden 来源：[`foreach_addcmul_scalar_list_tiling.cpp:130-145`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/arch35/foreach_addcmul_scalar_list_tiling.cpp#L130-L145)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
perCoreElements = std::max(SINGLE_CORE_MIN_ELEMENTS,
    (maxTensorElements + coreNum - 1) / coreNum);
perCoreElements = ((perCoreElements + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
needCoreNum = (maxTensorElements + perCoreElements - 1) / perCoreElements;
```

#### 机制与设计原因

Host 不按 tensor 总和分块，而以最大单 tensor 选择一次网格，并设 1024 元素/core 下限；所有 tensor 依次复用该网格。较短 tensor 由 SIMT `index<N` 自然裁剪，不需要逐 tensor POD 或多次 Host launch。该机制改变调度粒度；并不证明具体性能收益。

#### 迁移边界

- 适用：tensor list 内每个 tensor 可独立点运算、且能用 grid-stride 尾保护的 SIMT 算子。
- 前提：每次 VF 的 blockNum 与 Host `blockDim` 一致；tensor 之间无跨 tensor 依赖。
- 失效：需要 tensor 专属网格/共享状态时必须增加 POD 或拆分 launch；本算子无替代 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16 在线程内升至 FP32、零中间物化

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：模板 `CalcT=float`，输入 `T=half/bfloat16_t`。
- Golden 来源：[`foreach_addcmul_scalar_list_simt.h:76-89`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_simt.h#L76-L89)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
CalcT cx1 = static_cast<CalcT>(x1Ptr[index]);
CalcT cx2 = static_cast<CalcT>(x2Ptr[index]);
CalcT cx3 = static_cast<CalcT>(x3Ptr[index]);
CalcT result = cx1 + scalarVal * cx2 * cx3;
yPtr[index] = static_cast<T>(result);
```

#### 机制与设计原因

四个低精度操作数只在线程寄存器中转为 FP32，完成两次乘与一次加后一次性降回输出；没有 GM/UB 的 FP32 中间张量。这同时闭合精度顺序和“避免中间物化”的内存机制。

#### 迁移边界

- 适用：逐元素表达式且输出仍为原低精度 dtype。
- 前提：目标 SIMT 支持相应 cast；公式顺序可保持。
- 失效：要求逐步低精度舍入或需要 FP32 输出时不能直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 升精度 | R-002 FP32 | R-003 INT32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 0/3 | key 1 | key 2 | 仅 dtype switch，无 shape fallback。 |
| 模板与 dtype | `T=half/bf16, CalcT=float` | `float,float` | `int32,int32` | 低精度浮点升精度。 |
| 循环与分核 | 512-thread grid-stride | 同左 | 同左 | 共用最大 tensor blockDim。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程 FP32 locals，dynUB 0 | FP32 locals | INT32 locals | 无共享 buffer。 |
| 内存总占用与复用 | 无显式 UB/workspace | 同左 | 同左 | GM 直达线程。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar load/store | 同左 | 同左 | 单 Kernel SIMT。 |
| 同步、精度与 tail | FP32→T；`index<N` | FP32；同 tail | INT32；同 tail | dtype 决定 CalcT。 |
| fallback/失败边界 | 不支持 dtype/API 失败 | 同左 | 同左 | 无 Device default body。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；见无后缀卡 | 公开支持；见无后缀卡 | 公开支持 Ascend 950PR/DT。 |
| Host/Launcher | arch22 common Tiling | arch22 common Tiling | arch35 专用 Tiling，0 workspace/local memory，最多 128 tensors。 |
| Kernel/资源 | AIV Vector + UB/TQue | AIV Vector + UB/TQue | SIMT 512 threads，GM 直读直写，无显式 UB。 |

## 限制与失败边界

- arch35 Tiling 只扫描 `x1` 并从 `x1_0` 取 dtype；跨列表 shape/dtype/长度一致性依赖公开 ACLNN/OpDef 框架校验，固定仓没有 arch35 本体中的重复检查。
- `GetTensorElemCount` 以 tensor-list 首个 descriptor 的 `dimSize` 推导所有 descriptor 的步长；若同一 list 内 tensor rank 不同，固定源码未提供替代解析 Route。公开接口只要求对应列表 tensor shape 相同，未声明列表内部 rank 必须相同，因此这是需调用方谨慎处理的源码边界。
- `GetPlatformInfoFallback` 在 compile/platform 信息均不可用时使用 24 cores、1 MiB UB；UB 值随后不参与 950 算法。TilingParse 正常路径仍要求有效 AIV core/UB。
- `perCoreElements`、POD `needCoreNum` 不被 Device 读取；实现复用时不能误认为 Kernel 用它们限定线程循环。
- operator workspace 为 0 不等于公开 ACLNN 总 workspace 必为 0；AutoContiguous/executor 生成实现不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_addcmul_scalar_list/docs/aclnnForeachAddcmulScalarList.md` | `aclnnForeachAddcmulScalarList*` | 公开语义、参数、950 支持与错误。 |
| 入口实现 | `foreach/foreach_addcmul_scalar_list/op_host/foreach_addcmul_scalar_list_def.cpp` | `ops::ForeachAddcmulScalarList` | dtype/layout、AutoContiguous、ascend950 注册。 |
| Host/Tiling/Launcher | `foreach/foreach_addcmul_scalar_list/op_host/arch35/foreach_addcmul_scalar_list_tiling.cpp` | `ForeachAddcmulScalarListTilingFunc` | 校验、分核、key、workspace。 |
| Host/Tiling/Launcher | `foreach/foreach_addcmul_scalar_list/op_host/arch35/foreach_addcmul_scalar_list_tiling.h` | `ForeachAddcmulScalarListTilingData` | POD ABI。 |
| Kernel entry/body | `foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list.cpp` | `foreach_addcmul_scalar_list<schMode>` | key router。 |
| Kernel entry/body | `foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_simt.h` | `Process`, `OpForeachAddcmulScalarListSimt` | descriptor、SIMT loop、精度/tail。 |
| Kernel entry/body | `foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_tiling_data.h` | `ForeachAddcmulScalarListTilingData` | Device 侧 POD 同构定义。 |
| Kernel entry/body | `foreach/foreach_addcmul_scalar_list/op_kernel/arch35/foreach_addcmul_scalar_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 0–3 的编译期实例集合。 |
| 构建接线 | `foreach/foreach_addcmul_scalar_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 Host。 |
| 构建接线 | `foreach/foreach_addcmul_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch35 source→ascend950。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 16-byte 三字段 POD | 字段顺序/位宽；后两字段当前 Device inactive | `ForeachAddcmulScalarListTilingData` |
| 调用顺序 | 单 Kernel 内逐 tensor VF | descriptor 解析先于 `VF_CALL`；空 tensor 跳过 | `Process` |
| workspace/内存 | 0 local memory/operator workspace | ACLNN 总 workspace 仍由 executor 决定 | `SetTilingContext` |
| 同步 | 每 index 独占，无 atomic/barrier | 不能引入跨线程共享而不补同步 | `Op...Simt` |
| 精度与尾块 | FP16/BF16→FP32→T；grid-stride tail | cast 顺序与 `index<N` | `Op...Simt` |
| 失败边界 | >128 tensors 失败；四 dtype only | default key 0 不是公开 fallback | `ValidateAndGetTensorCount`, `GetTilingKeyByDtype` |