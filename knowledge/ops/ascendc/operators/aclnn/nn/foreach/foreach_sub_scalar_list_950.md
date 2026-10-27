---
type: Operator
title: "ForeachSubScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachSubScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/docs/aclnnForeachSubScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_host/foreach_sub_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_proto_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_host/config/ascend950/foreach_sub_scalar_list_binary.json
    role: implementation
created_at: "2026-08-25T16:22:40Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSubScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSubScalarListGetWorkspaceSize` + `aclnnForeachSubScalarList`，公开两段式 ACLNN；固定仓只公开接口文档，inner bridge 为生成框架边界。 |
| 计算语义 | 对列表位置 `t` 和元素 `j`，`y_t[j] = x_t[j] - scalars[t]`；只写 `0 <= j < numel(x_t)`。 |
| 输入输出 | `x/y` 为 ND TensorList；`x/y` dtype 相同，支持 FP32/FP16/INT32/BF16；对应 `scalars` 为 FP32/FP32/INT64/FP32。 |
| 执行类型 | 单节点 `SIMT`：AIV Kernel 内通过 `VF_CALL` 启动 1024-thread SIMT body；无 Cube/AIC 或 MIX_CV 节点。 |
| Kernel 路线 | R-001 FP32、R-002 FP16→FP32、R-003 INT32→INT64、R-004 BF16→FP32。 |
| 核心资源 | GM TensorList 指针表、GM scalars、每线程寄存器；body 未分配 `TQue/TBuf/shared/dynamic UB`，Tiling 仅设置 `localMemorySize = UB-128 KiB`；语义 workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档/生成边界→OpDef/InferShape→arch35 Tiling/POD/key→SIMT entry/body→Ascend 950 构建和 binary config 已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x=[x_0,...,x_{n-1}]`、`scalars=[s_0,...,s_{n-1}]`、`y=[y_0,...,y_{n-1}]`，则 `y_t[j]=x_t[j]-s_t`。950 Tiling 记录每个 `x_t` 的 storage-shape 元素数，SIMT body 只迭代该 count；因此即使公开接口允许 `y_t` 的 shape size 大于等于 `x_t`，额外区域也不由本 Kernel 写入。公共 InferShape 在图模式中把每个 `y_t` shape/dtype 设为对应 `x_t`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND，维数 0–8 | FP32、FP16、INT32、BF16 | 支持空 Tensor与非连续 Tensor；列表内 dtype 一致。OpDef 标记 `AutoContiguous`，生成 bridge 的具体临时张量与 workspace 公式固定仓不可见。 |
| `scalars` | ScalarList；元素数等于 `x` Tensor 数 | FP32（对应浮点 `x`）或 INT64（对应 INT32 `x`） | Device 看到按 `S` 连续排列的 GM 数组并读取 `scalars[t]`；ACL ScalarList 到该 GM 布局的生成 bridge 固定仓不可见。 |
| `out` | TensorList；每项 ND，维数 0–8 | 与 `x` 相同 | 列表内 dtype 一致，shape size 至少为对应 `x`；公开接口不支持非连续输出。Kernel 只写 `numel(x_t)` 个元素。 |
| `workspaceSize/executor` | Host 标量/句柄 | `uint64_t*`/`aclOpExecutor**` | 第一段返回；算子 Tiling 请求 0 B，框架连续化可能增加总 workspace，精确公式固定仓不可见。 |
| `workspace/stream` | Device 地址/stream | `void*`/`aclrtStream` | 第二段在指定 stream 执行 executor；调用方持有 workspace 至执行完成。 |

### 入口与版本

固定仓只公开一个 `aclnnForeachSubScalarList` 版本，没有更高版本同语义入口。文档产品表明确支持 Ascend 950PR/950DT，README、OpDef 的 `AddConfig("ascend950")`、Host/Kernel CMake 的 `ascend950→arch35` 选择和 `ascend950` binary JSON 相互一致。公开声明与参数校验文档可见；自动生成的 ACLNN C/C++ inner bridge、executor 构造和第二段 launch 包装未提交，按框架边界处理，不补造符号。

## Host/Launcher 执行流程

### 参数校验与前处理

公开文档给出的失败边界为：`x/scalars/out` 空指针返回 `ACLNN_ERR_PARAM_NULLPTR`；不支持 dtype 或 `x/out` dtype 不同返回 `ACLNN_ERR_PARAM_INVALID`；shape 不满足、TensorList 内 dtype 不一致或维数大于 8 可返回 `ACLNN_ERR_INNER_TILING_ERROR`。OpDef 进一步把 `x/scalars/y` 声明为 ND，并根据 `x` dtype 将 scalars 映射为 FP32 或 INT64。公共 InferShape 要求动态输入和输出个数一致，并逐项复制 shape/dtype。950 Tiling 自身检查平台信息、input-0 instance/desc、Tiling buffer、每个输入 shape 和 workspace 指针；生成 bridge 中各校验的具体顺序和 contiguous 临时对象不可见。

### 参数、Workspace 与生命周期

Tiling 从平台读取 AIV core 数 `C` 与 UB 字节数 `U`，从 input-0 instance 得到 `n`，逐项取 `e_t=numel(storageShape(x_t))` 并求 `E=sum(e_t)`。`blockDim = 1`（`E=0`），否则 `min(C, ceil(E/1024))`。它写 `workspace[0]=0`，并将 launch local memory 设为 `uint32_t(U-128 KiB)`；当前 body 没有对应的显式 local-buffer 分配。executor/contiguous 临时对象的 owner 与精确释放时点属于固定仓不可见的 ACLNN runtime；Device 的 x/y/scalars/POD 在 stream 中完成 Kernel 前必须保持有效。

### 启动图与失败传播

单节点 launch graph：

```text
aclnnForeachSubScalarListGetWorkspaceSize(...)
  -> 生成框架边界：校验/AutoContiguous/executor
  -> ForeachSubScalarListTilingFunc
     -> N1 foreach_sub_scalar_list<schMode><<<blockDim, 1024 threads/block>>>
aclnnForeachSubScalarList(workspace, workspaceSize, executor, stream)
  -> 在 stream 上提交 N1
```

N1 输入 `x/scalars/y` GM 对象、`ForeachSubScalarListTilingData` 和未使用的 workspace 参数，输出直接写 y；没有 Kernel 间 workspace edge。Tiling 任一空指针、平台资源、tensorNum、dtype 错误返回 `GRAPH_FAILED`，阻断 launch；第二段 runtime 错误由 ACLNN status 传播。stream 内提交顺序由 ACLNN runtime 保证，底层 launch/cleanup 实现固定仓不可见。

## Route 与参数设计

### 分核与资源计算

`E=sum_t e_t`，`B = E==0 ? 1 : min(AIVCoreNum, ceil(E/1024))`。Kernel grid 为 `B` 个 AIV block；每 block 的 `VF_CALL(Dim3(1024))` 产生 1024 个 SIMT thread。每线程全局 id `tid=blockIdx*1024+threadIdx`，全局步长 `stride=1024*B`；对每个 Tensor 独立执行 `j=tid; j<e_t; j+=stride`。因此分核不需要 former/tail POD，尾部由 `j<e_t` 保护。POD 上限为 256 Tensor；body 不分配 UB tile，`SetLocalMemorySize(U-128 KiB)` 只是 launcher 资源参数。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 | `schMode=0`；`Process<float,float>` | `dataType==DT_FLOAT` | 否则判断 FP16 |
| 2 | R-002 FP16→FP32 | `schMode=1`；`Process<half,float>` | `dataType==DT_FLOAT16` | 否则判断 INT32 |
| 3 | R-003 INT32→INT64 | `schMode=2`；`Process<int32_t,int64_t>` | `dataType==DT_INT32` | 否则判断 BF16 |
| 4 | R-004 BF16→FP32 | `schMode=3`；`Process<bfloat16_t,float>` | `dataType==DT_BF16` | 其它 dtype：`GRAPH_FAILED` |

```text
if DT_FLOAT: key=TPL(0) -> R-001
else if DT_FLOAT16: key=TPL(1) -> R-002
else if DT_INT32: key=TPL(2) -> R-003
else if DT_BF16: key=TPL(3) -> R-004
else: GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | `TPL(0)` | 950、x/y FP32、scalars FP32、`1<=n<=256` | 单 SIMT N1 | `Process<float,float>`→`OpForeachSubScalarListSimt<float,float>` | 直接 GM load/sub/store；寄存器 float | grid-stride 的 `j<count`；无 fallback |
| R-002 FP16→FP32 | `TPL(1)` | 950、x/y FP16、scalars FP32、`1<=n<=256` | 单 SIMT N1 | `Process<half,float>`；`ComputeType<half>::type=float` | x 升至 FP32、FP32 减、cast half 写回 | 同上 |
| R-003 INT32→INT64 | `TPL(2)` | 950、x/y INT32、scalars INT64、`1<=n<=256` | 单 SIMT N1 | `Process<int32_t,int64_t>`；compute INT64 | x 升至 INT64、INT64 减、cast INT32 写回 | 同上；窄化语义来自 C++ cast |
| R-004 BF16→FP32 | `TPL(3)` | 950、x/y BF16、scalars FP32、`1<=n<=256` | 单 SIMT N1 | `Process<bfloat16_t,float>`；compute FP32 | x 升至 FP32、FP32 减、cast BF16 写回 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `tensorNum<=256` | POD 的 `tensorElements[256]` 固定容量 | 全部 | `GRAPH_FAILED` | `ForeachSubScalarListTilingFunc::MAX_TENSOR_NUM` |
| 最小每核工作量 | `B=min(C,ceil(E/1024))`，空域 B=1 | 使总 SIMT thread 数约覆盖 E；不是 UB tile 条件 | 全部 | 无其它 Route | `MIN_PER_CORE`; `SetBlockDim` |
| 固定 threads/block | `THREAD_NUM=1024` | `LAUNCH_BOUND(1024)` 与 `VF_CALL(Dim3(1024))` 一致 | 全部 | 无 fallback | `OpForeachSubScalarListSimt`; `Process` |
| local memory 保留 | `localMemorySize=uint32_t(U-128KiB)` | 预留 DCache；body 不显式消费该区 | 全部 | 源码未检查 `U>=128KiB`，无 fallback | `DCACHE_SIZE`; `SetLocalMemorySize` |
| dtype | 四个 if/else 精确匹配 | router 只有四个模板实例 | 全部 | 其它 dtype `GRAPH_FAILED` | `ForeachSubScalarListTilingFunc`; entry `if constexpr` |
| 空 Tensor | `e_t==0` | 避免解析 Tensor 指针和标量计算 | 全部 | `continue` 到下一 Tensor | `OpForeachSubScalarListSimt` |

### POD/ABI

`ForeachSubScalarListTilingData` 是固定仓可见的普通标准布局字段序列：`sizeof=2064`、`alignof=8`；offset 分别为 0、4、8、16。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSubScalarListTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；总结构 2064 B/align 8 | AIV block 数 | `E==0?1:min(C,ceil(E/1024))` | body 不直接读取；block 数由 launch 使用 | `C>0`；有效但 Device 字段 inactive |
| `ForeachSubScalarListTilingData` | `tensorCount` | `int32_t`/32 | offset 4 | Tensor 数 | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT 外层 `t<tensorCount` | `1..256`；空列表在取得第0项 desc 时失败 |
| `ForeachSubScalarListTilingData` | `totalElements` | `int64_t`/64 | offset 8 | 元素 | `sum(storageShape.GetShapeSize())` | body 不直接读取；只参与 Host blockDim | 累加溢出未显式检查；Device 字段 inactive |
| `ForeachSubScalarListTilingData` | `tensorElements` | `int64_t[256]` | offset 16，2048 B | 元素/每 Tensor | 循环写 `tensorElements[i]=numel` | `elemCounts[t]` 控制 grid-stride 尾界 | 仅 `[0,tensorCount)` 活跃，其余由 memset 置 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | `TPL(0)` | `arch35/...tiling_arch35.cpp::ForeachSubScalarListTilingFunc` | `ForeachSubScalarListTilingData` | `arch35/foreach_sub_scalar_list.cpp::foreach_sub_scalar_list<0>` | `foreach_sub_scalar_list_simt.h::Process<float,float>/Op...` | Host/Kernel CMake + ascend950 binary JSON |
| R-002 FP16→FP32 | `TPL(1)` | 同上 | 同上 | `foreach_sub_scalar_list<1>` | `Process<half,float>`；`ComputeType<half>` | 同上 |
| R-003 INT32→INT64 | `TPL(2)` | 同上 | 同上 | `foreach_sub_scalar_list<2>` | `Process<int32_t,int64_t>`；`ComputeType<int32_t>` | 同上 |
| R-004 BF16→FP32 | `TPL(3)` | 同上 | 同上 | `foreach_sub_scalar_list<3>` | `Process<bfloat16_t,float>`；`ComputeType<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 共用单 Kernel graph。entry 用结构化 Tiling 宏注册/解析 POD，保留指向 GM Tiling 的指针，再按编译期 `schMode` 选择 `<T,S>`。`Process` 以 `offsetof(tensorElements)` 得到 count 数组地址，随后 `VF_CALL` 1024-thread SIMT body。每个线程遍历全部 Tensor；非空 Tensor 解析 x/y ListTensor 指针表、读取该位置 scalar，并以全 grid stride 处理互不重叠的元素。没有 queue/event/barrier/atomic，也没有跨节点 workspace。

### Route R-001：FP32 直接 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | `dataType==ge::DT_FLOAT` | `GET_TPL_TILING_KEY(0)` |
| Router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<schMode>` | `schMode==TILING_KEY_FLOAT` | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachSubScalarListTilingFunc -> N1 foreach_sub_scalar_list<0>
  -> NsForeachSubScalarList::Process<float,float>
    -> VF_CALL<OpForeachSubScalarListSimt<float,float>>(Dim3(1024), ...)
      -> SimtGetTensorAddr<float>(x/y,t) -> grid-stride load/sub/store
```

##### 源码等价伪码

```text
# 源码等价伪码
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0: continue
  xT=list_addr<float>(x,t); yT=list_addr<float>(y,t); s=scalars_float[t]
  tid=blockIdx*1024+threadIdx; stride=1024*blockNum
  for j=tid; j<count; j+=stride: yT[j]=float(xT[j])-float(s)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y ListTensor | `table[(base[0]>>3)+t]` | `e_t*sizeof(float)` | FP32 ND | runtime | SIMT thread | caller/runtime，Kernel 全程 | 每 j 唯一写；无 UB copy |
| GM scalars | `scalars+t` | `n*sizeof(float)` | FP32 dense | generated bridge | 每线程读取 `s_t` | runtime，Kernel 全程 | 同一 t 在各线程重复读取 |
| 寄存器 | 当前 thread 的 `xVal/sVal/result` | 3 个 compute 标量，精确寄存器字节固定仓不可见 | FP32 | thread load/cast | subtract/store | 单次 j 迭代 | 下一 j 覆盖 |

##### 分层内存布局

N/A：该 SIMT body 未声明 shared/dynamic UB、`TQue` 或 `TBuf`；数据从 GM 进入每线程标量寄存器后直接写 GM。Tiling 设置的 local-memory launch 上限没有仓内显式 consumer。

##### 字节账与复用顺序

```text
UB_total = 0 B（源码显式分配）
dynamic_ub = N/A；仅 localMemorySize=uint32_t(UB_platform-128 KiB)
workspace_total = 0 B（算子 Tiling）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread 标量寄存器 | 当前 j load/sub | 当前 j store | 下一 j | 单线程程序顺序 | 不跨线程共享；每个 j 只由一个 grid-stride thread 命中 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元素映射 | grid-stride | block/thread id | j loop | y 元素独占 | `j<count` 保护尾部；空 Tensor `continue` |
| 计算 | C++ float `-` | x/scalar load | y store | thread local | FP32 输入、计算、输出均 FP32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | key 0、blockDim、counts、workspace=0 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<0>` | 映射 `<float,float>` |
| Device body/helper | `op_kernel/arch35/foreach_sub_scalar_list_simt.h` | `Process/Op.../SimtGetTensorAddr<float>` | 1024-thread grid-stride FP32 subtract |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | 前一条件失败且 `dataType==DT_FLOAT16` | `TPL(1)` |
| Router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<1>` | `schMode==TILING_KEY_FLOAT16` | `Process<half,float>` |

#### Device 数据流

##### 函数级调用链

```text
N1 foreach_sub_scalar_list<1> -> Process<half,float>
  -> VF_CALL<OpForeachSubScalarListSimt<half,float>>(1024 threads)
    -> SimtGetTensorAddr<half> -> half load -> float subtract -> half store
```

##### 源码等价伪码

```text
# 源码等价伪码
for each nonempty tensor t, in every grid-stride thread:
  s=float(scalars_fp32[t])
  for j=tid; j<e_t; j+=1024*blockNum:
    result=float(x_half[t][j])-s
    y_half[t][j]=static_cast<half>(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | ListTensor table + `j` | `e_t*2` B | FP16 ND | runtime/thread | thread | Kernel 全程 | 直接 load/store |
| GM scalars | `t*4` B | `n*4` B | FP32 | bridge | thread | Kernel 全程 | 同一 t 重读 |
| registers | per-thread x/s/result | 固定仓不可见 | FP32 compute | cast/load | subtract/cast | 当前 j | 下一 j 覆盖 |

##### 分层内存布局

N/A：无显式 UB/shared buffer；升精度值保存在 thread-local compute 标量。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 temporaries | half/scalar cast | FP32 subtract与half cast | 下一 j | 单线程程序顺序 | 无跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 精度 | `ComputeType<half>::type=float` | half load | float subtract | thread temporary | 最后 `static_cast<half>`；具体舍入由编译器/硬件转换语义，源码未显式给 RoundMode |
| 尾部 | grid-stride condition | thread id | store | y 有效域 | `j<e_t`；空 Tensor跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | DT_FLOAT16→key 1 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<1>` | `<half,float>` |
| Device body/helper | `op_kernel/arch35/foreach_sub_scalar_list_simt.h` | `ComputeType<half>`；`Op...<half,float>` | FP32 compute、FP16 store |

### Route R-003：INT32 升位宽 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | 前两条件失败且 `dataType==DT_INT32` | `TPL(2)` |
| Router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<2>` | `schMode==TILING_KEY_INT32` | `Process<int32_t,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
N1 foreach_sub_scalar_list<2> -> Process<int32_t,int64_t>
  -> VF_CALL<Op...<int32_t,int64_t>>(1024 threads)
    -> int32 load -> int64 subtract with scalar int64 -> int32 store
```

##### 源码等价伪码

```text
# 源码等价伪码
s=int64(scalars_i64[t])
for j=tid; j<e_t; j+=stride:
  result=int64(x_i32[t][j])-s
  y_i32[t][j]=static_cast<int32_t>(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | ListTensor table + `j*4` | `e_t*4` B | INT32 ND | runtime/thread | thread | Kernel 全程 | 直接访问 |
| GM scalars | `t*8` B | `n*8` B | INT64 dense | bridge | thread | Kernel 全程 | 每 t 每线程读取 |
| registers | x/s/result | 固定仓不可见 | INT64 compute | cast/load | subtract/narrow | 当前 j | 下一 j 覆盖 |

##### 分层内存布局

N/A：无显式 UB/shared buffer；INT64 中间值在线程寄存器中。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread INT64 temporaries | cast/load | subtract与narrow store | 下一 j | 单线程程序顺序 | 每个输出唯一线程 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 位宽 | `ComputeType<int32_t>::type=int64_t` | int32/scalar load | int64 subtract | 中间结果 | 写回显式 `static_cast<int32_t>`；越界窄化结果由 C++/编译器目标语义决定，源码无饱和处理 |
| 尾部 | `j<count` | thread id | store | y 有效域 | 无越界 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | DT_INT32→key 2 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<2>` | `<int32_t,int64_t>` |
| Device body/helper | `op_kernel/arch35/foreach_sub_scalar_list_simt.h` | `ComputeType<int32_t>` | INT64 subtract后窄化 |

### Route R-004：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | 前三条件失败且 `dataType==DT_BF16` | `TPL(3)` |
| Router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<3>` | `schMode==TILING_KEY_BF16` | `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
N1 foreach_sub_scalar_list<3> -> Process<bfloat16_t,float>
  -> VF_CALL<Op...<bfloat16_t,float>>(1024 threads)
    -> BF16 load -> FP32 subtract -> BF16 store
```

##### 源码等价伪码

```text
# 源码等价伪码
s=float(scalars_fp32[t])
for j=tid; j<e_t; j+=stride:
  result=float(x_bf16[t][j])-s
  y_bf16[t][j]=static_cast<bfloat16_t>(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | ListTensor table + `j*2` | `e_t*2` B | BF16 ND | runtime/thread | thread | Kernel 全程 | direct GM |
| GM scalars | `t*4` B | `n*4` B | FP32 dense | bridge | thread | Kernel 全程 | repeated read |
| registers | x/s/result | 固定仓不可见 | FP32 compute | cast/load | subtract/cast | 当前 j | 下一 j 覆盖 |

##### 分层内存布局

N/A：无显式 AIV UB/shared buffer；BF16→FP32→BF16 全在线程标量中完成。

##### 字节账与复用顺序

```text
UB_total=0 B; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 temporaries | BF16/scalar cast | FP32 subtract与BF16 cast | 下一 j | 程序顺序 | 无共享/别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 精度 | `ComputeType<bfloat16_t>::type=float` | BF16 load | FP32 subtract | thread temp | 最后 `static_cast<bfloat16_t>`；源码未指定显式 RoundMode |
| 尾部 | grid-stride bound | ids | store | y 有效域 | `j<e_t`；空 Tensor跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | DT_BF16→key 3 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<3>` | `<bfloat16_t,float>` |
| Device body/helper | `op_kernel/arch35/foreach_sub_scalar_list_simt.h` | `ComputeType<bfloat16_t>` | FP32 subtract后 BF16 cast |

## 算子特定优化

### Trick T-001：跨 Tensor 复用同一 SIMT grid-stride 启动

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：`tensorCount<=256` 且 Tiling 成功；每个非空 Tensor 使用相同 `blockNum*1024` 线程网格。
- Golden 来源：[`foreach_sub_scalar_list_simt.h:56-79`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_simt.h#L56-L79)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t t = 0; t < tensorCount; t++) {
    int64_t count = tensorElements[t];
    if (count == 0) {
        continue;
    }
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);
    __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
    S scalarVal = scalarsGm[t];
    uint64_t tid = static_cast<uint64_t>(
        AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
        AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(
        AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        C xVal = static_cast<C>(xData[idx]);
        C sVal = static_cast<C>(scalarVal);
        C result = xVal - sVal;
        yData[idx] = static_cast<T>(result);
    }
}
```

#### 机制与设计原因

一个 Kernel launch 覆盖整个 TensorList，避免按 Tensor 启动独立 Kernel；每个 Tensor 内以全 grid stride 划分元素，`idx<count` 同时处理任意长度和尾部。标量在进入该 Tensor 的元素循环前读取，在线程内跨多个 stride 迭代复用。源码可证明这减少了 Tensor 级 launch 数和标量重复 load 的内层次数，但未提供性能百分比。

#### 迁移边界

- 适用：多个独立 Tensor 应用相同逐元素二元标量运算。
- 前提：列表指针 ABI 相同、每项输出无重叠、POD 能保存全部元素数、block/thread 上限支持 1024 threads。
- 失效：Tensor 数超过 256 时 Host 直接失败；需要跨 Tensor reduction/ordering 时不能复用该无同步方案。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：低位宽输入只在寄存器升精度，不物化 UB 中间张量

#### 触发与源码依据

- 适用 Route：R-002、R-003、R-004。
- 触发条件：`T` 为 half/int32/BF16，分别由 `ComputeType` 选 float/int64/float。
- Golden 来源：[`foreach_sub_scalar_list_simt.h:39-43`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_simt.h#L39-L43)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
struct ComputeType { using type = T; };
template <> struct ComputeType<half> { using type = float; };
template <> struct ComputeType<bfloat16_t> { using type = float; };
template <> struct ComputeType<int32_t> { using type = int64_t; };
```

#### 机制与设计原因

模板在编译期冻结计算类型，body 对单个元素执行 `T→C`、相减、`C→T`，不为整个 tile 建立 FP32/INT64 UB 副本。它同时改变精度顺序与内存行为：FP16/BF16 在 FP32 做减法，INT32 在 INT64 做减法，临时值生命周期只覆盖当前元素。

#### 迁移边界

- 适用：逐元素、无跨元素状态的低位宽运算。
- 前提：目标转换语义满足结果要求；输出仍需显式降回 T。
- 失效：需要可控 `RoundMode`、饱和转换或跨元素累积时，应换成显式 primitive/Route；当前源码没有此 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | DT_FLOAT 第一 | DT_FLOAT16 第二 | DT_INT32 第三 | DT_BF16 第四 | Host if/else 顺序 |
| 模板与 dtype | `<float,float>` | `<half,float>` | `<int32,int64>` | `<bf16,float>` | x dtype 与 scalar ABI |
| 循环与分核 | 1024-thread grid-stride，每 Tensor 重置 j | 同左 | 同左 | 同左 | `THREAD_NUM` 与 `blockDim` |
| Queue/TBuf/寄存器/dynamic UB/通信 window | FP32 registers；无 UB | FP32 registers；无 UB | INT64 registers；无 UB | FP32 registers；无 UB | `ComputeType` |
| 内存总占用与复用 | 显式 UB 0 | 显式 UB 0 | 显式 UB 0 | 显式 UB 0 | direct GM |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar load/store | 同左+casts | 同左+casts | 同左+casts | 无 Vector queue |
| 同步、精度与 tail | FP32；`j<count` | FP32算/FP16写 | INT64算/INT32写 | FP32算/BF16写 | 独立元素无需 barrier |
| fallback/失败边界 | 无 dtype fallback | 无 | 无 | 无 | 未知 dtype Host失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；另见无后缀卡 | 支持；另见无后缀卡 | 支持 |
| Host/Launcher | arch22 `ForeachCommonTiling` | 与 A2 共用 arch22 | arch35 专用 Tiling，256 Tensor、每 1024 元素估算 blockDim |
| Kernel/资源 | AIV Vector Queue/TBuf，workspace 请求 32 B | 同 A2 | SIMT direct-GM/register，算子 workspace 0 B |

## 限制与失败边界

- `tensorNum>256` 明确 `GRAPH_FAILED`；`totalElements` 的 int64 累加、`U-128 KiB` 无下溢检查，之后分别用于 blockDim 与 `uint32_t` localMemorySize。
- 空 Tensor受支持；空 TensorList会在 `GetDynamicInputDesc(0,0)` 后的空指针检查失败，不进入 Kernel。
- Kernel 以 `x` storage-shape count 控制写入，只写 y 的对应前缀；y 更大部分不写。950 Tiling 不自行重验 y shape/scalar 个数，相关 ACLNN 生成 bridge 逻辑固定仓不可见。
- `SimtGetTensorAddr` 依赖 ListTensor 首个 uint64 保存地址表字节偏移、其后表项保存实际 Tensor GM 指针；不得把 ACL Host 对象地址直接传入。
- FP16/BF16 的最终 cast 未指定显式 RoundMode；INT64→INT32 没有饱和处理。精确目标转换行为超出固定源码可见范围。
- 公开 ACLNN inner bridge、框架 AutoContiguous workspace 公式、executor cleanup 与底层 stream launch 实现固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sub_scalar_list/docs/aclnnForeachSubScalarList.md` | 两段式函数原型、产品/参数/错误表 | 公开 ACLNN 与 950 支持 |
| 入口声明/约束 | `foreach/foreach_sub_scalar_list/README.md` | 产品支持/公式 | 逻辑语义与平台复核 |
| 入口实现 | `foreach/foreach_sub_scalar_list/op_host/foreach_sub_scalar_list_def.cpp` | `ForeachSubScalarList` | dtype映射、AutoContiguous、ascend950 config |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_proto_utils.h` | `DtypeTensor2Scalar` | x dtype到scalar dtype映射 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`; `IMPL_OP_INFERSHAPE(ForeachSubScalarList)` | output count/shape/dtype |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar_list/op_host/arch35/foreach_sub_scalar_list_tiling_arch35.cpp` | `ForeachSubScalarListTilingFunc` | checks、counts、blockDim、workspace、key |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_tiling_data.h` | `ForeachSubScalarListTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_tiling_key.h` | enum/`ASCENDC_TPL_*` | 四个 key/模板值 |
| Kernel entry/body | `foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list.cpp` | `foreach_sub_scalar_list<schMode>` | key→`<T,S>` router |
| Kernel entry/body | `foreach/foreach_sub_scalar_list/op_kernel/arch35/foreach_sub_scalar_list_simt.h` | `Process`; `OpForeachSubScalarListSimt`; `SimtGetTensorAddr` | SIMT loop、地址、精度、tail |
| 构建接线 | `foreach/foreach_sub_scalar_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_sub_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 arch35 Kernel |
| 构建接线 | `foreach/foreach_sub_scalar_list/op_host/config/ascend950/foreach_sub_scalar_list_binary.json` | 四个 op_list 条目 | 950 dtype binaries |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B POD 可表达 256 Tensor counts | 字段顺序/offset/位宽，数组上限 256 | `ForeachSubScalarListTilingData` |
| 调用顺序 | entry→`Process`→`VF_CALL`→SIMT body | `Dim3(1024)` 与 `LAUNCH_BOUND(1024)` | `Process`; `OpForeachSubScalarListSimt` |
| workspace/内存 | 单 Kernel direct-GM，无语义 workspace | ListTensor 指针表协议、y 有效前缀、对象 lifetime | `SimtGetTensorAddr`; `workspace[0]=0` |
| 同步 | 独立 grid-stride 元素无需 barrier | thread id/stride 必须覆盖且不重叠 | `GetBlockIdx/GetThreadIdx/GetBlockNum` |
| 精度与尾块 | FP16/BF16→FP32、INT32→INT64 | cast顺序、`j<count`、空 Tensor continue | `ComputeType`; inner loop |
| 失败边界 | tensorNum/dtype/platform资源失败阻断 launch | 不得静默截断超过256的列表 | `ForeachSubScalarListTilingFunc` |
