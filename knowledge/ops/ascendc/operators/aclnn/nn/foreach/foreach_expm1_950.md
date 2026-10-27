---
type: Operator
title: "ForeachExpm1 950 ACLNN 算子知识"
description: "说明 aclnnForeachExpm1 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/docs/aclnnForeachExpm1.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_host/foreach_expm1_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_host/arch35/foreach_expm1_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/arch35/foreach_expm1.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_host/config/ascend950/foreach_expm1_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T15:32:07Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachExpm1 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachExpm1GetWorkspaceSize` + `aclnnForeachExpm1`，公开两段式 ACLNN；固定仓未见更高版本后缀。 |
| 计算语义 | 对 TensorList 中每个 Tensor 逐元素计算 `y_i[j]=expm1(x_i[j])`，等价于 `exp(x_i[j])-1`。 |
| 输入输出 | `x`/`out`：ND TensorList，FP16/FP32/BF16；rank 0–8；输入可非连续、输出必须连续。 |
| 执行类型 | 单 Kernel，节点 K1 为 SIMT VF；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP32 直接 `expm1f`；R-002 FP16→FP32→FP16；R-003 BF16→FP32→BF16。 |
| 核心资源 | GM TensorList 指针表、GM Tiling POD、每 block 1024 SIMT threads；Host 设置 local memory 为 `UB-128 KiB`，body 未声明 shared/dynamic UB 对象；算子节点 workspace slice 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN→自动生成 bridge 边界→OpDef/InferShape→arch35 Tiling/POD/key→templated entry→SIMT VF body→ascend950 构建与 binary config 已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设输入列表 `x=[x_0,...,x_{n-1}]`，对每个 Tensor 的线性有效元素 `j`，950 SIMT body 写 `y_i[j]=ComputeExpm1<T>(x_i[j])`。FP32 specialization 直接调用 `expm1f`；FP16/BF16 generic template 先 `static_cast<float>`，调用 `expm1f`，再 cast 回存储 dtype。每个 Tensor 独立，列表间没有归约或状态交接。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；ND；每 Tensor rank 0–8 | FP16、FP32、BF16 | 非空指针；支持空 Tensor；列表内 dtype 一致；输入支持非连续，OpDef `AutoContiguous()` 形成框架前处理边界。 |
| `out` | 对应 TensorList；ND；rank 0–8 | 与 `x` 相同 | 非空指针；输出必须连续；公开 API 允许对应 shape size≥输入，Device 只写输入元素数。 |
| `workspaceSize`/`executor` | 第一段输出 | `uint64_t*`/`aclOpExecutor**` | arch35 算子节点 workspace slice 为 0；公开 executor 总 workspace 仍可能包含输入 Contiguous 等框架节点，固定生成层不可见。 |
| `workspace`/`workspaceSize`/`stream` | 第二段输入 | opaque/`uint64_t`/stream | 调用者按第一段结果提供；K1 entry 参数保留 workspace 但不读取；stream 承载 launch。 |

### 入口与版本

实际公开入口为无后缀 `aclnnForeachExpm1GetWorkspaceSize`/`aclnnForeachExpm1`。README 明确列出 Ascend 950PR/950DT 支持，OpDef 添加 `ascend950` AICore config，Host CMake 映射 `ascend950→arch35`，Kernel CMake选择 `arch35/foreach_expm1.cpp`，binary config 含三种 dtype。因此 950 公开性与实现接线无冲突。自动生成 ACLNN bridge 不在固定仓，作为框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段 ACLNN 按公开文档检查空指针、FP16/FP32/BF16、输入输出及列表内部 dtype 一致、对应 shape size、rank≤8；错误为 `ACLNN_ERR_PARAM_NULLPTR`、`ACLNN_ERR_PARAM_INVALID` 或 `ACLNN_ERR_INNER_TILING_ERROR`。
2. OpDef 动态输入/输出均声明 ND 和 `AutoContiguous()`；输出非连续由接口约束排除，输入 contiguous 转换细节位于仓外生成 bridge。
3. `InferShape4ForeachCommon` 要求动态输入数等于输出数并逐项复制 shape；InferDataType 逐项复制 dtype。
4. `ForeachExpm1TilingFunc` 获取 AIV core 数和 UB；任一为 0、compute node/instance/desc/POD/workspace 指针为空、POD 清零失败或 dtype 非三者之一时返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

Host 从 input instance info 得到 `tensorNum`，逐个 storage shape 写 `tensorElements[i]` 并累加 `totalElements=E`。`E==0` 时 `needCoreNum=blockDim=1`；否则 `needCoreNum=min(AIV_core_num,ceil(E/1024))`。算子节点 `workspace[0]=0`，ACLNN executor 的框架 workspace 总量不可见。Host 另调用 `SetLocalMemorySize(uint32_t(ubSize-128 KiB))`；这是 K1 local-memory launch 配额，SIMT body 固定源码没有显式 shared/dynamic UB 分配或寻址。Tiling POD、TensorList 指针表和各 GM Tensor 必须存活到 stream 上 K1 完成。

### 启动图与失败传播

```text
aclnnForeachExpm1GetWorkspaceSize
  -> ForeachExpm1 executor（自动生成 bridge 边界）
  -> ForeachExpm1TilingFunc
       -> blockDim = (E==0 ? 1 : min(AIV_core_num, ceil(E/1024)))
       -> workspace = 0, localMemory = UB-128KiB, schMode by dtype
aclnnForeachExpm1(..., stream)
  -> K1: foreach_expm1<schMode><<<blockDim, stream>>>(xList,yList,null/0,tiling)
       -> NsForeachExpm1::Process<T>
          -> VF_CALL<OpForeachExpm1Simt<T>>(Dim3(1024), ...)
```

单节点图没有 workspace/GM 中间边。K1 直接从 x TensorList GM 读并写 y TensorList GM；同一 stream 提供 launch 顺序与完成可见性。Tiling 失败阻断 launch；schMode 是 binary specialization 的编译期模板参数，Host unsupported dtype 直接失败，不存在运行时 fallback。生成 bridge 的具体 launch 返回传播固定仓不可见。

## Route 与参数设计

### 分核与资源计算

令 `E=sum_i tensorElements[i]`。Host 仅用 E 决定 block 数 `B=(E==0?1:min(C,ceil(E/1024)))`，其中 C 为 AIV core 数；不对 TensorList 拼接域做连续分片。每个 VF block 启动 `THREAD_NUM=1024`：

```text
tid = blockIdx * threadNum + threadIdx
stride = threadNum * blockNum = 1024 * B
for every tensor t:
  for idx = tid; idx < tensorElements[t]; idx += stride:
    y[t][idx] = ComputeExpm1(x[t][idx])
```

因此每个线程依次访问所有 Tensor，但对每个 Tensor 使用相同的全局 grid-stride。Host 设置 `dynamic_local_bytes=uint32_t(UB-131072)`；body 无对应显式消费。算子节点 workspace 为 0；ACLNN executor 总 workspace 的框架部分固定仓不可见。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 direct | `schMode=0`；`Process<float>` | `dataType==DT_FLOAT`，Host `GET_TPL_TILING_KEY(FLOAT=0)` | 检查 FP16。 |
| 2 | R-002 FP16 promotion | `schMode=1`；`Process<half>` | `dataType==DT_FLOAT16`，Host key with mode 1 | 检查 BF16。 |
| 3 | R-003 BF16 promotion | `schMode=2`；`Process<bfloat16_t>` | `dataType==DT_BF16`，Host key with mode 2 | 其它 dtype→`GRAPH_FAILED`。 |

```text
FLOAT   -> schMode 0 -> foreach_expm1<0> -> Process<float>
FLOAT16 -> schMode 1 -> foreach_expm1<1> -> Process<half>
BF16    -> schMode 2 -> foreach_expm1<2> -> Process<bfloat16_t>
other   -> Host error, no launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 direct | mode 0 | FP32、ND/rank≤8、ascend950 | K1 SIMT VF，B blocks×1024 threads | `Process<float>`→`OpForeachExpm1Simt<float>`→float specialization | 仅 GM load/store 与寄存器标量；无 shared/atomic/barrier | `idx<count` 保护；空 Tensor continue；无 fallback |
| R-002 FP16 promotion | mode 1 | FP16；其余同上 | 同 K1 | generic `ComputeExpm1<half>` | 每线程 FP16 load→FP32 scalar→FP16 store | 同上 |
| R-003 BF16 promotion | mode 2 | BF16；其余同上 | 同 K1 | generic `ComputeExpm1<bfloat16_t>` | 每线程 BF16 load→FP32 scalar→BF16 store | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 最小每核元素 | `B=min(C,ceil(E/1024))` | 每 block 固定 1024 threads；避免为少于 1024 总元素继续增加 blocks | 全部 | E=0 强制 B=1 | `MIN_PER_CORE`/`ForeachExpm1TilingFunc` |
| thread launch bound | `THREAD_NUM=1024` | VF body声明 `LAUNCH_BOUND(1024)`，VF_CALL 同值 | 全部 | 无其它 thread 配置 Route | `OpForeachExpm1Simt`/`Process` |
| POD tensor数组 | `tensorElements[256]` | Host逐实例写静态数组，Device按 tensorCount 遍历 | 全部 | 固定源码未见 `tensorNum≤256` guard；超限不可依赖 | `ForeachExpm1TilingData`/`ForeachExpm1TilingFunc` |
| local memory扣减 | `local=uint32_t(ubSize-128KiB)` | 为 DCache 预留 128 KiB；body未显式使用剩余 local | 全部 | `ubSize==0`失败，但未单独检查 `ubSize<128KiB` | `DCACHE_SIZE`/`SetLocalMemorySize` |
| dtype specialization | 仅 FLOAT/FLOAT16/BF16 | entry `if constexpr` 只实例化三种 T | 全部 | Host `GRAPH_FAILED` | `ForeachExpm1TilingFunc`/`foreach_expm1<schMode>` |
| 空 Tensor | `count==0` | 避免解析/访问空 Tensor 的数据地址 | 全部 | continue 到下一 Tensor | `OpForeachExpm1Simt` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachExpm1TilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8 | blocks/cores | E==0→1，否则 cast `min(C,ceil(E/1024))` | 固定 Device body未读取；blockDim另由 context 设置 | int64→int32；B≤coreNum |
| `ForeachExpm1TilingData` | `tensorCount` | `int32_t`/32 | offset 4 | Tensor count | `static_cast<int32_t>(tensorNum)` | `Process`→VF arg→外层 Tensor loop | Host未检查 `tensorNum>INT32_MAX/256` |
| `ForeachExpm1TilingData` | `totalElements` | `int64_t`/64 | offset 8 | elements | 逐 Tensor numel 求和 | 固定 Device body未读取 | 求和溢出未显式检查 |
| `ForeachExpm1TilingData` | `tensorElements[256]` | `int64_t[256]` | offset 16；extent 2048 B；`sizeof=2064`，align 8 | elements/Tensor | `tiling->tensorElements[i]=storageShape.GetShapeSize()` | `elemCounts[t]` 决定每 Tensor grid-stride 终点 | 仅前 tensorCount 项活动；无 >256 guard |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | mode 0 | `arch35/foreach_expm1_tiling_arch35.cpp::ForeachExpm1TilingFunc` | `ForeachExpm1TilingData` | `arch35/foreach_expm1.cpp::foreach_expm1<0>` | `Process<float>`、`OpForeachExpm1Simt<float>`、float `ComputeExpm1` | ascend950 arch35 CMake + binary config |
| R-002 FP16 | mode 1 | 同上 | 同布局 | `foreach_expm1<1>` | generic `Process/Op.../ComputeExpm1<half>` | 同上 |
| R-003 BF16 | mode 2 | 同上 | 同布局 | `foreach_expm1<2>` | generic `Process/Op.../ComputeExpm1<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 以 compile-time `schMode` 映射 T，并把 GM tiling 转为 `ForeachExpm1TilingData*`。`Process<T>` 用 `offsetof(...,tensorElements)` 形成 GM 元素计数数组指针，读取 `tensorCount`，通过 `VF_CALL` 以 1024 threads 调用 SIMT body。body 对 TensorList 的地址协议为：把列表首地址视为 `uint64_t*`，首字是指针表相对列表首地址的 byte offset，右移 3 转为 64-bit 索引，再从表中取第 t 个 Tensor GM 指针。每个线程对每个 Tensor 执行全 grid-stride load/compute/store。

### Route R-001：FP32 直接 expm1f

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/foreach_expm1_tiling_arch35.cpp` | `ForeachExpm1TilingFunc` | `dataType==DT_FLOAT` | mode 0 |
| Template key | `op_kernel/arch35/foreach_expm1_tiling_key.h` | `FOREACH_EXPM1_TPL_SCH_MODE_FLOAT` | 常量 0 | 编译 `foreach_expm1<0>` |
| Router | `op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<schMode>` | `if constexpr(schMode==0)` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_expm1<0>(xList,yList,workspace,tiling)
  -> NsForeachExpm1::Process<float>
    -> VF_CALL<OpForeachExpm1Simt<float>>(Dim3(1024), tensorCount, elemCounts, x, y)
      -> SimtGetTensorAddr<float>(x/y,t)
      -> ComputeExpm1<float> -> expm1f
      -> yData[idx] store
```

##### 源码等价伪码

```text
globalTid = blockIdx*1024 + threadIdx
stride = 1024*blockNum
for t in [0,tensorCount):
  count = tensorElements[t]
  if count == 0: continue
  xData = TensorListAddress<float>(xList,t)
  yData = TensorListAddress<float>(yList,t)
  for idx=globalTid; idx<count; idx+=stride:
    yData[idx] = expm1f(xData[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x TensorList/GM | ptr table offset `(firstWord>>3)+t`; element `idx` | count[t] | FP32/ND linear | caller | SIMT thread | caller to stream completion | scalar GM load |
| per-thread scalar | register/compiler allocation不可见 | one xFloat/result | FP32 | load/expm1f | GM store | thread iteration | 无 shared alias |
| y TensorList/GM | 同 t 与 idx | count[t] written | FP32/ND linear | SIMT thread | caller | caller | scalar GM store |
| tiling GM | offset 4/16 | 2064 B struct | POD | Host | Process/VF | executor | read-only |
| local memory | launch quota | `uint32_t(UB-128KiB)` | opaque | Host | body无显式 consumer | K1 | N/A |

##### 分层内存布局

N/A：此 SIMT body 未声明 AIV `TQue/TBuf`、shared/dynamic UB 数组或通信 window；实际路径为 GM scalar load→线程局部标量→GM scalar store。Host 仍向 launch 设置 `UB-128 KiB` local-memory 配额，但固定 body 无显式寻址。

##### 字节账与复用顺序

```text
UB_total = N/A（无 AIV InitBuffer）
dynamic_ub_launch_quota = uint32_t(UB_platform - 131072)
explicit_dynamic_ub_consumed = 0 B（固定 body 无声明）
operator_node_workspace = 0 B；ACLNN executor 总 workspace 可能另含框架 slice
register_bytes = 固定编译产物不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程局部 `x/result` | 当前 idx load/expm1f | 当前 idx GM store | 下一 idx | 单线程程序顺序 | 各线程 idx 集合由 grid-stride 互斥，无 atomic |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread mapping | grid-stride | block/thread ids | loop guard | y element ownership | `idx<count` 保护 tail |
| compute/store | 单线程程序顺序 | `expm1f` | GM store | thread scalar | FP32；无 cast |
| inter-thread | N/A | 各线程 | N/A | disjoint idx | 无 barrier/atomic/cooperative group |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_expm1/op_host/arch35/foreach_expm1_tiling_arch35.cpp` | `ForeachExpm1TilingFunc` | B、POD、workspace/local memory、mode 0 |
| Kernel entry/router | `foreach_expm1/op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<0>` | mode→float |
| Device body/helper | `foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h` | `Process<float>/OpForeachExpm1Simt<float>/ComputeExpm1<float>` | VF launch、grid-stride、直接 expm1f |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/foreach_expm1_tiling_arch35.cpp` | `ForeachExpm1TilingFunc` | `dataType==DT_FLOAT16` | mode 1 |
| Template key | `op_kernel/arch35/foreach_expm1_tiling_key.h` | `...MODE_FLOAT16` | 1 | `foreach_expm1<1>` |
| Router | `op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<1>` | `if constexpr(schMode==1)` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_expm1<1> -> Process<half>
  -> VF_CALL<OpForeachExpm1Simt<half>>(1024 threads)
  -> grid-stride load half
  -> generic ComputeExpm1<half>: half->float -> expm1f -> float->half
  -> scalar GM store
```

##### 源码等价伪码

```text
使用与 R-001 相同的 tensor/grid-stride；对每个有效 idx：
  xFloat = static_cast<float>(xHalf[idx])
  resultFloat = expm1f(xFloat)
  yHalf[idx] = static_cast<half>(resultFloat)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | TensorList[t][idx] | count[t] | FP16/ND | caller/thread | thread/caller | caller | scalar 2 B load/store |
| thread scalars | compiler register allocation不可见 | one FP16 input/output + FP32 intermediate | scalar | cast/expm1f | cast/store | current iteration | loop-carried reuse |
| tiling/local/workspace | 同 R-001 | POD 2064 B；local quota UB-128KiB；workspace 0 | opaque | Host | body/POD | K1 | 无显式 shared |

##### 分层内存布局

N/A：无 AIV UB、shared/dynamic UB 数组；GM→线程标量 FP32→GM。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub_launch_quota = uint32_t(UB_platform-131072); explicit use = 0 B
operator_node_workspace = 0 B；ACLNN executor 总 workspace 可能另含框架 slice
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-thread FP32 scalar | FP16 cast-up/expm1f | cast-down | next idx | thread program order | 每 idx 独立 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promotion | C++ scalar cast | half load | `expm1f` | thread scalar | FP16→FP32 |
| demotion | C++ scalar cast | FP32 result | half GM store | thread scalar | round mode由 `static_cast<half>`/工具链定义，固定源码未另写模式 |
| tail/concurrency | `idx<count`；disjoint grid-stride | ids | loop | GM element | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_expm1/op_host/arch35/foreach_expm1_tiling_arch35.cpp` | dtype branch | FP16→mode1 |
| Kernel entry/router | `foreach_expm1/op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<1>` | `Process<half>` |
| Device body/helper | `foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h` | generic `ComputeExpm1<half>` | FP32 promotion、expm1f、demotion |

### Route R-003：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/foreach_expm1_tiling_arch35.cpp` | `ForeachExpm1TilingFunc` | `dataType==DT_BF16` | mode 2 |
| Template key | `op_kernel/arch35/foreach_expm1_tiling_key.h` | `...MODE_BF16` | 2 | `foreach_expm1<2>` |
| Router | `op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<2>` | `if constexpr(schMode==2)` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_expm1<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachExpm1Simt<bfloat16_t>>(1024 threads)
  -> grid-stride BF16 load
  -> generic ComputeExpm1: BF16->FP32 -> expm1f -> FP32->BF16
  -> scalar GM store
```

##### 源码等价伪码

```text
对每个 t 和线程拥有的 idx<count：
  xFloat = static_cast<float>(xBf16[idx])
  resultFloat = expm1f(xFloat)
  yBf16[idx] = static_cast<bfloat16_t>(resultFloat)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | TensorList[t][idx] | count[t] | BF16/ND | caller/thread | thread/caller | caller | scalar 2 B load/store |
| thread scalars | compiler allocation不可见 | BF16 + FP32 intermediate | scalar | cast/expm1f | cast/store | iteration | next idx reuse |
| tiling/local/workspace | 同 R-001 | POD 2064 B；local quota；workspace 0 | opaque | Host | Process | K1 | 无显式 shared |

##### 分层内存布局

N/A：无 AIV UB 或 shared/dynamic UB 对象；实际替代路径为 GM scalar load→FP32 线程局部计算→GM scalar store。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub_launch_quota = uint32_t(UB_platform-131072); explicit use = 0 B
operator_node_workspace = 0 B；ACLNN executor 总 workspace 可能另含框架 slice
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-thread FP32 scalar | BF16 cast-up/expm1f | cast-down | next idx | thread program order | 各 idx 独立 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promotion/compute | scalar cast + `expm1f` | BF16 load | FP32 result | thread scalar | BF16→FP32 |
| demotion | `static_cast<bfloat16_t>` | FP32 result | GM store | output scalar | 固定源码未指定独立 round mode |
| tail/concurrency | `idx<count`/grid-stride | thread IDs | loop | GM element | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_expm1/op_host/arch35/foreach_expm1_tiling_arch35.cpp` | BF16 branch | mode2 |
| Kernel entry/router | `foreach_expm1/op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<2>` | BF16 Process |
| Device body/helper | `foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h` | generic `ComputeExpm1<bfloat16_t>` | FP32 promotion 与 scalar store |

## 算子特定优化

### Trick T-001：所有 Tensor 复用同一全局 grid-stride，避免 per-Tensor launch

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：单 Kernel 收到 TensorList 指针表与 `tensorElements[]`；blockDim 由列表总元素数决定。
- Golden 来源：[`foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h:83-101`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h#L83-L101)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }

        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

        uint64_t tid = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
            AscendC::Simt::GetThreadIdx());
```

#### 机制与设计原因

Host 只启动一个 K1。每个线程对所有 Tensor 重用 `tid/stride`，并以 `idx<count` 保护各 Tensor 尾部；因此无需为不同 shape 的 Tensor 生成独立 launch 或 Host 侧 offset 拼接表。空 Tensor 在解析数据地址前跳过。各线程 idx 序列互斥，不需要 atomic 或 barrier。

#### 迁移边界

- 适用：TensorList 中各元素可独立计算、各 Tensor 地址可从统一指针表解析的逐元素算子。
- 前提：每个输出元素只有一个线程写；列表遍历成本可接受；POD 容量覆盖列表长度。
- 失效：跨 Tensor 归约、共享状态或单 Tensor 需不同 launch 配置时不能直接复用此图。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：用 `expm1f` 保留小 x 的专用数值路径并统一低精度升精度

#### 触发与源码依据

- 适用 Route：全部；R-002/R-003额外触发 FP32 promotion。
- 触发条件：SIMT `ComputeExpm1<T>`；float specialization 或 generic low-precision template。
- Golden 来源：[`foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h:50-70`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h#L50-L70)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__simt_callee__ inline T ComputeExpm1(T x)
{
    // Generic path: cast to float, compute expm1f, cast back
    float xFloat = static_cast<float>(x);
    float resultFloat = expm1f(xFloat);
    return static_cast<T>(resultFloat);
}

template <>
__simt_callee__ inline float ComputeExpm1<float>(float x)
{
    return expm1f(x);
}
```

#### 机制与设计原因

实现没有拆成普通 `expf(x)-1`，而是直接调用 `expm1f`；这保持专用 `expm1` 数值语义。FP16/BF16 在 FP32 中执行 transcendental，再降精度，避免在低精度上直接求指数与减一。每线程只保留标量中间值，不物化 GM/UB 临时 Tensor。

#### 迁移边界

- 适用：SIMT 逐元素 expm1，尤其低精度输入。
- 前提：平台 SIMT math 提供 `expm1f`；低精度 cast 语义满足接口精度要求。
- 失效：要求不同舍入模式或更高精度 transcendental 时需替换 compute specialization。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | mode0 | mode1 | mode2 | Host dtype 决策树 |
| 模板与 dtype | float specialization | generic half | generic bfloat16_t | 低精度统一 FP32 compute |
| 循环与分核 | B×1024，per-Tensor grid-stride | 同 | 同 | E只决定B |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf；线程标量；local quota未显式使用 | 同 | 同 | SIMT scalar path |
| 内存总占用与复用 | 算子节点 workspace 0；register bytes不可见 | 同 | 同 | 无物化中间 Tensor |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + `expm1f` | 另有 cast | 另有 cast | 全为 SIMT |
| 同步、精度与 tail | 无同步；FP32；idx guard | FP32 compute→FP16 | FP32 compute→BF16 | disjoint elements |
| fallback/失败边界 | Host unsupported失败 | 同 | 同 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 Tiling，32 B workspace，32 B block拼接分核 | 同 A2 | arch35 独立 Tiling，workspace 0，E/1024 选 blocks |
| Kernel/资源 | AIV Vector `TQue` 双缓冲；BF16 FP32 window | 同 A2 | SIMT VF 1024 threads；无显式 UB queue/shared |

## 限制与失败边界

- 950 POD `tensorElements` 固定 256 项，Host 未在写入前检查 TensorList 长度；`tensorCount` 还从 `uint64_t` 窄化为 `int32_t`。超过容量/位宽的行为不可依赖。
- `totalElements` 的 int64 累加和 `ubSize-128KiB` 后转 uint32 均没有显式溢出/下溢 guard；平台正常资源是隐含前提。
- 输出 shape size可大于输入，但 K1只写输入 count；多余区不由此算子定义/初始化。空 Tensor由 Device显式跳过。
- fixed entry读取了完整 `tilingData` 宏变量但实际 body仅消费 `tensorCount` 与 `tensorElements`；`needCoreNum/totalElements` 是 Host/launch 参数，不应在 Device 复刻用途。
- 自动生成 ACLNN inner bridge、实际寄存器数、SIMT math primitive 内部实现与 cast 具体硬件 round mode位于固定仓边界之外。本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_expm1/docs/aclnnForeachExpm1.md` | 两段式 API | 参数、错误、A2/A3/950 公开支持表 |
| 入口声明/约束 | `foreach/foreach_expm1/README.md` | 产品支持情况 | Ascend 950PR/950DT 公开支持与语义 |
| 入口实现 | `foreach/foreach_expm1/op_host/foreach_expm1_def.cpp` | `ForeachExpm1` | ascend950 OpDef、dtype/ND/AutoContiguous |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` | 动态列表 shape/dtype 传播 |
| Host/Tiling/Launcher | `foreach/foreach_expm1/op_host/arch35/foreach_expm1_tiling_arch35.cpp` | `ForeachExpm1TilingFunc` | POD、blockDim、workspace/local、mode |
| Host/Tiling/Launcher | `foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_tiling_data.h` | `ForeachExpm1TilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 模板集合 |
| Kernel entry/body | `foreach/foreach_expm1/op_kernel/arch35/foreach_expm1.cpp` | `foreach_expm1<schMode>` | compile-time router |
| Kernel entry/body | `foreach/foreach_expm1/op_kernel/arch35/foreach_expm1_simt.h` | `Process/OpForeachExpm1Simt/ComputeExpm1` | VF launch、地址、线程循环、精度 |
| 构建接线 | `foreach/foreach_expm1/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 |
| 构建接线 | `foreach/foreach_expm1/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 Kernel、AUTO_SYNC false |
| 构建接线 | `foreach/foreach_expm1/op_host/config/ascend950/foreach_expm1_binary.json` | 三个 `op_list` | FP16/FP32/BF16 binary |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `ForeachExpm1` | ascend950 活动全局 kernel config |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B plain struct，四字段 | offsets 0/4/8/16、tensorElements容量256 | `ForeachExpm1TilingData` |
| 调用顺序 | Tiling→templated K1→Process→VF_CALL | `schMode`与 T 一一对应；VF 1024 threads | `ForeachExpm1TilingFunc`/`foreach_expm1`/`Process` |
| workspace/内存 | 算子节点 workspace 0；无显式 UB buffer | ACLNN 总 workspace 可能含框架 slice；local quota扣128KiB；TensorList GM协议 | `SetLocalMemorySize`/`SimtGetTensorAddr` |
| 同步 | 单流 launch + 线程内程序顺序 | idx 必须由 grid-stride互斥；不得无依据加入 atomic/barrier | `OpForeachExpm1Simt` |
| 精度与尾块 | FP32 direct；FP16/BF16升精度；`idx<count` | 必须用 `expm1f` 与对应 cast顺序 | `ComputeExpm1` |
| 失败边界 | unsupported dtype/资源指针导致 Host失败 | 不得自造 Device fallback；Tensor数≤256是实现前提 | `ForeachExpm1TilingFunc` |