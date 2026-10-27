---
type: Operator
title: "ForeachSinh A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachSinh 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/docs/aclnnForeachSinh.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_host/foreach_sinh_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_host/arch22/foreach_sinh_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/foreach_sinh.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:20:50Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachSinh A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachSinhGetWorkspaceSize` → `aclnnForeachSinh`；固定仓未出现更高版本入口。 |
| 计算语义 | 对 TensorList 中每个输入张量逐元素计算 `y_i[j] = sinh(x_i[j]) = (exp(x_i[j]) - exp(-x_i[j])) / 2`。 |
| 输入输出 | `x`/`out` 均为 ND TensorList，单个 Tensor 0–8 维，列表内 dtype 一致；支持 FP16、FP32、BF16；输出连续且 dtype 与输入一致，输出元素容量不小于对应输入。 |
| 执行类型 | 单 Kernel、单个 `AIV_VECTOR` 节点；无 Cube/AIC 与 `MIX_CV` 可达节点。 |
| Kernel 路线 | R-001 FP16 直算；R-002 FP32 直算；R-003 BF16→FP32 分段计算→BF16。 |
| 核心资源 | TensorList GM 描述符、双缓冲 `VECIN`/`VECOUT` Queue；BF16 额外单缓冲 FP32 Queue；无算子 workspace 消费。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档 → 自动生成 ACLNN/OpDef 框架边界 → `Tiling4ForeachSinhTiling` → `ForeachCommonTiling` → `foreach_sinh` → `ForeachTriangle/KernelForeachUnary`。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入列表 `x=[x_0,...,x_{n-1}]`，输出列表 `y=[y_0,...,y_{n-1}]`。对每个列表位置 `i` 和输入有效元素 `j`，计算 `y_i[j]=sinh(x_i[j])`。Kernel 只遍历 `x_i` 的 storage element count；若 `out_i` 容量更大，多出的区域不在该 Kernel 的写域内。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 为 ND、0–8 维 | FLOAT16/FLOAT32/BFLOAT16 | 非空指针；列表内 dtype 一致；文档允许空 Tensor；OpDef 对输入启用 `AutoContiguous()`。 |
| `out` | TensorList；每个 Tensor 为 ND、0–8 维 | 与 `x` 相同 | 非空指针；不支持非连续输出；对应输出 storage element count 必须不少于输入。 |
| `workspaceSize` | 标量 | `uint64_t*` | 第一段接口输出；A2/A3 Host 固定请求 32 B，Kernel 将 `workspace` 忽略。 |
| `executor` | 框架对象 | `aclOpExecutor**` | 第一段接口生成，保存计算流程；内部结构固定仓不可见。 |
| `workspace`/`stream` | Device 地址/流 | — | 第二段接口使用；单 Kernel 按 executor 所在 stream 发射。 |

### 入口与版本

公开文档声明 `aclnnForeachSinhGetWorkspaceSize(x,out,workspaceSize,executor)` 与 `aclnnForeachSinh(workspace,workspaceSize,executor,stream)`。固定仓未保存自动生成的 ACLNN inner bridge 源码，故参数装箱、executor 内部布局和确切 launch 包装属于框架边界；`foreach_sinh_def.cpp` 的 `ForeachSinh` OpDef、arch22 Tiling 注册和 Kernel entry 已闭合该边界之后的实际实现。公开产品表明确支持 A2/A3，OpDef 的 `ascend910b`、`ascend910_93` 注册及构建计算单元与其一致。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 文档要求 `x/out` 非空，dtype 属于 FP16/FP32/BF16、相同位置输入输出 dtype 一致，TensorList 内 dtype 一致且维数不超过 8；失败分别落到参数或 tiling 错误。
2. OpDef 将动态输入与输出都声明为 ND，并启用 `AutoContiguous()`；固定仓未保存生成 bridge 中对非连续输入的具体物化顺序和临时空间切片。
3. `ForeachCommonTiling::Init(FOREACH_SINH_OP_CODE)` 最多读取 50 个动态输入实例，记录每个 storage element count、总元素数和首个输入 dtype；遇到空实例指针结束枚举。
4. `CheckOpInfo` 校验动态列表数量关系、每个输入维数 `<=8`、输出 dtype 与输入一致、输出 element count 不小于输入、列表内 dtype 一致。任一失败返回 `GRAPH_FAILED`，不启动 Kernel。

### 参数、Workspace 与生命周期

设 dtype 字节数 `s∈{2,4}`，`e=32/s` 个元素/32B block，输入总元素数 `N=Σ_i n_i`，平台 AIV 核数 `Cplat`。Host 计算 `C=min(Cplat,ceil(N/e))`，空域时将 `C` 提升为 1，最后再截断到 50。它按连续拼接的 TensorList 元素域分给各核，并将每核首尾 tensor index/offset 写入 POD。

A2/A3 Tiling 固定 `workspace[0]=32` B；Device entry 明确将 `userWS=nullptr`，因此这 32 B 不被算子 body 读取。workspace 由 ACLNN 调用方在第一段返回后分配，生命周期覆盖第二段执行；executor 与 stream 的具体 owner/销毁协议属于 ACLNN 框架边界。

### 启动图与失败传播

```text
aclnnForeachSinhGetWorkspaceSize
  -> [自动生成 ACLNN/OpDef 边界]
  -> Tiling4ForeachSinhTiling
     -> ForeachCommonTiling::Init(14)
     -> ForeachCommonTiling::RunBigKernelTiling
  -> 返回 workspaceSize=32 B 与 executor

aclnnForeachSinh(workspace, 32, executor, stream)
  -> node K0: foreach_sinh<<<blockDim=C, AIV>>>(xListDesc, yListDesc, workspace, tiling)
```

仅有 K0，无 Kernel 间边。输入/输出 TensorList 数据均由 ACLNN 调用方拥有；K0 在同一 stream 上读取 `x`、写 `out`。Host/Tiling 失败时 executor 不应进入 K0；固定仓未保存生成 bridge 的精确错误码转换。Kernel 对未知 key 没有 `else` 报错，只会不写输出，因此该状态必须由 Host key 闭包排除。

## Route 与参数设计

### 分核与资源计算

`needCoreNum=min(Cplat, max(1,ceil(N/e)))`，再限到 `MAX_CORE_CONT=50`。`AssignDataToEachCore` 先取 `B=ceil(N/e)` 个 32B block：每核基础元素数 `floor(B/C)*e`，前 `B mod C` 核再多 `e` 个元素；边界允许切在一个 Tensor 内，以 `(tensorStart,tensorStartOffset,tensorEnd,tensorEndOffset)` 表示闭区间。

令平台 UB 字节数为 `Uplat`，序列化 tiling POD 大小为 `Spod`。Sinh Host 预留 primitive 临时区 `E=calcPro*s*1024`：FP16/BF16 的 `calcPro=4`，FP32 为 1。FP16/FP32 取 `U=floor32((Uplat-Spod-E)/4)`；BF16 先除以 10，再除以 4，最后 64B 向下对齐，即 `U=floor64(floor((Uplat-Spod-E)/10)/4)`。`U` 即 POD 的 `inputsTensorUbSize`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 直算 | key 1；`ForeachTriangle<half,half,Sinh<half,false>>` | `dataType==DT_FLOAT16`，Host `GetTilingKeyByDtypeOnly` 返回 1 | 检查 key 2 |
| 2 | R-002 FP32 直算 | key 2；`ForeachTriangle<float,float,Sinh<float,false>>` | `dataType==DT_FLOAT`，Host 返回 2 | 检查 key 4 |
| 3 | R-003 BF16 升精度 | key 4；`ForeachTriangle<bfloat16_t,float,Sinh<float,false>>` | `dataType==DT_BF16` 且架构 guard 非 3003/3113；A2/A3 对应构建可达 | Host dtype 校验失败或 Kernel 无分支 |

```text
dtype == FLOAT16 -> key 1 -> R-001
else dtype == FLOAT32 -> key 2 -> R-002
else dtype == BFLOAT16 -> key 4 -> R-003
else -> Host key 0；公开/OpDef 校验应先失败
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1 | FP16、ND、每 Tensor `dim<=8`、输出容量足够、A2/A3 | 单 K0 AIV | `ForeachTriangle<half,half,Sinh<half,false>>` | 2×VECIN + 2×VECOUT；Sinh 前后 `PIPE_V` barrier | 完整 tile 用 `DataCopy`，末 tile 用 `DataCopyPad` |
| R-002 | 2 | FP32，其余同上 | 单 K0 AIV | `ForeachTriangle<float,float,Sinh<float,false>>` | 同 R-001，单 buffer 字节相同但元素数减半 | 同 R-001 |
| R-003 | 4 | BF16，其余同上 | 单 K0 AIV | `ForeachTriangle<bfloat16_t,float,Sinh<float,false>>` | 双缓冲各 9U，另有 2U FP32 Queue；Cast/Sinh/Cast 间 barrier | 外层 DMA tail + 内层 FP32 cast chunk tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor/核上限 | 动态输入枚举 `<50`，`blockDim<=50` | POD 五组数组均固定 50 项 | 全部 | 第 51 个输入不进入固定枚举；公开 API 上限未在文档给出 | `ForeachCommonTiling::Init/RunBigKernelTiling` |
| 32B 分核 | `e=32/s`，按 `ceil(N/e)` 分配 | 完整 DMA tile 与核边界优先 32B 对齐 | 全部 | 核尾仍由实际 tensor 边界与 tail copy 保护 | `GetNeedCoreNum/AssignDataToEachCore` |
| Sinh 临时区 | `Uplat > Spod+E`，其中 E 为 8KiB(FP16/BF16)或4KiB(FP32) | AscendC `Sinh` primitive 内部空间未显式暴露给 TPipe | 全部 | 源码无显式失败 guard，过小 UB 会发生无符号减法风险 | `DivideUbMemory4` |
| BF16 64B/10 倍折减 | `dataType==DT_BF16` | 显式 Queue 合计 38U，并为转换/primitive 留余量 | R-003 | 非 BF16 走 32B 对齐公式 | `DivideUbMemory4`、`KernelForeachUnary::Init` |
| BF16 编译 guard | `__CCE_AICORE__>=220` 且 NPU arch 非 3003/3113 | BF16 Cast 与 body 只在该 guard 下实例化 | R-003 | 不满足时 Kernel 无 key 4 body | `foreach_sinh`、`InnerComputer<bfloat16_t,...>` |
| 空域 | `N==0 -> blockDim=1` | 避免 0 block launch | 全部 | POD 默认区间为 0；固定源码没有显式空 TensorList descriptor guard | `GetNeedCoreNum` |

### POD/ABI

`ForeachCommonTilingData` 由 `BEGIN_TILING_DATA_DEF` 宏生成并序列化；固定仓未包含宏展开后的 C++ 对象布局，因此 `sizeof/alignof/offsetof` 不可见，不能用普通结构体规则替代。数组长度均为 50。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 宏展开 offset 不可见 | bytes/单 buffer 基准 U | `DivideUbMemory4` → `FillTilingData` | `KernelForeachBase::ParseTilingData`，推导 tile/cast 容量 | 32B 对齐；BF16 为 64B 对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 宏展开 offset 不可见 | elements | `Init` 逐 Tensor 写 storage size | `Process` 计算每 Tensor 结束 cursor | 未使用项为 0 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 宏展开 offset 不可见 | tensor index | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 仅 blockIdx `<blockDim<=50` 有效 |
| 同上 | `tensorEndList` | `uint16_t[50]` | 宏展开 offset 不可见 | tensor index | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 闭区间 |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 宏展开 offset 不可见 | elements | `AssignDataToEachCore` | `Process` 的首 Tensor cursor | 0 基 |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 宏展开 offset 不可见 | elements | `AssignDataToEachCore` | `Process` 的末 Tensor cursor | 闭区间；空域默认 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | `foreach_sinh_tiling_arch22.cpp::Tiling4ForeachSinhTiling`；`ForeachCommonTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_sinh.cpp::foreach_sinh` | `ForeachTriangle<half,...>` → `KernelForeachUnary` | `op_kernel/CMakeLists.txt::add_kernel_sources(ascend910b,ascend910_93)` |
| R-002 | key 2 | 同上 | 同上 | 同上 | `ForeachTriangle<float,...>` | 同上 |
| R-003 | key 4 | 同上 | 同上 | 同上 | `ForeachTriangle<bfloat16_t,float,...>`、BF16 `InnerComputer` | 同上 |

## Kernel 实现

### 公共执行流程

K0 解析 POD 后按 key 构造一个 `ForeachTriangle`。`KernelForeachBase` 用 `GetBlockIdx()` 选择本核首尾 Tensor/offset；`GetTensorAddr` 按 ListTensor 描述符首个 64 位 offset 定位指针表，再取第 `i` 个 GM 地址。每个核顺序遍历自己的 Tensor 区间，将单 Tensor 子区切成 `maxDataCount` tile，执行 `CopyIn → Compute → CopyOut`。Queue 的 EnQue/DeQue/FreeTensor 建立 DMA 与 Vector 阶段生命周期，`Sinh` 前后显式 `PipeBarrier<PIPE_V>()`。

### Route R-001：FP16 直算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_sinh/op_kernel/foreach_sinh.cpp` | `foreach_sinh` | `TILING_KEY_IS(1)` | `ForeachTriangle<half,half,Sinh<half,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sinh(x,y,workspace,tiling)
  -> ForeachTriangle<half,half,Sinh<half,false>>::Init
     -> KernelForeachBase<half>::ParseTilingData
     -> InitBuffer(dataQueue,2,U); InitBuffer(outQueue,2,U)
  -> KernelForeachUnary::Process
     -> SingleTensorProcess
        -> CopyIn -> ForeachTriangle::Compute -> Sinh<half,false> -> CopyOut
```

##### 源码等价伪码

```text
core = GetBlockIdx()
for tensor i in [start[core], end[core]]:
  begin = (i==start ? startOffset : 0)
  finish = (i==end ? endOffset : tensorElements[i]-1)
  n = finish-begin+1
  bind input/output GM at tensor_ptr(i)+begin
  for tile in ceil(n/(U/2)):
    valid = min(U/2, n-tile*(U/2))
    CopyIn(valid; tail uses DataCopyPad)
    PipeBarrier(V); Sinh<half,false>(outLocal,inLocal,valid); PipeBarrier(V)
    CopyOut(valid; tail uses DataCopyPad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `tensor_ptr(i)+begin+tile*(U/2)` | `valid` elements | FP16/ND flat | caller | CopyIn | caller；Kernel 只读 | tail 可非 32B |
| `dataQueue` | Queue slot | 2 slots × U B | FP16 | CopyIn | Sinh | TPipe；DeQue 后 Free | depth 2 |
| `outQueue` | Queue slot | 2 slots × U B | FP16 | Sinh | CopyOut | TPipe；CopyOut 后 Free | depth 2 |
| GM output | 同 input 相对元素 offset | `valid` elements | FP16/ND flat | CopyOut | caller | caller | 只写输入有效域 |
| workspace | 不绑定 | 32 B Host 请求 | opaque | ACLNN | 无 | caller | Kernel 强制 `userWS=nullptr` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
E = 4 * 2 * 1024 = 8192 B (Sinh primitive reserve)
U = floor32((Uplat - Spod - E) / 4)
UB_explicit = 2U + 2U = 4U
UB_explicit + Spod + E <= Uplat
dynamic_ub = N/A
workspace_total = 32 B requested, 0 B consumed by body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | CopyIn | Sinh | 后续 tile CopyIn | EnQue→DeQue→FreeTensor | Free 后才复用 |
| `outQueue` slot | Sinh | CopyOut | 后续 tile Compute | EnQue→DeQue→FreeTensor | CopyOut 完成后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue EnQue/DeQue | DataCopy | Compute | input slot | tail `blockCount=1, blockLen=valid*2 B` |
| 计算 | `PipeBarrier<PIPE_V>` | input ready | `Sinh` 前；输出入队前 | Vector pipeline | FP16 直接 `Sinh<half,false>`，无显式升精度 |
| UB→GM | Queue EnQue/DeQue | Compute | DataCopy | output slot | tail 精确写 `valid*2 B` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling/DivideUbMemory4` | key、blockDim、U 与 workspace |
| Kernel entry/router | `foreach_sinh/op_kernel/foreach_sinh.cpp` | `foreach_sinh`, key 1 | 映射 FP16 模板 |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_unary.h`、`foreach_triangle.h` | `SingleTensorProcess`、generic `InnerComputer` | tile DMA、Sinh、Queue 与 tail |

### Route R-002：FP32 直算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_sinh/op_kernel/foreach_sinh.cpp` | `foreach_sinh` | `TILING_KEY_IS(2)` | `ForeachTriangle<float,float,Sinh<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sinh -> ForeachTriangle<float,float,Sinh<float,false>>::Init/Process
  -> KernelForeachUnary::SingleTensorProcess
     -> CopyIn -> InnerComputer<float,float>::Compute
        -> Sinh<float,false>
     -> CopyOut
```

##### 源码等价伪码

```text
按 POD 的 core/tensor 闭区间遍历；tileCapacity=U/4。
每 tile 从 GM 搬 valid 个 float，PIPE_V barrier 后以 float Sinh 计算，
再次 barrier 后将 valid 个 float 写回同相对 offset；仅末 tile 使用字节型 DataCopyPad。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `tensor_ptr(i)+begin+tile*(U/4)` | `valid<=U/4` | FP32 flat | caller/CopyOut | CopyIn/caller | caller | tail 精确字节数 |
| `dataQueue` | Queue slot | 2U B | FP32 | CopyIn | Sinh | TPipe | depth 2 |
| `outQueue` | Queue slot | 2U B | FP32 | Sinh | CopyOut | TPipe | depth 2 |
| workspace | 不绑定 | 32 B requested | opaque | caller | 无 | caller | 不消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
E = 1 * 4 * 1024 = 4096 B
U = floor32((Uplat - Spod - E) / 4)
UB_explicit = 4U; UB_explicit + Spod + E <= Uplat
dynamic_ub = N/A; workspace_total = 32 B requested / 0 B body use
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 两个 Queue slot | CopyIn/Sinh | Sinh/CopyOut | 后续 tile | Queue EnQue/DeQue/FreeTensor | 生产消费闭合后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn/Out | Queue + DataCopy(Pad) | GM/UB | Compute/GM | Queue slot | tail `blockLen=valid*4 B` |
| Compute | `PipeBarrier<PIPE_V>` | DeQue input | Sinh/EnQue output | Vector result | 全程 FP32，无 cast/round |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory4` | FP32 `calcPro=1` |
| Kernel entry/router | `foreach_sinh/op_kernel/foreach_sinh.cpp` | key 2 branch | FP32 模板 |
| Device body/helper | `kernel_foreach_unary.h`、`foreach_triangle.h` | generic `InnerComputer<float,...>` | FP32 直算、tail 与同步 |

### Route R-003：BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `foreach_sinh/op_kernel/foreach_sinh.cpp` | key 4 branch | 编译 guard 非 arch 3003/3113 | BF16/FP32 模板 |
| Body | `foreach_utils/op_kernel/foreach_triangle.h` | `InnerComputer<bfloat16_t,float,...>` | `T=bfloat16_t,P=float` | 分段 Cast→Sinh→Cast |

#### Device 数据流

##### 函数级调用链

```text
foreach_sinh -> ForeachTriangle<bfloat16_t,float,Sinh<float,false>>::Init
  -> InitBuffer(dataQueue,2,9U); InitBuffer(outQueue,2,9U)
  -> InitBuffer(float32Queue,1,2U)
  -> Process -> SingleTensorProcess(maxDataCount=9U/2)
     -> CopyIn(BF16)
     -> InnerComputer::Compute
        -> repeated ComputePerCast(maxCastDataCount=U/4)
           -> Cast BF16→FP32 -> Sinh<float,false> -> Cast FP32→BF16(RINT)
     -> CopyOut(BF16)
```

##### 源码等价伪码

```text
for each core-owned tensor slice:
  for outerTile of at most 9U/2 BF16 elements:
    CopyIn outerTile
    for chunk of at most U/4 elements:
      f32[0:chunk] = Cast(bf16_chunk, CAST_NONE)
      f32[U/4 : U/4+chunk] = Sinh<float,false>(f32[0:chunk])
      bf16_out_chunk = Cast(f32[U/4:...], CAST_RINT)
    CopyOut outerTile
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | outer tile offset `tile*(9U/2)` | `valid<=9U/2` | BF16 flat | caller/CopyOut | CopyIn/caller | caller | outer tail 精确搬运 |
| `dataQueue` | slot | 2×9U B | BF16 | CopyIn | Cast | TPipe | depth 2 |
| `outQueue` | slot | 2×9U B | BF16 | final Cast | CopyOut | TPipe | depth 2 |
| `float32Queue` input half | byte 0 | U B=`U/4` floats | FP32 | BF16 Cast | Sinh | 单 slot，整个 Process 持有 | 与输出 half 不别名 |
| `float32Queue` output half | byte U | U B=`U/4` floats | FP32 | Sinh | BF16 Cast | 同上 | 每 chunk 覆盖 |
| workspace | 不绑定 | 32 B requested | opaque | caller | 无 | caller | 不消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["dataQueue · depth=2 · 18U B"]:2
  fIn["float32 input · U B"]:1
  fOut["float32 output · U B"]:1
  outQ["outQueue · depth=2 · 18U B"]:2
```

##### 字节账与复用顺序

```text
E = 4 * 2 * 1024 = 8192 B
U = floor64(floor((Uplat - Spod - E)/10)/4)
UB_explicit = 18U + 18U + 2U = 38U
Host 保守预算近似 40U，并另扣 Spod 与 E，因此 38U + Spod + E <= Uplat
dynamic_ub = N/A; workspace_total = 32 B requested / 0 B body use
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 input half | BF16 Cast | Sinh | 下一 cast chunk | Cast 后 `PIPE_V` barrier | Sinh 已消费 |
| FP32 output half | Sinh | BF16 Cast | 下一 Sinh chunk | Sinh/Cast 间及 Cast 后 barrier | 降精度完成后覆盖 |
| input/output Queue slot | DMA/Compute | Compute/DMA | 后续 outer tile | Queue FreeTensor | slot 已释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `Cast(...,CAST_NONE)` + `PIPE_V` | BF16 Queue | FP32 Sinh | FP32 input half | chunk tail=`dataCount % (U/4)` |
| 计算 | `Sinh<float,false>` + barrier | FP32 input | 降精度 Cast | FP32 output half | FP32 中间精度 |
| 降精度 | `Cast(...,CAST_RINT)` + barrier | FP32 result | CopyOut | BF16 output Queue | RINT 明确控制舍入 |
| DMA | DataCopy/DataCopyPad + Queue | GM/UB | 下一阶段 | outer tile | outer tail 精确 `valid*2 B` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory4` | BF16 10 倍折减与 64B 对齐 |
| Kernel entry/router | `foreach_sinh/op_kernel/foreach_sinh.cpp` | key 4 branch | BF16→FP32 模板与 guard |
| Device body/helper | `foreach_utils/op_kernel/foreach_triangle.h` | BF16 `InnerComputer::ComputePerCast` | FP32 双子区、Cast 顺序与 RINT |

## 算子特定优化

### Trick T-001：BF16 分段 FP32 双子区复用

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：`T=bfloat16_t,P=float`，且 BF16 编译 guard 成立。
- Golden 来源：[`foreach/foreach_utils/op_kernel/foreach_triangle.h:64-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h#L64-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    __aicore__ inline void ComputePerCast(
        const LocalTensor<bfloat16_t>& x1Local, const LocalTensor<bfloat16_t>& yLocal,
        LocalTensor<float>& float32Tensor, uint32_t maxCastDataCount, uint32_t index, int64_t dataCount)
    {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
        op(float32Tensor[offset], float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
```

#### 机制与设计原因

`paramsCount=2` 使输入和 `Sinh` 输出分别占 `float32Queue` 的两个 U-byte 子区，避免为每个 outer tile 物化新的 GM 临时张量。`maxCastDataCount=U/sizeof(float)` 将最大 `9U/2` 个 BF16 元素拆成可放入该 FP32 子区的 chunk；四个 Vector barrier 固定 Cast、Sinh、Cast 之间的覆盖顺序。源码可证明它以 FP32 计算并控制 BF16 回写舍入，但未提供性能百分比。

#### 迁移边界

- 适用：BF16 输入、支持 FP32 对应 math primitive、可用两个等长 FP32 子区的逐元素 unary。
- 前提：A2/A3 BF16 编译 guard、U 为 64B 对齐，降精度允许 `CAST_RINT`。
- 失效：FP16/FP32 使用各自直算 Route；不支持 BF16 guard 的平台不能复用 key 4 body。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在 `Sinh` 消费 FP32 输入前覆盖第一个子区，也不能在 BF16 Cast 消费结果前覆盖第二个子区；固定仓未提供除此同步违例外的可验证反例。

### Trick T-002：跨 Tensor 连续域按 32B block 分核

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003。
- 触发条件：TensorList 总元素数 `N>0`，`AssignDataToEachCore` 按 `elementsPerBlock=32/s` 分配。
- Golden 来源：[`foreach/foreach_utils/op_host/foreach_tiling_class.h:651-691`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L651-L691)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
        int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
        int64_t remainderCount = blockCount % needCoreNum;
```

#### 机制与设计原因

Host 把多个 Tensor 的元素视为连续总域进行负载分配，并用首尾 tensor index/offset 还原边界；这避免“一个小 Tensor 固定占一核”造成的并行度损失，同时让绝大多数核配额是 32B block 的整数倍。每个 Tensor 仍通过 ListTensor 指针表独立寻址，不要求 GM 实际连续。

#### 迁移边界

- 适用：独立逐元素 TensorList 运算。
- 前提：POD 能表达跨 Tensor 的首尾闭区间，Kernel 正确处理 tensor 边界 tail。
- 失效：Tensor 间存在依赖、归约或不同 dtype/layout 时不能直接使用同一连续域。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若只保存每核单一 GM 起点而不保存跨 Tensor 的 index/offset，无法穿越 ListTensor 的非连续指针表；固定仓未提供其它可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 4 | dtype 唯一决定 |
| 模板与 dtype | half→half Sinh | float→float Sinh | BF16→float Sinh→BF16 | BF16 采用 opmath 型升精度 |
| 循环与分核 | 跨 Tensor 32B block；tile U/2 | 同，tile U/4 | 同，outer tile 9U/2，inner U/4 | dtype 与 buffer 结构决定 tile |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4U Queue | 4U Queue | 38U Queue | 无 dynamic UB/通信 window |
| 内存总占用与复用 | 双 Queue slot | 双 Queue slot | 双 Queue + FP32 双子区 | BF16 子区跨 chunk 覆盖 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Sinh | 同 | DataCopy(Pad)+Cast+Sinh+Cast | 全部 AIV_VECTOR |
| 同步、精度与 tail | Sinh 前后 barrier；FP16 | 同；FP32 | 每 Cast/Sinh 间 barrier；RINT | tail 都按有效字节提交 |
| fallback/失败边界 | 无 Device fallback | 无 | 编译 guard | unknown key 无输出 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` | 公开支持；`ascend910_93` | 公开支持，见独立 `_950` 卡 |
| Host/Launcher | 共用 arch22 `ForeachCommonTiling` | 共用 arch22 `ForeachCommonTiling` | arch35 独立 SIMT Tiling |
| Kernel/资源 | 共用 AIV Vector Kernel | 共用 AIV Vector Kernel | SIMT、无显式 TQue/TBuf |

## 限制与失败边界

- A2/A3 公共 POD 只容纳 50 个 Tensor 和 50 个核；公开 ACLNN 文档没有给出 TensorList 长度上限，而固定实现枚举在 50 停止，这是迁移时必须保留并补充上层约束的边界。
- `DivideUbMemory4` 对 `Uplat-Spod-E` 没有显式下溢检查；平台 UB 必须足够容纳 POD、Sinh primitive 预留和显式 Queue。
- 输出容量可以大于输入，但 Kernel 仅写与输入相同的 element count；多出区域保持调用前状态，源码未定义其值。
- 自动生成 ACLNN bridge、executor 内部结构、`Sinh` primitive 内部 scratch 细节和 tiling 宏展开 ABI 在固定仓不可见；正文只把它们作为明确框架/primitive 边界，不推测内部实现。
- empty Tensor 支持由公开文档声明；空 TensorList 的 descriptor 安全性没有在 Kernel 中显式闭合，因此不把“列表长度可为 0”扩展为已证明约束。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sinh/docs/aclnnForeachSinh.md` | 两段式原型、产品表 | ACLNN、dtype/shape/连续性与 A2/A3 支持 |
| 入口实现 | `foreach/foreach_sinh/op_host/foreach_sinh_def.cpp` | `ops::ForeachSinh` | 动态 ND OpDef、AutoContiguous、平台配置 |
| Host/Tiling/Launcher | `foreach/foreach_sinh/op_host/arch22/foreach_sinh_tiling_arch22.cpp` | `Tiling4ForeachSinhTiling` | ForeachSinh 到公共 Tiling 接线 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace、POD producer |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD 字段与 ForeachSinh 注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype 到 key 1/2/4 的映射 |
| Kernel entry/body | `foreach/foreach_sinh/op_kernel/foreach_sinh.cpp` | `foreach_sinh` | key 到三种模板 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD consumer、ListTensor 地址解析 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、tile、DMA、tail |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | `ForeachTriangle/InnerComputer` | Sinh 与 BF16 升降精度 |
| 构建接线 | `foreach/foreach_sinh/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 arch22 Kernel 计算单元 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段公共 Foreach POD | 数组长度 50、字段位宽、闭区间 offset 单位 | `ForeachCommonTilingData` |
| 调用顺序 | Init→逐 Tensor→逐 tile→CopyIn/Compute/CopyOut | BF16 inner chunk 顺序 | `KernelForeachUnary::Process` |
| workspace/内存 | workspace body 不消费；UB 由 U 驱动 | depth 必须乘 2；BF16 为 18U+18U+2U | `InitBuffer` calls |
| 同步 | Queue 生命周期与 PIPE_V barrier | FP32 双子区消费前不得覆盖 | `InnerComputer::ComputePerCast` |
| 精度与尾块 | FP16/FP32 直算；BF16 FP32+RINT | tail 按有效字节搬运 | `CopyIn/CopyOut/InnerComputer` |
| 失败边界 | Host 校验阻断 launch | unknown key、UB 下溢不得当作 fallback | `CheckOpInfo`、entry branches |