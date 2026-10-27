---
type: Operator
title: "ForeachMinimumScalarList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachMinimumScalarList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/docs/aclnnForeachMinimumScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/foreach_minimum_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/arch22/foreach_minimum_scalar_list_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/foreach_minimum_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/config/ascend910b/foreach_minimum_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/config/ascend910_93/foreach_minimum_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_minimum_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:07:57Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachMinimumScalarList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMinimumScalarListGetWorkspaceSize` + `aclnnForeachMinimumScalarList`，公开 ACLNN 两段式接口；固定仓未出现带版本后缀的新入口。 |
| 计算语义 | 对第 `i` 个 Tensor 的每个有效元素 `j` 计算 `y_i[j] = min(x_i[j], scalars[i])`。 |
| 输入输出 | `x`/`y` 为同长度 ND TensorList；`x,y` 支持 FP16/FP32/INT32/BF16，内部 scalar Tensor 对应为 FP32/FP32/INT64/FP32。 |
| 执行类型 | 单节点 `AIV_VECTOR`：`foreach_minimum_scalar_list`；无 Cube/AIC 计算或 `MIX_CV` 节点。 |
| Kernel 路线 | R-001 FP16 原生 Mins；R-002 FP32 原生 Mins；R-003 INT32 原生 Mins；R-004 BF16 分段 FP32 Mins。 |
| 核心资源 | TensorList GM 指针表、scalar GM、输入/输出 `TQue` 双缓冲；BF16 另有单深度 FP32 `TQue`；算子 Kernel 不消费 user workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档/ACLNN 构建接线→OpDef/公共 infer→arch22 Tiling/POD→dtype key router→公共 Vector body、UB 与 tail 均闭合；自动生成 ACLNN bridge 属框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`，`scalars=[s_0,...,s_{n-1}]`。对每个 `0<=i<n` 及 `0<=j<numel(x_i)`，输出有效域为 `y_i[j]=min(x_i[j],s_i)`。Host 公共 infer 将每个 `y_i` 的 shape 和 dtype 设为对应 `x_i`；ACLNN 文档允许调用者提供 shape-size 大于等于输入的输出，但 Kernel 只写输入元素数对应的前缀，额外区域不在算子有效输出域。空 Tensor 合法且不触发搬运/计算。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 为 ND，0～8 维 | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；允许空 Tensor；OpDef `AutoContiguous()`，公开接口允许非连续输入。 |
| `scalars` | ScalarList；内部物化为长度 `n` 的 ND Tensor | FP32 或 DOUBLE（公开 ACLNN）；binary 为 FP32/INT64 | 元素数必须等于 `x` Tensor 数；浮点/BF16 的内部 binary scalar 为 FP32，INT32 的 scalar 为 INT64。DOUBLE 到内部 FP32 的生成 bridge 转换细节固定仓不可见。 |
| `out` | TensorList；ND，0～8 维 | 与 `x` 相同 | Tensor 数与 `x` 相同；公开接口不支持非连续输出；每项 shape-size 不小于输入，公共 infer 默认复制输入 shape。 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段返回执行所需空间与执行器；空指针或校验失败返回 ACLNN 错误。 |
| `workspace/stream` | Device workspace 与执行流 | 指针/句柄 | 第二段使用第一段结果异步提交；生命周期至少覆盖该次 stream 执行。 |

### 入口与版本

公开文档声明唯一两段式 `aclnnForeachMinimumScalarList` 入口，产品表明确支持 Atlas A2、Atlas A3；OpDef 和 binary/CMake 分别接入 `ascend910b` 与 `ascend910_93`。`ACLNNTYPE aclnn` 由构建框架生成声明/inner bridge，固定仓不含生成后的 `aclnn_foreach_minimum_scalar_list.h` 与 C 实现；该生成层作为明确框架边界，算子拥有的 OpDef、Tiling、POD 和 Kernel 已继续闭合。固定仓未出现更高版本后缀入口。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段检查 `x/scalars/out/workspaceSize/executor` 等必需对象；公开错误表覆盖空指针、dtype 不支持/不一致、shape 或维数错误。
2. OpDef 为动态 `x/y` 与 ScalarList `scalars` 建立 ND、dtype 和 `AutoContiguous()` 约束；生成 bridge 如何把 ACL ScalarList（含 DOUBLE）物化为内部 scalar Tensor 固定仓不可见。
3. `InferShape4ForeachCommon` 要求动态输入数等于输出数，并逐项复制 `x_i` shape；`InferDataType4ForeachCommon` 逐项复制 dtype。
4. `ForeachCommonTiling::Init` 最多扫描 50 个 `x_i`，记录每项 storage numel、总元素数、首个 dtype 和 32-byte block 元素数，并拒绝列表内 dtype 不一致。
5. `CheckOpInfo` 依次检查输入列表数量/shape、ScalarList 长度与维数、输出 dtype/容量、列表内 dtype；任何失败返回 `GRAPH_FAILED`，阻断 launch。

### 参数、Workspace 与生命周期

设 `P=ForeachCommonTilingData::GetDataSize()`，`B=32/sizeof(T)`（元素/32-byte block），`N=sum_i numel(x_i)`。Tiling 生成 `key`、`needCoreNum=min(AIVCoreNum,ceil(N/B),50)`（`N=0` 时为 1）、每核跨 Tensor 的闭区间、UB 单槽字节数 `U` 和 `blockDim`。Tiling 节点请求 32 bytes workspace，但 Kernel entry 显式令 `userWS=nullptr` 且 body 不读取 `workspace`；ACLNN executor 可能因 `AutoContiguous` 增加框架 workspace，其精确合并公式固定仓不可见。caller 拥有 workspace，executor 保存计算流程，第二段在传入 stream 上提交。

### 启动图与失败传播

单 Kernel 图：

```text
aclnnForeachMinimumScalarListGetWorkspaceSize
  -> 生成 bridge / ForeachMinimumScalarList OpDef（框架边界）
  -> Tiling4ForeachMinimumScalarListTiling
     -> ForeachCommonTiling::Init
     -> ForeachCommonTiling::RunBigKernelTiling
  -> executor + workspaceSize
aclnnForeachMinimumScalarList(workspace, workspaceSize, executor, stream)
  -> launch foreach_minimum_scalar_list<<<blockDim, stream>>>
     -> 单个 AIV_VECTOR Device 节点
```

节点输入为 `x` TensorList 指针表、内部 scalar GM、`y` TensorList 指针表与 POD；节点输出直接写 `y`，无 Kernel 间边。Tiling/launch 失败直接返回并阻断节点；单 stream 提交保证输入、可能的 Contiguous 前处理、Kernel 与可能的框架后处理顺序。生成 executor 内部的具体任务列表和临时切片 offset 固定仓不可见。

## Route 与参数设计

### 分核与资源计算

Host 将所有 Tensor 的有效元素逻辑串接后按 32-byte block 分核。`blockCount=ceil(N/B)`；普通情况每核先取 `floor(blockCount/needCoreNum)*B` 个元素，前 `blockCount%needCoreNum` 核再多取一个 `B`。`AssignDataToEachCore` 将每核区间投影为 `tensorStart/End` 和 inclusive `tensorStartOffset/tensorEndOffset`，所以单核可跨多个 Tensor。非 BF16：`U=floor(((UB-P)/4)/32)*32` bytes；BF16：`U=floor((((UB-P)/10)/4)/64)*64` bytes。无显式 `UB>P` guard，依赖已注册平台资源成立。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 原生 Mins | key 1；`<half,half,MinsAdapter<half>,2,1>` | `dataType==DT_FLOAT16` | dtype selector 下一分支 |
| 2 | R-002 FP32 原生 Mins | key 2；`<float,float,MinsAdapter<float>,2,1>` | `dataType==DT_FLOAT` | dtype selector 下一分支 |
| 3 | R-003 INT32 原生 Mins | key 3；`<int,int,MinsAdapter<int>,2,1>` | `dataType==DT_INT32`，内部 scalar binary 为 INT64，逐 Tensor 取值时转 INT32 | dtype selector 下一分支 |
| 4 | R-004 BF16 分段 FP32 Mins | key 4；`<bfloat16_t,float,MinsAdapter<float>,2,1>` | `dataType==DT_BF16` 且编译 guard `__CCE_AICORE__>=220` 且非 NPU arch 3003/3113 | Host 其他 dtype 得 key 0；router 无分支、不写输出 |

```text
if dtype == FP16: key = 1 -> R-001
else if dtype == FP32: key = 2 -> R-002
else if dtype == INT32: key = 3 -> R-003
else if dtype == BF16: key = 4 -> R-004
else: key = 0 -> 无可达 Kernel body（接口/OpDef 应先拒绝）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1 | FP16，列表约束通过，A2/A3 | 单 AIV Kernel | `ForeachOneScalarListBinary<half,half,...>`→generic `InnerComputer` | 2×inputQ + 2×outputQ；Queue + `PIPE_V` | 尾 tile 用 byte-count `DataCopyPad` |
| R-002 | 2 | FP32，列表约束通过，A2/A3 | 单 AIV Kernel | `ForeachOneScalarListBinary<float,float,...>`→generic `InnerComputer` | 同 R-001，容量按 bytes | 同 R-001 |
| R-003 | 3 | INT32/INT64 scalar，列表约束通过，A2/A3 | 单 AIV Kernel | `ForeachOneScalarListBinary<int,int,...>`→generic `InnerComputer` | 同 R-001；scalar 先窄化为 INT32 | 同 R-001；越界 scalar 无 Host range guard |
| R-004 | 4 | BF16/FP32 scalar，A2/A3，BF16 编译 guard 成立 | 单 AIV Kernel | BF16 specialized `InnerComputer` | 2×9U inputQ + 2×9U outputQ + 1×2U FP32Q；多次 cast + barrier | 外层 DMA tail + 内层 cast remainder |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `i<MAX_TENSOR_CONT(50)`；Scalar 长度须等于记录数 | POD 每项数组固定 50 | 全部 | 超长列表通常在 scalar 长度/输出检查失败；无独立“>50”错误分支 | `ForeachCommonTiling::Init/CheckScalarsTenorShapeInfo` |
| 32-byte 分核 | `B=32/sizeof(T)` | full-tile `DataCopy` 与核边界按 block 对齐 | 全部 | tail 只在 Tensor 边界/tile 尾由 Pad API 处理 | `GetNeedCoreNum/AssignDataToEachCore` |
| UB 四分 | `U=floor(((UB-P)/4)/32)*32` | 两个 depth=2 Queue 共需 `4U` | R-001～003 | 无 fallback；平台 UB 不满足时存在无符号下溢风险 | `DivideUbMemory3` |
| BF16 保守 UB | `U=floor((((UB-P)/10)/4)/64)*64` | 实际 Queue 为 `38U`，并为 BF16→FP32 分段留余量 | R-004 | 无非升精度 fallback | `DivideUbMemory3`、`KernelForeachUnary::Init` |
| BF16 编译门 | `__CCE_AICORE__>=220 && arch!=3003/3113` | BF16 body 和 FP32 Queue specialization 仅在该 guard 内 | R-004 | router 不执行 key 4 body | `foreach_minimum_scalar_list`、`InnerComputer<bfloat16_t,...>` |
| 空域 | `N==0 -> blockDim=1` | 保留合法 launch；每 Tensor `dataCount=0` 时 tile 循环为 0 | 全部 | 成功但不写输出 | `GetNeedCoreNum/SingleTensorProcess` |

### POD/ABI

`ForeachCommonTilingData` 由 tiling 宏生成；固定仓能确认序列化字段顺序、源码类型与数组长度，但宏定义来自外部 CANN 头，因此实际 `sizeof/alignof/offsetof` 固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 第 1 字段；精确 offset 不可见 | bytes/Queue 单槽基准 `U` | `DivideUbMemory3`→`FillTilingData` | `KernelForeachBase::ParseTilingData`，推导 Queue 容量与 tile | 必须按 32 bytes（BF16 为 64 bytes）向下对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第 2 字段；精确 offset 不可见 | elements/Tensor | `Init` 逐项写 storage numel | `Process` 计算每个 Tensor 默认末端 | 未活动项为 Host 零初始化数组 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 第 3 字段；精确 offset 不可见 | Tensor index/core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 活动 core 必须 `<50` |
| 同上 | `tensorEndList` | `uint16_t[50]` | 第 4 字段；精确 offset 不可见 | inclusive Tensor index/core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | inclusive |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 第 5 字段；精确 offset 不可见 | elements | `AssignDataToEachCore` | `Process` 绑定 GM base | inclusive start |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 第 6 字段；精确 offset 不可见 | elements | `AssignDataToEachCore` | `Process` 计算 `dataCount` | inclusive end；空 Tensor 可为 -1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | arch22 `Tiling4...`→`RunBigKernelTiling` | `ForeachCommonTilingData` | entry key-1 branch | scalar-list binary→unary/base→generic InnerComputer→Mins | A2/A3 CMake + binary JSON FP16 |
| R-002 | key 2 | 同上 | 同上 | entry key-2 branch | 同上，FP32 实例 | A2/A3 CMake + binary JSON FP32 |
| R-003 | key 3 | 同上 | 同上 | entry key-3 branch | 同上，INT32 实例 | A2/A3 CMake + binary JSON INT32/INT64 |
| R-004 | key 4 | 同上 | 同上 | entry guarded key-4 branch | scalar-list binary→BF16 specialized InnerComputer→Mins FP32 | A2/A3 CMake + binary JSON BF16/FP32 |

## Kernel 实现

### 公共执行流程

entry 解析同一 POD，按 key 构造一个 CRTP `ForeachOneScalarListBinary` 并调用 `Init/Process`。`GetTensorAddr` 读取 TensorList 首个 `uint64_t` 偏移，再从指针表取第 `i` 个数据地址。每个 AIV core 按 POD 的 inclusive Tensor/offset 区间遍历；每个 Tensor 片段再按 `maxDataCount` 切 tile，执行 `CopyIn→Compute→CopyOut`。满 tile 使用 `DataCopy`，尾 tile 使用单 block、精确 byte `blockLen` 的 `DataCopyPad`。scalar 在进入每个 Tensor 时读取一次并复用其全部 tile。无 Kernel 间 workspace 边。

### Route R-001：FP16 原生 Mins

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `op_kernel/foreach_minimum_scalar_list.cpp` | `foreach_minimum_scalar_list` | `TILING_KEY_IS(1)` | `<half,half,MinsAdapter<half>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_minimum_scalar_list(inputs, scalar, outputs, workspace, tiling)
  -> ForeachOneScalarListBinary<half,half,...>::Init
     -> KernelForeachUnary::Init -> KernelForeachBase::Init/ParseTilingData
  -> KernelForeachUnary::Process
     -> ProcessPlusInLoop（读取 scalars[i] 并转 half）
     -> SingleTensorProcess
        -> CopyIn -> ForeachOneScalarListBinary::Compute
           -> InnerComputer<half,half>::Compute -> MinsAdapter<half> -> Mins
        -> CopyOut
```

##### 源码等价伪码

```text
core = GetBlockIdx(); 载入该 core 的 tensorStart/end 与 inclusive offsets
for tensor i in [start,end]:
  begin = (i==start ? startOffset : 0)
  end   = (i==end ? endOffset : numel[i]-1)
  scalar16 = half(scalars_fp32[i])
  for tile k over count=end-begin+1, capacity=U/2 elements:
    CopyIn x_i[begin+k*capacity : +valid]
    PipeBarrier(V); Mins(outUB, inUB, scalar16, int32(valid)); PipeBarrier(V)
    CopyOut valid elements to y_i；尾 tile 使用精确 byte DataCopyPad
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | pointer-table `x[i]+begin+k*U/2` | `valid` elements | FP16/ND flatten | caller/framework | CopyIn | caller；launch 全程 | full tile 32-byte aligned，tail pad |
| scalar GM | `scalars+i` | 1 FP32/Tensor | FP32 vector | bridge | ProcessPlusInLoop | executor/input | 每 Tensor 只读一次后转 FP16 |
| dataQueue | Queue slot | `U` bytes×depth 2 | FP16 | CopyIn | Compute | TPipe；FreeTensor 后复用 | 双缓冲 |
| outQueue | Queue slot | `U` bytes×depth 2 | FP16 | Compute | CopyOut | TPipe；FreeTensor 后复用 | 双缓冲 |
| y GM | pointer-table `y[i]+begin+k*U/2` | `valid` elements | FP16/ND flatten | CopyOut | caller | caller | 仅写有效前缀 |
| workspace | N/A | Kernel 0；Tiling 请求 32 bytes | bytes | ACLNN caller | 未消费 | caller/executor | entry 强制 `userWS=nullptr` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ0["dataQueue slot0 · U bytes"]:1
  inQ1["dataQueue slot1 · U bytes"]:1
  outQ0["outQueue slot0 · U bytes"]:1
  outQ1["outQueue slot1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*U + 2*U = 4U
U = floor(((UB-P)/4)/32)*32
UB_total <= UB-P
dynamic_ub = N/A（AIV TPipe 静态 Queue）
workspace_total = 32 bytes requested；0 bytes consumed by Kernel body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue slot | CopyIn | DeQue→Mins | 后续 tile AllocTensor | EnQue/DeQue + `FreeTensor(dataLocal)` | 不在 Compute 释放前覆盖 |
| outQueue slot | Mins | DeQue→CopyOut | 后续 tile AllocTensor | EnQue/DeQue + `FreeTensor(outLocal)` | CopyOut 完成后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue | CopyIn/EnQue | Compute/DeQue | dataQueue slot | tail 用 `DataCopyPad`，有效 byte=`valid*2` |
| Vector | `PipeBarrier<PIPE_V>` | DeQue/前序 Vector | `Mins` 与后续提交 | input/output UB | FP32 scalar 先转 FP16；`valid` 窄化 INT32，UB 上界保证远小于 INT32_MAX |
| UB→GM | Queue | Compute/EnQue | CopyOut/DeQue | outQueue slot | 仅搬 `valid` 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `RunBigKernelTiling/DivideUbMemory3` | key、分核、U 和 32-byte workspace |
| Kernel entry/router | `foreach_minimum_scalar_list.cpp` | key 1 branch | FP16 模板映射 |
| Device body/helper | `kernel_foreach_unary.h` / `foreach_one_scalar_list_binary.h` | `Process/SingleTensorProcess/Compute` | 双 Queue、scalar 复用、tile/tail |
| Device primitive | `foreach_minimum_scalar_list.cpp` | `MinsAdapter<half>` | `Mins(...,valid)`；primitive 内部 NaN lane 细节固定仓不可见 |

### Route R-002：FP32 原生 Mins

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_minimum_scalar_list.cpp` | `foreach_minimum_scalar_list` | `TILING_KEY_IS(2)` | `<float,float,MinsAdapter<float>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
entry key2 -> ForeachOneScalarListBinary<float,float,...>::Init/Process
  -> KernelForeachUnary::Process -> CopyIn
  -> generic InnerComputer<float,float>::Compute -> MinsAdapter<float>
  -> CopyOut
```

##### 源码等价伪码

```text
for 本 core 的每个 Tensor 片段:
  scalar32 = scalars_fp32[i]
  for tile capacity=U/4 elements:
    CopyIn(valid); barrier; Mins(outUB,inUB,scalar32,int32(valid)); barrier; CopyOut(valid)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `ptr[i]+begin+k*U/4` | `valid` elements | FP32/ND | caller/Kernel | Kernel/caller | caller | full DataCopy；tail Pad |
| scalar GM | `scalars+i` | 1 FP32 | FP32 | bridge | Compute | input | Tensor 内复用 |
| dataQueue/outQueue | Queue slot | 各 `U` bytes×depth 2 | FP32 | CopyIn/Compute | Compute/CopyOut | TPipe | FreeTensor 后复用 |
| workspace | N/A | Kernel 0；Tiling 32 bytes | bytes | caller | 未消费 | executor | 无 body slice |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ0["dataQueue slot0 · U bytes"]:1
  inQ1["dataQueue slot1 · U bytes"]:1
  outQ0["outQueue slot0 · U bytes"]:1
  outQ1["outQueue slot1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=4U；U=floor(((UB-P)/4)/32)*32；dynamic_ub=N/A
workspace_total=32 bytes requested，Kernel consumed=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue/outQueue slots | CopyIn/Mins | Mins/CopyOut | 后续 tile | Queue + FreeTensor | producer/consumer 配对完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pipeline | Queue + `PIPE_V` barrier | CopyIn/Mins | Mins/CopyOut | UB slots | 全程 FP32；tail byte=`valid*4` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `GetNeedCoreNum/AssignDataToEachCore/DivideUbMemory3` | 分核、U |
| Kernel entry/router | `foreach_minimum_scalar_list.cpp` | key 2 branch | FP32 实例 |
| Device body/helper | `foreach_one_scalar_list_binary.h` / `foreach_one_scalar_binary.h` | `Compute` / generic `InnerComputer` | scalar 复用与原生 FP32 Mins |

### Route R-003：INT32 原生 Mins

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `op_kernel/foreach_minimum_scalar_list.cpp` | `foreach_minimum_scalar_list` | `TILING_KEY_IS(3)` | `<int,int,MinsAdapter<int>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
entry key3 -> ForeachOneScalarListBinary<int,int,...>::Init/Process
  -> ProcessPlusInLoop: int scalarVal = int(inScalarGM<int64>[i])
  -> CopyIn -> generic InnerComputer<int,int> -> MinsAdapter<int> -> CopyOut
```

##### 源码等价伪码

```text
scalar32 = int32(scalars_int64[i])
for tile capacity=U/4: y = Mins(x_int32, scalar32, valid)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `ptr[i]+begin+k*U/4` | valid | INT32/ND | caller/Kernel | Kernel/caller | caller | tail Pad |
| scalar GM | `scalars+i` | 1 INT64 | INT64 | bridge | ProcessPlusInLoop | input | 读取后显式转模板 `T=int` |
| dataQueue/outQueue | Queue slot | 各 `U` bytes×2 | INT32 | CopyIn/Mins | Mins/CopyOut | TPipe | 双缓冲 |
| workspace | N/A | Kernel 0；Tiling 32 bytes | bytes | caller | 未消费 | executor | 无 body slice |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ0["dataQueue slot0 · U bytes"]:1
  inQ1["dataQueue slot1 · U bytes"]:1
  outQ0["outQueue slot0 · U bytes"]:1
  outQ1["outQueue slot1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=4U；U=floor(((UB-P)/4)/32)*32；dynamic_ub=N/A
workspace_total=32 bytes requested，Kernel consumed=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Mins | Mins/CopyOut | 后续 tile | Queue + FreeTensor | 同一 slot 完成消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar | C++ 转换 | INT64 GM load | INT32 Mins | `scalarVal` | 固定 Host 未检查 INT64 是否落入 INT32；越界转换结果不可由本仓补齐 |
| pipeline | Queue + barrier | CopyIn/Mins | Mins/CopyOut | UB slots | INT32；tail byte=`valid*4` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_minimum_scalar_list_binary.json` / `foreach_tiling_class.h` | INT32 entry / key 3 | scalar binary INT64，key 按 x dtype |
| Kernel entry/router | `foreach_minimum_scalar_list.cpp` | key 3 branch | 模板 P/T 均为 int |
| Device body/helper | `foreach_one_scalar_list_binary.h` | `ProcessPlusInLoop` | INT64 scalar 读取后转 INT32 |

### Route R-004：BF16 分段 FP32 Mins

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `op_kernel/foreach_minimum_scalar_list.cpp` | guarded key 4 branch | CCE>=220 且非 arch3003/3113 | `<bfloat16_t,float,MinsAdapter<float>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
entry key4 -> ForeachOneScalarListBinary<bfloat16_t,float,...>::Init
  -> KernelForeachUnary::Init（2×9U BF16 input/output Queue + 1×2U FP32 Queue）
  -> Process -> CopyIn
  -> InnerComputer<bfloat16_t,float>::Compute
     -> ComputePerCast: Cast BF16→FP32 -> MinsAdapter<float> -> Cast FP32→BF16 RINT
  -> CopyOut
```

##### 源码等价伪码

```text
scalar32 = scalars_fp32[i]
outer tile capacity = 9U/2 BF16 elements
CopyIn(valid)
for cast chunk c, capacity=U/4 FP32 elements:
  Cast(fp32[0:chunk], bf16[c*capacity:], CAST_NONE)
  Mins(fp32[capacity:capacity+chunk], fp32[0:chunk], scalar32, chunk)
  Cast(out_bf16[c*capacity:], fp32[capacity:], CAST_RINT)
CopyOut(valid)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `ptr[i]+begin+k*(9U/2)` | valid | BF16/ND | caller/Kernel | Kernel/caller | caller | tail Pad |
| scalar GM | `scalars+i` | 1 FP32 | FP32 | bridge | FP32 Mins | input | Tensor 内复用 |
| dataQueue | slots | `9U` bytes×2 | BF16 | CopyIn | Cast | TPipe | double buffer |
| outQueue | slots | `9U` bytes×2 | BF16 | final Cast | CopyOut | TPipe | double buffer |
| float32Queue | 单 slot，子区 0/1 | `2U` bytes×1 | FP32 | input Cast / Mins | Mins / output Cast | 整个 Process 持有，末尾释放 | 两个 `U` 子区，不别名 |
| workspace | N/A | Kernel 0；Tiling 32 bytes | bytes | caller | 未消费 | executor | 无 body slice |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  in0["dataQueue slot0 · 9U"]:1
  in1["dataQueue slot1 · 9U"]:1
  out0["outQueue slot0 · 9U"]:1
  out1["outQueue slot1 · 9U"]:1
  fsrc["float32Queue src · U"]:1
  fdst["float32Queue dst · U"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*9U + 2*9U + 1*2U = 38U
U = floor((((UB-P)/10)/4)/64)*64
38U <= 0.95*(UB-P)（忽略向下对齐只会更小）
dynamic_ub=N/A
workspace_total=32 bytes requested，Kernel consumed=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 src 子区 | BF16→FP32 Cast | FP32 Mins | 下一 cast chunk | `PIPE_V` barrier | Mins 已完成读取 |
| FP32 dst 子区 | FP32 Mins | FP32→BF16 Cast | 下一 cast chunk | `PIPE_V` barrier | 降精度 Cast 已消费 |
| BF16 Queue slots | CopyIn/final Cast | Cast/CopyOut | 后续 tile | Queue + FreeTensor | slot 消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | Cast + `PIPE_V` | input Queue | Mins | FP32 src 子区 | `CAST_NONE`；内层 remainder 精确 count |
| Mins | `MinsAdapter<float>` + barrier | FP32 src | output Cast | FP32 dst 子区 | FP32 比较；primitive NaN 细节固定仓不可见 |
| FP32→BF16 | Cast `CAST_RINT` | FP32 dst | CopyOut | BF16 output slot | round-to-nearest integral mode 名称由源码固定；尾长精确 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory3` | BF16 `/10` 后再 `/4`、64-byte 对齐 |
| Kernel entry/router | `foreach_minimum_scalar_list.cpp` | guarded key 4 | BF16→FP32 模板 |
| Device body/helper | `kernel_foreach_base.h` / `kernel_foreach_unary.h` | BF16 `Init` | `9U` Queue 与 `2U` FP32 Queue |
| Device body/helper | `foreach_one_scalar_binary.h` | BF16 `InnerComputer::ComputePerCast` | 双 FP32 子区、barrier、CAST_RINT |

## 算子特定优化

### Trick T-001：每个 Tensor 只装载一次 scalar 并跨 tile 复用

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：进入 `KernelForeachUnary::Process` 的第 `i` 个 Tensor 片段。
- Golden 来源：[`foreach_one_scalar_list_binary.h:81-91`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h#L81-L91)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart)
{
#if __CCE_AICORE__ >= 220
    if (std::is_same_v<T, bfloat16_t>) {
        scalarVal = inScalarGM.GetValue(index);
    } else {
        scalarVal = T(inScalarGM.GetValue(index));
    }
#else
    scalarVal = T(inScalarGM.GetValue(index));
#endif
}
```

#### 机制与设计原因

`ProcessPlusInLoop` 位于 Tensor 循环、tile 循环之外；因此每个 core 处理某 Tensor 片段时只从 scalar GM 读取一次 `scalars[i]`，之后 `SingleTensorProcess` 的全部 tile 复用寄存器/标量成员 `scalarVal`。这减少了每 tile 重复 scalar GM load，并把 dtype 转换也移出 tile 循环。索引来自当前 Tensor `i`，与 TensorList 指针表选择保持一致。

#### 迁移边界

- 适用：一个标量广播到单个 Tensor 全有效域的 foreach 单目+scalar 算法。
- 前提：scalar 在一个 Tensor 的所有 tile 间不变，Tensor 索引与 scalar 列表一一对应。
- 失效：标量逐元素变化或 tile 间变化时必须移入相应循环，不能复用此成员。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：BF16 用固定双子区 FP32 scratch 分段升精度

#### 触发与源码依据

- 适用 Route：R-004。
- 触发条件：`T=bfloat16_t` 且 BF16 编译 guard 成立。
- Golden 来源：[`foreach_one_scalar_binary.h:69-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h#L69-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
op(float32Tensor[offset], float32Tensor, scalarVal, dataCount);
PipeBarrier<PIPE_V>();
Cast(outLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
```

#### 机制与设计原因

`paramsCount=2` 使 `offset=maxCastDataCount`，同一个 `2U` FP32 Queue 被固定切成输入与输出两个 `U` 子区；每次只处理 `U/4` 个元素，完成 BF16→FP32、FP32 Mins、FP32→BF16 后覆盖这两个子区。外层 BF16 Queue 可容纳 `9U` bytes，而 scratch 不随外层 tile 等比例扩张，从而在保留 FP32 比较顺序的同时控制 scratch。三个 `PIPE_V` barrier 保护同一 scratch 的阶段覆盖。

#### 迁移边界

- 适用：低精度输入、单个 FP32 中间结果且允许分段处理的逐元素算子。
- 前提：Host 按 BF16 路线给出 `/10`、`/4`、64-byte 对齐的 `U`，并保留两个不别名 FP32 子区。
- 失效：需要多个同时存活的 FP32 中间量时必须增大 `paramsCount`/scratch，不能沿用 `2U` 字节账。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key1 | key2 | key3 | guarded key4 | x dtype 唯一分发 |
| 模板与 dtype | half/half | float/float | int/int，scalar GM INT64 | bf16/float | binary 配置与计算精度 |
| 循环与分核 | 32-byte 串接分核、逐 Tensor/tile | 同左 | 同左 | 同左，另有 cast chunk | 公共基类与 BF16 specialization |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4U Queue | 4U Queue | 4U Queue | 38U Queue/scratch | BF16 升精度需要额外局部存储 |
| 内存总占用与复用 | depth2 输入/输出 | 同左 | 同左 | depth2 9U 输入/输出 + 两 FP32 子区 | Host UB divider 与 Device Init 对齐 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Pad + Mins | 同左 | 同左 | DataCopy/Pad + Cast/Mins/Cast | BF16 specialized InnerComputer |
| 同步、精度与 tail | FP16 原生 | FP32 原生 | scalar 先窄化 INT32 | FP32 比较、RINT 回 BF16 | 模板类型与 specialization |
| fallback/失败边界 | 无 dtype fallback | 无 | INT64 范围未校验 | guard 不成立无 body | Host 应在 launch 前拒绝不支持配置 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持（公开产品表，`ascend910b`） | 支持（公开产品表，`ascend910_93`） | 支持，但使用独立 `_950` 卡 |
| Host/Launcher | 共用 arch22 `ForeachCommonTiling` | 共用 arch22 `ForeachCommonTiling` | arch35 独立 POD/tiling，最多 256 Tensor |
| Kernel/资源 | 共用 AIV Vector + UB Queue | 共用 AIV Vector + UB Queue | SIMT、直接 GM、无 AIV Queue |

## 限制与失败边界

- A2/A3 公共 POD 最多表达 50 个 Tensor/core；公开 ACLNN 文档未写该上限，但源码无 >50 的正式 fallback。
- output 非连续 Tensor 不支持；input 可由 `AutoContiguous` 框架前处理，生成任务及其额外 workspace offset 固定仓不可见。
- Tiling 请求 32 bytes workspace，Kernel body 实际忽略 workspace；ACLNN 返回的总 workspace 还可能含框架连续化空间。
- INT32 Route 从 INT64 scalar GM 窄化到 INT32，固定仓无数值范围校验；不得假设越界 scalar 的可移植结果。
- `ForeachCommonTilingData` 的宏展开 ABI 数字、自动生成 ACLNN bridge、TensorList/ScalarList 打包 producer 以及 AscendC `Mins` primitive 内部 NaN lane 细节固定仓不可见。
- UB 减法无显式小资源 guard，依赖已注册 A2/A3 平台满足 `UB>P`；无资源 fallback。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_minimum_scalar_list/docs/aclnnForeachMinimumScalarList.md` | 两段式原型/产品表/错误表 | API、平台、dtype、shape、连续性 |
| 入口实现 | `foreach/foreach_minimum_scalar_list/op_host/foreach_minimum_scalar_list_def.cpp` | `ForeachMinimumScalarList` | ScalarList/AutoContiguous/平台注册 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` | 动态输入输出一一对应 |
| Host/Tiling/Launcher | `foreach/foreach_minimum_scalar_list/op_host/arch22/foreach_minimum_scalar_list_tiling_arch22.cpp` | `Tiling4ForeachMinimumScalarListTiling` | opCode/inputType 与公共 Tiling 入口 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD、workspace |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | 字段与注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | key 映射与 dtype bytes |
| Kernel entry/body | `foreach/foreach_minimum_scalar_list/op_kernel/foreach_minimum_scalar_list.cpp` | `foreach_minimum_scalar_list/MinsAdapter` | key router、模板、Mins |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h` | `ForeachOneScalarListBinary` | scalar list 读取与 Compute |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `InnerComputer` | native/BF16 计算与 barrier |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、tile、DMA、tail |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD 解析、指针表、tile capacity |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_host/config/ascend910b/foreach_minimum_scalar_list_binary.json` | `op_list` | A2 dtype binary |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_host/config/ascend910_93/foreach_minimum_scalar_list_binary.json` | `op_list` | A3 dtype binary |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3 使用 arch22，ACLNN 构建 |
| 构建接线 | `foreach/foreach_minimum_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 使用公共 Kernel，`AUTO_SYNC false` |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 6 字段表达 UB 与每核跨 Tensor 区间 | 数组长度 50、字段顺序、inclusive offsets | `ForeachCommonTilingData/ParseTilingData` |
| 调用顺序 | 每 Tensor load scalar，再逐 tile CopyIn→Compute→CopyOut | scalar 索引必须与 Tensor 索引一致 | `Process/ProcessPlusInLoop` |
| workspace/内存 | 普通 4U；BF16 38U；Kernel workspace 0 | depth=2 必须计入，BF16 `9U/2U` 不能按逻辑长度缩减 | `DivideUbMemory3/KernelForeachUnary::Init` |
| 同步 | Queue 生命周期 + Vector barrier | scratch 覆盖前必须等待前一阶段 | `InnerComputer/CopyIn/CopyOut` |
| 精度与尾块 | BF16 分段 FP32；full/tail 两种 DMA | `CAST_NONE→Mins→CAST_RINT` 与有效 byte 数 | `ComputePerCast/SingleTensorProcess` |
| 失败边界 | 校验失败阻断 launch；未知 key 无 body | 不得为 >50、低 UB、越界 INT64 scalar 编造 fallback | `CheckOpInfo/router` |