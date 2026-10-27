---
type: Operator
title: "ForeachMaximumScalarList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachMaximumScalarList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/docs/aclnnForeachMaximumScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/foreach_maximum_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/arch22/foreach_maximum_scalar_list_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/config/ascend910b/foreach_maximum_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_host/config/ascend910_93/foreach_maximum_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:47:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachMaximumScalarList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMaximumScalarListGetWorkspaceSize` + `aclnnForeachMaximumScalarList`，公开 ACLNN 两段式接口，无更高版本后缀 |
| 计算语义 | 对第 `i` 个 Tensor 的每个元素计算 `y_i[j] = max(x_i[j], scalars_i)` |
| 输入输出 | `x/out` 为 ND TensorList，单 Tensor 0–8 维；`x/out` 支持 FLOAT16/FLOAT32/INT32/BFLOAT16；算子层 scalar Tensor 分别为 FLOAT32/FLOAT32/INT64/FLOAT32 |
| 执行类型 | 单 Kernel：`AIV_VECTOR`；无 Cube/AIC、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | R-001 FP16 原生 `Maxs`；R-002 FP32 原生 `Maxs`；R-003 INT32 原生 `Maxs`；R-004 BF16→FP32→BF16 |
| 核心资源 | TensorList GM 指针表、scalar GM、双缓冲 `VECIN/VECOUT`；BF16 另有单缓冲 FP32 Queue；32 B executor workspace 不被 Kernel 使用 |
| 来源闭包 | `source_closed`：公开 ACLNN 声明/约束经框架生成的两段式 executor 边界，闭合到 `ForeachMaximumScalarList` OpDef、arch22 Tiling、`ForeachCommonTilingData`、Kernel router、共享 Vector body 与 A2/A3 构建接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`、`scalars=[s_0,...,s_{n-1}]`，输出 `y` 与 `x` 等长，且 `y_i` 的有效元素域与 `x_i` 相同：

```text
for i in [0,n):
  for j in [0,numel(x_i)):
    y_i[j] = max(x_i[j], s_i)
```

Host 按所有 Tensor 的 storage numel 拼成一个逻辑一维域分核，但 Device 仍通过 TensorList 指针表逐 Tensor 访问，不把 Tensor 数据物化为连续大数组。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 为 ND、0–8 维，可为空 | FLOAT16/FLOAT32/INT32/BFLOAT16，列表内一致 | 可为非连续 Tensor；OpDef 标记 `AutoContiguous()`，具体 Contiguous 实现属于框架边界 |
| `scalars` | ScalarList；元素数等于 `x` Tensor 数 | ACLNN 对浮点 `x` 接受 FLOAT32/DOUBLE，对 INT32 接受 INT64；算子配置实际为 FLOAT32 或 INT64 | ACLNN ScalarList 到算子 scalar Tensor 的生成/DOUBLE→FLOAT32 规范化代码由框架生成，固定仓不可见 |
| `out` | TensorList；ND、0–8 维 | 与 `x` 相同 | 输出不支持非连续；公开文档允许 shape size 大于等于输入，但公共 infer-shape 产生与输入完全相同的 shape；Kernel 只写输入有效 numel |

### 入口与版本

固定提交公开的最新入口是无版本后缀的 `aclnnForeachMaximumScalarList`。文档明确 A2、A3 支持；OpDef 注册 `ascend910b` 与 `ascend910_93`，Host CMake 将二者都映射到 `arch22`，Kernel CMake 为二者编译同一个非 `arch35` entry。两段式 ACLNN inner bridge 未以手写源文件出现，作为框架生成的 executor/Contiguous 边界处理；算子拥有的 Host/Tiling/POD/Kernel/构建链路均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段检查 `x/scalars/out` 非空、公开 dtype 组合和 TensorList 约束，并生成 executor；`x` 的非连续视图由 OpDef 的 `AutoContiguous()` 进入框架 Contiguous 边界。
2. 公共 infer-shape 要求动态输入实例数等于输出数，并逐项复制 `x_i` shape/dtype 到 `y_i`。
3. `Tiling4ForeachMaximumScalarListTiling` 以 `FOREACH_BINARY_SCALAR_OP_CODE`、`TYPE_SCALARS_TENSOR` 初始化 `ForeachCommonTiling`。`Init` 最多扫描 `MAX_TENSOR_CONT=50` 个 `x_i`，累计 `tensorDataCountList[i]=storageShape.GetShapeSize()` 和 `totalDataCount`。
4. `CheckOpInfo` 检查 scalar Tensor 元素数等于 Tensor 数、输入/输出单 Tensor 维数不超过 8、列表内 dtype 一致、输出 dtype 与输入一致，且输出 storage numel 不小于输入。
5. 任一 desc/shape/platform/workspace 指针缺失或校验失败都返回 `GRAPH_FAILED`，ACLNN 第一段将其报告为参数或 inner tiling 错误；第二段只按 executor 在给定 stream 上启动已建立的图。

### 参数、Workspace 与生命周期

设 `B=32` 字节，`e=B/sizeof(T)`，`E=sum_i numel(x_i)`，平台 AIV core 数为 `C`：

```text
N = min(C, max(1, ceil(E/e)), 50)
blockCount = ceil(E/e)
base = floor(blockCount/N) * e
前 blockCount mod N 个 core 各多处理 e 个元素
```

Tiling 以 Tensor 边界和首尾 offset 记录每核区间。workspace 固定为 32 B；entry 将 `userWS=nullptr`，算子 Kernel 不读写该 32 B。workspace 由 ACLNN 调用者按第一段返回值分配，executor 持有 launch 参数直到第二段提交；stream 决定单 Kernel 的执行顺序和可见性。

### 启动图与失败传播

```text
aclnnForeachMaximumScalarListGetWorkspaceSize
  -> framework executor / AutoContiguous boundary
  -> ForeachMaximumScalarList Host + arch22 tiling
     [失败：不创建可执行 launch]
aclnnForeachMaximumScalarList(workspace=32 B, executor, stream)
  -> node K0: foreach_maximum_scalar_list<<<blockDim=N>>>
       inputs=TensorList GM, scalar=contiguous scalar GM,
       outputs=TensorList GM, workspace=unused, tiling=ForeachCommonTilingData
     [失败：该单节点 launch 状态向 ACLNN 返回，无后继节点]
```

没有跨 Kernel 边。输入、scalar、输出和 tiling 均由 executor/框架持有至 K0 完成；K0 在给定 stream 上直接由 GM 读入、经 UB 计算并写回各 `y_i`。

## Route 与参数设计

### 分核与资源计算

公共 Tiling 先按 32 B block 分核，再计算单个 Queue buffer 的字节参数 `U=inputsTensorUbSize`。令 `P=tilingData.GetDataSize()`、平台 UB 字节数为 `UB`：

```text
非 BF16: U = floor_to_32((UB - P) / 4)
BF16:    U = floor_to_64(((UB - P) / 10) / 4)
```

非 BF16 的每 tile 最大元素数是 `U/sizeof(T)`；BF16 数据 Queue 的单 buffer 为 `9U`，最大 tile 为 `9U/2` 个 BF16，按 `U/4` 个元素一段在 FP32 Queue 中分段 cast/比较/cast。TensorList 跨核切割可从一个 Tensor 中间开始或结束，POD 的 start/end Tensor 与 offset 精确描述该片段。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 原生 Maxs | key `1`; `<half,half,MaxsAdapter<half>,2,1>` | `dataType == DT_FLOAT16`，`GetTilingKeyByDtypeOnly` 返回 1 | 检查 key 2 |
| 2 | R-002 FP32 原生 Maxs | key `2`; `<float,float,MaxsAdapter<float>,2,1>` | `dataType == DT_FLOAT`，返回 2 | 检查 key 3 |
| 3 | R-003 INT32 原生 Maxs | key `3`; `<int,int,MaxsAdapter<int>,2,1>` | `dataType == DT_INT32`，返回 3 | 检查 key 4 |
| 4 | R-004 BF16 升精度 Maxs | key `4`; `<bfloat16_t,float,MaxsAdapter<float>,2,1>` | `dataType == DT_BF16`，且编译条件 `__CCE_AICORE__ >= 220`、非 NPU arch 3003/3113 | 未知/未编译 key 无 body，entry 不写输出 |

```text
dtype FLOAT16 -> key 1 -> R-001
else dtype FLOAT32 -> key 2 -> R-002
else dtype INT32 -> key 3 -> R-003
else dtype BF16 -> key 4 -> R-004（受编译 guard）
else -> key 0；OpDef/binary config 不提供该 dtype，tiling/launch 失败
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | FP16、合法 TensorList、A2/A3 | 单 K0 AIV_VECTOR | `ForeachOneScalarListBinary<half,half,...,2,1>` → `InnerComputer<half,half>` | `dataQueue/outQueue` 各 depth 2；两次 `PIPE_V` barrier 包围 `Maxs` | 最后一 tile 用 `DataCopyPad`; 无 fallback |
| R-002 FP32 | 2 | FP32、合法 TensorList、A2/A3 | 单 K0 AIV_VECTOR | `<float,float,...,2,1>` → `InnerComputer<float,float>` | 同 R-001，元素容量由 4 B dtype 缩半 | 同 R-001 |
| R-003 INT32 | 3 | INT32 + INT64 scalar、合法 TensorList、A2/A3 | 单 K0 AIV_VECTOR | `<int,int,...,2,1>` → `InnerComputer<int,int>` | scalar 在 GM→寄存器时窄化为 int；其余同 R-002 | 同 R-001；超 INT32 scalar 无显式范围 guard |
| R-004 BF16 | 4 | BF16、合法 TensorList、A2/A3，编译 guard 成立 | 单 K0 AIV_VECTOR | `<bfloat16_t,float,...,2,1>` → BF16 specialization | 2×9U 输入、2×9U 输出、1×U FP32 Queue；cast 段内原地 `Maxs` | BF16 tile tail + cast tail；无其它 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `Init` 只枚举 `i < MAX_TENSOR_CONT(50)` | POD 只有 50 个 `tensorDataCountList` 槽位 | 全部 | 公共校验会把合法第 51 个 scalar 与已扫描的 50 比较而失败；无 fallback | `ForeachCommonTiling::Init`, `CheckScalarsTenorShapeInfo` |
| 32 B 分核 | `elementsPerBlock=32/sizeof(T)` | 每核边界以 DMA block 对齐，只有 Tensor 总尾可非对齐 | 全部 | 不满足整块的最终 tile 走 `DataCopyPad` | `GetNeedCoreNum`, `AssignDataToEachCore` |
| UB 正容量 | 非 BF16 `UB>P`；BF16 还需整除后 `U>0` | `maxDataCount` 用作除数并决定 Queue capacity | 全部 | Host 无显式 `U==0` guard；平台资源必须满足 | `DivideUbMemory3`, `SingleTensorProcess` |
| BF16 编译能力 | `__CCE_AICORE__ >= 220` 且非 arch 3003/3113 | BF16 cast/FP32 Queue specialization 才实例化 | R-004 | key 4 无匹配 body，不写输出 | Kernel entry 与 `KernelForeachBase::Init` guard |
| Vector count 窄化 | tile `dataCount` 传给 `Maxs` 时转 `uint32_t` | Vector API count 参数为 32 位 | 全部 | 固定平台 UB 限制使单 tile 远小于 2^32；源码无独立溢出检查 | `MaxsAdapter` |

### POD/ABI

`ForeachCommonTilingData` 由 tiling-data 宏序列化；固定仓给出字段顺序和类型，但宏实现不在本仓，故 `sizeof/alignof/offsetof` 不可见，不能把按普通 C struct 推算的值当作 ABI 事实。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 第 1 字段；精确 offset/sizeof 固定仓不可见 | 字节/单 Queue buffer 基数 U | `DivideUbMemory3` → `FillTilingData` | `KernelForeachBase::Init` 计算 tile/Queue | 必须 >0；BF16 再乘 9 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 第 2 字段；精确 offset 不可见 | 元素 | `Init`: 每个 storage numel | `KernelForeachUnary::Process` | 仅 `[0,totalTensorCount)` 有效 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 第 3 字段；精确 offset 不可见 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | 仅活动 core 读取 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 第 4 字段；精确 offset 不可见 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | 闭区间终点 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 第 5 字段；精确 offset 不可见 | 元素 | `AssignDataToEachCore` | `ParseTilingData` | 相对 start Tensor |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 第 6 字段；精确 offset 不可见 | 元素 | `AssignDataToEachCore` | `ParseTilingData` | 相对 end Tensor，闭区间 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | arch22 `Tiling4...`; `ForeachCommonTiling::RunBigKernelTiling` | `ForeachCommonTilingData`, U/start/end | entry key 1 | `KernelForeachUnary`; `ForeachOneScalarListBinary`; `InnerComputer<half>` | Host/Kernel CMake + ascend910b/910_93 binary config |
| R-002 | key 2 | 同上 | 同上 | entry key 2 | 同上，`InnerComputer<float>` | 同上 |
| R-003 | key 3 | 同上 | 同上 | entry key 3 | 同上，`InnerComputer<int>` | 同上 |
| R-004 | key 4 | 同上，BF16 U 分支 | 同上 | entry key 4 + compile guard | BF16 `InnerComputer` specialization | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 后按 key 构造唯一模板实例，显式令 `userWS=nullptr`。`KernelForeachBase` 用 `GetBlockIdx()` 读取本核 start/end Tensor 和 offset；TensorList 首字保存指针表相对偏移，`GetTensorAddr` 定位第 `i` 个真实 GM 数据指针。每核逐 Tensor、逐 tile 执行 `CopyIn → Compute(Maxs) → CopyOut`；Queue 的 `EnQue/DeQue/FreeTensor` 管理搬运与 Vector 阶段生命周期。四条 Route 都是一个 Device 节点，没有跨节点 workspace 或同步边。

### Route R-001：FP16 原生 Maxs

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | `TILING_KEY_IS(1)` | key 1 | `<half,half,MaxsAdapter<half>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
K0 foreach_maximum_scalar_list
  -> ForeachOneScalarListBinary<half,...>::Init
    -> KernelForeachUnary::Init -> KernelForeachBase::Init/ParseTilingData
  -> KernelForeachUnary::Process
    -> ProcessPlusInLoop: scalar FP32 GM -> half scalarVal
    -> SingleTensorProcess
      -> CopyIn -> ForeachOneScalarListBinary::Compute
        -> InnerComputer<half>::Compute -> MaxsAdapter<half> -> Maxs
      -> CopyOut
```

##### 源码等价伪码

```text
core = GetBlockIdx()
for tensor i in [startTensor[core], endTensor[core]]:
  begin = (i == start ? startOffset : 0)
  end   = (i == end ? endOffset : numel[i]-1)
  scalarVal = half(scalars_fp32[i])
  bind xGM/yGM to tensor_ptr(i) + begin
  for tile from 0 while tile*U/2 < end-begin+1:
    valid = min(U/2, remaining)
    CopyIn valid half (tail uses DataCopyPad)
    PIPE_V barrier; Maxs(outUB, inUB, scalarVal, valid); PIPE_V barrier
    CopyOut valid half (tail uses DataCopyPad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `tensor_ptr(i)+begin+tile*(U/2)` | `valid` 元素 | FP16/ND flattened | ACLNN input | `CopyIn` | caller 至 K0 完成 | full tile 32 B aligned；tail pad copy |
| scalar GM | `scalars+i` | 1 | FP32 contiguous | framework scalar Tensor | `ProcessPlusInLoop` | executor 至 K0 完成 | 每 Tensor 读一次/核片段 |
| `dataQueue` | Queue 管理 | depth 2 × U B | FP16 | `CopyIn` | `Compute` | K0 core | 双缓冲，消费后 `FreeTensor` |
| `outQueue` | Queue 管理 | depth 2 × U B | FP16 | `Compute` | `CopyOut` | K0 core | 双缓冲，写回后释放 |
| y GM | 同 x 地址公式 | `valid` 元素 | FP16/ND flattened | `CopyOut` | caller | caller | 只写输入有效域 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2U + 2U = 4U
U = floor_to_32((UB - POD_bytes)/4)
dynamic_ub = N/A；AIV Queue 使用静态 UB
workspace_total = 32 B（Kernel 实际不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` 当前 slot | `CopyIn` | `Compute`/`Maxs` | 后续 tile `AllocTensor` | `EnQue→DeQue→FreeTensor` + `PIPE_V` | 计算完成前不得复用 |
| `outQueue` 当前 slot | `Compute` | `CopyOut` | 后续 tile `AllocTensor` | `EnQue→DeQue→FreeTensor` | GM 写回完成前不得复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue + `DataCopy/DataCopyPad` | GM | `DeQue` | input slot | tail 的 blockLen=`valid*2` B |
| Vector | `PipeBarrier<PIPE_V>` | CopyIn/Maxs | `Maxs`/enqueue | Vector pipeline | scalar 先 FP32→FP16；比较和输出均 FP16 |
| UB→GM | Queue + copy | `Maxs` | `CopyOut` | output slot | 只提交 `valid` 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling`, `AssignDataToEachCore`, `DivideUbMemory3` | key、core 区间、U、workspace=32 B |
| Kernel entry/router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | key 1 branch | 路由到 FP16 模板 |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_unary.h`; `foreach_one_scalar_binary.h`; `foreach_one_scalar_list_binary.h` | `Process/CopyIn/Compute/CopyOut`, `InnerComputer<half>` | 双缓冲、scalar cast、`Maxs`、tail |

### Route R-002：FP32 原生 Maxs

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | `TILING_KEY_IS(2)` | key 2 | `<float,float,MaxsAdapter<float>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
K0 entry -> ForeachOneScalarListBinary<float,...>::Init/Process
  -> per Tensor scalarVal = scalars_fp32[i]
  -> CopyIn<float> -> InnerComputer<float>::Compute
  -> MaxsAdapter<float>/Maxs -> CopyOut<float>
```

##### 源码等价伪码

```text
沿 POD 指定的 core/Tensor 闭区间遍历；每 tile 最多 U/4 个 float。
tail 用字节数 valid*4 的 DataCopyPad；Vector 阶段以 valid 作为 Maxs count，结果写相同 GM offset。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensor_ptr(i)+begin+tile*(U/4)` | `valid<=U/4` | FP32 flattened ND | caller/CopyOut | CopyIn/caller | K0 | tail 精确字节搬运 |
| scalar GM | `scalars+i` | 1 | FP32 | framework | Compute | K0 | 不窄化 |
| `dataQueue` | Queue slot | depth 2 × U B | FP32 | CopyIn | Maxs | K0 | dequeue/free 后复用 |
| `outQueue` | Queue slot | depth 2 × U B | FP32 | Maxs | CopyOut | K0 | dequeue/free 后复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 4U; U = floor_to_32((UB-P)/4)
dynamic_ub = N/A
workspace_total = 32 B（unused by K0）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 两个 Queue slot | CopyIn/Maxs | Maxs/CopyOut | 后续 tile | Queue 生命周期 + PIPE_V barrier | 与 R-001 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| copy/compute/copy | Queue + 两次 `PIPE_V` barrier | GM/Maxs | Maxs/GM | 当前 slot | 全程 FP32；tail=`valid` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/common_dtype.h`; `foreach_tiling_class.h` | `GetTilingKeyByDtypeOnly`, `RunBigKernelTiling` | key 2、U/N |
| Kernel entry/router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | key 2 branch | FP32 实例 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `InnerComputer<float,float>` | 原生 FP32 `Maxs` |

### Route R-003：INT32 原生 Maxs

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | `TILING_KEY_IS(3)` | key 3 | `<int,int,MaxsAdapter<int>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
K0 entry -> ForeachOneScalarListBinary<int,...>::Process
  -> scalarVal = int(scalars_int64[i])
  -> CopyIn<int> -> InnerComputer<int>::Compute -> Maxs<int> -> CopyOut<int>
```

##### 源码等价伪码

```text
与 R-002 相同的 U/4 tile 与地址公式；唯一数值差异是每个 INT64 scalar 在比较前转为 int，
之后执行 INT32 Maxs。源码没有 scalar 值域检查。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensor_ptr(i)+begin+tile*(U/4)` | `valid<=U/4` | INT32 flattened ND | caller/CopyOut | CopyIn/caller | K0 | tail 字节数 valid*4 |
| scalar GM | `scalars+i` | 1 | INT64→INT32 register | framework | Maxs | K0 | 每 Tensor/核片段读取 |
| data/out Queue | Queue slot | 各 depth 2 × U B | INT32 | CopyIn/Maxs | Maxs/CopyOut | K0 | 与 R-002 相同 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 4U; U = floor_to_32((UB-P)/4)
dynamic_ub = N/A
workspace_total = 32 B（unused）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| data/out Queue slot | CopyIn/Maxs | Maxs/CopyOut | 后续 tile | Queue + barrier | 当前 tile 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar | C++ conversion | INT64 GM | `Maxs<int>` | scalar register | 超 INT32 值域行为未由 Host 限制 |
| tile | Queue + `PIPE_V` | GM/Maxs | Maxs/GM | UB slot | INT32 比较；tail 精确写回 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | INT32→key 3 |
| Kernel entry/router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | key 3 branch | `int` 实例 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_list_binary.h` | `ProcessPlusInLoop` | INT64 scalar 转 T；随后原生 Maxs |

### Route R-004：BF16 升精度 Maxs

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | key 4 + compile guard | guard 成立 | `<bfloat16_t,float,MaxsAdapter<float>,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
K0 entry -> ForeachOneScalarListBinary<bfloat16_t,float,...>::Init
  -> data/out Queue(depth 2, each slot 9U), float32Queue(depth 1, U)
  -> Process: dequeue persistent float32Tensor
    -> per Tensor scalarVal = FP32 scalar
    -> per tile CopyIn BF16
    -> InnerComputer BF16 specialization::Compute
       -> repeated ComputePerCast
          -> Cast BF16->FP32(CAST_NONE)
          -> MaxsAdapter<float> in-place on float32Tensor
          -> Cast FP32->BF16(CAST_RINT) into outLocal
    -> CopyOut BF16
  -> free persistent float32Tensor
```

##### 源码等价伪码

```text
tileCap = 9U/2 BF16 elements; castCap = U/4 elements
for each assigned Tensor segment:
  scalarVal = scalars_fp32[i]
  for each tile:
    CopyIn BF16 valid elements
    for k in [0, ceil(valid/castCap)):
      n = min(castCap, valid-k*castCap)
      fp32[0:n] = Cast(bf16[k*castCap:k*castCap+n], CAST_NONE)
      fp32[0:n] = Maxs(fp32[0:n], scalarVal)  # paramsCount=1, offset=0
      out_bf16[k*castCap:...] = Cast(fp32[0:n], CAST_RINT)
    CopyOut only valid BF16 elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensor_ptr(i)+begin+tile*(9U/2)` | `valid<=9U/2` | BF16 flattened ND | caller/CopyOut | CopyIn/caller | K0 | tail `valid*2` B |
| `dataQueue` | Queue slot | depth 2 × 9U B | BF16 | CopyIn | Cast | K0 | double buffer |
| `outQueue` | Queue slot | depth 2 × 9U B | BF16 | Cast RINT | CopyOut | K0 | double buffer |
| `float32Queue` | offset 0 | depth 1 × U B | FP32 | BF16 Cast / Maxs | Maxs / BF16 Cast | 整个 Process | 同一子区原地覆盖，逐 cast 段复用 |
| scalar GM | `scalars+i` | 1 | FP32 | framework | Maxs | K0 | 不先降为 BF16 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 18U B"]:2
  fp32["float32Queue · depth=1 · U B"]:1
  outQ["outQueue · depth=2 · 18U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*(9U) + 1*U + 2*(9U) = 37U
U = floor_to_64(((UB-P)/10)/4)
maxDataCount = 9U/2 BF16 elements
maxCastDataCount = U/4 FP32 elements
dynamic_ub = N/A
workspace_total = 32 B（unused）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `float32Tensor[0:castCap]` | BF16→FP32 Cast | 原地 Maxs、FP32→BF16 Cast | 下一 cast 段 | 三处 `PIPE_V` barrier；Process 末 `FreeTensor` | 降精度 Cast 完成后才能覆盖 |
| BF16 Queue slot | CopyIn/降精度 Cast | Cast/CopyOut | 后续 tile | Queue EnQue/DeQue/FreeTensor | 当前消费者完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `Cast(CAST_NONE)` + `PIPE_V` | dataQueue | Maxs | FP32 子区 | cast 段尾用实际 `dataCount` |
| FP32 Maxs | `Maxs` + `PIPE_V` | FP32 Cast | output Cast | 同一 FP32 子区 | scalar 保持 FP32；原地 dst=src |
| FP32→BF16 | `Cast(CAST_RINT)` | Maxs | outQueue/CopyOut | BF16 output slot | RINT 后只写有效 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory3` | BF16 先 `/10`，再 `/4`、64 B 对齐 |
| Kernel entry/router | `foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | key 4 branch | BF16/FP32 模板与 compile guard |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_base.h`; `kernel_foreach_unary.h`; `foreach_one_scalar_binary.h` | BF16 specializations | 9U Queue、U FP32 Queue、分段 cast 与原地 Maxs |

## 算子特定优化

### Trick T-001：BF16 分段升精度并原地复用单个 FP32 子区

#### 触发与源码依据

- 适用 Route：R-004
- 触发条件：`T=bfloat16_t`、key 4 且 BF16 compile guard 成立；模板 `paramsCount=1`
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

BF16 路线不直接执行 BF16 `Maxs`：输入按 `maxCastDataCount=U/4` 分段转成 FP32，`paramsCount=1` 令 `offset=0`，所以 `Maxs` 的 dst 与 src 使用同一 FP32 子区，随后立即 RINT 降回 BF16。该机制以单个 U 字节 FP32 Queue 完成升精度比较，避免为结果再物化第二个 FP32 子区；barrier 固定了 cast→原地 Maxs→cast-out 的覆盖顺序。

#### 迁移边界

- 适用：BF16 输入、每 Tensor 一个 FP32 scalar、逐元素一元+scalar Vector 运算。
- 前提：A2/A3 BF16 编译能力、`paramsCount=1`、Host 为 37U 实际 Queue 占用预留足够 UB。
- 失效：多输入中间量需要同时存活时 `paramsCount` 会改变 offset/容量；无 BF16 guard 时不能进入 R-004。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但复用必须保持源码中的三段 `PIPE_V` 次序，不能把原地子区在降精度 Cast 完成前覆盖。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 3 | key 4 + guard | dtype-only selector |
| 模板与 dtype | half/half | float/float | int/int | bf16/float | scalar/计算类型匹配 entry 实例 |
| 循环与分核 | 32 B block，U/2 tile | 32 B block，U/4 | 32 B block，U/4 | 32 B block，9U/2 + U/4 cast 段 | dtype 字节与 BF16 升精度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 双 Queue | 双 Queue | 双 Queue | 双 9U Queue + FP32 Queue | 无 TBuf/dynamic UB/通信 |
| 内存总占用与复用 | 4U | 4U | 4U | 37U，FP32 原地复用 | Host divider 对应实际分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Maxs | 同左 | 同左 | DataCopy(Pad)+Cast+Maxs+Cast | 全部 AIV_VECTOR |
| 同步、精度与 tail | FP16，Queue/barrier | FP32 | INT64 scalar 窄化到 INT32 | FP32 compare、RINT BF16 | 模板 specialization |
| fallback/失败边界 | 未知 key 无输出 | 同左 | scalar 值域无 guard | compile guard 不满足无 body | 无运行时备用 Kernel |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 支持；见独立 `_950` 卡 |
| Host/Launcher | `ascend910b`→arch22，共享 Tiling | `ascend910_93`→arch22，共享 Tiling | arch35 自有 Tiling，workspace=0，最多 256 Tensor |
| Kernel/资源 | AIV_VECTOR、Queue UB、workspace 32 B unused | 同 A2 | SIMT 直读写 GM，无 AIV Queue；FP16/BF16 FP32 compare、INT32 INT64 compare |

## 限制与失败边界

- A2/A3 公共 POD 只有 50 个 Tensor 槽位，`Init` 也只扫描 50 个；对满足公开约束的第 51 个 scalar，scalar-count 校验会因 `totalTensorCount==50` 而失败，正式复用不得扩展此边界。
- ACLNN 文档允许浮点 ScalarList 的 DOUBLE，但生成 bridge 的 DOUBLE→算子 FLOAT32 规范化细节固定仓不可见；Device 只看到配置中的 FLOAT32 scalar Tensor。
- `out` 文档允许更大 numel，但 infer-shape 默认与 `x` 相同；Kernel 仅写 `x_i` 的有效 numel，额外输出区不定义。
- INT32 路线把 INT64 scalar 转为 `int`，无值域 guard；超 INT32 范围不可作为已闭合语义使用。
- Host 未显式检查 `U>0`；实现依赖支持平台 UB 足以容纳 POD 与 Queue。未知 key 或 BF16 compile guard 不成立时 entry 没有 fallback，也不写输出。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_maximum_scalar_list/docs/aclnnForeachMaximumScalarList.md` | 两段式原型、产品表 | 公开 API、平台、dtype/shape/错误语义 |
| 入口实现 | `foreach/foreach_maximum_scalar_list/op_host/foreach_maximum_scalar_list_def.cpp` | `ForeachMaximumScalarList` | OpDef、AutoContiguous、平台配置 |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar_list/op_host/arch22/foreach_maximum_scalar_list_tiling_arch22.cpp` | `Tiling4ForeachMaximumScalarListTiling` | 当前算子选择公共 Tiling |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace、POD producer |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD 字段与注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key 与 dtype 字节数 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`, `InferDataType4ForeachCommon` | 动态列表输出 shape/dtype 推导 |
| Kernel entry/body | `foreach/foreach_maximum_scalar_list/op_kernel/foreach_maximum_scalar_list.cpp` | `foreach_maximum_scalar_list`, `MaxsAdapter` | key router 与模板实例 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD consumer、TensorList 地址、BF16 tile |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、循环、搬运、tail |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `InnerComputer` | 原生/BF16 计算与 barrier |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h` | `ForeachOneScalarListBinary` | scalar list 读取与计算接线 |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | A2/A3→arch22 |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_host/config/ascend910b/foreach_maximum_scalar_list_binary.json` | 四个 binary 条目 | A2 dtype 实例 |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_host/config/ascend910_93/foreach_maximum_scalar_list_binary.json` | 四个 binary 条目 | A3 dtype 实例 |
| 构建接线 | `foreach/foreach_maximum_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 entry 编译选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段公共 POD 与 TensorList 指针表解析 | 字段顺序、50 槽、闭区间 offset | `ForeachCommonTilingData`, `ParseTilingData`, `GetTensorAddr` |
| 调用顺序 | Init→Process→per Tensor→per tile→CopyIn/Compute/CopyOut | key 对应模板和 scalar 每 Tensor更新 | entry；`KernelForeachUnary::Process` |
| workspace/内存 | 32 B workspace unused；4U/37U UB 账 | depth=2、BF16 9U/U 容量和 Queue 生命周期 | `RunBigKernelTiling`; `InitBuffer` |
| 同步 | Queue 协议与 `PIPE_V` barrier | 原地 BF16 子区的 cast/Maxs/cast 顺序 | `InnerComputer::ComputePerCast` |
| 精度与尾块 | 原生 FP16/FP32/INT32；BF16 FP32+RINT | scalar conversion、实际 valid、DataCopyPad | `ProcessPlusInLoop`; `CopyIn/CopyOut` |
| 失败边界 | 校验失败终止；未知 key 无输出 | 不虚构 fallback；不扩展 50 Tensor | `CheckOpInfo`; entry router |