---
type: Operator
title: "ForeachMulScalarList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachMulScalarList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/docs/aclnnForeachMulScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_host/foreach_mul_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_kernel/foreach_mul_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_host/config/ascend910b/foreach_mul_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_host/config/ascend910_93/foreach_mul_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:58:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachMulScalarList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMulScalarListGetWorkspaceSize` + `aclnnForeachMulScalarList`，公开两段式 ACLNN；固定仓只保留公开声明文档，自动生成 bridge/executor 壳为框架边界 |
| 计算语义 | 对列表位置 `i` 与元素位置 `j`，`out[i][j] = x[i][j] * scalars[i]` |
| 输入输出 | `x/out` 为 ND TensorList，元素 dtype 同为 FP16/FP32/INT32/BF16；A2/A3 的 `scalars` 分别为 FP32/FP32/INT64/FP32 |
| 执行类型 | 单 Kernel：`foreach_mul_scalar_list`，`KERNEL_TYPE_AIV_ONLY`，AIV_VECTOR |
| Kernel 路线 | R-001 FP16；R-002 FP32；R-003 INT32；R-004 BF16 升精度 |
| 核心资源 | GM TensorList 描述/指针表、每核一对 VECIN/VECOUT Queue；BF16 另有单深度 FP32 Queue；无算子临时 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→OpDef/Infer→优先级 Tiling→POD→AIV entry→模板 body/helper→binary config/CMake 已闭合；自动 ACLNN bridge 为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x=[x_0,...,x_{n-1}]`，`scalars=[s_0,...,s_{n-1}]`，则 `y_i=x_i*s_i`。每个张量按 storage shape 展平处理；有效输出仅覆盖 `x_i` 的元素数。InferShape 将每个 `y_i` 设为对应 `x_i` 的 shape，InferDataType 将 dtype 设为 `x_i` dtype。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 0–8 维，ND | FP16/FP32/INT32/BF16 | 非空指针；列表内 dtype 一致；支持空 Tensor；非连续输入由 OpDef `AutoContiguous()` 归一化 |
| `scalars` | ScalarList，长度等于 `x` Tensor 数 | FP32（浮点 `x`）或 INT64（INT32 `x`） | 每个列表位置提供一个广播标量；A2/A3 不接受 DOUBLE |
| `out` | TensorList；ND，0–8 维 | 与 `x` 一致 | 列表长度与 `x` 相等；输出不支持非连续；公开文档允许 shape size 不小于输入，图推导路径则生成与输入完全相同 shape |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段返回；内部 Tiling 固定预留 32 B，Kernel 不使用该空间 |
| `workspace/stream` | Device 地址/运行时流 | — | 第二段由 caller 提供 workspace 并在指定 stream 上执行 |

### 入口与版本

固定 Commit 公开的最新入口是无版本后缀的 `aclnnForeachMulScalarList` 两段式接口。文档明确支持 Atlas A2/A3；OpDef 注册 `ascend910b` 与 `ascend910_93`，两份 binary config 都列出四组 dtype，CMake 将默认 Kernel 接到这两个 compute unit。仓内未出现更高版本 ACLNN 符号。自动生成的 C 声明头、inner bridge、executor/launch 壳未随该算子目录公开，按框架边界处理；算子拥有的 OpDef、Tiling、POD 与 Kernel 已继续闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段 ACLNN 检查 `x/scalars/out` 非空、dtype 组合、列表/shape/dim 约束；错误语义由公开文档给出。
2. OpDef 对 `x` 与 `scalars` 启用 `AutoContiguous()`，输出要求连续；ScalarList 按 `x` dtype 映射到 FP32 或 INT64。
3. `InferShape4ForeachCommon` 要求输入/输出列表数相等并逐项复制 shape；`InferDataType4ForeachCommon` 逐项复制 dtype。
4. `Tiling4ForeachMulScalarListTiling` 进入模板 Registry。优先级数字越小越先尝试：10000 的 `ForeachMulScalarListMembaseTiling` 在非 regbase A2/A3 上可用；30000 regbase 模板不再到达。
5. `ForeachCommonTiling::CheckOpInfo` 检查 TensorList 数、维数、ScalarList 长度、输出 shape/dtype 和列表内 dtype；失败立即返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

Tiling 从 platform/compile info 取得 AIV 核数和 UB 字节数。POD 与 executor 在第一段构造，caller 按返回值分配 workspace；第二段把 workspace、executor 和 stream 交给框架 launcher。Tiling 固定写 `workspaces[0]=32`，但 entry 把 `userWS` 强制设为 `nullptr`，因此这 32 B 仅为框架预留。输入、ScalarList、输出和 workspace 必须至少存活到 stream 中该 Kernel 完成；固定仓不公开 executor 内部释放代码。

### 启动图与失败传播

```text
aclnnForeachMulScalarListGetWorkspaceSize (自动 bridge，固定仓不可见)
  -> ForeachMulScalarList OpDef / Infer
  -> Tiling4ForeachMulScalarListTiling
  -> ForeachMulScalarListMembaseTiling::DoOpTiling
  -> 生成 ForeachCommonTilingData、key=1/2/3/4、blockDim、workspace=32 B
aclnnForeachMulScalarList(workspace, 32, executor, stream)
  -> 单节点 K0: foreach_mul_scalar_list(inputs, scalar, outputs, workspace, tiling)
     execution=AIV_VECTOR, blockDim=needCoreNum, POD=ForeachCommonTilingData
```

无 Kernel 间边。K0 直接读 GM 输入/标量并写 GM 输出；同一 stream 的可见性、launch 返回与 executor 失败传播属于自动框架边界。Tiling 任一步 `GRAPH_FAILED` 都阻止 launch；entry 对未知 key 无 body、无输出。

## Route 与参数设计

### 分核与资源计算

令 `s=sizeof(T)`，`e=32/s` 元素/32 B block，`N=sum_i numel(x_i)`。`needCoreNum=min(max(1,ceil(N/e)), platformAivCoreNum, 50)`。把 `ceil(N/e)` 个 32 B block 尽量均分：前 `remainder` 个核多一个 block；每核由 `[tensorStart,tensorEnd]` 和首尾元素 offset 描述，可跨越多个 Tensor。

非 BF16：`tilingBytes=ForeachCommonTilingData::GetDataSize()`，`U=floor32((UB-tilingBytes)/2)` 字节。entry 实例把 `bufferNum=1`，所以 VECIN 与 VECOUT 各分配 `U`，总计 `2U<=UB-tilingBytes`。BF16：`U=floor64(((UB-tilingBytes)/10)/2)`，Kernel 为输入/输出分别分配 `9U`，另分配 `U` FP32 临时区，总计 `19U<=19/20*(UB-tilingBytes)`。`maxDataCount=U/s`（非 BF16）或 `9U/2`（BF16）；BF16 单次 cast 子块为 `U/4` 个 FP32 元素。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | key 1；`ForeachOneScalarListBinary<half,half,MulsAdapter<half>,1,1>` | A2/A3，`x/out=FP16`、`scalars=FP32` | dtype 不匹配则下一 key；Host 非支持组合失败 |
| 2 | R-002 FP32 | key 2；`<float,float,...,1,1>` | A2/A3，`x/out=FP32`、`scalars=FP32` | 下一 key/失败 |
| 3 | R-003 INT32 | key 3；`<int,int,...,1,1>` | A2/A3，`x/out=INT32`、`scalars=INT64`，且编译目标 `__CCE_AICORE__>=220` | 下一 key/失败 |
| 4 | R-004 BF16 | key 4；`<bfloat16_t,float,...,1,1>` | A2/A3，`x/out=BF16`、`scalars=FP32`，`__CCE_AICORE__>=220` 且非 NPU arch 3003/3113 | 未编译时无 body；Host/config 不应下发 |

```text
priority 10000 Membase: !IsRegbaseSocVersion(context) ? continue : GRAPH_PARAM_INVALID
  -> dtype FP16 -> key 1 / R-001
  -> dtype FP32 -> key 2 / R-002
  -> dtype INT32 -> key 3 / R-003
  -> dtype BF16 -> key 4 / R-004
  -> other -> key 0，entry 无匹配 body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | A2/A3；合法列表、0–8 维；FP16/FP32 | 单 K0 AIV_VECTOR | `KernelForeachUnary<half,...>`→`InnerComputer<half,half>`→`MulsAdapter<half>` | 1×VECIN + 1×VECOUT；Queue 和两次 `PIPE_V` barrier | 非整 tile 用 `DataCopyPad`；`Muls` 元素数为有效 count |
| R-002 FP32 | 2 | 同上，FP32/FP32 | 单 K0 | `InnerComputer<float,float>` | 同 R-001 | 同 R-001 |
| R-003 INT32 | 3 | 同上，INT32/INT64 | 单 K0 | `InnerComputer<int,int>` | 标量先转 `int32`；其余同 R-001 | 同 R-001；超出 INT32 的 scalar 窄化语义由转换决定 |
| R-004 BF16 | 4 | 同上，BF16/FP32；编译 guard 成立 | 单 K0 | BF16 `InnerComputer` specialization | 输入/输出各 9U + FP32 U；BF16→FP32→Muls→BF16 RINT | 外层 GM tail + 内层 cast tail；arch guard 不满足无 route |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 非 regbase | `!IsRegbaseSocVersion(context_)` | 选择旧式 Queue/Vector body 与 key 1–4 | 全部 | Registry 尝试 priority 30000 | `foreach_tiling_def.h::MEMBASE_TILING::IsCapable` |
| 32 B 分核 | `elementsPerBlock=32/s` | core 边界按 DMA block 切分 | 全部 | 无 fallback；公式决定 blockDim | `ForeachCommonTiling::GetNeedCoreNum/AssignDataToEachCore` |
| UB 对齐 | 非 BF16 `U%32=0`；BF16 `U%64=0` | Queue 分配与 BF16 cast 子区需要对齐 | 全部/R-004 | tile 变小；源码无其它 route | `DivideUbMemory7` |
| 固定数组/核数 | Tiling 只遍历前 50 个 Tensor；`blockDim<=50` | POD Tensor/core 数组均固定 50 项 | 全部 | 核数截断；源码未显式拒绝 `x` 列表长度>50 | `MAX_TENSOR_CONT/MAX_CORE_CONT`、`ForeachCommonTiling::Init` |
| BF16 编译 guard | `__CCE_AICORE__>=220 && arch!=3003/3113` | BF16 specialization 和临时 FP32 Queue 条件编译 | R-004 | entry 无 key 4 body | `foreach_mul_scalar_list`、`KernelForeachUnary::Init` |
| 空域 | `needCoreNum` 至少 1；每段 `dataCount` 可为 0 | 保持合法 launch，tile loop 次数为 0 | 全部 | no-op，无写出 | `GetNeedCoreNum`、`SingleTensorProcess` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 声明首字段；宏展开后的 sizeof/alignof/offsetof 固定仓不可见 | 字节 | `DivideUbMemory7`→`FillTilingData` | `KernelForeachBase::ParseTilingData`，决定 Queue 和 tile | 32 B 对齐；BF16 为 64 B 对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 次字段；精确 offset 不可见 | 元素 | `Init` 最多收集前 50 个 storage shape size | `Process` 计算每 Tensor 末端 | 公开 A2/A3 文档未声明列表上限；>50 的尾项不会进入 POD |
| 同上 | `tensorStartList` | `uint16_t[50]` | 第三字段；精确 offset 不可见 | Tensor 索引 | `AssignDataToEachCore` | 当前 block 读取 `[blockIdx]` | 仅活动 block 有效 |
| 同上 | `tensorEndList` | `uint16_t[50]` | 第四字段；精确 offset 不可见 | Tensor 索引 | 同上 | 同上 | inclusive |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 第五字段；精确 offset 不可见 | 元素 | 同上 | 当前 block 首 Tensor offset | inclusive |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 第六字段；精确 offset 不可见 | 元素 | 同上 | 当前 block 末 Tensor offset | inclusive；空段为 -1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1 | `foreach_tiling_func.cpp::Tiling4...`、`foreach_tiling_class.h::RunBigScalarKernelTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_mul_scalar_list.cpp::foreach_mul_scalar_list` | `KernelForeachUnary`、`ForeachOneScalarListBinary<half,...>`、`MulsAdapter` | `ascend910b/...binary.json` + `op_kernel/CMakeLists.txt` |
| R-002 | 2 | 同上 | 同上 | 同上 | `<float,...>` | 同上 |
| R-003 | 3 | 同上 | 同上 | 同上 | `<int,...>` | 同上 |
| R-004 | 4 | 同上 | 同上 | 同上 | BF16 `InnerComputer` specialization | 同上 |

## Kernel 实现

### 公共执行流程

entry 声明 AIV-only，解析 POD，忽略 workspace，再按 key 构造一个 CRTP `ForeachOneScalarListBinary`。`KernelForeachBase` 取得 `blockIdx` 和本核 Tensor/offset 区间；`KernelForeachUnary::Process` 逐 Tensor 解析指针表、每 Tensor 只读取一次对应 scalar，再按 `maxDataCount` 切 tile。每 tile 为 GM→VECIN Queue→`InnerComputer/Muls`→VECOUT Queue→GM。输入 Queue 的 EnQue/DeQue 与输出 Queue 的 EnQue/DeQue/FreeTensor 建立阶段生命周期；Kernel 内无跨核通信和 workspace 边。

### Route R-001：FP16 直接 Muls

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_tiling_def.h` | `ForeachMulScalarListMembaseTiling::IsCapable` | 非 regbase | membase |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `op_kernel/foreach_mul_scalar_list.cpp` | `foreach_mul_scalar_list` | `TILING_KEY_IS(1)` | `<half,half,...,1,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar_list -> ForeachOneScalarListBinary<half,...>::Init/Process
  -> KernelForeachUnary::Process -> ProcessPlusInLoop(scalar[index])
  -> SingleTensorProcess -> CopyIn -> Compute -> InnerComputer::Compute
  -> MulsAdapter<half> -> Muls -> CopyOut
```

##### 源码等价伪码

```text
for tensor i in [start,end]:
  begin/end = POD 首尾 offset；scalar = half(scalars[i])
  for tile k: n=min(maxDataCount, remain)
    GM[x_i+begin+k*max] -> inQ(n)
    outQ = Muls(inQ, scalar, valid=n)
    outQ(n) -> GM[y_i+begin+k*max]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x_i/y_i` | descriptor pointer + core offset + `k*maxDataCount` | 本 tile `n` 元素 | FP16/ND-flat | caller/Kernel | CopyIn/caller | caller；stream 完成前有效 | full tile DataCopy；tail DataCopyPad |
| GM scalar | `scalars+i` | 1 FP32 | FP32 list | caller | `ProcessPlusInLoop` | caller | 每 Tensor 读一次并转 half |
| `dataQueue` | Queue 管理 | `U` B，depth 1 | FP16 | CopyIn | Compute | TPipe | 与下一 tile复用 |
| `outQueue` | Queue 管理 | `U` B，depth 1 | FP16 | Compute | CopyOut | TPipe | CopyOut 后 Free |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = U + U = 2U <= UB - tilingDataBytes
dynamic_ub = N/A
workspace_total = 32 B reserved；Kernel 实际使用 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQ slot | CopyIn | Compute/DeQue | 下一 tile Alloc | FreeTensor | Compute 完成后释放 |
| outQ slot | Compute | CopyOut/DeQue | 下一 tile Alloc | FreeTensor | GM CopyOut 提交后释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue EnQue/DeQue | CopyIn | Compute | inQ | tail 按 `n*sizeof(half)`，不 pad 值 |
| Vector | `PipeBarrier<PIPE_V>` 前后 | load/前序 V | Muls/后序 V | Vector pipeline | scalar FP32→FP16 后以 FP16 Muls；有效元素数 `n` |
| UB→GM | outQ EnQue/DeQue | Compute | CopyOut | outQ | 仅写 `n` 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `RunBigScalarKernelTiling/DivideUbMemory7` | key/core/U/workspace |
| Kernel entry/router | `op_kernel/foreach_mul_scalar_list.cpp` | key 1 branch | FP16 模板映射 |
| Device body/helper | `kernel_foreach_unary.h`、`foreach_one_scalar_list_binary.h` | `Process/Compute/ProcessPlusInLoop` | tile、Queue、标量与 Muls |

### Route R-002：FP32 直接 Muls

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_mul_scalar_list.cpp` | key 2 branch | `TILING_KEY_IS(2)` | `<float,float,...,1,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar_list -> ForeachOneScalarListBinary<float,...>::Process
  -> CopyIn -> InnerComputer<float,float>::Compute -> MulsAdapter<float> -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 相同；scalar=scalars[i]，输入/输出/Muls 全程 FP32。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM + inQ/outQ | 同 R-001 | GM `n` 元素；Queue 各 `U` B | FP32/flat | CopyIn/Compute | Compute/CopyOut | caller/TPipe | depth 1，tail `n*4` B |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=2U；workspace_total=32 B reserved/0 B consumed。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQ/outQ | CopyIn/Compute | Compute/CopyOut | 下一 tile | Queue FreeTensor | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | Queue + `PIPE_V` barrier | CopyIn/Muls | Muls/CopyOut | Queue/V pipeline | FP32 Muls；tail 仅处理 `n` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | key/core/U | 与 R-001 共用 producer |
| Kernel entry/router | `op_kernel/foreach_mul_scalar_list.cpp` | key 2 branch | FP32 mapping |
| Device body/helper | `foreach_one_scalar_binary.h` | generic `InnerComputer` | 两次 V barrier 包围 FP32 Muls |

### Route R-003：INT32 Muls

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `op_kernel/foreach_mul_scalar_list.cpp` | key 3 branch | `__CCE_AICORE__>=220 && TILING_KEY_IS(3)` | `<int,int,...>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> ForeachOneScalarListBinary<int,int>::Process
  -> scalarVal=int(inScalarGM.GetValue(i)) -> CopyIn -> Muls<int> -> CopyOut
```

##### 源码等价伪码

```text
scalar32 = int32(scalars_int64[i]); for each valid tile: y = Muls<int32>(x, scalar32, n)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| scalar GM | `scalars+i` | 1 INT64 | INT64 | caller | scalar load | caller | 转为 INT32 后复用一 Tensor |
| inQ/outQ | 同 R-001 | 各 `U` B | INT32 | CopyIn/Compute | Muls/CopyOut | TPipe | depth 1 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=2U；workspace_total=32 B reserved/0 B consumed。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slot | CopyIn/Compute | Muls/CopyOut | 下一 tile | FreeTensor | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 标量 | C++ cast | INT64 GM load | INT32 Muls | scalar value | INT64→INT32 窄化；固定仓无显式范围校验 |
| Vector/tail | Queue + barrier | input | output | valid `n` | INT32 Muls；溢出细则属于 primitive 边界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | key 3/core/U | INT32 route |
| Kernel entry/router | `op_kernel/foreach_mul_scalar_list.cpp` | key 3 branch | 编译 guard + int template |
| Device body/helper | `foreach_one_scalar_list_binary.h` | `ProcessPlusInLoop` | `T(inScalarGM.GetValue(index))` 窄化 |

### Route R-004：BF16 升精度 Muls

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `op_kernel/foreach_mul_scalar_list.cpp` | key 4 branch | CCE>=220 且非 arch 3003/3113 | BF16/FP32 specialization |

#### Device 数据流

##### 函数级调用链

```text
entry -> ForeachOneScalarListBinary<bfloat16_t,float>::Process
  -> GM BF16 -> inQ -> Cast(CAST_NONE) to float32Queue
  -> Muls<float> -> Cast(CAST_RINT) to outQ BF16 -> GM
```

##### 源码等价伪码

```text
for tile n:
  load BF16; for subchunk m<=U/4: fp32=Cast(bf16); fp32*=scalar_fp32; bf16=Cast_RINT(fp32)
  copy exactly n BF16 elements out
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQ | Queue | `9U` B | BF16 | CopyIn | Cast | TPipe | depth 1 |
| outQ | Queue | `9U` B | BF16 | Cast_RINT | CopyOut | TPipe | depth 1 |
| float32Queue | offset 0（`paramsCount=1`） | `U` B | FP32 | Cast | Muls/Cast_RINT | depth 1，Process 全程持有 | 每 subchunk 原地覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["dataQueue · depth=1 · 9U bytes"]:1
  fp32["float32Queue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · 9U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=9U+U+9U=19U <= 19/20*(UB-tilingDataBytes)
workspace_total=32 B reserved/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 temp offset 0 | BF16→FP32 Cast | FP32 Muls + BF16 Cast | 下一 subchunk | `PIPE_V` barriers | 后一 Cast 只能在回写后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast→Muls→Cast | 三段间 `PIPE_V` barrier | Cast/Muls | Muls/Cast | FP32 temp | BF16 升 FP32，FP32 乘，`CAST_RINT` 降回；subchunk remainder 单独处理 |
| GM tail | DataCopyPad | GM/UB | Compute/GM | 有效 BF16 | 仅 `n` 元素参与/写回 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory7` | BF16 1/10 衰减、1/2 Queue 预算、64 B 对齐 |
| Kernel entry/router | `op_kernel/foreach_mul_scalar_list.cpp` | key 4 branch | BF16/FP32 mapping 与 arch guard |
| Device body/helper | `foreach_one_scalar_binary.h` | BF16 `InnerComputer` specialization | 分段 cast/Muls/RINT |

## 算子特定优化

### Trick T-001：每个 Tensor 只加载一次广播标量

#### 触发与源码依据

- 适用 Route：R-001–R-004
- 触发条件：处理核进入 Tensor `index` 后、该 Tensor 的 tile 循环之前
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

`KernelForeachUnary::Process` 在 `SingleTensorProcess` 的 tile 循环外调用该 hook，因此 scalar 的 GM load 与 dtype 转换每个 Tensor 只做一次，后续所有 tile 复用成员 `scalarVal`；这减少了长 Tensor 上重复的 scalar GM 读取。BF16 保持 FP32 scalar，其他路线按计算 dtype 转换。

#### 迁移边界

- 适用：同一 Tensor 内共享一个标量、且 core 对该 Tensor 的分片内 scalar 不变的 foreach unary-scalar 算子。
- 前提：ScalarList 与 TensorList 等长；scalar 在 Kernel 期间只读；每次切换 Tensor 必须刷新。
- 失效：元素级 scalar、tile 内标量变化或异步修改 scalar 时不能复用；应改为相应逐元素输入路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 3 | key 4 + guard | dtype-only key |
| 模板与 dtype | half/half | float/float | int/int | bf16/float | scalar/计算精度 |
| 循环与分核 | 32 B block，U/s tile | 同 | 同 | 32 B block，9U/2 外 tile + U/4 cast tile | BF16 临时区 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | in/out各U | 同 | 同 | in/out各9U + FP32 U | BF16 升精度 |
| 内存总占用与复用 | 2U | 2U | 2U | 19U | Host UB 预算匹配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Pad + Muls | 同 | 同 | DataCopy/Pad + Cast/Muls/Cast | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue + V barrier，FP16 | FP32 | INT64 scalar→INT32 | FP32 计算、RINT | dtype 语义 |
| fallback/失败边界 | 非法 dtype/shape 失败 | 同 | CCE guard | arch guard | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开支持，见独立 `_950` 卡 |
| Host/Launcher | 共用 membase priority 10000、key 1–4、32 B 分核 | 同 A2 | regbase priority 30000、key 10001–10004、4 KiB 分核 |
| Kernel/资源 | 旧式 Queue/Vector body；最多 50 核 | 同 A2 | arch35 MicroAPI body；最多 80 核；ListTensorDesc |

## 限制与失败边界

- ScalarList 长度、列表内 dtype、0–8 维、输出 dtype/shape 不满足约束时 Host/Tiling 失败；未知 key 无 Kernel body。
- A2/A3 公开文档未声明 TensorList 最大长度，而 membase Tiling 的收集循环和 POD 仅有 50 项，源码也未显式拒绝长度>50；因此不得把超过 50 个 Tensor 的调用视为闭合支持。长度为 0 时 Tiling 也未显式拒绝，Kernel 随后仍解析 index 0 的 Tensor 地址，故空 TensorList 同样不属于可安全复用边界（单个空 Tensor 则有 no-op 路径）。
- INT32 route 把公开 INT64 scalar 显式转为 `int`，固定仓未见范围校验；调用方复用时不得忽略该窄化。
- Kernel 中 `inScalarGM.SetGlobalBuffer(..., 1)` 后按 Tensor `index` 读取；固定源码只证明地址和索引行为，GlobalTensor 长度元数据是否参与运行时检查属于 AscendC primitive 边界。
- 固定仓未公开自动生成 ACLNN bridge/executor/launch 壳的精确符号、executor 内部 owner 和 stream 提交实现；这些仅作为框架边界，不据此推测。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_mul_scalar_list/docs/aclnnForeachMulScalarList.md` | 两段式原型、产品表 | API、dtype、shape、错误语义 |
| 入口实现 | `foreach/foreach_mul_scalar_list/op_host/foreach_mul_scalar_list_def.cpp` | `ForeachMulScalarList`、`OP_ADD` | 内部参数 schema、平台注册、AutoContiguous |
| Infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`、注册 | 输出 shape/dtype |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachMulScalarListTiling`、MEMBASE 注册 | Registry、workspace |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD producer |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | 字段/数组长度 |
| Host helper | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key 1–4 |
| Host selector | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 升序与失败传播 |
| Kernel entry/body | `foreach/foreach_mul_scalar_list/op_kernel/foreach_mul_scalar_list.cpp` | `foreach_mul_scalar_list`、`MulsAdapter` | AIV entry、key router |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD 解析、Tensor 地址 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、循环、CopyIn/Out |
| Kernel helper | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `InnerComputer` | Muls 与 BF16 cast specialization |
| Kernel helper | `foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary.h` | `ForeachOneScalarListBinary` | ScalarList load 与 CRTP Compute |
| 构建接线 | `foreach/foreach_mul_scalar_list/op_host/config/ascend910b/foreach_mul_scalar_list_binary.json` | 四组 `op_list` | A2 dtype binary 实例 |
| 构建接线 | `foreach/foreach_mul_scalar_list/op_host/config/ascend910_93/foreach_mul_scalar_list_binary.json` | 四组 `op_list` | A3 dtype binary 实例 |
| 构建接线 | `foreach/foreach_mul_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 默认 Kernel 接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段分核描述 | 字段顺序、位宽、50 项数组、`inputsTensorUbSize` 为字节 | `ForeachCommonTilingData` |
| 调用顺序 | 每 Tensor load scalar，再逐 tile CopyIn→Compute→CopyOut | scalar hook 不得移入元素循环或遗漏 Tensor 切换 | `KernelForeachUnary::Process` |
| workspace/内存 | Kernel 无临时 workspace；Queue 字节账可复用 | BF16 必须保留 19U 与 cast 子块关系 | `DivideUbMemory7`、`InitBuffer` |
| 同步 | Queue 生命周期 + `PIPE_V` barriers | 释放/覆盖不得早于 consumer | `InnerComputer::Compute` |
| 精度与尾块 | FP16/FP32/INT32 直接 Muls；BF16 FP32 中间 | BF16 RINT、INT64→INT32、tail 有效长度 | `InnerComputer`、`CopyIn/CopyOut` |
| 失败边界 | Host 校验失败阻止 launch，unknown key 无 body | 不得把不支持 dtype 自动映射到相邻 key | `CheckOpInfo`、entry router |