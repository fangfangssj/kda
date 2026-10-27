---
type: Operator
title: "ForeachPowScalar A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachPowScalarV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/docs/aclnnForeachPowScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/op_api/foreach_pow_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/foreach_pow_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/arch22/foreach_pow_scalar_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:47:57Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachPowScalar A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachPowScalarV2GetWorkspaceSize` / `aclnnForeachPowScalarV2`，公开 ACLNN V2 |
| 计算语义 | 对列表中每个张量逐元素计算 `y_i[j] = pow(x_i[j], exponent)` |
| 输入输出 | 公开契约：`x/out` 为 ND TensorList，FLOAT16/FLOAT32/BFLOAT16，秩 0–8；`scalar` 为 `aclScalar` |
| 执行类型 | 单 Kernel，AIV_VECTOR；无 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | R-001 FP16；R-002 FP32；R-003 BF16→FP32→BF16 |
| 核心资源 | GM TensorList 指针表、AIV、`TQue<VECIN/VECOUT>`；BF16 另有 FP32 Queue；算子 Kernel 不使用 workspace |
| 来源闭包 | source_closed：公开 V2 API→ACLNN 参数处理→L0 launcher→arch22 Tiling/POD→Kernel router→公共 foreach body→构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`，输出 `y` 与列表位置一一对应；每个有效元素执行 `y_i[j]=x_i[j]^exponent`。各张量彼此独立，Host 可把一个张量切给多个 AIV，也可让一个 AIV 连续处理多个张量片段。公开文档声明默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；各 tensor ND，秩 0–8 | FLOAT16/FLOAT32/BFLOAT16 | 列表内 dtype 一致；公开文档标记非连续 Tensor 不支持；空 Tensor 支持 |
| `scalar` | `aclScalar`，无 tensor shape/layout | x=FP16 时 FP16/DOUBLE；x=FP32/BF16 时 FLOAT/DOUBLE | `abs(exponent) <= 2^31-1`；API 将其转换为一个元素的 Device tensor；DOUBLE 会窄化到计算 dtype |
| `out` | TensorList；ND | 与 `x` 相同 | 列表长度由调用者配对；API 实现逐项要求 shape 精确相等，公开文档只写 shape size 不小于输入，复用应遵守实现的更强约束 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段接口生成；源码有空列表返回 0 workspace 的分支，但该分支位于一次 element-0 dtype 读取之后，不能据此证明零长度列表安全 |
| `workspace/workspaceSize/executor/stream` | 第二段执行参数 | 框架类型 | `CommonOpExecutorRun` 在指定 stream 执行已生成流程 |

### 入口与版本

版本族包括旧版 `aclnnForeachPowScalar`（`scalar` 是单元素 `aclTensor`）与 V2（`scalar` 是 `aclScalar`）。V2 文档同时公开支持 A2、A3、950，且源码中存在完整 V2 实现，因此 A2/A3 取最新的 `aclnnForeachPowScalarV2`；其 API 把 `aclScalar` 转成 tensor 后复用内部 `ForeachPowScalar` body。旧版不进入本卡。

V2 文档只公开 FLOAT16/FLOAT32/BFLOAT16。ACLNN 实现的 `ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST` 与 Kernel router 还含 INT32，但 header/doc 未把它纳入 V2 契约；该实现信号不扩张公开 API Route。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecForeachPowScalarV2GetWorkspaceSize` 依次：创建 executor；检查 `x/scalar/out` 非空；按当前 NPU 架构取得 dtype 集；检查列表内 dtype、输出 dtype、scalar dtype 对应；逐输入检查秩不超过 8 且输入/输出 shape 精确相等；拒绝私有 storage format。参数检查成功后，源码对 `x->Size()==0` 返回空 executor 和 0 workspace；但 `CheckDtypePowScalar` 在这个分支之前已经读取 `(*x)[0]`，所以固定仓不能证明零长度列表安全。非空时逐 tensor 调用框架 `l0op::Contiguous` 并组装新的 TensorList；随后 `ConvertToTensor` 把 scalar 转为 x dtype，唯 BF16 转为 FLOAT；最后 `l0op::ForeachPowScalarV2` 用 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachPowScalar, OP_INPUT(x, scalar), OP_OUTPUT(out))` 加入一个 Device 节点。Contiguous、scalar tensor 分配和 executor 调度属于框架边界。

### 参数、Workspace 与生命周期

ACLNN 返回的 workspace 是整个 executor 的 `GetWorkspaceSize()`，可能包含框架 Contiguous/转换节点需要的空间；由调用者持有至第二段接口完成。算子 Tiling 自身写 `workspaces[0]=32`，但 Kernel entry 显式把 `userWS=nullptr`，所以 32 字节是公共 tiling 的系统占位，非算子数据交换区。输入 contiguous tensor、scalar tensor 和 launcher 参数均归 executor 所有，随 executor 流程在传入 stream 上存活。

### 启动图与失败传播

```text
aclnnForeachPowScalarV2GetWorkspaceSize
  -> [每个 x] l0op::Contiguous（仅框架需要时产生前置节点）
  -> ConvertToTensor(scalar, x_dtype；BF16 改用 FLOAT)
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachPowScalar)
     -> arch22 Tiling4ForeachPowScalarTiling
     -> foreach_pow_scalar<<<blockDim, stream>>>
aclnnForeachPowScalarV2
  -> CommonOpExecutorRun(workspace, executor, stream)
```

算子核心只有一个 AIV Kernel 节点：输入是 contiguous TensorList 指针表和单元素 scalar tensor，输出直写 `out`，POD 为 `ForeachCommonTilingData`，blockDim 为 `needCoreNum`。没有 Kernel 间 edge。executor 保证前处理节点与核心节点在同一执行流程中的顺序；任一 `CHECK_RET` 或 launcher 添加失败会在执行前返回，不留下可执行核心节点。

## Route 与参数设计

### 分核与资源计算

令 `E=sum_i numel(x_i)`，`s=sizeof(T)`，`epb=32/s` 元素。Host 计算 `needCoreNum=min(AIV_core_num, ceil(E/epb), 50)`，若 `E=0` 则取 1。`AssignDataToEachCore` 先把总元素按 32 字节 block 均分：前 `ceil(E/epb) mod needCoreNum` 个核多一个 block；再生成每核 `[tensorStart,tensorEnd]` 与首尾 tensor 的闭区间 offset，使一个 core 可以跨 tensor，一个 tensor 也可以跨 core。

设平台 UB 字节为 `U`，POD 序列化字节为 `D`，`N(dtype)` 对 FP16/FP32/BF16 分别为 14/4/14，Power 预留 `R=1024*N*s`。FP16/FP32 的每 Queue 容量 `I=floor32((U-D-R)/4)` 字节。BF16 先取 `(U-D-R)/10/4` 再向下 64 字节对齐得到 `I`；Device 将输入、输出 Queue 各扩大为 `9I`，另分配 `2I` 的 FP32 Queue。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | key 1；`ForeachOneScalarBinary<half,half,PowerAdapter<half>,1>` | V2 公开 dtype 为 FLOAT16；scalar 由 ACLNN 转成 FLOAT16 tensor | 检查下一 dtype |
| 2 | R-002 FP32 | key 2；`ForeachOneScalarBinary<float,float,PowerAdapter<float>,1>` | V2 公开 dtype 为 FLOAT32；scalar 转成 FLOAT32 tensor | 检查下一 dtype |
| 3 | R-003 BF16 | key 4；`ForeachOneScalarBinary<bfloat16_t,float,PowerAdapter<float>,1>` | V2 公开 dtype 为 BFLOAT16；scalar 强制转成 FLOAT32 tensor；非 3003/3113 编译 guard | 未知/不支持 key 时 entry 不执行 body |

```text
API dtype/shape/format 检查失败 -> ACLNN_ERR_PARAM_INVALID
else FLOAT16 -> key 1 -> R-001
else FLOAT32 -> key 2 -> R-002
else BFLOAT16 -> key 4 -> R-003
else -> 不属于 V2 公开契约
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | A2/A3；x/out FP16；scalar FP16/DOUBLE；公开 shape/format 约束通过 | 单 AIV Kernel | `Power<half,false>`，公共 unary foreach | depth-1 输入/输出 Queue；Power 前后 V pipe barrier | 对齐 tile 用 `DataCopy`，末 tile 用字节精确 `DataCopyPad` |
| R-002 FP32 | 2 | A2/A3；x/out FP32；scalar FLOAT/DOUBLE；约束通过 | 单 AIV Kernel | `Power<float,false>` | 同 R-001，元素容量按 4 字节换算 | 同 R-001 |
| R-003 BF16 | 4 | A2/A3；x/out BF16；scalar FLOAT/DOUBLE；约束通过；编译架构非 3003/3113 | 单 AIV Kernel | 分片 `Cast(BF16→FP32)`→`Power<float,false>`→`Cast(FP32→BF16, CAST_RINT)` | depth-1 的 9I 输入/输出 Queue，单 depth FP32 Queue 2I；V pipe barrier | GM tail 同上；内部 cast 再按 `I/4` 个 FP32 元素分片 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32B 分核 | `epb=32/sizeof(T)`；按 `ceil(E/epb)` 分核 | 完整 tile 的 `DataCopy` 以元素数搬运，核边界保持 block 粒度 | 全部 | 最后有效片由 `DataCopyPad` 处理 | `ForeachCommonTiling::GetNeedCoreNum/AssignDataToEachCore` |
| 最大 50 tensor/core | 初始化循环与 POD 数组均为 50；core 再 `min(...,50)` | POD 的列表和分核数组是定长 ABI | 全部 | 超出 50 的输入不会进入数组；公开 ACLNN 文档未给出此上限，应视为实现限制 | `MAX_TENSOR_CONT/MAX_CORE_CONT` |
| Power UB 预留 | `U > D + 1024*N(dtype)*s` 才能得到正的无符号剩余量 | Host 为 Power 计算预留空间，再按 4 分配 Queue 预算 | 全部 | 源码无显式小 UB 失败 guard；无符号减法下溢风险必须由平台资源前提避免 | `DivideUbMemory6` |
| BF16 临时区 | dtype=BF16 且编译 guard允许 | BF16 必须用 FP32 分片计算，Queue 放大 9 倍并需 2I FP32 区 | R-003 | 3003/3113 不编译 key 4 body | `KernelForeachBase::Init`、`KernelForeachUnary::Init` |
| 空域 | 参数检查后有空列表返回分支；非空列表但 E=0 时 blockDim=1 | 避免 0 block；每 tensor 的闭区间须谨慎 | 全部 | element-0 dtype 读取先于空列表分支；全空 tensor 又缺少专门 Device guard，均不能证明安全 | `CheckDtypePowScalar/ExecForeachPowScalarV2GetWorkspaceSize/GetNeedCoreNum` |

### POD/ABI

`ForeachCommonTilingData` 由仓外注册宏展开；固定仓能证明字段顺序、类型、数组长度和 producer/consumer，但宏展开后的 `sizeof/alignof/offsetof` 不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | uint64/64 | 首字段；精确 offset/sizeof 固定仓不可见 | 字节 | `DivideUbMemory6`；`set_inputsTensorUbSize(I)` | `KernelForeachBase::ParseTilingData`；Queue 容量和 tile 上限 | 32B 对齐；BF16 为 64B 对齐 |
| 同上 | `tensorDataCountList` | `int64[50]` | 第二字段；精确 offset 不可见 | 元素 | `Init`: 每 tensor storage shape size | `Process`: 每 tensor `cursorEnd` | 最多 50 项 |
| 同上 | `tensorStartList` | `uint16[50]` | 第三字段；精确 offset 不可见 | tensor 下标 | `AssignDataToEachCore` | 当前 block 的起始 tensor | 由 blockIdx 索引 |
| 同上 | `tensorEndList` | `uint16[50]` | 第四字段；精确 offset 不可见 | tensor 下标 | `AssignDataToEachCore` | 当前 block 的结束 tensor（闭区间） | 由 blockIdx 索引 |
| 同上 | `tensorStartOffsetList` | `int64[50]` | 第五字段；精确 offset 不可见 | 元素 | `AssignDataToEachCore` | 首 tensor 的 GM 基址偏移 | 闭区间起点 |
| 同上 | `tensorEndOffsetList` | `int64[50]` | 第六字段；精确 offset 不可见 | 元素 | `AssignDataToEachCore` | 尾 tensor 的闭区间终点 | `dataCount=end-start+1` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 1 | `arch22/...tiling...cpp::Tiling4ForeachPowScalarTiling`；`ForeachCommonTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_pow_scalar.cpp::foreach_pow_scalar` | `ForeachOneScalarBinary<half,...>`；`KernelForeachUnary` | Host/Kernel CMake: 910b、910_93 |
| R-002 FP32 | key 2 | 同上 | 同上 | 同上 | `ForeachOneScalarBinary<float,...>` | 同上 |
| R-003 BF16 | key 4 | 同上 | 同上 | 同上 | `ForeachOneScalarBinary<bfloat16_t,float,...>` 与 BF16 `InnerComputer` | 同上 |

## Kernel 实现

### 公共执行流程

entry 用 `GET_TILING_DATA` 解析 POD，按 key 实例化一个 `ForeachOneScalarBinary`，workspace 传空。`KernelForeachBase` 用 `GetBlockIdx()` 取得当前核的 tensor/offset 闭区间；TensorList GM 首字存地址表偏移，右移 3 转为 `uint64_t` 索引后取第 i 个 tensor 指针。每个 tensor 片段按 `maxDataCount` 分 tile，顺序执行 `CopyIn→Compute→CopyOut`。完整 tile 走 `DataCopy`，最后不足 tile 走单 block、精确字节 `DataCopyPad`。Queue 的 EnQue/DeQue/FreeTensor 加上 Vector pipe barrier约束生产、消费和复用。

### Route R-001：FP16 原生 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachPowScalarV2.md` | dtype 表 | x/out FLOAT16；scalar FLOAT16/DOUBLE | 公开支持 |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp` | `TILING_KEY_IS(1)` | key=1 | half 模板 |

#### Device 数据流

##### 函数级调用链

```text
Tiling4ForeachPowScalarTiling -> RunBigScalarKernelTiling
foreach_pow_scalar(key=1)
  -> ForeachOneScalarBinary<half,half,PowerAdapter<half>,1>::Init
  -> KernelForeachUnary::Process
     -> CopyIn -> InnerComputer::Compute -> PowerAdapter -> Power<half,false> -> CopyOut
```

##### 源码等价伪码

```text
scalarVal = *(half*)scalar
for tensor i in [tensorStart[blockIdx], tensorEnd[blockIdx]]:
  start = (i == first ? startOffset : 0)
  end = (i == last ? endOffset : numel[i]-1)
  bind xGM/yGM to tensorAddress(i)+start
  for tile k over count=end-start+1, tileCap=I/2 elements:
    n = min(tileCap, remaining)
    dataLocal = copy xGM[k*tileCap : +n]  # tail uses byte-exact DataCopyPad
    outLocal = alloc outQueue
    pipe_barrier(V); Power<half,false>(outLocal,dataLocal,scalarVal,n); pipe_barrier(V)
    free dataLocal; enqueue outLocal
    copy n elements to yGM[k*tileCap]; free outLocal
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y tensor | `TensorListAddr(i)+cursorStart+k*(I/2)` | 当前 tile `n` | half/ND | API/CopyOut | CopyIn/调用者 | caller；整个 Kernel | full tile 对齐；tail 精确字节 |
| `dataQueue` | Queue 分配 | depth 1 × I bytes | half | CopyIn | Compute | `TPipe`；FreeTensor 后复用 | tile 容量 I/2 元素 |
| `outQueue` | Queue 分配 | depth 1 × I bytes | half | Compute | CopyOut | `TPipe`；CopyOut 后释放 | 不与 input alias |
| scalar GM | offset 0 | 1 元素 | half | API ConvertToTensor | Init | executor | 每核读取一次 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · VECIN · depth=1 · I bytes"]:2
  outQ["outQueue · VECOUT · depth=1 · I bytes"]:2
```

##### 字节账与复用顺序

```text
R = 1024 * 14 * 2 = 28672 bytes
I = floor32((U - D - R) / 4)
UB_explicit_queues = I + I = 2I
Host additionally reserves R and leaves the remaining divider margin to the Power implementation/resource policy
dynamic_ub = N/A
workspace_kernel = 0; common tiling slot = 32 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue 当前 slot | CopyIn | Compute | 下一 tile CopyIn | EnQue→DeQue→FreeTensor | depth=1，必须消费后覆盖 |
| outQueue 当前 slot | Compute | CopyOut | 下一 tile Compute | EnQue→DeQue→FreeTensor | CopyOut 完成后释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue + DataCopy/DataCopyPad | GM x | Compute DeQue | dataQueue slot | tail 仅搬 `n*sizeof(half)` 字节 |
| Power | `PipeBarrier<PIPE_V>` | CopyIn/上一 Vector 指令 | `Power<half,false>`/CopyOut | Vector 数据依赖 | 直接 half 计算；固定仓未展开 Power 内部近似与舍入 |
| UB→GM | Queue + DataCopy/DataCopyPad | Power | GM y | outQueue slot | 只提交 n 个有效元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling/AssignDataToEachCore/DivideUbMemory6` | key、分核、I、POD、workspace |
| Kernel entry/router | `foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp` | key 1 branch | 映射到 half 模板，workspace 置空 |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_unary.h` | `Process/SingleTensorProcess/CopyIn/CopyOut` | tensor/core/tile 循环与 tail DMA |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary.h` | 通用 `InnerComputer`、`PowerAdapter<half>` | barrier→Power→barrier |

### Route R-002：FP32 原生 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachPowScalarV2.md` | dtype 表 | x/out FLOAT32；scalar FLOAT/DOUBLE | 公开支持 |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp` | `TILING_KEY_IS(2)` | key=2 | float 模板 |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar(key=2)
  -> ForeachOneScalarBinary<float,float,PowerAdapter<float>,1>::Init/Process
  -> KernelForeachUnary::{CopyIn,Compute,CopyOut}
  -> InnerComputer<float>::Compute -> Power<float,false>
```

##### 源码等价伪码

```text
scalarVal = *(float*)scalar
for assigned tensor slices and tiles of at most I/4 elements:
  CopyIn(n); barrier(V); Power<float,false>(out,in,scalarVal,n); barrier(V); CopyOut(n)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | tensor base + core start + `k*(I/4)` | n elements | float/ND | caller/CopyOut | CopyIn/caller | caller | tail byte count `4n` |
| dataQueue | Queue slot | I bytes | float | CopyIn | Power | `TPipe` | depth 1 |
| outQueue | Queue slot | I bytes | float | Power | CopyOut | `TPipe` | depth 1 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · I bytes"]
  outQ["outQueue · depth=1 · I bytes"]
```

##### 字节账与复用顺序

```text
R = 1024 * 4 * 4 = 16384 bytes
I = floor32((U - D - R) / 4)
UB_explicit_queues = 2I
dynamic_ub = N/A
workspace_kernel = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue/outQueue slot | CopyIn/Power | Power/CopyOut | 下一 tile | Queue EnQue/DeQue/FreeTensor | 单 slot 串行生命周期 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流水 | Queue + V barrier | CopyIn/Power | Power/CopyOut | 当前 tile | FP32 Power；tail 精确搬运 n 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `GetTilingN/DivideUbMemory6` | FP32 N=4，I 计算 |
| Kernel entry/router | `foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp` | key 2 branch | float 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `ForeachOneScalarBinary<float,...>` | scalar 读取和 Power 调用 |

### Route R-003：BF16 分片升精度 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_pow_scalar_v2.cpp` | `ConvertToTensor` | x dtype BF16 | scalar tensor 为 FLOAT |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `op_kernel/foreach_pow_scalar.cpp` | key 4 compile-guarded branch | 非 NPU arch 3003/3113 | BF16/float 模板 |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar(key=4)
  -> ForeachOneScalarBinary<bfloat16_t,float,PowerAdapter<float>,1>::Init
  -> KernelForeachUnary::Process
     -> CopyIn BF16
     -> BF16 InnerComputer::Compute
        -> ComputePerCast: Cast -> Power<float,false> -> Cast(CAST_RINT)
     -> CopyOut BF16
```

##### 源码等价伪码

```text
scalarVal = *(float*)scalar
tileCap = 9I/2 BF16 elements; castCap = I/4 FP32 elements
for each assigned tile of n elements:
  copy BF16 GM -> dataQueue
  for c in chunks(n, castCap):
    Cast(floatBuf[0], bf16In[c], CAST_NONE, m)
    Power<float,false>(floatBuf[castCap], floatBuf[0], scalarVal, m)
    Cast(bf16Out[c], floatBuf[castCap], CAST_RINT, m)
  copy only n BF16 elements -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dataQueue | slot | 9I bytes | BF16 | CopyIn | Cast | `TPipe` | depth 1；tileCap=9I/2 |
| outQueue | slot | 9I bytes | BF16 | final Cast | CopyOut | `TPipe` | depth 1 |
| float32Queue input half | offset 0 | I bytes = I/4 floats | FP32 | Cast | Power | 整个 Process | 与 output half 分区 |
| float32Queue output half | offset I bytes | I bytes = I/4 floats | FP32 | Power | final Cast | 整个 Process | `paramsCount=2` 决定 offset |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=1 · 9I bytes"]:2
  fIn["float32Queue[0:I] · I bytes"]:1
  fOut["float32Queue[I:2I] · I bytes"]:1
  outQ["outQueue · depth=1 · 9I bytes"]:1
```

##### 字节账与复用顺序

```text
R = 1024 * 14 * 2 = 28672 bytes
I = floor64(((U - D - R) / 10) / 4)
UB_total = 9I + 9I + 2I = 20I
UB_total <= U - D - R  （由 I 的 /40 预算保守保证，忽略向下对齐只会更小）
dynamic_ub = N/A
workspace_kernel = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| float32 input half | BF16→FP32 Cast | Power src | 下一 cast chunk | 每步间 `PipeBarrier<PIPE_V>` | Power 读取完成后才覆盖 |
| float32 output half | Power dst | FP32→BF16 Cast | 下一 cast chunk | V barrier 与顺序调用 | Cast 读取完成后才覆盖 |
| BF16 Queue slots | CopyIn/final Cast | Cast/CopyOut | 下一 tile | Queue FreeTensor | 当前 tile 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | Cast + V barrier | input Queue | Power | float input half | `CAST_NONE`，每片 m≤I/4 |
| Power | `Power<float,false>` + V barrier | FP32 Cast | output Cast | float output half | exponent 与 base 均 FP32 |
| FP32→BF16 | `Cast(...,CAST_RINT,m)` | Power | CopyOut | BF16 out Queue | 显式 round-to-nearest integer mode token；只转换 m 个有效元素 |
| GM tail | DataCopyPad | Queue | GM | tile tail | 只读写 `n*sizeof(BF16)` 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory6` | BF16 /10、/4、64B 对齐预算 |
| Kernel entry/router | `foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp` | key 4 branch | BF16 storage + float compute |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase::Init` | 9I Queue 容量、castCap |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary.h` | BF16 `InnerComputer::ComputePerCast` | 分片 cast/power/cast 与 barrier |

## 算子特定优化

### Trick T-001：按 tensor 边界连续均分 32B block

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：任意非空 TensorList；`AssignDataToEachCore(needCoreNum)`。
- Golden 来源：[`foreach/foreach_utils/op_host/foreach_tiling_class.h:651-660`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L651-L660)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
void AssignDataToEachCore(int64_t needCoreNum)
{
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
    // Divisible, representing the amount of data each core needs to process.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
```

#### 机制与设计原因

Host 不按 tensor 个数简单分核，而把整个列表视作逻辑连续的元素域，以 32B block 为分配单位，再用首尾 tensor/offset 把分区投影回 TensorList。这样大 tensor 可跨核、小 tensor 可在同一核连续处理，并使大多数核边界满足完整 `DataCopy` 的 block 对齐；只有实际 tensor/tile 尾部走 `DataCopyPad`。

#### 迁移边界

- 适用：TensorList 中尺寸差异大、但元素 dtype 一致的逐元素算子。
- 前提：POD 能表达每核 tensor 起止下标和首尾 offset；列表/core 不超过 50；32B block 元素数计算无溢出。
- 失效：需要 tensor 间归约、顺序依赖或不同 dtype/layout 时不能直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但源码明确要求保留 remainder core 的额外 block 与闭区间 `endOffset`，迁移时不得把它改成不等价的平均元素截断。

### Trick T-002：BF16 用双分区 FP32 临时区分片计算

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：`T=bfloat16_t`，每个 tile 再按 `maxCastDataCount` 分片。
- Golden 来源：[`foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h:64-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h#L64-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ComputePerCast(
    const LocalTensor<bfloat16_t>& dataLocal, const LocalTensor<bfloat16_t>& outLocal,
    LocalTensor<float>& float32Tensor, float scalarVal, uint32_t maxCastDataCount, uint32_t index,
    int64_t dataCount)
{
    PipeBarrier<PIPE_V>();
    Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
    op(float32Tensor[offset], float32Tensor, scalarVal, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(outLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
}
```

#### 机制与设计原因

BF16 tile 可以大于 FP32 临时区；实现把 tile 按 `maxCastDataCount=I/4` 切片，并把单个 2I-byte FP32 Queue 分成输入、输出两个不重叠的 I-byte 子区。每片只物化 FP32 base 和结果，完成降精度后复用相同子区，避免为整个 BF16 tile 同时保留两份 FP32 数据，同时把 `pow` 放在 FP32 精度执行。

#### 迁移边界

- 适用：低精度输入、必须升到 FP32 执行且可逐片独立计算的 unary 算子。
- 前提：两子区容量均至少 `maxCastDataCount*sizeof(float)`，V barrier 顺序保持，输出 cast 只覆盖有效片。
- 失效：算子需要跨片状态或 Power body 需要额外别名约束时必须重新做 UB 账。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码证据要求 `paramsCount=2` 时输出位于 `float32Tensor[maxCastDataCount]`，不能误写为与输入原地重叠。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 4 | Host dtype key |
| 模板与 dtype | half/half | float/float | BF16 storage/float compute | BF16 显式升精度 |
| 循环与分核 | 32B block；tile I/2 | 32B block；tile I/4 | 32B block；tile 9I/2，再分片 I/4 | dtype 字节宽度和临时区 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | data/out 各 I | data/out 各 I | data/out 各 9I；FP32 2I | BF16 cast 管线 |
| 内存总占用与复用 | 2I | 2I | 20I | Host 分母与 Queue 初始化闭合 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Power half | DataCopy(Pad)+Power float | DataCopy(Pad)+Cast/Power/Cast | 全部为 AIV Vector |
| 同步、精度与 tail | V barrier；half | V barrier；FP32 | 每阶段 V barrier；CAST_RINT | tail 均精确字节提交 |
| fallback/失败边界 | 未知 key 无 body | 同左 | 编译 guard 排除 3003/3113 | 无另一个 Host fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 文档支持 | V2 文档支持 | V2 文档支持，见 `_950` 卡 |
| Host/Launcher | ascend910b→arch22 公共 Tiling | ascend910_93→arch22 公共 Tiling | arch35 独立 Tiling，最多 256 tensor |
| Kernel/资源 | AIV Vector 公共 foreach body | 同 A2 | SIMT VF、无显式 AIV Queue |

## 限制与失败边界

- 公开 V2 契约只有 FLOAT16/FLOAT32/BFLOAT16；源码中的 INT32 key 3 与 ACLNN dtype 检查是未公开实现信号，不应当作为兼容性承诺。
- V2 文档写 out shape size 不小于 x，ACLNN `CheckShape` 实际逐项要求 shape 精确相等；本卡按更强的源码检查复用。
- 文档标记 x 非连续不支持，但 ACLNN 实现仍调用 `Contiguous`；是否接受某个非连续输入受框架前置边界影响，固定仓未闭合该框架实现。
- arch22 POD 最多保存 50 个 tensor 和 50 个 core；`Init` 到 50 项即停止，源码没有显式 tensorNum>50 错误，调用者不得依赖静默截断。
- `totalDataCount`、shape size 与 offset 是 int64；分核中部分 core 计数和 key 存在窄化。固定仓没有显式溢出检查。
- tile `dataCount` 从 int64 经函数签名窄化到 `PowerAdapter` 的 int32，再 cast 为 Power 的 uint32 count；平台 UB 通常限制单 tile，但固定仓没有独立的窄化 guard。
- 零长度 TensorList 的返回分支位于 `CheckDtypePowScalar` 的 `(*x)[0]` 读取之后；不能把该分支当作零长度列表安全证据。非空列表中所有 tensor 都为空时，arch22 的闭区间分核也缺少明确的全空保护。
- Power primitive 内部算法和额外临时资源属于 AscendC 库边界；Host 的 `R` 预留可见，但其内部布局不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_pow_scalar/docs/aclnnForeachPowScalarV2.md` | 产品表、原型、参数表 | 公开版本、平台、dtype/shape/format |
| 入口声明/约束 | `foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.h` | 两段式 API 声明 | C ABI |
| 入口实现 | `foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.cpp` | `CheckParams/Exec...` | 校验、Contiguous、scalar 转换、executor |
| 入口实现 | `foreach/foreach_pow_scalar/op_host/op_api/foreach_pow_scalar_v2.cpp` | `ForeachPowScalarV2` | 单 AICORE launcher 节点 |
| Host/Tiling/Launcher | `foreach/foreach_pow_scalar/op_host/foreach_pow_scalar_def.cpp` | `ForeachPowScalar` OpDef | dtype、ND、平台注册 |
| Host/Tiling/Launcher | `foreach/foreach_pow_scalar/op_host/arch22/foreach_pow_scalar_tiling_arch22.cpp` | `Tiling4ForeachPowScalarTiling` | arch22 入口与 compile info |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD、workspace |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD 字段和定长数组 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | key 映射和字节常量 |
| Kernel entry/body | `foreach/foreach_pow_scalar/op_kernel/foreach_pow_scalar.cpp` | `foreach_pow_scalar/PowerAdapter` | key router 与 Power 适配 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `ForeachOneScalarBinary/InnerComputer` | scalar、计算、BF16 cast |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、tile、DMA、tail |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD 解析、TensorList 地址、容量 |
| 构建接线 | `foreach/foreach_pow_scalar/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | 910b/910_93→arch22 |
| 构建接线 | `foreach/foreach_pow_scalar/op_kernel/CMakeLists.txt` | `add_kernel_sources` | 910b/910_93 编译公共 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 每核 tensor 闭区间描述 | 六字段顺序、数组长度 50、元素/字节单位 | `ForeachCommonTilingData` |
| 调用顺序 | CopyIn→Compute→CopyOut | Queue 生命周期与 BF16 cast/power/cast 顺序 | `KernelForeachUnary::SingleTensorProcess` |
| workspace/内存 | 核心 Kernel 无 workspace；Queue 按 I 分配 | Power reserve、divider、alignment、BF16 9I/2I 关系 | `DivideUbMemory6`、`KernelForeachUnary::Init` |
| 同步 | Queue + V barrier | BF16 三阶段间 barrier 不得删除 | `InnerComputer::Compute/ComputePerCast` |
| 精度与尾块 | BF16 FP32 Power 后 CAST_RINT；GM tail 精确字节 | scalar 转换 dtype、castCap、有效 n | `Exec...`、`ComputePerCast`、`CopyOut` |
| 失败边界 | API 检查、Tiling null/resource检查、未知 key | 不扩张 INT32 公共契约；不忽略 50 项 ABI 上限 | `CheckParams/CheckOpInfo/foreach_pow_scalar` |

- [Power（数学计算）](../../../../apis/simd_api/adv_api/math_compute/power_interface/power.md)
