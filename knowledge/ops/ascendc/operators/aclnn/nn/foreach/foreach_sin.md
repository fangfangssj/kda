---
type: Operator
title: "ForeachSin A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachSin 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/docs/aclnnForeachSin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/foreach_sin_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/arch22/foreach_sin_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/foreach_sin.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sin/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:20:33Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachSin A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSinGetWorkspaceSize` + `aclnnForeachSin`，公开 ACLNN 两段式接口；固定仓未提供生成后的 C 头和 inner bridge。 |
| 计算语义 | 对 TensorList 中每个张量逐元素计算 `y_i[j] = sin(x_i[j])`。 |
| 输入输出 | 动态 TensorList；ND；FLOAT16/FLOAT32/BFLOAT16；输出 dtype 与输入一致，输出元素容量不得小于对应输入。 |
| 执行类型 | 单 Kernel、单 AIV_VECTOR 节点；无 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 FP16 原精度；R-002 FP32 原精度；R-003 BF16→FP32→BF16。 |
| 核心资源 | TensorList GM 描述符、输入/输出双缓冲 `TQue`、BF16 专用 FP32 Queue；算子不消费 user workspace。 |
| 来源闭包 | source_closed：公开 API 文档→OpDef/公共 infer→arch22 Tiling/POD→dtype router→公共 unary/triangle body→A2/A3 构建接线闭合；生成式 ACLNN bridge 与 `Sin` primitive 内部为明确框架/库边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设输入列表 `x=[x_0,...,x_{n-1}]`，输出列表 `y=[y_0,...,y_{n-1}]`。对每个有效输入元素执行 `y_i[j]=sin(x_i[j])`。Host 依据 storage shape 的元素数切分，Device 把列表视为按张量边界串接的线性元素域，但输出仍写回各自张量。公开文档声明输入值域 `[-65504, 65504]` 时满足功能精度要求，且接口为默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 0–8 维、ND；可非连续，OpDef `AutoContiguous()` | FLOAT16/FLOAT32/BFLOAT16；列表内一致 | 非空指针；支持零元素 Tensor；每个元素是正弦输入。框架连续化细节固定仓不可见。 |
| `out` | TensorList；每个 Tensor 0–8 维、ND；公开接口不支持非连续输出 | 与 `x` 一致；列表内一致 | 非空指针；对应输出 storage 元素数不得小于输入。公共 infer 在图模式把 shape/dtype 逐项设为输入值。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 第一段返回 Device workspace 字节数；arch22 Tiling 写 `32`，但 Kernel 将 `userWS` 置空。 |
| `executor` | 执行器输出 | `aclOpExecutor**` | 第一段生成并持有计算流程，第二段消费。生成实现固定仓不可见。 |
| `workspace`、`stream` | Device 地址、执行流 | 指针/handle | 第二段按 executor 在给定 stream 启动；workspace owner 为调用者。 |

### 入口与版本

固定 Commit 只公开 `aclnnForeachSin` 这一版本族。公开文档明确支持 Atlas A2/A3；`ForeachSin` OpDef 注册 `ascend910b` 与 `ascend910_93`，Host CMake 将二者映射到 `arch22`，Kernel CMake也编译公共 `foreach_sin.cpp`，证据无冲突。生成后的 `aclnnop/aclnn_foreach_sin.h`、`GetWorkspaceSize` 实现和 executor→`ForeachSin` inner bridge 未纳入固定仓，按生成式 ACLNN 框架边界处理；算子拥有的校验、Tiling、POD、router 和 body 已继续闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口检查 `x/out` 空指针、支持 dtype 与 dtype 一致性，并把 shape 不满足、列表内 dtype 不一致和维数超过 8 归为 tiling 错误。OpDef 对动态输入和输出启用 `AutoContiguous()`；公开接口允许非连续输入但不允许非连续输出。arch22 `ForeachCommonTiling::Init` 最多枚举 `MAX_TENSOR_CONT=50` 个输入 Tensor，读取第一个输入描述 dtype，核对列表内 dtype，并记录每项 storage 元素数。`CheckOpInfo` 再核对输入列表实例数、每项维数、输出 dtype 和输出元素容量；任一步失败返回 `GRAPH_FAILED`。固定仓没有生成 bridge，因此 ACLNN 状态码到这些 graphStatus 的具体映射不可见。

### 参数、Workspace 与生命周期

`RunBigKernelTiling` 从平台读取 AIV 核数和 UB 大小，产生 key、每核 Tensor 起止索引/offset、每项元素数和每个 Queue 的单 buffer 字节容量 `I=inputsTensorUbSize`。它把 `blockDim=needCoreNum`，把 workspace slice 0 设为 32 字节。Kernel 签名接收 workspace，但显式令 `userWS=nullptr`，公共 unary body不访问 workspace；这 32 字节从第一段返回后由调用者持有到 stream 完成。Tiling blob 由 `ForeachCommonTilingData::SaveToBuffer` 序列化，executor/stream 的内部 owner 与释放协议固定仓不可见。

### 启动图与失败传播

单节点启动图：executor 在用户 stream 上启动 `foreach_sin(xListDesc, yListDesc, workspace, tiling)`，`blockDim=needCoreNum`，POD 为 `ForeachCommonTilingData`。节点直接读 TensorList GM 地址表与各输入数据，写对应输出；没有 Kernel 间边。Tiling/指针/shape/dtype 校验失败时不应启动节点；Kernel unknown key 没有显式错误分支而是无写出。stream 内 launch ordering 属 ACLNN 框架边界；公开样例在第二段后调用 `aclrtSynchronizeStream` 再读取/释放输出与 workspace。

## Route 与参数设计

### 分核与资源计算

令 `E=sum_i tensorDataCountList[i]`，`s=sizeof(dtype)`，`e32=32/s`。`needCoreNum=min(platformAiv, ceil(E/e32), 50)`，若 `E=0` 则取 1。`blockCount=ceil(E/e32)`；基础每核元素数是 `floor(blockCount/needCoreNum)*e32`，前 `blockCount mod needCoreNum` 个核各多 `e32`。`AssignDataToEachCore` 沿 TensorList 顺序推进 cursor，生成含首尾的 `[tensorStart,tensorEnd]` 以及起止元素 offset，因此同一 Tensor 可以被两个相邻核切开。

令 `U` 为平台 UB 字节数，`S=tilingData.GetDataSize()`，`A` 为 Queue 容量对齐（FP16/FP32 32B，BF16 64B）。Sin primitive 预留 `R=4*2048*calcPro*s`，其中 FP16/BF16 `calcPro=6`，FP32 `calcPro=2`。FP16/FP32：`I=floor(((U-S-R)/4)/A)*A`；BF16：先令 `U'=(U-S-R)/10`，再 `I=floor((U'/4)/64)*64`。Device 用 `I` 推导每 tile 最大元素数；BF16 的输入/输出 Queue 单 buffer 扩为 `9I`，FP32 临时 Queue 为 `2I`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 Vector | key 1；`ForeachTriangle<half,half,Sin<half,false>>` | `dataType==DT_FLOAT16`，`GetTilingKeyByDtypeOnly` 返回 1 | 否则继续 key 2 |
| 2 | R-002 FP32 Vector | key 2；`ForeachTriangle<float,float,Sin<float,false>>` | `dataType==DT_FLOAT`，返回 2 | 否则继续 key 4 |
| 3 | R-003 BF16 升精度 Vector | key 4；`ForeachTriangle<bfloat16_t,float,Sin<float,false>>` | `dataType==DT_BF16` 且非 `__NPU_ARCH__==3003/3113` | 其它 dtype Host 校验失败；router unknown key 无写出 |

```text
ForeachCommonTiling::Init/CheckOpInfo 成功
  if dtype == FLOAT16 -> key 1 -> R-001
  else if dtype == FLOAT32 -> key 2 -> R-002
  else if dtype == BFLOAT16 -> key 4 -> R-003
  else -> key 0 / Host 不支持，失败
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 Vector | 1 | FP16，A2/A3，校验成功 | 单 AIV_VECTOR `foreach_sin` | `ForeachTriangle<half,half,Sin<half,false>>` | depth-2 输入/输出 Queue；`PipeBarrier<PIPE_V>` 包围 Sin | 完整 tile 用 `DataCopy`；尾 tile 用 exact-byte `DataCopyPad` |
| R-002 FP32 Vector | 2 | FP32，A2/A3，校验成功 | 单 AIV_VECTOR `foreach_sin` | `ForeachTriangle<float,float,Sin<float,false>>` | 同 R-001，但 tile 元素数按 4B dtype | 同 R-001 |
| R-003 BF16 升精度 Vector | 4 | BF16，A2/A3，且编译 guard 允许 | 单 AIV_VECTOR `foreach_sin` | `ForeachTriangle<bfloat16_t,float,Sin<float,false>>` | 输入/输出 Queue 各 `2×9I`，FP32 Queue `2I`；Cast/Sin/Cast 间 barrier | 大 tile 再按 `I/4` 个 FP32 元素分段；尾搬运 exact-byte |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor/核数组上限 | 枚举 `<50`；`needCoreNum<=50` | POD 数组均固定 50 项 | 全部 | 固定代码未显式拒绝第 51 项，超界语义不可见 | `MAX_TENSOR_CONT/MAX_CORE_CONT`、`Init` |
| 32B 分核 | `e32=32/s`，按 32B block 切分 | 完整 DMA tile/核域对齐 | 全部 | 尾部由 `DataCopyPad` | `GetNeedCoreNum/AssignDataToEachCore` |
| Sin scratch | `U>S+R` 后才能做无符号减法 | 数学 primitive 需要 Host 预留临时 UB | 全部 | 固定代码无显式 underflow guard | `DivideUbMemory7` |
| BF16 UB 收缩 | dtype BF16 时 `(U-S-R)/10` 后再 `/4` | Queue 扩为 9 倍且增加 FP32 临时区 | R-003 | 不回退其它 Route | `DivideUbMemory7`、`KernelForeachBase::Init` |
| BF16 编译 guard | 排除 NPU arch 3003/3113 | 对应编译目标不实例化 BF16 router/body | R-003 | Host/OpDef 对 Kirin 另有约束；A2/A3 可达 | `foreach_sin` router |
| 空 tile | `dataCount==0` 时 `copyTimes=0` | 不进行 Queue 分配与 DMA/计算 | 全部 | 当前 Tensor 无输出写入 | `SingleTensorProcess` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 由 tiling 宏生成；固定仓未显式给出 `sizeof/alignof/offsetof` | 字节/Queue 单 buffer | `DivideUbMemory7` 写 `inputsTensorUbSize`，`FillTilingData` setter | `KernelForeachBase::ParseTilingData/Init` 推 `maxDataCount` | 32B 对齐；BF16 64B 对齐 |
| `ForeachCommonTilingData` | `tensorDataCountList[50]` | `int64_t[50]` | 同上 | 元素/张量 | `Init: shape.GetShapeSize()` | `Process` 计算每个核内张量片段长度 | 仅前 `totalTensorCount` 项有效 |
| `ForeachCommonTilingData` | `tensorStartList[50]` | `uint16_t[50]` | 同上 | 张量索引/核 | `AssignDataToEachCore` | `ParseTilingData`→`Process` 循环下界 | 由 `blockIdx` 索引 |
| `ForeachCommonTilingData` | `tensorEndList[50]` | `uint16_t[50]` | 同上 | 张量索引/核 | `AssignDataToEachCore` | `Process` 循环上界，含尾 | 由 `blockIdx` 索引 |
| `ForeachCommonTilingData` | `tensorStartOffsetList[50]` | `int64_t[50]` | 同上 | 元素 offset/核 | `AssignDataToEachCore` | 首 Tensor 的 `cursorStart` | 非首 Tensor inactive |
| `ForeachCommonTilingData` | `tensorEndOffsetList[50]` | `int64_t[50]` | 同上 | 元素 offset/核 | `AssignDataToEachCore` | 尾 Tensor 的 `cursorEnd` | 非尾 Tensor inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | `arch22/foreach_sin_tiling_arch22.cpp::Tiling4ForeachSinTiling`→`ForeachCommonTiling::RunBigKernelTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_sin.cpp::foreach_sin` | `KernelForeachUnary`、`ForeachTriangle<half,...>` | Host/Kernel CMake `ascend910b/ascend910_93` |
| R-002 | key 2 | 同上 | 同上 | 同上 | `ForeachTriangle<float,...>` | 同上 |
| R-003 | key 4 | 同上 | 同上 | 同上 | `ForeachTriangle<bfloat16_t,float,...>` + BF16 specialization | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 只有一个 `foreach_sin` AIV entry。router 解析 POD，按 key 实例化 `ForeachTriangle`；类的 `Init` 读取 `blockIdx` 对应核域、绑定 TensorList 描述符并建 Queue。`Process` 依次遍历本核涉及的 Tensor，借 `GetTensorAddr` 从 ListTensor GM 元数据取得真实数据指针，按 `maxDataCount` 分 tile，顺序执行 `CopyIn→Compute→CopyOut`。Queue 的 EnQue/DeQue/FreeTensor 管理搬运/Vector阶段所有权；Sin 前后显式 `PipeBarrier<PIPE_V>`。无跨 Kernel workspace 边。

### Route R-001：FP16 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_sin/op_kernel/foreach_sin.cpp` | `foreach_sin` | `TILING_KEY_IS(1)` | `ForeachTriangle<half,half,Sin<half,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sin(x,y,workspace,tiling)
  -> ForeachTriangle<half,half,Sin<half,false>>::Init
    -> KernelForeachBase<half>::Init/ParseTilingData
  -> KernelForeachUnary::Process
    -> GetTensorAddr + SingleTensorProcess
      -> CopyIn -> ForeachTriangle::Compute -> InnerComputer::Compute
      -> Sin<half,false> -> CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码
core = GetBlockIdx(); load [tensorStart,tensorEnd,startOffset,endOffset]
for tensor i in inclusive core range:
  begin = (i==start ? startOffset : 0)
  end = (i==end ? endOffset : tensorElements[i]-1)
  bind inGM/outGM = list_ptr[i] + begin
  for tile in ceil((end-begin+1)/(I/2)):
    valid = min(I/2, remaining)
    CopyIn exact valid half elements; tail uses DataCopyPad(valid*2 bytes)
    PipeBarrier(V); Sin<half,false>(outLocal,inLocal,valid); PipeBarrier(V)
    CopyOut valid elements; tail uses DataCopyPad(valid*2 bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `list[i]+cursorStart+tile*(I/2)` | `valid` 元素 | FP16/ND linear | ACLNN input | CopyIn | caller；stream 完成前有效 | full tile 32B 对齐，tail exact bytes |
| `dataQueue` | Queue slot | `depth 2 × I` 字节 | FP16 | CopyIn | Compute | `TPipe`；DeQue 后 Free | 双缓冲 |
| `outQueue` | Queue slot | `depth 2 × I` 字节 | FP16 | Compute | CopyOut | `TPipe`；CopyOut 后 Free | 双缓冲 |
| output GM | 同 input 元素 offset | `valid` 元素 | FP16/ND linear | CopyOut | caller | caller | 只写输入有效元素数 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2I bytes"]:2
  outQ["outQueue · depth=2 · 2I bytes"]:2
```

##### 字节账与复用顺序

```text
explicit_queue_UB = 2I + 2I = 4I
primitive_reserved = R = 4*2048*6*2 = 98304 bytes
4I + R + S <= U（I 由 Host 向下 32B 对齐）
dynamic_ub = N/A
workspace_total = 32 bytes（Kernel 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | CopyIn | Compute/DeQue | `FreeTensor` 后的后续 tile | Queue EnQue/DeQue/Free | 未 Free 前不得覆盖 |
| `outQueue` slot | Sin | CopyOut/DeQue | `FreeTensor` 后的后续 tile | Queue + Sin 后 `PipeBarrier` | CopyOut 完成后释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue + DataCopy/DataCopyPad | GM | Compute DeQue | input slot | FP16；尾部只搬 `valid*2` 字节 |
| Sin | `PipeBarrier<PIPE_V>` | CopyIn/Vector | Sin/CopyOut | Vector 读写次序 | `Sin<half,false>`，有效长度 `valid` |
| UB→GM | Queue + DataCopy/DataCopyPad | Sin | GM | output slot | 只提交有效长度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling/DivideUbMemory7` | 32B 分核、UB/Queue 容量、key/blockDim/workspace |
| Kernel entry/router | `foreach_sin/op_kernel/foreach_sin.cpp` | key 1 branch | 绑定 FP16 模板 |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_unary.h`、`foreach_triangle.h` | `SingleTensorProcess/InnerComputer<half,...>` | 双缓冲、tail DMA 与 Sin 调用 |

### Route R-002：FP32 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_sin/op_kernel/foreach_sin.cpp` | `foreach_sin` | `TILING_KEY_IS(2)` | `ForeachTriangle<float,float,Sin<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sin -> ForeachTriangle<float,float,Sin<float,false>>::Init/Process
  -> KernelForeachUnary<float>::SingleTensorProcess
    -> CopyIn -> InnerComputer<float,float,...>::Compute
    -> Sin<float,false> -> CopyOut
```

##### 源码等价伪码

```text
# 与 R-001 相同的 core/Tensor/tile 循环
tile_capacity = I/4 float elements
valid = min(tile_capacity, remaining)
CopyIn(valid); barrier; Sin<float,false>(out,in,valid); barrier; CopyOut(valid)
tail CopyIn/CopyOut 使用 blockLen=valid*4 bytes 的 DataCopyPad
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `list[i]+cursorStart+tile*(I/4)` | `valid` | FP32/ND linear | caller/CopyOut | CopyIn/caller | caller | tail exact-byte |
| `dataQueue` | Queue slot | `2I` | FP32 | CopyIn | Sin | `TPipe` | depth 2 |
| `outQueue` | Queue slot | `2I` | FP32 | Sin | CopyOut | `TPipe` | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2I bytes"]:2
  outQ["outQueue · depth=2 · 2I bytes"]:2
```

##### 字节账与复用顺序

```text
explicit_queue_UB = 4I
primitive_reserved = 4*2048*2*4 = 65536 bytes
4I + 65536 + S <= U
dynamic_ub = N/A
workspace_total = 32 bytes（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output Queue slots | CopyIn/Sin | Sin/CopyOut | Free 后下一 tile | Queue + `PipeBarrier` | slot 生命周期不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn/Sin/CopyOut | Queue + V barrier | 前一阶段 | 后一阶段 | 当前 slot | 全程 FP32；尾部长度为 `valid` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory7` | FP32 `calcPro=2`、32B 对齐 |
| Kernel entry/router | `foreach_sin/op_kernel/foreach_sin.cpp` | key 2 branch | FP32 模板映射 |
| Device body/helper | `foreach_utils/op_kernel/foreach_triangle.h` | `InnerComputer<float,float,...>` | 直接 FP32 Sin，无 cast |

### Route R-003：BF16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `foreach_sin/op_kernel/foreach_sin.cpp` | key 4 branch + arch guard | BF16 且非 3003/3113 | `ForeachTriangle<bfloat16_t,float,Sin<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sin -> ForeachTriangle<bfloat16_t,float,Sin<float,false>>::Init/Process
  -> KernelForeachBase::Init (totalTensorUbSize=9I, maxCastDataCount=I/4)
  -> SingleTensorProcess -> CopyIn
  -> InnerComputer<bfloat16_t,float,...>::Compute
    -> ComputePerCast: Cast BF16->FP32 -> Sin<float,false> -> Cast FP32->BF16 RINT
  -> CopyOut
```

##### 源码等价伪码

```text
# 外层分核、Tensor 和 tile 循环同 R-001
tile_capacity = 9I/2 BF16 elements
for cast_chunk in ceil(valid/(I/4)):
  n = min(I/4, remaining_in_tile)
  Cast(floatTmp[0:n], bf16In[offset:offset+n], CAST_NONE)
  Sin<float,false>(floatTmp[I/4 : I/4+n], floatTmp[0:n], n)
  Cast(bf16Out[offset:offset+n], floatTmp[I/4:I/4+n], CAST_RINT)
CopyOut exact valid BF16 elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | TensorList pointer + core/tile offset | `valid` | BF16 | caller/CopyOut | CopyIn/caller | caller | tail exact-byte |
| `dataQueue` | slot | `depth 2 × 9I=18I` | BF16 | CopyIn | BF16→FP32 Cast | `TPipe` | 双缓冲 |
| `outQueue` | slot | `depth 2 × 9I=18I` | BF16 | FP32→BF16 Cast | CopyOut | `TPipe` | 双缓冲 |
| `float32Queue` | `[0,I)` input temp；`[I,2I)` output temp | `depth 1 × 2I` | FP32 | Cast/Sin | Sin/Cast | Init 时 Alloc+EnQue，Process 中 DeQue，末尾 Free | 两个不重叠子区 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 18I bytes"]:2
  outQ["outQueue · depth=2 · 18I bytes"]:2
  castIn["float32Queue[0:I] · I bytes"]:1
  castOut["float32Queue[I:2I] · I bytes"]:1
```

##### 字节账与复用顺序

```text
explicit_queue_UB = 18I + 18I + 2I = 38I
primitive_reserved = 98304 bytes
Host first divides remaining budget by 10, then by 4, so before alignment 40I <= U-S-R
=> 38I + R + S <= U（保守留量）
dynamic_ub = N/A
workspace_total = 32 bytes（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| float temp `[0,I)` | BF16→FP32 Cast | Sin src | 下一 cast chunk | Cast 后 V barrier | Sin 读取完成前不覆盖 |
| float temp `[I,2I)` | Sin dst | FP32→BF16 Cast | 下一 cast chunk | Sin/Cast 间 V barrier | 降精度完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast up | `Cast(...CAST_NONE)` + V barrier | BF16 Queue | FP32 Sin | float input temp | 每段最多 `I/4` 元素 |
| Sin | `Sin<float,false>` + V barrier | float input | Cast down | float output temp | FP32 计算 |
| Cast down | `Cast(...CAST_RINT)` + V barrier | float output | CopyOut | BF16 output Queue | RINT 降精度；尾段用真实 `n` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory7` | BF16 `/10` 收缩与 64B 对齐 |
| Kernel entry/router | `foreach_sin/op_kernel/foreach_sin.cpp` | key 4 branch | BF16→FP32模板 |
| Device body/helper | `kernel_foreach_base.h`、`foreach_triangle.h` | BF16 `Init`/`ComputePerCast` | 9倍 Queue、双 FP32 子区与 RINT |

## 算子特定优化

### Trick T-001：按 Sin 临时区反推双缓冲 tile 容量

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：`opCode==FOREACH_SIN_OP_CODE`；dtype 决定 `calcPro`，BF16 再收缩可用量。
- Golden 来源：[`foreach/foreach_utils/op_host/foreach_tiling_class.h:1039-1055`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L1039-L1055)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (opCode == FOREACH_SIN_OP_CODE) {
    // foreach_sin
    uint32_t calcPro = SIN_HALF_CALC_FAC;
    if (dataTypeSize == BYTE_LEN_4) {
        calcPro = SIN_FLOAT_CALC_FAC;
    }
    uint32_t reservedUbSize = static_cast<uint32_t>(4U * SIN_BASIC_BLOCK * calcPro * dataTypeSize);
    uint32_t totalSize = static_cast<uint32_t>(
        ubSizePlatForm - static_cast<uint32_t>(tilingData.GetDataSize()) - reservedUbSize);
    if (dataType == ge::DT_BF16) {
        totalSize = static_cast<uint32_t>(totalSize / UB_DIVIDER_FOR_TEMP_CASTING);
    }
    uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / FOREACH_SIN_DIVIDER); // 4
    inputsTensorUbSize = static_cast<uint32_t>(canUseUbSize / BYTE_BLOCK * BYTE_BLOCK);
    inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                     canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
} else if (opCode == FOREACH_ABS_OP_CODE) {
```

#### 机制与设计原因

Host 先扣除 POD 与 Sin primitive 的 dtype 相关临时量，再按四份切给 depth-2 input/output Queue；这使 tile 容量直接满足 Device 的真实 Queue 图，而不是使用固定 tile。BF16 再除以 10，为 `9I` input/output slot 与 FP32 临时区保守留量。该机制改变 UB 上界和每次 DMA 的有效长度，不声称未经实测的性能比例。

#### 迁移边界

- 适用：具有固定 primitive scratch 且 input/output 双缓冲的逐元素 unary AIV Kernel。
- 前提：平台 UB、POD 字节数、primitive scratch 公式、Queue depth 和 BF16 扩容倍数保持一致。
- 失效：Queue 数、depth、primitive 临时区或 cast buffer 改变时必须重新做字节账，不能复用 `FOREACH_SIN_DIVIDER=4`。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码也未提供 `U<=S+R` 的显式失败 guard，因此迁移时不能省略容量前置检查。

### Trick T-002：BF16 分段升精度并在单一 FP32 Queue 内分区

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：`T=bfloat16_t`，`P=float`，`paramsCount=2`。
- Golden 来源：[`foreach/foreach_utils/op_kernel/foreach_triangle.h:68-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h#L68-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
op(float32Tensor[offset], float32Tensor, dataCount);
PipeBarrier<PIPE_V>();
Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

每个 BF16 tile再按 `maxCastDataCount=I/sizeof(float)` 分段，FP32 Queue 的前后两个 `I` 字节子区分别承载 Sin 输入和输出；Sin 在 FP32 执行，最后 RINT 回 BF16。显式 barrier 允许同一物理 Queue 跨 cast chunk 重用，同时防止 Vector 阶段提前覆盖。

#### 迁移边界

- 适用：低精度存储、FP32 unary primitive 计算、结果可按 RINT 回写的路线。
- 前提：两个 FP32 子区不重叠，Queue 容量至少 `2I`，每段不超过 `I/4` 个 float。
- 失效：若 primitive 需要额外输入/输出别名、不同 round mode 或跨段状态，必须拆 buffer 或改变同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 升精度 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 4 | dtype-only selector |
| 模板与 dtype | half→half Sin | float→float Sin | BF16→float Sin→BF16 | BF16 使用 FP32 opmath |
| 循环与分核 | 32B block，tile `I/2` | 32B block，tile `I/4` | 32B block，外 tile `9I/2`，内段 `I/4` | 元素字节数与 cast buffer |
| Queue/TBuf/寄存器/dynamic UB/通信 window | input/output 各 depth2 | 同左 | 两个 9I Queue + FP32 Queue | BF16 额外精度路径 |
| 内存总占用与复用 | 4I + 98304 + S | 4I + 65536 + S | 38I + 98304 + S，Host更保守 | primitive scratch/cast |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Sin | 同左 | DataCopy(Pad)+Cast+Sin+Cast | 全部 AIV_VECTOR |
| 同步、精度与 tail | Queue+V barrier，FP16 | Queue+V barrier，FP32 | 更多 V barrier，RINT；分段 tail | 防止子区覆盖 |
| fallback/失败边界 | unknown key 无写出 | 同左 | 编译 guard | 无运行时 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；`ascend910b` | 支持；`ascend910_93` | 支持，但使用独立 `_950` 卡 |
| Host/Launcher | 共用 arch22 `ForeachCommonTiling` | 共用 arch22 | arch35 独立 SIMT Tiling |
| Kernel/资源 | 共用 AIV Vector + Queue | 共用 AIV Vector + Queue | SIMT 直接 GM、无这些 Queue |

## 限制与失败边界

- 公开输入值域保证为 `[-65504,65504]`；超出时精度不保证。输出非连续 Tensor 不支持。
- arch22 固定 POD 只有 50 个 Tensor/50 个 core 项；`Init` 遇到第 51 项停止枚举但未显式报错，这是固定源码可见的闭包边界，不能推断额外 Tensor 语义。
- `DivideUbMemory7` 对 `U-S-R` 使用无符号减法，固定仓没有 UB 不足的显式 guard；移植时必须保留容量前置条件。
- `totalDataCount`和shape元素数使用`int64_t`，`CeilA2B(a,b)`会计算`a+b-1`；固定仓没有总元素数加法或向上取整的显式溢出检查。
- Kernel 对 unknown key 无显式错误或 fallback，只会不写输出。生成式 ACLNN bridge、executor 内部、ListTensor 描述符 ABI、tiling 宏的精确 `sizeof/alignof/offsetof` 与 `Sin` primitive 内部算法固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sin/docs/aclnnForeachSin.md` | `aclnnForeachSin*` | API 原型、平台、dtype/shape、错误与精度约束 |
| 入口实现 | `foreach/foreach_sin/op_host/foreach_sin_def.cpp` | `ForeachSin` | ND/dtype、AutoContiguous、A2/A3 注册 |
| Host infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon/InferDataType4ForeachCommon` | 输出数量、shape、dtype传播 |
| Host/Tiling | `foreach/foreach_sin/op_host/arch22/foreach_sin_tiling_arch22.cpp` | `Tiling4ForeachSinTiling` | op code 与公共 Tiling 入口 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace、POD producer |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | 字段与 ForeachSin 注册 |
| dtype key | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | key 1/2/4 |
| Kernel entry/body | `foreach/foreach_sin/op_kernel/foreach_sin.cpp` | `foreach_sin` | router 与模板实例 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD 解析、ListTensor 地址、BF16容量 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、Tensor/tile循环、DMA tail |
| Kernel body | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | `ForeachTriangle/InnerComputer` | Sin、BF16 Cast、barrier |
| 构建接线 | `foreach/foreach_sin/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3→arch22 Host |
| 构建接线 | `foreach/foreach_sin/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 公共 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段公共 foreach POD | 数组长度50、字段类型/顺序、元素/字节单位 | `ForeachCommonTilingData` |
| 调用顺序 | `Init→Process→CopyIn→Compute→CopyOut` | Tensor 与 tile 地址公式、含首尾 offset | `KernelForeachUnary::Process` |
| workspace/内存 | user workspace 未消费；Queue按 Host I 分配 | 32B workspace约定、Queue depth、BF16 9倍/2I临时区 | `RunBigKernelTiling`、`InitBuffer` |
| 同步 | Queue 生命周期与 V barrier | Cast/Sin/Cast 和复用顺序 | `InnerComputer::Compute` |
| 精度与尾块 | FP16/FP32原精度；BF16 FP32 opmath+RINT | tail exact-byte、有效长度、round mode | `ComputePerCast`、`CopyIn/CopyOut` |
| 失败边界 | Host校验失败不启动；unknown key无写出 | 不得把 unknown key 当 fallback | `CheckOpInfo`、`foreach_sin` router |