---
type: Operator
title: "ForeachAddList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachAddListV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/docs/aclnnForeachAddListV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/op_api/aclnn_foreach_add_list_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/op_api/foreach_add_list_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/arch22/foreach_add_list_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/foreach_add_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/foreach_add_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_ternary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:28:50Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachAddList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddListV2GetWorkspaceSize` / `aclnnForeachAddListV2`，公开 ACLNN 两段式接口；本平台版本族取最新 V2 |
| 计算语义 | 对列表位置 `i` 及其中每个元素 `j`，`y_i[j] = x1_i[j] + x2_i[j] * alpha` |
| 输入输出 | `x1/x2/out` 为等长 TensorList，逐位置 shape 相同，ND，统一 dtype 为 FP16/FP32/INT32/BF16；`alpha` 是 ACL Scalar，执行前物化为单元素 Tensor |
| 执行类型 | 单 Kernel；唯一 Device 节点为 `AIV_VECTOR`，未发现可达 Cube/AIC 或 `MIX_CV` |
| Kernel 路线 | R-001 FP16 Axpy；R-002 FP32 Axpy；R-003 INT32 Muls+Add；R-004 BF16→FP32→BF16 |
| 核心资源 | 列表描述符位于 GM；每核使用三个 depth=2 的 TQue，BF16 另有 FP32 临时 Queue；算子 Kernel 不读 workspace |
| 来源闭包 | `source_closed`：公开 V2 API→L0 launcher→arch22 Tiling/POD→dtype router→公共 Foreach body/helper→A2/A3 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入是两个长度相同的 TensorList。对每个列表位置 `i`，只处理 `x1_i` 的有效元素域，并计算 `out_i = x1_i + alpha * x2_i`。A2/A3 的 FP16/FP32 路线以 `Axpy(x1, x2, alpha)` 实现；INT32 先原地缩放第二输入的 UB 副本，再相加；BF16 逐段转为 FP32、计算、再以 `CAST_RINT` 回写 BF16。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；单 Tensor 0–8 维；公开格式 ND | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；V2 文档不支持空 Tensor 和非连续 Tensor |
| `x2` | 与 `x1` 列表等长，逐位置 shape 相同 | 与 `x1` 相同 | 乘以 `alpha` 后与 `x1` 相加 |
| `alpha` | ACL Scalar；API 内转为单元素 Tensor | FP16 输入支持 FP16/DOUBLE；FP32、BF16 支持 FP32/DOUBLE；INT32 支持 INT32/INT64 | 转换后 FP16/FP32/INT32 路线与数据 dtype 相同；BF16 路线为 FP32 |
| `out` | TensorList；V2 API 校验为逐位置 shape 与 `x1` 相同 | 与 `x1` 相同 | 输出不做 ViewCopy，公开接口要求连续 |

### 入口与版本

固定仓同时公开 `aclnnForeachAddList` 与 `aclnnForeachAddListV2`；V2 仅把 `alpha` 的入口类型从 `aclTensor*` 改为 `aclScalar*`，核心 `ForeachAddList` Host/Tiling/Kernel 相同。A2/A3 公开产品表均支持 V2，因此本卡只提取最新 V2。`ExecForeachAddListV2GetWorkspaceSize` 把 Scalar 转成 Tensor，再由 `l0op::ForeachAddListV2` 注册一个 `ForeachAddList` AICore launcher。

## Host/Launcher 执行流程

### 参数校验与前处理

`ForeachAddListV2CheckParams` 依次执行空指针、平台/dtype、shape、format 校验。空 TensorList 在任何 launcher 创建前返回成功、workspace 为 0。非空时，API 对 `x1`、`x2` 的每个元素调用框架 `l0op::Contiguous`，构造 executor 所有的连续 TensorList；公开 V2 合同仍标记输入不支持非连续 Tensor，不能据内部调用扩展公开支持面。`alpha` 在 BF16 数据路线转为 FP32 Tensor，其他路线转为 `x1[0]` 的 dtype。任一转换或 launcher 添加失败立即返回错误，不启动主 Kernel。

### 参数、Workspace 与生命周期

API 第一段创建并移交 `aclOpExecutor`；连续化临时 Tensor、Scalar Tensor 和 launcher 都由 executor 持有，第二段 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)` 按同一 stream 执行。arch22 Tiling 将 `workspaces[0]` 固定为 32 字节，但 Kernel entry 明确把 `userWS` 设为 `nullptr`；因此这 32 字节不承载 ForeachAddList 节点间数据。API 返回的总 workspace 仍可能包含框架 Contiguous 节点所需空间，固定仓在框架边界内不展开其公式。

### 启动图与失败传播

```text
aclnnForeachAddListV2GetWorkspaceSize
  -> [每个 x1/x2，若框架需要] l0op::Contiguous（框架节点，executor 持有）
  -> Scalar -> 单元素 Tensor
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachAddList)
aclnnForeachAddListV2
  -> CommonOpExecutorRun(..., stream)
     -> foreach_add_list<<<blockDim=needCoreNum, stream>>>(x1_list, x2_list, alpha, out_list, workspace, tiling)
```

主算子只有一个 Device 节点，无算子自有跨 Kernel 边。Contiguous 的产出 Tensor 由 executor 保存至主 Kernel 消费结束；同一 executor/stream 给出顺序，前置节点失败会阻断主 Kernel。主 launcher 添加失败时 `l0op::ForeachAddListV2` 返回空，第一段映射为 `ACLNN_ERR_INNER_NULLPTR`。

## Route 与参数设计

### 分核与资源计算

设全部 Tensor 元素数之和为 `N`，dtype 字节数为 `s`，`E=32/s`。`GetNeedCoreNum` 取 `min(platformAivCoreNum, ceil(N/E))`，零元素时先钳为 1，随后再钳到 `MAX_CORE_CONT=50`。`AssignDataToEachCore` 以 32 字节 block 为分配单位：`B=ceil(N/E)`，基础每核元素数为 `floor(B/C)*E`，前 `B mod C` 核各多 `E` 个元素；每核区间映射成首/尾 Tensor 索引和含首尾 offset。

设平台 UB 字节为 `Uplat`，POD 序列化字节为 `S`。非 BF16 时 `inputsTensorUbSize=floor32((Uplat-S)/6)`；BF16 时为 `floor64(((Uplat-S)/10)/6)`。该字段是每个基础 buffer 的字节容量，不是元素数。Kernel 使用 `maxDataCount=inputsTensorUbSize/s`；BF16 使用 `maxDataCount=9*inputsTensorUbSize/2`、`maxCastDataCount=inputsTensorUbSize/4`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 Axpy | key 1；`ForeachOneScalarTernary<half,half,AddListFloatAdapter<half>>` | `dataType == DT_FLOAT16` | 检查下一 dtype；API 不允许其他 dtype |
| 2 | R-002 FP32 Axpy | key 2；`ForeachOneScalarTernary<float,float,AddListFloatAdapter<float>>` | `dataType == DT_FLOAT` | 检查下一 dtype |
| 3 | R-003 INT32 Muls+Add | key 3；`ForeachOneScalarTernary<int,int,AddListNormalAdapter<int>>` | `dataType == DT_INT32` 且编译目标 `__CCE_AICORE__ >= 220` | 检查下一 dtype |
| 4 | R-004 BF16 cast | key 4；`ForeachOneScalarTernary<bfloat16_t,float,AddListFloatAdapter<float>>` | `dataType == DT_BF16`，`__CCE_AICORE__ >= 220`，且非 NPU arch 3003/3113；A2/A3 构建可达 | 未知 key 不执行 body |

```text
dtype FP16 -> key 1 -> R-001
else dtype FP32 -> key 2 -> R-002
else dtype INT32 -> key 3 -> R-003
else dtype BF16 -> key 4 -> R-004
else -> API dtype 校验失败；若异常 key 到达 entry，则无分支执行
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1 | FP16，公开 shape/列表约束通过，A2/A3 | 单 AIV Kernel | `AddListFloatAdapter<half>`→`Axpy<half,half>` | 三个 depth=2 Queue；V pipe barrier；无 workspace 数据 | 尾 tile 用 `DataCopyPad`，只回写有效字节 |
| R-002 | 2 | FP32，其余同上 | 单 AIV Kernel | `AddListFloatAdapter<float>`→`Axpy<float,float>` | 同 R-001 | 同 R-001 |
| R-003 | 3 | INT32，其余同上 | 单 AIV Kernel | `AddListNormalAdapter<int>`→`Muls`→`Add` | 三个 depth=2 Queue；Muls/Add 间 barrier | 同 R-001 |
| R-004 | 4 | BF16，其余同上 | 单 AIV Kernel | `InnerComputer<bfloat16_t,float>` + `AddListFloatAdapter<float>` | 三个放大为 `9U` 的 depth=2 Queue，加一个 `2U` FP32 Queue；多次 V barrier | 每个 cast 子段尾长独立处理，最终 `CAST_RINT` |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32 B 分核 | `E=32/s`，`C<=ceil(N/E)` | 每核边界按 DMA block 粒度分配 | 全部 | 零元素仍置 1 核 | `ForeachCommonTiling::GetNeedCoreNum/AssignDataToEachCore` |
| UB 六分 | `U=floor32((Uplat-S)/6)` | 三个 depth=2 Queue 共需 `6U` | R-001..003 | 固定仓无备用 Route | `DivideUbMemory8` |
| BF16 保守缩放 | `U=floor64(((Uplat-S)/10)/6)` | 三个 `9U×depth2` Queue 加 `2U` FP32 临时区，总计 `56U` | R-004 | 固定仓无备用 Route | `DivideUbMemory8`; `KernelForeachUnary::Init`; `ForeachOneScalarTernary::Init` |
| 尾块 | `dataCount % maxDataCount != 0` | 非整 tile 搬运不能用普通整块 DataCopy | 全部 | 使用 `DataCopyPad` | `SingleTensorProcess`; `CopyIn`; `CopyInPlus`; `CopyOut` |
| Axpy 输出别名检查 | `dstLocal.GetPhyAddr()!=srcLocal1.GetPhyAddr()` | Axpy 原地更新 `srcLocal1`，独立 outQueue 需要复制 | R-001/R-002/R-004 内 FP32 段 | 地址相同则省略复制 | `AddListFloatAdapter` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 第 1 字段；宏展开后的 `sizeof/offsetof` 固定仓不可见 | 字节 | `FillTilingData`: `inputsTensorUbSize` | `KernelForeachBase::ParseTilingData`，计算 tile/Queue 容量 | 32 B 或 64 B 向下对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第 2 字段；宏展开布局不可见 | 元素/每 Tensor | `Init`: `GetShapeSize()` | `Process` 计算 Tensor 有效尾 | 最多记录 50 项 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 第 3 字段；宏展开布局不可见 | Tensor 索引/每核 | `AssignDataToEachCore` | `ParseTilingData` | `blockIdx` 索引 |
| 同上 | `tensorEndList` | `uint16_t[50]` | 第 4 字段；宏展开布局不可见 | Tensor 索引/每核 | `AssignDataToEachCore` | `ParseTilingData` | 含首尾 |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 第 5 字段；宏展开布局不可见 | 元素/每核 | `AssignDataToEachCore` | `ParseTilingData` | 首 Tensor 内 offset |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 第 6 字段；宏展开布局不可见 | 元素/每核 | `AssignDataToEachCore` | `ParseTilingData` | 尾 Tensor 内含尾 offset |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | `foreach_add_list_tiling_arch22.cpp::Tiling4ForeachAddListTiling`; `ForeachCommonTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_add_list.cpp::foreach_add_list` | `AddListFloatAdapter<half>`；`ForeachOneScalarTernary`; `KernelForeachUnary/Base` | `op_kernel/CMakeLists.txt::add_kernel_sources(ascend910b,ascend910_93)` |
| R-002 | key 2 | 同上 | 同上 | 同上 | `AddListFloatAdapter<float>`；公共 body | 同上 |
| R-003 | key 3 | 同上 | 同上 | 同上 | `AddListNormalAdapter<int>`；公共 body | 同上 |
| R-004 | key 4 | 同上 | 同上 | 同上 | `InnerComputer<bfloat16_t,float>`；`AddListFloatAdapter<float>`；公共 body | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 后按 key 实例化一个 `ForeachOneScalarTernary`。`KernelForeachBase` 用 `blockIdx` 读取本核首尾 Tensor 与 offset；`KernelForeachUnary::Process` 遍历本核覆盖的 Tensor，把 ListTensor 描述符解引用为 GM 地址，并按 `maxDataCount` 分 tile。每 tile 顺序是 `CopyIn(x1)`、`CopyInPlus(x2)`、`Compute`、`CopyOut(y)`。TQue 的 EnQue/DeQue 建立 MTE2→V→MTE3 依赖，Vector adapter 内用 `PipeBarrier<PIPE_V>` 约束原地结果。

### Route R-001：FP16 Axpy

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_add_list_v2.cpp` | `ForeachAddListV2GetDtypeSupportList` | A2/A3 arch 且首 Tensor FP16 | 允许并物化 FP16 alpha Tensor |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `op_kernel/foreach_add_list.cpp` | `foreach_add_list` | `TILING_KEY_IS(1)` | FP16 模板实例 |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list(key=1)
  -> ForeachOneScalarTernary<half,half,AddListFloatAdapter<half>>::Init
  -> KernelForeachUnary::Process
     -> CopyIn(x1) -> CopyInPlus(x2)
     -> InnerComputer::Compute -> AddListFloatAdapter -> Axpy
     -> CopyOut(out)
```

##### 源码等价伪码

```text
for tensor i in [tensorStart, tensorEnd]:
  begin/end = POD offsets for boundary tensors, otherwise full tensor
  for tile k in ceil((end-begin)/maxDataCount):
    valid = min(maxDataCount, remaining)
    enqueue x1[begin+k*maxDataCount : +valid]
    enqueue x2[begin+k*maxDataCount : +valid]
    Axpy(x1_ub, x2_ub, alpha, valid)       # x1_ub <- x1 + alpha*x2
    if out_ub is not x1_ub: copy aligned-up bytes into out_ub
    dequeue and write only valid elements to out GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x1/x2/y | `ListTensor[index] + cursorStart + tile*maxDataCount` | `valid` 元素 | FP16/ND | caller/前置节点 | CopyIn/CopyOut | executor 外部 Tensor 生命周期 | 整 tile DataCopy；尾 tile DataCopyPad |
| `dataQueue` | Queue 管理 | `depth2 * U` 字节 | FP16 | CopyIn | Axpy | Kernel `TPipe` | 计算后 FreeTensor |
| `InQueue_2` | Queue 管理 | `depth2 * U` 字节 | FP16 | CopyInPlus | Axpy | Kernel `TPipe` | 计算后 FreeTensor |
| `outQueue` | Queue 管理 | `depth2 * U` 字节 | FP16 | adapter | CopyOut | Kernel `TPipe` | GM 写完释放 |
| workspace | N/A | 0 个有效字节 | N/A | N/A | N/A | Kernel 不绑定 | entry 强制 `userWS=nullptr` |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in1["dataQueue · depth=2 · 2U bytes"]:1
  in2["InQueue_2 · depth=2 · 2U bytes"]:1
  out["outQueue · depth=2 · 2U bytes"]:1
```

##### 字节账与复用顺序

```text
U = floor32((Uplat - sizeof(ForeachCommonTilingData))/6)
UB_total = 2U + 2U + 2U = 6U <= Uplat - POD reservation
dynamic_ub = N/A
workspace_total_for_kernel = 0（Tiling 保留 32 字节但 body 不读）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` 当前 slot | MTE2 CopyIn | Axpy 原地改写/读取 | `FreeTensor` 后下一 tile | EnQue/DeQue + V barrier | 不得在 Compute 完成前释放 |
| `outQueue` 当前 slot | adapter 从 x1 UB 拷贝 | MTE3 CopyOut | GM 提交后下一 tile | EnQue/DeQue/FreeTensor | 尾部只提交 valid 字节 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | TQue EnQue/DeQue | CopyIn | Compute | 两路输入 Queue slot | 尾部 `blockLen=valid*sizeof(half)` |
| 计算 | `Axpy<half,half>` + `PipeBarrier<PIPE_V>` | Axpy | UB copy | 原地 x1 UB | FP16 模板精度；`valid` 是有效 lane 数 |
| UB→GM | outQueue | adapter | CopyOut | 输出 slot | 非对齐内部 copy 向 32 B 上取整，GM 仅写 valid 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling/AssignDataToEachCore/DivideUbMemory8` | key、blockDim、每核区间和 UB 字段 |
| Kernel entry/router | `foreach/foreach_add_list/op_kernel/foreach_add_list.cpp` | `foreach_add_list`, key 1 | key→FP16 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_ternary.h`；`kernel_foreach_unary.h` | `Compute/CopyInPlus`; `SingleTensorProcess/CopyOut` | 双输入搬运、Axpy 与尾块 |

### Route R-002：FP32 Axpy

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_add_list_v2.cpp` | `ForeachAddListV2CheckDtypeValid` | 首 Tensor FP32，Scalar FP32/DOUBLE | 物化 FP32 alpha Tensor |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_add_list.cpp` | `foreach_add_list` | `TILING_KEY_IS(2)` | FP32 模板实例 |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list(key=2) -> ForeachOneScalarTernary<float,float,AddListFloatAdapter<float>>
  -> KernelForeachUnary::Process -> CopyIn×2 -> Axpy<float,float> -> CopyOut
```

##### 源码等价伪码

```text
沿 POD 分配的每核/每 Tensor 区间按 U/4 个元素切 tile；
每 tile 把 x1/x2 搬入 UB，执行 x1_ub <- x1_ub + alpha*x2_ub，
必要时复制到 outQueue，并只把 valid 个 FP32 元素写回。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x1/x2/y | 同 R-001，元素宽度 4 B | valid | FP32/ND | caller/CopyOut | Kernel/caller | Tensor 生命周期 | 尾部 Pad |
| 三个 TQue | Queue 管理 | 各 `depth2*U` | FP32 | CopyIn/compute | compute/CopyOut | `TPipe` | 总 6U |
| workspace | N/A | 0 | N/A | N/A | N/A | 未使用 | 保留值不参与计算 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in1["dataQueue · depth=2 · 2U bytes"]:1
  in2["InQueue_2 · depth=2 · 2U bytes"]:1
  out["outQueue · depth=2 · 2U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=6U；maxDataCount=U/4；dynamic_ub=N/A；kernel workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 Queue slot | CopyIn | Axpy 原地结果 | FreeTensor 后 | Queue + V barrier | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Compute | `Axpy<float,float>` | 两输入 Queue | outQueue | x1 UB | FP32；valid count 控制计算 |
| Tail | `DataCopyPad` | GM/UB | Queue/GM | 最后 tile | GM 回写仅 valid 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling` | key 2、U、blockDim |
| Kernel entry/router | `foreach/foreach_add_list/op_kernel/foreach_add_list.cpp` | key 2 分支 | FP32 实例 |
| Device body/helper | 同文件；`foreach_one_scalar_ternary.h` | `AddListFloatAdapter<float>`；`InnerComputer<float,float>` | FP32 Axpy 管线 |

### Route R-003：INT32 Muls+Add

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_add_list_v2.cpp` | `ForeachAddListV2CheckDtypeValid` | INT32 Tensor，Scalar INT32/INT64 | 物化 INT32 alpha |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `op_kernel/foreach_add_list.cpp` | key 3 分支 | `__CCE_AICORE__>=220` | INT32 模板 |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list(key=3) -> ForeachOneScalarTernary<int,int,AddListNormalAdapter<int>>
  -> CopyIn×2 -> Muls(x2_ub, x2_ub, alpha, valid)
  -> PipeBarrier<PIPE_V> -> Add(out_ub, x1_ub, x2_ub, valid) -> CopyOut
```

##### 源码等价伪码

```text
for each valid tile:
  x1_ub = GM x1; x2_ub = GM x2
  x2_ub[0:valid] *= alpha
  vector barrier
  out_ub[0:valid] = x1_ub[0:valid] + x2_ub[0:valid]
  write valid elements to GM y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | 同 R-002 | valid | INT32/ND | caller | Kernel/caller | Tensor | 32 B tail |
| 三个 TQue | Queue 管理 | 各 `depth2*U` | INT32 | CopyIn/Muls+Add | Compute/CopyOut | `TPipe` | x2 slot 原地缩放 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in1["dataQueue · depth=2 · 2U bytes"]:1
  in2["InQueue_2 · depth=2 · 2U bytes"]:1
  out["outQueue · depth=2 · 2U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=6U；maxDataCount=U/4；dynamic_ub=N/A；kernel workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x2 Queue slot | CopyInPlus | Muls 原地写；Add 读 | Add 后 FreeTensor | Muls/Add 间 V barrier | Add 前不得覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Scale→Add | `PipeBarrier<PIPE_V>` | `Muls` | `Add` | x2 UB | INT32 运算；源码未定义溢出扩展语义 |
| Tail | Queue + `DataCopyPad` | DMA | Vector/GM | valid 范围 | 仅 valid 元素参与 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `GetTilingKeyByDtypeOnly/DivideUbMemory8` | key 3 与 U |
| Kernel entry/router | `foreach/foreach_add_list/op_kernel/foreach_add_list.cpp` | key 3 分支 | INT32 实例 |
| Device body/helper | 同文件 | `AddListNormalAdapter<int>` | 原地 Muls 后 Add |

### Route R-004：BF16→FP32→BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_add_list_v2.cpp` | `ExecForeachAddListV2GetWorkspaceSize` | x1 dtype BF16 | Scalar 转 FP32 Tensor |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `op_kernel/foreach_add_list.cpp` | key 4 分支 | AICore>=220 且非 3003/3113 | BF16/FP32 模板 |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list(key=4)
  -> ForeachOneScalarTernary<bfloat16_t,float,AddListFloatAdapter<float>>::Init
  -> InnerComputer<bfloat16_t,float>::Compute
     -> ComputePerCast: Cast x1 BF16->FP32; Cast x2 BF16->FP32
     -> AddListFloatAdapter<float> -> Axpy<float,float>
     -> Cast FP32->BF16(CAST_RINT)
  -> CopyOut
```

##### 源码等价伪码

```text
for each BF16 tile of at most 9U/2 elements:
  for each cast chunk of at most U/4 elements:
    tmp0 = Cast(x1_bf16, FP32, CAST_NONE)
    tmp1 = Cast(x2_bf16, FP32, CAST_NONE)
    tmp0 = tmp0 + alpha_fp32 * tmp1
    out_bf16 = Cast(tmp0, BF16, CAST_RINT)
  CopyOut only the tile's valid BF16 bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| three BF16 TQue | Queue 管理 | 每个 `depth2 * 9U` | BF16 | DMA/compute | cast/CopyOut | `TPipe` | 双缓冲 |
| `float32Queue` | `tmp0[0:U/4]`, `tmp1[U/4:2U/4]` | `1 * 2U` 字节 | FP32 | Cast | Axpy/Cast | 整个 Process 持有 | 两个连续子区，跨 cast chunk 复用 |
| GM | 列表地址 + offset | valid BF16 元素 | BF16/ND | caller/CopyOut | Kernel/caller | Tensor | 尾部 Pad |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in1["dataQueue · depth=2 · 18U bytes"]:1
  in2["InQueue_2 · depth=2 · 18U bytes"]:1
  out["outQueue · depth=2 · 18U bytes"]:1
  fp32["float32Queue · depth=1 · 2U bytes"]:1
```

##### 字节账与复用顺序

```text
U = floor64(((Uplat-S)/10)/6)
UB_total = 18U + 18U + 18U + 2U = 56U <= Uplat-S（Host 以约 60U 保守预算）
maxDataCount=9U/2；maxCastDataCount=U/4；dynamic_ub=N/A；kernel workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 tmp0 | x1 Cast | Axpy 原地结果、BF16 Cast | 下一 cast chunk | 每步 `PipeBarrier<PIPE_V>` | BF16 Cast 完成后才覆盖 |
| FP32 tmp1 | x2 Cast | Axpy | 下一 cast chunk | V barrier | Axpy 完成后才覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `Cast(...CAST_NONE)` + V barrier | 两路 BF16 | Axpy | tmp0/tmp1 | FP32 中间计算 |
| FP32 compute | `Axpy<float,float>` | tmp0/tmp1 | downcast | tmp0 | FP32 alpha、FP32 运算 |
| FP32→BF16 | `Cast(...CAST_RINT)` | Axpy | CopyOut | BF16 outQueue | round-to-nearest-integer mode as coded；有效 count 控制 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory8` | BF16 `/10` 保守因子与 64 B 对齐 |
| Kernel entry/router | `foreach/foreach_add_list/op_kernel/foreach_add_list.cpp` | key 4 分支 | BF16 storage/FP32 compute 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_ternary.h` | `InnerComputer<bfloat16_t,float>::ComputePerCast` | 双 Cast、Axpy、CAST_RINT |

## 算子特定优化

### Trick T-001：Axpy 原地计算并按物理别名跳过冗余 UB Copy

#### 触发与源码依据

- 适用 Route：R-001、R-002，以及 R-004 的 FP32 cast chunk。
- 触发条件：选择 `AddListFloatAdapter<T>`；若 `dstLocal` 与 `srcLocal1` 物理地址相同，则无需第二次 UB 搬运。
- Golden 来源：[`foreach/foreach_add_list/op_kernel/foreach_add_list.cpp:37-51`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/foreach_add_list.cpp#L37-L51)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__aicore__ void AddListFloatAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal1, const LocalTensor<T>& srcLocal2,
    const T& scalarVal, const int32_t& uValue)
{
    Axpy<T, T>(srcLocal1, srcLocal2, scalarVal, uValue);
    if (dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()) {
        PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

`Axpy` 直接把 `x1 + alpha*x2` 写回 `srcLocal1`，省去独立计算临时区；只有 output LocalTensor 不与该区物理别名时才做 UB→UB copy。copy 长度不对齐 32 B 时向上补到完整 block，但后续 GM CopyOut 仍以 `valid` 字节限制有效域。

#### 迁移边界

- 适用：具有 `dst = src1 + scalar*src2` 语义、允许输入 UB 副本被覆盖的 Vector 路线。
- 前提：输入已搬入私有 UB；原地 Axpy 后有 V pipe 同步；尾部写回只提交有效元素。
- 失效：需要保留 x1 UB、dtype 不支持 Axpy，或物理输出区不一致时必须保留 copy/改用 R-003 类两步实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓可验证的错误边界是：在独立物理地址下直接省略 UB copy 会令 outQueue 未获得结果；源码以 `GetPhyAddr()` 判断避免该错误。

### Trick T-002：BF16 以两个 FP32 子区分段复用

#### 触发与源码依据

- 适用 Route：R-004。
- 触发条件：`T=bfloat16_t`，每段 `dataCount<=maxCastDataCount=U/4`。
- Golden 来源：[`foreach/foreach_utils/op_kernel/foreach_one_scalar_ternary.h:68-81`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_ternary.h#L68-L81)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
op(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], scalarVal, dataCount);
PipeBarrier<PIPE_V>();
Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
```

#### 机制与设计原因

一个 `2U` FP32 Queue 被静态切为两个 `U` 子区，分别承载 x1/x2 cast 结果；x1 子区再被 Axpy 原地覆盖为结果。每个 cast chunk 完成下转后即可复用两个子区，避免为完整 `9U/2` BF16 tile 同时物化两份 FP32 数据。

#### 迁移边界

- 适用：低精度存储、FP32 中间计算且可分段处理的逐元素算子。
- 前提：各阶段有效 count 相同，子区互不重叠，V barrier 保证覆盖顺序。
- 失效：需要跨 chunk 依赖或同时保留两个输入与结果时不可采用同样复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key1 | key2 | key3 | key4 | dtype 唯一映射 |
| 模板与 dtype | half/half | float/float | int/int | bf16/float | BF16 升精度 |
| 循环与分核 | 公共按核/Tensor/tile | 同左 | 同左 | tile 内另分 cast chunk | FP32 临时区有限 |
| Queue/TBuf/寄存器 | 3×depth2×U | 同左 | 同左 | 3×depth2×9U + 1×2U | Host 保守 UB 预算 |
| 内存总占用与复用 | 6U；x1 原地 | 6U；x1 原地 | 6U；x2 原地缩放 | 56U；tmp0 原地 | 运算 primitive 不同 |
| 搬运/Vector API | Axpy | Axpy | Muls+Add | Cast+Axpy+Cast | dtype 能力与精度 |
| 同步、精度与 tail | V barrier；FP16 | V barrier；FP32 | Muls/Add barrier | 多 V barrier；FP32 中间、CAST_RINT | tail 均 DataCopyPad |
| fallback/失败边界 | 未知 key 无计算 | 同左 | 需 AICore>=220 | 需 BF16 编译 guard | API 先拦截不支持 dtype |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 V2 支持 | 公开 V2 支持 | 支持，但见独立 `_950` 卡 |
| Host/Launcher | `arch22` 公共 Tiling，compute unit `ascend910b` | `arch22` 公共 Tiling，compute unit `ascend910_93` | `arch35` 独立 Tiling |
| Kernel/资源 | AIV Vector 公共 Kernel | AIV Vector 公共 Kernel | SIMT VF、无 AIV Queue |

## 限制与失败边界

- 公开 V2 合同要求列表等长、逐位置 shape 相同、维数不超过 8、输入输出 dtype 一致，且输入/输出不支持非连续 Tensor；内部 `Contiguous` 不作为扩大公开能力的依据。
- 空 TensorList 在 API 层成功短路；公开文档声明单个空 Tensor 不支持。固定 API/Tiling 未给出独立的单 Tensor `numel>0` ACLNN 检查，不能据此推测扩展支持。
- A2/A3 Tiling 最多记录 50 个 Tensor/50 个核；`Init` 遍历上限为 50，固定源码没有公开的更大列表支持保证。
- arch22 Tiling 预留 workspace 32 字节，但 Kernel 不读；executor 总 workspace 还可能包含框架 Contiguous 节点，公式在固定仓框架边界不可见。
- `ForeachCommonTilingData` 由外部 Tiling 宏生成 ABI；本仓能闭合字段顺序、类型和 producer/consumer，不能闭合宏展开后的 `sizeof/alignof/offsetof`。
- INT32 溢出、Vector primitive 的更底层舍入与异常值细节不在固定仓可见范围；不推测补齐。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_add_list/docs/aclnnForeachAddListV2.md` | `aclnnForeachAddListV2*` | 平台、原型、公开 dtype/shape/连续性约束 |
| 入口实现 | `foreach/foreach_add_list/op_host/op_api/aclnn_foreach_add_list_v2.cpp` | `ExecForeachAddListV2GetWorkspaceSize` | API 校验、Contiguous、Scalar 转 Tensor、executor |
| Launcher | `foreach/foreach_add_list/op_host/op_api/foreach_add_list_v2.cpp` | `l0op::ForeachAddListV2` | 单 AICore launcher 与失败传播 |
| Host/Tiling | `foreach/foreach_add_list/op_host/arch22/foreach_add_list_tiling_arch22.cpp` | `Tiling4ForeachAddListTiling` | 绑定 opCode 28 与公共 Tiling |
| OpDef/平台 | `foreach/foreach_add_list/op_host/foreach_add_list_def.cpp` | `ForeachAddList::ForeachAddList` | dtype/layout 与 A2/A3 compute-unit 注册 |
| Host 构建 | `foreach/foreach_add_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3 选择 arch22 Tiling |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace、POD producer |
| Host selector | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype 到 key 1–4 |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | 字段顺序和数组上限 |
| Kernel entry/body | `foreach/foreach_add_list/op_kernel/foreach_add_list.cpp` | `foreach_add_list`; adapters | key router 与实际 Vector 操作 |
| Kernel body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_ternary.h` | `ForeachOneScalarTernary`; `InnerComputer` | 第二输入、scalar、BF16 cast 管线 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、tile、CopyIn/CopyOut、tail |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD 解析和列表地址解引用 |
| 构建接线 | `foreach/foreach_add_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 compute unit 选择 arch22 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段表达每核列表区间和 tile 字节 | 字段顺序、50 项数组、含尾 offset | `ForeachCommonTilingData`; `ParseTilingData` |
| 调用顺序 | 每 tile x1 CopyIn→x2 CopyIn→Compute→CopyOut | Queue 的 EnQue/DeQue/FreeTensor 顺序 | `SingleTensorProcess` |
| workspace/内存 | 非 BF16 6U；BF16 56U；主 Kernel 不读 workspace | depth=2 和 BF16 9 倍/2U 切片 | 各 `InitBuffer` |
| 同步 | Queue 管线 + V barrier | 原地 Axpy/Muls 后不得提前覆盖 | adapters；`InnerComputer` |
| 精度与尾块 | BF16 FP32 中间、CAST_RINT；尾块 Pad | `valid` count 和 GM 有效字节 | `ComputePerCast`; `CopyOut` |
| 失败边界 | API 校验、Tiling graph failure、launcher failure | 未知 key 无 body；无源码 fallback | `ForeachAddListV2CheckParams`; router |