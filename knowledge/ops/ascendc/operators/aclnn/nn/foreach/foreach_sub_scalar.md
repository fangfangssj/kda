---
type: Operator
title: "ForeachSubScalar A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachSubScalarV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/docs/aclnnForeachSubScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/op_api/foreach_sub_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/arch22/foreach_sub_scalar_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/foreach_sub_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:22:25Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachSubScalar A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSubScalarV2GetWorkspaceSize` + `aclnnForeachSubScalarV2`，公开 ACLNN 两段式接口；同族旧入口 `aclnnForeachSubScalar` 的 `scalar` 是 `aclTensor`，V2 改为 `aclScalar`，A2/A3 取最新公开 V2。 |
| 计算语义 | 对列表中每个张量逐元素执行 `y_i[j] = x_i[j] - scalar`。 |
| 输入输出 | `x/out` 为 ND TensorList，元素 dtype 为 FP16/FP32/INT32/BF16；`scalar` 为 `aclScalar`，进入 L0 前转成与计算路线匹配的单元素 Tensor。 |
| 执行类型 | 单 Kernel：`ForeachSubScalar` 节点为 `AIV_VECTOR`；无 Cube/AIC、SIMT、NDDMA 或通信节点。 |
| Kernel 路线 | R-001 FP32 直算；R-002 INT32 直算；R-003 FP16→FP32→FP16；R-004 BF16→FP32→BF16。 |
| 核心资源 | 动态 TensorList GM 指针表、双缓冲 `dataQueue/outQueue`、32 B scalar `TBuf`；FP16/BF16 另有单深度 FP32 Queue；算子 Kernel 不消费 user workspace。 |
| 来源闭包 | `source_closed`：公开 V2 API→L0 launcher→arch22 tiling/POD→dtype router→公共 Foreach AIV body/helper→A2/A3 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`，输出 `y` 与 `x` 等长且逐项同 shape；对任意有效 `i,j`，`y_i[j]=x_i[j]-s`。FP32/INT32 在自身类型中执行 `Sub`；FP16/BF16 先将 tile 转为 FP32，与转为 FP32 的标量相减，再以 `CAST_RINT` 回写输入 dtype。各输出只写 `x_i` 的有效元素。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 0–8 维，公开格式 ND | FP32、FP16、BF16、INT32，列表内一致 | V2 文档将非连续输入标为不支持；实现仍逐项调用框架 `Contiguous`。允许空 Tensor；空 TensorList 在 API 层直接成功返回。 |
| `scalar` | `aclScalar`，无 Tensor shape/layout | 当 x=FP32/BF16 时 FP32/DOUBLE；x=FP16 时 FP16/DOUBLE；x=INT32 时 INT32/INT64 | executor 将其窄化/转换成 FP32、FP16 或 INT32 单元素 Tensor；BF16 路线固定生成 FP32 Tensor。 |
| `out` | TensorList；逐项 shape 与 `x` 相等，格式相等且非私有 | 与 `x` 一致 | V2 文档与 API 校验要求连续输出；实现要求列表内 dtype 一致。 |
| `workspaceSize` | 标量输出 | `uint64_t` | 由 executor 汇总框架 `Contiguous`、scalar-to-tensor 与算子任务所需空间；固定源码未公开 executor 内部切片公式。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段构造，第二段交给 `CommonOpExecutorRun` 在指定 stream 执行。 |

### 入口与版本

公开文档同时列出 `aclnnForeachSubScalar` 与 `aclnnForeachSubScalarV2`。V2 是同一数学语义的更新入口，差异是 `scalar` 从 `aclTensor*` 改成 `aclScalar*`；V2 的产品表明确支持 A2/A3，故两平台均选择 V2。`ExecForeachSubScalarV2GetWorkspaceSize` 将 scalar 转成 Tensor 后复用内部算子类型 `ForeachSubScalar`，不是不同语义变体。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 依次执行：非空检查；平台/dtype 检查（仅 DAV_2201 或 regbase arch，当前 A2/A3 为 DAV_2201 路线）；列表内 dtype 与 out dtype 检查；按 x dtype 检查 scalar 类型；逐项 shape 相等、维数不超过 8；输入输出 storage format 相等且输入不是私有格式。随后若 `x->Size()==0`，返回 `workspaceSize=0` 且不添加 Kernel。非空时逐项执行框架 `l0op::Contiguous`，构造新 TensorList；BF16 将 scalar 转为 FP32 Tensor，其余转为 x dtype；失败立即返回对应 ACLNN 错误。输出没有后置 ViewCopy。

### 参数、Workspace 与生命周期

`l0op::ForeachSubScalarV2` 以 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachSubScalar, OP_INPUT(x, scalar), OP_OUTPUT(out))` 添加单个语义 Kernel 节点。第一段把 executor 释放给调用者并返回 executor 汇总的 workspace 大小；调用者保持 `x/scalar/out/executor/workspace` 有效至第二段在 stream 上完成。arch22 tiling 将设备算子 workspace 设为 32 B，但 entry 明确把 `userWS=nullptr`，Device body 不读取该区；框架 `Contiguous` 与 scalar Tensor 的具体 workspace offset/owner 由外部 executor 实现管理，固定仓不可见。

### 启动图与失败传播

```text
aclnnForeachSubScalarV2GetWorkspaceSize
  -> [每个 x_i] l0op::Contiguous（框架边界；失败则不建后续任务）
  -> ConvertToTensor(scalar, target_dtype)（框架边界）
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachSubScalar)
       -> arch22 Tiling4ForeachSubScalarTiling
       -> foreach_sub_scalar<<<blockDim, stream>>>(xList, scalarTensor, yList, workspace, POD)
aclnnForeachSubScalarV2
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

核心语义是单 Device 节点：entry 为 `foreach_sub_scalar`，`blockDim=min(AIV核心数, ceil(totalDataCount/elementsPerBlock), 50)`，空总域取 1；POD 为 `ForeachCommonTilingData`。输入/输出是 executor 所持 TensorList GM 描述对象，scalar Tensor 在 executor 生命周期内有效。没有 Kernel 间 workspace 边；同一 executor/stream 维护框架前处理到 Kernel 的顺序。任何 Contiguous、ConvertToTensor、launcher 添加、tiling 或执行失败都会阻断该单节点或向 ACLNN 返回失败。

## Route 与参数设计

### 分核与资源计算

`Init` 扫描最多 50 个动态输入，记录每个 storage shape 的元素数，`dataTypeSize∈{2,4}`，`elementsPerBlock=32/dataTypeSize`。`GetNeedCoreNum` 先算 `ceil(totalDataCount/elementsPerBlock)`，零时修正为 1，再取 AIV 核数下界并封顶 50。`AssignDataToEachCore` 以 32 B block 为分配单位：前 `blockCount % needCoreNum` 核比其余核多一个 block，并用 `tensorStart/End` 及元素 offset 表达一个 core 跨 Tensor 边界的连续逻辑区间。

令平台 UB 为 `U`，POD 序列化大小为 `D`，`B=inputsTensorUbSize`：FP32/INT32 取 `B=floor32((U-D-32)/4)`；FP16/BF16 先令 `total=floor((U-D-32)/10)`，再取 `B=floor64(total/4)`。tile 上限为直算 `B/sizeof(T)` 元素；半精度路线令 `totalTensorUbSize=9B`、`maxDataCount=9B/2` 元素、每次 FP32 cast 子段 `maxCastDataCount=B/4` 元素。TilingKey 分别为 FP16=1、FP32=2、INT32=3、BF16=4。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 直算 | key 2；`<float,float,Sub,2,1>` | API 接受且首个 x dtype=`DT_FLOAT`；tiling `GetTilingKeyByDtypeOnly` 返回 2 | 检查下一 dtype；不支持 dtype 在 API/tiling 失败 |
| 2 | R-003 FP16 升精度 | key 1；`<half,float,Sub,2,1>` | x dtype=`DT_FLOAT16` | 检查下一 dtype |
| 3 | R-002 INT32 直算 | key 3；`<int,int,Sub,2,1>` | x dtype=`DT_INT32` | 检查下一 dtype |
| 4 | R-004 BF16 升精度 | key 4；`<bfloat16_t,float,Sub,2,1>` | x dtype=`DT_BF16` 且编译 guard `__CCE_AICORE__>=220` 并排除 NPU arch 3003/3113 | guard 未编译时无 key-4 body；公开 A2/A3 binary/config 使用支持该实例的平台 |

```text
if dtype == FP32: key=2 -> R-001
else if dtype == FP16: key=1 -> R-003
else if dtype == INT32: key=3 -> R-002
else if dtype == BF16: key=4 -> R-004
else: API/tiling failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 直算 | 2 | A2/A3；FP32；API shape/format 校验通过 | 单 AIV Kernel | `ForeachOneScalarBinaryLevelZeroApi<float,float,Sub,2,1>`→通用 `InnerComputer` | 双缓冲 B-byte in/out + 32 B scalar；Queue 与 V→MTE3/MTE3→MTE2 event | 尾 DMA 用 byte-length `DataCopyPad`；Vector 尾用单 repeat mask |
| R-002 INT32 直算 | 3 | A2/A3；INT32；其余同上 | 单 AIV Kernel | `<int,int,Sub,2,1>`→通用 `InnerComputer` | 同 R-001 | 同 R-001；INT32 在 int 类型中相减 |
| R-003 FP16 升精度 | 1 | A2/A3；FP16 | 单 AIV Kernel | `<half,float,Sub,2,1>`→half 专用 `InnerComputer` | 双缓冲 9B-byte in/out + 2B FP32 Queue + scalar TBuf；V pipe barriers | 每个 cast 子段独立处理，尾 mask 后 `CAST_RINT` |
| R-004 BF16 升精度 | 4 | A2/A3；BF16；编译 guard 成立 | 单 AIV Kernel | `<bfloat16_t,float,Sub,2,1>`→BF16 专用 `InnerComputer` | 同 R-003 | 同 R-003；guard 不成立时 router 无此分支 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 每核 32 B 粒度 | `elementsPerBlock=32/sizeof(T)`，核数按 `ceil(total/elementsPerBlock)` | 普通 `DataCopy` 要求整块，边界由 tail API处理 | 全部 | 总域为 0 时仍设 1 核 | `ForeachCommonTiling::GetNeedCoreNum/AssignDataToEachCore` |
| UB 四份 | 直算 `B=floor32((U-D-32)/4)` | in/out 各 depth=2，另留 32 B scalar | R-001/R-002 | 无备用 Route；资源计算固定缩小 tile | `DivideUbMemory9` |
| 半精度衰减 | `total=(U-D-32)/10`、`B=floor64(total/4)` | 9B input/output queue 与 FP32 cast Queue 需要更大物理容量 | R-003/R-004 | 无备用 Route | `DivideUbMemory9`、specialized `Init` |
| repeat 上限 | 每批最多 255 repeat；每 repeat `256/sizeof(float)=64` 元素 | L0 Vector API repeat 字段限制 | 全部；直算 P 与半精度计算 P 都为 4 B | 切为多个 255-repeat 批次，不换 Route | `InnerComputer::Compute/ComputePerCast` |
| BF16 编译 guard | `__CCE_AICORE__>=220` 且 arch 非 3003/3113 | BF16 type/cast body只在该 guard 下编译 | R-004 | body 不存在；不允许落入其他 key | `op_kernel/foreach_sub_scalar.cpp` |
| TensorList 容量 | 公共 POD/Host 数组长度 50，`Init` 循环上界 50 | POD 无法表达更多 Tensor 元数据 | 全部 | 固定源码未见 `>50` 的显式 API 拒绝；复用实现必须外部限制为 50 | `MAX_TENSOR_CONT`、`ForeachCommonTiling::Init` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 首字段；宏展开后的 `sizeof/alignof/offsetof` 固定仓不可见 | byte | `DivideUbMemory9`→`FillTilingData` | `KernelForeachBase::ParseTilingData`，决定 Queue/tile 容量 | 32/64 B 向下对齐，按 dtype Route 解释 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第二字段；精确 offset 固定仓不可见 | element/Tensor | `Init` 取 storage shape size | `Process` 设置每 Tensor 尾端 | 未填项为 0；最多 50 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 第三字段；精确 offset 固定仓不可见 | Tensor index/core | `AssignDataToEachCore` | 按 `blockIdx` 读取 `tensorStart` | blockDim≤50 |
| 同上 | `tensorEndList` | `uint16_t[50]` | 第四字段；精确 offset 固定仓不可见 | Tensor index/core | 同上 | 读取 `tensorEnd` | 含首尾 |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 第五字段；精确 offset 固定仓不可见 | element/core | 同上 | 首 Tensor GM base offset | 0-based |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 第六字段；精确 offset 固定仓不可见 | element/core | 同上 | 尾 Tensor 有效末元素 | 0-based、包含末端 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | key 2 | arch22 `Tiling4ForeachSubScalarTiling`→`RunBigScalarKernelTiling` | `ForeachCommonTilingData` | `foreach_sub_scalar` key-2 branch | `ForeachOneScalarBinaryLevelZeroApi<float,float,...>`、`KernelForeachUnary` | op_host `SUPPORT_COMPUTE_UNIT` + op_kernel `ascend910b/ascend910_93` |
| R-002 INT32 | key 3 | 同上 | 同上 | key-3 branch | `<int,int,...>` + 公共 helper | 同上 |
| R-003 FP16 | key 1 | 同上 | 同上 | key-1 branch | `<half,float,...>` + half `InnerComputer` | 同上 |
| R-004 BF16 | key 4 | 同上 | 同上 | guarded key-4 branch | `<bfloat16_t,float,...>` + BF16 `InnerComputer` | 同上及 BF16 binary config |

## Kernel 实现

### 公共执行流程

entry 解析 `ForeachCommonTilingData`，按 key 构造唯一模板并调用 `Init/Process`；unknown key 没有 body，也不写输出。TensorList GM 首个 `uint64_t` 是指针数组相对偏移，右移 3 后得到 `uint64_t*` 指针表，再以 Tensor index 取数据地址。每核按 POD 的 start/end Tensor 和 offset 遍历；每个 Tensor 片段再按 `maxDataCount` 切 tile，执行 GM→VECIN Queue、`Sub`、VECOUT Queue→GM。scalar 只从 GM 读一次并 `Duplicate` 到 32 B `TBuf`；workspace entry 参数不参与 Device 地址计算。

### Route R-001：FP32 直算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | `CheckDtypeValid` | 首个 x 为 `DT_FLOAT`，scalar 为 FLOAT/DOUBLE，out 与 x 同 dtype | scalar 转 FP32 Tensor并添加节点 |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_sub_scalar.cpp` | `foreach_sub_scalar` | `TILING_KEY_IS(2)` | `<float,float,Sub,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar -> ForeachOneScalarBinaryLevelZeroApi<float,float,...>::Init/Process
  -> KernelForeachUnary::Process -> SingleTensorProcess
    -> CopyIn -> InnerComputer<float,float>::Compute -> CopyOut
```

##### 源码等价伪码

```text
scalarBlock[0:8] = duplicate(float(scalarGM[0]))
for tensor = start[blockIdx] .. end[blockIdx]:
  bind xGM/yGM at tensor pointer + startOffset
  for tile in ceil(validCount/(B/4)):
    n = min(B/4, remaining); CopyIn n FP32 (tail uses byte DataCopyPad)
    for batches of <=255 repeats: Sub(out, in, scalarBlock, mask=64, repeats, strides={1,1,0,8,8,0})
    final remainder: Sub(mask=n%64, repeats=1)
    wait V->MTE3; copy exactly n FP32; wait MTE3->MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | TensorList pointer[t] + core/tile offset | n elements | FP32/ND contiguous | caller/executor | CopyIn/CopyOut | caller至 stream 完成 | tail byte copy |
| `dataQueue` | Queue slot | depth 2 × B B | FP32 | CopyIn | Compute DeQue | `TPipe`，Compute 后 Free | 双缓冲 |
| `outQueue` | Queue slot | depth 2 × B B | FP32 | Compute | CopyOut | CopyOut 后 Free | 双缓冲 |
| `scalarOneBlockBuf` | 独立 TBuf | 32 B | FP32 broadcast block | Init Duplicate | Sub src1 | Kernel | repeat block stride 0 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 2B"]:2
  outQ["outQueue · depth=2 · 2B"]:2
  scalar["scalar TBuf · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2B + 2B + 32 = 4B + 32 <= U - D
dynamic_ub = N/A
workspace_total(device op) = 32 B declared, 0 B consumed by body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue slot | CopyIn | Compute `DeQue` | 后续 tile CopyIn | `FreeTensor` | depth=2 且 Queue 协议 |
| outQueue slot | Compute | CopyOut `DeQue` | 后续 Compute | V→MTE3、MTE3→MTE2 event 后 `FreeTensor` | DMA 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | TQue EnQue/DeQue | MTE2 | Vector | input slot | 尾 tile按有效字节搬入 |
| Compute | `Sub` | input/scalar | output Queue | 有效 lane | FP32；64 lane/repeat，最大 255 repeats，末 repeat mask |
| Compute→CopyOut | `V_MTE3` event | Vector | MTE3 | output slot | 只搬 `n*sizeof(float)` |
| CopyOut→复用 | `MTE3_MTE2` event | MTE3 | 下一 MTE2 | Queue物理区 | 防止提前复用 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling/DivideUbMemory9` | key、core区间、B与32 B workspace |
| Kernel entry/router | `foreach_sub_scalar/op_kernel/foreach_sub_scalar.cpp` | key-2 branch | 映射 FP32 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h` | generic `InnerComputer::Compute` | repeat 批次、mask、broadcast stride |

### Route R-002：INT32 直算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | `CheckDtypeValid` | x=INT32，scalar=INT32/INT64 | scalar 转 INT32 Tensor |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `op_kernel/foreach_sub_scalar.cpp` | key-3 branch | `TILING_KEY_IS(3)` | `<int,int,Sub,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar -> ForeachOneScalarBinaryLevelZeroApi<int,int,...>
  -> KernelForeachUnary::SingleTensorProcess -> CopyIn -> generic InnerComputer::Compute -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 相同地址/tile/repeat网络；B/4 个 INT32/tile，mask=64。
对每个有效 lane 执行 Sub<int>(x, scalar)，不经过 float cast；尾部按有效字节提交。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | pointer[t]+offset | n | INT32/ND | caller | Kernel | stream完成前 | tail byte copy |
| data/out Queue | slot | 各 depth 2 × B B | INT32 | CopyIn/Compute | Compute/CopyOut | 每 tile | 双缓冲 |
| scalar TBuf | 独立 | 32 B | INT32 | Duplicate | Sub | Kernel | stride 0 广播 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 2B"]:2
  outQ["outQueue · depth=2 · 2B"]:2
  scalar["scalar TBuf · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4B + 32 <= U - D
dynamic_ub = N/A
workspace_total(device op) = 32 B declared, 0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| data/out Queue slots | CopyIn/Compute | Compute/CopyOut | 下一 tile | Queue + V_MTE3/MTE3_MTE2 events | 与 R-001 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | Queue + explicit events | MTE2/Vector/MTE3 | 下一阶段 | Queue slot | INT32 Sub；源码未增加饱和或宽化；mask/tail同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | INT32→3 |
| Kernel entry/router | `foreach_sub_scalar/op_kernel/foreach_sub_scalar.cpp` | key-3 | INT32 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h` | generic `InnerComputer<int,int>` | 无 cast 的 Sub/repeat/tail |

### Route R-003：FP16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | scalar conversion | x=FP16；scalar=FP16/DOUBLE | 转 FP16 Tensor，Kernel Init 再读为 FP32 值 |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `op_kernel/foreach_sub_scalar.cpp` | key-1 | `TILING_KEY_IS(1)` | `<half,float,Sub,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar -> specialized Init/Process<half,float>
  -> SingleTensorProcess -> CopyIn
  -> InnerComputer<half,float>::Compute -> ComputePerCast
     -> Cast half->float -> in-place Sub<float> -> Cast float->half(CAST_RINT)
  -> CopyOut
```

##### 源码等价伪码

```text
tileCapacity = 9B/2 half elements; castCapacity = B/4 float elements
for tile, then each cast segment:
  Cast(fp32Buf, halfInput[segment], CAST_NONE, n)
  Sub(fp32Buf, fp32Buf, scalarBlock, mask/repeat, src1BlockStride=0)
  Cast(halfOutput[segment], fp32Buf, CAST_RINT, n)
copy only valid half bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dataQueue | slot | depth 2 × 9B B | FP16 | CopyIn | cast | tile | 双缓冲 |
| outQueue | slot | depth 2 × 9B B | FP16 | final cast | CopyOut | tile | 双缓冲 |
| float32Queue | 单 slot | 2B B | FP32 | cast/Sub | final cast | Kernel，Process末释放 | Sub 原地覆盖该 buffer |
| scalar TBuf | 独立 | 32 B | FP32 values | Duplicate | Sub | Kernel | FP16 scalar先转 FP32 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["dataQueue · depth=2 · 18B"]:2
  fp32["float32Queue · depth=1 · 2B"]:1
  outQ["outQueue · depth=2 · 18B"]:2
  scalar["scalar · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*9B + 2B + 2*9B + 32 = 38B + 32
B = floor64(((U-D-32)/10)/4), hence UB_total stays below reserved U-D bound
dynamic_ub = N/A; workspace body consumption = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 Queue | Cast input | in-place Sub、Cast output | 下一 cast segment | `PipeBarrier<PIPE_V>` around Cast/Sub/Cast | 当前 segment output 已落到 outLocal |
| Queue slots | CopyIn/Compute | Compute/CopyOut | 下一 tile | Queue/events | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| half→float | `PipeBarrier<PIPE_V>` + Cast NONE | input | Sub | FP32 buffer | 仅有效 n 元素 |
| FP32 Sub | barriers + in-place Sub | Cast | final Cast | FP32 buffer | 64 lane/repeat、255 上限、尾 mask |
| float→half | Cast `CAST_RINT` | Sub | CopyOut | outLocal | 回写 FP16，尾 DMA仅有效字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory9` | 半精度衰减与 64 B 对齐 |
| Kernel entry/router | `foreach_sub_scalar/op_kernel/foreach_sub_scalar.cpp` | key-1 | half/float 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h` | `InnerComputer<half,float>::ComputePerCast` | cast、原地 Sub、RINT |

### Route R-004：BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | scalar conversion | x=BF16、scalar=FLOAT/DOUBLE | scalar固定转 FP32 Tensor |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `op_kernel/foreach_sub_scalar.cpp` | guarded key-4 | CCE/arch guard成立 | `<bfloat16_t,float,Sub,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar -> specialized Init/Process<bfloat16_t,float>
  -> CopyIn -> InnerComputer<bfloat16_t,float>::ComputePerCast
  -> Cast BF16->FP32 -> in-place Sub -> Cast FP32->BF16(CAST_RINT) -> CopyOut
```

##### 源码等价伪码

```text
与 R-003 相同 tile/cast/repeat 网络；输入/输出本地类型替换为 bfloat16_t，
scalarGM 本身为 FP32；所有 Sub 在 FP32 buffer 中完成后 CAST_RINT 回 BF16。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| data/out Queue | slot | 各 depth 2 × 9B B | BF16 | CopyIn/final Cast | Cast/CopyOut | tile | 双缓冲 |
| float32Queue | 单 slot | 2B B | FP32 | Cast/Sub | final Cast | Kernel | 原地 Sub |
| scalar TBuf | 独立 | 32 B | FP32 | Duplicate | Sub | Kernel | 广播 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["dataQueue · depth=2 · 18B"]:2
  fp32["float32Queue · 2B"]:1
  outQ["outQueue · depth=2 · 18B"]:2
  scalar["scalar · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 38B + 32；B 的 producer 与 R-003 相同
dynamic_ub = N/A；workspace body consumption = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32/Queue slots | Cast/CopyIn | Sub/Cast/CopyOut | 下一 segment/tile | V barriers、Queue、events | 与 R-003 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全计算 | Cast + `PipeBarrier<PIPE_V>` + Sub + CAST_RINT | BF16 input/FP32 scalar | BF16 output | FP32 temporary | BF16 不直接相减；尾 mask和有效字节输出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory9` | BF16 UB与 key producer |
| Kernel entry/router | `foreach_sub_scalar/op_kernel/foreach_sub_scalar.cpp` | guarded key-4 | BF16编译边界 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h` | BF16 `ComputePerCast` | FP32升精度和 RINT |

## 算子特定优化

### Trick T-001：标量单次装载并以零 block-stride 广播

#### 触发与源码依据

- 适用 Route：R-001–R-004。
- 触发条件：非空 TensorList，任一支持 dtype key。
- Golden 来源：[`foreach_one_scalar_binary_level_zero_api.h:331-335`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h#L331-L335)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Base::pipe.InitBuffer(scalarOneBlockBuf, BYTES_PER_BLOCK);
inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
scalarValue = P(inScalarGM.GetValue(0));
scalarOneBlockLM = scalarOneBlockBuf.Get<P>();
Duplicate(scalarOneBlockLM, scalarValue, BYTES_PER_BLOCK / sizeof(P));
```

#### 机制与设计原因

每核只从 scalar GM 读取一个值，扩展为恰好 32 B 的一个 Vector block。后续 `Sub` 的 `BinaryRepeatParams{1,1,0,8,8,0}` 将 scalar 的 block/repeat stride 都置 0，因此所有 lane/repeat 复用该 block，避免随 x tile 重复搬运或物化整形标量 Tensor。mask 来自当前完整 repeat 或尾元素数，故广播不会扩大有效输出域。

#### 迁移边界

- 适用：单一标量作用于长向量/多 Tensor 的 elementwise 二元算子。
- 前提：scalar 在整个 Kernel 生命周期不变；primitive 支持 stride-0 source；P 类型与实际计算类型一致。
- 失效：逐 Tensor scalar list、逐元素第二输入或无法 stride-0 广播时必须换相应 list/binary 实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：半精度分段升精度并原地复用 FP32 buffer

#### 触发与源码依据

- 适用 Route：R-003、R-004。
- 触发条件：T 为 half 或 bfloat16_t；每个 tile 继续按 `maxCastDataCount=B/4` 分段。
- Golden 来源：[`foreach_one_scalar_binary_level_zero_api.h:127-151`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h#L127-L151)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint32_t i = 0; i < repeatBatchCnt; i++) {
    PipeBarrier<PIPE_V>();
    op(float32Tensor[offset], float32Tensor[offset], oneBlockData, elementsPerRepeat, MAX_REPEATS,
       {1, 1, 0, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    offset += MAX_REPEATS * elementsPerRepeat;
}
```

#### 机制与设计原因

输入先 cast 到单个 FP32 Queue，`Sub` 的 dst 与 src0 都指向 `float32Tensor[offset]`，以原地覆盖避免第二个 FP32 中间区；完成后立即 cast 到输出 Queue。`maxCastDataCount` 把巨大 half/BF16 tile 切成受控 FP32 段，`PipeBarrier<PIPE_V>` 保护同一物理区的 Cast/Sub/Cast 顺序。代价与 UB 上界已由 Host 的 `/10` 衰减和 `9B/2B/9B` 分配共同约束。

#### 迁移边界

- 适用：半精度输入但要求在 FP32 中执行的逐元素运算。
- 前提：算子允许输入 FP32 临时被原地覆盖；输出降精度使用 `CAST_RINT`；Host 与 Device 共享 B 的字节单位。
- 失效：需要保留 FP32 输入中间量、存在多消费者或需要不同 rounding 时必须增加 buffer/改变同步与 UB 公式。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 INT32 | R-003 FP16 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key2 | key3 | key1 | guarded key4 | dtype 唯一决定，无 fallback |
| 模板与 dtype | float/float | int/int | half/float | bf16/float | P 是实际 Sub 类型 |
| 循环与分核 | `B/4` tile | `B/4` tile | `9B/2` tile、`B/4` cast段 | 同 FP16 | 半精度扩大搬运 tile、限制 FP32工作段 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2B+2B+32 | 同 | 18B+2B+18B+32 | 同 | AIV Queue；无 dynamic UB/通信 |
| 内存总占用与复用 | `4B+32` | 同 | `38B+32`，FP32原地 | 同 | Host B 公式匹配 Device 分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Sub | 同 | 加 Cast | 加 Cast | 全部 AIV Vector |
| 同步、精度与 tail | Queue/events，FP32 | Queue/events，INT32 | V barriers，FP32计算/RINT | 同 | dtype语义 |
| fallback/失败边界 | unknown dtype失败 | 同 | 同 | 编译 guard | 不跨 key fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 公开支持 | V2 公开支持 | V2 公开支持，但见独立 `_950` 卡 |
| Host/Launcher | `ascend910b`→arch22 | `ascend910_93`→arch22 | arch35 独立 tiling |
| Kernel/资源 | 共用 AIV Vector body | 共用 AIV Vector body | SIMT body，无 AIV Queue |

## 限制与失败边界

- x/out 空指针、dtype/shape/format 不符、维数超过 8 或平台不是 DAV_2201/regbase 时第一段失败；空 TensorList直接返回且不启动 Kernel。
- arch22 公共 POD 最多表达 50 个 Tensor，Host `Init` 也只扫描 50 项；固定仓未见 V2 API 对更长列表的显式拒绝，因此工程复用必须把 `n<=50` 作为实现边界。
- FP16/BF16 使用 FP32 Sub 后 `CAST_RINT`；INT32 无宽化/饱和逻辑。固定源码未说明 INT32 溢出的额外处理。
- executor 的 Contiguous/scalar Tensor 具体 workspace 切片与自动生成 launch ABI 属框架边界，固定仓不可见；核心 Kernel 自身声明 32 B workspace但不读取。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sub_scalar/docs/aclnnForeachSubScalarV2.md` | `aclnnForeachSubScalarV2*` | A2/A3公开性、dtype/shape/错误约束 |
| 入口实现 | `foreach/foreach_sub_scalar/op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | `ExecForeachSubScalarV2GetWorkspaceSize` | 校验、Contiguous、scalar转换、executor |
| 入口实现 | `foreach/foreach_sub_scalar/op_host/op_api/foreach_sub_scalar_v2.cpp` | `l0op::ForeachSubScalarV2` | 单节点 launcher |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar/op_host/arch22/foreach_sub_scalar_tiling_arch22.cpp` | `Tiling4ForeachSubScalarTiling` | arch22 注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 分核、UB、POD producer |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD字段/容量 |
| Kernel entry/body | `foreach/foreach_sub_scalar/op_kernel/foreach_sub_scalar.cpp` | `foreach_sub_scalar` | dtype router |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h` | `ForeachOneScalarBinaryLevelZeroApi` | scalar广播、Sub、cast、CopyOut |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | TensorList循环、CopyIn/tile Queue |
| 构建接线 | `foreach/foreach_sub_scalar/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3→arch22映射 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 每核用 start/end Tensor+offset 描述区间 | 字段顺序/位宽、50项上限、B为字节 | `ForeachCommonTilingData` |
| 调用顺序 | Contiguous→scalar Tensor→单 Kernel | 失败必须阻断后续节点 | `ExecForeachSubScalarV2GetWorkspaceSize` |
| workspace/内存 | Device op无需 userWS；Queue按 B 分配 | depth=2必须计入；半精度是9B/2B/9B | `Init`、`DivideUbMemory9` |
| 同步 | Queue连接 MTE2/V，显式 event连接 V/MTE3与复用 | out slot不可在 MTE3完成前覆盖 | `CopyOut` |
| 精度与尾块 | half/BF16升FP32并RINT；byte tail + Vector mask | 不得改 cast/round、255 repeat上限 | `InnerComputer` |
| 失败边界 | dtype唯一 key，unknown key无 body | BF16 guard与 TensorList≤50边界 | router、`MAX_TENSOR_CONT` |
