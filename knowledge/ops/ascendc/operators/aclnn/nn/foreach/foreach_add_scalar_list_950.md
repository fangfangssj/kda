---
type: Operator
title: "ForeachAddScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachAddScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar_list/docs/aclnnForeachAddScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar_list/op_host/foreach_add_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar_list/op_kernel/arch35/foreach_add_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar_list/op_kernel/arch35/foreach_add_scalar_list_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary_scalar_list.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:30:28Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddScalarListGetWorkspaceSize` + `aclnnForeachAddScalarList`，公开两段式 ACLNN；固定仓无更高版本后缀。 |
| 计算语义 | 对列表位置 `i`：`y_i[j] = x_i[j] + scalars_i`，有效域为 `x_i` 元素域。 |
| 输入输出 | ND TensorList `x/out`：FP16/FP32/INT32/BF16；ScalarList：对应 FP32/FP32/INT64/FP32。 |
| 执行类型 | 单 Kernel：一个 arch35 `AIV_VECTOR` MicroAPI 节点，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP16→FP32 register→FP16；R-002 FP32；R-003 INT32；R-004 BF16→FP32 register→BF16。 |
| 核心资源 | GM `ListTensorDesc`、GM scalar list、depth-2 `inQueue/outQueue`、MicroAPI RegTensor/MaskReg；32 B 预留 workspace 不被 body 读取。 |
| 来源闭包 | `source_closed`：公开 ACLNN→生成 bridge 边界→OpDef/Infer→priority selector→950 POD→arch35 entry/body/helper→950 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

TensorList 长度 `n`，第 `i` 个输入的元素数 `N_i`。Kernel 对 `0<=j<N_i` 计算 `y_i[j]=x_i[j]+scalars_i`。公开约束允许 `out_i` 容量大于 `x_i`；Device 只写前 `N_i` 个元素，额外区域在固定仓中未定义。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 非空 TensorList；每项 ND、0–8 维 | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；支持空 Tensor 和非连续输入；OpDef `AutoContiguous()`。950 Tiling 明确要求列表项数 1–50。 |
| `scalars` | ScalarList，元素数等于 Tensor 数 | 浮点输入对应 FP32；INT32 对应 INT64 | 每项被相同 Tensor index 的计算读取一次。 |
| `out` | TensorList；ND、0–8 维 | 与 `x` 相同 | Tensor 数/dtype 一致；容量不少于输入；公开 ACLNN 不支持非连续输出。 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段接口返回；Tiling 固定报 32 B。 |
| `workspace/stream` | Device 地址/stream | - | 第二段执行；算子 body 不消费 workspace。 |

### 入口与版本

固定 Commit 的最新且唯一公开版本是无后缀 `aclnnForeachAddScalarList`。ACLNN header/inner bridge 由 `add_modules_sources(... ACLNNTYPE aclnn ...)` 生成，固定仓无生成实现；此处把连续化、executor 与实际 runtime launch 保留为框架边界，并继续闭合算子拥有的全部链路。公开 README/API 产品表明确支持 Ascend 950PR/950DT；OpDef `AddConfig("ascend950")`、950 binary config 和 arch35 CMake 接线一致，无发布冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开第一段接口检查空指针、dtype、shape、列表一致性；生成 bridge 的逐语句实现不可见。
2. OpDef 声明 `x/scalars/y` 的 dtype 映射、ND 与 `AutoContiguous()`；输入连续化临时对象的 owner 属于框架边界。
3. `InferShape4ForeachCommon` 要求输出数等于输入 Tensor 数并复制 shape；`InferDataType4ForeachCommon` 复制 dtype。
4. `Tiling4ForeachAddScalarListTiling` 遍历优先级：priority 10000 membase 完成 shape/platform 前置阶段后因 `IsRegbaseSocVersion` 为真返回 `GRAPH_PARAM_INVALID`；priority 30000 `ForeachRegbaseTilingUnaryScalarList2` 命中。
5. regbase Tiling 校验 Tensor 数 1–50、维数不超过 8、各维非负、列表 dtype 一致、scalar dtype/数量组合、输出数/dtype/容量；任一步失败即 `GRAPH_FAILED`，阻断 launch。

### 参数、Workspace 与生命周期

令 `s=sizeof(T)`、`N=sum_i N_i`、`B=4096/s` 元素。平台 producer 提供 AIV 核数 `C_hw` 与 `UB_bytes`；`C=min(C_hw,80,ceil(N/B))`，`N=0` 特判 `C=1`。Host 按 `B` 元素块在 C 核间均分，写每核首尾 Tensor/offset。`U=floor_to_32B((UB_bytes/(4s))*s)/s` 是每个 Queue slot 的元素容量；两个 Queue 均 depth 2，所以总账恰为 `4*U*s<=UB_bytes`。Tiling 设置 key 10001–10004、`blockDim=C`、workspace 32 B并保存 POD。

### 启动图与失败传播

```text
aclnnForeachAddScalarListGetWorkspaceSize
  -> generated ACLNN bridge
  -> InferShape/InferDtype
  -> TilingRegistry: priority10000 reject -> priority30000 regbase
  -> executor{blockDim, key, ForeachSoloTilingDataRegbase, workspace=32}
aclnnForeachAddScalarList(..., stream)
  -> [K1 AIV_VECTOR] arch35::foreach_add_scalar_list(tensor1, scalars, outputs, workspace, tiling)
```

K1 是唯一节点。输入输出是 `ListTensorDesc` 指向的 GM Tensor，scalar 是连续 GM 列表；无中间 workspace 边。entry 在同一 stream 被 executor 启动，返回/失败传播细节由生成 bridge 保证；Host Tiling 返回失败时无有效 K1。

## Route 与参数设计

### 分核与资源计算

`blocks=ceil(N/B)`；基础每核元素 `floor(blocks/C)*B`，前 `blocks%C` 核多一个 `B`。每核 slice 可跨 Tensor，POD 使用含端点索引/offset。Device 对每个 Tensor slice 再切 `ceil(dataCount/U)` 个 tile，tile `currentDataCount=min(U,remain)`。每 tile 内 MicroAPI 每轮处理 `L=GetVRegSize()/sizeof(float)` 个 32-bit lane，`repeatTimes=ceil(currentDataCount/L)`；`UpdateMask<float>(sreg)` 递减保护最后一轮。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 register FP32 | 10001；`ForeachAddScalarListRegbase<half,float,...>` | 950 regbase，`x/out=FP16, scalars=FP32` | 下一 dtype key；未知 key 无 body。 |
| 2 | R-002 FP32 | 10002；`<float,float,...>` | 950，全部 FP32 | 下一 key。 |
| 3 | R-003 INT32 | 10003；`<int32_t,int64_t,...>` | 950，`x/out=INT32, scalars=INT64` | 下一 key。 |
| 4 | R-004 BF16 register FP32 | 10004；`<bfloat16_t,float,...>` | 950，`x/out=BF16, scalars=FP32` | 无匹配 body。 |

```text
priority 10000 membase: IsCapable = !regbase = false -> GRAPH_PARAM_INVALID
priority 30000 regbase: IsCapable = regbase = true
  DT_FLOAT16 -> 10001
  DT_FLOAT   -> 10002
  DT_INT32   -> 10003
  DT_BF16    -> 10004
  other      -> 0 -> router无分支
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 10001 | 950、1–50 Tensor、合法 shape、FP16/FP32/FP16 | 单 K1 AIV | Regbase base + `Compute<half,float>` | depth-2 in/out Queue；float RegTensor；MaskReg | DMA 按有效字节；MicroAPI mask；无 fallback。 |
| R-002 | 10002 | 同约束，全 FP32 | 单 K1 AIV | `Compute<float,float>` | 同 Queue，float 直接 load/store | 同。 |
| R-003 | 10003 | 同约束，INT32/INT64/INT32 | 单 K1 AIV | `Compute<int32_t,int64_t>` | int RegTensor；scalar 源码先转 float | 同。 |
| R-004 | 10004 | 同约束，BF16/FP32/BF16 | 单 K1 AIV | `Compute<bfloat16_t,float>` | float RegTensor，BF16 unpack/pack Cast | 同。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| regbase platform | `IsRegbaseSocVersion(context_)` | 选择 arch35 POD/body | 全部 | membase 仅非 950；两者都不命中则失败 | `ForeachRegbaseTiling::IsCapable` |
| Tensor 数 | `1<=totalTensorCount<=50` | Tensor count POD 仅 50 | 全部 | `GRAPH_FAILED` | `ForeachRegbaseTiling::GetShapeAttrsInfo` |
| 分核上限 | `C<=80` 且每核粗粒度 `4096/s` 元素 | per-core POD 数组 80；避免小于粗粒度的核切片 | 全部 | 使用更少核；空数据 C=1 | `ForeachRegbaseTiling::DoOpTiling` |
| UB tile | `U=floor32((UB/(4s))*s)/s` | 两个 depth-2 Queue | 全部 | tile 缩小；sizePerElem 非正则失败 | `ForeachRegbaseTilingUnaryScalarList2::DoOpTiling` |
| MicroAPI 尾 mask | `repeatTimes=uint16(ceil(dataCount/L))`, `sreg=uint32(dataCount)` | 每轮寄存器宽度固定，最后一轮不得写无效 lane | 全部 | 无另一路；依赖 U 使窄化安全 | `ForeachAddScalarListRegbase::Compute` |

### POD/ABI

固定仓只含 tiling 宏声明，没有展开后的 C struct，因此不编造 `sizeof/alignof/offsetof`；字段顺序和 consumer 如下。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | `uint32_t`/32 | 第 1 字段，offset 不可见 | 元素 | UnaryScalarList2 `set_inputsTensorUbSize(U)` | base `ParseTilingData`→Queue bytes/U | Host 由 UB 推导并 32 B 对齐。 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 第 2 字段 | 元素 | `GetShapeAttrsInfo` | 每 Tensor `cursorEnd` | shape size；空 Tensor 为 0。 |
| 同上 | `tensorStartList[80]` | `uint16_t[80]` | 第 3 字段 | Tensor index | `AssignDataToEachCore` | `[blockIdx]` | 含首端。 |
| 同上 | `tensorEndList[80]` | `uint16_t[80]` | 第 4 字段 | Tensor index | 同上 | `[blockIdx]` | 含尾端。 |
| 同上 | `tensorStartOffsetList[80]` | `int64_t[80]` | 第 5 字段 | 元素 | 同上 | GM base offset | 含首元素。 |
| 同上 | `tensorEndOffsetList[80]` | `int64_t[80]` | 第 6 字段 | 元素 | 同上 | `cursorEnd` | 含尾元素；全空域核 0 为 -1。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10001 | `ForeachRegbaseTilingUnaryScalarList2` | `ForeachSoloTilingDataRegbase` | arch35 entry→`ForeachAddScalarListImpl<half,float>` | `ForeachRegbaseUnaryScalarList` + `Compute` + load/store helpers | `op_kernel/CMakeLists.txt` ascend950 arch35 source |
| R-002 | 10002 | 同上 | 同上 | `<float,float>` | 同上 float specialization | 同上 |
| R-003 | 10003 | 同上 | 同上 | `<int32_t,int64_t>` | int direct RegTensor path | 同上 |
| R-004 | 10004 | 同上 | 同上 | `<bfloat16_t,float>` | BF16 unpack/cast/store helper | 同上 |

## Kernel 实现

### 公共执行流程

entry 建立 `TPipe`、解析 key/POD并在结束时 `Destroy()`。base 用 `ListTensorDesc` 解析输入输出地址，按每核 Tensor 范围循环；每 Tensor slice 按 U 元素切 tile。`CopyIn` 用单 block、`blockLen=dataCount*sizeof(T)` 的 `DataCopyPad` 入 depth-2 inQueue；派生 `Compute` DeQue 输入、Alloc 输出，逐寄存器轮次 load→Adds→store，再 EnQue；`CopyOut` 用相同有效字节数写 GM。Queue 协议承担 MTE/Vector ordering，MicroAPI mask 保护 lane 尾部。

### Route R-001：FP16→FP32 register→FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT16` | 10001。 |
| Router | `arch35/foreach_add_scalar_list.cpp` | key HALF | `ForeachAddScalarListImpl<half,DTYPE_SCALARS>` | `ScalarT=float` binary。 |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_scalar_list -> ForeachAddScalarListImpl<half,float>
  -> ForeachAddScalarListRegbase::Init -> ForeachRegbaseUnaryScalarList::Init/Process
  -> CopyIn -> derived Compute -> LoadOneTensorForDtypeT<half> -> Adds<float>
  -> StoreOneTensorForDtypeT<half> -> CopyOut
```

##### 源码等价伪码

```text
for tensor i in core slice:
  scalarVal=float(scalars[i]); bind x_i/y_i + core offsets
  for tile k with valid<=U:
    DataCopyPad(valid*2 B) x -> inQ
    for r in ceil(valid/L): mask=UpdateMask<float>(remaining)
      unpack half -> float reg; Adds(float reg, scalarVal, mask)
      Cast RINT float->half and packed masked store to outQ
    DataCopyPad(valid*2 B) outQ -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | ListTensorDesc ptr + `cursorStart+kU` | valid half | flattened ND | caller/CopyOut | CopyIn/caller | 调用期间 | exact byte tail。 |
| GM scalar | `scalars+i` | 1 FP32 | scalar list | caller | Compute | 调用期间 | 每 Tensor 读一次。 |
| inQueue | depth-2 slots | `2*U*2` B total | half | CopyIn | Compute | pipe | 双缓冲。 |
| outQueue | depth-2 slots | `2*U*2` B total | half | Compute | CopyOut | pipe | 双缓冲。 |
| RegTensor | register | 固定仓不暴露物理字节 | float | load/cast | Adds/store | 单 MicroAPI 轮次 | mask zeroing/packed store。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["inQueue slot0 · U·2 B"]:1
  in1["inQueue slot1 · U·2 B"]:1
  out0["outQueue slot0 · U·2 B"]:1
  out1["outQueue slot1 · U·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*U*s + 2*U*s = 4U*s <= UB_bytes, s=2
dynamic_ub=N/A; workspace_total=32 B reserved, body consumes 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | CopyIn | Compute DeQue | 两个 slot 周转后 | EnQue/DeQue/FreeTensor | Compute 完成后 Free。 |
| outQueue slot | register store | CopyOut | 周转后 | EnQue/DeQue/FreeTensor | GM write 完成后 Free。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA↔Vector | TQue | CopyIn/Compute | DeQue | tile slot | exact valid bytes。 |
| register | `MaskReg/UpdateMask<float>` | load | Adds/store | last lanes | FP16 unpack→FP32；store `NO_SAT+CAST_RINT`。 |
| Vector↔DMA | TQue | Compute | CopyOut | output slot | only valid bytes copied。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `DoOpTiling/GetTilingKey` | C/U/POD/key。 |
| Kernel entry/router | `arch35/foreach_add_scalar_list.cpp` | HALF branch | half,float instance。 |
| Device body/helper | `foreach_add_scalar_list_regbase.h`、`load_store_utils.h` | `Compute`、half load/store | FP32 add、mask、RINT。 |

### Route R-002：FP32 register Adds

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT` | 10002。 |
| Router | `arch35/foreach_add_scalar_list.cpp` | FLOAT branch | float,float | R-002。 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Impl<float,float> -> base Process/CopyIn -> Compute
  -> direct DataCopy UB float to RegTensor<float> -> Adds -> masked DataCopy to UB -> CopyOut
```

##### 源码等价伪码

```text
每 core/Tensor/tile：scalarVal=scalars[i]；搬 valid 个 float；
逐 L 个 lane 更新 mask，直接 load float reg，Adds，masked store；搬回 valid 个 float。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | list ptr + `cursorStart+kU` | valid FP32 | flattened ND | caller/body | body/caller | 调用期间 | exact byte tail。 |
| in/out Queue | two independent depth-2 queues | each `2U*4` B | FP32 | DMA/Adds | Adds/DMA | pipe | slot round-robin。 |
| RegTensor | one input/one output | physical size不可见 | FP32 | load/Adds | Adds/store | per repeat | mask tail。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["in slot0 · 4U B"]:1
  in1["in slot1 · 4U B"]:1
  out0["out slot0 · 4U B"]:1
  out1["out slot1 · 4U B"]:1
```

##### 字节账与复用顺序

```text
UB_total=4U*4 B; workspace_total=32 B reserved; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Compute | Compute/CopyOut | slot 周转 | FreeTensor | consumer 完成。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue + MaskReg | DMA/load/Adds | Compute/store/DMA | tile/lane | FP32 Adds；last repeat masked。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | key FLOAT | 10002。 |
| Kernel entry/router | `arch35/foreach_add_scalar_list.cpp` | FLOAT branch | float instance。 |
| Device body/helper | `load_store_utils.h` | float else branches | direct register load/store。 |

### Route R-003：INT32 register Adds

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_regbase_tiling.cpp` | scalar combination/GetTilingKey | INT32/INT64 | 10003。 |
| Router | `arch35/foreach_add_scalar_list.cpp` | INT branch | `<int32_t,DTYPE_SCALARS>` | ScalarT 来自 binary config INT64。 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Impl<int32_t,int64_t> -> base CopyIn -> derived Compute
  -> scalarCalcType=float; scalarVal=float(INT64 GM)
  -> DataCopy int32 UB->RegTensor<int32_t> -> Adds(...,scalarVal,mask) -> DataCopy -> CopyOut
```

##### 源码等价伪码

```text
scalarVal = float(scalars_int64[i]);
for each valid tile/repeat:
  mask=UpdateMask<float>(remaining); load int32 reg;
  Adds(int32 outReg, int32 inReg, scalarVal, mask); masked store int32; copy valid bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| scalar GM | `scalars+i` | 1 | INT64 | caller | float conversion | call | 源码未做范围 guard。 |
| in/out Queue | each `2U*4` B | U INT32/slot | INT32 | DMA/Adds | Adds/DMA | pipe | depth 2。 |
| RegTensor | input/output | physical size不可见 | INT32 | load/Adds | Adds/store | repeat | mask typed float but lane width同为 32 bit。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["in slot0 · 4U B"]:1
  in1["in slot1 · 4U B"]:1
  out0["out slot0 · 4U B"]:1
  out1["out slot1 · 4U B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16U B; workspace_total=32 B reserved; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Compute | Compute/CopyOut | round-robin | FreeTensor | consumer 完成。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar | C++ conversion | INT64 GM | Adds | scalar | 固定源码明确先转 FP32，再传给 INT32 RegTensor Adds；最终转换语义库内不可见。 |
| repeat | MaskReg | load | Adds/store | last lanes | valid lane only。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `SCALAR_LIST_SUPPORT_DTYPE_COMB/GetTilingKey` | INT32/INT64 合法且 key10003。 |
| Kernel entry/router | `arch35/foreach_add_scalar_list.cpp` | INT branch | int32/int64 instance。 |
| Device body/helper | `foreach_add_scalar_list_regbase.h` | `scalarCalcType/Compute` | INT64→float 和 int RegTensor Adds。 |

### Route R-004：BF16→FP32 register→BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_regbase_tiling.cpp` | `GetTilingKey` | DT_BF16 + scalar FP32 | 10004。 |
| Router | `arch35/foreach_add_scalar_list.cpp` | BF16 branch | `<bfloat16_t,float>` | R-004。 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Impl<bfloat16_t,float> -> CopyIn -> Compute
  -> DIST_UNPACK_B16 load + Cast BF16->FP32 -> Adds<float>
  -> Cast FP32->BF16(CAST_RINT) + DIST_PACK_B32 store -> CopyOut
```

##### 源码等价伪码

```text
对 valid BF16 tile：逐 L 个 32-bit计算 lane生成 mask；
unpack BF16并 zeroing cast 到 float reg；FP32 Adds(scalars[i])；
NO_SAT+RINT cast BF16并 masked packed store；只 DMA valid*2 B。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| in/out Queue | each `2U*2` B | U BF16/slot | BF16 | DMA/register | register/DMA | pipe | depth 2。 |
| RegTensor | BF16 temp + float calc | physical size不可见 | BF16/FP32 | unpack/Cast/Adds | Adds/Cast/pack | repeat | 无 UB FP32 scratch。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["in slot0 · 2U B"]:1
  in1["in slot1 · 2U B"]:1
  out0["out slot0 · 2U B"]:1
  out1["out slot1 · 2U B"]:1
```

##### 字节账与复用顺序

```text
UB_total=8U B; FP32 intermediates live only in registers
workspace_total=32 B reserved; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| RegTensor | unpack/Cast | Adds/Cast/store | next repeat | lexical MicroAPI order | store 完成。 |
| Queue slots | DMA/register | register/DMA | round-robin | Queue FreeTensor | consumer 完成。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| register | MicroAPI lexical dependency + mask | unpack/Cast | Adds/Cast/pack | registers/lanes | FP32 Add；BF16 store RINT/no-sat；tail masked。 |
| DMA | TQue | CopyIn/Compute | Compute/CopyOut | UB slots | exact valid bytes。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | BF16 key/UB formula | key10004、U。 |
| Kernel entry/router | `arch35/foreach_add_scalar_list.cpp` | BF16 branch | bf16,float instance。 |
| Device body/helper | `load_store_utils.h` | BF16 load/store helpers | unpack、cast traits、RINT/pack。 |

## 算子特定优化

### Trick T-001：FP16/BF16 只在寄存器升精度，不物化 FP32 UB

#### 触发与源码依据

- 适用 Route：R-001、R-004；R-002 复用同一 float RegTensor 管线。
- 触发条件：`T` 为 half/BF16/float 的编译期分支。
- Golden 来源：[`foreach/foreach_add_scalar_list/op_kernel/arch35/foreach_add_scalar_list_regbase.h:52-67`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar_list/op_kernel/arch35/foreach_add_scalar_list_regbase.h#L52-L67)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
uint32_t sreg = (uint32_t)dataCount;

__VEC_SCOPE__
{
    MaskReg maskReg;
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, float>::value) {
        RegTensor<float> inRegToFloat;
        RegTensor<float> outReg;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            maskReg = UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
            Adds(outReg, inRegToFloat, scalarVal, maskReg);
            ops::StoreOneTensorForDtypeT<T>(outUbAddr, outReg, maskReg, i * dataCountPerLoop);
        }
```

#### 机制与设计原因

FP16/BF16 由 load helper 在 RegTensor 内展开并 cast 为 FP32，Adds 后立刻 RINT pack 回原 dtype；UB 始终只有原 dtype 的双缓冲 input/output，没有 FP32 scratch。`dataCountPerLoop` 固定按 FP32 register lane 数，`UpdateMask<float>` 同时解决最后一轮 tail。这同时减少 UB 中间物化并明确升/降精度位置。

#### 迁移边界

- 适用：无跨 repeat 状态、结果可立即降精度的 elementwise。
- 前提：保留 `Load/StoreOneTensorForDtypeT` 的 cast trait、mask merge、pack distribution 与 RINT。
- 失效：需要 FP32 中间跨 tile 保存或跨元素归约时必须增加真实 scratch/同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：列表全域 4 KiB 粒度切核、每核跨 Tensor 执行

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：任意非空 TensorList；空元素总量走单核空域特判。
- Golden 来源：[`foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp:253-271`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp#L253-L271)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
OP_CHECK_IF(
    sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
int64_t elementsPerBlock = SINGLE_CORE_PROCESS_DATA / sizePerElem;

numBlocks_ = std::min<uint64_t>(aicoreParams_.numBlocks, MAX_CORE_CONT_950);
uint32_t tempCoreNum = Ops::Base::CeilDiv(totalDataCount_, elementsPerBlock);
if (tempCoreNum < numBlocks_) {
    numBlocks_ = tempCoreNum;
}
if (totalDataCount_ == 0) {
    numBlocks_ = 1;
    tensorStartList_[0] = 0;
    tensorEndList_[0] = 0;
    tensorStartOffsetList_[0] = 0;
    tensorEndOffsetList_[0] = -1;
} else {
    AssignDataToEachCore(numBlocks_, elementsPerBlock);
}
```

#### 机制与设计原因

Host 把所有 Tensor flatten 为一个总元素域，用 4096 字节等价元素块决定核数，再编码回 Tensor 索引/offset。这样单 launch 能跨 Tensor 分发，且极小总数据不会盲目占满 80 个 POD core slot；源码可证明的是调度粒度和 launch 合并，不宣称实测收益。

#### 迁移边界

- 适用：每 Tensor body 相同、scalar 可按 Tensor index 获取的列表算子。
- 前提：TensorList 最多 50，POD core slot 最多 80，Device 必须解释含端点 offset。
- 失效：不同 Tensor 需要不同 layout/body 或跨 Tensor 依赖时需拆 route/launch。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | 10001 | 10002 | 10003 | 10004 | regbase priority 30000 + dtype。 |
| 模板与 dtype | half,float | float,float | int32,int64 | bf16,float | binary config/entry。 |
| 循环与分核 | 4 KiB host slice；U tile；FP32 lanes | 同 | 同 | 同 | 公共 base。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2×depth2 Queue + float regs | 同 | int regs | float regs + BF16 temp regs | 无 TBuf/dynamic UB/通信。 |
| 内存总占用与复用 | 8U B | 16U B | 16U B | 8U B | 恒为 `4Us`。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+unpack/cast/Adds/pack | direct load/Adds/store | int load/Adds/store | BF16 unpack/cast/Adds/pack | 全部 AIV。 |
| 同步、精度与 tail | FP32 add，RINT half | FP32 | INT64先转float；INT Adds consumer语义库内不可见 | FP32 add，RINT BF16 | mask + Queue。 |
| fallback/失败边界 | unknown key无body | 同 | 同 | 同 | Host 必须闭合组合。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 公开支持。 |
| Host/Launcher | membase priority10000，50核上限 | 同 | regbase priority30000，80核上限、1–50 Tensor显式校验。 |
| Kernel/资源 | 旧 Vector body；BF16有 UB FP32 scratch | 同 | arch35 MicroAPI；FP16/BF16 FP32 中间仅寄存器；depth2 Queue。 |

## 限制与失败边界

- 950 Tiling 要求 TensorList 项数 1–50；单个 Tensor 可空，所有 Tensor 总元素为 0 时发一个空域核，不进行 DMA/计算。
- 输出大于输入时只写输入元素域；额外区域固定仓未定义。输出非连续由公开 ACLNN 明确不支持。
- R-003 固定源码把 INT64 scalar 先转换为 `float`，再传给 INT32 `RegTensor` 的 `Adds`；该 primitive 的最终 scalar 转换规则不在固定仓，不能推测。大整数精度未执行真机验证。
- `repeatTimes` 窄化为 `uint16_t`、`dataCount` 窄化为 `uint32_t`；固定 Host 未写独立 guard，但每 tile 受平台 UB 推导的 U 约束。不得绕过 Host 直接传超界 POD。
- ACLNN inner bridge、连续化临时对象和 executor launch 代码固定仓不可见；workspace 32 B 被 Host 报告但 Device base/派生 body未读取。
- unknown key 无 Device fallback 或显式错误写回；Host selector/binary config 是必要前提。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_add_scalar_list/docs/aclnnForeachAddScalarList.md` | 两段式 ACLNN 原型/产品表 | 公开入口与 950 支持。 |
| 入口实现 | `foreach/foreach_add_scalar_list/op_host/foreach_add_scalar_list_def.cpp` | `ForeachAddScalarList` | dtype/格式/AutoContiguous/950 config。 |
| Infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` | output shape/dtype。 |
| Host selector/注册 | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachAddScalarListTiling` | TilingRegistry 入口。 |
| Host selector predicate | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `MEMBASE_TILING` | priority10000 在 950 拒绝。 |
| Host/Tiling/POD | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `ForeachRegbaseTilingUnaryScalarList2` | 校验、切核、U、key、POD/workspace。 |
| Kernel router | `foreach/foreach_add_scalar_list/op_kernel/arch35/foreach_add_scalar_list.cpp` | `foreach_add_scalar_list` | 10001–10004→实例。 |
| Kernel operator body | `foreach/foreach_add_scalar_list/op_kernel/arch35/foreach_add_scalar_list_regbase.h` | `ForeachAddScalarListRegbase::Compute` | scalar、MicroAPI loop、mask/Adds。 |
| Kernel base/memory | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary_scalar_list.h` | `Init/Process/CopyIn/CopyOut` | ListTensorDesc、Queue、DMA、tile。 |
| Kernel dtype helper | `common/inc/op_kernel/load_store_utils.h` | `Load/StoreOneTensorForDtypeT` | unpack/cast/round/pack。 |
| Kernel vector width | `common/inc/op_kernel/platform.h` | `GetVRegSize` | register byte width producer。 |
| 构建接线 | `foreach/foreach_add_scalar_list/op_kernel/CMakeLists.txt` | ascend950 `KERNEL_SRC arch35/...` | 950 source选择。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | U 元素数与 per-core 含端点区间 | uint32 U、50 Tensor、80 core、字段顺序 | `ForeachSoloTilingDataRegbase` |
| 调用顺序 | ListDesc bind→tile CopyIn→register Compute→CopyOut | scalar index必须等于 Tensor index | `ForeachRegbaseUnaryScalarList::Process` |
| workspace/内存 | 32 B预留、0 B算子消费；`4Us` UB | Queue depth 2 必须计入 | `DoOpTiling/InitBuffer` |
| 同步 | Queue producer/consumer；repeat内数据依赖 | FreeTensor 前消费完成 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP16/BF16在寄存器升FP32并RINT；mask tail | cast trait、pack distribution、valid-byte DMA | `Compute/LoadOneTensorForDtypeT/StoreOneTensorForDtypeT` |
| 失败边界 | 1–50 Tensor、dtype组合、output校验 | unknown key不是fallback | `GetShapeAttrsInfo/CheckScalarList/CheckOutput` |