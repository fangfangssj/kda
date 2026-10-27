---
type: Operator
title: "ForeachNorm 950 ACLNN 算子知识"
description: "说明 aclnnForeachNorm 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/docs/aclnnForeachNorm.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_host/foreach_norm_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/arch35/foreach_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/arch35/foreach_norm_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:58:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachNorm 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachNormGetWorkspaceSize` / `aclnnForeachNorm`，公开 ACLNN 两段式接口。 |
| 计算语义 | 每个 Tensor 独立归约；FLOAT scalar 正无穷→max(abs(x))，`int(scalar)==1`→sum(abs(x))，其余→sqrt(sum(x*x))。 |
| 输入输出 | x/out: 最多256项，ND，FP16/FP32/BF16，同dtype，out每项1元素；scalar实际binary为FLOAT32或INT64。 |
| 执行类型 | 单 Kernel `foreach_norm`，AIV_VECTOR RegBase/MicroAPI；无可达 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 +Inf；R-002 L1；R-003 L2。dtype key 分别为10001/10002/10004。 |
| 核心资源 | 双缓冲 VECIN、单缓冲 VECOUT、RegTensor/MaskReg、最多336个FP32 userWS partial槽、`SyncAll`。 |
| 来源闭包 | source_closed：公开 API → OpDef/自动 ACLNN 边界 → `ForeachReduceRegbaseTiling` → RegBase POD → arch35 entry/body → 950构建接线。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

```text
FLOAT32 scalar 且 exponent==0xFF 且 scalarVal>0: y=max(abs(x))
否则 int(float(scalar))==1:                    y=sum(abs(x))
否则:                                          y=sqrt(sum(x*x))
空 Tensor:                                     y=0
```

每个输入 Tensor 全元素归约成一个同 dtype 输出。MicroAPI 始终把 FP16/BF16 load 后转为 FP32，FP32也在寄存器中计算；跨 tile、跨核 partial 均为 FP32，最后才按 `StoreOneTensorForDtypeT` 的 RINT/zeroing cast 降精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 1–256项 TensorList；每项ND，维数≤8 | FLOAT16/FLOAT32/BFLOAT16，列表内一致 | 维度值须非负；支持空 Tensor；OpDef AutoContiguous。 |
| `scalar` | shape size=1，维数≤8 | 固定 binary/OpDef: FLOAT32或INT64；ACLNN文档另列DOUBLE | INT64只识别转float后整数1；FLOAT32另识别正无穷。 |
| `out` | 与x等长；每项shape size=1、维数≤8、ND | 与x一致 | 输出不支持非连续；空输入Tensor写0。 |

### 入口与版本

公开产品表明确支持 Ascend 950PR/950DT。无更高版本后缀；950 通过相同 ACLNN 名称进入 `ascend950` OpDef 与 arch35 Kernel。自动生成 ACLNN inner bridge 不在固定仓，故其 executor/Contiguous 包装以框架边界标注；实际 Host/Tiling/POD/Device/构建均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口检查 null、dtype、列表一致性、out shape。RegBase Tiling 的 `GetShapeAttrsInfo` 要求 x列表长度在[1,256]，逐项检查dtype一致、维数≤8、所有轴非负，并检查 scalar descriptor、dtype为FLOAT/INT64、shape size=1；`CheckOutput` 校验输出数量、dtype、维数及shape size。OpDef 的输入/输出 `AutoContiguous()` 由生成桥接消费，具体ViewCopy不可见。

### 参数、Workspace 与生命周期

注册表按数值升序先试 priority10000 membase；950的regbase arch使其返回 `GRAPH_PARAM_INVALID`，继而 priority30000 `ForeachReduceRegbaseTiling` 命中。Host写 `blockDim=numBlocks`、scheduleMode=1、key 10001/10002/10004。workspace=`(80+256)*4 + 16MiB = 1,344B + 16MiB`；前336个FP32槽存每核每Tensor片段结果。executor/workspace/stream生命周期由ACLNN两段式契约管理。

### 启动图与失败传播

```text
ACLNN executor
  -> Tiling4ForeachNormTiling
     priority10000 membase: IsCapable=false
     priority30000 ForeachReduceRegbaseTiling: IsCapable=true
  -> foreach_norm<<<numBlocks, scheduleMode=1>>>
       Stage1 MicroAPI per-core segment reduction -> userWS
       SyncAll()
       Stage2 per-tensor partial reduction -> output
```

唯一跨阶段边是 userWS FP32 partial 数组：Stage1的 `DataCopyPad` 批量写每核覆盖的 Tensor partial，Stage2在 `SyncAll` 后按 `tensorMiddleStartList/count` 读取。Host校验/POD/workspace指针失败阻断launch；Device null workspace/sys workspace或未知key直接结束且无输出。

## Route 与参数设计

### 分核与资源计算

Host以总元素数 `N` 计算 `elementsPerBlock=4096/sizeof(T)`，`numBlocks=min(AIVCoreNum,80,ceil(N/elementsPerBlock))`，零数据强制1核。非零时按连续元素近似均分，记录每核Tensor闭区间及offset；`maxTensorNumPerCore` 以 `B=GetUbBlockSize` 对齐成 FP32 元素数 `M=ceil(maxTensorNumPerCore*4/B)*B/4`。输入tile容量（元素）为 `I=floor((UB-4M)/2/B)*B/sizeof(T)`。因此真实UB占用为双输入Queue `2*I*sizeof(T)` 加输出Queue `4M`，不超过Host的UB值。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 +Inf | key10001/10002/10004；model3 | 950；scalar binary为FLOAT32；指数域全1且`scalarVal>0` | R-002/R-003 |
| 2 | R-002 L1 | 同keys；model1 | 未命中+Inf，且`static_cast<int>(scalarVal)==1` | R-003 |
| 3 | R-003 L2 | 同keys；model2 | 其余所有scalar | 无fallback |

```text
dtype FP16 -> 10001; FP32 -> 10002; BF16 -> 10004
if DTYPE_SCALAR is float and positive exponent-all-ones: R-001
else if int(scalarVal)==1: R-002
else: R-003
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 +Inf | 10001/2/4 + model3 | 950合法输入，FLOAT scalar正无穷 | 单AIV，两阶段 | `ForeachNormNDRegbase<T,3,...>` | 双VECIN+单VECOUT+FP32 RegTensor/userWS+SyncAll | UpdateMask清尾；空Tensor→0 |
| R-002 L1 | keys + model1 | 未命中inf且int(s)=1 | 同上 | `<T,1,...>` | 同上 | Abs+sum；tail mask |
| R-003 L2 | keys + model2 | 其余 | 同上 | `<T,2,...>` | 同上 | square+sum+sqrt；无一般p fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| RegBase平台 | `IsRegbaseSocVersion(context_)` | 使用MicroAPI/RegTensor arch35 body | 全部 | membase跳过后regbase；非regbase相反 | 两个 `IsCapable` |
| 列表长度 | `1<=count<=256` | POD数组和ListTensorDesc上界 | 全部 | GRAPH_FAILED | `GetShapeAttrsInfo` |
| 核上界 | `min(aivCore,80,ceil(N/(4096/s)))` | POD core数组长度80，减少空核 | 全部 | 至少1核 | `DoOpTiling` |
| UB | `2*I*s+4M<=UB`，I按B向下对齐 | 双输入Queue和per-core输出partial Queue | 全部 | tile变小；无显式I=0 guard | `DoOpTiling`, `Init` |
| scalar +Inf | 仅FLOAT binary且指数全1、值>0 | INT64编译分支不检查inf；NaN比较>0为false | R-001 | R-002/R-003 | arch35 `foreach_norm` |
| tail mask | `repeatTimes=ceil(dataCount/(VL_SIZE/4))` + `UpdateMask(sreg)` | 防止脏lane进入reduce | 全部 | 最后repeat masked | `DoCompute`, `ReduceComputeImpl` |

### POD/ABI

底层 tiling-data 宏来自固定仓外头文件，故字段顺序/类型可证，`sizeof/alignof/offsetof` 固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachReduceTilingDataRegbase` | `inputsTensorUbSize` | uint32_t/32 | 第1字段 | 元素 | UB公式I | Init Queue/最大tile | B字节向下对齐后除s |
| 同上 | `needCoreNum` | uint32_t/32 | 第2字段 | core | numBlocks | Stage2步长 | 1–80 |
| 同上 | `totalTensorCount` | uint32_t/32 | 第3字段 | Tensor | input instance count | Stage2上界 | 1–256 |
| 同上 | `maxTensorNumPerCore` | uint16_t/16 | 第4字段 | FP32元素 | 对齐后M | outQueue容量 | 4M按B对齐 |
| 同上 | `tensorDataCountList[256]` | uint64_t[] | 第5字段 | 元素/Tensor | storage numel | Stage1/空域 | 可为0 |
| 同上 | `tensorStartList[80]` | uint16_t[] | 第6字段 | Tensor index/core | 分核 | Parse | active项 |
| 同上 | `tensorEndList[80]` | uint16_t[] | 第7字段 | Tensor index/core | 分核 | Parse | 闭区间 |
| 同上 | `tensorStartOffsetList[80]` | uint64_t[] | 第8字段 | 元素 | 分核 | Stage1 cursor | active项 |
| 同上 | `tensorEndOffsetList[80]` | uint64_t[] | 第9字段 | 元素 | 分核/全空特例-1写入无符号 | Stage1 cursor转int64 | 空域产生-1位型 |
| 同上 | `tensorMiddleCountList[256]` | uint16_t[] | 第10字段 | partial/Tensor | 片段计数 | Stage2长度 | 每Tensor计数加1 |
| 同上 | `tensorMiddleStartList[256]` | uint16_t[] | 第11字段 | FP32 slot | 计数前缀和 | Stage2 offset | ≤336槽 |
| 同上 | `coreMiddleOffsetList[80]` | uint16_t[] | 第12字段 | FP32 slot | 每核Tensor数前缀和 | Stage1写offset | active项 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | keys+model3 | `ForeachReduceRegbaseTiling` | RegBase POD | arch35 `foreach_norm`/`ForeachNormImpl` | `ForeachNormNDRegbase<T,3>`, load/store utils | 950 arch35 CMake |
| R-002 | keys+model1 | 同上 | 同上 | 同上 | `<T,1>` | 同上 |
| R-003 | keys+model2 | 同上 | 同上 | 同上 | `<T,2>` | 同上 |

## Kernel 实现

### 公共执行流程

`foreach_norm` 取得user workspace，读取scalar，选择modelCode，再按key选择T。`Init` 用ListTensorDesc绑定TensorList，分配双输入Queue和单输出Queue。Stage1每核对每个负责Tensor片段按I元素切tile，MicroAPI load/cast，寄存器内abs或square与归约，将每Tensor片段一个FP32放入outLocal；一次DMA把本核全部partial连续写入userWS。`SyncAll` 后，Stage2按Tensor搬入所有partial，寄存器归约，L2开方，按T存入outQueue再写GM。

### Route R-001：正无穷范数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_reduce_regbase_tiling.cpp` | `GetTilingKey` | dtype→10001/2/4 | dtype实例 |
| Router | `arch35/foreach_norm.cpp` | `foreach_norm` | FLOAT scalar指数全1且>0 | model3 |
| Body | `arch35/foreach_norm_regbase.h` | `DoCompute<...,3>` | modelOrd==POSITIVE_INF | Abs+Max |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormImpl<T> -> ForeachNormNDRegbase<T,3>::Init/Process
  -> SingleTensorProcess -> CopyIn -> DoCompute<first?,3>: Load/cast, Abs, Max, ReduceMax
  -> userWS -> SyncAll -> ReduceComputeImpl<...,3>: Max/ReduceMax -> Store T -> CopyOut
```

##### 源码等价伪码

```text
for assigned tensor segment:
  for tile: DMA GM->UB; for vector repeat with UpdateMask: load T->FP32, abs, lane-wise max
            ReduceMax vector; first tile store partial, later tile max into same outLocal slot
DMA all per-core partials -> userWS; SyncAll
for assigned output tensors: DMA partials; max across repeats; ReduceMax; cast/store T; DMA 1 element
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | TensorList ptr+cursor+tile*I | currentDataCount | T/ND | caller | CopyIn | caller | DataCopyPad no explicit pad |
| inQueue | UB | depth2×I*s B | T；Stage2按float | MTE2 | MicroAPI | TPipe | tile/Stage2复用 |
| outQueue Stage1 | perTensor offset | depth1×4M B | FP32 | DoCompute | userWS DMA | TPipe | first/later tile原地max |
| userWS | coreMiddleOffset/tensorMiddleStart | ≤336×4 B | FP32 | Stage1 | Stage2 | executor | SyncAll保护 |
| RegTensor | VReg | VL_SIZE bytes/寄存器，精确寄存器总占用不可见 | FP32 | load/abs | max/reduce | vector scope | mask清尾 |
| output GM | TensorList item0 | sizeof(T) | T | Store/CopyOut | caller | caller | 1元素 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 2I·s B"]:3
  outQ["outQueue · depth=1 · 4M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*I*sizeof(T) + 4M <= UB
dynamic_ub = N/A（AIV TPipe）
workspace_total = 1344 B user + 16 MiB system
VReg总字节：固定仓编译器分配不可见；生命周期限于__VEC_SCOPE__
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue槽 | CopyIn | DoCompute | FreeTensor后下一tile | Queue | tile≤I |
| outLocal[tensor] | first tile partial | later tile max / userWS DMA | 下个Process生命周期 | isFirst模板分支；EnQue/DeQue | offset<M |
| userWS partial | Stage1 | Stage2 | Kernel结束 | SyncAll | 唯一partial slot |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue | MTE2 | MicroAPI | inQueue | currentDataCount字节 |
| Vector | UpdateMask+Select | load | Max/ReduceMax | VReg | masked lane zero；abs非负故0中性 |
| 跨核 | SyncAll | userWS写 | Stage2读 | partial | FP32 |
| store | load_store_utils | FP32 result | CopyOut | outQueue | FP16/BF16 RINT/zeroing |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_regbase_tiling.cpp` | `DoOpTiling`, `GetTilingKey` | I/M、分核、key |
| Kernel entry/router | `arch35/foreach_norm.cpp` | +Inf bit test, `ForeachNormImpl` | model3 |
| Device body/helper | `arch35/foreach_norm_regbase.h` | `DoCompute<...,3>`, `ReduceComputeImpl<...,3>` | 两级abs-max、mask、tail |

### Route R-002：L1范数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `arch35/foreach_norm.cpp` | `foreach_norm` | 未命中inf且`int(scalarVal)==1` | model1 |
| Body | `arch35/foreach_norm_regbase.h` | `DoCompute<...,1>` | ONE model | Abs+Add+ReduceSum |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormNDRegbase<T,1>::Process
  -> CopyIn -> DoCompute: Load/cast -> Abs -> Add/ReduceSum -> userWS
  -> SyncAll -> ReduceComputeImpl: Add/ReduceSum -> Store -> CopyOut
```

##### 源码等价伪码

```text
沿R-001相同地址/循环；将max组合替换为sum组合；第一tile直接存FP32 partial，后续tile广播旧值并Add；二级求和后按T输出。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQueue | tile*I | depth2×I*s B | T | MTE2 | Abs/Sum | TPipe | double buffer |
| outQueue | localTensor offset / final0 | 4M B | Stage1 FP32；final T | DoCompute/ReduceCompute | userWS/GM | TPipe | 阶段间复用 |
| userWS | partial前缀 | ≤1344 B | FP32 | Stage1 | Stage2 | executor | SyncAll |
| VReg | repeat offset | VL_SIZE | FP32 | load/cast | Add/ReduceSum | scope | UpdateMask |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 2I·s B"]:3
  outQ["outQueue · depth=1 · 4M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2I*s + 4M
dynamic_ub = N/A
workspace_total = 1344 B + 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| outLocal slot | first tile sum | later tile broadcast/add | 下一Tensor/阶段 | isFirst + Queue | offset<M |
| userWS | per-core DMA | Stage2 | Kernel结束 | SyncAll | count/start一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Stage1 | UpdateMask, Abs, Add, ReduceSum | tile | outLocal | VReg/UB | T升FP32；tail masked |
| Stage1→2 | SyncAll | DMA partial | partial load | userWS | FP32 |
| output | StoreOneTensorForDtypeT | FP32 sum | GM | outQueue | RINT for 16-bit |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_regbase_tiling.cpp` | RegBase tiling | 与R-001相同资源 |
| Kernel entry/router | `arch35/foreach_norm.cpp` | int(scalar)==1 | model1 |
| Device body/helper | `arch35/foreach_norm_regbase.h` | `DoCompute<...,1>` | 两级abs-sum |

### Route R-003：L2范数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `arch35/foreach_norm.cpp` | `ForeachNormImpl` else | 非+Inf且int(s)!=1 | model2 |
| Body | `arch35/foreach_norm_regbase.h` | `DoCompute<...,2>`, `ReduceComputeImpl<...,2>` | TWO model | square+sum+sqrt |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormNDRegbase<T,2>::Process
  -> CopyIn -> DoCompute: Load/cast -> Mul -> Add/ReduceSum -> userWS
  -> SyncAll -> ReduceComputeImpl: Add/ReduceSum -> Sqrt -> Store -> CopyOut
```

##### 源码等价伪码

```text
逐tile将T加载为FP32，mask内平方并累加；每核每Tensor产生一个FP32 partial；
SyncAll后对该Tensor所有partial求和，只在二级归约后Sqrt，随后cast到T并写1元素。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/inQueue | cursor+tile*I | currentDataCount；2I*s B | T | GM/MTE2 | Mul/Sum | caller/TPipe | mask tail |
| outQueue | localTensor slot/final0 | 4M B | FP32 partial/T final | Stage1/Stage2 | userWS/GM | TPipe | stage reuse |
| userWS | core/tensor prefix | ≤336 floats | FP32 | Stage1 | Stage2 | executor | SyncAll |
| VReg | repeat offset | compiler-managed | FP32 | LoadOneTensor | Mul/Add/Reduce/Sqrt | scope | VL1 final store |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 2I·s B"]:3
  outQ["outQueue · depth=1 · 4M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2I*s + 4M <= UB
dynamic_ub = N/A
workspace_total = 1344 B + 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| outLocal partial | tile square-sum | latertile add/userWS DMA | 下一Tensor | MicroAPI store + Queue | FP32 |
| userWS | Stage1 | Stage2 | Kernel结束 | SyncAll | start/count不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile | UpdateMask + Mul/Add | load | ReduceSum | VReg | padding lane不参与 |
| cross-core | SyncAll | partial DMA | Stage2 load | userWS | FP32可见 |
| final | ReduceSum→Sqrt→Store | partials | output GM | VReg/outQueue | 只最终开方；16-bit RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_regbase_tiling.cpp` | `DoOpTiling` | 分核、I/M、workspace |
| Kernel entry/router | `arch35/foreach_norm.cpp` | default model2 | scalar fallback |
| Device body/helper | `arch35/foreach_norm_regbase.h` | model2 methods | FP32 square-sum-sqrt与tail mask |

## 算子特定优化

### Trick T-001：RegBase 掩码归约与跨 tile 原地 partial

#### 触发与源码依据

- 适用 Route：R-001–R-003。
- 触发条件：950 RegBase body；每个tile按`VL_SIZE/sizeof(float)`循环，最后repeat可能不足满VL。
- Golden 来源：[`foreach/foreach_norm/op_kernel/arch35/foreach_norm_regbase.h:207-248`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/arch35/foreach_norm_regbase.h#L207-L248)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
maskReg = MicroAPI::UpdateMask<float>(sreg);
ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
if constexpr (modelOrd == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
    Abs(inRegToFloat, inRegToFloat, maskReg);
    Max(reduceFloat, reduceFloat, inRegToFloat, pregMain);
} else if constexpr (modelOrd == ONE_SCALAR_NORM_MODEL_CODE) {
    Abs(inRegToFloat, inRegToFloat, maskReg);
    Add(reduceFloat, reduceFloat, inRegToFloat, pregMain);
} else {
    Mul(inRegToFloat, inRegToFloat, inRegToFloat, maskReg);
    Add(reduceFloat, reduceFloat, inRegToFloat, pregMain);
}
```

#### 机制与设计原因

同一编译期模板用 `modelOrd` 消除运行时内层分支，`UpdateMask`只让有效元素进入abs/square；每个repeat先在FP32 RegTensor纵向组合，再归约成一个partial。首tile用 `DIST_FIRST_ELEMENT_B32` 写outLocal槽，后续tile用广播load旧partial并原地max/add，避免为每tile物化UB数组。

#### 迁移边界

- 适用：可结合的max/sum类全量归约，输出每Tensor一个标量。
- 前提：T可经load_store_utils升FP32，尾块必须以UpdateMask控制，outLocal槽容量M覆盖本核Tensor数。
- 失效：非结合归约、需要保留每tile结果或不同中性元时必须重设寄存器初始化与合并逻辑。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码通过mask与编译期模型分支给出必要边界，不能删除尾mask或把+Inf路线的Max替换为Add。

## 模板与实现对比

| 对比维度 | R-001 +Inf | R-002 L1 | R-003 L2 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | FLOAT正无穷最先 | 其次int(s)=1 | 其余 | entry判断顺序 |
| 模板与 dtype | model3，T按key | model1 | model2 | 编译期消除内层模型分支 |
| 循环与分核 | 4096B/core粒度，两阶段 | 同左 | 同左 | Host统一分核 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2I*s+4M，Max RegTensor | 同占用，Add RegTensor | 同占用，Mul/Add/Sqrt | 无TBuf/dynamicUB/通信 |
| 内存总占用与复用 | outLocal原地max | 原地add | 原地add | 避免tile partial物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Pad, MicroAPI Abs/Max | Abs/Add | Mul/Add/Sqrt | 仅AIV |
| 同步、精度与 tail | SyncAll，FP32，mask | 同左 | 同左 | 最终才cast |
| fallback/失败边界 | 非FLOAT不可达 | 非1到R-003 | 终点 | 无一般p实现 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 同左 | 公开支持 |
| Host/Launcher | membase priority10000 | membase priority10000 | regbase priority30000，最多80核POD |
| Kernel/资源 | 传统Vector API、1KiB TBuf/cast Queue | 同A2 | MicroAPI RegTensor、无calc TBuf，+Inf route |

## 限制与失败边界

- README同一文件既给950 `scalar=+Inf`公式，又写“scalar仅支持1或者2”；固定Kernel确实实现FLOAT正无穷路线。该公开约束冲突在此保留，不扩展成其它无穷/一般p承诺。
- 除正无穷和`int(scalar)==1`外均走L2；NaN、负无穷、其它p没有独立数学实现。
- ACLNN文档声明DOUBLE scalar，但950 binary config仅FLOAT32/INT64；生成桥接是否转换DOUBLE固定仓不可见。
- `inputsTensorUbSize`无显式非零guard；若平台UB不足以容纳`4M`后再双分，Device的tile除法边界未在固定仓防护。
- 空Tensor Stage1可能在per-core outLocal留下未使用槽，但Stage2通过`tensorDataCountList[i]==0`直接写0，不读取该partial。
- null workspace/unknown key在Device静默返回；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_norm/README.md` | 950公式与参数约束 | +Inf与1/2公开说明冲突 |
| 入口声明/约束 | `foreach/foreach_norm/docs/aclnnForeachNorm.md` | API原型/产品表 | 公开950支持与契约 |
| 入口实现 | `foreach/foreach_norm/op_host/CMakeLists.txt` | `ACLNNTYPE aclnn` | 自动生成 ACLNN inner bridge 的活动构建边界 |
| 入口实现 | `foreach/foreach_norm/op_host/foreach_norm_def.cpp` | ascend950 config | dtype/layout/platform与生成ACLNN边界 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority升序尝试与fallthrough |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_util.h` | `IsRegbaseSocVersion` | 950 RegBase平台判定 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_reduce_tiling_def.h` | RegBase POD | ABI字段/key注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_reduce_regbase_tiling.cpp` | `ForeachReduceRegbaseTiling` | 校验、分核、UB、workspace、key |
| Kernel entry/body | `foreach/foreach_norm/op_kernel/arch35/foreach_norm.cpp` | `foreach_norm`, `ForeachNormImpl` | scalar/model/key router |
| Kernel entry/body | `foreach/foreach_norm/op_kernel/arch35/foreach_norm_regbase.h` | `ForeachNormNDRegbase` | MicroAPI body、内存、同步、tail |
| Kernel entry/body | `common/inc/op_kernel/load_store_utils.h` | `LoadOneTensorForDtypeT`, `StoreOneTensorForDtypeT` | FP16/BF16升降精度与round/mask |
| 构建接线 | `foreach/foreach_norm/op_kernel/CMakeLists.txt` | ascend950 arch35 source | 950活动构建 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 256 Tensor/80 core两阶段归约POD | 字段顺序、位宽、元素/slot单位、M/I含义 | `ForeachReduceTilingDataRegbase` |
| 调用顺序 | Stage1批量partial→SyncAll→Stage2 | 不得提前读userWS | `Process` |
| workspace/内存 | 336个FP32槽；UB=2I*s+4M | core/tensor前缀和及Queue depth | `DoOpTiling`, `Init` |
| 同步 | Queue与`SyncAll` | Stage1 DMA必须全体完成 | `Process` |
| 精度与尾块 | load升FP32、UpdateMask、最终RINT store | +Inf/L1/L2操作次序与中性元 | `DoCompute`, `ReduceComputeImpl`, load/store utils |
| 失败边界 | x列表1–256、维数≤8、scalar shape1 | 其它scalar默认L2；unknown key无body | Tiling checks, arch35 entry |