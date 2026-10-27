---
type: Operator
title: "ForeachLerpScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachLerpScalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_host/foreach_lerp_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_host/config/ascend950/foreach_lerp_scalar_binary.json
    role: implementation
created_at: "2026-08-25T15:46:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachLerpScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachLerpScalarGetWorkspaceSize` + `aclnnForeachLerpScalar`，公开 ACLNN 两段式接口。 |
| 计算语义 | `y_i = x1_i + weight * (x2_i - x1_i)`；950 公开约束要求三列表对应 shape size 一致，列表最多 50 项。 |
| 输入输出 | ND TensorList，FP16/FP32/BF16；节点 `weight` 为 FP32 scalar。 |
| 执行类型 | 单 Kernel AIV_VECTOR：arch35 entry + MicroAPI `RegTensor<float>` body。 |
| Kernel 路线 | R-001 FP32 RegTensor；R-002 FP16/BF16 FP32 RegTensor cast。 |
| 核心资源 | TensorList `ListTensorDesc`、3 个 depth=2 Queue、FP32 RegTensor/MaskReg；32 B workspace 预留但 body 不使用。 |
| 来源闭包 | `source_closed`：ACLNN 文档→OpDef/框架 bridge→priority selector/regbase POD→arch35 entry→共享 binary pipeline+MicroAPI helper→950 CMake/config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对列表中每个对应 Tensor 的每个元素执行 lerp。MicroAPI body 在 `weight∈(-0.5,0.5)` 时计算 `x1 + weight*(x2-x1)`；否则计算 `x2 + (weight-1)*(x2-x1)`。两式等价。950 输出 local 独立于两个输入 local，因此无论分支均存入 `outQueue`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 每项 0–8 维 ND | FP16/FP32/BF16 | 列表 1–50 项（公开文档最大 50，regbase Host 显式拒绝 0 或 >50）；可含空 Tensor。 |
| `x2` | 与 `x1` 对应 shape size、ND | 与 `x1` 相同 | 列表长度、dtype 一致；公开约束要求对应 shape size 一致。 |
| `weight` | scalar | ACLNN FLOAT32/DOUBLE；节点 FP32 | 单元素；框架完成 scalar lowering。 |
| `out` | 与输入对应 shape size、ND | 与 `x1` 相同 | 不支持非连续输出；公开 950 约束要求 shape size 一致。 |

### 入口与版本

固定仓只有该公开两段式版本。ACLNN inner bridge 为生成边界；OpDef 注册 `ascend950` 并开启 dynamic rank/shape，950 binary config 列出三种 dtype，README/API 产品表明确支持 Ascend 950PR/950DT，故公开支持与实现接线一致。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 第一段执行文档列出的 nullptr、dtype、列表内部一致性、shape 和 8 维上限检查。OpDef 将两个输入列表和 scalar 标记 `AutoContiguous`；输出公开约束为连续。Tiling selector 内，priority 10000 membase 先读取 shape/platform 后因 regbase SoC 返回 `GRAPH_PARAM_INVALID`，随后 priority 30000 `ForeachRegbaseTilingBinaryScalar` 执行：检查 x1 列表 1–50、非负维、x2 列表数/dtype/对应元素数、weight FP32 单元素、out 列表数/dtype/容量。

### 参数、Workspace 与生命周期

`ForeachBaseClass::GetPlatformInfo` 产生 `numBlocks=AIV core num` 与 `ubSize`。节点 workspace 固定 32 B；`ForeachRegbaseBinary::Init` 接收但不读取 workspace。ACLNN 总 workspace 由 executor 第一段返回，可能叠加 `AutoContiguous` 框架临时区；固定仓未暴露生成 bridge 的聚合公式。调用方拥有 workspace，第二段在 stream 上异步使用。

### 启动图与失败传播

唯一节点为 `foreach_lerp_scalar`，`blockDim=numBlocks_`，POD `ForeachSoloTilingDataRegbase`，输入/输出直接为 TensorList GM 描述符。没有中间 workspace 边、第二 Kernel 或跨核同步。priority 10000 的不适用状态继续到 30000；30000 的任何参数/资源失败返回 `GRAPH_FAILED` 并阻断 launch。entry 未处理未知 key，Host 只为支持 dtype 生成 10001/10002/10004。

## Route 与参数设计

### 分核与资源计算

令 `N=sum_i numel(x1_i)`、`s=sizeof(T)`、`E=4096/s` 元素。`numBlocks=min(AIV核数,80,ceil(N/E))`；若 `N=0` 强制 1 核并写 `start=end=0,startOffset=0,endOffset=-1`。非空时按 `E` 为粒度将数据块均分，前余数核多一个 `E`，允许核切片跨 Tensor。

UB Host 算式：`bytesPerElem = s*2*2 + s*2 = 6s`（两个输入与一个输出，各 depth 2），`L=floor_align((U/6s)*s,32)/s` 元素，写入 `inputsTensorUbSize`。Device 每个 Queue 的每 slot 为 `L*s` B，因此精确 `UB_total=6L*s<=U`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 RegTensor | priority 30000；key 10002；`ForeachLerpScalarRegbase<float>` | regbase SoC、FP32、全部检查成功 | 失败终止。 |
| 2 | R-002 FP16/BF16 FP32 RegTensor cast | priority 30000；key 10001/10004；`ForeachLerpScalarRegbase<half/bfloat16_t>` | regbase SoC、FP16/BF16、全部检查成功 | 失败终止。 |

```text
registry priority 10000 membase
  -> IsCapable = !IsRegbaseSocVersion = false -> GRAPH_PARAM_INVALID
registry priority 30000 ForeachRegbaseTilingBinaryScalar
  -> IsCapable = IsRegbaseSocVersion = true
  -> dtype FP16/FP32/BF16 -> key 10001/10002/10004
entry -> ForeachLerpScalarImpl<T> -> ForeachLerpScalarRegbase<T> + ForeachRegbaseBinary base
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 10002/regbase | 950、FP32、列表 1–50、shape size/dtype/scalar/output 校验成功 | 单 AIV Kernel | `ForeachLerpScalarRegbase<float,ForeachSoloTilingDataRegbase>` 继承式复用 `ForeachRegbaseBinary` | 3×depth2 Queue；FP32 RegTensor；Queue 同步 | 每 tile 均用实际字节 `DataCopyPad`；MaskReg 保护尾 lane。 |
| R-002 | 10001/10004/regbase | 同上，FP16/BF16 | 单 AIV Kernel | 同模板的 half/bfloat16_t 实例 | 相同 Queue；load 时 unpack+升 FP32，store 时 RINT+pack | 同上；无 fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| selector | priority 10000 不可用；priority 30000 `IsRegbaseSocVersion` | arch35 body 使用 1000x key 与 MicroAPI | 两 Route | 10000 fallthrough；30000 不适用则无模板失败 | `ForeachRegbaseTiling::IsCapable` |
| Tensor 数 | `1<=count<=50` | POD `tensorDataCountList[50]` | 两 Route | `GRAPH_FAILED` | `ForeachRegbaseTiling::GetShapeAttrsInfo` |
| UB 对齐 | `L=floor_align((U/6s)*s,32)/s` | 三个 depth2 Queue 完整占用 6Ls | 两 Route | 无源码 fallback；异常 0 容量不可用 | `ForeachRegbaseTilingBinaryScalar::DoOpTiling` |
| MicroAPI tail | `repeatTimes=ceil(dataCount/(VL_SIZE/4))`，每轮 `UpdateMask<float>(sreg)` | 所有算术在 FP32 lane，mask 排除无效尾 lane | 两 Route | 同 Route mask tail | `ForeachLerpScalarRegbase::DoCompute` |
| weight 分支 | strict `(-0.5,0.5)` | 切换锚点和 Axpy scalar | 两 Route | 同 Route 另一分支 | `ForeachLerpScalarRegbase::Compute` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | `uint32_t`/32 | 第 1 字段；宏展开后的精确 size/offset 固定仓不可见 | 元素 | `BinaryScalar::DoOpTiling` | `ForeachRegbaseBinary::ParseTilingData` | 32 B 对齐换算后窄化存储。 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 第 2 字段；精确 offset 不可见 | 元素 | base `GetShapeAttrsInfo`/`DoOpTiling` | `Process` | 列表 1–50。 |
| 同上 | `tensorStartList[80]` | `uint16_t[80]` | 第 3 字段；精确 offset 不可见 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | `[0,blockDim)` 有效。 |
| 同上 | `tensorEndList[80]` | `uint16_t[80]` | 第 4 字段；精确 offset 不可见 | Tensor 索引 | 同上 | 同上 | inclusive。 |
| 同上 | `tensorStartOffsetList[80]` | `int64_t[80]` | 第 5 字段；精确 offset 不可见 | 元素 | 同上 | `Process` | inclusive。 |
| 同上 | `tensorEndOffsetList[80]` | `int64_t[80]` | 第 6 字段；精确 offset 不可见 | 元素 | 同上 | `Process` | inclusive；全空时首核为 -1。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10002/regbase | `foreach_regbase_tiling.cpp::ForeachRegbaseTilingBinaryScalar` | `ForeachSoloTilingDataRegbase` | `arch35/foreach_lerp_scalar.cpp::foreach_lerp_scalar` | `ForeachLerpScalarRegbase<float>`；`ForeachRegbaseBinary`；load/store helper | 950 `add_kernel_sources` + ascend950 binary config |
| R-002 | 10001/10004/regbase | 同上 | 同上 | 同上 | half/bfloat16_t 实例 + load/store cast traits | 同上 |

## Kernel 实现

### 公共执行流程

entry 创建 `TPipe`，按 key 调用 `ForeachLerpScalarImpl<T>`，解析 POD 后构造派生谓词类；其 `Init` 先调用 `ForeachRegbaseBinary::Init` 绑定三个 `ListTensorDesc` 并分配 Queue，再读取 FP32 weight。base `Process` 按核的 Tensor 范围循环，每个切片按 `L` 元素 tile：两路 `DataCopyPad` 入 Queue→派生 `Compute`→独立 out Queue→`DataCopyPad` 到 GM。末尾 `TPipe::Destroy`。

### Route R-001：FP32 RegTensor

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host selector | `foreach_regbase_tiling.h` | `ForeachRegbaseTiling::IsCapable` | regbase SoC | regbase。 |
| Tiling | `foreach_regbase_tiling.cpp` | `BinaryScalar::GetTilingKey` | `DT_FLOAT` | 10002。 |
| Router | `arch35/foreach_lerp_scalar.cpp` | `foreach_lerp_scalar` | `TILING_KEY_IS(10002)` | `ForeachLerpScalarImpl<float>`。 |

#### Device 数据流

##### 函数级调用链

```text
Tiling4ForeachLerpScalarTiling -> registry -> ForeachRegbaseTilingBinaryScalar
foreach_lerp_scalar(key=10002)
  -> ForeachLerpScalarImpl<float>
    -> ForeachLerpScalarRegbase::Init -> ForeachRegbaseBinary::Init
    -> ForeachRegbaseBinary::Process -> CopyIn -> Compute(pred_.Compute) -> CopyOut
      -> ForeachLerpScalarRegbase::DoCompute<float>
```

##### 源码等价伪码

```text
for assigned tensor slice:
  for tile t, count=min(L, remaining):
    DataCopyPad x1,x2 GM[t*L], blockCount=1, blockLen=count*4 -> input Queues
    allocate outQueue slot
    for r in [0, ceil(count/(VL_SIZE/4))):
      mask = UpdateMask<float>(remaining_sreg)
      a=load_fp32(x1Local+r*VL/4); b=load_fp32(x2Local+r*VL/4)
      if -0.5 < weight < 0.5: out = a + weight*(b-a)
      else: out = b + (weight-1)*(b-a)
      masked store_fp32(outLocal+r*VL/4,out)
    enqueue out; DataCopyPad valid bytes to y GM[t*L]; free
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM TensorList | `ListTensorDesc.GetDataPtr(i)+cursorStart+tL` | `count` elements | FP32/ND | caller | CopyIn/CopyOut | caller | actual byte length。 |
| input Queue 1 | slot | depth2×`4L` B | FP32 | CopyIn | Compute | TPipe/tile | Free after Compute。 |
| input Queue 2 | slot | depth2×`4L` B | FP32 | CopyIn | Compute | 同上 | 同上。 |
| out Queue | slot | depth2×`4L` B | FP32 | MicroAPI store | CopyOut | 同上 | 独立输出，不覆盖输入。 |
| RegTensor | register scope | 源码声明 3 个 FP32 RegTensor（`inRegToFloat` 未使用，活跃 2 个）+ MaskReg；物理字节数不可见 | FP32 lanes | load/Sub/Axpy | store | 每 repeat | 结果原位复用 a 或 b reg；mask 保护有效 lane。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in1["tensorOneQueue · depth=2 · 2Ls B"]:1
  in2["tensorTwoQueue · depth=2 · 2Ls B"]:1
  out["outQueue · depth=2 · 2Ls B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2Ls + 2Ls + 2Ls = 6Ls
L = floor_align((U/(6s))*s,32)/s, s=4
UB_total <= U
dynamic_ub = N/A
workspace_total = 32 B 预留（body 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue slots | CopyIn | Compute | Free 后下一 tile | `DeQue/FreeTensor` | Compute 完成后才 Free。 |
| out Queue slot | masked store | CopyOut | Free 后下一 tile | `EnQue/DeQue/FreeTensor` | CopyOut 完成后 Free。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→Vector | Queue `EnQue/DeQue` | MTE2 | Vector | input slots | blockLen=count*4。 |
| Vector | `__VEC_SCOPE__` + MaskReg | load | Sub/Axpy/store | registers/out local | FP32；最后 repeat mask。 |
| Vector→GM | out Queue | Vector | MTE3 CopyOut | out slot | `DataCopyPad` 只写 count。 |
| Kernel lifetime | `TPipe::Destroy` | 所有 Queue ops | entry return | TPipe resources | 无跨核同步。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_regbase_tiling.cpp` | `BinaryScalar::DoOpTiling/GetTilingKey` | 6s UB、key 10002、校验。 |
| Kernel entry/router | `foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar.cpp` | `foreach_lerp_scalar` / `ForeachLerpScalarImpl<float>` | POD、pipe、模板映射。 |
| Device body/helper | `foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar_regbase.h` | `DoCompute<true/false>` | MaskReg、Sub/Axpy、FP32 store。 |
| Device body/helper | `foreach_utils/op_kernel/arch35/foreach_regbase_binary.h` | `Process/CopyIn/Compute/CopyOut` | Queue、tile、TensorList 地址与 tail DMA。 |

### Route R-002：FP16/BF16 FP32 RegTensor cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host selector | `foreach_regbase_tiling.h` | regbase `IsCapable` | 950 | regbase。 |
| Tiling | `foreach_regbase_tiling.cpp` | `BinaryScalar::GetTilingKey` | FP16/BF16 | 10001/10004。 |
| Router | `arch35/foreach_lerp_scalar.cpp` | entry | key 10001/10004 | half/bfloat16_t 实例。 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_scalar(key=10001/10004)
  -> ForeachLerpScalarImpl<T> -> ForeachLerpScalarRegbase<T>::Init
  -> ForeachRegbaseBinary<T>::Process/CopyIn/Compute/CopyOut
    -> DoCompute -> ops::LoadOneTensorForDtypeT<T>
                 -> FP32 Sub/Axpy
                 -> ops::StoreOneTensorForDtypeT<T>
```

##### 源码等价伪码

```text
for each count<=L tile:
  copy actual FP16/BF16 bytes to two input Queues
  for each FP32 vector-width repeat:
    mask=UpdateMask<float>(remaining)
    load T with DIST_UNPACK_B16; cast T->FP32 for a,b
    delta=b-a
    out = (-0.5<weight<0.5) ? a+weight*delta : b+(weight-1)*delta
    cast FP32->T with NO_SAT + CAST_RINT + ZEROING
    store T with DIST_PACK_B32 under mask into out Queue
  copy actual T bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | descriptor + `cursorStart+tL` | count T elements | FP16/BF16 ND | caller | CopyIn/CopyOut | caller | blockLen=count*2。 |
| 三个 Queue | slot | 各 depth2×`2L` B | T | DMA/Vector | Vector/DMA | TPipe | 总 `12L` B。 |
| FP32 RegTensor | repeat-local | 两个活跃 reg 承载 a、b/delta 并原位形成 out；另有一个未使用声明；物理字节不可见 | FP32 | unpack Cast/Sub/Axpy | pack Cast/store | `__VEC_SCOPE__` | 无 UB FP32 临时 Tensor。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in1["tensorOneQueue · depth=2 · 4L B"]:1
  in2["tensorTwoQueue · depth=2 · 4L B"]:1
  out["outQueue · depth=2 · 4L B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4L + 4L + 4L = 12L B = 6Ls (s=2)
L = floor_align((U/12)*2,32)/2
UB_total <= U
dynamic_ub = N/A
workspace_total = 32 B 预留（body 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Vector store | Compute/CopyOut | Free 后下一 tile | Queue 生命周期 | depth2 容量与 Host L 一致。 |
| FP32 registers | load/cast/Sub/Axpy | store helper | 下一 repeat | `__VEC_SCOPE__` 顺序 | MaskReg 必须随剩余元素更新。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load | `DIST_UNPACK_B16` + Cast ZEROING | UB | FP32 RegTensor | input lanes | FP16/BF16→FP32。 |
| compute | MaskReg/Sub/Axpy | FP32 regs | store helper | valid lanes | strict weight threshold。 |
| store | `CAST_RINT`, NO_SAT, ZEROING + `DIST_PACK_B32` | FP32 reg | out UB | output lanes | 一次降精度，mask tail。 |
| DMA | Queue + DataCopyPad | GM/UB | UB/GM | actual count | blockCount=1、stride=0。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_regbase_tiling.cpp` | `BinaryScalar::DoOpTiling/GetTilingKey` | key 10001/10004、L、blockDim。 |
| Kernel entry/router | `foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar.cpp` | `ForeachLerpScalarImpl<half/bfloat16_t>` | dtype router。 |
| Device body/helper | `foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar_regbase.h` | `DoCompute` | FP32 RegTensor 算术与 mask。 |
| Device body/helper | `common/inc/op_kernel/load_store_utils.h` | `LoadOneTensorForDtypeT` / `StoreOneTensorForDtypeT` | unpack/cast/round/pack。 |

## 算子特定优化

### Trick T-001：用 MaskReg 在一个 MicroAPI 循环统一主块与尾块

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：每个 tile 的 `repeatTimes=ceil(dataCount/(VL_SIZE/sizeof(float)))`，每轮由剩余 `sreg` 更新 mask。
- Golden 来源：[`foreach_lerp_scalar_regbase.h:58-79`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar_regbase.h#L58-L79)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
uint32_t sreg = (uint32_t)dataCount;
float weightVal = weightVal_ - FLOAT_NUM_ONE;

__VEC_SCOPE__
{
    MaskReg maskReg;
    RegTensor<float> inRegToFloat;
    RegTensor<float> tensorOneRegToFloat;
    RegTensor<float> tensorTwoRegToFloat;
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
        maskReg = UpdateMask<float>(sreg);
        ops::LoadOneTensorForDtypeT<T>(tensorOneUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
        ops::LoadOneTensorForDtypeT<T>(tensorTwoUbAddr, tensorTwoRegToFloat, maskReg, i * dataCountPerLoop);
        if constexpr (needExchange) {
            Sub(tensorTwoRegToFloat, tensorTwoRegToFloat, tensorOneRegToFloat, maskReg);
            Axpy(tensorOneRegToFloat, tensorTwoRegToFloat, weightVal_, maskReg);
            ops::StoreOneTensorForDtypeT<T>(outUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
        } else {
            Sub(tensorOneRegToFloat, tensorTwoRegToFloat, tensorOneRegToFloat, maskReg);
            Axpy(tensorTwoRegToFloat, tensorOneRegToFloat, weightVal, maskReg);
```

#### 机制与设计原因

循环步长固定按 FP32 lane 数 `VL_SIZE/4`，即使存储 dtype 为 16 位也先升到 FP32；`UpdateMask` 同时约束 load、Sub、Axpy 与 store，最后一次 repeat 不需要独立 scalar/tail body。`needExchange` 是编译期 bool，使每次调用只保留选中的 operand 顺序。

#### 迁移边界

- 适用：所有输入都能由 helper 映射到 FP32 RegTensor 的逐元素运算。
- 前提：循环粒度按计算 dtype FP32 而非存储 dtype；所有 primitive 使用同一 mask。
- 失效：存在跨 lane 邻接、reduce 或需要读取 padding lane 时，不能直接复用该 tail 方案。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：低精度只在寄存器升精度，不分配 FP32 UB 临时区

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`T=half` 或 `bfloat16_t`。
- Golden 来源：[`load_store_utils.h:59-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h#L59-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T *input, MicroAPI::RegTensor<float> &dst,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half *)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16,
                    ((__local_mem__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float *)(input) + (offset)));
    }
}
```

#### 机制与设计原因

UB 只保留原 dtype 的两输入一输出 Queue，升精度值驻留 `RegTensor<float>`；与 classic 路线相比无需 FP32 `TQue`。store helper 用 `CAST_RINT` 后 pack 回原 dtype，精度顺序仍是“输入升 FP32→Sub/Axpy→一次降精度”。

#### 迁移边界

- 适用：MicroAPI 可表达、每次只需有限个 FP32 RegTensor 的 elementwise 算子。
- 前提：950/regbase、对应 unpack/pack helper 存在、寄存器生命周期不跨 repeat。
- 失效：中间量需跨 tile/Kernel 保存或寄存器数量不足时，需要 UB/workspace 物化。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 RegTensor | R-002 FP16/BF16 FP32 RegTensor cast | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | regbase priority 30000，10002 | 同 priority，10001/10004 | dtype key。 |
| 模板与 dtype | `Regbase<float>` | `Regbase<half/bfloat16_t>` | load/store helper specialization。 |
| 循环与分核 | 4096 B block；tile L；repeat VL/4 | 完全相同 | 计算统一为 FP32 lanes。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3×depth2 Queue + FP32 regs | 相同，但 load/store 有低精度临时 regs | 无 dynamic UB/通信。 |
| 内存总占用与复用 | `24L` B | `12L` B | 均为 `6Ls`。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + FP32 MicroAPI | 另含 unpack/cast/pack | AIV_VECTOR。 |
| 同步、精度与 tail | Queue + mask，FP32 | FP32 中间 + RINT，mask | tail 同一 loop。 |
| fallback/失败边界 | 未知 key 无 body | 同上 | 无其它 Kernel。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | membase priority 10000 | membase priority 10000 | regbase priority 30000，显式 1–50 列表校验 |
| Kernel/资源 | classic AIV，低精度用 FP32 UB Queue | 同左 | arch35 MicroAPI，低精度 FP32 值仅驻留寄存器 |

## 限制与失败边界

- 公开 950 约束要求 `x1/x2/out` 对应 shape size 一致且列表最多 50；Host 对 x2 实际比较元素数，对 out 使用“容量不少于输入”的公共检查，正式使用应遵守更严格的公开约束。
- regbase Host 显式拒绝列表为空或超过 50、负维、维数超过 8、weight 非 FP32/非单元素、列表 dtype/长度不一致。
- `inputsTensorUbSize` 最终写入 uint32；固定源码未显式检查从 UB 计算结果到 uint32 的窄化，但平台注册 UB 规模是其实际前提。
- ACLNN 生成 bridge、scalar DOUBLE lowering、executor workspace 聚合在固定仓不可见；标为框架边界，不推测具体实现。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md` | 原型、950 产品表/最大 50 | 公开入口与约束。 |
| 入口声明/约束 | `foreach/foreach_lerp_scalar/README.md` | 公式和支持表 | 语义/平台。 |
| 入口实现 | `foreach/foreach_lerp_scalar/op_host/foreach_lerp_scalar_def.cpp` | `ForeachLerpScalar` | ascend950 OpDef。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachLerpScalarTiling` | registry 入口。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `ForeachRegbaseTilingBinaryScalar` | 校验、分核、UB、key/POD/workspace。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_regbase_tiling.h` | `ForeachRegbaseTiling` / `ForeachRegbaseTilingBinaryScalar` | 平台能力谓词与类层次。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachSoloTilingDataRegbase` | POD 字段与 membase guard。 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `DoTilingImpl` | priority/fallthrough。 |
| Kernel entry/body | `foreach/foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar.cpp` | entry/Impl | key router、pipe。 |
| Kernel entry/body | `foreach/foreach_lerp_scalar/op_kernel/arch35/foreach_lerp_scalar_regbase.h` | `ForeachLerpScalarRegbase` | lerp MicroAPI body。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_binary.h` | `ForeachRegbaseBinary` | Queue/tile/DMA/TensorList。 |
| Kernel entry/body | `common/inc/op_kernel/load_store_utils.h` | load/store helpers | dtype cast/round/pack。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_host/CMakeLists.txt` | `add_modules_sources` | Host 源与 `foreach_utils` 依赖。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 与编译选项。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_host/config/ascend950/foreach_lerp_scalar_binary.json` | op_list | 950 三 dtype binary。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段描述 L 与跨 Tensor 核切片 | uint32 L、50/80 数组、字段顺序、inclusive offset | `ForeachSoloTilingDataRegbase` |
| 调用顺序 | base 管理 `CopyIn→派生 Compute→CopyOut` | 派生对象引用与 TPipe 生命周期覆盖 Process | `ForeachLerpScalarImpl` / `ForeachRegbaseBinary` |
| workspace/内存 | 三 Queue 精确 `6Ls`；32 B workspace 不用 | depth2、L 的元素单位、32 B FloorAlign | `BinaryScalar::DoOpTiling` / base `Init` |
| 同步 | Queue 生命周期 | input 仅 Compute 后 Free，out 仅 CopyOut 后 Free | base `Compute/CopyOut` |
| 精度与尾块 | FP32 RegTensor、RINT store、MaskReg tail | FP32 loop 宽度、同一 mask、strict weight 阈值 | `DoCompute` / load-store traits |
| 失败边界 | 10000 不适用继续 30000；30000 校验失败终止 | 不得把未知 key 当 fallback | registry / regbase Host / entry |

- [Axpy](../../../../apis/simd_api/basic_api/reg_vector_compute/composite_compute/axpy.md)
