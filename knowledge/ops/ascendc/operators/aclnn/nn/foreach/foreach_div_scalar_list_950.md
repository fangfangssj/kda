---
type: Operator
title: "ForeachDivScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachDivScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/docs/aclnnForeachDivScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/foreach_div_scalar_list_def.cpp
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
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary_scalar_list.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_kernel/arch35/foreach_div_scalar_list_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_kernel/arch35/foreach_div_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/config/ascend950/foreach_div_scalar_list_binary.json
    role: implementation
created_at: "2026-08-25T15:32:31Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachDivScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachDivScalarListGetWorkspaceSize` + `aclnnForeachDivScalarList`，公开两段式 ACLNN |
| 计算语义 | 对列表位置 `i` 的每个有效元素 `j` 计算 `out[i][j] = x[i][j] / scalars[i]` |
| 输入输出 | `x/out`：1–50 个 ND Tensor、rank 0–8、统一 `FLOAT16/FLOAT32/BFLOAT16`；`scalars`：等长 `FLOAT32` ScalarList |
| 执行类型 | 单 Kernel `foreach_div_scalar_list`：`AIV_VECTOR`（arch35 regbase MicroAPI） |
| Kernel 路线 | R-001 FP16/FP32-reg；R-002 FP32；R-003 BF16/FP32-reg，分别由 key 10001/10002/10004 路由 |
| 核心资源 | TensorList 描述符 GM、标量 GM、双缓冲 `VECIN/VECOUT`、MaskReg 与 FP32 RegTensor；无 Device workspace 数据 |
| 来源闭包 | `source_closed`：公开 ACLNN 约束、OpDef、priority selector、regbase Host/Tiling/POD、全部 dtype entry/body/helper 与 Ascend 950 构建闭合；生成 ACLNN bridge/launcher 为框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`、`scalars=[s_0,...,s_{n-1}]`，输出有效域满足 `y_i[j]=x_i[j]/s_i`。950 三种 dtype 的除法统一在 FP32 vector register 中完成：FP16/BF16 load 时升为 FP32、store 时以 `CAST_RINT` 降回原 dtype；FP32 直接 load/store。输出若比输入容量大，Kernel 仍只写与 `x_i` storage element count 相同的前缀。除零没有算子级特殊处理。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；ND；每项 rank 0–8 | 列表内统一 FP16/FP32/BF16 | 1–50 项；维度允许 0，不允许负值；支持空 Tensor；OpDef `AutoContiguous()` |
| `scalars` | ScalarList，元素数=`len(x)` | FP32 | 每项对应一个 Tensor，Kernel 按 Tensor 索引读取 |
| `out` | TensorList；ND；每项 rank 0–8 | 与 `x` 相同 | 项数相同；storage size≥输入；公开接口不支持非连续输出 |

### 入口与版本

固定仓公开的最新且唯一入口是无版本后缀的 `aclnnForeachDivScalarList`。API 产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，binary config 发布 FP16/FP32/BF16 三种组合，Kernel CMake 为 `ascend950` 选择 arch35 源。固定仓未保存生成的 ACLNN bridge、executor/launcher 具体函数和公开 C header 实体；`ACLNNTYPE aclnn` 生成边界不作为算子自有源码缺口。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段公开 ACLNN 校验空指针、dtype、列表长度/一致性、rank 与输出 shape。进入 regbase Tiling 后，`ForeachRegbaseTilingUnaryScalarList::GetShapeAttrsInfo` 先调用基类：获取动态输入实例数并要求 `1≤n≤50`，逐 Tensor 校验描述符、统一 dtype、rank≤8、各维非负并累加 storage element count；再由 `CheckScalarList(1)` 要求 scalar dtype 为 FP32、x 为 FP16/FP32/BF16、scalar 元素数等于 `n`。`DoOpTiling` 后调用 `CheckOutput`，验证输出项数、dtype、rank、容量。输入的 `AutoContiguous` 复制属于生成框架；具体节点与临时区固定仓不可见。

### 参数、Workspace 与生命周期

设总元素 `N`、元素字节 `S`、每 4096 B 的元素数 `G=4096/S`、AIV 核数 `Caiv`。Host 取 `C=min(Caiv,80,ceil(N/G))`；`N=0` 时强制 `C=1` 并写 `endOffset=-1`。每核得到跨 TensorList 连续域的首尾 Tensor 与 offset。算子 `GetWorkspaceSize/PostTiling` 均给出 32 B，Kernel 虽接收 `workspace`，但 `ForeachRegbaseUnaryScalarList::Init` 不读取它；因此没有 Device workspace slice 或跨节点生命周期。ACLNN 总 workspace 仍可能包含生成的 contiguous 临时区，其精确合并公式固定仓不可见。

### 启动图与失败传播

启动图只有一个 Device 节点：生成 ACLNN bridge 调用注册算子；`Tiling4ForeachDivScalarListTiling` 经模板注册表选 regbase，随后在调用者 stream 上启动 `foreach_div_scalar_list(tensor1,scalars,outputs,workspace,tiling)`，blockDim=`C`，POD=`ForeachSoloTilingDataRegbase`。节点无前后 Kernel、workspace edge、event 或 atomic。任何校验、Tiling、POD 保存或 launch 失败会终止这一个节点；生成 launcher 的精确错误包装固定仓不可见。

## Route 与参数设计

### 分核与资源计算

注册表以较小 priority 先试：priority 10000 membase 在 950 `IsCapable=false`，返回 `GRAPH_PARAM_INVALID`；priority 30000 regbase 的 `IsCapable=IsRegbaseSocVersion` 后成功。分核以 4096 B 数据块为粒度，前 `ceil(N/G) mod C` 核多一个 `G` 元素，切片可以跨 Tensor。设 UB 字节数 `Uplat`、平台 UB block bytes `A=GetUbBlockSize(context)`，tile 元素数 `E` 为：FP32 `floorAlign(floor(Uplat/16)·4,A)/4`；FP16/BF16 `floorAlign(floor(Uplat/12)·2,A)/2`。POD 将 `E` 窄化到 `uint32_t inputsTensorUbSize`；固定源码未检查窄化溢出。实际 Queue 总字节为 `4ES`，FP16/BF16 Host 的额外 `4E` 字节/元素预算不对应 `InitBuffer`，为 FP32 register 计算路线留下保守资源余量。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | membase 候选 | priority 10000 | `!IsRegbaseSocVersion(context)`；950 为 false | `GRAPH_PARAM_INVALID`，继续 priority 30000 |
| 2 | regbase 公共模板 | priority 30000，`ForeachRegbaseTilingUnaryScalarList` | `IsRegbaseSocVersion(context)` 且校验通过 | 校验失败终止，无其它 fallback |
| 3 | R-001 FP16 | key 10001，`ForeachDivScalarListRegbase<half,...>` | `dataType_=DT_FLOAT16` | unsupported dtype→失败/key0 |
| 4 | R-002 FP32 | key 10002，`...<float,...>` | `dataType_=DT_FLOAT` | 同上 |
| 5 | R-003 BF16 | key 10004，`...<bfloat16_t,...>` | `dataType_=DT_BF16` | 同上 |

```text
priority 10000 membase: IsCapable(A5)=false -> fallthrough
priority 30000 ForeachRegbaseTilingUnaryScalarList: IsCapable(A5)=true
  validate -> split -> E -> workspace=32 -> key(dtype)
    10001 -> R-001
    10002 -> R-002
    10004 -> R-003
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 10001 | 950；x/out FP16；scalar FP32；公共检查通过 | 单 AIV arch35 entry | `ForeachDivScalarListRegbase<half,ForeachSoloTilingDataRegbase>` → CRTP base | 2×FP16 in + 2×FP16 out；FP32 RegTensor；Queue 同步 | GM `DataCopyPad` + per-VReg `UpdateMask<float>` |
| R-002 FP32 | 10002 | 同上，x/out FP32 | 同上 | `<float,...>` | 同 Queue；FP32 direct register load/store | 同上 |
| R-003 BF16 | 10004 | 同上，x/out BF16 | 同上 | `<bfloat16_t,...>` | 2×BF16 in/out；FP32 RegTensor | 同上，store `CAST_RINT` |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| selector order | 10000 先于 30000；越小优先 | 同入口共存两种 POD/Kernel 世代 | 全部 | capability false 才继续 | `TilingRegistry::DoTilingImpl` |
| 950 capability | `IsRegbaseNpuArch` 为 DAV_3510/DAV_5102 | 仅 regbase 平台使用 key 100xx | 全部 | membase 或失败 | `IsRegbaseSocVersion`、`IsCapable` |
| 列表/core 上限 | `1≤n≤50`、`C≤80` | POD 数组分别为 50 Tensor、80 core | 全部 | Tiling 失败/截核 | `MAX_TENSOR_CONT_950/MAX_CORE_CONT_950` |
| 4096 B 分核 | `G=4096/S` | 避免为少于一个 4 KiB 数据块启动更多核 | 全部 | 无其他 Route | `SINGLE_CORE_PROCESS_DATA`、`DoOpTiling` |
| UB tile | `E=floorAlign(floor(Uplat/D)·S,A)/S`，`D=16` 或 12 | 4 个 Queue slots；半精度另预算 FP32 计算 | 全部 | 无小 UB fallback | `ForeachRegbaseTilingUnaryScalarList::DoOpTiling` |
| VReg loop | `L=VL_SIZE/sizeof(float)`；`repeatTimes=ceil(dataCount/L)` | 三 dtype 都以 FP32 lane 计算 | 全部 | 循环加 tail mask | `ForeachDivScalarListRegbase::Compute` |
| 空域 | `N=0 -> C=1,endOffset=-1` | `dataCount=0 -> quotient=0`，不搬运 | 全部 | 成功空写 | `DoOpTiling/Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | `uint32_t/32` | 第1字段；宏展开的 `sizeof/alignof/offsetof` 固定仓不可见 | 元素/tile | `DoOpTiling: set_inputsTensorUbSize(E)` | base `Init/ParseTilingData`，Queue 容量与 `maxDataCount` | int64→uint32 窄化无显式 guard |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第2字段；精确 offset 固定仓不可见 | 元素/每 Tensor | `GetShapeAttrsInfo` | `Process` | 可为0 |
| 同上 | `tensorStartList` | `uint16_t[80]` | 第3字段；精确 offset 固定仓不可见 | Tensor 索引/每核 | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | `blockIdx<C≤80` |
| 同上 | `tensorEndList` | `uint16_t[80]` | 第4字段；精确 offset 固定仓不可见 | Tensor 索引/每核 | 同上 | 同上 | 空域为0 |
| 同上 | `tensorStartOffsetList` | `int64_t[80]` | 第5字段；精确 offset 固定仓不可见 | 元素/每核 | 同上 | 同上 | 首 Tensor 内 offset |
| 同上 | `tensorEndOffsetList` | `int64_t[80]` | 第6字段；精确 offset 固定仓不可见 | 元素/每核，含尾 | 同上 | 同上 | 空域为-1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | 10001 | `foreach_regbase_tiling.cpp:ForeachRegbaseTilingUnaryScalarList::{GetShapeAttrsInfo,DoOpTiling}` | `ForeachSoloTilingDataRegbase` | `arch35/foreach_div_scalar_list.cpp:foreach_div_scalar_list` | `ForeachDivScalarListRegbase<half>`、`ForeachRegbaseUnaryScalarList`、load/store helper | Kernel CMake ascend950 + ascend950 binary JSON |
| R-002 FP32 | 10002 | 同上 | 同上 | 同上 | `...<float>` 与 direct FP32 helper | 同上 |
| R-003 BF16 | 10004 | 同上 | 同上 | 同上 | `...<bfloat16_t>` 与 BF16 cast helper | 同上 |

## Kernel 实现

### 公共执行流程

entry 在栈上创建 `TPipe`，按 key 调用 `ForeachDivScalarListImpl<T>`；该函数用 `GET_TILING_DATA_WITH_STRUCT` 解析 POD，构造 CRTP body，`Init` 后 `Process`，最后统一 `pipeOp.Destroy()`。Base 用 `ListTensorDesc` 解引用每个输入/输出 GM 指针，按当前核首尾元数据遍历 Tensor；每 Tensor 切成 `E` 元素 tile。每 tile 使用 `DataCopyPad` 进入双缓冲 VECIN，派生 `Compute` 将 scalar 读入 FP32、在 FP32 VReg 中除法，再经 VECOUT 双缓冲 `DataCopyPad` 回 GM。Queue 的 Alloc/EnQue/DeQue/FreeTensor 提供 MTE/Vector 顺序；没有显式 event、workspace、atomic 或跨核同步。

### Route R-001：FP16 输入、FP32 寄存器计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Selector | `foreach_tiling_func.cpp` | `Tiling4ForeachDivScalarListTiling` | registry dispatch | priority 30000 regbase |
| Tiling | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `dataType_=DT_FLOAT16` | 10001 |
| Router | `arch35/foreach_div_scalar_list.cpp` | `foreach_div_scalar_list` | `TILING_KEY_IS(FOREACH_TILING_KEY_HALF)` | `ForeachDivScalarListImpl<half>` |

#### Device 数据流

##### 函数级调用链

```text
framework launch foreach_div_scalar_list
  -> ForeachDivScalarListImpl<half>
    -> ForeachDivScalarListRegbase::Init -> ForeachRegbaseUnaryScalarList::Init
    -> Base::Process -> SingleTensorProcess -> CopyIn
      -> Base::Compute -> ForeachDivScalarListRegbase::Compute
        -> LoadOneTensorForDtypeT<half> -> Cast half→float -> Div -> Cast float→half,RINT -> Store
      -> CopyOut
  -> TPipe::Destroy
```

##### 源码等价伪码

```text
for tensor t in [startTensor,endTensor]:
  count = clippedEnd(t)-clippedStart(t)+1
  bind GM pointers at clippedStart
  for tile k in [0,ceil(count/E)):
    valid = min(E,count-k*E)
    DataCopyPad(valid*sizeof(half)) GM -> inQ
    scale = float(scalars[t]); scaleReg = Duplicate(scale)
    sreg=valid
    for r in [0,ceil(valid/(VL_SIZE/4))):
      mask=UpdateMask<float>(sreg)
      unpack+cast half UB -> float reg
      Div(float outReg,float inReg,scaleReg,mask)
      cast RINT+pack float reg -> half UB under mask
    DataCopyPad(valid*sizeof(half)) outQ -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `inDesc[t]+coreStart+kE` | `valid≤E` | FP16/ND | caller | CopyIn | caller | byte length=`valid·2` |
| inQueue | Queue slot | depth2 × `2E` B=`4E` B | FP16 | MTE2 | MicroAPI load | TPipe | Free after Compute |
| RegTensor | register file | `inRegToFloat/scaleValReg/outReg`，精确字节固定仓不可见 | FP32 lanes | Load/Duplicate/Div | Div/Store | 单次 `Compute` | masked overwrite |
| outQueue | Queue slot | depth2 × `2E` B=`4E` B | FP16 | Store | MTE3 | TPipe | Free after CopyOut |
| output GM | `outDesc[t]+coreStart+kE` | `valid` | FP16/ND | CopyOut | caller | caller | only valid bytes |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 4E B"]:2
  outQ["outQueue · depth=2 · 4E B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*(E*2) + 2*(E*2) = 8E B
Host budget denominator = 12 B/element; E=floorAlign(floor(Uplat/12)*2,A)/2
UB_total <= Uplat；剩余 Host 预算不形成额外 InitBuffer
register_bytes = 固定仓不可见（3 个 FP32 RegTensor + 临时 half RegTensor）
dynamic_ub = N/A
workspace_total = 32 B reserved, Base ignores workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | CopyIn | Compute | 后续 tile | `EnQue→DeQue→FreeTensor` | depth2 |
| outQueue slot | Compute | CopyOut | 后续 tile | `EnQue→DeQue→FreeTensor` | CopyOut 已消费 |
| RegTensor | Load/Duplicate | Div/Store | 下一 VReg loop | `__VEC_SCOPE__` 顺序与 mask | 当前 store 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→Vector | TQue | CopyIn | DeQue | in slot | `DataCopyPad` 只搬有效字节，无 padding |
| load | `DIST_UNPACK_B16` + cast ZEROING | half UB | FP32 Div | invalid lanes | FP16→FP32 |
| compute | `Div(...,mask)` | input/scale regs | store | masked lanes | FP32 division |
| store | `CAST_RINT` + `DIST_PACK_B32` | FP32 result | half out UB | invalid lanes | FP32→FP16，no-sat/RINT |
| UB→GM | TQue + DataCopyPad | outQ | MTE3 | out slot | valid bytes only |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `ForeachRegbaseTilingUnaryScalarList::DoOpTiling` | E、blockDim、POD、workspace |
| Kernel entry/router | `arch35/foreach_div_scalar_list.cpp` | key 10001 | half 实例 |
| Device body/helper | `foreach_div_scalar_list_regbase.h` | `Compute<half>` | FP32 register Div、mask loop |
| Device body/helper | `load_store_utils.h` | `Load/StoreOneTensorForDtypeT<half>` | unpack/cast/RINT/pack |

### Route R-002：FP32 直接寄存器计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT` | 10002 |
| Router | `arch35/foreach_div_scalar_list.cpp` | key FLOAT | key 10002 | `ForeachDivScalarListImpl<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_div_scalar_list -> ForeachDivScalarListRegbase<float>::Compute
  -> LoadOneTensorForDtypeT<float> -> Div<float> -> StoreOneTensorForDtypeT<float>
```

##### 源码等价伪码

```text
同 R-001 的 core/Tensor/tile/mask 循环；Load/Store helper 对 float 走 direct DataCopy，
不执行 dtype cast；每 tile 仍把 scalars[t] Duplicate 到 scaleValReg。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | ListTensorDesc + core/tile offset | `≤E` | FP32/ND | caller/Kernel | Kernel/caller | stream | valid bytes |
| inQueue | slot | `2·4E=8E` B | FP32 | CopyIn | load | pipe | depth2 |
| FP32 RegTensor | register | input/scale/output | FP32 | load/Duplicate/Div | Div/store | VEC_SCOPE | masked reuse |
| outQueue | slot | `8E` B | FP32 | store | CopyOut | pipe | depth2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 8E B"]:2
  outQ["outQueue · depth=2 · 8E B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 8E + 8E = 16E B
E = floorAlign(floor(Uplat/16)*4,A)/4
UB_total <= Uplat
register_bytes = 固定仓不可见
dynamic_ub = N/A
workspace_total = 32 B reserved, unused
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Store | Load/CopyOut | 后续 tile | Queue 生命周期 | 同 R-001 |
| registers | direct load/Div | Div/direct store | 下一 VReg loop | VEC_SCOPE order | mask protects tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue pipeline | EnQue/DeQue/Free | MTE2/Vector/MTE3 | next stage | slots | valid byte length |
| Vector | direct FP32 load/Div/store + mask | input/scale | output | tail lanes | no cast/round |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `DoOpTiling/GetTilingKey` | FP32 E、key10002 |
| Kernel entry/router | `arch35/foreach_div_scalar_list.cpp` | `Impl<float>` | FP32 route |
| Device body/helper | `load_store_utils.h` | float `else` branches | direct FP32 register copy |

### Route R-003：BF16 输入、FP32 寄存器计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_BF16` | 10004 |
| Router | `arch35/foreach_div_scalar_list.cpp` | key BF16 | key 10004 | `ForeachDivScalarListImpl<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_div_scalar_list -> ForeachDivScalarListRegbase<bfloat16_t>::Compute
  -> LoadOneTensorForDtypeT<bfloat16_t> -> Cast BF16→FP32
  -> Div FP32 -> Cast FP32→BF16,RINT -> StoreOneTensorForDtypeT<bfloat16_t>
```

##### 源码等价伪码

```text
与 R-001 相同，只把 unpack 输入类型和 pack 输出类型换为 bfloat16_t；
scale 始终直接读取 FP32，计算全程 FP32，尾 lane 由同一个 float MaskReg 控制。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | descriptor + offsets | `≤E` | BF16/ND | caller/Kernel | Kernel/caller | stream | valid bytes |
| inQueue | slot | `4E` B total | BF16 | CopyIn | unpack/cast | pipe | depth2 |
| FP32/BF16 RegTensor | register | compiler allocation不可见 | register layouts | load/cast/Div | store | VEC_SCOPE | per-loop masked reuse |
| outQueue | slot | `4E` B total | BF16 | cast/pack | CopyOut | pipe | depth2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 4E B"]:2
  outQ["outQueue · depth=2 · 4E B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 4E + 4E = 8E B
E = floorAlign(floor(Uplat/12)*2,A)/2
UB_total <= Uplat；Host 另留 4E B 等价预算但未 InitBuffer
register_bytes = 固定仓不可见
dynamic_ub = N/A
workspace_total = 32 B reserved, unused
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/cast-store | cast-load/CopyOut | 后续 tile | Queue 生命周期 | 同 R-001 |
| FP32 registers | BF16 cast/Div | Div/BF16 cast | 下一 VReg loop | VEC_SCOPE order | UpdateMask tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load | unpack B16 + ZEROING cast | BF16 UB | Div | tail lanes | BF16→FP32 |
| Div | FP32 masked | input/scale | output | tail lanes | FP32 |
| store | no-sat `CAST_RINT` + pack B32 | FP32 result | BF16 UB | tail lanes | FP32→BF16 |
| GM path | Queue + DataCopyPad | MTE/Vector | next stage | slots | valid bytes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | key 10004 / E formula | BF16 route resources |
| Kernel entry/router | `arch35/foreach_div_scalar_list.cpp` | `Impl<bfloat16_t>` | BF16 route |
| Device body/helper | `load_store_utils.h` | BF16 load/store specializations | cast traits、RINT、mask |

## 算子特定优化

### Trick T-001：半精度数据在 FP32 VReg 中完成除法且不物化 FP32 UB Tensor

#### 触发与源码依据

- 适用 Route：R-001、R-003；R-002 复用同一 FP32 register 主干但无 cast
- 触发条件：950 regbase route，每个 tile 的 `dataCount>0`
- Golden 来源：[`foreach_div_scalar_list_regbase.h:59-67`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_kernel/arch35/foreach_div_scalar_list_regbase.h#L59-L67)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
            Duplicate(scaleValReg, scaleVal);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                Div(outReg, inRegToFloat, scaleValReg, maskReg);
                ops::StoreOneTensorForDtypeT<T>(outUbAddr, outReg, maskReg, i * dataCountPerLoop);
            }
```

#### 机制与设计原因

`LoadOneTensorForDtypeT` 对 FP16/BF16 直接在 load 后 cast 到 FP32 RegTensor，`Div` 使用 FP32 scalar register，`StoreOneTensorForDtypeT` 再以 `CAST_RINT` pack 回原 dtype。FP32 中间结果只存在寄存器，不需要额外 FP32 TBuf/Queue；同一个 `MaskReg` 同时约束 load、Div、store，使最后一个 VReg 的无效 lane 不进入输出。Host 仍以每元素额外 `sizeof(float)` 的分母保守限制 E，但 Device `InitBuffer` 只有双输入/输出 Queue。

#### 迁移边界

- 适用：逐元素算子能在一个 VReg pipeline 内完成，且半精度只需一次升/降精度。
- 前提：arch35 MicroAPI、`Load/StoreOneTensorForDtypeT` 的 cast trait 与 FP32 scalar 语义保持不变；tail 使用 float mask。
- 失效：计算需要跨 tile 保存 FP32 中间量、跨核归约或超过寄存器生命周期时，必须显式分配 UB/workspace，不能照搬该布局。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key10001 | key10002 | key10004 | priority30000 regbase 后按 dtype |
| 模板与 dtype | `<half>` + FP32 regs | `<float>` | `<bfloat16_t>` + FP32 regs | input/output storage dtype |
| 循环与分核 | 4096 B/core block；E tile；VL/4 lanes | 同左 | 同左 | 统一 float lane 数 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 8E UB + FP32 regs | 16E UB + FP32 regs | 8E UB + FP32 regs | Queue bytes随S变化 |
| 内存总占用与复用 | Host预算12E，实际8E | Host/实际16E | Host预算12E，实际8E | 半精度留保守余量 |
| 搬运/Vector/SIMT/NDDMA/通信 API | pad + unpack/cast/Div/pack | pad + direct/Div/direct | pad + unpack/cast/Div/pack | 全为AIV，无其它执行族 |
| 同步、精度与 tail | Queue；FP32；RINT；mask | Queue；FP32；mask | Queue；FP32；RINT；mask | helper specialization |
| fallback/失败边界 | 无 | 无 | 无 | unknown key不执行 body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持 |
| Host/Launcher | membase priority10000、32 B block 分核 | 同 A2 | regbase priority30000、4096 B 分核、最多80核 |
| Kernel/资源 | Level-0 API；BF16有FP32 UB Queue | 同 A2 | arch35 MicroAPI；FP32 register pipeline、无 FP32 UB 中间 Tensor |

## 限制与失败边界

- 公开约束为 TensorList 1–50、rank≤8、x/out 仅 FP16/FP32/BF16、scalars 仅 FP32 且等长、输出连续且容量足够；950 Host 还显式拒绝负维。
- `inputsTensorUbSize` producer 使用 int64 中间值，POD 字段为 uint32，固定源码未显式检查窄化；也未检查 E=0，没有更小 tile fallback。
- scalar GM 在每次 tile 的 `Compute` 中按 Tensor index 重新 `GetValue`；不得误改为元素 index，也不能在跨 Tensor 时复用旧 scalar。
- unknown key 没有错误 `else`，entry 只销毁 pipe 并结束；正常 ascend950 binary config 仅发布三种受支持 dtype。
- 生成 ACLNN bridge/launcher、AutoContiguous 节点及 executor 总 workspace 合并固定仓不可见；POD 宏外部定义使精确 `sizeof/alignof/offsetof` 也不可见。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_div_scalar_list/docs/aclnnForeachDivScalarList.md` | ACLNN 两段式原型、产品表 | 950公开支持、参数与错误 |
| 入口实现 | `foreach/foreach_div_scalar_list/op_host/foreach_div_scalar_list_def.cpp` | `ForeachDivScalarList` OpDef | dtype/layout/AutoContiguous/ascend950 config |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | dispatch + membase priority10000 | 外层 selector |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `ForeachRegbaseTilingUnaryScalarList` | 校验、分核、E、key、POD、workspace |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_regbase_tiling.h` | `ForeachRegbaseTiling::IsCapable` | 950 regbase capability |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachSoloTilingDataRegbase` | POD字段/注册 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority从小到大、fallthrough |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_util.h` | `IsRegbaseSocVersion` | DAV_3510/DAV_5102 判定 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary_scalar_list.h` | CRTP base | TensorList、Queue、core/tile/data copy |
| Kernel entry/body | `common/inc/op_kernel/load_store_utils.h` | typed load/store helpers | FP16/BF16 cast、FP32 direct、RINT/mask |
| Kernel entry/body | `common/inc/op_kernel/platform.h` | `GetVRegSize` | VReg 字节宽度 producer |
| Kernel entry/body | `foreach/foreach_div_scalar_list/op_kernel/arch35/foreach_div_scalar_list_regbase.h` | `ForeachDivScalarListRegbase::Compute` | scalar、VReg loop、Div |
| Kernel entry/body | `foreach/foreach_div_scalar_list/op_kernel/arch35/foreach_div_scalar_list.cpp` | entry/Impl | key router、POD、pipe lifetime |
| 构建接线 | `foreach/foreach_div_scalar_list/op_kernel/CMakeLists.txt` | ascend950 `add_kernel_sources` | arch35 source和编译选项 |
| 构建接线 | `foreach/foreach_div_scalar_list/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn ...)` | ACLNN Host生成接线 |
| 构建接线 | `foreach/foreach_div_scalar_list/op_host/config/ascend950/foreach_div_scalar_list_binary.json` | `op_list` | 三种 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段描述50 Tensor/80 core切片 | u32元素tile、数组顺序/单位、空域-1 | `ForeachSoloTilingDataRegbase` |
| 调用顺序 | priority selector→POD→单entry→CRTP Process | 950必须使用key100xx和arch35 body | registry、entry |
| workspace/内存 | 32 B保留但Device不用；仅双in/out Queue | depth2、E按Host分母和block对齐 | `PostTiling`、Base::Init |
| 同步 | Queue生命周期；无显式event/atomic | Alloc/EnQue/DeQue/Free顺序 | CRTP base methods |
| 精度与尾块 | FP16/BF16经FP32/RINT；FP32直通；float mask | cast traits、VL/4、UpdateMask链 | `Compute`、load/store helpers |
| 失败边界 | capability、列表/dtype/shape检查、key0 | 不得给unknown key补造fallback | Host checks、router |