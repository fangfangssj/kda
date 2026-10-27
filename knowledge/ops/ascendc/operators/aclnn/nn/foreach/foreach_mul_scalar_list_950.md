---
type: Operator
title: "ForeachMulScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachMulScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_kernel/arch35/foreach_mul_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_kernel/arch35/foreach_mul_scalar_list_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary_scalar_list.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/load_store_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_host/config/ascend950/foreach_mul_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:58:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMulScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMulScalarListGetWorkspaceSize` + `aclnnForeachMulScalarList`，公开两段式 ACLNN；自动生成 bridge/executor 壳为框架边界 |
| 计算语义 | `out[i][j] = x[i][j] * scalars[i]` |
| 输入输出 | 最多 50 个 ND Tensor；`x/out` 为 FP16/FP32/INT32/BF16；内部 Kernel scalar ABI 分别为 FP32/FP32/INT64/FP32 |
| 执行类型 | 单 Kernel：arch35 `foreach_mul_scalar_list`，AIV_VECTOR（MicroAPI/Vector register）；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 FP16；R-002 FP32；R-003 INT32；R-004 BF16，key 10001–10004 |
| 核心资源 | `ListTensorDesc`、GM ScalarList、双缓冲 VECIN/VECOUT Queue、Vector RegTensor；32 B 预留 workspace 不参与计算 |
| 来源闭包 | `source_closed`：公开 ACLNN→OpDef/Infer→Registry selector→regbase Tiling/POD→arch35 entry/body/helper→binary/CMake 已闭合；自动 bridge 中公开 DOUBLE scalar 到内部 FP32 ABI 的归一化细节为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 TensorList 位置 `i∈[0,n)`，用 ScalarList 同位置标量广播乘 `x_i` 的所有 storage 元素。输出有效 shape/dtype 由公共 Infer 逐项复制输入，Kernel 按展平元素序遍历，并仅写各 Tensor 的有效元素区间。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 1–50 个 Tensor；每个 0–8 维；ND | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；支持空 Tensor；输入可非连续，OpDef `AutoContiguous()` |
| `scalars` | ScalarList，长度等于 `x` | 公开 ACLNN：浮点 `x` 可 FP32/DOUBLE，INT32 `x` 用 INT64 | 内部 OpDef/binary/Kernel 只闭合 FP32 或 INT64 ABI；DOUBLE 的生成 bridge 归一化细节固定仓不可见 |
| `out` | 与 `x` 等长，ND，0–8 维 | 与 `x` 一致 | 最多 50；输出连续；Tiling 要求每项容量不小于输入，Infer 路径设为完全相同 shape |
| `workspaceSize/executor` | Host 输出 | — | 第一段返回；Tiling 固定 32 B |
| `workspace/stream` | Device 地址/stream | — | 第二段启动单 Kernel；workspace 参数传入但 body 未读取 |

### 入口与版本

公开产品表明确支持 Ascend 950PR/950DT，OpDef 添加 `ascend950` regbase config，binary config 有四个 dtype 实例，CMake 只为 `ascend950` 选择 arch35 源。未发现带版本后缀的更新入口。公开 ACLNN 文档允许浮点 scalar 为 DOUBLE，而内部 OpDef、binary config 和 Kernel scalar type 均为 FP32；自动生成 ACLNN bridge 不在固定仓，故只记录“存在公开到内部 ABI 的框架边界”，不推测 DOUBLE 的舍入、溢出或报错细节。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段 ACLNN 执行公开的空指针、dtype、列表长度、shape/dim 校验；自动生成实现固定仓不可见。
2. OpDef 对 `x/scalars` 启用连续化，注册内部 dtype 组合 FP16/FP32、FP32/FP32、BF16/FP32、INT32/INT64；输出 dtype 与输入一一对应。
3. 公共 Infer 要求输入/输出列表等长，逐项复制 shape/dtype。
4. `Tiling4ForeachMulScalarListTiling` 遍历优先级升序：10000 membase 先被尝试，但 950 `IsRegbaseSocVersion` 为真，所以返回 `GRAPH_PARAM_INVALID`；30000 的 `ForeachRegbaseTilingUnaryScalarList2` 命中。
5. regbase Host 检查 Tensor 数 1–50、列表内 dtype、非负 shape、维数≤8、ScalarList dtype/长度、输出列表数/dtype/容量；任一失败阻断 launch。

### 参数、Workspace 与生命周期

`GetPlatformInfo` 取得 AIV 核数和 UB 大小；Tiling 生成 `ForeachSoloTilingDataRegbase`、key、blockDim，并写 32 B workspace。caller 拥有输入/输出/workspace，executor 持有 launch 描述；它们至少存活到 stream 完成。Kernel entry 把 workspace 传到 `ForeachMulScalarListRegbase::Init`，Base `Init` 未读取它，因此没有中间 slice、跨 Kernel owner 或同步边。

### 启动图与失败传播

```text
aclnnForeachMulScalarListGetWorkspaceSize (自动 bridge)
  -> OpDef/Infer
  -> Tiling4ForeachMulScalarListTiling
     -> priority 10000 Membase: IsCapable=false
     -> priority 30000 ForeachRegbaseTilingUnaryScalarList2
  -> ForeachSoloTilingDataRegbase + key 10001..10004 + blockDim + 32 B workspace
aclnnForeachMulScalarList(..., stream)
  -> K0 arch35 foreach_mul_scalar_list(tensor1, scalars, outputs, workspace, tiling)
     -> ForeachMulScalarListImpl<T,ScalarT>
     -> ForeachMulScalarListRegbase::Process
```

只有 K0 一个 Device 节点，无节点间 workspace/GM 边。Host Tiling `GRAPH_FAILED` 或 launch 框架错误阻断执行；未知 key 时 entry 只销毁 `TPipe`，不产生输出。

## Route 与参数设计

### 分核与资源计算

令 `s=sizeof(T)`，`B=4096/s` 元素/4 KiB 分核 block，`N=sum_i numel(x_i)`。`blockDim=min(AIVCoreNum,80,ceil(N/B))`；若 `N=0` 则强制 1 核并写 `[start=end=0,startOffset=0,endOffset=-1]`。非空时把 `ceil(N/B)` 个 block 均分，前余数核多一个 block，核区间用 Tensor 索引与 inclusive offset 描述。

UB tile 元素数：`Q=ubSize/(4*s)`，再令 `inputsTensorUbSize=floor32(Q*s)/s`。`inputsTensorUbSize` 保存到 `uint32_t` POD；Kernel 为 inQueue/outQueue 各 `depth=2`、单 slot `inputsTensorUbSize*s` B，总账 `2*2*inputsTensorUbSize*s<=ubSize`。Host 未提供额外保留或显式 uint32 溢出 guard，实际 UB 尺度使该字段受平台资源约束。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | 10001；`ForeachMulScalarListImpl<half,float>` | 950 regbase；`x/out=FP16`、内部 scalar ABI FP32 | 下一 key/Host 失败 |
| 2 | R-002 FP32 | 10002；`<float,float>` | 950；FP32/FP32 | 下一 key/失败 |
| 3 | R-003 INT32 | 10003；`<int32_t,int64_t>` | 950；INT32/INT64 | 下一 key/失败 |
| 4 | R-004 BF16 | 10004；`<bfloat16_t,float>` | 950；BF16/FP32 | unknown key 无 body |

```text
Registry priority 10000 Membase -> IsRegbase=true -> GRAPH_PARAM_INVALID
Registry priority 30000 Regbase -> IsRegbase=true
  -> DT_FLOAT16:10001 | DT_FLOAT:10002 | DT_INT32:10003 | DT_BF16:10004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 10001 | 950、合法列表/shape、FP16/FP32 | 单 K0 AIV_VECTOR | `ForeachMulScalarListRegbase<half,float,...>` | 双缓冲 in/out；RegTensor<float>；Queue 同步 | `UpdateMask<float>` + DataCopyPad 精确字节数 |
| R-002 | 10002 | 同上，FP32/FP32 | 单 K0 | `<float,float>` | 双缓冲；FP32 register load/store | 同上 |
| R-003 | 10003 | 同上，INT32/INT64 | 单 K0 | `<int32_t,int64_t>` | 双缓冲；RegTensor<int32_t>；scalar 转 float | 同上；float scalar 转换是精度边界 |
| R-004 | 10004 | 同上，BF16/FP32 | 单 K0 | `<bfloat16_t,float>` | 双缓冲；BF16 unpack/cast FP32、RINT pack | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| regbase selector | `IsRegbaseSocVersion(context_)`，NPU arch DAV_3510/5102 | 选择 arch35 POD/key/body | 全部 | membase/最终失败 | `tiling_util.h::IsRegbaseSocVersion`、两模板 `IsCapable` |
| Tensor 数 | `1<=totalTensorCount<=50` | POD `tensorDataCountList[50]`、公开 950 上限 | 全部 | Tiling 失败 | `GetShapeAttrsInfo` |
| 核数 | `min(AIV,80,ceil(N/B))` | per-core 数组 80 项 | 全部 | 截断 blockDim | `DoOpTiling` |
| 4 KiB 分核 | `B=4096/s` | 较粗粒度分配，减少核边界数量 | 全部 | 无其它 route | `SINGLE_CORE_PROCESS_DATA` |
| UB/32 B 对齐 | `floor32((ub/(4s))*s)/s` | depth 2×输入输出四份 tile | 全部 | tile 变小；无 fallback | `UnaryScalarList2::DoOpTiling` |
| repeat 位宽 | `repeatTimes=ceil(dataCount/(VL/4))` 后窄为 `uint16_t` | MicroAPI loop counter | 全部 | 固定仓无显式超界 fallback | `ForeachMulScalarListRegbase::Compute` |
| 空域 | `N==0 -> blockDim=1,endOffset=-1` | entry 合法启动、循环 quotient=0 | 全部 | no-op | `DoOpTiling/SingleTensorProcess` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | `uint32_t`/32 | 首字段；宏展开的 sizeof/alignof/offsetof 固定仓不可见 | 元素 | `UnaryScalarList2::DoOpTiling` | Base `ParseTilingData`、`InitBuffer` | Host int64 setter 隐式收敛到 POD；无显式 overflow guard |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 次字段；精确 offset 不可见 | 元素 | `GetShapeAttrsInfo/DoOpTiling` | `Process` | 非负，最多 50 |
| 同上 | `tensorStartList` | `uint16_t[80]` | 第三字段 | Tensor 索引 | `AssignDataToEachCore` | `[blockIdx]` | inclusive |
| 同上 | `tensorEndList` | `uint16_t[80]` | 第四字段 | Tensor 索引 | 同上 | 同上 | inclusive |
| 同上 | `tensorStartOffsetList` | `int64_t[80]` | 第五字段 | 元素 | 同上 | 同上 | inclusive |
| 同上 | `tensorEndOffsetList` | `int64_t[80]` | 第六字段 | 元素 | 同上 | 同上 | inclusive；空域 -1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10001 | `ForeachRegbaseTilingUnaryScalarList2` | `ForeachSoloTilingDataRegbase` | arch35 `foreach_mul_scalar_list` | regbase Base + `<half,float>::Compute` + load/store helper | ascend950 binary JSON + arch35 CMake |
| R-002 | 10002 | 同上 | 同上 | 同上 | `<float,float>` | 同上 |
| R-003 | 10003 | 同上 | 同上 | 同上 | `<int32_t,int64_t>` | 同上 |
| R-004 | 10004 | 同上 | 同上 | 同上 | `<bfloat16_t,float>` + cast helper | 同上 |

## Kernel 实现

### 公共执行流程

entry 栈上创建 `TPipe`，按 key 解析 POD 并实例化模板，结束时 `Destroy()`。Base 使用 `ListTensorDesc` 解析每个输入/输出数据指针，按本核首尾 Tensor/offset 遍历；每 Tensor 再按 `inputsTensorUbSize` 元素切 tile。CopyIn/Out 都使用单 block、`blockLen=dataCount*sizeof(T)`、零 stride 的 `DataCopyPad`。Compute 从 inQueue DeQue、outQueue Alloc，在 Vector scope 中按 `VL_SIZE/sizeof(float)` 个 32-bit lane 循环，用 `UpdateMask<float>(sreg)` 消耗有效元素，原地 register Muls 后 Store。Queue depth 2 实现 copy/compute/copyout 的双缓冲生命周期；无 event、atomic、跨核同步或 workspace。

### Route R-001：FP16→FP32 Register Muls→FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT16` | 10001 |
| Router | arch35 entry | key 10001 | `<half,float>` | R-001 |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar_list -> ForeachMulScalarListImpl<half,float>
  -> ForeachMulScalarListRegbase::Init -> Base::Init
  -> Base::Process -> CopyIn -> pred.Compute -> CopyOut
  -> LoadOneTensorForDtypeT<half> -> Cast<float> -> Muls<float> -> StoreOneTensorForDtypeT<half>
```

##### 源码等价伪码

```text
for tensor i in core interval:
  scalar=float(scalars[i]); for tile n:
    GM half[n] -> inQ
    for r=0..ceil(n/(VL/4))-1: mask=next valid FP32 lanes; unpack+cast half->float; reg*=scalar; RINT cast+pack float->half
    outQ half[n] -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `ListTensorDesc[i]+coreOffset+k*max` | `n` FP16 | ND-flat | caller/CopyOut | CopyIn/caller | caller | exact byte DataCopyPad |
| inQueue | Queue slots | `2*E*2` B | FP16 | CopyIn | Compute | TPipe | depth 2 |
| outQueue | Queue slots | `2*E*2` B | FP16 | Compute | CopyOut | TPipe | depth 2 |
| RegTensor | register | one VReg FP32 lane group | FP32 | Load/cast | Muls/store | Vector scope | each repeat overwritten |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · depth=2 · 2×E×2 bytes"]:1
  outQ["outQueue · depth=2 · 2×E×2 bytes"]:1
```

##### 字节账与复用顺序

```text
E=inputsTensorUbSize；UB_total=4E*sizeof(half)<=ubSize
dynamic_ub=N/A；workspace_total=32 B reserved/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQ slot | CopyIn | Compute | 后两 tile 周期 | DeQue+FreeTensor | consumer 完成 |
| outQ slot | Compute | CopyOut | 后两 tile 周期 | DeQue+FreeTensor | GM copy 提交 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/Compute | Queue | CopyIn/Compute | Compute/CopyOut | Queue slot | exact `n` bytes |
| Vector | `UpdateMask<float>` | sreg | Load/Muls/Store | valid lanes | FP16 unpack→FP32 Muls→RINT FP16；无效 lane zeroing/not stored |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `DoOpTiling/GetTilingKey/PostTiling` | key/core/POD/workspace |
| Kernel entry/router | arch35 `foreach_mul_scalar_list.cpp` | key 10001 | half/float mapping |
| Device body/helper | `foreach_mul_scalar_list_regbase.h`、`load_store_utils.h` | `Compute`、load/store half | mask、FP32 Muls、RINT |

### Route R-002：FP32 Register Muls

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `GetTilingKey` | `DT_FLOAT` | switch | 10002 |
| Router | arch35 entry | key 10002 | `<float,float>` | R-002 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Regbase<float,float>::Process -> CopyIn -> DataCopy RegTensor<float> -> Muls -> DataCopy(mask) -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 同一 core/tile/repeat；省去 dtype cast，FP32 register 原地乘并按 mask 存回。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/inQ/outQ | 同 R-001 | Queue 各 `2*E*4` B | FP32 | CopyIn/Compute | Compute/CopyOut | caller/TPipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · depth=2 · 2×E×4 bytes"]:1
  outQ["outQueue · depth=2 · 2×E×4 bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=4E*sizeof(float)<=ubSize；workspace=32 B reserved/0 consumed。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQ/outQ | CopyIn/Compute | Compute/CopyOut | depth-2 周期 | FreeTensor | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | Queue + mask | GM/RegTensor | RegTensor/GM | valid lanes | FP32 Muls；tail mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | key 10002 | FP32 route |
| Kernel entry/router | arch35 entry | `<float,float>` | mapping |
| Device body/helper | `ForeachMulScalarListRegbase::Compute` | floating branch | direct FP32 register load/store |

### Route R-003：INT32 Register Muls

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `CheckScalarList` | `{DT_INT32,DT_INT64}` | dtype combination | valid |
| Tiling/Router | `GetTilingKey` / arch35 entry | INT32/key10003 | `<int32_t,int64_t>` | R-003 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Regbase<int32_t,int64_t>::Compute
  -> scalarCalcType=float; scaleVal=float(int64 scalar)
  -> DataCopy RegTensor<int32_t> -> Muls(inReg,inReg,scaleVal,mask) -> masked DataCopy out
```

##### 源码等价伪码

```text
scale = float(scalars_int64[i]); for each masked 32-bit lane group: reg_i32=load; reg_i32=Muls(reg_i32,scale); store valid lanes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| scalar GM | `scalars+i` | 1 INT64 | INT64 | caller | Compute | caller | 每 Tensor GetValue；转 float |
| inQ/outQ | 同 R-001 | 各 `2*E*4` B | INT32 | CopyIn/Compute | Muls/CopyOut | TPipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · depth=2 · 2×E×4 bytes"]:1
  outQ["outQueue · depth=2 · 2×E×4 bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=4E*sizeof(int32_t)<=ubSize；workspace=32 B reserved/0 consumed。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slot | CopyIn/Compute | Compute/CopyOut | depth-2 周期 | FreeTensor | consumer 已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar/Vector | C++ cast + masked Muls | INT64 scalar | INT32 register result | value/lanes | INT64→FP32 可能丢整数精度；Muls 对 INT32 与 float scalar 的精确转换/溢出规则属于 primitive 边界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `CheckScalarList/GetTilingKey` | INT32/INT64 validation |
| Kernel entry/router | arch35 entry | key 10003 | int32/int64 mapping |
| Device body/helper | `foreach_mul_scalar_list_regbase.h` | `scalarCalcType/scaleVal`、integer branch | scalar 转 float 后 INT32 Muls |

### Route R-004：BF16→FP32 Register Muls→BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `GetTilingKey` | `DT_BF16` | switch | 10004 |
| Router | arch35 entry | key 10004 | `<bfloat16_t,float>` | R-004 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Regbase<bfloat16_t,float>::Compute
  -> LoadOneTensorForDtypeT<bfloat16_t> (unpack + Cast FP32)
  -> Muls FP32 -> StoreOneTensorForDtypeT<bfloat16_t> (RINT + pack)
```

##### 源码等价伪码

```text
for each masked repeat: unpack BF16 lanes; zeroing cast to FP32; multiply scalar FP32; RINT cast+pack valid BF16 lanes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQ/outQ | 同 R-001 | 各 `2*E*2` B | BF16 | CopyIn/Compute | Compute/CopyOut | TPipe | depth 2 |
| FP32 RegTensor | repeat offset `r*(VL/4)` | one VReg | FP32 | BF16 unpack/cast | Muls/RINT store | Vector scope | 每 repeat 原地复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · depth=2 · 2×E×2 bytes"]:1
  outQ["outQueue · depth=2 · 2×E×2 bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=4E*sizeof(bfloat16_t)<=ubSize；FP32 中间仅在寄存器；workspace=32 B reserved/0 consumed。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 register | load/cast | Muls/store | 下一 repeat | 同一 Vector scope 顺序 | store 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Vector | mask + register program order | BF16 load/cast | FP32 Muls/BF16 store | valid lanes | FP32 计算，`CAST_RINT` 降回，ZEROING cast；tail mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | key10004/UB formula | BF16 route |
| Kernel entry/router | arch35 entry | `<bfloat16_t,float>` | mapping |
| Device body/helper | regbase header + `load_store_utils.h` | `Compute`、BF16 load/store | FP32 register + RINT pack |

## 算子特定优化

### Trick T-001：用 Masked MicroAPI 融合尾块与升精度计算

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-004；R-003 复用同一 repeat/mask 框架但走整数 register 分支
- 触发条件：任意 tile，特别是 `dataCount` 不是 `VL_SIZE/sizeof(float)` 整数倍时
- Golden 来源：[`foreach_mul_scalar_list_regbase.h:53-68`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar_list/op_kernel/arch35/foreach_mul_scalar_list_regbase.h#L53-L68)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
uint32_t sreg = (uint32_t)dataCount;

__VEC_SCOPE__
{
    MaskReg maskReg;
    if constexpr (
        IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, float>::value) {
        RegTensor<float> inRegToFloat;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            maskReg = UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
            Muls(inRegToFloat, inRegToFloat, scaleVal, maskReg);
            ops::StoreOneTensorForDtypeT<T>(outUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
        }
        }
```

#### 机制与设计原因

Host 只需给 tile 容量，Device 统一按 FP32 lane 宽度迭代；`UpdateMask<float>(sreg)` 每次消耗剩余有效 lane，使满 repeat 与尾 repeat 共享一套 load/Muls/store。FP16/BF16 的 unpack/cast、FP32 原地 Muls、RINT pack 都停留在寄存器，不在 UB 物化 FP32 临时 Tensor；这同时降低 UB 需求并避免独立 tail 分支。

#### 迁移边界

- 适用：可按 32-bit lane 统一处理、支持 masked load/store 的逐元素 broadcast 运算。
- 前提：`dataCount` 可安全窄化到 `uint32_t`，repeat 数可放入 `uint16_t`，load/store helper 与目标 dtype 的 pack/cast 语义匹配。
- 失效：tile 超出上述位宽、需要跨 lane 状态或不支持 masked store 时，必须缩小 Host tile 或采用不同实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | 10001 | 10002 | 10003 | 10004 | regbase dtype key |
| 模板与 dtype | half/float | float/float | int32/int64 | bf16/float | 内部 scalar ABI |
| 循环与分核 | 4 KiB block；FP32 lane repeats | 同 | 同 | 同 | 公共 Base/MicroAPI |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth2 in/out + FP32 reg | 同 | INT32 reg | depth2 + FP32 reg | 无 TBuf/dynamic UB/通信 |
| 内存总占用与复用 | 4E×2 B | 4E×4 B | 4E×4 B | 4E×2 B | `E` 随 s 反向变化，总体≤UB |
| 搬运/Vector/SIMT/NDDMA/通信 API | Pad + unpack/cast/Muls/pack | Pad + load/Muls/store | Pad + int load/Muls/store | Pad + unpack/cast/Muls/pack | 无其它执行族 |
| 同步、精度与 tail | FP32 乘、RINT FP16 | FP32 | INT64→float 边界 | FP32 乘、RINT BF16 | masked tail |
| fallback/失败边界 | Host 失败/unknown key | 同 | 同 | 同 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | membase key1–4、32 B block、最多50核 | 同 A2 | regbase key10001–10004、4 KiB block、最多80核 |
| Kernel/资源 | 旧式 Queue API；BF16 使用 UB FP32 临时区 | 同 A2 | arch35 MicroAPI；FP16/BF16 的 FP32 中间仅在 register；ListTensorDesc |

## 限制与失败边界

- 950 最多 50 个 Tensor；ScalarList 与 output 必须等长；shape 维数≤8、维值非负；列表内 dtype 必须一致；不满足时 Tiling 失败。
- 公开 ACLNN 文档允许浮点 scalar 为 DOUBLE，但固定仓内部仅公开 FP32 ABI。自动 bridge 的 DOUBLE→FP32 归一化/校验与舍入规则不可见，复用时不得自行假设；这是明确的框架边界。
- R-003 源码把 INT64 scalar 转成 `float` 后传给 INT32 `Muls`；大整数可能在 FP32 转换时丢失精度，primitive 最终转换/溢出语义固定仓不可见。
- `repeatTimes` 窄为 `uint16_t`、`dataCount` 窄为 `uint32_t`，Host 仅以 UB tile 间接约束，未写显式位宽 guard。
- workspace 固定返回 32 B，但 body 不使用；不得把它当作可用的跨 Kernel 中间区。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_mul_scalar_list/docs/aclnnForeachMulScalarList.md` | ACLNN 两段式原型/产品表 | 950 API、DOUBLE、shape/dtype/error |
| 入口实现 | `foreach/foreach_mul_scalar_list/op_host/foreach_mul_scalar_list_def.cpp` | `ForeachMulScalarList` | ascend950 schema/config |
| Infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | common Infer + registration | output shape/dtype |
| Host selector | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachMulScalarListTiling` | Registry 调用和模板注册 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | Base/UnaryScalarList2 methods | 校验、分核、UB、key、workspace |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.h` | 类定义 | selector 与状态字段 |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachSoloTilingDataRegbase` | ABI 字段 |
| Host selector | `common/inc/op_host/tiling_util.h` | `IsRegbaseSocVersion` | regbase NPU arch 判定 |
| Host selector | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 升序、fallthrough 和失败终点 |
| Kernel entry/body | `foreach/foreach_mul_scalar_list/op_kernel/arch35/foreach_mul_scalar_list.cpp` | entry/Impl | key router、TPipe |
| Kernel body | `foreach/foreach_mul_scalar_list/op_kernel/arch35/foreach_mul_scalar_list_regbase.h` | `ForeachMulScalarListRegbase::Compute` | scalar、repeat、mask、Muls |
| Kernel body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary_scalar_list.h` | Base `Init/Process/Copy*` | Queue、ListTensorDesc、GM 搬运 |
| Kernel helper | `common/inc/op_kernel/load_store_utils.h` | `Load/StoreOneTensorForDtypeT` | cast/pack/round |
| Kernel helper | `common/inc/op_kernel/platform.h` | `GetVRegSize` | Vector register width |
| 构建接线 | `foreach/foreach_mul_scalar_list/op_host/config/ascend950/foreach_mul_scalar_list_binary.json` | 四个 binary 实例 | 950 dtype wiring |
| 构建接线 | `foreach/foreach_mul_scalar_list/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 source/option |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段、50 Tensor/80 core | 首字段是元素数 `uint32_t`，首尾 offset inclusive | `ForeachSoloTilingDataRegbase` |
| 调用顺序 | descriptor→每 Tensor→tile→CopyIn/Compute/CopyOut | scalar 必须与 Tensor index 同步 | Base `Process`、pred `Compute` |
| workspace/内存 | 双缓冲输入输出、无 workspace 中间 | `UB_total=4E*s`、depth=2 必须计入 | `InitBuffer`、Host UB 公式 |
| 同步 | Queue 生命周期 + Vector scope program order | slot 在 FreeTensor 前不得覆盖 | Base Copy/Compute methods |
| 精度与尾块 | masked FP32 lane 管线 | FP16/BF16 RINT；INT64 scalar→float；tail mask | load/store traits、`Compute` |
| 失败边界 | 两级 selector、Host 校验、unknown key no-op | 不得把 A2 key 或非法 dtype 下发到 arch35 router | Registry/`GetTilingKey`/entry |