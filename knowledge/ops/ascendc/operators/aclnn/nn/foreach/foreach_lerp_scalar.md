---
type: Operator
title: "ForeachLerpScalar A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachLerpScalar 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_host/config/ascend910b/foreach_lerp_scalar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_host/config/ascend910_93/foreach_lerp_scalar_binary.json
    role: implementation
created_at: "2026-08-25T15:46:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachLerpScalar A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachLerpScalarGetWorkspaceSize` + `aclnnForeachLerpScalar`，公开 ACLNN 两段式接口；固定仓未出现其它公开版本。 |
| 计算语义 | 对列表位置 `i`：`y_i = x1_i + weight * (x2_i - x1_i)`。 |
| 输入输出 | `x1/x2/out` 为 ND TensorList，元素 dtype 为 FP16/FP32/BF16；`weight` 在 ACLNN 层接受 FLOAT32/DOUBLE，算子节点输入固定为 FP32 scalar。 |
| 执行类型 | 单 Kernel：`foreach_lerp_scalar`，`KERNEL_TYPE_AIV_ONLY`，AIV_VECTOR。 |
| Kernel 路线 | R-001 FP32 原位双公式；R-002 FP16/BF16 FP32 中间计算。 |
| 核心资源 | TensorList GM 描述符、3 个 depth=2 的 `TQue`、FP16/BF16 路线额外 1 个 FP32 `TQue`；节点预留 32 B workspace 但 Kernel 置空不使用。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→OpDef/自动生成 ACLNN 框架边界→Tiling registry/membase POD producer→AIV entry→完整 body→binary config/CMake。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设两个列表长度相等，各位置 Tensor 的 shape 相同。对 `i=0..n-1` 及其每个有效元素 `j`，输出为 `y_i[j] = x1_i[j] + weight * (x2_i[j] - x1_i[j])`。Device 按 `weight∈(-0.5,0.5)` 选择该式；否则用等价式 `x2_i[j] + (weight-1) * (x2_i[j]-x1_i[j])`。A2/A3 Host 允许 `out_i` 元素数大于 `x1_i`，Kernel 只写 `x1_i` 对应的元素域，多出的输出区域不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 每个 Tensor 0–8 维，ND | FP16/FP32/BF16 | 支持空 Tensor、输入可非连续并由 `AutoContiguous` 框架边界处理；列表内 dtype 一致。 |
| `x2` | 与 `x1` 同列表长度、对应 shape、ND | 与 `x1` 相同 | 支持空 Tensor、输入可非连续；Tiling 对应位置逐 shape 精确相等。 |
| `weight` | scalar | ACLNN FLOAT32/DOUBLE；节点 FP32 | 必需；框架将 ACLNN scalar lowering 到单元素 FP32 节点输入。 |
| `out` | 每项元素数不少于 `x1`，ND | 与 `x1` 相同 | 列表长度一致；不支持非连续输出；仅 `x1` 有效元素域被写入。 |

### 入口与版本

公开文档声明唯一两段式入口。第一段校验参数并构造 executor/返回 workspace，第二段在给定 stream 上执行。固定仓没有手写 ACLNN inner bridge，其参数 lowering、executor 壳和 `AutoContiguous` 属于允许的框架边界；算子拥有的 OpDef、selector、POD 和 Kernel 已闭合。README 与 ACLNN 文档均明确支持 Atlas A2/A3，OpDef 注册 `ascend910b` 与 `ascend910_93`，两个 binary config 均列出 FP16/FP32/BF16，因此没有公开支持冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 第一段先处理空指针、dtype/list 一致性、shape 与维数约束；文档列出的错误终点为 `ACLNN_ERR_PARAM_NULLPTR`、`ACLNN_ERR_PARAM_INVALID`、`ACLNN_ERR_INNER_TILING_ERROR`。OpDef 对 `x1/x2/weight/y` 标记 `AutoContiguous`，其中公开约束仍禁止非连续输出。进入算子 Tiling 后，`ForeachCommonTiling::Init` 扫描最多 50 个 `x1` Tensor，记录 storage 元素数、统一 dtype、`elementsPerBlock=32/sizeof(T)`；`CheckOpInfo` 再校验两个输入列表长度、对应 shape、列表内 dtype、输出 dtype/容量以及 scalar 单元素与维数。

### 参数、Workspace 与生命周期

`Tiling4ForeachLerpScalarTiling` 调用 `TilingRegistry::DoTilingImpl`。priority 10000 的 `ForeachLerpScalarMembaseTiling` 在非 regbase SoC 上可用，priority 30000 的 regbase 模板不适用。节点 workspace 固定为 32 B；entry 明确把 `userWS` 设为 `nullptr`，故此 32 B 不参与算法。ACLNN executor 总 workspace 还可能包含框架 `AutoContiguous` 的临时空间，其精确聚合公式在固定仓的生成代码边界外；owner 为调用方按第一段返回值申请，生命周期覆盖第二段异步提交，释放前需由调用方完成 stream 同步。

### 启动图与失败传播

单节点图：`aclnnForeachLerpScalar` executor 在同一 stream 上启动 `foreach_lerp_scalar(x1,x2,weight,y,workspace,tiling)`，`blockDim=needCoreNum`，POD 为 `ForeachCommonTilingData`。节点没有跨 Kernel 边；TensorList 描述符指向每项 GM 数据，输出直接落到各 `y_i`。Tiling 任一校验、平台/POD buffer、workspace 指针获取失败都返回 `GRAPH_FAILED`，不产生 launch；entry 对未知 key 没有 body 分支，因此必须由 Host dtype/OpDef 保证只生成 1/2/4。

## Route 与参数设计

### 分核与资源计算

令 `N=sum_i numel(x1_i)`，`s=sizeof(T)`，`E=32/s` 元素。`needCoreNum=min(max(ceil(N/E),1),AIV核数,50)`。Host 把以 32 B block 计的总数据近似均分：前 `blockCount % needCoreNum` 个核多一个 block，并通过 `tensorStart/EndList` 与首尾 offset 跨 Tensor 切片。

令 `P=tilingData.GetDataSize()`，`U=平台 UB 字节数`。对本算子先取 `total=U-P-128`；FP16/BF16 再令 `total=floor(total/10)`；`I=floor_align(total/6,64)`（FP16/BF16）或 `I=floor_align(total/6,32)`（FP32），`I` 的单位是字节并写入 `inputsTensorUbSize`。FP32 tile 容量为 `I/4` 元素；FP16/BF16 每个输入/输出 Queue buffer 分配 `9I` 字节，tile 容量为 `9I/2` 元素，FP32 中间区每半区容量 `I/4` 元素。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原位双公式 | priority 10000 membase；key 2；`ForeachLerpScalarND<float>` | 非 regbase SoC 且 `x1/x2/out=FP32`、节点 scalar=FP32、校验成功 | 参数失败终止；平台不符尝试 priority 30000。 |
| 2 | R-002 FP16/BF16 FP32 中间计算 | priority 10000 membase；key 1/4；`ForeachLerpScalarND<half/bfloat16_t>` | 非 regbase SoC 且 `x1/x2/out=FP16` 或 BF16；BF16 entry 还要求编译 guard 成立 | 参数失败终止；未知/未编译 key 无 body。 |

```text
TilingRegistry 按 priority 升序
  -> 10000 ForeachLerpScalarMembaseTiling::IsCapable = !IsRegbaseSocVersion
     -> true: Init(opCode=19, TYPE_SCALAR) -> RunBigScalarKernelTiling
        -> dtype FP16/FP32/BF16 -> key 1/2/4 -> R-002/R-001/R-002
     -> false: GRAPH_PARAM_INVALID，继续 30000 regbase
Kernel router: key 1 -> half；key 2 -> float；guard 内 key 4 -> bfloat16_t
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原位双公式 | 2 / membase | A2/A3；FP32；列表/shape/dtype/output/scalar 校验成功 | 单 AIV Kernel | `ForeachLerpScalarND<float>` + `InnerComputer<float>` | x1/x2/y Queue 均 depth 2；计算结果原位留在 x1 或 x2 Queue；显式 V→MTE3、MTE3→MTE2 event | 尾 tile 用 `DataCopyPad`；无其它 fallback。 |
| R-002 FP16/BF16 FP32 中间计算 | 1/4 / membase | A2/A3；FP16/BF16；同上；BF16 编译 guard 成立 | 单 AIV Kernel | `ForeachLerpScalarND<T>` + `InnerComputer<T>` | 三个 depth 2 Queue + 单个 FP32 Queue；分段 cast/FP32 Axpy/CAST_RINT；相同显式 event | 尾 tile 用 `DataCopyPad`；guard 不成立时 key 4 无 body。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 selector | `!IsRegbaseSocVersion(context_)` | classic body 使用 1/2/4 key 与 classic Vector API | 两 Route | 返回 `GRAPH_PARAM_INVALID`，registry 尝试下一模板 | `foreach_tiling_def.h::MEMBASE_TILING::IsCapable` |
| BF16 编译 guard | `__CCE_AICORE__ >= 220 && NPU_ARCH != 3003/3113` | 只在该 guard 内实例化 BF16 entry/body | R-002 key 4 | key 4 无执行分支 | `op_kernel/foreach_lerp_scalar.cpp::foreach_lerp_scalar` |
| UB 预算 | `U>P+128` 且除法/对齐后 `I>0` | `maxDataCount` 和 `maxCastDataCount` 不能为 0；Host 代码未显式拒绝 0 | 两 Route | 固定仓未提供安全 fallback | `ForeachCommonTiling::DivideUbMemory6` |
| 公式分支 | `-0.5 < weight < 0.5` | 选择 x1 锚点并原位写 x1；否则选 x2 锚点并用 `weight-1` | 两 Route | 同 Route 内另一计算分支 | `InnerComputer::ComputePerCast` / `InnerComputer<float>::Compute` |
| tail | `dataCount % maxDataCount != 0` | 非完整 tile 搬运字节数不一定 32 B 对齐 | 两 Route | 使用 `DataCopyPad`，无失败 | `SingleTensorProcess` / `CopyIn` / `ComputeAndCopyOut` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 第 1 字段；宏展开后的 `sizeof/alignof/offsetof` 在固定仓不可见 | 字节 | `DivideUbMemory6`→`FillTilingData` | `ParseTilingData`，决定 Queue bytes/tile | 必须非零并满足 UB 预算。 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 第 2 字段；精确 offset 不可见 | 元素 | `Init` 取 storage shape size | `Process` 计算各 Tensor 末元素 | 最多存 50 项。 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 第 3 字段；精确 offset 不可见 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | 仅 `[0,blockDim)` 有效。 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 第 4 字段；精确 offset 不可见 | Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData` | inclusive。 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 第 5 字段；精确 offset 不可见 | 元素 | `AssignDataToEachCore` | `Process` 首 Tensor GM base | inclusive。 |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 第 6 字段；精确 offset 不可见 | 元素 | `AssignDataToEachCore` | `Process` 尾 Tensor有效长度 | inclusive；空域可形成 `-1` 末偏移。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 2 / membase | `foreach_tiling_func.cpp::Tiling4ForeachLerpScalarTiling`；`foreach_tiling_class.h::RunBigScalarKernelTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_lerp_scalar.cpp::foreach_lerp_scalar` | `ForeachLerpScalarND<float>`；`InnerComputer<float>` | `op_kernel/CMakeLists.txt::add_kernel_sources`；`ascend910b/...binary.json` |
| R-002 | key 1/4 / membase | 同上 | 同上 | 同上，含 BF16 guard | `ForeachLerpScalarND<half/bfloat16_t>`；`InnerComputer<T>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 标记 `KERNEL_TYPE_AIV_ONLY`，解析 POD 后按 key 实例化一个 body，忽略传入 workspace。每核从 TensorList 的地址表中解析自己的首尾 Tensor 切片；每个 Tensor 切片再按 `maxDataCount` 分 tile，执行 `CopyIn→ComputeAndCopyOut`。完整 tile 使用按元素数的 `DataCopy`，尾 tile 使用 `blockCount=1, blockLen=dataCount*sizeof(T), stride=0` 的 `DataCopyPad`。没有跨 Kernel workspace、atomic 或跨核归约。

### Route R-001：FP32 原位双公式

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host selector | `foreach_utils/op_host/foreach_tiling_def.h` | `ForeachLerpScalarMembaseTiling::IsCapable` | `!IsRegbaseSocVersion(context_)` | 选择 membase。 |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2。 |
| Router | `foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp` | `foreach_lerp_scalar` | `TILING_KEY_IS(2)` | `ForeachLerpScalarND<float>`。 |

#### Device 数据流

##### 函数级调用链

```text
Tiling4ForeachLerpScalarTiling -> registry -> ForeachLerpScalarMembaseTiling
  -> ForeachCommonTiling::RunBigScalarKernelTiling
foreach_lerp_scalar(key=2)
  -> ForeachLerpScalarND<float>::Init
  -> Process -> SingleTensorProcess
    -> CopyIn -> InnerComputer<float>::Compute -> ComputeAndCopyOut -> GM y
```

##### 源码等价伪码

```text
for tensor i in [tensorStart, tensorEnd]:
  begin = (i==tensorStart ? tensorStartOffset : 0)
  end   = (i==tensorEnd ? tensorEndOffset : numel[i]-1)
  bind x1/x2/y GM at tensor_ptr[i] + begin
  for tile t over end-begin+1, tile_cap=I/4:
    copy x1,x2 GM[t*tile_cap : +count] -> Queue local
    if -0.5 < weight < 0.5:
      x2Local = x2Local - x1Local
      x1Local = x1Local + weight*x2Local
      resultLocal = x1Local
    else:
      x1Local = x2Local - x1Local
      x2Local = x2Local + (weight-1)*x1Local
      resultLocal = x2Local
    wait V->MTE3; copy resultLocal -> y GM; wait MTE3->MTE2; free local
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x1_i/x2_i/y_i` | descriptor pointer + `cursorStart + tile*I/4` | `dataCount` 元素 | FP32/ND | caller | `CopyIn`/`CopyOut` | caller；覆盖整个 API 执行 | tail 按实际字节。 |
| `x1Queue` | Queue slot | depth 2 × `I` B | FP32 | `CopyIn` | Compute/CopyOut | `TPipe`；tile 内 | 结果可原位留在 x1。 |
| `x2Queue` | Queue slot | depth 2 × `I` B | FP32 | `CopyIn` | Compute/CopyOut | 同上 | 结果可原位留在 x2。 |
| `yQueue` | Queue slot | depth 2 × `I` B | FP32 | 未写入 | 未读取 | `TPipe`；Kernel lifetime | 源码实际分配但 body 未使用，不可从字节账删除。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  x1Q["x1Queue · depth=2 · 2I B"]:1
  x2Q["x2Queue · depth=2 · 2I B"]:1
  yQ["yQueue · depth=2 · 2I B（未使用）"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2I + 2I + 2I = 6I
I = floor_align((U - POD_bytes - 128) / 6, 32)
UB_total <= U - POD_bytes - 128
dynamic_ub = N/A
workspace_total = 32 B 节点预留（Kernel 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 Queue slot | CopyIn；小权重 Compute 原位写结果 | CopyOut | slot Free 后下一 tile CopyIn | V→MTE3、MTE3→MTE2 event 后 Free | 小权重分支先释放 x2，复制 x1。 |
| x2 Queue slot | CopyIn；大权重 Compute 原位写结果 | CopyOut | slot Free 后下一 tile CopyIn | 同上 | 大权重分支先释放 x1，复制 x2。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | `EnQue/DeQue` | MTE2 | Vector | x1/x2 slot | FP32 原样。 |
| Vector pipeline | `PipeBarrier<PIPE_V>` | `Sub` | `Axpy`/后续 | 原位 LocalTensor | 严格 `(-0.5,0.5)`；边界 ±0.5 走 x2 公式。 |
| Compute→CopyOut | `SetFlag/WaitFlag<V_MTE3>` | Vector | MTE3 | 结果 local | 防止未完成计算即写 GM。 |
| CopyOut→复用 | `SetFlag/WaitFlag<MTE3_MTE2>` | MTE3 | 下一次 MTE2/Free | Queue slot | 尾 tile `DataCopyPad` 只提交 `count*sizeof(float)`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling`、`DivideUbMemory6` | key/core/POD/I/workspace 生产。 |
| Kernel entry/router | `foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp` | `foreach_lerp_scalar`, key 2 | 单 AIV entry 到 float body。 |
| Device body/helper | `foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h` | `ForeachLerpScalarND<float>`、`InnerComputer<float>::Compute` | 分 Tensor/tile、原位 Sub/Axpy、event、tail。 |

### Route R-002：FP16/BF16 FP32 中间计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host selector | `foreach_utils/op_host/foreach_tiling_def.h` | membase `IsCapable` | 非 regbase | membase。 |
| Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | FP16→1，BF16→4 | key 1/4。 |
| Router | `foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp` | `foreach_lerp_scalar` | key 1；guard 内 key 4 | `ForeachLerpScalarND<half/bfloat16_t>`。 |

#### Device 数据流

##### 函数级调用链

```text
foreach_lerp_scalar(key=1/4)
  -> ForeachLerpScalarND<T>::Init
  -> Process -> SingleTensorProcess -> CopyIn
  -> InnerComputer<T>::Compute
     -> ComputePerCast: Cast x1/x2 -> FP32 -> Sub -> Axpy -> Cast_RINT into x1/x2 local
  -> ComputeAndCopyOut -> GM y
```

##### 源码等价伪码

```text
tile_cap = 9I / sizeof(T); cast_cap = I / 4
for each assigned tensor slice and tile:
  DataCopy/DataCopyPad x1,x2 -> T local
  for cast chunk c of at most cast_cap elements:
    f0 = cast_fp32(x1[c]); f1 = cast_fp32(x2[c]); delta = f1-f0
    if -0.5 < weight < 0.5:
      f0 = f0 + weight*delta; cast_rint f0 -> x1Local[c]; result=x1Local
    else:
      f1 = cast_fp32(x2[c]); f1 = f1 + (weight-1)*delta
      cast_rint f1 -> x2Local[c]; result=x2Local
  event V->MTE3; copy valid result bytes to GM y; event MTE3->MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y Queue | Queue slot | 每 Queue depth 2 × `9I` B | FP16/BF16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | x1 或 x2 原位承载结果；yQueue 未使用。 |
| `float32Queue` | `[0,cast_cap)` 与 `[cast_cap,2*cast_cap)` | depth 1 × `2I` B | FP32 | Cast/Sub/Axpy | ComputePerCast | Init 时 Alloc+EnQue，Process 全程持有 | 两半区分别承载 x1/复用结果与 x2/delta。 |
| GM | descriptor + slice/tile offset | 当前 `dataCount` | T/ND | caller/CopyOut | CopyIn/caller | caller | tail 实际字节。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  x1Q["x1Queue · depth=2 · 18I B"]:1
  x2Q["x2Queue · depth=2 · 18I B"]:1
  yQ["yQueue · depth=2 · 18I B（未使用）"]:1
  f32Q["float32Queue · depth=1 · 2I B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 18I + 18I + 18I + 2I = 56I
I = floor_align(((U - POD_bytes - 128) / 10) / 6, 64)
56I <= U - POD_bytes - 128（Host /10、/6 提供保守余量）
dynamic_ub = N/A
workspace_total = 32 B 节点预留（Kernel 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 前半区 | x1 Cast；小权重 Axpy；大权重重新 Cast x2 | Axpy/Cast_RINT | 下一 cast chunk | 每步 `PipeBarrier<PIPE_V>` | chunk 不超过 `I/4` FP32 元素。 |
| FP32 后半区 | x2 Cast，随后 Sub 为 delta | Axpy | 下一 cast chunk | `PipeBarrier<PIPE_V>` | 与前半区物理不重叠。 |
| x1/x2 Queue slot | CopyIn，选中分支 Cast_RINT 原位写结果 | CopyOut | event 后 Free | V→MTE3、MTE3→MTE2 | 只复制与公式分支一致的 Queue。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast/compute | `PipeBarrier<PIPE_V>` | Cast/Sub/Axpy | 下一 Vector op | FP32 两半区 | FP16/BF16 升为 FP32，结果 `CAST_RINT` 回 T。 |
| Queue | `EnQue/DeQue/FreeTensor` | MTE2/Vector | Vector/MTE3 | 输入 Queue slot | depth 2。 |
| 尾搬运 | `DataCopyPad` | GM/UB | UB/GM | 有效 `dataCount` | `blockLen=dataCount*sizeof(T)`，padding 关闭。 |
| 结果提交 | V→MTE3、MTE3→MTE2 event | Vector/MTE3 | MTE3/下一 MTE2 | 原位结果 slot | 防提前读取/覆盖。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory6`、`FillTilingData` | FP16/BF16 /10、/6、64 B 对齐与 POD。 |
| Kernel entry/router | `foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp` | key 1/4 分支及 BF16 guard | dtype 实例映射。 |
| Device body/helper | `foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h` | `InnerComputer<T>::ComputePerCast` | FP32 chunk、双公式、CAST_RINT、原位结果。 |

## 算子特定优化

### Trick T-001：按 weight 区间选择锚点并原位提交结果

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：`weightVal < 0.5f && weightVal > -0.5f` 选 x1；否则选 x2。
- Golden 来源：[`foreach_lerp_scalar.h:99-110`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h#L99-L110)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (weightVal < FLOAT_NUM_POS && weightVal > FLOAT_NUM_NEG) {
    Sub(x2Local, x2Local, x1Local, dataCount);
    PipeBarrier<PIPE_V>();
    Axpy(x1Local, x2Local, weightVal, dataCount);
    PipeBarrier<PIPE_V>();
} else {
    Sub(x1Local, x2Local, x1Local, dataCount);
    PipeBarrier<PIPE_V>();
    weightVal = weightVal - FLOAT_NUM_ONE;
    PipeBarrier<PIPE_V>();
    Axpy(x2Local, x1Local, weightVal, dataCount);
    PipeBarrier<PIPE_V>();
}
```

#### 机制与设计原因

两条代数等价路径都只做一次 `Sub` 和一次 `Axpy`，但把最终值直接留在 x1 或 x2 的输入 Queue slot；随后 CopyOut 选择同一 slot，因此没有再物化独立结果。阈值同时决定 Axpy scalar 使用 `weight` 还是 `weight-1`，必须与 CopyOut 的 Queue 选择保持一致。

#### 迁移边界

- 适用：两个同 shape Tensor 的逐元素 lerp，且允许输入 local 被覆盖。
- 前提：分支谓词、Axpy operand 顺序、结果 Queue 选择三者一致；输出 GM 与输入 GM 的 owner 关系不能破坏读后写。
- 失效：若后续还需要未修改的两个 local，必须增加独立输出 local 或调整生命周期。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码可直接证明若公式分支与 CopyOut Queue 选择不一致会输出 delta/旧输入而非 lerp 结果。

### Trick T-002：低精度输入分块升到 FP32 后再一次性回写

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：模板 `T=half/bfloat16_t`，每个 cast chunk 不超过 `maxCastDataCount=I/sizeof(float)`。
- Golden 来源：[`foreach_lerp_scalar.h:68-83`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h#L68-L83)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
Cast(float32Tensor[maxCastDataCount], x2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
Sub(float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount], float32Tensor, dataCount);
PipeBarrier<PIPE_V>();
if (localWeight < FLOAT_NUM_POS && localWeight > FLOAT_NUM_NEG) {
    Axpy(float32Tensor, float32Tensor[maxCastDataCount], localWeight, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(x1Local[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
} else {
    localWeight = localWeight - FLOAT_NUM_ONE;
    PipeBarrier<PIPE_V>();
    Cast(float32Tensor, x2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Axpy(float32Tensor, float32Tensor[maxCastDataCount], localWeight, dataCount);
```

#### 机制与设计原因

FP16/BF16 输入在 FP32 临时区内完成减法与 Axpy，最后以 `CAST_RINT` 一次降精度；`float32Queue` 的两个相邻半区复用为 x1/结果与 x2/delta，使精度路径不要求为整个大 tile 同时物化更多 FP32 张量。关键 offset `maxCastDataCount` 来自 Host 写入的字节量 `I`。

#### 迁移边界

- 适用：低精度输入、FP32 中间运算、tile 可能大于 FP32 临时区的逐元素算子。
- 前提：临时区至少 `2I` B，chunk 上界为 `I/4` 元素，各 Vector op 之间保留 PIPE_V barrier。
- 失效：FP32 临时区不足或移除 barrier 时不能沿用；应缩小 chunk 或重新设计 Queue。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 原位双公式 | R-002 FP16/BF16 FP32 中间计算 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | membase priority 10000，key 2 | 同 priority，key 1/4 | dtype 决定 key/body。 |
| 模板与 dtype | `ND<float>` | `ND<half/bfloat16_t>` | 低精度需要 cast specialization。 |
| 循环与分核 | 32 B block 均分；tile `I/4` | 同分核；tile `9I/2`，内部 cast chunk `I/4` | 低精度 Queue 扩大、FP32 临时分块。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3×depth2 Queue | 同上 + 1×depth1 FP32 Queue | 无 dynamic UB/通信。 |
| 内存总占用与复用 | `6I`，x1/x2 原位 | `56I`，x1/x2 原位 + FP32 双半区 | Host 公式分别保守约束。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad), Sub, Axpy | 另含 Cast(CAST_NONE/RINT) | 均为 AIV Vector。 |
| 同步、精度与 tail | FP32；显式 event；tail pad | FP32 中间、一次降精度；相同 event/tail | dtype 精度需求。 |
| fallback/失败边界 | 未知 key 无 body | BF16 guard 不成立无 body | 无 SIMT/NDDMA fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开支持，见独立 `_950` 卡 |
| Host/Launcher | `ascend910b`，membase key 1/2/4 | `ascend910_93`，同 membase | regbase key 10001/10002/10004 |
| Kernel/资源 | classic AIV，共用 body | classic AIV，共用 body | arch35 MicroAPI + 独立输出 Queue |

## 限制与失败边界

- `x1/x2` 列表长度、对应 shape 与 dtype 必须一致；Tensor 维数不超过 8；输出 dtype 一致且容量不少于输入，公开接口禁止非连续输出。
- classic POD 数组上限为 50，`Init` 只扫描前 50 项；固定仓未显示对更长 A2/A3 列表的显式报错，因此不得据此宣称支持更长列表。
- Host UB 算式未显式检查减法下溢或对齐后 `I==0`；正常平台资源由平台注册保证，异常资源没有源码 fallback。
- ACLNN 自动生成 bridge、executor 总 workspace 聚合与 scalar FLOAT64→节点 FP32 lowering 的具体代码在固定仓不可见，作为框架边界保留。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md` | 两段式原型、产品表、参数/错误码 | ACLNN 公开性与约束。 |
| 入口声明/约束 | `foreach/foreach_lerp_scalar/README.md` | 产品支持与公式 | A2/A3/950 支持和语义。 |
| 入口实现 | `foreach/foreach_lerp_scalar/op_host/foreach_lerp_scalar_def.cpp` | `ForeachLerpScalar`、`OP_ADD` | dtype/format/AutoContiguous/平台。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachLerpScalarTiling`、membase 注册 | selector 入口与 priority。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key 1/2/4。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | POD 定义、`MEMBASE_TILING` | ABI 字段和平台能力谓词。 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 升序、fallback/失败。 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | `IsCapable` 与返回语义。 |
| Kernel entry/body | `foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp` | `foreach_lerp_scalar` | AIV 标记和 key router。 |
| Kernel entry/body | `foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h` | `ForeachLerpScalarND`、`InnerComputer` | 完整 Device 实现。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_host/CMakeLists.txt` | `add_modules_sources` | Host 源与 `foreach_utils` 依赖。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 编译选择。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_host/config/ascend910b/foreach_lerp_scalar_binary.json` | `ForeachLerpScalar` op_list | dtype binary 接线；A3 对应 config 同构。 |
| 构建接线 | `foreach/foreach_lerp_scalar/op_host/config/ascend910_93/foreach_lerp_scalar_binary.json` | `ForeachLerpScalar` op_list | A3 三 dtype binary 接线。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段共同描述 UB 与跨 Tensor 核切片 | 字段顺序、位宽、数组 50、inclusive offset | `ForeachCommonTilingData` |
| 调用顺序 | 每 Tensor、每 tile `CopyIn→Compute→CopyOut` | 公式分支与结果 Queue 一致 | `Process` / `SingleTensorProcess` |
| workspace/内存 | 32 B 预留但 Kernel 不用；Queue 字节账可复用 | depth=2 与 FP32 Queue 不能漏算，未用 yQueue 仍占 UB | `Init` / `RunBigScalarKernelTiling` |
| 同步 | Queue + V/MTE3/MTE2 event | 结果复制和 slot 复用前等待 | `ComputeAndCopyOut` |
| 精度与尾块 | 低精度升 FP32、CAST_RINT；尾 tile pad copy | strict weight 阈值、chunk offset、实际 `blockLen` | `ComputePerCast` / `CopyIn` |
| 失败边界 | selector 不适用继续；校验失败立即终止 | 不得把未知 key 或 BF16 guard 当 fallback | `TilingRegistry::DoTilingImpl` / entry |