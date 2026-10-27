---
type: Operator
title: "ForeachDivScalarList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachDivScalarList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/docs/aclnnForeachDivScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/foreach_div_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary_level_zero_api.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_kernel/foreach_div_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/config/ascend910b/foreach_div_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_div_scalar_list/op_host/config/ascend910_93/foreach_div_scalar_list_binary.json
    role: implementation
created_at: "2026-08-25T15:32:31Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachDivScalarList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachDivScalarListGetWorkspaceSize` + `aclnnForeachDivScalarList`，公开两段式 ACLNN |
| 计算语义 | 对列表位置 `i` 的每个有效元素 `j` 计算 `out[i][j] = x[i][j] / scalars[i]` |
| 输入输出 | `x/out` 为 1–50 个 ND Tensor，rank 0–8，统一 dtype `FLOAT16/FLOAT32/BFLOAT16`；`scalars` 为等长 `FLOAT32` 列表 |
| 执行类型 | 单 Kernel：`foreach_div_scalar_list`，`AIV_VECTOR` |
| Kernel 路线 | R-001 FP16 Level-0；R-002 FP32 Level-0；R-003 BF16→FP32→BF16 Level-0 |
| 核心资源 | TensorList 描述符 GM、标量 GM、双缓冲 `VECIN/VECOUT`、32 B 标量 TBuf；BF16 另有单缓冲 FP32 Queue |
| 来源闭包 | `source_closed`：公开 ACLNN 约束、OpDef、membase Host/Tiling、POD、entry、全部 dtype body/helper 与 A2/A3 构建接线闭合；自动生成 ACLNN bridge/launcher 作为框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x=[x_0,...,x_{n-1}]`、`scalars=[s_0,...,s_{n-1}]`，则对每个 `i∈[0,n)` 与 `x_i` 的存储元素域计算 `y_i=x_i/s_i`。Host 按每个 Tensor 的 storage shape size 计数，输出的有效写入长度等于对应输入元素数；即使公开约束允许 `out_i` 容量更大，超出 `x_i` 元素数的区域也不由本 Kernel 写入。除数为零时的浮点结果没有算子自定义分支，沿硬件 `Div` 语义产生。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND、rank 0–8 | 列表内统一 `FLOAT16/FLOAT32/BFLOAT16` | 列表长度公开约束为 1–50；支持空 Tensor；OpDef `AutoContiguous()`，公开 ACLNN 支持非连续输入 |
| `scalars` | ScalarList，元素数等于 `x` 长度 | `FLOAT32` | 第 `i` 项广播到整个 `x_i`；OpDef 将其注册为 `ScalarList()` 并 `AutoContiguous()` |
| `out` | TensorList；每项 ND、rank 0–8 | 与 `x` 相同 | 列表长度与 `x` 相同；每项 storage size 不小于对应 `x`；不支持非连续输出 |

### 入口与版本

固定仓公开唯一版本为无版本后缀的 `aclnnForeachDivScalarList`。第一段返回 workspace 大小和 executor，第二段携 workspace、executor、stream 执行。README/API 文档明确支持 Atlas A2、A3；OpDef 将 `ascend910b` 与 `ascend910_93` 接到 membase 配置，Kernel CMake 同时编译这两个 compute unit。固定仓未包含自动生成的 `aclnnop/aclnn_foreach_div_scalar_list.h` 实体、ACLNN bridge 与 executor launch 函数；它们由 `ACLNNTYPE aclnn`/OpDef 生成，作为框架边界，不补造源码符号。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口先拒绝 `x/scalars/out` 空指针与不支持 dtype，并要求 `x/out` dtype 一致。进入算子 Tiling 后，`ForeachCommonTiling::Init` 枚举至多 50 个动态 `x`，记录 storage element count、统一 dtype 和总元素数；`CheckOpInfo` 依次校验动态输入列表实例数、每项 rank≤8、`scalars` storage size 等于 Tensor 数、输出 dtype、输出 storage size 以及列表内 dtype 一致性。`x` 与 `scalars` 的非连续输入由 OpDef 的 `AutoContiguous()` 交给框架处理；对应复制节点和其 workspace 细节在固定仓不可见。任何检查失败均返回 `GRAPH_FAILED`，由 ACLNN 映射为文档列出的参数或 inner-tiling 错误。

### 参数、Workspace 与生命周期

设 `N=Σ_i numel(x_i)`、元素字节 `S∈{2,4}`、32 B 内元素数 `B=32/S`、平台 AIV 核数 `Cplat`。Tiling 取 `C=min(ceil(N/B),Cplat,50)`，空总域强制 `C=1`，再按 32 B block 数将连续的“Tensor 列表拼接域”均分到各核，产生每核首尾 Tensor 与首尾元素 offset。算子 Tiling 固定写 `workspace[0]=32`；Kernel 将用户 workspace 替换成 `nullptr`，故这 32 B 只是框架保留量，不承载节点间数据。ACLNN executor、stream 与可能的 contiguous 临时区由框架持有到第二段执行结束，其精确总 workspace 组成固定仓不可见。

### 启动图与失败传播

单节点启动图为：框架生成 ACLNN bridge → `Tiling4ForeachDivScalarListTiling` → 优先级模板 → 单次启动 `foreach_div_scalar_list(x_desc, scalars, out_desc, workspace, tiling)`。节点 blockDim=`C`，POD=`ForeachCommonTilingData`，输入/输出 TensorList 描述符的 GM 数据指针由 Kernel 解引用；无 Kernel 间边、event 或 workspace slice。Tiling/launch 失败阻止节点提交，第二段在调用者给定 stream 上执行；具体生成 launcher 符号与 launch 返回包装固定仓不可见。

## Route 与参数设计

### 分核与资源计算

`ForeachCommonTiling` 先以 32 B block 对 `N` 向上取整。每核基础份额为 `floor(ceil(N/B)/C)*B` 元素，前 `ceil(N/B) mod C` 核再多一个 `B`；份额可以跨越多个 Tensor，并用 `tensorStart/EndList` 与 `tensorStart/EndOffsetList` 截断。设 POD 序列化字节数为 `H`、平台 UB 为 `Uplat`，本算子 `opCode=31`：先令 `R=Uplat-H-32`；FP32 取 `U=floor32((R/4))` 字节，FP16/BF16 先令 `R'=floor(R/10)` 再取 `U=floor64(R'/4)` 字节。`U` 是 A2/A3 POD 的 `inputsTensorUbSize`（字节），同时决定 tile 上限。所有除法均为整数向下取整；固定源码没有 `U=0` 的显式 Host guard。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | 公共 membase selector | priority `10000`，`ForeachDivScalarListMembaseTiling` | 注册表按较小数值优先；`!IsRegbaseSocVersion(context)`，A2/A3 命中 | regbase 平台返回 `GRAPH_PARAM_INVALID` 后尝试 priority 30000 |
| 2 | R-001 FP16 | key `1`，`<half,half,Div,2,1>` | membase 且 `x.dtype=DT_FLOAT16` | dtype 不支持则 Tiling/OpDef 失败 |
| 3 | R-002 FP32 | key `2`，`<float,float,Div,2,1>` | membase 且 `x.dtype=DT_FLOAT` | dtype 不支持则失败 |
| 4 | R-003 BF16 | key `4`，`<bfloat16_t,float,Div,2,1>` | membase 且 `x.dtype=DT_BF16`；Kernel guard `__CCE_AICORE__>=220` 且非 NPU arch 3003/3113 | guard 未编译时无匹配 body；A2/A3 活动 binary config 包含 BF16 |

```text
priority 10000 ForeachDivScalarListMembaseTiling
  if !IsRegbaseSocVersion:
    validate -> split -> key=dtype
      key 1 -> R-001
      key 2 -> R-002
      key 4 -> R-003
  else GRAPH_PARAM_INVALID -> priority 30000 regbase template
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | `1` | A2/A3；统一 FP16；列表/shape/scalar/output 校验通过 | 单 AIV `foreach_div_scalar_list` | `ForeachOneScalarListBinaryLevelZeroApi<half,half,Div,2,1>` → generic `InnerComputer` | 2×in + 2×out，32 B half 标量块；Queue 与显式 V→MTE3/MTE3→MTE2 event | GM 尾块用 `DataCopyPad`；Vector 尾 repeat 用实际 mask 长度 |
| R-002 FP32 | `2` | A2/A3；统一 FP32；其余同上 | 同上 | `<float,float,Div,2,1>` → generic `InnerComputer` | 同上，标量块为 float；每 repeat 64 元素 | 同上 |
| R-003 BF16 | `4` | A2/A3；统一 BF16；编译 guard 成立 | 同上 | `<bfloat16_t,float,Div,2,1>` → BF16 specialization | 2×9U in + 2×9U out + 1×U FP32 Queue + 32 B 标量块；显式 `PIPE_V` barrier | 外层 tile、FP32 cast 子块、repeat 三层尾部均保护 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 模板优先级 | priority 10000 先于 30000；越小越优先 | membase/regbase 共享公开入口但 POD/key 不同 | 全部 | 当前模板 `IsCapable=false` 才 fallthrough | `TilingRegistry::DoTilingImpl`、`MEMBASE_TILING::IsCapable` |
| 列表上限 | 公开 `1≤n≤50`；POD 数组固定 50 | 防止数组和 per-core 元数据越界 | 全部 | ACLNN/tiling 失败 | `MAX_TENSOR_CONT/MAX_CORE_CONT` |
| 32 B 分核 | `B=32/S`，`C≤50` | 核份额以 DMA block 粒度构造 | 全部 | 无其他 Route | `GetNeedCoreNum`、`AssignDataToEachCore` |
| UB tile | `U=floorAlign((Uplat-H-32)/(4·attenuation),32/64)` | 双缓冲、标量块与 BF16 cast 临时量必须落入 UB | 全部 | 固定源码无小 UB fallback | `DivideUbMemory9` |
| repeat 上限 | full repeats 分批，每批 `≤255` | Level-0 binary API repeatTimes 为 8-bit 有效范围 | 全部 | 再循环一批；最后单 repeat | `InnerComputer::Compute/ComputePerCast` |
| BF16 编译 guard | `__CCE_AICORE__>=220 && arch!=3003/3113` | BF16 specialization/指令支持 | R-003 | 未编译时 router 无 BF16 分支 | `foreach_div_scalar_list` |
| 空域 | `N=0 -> C=1`，每核 `endOffset=-1` | `dataCount=0` 使 tile 循环为零 | 全部 | 成功但不写输出 | `GetNeedCoreNum`、`Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t/64` | 第 1 字段；`sizeof/alignof/offsetof` 的宏展开固定仓不可见 | 字节 | `FillTilingData: set_inputsTensorUbSize(U)` | `KernelForeachBase::ParseTilingData`；计算 Queue 容量与 tile | 必须非零；Host 未显式检查 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第 2 字段；精确 offset 固定仓不可见 | 元素/每 Tensor | `Init` 取 storage shape size | `Process` 计算 Tensor 有效域 | 空 Tensor 为 0 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 第 3 字段；精确 offset 固定仓不可见 | Tensor 索引/每核 | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | `blockIdx<C≤50` |
| 同上 | `tensorEndList` | `uint16_t[50]` | 第 4 字段；精确 offset 固定仓不可见 | Tensor 索引/每核 | 同上 | 同上 | 空域首核为 0 |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 第 5 字段；精确 offset 固定仓不可见 | 元素/每核 | 同上 | 同上 | 首 Tensor 内偏移 |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 第 6 字段；精确 offset 固定仓不可见 | 元素/每核，含尾元素 | 同上 | 同上 | 空域为 `-1` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 1 | `foreach_tiling_func.cpp:Tiling4ForeachDivScalarListTiling`；`foreach_tiling_class.h:RunBigScalarKernelTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_div_scalar_list.cpp:foreach_div_scalar_list` | `ForeachOneScalarListBinaryLevelZeroApi<half,half,...>`、`KernelForeachUnary`、generic `InnerComputer` | `op_kernel/CMakeLists.txt` + `ascend910b/...binary.json` |
| R-002 FP32 | key 2 | 同上 | 同上 | 同上 | `<float,float,...>`、generic `InnerComputer` | 同上 |
| R-003 BF16 | key 4 | 同上 | 同上 | 同上 | `<bfloat16_t,float,...>`、BF16 `InnerComputer` specialization | 同上 |

## Kernel 实现

### 公共执行流程

entry 声明 AIV-only，解析 POD 后按 key 构造一个 CRTP 实例。每核把 POD 的首尾 Tensor/offset 映射到 TensorList GM 指针；对每个相交 Tensor，先读取该 Tensor 的 `FLOAT32` scalar 并在 32 B TBuf 广播，再按 `maxDataCount` 分 tile：GM→`dataQueue`，Vector `Div`，结果→`outQueue`，最后写回 GM。Queue 的 `EnQue/DeQue/FreeTensor` 形成 MTE2→V 生命周期；派生 `CopyOut` 另以 V→MTE3、MTE3→MTE2 event 防止结果未完成或下一次输入覆盖。workspace 不参与 Device 数据流。

### Route R-001：FP16 Level-0

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_tiling_def.h` | `ForeachDivScalarListMembaseTiling::IsCapable` | `!IsRegbaseSocVersion(context_)` | 使用 membase |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_div_scalar_list.cpp` | `foreach_div_scalar_list` | `TILING_KEY_IS(1)` | `<half,half,Div,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
framework launch foreach_div_scalar_list(x, scalars, y, workspace, tiling)
  -> ForeachOneScalarListBinaryLevelZeroApi<half,half,Div,2,1>::Init
  -> KernelForeachUnary::Process
    -> ProcessPlusInLoop -> scalar GM load + Duplicate
    -> SingleTensorProcess
      -> CopyIn -> Compute -> InnerComputer<half,half>::Compute -> Div
      -> derived CopyOut
```

##### 源码等价伪码

```text
for tensor i in [tensorStart, tensorEnd]:
  begin/end = POD offsets clipped to tensor i
  scalarVal = half(scalars[i]); scalarUB[0:16] = scalarVal
  bind input/output GM at tensor i + begin
  for tile k over count=(end-begin+1), capacity=U/2 elements:
    DataCopy or DataCopyPad GM[k] -> inQ
    for batches of <=255 repeats, 128 half lanes/repeat:
      Div(out, in, scalarUB, mask, repeat, {1,1,0,8,8,0})
    V_MTE3 wait; DataCopy or DataCopyPad outQ -> GM[k]; MTE3_MTE2 wait
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `GetTensorAddr(i,x)+begin+k·U/2` | `tileCount≤U/2` 元素 | FP16/ND | ACLNN input | `CopyIn` | caller；执行期有效 | 尾块 `DataCopyPad`, no pad |
| `dataQueue` | Queue slot | `2·U` 总字节 | FP16 | MTE2 | Vector Div | Kernel pipe；FreeTensor 后复用 | depth 2 |
| `scalarOneBlockBuf` | 0 | 32 B | FP16 broadcast | `Duplicate` | Div src1 | Kernel 全程 | 单 TBuf，Tensor 间覆盖 |
| `outQueue` | Queue slot | `2·U` 总字节 | FP16 | Div | MTE3 | Kernel pipe；写回后释放 | depth 2 |
| output GM | `GetTensorAddr(i,y)+begin+k·U/2` | 有效 tile 元素 | FP16/ND | `CopyOut` | caller | caller | 超出 x size 不写 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 2U B"]:2
  scalar["scalar TBuf · 32 B"]:1
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2U + 32 + 2U = 4U + 32 B
U = floor64(floor((Uplat-H-32)/10)/4)
UB_total <= Uplat-H（Host 的 /10 为 FP16 路线留下保守余量）
dynamic_ub = N/A
workspace_total = 32 B reserved；Kernel userWS=nullptr
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQ slot | `CopyIn` | `Compute` | 后续 tile | `EnQue→DeQue→FreeTensor` | depth=2 |
| outQ slot | `Compute` | `CopyOut` | 后续 tile | `EnQue→DeQue` + V_MTE3 + `FreeTensor` | MTE3 完成后覆盖 |
| scalar TBuf | `ProcessPlusInLoop(i)` | i 的所有 tile | 下一 Tensor | 同一核顺序循环 | 当前 Tensor 全 tile 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | TQue | MTE2 | `DeQue` | input slot | GM 尾 tile 按有效字节搬运 |
| Compute | Level-0 `Div` | in/scalar UB | outQ | Vector lanes | FP32 scalar先 cast FP16；FP16 除法；repeat tail 使用实际 mask |
| Compute→CopyOut | `V_MTE3` event | Vector | MTE3 | output slot | 只提交 `dataCount` |
| CopyOut→下次 CopyIn | `MTE3_MTE2` event | MTE3 | MTE2 | 复用 Queue | 防止提前覆盖 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `RunBigScalarKernelTiling/DivideUbMemory9` | key、blockDim、U、32 B workspace |
| Kernel entry/router | `foreach_div_scalar_list.cpp` | `foreach_div_scalar_list` key 1 | AIV-only 与 FP16 实例 |
| Device body/helper | `foreach_one_scalar_list_binary_level_zero_api.h` | `ProcessPlusInLoop/Compute/CopyOut` | scalar broadcast、Queue 生命周期与 events |
| Device body/helper | `foreach_one_scalar_binary_level_zero_api.h` | generic `InnerComputer::Compute` | 255-repeat 分批与 tail mask |

### Route R-002：FP32 Level-0

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_div_scalar_list.cpp` | `foreach_div_scalar_list` | `TILING_KEY_IS(2)` | `<float,float,Div,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_div_scalar_list -> ForeachOneScalarListBinaryLevelZeroApi<float,float,Div,2,1>
  -> KernelForeachUnary::Process -> CopyIn -> generic InnerComputer::Compute -> Div -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 相同；scalarVal 保持 float，tile=U/4 元素，repeat=64 float lanes；
每个 full batch 最多 255 repeats，最后 remainder 以 mask=remaining elements 执行一次。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | TensorList pointer + core/tile offset | `≤U/4` 元素/tile | FP32/ND | caller/Kernel | Kernel/caller | stream 执行期 | tail 按有效字节 |
| dataQueue | slot | `2U` B | FP32 | CopyIn | Div | pipe | depth 2 |
| scalar TBuf | 0 | 32 B | FP32×8 | Duplicate | Div | pipe | Tensor 间覆盖 |
| outQueue | slot | `2U` B | FP32 | Div | CopyOut | pipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · 2U B"]:2
  scalar["scalar · 32 B"]:1
  outQ["outQueue · 2U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 4U + 32 B
U = floor32((Uplat-H-32)/4)
UB_total <= Uplat-H
dynamic_ub = N/A
workspace_total = 32 B reserved, unused by Kernel
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 双 Queue slots | CopyIn/Div | Div/CopyOut | 后续 tile | Queue + 两个 hard events | 同 R-001 |
| scalar TBuf | `Duplicate` | all Div repeats of current Tensor | 下一 Tensor | 核内顺序 | 当前 Tensor 已结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM↔UB | Queue/events | MTE2/Vector/MTE3 | DeQue/WaitFlag | slot | `DataCopyPad` 处理非整块尾部 |
| Div | FP32 Level-0 | input/scalar | output | 64 lane repeat | 无 cast；最多255 repeats/批，末尾 mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory9` | FP32 U 公式 |
| Kernel entry/router | `foreach_div_scalar_list.cpp` | key 2 | FP32 实例 |
| Device body/helper | `foreach_one_scalar_binary_level_zero_api.h` | generic `InnerComputer<float,float>` | 直接 FP32 Div 与 repeat 分批 |

### Route R-003：BF16→FP32→BF16 Level-0

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `foreach_div_scalar_list.cpp` | guarded key 4 branch | CCE≥220 且非 arch 3003/3113 | `<bfloat16_t,float,Div,2,1>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_div_scalar_list
  -> ForeachOneScalarListBinaryLevelZeroApi<bfloat16_t,float,Div,2,1>::Init/Process
    -> KernelForeachUnary::CopyIn
    -> InnerComputer<bfloat16_t,float>::Compute
      -> ComputePerCast -> Cast(BF16→FP32) -> Div(in-place FP32) -> Cast(FP32→BF16,RINT)
    -> derived CopyOut
```

##### 源码等价伪码

```text
U = POD bytes; copyCapacity = 9U/2 BF16 elements; castCapacity = U/4 float elements
for each GM tile up to copyCapacity:
  BF16 GM -> inQ
  for each cast chunk <= castCapacity:
    Cast BF16 chunk -> shared float32Queue
    divide FP32 in-place by broadcast scalar, in <=255-repeat batches
    Cast FP32 -> BF16 outQ using CAST_RINT
  outQ -> BF16 GM, writing only valid elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dataQueue | slot | `2·9U=18U` B | BF16 | CopyIn | Cast | pipe | depth 2 |
| float32Queue | 0 | `U` B | FP32 | Cast/Div in-place | Cast back | pipe，全 Process 只分配一次 | 单槽跨 cast chunk 复用 |
| scalar TBuf | 0 | 32 B | FP32×8 | Duplicate | Div | pipe | Tensor 间覆盖 |
| outQueue | slot | `18U` B | BF16 | Cast back | CopyOut | pipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["dataQueue · depth=2 · 18U B"]:2
  f32["float32Queue · depth=1 · U B"]:1
  scalar["scalar · 32 B"]:1
  outQ["outQueue · depth=2 · 18U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 18U + U + 32 + 18U = 37U + 32 B
U = floor64(floor((Uplat-H-32)/10)/4)
37U + 32 <= Uplat-H（整数对齐只减小 U）
dynamic_ub = N/A
workspace_total = 32 B reserved, unused by Kernel
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| float32Queue | BF16→FP32 Cast | Div、FP32→BF16 Cast | 下一 cast chunk | `PIPE_V` barriers；单槽在 Process 末释放 | 前一 cast back 完成 |
| input/output slots | MTE2/Vector | Vector/MTE3 | 后续 GM tile | Queue + events | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast→Div→Cast | `PipeBarrier<PIPE_V>` | 各 Vector 指令 | 下一 Vector 指令 | shared FP32 buffer | BF16 升 FP32；Div FP32；`CAST_RINT` 回 BF16 |
| cast tail | `castTimesRemainder` | outer tile | `ComputePerCast` | FP32 buffer | 只处理剩余元素 |
| repeat tail | mask length | Div | Cast back | Vector lanes | 无效 lane 不提交 |
| CopyOut | hard events + `DataCopyPad` | Vector | MTE3/MTE2 | Queue slots | 只写有效 BF16 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory9` | `/10` attenuation、64 B 对齐 |
| Kernel entry/router | `foreach_div_scalar_list.cpp` | guarded key 4 | BF16 实例与平台 guard |
| Device body/helper | `kernel_foreach_base.h` | `COPY_SPACE_MULTIPLE=9` | copy tile 与 cast tile 容量 |
| Device body/helper | `foreach_one_scalar_binary_level_zero_api.h` | BF16 `InnerComputer::ComputePerCast` | 升精度、in-place Div、RINT 回写 |

## 算子特定优化

### Trick T-001：标量单块广播复用与 repeat 批处理

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：进入每个 Tensor 的 `ProcessPlusInLoop`；随后该 Tensor 可包含任意数量 tile/repeat
- Golden 来源：[`foreach_one_scalar_list_binary_level_zero_api.h:73-77`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary_level_zero_api.h#L73-L77)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart)
    {
        scalarVal = P(inScalarGM.GetValue(index));
        Duplicate(scalarOneBlockLM, scalarVal, BYTES_PER_BLOCK / sizeof(P));
    }
```

#### 机制与设计原因

每个 Tensor 只在进入其循环时从标量 GM 取 `scalars[i]`，随后把它扩展成一个 32 B UB block。所有 tile 的 Level-0 `Div` 都以 scalar repeat stride `0` 和参数 `{1,1,0,8,8,0}` 重用这个 block；full repeat 又按最多 255 次分批，因此既避免逐元素 scalar GM 搬运，也满足 API repeat 上限。`P` 来自 Route：FP16 为 half，FP32/BF16 为 float。

#### 迁移边界

- 适用：一个 scalar 对一个 Tensor 全域广播、且同核顺序完成该 Tensor 所有 tile 的 foreach 算子。
- 前提：标量列表索引与 TensorList 索引一一对应；32 B local block 在该 Tensor 生命周期内不覆盖。
- 失效：scalar 随元素变化时必须改为 tensor 输入搬运；不同 dtype 必须保留相应 cast/精度 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key1 | key2 | key4+guard | dtype selector |
| 模板与 dtype | `<half,half>` | `<float,float>` | `<bf16,float>` | scalar/计算精度不同 |
| 循环与分核 | 32 B block；tile U/2 | tile U/4 | outer 9U/2 + inner U/4 | dtype 与 cast 临时量 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4U+32 | 4U+32 | 37U+32 | BF16 需要 FP32 Queue |
| 内存总占用与复用 | 双 Queue + scalar | 同左 | 双大 Queue + 单 FP32 Queue | Host attenuation 保证上界 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + half Div | DataCopyPad + float Div | Cast + float Div + Cast | 均为 AIV，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue/events；FP16 | Queue/events；FP32 | 加 `PIPE_V`；RINT | dtype specialization |
| fallback/失败边界 | 无 dtype fallback | 无 | guard 缺失即无 body | 未知 key 不执行 body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 支持；见独立 `_950` 卡 |
| Host/Launcher | `ascend910b` membase，共用 key 1/2/4 | `ascend910_93` membase，共用实现 | regbase priority 30000、key 10001/10002/10004 |
| Kernel/资源 | 通用 Level-0 body | 同 A2 | arch35 MicroAPI、VReg float pipeline |

## 限制与失败边界

- 公开列表长度为 1–50、rank≤8、dtype 仅 FP16/FP32/BF16，scalars 必须 FP32 且等长；输出连续、dtype 相同、容量不小于输入。Host POD 数组和 core 元数据上限均为 50。
- 固定仓未显式防护 `inputsTensorUbSize==0`；异常小 UB 配置会使 Device tile 除数失效，没有 fallback Route。
- unknown TilingKey 或 BF16 编译 guard 不成立时 entry 没有 `else` 错误分支，会直接结束而不写输出；正常 A2/A3 binary config 只发布 key 对应的三种 dtype。
- ACLNN 自动生成 bridge、输入 contiguous 复制节点、executor 内部 workspace 合并及 launch 包装固定仓不可见；算子自有 Tiling/Kernel 闭包已闭合。
- POD 宏定义来自外部 CANN 头，固定仓只能证明字段顺序/类型，不能证明精确 `sizeof/alignof/offsetof`。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_div_scalar_list/docs/aclnnForeachDivScalarList.md` | 两段式原型、产品表 | API、参数、错误边界、A2/A3 支持 |
| 入口实现 | `foreach/foreach_div_scalar_list/op_host/foreach_div_scalar_list_def.cpp` | `ForeachDivScalarList::ForeachDivScalarList` | dtype/layout、AutoContiguous、SoC config |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachDivScalarListTiling`、membase 注册 | selector、priority、workspace |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD producer |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD 字段与注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key 1/2/4 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 从小到大尝试 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | capability/fallthrough 与 Tiling 步序 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD consumer、TensorList ABI、BF16 容量 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、循环、CopyIn/Out |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary_level_zero_api.h` | `InnerComputer` | repeat、cast、precision、tail |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_list_binary_level_zero_api.h` | `ForeachOneScalarListBinaryLevelZeroApi` | scalar list、events、derived body |
| Kernel entry/body | `foreach/foreach_div_scalar_list/op_kernel/foreach_div_scalar_list.cpp` | `foreach_div_scalar_list` | key router、模板实例、AIV-only |
| 构建接线 | `foreach/foreach_div_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 compute units |
| 构建接线 | `foreach/foreach_div_scalar_list/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn ...)` | ACLNN Host 生成接线 |
| 构建接线 | `foreach/foreach_div_scalar_list/op_host/config/ascend910b/foreach_div_scalar_list_binary.json` | `op_list` | A2 三种 dtype binary |
| 构建接线 | `foreach/foreach_div_scalar_list/op_host/config/ascend910_93/foreach_div_scalar_list_binary.json` | `op_list` | A3 三种 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段描述每核跨 Tensor 切片 | 字段顺序、50 项数组、字节/元素单位 | `ForeachCommonTilingData` |
| 调用顺序 | scalar broadcast→逐 tile CopyIn→Div→CopyOut | Tensor 索引必须同步驱动 scalar 与 GM 描述符 | `KernelForeachUnary::Process` |
| workspace/内存 | Device 无 workspace；Queue 按 U 分配 | 32 B 保留量、depth=2、BF16 9×copy space | `GetWorkspaceSize`、`InitBuffer` |
| 同步 | Queue + V_MTE3 + MTE3_MTE2；BF16 加 PIPE_V | Free/覆盖不得早于等待点 | `CopyOut`、`ComputePerCast` |
| 精度与尾块 | FP16 direct；FP32 direct；BF16 经 FP32/RINT | scalar P、repeat mask、cast 顺序 | `InnerComputer` specializations |
| 失败边界 | 模板 capability、dtype key、公开校验 | 不得为 unknown key 自造 fallback | router 与 `CheckOpInfo` |