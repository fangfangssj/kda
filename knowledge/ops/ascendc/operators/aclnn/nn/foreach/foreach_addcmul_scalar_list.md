---
type: Operator
title: "ForeachAddcmulScalarList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachAddcmulScalarList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/docs/aclnnForeachAddcmulScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/foreach_addcmul_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/arch22/foreach_addcmul_scalar_list_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/foreach_addcmul_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_quaternary_implict_output.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_quaternary_implict_output.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcmul_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:30:28Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachAddcmulScalarList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddcmulScalarListGetWorkspaceSize` + `aclnnForeachAddcmulScalarList`，公开 ACLNN，两段式，无更高版本符号。 |
| 计算语义 | `y_i = x1_i + scalars[i] * x2_i * x3_i`，对 tensor list 每个位置逐元素计算。 |
| 输入输出 | `x1/x2/x3/y` 动态 tensor list，ND，FP16/FP32/INT32/BF16；`scalars.numel == tensorCount`；对应 tensor shape 相同。 |
| 执行类型 | 单 AIV_VECTOR Kernel；R-001/R-002/R-003 均无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP16/BF16 分段升 FP32；R-002 FP32 `Mul+Axpy`；R-003 INT32 `Mul+Muls+Add`。 |
| 核心资源 | GM tensor-list ABI；三路 depth=2 VECIN queues；低精度路线额外 depth=1 FP32 queue；operator workspace 32 B 但 Kernel 不读。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→生成式 ACLNN/AutoContiguous 框架边界→OpDef→arch22 common Tiling/POD→entry/router→公共 Vector body/helper→A2/A3 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

列表长度为 `n`。对 `i∈[0,n)` 和 `j∈[0,numel(x1_i))`，输出为 `x1_i[j] + scalars[i] × x2_i[j] × x3_i[j]`。FP16/BF16 路线把三路 tile 分段 cast 到 FP32，按 `Mul` 后 `Axpy` 计算，再以 `CAST_RINT` 回写；FP32 直接在原 dtype 上 `Mul` 后 `Axpy`；INT32 依次 `Mul`、将 float Host scalar 强转 `int32_t` 后 `Muls`、`Add`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 动态列表，各 tensor ND、0–8 维 | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；允许零元素 tensor。 |
| `x2` | 与 `x1` 等长，对应 tensor shape 相同 | 同 `x1` | 第一乘法输入。 |
| `x3` | 与 `x1` 等长，对应 tensor shape 相同 | 同 `x1` | 第二乘法输入。 |
| `scalars` | ND、0–8 维，元素数等于列表长度 | 同 `x1` | 每个 tensor 独立一个 scalar。 |
| `out/y` | 动态列表、ND；每项容量不少于对应 `x1` 元素数 | 同 `x1` | Kernel 写 `x1_i` 有效元素；公开接口不支持非连续输出。 |

### 入口与版本

公开文档只有无版本后缀的两段式 API。`GetWorkspaceSize` 返回 executor 和总 workspace，第二段在指定 stream 执行。OpDef 对全部输入输出声明 `AutoContiguous()`；自动生成的 ACLNN inner bridge、连续化/ViewCopy 与 executor 壳在固定仓不可见，按框架边界处理。公开产品表支持 A2 与 A3；OpDef 的 `ascend910b/ascend910_93`、arch22 Host Tiling 和 Kernel CMake 接线一致。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段校验五个 tensor 参数非空、dtype 属于四种且一致；生成框架完成输入连续化。
2. `Tiling4ForeachAddcmulScalarListTiling` 构造 `ForeachCommonTiling`，用 `FOREACH_POINTWISE_OP_CODE` 与 `TYPE_SCALARS_TENSOR` 初始化。
3. `Init` 最多读取 `MAX_TENSOR_CONT=50` 个 `x1` 动态 tensor，锁定 dtype、dtype 字节数、每 32 B 元素数，并收集 `tensorDataCountList` 与总元素数。
4. `CheckOpInfo` 按顺序检查三路列表长度相同、逐位置 shape 相同、维度≤8、`scalars.numel==tensorCount` 且维度≤8、输出 dtype/容量、三路列表所有 tensor dtype 一致；失败返回 `GRAPH_FAILED`。
5. `RunBigKernelTiling` 生产 key、分核边界、UB tile、POD、blockDim，并把 operator workspace 设为 32 B。第二段 executor 在 stream 上启动单 Kernel；entry 明确把 `userWS=nullptr`，不读取传入 workspace。

### 参数、Workspace 与生命周期

设 `S=sum_i numel(x1_i)`，元素字节数 `B`，`E=32/B`，平台 AIV 核数 `C`：

```text
needCoreNum = min(C, max(1, ceil(S/E)), 50)
blockCount = ceil(S/E)
basePerCore = floor(blockCount/needCoreNum) * E
前 blockCount%needCoreNum 个 core 再取 E 个元素
每 core 用 tensorStart/End + 两端 offset 表达跨 tensor 的连续逻辑区间
```

POD buffer 由 executor 保持到 launch；GM 输入/输出和 API workspace 由调用方保持到 stream 完成。operator Tiling 固定申请 32 B workspace，但 Kernel entry 将 `userWS` 置空；公开 API 总 workspace 还可包含 AutoContiguous 生成节点，固定仓没有总公式。

### 启动图与失败传播

```text
aclnn...GetWorkspaceSize
  -> generated contiguous/executor boundary
  -> Tiling4ForeachAddcmulScalarListTiling
     -> ForeachCommonTiling::Init/RunBigKernelTiling
        -> key + ForeachCommonTilingData + blockDim + workspace[0]=32
aclnnForeachAddcmulScalarList(stream)
  -> foreach_addcmul_scalar_list(x1,x2,x3,scalars,y,workspace,tiling)
     -> one selected ForeachOneScalarListQuaternaryImplictOutput<T,...>::Process
```

只有一个 Device 节点，无跨 Kernel 边。Host 检查或 workspace 获取失败即不启动；生成 executor 的 cleanup/错误码映射属于固定仓不可见框架边界。Kernel 对 unknown key 没有 `else` 错误写回，正常 Host 只生成 key 1–4。

## Route 与参数设计

### 分核与资源计算

Host 把所有 tensor 的逻辑元素串成一条序列，按 32 B block 分给最多 50 个 cores。每 core POD 记录首/尾 tensor 与偏移；Device 再逐 tensor 把本 core 的片段切为 `maxDataCount` tile。令 `U=inputsTensorUbSize`（字节）：

```text
FP16/BF16:
  U = align_down_64(floor(floor((UB - POD_bytes)/10)/8))
  maxDataCount = 9U / 2 elements
  maxCastDataCount = U / 4 elements
FP32/INT32:
  U = align_down_32(floor((UB - POD_bytes)/8))
  maxDataCount = U / 4 elements
```

低精度 `9U` tile 会在 FP32 scratch 中按 `U/4` 元素分 9 次（末次可短）处理。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 低精度升精度 | key 1 FP16；key 4 BF16 | `GetTilingKeyByDtypeOnly(DT_FLOAT16/BF16)`；BF16 entry 还要求 `__CCE_AICORE__>=220` 且非 NPU arch 3003/3113 | 其它 dtype 继续；不满足编译 guard 时无 BF16 body。 |
| 2 | R-002 FP32 | key 2 | dtype `DT_FLOAT` | 继续。 |
| 3 | R-003 INT32 | key 3 | dtype `DT_INT32` 且 `__CCE_AICORE__>=220` | 不支持 dtype/平台在 Host/构建边界失败。 |

```text
DT_FLOAT16 -> key1 -> Adapter<half> -> segmented FP32 Mul/Axpy/CAST_RINT
DT_FLOAT   -> key2 -> AdapterForFloat<float> -> Mul/Axpy
DT_INT32   -> key3 -> AdapterForInt<int> -> Mul/Muls/Add
DT_BF16    -> key4 -> Adapter<bfloat16_t> -> segmented FP32 path (compile guards apply)
else       -> unsupported Host failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1/4 | FP16/BF16、A2/A3、列表/shape/dtype checks pass | 单 AIV Kernel | `ForeachOneScalarList...<T,AddcMulScalarListAdapter<T>,2,3>` | 3×depth2×9U input queues + depth1×3U FP32 queue；V barriers/events | core/tensor/tile/cast 四级 tail；DataCopyPad。 |
| R-002 | 2 | FP32，同检查 | 单 AIV Kernel | `...<float,AdapterForFloat<float>,2,3>` | 3×depth2×U；inLocal1 原地结果 | tile tail DataCopyPad。 |
| R-003 | 3 | INT32、CCE≥220，同检查 | 单 AIV Kernel | `...<int,AdapterForInt<int>,2,3>` | 3×depth2×U；inLocal1 原地结果 | tile tail DataCopyPad。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32 B 分核 | `E=32/B`, `ceil(S/E)` | core 边界按 DMA block 粒度 | 全部 | 至少 1 core | `GetNeedCoreNum`, `AssignDataToEachCore` |
| core/tensor 数静态容量 | `needCoreNum<=50`; `Init` 只循环 `<50` | POD 数组固定 50 | 全部 | core cap 50；>50 tensors 无显式错误而被扫描上限截断 | `MAX_CORE_CONT`, `MAX_TENSOR_CONT`, `Init` |
| UB divider | low precision `(UB-POD)/10/8`; native `(UB-POD)/8` | 覆盖三路双缓冲及 FP32 scratch | 对应 Route | 无小 UB fallback | `DivideUbMemory1` |
| BF16/INT32 compile guard | `__CCE_AICORE__>=220`; BF16 排除 arch3003/3113 | entry 实例化条件 | R-001 BF16/R-003 | 无 Device fallback | Kernel router |
| tile tail | `dataCount % maxDataCount != 0` | 普通 `DataCopy` 需完整 tile；tail 用 byte-count Pad | 全部 | 同 Route tail | `SingleTensorProcess`, `CopyIn`, `Compute` |
| cast tail | `dataCount % (U/4) != 0` | FP32 scratch 每 slice 容量 `U` bytes | R-001 | 同 Route tail | `AddcMulScalarListAdapter` |

### POD/ABI

`ForeachCommonTilingData` 的字段顺序、类型和数组长度由固定源码明确；`BEGIN_TILING_DATA_DEF/TILING_DATA_FIELD_DEF*` 的宏定义在固定仓外，因此生成类型的精确 `sizeof/alignof/offsetof` 不可见，不能把自然 C++ 布局推测成序列化 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 精确 offset/sizeof/align 固定仓不可见；逻辑字段序号 0 | bytes/queue buffer base `U` | `DivideUbMemory1`→`set_inputsTensorUbSize` | `ParseTilingData`, `Init` | 32/64 B 向下对齐。 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 精确 offset 固定仓不可见；逻辑字段序号 1 | elements/tensor | `Init` shape size | `Process` tensor extent | 超过实际 count 的项为 0。 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 精确 offset 固定仓不可见；逻辑字段序号 2 | tensor index/core | `AssignDataToEachCore` | `ParseTilingData` | core<blockDim。 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 精确 offset 固定仓不可见；逻辑字段序号 3 | tensor index/core | 同上 | 同上 | inclusive。 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 精确 offset 固定仓不可见；逻辑字段序号 4 | element offset | 同上 | 同上 | inclusive start。 |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 精确 offset 固定仓不可见；逻辑字段序号 5 | element offset | 同上 | 同上 | inclusive end。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1/4 | `arch22/...tiling_arch22.cpp::Tiling4...`→`ForeachCommonTiling` | `foreach_tiling_def.h::ForeachCommonTilingData` | `op_kernel/...cpp::foreach_addcmul_scalar_list` | `foreach_one_scalar_list...h::{Init,Process,Compute}` + `kernel_foreach_unary.h` + adapters | `op_kernel/CMakeLists.txt` arch22 compute units |
| R-002 | 2 | 同上 | 同上 | 同上 key2 | `AdapterForFloat` + 公共 body | 同上 |
| R-003 | 3 | 同上 | 同上 | 同上 key3 | `AdapterForInt` + 公共 body | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 并按 key 构造一个模板实例。`Init` 解析本 core 的 tensor 区间，绑定四个 tensor-list 基址与 scalars GM，分配 UB queues。`Process` 逐 tensor 读取 `scalars[i]`，通过 tensor-list 首地址中的 pointer-table offset 取得实际数据指针，加本 core 的 `cursorStart`，再由 `SingleTensorProcess` 逐 tile 执行：三路 CopyIn→adapter 原地改写 x1 local→V_MTE3 event→直接 CopyOut→MTE3_MTE2 event→释放 x1 local。输出 queue 因模板 `needCopyOut=false` 不分配。

### Route R-001：FP16/BF16 分段升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | FP16→1，BF16→4 | key1/key4 |
| Router | `op_kernel/foreach_addcmul_scalar_list.cpp` | `foreach_addcmul_scalar_list` | `TILING_KEY_IS(1/4)` + compile guard | `...<half/bfloat16_t,Adapter<T>,2,3>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_addcmul_scalar_list(key1/4)
  -> ForeachOneScalarList...<T,Adapter<T>,2,3>::Init/Process
    -> KernelForeachUnary::SingleTensorProcess
      -> CopyIn(x1) + CopyInPlus(x2,x3)
      -> Compute -> InnerComputer -> AddcMulScalarListAdapter
        -> ComputerPerCast: Cast x2/x3/x1 -> Mul -> Axpy -> Cast_RINT x1
      -> direct DataCopy/DataCopyPad x1Local -> y
```

##### 源码等价伪码

```text
core range = [tensorStart:startOffset, tensorEnd:endOffset]
for each covered tensor i:
  s = float(scalars[i]); bind x1/x2/x3/y at cursorStart
  for tile t of at most 9U/2 elements:
    DMA x1,x2,x3 GM -> three depth-2 queues
    for slice k of at most U/4 elements:
      fp32[0]   = Cast(x2 slice)
      fp32[U/4] = Cast(x3 slice)
      fp32[2U/4]= Cast(x1 slice)
      fp32[0] *= fp32[U/4]
      fp32[2U/4] += s * fp32[0]
      x1 slice = Cast_RINT(fp32[2U/4])
    event V->MTE3; DMA valid x1Local bytes -> y; event MTE3->MTE2; free
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x1/x2/x3/y | tensor pointer + `cursorStart + t*maxDataCount` | valid `dataCount*sizeof(T)` | T, linear ND | caller | CopyIn/CopyOut | stream lifetime | tail byte copy. |
| `dataQueue` x1 | queue slot | depth 2 × `9U` bytes | T | CopyIn | Compute then CopyOut | alloc→free/tile | x1 local becomes result. |
| `InQueue_2` x2 | queue slot | depth 2 × `9U` bytes | T | CopyInPlus | Compute | alloc→free before output | no output copy. |
| `InQueue_3` x3 | queue slot | depth 2 × `9U` bytes | T | CopyInPlus | Compute | 同上 | no output copy. |
| `float32Queue` | three contiguous U-byte regions | depth 1 × `3U` bytes | FP32 | Cast | Mul/Axpy/Cast | whole Process | slices reused for each cast group/tile. |
| scalar GM/register | `scalars+i` | one T→float | T/FP32 | GM load | adapter | tensor iteration | once per tensor/core. |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  x1q["dataQueue x1/result · depth=2 · 18U B"]:2
  x2q["InQueue_2 · depth=2 · 18U B"]:2
  x3q["InQueue_3 · depth=2 · 18U B"]:2
  f0["FP32 x2/product · U B"]:2
  f1["FP32 x3 · U B"]:2
  f2["FP32 x1/result · U B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 3 * (depth2 * 9U) + (depth1 * 3U) = 57U bytes
U = align_down_64(floor(floor((UB_platform - POD_bytes)/10)/8))
UB_total <= UB_platform - POD_bytes  (Host 的 /10 与 /8 是保守预算)
dynamic_ub = N/A
operator_workspace_total = 32 B; Kernel usage = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 queue slot | CopyIn | Cast/adapter then CopyOut | `FreeTensor` 后下一 tile alloc | V_MTE3 + MTE3_MTE2 events | 输出 DMA 完成后才释放/覆盖。 |
| FP32 region 0 | Cast x2 then Mul product | Axpy | 下一 cast slice | PIPE_V barriers | 当前 Axpy 消费结束。 |
| FP32 regions 1/2 | Cast x3/x1 | Mul/Axpy/Cast_RINT | 下一 slice | PIPE_V barriers | 逐 slice 顺序。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | TQue EnQue/DeQue | MTE2 | Vector | 三路 local | tail `DataCopyPad`, valid bytes only。 |
| Vector stages | `PipeBarrier<PIPE_V>` | Cast/Mul/Axpy | 下一 Vector op | FP32 slices | FP32 product/add；回 T 用 `CAST_RINT`。 |
| Compute→CopyOut | `Set/WaitFlag<V_MTE3>` | Vector | MTE3 | x1 result | tail output byte length=`dataCount*sizeof(T)`。 |
| CopyOut→reuse | `Set/WaitFlag<MTE3_MTE2>` | MTE3 | next MTE2/free | queue slot | 防止输出尚未完成即覆盖。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling`, `DivideUbMemory1` | key/core/POD/U/workspace。 |
| Kernel entry/router | `foreach_addcmul_scalar_list/op_kernel/...cpp` | key1/key4 template instances | dtype→body/compile guard。 |
| Device body/helper | `foreach_utils/op_kernel/foreach_one_scalar_list...h` | `Init`, `Process`, `Compute` | queue、tensor/core/tile、events。 |
| Device body/helper | `foreach_addcmul_scalar_list/op_kernel/...cpp` | `AddcMulScalarListAdapter`, `ComputerPerCast...` | FP32 slice pipeline。 |

### Route R-002：FP32 Mul+Axpy

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key2 |
| Router | `op_kernel/...cpp` | key2 branch | `TILING_KEY_IS(2)` | `...<float,AdapterForFloat<float>,2,3>` |

#### Device 数据流

##### 函数级调用链

```text
entry key2 -> common Init/Process -> SingleTensorProcess
  -> CopyIn three FP32 tiles -> AdapterForFloat: Mul(x2,x2,x3); Axpy(x1,x2,s)
  -> direct CopyOut x1Local
```

##### 源码等价伪码

```text
for each core-covered tensor and U/4-element tile:
  x1Local,x2Local,x3Local = DMA inputs
  x2Local = x2Local * x3Local
  barrier V
  x1Local = x1Local + scalar * x2Local
  event V->MTE3; DMA valid x1Local -> y; event MTE3->MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | tensor base+core/tile offset | valid `dataCount*4` | FP32 | caller | DMA | stream | tail pad path. |
| x1/x2/x3 queues | slot | each depth2×U | FP32 | CopyIn | Mul/Axpy/output | tile | x2 product; x1 result. |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  x1["x1/result queue · depth=2 · 2U B"]
  x2["x2/product queue · depth=2 · 2U B"]
  x3["x3 queue · depth=2 · 2U B"]
```

##### 字节账与复用顺序

```text
UB_total = 3 * depth2 * U = 6U bytes
U = align_down_32(floor((UB_platform - POD_bytes)/8))
operator_workspace=32 B; Kernel usage=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x2 slot | CopyIn then Mul in-place | Axpy | FreeTensor then next alloc | PIPE_V + queue lifetime | Axpy 已消费。 |
| x1 slot | CopyIn then Axpy in-place | CopyOut | MTE3_MTE2 event then free | explicit events | 输出完成。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Mul→Axpy | PIPE_V barrier | `Mul` | `Axpy` | x2 product | FP32 原生。 |
| result→GM | V_MTE3/MTE3_MTE2 | Axpy/MTE3 | copy/free | x1 | tail valid bytes。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory1` | native U。 |
| Kernel entry/router | `op_kernel/...cpp` | key2 | FP32 instance。 |
| Device body/helper | `op_kernel/...cpp` | `AddcMulScalarListAdapterForFloat` | Mul→Axpy。 |
| Device body/helper | `kernel_foreach_unary.h` | `SingleTensorProcess`, `CopyIn` | tile/tail。 |

### Route R-003：INT32 Mul+Muls+Add

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key3 |
| Router | `op_kernel/...cpp` | key3 branch | `__CCE_AICORE__>=220 && TILING_KEY_IS(3)` | `...<int,AdapterForInt<int>,2,3>` |

#### Device 数据流

##### 函数级调用链

```text
entry key3 -> common Init/Process -> SingleTensorProcess
  -> CopyIn three INT32 tiles
  -> AdapterForInt: Mul(x2,x2,x3) -> Muls(x2,(int32_t)scalar) -> Add(x1,x1,x2)
  -> direct CopyOut x1Local
```

##### 源码等价伪码

```text
s_float = float(scalars[i]); s_int = int32(s_float)
for each tile:
  x2Local *= x3Local; barrier
  x2Local *= s_int; barrier
  x1Local += x2Local
  events; DMA x1Local -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | tensor base+core/tile offset | valid `dataCount*4` | INT32 | caller | DMA | stream | tail pad。 |
| three queues | slot | each depth2×U | INT32 | CopyIn | vector/output | tile | x2 holds both products; x1 holds result。 |
| scalar register | `scalars[i]`→float→int32 | one scalar | INT32 source, float intermediate | `valueScalar` | `Muls` | tensor iteration | conversion round semantics not otherwise specified。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  x1["x1/result queue · depth=2 · 2U B"]
  x2["x2/product queue · depth=2 · 2U B"]
  x3["x3 queue · depth=2 · 2U B"]
```

##### 字节账与复用顺序

```text
UB_total = 6U bytes
U = align_down_32(floor((UB_platform - POD_bytes)/8))
operator_workspace=32 B; Kernel usage=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x2 slot | CopyIn→Mul→Muls | Add | free after Add | PIPE_V barriers | Add 已消费。 |
| x1 slot | CopyIn→Add | CopyOut | after MTE3_MTE2 | events | output complete。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Mul→Muls→Add | PIPE_V barriers | each vector op | next op | x2/x1 | INT32；源码经 float scalar 再 cast int32。 |
| result→GM | V_MTE3/MTE3_MTE2 | Add/MTE3 | copy/free | x1 | tail byte count。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `RunBigKernelTiling` | key/core/U。 |
| Kernel entry/router | `op_kernel/...cpp` | key3 branch | CCE guard 与 INT32 instance。 |
| Device body/helper | `op_kernel/...cpp` | `AddcMulScalarListAdapterForInt` | Mul/Muls/Add。 |
| Device body/helper | `foreach_one_scalar_list...h` | `valueScalar(int)`, `Compute` | scalar float conversion、events。 |

## 算子特定优化

### Trick T-001：x1 Queue 原地承载结果并直接 CopyOut

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003。
- 触发条件：公共模板参数 `needCopyOut=false`；adapter 把结果写回 `inLocal1`。
- Golden 来源：[`foreach_one_scalar_list_quaternary_implict_output.h:53-78`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_quaternary_implict_output.h#L53-L78)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<T> inLocal1 = Base::dataQueue.template DeQue<T>();
InnerComputer<T, op> computer;
computer.Compute(inLocal1, inLocal2, inLocal3, float32Tensor, scalarValue, Base::maxCastDataCount, dataCount);
DataCopy(Base::outTensorsGM[index * Base::maxDataCount], inLocal1, dataCount);
```

#### 机制与设计原因

x1 的 VECIN queue 同时是加法累加器和最终输出源，模板不分配 VECOUT queue，也不物化额外 result tensor。显式 V_MTE3 与 MTE3_MTE2 event 保证原地结果完成后才输出、输出完成后才复用 slot。

#### 迁移边界

- 适用：输入允许在 UB 局部副本上覆盖、输出与输入 shape/dtype 一致的点运算。
- 前提：原始 x1 在后续计算中不再需要；输出 DMA 完成后才释放 slot。
- 失效：一个输入需被后续阶段再次读取或产生多输出时必须保留独立 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：9×低精度 tile 复用三段 FP32 scratch

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：`T` 为 FP16/BF16；`totalTensorUbSize=9U`、`maxCastDataCount=U/4`。
- Golden 来源：[`foreach_one_scalar_list_quaternary_implict_output.h:148-159`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_list_quaternary_implict_output.h#L148-L159)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t totalTensorUbSize = Base::Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
Base::Base::pipe.InitBuffer(Base::dataQueue, bufferNum, totalTensorUbSize);
Base::Base::pipe.InitBuffer(Base::float32Queue, 1, Base::Base::inputsTensorUbSize * paramsCount);
Base::Base::maxCastDataCount = Base::Base::inputsTensorUbSize / sizeof(float);
```

#### 机制与设计原因

每次 DMA 可装入 `9U` bytes 的低精度数据，而 FP32 scratch 始终只有三段 `U` bytes；adapter 把大 tile 分成最多 `U/4` 元素的小段，重复 Cast/Mul/Axpy/Cast。这样 FP32 中间内存不随完整低精度 tile 等比例扩张，同时维持较大的 GM tile。

#### 迁移边界

- 适用：低精度输入、FP32 中间、逐段无跨段依赖的点运算。
- 前提：每段所有 Vector op 完成并 barrier 后才能覆盖 scratch；Host UB divider 与 57U 实际账一致。
- 失效：跨段归约/扫描或需同时保留所有 FP32 中间时不能复用此布局。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16/BF16 | R-002 FP32 | R-003 INT32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key1/4 + guards | key2 | key3 + CCE≥220 | dtype-only。 |
| 模板与 dtype | `Adapter<T>` | `AdapterForFloat<float>` | `AdapterForInt<int>` | 精度/API 可用性。 |
| 循环与分核 | core→tensor→tile→cast slice | core→tensor→tile | 同 FP32 | 低精度需额外切片。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3×depth2×9U + FP32 3U | 3×depth2×U | 同 FP32 | 低精度升精度 scratch。 |
| 内存总占用与复用 | 57U，x1/FP32 原地复用 | 6U，x1/x2 原地 | 6U，x1/x2 原地 | adapter 数据流。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Cast/Mul/Axpy/Cast | Mul/Axpy | Mul/Muls/Add | AIV Vector only。 |
| 同步、精度与 tail | PIPE_V + events；CAST_RINT；双 tail | barriers/events；FP32 | barriers/events；INT32 | 同一公共搬运骨架。 |
| fallback/失败边界 | 编译 guard 无 fallback | 无 Device fallback | 编译 guard 无 fallback | Host 只发支持 key。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；ascend910b 接线 | 公开支持；ascend910_93 接线 | 公开支持，见 `_950` 卡。 |
| Host/Launcher | 共用 arch22 `ForeachCommonTiling` | 共用 arch22 `ForeachCommonTiling` | arch35 专用 max-tensor Tiling。 |
| Kernel/资源 | 共用 AIV Vector entry/body | 共用 AIV Vector entry/body | SIMT GM 直读直写，无显式 UB。 |

## 限制与失败边界

- `MAX_TENSOR_CONT=50`，但 `Init` 在第 50 项停止扫描而没有对第 51 项显式报错；实现复用应把 tensor list 长度限制为≤50，不能假设更长列表安全。
- 公开文档支持空 tensor；arch22 全部元素数为 0 时 Host 仍启动 1 core，而 POD 首尾 offset 保持零初始化，Device 会形成 `dataCount=1` 的源码路径。固定仓未提供专用 empty guard/fallback，空域是需谨慎验证的边界。
- Host 的 `uint32_t(ubSizePlatForm - tilingData.GetDataSize())` 未在减法前检查 UB 足够大，固定仓没有小 UB fallback。
- INT32 scalar 先由 `valueScalar(int)` 转为 float，再在 adapter 中强转回 int32；超出 FP32 精确整数范围的 scalar 可能无法保持原值，固定仓没有替代 Route。
- unknown TilingKey 在 entry 中静默不执行；正常公开入口只能由 Host 生成 1–4。
- operator workspace 32 B 不等于 ACLNN 总 workspace；AutoContiguous/executor 的总公式固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_addcmul_scalar_list/docs/aclnnForeachAddcmulScalarList.md` | `aclnnForeachAddcmulScalarList*` | 公开语义、参数、A2/A3 支持与错误。 |
| 入口实现 | `foreach/foreach_addcmul_scalar_list/op_host/foreach_addcmul_scalar_list_def.cpp` | `ops::ForeachAddcmulScalarList` | dtype/layout/AutoContiguous/平台注册。 |
| Host/Tiling/Launcher | `foreach/foreach_addcmul_scalar_list/op_host/arch22/foreach_addcmul_scalar_list_tiling_arch22.cpp` | `Tiling4ForeachAddcmulScalarListTiling` | 当前 op 到 common Tiling。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD、workspace。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD 注册与 ABI。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key 1–4。 |
| Kernel entry/body | `foreach/foreach_addcmul_scalar_list/op_kernel/foreach_addcmul_scalar_list.cpp` | entry + adapters | key router、计算。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_list_quaternary_implict_output.h` | template `Init/Process/Compute` | queues、scalar、tensor loop、events。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_quaternary_implict_output.h` | `InnerComputer` | adapter 调用与 Vector barrier。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `SingleTensorProcess`, `CopyIn` | tile/tail/queue pipeline。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `ParseTilingData`, `GetTensorAddr` | POD consumer 与 tensor-list pointer ABI。 |
| 构建接线 | `foreach/foreach_addcmul_scalar_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | A2/A3→arch22 Host。 |
| 构建接线 | `foreach/foreach_addcmul_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch22 compute units。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段/数组逻辑布局 | 数组长度 50、字段顺序、inclusive element offsets；宏展开后的字节 ABI 不可猜测 | `ForeachCommonTilingData` |
| 调用顺序 | 三路 CopyIn→compute→direct x1 CopyOut | x1 是 result owner；x2/x3 可先释放 | `SingleTensorProcess`, `Compute` |
| workspace/内存 | 32 B operator workspace 未被 Kernel 读；UB 57U/6U | depth=2 必须计入；ACLNN 总 workspace 另算 | `RunBigKernelTiling`, `Init` |
| 同步 | queue + PIPE_V + V_MTE3/MTE3_MTE2 | 输出完成前不得覆盖 x1 slot | adapters, `Compute` |
| 精度与尾块 | low precision FP32 slices/CAST_RINT；tail byte copy | cast 顺序、maxCastDataCount、valid bytes | `ComputerPerCast...`, `SingleTensorProcess` |
| 失败边界 | Host checks + compile guards | >50 tensors/全空域/unknown key 无安全 fallback | `Init`, `CheckOpInfo`, router |