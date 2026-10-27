---
type: Operator
title: "ForeachMulScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachMulScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/docs/aclnnForeachMulScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/op_api/foreach_mul_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/foreach_mul_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/config/ascend950/foreach_mul_scalar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
created_at: "2026-08-25T15:47:32Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMulScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMulScalarV2GetWorkspaceSize` / `aclnnForeachMulScalarV2`，公开 ACLNN 两段式 V2 入口 |
| 计算语义 | 对列表中每个张量逐元素计算 `out[i][j] = x[i][j] * scalar` |
| 输入输出 | `x/out` 为最多 50 个 ND Tensor 的列表，dtype FP16/FP32/BF16/INT32；`scalar` 为 `aclScalar`，内部转成单元素 Tensor |
| 执行类型 | 单 Kernel：arch35 `foreach_mul_scalar`，`AIV_VECTOR` |
| Kernel 路线 | R-001 FP16→FP32 直乘→FP16；R-002 FP32 直乘；R-003 INT32 直乘；R-004 BF16→FP32 直乘→BF16 |
| 核心资源 | TensorList 描述符、GM、depth=2 的输入/输出 TQue；FP16/BF16 另有 FP32 `TBuf`; Kernel 不读取 workspace |
| 来源闭包 | `source_closed`：公开 V2 ACLNN→L0 launcher→regbase priority/IsCapable→POD→key 10001–10004 router→arch35 body/基类→构建配置闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对列表长度 `N`、第 `i` 个 Tensor 的有效元素数 `L_i`：

```text
out[i][j] = x[i][j] * scalar, 0 <= i < N, 0 <= j < L_i
```

输出 dtype 与输入相同。公开 V2 文档声明默认确定性实现。TensorList 长度为 0 时，API 第一段直接返回 `workspaceSize=0`，因此 regbase Tiling 对列表长度 `[1,50]` 的要求不会影响公开 empty-list 快路径。列表非空但 Tensor 元素数为 0 时，Tiling 生成单核空域，Device 不搬运。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；各 Tensor ND、0–8 维 | FP16/FP32/BF16/INT32，列表内一致 | 与 `out` 等长且逐项 shape 相同；950 最多 50 个；V2 文档标注非连续 Tensor 不支持，内部仍经过框架 `Contiguous` 边界 |
| `scalar` | `aclScalar` | FP16/FP32/DOUBLE/INT32/INT64 | FP32/BF16 输入配 FP32/DOUBLE；FP16 配 FP16/FP32/DOUBLE；INT32 配 INT32/INT64 |
| `out` | TensorList；ND，逐项 shape 与 `x` 相同 | 与 `x` 相同 | 最多 50 个；API `CheckShape` 比底层 Tiling 的 size-only 检查更严格 |
| `workspaceSize` | 标量输出 | `uint64_t` | executor 汇总的 Device workspace |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有连续化、标量 Tensor 与单 AICore launcher |

### 入口与版本

V2 相对 `aclnnForeachMulScalar` 只改变公开 scalar 容器：从单元素 `aclTensor` 改为 `aclScalar`。公开产品表明确支持 Ascend 950PR/950DT，且 950 OpDef、ascend950 binary JSON 和 arch35 Kernel 构建同时闭合，所以本卡选择所有平台实际支持的最新版本 V2。内部仍使用 `ForeachMulScalar` OpType；不把 V2 当作不同数学语义。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckNotNull` 检查 `x/scalar/out`。
2. `GetDtypeSupportList` 因 `AclnnUtil::IsRegbase()` 接受 FP16/FP32/BF16/INT32；`CheckDtype` 检查列表内 dtype、输出 dtype 与 scalar 组合。
3. `CheckShape` 要求列表等长、`x` 不超过 8 维且逐项 shape 相同；`CheckFormat` 拒绝私有 storage format。
4. empty TensorList 直接返回 0 workspace，不创建 launcher。
5. 非空列表逐项经过框架 `l0op::Contiguous`，再创建内部 TensorList。
6. scalar 转换规则：BF16 输入总是转 FP32；FP16 且 scalar 为 FP32/DOUBLE 时转 FP32；FP16+FP16 转 FP16；FP32+DOUBLE 转 FP32；INT32+INT64 转 INT32。
7. `ForeachMulScalarV2` 通过 `ADD_TO_LAUNCHER_LIST_AICORE` 加入单个 `ForeachMulScalar` 节点。

参数、连续化、标量转换或 launcher 任一失败都会立即返回并阻断执行。

### 参数、Workspace 与生命周期

第一段的 `uniqueExecutor` 持有内部 TensorList、scalar Tensor 与 launcher，`GetWorkspaceSize()` 汇总 framework workspace。第二段在调用者给定 stream 上执行 executor。

Foreach regbase Tiling 的 `GetWorkspaceSize` 固定设置 32 B，`PostTiling` 写入该值；arch35 entry把原始 workspace 传给 `ForeachRegbaseUnary::Init`，但基类未绑定或读取它。因此节点的 semantic workspace 为 0 B，32 B 只是 ABI 预留。API 总 workspace 还可能含框架 `Contiguous` 空间，固定仓未展开其切片 offset/extent。

### 启动图与失败传播

```text
aclnnForeachMulScalarV2GetWorkspaceSize
  -> [框架边界] x[i] Contiguous
  -> aclScalar -> internal 1-element Tensor
  -> L0 ForeachMulScalarV2
  -> node-1 ForeachMulScalar / arch35 foreach_mul_scalar
       blockDim = numBlocks_
       key = 10001/10002/10003/10004
       POD = ForeachSoloTilingDataRegbase
       input = TensorList desc + scalar GM
       output = TensorList desc
       semantic workspace = 0 B

aclnnForeachMulScalarV2
  -> CommonOpExecutorRun(node-1, stream)
```

启动图只有一个 Device 节点，无跨 Kernel workspace/GM 边和事件。`AUTO_SYNC false` 不产生额外节点；节点内部依赖由 TQue 生命周期和同一 Kernel 中的 Vector 调用顺序表达。Tiling 返回 `GRAPH_FAILED` 会阻断 launch；unknown key 的 Device router 没有 fallback body。

## Route 与参数设计

### 分核与资源计算

令 `s=sizeof(T)`、`L=sum_i L_i`、`C=A5 AIV core count`，源码常量 `SINGLE_CORE_PROCESS_DATA=4096` 字节：

```text
elementsPerBlock = 4096 / s
numBlocks = min(C, 80, ceil(L / elementsPerBlock))
若 L==0：numBlocks=1，start=end=0，startOffset=0，endOffset=-1
否则：blockCount=ceil(L/elementsPerBlock)
        每核分配 floor(blockCount/numBlocks)*elementsPerBlock
        前 blockCount%numBlocks 核各多 elementsPerBlock 个元素
```

每核可跨越多个 Tensor，以 `tensorStart/End` 和首尾闭区间 offset 表示。令 `A=GetUbBlockSize(context)`，Host 为每个 tile 生成元素容量 `E=inputsTensorUbSize`：

```text
FP32/INT32: bytesPerElementBudget = 4*s
FP16/BF16:  bytesPerElementBudget = 4*s + sizeof(float)
E = floor_A((UB_platform / bytesPerElementBudget) * s) / s
```

Device 中 `maxDataCount=E`。`E*s` 先按平台 UB block 向下对齐；除法与数组 offset 分别使用 `int64_t/uint32_t/uint16_t`，迁移时需保持位宽。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 升精度 | key 10001；`ForeachMulScalarImpl<half,DTYPE_SCALAR>` | priority 10000 membase 先因 `IsRegbaseSocVersion=true` 返回 param invalid；priority 30000 regbase `IsCapable=true`；`x=FP16`，scalar Tensor=FP16或FP32 | dtype 不符继续 router |
| 2 | R-002 FP32 直乘 | key 10002；`ForeachMulScalarImpl<float,float>` | 同一 regbase selector，`x=FP32, scalar=FP32` | 继续 R-003 |
| 3 | R-003 INT32 直乘 | key 10003；`ForeachMulScalarImpl<int,int>` | 同一 selector，`x=INT32, scalar=INT32` | 继续 R-004 |
| 4 | R-004 BF16 升精度 | key 10004；`ForeachMulScalarImpl<bfloat16_t,float>` | 同一 selector，`x=BF16, scalar=FP32` | unknown key 无 body |

```text
TilingRegistry priority ascending:
  10000 ForeachMulScalarMembaseTiling
    IsCapable = !IsRegbaseSocVersion -> false on A5 -> GRAPH_PARAM_INVALID
  30000 ForeachRegbaseTilingUnaryScalar
    IsCapable = IsRegbaseSocVersion -> true
    GetShapeAttrsInfo -> CheckScalar -> DoOpTiling -> CheckOutput
    -> UB split -> PostTiling -> key by x dtype
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 10001 | 950 regbase；1–50 Tensor；FP16/FP16或FP16/FP32 internal scalar | 单 arch35 AIV Kernel | `ForeachMulScalarRegbase<half,DTYPE_SCALAR,...>` | depth2 input/output Queue + FP32 cast TBuf；Cast/Muls/Cast | 所有 tile 用精确字节 `DataCopyPad`; empty slice quotient=0 |
| R-002 FP32 | 10002 | 950；FP32/FP32 | 同上 | `<float,float,...>` | depth2 Queue；直接 `Muls` | 同上 |
| R-003 INT32 | 10003 | 950；INT32/INT32 | 同上 | `<int,int,...>` | depth2 Queue；直接整数 `Muls` | 同上 |
| R-004 BF16 | 10004 | 950；BF16/FP32 | 同上 | `<bfloat16_t,float,...>` | depth2 Queue + FP32 cast TBuf；Cast/Muls/Cast | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| regbase selector | `IsRegbaseSocVersion(context_)` | arch35 POD/key/body ABI | 全部 | membase 或最终 Tiling 失败 | `ForeachRegbaseTiling::IsCapable` |
| 列表长度 | `1 <= totalTensorCount_ <= 50` | POD Tensor 数组固定50 | 全部 | empty list 已在 API 返回；>50 Tiling失败 | `GetShapeAttrsInfo` |
| core 上限 | `numBlocks_=min(platform AIV,80,ceil(...))` | POD core数组固定80 | 全部 | 使用更少 core，无另一路线 | `DoOpTiling` |
| scalar 组合 | `{F32/F32, F16/F16, F16/F32, BF16/F32, I32/I32}` | binary config与模板 `ScalarT` | 全部 | `GRAPH_FAILED` | `CheckScalar`, `SUPPORT_DTYPE_COMB` |
| UB block 对齐 | `E*s = FloorAlign((UB/budget)*s, GetUbBlockSize)` | TQue/TBuf分配与 DMA block | 全部 | 无小 UB fallback；Tiling依赖正 `sizePerElem` | `ForeachRegbaseTilingUnaryScalar::DoOpTiling` |
| half/bf16 cast buffer | `T` 为 half 或 bfloat16 | Compute必须先升FP32 | R-001/R-004 | 编译期消除 castBuf 于 R-002/R-003 | `ForeachMulScalarRegbase::Init/Compute` |
| tail/空域 | `n` 为每 tile有效元素；空域 `end=-1` | 避免访问无效 GM | 全部 | `DataCopyPad` 精确 `n*s`; quotient 0不执行 | `ForeachRegbaseUnary::SingleTensorProcess/CopyIn/CopyOut` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | `uint32_t`/32 | 宏生成结构的 `sizeof/alignof/offsetof` 在固定仓依赖中不可见 | 元素，不是字节 | `ForeachRegbaseTilingUnaryScalar::DoOpTiling` | `ForeachRegbaseUnary::ParseTilingData/Init` | `E*s` 按 UB block 向下对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 同上 | 元素/张量 | `GetShapeAttrsInfo` | `Process` | shape各维非负；可为0 |
| 同上 | `tensorStartList` | `uint16_t[80]` | 同上 | Tensor 索引/core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | `blockIdx<numBlocks<=80` |
| 同上 | `tensorEndList` | `uint16_t[80]` | 同上 | Tensor 索引/core | 同上 | 同上 | 闭区间 |
| 同上 | `tensorStartOffsetList` | `int64_t[80]` | 同上 | 元素 | 同上 | 首 Tensor GM offset | 从0开始 |
| 同上 | `tensorEndOffsetList` | `int64_t[80]` | 同上 | 元素 | 同上 | 末 Tensor有效末元素 | 闭区间；空域为-1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10001 | `ForeachRegbaseTilingUnaryScalar` | `ForeachSoloTilingDataRegbase`; `DTYPE_SCALAR` binary specialization | arch35 `foreach_mul_scalar`→`ForeachMulScalarImpl<half,DTYPE_SCALAR>` | `ForeachMulScalarRegbase`; `ForeachRegbaseUnary` | arch35 CMake + ascend950 JSON FP16/FP16和FP16/FP32 entries |
| R-002 | 10002 | 同上 | 同一 POD | `Impl<float,float>` | 同一派生/基类 direct分支 | ascend950 FP32 entry |
| R-003 | 10003 | 同上 | 同一 POD | `Impl<int,int>` | 同上 | ascend950 INT32 entry |
| R-004 | 10004 | 同上 | 同一 POD | `Impl<bfloat16_t,float>` | cast分支 | ascend950 BF16/FP32 entry |

## Kernel 实现

### 公共执行流程

arch35 entry 创建一个 `TPipe`，按 key调用 `ForeachMulScalarImpl<T,ScalarT>`；该 helper 从 tiling GM 解出 regbase POD，构造 CRTP 派生类并调用 `Init/Process`。基类用 `ListTensorDesc` 解析输入/输出列表，blockIdx 选择本核的 Tensor 闭区间。每个 Tensor 片按 E 元素切 tile；`CopyIn` 和 `CopyOut` 始终使用 `DataCopyPad` 的单 block、`blockLen=n*sizeof(T)`、零 stride。scalar 在每个 core 的 `Init` 读取一次。最后显式 `pipeOp.Destroy()`。

### Route R-001：FP16 升精度乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckDtype/Exec...` | `x=FP16`; scalar=FP16/FP32/DOUBLE | internal scalar=FP16或FP32 |
| Tiling | `foreach_regbase_tiling.cpp` | `CheckScalar/GetTilingKey` | `{FP16,FP16}`或`{FP16,FP32}` | key10001、POD E |
| Router | `arch35/foreach_mul_scalar.cpp` | `foreach_mul_scalar` | `TILING_KEY_IS(10001)` | `Impl<half,DTYPE_SCALAR>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachMulScalarV2 launcher
  -> arch35 foreach_mul_scalar
    -> ForeachMulScalarImpl<half,DTYPE_SCALAR>
      -> ForeachMulScalarRegbase::Init -> ForeachRegbaseUnary::Init/ParseTilingData
      -> ForeachRegbaseUnary::Process
        -> SingleTensorProcess -> CopyIn -> ForeachMulScalarRegbase::Compute -> CopyOut
          -> Cast half→float -> Muls float in-place -> Cast float→half RINT
```

##### 源码等价伪码

```text
scalarVal = scalarGM<DTYPE_SCALAR>[0]
allocate inQueue depth2 E*2B, outQueue depth2 E*2B, castBuf E*4B
for tensor in this core's [start,end]:
  cursorStart/cursorEnd = POD closed interval
  bind input/output tensor address + cursorStart
  for tile in ceil(dataCount/E):
    n=min(E,remaining)
    DataCopyPad GM→half inQueue, blockLen=2n B
    Cast(castBuf, halfIn, CAST_NONE, n)
    Muls(castBuf, castBuf, float(scalarVal), n)
    Cast(halfOut, castBuf, CAST_RINT, n)
    DataCopyPad half outQueue→GM, blockLen=2n B
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `ListTensorDesc[i]+cursorStart+tile*E` | `n` | FP16/ND | caller/CopyOut | CopyIn/caller | caller | blockLen=2n B，零stride |
| scalar GM | 0 | 1 | FP16或FP32 | API conversion | Init | executor | 每core读一次 |
| `inQueue` | Queue slots | 每slot `E*2` B，depth2 | FP16 | CopyIn | Compute | TPipe/tile | double buffer |
| `outQueue` | Queue slots | 每slot `E*2` B，depth2 | FP16 | Compute | CopyOut | TPipe/tile | double buffer |
| `castBuf` | 唯一TBuf | `E*4` B | FP32 | Cast/Muls | Muls/Cast | TPipe/整个Kernel | Muls原地覆盖；逐tile复用 |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | ABI预留32 B未读取 | 不复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ0["inQueue[0] · E*2 B"]:1
  inQ1["inQueue[1] · E*2 B"]:1
  cast["castBuf · E*4 B"]:1
  outQ0["outQueue[0] · E*2 B"]:1
  outQ1["outQueue[1] · E*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*(E*2) + 2*(E*2) + E*4 = 12E bytes
budgetPerElement = 4*2 + 4 = 12 bytes
E = floor_A((UB_platform/12)*2)/2
UB_total <= UB_platform
dynamic_ub=N/A
workspace_total(node semantic)=0 B; Tiling reservation=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | CopyIn | Compute `DeQue` | 后续 tile CopyIn | `EnQue/DeQue/FreeTensor` | Free后覆盖 |
| outQueue slot | Compute | CopyOut `DeQue` | 后续 tile Compute | 同上 | CopyOut后Free |
| castBuf | upcast | in-place Muls/downcast | 下一 tile upcast | 同一 `Compute` 调用顺序 | downcast完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB/UB→GM | TQue + `DataCopyPad` | CopyIn/Compute | Compute/CopyOut | depth2 slots | 每tile只搬有效2n B |
| Cast/Muls/Cast | 同一 Vector 调用序列；源码无显式 `PipeBarrier` | upcast/Muls | Muls/downcast | castBuf | `CAST_NONE`→FP32乘→`CAST_RINT` |
| scalar | Init寄存值 | scalar GM read | 每tile Muls | scalar成员 | FP16 scalar先以half存储；FP32 scalar保留float后参与乘 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_regbase_tiling.cpp` | `ForeachRegbaseTilingUnaryScalar::DoOpTiling` | E预算、blockDim、key |
| Kernel entry/router | `foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar.cpp` | key10001 branch | half + binary `DTYPE_SCALAR`实例 |
| Device body/helper | `foreach_mul_scalar_regbase.h`; `foreach_regbase_unary.h` | `Init/Compute`; base `Process` | castBuf、三步计算、Queue/DMA/tail |

### Route R-002：FP32 直乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_mul_scalar_v2.cpp` | dtype/scalar conversion | `x=FP32`; scalar FP32/DOUBLE | internal FP32 |
| Tiling | `ForeachRegbaseTiling::GetTilingKey` | `dataType_=DT_FLOAT` | key10002 | FP32 Route |
| Router | `arch35/foreach_mul_scalar.cpp` | key10002 | true | `Impl<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar -> ForeachMulScalarImpl<float,float>
  -> ForeachMulScalarRegbase<float,float>::Init/Process
    -> base CopyIn -> Compute(Muls out,in,scalar,n) -> CopyOut
```

##### 源码等价伪码

```text
scalarVal=floatGM[0]
for core-owned tensor/tile n<=E:
  DataCopyPad 4n B -> inQueue
  Muls(outLocal,inLocal,scalarVal,n)
  DataCopyPad 4n B -> outputGM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 公共地址公式 | n | FP32/ND | caller/CopyOut | CopyIn/caller | caller | 4n B |
| scalar GM | 0 | 1 | FP32 | Host | Init | executor | 每core一次 |
| `inQueue` / `outQueue` | 各2个slot | 每slot E*4 B，depth2 | FP32 | CopyIn/Compute | Compute/CopyOut | tile | 无castBuf |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | N/A | 未消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ0["inQueue[0] · E*4 B"]:1
  inQ1["inQueue[1] · E*4 B"]:1
  outQ0["outQueue[0] · E*4 B"]:1
  outQ1["outQueue[1] · E*4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=2*E*4+2*E*4=16E B
E=floor_A((UB_platform/16)*4)/4
dynamic_ub=N/A; workspace_total(node semantic)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| in/out Queue slots | CopyIn/Compute | Compute/CopyOut | 后续tile | TQue FreeTensor | 消费后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/compute/copy | TQue；同一Kernel调用序列 | CopyIn/Muls | Compute/CopyOut | Queue slots | FP32直接乘；每tile精确4n B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `DoOpTiling/GetTilingKey` | 16B/元素预算、key10002 |
| Kernel entry/router | `arch35/foreach_mul_scalar.cpp` | `Impl<float,float>` | FP32实例 |
| Device body/helper | `foreach_mul_scalar_regbase.h` | non-half/bf16 Compute | 直接 `Muls`，编译期无castBuf分配 |

### Route R-003：INT32 直乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_mul_scalar_v2.cpp` | dtype/scalar conversion | `x=INT32`; scalar INT32/INT64 | internal INT32 |
| Tiling | `CheckScalar/GetTilingKey` | `{INT32,INT32}` | key10003 | INT Route |
| Router | `arch35/foreach_mul_scalar.cpp` | key10003 | true | `Impl<int,int>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar -> ForeachMulScalarImpl<int,int>
  -> base Process/CopyIn -> ForeachMulScalarRegbase::Compute(Muls<int>) -> CopyOut
```

##### 源码等价伪码

```text
与 R-002 相同，T=int32，scalarVal=int32；每tile读取/写回4n B。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 公共地址公式 | n | INT32/ND | caller/CopyOut | CopyIn/caller | caller | 4n B |
| scalar GM | 0 | 1 | INT32 | Host窄化 | Init | executor | INT64在Host转INT32 |
| `inQueue` / `outQueue` | 各2slot | 每slot E*4 B | INT32 | CopyIn/Compute | Compute/CopyOut | tile | 与R-002同布局 |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | N/A | 未消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ0["inQueue[0] · E*4 B"]:1
  inQ1["inQueue[1] · E*4 B"]:1
  outQ0["outQueue[0] · E*4 B"]:1
  outQ1["outQueue[1] · E*4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16E B; E=floor_A((UB_platform/16)*4)/4
dynamic_ub=N/A; workspace_total(node semantic)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Compute | Compute/CopyOut | 后续tile | TQue生命周期 | 与R-002一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/compute/copy | TQue/调用序列 | CopyIn/Muls | Compute/CopyOut | Queue slots | INT32直接 `Muls`；源码无额外升宽/饱和说明；tail 4n B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `CheckScalar/GetTilingKey` | INT组合与key10003 |
| Kernel entry/router | `arch35/foreach_mul_scalar.cpp` | `Impl<int,int>` | INT32实例 |
| Device body/helper | `foreach_mul_scalar_regbase.h` | direct Compute | INT32 Muls |

### Route R-004：BF16 升精度乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_mul_scalar_v2.cpp` | dtype/scalar conversion | `x=BF16`; scalar FP32/DOUBLE | internal FP32 |
| Tiling | `CheckScalar/GetTilingKey` | `{BF16,FP32}` | key10004、12B/元素预算 | BF16 Route |
| Router | `arch35/foreach_mul_scalar.cpp` | key10004 | true | `Impl<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar -> ForeachMulScalarImpl<bfloat16_t,float>
  -> ForeachMulScalarRegbase::Init/Process
    -> base CopyIn -> Cast BF16→FP32 -> in-place FP32 Muls -> Cast FP32→BF16 RINT -> CopyOut
```

##### 源码等价伪码

```text
scalarVal=floatGM[0]
for core-owned tensor/tile n<=E:
  DataCopyPad 2n B BF16 into inQueue
  Cast(castBuf,inLocal,CAST_NONE,n)
  Muls(castBuf,castBuf,scalarVal,n)
  Cast(outLocal,castBuf,CAST_RINT,n)
  DataCopyPad 2n B BF16 to outputGM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 公共地址公式 | n | BF16/ND | caller/CopyOut | CopyIn/caller | caller | 2n B |
| scalar GM | 0 | 1 | FP32 | Host conversion | Init | executor | 每core一次 |
| `inQueue` / `outQueue` | 各2slot | 每slot E*2 B | BF16 | CopyIn/Compute | Compute/CopyOut | tile | double buffer |
| `castBuf` | 唯一TBuf | E*4 B | FP32 | Cast/Muls | Muls/Cast | Kernel | 原地Muls、逐tile复用 |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | N/A | 未消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ0["inQueue[0] · E*2 B"]:1
  inQ1["inQueue[1] · E*2 B"]:1
  cast["castBuf · E*4 B"]:1
  outQ0["outQueue[0] · E*2 B"]:1
  outQ1["outQueue[1] · E*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=12E B
E=floor_A((UB_platform/12)*2)/2
dynamic_ub=N/A; workspace_total(node semantic)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castBuf | upcast | Muls/downcast | 下一tile | 同一Compute顺序 | downcast完成后覆盖 |
| Queue slots | CopyIn/Compute | Compute/CopyOut | 后续tile | TQue lifecycle | Free后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32→BF16 | 同一Vector调用序列，无显式barrier | Cast/Muls | Muls/Cast | castBuf | `CAST_NONE`，FP32乘，`CAST_RINT` |
| Queue/DMA | TQue + DataCopyPad | GM/Compute | Compute/GM | depth2 slots | 每tile精确2n B；无效lane不写GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_regbase_tiling.cpp` | `DoOpTiling/GetTilingKey` | 12B/元素预算与key10004 |
| Kernel entry/router | `arch35/foreach_mul_scalar.cpp` | `Impl<bfloat16_t,float>` | BF16实例 |
| Device body/helper | `foreach_mul_scalar_regbase.h` | cast `Compute` | BF16升精度、原地Muls、RINT |

## 算子特定优化

### Trick T-001：用编译期 dtype 分支消除不需要的 FP32 buffer

#### 触发与源码依据

- 适用 Route：R-001/R-004 分配 castBuf；R-002/R-003 编译期不分配。
- 触发条件：模板参数 `T` 是否为 `half` 或 `bfloat16_t`。
- Golden 来源：[`foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar_regbase.h:32-50`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar_regbase.h#L32-L50)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Base::Init(inputs, outputs, workspace, tilingData, tPipe);
inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)scalar, 1);
if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
    scalarVal_ = inScalarGM_.GetValue(0);
    tPipe->InitBuffer(castBuf_, tilingData->inputsTensorUbSize * sizeof(float));
    castLocal_ = castBuf_.Get<float>();
} else {
    scalarVal_ = T(inScalarGM_.GetValue(0));
}
```

#### 机制与设计原因

FP16/BF16 Route 显式保留 E 个 FP32 元素供升精度计算；FP32/INT32 Route 在 `if constexpr` 的未命中分支完全不初始化该 TBuf，Host 对应地把每元素 UB 预算从 `4s+4` 降为 `4s`。这不是运行时空 buffer，而是模板实例级资源消除；因此 direct Route 能把可用 UB 全部分给 depth2 输入/输出 Queue。

#### 迁移边界

- 适用：dtype 决定是否需要升精度中间量，且 binary config会生成独立模板实例的逐元素算子。
- 前提：Host `ubSizePerNumber` 必须与每个模板实际分配一致；FP16/BF16 budget含额外4 B/元素。
- 失效：若 FP32/INT32 后续算法也需要临时 TBuf，必须同步修改 Host预算，不能只添加Device分配。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16 在同一 FP32 TBuf 中原地乘

#### 触发与源码依据

- 适用 Route：R-001、R-004
- 触发条件：`T` 为 `half` 或 `bfloat16_t`。
- Golden 来源：[`foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar_regbase.h:43-50`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar_regbase.h#L43-L50)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
    Cast(this->castLocal_, inLocal, RoundMode::CAST_NONE, dataCount);
    Muls(this->castLocal_, this->castLocal_, float(this->scalarVal_), dataCount);
    Cast(outLocal, this->castLocal_, RoundMode::CAST_RINT, dataCount);
} else {
    Muls(outLocal, inLocal, this->scalarVal_, dataCount);
}
```

#### 机制与设计原因

升精度输入和乘法输出共用 `castLocal_` 同一物理子区，`Muls(dst=castLocal_,src=castLocal_)` 原地覆盖，随后立即降精度到 output Queue。这样只需一个 `E*sizeof(float)` TBuf，而不需分别物化 FP32 输入和 FP32 输出。scalar 由每core的GM单次读取值转成float，`dataCount`来自当前tail-aware tile。

#### 迁移边界

- 适用：支持原地 Vector运算且旧值在该指令后不再使用的逐元素链。
- 前提：Cast/Muls/Cast顺序不变；输出 Queue独立；Host为一个FP32 TBuf预留4 B/元素。
- 失效：后续步骤需要同时读取乘前和乘后 FP32值时，必须拆分buffer或重算，不能原地覆盖。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key10001；scalar half/float | key10002 | key10003 | key10004 | regbase priority30000在membase拒绝后命中 |
| 模板与 dtype | half/`DTYPE_SCALAR` | float/float | int/int | bf16/float | ascend950 binary dtype组合 |
| 循环与分核 | 4096B块、最多80核、E tile | 同左 | 同左 | 同左 | 公共regbase基类 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth2 Queues + FP32 TBuf | depth2 Queues | depth2 Queues | depth2 Queues + FP32 TBuf | compile-time分配 |
| 内存总占用与复用 | 12E B，TBuf原地 | 16E B | 16E B | 12E B，TBuf原地 | s与cast预算 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+Cast/Muls/Cast | DataCopyPad+Muls | 同左 | DataCopyPad+Cast/Muls/Cast | 全部AIV Vector |
| 同步、精度与 tail | TQue；FP32中间/RINT | FP32 direct | INT32 direct | FP32中间/RINT | dtype语义 |
| fallback/失败边界 | scalar binary specialization | unknown key无body | 同左 | 同左 | Host/build/router必须一致 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开 V2 支持 |
| Host/Launcher | membase key1–4，最多50 core | 同左 | regbase key10001–10004，最多80 core |
| Kernel/资源 | 公共 membase；FP16直接half；BF16分段FP32 Queue | 同左 | arch35；FP16/BF16均整tile FP32 TBuf；depth2 Queue；`--cce-no-dcache-flush` |

## 限制与失败边界

- 950 V2 API 最多50个输入/输出 Tensor；各维非负且不超过8维，`x/out`逐项shape和dtype相同。
- FP16 scalar可为FP16或FP32内部Tensor，binary config产生两个实例并通过 `DTYPE_SCALAR` 绑定；复用时不能固定假设 scalar GM 为half。
- INT64/DOUBLE只存在于公开 scalar入口，进入Kernel前分别转INT32/FP32（FP16+DOUBLE转FP32）；Device不直接读取64位 scalar。
- 生成式POD宏的数值 `sizeof/alignof/offsetof` 在固定仓依赖中不可见；字段顺序、类型和使用闭合，但不填造offset。
- 节点Tiling固定预留32 B，Kernel不使用；API executor可能还汇总框架连续化workspace，具体slice固定仓不可见。
- arch35 body没有显式 unknown-key错误分支；Host selector、key、binary config必须一致。
- 源码没有显式Vector `PipeBarrier`；不能把A2/A3 barrier协议虚构到950，实际依赖按TQue和同一调用序列描述。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_mul_scalar/docs/aclnnForeachMulScalarV2.md` | 产品表、参数表 | 950公开支持、scalar组合、最多50 |
| 入口声明/约束 | `foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.h` | 两段式声明 | V2 C ABI |
| 入口实现 | `foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckParams`, `Exec...`, phase-2 | 校验、连续化、950 scalar转换、executor |
| 入口实现 | `foreach/foreach_mul_scalar/op_host/op_api/foreach_mul_scalar_v2.cpp` | `ForeachMulScalarV2` | 单AICore launcher |
| Host注册 | `foreach/foreach_mul_scalar/op_host/foreach_mul_scalar_def.cpp` | `ForeachMulScalar` | ascend950 regbase dtype组合 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | registry与`Tiling4ForeachMulScalarTiling` | priority/selector入口 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.h` | `ForeachRegbaseTilingUnaryScalar` | IsCapable与状态字段 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `GetShapeAttrsInfo/CheckScalar/DoOpTiling/PostTiling/GetTilingKey` | 校验、分核、UB、POD、key、workspace |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachSoloTilingDataRegbase` | POD字段/数组容量 |
| Kernel entry/body | `foreach/foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar.cpp` | `foreach_mul_scalar`, `ForeachMulScalarImpl` | key router/TPipe/POD解析 |
| Kernel body | `foreach/foreach_mul_scalar/op_kernel/arch35/foreach_mul_scalar_regbase.h` | `ForeachMulScalarRegbase` | scalar load、castBuf、Muls |
| Kernel body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h` | `ForeachRegbaseUnary` | TensorList循环、Queue、DMA、tail |
| Kernel body | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_common.h` | key宏 | 10001–10004 |
| 构建接线 | `foreach/foreach_mul_scalar/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950专用源码/options |
| 构建接线 | `foreach/foreach_mul_scalar/op_host/config/ascend950/foreach_mul_scalar_binary.json` | `ForeachMulScalar` op_list | 五个dtype binary实例 |
| Tiling框架 | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority顺序与template fallthrough |
| Tiling框架 | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | `IsCapable`、Tiling阶段顺序与返回语义 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段描述每核Tensor闭区间 | Tensor数组50、core数组80、E单位为元素、offset闭区间 | `ForeachSoloTilingDataRegbase` |
| 调用顺序 | 单launch逐Tensor逐tile | empty list在API返回；scalar/Contiguous先于launcher | `Exec...`; `ForeachRegbaseUnary::Process` |
| workspace/内存 | Kernel semantic workspace0；depth2 Queue；条件castBuf | FP16/BF16为12E B，FP32/INT32为16E B；32 B只作ABI预留 | Host `DoOpTiling`; Device `InitBuffer` |
| 同步 | TQue生命周期与同一Vector调用序列 | slot Free前不可覆盖；950源码无显式Vector barrier | base `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP16/BF16升FP32、原地Muls、RINT；所有tile DataCopyPad | scalar实际dtype由binary实例决定；只写有效n元素 | derived `Compute`; base DMA |
| 失败边界 | membase拒绝后regbase命中；Host校验失败阻断 | unknown key无body；selector/key/binary需一致 | `IsCapable`; `GetTilingKey`; router |