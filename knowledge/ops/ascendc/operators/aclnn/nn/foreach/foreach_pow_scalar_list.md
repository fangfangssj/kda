---
type: Operator
title: "ForeachPowScalarList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachPowScalarList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/docs/aclnnForeachPowScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_host/foreach_pow_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_host/config/ascend910b/foreach_pow_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_host/config/ascend910_93/foreach_pow_scalar_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_host/config/ascend950/foreach_pow_scalar_list_binary.json
    role: implementation
created_at: "2026-08-25T16:03:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachPowScalarList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachPowScalarListGetWorkspaceSize` + `aclnnForeachPowScalarList`，公开无版本后缀的两段式 ACLNN |
| 计算语义 | 对列表位置 `i` 的每个有效元素 `j` 计算 `out[i][j] = pow(x[i][j], scalars[i])` |
| 输入输出 | `x/out` 为 ND TensorList，统一 `FLOAT16/FLOAT32/INT32/BFLOAT16`；`scalars` 为与列表等长的 `FLOAT32/INT64` ScalarList |
| 执行类型 | 单 Kernel：`foreach_pow_scalar_list`，`AIV_VECTOR`；没有 AIC/Cube 或 `MIX_CV` 节点 |
| Kernel 路线 | R-001 FP16；R-002 FP32；R-003 INT32；R-004 BF16 升精度 |
| 核心资源 | TensorList 指针表 GM、scalar GM、depth=2 的输入/输出 Queue；BF16 另有两组 depth=2 FP32 cast Queue；32 B 保留 workspace 不被 Kernel 使用 |
| 来源闭包 | `source_closed`：公开 ACLNN 约束、OpDef/推导、Host/Tiling、POD、entry、四个 dtype body 与 A2/A3 binary/CMake 接线闭合；自动生成 ACLNN bridge/launcher 属框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x=[x_0,...,x_{n-1}]`、`scalars=[e_0,...,e_{n-1}]`，输出 `y=[y_0,...,y_{n-1}]`。对每个列表位置 `i` 和 `x_i` storage shape 覆盖的元素 `j`，Kernel 调用 `Power<T,false>` 计算 `y_i[j]=x_i[j]^{e_i}`。指数对整个 `x_i` 广播。Host 允许 `out_i` storage size 大于 `x_i`，但 Device 只写与 `x_i` 元素数相同的前缀，额外区域不由本算子定义。FP16、FP32、INT32 直接在目标 dtype 上调用 `Power`；BF16 先转 FP32、执行 `Power<float,false>`，再以 `CAST_RINT` 转回 BF16。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND，rank 0–8 | 列表内统一 `FLOAT16/FLOAT32/INT32/BFLOAT16` | 支持空 Tensor；OpDef `AutoContiguous()`，公开 ACLNN 支持非连续输入 |
| `scalars` | ScalarList；元素数等于 `x` 中 Tensor 数 | `x` 为浮点/BF16 时 `FLOAT32`；`x` 为 `INT32` 时 `INT64` | 第 `i` 个指数广播到 `x_i`；公开约束要求绝对值不超过 `2^31-1` |
| `out` | TensorList；每项 ND，rank 0–8 | 与 `x` 相同且列表内统一 | 输出 Tensor 数与 `x` 相同；每项 storage size 不小于输入对应项；不支持非连续输出 |

### 入口与版本

固定仓只公开无版本后缀的 `aclnnForeachPowScalarList`，没有同语义 V2/V3 入口。第一段 `GetWorkspaceSize(x,scalars,out,workspaceSize,executor)` 完成校验并建立 executor；第二段以调用者提供的 workspace 和 stream 执行。README 与 ACLNN 文档明确支持 Atlas A2、A3，OpDef、`ascend910b/ascend910_93` binary config 和 Kernel CMake 与之相符。固定仓不包含自动生成的 `aclnnop/aclnn_foreach_pow_scalar_list.h` 实体、ACLNN inner bridge 和 executor launcher；这些由 `ACLNNTYPE aclnn`/OpDef 生成，作为框架边界，不补造其内部符号。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口拒绝 `x/scalars/out` 空指针、不支持 dtype 以及 `x/out` dtype 不一致。`ForeachCommonTiling::Init` 从动态输入 0 枚举最多 POD 可容纳的 50 个 Tensor，取 storage shape size、统一 dtype、元素字节数和总元素数。`CheckOpInfo` 随后检查：输入列表实例数关系；每项 rank≤8；`scalars` storage size 等于 Tensor 数；输出描述存在、dtype 等于输入、storage size 不小于输入；列表内所有 Tensor dtype 一致。公共 infer-shape/infer-dtype 在图推导路径中令每个输出 shape/dtype 等于对应输入。输入的非连续转连续由 OpDef `AutoContiguous()` 触发，生成 bridge 中的具体复制节点和临时 workspace 固定仓不可见。Tiling 任一检查失败返回 `GRAPH_FAILED`，由 ACLNN 框架映射为公开文档中的参数或 inner-tiling 错误。

### 参数、Workspace 与生命周期

设总元素数 `N=Σ_i numel_storage(x_i)`、元素字节 `S`、每个 32 B block 的元素数 `B=32/S`、平台 AIV 核数 `Cplat`。Host 生成 `C=min(ceil(N/B),Cplat,50)`；若 `N=0`，`GetNeedCoreNum` 强制 `C=1`。它把 TensorList 视为按列表次序连接的逻辑元素流，以 32 B block 为粒度近似均分，再把每核区间编码成首尾 Tensor 索引和首尾元素 offset。Tiling 固定写 `workspace[0]=32`，但 entry 将 `userWS` 设为 `nullptr`，算子 body 不消费这 32 B。executor、stream、自动连续化临时对象和整体 workspace 由框架持有到第二段任务完成；其精确额外字节组成在固定仓不可见。

### 启动图与失败传播

单节点启动图为：公开 ACLNN bridge（固定仓自动生成）→ `Tiling4ForeachPowScalarListTiling` → 一次启动 `foreach_pow_scalar_list(inputs,scalar,outputs,workspace,tiling)`。节点 `blockDim=C`，POD 为 `ForeachCommonTilingData`；`inputs/outputs` 是 TensorList 描述区，Kernel 从其中的偏移头和指针表解析每项真实 GM 地址。没有 Kernel 间 workspace 边、event 或跨节点生命周期。Tiling、executor 构建或 launch 失败均阻止该单节点提交；第二段使用调用者传入 stream，框架的精确 launch 返回包装固定仓不可见。

## Route 与参数设计

### 分核与资源计算

`blockCount=ceil(N/B)`，每核基础份额是 `floor(blockCount/C)·B` 元素，前 `blockCount mod C` 个核再多 `B` 元素。份额可以跨 Tensor，`AssignDataToEachCore` 用 `tensorStart/EndList` 和 `tensorStart/EndOffsetList` 将连续区间投影回各 Tensor。设平台 UB 为 `P`、序列化 POD 字节数为 `H`、`K=GetTilingN(dtype)`，其中 FP16/BF16 为 14、FP32/INT32 为 4；Host 先预留 `Rpow=1024·K·S` 字节给 `Power` 内部实现，再计算：非 BF16 `U=floor32((P-H-Rpow)/4)`；BF16 `U=floor64(((P-H-Rpow)/10)/4)`。这里 `U=inputsTensorUbSize`，单位为每个 Queue slot 的字节。整数除法均向下取整；固定源码没有对减法下溢或 `U=0` 的显式 fallback。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | key `1`，`ForeachPowScalarListND<half>` | OpDef/binary 支持，所有 Host 校验通过，`x.dtype=DT_FLOAT16` | 不支持 dtype 或校验失败则 `GRAPH_FAILED` |
| 2 | R-002 FP32 | key `2`，`ForeachPowScalarListND<float>` | 同上且 `x.dtype=DT_FLOAT` | 同上 |
| 3 | R-003 INT32 | key `3`，`ForeachPowScalarListND<int>` | 同上且 `x.dtype=DT_INT32`，scalars binary dtype 为 `INT64` | 同上 |
| 4 | R-004 BF16 | key `4`，`ForeachPowScalarListND<bfloat16_t>` | 同上且 `x.dtype=DT_BF16`；router 编译条件不是 NPU arch 3003/3113 | guard 未编译时无匹配 body；A2/A3 binary 含 BF16 |

```text
Tiling4ForeachPowScalarListTiling
  -> Init(opCode=21, inputType=TYPE_SCALARS_TENSOR)
  -> CheckOpInfo
  -> key = GetTilingKeyByDtypeOnly(x.dtype)
       DT_FLOAT16 -> 1 -> R-001
       DT_FLOAT   -> 2 -> R-002
       DT_INT32   -> 3 -> R-003
       DT_BF16    -> 4 -> R-004
       other      -> 0 / OpDef 或 Tiling 失败；router 对未知 key 不执行 body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | `1` | A2/A3；统一 FP16；scalar 为 FP32；列表、rank、shape、output 校验通过 | 单 AIV `foreach_pow_scalar_list` | `ForeachPowScalarListND<half>` + generic `InnerComputer<half>` | depth=2 的 `U` 输入/输出 Queue；Queue 跨 MTE/Vector 同步 | tile 尾用 `DataCopyPad`；`Power` 长度为有效元素数 |
| R-002 FP32 | `2` | A2/A3；统一 FP32；scalar 为 FP32；其余同上 | 同上 | `ForeachPowScalarListND<float>` + generic `InnerComputer<float>` | 同上 | 同上 |
| R-003 INT32 | `3` | A2/A3；统一 INT32；scalar binary 为 INT64且公开值域可转 INT32 | 同上 | `ForeachPowScalarListND<int>` + generic `InnerComputer<int>` | 同上 | 同上；无负指数特设分支 |
| R-004 BF16 | `4` | A2/A3；统一 BF16；scalar 为 FP32；编译 guard 成立 | 同上 | `ForeachPowScalarListND<bfloat16_t>` + `InnerComputer<bfloat16_t>` specialization | 输入/输出 Queue 各为 depth=2、slot=9U；两组 FP32 cast Queue 各 depth=2、slot=U；显式 `PIPE_V` barrier | 外层 tile 与内层 cast chunk 均处理 remainder，最后 `CAST_RINT` |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor/POD 上限 | `totalTensorCount≤50`，`blockDim≤50` | POD 的 Tensor 与 core 数组均固定 50 项 | 全部 | 公开/框架应限制；Tiling `Init` 超过 50 后不再枚举，固定源码无独立溢出 Route | `MAX_TENSOR_CONT/MAX_CORE_CONT`、`Init` |
| 32 B 分核 | `B=32/S`，`C=min(ceil(N/B),Cplat,50)` | 让大多数核边界落在 DMA block 粒度 | 全部 | 无其他 Route | `GetNeedCoreNum`、`AssignDataToEachCore` |
| Power scratch 预留 | `Rpow=1024·GetTilingN()·S` | 高层 `Power` primitive 需要 Host 留出未由算子 `InitBuffer` 声明的内部 UB 余量 | 全部 | 固定源码无小 UB fallback | `GetTilingN`、`DivideUbMemory6` |
| Queue 字节账 | 非 BF16 `4U+Rpow+H≤P`；BF16 `40U+Rpow+H≤P` 的保守构造 | depth=2 必须计入；BF16 需两套放大 Queue 和两套 FP32 Queue | 全部 | `U=0` 时 Device 除零风险，Host 未显式拒绝 | `ForeachPowScalarListND::Init` |
| BF16 chunk | `maxCastDataCount=U/4>0` | cast Queue 每 slot 只有 `U` 字节，外层 BF16 tile 最大 `9U/2` 元素，必须分段转换 | R-004 | 固定源码无 fallback；依赖 Host 保证非零 | `InnerComputer<bfloat16_t>::Compute` |
| BF16 编译 guard | `arch!=3003 && arch!=3113` | 这两个 Kirin arch 不编译 BF16 router 分支 | R-004 | 未命中时未知 key 静默离开 entry | `foreach_pow_scalar_list` |
| 空域 | `N=0 -> C=1` | 令启动维度合法；首核 `endOffset=-1` 时有效 `dataCount=0` | 全部 | 成功且不搬运/不写输出 | `GetNeedCoreNum`、`Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t/64` | 第 1 字段；序列化宏实现不在固定仓，`sizeof/alignof/offsetof` 固定仓不可见 | 字节/Queue slot | `DivideUbMemory6` 后 `FillTilingData::set_inputsTensorUbSize(U)` | `ParseTilingData`；`InitBuffer` 与 tile/cast 上限 | 必须使 `maxDataCount>0`；无显式 Host guard |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第 2 字段；精确 offset 固定仓不可见 | 每 Tensor 的 storage 元素数 | `Init: tempShape.GetShapeSize()` | `Process` 计算各 Tensor 的末元素 | 空 Tensor 为 0 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 第 3 字段；精确 offset 固定仓不可见 | 每核首 Tensor 索引 | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | `blockIdx<C≤50` |
| 同上 | `tensorEndList` | `uint16_t[50]` | 第 4 字段；精确 offset 固定仓不可见 | 每核尾 Tensor 索引 | 同上 | 同上 | 两端均为闭区间 |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 第 5 字段；精确 offset 固定仓不可见 | 首 Tensor 内元素 offset | 同上 | `Process` 的 `cursorStart` | 一般非负 |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 第 6 字段；精确 offset 固定仓不可见 | 尾 Tensor 内末元素 offset | 同上 | `Process` 的 `cursorEnd` | 空域首核为 `-1` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 1 | `foreach_tiling_func.cpp:Tiling4ForeachPowScalarListTiling`；`foreach_tiling_class.h:RunBigKernelTiling` | `ForeachCommonTilingData` 六字段 | `op_kernel/foreach_pow_scalar_list.cpp:foreach_pow_scalar_list` | `ForeachPowScalarListND<half>`、generic `InnerComputer::Compute` | Kernel/Host CMake + `ascend910b`、`ascend910_93` binary JSON |
| R-002 FP32 | key 2 | 同上 | 同上 | 同上 | `ForeachPowScalarListND<float>`、generic `InnerComputer::Compute` | 同上 |
| R-003 INT32 | key 3 | 同上 | 同上 | 同上 | `ForeachPowScalarListND<int>`、generic `InnerComputer::Compute` | 同上 |
| R-004 BF16 | key 4 | 同上 | 同上 | guarded key 4 branch | `ForeachPowScalarListND<bfloat16_t>`、BF16 `CopyPerCast/ComputerPerCast` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 `ForeachCommonTilingData`，按 key 构造一个 `ForeachPowScalarListND<T>`。每核读取自己的首尾 Tensor 和 offset；对相交 Tensor，先从 `scalar[i]` 取指数并转换为计算 dtype，再通过 TensorList 描述区的首个字节偏移找到 `uint64_t` 指针表，绑定当前 Tensor 的 GM 前缀。外层按 `maxDataCount` 分 tile：`CopyIn` 从 GM 进入 `dataQueue`，`InnerComputer` 执行 Power，结果入 `outQueue`，随后 `CopyOut` 写回 GM。整 tile 使用 `DataCopy`，尾 tile 使用单 block 的扩展参数 `DataCopyPad`；Queue 的 `Alloc/EnQue/DeQue/FreeTensor` 约束 MTE2、Vector 和 MTE3 的生产消费与 slot 生命周期。workspace 不参与 Device 数据流。

### Route R-001：FP16 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_scalar_list_def.cpp` | `ForeachPowScalarList::ForeachPowScalarList` | `x/y=DT_FLOAT16`，scalar 映射为 `DT_FLOAT` | A2/A3 config 可达 |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `dataType==ge::DT_FLOAT16` | key 1 |
| Router | `foreach_pow_scalar_list.cpp` | `foreach_pow_scalar_list` | `TILING_KEY_IS(1)` | `ForeachPowScalarListND<half>` |

#### Device 数据流

##### 函数级调用链

```text
framework launch foreach_pow_scalar_list(inputs, scalar, outputs, workspace, tiling)
  -> ForeachPowScalarListND<half>::Init
  -> ForeachPowScalarListND<half>::Process
    -> GetInputTensorAddr/GetOutputTensorAddr
    -> SingleTensorProcess
      -> CopyIn
      -> ComputeAndCopyOut
        -> InnerComputer<half>::Compute -> Power<half,false>
        -> DataCopy/DataCopyPad to output GM
```

##### 源码等价伪码

```text
U = tiling.inputsTensorUbSize; maxDataCount = U / 2
for tensor i in [tensorStart, tensorEnd]:
  begin = (i==tensorStart ? tensorStartOffset : 0)
  end   = (i==tensorEnd ? tensorEndOffset : tensorDataCount[i]-1)
  exponent = half(scalarGM[i])
  bind inGM/outGM = tensor_pointer_table[i] + begin
  count = end-begin+1
  for each full tile k of maxDataCount:
    DataCopy(inQ, inGM[k*maxDataCount], maxDataCount)
    Power<half,false>(outQ, inQ, exponent)
    DataCopy(outGM[k*maxDataCount], outQ, maxDataCount)
  if remainder>0:
    DataCopyPad exact remainder bytes -> Power(valid count) -> DataCopyPad exact bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input pointer table/GM | `dataAddr+(firstOffset>>3)`；`ptr[i]+coreBegin+k·U/2` | 每 tile `≤U/2` 元素 | FP16/ND | ACLNN/framework | `CopyIn` | caller；stream 任务期有效 | descriptor 以 64-bit 指针项解析 |
| `dataQueue` | Queue slot | depth `2` × slot `U` B | FP16 | MTE2 | `Power` | `TPipe`；`FreeTensor` 后 slot 可复用 | Host 将 U 向下对齐 32 B |
| `outQueue` | Queue slot | depth `2` × slot `U` B | FP16 | `Power` | MTE3 | 同上 | 同上 |
| output GM | `outPtr[i]+coreBegin+k·U/2` | 有效 tile 元素数 | FP16/ND | `CopyOut` | caller | caller | 超出输入长度的输出后缀不写 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
operator_owned_UB = 2U + 2U = 4U
Power_reserved_UB = 1024 * 14 * 2 = 28672 B
4U + Power_reserved_UB + H <= P（由 Host U 公式保守构造）
dynamic_ub = N/A
workspace_total = 32 B reserved；entry 令 userWS=nullptr
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | `CopyIn` | `ComputeAndCopyOut`/Power | 后续 tile | `EnQue→DeQue→FreeTensor` | 消费完成前不可覆盖 |
| `outQueue` slot | Power | `DataCopy/DataCopyPad` | 后续 tile | `EnQue→DeQue→FreeTensor` | MTE3 消费受 Queue 生命周期保护 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `dataQueue` | `DataCopy/Pad` | `DeQue` | input slot | 尾 tile `blockCount=1, blockLen=dataCount·2 B`，pad disabled |
| Compute | `Power<half,false>` | input Queue + half exponent | output Queue | 有效 lane | FP32 scalar 在核内先窄化为 half；有效长度为 `dataCount` |
| UB→GM | `outQueue` | Power | `DeQue` 后 MTE3 | output slot | 尾 tile只写 `dataCount·2 B` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling/DivideUbMemory6` | blockDim、key、U 与 workspace |
| Kernel entry/router | `foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.cpp` | key 1 branch | FP16 实例映射 |
| Device body/helper | `foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.h` | `ForeachPowScalarListND<half>`、generic `InnerComputer` | 指针表、tile、搬运与直接 half Power |

### Route R-002：FP32 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_scalar_list_def.cpp` | constructor | `x/y=DT_FLOAT`，scalar=`DT_FLOAT` | A2/A3 可达 |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_pow_scalar_list.cpp` | key 2 branch | `TILING_KEY_IS(2)` | `ForeachPowScalarListND<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar_list
  -> ForeachPowScalarListND<float>::Init/Process
    -> SingleTensorProcess -> CopyIn
    -> ComputeAndCopyOut -> InnerComputer<float>::Compute
      -> Power<float,false> -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 地址和循环相同；maxDataCount=U/4，scalarGM[i] 保持 float，
每个 tile 调用 Power<float,false>(outLocal,dataLocal,scalarValue)，尾 tile 只搬运和计算有效元素。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `ptr[i]+coreBegin+k·U/4` | `≤U/4` 元素/tile | FP32/ND | caller/Kernel | Kernel/caller | stream 任务期 | tail exact-byte copy |
| `dataQueue` | slot | `2U` B | FP32 | MTE2 | Power | pipe | depth=2 |
| `outQueue` | slot | `2U` B | FP32 | Power | MTE3 | pipe | depth=2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
operator_owned_UB = 4U
Power_reserved_UB = 1024 * 4 * 4 = 16384 B
4U + 16384 + H <= P
dynamic_ub = N/A
workspace_total = 32 B reserved, unused by body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output Queue slots | CopyIn/Power | Power/CopyOut | 后续 tile | `EnQue/DeQue/FreeTensor` | 与 R-001 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V→GM | TQue | MTE2/Power | Vector/MTE3 | 两组 slot | FP32 全程无 cast；尾部按有效字节与有效 `dataCount` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `GetTilingN/DivideUbMemory6` | FP32 reserve 和 U |
| Kernel entry/router | `foreach_pow_scalar_list.cpp` | key 2 branch | FP32 实例 |
| Device body/helper | `foreach_pow_scalar_list.h` | `InnerComputer<float>::Compute` | 直接 FP32 Power |

### Route R-003：INT32 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_scalar_list_def.cpp` | `DtypeTensor2Scalar` 使用点 | `x/y=DT_INT32`，scalar binary=`DT_INT64` | A2/A3 可达 |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `foreach_pow_scalar_list.cpp` | key 3 branch | `TILING_KEY_IS(3)` | `ForeachPowScalarListND<int>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar_list
  -> ForeachPowScalarListND<int>::Init/Process
    -> scalarValue = int(inScalarGM.GetValue(i))
    -> CopyIn -> InnerComputer<int>::Compute -> Power<int,false> -> CopyOut
```

##### 源码等价伪码

```text
与 R-002 tile/地址相同；maxDataCount=U/4。
从 DTYPE_SCALARS=int64 的 scalar GM 读 e_i，再窄化为 int，调用 Power<int,false>。
源码没有按 e_i 正负或特殊值分 Route；公开 abs(e_i)<=2^31-1 保证该窄化值域。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| scalar GM | `scalar[i]` | 1 × 8 B/Tensor | INT64/list | framework | `Process` | executor/task | 读后窄化 INT32 |
| input/output GM | `ptr[i]+coreBegin+k·U/4` | `≤U/4` 元素/tile | INT32/ND | caller/Kernel | Kernel/caller | stream 任务期 | tail exact-byte copy |
| `dataQueue` | slot | `2U` B | INT32 | MTE2 | Power | pipe | depth=2 |
| `outQueue` | slot | `2U` B | INT32 | Power | MTE3 | pipe | depth=2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 2U B"]:2
```

##### 字节账与复用顺序

```text
operator_owned_UB = 4U
Power_reserved_UB = 1024 * 4 * 4 = 16384 B
4U + 16384 + H <= P
dynamic_ub = N/A
workspace_total = 32 B reserved, unused
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Power | Power/CopyOut | 后续 tile | Queue lifecycle | 同一 slot 释放后再覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar load | GM scalar read + C++ cast | INT64 GM | integer Power | exponent | 文档值域可放入 INT32；源码没有负指数/溢出特设 |
| GM→V→GM | TQue | MTE2/Power | Vector/MTE3 | tile slots | INT32 直接计算；尾部按有效元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `GetTilingN/DivideUbMemory6` | INT32 reserve、U 和 key |
| Kernel entry/router | `foreach_pow_scalar_list.cpp` | key 3 branch | INT32 实例 |
| Device body/helper | `foreach_pow_scalar_list.h` | `Process`、`InnerComputer<int>` | INT64→INT32 exponent 和直接 integer Power |

### Route R-004：BF16 分段升精度 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_scalar_list_def.cpp` | constructor | `x/y=DT_BF16`，scalar=`DT_FLOAT` | A2/A3 binary 可达 |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `foreach_pow_scalar_list.cpp` | guarded key 4 branch | `TILING_KEY_IS(4)` 且 arch 非 3003/3113 | `ForeachPowScalarListND<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar_list
  -> ForeachPowScalarListND<bfloat16_t>::Init/Process
    -> SingleTensorProcess -> CopyIn
    -> ComputeAndCopyOut
      -> InnerComputer<bfloat16_t>::Compute
        -> CopyPerCast -> Cast(BF16->FP32, CAST_NONE)
        -> ComputerPerCast -> Power<float,false>
        -> Cast(FP32->BF16, CAST_RINT)
      -> CopyOut
```

##### 源码等价伪码

```text
U = tiling.inputsTensorUbSize
outerCapacity = (9U)/2 BF16 elements
castCapacity  = U/4 FP32 elements
for each core-owned Tensor interval:
  exponent = float(scalarGM[i])
  for each outer tile of at most outerCapacity:
    copy BF16 GM -> dataQueue
    q = floor(dataCount/castCapacity), r = dataCount mod castCapacity
    for chunk c in [0,q):
      Cast fp32In = BF16 input[c*castCapacity : +castCapacity], CAST_NONE
      Power<float,false>(fp32Out, fp32In, exponent)
      Cast BF16 output[c*castCapacity :], fp32Out, CAST_RINT
    if r>0: execute same three steps with valid length r
    copy only outer tile valid BF16 bytes -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `ptr[i]+coreBegin+k·(9U/2)` | outer tile `≤9U/2` elements | BF16/ND | caller/Kernel | Kernel/caller | stream 任务期 | tail exact-byte copy |
| `dataQueue` | slot | depth `2` × slot `9U` B | BF16 | MTE2 | BF16→FP32 Cast | pipe | depth=2；9 倍放大 |
| `outQueue` | slot | depth `2` × slot `9U` B | BF16 | FP32→BF16 Cast | MTE3 | pipe | depth=2；9 倍放大 |
| `dataCastQueue` | slot | depth `2` × slot `U` B | FP32 | first Cast | Power | pipe | 每 chunk `≤U/4` 元素 |
| `outCastQueue` | slot | depth `2` × slot `U` B | FP32 | Power | final Cast | pipe | 消费后释放 |

##### 分层内存布局

```mermaid
block-beta
  columns 10
  inQ["dataQueue · depth=2 · 18U B"]:3
  castIn["dataCastQueue · depth=2 · 2U B"]:2
  castOut["outCastQueue · depth=2 · 2U B"]:2
  outQ["outQueue · depth=2 · 18U B"]:3
```

##### 字节账与复用顺序

```text
operator_owned_UB = 18U + 18U + 2U + 2U = 40U
Power_reserved_UB = 1024 * 14 * 2 = 28672 B
Host: total=(P-H-28672)/10; U=floor64(total/4)
=> 40U + 28672 + H <= P（忽略 floor 留下的余量）
dynamic_ub = N/A
workspace_total = 32 B reserved, unused
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | `CopyIn` | 所有 BF16→FP32 chunks | 后续 outer tile | 外层 `DeQue` 后直到 `Compute` 返回才 `FreeTensor` | 当前 outer tile 所有 chunks 完成 |
| `dataCastQueue` slot | `CopyPerCast` | `ComputerPerCast` Power | 后续 chunk | `EnQue→DeQue→FreeTensor` + `PIPE_V` barrier | Cast 写完再 Power |
| `outCastQueue` slot | Power | FP32→BF16 Cast | 后续 chunk | `EnQue→DeQue→FreeTensor` + `PIPE_V` barrier | Power 写完再 Cast |
| `outQueue` slot | final Cast | `CopyOut` | 后续 outer tile | `EnQue→DeQue→FreeTensor` | 所有 chunks 已写入不同 offset |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→BF16 UB | `dataQueue` | MTE2 | `DeQue` | outer tile | 尾外层 tile使用 `DataCopyPad` |
| BF16→FP32 | `PipeBarrier<PIPE_V>` + cast Queue | Cast | Power | FP32 chunk | `CAST_NONE`，chunk remainder 使用真实长度 |
| FP32 Power | `PipeBarrier<PIPE_V>` + outCast Queue | `Power<float,false>` | final Cast | FP32 result | scalar 保持 FP32 |
| FP32→BF16 | `Cast(...,CAST_RINT,dataCount)` | FP32 result | BF16 outQueue | 对应 output offset | 最近偶数舍入；只转换有效元素 |
| BF16 UB→GM | `outQueue` | all chunks | MTE3 | outer tile | 只写有效 BF16 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `GetTilingN/DivideUbMemory6` | BF16 reserve、`/10` 衰减、`/4` Queue 分配与 64 B 对齐 |
| Kernel entry/router | `foreach_pow_scalar_list.cpp` | guarded key 4 branch | BF16 平台 guard 与实例 |
| Device body/helper | `foreach_pow_scalar_list.h` | BF16 `Init/Compute/CopyPerCast/ComputerPerCast` | 40U 自有 Queue、分块 cast、FP32 Power、RINT 回写 |

## 算子特定优化

### Trick T-001：BF16 以小 FP32 Queue 分段升精度，复用放大的 BF16 tile

#### 触发与源码依据

- 适用 Route：R-004。
- 触发条件：`T= bfloat16_t`、key 4，且 Host 产生的 `inputsTensorUbSize=U` 使 `U/4>0`。
- Golden 来源：[`foreach_pow_scalar_list.h:167-180`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// Init for bfloat16
if (std::is_same<T, bfloat16_t>::value) {
    uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
    pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
    pipe.InitBuffer(outQueue, BUFFER_NUM, totalTensorUbSize);
    maxDataCount = totalTensorUbSize / sizeof(T);
    reserved4CastingUbSize = inputsTensorUbSize;
    pipe.InitBuffer(dataCastQueue, BUFFER_NUM, reserved4CastingUbSize);
    pipe.InitBuffer(outCastQueue, BUFFER_NUM, reserved4CastingUbSize);
    maxCastDataCount = reserved4CastingUbSize / sizeof(float);
} else {
    pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
    pipe.InitBuffer(outQueue, BUFFER_NUM, inputsTensorUbSize);
    maxDataCount = inputsTensorUbSize / sizeof(T);
}
```

#### 机制与设计原因

BF16 路线没有为整个外层 tile 同时物化 FP32 输入和输出。它把 BF16 input/output slot 放大到 `9U`，一次从 GM 取得最多 `9U/2` 个元素；内部仅以两组 `U` 字节 FP32 Queue 按 `U/4` 元素分段完成 `BF16→FP32→Power→BF16`，每段消费后立即释放 cast slot。这样外层 GM tile 较大，而 FP32 中间态的同时存活量受控为两组 depth=2 的 `U` Queue。Host 的 BF16 `/10` 衰减恰好覆盖 `18U+18U+2U+2U=40U` 的算子自有 Queue；`COPY_SPACE_MULTIPLE=9`、`BUFFER_NUM=2` 和 `sizeof(float)` 分别决定放大倍数、双缓冲深度和 cast chunk 上限。

#### 迁移边界

- 适用：元素级高层 primitive 在 BF16 上需要 FP32 计算、且可把大 GM tile 拆成独立 FP32 chunk 的路线。
- 前提：A2/A3 key 4 可达；BF16/FP32 cast 与 `Power<float,false>` 可用；每个 chunk 独立，`U/4>0`；Host 必须按 40U 自有占用与 Power reserve 共同约束 UB。
- 失效：若平台不支持 BF16 router 分支、计算需要跨 chunk 状态，或 UB 不能容纳四组 Queue，就必须另设 Route 或拒绝；固定源码没有 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从当前字节账可直接确定，迁移时若遗漏 Queue depth=2 或把 `9U` 当成两组 Queue 的总容量而非单 slot 容量，会低估 UB，但固定仓没有对应错误实现。

### Trick T-002：按逻辑拼接域分核、Device 直接解析 TensorList 指针表

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：TensorList 非空或含空 Tensor，Host 已生成每核首尾 Tensor/offset。
- Golden 来源：[`foreach_pow_scalar_list.h:276-294`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__aicore__ inline __gm__ T* ForeachPowScalarListND<T>::GetInputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}
```

#### 机制与设计原因

Host 不把多个 Tensor 物化为一块连续数据，而只在 POD 中描述各核覆盖的列表区间。Device 从 TensorList 描述头取得指针表偏移，再按 `index` 直接读取真实 GM 指针，给首尾 Tensor 加元素 offset。其结果是列表可跨核、单个大 Tensor 也可被多核切分，同时没有额外 flatten/unflatten workspace 或 GM 数据复制；决定地址的 `tensorStart/End*` 全由 `AssignDataToEachCore` 产生。

#### 迁移边界

- 适用：运行时以相同 TensorList 描述 ABI 传递、各列表项可独立按元素区间处理的算子。
- 前提：描述区首项确为到 64-bit 指针表的字节偏移；指针表与实际 Tensor GM 在整个 stream 任务期间有效；Host 与 Device 对列表顺序和元素单位一致。
- 失效：描述 ABI 不同、需要跨 Tensor 连续邻域或输出重排时不能直接复用；应重新设计地址和交接。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；当前源码明确以 `tensorPtrOffset>>3` 将字节偏移转成 `uint64_t` 项偏移，若迁移时直接把该值当指针项数会得到错误地址。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 3 | key 4 + arch guard | dtype-only key，无形状 fallback |
| 模板与 dtype | `ND<half>` | `ND<float>` | `ND<int>` | `ND<bfloat16_t>` specialization | BF16 需 FP32 中间态 |
| 循环与分核 | 32 B block 分核；tile U/2 | tile U/4 | tile U/4 | outer tile 9U/2 + inner chunk U/4 | dtype 字节和 cast 容量不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 两组 depth=2 Queue；标量寄存值 | 同左 | 同左；INT64 标量窄化 | 四组 depth=2 Queue；标量 FP32 | 无 dynamic UB/通信 window |
| 内存总占用与复用 | 4U + 28672 reserve + H | 4U + 16384 reserve + H | 同 R-002 | 40U + 28672 reserve + H | BF16 Host `/10` 衰减匹配放大 Queue |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopy/Pad + Power<half>` | `...Power<float>` | `...Power<int>` | `DataCopy/Pad + Cast + Power<float> + Cast` | 全部为 AIV Vector，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue；FP32 exponent窄化 FP16 | Queue；FP32 | Queue；INT64 exponent窄化 INT32 | Queue + `PIPE_V` barrier；`CAST_RINT` | 外层和 BF16 内层均有 remainder |
| fallback/失败边界 | 无其他 Route | 无其他 Route | 无负指数特设 | guard/U 不满足无 fallback | 未知 key entry 静默不执行 body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` binary/config | 公开支持；`ascend910_93` binary/config | 公开 README 与 ACLNN 文档明确不支持；虽有 `ascend950` OpDef/config/CMake 接线，构成发布冲突，未生成正式卡 |
| Host/Launcher | 共用 `ForeachCommonTiling`、key 1–4、POD 与最多 50 核 | 同 A2 | 内部接线仍指向同 Host，但不得据此宣称公开 ACLNN 支持 |
| Kernel/资源 | 同一 `ForeachPowScalarListND` AIV body；具体 `P/Cplat` 由平台信息提供 | 同一 body；具体 `P/Cplat` 可不同 | Kernel CMake 含 `ascend950`，但公开“不支持”优先 |

## 限制与失败边界

- 公开接口要求 `x/scalars/out` 非空指针；支持空 Tensor；输入/输出 rank≤8，输出不支持非连续 Tensor。
- `scalars` 数量必须等于 Tensor 数；`x/out` dtype 必须一致，列表内 dtype 必须一致；输出 storage size 小于输入时报 Tiling 失败，较大时多余后缀不写。
- POD 数组固定 50 项；固定 Host `Init` 只枚举前 50 个 Tensor。公开文档未在本文件中显式给出 TensorList 最大 50 的条目，因此超过 50 的 ACLNN 前置拒绝细节固定仓不可见，不能依赖静默截断。
- Host 先做无符号 UB 减法再窄化为 `uint32_t`，且没有 `U=0` guard；异常小 UB、减法下溢与 Device 除零的失败行为固定仓不可见，不能推测为安全 fallback。
- INT32 路线把 INT64 exponent 窄化到 INT32；公开值域保证幅值可表示，但负指数、整数 Power 溢出和特殊底数没有算子自定义处理，最终数值边界取决于固定仓外的 `Power<int,false>` primitive。
- FP16 先把 FP32 exponent 窄化为 half；BF16 保持 FP32 计算并以 `CAST_RINT` 回写。`Power` primitive 内部算法、临时 UB 的精确布局、特殊值和误差界属于 AscendC 库边界，固定仓不可见。
- 自动生成 ACLNN bridge、连续化节点、executor/launcher、32 B 保留 workspace 的框架用途及 launch 错误包装固定仓不可见。
- 950 公开文档明确“不支持”，不得因 `ascend950` 实现接线存在而发布 `_950` 卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_pow_scalar_list/docs/aclnnForeachPowScalarList.md` | 两段式原型、产品支持表、参数/错误表 | 公开 API、A2/A3 支持、950 不支持、dtype/shape/连续性约束 |
| 入口声明/约束 | `foreach/foreach_pow_scalar_list/README.md` | 产品支持、公式、参数表 | 语义和第二份平台冲突证据 |
| 入口实现 | `foreach/foreach_pow_scalar_list/op_host/foreach_pow_scalar_list_def.cpp` | `ForeachPowScalarList`、`OP_ADD` | dtype 映射、AutoContiguous、平台注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon/InferDataType4ForeachCommon` | 图推导的输出 shape/dtype |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachPowScalarListTiling`、`IMPL_OP_OPTILING` | opCode/inputType 与 Tiling 注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace、POD producer |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData`、`REGISTER_TILING_DATA_CLASS` | POD 字段和算子绑定 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key |
| Kernel entry/body | `foreach/foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.cpp` | `foreach_pow_scalar_list` | 单 AIV entry、key router、BF16 guard |
| Kernel entry/body | `foreach/foreach_pow_scalar_list/op_kernel/foreach_pow_scalar_list.h` | `ForeachPowScalarListND`、`InnerComputer` | 全部循环、地址、Queue、搬运、Power/cast、tail |
| 构建接线 | `foreach/foreach_pow_scalar_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 与内部 ascend950 Kernel 编译接线 |
| 构建接线 | `foreach/foreach_pow_scalar_list/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn ...)` | Host/ACLNN 生成接线与 foreach_utils 依赖 |
| 构建接线 | `foreach/foreach_pow_scalar_list/op_host/config/ascend910b/foreach_pow_scalar_list_binary.json` | `op_list` | A2 四个 dtype binary |
| 构建接线 | `foreach/foreach_pow_scalar_list/op_host/config/ascend910_93/foreach_pow_scalar_list_binary.json` | `op_list` | A3 四个 dtype binary |
| 构建接线 | `foreach/foreach_pow_scalar_list/op_host/config/ascend950/foreach_pow_scalar_list_binary.json` | `op_list` | 950 内部四 dtype binary 信号；与公开不支持结论冲突，不能用于发布 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段描述 tile 容量、Tensor 长度与每核首尾区间 | 字段顺序、类型、数组长 50、字节/元素单位、闭区间语义 | `ForeachCommonTilingData`、`ParseTilingData` |
| 调用顺序 | 单 entry；每 Tensor 先读 scalar，再按 tile CopyIn→Power→CopyOut | BF16 每个 chunk 必须 Cast→Power→RINT Cast | `Process/SingleTensorProcess/ComputerPerCast` |
| workspace/内存 | workspace 固定保留 32 B 且 body 不用；非 BF16 4U，BF16 40U | Queue depth=2、BF16 slot=9U、Power reserve 和 Host 对齐不能漏算 | `RunBigKernelTiling/DivideUbMemory6/Init` |
| 同步 | 跨 MTE/Vector 依赖 Queue；BF16 Vector 链另有 `PIPE_V` barrier | tensor/slot 在消费和 `FreeTensor` 前不可覆盖 | `CopyIn/ComputeAndCopyOut/CopyPerCast/ComputerPerCast` |
| 精度与尾块 | FP16/FP32/INT32 直接 Power；BF16 升 FP32、RINT 回写；两层 remainder | exponent cast、有效 `dataCount`、tail exact-byte copy 不得改变 | `InnerComputer/CopyIn/ComputeAndCopyOut` |
| 失败边界 | Host 校验失败返回 `GRAPH_FAILED`；未知 key 无 body；950 冲突不发布 | 不得把未知 key、零 U 或 950 内部接线当成可用 fallback | `CheckOpInfo`、entry router、公开产品支持表 |

- [Power（数学计算）](../../../../apis/simd_api/adv_api/math_compute/power_interface/power.md)
- [ScalarList](../../../../apis/utils_api/prototype_register_management/op_param_def/scalar_list.md)
