---
type: Operator
title: "ForeachPowScalarAndTensor A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachPowScalarAndTensor 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/docs/aclnnForeachPowScalarAndTensor.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_host/foreach_pow_scalar_and_tensor_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_host/foreach_pow_scalar_and_tensor_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_host/config/ascend910b/foreach_pow_scalar_and_tensor_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_host/config/ascend910_93/foreach_pow_scalar_and_tensor_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:03:53Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachPowScalarAndTensor A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnForeachPowScalarAndTensorGetWorkspaceSize` + `aclnnForeachPowScalarAndTensor`；固定仓未出现带版本后缀的更新入口。 |
| 计算语义 | 对张量列表逐元素计算 `out[i][j] = scalar ^ x[i][j]`。 |
| 输入输出 | `scalar` 为标量；`x/out` 为 ND TensorList，元素 dtype 为 FP16、FP32、INT32 或 BF16；同一列表 dtype 一致，`out` dtype 与 `x` 一致。 |
| 执行类型 | 单 Kernel、单 `AIV_VECTOR` 节点；key 1/2/3 直接调用 `Power<T, false>`，key 4 先 BF16→FP32、以 FP32 求幂、再 FP32→BF16。 |
| Kernel 路线 | R-001：FP16/FP32/INT32 直接 Power；R-002：BF16 分段升精度 Power。 |
| 核心资源 | TensorList 地址表与数据位于 GM；每核有 VECIN/VECOUT `TQue`；BF16 另有常驻 FP32 `TQue`；算子 body 不读 workspace，Tiling 仍上报 32 B。 |
| 来源闭包 | `source_closed`：公开 API 文档→OpDef/infer→`ForeachCommonTiling`→`ForeachCommonTilingData`→kernel router→`ForeachPowScalarAndTensorND` body→A2/A3 binary/CMake 接线闭合；生成式 ACLNN bridge 与 AscendC `Power` primitive 内部按框架边界处理。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`，输出 `out=[out_0,...,out_{n-1}]`。对每个列表位置 `i` 和 `x_i` 的每个逻辑元素 `j`：

```text
out_i[j] = scalar ^ x_i[j]
```

每个 AIV core 获得拼接逻辑域中的一个连续区间；区间可跨越多个 Tensor，但 Kernel 在每个 Tensor 边界重新绑定 GM 指针。只写 `x_i` 的逻辑元素数，`out_i` 即使容量更大，超出该范围的区域也不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `scalar` | ACLNN 标量；内部 OpDef 标记 `.Scalar()`，Tiling 要求 storage size 为 1、维数不超过 8 | 当 `x` 为 FP16/FP32/BF16 时，公开 API 接受 FLOAT32/DOUBLE；当 `x` 为 INT32 时接受 INT64 | 底数。固定仓的 OpDef/binary 实际输入为 FLOAT32 或 INT64；DOUBLE→FLOAT32 的 ACLNN 生成 bridge 转换代码固定仓不可见。 |
| `x` | ND TensorList；每个 Tensor 0–8 维，可非连续输入，支持零元素 Tensor | FP16、FP32、INT32、BF16 | 所有成员 dtype 一致；公开文档要求元素绝对值不大于 `2^31-1`。OpDef `.AutoContiguous()` 表示非连续输入在框架边界连续化。 |
| `out` | ND TensorList；每项 storage shape size 不小于对应 `x`；0–8 维；必须连续 | 与 `x` 相同 | 列表成员数由 infer/执行图与 `x` 对齐；只写与 `x` 相同的逻辑元素数。 |
| `workspaceSize/executor` | 第一段输出 | `uint64_t*` / `aclOpExecutor**` | executor 保存已校验和已编排流程；Tiling 对 kernel workspace 固定写 32 B。 |
| `workspace/workspaceSize/executor/stream` | 第二段输入 | opaque | 在指定 stream 上执行 executor；workspace 所有权归调用者。 |

### 入口与版本

固定仓公开唯一版本族是 `aclnnForeachPowScalarAndTensorGetWorkspaceSize`/`aclnnForeachPowScalarAndTensor`。第一段负责参数检查、workspace 查询和 executor 生成，第二段接收 workspace 与 stream 执行。API 文档明确 A2、A3 支持；README、OpDef 的 `ascend910b`/`ascend910_93`、两个 binary config 和 Kernel CMake 彼此一致。

固定仓没有生成式 ACLNN C++ bridge 正文，因此 ACLNN 参数对象如何转换为内部 `ForeachPowScalarAndTensor(scalar, x, y)` 节点、DOUBLE scalar 如何窄化为 FLOAT32、executor 的具体 owner 实现属于固定仓不可见的框架边界；内部 Host/Tiling、POD、router/body 和构建接线均已继续追踪。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 API 第一段拒绝 `scalar/x/out` 空指针、非法 dtype 组合及 `x/out` dtype 不一致；Tiling/shape infer 失败映射为文档列出的 `ACLNN_ERR_INNER_TILING_ERROR`。
2. OpDef 将 `scalar` 声明为 Scalar，将 `x/y` 声明为 dynamic ND，并对 `scalar/x/y` 使用 `.AutoContiguous()`；公开语义允许非连续 `x`，不允许非连续 `out`。
3. `InferShape4ForeachPowScalarAndTensor` 要求 dynamic `x` 实例数等于输出数，并逐项复制 shape；`InferDataType4...` 逐项复制 `x` dtype。
4. `ForeachCommonTiling::Init(22, TYPE_SCALAR)` 从 dynamic input index 1 枚举最多 50 个 `x_i`，以 `GetInputDesc(1)` 冻结 dtype/元素字节数，累计每项 storage shape size 与总元素数。
5. `CheckOpInfo` 校验 TensorList 数量、每项维数、scalar storage size、输出 dtype/容量以及列表内 dtype 一致性；任何失败返回 `GRAPH_FAILED`，不启动 Kernel。

### 参数、Workspace 与生命周期

设 `U` 为平台 UB 字节数，`TD=ForeachCommonTilingData::GetDataSize()`，`d=sizeof(x dtype)`，`B=32/d` 个元素/32 B block，`N` 为 `GetTilingN()`（FP16/BF16 为 14，FP32/INT32 为 4）。Host 计算：

```text
total = sum_i numel(x_i)
blockDim = min(max(1, ceil(total / B)), AIV_core_count, 50)
reserve = 1024 * N * d
remain = U - TD - reserve
if dtype == BF16: remain = remain / 10
S = floor((remain / 4) / align) * align
align = 64 B (BF16), otherwise 32 B
```

`S` 写入 `inputsTensorUbSize`。每核区间按 `ceil(total/B)` 个 32 B block 均分，前 `blockCount % blockDim` 个核多一个 block；区间再转换为 `[tensorStart,tensorEnd]` 与首尾 Tensor 内元素 offset。

Tiling 固定令 `workspaces[0]=32`，但 kernel entry 将 `userWS=nullptr`，body 无 workspace consumer；这 32 B 是 Host 预留，不是跨 Kernel 数据边。workspace 由 ACLNN 调用者按第一段结果分配，在第二段调用期间有效。executor 和 stream 的具体释放/同步实现固定仓不可见。

### 启动图与失败传播

```text
aclnnForeachPowScalarAndTensorGetWorkspaceSize
  -> [生成式 ACLNN bridge，固定仓不可见]
  -> ForeachPowScalarAndTensor internal op
     -> InferShape/InferDataType
     -> Tiling4ForeachPowScalarAndTensorTiling
        -> ForeachCommonTiling::Init(22, TYPE_SCALAR)
        -> ForeachCommonTiling::RunBigKernelTiling

aclnnForeachPowScalarAndTensor(workspace, workspaceSize, executor, stream)
  -> 单 stream 启动 foreach_pow_scalar_and_tensor<<<blockDim>>>
     -> ForeachPowScalarAndTensorND<T>::Init/Process
     -> GM out
```

| 节点/边 | entry 与参数 | 输入/输出 | workspace/ordering/owner | 失败传播 |
|---|---|---|---|---|
| K1 单 AIV 节点 | `foreach_pow_scalar_and_tensor(scalar, inputs, outputs, workspace, tiling)`；`blockDim` 来自 Tiling，POD 为 `ForeachCommonTilingData` | GM scalar、TensorList 地址表/数据、POD → GM outputs | entry 明确不使用 workspace；唯一节点，无节点间边；第二段指定 stream 提供 launch ordering | Host 校验/Tiling 失败阻断 launch；router 未提供 unknown key 错误分支，合法 Host 只生成 key 1–4。 |

## Route 与参数设计

### 分核与资源计算

`elementsPerBlock=32/d`。`GetNeedCoreNum` 先按 `ceil(total/elementsPerBlock)` 得到所需 block 数，零总元素时强制至少 1 核，再与平台 AIV 核数和 POD 数组容量 50 取小。`AssignDataToEachCore` 用整 block 均分拼接后的总逻辑域，再以 `tensorDataCountList` 将每核范围映射回 Tensor 编号及首尾 offset；因此多数核为 32 B 整块域，列表边界和最后一核仍可形成非 32 B tail。

每个 Tensor 内再按 `maxDataCount=queue_bytes/d` 分 tile。完整 tile 走 `DataCopy`，最后不足 tile 的部分走 byte-length `DataCopyPad`。BF16 的 `maxDataCount=9S/2`，但升精度临时区每批最多处理 `maxCastDataCount=S/4` 元素，所以一个输入 tile 再被拆为多次 cast/Power/cast 子批。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 直接 Power | key 1 → `<half>`；key 2 → `<float>`；key 3 → `<int>` | `GetTilingKeyByDtypeOnly(dataType)` 依次映射 FP16/FP32/INT32；router 按 `if/else if` 检查 1、2、3 | key 4 进入 R-002；其它 key 无 body。 |
| 2 | R-002 BF16 升精度 Power | key 4 → `<bfloat16_t>` | Host dtype 为 BF16；router 在 A2/A3 编译时包含 key 4 分支 | 不存在下一 fallback；非法 dtype 在 OpDef/Host gate 失败。 |

```text
dataType = dtype(x[0])
if dataType == FP16: key=1 -> R-001<half>
else if dataType == FP32: key=2 -> R-001<float>
else if dataType == INT32: key=3 -> R-001<int>
else if dataType == BF16: key=4 -> R-002<bfloat16_t>
else: key=0; unsupported/Host failure（合法公开输入不可达）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 直接 Power | 1/2/3 | A2/A3；ND；FP16/FP32/INT32；列表内 dtype 一致；scalar 分别为 FLOAT32/FLOAT32/INT64 内部输入 | 单 `AIV_VECTOR` K1 | `ForeachPowScalarAndTensorND<half/float/int>` + 通用 `InnerComputer<T>::Compute` | VECIN `S` + VECOUT `S`，depth=1；Queue 事件隐式同步 Copy/Compute/CopyOut | 末 tile 用 `DataCopyPad`；不满足 dtype/shape 在 Host 失败。 |
| R-002 BF16 升精度 Power | 4 | A2/A3；ND；BF16；列表内 dtype 一致；scalar 内部输入 FLOAT32 | 单 `AIV_VECTOR` K1 | `ForeachPowScalarAndTensorND<bfloat16_t>` + 特化 `InnerComputer<bfloat16_t>` | VECIN `9S` + VECOUT `9S` + FP32 queue `2S`，均 depth=1；每个 cast/Power/cast 间 `PipeBarrier<PIPE_V>` | GM tail 用 `DataCopyPad`，每个 BF16 tile 内以 `S/4` 元素分段；不满足则 Host 失败。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `Init` 枚举 `i < MAX_TENSOR_CONT(50)`；blockDim 再限制 `<=MAX_CORE_CONT(50)` | POD 各列表固定 50 项 | 全部 | 固定仓没有显式“超过 50”错误；属于不可安全扩大边界 | `foreach_tiling_def.h::MAX_TENSOR_CONT/MAX_CORE_CONT`；`ForeachCommonTiling::Init` |
| 32 B 分核 | `B=32/d`，按 `ceil(total/B)` block 分核 | 完整 tile 的 `DataCopy` 需要整 block 路线，尾部另走 pad API | 全部 | 最后一段和 Tensor 边界由 tail 路线处理 | `GetNeedCoreNum`、`AssignDataToEachCore` |
| UB headroom | `U > TD + 1024*N*d` 是无下溢计算的必要条件；源码未显式检查 | Host 从无符号 `U` 中减 POD 与 Power 预留，再求 `S` | 全部 | 固定仓无 fallback；资源前提不成立不可复用该公式 | `DivideUbMemory6` |
| BF16 64 B 对齐 | BF16 的 `S=floor(canUse/64)*64` | BF16 升精度路径需要 9 倍输入/输出区及 FP32 双区，并以 `S/4` 元素切片 | R-002 | 非 BF16 使用 32 B 对齐的 R-001 | `DivideUbMemory6`、`ForeachPowScalarAndTensorND::Init` |
| BF16 编译 guard | key 4 仅在 `__NPU_ARCH__` 非 3003/3113 编译 | 这些 Kirin 架构不支持 BF16；A2/A3 key 4 可达 | R-002 | 被排除架构不能执行 key 4 | `foreach_pow_scalar_and_tensor` router |
| copy byte 长度窄化 | tail `dataCount*sizeof(T)` cast 到 `uint32_t` | 单 tile 由 UB 容量限制，正常远小于 4 GiB | 全部 | 固定仓无运行时 fallback | `CopyIn`、`ComputeAndCopyOut` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 生成序列化布局，固定仓未展开生成后的 C++ struct，因此 `sizeof/alignof/offsetof` 的数值不可直接验证；下表保留宏声明顺序、数组长度与 producer/consumer，不以普通 C++ 自然布局推测 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 第 1 字段；精确 offset/总 sizeof 固定仓不可见 | 字节 | `DivideUbMemory6` 计算 `S`；`FillTilingData::set_inputsTensorUbSize` | `ParseTilingData`；`InitBuffer`、`maxDataCount/maxCastDataCount` | 32 B 对齐；BF16 为 64 B 对齐；所有 Route 活跃 |
| `ForeachCommonTilingData` | `tensorDataCountList` | `int64_t[50]`/每项 64 | 第 2 字段；精确 offset 固定仓不可见 | 元素/每 Tensor | `Init`: `shape.GetShapeSize()`；`FillTilingData` | `Process` 计算每 Tensor 的尾坐标 | 最多 50 项；零元素允许 |
| `ForeachCommonTilingData` | `tensorStartList` | `uint16_t[50]`/每项 16 | 第 3 字段；精确 offset 固定仓不可见 | Tensor 索引/每 core | `AssignDataToEachCore`；`FillTilingData` | `ParseTilingData` 以 `blockIdx` 读取 | 仅 `[0,blockDim)` 活跃 |
| `ForeachCommonTilingData` | `tensorEndList` | `uint16_t[50]`/每项 16 | 第 4 字段；精确 offset 固定仓不可见 | Tensor 索引/每 core | `AssignDataToEachCore`；`FillTilingData` | `ParseTilingData`、`Process` 循环上界 | 含端点；仅活动 core 读取 |
| `ForeachCommonTilingData` | `tensorStartOffsetList` | `int64_t[50]`/每项 64 | 第 5 字段；精确 offset 固定仓不可见 | 元素 offset/每 core | `AssignDataToEachCore`；`FillTilingData` | `ParseTilingData`；首 Tensor 的 `cursorStart` | 含首元素，零起始 |
| `ForeachCommonTilingData` | `tensorEndOffsetList` | `int64_t[50]`/每项 64 | 第 6 字段；精确 offset 固定仓不可见 | 元素 offset/每 core | `AssignDataToEachCore`；`FillTilingData` | `ParseTilingData`；末 Tensor 的 `cursorEnd` | 含末元素；零元素时可为 `-1` 的等价空区间 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 直接 Power | key 1/2/3，`GetTilingKeyByDtypeOnly` | `foreach_tiling_func.cpp::Tiling4ForeachPowScalarAndTensorTiling`；`foreach_tiling_class.h::Init/RunBigKernelTiling/DivideUbMemory6` | `ForeachCommonTilingData` 六字段 | `foreach_pow_scalar_and_tensor.cpp::foreach_pow_scalar_and_tensor` → `<half/float/int>` | `foreach_pow_scalar_and_tensor.h::ForeachPowScalarAndTensorND`、`InnerComputer<T>` | A2 `ascend910b` JSON、A3 `ascend910_93` JSON、`op_kernel/CMakeLists.txt` |
| R-002 BF16 升精度 Power | key 4，`GetTilingKeyByDtypeOnly` | 同上，另含 BF16 `/10` 和 64 B 对齐 | 同一 POD；`S` 的 BF16 含义不变但 consumer 分配倍数不同 | 同一 entry → `<bfloat16_t>` | 同一 ND 类 + `InnerComputer<bfloat16_t>::Compute/ComputerPerCast` | 两个 JSON 均含 BF16 binary；同一 Kernel CMake target |

## Kernel 实现

### 公共执行流程

entry 用 `GET_TILING_DATA` 解析 POD，按 key 实例化一个 `ForeachPowScalarAndTensorND<T>`；workspace 被显式替换为 `nullptr`。`Init` 读取 `blockIdx` 对应的 Tensor 区间、从 GM scalar 取一个值、初始化 Queue；`Process` 逐 Tensor 绑定地址并按 UB tile 执行 `CopyIn → Power → CopyOut`。TensorList 不是连续拼接数据：`GetInputTensorAddr/GetOutputTensorAddr` 从输入对象首个 `uint64_t` 读取地址表相对 offset，再从表中取第 `i` 个实际 GM 地址。

### Route R-001：FP16/FP32/INT32 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachPowScalarAndTensor.md` | API 参数表 | `x/out` dtype 为 FP16/FP32/INT32 且一致 | 进入内部算子；不满足报参数错误 |
| Host/Tiling | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | FP16→1，FP32→2，INT32→3 | 产生 router key |
| Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling` | `CheckOpInfo()==GRAPH_SUCCESS` 后设置 key、blockDim、POD | 启动 K1；失败阻断 |
| Router | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.cpp` | `foreach_pow_scalar_and_tensor` | `TILING_KEY_IS(1/2/3)`，按源码顺序 | 实例化 `<half>/<float>/<int>` |

#### Device 数据流

##### 函数级调用链

```text
Host: Tiling4ForeachPowScalarAndTensorTiling
  -> ForeachCommonTiling::Init(22, TYPE_SCALAR)
  -> ForeachCommonTiling::RunBigKernelTiling
Host launch K1: foreach_pow_scalar_and_tensor(scalar, inputs, outputs, workspace, tiling)
  -> GET_TILING_DATA
  -> ForeachPowScalarAndTensorND<half|float|int>::Init
     -> ParseTilingData
     -> dataQueue/outQueue InitBuffer
  -> ForeachPowScalarAndTensorND<T>::Process
     -> GetInputTensorAddr/GetOutputTensorAddr
     -> SingleTensorProcess
        -> CopyIn
        -> ComputeAndCopyOut
           -> InnerComputer<T>::Compute
              -> Power<T, false>
        -> GM output
```

##### 源码等价伪码

```text
# 伪码；所有 offset 以 T 元素计
core = GetBlockIdx()
[t_begin,t_end,off_begin,off_end] = POD[core]
scalar_raw = GM_scalar[0]
scalarVal = cast<T>(scalar_raw)
S = POD.inputsTensorUbSize
maxElem = S / sizeof(T)
allocate VECIN[S, depth=1], VECOUT[S, depth=1]

for tensorId in inclusive(t_begin, t_end):
    begin = (tensorId == t_begin) ? off_begin : 0
    end   = (tensorId == t_end) ? off_end : numel[tensorId]-1
    count = end - begin + 1
    inGM  = inputAddressTable[tensorId] + begin
    outGM = outputAddressTable[tensorId] + begin
    full = count / maxElem; tail = count % maxElem
    for tile in [0, full):
        DataCopy(inQ, inGM + tile*maxElem, maxElem)
        Power<T,false>(outQ, scalarVal, inQ, maxElem)
        DataCopy(outGM + tile*maxElem, outQ, maxElem)
    if tail > 0:
        DataCopyPad(inQ, inGM + full*maxElem, bytes=tail*sizeof(T))
        Power<T,false>(outQ, scalarVal, inQ, tail)
        DataCopyPad(outGM + full*maxElem, outQ, bytes=tail*sizeof(T))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM scalar | base `scalar` | 1 个 `DTYPE_SCALAR` | FP32（FP16/FP32 route）或 INT64（INT32 route） | ACLNN/框架 | `Init::GetValue(0)` | 调用者；整个 launch 有效 | 每核读一次；FP16/INT32 转成 `T` |
| GM TensorList 地址表 | `base + ((*uint64_t(base)) >> 3)`，取 `[tensorId]` | 每项一个 64-bit GM 地址；表总长由列表数决定 | framework TensorList ABI | 框架 | `GetInputTensorAddr/GetOutputTensorAddr` | executor/框架；launch 期间有效 | 首字段是相对地址表 offset |
| GM input slice | `inputPtr[tensorId] + begin + tile*maxElem` | full=`maxElem`；tail=`tail` 元素 | `T`/ND contiguous storage | 调用者/AutoContiguous | `CopyIn` | 调用期间只读 | full 用 `DataCopy`，tail 用 byte `DataCopyPad` |
| `dataQueue` VECIN | Queue 管理 | `S` B × depth 1 | `T` | `CopyIn` | `InnerComputer<T>` | `Alloc→EnQue→DeQue→Free`/tile | 下个 tile 在 Free 后覆盖 |
| `outQueue` VECOUT | Queue 管理 | `S` B × depth 1 | `T` | `Power<T,false>` | `CopyOut` | `Alloc→EnQue→DeQue→Free`/tile | 直接物化最终输出，无额外 calc TBuf |
| GM output slice | `outputPtr[tensorId] + begin + tile*maxElem` | 与 input 有效元素数相同 | `T`/ND | `CopyOut` | 调用者 | 调用者长期持有 | `out` 容量多出的元素保持未写 |
| workspace | entry 接收但改为 `nullptr` | Host 报 32 B | N/A | ACLNN caller | 无 Device consumer | 第二段调用期间 | 不参与数据交接 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · VECIN · depth=1 · S bytes"]:1
  outQ["outQueue · VECOUT · depth=1 · S bytes"]:1
```

##### 字节账与复用顺序

```text
explicit_UB_total = dataQueue(1*S) + outQueue(1*S) = 2S
Host Power headroom = reserve = 1024*N*d
S = floor(((U - TD - reserve)/4)/32)*32
explicit_UB_total + reserve <= U - TD   # 由 /4 留有保守余量
dynamic_ub = N/A（非 SIMT）
workspace_total = 32 B reserved；Device body 使用 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` 唯一槽 | `CopyIn` | `InnerComputer<T>::Compute` | 下一 tile `AllocTensor` | `EnQue→DeQue→FreeTensor` | 必须在 Power 读取结束后 Free |
| `outQueue` 唯一槽 | `Power<T,false>` | GM `DataCopy/DataCopyPad` | 下一 tile `AllocTensor` | `EnQue→DeQue→FreeTensor` | 必须在 CopyOut 完成后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | VECIN Queue `EnQue/DeQue` | `DataCopy`/`DataCopyPad` | `ComputeAndCopyOut` DeQue | 当前 input tile | full 为 `maxElem`；tail 仅搬 `dataCount*sizeof(T)` |
| 求幂 | `Power<T,false>` | input Queue | output Queue EnQue | 当前有效 `dataCount` lanes | FP16 以 half scalar/half route，FP32 以 float，INT32 以 int；primitive 内算法固定仓不可见 |
| UB→GM | VECOUT Queue `EnQue/DeQue` | Power | `DataCopy`/`DataCopyPad` | 当前 output tile | tail 只写有效字节，不写 padding/额外 out 容量 |
| 跨核 | N/A | 各核独立不重叠区间 | N/A | GM output disjoint slices | 无 atomic、event 或跨核同步 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling`、`AssignDataToEachCore`、`DivideUbMemory6` | 生成 key/blockDim、core 区间和 `S` |
| Kernel entry/router | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.cpp` | `foreach_pow_scalar_and_tensor` key 1/2/3 | key 映射 `<half>/<float>/<int>`；workspace 不传 body |
| Device body/helper | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h` | `ForeachPowScalarAndTensorND<T>`、`InnerComputer<T>::Compute` | 地址表解析、tile 循环、`Power<T,false>` 和 GM tail |

### Route R-002：BF16 分段升精度 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachPowScalarAndTensor.md` | API 参数表 | A2/A3 上 `x/out` BF16，scalar FLOAT32/DOUBLE | 进入内部算子；DOUBLE bridge 细节不可见 |
| Host/Tiling | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `GetTilingN/DivideUbMemory6` | BF16：`N=14`、`remain/=10`、`S` 下取整到 64 B | 生成 BF16 资源上界 |
| Router | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.cpp` | `foreach_pow_scalar_and_tensor` | `TILING_KEY_IS(4)` 且非 arch 3003/3113 | `<bfloat16_t>`；A2/A3 可达 |

#### Device 数据流

##### 函数级调用链

```text
Host launch K1: foreach_pow_scalar_and_tensor(..., key=4)
  -> ForeachPowScalarAndTensorND<bfloat16_t>::Init
     -> dataQueue[9S], outQueue[9S], float32Queue[2S]
  -> Process
     -> DeQue persistent float32Tensor
     -> per Tensor: SingleTensorProcess
        -> CopyIn
        -> ComputeAndCopyOut
           -> InnerComputer<bfloat16_t>::Compute
              -> repeated ComputerPerCast
                 -> Cast(BF16 -> FP32)
                 -> Power<float, false>
                 -> Cast(FP32 -> BF16, CAST_RINT)
           -> CopyOut
     -> FreeTensor(float32Tensor)
```

##### 源码等价伪码

```text
# 伪码；S=POD.inputsTensorUbSize
allocate inQ[9S], outQ[9S], fp32Q[2S], all depth=1
maxElem = 9S / 2                 # BF16 elements per outer tile
castElem = S / 4                 # FP32 elements per inner batch
fp32 = DeQue(fp32Q)              # [0,castElem) input; [castElem,2*castElem) output
scalarVal = float(GM_scalar[0])

for each assigned Tensor slice:
  for each outer tile with valid count:
    copy BF16 GM -> inQ
    q = count / castElem; r = count % castElem
    for k in [0,q) plus optional remainder:
      n = (k<q) ? castElem : r
      PipeBarrier<V>()
      Cast(fp32[0], inQ[k*castElem], CAST_NONE, n)
      PipeBarrier<V>()
      Power<float,false>(fp32[castElem], scalarVal, fp32[0], n)
      PipeBarrier<V>()
      Cast(outQ[k*castElem], fp32[castElem], CAST_RINT, n)
      PipeBarrier<V>()
    copy valid BF16 outQ -> GM
free fp32 after all Tensor slices
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM scalar | base `scalar` | 1 FP32 | FP32 | ACLNN bridge | `Init::GetValue(0)` | launch 期间 | 保持 FP32，不先降为 BF16 |
| GM input/output | 与 R-001 相同地址表公式 | 每 outer tile 最多 `9S/2` 个 BF16 | BF16/ND | caller/Kernel | Kernel/caller | launch 期间/长期 | outer tail 用 byte `DataCopyPad` |
| `dataQueue` | Queue 管理 | `9S` B × depth 1 | BF16 | CopyIn | BF16→FP32 Cast | outer tile | 64 B 基准 `S`，Free 后复用 |
| `outQueue` | Queue 管理 | `9S` B × depth 1 | BF16 | FP32→BF16 Cast | CopyOut | outer tile | 直接承载最终 outer tile |
| `float32Queue` 物理槽 | Queue 管理；前半 `[0,S)`、后半 `[S,2S)` | `2S` B × depth 1 | FP32 | 前半由 Cast 写；后半由 Power 写 | Power/降精度 Cast | `Init` 后 DeQue，整个 `Process` 常驻，末尾 Free | 两个不重叠子区；跨 inner batch 原地覆盖 |
| workspace | 不读 | Host 32 B | N/A | caller | 无 | 第二段期间 | 非数据边 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · VECIN · depth=1 · 9S bytes"]:1
  fIn["float32Queue[0:S) · S bytes"]:1
  fOut["float32Queue[S:2S) · S bytes"]:1
  outQ["outQueue · VECOUT · depth=1 · 9S bytes"]:1
```

##### 字节账与复用顺序

```text
explicit_UB_total = dataQueue(9S) + outQueue(9S) + float32Queue(2S) = 20S
reserve = 1024*14*2
remain0 = U - TD - reserve
S = floor(((remain0/10)/4)/64)*64 = floor(remain0/40/64)*64
explicit_UB_total <= remain0/2 <= U - TD - reserve
dynamic_ub = N/A（非 SIMT）
workspace_total = 32 B reserved；Device body 使用 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 前半 `[0,S)` | BF16→FP32 `Cast` | `Power<float,false>` | 下一 `ComputerPerCast` | Cast 后、Power 前 `PipeBarrier<PIPE_V>` | `n<=S/4` FP32 元素 |
| FP32 后半 `[S,2S)` | `Power<float,false>` | FP32→BF16 `Cast` | 下一 `ComputerPerCast` | Power 后、Cast 前 `PipeBarrier<PIPE_V>` | 前后半不可重叠；base offset=`maxCastDataCount` FP32 元素 |
| `outQueue` 当前片段 | 降精度 Cast | CopyOut | 下一 outer tile | 每次 Cast 后 barrier；Queue EnQue/DeQue/Free | 所有 inner batch 完成后才 EnQue/CopyOut |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `Cast(..., CAST_NONE, n)` + `PipeBarrier<PIPE_V>` | input Queue | FP32 Power | FP32 前半 | `n=min(S/4, remaining)`，不处理无效 lanes |
| FP32 求幂 | `Power<float,false>(dst=fp32[S], scalarVal, src=fp32[0], n)` + barrier | FP32 前半 | 降精度 Cast | FP32 后半 | scalar 和指数均以 FP32 参与核心 Power |
| FP32→BF16 | `Cast(..., CAST_RINT, n)` + barrier | FP32 Power | output Queue/CopyOut | BF16 output 片段 | 显式 round-to-integral mode `CAST_RINT`；具体硬件舍入细节由 primitive 定义 |
| outer GM tail | `DataCopyPad` | GM/UB | Cast 或 GM output | 当前有效 BF16 字节 | 仅 `count*2` B；内层 remainder 仅处理 `r` 元素 |
| 跨核 | N/A | 各核 disjoint | N/A | GM output | 无 atomic/跨核同步 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `GetTilingN`、`DivideUbMemory6` | BF16 N=14、UB `/10`、64 B 对齐 |
| Kernel entry/router | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.cpp` | key 4 branch | A2/A3 实例化 `<bfloat16_t>`，仅 Kirin arch 宏排除 |
| Device body/helper | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h` | BF16 `Init`、`InnerComputer<bfloat16_t>::Compute/ComputerPerCast` | 9S/9S/2S 分配、分段 cast/Power/cast、四个 V pipe barrier |

## 算子特定优化

### Trick T-001：TensorList 拼接域按 32 B block 均分并回映 Tensor 边界

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：`RunBigKernelTiling` 通过检查后，对所有 Tensor 的 `totalDataCount` 分核。
- Golden 来源：[`foreach/foreach_utils/op_host/foreach_tiling_class.h:649-695`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L649-L695)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
int64_t remainderCount = blockCount % needCoreNum;
// ...
if (remainderCount && static_cast<int64_t>(coreIndex) < remainderCount) {
    curCmpCount = tempPerCoreCount + static_cast<int64_t>(elementsPerBlock);
} else {
    curCmpCount = tempPerCoreCount;
}
```

#### 机制与设计原因

Host 不以 Tensor 个数静态分核，而先把所有 Tensor 视为一个拼接元素域，以 `32/d` 个元素为 block 均分，再输出首尾 Tensor 及 offset。这样大 Tensor 可跨核、小 Tensor 可由一个核连续处理，完整块继续满足 `DataCopy` 的 32 B 粒度；Tensor 边界和最终余数由 `DataCopyPad` 收口。分核只改变调度和搬运粒度，不改变元素运算顺序。

#### 迁移边界

- 适用：TensorList 各项 dtype 相同、元素级独立、输出切片互不重叠的单 Kernel Vector 算子。
- 前提：列表不超过 50 项；POD 能表达每核首尾 Tensor/offset；完整块 32 B；tail 有 byte-length copy 路线。
- 失效：需要跨元素归约、跨 Tensor 状态或输出别名时不能直接复用；固定实现没有另一 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：BF16 以一个双区 FP32 Queue 分段升精度计算

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：模板类型为 `bfloat16_t`，每个 inner batch 不超过 `maxCastDataCount=S/4`。
- Golden 来源：[`foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h:70-80`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h#L70-L80)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
Power<float, false>(float32Tensor[maxCastDataCount], scalarVal, float32Tensor, dataCount);
PipeBarrier<PIPE_V>();
Cast(outLocal[index * maxCastDataCount], float32Tensor[maxCastDataCount], RoundMode::CAST_RINT, dataCount);
PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

同一个 `float32Queue` 的前半承载升精度指数，后半承载 FP32 Power 结果；每一批依次覆盖这两个区域，避免为每个 outer BF16 tile 物化一个同尺寸 FP32 Tensor。`maxCastDataCount=S/4` 同时决定前半容量和后半 base offset，四个 V pipe barrier 保证写后读与下一批覆盖顺序。Power 在 FP32 中执行，最终以 `CAST_RINT` 降回 BF16，精度顺序是源码硬约束。

#### 迁移边界

- 适用：低精度输入需要 FP32 primitive、且计算可按独立连续片段执行的 Vector Kernel。
- 前提：分配 `2S` FP32 空间；inner count 不超过 `S/4`；前后半不别名；保留 V pipe barriers 和最终 round mode。
- 失效：若 primitive 需要跨片段状态、输出扩张或前后区容量不同，必须重算布局，不能沿用该 offset。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 直接 Power | R-002 BF16 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 1/2/3；FP16/FP32/INT32 | key 4；BF16 | Host dtype→key，router 同序匹配 |
| 模板与 dtype | `<half>/<float>/<int>`；scalar 先转为 T | `<bfloat16_t>`；scalar 保持 float | BF16 使用特化 `InnerComputer` |
| 循环与分核 | 公共 32 B 拼接域分核；每 tile 最多 `S/d` | 同一分核；outer tile `9S/2` BF16，再按 `S/4` 分 inner batch | BF16 需要 FP32 双区复用 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | VECIN S + VECOUT S；无 TBuf/dynamic UB/window | VECIN 9S + VECOUT 9S + FP32 Queue 2S；无 TBuf/dynamic UB/window | 源码 `InitBuffer` 条件分配 |
| 内存总占用与复用 | 显式 2S；逐 tile Queue 槽复用 | 显式 20S；FP32 前后半逐 inner batch 覆盖 | Host BF16 `/10` 收缩后再 `/4` |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopy/DataCopyPad` + `Power<T,false>` | 同上 + 两次 Cast + V barriers | 仅 AIV Vector，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue 同步；以目标 T 直接 Power；GM tail pad | Queue + V barrier；BF16→FP32 Power→RINT BF16；双层 tail | 模板类型与 BF16 特化决定 |
| fallback/失败边界 | Host 非法输入失败；unknown key 无 body | BF16 编译 guard；Host 非法输入失败 | 无 Device fallback Kernel |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；公开文档 √ | 支持；公开文档 √ | 公开 README/API 文档明确 ×；虽然 OpDef/config/CMake 有 `ascend950` 接线，构成发布冲突，本卡不声明支持。 |
| Host/Launcher | `ascend910b` binary config；共用 `ForeachCommonTiling` | `ascend910_93` binary config；共用同一 Tiling | 实现信号存在，但不得覆盖公开“不支持”结论。 |
| Kernel/资源 | 同一 key 1–4 entry/body，`AUTO_SYNC false` | 同一 key 1–4 entry/body，`AUTO_SYNC false` | 不在本卡闭包范围；未发布 `_950` 卡。 |

## 限制与失败边界

- 公开 API 文档支持 `scalar=DOUBLE` 搭配浮点 `x`，而固定仓 OpDef/binary 只列内部 FLOAT32 scalar；生成式 ACLNN bridge 的 DOUBLE→FLOAT32 转换、溢出/舍入检查固定仓不可见。
- `ForeachCommonTilingData` 由宏生成；固定仓没有展开后的 `sizeof/alignof/offsetof`，不得凭普通 C++ 布局重建序列化 ABI。
- 最多 50 个 Tensor、最多 50 个活动 core 由固定 POD 数组决定；超过 50 个 Tensor 的显式拒绝路径固定仓不可见。
- Host 的 UB 公式使用无符号减法，源码没有在 `U<=TD+reserve` 时提供 guard/fallback；迁移时必须保证资源前提。
- Kernel router 对 unknown key 没有显式错误返回或 fallback；合法公开 dtype 经 Host 只产生 1–4。
- `Power` AscendC primitive 的内部逼近算法、临时资源和特殊值细节属于固定仓外工具链实现；本卡只承诺调用 dtype、参数顺序与 BF16 外围 cast 顺序。
- API 文档称默认确定性实现；源码可证明各核输出区间不重叠且无 atomic，但未执行真机确定性验证。
- 950 文档与实现接线冲突，按平台 gate 不生成正式 950 卡。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_pow_scalar_and_tensor/docs/aclnnForeachPowScalarAndTensor.md` | 两段式函数原型、产品支持表 | 公开入口、参数、错误与 A2/A3/950 支持结论 |
| 入口声明/约束 | `foreach/foreach_pow_scalar_and_tensor/README.md` | 产品支持、功能、参数表 | 逻辑语义、950 不支持冲突 gate |
| 入口实现 | `foreach/foreach_pow_scalar_and_tensor/op_host/foreach_pow_scalar_and_tensor_def.cpp` | `ops::ForeachPowScalarAndTensor` | 内部输入顺序、dtype/layout、AutoContiguous、SoC config |
| 入口实现 | `foreach/foreach_pow_scalar_and_tensor/op_host/foreach_pow_scalar_and_tensor_infershape.cpp` | `InferShape4...`、`InferDataType4...` | TensorList 数量、shape 与 dtype 推导 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachPowScalarAndTensorTiling` | op code 22、TYPE_SCALAR、Tiling 注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace、POD producer |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData`、注册宏 | POD 字段与数组上限 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key 1/2/3/4 |
| Kernel entry/body | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.cpp` | `foreach_pow_scalar_and_tensor` | entry、workspace 边界与 key router |
| Kernel entry/body | `foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h` | `ForeachPowScalarAndTensorND`、`InnerComputer` | 全部 Device 循环、地址、Queue、Power/cast/tail |
| 构建接线 | `foreach/foreach_pow_scalar_and_tensor/op_host/config/ascend910b/foreach_pow_scalar_and_tensor_binary.json` | `ForeachPowScalarAndTensor` op list | A2 四个 dtype binary 实例 |
| 构建接线 | `foreach/foreach_pow_scalar_and_tensor/op_host/config/ascend910_93/foreach_pow_scalar_and_tensor_binary.json` | `ForeachPowScalarAndTensor` op list | A3 四个 dtype binary 实例 |
| 构建接线 | `foreach/foreach_pow_scalar_and_tensor/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 Kernel target 与 `AUTO_SYNC false` |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段可表达 TensorList 拼接域的每核区间 | 字段顺序、类型、50 项数组、元素/字节单位；精确生成 ABI 需官方宏工具 | `ForeachCommonTilingData`、`FillTilingData`、`ParseTilingData` |
| 调用顺序 | 单 Kernel `Init→Process→per Tensor→per tile` | TensorList 地址表解析、首尾含端点语义、先 CopyIn 后 Power 后 CopyOut | `foreach_pow_scalar_and_tensor`、`Process` |
| workspace/内存 | Device 无 workspace；R-001 显式 2S，R-002 显式 20S | Host 仍上报 32 B；BF16 9S/9S/2S 倍数、对齐和 lifetime 不可改 | `RunBigKernelTiling`、`Init` |
| 同步 | Queue 生命周期同步通用流水；BF16 用 V pipe barriers | BF16 每个 cast/Power/cast 边界及下一批覆盖前的 barrier 必须保留 | `ComputeAndCopyOut`、`ComputerPerCast` |
| 精度与尾块 | dtype key 决定直接或升精度路线；GM tail 用 byte copy | BF16 的 FP32 Power 和 `CAST_RINT` 顺序、仅写有效元素 | `InnerComputer<T>`、`InnerComputer<bfloat16_t>`、`CopyIn/CopyOut` |
| 失败边界 | Host 检查失败阻断 launch | 不得为 unknown key、>50 Tensor、UB 下溢虚构 fallback | `CheckOpInfo`、router、`DivideUbMemory6` |
