---
type: Operator
title: "ForeachPowList A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachPowList 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/docs/aclnnForeachPowList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_host/foreach_pow_list_def.cpp
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
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_kernel/foreach_pow_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_kernel/foreach_pow_list.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_host/config/ascend910b/foreach_pow_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_host/config/ascend910_93/foreach_pow_list_binary.json
    role: implementation
created_at: "2026-08-25T15:48:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachPowList A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachPowListGetWorkspaceSize` + `aclnnForeachPowList`，公开两段式 ACLNN；固定仓未见版本后缀入口。 |
| 计算语义 | 对等长 TensorList 的对应 Tensor 逐元素计算 `out[i][j] = x1[i][j] ^ x2[i][j]`。 |
| 输入输出 | `x1/x2/out` 均为 ND TensorList；FP16、FP32、INT32、BF16；单 Tensor rank 0–8；输入可非连续，输出必须连续。 |
| 执行类型 | 单 Kernel `foreach_pow_list`，全部 Route 均为 `AIV_VECTOR`；无 Cube/AIC/MIX_CV 可达节点。 |
| Kernel 路线 | R-001 FP16、R-002 FP32、R-003 INT32、R-004 BF16 升 FP32 计算。 |
| 核心资源 | TensorList GM 指针表；三组 depth=2 的 `VECIN/VECOUT` Queue；一组 Power 临时 Queue；BF16 另有 FP32 Queue；算子预留 32 B workspace 但 Kernel 不读取。 |
| 来源闭包 | `source_closed`：公开 ACLNN→生成 bridge 边界→OpDef/InferShape→公共 Tiling/POD→dtype router→AIV body→A2/A3 CMake/binary config 已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x1=[x1_0,...,x1_(n-1)]`、`x2=[x2_0,...,x2_(n-1)]`，输出有效域为每个输入 Tensor 的 storage element count，逐元素计算 `y_i[j]=pow(x1_i[j],x2_i[j])`。FP16、FP32、INT32 Route 直接实例化 `Power<T,false>`；BF16 Route 明确先把两路输入分段 cast 为 FP32，再执行 `Power<float,false>`，最后按 `CAST_RINT` 降回 BF16。不同列表位置及元素间没有归约或状态依赖。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；ND；单 Tensor rank 0–8 | 列表内统一 FP16/FP32/INT32/BF16 | 非空指针；支持零元素 Tensor；与 `x2` 对应 shape 相同；OpDef `AutoContiguous()`；固定 Tiling 最多保存 50 个 Tensor。 |
| `x2` | TensorList；ND；对应 `x1` shape | 与 `x1` 完全一致 | 非空指针；指数绝对值公开约束为 `<=2^31-1`；支持输入非连续，经框架 contiguous 边界后进入算子。 |
| `out` | TensorList；ND；对应项容量不小于 `x1` | 与 `x1` 完全一致 | 非空指针；公开接口不支持非连续输出；Kernel 只写 `x1` 对应项的有效元素数。 |
| `workspaceSize`/`executor` | 第一段输出 | `uint64_t*`/`aclOpExecutor**` | 执行器包含前处理和算子 launch；算子 Tiling 固定写 workspace 32 B，executor 因 contiguous 产生的总 workspace 公式固定仓不可见。 |
| `workspace`/`workspaceSize`/`stream` | 第二段输入 | opaque/`uint64_t`/stream | 调用者使用第一段结果；Kernel entry 保留 workspace 形参，但主动置 `userWS=nullptr`，Device body 不读取 workspace。 |

### 入口与版本

固定仓公开的最新且唯一版本是无后缀 `aclnnForeachPowListGetWorkspaceSize`/`aclnnForeachPowList`。公开产品表明确支持 Atlas A2 与 A3；OpDef 同时注册 `ascend910b`、`ascend910_93`，两套 binary config 均发布 FP16/FP32/INT32/BF16，Kernel CMake 将二者接到同一源码，因此本卡覆盖 A2/A3。公开 ACLNN C header 与 bridge/executor 实体由构建框架生成，固定仓仅保存接口文档及 `ACLNNTYPE aclnn` 接线，作为允许的生成框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段 ACLNN 按公开约束检查 `x1/x2/out` 空指针、四种 dtype、三者 dtype 一致、两输入列表长度与对应 shape、列表内部 dtype、输出容量和 rank≤8；分别可能返回 `ACLNN_ERR_PARAM_NULLPTR`、`ACLNN_ERR_PARAM_INVALID` 或 `ACLNN_ERR_INNER_TILING_ERROR`。
2. OpDef 将两路动态输入与动态输出声明为 ND、四种 dtype，并设置 `AutoContiguous()`；输入非连续转换与 executor 节点编排属于固定仓不可见的生成 bridge。
3. `InferShape4ForeachCommon` 要求动态输入 0 的实例数等于输出数，逐项复制 `x1` shape；`InferDataType4ForeachCommon` 逐项复制 `x1` dtype。
4. `Tiling4ForeachPowListTiling` 构造 `ForeachCommonTiling`，以 `FOREACH_POW_TENSOR_OP_CODE=8` 初始化并调用 `RunBigKernelTiling`。`Init` 最多扫描 50 个 `x1` Tensor，累加 storage element count；`CheckOpInfo` 再验证两输入列表实例数、对应 shape、各 Tensor rank、列表内/列表间 dtype 和输出容量。
5. 任一 desc/shape/compute-node/workspace 指针为空、校验失败或 POD 保存失败会返回 `GRAPH_FAILED`，阻断 Kernel launch。

### 参数、Workspace 与生命周期

令总元素数 `N=sum_i n_i`，dtype 字节数 `S`，每 32 B 元素数 `E32=32/S`，平台 AIV 核数 `Cplat`。Host 取 `B=min(Cplat,ceil(N/E32))`，`N=0` 时强制 `B=1`，再限制 `B<=50`。它按拼接后的 TensorList 元素域、以 32 B block 数均分：基础份额 `floor(ceil(N/E32)/B)*E32`，前 `ceil(N/E32)%B` 个 core 多 `E32` 个元素，并将每核区间反解为 `tensorStart/End` 与首尾 offset。

Host 从平台 UB 字节数 `U` 中扣除 POD 序列化区 `P=tilingData.GetDataSize()`。令写入 POD 的基础字节片 `Q=inputsTensorUbSize`：FP16 为 `floor32((U-P)/18)`，FP32 为 `floor32((U-P)/9)`，INT32 为 `floor32((U-P)/11)`，BF16 为 `floor64((U-P)/63)`。`floor32/floor64` 表示向下对齐到 32/64 B。算子节点 workspace 固定为 32 B，由 executor/framework 持有到 stream launch 完成；Kernel 明确不消费它。TensorList 指针描述、各 GM Tensor 与 POD 同样必须存活至 K1 完成。

### 启动图与失败传播

```text
aclnnForeachPowListGetWorkspaceSize(x1,x2,out)
  -> 生成 bridge：校验/输入 Contiguous/构建 ForeachPowList executor
  -> InferShape/InferDataType
  -> Tiling4ForeachPowListTiling
       -> key=dtype(1/2/3/4), blockDim=B, POD=ForeachCommonTilingData, workspace[0]=32
aclnnForeachPowList(workspace,workspaceSize,executor,stream)
  -> K1: foreach_pow_list<<<B,stream>>>(x1List,x2List,outList,workspace,tiling)
       -> ForeachPowList<T>::Init/Process
```

这是单 Device 节点图，不存在 Kernel 间 workspace/GM 边。K1 输入为三份 TensorList GM 描述及 POD，输出直接写 `out` GM；同一 stream 承载 launch。Tiling 失败阻断 K1；entry 对 key 1–4 分流，未知 key 没有 `else` 错误分支而是静默不执行，但正常 Host 只会为四种已校验 dtype 生成这些 key。生成 bridge 的精确 launch 返回包装固定仓不可见。

## Route 与参数设计

### 分核与资源计算

每核以 POD 的 `[tensorStart,tensorStartOffset]..[tensorEnd,tensorEndOffset]` 遍历一个跨 Tensor 的连续切片。对每个 Tensor 片段，`SingleTensorProcess` 以 `M=maxDataCount` 分 tile；完整 tile 使用 `DataCopy(...,M)`，末 tile 使用单 block 的 `DataCopyPad` 参数 `{blockCount=1, blockLen=dataCount*S bytes, srcStride=0,dstStride=0}`，Vector `Power` 仅处理真实 `dataCount`。FP16/FP32/INT32 的 `M=Q/S`；BF16 扩大搬运 tile 为 `M=9Q/2` 元素，同时以 `C=Q/4` 元素为一次 FP32 cast/Power 子块。

三组主 Queue 都是 depth 2。非 BF16 的真实 UB 为 FP16 `6Q+12Q=18Q`、FP32 `6Q+3Q=9Q`、INT32 `6Q+5Q=11Q`。BF16 主 Queue 为 `3*2*(9Q)=54Q`，FP32 Queue 为 `3Q`，Power 临时 Queue 为 `3Q`，真实分配 `60Q`；Host 用 `63Q<=U-P` 作保守上界。固定源码没有在 `Q==0` 时拒绝 launch，依赖受支持平台提供足够 UB。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | key 1，`ForeachPowList<half>` | `dataType=DT_FLOAT16`，且公共 shape/dtype/output 校验成功 | dtype 不同继续 key 映射；校验失败终止 |
| 2 | R-002 FP32 | key 2，`ForeachPowList<float>` | `dataType=DT_FLOAT`，同上 | 同上 |
| 3 | R-003 INT32 | key 3，`ForeachPowList<int>` | `dataType=DT_INT32`，同上 | 同上 |
| 4 | R-004 BF16 | key 4，`ForeachPowList<bfloat16_t>` | `dataType=DT_BF16`，且非 NPU arch 3003/3113；A2/A3 config 发布 BF16 | unsupported/unknown key 静默无 body |

```text
validate tensor lists/output
  -> dtype FP16  -> key 1 -> R-001
  -> dtype FP32  -> key 2 -> R-002
  -> dtype INT32 -> key 3 -> R-003
  -> dtype BF16  -> key 4 -> R-004
  -> otherwise   -> OpDef/Host failure（若绕过则 key 0，entry 无动作）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | A2/A3；两输入/输出统一 FP16；列表数、对应 shape、rank、输出容量通过公共检查 | 单 K1 `AIV_VECTOR` | `ForeachPowList<half>`→generic `InnerComputer<half>`→`Power<half,false>` | `18Q` UB；三组 depth-2 Queue + `12Q` Power temp；Queue 同步 | 完整 tile `DataCopy`，尾 tile exact-byte `DataCopyPad`；无 Device fallback |
| R-002 FP32 | 2 | 同上，统一 FP32 | 单 K1 `AIV_VECTOR` | `ForeachPowList<float>`→`Power<float,false>` | `9Q` UB；Power temp `3Q` | 同上 |
| R-003 INT32 | 3 | 同上，统一 INT32 | 单 K1 `AIV_VECTOR` | `ForeachPowList<int>`→`Power<int,false>` | `11Q` UB；Power temp `5Q` | 同上；整数 Power 边界由 primitive 定义，固定仓不可见 |
| R-004 BF16 | 4 | 同上，统一 BF16；A2/A3 编译 guard 可达 | 单 K1 `AIV_VECTOR` | `ForeachPowList<bfloat16_t>`→specialized `InnerComputer`→cast/`Power<float,false>`/cast | 实配 `60Q`，Host 保守预算 `63Q`；Queue + `PIPE_V` barrier | 外层 tile exact tail；内部按 `Q/4` 元素分 cast 子块；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor/core 上限 | `totalTensorCount<=50` 由扫描数组容量隐式限制；`B<=50` | POD 数组分别只有 50 个 Tensor/core 槽 | 全部 | 第 51 个及以后不会进入 `Init` 扫描；公开 ACLNN 未单列该上限 | `foreach_tiling_def.h::MAX_TENSOR_CONT/MAX_CORE_CONT`；`ForeachCommonTiling::Init` |
| 32 B 分核粒度 | `E32=32/S`，`B=min(Cplat,ceil(N/E32),50)` | 以 DMA block 粒度控制每核覆盖量 | 全部 | `N=0` 强制 1 core | `ForeachCommonTiling::GetNeedCoreNum/AssignDataToEachCore` |
| dtype UB 分母 | FP16/FP32/INT32 分母 `18/9/11`；BF16 分母 `63` | depth-2 输入/输出 Queue 与 Power 临时区；BF16 还需 FP32 staging | 分别 R-001..R-004 | 没有小 UB fallback；`Q` 可能为 0 | `ForeachCommonTiling::DivideUbMemory2` |
| 搬运对齐 | `Q` 对齐 32 B，BF16 对齐 64 B；完整 tile `M`，尾 tile走 `DataCopyPad` | 完整 DMA 按对齐容量；尾部按字节复制避免越界 | 全部 | 尾块切换 padded API，不切 Route | `DivideUbMemory2`；`ForeachPowList::CopyIn/CopyOut` |
| BF16 编译 guard | `key=4` 且 `__NPU_ARCH__` 非 3003/3113 | 这两个 arch 不实例化 BF16 body | R-004 | entry 无 BF16 分支；A2/A3 不命中该排除 | `foreach_pow_list.cpp::foreach_pow_list` |
| offset 位宽 | tile `offset` 为 `uint32_t`，POD Tensor offset 为 `int64_t` | `SingleTensorProcess` 的 `i*M` 会窄化到 32 位 tile offset | 全部 | 固定 Host 未检查单核单 Tensor 片段是否超过 32 位 offset | `ForeachPowList::SingleTensorProcess/CopyIn/CopyOut` |

### POD/ABI

`BEGIN_TILING_DATA_DEF`/`TILING_DATA_FIELD_DEF(_ARR)` 的具体展开来自仓外注册头，固定仓无法可靠给出生成 C++ 对象的 `sizeof/alignof/offsetof`；下表保留声明顺序、元素类型与可见容量，不猜测 padding。所有数组仅 active 前缀会被 Host 写入/Device 读取。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 第 1 字段；具体 offset/总大小固定仓不可见 | bytes/每个基础片 `Q` | `DivideUbMemory2`→`FillTilingData::set_inputsTensorUbSize(Q)` | `ParseTilingData`；`InitBuffer`、`maxDataCount` | 32 B 对齐；BF16 为 64 B 对齐；无显式非零 guard |
| 同上 | `tensorDataCountList` | `int64_t[50]`/每项64 | 第 2 字段；具体 offset 固定仓不可见 | elements/Tensor | `Init`: `storageShape.GetShapeSize()` | `Process`: 取当前 Tensor 总元素数 | active `[0,totalTensorCount)`；空 Tensor 为 0 |
| 同上 | `tensorStartList` | `uint16_t[50]`/每项16 | 第 3 字段；具体 offset 固定仓不可见 | Tensor index | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | active `[0,B)`；最多索引 49 |
| 同上 | `tensorEndList` | `uint16_t[50]`/每项16 | 第 4 字段；具体 offset 固定仓不可见 | Tensor index（含端点） | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | active `[0,B)` |
| 同上 | `tensorStartOffsetList` | `int64_t[50]`/每项64 | 第 5 字段；具体 offset 固定仓不可见 | elements/Tensor | `AssignDataToEachCore` | `Process` 的 `cursorStart` 与 GM base | active `[0,B)`；首核通常为 0 |
| 同上 | `tensorEndOffsetList` | `int64_t[50]`/每项64 | 第 6 字段；具体 offset 固定仓不可见 | elements/Tensor，含端点 | `AssignDataToEachCore`: `cursorPosition-1` 或末 Tensor `n-1` | `Process`: `dataCount=endOffset+1-cursorStart` | 空切片可为 `-1` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 1 | `foreach_tiling_func.cpp::Tiling4ForeachPowListTiling`；`foreach_tiling_class.h::RunBigKernelTiling` | `ForeachCommonTilingData`，`Q/arrays` | `foreach_pow_list.cpp::foreach_pow_list`→`ForeachPowList<half>` | `foreach_pow_list.h::Init/Process/InnerComputer<T>::Compute` | op_host/op_kernel CMake；ascend910b/ascend910_93 binary JSON |
| R-002 FP32 | key 2 | 同上 | 同上，`Q=floor32((U-P)/9)` | entry→`ForeachPowList<float>` | generic `Power<float,false>` body | 同上 |
| R-003 INT32 | key 3 | 同上 | 同上，`Q=floor32((U-P)/11)` | entry→`ForeachPowList<int>` | generic `Power<int,false>` body | 同上 |
| R-004 BF16 | key 4 | 同上 | 同上，`Q=floor64((U-P)/63)` | entry guard→`ForeachPowList<bfloat16_t>` | specialized `InnerComputer<bfloat16_t>::Compute/ComputerPerCast` | 同上，binary JSON 含 BF16 |

## Kernel 实现

### 公共执行流程

K1 读取 POD 后只按 dtype key 实例化一个 AIV body。`Init` 记录 `blockIdx`、三份 TensorList GM 描述地址，读取本核首尾 Tensor/offset，按 Route 分配 Queue。`Process` 从 `tensorStart` 到 `tensorEnd` 逐项绑定真实 Tensor 地址；TensorList ABI 由 `GetTensorAddr` 解释为“首 `uint64_t` 是指针表相对首地址的字节偏移，右移 3 转成 `uint64_t` 索引，再读取第 `index` 个 GM 指针”。每个 Tensor 片段按 tile 执行 `CopyIn→Compute→CopyOut`。三组主 Queue 提供 MTE2→V→MTE3 producer/consumer 顺序；没有跨核同步、Atomic、event 或 Device workspace。

### Route R-001：FP16 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_list_def.cpp` | `ForeachPowList::ForeachPowList` | 动态输入/输出 dtype 集包含 `DT_FLOAT16`，平台含 `ascend910b/ascend910_93` | 进入公共 Tiling |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_pow_list.cpp` | `foreach_pow_list` | `TILING_KEY_IS(1)` | `ForeachPowList<half>` |

#### Device 数据流

##### 函数级调用链

```text
Tiling4ForeachPowListTiling -> ForeachCommonTiling::RunBigKernelTiling(key=1,blockDim=B)
  -> foreach_pow_list(x1List,x2List,outList,workspace,tiling)
    -> ForeachPowList<half>::Init
    -> ForeachPowList<half>::Process
      -> GetTensorAddr(each list,i,cursorStart)
      -> SingleTensorProcess -> CopyIn -> InnerComputer<half>::Compute -> Power<half,false> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for tensor i in [startTensor,endTensor]:
  begin = (i==startTensor ? startOffset : 0)
  count = n[i]-begin
  if i==endTensor: count=endOffset+1-begin
  bind x1/x2/y GM to tensor i + begin
  for tile in full tiles of M=Q/2 elements:
    DataCopy x1,x2 -> depth-2 input queues
    Power<half,false>(out,x1,x2,tmp12Q,M)
    DataCopy out -> y
  if tail>0:
    DataCopyPad exact tail bytes; Power(...,tail); DataCopyPad exact tail bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 GM | `tensorPtr[i]+cursorStart+tileOffset` | `dataCount` elements | FP16/ND | ACLNN inputs | `CopyIn` | caller/executor；K1 完成前有效 | 完整 tile `Q` bytes/路；tail exact bytes |
| `InQueue_1/2` | Queue slot 0/1 | 每路 `depth 2 * Q` bytes | FP16 | `CopyIn` | `Compute` | `TPipe`/tile | 双缓冲，DeQue 后 Free |
| `OutQueue` | Queue slot 0/1 | `depth 2 * Q` bytes | FP16 | `Power` | `CopyOut` | `TPipe`/tile | 双缓冲，写 GM 后 Free |
| `bufferQueue` | 单 slot | `12Q` bytes | uint8 temp | `Init` 分配 | `Power<half,false>` | 整个 Process；只分配/DeQue 一次 | 跨所有 tile 复用，Process 末 Free |
| y GM | `tensorPtr[i]+cursorStart+tileOffset` | `dataCount` elements | FP16/ND | `CopyOut` | caller | caller | 只写输入有效前缀 |
| workspace | N/A | Host 预留 32 B | opaque | framework | 无 Device consumer | executor/stream | Kernel `userWS=nullptr` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in1["InQueue_1 · depth=2 · 2Q B"]:1
  in2["InQueue_2 · depth=2 · 2Q B"]:1
  out["OutQueue · depth=2 · 2Q B"]:1
  tmp["Power tmp · depth=1 · 12Q B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2Q + 2Q + 2Q + 12Q = 18Q
Q = floor32((U-P)/18) => UB_total <= U-P
dynamic_ub = N/A（AIV TPipe 静态 Queue）
workspace_total = 32 B reservation；Device 有效 slice = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue slot | `CopyIn` | `Compute` DeQue | `FreeTensor` 后后续 tile | EnQue→DeQue→FreeTensor | depth=2，消费前不得覆盖 |
| output Queue slot | `Power` 后 EnQue | `CopyOut` | `FreeTensor` 后后续 tile | EnQue→DeQue→FreeTensor | GM copy 提交后释放 |
| `bufferQueue` slot | Power primitive 临时写 | 同次 Power | 下一 tile Power | 单次 DeQue，串行函数调用，Process 末 Free | 不跨 core；tile 顺序执行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | TQue EnQue/DeQue | `DataCopy/DataCopyPad` | `Compute` | 两路 input Queue slot | FP16 原 dtype；tail 仅搬真实字节 |
| Compute | `Power<half,false>` | 两路 DeQue | OutQueue EnQue | 输入与 output/tmp UB | 固定源码无显式升精度；primitive 内部固定仓不可见；count 精确到有效元素 |
| UB→GM | OutQueue DeQue/Free | `Power` | `CopyOut` | output Queue slot | tail 用 exact-byte `DataCopyPad`，不写扩展区域 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `RunBigKernelTiling/DivideUbMemory2/AssignDataToEachCore` | key 1、`B`、`Q=floor32((U-P)/18)` 与每核区间 |
| Kernel entry/router | `foreach_pow_list.cpp` | `foreach_pow_list` key 1 | 实例化 `ForeachPowList<half>` |
| Device body/helper | `foreach_pow_list.h` | `Process/SingleTensorProcess/InnerComputer<half>::Compute` | TensorList 寻址、tile、`Power<half,false>`、tail 搬运与 Queue 生命周期 |

### Route R-002：FP32 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_list_def.cpp` | 构造函数 | dtype 集含 `DT_FLOAT`，A2/A3 config 可达 | 公共 Tiling |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `foreach_pow_list.cpp` | `foreach_pow_list` | `TILING_KEY_IS(2)` | `ForeachPowList<float>` |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling(key=2,blockDim=B)
  -> foreach_pow_list -> ForeachPowList<float>::Init/Process
    -> per Tensor slice -> SingleTensorProcess
      -> CopyIn -> InnerComputer<float>::Compute -> Power<float,false> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；分核/Tensor 绑定同 R-001
M=Q/4
for each full tile: CopyIn 2*M FP32 -> Power<float,false>(...,tmp3Q,M) -> CopyOut M FP32
if tail: DataCopyPad(blockLen=tail*4 B) -> Power(...,tail) -> DataCopyPad(blockLen=tail*4 B)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | 当前 Tensor pointer + `cursorStart+tileOffset` | 每路 tile≤`Q/4` elements | FP32/ND | caller/CopyOut | CopyIn/caller | caller，K1 完成前有效 | full tile `Q` B；tail exact bytes |
| 三组主 Queue | 独立 slots | `3*2Q=6Q` B | FP32 | CopyIn/Power | Power/CopyOut | TPipe/tile | depth 2 |
| `bufferQueue` | 单 slot | `3Q` B | uint8 temp | Init | `Power<float,false>` | Process | 跨 tile 串行复用 |
| workspace | N/A | 32 B reservation | opaque | framework | 无 | executor | Device 不读 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in1["InQueue_1 · depth=2 · 2Q B"]:1
  in2["InQueue_2 · depth=2 · 2Q B"]:1
  out["OutQueue · depth=2 · 2Q B"]:1
  tmp["Power tmp · depth=1 · 3Q B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 6Q + 3Q = 9Q
Q = floor32((U-P)/9) => UB_total <= U-P
dynamic_ub = N/A
workspace_total = 32 B reservation；Device 有效 slice = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output slots | CopyIn/Power | Power/CopyOut | FreeTensor 后 | Queue EnQue/DeQue/Free | 与 R-001 相同 |
| Power temp | `Power<float,false>` | 同次 primitive | 下一 tile | Process 串行调用，末尾 Free | 单 core 私有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn/Compute | TQue | MTE2 copy | Power | input slots | FP32 原 dtype；有效 count 控制 tail |
| Compute | `Power<float,false>` | input Queue | output Queue | output/tmp | 无算子级 cast；primitive 的特殊值语义固定仓不可见 |
| CopyOut | TQue + `DataCopyPad` tail | Power | GM | output slot | 只写有效元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory2` | FP32 key index对应 calc factor 3，产生 `Q` |
| Kernel entry/router | `foreach_pow_list.cpp` | key 2 branch | `ForeachPowList<float>` |
| Device body/helper | `foreach_pow_list.h` | generic `InnerComputer<float>::Compute` | 直接 `Power<float,false>`，三 Queue 与 `3Q` temp |

### Route R-003：INT32 直接 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_list_def.cpp` | 构造函数 | dtype 集含 `DT_INT32`，A2/A3 config 可达 | 公共 Tiling |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `foreach_pow_list.cpp` | `foreach_pow_list` | `TILING_KEY_IS(3)` | `ForeachPowList<int>` |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling(key=3,blockDim=B)
  -> foreach_pow_list -> ForeachPowList<int>::Init/Process
    -> CopyIn -> InnerComputer<int>::Compute -> Power<int,false> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；分核/Tensor 绑定同 R-001
M=Q/4
for tile of current Tensor slice:
  copy exact valid INT32 x1/x2 to UB
  Power<int,false>(out,x1,x2,tmp5Q,validCount)
  copy valid out to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | Tensor pointer + element offset | tile≤`Q/4` elements | INT32/ND | caller/CopyOut | CopyIn/caller | caller | full tile aligned；tail exact bytes |
| 三组主 Queue | 独立 slots | `6Q` B | INT32 | CopyIn/Power | Power/CopyOut | TPipe/tile | depth 2 |
| `bufferQueue` | 单 slot | `5Q` B | uint8 temp | Init | `Power<int,false>` | Process | 跨 tile 复用 |
| workspace | N/A | 32 B reservation | opaque | framework | 无 | executor | Device 不读 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in1["InQueue_1 · depth=2 · 2Q B"]:1
  in2["InQueue_2 · depth=2 · 2Q B"]:1
  out["OutQueue · depth=2 · 2Q B"]:1
  tmp["Power tmp · depth=1 · 5Q B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 6Q + 5Q = 11Q
Q = floor32((U-P)/11) => UB_total <= U-P
dynamic_ub = N/A
workspace_total = 32 B reservation；Device 有效 slice = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output slots | CopyIn/Power | Power/CopyOut | FreeTensor 后 | Queue protocol | 不提前覆盖 |
| Power temp | `Power<int,false>` | 同次 primitive | 下一 tile | 串行调用，Process 末 Free | 单 core 私有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM↔UB | TQue/DataCopy(Pad) | CopyIn/Power | Power/CopyOut | Queue slots | INT32 不 cast；tail exact count |
| Compute | `Power<int,false>` | input Queue | output Queue | temp/output | 负指数、溢出等整数 primitive 细节固定仓不可见；算子无额外修正 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory2` | INT32 calc factor 5，产生 `Q=floor32((U-P)/11)` |
| Kernel entry/router | `foreach_pow_list.cpp` | key 3 branch | `ForeachPowList<int>` |
| Device body/helper | `foreach_pow_list.h` | generic `InnerComputer<int>::Compute` | `Power<int,false>` 与 `5Q` temp |

### Route R-004：BF16 分段升 FP32 Power

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_pow_list_def.cpp` | 构造函数 | dtype 集含 `DT_BF16`，A2/A3 config 可达 | 公共 Tiling |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4，`Q=floor64((U-P)/63)` |
| Router | `foreach_pow_list.cpp` | `foreach_pow_list` | 非 arch 3003/3113 且 `TILING_KEY_IS(4)` | `ForeachPowList<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling(key=4,blockDim=B)
  -> foreach_pow_list -> ForeachPowList<bfloat16_t>::Init/Process
    -> SingleTensorProcess(M=9Q/2)
      -> CopyIn(BF16)
      -> InnerComputer<bfloat16_t>::Compute
        -> ComputerPerCast(C=Q/4): Cast x1 -> Cast x2 -> Power<float,false> -> Cast_RINT BF16
      -> CopyOut(BF16)
```

##### 源码等价伪码

```text
# 伪码
for each per-core Tensor slice:
  for outer tile of at most M=9Q/2 BF16 elements:
    CopyIn both inputs (tail uses exact-byte DataCopyPad)
    full=C? count/C : 0; rem=count%C, where C=Q/4
    for k in [0,full):
      barrier; f32[0:C]=Cast(x1[k*C:C])
      barrier; f32[C:2C]=Cast(x2[k*C:C])
      barrier; f32[2C:3C]=Power<float,false>(f32[0:C],f32[C:2C],tmp3Q,C)
      barrier; out[k*C:C]=Cast_RINT(f32[2C:3C])
      barrier
    if rem>0: repeat same sequence with valid count=rem
    CopyOut valid BF16 count
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | Tensor pointer + element offset | outer tile≤`9Q/2` elements | BF16/ND | caller/CopyOut | CopyIn/caller | caller | full tile每路 `9Q` B；tail exact bytes |
| 三组主 Queue | independent slots | 每组 `depth 2 * 9Q`，合计 `54Q` B | BF16 | CopyIn/Compute | Compute/CopyOut | TPipe/outer tile | depth 2 |
| `float32Queue` | `[0,C)`,`[C,2C)`,`[2C,3C)` | `3Q` B，总 `3C` FP32 elements | FP32 staging | Cast/Power | Power/Cast_RINT | Process | 同一物理 Queue slot跨子块复用；barrier 隔离依赖 |
| `bufferQueue` | 单 slot | `3Q` B | uint8 temp | Init | `Power<float,false>` | Process | 每个 cast 子块复用 |
| workspace | N/A | 32 B reservation | opaque | framework | 无 | executor | Device 不读 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  in1["InQueue_1 · d=2 · 18Q B"]:1
  in2["InQueue_2 · d=2 · 18Q B"]:1
  out["OutQueue · d=2 · 18Q B"]:1
  f32["FP32 stage · d=1 · 3Q B"]:1
  tmp["Power tmp · d=1 · 3Q B"]:1
```

##### 字节账与复用顺序

```text
UB_actual = 18Q + 18Q + 18Q + 3Q + 3Q = 60Q
Host_budget = (6*10 + 3)Q = 63Q <= U-P
C = Q/sizeof(float) = Q/4 elements
M = 9Q/sizeof(bfloat16_t) = 9Q/2 elements
dynamic_ub = N/A
workspace_total = 32 B reservation；Device 有效 slice = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `f32[0:C)` | x1 Cast | FP32 Power src1 | 下一 cast 子块 x1 | `PipeBarrier<PIPE_V>` before/after Cast/Power | 当前 Power 已完成读取 |
| `f32[C:2C)` | x2 Cast | FP32 Power src2 | 下一子块 x2 | 同上 | 同上 |
| `f32[2C:3C)` | FP32 Power dst | BF16 Cast_RINT | 下一子块 Power | Power 后与 Cast 后 barrier | 降精度完成后才覆盖 |
| Power temp | `Power<float,false>` | 当前 Power | 下一子块 | vector barriers与串行调用 | 单 core 私有 |
| 主 Queue slots | CopyIn/Compute | Compute/CopyOut | FreeTensor 后下一 outer tile | Queue protocol | 与其他 Route 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16 GM→UB | TQue + DataCopy(Pad) | MTE2 | Compute DeQue | input slots | 只搬有效 BF16 元素 |
| BF16→FP32 | `Cast(...,CAST_NONE,count)` + `PIPE_V` barrier | x1/x2 local | FP32 Power | `f32[0:2C)` | 分段最多 C 个元素；不对无效 tail lane计算 |
| FP32 Power | `Power<float,false>` + barriers | two FP32 regions | BF16 Cast | `f32[2C:3C)` 与 `tmp` | 在 FP32 中计算 |
| FP32→BF16 | `Cast(...,CAST_RINT,count)` + barrier | FP32 result | CopyOut | BF16 out Queue | round-to-integral cast mode由源码指定；count 为实际子块 |
| BF16 UB→GM | TQue + DataCopy(Pad) | Compute | GM | output slot | 只写有效外层 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory2` | BF16 以分母 63 预算并 64 B 向下对齐 `Q` |
| Kernel entry/router | `foreach_pow_list.cpp` | guarded key 4 branch | A2/A3 实例化 `ForeachPowList<bfloat16_t>` |
| Device body/helper | `foreach_pow_list.h` | `Init`；specialized `InnerComputer::Compute/ComputerPerCast` | `9Q` 主 slot、`3Q` FP32/Power temp、分段 cast、PIPE_V barrier与 tail |

## 算子特定优化

### Trick T-001：BF16 分片升 FP32 幂计算并复用三段 staging

#### 触发与源码依据

- 适用 Route：R-004。
- 触发条件：Host 生成 dtype key 4，entry 在 A2/A3 编译架构进入 `ForeachPowList<bfloat16_t>`；每个内部子块 `dataCount<=Q/4`。
- Golden 来源：[`foreach_pow_list.h:84-95`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_list/op_kernel/foreach_pow_list.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
Power<float, false>(
    float32Tensor[BUFFER_NUM * maxCastDataCount], float32Tensor, float32Tensor[maxCastDataCount], bufferTensor,
    dataCount);
PipeBarrier<PIPE_V>();
Cast(
    outLocal[index * maxCastDataCount], float32Tensor[BUFFER_NUM * maxCastDataCount], RoundMode::CAST_RINT,
    dataCount);
PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

同一个 `3Q` FP32 Queue slot 被切成三个连续的 `Q` 字节区：底数、指数和结果；`maxCastDataCount=Q/4` 决定每段 FP32 元素数。外层 BF16 tile 可达 `9Q/2` 个元素，但内层反复在这三段上完成 cast→Power→cast，不为整块 BF16 tile 物化两份 FP32 输入和一份 FP32 输出；这把 FP32 staging 固定为 `3Q`，并由 `PIPE_V` barrier 证明每段在下一子块覆盖前已经消费。FP32 Power 避免直接 BF16 Power 的精度路径，最终 `CAST_RINT` 明确降精度模式。

#### 迁移边界

- 适用：低精度输入需要在 FP32 中完成复杂逐元素运算，且大 tile 可按固定 staging 容量分段的 AIV Kernel。
- 前提：拥有三段互不重叠 FP32 staging、临时区满足 `Power<float,false>`、每次 Vector count 不超过 `Q/4`，并维持源码中的 `PIPE_V` 依赖。
- 失效：FP16/FP32/INT32 使用 R-001/R-002/R-003 直接 `Power<T,false>`；被排除的 arch 3003/3113 没有 key-4 body，不能照搬。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；复用 staging 时若删除相邻 Vector barrier，会破坏源码明确建立的 cast/Power/覆盖依赖，属于超出该实现证据的改变。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 3 | key 4 + arch guard | dtype 唯一决定 Route |
| 模板与 dtype | `ForeachPowList<half>` | `<float>` | `<int>` | `<bfloat16_t>` specialized InnerComputer | BF16 单独升精度 |
| 循环与分核 | `M=Q/2` | `M=Q/4` | `M=Q/4` | outer `M=9Q/2`，inner `C=Q/4` | 同一 32 B 分核；tile按 UB布局变化 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 三组 depth-2 Queue + temp；无 TBuf/dynUB | 同左 | 同左 | 同左并加 FP32 Queue | 全部为 AIV，无通信/动态 UB |
| 内存总占用与复用 | `18Q` | `9Q` | `11Q` | 实配`60Q`，Host预算`63Q` | Power temp factor与 BF16 staging决定 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Power half | DataCopy(Pad)+Power float | DataCopy(Pad)+Power int | DataCopy(Pad)+Cast+Power float+Cast | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue；直接 FP16 | Queue；FP32 | Queue；INT32 | Queue + PIPE_V；FP32计算/CAST_RINT | 所有 tail 都以真实 count 执行 |
| fallback/失败边界 | 无 Device fallback | 同左 | 同左 | arch guard不满足则无 body | unknown key 均静默无动作 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 产品表支持；`ascend910b` 接线 | 公开 ACLNN 产品表支持；`ascend910_93` 接线 | 公开 ACLNN 产品表明确不支持；尽管 OpDef/CMake/binary config 存在 `ascend950` 实现信号，按发布门禁不生成 950 卡 |
| Host/Launcher | 共用 `ForeachCommonTiling` 与四 dtype key | 同 A2 | 实现接线复用公共 Tiling，但与公开支持冲突，不作为正式知识 |
| Kernel/资源 | 共用 `foreach_pow_list.cpp/.h`；四 Route | 共用源码；FP16 binary 文件名与 A2 有哈希差异，但固定源码 body 相同 | 文件存在不证明公开支持；不在本卡展开 |

## 限制与失败边界

- 固定 Tiling POD 最多容纳 50 个 Tensor 和 50 个 core；`Init` 到第一个空 dynamic input 为止，公开 ACLNN 文档未显式给出 TensorList 长度上限。
- Host 只验证输出 storage size 不小于输入；Kernel 只写输入元素数，输出额外区域保持调用前内容。
- `SingleTensorProcess` 的 tile offset 为 `uint32_t`，固定 Host 没有对超大单核 Tensor 片段给出 32 位溢出检查。
- `N=0` 时仍启动 1 core；零元素 Tensor 由 `dataCount=0` 跳过 tile。空 TensorList（列表项数为 0）的 Device 指针表行为未在公开约束和固定源码中闭合，不应假定支持。
- Host 未显式检查 `Q>0`；受支持硬件通常具有足够 UB，但固定源码没有小 UB fallback。
- INT32 `Power<int,false>` 对负指数、溢出与特殊值的 primitive 级语义固定仓不可见；FP16/FP32 `Power` 内部算法、误差界也固定仓不可见。
- 公开 ACLNN header、自动生成 bridge/executor、输入 Contiguous 临时区合并公式和具体 launch 错误包装固定仓不可见；这些是框架边界。算子自有 selector、POD、entry/body、内存与构建接线已闭合。
- 950 公开文档与实现接线冲突，`foreach_pow_list_950.md` 不创建；不得仅凭 `ascend950` config 推断公开 API 支持。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_pow_list/docs/aclnnForeachPowList.md` | 两段式函数原型、产品表、参数/错误约束 | ACLNN 公开性、A2/A3 支持与 950 不支持 |
| 入口声明/约束 | `foreach/foreach_pow_list/README.md` | 产品支持、公式、调用方式 | 逻辑语义与公开入口 |
| OpDef | `foreach/foreach_pow_list/op_host/foreach_pow_list_def.cpp` | `ForeachPowList` | dtype/layout/AutoContiguous与平台 config |
| InferShape | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`、ForeachPowList注册 | 输出数、shape/dtype传播 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachPowListTiling`、`IMPL_OP_OPTILING(ForeachPowList)` | Tiling入口与公共类接线 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD、workspace、key |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | 字段声明、容量与注册 |
| key helper | `foreach/foreach_utils/op_host/common_dtype.h` | `GetDataTypeSize/GetTilingKeyByDtypeOnly` | dtype字节数和 key 1–4 |
| Kernel entry | `foreach/foreach_pow_list/op_kernel/foreach_pow_list.cpp` | `foreach_pow_list` | key router、BF16 arch guard、workspace不使用 |
| Kernel body | `foreach/foreach_pow_list/op_kernel/foreach_pow_list.h` | `ForeachPowList`、`InnerComputer` | Queue、循环、搬运、Power、cast、同步与 tail |
| 构建接线 | `foreach/foreach_pow_list/op_host/CMakeLists.txt` | `add_modules_sources(...ACLNNTYPE aclnn...)` | 生成 ACLNN Host 接线及 foreach_utils依赖 |
| 构建接线 | `foreach/foreach_pow_list/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3/950 compute-unit源码选择信号 |
| binary config | `foreach/foreach_pow_list/op_host/config/ascend910b/foreach_pow_list_binary.json` | `ForeachPowList` 四项 `op_list` | A2 四 dtype binary 发布 |
| binary config | `foreach/foreach_pow_list/op_host/config/ascend910_93/foreach_pow_list_binary.json` | `ForeachPowList` 四项 `op_list` | A3 四 dtype binary 发布 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 可复用六字段的分核描述语义 | 字段声明顺序、50项容量、元素/字节单位、含端点 `tensorEndOffset` 不得改变；具体生成 offset不可猜测 | `ForeachCommonTilingData`；`FillTilingData/ParseTilingData` |
| 调用顺序 | 单 K1 与每 tile `CopyIn→Compute→CopyOut` | TensorList 地址表解释、首尾 offset修正和 inclusive end 语义不变 | `foreach_pow_list`；`Process/SingleTensorProcess/GetTensorAddr` |
| workspace/内存 | AIV Queue 字节账与 32 B reservation 可复用 | depth=2 必须计入；dtype temp factor、BF16 `9Q/3Q/3Q` 布局及 owner/lifetime不变 | `DivideUbMemory2`；`Init/GetTmpBufferUBSize` |
| 同步 | Queue protocol；BF16三段 staging的 `PIPE_V` barrier | BF16 cast/Power/cast之间及复用前 barrier 不得删除 | `CopyIn/Compute/CopyOut`；`ComputerPerCast` |
| 精度与尾块 | FP16/FP32/INT32直接 Power，BF16 FP32 Power后CAST_RINT | dtype模板、cast顺序、实际 `dataCount` 和 tail exact-byte搬运不变 | `InnerComputer`；`CopyIn/CopyOut` |
| 失败边界 | Host校验失败阻断；unknown key无动作 | 不得把未知 key当作 fallback；不得发布公开文档明确不支持的 950 | `CheckOpInfo`；`foreach_pow_list` router；公开产品表 |
