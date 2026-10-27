---
type: Operator
title: "Relu A2 ACLNN 算子知识"
description: "说明 aclnnReluV3 在 A2 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐 dtype AIV 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/docs/aclnnReluV3.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_host/relu_v3_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_host/relu_v3_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_kernel/relu_v3_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_kernel/relu_v3_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_kernel/relu_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_kernel/relu_v3.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu/docs/aclnnRelu&aclnnInplaceRelu.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu/op_host/relu_def.cpp
    role: implementation
created_at: "2026-08-25T18:53:24Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# Relu A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnReluV3GetWorkspaceSize` / `aclnnReluV3`；生成式 ACLNN bridge 落到 `ReluV3` OpDef。 |
| 计算语义 | `y_i=max(x_i,0)`；同时按输入元素顺序将 `(x_i>0)` 打包为 1 bit，`mask[j]` 保存元素 `8j..8j+7`。 |
| 输入输出 | `x,y`：5-D `NC1HWC0`，同 shape/dtype；dtype 为 FP16/FP32/INT32/INT8/UINT8/BF16。`mask`：`UINT8/ND`，前四维同 `x`，末维为 `x_last/8`。 |
| 执行类型 | 单节点 `AIV_VECTOR`，无 Cube/AIC、MIX_CV、SIMT、NDDMA、AICPU 或第二 Kernel。 |
| Kernel 路线 | R-001 FP16；R-002 FP32；R-003 INT32；R-004 INT8/UINT8；R-005 BF16。 |
| 核心资源 | 三个 depth=2 的 `TQue`，每个 slot 固定 30 KiB，总 UB 180 KiB；无算子自用 workspace。 |
| 来源闭包 | `source_closed`：公开 API 约束与生成接线、`ReluV3` OpDef/InferShape/Tiling/POD、模板 key、AIV entry、五个 dtype body 及活动 experimental 构建均可追溯。生成式 ACLNN executor 与 AscendC primitive 内部作为框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对存储 shape 中的每个元素执行：

```text
y[i] = max(x[i], 0)
mask[i / 8].bit(i % 8) = (x[i] > 0)
```

`mask` 是压缩位图而不是逐元素 `UINT8` 布尔张量。`relu_v3_infershape.cpp::InferShape` 复制 `x` shape 到 `y` 和 `mask`，随后把 `mask` 最后一维改为 `(last_dim + 7) / 8`；Tiling 又要求非标量输入的最后一维能被 8 整除，所以活动成功路线实际为 `last_dim/8`。比较模式固定为 `CMPMODE::GT`，因此零不置位。NaN、带符号零及底层 `Relu/Maxs` 的特殊值细节由 AscendC primitive 定义，固定仓未展开，不在本卡推测。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 公开 ACLNN 约束为 5-D `NC1HWC0` | FP16、FP32、INT32、INT8、UINT8、BF16 | 必选；末维须为 8 的倍数；支持非连续 Tensor，由生成式 ACLNN 框架处理。 |
| `yOut` | 与 `x` 同 shape，`NC1HWC0` | 与 `x` 相同 | 必选；保存 ReLU 结果；公开文档声明支持非连续 Tensor。 |
| `maskOut` | 5-D `ND`；实现为前四维同 `x`、末维 `x_last/8` | UINT8 | 必选；每字节保存 8 个比较结果。文档参数表“shape 与 x 一致”的文字与 InferShape、示例的压缩 shape 冲突，复用时以实际 `InferShape` 和 Kernel 地址公式为准。 |
| 属性 | N/A | N/A | 无属性。 |

### 入口与版本

本卡选择 A2 上公开且活动实现最新的 `aclnnReluV3`：公开文档明确 A2 支持、A3/950 不支持；`relu_v3_def.cpp::ReluV3` 仅注册 `ascend910b`；`op_host/CMakeLists.txt` 以 `OPTYPE relu_v3 ACLNNTYPE aclnn` 接入生成式 ACLNN。固定仓没有手写 `aclnnReluV3` C++ bridge，生成代码、`Contiguous/ViewCopy` 与 executor 壳按框架边界处理，但算子拥有的 OpDef、shape/dtype 推导、Tiling、POD、entry 与 body 均闭合。

稳定目录 `activation/relu/` 中的 `aclnnRelu` 是 950 的活动实现；其 950 闭包缺失见“平台差异”和“限制与失败边界”。本卡不处理 `aclnnInplaceRelu` 或 `aclnnAddRelu`。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 ACLNN 第一段接口要求 `x/yOut/maskOut/workspaceSize/executor` 有效；文档声明 dtype、format、rank 或 shape 不合法返回 `ACLNN_ERR_PARAM_INVALID`，空指针返回 `ACLNN_ERR_PARAM_NULLPTR`。生成式校验代码在固定仓不可见。
2. `ReluV3` OpDef 将 `x/y` 限定为六种相同 dtype 的 `NC1HWC0`，将 `mask` 限定为 `UINT8/ND`，并仅添加 `ascend910b` AICore config。
3. `InferShape` 令 `y.shape=x.shape`，令 `mask.shape=x.shape` 后压缩末维；`InferDataType` 令 `y.dtype=x.dtype`、`mask.dtype=UINT8`。
4. `TilingFunc` 读取 `x` 的 storage shape；若有维度且末维 `%8!=0` 则失败。它以 `long size=1` 顺序相乘所有 storage 维度，并在 `size > floor(INT_MAX/512)*512` 时失败。
5. 生成式 ACLNN 对非连续输入输出的物化/回写属于框架边界；固定仓没有暴露这些节点的 workspace slice 和内部 kernel。核心 `ReluV3` 节点的输入输出仍按上述 storage shape 进入 Tiling。

### 参数、Workspace 与生命周期

`TilingFunc` 唯一写入 `ReluV3TilingData::size = size`，单位为元素。`dtype` 仅用于 `ge::GetSizeInBytes(size,dtype)` 计算分核阈值，不进入 POD。`blockDim` 取 AIV 核数，并按字节数修正：`data_size < 32 KiB` 时为 1，`32 KiB <= data_size < 4 MiB` 时整数除以 2，其他情况保持全部 AIV 核。

Tiling 不调用 `GetWorkspaceSizes`，Kernel entry 的 `workspace` 参数也从不读取，因此核心算子节点的自用 workspace 为 0。第一段 ACLNN 返回的总 workspace 仍可能包含生成式非连续 Tensor 物化/回写所需空间；其精确公式与 owner 在固定仓不可见，用户必须使用第一段实际返回值，并让该内存至少存活到同一 executor 的第二段调用完成。

### 启动图与失败传播

```text
aclnnReluV3GetWorkspaceSize(x, yOut, maskOut)
  -> [生成式 ACLNN 校验/必要的连续化：框架边界]
  -> ReluV3::InferShape / InferDataType
  -> optiling::TilingFunc
  -> 单节点 relu_v3<schMode=0>(x, y, mask, workspace_unused, tiling)
  -> [必要的非连续输出回写：框架边界]
aclnnReluV3(workspace, workspaceSize, executor, stream)
```

核心启动节点由 `ReluV3` OpDef 的 A2 AICore config 和模板 key 0 选择，entry 为 `relu_v3<0>`，`blockDim` 来自 `TilingFunc`，POD 为单字段 `ReluV3TilingData`。`x/y/mask` 均直接是 GM 对象，算子节点之间没有 workspace 边。executor 在传入 stream 上提交框架节点与核心 Kernel；生成式提交细节不可见。Tiling 的末维、size、platform 空指针检查任一失败都会阻断 Kernel；entry 对 `schMode!=0` 只返回且不写输出，但 Host 活动 key 恒为 0。

## Route 与参数设计

### 分核与资源计算

令 `N=product(x.storage_shape)`，`S=sizeof(T)`，`B=ceil(N/512)`：

```text
data_size = N * S bytes
blockDim = 1                               if data_size < 32 KiB
           floor(AIV_core_num / 2)         if 32 KiB <= data_size < 4 MiB
           AIV_core_num                    otherwise

core k start = floor(B * k / blockDim) * 512 elements
core k end   = min(floor(B * (k+1) / blockDim) * 512, N) elements
```

每核得到连续区间，区间边界以 512 元素块划分；最后一核的 `end` 被 `N` 截断。每个 core 以 `MAX_TILE_SIZE=30 KiB/sizeof(U)` 迭代。FP16 与 INT8/UINT8 的 `U=half`，tile 上限 15360 元素；FP32、INT32、BF16 的 `U` 为 4-byte 类型，tile 上限 7680 元素。三个队列均为 depth=2、每 slot 30 KiB，总 UB 为 `3*2*30 KiB=180 KiB`。Host 没有查询 UB 或执行 `IsCapable`；180 KiB 可用 UB 是这一路线的隐式平台前提。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | key 0；`relu_v3<half,half,half>` | `x.dtype=DT_FLOAT16`，OpDef/生成编译实例选择 `DTYPE_X=half` | 按 dtype 选择下一实例；无运行时 fallback。 |
| 2 | R-002 FP32 | key 0；`relu_v3<float,float,float>` | `x.dtype=DT_FLOAT` | 下一 dtype 实例或生成阶段失败。 |
| 3 | R-003 INT32 | key 0；`relu_v3<int,int,float>` | `x.dtype=DT_INT32` | 下一 dtype 实例或生成阶段失败。 |
| 4 | R-004 INT8/UINT8 | key 0；`relu_v3<int8_t,half,half>` / `relu_v3<uint8_t,half,half>` | `x.dtype=DT_INT8` 或 `DT_UINT8` | 下一 dtype 实例或生成阶段失败。 |
| 5 | R-005 BF16 | key 0；`relu_v3<bfloat16_t,float,float>` | `x.dtype=DT_BF16` | 不支持 dtype 在 ACLNN/OpDef 阶段失败。 |

```text
平台必须为 ascend910b，shape/rank/format/dtype 先通过公开 ACLNN 与 OpDef 约束
  -> Tiling: last_dim % 8 == 0 且 N <= floor(INT_MAX/512)*512
  -> 唯一 schMode = ELEMENTWISE_TPL_SCH_MODE_0 (0)
  -> 编译 dtype 实例：half | float | int | int8_t/uint8_t | bfloat16_t
  -> 对应 R-001 | R-002 | R-003 | R-004 | R-005
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 0 | A2；5-D NC1HWC0；FP16；末维 `%8==0`；size 上界通过 | 单 AIV `relu_v3<0>` | `relu_v3<half,half,half>`；`Relu+Compare` | 三个双缓冲 Queue，共 180 KiB；Queue 顺序 | x/y 以 16 元素对齐，Compare 以 128 元素对齐，mask 以 32 bytes 对齐；无 fallback。 |
| R-002 FP32 | key 0 | 同上，FP32 | 同上 | `relu_v3<float,float,float>`；`Maxs+Compare` | 同上 | x/y 以 8 元素对齐，Compare 以 64 元素对齐，mask 以 32 bytes 对齐。 |
| R-003 INT32 | key 0 | 同上，INT32 | 同上 | `relu_v3<int,int,float>`；`Maxs`，输入 buffer 原地 reinterpret/cast 为 FP32 后 Compare | 同上；x Queue 被原地覆盖 | x/y 8 元素对齐，Compare 64 元素对齐，mask 32 bytes 对齐。 |
| R-004 INT8/UINT8 | key 0 | 同上，INT8 或 UINT8 | 同上 | `relu_v3<T,half,half>`；升 FP16、Relu、降精度、Compare | 同上；mask Queue 先作 FP16 临时值后原地生成 bitmask | x/y 32 元素对齐，Compare 128 元素对齐，mask 32 bytes 对齐。 |
| R-005 BF16 | key 0 | 同上，BF16 | 同上 | `relu_v3<bfloat16_t,float,float>`；升 FP32、Maxs、RINT 降 BF16、Compare | 同上；mask Queue 先作 FP32 临时值后原地生成 bitmask | x/y 16 元素对齐，Compare 64 元素对齐，mask 32 bytes 对齐。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| A2 平台 | OpDef config 为 `ascend910b` | 仅该平台生成/注册当前实现 | 全部 | 其他平台无此 V3 路线 | `relu_v3_def.cpp::ReluV3` |
| 压缩 mask 对齐 | `dim_num>0 && last_dim%8!=0` 时失败 | `mask` GM offset 为 `i/8`，完整 tile 输出 `_ / 8` bytes | 全部 | `GRAPH_FAILED` | `relu_v3_tiling.cpp::TilingFunc` |
| POD int 上界 | `size > floor(INT_MAX/512)*512` 时失败 | POD、core/tile 地址和循环变量均为 32-bit `int` | 全部 | `GRAPH_FAILED` | `relu_v3_tiling.cpp::TilingFunc` |
| 分核阈值 | `<32 KiB` 用 1 核；`<4 MiB` 用半数 AIV；否则全核 | 减少小数据上的空核；改变每核连续区间 | 全部 | 不是失败，进入下一阈值档 | `relu_v3_tiling.cpp::TilingFunc` |
| 固定 UB | 无 Host guard；Device 固定 `3*2*30 KiB` | 三 Queue 双缓冲需要 180 KiB | 全部 | 可用 UB 不足时没有源码 fallback | `relu_v3.h::RELU_V3_SETUP` |
| DMA tail | `_ < MAX_TILE_SIZE` 时 x/y 向上取整到 32 bytes，mask 向上取整到 32 bytes | `DataCopy` 使用块对齐长度 | 全部 | 无 unaligned fallback | `relu_v3.h::relu_v3<T,U,V>` |
| Compare tail | count 向上取整到 `256/sizeof(V)` 元素 | Compare 产生 packed bitmask 的 vector 粒度 | 全部 | 无标量 fallback | `relu_v3.h::ReluV3<T,U,V>` |
| key guard | `schMode==0` | entry 只实例化模式 0 | 全部 | 其他 key 静默返回，不写输出 | `relu_v3.cpp::relu_v3` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ReluV3TilingData` | `size` | `int` / 32-bit | 唯一字段；`offsetof=0`、`sizeof=4`、`alignof=4`（目标 ABI 的普通 `int`） | 元素 | `TilingFunc`: `tiling.size = product(x.storage_shape)`，从 `long` 窄化 | `RELU_V3_SETUP`: 计算 512 元素块数与每核 `[start,end)` | Host 先限制到 `floor(INT_MAX/512)*512`；乘法本身无逐维溢出检查。 |

模板 key 不在 POD 内：Host 通过 `GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0)` 设置 key，构建系统将 `schMode=0` 实例化到 entry。`REGISTER_TILING_DEFAULT`/`GET_TILING_DATA` 的序列化宏实现属于外部框架，但 Host/Device 使用同一 `ReluV3TilingData` 定义。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | 0 / `DTYPE_X=half` | `relu_v3_tiling.cpp::TilingFunc` | `ReluV3TilingData::size` | `relu_v3.cpp::relu_v3<0>` | `relu_v3.h::relu_v3<half,half,half>`、FP16 `ReluV3` overload | `op_host/CMakeLists.txt::add_modules_sources`；`relu_v3_def.cpp::ReluV3` |
| R-002 FP32 | 0 / `DTYPE_X=float` | 同上 | 同上 | 同上 | `relu_v3<float,float,float>`、FP32 overload | 同上 |
| R-003 INT32 | 0 / `DTYPE_X=int` | 同上 | 同上 | 同上 | `relu_v3<int,int,float>`、INT32 overload | 同上 |
| R-004 INT8/UINT8 | 0 / `DTYPE_X=int8_t,uint8_t` | 同上 | 同上 | 同上 | `relu_v3<T,half,half>`、8-bit overload | 同上 |
| R-005 BF16 | 0 / `DTYPE_X=bfloat16_t` | 同上 | 同上 | 同上 | `relu_v3<bfloat16_t,float,float>`、BF16 overload | 同上 |

## Kernel 实现

### 公共执行流程

entry `relu_v3<schMode>` 声明 `KERNEL_TYPE_AIV_ONLY`，解析 `ReluV3TilingData`，仅在 `schMode==0` 时按 `DTYPE_X` 调用真实 body。body 将 GM 绑定为 `x:T[]`、`y:T[]`、`mask:uint8_t[]`，创建 `x_t_que(VECIN)`、`y_t_que/mask_t_que(VECOUT)`，三者均 depth=2。每核按 512 元素块获得连续区间，再按 dtype 对应的 `MAX_TILE_SIZE` 循环执行：对齐 CopyIn → dtype 专用 ReLU 与 packed Compare → 对齐 CopyOut。`workspace` 完全未读，单 Kernel 内没有跨节点边。

### Route R-001：FP16 原生 Relu

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnReluV3.md` | `aclnnReluV3*` | A2 支持；FP16；5-D NC1HWC0 | 生成 `ReluV3` 节点。 |
| Host | `op_host/relu_v3_def.cpp` | `ReluV3` | `DT_FLOAT16` + `ascend910b` | FP16 binary 实例。 |
| Tiling | `op_host/relu_v3_tiling.cpp` | `TilingFunc` | shape guards 通过；key=0 | `relu_v3<0>`。 |
| Router | `op_kernel/relu_v3.h` | `relu_v3<T>` | `T=half` 命中 `is_one_of_v<T,half,float>` | `relu_v3<half,half,half>`。 |

#### Device 数据流

##### 函数级调用链

```text
TilingFunc -> relu_v3<0>(x,y,mask,workspace_unused,tiling)
  -> relu_v3<half>(...)
    -> relu_v3<half,half,half>(...)
      -> CopyIn(DataCopy + x_t_que.EnQue)
      -> ReluV3<half,half,half>(Relu, Duplicate, Compare)
      -> CopyOut(DataCopy y + DataCopy packed mask)
```

##### 源码等价伪码

```text
# 伪码；k 为 AIV core id
[start,end) = 按 ceil(N/512) 个块比例切给 core k
for i in range(start,end,15360):
    valid = min(end-i,15360)
    x_dma = valid if full_tile else ceil(valid/16)*16
    xq = alloc xQueue; DataCopy(xq, xGM+i, x_dma); enqueue/dequeue
    yq = alloc yQueue; mq = alloc maskQueue
    Relu(yq,xq,valid)
    Duplicate(mq_as_half,0,valid)
    Compare(mq,xq,mq_as_half,GT,ceil(valid/128)*128)
    free xq; enqueue/dequeue yq,mq
    y_dma = valid if full_tile else ceil(valid/16)*16
    m_dma = valid/8 if full_tile else ceil(ceil(valid/8)/32)*32
    DataCopy(yGM+i,yq,y_dma); DataCopy(maskGM+i/8,mq,m_dma)
    free mq,yq
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `i` elements | logical `N` FP16 | FP16/NC1HWC0 storage | caller/framework | CopyIn | caller；executor 期间有效 | tail read count `ceil(valid/16)*16`。 |
| `x_t_que` | queue slot 0/1 | `2*15360*2=61440 B` | physical FP16 | DataCopy | `Relu`/`Compare` | Kernel `TPipe` | depth=2；计算后 Free。 |
| `y_t_que` | queue slot 0/1 | 61440 B | FP16 | `Relu` | CopyOut | Kernel `TPipe` | depth=2。 |
| `mask_t_que` | queue slot 0/1 | 61440 B，实际 bitmask 只占前 `ceil(valid/8)` bytes | zero FP16 temporary then packed bits | `Duplicate`/`Compare` | mask CopyOut | Kernel `TPipe` | 同一 slot 先作 zero source，再保存 packed dst。 |
| GM `y` | `i` elements | logical `N` FP16 | FP16/NC1HWC0 storage | y CopyOut | caller | caller | tail write 32-byte 圆整。 |
| GM `mask` | `i/8` bytes | logical `N/8` bytes | UINT8/ND packed | mask CopyOut | caller | caller | tail write 32-byte 圆整。 |
| workspace | N/A | 0 operator-owned bytes | N/A | N/A | entry 不读取 | N/A | ACLNN 框架 workspace 另计。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq["x_t_que · VECIN · depth=2 · 2×30 KiB"]:1
  yq["y_t_que · VECOUT · depth=2 · 2×30 KiB"]:1
  mq["mask_t_que · VECOUT · depth=2 · 2×30 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*30 KiB + 2*30 KiB + 2*30 KiB = 180 KiB
UB_total <= 平台可用 UB（Host 未查询，固定实现的必要前提）
dynamic_ub = N/A
workspace_total = 0（核心 Kernel）；生成式 ACLNN 框架 workspace 固定仓不可见
=> MAX_TILE_SIZE = 15360 FP16 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x Queue slot | CopyIn | `Relu`、`Compare` | 下一 tile CopyIn | `EnQue→DeQue→FreeTensor` | 两个计算均在 Free 前完成。 |
| mask Queue slot | `Duplicate` 写 FP16 zeros；`Compare` 写 packed bits | mask CopyOut | 下一 tile | `EnQue→DeQue→FreeTensor` | Compare 完成后才入队；packed 结果位于同一 slot 前部。 |
| y Queue slot | `Relu` | y CopyOut | 下一 tile | Queue 生命周期 | CopyOut 后释放。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE→V | `x_t_que EnQue/DeQue` | DataCopy | Compute | x slot | CopyIn tail 16 FP16 元素对齐。 |
| V sequence | 同一计算作用域 | `Relu`、`Duplicate` | `Compare` | x/y/mask slot | FP16 原生计算；Compare count 向上到 128 元素。 |
| V→MTE | `y/mask EnQue/DeQue` | Compute | CopyOut | y/mask slot | 只定义前 `valid` 个 y 与前 `ceil(valid/8)` 个 mask bytes；DMA 向上到 32 bytes。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp` | `TilingFunc` | size、blockDim、key 0。 |
| Kernel entry/router | `experimental/activation/relu_v3/op_kernel/relu_v3.cpp` | `relu_v3<0>` | AIV-only，POD 解析，调用 `relu_v3<DTYPE_X>`。 |
| Device body/helper | `experimental/activation/relu_v3/op_kernel/relu_v3.h` | `relu_v3<half,half,half>`、FP16 `ReluV3` | 15K tile、Relu、packed Compare、双缓冲 DMA。 |

### Route R-002：FP32 Maxs

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `docs/aclnnReluV3.md`；`op_host/relu_v3_def.cpp` | `aclnnReluV3*`；`ReluV3` | A2、FP32、5-D NC1HWC0 | FP32 instance。 |
| Tiling | `op_host/relu_v3_tiling.cpp` | `TilingFunc` | guards 通过，key=0 | entry 0。 |
| Router | `op_kernel/relu_v3.h` | `relu_v3<T>` | `T=float` | `relu_v3<float,float,float>`。 |

#### Device 数据流

##### 函数级调用链

```text
TilingFunc -> relu_v3<0> -> relu_v3<float> -> relu_v3<float,float,float>
  -> DataCopy x -> Maxs(y,x,0.0f,valid)
  -> Duplicate(zero) -> Compare(GT, rounded_count)
  -> DataCopy y and packed mask
```

##### 源码等价伪码

```text
# 伪码；公共 512 元素分核
for i in core_range step 7680:
  valid=min(remaining,7680); copy ceil(valid/8)*8 FP32 elements on tail
  Maxs(y,x,0.0f,valid)
  Duplicate(mask_as_float,0.0f,valid)
  Compare(mask,x,mask_as_float,GT,ceil(valid/64)*64)
  copy y with 8-element alignment; copy packed mask with 32-byte alignment
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `i` FP32 elements | N elements each | FP32/NC1HWC0 | caller / CopyOut | CopyIn / caller | caller | tail 8-element alignment。 |
| x Queue | slot | `2*7680*4=61440 B` | FP32 | CopyIn | Maxs/Compare | TPipe | depth=2。 |
| y Queue | slot | 61440 B | FP32 | Maxs | CopyOut | TPipe | depth=2。 |
| mask Queue | slot | 61440 B | zero FP32 + packed UINT8 | Duplicate/Compare | mask CopyOut | TPipe | slot 内复用。 |
| GM mask | `i/8` | N/8 bytes | packed UINT8 | CopyOut | caller | caller | tail 32 bytes。 |
| workspace | N/A | 0 | N/A | N/A | unused | N/A | framework workspace 另计。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq["x Queue · 2×30 KiB"]:1
  yq["y Queue · 2×30 KiB"]:1
  mq["mask Queue · 2×30 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 180 KiB <= 平台可用 UB（隐式前提）
dynamic_ub = N/A
workspace_total = 0 operator-owned bytes
=> MAX_TILE_SIZE = 7680 FP32 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mask slot | Duplicate FP32 zeros；Compare packed dst | CopyOut | 下一 tile | Queue EnQue/DeQue/Free | Compare 在入队前完成。 |
| x/y slots | CopyIn / Maxs | Compute / CopyOut | 下一 tile | Queue protocol | Free 后才复用。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | VECIN Queue | MTE | Maxs/Compare | x | FP32，尾块 8 元素对齐。 |
| Compute | 顺序 `Maxs→Duplicate→Compare` | Vector | Queue enqueue | y/mask | 无 cast；Compare 64 元素粒度。 |
| CopyOut | VECOUT Queue | Vector | MTE | y/mask | y 8 元素、mask 32 bytes 对齐。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp` | `TilingFunc` | 字节阈值和 key。 |
| Kernel entry/router | `experimental/activation/relu_v3/op_kernel/relu_v3.cpp` | `relu_v3<0>` | FP32 binary 进入相同 entry。 |
| Device body/helper | `experimental/activation/relu_v3/op_kernel/relu_v3.h` | `relu_v3<float,float,float>` | `Maxs` 与 FP32 Compare。 |

### Route R-003：INT32 原地转 FP32 比较

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `docs/aclnnReluV3.md`；`relu_v3_def.cpp` | `aclnnReluV3*`；`ReluV3` | A2、INT32 | INT32 instance。 |
| Tiling | `relu_v3_tiling.cpp` | `TilingFunc` | key 0 | entry 0。 |
| Router | `relu_v3.h` | `relu_v3<T>` | `T=int` | `relu_v3<int,int,float>`。 |

#### Device 数据流

##### 函数级调用链

```text
relu_v3<int,int,float>
  -> DataCopy INT32 x
  -> Maxs(y_int,x_int,0,valid)
  -> x_float aliases x_int; Cast(x_float,x_int,CAST_RINT,valid)
  -> Duplicate(mask_as_float,0); Compare(mask,x_float,zero,GT,round64)
  -> CopyOut INT32 y + packed mask
```

##### 源码等价伪码

```text
# 伪码；tile=7680
copy aligned INT32 x
y_int = max(x_int,0)
x_buffer_as_float = Cast_RINT(x_int)       # 同一 x Queue slot 原地覆盖
zero_float = 0 in mask Queue
mask_bits = Compare_GT(x_buffer_as_float,zero_float,ceil(valid/64)*64)
copy aligned INT32 y and 32-byte-aligned mask
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `i` | N INT32 | INT32/NC1HWC0 | caller/CopyOut | CopyIn/caller | caller | 8-element tail alignment。 |
| x Queue | slot | 61440 B | INT32，随后同址 FP32 | CopyIn、Cast | Maxs、Compare | TPipe | Maxs 必须先消费原 INT32，再由 Cast 覆盖。 |
| y Queue | slot | 61440 B | INT32 | Maxs | CopyOut | TPipe | depth=2。 |
| mask Queue | slot | 61440 B | zero FP32，随后 packed UINT8 | Duplicate/Compare | CopyOut | TPipe | alias 复用。 |
| GM mask | `i/8` | N/8 bytes | packed UINT8 | CopyOut | caller | caller | tail 32 bytes。 |
| workspace | N/A | 0 | N/A | N/A | unused | N/A | N/A。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq["x Queue · INT32→FP32 in-place · 2×30 KiB"]:1
  yq["y Queue · INT32 · 2×30 KiB"]:1
  mq["mask Queue · FP32 zero→bits · 2×30 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 180 KiB；x 的 INT32/FP32 是同一物理 slot，不另加 TBuf
dynamic_ub = N/A
workspace_total = 0
=> tile = 7680 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x slot | CopyIn INT32 | Maxs | Cast 原地写 FP32；之后下一 tile | 源码调用顺序、最后 FreeTensor | Maxs 必须在 Cast 之前。 |
| mask slot | Duplicate FP32 | Compare source/destination | 下一 tile | Compare 后 enqueue | packed dst 与 zero source 共址行为依赖 AscendC Compare 的源码调用约定。 |
| y slot | Maxs | CopyOut | 下一 tile | Queue protocol | y 与 x 分离。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| ReLU | `Maxs<int>` | x INT32 | y Queue | y | ReLU 结果保持 INT32 精确。 |
| mask conversion | `Cast(...CAST_RINT)` | x INT32 | FP32 Compare | x slot | INT32→FP32 对绝对值大于 `2^24` 的整数不保证逐值精确，但正负号在有限范围转换中保留；本次未做数值测试。 |
| tail | Queue + aligned DMA | Compute | CopyOut | buffers/GM | 8 INT32 元素、64 compare 元素、32 mask bytes 粒度。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp` | `TilingFunc` | 通用分核。 |
| Kernel entry/router | `experimental/activation/relu_v3/op_kernel/relu_v3.cpp` | `relu_v3<0>` | AIV-only。 |
| Device body/helper | `experimental/activation/relu_v3/op_kernel/relu_v3.h` | INT32 `ReluV3` | `Maxs<int>` 后 x slot 原地 `Cast`，再 Compare。 |

### Route R-004：INT8/UINT8 升 FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `docs/aclnnReluV3.md`；`relu_v3_def.cpp` | `aclnnReluV3*`；`ReluV3` | A2、INT8 或 UINT8 | 对应 8-bit binary。 |
| Tiling | `relu_v3_tiling.cpp` | `TilingFunc` | key 0 | entry 0。 |
| Router | `relu_v3.h` | `relu_v3<T>` | `is_one_of_v<T,int8_t,uint8_t>` | `relu_v3<T,half,half>`。 |

#### Device 数据流

##### 函数级调用链

```text
relu_v3<T,half,half>, T=int8_t|uint8_t
  -> DataCopy T x into oversized x Queue slot
  -> Cast(maskQueue_as_half, xQueue_as_T, CAST_NONE, valid)
  -> Relu(yQueue_half, maskQueue_half, valid)
  -> Cast(yQueue_as_T, yQueue_half, CAST_TRUNC, valid)
  -> Duplicate(xQueue_as_half,0)
  -> Compare(maskQueue_bytes, maskQueue_half, xQueue_half,GT,round128)
  -> CopyOut T y + packed mask
```

##### 源码等价伪码

```text
# 伪码；tile=15360，U=FP16
copy ceil(valid/32)*32 T elements to x slot
mask_slot_half = Cast_NONE(x_slot_T)
y_slot_half = Relu(mask_slot_half)
y_slot_T = Cast_TRUNC(y_slot_half)          # y slot 原地降精度
x_slot_half = 0                             # 输入已消费，复用作 Compare zero
mask_slot_bits = Compare_GT(mask_slot_half,x_slot_half,ceil(valid/128)*128)
copy ceil(valid/32)*32 T y; copy mask 32-byte aligned
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `i` | N bytes each | INT8/UINT8 NC1HWC0 | caller/CopyOut | CopyIn/caller | caller | 32-element tail。 |
| x Queue | slot | 61440 B | 前部 T 输入，随后 FP16 zero | CopyIn/Duplicate | Cast/Compare | TPipe | 输入消费后原地改作 zero buffer。 |
| y Queue | slot | 61440 B | FP16 y，随后前部 T y | Relu/Cast | CopyOut | TPipe | 降精度原地写回。 |
| mask Queue | slot | 61440 B | FP16 widened x，随后 packed bits | Cast/Compare | Relu/Compare/CopyOut | TPipe | 一块同时承担计算输入和 mask 输出。 |
| GM mask | `i/8` | N/8 bytes | packed UINT8 | CopyOut | caller | caller | 32-byte tail。 |
| workspace | N/A | 0 | N/A | N/A | unused | N/A | N/A。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq["x Queue · T input→FP16 zero · 2×30 KiB"]:1
  yq["y Queue · FP16 result→T result · 2×30 KiB"]:1
  mq["mask Queue · widened x→packed mask · 2×30 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 180 KiB；所有升/降精度临时值均落在三个 Queue slot 内
dynamic_ub = N/A
workspace_total = 0
=> tile = 30 KiB / sizeof(half) = 15360 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mask slot | Cast 生成 FP16 x | Relu、Compare src | Compare packed dst | 源码顺序；Compare 后 enqueue | Relu 必须先完成；Compare 的 dst 与 src 共址。 |
| x slot | CopyIn T | Cast | Duplicate FP16 zero | 源码顺序 | Cast 消费完成后才能覆盖。 |
| y slot | Relu FP16 | Cast source | Cast 原地 T；CopyOut | 源码顺序 + Queue | 降精度后按 T 视图出队。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `Cast(...CAST_NONE)` | INT8/UINT8 x | FP16 Relu/Compare | mask slot | 8-bit 整数可被 FP16 精确表示。 |
| 降精度 | `Cast(...CAST_TRUNC)` | FP16 Relu | T CopyOut | y slot | ReLU 对整数产生整数值；UINT8 输入本已非负。 |
| Queue/DMA | EnQue/DeQue/Free | MTE/Vector | 后续 stage | 三 slots | T DMA 32 元素；Compare 128；mask DMA 32 bytes。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp` | `TilingFunc` | dtype 字节数影响 blockDim。 |
| Kernel entry/router | `experimental/activation/relu_v3/op_kernel/relu_v3.cpp` | `relu_v3<0>` | DTYPE_X 编译实例。 |
| Device body/helper | `experimental/activation/relu_v3/op_kernel/relu_v3.h` | 8-bit `ReluV3` | FP16 widen、三 Queue 内原地复用、packed Compare。 |

### Route R-005：BF16 升 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `docs/aclnnReluV3.md`；`relu_v3_def.cpp` | `aclnnReluV3*`；`ReluV3` | A2、BF16 | BF16 instance。 |
| Tiling | `relu_v3_tiling.cpp` | `TilingFunc` | key 0 | entry 0。 |
| Router | `relu_v3.h` | `relu_v3<T>` | `T=bfloat16_t` | `relu_v3<bfloat16_t,float,float>`。 |

#### Device 数据流

##### 函数级调用链

```text
relu_v3<bfloat16_t,float,float>
  -> DataCopy BF16 x
  -> Cast(maskQueue_float, xQueue_bf16, CAST_NONE)
  -> Maxs(yQueue_float, maskQueue_float, 0)
  -> Cast(yQueue_bf16, yQueue_float, CAST_RINT)
  -> Duplicate(xQueue_float,0)
  -> Compare(maskQueue_bytes,maskQueue_float,xQueue_float,GT,round64)
  -> CopyOut BF16 y + packed mask
```

##### 源码等价伪码

```text
# 伪码；U=FP32，tile=7680
copy BF16 x, tail ceil(valid/16)*16
mask_slot_f32 = cast(x_bf16)
y_slot_f32 = max(mask_slot_f32,0)
y_slot_bf16 = Cast_RINT(y_slot_f32)
x_slot_f32 = 0
mask_slot_bits = Compare_GT(mask_slot_f32,x_slot_f32,ceil(valid/64)*64)
copy BF16 y and packed mask with respective 32-byte alignment
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `i` | N BF16 | BF16/NC1HWC0 | caller/CopyOut | CopyIn/caller | caller | 16-element tail。 |
| x Queue | slot | 61440 B | BF16 input，随后 FP32 zero | CopyIn/Duplicate | Cast/Compare | TPipe | 输入消费后复用。 |
| y Queue | slot | 61440 B | FP32 y，随后 BF16 y | Maxs/Cast | Cast/CopyOut | TPipe | 原地降精度。 |
| mask Queue | slot | 61440 B | FP32 widened x，随后 packed mask | Cast/Compare | Maxs/Compare/CopyOut | TPipe | 共址。 |
| GM mask | `i/8` | N/8 bytes | packed UINT8 | CopyOut | caller | caller | 32-byte tail。 |
| workspace | N/A | 0 | N/A | N/A | unused | N/A | N/A。 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq["x Queue · BF16 input→FP32 zero · 2×30 KiB"]:1
  yq["y Queue · FP32 result→BF16 result · 2×30 KiB"]:1
  mq["mask Queue · FP32 x→packed mask · 2×30 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 180 KiB；不额外分配 FP32 TBuf
dynamic_ub = N/A
workspace_total = 0
=> tile = 30 KiB / sizeof(float) = 7680 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mask slot | BF16→FP32 Cast | Maxs、Compare source | Compare packed dst | 源码顺序 + enqueue | Maxs 在 Compare 覆盖前完成。 |
| x slot | BF16 CopyIn | Cast | Duplicate FP32 zero | 源码顺序 | Cast 先完成。 |
| y slot | FP32 Maxs | BF16 Cast | 下一 tile | Queue protocol | CopyOut 以 BF16 视图读取。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `Cast(...CAST_NONE)` | BF16 x | FP32 Maxs/Compare | mask slot | 比较和 ReLU 在 FP32 中执行。 |
| 降精度 | `Cast(...CAST_RINT)` | FP32 y | BF16 CopyOut | y slot | 显式 RINT。 |
| tail | Queue + DMA | stages | stages | all | BF16 DMA 16 元素；Compare 64；mask DMA 32 bytes。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp` | `TilingFunc` | size/blockDim/key。 |
| Kernel entry/router | `experimental/activation/relu_v3/op_kernel/relu_v3.cpp` | `relu_v3<0>` | BF16 instance。 |
| Device body/helper | `experimental/activation/relu_v3/op_kernel/relu_v3.h` | BF16 `ReluV3` | FP32 widen/Maxs、RINT、三 Queue 复用。 |

## 算子特定优化

### Trick T-001：单次读取同时生成 ReLU 与压缩 mask

#### 触发与源码依据

- 适用 Route：R-001～R-005。
- 触发条件：key 0 的任一支持 dtype；每个 tile 在 x Queue 中就绪。
- Golden 来源：[`experimental/activation/relu_v3/op_kernel/relu_v3.h:131-154`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_kernel/relu_v3.h#L131-L154)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<U> x = x_t_que.DeQue<U>();
LocalTensor<U> y = y_t_que.AllocTensor<U>();
LocalTensor<uint8_t> mask = mask_t_que.AllocTensor<uint8_t>();
ReluV3<T, U, V>(y, mask, x, _);
x_t_que.FreeTensor(x);
y_t_que.EnQue(y);
mask_t_que.EnQue(mask);
```

#### 机制与设计原因

同一 x tile 在一次 UB 驻留期间同时供 ReLU 与 `Compare(GT)` 使用，随后 y 与 packed mask 一起入 VECOUT 队列；没有第二次 GM 读取 x，也没有第二个 mask Kernel。Compare 的最后参数由 `ceil_round(valid,256/sizeof(V))` 生成，输出直接为 packed bits；Host 的末维 `%8==0` guard 保证各 core 的 512 元素块起点都对应整数 mask byte offset。

#### 迁移边界

- 适用：需要同时输出逐元素激活和 1-bit predicate 的 elementwise 算子。
- 前提：输出 bit 顺序与 `Compare` packed layout 一致；分核起点能被 8 整除；tail padding 可满足 vector/DMA 粒度。
- 失效：若 mask 必须逐元素 `UINT8`、采用其他 bit order，或 core 起点非 byte-aligned，需改写输出布局或增加转换路线；固定仓没有 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能在未同步消费 x Queue 前提前 `FreeTensor` 或发起下一次覆盖。

### Trick T-002：按 dtype 选择计算宽度并在三个 Queue 内原地复用

#### 触发与源码依据

- 适用 Route：R-003～R-005，R-004/R-005收益路径最完整。
- 触发条件：INT32 需要 FP32 mask 比较；8-bit 使用 FP16；BF16 使用 FP32。
- Golden 来源：[`experimental/activation/relu_v3/op_kernel/relu_v3.h:161-171`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_kernel/relu_v3.h#L161-L171)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (is_one_of_v<T, half, float>)
    relu_v3<T, T, T>(x, y, mask, tiling);
else if constexpr (std::is_same_v<T, int>)
    relu_v3<T, T, float>(x, y, mask, tiling);
else if constexpr (is_one_of_v<T, int8_t, uint8_t>)
    relu_v3<T, half, half>(x, y, mask, tiling);
else if constexpr (std::is_same_v<T, bfloat16_t>)
    relu_v3<T, float, float>(x, y, mask, tiling);
```

#### 机制与设计原因

模板 `U` 决定计算宽度与 30 KiB slot 的元素容量，`V` 决定 Compare 对齐粒度。8-bit 和 BF16 路线先把 x 扩展到 mask Queue，再在 y Queue 中计算并原地降精度；已消费的 x Queue 被覆盖为 zero compare source，mask Queue 最后被 packed result 覆盖。这样无需额外 `TBuf`，UB 始终为 180 KiB。INT32 路线也用同尺寸 reinterpret 在 x slot 内原地转换为 FP32，仅用于符号比较。

#### 迁移边界

- 适用：输入/输出窄类型、计算或比较需要更宽类型，且各阶段生命周期严格串行的 elementwise pipeline。
- 前提：宽类型 slot 已按 `sizeof(U)` 分配；原始输入在覆盖前已被所有消费者使用；原地 Cast/Compare alias 受目标 AscendC API 支持。
- 失效：若原始 x 还需在 Compare 后使用、存在并发 stage，或 primitive 禁止 dst/src alias，必须增加独立 buffer 并重新计算 UB；固定仓没有替代 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；把 Queue 容量按 `sizeof(T)` 而不是 `sizeof(U)` 分配会使 INT8/BF16 的 widened tile 越界。

### Trick T-003：按输入字节数分三级限制 AIV 核数

#### 触发与源码依据

- 适用 Route：R-001～R-005。
- 触发条件：Tiling 成功后，根据 `N*sizeof(dtype)` 命中 `<32 KiB`、`<4 MiB` 或其他档。
- Golden 来源：[`experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp:35-46`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp#L35-L46)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
auto data_size = ge::GetSizeInBytes(size, dtype);
auto block_dim = plaform.GetCoreNumAiv();
if (data_size < 1 << 15)
{
    block_dim = 1;
}
else if (data_size < 1 << 22)
{
    block_dim /= 2;
}
context->SetBlockDim(block_dim);
```

#### 机制与设计原因

小于 32 KiB 只启一核，中等数据用半数 AIV，较大数据用全核；Device 再按 512 元素块比例切连续区间。这是调度 specialization：避免小数据把少量块分散到全部 core，同时保持大数据的核级并行。固定源码只证明分核行为，不提供性能百分比。

#### 迁移边界

- 适用：单次 elementwise、每核无跨核归约、可以用连续块切分的数据域。
- 前提：AIV 核数在半核档整数除二后仍大于 0；阈值按输入字节而非元素复用。
- 失效：跨核依赖、特殊 layout 或 core count=1 的半核档需要单独 guard；当前源码无 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若将阈值误按元素数判断，不同 dtype 会改变原本的字节工作量分档。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 INT8/UINT8 | R-005 BF16 | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | FP16 | FP32 | INT32 | INT8/UINT8 | BF16 | 编译 dtype 实例；均为 key 0。 |
| 模板与 dtype | `<half,half,half>` | `<float,float,float>` | `<int,int,float>` | `<T,half,half>` | `<bf16,float,float>` | `U` 是计算/slot 宽度，`V` 是 Compare 对齐类型。 |
| 循环与分核 | 15360/tile | 7680/tile | 7680/tile | 15360/tile | 7680/tile | 每 slot 固定 30 KiB。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 三 Queue；无 TBuf | 同左 | x INT32→FP32 alias | 三 Queue多阶段 alias | 三 Queue多阶段 alias | 均无 dynamic UB/通信。 |
| 内存总占用与复用 | 180 KiB | 180 KiB | 180 KiB，x 原地 | 180 KiB，x/y/mask 都复用 | 180 KiB，x/y/mask 都复用 | 物理 slot 相同，不按逻辑 tensor 重复计费。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy, Relu, Compare | DataCopy, Maxs, Compare | +Cast RINT | +Cast NONE/TRUNC | +Cast NONE/RINT | 只有 AIV Vector。 |
| 同步、精度与 tail | 原生 FP16，16/128/32B | 原生 FP32，8/64/32B | y INT32，mask 经 FP32，8/64/32B | 经 FP16，32/128/32B | 经 FP32，16/64/32B | 分别为 x/y DMA 元素、Compare 元素、mask DMA bytes。 |
| fallback/失败边界 | 无 | 无 | 无 | 无 | 无 | 非法 dtype/shape 在 Host 失败；未知 key 静默无输出。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | `aclnnReluV3` 公开支持；`ReluV3` 注册 `ascend910b`；本卡发布。 | `aclnnReluV3` 公开文档明确不支持。experimental `aclnnRelu` 文档虽写支持 A3，但其 `Relu` OpDef 只注册 `ascend910b`，构建也未声明 `ascend910_93`，形成发布冲突；不发布 A3 内容。 | `aclnnReluV3` 明确不支持；稳定 `aclnnRelu` 文档支持 950，但本次无法达到源码闭包。 |
| Host/Launcher | 自包含 `TilingFunc`，key 0，按 32 KiB/4 MiB 分核。 | N/A。 | 稳定 `ReluTiling::RunTiling` 调用仓内不存在的 `ElewiseBaseTiling::DoTiling`。 |
| Kernel/资源 | 单 AIV Kernel；三双缓冲 Queue，180 KiB。 | N/A。 | 稳定 entry 调用仓内不存在的 `ElementwiseSch16B::Init/Process` 与 `Vec::*` DAG primitives；UB、POD 字段、循环和同步无法闭合。 |

## 限制与失败边界

- 仅覆盖公开 `aclnnReluV3` 的 A2 路线；不覆盖 `aclnnRelu` 的原地或 AddRelu 变体。
- API 文档要求 5-D，但 Tiling 本身只检查末维 `%8`；rank/format/dtype 的生成式 ACLNN 校验实现固定仓不可见。
- 文档参数表称 `maskOut.shape` 与 `x` 一致，与 `InferShape` 的末维 `/8`、示例和 Kernel packed 地址公式冲突；复用必须使用压缩 mask shape。
- `size` 先在 `long` 中逐维连乘，再检查并窄化到 `int`；源码没有逐维乘法溢出、负维或 `block_dim/2==0` guard。上界比较为 `floor(INT_MAX/512)*512=2147483136` 元素。
- 三个 Queue 固定占 180 KiB，Host 不查询 UB，也没有小 UB fallback。
- 尾 tile 的 x/y `DataCopy` 向上到 32 bytes，mask CopyOut 也向上到 32 bytes；固定仓没有暴露用户 GM 对象在逻辑 extent 之外的物理 padding 保证。迁移实现必须显式保证尾部可访问，或改用支持非对齐尾块的搬运；不能把逻辑 `N`/`N/8` 当作已证明的物理 extent。
- Compare count 比有效元素向上到 64 或 128 元素；padding lane 的值可能参与生成逻辑 mask 末尾之后的 bits，但成功 shape 的有效 mask 只消费前 `N/8` bytes。
- 未知 `schMode` 在 entry 中没有错误返回，只是不调用 body；Host 活动 key 恒为 0。
- A3 未发布：V3 文档明确不支持；experimental `aclnnRelu` 文档声称支持 A3，但实际 OpDef 仅 `.AddConfig("ascend910b")`，`op_host/CMakeLists.txt` 也没有 `ascend910_93` compute-unit 接线。依照“公开支持与活动实现必须一致”的 gate，不能只凭文档发布。
- 950 未生成 `relu_950.md`：固定仓缺少 `op_common/atvoss/elewise/elewise_tiling.h`、`atvoss/elewise/elewise_sch_16b.h`、`atvoss/util/{dag,vec}.h` 等直接依赖的实现，无法闭合 `EleBaseTilingData16B` ABI、tile/loop、UB、Queue、同步与实际 Vector 参数。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `experimental/activation/relu_v3/docs/aclnnReluV3.md` | `aclnnReluV3GetWorkspaceSize`、`aclnnReluV3` | A2 支持、公开签名、dtype/format/rank/错误语义。 |
| 入口实现 | `experimental/activation/relu_v3/op_host/CMakeLists.txt` | `add_modules_sources(... OPTYPE relu_v3 ACLNNTYPE aclnn)` | 生成式公共 ACLNN 接线。 |
| Host/OpDef | `experimental/activation/relu_v3/op_host/relu_v3_def.cpp` | `ops::ReluV3`、`OP_ADD` | 输入输出签名和 `ascend910b` 注册。 |
| Host/Infer | `experimental/activation/relu_v3/op_host/relu_v3_infershape.cpp` | `InferShape`、`InferDataType` | y 同 shape、mask 末维压缩与 dtype。 |
| Host/Tiling | `experimental/activation/relu_v3/op_host/relu_v3_tiling.cpp` | `TilingFunc` | shape guard、size、blockDim、key。 |
| POD | `experimental/activation/relu_v3/op_kernel/relu_v3_tiling_data.h` | `ReluV3TilingData` | 单字段 ABI。 |
| 模板 key | `experimental/activation/relu_v3/op_kernel/relu_v3_tiling_key.h` | `ELEMENTWISE_TPL_SCH_MODE_0`、`ASCENDC_TPL_ARGS_DECL/SEL` | key 0 的声明与选择。 |
| Kernel entry | `experimental/activation/relu_v3/op_kernel/relu_v3.cpp` | `relu_v3<schMode>` | AIV-only、POD 解析和 router。 |
| Kernel body | `experimental/activation/relu_v3/op_kernel/relu_v3.h` | `RELU_V3_SETUP`、`ReluV3` overloads、`relu_v3<T,U,V>` | 分核、UB、DMA、dtype 计算、同步与 tail。 |
| A3 发布审计 | `experimental/activation/relu/docs/aclnnRelu&aclnnInplaceRelu.md` | 产品支持表 | 文档声称 A3 支持。 |
| A3 构建审计 | `experimental/activation/relu/op_host/CMakeLists.txt` | `add_modules_sources` | 未声明 `ascend910_93` compute unit。 |
| A3 注册审计 | `experimental/activation/relu/op_host/relu_def.cpp` | `ops::Relu` | 实际只添加 `ascend910b` AICore config。 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单 `int size` 即可驱动 Device 分核 | 字段顺序、32-bit、元素单位、上界与 Host/Device 共用结构 | `ReluV3TilingData`、`TilingFunc` |
| 调用顺序 | CopyIn→dtype compute→双输出 CopyOut | 8-bit/BF16 的 Cast、Relu/Maxs、Compare 覆盖顺序 | `relu_v3<T,U,V>`、各 `ReluV3` overload |
| workspace/内存 | 核心节点无 workspace；三 Queue 共 180 KiB | depth=2、每 slot 30 KiB、按 `sizeof(U)` 计算 tile | `RELU_V3_SETUP` |
| 同步 | Queue 协议连接 MTE/Vector stages | `EnQue/DeQue/FreeTensor` 顺序，alias buffer 覆盖点 | `relu_v3<T,U,V>` |
| 精度与尾块 | dtype 专用 `U/V` 与 round mode | INT8/UINT8 `CAST_TRUNC`、BF16 `CAST_RINT`、Compare 与 DMA 对齐粒度 | `ReluV3` overloads |
| 失败边界 | Host 拒绝末维非 8 倍数和 size 超限 | 不得把未知 key 静默返回或 180 KiB UB 前提改成成功 fallback | `TilingFunc`、`relu_v3<schMode>` |
