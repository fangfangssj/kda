---
type: Operator
title: "DynamicBlockQuant 950 ACLNN 算子知识"
description: "说明 aclnnDynamicBlockQuantV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/docs/aclnnDynamicBlockQuantV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/dynamic_block_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/dynamic_block_quant_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_kernel/dynamic_block_quant_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_small_blocksize_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_single_row_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_large_blocksize_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/config/ascend950/dynamic_block_quant_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:22:45Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# DynamicBlockQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnDynamicBlockQuantV2GetWorkspaceSize` + `aclnnDynamicBlockQuantV2`，公开两段式 ACLNN；V2 通过版本 2 属性 `dst_type_max` 区别于旧入口，是 950 实际支持的最新公开版本。 |
| 计算语义 | 对每个 `rowBlockSize × colBlockSize` 块求绝对值最大值 `amax`，输出 `scale=min(amax/dstMax, 1/minScale)`，再以 `y=cast(x/scale)` 量化；`minScale=0` 时 `1/minScale` 依 IEEE 除法成为上界 Inf。 |
| 输入输出 | `x` 为 2D/3D ND FP16/BF16；`y` 同 shape，950 为 FP8_E5M2/FP8_E4M3FN/HiFloat8；`scaleOut` 为 FP32，末两维分别按行块、列块向上取整。 |
| 执行类型 | 单 Kernel `dynamic_block_quant`，entry 显式 `KERNEL_TYPE_AIV_ONLY`；三个 Host 可达节点体均为 AIV Vector/MicroAPI，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 `rowBlockSize=1` 的 SingleRow；R-002 非单行且 UB 可容纳至少一块的 SmallBlockSize；R-003 非单行且一块也不能整体放入 UB 的 LargeBlockSize。 |
| 核心资源 | GM `x/y/scale`；R-001/R-002 三个 depth=2 `TQue`；R-003 另有一个 `VECTOR_REG_WIDTH` TBuf；寄存器内绝对值归约、FP32 scale/除法与饱和降精度；算子自有 workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 V2 文档→框架生成 ACLNN bridge→V2 OpDef/shape inference→950 Tiling/POD→AIV-only router→三个完整 MicroAPI body→ascend950 binary/CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

把 2D `x[M,N]` 看成 `B=1`，把 3D `x[B,M,N]` 保持 batch 维。对块 `(b, rb, cb)`，有效域为 `r∈[rb*R,min(M,(rb+1)R))`、`c∈[cb*C,min(N,(cb+1)C))`：

```text
amax[b,rb,cb] = max(abs(x[b,r,c]))
dstMax = 57344 (FP8_E5M2), 448 (FP8_E4M3FN),
         dstTypeMax != 0 ? dstTypeMax : 32768 (HiFloat8)
scale[b,rb,cb] = min(amax / dstMax, 1 / minScale)
y[b,r,c] = saturating_cast(x[b,r,c] / scale[b,rb,cb])
```

Kernel 以清除符号位后的 16-bit 表示做最大值归约；随后升为 FP32 计算 scale 和除法。FP8 使用 `SAT + CAST_RINT`；HiFloat8 的公开 950 入口只允许 `round`，对应模板 `RMode=4` 和 `CAST_ROUND`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，2D `[M,N]` 或 3D `[B,M,N]` | FP16/BF16 | 非空；OpDef `AutoContiguous`，因此 ACLNN 可接受非连续输入并在框架边界连续化。 |
| `minScale` | scalar | double→Host float | `>=0`；默认 0。限制 scale 的上界为 `1/minScale`。 |
| `roundModeOptional` | string | char* | FP8_E5M2/E4M3FN 必须 `rint`；HiFloat8 必须 `round`。 |
| `dstType` | scalar | int64 | 950 仅 34/35/36，即 HiFloat8/FP8_E5M2/FP8_E4M3FN，且必须等于 `yOut` dtype。 |
| `rowBlockSize` | scalar | int64 | 950 为 1/128/256/512。 |
| `colBlockSize` | scalar | int64 | 950 为 64/128/192/256。 |
| `dstTypeMax` | scalar | double→Host float | 0/15/56/224/32768；非零仅对 HiFloat8 合法。 |
| `yOut` | 与 `x` 相同，ND | FP8_E5M2/FP8_E4M3FN/HiFloat8 | 非空；每元素 1 byte。 |
| `scaleOut` | 2D `[ceil(M/R),ceil(N/C)]` 或 3D `[B,ceil(M/R),ceil(N/C)]` | FP32 | 非空；每量化块一个 scale。 |

### 入口与版本

公开文档同时存在 `aclnnDynamicBlockQuant` 与 `aclnnDynamicBlockQuantV2`。固定 OpDef 将 `dst_type_max` 标为 `.Version(2)`，因此本文选择 950 最新公开 V2。V2 第一段产生 workspace 大小和 executor，第二段在给定 stream 上执行；生成的 inner bridge/executor 源码不在该算子目录，按框架边界处理，但算子自有 OpDef、校验、Tiling、POD、launch key、entry/body 和 binary 均可追溯。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段先检查 `x/yOut/scaleOut` 非空、dtype/format/shape。OpDef 对 `x` 声明 `AutoContiguous`；输出由 executor 持有实际地址。算子 Host 依次执行：

1. `InferShapeForDynamicBlockQuant` 令 `y.shape=x.shape`，并按 `R/C` 推导 `scale.shape`；未知 rank 传播为未知 rank。
2. `Tiling4DynamicBlockQuant` 在 950 分支执行 `CheckDtype → GetAttr → CheckShape`。`GetAttr` 校验 `minScale`、输出类型/属性一致、round mode、块尺寸与 `dstTypeMax`。
3. 从 `PlatformAscendC` 取得 AIV 核数、UB 字节数和 vector register 字节数；任何空指针、非法值、核数/UB 非正或 POD 容量不足均返回 `GRAPH_FAILED`。
4. `DoTiling` 计算二维核网格、normal/tail core、UB 因子与 key，写入 `DynamicBlockQuantTilingData`，设置 `blockDim=usedCoreNum`、key 和 workspace 0。

### 参数、Workspace 与生命周期

`rowBlockLoopNum=B*ceil(M/R)`，`colBlockLoopNum=ceil(N/C)`。`AutoTiling` 先令 `usedCoreNum=min(AIV核数,rowBlocks*colBlocks)`（零块时强制 1），枚举其因子 `(m,n)`，优先完整整除；否则按 `(delta,n)` 升序选择二维切核。核心 Kernel 的 `workspaces[0]=0`；entry 参数保留 `workspace` ABI 位置但 950 body 不解引用。executor、连续化临时量及 stream 的 owner/lifetime 属 ACLNN 框架；算子 POD 至 Kernel 返回前只读有效。

### 启动图与失败传播

```text
aclnnDynamicBlockQuantV2GetWorkspaceSize
  -> framework generated DynamicBlockQuant V2 bridge/executor
  -> InferShapeForDynamicBlockQuant
  -> Tiling4DynamicBlockQuant (POD, key, blockDim, workspace=0)
aclnnDynamicBlockQuantV2(workspace, executor, stream)
  -> dynamic_block_quant<<<usedCoreNum, stream>>>(x, y, scale, workspace, tiling)
     -> exactly one of SingleUB / SingleRow / LargeBlockSize bodies
```

唯一 Device 节点为 `dynamic_block_quant`：输入 `x`、输出 `y/scale`、POD `DynamicBlockQuantTilingData`，无节点间 workspace 边。Host 校验/Tiling 失败则不 launch；Kernel 未识别 key 时三个 router helper 均不命中，源码没有 Device 错误回传，合法 Host key 是避免静默无输出的必要条件。

## Route 与参数设计

### 分核与资源计算

令 `P=R*C*(2+1)+32` 字节（一块 FP16/BF16 输入、1-byte 输出和 32-byte scale slot），Host 计算：

```text
maxUbAvailable = floor((ubSize - 1024) / 2 / P)     # 可双缓冲的块数
rowBlocks = B * ceil(M/R); colBlocks = ceil(N/C)
usedCoreNum = rowTileNum * colTileNum
normalRowBlocks = ceil(rowBlocks/rowTileNum)
tailRowBlocks   = floor(rowBlocks/rowTileNum)
normalColBlocks = ceil(colBlocks/colTileNum)
tailColBlocks   = floor(colBlocks/colTileNum)
```

若 `maxUbAvailable>0`，先令 `colUbBlockLoopNum=min(maxUbAvailable,normalColBlocks)`，再除后令 `rowUbBlockLoopNum=min(quotient,normalRowBlocks)`；`rowUbFactor=rowUbBlockLoopNum*R`，`colUbFactor=colUbBlockLoopNum*C`。若为 0，两方向 loop 均为 1，转入大块切行路线。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 SingleRow | 个位 1；`1101/1201/1111/1211/4121/4221` | `blockSizeRow==1`，优先于 UB 不足判断 | `blockSizeRow!=1` → R-002/R-003 |
| 2 | R-003 LargeBlockSize | 个位 2；`1102/1202/1112/1212/4122/4222` | `blockSizeRow!=1 && maxUbAvailable==0` | `maxUbAvailable>0` → R-002 |
| 3 | R-002 SmallBlockSize | 个位 0；`1100/1200/1110/1210/4120/4220` | `blockSizeRow!=1 && maxUbAvailable>0` | 无；合法输入必命中 |

```text
roundDigit = rint ? 1 : round ? 4 : 7
inputDigit = FP16 ? 1 : 2
outputDigit = E5M2 ? 0 : E4M3FN ? 1 : 2
if R == 1: familyDigit = 1
else if maxUbAvailable == 0: familyDigit = 2
else: familyDigit = 0
key = 1000*roundDigit + 100*inputDigit + 10*outputDigit + familyDigit
```

Router 还编译了 HiFloat8 `hybrid` 的 7xxx key，但 Host 的公开可达校验要求 HiFloat8 为 `round`，因此 7xxx 是 dormant，不进入正式 Route。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 SingleRow | 6 个 `...1` key | 950 合法 dtype/shape，`R=1` | 单 AIV Kernel | `DynamicBlockQuantSingleRow<T,U,RMode>::Process/ComputeVF` | 三条 depth=2 queue；逐行、逐列块 MicroAPI；queue 隐式同步 | `UpdateMask` 和独立 tail column block；无 fallback |
| R-002 SmallBlockSize | 6 个 `...0` key | 950 合法域，`R!=1 && maxUbAvailable>0` | 单 AIV Kernel | `DynamicBlockQuantSmallBlockSize<T,U,RMode>::Process/ComputeVF` | 多块合并入三条 depth=2 queue；每块 scale 占 32B slot | row/col normal-tail core + MicroAPI mask |
| R-003 LargeBlockSize | 6 个 `...2` key | 950 合法域，`R!=1 && maxUbAvailable==0` | 单 AIV Kernel | `DynamicBlockQuantLargeBlockSize<T,U,RMode>::ProcessBlock` | 分段重读块：TQue + `VECTOR_REG_WIDTH` max TBuf | 行分段、尾行/尾列；无进一步 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 单行专用 | `blockSizeRow==1` | 避免通用二维块的 per-block 嵌套，直接按行/列块循环 | R-001 | R-002/R-003 | `CalcTilingKey`; `DynamicBlockQuantSingleRow` |
| 整块双缓冲可容纳 | `(ubSize-1024)/2/P > 0` | R-002/R-001 的三条 depth=2 queue 至少要容纳一块 | R-002 | R-003（R=1 优先仍为 R-001） | `CalcPerBlockUbSize`; `DoTiling` |
| 32B scale slot | 每块 `BLOCK_BYTE_32`，不是仅 4B | 单个 FP32 scale 的 GM 尾写需安全 queue slot/对齐 | 全部 | Host UB 因子缩小 | 三个 body 的 `InitBuffer` |
| AIV 核/UB 可用 | `coreNum>0 && ubSize>0` | blockDim 和本地内存生产不可缺失 | 全部 | `GRAPH_FAILED` | `Tiling4DynamicBlockQuant` |
| dtype-round 绑定 | HiF8→round；FP8→rint | template cast trait 固定在 key 中 | 全部 | `GRAPH_FAILED` | `GetAttr`; `dynamic_block_quant_apt.cpp` |
| 尾块 mask | `UpdateMask(remaining)` | 防止无效 lane 进入 max、除法和 store | 全部 | N/A，路线内部尾块 | 各 `ComputeVF`/`Compute*VF` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 生成的序列化结构在固定仓没有展开 `sizeof/alignof/offsetof`；下表保持声明顺序，类型/位宽与 producer/consumer 可闭合，但不臆测具体 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `DynamicBlockQuantTilingData` | `tilingKey` | int64/64 | 宏生成，offset 不可见 | key | `CalcTilingKey` | entry `TILING_KEY_IS` | 公开可达 18 key |
| 同上 | `totalCoreNum` | int64/64 | 同上 | AIV core | `GetCoreNumAiv` | 三 body parse | `>0` |
| 同上 | `ubSize` | int64/64 | 同上 | byte/core | `GetCoreMemSize(UB)` | Large buffer；其余留存 | `>0` |
| 同上 | `vfLen` | int64/64 | 同上 | byte/register | `GetVRegSize` | parse；body 多用编译期 VReg size | 950 分支有效 |
| 同上 | `minScale` | float/32 | 同上 | scale | attr cast | 三 body FP32 reciprocal | `>=0` |
| 同上 | `roundMode` | int64/64 | 同上 | enum | `GetRoundMode` | 模板 key/parse | 1 或 4 可达 |
| 同上 | `dstType` | int64/64 | 同上 | GE dtype enum | attr | parse；模板由 key确定 | 34/35/36 |
| 同上 | `blockSizeRow` | int64/64 | 同上 | element | attr | 分块/循环 | 1/128/256/512 |
| 同上 | `blockSizeCol` | int64/64 | 同上 | element | attr | 分块/地址 | 64/128/192/256 |
| 同上 | `dstTypeMax` | float/32 | 同上 | numeric max | attr cast | HiF8 body | 非 HiF8 必须 0 |
| 同上 | `batchNum` | int64/64 | 同上 | batch | `CalcAxisSize` | GM row-block flatten | 2D 时 1 |
| 同上 | `rowNum` | int64/64 | 同上 | row/batch | `CalcAxisSize` | 地址/tail | shape dim -2 |
| 同上 | `colNum` | int64/64 | 同上 | element/row | `CalcAxisSize` | GM stride/tail | shape dim -1 |
| 同上 | `singleBatchRowBlockLoopNum` | int64/64 | 同上 | row block | `ceil(rowNum/R)` | batch 边界映射 | 正常非空 |
| 同上 | `rowBlockLoopNum` | int64/64 | 同上 | row block | 上项×batch | 分核 | 全 batch |
| 同上 | `colBlockLoopNum` | int64/64 | 同上 | col block | `ceil(colNum/C)` | 分核/scale stride | 正常非空 |
| 同上 | `rowUbBlockLoopNum` | int64/64 | 同上 | block/UB | `SpliteUB` 或 1 | queue loop | `>=1` |
| 同上 | `colUbBlockLoopNum` | int64/64 | 同上 | block/UB | `SpliteUB` 或 1 | queue loop | `>=1` |
| 同上 | `rowUbFactor` | int64/64 | 同上 | row element | row UB blocks×R | queue capacity | route dependent |
| 同上 | `colUbFactor` | int64/64 | 同上 | col element | col UB blocks×C | queue capacity | route dependent |
| 同上 | `usedCoreNum` | int64/64 | 同上 | core | `rowTileNum*colTileNum` | entry/block guard | blockDim 同值 |
| 同上 | `rowTileNum` | int64/64 | 同上 | core grid row | `AutoTiling` | `blockIdx/colTileNum` | `>=1` |
| 同上 | `colTileNum` | int64/64 | 同上 | core grid col | `AutoTiling` | `% colTileNum` | `>=1` |
| 同上 | `normalCoreRowTileNum` | int64/64 | 同上 | row block/core | ceil division | normal core | 至少 1 |
| 同上 | `normalCoreColTileNum` | int64/64 | 同上 | col block/core | ceil division | normal core | 至少 1 |
| 同上 | `tailCoreRowTileNum` | int64/64 | 同上 | row block/core | floor division | tail core | 可比 normal 少 1 |
| 同上 | `tailCoreColTileNum` | int64/64 | 同上 | col block/core | floor division | tail core | 可比 normal 少 1 |
| 同上 | `rowNormalCoreNum` | int64/64 | 同上 | core | remainder（零时=rowTileNum） | normal/tail 判定 | `[1,rowTileNum]` |
| 同上 | `colNormalCoreNum` | int64/64 | 同上 | core | remainder（零时=colTileNum） | normal/tail 判定 | `[1,colTileNum]` |
| 同上 | `rowTailCoreNum` | int64/64 | 同上 | core | rowTile-normal | parse | 正文 body 不直接读取 |
| 同上 | `colTailCoreNum` | int64/64 | 同上 | core | colTile-normal | parse | 正文 body 不直接读取 |
| 同上 | `perCoreRowNum` | int64/64 | 同上 | row/core | A2/A3 I8 producer | A2/A3 body | 950 inactive为默认 0 |
| 同上 | `tailRowList[50]` | int64[50] | 同上 | row index | A2/A3 I8 producer | A2/A3 body | 950 inactive |
| 同上 | `tailColBlockStartList[50]` | int64[50] | 同上 | col block | A2/A3 I8 producer | A2/A3 body | 950 inactive |
| 同上 | `tailColBlockEndList[50]` | int64[50] | 同上 | col block | A2/A3 I8 producer | A2/A3 body | 950 inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 SingleRow | 6 个 `...1` | `dynamic_block_quant_tiling.cpp::DoTiling/CalcTilingKey` | `DynamicBlockQuantTilingData` | `dynamic_block_quant_apt.cpp::SingleRow` | `dynamic_block_quant_single_row_kernel.h::DynamicBlockQuantSingleRow` + common | ascend950 binary JSON + op_host CMake |
| R-002 SmallBlockSize | 6 个 `...0` | 同上 | 同上 | `dynamic_block_quant_apt.cpp::SingleUB` | `dynamic_block_quant_small_blocksize_kernel.h::DynamicBlockQuantSmallBlockSize` + common | 同上 |
| R-003 LargeBlockSize | 6 个 `...2` | 同上 | 同上 | `dynamic_block_quant_apt.cpp::LargeBlockSize` | `dynamic_block_quant_large_blocksize_kernel.h::DynamicBlockQuantLargeBlockSize` + common | 同上 |

## Kernel 实现

### 公共执行流程

三个 helper 顺序被 entry 调用，但只有与当前 key 相等的一个分支实例化执行。它们共享：二维 core grid 映射、GM→depth=2 VECIN queue、寄存器清符号位求绝对最大值、FP32 `scale=min(amax/dstMax,1/minScale)`、FP32 `x/scale`、饱和 cast、VECOUT queue→GM。`DataCopyPad` 处理不对齐矩形，MicroAPI `UpdateMask` 保护尾 lane；queue 的 `Alloc/EnQue/DeQue/Free` 建立搬运和计算阶段的生产消费次序。

### Route R-001：SingleRow 单行块

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/dynamic_block_quant_tiling.cpp` | `CalcTilingKey` | `blockSizeRow==1` | family digit 1 |
| Router | `op_kernel/dynamic_block_quant_apt.cpp` | `SingleRow` | 六个公开 `TILING_KEY_IS(...SINGLE)` | `<T,U,RMode>` 实例 |
| Body | `op_kernel/arch35/dynamic_block_quant_single_row_kernel.h` | `Init/Process` | `blockIdx<usedCoreNum` | 处理核内矩形 |

#### Device 数据流

##### 函数级调用链

```text
dynamic_block_quant -> SingleRow -> DynamicBlockQuantSingleRow::Init
  -> Process -> ProcessRow -> ProcessCol
    -> CopyIn -> Compute -> ComputeVF -> CopyOut -> CopyOutScale
```

##### 源码等价伪码

```text
core=(blockIdx/colTileNum, blockIdx%colTileNum)
derive normal/tail row and col ranges
for row UB slice, including tail:
  for col UB slice, including tail:
    DataCopyPad x rectangle to inQueue
    for each row and each col block:
      mask-load lanes; clear sign bit; ReduceMax
      scale=min(float(amax)/dstMax, 1/minScale); store one FP32
      reload block; cast to FP32; divide by scale; saturating cast U; packed store
    copy valid y rectangle and compact scale rectangle to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `rowBlockOffset*N+colBlockOffset*C + local` | 核内 normal/tail rectangle | T/U ND | ACLNN / Kernel | CopyIn / caller | caller-owned | DataCopyPad tail |
| GM scale | `rowBlockOffset*colBlocks+colBlockOffset` | 核内 block grid | FP32 ND | CopyOutScale | caller | caller-owned | compact 4B values |
| `inQueue_` | queue slot | `rowUbFactor*colUbFactor*2B` each | T compact rows | CopyIn | ComputeVF | TPipe，DeQue 后释放 | depth=2 |
| `outQueue_` | queue slot | `rowUbFactor*colUbFactor*1B` each | U compact rows | ComputeVF | CopyOut | TPipe | depth=2 |
| `scaleQueue_` | 8-float slot per block | `rowUbBlocks*colUbBlocks*32B` each | FP32 padded | ComputeVF | CopyOutScale | TPipe | depth=2 |
| RegTensor | vector lanes | implementation-defined | T/FP32/U | MicroAPI load/math | packed store | `__VEC_SCOPE__` | masks protect tail |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inQueue · depth=2 · 2*rowUbFactor*colUbFactor*2B"]:1
  outQ["outQueue · depth=2 · 2*rowUbFactor*colUbFactor*1B"]:1
  scaleQ["scaleQueue · depth=2 · 2*rowUbBlocks*colUbBlocks*32B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4*rowUbFactor*colUbFactor
         + 2*rowUbFactor*colUbFactor
         + 64*rowUbBlocks*colUbBlocks
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| in queue slot | `CopyIn` | `Compute` | 下一 UB slice | `EnQue→DeQue→FreeTensor` | Compute 完成后释放 |
| out queue slot | `ComputeVF` | `CopyOut` | 下一 slice | `EnQue→DeQue→FreeTensor` | DMA 消费后释放 |
| scale queue slot | `ComputeVF` | `CopyOutScale` | 下一 slice | 同上 | GM scale 写完后释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | TQue protocol | CopyIn | Compute DeQue | x slot | DataCopyPad row/col tail |
| reduce | `UpdateMask/And/Max/ReduceMax` | register load | scale math | valid x lanes | 16-bit abs ordering |
| quant | FP32 Div + CastTrait | scale and x cast | packed store | result lanes | FP8 RINT；HiF8 ROUND；SAT |
| UB→GM | TQue protocol | Compute | CopyOut | y/scale slots | only valid extents written |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/dynamic_block_quant_tiling.cpp` | `CalcTilingKey` | R=1 优先 key |
| Kernel entry/router | `op_kernel/dynamic_block_quant_apt.cpp` | `SingleRow` | 6 key→模板映射 |
| Device body/helper | `op_kernel/arch35/dynamic_block_quant_single_row_kernel.h` | `ProcessCol/ComputeVF` | 矩形 DMA、逐块归约和 tail mask |

### Route R-002：SmallBlockSize 多块合并

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/dynamic_block_quant_tiling.cpp` | `CalcTilingKey` | `R!=1 && maxUbAvailable!=0` | family digit 0 |
| Router | `op_kernel/dynamic_block_quant_apt.cpp` | `SingleUB` | 六个公开 normal key | SmallBlockSize |
| Body | `...small_blocksize_kernel.h` | `Init/Process` | normal/tail 2D core | 多 block/UB |

#### Device 数据流

##### 函数级调用链

```text
dynamic_block_quant -> SingleUB -> DynamicBlockQuantSmallBlockSize::Init
  -> Process -> CopyIn -> Compute -> ComputeVF -> CopyOut
```

##### 源码等价伪码

```text
for rowUbLoop in core row-block range:
  blockRow = min(rowUbBlocks, remaining)
  for colUbLoop in core col-block range:
    blockCol = min(colUbBlocks, remaining)
    for each block: DataCopyPad valid rectangle into 32B-aligned compact UB subrange
    for each block: masked abs-max; scale; FP32 divide; saturating output cast
    for each block: DataCopyPad valid y rectangle and one FP32 scale
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM rectangles | batch/row-block/col-block offsets from `preRowBlockNum_` | UB-selected block grid | ND T/U | caller/CopyOut | CopyIn/caller | caller | each compact subblock 32B aligned |
| input/output queues | cumulative aligned block extents | per slot `rowUbFactor*colUbFactor*sizeof(T/U)` | compact blocks | CopyIn/Compute | Compute/CopyOut | TPipe | depth=2 |
| scale queue | `(rowIdx*blockCol+colIdx)*8` floats | per slot `rowUbBlocks*colUbBlocks*32B` | FP32 padded | ComputeVF | CopyOut | TPipe | depth=2 |
| registers | per current block | VReg | T/FP32/U | loads | stores | vec scope | UpdateMask |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inQueue · depth=2 · 4*rowUbFactor*colUbFactor B"]:1
  outQ["outQueue · depth=2 · 2*rowUbFactor*colUbFactor B"]:1
  scaleQ["scaleQueue · depth=2 · 64*rowUbBlocks*colUbBlocks B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 6*rowUbFactor*colUbFactor + 64*rowUbBlocks*colUbBlocks
         = 2*rowUbBlocks*colUbBlocks*(R*C*3 + 32)
UB_total <= ubSize - 1024
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三 queue slots | CopyIn/Compute | Compute/CopyOut | 下一双缓冲迭代 | queue EnQue/DeQue/Free | 物理 slot 释放后才覆盖 |
| scale 32B slots | ComputeVF | per-block CopyOut | 下一 UB grid | scale queue 生命周期 | 每块只首个 FP32 有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| blocks DMA | queue + DataCopyPad | CopyIn | Compute | compact subblocks | tail row/col exact bytes |
| reduction | register mask | x load | scale | invalid lanes | bitwise abs max |
| output | CastTrait + queue | FP32 division | CopyOut | y/scale | saturated 8-bit store |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/dynamic_block_quant_tiling.cpp` | `SpliteUB/DoTiling` | 多块 UB 因子与 key |
| Kernel entry/router | `op_kernel/dynamic_block_quant_apt.cpp` | `SingleUB` | 公开 normal key |
| Device body/helper | `op_kernel/arch35/dynamic_block_quant_small_blocksize_kernel.h` | `CopyIn/ComputeVF/CopyOut` | 多块合并 DMA、32B scale slot |

### Route R-003：LargeBlockSize 分段重读

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/dynamic_block_quant_tiling.cpp` | `CalcTilingKey` | `R!=1 && maxUbAvailable==0` | family digit 2 |
| Router | `op_kernel/dynamic_block_quant_apt.cpp` | `LargeBlockSize` | 六个公开 large key | LargeBlockSize |
| Body | `...large_blocksize_kernel.h` | `ProcessBlock` | 单块跨多个 UB row slice | 两遍处理 |

#### Device 数据流

##### 函数级调用链

```text
dynamic_block_quant -> LargeBlockSize -> InitGmOffset -> InitBuffer
  -> Process -> ProcessBlock
    -> repeated CopyIn/ComputeXTmpMax
    -> ReduceMax -> ComputeScaleVF
    -> retained last slice ComputeOutVF/CopyOutY
    -> re-CopyIn earlier/later slices -> ComputeOutVF/CopyOutY
    -> CopyOutScale
```

##### 源码等价伪码

```text
for each assigned logical block:
  zero xLocalMaxBuffer
  for rowSlice from last down to first:
    load slice; accumulate sign-cleared max into xLocalMaxBuffer
    retain first logical slice's queue tensor, free others
  reduce accumulated max; scale=min(max/dstMax,1/minScale)
  quantize retained slice; free it; write y
  for remaining slices from 1 upward:
    reload x; quantize with same scale; write y
  write one FP32 scale
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `base + rowSlice*sliceRows*N` | `sliceRows*blockCols` | strided ND T | caller | CopyIn | caller | reread for quant pass |
| in/out queue | slot | `sliceRows*blockCols*sizeof(T/U)` | compact | CopyIn/ComputeOut | max/quant/CopyOut | TPipe | depth=2 |
| scale queue | slot | 32B | one FP32 valid | ComputeScaleVF | CopyOutScale | TPipe | depth=2 |
| `xLocalMaxBuffer_` | 0 | `VECTOR_REG_WIDTH` B | T bits | ComputeXTmpMax | ReduceMax/scale | TPipe whole block | reused next block after complete |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 2*sliceRows*blockCols*2B"]:1
  outQ["outQueue · depth=2 · 2*sliceRows*blockCols*1B"]:1
  scaleQ["scaleQueue · depth=2 · 64B"]:1
  maxBuf["xLocalMaxBuffer · VECTOR_REG_WIDTH B"]:1
```

##### 字节账与复用顺序

```text
perBlockSize = 2*normalBlockRow*normalBlockCol*(sizeof(T)+sizeof(U)) + 64
sliceRows = ceil(normalBlockRow / ceil(perBlockSize/(ubSize-VECTOR_REG_WIDTH)))
UB_total = 2*sliceRows*normalBlockCol*2
         + 2*sliceRows*normalBlockCol*1 + 64 + VECTOR_REG_WIDTH
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| max TBuf | each max pass | final ReduceMax | next block zero-fill | block-serial ProcessBlock | scale 已产生 |
| retained in slot | reverse max loop last iteration (`rowLoopIdx==0`) | first quant output | then FreeTensor | explicit conditional free | 不可在 scale 前释放 |
| other in/out slots | CopyIn/ComputeOut | max/CopyOut | next slice | queue lifecycle | FreeTensor after consumer |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| max pass | queue + TBuf accumulation | CopyIn | ReduceMax | all row slices | `UpdateMask(xTotalNum)` |
| scale pass | FP32 Div/Min | ReduceMax | quant passes | one scale | Inf compare mask gates Min |
| quant pass | GM reread + broadcast scale | scale | CopyOutY | each slice | last slice exact rows/cols |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/dynamic_block_quant_tiling.cpp` | `CalcTilingKey` | zero whole-block capacity selects large |
| Kernel entry/router | `op_kernel/dynamic_block_quant_apt.cpp` | `LargeBlockSize` | 6 key→body |
| Device body/helper | `op_kernel/arch35/dynamic_block_quant_large_blocksize_kernel.h` | `InitBuffer/ProcessBlock/Compute*VF` | 分段 max、一次 scale、重读 quant |

## 算子特定优化

### Trick T-001：按整块 UB 容量切换“合并多块”与“两遍分段”

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：`blockSizeRow!=1`，以 `maxUbAvailable == 0` 为边界。
- Golden 来源：[`dynamic_block_quant_tiling.cpp:493-509`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.cpp#L493-L509)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t maxUbAvailable = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER / perBlockUbSize;

CalcTilingKey(inDtype, outDtype, tilingParam, maxUbAvailable);
if (maxUbAvailable == 0) {
    tilingParam.colUbBlockLoopNum = 1;
    tilingParam.rowUbBlockLoopNum = 1;
```

#### 机制与设计原因

能整体容纳时，R-002 把多个逻辑块合入一次双缓冲 DMA/compute grid；不能容纳时，R-003 保留一个 VReg 宽 max TBuf，以多 row slice 完成全块 max，再用同一 scale 量化各 slice。这避免按完整超大块分配 UB，同时只物化一个归约状态；代价是量化阶段重读 GM，源码没有声称具体性能百分比。

#### 迁移边界

- 适用：先全域归约得到一个标量、再对全域逐元素变换的块算法。
- 前提：950 AIV、可将归约状态压缩为 VReg/TBuf、第二遍 GM 重读语义安全。
- 失效：若块可整体容纳应走 R-002；若算法不可重读或 scale 依赖逐元素状态，不能直接迁移。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的真实反例边界是强行把 `maxUbAvailable==0` 的块分配到 SmallBlockSize：其 depth=2 input/output/scale 容量无法由 Host UB 公式保证；官方实现转而使用 LargeBlockSize。

### Trick T-002：16-bit 符号位清除后直接做绝对值最大值归约

#### 触发与源码依据

- 适用 Route：全部
- 触发条件：输入仅 FP16/BF16，非 NaN 数值按正数 bit pattern 可比较。
- Golden 来源：[`dynamic_block_quant_small_blocksize_kernel.h:370-381`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_small_blocksize_kernel.h#L370-L381)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::DataCopy(vreg1, xAddr + i * vfLen);
AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3,
    (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1,
    (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, preg0);
AscendC::MicroAPI::Max<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
    expMaxRegTensor, expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3, preg0);
```

#### 机制与设计原因

`maxValue_=0x7fff` 清除 sign bit，直接得到 FP16/BF16 的绝对值编码；对非负有限 16-bit 浮点，编码序与数值序一致，因此可用 uint16 Max/ReduceMax，直到得到块最大值后才 cast FP32。它减少逐 lane FP32 abs 中间物化，并让 tail 由同一 `preg0` 保护。

#### 迁移边界

- 适用：IEEE-like 16-bit 浮点的绝对最大值归约。
- 前提：输入 dtype 编码与该顺序性质一致，NaN/Inf 语义满足目标接口；mask 必须覆盖尾 lane。
- 失效：FP32、非 IEEE 编码、需要保留 NaN 次序或符号信息时须改用数值 Abs/compare。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 SingleRow | R-002 SmallBlockSize | R-003 LargeBlockSize | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | R=1，最高 | R!=1 且整块可容纳 | R!=1 且整块不可容纳 | 行块特化优先，再看 UB |
| 模板与 dtype | 6 个 `<T,U,RMode>` | 同 6 组合 | 同 6 组合 | dtype 只改变模板，不改变拓扑 |
| 循环与分核 | row slice→col slice→逐块 | row-block UB grid→col-block grid | 每逻辑块→row slice 两遍 | 与块大小/UB 相配 |
| Queue/TBuf/寄存器 | 3×depth2 queue | 3×depth2 queue | 3×depth2 + max TBuf | 大块保存压缩归约状态 |
| 内存总占用与复用 | UB factor 公式 | Host 精确约束≤UB-1024 | slice 公式≤UB，另留 VReg | 防止超配 |
| 搬运/Vector API | 矩形 DataCopyPad + MicroAPI | 多块 compact DataCopyPad | GM 重读 + MicroAPI | 整块可容纳时减少重读 |
| 同步、精度与 tail | queue；FP32 scale/div；mask | 同左 | 同左，跨 slice max TBuf | 语义一致 |
| fallback/失败边界 | 无 | UB=0 capacity→R-003 | 无更低 fallback | Host 合法 key 必须命中 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 文档支持，但固定源码与公开 `colBlockSize=1` 域冲突，未在本文发布 | 同 A2 | 支持，本文闭合 |
| Host/Launcher | I8 专用 `DynamicBlockQuantI8` | 同 A2 | FP8/HiF8 通用 `DoTiling`，三 family key |
| Kernel/资源 | 旧 `DynamicBlockQuantND<T>`，INT8 | 同 A2 | arch35 三 MicroAPI body，AIV-only |

## 限制与失败边界

- 950 不支持 INT8 输出；HiFloat8 不允许 `rint`，FP8 不允许 `round`；router 中 hybrid key 不属于公开 Host 可达域。
- `dstTypeMax` 非零时只允许 HiFloat8；块尺寸集合固定；shape 仅 2D/3D且非空。
- `DynamicBlockQuantTilingData` 的宏展开 ABI offset/sizeof 在固定源码不可见；复用必须保持声明顺序、字段位宽和同一序列化宏。
- `InferShapeForDynamicBlockQuant` 的 3D 日志局部变量 `dim2` 读取 `dim0`，但实际 `scaleShape` 第二维使用 `inputXShape->GetDim(2)`；这只影响日志值，不改变推导结果。
- unknown key 的 Device 分支静默不写输出；Host key 校验与 entry/binary 一致性不可破坏。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/dynamic_block_quant/docs/aclnnDynamicBlockQuantV2.md` | 两段式原型/产品表 | V2 API、950 支持和参数域 |
| 入口实现 | `quant/dynamic_block_quant/op_host/dynamic_block_quant_def.cpp` | `DynamicBlockQuant`, `.Version(2)` | V2 属性、950 OpDef、apt 文件名 |
| shape | `quant/dynamic_block_quant/op_host/dynamic_block_quant_infershape.cpp` | `InferShapeForDynamicBlockQuant` | y/scale shape 和 dtype |
| Host/Tiling/POD | `quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.h` | `DynamicBlockQuantTilingData` | POD 字段 |
| Host/Tiling/Launcher | `quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.cpp` | `Tiling4DynamicBlockQuant` | 校验、分核、UB、key、workspace |
| Kernel entry/router | `quant/dynamic_block_quant/op_kernel/dynamic_block_quant_apt.cpp` | `dynamic_block_quant`, `SingleUB/SingleRow/LargeBlockSize` | AIV-only 和 key 映射 |
| Kernel common | `quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_common.h` | constants/CastTrait | dtype max 与 cast mode |
| Kernel body | `quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_single_row_kernel.h` | `DynamicBlockQuantSingleRow` | R-001 |
| Kernel body | `quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_small_blocksize_kernel.h` | `DynamicBlockQuantSmallBlockSize` | R-002 |
| Kernel body | `quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_large_blocksize_kernel.h` | `DynamicBlockQuantLargeBlockSize` | R-003 |
| 构建接线 | `quant/dynamic_block_quant/op_host/config/ascend950/dynamic_block_quant_binary.json` | `DynamicBlockQuant_*` | 950 六 dtype binaries |
| 构建接线 | `quant/dynamic_block_quant/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN Host 注册 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Host 统一 POD 支持三 950 family | 声明顺序、int64/float 位宽、inactive A2/A3 数组 | `DynamicBlockQuantTilingData` |
| 调用顺序 | 单 Kernel，router 仅一个 key body命中 | key digit 编码和 Host 优先级 | `CalcTilingKey`; entry helpers |
| workspace/内存 | workspace 0；三 queue；large max TBuf | depth=2、32B scale slot、1024B Host reserve | `DoTiling`; `InitBuffer` |
| 同步 | queue 生命周期和 MicroAPI vec scope | DeQue/Free 前后关系；retained slice 特例 | 三 body `Process*` |
| 精度与尾块 | 16-bit abs max→FP32 scale/div→8-bit SAT cast | dtype max、round trait、UpdateMask | common + `ComputeVF` |
| 失败边界 | Host 非法域失败；unknown key 无输出 | 不得使 dormant hybrid/未知 key 可达 | `GetAttr`; router |