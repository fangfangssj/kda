---
type: Operator
title: "GroupedDynamicBlockQuant 950 ACLNN 算子知识"
description: "说明 aclnnGroupedDynamicBlockQuant 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/docs/aclnnGroupedDynamicBlockQuant.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_kernel/grouped_dynamic_block_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_kernel/arch35/grouped_dynamic_block_quant_small_block.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_kernel/arch35/grouped_dynamic_block_quant_large_block.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_host/config/ascend950/grouped_dynamic_block_quant_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
created_at: "2026-08-25T18:28:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GroupedDynamicBlockQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnGroupedDynamicBlockQuantGetWorkspaceSize` + `aclnnGroupedDynamicBlockQuant`，公开 ACLNN 两段式接口；固定仓仅有这一公开版本。 |
| 计算语义 | 对每个 batch、cumsum group、`rowBlockSize × colBlockSize` 有效块求 `amax(abs(x))`，计算 `scale=min(amax/FP8_MAX, 1/minScale)`，再以指定 round mode 饱和转换 `y=cast(x/scale)`。 |
| 输入输出 | `x`: `[M,N]`/`[B,M,N]` FP16/BF16 ND；`groupList`: `[g]` INT32；`y`: 同 shape 的 HiF8/FP8；`scale`: `[M/r+g,ceil(N/c)]` 或 `[B,M/r+g,ceil(N/c)]` FP32。 |
| 执行类型 | 单 Kernel、单 AIV 节点 `AIV_VECTOR`；entry 显式 `KERNEL_TYPE_AIV_ONLY`，所有公开可达 Route 均无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 SmallBlock：一个行量化块可由 UB tile 覆盖；R-002 LargeBlock：一个行量化块需按 `maxUbRow` 分段并两遍读取。每条 Route 内含 8 个 dtype/round 编译期实例。 |
| 核心资源 | GM `x/groupList/y/scale`；双缓冲 `VECIN/VECOUT` queues；R-002 另有一个 vector-register-width 的 `TBuf`；workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→`ACLNNTYPE aclnn` OpDef 接线→infer/tiling/POD→AIV-only router→Small/Large MicroAPI body→Ascend 950 binary/config 闭合；自动生成 ACLNN bridge/executor 是框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `r=rowBlockSize`、`c=colBlockSize`，`groupList[k]` 是第 `k` 个 group 在 M 轴的累积结束位置，`s_k=(k==0?0:groupList[k-1])`、`e_k=groupList[k]`。对 batch `b`、group `k`、行块 `q`、列块 `n`，有效域为：

```text
rows = [s_k + q*r, min(s_k + (q+1)*r, e_k))
cols = [n*c, min((n+1)*c, N))
amax  = max(abs(x[b, rows, cols]))
Fmax  = 57344 (FP8_E5M2) / 448 (FP8_E4M3FN) / 32768 (HiF8)
scale = min(amax / Fmax, 1 / minScale)
y     = saturating_cast_target(x / scale, RINT/ROUND/HYBRID)
```

`scale` 的 M 向物理索引为 `s_k/r + k + q`，额外的 `+k` 为每个 group 的不足整块预留一行；N 向索引为 `n`。Device 以清除 FP16/BF16 符号位的方式求绝对值，再先在寄存器内归约，必要时跨 UB 分段归约。`minScale==0` 时源码执行 FP32 `1/0`；后续只在候选 scale 小于 `+Inf` 时用该上限做 `Min`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[M,N]` 或 `[B,M,N]`，ND | FP16/BF16 | 非空；OpDef `AutoContiguous()`，公开接口允许非连续输入。 |
| `groupList` | `[g]`，ND | INT32 | 非空、值非负且非递减，末值等于 M；公开 API 仅允许 cumsum 语义。OpDef 同样 `AutoContiguous()`。 |
| `minScale` | scalar | ACLNN 为 `double`，OpDef/POD 为 FP32 | `>=0`；进入 tiling 时读取为 `float`。 |
| `roundModeOptional` | string | char pointer | 空指针的公开默认值为 `rint`；FP8 仅 `rint`，HiF8 仅 `round`/`hybrid`。 |
| `dstType` | scalar | INT64 | `34/35/36` 对应 HiF8/FP8_E5M2/FP8_E4M3FN，且必须与 `yOut` dtype 一致。 |
| `rowBlockSize` | scalar | INT64 | 仅 `1/128/256/512`。 |
| `colBlockSize` | scalar | INT64 | 仅 `64/128/192/256`。 |
| `groupListType` | scalar | INT64 | 公开 ACLNN 仅 `0`；Host tiling 代码还接受 `1`，但 Device 仍按 cumsum 读取，故 `1` 不属于本卡公开可达域。 |
| `yOut` | 与 `x` 相同，ND | HiF8/FP8_E4M3FN/FP8_E5M2 | 每个有效输入元素写一次。 |
| `scaleOut` | 2D/3D，ND | FP32 | Host 实际要求 N 维为 `ceil(N/c)`；可为零元素 Tensor，但 `x/groupList` 不可为空。 |

### 入口与版本

公开版本只有无版本后缀的 `aclnnGroupedDynamicBlockQuant`。第一段接口验证元数据、形成 executor 并返回 workspace 大小；第二段在调用者提供的 stream 上执行。固定仓未提交自动生成的 ACLNN bridge 源码，`CMakeLists.txt` 的 `ACLNNTYPE aclnn` 与 OpDef/公开文档共同确定该框架边界；算子自有的 infer、tiling、POD、router 和 body 均可见。README/API 产品表明确仅 Ascend 950PR/950DT 支持，A2/A3 明确为不支持。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段要求 `x/groupList/yOut/scaleOut` 非空，检查 dtype/format/rank/shape 与属性；文档给出的失败为 `ACLNN_ERR_PARAM_NULLPTR` 或 `ACLNN_ERR_PARAM_INVALID`。
2. OpDef 对 `x`、`group_list` 启用 `AutoContiguous()`；连续化的具体 workspace/executor 代码由框架生成，固定仓不可见。
3. `InferShapeForGroupedDynamicBlockQuant` 令 `y.shape=x.shape`；2D scale 为 `[M/r+g,ceil(N/c)]`，3D scale 为 `[B,M/r+g,ceil(N/c)]`；`InferDataTypeForGroupedDynamicBlockQuant` 令 `y.dtype=dst_type`、`scale.dtype=FP32`。
4. `Tiling4GroupedDynamicBlockQuant` 依次执行 `CheckDtype`、`GetAttr`、`CheckShape`，再读取 AIV 核数、UB 字节数与 vector-register 字节数，执行 `DoTiling`、序列化 POD、设置 `blockDim/key/workspace`。任一步失败立即返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

`totalCoreNum=GetCoreNumAiv()`，`ubSize=GetCoreMemSize(UB)`，`vfLen=Ops::Base::GetVRegSize(context)`。`workspace[0]=0`，Kernel 的 `workspace` 参数未读取；因此算子自有节点间 workspace 为 N/A。executor 与可能的输入连续化临时区由 ACLNN 框架拥有，固定仓未暴露其 offset/lifetime；算子 Kernel 直接消费连续 GM 地址，输出由调用者拥有并在 stream 完成后可见。

### 启动图与失败传播

```text
aclnnGroupedDynamicBlockQuantGetWorkspaceSize
  -> generated ACLNN bridge / AutoContiguous (framework boundary)
  -> InferShapeForGroupedDynamicBlockQuant + InferDataTypeForGroupedDynamicBlockQuant
  -> Tiling4GroupedDynamicBlockQuant
       -> SetBlockDim(usedCoreNum), SetTilingKey(tilingKey), workspace[0]=0
  -> executor
aclnnGroupedDynamicBlockQuant(workspace=unused, executor, stream)
  -> grouped_dynamic_block_quant<<<usedCoreNum, stream>>>(x, groupList, y, scale, workspace, tiling)
       -> exactly one SmallBlock or LargeBlock AIV template instance
```

唯一 Device 节点的 entry 是 `grouped_dynamic_block_quant`，POD 为 `GroupedDynamicBlockQuantTilingData`，输入/输出直接为调用者 GM tensor，无 kernel 间边。Host tiling 失败则 executor 不形成/launch 不发生；entry 对已知 key 进入一个分支，源码没有显式 unknown-key 错误分支，未知 key 会跳过计算。stream 内可见性由 ACLNN launch 框架保证，固定仓未暴露更低层 launch 包装。

## Route 与参数设计

### 分核与资源计算

设 `uo=ceil(N/c)`、`S=uo*B`。若 `S>=AIVCoreNum`，使用全部 AIV 核，将 `(batch,n)` tile 均分为 head/tail core，再把每核 tile 数乘 `g`；否则先把工作量扩为 `S*g` 后取 `usedCoreNum=min(S*g,AIVCoreNum)`。统一有：

```text
headWork = ceil(work / usedCoreNum)
tailWork = headWork - 1
tailCoreNum = headWork * usedCoreNum - work
headCoreNum = usedCoreNum - tailCoreNum
```

其中 `work=S`（随后每核工作数乘 `g`）或 `S*g`。列 tile 的 `blockFactor=(N<=c?0:c)`，末 tile `tailBlockFactor=(N%c==0?c:N%c)`；`blockFactor=0` 只用于 `nIdx=0` 的基址，实际 `dataLen` 仍取 `tailBlockFactor=N`。

Host 以固定字节常量 `sizeof(input)=2`、`sizeof(output)=1`、scale 槽对齐 `8*4=32`、双缓冲深度 2 计算：

```text
E = 32 + r*c*(2+1)
maxUbAvailableRows = r * floor((UB/2) / E)
small = (r <= maxUbAvailableRows)
small.maxUbRow = floor(maxUbAvailableRows/r) * r
large.maxUbRow = floor(r * floor((UB-vfLen)/2) / E)
```

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 SmallBlock | `1 T U R`：1111/1211、1121/1221、1134/1234、1137/1237 | `rowBlockSize <= rowBlockSize*floor((UB/2)/E)`；百位 FP16/BF16，十位 E4M3/E5M2/HiF8，个位 rint/round/hybrid。 | 条件假进入 R-002。 |
| 2 | R-002 LargeBlock | `2 T U R`：2111/2211、2121/2221、2134/2234、2137/2237 | `rowBlockSize > rowBlockSize*floor((UB/2)/E)`；其余 key 位含义同上。 | 已通过 dtype/attr/shape gate 后无下一 Route。 |

```text
if r <= r * floor((UB/2)/(32 + 3*r*c)):
    thousand = 1; body = GroupedDynamicBlockQuantSmallBlock
else:
    thousand = 2; body = GroupedDynamicBlockQuantLargeBlock
hundred = 1 if x==FP16 else 2
ten = 1 if y==E4M3FN else 2 if y==E5M2 else 3
ones = 1(rint) / 4(round) / 7(hybrid)
key = 1000*thousand + 100*hundred + 10*ten + ones
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 SmallBlock | 1111/1211 | small；FP16/BF16→E4M3FN；rint；950 | 单 AIV Kernel | `SmallBlock<half/bfloat16_t,fp8_e4m3fn_t,1>::Process/ComputeAlign` | 三个 depth-2 queue；每 UB tile 同时产出多个 row-block 的 y/scale。 | `UpdateMask` + GM 2D DMA tail。 |
| R-001 SmallBlock | 1121/1221 | small；FP16/BF16→E5M2；rint；950 | 同上 | `SmallBlock<half/bfloat16_t,fp8_e5m2_t,1>` | 同上，Fmax=57344。 | 同上。 |
| R-001 SmallBlock | 1134/1234 | small；FP16/BF16→HiF8；round；950 | 同上 | `SmallBlock<half/bfloat16_t,hifloat8_t,4>` | 同上，cast trait 为 `CAST_ROUND`。 | 同上。 |
| R-001 SmallBlock | 1137/1237 | small；FP16/BF16→HiF8；hybrid；950 | 同上 | `SmallBlock<half/bfloat16_t,hifloat8_t,7>` | 同上，cast trait 为 `CAST_HYBRID`。 | 同上。 |
| R-002 LargeBlock | 2111/2211 | large；FP16/BF16→E4M3FN；rint；950 | 单 AIV Kernel | `LargeBlock<half/bfloat16_t,fp8_e4m3fn_t,1>::Process` 与三个 VF helper | 三个 depth-2 queue + `vfLen` TBuf；先归约所有分段，再量化。 | 每段 `UpdateMask`；首段保留，其他段二次读 GM。 |
| R-002 LargeBlock | 2121/2221 | large；FP16/BF16→E5M2；rint；950 | 同上 | `LargeBlock<half/bfloat16_t,fp8_e5m2_t,1>` | 同上，Fmax=57344。 | 同上。 |
| R-002 LargeBlock | 2134/2234 | large；FP16/BF16→HiF8；round；950 | 同上 | `LargeBlock<half/bfloat16_t,hifloat8_t,4>` | 同上，cast `CAST_ROUND`。 | 同上。 |
| R-002 LargeBlock | 2137/2237 | large；FP16/BF16→HiF8；hybrid；950 | 同上 | `LargeBlock<half/bfloat16_t,hifloat8_t,7>` | 同上，cast `CAST_HYBRID`。 | 同上。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 整行块可驻留 | `r <= maxUbAvailableRows` | R-001 同一 UB tile 同时完成多个 row block 的 reduce/scale/cast。 | R-001 | R-002 两遍分段。 | `DoTiling` |
| R-002 预留归约 TBuf | `maxUbRow` 使用 `(UB-vfLen)/2` | `xLocalMaxBuffer_` 固定分配 `VECTOR_REG_WIDTH` 字节。 | R-002 | 无 fallback；由 Host 减去 `vfLen`。 | `DoTiling`; `LargeBlock::Init` |
| GM 2D DMA 计数窄化 | `rowNum/blockCount` 最终转 `uint16_t`，`dataLen*2`/stride 转 `uint32_t` | `DataCopyExtParams` 字段位宽要求。 | 两 Route | Host 未显式检查溢出；越界属于限制。 | 两 body 的 `CopyIn/CopyOut*` |
| R-001 左填充 | `dataLen%32 != 0 && dataLen%32 <=16` | 输入 UB 起始右移 16 个 T 元素，保持后续 `LoadAlign` 的有效片段布局。 | R-001 | 其余余数不加左 pad。 | `SmallBlock::CopyIn/SplitMaxUbRowCompute` |
| 有效 lane | 每轮 `UpdateMask(remaining)` | 防止 padding/tail lane 进入 reduce、除法与输出。 | 两 Route | N/A。 | `ComputeAlign/ComputeXTmpMax/ComputeOutVF` |
| 平台 guard | public=950，OpDef/config/CMake=`ascend950`，body `__NPU_ARCH__==3510` 控制 overflow SPR | HiF8/FP8 与 MicroAPI 实现在 arch35/950 构建。 | 全部 | A2/A3 不发布。 | README/API、OpDef、CMake、kernel entry/body |

### POD/ABI

`GroupedDynamicBlockQuantTilingData` 的字段顺序和源码类型如下。结构由外部 `tiling/tiling_api.h` 宏生成；固定仓未包含宏展开，因此 `sizeof/alignof/offsetof` 不可见，不能以本地 C++ 自然对齐推测。所有字段由 `SetTilingData` 同名 setter 生产。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `GroupedDynamicBlockQuantTilingData` | `tilingKey` | int64/64 | 固定仓不可见 | key | `CalcTilingKey` | body 不读；router 用框架 `TILING_KEY_IS` | 4 位十进制编码 |
| 同上 | `totalCoreNum` | int64/64 | 固定仓不可见 | AIV core | `GetCoreNumAiv` | 两 body 解析但不参与循环 | 必须 >0 |
| 同上 | `ubSize` | int64/64 | 固定仓不可见 | byte | `GetCoreMemSize(UB)` | 解析但不读 | 必须 >0 |
| 同上 | `vfLen` | int64/64 | 固定仓不可见 | byte | `GetVRegSize(context)` | 解析但不读；已参与 Host R-002 公式 | 950 等于 `VECTOR_REG_WIDTH` 的具体值由工具链给出 |
| 同上 | `usedCoreNum` | int64/64 | 固定仓不可见 | core | `min(work,totalCoreNum)` | `Init/Process` core guard | 同时为 `blockDim` |
| 同上 | `headCoreNum` | int64/64 | 固定仓不可见 | core | `used-tail` | `Process` 选择 head/tail work | `[0,used]` |
| 同上 | `tailCoreNum` | int64/64 | 固定仓不可见 | core | `headWork*used-work` | R-002 解析、R-001 未解析；算法不用 | 诊断冗余字段 |
| 同上 | `minScale` | float/32 | 固定仓不可见 | FP32 scale | ACLNN double→OpDef float→`GetAttr<float>` | `ComputeAlign/ComputeScaleVF` | `>=0`，发生窄化 |
| 同上 | `roundMode` | int64/64 | 固定仓不可见 | enum | `GetRoundMode` | 解析但运行时不读；已编码到模板 `RMode` | 1/4/7 |
| 同上 | `dstType` | int64/64 | 固定仓不可见 | GE dtype enum | attr | 解析但运行时不读；已编码到模板 `U` | 34/35/36 |
| 同上 | `rowBlockSize` | int64/64 | 固定仓不可见 | row | attr | 分块、scale stride/reduce | 1/128/256/512 |
| 同上 | `colBlockSize` | int64/64 | 固定仓不可见 | element | attr | queue 容量 | 64/128/192/256 |
| 同上 | `batchNum` | int64/64 | 固定仓不可见 | batch | `CalcAxisSize` | 解析但 body 用地址反推的 `bIdx` | >=1（公开非空域） |
| 同上 | `rowNum` | int64/64 | 固定仓不可见 | row | x shape | GM batch stride/group assert | M |
| 同上 | `colNum` | int64/64 | 固定仓不可见 | element | x shape | GM row stride/DMA stride | N |
| 同上 | `scaleRowNum` | int64/64 | 固定仓不可见 | row | `M/r+g` | scale batch stride | 整数除法加预留 |
| 同上 | `scaleColNum` | int64/64 | 固定仓不可见 | element | `ceil(N/c)` | scale row stride | >=1 |
| 同上 | `uo` | int64/64 | 固定仓不可见 | N tile | `ceil(N/c)` | `(b,n,g)` 解码 | >=1 |
| 同上 | `groupNum` | int64/64 | 固定仓不可见 | group | `groupList.shape[0]` | `(b,n,g)` 解码 | 公开 groupList 非空 |
| 同上 | `blockFactor` | int64/64 | 固定仓不可见 | element | `N<=c?0:c` | 非尾 N tile 基址/长度 | `0` 或 c |
| 同上 | `tailBlockFactor` | int64/64 | 固定仓不可见 | element | `N%c==0?c:N%c` | 尾 N tile 长度 | `[1,c]` |
| 同上 | `groupBlockNumHeadCore` | int64/64 | 固定仓不可见 | `(b,n,g)` task | 分核公式 | head core 循环数 | 非负 |
| 同上 | `groupBlockNumTailCore` | int64/64 | 固定仓不可见 | `(b,n,g)` task | 分核公式 | tail core 循环数 | 可为 0 |
| 同上 | `maxUbRow` | int64/64 | 固定仓不可见 | row | UB 公式 | UB tile/分段行数与 queue 容量 | 必须为正；Host 未单独校验 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 SmallBlock | 1111/1211/1121/1221/1134/1234/1137/1237 | `grouped_dynamic_block_quant_tiling.cpp::DoTiling/CalcTilingKey` | `GroupedDynamicBlockQuantTilingData` | `grouped_dynamic_block_quant.cpp::grouped_dynamic_block_quant` 前 8 分支 | `GroupedDynamicBlockQuantSmallBlock::{Init,Process,CopyIn,SplitMaxUbRowCompute,ComputeAlign,CopyOut}`；`common/inc/op_kernel/platform.h::GetVRegSize` | 算子 CMake `ascend950`; ascend950 binary JSON 六组 dtype pair；全局 config entry |
| R-002 LargeBlock | 2111/2211/2121/2221/2134/2234/2137/2237 | 同上，`blockIsSmallThanUB=false` | 同上 | 同一 entry 后 8 分支 | `GroupedDynamicBlockQuantLargeBlock::{Init,Process,CopyIn,ComputeXTmpMax,ComputeScaleVF,ComputeOutVF,CopyOutY,CopyOutScale}` | 同上 |

## Kernel 实现

### 公共执行流程

entry 保存 950 overflow control SPR，声明 `KERNEL_TYPE_AIV_ONLY`，解析 POD，按 key 构造唯一模板实例并依次调用 `Init/Process`，最后恢复 SPR。两类 body 都把 `blockIdx` 映射为连续的 `(batch,nTile,group)` task；每个 task 从 `groupList[g-1:g]` 得到行域，从 `nTile` 得到有效 `dataLen`。GM→双缓冲 `inQueue`，MicroAPI 在寄存器中做绝对值最大值、FP32 scale 与 FP32 除法，饱和 cast 到 8 bit，经 `outQueue/scaleQueue` 返回 GM。无 workspace、atomic、跨核归约或多 Kernel 边；每个 `(b,n,g,rowBlock)` 只有一个 core owner。

### Route R-001：SmallBlock 单遍融合

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnGroupedDynamicBlockQuant.md` | 两段式函数原型 | 950、合法 dtype/attr/shape | 进入 GroupedDynamicBlockQuant OpDef |
| Host/Tiling | `op_host/grouped_dynamic_block_quant_tiling.cpp` | `DoTiling` | `rowBlockSize <= maxUbAvailableRows` | key 千位=1 |
| Router | `op_kernel/grouped_dynamic_block_quant.cpp` | `grouped_dynamic_block_quant` | 8 个 `TILING_KEY_IS(1xxx)` 顺序分支 | `SmallBlock<T,U,RMode>` |

#### Device 数据流

##### 函数级调用链

```text
grouped_dynamic_block_quant(x,groupList,y,scale,unusedWorkspace,tiling)
  -> SmallBlock<T,U,RMode>::Init
     -> ParseTilingData + InitBuffer(in/out/scale queues)
  -> SmallBlock<T,U,RMode>::Process
     -> CopyIn -> SplitMaxUbRowCompute -> ComputeAlign
     -> CopyOut(y,scale)
```

##### 源码等价伪码

```text
taskBegin = head/tail prefix(blockIdx)
for task in thisCoreTasks:
  (b,n,g) = decode(task, uo, groupNum)
  dataLen = (n==uo-1 ? tailBlockFactor : blockFactor)
  [gs,ge) = [g?groupList[g-1]:0, groupList[g]]
  for rowTileStart in range(gs, ge, maxUbRow):
    rows = min(maxUbRow, ge-rowTileStart)
    DataCopyPad x[b,rowTileStart:rowTileStart+rows,nTile] -> inQueue
    for each rowBlock q in this UB tile:
      valid = min(r, remainingRows) * paddedDataLen minus left-pad correction
      amax16 = ReduceMax(bitcast_u16(x) & 0x7fff, UpdateMask(valid))
      scale32 = min(cast_fp32(amax16)/Fmax, 1/minScale) when finite
      store scale32 at scaleQueue[q*8]
      y = cast_U(cast_fp32(x)/scale32, SAT + compile-time round mode)
    DataCopyPad compact y -> GM at x offset
    2D DataCopyPad first float of each 32-byte scale slot -> scale GM column n
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `b*M*N+n*blockFactor+groupStart*N+i*maxUbRow*N` | `blockCount × dataLen` effective | T/U, strided rows | caller/ComputeAlign | CopyIn/caller | caller；stream lifetime | 2D DMA row stride `N-dataLen` |
| GM scale | `b*scaleRows*scaleCols+n+(groupStart/r+g+i*maxUbRow/r)*scaleCols` | `ceil(blockCount/r)` floats | FP32 strided rows | CopyOut | caller | caller | destination stride `(scaleCols-1)*4` bytes |
| `inQueue_` | queue-managed | depth 2 × `c*maxUbRow*2` bytes | T, padded compact tile | CopyIn | ComputeAlign | kernel/core | left pad 16 T for selected N tails |
| `outQueue_` | queue-managed | depth 2 × `c*maxUbRow*1` bytes | U | ComputeAlign | CopyOut | kernel/core | only effective `dataLen` copied |
| `scaleQueue_` | slot `q*8` FP32 | depth 2 × `ceil(maxUbRow/r)*8*4` bytes | FP32, 32-byte slot | ComputeAlign | CopyOut | kernel/core | only first float per slot copied |
| registers | compiler assigned | 固定仓不可见 | T/u16/fp32/U + masks | ComputeAlign | same scope | one `__VEC_SCOPE__` | no cross-iteration state |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue VECIN · depth=2 · 4*c*maxUbRow B"]:2
  outQ["outQueue VECOUT · depth=2 · 2*c*maxUbRow B"]:1
  scaleQ["scaleQueue VECOUT · depth=2 · 64*ceil(maxUbRow/r) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*c*maxUbRow*2 + 2*c*maxUbRow*1
         + 2*ceil(maxUbRow/r)*8*4
         = 6*c*maxUbRow + 64*ceil(maxUbRow/r) bytes
UB_total <= Host-reported UB by the small-route maxUbRow construction
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueue` slot | `CopyIn` | `SplitMaxUbRowCompute/ComputeAlign` | 下一 row tile | `EnQue→DeQue→FreeTensor` | compute 完成后才 free |
| `outQueue` slot | `ComputeAlign` | `CopyOut` | 下一 row tile | `EnQue→DeQue→FreeTensor` | GM DMA consumption precedes free |
| `scaleQueue` slot | `ComputeAlign` | `CopyOut` | 下一 row tile | 同上 | 每 scale 32-byte slot 只提交首 FP32 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + queue | CopyIn | DeQue | input slot | N tail 用 right pad；特定余数另加 16 T left pad |
| reduce | `And 0x7fff`、`Max`、`ReduceMax` | register loads | scale math | active lanes | FP16/BF16 bit-level abs；`UpdateMask` 排除 padding |
| scale | cast FP32、`Div/CompareScalar/Min` | amax | quantize/store | one scale per row block | FP32；只对有限候选执行上限 Min |
| quantize | unpack B16→FP32→Div→saturating cast→pack B8 | x + scale | outQueue | valid y lanes | RMode 1/4/7 编译期选择；tail mask zeroing |
| UB→GM | queue + `DataCopyPad` | ComputeAlign | CopyOut | y/scale slots | 2D DMA 只写有效 N 与有效 row-block scale |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/grouped_dynamic_block_quant_tiling.cpp` | `DoTiling/CalcTilingKey` | small 判定、maxUbRow、16 个 key 编码与 blockDim |
| Kernel entry/router | `op_kernel/grouped_dynamic_block_quant.cpp` | `grouped_dynamic_block_quant` 1xxx branches | key→8 个 SmallBlock 实例；AIV-only |
| Device body/helper | `op_kernel/arch35/grouped_dynamic_block_quant_small_block.h` | `Process/ComputeAlign` | row tile 单遍融合 reduce、scale、quantize 与 tail padding |

### Route R-002：LargeBlock 两遍分段

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnGroupedDynamicBlockQuant.md` | 两段式函数原型 | 950、合法 dtype/attr/shape | 进入同一 OpDef |
| Host/Tiling | `op_host/grouped_dynamic_block_quant_tiling.cpp` | `DoTiling` | `rowBlockSize > maxUbAvailableRows`；`maxUbRow=r*((UB-vfLen)/2)/E` | key 千位=2 |
| Router | `op_kernel/grouped_dynamic_block_quant.cpp` | `grouped_dynamic_block_quant` | 8 个 `TILING_KEY_IS(2xxx)` 顺序分支 | `LargeBlock<T,U,RMode>` |

#### Device 数据流

##### 函数级调用链

```text
grouped_dynamic_block_quant
  -> LargeBlock<T,U,RMode>::Init
     -> InitBuffer(in/out/scale queues, xLocalMaxBuffer)
  -> LargeBlock<T,U,RMode>::Process
     -> [reverse segments] CopyIn -> ComputeXTmpMax
     -> ReduceMax(xLocalMaxBuffer) -> ComputeScaleVF
     -> ComputeOutVF(retained segment 0) -> CopyOutY
     -> [segments last..1] CopyIn -> ComputeOutVF -> CopyOutY
     -> CopyOutScale
```

##### 源码等价伪码

```text
for owned (b,n,g):
  for q in ceil((groupEnd-groupStart)/r):
    blockRows = tail-aware rows in this row block
    ubLoops = ceil(blockRows/maxUbRow)
    xLocalMax = 0
    for seg = ubLoops-1 downto 0:
      CopyIn compact segment
      xLocalMax = lane_max(xLocalMax, abs(segment), UpdateMask(valid))
      if seg != 0: FreeTensor(segment)
    amax = ReduceMax(xLocalMax); scale = min(amax/Fmax,1/minScale)
    quantize retained seg0 with broadcast scale; copy y seg0
    for seg = ubLoops-1 downto 1:
      reread segment; quantize with same scale; copy y segment
    copy one FP32 scale to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `b*M*N+n*blockFactor+(groupStart+q*r+seg*maxUbRow)*N` | `rowNum × dataLen` | T/U compact DMA tile | caller/ComputeOutVF | CopyIn/caller | caller | source/destination row stride `N-dataLen` |
| GM scale | `b*scaleRows*scaleCols+n+(groupStart/r+g+q)*scaleCols` | one FP32 | FP32 | CopyOutScale | caller | caller | 4-byte effective copy |
| `inQueue_` | queue-managed | depth 2 × `c*maxUbRow*2` | T compact | CopyIn | max/quantize helpers | core | seg0 retained across scale compute |
| `outQueue_` | queue-managed | depth 2 × `c*maxUbRow` | U | ComputeOutVF | CopyOutY | core | per segment |
| `scaleQueue_` | queue-managed | depth 2 × 32 | FP32 | ComputeScaleVF | all segment quantize + CopyOutScale | core | held until every segment done |
| `xLocalMaxBuffer_` | base 0 | `VECTOR_REG_WIDTH` bytes | T/u16 reduction state | ComputeXTmpMax | ReduceMax/ComputeScaleVF | core, one row block | repeatedly overwritten after consumption |
| registers | compiler assigned | 固定仓不可见 | T/u16/fp32/U + masks | VF helpers | same helper | scope local | scale broadcast in `ComputeOutVF` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue VECIN · depth=2 · 4*c*maxUbRow B"]:2
  outQ["outQueue VECOUT · depth=2 · 2*c*maxUbRow B"]:1
  scaleQ["scaleQueue VECOUT · depth=2 · 64 B"]:1
  maxBuf["xLocalMax TBuf · vfLen B"]:4
```

##### 字节账与复用顺序

```text
UB_total = 4*c*maxUbRow + 2*c*maxUbRow + 64 + vfLen
         = 6*c*maxUbRow + 64 + vfLen bytes
required safety condition: 6*c*maxUbRow + 64 + vfLen <= Host-reported UB
dynamic_ub = N/A
workspace_total = 0
```

Host 的 `maxUbRow=r*floor((UB-vfLen)/2)/E` 把 scale 的 32-byte 单缓冲项按 `maxUbRow/r` 比例纳入 `E`，而 Device 在 LargeBlock 中固定申请两个 32-byte scale slot。固定源码没有在选出 `maxUbRow` 后复核上述精确不等式；因此这里只给出真实字节账和必要条件，不声称 Host 公式对所有整数截断组合构成严格证明。

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `xLocalMaxBuffer` | 每段 `ComputeXTmpMax` | final `ReduceMax/ComputeScaleVF` | 下一 row block 的 `Duplicate(0)` | 同一 core 顺序执行 | 全部分段归约结束后才清零 |
| retained `inQueue` seg0 | 归约阶段最后一次 `CopyIn` | 首次 `ComputeOutVF` | `FreeTensor` 后 | 源码对 `seg==0` 不 free | 归约循环必须倒序 |
| `scaleQueue` slot | `ComputeScaleVF` | 每段 `ComputeOutVF` + `CopyOutScale` | 全部 y segment 后 free | enqueue 延迟至 y 完成后 | scaleLocal 地址在量化阶段保持有效 |
| other input/output slots | CopyIn/ComputeOutVF | VF/CopyOutY | 下一 segment | queue EnQue/DeQue/FreeTensor | 单 core 顺序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分段归约 | queue + `xLocalMaxBuffer` | reverse CopyIn/ComputeXTmpMax | final ReduceMax | cross-segment max | `UpdateMask<T>(xTotalNum)`；符号位 mask |
| scale 广播 | `scaleQueue` local tensor | ComputeScaleVF | every ComputeOutVF | one row-block scale | FP32 `Div/Min`；`DIST_BRC_B32` |
| 分段量化 | queue order | retained/reread segment | CopyOutY | segment y | FP32 divide then saturating compile-time cast；last segment row/N tails masked |
| SPR | save/set/restore overflow control | entry/Init | entry epilogue | 950 floating overflow mode | `__NPU_ARCH__==3510`；每个 body Init 设 0，entry 最终恢复原值 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/grouped_dynamic_block_quant_tiling.cpp` | `DoTiling` | large 判定、为归约 TBuf 扣除 `vfLen`、maxUbRow 与 core/task 参数 |
| Kernel entry/router | `op_kernel/grouped_dynamic_block_quant.cpp` | `grouped_dynamic_block_quant` 2xxx branches | key→8 个 LargeBlock 实例；AIV-only |
| Device body/helper | `op_kernel/arch35/grouped_dynamic_block_quant_large_block.h` | `Process/ComputeXTmpMax/ComputeScaleVF/ComputeOutVF` | 两遍分段、seg0 保留、FP32 scale 广播、有效 lane mask |

## 算子特定优化

### Trick T-001：SmallBlock 在单 UB tile 内融合多行块的 scale 与量化

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：`rowBlockSize <= maxUbAvailableRows`。
- Golden 来源：[`grouped_dynamic_block_quant_small_block.h:349-395`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_kernel/arch35/grouped_dynamic_block_quant_small_block.h#L349-L395)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint16_t stepBlock = 0; stepBlock < static_cast<uint16_t>(tailBlockIdx); stepBlock++) {
    uint32_t pnum16 = normalBlockDataNum;
    AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0x0000);
    for (uint16_t step128 = 0; step128 < static_cast<uint16_t>(normalStep128); step128++) {
        dataLenMask16 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum16);
        AscendC::MicroAPI::LoadAlign(xRegTensor, xAddr + stepBlock * blockDataNumOffset + step128 * vfLen);
        AscendC::MicroAPI::And(
            xAbsRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskRegTensor,
            dataLenMask16);
        AscendC::MicroAPI::Max<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
            expMaxRegTensor, expMaxRegTensor, xAbsRegTensor, dataLenMask16);
    }
    // ... reduce, scale, divide and cast this row block before advancing ...
}
```

#### 机制与设计原因

一个 `maxUbRow × c` 输入 tile 只从 GM 读一次，`stepBlock` 在同一 UB 输入上依次完成每个 `r × dataLen` 块的 reduce、FP32 scale 和量化。scale 直接写入 32-byte 间隔的 output queue 槽，y 直接写 output queue，不物化额外 FP32 tile。`vfLen`/`vfNum` 来自 950 vector-register 字节数除以 2/4，`UpdateMask` 来自当前块的有效元素数。

#### 迁移边界

- 适用：多个独立 reduction block 能共同放入一个 UB tile，且输出可立即由各自 scale 量化。
- 前提：950 MicroAPI、2-byte 输入/1-byte 输出、32-byte scale slot、合法 block sizes 与同样 tail padding。
- 失效：一个 row block 不能驻留时必须使用 R-002 的跨段归约。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不可在没有 tail mask/left-pad 地址修正时照搬对齐 load。

### Trick T-002：LargeBlock 倒序归约保留 seg0，避免量化阶段重读一个分段

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：一个 row block 被切成 `ubLoopNum>1` 个 `maxUbRow` 分段。
- Golden 来源：[`grouped_dynamic_block_quant_large_block.h:251-295`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_block_quant/op_kernel/arch35/grouped_dynamic_block_quant_large_block.h#L251-L295)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int64_t ubLoopNumIdx = ubLoopNum - 1; ubLoopNumIdx >= 0; ubLoopNumIdx--) {
    xGmOffset = xBlockGmOffset + ubLoopNumIdx * maxUbRow_ * colNum_;
    rowNum = (ubLoopNumIdx == ubLoopNum - 1) ? blockRowNum - ubLoopNumIdx * maxUbRow_ : maxUbRow_;
    CopyIn(xGmOffset, rowNum, dataLen);
    inLocal = inQueue_.DeQue<T>();
    xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    ComputeXTmpMax(rowNum, dataLen, xLocalAddr, xLocalMaxTmpAddr);
    if (ubLoopNumIdx != 0) {
        inQueue_.FreeTensor(inLocal);
    }
}
```

#### 机制与设计原因

scale 依赖所有分段的最大值，因此多数分段在归约后必须二次从 GM 读取才能量化。源码倒序扫描并故意不释放最后处理的 seg0；scale 形成后先用该仍驻留 UB 的 tensor 生成 y，然后只重读 seg `ubLoopNum-1..1`。这由 queue slot 的延长 lifetime 减少一次 GM→UB 分段搬运，代价是 scale 形成前占用一个 input slot。

#### 迁移边界

- 适用：先全局/块级归约、后按归约结果逐分段变换的两遍算法。
- 前提：至少一个 input queue slot可跨 scale 计算保留；处理顺序和 offset 必须让被保留段唯一且最终仍需输出。
- 失效：单段时自然退化为无重读；若中间操作需要复用该 slot，则必须显式保存或重读全部分段。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若归约循环对 seg0 也立即 `FreeTensor`，后续首段量化使用的 `xLocalAddr` 生命周期失效；固定源码用 `if (ubLoopNumIdx != 0)` 明确避免该错误。

## 模板与实现对比

| 对比维度 | R-001 SmallBlock | R-002 LargeBlock | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `r<=maxUbAvailableRows`，key 1xxx | 其补集，key 2xxx | 行量化块是否可由 UB tile 覆盖 |
| 模板与 dtype | `SmallBlock<T,U,RMode>`，8 实例 | `LargeBlock<T,U,RMode>`，8 实例 | T=FP16/BF16；U=三种 8-bit；RMode=1/4/7 |
| 循环与分核 | core→(b,n,g)→row tile→row block | core→(b,n,g)→row block→UB segment 两遍 | 是否需要跨段 max |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 个 depth-2 queue；无 TBuf/dynamic UB | 同样 queues + `vfLen` TBuf | Large 保存跨段最大值 |
| 内存总占用与复用 | `6cR+64ceil(R/r)`；tile 内融合 | `6cR+64+vfLen`；seg0 延长 lifetime | Host 分别构造 R |
| 搬运/Vector/SIMT/NDDMA/通信 API | padded 2D copy；MicroAPI reduce/div/cast | compact 2D copy；两遍 MicroAPI；部分 GM 重读 | 均为 AIV_VECTOR，无其他执行族 |
| 同步、精度与 tail | queue；block/tile masks；可有 left pad | queue + TBuf 顺序；segment masks | FP32 scale、饱和目标 cast 相同 |
| fallback/失败边界 | 资源条件假转 R-002 | 无第三 Route | unknown key 静默无输出；tiling gate 提前失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持 | 公开 README/API 明确不支持 | Ascend 950PR/950DT 支持 |
| Host/Launcher | 无公开支持链路 | 无公开支持链路 | `SUPPORT_COMPUTE_UNIT=ascend950`，AIV core/UB/vreg 参数化 |
| Kernel/资源 | 不发布 | 不发布 | arch35 body，`KERNEL_TYPE_AIV_ONLY`，3510 SPR guard，Ascend 950 binary config |

## 限制与失败边界

- ACLNN 公开域仅 `groupListType=0`；tiling 源码接受 1 与文档冲突，且 Device 没有第二种语义实现，不能把 1 视为公开支持。
- 文档约束 `groupList` 值非负、非递减、末值等于 M；R-001 Device 有范围/单调 assert，R-002 没有相同 assert。自动生成 ACLNN bridge 是否读取 Device tensor 值做校验在固定仓不可见，调用者必须满足文档约束。
- Host 将 ACLNN `double minScale` 收窄到 tiling FP32；具体 bridge 转换表达式不可见。`minScale=0` 依赖设备 FP32 除零/比较语义。
- `maxUbRow`、`usedCoreNum` 依赖非空 M/N/groupList；公开文档排除空 `x/groupList`，但 Host 没有独立的 `maxUbRow>0` 检查。
- DMA 的 `blockCount` 收窄为 `uint16_t`，block byte length/stride 收窄为 `uint32_t`；Host 未给出相应 shape 上界检查，超位宽行为不应推测。
- POD 宏展开、ACLNN generated bridge、executor 内部 launch 包装与 AutoContiguous 临时区布局不在固定仓；这些仅作为框架边界，不据此推测 offset 或额外 workspace。
- router 无显式 unknown-key 分支；未知 key 不执行 body。固定仓未显示错误上报。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/grouped_dynamic_block_quant/docs/aclnnGroupedDynamicBlockQuant.md` | 两段式 API 原型/产品支持/参数表 | 公开入口、950-only、接口约束 |
| 入口实现 | `quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_def.cpp` | `GroupedDynamicBlockQuant::GroupedDynamicBlockQuant` | dtype/format/AutoContiguous、ascend950 OpDef |
| Host infer | `quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_infershape.cpp` | `InferShapeForGroupedDynamicBlockQuant/InferDataTypeForGroupedDynamicBlockQuant` | 输出 shape/dtype |
| Host/Tiling/POD | `quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_tiling.h` | `GroupedDynamicBlockQuantTilingData` | 字段顺序、类型、round enum |
| Host/Tiling/Launcher | `quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_tiling.cpp` | `GetAttr/CheckDtype/CheckShape/DoTiling/Tiling4GroupedDynamicBlockQuant` | 校验、Route、core/UB/key/workspace/POD |
| Kernel entry/body | `quant/grouped_dynamic_block_quant/op_kernel/grouped_dynamic_block_quant.cpp` | `grouped_dynamic_block_quant` | AIV-only entry 与 16 key router |
| Kernel entry/body | `quant/grouped_dynamic_block_quant/op_kernel/arch35/grouped_dynamic_block_quant_small_block.h` | `GroupedDynamicBlockQuantSmallBlock` | R-001 全实现 |
| Kernel entry/body | `quant/grouped_dynamic_block_quant/op_kernel/arch35/grouped_dynamic_block_quant_large_block.h` | `GroupedDynamicBlockQuantLargeBlock` | R-002 全实现 |
| Kernel helper | `common/inc/op_kernel/platform.h` | `platform::GetVRegSize` | 950 vector-register 宽度来源 |
| 构建接线 | `quant/grouped_dynamic_block_quant/op_host/config/ascend950/grouped_dynamic_block_quant_binary.json` | `op_type/op_list` | 六组 input/output dtype binary 组合 |
| 构建接线 | `quant/grouped_dynamic_block_quant/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/add_modules_sources` | ascend950 + ACLNN 接线 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `GroupedDynamicBlockQuant` entry | kernel 只编译 ascend950 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 同一 POD 服务 Small/Large 与全部 dtype 实例 | 字段顺序/位宽、字节/元素单位；宏布局不可臆测 | `GroupedDynamicBlockQuantTilingData/SetTilingData/ParseTilingData` |
| 调用顺序 | 单 Kernel；Init 后 Process | Large 必须先完成所有段 max，形成 scale 后再量化 | entry；两 body `Process` |
| workspace/内存 | workspace=0；双缓冲公式可直接复用 | depth=2 必须计入；Large 必须额外保留 `vfLen` TBuf 与 seg0 lifetime | `Tiling4...`; 两 body `Init` |
| 同步 | queue 生命周期承担 GM/UB producer-consumer 顺序 | `EnQue/DeQue/FreeTensor` 次序和 Large seg0 特例不可改变 | `CopyIn/SplitMaxUbRowCompute/Process/CopyOut*` |
| 精度与尾块 | abs-bit reduce→FP32 scale/div→饱和 8-bit cast | Fmax、RMode、finite compare、left/right pad、mask 与 scale 32-byte slot | `ComputeAlign/ComputeScaleVF/ComputeOutVF` |
| 失败边界 | Host gate 失败不 launch；资源 gate在两 Route 间切换 | A2/A3 不支持；groupListType 公开仅 0；unknown key 无错误 body | docs、`GetAttr/DoTiling/router` |
