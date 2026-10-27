---
type: Operator
title: "DynamicBlockMxQuant 950 ACLNN 算子知识"
description: "说明 aclnnDynamicBlockMxQuant 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 AIV 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/docs/aclnnDynamicBlockMxQuant.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_kernel/dynamic_block_mx_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_host/config/ascend950/dynamic_block_mx_quant_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_kernel/platform.h
    role: implementation
created_at: "2026-08-25T18:24:04Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# DynamicBlockMxQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | CANN 9.1.0 两段式公开接口 `aclnnDynamicBlockMxQuantGetWorkspaceSize` → `aclnnDynamicBlockMxQuant`；仓内没有后缀版本，故它是 950 实际支持的最新公开版本。 |
| 计算语义 | 对最后两维按 `32×32` MX 块求共享 E8M0 scale，输出量化值 `y`，并分别沿最后一维、倒数第二维广播为 `scale1`、交织为 `scale2`；`scale_alg=0` 为 OCP MX，`scale_alg=2` 为仅限 E2M1 的 Dynamic Dtype Range。 |
| 输入输出 | `x`: FP16/BF16、ND、rank 2/3；`y`: FP4 E2M1/E1M2 或 FP8 E4M3FN/E5M2；`scale1/scale2`: E8M0。FP4 要求 `x[-1]` 为偶数。 |
| 执行类型 | 单 Kernel、`AIV_VECTOR`；entry 声明 `KERNEL_TYPE_MIX_AIV_1_0`，可达 body 只含 AIV/Vector 与 GM↔UB 搬运，无 Cube/AIC、MIX_CV、SIMT、NDDMA 或 AICPU 节点。 |
| Kernel 路线 | R-001～R-016 为 `scale_alg=0` 的 OCP 路线，R-017～R-022 为 `scale_alg=2` 的 E2M1 DDR 路线；按 `DTYPE_X × DTYPE_Y × roundMode × scaleAlg` 的实际模板实例拆分。 |
| 核心资源 | 每核 4 个 depth=2 `TQue`、2 个 `TBuf`、RegTensor；无算子自有跨 Kernel workspace。Tiling 仍申请 `GetLibApiWorkSpaceSize()` 供 AscendC 库/框架使用。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档 → `ACLNNTYPE aclnn` 生成边界/OpDef → infer/tiling/POD → 模板 key → 单 AIV entry → `DynamicBlockMxQuantBase` 全部 body/helper → 950 binary/config/CMake 已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x` 的末两维为 `M,N`，不足 `32×32` 的边界在 UB 中补零。对每个 `32×32` 子块，`scale_alg=0` 计算

```text
shared_exp = floor(log2(max(abs(x_block)))) - emax(dst_type)
scale      = 2^shared_exp
y_i        = cast_dst(x_i / scale, round_mode)
```

其中 `emax(E2M1,E1M2,E4M3FN,E5M2)=(2,0,8,15)`。`scale_alg=2` 仅允许 E2M1：从块内绝对值最大值的 BF16 表示提取指数/尾数，按 `dst_type_max∈{0,6,7}` 选择尾数阈值并向上调整 E8M0 scale，再乘其倒数并按指定 round mode 转 E2M1。`scale1` 为每行、每 32 列一份 scale，按 64 列补成偶数 pair；`scale2` 为每列、每 32 行一份 scale，两个 32 行半块交织输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[M,N]` 或 `[B,M,N]`, ND | FP16/BF16 | 非空；FP4 输出时 `N` 必须为偶数；OpDef 使用 `AutoContiguous()`，非连续输入由框架边界处理。 |
| `roundModeOptional` / `round_mode` | 标量字符串 | STRING | 默认/空指针为 `rint`；FP4 支持 `rint/floor/round`，FP8 仅支持 `rint`。 |
| `dstType` / `dst_type` | 标量 | INT64 | `35/36/40/41` 分别对应 E5M2/E4M3FN/E2M1/E1M2，必须与 `y` dtype 一致。 |
| `scaleAlg` / `scale_alg` | 标量 | INT64 | 仅 `0` 或 `2`；`2` 仅允许 E2M1。 |
| `dstTypeMax` / `dst_type_max` | 标量 | ACLNN 为 double，OpDef/Tiling 为 float | 默认 `0.0`；仅 `scale_alg=2 && y=E2M1` 可取 `0/6/7`，其它组合必须为 0。ACLNN→OpDef 的 double→float 转换属于生成 bridge 边界。 |
| `yOut` | 与 `x` 相同，ND | E2M1/E1M2/E4M3FN/E5M2 | FP4 在 GM 中两个 4-bit 元素打包为一字节；FP8 一元素一字节。 |
| `scale1Out` | `[B?,M,ceil(N/64),2]`, ND | E8M0 | 最后维为 pair；不足偶数个 32 列 group 的位置补 0。 |
| `scale2Out` | `[B?,ceil(M/64),N,2]`, ND | E8M0 | 两个相邻 32 行 group 交织；不足第二个 group 时补 0。 |

### 入口与版本

公开文档给出唯一两段式版本：第一段校验并返回 workspace/executor，第二段在传入 stream 上执行。固定仓未落盘手写 `op_api/*.cpp`；`CMakeLists.txt::add_modules_sources(... ACLNNTYPE aclnn)` 和 `DynamicBlockMxQuant` OpDef 是 ACLNN 生成 bridge 的明确框架边界，不把不可见生成代码推测为手写实现。README、ACLNN 文档、OpDef、`SUPPORT_COMPUTE_UNIT=ascend950`、950 binary config 与全局 kernel config 一致支持 Ascend 950；公开文档明确 A2/A3 不支持。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段按文档检查空指针、dtype/format/shape、属性值与当前平台；非连续 `x` 经 OpDef `AutoContiguous` 框架边界转换。
2. `InferShapeForDynamicBlockMxQuant` 令 `y.shape=x.shape`，`scale1[-2]=ceil(ceil(N/32)/2)=ceil(N/64)`、`scale2[-3]=ceil(ceil(M/32)/2)=ceil(M/64)`，二者追加末维 2；未知 rank 传播为 unknown rank。
3. `DoTiling` 严格按 `GetPlatformInfo → CheckDtype → GetAttr → CheckShape → SetTilingParams → SetTilingKey → SetTilingData` 执行，任一步失败返回 `GRAPH_FAILED`，不启动 Kernel。
4. `CheckShape` 再次拒绝空 tensor、非 2/3 维、FP4 下奇数 `N`、以及调用方给出的三个输出 shape 不匹配；`GetAttr` 检查 round/dst/algorithm 的组合约束。

### 参数、Workspace 与生命周期

`GetPlatformInfo` 产生 `totalCoreNum=GetCoreNumAiv()`、`ubSize=GetCoreMemSize(UB)` 与 `workspaceSize=GetLibApiWorkSpaceSize()`。Tiling 写 `workspaces[0]=workspaceSize`；Kernel entry 虽接收 `workspace`，body 从未解引用它，故算子自有中间 slice 为 0，非零部分归 AscendC 库/ACLNN executor 框架所有。若 `AutoContiguous` 需要临时区，其总量与 owner 由未落盘的生成 executor 管理，固定仓不能给出额外 offset；调用者必须使用第一段返回值，不可仅按 Kernel 的 0 字节自行替代。

POD 由 Tiling 在第一段 executor 构建/运行准备期写入，Device 只读到单 Kernel 完成；4 个 GM 输出由调用者提供并在同一 stream launch 完成前保持有效。

### 启动图与失败传播

```text
aclnnDynamicBlockMxQuantGetWorkspaceSize
  -> generated ACLNN executor boundary
  -> DynamicBlockMxQuant OpDef / InferShape / TilingForDynamicBlockMxQuant
  -> key = GET_TPL_TILING_KEY(roundMode, scaleAlg)
aclnnDynamicBlockMxQuant(workspace, workspaceSize, executor, stream)
  -> [K1, AIV_VECTOR] dynamic_block_mx_quant<roundMode,scaleAlg>
       grid/blockDim = usedCoreNum
       inputs: x; outputs: y, scale1, scale2; POD: DynamicBlockMxQuantTilingData
       workspace argument: passed but not consumed
```

只有一个 Device 节点，因此没有跨 Kernel workspace 边。executor 在同一指定 stream 上提交 K1；Tiling/launch 失败则 ACLNN 返回错误且不应产生后续节点。固定仓未公开生成 executor 的内部 cleanup 细节。

## Route 与参数设计

### 分核与资源计算

记 `BH=64`、`BW=256`、逻辑 MX block 为 `32×32`：

```text
batch = rank(x)==2 ? 1 : x[0]
M,N   = x[-2],x[-1]
rowBlocks = batch * ceil(M/64)
colBlocks = ceil(N/256)
candidateCores = min(totalAiv, rowBlocks*colBlocks), 至少 1
```

`AutoTiling` 枚举候选核数的因子 `(m,n)`，过滤 `m<=rowBlocks,n<=colBlocks`，按 `(delta,n)` 升序选最小项作为 `rowTileNum,colTileNum`。每维采用 `ceil` 的 normal-core tile 与 `floor` 的 tail-core tile，并据余数算 normal/tail core 数。

每个 `64×256` 基块的实际 UB 系数为：

```text
perBlock = 2 * [64*256*(2 input bytes + 1 output byte) + 64*32 + (64/32)*256]
         + (64/32)*align_up(256/32,32)*2 reciprocal bytes
         = 103552 bytes
maxBlocks = floor((ubSize - 256 gather - 1024 reserve) / 103552)
c = min(maxBlocks, normalCoreColTileNum)
r = min(floor(maxBlocks/c), normalCoreRowTileNum)
rowUbFactor=64*r; colUbFactor=256*c
```

随后每核以 `rowUbLoop=ceil(coreRowTiles/r)`、`colUbLoop=ceil(coreColTiles/c)` 遍历，最后循环分别缩成剩余 block 数。Host 公式要求 `maxBlocks>=1`；源码没有显式 0 防护，950 可用 UB 必须满足 `ubSize>=103552+256+1024`。

### Route 选择

表中 `ri∈{rint,round,floor}`；每一行是独立模板/精度 Route，不能仅因共享源模板合并。

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001～R-003 FP16→E2M1 OCP | `(ri,0)` | `x=FP16,y=E2M1,scaleAlg=0,round=ri` | 对应下一 dtype/alg 组合或校验失败 |
| 2 | R-004～R-006 BF16→E2M1 OCP | `(ri,0)` | `x=BF16,y=E2M1,scaleAlg=0,round=ri` | 同上 |
| 3 | R-007～R-009 FP16→E1M2 OCP | `(ri,0)` | `x=FP16,y=E1M2,scaleAlg=0,round=ri` | 同上 |
| 4 | R-010～R-012 BF16→E1M2 OCP | `(ri,0)` | `x=BF16,y=E1M2,scaleAlg=0,round=ri` | 同上 |
| 5 | R-013 FP16→E4M3FN OCP | `(rint,0)` | `x=FP16,y=E4M3FN,scaleAlg=0,round=rint` | 非 rint 失败 |
| 6 | R-014 BF16→E4M3FN OCP | `(rint,0)` | `x=BF16,y=E4M3FN,scaleAlg=0,round=rint` | 非 rint 失败 |
| 7 | R-015 FP16→E5M2 OCP | `(rint,0)` | `x=FP16,y=E5M2,scaleAlg=0,round=rint` | 非 rint 失败 |
| 8 | R-016 BF16→E5M2 OCP | `(rint,0)` | `x=BF16,y=E5M2,scaleAlg=0,round=rint` | 非 rint 失败 |
| 9 | R-017～R-019 FP16→E2M1 DDR | `(ri,2)` | `x=FP16,y=E2M1,scaleAlg=2,dstTypeMax∈{0,6,7},round=ri` | 非 E2M1/非法 max 失败 |
| 10 | R-020～R-022 BF16→E2M1 DDR | `(ri,2)` | `x=BF16,y=E2M1,scaleAlg=2,dstTypeMax∈{0,6,7},round=ri` | 非 E2M1/非法 max 失败 |

```text
validate dtype/shape
  -> validate round: FP8 only rint; FP4 rint|round|floor
  -> validate scaleAlg: 0, or 2 only for E2M1
  -> roundMode_ = rint:4 | round:0 | floor:1
  -> scaleAlg_ = 0 | 2
  -> GET_TPL_TILING_KEY(roundMode_, scaleAlg_)
  -> binary DTYPE_X/DTYPE_Y specialization
  -> DynamicBlockMxQuantBase<DTYPE_X,DTYPE_Y,mappedRound,scaleAlg>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001/R-002/R-003 | `(4,0)/(0,0)/(1,0)` | FP16→E2M1，FP4 shape guard，950 | K1 AIV | `Base<half,fp4x2_e2m1_t,RINT/ROUND/FLOOR,0>`；`ComputeOcp+ComputeY1ToFP4+ComputeFP4FromHalf(E2M1)` | 公共 4Q+2Buf；FP16 先升 FP32 | GM tail 补零/只回写有效区 |
| R-004/R-005/R-006 | 同左 | BF16→E2M1，FP4 shape guard，950 | K1 AIV | `Base<bfloat16_t,fp4x2_e2m1_t,...,0>` | BF16 原位乘 reciprocal 后转 FP4 | 同上 |
| R-007/R-008/R-009 | 同左 | FP16→E1M2，FP4 shape guard，950 | K1 AIV | `Base<half,fp4x2_e1m2_t,...,0>`；E1M2 `×4→Truncate→×1/4` | FP16→FP32 乘缩放 | 同上 |
| R-010/R-011/R-012 | 同左 | BF16→E1M2，FP4 shape guard，950 | K1 AIV | `Base<bfloat16_t,fp4x2_e1m2_t,...,0>` | BF16 路径 | 同上 |
| R-013 | `(4,0)` | FP16→E4M3FN，950 | K1 AIV | `Base<half,fp8_e4m3fn_t,CAST_RINT,0>` | FP32 乘 reciprocal，饱和转 FP8 | 同上 |
| R-014 | `(4,0)` | BF16→E4M3FN，950 | K1 AIV | `Base<bfloat16_t,fp8_e4m3fn_t,CAST_RINT,0>` | BF16 先乘，拆成 FP32 后转 FP8 | 同上 |
| R-015 | `(4,0)` | FP16→E5M2，950 | K1 AIV | `Base<half,fp8_e5m2_t,CAST_RINT,0>` | E5M2 `dtypeYMaxExp=0x0780` | 同上 |
| R-016 | `(4,0)` | BF16→E5M2，950 | K1 AIV | `Base<bfloat16_t,fp8_e5m2_t,CAST_RINT,0>` | BF16 路径 | 同上 |
| R-017/R-018/R-019 | `(4,2)/(0,2)/(1,2)` | FP16→E2M1，max guard，950 | K1 AIV | `Base<half,fp4x2_e2m1_t,RINT/ROUND/FLOOR,2>`；`ComputeDdr` | max 保留尾数；其后 FP16→FP32 | 同上 |
| R-020/R-021/R-022 | 同左 | BF16→E2M1，max guard，950 | K1 AIV | `Base<bfloat16_t,fp4x2_e2m1_t,...,2>` | BF16 max/量化路径 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950-only | README/API 支持 950，CMake `ascend950` | 只有 arch35 body/binary | 全部 | A2/A3 公开不支持 | `README.md::产品支持情况`; `CMakeLists.txt` |
| UB 至少一块 | `maxBlocks=(ub-1280)/103552 >= 1`（源码隐含） | 否则 `c=0` 后 Host 除零 | 全部 | 源码没有 fallback，Tiling 前提失败边界 | `SplitCore` |
| FP4 偶数 N | `N%2==0` | GM/UB 的 FP4 `y` 以两个元素/字节打包 | FP4 Route | `GRAPH_FAILED` | `CheckShape`; `CopyOut` |
| FP8 round | `y∈FP8 => round=rint` | `RoundModeMapper` 对 FP8 固定 `CAST_RINT` | R-013～016 | `GRAPH_FAILED` | `GetAttr`; `RoundModeMapper` |
| DDR 限域 | `scaleAlg=2 => y=E2M1 && max∈{0,6,7}` | 仅 E2M1 实例进入 `ComputeDdr`，max 决定 mantissa 常量 | R-017～022 | `GRAPH_FAILED` | `GetAttr`; `InitParams`; `Compute` |
| 尾列补零 | `blockColNum%BW!=0` | Vector body固定按 256 列 load/reduce，须隔离无效 lane | 全部 | 先 `Duplicate(0)` 再 `DataCopyPad` | `CopyIn` |
| 64 行拆半 | 每块 `row1=min(validRows,32)`，`row2=max(validRows-32,0)` | scale2 每 32 行一份并以 pair 交织 | 全部 | 第二半为空则补 256B 零 scale | `Compute` |

### POD/ABI

`DynamicBlockMxQuantTilingData` 使用 `#pragma pack(push,8)`；33 个 `int64_t` 后接一个 `float`，因此字段 offset 可由声明顺序确定，结构总大小为 272B、对齐 8B，末尾 4B padding。Host 在 `SetTilingData` 逐字段写入并 `memcpy_s` 序列化；下表“未读”表示当前 Kernel 模板不消费，不能擅自复用该 ABI 槽。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `DynamicBlockMxQuantTilingData` | `tilingKey` | int64/64 | off 0, size 8 | key | `tilingParams.tilingKey`（保持初始化 0） | 未读；entry 用 launch template key | 不可用它路由 |
| 同上 | `totalCoreNum` | int64/64 | off 8 | AIV core | `GetCoreNumAiv` | `InitParams→totalCoreNum_`，其后未读 | >0 |
| 同上 | `usedCoreNum` | int64/64 | off 16 | AIV core | `rowTileNum*colTileNum` | `Init/Process` 空闲核 guard | ≥1 |
| 同上 | `ubSize` | int64/64 | off 24 | byte | `GetCoreMemSize(UB)` | 读入 `ubSize_` 后未用 | Host 选 tile |
| 同上 | `roundMode` | int64/64 | off 32 | enum | `GetRoundMode` | 未读；编译期模板消费 | 0/1/4 |
| 同上 | `dstType` | int64/64 | off 40 | ge dtype id | ACLNN attr | 未读；binary dtype 选择消费 | 35/36/40/41 |
| 同上 | `scaleAlg` | int64/64 | off 48 | enum | ACLNN attr | 未读；模板 `scaleAlg` 消费 | 0/2 |
| 同上 | `blockSizeRow` | int64/64 | off 56 | element | 常量 32 | `InitParams`; scale 半块逻辑 | 固定 32 |
| 同上 | `blockSizeCol` | int64/64 | off 64 | element | 常量 32 | scale offset/长度 | 固定 32 |
| 同上 | `batchNum` | int64/64 | off 72 | batch | shape | 读入后未读 | rank2 为1 |
| 同上 | `rowNum` | int64/64 | off 80 | row/batch | `x[-2]` | core/GM/tail 地址 | >0 |
| 同上 | `colNum` | int64/64 | off 88 | element/row | `x[-1]` | GM stride/tail | >0；FP4 偶数 |
| 同上 | `singleBatchRowBlockLoopNum` | int64/64 | off 96 | 64-row block | `ceil(M/64)` | batch 边界与尾行 | ≥1 |
| 同上 | `rowBlockLoopNum` | int64/64 | off 104 | 64-row block | `batch*ceil(M/64)` | 读入后未读 | Host 分核 |
| 同上 | `colBlockLoopNum` | int64/64 | off 112 | 256-col block | `ceil(N/256)` | 读入后未读 | Host 分核 |
| 同上 | `rowUbBlockLoopNum` | int64/64 | off 120 | 64-row block/loop | `SplitCore:r` | UB loop 与 buffer | ≥1 |
| 同上 | `colUbBlockLoopNum` | int64/64 | off 128 | 256-col block/loop | `SplitCore:c` | UB loop 与 buffer | ≥1 |
| 同上 | `rowUbFactor` | int64/64 | off 136 | row | `64*r` | queue capacity | 64 对齐 |
| 同上 | `colUbFactor` | int64/64 | off 144 | element/row | `256*c` | queue capacity | 256 对齐 |
| 同上 | `rowTileNum` | int64/64 | off 152 | core-grid row | `AutoTiling:m` | `blockIdx/colTileNum` 映射 | ≥1 |
| 同上 | `colTileNum` | int64/64 | off 160 | core-grid col | `AutoTiling:n` | core 坐标 | ≥1 |
| 同上 | `normalCoreRowTileNum` | int64/64 | off 168 | block/core | `ceil(rowBlocks/m)` | normal core extent | ≥tail |
| 同上 | `normalCoreColTileNum` | int64/64 | off 176 | block/core | `ceil(colBlocks/n)` | normal core extent | ≥tail |
| 同上 | `tailCoreRowTileNum` | int64/64 | off 184 | block/core | `floor(rowBlocks/m)` | tail core extent | 可与 normal 相等 |
| 同上 | `tailCoreColTileNum` | int64/64 | off 192 | block/core | `floor(colBlocks/n)` | tail core extent | 可与 normal 相等 |
| 同上 | `rowNormalCoreNum` | int64/64 | off 200 | core | 余数或全部 row cores | 判断/定位行尾核 | 1..rowTileNum |
| 同上 | `colNormalCoreNum` | int64/64 | off 208 | core | 余数或全部 col cores | 判断/定位列尾核 | 1..colTileNum |
| 同上 | `rowTailCoreNum` | int64/64 | off 216 | core | `rowTileNum-rowNormalCoreNum` | 读入后未读 | 0..rowTileNum-1 |
| 同上 | `colTailCoreNum` | int64/64 | off 224 | core | `colTileNum-colNormalCoreNum` | 读入后未读 | 0..colTileNum-1 |
| 同上 | `blockH` | int64/64 | off 232 | row | 常量 64 | 未读；Device 成员固定 64 | 固定 64 |
| 同上 | `blockW` | int64/64 | off 240 | element/row | 常量 256 | 未读；Device 成员固定 256 | 固定 256 |
| 同上 | `rowScaleNum` | int64/64 | off 248 | E8M0/column | `2*ceil(M/64)` | 读入后未读 | 偶数 |
| 同上 | `colScaleNum` | int64/64 | off 256 | E8M0/row | `2*ceil(N/64)` | scale1 GM stride | 偶数 |
| 同上 | `dstTypeMax` | float/32 | off 264, tail pad 4 | scalar | ACLNN double 经 OpDef float attr | `InitParams→bitForMantissa_` | DDR: 0/6/7 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-016 | `GET_TPL_TILING_KEY(round,0)` + binary dtype | `DoTiling/SetTilingKey` | `DynamicBlockMxQuantTilingData` | `dynamic_block_mx_quant<round,0>` | `Base::Init/Process/CopyIn/ComputeOcp/ComputeY*/CopyOut` | 950 CMake + 8 dtype binaries + global config |
| R-017～R-022 | `GET_TPL_TILING_KEY(round,2)` + E2M1 binary | 同上 | 同上，`dstTypeMax` active | `dynamic_block_mx_quant<round,2>` | `Base::Init/Process/CopyIn/ComputeDdr/ComputeY1ToFP4/CopyOut` | 同上；Host guard 限 E2M1 |

## Kernel 实现

### 公共执行流程

entry 保存 950 的 overflow control SPR，解析 272B POD，构造 `TPipe` 和 `DynamicBlockMxQuantBase<DTYPE_X,DTYPE_Y,mappedRound,scaleAlg>`，调用 `Init→Process` 后恢复 SPR。`Init` 绑定每核 GM 起点并分配 UB；`Process` 生成 8×32B gather index，对每个 `(rowUbLoop,colUbLoop)` 严格执行 `CopyIn→Compute→CopyOut`。

`CopyIn` 将若干 `64×256` tile 逐块搬入连续 UB，列尾先清零；`Compute` 把每个 64 行块拆成两个 32 行半块，先用 OCP 或 DDR 求 E8M0 scale 与 BF16 reciprocal，再用 `ComputeY1ToFP4/FP8` 量化，第二半不存在时给 scale2 补零，最后交织两半 scale2；`CopyOut` 仅写有效行列，scale1 写偶数 pair，scale2 以两行 scale 为一个交织块。

所有 Route 的单节点函数链为：

```text
TilingForDynamicBlockMxQuant -> DynamicBlockMxQuantTiling::DoTiling
aclnn executor -> dynamic_block_mx_quant<roundMode,scaleAlg>(x,y,scale1,scale2,workspace,tiling)
  -> Base::Init -> InitParams + InitBuffer + GM binding
  -> Base::Process
    -> CopyIn
    -> Compute
       -> ComputeOcp [scaleAlg=0] | ComputeDdr [scaleAlg=2,E2M1]
       -> ComputeYVf -> ComputeY1ToFP4 | ComputeY1ToFP8
       -> Interleave(scale2 half0,half1)
    -> CopyOut
```

公共源码等价伪码：

```text
core=(blockIdx/colTileNum, blockIdx%colTileNum)
derive normal/tail rowBlocks,colBlocks and GM bases
for rowChunk in ceil(coreRowBlocks/r):
  for colChunk in ceil(coreColBlocks/c):
    zero whole input UB if this chunk has a short final 256-column block
    for each 64x256 block: DataCopyPad(validRows,validCols*2B, GM row stride, UB pad)
    for each block:
      split validRows -> row1<=32,row2<=32
      scale/reciprocal = ComputeOcp(...) or ComputeDdr(...)
      packedY = ComputeY1ToFP4/FP8(x,reciprocal,roundMode)
      if row2==0: scale2_half1=0 else compute second half
      Interleave(scale2_half0,scale2_half1)
    DataCopyPad valid y, padded scale1, interleaved scale2 back to GM
```

公共资源账（`r=rowUbBlockLoopNum,c=colUbBlockLoopNum,V=GetVRegSize()`）：

```text
inQueue       = 2*(64r*256c*2) = 65536rc B
outQueue      = 2*(64r*256c)   = 32768rc B
scale1Queue   = 2*(64r*c*32)   = 4096rc B
scale2Queue   = 2*(r*2*c*256)  = 1024rc B
reciprocalBuf = r*2*c*32*2     = 128rc B
tempIndexBuf  = V B (Host 以 256B gather reserve 计)
UB_total      = 103552rc + V B <= ubSize-1024 B（当 V=256，与 Host guard 等价）
dynamic_ub    = N/A；AIV TPipe 静态分配
workspace_total(operator-owned) = 0；library workspace = GetLibApiWorkSpaceSize()
```

```mermaid
block-beta
  columns 4
  inQ["inQueue VECIN · depth=2 · 65536rc B"]:2
  outQ["outQueue VECOUT · depth=2 · 32768rc B"]:2
  s1Q["scale1Queue VECOUT · depth=2 · 4096rc B"]:1
  s2Q["scale2Queue VECOUT · depth=2 · 1024rc B"]:1
  recip["mxScaleReciprocalBuf · 128rc B"]:1
  index["tempIndexBuf · V B"]:1
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueue` 当前 slot | `CopyIn` MTE2 | `Compute` Vector | `FreeTensor` 后下一 chunk | `EnQue→DeQue→FreeTensor` | depth=2 容量计足 |
| `outQueue` 当前 slot | `ComputeY*` Vector | `CopyOut` MTE3 | `FreeTensor` 后下一 chunk | `EnQue→DeQue→FreeTensor` | 仅搬有效 tail |
| `scale1/2Queue` 当前 slot | `ComputeOcp/Ddr` | `CopyOut` | `FreeTensor` 后 | Queue 协议 | scale2 交织在 EnQue 前完成 |
| `mxScaleReciprocalBuf` | `ComputeOcp/Ddr` | 紧随其后的 `ComputeY*` | 下一个 32-row half | 同一 Vector 顺序 | 不跨 half 保留 |
| `tempIndexBuf` | `Process` 开始时 8 次 `Duplicate` | 每次 `Gather` | 整个 Kernel 不再覆盖 | 同一核程序顺序 | 容量至少一个 VReg |

下面各 Route 章节给出该实例的选择、计算和精度差异；地址、资源与 tail 均采用上述逐字段闭合的公共实现，不存在隐藏的第二 Kernel。

### Route R-001：FP16→E2M1 OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/dynamic_block_mx_quant_tiling.cpp` | `GetAttr/SetTilingKey` | `x=FP16,y=E2M1,scaleAlg=0,round="rint"` | key tuple `(4,0)` |
| Router | `op_kernel/dynamic_block_mx_quant.cpp` | `RoundModeMapper<4>` | FP4 + `TPL_RINT` | `CAST_RINT`; `Base<half,fp4x2_e2m1_t,CAST_RINT,0>` |

#### Device 数据流

##### 函数级调用链

```text
dynamic_block_mx_quant<4,0> -> Init -> Process -> CopyIn
  -> ComputeOcp -> ComputeY1ToFP4 -> ComputeFP4FromHalf(E2M1,RINT) -> CopyOut
```

##### 源码等价伪码

```text
FP16 exponent -> BF16 exponent; reduce each 32x32 -> E8M0 scale + BF16 reciprocal
FP16 -> FP32; multiply reciprocal; E2M1 exponent-aware Truncate<RINT>; cast/pack two FP4 per byte
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM↔4Q+2Buf | core/chunk 公式见公共流程 | `103552rc+V` B | FP16 input, packed E2M1 output | `CopyIn/ComputeOcp` | `ComputeY1ToFP4/CopyOut` | 单核 TPipe | 32B，depth=2 |

##### 分层内存布局

使用公共 `inQ/outQ/scale1Q/scale2Q/reciprocal/index` 六区布局；本 Route 没有条件增删 buffer。

##### 字节账与复用顺序

`UB_total=103552rc+V`，FP4 有效输出为 `validRows*validCols/2` B；reciprocal 在每个 32-row half 内由 `ComputeOcp` 写、由 `ComputeY1ToFP4` 读后覆盖。workspace 自有 0B。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| copy/compute/out | Queue + tail `V_MTE2` event | MTE2/Vector | `DeQue`/`WaitFlag`/MTE3 | 当前 slot | FP16→BF16 exponent，量化乘法 FP32，RINT；列尾零化且只回写有效字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/dynamic_block_mx_quant_tiling.cpp` | `GetAttr`, `SetTilingKey` | 校验并生成 `(4,0)` |
| Kernel entry/router | `op_kernel/dynamic_block_mx_quant.cpp` | `dynamic_block_mx_quant<4,0>` | RINT 映射和模板构造 |
| Device body/helper | `op_kernel/arch35/dynamic_block_mx_quant_base.h` | `ComputeOcp`, `ComputeY1ToFP4`, `ComputeFP4FromHalf` | OCP scale、FP32 乘 reciprocal、E2M1 RINT 打包 |

### Route R-002：FP16→E2M1 OCP ROUND

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `op_host/dynamic_block_mx_quant_tiling.cpp`; `op_kernel/dynamic_block_mx_quant.cpp` | `SetTilingKey`; `RoundModeMapper<0>` | FP16→E2M1，`scaleAlg=0,round="round"` | `(0,0)` → `Base<half,E2M1,CAST_ROUND,0>` |

#### Device 数据流

##### 函数级调用链

`entry<0,0> → CopyIn → ComputeOcp → ComputeY1ToFP4 → ComputeFP4FromHalf(E2M1,ROUND) → CopyOut`。

##### 源码等价伪码

与 R-001 地址/scale 顺序相同；唯一精度分支是 E2M1 FP32 中间值执行 `Truncate<CAST_ROUND>`，随后饱和 cast/pack。

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | 容量 | 路径 | tail |
|---|---|---|---|
| 公共 4Q+2Buf | `103552rc+V` B | FP16 GM→UB→packed E2M1 GM | 列尾先清零、有效 `N/2` 字节回写 |

##### 分层内存布局

与公共六区布局完全相同，无 alias 差异。

##### 字节账与复用顺序

`UB_total=103552rc+V`；reciprocal half 内复用，Queue 以 `EnQue/DeQue/FreeTensor` 轮转；workspace 自有 0B。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer→consumer | 精度/尾块 |
|---|---|---|---|
| 全流程 | Queue；尾列 `V_MTE2` | `CopyIn→Compute→CopyOut` | FP32 乘法 + ROUND；GM tail 保护同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/entry/body | `op_host/dynamic_block_mx_quant_tiling.cpp`; `op_kernel/dynamic_block_mx_quant.cpp`; `op_kernel/arch35/dynamic_block_mx_quant_base.h` | `SetTilingKey`; `RoundModeMapper<0>`; `ComputeFP4FromHalf` | key `(0,0)`；ROUND 传入 `Reg::Truncate` |

### Route R-003：FP16→E2M1 OCP FLOOR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `op_host/dynamic_block_mx_quant_tiling.cpp`; `op_kernel/dynamic_block_mx_quant.cpp` | `SetTilingKey`; `RoundModeMapper<1>` | FP16→E2M1，`scaleAlg=0,round="floor"` | `(1,0)` → `Base<half,E2M1,CAST_FLOOR,0>` |

#### Device 数据流

##### 函数级调用链

`entry<1,0> → CopyIn → ComputeOcp → ComputeY1ToFP4 → ComputeFP4FromHalf(E2M1,FLOOR) → CopyOut`。

##### 源码等价伪码

按 32×32 指数归约求 scale；FP16 升 FP32 乘 reciprocal，E2M1 helper 用 `Truncate<CAST_FLOOR>`，然后 pack。

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲区 | 容量 | producer→consumer | 复用 |
|---|---|---|---|
| 公共六区 | `103552rc+V` B | `CopyIn/ComputeOcp→ComputeY1ToFP4/CopyOut` | half 内 reciprocal，chunk 间 Queue slot |

##### 分层内存布局

同公共六区，无条件 buffer。

##### 字节账与复用顺序

`UB_total=103552rc+V`；FP4 output Queue 逻辑只消费一半有效字节，物理仍按 `rowUbFactor*colUbFactor` 每 slot 分配。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 顺序 | 精度 | tail |
|---|---|---|---|
| Queue + `V_MTE2` | MTE2→Vector→MTE3 | FP32 + FLOOR | 零填无效 lane，只提交有效区 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| `dynamic_block_mx_quant_tiling.cpp`; `dynamic_block_mx_quant.cpp`; `dynamic_block_mx_quant_base.h` | `SetTilingKey`; `RoundModeMapper<1>`; `ComputeFP4FromHalf` | `(1,0)`、FLOOR E2M1 OCP |

### Route R-004：BF16→E2M1 OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<4>` | BF16→E2M1, alg0, rint | `(4,0)` → `Base<bfloat16_t,E2M1,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp(BF16 exponent) → ComputeY1ToFP4(BF16) → ComputeFP4FromHalf(E2M1,RINT，仅 FP16 子分支不调用) → CopyOut`；实际 BF16 Y 路径在 `ComputeY1ToFP4` 直接 BF16 multiply/interleave/cast。

##### 源码等价伪码

`BF16 exponent reduce → reciprocal(BF16 bits); BF16 x*=reciprocal → interleave → saturating Cast<E2M1,RINT> → pack`。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | dtype/layout | 生命周期 |
|---|---|---|---|
| 公共六区 | `103552rc+V` B | BF16 input / packed FP4 y | 当前 core Kernel |

##### 分层内存布局

同公共布局；BF16 和 FP16 都是 2B，UB 容量不变。

##### 字节账与复用顺序

`UB_total=103552rc+V`；reciprocal 在 BF16 `Reg::Mul` 后即可覆盖，Queue slot 在 `FreeTensor` 后复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | producer→consumer | 精度 | tail |
|---|---|---|---|
| Queue/event | copy→BF16 vector→copyout | BF16 乘 reciprocal，RINT 饱和转 FP4 | 同公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| `op_kernel/arch35/dynamic_block_mx_quant_base.h` | `ComputeOcp`; `ComputeY1ToFP4` BF16 branch | 直接提取 BF16 指数，BF16 原位乘后 cast |

### Route R-005：BF16→E2M1 OCP ROUND

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<0>` | BF16→E2M1, alg0, round | `(0,0)` → `Base<bfloat16_t,E2M1,ROUND,0>` |

#### Device 数据流

##### 函数级调用链

`entry<0,0> → CopyIn → ComputeOcp → ComputeY1ToFP4(BF16 multiply/interleave, ROUND cast) → CopyOut`。

##### 源码等价伪码

BF16 exponent OCP scale；BF16 值乘 reciprocal 后以 `castTraitBF16toFp4` 的 `CAST_ROUND` 饱和量化。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | 数据流 | tail |
|---|---|---|---|
| 4Q+2Buf | `103552rc+V` B | BF16 GM→UB→packed E2M1 GM | 有效列 `/2` 字节 |

##### 分层内存布局

同公共六区。

##### 字节账与复用顺序

公共字节账和 Queue/TBuf 生命周期原样适用；无额外 FP32 UB 物化，FP32 不落 UB。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 顺序 | 精度/尾块 |
|---|---|---|
| Queue + event | MTE2→Vector→MTE3 | BF16 multiply + ROUND saturating cast；tail 补零/截写 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<0>`; `ComputeY1ToFP4` | `(0,0)` 和 BF16 ROUND 路径 |

### Route R-006：BF16→E2M1 OCP FLOOR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<1>` | BF16→E2M1, alg0, floor | `(1,0)` → `Base<bfloat16_t,E2M1,FLOOR,0>` |

#### Device 数据流

##### 函数级调用链

`entry<1,0> → CopyIn → ComputeOcp → ComputeY1ToFP4(BF16,FLOOR) → CopyOut`。

##### 源码等价伪码

OCP 指数归约和 reciprocal 不变；最终 BF16→FP4 cast 的 round trait 为 FLOOR。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | owner/lifetime | 对齐 |
|---|---|---|---|
| 公共六区 | `103552rc+V` B | TPipe / 单 Kernel | 32B |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；reciprocal 只在当前 32-row half 有效，scale/y Queue 在 CopyOut 后释放。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度 | tail |
|---|---|---|
| Queue/event + `Cast` | BF16 multiply + FLOOR | 固定 256 列 Vector，GM 只写有效列 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<1>`; `ComputeY1ToFP4` | `(1,0)` BF16 FLOOR OCP |

### Route R-007：FP16→E1M2 OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<4>` | FP16→E1M2, alg0, rint | `(4,0)` → `Base<half,E1M2,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp(yMaxExp=0) → ComputeY1ToFP4 → ComputeFP4FromHalf(E1M2,RINT) → CopyOut`。

##### 源码等价伪码

FP16→FP32 乘 reciprocal；E1M2 helper 执行 `v*=4; Truncate<RINT>(v); v*=0.25`，保留负零规则后 cast/pack。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | dtype | 复用 |
|---|---|---|---|
| 公共六区 | `103552rc+V` B | FP16 / packed E1M2 | Queue + half reciprocal |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

`UB_total=103552rc+V`；全部 FP32 仅在 RegTensor，未增加 UB buffer。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | FP16→FP32，E1M2 RINT，保留 `-0` | 公共零填与有效回写 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| `dynamic_block_mx_quant_base.h` | `InitParams`; `ComputeFP4FromHalf` | E1M2 `dtypeYMaxExp=0`，`×4/truncate/×1/4` |

### Route R-008：FP16→E1M2 OCP ROUND

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<0>` | FP16→E1M2, alg0, round | `(0,0)` → `Base<half,E1M2,ROUND,0>` |

#### Device 数据流

##### 函数级调用链

`entry<0,0> → CopyIn → ComputeOcp → ComputeY1ToFP4 → ComputeFP4FromHalf(E1M2,ROUND) → CopyOut`。

##### 源码等价伪码

`FP16→FP32; *=reciprocal; *=4; Truncate<ROUND>; *=0.25; preserve -0; cast E1M2; pack`。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | 数据流 |
|---|---|---|
| 公共六区 | `103552rc+V` B | FP16 input、E8M0 scales、packed E1M2 output |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；RegTensor 临时量不计入可证明 UB，生命周期局限于当前 Vector scope。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | producer→consumer | 精度/tail |
|---|---|---|
| Queue/event + `Truncate<ROUND>` | copy→compute→copyout | E1M2 ROUND；尾块同公共规则 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<0>`; `ComputeFP4FromHalf` | `(0,0)` 与 E1M2 ROUND |

### Route R-009：FP16→E1M2 OCP FLOOR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<1>` | FP16→E1M2, alg0, floor | `(1,0)` → `Base<half,E1M2,FLOOR,0>` |

#### Device 数据流

##### 函数级调用链

`entry<1,0> → CopyIn → ComputeOcp → ComputeY1ToFP4 → ComputeFP4FromHalf(E1M2,FLOOR) → CopyOut`。

##### 源码等价伪码

与 R-008 相同，但 `Truncate<FLOOR>`；负零 mask 在 truncate 后恢复。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | 对齐/复用 |
|---|---|---|
| 公共六区 | `103552rc+V` B | 32B，Queue depth=2，reciprocal half 内覆盖 |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；packed output 实际有效量为元素数一半。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | FP32 E1M2 FLOOR + `-0` 修复 | 无效 lane 零、GM 截写 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<1>`; `ComputeFP4FromHalf` | `(1,0)` 与 E1M2 FLOOR |

### Route R-010：BF16→E1M2 OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<4>` | BF16→E1M2, alg0, rint | `(4,0)` → `Base<bfloat16_t,E1M2,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp → ComputeY1ToFP4(BF16 multiply/interleave, E1M2 RINT cast) → CopyOut`。

##### 源码等价伪码

BF16 exponent OCP scale；`x*=BF16 reciprocal` 后直接通过 `castTraitBF16toFp4<RINT>` 转 E1M2 并 pack，不走 FP16 专用 FP32 helper。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | dtype/layout |
|---|---|---|
| 公共六区 | `103552rc+V` B | BF16 / packed E1M2 / E8M0 |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账与 Queue 生命周期；无额外落 UB 的 FP32 临时区。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | BF16 multiply + RINT saturating cast | 公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| `dynamic_block_mx_quant_base.h` | `ComputeOcp`; `ComputeY1ToFP4` | BF16 OCP/E1M2 RINT 实体 |

### Route R-011：BF16→E1M2 OCP ROUND

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<0>` | BF16→E1M2, alg0, round | `(0,0)` → `Base<bfloat16_t,E1M2,ROUND,0>` |

#### Device 数据流

##### 函数级调用链

`entry<0,0> → CopyIn → ComputeOcp → ComputeY1ToFP4(BF16,ROUND) → CopyOut`。

##### 源码等价伪码

BF16 `x*=reciprocal; Interleave; Cast<E1M2,ROUND,SAT>; PACK4`。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | 生命周期 |
|---|---|---|
| 4Q+2Buf | `103552rc+V` B | 单核单 Kernel；chunk 间复用 |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账，FP4 Queue 物理容量不因打包减半。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度/tail |
|---|---|
| Queue/event | BF16 ROUND；尾列零化、有效打包字节回写 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<0>`; `ComputeY1ToFP4` | BF16→E1M2 ROUND |

### Route R-012：BF16→E1M2 OCP FLOOR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `SetTilingKey`; `RoundModeMapper<1>` | BF16→E1M2, alg0, floor | `(1,0)` → `Base<bfloat16_t,E1M2,FLOOR,0>` |

#### Device 数据流

##### 函数级调用链

`entry<1,0> → CopyIn → ComputeOcp → ComputeY1ToFP4(BF16,FLOOR) → CopyOut`。

##### 源码等价伪码

BF16 OCP scale 和乘法；最终 `castTraitBF16toFp4` 以 FLOOR 饱和转换并打包。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | producer→consumer |
|---|---|---|
| 公共六区 | `103552rc+V` B | CopyIn→OCP/FP4→CopyOut |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；所有复用在 Queue/当前 half 顺序内闭合。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度 | tail |
|---|---|---|
| Queue/event + Cast | BF16 FLOOR | 公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<1>`; `ComputeY1ToFP4` | BF16→E1M2 FLOOR |

### Route R-013：FP16→E4M3FN OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<4>` | FP16→E4M3FN, alg0；FP8 强制 rint | `(4,0)` → `Base<half,fp8_e4m3fn_t,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp(dtypeYMaxExp=0x0400) → ComputeY1ToFP8(FP16) → CopyOut`。

##### 源码等价伪码

FP16 exponent 先按异常 mask 转 BF16 exponent并归约；FP16 拆为两组 FP32，乘 BF16 reciprocal 的 FP32 展开值，交织后 `Cast<E4M3FN,RINT,SAT>`，每行写 256B。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | dtype/layout | tail |
|---|---|---|---|
| 公共六区 | `103552rc+V` B | FP16 / FP8 / E8M0 | y 每元素 1B，只回写有效列 |

##### 分层内存布局

同公共六区；FP8 output 的逻辑有效量等于输入元素数。

##### 字节账与复用顺序

公共字节账；四次 `DIST_PACK4_B32` 各写 64B，合成每行 256B，RegTensor 不落 UB。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | FP16→FP32 multiply→E4M3FN RINT/SAT | 无效 lane 为0，MTE3 截到 `validCols` |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| entry/base | `RoundModeMapper`; `InitParams`; `ComputeOcp`; `ComputeY1ToFP8` | FP8 固定 RINT、E4M3 exponent 常量及 FP16 分支 |

### Route R-014：BF16→E4M3FN OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper` | BF16→E4M3FN, alg0, rint | `(4,0)` → `Base<bfloat16_t,E4M3FN,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp(BF16) → ComputeY1ToFP8(BF16) → CopyOut`。

##### 源码等价伪码

BF16 exponent reduce；BF16 x 原位乘 reciprocal，再拆/交织成 FP32 lanes，饱和 RINT 转 E4M3FN，pack 到 256B/row。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | 数据流 |
|---|---|---|
| 公共六区 | `103552rc+V` B | BF16 GM→UB；FP8/E8M0 UB→GM |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；BF16 与 FP16 同为2B，buffer 无变化；reciprocal half 内覆盖。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度 | tail |
|---|---|---|
| Queue/event + Reg Cast | BF16 multiply→FP32 lane→E4M3FN RINT/SAT | 公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| `dynamic_block_mx_quant_base.h` | `ComputeOcp`; `ComputeY1ToFP8` BF16 branch | BF16→E4M3FN 路径 |

### Route R-015：FP16→E5M2 OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper` | FP16→E5M2, alg0, rint | `(4,0)` → `Base<half,fp8_e5m2_t,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp(dtypeYMaxExp=0x0780) → ComputeY1ToFP8(FP16) → CopyOut`。

##### 源码等价伪码

与 R-013 相同的数据布局；scale 指数阈值换成 E5M2 `0x0780`，最终 cast target 为 E5M2。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | dtype/layout |
|---|---|---|
| 公共六区 | `103552rc+V` B | FP16 / E5M2 / E8M0 |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；每有效元素一个 FP8 byte，Queue slot 完整按 64×256 分配。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | FP32 multiply→E5M2 RINT/SAT | 公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| base | `InitParams`; `ComputeOcp`; `ComputeY1ToFP8` | E5M2 exponent 与 FP16 量化 |

### Route R-016：BF16→E5M2 OCP RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper` | BF16→E5M2, alg0, rint | `(4,0)` → `Base<bfloat16_t,E5M2,RINT,0>` |

#### Device 数据流

##### 函数级调用链

`entry<4,0> → CopyIn → ComputeOcp(BF16,E5M2) → ComputeY1ToFP8(BF16) → CopyOut`。

##### 源码等价伪码

BF16 exponent reduce，以 `0x0780` 得 scale；BF16 原位乘 reciprocal，转 FP32 lanes，再 RINT/SAT 转 E5M2。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | owner/lifetime |
|---|---|---|
| 公共六区 | `103552rc+V` B | TPipe；当前 K1/core |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账及 Queue/TBuf 复用；无 workspace edge。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度/tail |
|---|---|
| Queue/event + Reg Cast | BF16→FP32 lane→E5M2 RINT；tail 同公共规则 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| base | `ComputeOcp`; `ComputeY1ToFP8` BF16 branch | BF16→E5M2 路径 |

### Route R-017：FP16→E2M1 DDR RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Tiling | `op_host/dynamic_block_mx_quant_tiling.cpp` | `GetAttr/SetTilingKey` | FP16→E2M1, alg2, max 0/6/7, rint | `(4,2)` |
| Device | base | `Compute` constexpr branch | `scaleAlg==2 && U==E2M1` | `ComputeDdr`; Y 为 E2M1 RINT |

#### Device 数据流

##### 函数级调用链

`entry<4,2> → CopyIn → ComputeDdr(FP16 abs/cast BF16) → ComputeY1ToFP4 → ComputeFP4FromHalf(E2M1,RINT) → CopyOut`。

##### 源码等价伪码

`abs(FP16 bits) → finite mask → Cast<BF16,RINT> → reduce max(exp+mantissa) → mantissa threshold(0/6:c1,7:e1) → E8M0 scale/reciprocal → FP32 multiply → E2M1 RINT pack`。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | producer→consumer | 差异 |
|---|---|---|---|
| 公共六区 | `103552rc+V` B | `ComputeDdr→ComputeY1ToFP4` | 与 OCP 相同物理 buffer，仅 scale Reg 算法不同 |

##### 分层内存布局

同公共布局，无 DDR 额外 UB；这里的 DDR 是属性算法名，不是 NDDMA 执行族。

##### 字节账与复用顺序

`UB_total=103552rc+V`；mantissa/abs/max 全在 RegTensor，reciprocal 仍使用同一 TBuf half 内复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | 机制 | 精度 | tail |
|---|---|---|---|
| scale/Y | Vector program order + Queue | FP16 max 先 RINT 到 BF16；Y 乘法 FP32，E2M1 RINT | padding 0 参与完整块但不改变绝对值 max |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `GetAttr`; `dynamic_block_mx_quant<4,2>`; `ComputeDdr`; `ComputeFP4FromHalf` | DDR 限域、FP16 max 与 RINT Y |

### Route R-018：FP16→E2M1 DDR ROUND

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<0>` | FP16→E2M1, alg2, max 0/6/7, round | `(0,2)` → `Base<half,E2M1,ROUND,2>` |

#### Device 数据流

##### 函数级调用链

`entry<0,2> → CopyIn → ComputeDdr(FP16) → ComputeY1ToFP4 → ComputeFP4FromHalf(E2M1,ROUND) → CopyOut`。

##### 源码等价伪码

DDR scale 路线与 R-017 相同；FP16→BF16 max 的内部 cast 固定 RINT，而最终 E2M1 `Truncate` 使用 ROUND，二者不可混同。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | 复用 |
|---|---|---|
| 公共六区 | `103552rc+V` B | Queue chunk 轮转；reciprocal half 内覆盖 |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；FP32/Reg max 不物化 UB。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | max cast RINT；Y cast ROUND | 公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| entry/base | `RoundModeMapper<0>`; `ComputeDdr`; `ComputeFP4FromHalf` | `(0,2)` 与双重 rounding 语义 |

### Route R-019：FP16→E2M1 DDR FLOOR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<1>` | FP16→E2M1, alg2, max 0/6/7, floor | `(1,2)` → `Base<half,E2M1,FLOOR,2>` |

#### Device 数据流

##### 函数级调用链

`entry<1,2> → CopyIn → ComputeDdr(FP16) → ComputeY1ToFP4 → ComputeFP4FromHalf(E2M1,FLOOR) → CopyOut`。

##### 源码等价伪码

FP16 max 路径仍固定 RINT 转 BF16；最终 scaled FP32 值以 FLOOR 量化到 E2M1。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | tail |
|---|---|---|
| 公共六区 | `103552rc+V` B | 列尾零，FP4 GM 截写 `validCols/2` |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；scale1/2 和 y 同 chunk 在三条 VECOUT Queue 中独立排队。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度/tail |
|---|---|
| Queue/event | max RINT→BF16，Y FLOOR→E2M1；padding zero 安全参与 max |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| entry/base | `RoundModeMapper<1>`; `ComputeDdr`; `ComputeFP4FromHalf` | `(1,2)` FP16 DDR FLOOR |

### Route R-020：BF16→E2M1 DDR RINT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<4>` | BF16→E2M1, alg2, max 0/6/7, rint | `(4,2)` → `Base<bfloat16_t,E2M1,RINT,2>` |

#### Device 数据流

##### 函数级调用链

`entry<4,2> → CopyIn → ComputeDdr(BF16 abs/max) → ComputeY1ToFP4(BF16,RINT) → CopyOut`。

##### 源码等价伪码

`abs(BF16 bits) → reduce exp+mantissa → threshold from dstTypeMax → scale/reciprocal; BF16 x*=reciprocal → E2M1 RINT/SAT → pack`。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | dtype/layout |
|---|---|---|
| 公共六区 | `103552rc+V` B | BF16 / packed E2M1 / E8M0 |

##### 分层内存布局

同公共布局；DDR 不增加 TBuf。

##### 字节账与复用顺序

公共字节账；BF16 reciprocal 在当前 half 的 Y 计算结束后覆盖。

#### 同步、精度与尾块

##### 同步与数值边界表

| 机制 | 精度 | tail |
|---|---|---|
| Queue/event | BF16 max/乘法，E2M1 RINT/SAT | 公共 tail |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/base | `GetAttr`; `ComputeDdr`; `ComputeY1ToFP4` BF16 branch | DDR max guard 与 BF16 RINT Y |

### Route R-021：BF16→E2M1 DDR ROUND

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<0>` | BF16→E2M1, alg2, max 0/6/7, round | `(0,2)` → `Base<bfloat16_t,E2M1,ROUND,2>` |

#### Device 数据流

##### 函数级调用链

`entry<0,2> → CopyIn → ComputeDdr(BF16) → ComputeY1ToFP4(BF16,ROUND) → CopyOut`。

##### 源码等价伪码

BF16 raw magnitude 归约求 DDR scale；BF16 乘 reciprocal，interleave 后 ROUND/SAT 转 E2M1。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | producer→consumer |
|---|---|---|
| 公共六区 | `103552rc+V` B | CopyIn→DDR scale/Y→CopyOut |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

公共字节账；所有 RegTensor 生命周期局限于 `__VEC_SCOPE__`。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度/tail |
|---|---|
| Queue/event + Cast | BF16 ROUND→E2M1；padding 0 和有效 GM 截写 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| entry/base | `RoundModeMapper<0>`; `ComputeDdr`; `ComputeY1ToFP4` | `(0,2)` BF16 DDR ROUND |

### Route R-022：BF16→E2M1 DDR FLOOR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 谓词 | 结果 |
|---|---|---|---|---|
| Host/Router | tiling + entry | `GetAttr`; `RoundModeMapper<1>` | BF16→E2M1, alg2, max 0/6/7, floor | `(1,2)` → `Base<bfloat16_t,E2M1,FLOOR,2>` |

#### Device 数据流

##### 函数级调用链

`entry<1,2> → CopyIn → ComputeDdr(BF16) → ComputeY1ToFP4(BF16,FLOOR) → CopyOut`。

##### 源码等价伪码

BF16 magnitude/threshold 产生 E8M0 scale 与 reciprocal；BF16 scaled value 以 FLOOR/SAT 转 E2M1 并 pack。

#### 内存与数据搬运

##### 资源与搬运表

| 区域 | 容量 | owner/lifetime |
|---|---|---|
| 公共六区 | `103552rc+V` B | 当前 AIV core；Kernel 返回释放 |

##### 分层内存布局

同公共布局。

##### 字节账与复用顺序

`UB_total=103552rc+V`；workspace 自有 0B；Queue 和 reciprocal 复用顺序与公共表一致。

#### 同步、精度与尾块

##### 同步与数值边界表

| primitive | 精度 | tail |
|---|---|---|
| Queue/event + Cast | BF16 FLOOR→E2M1 | 公共 tail，第二个 32-row half 缺失则 scale2 补0 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| tiling/entry/base | `SetTilingKey`; `RoundModeMapper<1>`; `ComputeDdr`; `ComputeY1ToFP4` | `(1,2)` BF16 DDR FLOOR 全链路 |

## 算子特定优化

### Trick T-001：Host 字节账与 Device `InitBuffer` 同构的二维 UB 切块

#### 触发与源码依据

- 适用 Route：R-001～R-022。
- 触发条件：任意合法输入；950 UB 满足至少一个 `64×256` tile 后，Host 先沿列、再沿行把尽可能多的 tile 放入 UB。
- Golden 来源：[`dynamic_block_mx_quant_tiling.cpp:337-359`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_tiling.cpp#L337-L359)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
perBlockTmpUbSize += tilingParams.blockH * tilingParams.blockW * (BYTES_OF_INPUT_TYPE + BYTES_OF_OUTPUT_Y_TYPE);
perBlockTmpUbSize += tilingParams.blockH * BLOCK_SIZE + tilingParams.blockH / BLOCK_SIZE * tilingParams.blockW;
perBlockTmpUbSize *= N_BUFFER;
int64_t scaleAlignedBlockW = Ops::Base::CeilAlign((tilingParams.blockW / BLOCK_SIZE), BLOCK_SIZE);
perBlockUbSize =
    perBlockTmpUbSize + (tilingParams.blockH / BLOCK_SIZE) * scaleAlignedBlockW * BYTES_OF_RECIPROCAL_SCALE_TYPE;
```

#### 机制与设计原因

这段公式逐项对应 Device 的 input/output/scale1/scale2 四个双缓冲 Queue 和 reciprocal TBuf，得到每 `64×256` tile 103552B。Host 先取 `c=min(maxBlocks,normalCoreColTileNum)`，再以剩余倍数取 `r`，使一次 UB chunk 连续覆盖 `r×c` 个 tile；Device 使用完全相同的 `rowUbFactor=64r,colUbFactor=256c` 初始化队列，从而不会以“有效 tail 长度”误算物理容量。二维 chunk 减少了每个 core 的外层 chunk 次数，同时保持每块 scale2 的两半相邻。

#### 迁移边界

- 适用：多个固定二维 tile 共用相同 Queue/TBuf 账本的 AIV 算子。
- 前提：950、2B input、1B 物理 output Queue、32B UB block、四个 Queue depth=2；改变 dtype 宽度或 buffer 深度必须重算 103552。
- 失效：`maxBlocks<1` 时源码没有小 tile fallback；不能照搬。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把 FP4 的逻辑输出字节减半后缩小 `outQueue_`：源码仍按每输入元素 1B 的物理 Queue 容量初始化。固定仓未提供可验证的更小 Queue 反例。

### Trick T-002：尾列先整块清零，再用 `DataCopyPad` 覆盖有效区

#### 触发与源码依据

- 适用 Route：R-001～R-022。
- 触发条件：当前 chunk 的 `blockColNum % 256 != 0`。
- Golden 来源：[`dynamic_block_mx_quant_base.h:436-460`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_base.h#L436-L460)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (blockColNum % blockW_ != 0) {
    Duplicate<T>(inLocal, static_cast<T>(0), rowUbFactor_ * colUbFactor_);
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
}
```

#### 机制与设计原因

Vector scale body 始终按 256 列 load，并按每 32 列归约；只给 DMA 行尾做局部 padding不足以证明所有未覆盖 tile/lane 已清零。源码先对整个本 chunk input Queue slot 清零，再以 `V_MTE2` event 保证 MTE2 不会在 Vector clear 完成前覆盖，之后 `DataCopyPad` 写有效行列。这样无效 lane 的绝对值为0，不污染 OCP/DDR 的块最大值，同时 `CopyOut` 仍只提交有效 `colBlockSize`。

#### 迁移边界

- 适用：固定宽 Vector reduction 处理不规则 GM 尾列。
- 前提：0 是 reduction 的中性元素，且使用相同的 Vector→MTE2 硬事件协议。
- 失效：若 reduction 中性元不是0，必须换 padding 值；整块本来对齐时该 clear 不执行。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

直接对未初始化的 256-column UB 尾部做 `ReduceMax` 会让无效 lane 参与 scale；源码通过整块清零显式避免该错误。

### Trick T-003：一次 scale 归约同时派生 scale1、scale2 与 reciprocal

#### 触发与源码依据

- 适用 Route：R-001～R-022；OCP 使用 `ComputeOcp`，DDR 使用等价的 gather/store 顺序。
- 触发条件：每个有效 32-row half、每个 256-column tile。
- Golden 来源：[`dynamic_block_mx_quant_base.h:574-596`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_base.h#L574-L596)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScaleB16);
for (uint16_t i = 0; i < rowNum; i++) {
    Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
        mxScale1Addr, mxScale1B8, ubBlockSize_ / sizeof(uint8_t), maskReduceB8);
}
Reg::LoadAlign(gatherIndex, gatherIndexAddr);
Reg::Gather(mxScale2B8, mxScale1B8, gatherIndex);
Reg::StoreAlign(mxScale2Addr, mxScale2B8, maskB8);
```

#### 机制与设计原因

每个 32×32 group 的最大指数只归约一次：pack 后的 8 个 E8M0 scale 被按行重复 store 为 scale1；预先生成的 `[0×32B,…,7×32B]` gather index 把同一向量扩展成每列 scale2；同一 `expMaxDim` 又生成 reciprocal 写入 TBuf，紧接着供 y 量化。因而不需要为两个 scale 输出和 y 分别重读 GM 或重复归约。scale2 的两个 32-row half 最后原位 `Interleave`，也不引入 workspace。

#### 迁移边界

- 适用：一个块统计量需广播到多个输出方向并立即用于量化的 Vector 算子。
- 前提：256-column tile 对应 8 个 32-column group，gather index 的每段恰为32B，scale 输出为 E8M0 byte。
- 失效：block/group 宽度或 scale dtype 改变时，`ScaleGatherIndex=8`、index buffer 与 Store mask 都必须重建。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

分别从 GM 重算 scale1、scale2 会重复读取和归约；本固定仓没有该反例实现，但当前共享 RegTensor 数据流证明额外重算并非所需。

## 模板与实现对比

| 对比维度 | R-001～R-012 OCP FP4 | R-013～R-016 OCP FP8 | R-017～R-022 DDR E2M1 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | alg0，FP4，三种 round | alg0，FP8，仅 rint | alg2，E2M1，max 0/6/7，三种 round | Host 先做组合校验，无运行时 fallback |
| 模板与 dtype | `T=half/bf16,U=E2M1/E1M2` | `T=half/bf16,U=E4M3/E5M2` | `T=half/bf16,U=E2M1,scaleAlg=2` | dtype 由 binary，round/alg 由 key |
| 循环与分核 | 统一 64×256，拆两个32行 half | 相同 | 相同 | scale1/2 布局统一 |
| Queue/TBuf/寄存器 | `103552rc+V` B，FP4 Reg/helper | 同容量，FP8 Reg | 同容量，DDR max Reg | output Queue 统一按1B/元素物理分配 |
| 内存总占用与复用 | packed GM y 为半字节/元素 | GM y 为1B/元素 | packed GM y | UB 账不随目标 dtype 改变 |
| 搬运/Vector API | exponent reduce；FP4 pack | exponent reduce；四路 FP8 pack | abs+mantissa reduce；FP4 pack | alg2 只替换 scale producer |
| 同步、精度与 tail | FP16 路径升 FP32；BF16 路径原位乘；round 可变 | 最终 RINT/SAT | max 的 FP16→BF16 固定 RINT，Y round 可变 | compile-time trait 消除运行时 round 分支 |
| fallback/失败边界 | 非法 dtype/round/shape 失败 | 非 rint 失败 | 非 E2M1/非法 max 失败 | 没有 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持；未生成卡 | 公开 README/API 明确不支持；未生成卡 | Ascend 950PR/950DT 公开支持 |
| Host/Launcher | N/A | N/A | `GetCoreNumAiv/GetCoreMemSize`，`SUPPORT_TILING_DIR=arch35` |
| Kernel/资源 | N/A | N/A | 单 `KERNEL_TYPE_MIX_AIV_1_0` AIV Kernel；950 8 个 input/output dtype binaries |

## 限制与失败边界

- 固定仓 Commit 为 `955f8b33985b6d72989ebff341bb9c2c564a51bc`；所有结论仅覆盖 CANN 9.1.0 此版本。
- 公开接口不支持 A2/A3、空 tensor、rank 非2/3、FP4 下奇数末维，以及表外 dtype/round/algorithm/max 组合；这些情况在 ACLNN 或 Tiling 返回错误，没有 Kernel fallback。
- `SplitCore` 没有 `maxBlocks==0` 的显式 guard，且随后用 `colUbBlockLoopNum` 作除数；950 运行前提是可用 UB 至少容纳一个 103552B tile、256B gather 与1024B reserve。
- `DynamicBlockMxQuantTilingData::tilingKey` 的 Host 参数对象未被 `SetTilingKey` 回填，因此 POD 中该字段保持0；当前 Kernel 不读取它，真实路由只由 launch template key 决定。复用 ABI 时不能误用 POD 字段做二次分发。
- ACLNN 生成 bridge/executor 和框架 Contiguous 的具体源码未落盘；它们被限定为框架边界。可证明的算子自有 Kernel workspace 为0，ACLNN 总 workspace 必须仍使用第一段返回值。
- `RegTensor` 的物理寄存器占用未由固定仓暴露；本文只记录作用域和数据依赖，不推测寄存器字节数。
- scale 的 INF/NaN、零与 subnormal 位级处理严格依赖 `ComputeOcp/ComputeDdr` 的 mask/select 顺序；迁移时不得用普通浮点 `log2/div` 替换并假定特殊值完全等价。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开支持 | `quant/dynamic_block_mx_quant/README.md` | `产品支持情况`, `调用说明` | 950 支持、A2/A3 不支持、ACLNN 暴露 |
| 入口声明/约束 | `quant/dynamic_block_mx_quant/docs/aclnnDynamicBlockMxQuant.md` | `aclnnDynamicBlockMxQuantGetWorkspaceSize`, `aclnnDynamicBlockMxQuant` | 公开 API、产品支持、参数/错误与公式 |
| 入口实现/OpDef | `quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_def.cpp` | `DynamicBlockMxQuant::DynamicBlockMxQuant`, `OP_ADD` | dtype 组合、默认属性、AutoContiguous、950 注册 |
| shape/dtype 推导 | `quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_infershape.cpp` | `InferShapeForDynamicBlockMxQuant`, `InferDataTypeForDynamicBlockMxQuant` | y/scale shape 与 dtype |
| Host/Tiling | `quant/dynamic_block_mx_quant/op_host/dynamic_block_mx_quant_tiling.cpp` | `DynamicBlockMxQuantTiling::*`, `TilingForDynamicBlockMxQuant` | 校验、分核、UB、key、POD、workspace |
| POD | `quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_tilingdata.h` | `DynamicBlockMxQuantTilingData` | 272B Host/Device ABI |
| 模板 key | `quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | roundMode×scaleAlg 值域 |
| Kernel entry | `quant/dynamic_block_mx_quant/op_kernel/dynamic_block_mx_quant.cpp` | `RoundModeMapper`, `dynamic_block_mx_quant` | key→round trait→body；单 AIV 节点 |
| Kernel body | `quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_base.h` | `DynamicBlockMxQuantBase::*` | GM/UB、OCP/DDR、FP4/FP8、同步和 tail |
| binary 接线 | `quant/dynamic_block_mx_quant/op_host/config/ascend950/dynamic_block_mx_quant_binary.json` | `op_list` 八项 | FP16/BF16×四种 y dtype binaries |
| 构建接线 | `quant/dynamic_block_mx_quant/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `add_modules_sources` | ascend950/arch35 与 `ACLNNTYPE aclnn` |
| 构建框架 | `cmake/func.cmake` | `add_modules_sources` | `ACLNNTYPE aclnn` 对 OpDef 的收集与 ACLNN 分类边界 |
| Kernel 全局配置 | `scripts/kernel/binary_config/ascendc_config.json` | `DynamicBlockMxQuant` 条目 | `compute_units=[ascend950]`, `auto_sync=false` |
| 公共平台依赖 | `common/inc/op_kernel/platform.h` | `GetUbBlockSize`, `GetVRegSize` | 32B UB block 与 950 vector-register size 来源 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 272B、8B pack、33×int64+float | 字段顺序/offset；`tilingKey` 当前不用于 Device 路由 | `DynamicBlockMxQuantTilingData`; `SetTilingData` |
| 调用顺序 | 单 K1；每 chunk `CopyIn→Compute→CopyOut` | OCP/DDR scale 必须先于同 half 的 Y，scale2 必须在 EnQue 前交织 | `Process`; `Compute` |
| workspace/内存 | `UB=103552rc+V`，operator workspace 0 | 4 Queue depth=2、reciprocal/index TBuf；ACLNN 使用第一段 workspace | `SplitCore`; `Init`; `DoTiling` |
| 同步 | Queue 协议；尾列 clear 后 `V_MTE2` event | 不得让 MTE2 覆盖尚未完成的 Vector clear | `CopyIn` |
| 精度与尾块 | OCP exponent、DDR mantissa、FP16/BF16 分支、FP4 pack | round trait、异常 mask、padding zero、有效 GM extent | `ComputeOcp`; `ComputeDdr`; `ComputeY*`; `CopyOut` |
| 失败边界 | Host 组合校验，无 Device fallback | 950-only、非空、rank2/3、FP4偶数N、UB至少一 tile | `GetAttr`; `CheckDtype`; `CheckShape`; `SplitCore` |
