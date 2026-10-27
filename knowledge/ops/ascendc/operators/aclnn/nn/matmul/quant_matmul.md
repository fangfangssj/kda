---
type: Operator
title: "QuantMatmul A2/A3 ACLNN 算子知识"
description: "说明 aclnnQuantMatmulV5 在 A2/A3 上的接口语义、Host 参数生成、QuantBatchMatmulV3/V4 Route 选择与 Cube/MIX_CV Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_api/quant_matmul_common_check.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/op_host/op_api/quant_matmul_v4.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_kernel_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_cube_basic.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16_opt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_basic.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_init_output.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_tiling_registry.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_msd_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_perblock_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_pergroup_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_msd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_perblock.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_pergroup.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_host/config/ascend910b/quant_batch_matmul_v3_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_host/config/ascend910_93/quant_batch_matmul_v3_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/config/ascend910b/quant_batch_matmul_v4_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/config/ascend910_93/quant_batch_matmul_v4_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:39:30Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# QuantMatmul A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnQuantMatmulV5GetWorkspaceSize` + `aclnnQuantMatmulV5`，公开 ACLNN 最新版本 |
| 计算语义 | 量化矩阵乘；按模式执行 `x1*x2` 后应用 per-tensor/per-channel/per-token/per-group scale，可选 offset/bias，并转换为目标 dtype |
| 输入输出 | `x1/x2` 为 2～6 维 ND（`x2` 还可为 FRACTAL_NZ）；A2/A3 公开组合含 INT8、INT4，输出 FP16/BF16/INT8/INT32；具体组合由 V3/V4 binary 限定 |
| 执行类型 | V3 整数 scale 路线为 AIC_CUBE；V3 浮点 scale/per-token 路线为 MIX_CV；V4 MSD、per-block、per-group 路线为 MIX_CV |
| Kernel 路线 | R-001 V3 Cube；R-002 V3 Cube+AIV 后处理；R-003 V4 A8W4 MSD；R-004 V4 A8W8 G-B；R-005 V4 A4W4 非对称 K-G |
| 核心资源 | GM、system/user workspace、L1、L0A/L0B/L0C、UB/寄存器；MIX 路线以 workspace ping-pong 或 L0C→UB 交接 |
| 来源闭包 | source_closed：V5 ACLNN 校验/预处理→V3/V4 L0 节点→A2/A3 Tiling/POD→可达 Cube/MIX body→binary/CMake 在固定 Commit 内闭合；Contiguous/Transpose/TransData/ViewCopy 与通用 Matmul 为标注的框架/库边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `A=op(x1)`、`B=op(x2)`，`M/N/K` 为末两维矩阵轴，前导轴按广播得到 batch。核心累加为 `acc=A@B`。T-C/T-T 路线按列 scale，K-C/K-T 再乘按行 `x1Scale`；G-B/K-G 按 `groupSizeM/N/K` 的块 scale。A4W4 非对称 K-G 还计算 `x2Offset` 修正。bias 在 Cube 或 AIV epilogue 中加入；浮点后处理以 FP32 中间执行，再以 `CAST_RINT` 转 FP16/BF16。INT32 输出保留 Cube 累加结果。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| x1 | 2～6D ND | INT8/INT4（INT32 可作为 INT4 pack） | 非连续仅支持末两轴转置识别，否则先 Contiguous；`transposeX1` 参与 M/K 解释 |
| x2 | 2～6D ND，或 4～8D FRACTAL_NZ storage | INT8/INT4（INT32 pack） | `transposeX2` 参与 K/N；NZ 仅作为本入口的输入格式，不等同于 WeightNz API |
| x1Scale | 可空；标量、`[m]` 或块网格 | FP32 | K-C/K-G/G-B 的左矩阵 scale |
| x2Scale | 必选；`[1]`、`[n]` 或块网格 | UINT64/INT64/FP32/BF16 | V5 将 INT64 视图改为 UINT64；形状决定 per-tensor/per-channel/per-group |
| yScale/x2Offset/yOffset/bias | 可选 ND | 依模式 | A2/A3 不支持 `x1Offset`；A4W4 K-G 使用 FP16 `x2Offset`；bias 可在 Cube 或 AIV 融合 |
| groupSize | 标量编码 | int64 | 普通路径由 checker 解码；A8W4 INT per-channel 为 0、per-group 为 256 |
| out | 广播 batch + `[M,N]`，ND | FP16/BF16/INT8/INT32 | 形状和 dtype 必须命中公开表及 binary；不支持空 Tensor |

### 入口与版本

固定仓中 V1～V4 文档均提示迁移到 `aclnnQuantMatmulV5`；V5 产品表明确支持 Atlas A2/A3，因此本卡只写 V5，不拆写旧版本。V5 对普通量化调用 `l0op::QuantBatchMatmulV3`；仅 `isA8W4 || IsA8W8Perblock || IsA4W4PergroupNonSymmetric` 时调用 `l0op::QuantBatchMatmulV4`。WeightNz 是独立公开语义变体，不在本卡范围。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先创建 executor，检查必选指针、维数、dtype/format、M/N/K、batch 广播、scale/offset/bias shape 与 groupSize。`TensorContiguousProcess` 对末两轴转置视图仅交换 view shape 并翻转 transpose 属性，其余非连续输入加入 `Contiguous`；INT32 pack 通过 `CreateView` 改为 INT4；INT64 scale 通过 dtype view 变为 UINT64。310P 专属 Transpose/TransData 分支不属于 A2/A3。计算结果经必要的 `Reshape` 和 `ViewCopy` 写入用户 out。

### 参数、Workspace 与生命周期

ACLNN workspace 是 executor 内所有前处理、核心节点和 ViewCopy workspace 的总和。V3 从平台 library workspace 起算；需要 UB 后处理时再加 `2*usedCoreNum*baseM*baseN*sizeof(int32_t)`，atomic-clean 无 L0C2OUT 时再加 `aicNum*32B`。V4 MSD 为 system workspace 加 `parallNum*usedCoreNum*baseM*baseN*sizeof(int16_t)`；per-block/per-group 用双份 `baseM*baseN*sizeof(int32_t)` 核间交接区。调用方必须让输入、输出、workspace 和 executor 存活至同一 stream 完成。

### 启动图与失败传播

```text
aclnnQuantMatmulV5GetWorkspaceSize
  -> checker + view/Contiguous
  -> if A8W4/A8W8-perblock/A4W4-nonsym: QuantBatchMatmulV4[MIX_CV]
     else: QuantBatchMatmulV3[AIC_CUBE 或 MIX_CV]
  -> optional Reshape -> ViewCopy
aclnnQuantMatmulV5 -> CommonOpExecutorRun(workspace, stream)
```

核心节点均在 executor 的同一 stream 上按加入顺序执行；任一检查、InferShape、Tiling、launcher 或 ViewCopy 失败即返回错误，后继节点不提交。单 Kernel MIX 内部使用 CrossCore flag/`SyncAll`，不能仅依赖 Host 顺序。

## Route 与参数设计

### 分核与资源计算

`M/N/K` 从 transpose 后末两轴获得，batch 最多展开为四级 `batchA1..A4/batchB1..B4/batchC1..C4`。V3 Tiling 先选 template，再产生 `baseM/baseN/baseK`、`singleCoreM/N/K`、`stepKa/stepKb`、`depthA1/depthB1`、`dbL0C`、L2 tile 与 UB tile；`blockDim=usedCoreNum`。V4 MSD/per-block/per-group分别产生专属 POD；MIX 需 AIC:AIV=1:2，且使用独占调度以保护全核同步。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-003 A8W4 MSD | V4 `(TRANS,K_C/K_G,0,weightNz,MSD_BASIS)` | V5 公开输入为 INT8×INT32-pack，前处理将 x2 view 为 INT4；A2/A3 V4 registry 首试 MSD；K<=29576，per-channel group=0 或 per-group group=256 | 参数不满足则 Tiling 失败，不回落到 V3 |
| 2 | R-004 A8W8 G-B | V4 `(B_TRANS,MX,0,0,PERBLOCK_BASIS)` | INT8×INT8，x1/x2Scale 均 FP32 且 rank 与输入一致，输出 BF16；MSD 不 capable 后命中 per-block | 进入下一 V4 template 或失败 |
| 3 | R-005 A4W4 K-G | V4 `(NOT_TRANS,MX,0,0,PERGROUP_BASIS)` | INT4×INT4、FP32 双 scale、FP16 x2Offset、非对称 per-group checker 成立 | 失败 |
| 4 | R-001 V3 Cube | V3 `(TRANS,TBE/BASIC,not-per-token,none/atomic-clean)` | 非 V4 条件；UINT64/INT64 scale 或 INT32 输出，且无需 Vector epilogue | R-002 |
| 5 | R-002 V3 MIX | V3 `(TRANS,TBE/OPT/BASIC,per-token?,none)` | 非 V4 条件；FP32/BF16 scale、per-token 或 BF16/float bias 需 UB epilogue | Tiling/编译 key 不支持则失败 |

```text
if isA8W4 || isA8W8Perblock || isA4W4PergroupNonSymmetric:
    910B/910_93 registry: MSD -> per-block -> per-group
else:
    V3 registry: Basic(priority 0) -> generic TBE(priority 2)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 V3 Cube | TRANS×TBE/BASIC×not-per-token | A2/A3 V3 binary 的 INT8/INT4、integer scale 或 INT32 output 组合 | 单 AIC_CUBE | `BmmDequant` 或 `QuantBatchMatmulV3BaseKernel` | GM→L1→L0A/B→Mmad→L0C→Fixpipe/GM；split-K 可先清零 | M/N/K 与 batch tail；无运行 fallback |
| R-002 V3 MIX | TRANS×TBE/OPT/BASIC×per-token flag | V3 binary 的 FP32/BF16 scale 或 per-token组合，out!=INT32 | 单 MIX_CV | `BmmDequantBf16*` / `BmmDequantPertoken*` | AIC L0C 写共享 VECIN/或 workspace，AIV UB FP32 dequant；AIC:AIV flag | UB tile/tail；不适用回 R-001 由 Host key 决定 |
| R-003 V4 MSD | K_C/K_G + MSD | 公开 INT8×INT32-pack；核心 view 为 INT4，ND/NZ，FP16/BF16 | 单 MIX_CV | `QuantBatchMatmulV4MsdPre` + `QuantBatchMatmulV4Msd` | AIV 将 INT8 activation 拆为高/低 INT4；`SyncAll`；Cube 结果写 workspace，AIV合成/缩放 | K<=29576；group 0/256；失败终止 |
| R-004 V4 per-block | MX + PERBLOCK | INT8×INT8 G-B、双 FP32 scale、BF16 | 单 MIX_CV | `QuantBatchMatmulV4Perblock` | L1 A/B ping-pong；L0C→workspace ping/pong→AIV 192KB UB | group 1×128×128；尾 M/N mask |
| R-005 V4 per-group | MX + PERGROUP | INT4×INT4 K-G 非对称 | 单 MIX_CV | `QuantBatchMatmulV4Pergroup` | L1 int4 load、int32 workspace、AIV x2Offset 修正与 scale | K-group 与 M/N tail；失败终止 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| V4 分流 | 三个 V5 布尔条件之一为真 | V4 有专用量化 body/POD | R-003～005 | V3 或参数错误 | `aclnnQuantMatmulGetWorkspaceSizeCommonProcess` |
| MSD K/group | `K<=29576`；K-G 要 `K%256==0`、group=256 | MSD activation 拆分和 workspace 上界 | R-003 | Tiling failure | `QuantBatchMatmulV4MsdTiling::IsCapable` |
| MIX 核比 | `aivNum==2*aicNum` | 每个 AIC 的后处理由两个 AIV 分摊 | R-002～005 | Tiling failure | `CheckCoreNum` |
| V3 atomic-clean | split-K/输出覆盖要求 `NeedAtomiClean()` | 防止跨核累加读旧值 | R-001 | 加清零及 workspace；无安全条件则失败 | `GetTilingKey/GetWorkspaceSize` |
| L1/L0 DB | tile 字节能装入平台 L1/L0C | 决定 depth/step/dbL0C | 全部 | 降为单 buffer 或其它 tile | `CalcL1Tiling` / `TCubeTiling` |
| per-block UB | 固定 192KB 内容纳 accum/scale/bias/output | AIV 后处理不溢出 UB | R-004 | Tiling failure | `QuantBatchMatmulV4Perblock::Init` |

### POD/ABI

所有本仓自定义 POD 均 `#pragma pack(8)`/`alignas(8)`；`TCubeTiling` 的内部 `sizeof/offsetof` 来自外部 CANN 生成头，固定仓不可见。下表逐字段列出固定仓可见 ABI；含 `TCubeTiling` 的后继字段只能给出声明顺序，不能推测绝对 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `QuantBatchMatmulV3Params` | `batchA` | uint32 | offset 0 | batch | V3 Tiling | A batch offset | >=1 |
| 同上 | `batchB` | uint32 | offset 4 | batch | V3 Tiling | B batch offset | >=1 |
| 同上 | `batchC` | uint32 | offset 8 | batch | V3 Tiling | output batch/core | >=1 |
| 同上 | `batchA1` | uint32 | offset 12 | batch axis | broadcast analysis | A offset | 缺省0 |
| 同上 | `batchA2` | uint32 | offset 16 | batch axis | broadcast analysis | A offset | 缺省0 |
| 同上 | `batchA3` | uint32 | offset 20 | batch axis | broadcast analysis | A offset | 缺省0 |
| 同上 | `batchA4` | uint32 | offset 24 | batch axis | broadcast analysis | A offset | 缺省0 |
| 同上 | `batchB1` | uint32 | offset 28 | batch axis | broadcast analysis | B offset | 缺省0 |
| 同上 | `batchB2` | uint32 | offset 32 | batch axis | broadcast analysis | B offset | 缺省0 |
| 同上 | `batchB3` | uint32 | offset 36 | batch axis | broadcast analysis | B offset | 缺省0 |
| 同上 | `batchB4` | uint32 | offset 40 | batch axis | broadcast analysis | B offset | 缺省0 |
| 同上 | `batchC1` | uint32 | offset 44 | batch axis | broadcast analysis | output offset | 缺省0 |
| 同上 | `batchC2` | uint32 | offset 48 | batch axis | broadcast analysis | output offset | 缺省0 |
| 同上 | `batchC3` | uint32 | offset 52 | batch axis | broadcast analysis | output offset | 缺省0 |
| 同上 | `batchC4` | uint32 | offset 56 | batch axis | broadcast analysis | output offset | 缺省0 |
| 同上 | `singleCoreBatch` | uint32 | offset 60 | batch/core | core split | block scheduler | >=1 |
| 同上 | `isPerTensor` | uint32 | offset 64 | bool | scale analysis | epilogue | 0/1 |
| 同上 | `isPertoken` | uint32 | offset 68 | bool | x1Scale analysis | AIV epilogue | 0/1 |
| 同上 | `isDoubleScale` | uint32 | offset 72 | bool | scale analysis | epilogue | 0/1 |
| 同上 | `biasThreeDim` | uint32 | offset 76 | bool | bias shape analysis | bias offset | 0/1 |
| 同上 | `ubCalcM` | uint32 | offset 80 | element | UB tiling | AIV M tile | MIX有效 |
| 同上 | `ubCalcN` | uint32 | offset 84 | element | UB tiling | AIV N tile | MIX有效 |
| 同上 | `needUbBuffer` | uint32 | offset 88 | bool | route analysis | output path | 0/1 |
| 同上 | `realSingleCoreM` | uint32 | offset 92 | element | core split | M tail | >0 |
| 同上 | `realSingleCoreN` | uint32 | offset 96 | element | core split | N tail | >0 |
| 同上 | `biasDtype` | uint32 | offset 100 | dtype enum | bias analysis | epilogue cast/load | route-dependent |
| 同上 | `ubSize` | uint32 | offset 104 | byte | platform/UB analysis | UB allocation | <=平台UB |
| 同上 | `isMClash` | uint32 | offset 108 | bool | L2 analysis | scheduler | 0/1 |
| 同上 | `isNClash` | uint32 | offset 112 | bool | L2 analysis | scheduler | 0/1 |
| 同上 | `groupSizeM` | uint32 | offset 116 | element | group decode | group epilogue | mode-dependent |
| 同上 | `groupSizeN` | uint32 | offset 120 | element | group decode | group epilogue | mode-dependent |
| 同上 | `groupSizeK` | uint32 | offset 124 | element | group decode | K-group loop | mode-dependent |
| `L2cacheTileParam` | `mTileCntL2` | uint32 | offset 0 | tile | L2 tiling | scheduler | >=1或inactive |
| 同上 | `nTileCntL2` | uint32 | offset 4 | tile | L2 tiling | scheduler | >=1或inactive |
| 同上 | `mTileBlock` | uint32 | offset 8 | block | L2 tiling | scheduler | route-dependent |
| 同上 | `nTileBlock` | uint32 | offset 12 | block | L2 tiling | scheduler | route-dependent |
| 同上 | `calOrder` | uint32 | offset 16 | enum | L2 tiling | traversal | supported enum |
| 同上 | `isBasicTiling` | uint32 | offset 20 | bool | template selector | traversal | 0/1 |
| `SlidingWindowParam` | `mTailTile` | uint32 | offset 0 | tile | adaptive tiling | M tail | >=0 |
| 同上 | `nTailTile` | uint32 | offset 4 | tile | adaptive tiling | N tail | >=0 |
| 同上 | `mBaseTailSplitCnt` | uint32 | offset 8 | count | adaptive tiling | M schedule | >=1 |
| 同上 | `nBaseTailSplitCnt` | uint32 | offset 12 | count | adaptive tiling | N schedule | >=1 |
| 同上 | `mTailMain` | uint32 | offset 16 | element | adaptive tiling | M tail | >=0 |
| 同上 | `nTailMain` | uint32 | offset 20 | element | adaptive tiling | N tail | >=0 |
| `QuantBatchMatmulV3TilingData` | `params` | nested/128B | offset 0 | mixed | V3 Tiling | V3 entry/body | 必选 |
| 同上 | `matmulTiling` | `TCubeTiling` | 紧随params；内部offset缺失 | mixed | matmul tiler | Matmul body | 外部CANN ABI |
| 同上 | `tileL2cacheTiling` | nested/24B | `TCubeTiling`之后 | mixed | L2 tiling | scheduler | route-dependent |
| 同上 | `adaptiveSlidingWin` | nested/24B | 结构尾 | mixed | adaptive tiling | scheduler/tail | route-dependent |
| `QuantBatchMatmulV4MsdTilingData` | `coreNum` | uint8 | offset 0 | core | MSD Tiling | core mapping | <=aicNum |
| 同上 | `vBaseM` | uint32 | offset 4（1～3为padding） | element | MSD Tiling | AIV tile | >0 |
| 同上 | `ubRestBytes` | uint32 | offset 8 | byte | UB analysis | temp allocation | <=UB |
| 同上 | `parallNum` | uint32 | offset 12 | count | MSD Tiling | WS ping-pong | >0 |
| 同上 | `ubCalSize` | uint32 | offset 16 | element | MSD Tiling | UB queues | >0 |
| 同上 | `mSize` | uint32 | offset 20 | element | shape analysis | bounds | >0 |
| 同上 | `kSize` | uint32 | offset 24 | element | shape analysis | bounds | 1..29576 |
| 同上 | `nSize` | uint32 | offset 28 | element | shape analysis | bounds | >0 |
| 同上 | `groupSize` | uint32 | offset 32 | element | group decode | scale loop | 0或256 |
| 同上 | `matmulTiling` | `TCubeTiling` | 结构尾；绝对offset缺失 | mixed | matmul tiler | MSD Matmul | 外部CANN ABI |
| `QuantBatchMatmulV4PerblockTilingData` | `matmulTiling` | `TCubeTiling` | offset 0；内部大小缺失 | mixed | matmul tiler | perblock Matmul | 外部CANN ABI |
| 同上 | `tileL2cacheTiling` | nested/24B | `TCubeTiling`之后 | mixed | L2 tiling | scheduler | route-dependent |
| 同上 | `groupSizeM` | uint32 | nested字段之后 | element | group decode | block-scale indexing | G-B=1 |
| 同上 | `groupSizeN` | uint32 | 声明序 | element | group decode | block-scale indexing | G-B=128 |
| 同上 | `groupSizeK` | uint32 | 声明序 | element | group decode | K-loop | G-B=128 |
| 同上 | `ubCalcM` | uint32 | 声明序 | element | UB tiling | AIV M tile | >0 |
| 同上 | `ubCalcN` | uint32 | 声明序 | element | UB tiling | AIV N tile | >0 |
| 同上 | `transA` | bool/8bit | 结构尾前1B | bool | attrs | A layout | checker范围 |
| 同上 | `transB` | bool/8bit | 结构尾1B | bool | attrs | B layout | checker范围 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | V3 key | V3 generic/basic tiling | V3 TilingData | `quant_batch_matmul_v3` | base/cube/basic/dequant body | V3 A2/A3 binary + kernel CMake |
| R-002 | V3 MIX keys | V3 generic/basic tiling | V3 TilingData | 同 entry | bf16/per-token/opt body | 同上 |
| R-003 | V4 MSD key | `QuantBatchMatmulV4MsdTiling` | MSD TilingData | `quant_batch_matmul_v4` | `quant_batch_matmul_v4_msd.h` | V4 A2/A3 binary + CMake |
| R-004 | V4 PERBLOCK key | `QuantBatchMatmulV4PerblockTiling` | Perblock TilingData | 同 entry | `quant_batch_matmul_v4_perblock.h` | 同上 |
| R-005 | V4 PERGROUP key | `QuantBatchMatmulV4PergroupTiling` | V3-compatible TilingData | 同 entry | `quant_batch_matmul_v4_pergroup.h` | 同上 |

## Kernel 实现

### 公共执行流程

核心 entry 首先拒绝空 workspace，设置 system workspace 并取得 user workspace，再按 tiling key 实例化 body。Cube 路线均执行 `GM A/B→L1→L0A/L0B→Mmad(int32/float accum)→L0C`；纯 Cube 由 Fixpipe/Matmul epilogue应用 scale/bias并写 GM，MIX 则把 L0C片写入共享 VECIN 或 user workspace，由 AIV 在 UB/寄存器完成 scale、offset、bias 和 cast。MIX 的 producer/consumer 通过明确 CrossCore flag 或 `SyncAll<false>` 交接。

### Route R-001：V3 Cube

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `aclnn_quant_matmul_v5.cpp` | common process | 三个 V4 条件均 false | `QuantBatchMatmulV3` |
| Tiling | `quant_batch_matmul_v3_tiling.cpp` | `GetTilingKey` | no per-token；integer scale/INT32 output | TBE/BASIC Cube key |
| Router | `quant_batch_matmul_v3.cpp` | entry | TRANS/TEMPLATE/PERTOKEN/OPTIONATTR | `BmmDequant` 或 BaseKernel |

#### Device 数据流

##### 函数级调用链

```text
quant_batch_matmul_v3 -> GET_TILING_DATA -> BmmDequant/BaseKernel::Init -> Process -> Matmul Iterate/GetTensorC
```

##### 源码等价伪码

```text
for batch,tileM,tileN:
  load A/B GM->L1->L0A/L0B; accumulate K tile into L0C
  apply integer dequant scale and optional bias in Fixpipe
  store valid M/N tail to GM; split-K key先清零再累加
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM A/B/Y | batch/tile offset | logical tensor | ND/NZ | caller | Cube/Fixpipe | caller/stream | 16/32B tail guard |
| L1 A/B | tile offsets | `depthA1*A_tile + depthB1*B_tile` | fractal | MTE2 | L0 load | core/tile | depth/step reuse |
| L0A/B/C | Matmul internal | baseM*baseK, baseK*baseN, `dbL0C*baseM*baseN*sizeof(acc)` | Cube | LoadData/Mmad | Mmad/Fixpipe | core/tile | db optional |
| UB | N/A | N/A | N/A | N/A | N/A | N/A | pure Cube |

##### 分层内存布局

```mermaid
flowchart LR
  A[GM A/B] --> L1[L1 A1/B1] --> L0[L0A/L0B→Mmad→L0C] --> F[Fixpipe scale/bias] --> Y[GM Y]
```

##### 字节账、交接边与复用顺序

```text
workspace = library_workspace + optional atomic_clean 32B*aicNum
L1 = depthA1*A_tile_bytes + depthB1*B_tile_bytes
L0C = dbL0C*baseM*baseN*sizeof(acc)
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| L0C→Y | Mmad | base tile | int32/float L0C | Fixpipe | Cube pipeline | core/tile | valid tail only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cube pipe | Matmul internal events | MTE/L0 | Mmad/Fixpipe | L1/L0 buffer | K 累加 int32/float；M/N/K tail由 TCubeTiling |
| split-K | scheduleMode/atomic-clean | clear | accum | output | 仅 key 要求时启用 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `quant_batch_matmul_v3_tiling.cpp` | `GetTilingKey/PostTiling` | key、blockDim、workspace |
| Kernel entry/router | `quant_batch_matmul_v3.cpp` | `quant_batch_matmul_v3` | key→Cube body |
| Device body/helper | 同目录 base/cube headers | `BmmDequant`/`QuantBatchMatmulV3BaseKernel` | Cube层级与Fixpipe |

### Route R-002：V3 MIX_CV 浮点后处理

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | V3 Tiling | `isPertoken/isBf16Opt/isUbQuant` | 浮点 scale、per-token 或浮点 bias | MIX key |
| Router | V3 entry | TBE/OPT/BASIC | key 与 dtype compile guard | 相应 bf16/per-token body |

#### Device 数据流

##### 函数级调用链

```text
entry -> BmmDequantBf16*/BmmDequantPertoken*::Init -> Cube Matmul -> AIV Dequant -> GM
```

##### 源码等价伪码

```text
AIC: Mmad(A,B[,int32 bias]) -> L0C/VECIN or workspace -> set C2V
AIV: wait C2V -> FP32(acc)*x2Scale[*x1Scale] + optional bias -> CAST_RINT -> GM -> set V2C
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| user workspace | per-core DB slice | `2*usedCoreNum*baseM*baseN*sizeof(int32)` when basic UB path | int32 ND-align | AIC | AIV | executor/kernel | ping-pong |
| UB accum/scale/output | core local | `ubCalcM*ubCalcN*acc + scale/bias + output` | int32/FP32/out | MTE/Fixpipe | Vector | AIV/tile | aligned tail |
| L1/L0 | TCubeTiling | same hierarchy as R-001 | Cube | MTE/Mmad | Mmad/Fixpipe | AIC/tile | DB by tiling |

##### 分层内存布局

```mermaid
flowchart LR
  GM[GM A/B] --> L1[L1] --> C[L0A/B→Mmad→L0C]
  C --> X[VECIN/workspace ping-pong]
  X --> U[UB→FP32 regs: scale/bias/cast] --> Y[GM Y]
```

##### 字节账、交接边与复用顺序

```text
workspace_total = library_workspace + 2*usedCoreNum*baseM*baseN*sizeof(int32)（需要时）
UB_total = accum_tile + scale(baseN) + perToken(baseM) + bias(baseN) + output_ping/pong
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| Cube→Vector | Mmad/GetTensorC | per-core base tile | int32/float ND-align | AIV epilogue | CrossCore C2V/V2C | one launch | AIC不得覆盖未消费 ping/pong |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CV handoff | CrossCore flag | AIC | AIV | accum tile | 每轮成对；末轮 AIC补 wait |
| Vector | UB event + mask | MTE2/V | V/MTE3 | scale/output UB | FP32 mul/add，CAST_RINT；mask保护 N tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `quant_batch_matmul_v3_basic_tiling.cpp` | `GetTilingKey/GetUbDequantExtreSpace` | MIX key/workspace |
| Kernel entry/router | `quant_batch_matmul_v3.cpp` | INVOKE macros | key→后处理 specialization |
| Device body/helper | `quant_batch_matmul_v3_pertoken.h` 等 | `BmmDequantPertoken*` | AIC/AIV交接、UB后处理 |

### Route R-003：V4 A8W4 MSD

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | V5 API | `isA8W4Int` | INT8×INT32-pack；随后 x2 view 为 INT4 | V4 |
| Tiling | `quant_batch_matmul_v4_msd_tiling.cpp` | `IsCapable/GetTilingKey` | K/group约束 | MSD key |

#### Device 数据流

##### 函数级调用链

```text
entry -> AIV QuantBatchMatmulV4MsdPre -> SyncAll -> QuantBatchMatmulV4Msd::Process -> Cube/AIV ping-pong
```

##### 源码等价伪码

```text
AIV preprocess: INT8 row -> high/low INT4 planes in workspace
SyncAll
AIC: two/folded INT4 matmuls -> FP16 partial workspace -> C2V
AIV: reconstruct FP32, apply x1Scale/x2Scale/yOffset/bias -> cast -> GM -> V2C
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| activation staging | userWS start | aligned A high/low planes | int4 packed | AIV pre | Cube | launch | after SyncAll |
| MM result | per-core parallel slice | `parallNum*core*baseM*baseN*sizeof(int16)` | FP16 | AIC | AIV | launch | V2C/C2V |
| UB | `ubCalSize` queues + `ubRestBytes` temp | explicit POD sizes | half/float/out | MTE | Vector | AIV/tile | single queues |

##### 分层内存布局

```mermaid
flowchart LR
  A[GM INT8 A] --> P[AIV UB split] --> W[workspace INT4 planes]
  W --> C[AIC L1/L0/Mmad] --> R[workspace FP16 partials] --> V[AIV FP32 combine/scale] --> Y[GM Y]
```

##### 字节账、交接边与复用顺序

```text
workspace = SYS + aligned(preprocess planes) + parallNum*usedCoreNum*baseM*baseN*sizeof(int16)
UB = input/output queues from ubCalSize + ubRestBytes temporary
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| pre→Cube | AIV MSDPre | aligned userWS | packed INT4 | AIC Matmul | `SyncAll<false>` | launch | scheduleMode=1 |
| Cube→AIV | GetTensorC | workSpaceOffset/base tile | half partial | ComputeDequant | C2V/V2C flags | ping-pong | wait before overwrite |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| preprocessing | `SyncAll` | AIV | AIC | activation planes | K aligned/padded |
| main | flag 2 PIPE_FIX/MTE2 | AIC/AIV | peer | result slice | FP32 combine; M/N tail by cur sizes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `quant_batch_matmul_v4_msd_tiling.cpp` | MSD Tiling | key/workspace/schedule |
| Kernel entry/router | `quant_batch_matmul_v4.cpp` | `QBMM_MSD_IMPL_CLASS` | pre+SyncAll+main |
| Device body/helper | `quant_batch_matmul_v4_msd.h` | MSDPre/MSD | split、Cube、dequant |

### Route R-004：V4 A8W8 G-B per-block

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | V5 | `IsA8W8Perblock` | INT8×INT8、双 FP32 block scales | V4 |
| Tiling | perblock | key producer | transB=true、G-B grid | PERBLOCK |

#### Device 数据流

##### 函数级调用链

```text
entry -> QuantBatchMatmulV4Perblock::Init -> Process -> MMCompute + VectorProcess
```

##### 源码等价伪码

```text
AIC load A/B to L1, Mmad per K-group, GetTensorC(workspace ping/pong), set C2V
AIV wait, load int32 tile, multiply x1/x2 block scales, accumulate FP32, add bias, cast/store, set V2C
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace ping/pong | per core | `2*baseM*baseN*sizeof(int32)` | int32 ND | AIC | AIV | launch | flags |
| L1 | A/B ping/pong | <= platform L1 | int8 NZ | MTE2 | Matmul | AIC | double buffer |
| UB | fixed arena | 192KB | int32/FP32/BF16 | AIV | AIV/MTE3 | tile | offsets in Init |

##### 分层内存布局

```mermaid
flowchart LR
  GM[GM A/B] --> L1[L1 ping/pong] --> C[L0 Mmad] --> W[workspace int32 DB] --> U[UB FP32 block-scale] --> Y[GM BF16]
```

##### 字节账、交接边与复用顺序

```text
workspace_user = 2*usedCoreNum*baseM*baseN*sizeof(int32)
UB_total = 192KB fixed arena
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| WS ping/pong | AIC GetTensorC | offsetWorkspaceC/Pong | int32 ND | AIV CopyInMMOut | C2V/V2C ping/pong flags | launch | pair wait/set |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| handoff | CrossCore ping/pong | AIC/AIV | peer | workspace DB | int32→FP32，block scale；tail sizes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `quant_batch_matmul_v4_perblock_tiling.cpp` | perblock Tiling | group/UB/key |
| Kernel entry/router | `quant_batch_matmul_v4.cpp` | PERBLOCK branch | exact template |
| Device body/helper | `quant_batch_matmul_v4_perblock.h` | `QuantBatchMatmulV4Perblock` | 192KB UB和DB协议 |

### Route R-005：V4 A4W4 非对称 K-G

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | V5 checker | `IsA4W4PergroupNonSymmetric` | INT4双输入、FP32 scales、x2Offset | V4 |
| Tiling | pergroup | `GetTilingKey` | not-trans, group grid | PERGROUP |

#### Device 数据流

##### 函数级调用链

```text
entry -> QuantBatchMatmulV4Pergroup::Init -> Process -> CubeProcess + VectorProcess
```

##### 源码等价伪码

```text
AIC: unpack/load INT4 A/B to L1, Mmad each K-group -> int32 workspace -> C2V
AIV: convert, apply x1Scale*x2Scale and x2Offset correction, FP32 accumulate, add bias, cast/store -> V2C
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| L1 A/B | K-group ping/pong | groupSizeK*baseM/N packed | INT4/NZ | MTE | Mmad | AIC/group | DB |
| workspace | ping/pong per core | `2*baseM*baseN*sizeof(int32)` | int32 | AIC | AIV | launch | flags |
| UB | fixed arena offsets | <=192KB | int32/FP32/half/out | AIV | AIV/MTE3 | tile | includes row-sum/offset correction |

##### 分层内存布局

```mermaid
flowchart LR
  I[GM INT4] --> L[L1] --> C[L0 Mmad] --> W[workspace int32] --> U[UB scales+offset correction] --> O[GM out]
```

##### 字节账、交接边与复用顺序

```text
workspace_user = 2*usedCoreNum*baseM*baseN*sizeof(int32)
UB = mmOut + FP32 accum + scales + X1 row-sum + x2Offset correction + output
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| Cube→AIV | GetTensorC | C/Pong | int32 ND | VectorProcess | C2V/V2C flags | launch | ping/pong not overwritten |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| K-group | CrossCore flags | AIC | AIV | int32 partial | FP32 accumulate and offset correction；tail masks |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `quant_batch_matmul_v4_pergroup_tiling.cpp` | pergroup Tiling | fixed base/group/key |
| Kernel entry/router | `quant_batch_matmul_v4.cpp` | PERGROUP branch | exact template |
| Device body/helper | `quant_batch_matmul_v4_pergroup.h` | `QuantBatchMatmulV4Pergroup` | INT4/offset/FP32 accum |

## 算子特定优化

### Trick T-001：V5 以量化语义选择 V3/V4 内核而非按 API 版本硬绑定

#### 触发与源码依据

- 适用 Route：全部。
- 触发条件：A8W4、A8W8 per-block 或 A4W4 非对称 per-group。
- Golden 来源：[`aclnn_quant_matmul_v5.cpp:1139-1150`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp#L1139-L1150)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isA8W4 || isA8W8Perblock || isA4W4PergroupNonSymmetric) {
    matmulRet = l0op::QuantBatchMatmulV4(reformatedX1, reformatedX2, reformatedBias, reformatedX1Scale,
                                         castedScale, reformatedYScale, x1Offset, x2Offset, yOffset,
                                         nullptr, dtype, -1, transposeX1, transposeX2, groupSize, executor);
} else {
    matmulRet = l0op::QuantBatchMatmulV3(reformatedX1, reformatedX2, castedScale, x2Offset, reformatedBias,
                                         reformatedX1Scale, dtype, transposeX1, transposeX2, groupSizeReal, executor);
}
```

#### 机制与设计原因

公开 V5 保持一个 ABI，但把需要 activation/weight 双侧分组 scale、INT4 offset 修正或 MSD 展开的模式送入 V4，普通 T-C/T-T/K-C/K-T 复用 V3。迁移实现必须复刻此谓词，否则 POD、输入顺序和 kernel key 会错配。

#### 迁移边界

- 适用：V5 A2/A3 公布且 binary 列出的组合。
- 前提：checker、groupSize 解码、dtype view 与 L0 参数顺序一致。
- 失效：新模式不满足三谓词时必须走 V3；不得自行送 V4。
- 反例/失效证据：同一 V5 common process 的 `else` 明确把其余公开组合送入 `QuantBatchMatmulV3`。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把“V5”直接等同于 `QuantBatchMatmulV4` 会让普通模式读错输入和 POD。

### Trick T-002：MIX_CV 用双缓冲 workspace 和 CrossCore flag 覆盖 Cube/Vector 阶段

#### 触发与源码依据

- 适用 Route：R-004、R-005。
- 触发条件：V4 per-block/per-group MIX_CV。
- Golden 来源：[`quant_batch_matmul_v4_perblock.h:284-306`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_perblock.h#L284-L306)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
CrossCoreWaitFlag(wsPingFlag ? V2C_PING_FLAG : V2C_PONG_FLAG);
mmObj_.GetTensorC(mmOutGlobal_[wsPingFlag ? offsetWorkspaceC_ : offsetWorkspacePong_], 0);
CrossCoreSetFlag<CV_SYNC_FLAG, PIPE_FIX>(wsPingFlag ? C2V_PING_FLAG : C2V_PONG_FLAG);
```

#### 机制与设计原因

AIC 仅在 AIV 归还对应 ping/pong 后覆盖 workspace，写完通过 PIPE_FIX 通知 AIV；AIV 消费后以 PIPE_MTE2 归还。交替 slice 允许下一组 Cube 与上一组 Vector epilogue 重叠，同时保持可见性。

#### 迁移边界

- 适用：AIC:AIV=1:2 且 workspace 按双份 core tile 分配。
- 前提：flag ID、slice offset、scheduleMode 和轮次完全一致。
- 失效：单缓冲或核比变化需重做协议。
- 反例/失效证据：固定仓未提供删除该 flag 协议后仍可正确运行的可验证反例。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

仅按源码语句顺序读写 workspace、删去 flag，会造成跨核读写竞争。

## 模板与实现对比

| 对比维度 | R-001 V3 Cube | R-002 V3 MIX | R-003 MSD | R-004/R-005 block/group | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | integer scale/INT32 | float scale/per-token | A8W4 | A8W8 G-B/A4W4 K-G | V5 semantic predicate + tiling registry |
| 执行族/节点图 | AIC-only | MIX_CV | MIX_CV pre+main | MIX_CV | 是否需要 Vector量化处理 |
| M/N/K、tile 与 scheduler | TBE/basic | TBE/opt/basic | MSD vBaseM | 固定 group-aware tile | dtype/scale/group与资源 |
| workspace/L1/L0/UB | 少量/纯Cube | per-core accum DB | activation+FP16 partial | int32 partial DB | 跨族交接内容不同 |
| AIC/AIV 或 kernel 交接 | N/A | flag/VECIN或WS | SyncAll+flags | ping/pong flags | producer可见性 |
| 同步、精度与 tail | Fixpipe | FP32 UB epilogue | FP32重构 | FP32分组累加 | 数值语义 |
| fallback/失败边界 | key miss失败 | key miss失败 | constraint failure | constraint failure | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V5公开支持 | V5公开支持 | V5公开支持，见 `_950` 卡 |
| Host/Launcher | 共用 910B/910_93 V3/V4路径 | 同左 | DAV_3510 使用 arch35 selector/body |
| Kernel/资源 | V3 generic + V4 MSD/perblock/pergroup | 与 A2同族，binary分别接线 | CMCT/Blaze/RegBase 与新量化类型 |

## 限制与失败边界

- A2/A3 不支持空 Tensor；产品表不支持 Atlas 200I/500 A2。
- API checker 接受范围不等于平台 binary 实际覆盖；正式 Route 仅取 A2/A3 binary 列出的组合。
- V4 MSD 的 K、group、transpose/format，per-block/per-group 的 shape 与核比不满足时均失败，不存在隐式 V3 fallback。
- `TCubeTiling` 和框架 executor 的精确 ABI 常数在固定仓不可见；必须使用 CANN 9.1 匹配头，不得手写布局。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md` | 产品表/约束 | 最新版本、A2/A3支持 |
| 入口声明/约束 | `matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.h` | V5 declarations | 两段式 ABI |
| 入口实现 | `matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp` | common process | 校验、预处理、V3/V4分流 |
| 入口实现 | `matmul/quant_batch_matmul_v4/op_host/op_api/quant_matmul_common_check.cpp` | preprocess helpers | view/format/INT4处理 |
| Host/Launcher | `matmul/common/op_host/op_api/quant_matmul_v4.cpp` | `QuantBatchMatmulV4` | InferShape/launcher node |
| Host/Tiling | `matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp` | V3 Tiling | key/workspace/blockDim |
| Host/Tiling | `matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_tiling_registry.cpp` | registry | A2/A3模板顺序 |
| Kernel entry/body | `matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp` | V3 entry | Cube/MIX router |
| Kernel entry/body | `matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4.cpp` | V4 entry | MSD/perblock/pergroup router |
| Kernel entry/body | `matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_msd.h` | MSDPre/MSD | A8W4数据流 |
| Kernel entry/body | `matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_perblock.h` | perblock | G-B同步/UB |
| Kernel entry/body | `matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_pergroup.h` | pergroup | K-G offset修正 |
| 构建接线 | `matmul/quant_batch_matmul_v3/op_host/config/ascend910b/quant_batch_matmul_v3_binary.json`、`ascend910_93/quant_batch_matmul_v3_binary.json` | op_list | V3 A2/A3组合 |
| 构建接线 | `matmul/quant_batch_matmul_v4/op_host/config/ascend910b/quant_batch_matmul_v4_binary.json`、`ascend910_93/quant_batch_matmul_v4_binary.json` | op_list | V4 A2/A3组合 |
| 构建接线 | `matmul/quant_batch_matmul_v3/op_kernel/CMakeLists.txt`、`matmul/quant_batch_matmul_v4/op_kernel/CMakeLists.txt` | `add_kernel_sources` | V3/V4 generic与arch35接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | V3/V4各 POD 字段语义 | 8B pack、字段顺序、key匹配 | tiling data headers |
| 调用顺序 | checker→preprocess→V3/V4→ViewCopy | 三个 V4 分流谓词 | V5 common process |
| workspace/内存 | per-core DB、MSD staging公式 | offset、alignment、owner/lifetime | Tiling GetWorkspaceSize + body Init |
| 同步 | SyncAll、C2V/V2C ping-pong | flag ID、轮次、AIC:AIV核比 | MSD/perblock/pergroup body |
| 精度与尾块 | int32/float accum，FP32 epilogue，CAST_RINT | scale/bias/offset顺序与tail mask | Device body |
| 失败边界 | checker/Tiling/key miss均终止 | 不得跨 V3/V4 猜测 fallback | API/registry/entry |