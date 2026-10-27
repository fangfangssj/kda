---
type: Operator
title: "QuantMatmulAlltoAll A2 ACLNN 算子知识"
description: "固定提交中 aclnnQuantMatmulAlltoAllV2 在 A2/Ascend 910B 的公开语义、Host/Tiling、INT8 KC Kernel 与 AIV AlltoAll 全链路。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/docs/aclnnQuantMatmulAlltoAllV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_api/aclnn_quant_matmul_allto_all_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_api/aclnn_quant_matmul_allto_all_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_api/quant_matmul_allto_all_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_api/matmul_allto_all_util.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/arch22/matmul_allto_all_tiling_910b.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/matmul_allto_all_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/matmul_allto_all_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/matmul_allto_all.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/matmul_allto_all.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/quant_matmul.hpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/block_epilogue_dequant.hpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/matmul_allto_all_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/arch.hpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/matmul_allto_all_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:41:43Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# QuantMatmulAlltoAll A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnQuantMatmulAlltoAllV2GetWorkspaceSize` / `aclnnQuantMatmulAlltoAllV2`；公开 ACLNN V2 |
| 计算语义 | INT8 KC MatMul；AIV 做 per-token/per-channel 反量化与 bias，再按 rank 换位并 AlltoAll |
| 输入输出 | x1/x2 INT8 ND，scale F32，bias FP16/BF16/F32，y FP16/BF16；输出 `[rank*BS,H2/rank]` |
| 执行类型 | MIX_CV：`KERNEL_TASK_TYPE_MIX_AIC_1_2`；AIC Cube + 两个 AIV epilogue/搬运/通信节点 |
| Kernel 路线 | R-001：`m0=128,n0=256`；R-002：`m0=256,n0=128`；各 key 变体仍是独立实例 |
| 核心资源 | 16MiB system workspace + 双 stage INT32 workspace；A2 L1/L0/UB；200MiB HCCL window |
| 来源闭包 | `source_closed`：ACLNN→base/Inner→A2 Tiling/POD/key→MIX router→Cube/AIV→HCCL window→构建闭合于 `be33d95...2540` |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `x1:[BS,H1]`、`x2:[H1,H2]`（`transposeX2=true` 时接受物理 `[H2,H1]`），A2 KC 路径先得到 INT32 累加，再按 Kernel 的 `BlockEpilogue` 顺序计算 `cast(x1@x2,F32) * x2Scale[n] * x1Scale[m] + bias[n]`，最后 cast 为 `y` dtype。逻辑 MatMul 结果 `[BS,H2]` 被视作 `[BS,rankSize,H2/rankSize]`，换位为 `[rankSize,BS,H2/rankSize]` 后做 AlltoAll，输出为 `[rankSize*BS,H2/rankSize]`。

### 输入、输出与属性

| 参数 | shape / layout | A2 dtype | 实现约束 |
|---|---|---|---|
| `x1` | `[BS,H1]`, ND、连续 | INT8 | 必选；`BS>=1`，`1<=H1<=65535`；`transposeX1` 必须 false |
| `x2` | `[H1,H2]` 或转置物理形状，ND、连续 | INT8 | 必选；A2 不接受非连续 transpose view；`H2>=2` 且可被 `rankSize` 整除 |
| `x1Scale` | `[BS]` | FLOAT32 | 必选，per-token |
| `x2Scale` | `[H2]` | FLOAT32 | 必选，per-channel |
| `bias` | `[H2]` | FLOAT16 / BFLOAT16 / FLOAT32 | A2 Tiling 要求非空；可达组合为 FP16 bias→FP16 y，BF16/F32 bias→BF16 y |
| `y` | `[rankSize*BS,H2/rankSize]` | FLOAT16 / BFLOAT16 | 必选且非空；`BS*rankSize`、`H2` 不超过 INT32_MAX |
| `group` | 字符串 | — | 必选，长度 `(0,128)`；Tiling 由该 group 查询 rankSize/rankId |
| `commMode` | 字符串 | — | A2 只接受 `"aiv"` |
| 量化属性 | 标量/保留 tensor | — | `x1Mode=3`、`x2Mode=2`、`commQuantMode=0`；offset/commScale 不支持，commQuantDtype 固定 GE undefined |
| `alltoallAxes` | null 或 `[-1,-2]` | INT64 list | 其余取值拒绝 |

公开文档把 FLOAT32 bias/FP16 y 也列入 A2 表格，但固定提交 `CheckKCBias` 与 Tiling dtype 列表不接受该组合；本卡按可到达源码组合记录。

### 入口与版本

V2 头文件给出两段式 ACLNN；实现只转发 `QuantMatmulAlltoAllBaseGetWorkspaceSize(...,commMode,...)` 与 `QuantMatmulAlltoAllBase(...)`。GetWorkspaceSize 构造 executor、返回 workspace；执行段从 executor user handle 取通信引擎并在调用者 stream 上调用 Inner op。V1 无 `commMode`，不作为本卡入口。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckAndHandleParams` 检查必选/空 tensor、二维 shape、K 与输出 shape、scale/bias shape、dtype、private format、ND 转换、axes、`transposeX1=false`、group 与 `commMode`。A2 要求输入连续；950 专属 x2 transpose-view 重描述不启用。失败不进入 Tiling/Kernel。

### 参数、Workspace 与生命周期

base 将 `worldSize=-1` 交 Tiling 从 group 查询，将 y dtype 编码成 GE 0/1/27，调用 `aclnnInnerMatmulAlltoAllGetWorkspaceSize`，并保存 `"aiv"`。A2 workspace 为 `16 MiB + tileM0*blockDim*2*tileN0*4` 字节；两 route 均化简为 `16 MiB + 262144*blockDim`，后项是 INT32 Cube 双 stage。executor、临时 tensor、workspace 必须存活到执行段完成。

### 启动图与失败传播

```text
V2GetWorkspaceSize -> QuantMatmulAlltoAllBaseGetWorkspaceSize
  -> InnerGetWorkspaceSize -> MatmulAlltoAllTiling910B::DoTiling
V2(stream, executor) -> QuantMatmulAlltoAllBase
  -> Inner(stream) -> matmul_allto_all<key>
  -> MatmulAlltoAll::QuantCatlassMatmul -> SyncAll
```

执行段把 A2 server type 设为 MTE；inner 失败转换为 `ACLNN_ERR_INNER`。平台、HCCL rank/buffer、shape、dtype、资源或 key 失败都阻止 launch。

## Route 与参数设计

### 分核与资源计算

Tiling 仅在 `Ascend910B` 可用，rankSize 必须 2/4/8；HCCL buffer 查询成功时至少 200 MiB。`DoMmCommTiling` 按 rank 的有序 shape 区间表首个匹配项选 `(m0,pValue,ubMoveNum)`，无匹配默认 rank2/4 `(128,10,40*512B)`、rank8 `(128,10,80*512B)`。`m0>=128` 时 `k0=256`；`m0=128` 对应 `n0=256`，否则 `n0=128`。若窗口需求超过 `(200MiB-20MiB)/2 stages`，回退 `m0=128,n0=256` 并收缩 p。`blockDim=CalcTschBlockDim(aivNum,aicNum,aivNum)`，schedule mode 因 `SyncAll` 为 1。

### Route 选择

| 顺序 | Route | 选择条件 | Key 细分 | 不满足 |
|---:|---|---|---|---|
| 1 | R-001 `m0=128` | shape map/默认值选 128，或窗口容量回退 | `GET_TPL_TILING_KEY(transposeX2,1,biasDtype,SOC_ASCEND910B)` | R-002 或失败 |
| 2 | R-002 `m0=256` | shape map 选 256 且不触发窗口回退 | 同 key；body 由 `m0!=128` 选模板 | R-001 或失败 |

~~~text
Ascend910B 且 rankSize in {2,4,8}
  -> API/Tiling dtype、shape、bias、aiv 校验
  -> 有序 shape map 首个匹配或默认 m0=128
  -> HCCL 数据区容量不足：强制 m0=128,n0=256 并收缩 pValue -> R-001
  -> m0==128 -> R-001
  -> m0==256 -> R-002
  -> 其他值/任一校验失败 -> 失败，不 launch
~~~

### 模板/实现执行矩阵

| Route | A/B | accumulator | L1 tile | L0 tile | epilogue | 通信 |
|---|---|---|---|---|---|---|
| R-001 | INT8 RowMajor；B 随 transpose key | INT32 | A `128x512`、B `512x256`，2-stage | A `128x128`、B `128x256`、C `128x256` | `32x256` | AIV HCCL window，2-stage |
| R-002 | 同上 | INT32 | A `256x512`、B `512x128`，2-stage | A `256x128`、B `128x128`、C `256x128` | `64x128` | 同上 |

### 隐藏性能 Guard

- N 还受 `(200MiB-20MiB)/(2*sizeof(y)*128)` 限制；它可能早于接口 INT32 上限生效。
- `pValue` 决定一次通算批次的 M tile 数；窗口超限时收缩 p，保证双 stage 留在 180MiB 数据区。
- `ubMoveNum` 的表值乘 512B，AIV 再减半做 ping-pong并按 `n0` 对齐。
- 两 route 的 L0C 都为 128KiB；R-001 偏 N，R-002 偏 M，离散表负责通算平衡。

### POD/ABI

| 结构 | 字段 | 类型 | offset/size/alignment | 单位 | producer→consumer | 边界 |
|---|---|---|---|---|---|---|
| `MatmulAlltoAllInfo` | `M` | uint32 | 0/4/align4 | element | Tiling→MM/comm | `M>0` |
| 同上 | `K` | uint32 | 4/4/align4 | element | Tiling→MM | `1..65535` |
| 同上 | `N` | uint32 | 8/4/align4 | element | Tiling→rank slice | `N%worldSize=0` |
| 同上 | `worldSize` | uint32 | 12/4/align4 | rank | HCCL query→kernel | 2/4/8 |
| `CoCTiling` | `m0` | int32 | 0/4/align4 | row | route table→template | 128/256 |
| 同上 | `k0` | int32 | 4/4/align4 | element | Tiling→MM | 256 |
| 同上 | `n0` | int32 | 8/4/align4 | column | Tiling→MM/AIV | 256/128 |
| 同上 | `pValue` | int32 | 12/4/align4 | tile | Tiling→scheduler | 容量收缩后正值 |
| 同上 | `ubMoveNum` | int32 | 16/4/align4 | byte | Tiling→`CommBase` | 512B 倍数 |

`sizeof(MatmulAlltoAllInfo)=16`、`sizeof(CoCTiling)=20`。外层还嵌入 SDK `Mc2InitTiling`、`Mc2CcTiling`；其定义来自外部 `kernel_tiling`，固定仓无法给出完整 outer `offsetof/sizeof`，故不推测。

### Route 源码闭包

| Route | selector | POD/key | entry | body/依赖 | 构建 |
|---|---|---|---|---|---|
| R-001 | `DoMmCommTiling/CalTilingParam` | A2 tiling/key headers | `matmul_allto_all.cpp` | `QuantCatlassMatmul`→`QuantMatmul`→epilogue→`CommBase` | op_def/CMake |
| R-002 | 同一 selector，`m0=256` | 同上 | 同上 | 同上，tile 参数改变 | 同上 |

## Kernel 实现

### 公共执行流程

entry 以 `MIX_AIC_1_2` 解包 Tiling。`Init` 获取 HCCL context/window、rankId，把 peer window 作为反量化输出目标并绑定 workspace；INT8 `Process` 调 `QuantCatlassMatmul()`，最后 `SyncAll<false>`。AIC 生产 INT32 tile，AIV 反量化写 peer window，再按批跨 rank 同步并搬到 y。

### Route R-001：128×256 通算流水

#### 适用条件

##### 选择证据

Host 令 `coCTiling.m0==128`；entry key 匹配 x2 transpose 与 bias dtype，`QuantCatlassMatmul` 的 128 分支实例化 L1 `<128,256,512>`、L0 `<128,256,128>`、epilogue `<32,256>`。

#### Device 数据流

##### 函数级调用链

```text
matmul_allto_all -> MatmulAlltoAll::Init/Process
  -> QuantCatlassMatmul -> QuantMatmul<128x256>::operator()
    AIC: MmadAtlasA2PreloadAsyncWithCallback -> workspace[int32]
    AIV: BlockEpilogue -> peer window[D]
    AIV task core: CrossRankSync -> Gm2Gm -> y -> CrossRankSync
```

##### 源码等价伪码

```text
for group of pValue M-tiles:
  AIC owner computes tile into workspace[stage] and sets FIX flag
  AIV waits; INT32->F32; multiply x2Scale; multiply broadcast x1Scale
  add bias; cast D; write target-rank slice to peer window[stage]
  ranks rendezvous; copy local received slices to y; rendezvous; reuse stage
```

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲 | offset/extent | 数据流 | 生命周期/对齐 |
|---|---|---|---|
| workspace | row-major `{128*blockDim*2,256}` INT32 | AIC Fixpipe→AIV GM2UB | 双 stage，`262144*blockDim` B |
| L1/L0 | L1 A `128x512`, B `512x256`; L0 A `128x128`, B `128x256`, C `128x256` | GM→L1→L0→MMAD→workspace | A/B 2-stage，C 1-stage |
| epilogue UB | tile `32x256`；双份 C/scale/per-token/bias/D 与共享 FP32 中间量 | workspace→UB→VF→UB→peer GM | 容量式 `2*(MN*4+N*4+M*4+N*sizeof(Bias)+MN*sizeof(D))+2*MN*4+M*32` |
| HCCL window | 每 stage `m0*p*N*sizeof(D)`，flag 从 180MiB 起 | 写 peer、读本 rank | 双 stage |
| y | rank-major receive slice | window→UB ping-pong→GM | 只写真实 tail |

##### 分层内存布局

~~~mermaid
flowchart LR
  X1["GM x1 INT8"] --> L1A["L1 A 128x512，双 stage"]
  X2["GM x2 INT8"] --> L1B["L1 B 512x256，双 stage"]
  L1A --> L0["L0A/L0B -> MMAD -> L0C INT32"]
  L1B --> L0
  L0 --> WS["GM workspace INT32，双 stage"]
  WS --> UB["AIV UB：cast/scale/bias/cast"]
  UB --> WIN["HCCL peer window"]
  WIN --> Y["AIV GM2GM -> GM y"]
~~~

##### 字节账、交接边与复用顺序

~~~text
workspace_total = 16 MiB + 2 * blockDim * 128 * 256 * 4
L1 = 2 * (128*512 + 512*256) = 393216 B
L0 = 2*128*128 + 2*128*256 + 4*128*256 = 229376 B
epilogue_UB_actual = 169216 B（FP16/BF16 bias 的最大可达切片）
~~~

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| workspace stage | AIC MMAD/Fixpipe | (stage*blockDim+core)*128*256，真实 tail 截断 | INT32 row-major | AIV epilogue | PIPE_FIX flag | 当前 kernel | AIV 回收 flag 后 AIC 方可复用 |
| peer-window stage | AIV epilogue | stage*128*p*N + rank slice | y dtype，rank slice | AIV GM2GM | 两次 rank rendezvous | HCCL window | p 经 180MiB 数据区容量收缩 |
| UB 复用 | AIV epilogue | ubMul 与 ubBiasAdd 别名 | FP32 | 同一 VF 顺序 | TEvent/队列顺序 | 单 epilogue tile | bias add 只能在 scale 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 边 | 机制 | 保护对象 | 精度/tail |
|---|---|---|---|
| AIC→AIV | `CrossCoreSet/WaitFlag<0x2,PIPE_FIX>` | workspace stage | `mActual/nActual` |
| AIV→AIC | `PIPE_MTE3` flag | stage reuse | 超过两 stage 前等待 |
| rank→rank | window flag 轮询，搬运前后各一次 | window 可见性/复用 | D dtype 不变 |
| VF | RINT INT32→F32，scale×scale，bias add，RINT cast | 数值顺序 | mask 有效 M×N |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 事实 |
|---|---|---|
| `quant_matmul.hpp` | `QuantMatmul::operator()/BlockMmad/BlockEpilogue` | scheduler、双 stage callback、tail offset |
| `block_epilogue_dequant.hpp` | `EpilogueAtlasA2PerTokenDequant` | scale/bias/cast 与 UB 式 |
| `matmul_allto_all_util.h` | `CommBase::Gm2Gm/CrossRankSync` | peer window、flag、GM2GM |

### Route R-002：256×128 通算流水

#### 适用条件

##### 选择证据

shape map 令 `m0=256` 且容量检查未回退；body 的非 128 分支实例化 L1 `<256,128,512>`、L0 `<256,128,128>`、epilogue `<64,128>`。

#### Device 数据流

##### 函数级调用链

与 R-001 相同；只改变 tile shape、循环数、offset 与 tail。

##### 源码等价伪码

```text
compute 256x128 tiles
repeat AIC workspace -> AIV dequant -> peer window -> local y
tail = min(256,Mremain) x min(128,NrankRemain)
```

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲 | extent | 差异 |
|---|---|---|
| workspace | `{256*blockDim*2,128}` INT32 | 总字节仍 `262144*blockDim` |
| L1/L0 | A L1 128KiB/stage、B L1 64KiB/stage；L0C 128KiB | 偏大 M |
| epilogue UB | `64x128`，同一容量式 | `MN=8192` |
| window | `256*p*N*sizeof(D)`/stage | p 受 180MiB 限制 |

##### 分层内存布局

~~~mermaid
flowchart LR
  X1["GM x1 INT8"] --> L1A["L1 A 256x512，双 stage"]
  X2["GM x2 INT8"] --> L1B["L1 B 512x128，双 stage"]
  L1A --> L0["L0A/L0B -> MMAD -> L0C INT32"]
  L1B --> L0
  L0 --> WS["GM workspace INT32，双 stage"]
  WS --> UB["AIV UB 64x128 epilogue"]
  UB --> WIN["HCCL peer window"]
  WIN --> Y["AIV GM2GM -> GM y"]
~~~

##### 字节账、交接边与复用顺序

~~~text
workspace_total = 16 MiB + 2 * blockDim * 256 * 128 * 4
L1 = 2 * (256*512 + 512*128) = 393216 B
L0 = 2*256*128 + 2*128*128 + 4*256*128 = 229376 B
epilogue_UB_actual = 168448 B（FP16/BF16 bias 的最大可达切片）
~~~

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| workspace stage | AIC MMAD/Fixpipe | (stage*blockDim+core)*256*128 | INT32 row-major | AIV epilogue | PIPE_FIX flag | 当前 kernel | 按真实 M/N tail 消费 |
| peer-window stage | AIV epilogue | stage*256*p*N + rank slice | y dtype | AIV GM2GM | rank rendezvous | HCCL window | 容量检查可回退 R-001 |
| UB 复用 | AIV epilogue | ubMul/ubBiasAdd 同址 | FP32 | VF 后继步骤 | event/队列顺序 | 单 tile | 禁止跨步骤提前覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

与 R-001 同 primitive/数值顺序；M tail 上界 256、N tile tail 上界 128，flag 仍以 stage 为复用粒度。

#### 关键实现

##### 符号证据表

`MatmulAlltoAll::QuantCatlassMatmul` 的非 128 模板实例是唯一区分证据；其余实现完全复用。

## 算子特定优化

### Trick T-001：Cube 回调与 AIV 反量化双 stage 重叠

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：A2、INT8 KC、MIX_AIC_1_2，workspaceStage=2。
- Golden 来源：[quant_matmul.hpp:324-326](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/quant_matmul.hpp#L324-L326)

官方源码的决定性连续片段：

~~~cpp
Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
blockEpilogue(blockOffset, actualBlockShape, gmBlockC, layoutBlockC, gmD[gmOffsetD], layoutD);
Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);
~~~

#### 机制与设计原因

AIC 写两份 INT32 workspace stage并置 FIX flag，AIV 等待后完成 scale/bias/cast 再置回收 flag，使相邻 Cube 与 epilogue 重叠。

#### 迁移边界

- 适用：同样具有一对 AIC/AIV 子块和 GM workspace 交接的 A2 KC 实现。
- 前提：workspace、flag/stage 索引与两 stage 生命周期保持一致。
- 失效：950 KC/MX 使用另一组 QBMM/pipeline 协议，不可直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在 PIPE_MTE3 回收信号前让 AIC 覆盖同一 stage，会造成 AIV 读到被覆盖的 INT32 tile；固定仓没有允许省略该等待的 fallback。

### Trick T-002：直接写 peer window 融合换位与发送

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：commMode=aiv，rankSize 为 2/4/8，HCCL window 布局初始化成功。
- Golden 来源：[quant_matmul.hpp:317-323](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch22/quant_matmul.hpp#L317-L323)

官方源码的决定性连续片段：

~~~cpp
int64_t gmOffsetD =
    static_cast<int64_t>(flagIdx) * L1TileShape::M * params.commUtil.p_value *
        params.problemShape.n() +
    rankIdx * rankOffset +
    static_cast<int64_t>(blockIdxCoord.m()) * L1TileShape::M *
        (params.problemShape.n() / params.commUtil.rank_size) +
    blockIdxCoord.n() % nLoopPerRank * L1TileShape::N;
~~~

#### 机制与设计原因

epilogue 按目标 rank 写 peer window，省去本地完整 `[M,N]` 中间结果；每 p 批次从本 rank window 收取所有 source rank 切片。

#### 迁移边界

- 适用：A2 peer-window 直写路线。
- 前提：rankSize 2/4/8、HCCL window 至少 200MiB，数据区/flag 区与 pValue 一致。
- 失效：950 先写 MM workspace、再转置、再提交 communicator，不是该布局。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 gmDOffset 当作普通本地行主序地址会破坏目标 rank 分片；删除批前/批后 rendezvous 会破坏窗口可见性或提前复用。

## 模板与实现对比

| 维度 | R-001 | R-002 |
|---|---|---|
| selector | `m0==128` 或容量回退 | map 选 256 |
| tile | M 小/N 大 | M 大/N 小 |
| workspace/精度/通信 | 相同总式与协议 | 相同 |
| 差异边界 | 编译期 tile 参数 | 编译期 tile 参数 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| V2 | 支持 | 不支持 | 支持 |
| quant | INT8 KC `(3,2)` | 无 route | FP8 KC、FP8/FP4 MX |
| commMode | `aiv` | — | `ccu`/`ai_cpu` |
| Kernel | arch22 peer-window | — | arch35 communicator+transpose |

## 限制与失败边界

- 仅固定仓固定提交 A2 V2；rankSize 2/4/8，`H2%rankSize==0`，不支持空 tensor、跨超节点或同通信域并发。
- A2 bias 必选；FLOAT32 bias 仅与 BF16 y 可达。文档的 FLOAT32 bias+FP16 y 与源码冲突，固定实现拒绝。
- 不接受非连续 tensor；x2 transpose 必须由连续存储与属性表达。
- HCCL buffer、workspace、POD 或 key 不满足时无降级；外部 HCCL POD ABI 数字缺失，不推测。
- 未执行编译、测试、validator、review、真机数值或性能验证。

## 源码索引

### 链路索引

| 阶段 | 文件 | 关键符号/用途 |
|---|---|---|
| 公开约束 | `docs/aclnnQuantMatmulAlltoAllV2.md` | V2 公式、dtype/shape/platform |
| API 声明 | `op_api/aclnn_quant_matmul_allto_all_v2.h` | 两段式 V2 入口声明 |
| API 实现 | `op_api/aclnn_quant_matmul_allto_all_v2.cpp` | V2 到 base 转发 |
| API base | `op_api/quant_matmul_allto_all_base.cpp` | 校验、Inner op、错误传播 |
| API util | `op_api/matmul_allto_all_util.cpp` | axes/group/commMode |
| Tiling | `op_host/op_tiling/arch22/matmul_allto_all_tiling_910b.cpp` | capability、map、workspace、HCCL |
| POD | `op_kernel/arch22/matmul_allto_all_tiling.h` | 本地 ABI 字段 |
| key | `op_kernel/arch22/matmul_allto_all_tiling_key.h` | router key |
| entry | `op_kernel/arch22/matmul_allto_all.cpp` | MIX router |
| body | `op_kernel/arch22/matmul_allto_all.h` | Init/Process 与模板选择 |
| Cube | `op_kernel/arch22/quant_matmul.hpp` | scheduler、MM、callback |
| epilogue | `op_kernel/arch22/block_epilogue_dequant.hpp` | scale/bias/cast/UB |
| comm | `op_kernel/arch22/matmul_allto_all_util.h` | peer window/flags/GM2GM |
| 片上资源 | `mc2/3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/arch.hpp` | AtlasA2 UB/L1/L0 容量常量 |
| 注册 | `op_host/matmul_allto_all_def.cpp` | 910B ACLNN/op file |
| 构建 | `CMakeLists.txt` | 源文件与 target 接线 |

### 实现复用检查

| 可复用项 | 不得改变 | 证据 |
|---|---|---|
| ACLNN | executor/workspace/stream 生命周期 | V2/base |
| route | shape map、窗口回退、m0模板映射 | Tiling/body |
| ABI | 本地 POD 顺序/位宽；不重建外部 POD | tiling header |
| workspace | 16MiB 与双 stage INT32 公式 | Host/QuantMatmul |
| 精度 | INT32→F32→两 scale→bias→cast | epilogue |
| 同步 | AIC/AIV flag 与 rank rendezvous | QuantMatmul/CommBase |