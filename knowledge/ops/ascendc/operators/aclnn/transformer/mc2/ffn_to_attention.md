---
type: Operator
title: "FfnToAttention A3 ACLNN 算子知识"
description: "说明 aclnnFFNToAttention 在 A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/docs/aclnnFFNToAttention.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_api/aclnn_ffn_to_attention.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_api/aclnn_ffn_to_attention.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_host/ffn_to_attention_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_host/op_tiling/ffn_to_attention_tilling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_kernel/ffn_to_attention_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_kernel/ffn_to_attention_tilling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_kernel/arch22/ffn_to_attention.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_kernel/ffn_to_attention.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_graph/ffn_to_attention_gen_task.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_graph/mc2_moe_gen_task_ops_utils.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_graph/mc2_gen_task_ops_utils.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/moe_distribute_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_kernel_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_host/config/ascend910_93/ffn_to_attention_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:28:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a3]
cann_versions: ["9.1.0"]
---

# FfnToAttention A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnFFNToAttentionGetWorkspaceSize` / `aclnnFFNToAttention`，无版本后缀的公开两段式 ACLNN |
| 计算语义 | 对每个有效 FFN token，依据 Attention worker、micro-batch、batch token 和 expert offset，把 `x[y, 0:H]` 写入目标 rank 的 HCCL window 数据表，并在对应 token-info 槽写 `int32(1)` |
| 输入输出 | 7 个输入（`attnRankTable` 可选）、4 个属性、无输出 Tensor；结果是远端通信 window 的副作用 |
| 执行类型 | `HYBRID`：AIV 数据搬运 + HCCL 隐藏上下文/远端 window；无 Cube、无本算子 AICPU Kernel 节点 |
| Kernel 路线 | R-001 默认 rank 映射；R-002 显式 `attnRankTable` 映射 |
| 核心资源 | 输入 GM、HCCL remote/user window、每核双缓冲 `xQueue_`、32 B `statusBuf_`；Host 返回 lib-api workspace，但算子 body 不切片或读取它 |
| 来源闭包 | `source_closed`：公开 ACLNN→生成式 inner bridge→OpDef/Tiling→任务改写→AIV entry/body→远端 window 写入闭合；生成式 inner bridge与 CANN `Mc2*` POD 定义是明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `Y=x.shape[0]`、`H=x.shape[1]`，`B=BS`，`E=ExpertNumPerToken`，`S=HS`，`m=microBatchIds[y]`，`b=tokenIds[y]`，`e=expertOffsets[y]`。对 `0 <= y < actualTokenNum[0]`：

- R-001 取目标 rank `r=sessionIds[y]`；R-002 取 `r=attnRankTable[sessionIds[y]]`。
- 目标 window 的 token-info 字节地址为 `win(r) + (m*B*E + b*E + e)*4`。
- token-data 区起点为 `win(r) + align_up(microBatchNum*B*E*4, 512)`；数据字节地址再加 `(m*B*E*S + b*E*S + e*S)*sizeof(xType)`。
- Kernel 原样复制 `x[y, 0:H]` 的 `H*sizeof(xType)` 字节，不做算术、cast、量化或 round；随后向 token-info 地址写 `int32(1)`。`HS-H` 是 window 中为 scale 等数据预留的步幅，不由本算子填充。

公开算子没有输出 Tensor；可观察结果是 Attention worker 对应 HCCL window 的数据与状态更新。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `(Y,H)` / ND | FP16、BF16 | `1024 <= H <= 8192`；可非连续，OpDef `AutoContiguous()` 后进入 Kernel |
| `sessionIds` | `(Y)` / ND | INT32 | 可非连续；Attention worker id；公开有效域 `[0, attnRankNum-1]`，Kernel 直接信任其值 |
| `microBatchIds` | `(Y)` / ND | INT32 | 可非连续；micro-batch 下标；当前公开域只有 0，因为 `MicroBatchNum=1` |
| `tokenIds` | `(Y)` / ND | INT32 | 可非连续；batch token 下标，公开有效域 `[0, BS-1]` |
| `expertOffsets` | `(Y)` / ND | INT32 | 可非连续；每 token 的 expert 槽下标，公开有效域 `[0,E-1]` |
| `actualTokenNum` | `(1)` / ND | INT64 | 可非连续；实际有效 token 数，公开范围 `[0,Y]`；Device 读标量后分核 |
| `attnRankTable` | `(A)` / ND，可选、可非连续 | INT32 | 缺省时 `rank=sessionIds[y]`；存在时 `rank=attnRankTable[sessionIds[y]]`，Host 要求 `A <= worldSize-1` |
| `group` | 字符串 | STRING | 必选，长度 `[1,128)`；该通信域内不允许有其他算子 |
| `worldSize` | 标量 | INT64 API / Host 以 `int` 读取 | `[2,768]`；所有卡及网络不同层必须一致 |
| `tokenInfoTableShape` | `[MicroBatchNum,BS,E]` | ListInt | 长度 3；与 `tokenDataShape` 前三维相等；当前 `MicroBatchNum=1` |
| `tokenDataShape` | `[MicroBatchNum,BS,E,HS]` | ListInt | 长度 4；公开 `1152 <= HS <= 8320`，Host 实现实际接受 `[1024,8320]`，正式 API 域按公开文档较窄范围 |
| 输出 | 无 | N/A | 远端 window 是副作用，不返回本地 Tensor |

### 入口与版本

固定 Commit 只公开无版本后缀的 `aclnnFFNToAttentionGetWorkspaceSize` 与 `aclnnFFNToAttention`，目录中没有 V2/V3 等同语义公开入口，因此它就是 A3 的最新公开版本。第一段检查必选 Tensor 与 `group` 后调用生成头 `aclnnInner_ffn_to_attention.h` 提供的 inner 接口；第二段在弱符号存在时把 executor 的 HCCL server 类型设为 `NNOPBASE_HCCL_SERVER_TYPE_MTE`，再执行 inner 接口。生成式 inner bridge 不在固定仓中，其参数绑定、executor 与 stream 提交属于框架边界。

README、接口文档和 `OpAICoreConfig::AddConfig("ascend910_93", ...)` 一致证明仅 A3 公开支持；A2、200I/500 A2、旧 Atlas 训练/推理和 950 均明确不支持。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 只检查 `x/sessionIds/microBatchIds/tokenIds/expertOffsets/actualTokenNum` 非空、`group` 非空且长度小于 128；`attnRankTable` 允许为空。OpDef 为所有 Tensor 注册 `AutoContiguous()`。随后 inner bridge 建立 `FFNToAttention` 节点；Tiling 按以下顺序执行：

1. `CheckAndSetAttrs` 检查属性和两个 shape list，要求前三维相等、`worldSize in [2,768]`、`MicroBatchNum=1`、`BS in [1,512]`、`HS in [1024,8320]`。
2. `CheckInputDim` 检查 `x` 为 2D、其余必选 Tensor 为 1D、各索引 Tensor 的长度等于 `Y`、`actualTokenNum.shape[0]=1`；可选 rank table 为 1D 且 `A<=worldSize-1`。
3. `CheckInputDataType` 检查 `x` 为 FP16/BF16、索引为 INT32、实际 token 数为 INT64；`CheckInputFormat` 排除 `FRACTAL_NZ`。
4. `CalWinSize`、`SetWorkSpace`、`SetHcommCfg`、tiling key 与 `blockDim` 依次产生。任一步失败均返回 `GRAPH_FAILED`，不产生可达 Kernel route。

### 参数、Workspace 与生命周期

- window 逻辑数据字节为 `microBatchNum*BS*E*HS*2`，逻辑 info 字节为 `microBatchNum*BS*E*4`；`CalWinSize` 以 `floor((data+info)/MiB)+1` 与 `floor(GetMaxWindowSize()/MiB)` 比较。它是 window 的 MiB 桶 guard，不是 Device workspace 分配；因为 Device 另把 info 区向上对齐到 512 B，逻辑总量距 MiB 边界不足该 padding 时，这个 guard 不能单独证明物理布局一定装得下。
- 实际 window 内部把 info 区物理容量对齐到 512 B：`winInfoBytes=ceil(infoBytes/512)*512`；data 区紧随其后。
- `SetWorkSpace` 返回 `PlatformAscendC::GetLibApiWorkSpaceSize()`；其具体字节数由平台库提供，固定仓没有常量。entry 把 `workspaceGM` 传入 `Init`，但 `Init/Process` 不读取它，因此只能确认没有算子 body 自有的 workspace slice，不能把 ACLNN 返回的总 workspace 记成 0。
- HCCL window 由 `GetHcclContext<HCCL_GROUP_ID_0>()` 隐藏输入拥有。`userMemType==0` 时取 `remoteRes[r].nextDevicePtr->windowsIn`，否则取 `userMemRes[r].addr`；其创建、释放和跨算子消费由 HCCL/执行框架负责，固定仓没有显式 release。
- executor 覆盖两段式 ACLNN 的任务图生命周期；第二段把任务提交到调用方 stream。所有卡的通信参数与 `HCCL_BUFFSIZE` 必须一致。

### 启动图与失败传播

```text
aclnnFFNToAttentionGetWorkspaceSize
  -> aclnnInnerFFNToAttentionGetWorkspaceSize（生成式框架边界）
    -> FFNToAttentionTilingFunc
      -> SetHcommCfg(group, Alltoall=level0:fullmesh;level1:pairwise)
      -> key = GET_TPL_TILING_KEY(isInputRankTable, A3)
      -> blockDim = CalcTschBlockDim(aivNum, 0, aivNum)
    -> FFNToAttentionGenTaskFunc
      -> Mc2MoeGenTaskCallbackV2
        -> 向唯一 AICore task 插入 1 个 kHcom 隐藏输入
        -> FFNToAttention 命中 useAiCpu=false，不插入 wait/AICPU/record task
aclnnFFNToAttention
  -> 可选 NnopbaseSetHcclServerType(executor, MTE)
  -> aclnnInnerFFNToAttention(..., stream)
    -> ffn_to_attention<RankTableMode, A3>
      -> FFNToAttention<DTYPE_X, RankTableMode>::Init/Process
```

任务生成要求一个名为 `aicpu kfc server`、复用键 `kfc_stream` 的 attached stream 资源，但 V2 回调对 `FFNToAttention` 明确不插入 AICPU task。AICore task 或 hidden Hcom 注入失败时返回 `GRAPH_FAILED`；ACLNN inner 调用的失败状态原样返回。Device body 没有错误码通道，地址/索引合法性依赖公开前置约束。

## Route 与参数设计

### 分核与资源计算

Host 取平台 `aivNum=GetCoreNumAiv()`，`blockDim=CalcTschBlockDim(aivNum,0,aivNum)`，并把 `aivNum` 写入 POD。Device 对 `N=actualTokenNum[0]` 采用循环交错分核：AIV `i` 处理 `y=i+t*aivNum`，处理数为 `floor(N/aivNum)+(i < N%aivNum)`；`i>=N` 的核处理 0 个 token。

每个核向 `InitBuffer` 请求深度 2、单槽 `H*sizeof(xType)=2H` B 的 `xQueue_`，以及 32 B `statusBuf_`，故源码可见请求量为 `2*(2H)+32=4H+32` B，`H=8192` 时为 `32800` B。`InitBuffer` 对非 32 B 长度的内部舍入规则属于仓外 primitive，固定仓不能把请求量进一步等同为物理占用；Host 虽记录平台总 UB 字节数，却没有执行 `UB_requested<=totalUbSize` selector。L1、L0A/B/C、显式寄存器 tile 和 dynamic UB 均不使用。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 默认 rank 映射 | `GET_TPL_TILING_KEY(false,A3)`；`FFNToAttention<DTYPE_X,false>` | `GetOptionalInputShape(attnRankTable)==nullptr`，其余 Host 检查及 window guard 全通过 | optional input 存在则转 R-002；其他失败终止 Tiling |
| 2 | R-002 显式 rank table | `GET_TPL_TILING_KEY(true,A3)`；`FFNToAttention<DTYPE_X,true>` | `attnRankTable` 存在、1D、INT32、非 NZ 且 `A<=worldSize-1`，其余检查全通过 | 任一条件失败终止 Tiling，无 Device fallback |

```text
if common API/Host checks fail or required window MiB > available window MiB:
    GRAPH_FAILED
else if attnRankTable is absent:
    R-001: key(false, A3), rank = sessionIds[y]
else:
    R-002: key(true, A3), rank = attnRankTable[sessionIds[y]]
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 默认映射 | `rankTableMode=false, ARCH_TAG=A3` | A3；FP16/BF16；公共 shape/attr/format/window guard；rank table 缺省 | 单 AIV Kernel + HCCL remote-window MTE 写 | `ffn_to_attention<false,A3>` → `FFNToAttention<DTYPE_X,false>::Process` | GM→双缓冲 UB→remote data；`S_MTE3` 首轮同步；`PIPE_MTE3` 后写 info flag | token 数按核取模；0 token 核不进入循环；失败只在 Host |
| R-002 rank table | `rankTableMode=true, ARCH_TAG=A3` | 同 R-001，并要求 rank table 1D/INT32/非 NZ/`A<=worldSize-1` | 同 R-001 | `ffn_to_attention<true,A3>` → `ReadTokenMetaData` 增加 rank table 间接读 | 多一次 GM 标量读与 cache clean/invalidate；其余相同 | 同 R-001；不存在无表 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 | README 仅 A3=√；OpDef 仅 `ascend910_93` | entry 的 `ArchTag` 仅执行 A3 分支 | R-001/R-002 | 不发布其他平台 route | `FFNToAttention::FFNToAttention`、`ffn_to_attention` |
| window 容量 | `floor((dataBytes+infoBytes)/MiB)+1 <= floor(maxWindowBytes/MiB)` | 远端 window 必须容纳 info 与 token data | 两者 | `GRAPH_FAILED` | `CalWinSize`、`FFNToAttentionTilingFunc` |
| UB | `H in [1024,8192]`；无显式 `UB_requested=4H+32<=totalUbSize` 判断 | 双缓冲 queue 加 status buffer 的源码请求量为 `4H+32` B；primitive 物理舍入仓内不可见 | 两者 | H 越界时 Host 失败；UB 不足没有独立 fallback | `CheckInputDim0Dim1`、`InitBuffer` |
| 32-bit 窄化与乘积 | 公共域依赖 `BS<=512`、`HS<=8320` 及文档中的 `E=K+sharedExpertNum<=20`；无独立乘法溢出 guard | `batchSizeSendCnt_`、`perMicroBatchOffset_`、`perExpertOffset_` 的部分乘法先在 `uint32_t` 中完成；`sendTokenNum` 也把 `uint64_t N/aivNum` 窄化到 `uint32_t` | 两者 | 超出公开值域没有安全 fallback | `CheckAndSetAttrs`、`Init`、`Process` |
| rank table | absent 或 `A<=worldSize-1` | 决定是否额外间接寻址目标 rank | R-001/R-002 | 有表但非法则 Host 失败 | `CheckInputDim`、`GET_TPL_TILING_KEY` |
| token 索引有效域 | 文档要求 `session/microBatch/token/expert` 均在各自范围 | 这些值直接进入 remote rank 与字节地址公式 | 两者 | 固定 Host 未逐值校验，无安全 fallback | `ReadTokenMetaData`、`Process` |
| 实际 token 数 | 文档要求 `0<=actualTokenNum<=Y` | 决定循环次数和 `x[y]` 地址 | 两者 | 固定 Host 未读值校验，无安全 fallback | `actualTokenNumGMTensor_.GetValue(0)`、`Process` |
| 手工同步 | 构建对 `ascend910_93` 增加 `--cce-auto-sync=off` | 源码自行承担 MTE pipeline 顺序 | 两者 | 无另一编译 route | `op_host/CMakeLists.txt`、`SyncFunc`、`PipeBarrier` |

### POD/ABI

`FFNToAttentionInfo` 的仓内可见 ABI 为 `alignof=8`、`sizeof=56`：前 8 个 `uint32_t` 占偏移 0..31，两个 1 B 字段位于 32、33，补齐到偏移 40 后放两个 `uint64_t`。外层前置的 `Mc2InitTiling`、`Mc2CcTiling` 类型定义来自仓外 `kernel_tiling/kernel_tiling.h`，故外层 `sizeof/offsetof` 在固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `FFNToAttentionTilingData` | `mc2InitTiling` | `Mc2InitTiling` | offset 0；size/align 固定仓不可见 | 通信配置 POD | `SetHcommCfg::GetTiling` | body 不读取；运行时 Hcom 资源消费属于框架 | 不得臆造字段 |
| 同上 | `mc2CcTiling1` | `Mc2CcTiling` | 紧随前字段；offset/size/align 固定仓不可见 | 通信配置 POD | `SetHcommCfg::GetTiling` | body 不读取 | `OP_TYPE_ALL_TO_ALL=8`、fullmesh/pairwise 配置 |
| 同上 | `ffnToAttentionInfo` | `FFNToAttentionInfo` / 56 B | offset 受前两个仓外类型影响 | 算子 POD | 各 Host helper | `Init`/entry | 外层 offset 固定仓不可见 |
| `FFNToAttentionInfo` | `H` | `uint32_t` / 32 bit | offset 0 | 元素 | `CheckInputDim0Dim1: xDim1` | `axisH_`、copy 长度、UB 容量 | `[1024,8192]` |
| 同上 | `A` | `uint32_t` / 32 bit | offset 4 | rank-table 元素 | 有表时 `attnRankTableDim0` | `axisA_`，之后未使用 | 无表时 producer 未显式赋值 |
| 同上 | `microBatchNum` | `uint32_t` / 32 bit | offset 8 | 个 | `tokenDataShape[0]` | info 对齐容量 | 固定为 1 |
| 同上 | `BS` | `uint32_t` / 32 bit | offset 12 | token 槽/批 | `tokenDataShape[1]` | `batchSizeSendCnt_` | `[1,512]` |
| 同上 | `expertNumPerToken` | `uint32_t` / 32 bit | offset 16 | expert 槽/token | `tokenInfoTableShape[2]` | info/data stride | 文档值 `K+sharedExpertNum` |
| 同上 | `HS` | `uint32_t` / 32 bit | offset 20 | 元素/槽 | `tokenDataShape[3]` | data stride | 公开 `[1152,8320]` |
| 同上 | `aivNum` | `uint32_t` / 32 bit | offset 24 | AIV 核 | `GetCoreNumAiv()` | 分核除数/步长 | 必须非 0，源码未单独 guard |
| 同上 | `worldSize` | `uint32_t` / 32 bit | offset 28 | rank | `worldSize` attr | `worldSize_`，之后未使用 | Host `[2,768]` |
| 同上 | `isInputRankTable` | `bool` / 8 bit | offset 32 | 布尔 | optional input 是否存在 | 不在 body 读取；用于 Host key | key 与字段必须一致 |
| 同上 | `windowType` | `uint8_t` / 8 bit | offset 33 | 枚举值 | 固定 Tiling 未赋值 | 复制到 `windowType_` 后未使用 | inactive；不得依赖其值 |
| 同上 | padding | 6 B | offset 34..39 | 字节 | ABI 对齐 | 无 | 固定仓未显式清零证据 |
| 同上 | `totalUbSize` | `uint64_t` / 64 bit | offset 40 | 字节/核 | `GetCoreMemSize(UB)` | body 不读取 | 仅日志/观测字段 |
| 同上 | `totalWinSize` | `uint64_t` / 64 bit | offset 48 | 字节 | `GetMaxWindowSize()` | body 不读取 | 仅 Host window guard/日志 |
| Kernel launch ABI | `x..attnRankTable,workspaceGM,tilingGM` | 9 个 `GM_ADDR` / 各 64 bit | 非聚合参数，无 `offsetof` | GM 地址 | generated inner + hidden Hcom task rewrite | `ffn_to_attention` | 参数顺序不可改变；Hcom 隐藏输入由框架插入 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 默认映射 | `rankTableMode=false,A3` | `FFNToAttentionTilingFunc` | `FFNToAttentionTilingData`；无表 GM 地址 inactive | `ffn_to_attention<false,A3>` | `FFNToAttention<DTYPE_X,false>::Init/Process`；`HcclOpResParam` | `ffn_to_attention_def.cpp`、binary JSON、Host CMake |
| R-002 rank table | `rankTableMode=true,A3` | `CheckInputDim` + `FFNToAttentionTilingFunc` | 同上；`attnRankTable` active | `ffn_to_attention<true,A3>` | `FFNToAttention<DTYPE_X,true>::ReadTokenMetaData/Process` | 同 R-001 |

## Kernel 实现

### 公共执行流程

entry 注册并读取 `FFNToAttentionTilingData`，仅当编译期 `ArchTag==TILINGKEY_TPL_A3` 时构造 `FFNToAttention<DTYPE_X,RankTableMode>`。任务生成器把 1 个 Hcom hidden input 插入既有 AICore task，V2 对本算子不增加 wait/AICPU/record 节点。

`Init` 取得 block id 与 HCCL context，绑定全部 GM，cache clean/invalidate 后读取 `actualTokenNum[0]`，初始化 `xQueue_` 和值为 1 的 status buffer，并推导 window stride。`Process` 只在 `ASCEND_IS_AIV` 分支执行；各核循环读取 token、读取 4 个元数据标量、选择远端 window、先写 token data 再写状态位。没有 AIC 分支、Cube、数值计算或本地输出。

### Route R-001：默认 rank 映射

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_ffn_to_attention.cpp` | `aclnnFFNToAttentionGetWorkspaceSize` | 可选 `attnRankTable==nullptr` 合法；公共必选项通过 | inner op |
| Host/Tiling | `op_host/op_tiling/ffn_to_attention_tilling.cpp` | `CheckInputDim` | `GetOptionalInputShape(6)==nullptr` | `isInputRankTable=false` |
| Host/Tiling | 同上 | `FFNToAttentionTilingFunc` | `GET_TPL_TILING_KEY(false,TILINGKEY_TPL_A3)` | R-001 key |
| Router | `op_kernel/arch22/ffn_to_attention.cpp` | `ffn_to_attention<false,A3>` | `ArchTag==A3` | `FFNToAttention<DTYPE_X,false>` |

#### Device 数据流

##### 函数级调用链

```text
ffn_to_attention<false,A3>(x,...,attnRankTable,workspaceGM,tilingGM)
  -> FFNToAttention<DTYPE_X,false>::Init
    -> GetHcclContext<HCCL_GROUP_ID_0>
    -> InitBuffer(xQueue_, 2, H*2) + InitBuffer(statusBuf_, 32)
  -> FFNToAttention<DTYPE_X,false>::Process
    -> ReadTokenMetaData（rank=sessionIds[y]）
    -> GetWindowAddr(rank)
    -> DataCopyPad(GM x -> UB)
    -> DataCopyPad(UB -> remote token data)
    -> PipeBarrier<PIPE_MTE3>
    -> DataCopyPad(status=1 -> remote token info)
```

##### 源码等价伪码

```text
N = cache_invalidate_then_load(actualTokenNum[0])
for AIV i:
    count = N / aivNum + (i < N % aivNum)
    for t in [0,count):
        y = i + t*aivNum
        ub = alloc xQueue slot
        DataCopyPad(ub, x + y*H, H*2 bytes)
        worker = load sessionIds[y]
        m = load microBatchIds[y]; b = load tokenIds[y]; e = load expertOffsets[y]
        win = GetWindowAddr(worker)
        data = win + align_up(microBatchNum*BS*E*4,512)
                   + (m*BS*E*HS + b*E*HS + e*HS)*2
        info = win + (m*BS*E + b*E + e)*4
        if t == 0: SyncFunc<S_MTE3>()
        enqueue/dequeue ub
        DataCopyPad(data, ub, H*2 bytes)
        free ub; PipeBarrier<PIPE_MTE3>()
        DataCopyPad(info, statusTensor(value=1), 4 bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 输入 GM `x` | `y*H` 元素 | `Y*H*2` B | FP16/BF16 ND | 调用方 | MTE2 `DataCopyPad` | 调用方至 stream 完成 | 每次 `H*2` B |
| 元数据 GM | 各 `[y]` | 4 B×4 + 8 B actual count | INT32/INT64 ND | 调用方 | scalar load | 同上 | 每次先 DCCI cache clean/invalidate |
| HCCL remote/user window info | `(m*BS*E+b*E+e)*4` | 物理 header `ceil(microBatch*BS*E*4/512)*512` B | INT32 flat | `statusTensor_` 经 MTE3 | Attention worker | HCCL context/window | 512 B 后接 data；每槽写 4 B |
| HCCL remote/user window data | `winInfoBytes+(m*BS*E*HS+b*E*HS+e*HS)*2` | `microBatch*BS*E*HS*2` B | xType flat | UB 经 MTE3 | Attention worker | HCCL context/window | 槽 stride `HS*2` B，只写前 `H*2` B |
| UB `xQueue_` | queue slot 0/1 | `2*(H*2)=4H` B | xType | MTE2 | MTE3 | 当前 AIV/循环；Free 后复用 | 深度 2，单槽恰为一行 H |
| UB `statusBuf_` | offset 0 | 32 B | INT32 | scalar `statusTensor_(0)=1` | MTE3 | Kernel 全生命周期 | 32 B 对齐；仅前 4 B 有效 |
| lib-api workspace | 框架返回的基址 | `GetLibApiWorkSpaceSize()`，固定仓未给出常量 | opaque | 平台库/ACLNN executor | 算子 body 不读取 | executor 至 stream 完成 | 无算子自有 slice；不能当作 window |
| L1/L0A/L0B/L0C/寄存器 | N/A | 0 | N/A | N/A | 由 GM↔UB MTE 路径替代 | N/A | 无 Cube/算术 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq0["xQueue_ slot 0 · H×2 B"]:1
  xq1["xQueue_ slot 1 · H×2 B"]:1
  status["statusBuf_ · 32 B"]:1
```

GM→UB→远端 window 的地址、owner 与同步边见上表；图中只画 `InitBuffer` 实际请求的 AIV UB 子区。

##### 字节账与复用顺序

```text
workspace_total = PlatformAscendC::GetLibApiWorkSpaceSize()（固定仓未给出常量）
workspace_operator_slice = 0 B（Kernel 不消费 workspaceGM）
window_info_physical = ceil(microBatchNum*BS*E*4 / 512) * 512 B
window_data = microBatchNum*BS*E*HS*2 B
UB_xQueue_requested = 2 * H * 2 = 4H B
UB_status_requested = 32 B
UB_total_requested = 4H + 32 B <= 32800 B（H<=8192）
UB_total_physical = 固定仓不可见（InitBuffer 对非 32 B 长度的内部舍入属于仓外 primitive）
dynamic_ub = N/A（AIV 静态 TPipe buffer）
L1 = L0A = L0B = L0C = 0 B
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| GM→UB slot | MTE2 `DataCopyPad` | `x+y*H` / `H*2` B | xType ND row | MTE3 copy-out | `TQueBind` EnQue/DeQue | 当前 AIV；Free 后下一轮复用 | `y<actualTokenNum<=Y` |
| UB→remote data | MTE3 `DataCopyPad` | 上述 data offset / `H*2` B | xType flat | Attention window consumer | 首轮 `SyncFunc<S_MTE3>`；queue 协议 | HCCL window | rank/m/b/e 有效，window 足够 |
| data→info 发布 | data-copy MTE3 | data `H*2` B 后 info 4 B | xType 后 INT32 | Attention consumer 以 info=1 判断数据有效 | `PipeBarrier<PIPE_MTE3>` 后 flag copy | HCCL window，跨算子生命周期 | 不可把 flag 提前到 data copy 前 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| actual count/元数据读取 | `DataCacheCleanAndInvalid<...,CACHELINE_OUT>` | 调用方 GM | Device scalar load | cache line | INT64/INT32 原值，不转换 |
| 首个 token MTE3 | `SyncFunc<HardEvent::S_MTE3>` | scalar/queue 初始化 | 首次 remote copy | scalar→MTE3 可见性 | 只在 `t==0` |
| x queue | `Alloc→EnQue→DeQue→Free` | MTE2 | MTE3 | 双缓冲 slot | 原样 FP16/BF16，copy 长度恰 H，无行内 tail |
| data 发布 flag | `PipeBarrier<PIPE_MTE3>` | remote data copy | remote info copy | 同一 token 数据/状态对 | 先完整 `H*2` B，再写 4 B 状态 |
| token tail | 循环次数公式 | 分核 | 每核循环 | `actualTokenNum` 域 | 余数前若干 AIV 多处理 1 个；无 padding token |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/op_tiling/ffn_to_attention_tilling.cpp` | `FFNToAttentionTilingFunc` | 无表 key、A3 tag、全 AIV blockDim、window/UB 字段 |
| Kernel entry/router | `op_kernel/arch22/ffn_to_attention.cpp` | `ffn_to_attention<false,A3>` | 读取 POD，实例化 `DTYPE_X,false` |
| Device body/helper | `op_kernel/ffn_to_attention.h` | `Init/ReadTokenMetaData/GetWindowAddr/Process` | 默认 rank、循环交错分核、远端 window data-before-flag |

### Route R-002：显式 rank table 映射

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_ffn_to_attention.h` | `aclnnFFNToAttentionGetWorkspaceSize` | `attnRankTable` 非空可选输入 | inner op 保留该 Tensor |
| Host/Tiling | `op_host/op_tiling/ffn_to_attention_tilling.cpp` | `CheckInputDim/DataType/Format` | 有表且 1D/INT32/非 NZ、`A<=worldSize-1` | `isInputRankTable=true` |
| Host/Tiling | 同上 | `FFNToAttentionTilingFunc` | `GET_TPL_TILING_KEY(true,TILINGKEY_TPL_A3)` | R-002 key |
| Router | `op_kernel/arch22/ffn_to_attention.cpp` | `ffn_to_attention<true,A3>` | `ArchTag==A3` | `FFNToAttention<DTYPE_X,true>` |

#### Device 数据流

##### 函数级调用链

```text
ffn_to_attention<true,A3>
  -> FFNToAttention<DTYPE_X,true>::Init（绑定 attnRankTable GM）
  -> Process
    -> ReadTokenMetaData
      -> workerId = sessionIds[y]
      -> targetRank = attnRankTable[workerId]
    -> 与 R-001 相同的 GetWindowAddr/data copy/barrier/info flag
```

##### 源码等价伪码

```text
worker = cache_invalidate_then_load(sessionIds[y])
targetRank = cache_invalidate_then_load(attnRankTable[worker])
m = load microBatchIds[y]; b = load tokenIds[y]; e = load expertOffsets[y]
win = GetWindowAddr(targetRank)
copy x[y,0:H] -> UB -> win.data[m,b,e,0:H]
PipeBarrier<MTE3>()
copy int32(1) -> win.info[m,b,e]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `attnRankTable` GM | `sessionIds[y]*4` B | `A*4` B | INT32 ND | 调用方 | scalar rank load | 调用方至 stream 完成 | 每 token 读 1 元素 |
| 输入/元数据 GM | 同 R-001 | 同 R-001 | 同 R-001 | 调用方 | AIV | 同 R-001 | 增加 rank-table 间接寻址 |
| remote/user window | `targetRank=table[worker]` 后地址公式同 R-001 | info/data 公式同 R-001 | INT32 + xType flat | AIV MTE3 | Attention worker | HCCL context | data-before-flag |
| UB/L1/L0/workspace | 与 R-001 完全相同 | `UB_requested=4H+32` B；lib-api workspace 由框架拥有、body 不读取；L1/L0 不使用 | xType/INT32/opaque | 同 R-001 | 同 R-001 | 同 R-001 | route 不改变容量；physical UB rounding 固定仓不可见 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  xq0["xQueue_ slot 0 · H×2 B"]:1
  xq1["xQueue_ slot 1 · H×2 B"]:1
  status["statusBuf_ · 32 B"]:1
```

R-002 的 UB 布局与 R-001 相同；额外的 `attnRankTable` 是 GM 标量读取，不占用另一个 `TQue/TBuf`。

##### 字节账与复用顺序

```text
额外 GM 读取 = 每有效 token 4 B rank table 元素（不计 cache-line 物理事务放大）
workspace_total = PlatformAscendC::GetLibApiWorkSpaceSize()（固定仓未给出常量）
UB_total_requested = 4H + 32 B；UB_total_physical 的 primitive 舍入固定仓不可见
dynamic_ub = N/A；其余 window/L1/L0 字节账与 R-001 相同
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| worker→rank | sessionIds scalar load | worker 下标、table 元素 4 B | INT32 | `GetWindowAddr` | 两次 load 间程序顺序，均先 DCCI | 输入 Tensor | worker 必须落在 `[0,A)`，Host 不逐值检查 |
| data→info | 同 R-001 | 同 R-001 | xType→INT32 flag | Attention worker | `PipeBarrier<PIPE_MTE3>` | HCCL window | table 值必须是合法通信 rank |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| rank table 读取 | `DataCacheCleanAndInvalid` | rank table GM | scalar load | table cache line | INT32 原值，无 cast |
| 后续搬运/发布 | 与 R-001 相同 | 同 R-001 | 同 R-001 | UB 与远端 token 槽 | FP16/BF16 bitwise 搬运；token tail 同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/op_tiling/ffn_to_attention_tilling.cpp` | `CheckInputDim`、`FFNToAttentionTilingFunc` | 有表检查并生成 true key |
| Kernel entry/router | `op_kernel/arch22/ffn_to_attention.cpp` | `ffn_to_attention<true,A3>` | 编译期启用 rank-table specialization |
| Device body/helper | `op_kernel/ffn_to_attention.h` | `Init`、`ReadTokenMetaData` | 绑定 table，并以 `table[sessionIds[y]]` 选择 remote window |

## 算子特定优化

### Trick T-001：可选 rank table 的编译期 specialization

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：A3 且 optional `attnRankTable` 的存在性决定 `rankTableMode`。
- Golden 来源：[`ffn_to_attention_tilling.cpp:507-520`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_host/op_tiling/ffn_to_attention_tilling.cpp#L507-L520)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // Set TilingKey
    bool rankTableMode = tilingData->ffnToAttentionInfo.isInputRankTable;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A3);
    OP_LOGD(nodeName, "cur case tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // Set numBlocks
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
```

#### 机制与设计原因

Host 把 optional 输入存在性编码进模板 key，entry 直接实例化 `FFNToAttention<DTYPE_X,RankTableMode>`。R-001 的 `if constexpr` 消除 rank table 绑定、DCCI 和间接 GM 读取；R-002 只为需要自定义 worker→rank 映射的场景支付该访问。

#### 迁移边界

- 适用：同一主体搬运、仅地址映射 optional 的 AIV 通信算子。
- 前提：Host key、template selection 和 optional GM 参数的 active 状态严格一致。
- 失效：若映射影响 window 布局、dtype 或同步图，必须拆成不同 body/POD，不能只加布尔分支。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时必须保持 Host key、optional 参数 active 状态与 `if constexpr` specialization 一致。

### Trick T-002：全 AIV 循环交错分核与每核双缓冲

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：所有合法 `actualTokenNum`；A3 `aivNum` 个 Vector 核。
- Golden 来源：[`ffn_to_attention.h:169-195`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_kernel/ffn_to_attention.h#L169-L195)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    if ASCEND_IS_AIV {
        uint32_t calStartTokenId = 0;
        uint32_t sendTokenNum = actualTokenNum_ / aivNum_;
        uint32_t remainderRankNum = actualTokenNum_ % aivNum_;
        if (aivId_ < remainderRankNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
            sendTokenNum += 1;
            calStartTokenId += aivId_;
        } else {
            calStartTokenId += remainderRankNum;
        }
        if (calStartTokenId >= actualTokenNum_) {
            sendTokenNum = 0;
        }
        GM_ADDR curRankWinAddr;
        GlobalTensor<xType> tokenDataGMTensor;
        GlobalTensor<int32_t> tokenInfoTableGMTensor;
        DataCopyExtParams dataCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(axisH_ * sizeof(xType)), 0U, 0U, 0U};
        DataCopyPadExtParams<xType> copyPadExtParams{false, 0U, 0U, 0U};
        for (uint32_t tokenCnt = 0; tokenCnt < sendTokenNum; ++tokenCnt) {
            uint32_t YOffset = aivId_ + (tokenCnt * aivNum_); // 当前token在Y的偏移

            // 1. 加载token
            xTmpTensor_ = xQueue_.AllocTensor<xType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[YOffset * axisH_], xCopyParams, copyPadExtParams);
```

#### 机制与设计原因

循环交错映射使有效 token 数的负载差最多为 1，并让 `N<aivNum` 时仅前 N 个核工作。每核 queue 深度为 2，单槽恰好容纳一行 `H`，以 queue 生命周期连接输入搬运与远端 MTE3 写；代码没有证明具体重叠周期，因此只能确认双缓冲容量与复用协议，不能宣称量化性能收益。

#### 迁移边界

- 适用：每个 token 的搬运字节相同、token 之间无归约依赖的 scatter。
- 前提：`aivNum>0`、`actualTokenNum<=Y`，且平台 UB 能容纳 `4H+32` B 源码请求量及仓外 primitive 可能施加的对齐舍入。
- 失效：可变长 token 或单 token 需要跨核协作时，需重新设计分核与 queue 容量。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但不得把 `calStartTokenId` 误写成实际输入起点，真正地址始终是 `aivId_ + tokenCnt*aivNum_`。

### Trick T-003：远端数据先行、状态位后发布

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：每个有效 token 的 remote window 写入。
- Golden 来源：[`ffn_to_attention.h:204-219`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/ffn_to_attention/op_kernel/ffn_to_attention.h#L204-L219)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
            uint64_t tokenInfoTableOffset = metaDataStruct.curMicroBatchIds * batchSizeSendCnt_;
            uint64_t tokenDataOffset = (metaDataStruct.curMicroBatchIds * perMicroBatchOffset_ + metaDataStruct.curTokenBatchOffset * perExpertOffset_ + metaDataStruct.curTokenTopkOffset * axisHS_) * sizeof(xType);
            tokenInfoTableGMTensor.SetGlobalBuffer((__gm__ int32_t*)(curRankWinAddr + (tokenInfoTableOffset + metaDataStruct.curTokenBatchOffset * expertNumPerToken_ + metaDataStruct.curTokenTopkOffset) * sizeof(int32_t))); // win区分布，前面放info 后面放data
            tokenDataGMTensor.SetGlobalBuffer((__gm__ xType*)(curRankWinAddr + winTokenInfoTableSize_ + tokenDataOffset));

            // 5. 复制数据到win区
            if(tokenCnt == 0) {
                SyncFunc<AscendC::HardEvent::S_MTE3>();
            }
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<xType>();
            DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams); // 数据搬入token_data位置上
            xQueue_.FreeTensor<xType>(xTmpTensor_);
            PipeBarrier<PIPE_MTE3>();
            DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams); // 状态位
```

#### 机制与设计原因

window 把固定 512 B 对齐的 info header 与 `HS` stride 的 token data 分离。生产者先复制实际 `H` 元素，再用 MTE3 barrier 保证同 token 的 data copy 先于 4 B `status=1`；消费端可把状态位作为数据已发布标记。该协议避免消费者看到 flag 后读取尚未完成的 token data。

#### 迁移边界

- 适用：远端 window 采用 data+valid-flag 的单生产者槽位协议。
- 前提：`m/b/e/rank` 唯一定位合法槽位，生产者不会并发覆盖同槽，消费者遵守 flag 协议。
- 失效：多生产者可能写同一槽、状态需计数或 epoch 时，单个常量 1 不足以防 ABA/覆盖。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把状态位 copy 移到 data copy 之前或删除 `PIPE_MTE3` barrier，会破坏源码明确的数据先于状态发布顺序。

## 模板与实现对比

| 对比维度 | R-001 默认 rank 映射 | R-002 显式 rank table | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | optional rank table 缺省 | optional rank table 存在且合法 | Host 以存在性直接生成 bool template key |
| 模板与 dtype | `FFNToAttention<FP16/BF16,false>` | `FFNToAttention<FP16/BF16,true>` | dtype 由 binary specialization 提供；bool 模板只控制 rank 映射 |
| 循环与分核 | `y=aivId+t*aivNum`，每 token 搬一行 H | 同左 | 两条 Route 的 token 域与 tail 相同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 双槽 `xQueue_` + 32 B `statusBuf_`；无 dynamic UB；直接取 `sessionIds[y]` 对应 window | UB 同左；增加 `attnRankTable[worker]` GM 读取后选 window | optional table 不改变 UB 或 window layout |
| 内存总占用与复用 | `UB_requested=4H+32`；workspace 为 opaque lib-api workspace；queue slot Free 后复用 | 同左，外加每 token 4 B 逻辑 GM 读取 | Route 只改变目标 rank 寻址；physical UB rounding 固定仓不可见 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad`：GM→UB→remote window；Hcom hidden context 提供 window | 同左，先 DCCI/load rank table | 无 Vector 算术、SIMT、NDDMA 或 Cube；均为 AIV MTE + remote window |
| 同步、精度与 tail | MTE3 data→barrier→flag；FP16/BF16 原样；token 余数分核 | 同左 | rank table 不改变数据协议 |
| fallback/失败边界 | 非法公共参数 Host 失败 | 表非法 Host 失败 | Device 无互相 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | README/API 明确不支持 | 支持：Atlas A3 训练/推理，单“卡”语义为单 DIE | 950PR/950DT 明确不支持 |
| Host/Launcher | 无正式 route | `ascend910_93` OpDef/config；AIV blockDim；MTE server type | 无正式 route |
| Kernel/资源 | 虽 key 头声明 A2 tag 值，但无 selector 实例和公开支持，不可达 | `arch22/ffn_to_attention.cpp` 仅执行 `ArchTag==A3`；remote window + UB | 虽 key 头声明 950 tag 值，但无 selector 实例和公开支持，不可达 |

## 限制与失败边界

- 所有卡、不同网络层的 `group/worldSize/tokenInfoTableShape/tokenDataShape/HCCL_BUFFSIZE` 必须一致；FFNToAttention 通信域不能混入其他算子。
- `sessionIds/microBatchIds/tokenIds/expertOffsets/actualTokenNum` 的值域只在公开文档声明，固定 Host 不逐元素检查；越界会直接影响 HCCL rank 或远端 window 地址，Device 没有错误返回或安全 fallback。
- `CheckAndSetAttrs` 直接对两个 required ListInt 属性调用 `GetSize/GetData`，固定函数内没有先检查 ListInt 指针；其存在性依赖公开 ACLNN 入口与 OpDef required-attr 框架边界。
- Host 接受 `HS>=1024`，公开文档只支持 `HS>=1152`；正式 API 复用必须遵守较窄公开范围。文档还声明 K、shared expert 与 worker 数范围，但固定 Tiling 不从这些派生量逐项验证。
- `CalWinSize` 按未含 512 B info padding 的逻辑总量计算 MiB 桶，而 Device 以 `align_up(infoBytes,512)+dataBytes` 布局 window；逻辑总量紧邻 MiB 边界时，固定 Host guard 本身不足以证明物理范围安全。
- `sendTokenNum` 是 `uint32_t`，由 `uint64_t actualTokenNum_/aivNum_` 赋值；若调用者违反公开域并提供超大实际 token 数会窄化。`BS*E`、`BS*E*HS` 等中间量也有先在 `uint32_t` 计算的表达式，安全性依赖公开 `BS/HS/K/sharedExpertNum` 约束，源码没有独立溢出 guard。
- Host 仅记录 `totalUbSize`，未做 UB selector；`4H+32` B 是传给 `InitBuffer` 的请求量，仓外 primitive 的物理对齐规则不可见，不能把该请求量宣称为精确物理 UB 占用。
- `windowType`、无表 route 的 `A`、POD padding 没有显式初始化证据，且当前 body 不使用；迁移实现不得读取这些 inactive 值。
- `totalUbSize/totalWinSize/mc2InitTiling/mc2CcTiling1/workspaceGM` 不被当前 Device body读取；ACLNN 总 workspace 仍是 `GetLibApiWorkSpaceSize()` 的平台值，不能据 body 未读取而把它写成 0，也不能据这些字段臆造 Device collective、workspace slice 或额外 Kernel。
- outer POD 中 `Mc2InitTiling/Mc2CcTiling` 的定义与 generated inner bridge 不在固定仓；本卡只把它们标为 CANN 框架 ABI 边界，不推测 `sizeof/offsetof`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开支持/语义 | `mc2/ffn_to_attention/README.md` | 产品表、参数与约束 | A3-only、shape/通信域约束 |
| 公开接口文档 | `mc2/ffn_to_attention/docs/aclnnFFNToAttention.md` | 两段式原型 | 参数、确定性、公开值域 |
| 入口声明 | `mc2/ffn_to_attention/op_api/aclnn_ffn_to_attention.h` | `aclnnFFNToAttention*` | 公开 ABI |
| 入口实现 | `mc2/ffn_to_attention/op_api/aclnn_ffn_to_attention.cpp` | `CheckParams`、两段式函数 | 校验、inner bridge、MTE server type |
| OpDef | `mc2/ffn_to_attention/op_host/ffn_to_attention_def.cpp` | `ops::FFNToAttention` | dtype/format/AutoContiguous/A3/HCCL group |
| Host/Tiling | `mc2/ffn_to_attention/op_host/op_tiling/ffn_to_attention_tilling.cpp` | `FFNToAttentionTilingFunc` | guards、window/workspace、key、blockDim、Hcomm config |
| POD | `mc2/ffn_to_attention/op_kernel/ffn_to_attention_tiling.h` | `FFNToAttentionInfo/TilingData` | 算子字段与外层 POD |
| key selector | `mc2/ffn_to_attention/op_kernel/ffn_to_attention_tilling_key.h` | `ASCENDC_TPL_SEL` | 两个 rank-table bool specialization，仅 A3 可选 |
| Kernel entry | `mc2/ffn_to_attention/op_kernel/arch22/ffn_to_attention.cpp` | `ffn_to_attention` | A3 router 与模板实例 |
| Kernel body | `mc2/ffn_to_attention/op_kernel/ffn_to_attention.h` | `Init/GetWindowAddr/ReadTokenMetaData/Process` | 分核、UB、远端地址、同步、发布 |
| 任务生成 | `mc2/ffn_to_attention/op_graph/ffn_to_attention_gen_task.cpp` | `CalcParam/GenTask` | attached stream 资源和 V2 回调 |
| 任务改写公共依赖 | `mc2/common/op_graph/mc2_moe_gen_task_ops_utils.cpp` | `Mc2MoeGenTaskCallbackV2` | Hcom hidden input；FFNToAttention 不插 AICPU task |
| attached stream 公共依赖 | `mc2/common/op_graph/mc2_gen_task_ops_utils.cpp` | `CommonKFCMc2CalcParamFunc`、`InsertHiddenInputsForAicoreTask` | required attached stream 与 hidden Hcom 参数插入 |
| Device 公共依赖 | `mc2/common/op_kernel/moe_distribute_base.h` | `HcclOpResParam`、`ReadTokenMetaDataStruct` | window context 与元数据结构 |
| Device 同步 helper | `mc2/common/op_kernel/mc2_kernel_utils.h` | `SyncFunc<event>` | `FetchEventID→SetFlag→WaitFlag` 的首轮 `S_MTE3` 同步 |
| 二进制配置 | `mc2/ffn_to_attention/op_host/config/ascend910_93/ffn_to_attention_binary.json` | FP16/BF16 entries | A3 dtype/输入组合 |
| 构建接线 | `mc2/ffn_to_attention/op_host/CMakeLists.txt` | `add_modules_sources_with_soc` | API/Host/Kernel MC2 接线与 auto-sync off |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | bool key + `FFNToAttentionInfo` 字段推导 | launch 参数顺序、A3 tag、56 B 内部布局；仓外 Mc2 POD 不得重定义 | `FFNToAttentionTilingData`、`ffn_to_attention` |
| 调用顺序 | ACLNN check→inner→Tiling→hidden Hcom→AIV body | 第二段 MTE server 设置与同一 executor/stream；失败不得继续 | `aclnnFFNToAttention*`、`Mc2MoeGenTaskCallbackV2` |
| workspace/内存 | info 512 B 对齐、data 以 HS stride、UB 请求量 `4H+32` | lib-api workspace 不是 window；只复制 H，不得覆盖 HS-H；physical UB rounding 与 window 边界风险不可隐去 | `SetWorkSpace`、`CalWinSize`、`Init`、`Process` |
| 同步 | queue + 首轮 `S_MTE3` + `PIPE_MTE3` data-before-flag | flag 不得先于 data；`--cce-auto-sync=off` 下不能依赖编译器补同步 | `Process`、Host CMake |
| 精度与尾块 | FP16/BF16 原样 2 B 搬运；token 循环交错分核 | 无 cast/scale/round；`actualTokenNum<=Y` | `xCopyParams`、`YOffset` |
| 失败边界 | Host shape/dtype/format/window guard | 不得为越界索引、未支持平台或未知 key 发明 Device fallback | `CheckAndSetAttrs`、`CheckInput*`、`ASCENDC_TPL_SEL` |