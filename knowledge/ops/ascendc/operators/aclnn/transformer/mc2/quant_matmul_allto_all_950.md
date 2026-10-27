---
type: Operator
title: "QuantMatmulAlltoAll 950 ACLNN 算子知识"
description: "固定提交中 aclnnQuantMatmulAlltoAllV2 在 Ascend 950 的 KC/MX 量化、Host/Tiling、MIX Kernel 与 CCU/AI-CPU AlltoAll 全链路。"
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
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/arch35/kc_quant_matmul_allto_all_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/arch35/mx_quant_matmul_allto_all_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/arch35/matmul_allto_all_fit_balance_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/arch35/matmul_allto_all_fit_balance_tiling.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/matmul_allto_all_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/matmul_allto_all_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/matmul_allto_all_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/matmul_allto_all_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/kc_quant_matmul_allto_all_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/mx_quant_matmul_allto_all_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/matmul_allto_all_pipeline.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_templates/computation/math/mc2_vec_transpose.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_templates/communication/communicator/hccl_communicator.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_templates/computation/matmul/mc2_kc_quant_matmul.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_templates/computation/matmul/mc2_mx_quant_matmul.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/matmul_allto_all_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:41:43Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# QuantMatmulAlltoAll 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnQuantMatmulAlltoAllV2GetWorkspaceSize` / `aclnnQuantMatmulAlltoAllV2`；公开 ACLNN V2 |
| 计算语义 | KC：FP8 Cube + F32 两级 scale；MX：FP8/FP4 + E8M0 block scale；均接 rank transpose + AlltoAll |
| 输入输出 | KC FP8/F32-scale；MX FP8/FP4/E8M0-scale；y 为 FP16/BF16/F32，输出 `[rank*BS,H2/rank]` |
| 执行类型 | 一个 MIX_CV `MIX_AIC_1_2` kernel：AIC Cube、AIV VF/transpose、CCU 或 AI-CPU 通信节点 |
| Kernel 路线 | R-001 KC+CCU；R-002 KC+AI-CPU；R-003 MX+CCU；R-004 MX+AI-CPU |
| 核心资源 | libApi workspace + 两个 512B 对齐的 M*N*yBytes GM slice；nested L1/L0/UB 与 communicator handles |
| 来源闭包 | `source_closed`：API→KC/MX selector/FitBalance→POD/key→MIX entry→QBMM/transpose/communicator→构建闭合于 `be33d95...2540` |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公共形状语义与 A2 相同：MatMul 逻辑结果 `[BS,H2]` reshape 为 `[BS,rankSize,H2/rankSize]`，交换前两维，再 AlltoAll 得到 `[rankSize*BS,H2/rankSize]`。KC 公共公式为 `(x1@x2 + bias) * x1Scale * x2Scale`；当前 FP8 Kernel 在 Cube 中施加可选 F32 bias，再由 AIV VF 乘 per-channel 与 per-token scale。MX 按 K 方向 32 元素 block 累加 `x1@x2*x1Scale*x2Scale` 后加 bias；scale block 编码由 Cube-on-the-fly MatMul 库消费。

### 输入、输出与属性

| 模式/参数 | shape/layout | dtype | 950 约束 |
|---|---|---|---|
| KC `x1` | `[BS,H1]`, ND | FLOAT8_E4M3FN / FLOAT8_E5M2 | 必选；`x1Mode=3`，`BS>=1`，`1<=H1<=65535` |
| KC `x2` | `[H1,H2]` 或物理转置 | 两种 FP8 可独立组合 | `x2Mode=2`；连续或 API 可识别的 transpose view |
| KC scale | x1 `[BS]`, x2 `[H2]` | FLOAT32 | 必选 |
| MX `x1` | `[BS,H1]` | FP8 E4M3/E5M2 或 FLOAT4_E2M1 | `x1Mode=6` |
| MX `x2` | 物理 `[H2,H1]` transpose | FP8 组合；FP4 时两输入都 FP4 | `x2Mode=6`，`transposeX2=true` 必选 |
| MX scale | x1 `[BS,ceil(H1/64),2]`；x2 `[H2,ceil(H1/64),2]` | FLOAT8_E8M0 | K blockSize=32；FP4 K 为偶数且 `ceil(K/32)` 为偶数 |
| `bias` | `[H2]` | FLOAT32 | 950 可空；非空时 shape/dtype 固定 |
| `y` | `[rankSize*BS,H2/rankSize]` | FP16/BF16/F32 | 必选非空；`H2>=2` 且 `H2%rankSize=0` |
| `groupSize` | 16-bit packed M/N/K fields | INT64 | 0 字段由 shape 推导，最终只允许 `[1,1,32]` |
| `group` / rank | string / HCCL | — | group 长 `(0,128)`；rankSize 2/4/8/16；不跨超节点 |
| `commMode` | string | — | 只接受 `"ccu"` 或 `"ai_cpu"` |
| axes/保留量化项 | null 或 `[-1,-2]` | — | `commQuantMode=0`；commScale/offset 不支持；commQuantDtype 固定 undefined |

`BS*rankSize` 与 `H2` 不得超过 INT32_MAX；不支持空 tensor。950 只对 `x2` 的纯 transpose view 做 descriptor 重描述，不复制数据；其他任意非连续布局不受支持。

### 入口与版本

V2 比 V1 增加 `commMode`，公开文档、头文件与 op_api 均以 V2 为当前版本。V2 wrapper 调 base；GetWorkspaceSize 构造 executor/Inner op，执行段按保存的枚举将 Ascend 950 server type 设为 CCU 或 AICPU，再在用户 stream 发起 inner launch。

## Host/Launcher 执行流程

### 参数校验与前处理

base 先做公共非空/shape/K/output/scale/bias/dtype/format/axes/group/commMode 校验。950 如果发现 x2 为严格的二维 transpose view，就用 `TransX2Tensor` 复用同一 storage、交换 descriptor shape/stride 并把 `transposeX2` 置 true；若调用者又显式置 true 则拒绝，其他非连续模式也拒绝。KC/MX Tiling 再分别核验 quantMode、dtype、scale shape、MX groupSize 与 transpose 条件。

### 参数、Workspace 与生命周期

KC/MX 都从平台信息取 AIC 核数与 libApi workspace，调用 FitBalance 得到长/短 M tile，再分别生成 main/tail 两份 QuantBatchMatmulV3 Tiling 和 HCCL Tiling。`mmResultLen=AlignUp(M*N*sizeof(y),512)`，`permuteLen=mmResultLen`；总 workspace 为 `libApiWorkspace + mmResultLen + permuteLen`。Device 以传入的 `workspaceGM` 为 MatMul 临时结果起点，以 `workspaceGM+mmResultLen` 为 transpose 结果；这两段从 launch 到所有通信完成前不可释放/复用。

### 启动图与失败传播

```text
V2GetWorkspaceSize -> base -> InnerGetWorkspaceSize
  -> KC(priority 1, mode 3/2) or MX(priority 2, mode 6/6)
  -> FitBalance -> QBMM main/tail tiling + HCCL tiling + key/blockDim/workspace
V2(stream,executor) -> base(CCU/AICPU) -> Inner
  -> matmul_allto_all_apt<key> -> route body -> MatmulAlltoAllPipeLine
```

任何 API/Tiling/HCCL builder/QBMM tiling/launch 错误都上抛；执行段 inner 失败转换为 `ACLNN_ERR_INNER`。四 route 之间不做 Device 动态 fallback。

## Route 与参数设计

### 分核与资源计算

KC selector `IsCapable` 只认 `(x1Mode,x2Mode)=(3,2)`，MX 只认 `(6,6)`，二者不重叠。FitBalance 估算 Cube/通信时间；KC 计算时间乘 1.43 计入 AIV 开销。输入 A/B 超 128MiB L2 时最小 tile 加倍；小 M、低并行或 overhead shape 会把最小 tile 抬到约三块；长 tile 对齐 256。AI-CPU 模式若 MM 或通信估时小于 80us 则不切，且 `M>10*N` 时最多三块。最终 main 是 long/short 中排在前且存在的一组，另一组作为 tail；`blockDim` 为 AIC 核数，schedule mode=1。

### Route 选择

| 顺序 | Route | key | 精确 selector | 不适用 |
|---:|---|---|---|---|
| 1 | R-001 KC+CCU | `(KC_QUANT_MODE,transposeX2,FP32 bias,CCU)` | mode 3/2、KC dtype/scale/bias 合法、`commMode=ccu` | R-002 或失败 |
| 2 | R-002 KC+AI-CPU | `(KC_QUANT_MODE,transposeX2,FP32 bias,AICPU)` | 同 KC 条件、`commMode=ai_cpu` | 失败 |
| 3 | R-003 MX+CCU | `(MX_QUANT_MODE,true,FP32 bias,CCU)` | mode 6/6、MX dtype/scale/groupSize 合法、transposeX2、`ccu` | R-004 或失败 |
| 4 | R-004 MX+AI-CPU | `(MX_QUANT_MODE,true,FP32 bias,AICPU)` | 同 MX 条件、`ai_cpu` | 失败 |

~~~text
Ascend 950 + 公共 API/shape/rank/commMode 校验
  -> quant mode (3,2) -> KC capability/dtype/scale/bias
       -> commMode=ccu -> R-001
       -> commMode=ai_cpu -> R-002
  -> quant mode (6,6) -> MX capability/dtype/E8M0/groupSize/transposeX2
       -> commMode=ccu -> R-003
       -> commMode=ai_cpu -> R-004
  -> selector 不命中或任一校验失败 -> 失败，不 launch
~~~

### 模板/实现执行矩阵

| Route | compute 实例 | AIC/AIV | transpose | communicator |
|---|---|---|---|---|
| R-001 | `MC2KCQuantMatmul<Mc2QuantBmmPertokenRegbaseKernel<FP8,...,F32>>` | AIC Cube + AIV KC dequant | AIV | CCU server |
| R-002 | 同 KC 实例 | 同上 | AIV | AI-CPU server |
| R-003 | `MC2MXQuantMatmul<MatMulASWKernel<FP8/FP4,E8M0,F32,Y>>` | AIC Cube，AIV 在外层 transpose | AIV | CCU server |
| R-004 | 同 MX 实例 | 同上 | AIV | AI-CPU server |

### 隐藏性能 Guard

- `MC2AlltoAllContext`/communicator 的 task handle 数组固定 63 项；当前 body 以 `tileCnt+tailCnt` 传 `PrepareAll`，源码未见显式 `taskCnt<=63` guard，Tiling 切块策略必须隐含守住该上限。
- transpose 以 `M` 方向 64、`N/rank` 方向 256 切块，使用核数 `min(blockNum*2,totalTiles)`；过小 shape 会空闲部分 AIV。
- KC FitBalance 的 1.43 系数、AI-CPU 80us 不切阈值、`M/N>10` 三块上限都是选择性能 guard，而非语义约束。
- MX FP4 的字节 offset 用 pack factor 2 计算；用普通元素字节公式会把 main/tail x1 起点放错。

### POD/ABI

| 结构 | 字段 | 类型 | offset/size/alignment | 单位 | producer→consumer | 边界 |
|---|---|---|---|---|---|---|
| `MatmulAlltoAllTilingInfo` | `rankDim` | uint32 | 0/4/align8 outer | rank | HCCL query→all stages | 2/4/8/16 |
| 同上 | `tileM` | uint32 | 4/4 | row | FitBalance→main compute/trans/comm | main extent |
| 同上 | `tileCnt` | uint32 | 8/4 | task | FitBalance→pipeline | main count |
| 同上 | `tailM` | uint32 | 12/4 | row | FitBalance→tail | 0 or tail extent |
| 同上 | `tailCnt` | uint32 | 16/4 | task | FitBalance→pipeline | 0/1 or cut count |
| 同上 | `rankM` | uint32 | 20/4 | row | shape→offset | BS |
| 同上 | `rankN` | uint32 | 24/4 | column | shape→transpose/comm | H2 |
| 同上 | `rankK` | uint32 | 28/4 | reduction | shape→compute | H1 |
| 同上 | `mmResultLen` | uint64 | 32/8 | byte | workspace tiling→route Init | 512B aligned |
| 同上 | `permuteLen` | uint64 | 40/8 | byte | workspace tiling→accounting | 等于 mmResultLen |
| 同上 | `biasLen` | uint32 | 48/4 | element | bias shape→compute | 0 or H2 |
| 同上 | `aicCoreNum` | uint32 | 52/4 | core | platform→compute | 平台 AIC 数 |
| 同上 | `hcclDataType` | uint64 | 56/8 | enum | y dtype→comm | FP16/BF16/F32 对应 HCCL enum |
| outer | `mc2InitTiling` | `Mc2InitTiling` | 外部 SDK ABI | — | HCCL builder→communicator | 固定仓无定义 |
| outer | `mc2CcTiling` | `Mc2CcTiling` | 外部 SDK ABI | — | HCCL builder→communicator | 固定仓无定义 |
| outer | `matmulAlltoAllTilingInfo` | 上述 struct | 位于两个外部字段后 | — | Host→route | outer offset 不推测 |
| outer | `mc2Quant...Params` | nested QBMM POD | 外部/3rd 定义 | — | main QBMM Tiling→compute | baseM/N/K、depth、shape、quant/group fields |
| outer | `mc2Quant...TailParams` | nested QBMM POD | 同上 | — | tail QBMM Tiling→compute | tailCnt=0 时不消费 |

本地 `MatmulAlltoAllTilingInfo` 的 offset/size 可由字段定义闭合；outer 前缀 HCCL struct 与 nested `TCubeTiling` 由 SDK `kernel_tiling` 提供，固定仓不能给出完整 outer `offsetof/sizeof`，不以推测数字填充。nested QBMM 的 producer/consumer 同时引用同一 `Mc2QuantBatchMatmulV3TilingDataParams`，其中 shape/batch/base/step/depth、L2/cache、sliding-window字段由第三方 QBMM tiler逐字段写入并由 current-instance 读取。

### Route 源码闭包

| Route | selector/producer | key/POD | entry | body/直接依赖 | 构建 |
|---|---|---|---|---|---|
| R-001 | KC Tiling + FitBalance | arch35 key/data | `matmul_allto_all_apt.cpp` KC+CCU macro | KC arch35→KC compute→QBMM mix→transpose→HCCL communicator | op_def/CMake |
| R-002 | KC Tiling + FitBalance | 同上，AICPU key | 同上 | 同上，仅 communicator server template 改变 | 同上 |
| R-003 | MX Tiling + FitBalance | arch35 key/data | MX+CCU macro | MX arch35→MX compute→Cube-on-the-fly→transpose→communicator | 同上 |
| R-004 | MX Tiling + FitBalance | 同上，AICPU key | 同上 | 同上，仅 server template 改变 | 同上 |

## Kernel 实现

### 公共执行流程

一个 MIX entry 依据 key 实例化 KC/MX、x2 transpose 与 CCU/AICPU server。KC/MX body 把 workspace 分成 MatMul ND 结果和 rank-transpose 结果，分别构造 main/tail compute、transpose、comm context，交给 `MatmulAlltoAllPipeLine`。每个通信 task 的 send 指向 transpose slice，recv 直接指向 y；`sendCount=tileM*N/rank`，`strideCount=M*N/rank`，HCCL dtype 来自 POD。

### Route R-001：KC + CCU

#### 适用条件

##### 选择证据

KC Tiling priority 1 命中 3/2，生成 `KC_QUANT_MODE` 与 CCU key；entry 的 `MC2KCQuantMatmul` 宏实例化 FP8 input、F32 scale/bias/accumulator 与 y 模板，server type 为 CCU。

#### Device 数据流

##### 函数级调用链

```text
entry -> KcQuantMatmulAlltoAllArch35::Process
  -> MatmulAlltoAllPipeLine::ProcessKC
    -> Mc2QuantBmmPertokenRegbaseKernel::Process
    -> MC2VecTranspose::Process
    -> HcclCommunicator::Process(CCU)
```

##### 源码等价伪码

```text
PrepareAll(taskCnt)
for task: KC MIX computes tile into mmWorkspace; AIV multiplies x2/x1 scales
SyncAll; AIV transpose [tileM,N] -> [rank,tileM,N/rank]
SyncAll; commit CCU AlltoAll task; SyncAll
End waits outstanding tasks and finalizes communicator
```

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲 | offset/extent | producer→consumer | 复用/对齐 |
|---|---|---|---|
| x1 tile | `task*tileM*K*sizeof(x1)` | GM→KC QBMM | tail 从 `tileCnt*tileM` 起 |
| x1Scale | `task*tileM*4` | GM→AIV VF | per-token |
| x2/scale/bias | 全局固定 | GM→Cube/AIV | 跨 task 复用 |
| MM workspace | `workspace + task*tileM*N*sizeof(y)` | QBMM→transpose | 总 slice `AlignUp(M*N*ybytes,512)` |
| transpose workspace | `workspace+mmResultLen + task*tileM*(N/rank)*ybytes` 每 rank布局 | AIV transpose→HCCL send | 第二个同尺寸 slice |
| y | task stride `M*N/rank` | HCCL recv | recv 直接落最终 rank-major y |
| L1/L0/UB | nested `TCubeTiling` 的 baseM/N/K、depth 与 QBMM VF block | Cube/AIV current instance | 运行 shape 决定，无单一常数 |

##### 分层内存布局

~~~mermaid
flowchart LR
  X["GM FP8 x1/x2"] --> CUBE["AIC Cube -> L0C INT32"]
  CUBE --> VUB["AIV UB：INT32->F32，x2Scale*x1Scale，cast"]
  VUB --> MMWS["GM MM workspace [tileM,N]"]
  MMWS --> TUB["AIV UB transpose 64x256"]
  TUB --> PWS["GM rank-major workspace"]
  PWS --> H["HCCL CCU AlltoAll"]
  H --> Y["GM y"]
~~~

##### 字节账、交接边与复用顺序

~~~text
workspace_total = libApiWorkspace + AlignUp(M*N*yBytes,512) * 2
KC_VF_UB = ceil(baseM/2)*baseN*4 + baseN*4
           + Align32(ceil(baseM/2)*4)
           + 2*ceil(ceil(baseM/2)/splitNum)*baseN*yBytes
L1/L0 = nested TCubeTiling(baseM,baseN,baseK,step,depth)，固定仓无单一常量
~~~

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| QBMM tile | KC AIC/AIV current instance | task*tileM*N*yBytes | y dtype ND | transpose | QBMM flags + SyncAll(false) | mmResultLen slice | main/tail 各用自身 POD |
| rank-major tile | AIV transpose | rank/task slice，tileM*N*yBytes | y dtype | communicator | SyncAll(true) | permuteLen slice | N 可整除 rank |
| HCCL handle | communicator Prepare/Process | task index | control handle | End | outstanding wait | 单 kernel | taskCnt 不得超过固定数组能力 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive | 精度/tail |
|---|---|---|
| KC AIC↔AIV | QBMM current-instance cross-core flags | FP8 Cube；bias 在非 INT8实例的 MM 中，AIV F32 乘两 scale，最后 cast y |
| compute→transpose | `SyncAll<false>` | main/tail 使用各自 QBMM tiling |
| transpose→comm | `SyncAll<true>` | AIV 以 64x256 tile，DataCopyPad 保护边界 |
| comm | communicator task handle + End wait | y dtype 不变，sendCount 仅有效 tile |

#### 关键实现

##### 符号证据表

KC arch35 的 `Init/Process` 给出 main/tail 地址；`qbmm_mix_online_dynamic.h::VectorCompute` 给出 F32 scale/cast；pipeline 的 `ProcessKC` 给出 barrier/commit 次序。

### Route R-002：KC + AI-CPU

#### 适用条件

##### 选择证据

与 R-001 相同 KC 条件，仅 `commMode="ai_cpu"` 令 key 与 executor server type 选择 AICPU；FitBalance 同时启用 80us no-cut 与 M/N 三块 guard。

#### Device 数据流

##### 函数级调用链

调用链与 R-001 相同，`HcclCommunicator` 的 server template 改为 AI-CPU。

##### 源码等价伪码

```text
KC compute -> AIV transpose -> commit AI-CPU-backed AlltoAll -> wait/finalize
```

#### 内存与数据搬运

##### 资源与搬运表

所有 pointer/extent 与 R-001 相同；通信控制面不同，不改变 send/recv 数据平面或 y layout。

##### 分层内存布局

~~~mermaid
flowchart LR
  X["GM FP8 x1/x2"] --> CV["KC MIX：AIC Cube + AIV VF"]
  CV --> MMWS["GM MM workspace"]
  MMWS --> TV["AIV transpose"]
  TV --> PWS["GM rank-major workspace"]
  PWS --> H["HCCL AI-CPU AlltoAll"]
  H --> Y["GM y"]
~~~

##### 字节账、交接边与复用顺序

~~~text
workspace_total = libApiWorkspace + 2*AlignUp(M*N*yBytes,512)
compute/transpose 字节式与 R-001 相同；AI-CPU 只改变 communicator server 与 Host 切块结果
~~~

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| MM workspace | KC MIX | task*tileM*N*yBytes | y dtype ND | AIV transpose | SyncAll(false) | 当前 launch | 按 AI-CPU tile POD 取址 |
| permute workspace | AIV transpose | rank/task slice | rank-major | HCCL | SyncAll(true) | 当前 launch | 小任务 no-cut 时只有一 task |
| y slice | AI-CPU communicator | task*M*N/rank stride | y dtype | final y | End wait | executor stream | outstanding 完成前不可释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

KC MIX 同步与数值完全相同；AI-CPU 小任务可能由 Host 不切块，tailCnt 可为 0。

#### 关键实现

##### 符号证据表

base API 的 server-type 分支、KC Tiling 的 commMode key 与 entry AICPU macro 共同闭合该 route；不存在独立 compute body。

### Route R-003：MX + CCU

#### 适用条件

##### 选择证据

MX Tiling priority 2 命中 6/6，检查 E8M0 scale/groupSize/FP4 pack 与 `transposeX2=true`，生成 MX+CCU key；entry 实例化 `MC2MXQuantMatmul<MatMulASWKernel<...,E8M0,F32,Y>>`。

#### Device 数据流

##### 函数级调用链

```text
entry -> MxQuantMatmulAlltoAllArch35::Process
  -> MatmulAlltoAllPipeLine::ProcessDefault
    AIC: qbmm_cube_on_the_fly MatMulASWKernel
    AIV: MC2VecTranspose
    HcclCommunicator(CCU)
```

##### 源码等价伪码

```text
PrepareAll
for task:
  AIC set E8M0 scale A/B, optional F32 bias, Iterate and write MM workspace
  AIC signals 8 then 9; AIV waits 9, transposes tile, SyncAll, commits CCU task
End waits all outstanding and Finalize
```

#### 内存与数据搬运

##### 资源与搬运表

| 缓冲 | offset/extent | 特有事实 |
|---|---|---|
| x1 | `ComputeByteOffset<T>(task*tileM*K)` | FP4 offset 除 pack factor 2 |
| x1Scale | `task*tileM*ceil(K/64)*2` bytes | E8M0 两 scale/64 K 对应 32-block |
| x2/x2Scale | 固定 transposed base | x2Scale `[N,ceil(K/64),2]` |
| MM/trans/y | 与 KC 相同两 workspace slice与 comm stride | 输出 dtype FP16/BF16/F32 |
| L1/L0 | `TCubeTiling` 动态 base/depth | AIC-only compute；AIV不进入 MX QBMM body |

##### 分层内存布局

~~~mermaid
flowchart LR
  X["GM FP8/FP4 x1/x2 + E8M0 scales"] --> L1["AIC MatmulWithScale：L1/L0"]
  L1 --> MMWS["GM y-dtype MM workspace"]
  MMWS --> TUB["AIV UB transpose 64x256"]
  TUB --> PWS["GM rank-major workspace"]
  PWS --> H["HCCL CCU AlltoAll"]
  H --> Y["GM y"]
~~~

##### 字节账、交接边与复用顺序

~~~text
workspace_total = libApiWorkspace + 2*AlignUp(M*N*yBytes,512)
x1_task_offset_bytes = task*tileM*K*sizeof(T)/PackFactor(T)
x1Scale_task_offset_bytes = task*tileM*ceil(K/64)*2
L1/L0 = nested TCubeTiling 动态值；本 route 的 MX body 不分配 QBMM AIV 去量化 UB
~~~

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| MX output tile | MatMulASWKernel AIC | task*tileM*N*yBytes | y dtype ND | AIV transpose | flags 8/9 | mmResultLen slice | FP4 offset 除 pack factor 2 |
| rank-major tile | AIV transpose | rank/task slice | y dtype | CCU communicator | SyncAll(true) | permuteLen slice | DataCopyPad 只提交真实 tail |
| scale address | Host shape/POD | x1 按 task，x2 全局 | E8M0 | MatmulWithScale | AIC program order | 输入 tensor | scale shape 满足 64K 两段编码 |

#### 同步、精度与尾块

##### 同步与数值边界表

AIC 用 FIX flags 8/9 通知两个 AIV子块；AIV transpose 完成后 SyncAll(true) 再 commit。MX scale 由 MatMul scale policy 在 Cube 路径消费，F32 bias 由 MatMul optional bias 接口加入；tail 使用独立 nested tail Tiling。

#### 关键实现

##### 符号证据表

MX arch35 的 `ComputeByteOffset` 保护 FP4 packing；`qbmm_cube_on_the_fly.h` 在 AIV early-return、AIC 设置 scale/bias并直接 `GetTensorC` 到 GM；pipeline default 给出 8/9 flag。

### Route R-004：MX + AI-CPU

#### 适用条件

##### 选择证据

MX 的 dtype/group/transpose 条件与 R-003 相同，仅 `commMode="ai_cpu"` 选择 AICPU key/server。

#### Device 数据流

##### 函数级调用链

与 R-003 相同，communicator server template 改为 AI-CPU。

##### 源码等价伪码

```text
MX scaled Cube -> AIV transpose -> AI-CPU-backed AlltoAll -> y
```

#### 内存与数据搬运

##### 资源与搬运表

与 R-003 完全相同；只替换通信控制面。AI-CPU FitBalance guard 可能改变 tileCnt/tileM，因此 main/tail offset 始终以 POD 为准。

##### 分层内存布局

~~~mermaid
flowchart LR
  X["GM FP8/FP4 + E8M0"] --> CUBE["AIC scaled Cube"]
  CUBE --> MMWS["GM MM workspace"]
  MMWS --> TV["AIV transpose"]
  TV --> PWS["GM rank-major workspace"]
  PWS --> H["HCCL AI-CPU AlltoAll"]
  H --> Y["GM y"]
~~~

##### 字节账、交接边与复用顺序

~~~text
workspace_total = libApiWorkspace + 2*AlignUp(M*N*yBytes,512)
FP4/E8M0 地址式与 R-003 相同；AI-CPU guard 只改变 tileM/tileCnt/tailM/tailCnt
~~~

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| MX output | AIC | POD task slice | y dtype | AIV transpose | flags 8/9 | mmResultLen | tail 用独立 nested POD |
| permuted output | AIV | rank/task slice | rank-major | AI-CPU communicator | SyncAll(true) | permuteLen | 按真实 sendCount |
| final y | communicator | task stride M*N/rank | y dtype | caller | End wait | executor stream | 完成前不可回收 workspace/executor |

#### 同步、精度与尾块

##### 同步与数值边界表

8/9 flag、AIV barrier、MX scale/bias 与 tail 规则均不变；End 必须等待 AI-CPU communicator outstanding task。

#### 关键实现

##### 符号证据表

API server-type 分支、MX Host key 与 entry AICPU macro 是 route 差异证据；没有复制 MX compute 实现。

## 算子特定优化

### Trick T-001：FitBalance 以预测时间生成长短 tile

#### 触发与源码依据

- 适用 Route：R-001 至 R-004。
- 触发条件：Ascend 950 Host Tiling 进入 FitBalance；KC 另计 AIV 开销。
- Golden 来源：[matmul_allto_all_fit_balance_tiling.cpp:40-46](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_host/op_tiling/arch35/matmul_allto_all_fit_balance_tiling.cpp#L40-L46)

官方源码的决定性连续片段：

~~~cpp
double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
// 当 quantMode 为 KC_QUANT 时，使用的模版为mix，需要额外考虑vector的开销
if (quantMode_ == QuantMode::KC_QUANT) {
    totalMatmulTime *= COMPUTE_TIME_SCALE_FACTOR;
}
double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
~~~

#### 机制与设计原因

Host 同时估计 MatMul 与通信时间，用反函数求另一阶段的 tileM，使流水线相邻 task 更接近平衡；计算占优时 short tile 放后，否则放前。

#### 迁移边界

- 适用：本卡 KC/MX 的 CCU 与 AI-CPU route。
- 前提：性能模型、baseM 对齐、rank 拓扑和 commMode 与 Host Tiling 一致；KC 估时系数为 1.43。
- 失效：A2 使用离散 shape map/pValue，不使用此 FitBalance。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把估算的 long/short tile 当成运行期自适应会错误：它们在 Host 已固化为 main/tail POD；绕过 AI-CPU 80us no-cut 与 tile 上限会改变通信资源假设。

### Trick T-002：AIV rank transpose 与通信流水化

#### 触发与源码依据

- 适用 Route：R-001 至 R-004。
- 触发条件：compute tile 完成，N 可整除 rank，AIV 进入 MC2VecTranspose。
- Golden 来源：[mc2_vec_transpose.h:114-118](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_templates/computation/math/mc2_vec_transpose.h#L114-L118)

官方源码的决定性连续片段：

~~~cpp
DataCopyPadExtParams<transposeDataType> padExtParams{true, 0, 0,
                                                     *reinterpret_cast<transposeDataType *>(uint8_t(0))};

DataCopyPad<transposeDataType>(vecInBuf, tranposeGm_[srcGmOffset], loadGm2UbParams, padExtParams);
~~~

#### 机制与设计原因

MatMul 先写连续 `[tileM,N]`，AIV 按 rank、M64、innerN256 分块转为 rank-major workspace，随后 communicator 可用连续 sendCount 直接发送。

#### 迁移边界

- 适用：KC/MX、CCU/AI-CPU 共用的 arch35 pipeline。
- 前提：N%rankSize=0、两个 workspace slice 足额，AIV 等待对应 compute tile。
- 失效：A2 epilogue 直接写 peer window，不经过这一 rank transpose。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

让 communicator 在 SyncAll(true) 前读取 permute workspace，会发送尚未完成的 rank-major tile；把 source stride 当成 N/rank 会读错原始行主序矩阵。

### Trick T-003：MX FP4 pack-aware main/tail offset

#### 触发与源码依据

- 适用 Route：R-003、R-004。
- 触发条件：MX 且 x1/x2 为 FLOAT4_E2M1，main/tail 构造输入地址。
- Golden 来源：[mx_quant_matmul_allto_all_arch35.h:39-42](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_allto_all/op_kernel/arch35/mx_quant_matmul_allto_all_arch35.h#L39-L42)

官方源码的决定性连续片段：

~~~cpp
template <typename T>
__aicore__ inline uint64_t ComputeByteOffset(uint64_t elementCount)
{
    return elementCount * sizeof(T) / PackFactor<T>::value;
}
~~~

#### 机制与设计原因

`ComputeByteOffset` 对 FP4 两元素/字节做除 2，保证 tile/tail x1 起点与真实压缩存储一致；scale offset则按 `ceil(K/64)*2` 独立计算。

#### 迁移边界

- 适用：R-003/R-004 的 FP4 specialization。
- 前提：两输入同为 FP4，K 与 scale shape 已通过 Host 偶数/分块约束。
- 失效：FP8 的 PackFactor 为 1；KC 不使用 MX scale 地址。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

对 FP4 直接使用 `elements*sizeof(T)`，或把 scale offset也除以 2，会分别导致 x1 tile 起点或 E8M0 scale 起点错误；非法 K 应由 Host 拒绝。

## 模板与实现对比

| 维度 | KC | MX | CCU vs AI-CPU |
|---|---|---|---|
| quant selector | 3/2 | 6/6 | 不改变 quant |
| compute | MIX QBMM，AIC+内部 AIV dequant | AIC Cube-on-the-fly scale | body 不变 |
| bias | KC FP8 Cube bias后乘 scale | MX scaled accumulation后 bias | 不变 |
| x2 transpose | true/false key | public route只 true | 不变 |
| pipeline | `ProcessKC` 多次全核同步 | default 8/9 flags | communicator server 不同 |
| workspace | 两个全输出尺寸 slice | 同左 | 同左，切块可能不同 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 公开 V2 | 支持 | 不支持 | 本卡支持 |
| quant | INT8 KC | — | FP8 KC、FP8/FP4 MX |
| rankSize | 2/4/8 | — | 2/4/8/16 |
| commMode | aiv | — | ccu/ai_cpu |
| 数据交换 | 反量化直写 peer window | — | workspace transpose + HCCL task |

## 限制与失败边界

- 仅覆盖固定仓固定提交 Ascend 950 V2；A3 公开矩阵明确不支持。
- rankSize 2/4/8/16，`H2%rankSize=0`；不支持空 tensor、跨超节点或同通信域并发。
- MX 强制 x2 transpose，FP4 两输入必须同为 FP4并满足 K/scale pack；groupSize最终只能 `[1,1,32]`。
- 只支持 x2 的规范 transpose view；其他非连续 tensor 拒绝。
- task handle 固定 63 项但未见显式 guard，是源码边界；不得在外部擅自增加 tile 数。
- outer HCCL/TCube ABI 属外部 SDK，固定仓缺少完整定义，不推测 offset/sizeof。
- 未执行编译、测试、validator、review、真机数值或性能验证。

## 源码索引

### 链路索引

| 阶段 | 文件 | 关键符号/用途 |
|---|---|---|
| 公开约束 | `docs/aclnnQuantMatmulAlltoAllV2.md` | 950 KC/MX 公式、dtype/shape/platform |
| API 声明 | `op_api/aclnn_quant_matmul_allto_all_v2.h` | V2 两段式声明 |
| API 实现 | `op_api/aclnn_quant_matmul_allto_all_v2.cpp` | V2 到 base 转发 |
| API base | `op_api/quant_matmul_allto_all_base.cpp` | 校验、x2 view、server type、错误传播 |
| API util | `op_api/matmul_allto_all_util.cpp` | axes/group/commMode 与 transpose view helper |
| KC Tiling | `op_host/op_tiling/arch35/kc_quant_matmul_allto_all_tiling_base.cpp` | KC capability/QBMM/HCCL/key |
| MX Tiling | `op_host/op_tiling/arch35/mx_quant_matmul_allto_all_tiling_base.cpp` | MX dtype/group/transpose/QBMM/key |
| 公共 Tiling | `op_host/op_tiling/matmul_allto_all_tiling_base.cpp` | 平台、workspace 与 main/tail producer |
| 公共校验 | `op_host/op_tiling/common/matmul_allto_all_util_tiling.cpp` | rank/shape/dtype/commMode |
| 平衡实现 | `op_host/op_tiling/arch35/matmul_allto_all_fit_balance_tiling.cpp` | 长短 tile、1.43 与性能 guards |
| 平衡声明 | `op_host/op_tiling/arch35/matmul_allto_all_fit_balance_tiling.h` | AICPU 最大 tile 与公式化 fallback |
| POD | `op_kernel/arch35/matmul_allto_all_tiling_data.h` | 本地 ABI 与 nested POD |
| key | `op_kernel/arch35/matmul_allto_all_tiling_key.h` | KC/MX、transpose、commMode key |
| entry | `op_kernel/arch35/matmul_allto_all_apt.cpp` | MIX macro router |
| KC body | `op_kernel/arch35/kc_quant_matmul_allto_all_arch35.h` | KC main/tail 地址与 context |
| MX body | `op_kernel/arch35/mx_quant_matmul_allto_all_arch35.h` | MX pack-aware main/tail 地址 |
| pipeline | `op_kernel/arch35/matmul_allto_all_pipeline.h` | KC/default 顺序和同步 |
| transpose | `mc2/common/.../mc2_vec_transpose.h` | 64x256 AIV transpose |
| communicator | `mc2/common/.../hccl_communicator.h` | Prepare/commit/wait/finalize |
| KC wrapper | `mc2/common/op_kernel/mc2_templates/computation/matmul/mc2_kc_quant_matmul.h` | KC current instance |
| MX wrapper | `mc2/common/op_kernel/mc2_templates/computation/matmul/mc2_mx_quant_matmul.h` | MX current instance |
| KC QBMM | `mc2/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h` | KC Cube/AIV VF |
| MX QBMM | `mc2/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h` | MX scaled Cube |
| 注册 | `op_host/matmul_allto_all_def.cpp` | Ascend 950 op file |
| 构建 | `CMakeLists.txt` | 编译接线 |

### 实现复用检查

| 可复用项 | 不得改变 | 证据 |
|---|---|---|
| ACLNN | executor handle、x2 view descriptor、server type | base/util |
| selector | KC/MX quantMode 不交叉；commMode进入 key | 两套 Tiling/key |
| ABI | 本地 info 字段与 nested producer/consumer同型；不重建 SDK POD | tiling data |
| workspace | 两个 512B aligned 全输出 slice及main/tail offset | Tiling + KC/MX body |
| KC 精度 | Cube bias与 AIV F32 scale/cast顺序 | QBMM mix |
| MX packing | FP4 byte offset与E8M0 scale offset | MX body/Cube-on-the-fly |
| 同步 | KC barriers、MX 8/9 flags、communicator End | pipeline/communicator |