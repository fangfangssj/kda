---
type: Operator
title: "FlashAttentionScoreGrad A2/A3 ACLNN 算子知识"
description: "说明 aclnnFlashAttentionScoreGradV3 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV3.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_api/flash_attention_score_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_bngs1s2_b.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_ngs1s2_bn.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_det.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_bngs1s2_b.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_ngs1s2_bn.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2_basic_det.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_empty_tensor.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_pre.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_post.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T00:00:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# FlashAttentionScoreGrad A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnFlashAttentionScoreGradV3GetWorkspaceSize` + `aclnnFlashAttentionScoreGradV3`，公开二段式 ACLNN |
| 计算语义 | 从 Q/K/V、dY、前向 softmax 统计量和 attention 输出计算 dQ/dK/dV；V3 相比 V2 新增 sink 输入和 dSink 输出。 |
| 输入输出 | 公开 V3 仅 BSH/SBH/BNSD/BSND；主 dtype FP16/BF16/FP32，mask/PSE/dropout/sink 可选。共享内部 TND 模板不属于该入口闭包。 |
| 执行类型 | 非空计算是单 launch `MIX_CV`，entry 固定 `KERNEL_TYPE_MIX_AIC_1_2`；AIC Matmul 与 AIV softmax-grad/后处理经 GM workspace 和 flag 交接。全空 API 终点不 launch，key 0 终点是 AIV clear。 |
| Kernel 路线 | R-001 空域；R-002 确定性/Basic；R-003 B/BN 小工作集；R-004 SameAB/sink；R-005 BN2/通用 S1S2 |
| 核心资源 | GM user workspace、system workspace、L1/L0A/L0B/L0C、UB、Matmul 对象、跨核/pipe flag |
| 来源闭包 | `source_closed`：V3 公开文档/API→L0→arch22 template registry/POD→MIX_CV entry/body→CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `A=QKᵀ`。`pseType=1` 时 `S=Mask(scale·(A+pse))`，其它值时 `S=Mask(scale·A+pse)`；随后 `P=Dropout(Softmax(S),keepProb)`、`Y=PV`。反向计算 `dV=PᵀdY`、`dP=dYVᵀ`、`dS=SoftmaxGrad(dP,P)`、`dQ=scale·dSK`、`dK=scale·dSᵀQ`。V3 的 sink 参与 softmax max/sum，并输出归约的 dSink；V3 文档明确 A2/A3 支持该新增语义。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| query/keyIn/value/dy | BSH/SBH/BNSD/BSND | FP16/BF16/FP32 | 必选；Q/K/V dtype 一致，`headNum>0`；V3 文档和原型均没有 actual-sequence 参数。 |
| attentionIn/softmaxMax/softmaxSum | 前向结果/统计量 | 主输入族/FP32 | attentionIn 必选；统计量 shape/dtype 由 API/Tiling 检查。 |
| pseShift/dropMask/attenMask | 广播 PSE、bit/byte mask、attention mask | FP16/BF16/FP32、UINT8、BOOL/UINT8 | 可选；keepProb<1 激活 dropout 路线。 |
| sinkIn/dsinkOut | 1-D sink | 文档支持类型；dSink 为对应梯度 | V3 新增；A2/A3 支持，950 的 V3 暂不支持但本卡不覆盖 950。 |
| prefix/qStartIdx/kvStartIdx | 一维数组 | INT64 | 由 L0 转为 ND tensor并进入 Tiling dependency；不能据此把 V3 扩展为 TND/变长入口。 |
| scale/keepProb/tokens/headNum/layout/innerPrecise/sparseMode/pseType | 标量 | double/int64/string | keepProb (0,1]；sparseMode 决定有效 S1×S2 窗。 |
| dq/dk/dv/dpse/dsink | 对应输入/广播 shape | 主输出族/dsink 约定 | dq/dk/dv 必选；可选输出按输入启用。 |

### 入口与版本

V3 文档的产品表把 A2/A3 标为支持，V4 文档把 A2/A3 明确标为不支持，因此本卡选择 A2/A3 实际支持的最新版本 V3，而不是现有旧卡曾写的 V4。V3 第一段复用共享 `FlashAttentionScoreGradV5GetWorkspace`/L0 节点构图，第二段调用 `CommonOpExecutorRun`。

## Host/Launcher 执行流程

### 参数校验与前处理

V3 第一段创建 executor，检查必选输入/输出、`headNum>0`、dtype/shape/layout。共享 workspace helper 连续化必选和可选张量，必要时 reshape、D 维 Pad、Transpose，并根据 `softmaxInLayout` 处理 softmax 统计量；然后调用 `l0op::FlashAttentionScoreGrad`。L0 分配 dQ/dK/dV/dPSE/dSink 等内部输出，将数组转 INT64 tensor，执行 infer-shape 与一次 AICore launcher 添加。Host 最后 transpose/slice/reshape/ViewCopy 到用户输出。

### 参数、Workspace 与生命周期

executor 的总 workspace 包含前后处理临时量和算子 user workspace。arch22 模板的 user workspace 从多核同步区开始，顺序放 dQ/dK/dV FP32 梯度区、可选 sink、drop-mask 展开、mm1/mm2 交接、drop/mul 双缓冲和 PSE-alibi 区，全部按 `GM_ALIGN` 向上对齐，末尾含固定 buffer。V3 向共享 L0 显式传入空 rope，因此 rope slice/模板不活跃。workspace 和 executor 在同一 stream 完成前有效。

### 启动图与失败传播

```text
V3 GetWorkspaceSize
  -> dQ/dK/dV 全空且 dPSE 缺省/空：workspace=0，返回 executor（无节点）
  -> otherwise validation/Contiguous/Pad/Transpose
  -> l0op::FlashAttentionScoreGrad
     -> INFER_SHAPE
     -> ADD_TO_LAUNCHER_LIST_AICORE [唯一 FlashAttentionScoreGrad 节点]
        -> TilingFlashAttentionGradScore -> arch22 registry
        -> flash_attention_score_grad<MIX_AIC_1_2>
           -> Pre -> Matmul+Vector main -> Post
  -> output ViewCopy
V3 execute -> CommonOpExecutorRun
```

任何参数、前处理、infer、launcher、registry、Tiling、workspace 或 Matmul key 失败都停止构图/执行；未知编译 key 没有通用 Device fallback。

## Route 与参数设计

### 分核与资源计算

顶层从 platform 读取 AIC/AIV、UB/L1/L0。空域以 `CalculateTschBlockDim(sliceNum,aicNum,aivNum)` 编码 AIC:AIV 比。常规 template 把公开布局规范化为 B/N2/G/S1/S2/D，推导 `s1Inner/s2Inner/s1CvInner/s2CvInner/baseMN`、outer、coreNum、Matmul `baseM/baseN/baseK/step/depth/dbL0C`，再计算 workspace、tiling key、blockDim，并对需要全核 barrier 的模板设置 schedule mode 1。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 空域 | API 无 launch 终点，或 key 0 | dQ/dK/dV 全空且 dPSE 缺省/空时 API 直接返回 workspace=0；若 L0 已构图且任一主输出 storage size 为 0，则 Tiling key 0 清零其余输出 | 非空进入 registry |
| 2 | R-002 确定性/Basic | BN2 deter 1000、MLA 1001、BasicDet 1002、SameABDet 1100 | 对应 `IsCapable`：deterministic、Basic/MLA shape/dtype 契约 | R-003/R-004/R-005 |
| 3 | R-003 B/BN 小工作集 | B split 10000、BN split 11000 | 无 sink、`pseType==1`、K/V D 相等；B 路线拒绝 FP32且总块量不超过阈值，BN 路线另要求 `G==1`、`S2Align<=MAX_KV_SEQLEN`、`G*S1*S2Align<=BEST_BASIC_BLOCK_SIZE` | R-004/R-005 |
| 4 | R-004 SameAB/sink | SameAB priority 15500；sink 一维时强制该族 | 同一 A/B tile 可复用；非 sink 时还要求非 FP32及 SameAB shape/resource guard | R-005 |
| 5 | R-005 BN2/通用 S1S2 | BN2 priority 15000、general 16000 | BN2 的独立 capability 先判定；general 对无 sink场景兜底，sink 明确拒绝 | GRAPH_FAILED |

```text
dQ/dK/dV empty and dPSE absent-or-empty -> API no-launch R-001
else output-empty after L0 construction -> key-0 R-001
else registry by ascending priority number:
  deterministic/basic candidates -> R-002
  B/BN axis candidates -> R-003
  BN2 axis candidate -> R-005
  SameAB/sink -> R-004
  general S1S2-BN2GS1S2 (reject sink) -> R-005
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | API no-launch / key 0 | A2/A3 全输出空，或已构图后的 output-empty | 无 Device 节点，或 AIV clear | API early return / `EmptyTensor<T>` | workspace=0，或保留 100 MiB 但 clear body 不消费 | immediate return / former-tail |
| R-002 | key low digits 999/1999/BN2 deter | deterministic 或 Basic/MLA capability | MIX_CV single launch | Basic/BasicDet 或 BN2 deter/SameAB deter | FP32 deterministic slices、atomic/ordered post | template fallback |
| R-003 | suffix `0x099` / `0x199` | B/BN 小工作集 capability | MIX_CV single launch | `FlashAttentionScoreGradUbngs1s2Bb` / `FlashAttentionScoreGradUngs1s2Bbn` | 各自 POD、Matmul/UB 与 workspace | capability false 后继续 |
| R-004 | key suffix 1434 / `IsSameAB=1` | sink，或 non-FP32 SameAB capability | MIX_CV pre/SFMG/main/post | SameAB specialization | L1 custom、mm workspace、SyncAll | FP32去 R-005 |
| R-005 | suffix `0x134` 或 packed S1S2 key | FP16/BF16/FP32、layout、mask/PSE/drop、BN2/general selector | MIX_CV pre/main/post | BN2 或 S1S2-BN2GS1S2 body | Matmul+UB；GM ping-pong | general reject sink；unknown key return |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| sink route | sink shape 非空且 dimNum=1 | general body明确拒绝 sink，SameAB处理 sink/dsink | R-004 | 无可达 fallback则失败 | SameAB/general `IsCapable` |
| SameAB dtype | query != FP32 | L1 custom/SameAB specialization未接 FP32 | R-004 | R-005 | SameAB `IsCapable` |
| deterministic | context deterministic=1 | 固定归并顺序和额外 FP32 DB workspace | R-002/R-004 | nondeter route | Deter `IsCapable` |
| L1 custom | enableL1Custom + D=64/128/(128,256] | key 固化 S1/S2=512 和 D specialization | R-004 | nonaligned SameAB | `GetTilingKey` |
| dropout bit expansion | `dropoutIsDivisibleBy8==0` | byte/bit mask对齐不足，需 GM展开区 | R-004/R-005 | 直接读取 packed mask | `GetWorkspaceSize` |
| PSE inner mode | pseType 2/3 | PSE alibi tile需每核额外对齐 workspace | R-004/R-005 | 无额外区 | workspace planner |
| blockDim/schedule | CalcTschBlockDim !=0；SyncAll route schedule mode 1 | MIX_AIC_1_2 核比例和全核同步 | all nonempty | GRAPH_FAILED | `PostTiling` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| BaseParams | b/n2/g/s1/s2/d/rope_d/value_d | int64_t/64，各字段独立 | arch22 tiling class顺序；固定仓未给编译后 offsetof | 元素 | SaveToTilingData | main/pre/post | 正值/文档范围 |
| BaseParams | scaleValue/keepProb | float/32，各字段独立 | 同上 | scalar | attrs | vector/math | keepProb (0,1] |
| BaseParams | layout/pse/mask/sparse/sink | uint32_t/32，各字段独立 | 同上 | enum/switch | selector | body | inactive 不读 |
| SplitCoreParams | s1/s2 outer/inner/tail、blockOuter | int64/uint32，各字段独立 | 同上 | tile/block | DoSplit | scheduler | ceil-div/tail |
| PreParams | q/kv/v factor,total,tail | uint32_t/32，各字段独立 | tilingdata_base macro；数值 offsetof 不可见 | elements | pre tiling | FlashAttentionScoreGradPre | former/tail |
| PreParams | maskPreBlockTotal/dropBeginAddr | uint64_t/64，各字段独立 | 8B field | elements/address | pre tiling | bit2bool | active dropout |
| PostParams | q/kv/v size align | uint64_t/64，各字段独立 | macro layout | elements | workspace planner | post | aligned size |
| PostParams | dq/dk/dv/dsink workspace offsets | uint64_t/64，各字段独立 | macro layout | bytes | GetWorkspaceSize | main/post | GM align/non-overlap |
| TCubeTiling ×3 | M/N/K/base/step/depth | external POD | kernel_tiling ABI | tile/core | Matmul tiling API | mm1/mm3/mm4 | key/body一致 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | API empty / top-level output-empty | API + flash_attention_score_grad_tiling.cpp | no POD / EmptyTensor POD | no entry / flash_attention_score_grad.cpp | no body / empty body | op_host CMake |
| R-002 | deterministic/basic registry | arch22 basic/det/bn2/sab tiling files | Basic/BN2/SameAB POD | entry macros | Basic/BasicDet/BN2/SameAB | same CMake |
| R-003 | priorities 10000/11000 | tiling_bngs1s2_b.cpp / tiling_ngs1s2_bn.cpp | B/BN 独立 POD | suffix 099/199 router | B/BN 独立 body | same |
| R-004 | SameAB 15500 | tiling_s1s2_bn2gs1s2_sab.cpp | SameAB POD | suffix 1434 router | SameAB/SFMG/Post | same |
| R-005 | axis/general | b/ng/bn2/general tiling | matching POD | packed template key | corresponding body | same |

## Kernel 实现

### 公共执行流程

公开 API 仅加入一个 AICore 节点。entry 声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`，读取 user workspace 与 tiling POD。大多数 macros 按 `Pre::Process→SyncALLCores→Matmul main::Process→SyncALLCores→Post::Process`；main 中 AIC Matmul 通过 Fixpipe 写 FP32 workspace，AIV 从该区做 mask/PSE/dropout/softmax-grad，再把 dS 等交回 Cube 计算梯度。

### Route R-001：空域清零

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | op_api/aclnn_flash_attention_score_grad.cpp | `aclnnFlashAttentionScoreGradV3GetWorkspaceSize` | dQ/dK/dV 全空且 dPSE 缺省/空 | workspace=0，无 launcher |
| Tiling | op_host/...tiling.cpp | `IsEmptyOutput` | 已进入 L0 且输出 storage size 为 0 | key 0 |
| Router | op_kernel/...cpp | UB0=UB1=Block=0 | empty branch | clear + return |

#### Device 数据流

##### 函数级调用链

```text
entry -> EmptyTensor<T>::Init -> Process -> GM outputs zero
```

##### 源码等价伪码

```text
count=former_or_tail(blockIdx); Duplicate(UB,0); copy valid count to each output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM output | former/tail prefix | output elements | T/ND | AIV | user | executor | tail |
| L1/L0/workspace | N/A | Tiling 保留 100 MiB槽 | N/A | N/A | N/A | executor | N/A |
| UB | per-core zero tile | body tile | T | Duplicate | copy | loop | tail mask |

##### 分层内存布局

```mermaid
flowchart LR
  Z["AIV zero"] --> UB --> O["GM outputs"]
```

##### 字节账、交接边与复用顺序

```text
API 全空终点：workspace_total=0，且无 Device body
key 0 终点：workspace_total=100 MiB（Tiling 顶层约定）；L1/L0=N/A；UB=当前 clear tile
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| UB→GM | Duplicate | former/tail | T | output | AIV pipe | loop | Host count |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| clear | sequential AIV | zero | copy | output shard | exact zero |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | op_api/aclnn_flash_attention_score_grad.cpp / op_host/flash_attention_score_grad_tiling.cpp | V3 empty return / `RunEmptyTiling` | no-launch 或 former/tail |
| Kernel entry/router | op_kernel/flash_attention_score_grad.cpp | empty branch | dtype route |
| Device body/helper | arch22/flash_attention_score_grad_empty_tensor.h | Process | clear |

### Route R-002：确定性与 Basic

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | basic_det/bn2/sab tiling | IsCapable | deterministic 或 Basic/MLA contract | keys 1999/999/deter variants |
| Router | entry | UB/Block compile constants | compiled match | Basic/BasicDet/BN2/SameAB deter |

#### Device 数据流

##### 函数级调用链

```text
entry -> Basic/BasicDet::Process
or entry -> Pre -> BN2/SameAB main -> ordered post
```

##### 源码等价伪码

```text
partition fixed task rounds; accumulate FP32 partial dQ/dK/dV in deterministic workspace;
wait producer completion; reduce in fixed order; cast valid tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | route address helper | logical shape | input | caller | Cube/Vec | caller | tail |
| deter workspace | POD offsets | per-AIC FP32 DB × gradient extents | FP32 | Cube | ordered post | executor | no alias |
| L1/L0/UB | tiling specialization | base/step/depth | fractal/FP32 | main | post | task | barrier |

##### 分层内存布局

```mermaid
flowchart LR
  GM --> C["Cube partials"] --> W["FP32 deterministic workspace"] --> V["ordered AIV reduce/post"] --> O["GM"]
```

##### 字节账、交接边与复用顺序

```text
deter_extra = aligned(S1inner*Dalign*4*AIC*2)
            + aligned(S2inner*Dalign*4*AIC*2)
            + aligned(S2inner*D1align*4*AIC*2)（按实际 template active）
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| partial→reduce | Cube | per-core FP32 slice | FP32 | deterministic post | SyncAll/flags | launch | fixed round |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| partial reduction | all-core sync/fixed order | Cube | post | FP32 slices | deterministic order |
| output | cast tail | post | GM | logical gradient | FP32→T last |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch22/...basic_det.cpp | IsCapable/GetWorkspaceSize | deter key/workspace |
| Kernel entry/router | op_kernel/flash_attention_score_grad.cpp | INVOKE_FAG_DETERMINISTIC_BASIC_IMPL | router |
| Device body/helper | arch22/...basic_det.h | FlashAttentionScoreGradBasicDet | ordered body |

### Route R-003：B/BN 小工作集 MIX_CV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | tiling_bngs1s2_b.cpp | `FlashAttentionScoreGradUbngs1s2BbTiling::IsCapable` | 无 sink、非 FP32、`pseType==1`、K/V D 相等且 `N*G*align(S1)*align(S2)` 不超过阈值 | priority 10000；否则下一模板 |
| Tiling | tiling_ngs1s2_bn.cpp | `FlashAttentionScoreGradUngs1s2BbnTiling::IsCapable` | 无 sink、`pseType==1`、`G==1`、K/V D 相等、`S2Align<=MAX_KV_SEQLEN` 且工作集不超过阈值 | priority 11000；否则下一模板 |
| Router | op_kernel/flash_attention_score_grad.cpp | `UB0=UB1=9, Block=0/1` | key suffix `0x099/0x199` | B/BN 独立 body |

#### Device 数据流

##### 函数级调用链

```text
entry -> B/BN POD parse -> Pre -> selected B/BN Matmul+Vector body -> Post
```

##### 源码等价伪码

```text
if B route: assign one B-oriented task range per core
if BN route: assign one B*N2-oriented task range per core
mm1/mm2 Fixpipe FP32 tile -> AIV mask/PSE/dropout/softmax-grad -> mm3/mm4/mm5 -> post valid S/D tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | B/N/S/D layout stride | B×N×S×D | input ND | caller | B/BN body | caller | S/D tail |
| workspace/L1/L0/UB | B/BN 各自 Tiling 公式 | active tile + FP32 mm/gradient slices | FP32/fractal | AIC/AIV | peer/post | launch/task | flags |

##### 分层内存布局

```mermaid
flowchart LR
  T["GM B/N/S/D"] --> C["AIC Matmul"] --> W["FP32 workspace"] --> V["AIV softmax-grad"] --> C2["AIC gradient MM"] --> O["GM output"]
```

##### 字节账、交接边与复用顺序

```text
workspace = system/sync + mm1/mm2 per-core FP32 slices + dQ/dK/dV FP32 slices + active drop/PSE slices
B 与 BN 使用各自 GetWorkspaceSize；不得互换 POD offset
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| Cube→AIV | Matmul/Fixpipe | B/BN POD 的 per-core tile | FP32 | vector | flags/pipe | task | consume-before-reuse 与 S tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| stages | SyncALLCores | pre/main | main/post | workspace | S/D tail |
| tile | pipe flags | Cube/AIV | peer | current tile | FP32 accum |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch22/flash_attention_score_grad_tiling_bngs1s2_b.cpp / tiling_ngs1s2_bn.cpp | `IsCapable/GetWorkspaceSize` | B/BN 独立 selector 与 POD |
| Kernel entry/router | op_kernel/flash_attention_score_grad.cpp | suffix `0x099/0x199` | B/BN 分派 |
| Device body/helper | arch22/flash_attention_score_grad_bngs1s2_b.h / flash_attention_score_grad_ngs1s2_bn.h | `Process` | B/BN 各自 body |

### Route R-004：SameAB/sink MIX_CV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | tiling_s1s2_bn2gs1s2_sab.cpp | SameAB IsCapable | sink dim1→true；否则 non-FP32 + layout guards | SameAB key |
| Router | entry | key suffix 1434/IsSameAB=1 | SameAB macro | route |

#### Device 数据流

##### 函数级调用链

```text
entry -> Pre -> optional SFMG -> SameAB::ProcessFirstMM
  -> InitBuffer -> Process -> SyncAll -> Post/dSink
```

##### 源码等价伪码

```text
preclear; if AIV run softmax-grad front; share L1 AB tiles;
Cube writes mm outputs; AIV computes dS/sink; Cube completes gradients; post casts/reduces dSink
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| FP32 q/k/v workspace | sequential GM-aligned offsets | aligned sizes×4 | FP32 | Cube | post | executor | non-overlap |
| SFMG/dSink | after gradient slices | axisSize×reduce×4 / core*n2*g×4 | FP32 | AIV | post | launch | sink active |
| mm1/mm2 | per-core S1cv×S2cv×4×2 | two Matmul inputs | FP32 | Cube | AIV | task | flag |
| L1/L0 | shareL1/shareL0C | platform tiling | fractal | SameAB Cube | Fixpipe | task | optional dbL0C=2 |
| UB | 150KiB + 33KiB + 8KiB in general body analogue | fixed body buffers | FP32/T | GM/workspace | vector | task | sub-buffered |

##### 分层内存布局

```mermaid
flowchart LR
  GM --> L1["shared L1 AB"] --> L0["MMAD/L0C"] --> W["FP32 mm workspace"]
  W --> V["AIV SFMG/mask/drop"] --> DS["dS"] --> C["Cube gradients"] --> P["post+dSink"] --> O["GM"]
```

##### 字节账、交接边与复用顺序

```text
workspace = sync + align(q*4)+align(k*4)+align(v*4)
          + optional sink/core*n2*g*4 + optional dropMask
          + align(sfmgAxis*reduce*4)
          + align(core*s1Cv*s2Cv*4*2) + optional PSE + fixed buffer
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| mm→vector | AIC Fixpipe | per-core tile | FP32 | AIV | CrossCore/pipe flags | task | wait before reuse |
| main→post | AIC/AIV | gradient/dsink slices | FP32 | Post | SyncAll | launch | active output |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pre/main/post | SyncALLCores | each stage | next | workspace | logical S/D |
| tile | HardEvent/PipeBarrier | MTE/Vector | next pipe | UB/workspace | mask/tail |
| sink | FP32 reduction | AIV | post | dSink | deterministic route ordered |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch22/...tiling_s1s2_bn2gs1s2_sab.cpp | IsCapable/GetTilingKey/GetWorkspaceSize | sink/SameAB/L1 key |
| Kernel entry/router | op_kernel/flash_attention_score_grad.cpp | SAMEAB macros | 1434 mapping |
| Device body/helper | arch22/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h | SameAB Process | main body |

### Route R-005：通用 axis-template MIX_CV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Registry | b/ng/bn2/general tiling files | IsCapable | B/BN/BN2/S1S2 resource predicates；general reject sink | axis key |
| Router | entry macros | UB0/UB1/Block + layout/dtype/options | compiled key | matching body |

#### Device 数据流

##### 函数级调用链

```text
entry -> Pre -> selected B/BN/BN2/S1S2 main(Matmul+Vector) -> Post
```

##### 源码等价伪码

```text
for assigned axis tile:
  mm1/mm2 -> workspace; vector mask+pse+drop+softmaxGrad -> dS;
  mm3/mm4/mm5 -> FP32 dQ/dK/dV; post valid tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | layout-specific B/N/S/D | logical tensors | T/ND | caller | body | caller | tail |
| workspace | tiling offsets | q/k/v FP32 + per-core mm/drop/mul | FP32/ND/NZ | Cube/AIV | peer/post | executor | GM align/pingpong |
| L1/L0A/B/C | TCubeTiling | base/step/depth | fractal/FP32 | Matmul | Fixpipe | tile | DB |
| UB | InitBuffer subregions | 150KiB/33KiB/8KiB in general S1S2 body | FP32/T/mask | CopyIn | vector/CopyOut | task | hard events |

##### 分层内存布局

```mermaid
flowchart LR
  GM --> L1 --> L0["L0A/L0B -> MMAD -> L0C"] --> W["FP32 workspace"]
  W --> UB["AIV UB softmax-grad"] --> DS["dS"] --> L02["gradient MM"] --> P["post"] --> O["GM"]
```

##### 字节账、交接边与复用顺序

```text
workspace = sync + Σalign(q/k/v logical-aligned size*4)
          + optional align(dropMask) + core*s1Ratio*s2Ratio*baseMN*4*2
          + two ping-pong T buffers + optional PSE + fixed buffer
UB_general = 150KiB + 33KiB + 8KiB（对应三个 InitBuffer；内部再切片）
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| mm1/mm2→AIV | Fixpipe | per-core baseMN tile | FP32 | vector | MTE/Vector flags | task | pingpong |
| AIV dS→gradient MM | Vector | dS tile | FP32/T | AIC | pipe/cross-core | task | valid S tail |
| gradient→post | Cube | POD slices | FP32 | post | SyncAll | launch | D tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2↔V↔MTE3 | SetFlag/WaitFlag/PipeBarrier | copy/vector | next pipe | UB tile | mask S tail |
| AIC↔AIV | CrossCore/GM workspace protocol | Cube/Vec | peer | mm/dS tile | FP32 intermediate |
| post | SyncAll then cast | main | post | gradients | output dtype last |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch22/...tiling_s1s2_bn2gs1s2.cpp | GetTilingKey/GetWorkspaceSize/PostTiling | general route |
| Kernel entry/router | op_kernel/flash_attention_score_grad.cpp | INVOKE_FAG_GENERAL_* | axis mapping |
| Device body/helper | arch22/flash_attention_score_grad_s1s2_bn2gs1s2.h | Process/Calc | Matmul+Vector body |

## 算子特定优化

### Trick T-001：单 MIX_CV launch 内复用 pre/main/post 与 user workspace

#### 触发与源码依据

- 适用 Route：R-003、R-004、R-005
- 触发条件：entry macro 命中非空通用/SameAB key。
- Golden 来源：[`flash_attention_score_grad.cpp:62-91`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp#L62-L91)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
opPre.Process();
opPre.SyncALLCores();
pipeIn.Destroy();
TPipe pipeBase;
REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling,
                  op.mm3, bmm2tiling, op.mm4, bmm3tiling);
op.Process();
op.SyncALLCores();
```

#### 机制与设计原因

Host 只 launch 一次；Device 在同一 entry 内显式销毁/重建阶段 pipe，并在阶段边界全核同步。三个 Matmul 对象共享 system workspace，user workspace 保存跨 AIC/AIV 和 post 的 FP32 数据，避免把三个阶段误建成独立 Host 节点。

#### 迁移边界

- 适用：同一 MIX_AIC_1_2 launch、相同 Matmul/POD/workspace 布局。
- 前提：schedule mode 1、SyncAll、pipe Destroy 顺序和 workspace owner 均保留。
- 失效：拆成多 kernel 时必须由 stream/event 重新证明可见性和 lifetime。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 pre/main/post 写成三个 Host launch 会改变 workspace 可见性、system Matmul workspace 和失败传播，不能视为等价搬运。

### Trick T-002：SameAB L1 constant specialization

#### 触发与源码依据

- 适用 Route：R-004
- 触发条件：SameAB 命中且 `enableL1Custom`；S1/S2 固化 Aligned512，D 按 64/128/192 类别编码。
- Golden 来源：[`flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp:232-263`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp#L232-L263)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (fBaseParams.enableL1Custom) {
    s1TemplateType = STemplateTypeSab::Aligned512;
    s2TemplateType = STemplateTypeSab::Aligned512;
    if (fBaseParams.d == 64) {
        dTemplateType = DTemplateTypeSab::Aligned64;
    }
}
```

#### 机制与设计原因

Host 把可证明的固定 S/D 对齐编码进 key，Device 选择 L1 custom specialization，并把 `shareL1Size/shareL0CSize` 写入 Matmul tiling；部分实例启用 `dbL0C=2`。这改变真实 L1/L0C 容量与复用，不只是标签。

#### 迁移边界

- 适用：SameAB、非 FP32、相同 512 对齐和 D 类别。
- 前提：key、Matmul share size、Device template 参数一致。
- 失效：非对齐、FP32 或 D 不在枚举时走 nonaligned/general。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓 fallback 是 `NonAligned` template；不得在 Host 未设置 `enableL1Custom` 时强行实例化 Aligned512。

## 模板与实现对比

| 对比维度 | R-001 空域 | R-002 确定性 | R-003 B/BN | R-004 SameAB/sink | R-005 BN2/通用 | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | top-level | 1000-1100 | 10000/11000 | 15500 | 15000/16000 | registry/capability |
| 执行族/节点图 | AIV | MIX_CV | MIX_CV | MIX_CV | MIX_CV | semantic domain |
| M/N/K/tile | N/A | fixed rounds | B/BN-oriented | shared AB/L1 | BN2/S1S2-oriented | shape/resources |
| workspace | none semantic | deter FP32 DB | actual T | sink/SFMG/L1 | general pingpong | active features |
| AIC/AIV交接 | none | ordered slices | workspace/flags | workspace/flags | workspace/flags | producer-consumer |
| 同步/精度/tail | former/tail | fixed reduce | S/D tail | SyncAll/dsink | SyncAll/S-D tail | correctness |
| fallback | return | next template | next | general or fail | fail | closed registry |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V3 公开支持 | V3 公开支持 | V3 也有入口但 950 最新为 V4，见 `_950` 卡 |
| Host/Launcher | DAV_2201 arch22 registry | 同一 DAV_2201 接线 | DAV_3510 RegBase，不由本卡外推 |
| Kernel/资源 | MIX_AIC_1_2 arch22 | 同上 | APT/arch35，独立 POD/key |

## 限制与失败边界

- A2/A3 最新公开版本是 V3；V4 文档明确不支持这两个平台。
- V3 的 sink/dSink 是独立新增语义；不能用 V2 卡替代，也不能把 950 的 V4 rope/seed/scale 功能倒推到本卡。
- arch22 的 unpadded/TND template 是共享内部实现，但 V3 原型没有 `actualSeqQLen/actualSeqKvLen` 且公开布局不含 TND，因此不是本卡的 Host 可达 Route。
- SameAB/general/BN2/BN/B/Basic 各有独立 capability/key/POD；本卡矩阵保留所有可达 selector，Device 不提供未知 key fallback。
- workspace offset 为字节，double buffer 必须乘 2；逻辑元素数不能代替 GM_ALIGN 后的 FP32 容量。
- POD 宏展开后的编译期 `sizeof/alignof/offsetof` 数值固定仓未列出，标记为不可见而不猜测。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV3.md | V3 产品表/原型 | A2/A3 最新入口 |
| 入口实现 | attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp | V3GetWorkspaceSize | 校验/executor/pre/post |
| L0 launcher | attention/flash_attention_score_grad/op_api/flash_attention_score_grad.cpp | l0op::FlashAttentionScoreGrad | infer/单节点 launch |
| Host/Tiling | attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp | TilingFlashAttentionGradScore | empty/EOD/registry |
| POD/共同约束 | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_common.h | PreParams/PostParams | Host POD 定义 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp | general template | R-005 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp | SameAB | R-004 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2.cpp | BN2/deterministic | R-002/R-005 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_bngs1s2_b.cpp | B split | R-003 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_ngs1s2_bn.cpp | BN split | R-003 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic.cpp | Basic/MLA | R-002 |
| Host/Tiling | attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_det.cpp | Basic deterministic | R-002 |
| Device POD | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_tiling.h | tiling classes | Device ABI |
| Kernel entry | attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp | entry/macros | MIX_CV router |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2.h | S1S2 Process | Matmul/Vector body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h | SameAB Process | R-004 Cube/Vector body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2.h | BN2 Process | R-002/R-005 body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_bngs1s2_b.h | B Process | R-003 body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_ngs1s2_bn.h | BN Process | R-003 body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h | Basic Process | R-002 body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2gs1s2_basic_det.h | BasicDet Process | R-002 deterministic body |
| Kernel body | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_empty_tensor.h | EmptyTensor Process | R-001 clear body |
| Kernel pre | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_pre.h | Pre | clear/mask preparation |
| Kernel post | attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_post.h | Post | cast/reduction/tail |
| 构建接线 | attention/flash_attention_score_grad/op_host/CMakeLists.txt | add_op_to_compiled_list/options | Host/Kernel 编译注册 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Base/Split/Pre/Post/TCubeTiling | 字段顺序、位宽、单位、active 开关 | tiling_common.h/tiling.h |
| 调用顺序 | V3→L0→Tiling→单 MIX entry→pre/main/post | 单 launch、pipe Destroy、SyncAll | V3GetWorkspaceSize/entry macros |
| workspace/内存 | FP32 gradients、mm/drop/mul pingpong、PSE/sink | offsets、GM align、DB、owner | GetWorkspaceSize |
| 同步 | stage SyncAll + pipe/hard events | schedule mode、flag方向、复用前等待 | entry/body |
| 精度与尾块 | FP32 Matmul/vector intermediate，post cast | 不提前 cast，不写 padding/EOD | main/Post |
| 失败边界 | capability/key/shape/workspace fail closed | 不执行未接线模板 | registry/entry |