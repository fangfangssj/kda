---
type: Operator
title: "FlashAttentionVarLenScore 950 ACLNN 算子知识"
description: "说明 aclnnFlashAttentionVarLenScoreV5 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV5.md
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_api/aclnn_flash_attention_score.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attn/op_api/aclnn_flash_attn.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attn/op_api/flash_attn.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attn/op_host/flash_attn_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attn_metadata/op_host/op_api/aclnn_flash_attn_metadata.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_api/aclnn_flash_attention_score.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_dropmask.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/flash_attention_score_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_template_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_entry_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_drop_mask_adapter_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_empty_tensor_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_kernel_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_block_cube.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_block_vec_train.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/flash_attention_score_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:27:18Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# FlashAttentionVarLenScore 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize` / `GetMaxWorkspaceSize` / `aclnnFlashAttentionVarLenScoreV5`，公开 ACLNN |
| 计算语义 | TND 变长训练 attention；950 V5 不支持 sink 与 softmaxOutLayout |
| 输入输出 | FP16/BF16/FP32 TND Q/K/V与optional；FP32 max/sum、attentionOut；用户softmax输出未提交 |
| 执行类型 | RegBase 单 Kernel `MIX_CV`，`KERNEL_TYPE_MIX_AIC_1_2` |
| Kernel 路线 | R-001 standard RegBase；R-002 optionalDn；R-003 empty初始化 |
| 核心资源 | GM/user workspace、L1/L0A/B/C、UB/寄存器、cross-core buffers、三重ping-pong |
| 来源闭包 | `source_closed`：V5 ACLNN→DAV_3510 Tiling/POD/key→MIX KernelTrain→Cube/Vec body→ascend950构建 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

按 actual-length TND 段计算 `S=scaleValue*(QK^T+QropeKrope^T)+PSE`，执行 sparse/mask、FP32在线stable softmax、可选dropout与 `PV`。950 V5公开文档明确暂不支持 `sinkOptional` 与 `softmaxOutLayout`；不得因共享L0仍带这些指针而宣称可用。V5 API只接受 FP16/BF16/FP32，且把 `dScaleQ/K/V` 固定为 null、outDtype固定0；共享entry中的 HIFLOAT8/FP8 specialization对该API不可达。固定Host也不把内部softmax第3输出复制到用户 `softmaxOutOut`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `query/key/value` | ND，逻辑TND `[T,N,D]` | FP16/BF16/FP32 | dtype一致；`D,Dv<=768`；GQA ratio为正整数 |
| `queryRope/keyRope` | optional TND | 文档约束BF16 | 成对参与QK；组合受Host校验 |
| `realShift/dropMask/paddingMask/attenMask` | optional | 浮点/UINT8/BOOL | PSE、dropout、稀疏mask；drop可先adapter |
| `sinkOptional` | API参数 | FLOAT32 | 950 V5文档明确暂不支持；正式Route按null使用 |
| `prefix/actualSeq*/startIdx*` | `[B]` | INT64 | TND分段、sparse与PSE起点 |
| 属性 | scalar/string | double/int64/string | scale/keep/pre/next/head/layout/precise/sparse/pse；950不支持softmaxOutLayout |
| `softmaxMax/Sum` | `[T,N,8]`逻辑 | FP32 | Host ViewCopy提交 |
| `softmaxOut` | API参数 | 文档定义 | 950不支持布局参数，且固定Host未提交L0 softmax输出 |
| `attentionOut` | TND `[T,N,Dv]` | 与输入约束一致 | outDtype固定0；Postprocess/ViewCopy |

### 入口与版本

固定仓 `FlashAttentionVarLenScore` 版本族的最新公开变长接口为V5，产品表列 Ascend 950PR/950DT 支持；本卡仅记录950。递归扫描 `attention/flash_attn/` 与 `attention/flash_attn_metadata/` 另见独立公开符号 `aclnnFlashAttn` 和 `aclnnFlashAttnMetadata`：前者是训练/推理统一的新签名并只为 `FlashAttn` 注册 ascend950，后者单独生成该接口消费的 metadata；它们没有进入 V5 的 `FlashAttentionScore` Host/Tiling/Kernel 链，故不是 `FlashAttentionVarLenScoreV6`，也不替代本卡的V5版本选择。V5 API执行与A2/A3共用helper，但平台分发到 `DAV_3510`、arch35 Simplified POD和 `flash_attention_score_apt`。`GetMaxWorkspaceSize` 强制drop=null、keepProb=1并传max-workspace标志；公开执行由 `CommonOpExecutorRun`。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckFaParam`→全输出空零workspace快返→TND/format/dtype/shape/actual-length校验→全部tensor连续化→`PreprocessQKV`→multi-input校验→L0 `FlashAttentionScore`。空shape sink转null；950正式调用应始终不提供sink。内部max/sum/attention必须非null，attention执行Postprocess，随后max/sum/attention ViewCopy；softmaxOut不提交。

### 参数、Workspace 与生命周期

priority81构造时先把raw tiling data清零，若需adapter则生成其POD，然后返回 `GRAPH_PARAM_INVALID` 继续priority82。非空主区：若 `D>128`，`bmm2Bytes=s1Base*bmm2ResBlockSize*calcTypeSize`；仅当 `D>128` 且D模板>256时，另有 `vec2Bytes=s1Base*dBasicBlock*calcTypeSize`；两者各512B对齐，乘 `PING_PONG_VALUE=3` 与 `coreNum`。drop区追加 `AlignUp(N2*G*Σ_b S1_b*S2_b,512)`，末尾加16MiB reserve。empty固定100MiB。executor与by-value POD活到stream完成。

### 启动图与失败传播

```text
V5 API -> L0 -> TilingFlashAttentionScore
  -> DAV3510 empty? R-003
  -> priority81 raw-POD memset/dropmask probe
  -> priority82 VarLen: optionalDn ? R-002 : R-001
  -> flash_attention_score_apt single MIX launch
     optional DropMaskAdapter -> TPipe::Reset
     KernelTrain 4-stage software pipeline: bmm1 -> vec1 -> bmm2 -> vec2
-> post/ViewCopy -> CommonOpExecutorRun(stream)
```

任一shape/sparse/PSE、D/Dv>768、workspace、key白名单或节点创建失败均终止；priority探测的PARAM_INVALID仅表示继续模板。

## Route 与参数设计

### 分核与资源计算

默认 `s1Base=s2Base=128`；D>256时FP32令s1=64，其余令s1=128；D模板按 `AlignUp(D+DRope,64)` 分64/128/192/256，>256统一768。低占核、非FP32、无RoPE且S2>1024且无mask/PSE/drop时可把s1改64、s2改256。`totalSize=Σ_b ceil(S1_b/s1Base)*N2*G`，`coreNum=min(totalSize,aicNum)`。TND splitCoreMode可正/倒序分配；sparseStartIdx[48]经均分、粗调、精调择优。blockDim由 `CalcTschBlockDim(coreNum*2,aicNum,aivNum)` 生成；drop adapter时按全部AIV重算。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-003 empty | KernelType=1，其余key位0 | Q/K/V任一为空且attention或sum非空 | 全输出空API快返；非空继续 |
| 2 | R-002 optionalDn | priority82、`optionalDn=1` | 有attenMask且compress为LEFT_UP或RIGHT_DOWN causal；无PSE/RoPE/drop；非FP32/FP8；D模板<=256且D=DV模板；s1模板!=64；平均S1、S2均>=1536 | 任一条件假R-001 |
| 3 | R-001 standard | priority82、`optionalDn=0` | DAV3510、TND、非空且varlen capable；V5 dtype FP16/BF16/FP32 | 无白名单实例失败 |

```text
all outputs empty -> API return
empty compute input with output -> R-003
non-TND/non-DAV3510 -> fail
full optionalDn predicate -> R-002
otherwise -> R-001
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 14段key，Kernel0/Impl/Layout4/S1/S2/D/DV/PSE/mask/drop/rope/out/regbase/optionalDn0 | 950 TND非空；FP16/BF16/FP32 | 单MIX 1:2，4阶段 | `KernelTrain<FANoQuantBlockCube,FANoQuantBlockVecTrain>` | cross-core UB/L1/GM buffers，3槽state | runtime/template双tail；key失败 |
| R-002 | 同key但optionalDn1 | 完整Dn谓词 | 同一MIX，Cube `useDn` specialization | bmm1 Dn/DnSplitK；其余同KernelTrain | L1全载/驻留与不同Matmul layout | guard假R-001 |
| R-003 | Kernel1 | 空输入、非空输出 | 纯AIV init | `EmptyTensorRegbase<T>` | UB→GM；100MiB reserve | former/tail/last |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| optionalDn | 上表完整析取否定后的谓词 | 只在无附加算子、大平均序列、相同小D/Dv下采用Dn搬运 | R-002 | R-001 | `AnalyzeOptionalDn` |
| D/Dv上界 | 两者`<=768` | key模板与L1/L0切分上限 | R-001/R-002 | Tiling失败 | `DoOpTiling` |
| workspace门槛 | `D>128`才分配bmm2区；D模板>256再分vec2区 | 小D可L0C→UB，较大D需GM中转/分段 | R-001 | 0字节主区 | `GetWorkspaceSize` |
| triple buffer | `(bmm2Bytes+vec2Bytes)*3*coreNum` | KernelTrain保留跨4阶段的三个活跃结果槽 | R-001/R-002 | N/A | `PING_PONG_VALUE`/taskId mod3 |
| drop UB | `baseUbCalSize=AlignDown(ceil(ubSize/DROP_MASK_CAL_SIZE),2048)` | bit展开的UB组合容量 | R-001 | no adapter/失败 | dropmask tiling |
| sparse balance | 粗调+`BalanceLoad`精调仅在最大负载更小时替换均分 | S2有效长度不均 | R-001/R-002 | 保留均分 | `SetSparseStartIdx` |
| empty 32B | blocks/core三种partition | 避免GM越界 | R-003 | API快返 | `IsEmptyInputRegbase` |

### POD/ABI

仓内无整结构 `sizeof/alignof/offsetof` static_assert；priority81先清零raw data。以下每个字段独占一行；ABI布局由类声明顺序决定，实际offset不可外推，且不能与arch22 POD互换。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `FlashAttentionScoreSimplifiedTilingData` | `inputParamsRegbase` | nested | 第1成员；结构offset固定仓不可见 | 参数 | arch35 analyzers | KernelTrain/Cube/Vec blocks | priority81先清零raw data |
| `InputParamsRegbase` | `bSize` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `t1Size` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `t2Size` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `n2Size` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `gSize` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `s1Size` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `s2Size` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `alignedS2` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `dSize` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `dSizeV` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `dSizeRope` | int64/64 | 声明顺序；offset固定仓不可见 | element/count | TND shape parser | address/tile mapping | D/Dv<=768；乘积64位 |
| `InputParamsRegbase` | `keepProb` | float/32 | 声明顺序；offset固定仓不可见 | scalar | 属性double窄化 | dropout | max-workspace固定1 |
| `InputParamsRegbase` | `scaleValue` | float/32 | 声明顺序；offset固定仓不可见 | scalar | 属性double窄化 | score scale | 接口范围 |
| `InputParamsRegbase` | `preTokens` | int64/64 | 声明顺序；offset固定仓不可见 | token | sparse analyzer | valid-S2 range | 须满足sparse约束 |
| `InputParamsRegbase` | `nextTokens` | int64/64 | 声明顺序；offset固定仓不可见 | token | sparse analyzer | valid-S2 range | 须满足sparse约束 |
| `InputParamsRegbase` | `pseS1Size` | int64/64 | 声明顺序；offset固定仓不可见 | element | PSE analyzer | PSE offset | inactive为0 |
| `InputParamsRegbase` | `pseS2Size` | int64/64 | 声明顺序；offset固定仓不可见 | element | PSE analyzer | PSE offset | inactive为0 |
| `InputParamsRegbase` | `pseBSize` | uint32/32 | 声明顺序；offset固定仓不可见 | count | PSE analyzer | PSE broadcast | inactive为0 |
| `InputParamsRegbase` | `bandIndex` | uint32/32 | 声明顺序；offset固定仓不可见 | index | sparse analyzer | band route | inactive为0 |
| `InputParamsRegbase` | `layoutType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | V5 TND为4 |
| `InputParamsRegbase` | `pseShapeType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `attenMaskShapeType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `attenMaskDataType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `attenMaskCompressMode` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `implMode` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `sparseType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `needDropMaskOp` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `dropMaskOuter` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `pseEncodeType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 枚举/布尔有效域 |
| `InputParamsRegbase` | `tndSoftmaxOut` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | 950 V5正式Route为0 |
| `InputParamsRegbase` | `remain` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/bool | arch35 analyzers | runtime/template branch | reserved=0 |
| `InputParamsRegbase` | `attenMaskS2Size` | uint32/32 | 声明顺序；offset固定仓不可见 | element | mask analyzer | Vector mask address | shape边界 |
| `InputParamsRegbase` | `pseType` | uint32/32 | 声明顺序；offset固定仓不可见 | enum | PSE analyzer | PSE branch | 接口枚举 |
| `InputParamsRegbase` | `rsv1` | uint32/32 | 声明顺序；offset固定仓不可见 | reserved | raw memset | N/A | 0 |
| `InputParamsRegbase` | `qStartIdx` | int64/64 | 声明顺序；offset固定仓不可见 | token | start-index analyzer | PSE/sparse address | optional为0 |
| `InputParamsRegbase` | `kvStartIdx` | int64/64 | 声明顺序；offset固定仓不可见 | token | start-index analyzer | PSE/sparse address | optional为0 |
| `InputParamsRegbase` | `s1SparseValidSize` | int64/64 | 声明顺序；offset固定仓不可见 | element | sparse producer | scheduler/Vector | 按sparse Route |
| `InputParamsRegbase` | `s2SparseValidSize` | int64/64 | 声明顺序；offset固定仓不可见 | element | sparse producer | scheduler/Vector | 按sparse Route |
| `InputParamsRegbase` | `seed` | int64/64 | 声明顺序；offset固定仓不可见 | scalar | shared analyzer | drop branch | V5 seed/offset固定0 |
| `InputParamsRegbase` | `offset` | int64/64 | 声明顺序；offset固定仓不可见 | scalar | shared analyzer | drop branch | V5 seed/offset固定0 |
| `InputParamsRegbase` | `keepProbUint8` | int64/64 | 声明顺序；offset固定仓不可见 | scalar | shared analyzer | drop branch | V5 seed/offset固定0 |
| `InputParamsRegbase` | `pseAlibiBaseS1` | int64/64 | 声明顺序；offset固定仓不可见 | element | PSE analyzer | PSE address | inactive为0 |
| `InputParamsRegbase` | `pseAlibiBaseS2` | int64/64 | 声明顺序；offset固定仓不可见 | element | PSE analyzer | PSE address | inactive为0 |
| `InputParamsRegbase` | `deqScaleFlag` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared analyzer | shared blocks | V5 dequant inactive |
| `InputParamsRegbase` | `deqScale2Flag` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared analyzer | shared blocks | V5 dequant inactive |
| `InputParamsRegbase` | `isActualSeqLengthsNull` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared analyzer | shared blocks | V5 dequant inactive |
| `InputParamsRegbase` | `isActualSeqLengthsKVNull` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared analyzer | shared blocks | V5 dequant inactive |
| `InputParamsRegbase` | `actualSeqLengthsSize` | uint32/32 | 声明顺序；offset固定仓不可见 | count | length parser | bounds check/address | <=B array |
| `InputParamsRegbase` | `actualSeqLengthsKVSize` | uint32/32 | 声明顺序；offset固定仓不可见 | count | length parser | bounds check/address | <=B array |
| `InputParamsRegbase` | `isKvContinuous` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `fromFused` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `isBSNDOut` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `transposeLayout` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `isGqa` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `isSoftMaxLseEnable` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `isActualSharedPrefixLenNull` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `isQHasLeftPadding` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `isKVHasLeftPadding` | uint8/8 | 声明顺序；offset固定仓不可见 | flag | shared base analyzer | address/output branches | V5 inactive项为0或由GQA推导 |
| `InputParamsRegbase` | `ropeHeadSize` | uint32/32 | 声明顺序；offset固定仓不可见 | element/ratio | shape parser | RoPE/GQA address | conditional |
| `InputParamsRegbase` | `prefixSeqInnerSize` | uint32/32 | 声明顺序；offset固定仓不可见 | element/ratio | shape parser | RoPE/GQA address | conditional |
| `InputParamsRegbase` | `headNumRatio` | uint32/32 | 声明顺序；offset固定仓不可见 | element/ratio | shape parser | RoPE/GQA address | conditional |
| `InputParamsRegbase` | `blockSize` | int32/32 | 声明顺序；offset固定仓不可见 | element/count | shared base analyzer | PA/mask branch | V5 PA项为0 |
| `InputParamsRegbase` | `blockTableDim2` | int32/32 | 声明顺序；offset固定仓不可见 | element/count | shared base analyzer | PA/mask branch | V5 PA项为0 |
| `InputParamsRegbase` | `paBlockNumSum` | int32/32 | 声明顺序；offset固定仓不可见 | element/count | shared base analyzer | PA/mask branch | V5 PA项为0 |
| `InputParamsRegbase` | `attenMaskS1Size` | int32/32 | 声明顺序；offset固定仓不可见 | element/count | shared base analyzer | PA/mask branch | V5 PA项为0 |
| `InputParamsRegbase` | `kvSplitPart` | uint32/32 | 声明顺序；offset固定仓不可见 | part/byte | shared base analyzer | workspace/output branch | V5 split项inactive |
| `InputParamsRegbase` | `accumOutSize` | uint32/32 | 声明顺序；offset固定仓不可见 | part/byte | shared base analyzer | workspace/output branch | V5 split项inactive |
| `InputParamsRegbase` | `logSumExpSize` | uint32/32 | 声明顺序；offset固定仓不可见 | part/byte | shared base analyzer | workspace/output branch | V5 split项inactive |
| `InputParamsRegbase` | `paLayoutType` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/flag | shared base analyzer | PA/quant branch | V5 quant/PA inactive |
| `InputParamsRegbase` | `isRowInvalid` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/flag | shared base analyzer | PA/quant branch | V5 quant/PA inactive |
| `InputParamsRegbase` | `isPostQuantPerChnl` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/flag | shared base analyzer | PA/quant branch | V5 quant/PA inactive |
| `InputParamsRegbase` | `isPostQuantBF16` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/flag | shared base analyzer | PA/quant branch | V5 quant/PA inactive |
| `InputParamsRegbase` | `antiquantPageAttentionFlag` | uint8/8 | 声明顺序；offset固定仓不可见 | enum/flag | shared base analyzer | PA/quant branch | V5 quant/PA inactive |
| `InputParamsRegbase` | `antiquantPerTensorFlag` | uint16/16 | 声明顺序；offset固定仓不可见 | bitset | shared base analyzer | quant branch | V5为0 |
| `InputParamsRegbase` | `antiquantPerHeadFlag` | uint16/16 | 声明顺序；offset固定仓不可见 | bitset | shared base analyzer | quant branch | V5为0 |
| `InputParamsRegbase` | `antiquantParaSeqSize` | uint32/32 | 声明顺序；offset固定仓不可见 | count | shared base analyzer | quant branch | V5为0 |
| `FlashAttentionScoreSimplifiedTilingData` | `multiCoreParamsRegbase` | nested | 第2成员；结构offset固定仓不可见 | task | split/sparse producer | KernelTrain scheduler | 数组固定48项 |
| `MultiCoreParamsRegbase` | `coreNum` | int32/32 | 声明顺序；offset固定仓不可见 | core | SetMultiCoreParamsRegbase | KernelTrain/core mapping | <=aicNum且<=48 |
| `MultiCoreParamsRegbase` | `totalSize` | int64/64 | 声明顺序；offset固定仓不可见 | task/loop | tiling split producer | KernelTrain task loop | 向上取整与tail |
| `MultiCoreParamsRegbase` | `s1OuterSize` | int64/64 | 声明顺序；offset固定仓不可见 | task/loop | tiling split producer | KernelTrain task loop | 向上取整与tail |
| `MultiCoreParamsRegbase` | `splitFactorSize` | int64/64 | 声明顺序；offset固定仓不可见 | task/loop | tiling split producer | KernelTrain task loop | 向上取整与tail |
| `MultiCoreParamsRegbase` | `splitFactorTailSize` | int64/64 | 声明顺序；offset固定仓不可见 | task/loop | tiling split producer | KernelTrain task loop | 向上取整与tail |
| `MultiCoreParamsRegbase` | `bnStartIdx[48]` | uint32[48] | 声明顺序；offset固定仓不可见 | task index | scheduler producer | KernelTrain task decode | 有效索引<48 |
| `MultiCoreParamsRegbase` | `sparseStartIdx[48]` | int64[48] | 声明顺序；offset固定仓不可见 | task index | SetSparseStartIdx | sparse scheduler | 有效索引<48 |
| `MultiCoreParamsRegbase` | `firstFullLoadS1OuterIdx` | int64/64 | 声明顺序；offset固定仓不可见 | task index | split-core analyzer | KernelTrain scheduler | conditional |
| `MultiCoreParamsRegbase` | `splitCoreMode` | uint8/8 | 声明顺序；offset固定仓不可见 | enum | SetSplitCoreModeParam | KernelTrain scheduler | 0/1 |
| `MultiCoreParamsRegbase` | `reserve[3]` | uint8[3]/24 | 尾部padding；offset固定仓不可见 | byte | raw memset | N/A | 必须为0 |
| `FlashAttentionScoreSimplifiedTilingData` | `dropmaskParamsRegbase` | nested | 第3成员；结构offset固定仓不可见 | adapter POD | priority81/PostTiling | drop adapter/main | needDropMaskOp条件有效 |
| `DropmaskParamsRegbase` | `multiCoreFactorSize` | int32/32 | 声明顺序；offset固定仓不可见 | loop | dropmask tiling | adapter | active时>0 |
| `DropmaskParamsRegbase` | `baseUbCalSize` | int32/32 | 声明顺序；offset固定仓不可见 | element | dropmask tiling | adapter | 2048对齐 |
| `DropmaskParamsRegbase` | `multiCoreTotalSize` | int64/64 | 声明顺序；offset固定仓不可见 | loop | dropmask tiling | adapter | inactive为0 |
| `DropmaskParamsRegbase` | `shapeTotalSize` | int64/64 | 声明顺序；offset固定仓不可见 | element | dropmask tiling | adapter | TND有效面积 |
| `DropmaskParamsRegbase` | `dropMaskAddrOffset` | int64/64 | 声明顺序；offset固定仓不可见 | byte | PostTiling | adapter/main drop地址 | 等于主区末尾 |
| `FlashAttentionScoreSimplifiedTilingData` | `initOutputParams` | nested | 第4成员；结构offset固定仓不可见 | init POD | shared init producer | shared base | 正常V5多为0 |
| `InitOutputParams` | `singleCoreSize` | uint32/32 | 声明顺序；offset固定仓不可见 | element | init producer | init loop | conditional |
| `InitOutputParams` | `needInit` | uint8/8 | 声明顺序；offset固定仓不可见 | bool | init producer | shared base | 正常V5多为0 |
| `InitOutputParams` | `isOneN` | uint8/8 | 声明顺序；offset固定仓不可见 | bool | init producer | shared base | conditional |
| `InitOutputParams` | `rsvd[2]` | uint8[2]/16 | 显式padding；offset固定仓不可见 | byte | raw memset | N/A | 必须为0 |
| `InitOutputParams` | `totalOutputSize` | int64/64 | 声明顺序；offset固定仓不可见 | element | init producer | shared base | conditional |
| `InitOutputParams` | `totalSoftMaxLseOutputSize` | int64/64 | 声明顺序；offset固定仓不可见 | element | init producer | shared base | conditional |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `coreNum` | uint32/32 | 声明顺序；offset固定仓不可见 | core | GetEmptyArgs | empty scheduler | key1/32B partition |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `attentionOutFormerNum` | uint32/32 | 声明顺序；offset固定仓不可见 | count | GetEmptyArgs | empty scheduler | key1/32B partition |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `attentionOutTailNum` | uint32/32 | 声明顺序；offset固定仓不可见 | count | GetEmptyArgs | empty scheduler | key1/32B partition |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `softmaxMaxFormerNum` | uint32/32 | 声明顺序；offset固定仓不可见 | count | GetEmptyArgs | empty scheduler | key1/32B partition |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `softmaxMaxTailNum` | uint32/32 | 声明顺序；offset固定仓不可见 | count | GetEmptyArgs | empty scheduler | key1/32B partition |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `attentionOutSingleCoreDataSize` | uint64/64 | 声明顺序；offset固定仓不可见 | element/index | GetEmptyArgs | empty copy loops | 32B尾界 |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `attentionOutTailCoreDataSize` | uint64/64 | 声明顺序；offset固定仓不可见 | element/index | GetEmptyArgs | empty copy loops | 32B尾界 |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `softmaxMaxSingleCoreDataSize` | uint64/64 | 声明顺序；offset固定仓不可见 | element/index | GetEmptyArgs | empty copy loops | 32B尾界 |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `softmaxMaxTailCoreDataSize` | uint64/64 | 声明顺序；offset固定仓不可见 | element/index | GetEmptyArgs | empty copy loops | 32B尾界 |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `attentionOutLastCoreDataSize` | uint64/64 | 声明顺序；offset固定仓不可见 | element/index | GetEmptyArgs | empty copy loops | 32B尾界 |
| `FlashAttentionScoreEmptyInputTilingDataRegbase` | `attentionOutLastCoreIndex` | uint64/64 | 声明顺序；offset固定仓不可见 | element/index | GetEmptyArgs | empty copy loops | 32B尾界 |
### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | priority82 optionalDn0 | regbase+varlen+drop tiling | Simplified POD | apt→regbase entry | KernelTrain→Cube/VecTrain | ascend950 opFile+CMake SSBUF |
| R-002 | priority82 optionalDn1 | `AnalyzeOptionalDn` | 同POD，key最后位1 | 同entry | Cube Dn specialization+同Vec | 同上 |
| R-003 | Kernel1 | `IsEmptyInputRegbase` | empty POD | apt key1 | EmptyTensorRegbase | 同上 |

## Kernel 实现

### 公共执行流程

R-001/R-002 的entry显式 `KERNEL_TYPE_MIX_AIC_1_2`，用core type选择真实Cube/Vec或Dummy。optional adapter结束后reset pipe。KernelTrain在runInfo[4]与三个伪循环flush上形成bmm1(t)→vec1(t-1)→bmm2(t-2)→vec2(t-3)流水。Cube/AIV用 `Buffer<...,CROSS_CORE_SYNC_*>` 的 `WaitCrossCore/SetCrossCore` 交接；小D bmm2直接L0C→UB，大D通过GM workspace。

### Route R-001：standard RegBase MIX

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | API cpp | V5 helper | 950公开参数合法、非空 | L0 |
| Tiling | arch35 varlen cpp | `IsCapable/GetTilingKey` | DAV3510、TND、optionalDn假 | 14段key |
| Router | apt/entry headers | `KernelType=0` / `flash_attention_score_regbase` | dtype FP16/BF16/FP32 | no-quant KernelTrain |

#### Device 数据流

##### 函数级调用链

```text
flash_attention_score -> flash_attention_score_regbase
 -> optional DropMaskAdapterRegbase -> KernelTrain::InitBaseAPI -> Process
 -> Cube::IterateBmm1 -> VecTrain::ProcessVec1 -> Cube::IterateBmm2 -> VecTrain::ProcessVec2
```

##### 源码等价伪码

```text
按sparseStartIdx/splitCoreMode解析TND task；对每个S2 tile：
AIC GM Q/K->L1->L0A/B->FP32 L0C->cross-core score；
AIV scale/PSE/mask/online softmax/drop，写P到cross-core L1；
AIC等待P，P/V->L0->FP32 PV，按D选择UB或GM结果；
AIV等待PV，在线rescale累加，最终除sum、cast并写max/sum/attention。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM Q/K/V | actual-prefix+head+tile stride | TND真实域 | input ND | caller | Cube | stream | runtime tail |
| L1 | Q/K/P/V tiles | template S1/S2/D/DV | NZ/fractal | MTE2/AIV | L0/Mmad | AIC/task | ping-pong/cross-core |
| L0A/B/C | Matmul tiles | template base；L0C FP32 | fractal/FP32 | Load/Matmul | Fixpipe | AIC/mmad | HardEvent |
| UB/Reg | score、mask、max/sum/exp、PV | template buffers；max/sum/exp各3×256B | FP32/input | Fixpipe/GM | Vec | AIV/task槽 | predicate tail |
| workspace | system后user基址 | 主区+drop区+16MiB | byte/FP32 | Cube/adapter/Vec | later stage | executor | 512B对齐、3槽 |

##### 分层内存布局

```mermaid
flowchart LR
  QK["GM Q/K"] --> L1["L1"] --> MM1["L0A/B->Mmad->L0C"] --> UB["cross-core UB score"]
  UB --> V1["AIV softmax/drop"] --> PL1["cross-core L1 P"] --> MM2["AIC PV"]
  MM2 --> RES["UB or GM workspace"] --> V2["AIV rescale/div/cast"] --> OUT["GM outputs"]
```

##### 字节账、交接边与复用顺序

```text
bmm2Bytes=AlignUp(D>128 ? s1Base*bmm2ResBlockSize*calcTypeSize : 0,512)
vec2Bytes=AlignUp((D>128 && DTemplate>256) ? s1Base*dBasicBlock*calcTypeSize : 0,512)
main=((bmm2Bytes+vec2Bytes)*3)*coreNum
drop=needDrop?AlignUp(N2*G*sum(S1_b*S2_b),512):0
workspace_total=main+drop+16MiB
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| bmm1→vec1 | Cube Fixpipe | UB score buffer | FP32 row-major | Vec1 | Wait/SetCrossCore | taskId槽 | realM/N mask |
| vec1→bmm2 | Vec1 | L1 P buffer | compute/NZ | Cube bmm2 | CROSS_CORE_SYNC_FORWARD | task槽 | Cube WaitCrossCore |
| bmm2→vec2 | Cube Fixpipe | UB或GM bmm2Res，3槽 | FP32 | Vec2 | Set/WaitCrossCore | mod3槽 | consumer后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cube/AIV | `WaitCrossCore/SetCrossCore` | Fixpipe/P/PV | opposite core | UB/L1/GM buffers | dualDst M偶数、N 8元素对齐后mask |
| intra-core | HardEvent MTE/M/FIX/V | copy/mmad/vector | next pipe | L1/L0/UB | runtime realM/N/K |
| online state | mod3 max/sum/exp | Vec1/Vec2 | later S2 tile | FP32 state | final `LastDiv` |
| output | `CAST_ROUND`+MTE3 | FP32 accumulator | GM | attention rows | max/sum FP32；O target dtype |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 varlen/regbase cpp | `DoOpTiling/GetWorkspaceSize/GetTilingKey` | tile/key/workspace |
| Kernel entry/router | apt+entry headers | `flash_attention_score_regbase` | noquant dtype实例 |
| Device body/helper | local kernel_train.h | `Process` | 四阶段调度 |
| Device body/helper | common cube/vec headers | `IterateBmm1/2,ProcessVec1/2` | 内存/同步/精度 |

### Route R-002：optionalDn RegBase MIX

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | API cpp | V5 helper | FP16/BF16、无额外PSE/RoPE/drop等 | L0 |
| Tiling | varlen cpp | `AnalyzeOptionalDn` | 完整Dn谓词与avg>=1536 | optionalDn=1 |
| Device | cube header | `useDn` constexpr | D<=128/256档 | Dn/DnSplitK body |

#### Device 数据流

##### 函数级调用链

```text
同R-001 KernelTrain -> Cube::IterateBmm1 -> useDn ? IterateBmm1Dn/IterateBmm1DnSplitK -> same Vec/PV pipeline
```

##### 源码等价伪码

```text
保持R-001四阶段；区别仅bmm1：满足小D时以Dn layout全载/驻留L1，128<D<=256走DnSplitK；mask只允许压缩causal模式。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/workspace/UB | 同R-001 | 同R-001公式 | 同R-001 | 同R-001 | 同R-001 | 同R-001 | optional inputs减少 |
| L1/L0A/B | Dn specialization | S1=128,S2=128,D<=256 | Dn/fractal | MTE/Load | Mmad | AIC | L1全载/驻留；D>128 splitK |

##### 分层内存布局

```mermaid
flowchart LR
  GM["GM Q/K"] --> DNL1["Dn L1 full-load/resident"] --> SPLIT["L0 D or split-K"] --> L0C["FP32 L0C"] --> V["同R-001 AIV/PV流水"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total沿用R-001；D<=256且D=DV模板，D>128时主区按三槽分配。
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| Dn L1→L0 | MTE2 | full resident tile | input/Dn | Dn Matmul | MTE1/M events | AIC task | 完整optionalDn guard |
| 后续边 | 同R-001 | 同R-001 | 同R-001 | 同R-001 | cross-core | task槽 | runtime tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Dn bmm1 | L1/L0 HardEvents | Dn copy | Mmad | resident L1/L0 split | D>128 splitK FP32累加 |
| later | 同R-001 cross-core | QK/P/PV | next stage | buffers | actualSeq tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | varlen cpp | `AnalyzeOptionalDn` | 完整guard/key |
| Kernel entry/router | entry header | optionalDn template arg | useDn实例 |
| Device body/helper | cube header | `IterateBmm1Dn*` | Dn搬运/Matmul |

### Route R-003：empty 初始化

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | API cpp | V5 helper | 输出不全空 | L0 |
| Tiling | common tiling cpp | `IsEmptyInputRegbase` | Q/K/V任一空且输出非空 | Kernel1/100MiB |
| Router | apt cpp | KernelType1 | FP16/BF16/FP32 | EmptyTensorRegbase |

#### Device 数据流

##### 函数级调用链

```text
IsEmptyInputRegbase -> GetEmptyArgs -> apt<Kernel1> -> EmptyTensorRegbase::Init -> Process
```

##### 源码等价伪码

```text
按32B block划分former/tail/last-core，UB生成初值并写max/sum/attention有效域；核间区间不重叠。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM outputs | POD prefix | empty extents | target/FP32 | UB init | caller | stream | 32B+last tail |
| UB | local tile | body fixed | target/FP32 | Duplicate | DataCopy | AIV | mask tail |
| workspace | 0 | 100MiB requested/body unused | bytes | runtime | N/A | executor | L1/L0 N/A |

##### 分层内存布局

```mermaid
flowchart LR
  POD["empty POD"] --> UB["AIV UB"] --> GM["outputs"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total=100*1024*1024; L1=L0A=L0B=L0C=0
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| UB→GM | Duplicate | per-core extent | target/FP32 | DataCopy | pipe | AIV | last mask |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init/store | queue/pipe | UB | GM | disjoint ranges | max/sum FP32；tail POD |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | common tiling cpp | `IsEmptyInputRegbase` | key/workspace/partition |
| Kernel entry/router | apt cpp | key1 branch | dtype router |
| Device body/helper | empty regbase header | `Init/Process` | initialization |

## 算子特定优化

### Trick T-001：严格guard的optionalDn

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：causal压缩mask、无PSE/RoPE/drop、非FP32/FP8、D/DV同档且<=256、s1模板非64、平均S1/S2均>=1536。
- Golden 来源：[`attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp:246`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp#L246)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (s1AvgLen < NUM_1536 || s2AvgLen < NUM_1536) {
    return;
}
optionalDn = true;
```

#### 机制与设计原因

Dn改变Q/K进入L1/L0的layout及D方向切分；Host只在无额外融合负担、相同小D/Dv和足够长平均序列时实例化，避免把共享库中存在的Dn误当默认路线。

#### 迁移边界

- 适用：950 TND FP16/BF16 causal mask长序列。
- 前提：完整guard和key最后位一致。
- 失效：任一条件假回R-001。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只按D<=256开启Dn会绕过PSE/RoPE/drop、mask模式与平均长度guard。

### Trick T-002：四阶段跨核软件流水

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：非空RegBase MIX。
- Golden 来源：[`attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h:300`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h#L300)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (notLastThreeLoop) {
                RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, realCoreInnerIdx);
                if (this->sharedParams.splitCoreMode == 1) {
                    runInfo1.multiCoreIdxMod3 = realMultiCoreInnerIdxInS2 % 3;
                }
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                }
            }
            if (taskId > 0 && notLastTwoLoop) {
                if ASCEND_IS_AIV {
                    auto &runInfo3 = runInfo[(taskId + 3) & 3];
                    this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                        this->constInfo);
                }
            }
            if (taskId > 1 && notLast) {
                RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                if ASCEND_IS_AIC {
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    } else {
                        this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    }
                }
            }
            if (taskId > 2) {
                if ASCEND_IS_AIV {
                    RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                    }
                }
            }
            taskId++;
        }
```

#### 机制与设计原因

runInfo[4]和三个flush伪循环让同一task序列重叠bmm1、vec1、bmm2、vec2；Buffer的cross-core wait/set而非Host顺序保证AIC/AIV可见性，mod3状态与workspace三槽匹配。

#### 迁移边界

- 适用：950 RegBase训练body。
- 前提：1:2 core映射、SSBUF编译选项、三槽容量和flags一致。
- 失效：empty不使用该流水。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

仅复制四个函数调用而省略三次flush和cross-core协议会丢最后任务或覆盖活跃槽。

### Trick T-003：按D选择UB或三槽GM中转

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：`D>128`申请bmm2区；仅当 `D>128 && DTemplate>256` 时再申请vec2区。
- Golden 来源：[`attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp:295`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp#L295)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
workspaces[0] = static_cast<size_t>(((bmm2Bytes + vec2Bytes) * PING_PONG_VALUE) *
    multiCoreParamsRegbase_->get_coreNum());
```

#### 机制与设计原因

小D让Fixpipe直接写cross-core UB；大D把PV或Vector2累积放入512B对齐GM workspace，并以三槽配合四阶段流水，避免UB容量超限。

#### 迁移边界

- 适用：950非空路线。
- 前提：Host字节公式与 `GetC2Position`、mod3地址完全一致。
- 失效：D<=128时主workspace可为0，但仍有16MiB reserve及可选drop区。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把三槽乘数省略或用逻辑D替代模板对齐D会导致跨stage覆盖。

## 模板与实现对比

| 对比维度 | R-001 standard | R-002 optionalDn | R-003 empty | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | priority82其余非空TND | priority82严格长序列子集 | registry前empty | 空域/搬运特化 |
| 执行族/节点图 | MIX四阶段 | MIX四阶段，Dn bmm1 | AIV init | Cube路径差异使R1/R2分开 |
| M/N/K、tile与scheduler | S1/S2通常128，D分档 | 同但Dn/full-load/splitK | N/A | optionalDn guard |
| workspace/L1/L0/UB | D门槛+3槽 | 同公式，不同L1 layout | 100MiB | 物理容量 |
| AIC/AIV交接 | cross-core UB/L1/GM | 同协议 | N/A | Buffer sync types |
| 同步、精度与tail | FP32 state，四阶段flush | 同+Dn K tail | disjoint GM | template+runtime双界 |
| fallback/失败边界 | key失败 | guard假回R1 | 全输出空API快返 | 无旧V fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V5支持，arch22卡 | V5支持，arch22卡 | V5支持，本卡 |
| Host/Launcher | DAV2201 priority90/94 | 同A2 | DAV3510 priority81/82、Simplified POD |
| Kernel/资源 | classic general/SameAB | 同A2 | RegBase四阶段/optionalDn/SSBUF；V5不支持sink/softmax layout |

## 限制与失败边界

- 950 V5公开输入只有FP16/BF16/FP32；共享entry的HIFLOAT8/FP8分支对本API不可达，不能发布为V5能力。
- 文档明确950暂不支持sink与softmaxOutLayout；固定Host也不提交用户softmaxOut。
- D/Dv>768、非TND、sparse/PSE/actual-length不合法或14段key无实例均失败。
- empty请求100MiB；非空末尾16MiB，另有512B对齐主区/drop区；不得省略三槽。
- POD无整结构static_assert；48项数组、inactive清零和声明顺序不可改变。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV5.md` | 产品表/950限制 | 公开支持/不支持项 |
| 入口声明/约束 | `attention/flash_attention_score/op_api/aclnn_flash_attention_score.h` | V5声明 | ABI |
| 入口声明/约束 | `attention/flash_attn/op_api/aclnn_flash_attn.h` | `aclnnFlashAttn`声明 | 独立950统一接口，不属于VarLenScore版本族 |
| 入口实现 | `attention/flash_attn/op_api/flash_attn.cpp` | `l0op::FlashAttn` | 单独注册并添加 `FlashAttn` launcher，不进入 `FlashAttentionScore` 链 |
| 构建接线 | `attention/flash_attn/op_host/flash_attn_def.cpp` | `FlashAttn` / `this->AICore().AddConfig("ascend950")` | 独立统一接口的平台边界 |
| 入口声明/约束 | `attention/flash_attn_metadata/op_host/op_api/aclnn_flash_attn_metadata.h` | `aclnnFlashAttnMetadata`声明 | 独立metadata接口，不属于VarLenScore版本族 |
| 入口实现 | `attention/flash_attention_score/op_api/aclnn_flash_attention_score.cpp` | V5 helper | Host流程/不可达dequant/softmax缺口 |
| Host/Tiling/Launcher | `attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp` | platform/empty | DAV3510/empty |
| Host/Tiling/Launcher | `attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.h` | constants/base | templates/resources |
| Host/Tiling/Launcher | `attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.cpp` | Do/PostTiling | analyzer/blockDim/drop offset |
| Host/Tiling/Launcher | `attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp` | VarLenConst | key/optionalDn/workspace |
| Host/Tiling/Launcher | `attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_dropmask.cpp` | priority81 | memset/adapter POD |
| Host/Tiling/Launcher | `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h` | POD classes | ABI |
| Kernel entry/body | `attention/flash_attention_score/op_kernel/flash_attention_score_apt.cpp` | entry | KernelType router |
| Kernel entry/body | `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_template_tiling_key.h` | 14段key | 白名单 |
| Kernel entry/body | `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_entry_regbase.h` | regbase entry | dtype/core实例 |
| Kernel entry/body | `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h` | KernelTrain | 四阶段 |
| Kernel entry/body | `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_drop_mask_adapter_regbase.h` | adapter | drop转换 |
| Kernel entry/body | `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_empty_tensor_regbase.h` | EmptyTensor | empty |
| Kernel entry/body | `attention/common/op_kernel/arch35/flash_attention_score_kernel_base.h` | noquant base | buffers/Init |
| Kernel entry/body | `attention/common/op_kernel/arch35/flash_attention_score_block_cube.h` | FABlockCube | L1/L0/Mmad/Fixpipe/sync |
| Kernel entry/body | `attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h` | FABlockVecBase | softmax/PV/tail |
| Kernel entry/body | `attention/common/op_kernel/arch35/flash_attention_score_block_vec_train.h` | VecTrain | training outputs |
| 构建接线 | `attention/flash_attention_score/op_host/flash_attention_score_def.cpp` | ascend950 config | apt opFile |
| 构建接线 | `attention/flash_attention_score/op_host/CMakeLists.txt` | SSBUF/dependency | 构建 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Simplified四nested+empty POD | 顺序/48数组/raw清零/inactive0 | common tiling header/priority81 |
| 调用顺序 | preprocess→L0→post/ViewCopy | no dequant/outDtype0；softmax不提交 | V5 helper |
| workspace/内存 | D门槛、512B、3槽、drop、16/100MiB | offset/extent/coreNum/owner | VarLen GetWorkspace/PostTiling |
| 同步 | MIX1:2、四阶段、cross-core buffers | wait/set方向、flush、mod3 | KernelTrain/Cube/Vec |
| 精度与尾块 | FP32 QK/softmax/PV | cast round、realM/N/K、actualSeq | Cube/Vec bodies |
| 失败边界 | DAV3510/TND/key/D上限 | 不回退arch22/旧V/共享FP8 | IsCapable/key selector |