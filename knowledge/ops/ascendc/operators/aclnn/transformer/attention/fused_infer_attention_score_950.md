---
type: Operator
title: "FusedInferAttentionScore 950 ACLNN 算子知识"
description: "说明 aclnnFusedInferAttentionScoreV5 在 950 上的接口语义、Host/Tiling 分流、POD 与 arch35 MIX_CV Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v5.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v5.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_inner.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v4.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_host/arch35/fia_tiling_fullquant_mx.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/fia_tiling_data_public.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/fia_tiling_data_noquant_gqa.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/fia_tiling_data_fullquant_mx.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/fia_template_dispatcher.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/fia_kernel_fullquant_mx.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_kernel_infer.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_noquant_kernel_infer.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_mla_fullquant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_gqa_fullquant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_mx_fullquant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T00:00:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# FusedInferAttentionScore 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnFusedInferAttentionScoreV5GetWorkspaceSize` / `aclnnFusedInferAttentionScoreV5`，公开 ACLNN 两段式接口 |
| 计算语义 | `S=QK^T*scale+pse+mask`，逐行 softmax 后计算 `O=softmax(S)V`；覆盖 decode/prefill、page attention、RoPE、量化与 LSE |
| 输入输出 | Q、Key/Value 列表、可选 mask/长度/cache/量化/PSE/RoPE/前缀张量；输出 `attentionOut` 与可选 FP32 `softmaxLse` |
| 执行类型 | arch35 单 Kernel；R-001/R-002/R-004/R-005 为 `MIX_CV`，AIC 计算 QK/PV、AIV 完成 softmax/更新/量化/cast；R-003 为空域 `AIV_VECTOR` zero-op |
| Kernel 路线 | R-001 重构非量化 GQA；R-002 重构 MX 全量化；R-003 空域初始化；R-004 legacy antiquant IFA；R-005 legacy PFA |
| 核心资源 | GM、user workspace、L1/L0A/L0B/L0C、UB；AIC/AIV 通过 cross-core buffer/event 交接 |
| 来源闭包 | `source_closed`：V5 API→generated inner 边界→arch35 parser/checker/selector→POD/key→entry/router→body/helper→CMake 闭合 |
| 验证状态 | 固定提交源码提取；按任务要求未执行编译、测试、validator 或 review。 |

## 接口与语义

### 计算语义

对每个 batch、query head 与有效 query token，按 `G=numHeads/numKeyValueHeads` 映射到 KV head，分块形成 QK 分数，乘 `scaleValue`，再应用 PSE、显式 attention mask 或稀疏窗口。Vector 侧执行行最大值、指数与行和；跨 S2 块使用在线 softmax 更新旧 `(max,sum,O)`，之后与 V 做第二次矩阵乘并提交输出。`softmaxLseFlag` 打开时同时输出 FP32 LSE。V5 相对 V4 新增 `qStartIdxOptional`、`kvStartIdxOptional` 和 `pseType`，它们参与 PSE 地址与模式选择。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `query` | 由 BSH/BSND/BNSD/TND/NTD 等已校验布局解释 | FP16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN | 必选；给出 B、Q_N、Q_S、D；文档标明不支持非连续 tensor |
| `key`,`value` | `aclTensorList`；连续 KV、TensorList 或 page cache ND/NZ | FP16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN/INT4(INT32)/FLOAT4_E2M1 | K/V 列表、dtype、shape 必须配对；INT32 仅作为 INT4 打包 view |
| PSE/mask/长度 | PSE、mask、`actualSeqLengths*`、`qStartIdx`、`kvStartIdx` | PSE FP16/BF16；mask BOOL/INT8/UINT8；数组 INT64 | 空指针表示关闭或采用 shape 长度；TND 长度按累计序列解释；起始索引供 PSE 定位 |
| 量化参数 | dequant/quant/offset、K/V antiquant、query dequant、K-RoPE antiquant | 由 checker 组合约束 | 是否存在及 query/K/V dtype 决定 no/anti/full/post quant；不允许拼出未支持组合 |
| cache/前缀/RoPE/padding/sink | block table、shared prefix、Q/K RoPE、左右 padding、learnable sink | block table INT32；其余见公开表 | block table 决定 PA；RoPE/前缀/padding 改变地址和有效长度；注册非量化模板会排除部分特征 |
| 属性 | `numHeads,scaleValue,preTokens,nextTokens,inputLayout,numKeyValueHeads,sparseMode,innerPrecise,blockSize,antiquantMode,softmaxLseFlag,key/valueAntiquantMode,queryQuantMode,pseType` | 标量 | parser/checker 冻结为 `FiaTilingInfo`；非法枚举、尺寸、dtype/layout 组合失败关闭 |
| 输出 | `attentionOut`,`softmaxLse` | attentionOut 依量化组合；LSE FP32 | LSE 关闭时 API 创建 shape `{0}` 的 FP32 占位；空域路径按 `needInit` 清零有效输出 |

### 入口与版本

950 最新公开入口是 V5。`docs/aclnnFusedInferAttentionScoreV5.md` 的产品表仅对 Ascend 950PR/950DT 标为支持，并明确 A2/A3 不支持；README 同样把 Ascend 950 示例指向 V5。V5 API 的普通入口与最大 workspace 边界最终都调用 generated `aclnnInnerFusedInferAttentionScore*`，固定仓从 OpDef/Host Tiling 重新闭合下游链路。

## Host/Launcher 执行流程

### 参数校验与前处理

`TensorPreProcess` 在 K/V 首元素均为 INT32 时逐元素把末轴乘 8，并把 view dtype 改为 INT4；`PrefixTensorPreProcess` 对共享 K/V 前缀执行相同变换。V5 对 K/V、K/V antiquant scale 和 `keyRope` 做连续性检查：旧 opbase 缺少弱符号 `NnopbaseSupportTensorV2` 且发现非连续输入时返回 `ACLNN_ERR_INNER_TILING_ERROR`。LSE 关闭时创建 `{0}` FP32 占位。最大 workspace API 还把五个 `aclIntArray` 转为临时 tensor，并在调用后逐一释放。

Host 侧 `DoOpTilingFusedInferAttentionScore` 读取 NPU arch；`DAV_3510` 固定进入 `TilingFusedInferAttentionScoreV4`。该函数执行 `FiaInfoParser::Parse`、`FIAChecker::Process`，校验过程不得修改 `FiaTilingInfo`，然后先调用模板 registry；registry 全部拒绝时才进入旧 `FusedInferAttentionScoreTilingImpl::DoOpTiling`。

### 参数、Workspace 与生命周期

第一段 ACLNN 返回 workspace 字节数与 executor；调用方分配 workspace 后再用第二段接口提交。executor、临时 tensor、输出和 workspace 必须存活到 stream 完成。重构 R-001/R-002 workspace 为 system 区，加每核三重预载的 BMM2/Vec2 FP32 slice；FD 再加 `accumOut+2*logSumExp`。R-004 antiquant 额外按核预留 256 KiB L1 模拟区与 128 KiB L0C 模拟区、100 MiB dump 区，PA 再加每核两份 64 B；R-003 只保留 system workspace。

### 启动图与失败传播

```text
V5GetWorkspaceSize
  -> INT4 view / contiguous guard / LSE placeholder
  -> generated aclnnInnerFusedInferAttentionScoreGetWorkspaceSize
  -> DoOpTilingFusedInferAttentionScore(DAV_3510)
     -> Parse -> FIAChecker -> registry(p28,p210)
        -> success: reconstructed POD, isReconstructTemp=true
        -> all reject: legacy DoOpTiling, isReconstructTemp=false
  -> executor records fused_infer_attention_score
V5(workspace,executor,stream) -> generated inner submits one MIX_CV kernel
```

API 临时对象创建、parser/checker、任一 Tiling setter、POD capacity、workspace、tiling key 或 executor 提交失败都阻断后继计算。非空路径因 Device 使用 `SyncAll` 设置 batch schedule；空域提前返回，不设置该 schedule。

## Route 与参数设计

### 分核与资源计算

共同依赖顺序为 `layout/shape -> B,N2,G,S1,S2,D,Dv,ropeD -> actual lengths/mask -> quant/feature route -> sOuter/sInner -> core metadata -> workspace/POD/key`。重构模板用 `CalcTschBlockDim(aivNum,aicNum,aivNum)` 编码 CV 核配比，并将 AIC 的 FA 工作拆分写入 `FAMetadata[36][8]`、AIV 的 FD 归约拆分写入 `FDMetadata[72][8]`。legacy 也以平台全部 AIV/AIC 生成 blockDim；S1 外切与 FD 标志进入 key。

重构普通 workspace 公式为：

```text
W = Wsys + 3 * usedCore * (bmm2Bytes + vec2Bytes)
    + isFD * 4 * (accumOutElems + 2 * logSumExpElems)
bmm2Bytes = (Dv > 128) ? mSize * alignedDvForBmm2 * 4 : 0
vec2Bytes = (alignedDv > 256) ? mSize * alignedDv * 4 : 0
```

最大 workspace 另按每核两次预载、两个 BMM 的 64 B 双缓冲和最大 FD 临时区计算。PA 的 legacy normal 路径加 `coreNum*2*2*64`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 重构非量化 GQA | registry priority 28；`quantMode=31,isReconstructTemp=1` | 非空；`FiaQuantMode::NO_QUANT`；非 split-RoPE D512/D128；无 prefix/PSE/ALiBi/padding/postquant/sink/D 不等/TensorList；sparse/layout/D/S1-out guard 全通过 | 任一 false 继续 p210，再落 legacy |
| 2 | R-002 重构 MX 全量化 | registry priority 210；`quantMode=19/20,isReconstructTemp=1` | 非空且 `fullQuantMode==QKV_MXFP8_FULL_QUANT` | false 落 legacy；tiling setter 失败终止 |
| 3 | R-003 空域初始化 | `emptyTensor=1` | registry 两模板都拒绝空域；legacy `DoOpTiling` 首先调用 `SetEmptyTensor` 并提前成功返回 | 非空继续 R-004/R-005 |
| 4 | R-004 legacy antiquant IFA | `quantMode=0/2/6,isReconstructTemp=0` | registry 均拒绝；合法 antiquant 被 key 归并为 per-channel、K-per-channel/V-per-token 或 K-per-token；entry 判定 `<15` | 非 antiquant 进入 R-005；未编译 key 失败 |
| 5 | R-005 legacy PFA | `quantMode=17/18/19/20/30/31,isReconstructTemp=0` | registry 未接管的合法 no/full quant、MLA/特征组合；entry 判定 `>=15` | 无 Host fallback；未编译 key 失败 |

```text
registry p28 nonquant capable ? R-001
  : registry p210 MX fullquant capable ? R-002
  : emptyTensor ? R-003
  : legacy key.quantMode < 15 ? R-004 : R-005
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | p28 + `isReconstructTemp=1` | FP16/BF16 noquant；布局 BSH/BSND/BNSD/TND；D=128 或 combine-RoPE D128；精确 feature/sparse guard | 单 arch35 MIX_CV | `FlashAttentionNoQuantGqaKernel<FANoQuantGqaBlockCube,FANoQuantGqaBlockVec,FiaBlockVecFlashDecode>` | metadata 驱动；BMM2 直写 UB；cross-core buffer；FD workspace | `min(end,shape)`；拒绝回 legacy |
| R-002 | p210 + mode 19/20 + reconstruct | MXFP8 full quant；非空；其余 checker/tiling 必须成功 | 单 arch35 MIX_CV | `FlashAttentionFullQuantMxKernel<FAFullQuantMxBlockCube,FAFullQuantMxBlockVec,FiaBlockVecFlashDecodeFullQuant>` | prefill `useDn=true`；BMM2 位置编译期选择；Dv>256 splitD | metadata tail；拒绝回 legacy |
| R-003 | `emptyTensor=1` | 输出逻辑元素为 0 或需显式初始化的空输入组合 | 单 `AIV_VECTOR` zero-op；PFA/IFA entry 均在 constexpr 空域分支后立即返回 | `INVOKE_PFA_ZERO_OP_IMPL_V2` | 仅 system workspace；AIV 用 UB fill/CopyOut 分段清零 attention/LSE，无 L1/L0/Cube | `singleCoreSize` 和总 size 限界；立即 return |
| R-004 | mode 0/2/6 | FP query + INT8/INT4/HIFLOAT8 等合法 antiquant 组合；legacy checker/tiling 接线 | RegBase IFA | `FlashAttentionScoreAntiquantKernel` 及其 dtype 模板 | GM 模拟 L1/L0C slice；MIX AIC 1:2；FD/PA workspace | IFA POD tail；无第二 fallback |
| R-005 | mode 17/18/19/20/30/31 | R-001/R-002 未接管的合法 full/noquant、MLA、prefix/PSE/padding/sink/TensorList 等 | RegBase PFA | legacy noquant/fullquant/MLA 模板 | PFA BMM/Vector 管线；normal/FD/PA workspace；`isReconstructTemp=false` | PFA POD tail；无 Host fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 重构非量化基础 guard | 非空、`quantMode==NO_QUANT`、非 split-RoPE D512/D128 | 当前 GQA body 固定 noquant 与 D128 类模板 | R-001 | registry 下一项/legacy | `FiaTilingNonQuantArch35::IsCapableBasicCheckGqa` |
| feature guard | 无 prefix/PSE/ALiBi/padding/postquant/sink/D 不等/TensorList | 新 POD/body 未接这些路径或地址模式 | R-001 | R-005 | `IsCapableFeatureCheckGqa` |
| sparse/layout guard | 排除 sparse 1/2/4；sparse0 不得传 mask；layout 限 BSH/BSND/BNSD/TND；D=128 或 combine-RoPE D128；无 S1 外切 | metadata 与编译 tile 的地址/掩码域有限 | R-001 | R-005 | `IsCapableSparseLayoutCheckGqa` |
| MX 全量化 guard | `fullQuantMode==QKV_MXFP8_FULL_QUANT` | 重构 body 的 scale/POD/输入类型固定为 MXFP8 | R-002 | legacy R-005 | `FiaTilingFullQuantMxArch35::IsCapable` |
| entry 阈值 | `quantMode < 15` / `>=15` | 一个公开 kernel 内静态选择 IFA 或 PFA entry | R-004/R-005 | 无运行时重选 | `fused_infer_attention_score` |
| reconstruct bit | 新模板置 true，fallback 置 false | PFA entry 用同一 dtype/key 区分新 dispatcher 与旧 body | R-001/R-002/R-005 | 编译分支固定 | `UpdateTilingKeyInfo`,`prompt_flash_attention_FIAS_regbase` |
| batch schedule | 非空路径设置 mode 1 | `SyncAll` 需要所有核同时启动 | R-001/R-002/R-004/R-005 | setter/调度失败 | `FusedInferAttentionScoreTilingImpl::DoOpTiling`,`CalcScheduleMode` |

### POD/ABI

重构 POD 由普通 C++ struct 声明；固定仓没有提供平台编译后的 `sizeof/alignof/offsetof` 数字，因此只记录字段顺序和声明类型，不推测 padding。legacy 的 `PromptFlashAttentionTilingDataV2`、`IncreFlashAttentionTilingDataRegbase`、`FlashAttentionScoreSimplifiedTilingData` 由依赖 POD/注册宏消费，固定仓同样未暴露完整数字 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `stridesParams` | `bnStride,n2Stride` | uint64/64（逐字段） | 声明顺序；数值 offset 不可见 | elements/bytes 依地址公式 | arch35 common tiling | Q/K/V/scale address | 乘法检查 |
| `FiaBaseParams` | `bSize,t1Size,t2Size,n2Size,gSize,s1Size,s2Size,dSize,dSizeV,dSizeRope,actualSeqLengthsQSize,actualSeqLengthsKVSize` | uint32/32（逐字段） | 同上 | count/elements | `SetTilingData` | common kernel Init/loop | parser 窄化 |
| 同上 | `scaleValue` | float/32 | 同上 | scalar | API double→Host float | Vec softmax | FP32 scale |
| 同上 | `isKvContinuous,isSoftMaxLseEnable` | uint8/8（逐字段） | 同上 | flag | parser/tiling | address/CopyOut | 0/1 |
| 同上 | `coreNum,outputLayout` | uint32/32（逐字段） | 同上 | count/enum | split/tiling | scheduler/address | core metadata 上限 |
| 同上 | `keyStrides,valueStrides,kRopeStrides,kScaleStrides,vScaleStrides` | `stridesParams` | 嵌套声明顺序 | stride | tiling | GM address | 非对应特征时 inactive |
| `FiaAttenMaskParams` | `sparseMode` | uint8/8 | 声明顺序 | enum | checker/tiling | Vec mask | enum 已校验 |
| 同上 | `preTokens,nextTokens` | int32/32（逐字段） | 同上 | tokens | int64 attrs 窄化 | row window | 窄化/裁剪 |
| 同上 | `attenMaskBatch,attenMaskS1Size,attenMaskS2Size` | uint32/32（逐字段） | 同上 | elements | mask tiling | mask address | shape 限界 |
| 同上 | `isRowInvalidOpen,isExistRowInvalid` | uint8/8（逐字段） | 同上 | flag | tiling | row guard | 0/1 |
| `FiaPseParams` | `pseShiftByBatch` | uint8/8 | 声明顺序 | flag | PSE tiling | PSE address | 0/1 |
| 同上 | `pseS1Size,pseS2Size,pseStride,qStartIdx,kvStartIdx` | uint32/32（逐字段） | 同上 | elements/index | V5 attrs/tiling | PSE reader | 索引在 shape 内 |
| `FiaSystemPrefixParams` | `isActualSharedPrefixLenNull,prefixSeqInnerSize` | uint8/8 + uint32/32 | 声明顺序；padding 数值不可见 | flag/tokens | prefix tiling | KV address | 非 prefix inactive |
| `FiaPageAttentionParams` | `paLayoutType,blockSize,maxBlockNumPerBatch` | uint8/8 + 2×uint32/32 | 声明顺序 | enum/elements/blocks | PA tiling | block-table address | block index 限界 |
| `FiaLeftPaddingParams` | `isQHasLeftPadding,isKVHasLeftPadding` | uint8/8（逐字段） | 声明顺序 | flag | padding tiling | Q/KV address | 0/1 |
| `FiaPostQuantParams` | `isPostQuantPerChnl,isPostQuantBF16` | uint8/8（逐字段） | 声明顺序 | flag | quant tiling | output epilogue | 0/1 |
| `FiaWorkspaceParams` | `accumOutSize,logSumExpSize` | uint32/32（逐字段） | 声明顺序 | FP32 elements | FD workspace tiling | FD vector | bytes=elements×4 |
| `FiaS1OuterSplitCoreParams` | `enableS1OutSplit,totalSize` | bool/实现位宽 + uint64/64 | 声明顺序；padding 不可见 | flag/elements | split tiling | scheduler | 新 noquant guard 要求 false |
| `FiaEmptyTensorParams` | `singleCoreSize,needInit,totalOutputSize,totalSoftMaxLseOutputSize` | uint32/32,uint8/8,2×uint64/64 | 声明顺序 | elements/flag | empty tiling | zero-op | 每核范围裁剪 |
| `FiaMetaData` | `FAMetadata[36][8]` | uint32/32 | 紧邻二维数组 | index/count | split-core v2 | AIC scheduler | index 0..7 |
| 同上 | `FDMetadata[72][8]` | uint32/32 | 紧邻二维数组 | index/count | FD split | AIV FD scheduler | index 0..7 |
| `FusedInferAttentionScoreTilingData` | `baseTiling,fiaMetaData` | `NoQuantTilingArch35` + `FiaMetaData` | 成员声明顺序 | POD | R-001 tiling | `run_fia_noquant_gqa_kernel` | `offsetof(...,fiaMetaData)` |
| `FusedInferAttentionScoreFullQuantTilingData` | `baseTiling,fiaMetaData` | `FullQuantTiling` + `FiaMetaData` | 成员声明顺序 | POD | R-002 tiling | `run_fia_fullquant_mx_kernel` | `offsetof(...,fiaMetaData)` |

`FAMetadata` 的 8 列依次是 core-enable、BN2 start、M start、S2 start、BN2 end、M end、S2 end、首个 FD workspace index；`FDMetadata` 的已命名列依次是 core-enable、BN2 index、M index、S2 split count、workspace index、M start、M count，第 7 列在该固定头文件中未命名，不能推测语义。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | registry p28 + reconstruct bit | `FiaTilingNonQuantArch35` | `FusedInferAttentionScoreTilingData` | `_apt`→PFA entry reconstruct branch | `run_fia_noquant_gqa_kernel`→`FlashAttentionNoQuantGqaKernel` | operator CMake arch35 common sources/deps |
| R-002 | registry p210 + mode 19/20 + reconstruct bit | `FiaTilingFullQuantMxArch35` | `FusedInferAttentionScoreFullQuantTilingData` | `_apt`→PFA entry reconstruct branch | `run_fia_fullquant_mx_kernel`→`FlashAttentionFullQuantMxKernel` | same |
| R-003 | fallback `emptyTensor=1` | `FusedInferAttentionScoreTilingImpl::SetEmptyTensor` | legacy empty/init fields | `_apt`→IFA/PFA zero-op branch | `INVOKE_PFA_ZERO_OP_IMPL_V2` | same |
| R-004 | fallback mode 0/2/6 | legacy `ComputeTilingData/GenTilingKey` | simplified/IFA RegBase POD | `_apt`→`incre_flash_attention_FIAS_regbase` | antiquant BaseAPI bodies | IFA dependency |
| R-005 | fallback mode 17/18/19/20/30/31 | legacy `ComputeTilingData/GenTilingKey` | PFA/RegBase POD | `_apt`→`prompt_flash_attention_FIAS_regbase` | legacy PFA no/fullquant/MLA bodies | PFA dependency |

## Kernel 实现

### 公共执行流程

`fused_infer_attention_score` 的 12 个编译期字段为 layout、config、PSE mode、quant mode、mask、RoPE、KV layout、FD、empty、KV prefix、S1 外切、reconstruct。它不在 Device 重新评估 Host guard，只按 `quantMode >= 15` 静态调用 PFA，否则调用 IFA。两侧先用 `GetUserWorkspace` 跳过 system 区；PFA 再用 reconstruct bit 选择新 dispatcher 或旧 body。非空 body 以 core metadata 划分 GM 区域，AIC/AIV 通过 cross-core buffer、MTE/FIX event 和必要的 `SyncAll` 交接，尾块始终由有效区间限制。

### Route R-001：重构非量化 GQA

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling registry | `attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp` | `FiaTilingNonQuantArch35::IsCapable` | 三组 guard 全 true，priority 28 | 生成 mode31、reconstruct=true |
| Kernel router | `attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h` | `prompt_flash_attention_FIAS_regbase` | FP16/FP16/FP16 或 BF16/BF16/BF16 且 reconstruct=true | 调用 noquant dispatcher |

#### Device 数据流

##### 函数级调用链

```text
fused_infer_attention_score<...,31,...,true>
  -> prompt_flash_attention_FIAS_regbase
  -> run_fia_noquant_gqa_kernel
  -> FlashAttentionNoQuantGqaKernel::Init
  -> FlashAttentionNoQuantGqaKernel::Process
     -> FANoQuantGqaBlockCube / FANoQuantGqaBlockVec
     -> FiaBlockVecFlashDecode (isFd)
```

##### 源码等价伪码

```text
read baseTiling and fiaMetaData via offsetof
if AIC metadata enabled: for assigned (bn2,m,s2): QK -> workspace/UB -> PV
if AIV metadata enabled: wait QK; mask/scale/online-softmax; wait PV; update O
if isFd: write FP32 partial O/max/sum; SyncAll; assigned AIV reduces splits
cast and CopyOut only valid M,Dv prefix; optionally write LSE
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Q/K/V GM | metadata `(bn2,m,s2)` + POD strides | assigned token/head tile | FP16/BF16, API layout | API/executor | Cube CopyIn | stream | tail按 metadata end |
| BMM/Vec workspace | `coreSlice + preloadSlot*sliceBytes` | `3*(bmm2Bytes+vec2Bytes)`/core | FP32 | AIC/Vec | downstream Vec | tile/preload slot | 复用前 cross-core wait |
| FD workspace | POD `accumOutSize/logSumExpSize` + FD index | O + max + sum FP32 | FP32 | each FA split | FD AIV | kernel | two LSE arrays |
| L1/L0/UB | block class compile-time tile | config/Dv bounded | cube fractal/FP32 | CopyIn/MMAD | softmax/update | loop | double/preload buffer |

##### 分层内存布局

```text
GM Q/K/V -> L1 -> L0A/L0B -> L0C(QK)
                                 -> UB Vec1(mask/softmax)
UB P + L1 V -> L0C/UB(PV) -> UB Vec2(update/cast) -> GM O/LSE
FD: per-split GM FP32 O,max,sum -> AIV reduce -> final GM
```

##### 字节账、交接边与复用顺序

Host 的 `bmm2Bytes/vec2Bytes` 与 Device config/Dv 模板一致；每核按三份预载 slice 分离正在生产、正在消费和下一块。`FAMetadata` 绑定 AIC 区间，`FDMetadata` 绑定 AIV 归约与 workspace index；覆盖旧 slot 前必须完成对应 `WaitCrossCore`。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| QK/PV→Vector | `Buffer::SetCrossCore/WaitCrossCore` | AIC block | AIV Vec block | BMM1/BMM2 buffer | FP32 累加；有效 S2 mask |
| pipe 内复用 | MTE/FIX event | DMA/MMAD | 下一阶段/下一 loop | L1/L0/UB | event 后才覆盖 slot |
| FD | `SyncAll` + metadata | FA splits | FD AIV | FP32 O/max/sum | 在线 rescale 后归约，尾 M/Dv 裁剪 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host selector | `attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp` | `IsCapable`,`GenTilingKey`,`CalcWorkspaceSize` | p28、mode31、reconstruct bit、metadata/workspace |
| entry/router | `attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h` | `prompt_flash_attention_FIAS_regbase` | reconstruct=true 转专用 dispatcher |
| body | `attention/common/op_kernel/arch35/fia_template_dispatcher.h` | `run_fia_noquant_gqa_kernel` | `useDn=false,bmm2Write2Ub=true,splitD=false` |
| helper | `attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h` | `FlashAttentionNoQuantGqaKernel` | metadata、cross-core、FD 和尾块实现 |

### Route R-002：重构 MX 全量化

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling registry | `attention/common/op_host/arch35/fia_tiling_fullquant_mx.cpp` | `FiaTilingFullQuantMxArch35::IsCapable` | 非空且 fullQuantMode 为 QKV_MXFP8_FULL_QUANT，priority 210 | mode19/20、reconstruct=true |
| Kernel router | PFA arch35 entry | `prompt_flash_attention_FIAS_regbase` | MXFP8 query/key 与 FP16/BF16 output，reconstruct=true | fullquant MX dispatcher |

#### Device 数据流

##### 函数级调用链

```text
fused_infer_attention_score<...,19|20,...,true>
  -> prompt_flash_attention_FIAS_regbase
  -> run_fia_fullquant_mx_kernel
  -> FlashAttentionFullQuantMxKernel::Init/Process
     -> FAFullQuantMxBlockCube / FAFullQuantMxBlockVec
     -> FiaBlockVecFlashDecodeFullQuant (isFd)
```

##### 源码等价伪码

```text
read MX POD + metadata; load Q/K/V and dequant scales
AIC: MXFP8 QK with scale -> FP32 score
AIV: scale/mask/softmax; prepare P and scales
AIC/AIV: PV, online update, optional FD partial write/reduce
convert to FP16/BF16 attentionOut; write LSE when enabled
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Q/K/V/scale GM | metadata tile + POD strides | tile plus scale blocks | FP8 E4M3/MX scale | executor | MX Cube/Vec | stream | scale地址与数据tile同界 |
| BMM2 workspace | core/preload slice | `3*bmm2Bytes`/core when Dv>128 | FP32 | MX Cube | Vec2 | slot | 复用前 flag |
| Vec2 workspace | adjacent core slice | `3*vec2Bytes`/core when Dv>256 | FP32 | first D slice | merge/copy | slot | splitD 时非零 |
| FD workspace | metadata index | accum O + two LSE arrays | FP32 | FA splits | FD AIV | kernel | batch barrier |

##### 分层内存布局

```text
GM MXFP8 Q/K + scale -> Cube QK(FP32) -> Vec softmax
GM MXFP8 V + P -> Cube/UB PV(FP32) -> Vec update -> GM FP16/BF16
                 Dv>256: split-D intermediate workspace
```

##### 字节账、交接边与复用顺序

`CalcWorkspaceSize` 与非量化共用 `Wsys+3*core*(bmm2Bytes+vec2Bytes)+FD` 形状。`GetC2Position` 在编译期决定 BMM2 写 VECCALC 还是其他位置；`splitD=(DvTemplate>256)` 决定额外 Vec2 合并区，不能只按 runtime Dv 改写。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MX QK/PV | cross-core buffer | AIC | AIV | BMM1/BMM2 tile | scale 后 FP32 softmax/累加 |
| split-D | event + workspace | first D slice | Vec merge | PV partial | Dv>256 才启用，尾 Dv mask |
| FD | `SyncAll` | FA splits | FD vector | O/max/sum | FP32 归约后 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host selector | `attention/common/op_host/arch35/fia_tiling_fullquant_mx.cpp` | `IsCapable`,`UpdateTilingKeyQuantMode`,`CalcWorkspaceSize` | p210，prefill 19/decode 20，metadata/workspace |
| entry/router | PFA arch35 entry | reconstruct MX branches | FP8 E4M3→FP16/BF16 dispatcher |
| body | `fia_template_dispatcher.h` | `run_fia_fullquant_mx_kernel` | prefill `useDn=true`，C2 position 与 splitD 编译期化 |
| helper | `fia_kernel_fullquant_mx.h` | `FlashAttentionFullQuantMxKernel` | MX 数据流、flags、FD 与尾块 |

### Route R-003：空域初始化

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| registry | 两个 arch35 tiling | `IsCapable` | `emptyTensorFlag` 均返回 false | 进入 legacy |
| fallback Host | `fused_infer_attention_score_tiling_impl.cpp` | `DoOpTiling`,`SetEmptyTensor` | empty true 时设置空域 tiling 后提前 return | key `emptyTensor=1` |
| Device | IFA/PFA entry | zero-op branch | `if constexpr(emptyTensor==true)` | 初始化并 return |

#### Device 数据流

##### 函数级调用链

```text
fused_infer_attention_score<...,emptyTensor=true,...>
  -> IFA or PFA entry
  -> INVOKE_PFA_ZERO_OP_IMPL_V2<output type>
  -> zero valid attentionOut/LSE slice
```

##### 源码等价伪码

```text
start = coreIdx * singleCoreSize
end = min(start + singleCoreSize, totalOutputSize)
if needInit: fill attentionOut[start:end] with zero
if LSE enabled: fill assigned LSE range with zero
return before attention math
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| attentionOut GM | `core*singleCoreSize` | min(singleCoreSize,remain) | output dtype/layout | zero-op | caller | stream | 尾核裁剪 |
| LSE GM | per-core assigned offset | total LSE elements | FP32 | zero-op | caller | stream | flag false 时不写 |
| workspace | user begins after system area | `Wsys` only | unused user area | Host | entry | executor | 无 BMM slice |

##### 分层内存布局

```text
constant zero -> UB fill tile -> GM attentionOut
                         `----> GM softmaxLse (enabled only)
```

##### 字节账、交接边与复用顺序

Host 空域仅返回 default system workspace。Device 不读取 Q/K/V，不建立 BMM producer-consumer 边；每核区间互斥，最后一核以总元素数截断。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| fill/copy | zero-op 内部 vector pipe | UB zero | GM CopyOut | 每核输出区间 | 零按目标 dtype；last core 裁剪 |
| return | compile-time branch | zero-op | kernel end | 避免访问空 Q/K/V | 无 softmax 数值路径 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | legacy arch35 tiling impl | `SetEmptyTensor`,`DoOpTiling` | 空域早返回并只保留 system workspace |
| entry | IFA/PFA arch35 entry | `emptyTensor` constexpr branch | 选择 output dtype zero-op |
| body | included zero implementation | `INVOKE_PFA_ZERO_OP_IMPL_V2` | 分核初始化有效 attention/LSE 区间 |

### Route R-004：legacy antiquant IFA

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| fallback Host | legacy arch35 tiling impl | `UpdateTilingKeyQuantMode` | antiquant 映射到 mode 0/2/6，reconstruct=false | 生成 IFA key/POD |
| top entry | `_apt.cpp` | `fused_infer_attention_score` | `quantMode < 15` | `incre_flash_attention_FIAS_regbase` |
| IFA entry | arch35 IFA entry | dtype template guards | 合法 FP query、quant KV、output 组合 | antiquant BaseAPI body |

#### Device 数据流

##### 函数级调用链

```text
fused_infer_attention_score<...,0|2|6,...,false>
  -> incre_flash_attention_FIAS_regbase
  -> PARSE_PARAMS_AntiQuant
  -> FlashAttentionScoreAntiquantKernel<dtype,mode,...>::Init/Process
  -> MM1 -> Vec1(dequant/softmax) -> MM2 -> Vec2/update/CopyOut
```

##### 源码等价伪码

```text
load packed/quantized K,V and matching scale/offset
dequantize the current K/V tile according to mode 0/2/6
QK -> scale/mask/online softmax -> PV
merge previous O/max/sum; optional FD reduce; cast/copy valid output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| K/V GM | layout/PA block address | quant tile | INT8/INT4/HIFLOAT8 etc. | executor | antiquant CopyIn | stream | INT4 view axis ×8 |
| scale/offset GM | per tensor/channel/token/head/page address | checker-defined shape | scale dtype | executor | Vec dequant | stream | mode与shape绑定 |
| emulated L1 workspace | `user + core*256KiB` | 256 KiB/core | byte/quant tile | CopyIn | Cube/Vec | core | core slices disjoint |
| emulated L0C workspace | after all L1 slices + `core*128KiB` | 128 KiB/core | FP32/intermediate | Cube | Vec | core | 复用前 event |
| dump/FD/PA | following fixed slices | 100 MiB + optional FD + PA flags | mixed | body | body/FD | kernel | Host精确容量 |

##### 分层内存布局

```text
GM quant K/V + scale/offset -> user-workspace L1/L0C emulation
Q + dequant K -> QK FP32 -> UB softmax -> dequant V/PV -> GM O
FD partial O,max,sum -> SyncAll -> final output
```

##### 字节账、交接边与复用顺序

`SetWorkspaceAntiQuant` 的固定区间顺序是 system、所有 core 的 256 KiB L1、所有 core 的 128 KiB L0C、可选 FD、100 MiB dump、可选 PA 两份 64 B/core。Device offset 必须与此 producer 顺序一致；INT4 物理载荷使用 API 已扩展后的逻辑末轴。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dequant→MMAD | queue/event | Vector/MTE | Cube | reconstructed tile | scale/offset 在乘法前应用 |
| Cube→Vector | cross-core flag | AIC | AIV | QK/PV workspace | FP32 softmax/累加 |
| FD | `SyncAll` | split cores | reducer | O/max/sum | 尾 S2/M/Dv mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | legacy arch35 tiling impl | `UpdateTilingKeyQuantMode`,`SetWorkspaceAntiQuant` | mode 0/2/6 与精确 workspace |
| router | `_apt.cpp` | `quantMode < FullQuantTiling` | 静态进入 IFA |
| body | IFA arch35 entry | `incre_flash_attention_FIAS_regbase` | dtype guard、POD parser、antiquant kernel 实例 |

### Route R-005：legacy PFA

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| registry/fallback | arch35 V4 top | `TilingFusedInferAttentionScoreV4` | p28/p210 均返回失败 | legacy `DoOpTiling` |
| fallback key | legacy arch35 tiling impl | `UpdateTilingKeyInfo` | mode 17/18/19/20/30/31，reconstruct=false | PFA compiled key |
| top/device | `_apt.cpp`, PFA entry | threshold + reconstruct branch | mode>=15 且 reconstruct=false | legacy PFA no/fullquant/MLA body |

#### Device 数据流

##### 函数级调用链

```text
fused_infer_attention_score<...,quantMode>=15,...,false>
  -> prompt_flash_attention_FIAS_regbase
  -> PARSE_PARAMS_NoQuant / corresponding fullquant parser
  -> legacy FlashAttention noquant/fullquant/MLA kernel
  -> Init -> Process -> CopyOut / FD reduce
```

##### 源码等价伪码

```text
parse legacy PFA POD selected by tiling-key registration
for each assigned S1/S2/head tile: QK; apply feature-specific PSE/mask/RoPE
online softmax; PV; update FP32 output state
if FD reduce split states; apply postquant/cast; copy valid output/LSE
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | legacy POD layout/stride | assigned S1/S2/D tile | no/fullquant/MLA dtype | executor | PFA body | stream | page/prefix/rope branch-specific |
| BMM2 workspace | core/preload slot | normal formula, Dv thresholded | FP32 | Cube | Vec2 | slot | three preload slices |
| Vec2/FD workspace | following BMM2 slices | Dv>256 + FD O/max/sum | FP32 | body/splits | merge/reducer | kernel | batch schedule |
| PA flags | trailing core slice | `2*2*64`/core | cacheline flags | AIC/AIV | peer core | kernel | 64 B alignment |

##### 分层内存布局

```text
GM Q/K/V/features -> L1/L0 QK -> UB softmax -> L0/UB PV
      optional RoPE/PSE/prefix/PA address branches       |
GM O/LSE <- cast/postquant/update <- workspace/UB <------'
```

##### 字节账、交接边与复用顺序

Normal fallback 为 `Wsys+3*core*(bmm2Bytes+vec2Bytes)+FD`，PA 再加两 BMM×双缓冲×64 B/core。`isReconstructTemp=false` 是 POD 与 body 的兼容边界：不能把 legacy POD 交给 R-001/R-002 的 `offsetof` 解析器。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| QK/PV pipeline | MTE/FIX event + cross-core flag | Cube/DMA | Vector/next loop | local/workspace tile | FP32 accumulator，S1/S2 tail mask |
| online softmax | max/sum rescale order | current score tile | PV/update | previous O/max/sum | 先 rescale 再累加，禁止提前 cast |
| FD | `SyncAll` + batch schedule | all splits | reducer | FP32 partial state | only valid splits/lanes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | legacy arch35 tiling impl | `DoOpTiling`,`ComputeTilingData`,`GenTilingKey`,`GetWorkspace` | fallback POD/key/blockDim/workspace/schedule |
| router | `_apt.cpp` | `quantMode >= 15` | 静态进入 PFA |
| entry/body | PFA arch35 entry | `prompt_flash_attention_FIAS_regbase` reconstruct=false 分支 | legacy no/fullquant/MLA 已编译模板 |

## 算子特定优化

### Trick T-001：registry 渐进替换与 reconstruct bit 共用入口

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-004、R-005
- 触发条件：`Parse` 和 `FIAChecker::Process` 成功后，registry 按 priority 28/210 尝试重构模板；registry 未成功才进入 legacy `FusedInferAttentionScoreTilingImpl::DoOpTiling`。重构模板生成 `isReconstructTemp=true`，legacy 生成 false。
- Golden 来源：[`fused_infer_attention_score_tiling_v4.cpp:45-52`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v4.cpp#L45-L52)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo) == ge::GRAPH_SUCCESS) {
    return ge::GRAPH_SUCCESS;
} else {        // 假设，老的模板也注册，把else分支和下面的逻辑删掉
    OP_LOGD(context, "reconstruct template do not support, routing to old template.");
}

if (fiav4.DoOpTiling(context, fiaInfo) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
}
```

#### 机制与设计原因

同一 ACLNN、同一 kernel symbol 和同一大部分 key 维度下，可以只迁移已闭合的 noquant GQA 与 MXFP8 场景，同时保留复杂 prefix/PSE/MLA/量化组合。Host guard 与 Device reconstruct 分支共同保证不同 POD 不混读。

#### 迁移边界

- 适用：在同一公开入口和 kernel symbol 下逐步替换已闭合的 specialization。
- 前提：registry `IsCapable`、独立 POD producer、key bit、PFA entry dtype 分支和 dispatcher/body 同时接线。
- 失效：registry 内部 tiling 失败或 legacy 也失败时直接返回失败，不存在第三套 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要把 reconstruct 位当成性能提示而忽略；也不要让 legacy POD 配上 true，或让新 `FusedInferAttentionScore*TilingData` 配上 false。

### Trick T-002：元数据驱动的 AIC/AIV 异构拆分

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：重构模板完成 split-core 后，`SetSplitOutput` 将 FA 区间写入 `FAMetadata[36][8]`，并从同一 split 结果写入 `FDMetadata[72][8]`。
- Golden 来源：[`fia_tiling_nonquant_gqa.cpp:490-520`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp#L490-L520)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
void FiaTilingNonQuantArch35::SetSplitOutput(const split_core_v2::FAMetaData &splitRes)
{
    // FA Metadata Generate
    for (size_t i = 0; i < NPU_AIC_CORE_NUM; ++i) {
        if (i >= splitRes.usedCoreNum) {
            tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_START_INDEX] = (i == 0) ? 0 : splitRes.bN2End[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_START_INDEX] = (i == 0) ? 0 : splitRes.mEnd[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_START_INDEX] = (i == 0) ? 0 : splitRes.s2End[i - 1];
        // FA END
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_END_INDEX] = splitRes.mEnd[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
        // FA idx
        tilingData_.fiaMetaData.FAMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }

    // FD Metadata Generate
    const split_core_v2::FlashDecodeResult &fdRes = splitRes.fdRes;
    for (size_t i = 0; i < NPU_AIV_CORE_NUM; ++i) {
        if (i >= fdRes.fdUsedVecNum) {
            tilingData_.fiaMetaData.FDMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
```

#### 机制与设计原因

Host 可根据实际 B/N2/G/S1/S2 与 FD split 生成不规则工作区间，Device 避免再做昂贵的全局分配计算；AIC 负责 Cube 密集段，AIV 负责 softmax、尾块和归约。

#### 迁移边界

- 适用：Host 能预计算不规则 FA/FD 工作区间的 AIC/AIV 混合路线。
- 前提：metadata 数组规模、列含义、core index 映射、workspace index 和 `CalcTschBlockDim` 保持 Host/Device 一致。
- 失效：目标平台核数、CV 比或 metadata ABI 改变时，必须同时重做 producer 与 consumer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要推测未命名的 `FDMetadata[][7]`，不要把 36/72 直接替换成运行时核数，也不要交换 FA/FD metadata 的 consumer。

### Trick T-003：BMM2 输出位置与 split-D 编译期特化

#### 触发与源码依据

- 适用 Route：R-002；R-001 作为固定 `bmm2Write2Ub=true,splitD=false` 的对照。
- 触发条件：MX dispatcher 根据完整模板的 `dVTemplateType`、S1/S2 tile、mask/PSE/RoPE 条件计算 C2 落点；`dVTemplateType>Aligned256` 时编译期启用 split-D。
- Golden 来源：[`fia_template_dispatcher.h:99-106`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/common/op_kernel/arch35/fia_template_dispatcher.h#L99-L106)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr TPosition bmm2OutPos =
    GetC2Position(dVTemplateType,
                  UbOutCondition<INPUT_T>(false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                                          (uint32_t)s1TemplateType == 64),
                  ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
constexpr bool useDn = (quantMode == FULLQUANT_MODE_QKV_MXFP8_PREFILL);
constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
```

#### 机制与设计原因

把 C2 落点和 D 方向拆分固化为模板常量，可删除 Device 热路径分支，并让 workspace 只在需要跨阶段/跨 D slice 时出现。

#### 迁移边界

- 适用：C2 落点和 D 方向拆分可由模板桶静态决定的 MXFP8 attention。
- 前提：Host 的 Dv 对齐桶、workspace 公式与 Device `DTemplateType`/`TPosition` 选择一致。
- 失效：Dv 桶、tile 或 UB 条件变化时必须同步修改 Host workspace 与已编译 key；R-001 不启用此 split-D。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要根据 runtime 原始 Dv 单独改变 workspace，也不要把 R-002 的 splitD 逻辑套到固定 `splitD=false` 的 R-001。

## 模板与实现对比

| 对比维度 | R-001 重构 noquant | R-002 重构 MX | R-003 empty | R-004 legacy antiquant | R-005 legacy PFA | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | registry p28 严格 guard | registry p210 `fullQuantMode` | fallback early return | fallback quant mapper | fallback remainder | 先重构模板、后 legacy；空域在 legacy 最先结束 |
| 执行族/节点图 | 单 MIX_CV：GQA Cube/Vec/FD | 单 MIX_CV：MX Cube/Vec/FD | AIV zero-op | 单 MIX_CV antiquant IFA | 单 MIX_CV no/fullquant/MLA PFA | key 与 entry 编译期分支冻结执行 body |
| M/N/K、tile 与 scheduler | metadata 的 BN2/M/S2 区间 | metadata + MX Dv 桶 | 每核连续输出区间 | legacy IFA split/POD | legacy PFA S1/S2 split/POD | shape、量化与特征 guard 决定 specialization |
| workspace/L1/L0/UB | normal+FD | normal+FD+splitD | system only | L1/L0C 模拟区+100MiB+FD/PA | normal+FD/PA | Host slice 顺序必须匹配 Device consumer |
| AIC/AIV 或 kernel 交接 | SSBUF/cross-core + metadata | SSBUF/cross-core + metadata | 无 AIC 交接 | antiquant Vec→Cube 与 Cube→Vec | QK→softmax→PV→update | MIX_CV 单 launch 内按 flag/event 防覆盖 |
| 同步、精度与 tail | FP32 online/FD，metadata tail | MX scale 后 FP32，split-D/FD tail | 目标 dtype 零填充 | 反量化后 FP32 softmax/累加 | FP32 online 状态后 cast/postquant | 非空 `SyncAll` 与 batch schedule 配对 |
| fallback/失败边界 | registry 失败转 legacy | registry 失败转 legacy | 无 fallback | 无 Device fallback | 无 Device fallback | Host false 可继续，tiling/key/body 未注册则失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 最新公开入口 | V4 | V4 | V5 |
| Host arch | arch22 | arch22 | `DAV_3510`/arch35 |
| 新 registry 模板 | 非量化 MLA/general 等 arch22 模板 | 同左 | p28 noquant GQA、p210 MX fullquant，剩余 legacy |
| POD/分核 | arch22 POD，最多 26 路数组 | 同左 | `FAMetadata[36][8]` + `FDMetadata[72][8]` |
| Kernel/同步 | arch22 flags/events | 同左 | arch35 SSBUF cross-core；CMake `ENABLE_CV_COMM_VIA_SSBUF=true` |
| V5 新语义 | 不支持 V5 | 不支持 V5 | `qStartIdx`,`kvStartIdx`,`pseType` |

## 限制与失败边界

- 本卡仅覆盖仓库 `ops-transformer` 固定提交 `be33d95bb755cedec3108f70805c1874d1792540` 中 950/Ascend 950 的公开 V5 链路；不把 V4 或 arch38 事实混入 950 route。
- generated `aclnnInner_fused_infer_attention_score.h` 与生成实现不在固定源码树；该边界标为 generated bridge，并由下游 OpDef/Host 接口重新闭合，不能补写其内部节点细节。
- 非连续 K/V、K/V antiquant scale 或 `keyRope` 在旧 opbase 缺少 `NnopbaseSupportTensorV2` 时失败；这不是自动 Contiguous 保证。
- registry 返回失败既可能是 `IsCapable=false`，也可能是模板内部 tiling 失败；源码统一尝试 legacy。legacy 再失败时没有第三套 Host fallback。
- key 只表示已注册维度组合；dtype/layout/feature checker 仍是前置必要条件。不能仅凭 `quantMode` 数字声称任意输入可执行。
- POD 数字 offset、padding 与 legacy POD 全布局在固定仓中缺失时保持“缺失”；不得按宿主编译器猜测。
- 本次任务未执行编译、测试、validator、review、真机数值或性能实验；不把源码结构写成已测性能结论。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开文档 | `attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md` | `aclnnFusedInferAttentionScoreV5` | 产品支持、V5 新增参数、函数原型与输入约束 |
| API 声明 | `attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v5.h` | V5 GetWorkspaceSize/execute | 公开 ABI |
| API 实现 | `attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v5.cpp` | `CheckTensorContiguous`,V5 GetWorkspaceSize | INT4前处理、连续性弱符号、数组/LSE占位、generated bridge |
| API 共用前处理 | `attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_inner.cpp` | `TensorPreProcess`,`PrefixTensorPreProcess` | INT32 打包载体到 INT4 view |
| 顶层 Host 路由 | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp` | `DoOpTilingFusedInferAttentionScore` | `DAV_3510`→arch35 V4 tiling 与 data-class 接线 |
| arch35 selector | `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v4.cpp` | `TilingFusedInferAttentionScoreV4` | parse/check/registry/fallback 顺序 |
| arch35 fallback | `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp` | `DoOpTiling`,`GenTilingKey`,`GetWorkspace` | legacy split/POD/key/blockDim/workspace/schedule |
| noquant tiling | `attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp` | `FiaTilingNonQuantArch35` | p28 guards、metadata、workspace、reconstruct bit |
| MX tiling | `attention/common/op_host/arch35/fia_tiling_fullquant_mx.cpp` | `FiaTilingFullQuantMxArch35` | p210、mode19/20、metadata、workspace |
| 公共 POD | `attention/common/op_kernel/fia_tiling_data_public.h` | `FiaBaseParams`,`FiaMetaData` | 字段顺序、类型、metadata indices |
| noquant POD | `attention/common/op_kernel/arch35/fia_tiling_data_noquant_gqa.h` | `FusedInferAttentionScoreTilingData` | noquant base + metadata |
| MX POD | `attention/common/op_kernel/arch35/fia_tiling_data_fullquant_mx.h` | `FusedInferAttentionScoreFullQuantTilingData` | fullquant base + metadata |
| Kernel 顶层 | `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_apt.cpp` | `fused_infer_attention_score` | quantMode 15 阈值与统一参数桥 |
| key 定义 | `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h` | `QuantMode`,`IsReconstructTemp` | mode 数值和 key 位域 |
| PFA 路由/body | `attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h` | `prompt_flash_attention_FIAS_regbase` | empty、reconstruct 与 legacy PFA 分派 |
| IFA 路由/body | `attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h` | `incre_flash_attention_FIAS_regbase` | empty 与 antiquant dtype 模板 |
| 新 body dispatcher | `attention/common/op_kernel/arch35/fia_template_dispatcher.h` | `run_fia_noquant_gqa_kernel`,`run_fia_fullquant_mx_kernel` | block type、POD offset、Init/Process |
| noquant kernel | `attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h` | `FlashAttentionNoQuantGqaKernel` | metadata、cross-core、FA/FD 数据流 |
| MX kernel | `attention/common/op_kernel/arch35/fia_kernel_fullquant_mx.h` | `FlashAttentionFullQuantMxKernel` | MX 数据流、cross-core、FA/FD |
| legacy antiquant body | `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h` | `FlashAttentionScoreAntiquantKernel` | R-004 反量化、FA/FD 调度与 `SyncAll` |
| legacy PFA body | `attention/common/op_kernel/arch35/flash_attention_score_kernel_infer.h` | `FlashAttentionScoreKernel` | R-005 普通全量化 Cube/Vector 管线 |
| legacy PFA body | `attention/common/op_kernel/arch35/flash_attention_noquant_kernel_infer.h` | `FlashAttentionNoQuantKernel` | R-005 普通非量化 Cube/Vector 管线 |
| legacy PFA body | `attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_mla_fullquant.h` | MLA fullquant kernel | R-005 MLA 全量化 specialization |
| legacy PFA body | `attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_gqa_fullquant.h` | GQA fullquant kernel | R-005 GQA 全量化 specialization |
| legacy PFA body | `attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_mx_fullquant.h` | MX fullquant kernel | R-005 legacy MX specialization |
| legacy PFA body | `attention/common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h` | noquant MLA kernel | R-005 非量化 MLA specialization |
| 构建接线 | `attention/fused_infer_attention_score/op_host/CMakeLists.txt` | `add_op_to_compiled_list`,arch35 source list | common/PFA/IFA 依赖与 Ascend950 编译选项 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| API 前处理 | INT32→INT4 view、LSE占位、连续性弱符号 | 末轴×8、临时对象释放、旧 opbase fail-close | `TensorPreProcess`,`CheckTensorContiguous` |
| selector | registry p28/p210 后 legacy | priority、`IsCapable` 与 fallback 顺序 | `TilingFusedInferAttentionScoreV4` |
| ABI/POD | common base + route-specific POD + metadata | 字段顺序/类型、`offsetof`、36/72×8 | `FiaMetaData`,two tiling data classes |
| key | quant 0/2/6 与 17/18/19/20/30/31、reconstruct bit | 15 阈值和 POD/body 配对 | `UpdateTilingKeyInfo`,`fused_infer_attention_score` |
| workspace | system、per-core preload、FD、antiquant 模拟区、PA flags | bytes/元素单位、slice 顺序、owner/lifetime | `CalcWorkspaceSize`,`SetWorkspaceNormal`,`SetWorkspaceAntiQuant` |
| 分核 | AIC FA metadata、AIV FD metadata | core index 和 start/end 列语义 | `SetSplitOutput`,`FiaMetaData` |
| 同步 | cross-core buffer、pipe event、`SyncAll`/batch mode | producer-consumer 边与 slot 复用前等待 | noquant/MX kernel,`DoOpTiling` |
| 精度与尾块 | FP32 score/softmax/update/FD 后 cast | online rescale 顺序、有效前缀提交 | `FlashAttentionNoQuantGqaKernel`,`FlashAttentionFullQuantMxKernel` |
| 构建 | arch35 sources + common/PFA/IFA dependencies | Ascend950 SSBUF 与 auto-sync 选项 | operator `CMakeLists.txt` |
| 失败边界 | checker/registry/fallback/key/POD/workspace fail-close | 不执行未注册 key，不猜 generated bridge | V5 API,arch35 selector/fallback |