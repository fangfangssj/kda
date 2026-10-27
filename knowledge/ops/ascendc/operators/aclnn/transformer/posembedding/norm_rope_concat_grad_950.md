---
type: Operator
title: "NormRopeConcatGrad 950 ACLNN 算子知识"
description: "说明 aclnnNormRopeConcatBackwardGetWorkspaceSize/aclnnNormRopeConcatBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 Device 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/docs/aclnnNormRopeConcatBackward.md
    role: primary
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/op_api/aclnn_norm_rope_concat_grad.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/op_api/aclnn_norm_rope_concat_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/op_api/norm_rope_concat_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_def.cpp
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/config/ascend950/norm_rope_concat_grad_binary.json
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/cmake/level2_ops_list
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/scripts/ci/ascend950/ops_transformer_operator_list.yaml
    role: constraint
created_at: "2026-08-24T16:30:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# NormRopeConcatGrad 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnNormRopeConcatBackwardGetWorkspaceSize` / `aclnnNormRopeConcatBackward`，固定仓唯一公开 Backward 版本 |
| 计算语义 | 从 concat 后的 BHSD 上游梯度拆回 BSHD；Q/K 可做 RoPE 与 LayerNorm 反向，V 只切片；affine 时再归约四组 dγ/dβ |
| 输入输出 | 21 tensor+4 int64 属性→14 tensor；主 dtype FP16/BF16/FP32，mean/rstd FP32，ND |
| 执行类型 | `AIV_VECTOR` 单 Kernel；固定范围只出现 VECCALC/VECIN/VECOUT、Vector API 与数据搬运/同步原语 |
| Kernel 路线 | 54 个 key 按 Norm 资源族与 Rope tail 策略拆为 R-001～R-009 |
| 核心资源 | GM；七个顺序复用 TBufPool；FP32 TBuf；TQue；affine 两级 FP32 workspace；无显式 RegTensor |
| 来源闭包 | `source_closed`：ACLNN→L0→OpDef/InferShape→Tiling/POD→54-key entry→Vector body/helper→950 config/CMake 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `T=gradQueryOutput.dtype`，`D=headDim`。`concatOrder=0/1` 分别表示 `[主,encoder]`/`[encoder,主]`。上游为 `[B,H,S_total,D]`，输出为 `[B,S,H,D]`。RoPE NONE 时 `g=dy`；INTERLEAVE/HALF 时在 FP32 中执行 `g=dy*cos+rotate(dy*sin)`，反向符号由 mask/Muls 编码。LayerNorm 使用 `xhat=(x-mean)*rstd`、`scale=1/D`，`dx=rstd*(dxhat-mean_D(dxhat)-xhat*mean_D(dxhat*xhat))`；affine 时 `dxhat=dy*weight`、`dβ=Σdy`、`dγ=Σ(dy*xhat)`。V/eV 只做 BHSD→BSHD 切片。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradQueryOutput/gradKeyOutput/gradValueOutput` | `[B,H,S主+Senc,D]`，ND | T∈FP16/BF16/FP32 | 三者必选同 dtype；K/V 两组序列长度相等 |
| `query/key` | `[B,S,H,D]`，ND | T | 必选；B/H/D 相同，D 为偶数且≤1024 |
| `encoderQuery/encoderKey` | `[B,Senc,H,D]`，ND | T | 可选；空指针令对应 Senc=0 |
| `normQueryWeight/normKeyWeight` | `[D]`，ND | T | `normType=2` 时 γ；属性配对非空主要由公开契约保证 |
| `normQueryMean/Rstd` | `[B,Sq,H,1]`，ND | FP32 | `normType=1/2` 保存统计量 |
| `normKeyMean/Rstd` | `[B,Sk,H,1]`，ND | FP32 | 同上 |
| `normAddedQueryWeight/normAddedKeyWeight` | `[D]`，ND | T | `normAddedType=2` 时 γ |
| `normAddedQueryMean/Rstd` | `[B,Seq,H,1]`，ND | FP32 | encoderQuery 统计量 |
| `normAddedKeyMean/Rstd` | `[B,Sek,H,1]`，ND | FP32 | encoderKey 统计量 |
| `ropeSin/ropeCos` | `[Sr,D]`，ND | T | Rope启用时配对；公开契约 `1≤Sr≤min(totalQ,totalK)` |
| `normType/normAddedType` | Host标量 | int64 | 0 NONE、1 LN、2 affine LN；默认0 |
| `ropeType` | Host标量 | int64 | 0 NONE、1 INTERLEAVE、2 HALF；默认0 |
| `concatOrder` | Host标量 | int64 | 0/1；默认0 |
| `gradQuery/gradKey/gradValue` | `[B,Sq/Sk/Sk,H,D]`，ND | T | 三个必选输出 |
| `gradEncoderQuery/Key/Value` | `[B,Seq/Sek/Sek,H,D]`，ND | T | 可选输出 |
| `gradNormQueryWeight/Bias` | 各 `[D]`，ND | T | `normType=2` 的 query dγ/dβ |
| `gradNormKeyWeight/Bias` | 各 `[D]`，ND | T | `normType=2` 的 key dγ/dβ |
| `gradNormAddedQueryWeight/Bias` | 各 `[D]`，ND | T | `normAddedType=2` 的 encoderQuery dγ/dβ |
| `gradNormAddedKeyWeight/Bias` | 各 `[D]`，ND | T | `normAddedType=2` 的 encoderKey dγ/dβ |

### 入口与版本

公开接口只存在 Backward，无 V2/V3。第一段创建 executor，检查必选指针及 dtype，L0 分配14个 ND 临时输出、推形并加入 launcher，再对非空 L0 输出调用 `ViewCopy(l0Tensor, userTensor, executor)`；第二段在用户 stream 上 `CommonOpExecutorRun`。该 Grad 不能并入前向 `norm_rope_concat` 版本族。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 强制三份上游梯度、query/key、gradQuery/gradKey/gradValue 非空；所有存在的普通 tensor 与主 dtype 相同，八个 mean/rstd 必须 FP32。OpDef 为21输入启用 `AutoContiguous`。InferShape 检查 BHSD 上游、BSHD 前向及 B/H/D；当三份上游的 S 与主/encoder 序列和不等时只执行 `OP_LOGE`，不返回失败。Tiling 的 `total*Seq_` 又由 query/key 与 encoder shape 自行相加，相关等式检查是恒真的；它仅另查上游 S 不小于 `ropeSeq_`，因此不能关闭该不匹配。Tiling 还检查 D 为偶数且≤1024、属性范围以及 K/V 主/encoder 序列相等。

### 参数、Workspace 与生命周期

平台提供 AIV核数、UB和system workspace。仅 affine 时追加 `8*usedCore*alignedD*4` B 主归约区和 Host 保守上界 `40*usedCore*max(avgHeads,2)*2*alignedD*4` B 级联区；Device 实际第二段为 `affineBlockNums*usedCore*avgHeadsLow*2*alignedD*4` B。workspace、连续化tensor、L0临时输出与ViewCopy节点均由 executor 持有到第二段完成。

### 启动图与失败传播

`GetWorkspaceSize→CheckParam→l0op::NormRopeConcatGrad→INFER_SHAPE→ADD_TO_LAUNCHER_LIST_AICORE→TilingNormRopeConcatGrad→norm_rope_concat_grad<<<usedCore>>>→ViewCopy`。任一 executor/L0/必选输出/ViewCopy/Tiling/运行错误立即返回；key≥100000设置schedule mode1，较小key保持默认。

## Route 与参数设计

### 分核与资源计算

`usedCore=min(max(Sq,Sk,Seq,Sek),aivNum)`；每分支令 `each=S/usedCore, tail=S%usedCore`，前tail核取each+1行。`alignedD=align(D,32/sizeof(T))`。Tiling 先算 `capacityHeads=floor((UB-ropeCoef*ropeFixed-normCoef*normFixed-1024)/oneHead)`，截到4096后要求其≥2，再执行 `avgHeads=min(capacityHeads,H)`；所以 H=1 时最终 `avgHeads=1` 合法，Device 以 `A=max(avgHeads,2)` 分配 UB。`splitHeadNum=ceil(H/avgHeads)`，最后块 `tailHeads=H-(split-1)*avgHeads`。主FP32 TBuf=`3*A*alignedD*4+256` B。源码 `DOUBLE_BUFFER=1`，所有相应 queue 实际 depth=1。七个 pool 链到queryPool并在各分支后Reset，是顺序别名而非并存七份。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 纯拆分 | 0/10，`<0,0,0,C>` | N=A=R=0，C=0/1 | 属性组合不满足时继续按表中顺序匹配 |
| 2 | R-002 Interleave拆分 | 100/110，`<0,0,1,C>` | N=A=0,R=1 | 属性组合不满足时继续按表中顺序匹配 |
| 3 | R-003 Half拆分 | 200/210，`<0,0,2,C>` | N=A=0,R=2 | 属性组合不满足时继续按表中顺序匹配 |
| 4 | R-004 非affine Norm | 100000/100010；100000000/100000010；100100000/100100010 | `(N,A)=(0,1)/(1,0)/(1,1),R=0` | 属性组合不满足时继续按表中顺序匹配 |
| 5 | R-005 非affine+Interleave | 100100/100110；100000100/100000110；100100100/100100110 | `(N,A)=(0,1)/(1,0)/(1,1),R=1` | 属性组合不满足时继续按表中顺序匹配 |
| 6 | R-006 非affine+Half | 100200/100210；100000200/100000210；100100200/100100210 | `(N,A)=(0,1)/(1,0)/(1,1),R=2` | 属性组合不满足时继续按表中顺序匹配 |
| 7 | R-007 affine Norm | 200000/200010；100200000/100200010；200000000/200000010；200100000/200100010；200200000/200200010 | N/A至少一个2，R=0 | 属性组合不满足时继续按表中顺序匹配 |
| 8 | R-008 affine+Interleave | 200100/200110；100200100/100200110；200000100/200000110；200100100/200100110；200200100/200200110 | N/A至少一个2，R=1 | 属性组合不满足时继续按表中顺序匹配 |
| 9 | R-009 affine+Half | 200200/200210；100200200/100200210；200000200/200000210；200100200/200100210；200200200/200200210 | N/A至少一个2，R=2 | 非法属性Tiling失败 |

```text
校验 N,A,R∈[0,2]、C∈[0,1]
key=N*100000000+A*100000+R*100+C*10
含affine按R分到R-007/8/9；仅LN按R分到R-004/5/6；无Norm按R分到R-001/2/3
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 实际模板/类 | body/helper | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 0/10 | `N=A=R=0`，`C=0/1`，T∈{FP16,BF16,FP32}，平台接线可达 | `NormRopeConcatGrad<NONE,NONE,NONE,C>` | `ProcessQK/ProcessV` | FP32 主buf+输入/输出 queue，无额外 workspace | core/head tail；非法属性失败 |
| R-002 | 100/110 | 上述且 `R=1` | `<NONE,NONE,INTERLEAVE,C>` | `RopeOperationBackward<INTERLEAVE>` | 增加 sin/cos queue、FP32 sin/cos 与 u32 mask | `curSeq>=ropeActualSeq` 跳过 Rope |
| R-003 | 200/210 | 上述且 `R=2` | `<NONE,NONE,HALF,C>` | `RopeOperationBackward<HALF>` | 与 R-002 同容量，mask/符号规则不同 | D 必须偶数；无 fallback |
| R-004 | 6 key | `N,A∈{0,1}` 且至少一个1，`R=0` | `NormRopeConcatGrad<N,A,NONE,C>` | `NormOperationBackward::{CopyIn,Compute,CopyOut}` | 新增 x/stats queue 与 FP32 Norm 分片 | 未开 Norm 的分支走 NONE body |
| R-005 | 6 key | R-004 的 N/A 集，`R=1` | `<N,A,INTERLEAVE,C>` | Rope Interleave→Norm | Rope 与 Norm 共用主buf，pool 包含两类容量 | rope/core/head tail |
| R-006 | 6 key | R-004 的 N/A 集，`R=2` | `<N,A,HALF,C>` | Rope Half→Norm | 同 R-005，Half mask 不同 | D/2 + rope/core/head tail |
| R-007 | 10 key | `N=2` 或 `A=2`，`R=0` | `NormRopeConcatGrad<N,A,NONE,C>` | `AffineAccSum/PostProcess` | 增加 weight/dγ/dβ UB 片与两区 FP32 GM workspace，两次 `SyncAll` | 40 级 Host 保守上界；无显式超限检查 |
| R-008 | 10 key | R-007 的 N/A 集，`R=1` | `<N,A,INTERLEAVE,C>` | Rope Interleave→affine Norm→归约 | R-007+Rope 分片/HardEvent | rope/core/head/级联 tail |
| R-009 | 10 key | R-007 的 N/A 集，`R=2` | `<N,A,HALF,C>` | Rope Half→affine Norm→归约 | R-007+Rope 分片/HardEvent | D/2+rope/core/head/级联 tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| headDim 偶数且≤1024 | `(headDim_ % 2 == 0) && headDim_ <= 1024` | Half 使用 `D/2`，UB 公式也以 D 为上界 | 全部 | `GRAPH_FAILED` | `CheckInput` |
| UB 容量候选至少2 heads | `capacityHeads=(UB-fixed-1024)/oneHead` 截到4096后必须≥2；随后 `avgHeads=min(capacityHeads,H)` | 主buf/pool 按 `A=max(avgHeads,2)` 分配；H=1 时最终 avg 可为1 | 全部 | 容量候选<2时 `GRAPH_FAILED`；H=1不失败 | `ComputeUBTilingStrategy` |
| 32B dtype 对齐 | `alignedD=CeilAlign(D,32/sizeof(T))` | `DataCopyPad`、Queue 和 FP32 Vector 分片要求对齐容量 | 全部 | 对齐保留，仅 CopyOut 写 D 元素 | `ComputeUBTilingStrategy/Init` |
| Rope 有效序列 | `ropeActualSeq=ropeSin.dim0`，Device `curSeq<actualSeq` | 避免读取 sin/cos 越界 | R-002/3/5/6/8/9 | 当前行跳过 Rope，继续 Norm/copy | `RopeOperation::{PreProcess,Compute}` |
| 分核非空 | `usedCore=min(max(Sq,Sk,Seq,Sek),aivNum)` | `GetSeqRange` 会除以 `usedCore` | 全部 | 固定仓没有空域 ACLNN 快返回 | `ComputeCoreTilingStrategy/GetSeqRange` |
| 上游 S 等于主/encoder 序列和 | InferShape 只 `OP_LOGE`；Tiling 未比较上游 S 与自行计算的 `total*Seq_`，仅检查上游 S≥`ropeSeq_` | Kernel 用自行计算的 `total*Seq_` 作 BHSD stride | 全部 | 固定实现无失败终点；调用者必须满足公开 shape 契约 | `InferShape4NormRopeConcatGrad/CheckInput/GetShapeInfo` |
| affine 级联高度 | Host 预留 `MAX_AFFINE_BLOCK_NUM=40` 层 | Device `affineArray_[40]` 保存每层占用 | R-007/8/9 | 无显式 `affineBlockNums<=40` 检查 | `ComputeWorkSpace/NormOperationBackward::AffineAccSum` |
| Router key 集合 | Host 属性校验只生成 `3*3*3*2=54` 个 key | entry 只有54个 `Impl`，末尾没有 unknown-key 错误分支 | 全部 | 非法/损坏 key 会无输出直接返回 | `ComputeTilingKey/norm_rope_concat_grad` |
| schedule mode | `tilingKey>=100000` | 所有含 Norm 的 key 使用 batch schedule mode 1 | R-004～R-009 | key<100000 保持默认 mode | `DoTiling` |

### POD/ABI

宏实现属于固定仓外依赖，确切 `sizeof/alignof/offsetof` 固定仓不可见；按声明顺序逐字段如下。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `NormRopeConcatGradTilingData` | `tilingKey` | uint64/64 | 固定仓不可见；序号0 | selector | `ComputeTilingKey/FillTilingData` | entry router | 54 key |
| 同上 | `batch` | uint64/64 | 固定仓不可见；序号1 | batch | query dim0 | `Init→batch_` | 窄化uint32 |
| 同上 | `querySeq` | uint64/64 | 固定仓不可见；序号2 | 行 | query dim1 | `ProcessQK` | 窄化uint32 |
| 同上 | `keySeq` | uint64/64 | 固定仓不可见；序号3 | 行 | key dim1 | `ProcessKey/V` | 同上 |
| 同上 | `valueSeq` | uint64/64 | 固定仓不可见；序号4 | 行 | `=keySeq` | `ProcessValue` | 必须相等 |
| 同上 | `encoderQuerySeq` | uint64/64 | 固定仓不可见；序号5 | 行 | optional dim1/0 | `ProcessEncoderQK` | 可0 |
| 同上 | `encoderKeySeq` | uint64/64 | 固定仓不可见；序号6 | 行 | optional dim1/0 | `ProcessEncoderQK` | 可0 |
| 同上 | `encoderValueSeq` | uint64/64 | 固定仓不可见；序号7 | 行 | `=encoderKeySeq` | `ProcessValue` | 必须相等 |
| 同上 | `totalQuerySeq` | uint64/64 | 固定仓不可见；序号8 | 行 | Sq+Seq | BHSD stride | 匹配上游S |
| 同上 | `totalKeySeq` | uint64/64 | 固定仓不可见；序号9 | 行 | Sk+Sek | BHSD stride | 同上 |
| 同上 | `totalValueSeq` | uint64/64 | 固定仓不可见；序号10 | 行 | Sv+Sev | BHSD stride | 同上 |
| 同上 | `ropeActualSeq` | uint64/64 | 固定仓不可见；序号11 | 行 | ropeSin dim0/0 | `isActive/PreProcess` | curSeq越界跳过Rope |
| 同上 | `splitHeadNum` | uint64/64 | 固定仓不可见；序号12 | tile数 | ceil(H/avg) | head循环 | ≥1 |
| 同上 | `avgHeads` | uint64/64 | 固定仓不可见；序号13 | head/tile | 容量候选截到4096后再 `min(...,H)` | queue extent；Device另取`max(avgHeads,2)`分配 | 最终≥1；H=1时可为1 |
| 同上 | `tailHeads` | uint64/64 | 固定仓不可见；序号14 | head | H-(split-1)*avg | 最后head块 | 1..avg |
| 同上 | `normDim` | uint64/64 | 固定仓不可见；序号15 | 元素 | `=D` | Norm blockLen/Sum | 偶数≤1024 |
| 同上 | `ropeDim` | uint64/64 | 固定仓不可见；序号16 | 元素 | ropeSin dim1/0 | Rope mask/copy | R>0时=D |
| 同上 | `headNum` | uint64/64 | 固定仓不可见；序号17 | head | query dim2 | 地址公式 | 窄化uint32 |
| 同上 | `headDim` | uint64/64 | 固定仓不可见；序号18 | 元素 | query dim3 | 地址stride | body主用normDim/ropeDim |
| 同上 | `usedCore` | uint64/64 | 固定仓不可见；序号19 | AIV核 | min(maxSeq,aivNum) | 分核/workspace | 非空应>0 |
| 同上 | `eps` | float/32 | 固定仓不可见；序号20 | 标量 | 默认1e-5 | 仅存`eps_` | Compute未读取 |
| 同上 | `scale` | float/32 | 固定仓不可见；序号21 | 1/D | `CheckInput` | 两次Muls | D非0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 纯拆分 | 0/10 | `tiling.cpp:ComputeTilingKey` | POD N=A=R=0 | `kernel.cpp:Impl` | `ProcessQK/ProcessV` | `op_host/CMakeLists.txt:add_modules_sources` |
| R-002 Interleave拆分 | 100/110 | 同上 | R=1 | 同上 | `RopeOperation<INTERLEAVE>` | 同上 |
| R-003 Half拆分 | 200/210 | 同上 | R=2 | 同上 | `RopeOperation<HALF>` | 同上 |
| R-004 非affine Norm | 6 key | 同上 | N/A∈{0,1} | 同上 | `NormOperationBackward::Compute` | 同上 |
| R-005 非affine+Interleave | 6 key | 同上 | 上项+R=1 | 同上 | Norm+Interleave helper | 同上 |
| R-006 非affine+Half | 6 key | 同上 | 上项+R=2 | 同上 | Norm+Half helper | 同上 |
| R-007 affine Norm | 10 key | `ComputeWorkSpace/Key` | N/A含2 | 同上 | `AffineAccSum/PostProcess` | 同上 |
| R-008 affine+Interleave | 10 key | 同上 | 含2,R=1 | 同上 | affine+Interleave | 同上 |
| R-009 affine+Half | 10 key | 同上 | 含2,R=2 | 同上 | affine+Half | 同上 |

## Kernel 实现

### 公共执行流程

entry 显式54次 `Impl` 实例化 `NormRopeConcatGrad<N,A,R,C>`。固定顺序为 `Prepare→ProcessQuery→ProcessKey→ProcessValue→PostProcess`。上游 offset=`b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`，输出 offset=`(b*S+s)*H*D+h*D`。每个Q/K/eQ/eK/V/eV分支后Reset其pool；V固定用NONE Rope+NONE Norm。

### Route R-001：纯拆分

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey` | N=A=R=0,C=0/1 | 0/10 |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `Impl(0/10)` | TILING_KEY_IS | NONE模板 |

#### Device 数据流

##### 函数级调用链

```text
norm_rope_concat_grad -> NormRopeConcatGrad::Process -> ProcessQK/ProcessV
  -> RopeOperationBackward<NONE>::CopyIn/Compute -> NormOperationBackward<NONE>::Compute/CopyOut
```

##### 源码等价伪码

```text
for branch,s in GetSeqRange(blockIdx),b,headTile:
  outOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D
  inOffset=(b*S+s)*H*D+h*D
  DataCopyPad(BHSD,outGradQue,blockCount=heads,srcStride=(totalS-1)*D*sizeof(T))
  Cast/Adds到FP32主buf，再CAST_ROUND/Adds到inGradQue，写BSHD前D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM上游/输出 | 上述BHSD/BSHD公式 | 逻辑tensor | T | caller/queue | queue/caller | launch | 只写D |
| 主FP32 `buf_` | TPipe offset | `3*max(avg,2)*alignedD*4+256` | FP32 UB | Cast | Norm NONE | tile | branch复用 |
| `outGradQue_` | pool offset不可见 | `avg*alignedD*sizeof(T)` | T VECIN | GM | Cast | tile | depth1 |
| `inGradQue_` | pool offset固定仓不可见 | `avg*alignedD*sizeof(T)` | T VECOUT | round | GM | tile | depth1 |
| 七个TBufPool | queryPool后链接别名 | `poolSize=max(2*avg*alignedD*sizeof(T),24*alignedD*4)` | UB | Init | 六分支/post | branch | Reset复用 |
| workspace | N/A | 0 extra | N/A | N/A | N/A | N/A | 非affine |
| RegTensor | N/A | 0显式 | intrinsic | Vector API | Vector API | 指令 | 固定仓无对象 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked TBufPool · max(2·A·E·s, 96·E) B"]:1
```

`A=max(avgHeads,2)`、`E=alignedNormDim`、`s=sizeof(T)`。七个 pool 以 `queryPool_` 为同一物理起点链接，不乘七。

##### 字节账与复用顺序

```text
UB_total = (12·A·E + 256) + max(2·A·E·s, 96·E)
Host 以 ComputeUBTilingStrategy 的 oneHead/fixed 系数选取容量候选并要求候选≥2；最终 avgHeads 可因 H=1 截为1，A仍取2
固定仓没有再显式断言 UB_total<=platform UB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| linked pool | 当前 Q/K/eQ/eK/V/eV 分支 | 当前分支 queue | 下一分支 | `PIPE_ALL` 后 `bufPool.Reset()` | 分支按 `Process` 固定顺序串行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬运 | TQue EnQue/DeQue | MTE | Vector/GM | queue slot | core/head tail |
| Vector | `PipeBarrier<PIPE_V>` | Cast/Adds | queue提交 | 主buf | T<FP32 CAST_ROUND |
| pool | `PipeBarrier<PIPE_ALL>` | branch | Reset | pool | 空分支不分配 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeCoreTilingStrategy/ComputeUBTilingStrategy` | core/head分片 |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `Impl(0/10)` | NONE实例 |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | `ProcessQK/ProcessV/GetSeqRange` | 循环地址与copy |

### Route R-002：Interleave拆分

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey` | 0,0,1,C | 100/110 |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `Impl(100/110)` | 显式key | INTERLEAVE |

#### Device 数据流

##### 函数级调用链

```text
entry -> ProcessQK -> RopeOperation::Prepare/PreProcess
  -> RopeOperationBackward::CopyIn/Compute -> Rotate/Collect -> NormOperationBackward<NONE>::CopyOut
```

##### 源码等价伪码

```text
mask[2i]=(2i+1)*4; mask[2i+1]=2i*4
for s,b,headTile: outOffset=b*H*totalS*D+(outputSeq+s)*D+headTile*avg*totalS*D
  加载sin/cos并cast FP32；masked Muls(sin,-1)
  搬dy；for head: rotated=Gather(dy*sin,mask); dy=dy*cos+rotated；round写D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM dy/out | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`outOffset=(b*S+s)*H*D+h*D` | `dy=B*H*totalS*D`；`out=B*S*H*D` 元素 | T | caller | route | launch | BHSD→BSHD |
| GM sin/cos | `(outputSeq+s)*D` | Sr*D各 | T | caller | ropeQueue | seq | 末行右padding |
| `ropeQueue_` | pool offset | `2*alignedD*sizeof(T)` | T VECIN | GM | Cast | seq | depth1 |
| Rope sin TBuf片 | 0 | alignedD*4 | FP32 | Cast/Muls | Mul | branch | repeat/tail mask |
| Rope cos TBuf片 | alignedD*4 | alignedD*4 | FP32 | Cast | Mul | branch | 独立 |
| Rope mask TBuf片 | 2*alignedD*4 | alignedD*4 | u32 | Prepare | Gather | branch | 字节索引 |
| out/inGrad queues | pool offsets | 各avg*alignedD*sizeof(T) | T | GM/round | Vector/GM | tile | depth1 |
| workspace/RegTensor | N/A | 0 extra/0显式 | N/A | N/A | N/A | N/A | pool含ropeFixed |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(2·A·E·s+2·R·s+12·R, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(2·A·E·s + 2·R·s + 12·R, 96·E)
R=alignedRopeDim；Rope 区=ropeQueue(2·R·s)+sin/cos/mask(12·R)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| sin/cos/mask 与 branch pool | `Prepare/PreProcess` | `Rotate/Collect` | 下一 seq/分支 | queue `FreeTensor`、`PIPE_V`、分支末 `PIPE_ALL+Reset` | `isActive_` 为真才分配/读取 Rope |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| queue | EnQue/DeQue | GM | Cast | rope/dy slots | 32B padding |
| Rope | 多次`PIPE_V` | Cast/Muls/Mul/Gather | Collect | FP32三片 | 64-lane repeat+tail mask |
| 输出 | CAST_ROUND+TQue | Collect | GM | D前缀 | curSeq≥Sr跳过Rope |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeUBTilingStrategy/ComputeTilingKey` | Rope容量/key |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `Impl(100/110)` | route映射 |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h | `RopeOperation<INTERLEAVE,false>` | mask/repeat/tail |

### Route R-003：Half拆分

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey` | 0,0,2,C | 200/210 |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `Impl(200/210)` | 显式key | HALF |

#### Device 数据流

##### 函数级调用链

```text
entry -> ProcessQK -> RopeOperation<HALF,false>::Prepare/PreProcess
  -> RopeOperationBackward<HALF>::CopyIn/Compute -> Rotate/Collect -> CopyOut
```

##### 源码等价伪码

```text
half=D/2; mask[half+i]=i*4; mask[i]=(half+i)*4
for s,b,headTile: outOffset=b*H*totalS*D+(outputSeq+s)*D+headTile*avg*totalS*D
  加载sin/cos；Muls(sin[0:half],-1)
  tmp=dy*sin; rotated=Gather(tmp,mask); dy=dy*cos+rotated；round写D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM dy/out | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`outOffset=(b*S+s)*H*D+h*D` | `dy=B*H*totalS*D`；`out=B*S*H*D` 元素 | T | caller | route | launch | concat offset |
| GM sin/cos | `(outputSeq+s)*D` | Sr*D各 | T | caller | ropeQueue | seq | padding |
| ropeQueue | pool | 2*alignedD*sizeof(T) | T | GM | Cast | seq | depth1 |
| sin TBuf片 | 0 | alignedD*4 | FP32 | Cast/Muls | Mul | branch | 前half取负 |
| cos TBuf片 | alignedD*4 | alignedD*4 | FP32 | Cast | Mul | branch | 独立 |
| mask TBuf片 | 2*alignedD*4 | alignedD*4 | u32 | Prepare | Gather | branch | half交换 |
| out/inGrad queues | pool | 各avg*alignedD*sizeof(T) | T | GM/round | Vector/GM | tile | depth1 |
| workspace/RegTensor | N/A | 0/0显式 | N/A | N/A | N/A | N/A | 非affine |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(2·A·E·s+2·R·s+12·R, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(2·A·E·s + 2·R·s + 12·R, 96·E)
Half 与 Interleave 的容量相同，只有 mask 与负号规则不同
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Rope pool | `Prepare/PreProcess` | `Rotate/Collect` | 下一 seq/分支 | `PIPE_V`、`FreeTensor`、`PIPE_ALL+Reset` | D 偶数，尾部 padding 不写回 GM |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Rope搬运 | TQue | GM | Cast | ropeQueue | alignedD padding |
| HALF | `PIPE_V` | Muls/Mul/Gather | Collect | FP32 | D偶数，half=D/2 |
| 输出 | TQue | round | GM | D前缀 | core/head/rope tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `CheckInput/ComputeTilingKey` | D偶数/key |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `Impl(200/210)` | HALF实例 |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h | `Prepare/PreProcess/Rotate/Collect` | half符号和mask |

### Route R-004：非affine Norm

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey` | N/A∈{0,1}至少一个1,R=0 | 6key |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | 对应`Impl` | 编译期N/A | LN/NONE |

#### Device 数据流

##### 函数级调用链

```text
entry -> Prepare/SyncAll -> ProcessQK/ProcessEncoderQK
  -> NormOperationBackward::CopyIn/Compute -> Sum/BroadCast/MulAddDst -> CopyOut
```

##### 源码等价伪码

```text
for 启用LN的branch,s,b,headTile: dyOffset=b*H*totalS*D+(outputSeq+s)*D+headTile*avg*totalS*D；inOffset=(b*S+s)*H*D+headTile*avg*D；搬dy,x,mean,rstd到UB并cast FP32
  xhat=(x-mean)*rstd; du=-Sum_D(dy)/D; ds=-Sum_D(dy*xhat)/D
  dx=rstd*(dy+du+xhat*ds); round写BSHD D前缀
未启用LN侧与V走NONE copy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM dy/x/dx | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`x/dxOffset=(b*S+s)*H*D+h*D` | `dy=B*H*totalS*D`；`x/dx=B*S*H*D` 元素 | T | caller | Norm | launch | D有效 |
| GM mean/rstd | `(b*S+s)*H+h` | B*S*H各 | FP32 | forward | normQue | tile | blockLen=heads*4 |
| out/inGradQue | pool | 各avg*alignedD*sizeof(T) | T | GM/round | Compute/GM | tile | depth1 |
| `inQue_` | pool | avg*alignedD*sizeof(T) | T | xGM | Cast | tile | depth1 |
| `normQue_` | pool | 2*alignedNormNum*4 | FP32 | mean/rstd | Adds | tile | depth1 |
| `inX_` | Norm TBuf offset0 | avg*alignedD*4 | FP32 | Cast | xhat | branch | 原地 |
| `share_` | offset avg*alignedD | avg*alignedD*4 | FP32 | Broadcast | Add/MulAdd | tile | 临时 |
| `mean_/rstd_` | offset2*avg*alignedD与后加alignedNormNum | 各alignedNormNum*4 | FP32 | normQue | Sum/Broadcast | tile | 分片 |
| weight/workspace/RegTensor | N/A | 0 | N/A | N/A | N/A | N/A | 非affine |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(3·A·E·s+8·A·E+16·N, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(3·A·E·s + 8·A·E + 16·N, 96·E)
N=alignedNormNum；queue depth 常量虽名 DOUBLE_BUFFER，值为1
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inX_/share_/mean_/rstd_` | Cast/Broadcast/stats copy | LN 连续 Vector 管线 | 下一 head tile | 每步 `PIPE_V`，queue `FreeTensor` | Sum 只以 D 为有效列数 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 前置 | `SyncAll` | kernel | branch | 全核阶段 | 非affine仍执行 |
| queues | EnQue/DeQue | GM | Norm | input/stats | T→FP32 CAST_NONE |
| Norm | 每步`PIPE_V` | Sum/Broadcast/Mul | 后继 | FP32片 | SumParams(heads,alignedD,D) |
| 输出 | CAST_ROUND+TQue | dx | GM | D前缀 | head/core tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeUBTilingStrategy/FillTilingData` | norm资源/scale |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | R-004 `Impl` | 6key |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | `NormOperationBackward::CopyIn/Compute/CopyOut` | LN body/资源 |

### Route R-005：非affine+Interleave

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey` | R-004 Norm集,R=1 | 6key |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | 对应`Impl` | 显式key | Rope→LN |

#### Device 数据流

##### 函数级调用链

```text
entry -> ProcessQK -> RopeOperationBackward<INTERLEAVE>::Compute
  -> Rotate/Collect -> NormOperationBackward<N>::CopyIn/Compute/CopyOut
```

##### 源码等价伪码

```text
for branch,s,b,headTile: 按concat offset加载sin/cos与dy，FP32 Interleave反向
  若branch N/A=1则加载x/mean/rstd并做FP32 LN；否则直接copy
  round写D；PIPE_ALL后Reset pool
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM dy/x/dx/sin/cos/stats | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`xOffset=(b*S+s)*H*D+h*D`；`ropeOffset=(outputSeq+s)*D`；`statOffset=(b*S+s)*H+h` | `dy=B*H*totalS*D`；`x/dx=B*S*H*D`；`sin/cos=Sr*D`；`stats=B*S*H` 元素 | T/FP32 | caller | queues | launch | concat/rope tail |
| ropeQueue | pool | 2*alignedD*sizeof(T) | T | GM | Cast | seq | depth1 |
| Rope sin/cos/mask | 0/alignedD*4/2*alignedD*4 | 各alignedD*4 | FP32/FP32/u32 | Cast/Prepare | Rope | branch | 三片 |
| out/inGradQue | pool | 各avg*alignedD*sizeof(T) | T | GM/round | Rope/GM | tile | depth1 |
| inQue | pool | avg*alignedD*sizeof(T) | T | x | Norm | tile | LN侧 |
| normQue | pool | 2*alignedNormNum*4 | FP32 | stats | Norm | tile | LN侧 |
| Norm inX/share/mean/rstd | offsets同R-004 | 对应四片 | FP32 | queues | LN | branch | 不别名 |
| workspace/weight/RegTensor | N/A | 0 | N/A | N/A | N/A | N/A | 非affine |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(3·A·E·s+8·A·E+16·N+2·R·s+12·R, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(3·A·E·s+8·A·E+16·N+2·R·s+12·R, 96·E)
Rope 与 Norm 的分配同属当前 branch pool，主buf把 Rope FP32 结果直接交给 Norm
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 主buf FP32 `buf` | Rope `Collect` | `NormOperationBackward::Compute` | 下一 head tile | 两者之间的 `PIPE_V`，输出 queue 提交 | 不在 Rope/Norm 之间插入降精度 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 前置 | SyncAll | kernel | branch | 全核 | schedule1 |
| Rope→Norm | TQue+`PIPE_V` | GM/Rope | LN | FP32主buf | 无中间round |
| Norm→输出 | `PIPE_V`+TQue | LN | GM | D前缀 | core/head/rope tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeUBTilingStrategy/Key` | 两类UB |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | R-005 `Impl` | 6key |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | `RopeOperationBackward::Compute/NormOperationBackward::Compute` | 串联顺序 |

### Route R-006：非affine+Half

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey` | R-004 Norm集,R=2 | 6key |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | 对应`Impl` | 显式key | Half→LN |

#### Device 数据流

##### 函数级调用链

```text
entry -> ProcessQK -> RopeOperationBackward<HALF>::Compute/Rotate/Collect
  -> NormOperationBackward<N>::CopyIn/Compute/CopyOut
```

##### 源码等价伪码

```text
for branch,s,b,headTile: dyOffset=b*H*totalS*D+(outputSeq+s)*D+headTile*avg*totalS*D；inOffset=(b*S+s)*H*D+headTile*avg*D；以half mask和sin前半负号做FP32 RoPE反向
  N/A=1侧继续FP32 LN反向，N/A=0侧copy；round并写BSHD D前缀
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs/outputs | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`x/outOffset=(b*S+s)*H*D+h*D`；`ropeOffset=(outputSeq+s)*D` | `dy=B*H*totalS*D`；`x/out=B*S*H*D`；`sin/cos=Sr*D` 元素 | T/FP32 | caller | route | launch | D有效 |
| ropeQueue | pool | 2*alignedD*sizeof(T) | T | sin/cos | Cast | seq | depth1 |
| Rope sin/cos/mask | 0/alignedD*4/2*alignedD*4 | 各alignedD*4 | FP32/FP32/u32 | Cast/Prepare | Half | branch | 三片 |
| out/inGradQue | pool | 各avg*alignedD*sizeof(T) | T | GM/round | body/GM | tile | depth1 |
| inQue/normQue | pool | avg*alignedD*sizeof(T)；2*alignedNormNum*4 | T/FP32 | x/stats | LN | tile | depth1 |
| Norm inX/share/mean/rstd | offsets同R-004 | 四片 | FP32 | queues | LN | branch | 分片 |
| workspace/weight/RegTensor | N/A | 0 | N/A | N/A | N/A | N/A | 非affine |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(3·A·E·s+8·A·E+16·N+2·R·s+12·R, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(3·A·E·s+8·A·E+16·N+2·R·s+12·R, 96·E)
Half 只改变 mask/符号，不改变分配式
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 主buf/pool | Half `Rotate/Collect` | LN/CopyOut | 下一 tile/分支 | `PIPE_V`、`PIPE_ALL+Reset` | D 偶数，仅 D 前缀写回 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 前置 | SyncAll | kernel | route | 全核 | schedule1 |
| Half/LN | TQue+逐指令`PIPE_V` | dy/sin/x/stats | dx | FP32 | D/2与Sum有效D |
| 输出 | CAST_ROUND+TQue | dx | GM | D前缀 | 三类tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `FillTilingData/Key` | Half+norm参数 |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | R-006 `Impl` | 6key |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h | `RopeOperation<HALF,false>` | half tail |

### Route R-007：affine Norm

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeWorkSpace/Key` | N/A含2,R=0 | 10key+workspace |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | 对应`Impl` | affine模板 | affine body |

#### Device 数据流

##### 函数级调用链

```text
entry -> Prepare/SyncAll -> ProcessQK/EncoderQK -> NormOperationBackward::Prepare/Compute
  -> BinarySum/CopyOut/AffineAccSum/PostProcess -> global PostProcess/SyncAll/block0 output
```

##### 源码等价伪码

```text
block0清零8输出及实际workspace；SyncAll
for affine branch,s,b,headTile: weight广播；FP32计算dx、dW=sum(dy*xhat)、dB=sum(dy)
  [dW,dB]写级联workspace；满avgHeadsLow读回BinarySum并推下一层
branch结束写该core主workspace片；SyncAll
if blockIdx==0: for core累加8*alignedD，round到T，分别写八输出前D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM数据/weight/stats | 公共offset；weight0:D | 逻辑extent | T/FP32 | caller | queues | launch | affine侧 |
| GM八dγ/dβ | 各0:D | 各D*sizeof(T) | T | block0 | caller | launch | 先清零 |
| 主workspace | `core*8*alignedD+workspaceId*2*alignedD` | 8*usedCore*alignedD*4 | FP32 GM | branch post | block0 | launch | workspaceId Q/eQ/K/eK |
| 级联workspace | `(core*L+i)*avgLow*2*alignedD+slot*2*alignedD` | affineBlockNums*usedCore*avgLow*2*alignedD*4 | FP32 GM | local dW/dB | 层级sum | branch | Host40层上界 |
| out/inGradQue | pool | 各avg*alignedD*sizeof(T) | T | GM/round | Norm/GM | tile | depth1 |
| inQue | pool | avg*alignedD*sizeof(T) | T | x | Cast | tile | depth1 |
| normQue | pool | 2*alignedNormNum*4 | FP32 | stats | Norm | tile | depth1 |
| weightQue | pool | alignedD*sizeof(T) | T | weight | Cast | branch | depth1 |
| inX/share/mean/rstd | offsets0、avg*alignedD、2*avg*alignedD、后加alignedNormNum | 前两avg*alignedD，后两alignedNormNum floats | FP32 | queues | LN | branch | 四片 |
| weight LocalTensor | 2*avg*alignedD+2*alignedNormNum | avg*alignedD floats | FP32 | Broadcast | dxhat | branch | 独立 |
| dW LocalTensor | 3*avg*alignedD+2*alignedNormNum | alignedD floats | FP32 | BinarySum | workspace | branch | 累计 |
| dB LocalTensor | 上项+alignedD | alignedD floats | FP32 | BinarySum | workspace | branch | 独立 |
| workspaceGetQue | postPool | 8*alignedD*4 | FP32 | GM | Add | final | depth1 |
| workspacePushQue | postPool | 8*alignedD*sizeof(T) | T | Cast | GM | final | depth1 |
| workspaceBuf | postPool | 8*alignedD*4 | FP32 TBuf | Duplicate/Add | Cast | block0 | 独立 |
| RegTensor | N/A | 0显式 | intrinsic | Vector | Vector | 指令 | LocalTensor归约 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(3·A·E·s+12·A·E+16·N+E·s+16·E, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(3·A·E·s+12·A·E+16·N+E·s+16·E, 96·E)
postPool 需要 8·E·4 + 8·E·s + 8·E·4，对 s≤4 被 max(96·E) 覆盖
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dγ/dβ LocalTensor | `BinarySum` | GM 级联 workspace | 下一 tile | `VToMTE3Sync/MTE3ToMTE2Sync` | 每个 slot 读回前必须等待 GM 写完 |
| postPool | block0 终局 Add/Cast | 八个 GM 输出 | Kernel 结束 | 第二次 `SyncAll`、queue 释放 | 其他 core 已提交主 workspace |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零 | 首次SyncAll | block0 | all cores | outputs/workspace | affine必需 |
| local | queues+PIPE_V | GM/Vector | dx,dW,dB | UB slices | FP32累加 |
| 级联 | MTE3→MTE2、MTE2→V、V→MTE3 SetFlag/WaitFlag | GM write/read/Vector | 下一层 | workspace slot | 2*alignedD粒度 |
| branch | PIPE_ALL | PostProcess | Reset | pool/main slice | 非满层也收口 |
| 全核 | 第二次SyncAll | cores | block0 | 主workspace | usedCore次FP32 Add |
| final | CAST_ROUND+TQue | FP32总和 | GM | D前缀 | T=FP32用Adds |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeWorkSpace/FillTilingData` | 两区workspace |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | R-007 `Impl` | 10key |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | `Prepare/Compute/AffineAccSum/PostProcess` | 两级归约 |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h | `VToMTE3Sync/MTE3ToMTE2Sync/MTE2ToVSync` | 硬事件边 |

### Route R-008：affine+Interleave

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeWorkSpace/Key` | N/A含2,R=1 | 10key |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | 对应`Impl` | affine+Rope | R-008 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Prepare/SyncAll -> RopeOperationBackward<INTERLEAVE>::Compute
  -> NormOperationBackward::Compute/AffineAccSum/PostProcess -> global PostProcess/SyncAll/block0
```

##### 源码等价伪码

```text
清零后SyncAll；for branch,s,b,headTile: dyOffset=b*H*totalS*D+(outputSeq+s)*D+headTile*avg*totalS*D；inOffset=(b*S+s)*H*D+headTile*avg*D；FP32 Interleave反向
  branch=0/1/2分别copy/LN/affine LN；affine dW/dB走级联GM
写数据梯度；SyncAll后block0归约并round八个向量到GM D前缀
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM数据/sin/cos/weight/stats | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`x/outOffset=(b*S+s)*H*D+h*D`；`ropeOffset=(outputSeq+s)*D`；`statOffset=(b*S+s)*H+h` | `dy=B*H*totalS*D`；`x/out=B*S*H*D`；`sin/cos=Sr*D`；`weight=D`；`stats=B*S*H` 元素 | T/FP32 | caller | route | launch | concat+rope tail |
| 主workspace | core*8*alignedD+workspaceId*2*alignedD | 8*usedCore*alignedD*4 | FP32 | branch | block0 | launch | 四片 |
| 级联workspace | `(core*L+i)*avgLow*2*alignedD+slot*2*alignedD` | affineBlockNums*usedCore*avgLow*2*alignedD*4 | FP32 | dW/dB | sum | branch | 分层 |
| ropeQueue | pool | 2*alignedD*sizeof(T) | T | GM | Cast | seq | depth1 |
| Rope sin/cos/mask | 0/alignedD*4/2*alignedD*4 | 各alignedD*4 | FP32/FP32/u32 | Cast/Prepare | Rope | branch | 三片 |
| out/inGradQue | pool | 各avg*alignedD*sizeof(T) | T | GM/round | body/GM | tile | depth1 |
| inQue/normQue/weightQue | pool | avg*alignedD*sizeof(T)；2*alignedNormNum*4；alignedD*sizeof(T) | T/FP32/T | GM | Norm | tile/branch | 各depth1 |
| Norm inX/share/mean/rstd | offsets同R-007 | 四片 | FP32 | queues | LN | branch | 分片 |
| Norm weight/dW/dB | offsets同R-007 | 三片 | FP32 | Broadcast/Sum | dx/workspace | branch | 分片 |
| final get/push queues | postPool | 8*alignedD*4；8*alignedD*sizeof(T) | FP32/T | GM/Add/Cast | outputs | final | depth1 |
| final workspaceBuf | postPool | 8*alignedD*4 | FP32 TBuf | Add | Cast | block0 | 独立 |
| RegTensor | N/A | 0显式 | intrinsic | Vector | Vector | 指令 | 无对象 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(3·A·E·s+12·A·E+16·N+E·s+16·E+2·R·s+12·R, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(affine_pool_bytes + 2·R·s + 12·R, 96·E)
affine_pool_bytes=3·A·E·s+12·A·E+16·N+E·s+16·E
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 主buf FP32 | Interleave `Collect` | affine LN | 下一 tile | `PIPE_V` | Rope→Norm 中间不 round |
| linked/post pool | branch 计算/终局归约 | workspace/输出 | 下一 branch | `PIPE_ALL+Reset`、`SyncAll` | 终局仅 block0 执行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零/最终 | 两次SyncAll | block0/cores | cores/block0 | workspace | 跨核协议 |
| Rope→Norm | TQue+PIPE_V | GM/Rope | Norm | FP32 dy | repeat/tail，不中间round |
| affine级联 | 三种HardEvent | MTE/Vector | 下一层 | slot | 2*alignedD |
| final | CAST_ROUND | FP32 | outputs | D前缀 | core/head/rope tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeUBTilingStrategy/WorkSpace` | Rope+affine容量 |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | R-008 `Impl` | 10key |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | `RopeOperationBackward/NormOperationBackward/PostProcess` | 串联及归约 |

### Route R-009：affine+Half

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeWorkSpace/Key` | N/A含2,R=2 | 10key |
| Router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | 对应`Impl` | affine+Half | R-009 |

#### Device 数据流

##### 函数级调用链

```text
entry -> Prepare/SyncAll -> RopeOperationBackward<HALF>::Compute/Rotate/Collect
  -> NormOperationBackward::Compute/AffineAccSum/PostProcess -> global PostProcess/SyncAll/block0
```

##### 源码等价伪码

```text
清零后SyncAll；for branch,s,b,headTile: dyOffset=b*H*totalS*D+(outputSeq+s)*D+headTile*avg*totalS*D；inOffset=(b*S+s)*H*D+headTile*avg*D；sin前half取负并Gather交换half，得到FP32 RoPE梯度
  执行copy/LN/affine LN；dW/dB级联，数据梯度写D
SyncAll；block0按core加8*alignedD，round并写八输出D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM数据/sin/cos/weight/stats | `dyOffset=b*H*totalS*D+(outputSeq+s)*D+h*totalS*D`；`x/outOffset=(b*S+s)*H*D+h*D`；`ropeOffset=(outputSeq+s)*D`；`statOffset=(b*S+s)*H+h` | `dy=B*H*totalS*D`；`x/out=B*S*H*D`；`sin/cos=Sr*D`；`weight=D`；`stats=B*S*H` 元素 | T/FP32 | caller | route | launch | D偶数 |
| 主workspace | core*8*alignedD+workspaceId*2*alignedD | 8*usedCore*alignedD*4 | FP32 | branch | block0 | launch | 四片 |
| 级联workspace | `(core*L+i)*avgLow*2*alignedD+slot*2*alignedD` | affineBlockNums*usedCore*avgLow*2*alignedD*4 | FP32 | local | sum | branch | Host上界 |
| ropeQueue | pool | 2*alignedD*sizeof(T) | T | GM | Cast | seq | depth1 |
| Rope sin/cos/mask | 0/alignedD*4/2*alignedD*4 | 各alignedD*4 | FP32/FP32/u32 | Cast/Prepare | Half | branch | 三片 |
| out/inGradQue | pool | 各avg*alignedD*sizeof(T) | T | GM/round | route/GM | tile | depth1 |
| inQue/normQue/weightQue | pool | avg*alignedD*sizeof(T)；2*alignedNormNum*4；alignedD*sizeof(T) | T/FP32/T | GM | Norm | tile/branch | depth1 |
| Norm inX/share/mean/rstd | offsets同R-007 | 四片 | FP32 | queues | LN | branch | 分片 |
| Norm weight/dW/dB | offsets同R-007 | 三片 | FP32 | Broadcast/Sum | dx/workspace | branch | 分片 |
| final get/push queues | postPool | 8*alignedD*4；8*alignedD*sizeof(T) | FP32/T | GM/Add/Cast | outputs | final | depth1 |
| workspaceBuf | postPool | 8*alignedD*4 | FP32 | Add | Cast | block0 | TBuf |
| RegTensor | N/A | 0显式 | intrinsic | Vector | Vector | 指令 | 固定仓无对象 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  main["buf_ · 12·A·E+256 B"]:1
  pool["linked pool · max(3·A·E·s+12·A·E+16·N+E·s+16·E+2·R·s+12·R, 96·E) B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 12·A·E+256 + max(affine_pool_bytes + 2·R·s + 12·R, 96·E)
Half 和 Interleave 仅替换 mask/符号规则，不替换资源上界
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 主buf/pool/workspace slot | Half/affine Norm | 数据梯度与级联归约 | 下一 tile/层 | `PIPE_V`、三种 HardEvent、`PIPE_ALL+Reset` | D 偶数；slot 读写严格交接 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 跨核 | 两次SyncAll | block0/cores | cores/block0 | workspace/output | 顺序固定 |
| Half→Norm | TQue+PIPE_V | sin/dy | LN | FP32 | half=D/2，aligned padding |
| 级联 | MTE3→MTE2、MTE2→V、V→MTE3 | pipeline producer | WaitFlag | slots | 非满层也收口 |
| final | CAST_ROUND+TQue | FP32 | GM | D前缀 | 三类tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `ComputeTilingKey/WorkSpace` | Half affine配置 |
| Kernel entry/router | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | R-009 `Impl` | 10key |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h | `RopeOperation<HALF,false>/HardEvent helpers` | half和同步 |
| Device body/helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | `AffineAccSum/PostProcess` | 归约收口 |

## 算子特定优化

### Trick T-001：链接 TBufPool 让六个数据分支与终局归约共用一块 UB

#### 触发与源码依据

- 适用 Route：R-001～R-009。
- 触发条件：单个 Kernel 依次执行 Query、Key、Value 分支，各分支末在所有 queue/Vector 消费完成后 `Reset`。
- Golden 来源：[`posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h:709-716`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h#L709-L716)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
pipe_->InitBuffer(buf_, bufSize);
pipe_->InitBufPool(queryPool_, poolSize);
pipe_->InitBufPool(keyPool_, poolSize, queryPool_);
pipe_->InitBufPool(valuePool_, poolSize, queryPool_);
pipe_->InitBufPool(encoderQueryPool_, poolSize, queryPool_);
pipe_->InitBufPool(encoderKeyPool_, poolSize, queryPool_);
pipe_->InitBufPool(encoderValuePool_, poolSize, queryPool_);
pipe_->InitBufPool(postPool_, poolSize, queryPool_);
```

#### 机制与设计原因

后六个 pool 都以 `queryPool_` 作为链接起点，因此物理字节账是 `bufSize+poolSize`，而不是 `bufSize+7*poolSize`。`ProcessQK`、`ProcessEncoderQK` 和 `ProcessV` 在分支末先做 `PipeBarrier<PIPE_ALL>()`再 `bufPool.Reset()`（同文件 911-915、959-963、1007-1008），所以下一分支覆盖时，上一分支的 MTE/Vector 消费已结束。`poolSize` 取数据分支最大需求与 `3*TOTAL_NORM_NUM*alignedNormDim*sizeof(float)` 的最大值，因而 affine 最后 block0 的 get/push/TBuf 也能在同一块 pool 中完成，不另保留一块并存 UB。

#### 迁移边界

- 适用：多个串行分支的局部工作集上界不同，但生命期不重叠的纯 Vector Kernel。
- 前提：分支之间必须有能证明所有 queue、Vector 和 GM 写完成的 barrier/释放边；一次只能有一个链接 pool owner 活跃。
- 失效：若改为分支并发、去掉 `PIPE_ALL`，或让 post 与数据分支重叠，必须拆成独立物理 pool 并重算 UB，不能保留当前字节账。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把七个 `InitBufPool` 当成七份并存 UB，会与链接起点和 `Reset` 生命期证据冲突；反过来，只看链接关系而删掉 `PIPE_ALL` 也会使覆盖早于消费完成。

## 模板与实现对比

| 对比维度 | R-001 纯拆分 | R-002/3 Rope | R-004/5/6 非affine Norm | R-007/8/9 affine | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | N=A=R=0 | N=A=0，R=1/2 | N/A∈{0,1}且有1，R=0/1/2 | N/A有2，R=0/1/2 | key 是属性的位权和，非运行时性能 selector |
| 模板与 dtype | NONE/NONE/NONE | Interleave/Half | LN 或 NONE | affine LN/LN/NONE | 三种 dtype 共用模板，由 binary config 生成独立签名 |
| 循环与分核 | 全部共用 seq→batch→head tile | 增加每 seq 的 sin/cos 预处理 | 每 tile 加 x/stats 与 D 维 Sum | 再加 dγ/dβ 层级归约 | `usedCore` 只按最大 seq 生产，head 用 `avgHeads` 分 tile |
| Queue/TBuf/寄存器 | dy/dx queue+主buf | +rope queue/sin/cos/mask | +x/stats queue和 Norm 分片 | +weight/dγ/dβ/post 分片 | 无显式 RegTensor；所有 queue depth=1 |
| UB 总占用与复用 | 最小 pool | +Rope fixed | +Norm per-head/fixed | +affine/post，可再+Rope | 七 pool 链接复用，逐 Route 字节账见上 |
| 搬运/Vector API | BHSD→BSHD、Cast/Adds | Gather/Mul/Add | BroadCast/Sum/MulAddDst | BinarySum+GM 级联 | Rope→Norm 在 FP32 主buf直接交接 |
| 同步、精度与 tail | queue/PIPE_V；三类 tail | +Rope mask/active tail | +SyncAll/Sum 有效 D | +HardEvent/二次 SyncAll | 输入 CAST_NONE→FP32→输出 CAST_ROUND |
| fallback/失败边界 | 非法属性失败 | rope 越界行跳过 | optional 配对无安全 fallback | 40 层和空域无显式保护 | 不存在路线间性能 fallthrough |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持：公开文档、OpDef 与 `ascend910b` 接线；详见无后缀卡 | 支持：公开文档、OpDef 与 `ascend910_93` 接线；详见无后缀卡 | 支持：公开文档、OpDef、`ascend950` 三 dtype binary config 与 CI 接线 |
| Host/Launcher | 与 A3/950 共用 ACLNN/Tiling；平台资源值不写入本卡 | 与 A2/950 共用 54-key 逻辑；平台资源值不写入本卡 | `PlatformAscendC` 在运行时提供 AIV 数、UB 与 system workspace，其他 Host 逻辑同源 |
| Kernel/资源 | 同源纯 Vector body，平台参数由各自 Tiling 生产 | 同源 body/POD/key | 本卡覆盖 950；固定仓没有 950 专用 Kernel body，不能把动态 UB/核数写成常量 |

## 限制与失败边界

- optional weight/mean/rstd/sin/cos 与属性的非空配对主要是公开契约；Host允许空optional，命中模板后没有安全fallback。
- POD uint64字段在Device窄化为uint32，固定仓没有乘法/窄化溢出检查；`usedCore=0`空域也没有独立ACLNN快返回。
- UB 容量候选 `<2` 直接 `GRAPH_FAILED`，但其后 `avgHeads=min(capacityHeads,H)`，H=1 时最终 avg 为1且 Device 用 A=2 分配；无小UB fallback。`DOUBLE_BUFFER=1`，不得描述成双缓冲。
- Device 物理分配包含 `poolSize=max(routePool,96*alignedD)`，Host 用分项系数估算 `avgHeads`，但固定仓没有在 `InitBuffer` 后再显式校验 `bufSize+poolSize<=UB`；不得把该不等式写成已验证保证。
- affine Host按40层保守分配，Device数组也是40项，但没有显式 `affineBlockNums<=40` 检查。
- InferShape 对三份上游 S 与主/encoder 序列和不等只记日志；Tiling 由主/encoder shape 重算 `total*Seq_`，没有再与上游 S 比较，不能把它描述成已失败关闭。调用者必须遵守公开 shape 契约，否则 Kernel 的 BHSD stride 依据与实际上游布局可能不一致。
- entry 对未知 tiling key 没有显式错误/fallback，54 个 `Impl` 全不命中时直接到函数末尾且不写输出；正常 Host 属性校验不会生成该状态。
- 默认非确定性/可开启确定性只见公开文档，Kernel无独立确定性key；未真机验证。
- 宏ABI确切offset与system/extra workspace指针映射固定仓不可见；无显式RegTensor、Atomic或跨执行族资源。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开文档 | posembedding/norm_rope_concat_grad/docs/aclnnNormRopeConcatBackward.md | API/公式/平台 | 语义与支持范围 |
| 入口声明/约束 | posembedding/norm_rope_concat_grad/op_host/op_api/aclnn_norm_rope_concat_grad.h | 两段接口 | 参数ABI |
| 入口实现 | posembedding/norm_rope_concat_grad/op_host/op_api/aclnn_norm_rope_concat_grad.cpp | `CheckParam/GetWorkspaceSize/Backward` | ACLNN流程 |
| L0 Launcher | posembedding/norm_rope_concat_grad/op_host/op_api/norm_rope_concat_grad.cpp | `l0op::NormRopeConcatGrad` | 输出与launcher |
| OpDef/平台 | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_def.cpp | `NormRopeConcatGrad` | tensor/属性/平台 |
| InferShape | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_infershape.cpp | `InferShape4NormRopeConcatGrad` | shape/dtype |
| POD | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.h | `NormRopeConcatGradTilingData` | 22字段 |
| Host/Tiling | posembedding/norm_rope_concat_grad/op_host/norm_rope_concat_grad_tiling.cpp | `DoTiling/Compute*` | key/资源/workspace |
| Kernel entry | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.cpp | `norm_rope_concat_grad/Impl` | 54key |
| Kernel body | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad.h | 三类模板 | 数据流/资源/同步 |
| Kernel helper | posembedding/norm_rope_concat_grad/op_kernel/norm_rope_concat_grad_base.h | Rope/HardEvent | mask/tail/event |
| 950 binary config | posembedding/norm_rope_concat_grad/op_host/config/ascend950/norm_rope_concat_grad_binary.json | `NormRopeConcatGrad_{bf16,fp16,fp32}` | 950 三 dtype 二进制签名接线 |
| Host构建 | posembedding/norm_rope_concat_grad/op_host/CMakeLists.txt | `add_modules_sources` | 编译接入 |
| 模块构建 | posembedding/norm_rope_concat_grad/CMakeLists.txt | `add_subdirectory` | 目录接入 |
| ACLNN 头接线 | cmake/level2_ops_list | `aclnn_norm_rope_concat_grad.h` | 公开头进入 level2 列表 |
| 950 CI | scripts/ci/ascend950/ops_transformer_operator_list.yaml | `norm_rope_concat_grad` | 950 流水线枚举 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三平台共用22字段 | 顺序/位宽/单位固定；offset不可猜 | `TilingData/Fill/Init` |
| 调用顺序 | 两段ACLNN+单Kernel | executor/L0/ViewCopy/同stream顺序 | `GetWorkspaceSize/CommonOpExecutorRun` |
| workspace/内存 | pool复用和两级归约 | depth1、各offset/extent/alignedD不可合并 | `Init/AffineAccSum/PostProcess` |
| 同步 | queue、barrier、HardEvent、SyncAll | GM读写及block0等待边不可删 | 三HardEvent helper/Prepare/PostProcess |
| 精度与尾块 | FP32中间与三重tail | CAST_NONE→FP32→CAST_ROUND、Sum有效D、Rope mask不可换序 | `Compute/GetSeqRange` |
| 失败边界 | Host/Tiling校验 | optional配对、窄化、40层、avgHeads缺口不得伪造fallback | `CheckParam/CheckInput/ComputeUBTilingStrategy` |