---
type: Operator
title: "InplaceQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识"
description: "固定 CANN 9.1 源码中 aclnnInplaceQuantMatmulAllReduceAddRmsNorm 的公开接口、A2 Host/Tiling、MIX Kernel、通信和 AddRMSNorm 路由知识。"
tags: [task.implementation, task.lookup, topic.data_movement.communication.collective_communication, topic.data_movement, topic.memory, topic.precision]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/README.md
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_api/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_api/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_api/aclnn_quant_matmul_all_reduce_add_rms_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_api/matmul_all_reduce_arn_util.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_host/inplace_matmul_all_reduce_add_rms_norm_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/inplace_quant_matmul_all_reduce_add_rms_norm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/quant_matmul_all_reduce_add_rms_norm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/common_add_rms_norm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce/op_host/op_tiling/arch22/quant_matmul_all_reduce_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce/op_kernel/arch22/quant_matmul_all_reduce_tiling_data.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_host/op_tiling/mc2_tiling_struct.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_kernel/mc2_tiling_struct.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_data.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_cache.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/mm_allreduce_add_rms_norm_quant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_quant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_split_d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_merge_n.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_single_n.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_multi_n.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_host/config/ascend910b/inplace_matmul_all_reduce_add_rms_norm_binary.json
    role: evidence
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_graph/inplace_matmul_all_reduce_add_rms_norm_gen_task.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_graph/matmul_all_reduce_gen_task_ops_utils.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/common/op_graph/mc2_gen_task_ops_utils.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T16:28:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# InplaceQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize` / `aclnnInplaceQuantMatmulAllReduceAddRmsNorm` |
| 计算语义 | INT8 MatMul + INT32 bias → dequantize → HCCL AllReduce(SUM) → add residual → RMSNorm；`residual` 同时承载原位 `y` |
| 输入输出 | x1/x2 为 INT8，bias 为可选 INT32，dequantScale 为 UINT64/INT64 或 BF16，residual/gamma/y/normOut 为 FP16 或 BF16；`y` 与 `residual` 同址 |
| 执行类型 | `MIX_CV` 可达节点：A2 `KERNEL_TYPE_MIX_AIC_1_2`；同图还有附属流 AICPU KFC 通信服务和 Hcom wait/record 任务 |
| Kernel 路线 | 顶层四路：FP16/BF16 × x2 不转置/转置；ARN 子 key 为 FP16 `10..14`、BF16 `30..33` |
| 核心资源 | GM 输入/原位输出、user workspace、Matmul L1/L0A/L0B/L0C、AIV UB/寄存器、HCCL window/output segment；Cube 做 QBMM，AIV 做 BF16 dequant/ARN |
| 来源闭包 | `source_closed/kernel_tiling_closed`：固定 commit 的 ACLNN→OpDef→arch22 Tiling/POD→任务图→MIX Kernel→构建配置已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 x1 展平后的矩阵为 \(X\in\mathbb{Z}_8^{M\times K}\)，x2 为 \(W\in\mathbb{Z}_8^{K\times N}\)，bias 为 \(b\in\mathbb{Z}_{32}^{N}\)，反量化尺度为 \(s\)，残差为 \(r\)，gamma 为 \(\gamma\)。每个 rank 先计算局部量化矩阵乘：

```text
q = MatMul(x1, x2, transposeX1=false, transposeX2=derived-from-x2-view) + bias
z_local = Dequant(q, dequantScale)
z = AllReduceSUM(z_local, group)
y = cast_out(z + residual)
normOut = y * gamma / sqrt(mean(y * y, axis=-1) + epsilon)
```

公开 inplace 封装把同一 `residual` 指针同时传给非原地 inner op 的 `residual` 输入和 `y` 输出，所以 `y` 覆盖 `residual`。Kernel 宏也以 `{residual, gamma, y, normOut}` 绑定 AddRMSNorm，其中 `y` 与 `residual` 是同一 GM 地址。`normOut` 不别名该地址。

### 输入、输出与属性

| 参数 | shape / format | dtype | 公开约束与语义 |
|---|---|---|---|
| `x1` | `[M,K]` 或 `[B,S,K]`，ND | INT8 | 必需；不支持转置；三维时 `M=B*S` |
| `x2` | `[K,N]` 的逻辑矩阵，ND | INT8 | 必需；可由非连续最后两维 view 表达转置，OpDef 标记 `IgnoreContiguous` |
| `bias` | `[N]`，ND | INT32 | 可选；加在 INT32 MatMul 结果上 |
| `residual` | `[B,S,N]`，ND | FP16 或 BF16 | 必需；既是 residual 输入又是原地 `y` 输出 |
| `dequantScale` | `[1]`、`[N]` 或 `[1,N]`，ND | 输出 FP16 时 UINT64/INT64；输出 BF16 时 BF16 | 必需；INT64 descriptor 在 API 内改为 UINT64 解释；`[1]` 为 per-tensor，后两者为 per-channel |
| `gamma` | `[N]`，ND | 与 `residual` 相同 | 必需；RMSNorm 逐列缩放 |
| `antiQuantScale` / `antiQuantOffset` / `pertokenScale` | 本入口不暴露 | — | inner op 固定传空；本卡不得套用 weight-quant 或 per-token 路线 |
| `group` | HCCL 通信域字符串 | string | 必需；1/2/4/8 卡，卡间 HCCS 全互联并处于同一通信域 |
| `reduceOp` | 固定语义 | string | inner op 固定 `sum` |
| `epsilon` | 标量 | float | RMSNorm 分母稳定项，公开约束 `(0,1)`，写入 main/tail 两套 tiling |
| `commTurn` | 标量 | int64 | 公开文档当前只支持 0；由公共 MC2 Host 校验 |
| `streamMode` | 标量 | int64 | 仅接受 `1`（`NUM_ACL_STOP_ON_FAILURE`） |
| 原地 `y` | 与 `residual` 相同 | 与 `residual` 相同 | 写回 `residual` |
| `normOut` | 与 `residual` 相同 | 与 `residual` 相同 | 独立输出 |

维度限制来自公开文档：`M <= INT32_MAX`；相关末维不超过 `65535`。`B*S=0` 或 `N=0` 走空张量 executor；`K=0` 明确报错。确定性可由运行环境的 `HCCL_DETERMINISTIC` 控制，不能由该 ACLNN 参数单独打开。

### 入口与版本

该接口是公开两段式 ACLNN：GetWorkspaceSize 负责检查、建立 executor 和返回 workspace，执行接口在传入 stream 上运行。文档已把它标为废弃接口并建议拆成多个算子，但固定 commit 仍提供完整公开头、实现、OpDef、Tiling 和 Kernel，因而可发布全链路知识。

固定目录的支持表为：A2 支持；A3 不支持；Ascend 950 不支持。OpDef 只配置 `ascend910b`，binary 配置也只有 `op_host/config/ascend910b/`。因此无后缀卡的最新实际支持代际是 A2，不把仓内其他 MC2 算子的 arch31/arch35 实现误归入本入口。

## Host/Launcher 执行流程

### 参数校验与前处理

1. inplace GetWorkspaceSize 直接调用 `aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize`，把 `residual` 同时放到 inner op 的 residual 输入与 `y` 输出槽。
2. 非原地量化 API 检查 `x1`、`x2`、`residual`、`dequantScale`、`gamma` 及 `normOut` 等必需指针，并调用公共 shape/dtype 检查。
3. `dequantScale` 为 INT64 时，descriptor 被改写为 UINT64，以匹配 FP16 反量化尺度的位模式。
4. 任一必需张量为空时先区分 K：`K=0` 报错；其他空域建立 empty executor，workspace 为 0。
5. `streamMode` 不等于 `NUM_ACL_STOP_ON_FAILURE(1)` 时返回参数错误。
6. `MatmulAllReduceArnUtil` 固定 `transposeX1=false`，通过 `IsTransposeLastTwoDims(x2)` 推导 `transposeX2`；anti-quant 与 per-token 参数固定为空。

### 参数、Workspace 与生命周期

公共量化 MC2 tiling 先为 main/tail 各生成一份 `Mc2QuantBatchMatmulV3TilingData`，再生成 HCCL 描述和两份 AddRMSNorm tiling。FP16 QBMM 在 Matmul/Fixpipe 内完成 INT32 累加与 UINT64 scale 反量化，输出直接落到 FP16 `normOut` 临时区。BF16 QBMM 则把每个有效 Cube 核的 INT32 结果放到 user workspace；单核跨度是

```text
singleOffset = max(singleTimeM, baseM) * max(singleTimeN, baseN) * 4 B
bf16ExtraWorkspace = singleOffset * usedCoreNum
```

Host 在 1 GiB workspace 上限相关分支中可能进一步切分 `singleCoreM`，所以不能把 BF16 额外区简化成固定的 `Mtile*N*4`。该区由 Cube 生产、AIV 反量化消费，消费结束前不能复用。

融合层要求 main/tail AddRMSNorm workspace 相同。ARN 报告的空间是 `16 MiB + 256 B`；融合层扣掉 MC2 已包含的公共 `16 MiB`，最终在 `mc2Workspace` 后增加 `256 B`。entry 同时检查原始 `workspaceGM` 和 `GetUserWorkspace(workspaceGM)`，任一为空即返回。非空调用中，调用方从 GetWorkspaceSize 返回后取得 workspace 所有权，并保持它以及全部输入输出、`group` 关联通信资源有效，直到传入 stream 完成；`residual` 原值在同一时点前仍可能被后续 segment 读取，不能提前释放或复用。

### 启动图与失败传播

```text
ACLNN GetWorkspaceSize
  -> inplace wrapper: residual input == y output
  -> inner op / OpDef(ascend910b, HCCL group)
  -> InplaceQuant...Tiling (类型别名)
     -> QuantMatmulAllReduceTiling(main/tail QBMM + HCCL)
     -> CommonAddRmsNormTiling(main)
     -> CommonAddRmsNormTiling(tail, 仅 tailCnt!=0)

ACLNN Execute(stream)
  -> task generator 请求 attached stream: "aicpu kfc server", reuse key "kfc_stream"
  -> attached stream: Hcom wait -> libccl_kernel.so::RunAicpuKfcSrvLaunch
  -> main stream: Hcom record -> AscendC MIX AIC:2*AIV task
  -> task generator 把隐藏 HCOM context 注入已有 MIX task 参数
```

这不是“仅启动一个 Kernel”：主计算是一个 AscendC MIX task，通信控制还依赖附属流上的 AICPU KFC server 及 Hcom wait/record。任一步获取 context/stream、格式化 task、构造 Hcom task 或注入隐藏输入失败，任务图构造返回 `GRAPH_FAILED`，不会得到完整可执行图。

API/inner op 的非零状态直接向上返回。执行接口允许 `workspace=nullptr, workspaceSize=0` 的空张量 executor；非空执行转交 inner op，inner 失败映射为 `ACLNN_ERR_INNER`。Host tiling 的 shape、dtype、通信、workspace 或 AddRMSNorm 条件失败时不 launch；Device entry 的 workspace guard 失败时不产生有效结果。

## Route 与参数设计

### 分核与资源计算

公共 MC2 Host 从 x1 得到 `M`（二维为 dim0，三维为 dim0×dim1）、`K`（末维），从输出末维得到 `N`。`DoSplitM` 生成 `tileM/tailM` 与对应计数，主、尾分别完成 QBMM tiling。HCCL 参数固定 `commOrder=1`（MatMul 后 AICPU 通信）、`waitPolicy=1`、`rspPolicy=1`、`exitPolicy=0`、`preparePosition=1`、`hasCommOut=1`，并复用 tile/tail 通信窗口。

当前 arch22 量化路线直接实例化仓内 QBMM cache tiling。对每份 main/tail 参数，它先写逻辑 `M/N/Ka/Kb`，再把 TBE cache 单位转换为 Cube 尺度：

```text
baseM = m_l0 * 16
baseN = n_l0 * 16
baseK = k_l0 * (32 / sizeof(int8)) = k_l0 * 32
depthA1 = ceil(kal1_16/k_l0) * m_al1 * db_al1
depthB1 = ceil(kbl1_16/k_l0) * n_bl1 * db_bl1
stepM = m_al1; stepN = n_bl1
stepKa = ceil(kal1_16/k_l0); stepKb = ceil(kbl1_16/k_l0)
```

`usedCoreNum=min(逻辑 block 数, AIC 核数)`，`dbL0A=dbL0B=2`，`dbL0C` 和遍历次序取 TBE cache 结果。显式基本容量为 `a1Length=baseM*baseK` B、`b1Length=baseN*baseK` B、`c1Length=baseM*baseN*4` B；共享 L1 按 `a1Length*depthA1+b1Length*depthB1+biasL1` 计，共享 L0C 按 `c1Length` 计，并受 cache tiling 的 fallback/可行性检查约束。矩阵实际搬运、Mmad 与 Fixpipe 属于外部 Matmul API 原语边界，本卡不伪造其内部指令调度。

FP16 dequant 在 Matmul body 中根据 `isPerTensor` 选择 `SetQuantScalar(uint64)` 或 `SetQuantVector`，随后 `IterateAll` 直出 FP16。BF16 Host 令 `isUbQuant=true`，以 `ubCalcN=baseN` 起算；16-bit 路线每个计算元素的显式常数项为 `2*(sizeof(int32)+sizeof(bfloat16)) + UB_EXTRE_BYTE`，per-channel 时再加入 scale queue，利用真实 UB 容量反推 `ubCalcM<=min(baseM,M)` 和 `needUbBuffer`。

AllReduce 的 `sendCnt/recvCnt=tileM*N`、tail count=`tailM*N`，对应 byte stride 均乘输出元素 2 B；`totalCnt=M*N`。全部 segment 的累计跨度低于当前 window 容量时 Host 选择 `WINDOW_IN`，否则选择 `OUTPUT`。默认容量为 200 MiB，但可由 `HCCL_BUFFSIZE` 改写。A2 只接受该实现支持的通信模式，rank 数公开约束为 1/2/4/8。融合 PostTiling 把 `blockDim` 设为 AIC 核数，并要求 `aicCoreNum * 2 >= max(mainArnBlocks, tailArnBlocks)`；schedule mode 设为 1。MIX 比例是 1 AIC 对 2 AIV。

AddRMSNorm 令 `numRow=M_tile`、`numCol=N`、`alignN=AlignUp(N,16)`，以 AIV 核数算 `blockFactor=ceil(numRow/numCore)` 和 `numBlocks=ceil(numRow/blockFactor)`。FP16 dtype prefix 为 1，BF16 为 3，最终 key=`prefix*10+mode`。

### Route 选择

顶层路由由输出 dtype 与 x2 转置组合决定；同一路由中的主 tile 和 tail tile各自再选择 AddRMSNorm 子 key。

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 no-trans | key 8；`BmmDequant<...,uint64_t,DTYPE_Y,false,false>` | A8W8；output/residual/gamma FP16；x1 不转置；x2 不转置 | x2 转置→R-002；BF16→R-003/004；其他 dtype/量化语义失败 |
| 2 | R-002 FP16 B-trans | key 4104；`BmmDequant<...,uint64_t,DTYPE_Y,false,true>` | 与 R-001 同 dtype，且 `IsTransposeLastTwoDims(x2)=true` | no-trans→R-001；BF16→R-003/004 |
| 3 | R-003 BF16 no-trans | key 8；`BmmDequantBf16<...,DTYPE_Y,DTYPE_Y,false,false,true>` | A8W8；output/residual/gamma/dequantScale BF16；x2 不转置 | x2 转置→R-004；FP16→R-001/002 |
| 4 | R-004 BF16 B-trans | key 4104；`BmmDequantBf16<...,DTYPE_Y,DTYPE_Y,false,true,true>` | 与 R-003 同 dtype，且 `IsTransposeLastTwoDims(x2)=true` | no-trans→R-003；不满足 API/tiling guard 则失败 |

AddRMSNorm 子路由 selector：

```text
if N > 12288:
    mode=1 SplitD; ubFactor is recomputed and 16-aligned
else if blockFactor == 1:
    mode=3 SingleN
else if alignN <= 2000:
    mode=2 MergeN; choose rowFactor/ubFactor from UB capacity
else if dtype == FP16 and N == alignN:
    mode=4 MultiN; rowFactor=(UB-1024-256-2*N)/(16*N+64)
    if rowFactor == 0: fallback mode=0 Normal
else:
    mode=0 Normal
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 no-trans | 8；FP16 branch | A2、A8W8、x1 2D/3D、B no-trans、FP16 output | 附属流 AICPU KFC + 单个 MIX AIC:2*AIV task | `BmmDequant<...,false,false>` + ARN 10..14 | AIC Matmul/Fixpipe 直出 FP16；HCCL count→AIV ARN | main/tail 各自 QBMM/ARN；selector 失败即 Host 失败 |
| R-002 FP16 B-trans | 4104；B-trans | 同 R-001，但 B 为末两维转置 view | 同 R-001 | `BmmDequant<...,false,true>` + ARN 10..14 | B 地址为 `nOffset*K+kOffset`；其余同步同 R-001 | 同 R-001；不生成 transpose buffer |
| R-003 BF16 no-trans | 8；BF16 branch | A2、A8W8、B no-trans、BF16 output/scale | 附属流 AICPU KFC + 单个 MIX task；AIC Cube、AIV dequant/ARN | `BmmDequantBf16<...,false,false,true>` + ARN 30..33 | per-core INT32 workspace→AIV BF16；HCCL count→ARN | main/tail 独立；无 MultiN；1 GiB 分支可重切单核 M |
| R-004 BF16 B-trans | 4104；BF16+B-trans | 同 R-003，但 B 为末两维转置 view | 同 R-003 | `BmmDequantBf16<...,false,true,true>` + ARN 30..33 | BF16 scratch 同 R-003；B 地址改为 trans 公式 | 同 R-003；不生成 transpose buffer |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| ARN SplitD | `N > 12288` | 整行不能按普通 16-bit UB 因子常驻 | 全部 | key 11/31，按列 chunk 两遍处理 | `CommonAddRmsNormTiling::DoTiling` |
| ARN SingleN | `N<=12288 && blockFactor==1` | 一核一行采用固定队列组织 | 全部 | key 13/33；优先于 MergeN/MultiN | `CommonAddRmsNormTiling::DoTiling` |
| ARN MergeN | 前两项未命中且 `alignN<=2000` | UB 可合并多行，按 `18*alignN+260` 反推 | 全部 | key 12/32；否则继续 MultiN/Normal | `CommonAddRmsNormTiling::DoTiling` |
| ARN MultiN | 前三项未命中、FP16、`N==alignN` | 只实现 FP16 对齐多行 body | R-001/002 | `rowFactor==0` 回退 key 10；BF16 直接 Normal | `CommonAddRmsNormTiling::DoTiling` |
| BF16 INT32 暂存 | output BF16 | Cube 与 AIV dequant 间需 INT32 GM 交接 | R-003/004 | 受 1 GiB 分支约束时调整单核 M；tiling 失败则无 launch | `Mc2QuantBatchMatmulV3Tiling::GetWorkspaceSize` / `BmmDequantBf16` |
| HCCL buffer | `tileSendOff+tailSendOff < windowSize` 且无 uint64 溢出 | window 必须容纳全部复用 segment | 全部 | 命中→WINDOW_IN，否则→OUTPUT；确定性时 Device 仍保留 output | `CalcBufferTypeByWindowSize` / `MatmulAllReduceBase::Init` |
| MIX 核比 | `max(mainArnBlocks,tailArnBlocks) <= 2*aicCoreNum` | entry 固定 1 AIC:2 AIV | 全部 | 不满足则 fused PostTiling 失败 | `QuantMatmulAllReduceAddRmsNormTiling::PostTiling` |
| QBMM 可行性 | A8W8、shape/dtype、RCS、split-M、QBMM、HCCL 依次成功 | Device 不自选 shape/layout | 全部 | 任一步失败，无 Device fallback | `QuantMatmulAllReduceTiling::IsCapable/DoOpTiling` |

### POD/ABI

显式 Device ABI 的 12 个 GM 参数严格依次为：`aGM, bGM, biasGM, residualGM, gammaGM, antiquantScaleGM, antiquantOffsetGM, dequantGM, yGM, normOutGM, workspaceGM, tilingGM`。本入口不暴露 antiquant，但占位参数仍存在；任务生成器还会向任务参数注入不属于该 C 签名的隐藏 HCOM context。不能按 ACLNN 参数表推测 Device 参数顺序。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| top | `quantMatmulAllReduceTilingData` | struct | offset 0，size `Q` | bytes | quant MC2 tiling | MatMul/HCCL wrapper | `Q=608+2*sizeof(TCubeTiling)` |
| top | `addRMSNormTileTilingData` | struct | offset `Q`，size 28 | bytes | main ARN tiling | ARN main | 总会生成 |
| top | `addRMSNormTailTilingData` | struct | offset `Q+28`，size 28 | bytes | tail ARN tiling | ARN tail | 仅 `tailCnt!=0` 写/读 |
| top | `addRmsNormTilingeKeyData` | struct | offset `Q+56`，size 16 | bytes | fused PostTiling | ARN router | top size `Q+72`，align 8 |
| ARN | `num_row` | uint32/32 | offset 0 | elements | `SetTilingData` | 行映射 | main/tail 不同 |
| ARN | `num_col` | uint32/32 | offset 4 | elements | `SetTilingData` | 有效列/tail copy | 等于 N |
| ARN | `block_factor` | uint32/32 | offset 8 | rows/block | `SetTilingData` | block→row | ceil 分核 |
| ARN | `row_factor` | uint32/32 | offset 12 | rows | `SetTilingData` | MergeN/MultiN | mode 相关 |
| ARN | `ub_factor` | uint32/32 | offset 16 | elements | `SetTilingData` | UB/chunk | mode 相关 |
| ARN | `epsilon` | float/32 | offset 20 | scalar | ACLNN attr→`SetTilingData` | RMS denominator | `(0,1)` |
| ARN | `avg_factor` | float/32 | offset 24 | scalar | `1.0F/numCol` | FP32 mean | N=0 时 0，但空 executor 不 launch |
| ARNKey | `ARNKeyTile` | uint32/32 | offset 0 | key | main selector | ARN router | FP16 10..14/BF16 30..33 |
| ARNKey | `ARNKeyTail` | uint32/32 | offset 4 | key | tail selector | ARN router | tail 存在才读 |
| ARNKey | `ARNNumBlocksTile` | uint32/32 | offset 8 | blocks | main selector | AIV guard | `<=2*aicCoreNum` |
| ARNKey | `ARNNumBlocksTail` | uint32/32 | offset 12 | blocks | tail selector | AIV guard | tail 存在才读 |
| Quant | `msg` | struct | offset 0，size 128，align 8 | bytes | HCCL tiling | communication base | 固定仓闭合 |
| Quant | `param` | struct | offset 128，size 128，align 8 | bytes | MC2 base tiling | QBMM/scheduler | 固定仓闭合 |
| Quant | `tilematmulTiling` | struct | offset 256，size `176+sizeof(TCubeTiling)` | bytes | main QBMM tiling | main MM | 总会生成 |
| Quant | `tailmatmulTiling` | struct | offset `432+sizeof(TCubeTiling)` | bytes | tail QBMM tiling | tail MM | tail 存在才消费 |
| Mc2Msg | `preparePosition` | uint32/32 | offset 0 | enum | `DoAllReduceTiling=1` | HCCL prepare | HCCL API 路线 |
| Mc2Msg | `sendOff` | uint64/64 | offset 8 | bytes | `CalcAllReduceSendRecvParams` | main send step | offset 4..7 为 padding |
| Mc2Msg | `recvOff` | uint64/64 | offset 16 | bytes | 同上 | main recv step | — |
| Mc2Msg | `tailSendOff` | uint64/64 | offset 24 | bytes | 同上 | tail send step | tail 条件 |
| Mc2Msg | `tailRecvOff` | uint64/64 | offset 32 | bytes | 同上 | tail recv step | tail 条件 |
| Mc2Msg | `sendCnt` | uint64/64 | offset 40 | elements | 同上 | main AllReduce | `tileM*N` |
| Mc2Msg | `recvCnt` | uint64/64 | offset 48 | elements | 同上 | main AllReduce | `tileM*N` |
| Mc2Msg | `tailSendCnt` | uint64/64 | offset 56 | elements | 同上 | tail AllReduce | `tailM*N` |
| Mc2Msg | `tailRecvCnt` | uint64/64 | offset 64 | elements | 同上 | tail AllReduce | `tailM*N` |
| Mc2Msg | `totalCnt` | uint64/64 | offset 72 | elements | `rankM*rankN` | total count | `M*N` |
| Mc2Msg | `turnNum` | uint32/32 | offset 80 | turns | `tileCnt+tailCnt` | KFC turns | — |
| Mc2Msg | `tailNum` | uint32/32 | offset 84 | turns | `tailCnt` | KFC tail | — |
| Mc2Msg | `stride` | uint32/32 | offset 88 | elements | `DoAllReduceTiling=0` | KFC | 当前为 0 |
| Mc2Msg | `workspaceOff` | uint32/32 | offset 92 | bytes | `libApiWorkSpaceSize_` | HCCL workspace | uint32 上界 |
| Mc2Msg | `notifyOff` | uint32/32 | offset 96 | bytes | `sizeof(KFCMsgBody)` | notify queue | — |
| Mc2Msg | `notifyBeginCnt` | uint16/16 | offset 100 | count | `NOTIFY_WRITE_CNT` | KFC | — |
| Mc2Msg | `notifyEndCnt` | uint16/16 | offset 102 | count | 常量 1 | KFC | — |
| Mc2Msg | `useBufferType` | uint8/8 | offset 104 | enum | `setUseBufferType` | `MatmulAllReduceBase::Init` | WINDOW_IN/OUTPUT |
| Mc2Msg | `funID` | uint8/8 | offset 105 | enum | `ALL_REDUCE_FUNC_ID` | HCCL | 固定 AllReduce |
| Mc2Msg | `dataType` | uint8/8 | offset 106 | enum | `GetDataType(geCType)` | HCCL | FP16/BF16 |
| Mc2Msg | `groupNum` | uint8/8 | offset 107 | groups | 常量 1 | HCCL | 单 group |
| Mc2Msg | `reuseMode` | uint8/8 | offset 108 | turns | `tileCnt+tailCnt` | KFC/window | uint8 容量受 Host guard |
| Mc2Msg | `commType` | uint8/8 | offset 109 | enum | `RCSTiling.commtype` | KFC | A2 comm mode |
| Mc2Msg | `reduceOp` | uint8/8 | offset 110 | enum | `RCSTiling.subtype` | HCCL | SUM |
| Mc2Msg | `commOrder` | uint8/8 | offset 111 | enum | 常量 1 | KFC | MM 后通信 |
| Mc2Msg | `waitPolicy` | uint8/8 | offset 112 | enum | 常量 1 | KFC wait | 每轮等待 |
| Mc2Msg | `rspPolicy` | uint8/8 | offset 113 | enum | 常量 1 | KFC response | 每轮响应 |
| Mc2Msg | `exitPolicy` | uint8/8 | offset 114 | enum | 常量 0 | KFC exit | 下发完退出 |
| Mc2Msg | `commAlg` | uint8/8 | offset 115 | enum | 常量 0 | HCCL | default |
| Mc2Msg | `taskType` | uint8/8 | offset 116 | enum | `KFC_TASK_HCC_TASK_DELIVER` | KFC | 参数取任务 |
| Mc2Msg | `debugMode` | uint8/8 | offset 117 | bitmask | base tiling | Host/Device debug route | 普通执行为非 debug |
| Mc2Msg | `stepSize` | uint8/8 | offset 118 | turns | base tiling | KFC | 当前路径无特殊步长 |
| Mc2Msg | `sendArgIndex` | uint8/8 | offset 119 | arg index | 常量 0 | KFC | x1 起点 |
| Mc2Msg | `recvArgIndex` | uint8/8 | offset 120 | arg index | IR inputs+outputs-1 | KFC | 由 OpDef 数量决定 |
| Mc2Msg | `commOutArgIndex` | uint8/8 | offset 121 | arg index | 本路径未见显式赋值 | 未见稳定消费契约 | 不得假定为 0 |
| Mc2Msg | `hasCommOut` | uint8/8 | offset 122 | bool | HCCL API 路线置 1 | KFC | — |
| Mc2Msg | `reserve` | uint8/8 | offset 123 | reserved | 未定义业务赋值 | 不应消费 | inactive 值非契约 |
| Mc2Msg | `reserve2` | uint32/32 | offset 124 | reserved | 未定义业务赋值 | 不应消费 | inactive 值非契约 |
| RCSTiling | `rankDim` | uint32/32 | offset 0 | ranks | group query | HCCL/MM | 文档限 1/2/4/8 |
| RCSTiling | `rankID` | uint32/32 | offset 4 | rank | group query | HCCL | `<rankDim` |
| RCSTiling | `commtype` | uint32/32 | offset 8 | enum | comm-mode tiling | Mc2Msg | A2 支持域 |
| RCSTiling | `subtype` | uint32/32 | offset 12 | enum | reduceOp parse | Mc2Msg | SUM |
| RCSTiling | `tileCnt` | uint32/32 | offset 16 | segments | `DoSplitM` | main loop/router | — |
| RCSTiling | `tailM` | uint32/32 | offset 20 | rows | `DoSplitM` | tail QBMM/ARN | tail 条件 |
| RCSTiling | `tailCnt` | uint32/32 | offset 24 | segments | `DoSplitM` | tail loop/router | 通常 0/非零 |
| RCSTiling | `biasLen` | uint32/32 | offset 28 | elements | bias shape | QBMM | optional bias |
| RCSTiling | `isAdd` | uint32/32 | offset 32 | bool | base attr | base postproc | 本融合的 residual add 在 ARN |
| RCSTiling | `rankM` | uint32/32 | offset 36 | rows | shape flatten | QBMM/HCCL | M≤INT32_MAX |
| RCSTiling | `rankN` | uint32/32 | offset 40 | cols | output shape | QBMM/HCCL | N≤65535 |
| RCSTiling | `rankK` | uint32/32 | offset 44 | cols | x1 last dim | QBMM | K≤65535，K=0 失败 |
| RCSTiling | `gatherIndex` | uint32/32 | offset 48 | index | base tiling | gather path | 本 AllReduce 主路不消费 |
| RCSTiling | `isTransposeA` | uint32/32 | offset 52 | bool | API util/Host | QBMM | 固定 0 |
| RCSTiling | `isTransposeB` | uint32/32 | offset 56 | bool | API util/Host | QBMM/key | 0→8，1→4104 |
| RCSTiling | `storageGather` | uint32/32 | offset 60 | enum | base tiling | gather path | inactive 非契约 |
| RCSTiling | `nd2NzWorkLen` | uint64/64 | offset 64 | bytes | base workspace tiling | ND2NZ path | 对应路径启用才消费 |
| RCSTiling | `cToFloatLen` | uint64/64 | offset 72 | bytes | base workspace tiling | cast path | 对应路径启用才消费 |
| RCSTiling | `gatherLen` | uint64/64 | offset 80 | bytes | base workspace tiling | gather path | 本路不应依赖 |
| RCSTiling | `workspaceAddr4` | uint32/32 | offset 88 | bytes | workspace tiling | MM helper | uint32 地址偏移 |
| RCSTiling | `aicCoreNum` | uint32/32 | offset 92 | cores | platform query | core map | blockDim 同源 |
| RCSTiling | `needUbBuffer` | uint32/32 | offset 96 | bytes | quant QBMM tiling | BF16 dequant | FP16 路径不同 |
| RCSTiling | `addX3UbCnt` | uint32/32 | offset 100 | elements | base tiling | base add helper | 本 ARN add 不依赖其语义 |
| RCSTiling | `commWorkSpaceSize` | uint32/32 | offset 104 | bytes | HCCL tiling | HCCL | — |
| RCSTiling | `isInputCommQuantScale` | uint32/32 | offset 108 | bool | comm quant route | buffer selector | 本入口为非 comm-int8 |
| RCSTiling | `dataType` | uint32/32 | offset 112 | enum | dtype route | MM/dequant | FP16/BF16 |
| RCSTiling | `commInt8WorkSpace` | uint32/32 | offset 116 | bytes | comm-int8 route | comm-int8 body | 本入口 inactive |
| RCSTiling | `dynamicQuantTempBuffSize` | uint32/32 | offset 120 | bytes | dynamic quant route | dynamic quant body | 本入口 inactive；struct size 128 |

QBMM 子 POD 也按字段闭合；下表偏移均相对各自子结构起点，main/tail 各有一份。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| QBMMParams | `batchA` | uint32/32 | offset 0 | count | QBMM tiling | batch map | 本 op 无额外 batch 语义 |
| QBMMParams | `batchB` | uint32/32 | offset 4 | count | QBMM tiling | batch map | 同上 |
| QBMMParams | `batchC` | uint32/32 | offset 8 | count | QBMM tiling | batch map | 同上 |
| QBMMParams | `batchA1` | uint32/32 | offset 12 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchA2` | uint32/32 | offset 16 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchA3` | uint32/32 | offset 20 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchA4` | uint32/32 | offset 24 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchB1` | uint32/32 | offset 28 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchB2` | uint32/32 | offset 32 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchB3` | uint32/32 | offset 36 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchB4` | uint32/32 | offset 40 | count | QBMM tiling | batch broadcast | 机械填充 |
| QBMMParams | `batchC1` | uint32/32 | offset 44 | count | QBMM tiling | batch output | 机械填充 |
| QBMMParams | `batchC2` | uint32/32 | offset 48 | count | QBMM tiling | batch output | 机械填充 |
| QBMMParams | `batchC3` | uint32/32 | offset 52 | count | QBMM tiling | batch output | 机械填充 |
| QBMMParams | `batchC4` | uint32/32 | offset 56 | count | QBMM tiling | batch output | 机械填充 |
| QBMMParams | `singleCoreBatch` | uint32/32 | offset 60 | batches/core | QBMM tiling | core loop | — |
| QBMMParams | `isPerTensor` | uint32/32 | offset 64 | bool | scale shape | FP16/BF16 dequant | 与 per-channel 互斥 |
| QBMMParams | `isPertoken` | uint32/32 | offset 68 | bool | quant API route | dequant body | 本入口固定 false |
| QBMMParams | `isDoubleScale` | uint32/32 | offset 72 | bool | QBMM tiling | scale path | 本入口不扩展双 scale 语义 |
| QBMMParams | `biasThreeDim` | uint32/32 | offset 76 | bool | bias shape | bias address | 本入口 bias 为 `[N]` |
| QBMMParams | `ubCalcM` | uint32/32 | offset 80 | rows | BF16 UB tiling | AIV dequant | FP16 不按此 Vector 路线消费 |
| QBMMParams | `ubCalcN` | uint32/32 | offset 84 | cols | BF16 UB tiling | AIV dequant | 通常从 baseN 开始 |
| QBMMParams | `needUbBuffer` | uint32/32 | offset 88 | bytes | BF16 UB tiling | UB init | `ubCalcM*ubCalcN` 派生 |
| QBMMParams | `realSingleCoreM` | uint32/32 | offset 92 | rows | workspace/core tiling | BF16 core loop | 1 GiB 分支可调整 |
| QBMMParams | `realSingleCoreN` | uint32/32 | offset 96 | cols | workspace/core tiling | BF16 core loop | — |
| QBMMParams | `biasDtype` | uint32/32 | offset 100 | enum | bias dtype | QBMM body | 本入口 INT32 |
| QBMMParams | `ubSize` | uint32/32 | offset 104 | bytes | platform query | BF16 UB init | 平台相关 |
| QBMMParams | `isMClash` | uint32/32 | offset 108 | bool | QBMM scheduler | core map | 机械调度字段 |
| QBMMParams | `isNClash` | uint32/32 | offset 112 | bool | QBMM scheduler | core map | 机械调度字段 |
| QBMMParams | `groupSizeM` | uint32/32 | offset 116 | blocks | QBMM scheduler | grouped core map | 机械调度字段 |
| QBMMParams | `groupSizeN` | uint32/32 | offset 120 | blocks | QBMM scheduler | grouped core map | 机械调度字段 |
| QBMMParams | `groupSizeK` | uint32/32 | offset 124 | blocks | QBMM scheduler | grouped core map | params size 128 |
| QBMML2 | `mTileCntL2` | uint32/32 | offset 0 | tiles | QBMM L2 tiling | L2 scheduler | — |
| QBMML2 | `nTileCntL2` | uint32/32 | offset 4 | tiles | QBMM L2 tiling | L2 scheduler | — |
| QBMML2 | `mTileBlock` | uint32/32 | offset 8 | blocks | QBMM L2 tiling | L2 scheduler | — |
| QBMML2 | `nTileBlock` | uint32/32 | offset 12 | blocks | QBMM L2 tiling | L2 scheduler | — |
| QBMML2 | `calOrder` | uint32/32 | offset 16 | enum | QBMM L2 tiling | L2 scheduler | — |
| QBMML2 | `isBasicTiling` | uint32/32 | offset 20 | bool | QBMM L2 tiling | L2 scheduler | struct size 24 |
| QBMMSliding | `mTailTile` | uint32/32 | offset 0 | tiles | QBMM scheduler | sliding loop | — |
| QBMMSliding | `nTailTile` | uint32/32 | offset 4 | tiles | QBMM scheduler | sliding loop | — |
| QBMMSliding | `mBaseTailSplitCnt` | uint32/32 | offset 8 | splits | QBMM scheduler | sliding loop | — |
| QBMMSliding | `nBaseTailSplitCnt` | uint32/32 | offset 12 | splits | QBMM scheduler | sliding loop | — |
| QBMMSliding | `mTailMain` | uint32/32 | offset 16 | rows | QBMM scheduler | sliding loop | — |
| QBMMSliding | `nTailMain` | uint32/32 | offset 20 | cols | QBMM scheduler | sliding loop | struct size 24 |

`Mc2QuantBatchMatmulV3TilingData` 的布局为 128 B params + 仓外 `kernel_tiling.h` 定义的 `TCubeTiling` + 24 B L2 参数 + 24 B sliding-window 参数，即 `176+sizeof(TCubeTiling)`。因此 `Q=608+2*sizeof(TCubeTiling)`，top 总长为 `Q+72` 且按 8 B 对齐；固定仓不能证明 `TCubeTiling` 的字段、绝对大小或绝对顶层偏移，不能伪造这些常数。

fused tiling 对顶层存储没有可证明的整体清零契约。main ARN 总会写；tail ARN 仅在 `tailCnt!=0` 时写，Kernel 也只应在该条件下读取。保留字段和未启用路径字段不得被当作稳定零值或公开 ABI。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 no-trans | key 8；ARN 10..14 | inplace alias + quant MC2 + common ARN | top POD；B_TRANS=0 | arch22 FP16/key8 | `BmmDequant` + HCCL base + 5 ARN bodies | ascend910b binary JSON + CMake |
| R-002 FP16 B-trans | key 4104；ARN 10..14 | 同上，B_TRANS=1 | top POD；B_TRANS=1 | arch22 FP16/key4104 | `BmmDequant<B_TRANS=true>` + 同上 | 同上 |
| R-003 BF16 no-trans | key 8；ARN 30..33 | 同上 + BF16 workspace/UB producer | top POD；isUbQuant；B_TRANS=0 | arch22 BF16/key8 | `BmmDequantBf16` + HCCL base + 4 ARN bodies | 同上 |
| R-004 BF16 B-trans | key 4104；ARN 30..33 | 同上，B_TRANS=1 | top POD；isUbQuant；B_TRANS=1 | arch22 BF16/key4104 | `BmmDequantBf16<B_TRANS=true>` + 同上 | 同上 |

## Kernel 实现

### 公共执行流程

entry 读取顶层 tiling，取得 user workspace 并构造 `TPipe`。四个模板实例均以 `KERNEL_TYPE_MIX_AIC_1_2` 编译。quant wrapper 的地址集合为：MatMul `{x1,x2,bias,null,normOut,workspace,normOut}`，quant `{null,null,dequantScale,null}`，ARN `{residual,gamma,y,normOut}`。因此 `normOut` 先是每段 QBMM/HCCL 的通信输出缓冲，HCCL 完成且 ARN 读入该段后才被覆盖为最终归一化结果；`y` 与 `residual` 是同一 GM 地址。

Device `Process` 先提交全部 main MatMul segment，再提交 tail segment；每段 AIC 结束后 `Mc2SyncAll`，由 AIV0 commit 对应 HCCL handle。全部 MatMul 提交结束后再次同步，AIV 才进入 ARN 阶段。ARN router 中仅 AIV block 0 调 `Query(handle)`，把累计完成数写到 workspace 起始处并执行 DCCI clean/invalidate；所有 AIV `SyncAll` 后只消费相对上次新增的连续 segment。main 计数消费完再消费 tail，指针步长是 `num_row*num_col*2` B。循环直到 `Query >= tileCnt+tailCnt`，随后全核同步，AIV0 finalize 通信。

因此有源码保证的是“后续 Cube segment 与此前 HCCL 可重叠，以及进入 ARN 阶段后可按 HCCL 完成进度增量消费”；由于 ARN 阶段位于全部 MatMul 提交之后，不能把它概括成始终成立的 Cube/HCCL/ARN 三者同时并行。

原位生命周期同样受 segment 约束：ARN 对一个 segment 先把 `normOut` 通信结果与原始 `residual` 搬入本地，再写 `y(==residual)`；Normal/MergeN/SingleN/MultiN 随后使用本地结果完成 norm，SplitD 第二遍则重读已经写出的 `y`。因此单段内部不会在读取前覆盖 residual，但调用完成后原 residual 不再存在，且调用方必须保持该地址有效直到 stream 完成。

### Route R-001：FP16、x2 不转置

#### 适用条件

输出/residual/gamma 为 FP16，dequantScale 为 UINT64/INT64 位模式，x2 不是最后两维转置 view；顶层 key 8。

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `mc2/matmul_all_reduce_add_rms_norm/op_api/matmul_all_reduce_arn_util.cpp` | `IsTransposeLastTwoDims` | x2 不构成末两维转置 view | `transposeX2=false` |
| Tiling | `mc2/matmul_all_reduce/op_host/op_tiling/arch22/quant_matmul_all_reduce_tiling.cpp` | `GetTilingKey` | A8W8、FP16、A_TRANS=0、B_TRANS=0 | key 8 |
| Router | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp` | entry FP16/key 8 branch | `TILING_KEY_IS(8)` | `BmmDequant<...,false,false>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> MatmulAllReduceAddRmsNormQuant::Process
      -> BmmDequant::Init/Process (AIC)
      -> MatmulAllReduceBase HCCL commit/wait
      -> AddRmsNormKernel::Process (AIV, ARN key 10..14)
      -> Mc2SyncAll/finalize
```

##### 源码等价伪码

```text
for segment in [main tiles, tail tiles]:
  AIC: int32_acc = MM_INT8(x1_segment, x2_KN, bias)
       fp16_segment = AscendDequant(int32_acc, uint64_scale)
       submit AllReduceSUM(fp16_segment)
sync after all MM segments have been submitted
AIV loop:
  completed = Query(HCCL handle)
  for each newly completed segment:
    y_segment = fp16(allreduce_segment + residual_segment)
    normOut_segment = RMSNorm_FP32(y_segment, gamma, epsilon)
sync all MIX cores; finalize HCCL
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲 | offset/地址 | extent | dtype/layout | producer | consumer | 所有权/生命周期 | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 GM | `mOffset*K` elements | segmentM×K | INT8 ND | caller | L1/L0A/Cube | stream 完成前由 caller 保持 | QBMM 内部按 tile 对齐 |
| x2 GM | `nOffset+kOffset*N` | Ktile×Ntile | INT8 K×N view | caller | L1/L0B/Cube | 全调用期只读 | 无显式 transpose buffer |
| bias GM | N tile offset | Ntile | INT32 ND | caller | Cube/Fixpipe | 全调用期只读 | per-channel |
| dequantScale GM | 标量或 N tile | 1 或 N | UINT64 位模式 | caller | Matmul `SetQuantScalar/Vector` | 当前 MM segment | scalar/vector 两路 |
| `normOut` 临时段 | segment index×`segmentM*N*2` B | segmentM×N | FP16 ND | `BmmDequant`，随后 HCCL | ARN x1 | HCCL 完成且 ARN 读入前禁止覆盖 | segment 完成后原址写最终 norm |
| residual/y GM | 同一地址 | segmentM×N | FP16 ND | caller residual / ARN y | ARN load / caller output | 先读 residual 后原位写 y；stream 完成前有效 | 不可跨 segment 提前复用 |
| gamma GM | 0..N | N | FP16 ND | caller | ARN Vector | 全调用期只读 | 逐列广播 |
| completion word | user workspace 起始状态区 | 8 B slot | int64 存储槽，值域来自 uint32 count | AIV0 `Query`+DCCI | 全部 AIV | ARN 循环期 | `SyncAll` 后可见 |
| ARN UB queues | 由 sub-key 分配 | 当前 row/chunk group | FP16/FP32 | GM copy-in | Vector compute/copy-out | 当前 ARN batch | queue/event 后复用 |

##### 分层内存布局

```text
x1/x2 GM -> L1 -> L0A/L0B -> L0C(INT32) -> dequant(FP16 GM)
                                             -> HCCL SUM
residual/gamma GM ----------------------------> AIV UB -> y/normOut GM
```

##### 字节账、交接边与复用顺序

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| QBMM/HCCL segment | `BmmDequant`/HCCL | segment×`segmentM*N*2` B | FP16 ND | ARN | handle Query+DCCI+SyncAll | `normOut` 同段 | handle 完成且 ARN 已读入 |
| ARN Normal UB | GM copy-in | `14*ubFactor+256` B | FP16/FP32 | Vector/copy-out | TQue/event | 当前行 | copy-out 完成 |
| ARN SplitD UB | first pass | `16*ubFactor+36*rowFactor+256` B | FP16/FP32 | second pass | pass 顺序 | 当前 chunk/行 | y/sum 完成后方可第二遍 |
| ARN MergeN UB | GM copy-in | `18*ubFactor+260*rowFactor` B | FP16/FP32 | Vector/copy-out | TQue/event | 当前行组 | 合并行组完成 |
| ARN SingleN UB | GM copy-in | 195584 B | FP16/FP32 | Vector/copy-out | TQue/event | 单行 | 单行完成 |
| ARN MultiN UB | GM copy-in | `16*ubFactor+2*alignN+64*rowFactor+256` B | FP16/FP32 | Vector/copy-out | 双缓冲队列 | 当前多行 batch | batch 完成 |

这里的 UB 账是显式 queue/tmp/reduce buffer 之和，不含 Matmul API 内部 L1/L0 配额。通信 GM 片段必须等 HCCL handle 完成并被 ARN 消费后才能覆盖为最终 `normOut`。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cube→HCCL | handle register、`Mc2SyncAll`、AIV0 commit | `BmmDequant` AIC | AICPU KFC/HCCL | FP16 segment | INT32 accum→UINT64 scale→FP16；tail 独立 QBMM |
| HCCL→ARN | AIV0 Query、workspace count、DCCI、`SyncAll` | HCCL | 全部 AIV ARN | `normOut` 临时段 | FP16 SUM；只消费新增完成段，tail 独立 ARN |
| add→RMS | TQue/event 与 Vector pipeline | residual+reduced | RMS reduce/copy-out | y 与 ARN UB | FP16 add→FP32 square/mean/rsqrt；N tail 由有效列限定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 事实 |
|---|---|---|---|
| entry | `.../op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp` | key 8 FP16 branch | no-trans 模板实例 |
| MatMul | `.../op_kernel/mm_allreduce_add_rms_norm_quant.h` | `BmmDequant` | AIC INT8 MM 与 FP16 dequant |
| ARN | `add_rms_norm_kernel.h` + mode headers | `AddRmsNormKernel` | key 10–14 路由 |

### Route R-002：FP16、x2 转置

#### 适用条件

与 R-001 相同，但 `IsTransposeLastTwoDims(x2)` 为 true；顶层 key 4104。

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `mc2/matmul_all_reduce_add_rms_norm/op_api/matmul_all_reduce_arn_util.cpp` | `IsTransposeLastTwoDims` | x2 构成末两维转置 view | `transposeX2=true` |
| Tiling | `mc2/matmul_all_reduce/op_host/op_tiling/arch22/quant_matmul_all_reduce_tiling.cpp` | `GetTilingKey` | A8W8、FP16、A_TRANS=0、B_TRANS=1 | key 4104 |
| Router | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp` | entry FP16/key 4104 branch | `TILING_KEY_IS(4104)` | `BmmDequant<...,false,true>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> FP16 B-trans BmmDequant -> HCCL -> ARN key 10..14 -> sync/finalize
```

##### 源码等价伪码

```text
same as R-001, except x2 tile base = nOffset*K + kOffset
and Matmul template parameter B_TRANS=true
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲 | offset/地址 | extent | dtype/layout | producer | consumer | 所有权/生命周期 | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x2 trans GM | `nOffset*K+kOffset` | Ntile×Ktile | INT8 N×K 物理 view、逻辑 K×N | caller | L1/L0B/Cube | 全调用期只读 | `B_TRANS=true`，无临时转置 |
| x1/bias/scale | 同 R-001 | 同 R-001 | 同 R-001 | caller | QBMM | 同 R-001 | 同 R-001 |
| `normOut` 临时段 | segment×`segmentM*N*2` B | segmentM×N | FP16 ND | B-trans QBMM→HCCL | ARN | HCCL+ARN handoff 前不可覆盖 | 最终复用为 normOut |
| residual/y、gamma、ARN UB | 同 R-001 | 同 R-001 | FP16/FP32 | caller/ARN | ARN/caller | 同 R-001 | 同 R-001 |

##### 分层内存布局

```text
x2[N,K] GM --B_TRANS--> L1/L0B --MM--> L0C -> FP16 dequant -> HCCL -> ARN
```

##### 字节账、交接边与复用顺序

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| B-trans GM→L0B | caller/Matmul copy | Ktile×Ntile B | INT8 N×K view | Cube | Matmul pipeline | 当前 K/N tile | B_TRANS 地址公式一致 |
| QBMM→HCCL→ARN | B-trans QBMM/HCCL | `segmentM*N*2` B | FP16 ND | ARN | Query/DCCI/SyncAll | `normOut` 同段 | HCCL 完成后才读/覆写 |
| ARN UB | GM copy-in | 与 R-001 相同 | FP16/FP32 | Vector | TQue/event | 当前 sub-key batch | batch 完成后复用 |

差异仅是 B 矩阵地址公式和 Matmul 模板布局；转置是 view 语义，不生成显式 transpose workspace。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| B 搬运→Cube | Matmul API 内部 pipeline | B-trans GM copy | Mmad | L1/L0B tile | B-trans 地址；K/N tail 由 QBMM tiling 限定 |
| Cube→HCCL→ARN | 同 R-001 | B-trans `BmmDequant` | HCCL/ARN | segment 与原位 y | FP16；main/tail 同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 事实 |
|---|---|---|---|
| selector | quant MC2 tiling | key 4104 | B-trans specialization |
| entry | arch22 entry | FP16/4104 branch | 模板末参数为 true |
| body | quant wrapper | B offset branch | `nOffset*K+kOffset` |

### Route R-003：BF16、x2 不转置

#### 适用条件

输出/residual/gamma/dequantScale 为 BF16，x2 不转置；顶层 key 8。BF16 不可使用 ARN MultiN。

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `mc2/matmul_all_reduce_add_rms_norm/op_api/aclnn_quant_matmul_all_reduce_add_rms_norm.cpp` | dtype check | output/residual/gamma/dequantScale 为 BF16 | BF16 inner op |
| Tiling | `mc2/matmul_all_reduce/op_host/op_tiling/arch22/quant_matmul_all_reduce_tiling.cpp` | QBMM/workspace tiling | A8W8、BF16、B_TRANS=0 | key 8；INT32 workspace |
| Router | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp` | entry BF16/key 8 branch | `TILING_KEY_IS(8)` | `BmmDequantBf16<...,false,false,true>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> BmmDequantBf16::Init/Process
      -> registered Matmul object produces INT32 VECIN/workspace
      -> AIV AscendDequant to BF16 GM
      -> HCCL AllReduce
      -> ARN key 30..33
      -> sync/finalize
```

##### 源码等价伪码

```text
effectiveAicIdx = blockIdx / MIX_RATIO
if subBlockIdx > 0 on duplicated AIC task: return from MM producer path
int32Tile = MM_INT8(x1, x2, bias)
bf16Tile = AscendDequant(int32Tile, bf16Scale)
submit AllReduceSUM(bf16Tile)
wait completion
sumFP32 = fp32(reduced) + fp32(residual)
y = RINT_BF16(sumFP32)
normOut = RINT_BF16(RMSNorm_FP32(fp32(y), gamma, epsilon))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲 | offset/地址 | extent | dtype/layout | producer | consumer | 所有权/生命周期 | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 GM | `mOffset*K` | segmentM×K | INT8 ND | caller | Cube | stream 完成前 | QBMM tile 对齐 |
| x2 GM | `nOffset+kOffset*N` | Ktile×Ntile | INT8 K×N | caller | Cube | 全调用期只读 | no-trans |
| bias/scale GM | N tile / 标量或 N tile | N / 1或N | INT32 / BF16 | caller | Cube / dequant AIV | 当前 segment | per-tensor/per-channel |
| BF16 MM scratch | `blockIdx*singleOffset` | `singleOffset=max(singleTimeM,baseM)*max(singleTimeN,baseN)*4` B/有效核 | INT32 ND | 注册 Matmul/Cube | AIV `AscendDequant` | 当前 QBMM tile dequant 完成前 | 各有效核不重叠 |
| BF16 dequant UB | `ubCalcM*ubCalcN` 元素相关队列 | `needUbBuffer` | INT32/BF16/FP32 scale | GM/workspace | Vector dequant | 当前 UB tile | 由 `ubCalcM/N` 约束 |
| `normOut` 临时段 | segment×`segmentM*N*2` B | segmentM×N | BF16 ND | AIV dequant，随后 HCCL | ARN | HCCL 完成并读入前 | 最终覆写 normOut |
| residual/y GM | 同一地址 | segmentM×N | BF16 ND | caller / ARN | ARN / caller | FP32 add 后 RINT 写 y | stream 完成前不复用 |
| ARN UB | sub-key 分配 | 当前 row/chunk | BF16/FP32 | GM copy-in | ARN Vector | 当前 batch | BF16 无 MultiN |

##### 分层内存布局

```text
INT8 GM -> Cube -> INT32 workspace/VECIN -> AIV dequant BF16 GM
         -> HCCL SUM -> AIV FP32 add/RINT BF16 -> FP32 RMS -> BF16 outputs
```

##### 字节账、交接边与复用顺序

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| 每核 Cube→AIV dequant | registered Matmul | `blockIdx*singleOffset`；每核 `singleOffset` B | INT32 ND | AIV dequant | Matmul 协议 | user workspace 分片 | 对应 tile dequant 结束 |
| BF16 dequant UB | INT32/scale copy-in | `needUbBuffer`；单位列 `2*(4+2)+UB_EXTRE_BYTE` B，另含 scale queue | INT32/BF16 | `AscendDequant`/copy-out | Vector queue | 当前 UB tile | `ubCalcM*ubCalcN` 完成 |
| BF16 segment→HCCL→ARN | AIV dequant/HCCL | `segmentM*N*2` B | BF16 ND | ARN | Query/DCCI/SyncAll | `normOut` 同段 | handle 完成且 ARN 已读入 |
| ARN UB | GM copy-in | R-001 的 Normal/SplitD/MergeN/SingleN 公式 | BF16/FP32 | Vector/copy-out | TQue/event | 当前 ARN batch | BF16 禁止 MultiN |
| 融合 ARN 增量 | fused Host | MC2 workspace 尾部 256 B | byte buffer | ARN state | 全调用期 | caller workspace | stream 完成后释放 |

BF16 workspace 总额由 QBMM Host 根据实际 `singleCoreM/N`、`baseM/N` 与 `usedCoreNum` 得出，并受 1 GiB 分支影响；不能写成固定的 `max(tileM,tailM)*N*4`。INT32 分片必须等对应 AIV dequant 完成后才能复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cube→dequant AIV | 注册 MM 对象与 Matmul 内部同步 | AIC INT32 | AIV `AscendDequant` | per-core scratch | INT32→BF16；tail 独立 QBMM |
| dequant→HCCL→ARN | commit、Query、DCCI、`SyncAll` | AIV BF16/HCCL | ARN AIV | BF16 segment | BF16 SUM；tail 独立 ARN key |
| add→norm | Vector pipe、RINT | reduced+residual | y write/RMS/copy-out | 原位 y 与 UB | FP32 add→RINT BF16 y→FP32 RMS→RINT BF16 norm；有效 N |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 事实 |
|---|---|---|---|
| Host | quant MC2 tiling | BF16 workspace branch | 预留 INT32 MM 输出 |
| entry | arch22 entry | BF16/key8 branch | `BmmDequantBf16` no-trans |
| ARN | `add_rms_norm.h` 等 | BF16 path | FP32 add + RINT BF16；key 30–33 |

### Route R-004：BF16、x2 转置

#### 适用条件

与 R-003 相同，但 x2 为最后两维转置 view；顶层 key 4104。

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `mc2/matmul_all_reduce_add_rms_norm/op_api/matmul_all_reduce_arn_util.cpp` | `IsTransposeLastTwoDims` | BF16 且 x2 trans view | `transposeX2=true` |
| Tiling | `mc2/matmul_all_reduce/op_host/op_tiling/arch22/quant_matmul_all_reduce_tiling.cpp` | `GetTilingKey` | A8W8、BF16、B_TRANS=1 | key 4104 |
| Router | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp` | entry BF16/key 4104 branch | `TILING_KEY_IS(4104)` | `BmmDequantBf16<...,false,true,true>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> BF16 B-trans registered MM -> AIV dequant -> HCCL -> ARN 30..33
```

##### 源码等价伪码

```text
same as R-003, except x2 tile base = nOffset*K+kOffset
and the BF16 Matmul B_TRANS template parameter is true
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲 | offset/地址 | extent | dtype/layout | producer | consumer | 所有权/生命周期 | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x2 trans GM | `nOffset*K+kOffset` | Ntile×Ktile | INT8 N×K 物理 view | caller | registered BF16 MM | 全调用期只读 | `B_TRANS=true` |
| INT32 per-core scratch | `blockIdx*singleOffset` | 同 R-003 | INT32 ND | Cube | AIV dequant | 对应 tile dequant 前 | 同 R-003 |
| `normOut` 临时段 | segment×`segmentM*N*2` B | segmentM×N | BF16 ND | AIV dequant→HCCL | ARN | HCCL+ARN handoff 前 | 最终覆写 normOut |
| residual/y、ARN UB | 同 R-003 | 同 R-003 | BF16/FP32 | caller/ARN | ARN/caller | 同 R-003 | 同 R-003 |

##### 分层内存布局

```text
x2[N,K] B_TRANS + x1 -> INT32 -> BF16 dequant -> HCCL -> BF16 ARN
```

##### 字节账、交接边与复用顺序

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| B-trans GM→Cube | caller/Matmul copy | Ktile×Ntile B | INT8 N×K view | Cube | Matmul pipeline | 当前 K/N tile | `B_TRANS=true`，无临时转置 |
| Cube→AIV dequant | registered Matmul | 每有效核 `singleOffset` B | INT32 ND | AIV | Matmul 协议 | user workspace 分片 | dequant 完成后复用 |
| BF16 communication/ARN | AIV dequant/HCCL | `segmentM*N*2` B + ARN UB | BF16/FP32 | ARN/caller | Query/DCCI/SyncAll | `normOut`/UB | handle 完成、ARN 消费、最终写出 |

其余字节账同 R-003；转置不增加独立 transpose buffer。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| B-trans MM→AIV dequant | registered Matmul 协议 | AIC INT32 | AIV dequant | per-core scratch | INT32→BF16；QBMM tail 独立 |
| HCCL→ARN | Query/DCCI/`SyncAll` | HCCL | ARN AIV | BF16 segment | ARN tail tiling/key 独立 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号 | 事实 |
|---|---|---|---|
| selector | quant MC2 tiling | key 4104 | BF16 B-trans |
| entry | arch22 entry | BF16/4104 branch | B_TRANS=true |
| body | `BmmDequantBf16` | trans offset | `nOffset*K+kOffset` |

## 算子特定优化

### Trick T-001：API 到 Device 的 residual/y 原位别名

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003/R-004，仅公开 Inplace 入口。
- 触发条件：调用方接受 `residual` 被原位覆盖，且 Host/Device 均保留 residual/y 双槽同址。
- Golden 来源：[`aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.cpp:L38-L45`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/inplace_matmul_all_reduce_add_rms_norm/op_api/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.cpp#L38-L45)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
return aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    x1, x2, bias, dequantScale, residual, gamma, epsilon,
    group, reduceOp, commTurn, streamMode, residual, normOut,
    workspaceSize, executor);
```

Device 地址集合进一步固定为 `ArnGmAddrs {residualGM, gammaGM, yGM, normOutGM}`，而 entry 的 `yGM` 来自同一别名输出。

#### 机制与设计原因

ARN 每个 segment 先把 residual 与已完成的 AllReduce 结果搬进本地，再写 `y==residual`；这让同一 GM 存储承担输入 residual 和最终 y，不需要额外 y 地址。`normOut` 保持独立，因为它在前半程还承担 QBMM/HCCL 临时输出，后半程才被 ARN 覆盖。

#### 迁移边界

- 适用：同语义的原位 AddRMSNorm 融合入口。
- 前提：保留 Host 双槽别名、Device residual/y 两个形参与 ARN 先读后写；调用方持有存储至 stream 完成。
- 失效：非 inplace API，或任何会在 ARN copy-in 前覆盖 residual 的调度；SplitD 第二遍读取的是已写出的 y。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要把该契约移植到非 inplace API；不要在 ARN 读 residual 之前写 y；不要假设调用后还能读取原 residual。

### Trick T-002：HCCL 完成计数驱动的 segment 增量消费

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003/R-004。
- 触发条件：main/tail split、HCCL handle 注册、MIX 1:2 和 workspace guard 均成功。
- Golden 来源：[`mm_allreduce_add_rms_norm_quant.h:L47-L66`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/mm_allreduce_add_rms_norm_quant.h#L47-L66)、[`add_rms_norm_kernel.h:L61-L95`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_kernel.h#L61-L95)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
const uint32_t curCnt = GetCurFinishedCnt();
*rcvCntGM = curCnt;
SyncAll();
lastCnt = *rcvCntGM;
```

#### 机制与设计原因

AIV0 是完成状态发布者：Query 后写 workspace、DCCI clean/invalidate，再通过 `SyncAll` 让其他 AIV 看见同一累计值。router 维护 `addRmsNormCount`，只对 `(oldCount,newCount]` 做 ARN；main 和 tail 用各自 tiling/key，GM 指针按 segment 字节跨度推进。此前的 HCCL 可以与后续 Cube segment 重叠，ARN 阶段又不必等待全部 HCCL segment 一次性结束。

#### 迁移边界

- 适用：分段 Cube→HCCL→Vector consumer 且 HCCL 提供累计完成计数的实现。
- 前提：单发布者、DCCI、AIV 全核同步、main-before-tail 计数解释与最终 `Finalize` 均保留。
- 失效：完成状态不是单调累计值，或 consumer 与 producer 使用不同 segment 次序；源码不证明 ARN 与 Cube 始终同时运行。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要在 handle 完成前读取或覆盖 `normOut` segment；不要让所有 AIV 同时 Query/写计数；不要删除 MM→ARN 的 `Mc2SyncAll` 或最终 finalize。

### Trick T-003：按 N、每核行数和 dtype 路由 ARN

#### 触发与源码依据

- 适用 Route：R-001/R-002 的 10..14，以及 R-003/R-004 的 30..33。
- 触发条件：main/tail 各自完成 `numRow/numCol/blockFactor` 与真实 UB 容量计算。
- Golden 来源：[`common_add_rms_norm_tiling.cpp:L222-L251`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/common_add_rms_norm_tiling.cpp#L222-L251)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (numCol > ubFactor) modeKey = K_SPLIT_D;
else if (blockFactor == 1) modeKey = K_SINGLE_N;
else if (numColAlign <= SMALL_REDUCE_NUM) modeKey = K_MERGE_N;
else if (dataType == ge::DT_FLOAT16 && numCol == numColAlign)
    modeKey = K_MULTI_N;
```

#### 机制与设计原因

SplitD 把 `N>12288` 的行按列切块并采用两遍流程；SingleN 对一核一行使用固定 UB 组织；MergeN 把小 N 的多行合并；MultiN 仅对 16 对齐 FP16 行启用；其余走 Normal。selector 同时从真实 UB 容量反推 `rowFactor/ubFactor`，并把 dtype prefix 与 mode 合成为 Device key。

#### 迁移边界

- 适用：具有相同 ARN bodies、UB 公式与 16 元素对齐规则的实现。
- 前提：保持 selector 顺序；`blockFactor==1` 优先于 small-N MergeN；main/tail 独立计算。
- 失效：BF16 请求 MultiN，或 FP16 MultiN `rowFactor==0`；前者改走 30，后者回退 key 10。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要只按 N 选路；不要给 BF16 下发 MultiN；不要忽略 16 元素对齐、有效列 tail 或 UB 容量公式。

### Trick T-004：通信 window 容量与确定性联合约束缓冲归属

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003/R-004 的 A2 普通复用模式。
- 触发条件：非 K-zero、非 AICPU-only debug、`reuseMode!=0`，并能读取默认或环境配置的 window 容量。
- Golden 来源：[`matmul_all_reduce_tiling_base.cpp:L204-L237`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp#L204-L237)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (UINT64_MAX - tileSendOff < tailSendOff ||
    tileSendOff + tailSendOff >= maxWindowSize)
    buffer_type = MC2_BUFFER_TYPE_OUTPUT;
else
    buffer_type = MC2_BUFFER_TYPE_WINDOW_IN;
```

#### 机制与设计原因

Host 依据所有 main/tail segment 的累计跨度判断通信 window 是否容纳，写入 `useBufferType`。Device 只有在该字段为 `WINDOW_IN` 且 HCCL context 的 `determinism != 1` 时，才把 `cGM` 改到本 rank window；确定性模式继续使用原 `normOut` 输出地址。该双重条件维持了 Host 容量规划和 Device 运行时确定性策略的一致性。

#### 迁移边界

- 适用：同样以累计 segment 跨度复用 HCCL window 的 A2 MC2 实现。
- 前提：重新闭合 Host 阈值、`Mc2Msg.useBufferType` 与 Device `Init`；200 MiB 仅是当前默认值。
- 失效：debug、K-zero、`reuseMode==0`、非 910B 或确定性 context；转用 OUTPUT 或禁止 Device 重定向。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要把判断简化成单个 segment 字节数；不要忽略加法溢出；不要在确定性模式下无条件重定向 `cGM` 到 window。

## 模板与实现对比

| 对比维度 | R-001/R-002 FP16 | R-003/R-004 BF16 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | A8W8、FP16 output、scale 为 UINT64/INT64 位模式；B view 决定 8/4104 | A8W8、BF16 output/scale；B view 决定 8/4104 | API dtype/transpose 检查与量化 tiling key |
| 执行族/节点图 | AIC Matmul/Fixpipe dequant + HCCL + AIV ARN | AIC Matmul→INT32 workspace + AIV BF16 dequant + HCCL + AIV ARN | BF16 需要显式 Vector 反量化 |
| M/N/K、tile 与 scheduler | main/tail split-M；B no-trans/trans 地址分支 | 同样 split-M；B 地址分支相同，额外 realSingleCoreM/N | 公共 MC2 scheduler + 两份 QBMM tiling |
| workspace/L1/L0/UB | Matmul API 内部 L1/L0；ARN 5 个 mode | per-core INT32 workspace + dequant UB；ARN 4 个 mode | dtype specialization 的中间表示不同 |
| AIC/AIV 或 kernel 交接 | AIC 直出 FP16 segment，HCCL 后交 AIV | AIC INT32 先交 AIV 反量化，再提交 HCCL，随后 ARN | MIX 1:2 与 KFC 完成计数 |
| 同步、精度与 tail | INT32→FP16；FP16 add→FP32 RMS；main/tail key 独立 | INT32→BF16；FP32 add→RINT y→FP32 RMS→RINT norm；main/tail独立 | 固定 Device body 的 cast/round 顺序 |
| fallback/失败边界 | MultiN 容量不足回 Normal；其他 tiling guard 失败 | 无 MultiN；1 GiB workspace 分支可重切，仍失败则终止 | Device 不猜测 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持（公开文档称 A2；配置为 Ascend 910B） | 不支持 | Ascend 950 支持表明确不支持 |
| Host/Launcher | arch22 inplace alias + quant/ARN；AICPU KFC attached stream | 无入口闭包 | 无入口闭包 |
| Kernel/资源 | arch22 `KERNEL_TYPE_MIX_AIC_1_2`，Cube+Vector+HCCL | 无 | 无 |
| 正式卡结论 | 本文件为无后缀 A2 卡，CANN 9.1 固定仓最新支持版本 | 不发布 | 不发布 `_950` 卡 |

## 限制与失败边界

- 接口已废弃；固定文档建议拆分算子，但这不改变当前 commit 的语义。
- 仅 A2，且只支持公开文档列出的 1/2/4/8 卡 HCCS 全互联同通信域场景。
- x1 不能转置；x2 仅支持可被 `IsTransposeLastTwoDims` 识别的最后两维转置 view。
- x1/x2 必须 INT8，bias 必须 INT32；FP16 输出的 scale 必须 UINT64/INT64，BF16 输出的 scale 必须 BF16。
- antiQuant、antiQuantOffset、perTokenScale 不属于此入口；不同量化语义必须另建逻辑算子卡。
- `streamMode` 只能为 1；非法 group、rank、comm mode、shape、dtype、workspace 或 tiling 均失败，不存在 Device 猜测 fallback。
- `K=0` 失败；其他必需输入出现空域时仅走 empty executor，不能读取 workspace。
- FP16 与 BF16 的 add/cast/round 顺序不同；尤其 BF16 先 FP32 add、RINT 写 y，再参与 RMSNorm，不能任意重排。
- `TCubeTiling` 的绝对 ABI 来自仓外头文件，固定仓只能证明符号组合公式；跨版本复用 POD 必须重新编译。
- 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 支持范围 | `mc2/inplace_matmul_all_reduce_add_rms_norm/README.md` | 支持表 | A2 √、A3 × |
| 公开约束 | `mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md` | ACLNN 文档 | 签名、dtype/shape、空张量、950/A3/A2 与废弃说明 |
| 入口声明 | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_api/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h` | 两个 ACLNN 符号 | 两段式公开 API |
| 入口实现 | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_api/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.cpp` | inplace GetWorkspaceSize/Execute | residual/y alias 与执行委托 |
| inner API | `mc2/matmul_all_reduce_add_rms_norm/op_api/aclnn_quant_matmul_all_reduce_add_rms_norm.cpp` | quant GetWorkspaceSize/Execute | null/dtype/shape/empty/streamMode/error |
| API util | `mc2/matmul_all_reduce_add_rms_norm/op_api/matmul_all_reduce_arn_util.cpp` | `MatmulAllReduceArnUtil` | transposeX1=false、推导 transposeX2、禁用 antiquant/pertoken |
| OpDef | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_host/inplace_matmul_all_reduce_add_rms_norm_def.cpp` | `InplaceQuantMatmulAllReduceAddRmsNorm` | IO、IgnoreContiguous、ascend910b、HCCL |
| inplace tiling | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/inplace_quant_matmul_all_reduce_add_rms_norm_tiling.cpp` | `using InplaceQuant...` | alias 到 quant ARN tiling |
| fused tiling | `mc2/matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/quant_matmul_all_reduce_add_rms_norm_tiling.cpp` | `DoOpTiling/PostTiling/GetWorkspaceSize` | main/tail ARN、256 B 增量、blockDim/key |
| ARN selector | `mc2/matmul_all_reduce_add_rms_norm/op_host/op_tiling/arch22/common_add_rms_norm_tiling.cpp` | `CommonAddRmsNormTiling::DoTiling` | key 10..14/30..33 与 UB 公式 |
| quant MC2 tiling | `mc2/matmul_all_reduce/op_host/op_tiling/arch22/quant_matmul_all_reduce_tiling.cpp` | `QuantMatmulAllReduceTiling` | A8W8、key 8/4104、QBMM、BF16 workspace |
| MC2 base tiling | `mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp` | `DoSplitM/DoAllReduceTiling/CalcBufferTypeByWindowSize` | M/K/N、split、HCCL、window |
| top POD | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h` | `QuantMatmulAllReduceAddRmsNormTilingData` | quant MC2 + 2×ARN + key |
| quant POD | `mc2/matmul_all_reduce/op_kernel/arch22/quant_matmul_all_reduce_tiling_data.h` | `QuantMatmulAllReduceTilingData` | Mc2Msg+RCS+2×QBMM |
| comm POD | `mc2/common/op_host/op_tiling/mc2_tiling_struct.h` | `Mc2Msg` / `RCSTiling` | Host POD 注册、字段宽度与次序 |
| comm Device POD | `mc2/common/op_kernel/mc2_tiling_struct.h` | `Mc2Msg` / `RCSTiling` | Device C struct、sizeof/offset 闭合 |
| QBMM POD | `mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_data.h` | `Mc2QuantBatchMatmulV3TilingData` | params/TCube/L2/sliding 组合 |
| QBMM cache tiling | `mc2/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_cache.cpp` | cache tiling producer | M/N/K、base/single-core/depth/step/L1/L0 字段 |
| Kernel entry | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_kernel/arch22/inplace_matmul_all_reduce_add_rms_norm.cpp` | Device entry | 显式 ABI、MIX 1:2、FP16/BF16、key 8/4104 |
| fused body | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/mm_allreduce_add_rms_norm_quant.h` | `MatmulAllReduceAddRmsNormQuantBF16::Process` | 地址绑定、MM→ARN 阶段与 finalize |
| comm base | `mc2/matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_base.h` | `Init/InnerProcess/PostProcEachTurn` | HCCL init/register/commit、确定性 buffer |
| quant wrapper | `mc2/matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_quant.h` | quant MC2 template | QBMM body 接入 MC2 segment |
| FP16 QBMM | `mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h` | `BmmDequant` | AIC Matmul/Fixpipe、scale scalar/vector、B offset |
| BF16 QBMM | `mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h` | `BmmDequantBf16` | INT32 workspace、AIV AscendDequant、RINT BF16 |
| ARN router | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_kernel.h` | `ComputeAddRmsNorm` | Query count、DCCI、main/tail 分派 |
| ARN Normal | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm.h` | `KernelAddRmsNorm` | key 10/30、add/cast/RMS |
| ARN SplitD | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_split_d.h` | `KernelAddRmsNormSplitD` | key 11/31 两遍按列处理 |
| ARN MergeN | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_merge_n.h` | `KernelAddRmsNormMergeN` | key 12/32 多行合并 |
| ARN SingleN | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_single_n.h` | `KernelAddRmsNormSingleN` | key 13/33 单行路线 |
| ARN MultiN | `mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_multi_n.h` | `KernelAddRmsNormMultiN` | FP16 key 14，多行双缓冲 |
| binary config | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_host/config/ascend910b/inplace_matmul_all_reduce_add_rms_norm_binary.json` | dtype 实例 | A8W8 FP16/BF16 ascend910b |
| task callback | `mc2/inplace_matmul_all_reduce_add_rms_norm/op_graph/inplace_matmul_all_reduce_add_rms_norm_gen_task.cpp` | calc-param/gen-task callback | attached stream 名称与公共 MatmulAllReduce helper |
| task graph | `mc2/common/op_graph/matmul_all_reduce_gen_task_ops_utils.cpp` | MatmulAllReduce gen task | Hcom wait/record 与公共 KFC graph |
| KFC graph | `mc2/common/op_graph/mc2_gen_task_ops_utils.cpp` | `CommonKFCMc2GenTask` | AICPU server、隐藏 HCOM context、失败传播 |
| build | `mc2/inplace_matmul_all_reduce_add_rms_norm/CMakeLists.txt` | target/source wiring | mc2/3rd/common/shared 依赖、910B auto-sync off |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| inplace wrapper | residual 同时作为 residual/y | 非 inplace 不得继承别名 | inplace GetWorkspaceSize |
| transpose | x1 固定 false；x2 view 决定 key 8/4104 | 不得显式搬运后仍沿用原 key | `IsTransposeLastTwoDims` |
| workspace | MC2 基础 + BF16/QBMM + ARN 256 B | offset、容量和 producer/consumer lifetime 一致 | `DoWorkspaceTiling` |
| POD | ARN 28 B、key 16 B、Mc2/RCS 各 128 B | `TCubeTiling` 绝对布局未知 | tiling data headers |
| communication | AllReduce SUM、commOrder=1、completion count | wait 前不得复用 segment | `MatmulAllReduceBase` / ARN router |
| precision | FP16 与 BF16 specialization | 不得重排 dequant/add/RINT/RMS | `BmmDequant*` / ARN modes |
| tail | QBMM 与 ARN 均独立 tiling | main key 不能替代 tail key | fused `DoOpTiling` / Process |
| platform | ascend910b/arch22 | 不得套用其他算子的 A3/950 实现 | OpDef/README/binary/CMake |
