---
type: Operator
title: "HardtanhBackward 950 ACLNN 算子知识"
description: "说明 aclnnHardtanhBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/docs/aclnnHardtanhBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_api/aclnn_hardtanh_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_api/aclnn_hardtanh_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_api/hardtanh_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_host/hardtanh_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_host/arch35/hardtanh_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_kernel/hardtanh_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_host/config/ascend950/hardtanh_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/node.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
  - resource: https://gitcode.com/cann/asc-devkit/blob/7a7156352f594b5ca79b533f72c5a99f19b87983/include/utils/tiling/template_argument.h
    role: implementation
created_at: "2026-08-25T18:48:46Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# HardtanhBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnHardtanhBackwardGetWorkspaceSize` + `aclnnHardtanhBackward`；固定仓没有更高版本后缀。 |
| 计算语义 | 实际 Kernel 为 `out[i] = (isnan(self[i]) \|\| (self[i] > min && self[i] < max)) ? gradOutput[i] : 0`。这意味着有限值恰等于端点时输出 0；公开文档公式把端点归入通过区，存在可执行语义冲突。 |
| 输入输出 | `gradOutput/self/out` 公开契约为同 shape、同 dtype、ND、rank 0–8、可非连续；950 支持 FLOAT16、FLOAT、BFLOAT16。`min/max` 在 Host 转为 FP32 属性。 |
| 执行类型 | 核心 `HardtanhGrad` 是单个 `AIV_VECTOR` Kernel；`Contiguous/ViewCopy` 为 executor 框架边界。无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP32 原位宽比较；R-002 FP16/BF16 仅比较路径升为 FP32。两者的活动 TilingKey 均为 `3`。 |
| 核心资源 | GM + 单个 `TBuf<VECCALC>` tensor pool；R-001 为 9 个等长 slot，R-002 为 10 个等长 slot；输入/输出 slot ping-pong，中间 slot 按 DAG 活跃期复用。核心节点申报 16 MiB workspace，但 entry 不读取。 |
| 来源闭包 | `source_closed`：公开 ACLNN→executor 组图→L0 launcher→950 Tiling/POD→模板 entry→ATVOSS DAG/scheduler/Vector body→Ascend950 OpDef/binary/CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=self[i]`、`g=gradOutput[i]`，950 body 的真实逐元素语义为：

```text
inside = (x < float(max)) && (x > float(min))
nan    = (float(x) != float(x))
out[i] = (inside || nan) ? g : 0
```

FP32 的 `float(x)` 是同型值；FP16/BF16 先以 `Cast<float,U,CAST_NONE>` 将 `self` 升到 FP32，然后完成 `LT(max)`、`GT(min)` 与 `NE(x,x)`。`gradOutput` 不升精度，`Select<uint8_t,U,VSEL_TENSOR_SCALAR_MODE>` 直接在输入 dtype 中选择 `gradOutput` 或同 dtype 常量零。

公开 README/ACLNN 文档把条件写成仅 `self>max` 或 `self<min` 时为 0，因此把 `self==min/max` 归到通过区；DAG 文件注释和实际 `LT && GT` body 均把端点输出置零。本卡为实现复用记录可执行 body，并将公开语义冲突保留为失败边界。NaN 因显式 `x!=x` 分支通过梯度；固定源码不校验 `min<=max`，当区间为空或反向时，所有非 NaN 元素都输出 0。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | 与 `self/out` 同 shape；ND；公开 rank 0–8；可非连续 | FLOAT16、FLOAT、BFLOAT16 | 上游梯度。API 检查其 dtype 白名单、rank、与 `self` 的 dtype/shape 一致。 |
| `self` | 与 `gradOutput/out` 同 shape；ND；公开 rank 0–8；可非连续 | 与 `gradOutput/out` 相同 | 正向输入。内部 op 的 `result`/Kernel 的 `result` 参数实际承载该 Tensor。 |
| `min` | Host scalar | 文档列 FLOAT16、FLOAT、BFLOAT16 | 必须非空；Host 用 `aclScalar::ToFloat()` 归一成 FP32。源码没有独立 scalar dtype 或 `min<=max` 检查。 |
| `max` | Host scalar | 同 `min` | 必须非空；Host 用 `ToFloat()` 归一成 FP32。 |
| `out` | 与输入同 shape；ND；可非连续 | 与输入相同 | API 白名单检查 `out` dtype，但 API 层未显式比较 `out` 与输入 dtype/shape；950 Tiling 的 `CalcOutputDtype/CheckShape` 再要求三者一致。 |
| `workspaceSize` | Host 标量输出 | `uint64_t*` | 第一段返回整个 executor 的总 workspace；核心 Tiling slice 固定为 16,777,216 B。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并转移所有权，内含连续化、核心 op 与回写任务。 |
| `workspace/stream` | Device 地址/runtime stream | `void*` / `aclrtStream` | 第二段传给 `CommonOpExecutorRun`；任务完成前均须有效。 |

公开文档写“不支持空 Tensor”，但 ACLNN 实现在任一输入为空时令 `workspaceSize=0` 并成功返回，不添加核心任务；这是公开约束与实现行为的另一处冲突，调用方不能据此把空 Tensor 行为迁移成通用底层契约。

### 入口与版本

固定 Commit `955f8b33985b6d72989ebff341bb9c2c564a51bc` 的稳定目录只公开无后缀 `aclnnHardtanhBackward` 两段式版本。第一段创建 executor、规范化两个输入和输出、添加 `HardtanhGrad` 与 `ViewCopy`；第二段只调用 `CommonOpExecutorRun`。

公开产品表列出 A2、A3 与 Ascend 950 支持，但稳定活动构建的 `SUPPORT_COMPUTE_UNIT` 只有 `ascend950`，OpDef 仅向 `ascend950` 注册 `hardtanh_grad_apt`，binary config 也只存在 `ascend950`。同名 experimental 路线由顶层 `ENABLE_EXPERIMENTAL`（默认 OFF）替换稳定 category 后才参与构建，不是稳定 A2/A3 的活动实现，不能用于补齐无后缀卡。因此本卡只覆盖 950；A2/A3 不满足 `source_closed`。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnHardtanhBackwardGetWorkspaceSize` 按以下顺序执行：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`；`CheckNotNull` 检查三个 Tensor 与两个 scalar。
2. `GetDtypeSupportList` 在 `DAV_2201` 或 RegBase 平台返回 `{FLOAT,FLOAT16,BFLOAT16}`；950 属于 RegBase。随后检查三个 Tensor 均在白名单，并检查 `gradOutput.dtype==self.dtype`。
3. `CheckShapeValid` 检查两个输入 rank 不超过 8 且 shape 相同；输出一致性留到 Host/Tiling。
4. 将 `min/max` 分别用 `ToFloat()` 变为 `minValue/maxValue`，再创建 executor。
5. 任一输入为空时写 `workspaceSize=0`、转移 executor 并成功返回，不添加任何 Device 节点。
6. 非空时依次执行 `Contiguous(gradOutput)`、`Contiguous(self)`、`Contiguous(out)`；任一步返回空指针即返回内部错误。
7. `l0op::HardtanhGrad(gradOutputContiguous,selfContiguous,minValue,maxValue,outContiguous,executor)` 在 launcher 中重排为 `OP_INPUT(self,gradOutput)`，以匹配内部 schema 的 `result,grad` 和 Kernel ABI 的 `result,grad`。
8. `ViewCopy(hardtanhBackwardOpOut,out)` 把连续结果提交到用户输出视图；最后读取 executor 总 workspace 并转移所有权。

### 参数、Workspace 与生命周期

内部 `HardtanhGrad` 不分配新的 L0 输出，而是把 `Contiguous(out)` 得到的 Tensor直接作为 `OP_OUTPUT(out)` 和返回值。连续输入、连续输出及可能的框架 workspace 由 executor 持有；第二段结束前，用户输入、用户输出、executor、workspace 与 stream 必须保持有效。

```text
executor_workspace = framework Contiguous/ViewCopy slices
                   + HardtanhGrad slice(16,777,216 B)
```

Host/Tiling 的 `SetTilingData` 固定设置 `currentWorkspace[0]=16777216`。Kernel entry 保留 `workspace` ABI 参数，但 body 没有读取，因此该 slice 是框架/运行时预留，不承载本算法中间结果。executor 的最终对齐、各框架 slice 的 offset 和 extent 在算子源码中不可见，不能把 ACLNN 返回值恒等于 16 MiB。

### 启动图与失败传播

```text
aclnnHardtanhBackwardGetWorkspaceSize
  -> Contiguous(gradOutput)             [框架边界]
  -> Contiguous(self)                   [框架边界]
  -> Contiguous(out)                    [框架边界]
  -> l0op::HardtanhGrad
       -> ADD_TO_LAUNCHER_LIST_AICORE(HardtanhGrad,
                                      OP_INPUT(self,gradOutput),
                                      OP_OUTPUT(outContiguous),
                                      OP_ATTR(min,max))
       -> hardtanh_grad<schMode=1,dType=1>
          [单 AIV_VECTOR，blockDim=baseTiling.blockNum，POD=HardtanhGradTilingData]
  -> ViewCopy(outContiguous,out)         [框架边界]

aclnnHardtanhBackward
  -> CommonOpExecutorRun(workspace,workspaceSize,executor,stream)
```

| 节点/边 | entry 与参数 | 交接介质 | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|
| `Contiguous(gradOutput/self)` | 框架 L0 节点 | 用户 Tensor→executor 连续 Tensor；具体 workspace slice 不可见 | executor 节点列表中先于核心；临时 Tensor 持有至核心消费完成 | 空返回立即终止第一段 |
| `Contiguous(out)` | 框架 L0 节点 | 用户输出视图→连续输出 Tensor | 连续 Tensor 由 executor 持有至 `ViewCopy` 完成 | 空返回阻断核心节点 |
| `HardtanhGrad` | `hardtanh_grad(result,grad,y,workspace,tiling)` | 两个连续 GM 输入→连续 GM 输出；POD 与 16 MiB slice | 单 Kernel；同一 executor/stream 上位于 `ViewCopy` 前 | launcher/Tiling/Kernel 失败阻断回写并由 executor 返回 |
| `HardtanhGrad -> ViewCopy` | producer 为核心 Kernel，consumer 为框架 `ViewCopy` | `outContiguous`，shape/dtype 与用户 out 的公开契约一致 | executor/stream 任务顺序保证可见；完成前不得释放输出或 executor | producer 失败时没有有效结果可提交 |
| `ViewCopy` | 框架 L0 节点 | 连续输出→用户可能非连续视图 | executor 持有框架资源至第二段完成 | 失败由 `CommonOpExecutorRun` 返回 |

## Route 与参数设计

### 分核与资源计算

令 `N=product(output.storage_shape)`、`P=平台 AIV 核数`、`B=平台 UB 字节数`、`S=sizeof(U)`。DAG 中既有 FP32 比较值又有 `uint8_t` mask，因此所有 Route 的 `MaxDtypeBytes=4`、`MinDtypeBytes=1`。LEVEL_2 liveness 分析得 R-001 `BufferNum=9`、R-002 `BufferNum=10`：

```text
C = min(ceil(N * 8 / 32768), P) = min(ceil(N / 4096), P)
blockFormer = ceil(ceil(N / C) / 512) * 512       # 元素/常规核
blockNum    = ceil(N / blockFormer)               # 实际 blockDim
blockTail   = N - (blockNum - 1) * blockFormer

alignFactor = 256 * 8 / 8 = 256                   # 元素；由最小 dtype uint8 推导
R-001 maxElem = floor(B / (9*4))
R-002 maxElem = floor(B / (10*4))
ubFormer = floor(maxElem / 256) * 256              # 元素/tile

ubLoopFormer = ceil(blockFormer / ubFormer)
ubLoopTail   = ceil(blockTail / ubFormer)
ubTailFormer = blockFormer - (ubLoopFormer-1)*ubFormer
ubTailTail   = blockTail - (ubLoopTail-1)*ubFormer
```

`ElewiseTiling` 对零维长度、`P==0`、UB 小于 extra size、`ubFormer==0` 返回失败；`HardtanhGradTiling` 还在调用通用 Tiling 前校验输入/输出 shape、dtype。`blockNum` 写入 `SetBlockDim`。标量 storage shape 在算子 `CheckShape` 中用 `EnsureNotScalar` 比较，通用 Tiling 的空维乘积仍得到 `N=1`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原位宽 | key `3`；`ElementwiseSch<1,HardtanhGradDag<float>::OpDag>` | 950；公开 ACLNN 校验通过且非空；三 Tensor最终为 FLOAT、同 shape；通用 Tiling 成功 | FLOAT16/BFLOAT16 到 R-002；其它 dtype 在 API/Host 失败 |
| 2 | R-002 FP16/BF16 比较升精度 | key `3`；`HardtanhGradDag<half,float>` 或 `<bfloat16_t,float>` | 950；非空；三 Tensor最终为 FLOAT16 或 BFLOAT16、同 shape；通用 Tiling 成功 | 无下一 Device Route；资源或 Tiling 条件失败 |

```text
if any required pointer is null: ACLNN error
else if tensor dtype/rank/input-shape validation fails: ACLNN error
else if gradOutput.IsEmpty() || self.IsEmpty(): success, workspaceSize=0, no core task
else:
    make gradOutput/self/out contiguous
    Host/Tiling validates result/grad/y shape and dtype
    if dtype == FLOAT:                 R-001, key 3
    else if dtype in {FLOAT16,BF16}:   R-002, key 3
    else:                              failure
    ViewCopy(core output, user out)
```

key `3` 的编码为：`schMode=1` 在 `{0,1}` 中的索引 1 占低 1 bit，固定 `dType=HardtanhGrad_TPL=1` 作为 8-bit dtype 值左移 1 bit，因此 `1 | (1<<1)=3`。dtype 本身不编码进该 key；FP16/BF16/FP32 由各 binary 的 `DTYPE_RESULT` 编译类型决定。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原位宽 | key 3；binary dtype FLOAT | 950；FLOAT；输入/输出同 shape；非空；Tiling资源 guard 成功 | 单 `AIV_VECTOR`：`hardtanh_grad` | `ElementwiseSch<1,HardtanhGradDag<float,float>::OpDag>`；CastNoOp 被 DAG 消除 | 9×`ubFormer*4` B pool；2 输入与 1 输出 ping-pong；3 temp slot；MTE2/V/MTE3 Mutex | former/tail core、尾 tile真实 count；无 Device fallback |
| R-002 FP16/BF16 比较升精度 | key 3；binary dtype FLOAT16/BFLOAT16 | 950；FP16/BF16；其余同 R-001 | 单 `AIV_VECTOR` | `ElementwiseSch<1,HardtanhGradDag<U,float>::OpDag>`；self Cast FP32 | 10×`ubFormer*4` B pool；4 temp slot；GM DMA仍为 `count*2` B | 同 R-001；Tiling失败终止 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 稳定 950 构建 | `SUPPORT_COMPUTE_UNIT="ascend950"` 且 `TILING_DIR="arch35"` | 当前 body、OpDef 与 binary 只接到 Ascend950 | 全部 | A2/A3 不生成卡 | `activation/hardtanh_grad/CMakeLists.txt`；`HardtanhGrad::HardtanhGrad` |
| shape/dtype 一致 | Host 要求 `result==grad==y` 的 storage shape 和 dtype | scheduler 对三路使用相同 block/tile 元素 offset | 全部 | `GRAPH_FAILED` | `HardtanhGradTiling::CalcInputDtype/CalcOutputDtype/CheckShape` |
| 空域 | ACLNN 提前返回；底层 `ElewiseTiling` 对任一 dim 0 失败 | 避免 `C=0`、无效 block/tile 除法 | 全部 | ACLNN 路径成功但无 Kernel；绕过 ACLNN 则 Tiling 失败 | `aclnnHardtanhBackwardGetWorkspaceSize`; `ElewiseTiling` |
| AIV 核数 | `GetCoreNumAiv()>0` | 分核公式需要正核数 | 全部 | Tiling 失败 | `ElewiseBaseTiling::GetPlatformInfo`; `ElewiseTiling` |
| UB tile 非零 | `align_down(B/(BufferNum*4),256)>0` | tensor pool 至少容纳 9/10 个等长 slot，并满足 mask 的 256 元素对齐 | 全部 | Tiling 失败 | `GetMaxElemNum`; `ElewiseTiling` |
| FP16/BF16 多一 temp | `HardtanhGradDag<U,float>` 的 Cast 不是 NoOp | FP32 self 必须与多个比较 mask 同时覆盖 DAG 活跃区 | R-002 | 不可沿用 R-001 的 9-slot 预算 | `GetRealArgs`; `DAGSch::BufferNum`; `HardtanhGradDag` |
| 活动 schMode | 通用 `GetScheMode()` 恒为 1 | Host 只产生 key 3；声明的 mode 0 没有当前 Host 路径 | 全部 | mode 0 为 Host 不可达实现，不纳入正文 | `ElewiseBaseTiling::GetScheMode`; `hardtanh_grad_struct.h` |
| count 窄化 | scheduler 的 `tileLength:uint64_t` 传给多个 Vector wrapper 的 `uint32_t/int32_t count` | Vector API count 位宽有限 | 全部 | 固定源码无独立超界 fallback；实际上界依赖平台 UB | `ElementwiseSch::RunOp`; `Vec::Compare/Select/CopyIn` |

### POD/ABI

`HardtanhGradTilingData` 是标准布局：`EleBaseTilingData` 为 80 B、对齐 8 B；外层 `minVal` offset 80、`maxVal` offset 84，因此外层 `sizeof=88` B、`alignof=8` B。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `HardtanhGradTilingData.baseTiling` | `dim0` | `int64_t`/64 | offset 0 | 元素 | `AdaptEleBaseTilingData:=product(shape)` | 当前 scheduler不读取 | shape 乘法溢出显式检查不可见 |
| 同上 | `coreNum` | `int32_t`/32 | offset 8 | AIV 核 | `static_cast<int32_t>(ElewiseTilingData.coreNum)` | 当前 scheduler不读取 | Host 已取 `min(目标核数,P)` |
| 同上 | `ubFormer` | `int32_t`/32 | offset 12 | 元素/tile | `GetMaxElemNum` 后按 256 元素对齐 | `ElementwiseSch::Init/Process` | 必须正数；隐式窄化的独立范围检查不可见 |
| 同上 | `blockFormer` | `int64_t`/64 | offset 16 | 元素/常规核 | `ceil(ceil(N/C)/512)*512` | `CalcBlockOffset` | 最后核用 tail 参数 |
| 同上 | `blockNum` | `int64_t`/64 | offset 24 | block | `ceil(N/blockFormer)`；同值 `SetBlockDim` | 识别最后 block | 必须与实际 launch blockDim 一致 |
| 同上 | `ubLoopOfFormerBlock` | `int64_t`/64 | offset 32 | tile loop | `ceil(blockFormer/ubFormer)` | 非尾核 `loopNum` | 至少 1 |
| 同上 | `ubLoopOfTailBlock` | `int64_t`/64 | offset 40 | tile loop | `ceil(blockTail/ubFormer)` | 尾核 `loopNum` | 至少 1 |
| 同上 | `ubTailOfFormerBlock` | `int64_t`/64 | offset 48 | 元素 | `blockFormer-(loops-1)*ubFormer` | 非尾核最后 tile | `1..ubFormer` |
| 同上 | `ubTailOfTailBlock` | `int64_t`/64 | offset 56 | 元素 | `blockTail-(loops-1)*ubFormer` | 尾核最后 tile | `1..ubFormer` |
| 同上 | `elemNum` | `int64_t`/64 | offset 64 | 元素 | `=maxElemNum` | 当前 scheduler不读取 | inactive；不是逻辑 N |
| 同上 | `scheMode` | `uint64_t`/64 | offset 72 | 模板枚举 | `GetScheMode()=1` | 值编码到 key；body 中不读 POD 字段 | mode 0 Host 不可达 |
| `HardtanhGradTilingData` | `minVal` | `float`/32 | offset 80 | 数值 | `runtimeAttrs[0]` 非空时赋值 | entry `SetVar<float,0>`，GT 比较 scalar | 属性指针为空时保持 Tiling buffer初值；固定源码未显式初始化 |
| 同上 | `maxVal` | `float`/32 | offset 84 | 数值 | `runtimeAttrs[1]` 非空时赋值 | entry `SetVar<float,1>`，LT 比较 scalar | 同上 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | key 3 + FLOAT binary | `HardtanhGradTiling::RunTiling/SetTilingData`; `ElewiseTiling` | `HardtanhGradTilingData{baseTiling,minVal,maxVal}` | `hardtanh_grad<1,HardtanhGrad_TPL>` | `ElementwiseSch`; `HardtanhGradDag<float>`；`Vec::Compare/And/Or/Select/CopyIn/CopyOut` | OpDef `ascend950`; FP32 binary config；算子 CMake |
| R-002 FP16/BF16 | key 3 + FP16/BF16 binary | 同上，dtype 分支实例化 `<half>`/`<bfloat16_t>` DAG | 同一 POD | 同一 entry specialization | 同上并增加 `Vec::Cast<float,U,CAST_NONE>` | FP16/BF16 binary config；同一 950 接线 |

## Kernel 实现

### 公共执行流程

两条 Route 共用单 Kernel launch graph。entry 解析 POD、声明 `KERNEL_TYPE_AIV_ONLY`，构造一个 `TPipe` 与 `ElementwiseSch<schMode,HardtanhGradDag<DTYPE_RESULT>::OpDag>`，将 `minVal/maxVal` 写入 scheduler 的两个 FP32 Var，再按 ABI 顺序 `Init(result,grad,y)` 和 `Process()`。

scheduler 把第 `b` 核的 GM base 绑定到 `blockFormer*b*sizeof(U)`，每核按 `ubFormer` 切 tile，最后核与最后 tile 分别使用对应 tail 字段。每 tile 依 DAG 拓扑执行 self CopyIn、可选 Cast、三次 Compare、And、Or、grad CopyIn、Select 与 CopyOut。所有计算 LocalTensor 来自一个 `TBuf<VECCALC>` pool；input/output slot 通过 tile 奇偶 ping-pong，中间 slot 由编译期 liveness 复用。

### Route R-001：FP32 原位宽比较

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/hardtanh_grad/op_api/aclnn_hardtanh_backward.cpp` | `GetDtypeSupportList/CheckParams` | 950 RegBase 白名单含 FLOAT；非空；输入 shape/dtype 合法 | 连续化；否则 ACLNN错误 |
| L0 | `activation/hardtanh_grad/op_api/hardtanh_grad.cpp` | `l0op::HardtanhGrad` | `OP_INPUT(self,gradOutput)`、`OP_OUTPUT(out)` | 添加 AICore launcher |
| Tiling | `activation/hardtanh_grad/op_host/arch35/hardtanh_grad_tiling_arch35.cpp` | `HardtanhGradTiling::RunTiling` | `outputDtype==DT_FLOAT` | `DoTiling<HardtanhGradDag<float>>` |
| Key | 同上 + `hardtanh_grad_struct.h` | `SetTilingData`; template declaration | `scheMode=1,dType=1` | key `3`、`blockDim=blockNum` |
| Router | `activation/hardtanh_grad/op_kernel/hardtanh_grad_apt.cpp` | `hardtanh_grad<1,1>` | `dType==HardtanhGrad_TPL` | `ElementwiseSch<1,HardtanhGradDag<float>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(HardtanhGrad,self,gradOutput,out,min,max)
  -> hardtanh_grad<1,1>(result,grad,y,workspace,tiling)
    -> GET_TILING_DATA(HardtanhGradTilingData)
    -> ElementwiseSch<1,HardtanhGradDag<float>::OpDag>
       .SetVar<float,0>(minVal).SetVar<float,1>(maxVal)
    -> Init(result,grad,y) -> InitBuffer(pool,9*ubFormer*4)
    -> Process -> Run(tileOffset,count,pingPong)
      -> CopyIn<float>(self)
      -> CompareScalar<float,LT>(self,max)
      -> CompareScalar<float,GT>(self,min)
      -> And<uint8_t>
      -> Compare<float,NE>(self,self)
      -> Or<uint8_t>
      -> CopyIn<float>(gradOutput)
      -> Select<uint8_t,float,VSEL_TENSOR_SCALAR_MODE>(mask,gradOutput,0)
      -> CopyOut<float>(out)
  -> executor stream edge -> ViewCopy(outContiguous,out)
```

##### 源码等价伪码

```text
# 源码等价伪码
gmBase = blockFormer * blockIdx
loops  = (blockIdx == blockNum-1) ? ubLoopOfTailBlock : ubLoopOfFormerBlock
tail   = (blockIdx == blockNum-1) ? ubTailOfTailBlock : ubTailOfFormerBlock

for tile in [0,loops):
    count = (tile == loops-1) ? tail : ubFormer
    off   = tile * ubFormer
    pp    = tile & 1
    DataCopyPad(selfUB(pp), selfGM[gmBase+off], 1,count*4,0,0)
    lt  = CompareScalar<LT>(selfUB,maxVal,count)
    gt  = CompareScalar<GT>(selfUB,minVal,count)
    in  = And(lt,gt,count)
    nan = Compare<NE>(selfUB,selfUB,count)
    sel = Or(in,nan,count)
    DataCopyPad(gradUB(pp),gradGM[gmBase+off],1,count*4,0,0)
    outUB = Select(sel,gradUB,0.0f,VSEL_TENSOR_SCALAR_MODE,count)
    DataCopyPad(outGM[gmBase+off],outUB,1,count*4,0,0)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `self` | `(blockFormer*blockIdx+tile*ubFormer)*4` B | `count*4` B | FP32/连续 ND | Contiguous/user | self CopyIn | 用户/executor；Kernel期间有效 | tail 按 count |
| GM `gradOutput` | 同上 | `count*4` B | FP32/连续 ND | Contiguous/user | grad CopyIn | 同上 | 与 self同元素域 |
| MTE2 self ping/pong | pool 的两个 input slot | 每 slot `L=ubFormer*4` B | FP32线性 | `Vec::CopyIn` | Compare节点 | TPipe/Kernel | tile奇偶切换；MTE2→V Mutex |
| MTE2 grad ping/pong | 另两个 input slot | 每 slot L | FP32线性 | `Vec::CopyIn` | Select | 同上 | grad 延后搬入，仍按 pp 选择 |
| temp slots ×3 | pool 中间区 | 每 slot L，合计 `3L` | FP32或uint8逻辑视图 | Compare/And/Or | 后继 mask节点/Select | DAG活跃期 | LEVEL_2 按最后消费者释放后复用；物理容量按最大 dtype FP32 |
| MTE3 out ping/pong | 两个 output slot | 每 slot L | FP32线性 | Select | CopyOut | TPipe/Kernel | V→MTE3 Mutex |
| GM `out` | `(blockFormer*blockIdx+tile*ubFormer)*4` B | `count*4` B | FP32/连续 ND | CopyOut | ViewCopy/user | executor输出 Tensor | 只提交有效 count |
| 核心 workspace | 节点 slice | 16,777,216 B | opaque | Host/Tiling | entry 未读 | executor/节点期间 | 不参与算法复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 9
  i0p["self ping · L"]
  i1p["grad ping · L"]
  op["out ping · L"]
  t0["temp0 · L"]
  t1["temp1 · L"]
  t2["temp2 · L"]
  i0q["self pong · L"]
  i1q["grad pong · L"]
  oq["out pong · L"]
```

##### 字节账与复用顺序

```text
L = ubFormer * MaxDtypeBytes = ubFormer * 4
UB_total = (2 inputs*2 + 1 output*2 + 3 temp) * L
         = 9 * ubFormer * 4 <= B
dynamic_ub = N/A（AIV TBuf pool，不是 SIMT dynamic UB）
workspace_total(core) = 16,777,216 B；body 消费 0 B
=> ubFormer = align_down(floor(B/36),256 elements)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| self/grad ping-pong slot | `Vec::CopyIn` | Compare/Select | 同奇偶性的下一 tile | `Get/ReleaseTensor<VECIN>` 与计算节点 `VECCALC` Mutex | 最后消费者释放后 MTE2 才可覆盖 |
| temp0..2 | Compare/And/Or | DAG后继节点 | liveness 判定无后续引用的计算节点 | `PrepareArgs/PostArgs` 对输入 `Get/ReleaseTensor<VECCALC>`，输出也有 VECCALC lock | DAG拓扑与 bufferIds必须保持一致 |
| out ping-pong slot | Select | `Vec::CopyOut` | 同奇偶性的下一 tile | VECCALC 与 `Get/ReleaseTensor<VECOUT>` | MTE3完成搬出后才复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + `Mutex PIPE_MTE2` | self/grad GM | DAG计算节点 | input ping/pong slot | `blockCount=1,blockLen=count*4 B,stride=0`，padding关闭 |
| 比较 | `CompareScalar LT/GT`、`Compare NE` | self LocalTensor | And/Or | temp mask与 self slot | 全 FP32；严格不等端点；真实 count传 Vector API |
| mask | `And<uint8_t>`、`Or<uint8_t>` | Compare mask | Select | 3 个可复用 temp slot | NaN mask显式并入通过条件 |
| 选择 | `Select` tensor-scalar mode | mask、grad slot、常量 0 | CopyOut | out slot | FP32原位宽；不构造或乘 `grad_self` Tensor |
| UB→GM | `DataCopyPad` + `Mutex PIPE_MTE3` | out slot | GM output | output ping/pong slot | `count*4` B，仅有效 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/hardtanh_grad/op_host/arch35/hardtanh_grad_tiling_arch35.cpp` | `HardtanhGradTiling::RunTiling/SetTilingData` | FP32 DAG、key 3、blockDim、16 MiB workspace、min/max POD |
| Kernel entry/router | `activation/hardtanh_grad/op_kernel/hardtanh_grad_apt.cpp` | `hardtanh_grad<1,1>` | AIV-only；SetVar；实例化 dtype binary 对应 DAG |
| Device body/helper | `activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h` | `HardtanhGradDag<float>::OpDag` | strict-interior、NaN、Select与 LEVEL_2策略 |
| Scheduler dependency | `opbase/.../elewise_sch.h` | `ElementwiseSch::Init/Process/Run` | GM offset、tile循环、pool、Mutex与Vector调用顺序 |

### Route R-002：FP16/BF16 比较升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/hardtanh_grad/op_api/aclnn_hardtanh_backward.cpp` | `GetDtypeSupportList/CheckParams` | 950 RegBase；dtype为 FLOAT16/BFLOAT16；输入合法非空 | 连续化 |
| Tiling | `activation/hardtanh_grad/op_host/arch35/hardtanh_grad_tiling_arch35.cpp` | `HardtanhGradTiling::RunTiling` | `outputDtype==DT_FLOAT16` 或 `DT_BF16` | `DoTiling<HardtanhGradDag<half/bfloat16_t>>` |
| Key | 同上 + `hardtanh_grad_struct.h` | `SetTilingData` | 与 R-001同为 `scheMode=1,dType=1` | key 3 |
| Router | `activation/hardtanh_grad/op_kernel/hardtanh_grad_apt.cpp` | `hardtanh_grad<1,1>` | binary 的 `DTYPE_RESULT` 为 half/bfloat16_t | 对应 `HardtanhGradDag<U,float>` |

#### Device 数据流

##### 函数级调用链

```text
hardtanh_grad<1,1>
  -> ElementwiseSch<1,HardtanhGradDag<U,float>::OpDag>::Init/Process
    -> CopyIn<U>(self,count*2 B)
    -> Cast<float,U,CAST_NONE>(self)
    -> CompareScalar<float,LT/GT>(selfFp32,max/min)
    -> Compare<float,NE>(selfFp32,selfFp32)
    -> And<uint8_t> -> Or<uint8_t>
    -> CopyIn<U>(gradOutput,count*2 B)
    -> Select<uint8_t,U,VSEL_TENSOR_SCALAR_MODE>(mask,gradOutput,U(0))
    -> CopyOut<U>(out,count*2 B)
```

##### 源码等价伪码

```text
# 源码等价伪码；分核与 tile 循环同 R-001
DataCopyPad(selfU,selfGM,count*2 B)
selfF = Cast<float,U,CAST_NONE>(selfU,count)
lt    = CompareScalar<LT>(selfF,maxVal,count)
gt    = CompareScalar<GT>(selfF,minVal,count)
inside= And(lt,gt,count)
nan   = Compare<NE>(selfF,selfF,count)
mask  = Or(inside,nan,count)
DataCopyPad(gradU,gradGM,count*2 B)
outU  = Select(mask,gradU,U(0),VSEL_TENSOR_SCALAR_MODE,count)
DataCopyPad(outGM,outU,count*2 B)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM self/grad | `(blockFormer*blockIdx+tile*ubFormer)*2` B | 各 `count*2` B | FP16/BF16连续 ND | 框架/用户 | CopyIn | 用户/executor | tail真实 count |
| input ping/pong ×4 | pool input slots | 每 slot `L=ubFormer*4` B，逻辑搬入仅 `count*2` B | U视图 | CopyIn | Cast或Select | TPipe/Kernel | 容量按 max dtype FP32统一 |
| temp slots ×4 | pool中间区 | 每 slot L | selfFp32或uint8 mask逻辑视图 | Cast/Compare/And/Or | 后继节点 | DAG活跃期 | 比 FP32 Route 多一个 Cast活跃 slot |
| output ping/pong ×2 | pool output slots | 每 slot L，逻辑有效 `count*2` B | U视图 | Select | CopyOut | TPipe/Kernel | MTE3同步 |
| GM out | 同输入元素 offset×2 B | `count*2` B | U连续 ND | CopyOut | ViewCopy/user | executor输出 | 只写有效 tail |
| 核心 workspace | 节点 slice | 16,777,216 B | opaque | Host/Tiling | entry未读 | executor | 不参与算法 |

##### 分层内存布局

```mermaid
block-beta
  columns 10
  i0p["self ping · L"]
  i1p["grad ping · L"]
  op["out ping · L"]
  t0["temp0 · L"]
  t1["temp1 · L"]
  t2["temp2 · L"]
  t3["temp3 · L"]
  i0q["self pong · L"]
  i1q["grad pong · L"]
  oq["out pong · L"]
```

##### 字节账与复用顺序

```text
L = ubFormer * 4                  # 虽然U为2 B，pool按MaxDtypeBytes分槽
UB_total = (2 inputs*2 + 1 output*2 + 4 temp) * L
         = 10 * ubFormer * 4 <= B
dynamic_ub = N/A（AIV TBuf pool）
workspace_total(core) = 16,777,216 B；body消费0 B
=> ubFormer = align_down(floor(B/40),256 elements)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| self input slot | CopyIn<U> | Cast<float,U> | 同奇偶下一 tile | VECIN/VECCALC Mutex | Cast读取完成 |
| FP32/temp mask slots | Cast/Compare/And/Or | 比较与后继 mask | 最后消费者后由后续 DAG node或下一 tile覆盖 | DAG liveness bufferIds + VECCALC lock/release | 不得把 Cast结果在 NaN Compare前覆盖 |
| grad input slot | CopyIn<U> | Select<U> | 同奇偶下一 tile | VECIN/VECCALC Mutex | Select读取完成 |
| output slot | Select<U> | CopyOut<U> | 同奇偶下一 tile | VECCALC/VECOUT Mutex | 搬出完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad`、PIPE_MTE2 Mutex | GM输入 | Cast/Select | input slots | 每路 `count*2` B，padding关闭 |
| 升精度 | `Cast<float,U,CAST_NONE>` | self U | 三个 Compare | FP32 temp slot | 仅 self 升精度；FP16/BF16值可精确表示为FP32 |
| 比较/mask | LT、GT、NE、And、Or | FP32 self | Select | temp slots | min/max为FP32；端点严格排除，NaN通过 |
| 选择 | `Select<uint8_t,U>` | mask、原 dtype grad | CopyOut | output slot | 不对梯度升/降精度，无 round步骤 |
| UB→GM | `DataCopyPad`、PIPE_MTE3 Mutex | output slot | GM | output slots | `count*2` B，有效 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/hardtanh_grad/op_host/arch35/hardtanh_grad_tiling_arch35.cpp` | `RunTiling` dtype分支 | FP16/BF16 DAG、key/blockDim/workspace |
| Kernel entry/router | `activation/hardtanh_grad/op_kernel/hardtanh_grad_apt.cpp` | `hardtanh_grad<1,1>` + `DTYPE_RESULT` | 选择编译 binary 对应的 U |
| Device body/helper | `activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h` | `HardtanhGradDag<U,float>` | self-only Cast、FP32 Compare、U型 Select |
| Scheduler dependency | `opbase/.../dag.h/node.h/elewise_sch.h` | CastNoOp与LEVEL_2 liveness | R-002 多1 temp slot、pool/同步/复用 |

## 算子特定优化

### Trick T-001：用 strict-interior mask 直接选择梯度，消除导数 Tensor 与逐元素乘法

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：所有非空 950 核心调用；`self`、`gradOutput` 与输出满足同元素域。
- Golden 来源：[`activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h:41-55`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h#L41-L55)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // compare
    using OpCompareNaN = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_NE>, OpCopyIn0Cast, OpCopyIn0Cast>;
    using OpCompareLowerThanMax = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LT>, OpCopyIn0Cast, Placeholder::Var<T, 1>>;
    using OpCompareGreaterThanMin = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpCopyIn0Cast, Placeholder::Var<T, 0>>;

    // mask
    using OpMaskAnd = Bind<Vec::And<uint8_t>, OpCompareLowerThanMax, OpCompareGreaterThanMin>;
    using OpMaskOr = Bind<Vec::Or<uint8_t>, OpMaskAnd, OpCompareNaN>;

    // select
    using OpSelect = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpMaskOr, OpCopyIn1, ConstValueZero>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpSelect>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
```

#### 机制与设计原因

实现不先生成 `grad_self` 的 0/1 Tensor再与 `gradOutput` 相乘，而是用三个 Compare 形成 `(x<max && x>min) || isnan(x)` 的 uint8 mask，随后 `Select(mask,gradOutput,zero)` 直接产生最终输出。这样源码可证明省去一个同 shape 导数 Tensor和一次逐元素 Mul；`SELECT_MODE_VS=1` 落到 tensor-scalar Select，零值由 `MAKE_CONST(U,0.0)` 提供，不需要从 GM 搬入。`MemLevel::LEVEL_2` 又允许中间 mask 在最后消费者之后复用 temp slot。

#### 迁移边界

- 适用：导数只能取“原梯度或零”，且选择谓词可以逐元素形成 mask 的反向算子。
- 前提：必须保留 strict-interior 与 NaN分支的真实语义、`uint8_t` mask dtype、tensor-scalar Select mode 和 DAG拓扑。
- 失效：若导数包含非 0/1 比例、需要对梯度做算术变换，必须增加计算节点；若接口要求端点通过，则必须改变比较谓词，不能直接复用本 DAG。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的可验证错误边界是把 `LT/GT` 改成公开文档所暗示的端点通过语义会改变 `self==min/max` 结果；固定仓未提供其它可验证反例。

### Trick T-002：FP16/BF16 只提升比较输入，不提升梯度数据通路

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：`U` 为 `half` 或 `bfloat16_t`，模板第二参数保持默认 `T=float`。
- Golden 来源：[`activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h:36-53`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h#L36-L53)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // copy in & cast to float
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    // compare
    using OpCompareNaN = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_NE>, OpCopyIn0Cast, OpCopyIn0Cast>;
    using OpCompareLowerThanMax = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LT>, OpCopyIn0Cast, Placeholder::Var<T, 1>>;
    using OpCompareGreaterThanMin = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpCopyIn0Cast, Placeholder::Var<T, 0>>;

    // mask
    using OpMaskAnd = Bind<Vec::And<uint8_t>, OpCompareLowerThanMax, OpCompareGreaterThanMin>;
    using OpMaskOr = Bind<Vec::Or<uint8_t>, OpMaskAnd, OpCompareNaN>;

    // select
    using OpSelect = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpMaskOr, OpCopyIn1, ConstValueZero>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpSelect>;
```

#### 机制与设计原因

边界属性已经由 Host 归一为 FP32；模板只把 `self` Cast 为 FP32后比较，避免以 FP16/BF16 量化 `min/max`。梯度路径仍为 `CopyIn<U> -> Select<U> -> CopyOut<U>`，因此不增加 `gradOutput` 的 FP32物化，也没有输出降精度 round。代价是 Cast结果在 NaN与上下界比较全部结束前保持活跃，LEVEL_2预算由 R-001 的9个 slot增加到10个 slot；Host Tiling通过 `OpDag::BufferNum`自动缩小 `ubFormer`。

#### 迁移边界

- 适用：低精度输入只参与 FP32标量边界判断，而有效输出可原样选择低精度 Tensor的场景。
- 前提：低精度值向 FP32转换保持精确；输出确实只是原梯度或零；Host 与 Device都使用 FP32属性。
- 失效：若输出需要基于 FP32中间量做算术并回写低精度，则必须明确 round mode和额外 buffer；若删除 Cast，比较边界将被低精度量化。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能沿用 FP32 Route 的9-slot UB分母：Cast增加了最大活跃 temp 数，`OpDag::BufferNum`实际为10。固定仓未提供真机性能反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 原位宽 | R-002 FP16/BF16 比较升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 950 FLOAT | 950 FLOAT16/BFLOAT16 | Host dtype分支；不存在相互 fallback |
| 模板与 dtype | `HardtanhGradDag<float,float>`；CastNoOp消除 | `HardtanhGradDag<U,float>`；保留 Cast | 属性/比较统一 FP32，梯度保持 U |
| 循环与分核 | `C=min(ceil(N/4096),P)`；512元素核对齐；256元素tile对齐 | 相同 | uint8 mask决定最小 dtype和对齐 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单 TBuf pool，9 slots；无 Queue/dynamic UB/通信 | 单 TBuf pool，10 slots | FP16/BF16 多一个活跃 FP32 Cast结果 |
| 内存总占用与复用 | `36*ubFormer` B；3 temp | `40*ubFormer` B；4 temp | LEVEL_2 liveness与统一4 B slot |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad + Compare/And/Or/Select` | 多一个 Cast；GM DMA仍为2 B/元素 | 只提升比较路径 |
| 同步、精度与 tail | MTE2/V/MTE3 Mutex；FP32比较/输出；真实 count | 同步相同；FP32比较、U输出，无降精度 | scheduler统一处理 tail |
| fallback/失败边界 | Tiling失败，无 Device fallback | 同上 | 单 key、单 Kernel |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档标支持，但固定 Commit 稳定活动源码链未闭合；不生成无后缀卡 | 同 A2 | 公开支持且 `source_closed` |
| Host/Launcher | experimental 候选默认不活动，不能补稳定链 | 同 A2 | 稳定 ACLNN + `arch35` ATVOSS Tiling，key 3 |
| Kernel/资源 | 稳定活动 Kernel/Tiling/config缺失 | 同 A2 | `hardtanh_grad<1,1>`，单 AIV_VECTOR；9/10-slot TBuf pool；核心申报16 MiB workspace |

## 限制与失败边界

- 公开公式与实际 DAG 在端点处冲突：实际 `self==min/max` 输出 0。实现复用必须明确选择接口契约还是可执行 body，不能静默混用。
- 公开文档写不支持空 Tensor，ACLNN 源码却提前成功返回；本卡只记录差异，不把它推广为稳定契约。
- API 未显式校验 `out` 与输入 shape/dtype一致，950 Host/Tiling会失败；绕过稳定 Tiling或更换 launcher 时不得删除该约束。
- `min/max` 只做 `ToFloat()`，没有 `min<=max` 或 scalar dtype白名单的源码检查；反向区间下非 NaN元素全部为0是 body推论，不代表公开保证。
- `minVal/maxVal` 仅在 runtime attr指针非空时赋值；schema将属性设为 REQUIRED，若绕过schema使指针为空，POD值的初始化保证不可见。
- shape乘积、`int64_t -> int32_t` 的 `ubFormer/coreNum` 窄化以及 Vector count窄化缺少独立显式上界检查；安全边界依赖平台资源和框架shape约束。
- 核心固定申报16 MiB workspace但不读取；ACLNN总 workspace还可能包含框架节点，不能硬编码核心常量替代第一段返回值。
- A2/A3虽在公开产品表标支持，但稳定活动实现未闭合；`hardtanh_backward.md` 未生成。experimental 目录不是默认稳定构建来源。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/hardtanh_grad/docs/aclnnHardtanhBackward.md` | 产品支持表、函数原型、参数表 | 公开平台、dtype/shape/layout与两段式接口 |
| 入口声明/约束 | `activation/hardtanh_grad/op_api/aclnn_hardtanh_backward.h` | 两个 ACLNN声明与组图注释 | ABI和公开入口 |
| 入口实现 | `activation/hardtanh_grad/op_api/aclnn_hardtanh_backward.cpp` | `CheckParams`; 两段式函数 | 校验、空域、Contiguous/HardtanhGrad/ViewCopy、executor生命周期 |
| L0 launcher | `activation/hardtanh_grad/op_api/hardtanh_grad.cpp` | `l0op::HardtanhGrad` | `self,gradOutput` 到 `result,grad` 的重排与 AICore launcher |
| 平台注册 | `activation/hardtanh_grad/op_host/hardtanh_grad_def.cpp` | `HardtanhGrad::HardtanhGrad` | dtype/format/schema与 ascend950 `hardtanh_grad_apt` |
| Host/Tiling | `activation/hardtanh_grad/op_host/arch35/hardtanh_grad_tiling_arch35.cpp` | `HardtanhGradTiling::*` | dtype/shape、DAG分支、key/blockDim/workspace、属性POD |
| POD | `activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_tilingdata.h` | `HardtanhGradTilingData` | 外层POD布局 |
| 模板 selector | `activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode/dType实例空间 |
| Device DAG | `activation/hardtanh_grad/op_kernel/arch35/hardtanh_grad_dag.h` | `HardtanhGradDag` | Cast/Compare/mask/Select/CopyOut与LEVEL_2 |
| Kernel entry | `activation/hardtanh_grad/op_kernel/hardtanh_grad_apt.cpp` | `hardtanh_grad` | AIV-only、POD、SetVar、scheduler启动 |
| binary config | `activation/hardtanh_grad/op_host/config/ascend950/hardtanh_grad_binary.json` | `HardtanhGrad_*_ND` | 950 FP16/BF16/FP32 binary |
| 构建接线 | `activation/hardtanh_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR/add_modules_sources` | 仅Ascend950活动稳定实现 |
| 构建选择 | `cmake/func.cmake` | `add_category_subdirectory` | ENABLE_EXPERIMENTAL在稳定与experimental category间择一 |
| 通用 POD | `opbase/.../elewise_base_struct.h` | `EleBaseTilingData` | 字段类型与ABI顺序 |
| 通用 Tiling声明 | `opbase/.../elewise_tiling.h` | `ElewiseBaseTiling::DoTiling` | DAG BufferNum/dtype到资源参数 |
| 通用 Tiling实现 | `opbase/.../elewise_tiling.cpp` | `ElewiseTiling`; `AdaptEleBaseTilingData` | 分核、tile、tail与guard公式 |
| 通用 scheduler | `opbase/.../elewise_sch.h` | `ElementwiseSch` | pool、GM offset、tile循环、逐节点调用和同步 |
| DAG元编程 | `opbase/.../dag.h` | `DAGSch` | CastNoOp、FunList、BufferNum |
| 内存liveness | `opbase/.../node.h` | `DagNodeInfo::GetBufferNumLevel2` | input/output ping-pong与temp slot计数 |
| Vector wrapper | `opbase/.../vec.h` | `Cast/Compare/And/Or/Select/CopyIn/CopyOut` | 实际 AscendC primitive和参数 |
| 同步helper | `opbase/.../sync.h` | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 Mutex协议 |
| key编码 | `asc-devkit/.../template_argument.h` | `FastEncodeTilingKeyDirect` | key 3的bit编码规则 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 80 B base + 两个FP32属性，外层88 B | 字段顺序、offset、scheMode=1、min/max Var索引 | `HardtanhGradTilingData`; `EleBaseTilingData`; `SetVar` |
| 调用顺序 | `Contiguous inputs/out -> HardtanhGrad -> ViewCopy`；单核心Kernel | launcher将 API `(gradOutput,self)` 重排为内部 `(result=self,grad=gradOutput)` | `aclnnHardtanhBackwardGetWorkspaceSize`; `l0op::HardtanhGrad` |
| workspace/内存 | 核心申报16 MiB但body不读；9/10×4 B slot预算 | ACLNN总workspace必须用第一段值；FP16/BF16不可用9-slot预算 | `SetTilingData`; `ElementwiseSch::Init`; `DAGSch::BufferNum` |
| 同步 | input/output ping-pong，temp按liveness复用 | `Get/ReleaseTensor`的MTE2/V/MTE3协议和bufferIds必须匹配 | `ElementwiseSch`; `sync.h` |
| 精度与尾块 | FP16/BF16仅self升FP32；Select保持grad dtype；真实count搬运 | strict LT/GT端点、NaN NE分支、无输出降精度、former/tail字段 | `HardtanhGradDag`; `ElementwiseSch::Process` |
| 失败边界 | 空域ACLNN提前返回；Host/Tiling资源或一致性失败无fallback | 不得用experimental路线补稳定A2/A3，不得忽略公开/body冲突 | `CheckParams`; `HardtanhGradTiling`; `add_category_subdirectory` |
