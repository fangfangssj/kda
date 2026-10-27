---
type: Operator
title: "StridedSlice 950 Direct Invoke 算子知识"
description: "说明 cann_bench::StridedSliceNpu 与 launch_strided_slice_* 在 950 上的接口语义、Launcher 参数生成、Route 选择、单 Kernel 启动图与逐路线实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_host/strided_slice_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/strided_slice_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_two_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_gather.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_nddma.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_nddma_last_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:39:07Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# StridedSlice 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `torch.ops.cann_bench.strided_slice` → `cann_bench::StridedSliceNpu` → dtype 专用 `launch_strided_slice_{float16,bfloat16,float32,int8,uint8,int32,int64}`；Direct Invoke |
| 计算语义 | 按 `begin/end/strides` 与五类 mask 对连续输入做 Python/TensorFlow 风格切片；Host 把切片折叠为输入基址、稠密输出后缀积和有符号源步长，Device 只搬运张量值 |
| 输入输出 | 输入 rank 0..8、contiguous ND；输出保留 `new_axis` 的 size-1 维并移除 `shrink_axis`；直调 dtype 为 FP16/BF16/FP32/INT8/UINT8/INT32/INT64 |
| 执行类型 | 单 Device Kernel；R-001 内含 SIMT，R-002/R-003 为 AIV 搬运，R-004 为 NDDMA+AIV 搬运，R-005 为 AIV_VECTOR MicroAPI gather，R-006 为 AIV 标量 GM 索引；无可达 Cube/AIC 或 MIX_CV 节点 |
| Kernel 路线 | 按重要性降序：P0 R-001 rank-1..4 SIMT、P1 R-002 2-D row MoveAlign、P2 R-004 NDDMA、P3 R-003 generic MoveAlign、P4 R-005 32-bit gather、P5 R-006 signed-64 fallback；另有两个 selector 不可达的 gather 模板分支 |
| 核心资源 | GM；SIMT/fallback 无显式 UB；MoveAlign 单 bind queue 两个 88 KiB slot；NDDMA 两个 64 KiB slot；gather 为 80 KiB input Queue + 64 KiB output Queue + MicroAPI registers；workspace=0 |
| 来源闭包 | `direct_implementation_closed`：Direct 注册、归一化、稳定 POD、block producer、单节点 launch、全部可达 body 与构建接线由转换交付闭合；固定官方源码只证明 950/arch35 支持及相关算法血缘，不把 Direct 符号冒充为官方接口 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设连续输入为 `x`。Host 先把稀疏切片规格（ellipsis/new/shrink/mask）展开到输入轴。对每个未 shrink 的有效输入轴 `d` 得到归一化起点 `b_d`、切片步长 `s_d≠0` 和长度 `L_d`；对正步长，`L=max(0,ceil((end-begin)/s))`，对负步长，`L=max(0,ceil((begin-end)/(-s)))`。Host 把 `Σ b_d*x.stride(d)` 折入 `xPtr`，把每个有效轴的 `x.stride(d)*s_d` 写入 folded physical `sourceSteps`。切片步长非零不蕴含 `sourceSteps` 非零：contiguous 张量的 size-1 物理轴允许 stride 为 0，此时该轴的 folded step 也是 0，且输出轴长度至多 1。Device 对输出线性索引 `q` 解码坐标 `c_d`，读取：

```text
y[q] = xPtr[ Σ_d c_d * sourceSteps[d] ]
```

`new_axis` 只向返回 shape 插入 1，不增加 Device 有效 rank；`shrink_axis` 只推进输入基址，不进入输出坐标。所有 Route 都是元素位搬运，没有 cast、算术舍入或累加。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | rank 0..8，contiguous ND | FP16、BF16、FP32、INT8、UINT8、INT32、INT64 | 非连续输入直接失败；Tensor 值只由 Device 读取 |
| `begin` | Host list | int64 | 与 `end/strides` 等长；负值先加轴长；mask 可覆盖 |
| `end` | Host list | int64 | 与 `begin/strides` 等长；按步长方向裁剪；mask 可覆盖 |
| `strides` | Host list | int64 | 每项非零；可正可负。生产实现未拒绝 `INT64_MIN`，负向长度计算中的 `-step` 会先发生 signed overflow，不能依赖后续 Route fallback |
| `begin_mask/end_mask` | 标量属性 | int64，默认 0 | 对置位参数轴分别采用方向相关的整轴起点/终点 |
| `ellipsis_mask` | 标量属性 | int64，默认 0 | 最多一位；展开为足够的输入整轴 |
| `shrink_axis_mask` | 标量属性 | int64，默认 0 | 对应轴只选择一个 index 并从输出 rank 删除；不得与 new-axis 同位 |
| `new_axis_mask` | 标量属性 | int64，默认 0 | 在输出 shape 插入 size-1 轴，不消费输入轴 |
| `y` | contiguous ND；shape 由上述规则生成 | 与 `x` 相同 | 由入口新分配；空输出不 launch；无原地 alias |

### 入口与版本

Direct schema 是 `strided_slice(Tensor x, int[] begin, int[] end, int[] strides, int begin_mask=0, int end_mask=0, int ellipsis_mask=0, int shrink_axis_mask=0, int new_axis_mask=0) -> Tensor y`。Meta 与 PrivateUse1 都调用相同归一化器；PrivateUse1 额外检查 contiguous、建立当前设备 guard、取当前 NPU stream 并按 dtype 调用 launcher。它不是 ACLNN，也不复用官方 registry tiling buffer。

固定官方 `README.md` 明确列出 Ascend 950PR/950DT 支持，官方 `CMakeLists.txt` 把 `ascend950` 映射到 `arch35`；因此官方相关实现的平台身份闭合。Direct 转换目标固定为 950/Ascend 950、`dav-3510`、CANN 9.1.0。官方固定源码中的 key 100/101/102/103/150/200/201/300..303 与本卡 Direct `route=0/1/2` 不是同一 ABI；本卡后续所有精确 selector 均来自 Direct 转换实现，官方文件只作相关 body/设计来源。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序如下：

1. `BuildSlicePlan` 校验输入 rank 0..8、三列表等长、ellipsis 至多一位、shrink/new 不重叠。
2. `NormalizeAxis` 拒绝零步长；负 begin/end 先加轴长，随后按步长符号应用 mask 和裁剪，计算该轴长度。生产 `NegativeSliceLength` 直接执行 `positiveStep=-step`，故 `step==INT64_MIN` 在生成 `SlicePlan` 前已有 signed-overflow 缺口，尚未形成可传播的 Host 失败边。
3. new-axis 只写 `outputShape.push_back(1)`；ellipsis 展开为输入整轴；shrink 只累加 `baseOffset`；普通轴同时写输出长度和 `x.stride*step`。
4. 检查输出维度非负以及元素总数乘法不溢出 int64；分配 `y`。`totalLength==0` 时直接返回，不启动 Device。
5. `MakeConfig` 计算输出后缀积、blockNum 和外层 route。`StridedSliceNpu` 再把 `baseOffset*dtypeBytes` 折入 `xPtr`，取当前 stream，在 `OpCommand::RunOpApi("StridedSliceDirect", lambda)` 内按 dtype launch。

任一 `TORCH_CHECK` 在输出 launch 前终止；不含 CPU/ATen 张量值 fallback，也不读取 Device tensor value。

### 参数、Workspace 与生命周期

`SlicePlan` 和 `StridedSliceConfig` 都由 Host 栈对象持有；launch 时 config 按值传入 Kernel。输入/输出 Tensor 持有 GM，`xPtr` 是输入 allocation 内的派生基址，要求输入 Tensor 至少覆盖整个异步 Kernel 生命周期；输出由返回 Tensor 持有。launcher 使用当前 stream，只有一个 Kernel 节点，因此无跨 Kernel event、workspace edge 或 cleanup。

```text
AIV = max(platform.GetCoreNumAiv(), 1)
blockNum = min(AIV, max(ceil(max(totalLength,1)/minElementsPerCore),1))
minElementsPerCore = 32   if route=2 && rank=1 && sourceSteps[0]>1
                   = 4096 otherwise
workspace_total = 0
```

Direct 实现不查询 UB 大小；88 KiB、64 KiB、80 KiB 等容量是面向 `dav-3510` 的编译期常量。这一点与官方 Host 读取 `GetCoreMemSize(UB)` 后动态计算 tiling 不同。

### 启动图与失败传播

```text
cann_bench::StridedSliceNpu
  -> BuildSlicePlan -> MakeConfig
  -> OpCommand::RunOpApi("StridedSliceDirect", lambda)
     -> launch_strided_slice_{dtype}(xPtr, yPtr, config, stream)
        -> strided_slice_{dtype}<<<config.blockNum, nullptr, stream>>>
           -> DispatchStridedSlice<T> -> exactly one R-001..R-006 body
```

| 节点 | Host 启动 | entry/grid/POD | 输入→输出 | ordering / lifetime / 失败 |
|---|---|---|---|---|
| K-001 | dtype 专用 `launch_strided_slice_*` | `strided_slice_*<<<blockNum,nullptr,stream>>>`；按值 `StridedSliceConfig` | `xPtr GM → y GM`；无中间边 | 单 stream 提交；Host 归一化/dtype 失败阻断 launch；空输出跳过 K-001；Device route 无返回码 |

所有 Route 都只有 K-001。官方 registry entry 还接收 begin/end/strides/workspace/tiling GM；Direct entry 只接收折叠后的 `xPtr/y/config`，不要混用两种调用约定。

## Route 与参数设计

### 分核与资源计算

`outputShapeProd[d]=∏_{j=d}^{rank-1} outputDims[j]`，`sourceSteps[d]=x.stride(inputAxis_d)*normalizedStep_d`。Route 先由 Host 分成三组：任何负 source step、任何 step 超过 int32、任何 suffix 超过 uint32、有效 rank 不在 1..4 或总元素超过 uint32 均为 `route=1`；其余在 `totalLength>=4096 || innermostSourceStep!=1` 时为 `route=2`；剩余为 `route=0`。这里的 fast domain 是 **非负** folded step；0 可来自 size-1 物理轴，末轴为 0 时因 `!=1` 进入 route2。Device 对 `route=2` 再按 rank、尾维 bytes、步长、dtype 和规模选择搬运 body。

SIMT 每个 AIV 启动 2048 个 SIMT threads，线程 `blockIdx*2048+threadIdx` 起步并以 `blockNum*2048` 跨步。MoveAlign/NDDMA 以 segment 或完整 row 为调度单位，`segment=blockIdx; segment<segmentCount; segment+=blockNum`。fallback 以单元素为单位，`idx=blockIdx; idx<totalLength; idx+=blockNum`。

### Route 选择

下表以重要性降序组织 Route；`Host 顺序` 不参与重要性排序，专门保留源码的真实控制流。记 `H1` 为 `MakeConfig` 的逐轴扫描（命中负/过宽 step 或过宽 suffix 时把 route 粘滞地置 1，但不提前退出循环），`H2` 为扫描后的 rank/total 宽度判断，`H3` 为仅在 route 仍为 0 时把大输出或非单位末步长置 2，`H4` 为保留初值 0。Device 端 `D1` 先处理 route2，内部依次为连续尾行分支 `D1.a`、rank≤2 稀疏分支 `D1.b` 和其余 `D1.c`；随后 `D2` 接 route0，最终 `D3` 是 fallback。

| 重要性 | Route | 路线特征 | Key/模板 | 核心资源与 tail |
|---|---|---|---|---|
| P0 | R-001 rank-1..4 SIMT | 小连续切片的低固定成本主体，也承接 rank2 8/32/64-bit 稀疏短路径 | `route=0` 或 route2 sparse fallback / `StridedSliceOfficial::Run<T>` | GM→GM、2048-thread grid stride；`idx<outputSize` 保护尾线程 |
| P1 | R-002 2-D row MoveAlign | 末轴连续且整行可装入 slot 时，完整行归核并多行合并搬运 | `route=2` / `RunTwoDim<T>` | bind Queue 2×88 KiB；core row tail + tile row tail |
| P2 | R-004 NDDMA | 用最多 5-D descriptor 把非负 folded-step 矩形直接压实到 UB；size-1 物理轴允许 source step=0 | `route=2` / `StridedSliceNddma::Run<T>` | bind Queue 2×64 KiB；split-axis `actualFactor` tail |
| P3 | R-003 generic MoveAlign | 连续大后缀无法使用完整行专线时，用 blockCount+两级 loop-mode 搬运 | `route=2` / `Run<T>` | bind Queue 2×88 KiB；segment tail 与 packed/row-wise CopyOut |
| P4 | R-005 32-bit gather | 大规模 rank2 FP32/INT32 高步长列采样，用 register index 和 mask 压实 | `route=2` / `GatherDirect::Run<T>`，`sizeof(T)==4` | input/output Queue 80+64 KiB；row/vector mask tail |
| P5 | R-006 signed-64 fallback | 负步长、宽字段、高有效 rank 或未被 fast selector 接收的语义闭包 | `route=1` 或 selector fallback / `Supplemental::Run<T>` | 无显式 UB；signed-int64 mixed-radix；逐元素尾界 |

| 重要性 | Route | Key/模板 | 精确选择条件与优先级 | Host 顺序（真实短路/fallback） | 不适用去向 |
|---|---|---|---|---|---|
| P0 | R-001 rank-1..4 SIMT | `route=0` 或 route2 sparse fallback | ①route=0 且 rank 1..4；或②route2、内层 step>1、rank=2、`sizeof(T)` 为 1/4/8，且 4-byte 类型不满足 R-005 | ①H4→D2；②H3→D1.b，在 rank1/16-bit NDDMA 与 32-bit gather 判断之后走 `D1.b` 的最后 `else` | 宽字段/负步长到 R-006 |
| P1 | R-002 2-D row MoveAlign | `route=2` / `RunTwoDim<T>` | rank 1..4；内层 source step=1；`rowBytes=lastDim*sizeof(T)>=128`；且 rank=2、`rowBytes<=88 KiB`、rows≥blockNum | H3→D1.a；是 D1.a 内第一个嵌套命中，命中即返回 body | 嵌套 guard 不满足到 R-003 |
| P2 | R-004 NDDMA | `route=2` / `StridedSliceNddma::Run<T>` | ①内层 step>1、rank≤2，且 rank=1 或 `sizeof(T)==2`；②既不满足连续尾行≥128 B，也不属于内层 step>1 且 rank≤2的其余 route2 情况，包括末 folded step=0 | H3→D1.b 的首个 dtype/rank短路，或 H3→D1.c 的末尾 `else` | D1.b 首个 guard 不满足时继续 R-005/R-001 |
| P3 | R-003 generic MoveAlign | `route=2` / `Run<T>` | rank 1..4；内层 source step=1；rowBytes≥128；未命中 R-002 | H3→D1.a；仅在 R-002 嵌套 guard 失败后走同分支 `else` | D1.a 不成立后继续 D1.b/D1.c |
| P4 | R-005 32-bit gather | `route=2` / `GatherDirect::Run<T>` | rank=2、`sizeof(T)==4`、内层 step≥4、`totalLength>=2^20`；注册 dtype 中即 FP32 或 INT32 | H3→D1.b；先经过 R-004 的 rank1/16-bit 短路，再在 4-byte specialization 内判断，命中即返回 body | guard 不满足到 R-001 |
| P5 | R-006 signed-64 fallback | `route=1` 或异常 selector | H1/H2 产生 route1；或 Device 未满足 `route=0/2 + rank1..4` guard | H1/H2→D3；D3 是整个 `if route2 ... else if route0 ... else` 的最终 `else` | 终点 |

```text
MakeConfig:
  if negative/too-wide sourceStep or suffix>u32 or rank outside 1..4 or total>u32: route=1
  else if total>=4096 or lastSourceStep!=1: route=2
  else: route=0

Dispatch:
  if route=2 && rank in 1..4:
    if lastStep==1 && rowBytes>=128:
      if rank==2 && rowBytes<=88KiB && rows>=blockNum: R-002
      else: R-003
    else if lastStep>1 && rank<=2:
      if rank==1 || sizeof(T)==2: R-004
      else if sizeof(T)==4 && lastStep>=4 && total>=2^20: R-005
      else: R-001
    else: R-004
  else if route=0 && rank in 1..4: R-001
  else: R-006
```

#### 可达性盘点

这里的 `active/conditional/dormant` 描述生产调用链可达性，不把“输入谓词成立”本身误写为 `conditional`：R-001..R-006 都能由公开 schema 的合法输入命中，因而均为 `active`；没有依赖编译开关、能力探测结果或外部 selector 的 `conditional` Route。

| 状态 | Route/分支 | 生产链证明 | 重新启用条件 |
|---|---|---|---|
| active | R-001 | rank1..4、非负且窄的 source step；例如小型末轴连续输出由 H4→D2 命中，rank2 的 1/4/8-byte 稀疏输入也可由 D1.b 末支命中；size-1 外轴可带 step0 | N/A |
| active | R-002 | rank2、末步长1、128 B≤rowBytes≤88 KiB、rows≥blockNum 且 Host 先产生 route2 | N/A |
| active | R-004 | rank1 稀疏、rank2 16-bit 稀疏、末 folded step=0，或 rank3/4 等不落入连续 MoveAlign 与 rank≤2 sparse 子树的 route2 输入 | N/A |
| active | R-003 | rank1/3/4 连续大后缀，或不满足 R-002 完整行 ownership/capacity guard 的 rank2 连续大后缀 | N/A |
| active | R-005 | rank2 FP32/INT32、末步长≥4、total≥2^20；选择单行 footprint 不超过 Queue slot 的形状即可同时满足当前安全前提 | N/A |
| active | R-006 | 任一负 source step、超 i32 step、超 u32 suffix/total、有效 rank0或5..8 都由 H1/H2→D3 命中 | N/A |
| dormant | DR-001 `GatherDirect::Run<T>` 的 `rank==1` 分支 | `D1.b` 对 rank1 先执行 `if (rank==1 \|\| sizeof(T)==2) Nddma::Run`，因此 rank1 不可能调用 `GatherDirect::Run` | 必须先修改上游 selector，使某类 rank1/正步长输入显式选择 gather，并同时证明单 tile 容量、index 位宽与相对 NDDMA 的价值；仅直接调用 body 不构成生产 Route |
| dormant | DR-002 gather 的非 4-byte 模板分支（含 `sizeof(T)==1` 时转 NDDMA） | `GatherDirect::Run<T>` 的唯一调用点位于 `if constexpr(sizeof(T)==4)`；当前只会实例化 FP32/INT32，1/2/8-byte dtype 均在调用前被 NDDMA 或 SIMT 截断 | 必须扩展 dispatcher 的 dtype selector，并为对应 `RegTensor/IdxType/vl`、UB footprint 和 API 支持补齐约束；其中 1-byte 分支即使启用仍立即转 R-004，不应另算 gather Route |

`DispatchStridedSlice` 的最终 `else` 还会接收理论上的未知 `route` 值，但 Host producer 只写 0/1/2；未知值只是与 R-006 共用 body 的不可达防御边，不是第七条生产 Route。若未来扩展 key，必须新增显式分支，不能依赖该静默归并。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 SIMT | route0 或 route2 sparse fallback | 七 dtype；rank1..4；满足上表两组谓词 | K-001，AIV entry 内 `asc_vf_call` | `StridedSliceOfficial::Run<T>` → `StridedSliceDim{1..4}Compute` | GM→GM；每 core 2048 threads；无显式 UB/Queue | thread grid-stride guard `idx<outputSize`；超宽/负步长转 R-006 |
| R-002 row MoveAlign | route2 + row guard | 七 dtype；rank2；step-last=1；128 B≤row≤88 KiB；rows≥blocks | K-001 | `StridedSliceMoveAlignDirect::RunTwoDim<T>` | bind Queue 两个 88 KiB slot；compact DataCopyPad；整行归核 | `tileRows` 与 `coreEnd` 处理双重 tail；guard 失败到 R-003 |
| R-004 NDDMA | route2 NDDMA predicates | 七 dtype；rank1..4；非负 source step；符合上表 NDDMA 分支，含 size-1 轴产生的 step0 | K-001 | `StridedSliceNddma::Run<T>` | bind Queue depth2×64 KiB；5-D MultiCopy→packed CopyOut | 末 segment `actualFactor`；不设运行时 fallback |
| R-003 generic MoveAlign | route2 + dense suffix guard | 七 dtype；rank1..4；step-last=1；row≥128 B；非 R-002 | K-001 | `StridedSliceMoveAlignDirect::Run<T>` | 同一 bind Queue；loop-mode 最多两层；packed copyout | splitFactor/segment tail；descriptor width由 fixed integer 限制 |
| R-005 32-bit gather | route2 gather predicate | FP32/INT32；rank2；step-last≥4；total≥2^20 | K-001 | `StridedSliceGatherDirect::Run<T>` → `ProcessTile/GatherRows` | input 80 KiB + output 64 KiB Queue；MicroAPI register/mask | rows/tile、vector mask 和 copyout row tail；单行 footprint 无 selector guard |
| R-006 signed-64 fallback | route1/selector fallback | 七 dtype；rank0..8；负/宽 step、大 suffix/total 或其它未接收组合 | K-001 | `StridedSliceSupplemental::Run<T>` | `GlobalTensor::GetValue/SetValue`，无显式 UB | int64 坐标与地址；逐元素 `idx<totalLength` |

### 关键性能 Guard

下表同时区分“生产已有”与“安全实现要求”。标为“生产缺失”的 guard 不能当作当前代码已经提供的保护；建议出边只有在改造 selector 后才成立。

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32-bit SIMT ABI（生产已有但不充分） | rank 1..4、total/suffix≤u32、0≤sourceStep≤i32 | SIMT output index 为 uint32、source stride/coordinate 为 int32 | R-001 fast | 生产只在这些字段本身越界时选 R-006 | Direct `MakeConfig`；官方相关 `StridedSliceSIMT::ProcessNormalShape` |
| SIMT 乘加闭包（生产缺失） | 安全实现还需逐轴证明 `coordMax=outputDim-1≤INT32_MAX`、`sourceStep*coordMax≤INT32_MAX`，并证明所有轴项之和≤UINT32_MAX；乘法应在 int64/checked arithmetic 中验证后再窄化 | body 的 `stridesN * outputIndex[N]` 先按 int32 做有符号乘法，可能在加到 `uint32 inputIndex` 前已经溢出；只检查最终数学和不够 | R-001 | guard 失败先转 R-006；若 Host 的 int64 folded offset/step 也未证明无溢出，则应在 launch 前显式失败 | Direct `StridedSliceDim{1..4}Compute/MakeConfig` |
| small fixed-cost | total<4096 且 lastStep=1 | 保留 SIMT 小规模路径 | R-001 | route2 selector | Direct `MakeConfig` |
| row width | lastStep=1 且 rowBytes≥128 | 与官方 MoveAlign 最小连续搬运尺度一致 | R-002/R-003 | NDDMA/SIMT | Direct `DispatchStridedSlice`；官方 `MIN_MOVE_ALIGN_LEN` |
| row ownership | rank2、rowBytes≤88 KiB、rows≥blockNum | 一个 slot 容纳至少一行且每个已启动 core 有完整行 | R-002 | R-003 | Direct `DispatchStridedSlice/RunTwoDim` |
| NDDMA rank | Direct rank≤4，descriptor 固定 5 维 | `MultiCopyParams<T,5>`；未用维补 1 | R-004 | rank>4 为 R-006 | Direct `StridedSliceNddma::Run`；官方 `SetNddmaTilingMode` |
| loop-mode width | rank≤4，最多 blockCount+loop1+loop2 覆盖四轴 | dav-3510 MoveAlign loop-mode 只有两级外层寄存器 | R-003 | Host rank>4 先转 R-006 | Direct `Run`；官方 `MovAlignV2UbSplitLastFourDim` |
| gather dtype/规模 | 4-byte dtype（FP32/INT32）、rank2、stride≥4、total≥2^20 | 避免小/低 stride 路线支付 index-register 设置与额外 UB 搬运 | R-005 | R-001 | Direct `DispatchStridedSlice` |
| descriptor narrow | MoveAlign 的源行 gap 使用 `srcStride:uint32`；Direct 未检查 `sourceSteps*bytes-rowBytes` | raw tile 约束会限制 blockLen/blockCount，但不能保证物理源行 gap 可装入 uint32；窄化后无运行时修复 | R-002/R-003 | 无运行时 fallback | Direct `Run/RunTwoDim`；官方 `MovAlignV2UbSplitLast{2,3,4}Dim` |
| gather Queue/寄存器/descriptor（生产缺失） | 安全实现需先以 checked int64 计算 `inputSpan=(columns-1)*stride+1`；要求 `columns≤INT32_MAX`、最大 gather index≤INT32_MAX、`ceil(columns/64)≤UINT16_MAX`、单行 input/output footprint≤80/64 KiB、`tileRows≤UINT16_MAX`、各 `blockLen≤UINT32_MAX`，多行时还需 `0≤sourceRowStep*4-inputBytes≤UINT32_MAX` | 生产把 `columns/inputRowAligned/outputRowAligned` 窄为 int32，`rowCount/loops/blockCount` 窄为 uint16，`blockLen/srcStride` 窄为 uint32；`rowsPerTile` 归零后强制设1不能修复单行超槽 | R-005 | 当前无完整 fallback；安全 selector 应先尝试满足 SIMT 乘加 guard的 R-001，否则转 R-006；Host folded int64 未闭合则 launch 前失败 | Direct `Run/GatherRows/ProcessTile` |
| fixed UB | 88/64/80/64 KiB 常量，无 Host UB capability check | 只对 950/dav-3510 资源预算成立 | R-002..R-005 | 无运行时 fallback | Direct route constants；官方 `TilingPrepare4StridedSlice` 对照 |

### POD/ABI

`StridedSliceConfig` 由 Direct Host/Device 共用同一头文件并按值传递；标准 64-bit C++ 布局下 `alignof=8`、`sizeof=152`。官方固定仓不存在该 Direct POD；官方使用按 key 分开的 `StridedSlice*TilingData`，不可替换本结构。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `StridedSliceConfig` | `totalLength` | int64/64 | off 0 | element | `plan.totalLength` | 全部 body 的循环/输出上界 | ≥1 才 launch |
| 同上 | `outputShapeProd` | int64[8]/512 | off 8 | element product | `MakeConfig` 逆序 suffix 乘积 | 坐标解码、tile/row/segment | 前 `rank` 项有效 |
| 同上 | `sourceSteps` | int64[8]/512 | off 72 | input element | `x.stride(axis)*step` | 所有输入地址/descriptor stride | 可负或为0；0 可来自 size-1 物理轴，fast Route 接收非负值；生产乘法本身未做 checked overflow |
| 同上 | `rank` | int32/32 | off 136 | axis count | `plan.rank`（不计 new-axis/shrink-axis） | router 和 mixed-radix 循环 | 0..8；fast 为1..4 |
| 同上 | `blockNum` | int32/32 | off 140 | AIV core | `calc_strided_slice_block_num` | launch grid、core stride | 1..平台 AIV 核数 |
| 同上 | `route` | int32/32 | off 144 | selector | `MakeConfig` | `DispatchStridedSlice` | 0/1/2 |
| 同上 | `reserved` | int32/32 | off 148；sizeof=152 | N/A | 零初始化 | Device 不读取 | 必须保持 0/保留 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | Direct route0/route2 sparse | `BuildSlicePlan/MakeConfig` | `StridedSliceConfig` | dtype entry → `DispatchStridedSlice<T>` | Direct rank1..4 adapter；官方相关 `strided_slice_simt.h` | Direct 单算子 registration；950→dav-3510 |
| R-002 | Direct row predicate | 同上 | 同上 | 同上 | Direct `RunTwoDim`；官方相关 `strided_slice_move_align_two_dim.h` | 同上 |
| R-004 | Direct NDDMA predicate | 同上 | 同上 | 同上 | Direct NDDMA descriptor；官方相关 `strided_slice_nddma{,_last_dim}.h` | 同上 |
| R-003 | Direct dense suffix predicate | 同上 | 同上 | 同上 | Direct `Run`；官方相关 `strided_slice_move_align.h` | 同上 |
| R-005 | Direct 4-byte gather predicate | 同上 | 同上 | 同上 | Direct `GatherRows/ProcessTile`；官方相关 `strided_slice_move_align_gather.h` | 同上 |
| R-006 | Direct route1 | 同上 | 同上 | 同上 | Direct signed-int64 scalar body；官方无相同 body | 同上 |

## Kernel 实现

### 公共执行流程

七个 dtype entry 都标记 `KERNEL_TYPE_AIV_ONLY`，只把模板 `T` 交给同一个 `DispatchStridedSlice<T>`。每次 launch 恰好进入一个 body；没有官方 TilingKey router、workspace 或第二个 Kernel 节点。Host 已将 begin 折入 `xPtr`，所以所有 body 的源索引都相对该派生基址。R-001 的 1..4-D SIMT 计算体与固定官方 `strided_slice_simt.h` 的对应 body 保持同构，但 Direct adapter 把 begin 传 0、input shape product 传 1，并把完整物理跨度提前折入 `sourceSteps`。R-002..R-005 是沿官方 arch35 搬运结构的 Direct 派生；R-006 是转换层新增的语义兜底。

### Route R-001：rank-1..4 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | Direct 转换层 | `MakeConfig` | 32-bit field 可容纳；小连续为 route0，大/稀疏为 route2 | route0 或 route2 |
| Direct router | Direct 转换层 | `DispatchStridedSlice<T>` | route0 rank1..4；或 route2 rank2 sparse 且未命中 NDDMA/gather | `StridedSliceOfficial::Run<T>` |
| 官方相关 | `op_kernel/arch35/strided_slice_simt.h` | `ProcessNormalShape` | 按 inputDims=1..4 选择 SIMT body，threadDim=2048 | 对应 `StridedSliceDim*Compute` |

#### Device 数据流

##### 函数级调用链

```text
launch_strided_slice_{dtype} -> strided_slice_{dtype}
  -> DispatchStridedSlice<T> -> StridedSliceOfficial::Run<T>
     -> GetUintDivMagicAndShift(outputShapeProd[d+1])
     -> asc_vf_call<StridedSliceDim{1..4}Compute<T>>(dim3(2048), ...)
        -> GM input[sourceIndex] -> GM output[idx]
```

##### 源码等价伪码

```text
for thread idx = blockIdx*2048 + threadIdx; idx < totalLength; idx += blockNum*2048:
  remainder = idx
  for d in 0..rank-2:
    coord[d] = fast_uint_div(remainder, outputShapeProd[d+1])
    remainder -= coord[d] * outputShapeProd[d+1]
  coord[rank-1] = remainder
  sourceIndex = Σ coord[d] * sourceSteps[d]
  y[idx] = xPtr[sourceIndex]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `xPtr + Σcoord[d]*sourceSteps[d]` | one element/thread | T/strided source | Host base fold | SIMT thread | input Tensor/Kernel | 无 UB padding |
| output GM | `y + idx` | one element/thread | T/dense | SIMT thread | caller | output Tensor | `idx<totalLength` |
| SIMT registers | thread-local coords/magic operands | rank≤4 | int32/uint32 | `Run`/thread | same thread | thread lifetime | 无共享/atomic |

##### 分层内存布局

N/A：该 Route 不分配 AIV Queue/TBuf 或 dynamic UB，使用 SIMT thread-local register 和 GM 直读直写。

##### 字节账与复用顺序

```text
UB_total = 0（无显式 InitBuffer）
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local coord/index | 当前 SIMT iteration | 当前 GM load/store | 下一 grid-stride iteration | 单线程程序序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| index→store | N/A，独立线程 | 当前 thread | 同 thread store | 单一 `y[idx]` | 无 cast；`idx<outputSize` 只保护尾线程。生产未保护 int32 `sourceStep*coordinate` 的先行有符号溢出，也未证明各项累加可装入 uint32；安全实现须在 Host 用 checked int64 逐项验算，失败转 R-006，若 folded int64 本身不安全则 launch 前失败 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | Direct 转换层 | `MakeConfig` | 生产 32-bit fast guard、route、suffix/source step |
| Kernel entry/router | Direct 转换层 | `DispatchStridedSlice<T>` | route0 与 sparse rank2 fallback 映射到 SIMT |
| Device body/helper | `op_kernel/arch35/strided_slice_simt.h` | `StridedSliceDim1..4Compute` | 官方相关 body 使用 2048 threads、magic division 与 grid-stride loop |

### Route R-002：2-D 完整行 MoveAlign

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | Direct 转换层 | `MakeConfig` | total≥4096 或末 sourceStep≠1，且无宽/负字段 | route2 |
| Direct router | Direct 转换层 | `DispatchStridedSlice<T>` | rank2、lastStep=1、128 B≤rowBytes≤88 KiB、rows≥blockNum | `RunTwoDim<T>` |
| 官方相关 | `op_kernel/arch35/strided_slice_move_align_two_dim.h` | `Process` | 按 core 分完整行，compact CopyIn→packed CopyOut | 2-D MoveAlign body 血缘 |

#### Device 数据流

##### 函数级调用链

```text
K-001 -> DispatchStridedSlice<T> -> RunTwoDim<T>
  -> core row range
  -> AllocTensor -> DataCopyPad<Compact> rows GM->UB -> EnQue/DeQue
  -> DataCopyPad packed UB->GM -> FreeTensor
```

##### 源码等价伪码

```text
rowsPerTile = max(floor(88KiB/align32(rowBytes)),1)
rowsPerCore = ceil(rows/blockNum)
[coreBegin,coreEnd) = clipped blockIdx range
copyIn = {blockCount=tileRows, blockLen=rowBytes,
          srcStride=sourceSteps[0]*sizeof(T)-rowBytes, dstStride=rowBytes}
for row in [coreBegin,coreEnd) step rowsPerTile:
  CompactCopy xPtr[row*sourceSteps[0] : tileRows rows] -> queue slot
  CopyOut one packed burst of tileRows*rowBytes -> y[row*columns]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM rows | `row*sourceSteps[0]` | tileRows×rowBytes with source gaps | T/2-D source | caller | MTE2 | input Tensor | rowBytes≥128 |
| bind queue | slot | `2*88 KiB` physical allocation | T/compact rows | MTE2 | MTE3 | TPipe/Kernel | two slots；row footprint align32 |
| output GM | `row*columns` | tileRows×rowBytes | T/dense | MTE3 | caller | output Tensor | only valid bytes written |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["bind queue slot 0 · 88 KiB"]:1
  q1["bind queue slot 1 · 88 KiB"]:1
```

##### 字节账与复用顺序

```text
rowFootprint = align32(columns*sizeof(T))
rowsPerTile = floor(88KiB/rowFootprint)
UB_total = 2 * 88KiB = 176KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bind Queue slot | `DataCopyPad<Compact>` | output `DataCopyPad` | 后续 tile | EnQue→DeQue→FreeTensor | copyout 完成才释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→GM | bind Queue | MTE2 | DeQue/MTE3 | 当前 tile slot | 位搬运；`coreEnd` 与 `tileRows` 双重裁剪；padding 不写回 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | Direct 转换层 | `MakeConfig/calc_strided_slice_block_num` | route2、blockNum、row predicate inputs |
| Kernel entry/router | Direct 转换层 | `DispatchStridedSlice<T>` | 完整行 guard 优先于 generic MoveAlign |
| Device body/helper | `op_kernel/arch35/strided_slice_move_align_two_dim.h` | `StridedSliceMoveAlignTwoDim::Process` | 官方相关完整行 ownership、compact Queue 搬运与 tail |

### Route R-004：NDDMA 矩形搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | Direct 转换层 | `MakeConfig` | 非负 folded source step 且 route2；size-1 物理轴可产生 step0 | Device 二次选择 |
| Direct router | Direct 转换层 | `DispatchStridedSlice<T>` | rank1 sparse、rank2 16-bit sparse，或其余 route2 非 MoveAlign/非 sparse-SIMT 情况 | `Nddma::Run<T>` |
| 官方相关 | `op_kernel/arch35/strided_slice_nddma.h` | `SetLoopInfo/ProcessPerBlock` | 5-D descriptor 从 source/dst stride 生产并搬入 bind Queue | NDDMA body 血缘 |

#### Device 数据流

##### 函数级调用链

```text
K-001 -> Dispatch -> StridedSliceNddma::Run<T>
  -> choose splitAxis/splitFactor within 64KiB slot
  -> build MultiCopyParams<T,5>
  -> DataCopy<T,5>(GM rectangle -> local)
  -> EnQue/DeQue -> DataCopyPad(local -> packed GM) -> FreeTensor
```

##### 源码等价伪码

```text
capacity = 64KiB/sizeof(T)
tileBudget = min(ceil(total/blockNum), capacity)
choose largest right-aligned output hyper-rectangle fitting tileBudget
for segment = blockIdx; segment < segmentCount; segment += blockNum:
  decode prefix and splitBegin
  for axis from last down to splitAxis:
    loopSize = actualFactor on splitAxis else outputDim(axis)
    loopSrcStride = sourceSteps[axis]            # elements
    loopDstStride = 1 or outputShapeProd[axis+1] # elements
  unused of 5 loops := size1/stride0
  NDDMA GM->UB; packed DataCopyPad actualElements*sizeof(T) UB->GM
```

`loopSrcStride=0` 在此不是“切片 stride 为 0”：Host 已拒绝后者。它表示 contiguous 输入的 size-1 物理轴本身 stride 为 0；该轴输出长度至多 1，descriptor 可合法复用同一物理地址。末轴 folded step=0 时，D1.a/D1.b 均不命中而由 D1.c 进入本 Route。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM rectangle | decoded prefix + splitBegin*sourceSteps[splitAxis] | descriptor loop product | T/strided up to 4-D | caller | NDDMA | input Tensor | loop stride单位 element |
| bind queue | slot | `2*64 KiB` | T/dense packed | NDDMA | MTE3 | TPipe/segment | depth2 |
| output GM | dense segment offset | actualFactor*denseSuffix | T/dense | MTE3 | caller | output Tensor | exact valid bytes |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["NDDMA bind queue slot 0 · 64 KiB"]:1
  q1["NDDMA bind queue slot 1 · 64 KiB"]:1
```

##### 字节账与复用顺序

```text
capacity = 64KiB/sizeof(T)
actualElements <= capacity
UB_total = 2 * 64KiB = 128KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| NDDMA bind Queue slot | `DataCopy<T,5>` | packed `DataCopyPad` | 下一 segment | EnQue→DeQue→FreeTensor | descriptor 完成且 copyout 消费后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| descriptor→MTE3 | bind Queue | NDDMA | DeQue/MTE3 | packed tile | 无 cast；`actualFactor` 改写 split loop size；未用维补1/stride0，size-1 物理轴也可消费真实 stride0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | Direct 转换层 | `MakeConfig` | route2 与 rank1 sparse 的 32 elems/core block 策略 |
| Kernel entry/router | Direct 转换层 | `DispatchStridedSlice<T>` | NDDMA 多分支合并到相同 body |
| Device body/helper | `op_kernel/arch35/strided_slice_nddma.h` | `StridedSliceNDDMA::SetLoopInfo/ProcessPerBlock` | 官方相关 5-D loop descriptor、Queue 与 main/tail copyout |

### Route R-003：generic MoveAlign loop-mode

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | Direct 转换层 | `MakeConfig` | route2 | 进入 Device selector |
| Direct router | Direct 转换层 | `DispatchStridedSlice<T>` | rank1..4、lastStep=1、rowBytes≥128、未命中 row route | `MoveAlignDirect::Run<T>` |
| 官方相关 | `op_kernel/arch35/strided_slice_move_align.h` | `ProcessPerBlock` | SetLoopMode→Compact Copy→Reset→packed CopyOut | generic MoveAlign 血缘 |

#### Device 数据流

##### 函数级调用链

```text
K-001 -> Dispatch -> MoveAlignDirect::Run<T>
  -> choose splitAxis/splitFactor within 64KiB raw budget
  -> for segment assigned to core
     -> derive prefix, inputOffset, outputOffset
     -> build blockCount/blockLen + loop1/loop2 descriptors
     -> SetLoopMode -> Compact DataCopyPad -> ResetLoopMode
     -> Queue handoff -> packed/row-wise DataCopyPad output
```

##### 源码等价伪码

```text
tileBudget = min(ceil(total/blockNum), 64KiB/sizeof(T))
scan axes right->left; choose largest dense suffix rectangle <= tileBudget
segmentCount = prefixCount * ceil(splitDim/splitFactor)
for segment = blockIdx; segment < segmentCount; segment += blockNum:
  actualFactor = min(splitFactor, splitDim-splitBegin)
  decode prefix to inputOffset using sourceSteps
  last axis -> blockLen; second-last -> blockCount/srcStride
  third/fourth-last -> loop1/loop2 size and src/dst stride
  CompactCopy GM->UB; choose one packed copyout if <=2 axes or innerRowsBytes%32==0,
  otherwise row-block copyout; write actualFactor*denseSuffix elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM rectangle | prefix/sourceSteps + splitBegin | actualFactor×denseSuffix valid elems plus gaps | T/strided prefix+dense last axis | caller | loop-mode MTE2 | input Tensor | blockLen bytes；stride bytes |
| bind queue | slot | `2*88 KiB` | T/compact with possible 32-B row padding | MTE2 | MTE3 | TPipe/segment | raw tile≤64 KiB；padding headroom≤88 KiB |
| output GM | `prefix*outputShapeProd[splitAxis]+splitBegin*denseSuffix` | actualFactor×denseSuffix | T/dense | MTE3 | caller | output Tensor | packed or blockCount rows |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["bind queue slot 0 · 88 KiB"]:1
  q1["bind queue slot 1 · 88 KiB"]:1
```

##### 字节账与复用顺序

```text
raw_tile <= 64KiB
padded_tile <= 88KiB（Direct 假设 innermost row >=128B 时 padding 不超过 25%）
UB_total = 2 * 88KiB = 176KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bind Queue slot | loop-mode Compact CopyIn | packed CopyOut | 下一 segment | ResetLoopMode 后 EnQue/DeQue/FreeTensor | 必须每次 Reset，防止参数污染下一 DMA |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| loop-mode | `SetLoopModePara/ResetLoopModePara` | descriptor producer | DataCopy/后续迭代 | MTE2 寄存器状态 | loop size 禁用值为 1；0 会抑制搬运 |
| Queue | EnQue/DeQue/Free | MTE2 | MTE3 | 当前 compact tile | 无 cast；`actualFactor` 处理最后 segment |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | Direct 转换层 | `MakeConfig` | 生产 rank/steps/products/blockNum |
| Kernel entry/router | Direct 转换层 | `DispatchStridedSlice<T>` | dense last-axis guard |
| Device body/helper | `op_kernel/arch35/strided_slice_move_align.h` | `ParseLoopModeAndMoveAlignParams/ProcessPerBlock` | 官方相关 loop-mode、compact DMA、Queue 顺序和 tail |

### Route R-005：大规模 32-bit MicroAPI gather

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | Direct 转换层 | `MakeConfig` | route2 | Device 二次选择 |
| Direct router | Direct 转换层 | `DispatchStridedSlice<T>` | rank2、lastStep≥4、total≥2^20、sizeof(T)=4；当前注册实例为 FP32/INT32 | `GatherDirect::Run<T>` |
| 官方相关 | `op_kernel/arch35/strided_slice_move_align_gather.h` | `GenGatherIndex/GatherInOrder` | Arange×stride 生成 index，DataCopyGather + mask 压实 | gather 血缘 |

#### Device 数据流

##### 函数级调用链

```text
K-001 -> Dispatch -> GatherDirect::Run<T>  # T=float 或 int32_t
  -> per-core row range, rowsPerTile
  -> ProcessTile
     -> DataCopyPad source span GM->inputQueue
     -> GatherRows -> Arange/Muls/Adds/DataCopyGather/DataCopy
     -> outputQueue -> DataCopyPad packed GM
```

##### 源码等价伪码

```text
assert sizeof(T)==4; vl = 256/sizeof(T) = 64
inputSpan = (columns-1)*stride + 1
rowsPerTile = min(80KiB/align32(inputSpan*sizeof(T)),
                  64KiB/align32(columns*sizeof(T)))
for each core-owned row tile:
  rowCount:u16 = tileRows
  loops:u16 = ceil(columns/64)
  copyIn = {blockCount:u16=tileRows, blockLen:u32=inputBytes,
            srcStride:u32=(sourceRowStep*sizeof(T)-inputBytes if tileRows>1 else 0)}
  copy full source span(s) GM->input UB
  baseIndex[lane] = lane*stride
  for row, for loop in ceil(columns/vl):
    mask = min(vl, columns-loop*vl)
    index = baseIndex + loop*vl*stride
    data = gather(inputRow, index, mask)
    store packed outputRow, mask
  copyOut = aligned row ? {blockCount=1, blockLen:u32=tileRows*outputBytes}
                        : {blockCount:u16=tileRows, blockLen:u32=outputBytes}
  copy packed row(s) output UB->GM
```

这些窄化是生产 body 的真实顺序，不是已验证 guard：`columns` 先转 int32，`rowCount/loops/copy*.blockCount` 转 uint16，`copy*.blockLen/srcStride` 转 uint32。安全实现必须在任何 cast 与 Queue 分配前完成 checked 乘加和范围判断。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM span | `row*sourceSteps[0]` | rows×`((columns-1)*stride+1)` | T（FP32/INT32）/row spans | caller | MTE2 | input Tensor | row span align32 in UB；selector 未证明单行≤80 KiB |
| inputQueue | slot | 80 KiB | T/padded row spans | MTE2 | MicroAPI gather | TPipe/tile | depth1 |
| outputQueue | slot | 64 KiB | T/packed rows | MicroAPI | MTE3 | TPipe/tile | depth1；selector 未证明单行≤64 KiB |
| MicroAPI registers | lane index、data、mask | 64 lanes per vector | int32/uint32/T | Arange/Muls/Gather | store | vector scope | tail mask；index 加乘需落在 int32 |
| output GM | `row*columns` | rows×columns | T/dense | MTE3 | caller | output Tensor | exact row bytes |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inputQueue · depth=1 · 80 KiB"]:3
  outQ["outputQueue · depth=1 · 64 KiB"]:2
```

##### 字节账与复用顺序

```text
intended input_tile_bytes <= 80KiB
intended output_tile_bytes <= 64KiB
UB_total = 80KiB + 64KiB = 144KiB
register_bytes = 编译器分配，固定仓不可见；生命周期限于 __VEC_SCOPE__
dynamic_ub = N/A
workspace_total = 0
```

生产 selector 没有证明两个 `intended` 不等式；单行 footprint 超槽时 `rowsPerTile` 被钳到 1，仍会把过大的 `blockLen` 交给同一 Queue slot。

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputQueue slot | MTE2 | `GatherRows` | 下一 tile | DeQue 后 gather 完成再 FreeTensor | output 已独立写入 outputQueue |
| outputQueue slot | `GatherRows` | MTE3 | 下一 tile | EnQue→DeQue→FreeTensor | copyout 完成后释放 |
| index/data RegTensor | 当前 vector loop | gather/store | 下一 loop | vector 程序序 | UpdateMask 排除无效 lane |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→gather | input Queue | MTE2 | DeQue/MicroAPI | source span | FP32/INT32 都按 32-bit 元素搬运，无 cast/数值算术 |
| gather loop | `UpdateMask<T>` | remaining count | Gather/Store | 最后 vector lanes | `remaining` 形成 mask；生产 `loops:uint16` 与 int32 index 乘加未受完整 guard |
| gather→CopyOut | output Queue | MicroAPI | DeQue/MTE3 | packed output | rowBytes%32 决定单 burst或逐行 blockCount；生产 blockCount:uint16、blockLen:uint32、srcStride:uint32 均需 cast 前检查 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | Direct 转换层 | `MakeConfig` | total/rank/sourceSteps/blockNum |
| Kernel entry/router | Direct 转换层 | `DispatchStridedSlice<T>` | `sizeof(T)==4`、stride和规模 guard，因此覆盖 FP32/INT32 |
| Device body/helper | `op_kernel/arch35/strided_slice_move_align_gather.h` | `GenGatherIndex/GatherInOrder` | 官方相关 Arange×stride、DataCopyGather、mask与Queue 管线 |

### Route R-006：signed-int64 通用 fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | Direct 转换层 | `MakeConfig` | 负/超 i32 sourceStep、suffix/total超 u32、rank不在1..4 | route1 |
| Direct router | Direct 转换层 | `DispatchStridedSlice<T>` | 非 fast 条件 | `Supplemental::Run<T>` |
| 官方相关 | `op_host/arch35/strided_slice_tiling_arch35.cpp` | `SetSIMTTilingMode`/big-shape key | 官方对 shape>u32 使用另一 SIMT big-shape body | 仅为边界对照；不是本 Direct body |

#### Device 数据流

##### 函数级调用链

```text
K-001 -> DispatchStridedSlice<T> -> StridedSliceSupplemental::Run<T>
  -> mixed-radix int64 decode
  -> GlobalTensor<T>::GetValue(sourceIndex)
  -> GlobalTensor<T>::SetValue(idx, value)
```

##### 源码等价伪码

```text
for idx = blockIdx; idx < totalLength; idx += blockNum:
  remainder = idx; sourceIndex = 0
  for dim = 0..rank-1:
    coord = (dim+1<rank) ? remainder/outputShapeProd[dim+1] : remainder
    if not last: remainder -= coord*outputShapeProd[dim+1]
    sourceIndex += coord*sourceSteps[dim]  # signed int64, may be negative relative to xPtr
  y[idx] = xPtr[sourceIndex]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | signed `Σcoord*sourceSteps` relative to folded base | one element/iteration | T/arbitrary signed stride | caller/Host base | AIV scalar load | input Tensor | no vector alignment requirement |
| output GM | `idx` | one element/iteration | T/dense | AIV scalar store | caller | output Tensor | `idx<totalLength` |
| scalar state | remainder/coord/sourceIndex | O(rank) registers | int64 | current iteration | address generator | core iteration | no shared state |

##### 分层内存布局

N/A：该 Route 不调用 `InitBuffer`，以 `GlobalTensor::GetValue/SetValue` 和标量寄存器替代 AIV UB 管线。

##### 字节账与复用顺序

```text
UB_total = 0（无显式本地缓冲）
register_bytes = 编译器分配，固定仓不可见
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-core scalar state | 当前 idx | 当前 GM load/store | 下一 idx | 单核程序序 | 每个 output idx 只被一个 core 处理 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar load/store | N/A | current core | same core | unique output element | 无 cast；全程 signed int64；`idx<totalLength` 覆盖尾部和 scalar rank0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | Direct 转换层 | `MakeConfig` | route1 汇集负 stride、宽字段和高 rank |
| Kernel entry/router | Direct 转换层 | `DispatchStridedSlice<T>` | 所有 fast guard 失败进入 fallback |
| Device body/helper | Direct 转换层 | `StridedSliceSupplemental::Run<T>` | signed-int64 mixed-radix 与 GM 标量搬运；官方无完全相同实现 |

## 算子特定优化

### Trick T-001：Host 折叠 mask/new/shrink 与输入基址

#### 触发与源码依据

- 适用 Route：R-001..R-006。
- 触发条件：所有 Direct 调用；ellipsis/new/shrink/mask 先在 Host 归一化。
- Golden 来源：[`op_host/strided_slice_util.h:305-315`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_host/strided_slice_util.h)

**官方源码（固定 Commit 连续摘录；lambda 捕获后加入不改变 C++ 语义的 Markdown-safe 注释）**：

```cpp
auto canonical = [stride_i, dim_i, masks, valid_range, real_valid] /* Markdown-safe */ (int64_t x, int c) {
    if (masks[c]) {
        return stride_i > 0 ? valid_range[c] :
                              valid_range[static_cast<uint64_t>(c + 1) & static_cast<uint64_t>(1)];
    } else {
        if (!real_valid[c]) {
            return -3L; // -3 invalid, diff from valid_range
        }
        int64_t x_fwd = x < 0 ? dim_i + x : x; // make negative indices positive
        return x_fwd < valid_range[0] ? valid_range[0] : std::min(x_fwd, valid_range[1]);
    }
};
```

#### 机制与设计原因

官方相关实现先把方向、mask、负 index 和有效范围 canonicalize。Direct 转换在相同语义之后进一步把所有 begin 累加为 `baseOffset`，把物理 input stride 与 slice step 相乘为 `sourceSteps`，并移除 Device 不消费的 new/shrink 轴。这样 K-001 无需读取 begin/end/strides GM，也不需在每个输出元素重复 mask/ellipsis 分支；所有 Route 共用一个稳定 POD。

#### 迁移边界

- 适用：属性在 Host 可见、输入 stride 可查询、输出 shape 可在 launch 前确定的切片类 Direct Invoke。
- 前提：输入连续；Host 与 Device 对 `sourceSteps` 的元素单位和基址折叠完全一致。
- 失效：动态 Device-side begin/end 或无法在 Host 确定 shape 时，必须恢复官方动态参数路径或设计新的 Device normalization 节点。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要把 `new_axis` 计入 Device 有效 rank，或把 shrink 轴留在输出坐标中；这会改变 mixed-radix product。固定仓未提供 Direct POD 的可验证反例。

### Trick T-002：loop-mode Compact 搬运多维稠密后缀

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：route2、rank1..4、末 source step=1、rowBytes≥128，且不使用 2-D 完整行专线。
- Golden 来源：[`op_kernel/arch35/strided_slice_move_align.h:190-204`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
    LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
    SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<T, PaddingMode::Compact>(inputLocal, inputGM_[inputGmAddr + loops * this->ubInLoopSteps_],
                                         copyInParam_, padParams_);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    vecQue_.EnQue(inputLocal);
    inputLocal = vecQue_.DeQue<T>();

    DataCopyPad(outputGM_[outputGmAddr + loops * this->ubOutLoopSteps_], inputLocal, copyOutParamsMain);
    vecQue_.FreeTensor(inputLocal);
}
```

#### 机制与设计原因

Direct 路线把最后轴设为 `blockLen`、倒数第二轴设为 `blockCount`，再把最多两层更外轴放进 loop-mode，一次 MTE2 描述一个最多四维的矩形。`PaddingMode::Compact` 把源间隙消去，随后 MTE3 以一个 packed burst 或若干无 gap row burst写出。关键实参都来自 `sourceSteps`、`outputShapeProd` 和当前 `actualFactor`；每次 DMA 后立即 Reset，防止 loop 寄存器泄漏到后续搬运。

#### 迁移边界

- 适用：末轴连续、rank≤4、可把外层跨度写入 MoveAlign loop-mode 的矩形 gather。
- 前提：blockCount/loop size/stride 能装入硬件字段，且 padding 后 tile 不超过 88 KiB slot。
- 失效：末轴不连续时转 NDDMA/SIMT/gather；rank>4 或宽字段进入 R-006。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

打开 loop-mode 后不调用 `ResetLoopModePara` 会污染下一次 DataCopy；把禁用外环写成 0 会在 Direct 注释所述 dav-3510 语义下抑制搬运。

### Trick T-003：5-D NDDMA 直接生成 packed tile

#### 触发与源码依据

- 适用 Route：R-004。
- 触发条件：Direct route2 的 NDDMA 分支，非负 folded source step、rank1..4；包含 size-1 物理轴产生的 source step=0。
- Golden 来源：[`op_kernel/arch35/strided_slice_nddma.h:156-168`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_nddma.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int64_t idx = 0; idx < curCoreLoopsNum_; idx++) {
    inputGmAddr = this->GetInputGmAddr(curCoreRowsOffset_ + idx * this->rowsOffsetSteps_[this->ubIndex_]);
    outputGmAddr = this->GetOutputGmAddr(curCoreRowsOffset_ + idx * this->rowsOffsetSteps_[this->ubIndex_]);

    for (int64_t loops = 0; loops < ubSplitLoopsNum_; loops++) {
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        DataCopy<T, NDDMA_MAX_DIMS, nddmaConfig_>(inputLocal,
                                                  inputGM_[inputGmAddr + loops * this->ubInLoopSteps_],
                                                  paramsMain);
        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();
        DataCopyPad(outputGM_[outputGmAddr + loops * this->ubOutLoopSteps_], inputLocal, copyOutParamsMain);
        vecQue_.FreeTensor(inputLocal);
```

#### 机制与设计原因

Direct `Run` 从输出尾轴向左选择能放进 64 KiB slot 的最大稠密超矩形，把每轴 `sourceSteps` 直接写入 NDDMA source stride，把输出后缀积写入 destination stride。DMA engine 在 GM→UB 阶段完成稀疏采样和压实，Device 不再逐元素算 source index；随后一次 packed MTE3 写回有效元素。未使用的第 5 维补 size=1/stride=0。

#### 迁移边界

- 适用：最多 5 个 descriptor loop 可表达、目标是 dense packed 输出的非负 folded-step 切片；真实 step0 只用于长度至多1的 size-1 物理轴，未用 descriptor 维同样补 stride0。
- 前提：所有 loop size/stride 符合 NDDMA API 约束，tile≤64 KiB。
- 失效：负 source step 进入 R-006；需要更多维或字段超限时需拆阶段或使用通用索引。不要把 folded step0 误判成非法切片 stride0。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把 `loopSrcStride/loopDstStride` 当字节；固定官方 tiling 注释与 Direct descriptor 都以 element 为单位。固定仓未提供 Direct descriptor 超界后的可验证 fallback。

### Trick T-004：只在可摊薄设置成本时启用 32-bit register gather

#### 触发与源码依据

- 适用 Route：R-005。
- 触发条件：rank2、4-byte dtype（当前为 FP32/INT32）、末步长≥4、输出至少 `2^20` 元素。
- Golden 来源：[`op_kernel/arch35/strided_slice_move_align_gather.h:289-309`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_gather.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__VEC_SCOPE__
{
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<RangeType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<RangeType> indexReg;
    MicroAPI::Arange(indexReg, 0);
    MicroAPI::Muls(indexReg, indexReg, RangeType(lastDimStride_), mask);
    MicroAPI::DataCopy(idxAddr, indexReg, mask);
}
```

#### 机制与设计原因

官方相关实现用 `Arange×stride` 一次生成 gather index。Direct 4-byte 路线保留同一模式，但把 base index 留在 `RegTensor<int32_t>`，并按 vector loop 只加 `loop*vl*stride`；`UpdateMask<T>` 控制最后 64-lane vector。选择器写的是 `sizeof(T)==4` 而不是浮点类型判断，因此 FP32 与 INT32 都可达；值只被 gather/store，不参与浮点或整数算术。阈值使额外的源 span CopyIn、index register 初始化和 output Queue 管线有足够工作量可摊薄；其它 1/8-byte rank2 sparse 路线留在 SIMT，2-byte 路线先被 NDDMA 接收。

#### 迁移边界

- 适用：固定步长列采样、dtype 与 index register 宽度匹配、输出规模足够大的矩阵。
- 前提：输入/输出 row footprints 分别落入 80/64 KiB slot，`columns/inputRowAligned/outputRowAligned` 可由 int32 表示，`lane*stride + loop*vl*stride` 可由 int32 index 表示，`rowCount/loops/blockCount` 可由 uint16 表示，`blockLen/srcStride` 可由 uint32 表示；这些前提当前没有被 selector 完整检查。
- 失效：FP16/BF16 走 NDDMA；FP32/INT32 小规模或 stride2 走 R-001；负步长走 R-006。安全改造应在 cast 前检查所有 footprint、loop 和 row-gap descriptor 字段：满足 R-001 乘加 guard 才退 R-001，否则退 R-006；若 Host folded int64 运算未闭合则直接失败。生产版本尚无这组自动出边。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要无条件对小矩阵启用 gather，也不要忘记最后 vector 的 `UpdateMask`；否则会增加固定开销或写入无效 lane。固定仓未提供 Direct 阈值的官方反例。

## 模板与实现对比

| 对比维度 | R-001 SIMT | R-002 row MA | R-004 NDDMA | R-003 generic MA | R-005 gather | R-006 fallback | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | 小连续或 rank2 1/4/8-byte sparse | rank2 完整行可归核 | 低精度/高 rank sparse等 | 连续大后缀 | 大 4-byte stride≥4 | 负/宽/高 rank | 先专用搬运，再通用语义 |
| 模板与 dtype | rank1..4，七 dtype | `T`，七 dtype | `T`，七 dtype | `T`，七 dtype | `T`，仅 FP32/INT32 可达 | `T`，七 dtype | dtype 宽度改变搬运容量/selector |
| 循环与分核 | 2048-thread grid stride | core→row→tile | core-strided segment | core-strided segment | core row range→tile→vector | core-strided element | blockNum统一由 Host 产生 |
| Queue/TBuf/寄存器 | SIMT registers | bind Queue 2×88KiB | bind Queue 2×64KiB | bind Queue 2×88KiB | Queue 80+64KiB + RegTensor | scalar registers | 950 固定容量，无 dynamic UB |
| 内存总占用与复用 | 显式0 | 176KiB | 128KiB | 176KiB | 144KiB | 显式0 | Queue 都在 FreeTensor 后复用 |
| 搬运/Vector/SIMT/NDDMA API | `asc_vf_call` GM直访 | Compact DataCopyPad | 5-D DataCopy | loop-mode Compact DataCopyPad | DataCopyGather | Get/SetValue | 末轴连续度和规模决定机制 |
| 同步、精度与 tail | thread guard，无 cast | Queue；row/tile tail | Queue；actualFactor | Queue+Reset；segment tail | 双 Queue+mask | 程序序；idx guard | 所有路线位搬运 |
| fallback/失败边界 | 宽字段到 R-006 | guard失败到R-003 | 无 descriptor runtime fallback | 无 descriptor runtime fallback | guard失败到R-001 | 最终语义路线 | UB/descriptor 超限无运行时修复 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方 README 支持；本 Direct 卡不覆盖 | 官方 README 支持；本 Direct 卡不覆盖 | 官方 README 支持；本卡目标平台 |
| Host/Launcher | 固定交付未提供对应 Direct 构建闭包 | 固定交付未提供对应 Direct 构建闭包 | `ascend950`/`dav-3510` Direct launcher；CANN 9.1.0 |
| Kernel/资源 | 不得套用 950 固定 UB/loop-mode 参数 | 不得套用 950 固定 UB/loop-mode 参数 | AIV-only；SIMT、NDDMA、MoveAlign、MicroAPI；88/64/80 KiB 固定 slot |

## 限制与失败边界

- Direct 输入必须 contiguous；不会隐式 Contiguous，也没有 ACLNN/CPU/ATen 数值 fallback。
- Direct 只注册 FP16/BF16/FP32/INT8/UINT8/INT32/INT64；官方 README 的更宽 dtype 集不能外推到该入口。
- 输入 rank 校验为 0..8；Device fast 路线只接收有效 rank 1..4，其余由 R-006 处理。new-axis 不进入有效 rank，但仍保留在输出 shape。
- `begin/end/strides` 必须等长、stride 非零、ellipsis 至多一位、shrink/new mask 不重叠；空输出直接返回。生产的“stride 非零”检查仍接受 `INT64_MIN`，随后 `NegativeSliceLength` 的 `positiveStep=-step` 发生 signed overflow，且位置早于 `MakeConfig`，所以 R-006 无法接管。安全实现应在 `NormalizeAxis` 开头显式拒绝 `INT64_MIN`，或用 checked/unsigned magnitude（例如不先求有符号相反数）计算绝对值；失败必须在分配输出和 launch 前由 Host 明确传播。
- 输出元素总数显式检查 int64 乘法溢出；`baseOffset += begin*x.stride` 与 `sourceSteps=x.stride*step` 没有独立 int64 溢出检查。
- `paramCount` 与输入 rank 没有显式上界/消费数一致性检查：输入轴耗尽后的非 new-axis 参数会被循环递增跳过；ellipsis 展开维数为负时又被截为 0。超额切片规格不具备“显式拒绝或完整消费”的证明，复用时应补关系校验。
- mask 运算使用 `int64_t{1} << paramDim`；Direct 只扫描 ellipsis 的低 32 位，也没有把 `paramDim<64` 固化为前置条件。超长参数列表或高位 mask 的语义没有闭合到显式失败，应在复用时增加边界校验。
- MoveAlign descriptor 含 uint16/uint32 字段，Direct 没有复刻官方逐级回退检查；极宽 blockCount/blockLen/stride 可能窄化。NDDMA 接收非负 source step，size-1 物理轴的 step0 是合法输入而不是错误；但 NDDMA API 的其它硬件字段上限在固定仓外不可见。
- SIMT Host guard 只要求 output suffix 可装入 uint32，但 body 的 `outputIndex[]` 和 `stridesN*outputIndex[N]` 是 int32、`inputIndex` 是 uint32；乘积会先在 int32 中溢出，不能用“最终数学和≤uint32”替代逐项检查。安全 selector 应证明每轴 coordinate 和 int32 乘积均可表示、再证明 uint32 总和；失败先退 R-006，Host int64 folded 运算不安全则 launch 前失败。
- 32-bit gather 生产代码把 `columns/inputRowAligned/outputRowAligned` 窄为 int32，`rowCount/loops/blockCount` 窄为 uint16，`blockLen/srcStride` 窄为 uint32；selector 未检查最大 gather index、loop 数、单行 source/output footprint、tileRows、合并 copyout 字节数或多行 source row gap。安全 selector 必须在 cast 前 checked 计算这些字段；满足 SIMT 乘加边界才退 R-001，否则退 R-006。当前生产没有该完整 fallback 链。
- Direct 不查询 UB capability，176/144/128 KiB 预算只对目标 950/dav-3510 假设成立；不能直接移植到 A2/A3。
- 所有 route 单 Kernel、workspace=0；不存在跨节点同步或可复用 workspace slice。
- R-006 只能兜底 selector 表达宽度，不能修复 Host 已经发生的 `baseOffset/sourceSteps` 有符号乘加溢出，也不能补回被 Host 跳过的超额切片参数。
- 本次知识提取未执行编译、真机、数值或性能测试；不对实际性能收益作量化承诺。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 官方平台/语义 | `conversion/strided_slice/README.md` | 产品支持表、功能说明 | Ascend 950 支持及切片语义 |
| 官方语义归一化 | `conversion/strided_slice/op_host/strided_slice_util.h` | `BuildDenseSpec/BuildProcessingData/InferShape` | mask、ellipsis、new/shrink、正负步长 canonicalization |
| 官方 POD/Route 声明 | `conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h` | `StridedSlice*TilingData/StrideSliceTiling` | key、字段、资源与 selector 状态 |
| 官方 Host/Tiling | `conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.cpp` | `RunStrideSliceTiling/Set*TilingMode/FillTilingData` | arch35 Route、block/UB、descriptor producer |
| 官方 Kernel router | `conversion/strided_slice/op_kernel/strided_slice_apt.cpp` | `strided_slice` | key→POD→body 分发及 AIV-only |
| 官方 body wrapper | `conversion/strided_slice/op_kernel/arch35/strided_slice.h` | `StridedSlice*Process` | dtype宽度实例和 Init/Process 接线 |
| 官方 SIMT body | `conversion/strided_slice/op_kernel/arch35/strided_slice_simt.h` | `StridedSliceSIMT/Dim1..4Compute` | SIMT index、thread grid与32-bit边界 |
| 官方 MoveAlign body | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align.h` | `ProcessPerBlock` | loop-mode Compact DMA、Queue、tail |
| 官方 2-D body | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_two_dim.h` | `Process` | 完整行归核与多行 descriptor |
| 官方 gather body | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_gather.h` | `GenGatherIndex/GatherInOrder` | MicroAPI gather、index和mask |
| 官方 NDDMA body | `conversion/strided_slice/op_kernel/arch35/strided_slice_nddma.h` | `SetLoopInfo/ProcessPerBlock` | 5-D descriptor、Queue、main/tail |
| 官方 last-dim NDDMA | `conversion/strided_slice/op_kernel/arch35/strided_slice_nddma_last_dim.h` | `ProcessPerBlockInLastDim/SetLoopInfo` | 单轴切分与 tail 对照 |
| 官方构建接线 | `conversion/strided_slice/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 |

Direct 转换源码不是固定官方仓文件，因此不进入 Frontmatter `sources`；其闭合事实在正文中统一标为“Direct 转换层”，避免将派生入口、POD、阈值和 fallback 冒充为官方实现。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单个按值 `StridedSliceConfig` 驱动七 dtype entry | 字段顺序、int64/int32 位宽、元素单位、sizeof=152、reserved不消费 | Direct `StridedSliceConfig`；官方 POD 仅作对照 |
| 调用顺序 | normalize→base fold→MakeConfig→dtype launch→单 body | 空输出必须在 launch 前返回；不得混入官方 begin/end/tiling GM ABI | Direct `StridedSliceNpu/launch_strided_slice_*` |
| workspace/内存 | workspace=0；各 Route 固定 Queue 字节账 | MoveAlign 2×88KiB、NDDMA 2×64KiB、gather 80+64KiB；不能按逻辑有效长度少算 slot | Direct `InitBuffer`；官方相关 body |
| 同步 | Queue Route 均 Alloc→EnQue→DeQue→Free；MoveAlign 必须 Reset loop-mode | 不得提前复用 slot，不得让 loop 参数跨迭代残留 | 官方 `ProcessPerBlock` 与 Direct 派生顺序 |
| 精度与尾块 | 所有 dtype 位搬运；SIMT/mask/actualFactor/tileRows 保护 tail | 不得添加 cast；只写有效 GM extent | `Dim*Compute`、`RunTwoDim/Run/RunNddma/GatherRows` |
| 失败边界 | Host 显式拒绝非法列表、stride0、mask冲突、非连续和不支持 dtype | 生产尚未拒绝 `INT64_MIN` 的 `-step` overflow，也未闭合 folded int64、SIMT int32 乘积及 gather descriptor 窄化；安全实现须在 cast/launch 前 checked 计算，按 R-001→R-006→Host error 的可表示性顺序失败关闭 | `NormalizeAxis/NegativeSliceLength/BuildSlicePlan/MakeConfig/DispatchStridedSlice` |
