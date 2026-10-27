---
type: Operator
title: "MoeGatingTopKSoftmax 950 Direct Invoke 算子知识"
description: "说明 calc_moe_gating_top_k_softmax_tiling 与三个 dtype 直调 launcher 在 950 上的接口语义、四条 active Route、三个 Device TopK 子路线、POD/checked arithmetic、单 AIV Kernel 启动图与 RegBase 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_kernel/moe_gating_top_k_softmax_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_tiling_def.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_host/moe_gating_top_k_softmax_tiling_arch35.cpp
    role: implementation
created_at: "2026-08-25T19:39:15Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MoeGatingTopKSoftmax 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `calc_moe_gating_top_k_softmax_tiling`；`launch_moe_gating_top_k_softmax_{float,half,bfloat16}`，Direct Invoke |
| 计算语义 | 对最后一维专家 logits 做稳定 softmax，再按行取 top-k；同时给出 expert index 和展平 row index |
| 输入输出 | `x:[R,E]` 或把更高维前缀展平为 `R`，`finished:[R]` 可选；`y/expertIdx/rowIdx:[R,k]` |
| 执行类型 | 单 Kernel：`AIV_VECTOR`；entry 在 `AIC` 上立即返回 |
| Kernel 路线 | 四条 key Route 均为 `active`：P0 R-001 无 finished/对齐；P1 R-002 有 finished/对齐；P2 R-003 无 finished/补齐；P3 R-004 有 finished/补齐。每条再进入 ReduceMax、Sort32 或通用 Sort 三个互斥 TopK 子路线之一。 |
| 核心资源 | GM、5 个 depth=1 `TQue`、4 个 `TBuf`、MicroAPI 寄存器；Direct Invoke 不分配 workspace |
| 来源闭包 | `direct_implementation_closed`：官方 950 Host tiling、15 字段 POD、四 key router、RegBase Device body 已闭合；Direct 胶水只显式化 dtype/key launcher、平台查询并去掉未消费的保留 workspace |
| 证据边界 | 固定 HTTP 源码支撑官方 950 算法/POD/Host 事实；普通 POD、显式 entry、launcher、Torch 绑定和 workspace 省略是 canonical Direct 交付的转换事实，不提升为官方原生实现。 |

## 接口与语义

### 计算语义

令输入展平为 `x[r,e]`，`0<=r<R`、`0<=e<E`。Device 对每行先在 FP32 中计算：

```text
m[r] = max_e x[r,e]
p[r,e] = exp(float(x[r,e]) - m[r]) / sum_j exp(float(x[r,j]) - m[r])
(y[r,t], expertIdx[r,t]) = 第 t 个降序 top-k (p[r,:], expert id)
```

`float16`/`bfloat16` 输入在加载后转 FP32；softmax、排序值和归约保持 FP32，写 `y` 时再以 `CAST_RINT` 降回输入 dtype。`float32` 不发生边界 cast。`E` 非 32 倍数时，补齐 lane 在求和和排序前被写成 `0.0f`；它不增加 softmax 分母，也不会压过正的有效概率，零值 tie 的边界见下文。

`rowIdx` 由 `ArithProgression(rowIdxBase, 0, rows, k)` 生成长度为 `k`、步长为 `rows` 的基础序列，再加当前展平行号 `r=blockIdx*perCoreRowCount+loop*perLoopRowCount+j`，故精确契约是 `rowIdx[r,t]=t*R+r`。它等价于 `arange(R*k).reshape(k,R).transpose(0,1)`，不能改写成逐行重复的 `[r,r,...]`。若传入 `finished[r]=true`，`y` 与 `rowIdx` 不变，所选 expert index 通过 `max(index,E)` 变为哨兵 `E`。

对齐字段必须区分“余数”和“补齐数”：`q=E%32`、`A=ceil_div(E,32)*32`、`pad=A-E`，POD 的 `padNegInfCount=q`，并不是 `pad`。非对齐 Route 先把整个 `softmaxBuf` 清零，再仅向有效 E lane 写 `exp(x-max)`；尾 32-lane 块又用由 q 构造的 mask 写零。有限且未下溢的有效 softmax 值为正，因此零 lane 排在有效值之后；极端有限值造成 FP32 exp 下溢为零、NaN/Inf 或相等值时，源码没有额外的 pad/tie/特殊值裁决，必须保留官方 Sort/ReduceMax 顺序，不能把“补零”外推成任意输入下的严格负无穷哨兵。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 连续，最后一维为 `E`；Direct 胶水接受 2D/3D 并展平前缀为 `R` | `float16`/`bfloat16`/`float32` | `R>0`，`1<=E<=2048` |
| `finished` | 可选连续 `[R]`，对应 `x` 的前缀 shape | `bool` | 缺省为空；存在时选择 `hasFinished=true` 模板，并把该行 expert index 置为 `E` 哨兵 |
| `k` | Host 标量 | `int64` | `1<=k<=min(E,1024)` |
| `y` | 与 `x` 相同前缀，末维为 `k` | 与 `x` 相同 | top-k softmax 值 |
| `expertIdx` | 与 `y` 相同 | `int32` | top-k expert id；finished 行为哨兵 `E` |
| `rowIdx` | 与 `y` 相同 | `int32` | Device 生成的展平 row 索引布局 |

### 入口与版本

固定版本为 CANN 9.1.0、950/Ascend 950。官方固定提交没有同名的 Direct Invoke launcher 文件；转换从 `moe_gating_top_k_softmax` 官方 entry、`MoeGatingTopKSoftmaxRegbaseTilingData`、`MoeGatingTopKSoftmaxRegbaseTiling` 和 `MoeGatingTopKSoftmaxFullloadGenerlized` 派生：

- 官方宏解析的 tiling POD 被等字段、等顺序的普通 POD 替代；
- 官方 `DTYPE_X × 4 key` router 被展开为 `3 dtype × 4 bool specialization` 的 12 个显式 entry，再由三个 dtype launcher 选择；
- `gert` 平台信息改由 `PlatformAscendCManager` 提供 AIV 数和 UB 字节数；
- 官方 Host 预留 16 MiB workspace，但官方 `Init/Process` 不读取 `workspace`，因此 Direct Invoke 的参数和启动图中没有 workspace。

除把两个跨目录公共 include 换成其命中符号的局部闭包外，官方 Kernel 类及成员函数保持字节一致；Host 的分核、UB row split、`E/k` 对齐、padding 和 key 公式保持同一运算顺序。

这是源码派生边界，不把 Direct Invoke 伪装成公开 ACLNN/API。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 调用侧检查 `x` 为连续 2D/3D 且 dtype 为 FP16/BF16/FP32；检查 `finished`（若有）为同设备、连续 bool，shape 等于 `x.shape[:-1]`。
2. 取 `E=x.size(-1)`、`R=x.numel()/E`，检查 `R>0`、`1<=E<=2048`、`1<=k<=min(E,1024)`。
3. 为三个输出建立 `x.shape[:-1]+[k]` 的连续 GM 张量。
4. `calc_moe_gating_top_k_softmax_tiling(R,E,k,hasFinished,&params)` 查询 `aivNum` 和 `ubSize`；空 `params`、非法 shape、平台句柄为空、`aivNum<=0`、`ubSize==0` 或 UB 放不下一行时返回 `false`。
5. 从当前 NPU stream 取裸 stream；按 `x` dtype 调用对应 launcher。launcher 再按 `tilingKey` 选择 bool specialization。

### Shape 折叠与 checked arithmetic 边界

生产代码只显式检查 rank、dtype、连续性、`0<E<=2048`、`0<k<=min(E,1024)` 和 `R>0`，其余计算使用有符号 `int64_t`，没有通用 checked-add/mul。Torch 的 `numel()` 与输出分配是前置约束，但不能替代 Direct ABI 的整数证明。迁移或独立调用 calculator 前必须先验证：

```text
R = numel(x) / E，且 numel(x) % E == 0
0 < C=aivNum <= UINT32_MAX
ceil_div(R,C)、needCore、各 loop/row 字段均在 int64_t 正范围
R*E、R*k、perCore*E、perCore*k 及所有 loopBase/rowBase 乘积不溢出 int64_t
R <= INT32_MAX                             # progression 的 rows 显式窄化
R*k-1 <= INT32_MAX                         # rowIdx=t*R+r 的最大值
A=align_up(E,32)，K8=align_up(k,8)
UB 公式中的 A/K8 乘加不溢出，且 ubSize<=INT64_MAX
0 < needCore <= UINT32_MAX                 # launcher 的 blockDim 窄化
perLoopRows <= UINT16_MAX，loopCount <= INT_MAX
DMA 的 blockCount/blockLen/总连续字节适配其 uint16/uint32 字段
```

当前声明的 E/k 上限使 A/K8 自身安全，但源码没有约束 R，也没有检查上述窄化。`InitIndices` 显式把 rows 转为 int32，`ComputeRowIdx` 把行基址落到 int32，多个 `rowLoops`/DMA blockCount 又落到 uint16；因此过大的 R、异常大的平台 UB 或由其导出的 row tile 都是未拒绝的 ABI 边界，不能宣称 calculator 对任意正 int64 rows 都安全。ceil-div 也应使用 `a/b + (a%b!=0)` 或 checked-add，避免照搬 `(a+b-1)/b` 的潜在溢出。

### 参数、Workspace 与生命周期

```text
C = AIV core 数
A = align_up(E, 32)
K8 = align_up(k, 8)
perCore = ceil_div(R, C)
blockDim = needCore = ceil_div(R, perCore)，且 needCore <= min(R,C)
lastCore = R - (needCore-1)*perCore
```

Host 的 UB 保守计账为：

```text
rowBytes = (4*A + 32) + 12*K8 + 12*A + 16*A
fixedBytes = 8*A
maxRowsInUb = floor((ubSize - fixedBytes) / rowBytes)
```

随后对普通 core 的 `perCore` 与尾 core 的 `lastCore` 分别执行：

```text
loopCount = ceil_div(coreRows, maxRowsInUb)
perLoopRows = ceil_div(coreRows, loopCount)
lastLoopRows = coreRows - (loopCount-1)*perLoopRows
```

`params` 由 120 字节 tiling POD、8 字节 `tilingKey`、8 字节 `blockDim` 组成。它在 Host 栈上生成，launcher 在入队前按值复制 tiling POD 给 Kernel 参数；Kernel 接收 `const MoeGatingTopKSoftmaxRegbaseTilingData` 按值参数，不持有 Host 栈地址。`TPipe` 和所有 UB 资源只在单次 Kernel block 生命周期内有效，`workspace_total=0`。

GM owner 是另一层契约。lambda 的 `[=]` 因读取 `x.scalar_type()` 而复制一个 `x` Tensor handle，但只保证该 handle 活到 `RunOpApi` 返回；`finished` Tensor handle 没有被捕获，lambda 只保存 `finishedPtr`，三个输出也只以裸指针传入。源码本身因此只证明 launch 提交期 owner，不证明异步 Device 完成期 owner。当前 Torch/allocator 集成必须在外部保证全部 storage 在当前 stream 到达 K0 前不回收；裸 launcher 的调用者则必须持有 GM allocation 直到 stream 同步或完成 event。

可复用的 owner 闭包有三种等价形态：在异步命令对象中捕获 `x/finished/y/expertIdx/rowIdx` Tensor handle 并持有到 completion；或对五个 storage 执行当前 stream 的 allocator `record_stream` 等价登记；或把这些 owner 绑定到 completion event/callback 后再释放。只捕获原始地址、只延长到 `RunOpApi` 返回，或假设输出返回值能连带保活输入，均不构成 Device 生命周期证明。launcher 自身也不做空指针、key、grid 或 dtype 复核。

### 启动图与失败传播

```text
Direct caller
  -> calc_moe_gating_top_k_softmax_tiling
  -> dtype launcher(params, stream)
     -> 单个 AIV entry<<<blockDim, nullptr, stream>>>(x, finished, y, expertIdx, rowIdx, tilingData)
        -> op.Init(...)
        -> op.Process()
```

| 节点/边 | entry/介质 | 范围与布局 | ordering / owner / lifetime | 失败传播 |
|---|---|---|---|---|
| Host tiling | `calc_moe_gating_top_k_softmax_tiling` | 生产 key、`blockDim`、15 字段 POD | Host 同步完成后才 launch；调用者持有 `params` | `false` 阻断 launch |
| Kernel K0 | 12 个 dtype/bool/pad 显式 entry 之一 | 单 AIV Kernel；每个 block 覆盖连续 row 段 | 当前 stream 入队；GM completion owner 由调用方/allocator 契约提供，UB 由 K0 持有 | direct wrapper 无 Device 返回码；launch/runtime 错误由上层运行时处理 |
| K0 输出边 | `y/expertIdx/rowIdx` GM | 均为 `[R,k]` 连续布局 | 同一 Kernel 内 Queue→GM；completion owner 保持到 stream 越过 K0 | 无后继 Kernel |
| workspace | N/A | Direct Invoke 没有中间 slice | N/A | N/A |

`calc...` 返回 `false` 时 Torch 的 `TORCH_CHECK` 在 launch 前终止。正常 launch 后，lambda 固定返回 0；Kernel 没有 Device 状态输出，也没有 CPU、ACLNN 或第二个 Kernel fallback，异步 launch/runtime 故障只能由上层 stream/runtime 观察。各 AIV block 写互不重叠的 GM row 区间，无跨核生产者/消费者，也无跨核 barrier。

## Route 与参数设计

### 分核与资源计算

四条 Route 共用相同分核和行切片公式。`GetBlockIdx()` 决定 core；最后一个 block 读 `lastCore*` 字段，其余 block 读 `perCore*` 字段。GM row 基址分别是 `blockIdx*perCoreRowCount*E` 和 `blockIdx*perCoreRowCount*k`。每次循环的有效 `rowCount` 是普通 loop 的 `perLoopRowCount` 或最后 loop 的 `lastLoopRowCount`。

Host 按 FP32 最坏输入容量计账，故 FP16/BF16 的 `xInQueue` 实际字节更少。所有 TQue depth 都是 1；Kernel 没有 double buffer，也没有跨 Kernel workspace。

### Route 分类、重要性与选择

四个 key 组合都能由 Torch 生产链的合法输入触发，因而都是 `active`，没有只靠外部 caller 补 POD 才成立的 `conditional` key Route。P0→P3 是维护顺序，不改变稳定 Route ID；排序先保留生产 selector 的短路顺序，再考虑默认无 finished 的较短数据边、对齐 Route 的更小尾处理和 finished Route 的语义附加价值。四个输入域互斥，calculator 一次直接产出唯一 key，不会在某条 Route 执行失败后重试下一条。

| Host顺序 | 重要性 | Route | 可达性 | 特征 | 适用域 | 实现前提 | Key/模板 | 选择结果与失败出边 |
|---:|---|---|---|---|---|---|---|---|
| 1 | P0 | R-001 无 finished/对齐 | active | 最短数据路：不搬 `finished`，不清 pad lane | `finished` 缺省，`E%32==0`；三种输入 dtype | calculator 生成合法 POD；`A=E`；实例化 `<T,false,false>` | `100000`；`<T,false,false>` | 首个 `if` 命中；前置校验/UB 失败则不 launch，无 Device fallback |
| 2 | P1 | R-002 有 finished/对齐 | active | 在 R-001 上增加 bool 搬运与 expert 哨兵合并 | `finished` 存在且 shape/dtype 合法，`E%32==0` | finished storage 的 completion owner 覆盖 K0；实例化 `<T,true,false>` | `100001`；`<T,true,false>` | 第一个 `else if` 命中；前置校验/UB 失败则不 launch，无 Device fallback |
| 3 | P2 | R-003 无 finished/补齐 | active | 不搬 `finished`，但对 softmax 尾 lane 清零 | `finished` 缺省，`E%32!=0` | `A=align_up(E,32)`；pad mask 精确覆盖 `[E,A)`；实例化 `<T,false,true>` | `100010`；`<T,false,true>` | 第二个 `else if` 命中；失败不退到对齐 body |
| 4 | P3 | R-004 有 finished/补齐 | active | 同时启用 bool 哨兵与 pad lane 处理 | `finished` 存在，`E%32!=0` | 同时满足 R-002 的 completion owner 和 R-003 的 pad mask 前提；实例化 `<T,true,true>` | `100011`；`<T,true,true>` | 合法 key 到最终 `else`；前置失败则不 launch，不退到其它 body |

```text
校验失败 -> 返回 false，阻断 launch
key = 100000
if hasFinished: key += 1                  # 独立判断，不短路下一判断
if E % 32 != 0: key += 10                 # 独立判断

if dtype == float32:
  float launcher
else if dtype == float16:
  half launcher
else:
  bfloat16 launcher                       # dtype 已在入口校验

launcher 内：
if key == 100000: R-001，短路返回该 entry
else if key == 100001: R-002，短路返回该 entry
else if key == 100010: R-003，短路返回该 entry
else: R-004 entry                         # 生产 key 100011 的合法终点；未知 key 也会误落此 sink
```

### Device TopK 子路线

key Route 进入同一 `ComputeTopK` 后再按 A/E/k 形成三个互斥、生产可达的 `active-subroute`。它们不新增 tiling key 或 POD；排序价值按更少工作量到通用覆盖排列。每个 key Route 恰好与其中一个子路线组合，因此完整实现域是“四个 key Route × 一个 TopK 子路线 × 一个 dtype 实例”，不是另有未接线的 Kernel。

| 子路线 | 重要性 | 精确谓词 | 入口/body | 核心实现与 tail | 不满足时出边 |
|---|---|---|---|---|---|
| S-001 top1 ReduceMax | SP0 | `A==32 && k==1 && E<=8` | `HandleOneRepeatSortNum` | 每行读取一 repeat，以有效 E mask 做 `ReduceMax`，按 `VL2` value/index 交错写 `sortedTensor`；输出阶段只取 1 项 | `E>8` 或 `k>1` → S-002；`A>32` 在外层直接走 S-003 |
| S-002 Sort32 | SP1 | `A==32 && !(k==1 && E<=8)` | `HandleOneRepeatSortNum` → `Sort32` | 以 `softmaxTensor` 和预建 `expertIdxTensor` 排 32 lane；后续按 k mask 解交错 | `A>32` → S-003 |
| S-003 通用 Sort | SP2 | `A>32` | `ComputeTopK` 行循环 → `Sort<float,true>` | 每行 `sortRepeatTimes=A/32`，共享 `sortTmpTensor`，结果写 value/index 交错区；按行程序序复用 scratch | 无更慢 body；POD/UB 不合法时 Host 已拒绝 |

```text
if A == 32:
  if k == 1 and E <= 8: ReduceMax(valid E) -> interleaved(value,index)
  else:                 Sort32(softmax[0:32], expertId[0:32])
else:
  for row in tile: Sort<float,true>(repeat=A/32, sharedScratch)
extract first k interleaved pairs with lastLoopKCount mask
```

S-001/S-002 可与 R-003/R-004 组合（`E<32` 且非对齐）；R-001/R-002 在 `E==32` 时只能走 S-002，在 `E>=64` 时走 S-003。R-003/R-004 的 `E>32` 非对齐域走 S-003。dtype 只改变 GM load/store cast，不改变子路线选择。

### 模板/实现执行矩阵

| 重要性 | Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|---|
| P0 | R-001 | `100000` | `T∈{float,half,bfloat16}`，无 finished，`E%32==0`，950 | K0 单 AIV | `MoeGatingTopKSoftmaxFullloadGenerlized<T,false,false>::Process` | finished Queue 分配但不读；Queue 同步 | E 无 pad；row/k 用 mask 与 `lastLoopKCount`；校验失败则不 launch |
| P1 | R-002 | `100001` | 同上但有 finished | K0 单 AIV | `<T,true,false>` | finished GM→Queue→寄存器；索引取 `max(idx,finished*E)` | E 无 pad；校验失败则不 launch |
| P2 | R-003 | `100010` | 无 finished，`E%32!=0` | K0 单 AIV | `<T,false,true>` | 对齐 lane 在 softmax UB 清零 | `DataCopyPad` + pad mask；校验失败则不 launch |
| P3 | R-004 | `100011` | 有 finished，`E%32!=0` | K0 单 AIV | `<T,true,true>` | R-002 与 R-003 的编译期路径同时启用 | finished 哨兵 + pad mask；未知 key 落此 entry 属于 dormant sink，不扩大合法域 |

每一行内部有 float/half/bfloat16 三个编译实例；Route key 和数据流不因 dtype 改变，差异仅在 GM↔FP32 的编译期 load/store cast。

### 性能与安全 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 RegBase | 官方 `IsRegbaseSocVersion(context_)`；Direct 平台需返回 AIV/UB | body 使用 `__VEC_SCOPE__`、MicroAPI 与 RegTensor | 全部 | Host 失败，不 launch | `MoeGatingTopKSoftmaxRegbaseTiling::IsCapable`；`calc_moe_gating_top_k_softmax_tiling` |
| expert 上限 | `0<E<=2048` | Host 明确限制，排序 repeat 与 UB 随 E 增长 | 全部 | `false` | `DoOpTiling`；direct tiling calculator |
| UB 至少一行 | `ubSize > 8*A` 且 `maxRowsInUb>0` | 5 Queue、4 TBuf 必须同时初始化 | 全部 | `false` | `CalcMaxRowCountInUb`；`SplitRowInUb` |
| E 的 32 对齐 | `E%32==0` / `!=0` | `Sort32`/`Sort` 消费 `A=align_up(E,32)` | R-001/2 vs R-003/4 | 切 padding Route | `GetTilingKey`；`ComputeSoftmax` |
| 单 repeat sort | `A==32` | 可用 `Sort32`；更大 A 按行调用通用 `Sort` | 全部内部子支 | `A>32` 用通用 Sort | `ComputeTopK` |
| top1 reduce | `k==1 && E<=8 && A==32` | `ReduceMax` 直接产生 value/index，跳过完整排序 | 全部内部子支 | `Sort32` | `HandleOneRepeatSortNum` |
| k tail | `k % B32_VF_COUNT` | unaligned store 只提交有效 top-k | 全部 | `lastLoopKCount=B32_VF_COUNT`（整除）或余数 | `ComputeTopK` |
| core type | `g_coreType != AIC` | 算子仅有 AIV body | 全部 | AIC 立即返回 | 官方 entry 与 Direct entry body |
| launcher key | 只接受 tiling calculator 生成的四 key | direct wrapper 最后 `else` 会落入 finished+pad entry | 全部 | 禁止手工构造未知 key | dtype launcher wrapper |

### Conditional 与 dormant 分离

- `conditional key Route`：无。三个 dtype、finished 有无、E 对齐与否都由公开 Torch 绑定和 calculator 完整生产，不需要调用者伪造 POD。
- `D-001 AIC no-op`：12 个 entry 都先检查 `g_coreType==AIC` 并立即返回。生产 grid 来自 AIV 数，计算 body 只在 AIV 执行；这个分支不读写输出，不是 Cube Route。只有重新设计 AIC/AIV 混合 launch、给 AIC 分配任务并闭合跨核同步后，才可新增 AIC 路线。
- `D-002 unknown-key sink`：launcher 的最终 `else` 会把任何非 100000/100001/100010 key 送到 `*_f_p`。生产 calculator 只生成四个合法值，因此未知 key 在生产链不可达；它也不是安全 fallback，因为 POD、finished 指针与 padding 字段可能不匹配。恢复为可消费路径的前提是 launcher 显式验证 key 并返回错误，或为新 key 增加完整 Host producer、POD 和 body。
- 官方 Host 的保留 workspace 在 Direct 参数和 launch 图中被删除，Device body也没有 workspace 形参或 GM 访问；它是转换差异，不是 dormant 数据路线。

### POD/ABI

`MoeGatingTopKSoftmaxRegbaseTilingData` 是 15 个连续 `int64_t`：`sizeof=120`、`alignof=8`。Direct 普通 POD 保持官方字段名、类型与顺序。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| tiling POD | `rows` | `int64_t/64` | offset 0 | row | `rows=R` | `InitIndices` 的 progression | `0<R<=INT32_MAX`，且 `R*k-1<=INT32_MAX` |
| tiling POD | `perCoreRowCount` | `int64_t/64` | offset 8 | row/core | `ceil_div(R,C)` | `Init` 普通 core 与 GM 基址 | 正数 |
| tiling POD | `lastCoreRowCount` | `int64_t/64` | offset 16 | row | `R-(needCore-1)*perCore` | 最后 block 的 `Init` | 正数 |
| tiling POD | `perCoreLoopCount` | `int64_t/64` | offset 24 | loop | `SplitRowInUb(perCore)` | 普通 block 的 `Process` 循环 | >=1 |
| tiling POD | `lastCoreLoopCount` | `int64_t/64` | offset 32 | loop | `SplitRowInUb(lastCore)` | 最后 block 的循环 | >=1 |
| tiling POD | `perCorePerLoopRowCount` | `int64_t/64` | offset 40 | row/loop | 普通 core `ceil_div(coreRows,loops)` | 普通 block UB 容量与 loop | <=maxRowsInUb |
| tiling POD | `perCoreLastLoopRowCount` | `int64_t/64` | offset 48 | row | 普通 core 尾式 | 普通 block 最后 loop | 1..perLoop |
| tiling POD | `lastCorePerLoopRowCount` | `int64_t/64` | offset 56 | row/loop | 尾 core split | 最后 block UB 容量与 loop | <=maxRowsInUb |
| tiling POD | `lastCoreLastLoopRowCount` | `int64_t/64` | offset 64 | row | 尾 core 尾式 | 最后 block 最后 loop | 1..perLoop |
| tiling POD | `expertCount` | `int64_t/64` | offset 72 | element/row | `E` | GM stride、mask、finished 哨兵 | 1..2048 |
| tiling POD | `k` | `int64_t/64` | offset 80 | element/row | Host 属性 | output stride、mask、top1 guard | 1..min(E,1024) |
| tiling POD | `expertCountAlign` | `int64_t/64` | offset 88 | FP32 element/row | `align_up(E,32)` | UB stride、Sort/Sort32 | 32 对齐 |
| tiling POD | `kAlign` | `int64_t/64` | offset 96 | FP32/int32 element/row | `align_up(k,8)` | output Queue stride | 8 对齐 |
| tiling POD | `padNegInfCount` | `int64_t/64` | offset 104 | remainder | `q=E%32` | `duplicateIndex=E-q`、mask 起点和 key | 0..31；真实补齐数是 `A-E` |
| tiling POD | `sortRepeatTimes` | `int64_t/64` | offset 112 | 32-element repeat | `A/32` | 通用 `Sort<float,true>` | 1..64 |
| launch params | `tiling` | nested POD | offset 0；size 120 | bytes/POD | direct tiling calculator | launcher 按值复制给 entry | 生命周期覆盖 launch 调用 |
| launch params | `tilingKey` | `uint64_t/64` | offset 120 | key | `100000+[hasFinished]+10*[q>0]` | dtype launcher | 仅四合法值 |
| launch params | `blockDim` | `int64_t/64` | offset 128；struct size 136 | AIV block | `needCoreNum` | launch grid | 转 `uint32_t` 前必须为正且可表示 |

### Route 源码闭包

| 重要性 | Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|
| P0 | R-001 | `100000` | `DoOpTiling/GetTilingKey/PostTiling` → direct calculator | 15 字段 POD，`false,false` | 官方 key branch → direct `*_nf_np` | `MoeGatingTopKSoftmaxFullloadGenerlized<T,false,false>`；load/store helper | 官方 APT entry；Direct `register_direct_launch_op` 派生胶水 |
| P1 | R-002 | `100001` | 同上，`hasFinished=true` | 同 POD，`true,false` | 官方 key branch → direct `*_f_np` | `<T,true,false>` | 同上 |
| P2 | R-003 | `100010` | 同上，`padNegInfCount>0` | 同 POD，`false,true` | 官方 key branch → direct `*_nf_p` | `<T,false,true>` | 同上 |
| P3 | R-004 | `100011` | 两条件都成立 | 同 POD，`true,true` | 官方 key branch → direct `*_f_p` | `<T,true,true>` | 同上 |

## Kernel 实现

### 公共执行流程

四条 Route 都是单 AIV Kernel：`Init` 解析 POD、绑定每核 GM 区间并一次性初始化 Queue/TBuf；`Process` 先用 `InitIndices` 建 expert/row 基础索引，然后逐 UB row tile 严格执行：

```text
ComputeRowIdx -> CopyOutRowIdx -> CopyInX[/finished]
-> ComputeSoftmax(max, exp, ReduceSum, div)
-> ComputeTopK(Sort/Sort32/ReduceMax, finished sentinel)
-> CopyYExpertIdxOut
```

Queue 的 `AllocTensor/EnQue/DeQue/FreeTensor` 提供阶段可见性和 depth=1 复用顺序。softmax 和排序在同一个 Kernel 内通过 TBuf 交接；没有跨 Kernel GM/workspace 边。

核与任务映射是“一个 AIV block 拥有一段连续 row，block 内按 UB 容量循环多个 row tile”。softmax 的 max/exp/div 和 top-k 抽取在 `__VEC_SCOPE__` 中以 `B32_VF_COUNT=VRegBytes/4` lane 分段；通用 Sort 是当前 block 内逐行调用，不产生跨 block 合并。GM 地址关系为：

```text
coreBase = blockIdx * perCoreRowCount
rowBase  = coreBase + loop * perLoopRowCount
xGM      = x + rowBase*E
finishedGM = finished + rowBase                 # 仅 hasFinished=true 时搬运
yGM/expertGM/rowGM = output + rowBase*k
```

`DataCopyPad` 的 x `blockLen=E*sizeof(T)`，额外 `dstStride=(A-E)*sizeof(T)/32` 个 32B block；结合每行 blockLen 的 32B 向上取整，UB 物理行跨度恰为 `A*sizeof(T)`。finished 每行逻辑 1B、物理占一个 32B block。y/expert Queue 虽各预留 `4*K8*n` 容量，`DataCopyUnAlign/Post` 的目标指针却按每次有效 k 连续推进，所以 live 数据紧密占据前 `n*k` 项，并分别用一个 `blockCount=1, blockLen=n*k*sizeof(T/int32)` 搬出；只有 rowIdx 按 `row*K8+t` 写 UB，再逐行跳过 `K8-k` 尾槽搬出。该实现没有 L1、L0A/L0B/L0C、Cube、动态 UB 或 workspace；不得为这些不存在的层级虚构 producer/consumer。

`Init` 无条件建立 `finishedGm_` view；finished 缺省时 Torch 传 `nullptr`，但 `<T,false,*>` 的 `if constexpr` 会消除 `CopyInX` 和 `ComputeTopK` 中的 finished GM/Queue 消费。因而“无 finished”契约只允许保留未消费的 view/Queue，不能把 bool load 从编译期分支外提；未知 key 把空指针送入 `<T,true,true>` 正是 D-002 不安全的原因。

### 公共 body、资源与 R-001（P0）基线

#### 适用条件

| 项目 | 内容 |
|---|---|
| 特征 | 最短数据路；`finished` 物理 Queue 保留，但模板不搬运、不读取；E 无 pad 清零 |
| 适用域 | `finished` 缺省，`E%32==0`，`T∈{float,half,bfloat16}` |
| 实现前提 | calculator 生成 key `100000` 与合法 POD；launcher 首个分支命中 `<T,false,false>` |

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/moe_gating_top_k_softmax_tiling_arch35.cpp` | `GetTilingKey` | `!hasFinished && padNegInfCount_==0` | `100000` |
| Router | `op_kernel/moe_gating_top_k_softmax_apt.cpp` | `moe_gating_top_k_softmax` | key `100000` | `<DTYPE_X,false,false>` |
| Direct launcher | 派生 launcher | `launch_moe_gating_top_k_softmax_{dtype}` | key `100000` | `moe_gating_{dtype}_nf_np` |

#### Device 数据流

##### 函数级调用链

```text
dtype launcher -> *_nf_np(x,null,y,expertIdx,rowIdx,tilingData)
  -> MoeGatingTopKSoftmaxFullloadGenerlized<T,false,false>::Init
  -> Process
     -> InitIndices
     -> [loop] ComputeRowIdx -> CopyOutRowIdx
     -> CopyInX -> ComputeSoftmax -> ComputeTopK -> CopyYExpertIdxOut
```

##### 源码等价伪码

```text
coreBase = blockIdx * perCoreRowCount
for loop in [0, coreLoopCount):
  n = (last loop) ? lastLoopRows : perLoopRows
  rowIdxUB[j,t] = t*R + coreBase + loop*perLoopRows + j
  GM.rowIdx[loopBase : loopBase+n*k] <- rowIdxUB[:,0:k]
  xUB[j,0:E] <- GM.x[coreBase+loop*perLoopRows+j,:]
  for each row: max <- ReduceMax(x); expUB <- exp(float(x)-max)
  sum <- ReduceSum(expUB[0:A]); softmaxUB <- expUB/sum
  if A==32: (k==1 && E<=8 ? ReduceMax : Sort32)
  else: Sort<float,true>(repeat=A/32)
  extract first k interleaved (value,index), cast value to T if needed
  GM.y, GM.expertIdx <- contiguous valid n*k
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `xGm` | `coreBase*E + loop*perLoopRows*E` | `n*E` | T，row-major | caller | `CopyInX` | caller/stream | DMA pad 到 A |
| `xInQueue` | UB queue | `A*sizeof(T)*Rloop` | T | `CopyInX` | `ComputeSoftmax` | TPipe/一次 loop | depth 1 |
| `finishedInQueue` | UB queue | `32*Rloop` | bool block | 未写 | 未读 | TPipe/Kernel | 条件路径关闭但物理分配保留 |
| `softmaxBuf` | UB TBuf | `4*A*Rloop` | FP32 | `ComputeSoftmax` | `ComputeTopK` | TPipe/Kernel | exp 与归一结果原地复用 |
| `calcTmpBuf` sorted 区 | offset 0 | `8*A*Rloop` | value/index 交错 | Sort | TopK extract | TPipe/loop | 与 reduce 临时区分时复用 |
| `calcTmpBuf` sort tmp | `2*A*n` FP32 element | `16*A` bytes | FP32 scratch | Sort | Sort | TPipe/loop | 固定尾区 |
| `expertIdxBuf` | UB TBuf | `4*A*Rloop` | int32 | `InitIndices` | Sort | TPipe/Kernel | 同物理容量按 float 计 |
| `rowIdxBaseBuf` | UB TBuf | `4*A*Rloop` | int32 | `InitIndices` | `ComputeRowIdx` | TPipe/Kernel | 每 loop 只读 |
| `yOutQueue` | UB queue | 容量 `4*K8*Rloop`；live `n*k*sizeof(T)` | T，跨 row 连续 | TopK extract | 单 block CopyOut | TPipe/loop | depth 1；容量按 FP32 保守分配 |
| `expertIdxOutQueue` | UB queue | 容量 `4*K8*Rloop`；live `4*n*k` | int32，跨 row 连续 | TopK extract | 单 block CopyOut | TPipe/loop | depth 1 |
| `rowIdxOutQueue` | UB queue | 容量 `4*K8*Rloop`；live 行跨度 K8 | int32，`row*K8+t` | `ComputeRowIdx` | 逐行 CopyOut | TPipe/loop | depth 1；跳过 `K8-k` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xInQueue d=1 · A·sizeof(T)·Rloop"]:2
  fq["finishedInQueue d=1 · 32·Rloop"]:2
  yq["yOutQueue cap 4·K8·Rloop · live n·k contiguous"]:1
  eq["expertIdxOutQueue cap 4·K8·Rloop · live n·k contiguous"]:1
  rq["rowIdxOutQueue cap 4·K8·Rloop · row stride K8"]:2
  ebuf["expertIdxBuf · 4·A·Rloop"]:1
  rbuf["rowIdxBaseBuf · 4·A·Rloop"]:1
  sbuf["softmaxBuf · 4·A·Rloop"]:2
  cbuf["calcTmpBuf · 8·A·Rloop + 16·A"]:4
```

##### 字节账与复用顺序

```text
UB_actual(T) = A*sizeof(T)*Rloop + 32*Rloop
             + 3*(4*K8*Rloop)
             + 3*(4*A*Rloop)
             + (8*A*Rloop + 16*A)
UB_actual(float) = Rloop*(24*A + 12*K8 + 32) + 16*A
UB_host_bound = Rloop*(32*A + 12*K8 + 32) + 8*A
UB_actual(float) <= UB_host_bound  当 Rloop>=1；Rloop=1 时相等
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `softmaxBuf` | max-exp 阶段写 FP32 exp | ReduceSum、Div、Sort | 下一 loop 的 Duplicate/exp | 同函数顺序；x Queue 在末尾 Free | Sort 完成后才进入下一 loop |
| `calcTmpBuf` 前区 | ReduceSum/Brcb 临时，再由 Sort 写 interleaved 结果 | Div、TopK extract | 下一 loop | 单线程程序序；Sort 完成后输出已入 Queue | sort scratch offset 不得侵入前区 |
| output Queue | Compute/TopK | CopyOut | `FreeTensor` 后下个 loop | EnQue→DeQue→FreeTensor | depth=1，不能跨 loop 提前 Alloc |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + `EnQue` | `CopyInX` | `ComputeSoftmax::DeQue` | x Queue | 每行 `blockLen=E*sizeof(T)`，dst stride 补到 A |
| softmax | MicroAPI mask、`ReduceMax/Exp/ReduceSum/Brcb/Div` | x Queue | Sort | softmax TBuf | FP16/BF16→FP32；mask 限制 E；A 对齐无 pad Route |
| top-k | `ReduceMax`/`Sort32`/`Sort<float,true>` | softmax TBuf | unaligned extract | calc TBuf | k 尾由 mask 与 `lastLoopKCount` 控制 |
| UB→GM | Queue + `DataCopyPad` | output Queue | GM | 三路输出 | y/expert 连续 `n*k`；rowIdx 逐行去 K8 pad |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_gating_top_k_softmax_tiling_arch35.cpp` | `DoOpTiling/CalcMaxRowCountInUb/SplitRowInUb/PostTiling` | core、UB row tile、key、POD |
| Kernel entry/router | `op_kernel/moe_gating_top_k_softmax_apt.cpp` | `moe_gating_top_k_softmax` | key 100000 到 `<DTYPE_X,false,false>` |
| Device body/helper | `op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h` | `Init/Process/ComputeSoftmax/ComputeTopK` | GM/UB、FP32 softmax、sort、tail |

### 四 Route 的 finished/pad 差分闭包

R-001 上一节给出四 Route 共用的分核、Queue/TBuf、softmax、TopK、输出和同步 body。下面只保留两个编译期轴的真实差分；四个谓词互斥且穷尽合法输入域。

| Route | active 谓词 | key/模板/entry | finished 差分 | pad 差分 | fallback 与边界 |
|---|---|---|---|---|---|
| R-001（P0） | `!hasFinished && q==0` | `100000`；`<T,false,false>`；`*_nf_np` | Queue 物理分配但不搬运、不消费 | `A=E`，不执行尾清零 | Host 拒绝则不 launch；无 Device fallback |
| R-002（P1） | `hasFinished && q==0` | `100001`；`<T,true,false>`；`*_f_np` | bool GM→32B/row Queue；TopK 前转 int32 并取 `max(idx,finished*E)` | `A=E`，不执行尾清零 | finished storage 必须有 completion owner；失败不退 R-001 |
| R-003（P2） | `!hasFinished && q>0` | `100010`；`<T,false,true>`；`*_nf_p` | Queue 不搬运、不消费 | `A=align_up(E,32)`；exp 后把 `[E,A)` 清零再做 A-wide sum/sort | 零 tie 受公共精度边界约束；失败不退对齐 body |
| R-004（P3） | `hasFinished && q>0` | `100011`；`<T,true,true>`；`*_f_p` | 同 R-002 | 同 R-003 | 合法 key 走最终 `else`；未知 key 落此处是 D-002，不扩大合法域 |

#### Host producer、router 与 consumer

| Route | Host/Tiling producer | launcher selector | Device consumer/body | 构建节点 |
|---|---|---|---|---|
| R-001 | `key=100000`，两个增量均不加 | 首个 `if` → `*_nf_np` | `CopyInX/ComputeTopK` 的 finished 分支编译消除；`needPadNegInf=false` | 12 entry 与 3 dtype wrapper 由 `register_direct_launch_op` 接线 |
| R-002 | `hasFinished` 使 key `+1` | 第一个 `else if` → `*_f_np` | `CopyInX` 生产 finished Queue，`ComputeTopK` DeQue/Free 并写 E 哨兵 | 同上 |
| R-003 | `q=padNegInfCount>0` 使 key `+10` | 第二个 `else if` → `*_nf_p` | `ComputeSoftmax` 消费 q，构造 pad mask；finished 分支编译消除 | 同上 |
| R-004 | 两个增量叠加为 `100011` | 最终 `else` → `*_f_p` | 同时消费 finished Queue 与 q/pad mask | 同上 |

#### 公共 body 加差分的可落地伪码

```text
q = E % 32; A = align_up(E,32)
key = 100000 + int(hasFinished) + 10*int(q>0)
entry = dispatch(dtype,key) -> MoeGatingTopKSoftmaxFullloadGenerlized<T,hasFinished,q>0

for each owned row tile:
  rowIdx[r,t] = t*R+r; copy rowIdx by row from K8-strided UB
  copy x GM -> A-strided UB
  if hasFinished: copy finished GM -> 32B/row Queue
  p = exp(float(x)-reduce_max(float(x)))
  if q>0: p[E:A] = 0 using tail-block mask
  p /= reduce_sum(p[0:A])
  (value,index) = ReduceMax/Sort32/Sort(p, expertId)
  if hasFinished: index = max(index, int32(finished)*E)
  append first k values/indices to contiguous y/expert UB live regions
  copy y and expertIdx once each with blockLen=n*k; free all depth-1 Queues
```

#### 差分资源、同步与生命周期

| 差分对象 | Route | GM/UB extent 与布局 | producer→consumer | 同步/释放 | 安全条件 |
|---|---|---|---|---|---|
| `finishedGm/finishedInQueue` | R-002/R-004 | GM `rowBase+[0,n)` bool；UB `32*n`，32B/row | `CopyInX::DataCopyPad/EnQue` → `ComputeTopK::DeQue` | expert index 形成后 `FreeTensor`，下一 loop 才覆盖 | Tensor/storage completion owner 覆盖 K0；R-001/R-003 不得外提 bool load |
| `softmaxBuf[E:A)` | R-003/R-004 | 每行 `A-E` 个 FP32 lane | exp 后 masked `Duplicate(0)` → ReduceSum/Sort | 同函数程序序；下一 loop 覆盖 | q 是余数而非 pad 数；mask 精确覆盖 `[E,A)` |
| 两差分同时存在 | R-004 | UB 总容量不增加：finished Queue 已常驻分配，pad 已计入 A | 两条独立 producer/consumer 边在公共 body 汇合 | Queue 协议 + 程序序 | 未知 key 不能借此获得合法 POD/pointer 语义 |

四条 Route 的 `UB_actual(T)`、Host 保守上界、y/expert 连续 live 布局、rowIdx K8 布局、TopK 三子路线、row/core/k tail 和 `workspace_total=0` 均使用 R-001 公共公式；差分不会产生新 Kernel、L1/L0 或跨核同步。

## 跨 Route 精度、排序与尾部不变量

| 不变量 | 精确顺序/机制 | Route 差异 | 迁移边界 |
|---|---|---|---|
| softmax | T load；FP16/BF16 cast FP32；每行 `ReduceMax`；`x-max`；FP32 `Exp`；pad 清零；`ReduceSum`；广播 sum；FP32 `Div` | 仅 R-003/R-004 执行显式尾块清零 | 不得把 pad 清零提前到 x 的 max 归约，也不得删去减最大值 |
| y 写回 | sorted FP32 value；FP16/BF16 用 `CAST_RINT` 再 pack/unaligned store；FP32 直接 store | dtype 编译实例不同，Route key 不变 | 与另一种舍入或先降精度再排序不等价 |
| expert index | expert id 由 `Arange` 生成；Sort/ReduceMax 随 value 产生 index；finished 时做 `max(index,finished*E)` | R-002/R-004 增加哨兵 | 不能只改 y，不改配对 index；相等值和特殊值顺序依赖原 primitive |
| row index | `rowIdx[r,t]=t*R+r`，独立于 top-k expert 和 finished | 四 Route 相同 | 最大值 `R*k-1` 必须可表示 int32 |
| E tail | x DMA 物理行对齐到 A；有效 load/reduction mask 以 E 控制；非对齐 Route 的 softmax pad 为 0 | q=0 不执行尾清零，q>0 执行 | `padNegInfCount` 是 q 不是 A-E；mask 起点不能写成 E |
| k tail | `loopK=ceil_div(k,VF)`；末次 `lastLoopKCount=(k%VF==0?VF:k%VF)`；DataCopyUnAlign/Post 只提交有效项 | 四 Route 相同 | y/expert 的 UB live 区是连续 `n*k`；只有 rowIdx 使用 K8 行跨度，二者不得混用 |
| row/core tail | Host 分别生成普通 core、最后 core、普通 loop、最后 loop 的 row 数；Device 用最后 block/最后 loop 选择 | 四 Route 相同 | `lastCore*` 与 `perCore*` 不可互换；无跨核补写 |

对有限普通输入，四个 key Route 共享同一 FP32 softmax 与同一 TopK body，差异只来自 padding 和 finished index 后处理。NaN、正负无穷、FP32 exp 下溢及相等值没有独立 Host guard 或 Device canonicalization；知识迁移只能保留源码顺序和 value/index 配对，不能宣称额外的 NaN 顺序、稳定 tie-break 或 pad 在所有特殊输入下都不可被选中。

## 算子特定优化

### Trick T-001：FP32 稳定 softmax 与 UB 原地复用

#### 触发与源码依据

- 适用 Route：P0 R-001～P3 R-004，所有 dtype。
- 触发条件：每个有效 row tile；FP16/BF16 由 `LoadOneTensorForDtypeT` 升到 FP32。
- 固定官方源码：[`moe_gating_top_k_softmax_fullload_generalized_regbase.h:272-281`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::ReduceMax(reduceVreg, reduceMidRreg, mask);
AscendC::MicroAPI::Duplicate(dupVreg, reduceVreg, mask);
for (uint16_t j = 0; j < expertCountLoops; j++) {
    offset = rowLoopsOffset + j * repeatCount;
    ops::LoadOneTensorForDtypeT<T>(xTensorAddr, vreg0, mask, offset);
    AscendC::MicroAPI::Sub(vreg0, vreg0, dupVreg, mask);
    AscendC::MicroAPI::Exp(vreg0, vreg0, mask);
    AscendC::MicroAPI::DataCopy(xTensorFp32Addr + offset, vreg0, mask);
    mask = AscendC::MicroAPI::UpdateMask<int32_t>(remain);
}
```

#### 机制与设计原因

先求每行最大值再做 `exp(x-max)`，避免直接指数化的溢出；exp 直接写 `softmaxBuf`，后续 `Div` 又原地覆盖同一 TBuf，省去第二个整行 FP32 中间张量。MicroAPI mask 来自当前有效 E，repeat 来自 `B32_VF_COUNT=VRegBytes/4`。

#### 迁移边界

- 适用：最后一维独立 softmax，且整行或分段 row tile 能驻留 UB。
- 前提：950 RegBase、FP32 exp/reduce/div；保持 max、exp、sum、div 顺序。
- 失效：若 E/UB 超出本 Host 计账上界或平台不是 RegBase，当前 Direct tiling 返回失败，不能静默切到该 body。
- 证据边界：固定源码直接支撑该计算顺序；Direct 显式 entry 与局部 helper 闭包属于转换实现。

#### 错误做法

迁移时不得删去减最大值，也不得让 pad lane 以非零或未初始化值进入 A-wide ReduceSum/Sort。

### Trick T-002：单 repeat 的 top1 归约捷径

#### 触发与源码依据

- 适用 Route：P0 R-001～P3 R-004 的内部子支。
- 触发条件：`expertCountAlign==32 && k==1 && expertCount<=8`。
- 固定官方源码：[`moe_gating_top_k_softmax_fullload_generalized_regbase.h:335-357`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (k_ == 1 && expertCount_ <= B32_BLOCK_COUNT) {
    __VEC_SCOPE__ {
        __local_mem__ float *softmaxTensorAddr = (__local_mem__ float *)softmaxTensor.GetPhyAddr();
        __local_mem__ float *sortedTensorAddr = (__local_mem__ float *)sortedTensor.GetPhyAddr();
        AscendC::MicroAPI::RegTensor<float> valueAndIndexReg;
        AscendC::MicroAPI::MaskReg maskForValueAndIndex = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL2>();
        AscendC::MicroAPI::MaskReg maskForExpertCount;
        uint32_t uint32ExpertCount_ = static_cast<uint32_t>(expertCount_);
        int64_t kvExpertCountAlign_ = expertCountAlign_ * KEY_VALUE_FACTOR;


        for (uint16_t i = 0; i < static_cast<uint16_t>(rowCount); i++) {
            AscendC::MicroAPI::DataCopy(valueAndIndexReg, softmaxTensorAddr + i * expertCountAlign_);
            uint32_t expertCountForMask = uint32ExpertCount_;
            maskForExpertCount = AscendC::MicroAPI::UpdateMask<uint32_t>(expertCountForMask);
            AscendC::MicroAPI::ReduceMax(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
            DataCopy(sortedTensorAddr + kvExpertCountAlign_ * i, valueAndIndexReg, maskForValueAndIndex);
        }
    }
} else {
    Sort32(sortedTensor, softmaxTensor, expertIdxTensor, rowCount);
}
```

#### 机制与设计原因

只需 top1 且专家数不超过 8 时，`ReduceMax` 直接生成 value/index 对，跳过 `Sort32` 的完整 32 元素排序网络；输出仍落到与通用排序相同的 interleaved `sortedTensor`，所以后续提取代码无需分叉。

#### 迁移边界

- 适用：只取一个最大值且 MicroAPI 归约能保留匹配 index 的场景。
- 前提：`A==32`、`E<=8`、`k==1`，并保留 `VL2` value/index store 布局。
- 失效：`E>8` 或 `k>1` 用 `Sort32`；`A>32` 用逐行 `Sort<float,true>`。
- 证据边界：固定源码直接支撑该分支与 interleaved 布局；相等值次序不另作保证。

#### 错误做法

把该捷径扩大到 `k>1` 会缺少其余排序项；源码明确以 `else` 回到 Sort 路径。

### Trick T-003：Host 保守 UB 行计账统一三种 dtype

#### 触发与源码依据

- 适用 Route：P0 R-001～P3 R-004。
- 触发条件：计算 `maxRowCountInUb`。
- 固定官方源码：[`moe_gating_top_k_softmax_tiling_arch35.cpp:92-97`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_gating_top_k_softmax/op_host/moe_gating_top_k_softmax_tiling_arch35.cpp)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t inQueue = static_cast<int64_t>(expertCountAlign_ * sizeof(float) + BLOCK_BYTES);
int64_t outQueue = static_cast<int64_t>(kAlign_ * sizeof(float) + kAlign_ * sizeof(float) + kAlign_ * sizeof(float));
int64_t bufQueue = static_cast<int64_t>(expertCountAlign_ * sizeof(float) + expertCountAlign_ * sizeof(float) + expertCountAlign_ * sizeof(float));
int64_t sortTmpQueue =  static_cast<int64_t>(expertCountAlign_ * sizeof(float) * CONSTANT_TWO * CONSTANT_TWO);
int64_t sortedQueue = static_cast<int64_t>(expertCountAlign_ * sizeof(float) * CONSTANT_TWO);
maxRowCountInUb = static_cast<int64_t>((ubSize - sortedQueue) / (inQueue + outQueue + bufQueue + sortTmpQueue));
```

#### 机制与设计原因

Host 把 x 输入也按 FP32 容量计，使一个 tiling 公式可安全覆盖 float/half/bfloat16；并把 `8*A` 作为固定项从 UB 先扣除，再按每行成本求容量。对实际 `calcTmpBuf=8*A*Rloop+16*A`，该计账在 `Rloop>=1` 时不低估总 UB：一行时相等，多行时更保守。

#### 迁移边界

- 适用：多个输入 dtype 复用一个 FP32 中间计算 body。
- 前提：所有 Queue/TBuf 公式和 depth 与本实现相同，`A/K8` 对齐不变。
- 失效：增加 Queue depth、改变 sort scratch 布局或引入额外 TBuf 后必须重算，不能沿用该上界。
- 证据边界：固定源码支撑该保守公式；若改变任何 Queue/TBuf，必须重新做逐项容量证明。

#### 错误做法

仅按逻辑有效 `E` 或 FP16 输入字节估算 UB 会漏掉 FP32 对齐 buffer；固定仓 Host 明确以 `expertCountAlign*sizeof(float)` 计账。

## 模板与实现对比

| 对比维度 | P0 R-001 无完成/对齐 | P1 R-002 有完成/对齐 | P2 R-003 无完成/补齐 | P3 R-004 有完成/补齐 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `!finished,E%32=0` | `finished,E%32=0` | `!finished,E%32!=0` | `finished,E%32!=0` | 两个独立 bool 编译维度形成 4 key |
| 模板与 dtype | `<T,false,false>` | `<T,true,false>` | `<T,false,true>` | `<T,true,true>` | 每 Route 各有 3 个 T 实例 |
| 循环与分核 | 共用 | 共用 | 共用 | 共用 | 与 bool specialization 无关 |
| Queue/TBuf/寄存器 | finished Queue 只分配 | finished Queue 搬运 | pad lane 清零 | 两者同时 | UB 物理布局保持一致 |
| 内存总占用与复用 | 基线公式 | 相同 | A 对齐后同公式 | 相同 | Host 按最坏 FP32 容量计账 |
| 搬运/Vector API | softmax + sort | 加 bool copy/index Max | 加 pad Duplicate | 两者同时 | 编译期消除无关计算分支 |
| 同步、精度与 tail | Queue；FP32；k mask | 同上 + finished Queue | 同上 + E pad mask | 两种 mask | row/k tail 共用 |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | 无；非法手工 key 的 wrapper else 会误落此 entry | 只能使用 calculator 生成 params |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本 Direct 交付固定源码范围不可见 | 本 Direct 交付固定源码范围不可见 | 支持；目标为 Ascend 950/RegBase |
| Host/Launcher | 未纳入本卡 | 未纳入本卡 | `PlatformAscendCManager` 查询 AIV/UB，3 dtype launcher × 4 entry |
| Kernel/资源 | 未纳入本卡 | 未纳入本卡 | arch35 `MoeGatingTopKSoftmaxFullloadGenerlized`，单 AIV、无 workspace |

## 限制与失败边界

- 仅覆盖 CANN 9.1.0、950/Ascend 950 Direct Invoke；不外推 ACLNN、A2/A3 或 v2 语义。
- 输入必须连续，`R>0`、`1<=E<=2048`、`1<=k<=min(E,1024)`；Host tiling 还要求平台句柄、正 AIV 数和足够 UB。
- `rowIdx` 的基础 progression 是 ABI 语义的一部分；复用时必须按官方 `ArithProgression(...,0,rows,k)` 与行基址加法保持布局。
- Direct wrapper 的最终 `else` 不是合法 key 校验：未知手工 key 会进入 R-004 entry。调用者只能传入 calculator 生成的 100000/100001/100010/100011。
- `blockDim` 在 launcher 中窄化为 `uint32_t`；当前值由 `perCore=ceil_div(R,aivNum)`、`needCore=ceil_div(R,perCore)` 产生，满足 `needCore<=min(R,aivNum)`，禁止绕过 calculator 写入任意 64 位值。
- 官方 Host 的 16 MiB workspace 是保留量而非 Device 数据边；Direct 转换去除它的前提是当前固定 body 不读取 workspace。若替换 body，必须重新审计。
- 固定官方 Device helper `LoadOneTensorForDtypeT/StoreUnAlignOneTensor` 来自 CANN 公共头；Direct 转换只局部闭合命中符号，未把公共库其余实现当作本算子知识。
- 本卡只声明固定源码与 canonical Direct 交付能够闭合的 950 路径，不外推其它平台或改版 body。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口/router | `moe/moe_gating_top_k_softmax/op_kernel/moe_gating_top_k_softmax_apt.cpp` | `moe_gating_top_k_softmax` | 四 key、AIC return、模板实例、Init/Process |
| POD | `moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_tiling_def.h` | `MoeGatingTopKSoftmaxRegbaseTilingData` | 15 字段顺序和类型 |
| Kernel body | `moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h` | `MoeGatingTopKSoftmaxFullloadGenerlized` | Queue/TBuf、softmax、top-k、finished、tail |
| Host/Tiling | `moe/moe_gating_top_k_softmax/op_host/moe_gating_top_k_softmax_tiling_arch35.cpp` | `MoeGatingTopKSoftmaxRegbaseTiling` | 950 capability、分核、UB、key、workspace、POD producer |

Direct 的普通 POD、12 个显式 entry、3 个 dtype launcher、平台查询和 `register_direct_launch_op` 是从上述四个官方文件派生的交付胶水；固定提交不存在可引用的同名 Direct 文件，因此不把不可达或非官方 URL 写入 `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 个连续 int64 字段 | 字段顺序、offset、单位；launcher 按值传 POD | `MoeGatingTopKSoftmaxRegbaseTilingData` |
| 调用顺序 | 单 Kernel 内 rowIdx→x/finished→softmax→top-k→输出 | Queue producer/consumer 和每 loop 顺序 | `Process` |
| workspace/内存 | Direct workspace=0；5 Queue+4 TBuf | depth=1、A/K8 对齐、calcTmp 分区、Host 保守式 | `Init`；`CalcMaxRowCountInUb` |
| 同步 | TQue EnQue/DeQue/Free，单 stream 单 Kernel | depth=1 buffer 未消费前不得覆盖 | `CopyInX/ComputeSoftmax/ComputeTopK/CopyOut*` |
| 精度与尾块 | FP32 stable softmax，RINT 写回，E/k/row tail | max-exp-sum-div 顺序、pad mask、unaligned post update | `ComputeSoftmax/ComputeTopK` |
| 失败边界 | Host `false` 阻断 launch | 不绕过 shape/UB/platform 校验，不构造未知 key | direct calculator；`IsCapable/DoOpTiling` |
