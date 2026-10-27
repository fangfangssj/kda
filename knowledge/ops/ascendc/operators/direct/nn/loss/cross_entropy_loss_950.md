---
type: Operator
title: "CrossEntropyLoss 950 Direct Invoke 算子知识"
description: "说明 cann_bench::cross_entropy_loss、calc_cross_entropy_loss_tiling 与 launch_cross_entropy_loss_* 在 950 上的接口语义、Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss.cpp"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_tiling_data.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_full_load.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_ar_big_c.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/softmax_v2/op_kernel/arch35/softmax_v2_base.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/config/ascend950/cross_entropy_loss_binary.json"
    role: implementation
created_at: "2026-08-25T19:31:25Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# CrossEntropyLoss 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | PyTorch custom op `cann_bench::cross_entropy_loss`，PrivateUse1 实现 `cross_entropy_loss_npu`，Host producer `calc_cross_entropy_loss_tiling`，C ABI launcher `launch_cross_entropy_loss_{float16,float32,bfloat16}`；入口类型为 `Direct Invoke`。 |
| 计算语义 | 类别轴固定为 dim 1；hard target 计算稳定 `-log_softmax(input)[target]`，soft target 计算 `-sum(target*log_softmax(input), C)`，再执行 `none/mean/sum`。 |
| 输入输出 | 逻辑 `input[N,C,*S]`；hard target 为 `int32/int64 [N,*S]`，soft target 与 input 同 shape/dtype；input/output 为 FP16、FP32 或 BF16。非连续 view 由 Host 在 launch 前物化为连续 NPU tensor。 |
| 执行类型 | R-001 为 AIV 标量 GM 写；R-002/R-003/R-004 的 primary 为 AIV entry 内 SIMT，reduced multi-core 时追加 AIV 标量 finalize；R-005 的 primary 为单 Kernel `AIV_VECTOR+SIMT` Hybrid，同条件追加 AIV 标量 finalize；R-006/R-007 为官方来源的 `AIV_VECTOR`。所有 Host 可达节点均无 Cube/AIC 或 `MIX_CV`。 |
| Kernel 路线 | 按实现重要性：P0 R-007 big-C；P1 R-005 channel-first lean Hybrid；P2 R-006 full-load；P3 R-004 hard target-only SIMT；P4 R-003 soft warp SIMT；P5 R-002 soft scalar SIMT；P6 R-001 empty。Route ID 保持与 Host `route` 枚举映射一致。 |
| 核心资源 | GM、AIV UB `TQue/TBuf`、MicroAPI 寄存器、SIMT thread scratch；reduced SIMT/Hybrid 使用 `2*lossCoreNum*sizeof(float)` workspace，官方 full-load/big-C 使用 `coreNum*384B` workspace。 |
| 来源闭包 | `direct_implementation_closed`：固定官方来源闭合 950 arch35 full-load/big-C body、POD 与构建平台；最终交付源码闭合 direct schema、selector、融合 SIMT/Hybrid、launcher、workspace 和构建接线。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `S=prod(input.shape[2:])`、`R=N*S`，row `r=n*S+s` 对应 logits `x[n,:,s]`。稳定统计为 `m=max_c x_c`、`z=sum_c exp(x_c-m)`。

- hard target `y`：若 `y_r==ignore_index`，该 row 的 loss 和 mean 计数均为 0；否则 `l_r=m+log(z)-x_y`。
- soft target `p`：`l_r=sum_c p_c*(m+log(z)-x_c)`；`ignore_index` 不参与 soft 分支。
- `none` 返回 `[N,*S]`；`sum` 返回 `sum_r l_r`；hard `mean` 除以非 ignore row 数，soft `mean` 除以 `R`。

Direct 融合路线不会物化完整 `log_prob`。两条 official-derived 转换的处理不同：full-load 删除官方 `logPropQueue_` 的分配、入队与完整 GM CopyOut，直接在 FP32 `tempUb_` 中 gather loss；big-C 仍保留 `outQueue_`，因为它承载 chunk 内 FP32 exp/log、fold 与 gather 的 UB 生命周期，只删除/抑制 `DataCopyOut1` 对完整 `log_prob` 的 GM 写回。两者都只向任务返回 loss；这是交付件转换事实，不是上述固定官方文件的原样行为。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `input` | 逻辑 `[N,C,*S]`，rank 2～8 | FP16、FP32、BF16 | `N>=0`、`C>0`；类别轴固定 dim 1；任意合法 stride 可进入，Device 只消费 Host 产生的连续副本。 |
| `target` hard | 逻辑 `[N,*S]` | int32、int64 | rank 比 input 少 1；batch/spatial 逐维相等；除 `ignore_index` 外标签须在 `[0,C)`。越界标签不属于语义输入域；active body 在形成地址前做 ignore/bounds 处理，避免非法 GM/UB 访问。 |
| `target` soft | 与 input 完全同 shape/layout | 与 input 相同 | 概率值域/归一性不由插件检查；走 soft SIMT。 |
| `reduction` | 字符串属性 | `none/mean/sum` | 默认 `mean`；映射为 0/1/2。 |
| `ignore_index` | 标量属性 | int64 | 默认 `-100`；只影响 hard target。 |
| `loss` | `none` 为 `[N,*S]`，否则标量 | 与 input 相同 | ignored hard row 写 0；空域 mean 写 NaN，sum 写 0。 |
| `logProb` | direct ABI 占位 | 与 input 相同 | 插件仅分配 8 元素有效指针；所有 Host 可达交付路线均不写完整 tensor。 |
| `workspace` | uint8 tensor | byte | 配置为 0 时仍分配 1B 占位；非零时按下文公式分配并记录当前 stream。非连续 input/target 的连续副本也记录到当前 stream。 |

### 入口与版本

`TORCH_LIBRARY_FRAGMENT` 声明 `cross_entropy_loss(Tensor input, Tensor target, str reduction="mean", int ignore_index=-100)`；Meta 完成 shape/dtype 检查并推导输出，PrivateUse1 绑定 `cross_entropy_loss_npu`。后者取得当前 NPU stream，生产 `CrossEntropyLossLaunchConfig`，按 input dtype 调用一个 C ABI launcher，并用 `OpCommand::RunOpApi("CrossEntropyLossDirect", aclCall)` 接入调度。

目标版本为 CANN 9.1.0、950/Ascend 950。固定官方构建文件把 `arch35/cross_entropy_loss.cpp` 接到 `ascend950`；Direct 的 torch schema、融合 SIMT/Hybrid、定制 Host selector 与 direct CMake 是交付件补充，不宣称存在于固定官方 Commit。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckInputs` 检查 input dtype/rank/C；hard target 检查 rank 和 `[N,*S]`，soft target 检查同 dtype/同 shape。Meta 还要求 soft target 时 `ignore_index<0`；它不要求原 tensor 连续。
2. `ReductionCode` 只接受 `none/mean/sum`；Meta 按 reduction 分配输出。
3. `cross_entropy_loss_npu` 要求同 device，复制最多 8 个 input 维度到固定数组，零初始化 config。
4. `calc_cross_entropy_loss_tiling` 再检查指针、rank/dtype/reduction/targetType、`N/C`；`CheckedMul` 依次计算 `S`、`R=N*S`、`N*C*S` 并拒绝 int64 溢出。
5. 从 `PlatformAscendCManager` 读取 UB 字节数 `U` 与 AIV 核数 `P`；任一为 0 则失败。selector 依真实顺序只命中首条 Route。
6. `CheckHostResourceEnvelope` 在 device guard、输出/临时分配、连续化、取 stream 和 launch 之前，复核 route 范围、`blockDim>0`、workspace/output/input/target 字节乘法及全部新分配字节的无符号加法；非连续 tensor 还计入连续副本字节。
7. 校验后才建立 device guard，并严格按插件语句顺序执行：先分配 `loss`，再以 `input.contiguous()`、`target.contiguous()` 取得 Device 消费的连续 tensor，最后分配 8-element `logProb` 与 workspace；原本连续时 `contiguous()` 可复用原 storage。tiling 非零返回由 `TORCH_CHECK` 阻止 launch；launcher 本身不返回 Device 异步错误。

### 参数、Workspace 与生命周期

`CrossEntropyLossLaunchConfig` 在调用栈上零初始化并按值捕获进 `aclCall`；各 kernel 再把 `officialTiling` 或 `channelFirstTiling` 按值传入。loss 由 torch 持有；临时 `logProb`、workspace 与可能新建的连续 input/target 副本都由 caching allocator 管理并显式 `recordStream`，可在 stream 完成后回收。

- SIMT/Hybrid reduced：`workspaceBytes=2*lossCoreNum*4`，slice 0 为 partial sum，slice `lossCoreNum` 为 partial count。
- full-load/big-C reduced：`workspaceBytes=P*32*4*3=P*384`；三段分别承载 loss、count/weight、smooth（Direct 固定无 weight/smoothing，第三段仍按官方 ABI 预留）。
- `none`/empty：逻辑 workspace 为 0；插件分配 1B 占位但 Device 不读。

### 启动图与失败传播

```text
cross_entropy_loss_npu
  -> calc_cross_entropy_loss_tiling
  -> launch_cross_entropy_loss_<dtype>(..., config, stream)
     -> primary entry <<<config.blockDim, dynUB-or-null, stream>>>
     -> [reduction!=none && lossCoreNum>1]
          finalize entry <<<1, nullptr, same stream>>>
     -> loss
```

| Route | 节点与 grid | GM/workspace 边、ordering、owner/lifetime | 失败传播 |
|---|---|---|---|
| R-001 | empty entry `<<<1>>>` | 无边；直接写标量 loss | tiling 失败不 launch；entry 无返回码 |
| R-002/R-003 | soft entry `<<<lossCoreNum>>>`；可选 finalize `<<<1>>>` | primary 写 `[sum,count]` 两段 FP32；同一 stream 保证 finalize 可见；torch workspace 持有至 stream 完成 | Host 只在成功 tiling 后依次提交；异步错误不映射到 lambda 返回值 |
| R-004 | fused hard entry；可选同 dtype finalize | 与 soft 相同 | 同上 |
| R-005 | fused-vector entry `<<<usedCoreNums,localMemoryBytes>>>`；可选 finalize | Hybrid primary 写 partials；UB 内有 `DataSyncBarrier<UB>`；跨 kernel 由同 stream 排序 | 同上 |
| R-006/R-007 | official-derived entry `<<<realCoreNum>>>` | 每核写间隔 32 FP32 的三段 workspace，`SyncAll` 后 core 0 汇总并写 loss；owner 为 torch workspace | 单 Kernel 内同步；Host 无第二节点 |

#### Dormant 生产节点

| dormant 节点 | 生产链不可达证据 | 重新启用的必要条件 |
|---|---|---|
| `CROSS_ENTROPY_LOSS_CHANNEL_FIRST=2` | selector 从不写 `route=2`，`BuildChannelFirstFullLoad/Recompute` 也无调用者；因此 launcher 中 softmax→NLL→可选 finalize 的 2/3 节点图不可达 | Host 调用对应 builder、完整生产 `softmaxRoute/blockDim/lossCoreNum/workspace`；插件必须把 8-element dummy `logProb` 改为 `N*C*S` 输出，并为 int32 hard target 增加 NLL dispatch 或限定 int64 |
| `RunChannelFirstFusedVector` | 三个 fused-vector entry 均直接调用 `RunChannelFirstLean`，无 router 指向旧 full-load/recompute fused body | 新 selector/key 与 entry router；按该 body 的 scratch 写入范围分配 GM，闭合 producer→scratch-reduce 消费关系 |
| `cross_entropy_loss_scratch_reduce_*_entry` | launcher 没有任何 launch 节点指向它们，且 active Route 不生产其所需的完整 row scratch | 增加匹配的 scratch producer、scratch 容量、同 stream launch 顺序和 reduced workspace 协议 |
| 官方 `CrossEntropyLossEmpty` 类 | folded 进翻译单元但无实例化；R-001 直接调用 `RunCrossEntropyEmpty<T>` | direct empty entry 改为构造官方类并传完整 ABI/POD，或删除重复实现后二选一闭合 |

这些节点只说明源码库存，不计入 active/conditional Route，也不能据其函数存在推断生产可达。

## Route 与参数设计

### 分核与资源计算

记 `P=GetCoreNumAiv()`、`U=UB bytes`、`b=sizeof(input)`、`bt=sizeof(target)`、`A=32/b`、`R=N*S`。

- SIMT：固定 512 threads/block；hard scalar 每线程跨 `lossCoreNum*512` 个 row；soft scalar 覆盖 `C<128` 或 `C>2^24`，每线程一 row；soft warp 只覆盖 `128<=C<=2^24`，每 warp 一 row、每 block 16 rows。上界保证用 FP32 scratch 保存的最大 class index 仍精确。
- lean Hybrid：`spatialAlign=A`，`maxTile=floor((U-64KiB)/(C*b+8)/A)*A` 再截到 1024；`tileLen=align_up(ceil(S/max(1,ceil(P/N))),A)` 后夹在 `[A,maxTile]`。tile 数 `N*ceil(S/tileLen)`，均匀分给 `usedCoreNums`。
- rank-2 full-load：`realCoreNum=min(N,P)`，former 核多 1 row；Host 先尝试 `db=2`，不足则 `db=1`，两者都不能容纳至少 `A` rows 才转 big-C。
- big-C：每次最多 64 rows；`cOnceNum` 从 `(U-fixed)/bytesPerC` 向 32B 对齐，并在 cache tree 超 UB 时逐 block 缩小；`cLoopTimes=ceil(C/cOnceNum)`。

### Route 选择

下表是实现知识的阅读与复用优先级，不代表 Host 判断顺序。

| 重要性 | Route | 关键特征 | 适用域 | 实现前提 |
|---|---|---|---|---|
| P0 | R-007 big-C | rank-2 hard target 的 class 分块高吞吐实现；用二叉 cache 合并 chunk exp sum，单 Kernel 内跨核归约 | rank 2、hard target、先通过 official-fast 总 guard，但 full-load UB/编码 capability 不成立 | `cOnceNum` 32B 对齐且非零；Host/Device 的 `kTimes/kTimesTail/cacheStart` 一致；BigC 编码复核与 reduced workspace 均成立，否则回 R-004 |
| P1 | R-005 channel-first lean Hybrid | 大工作量 rank-N hard target 主实现；一次 DMA 装入 `[C,tileS]`，AIV 求 max/sum，SIMT 只 gather target | rank>2、`N*C*S>=10^6`、未命中前序 hard SIMT、lean 全部容量/编码 guard 成立 | 12-bit `DataCopyPad.blockCount` 合法；64KiB reserve、256B guard、dynamic UB 与 SIMT thread POD 一致 |
| P2 | R-006 full-load | rank-2 整 C 驻留；Host 在 db=2/db=1 间选择，MicroAPI 一次完成 row softmax 与 target gather | rank 2、先通过 official-fast 总 guard，且至少一个对齐 row 可 full-load、编码复核成立 | `TryFullLoad` 资源估算覆盖 Queue/TBuf；`db` 必须乘入 UB；FP32/none 的近零 loss 精化与 reduced workspace 协议一致 |
| P3 | R-004 hard target-only SIMT | 不物化 log-prob、不分配 C-sized UB slab的 int64-index streaming 完整安全兜底 | hard `none` 小域、rank-N 小工作量/lean 不可用、rank-2 小 N/二分类，或任一 official-fast/编码 guard 失败 | 512 threads/block；FP32 compensated exp tail；reduced 两段 partial workspace与可选 finalize；合法 shape 不因 fast-path 编码受限而失败 |
| P4 | R-003 soft warp SIMT | 每 warp 一 row，lane 跨 class；scratch 分阶段别名复用 | soft target、`128<=C<=2^24` | 所有 lane 包括 inactive row 都必须到达 block barrier；两块 512-FP32 scratch；class index 可由 FP32 精确表示 |
| P5 | R-002 soft scalar SIMT | 每线程一 row、串行扫描 class，也是超出 warp-index 精确域后的 fallback | soft target、`C<128 \|\| C>2^24` | 512 threads/block；稳定 FP32 class scan；int64 class index；reduced partial/finalize 协议 |
| P6 | R-001 empty | 仅 core 0 写空域 scalar reduction 结果 | `N*S==0` | grid=1；`none` 不写、`mean` 写 NaN、`sum` 写 0 |

以下才是 Host 的真实判断顺序、短路和 fallback：

| Host 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 empty | `EMPTY=6` | `R==0` | R-002 |
| 2 | R-002 soft scalar SIMT | `SOFT_SIMT=5` / scalar body | `targetType==SOFT && (C<128 \|\| C>2^24)`；Host key 与 warp 共用，第二项由 Device router 判定 | `128<=C<=2^24` 转 R-003 body |
| 3 | R-003 soft warp SIMT | `SOFT_SIMT=5` / warp body | `targetType==SOFT && 128<=C<=2^24`；Host 用 `C>=128` 计算 16 rows/block，Device 再以 `C<=2^24` 收窄 | 超上界在同 entry 内转 R-002 body；soft 分支在此终止 |
| 4 | R-004 hard target-only SIMT | `GENERIC_SIMT/FUSED_SIMT=3` | hard 且首个成立：`none && N*C*S<=1048576`；或 `rank>2 && (N*C*S<10^6 \|\| BuildChannelFirstLean/动态 UB 复核失败)`；或 `rank==2 && (N<32 \|\| C==2)`；或 rank-2 official-fast 总 guard/FullLoad/BigC 编码复核失败 | guard 通过则 R-005/R-006/R-007 |
| 5 | R-005 channel-first lean Hybrid | `FUSED_VECTOR=4` | hard、未命中 `none && N*C*S<=1048576`、`rank>2`、`N*C*S>=10^6`，且 `C<=4095`、`maxTile>=32/b`、full/tail DMA 字段可编码、`localMemoryBytes<=min(U,UINT32_MAX)` | 任一 guard 失败回 R-004 |
| 6 | R-006 full-load | `FULL_LOAD=1` | hard、未命中 `none && N*C<=1048576`、rank 2、`N>=32`、`C!=2`，且 official-fast 总 guard：`N*C>=10^6, N<=2097152, 2<=C<=16384, ignoreIndex` 可表示为 int32；`TryFullLoad(db2→db1)` 与 `FullLoadEncodingIsSafe` 均成立 | full-load 不成立则尝试 R-007；总 guard 失败直接 R-004 |
| 7 | R-007 big-C | `BIG_C=0` | 与 R-006 相同的前序/official-fast guard 已成立，full-load capability/编码不成立，且 `BigCEncodingIsSafe` 成立 | BigC 编码复核失败回 R-004 |

```text
R==0 -> R-001
soft -> (128<=C<=2^24) ? R-003 : R-002
hard none && N*C*S<=1Mi elements -> R-004
rank>2 -> (N*C*S>=10^6 && BuildLean与dyn-UB编码安全) ? R-005 : R-004
rank==2 && (N<32 || C==2) -> R-004
rank2 official-fast总guard失败 -> R-004
TryFullLoad(db2 then db1) && FullLoadEncodingIsSafe -> R-006
ComputeBigC && BigCEncodingIsSafe -> R-007
else -> R-004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| P0 · R-007 | 0 | hard rank2、official-fast 总 guard 成立、FullLoad 不 capable/编码不安全、BigC 编码安全 | 单 AIV kernel | `CrossEntropyLossBigC<T,TargetT,reduction,0,0,ignorePositive>` | depth2 class chunk、cache tree、workspace+`SyncAll` | C chunk tail、N tail；Host 编码失败回 R-004，Device 内无 fallback |
| P1 · R-005 | 4 | hard rank>2、`C<=4095`、C-sized tile+guard 可放 UB | 单 Hybrid primary；reduced multi-core 时 finalize | `RunChannelFirstLean`→`CrossEntropyLeanStats`→`CrossEntropyLeanGatherSimt` | C×tile Queue、max/sum、thread partial；SIMT→scalar UB DSB | S 尾 tile；失败回 R-004 |
| P2 · R-006 | 1 | hard rank2、FullLoad capable | 单 AIV kernel | `CrossEntropyLossFullLoad<T,TargetT,reduction,0,0,ignorePositive>` | db1/db2 input Queue；MicroAPI FP32；reduced 用 workspace+`SyncAll` | former/tail core、N tail、C padding |
| P3 · R-004 | 3 | hard 的 accurate-small、rank-N fast guard、rank-2 shape/official/编码 guard 任一有序 fallback | primary SIMT；reduced multi-core 时 finalize | `CrossEntropyFusedTargetSimt<T,TargetT>` | 两个 512-FP32 TBuf；每 block tree reduce | global-thread stride；ignore row 写 0/不计数 |
| P4 · R-003 | 5 | soft、`128<=C<=2^24` | primary SIMT；reduced multi-core 时 finalize | `CrossEntropySoftWarpSimt<T>` | 每 warp 一 row，两个 512-FP32 scratch，block/warp barriers | inactive row 仍参加 barrier；rowActive 守护；上界外回 scalar body |
| P5 · R-002 | 5 | soft、`C<128 \|\| C>2^24` | primary SIMT；reduced multi-core 时 finalize | `CrossEntropySoftScalarSimt<T>` | 2×512 FP32 TBuf，block tree barrier | row grid-stride；int64 class index；无 ignore |
| P6 · R-001 | 6 | `R==0`；3 input dtype；950 | 单 Kernel | `RunCrossEntropyEmpty<T>` | 无 UB Queue/workspace；dcache flush | none 不写；mean NaN；sum 0 |

### 性能与安全 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| big-C block alignment | `cOnceNum` 为 `32/b` 倍数且 cache bytes 可放剩余 UB | DMA 与缓存树均依赖对齐 class chunk | R-007 | Host 继续缩小 chunk | `ComputeBigC`（交付件）/`BigC::Init` |
| 12-bit DMA blockCount | `C<=4095` | lean `DataCopyPad.blockCount=C`，DAV-3510 编码宽度为 12 bit | R-005 | R-004 streaming SIMT | `BuildChannelFirstLean`（交付件） |
| lean UB reserve | `maxTile=floor((U-64KiB)/(C*b+8))` 且 `>=32/b` | C×tile slab、两份 FP32 statistic、guard 与 SIMT scratch | R-005 | R-004 | 同上 |
| lean tile cap | `maxTile<=1024`，threads 为 `[32,512]` 中覆盖 tileLen 的最小 2 次幂 | 限制单 tile vector/SIMT 工作集和 scratch | R-005 | 同 Route 参数变化 | `BuildChannelFirstLean/calc_cross_entropy_loss_tiling`（交付件） |
| rank-N 工作量门槛 | `N*C*S>=10^6` | 小工作量不承担 C-slab DMA、动态 UB 与 Hybrid 启动固定成本 | R-005 | R-004 | `kFastPathMinElements`（交付件） |
| full-load double buffer | db2 capable 优先，失败再 db1 | input Queue depth 影响 UB 与流水 | R-006 | db1 或 R-007 | `TryFullLoad`（交付件）/`FullLoad::Init` |
| rank-2 official-fast 总 guard | `N*C>=10^6, N<=2097152, 2<=C<=16384, ignoreIndex∈int32` | official body 内含 int32 tile/gather 字段及 DMA 编码边界 | R-006/R-007 | R-004 | `CanUseOfficialRank2FastPath`（交付件） |
| full/big 编码复核 | 分别检查 blockCount/blockLen、`onceN*C<=INT32_MAX`；BigC 另查三段 workspace stride、loop/cache 字段 | Host 的 int64 POD 最终会落入更窄的 Device/DMA 字段，容量可放不代表可编码 | R-006/R-007 | R-007 或 R-004 | `FullLoadEncodingIsSafe/BigCEncodingIsSafe`（交付件） |
| positive ignore specialization | `ignoreIndex>=0` | 编译期 `ignorex` 决定目标比较/mask 路径 | R-006/R-007 | 选 `ignorex=0` 实例 | `DispatchOfficial`（交付件） |
| SIMT accurate-small domain | `reduction==none && N*C*S<=1048576` | 目标值直接以稳定 `log1p` 路径计算，避免物化 log_prob | R-004 | 后续 rank selector | `kAccurateHardMaxElements`（交付件） |
| soft warp 精确域 | `128<=C<=2^24` | 每 warp 32 lane 分担类别循环；winning class 暂存 FP32，故上界外必须回到 int64-index scalar body | R-003/R-002 | 两者互切 | `RunCrossEntropySoft`（交付件） |
| shape 乘法 | 每次 `CheckedMul(a,b)` | 防止 GM extent/offset 的 int64 溢出 | 全部 | error -6/-7/-8 | `cross_entropy_loss_kernel.cpp::CheckedMul`（交付件） |

### POD/ABI

所有结构无显式 packing/static_assert；下列 offset/size 按声明顺序和标准自然对齐推导，跨工具链复用应补 `sizeof/offsetof` 断言。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CrossEntropyLossLaunchConfig` | `officialTiling` | struct/152B | offset 0 | POD | `FillCommonTiling/TryFullLoad/ComputeBigC` | official entry | R-006/R-007 active |
| 同上 | `channelFirstTiling` | struct/344B | offset 152 | POD | soft/fused/lean branches | SIMT/Hybrid entry | R-002～R-005 active |
| 同上 | `workspaceBytes` | uint64/64 | offset 496 | byte | selector formulas | plugin allocator | 0 时分配 1B 占位 |
| 同上 | `blockDim` | uint32/32 | offset 504 | AIV block | selector | launch grid | 必须 >0 |
| 同上 | `localMemoryBytes` | uint32/32 | offset 508 | byte | lean UB 公式，32B 对齐 | R-005 dynamic UB launch | 其它 Route 为 0 |
| 同上 | `route` | int32/32 | offset 512 | enum | selector | launcher router/entry | 0～6；4/5/6 等如上 |
| 同上 | `reduction` | int32/32 | offset 516 | enum | plugin | launcher/body | 0～2 |
| 同上 | `targetType` | int32/32 | offset 520 | enum | plugin | dtype router/body | 0～2 |
| 同上 | `reserved` | int32/32 | offset 524；size 528 align 8 | N/A | zero-init | 不读 | inactive |
| `CrossEntropyLossChannelFirstTilingData` | `fullLoad` | struct/160B | offset 0 | POD | 当前可达 Route 保持零 | dormant channel-first body | inactive |
| 同上 | `recompute` | struct/128B | offset 160 | POD | R-005 `BuildChannelFirstLean` | `RunChannelFirstLean` | 仅部分字段 active |
| 同上 | `batch` | int64/64 | offset 288 | batch | `sizes[0]` | row/tile 地址 | `>=0` |
| 同上 | `classes` | int64/64 | offset 296 | class | `sizes[1]` | class loops/DMA | `>0`；R-005 `<=4095` |
| 同上 | `spatial` | int64/64 | offset 304 | element | `prod(shape[2:])` | row 解码/GM stride | 可为 0 |
| 同上 | `totalTargets` | int64/64 | offset 312 | row | `batch*spatial` | SIMT/grid loop | R>0 except empty |
| 同上 | `ignoreIndex` | int64/64 | offset 320 | label | plugin attr | hard bodies | soft 不读 |
| 同上 | `softmaxRoute` | int32/32 | offset 328 | enum | R-005 写 2 | lean 当前不读 | 历史/dormant 字段 |
| 同上 | `lossCoreNum` | int32/32 | offset 332 | core | selector | partial offsets/finalize | `1..P` |
| 同上 | `reduction` | int32/32 | offset 336 | enum | selector | body | 0～2 |
| 同上 | `reserved` | int32/32 | offset 340；size 344 align 8 | SIMT threads | R-005 power-of-two producer | lean `asc_vf_call` | 32～512；其它 Route 0 |
| `SoftmaxV2ARARecomputeTilingData` | `totalRLen` | int64/64 | offset 0 | class | R-005 写 `C` | lean stats | `<=4095` |
| 同上 | `totalA0Len` | int64/64 | offset 8 | spatial | R-005 写 `S` | lean tile address | `>0` |
| 同上 | `totalTiles` | int64/64 | offset 16 | tile | `N*tileA0Outer` | block-stride tile loop | `>0` |
| 同上 | `tilesPerCore` | int64/64 | offset 24 | tile/core | `ceil(totalTiles/P)` | 当前 lean body 不读 | 保留 |
| 同上 | `usedCoreNums` | int64/64 | offset 32 | core | `ceil(totalTiles/tilesPerCore)` | block stride/grid | `1..P` |
| 同上 | `tileA0Outer` | int64/64 | offset 40 | tile/batch | `ceil(S/tileA0Len)` | tile 解码 | `>0` |
| 同上 | `tileA0Len` | int64/64 | offset 48 | element | 对齐 tile 公式 | UB/DMA stride | `32/b` 倍数，`<=1024` |
| 同上 | `tileA0Tail` | int64/64 | offset 56 | element | `S-tileLen*(outer-1)` | 尾 tile | `1..tileA0Len` |
| 同上 | `binAddRFactor` | int64/64 | offset 64 | class | zero-init | lean 不读 | inactive |
| 同上 | `binAddRLoop` | int64/64 | offset 72 | loop | zero-init | lean 不读 | inactive |
| 同上 | `binAddRTotalLoop` | int64/64 | offset 80 | loop | zero-init | lean 不读 | inactive |
| 同上 | `binAddRTail` | int64/64 | offset 88 | class | zero-init | lean 不读 | inactive |
| 同上 | `binAddBasicBlockLoop` | int64/64 | offset 96 | loop | zero-init | lean 不读 | inactive |
| 同上 | `binAddMainFoldCount` | int64/64 | offset 104 | loop | zero-init | lean 不读 | inactive |
| 同上 | `binAddCacheBufferCount` | int64/64 | offset 112 | slot | zero-init | lean 不读 | inactive |
| 同上 | `binAddResultCacheID` | int64/64 | offset 120；size 128 | slot | zero-init | lean 不读 | inactive |
| `CrossEntropyLossRegBaseTilingData` | `realCoreNum` | int64/64 | offset 0 | core | `min(N,P)` | Process/SyncAll | `>0` |
| 同上 | `frontCoreNum` | int64/64 | offset 8 | core | `N-frontCoreNSize*realCoreNum` | former/tail split | `[0,realCoreNum)` |
| 同上 | `frontCoreNSize` | int64/64 | offset 16 | row/core | `N/realCoreNum` | base offsets | `>0` |
| 同上 | `onceNSize` | int64/64 | offset 24 | row/tile | full/big producer | main loop | full varies；big `<=64` |
| 同上 | `onceNSizeT` | int64/64 | offset 32 | row/tile | producer | tail-core loop | big `<=64` |
| 同上 | `nLoopTimesB` | int64/64 | offset 40 | loop | producer | former cores | 非负 |
| 同上 | `nLoopTimesT` | int64/64 | offset 48 | loop | producer | tail cores | 非负 |
| 同上 | `nTailNumB` | int64/64 | offset 56 | row | producer | former last tile | `1..onceNSize` 或 0 |
| 同上 | `nTailNumT` | int64/64 | offset 64 | row | producer | tail-core last tile | 同上 |
| 同上 | `ignoreIndex` | int64/64 | offset 72 | label | attr | gather/mask | int64；部分官方 gather 窄化到 int32 |
| 同上 | `cLoopTimes` | int64/64 | offset 80 | loop | big-C producer | class chunk loops | R-007 active |
| 同上 | `cOnceNum` | int64/64 | offset 88 | class/aligned class | full/big producer | Queue/stride | 32B 对齐 |
| 同上 | `cOnceNumTail` | int64/64 | offset 96 | class | big-C producer | last chunk | `1..cOnceNum` 或 0 |
| 同上 | `C` | int64/64 | offset 104 | class | `sizes[1]` | address/loops | `>0` |
| 同上 | `kTimesTail` | int64/64 | offset 112 | loop | big-C tree producer | paired head/tail folds | R-007 active |
| 同上 | `kTimes` | int64/64 | offset 120 | loop | floor power of two | cache reduction | 2 的幂 |
| 同上 | `cacheStart` | int64/64 | offset 128 | slot | popcount-derived | final cache address | 非负 |
| 同上 | `db` | int64/64 | offset 136 | Queue depth | TryFullLoad | full-load InitBuffer | 1/2；big default 1 不读 |
| 同上 | `labelSmooth1` | float/32 | offset 144 | scalar | 固定 1.0 | official body | Direct 无 smoothing |
| 同上 | `labelSmoothC` | float/32 | offset 148；size 152 align 8 | scalar | 固定 0.0 | official body | Direct 无 smoothing |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| P0 · R-007 | 0 | `ComputeBigC`（交付件） | `CrossEntropyLossRegBaseTilingData` | dtype entry→`DispatchOfficial` | 官方 `CrossEntropyLossBigC` 的 direct loss-only 转换 | 官方 950 arch35 CMake + direct CMake |
| P1 · R-005 | 4 | `BuildChannelFirstLean` | channelFirst+recompute tile fields | `cross_entropy_loss_fused_vector_*_entry` | lean AIV stats + SIMT gather/reduce；官方 softmax cast trait | direct op CMake（交付件） |
| P2 · R-006 | 1 | `TryFullLoad`（交付件） | `CrossEntropyLossRegBaseTilingData` | dtype entry→`DispatchOfficial` | 官方 `CrossEntropyLossFullLoad` 的 direct loss-only 转换 | 官方 950 arch35 CMake + direct CMake |
| P3 · R-004 | 3 | `FillStreamingHardConfig` 的全部有序出口 | channelFirst common fields | `cross_entropy_loss_fused_*_entry` | `RunCrossEntropyFusedTarget`→SIMT | direct op CMake（交付件） |
| P4 · R-003 | 5, `128<=C<=2^24` | Host soft branch + Device 精确域 router | channelFirst common fields | `cross_entropy_loss_soft_*_entry` | `RunCrossEntropySoft`→`CrossEntropySoftWarpSimt` | 同上 |
| P5 · R-002 | 5, `C<128 \|\| C>2^24` | Host soft branch + Device fallback router | 同上 | 同 R-003 entry | `RunCrossEntropySoft`→`CrossEntropySoftScalarSimt` | 同上 |
| P6 · R-001 | 6 | `calc_cross_entropy_loss_tiling`（交付件） | reduction | `cross_entropy_loss_empty_*_entry` | `RunCrossEntropyEmpty<T>` | 同上 |

## Kernel 实现

### 公共执行流程

三个 dtype launcher 具有相同 Route 图，只替换 `T=half/float/bfloat16_t`。R-002/R-003/R-004 的 SIMT body 采用 `max + log1p(sum_{c!=argmax} exp(x_c-max))`：线程/每 lane 的 tail 用补偿加法，`tail<0.25` 时用 12 项交错级数；warp 的跨 lane 合并仍按树序 FP32 相加。R-005 Hybrid 则由 Vector 直接累计包含最大项的 `sum(exp(x-max))`，SIMT gather 使用普通 `logf(sum)`，没有补偿或小 tail 多项式。R-006/R-007 也在 FP32 UB 中完成 max/sub/exp/reduce，R-006 仅对 FP32+none 额外启用 target-excluded 精化。

写回顺序同样随 body 区分：SIMT/lean 的 primary 通过 `CrossEntropySimtFromFloat<T>` 做最终模板转换；跨 block finalize 与 empty helper 的 BF16 路径由 `CrossEntropyFloatToValue<bfloat16_t>` 显式执行 round-to-nearest-even 位级截取；official-derived 路径以 MicroAPI/AscendC cast 在 FP32 计算后降到输出 dtype。不得把某一路的转换方式概括成所有 Route 的统一指令。

reduced SIMT/Hybrid 的 primary 每 block 写 `(partialSum,partialCount)`；单 block 就地写 loss，多 block 由同 stream 的 finalize 顺序汇总。full-load/big-C 在单 kernel 内以 workspace+`SyncAll` 完成跨核归约。

### Route R-007：rank-2 big-C（P0）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | `CanUseOfficialRank2FastPath` | hard、rank 2、`N>=32,C!=2,N*C>=10^6,N<=2097152,2<=C<=16384,ignoreIndex∈int32` | 允许尝试 official-derived 路线；否则 R-004 |
| Host | direct kernel | `calc_cross_entropy_loss_tiling` | 上述 guard 成立，且 `TryFullLoad/FullLoadEncodingIsSafe` 未共同成立 | 调用 `ComputeBigC` |
| Host | direct kernel | `BigCEncodingIsSafe` | chunk/loop/cache/workspace-stride/DMA 字段全部可编码 | `route=BIG_C`；否则 R-004 |
| Router | direct kernel | `DispatchOfficialRoute` | route 非 full-load | `CrossEntropyLossBigC` 模板实例 |

#### Device 数据流

##### 函数级调用链

```text
cross_entropy_loss_<T>_entry
  -> DispatchOfficial<T,TargetT>
  -> RunOfficialVariant<...,FullLoad=false>
  -> CrossEntropyLossBigC::Init/Process
     -> ComputeMax over C chunks
     -> ComputeReduceTail/ComputeReduceAfter -> UpdateCache
     -> ComputeSubAndLoss -> GatherLoss
     -> ComputeMul -> none CopyOut / reduced ComputeReduceN
     -> reduced workspace -> SyncAll -> core0 ComputeNLast
```

##### 源码等价伪码

```text
for N tile assigned to this core, at most 64 rows:
  DMA target tile
  for each row:
    max = -inf
    for each C chunk: max = max(max, reduce_max(inputChunk))
    for paired/main chunks:
      chunkSum = reduce_sum(exp(inputChunk-max))
      UpdateCache(chunkSum, binary cache slot)
    total = cache[cacheStart]
    for each C chunk:
      shiftedLog = inputChunk-max-log(total)
      if target belongs to chunk: gather shiftedLog[target]
  apply ignore mask and negate gathered values
  none: cast/store rows
  reduced: accumulate per-core loss/count
if reduced: strided workspace write; SyncAll; core0 final mean/sum
```

这里的 fallback 只发生在 Host 选择阶段；一旦 R-007 launch，Device body 内不再改 Route。ignore 与 label 范围判断先于 label-derived class 地址：ignore row 写 0/不计数，语义域外 label 被阻止形成非法地址，其数值结果不作为 API 语义保证。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQueue depth2 | `row*C+k*onceC` | `2*onceC*b` | T/class chunk | `DataCopyInputx` | max/exp/log | TPipe/entry | 双 slot 随 chunk 复用 |
| outQueue + tmpUb | chunk-local | 各 `onceC*4` | FP32 physical capacity | exp/log | fold/reduce/gather | TPipe/entry | 分阶段复用 |
| maxUb/sumUb | scalar block | 各 32B | FP32 | chunk reduce | next chunk/cache | TPipe/entry | VL1 有效 |
| cacheUb | `cacheID*8` FP32 | `(cacheStart+1)*32` B | FP32 | `UpdateCache` | later update/final log | TPipe/row | binary tree slots |
| target/lossUb | row tile | `64*bt + 64*4` | TargetT/FP32 | DMA/gather | mask/output | TPipe/N tile | target depth1 |
| weightUb | row tile | mean only 256B | FP32 count | ignore mask | denominator | TPipe/N tile | Direct `isWeight=0` |
| clearUb/workspace | 0/8/16 slots；三段 GM | 96B；reduced `P*384B` | FP32 | per-core reduction | core0 | TPipe/torch | `SyncAll` 前不可覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["inputQ d2 · 2*onceC*b"]:2
  outtmp["outQ+tmp · 2*onceC*4"]:2
  stats["max+sum+cache"]
  rows["target+loss+conditional count/clear"]
```

##### 字节账与复用顺序

```text
UB_total = 2*onceC*b + 2*onceC*4 + 64*bt + 32 + 32
         + (cacheStart+1)*32 + 64*4
         + (mean ? 64*4 : 0) + (reduced ? 96 : 0)
workspace_total = reduced ? P*384 : 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input depth2 slots | head/tail chunk DMA | max/exp/log | next chunk pair | Queue EnQue/DeQue/FreeTensor | `tailC/alignC` 与真实 chunk 一致 |
| tmp/out chunk | exp/log | FoldAdd/ReduceSum/gather | next phase | Queue release与顺序调用 | gather 完成后才覆盖 |
| cache slot | `UpdateCache(index)` | later fold/final log | next row | cacheID 递增规则 | Host `cacheStart` 覆盖最大 slot |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| chunk DMA | Queue | MTE2 | Vector | input slot | `tailC` 与 32B `alignC` 保护最后块 |
| scalar gather | V→S/S→V event | vector log chunk | scalar target gather | lossUb | 只在 target 落入当前 chunk 时写 |
| cache tree | MicroAPI Add | chunk sums | final log | cacheUb | FP32、power-of-two head/tail fold |
| cross-core | V→MTE3 + `SyncAll` + MTE2→V | each core | core0 | workspace | mean count 排除 ignored row |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | `ComputeBigC` | chunk、cache、N/core 参数生产 |
| Kernel entry/router | direct kernel + official entry | `DispatchOfficial` / `schId==0` | dtype/target/reduction 到 BigC |
| Device body/helper | official big-C header的 direct loss-only转换 | `ComputeMax/ComputeReduce*/UpdateCache/GatherLoss/Process` | class chunk、cache与跨核归约 |

### Route R-005：channel-first lean Hybrid（P1）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | rank-N branch | hard、未命中 `none && N*C*S<=1048576`、rank>2、`N*C*S>=10^6` | 才尝试 lean；小工作量直接 R-004 |
| Host | direct kernel | `BuildChannelFirstLean` | `C<=4095 && maxTile>=32/b`，且 full/tail `blockLen/srcStride/dstStride` 与 tile/class 窄字段可编码 | 生产 channel-first tile POD；否则 R-004 |
| Host | direct kernel | dynamic UB复核 | `align32(inputSlab+2*stat+2*thread)<=min(U,UINT32_MAX)` | route 4；否则 R-004 |
| Router | direct kernel | fused-vector dtype entry | targetType int32/int64 | `RunChannelFirstLean<T,TargetT>` |

#### Device 数据流

##### 函数级调用链

```text
fused_vector_<T>_entry -> RunChannelFirstLean<T,TargetT>
  -> per tile DataCopyPad C blocks GM->inputQueue
  -> CrossEntropyLeanStats<T> (MicroAPI max then exp-sum)
  -> asc_vf_call<CrossEntropyLeanGatherSimt<T,TargetT>>(reserved threads)
  -> DataSyncBarrier<UB> -> FreeTensor
  -> [reduced] asc_vf_call<CrossEntropyLeanBlockReduceSimt<T>>
  -> [reduced && cores>1] finalize entry
```

##### 源码等价伪码

```text
for tile=blockIdx; tile<totalTiles; tile+=usedCoreNums:
  DMA C blocks, each curTileLen elements, into [C,tileLen] UB
  for each FP32 vector segment:
    max[s] = max_c float(local[c,s])
    sum[s] = sum_c exp(float(local[c,s])-max[s])
  SIMT reads label before any class-derived UB address
  ignore -> loss=0 and count+=0; malformed -> loss=NaN without gather
  valid -> gather local[label,tilePosition]
           loss = log(sum)-(targetValue-max)
  none: write valid rows; reduced: retain per-thread partial across local tiles
  UB DSB; release input slab
if reduced: SIMT block tree writes workspace or scalar loss
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQueue depth1 | `c*tileLen+s` | `tileLen*C*b+256` B | T `[C,tile]` | `DataCopyPad` | stats+gather | TPipe/tile | 256B legal guard |
| maxBuffer | s | `tileLen*4+256` B | FP32 | MicroAPI max | SIMT gather | TPipe/entry | 256B guard |
| sumBuffer | s | `tileLen*4+256` B | FP32 | MicroAPI exp sum | SIMT gather | TPipe/entry | 256B guard |
| threadSum/Count | thread | each `threads*4` B | FP32 | SIMT gather | block reduce | TPipe/entry | 跨本 core tile 保留 |
| workspace | `[0,L)`/`[L,2L)` | reduced `8*usedCoreNums` B | FP32 | primary | finalize | torch | same stream |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inputQueue d1 · tile*C*b+256"]:2
  max["max · tile*4+256"]
  sum["sum · tile*4+256"]
  partial["sum/count · 2*threads*4"]
```

##### 字节账与复用顺序

```text
UB_total = tile*C*b + 2*tile*4 + 2*threads*4 + 768
localMemoryBytes = align_up(UB_total,32)
selector requires tile*(C*b+8) <= U-64KiB and threads<=512
workspace_total = reduced ? 8*usedCoreNums : 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputQueue slab | MTE2 | Vector+SIMT | next tile | EnQue/DeQue + UB DSB + FreeTensor | SIMT VF 完成后才允许复用 |
| thread partials | gather tile k | gather tile k+1/block reduce | entry end | `firstTile` + final barriers | reduced only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→Vector | Queue | MTE2 | Vector | input slab | curTileLen 与 dst stride 补尾 |
| Vector→SIMT | program order/`asc_vf_call` | max/sum | gather | statistic TBuf | B16 unpack→FP32；`sum` 含最大项，gather 用普通 `logf(sum)`，无 SIMT accurate-log1p 分支 |
| SIMT→reuse | `DataSyncBarrier<UB>` | gather threads | FreeTensor/next DMA | input slab | 防止 scalar control 提前释放 |
| cross-kernel | same stream | primary | finalize | workspace | multi-core reduced only |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | `BuildChannelFirstLean` | 4095、UB、tile、core guard |
| Kernel entry/router | direct kernel | `cross_entropy_loss_fused_vector_*_entry` | Hybrid single-primary mapping |
| Device body/helper | direct kernel + official softmax base | `RunChannelFirstLean`、`CrossEntropyLeanStats/Gather/BlockReduce`、cast trait | DMA、Vector、SIMT与UB barrier |

### Route R-006：rank-2 full-load（P2）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | `CanUseOfficialRank2FastPath` | hard、未命中 `none && N*C<=1048576`、rank2、`N>=32,C!=2,N*C>=10^6,N<=2097152,2<=C<=16384,ignoreIndex∈int32` | 才尝试 full-load；否则 R-004 |
| Host | direct kernel | `TryFullLoad` | db=2 优先、失败再 db=1；`aNum>=32/b` | 继续编码复核；容量失败转 R-007 候选 |
| Host | direct kernel | `FullLoadEncodingIsSafe` | core/row blockCount、input/target blockLen、`onceN*cOnceNum<=INT32_MAX` | route 1；失败转 R-007 候选 |
| Router | direct kernel | `DispatchOfficial` | route 1 + reduction + ignore sign + target dtype | FullLoad 模板实例 |

#### Device 数据流

##### 函数级调用链

```text
cross_entropy_loss_<T>_entry
  -> DispatchOfficial<T,TargetT>
  -> RunOfficialVariant<...,FullLoad=true>
  -> CrossEntropyLossFullLoad::Init/Process
     -> CopyInInput + CopyInTarget
     -> VfReduceMax -> VfSubExp -> ReduceSum
     -> [FP32 && none] accurate near-one log + target-exp exclusion/tail reduce
     -> VfComputeLogSub -> VfComputeIgnore/safe gather -> VfComputeLossOut
     -> [FP32 && none] tiny-loss target-relative refinement
     -> none: cast+CopyOut; reduced: workspace -> SyncAll -> core0 final
```

##### 源码等价伪码

```text
rows = former/tail-core assignment
for N tile:
  DMA [nTail,C], pad C to cOnceNum with dtype minimum
  FP32 max; shifted=x-max; exp; rowSum; shiftedLog=shifted-log(rowSum)
  FP32+none: for rowSum-1<0.01 use five-term log1p polynomial;
             save exp(target), zero that lane, reduce target-excluded tail
  clamp target before gather; select 0 for ignore and NaN for malformed; negate
  FP32+none and tiny gathered loss: if tail/targetExp<0.01,
             replace by five-term log1p(tail/targetExp)
  none: cast_rint and GM loss
  reduced: accumulate per-core loss/count
if reduced: strided workspace write; SyncAll; core0 gather/finalize
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQueue | `row*C` | `db*onceN*cAlign*b` | T `[Ntile,cAlign]` | CopyInInput | max/subexp | TPipe | db 1/2 |
| targetQueue | row | `align(onceN,32/bt)*bt` | TargetT | CopyInTarget | gather | TPipe | depth1 |
| subUb/tempUb | `row*cAlign` | each `onceN*cAlign*4` | FP32 | sub/exp/log | reduce/gather | TPipe | tile-local |
| row TBufs | row | 5×`align(onceN,8)*4` | FP32 | max/sum/gather | output/reduce | TPipe | five distinct regions |
| lossQueue | row | none only `align(onceN,32/b)*b` | T | cast | GM | TPipe | depth1 |
| partSum+clear/workspace | local scalar slots；three GM segments | reduced 192B；`P*384B` | FP32 | tile/core reduce | core0 | TPipe/torch | events + `SyncAll` |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["inputQ · db*onceN*cA*b"]:2
  target["targetQ · alignN*bt"]
  nc["sub+temp · 2*onceN*cA*4"]:2
  rows["5 row TBuf + conditional lossQ/partials"]
```

##### 字节账与复用顺序

```text
UB_total = db*onceN*cA*b + alignNTarget*bt
         + 2*onceN*cA*4 + 5*alignNF32*4
         + (none ? alignNInput*b : 192)
Direct 不含官方 logProbQueue 的 onceN*cA*b
workspace_total = reduced ? P*384 : 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue slot | CopyIn | max/subexp | next tile | DeQue/FreeTensor | Host 与 Device `db` 一致 |
| temp/sub TBuf | exp/shift/log | ReduceSum/gather | next tile | 顺序 MicroAPI | FP32+none 时 temp 中 target exp 被清零后再做 auxiliary tail reduce；gather/refine 后覆盖 |
| clearUb | per-tile partial | workspace/final | kernel end | V→MTE3 event、`SyncAll` | reduced only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue | Alloc/EnQue/DeQue/Free | MTE2 | Vector | input/target | C padding=min，mask限定真实 C |
| Vector | MicroAPI | input | gather/reduce | FP32 TBuf | B16 unpack/cast，最终 CAST_RINT；FP32+none 对 `rowSum-1<0.01` 与 `tail/targetExp<0.01` 使用五项交错 Horner 多项式 |
| Vector→Scalar→Vector | `VToS/MTE2ToS/SToV` | exp tensor + target DMA | target-exp 排除后 auxiliary reduce | temp/max dead buffer alias | 仅 FP32+none；先校验 target 范围再做 UB 索引，ignore/malformed 不形成非法地址 |
| cross-core | V→MTE3 + `SyncAll` + MTE2→V | each core | core0 | workspace | former/tail cores均写入 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | `FillCommonTiling/TryFullLoad` | db、core、tile producer |
| Kernel entry/router | direct kernel + official entry | `DispatchOfficial` / `schId==2` | FullLoad body mapping |
| Device body/helper | official full-load header的 direct loss-only转换 | `CrossEntropyLossFullLoad::*` | UB、MicroAPI、workspace、tail |

### Route R-004：hard target-only SIMT（P3）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | `calc_cross_entropy_loss_tiling` | accurate-small、rank-N 工作量/lean/dynamic-UB guard、rank-2 小 shape/official-fast/FullLoad/BigC 编码 guard 任一有序出口 | `FillStreamingHardConfig` 写 route 3、`lossCoreNum` 与 workspace |
| Router | direct kernel | fused dtype entry | targetType int32/int64 | 对应 `TargetT` 模板 |

#### Device 数据流

##### 函数级调用链

```text
fused_<T>_entry -> RunCrossEntropyFusedTarget<T,TargetT>
  -> asc_vf_call<CrossEntropyFusedTargetSimt<T,TargetT>>(512)
  -> [reduced && cores>1] finalize
```

##### 源码等价伪码

```text
for row=globalThread; row<R; row+=grid*512:
  if target==ignore: none writes 0; continue
  if target<0 or target>=C: write/accumulate NaN; continue
  scan C for max and argmax
  compensated_sum exp(x[c]-max), excluding argmax
  value = accurate_log1p(tail)+(max-x[target])
  none: write value; reduced: localSum+=value, localCount++
if reduced: 512-thread block tree -> workspace or direct scalar
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `n*C*S+c*S+s` | `N*C*S` | T | caller | scalar class scans | caller | no C-sized UB |
| GM target/loss | row | R | TargetT/T | caller/body | body/caller | caller | ignore guarded |
| threadSums/Counts | thread | each 512×4B | FP32 | threads | block tree | TPipe/entry | reduced only |
| workspace | two core slices | `8*lossCoreNum` B | FP32 | block thread0 | finalize | torch | same stream |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  sums["threadSums · 2048B"]
  counts["threadCounts · 2048B"]
```

##### 字节账与复用顺序

```text
UB_total=4096B
dynamic_ub=N/A
workspace_total=reduced ? 8*lossCoreNum : 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread TBuf | each thread | tree peer/thread0 | entry end | `asc_syncthreads` each level | reduced only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| row stream | grid-stride | GM | same thread | logits | FP32 max/exp/compensated sum |
| ignore | branch | target | output/count | row | ignored row不访问 class、不计数 |
| reduction | barriers + same stream | threads/blocks | finalize | UB/workspace | odd tree guarded；count可为0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | hard selector branches | specialization/fallback |
| Kernel entry/router | direct kernel | `cross_entropy_loss_fused_*_entry` | int32/int64 dispatch |
| Device body/helper | direct kernel | `CrossEntropyFusedTargetSimt`、`CrossEntropyAccurateLog1p` | target-only stable streaming |

### Route R-003：soft warp SIMT（P4）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | soft branch | `targetType==SOFT && C>=128` 时按 rows/block=16 生产 `lossCoreNum` | route 5；Host key 与 scalar 共用 |
| Body router | direct kernel | `RunCrossEntropySoft` | `128<=C<=2^24` | warp SIMT；上界外转 scalar SIMT |

#### Device 数据流

##### 函数级调用链

```text
soft_<T>_entry -> RunCrossEntropySoft<T>
  -> asc_vf_call<CrossEntropySoftWarpSimt<T>>(512)
  -> [reduced && cores>1] finalize
```

##### 源码等价伪码

```text
warp=thread/32; lane=thread%32; global warp owns one row
lane scans c=lane;c<C;c+=32 for max/index
scratch reduction -> max
lane scans classes for compensated exp tail; scratch reduction -> sum
lane scans classes for p[c]*(logZ+max-x[c]); scratch reduction -> rowLoss
lane0 writes row or local block partial; final block tree writes workspace
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| scratchA | thread | 512×4B | FP32 | max then loss then sum | warp/block reduce | TPipe | phase alias after barrier |
| scratchB | thread | 512×4B | FP32 | index then exp then count | warp/block reduce | TPipe | phase alias after barrier |
| GM input/target | channel-first stride S | full tensors | T | lanes | lane reductions | caller | class-striped |
| workspace | two core slices | reduced `8*lossCoreNum` B | FP32 | primary | finalize | torch | same stream |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  a["scratchA · 2048B · max/loss/sum alias"]
  b["scratchB · 2048B · index/exp/count alias"]
```

##### 字节账与复用顺序

```text
UB_total=4096B; dynamic_ub=N/A
workspace_total=reduced ? 8*lossCoreNum : 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| scratchA/B | all lanes | warp tree | next phase | barriers before/within/after phase | inactive rows也必须到达 barrier |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| warp phases | block-wide barrier | lanes | lower lanes | aliased scratch | `rowActive` 只守护 GM，不能跳 barrier |
| class reduction | lane-local + tree | lanes | lane0 | FP32 | class tail由 `c<C` 守护 |
| cross-kernel | same stream | primary | finalize | workspace | multi-core reduced only |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | soft branch | 16 rows/block producer |
| Kernel entry/router | direct kernel | `RunCrossEntropySoft` | `128<=C<=2^24` internal branch；上界保证 FP32 class index 精确 |
| Device body/helper | direct kernel | `CrossEntropySoftWarpSimt<T>` | warp/class mapping与scratch复用 |

### Route R-002：soft scalar SIMT（P5）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | soft branch | `targetType==SOFT`；`C<128` 时按 rows/block=512，`C>=128` 时按 16 生产 `lossCoreNum` | route 5、lossCoreNum |
| Body router | direct kernel | `RunCrossEntropySoft` | `C<128 \|\| C>2^24` | scalar SIMT；超上界时虽沿用 Host 的 16 rows/block grid 估算，body 仍以全局 thread stride 覆盖全部 row |

#### Device 数据流

##### 函数级调用链

```text
soft_<T>_entry -> RunCrossEntropySoft<T>
  -> asc_vf_call<CrossEntropySoftScalarSimt<T>>(512)
  -> [reduced && cores>1] finalize
```

##### 源码等价伪码

```text
for row=globalThread; row<R; row+=grid*512:
  scan C for max/index
  compensated_sum exp(x[c]-max), excluding max index
  logZ=accurate_log1p(tail)
  rowLoss=sum_c target[c]*(logZ+max-x[c])
  none: write row; reduced: accumulate rowLoss/count
if reduced: 512-thread block tree -> workspace/direct scalar
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/target | `n*C*S+c*S+s` | each `N*C*S` | T | caller | thread class scans | caller | streamed |
| sumBuffer/countBuffer | thread | each 512×4B | FP32 | thread partial | block tree | TPipe | reduced only |
| workspace | two core slices | reduced `8*lossCoreNum` B | FP32 | primary | finalize | torch | same stream |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  sums["sumBuffer · 2048B"]
  counts["countBuffer · 2048B"]
```

##### 字节账与复用顺序

```text
UB_total=4096B; dynamic_ub=N/A
workspace_total=reduced ? 8*lossCoreNum : 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| sum/count TBuf | threads | block tree | entry end | `asc_syncthreads` each level | all threads reach barriers |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| row compute | grid-stride | each thread | itself | row | `row<R`；FP32 compensated tail |
| block reduce | `asc_syncthreads` | active slots | next level | two TBuf | odd active count guarded |
| cross-kernel | same stream | primary | finalize | workspace | mean divides valid row count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | soft branch | scalar rows/block 512 |
| Kernel entry/router | direct kernel | `cross_entropy_loss_soft_*_entry`→`RunCrossEntropySoft` | `C<128 \|\| C>2^24` scalar branch |
| Device body/helper | direct kernel | `CrossEntropySoftScalarSimt<T>` | stable row streaming与block reduction |

### Route R-001：empty（P6）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | direct kernel | `calc_cross_entropy_loss_tiling` | `totalTargets==0`，最先短路 | route 6、blockDim 1 |
| Router | direct kernel | dtype launcher | route 6 | empty dtype entry |

#### Device 数据流

##### 函数级调用链

```text
empty_<T>_entry(loss,reduction) -> RunCrossEntropyEmpty<T>
```

##### 源码等价伪码

```text
if blockIdx!=0 or reduction==none: return
loss[0] = reduction==mean ? NaN : 0
flush loss dcache
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM loss | 0 | one scalar for mean/sum | T | core0 | caller | torch output | dcache flush |

##### 分层内存布局

N/A：不分配 AIV UB、dynamic UB 或 workspace，仅由 core 0 标量写 GM。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | single GM write | blockIdx==0 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar output | dcache clean+invalidate | core0 | stream consumer | loss[0] | none无元素；mean NaN；sum 0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | direct kernel | `calc_cross_entropy_loss_tiling` | 空域优先短路 |
| Kernel entry/router | direct kernel | `cross_entropy_loss_empty_*_entry` | grid 1 |
| Device body/helper | direct kernel | `RunCrossEntropyEmpty<T>` | scalar empty semantics |

## 算子特定优化

### Trick T-001：full-load 的 Host 可选 Queue depth

#### 触发与源码依据

- 适用 Route：R-006。
- 触发条件：Direct `TryFullLoad` 先证明 db=2 的 UB 容量；不足时重算 db=1；两者都要求一次至少处理 `32/sizeof(T)` rows。
- 固定源码来源：[`loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_full_load.h:154-162`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_full_load.h#L154-L162)

**官方源码（固定 Commit 节选；省略号表示未展示的连续行）**：

```cpp
int32_t ncOnce = static_cast<int32_t>(tilingData_->onceNSize * tilingData_->cOnceNum);
// ...
uint32_t db = tilingData_->db;
pipe_->InitBuffer(inputQueue_, db, ncOnce * sizeof(T1));
```

#### 机制与设计原因

`db` 是 Host 写入的真实 Queue depth，直接乘入 input Queue 字节账。容量充足时 depth 2 允许 CopyIn 与 Vector 阶段使用不同 slot；容量不足时 depth 1 保留 full-load 的整 C 算法而降低 UB 占用，再不足才切 big-C。Direct loss-only 转换还删除官方 `logPropQueue_`，但该删除属于交付件事实，不在摘录中。

#### 迁移边界

- 适用：整条 class 轴可驻留、Host 能精确计算 `InitBuffer` 总量的 AIV reduction。
- 前提：Host/Device 对 `db`、`onceNSize*cOnceNum` 与 dtype 字节数解释一致。
- 失效：db=1 仍放不下至少一个 DMA 对齐 row 时必须切 R-007。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能只按逻辑有效元素计算 Queue 而漏乘 depth；固定源码以 `InitBuffer(inputQueue_, db, ...)` 明确证明 depth 是物理容量因子。

### Trick T-002：big-C 二叉 cache 避免物化全部 chunk sum

#### 触发与源码依据

- 适用 Route：R-007。
- 触发条件：C 被切成多个 `cOnceNum` chunk，Host 生产 `kTimes/kTimesTail/cacheStart`，Device 对每个 chunk 的 FP32 exp sum 调用 `UpdateCache`。
- 固定源码来源：[`loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_ar_big_c.h:431-453`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_ar_big_c.h#L431-L453)

**官方源码（固定 Commit 节选；省略号表示未展示的连续行）**：

```cpp
const int64_t cacheID = ScalarGetCountOfValue<1>(index ^ (index + 1)) - 1;
auto cahUbAddr = (__ubuf__ float*)dstUb.GetPhyAddr() + cacheID * stride;
// ...
for (uint16_t j = 0; j < innerLoopTimes; ++j) {
    AddrReg srcOffsetJ = AscendC::MicroAPI::CreateAddrReg<float>(j, stride);
    AscendC::MicroAPI::DataCopy(bReg, dstUbAddr, srcOffsetJ);
    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
}
AscendC::MicroAPI::DataCopy(cahUbAddr, aReg, pMask);
```

#### 机制与设计原因

`index^(index+1)` 的 1-bit 个数把连续 chunk 映射到二叉进位层级；当前 sum 只与需要折叠的低层 cache 相加并写入 `cacheID`，所需容量仅 `(cacheStart+1)*32B`，而不是 `cLoopTimes` 个完整 chunk 或标量数组。Host 在选择 `cOnceNum` 时把该 cache 字节数计入剩余 UB。

#### 迁移边界

- 适用：可结合的 FP32 chunk reduction，且 Host 能同步生产 tree 元数据。
- 前提：更新顺序严格递增，cache stride 为 8 个 FP32，最终读取 `cacheStart` 对应 slot。
- 失效：任意乱序并行更新都需要原子/额外同步或另一归约网络，不能直接复用该 cacheID 规则。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若只分配一个 32B slot，`cacheID>0` 会越界；若 Host 的 `cacheStart` 与 Device 的 popcount 规则不一致，最终 log-normalizer 会遗漏 chunk。

### Trick T-003：FP32 full-load 的 target-excluded 小损失精化

#### 触发与转换依据

- 适用 Route：R-006，且模板条件严格为 `T=float && reduction=none && isWeight=0 && labelS=0`。
- 第一层：`VfComputeAccurateLogSum` 对 `x=rowSum-1<0.01` 以五项交错 Horner 多项式计算 `log1p(x)`，否则保留 Vector `Log(rowSum)`。
- 第二层：`PrepareTargetExcludedTail` 在 target 合法且非 ignore 时保存 `targetExp=exp(x_target-max)`、把对应 exp lane 清零，再 ReduceSum 得到 `tail=sum_{c!=target} exp(x_c-max)`；`VfRefineLossFromTargetExcludedTail` 只在初算 loss `<=2^-12`、`tail/targetExp<0.01` 且非 ignore 时，用同一五项多项式替换为 `log1p(tail/targetExp)`。
- 这是交付件对官方 full-load body 的精度派生补充；固定官方源码只证明原始 full-load 的 Queue/MicroAPI 骨架，不证明这组新增函数。

#### 可落地伪码

```text
sum = reduce_sum(exp(logits-max))
logSum = (sum-1 < 0.01) ? poly5_log1p(sum-1) : log(sum)
loss0 = logSum - (logit[target]-max)

targetExp = exp(logit[target]-max)
tail = reduce_sum(expValues with target lane set to 0)
ratio = tail / targetExp
if loss0 <= 2^-12 && ratio < 0.01 && target is valid/non-ignore:
    loss = poly5_log1p(ratio)
else:
    loss = loss0
```

#### 存储、同步与边界

不增加 UB：`maxUb` 在 max/sub-exp 后改作每 row `targetExp`，无 smoothing 时闲置的 `smoothLossUb` 改作 `tail`。Vector 产生 exp、MTE2 产生 target，标量索引前执行 `VToS` 与 `MTE2ToS`，清零 target lane 后执行 `SToV` 再归约。合法性判断先于 UB index；ignore/malformed row 写占位 `targetExp=1` 且被最终 mask 排除。

这条精化不能泛化到 reduced、FP16/BF16、weight 或 smoothing 模板：这些实例不满足编译期条件，仍走常规 FP32 `Log(sum)`/gather 路径。也不能在 `sum` 已把微小 tail 舍入为 1 后才简单改用 `log1p(sum-1)`，因此必须保留 target-excluded auxiliary reduction 的先后顺序。

## 模板与实现对比

| 对比维度 | P0 R-007 big-C | P1 R-005 lean Hybrid | P2 R-006 full-load | P3 R-004 hard SIMT | P4/P5 R-003/R-002 soft SIMT | P6 R-001 empty | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | rank2通过总 guard、整C不可驻留/编码不安全 | hard rank>2、大工作量且C slab/编码安全 | rank2通过总 guard、整C可驻留且编码安全 | hard 小域/lean或official guard fallback | soft，C 阈值与 `2^24` 精确域拆 body | R=0 | Host 按空域→target类型→rank/资源/编码短路 |
| 模板与 dtype | 官方模板 T/TargetT | Vector+SIMT T/TargetT | 官方模板 T/TargetT | SIMT T/TargetT | SIMT T | 标量 T | 三种 input dtype、两种 hard target dtype |
| 循环与分核 | 每核≤64 row、C chunks | tile跨 AIV core | former/tail row core | thread row stride | warp/thread row stride | core0 | 匹配行数、C与UB |
| Queue/TBuf/寄存器/dynamic UB | depth2 chunk+cache | C×tile Queue+stats+partials | db1/2 input+FP32 NC/row buffers | 2×512 FP32 | 2×512 FP32 | 无 | 资源随实现族改变 |
| 内存总占用与复用 | O(cOnceNum)+cache | 公式化 dyn UB，64KiB reserve | full C×N tile | 4096B | 4096B | 0 | 避免不安全 C slab |
| 搬运/Vector/SIMT API | chunk DMA+MicroAPI | DataCopyPad+MicroAPI+SIMT | DataCopyPad+MicroAPI Reduce | GM scalar+SIMT | GM scalar+SIMT | scalar GM | 以空间连续性选择 |
| 同步、精度与 tail | Queue/events/cache/SyncAll | Queue+UB DSB+stream | Queue/events/SyncAll；FP32 none target-excluded 精化 | barriers/FP32/ignore | barriers/FP32 | dcache | 所有降精度在最终写回 |
| fallback/失败边界 | 编码失败回 R-004 | 工作量/C/UB/DMA任一 guard 失败回 R-004 | 容量/编码失败先到 R-007，BigC 仍失败则 R-004 | 最终 streaming fallback | warp index超精确域回 scalar body | 无 | selector 保持合法输入可执行 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本卡不覆盖 | 本卡不覆盖 | 支持；固定官方 CMake 对 `ascend950` 编译 arch35 body，Direct 交付目标为 950 |
| Host/Launcher | 未提取 | 未提取 | Direct 自定义 selector、torch PrivateUse1 与 C ABI launcher |
| Kernel/资源 | 未提取 | 未提取 | DAV-3510；32B UB block、256B vector guard、4095 DMA blockCount gate |

## 限制与失败边界

- Device 只支持连续 dense layout、class axis=1、rank 2～8、FP16/FP32/BF16；Host 接受合法非连续 view 并在 launch 前物化连续副本。不支持 weight、label smoothing、zloss 或 alternate axis。
- hard 标签除 `ignore_index` 外必须在 `[0,C)`；插件未逐元素校验。各 active body 必须在形成 label-derived 地址前处理 ignore/bounds；语义域外标签的数值结果不是 API 保证。所有 hard row 被 ignore 时 mean 的分母为 0，结果按 FP32 除法产生 NaN。
- R-005 的 `C<=4095` 是 `DataCopyPad.blockCount` 编码安全边界；不能只凭 UB 足够就放宽。大 C 由 R-004 从 GM streaming。
- R-006/R-007 还受 `N*C>=10^6`、`N<=2097152`、`2<=C<=16384`、int32 `ignoreIndex` 与各自窄字段编码复核约束；这些是 fast-path capability，不是合法输入上限，失败时走 R-004。
- soft warp 的 FP32 index scratch 只覆盖 `C<=2^24`；更大的合法 C 在同 key/entry 中回 soft scalar body。
- `localMemoryBytes`、workspace、POD offset 均依赖 Host/Device ABI 一致；源码没有 static ABI assertions。
- launcher 返回 `void`，`aclCall` 固定返回 0；异步 kernel 错误不在该返回值中传播。
- 固定官方来源包含多输出、weight/smoothing 等更宽语义；本卡只描述交付件 Direct 子集。融合 SIMT/Hybrid、direct selector、4095 gate、dummy logProb，以及 full-load 删除 `logPropQueue_`、big-C 保留 `outQueue_` 但抑制完整 GM CopyOut，均是交付件转换事实。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 官方 950 entry 基线 | `loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss.cpp` | `cross_entropy_loss`、schId 0/2 | big-C/full-load 官方映射与 VV task type |
| 官方 POD | `loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_tiling_data.h` | `CrossEntropyLossRegBaseTilingData` | official-derived ABI 字段 |
| full-load body | `loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_full_load.h` | `CrossEntropyLossFullLoad` | Queue、MicroAPI、workspace、tail |
| big-C body | `loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_ar_big_c.h` | `CrossEntropyLossBigC`、`UpdateCache` | class chunk 与 cache tree |
| softmax helper | `activation/softmax_v2/op_kernel/arch35/softmax_v2_base.h` | cast traits | lean B16↔FP32 MicroAPI trait 基线 |
| 构建接线 | `loss/cross_entropy_loss/op_kernel/CMakeLists.txt` | `COMPUTE_UNITS ascend950` | 950 arch35 编译选择 |
| 平台 binary | `loss/cross_entropy_loss/op_host/config/ascend950/cross_entropy_loss_binary.json` | CrossEntropyLoss dtype entries | ascend950 官方实现信号 |

上述每个文件均存在于 Frontmatter `sources`。Direct plugin、launch header、定制 kernel/router 与 direct CMake 不伪造官方 URL；其事实在正文统一标注为交付件转换事实。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 152B official POD、344B channel POD、528B config 的字段语义 | 字段顺序、位宽、单位、inactive Route 必须一致 | official tiling data + direct launch config |
| 调用顺序 | selector→dtype launcher→primary→可选 finalize | Route 优先级及同-stream节点顺序不能改 | direct tiler/launcher（交付件） |
| workspace/内存 | SIMT 两段 partial；official 三段32-FP32 stride；lean精确 UB 公式 | offset、extent、depth、guard、owner/lifetime不能缩写 | 各 Run/Init/Process |
| 同步 | SIMT barriers、lean UB DSB、official events+SyncAll、跨 kernel same stream | 任何别名覆盖都必须晚于等待点 | `asc_syncthreads`、`DataSyncBarrier`、`SyncAll` |
| 精度与尾块 | max-shift、FP32 sum/log、最终降精度；mask/rowActive/curTileLen | 不得把 padding lane、ignored row 或 inactive warp计入结果 | SIMT bodies、FullLoad/BigC MicroAPI |
| 失败边界 | 4095 DMA gate、UB capability、int64 CheckedMul、unknown/invalid参数失败 | 不得把 valid fallback 改成非法 Vector launch | direct selector（交付件） |