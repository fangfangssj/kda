---
type: Operator
title: "AdaptiveAvgPool3D 950 Direct Invoke 算子知识"
description: "说明 cann_bench::adaptive_avg_pool_3d、calc_adaptive_avg_pool_3d_config 与 dtype launcher 在 950 上的接口语义、参数生成、Route 选择、单 Kernel 启动图与逐 Route 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_pool3d_tiling_struct.h"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_avg_pool3d/op_kernel/adaptive_avg_pool3d_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_simt.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_pool3d_big_kernel.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_big_kernel.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_parall_pool.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_host/arch35/adaptive_avg_pool3d_simt_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_host/arch35/adaptive_avg_pool3d_big_kernel_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_host/arch35/adaptive_avg_pool3d_para_tiling.cpp"
    role: implementation
created_at: "2026-08-25T19:30:49Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# AdaptiveAvgPool3D 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::adaptive_avg_pool_3d`（Torch schema）→ `adaptive_avg_pool_3d_npu` → `calc_adaptive_avg_pool_3d_config` → `launch_adaptive_avg_pool_3d_{float32,float16,bfloat16}`；Direct Invoke |
| 计算语义 | 对每个 `N,C,od,oh,ow`，按三轴 `floor(o*I/O)` 到 `ceil((o+1)*I/O)` 的自适应窗口做算术平均 |
| 输入输出 | 连续五维 NCDHW；`float16/float32/bfloat16`；输出 `[N,C,outD,outH,outW]` 且 dtype 不变 |
| 执行类型 | R-001/R-002/R-003/R-005 为 SIMT；R-004/R-006 为 AIV_VECTOR；R-007 为 AIV 外壳内 `asc_vf_call` 的 SIMT。全部是单 Kernel，无 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | 七条 terminal Route 都是生产可达的 `active`：R-001 ×2、R-002 /2、R-003 固定因子 4/5/8/9、R-004 parallel、R-005 通用 u32 SIMT、R-006 big、R-007 u64 SIMT 安全终点。无独立 `conditional` terminal Route；保留的 official-SIMT/u32 entry 和 NDHWC template leaf 是 `dormant source`。 |
| 核心资源 | GM；SIMT 寄存器/warp shuffle；parallel 的 3 个 depth-1 Queue 与 4 个 TBuf；big 的 depth-2 输入 Queue、输出 TBuf、累加 TBuf；无用户 workspace |
| 来源闭包 | `direct_implementation_closed`：Direct schema、Meta/NPU glue、Host selector、局部 POD、全部 entry/body/helper 与构建注册已闭合；Frontmatter 的固定官方源码仅证明所转换的 arch35 SIMT/big/parallel 机制，不替代 Direct 交付事实。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入为 `x[n,c,d,h,w]`，输入空间为 `(ID,IH,IW)`，输出空间为 `(OD,OH,OW)`。对轴 `A∈{D,H,W}`，窗口为 `sA=floor(oA*IA/OA)`、`eA=ceil((oA+1)*IA/OA)`；输出为 `sum(x[n,c,sD:eD,sH:eH,sW:eW]) / ((eD-sD)(eH-sH)(eW-sW))`。三个空间轴都允许上采样；窗口可能重叠。Direct 交付只实现 NCDHW，不包含 indices 输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[N,C,ID,IH,IW]`，连续 NCDHW | FP16/FP32/BF16 | rank 必须为 5，所有维度正数；Host 不读取 tensor 值 |
| `output_size` | Host 长度 3 | `int[]` | `(OD,OH,OW)`，每项必须正数；Direct 接口不接受官方接口的长度 0/1 扩展形式 |
| `y` | `[N,C,OD,OH,OW]`，连续 NCDHW | 与 `x` 相同 | 由 Meta/NPU glue 分配，每个有效元素恰好由一个 block/thread 或 AIV core 写入 |

### 入口与版本

Direct 交付的 `op_plugin/adaptive_avg_pool_3d_plugin.cpp` 通过 `TORCH_LIBRARY_FRAGMENT` 注册 schema，以 Meta 实现做 shape/dtype 校验与输出分配，以 `PrivateUse1` 实现取得当前 NPU stream，随后按 dtype 调用本地 launcher；不是 ACLNN 入口。`op_kernel/adaptive_avg_pool_3d_kernel.cpp` 与局部 headers 闭合 selector、POD、entry 和 body，算子目录的 `CMakeLists.txt` 调用 `register_direct_launch_op`。目标身份冻结为 Ascend 950/dav-3510、CANN 9.1.0。固定官方 arch35 源码的 `AdaptiveAvgPool3d` router、SIMT、big、parallel 与 POD 是转换相关来源；ratio-2、固定因子、通用 u32 SIMT、按值 POD 和 Direct selector 属于交付实现，不能反向声称为官方公开 API 行为。

## Host/Launcher 执行流程

### 参数校验与前处理

`adaptive_avg_pool_3d_npu` 先建立 device guard，拒绝非连续输入，再复用输出分配函数检查 rank、三种 dtype、非空输入和正的三项输出尺寸。`calc_adaptive_avg_pool_3d_config` 仅读取 shape、元素字节数、AIV 核数和 UB 大小；用 u64 计算 `inputSize/outputSize/maxDivUse`，再按固定优先级返回第一个 Route。launcher 根据 dtype 与 `route/useIndex64` 选择唯一 kernel entry，在当前 stream 提交。

### 参数、Workspace 与生命周期

`AdaptiveAvgPool3DLaunchConfig` 是 Host-only 聚合体；实际 entry 只按值接收其中一个 POD。`blockDim` 来自 `min(工作项数, AIV 核数)` 或官方式 tiling 的 `useCoreNum/coreNums`，至少为 1。没有显式用户 workspace，launch 的第二参数为 `nullptr`；`x/y/config/stream` 被值捕获到 `OpCommand::RunOpApi` 回调，输出 tensor 持有 GM，直到当前 stream 完成。

### 启动图与失败传播

每次调用只启动一个 Device 节点：

```text
adaptive_avg_pool_3d_npu
  -> calc_adaptive_avg_pool_3d_config
  -> dtype launcher
       -> exactly one of R-001..R-007 entry(x, y, POD) on current stream
```

没有 Kernel 间 workspace 边、event 或 owner 交接。Host 校验失败会在 launch 前抛出；Device launcher 返回 `void`，Direct glue 的回调固定返回 0，因此交付层没有逐 launch 返回码传播。

## Route 与参数设计

### 分核与资源计算

记 `I=N*C*ID*IH*IW`、`O=N*C*OD*OH*OW`。selector 对每个空间轴计算 `B_A=CalKernelSizeOneDimMax(IA,OA)`，再令 `K_sel=B_D*B_H*B_W`：`OA<=10000` 时 helper 穷举该轴所有输出并给出精确最大窗长；`OA>10000` 时直接返回 `ceil(IA/OA)+1`，只是保守上界。因此下文的 350 分流均是 `K_sel<350` / `K_sel>=350` 的 selector bound；当三个输出轴都不超过 10000 时，`K_sel` 等于真实最大三维窗口体积，若有超限轴，仅在每个近似轴的 `ceil(IA/OA)+1` 恰好等于该轴真实最大窗长时两者才相等。R-001/2/3/5 将 `[0,total)` 按 `perBlock=total/blockDim`、`remainder=total%blockDim` 连续均分，每块内的 1024 个 SIMT thread 跨步遍历；R-001 的 `total=I`，其余三条为 `O`，`blockDim=min(total,AIV cores)`。R-007 不做连续块分段，而是 `idx=block*1024+thread; idx+=grid*1024`。R-006 用 `blockFactor=O/coreCount`、`blockTail=O%coreCount`，前 `blockTail` 核多一个输出。R-004 令 `vfLen=256/dtypeBytes`、`ncFactor=vfLen`，依次二分 `do/ho/woFactor` 使 UB 可容纳，再缩因子而不增加每核 outer 数，形成最后一核独立 `blockTail` 的 outer-block 分配。

### 生产可达性分类与穷尽证明

`active/conditional/dormant` 表示生产调用链可达性，不把普通 shape 谓词本身误写成 `conditional`。七条 Route 都由 `calc_adaptive_avg_pool_3d_config` 的合法输入分支直接产生，且各有 dtype launcher、全局 entry 和 Device body，因此都是 `active` terminal Route。没有依赖编译开关、运行能力探测或前置 Kernel 成功才能发射的独立 `conditional` Route；R-004 的 `ParaTiling::Evaluate().capable` 是同一 Host selector 内的纯 shape/platform/UB 谓词，命中即为 active 终点。

| 类别 | Route | 生产 producer → launcher/entry 闭包 | terminal/fallback 出边 |
|---|---|---|---|
| `active` | R-001 | selector 写 `route=3,FastTiling(input)` → typed launcher 的 `route==3` → `*_up2_*` | Host 谓词不成立才继续 R-002；发射后无重试 |
| `active` | R-002 | selector 写 `route=4,FastTiling(output)` → `route==4` → `*_down2_*` | 未命中才继续 R-003；Device 无 fallback |
| `active` | R-003 | selector 写 `route=6/7/8/9,GeneralTiling` → launcher 因子分支 → 12 个 `*_exactF_*` entry | 未命中才评估 R-004；entry 内 group8/scalar 是 body 变体而非新 Route |
| `active` | R-004 | `CalcOfficialParaTiling` 返回 `capable=true`，写 `route=2,ParaTiling,useIndex64` → i32/i64 para entry | capability 为 false 才继续 R-005/R-006/R-007；发射后无回退 |
| `active` | R-005 | 两个 Host 出口均写 `route=5,GeneralTiling` → general entry → group8/scalar | 大窗口受控域未命中才转 R-006；u32 普通末支已是该 index 域终点 |
| `active` | R-006 | `K_sel>=350` 末支写 `route=1,BigTiling` → big entry | 正常 selector 不提供发射失败后回退 |
| `active` | R-007 | route 保持零且 `useIndex64=1`，通用 launcher `else` → `*_u64` → `AdaptiveAvgPool3DSimt<T,u64,1>` | 合法 config 的大索引安全终点；无后续 Route |

该分支链对通过 Host 合同检查的输入是完备的：前四条专用谓词均不成立后，`K_sel>=350` 时不是受控 R-005 就是 R-006；`K_sel<350` 时则由 `maxDivUse<INT32_MAX` 的 R-005 与其补集 R-007 封口。dtype 只在已选 Route 之后三分，不增加 Route。

下列是从 selector 谓词直接构造的可达性 witness（shape 写为 `N,C,ID,IH,IW → OD,OH,OW`），用于排除“有 entry 但 Host 永不产生”的假 Route：

| Route | 谓词 witness | 为何绕过前置分支 |
|---|---|---|
| R-001 | `1,1,2,2,2 → 4,4,4` | 三轴精确 ×2，输入轴均是 2 的幂 |
| R-002 | `1,1,4,4,4 → 2,2,2` | 不是上采样；三轴精确 /2，输出轴均是 2 的幂 |
| R-003 | `1,1,8,8,8 → 2,2,2` | 不是 /2；三轴统一 factor 4 且 outputSpatial=8 |
| R-004 | FP32 `1,64,1,1,8 → 1,1,7` | 非统一 exact factor；`K_sel=2`、`NC=64>=vfLen/2`；`wiDataLen=7*2=14`、`alignNum=8`，余数 6>4；最小/full tile 的 UB 账约 12 KiB |
| R-005 | `1,1,3,3,3 → 2,2,2` | 非 ratio-2 power-of-two/固定 factor，NC 不足以走 parallel，`K_sel<350`且 `maxDivUse<INT32_MAX` |
| R-006 | `1,1,8,8,8 → 1,1,1` | factor 8 但 `outputSpatial==1` 使 R-003 拒绝；`K_sel=512`，受控 R-005 也要求 outputSpatial>1 |
| R-007 | `1,1,46341,1,1 → 46341,1,1` | `ID*OD=2147488281>=INT32_MAX`，但 I/O 各只 46341；非专用 ratio，NC 不足 parallel，`K_sel=1`不走 big |

### Dormant source 与官方图外节点

| dormant 源 | 保留语义 | 生产链不可达原因 | 重新启用条件 |
|---|---|---|---|
| D-001 `adaptive_avg_pool_3d_{f32,f16,bf16}_u32` | official-related `AdaptiveAvgPool3DSimt<T,u32,1>`；64 B tiling + 32 B division 元数据后做 NCDHW SIMT | launcher 的最后 `else` 虽保留 u32 分支，但 Host 在 `useIndex64=0` 时总是先写 `route=5`，会于 general launcher 分支提前返回 | 增加明确 Host 谓词产生 `route=0,useIndex64=0`，并重新核对它与 R-005 的精度顺序、分核和性能边界 |
| D-002 `AdaptiveAvgPool3DNdSimtCompute` / `FORMAT_TYPE=0` | NDHWC 的 N/D/H/W 解码，C 在最内层 | Direct 六个 official-SIMT entry 全部实例化 `FORMAT_TYPE=1`，plugin 又固定连续 NCDHW | 先扩展 layout 合同和 Host POD 解释，再增加 FORMAT=0 的显式 entry/launcher/selector；该 leaf 的 `magicC/shiftC` 写在现有 UB barrier 之后，重启时还必须将 barrier 移到 slots 6/7 写完之后，或在写后新增第二个 `DataSyncBarrier<UB>`；只有文档化契约另外证明当前 scalar→SIMT UB 可见性时才可不改 |

官方 `adaptive_avg_pool3d_apt.cpp` 的 `workspace,tiling` GM 入参、template key router 与 registry Host 的 16 MiB system workspace 没有进入 Direct 生产图；Direct 改为按值 POD 和显式 dtype/route entry。它们是转换基线关系，不能作为 Direct fallback 或资源占用的证明。

### Route 选择与重要性

以下重要性视图用于安排实现阅读和复用顺序，不参与 Host 分发；真实短路关系仍以其后的 Host 顺序表和决策树为准。

| 重要性 | Route | 状态 | 路线特征 | 适用域 | 实现前提 |
|---:|---|---|---|---|---|
| P0 | R-005 通用 u32 SIMT | active coverage | magic division 解码；按输出规模和窗口上界选择标量或 8-lane 协作归约 | 未命中更专用分支的常规自适应窗口，以及受控的大窗口中等输出域 | `maxDivUse<INT32_MAX`；按值 `GeneralTiling` 的 magic/shift 与 OD/OH/OW 一致 |
| P1 | R-006 big | active safety/large-window | UB 分块搬运，按 global 或 D→H→W 降级切窗，FP32 partial 累加并批量写回 | selector 保守窗口上界 `K_sel` 不小于阈值，且未被受控通用 SIMT 分支接管 | `maxCount>0`；depth-2 输入 Queue、输出区和累加区满足 UB 账；NCDHW stride 不变 |
| P2 | R-003 固定因子 4/5/8/9 | active specialized | 编译期固定三维窗口；标量或 group8 SIMT | 三轴统一整除缩小，固定因子属于 `{4,5,8,9}` 且输出空间不退化为单点 | `I,O<=UINT32_MAX`；factor 与三轴比例完全一致 |
| P3 | R-002 power-of-two /2 | active specialized | 位移/掩码解码，显式读取八个输入并做 FP32 平均 | 三轴均精确 /2 下采样 | 输出三轴为 2 的幂；`I,O<=UINT32_MAX` |
| P4 | R-001 power-of-two ×2 | active specialized | 将 16/32-bit 输入值打包复制到八个输出位置 | 三轴均精确 ×2 上采样 | 输入三轴为 2 的幂；`I,O<=UINT32_MAX` |
| P5 | R-004 parallel | active fast | NC lane 转置，W→H→D 三级 FP32 staged reduction | selector 窗口上界 `K_sel<350`、NC 充足且 tile 的 UB 利用和 W 尾条件合适 | capability、UB byte formula、16 行 transpose 与 32 B 对齐全部成立 |
| P6 | R-007 u64 SIMT fallback | active safety | AIV 外壳生成 u64 magic 元数据，UB barrier 后 grid-stride SIMT | 其余需要 u64 索引的非 big 窗口 | `maxDivUse>=INT32_MAX`；128 B UB 元数据在 `asc_vf_call` 前可见 |

| Host 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 ×2 上采样 | `route=3` | 三轴 `O=2I`，`ID/IH/IW` 均为 2 的幂，且 `I,O<=UINT32_MAX` | R-002 |
| 2 | R-002 /2 下采样 | `route=4` | 三轴 `I=2O`，`OD/OH/OW` 均为 2 的幂，且 `I,O<=UINT32_MAX` | R-003 |
| 3 | R-003 固定因子 | `route=6/7/8/9` | `ID/OD=IH/OH=IW/OW=F∈{4,5,8,9}` 且整除，`OD*OH*OW>1`，`I,O<=UINT32_MAX` | R-004 |
| 4 | R-004 parallel | `route=2`，`ID_T=i32/i64` | `K_sel<350`、`N*C>=vfLen/2`、最小块 UB 可容纳、调因子后的 UB 利用/尾 W guard 全部成立 | R-005/R-006/R-007 |
| 5 | R-005 通用 u32 SIMT | `route=5` | R-001..4 未命中且 `maxDivUse<INT32_MAX`；若 `K_sel>=350`，还要求 `OD*OH*OW>1 && O<=coreCount*512` 才抢在 big 前 | R-006 或 R-007 |
| 6 | R-006 big | `route=1` | 上述均未命中且 `K_sel>=350` | 无运行 fallback |
| 7 | R-007 官方式 u64 SIMT fallback | `route=0,useIndex64=1` | 上述均未命中、`K_sel<350` 且 `maxDivUse>=INT32_MAX` | 无运行 fallback |

```text
validate
  -> up2 power-of-two?
  -> down2 power-of-two?
  -> exact factor 4/5/8/9?
  -> official parallel IsCapable?
  -> K_sel>=350 and bounded medium u32? general-u32
  -> K_sel>=350? big
  -> u32-safe? general-u32
  -> official-style u64 SIMT fallback
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-005 | 5 | 剩余 u32-safe | 单 SIMT | `AdaptiveAvgPool3DGeneralFast/Group8<T>` | magic division；标量或 8-lane shuffle | `total<=grid*256` 或大窗且 `<=grid*512` 用 group8 |
| R-006 | 1 | `K_sel>=350` 且优先路线失败 | 单 AIV_VECTOR | `AdaptiveAvgPool3dBigKernel<T>` | depth-2 CopyIn；FP32 ReduceSum；D/H/W split | 1024 输出批次、core tail、split tail |
| R-003 | 6..9 | `F=4/5/8/9` | 单 SIMT | `AdaptiveAvgPool3DExactFactor{Group8}<T,F>` | 标量或 8-lane shuffle 归约 | `total<=grid*256` 用 group8，否则标量 |
| R-002 | 4 | 上表谓词，三 dtype，950 | 单 SIMT | `AdaptiveAvgPool3DDownsample2Power2<T>` | 8 次 GM load，FP32 累加 | block 连续尾 + thread 跨步 |
| R-001 | 3 | 上表谓词，三 dtype，950 | 单 SIMT | `AdaptiveAvgPool3DUpsample2Power2<T>` | GM 直接读写，无 UB；16/32-bit 值分别打包到 32/64-bit 写 | block 连续尾 + thread 跨步 |
| R-004 | 2 | parallel capability | 单 AIV_VECTOR | `AdaptiveAvgPool3dParaPool<T,i32/i64>` | CopyInput 与下一块预取；transpose；W/H/D 分段 FP32 reduction | nc/do/ho/wo 各自 tail |
| R-007 | 0/u64 | 大索引 fallback | 单 AIV 外壳+SIMT | `AdaptiveAvgPool3DSimt<T,uint64_t,1>`→`AdaptiveAvgPool3DNcSimtCompute` | 128 B UB 元数据，UB barrier 后 1024-thread VF | grid-stride tail |

### 关键性能与安全 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| u32 边界 | `max(ID*OD,IH*OH,IW*OW,I,O)<INT32_MAX` | magic division 与线性 offset 为 u32 | R-005 | R-006/R-007 | `calc_adaptive_avg_pool_3d_config` |
| ratio-2 幂次 | 对被位掩码拆分的三轴要求 power-of-two | `idx&(dim-1)` 与 shift 代替除模 | R-001/R-002 | 后续 Route | 同上 |
| parallel 窗口 | `K_sel<350` | staged UB reduction 的目标实现域 | R-004 | R-005/R-006/R-007 | `ParaTiling::Evaluate` |
| parallel NC | `N*C>=vfLen/2` | transpose 以一个 vector-length NC tile 工作 | R-004 | 后续 Route | 同上 |
| parallel UB | `M*s+2*M*4+四个对齐 i32 区<=UB` | 三 Queue 与索引 TBuf 必须同时存在 | R-004 | 后续 Route | `CalOccupySize` |
| parallel W 尾 | `(woFactor*kernelWMax)%alignNum>alignNum/2`，并受 224 KiB 利用规则约束 | 避免低效短 W DMA/低 UB 利用 | R-004 | 后续 Route | `ParaTiling::Evaluate` |
| big UB | `available=UB-1024*s-4096`，`maxCount=floor_align(available/2,256)/4` | depth-2 input queue、输出与累加保留 | R-006 | selector 不提供失败回退 | `calc_adaptive_avg_pool_3d_config` |
| group8 | `total<=grid*256`，通用路线另允许 `maxWindowBound>=256 && total<=grid*512` | 8 lanes 协作一个输出，避免小输出/大窗口欠并行 | R-003/R-005 | 同 entry 的标量 body | entry 宏 |

### POD/ABI

所有 Device POD 都是同类型标量连续声明；无显式 `alignas`。下表 offset 按其声明顺序与自然对齐给出；Host-only `LaunchConfig` 不整体传 Device。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `FastTiling` | `total` | u32 | 28 B，offset 0 | element | `I` 或 `O` | SIMT loop bound | u32-safe |
| 同上 | `d` | u32 | offset 4 | element | `ID` 或 `OD` | bit decode | power-of-two |
| 同上 | `h` | u32 | offset 8 | element | `IH` 或 `OH` | bit decode | power-of-two |
| 同上 | `w` | u32 | offset 12 | element | `IW` 或 `OW` | bit decode/packed stride | power-of-two |
| 同上 | `shiftD` | u32 | offset 16 | bit | `log2(d)` | right shift | active R-001/2 |
| 同上 | `shiftH` | u32 | offset 20 | bit | `log2(h)` | right shift | active R-001/2 |
| 同上 | `shiftW` | u32 | offset 24 | bit | `log2(w)` | right shift | active R-001/2 |
| `GeneralTiling` | `total` | u32 | 52 B，offset 0 | output element | `O` | block range | u32-safe |
| 同上 | `inD` | u32 | offset 4 | element | `ID` | window/address | positive |
| 同上 | `inH` | u32 | offset 8 | element | `IH` | window/address | positive |
| 同上 | `inW` | u32 | offset 12 | element | `IW` | window/address | positive |
| 同上 | `outD` | u32 | offset 16 | element | `OD` | decode/window | positive |
| 同上 | `outH` | u32 | offset 20 | element | `OH` | decode/window | positive |
| 同上 | `outW` | u32 | offset 24 | element | `OW` | decode/window | positive |
| 同上 | `magicD` | u32 | offset 28 | magic | Host division helper | `UintDiv` | divisor `OD` |
| 同上 | `magicH` | u32 | offset 32 | magic | Host division helper | `UintDiv` | divisor `OH` |
| 同上 | `magicW` | u32 | offset 36 | magic | Host division helper | `UintDiv` | divisor `OW` |
| 同上 | `shiftD` | u32 | offset 40 | bit | Host division helper | `UintDiv` | paired with magicD |
| 同上 | `shiftH` | u32 | offset 44 | bit | Host division helper | `UintDiv` | paired with magicH |
| 同上 | `shiftW` | u32 | offset 48 | bit | Host division helper | `UintDiv` | paired with magicW |
| `SimtTiling` | `nDim` | i64 | 64 B，offset 0 | element | `N` | output size | positive |
| 同上 | `cDim` | i64 | offset 8 | element | `C` | output size/NC base | positive |
| 同上 | `dInDim` | i64 | offset 16 | element | `ID` | window/address | positive |
| 同上 | `hInDim` | i64 | offset 24 | element | `IH` | window/address | positive |
| 同上 | `wInDim` | i64 | offset 32 | element | `IW` | window/address | positive |
| 同上 | `dOutDim` | i64 | offset 40 | element | `OD` | decode/window | positive |
| 同上 | `hOutDim` | i64 | offset 48 | element | `OH` | decode/window | positive |
| 同上 | `wOutDim` | i64 | offset 56 | element | `OW` | decode/window | positive |
| `BigTiling` | `nc` | i64 | 104 B，offset 0 | element | `N*C` | 元数据；body 不依赖分核 | positive |
| 同上 | `dInDim` | i64 | offset 8 | element | `ID` | address/window | positive |
| 同上 | `hInDim` | i64 | offset 16 | element | `IH` | address/window | positive |
| 同上 | `wInDim` | i64 | offset 24 | element | `IW` | address/window | positive |
| 同上 | `dOutDim` | i64 | offset 32 | element | `OD` | output decode | positive |
| 同上 | `hOutDim` | i64 | offset 40 | element | `OH` | output decode | positive |
| 同上 | `wOutDim` | i64 | offset 48 | element | `OW` | output decode | positive |
| 同上 | `blockFactor` | i64 | offset 56 | output/core | `O/coreCount` | block range | may be 0 only when coreNums=O |
| 同上 | `blockTail` | i64 | offset 64 | output | `O%coreCount` | former-core branch | `[0,coreCount)` |
| 同上 | `totalIdx` | i64 | offset 72 | output | `O` | tiling identity | positive |
| 同上 | `coreNums` | i64 | offset 80 | core | `min(O,coreCount)` | launch blockDim | positive |
| 同上 | `maxCount` | i64 | offset 88 | FP32 element budget | UB formula | CopyIn/split | must be positive |
| 同上 | `batchCount` | i64 | offset 96 | window | `maxCount/trueWindowVolume` | 保留字段；交付 body 不读 | inactive consumer |
| `ParaTiling` | `dIn` | i64 | 184 B，offset 0 | element | `ID` | D window/copy | positive |
| 同上 | `hIn` | i64 | offset 8 | element | `IH` | H window/copy | positive |
| 同上 | `wIn` | i64 | offset 16 | element | `IW` | W window/copy | positive |
| 同上 | `dOut` | i64 | offset 24 | element | `OD` | D block/window | positive |
| 同上 | `hOut` | i64 | offset 32 | element | `OH` | H block/window | positive |
| 同上 | `wOut` | i64 | offset 40 | element | `OW` | W block/window | positive |
| 同上 | `useCoreNum` | i64 | offset 48 | core | outer tiling | Init/Process guard | positive |
| 同上 | `blockFactor` | i64 | offset 56 | outer/core | ceil division | core range | positive |
| 同上 | `blockTail` | i64 | offset 64 | outer | tail formula | last-core range | positive |
| 同上 | `ncFactor` | i64 | offset 72 | element | `vfLen` | NC tile/address | 256/s |
| 同上 | `doFactor` | i64 | offset 80 | element | UB search | D tile | positive |
| 同上 | `hoFactor` | i64 | offset 88 | element | UB search | H tile | positive |
| 同上 | `woFactor` | i64 | offset 96 | element | UB search | W tile | positive |
| 同上 | `ncOuter` | i64 | offset 104 | outer block | `ceil(NC/ncFactor)` | block decode | positive |
| 同上 | `doOuter` | i64 | offset 112 | outer block | `ceil(OD/doFactor)` | block decode | positive |
| 同上 | `hoOuter` | i64 | offset 120 | outer block | `ceil(OH/hoFactor)` | block decode | positive |
| 同上 | `woOuter` | i64 | offset 128 | outer block | `ceil(OW/woFactor)` | block decode | positive |
| 同上 | `ncTail` | i64 | offset 136 | element | NC remainder | last NC tile | positive |
| 同上 | `doTail` | i64 | offset 144 | element | OD remainder | last D tile | positive |
| 同上 | `hoTail` | i64 | offset 152 | element | OH remainder | last H tile | positive |
| 同上 | `woTail` | i64 | offset 160 | element | OW remainder | last W tile | positive |
| 同上 | `maxInputSize` | i64 | offset 168 | element | aligned max tile×ncFactor | three Queue capacities | UB-bounded |
| 同上 | `maxDimOut` | i64 | offset 176 | element | max factor | start-index TBuf | positive |

### Route 源码闭包

| Route | Key/selector | Direct Host/Tiling producer | POD/参数 | Direct launcher/entry | Device body/转换关系 | 构建接线 |
|---|---|---|---|---|---|---|
| R-005 | 5 | `adaptive_avg_pool_3d_kernel.cpp::calc_adaptive_avg_pool_3d_config` 的受控大窗口出口或最后 u32 出口 | `GeneralTiling` 52 B，按值 | typed launcher `route==5` → `adaptive_avg_pool_3d_general_<dtype>` | Direct 派生 `GeneralFast/GeneralGroup8`；复用官方 magic-division 思路，不是官方原生 entry | 算子 `CMakeLists.txt` 将 kernel/plugin 交给 `register_direct_launch_op` |
| R-006 | 1 | 同文件 `K_sel>=350` selector-bound 分支，按固定公式填 `BigTiling` | `BigTiling` 104 B，按值 | typed launcher `route==1` → `*_big_<dtype>` | official big base/average 本地化；Direct 另增 global contiguous `CopyInGlobalContiguous/ComputeSplitGlobal` | 同上 |
| R-003 | 6..9 | 同文件 uniform exact-factor 分支 | `GeneralTiling` 52 B，按值 | typed launcher 的 4 个 factor 分支 → 12 个 `*_exactF_<dtype>` entry | Direct 派生 fixed-factor scalar/group8 SIMT | 同上 |
| R-002 | 4 | 同文件 exact-down2 分支 | `FastTiling` 28 B，按值 | typed launcher `route==4` → `*_down2_<dtype>` | Direct 派生 power-of-two /2 SIMT | 同上 |
| R-001 | 3 | 同文件 exact-up2 分支 | `FastTiling` 28 B，按值 | typed launcher `route==3` → `*_up2_<dtype>` | Direct 派生 power-of-two ×2 packed-replication SIMT | 同上 |
| R-004 | 2 | `adaptive_avg_pool_3d_official_tiling.h::ParaTiling::Evaluate` 由 config 调用并值拷贝结果 | `ParaTiling` 184 B，按值 | typed launcher `route==2` 再按 `useIndex64` 选 i32/i64 para entry | 官方 parallel Host/device 机制的自包含本地化；local deps 替代原公共 helper | 同上 |
| R-007 | 0/u64 | config 预先填八个 `SimtTiling` 字段，最后写 `useIndex64=1` 而 route 保持 0 | `SimtTiling` 64 B，按值 | typed launcher 通用 `else` 的 u64 分支 → `*_u64` | 官方 `AdaptiveAvgPool3DSimt<T,u64,FORMAT=1>` 本地化；Direct 用显式 entry 取代 template-key router | 同上 |

### 窗口、精度顺序与边界闭包

除 R-001 的位级复制外，所有 Route 都用同一三轴窗口 `start=floor(o*I/O)`、`end=ceil((o+1)*I/O)`，输入小于输出时窗口可重叠但仍非空。精度等价不意味浮点加法顺序相同；复用路线时必须保留下表的实际顺序和末次 cast。

| Route | 窗口/索引实现 | FP32 归约顺序 | 尾块与最终 cast |
|---|---|---|---|
| R-001 | power-of-two mask/shift 把每个输入映射到 2×2×2 输出 | 无算术；16/32-bit raw bits 分别复制成 32/64-bit packed word，保留 NaN payload、符号零和 Inf 位型 | `idx<blockEnd`；四个 packed store 恰覆盖八个输出 |
| R-002 | mask/shift 解码输出，窗口固定 2×2×2 | 严格按当前 D 面 `{0,1,+inW,+inW+1}`，再下一 D 面同序累加，乘 `0.125F` | `idx<blockEnd`；只在累加后 cast 一次 |
| R-003 | `start=(od,oh,ow)*F`，窗固定 `F³` | scalar 按 D→H→W 顺序；group8 各 lane 按 D→H→`W=lane+8k`，再 shuffle 4→2→1 | scalar/thread 或 group/index guard；lane0 除 `F³` 后 cast/store |
| R-004 | 每个 tile 先向 i32 TBuf 写三轴 start/kernelSize | 转置后每个 NC lane 按 W 窗、H 窗、D 窗三级求和，最后一次除 `kD*kH*kW` | `nc/do/ho/woTail` 独立；FP16 `CAST_NONE`、BF16 `CAST_RINT`，FP32 直接 CopyOut |
| R-005 | magic division 同时解码 output 和 floor/ceil 窗口 | scalar 按 D→H→W；group8 按 D→H→`W=lane+8k`，再 shuffle 4→2→1 | `idx<blockEnd`；lane0 或 thread 除真实 window 后 cast |
| R-006 | i64 解码；非 global 窗按 W 32 B padding，global 使用真实连续元素数 | 每个 VF mask 下 `ReduceSum`，repeat 之间 Add；split 再按 global/D/H/W chunk 顺序累加 `storeAddUB[0]`，最后除真实 `kD*kH*kW` | vector mask 排除 repeat 尾；chunk 尾、1024-output flush 尾和 core 商余全部独立 |
| R-007 | u64 magic division 解码与 floor/ceil | 每线程按 D→H→W 标量 FP32 累加 | grid-stride loop 以 `idx<O` 收尾，除真实 window 后 cast |

## Kernel 实现

### 公共执行流程

所有 Route 的 entry 都声明 `KERNEL_TYPE_AIV_ONLY`，但 R-001/2/3/5/7 在 `asc_vf_call` 内按 SIMT 执行；R-004/6 直接执行 AIV vector body。执行族按实际 body 标记，不按 entry 宏的统一名称推断。三种 dtype 仅改变 load/store/cast 模板，窗口边界一致。FP16/BF16 的所有平均路线先转 FP32 累加，再除窗口体积并降回原 dtype。所有路线都不分配 L1、L0A/L0B/L0C、dynamic UB 或 GM workspace；单 Kernel 图没有跨节点同步，同步只发生在同一 entry 的 UB、Queue、MTE 或 warp 内。

### Route R-005：通用 u32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | remaining u32-safe，含 bounded medium override | route 5 |
| Entry | 同上 | general entry | total/maxWindowBound | group8 或 scalar |

#### Device 数据流

##### 函数级调用链

```text
general entry -> AdaptiveAvgPool3DGeneralGroup8<T> OR AdaptiveAvgPool3DGeneralFast<T>
```

##### 源码等价伪码

```text
decode idx to nc,od,oh,ow with magic division
compute floor starts and ceil ends with the same magic division
sum FP32 over adaptive 3D window
group8 partitions W by lane and shuffle-reduces; scalar uses one thread/output
cast(sum/window) to T
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `nc*ID*IH*IW+(d*IH+h)*IW+w` | adaptive window | T | caller | thread/lane | caller | scalar loads |
| warp registers | lane-local | one FP32/lane | FP32 | group8 | lane0 | one output | width 8 |
| GM y | idx | O | T | thread/lane0 | caller | output | guarded store |

##### 分层内存布局

N/A：无 AIV UB，寄存器直接归约 GM 数据。

##### 字节账与复用顺序

```text
UB_total = 0; dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| lane/thread sum | window loop | divide/store | next output | thread/warp program order | output assignment disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group reduction | shuffle 4/2/1 | lanes | lane0 | one output | FP32；idx<blockEnd and iw<endW |
| scalar | N/A | one thread | store | one output | FP32；thread-stride tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | config + division helper | route、magic/shift、blockDim |
| Kernel entry/router | 同上 | general dtype entries | group8/scalar predicate |
| Device body/helper | 同上 | general bodies | adaptive bounds、FP32 average |

### Route R-006：big split vector reduction

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | priority routes failed and `K_sel>=350`；直接填 `BigTiling` | route 1 |
| Direct launcher | 同上 | `launch_adaptive_avg_pool_3d_<dtype>` 的 `route==1` | dtype 已由 plugin 确定 | 显式 big dtype entry |
| 固定官方转换基线 | `adaptive_avg_pool3d_big_kernel_tiling.cpp` / `adaptive_avg_pool3d_apt.cpp` | `IsCapable/DoBlockTiling` / template mode 1 | 解释 350 阈值、POD 字段和 BigKernel 关系；不是 Direct 运行 router | Direct 以按值 POD 和显式 entry 取代 |

#### Device 数据流

##### 函数级调用链

```text
big entry -> BigKernel::Init -> Process
  -> CalcWindowSize -> BaseCompute
     -> NoSplitProcess OR SplitProcess -> ComputeSplitGlobal/D/H/W
     -> ComputeSum<float> -> ComputeAvg<float>
  -> CopyOut every 1024 outputs and final tail
```

##### 源码等价伪码

```text
partition linear outputs by former/tail core
for each output: calculate adaptive window and aligned W/H/D extent
  if global output [1,1,1]: contiguous DMA, split into maxCount chunks if needed
  else copy padded rows; split on D else H else W when window exceeds maxCount
  vector load/cast to FP32; ReduceSum each vector; accumulate split partial in storeAddUB
  divide once by true, unpadded kD*kH*kW; cast into outputUB
flush outputUB to GM in groups of 1024
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inputQue` depth 2 | current window/chunk | `2*maxCount*s` bytes | T, W padded unless global | CopyIn | ComputeSum | core/process | 32 B padding |
| `outputUB` | local output index | `1024*s` bytes | T | ComputeAvg | CopyOut | core batch | reused each flush |
| `storeAddUB` | offset 0 | Device `InitBuffer` 声明 1024 B；FP32 的 16 次连续 64-float store 触达 `[0,4096)` B；FP16/BF16 的 8 次 64-float store 分别从 0/512/…/3584 B 开始，最高边界 3840 B | FP32 | Init/ComputeSum | ComputeAvg | current output | partial sum 只在 offset 0；三种 dtype 均越过 1024 B 声明范围 |
| GM y | `outputOffset` | up to 1024 T/flush | T/NCDHW | CopyOut | caller | output | V→MTE3 event |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inputQue · depth=2 · 2*maxCount*s"]:2
  sum["storeAddUB · declared 1024 B / highest boundary 4096 or 3840 B"]:1
  out["outputUB · 1024*s"]:1
```

##### 字节账与复用顺序

```text
Device declared UB = 2*maxCount*s + 1024 + 1024*s
Device highest touched boundary = 2*maxCount*s + (4096 if FP32 else 3840) + 1024*s
Host conservative reserve: available=UB-1024*s-4096;  # FP32 恰好，FP16/BF16 保守多留 256 B
maxCount=floor_align(available/2,256)/4
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputQue slot | CopyIn | ComputeSum after DeQue | FreeTensor then next alloc | Queue protocol | depth 2 capacity |
| storeAddUB | `InitStoreOutBuffer<float>`：FP32 清 1024 个连续 float；FP16/BF16 清 8 个离散的 64-float 区间；partial 只写 offset 0 | next partial/ComputeAvg | next output init | local mem barrier in UpdateSum | Host 的 4096 B 保留只是 selector 算式，不能闭合 TPipe 只声明 1024 B 而 Device 越界触达的缺口；需修正 Device 声明或由 API 契约额外证明 |
| outputUB | ComputeAvg | CopyOut | InitOutputBuffer after flush | MTE3_V and V_MTE3 events | <=1024 outputs |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| output init | MTE3→V event | prior CopyOut | vector zero | outputUB | before reuse |
| output DMA | V→MTE3 event | ComputeAvg | CopyOut | outputUB | exact copyCount bytes |
| partial | local mem barrier | store/load partial | next Add | storeAddUB[0] | D/H/W/global split tail |
| average | MicroAPI ReduceSum/Add/Div/Cast | T loads | T output | true window | FP32 accumulation; `UpdateMask(dataCount)` excludes vector tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Direct Host/Tiling producer | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | 350 阈值、受控 general override、core split、UB reserve、`route=1` |
| Direct entry/router | 同上 | typed launcher + big dtype entries | route 1 直接映射到按值 `BigTiling` entry |
| Device body/helper | `adaptive_pool3d_official_big_base.h` + `adaptive_avg_pool3d_official_big.h` | `CopyIn/ComputeSum/ComputeSplitGlobal/Process` | 官方 big 机制本地化；Direct 补 global contiguous DMA；padded/split FP32 sum 和 batched output |
| 官方关系 | `adaptive_avg_pool3d_big_kernel_tiling.cpp` + `adaptive_avg_pool3d_apt.cpp` | `IsCapable/DoBlockTiling` + mode 1 | 只证明转换基线的 threshold/POD/body 关系，不是 Direct Host/router |

### Route R-003：固定因子 4/5/8/9

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | uniform exact `F`、outputSpatial>1、u32-safe | route 6/7/8/9 |
| Entry | 同上 | exact entry macro | `total<=grid*256` | group8；否则 scalar |

#### Device 数据流

##### 函数级调用链

```text
launcher -> exactF_<dtype>
  -> AdaptiveAvgPool3DExactFactorGroup8<T,F> OR AdaptiveAvgPool3DExactFactor<T,F>
```

##### 源码等价伪码

```text
decode output; start=(od,oh,ow)*F
sum FP32 over compile-time F×F×F window
group8: lane traverses iw=lane; shuffle-down 4,2,1; lane0 stores
scalar: one thread traverses whole window
y = T(sum / (F^3))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `ncBase+((F*od+d)*IH+F*oh+h)*IW+F*ow+w` | `F^3`/output | T/NCDHW | caller | lane/thread | caller | scalar loads |
| warp registers | lane-local | one FP32 sum/lane | FP32 | lane | shuffle/lane0 | one output | width 8 |
| GM y | `idx` | O | T | lane0/thread | caller | output | one store |

##### 分层内存布局

N/A：无 AIV UB；8-lane 路线使用 warp 寄存器交换。

##### 字节账与复用顺序

```text
UB_total = 0; dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| lane sum | lane loop | shuffle tree | 下一 output | warp shuffle sync | group width fixed 8 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| reduction | `WarpShflDownSync` 4/2/1 | 8 lanes | lane0 | current window | FP32 sum；只有 lane0 cast/store；最后 group 受 idx guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | factor selector、magic/shift |
| Kernel entry/router | 同上 | exact entry macros | group8/scalar split |
| Device body/helper | 同上 | fixed-factor bodies | 编译期窗口与 FP32 reduction |

### Route R-002：power-of-two /2 下采样

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | 三轴输入=输出×2、输出三轴 power-of-two、I/O u32-safe | `route=4` |
| Router | 同上 | dtype launcher | `route==4` | down2 dtype entry |

#### Device 数据流

##### 函数级调用链

```text
launcher -> down2_<dtype> -> asc_vf_call<AdaptiveAvgPool3DDownsample2Power2<T>>(1024)
```

##### 源码等价伪码

```text
for output idx in block连续域 by thread stride:
  decode nc,od,oh,ow by mask/shift
  load the 8 corners of input [2od:2od+2,2oh:2oh+2,2ow:2ow+2]
  y[idx] = T(fp32_sum * 0.125)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `inBase + {0,1,inW,inW+1,inPlane...}` | 8 values/output | T/NCDHW | caller | thread | caller | scalar loads |
| GM y | `idx` | O | T/NCDHW | thread | caller | output | one store |

##### 分层内存布局

N/A：线程寄存器替代 AIV UB。

##### 字节账与复用顺序

```text
UB_total = 0; dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 sum register | 8 loads | scale/store | 下一 idx | 程序顺序 | 独立输出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| average | N/A | 8 FP32 casts/adds | T cast/store | current output | 固定乘 1/8；block/thread tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | /2 guard与 shifts |
| Kernel entry/router | 同上 | down2 dtype entries | 1024 threads |
| Device body/helper | 同上 | `AdaptiveAvgPool3DDownsample2Power2<T>` | eight-value FP32 average |

### Route R-001：power-of-two ×2 上采样

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | 三轴输出=输入×2、输入三轴 power-of-two、I/O u32-safe | `route=3` |
| Router | 同上 | dtype launcher | `route==3` | up2 dtype entry |

#### Device 数据流

##### 函数级调用链

```text
launcher -> up2_<dtype> -> asc_vf_call<AdaptiveAvgPool3DUpsample2Power2<T>>(1024)
```

##### 源码等价伪码

```text
for input idx in block连续域 by thread stride:
  iw/ih/id/nc <- mask + shift 解码
  raw <- GM x[idx]
  packed <- raw 复制到两个相邻 lane
  向两个 H 行、两个 D 面各写一个 packed word，共覆盖 2×2×2 输出
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `idx` | I | T/NCDHW | caller | thread | caller | 单次 scalar load |
| GM y | `packedBase + {0,row,plane,plane+row}` | O | T/NCDHW | thread | caller | output tensor | 32/64-bit packed store |

##### 分层内存布局

N/A：无 AIV UB/dynamic UB；只使用线程寄存器与 GM。

##### 字节账与复用顺序

```text
UB_total = 0; dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | 当前 idx | packed stores | 下一 idx | 程序顺序 | block 域互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| replicate | N/A | scalar load | packed store | 当前输入对应 8 输出 | 位级复制，不做算术；`idx<blockEnd` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | 幂次 guard、shift、blockDim |
| Kernel entry/router | 同上 | up2 dtype entries | 1024-thread SIMT |
| Device body/helper | 同上 | `AdaptiveAvgPool3DUpsample2Power2<T>` | packed replication |

### Route R-004：parallel transpose + W/H/D staged reduction

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host producer | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` + `adaptive_avg_pool_3d_official_tiling.h` | config 调用 `ParaTiling::Evaluate` | `K_sel<350 && NC>=vfLen/2 && UB/利用/W尾 guard`；`K_sel` 是 helper bound | 成功写 route 2，否则继续 |
| Direct launcher/entry | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | typed launcher `route==2` + para entries | dtype + `useIndex64` | `ParaPool<T,i32/i64>` |
| 固定官方转换基线 | `adaptive_avg_pool3d_para_tiling.cpp` + `adaptive_avg_pool3d_apt.cpp` | `IsCapable` / template mode 0 | 解释 capability、factor search 和 body 映射；不是 Direct 运行 router | Direct 以自包含 Host class 和显式 entry 取代 |

#### Device 数据流

##### 函数级调用链

```text
para entry -> ParaPool::Init -> Process
  -> CalInputBlockPara -> CopyInput
  -> TransInput -> [prefetch next CopyInput]
  -> AvgPoolOnW -> AvgPoolOnH -> AvgPoolOnD
  -> CalAvg -> TransOut -> CopyOut
```

##### 源码等价伪码

```text
for assigned outer block:
  derive NC/D/H/W tile and adaptive input extent
  DMA strided NCDHW tile to inputQue; transpose NC vector lanes contiguous
  for each W output: sum its W window to FP32
  for each H output: sum previous H slices to FP32
  for each D output: sum previous D slices to FP32
  divide by kD*kH*kW; transpose back; cast; strided DMA to GM
  except last block, prefetch next input before current staged reductions
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inputQue` depth 1 | tile GM offset | `M*s` bytes | T, padded W | CopyInput | TransInput | TPipe/block | 32 B W align |
| `avgQue` depth 1 | 0 | `M*4` bytes | T then FP32 | transpose/H/TransOut | W/D/CopyOut | TPipe/block | type-view reuse |
| `avgTransQue` depth 1 | 0 | `M*4` bytes | FP32/T cast | W/D/CopyOut | H/CalAvg | TPipe/block | staged ping-pong |
| index TBufs | 0 | `A(maxDimOut*4)+A(do*4)+A(ho*4)+A(wo*4)` | i32 | CalKernelSize | W/H/D/CalAvg | whole kernel | 32 B align |
| GM y | tile offset | `ncNum*doNum*hoNum*woNum` | T/NCDHW | CopyOut | caller | output | strided W rows |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inputQ["inputQue d=1 · M*s"]:1
  avgQ["avgQue d=1 · M*4"]:1
  avgTQ["avgTransQue d=1 · M*4"]:1
  start["startIdx · A(maxDimOut*4)"]:1
  ker["D/H/W size · ΣA(factor*4)"]:1
```

##### 字节账与复用顺序

```text
UB_total = M*s + 2*M*4 + A(maxDimOut*4,32)
         + A(doFactor*4,32)+A(hoFactor*4,32)+A(woFactor*4,32)
UB_total <= platform UB (Host capability)
dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| avgQue | TransInput/H/TransOut | W/D/CopyOut | next stage | Queue DeQue/EnQue/FreeTensor | stage order fixed |
| avgTransQue | W/D/cast | H/CalAvg/CopyOut | next stage | Queue protocol | no early overwrite |
| inputQue | CopyInput prefetch | TransInput | next prefetch | depth 1 queue | current item dequeued first |

Queue token 的精确生命期是：`TransInput` 把 T 转置结果放入 `avgQue`；W 阶段消费它并把 FP32 W-sum 放入 `avgTransQue`；H 阶段消费 W-sum，同时把被消费的 token 重新 EnQue 到 `avgTransQue` 作为 D 阶段的输出区，H-sum 放入 `avgQue`；D 阶段消费两者，在前一 W-sum 空间中生成 D-sum；`CalAvg` 就地除法后再 EnQue；`TransOut` 复用保留的 H-sum 空间写反转置结果；`CopyOut` 在最后一次 DeQue 后释放，低精度时临时借 `avgTransQue` 做 cast 目标。因此应保留源码中看似多余的 re-EnQue，不能用普通 FreeTensor 替换。

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/compute | Queue | CopyInput | TransInput DeQue | input tile | W padding不进入有效 kernel |
| index read | `V_S` flag/wait | CalKernelSize vector store | scalar kernelSize reads | index TBuf | i64 arithmetic route narrows stored local offsets/kernel size to i32，须满足下表独立窄化前提 |
| reduction | MicroAPI Add/Div | W→H→D sums | CalAvg | FP32 buffers | FP16/BF16 load升 FP32；输出 FP16 CAST_NONE、BF16 CAST_RINT |
| tails | outer factors | Host | Process | core/tile ranges | nc/do/ho/wo independent tails |

`ID_T=i64` 只保护 `CalKernelSize` 中的主索引算术，不能自动证明所有 Device ABI 窄化安全。下表把显式 cast、形参转换和 API 结构字段赋值分开；“UB 推导闭合”都以 Host 的 u64/i64 容量算术本身不溢出为前提，表示由 `CalMaxUbSplitSize`/`CalOccupySize` 与最终 factor search 对 950 UB 的上界可推导，不表示 Host 有独立的位宽检查。

| 窄化点 | 源值 → 目标位宽 | 严格安全前提 | 现有证明状态 |
|---|---|---|---|
| `CalInputBlockPara` 的 D 绝对 end | `Ceil((doIdx*doFactor+doNum)*dIn,dOut)` → `int32_t kerDEndIdx`，再减 i64 `kerDStartIdx` | 乘加分子先在 i64 内不溢出；绝对 ceil end 在 `[0,INT32_MAX]`；窄化后 `kerDEndIdx>=kerDStartIdx` | **缺失 Host 前提**：`useIndex64` 只选 `CalKernelSize` 模板，没有检查这个绝对 i32 end |
| 同上 H 绝对 end | `Ceil((hoIdx*hoFactor+hoNum)*hIn,hOut)` → `int32_t kerHEndIdx`，再减 i64 `kerHStartIdx` | 同样要求 i64 乘加无溢出、ceil end 不超过 `INT32_MAX`且 end 不小于 i64 start | **缺失 Host 前提**：同 D |
| 同上 W 绝对 end | `Ceil((woIdx*woFactor+woNum)*wIn,wOut)` → `int32_t kerWEndIdx`，再减 i64 `kerWStartIdx` | 同样要求 i64 乘加无溢出、ceil end 不超过 `INT32_MAX`且 end 不小于 i64 start | **缺失 Host 前提**：同 D |
| `BlockSplitParam` 到 helper 形参 | i64 `ncNum/diDataLen/hiDataLen/wiDataLen/doNum/hoNum/woNum` → `CopyInput`/`TransInput`/`AvgPoolOnW/H/D` 的 u32 形参 | 各值先须非负且 `<=UINT32_MAX` | `ncNum<=ncFactor=vfLen`；其余是 maxD/H/W 参与量，**UB 推导闭合**，但 Host 无单独 cast guard |
| `CopyInput` DMA 行数/行长 | u32 `hiDataLen` → `DataCopyExtParams.blockCount:u16`；`wiDataLen*sizeof(T)` → `blockLen:u32` | `hiDataLen<=UINT16_MAX`；`wiDataLen<=floor(UINT32_MAX/sizeof(T))` | 两者都是 maxInputSize 的参与量，**UB 推导闭合**；源码只对目标类型做 cast |
| `CopyInput` GM 行 stride | `(wIn-wiDataLen)*sizeof(T)` → 显式 `uint32_t`，再传给 950 可承载 i64 stride 的 `DataCopyExtParams` | 每个 tile 都须 `0<=wIn-wiDataLen<=floor(UINT32_MAX/sizeof(T))`；充分但更强的 Host guard 是 `wIn<=floor(UINT32_MAX/sizeof(T))` | **缺失 Host 前提**：UB 只约束 tile 的 `wiDataLen`，不约束全局 `wIn` 及字节 stride |
| `TransposeB16/B32` 入参 | `TransInput` 的 u32 `diDataLen*hiDataLen*wiDataAlign` 或 `TransOut` 的 u64 `rowNumAlign` → `TransposeB*` u32 `rowNum/colNum` | 空间乘积和 `rowNumAlign` 须 `<=UINT32_MAX`，乘加/对齐不溢出 | 这些乘积都不超过 maxDHW，**UB 推导闭合** |
| `TransDataTo5HDParams.repeatTimes` | 分支中 `rowNum/16` 或 `colNum/transPoseAlign` → u8 | 对应商必须在 `[1,UINT8_MAX]` | `TransInput` 的 `rowNum=vfLen`，`TransOut` 的 `colNum=vfLen`；另一维由 maxDHW 控制，**shape 常量 + UB 推导闭合**，无独立 u8 guard |
| `TransDataTo5HDParams` repeat stride | `rowNum` 或常量 `1/16`（B32 对齐分支还有常量 2）→ `dstRepStride/srcRepStride:u16` | 非常量的 `rowNum<=UINT16_MAX` | `TransInput` 的 rowNum 是 `vfLen`；`TransOut` 的 rowNum 受 maxDHW 约束，**shape 常量 + UB 推导闭合** |
| `CalKernelSize` 批长 | `BlockSplitParam` i64 tile 输出数 → `AvgPoolOn*` u32 → `CalKernelSize(int32_t kernelNum)` → `static_cast<uint16_t>(kernelNum)` | 先要 `1<=kernelNum<=INT32_MAX`以保真传参，再要 `kernelNum<=UINT16_MAX`以保真 `loopSize=ceil(kernelNum/vfLen)` | tile 输出数是 do/ho/woFactor 或 tail，**UB 推导闭合**，无独立 i32/u16 guard |
| i64 `CalKernelSize` 的 UB 写回 | i64 local start offset/window size → i32 index TBuf；window size 再→ `CustomSelect(uint16_t kernelSize)` | local offset 在 `[0,INT32_MAX]`；每轴窗长在 `[1,UINT16_MAX]` | local offset 不超过 UB-bounded input extent；窗长又受 `K_sel<350`，**UB + shape guard 推导闭合** |
| `CustomSelect` 控制量 | `diDataLen*hiDataLen`、`woNum*diDataLen`、`hoNum*woNumAlign` → `repeatTimes:u16`；`wiDataAlign/hiDataLen/diDataLen` → `srcStride:u16` | 三个 repeat 表达式和对应 stride 均须 `<=UINT16_MAX`，且表达式计算不溢出 | 都是 maxDHW 的因子或子乘积，**UB 推导闭合**；无独立 u16 guard |
| `CalAvg` 三重循环 | **i64** `doNum/hoNum/woNum` → 各自 `static_cast<uint16_t>(...)`；`CeilAlign(woNum,...)` → u32 `woNumAlign` | `doNum,hoNum,woNum` 分别在 `[1,UINT16_MAX]`，对齐后 W 数在 `[1,UINT32_MAX]` | tile 数是 maxDHW 参与量，**UB 推导闭合**；源类型是 i64，不是 u32 |
| `CopyOut` DMA/LoopMode 控制字段 | i64 `hoNum` → `blockCount:u16`；`woNum*sizeof(T)` → `blockLen:u32`；i64 `doNum/ncNum` → `loop1Size/loop2Size:u32`；`CeilAlign(i64 woNum,...)` → u32 `woNumAlign` | `hoNum<=UINT16_MAX`；行字节数、doNum、ncNum 及对齐 W 数分别不超过目标位宽 | do/ho/wo 由 maxDHW 约束，`ncNum<=vfLen`，**UB + shape 常量推导闭合**；950 `srcStride/dstStride` 字段本身为 i64，不在此列重复计入窄化 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Direct Host/Tiling producer | `adaptive_avg_pool_3d_official_tiling.h` | `ParaTiling::Evaluate/CalOccupySize` | capability、factor search、UB byte formula、POD |
| Direct entry/router | `adaptive_avg_pool_3d_kernel.cpp` | typed launcher + i32/i64 para entries | route 2 按 index width 映射 `ParaPool` |
| Device body/helper | `adaptive_avg_pool3d_official_para.h` | `Process/AvgPoolOnW/H/D/CalAvg` | 官方 parallel body 本地化；transpose、三级 reduction、cast/tail |
| 官方关系 | `adaptive_avg_pool3d_para_tiling.cpp` + `adaptive_avg_pool3d_apt.cpp` | `IsCapable/CalOccupySize` + mode 0 | 证明转换基线，不参与 Direct 运行分发 |

### Route R-007：u64 official-style SIMT fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct Host | `op_kernel/adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | route remains 0 and `maxDivUse>=INT32_MAX` | `useIndex64=1`，blockDim 取 `min(O,cores)` |
| Direct launcher/entry | 同上 | typed launcher 通用 `else` + `useIndex64!=0` | dtype + u64 + NCDHW | 显式 `*_u64` entry |
| 固定官方转换基线 | `adaptive_avg_pool3d_simt_tiling.cpp` + `adaptive_avg_pool3d_apt.cpp` | `GetTilingKey` / template mode 2 | 解释 index width、FORMAT=1 和 NC SIMT body 关系；不是 Direct 运行 router | Direct 以按值 POD 和显式 entry 取代 |

#### Device 数据流

##### 函数级调用链

```text
u64 entry -> AdaptiveAvgPool3DSimt<T,u64,1>::Init -> Process
  -> GetUintDivMagicAndShift -> UB barrier
  -> asc_vf_call<AdaptiveAvgPool3DNcSimtCompute<T,u64>>(1024)
```

##### 源码等价伪码

```text
copy 8 i64 tiling fields to UB; derive magic/shift for OD/OH/OW
barrier
for idx=block*1024+thread; idx<O; idx+=grid*1024:
  u64 decode nc/od/oh/ow; compute adaptive starts/ends
  FP32 scalar sum window; y[idx]=T(sum/window)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tilingDataBuf | 0 | `8*8=64 B` | i64 | AIV scalar setup | SIMT | entry | fixed |
| uintDivBuf | 0 | `8*8=64 B` | u64 | magic helper | SIMT UintDiv | entry | fixed |
| GM x/y | adaptive window / idx | input/O | T/NCDHW | caller/thread | thread/caller | tensors | u64 address math |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  tiling["tilingDataBuf · 64 B"]:1
  div["uintDivBuf · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 64 + 64 = 128 B
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| two TBufs | Process setup | SIMT body | kernel end | UB DataSyncBarrier | no overwrite after launch |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| setup→SIMT | `DataSyncBarrier<UB>` | scalar UB writes | asc_vf_call | POD/magic buffers | u64 index；FP32 sum；grid-stride tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Direct Host/Tiling producer | `adaptive_avg_pool_3d_kernel.cpp` | `calc_adaptive_avg_pool_3d_config` | 八个 i64 维度、blockDim、`useIndex64=1`、route 0 |
| Direct entry/router | 同上 | typed launcher + `*_u64` entries | route 0/u64 映射 FORMAT=1 NC SIMT |
| Device body/helper | `adaptive_avg_pool3d_official_simt.h` | `Process/NcSimtCompute` | 官方 SIMT body 本地化；128 B UB setup、u64 windows、FP32 average |
| 官方关系 | `adaptive_avg_pool3d_simt_tiling.cpp` + `adaptive_avg_pool3d_apt.cpp` | `DoOpTiling/GetTilingKey` + mode 2 | 证明 block/index/format/body 转换关系，不参与 Direct 运行分发 |

## 算子特定优化

### Trick T-001：SIMT 用 magic division 解码与构造自适应窗口

#### 触发与源码依据

- 适用 Route：R-005/R-007；R-003 仅复用输出索引解码。
- 触发条件：Host/entry 已为每个输出维度生成 magic/shift；R-007 依据最大乘积选择 u64。
- 固定来源：[`adaptive_avg_pool3d_simt.h:66-74`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_simt.h#L66-L74)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
        DIV_T divW = Simt::UintDiv<DIV_T>(idxOut, uintDivData[IDX0], uintDivData[IDX1]);
        DIV_T divH = Simt::UintDiv<DIV_T>(divW, uintDivData[IDX2], uintDivData[IDX3]);
        DIV_T divD = Simt::UintDiv<DIV_T>(divH, uintDivData[IDX4], uintDivData[IDX5]);
        DIV_T curWOutIdx = idxOut - divW * tilingData->wOutDim;
        DIV_T curHOutIdx = divW - divH * tilingData->hOutDim;
        DIV_T curDOutIdx = divH - divD * tilingData->dOutDim;

        DIV_T startInD = CalStartIdx<DIV_T>(curDOutIdx, uintDivData[IDX4], uintDivData[IDX5], tilingData->dInDim);
        DIV_T endInD = CalEndIdx<DIV_T>(curDOutIdx, uintDivData[IDX4], uintDivData[IDX5], tilingData->dInDim, tilingData->dOutDim);
```

#### 机制与设计原因

同一维度的 magic/shift 同时服务线性索引拆解和 `floor/ceil` 窗口边界，减少热循环中的普通整数除法。Direct 通用 u32 路线把 magic/shift 放进按值 POD；官方式 fallback 则在 entry 内写 128 B UB 元数据并 barrier 后调用 SIMT。

#### 迁移边界

- 适用：正的固定 divisor、Host 可预计算 magic/shift、索引乘积有明确 u32/u64 guard。
- 前提：`CalEndIdx` 的 `+outLen-1` 必须保留，magic 与 divisor 一一对应。
- 失效：乘积达到 u32 边界时必须切 u64 或其它安全 Route，不能只扩大最终指针类型。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 u32 magic 用于已溢出的 `outIdx*inLen` 会在除法前产生错误；固定官方 tiling 以最大参与乘积选择 index width。

### Trick T-002：big Route 按 D→H→W 层级切窗并跨 chunk 累加

#### 触发与源码依据

- 适用 Route：R-006。
- 触发条件：真实对齐窗口超过 `maxCount`；若 `AlignHW<=maxCount` 切 D，否则若 `AlignW<=maxCount` 切 H，否则切 W。
- 固定来源：[`adaptive_avg_pool3d_big_kernel.h:296-302`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_big_kernel.h#L296-L302)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    if (this->AlignHW <= this->tilingData_.maxCount) {
        ComputeSplitD(curIdx);
    } else if (this->AlignW <= this->tilingData_.maxCount) {
        ComputeSplitH(curIdx);
    } else {
        ComputeSplitW(curIdx);
    }
```

#### 机制与设计原因

优先沿最外层 D 切分，可让每次 DMA 保留完整 H×W 行块；只有 UB 连一层 HW 都容不下才逐步降到 H/W。`ComputeSum<SPLIT_*>` 把每 chunk 的 FP32 ReduceSum 加到 `storeAddUB[0]`，最后只做一次真实窗口体积除法，避免 materialize 每个 partial output。Direct 交付另外对全局 `[1,1,1]` 窗口使用连续 DMA；该增强是交付事实，不在上述官方片段中。

#### 迁移边界

- 适用：NCDHW 大归约窗、单输出窗口可被至少按 W 分片装入 UB。
- 前提：`maxCount>0`、32 B DMA padding、split partial 严格串行并以 FP32 保存。
- 失效：selector 窗口上界小于阈值时应走 parallel/general；若改变 layout，D/H/W 优先级与 stride 必须重算。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把 padding 后的 `AlignDHW` 当作除数；源码始终用真实 `curkDHW_`。固定仓未提供其它可验证反例。

### Trick T-003：parallel Route 先转置 NC lane，再做 W/H/D 三段 FP32 归约

#### 触发与源码依据

- 适用 Route：R-004。
- 触发条件：parallel capability 全部成立，`ncFactor=256/dtypeBytes`，tile 满足 UB 与 W 尾 guard。
- 固定来源：[`adaptive_avg_pool3d_parall_pool.h:667-674`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_parall_pool.h#L667-L674)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
        /* outshape: [woNum, hiDataLen, diDataLen, vl] */
        AvgPoolOnW(diDataLen, hiDataLen, wiDataLen, woIdx, woNum);

        /* outshape: [hoNum, woNumAlign, diDataLen, vl] */
        AvgPoolOnH(diDataLen, hiDataLen, woNum, hoIdx, hoNum);

        /* outshape: [doNum, hoNum, woNumAlign, vl] */
        AvgPoolOnD(diDataLen, hoNum, woNum, doIdx, doNum);
```

#### 机制与设计原因

转置把 `ncFactor` 个 NC 值放成 vector lane，`CustomSelect` 便可对同一空间窗口同时处理整条 NC 向量。W、H、D 逐轴先求和，计算量从重复遍历三维窗口改为复用前一级 partial；只有末尾 `CalAvg` 做一次除法。三 Queue 交替承载 T/FP32 阶段，避免额外 GM workspace。

#### 迁移边界

- 适用：NC 至少半个 vector、selector 窗口上界 `K_sel<350`、NCDHW。
- 前提：transpose 的 16 行和 32 B 对齐、UB 字节账、FP32 中间布局、V→S 索引可见性。
- 失效：NC 太短、UB 不足、W 尾利用不足或 `K_sel>=350` 时进入 general/big/SIMT fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在每一轴归约后立刻除以该轴窗口长度再降回低精度；固定实现保留 FP32 partial，到三轴求和完成后一次除以 `kD*kH*kW`。

## 模板与实现对比

| 对比维度 | P0 R-005 | P1 R-006 | P2 R-003 | P3 R-002 | P4 R-001 | P5 R-004 | P6 R-007 | 设计选择依据 |
|---|---|---|---|---|---|---|---|---|
| 命中域 | remaining u32 | `K_sel>=350` | exact F | exact /2 | exact ×2 | small `K_sel` + NC/UB | remaining u64 | 重要性视图不改变 Host 短路 |
| 模板与 dtype | SIMT T | AIV T | SIMT T,F | SIMT T | SIMT T | AIV T,ID_T | SIMT T,u64 | 三 dtype 均 FP32 average |
| 循环与分核 | output linear | outputs/core | output linear | output linear | input linear | outer tiles/core | grid stride | 按工作粒度均分 |
| Queue/TBuf/寄存器 | registers+shuffle | depth-2 Queue+2 TBuf | registers+shuffle | registers | registers | 3 Queue+4 TBuf | 2 TBuf+registers | 与复用需求匹配 |
| 内存总占用与复用 | 0 UB | `2*maxCount*s+1024*s+1024` 为 Device 声明账，但 init 最高触达 4096/3840 B | 0 UB | 0 UB | 0 UB | `M*s+8M+indices` | 128 B | Host 按 4096 B 预留；Device 声明缺口未闭合 |
| 搬运/计算 API | scalar+shuffle | DMA+ReduceSum | scalar+shuffle | 8 loads+average | packed GM | DMA+transpose+MicroAPI | scalar GM+UintDiv | shape 专用化 |
| 同步、精度与 tail | warp/程序序 | Queue/events/barrier | warp | 程序序 | 程序序 | Queue/V_S | UB barrier | 全部 FP32 累加 |
| fallback/失败边界 | body 内标量 fallback | 无 | body 内标量 fallback | Host 后续 Route | Host 后续 Route | Host 后续 Route | 无 | Host 去向以选择表为准 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本 Direct 交付未覆盖 | 本 Direct 交付未覆盖 | 交付目标为 950/dav-3510、CANN 9.1.0 |
| Host/Launcher | N/A | N/A | 查询 AIV core/UB，按值 POD，当前 stream 单 Kernel |
| Kernel/资源 | N/A | N/A | arch35 相关 SIMT/big/parallel + Direct supplemental SIMT；无 Cube |

固定官方源码在 base tiling 中显式检查 `DAV_3510`，但本卡的 950 Direct 支持结论来自交付目标与完整 Direct 实现闭包；不把目录名本身当作公开 API 支持证明。

## 限制与失败边界

- 仅连续、非空五维 NCDHW；三种浮点 dtype；`output_size` 必须恰为三个正数。
- Host 的 `I/O/shape products` 用 u64 连乘但没有显式乘法溢出检查；超过 u64 表示范围时行为未闭合。Fast/General POD 只在 selector 证明 u32-safe 后窄化。
- big Route 假设 UB 大到使 `ubSize-1024*s-4096` 与 `maxCount` 为正；Direct selector 没有为极小 UB 提供错误返回或二次 fallback。
- big Route 的 Host 统一按 4096 B 为 `storeAddUB` 保留空间：对 FP32，`InitStoreOutBuffer<float>` 的 16 个连续 64-float store 触达 `[0,4096)` B；对 FP16/BF16，`maxVfCount=128` 使 8 个 64-float store 以 512 B 为起点步长离散分布，最后一段为 `[3584,3840)` B，所以 4096 B 只是保守预留。Device `InitBuffer(storeAddUB_, STORE_ADD_BUFFER)` 对三种 dtype 却都只声明 `STORE_ADD_BUFFER=1024` B；Host 保留不能补齐这个 Device 声明/触达缺口，未修正或未有额外 API 证明前，不应把该存储边界记为严格安全。
- parallel 的 i64 index 路线仍有多级窄化。其中 tile extent、transpose repeat/stride、`CalKernelSize/CalAvg/CustomSelect` 与 `CopyOut` 控制字段可在 Host 容量算术不溢出的前提下，由 950 UB 账和 `K_sel<350`/`ncFactor=vfLen` 推导到目标位宽；但这些不是独立 Host cast guard。三轴绝对 ceil end → i32，以及 `CopyInput` 中 `(wIn-wiDataLen)*sizeof(T)` → u32 byte stride，则不由 UB 上界闭合，当前 selector 缺少对应 Host 前提。不能仅依赖 `useIndex64` 就宣称 R-004 窄化安全。
- Direct launcher 为 `void`，回调固定返回 0；Device launch 异常的同步反馈不在固定交付源码中可见。
- 官方链接证明转换相关实现，不证明 supplemental ratio/factor/general 路线是官方接口。未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| POD/模板 | `pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_pool3d_tiling_struct.h` | 三个 tiling class 与 template selection | 官方相关 ABI 与 AIV-only 模板族 |
| Kernel router | `pooling/adaptive_avg_pool3d/op_kernel/adaptive_avg_pool3d_apt.cpp` | `adaptive_avg_pool3d` | template→SIMT/big/parallel body |
| SIMT body | `pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_simt.h` | `AdaptiveAvgPool3DSimt` | magic division、UB barrier、u32/u64 SIMT |
| big base | `pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_pool3d_big_kernel.h` | `CopyIn/Init/CalcWindowSize` | Queue、DMA、window address |
| big average | `pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_big_kernel.h` | `ComputeSum/Process` | split reduction、FP32 average、tail |
| parallel body | `pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_avg_pool3d_parall_pool.h` | `Process` | transpose、W/H/D reduction、cast/copy |
| Host base header | `pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.h` | `BaseInput` | shape/platform/UB 参数结构 |
| Host base | `pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.cpp` | `GetShapeAttrsInfo/GetPlatformInfo` | DAV_3510、dtype/shape、AIV/UB |
| SIMT tiling | `pooling/adaptive_pool3d_common/op_host/arch35/adaptive_avg_pool3d_simt_tiling.cpp` | `DoOpTiling/GetTilingKey` | blockDim 与 index width |
| big tiling | `pooling/adaptive_pool3d_common/op_host/arch35/adaptive_avg_pool3d_big_kernel_tiling.cpp` | `IsCapable/DoBlockTiling` | 350 threshold、UB/core split |
| parallel tiling | `pooling/adaptive_pool3d_common/op_host/arch35/adaptive_avg_pool3d_para_tiling.cpp` | `IsCapable/CalOccupySize` | capability、factor search、UB formula |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三个 official-related POD 字段顺序与 Direct 按值副本一致 | 字段均 i64；Fast/General 是 Direct-only u32 ABI | tiling struct、Direct launch config |
| 调用顺序 | selector 后只启动一个 entry | Route 优先级不能重排；parallel 内预取/三级归约顺序固定 | config、`Process` |
| workspace/内存 | 用户 workspace 为 0；中间量在 UB/寄存器 | Queue depth、容量、32 B/16-row align、owner/lifetime | `InitBuffer/CalOccupySize` |
| 同步 | Queue、V_S、MTE events、UB barrier、warp shuffle | producer→consumer 与 buffer复用点不能省略 | 各 Route body |
| 精度与尾块 | FP32 accumulate/divide，再 cast T | true window size、mask、former/tail、lane0 store | `ComputeAvg/CalAvg/*SimtCompute` |
| 失败边界 | Host validation；selector 最终 u64 fallback | u32 narrowing、UB subtraction、void launch 边界 | plugin/config/launcher |
