---
type: Operator
title: "GroupedMatmul 950 Direct Invoke 算子知识"
description: "说明 cann_bench::grouped_matmul、grouped_matmul_npu 与 launch_grouped_matmul 在 950 上的接口语义、Host/POD、Route 选择及 Cube/AIV 协作。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/gmm/grouped_matmul/op_api/grouped_matmul.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_basic_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/gmm/grouped_matmul/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T19:34:28Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GroupedMatmul 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| Direct 入口 | `cann_bench::grouped_matmul`（Torch schema）→ `grouped_matmul_npu`（PrivateUse1）→ `launch_grouped_matmul`（C ABI launcher） |
| 数学语义 | 对累积边界 `r_g=group_list[g]`，计算 `Y[r_{g-1}:r_g,:]=X[r_{g-1}:r_g,:]×W[g]+bias[g]`，其中 `r_-1=0`；bias 可缺省。 |
| 合法域 | 连续 ND；`X:[M,K]`；`W:[E,K,N]` 或转置存储 `[E,N,K]`；`1<=E<=128`；FP16/BF16/FP32；允许空 group。 |
| P0 R-001 | 公共 cgmct no-quant Cube：BF16、普通 FP16，以及其它专用 Route 的基础 producer。 |
| P1 R-002 | 非小型 FP32：CATLASS grouped Slice-M，HF32 关闭；有 bias 时追加 AIV FP32 bias。 |
| P2 R-003 | FP16 数值风险域：公共 Cube → AIV 扫描压缩 → 最多 4096 个索引 dot 的 AIV 重算。 |
| P3 R-004 | 小型单专家非转置 FP32：保留公共 Cube，短路 CATLASS。 |
| P4 R-005 | 小 N 转置 FP16/FP16-bias ABI：公共 Cube body，但将 launch blocks 与 `usedCoreNum` 收缩为 E。 |
| 执行分工 | AIC/Cube 始终产生完整矩阵乘输出；AIV 只承担 FP32 bias，或 FP16 候选扫描与有界重算。 |
| GM workspace | 固定 descriptor/group-list 区 1280 B；R-003 再加 32776 B 数值 scratch。数学中间结果直接使用 Y，不另建中间 GM tensor。 |

本卡中的 direct schema、CATLASS Route、AIV 数值节点和固定 workspace 布局来自转换实现。Frontmatter 的 HTTP 源是同算子 950 no-quant 官方映射，不能据此把 direct-only 分支宣称为公开 API 的固有行为。

## 接口、语义与边界

### 分组矩阵乘

令：

```text
r[-1] = 0
r[g]  = group_list[g]
M_g   = r[g] - r[g-1]
X_g   = X[r[g-1]:r[g], :]
```

每个专家的地址与逻辑矩阵为：

```text
X_g offset = r[g-1] * K
W_g offset = g * K * N
Y_g offset = r[g-1] * N
bias_g offset = g * N                  # only when hasBias

transpose_weight = false: W_g storage [K,N], RowMajor B
transpose_weight = true : W_g storage [N,K], ColumnMajor logical B
```

`transpose_weight` 只改变 weight descriptor 的后两维、B layout 和重算时的索引公式；每个专家的连续 storage stride 始终是 `K*N`。`split_item` 只改变返回容器：0/1 返回 E 个共享底层 storage 的行 views，2/3 返回一个 `[M,N]` Tensor，不改变 device 数学。

### 输入输出契约

| 参数 | shape/layout | dtype | Host 契约 |
|---|---|---|---|
| `x` | `[M,K]`，连续 Strided/ND | FP16/BF16/FP32 | 已定义；`M,K>0`。 |
| `weight` | `[E,K,N]` 或 `[E,N,K]`，连续 Strided/ND | 与 X 相同 | `E,N>0`；weight 的 K 必须等于 X.K。 |
| `bias` | 可选 `[E,N]`，连续 Strided/ND | 同 X，或 FP16/BF16 X 配 FP32 bias | 同设备；FP32 X 只能配 FP32 bias。 |
| `group_list` | Host `int[]`，长度 E | int64 语义 | 累积、非递减、每项不超过 M、末项等于 M。 |
| `split_item` | 标量 | int64 | 只允许 0、1、2、3。 |
| `transpose_weight` | 标量 | bool | 决定 weight shape 解释与 B layout。 |
| 输出 | 底层 `[M,N]` 连续 Tensor 或 E 个行 views | 与 X 相同 | views、workspace 和输出共用一块 storage。 |

所有 tensor 必须同设备，代码不隐式执行 `contiguous()`、transpose、dtype cast 或 CPU 数学。BF16 无 bias 时，Host 将 dormant bias template 归一到 BF16-X/FP32-bias entry；`hasBias=0` 保证 Device 不读 bias。

### group 边界与空专家

Host 从 0 开始逐项验证 `previous<=end<=M` 并计算 `maxGroupM=max(end-previous)`。相等的相邻边界是合法空 group：

- 公共 cgmct body 从累积边界恢复当前 `M_g`，空 group 不产生 tile；X/Y 的行前缀不前进，weight/bias 的专家索引仍前进。
- CATLASS Slice-M 的 `currentM` 是相邻边界差。空 group 的 `coreLoops=0`，但 `gmGroupOffsetB` 仍增加 `K*N`，所以后续专家仍绑定正确 weight。
- AIV bias/refine 用 `while (expert+1<E && row>=group_list[expert]) ++expert` 找专家；重复边界会连续跳过空专家。

最后边界必须是 M，因此每个有效输出行恰好属于一个非空专家。不得把 `group_list` 当成每组长度，也不得在重复边界处读空专家的 X/Y。

## Host、storage 与异步生命周期

### 校验结果与整数域

`CheckInputs` 生成：

```text
CheckedInputs {
  e, m, k, n, maxGroupM,
  xDtype, biasDtype,
  hasBias, transposeWeight, splitItem,
  ends[E]
}
```

`M/K/N<=INT32_MAX` 是公共 `TCubeTiling` 和 Device shape 的显式边界；`E<=128` 是 metadata/group loop 边界。output 大小使用 uint64 计算：

```text
elementBytes = (xDtype == FP32 ? 4 : 2)
outputElements = uint64(M) * N
outputBytes = outputElements * elementBytes
workspaceOffset = align_up(outputBytes, 32)
storageElements = ceil((workspaceOffset + workspaceCapacity) / elementBytes)
```

这些显式 shape 上界足以使 `M*N*elementBytes` 留在 uint64 域，且 `outputElements` 留在 int64 allocator 参数域。另一方面，tiling 的 `E*K*N*dtypeBytes` L2 启发式和 CATLASS 的专家 weight 累积偏移没有单独的 checked-multiply；实现依赖实际 tensor 可表示/可分配。这是边界，不应把 L2 hint 的未检查乘法当成接口保证。

### 固定 metadata 与可选 scratch

输出后 32 B 对齐的 workspace 布局如下：

```text
workspace +    0 : X ListTensorDesc slot       [0,64)
workspace +   64 : weight ListTensorDesc slot  [64,128)
workspace +  128 : bias ListTensorDesc slot    [128,192)
workspace +  192 : Y ListTensorDesc slot       [192,256)
workspace +  256 : int64 group_list[128]        [256,1280)

# only R-003
workspace + 1280 : uint64 candidateCount        [1280,1288)
workspace + 1288 : uint64 candidateIndices[4096][1288,34056)
```

因此：

```text
descriptorBytes = 256 + 128*8 = 1280 B
numericalScratchBytes = (1+4096)*8 = 32776 B
workspaceCapacity = 1280 B                         # R-001/R-002/R-004/R-005
workspaceCapacity = 1280 + 32776 = 34056 B        # R-003
```

四个 descriptor 的序列化 footprint 分别是 40、48、40、40 B，slot 间空洞由 Host 清零。无 bias 时 bias descriptor 的 data pointer 填 X 地址以保持合法 descriptor；raw `bias` 参数仍可为 null，但所有 no-bias body 都由 `hasBias` 或编译期 `HasBias=false` 阻止读取。

Host 创建长度为 `workspaceCapacity/8` 的零初始化 `vector<uint64_t>`，填 descriptor 和前 E 个 group end 后，把整个 capacity 异步 H2D。R-003 因而在每次调用前同时把 `candidateCount` 和全部 indices 清零，不需要额外清零 kernel。

### 三类 owner 与同 stream 顺序边界

生命周期必须分开审计，不能用同 stream 顺序替代 owner 证明：

| 对象 | 源码内可证明事实 | 完成期边界 |
|---|---|---|
| Y、descriptor/group-list、R-003 scratch | 它们位于同一个 `[Y \| aligned workspace]` storage；返回的 Y 或行 views 持有该 storage | 返回 Tensor 对 Device storage 的 owner 由本文件闭合 |
| `descriptorKeep` Host vector | callable 按值捕获 vector，lambda 执行时它是 `aclrtMemcpyAsync` 的 Host source | 源码没有 event、callback 或同步等待证明 vector 覆盖异步 H2D 消费期；依赖 `RunOpApi`/runtime 对 callable 与 H2D source 的契约 |
| X、weight、optional bias | callable 按值捕获输入 Tensor，descriptor/raw 参数保存其 device pointer | 源码没有 `recordStream` 或 completion event 证明 capture 覆盖 Device 完成期；依赖 `RunOpApi` 与 NPU allocator/stream 的延迟回收契约 |

源码注释只说 `RunOpApi` “may retain/copy” callable，因此不能把 lambda capture 本身写成完成期保证。返回 storage 的 owner 是明确事实；Host copy source 和输入 Device storage 的最终安全期是框架/runtime 边界。

```text
grouped_matmul_npu
  -> validate / allocate [Y | aligned workspace]
  -> RunOpApi callable on current stream
       -> aclrtMemcpyAsync(H2D, full workspaceCapacity)
       -> launch_grouped_matmul
            -> unconditionally QueryPlatform + CalcOfficialTiling
            -> exactly one terminal Route graph
```

H2D 返回非成功时立即 `TORCH_CHECK`。Device kernel launch 语句不读取独立 launch status；同 stream 只证明 H2D→AIC、AIC→AIV、scan→refine 的提交与 Device 可见性顺序，不证明 Host vector 或输入 Tensor 的 owner 生命周期。Route 内无跨 stream event，也没有 host-side output 同步。

## Route 选择与不可变项

### 先计算、后分派

launcher 在 route 判断前无条件查询 AIC/AIV/L1/L0/L2 并计算公共 `GMMNoQuantTilingData`。因此即使终点是 CATLASS，平台查询或公共 tiling 失败仍会阻止启动；不存在“official tiling 失败后改走 CATLASS”的运行时 fallback。

定义：

```text
RISK16 = (X==FP16) &&
         (E>8 || K>=2048 ||
          (hasBias && T && K>=1024 && N>=1536 && maxGroupM>=256))

SMALL32 = (X==FP32 && E==1 && !T && M<=256 && K<=256 && N<=256)

SMALL_T16 = (!RISK16 && X==FP16 && biasDtype==FP16 &&
             T && N<=128 && K<=512 && maxGroupM<=128)
```

精确 selector 顺序：

```text
tiling = CalcOfficialTiling(...)                 # unconditional

if RISK16:
    official Cube full Y
    AIV compact unreliable FP16 cells
    AIV refine min(candidateCount, 4096) indexed dots
    return                                       # R-003

if SMALL32:
    official f32/f32/NT Cube
    return                                       # R-004

if X == FP32:
    CATLASS grouped Slice-M, T or NT
    if hasBias: AIV add FP32 bias
    return                                       # R-002

if SMALL_T16:
    set coreNum=usedCoreNum=E
    official f16/f16/T Cube
    return                                       # R-005

official dtype/bias/layout Cube                  # R-001
```

所有合法 FP32 都被 R-004 或 R-002 提前返回；源码末尾的公共 FP32 dispatch 是防御性分支，在当前已校验 dtype 与上述顺序下不可达。Route 不在 Device 内根据数据切换；R-003 只在已选择的图内根据 Cube 输出筛选需要重算的 cell。

### Route 矩阵

| 重要性 | Route | 精确适用条件 | kernel 图 | blocks | false/结束行为 |
|---:|---|---|---|---|---|
| P0 | R-001 公共 official Cube | BF16 全域；未命中 R-003/R-005 的 FP16 | 一个 AIC-only cgmct entry | `tiling.coreNum` | terminal fallback；非法 dtype 已由 Host 拒绝 |
| P1 | R-002 FP32 CATLASS | FP32 且 `!SMALL32` | AIC-only Slice-M；有 bias 再接 AIV-only bias | AIC=`aicNum`；AIV=`aivNum` | 分支末 return；无失败后 official 回退 |
| P2 | R-003 FP16 有界修复 | `RISK16` | official AIC Cube → AIV scan/atomic compact → AIV indexed refine | Cube=`tiling.coreNum`；两个 AIV 均 `aivNum` | 分支末 return；超过 4096 的候选保留 Cube 值 |
| P3 | R-004 小 FP32 official | `SMALL32`，bias 可有可无 | 一个 `f32_f32_nt` AIC-only entry | `tiling.coreNum` | false 才进入 CATLASS；true 立即 return |
| P4 | R-005 小转置 FP16 | `SMALL_T16`；no-bias 也因 bias ABI 归一化命中 | 一个 `f16_f16_t` AIC-only entry | E | false 进入 R-001；true 立即 return |

### active、conditional 与 dormant 分支

| 类别 | 分支 | 可达性结论 |
|---|---|---|
| active | R-003 `RISK16`、R-004 `SMALL32`、R-002 FP32 CATLASS、R-005 `SMALL_T16`、R-001 public Cube | 五条均由生产 selector 对合法输入直接可达，必须完整保留 |
| conditional | R-003 的 T/NT × no-bias/FP16-bias/FP32-bias；R-002 的 T/NT × bias/no-bias；R-001 的 FP16/BF16 × T/NT ×合法 bias ABI | 都是 active Route 内由 dtype/layout/bias 继续分派的合法叶路径，不得压成单一 entry |
| dormant branch | public dispatch 末尾的 FP32 `else if (T) ... else ...` | 所有合法 FP32 已由 R-004 或 R-002 return；该尾部分支当前生产链不可达。`f32_f32_nt` entry 本身仍被 R-004 使用，只有“尾部再次 dispatch”不可达；若移除前置 FP32 return 才会重新启用 |

合法 dtype 域形成无重叠、无遗漏的 partition：

| dtype | 子域 | terminal Route |
|---|---|---|
| FP16 | `RISK16` | R-003 |
| FP16 | `!RISK16 && SMALL_T16` | R-005 |
| FP16 | `!RISK16 && !SMALL_T16` | R-001 |
| BF16 | 全部合法 T/NT 与 bias 状态 | R-001 |
| FP32 | `SMALL32` | R-004 |
| FP32 | `!SMALL32` | R-002 |

### Route 间不变量

- `group_list` 始终是 int64 累积边界，X/Y 以行前缀推进，weight 以固定 `K*N` 专家 stride 推进。
- 所有 AIC 节点独占其 Y tile；AIV 节点按线性 cell 或唯一 candidate ordinal 分工，不使用 atomic 累加 Y。
- AIV 只在 AIC 完成后读/改正式 Y；没有中间 output tensor。
- `transpose_weight` 不改变输出 layout；Y 始终 RowMajor/连续 `[M,N]`。
- 任何专用分支都不放宽 Host dtype、shape、layout、device 和 group 边界检查。

## P0 R-001：公共 cgmct no-quant Cube

### 类型闭包与入口

公共 body 的动态 tile shape 由 `TCubeTiling` 提供：

```cpp
using L1TileShape = AscendC::Shape<Cgmct::Gemm::_0,
                                  Cgmct::Gemm::_0,
                                  Cgmct::Gemm::_0>;
using BlockScheduler = GroupedMatmulAswtScheduler;
using BlockMmad = BlockGroupedMatmulBuilder<
    XType, RowMajor, XType, LayoutB, XType, RowMajor,
    BiasType, RowMajor,
    L1TileShape, L1TileShape,
    BlockScheduler, MatmulMultiBlockBias<>>;
using Kernel = KernelGroupedMatmul<
    MatmulShape, BlockMmad, BlockEpilogueEmpty, BlockScheduler>;
```

10 个 `DEFINE_GMM_GLOBAL` entry 覆盖：

```text
FP16  × {FP16,FP32 bias ABI} × {NT,T}
BF16  × {BF16,FP32 bias ABI} × {NT,T}
FP32  × FP32 bias ABI        × {NT,T}
```

每个 entry 创建 `TPipe`、设置 overflow 行为并声明 `KERNEL_TYPE_AIC_ONLY`。`OfficialGmmNoQuantAswt` 把 `GMMNoQuantBaseParams` 和 `TCubeTiling` 适配为 `KernelGroupedMatmul::Params`；传入的是四个 `ListTensorDesc` 地址及 group-list 地址，不是裸 tensor 指针。

### Device 数据流与 tile owner

```text
GM descriptors/group ends
  -> group shape/offset state
GM X/W/[bias]
  -> L1 A1/B1
  -> L0A/L0B
  -> MMAD / FP32-sized L0C
  -> Fixpipe cast/write
  -> GM Y
```

一个 AIC entry 内创建并初始化一次 block MMAD 对象，然后遍历所有 group。跨 group 的逻辑 tile 计数持续轮转，不按专家重新启动 Matmul 对象。当前 group 的 tile 通过 `SetSingleShape → SetTensorA/B → [SetBias] → IterateAll` 完成；`hasBias=0` 时不得调用 `SetBias`。

ASWT 以最多 4 个 M tiles 为 window，奇偶 window 反向遍历 N。单 group 时，最后一轮剩余 tile 还可由 `mTailCnt*nTailCnt` 个 core 沿 M/N 拆分。`GetBlockShape` 同时给出 tail 的真实 extent 与地址偏移；只复制蛇形索引而不复制 tail shape/offset 会导致重叠或漏写。

### Host tiling 与资源账

初始资源公式：

```text
baseN = (T && N<256) ? align_up(N,16) : 256
baseK = floor((L0B/2)/(baseN*dtypeBytes), 16)
baseM = min((L0A/2)/(baseK*dtypeBytes),
            L0C/(baseN*4), 256)

totalL1 = L1 - (hasBias ? 1024 : 0)
l1A/l1B = 一侧固定 256 KiB、另一侧取余量；方向由 baseM>baseN 决定
stepKa = floor(l1A/2/baseM/baseK/dtypeBytes)
stepKb = floor(l1B/2/baseN/baseK/dtypeBytes)
将较大 step 向下调到较小 step 的整数倍
depthA1=2*stepKa; depthB1=2*stepKb
dbL0A=2; dbL0B=2; dbL0C=1
```

初始 `baseK/baseM/stepKa/stepKb` 为 0 时抛异常。单 group 会调用 `FormulateBasicBlock` 重新平衡 M/N tile，并令 `baseK=min(align_up(K,16), floor((L0A/2)/(dtypeBytes*max(baseM,baseN)),16))`、`usedCoreNum=min(ceil(M/baseM)*ceil(N/baseN),aicNum)`。随后用 `aswtL1=L1+256` 重算 A/B depth；若两侧 bytes 超过 `aswtL1-biasReserve`，按 `baseM<=baseN` 减半 A depth，否则减半 B depth，再恢复 step 的整数倍关系和双 stage。尾块拆分也在此时计算。多 group 初始 `usedCoreNum=aicNum`。R-005 最后把 `coreNum` 与 `usedCoreNum` 同时覆盖为 E，不能只改 launch blocks。

L2 hint 以 X+W+Y 总 bytes 是否小于 L2，以及 B 内层跨度/128 B divisibility 决定 `weightNoL2Cache`。它只改变 cache hint，不改变数学或地址。

### POD 字段消费

| 字段 | Host producer | Device consumer | 不变量 |
|---|---|---|---|
| `groupNum/groupType/groupListType` | E、0、0 | group loop / `SPLIT_M` / 累积差分 | `groupNum=E`，只支持 split-M cumsum |
| `singleX/singleWeight/singleY` | 全为 1 | descriptor index 与连续偏移 | 一个大 X、W、Y storage |
| `hasBias` | optional bias | `SetBias` gate | false 时 bias pointer 不得读 |
| `mTailCnt/nTailCnt` | 单 group tail 计算 | ASWT `GetBlockShape` | 与 baseM/baseN/core 同源 |
| `weightNoL2Cache` | L2/128 B 规则 | B cache hint | 不影响结果 |
| `coreNum/usedCoreNum` | tiling 或 R-005 覆盖 | launch/scheduler | 二者必须一致 |
| `M/N/Ka/Kb` | M/N/K/K | original shape | 正数、int32 域 |
| `baseM/baseN/baseK` | L0 公式 | tile 几何 | baseK/N 16 对齐起点 |
| `depthA1/depthB1/stepKa/stepKb` | L1 公式 | A1/B1 pipeline | step 整倍数关系 |
| `dbL0A/dbL0B/dbL0C` | 2/2/1 | L0 pipeline | 固定 stage 数 |

### R-001 selector/entry/边界闭包

```text
# 已排除 RISK16、全部 FP32 和 SMALL_T16
if X == FP16:
  entry = (biasDtype==FP32) ? f16_f32_{t|nt} : f16_f16_{t|nt}
else if X == BF16:
  entry = (biasDtype==FP32) ? bf16_f32_{t|nt} : bf16_bf16_{t|nt}
entry<<<tiling.gmmNoQuantParam.coreNum, nullptr, stream>>>(
    xDesc, weightDesc, biasDesc, groupListGm, yDesc, tiling)
```

| 闭包项 | R-001 事实 |
|---|---|
| Host/selector | `CheckInputs` 生成 dtype/bias/layout；`launch_grouped_matmul` 的末尾 public dispatch 是 terminal active 分支 |
| POD/entry | 1280 B metadata；按值 `GMMNoQuantTilingData`；8 个 active FP16/BF16 entry 从 10 个 `DEFINE_GMM_GLOBAL` 实例中选择 |
| template/body | `OfficialGmmNoQuantAswt<XType,BiasType,LayoutB>` → `KernelGroupedMatmul` → ASWT/`MatmulMultiBlockBias` |
| Cube/AIV | 单 AIC-only producer，无 AIV；GM descriptor→L1→L0A/B→L0C→Fixpipe→Y |
| workspace/资源 | 只需固定 1280 B metadata；动态 L1/L0 由 `TCubeTiling`，null launch workspace |
| tail/同步 | ASWT M/N tail 与单 group tail split；K 真实 extent；Matmul pipe 内同步 |
| fallback 出边 | active terminal fallback；Host 非法域或 tiling 异常抛错，不再换 Route；尾部 FP32 branch dormant |

## P1 R-002：FP32 CATLASS grouped Slice-M

### 适用与模板

除 `SMALL32` 外，所有合法 FP32 都走这条 Route，与 transpose、bias、E 和空 group 无关。核心模板为：

```text
ArchTag       = Ascend950
Dispatch      = MmadPingpong<Ascend950, enableUnitFlag=true,
                              useHF32=false>
L1 tile       = [256,256,128]
L0 tile       = [256,256,32]
A/C layout    = RowMajor
B layout      = NT:RowMajor, T:ColumnMajor
Scheduler     = GemmIdentityBlockSwizzle<3,0>
Kernel        = GroupedMatmulSliceMTla<..., int64 group-list>
```

HF32 显式关闭，A/B/C/accumulator 都沿 full-FP32 path。CATLASS Matmul 模板不接 bias；`hasBias` 时由独立 AIV 节点在完整 Y 上加一次 FP32 bias。

### 分组调度、尾块与资源

每个 AIC core 遍历全部专家，但以旋转的 `startCoreIdx` 分摊逻辑 tile：

```text
totalM = 0; gmGroupOffsetB = 0; startCoreIdx = 0
for g:
  currentM = ends[g] - ends[g-1]
  loops = ceil(currentM/256) * ceil(N/256)
  core 从 (coreIdx-startCoreIdx mod aicNum) 开始，每 aicNum 取一个 loop
  A/C tile row offset += totalM
  B base += gmGroupOffsetB
  actualBlockShape 裁剪 M/N tail；block MMAD 内处理 K tail
  totalM += currentM
  gmGroupOffsetB += K*N
  startCoreIdx = (startCoreIdx + loops) mod aicNum
```

静态资源上界由模板编译期检查：L1 A/B 各 `256*128*4*2=256 KiB`；L0A/L0B 各 `256*32*4*2=64 KiB`；L0C 为 `256*256*4=256 KiB`。资源对象和 event 生命周期都在单次 AIC entry 内；kernel 末尾执行 block 同步与 `PIPE_ALL` barrier。`GetWorkspaceSize=0`，CATLASS 数学不消费额外 GM workspace。

### AIV bias 节点

有 bias 时，`aivNum` 个 block 把 `[0,M*N)` 切成连续区间，512 个 SIMT thread 在区间内 stride 扫描。每个 cell 从 row 找 expert 后执行：

```text
Y[row,col] = Y[row,col] + bias[expert(row), col]
```

区间互斥，不需要 atomic；末 block 用 `count=min(perCore,total-begin)` 裁剪。动态 UB 参数为 null，节点只使用 GM 与线程标量。无 bias 时不启动此节点。

### R-002 launch/POD/fallback 闭包

```text
problem = GemmCoord{M,N,K}
params = {problem, E, groupListGm,
          rawX, RowMajor(M,K), rawW, layoutB(T), rawY, RowMajor(M,N)}
catlass_f32_{t|nt}<<<aicNum, nullptr, stream>>>(params)
if hasBias:
  catlass_add_f32_bias<<<aivNum, nullptr, stream>>>(
      rawBias, groupListGm, rawY, E, M, N)
return
```

| 闭包项 | R-002 事实 |
|---|---|
| Host/selector | `X==FP32 && !SMALL32`；T/NT 与 bias/no-bias 都是 active conditional 叶 |
| POD/entry | CATLASS `Params` 按值含 problem/layout/raw pointers；`f32_t` 或 `f32_nt` entry；有 bias 再接 `grouped_matmul_catlass_add_f32_bias` |
| template/body | `GroupedMatmulSliceMTla<BlockMmadTla<MmadPingpong<Ascend950,true,false>>,void,GemmIdentityBlockSwizzle<3,0>,int64_t>` |
| Cube/AIV | AIC full-FP32、HF32 off；AIV 只加一次 FP32 bias，不参与 dot |
| workspace/资源 | 固定 1280 B 中只直接消费 group-list/raw Y；CATLASS `GetWorkspaceSize=0`；静态 L1/L0 资源见上文。public tiling 仍被无条件创建但不被 entry 消费 |
| tail/同步 | Slice-M `GetActualBlockShape` 裁 M/N，block MMAD 处理 K tail；AIC kernel 完成后同 stream AIV 线性 count tail |
| fallback 出边 | 分支末 return；CATLASS launch/runtime 失败不回退 public Cube；predicate false 只可能是前置 R-004 或非 FP32 |

## P2 R-003：FP16 scan/compaction/有界重算

### Route 条件与 launch 图

这条 Route 覆盖以下任一公开结构条件：专家数大于 8、K 至少 2048，或带 bias 的大型转置域。它并不限于转置 weight：

```text
official FP16 Cube full Y
  -> AIV compact_f16_unreliable(Y, candidateCount, indices, limit)
  -> AIV refine_indexed_f16 variant over X/W/bias/ends/Y
```

Cube 仍是全域 producer，按 FP16/FP32 bias dtype 和 T/NT 选择四个 official entry。scan 与 refine 都用 `aivNum` blocks、每 block 512 threads、null dynamic UB。

### scan 与容量边界

scan 对完整 `M*N` 线性输出做互斥区间筛选：

```text
limit = (maxGroupM >= 512) ? 4.0 : 0.03125
candidate := (-limit < float(Y[i]) < limit)
slot = atomic_add(candidateCount, 1)
if slot < 4096: candidateIndices[slot] = i
```

严格开区间意味着等于阈值、NaN 和 Inf 都不进入列表。`candidateCount` 记录全部命中数，indices 只保存最先取得 slot 的 4096 个；AIV core 的原子竞争顺序不是稳定排序。若候选超过容量，refine 只处理已记录的 4096 个，其余单元保留 Cube 值。这是明确的有界退化，不是全输出重算。

### indexed dot 与专家定位

refine 令 `available=min(candidateCount,4096)`，并把 ordinal 跨 AIV core 交错：

```text
ordinal = blockIndex + threadIdx * blockCount
ordinalStride = 512 * blockCount
```

每个 ordinal 只有一个线程。对 `linear` 恢复 row/col 和 expert 后，索引公式是：

```text
xBase = row*K
wBase(T)  = (expert*N + col)*K
wBase(NT) = expert*K*N + col
wIndex(T)  = wBase(T) + kk
wIndex(NT) = wBase(NT) + kk*N
```

重算遵循固定 192 元素块：块内用 FP32 `fmaf(av,bv,blockSum)` 顺序归约，各块结果再用普通 FP32 `result += blockSum` 汇总。若有 bias，在全部 dot 完成后按真实 bias dtype 转 FP32 并只加一次，最后 cast 回 FP16 Y。该实现不是 hi/lo 补偿和，也不是 `M*N*K` 全量 scalar 重算。

### 同步与写安全

- H2D copy 清零 scratch；Cube 后的同 stream 顺序保证 scan 看到正式 Y。
- scan 内只有 `candidateCount` 使用 global atomic；indices 的 slot 唯一，不冲突。
- refine launch 在 scan 完成后读取 count/indices；ordinal 映射唯一，因此每个被选 Y cell 只有一个 writer。
- scan 不修改 Y；未命中、未收录或超容量的 cell 保持 Cube 输出。

### R-003 entry/POD/fallback 闭包

```text
cubeEntry = (biasDtype==FP32) ? f16_f32_{t|nt} : f16_f16_{t|nt}
cubeEntry<<<coreNum, nullptr, stream>>>(descriptors, groupListGm, tiling)
compact_f16_unreliable<<<aivNum, nullptr, stream>>>(
    rawY, workspace+1280, workspace+1288, M*N, limit)
refineEntry = !hasBias ? refine_indexed_f16
            : biasDtype==FP32 ? refine_indexed_f16_f32_bias
            : refine_indexed_f16_bias
refineEntry<<<aivNum, nullptr, stream>>>(
    rawX, rawW, rawBias, groupListGm, rawY,
    workspace+1280, workspace+1288, E, K, N, T)
return
```

| 闭包项 | R-003 事实 |
|---|---|
| Host/selector | `RISK16` 在 plugin scratch 规划和 launcher 中使用同一谓词；T/NT、三种 bias 状态均合法可达 |
| POD/entry | Cube 消费 `GMMNoQuantTilingData`；AIV 消费 raw pointers、E/K/N/T、count/indices；4 个 Cube 叶和 3 个 refine 叶 |
| template/body | Cube 复用 `OfficialGmmNoQuantAswt`；scan=`CompactUnreliableOutputs<half>`；refine=`RefineIndexedDots<half,BiasT,HasBias>` |
| Cube/AIV | AIC 先写全 Y；第一个 AIV 只筛选/原子压缩；第二个 AIV 只覆盖已收录的唯一 cell |
| workspace/资源 | 1280 B metadata + 32776 B scratch；容量不足在任何 kernel launch 前抛错；AIV null dynamic UB、线程标量 |
| tail/同步 | Cube ASWT tail；scan 连续 count tail；refine ordinal tail；dot 最后一块 `min(kBase+192,K)` |
| fallback 出边 | terminal return；候选超 4096 的定义退化是保留 Cube 值，不启动第二轮；workspace 不足或 launch 失败不回退 R-001 |

## P3/P4 小域专用 Route

### R-004：小型单专家非转置 FP32

条件是 `E=1 && !T && M,K,N<=256`，bias 可有可无。它在 CATLASS 分支前启动 `grouped_matmul_f32_f32_nt`，使用公共 official tiling/core 数并立即 return。该 Route 的意义是避免 grouped Slice-M 固定 setup；它不是数值或 layout fallback。

```text
if X==FP32 && E==1 && !T && M<=256 && K<=256 && N<=256:
  grouped_matmul_f32_f32_nt<<<coreNum, nullptr, stream>>>(
      xDesc, weightDesc, biasDesc, groupListGm, yDesc, tiling)
  return
```

| 闭包项 | R-004 事实 |
|---|---|
| Host/POD/entry | `SMALL32`；1280 B metadata + 按值 public tiling；`grouped_matmul_f32_f32_nt` |
| template/Cube-AIV | `OfficialGmmNoQuantAswt<float,float,RowMajor>`；一个 AIC-only Cube，bias 由 `hasBias` 控制在 Matmul 中融合，无 AIV |
| workspace/tail | 无 scratch；public L1/L0/ASWT、M/N/K tail 与单 group重整全部保留 |
| fallback | true 立即 return；false 后 FP32 必进 R-002；launch 失败无 CATLASS 重试 |

### R-005：小 N 转置 FP16

代码首先令 `baseN=align_up(N,16)`（所有 `T && N<256` 的 official tiling 都如此），再在更窄子域：

```text
T && N<=128 && K<=512 && maxGroupM<=128 &&
X==FP16 && biasDtype==FP16 && !RISK16
```

覆盖 `gmmNoQuantParam.coreNum=E` 和 `mmTilingData.usedCoreNum=E`，以 E blocks 启动 `f16_f16_t`。由于 R-003 在前，`E>8` 已先进入数值 Route；所以该分支当前可达域实际也满足 `E<=8`。FP32 bias 的 FP16 输入不命中该 specialization，走 R-001；不得只看 `hasBias` 而忽略 `biasDtype`。

缩小 baseN 后必须重算 baseK/baseM、L1 steps、depth 和 tail 字段；只改 launch blocks 或只改 baseN 都会破坏 POD 与资源账的一致性。

```text
if !RISK16 && X==FP16 && biasDtype==FP16 &&
   T && N<=128 && K<=512 && maxGroupM<=128:
  tiling.gmmNoQuantParam.coreNum = E
  tiling.mmTilingData.usedCoreNum = E
  grouped_matmul_f16_f16_t<<<E, nullptr, stream>>>(
      xDesc, weightDesc, biasDesc, groupListGm, yDesc, tiling)
  return
```

| 闭包项 | R-005 事实 |
|---|---|
| Host/POD/entry | `SMALL_T16`；1280 B metadata；覆盖 public tiling 的两个 core 字段；`grouped_matmul_f16_f16_t` |
| template/Cube-AIV | `OfficialGmmNoQuantAswt<half,half,ColumnMajor>`；一个 AIC-only Cube，无 AIV；no-bias 与 FP16 bias 都由同 entry 覆盖 |
| workspace/资源 | 无 scratch；`baseN=align_up(N,16)` 后完整重算 baseK/baseM/L1 depth；E blocks 与 `usedCoreNum` 同源 |
| tail/同步 | public ASWT M/N/K tail；空 group跳过；一个 entry 内 Matmul pipe 同步 |
| fallback | false 落 R-001；FP32 bias 的 FP16 输入明确落 R-001；E>8 先命中 R-003；true 后立即 return，无运行时回退 |

## POD、launch 参数与 ABI

### Descriptor ABI

| 对象/字段 | byte offset | 类型/值 | Device 用途 |
|---|---:|---|---|
| X payload/rank/dims/pointer | 0/8/16/24/32 | u64；payload=32，rank=2，M/K，GM ptr | cgmct ListTensorDesc |
| W payload/rank/dims/pointer | 64/72/80/88/96/104 | u64；payload=40，rank=3，E/(K,N 或 N,K)，GM ptr | B shape/address |
| bias payload/rank/dims/pointer | 128/136/144/152/160 | u64；payload=32，rank=2，E/N，GM ptr | bias；`hasBias=0` 不读 |
| Y payload/rank/dims/pointer | 192/200/208/216/224 | u64；payload=32，rank=2，M/N，GM ptr | output shape/address |
| `groupEnds[i]` | `256+8*i` | int64，`0<=i<E` | official、CATLASS 与 AIV 共用 |

Host 将未使用的 group-list slot `[E,128)` 和 descriptor gap 保持为 0；Device consumer 仍必须以 E 为读取上界。R-003 scratch 紧接固定 1280 B 区，不能把 group-list 区截短。

### launcher 标量/POD

`launch_grouped_matmul` 接收裸 X/W/bias/Y、workspace pointer/capacity、E/maxGroupM/M/K/N、X/bias dtype code、hasBias、transpose 和 stream。裸指针供 CATLASS/AIV 使用，descriptor 地址供 public cgmct body 使用；两套地址必须指向相同 tensors。

`GMMNoQuantBaseParams` 固定 48 B，字段顺序为：

```text
groupNum, coreNum, singleWeight, singleX, singleY,
groupType, groupListType, hasBias,
mTailCnt, nTailCnt, weightNoL2Cache, placeHolder
```

外层 `GMMNoQuantTilingData={48-B base params, TCubeTiling}` 按值传给 official entry。`TCubeTiling` 的完整 size/alignment/offset 由 CANN 9.1.0 `kernel_tiling.h` 决定，Host producer 和 Device consumer必须使用同一工具链；不能手写猜测其数字 ABI。CATLASS entry 传自己包含 `GemmCoord`、layouts、group-list 与裸指针的 `Params`，不消费 `GMMNoQuantTilingData`，但 launcher 仍会预先生成它。

## 资源、尾块、同步与 fallback 审计

| 维度 | official Cube | FP32 CATLASS | FP32 bias AIV | FP16 scan AIV | FP16 indexed refine AIV |
|---|---|---|---|---|---|
| GM 输入 | descriptors→X/W/[bias] | raw X/W/group ends | raw bias/group ends/Y | Y/count/indices | raw X/W/[bias]/ends/Y/count/indices |
| 片上资源 | 动态 L1/L0 tiling；L0A/B 2 stage、L0C 1 stage | 静态 L1/L0 ping-pong；HF32 off | thread scalars | thread scalars；global atomic count | thread scalars |
| 动态 UB | 无独立参数 | 无独立参数 | null | null | null |
| 分核 | `coreNum`，R-005 为 E | `aicNum` | `aivNum` 连续 cell ranges | `aivNum` 连续 cell ranges | `aivNum` 交错 ordinals |
| M/N tail | ASWT `GetBlockShape` | `GetActualBlockShape` | count clamp | count clamp | ordinal<available |
| K tail | TCubeTiling/Matmul真实 extent | block MMAD 内部 K loop/tail | N/A | N/A | 最后一段 `min(kBase+192,K)` |
| 跨节点同步 | N/A | AIC 内 event/barrier | 同 stream 等 CATLASS | 同 stream 等 official Cube | 同 stream 等 scan |

失败与退化关系：

- 平台对象为空，AIC/AIV/L1/L0 任一为 0，初始 `baseK/baseM` 为 0，或初始 L1 step 为 0时抛异常；所有 Route 都受此约束。
- R-003 还检查 `workspaceCapacity>=1280+32776`，不足时抛异常，不退回只跑 Cube。
- Host 非法输入在任何 launch 前终止；空 group 是合法 Device no-op，不是失败。
- 专用 predicate 为 false 时按 selector 落到下一 Route；kernel 启动后的 runtime 失败不会改选其它 Route。
- R-003 candidate overflow 的定义退化是“未收录 cell 保留 Cube 值”；R-002 无 bias 时省略 AIV；R-001/R-004/R-005 都是单 AIC 节点。
- launch 语句没有同步返回码检查，异步 device 错误由 stream/runtime 外层报告。

## 实现文件闭环

| 文件职责 | 关键符号 | 必须保留的实现事实 |
|---|---|---|
| plugin | `CheckInputs`、`MakeOutputs`、`grouped_matmul_npu` | E<=128；cumsum；共享 storage；full-capacity zero H2D；返回 storage owner 与 Host/input capture 外部边界 |
| launch header | dtype enum、descriptor offsets、`GMM_DIRECT_MAX_GROUPS`、C ABI | 固定 workspace 1280 B，不是 320 B |
| official ABI | `GMMNoQuantBaseParams`、`GMMNoQuantTilingData` | 48 B base params；SDK `TCubeTiling` 同工具链 |
| kernel TU | `CalcOfficialTiling`、selector、10 official entries、2 CATLASS entries、4 AIV entry 族 | Route 顺序与同 stream graph |
| cgmct closure | `KernelGroupedMatmul`、ASWT scheduler、builder/MMAD/tile copy/fixpipe | public Cube body、group offsets、tail 与资源生命周期 |
| CATLASS closure | `GroupedMatmulSliceMTla`、`BlockMmadTla<MmadPingpong>`、TLA layouts | full-FP32、HF32 off、Slice-M group/tail 调度 |
| operator CMake | direct kernel/plugin registration 与 private include root | cgmct/CATLASS/TLA 依赖随 operator-private source 构建 |

公共 cgmct 与 CATLASS 都不是 SDK 中可假设存在的隐式依赖。若只复制最外层 kernel TU 而遗漏实际实例化的 private header 闭包，构建与类型语义都不闭合。

## 算子特定实现要点

### T-001：一个 cumsum 同时服务三类 consumer

同一 `[E]` int64 累积数组的 Device 副本同时被 official cgmct、CATLASS Slice-M 和 AIV expert lookup 消费。它位于返回 Tensor 持有的 workspace storage 内；Host source vector 对异步 H2D 的安全期仍属于 `RunOpApi`/runtime 契约。三类 consumer 必须使用相同 offset 与 cumsum 解释；若把其中一条路径改成 group lengths，Route 之间会在空专家和 weight offset 上分叉。

### T-002：FP16 修复是“扫描全 Y、有限 dot”

复杂度边界为 `O(M*N)` 扫描加至多 `4096*K` 的 scalar dot，不是 `O(M*N*K)`。容量限制、原子 slot、严格阈值、192 元素块归约和 bias 后加顺序共同构成这条 Route；任何一项改变都会改变候选集合、数值顺序或成本上界。

### T-003：FP32 bias 与 CATLASS Matmul解耦

CATLASS 模板将 bias type 设为 void，确保 MMAD 只写 matmul 结果；AIV 在同 stream 对每个 cell 加一次 FP32 bias。把 bias 同时接入 AIC 和 AIV 会重复加，省略同 stream 顺序则可能读到未完成 Y。

### T-004：small Route 是前置短路，不是尾部 fallback

R-004 必须位于通用 FP32 CATLASS 之前；R-005 必须位于 common FP16/BF16 dispatch 之前；R-003 又必须先于 R-005。仅复制谓词而打乱 return 顺序，会使同一 Y 被两个 producer 写或让 scratch 未分配却启动 scan。

## 限制与待验证边界

- 只覆盖 no-quant、连续 ND、split-M cumsum、`E<=128`；不覆盖量化、NZ weight、非连续输入或其它 group type。
- R-003 最多修复 4096 个候选，超额候选的选择顺序受多核 atomic 到达顺序影响；源码没有第二轮或容量扩展。
- public Cube 单 group 重整后会再次计算 baseK，源码没有第二个显式 `baseK==0` guard；目标 950 资源配置是该公式的前提。
- `CalcOfficialTiling` 即使在 CATLASS Route 也先执行；不能删除它而不重新审计 selector、平台资源异常和 small Route。
- `TCubeTiling` 数字 ABI、CATLASS resource 常量及 task-type 构建元数据受 CANN 9.1.0/dav-3510 工具链约束，不向 A2/A3 外推。
- 本卡没有把 direct-only CATLASS/AIV 分支归因于 Frontmatter 的公开源码；公开链接只闭合相关 950 no-quant 映射。

## 源码索引

| 链路 | 官方映射 | direct 实现符号 |
|---|---|---|
| API/shape 语义 | `op_api/grouped_matmul.h` | Torch schema、`CheckInputs`、`MakeOutputs` |
| 950 no-quant tiling | `grouped_no_quant_matmul_tiling.cpp` | `CalcOfficialTiling`、`FormulateBasicBlock` |
| public Cube body | `grouped_matmul_basic_kernel.h` | `OfficialGmmNoQuantAswt`、10 个 `DEFINE_GMM_GLOBAL` entry |
| direct launcher | 相关 APT router | `launch_grouped_matmul` 的五条 terminal Route |
| FP32 specialization | direct-only | `CatlassF32Grouped::Route`、`GroupedMatmulSliceMTla`、`AddFp32Bias` |
| FP16 specialization | direct-only | `CompactUnreliableOutputs`、`RefineIndexedDots` |
| 构建 | operator CMake | direct registration + operator-private cgmct/CATLASS/TLA closure |