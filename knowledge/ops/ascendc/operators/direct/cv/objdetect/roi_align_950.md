---
type: Operator
title: "ROIAlign 950 Direct Invoke 算子知识"
description: "说明 cann_bench::roi_align、roi_align_npu 与 launch_roi_align 在 950 上的接口语义、Host 参数生成、四路静态分发、双 Kernel 启动图与 SIMT 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/docs/aclnnRoiAlignV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/op_graph/roi_align_proto.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/op_host/op_api/aclnn_roi_align_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/op_host/op_api/aclnn_roi_align.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/op_host/op_api/roi_align.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:39:32Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ROIAlign 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::roi_align` → `roi_align_npu` → `launch_roi_align`；Direct Invoke |
| 计算语义 | 对每个 ROI 的每个 `(channel, pooled_h, pooled_w)` bin 做规则网格双线性采样，再按采样点数求平均 |
| 输入输出 | `x[B,C,H,W]`、`boxes[R,5]` 与 `y[R,C,OH,OW]`；FP16/FP32、contiguous NCHW/ND |
| 执行类型 | 四条 Route 均为 `SIMT geometry + SIMT forward` 两节点图；两个 entry 都是 `KERNEL_TYPE_AIV_ONLY` wrapper，无 Cube/AIC 或 MIX_CV |
| Kernel 路线 | 复用价值顺序：P0 R-003 FP32 unaligned；P1 R-001 FP16 unaligned；P2 R-002 FP16 aligned；P3 R-004 FP32 aligned。四路由 dtype×aligned 唯一决定；fixed/adaptive sampling 与 32/64 位 flat decode 是 Route 内局部分支 |
| 核心资源 | boxes/features/geometry/output GM 与线程私有标量；geometry 为 `R×12×4` B 的 invocation-local GM Tensor；无 AIV UB、dynamic UB、Device workspace、atomic 或 barrier |
| 来源闭包 | `direct_implementation_closed`：交付候选闭合 Torch schema、Meta/PrivateUse1、Direct POD、四实例 router、双 Kernel body、同流 GM 边与构建；固定官方源码仅支撑接口/语义/Host 编排相关事实，不把 Direct 派生符号表述为官方实现 |
| 证据边界 | 已闭合转换载荷的 Host、POD、launcher、可达 Device 实例与构建接线；固定 HTTP 源码只支撑公开接口和 ROIAlignGrad 950 同族契约，不支撑将 Direct forward 派生体称为官方原生实现 |

## 接口与语义

### 计算语义

对 `boxes[r]=(batch,x0,y0,x1,y1)`，先把四个坐标乘 `spatial_scale`。`aligned=true` 时四端点再减 `0.5`；`aligned=false` 时不偏移，并将 `roiWidth=max(x1-x0,1)`、`roiHeight=max(y1-y0,1)`。令：

```text
binH = roiHeight / OH
binW = roiWidth  / OW
gridH = sampling_ratio > 0 ? sampling_ratio : max(ceil(binH), 1)
gridW = sampling_ratio > 0 ? sampling_ratio : max(ceil(binW), 1)
stepY = binH / gridH
stepX = binW / gridW
sampleY = roiStartH + ph*binH + (iy+0.5)*stepY
sampleX = roiStartW + pw*binW + (ix+0.5)*stepX
y[r,c,ph,pw] = Σ bilinear(x[batch,c], sampleY, sampleX) / (gridH*gridW)
```

坐标小于 `-1` 或大于对应空间尺寸时该样点贡献 0；`[-1,0]` 夹到 0；末像素区域把 low/high 都夹到 `size-1`。四个邻点按 `w1*v1 + w2*v2 + w3*v3 + w4*v4` 的固定次序相加。FP16 输入先转 FP32 做几何、插值和归约，仅最终结果转回 FP16。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[B,C,H,W]`，Host materialize 为 contiguous NCHW | FP16/FP32 | rank=4；Direct Host 未单独拒绝零 `B/C/H/W`，但有效 ROI 与除数要求合法正维 |
| `boxes` | `[R,5]`，contiguous ND；列为 `(batch,x0,y0,x1,y1)` | 与 `x` 相同 | rank=2 且 dim1=5；batch 列先转 FP32 再转 int32；Direct 不做越界 batch 恢复 |
| `outputHeight` | Host 标量 | int64→int32 | 必须 `>0`；写入 POD 时窄化为 int32 |
| `outputWidth` | Host 标量 | int64→int32 | 必须 `>0`；写入 POD 时窄化为 int32 |
| `spatial_scale` | Host 标量 | double→float32 | 必须 `>0`；Direct 未显式拒绝 `+Inf` |
| `sampling_ratio` | Host 标量 | int64→int32 | Host 接受 `>=-1`；`<=0` 走 adaptive，`>0` 为固定 H/W 网格 |
| `aligned` | Host 标量 | bool | 选择编译期 `ALIGNED` 实例；true 偏移 `-0.5`，false 做最小 extent 1 |
| `y` | `[R,C,OH,OW]`，contiguous NCHW | 与 `x` 相同 | 每个输出标量由唯一 SIMT lane 完整写出，不与输入别名 |

### 入口与版本

交付注册 `roi_align(Tensor x, Tensor boxes, int outputHeight, int outputWidth, float spatial_scale, int sampling_ratio, bool aligned) -> Tensor y`，并绑定 Meta 与 `PrivateUse1`，不是 ACLNN 两段式接口。目标是 CANN 9.1.0、950/Ascend 950。

固定官方版本中的 `aclnnRoiAlignV2` 是最接近的公开 forward 语义边界：同样接收 `[K,5] boxes`、FP16/FP32、`aligned` 与平均池化；其 Host 把 false/true 分别映射到 `roiEndMode=0/2`。但该文档明确标记 Ascend 950PR/950DT 不支持，且该版本的 forward `objdetect/roi_align` 没有 `op_host/arch35` 或 `op_kernel/arch35`。另一固定官方版本中的 `roi_align_grad` 则公开了 950 Tiling、AIV-only SIMT body 和模板 key，可用于核对同族的五列 ROI、mode 0 与 half-pixel mode 2/3 geometry、fixed/adaptive grid、每核最少 1024 元素与 32 元素对齐。因此：

- **官方 forward 来源**：Frontmatter 中固定到 `bb717d...` 的 V1/V2 文档、Op API、L0 launcher 与 OpProto，只证明参数布局、输出 shape、contiguous/格式转换、`aligned` 语义和 launcher 编排；不证明 950 ACLNN 可用或 Direct body 来源。
- **官方 950 同族来源**：固定到 `fcc3b3...` 的 ROIAlignGrad arch35 文件直接证明 256-lane AIV-only、`roiEndMode` 编译期派发、五列 ROI geometry 和 Host 分核公式。Direct forward 复用了这些同族契约，但将 lane 数改为 512，且没有复制 backward 的 gather body。
- **Direct Invoke 适配**：Torch schema/Meta/PrivateUse1、`ROIAlignTilingData`、C ABI launcher、current-stream 双节点和 direct CMake 是转换层适配。
- **Device 派生补充路线**：四个 forward 模板实例、`geometry[R,12]` GM 协议、输出独占插值、8 路顺序归约、FFMA 乘法舍入与正整数除法修正都是转换实现新增的 forward 体，不是官方 forward 950 原生源码。Frontmatter 仅保留可复核的固定 HTTP 源。

## Host/Launcher 执行流程

### 参数校验与前处理

`roi_align_meta` 只用 `boxes.size(0)`、`x.size(1)` 与两个 pooled 属性构造输出，不执行 rank、dtype 或属性校验。真实 NPU 路径中，`roi_align_npu` 按顺序执行：设备 guard 绑定 `x.device()`；检查 `x.dim()==4`；检查 `boxes.dim()==2 && boxes.size(1)==5`；检查两输入 dtype 相同且为 FP16/FP32；检查 `OH/OW>0`、`spatialScale>0`、`samplingRatio>=-1`。之后分别执行 `x.contiguous()`、`boxes.contiguous()`，分配 `output[R,C,OH,OW]` 和 `geometry[R,12]`（INT32）。

Host 未检查 `boxes.device()==x.device()`、batch 列的有限性/整数性/范围、四坐标的有限性与有序性、正 `B/C/H/W`、`samplingRatio` 上限、int64→int32 窄化范围及乘法溢出。`spatialScale>0` 会拒绝 NaN 和非正值，但会接受 `+Inf`，且 double→float 还可溢为 `+Inf`。任一显式 `TORCH_CHECK` 失败都在 contiguous/分配/launch 之前终止；未编码的条件必须由调用方保证。

### 参数、Workspace 与生命周期

设 `O=R*C*OH*OW`。`calc_roi_align_tiling` 把 shape/属性写入 POD，并查询 `GetCoreNumAiv()`：

```text
maxAiv = max(GetCoreNumAiv(), 1)
perCore0 = ceil(O / maxAiv)
perCore1 = max(perCore0, 1024)
perCore  = align_up(perCore1, 32)
blockDim = O==0 ? 1 : max(ceil(O / perCore), 1)
```

`O` 在 POD 中是 uint64，但分核路径把它转为 int64；源码的 `CeilDiv(x,y)=(x+y-1)/y` 使用普通有符号加法。除分母为正外，三段算术还分别要求：初次均分满足 `O<=INT64_MAX-(maxAiv-1)`，保证 `int64(O)+maxAiv-1` 可表示；令 `p=max(CeilDiv(O,maxAiv),1024)`，32 元素对齐满足 `p<=INT64_MAX-31`，保证 `p+31` 及 `((p+31)/32)*32` 可表示；最终求核数在 `O==0` 时直接取 1，否则满足 `O<=INT64_MAX-(perCore-1)`，保证 `int64(O)+perCore-1` 可表示。源码没有显式 checked/saturating overflow 保护，这些都是有效调用前提。官方 ROIAlignGrad 950 Tiling 直接支撑“核数均分→每核最少 1024→32 元素对齐”这一同族公式；Direct forward 将工作量换成 `O`，并自行添加 `maxAiv<=0` 时的 1-core 保护。

两个 Kernel 共享同一 POD。K-001 geometry 固定 `blockDim=1`；K-002 forward 使用上式 `blockDim`。无 opaque Device workspace；唯一跨节点中间量是 `geometry` GM Tensor，容量 `48R` B。`features`、`rois`、`geometry` 是 `roi_align_npu` 的局部 Tensor，`output` 被返回；`RunOpApi` 回调只按值捕获裸指针、POD 和 stream，并不捕获 Tensor owner。局部 Tensor 在回调执行及两次 launch 提交期间有效；函数返回后的异步保活依赖 Torch NPU 当前流 allocator 语义，源码未显式 `record_stream` 或持有 Tensor 的 event。

### 启动图与失败传播

```text
roi_align_npu
  -> calc_roi_align_tiling(shape, attrs)
  -> dtypeTag × aligned router
  -> K-001 roi_align_geometry_entry<T,ALIGNED><<<1,current_stream>>>
       boxes[R,5] -> geometry[R,12] uint32 words
  -> same-current-stream ordering
  -> K-002 roi_align_entry<T,ALIGNED><<<blockDim,current_stream>>>
       features + geometry -> output[R,C,OH,OW]
```

| 节点/边 | entry/body | grid/lane | 输入→输出 | ordering、owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| K-001 geometry | `roi_align_geometry_entry<T,ALIGNED>` → `BuildROIGeometry` | 1 AIV block × 512 SIMT lane | boxes→geometry | 当前 stream；geometry Tensor 属于本次调用 | launcher 为 `void`，无逐节点返回码 |
| E-001 | geometry GM，offset `48*r`、extent `48R` B、12-word layout | N/A | K-001→K-002 | 同 current stream 提交保证 producer 先于 consumer；allocator 必须在 K-002 完成前保留 storage | 无显式 event/错误短路 |
| K-002 forward | `roi_align_entry<T,ALIGNED>` → `ROIAlignForward` → `EvaluateOutput` | `blockDim` AIV blocks，每块 512 lanes | features+geometry→output | 当前 stream；每 lane 唯一 output owner | 同上 |

`OpCommand::RunOpApi` 只包装 direct-launch lambda 以接入框架 stream/error 机制；lambda 无条件调用 `launch_roi_align` 并固定返回 0。代码没有 ACLNN ROIAlign、ATen compute、CPU 或外部算子 fallback，也没有 K-001 失败后抑制 K-002 的 launcher 返回码。

## Route 与参数设计

### 分核与资源计算

K-001 以 `roiIndex=threadIdx.x; roiIndex<R; roiIndex+=512` 分配 ROI。K-002 先按 AIV block 切连续输出区间：

```text
devicePerCore = ceil(O / GetBlockNum())
begin = devicePerCore * blockIdx
count = min(devicePerCore, O-begin), begin>=O 时为 0
for local = threadIdx.x; local<count; local+=512:
    outIndex = begin+local
```

Host 的 `perCore` 只用于求 `blockDim`；Device 再以实际 `GetBlockNum()` 均分，所以两者不应混为一个 ABI 字段。`roi_align_entry` 在 `count==0` 时直接返回；任一 `count>0` block 都会在 `asc_vf_call` 前无条件为 `OW/OH/C` 预计算三组 magic/shift。`O<=UINT32_MAX` 只决定 `ROIAlignForward` 是否消费这些参数并用 `Simt::UintDiv` 解码 `(r,c,ph,pw)`；更大输出忽略已预计算的参数，改用 uint64 `%`/`/`。两个节点都直接 GM→线程私有标量→GM，不申请 UB。

### Route 选择

可达静态 Route 只由 `dtypeTag×aligned` 的 2×2 直积产生，共四条。R-001…R-004 是本卡的稳定标签，不是源码中的数字 tiling key；可执行的真实 key 是表中的 dtype/bool 对。下表按复用价值 P0→P3 排列，不伪装成 launcher 源码顺序；“实际短路”栏保留 `dtypeTag` 外层、`aligned` 内层的真实判断关系。

| 价值 | Route | Key/静态实例 | 完整命中条件 | 实际短路与不适用去向 |
|---:|---|---|---|---|
| P0 | R-003 FP32 unaligned | `FLOAT32/false`→`<float,false>` | `x/boxes` 均为 Float，`aligned=false`，Host 显式校验通过 | H0 走 `dtypeTag!=FLOAT16` else，H1 走 `aligned==false` else；true 去 R-004；其它 dtype 在 Host 失败 |
| P1 | R-001 FP16 unaligned | `FLOAT16/false`→`<half,false>` | `x/boxes` 均为 Half，`aligned=false`，Host 显式校验通过 | H0 命中 `dtypeTag==FLOAT16`，H1 走 false 分支；true 去 R-002；无 Device fallback |
| P2 | R-002 FP16 aligned | `FLOAT16/true`→`<half,true>` | `x/boxes` 均为 Half，`aligned=true`，Host 显式校验通过 | H0 命中 FP16，H1 首分支命中 true；false 去 R-001 |
| P3 | R-004 FP32 aligned | `FLOAT32/true`→`<float,true>` | `x/boxes` 均为 Float，`aligned=true`，Host 显式校验通过 | H0 走 dtype else，H1 首分支命中 true；false 去 R-003 |

排序是知识复用顺序，不是运行时或速度优先级：先放 Python wrapper 默认的 unaligned 语义，以 FP32 作为无末端 cast 的基线，再放 FP16 升/降精度变体；然后保留同样的 FP16/FP32 aligned 配对。launcher 不读 P0–P3，只执行下方决策树。

```text
dtypeTag = (features.dtype == Half) ? FLOAT16 : FLOAT32
if dtypeTag == FP16:
    if aligned: R-002 else: R-001
else:
    if aligned: R-004 else: R-003
# samplingRatio 不参与 Host Route 分发；进入 K-001 后：
if samplingRatio > 0: fixed grid
else: adaptive grid
# totalOutputElements 不改变 Route；进入 K-002 后：
if O <= UINT32_MAX: magic/shift UintDiv decode
else: uint64 / and % decode
```

#### 分类穷尽性

| 分支轴 | 可达值 | 是否独立 Route | 理由 |
|---|---|---|---|
| dtype | Half / Float | 是 | 改变 GM load/store 类型与末端 cast，launcher 实例不同 |
| `aligned` | false / true | 是 | 改变 `BuildROIGeometry<T,ALIGNED>` 编译期 body |
| `samplingRatio` | `>0` fixed / `<=0` adaptive | 否 | 同一 geometry body 内的运行时局部分支，不改 entry、资源或同步图 |
| flat index | `O<=UINT32_MAX` / `O>UINT32_MAX` | 否 | 同一 forward body 内切换 magic `UintDiv` 与 uint64 `/,%` |
| 正整数除法 | 2 的幂 / 普通热路修正 / 整数有效数 fallback | 否 | `CorrectedDivByPositiveInt` helper 内的数值分支 |
| 取样边界 | 无效为 0 / 低端 clamp / 末像素 clamp / 内部四点 | 否 | 逐 sample 的 `BuildBilinearPoint` 分支 |
| `EXACT` 模板 | true / false | 只有 true 可达 | `ROIAlignForward` 无条件实例化 `EvaluateOutput<...,true>`；false 体、`maxMagnitudeBits` 跟踪与 `ExactMulFloat` 未被现行入口调用，不得列为 fallback Route |

### 模板/实现执行矩阵

| 重要性 | Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---:|---|---|---|---|---|---|---|
| P0 | R-003 | float/false | FP32、unaligned、Host shape/attr 校验通过、950 | K-001 geometry SIMT→K-002 forward SIMT | `BuildROIGeometry<float,false>`；`ROIAlignForward<float,false>`；`EvaluateOutput<float,false,true>` | 48R B geometry GM；同流；FP32 I/O | extent clamp≥1；ROI/output 双 grid-stride tail；无 fallback |
| P1 | R-001 | half/false | FP16、unaligned、Host shape/attr 校验通过、950 | K-001→K-002 双 SIMT | `BuildROIGeometry<half,false>`；`ROIAlignForward<half,false>`；`EvaluateOutput<half,false,true>` | 48R B geometry GM；同流；FP32 intermediate→half | extent clamp≥1；双 tail guard；无 fallback |
| P2 | R-002 | half/true | FP16、aligned、Host shape/attr 校验通过、950 | K-001 geometry SIMT→K-002 forward SIMT | `BuildROIGeometry<half,true>`；`ROIAlignForward<half,true>`；`EvaluateOutput<half,true,true>` | 48R B geometry GM；same-current-stream；FP32 intermediate→half | 端点减0.5、不做 extent clamp；ROI/output 双 tail guard；无 fallback |
| P3 | R-004 | float/true | FP32、aligned、Host shape/attr 校验通过、950 | K-001 geometry SIMT→K-002 forward SIMT | `BuildROIGeometry<float,true>`；`ROIAlignForward<float,true>`；`EvaluateOutput<float,true,true>` | 48R B geometry GM；same-current-stream；FP32 I/O | 端点减0.5、不做 extent clamp；ROI/output 双 tail guard；无 fallback |

### 执行边界与性能保护

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV core floor | `maxCoreNum=max(GetCoreNumAiv(),1)` | 防止平台查询异常造成除零 | 全部 | 退为1 core | Direct `calc_roi_align_tiling` |
| 每核最小工作 | `perCore=max(ceil(O/maxAiv),1024)` | 限制小任务启动过多 AIV block | 全部 | 用更多每核输出换较少 block | 同上 |
| 32元素对齐 | `perCore=align_up(perCore,32)` | 复用官方 ROIAlignGrad 950 同族调度粒度；只影响 block 数，不分配 UB tile | 全部 | 无其它 Route | Direct `calc_roi_align_tiling`；official `CalcCoreDistribution` |
| SIMT lane 数 | `kThreadNum=512` | 决定两个 `asc_vf_call` 的 lane 数和循环步长 | 全部 | 无动态 lane Route | Direct `kThreadNum`；文件头历史注释的 256 与现行常量不一致，以常量为准 |
| 32位 flat decode | `O<=0xffffffff` | `Simt::UintDiv` 的 divisor/index 为 uint32 | 全部 | 大 O 用 uint64 `/`、`%` | `roi_align_entry` |
| fixed/adaptive | `samplingRatio>0` | fixed 可直接复用属性；非正值需由 ROI/bin 尺寸计算 H/W grid | 全部 | `<=0` 到 adaptive | `BuildROIGeometry` |
| 空输出 | `O==0` 时 `blockDim=1` | K-002 `count=0` 直接返回；K-001 仍按 `R` 决定是否生成 geometry | 全部 | `R==0` 时两节点无有效迭代；`R>0,C==0` 时只执行 geometry | `calc_roi_align_tiling`、`BuildROIGeometry`、`roi_align_entry` |

### POD/ABI

`ROIAlignTilingData` 的 `sizeof=64, alignof=8`；`ROIAlignLaunchTiling` 的 `sizeof=72, alignof=8`。offset 为字节，均按标准 C++ ABI 顺序布局。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| ROIAlignTilingData | `batch` | int32/32 | off 0 | element-axis | `features.size(0)` 窄化 | 当前 body不读 | 调用约束用于 batch index |
| 同上 | `channels` | int32/32 | off 4 | element-axis | `features.size(1)` | feature stride、flat decode | 必须正且适合 uint32 divisor |
| 同上 | `height` | int32/32 | off 8 | element-axis | `features.size(2)` | feature stride、boundary | 有效调用需正 |
| 同上 | `width` | int32/32 | off 12 | element-axis | `features.size(3)` | feature stride、boundary | 有效调用需正 |
| 同上 | `numRois` | int32/32 | off 16 | ROI | `static_cast<int32_t>(rois.size(0))` | K-001 loop，并与 C/OH/OW 生成 O | 可为0；窄化后为负会在 Device 转 uint32，必须由 Host 调用域排除 |
| 同上 | `pooledHeight` | int32/32 | off 20 | element-axis | `outputHeight` | bin除数、flat decode | Host保证>0，窄化未审计 |
| 同上 | `pooledWidth` | int32/32 | off 24 | element-axis | `outputWidth` | bin除数、flat decode | 同上 |
| 同上 | `samplingRatio` | int32/32 | off 28 | samples/axis | `static_cast<int32_t>(samplingRatio)` | fixed/adaptive selector；正值还参与 `gridH*gridW` | Host 只在窄化前保证>=-1；还需 int32 可表示且乘积不溢出 |
| 同上 | `groupWidth` | int32/32 | off 32 | output/group | 固定1 | 当前 body不读 | inactive 保留字段 |
| 同上 | `tilesPerChannel` | int32/32 | off 36 | tile/channel | 固定1 | 当前 body不读 | inactive 保留字段 |
| 同上 | `spatialScale` | float/32 | off 40 | scale | `static_cast<float>(spatialScale)` | K-001 坐标缩放 | Host只检查>0 |
| 同上 | padding | 4 B | off 44 | byte | ABI 自动填充 | 不读 | 对齐后续 uint64 |
| 同上 | `totalOutputElements` | uint64/64 | off 48 | output element | `uint64(numRois)*channels*uint64(pooledHeight)*pooledWidth` | 32/64 decode selector | 所有 int32 维必须先合法且乘积无 uint64 回绕；Host 分核另需可转 int64 |
| 同上 | `totalWorkItems` | uint64/64 | off 56 | output element | 等于上一字段 | block分段/count | 当前实现一 work item=一 output |
| ROIAlignLaunchTiling | `data` | struct/512 | off 0 | POD | `calc_roi_align_tiling` | 两 entry 按值接收 | 结构必须字节一致 |
| 同上 | `blockDim` | uint32/32 | off 64 | AIV block | Host core公式 | K-002 launch grid | 至少1 |
| 同上 | tail padding | 4 B | off 68 | byte | ABI 自动填充 | 不读 | struct总长72 |

geometry 不是 POD，而是跨节点 GM 协议：每 ROI 分配 12 个 uint32 word，K-001 实际写 9 个。`0=batch`、`1=gridH`、`2=gridW`、`4=startW bits`、`5=startH bits`、`6=binH bits`、`7=binW bits`、`8=stepY bits`、`9=stepX bits`；word 3/10/11 未初始化也不读。这个稀疏布局不得压缩，除非 producer/consumer 的 stride 与 offset 一起原子修改。

### Route 源码闭包

| 重要性 | Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---:|---|---|---|---|---|---|---|
| P0 | R-003 | float/false | Direct plugin `roi_align_npu`；kernel TU `calc_roi_align_tiling` | `ROIAlignLaunchTiling`，`dtypeTag=FLOAT32`，`aligned=false` | `launch_roi_align`→`roi_align_geometry_entry<float,false>`→`roi_align_entry<float,false>` | `BuildROIGeometry<float,false>`、`ROIAlignForward<float,false>`、`EvaluateOutput<float,false,true>`、bilinear/rounding helpers | Direct op CMake `register_direct_launch_op`；kernel `-ffp-contract=off` |
| P1 | R-001 | half/false | Direct plugin `roi_align_npu`；kernel TU `calc_roi_align_tiling` | `ROIAlignLaunchTiling`，`dtypeTag=FLOAT16`，`aligned=false` | `launch_roi_align`→`roi_align_geometry_entry<half,false>`→`roi_align_entry<half,false>` | `BuildROIGeometry<half,false>`、`ROIAlignForward<half,false>`、`EvaluateOutput<half,false,true>`、bilinear/rounding helpers | Direct op CMake `register_direct_launch_op`；kernel `-ffp-contract=off` |
| P2 | R-002 | half/true | Direct plugin `roi_align_npu`；kernel TU `calc_roi_align_tiling` | `ROIAlignLaunchTiling`，`dtypeTag=FLOAT16`，`aligned=true` | `launch_roi_align`→`roi_align_geometry_entry<half,true>`→`roi_align_entry<half,true>` | `BuildROIGeometry<half,true>`、`ROIAlignForward<half,true>`、`EvaluateOutput<half,true,true>`、bilinear/rounding helpers | Direct op CMake `register_direct_launch_op`；kernel `-ffp-contract=off` |
| P3 | R-004 | float/true | Direct plugin `roi_align_npu`；kernel TU `calc_roi_align_tiling` | `ROIAlignLaunchTiling`，`dtypeTag=FLOAT32`，`aligned=true` | `launch_roi_align`→`roi_align_geometry_entry<float,true>`→`roi_align_entry<float,true>` | `BuildROIGeometry<float,true>`、`ROIAlignForward<float,true>`、`EvaluateOutput<float,true,true>`、bilinear/rounding helpers | Direct op CMake `register_direct_launch_op`；kernel `-ffp-contract=off` |

这里的 Direct 文件/符号是交付闭包身份，不是 Frontmatter 官方来源；固定官方 `roi_align` 文件仅提供相关接口与编排对照。

## Kernel 实现

### 公共执行流程

下列逐 Route 章节按复用价值 P0→P3 排列，R-ID 保持不变；这不是 Host 判断顺序，真实 dtype→aligned 嵌套短路见“Route 选择”。所有 Route 共享两节点关系：K-001 的 ROI owner lane 从 boxes GM 读 5 个标量，向 geometry 切片写 9 个 word；K-002 的 output owner lane 从该 ROI 切片读 9 个 word，最多从 feature GM 读 `4*gridH*gridW` 个标量，再只写 1 个 output。两节点均是 512-lane AIV-only SIMT，无显式 UB/shared/atomic/barrier；可见性依赖同 current stream 的 K-001→K-002 提交顺序。

### P0 / Route R-003：FP32 unaligned

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | Direct plugin | `roi_align_npu` | Float、`aligned=false` | `dtypeTag=FLOAT32` |
| Router | Direct kernel | `launch_roi_align` | dtypeTag else && !aligned | 两个 `<float,false>` entry |
| Geometry | Direct kernel | `BuildROIGeometry<float,false>` | `samplingRatio>0` / `<=0` | fixed/adaptive 局部分支；写 geometry |
| Forward | Direct kernel | `ROIAlignForward<float,false>` | `O<=UINT32_MAX` / `>UINT32_MAX` | magic-u32 / u64 flat decode；两者都调 `EvaluateOutput<...,true>` |

#### Device 数据流

##### 函数级调用链

```text
launch_roi_align
  -> roi_align_geometry_entry<float,false><<<1>>>
    -> BuildROIGeometry<float,false>(512 lanes) -> geometry GM
  -> same stream
  -> roi_align_entry<float,false><<<blockDim>>>
    -> GetUintDivMagicAndShift(W/H/C)
    -> ROIAlignForward<float,false>(512 lanes)
      -> EvaluateOutput -> BuildBilinearPoint -> SampleBilinear
      -> AccumulateSample -> FoldAccumulator -> CorrectedDivByPositiveInt
      -> FP32 output GM
```

##### 源码等价伪码

```text
# K-001
for r = lane; r<R; r+=512:
  batch = int(boxes[5r])
  endpoints = ffma(boxes[5r+1:5r+5], spatialScale, 0)
  roiW/H = max(end-start, 1)
  binW/H = corrected_div_positive_int(roiW/H, OW/OH)
  gridW/H = ratio>0 ? ratio : max(ceil(binW/H),1)
  stepX/Y = corrected_div_positive_int(binW/H, gridW/H)
  geometry[12r + {0,1,2,4..9}] = batch, grid, start, bin, step
# K-002
for local = lane; local<count; local+=512:
  outIndex=begin+local
  if O<=UINT32_MAX: magic_uint_div_decode(outIndex) -> (r,c,ph,pw)
  else: uint64_div_mod_decode(outIndex) -> (r,c,ph,pw)
  acc[0..7]=0
  for iy,ix in grid:
    sample = ordered_fp32_bilinear(features, geometry, ph,pw,iy,ix)
    acc[(iy*gridW+ix)&7] += sample
  # EXACT 固定为 true；乘法用 ffma(x,y,0)，加法用 fadd_rn
  raw = fold_left(acc0..acc7)
  output[outIndex] = corrected_div_positive_int(raw, uint32(gridH*gridW))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boxes GM | `5r` | `20R` B | FP32/ND | caller | K-001 | contiguous Tensor | 每 ROI 5 scalar loads |
| geometry GM | `12r` | 分配 `48R` B，每 ROI 写/读 36 B | uint32 words | K-001 | K-002 | invocation-local Tensor | 12-word stride，3 holes |
| features/output GM | NCHW flat | `4BCHW`/`4O` B | FP32 | caller/K-002 | K-002/caller | Tensor | 唯一输出写 |
| thread registers | endpoints、weights、4 values、8 accumulators、index magic | 固定标量 | FP32/u32/u64 | 当前 lane | 当前 lane | 单 output iteration | 无跨 lane 共享 |

##### 分层内存布局

N/A：无 AIV UB/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0; inter_kernel_GM=48R B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| geometry `[12r,12r+12)` | K-001 ROI owner lane | 属于 ROI r 的所有 K-002 output owner lanes | allocator 回收后 | same-current-stream K-001→K-002 | K-002 完成前 storage 不得复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入/输出 | FP32 GM scalar | caller/K-002 | interpolation/caller | values | 无 dtype cast |
| K-001→K-002 | current stream ordering | geometry store | K-002 geometry load | 48R B GM | 无显式 event/barrier |
| 乘法 | `__fmaf_rn(a,b,0)` | FP32 operands | geometry/weight/product | 单次 RN rounding | build禁用 contraction以保持加减边界 |
| 归约 | 8个有序 accumulator | sample values | 固定 lane0..7 fold | FP32 sum | sample编号 `&7`，不跨线程 |
| 除法 | power-of-two scale / native quotient+FMA residual / exact fallback | rawSum、正整数count | output store | IEEE RN ties-to-even | subnormal/大除数有 fallback |
| tail | loop bounds | R/O/count | lane return | output/geometry | `r<R`、`local<count`、`count==0 return` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Router | Direct kernel | `launch_roi_align` | FP32/false映射 |
| Geometry | Direct kernel | `BuildROIGeometry<float,false>` | FP32 unaligned geometry |
| Forward | Direct kernel | `ROIAlignForward<float,false>` | FP32最终store |

### P1 / Route R-001：FP16 unaligned

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | Direct plugin | `roi_align_npu` | dtype相等且为 Half、`aligned=false` | `dtypeTag=FLOAT16` |
| Router | Direct kernel | `launch_roi_align` | `dtypeTag==FLOAT16 && !aligned` | 两个 `<half,false>` entry |
| Geometry | Direct kernel | `BuildROIGeometry<half,false>` | `samplingRatio>0` 或 `<=0` | fixed/adaptive grid |
| Forward | Direct kernel | `ROIAlignForward<half,false>` | `O<=UINT32_MAX` / `>UINT32_MAX` | 分别解码后都调 `EvaluateOutput<half,false,true>` |

#### Device 数据流

##### 函数级调用链

```text
launch_roi_align
  -> roi_align_geometry_entry<half,false><<<1>>>
    -> BuildROIGeometry<half,false>(512 lanes) -> geometry GM
  -> same stream
  -> roi_align_entry<half,false><<<blockDim>>>
    -> GetUintDivMagicAndShift(W/H/C)
    -> ROIAlignForward<half,false>(512 lanes)
      -> EvaluateOutput -> BuildBilinearPoint -> SampleBilinear
      -> AccumulateSample -> FoldAccumulator -> CorrectedDivByPositiveInt
      -> cast half -> output GM
```

##### 源码等价伪码

```text
# K-001
for r = lane; r<R; r+=512:
  batch = int(float(boxes[5r]))
  endpoints = ffma(float(boxes[5r+1:5r+5]), spatialScale, 0)
  roiW/H = max(end-start, 1)
  binW/H = corrected_div_positive_int(roiW/H, OW/OH)
  gridW/H = ratio>0 ? ratio : max(ceil(binW/H),1)
  stepX/Y = corrected_div_positive_int(binW/H, gridW/H)
  geometry[12r + {0,1,2,4..9}] = batch, grid, start, bin, step
# K-002
for local = lane; local<count; local+=512:
  outIndex=begin+local
  if O<=UINT32_MAX: magic_uint_div_decode(outIndex) -> (r,c,ph,pw)
  else: uint64_div_mod_decode(outIndex) -> (r,c,ph,pw)
  acc[0..7]=0
  for iy,ix in grid:
    sample = ordered_fp32_bilinear(features, geometry, ph,pw,iy,ix)
    acc[(iy*gridW+ix)&7] += sample
  raw = fold_left(acc0..acc7)
  q = corrected_div_positive_int(raw, uint32(gridH*gridW))
  output[outIndex] = half(q)  # 唯一降精度点
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boxes GM | `5*r` | `10R` B | half/ND | caller | K-001 lane | contiguous Tensor | 每 ROI 读5元素 |
| geometry GM | `12*r` | 分配 `48R` B，每 ROI 写/读 36 B | uint32 word ABI | K-001 | K-002 | invocation-local Tensor | word 3/10/11 hole |
| features GM | `uint64(batch)*C*HW+c*HW+pointOffset` | `2BCHW` B | half/NCHW | caller | K-002 | input Tensor | 每有效 sample 4 个不规则读 |
| thread registers | endpoints、weights、4 values、8 accumulators、index magic | 固定标量 | FP32/u32/u64 | 当前 lane | 当前 lane | 单 output iteration | 无跨 lane 共享 |
| output GM | `outIndex` | `2O` B | half/NCHW | K-002 lane | caller | output Tensor | 唯一写 |

##### 分层内存布局

N/A：两个 SIMT 节点均不分配 AIV UB 或 dynamic UB；实际路径为 GM→线程私有标量→GM，中间 geometry 位于 GM。

##### 字节账与复用顺序

```text
UB_total(K-001)=0; UB_total(K-002)=0
dynamic_ub=N/A
workspace_total=0
inter_kernel_GM=alignof(uint32) naturally aligned, 48R bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| geometry `[12r,12r+12)` | K-001 ROI owner lane | 所有属于 ROI r 的 K-002 lanes | 调用结束后 allocator 回收 | 同 stream K-001→K-002 | K-002 完成前 Tensor storage 不回收 |
| `Accumulator8` | 当前 sample | `FoldAccumulator` | 下一 output iteration | 单线程程序序 | 每 lane独占 output |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| K-001→K-002 | current stream ordering | geometry store | K-002 geometry load | 48R B GM | 无显式 event/barrier |
| 乘法 | `__fmaf_rn(a,b,0)` | FP32 operands | geometry/weight/product | 单次 RN rounding | build禁用 contraction以保持加减边界 |
| 归约 | 8个有序 accumulator | sample values | 固定 lane0..7 fold | FP32 sum | sample编号 `&7`，不跨线程 |
| 除法 | power-of-two scale / native quotient+FMA residual / exact fallback | rawSum、正整数count | final cast | IEEE RN ties-to-even | subnormal/大除数有 fallback |
| tail | loop bounds | R/O/count | lane return | output/geometry | `r<R`、`local<count`、`count==0 return` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | Direct plugin/kernel | `roi_align_npu`、`calc_roi_align_tiling` | 校验、contiguous、POD、core公式 |
| Kernel router | Direct kernel | `launch_roi_align` | FP16/unaligned映射、双节点顺序 |
| Device body | Direct kernel | `BuildROIGeometry<half,false>`、`ROIAlignForward<half,false>` | geometry hoist、输出独占SIMT |
| Helper | Direct kernel | `BuildBilinearPoint`、`SampleBilinear`、`CorrectedDivByPositiveInt` | boundary、四点插值、舍入 |

### P2 / Route R-002：FP16 aligned

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | Direct plugin | `roi_align_npu` | Half、`aligned=true` | `dtypeTag=FLOAT16` |
| Router | Direct kernel | `launch_roi_align` | FP16 && aligned | 两个 `<half,true>` entry |
| Geometry | Direct kernel | `BuildROIGeometry<half,true>` | `samplingRatio>0` / `<=0` | 端点-0.5 后走 fixed/adaptive grid |
| Forward | Direct kernel | `ROIAlignForward<half,true>` | `O<=UINT32_MAX` / `>UINT32_MAX` | 两种 flat decode 都进入 `EvaluateOutput<half,true,true>` |

#### Device 数据流

##### 函数级调用链

```text
launch_roi_align(Half,true)
  -> geometry_entry<half,true><<<1>>>
    -> BuildROIGeometry<half,true>: boxes half→FP32，端点缩放后-0.5→geometry GM
  -> same current stream
  -> roi_align_entry<half,true><<<blockDim>>>
    -> ROIAlignForward<half,true>
      -> EvaluateOutput<half,true,true>
      -> BuildBilinearPoint/SampleBilinear→8路归约→corrected divide→half store
```

##### 源码等价伪码

```text
# K-001, T=half, ALIGNED=true
for r = lane; r < uint32(R); r += 512:
  batch = int32(float(boxes[5r]))
  x0,y0,x1,y1 = ffma(float(boxes[5r+1:5r+5]), scale, 0) - 0.5
  roiW = x1-x0; roiH = y1-y0                 # 不做 min-1 clamp
  binW = corrected_div(roiW, OW); binH = corrected_div(roiH, OH)
  if ratio > 0: gridW=gridH=ratio
  else: gridW=max(int32(ceil(binW)),1); gridH=max(int32(ceil(binH)),1)
  stepX=corrected_div(binW,gridW); stepY=corrected_div(binH,gridH)
  geometry[12r+{0,1,2,4..9}] = bitcast(batch,grid,start,bin,step)
# K-002
for local = lane; local < count; local += 512:
  out = begin+local; magic-u32-or-u64-decode(out) -> (r,c,ph,pw)
  load geometry[12r+{0,1,2,4..9}]
  for iy in [0,gridH), ix in [0,gridW):
    point = boundary_and_four_neighbors(start + bin*(ph,pw) + step*(iy+0.5,ix+0.5))
    acc[(iy*gridW+ix)&7] = fadd_rn(acc[...], bilinear_ffma_product(point))
  output[out] = half(corrected_div(fold_left(acc[0..7]), uint32(gridH*gridW)))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boxes GM | `5r` | `10R` B | half/ND | caller | K-001 | contiguous Tensor | 每 ROI 5 标量 |
| geometry GM | `12r` | 分配 `48R` B，每 ROI 有效 36 B | uint32 words | K-001 | K-002 | invocation-local Tensor | 12-word stride |
| features GM | `uint64(batch)*C*HW + c*HW + pointOffset` | `2BCHW` B | half/NCHW | caller | K-002 | contiguous Tensor | 每有效 sample 4 scalar loads |
| output GM | `outIndex` | `2O` B | half/NCHW | K-002 owner lane | caller | returned Tensor | 唯一写 |
| thread registers | 端点、bin/step、magic/index、4 weights/values、8 accumulators | 固定标量 | FP32/u32/u64 | lane | lane | 单 ROI 或单 output | 无跨 lane 共享 |

##### 分层内存布局

N/A：K-001 为 boxes GM→线程 FP32 标量→geometry GM；K-002 为 geometry/features GM→线程 FP32 标量→half output GM。无 AIV UB/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0; inter_kernel_GM=48R B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| geometry `[12r,12r+12)` | K-001 ROI owner lane | 属于 ROI r 的所有 K-002 output owner lanes | allocator 回收后 | same-current-stream K-001→K-002 | K-002 完成前 storage 不得复用 |
| `Accumulator8` | 当前 lane 的 sample loop | 当前 lane 的 fold/divide | 下一 output iteration | 单线程程序序 | 一 output 只有一 owner |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| K-001→K-002 | current stream ordering | 9-word geometry store | geometry loads | 48R B allocation | 无显式 event/barrier |
| alignment | compile-time `if constexpr(ALIGNED)` | scaled endpoints | bin geometry | geometry words | FP32 减0.5；不做 extent clamp |
| 插值/归约 | FFMA product + `__fadd_rn` + 8 accumulators | feature/weight samples | fold | 线程寄存器 | `EXACT=true`，无跨 lane 归约 |
| 除法/输出 | power-of-two / residual correction / integer-significand fallback | raw FP32 sum | half store | output scalar | 正整数除数；末端唯一 half cast |
| tail | `r<R`、`local<count`、`count==0` | block/lane split | return/store | geometry/output | K-001 与 K-002 分别守护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Router | Direct kernel | `launch_roi_align` | true映射 `<half,true>` |
| Body | Direct kernel | `BuildROIGeometry<half,true>` | 四端点-0.5、无min-1 |
| Forward | Direct kernel | `ROIAlignForward<half,true>`→`EvaluateOutput<half,true,true>` | magic-u32/u64 decode、FP32 插值与 8 路归约、corrected divide、末端 half store |

### P3 / Route R-004：FP32 aligned

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | Direct plugin | `roi_align_npu` | Float、`aligned=true` | `dtypeTag=FLOAT32` |
| Router | Direct kernel | `launch_roi_align` | dtypeTag else && aligned | 两个 `<float,true>` entry |
| Geometry | Direct kernel | `BuildROIGeometry<float,true>` | `samplingRatio>0` / `<=0` | 端点-0.5 后走 fixed/adaptive grid |
| Forward | Direct kernel | `ROIAlignForward<float,true>` | `O<=UINT32_MAX` / `>UINT32_MAX` | 两种 flat decode 都调 `EvaluateOutput<float,true,true>` |

#### Device 数据流

##### 函数级调用链

```text
launch_roi_align(Float,true)
  -> geometry_entry<float,true><<<1>>>
    -> BuildROIGeometry<float,true>: FP32 端点缩放后-0.5→geometry GM
  -> same current stream
  -> roi_align_entry<float,true><<<blockDim>>>
    -> ROIAlignForward<float,true>
      -> EvaluateOutput<float,true,true>
      -> BuildBilinearPoint/SampleBilinear→8路归约→corrected divide→FP32 store
```

##### 源码等价伪码

```text
# K-001, T=float, ALIGNED=true
for r = lane; r < uint32(R); r += 512:
  batch = int32(boxes[5r])
  x0,y0,x1,y1 = ffma(boxes[5r+1:5r+5], scale, 0) - 0.5
  roiW=x1-x0; roiH=y1-y0                       # 不做 min-1 clamp
  binW=corrected_div(roiW,OW); binH=corrected_div(roiH,OH)
  if ratio>0: gridW=gridH=ratio
  else: gridW=max(int32(ceil(binW)),1); gridH=max(int32(ceil(binH)),1)
  stepX=corrected_div(binW,gridW); stepY=corrected_div(binH,gridH)
  geometry[12r+{0,1,2,4..9}] = bitcast(batch,grid,start,bin,step)
# K-002
for local = lane; local<count; local+=512:
  out=begin+local; magic-u32-or-u64-decode(out) -> (r,c,ph,pw)
  load geometry slice; acc[0..7]=0
  for iy in [0,gridH), ix in [0,gridW):
    point=boundary_and_four_neighbors(start + bin*(ph,pw) + step*(iy+0.5,ix+0.5))
    acc[(iy*gridW+ix)&7] = fadd_rn(acc[...], bilinear_ffma_product(point))
  output[out] = corrected_div(fold_left(acc[0..7]), uint32(gridH*gridW))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boxes GM | `5r` | `20R` B | FP32/ND | caller | K-001 | contiguous Tensor | 每 ROI 5 scalar loads |
| geometry GM | `12r` | 分配 `48R` B，每 ROI 有效 36 B | uint32 words | K-001 | K-002 | invocation-local Tensor | 12-word stride |
| features GM | `uint64(batch)*C*HW+c*HW+pointOffset` | `4BCHW` B | FP32/NCHW | caller | K-002 | contiguous Tensor | 每有效 sample 4 scalar loads |
| output GM | `outIndex` | `4O` B | FP32/NCHW | K-002 owner lane | caller | returned Tensor | 唯一写，无 dtype cast |
| registers | 端点、bin/step、magic/index、4 weights/values、8 accumulators | 固定标量 | FP32/u32/u64 | lane | lane | 单 ROI 或单 output | 无共享 |

##### 分层内存布局

N/A：K-001 为 boxes GM→线程 FP32 标量→geometry GM；K-002 为 geometry/features GM→线程 FP32 标量→FP32 output GM。无 AIV UB/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0; inter_kernel_GM=48R B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| geometry `[12r,12r+12)` | K-001 ROI owner lane | 属于 ROI r 的 K-002 lanes | allocator 回收后 | same-current-stream K-001→K-002 | K-002 完成前 storage 不得复用 |
| `Accumulator8` | sample loop | fold/divide | 下一 output iteration | 单线程程序序 | 一 output 只有一 owner |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| K-001→K-002 | current stream ordering | 9-word geometry store | geometry loads | 48R B allocation | 无显式 event/barrier |
| alignment | compile-time true | scaled endpoints | bin geometry | geometry | FP32 减0.5，无 min-1 |
| 插值/归约 | FFMA product + `__fadd_rn` + 8 accumulators | feature/weight samples | fold | registers | `EXACT=true`，固定顺序 |
| 除法/输出 | power-of-two / residual correction / integer-significand fallback | raw FP32 sum | FP32 store | output scalar | 无末端降精度 |
| tail | `r<R`、`local<count`、`count==0` | block/lane split | return/store | geometry/output | K-001/K-002 各自守护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Router | Direct kernel | `launch_roi_align` | FP32/true映射 |
| Geometry | Direct kernel | `BuildROIGeometry<float,true>` | aligned geometry |
| Forward | Direct kernel | `ROIAlignForward<float,true>` | FP32输出 |

## 算子特定优化

### Trick T-001：直接消费五列 boxes，避免 V1 重建 ROI

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：调用接口已提供 `[R,5]` 的 `(batch,x0,y0,x1,y1)` 且 dtype 与 features 相同。
- 固定源码依据：[`objdetect/roi_align/op_host/op_api/aclnn_roi_align_v2.cpp:191-192`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align/op_host/op_api/aclnn_roi_align_v2.cpp)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
auto boxesContiguous = l0op::Contiguous(boxes, uniqueExecutor.get());
CHECK_RET(boxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

#### 机制与设计原因

官方 V2 Host 把连续化后的五列 `boxes` 直接交给 ROIAlign 核心；相对地，官方 V1 Host 会对独立 `batchIndices` 执行 Reshape、Cast，再与四列 rois 做 ConcatD。Direct 入口本来就采用 V2 形态，K-001 直接从 `boxes[5r:5r+5]` 读取，因而不物化 reshape/cast/concat 中间 Tensor；这减少 Host executor 节点与 GM 中间量。Direct 仍必须先 `contiguous()`，因为 Device 使用固定 `5*r+k` 地址公式。

#### 迁移边界

- 适用：上游能稳定提供 `[R,5]`、batch 列与坐标同 dtype 的 ROIAlign 接口。
- 前提：batch 值是可精确转为 int32 的合法整数；boxes contiguous 或在 launch 前显式物化。
- 失效：若 ABI 是四列 rois + 独立 INT batchIndices，必须保留转换/拼接，或重写 Device 为双输入；不能按五列越界读取。
- 证据边界：固定 forward 源码只直接支撑五列 boxes 的 contiguous 契约；Direct 固定地址公式是转换实现事实。

#### 错误做法

把非连续 boxes 的 `data_ptr` 直接交给固定 `5*r+k` Device 地址公式会忽略 stride；固定官方 V2 源码明确先做 `Contiguous`。固定仓未提供 Direct 950 body 的可验证反例。

### Trick T-002：两节点 geometry hoist 与输出标量独占

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：同一 ROI 的几何量被 `C*OH*OW` 个输出复用，且两个 Kernel 在同一 stream 顺序提交。
- 固定源码依据：[`objdetect/roi_align_grad/op_kernel/arch35/roi_align_grad_simt.h:64-68`](https://gitcode.com/cann/ops-cv/blob/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_align_grad/op_kernel/arch35/roi_align_grad_simt.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <uint32_t ROI_END_MODE>
__simt_callee__ inline bool ComputeRoiBin(int32_t n, int32_t numRois, int32_t batch, int32_t pooledHeight,
                                          int32_t pooledWidth, float spatialScale, int32_t sampleNum,
                                          __gm__ float* rois, RoiBinInfo& info)
```

#### 机制与设计原因

官方 ROIAlignGrad 950 的 `ComputeRoiBin` 证明 endpoints、bin 与 grid 是 ROI-only 几何，但 backward body 仍在自己的 gather 循环中计算它；该固定源码不包含 forward geometry-hoist。Direct 派生路线把 scaled endpoints、bin、grid、step 先物化为 `48R` B geometry，使 K-002 不再对每个 `(c,ph,pw)` 重算这些值；随后一 lane 独占一个输出，局部完成所有 sample 累加并只写一次 output，避免 atomic、跨 lane reduction 和 partial-output GM。代价是额外一次 Kernel launch 与 geometry GM 交接。

#### 迁移边界

- 适用：`C*OH*OW` 足以摊薄 geometry Kernel 与 `48R` B 中间量，且 ROI 几何与 channel 无关。
- 前提：geometry word ABI、同流可见性及 Tensor lifetime 全部保持；每 output 的采样循环可由单 lane承载。
- 失效：跨 stream 启动必须增加 event；若几何依赖 channel/output value，则不能按 ROI 缓存；极小工作量可能不值得多一次 launch。
- 证据边界：固定同族源码只支撑 ROI-only geometry 的可分离性；双节点 hoist、48R B 协议和 512-lane 选择是 Direct 派生实现事实。

#### 错误做法

只按 Host 书写顺序假设跨流可见，或提前释放 geometry，都会使 K-002 读取未完成/失效数据。固定仓没有 forward 950 body，未提供同机制的可验证官方反例。

## 模板与实现对比

| 对比维度 | P0 R-003 FP32 false | P1 R-001 FP16 false | P2 R-002 FP16 true | P3 R-004 FP32 true | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | Float、false | Half、false | Half、true | Float、true | 重要性与Host嵌套顺序分离；真实短路见Route表 |
| 模板与 dtype | `<float,false>` | `<half,false>` | `<half,true>` | `<float,true>` | dtype决定GM load/store与末端cast；bool消除几何运行时分支 |
| 循环与分核 | 双Kernel、512 lanes | 同左 | 同左 | 同左 | geometry按ROI；forward按output |
| Queue/TBuf/寄存器/dynamic UB | 仅线程寄存器 | 同左 | 同左 | 同左 | 无UB/shared/atomic |
| 内存总占用与复用 | geometry 48R B；FP32 I/O | geometry 48R B；FP16 I/O | 同左 | geometry 48R B；FP32 I/O | ROI-only geometry跨输出复用 |
| 搬运/SIMT API | GM scalar、`asc_vf_call` | 同左 | 同左 | 同左 | 不规则四邻点读适合输出独占SIMT |
| 同步、精度与 tail | FP32全程；min-1 | 同流；FP32中间→half；min-1 | 同流；FP32中间→half；-0.5 | FP32全程；-0.5 | 8路有序归约、corrected divide、grid-stride bounds |
| fallback/失败边界 | dtype else由Host封闭为Float | 无外部fallback | 同左 | dtype else由Host封闭为Float | 大O仅切换u64 flat decode，不切Route |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 固定官方 ACLNN V2 文档支持；本卡不覆盖 Direct | 固定官方 ACLNN V2 文档支持；本卡不覆盖 Direct | 固定官方 ACLNN V2 文档明确不支持；本卡仅覆盖交付 Direct 950 实现 |
| Host/Launcher | 官方 executor + TransData + L0 launcher | 同左 | Torch PrivateUse1 + Direct POD + 两次当前流 launch |
| Kernel/资源 | 固定来源未纳入本卡的官方 body | 同左 | AIV-only SIMT geometry→forward；geometry 48R B；无UB/workspace |

官方“不支持 950”与 Direct 950 交付不冲突：前者限定公开 ACLNN/API，后者是独立注册的 Direct Invoke 入口。不得据此把 `aclnnRoiAlignV2` 宣称为 950 支持。

## 限制与失败边界

- Direct Host 仅检查 rank/boxes列数/dtype/属性下界；batch index、ROI 坐标、同设备、正 `B/C/H/W`、有限 `spatialScale` 与 int32 窄化范围由调用方保证。
- `sampling_ratio=-1` 是 Direct 补充；固定官方 V2 只接受 `>=0`。Direct 中 `-1` 与 `0` 都走 `<=0` adaptive branch。
- `launch_roi_align` 的 dtype router 把任何非 `ROI_ALIGN_FLOAT16` 值当作 Float；现行 Host 只产生 0/1，所以这不是可达 fallback。绕过 Host 直接调 C ABI 必须保证 tag 与 GM 实际 dtype 一致。
- geometry 的 word 3/10/11 是未写空洞；consumer也不读。复用实现必须保持12-word stride，或同时原子地修改 producer/consumer ABI。
- `gridHeight*gridWidth` 先在 int32 中相乘再转 uint32；过大的 fixed 或 adaptive 网格可能发生有符号溢出。Host 未编码 sampling-ratio 上限，调用方必须保证两轴 grid 及乘积可表示且为正。
- output 总量从 `uint64_t(numRois)*channels*uint64_t(pooledHeight)*pooledWidth` 计算；shape/属性先窄化 int32，超范围调用可能在乘法前改变语义。分核还把 uint64 总量转 int64，并以 `(x+y-1)/y` 做两次均分和一次 32 元素对齐；有效域须同时满足上文三条有符号加法可表示前提，只有 `O<=INT64_MAX` 并不足够。
- K-001/K-002 launcher 返回 `void`，无逐节点错误码和 consumer 抑制；依赖框架异步错误机制。
- `RunOpApi` lambda 只捕获裸指针；若将两节点改到其它 stream，除了添加 event 还必须显式处理 `features/rois/geometry` 的 allocator stream 记录和生命期。
- 无 CPU、ACLNN、ATen compute 或外部算子 fallback；不合法 dtype 在 Host 已失败，大 output 仅回退到 uint64 解码。
- 固定官方 forward 版本没有 950 Host/Tiling/Kernel body；Direct 的 512-lane、精确算术和 geometry 协议是派生补充，不可宣称为官方 forward 实现。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `objdetect/roi_align/docs/aclnnRoiAlignV2.md` | `aclnnRoiAlignV2GetWorkspaceSize` | 官方V2 `[K,5] boxes`、dtype/shape/aligned语义与950不支持边界 |
| 入口声明/约束 | `objdetect/roi_align/op_graph/roi_align_proto.h` | `REG_OP(ROIAlign)` | ROI五列语义、属性与dtype |
| 官方Host实现 | `objdetect/roi_align/op_host/op_api/aclnn_roi_align_v2.cpp` | `CheckParams`、`aclnnRoiAlignV2GetWorkspaceSize` | V2校验、Contiguous、aligned映射、executor链 |
| 相关旧版实现 | `objdetect/roi_align/op_host/op_api/aclnn_roi_align.cpp` | `aclnnRoiAlignGetWorkspaceSize` | V1 Reshape/Cast/ConcatD 对照 |
| L0 launcher | `objdetect/roi_align/op_host/op_api/roi_align.cpp` | `ROIAlignV2`、`ADD_TO_LAUNCHER_LIST_AICORE` | 输出shape与核心属性顺序 |
| 构建接线 | `objdetect/roi_align/op_host/CMakeLists.txt` | `add_modules_sources` | 官方 ACLNN Host module接线；非Direct构建 |
| 950 同族 Tiling | `objdetect/roi_align_grad/op_host/arch35/roi_align_grad_tiling_arch35.cpp` | `CalcCoreDistribution`、`FillTilingData` | 1024 每核下界、32 元素对齐、AIV 核数与 `roiEndMode` key |
| 950 同族 POD | `objdetect/roi_align_grad/op_kernel/arch35/roi_align_grad_tiling_data.h` | `ROIAlignGradTilingData` | 近邻 backward 算子的 shape/geometry 字段；不是 Direct POD ABI |
| 950 同族 body | `objdetect/roi_align_grad/op_kernel/arch35/roi_align_grad_simt.h` | `ComputeRoiBin`、`ROIAlignGradSimt` | 256-lane AIV-only、五列 ROI、mode 0 与 half-pixel mode 2/3 geometry、fixed/adaptive sampling |
| 950 同族 entry | `objdetect/roi_align_grad/op_kernel/roi_align_grad_apt.cpp` | `roi_align_grad<schMode>` | `schMode` 到 SIMT 模板的编译期接线 |
| 950 同族构建 | `objdetect/roi_align_grad/CMakeLists.txt` | `add_all_modules_sources` | `COMPUTE_UNIT=ascend950`、`TILING_DIR=arch35` 的官方构建范围 |

每行文件均存在于 Frontmatter `sources`，且 URL 固定到 40 位 commit。Direct 转换文件不伪造公开 blob URL；本卡对 Direct 符号的描述是转换载荷闭包事实，不提升为官方原生事实。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 64 B POD、72 B launch struct、12-word geometry | 字段顺序/位宽/offset；geometry稀疏word位置 | Direct `ROIAlignTilingData`、`GeometryWord` |
| 调用顺序 | geometry→forward 两节点 | 同一 stream；K-001必须先于K-002 | Direct `launch_roi_align` |
| workspace/内存 | Device workspace=0；geometry=48R B | geometry owner/lifetime、12-word stride | Direct `roi_align_npu`、`BuildROIGeometry` |
| 同步 | 无barrier/atomic/event；每output唯一owner | 跨节点必须同流或补event | Direct `ROIAlignForward` |
| 精度与尾块 | FP32 intermediate、显式RN、8路fold、corrected integer division | 操作次序、末端cast、bounds guard | Direct arithmetic helpers、`EvaluateOutput` |
| 失败边界 | Host校验失败不launch；Device无外部fallback | 不得把dtype else开放给非FP32；不得忽略窄化/非法batch | Direct `roi_align_npu`、`launch_roi_align` |
