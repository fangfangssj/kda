---
type: Operator
title: "CircularPad2dBackward 950 ACLNN 算子知识"
description: "说明 aclnnCircularPad2dBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/docs/aclnnCircularPad2dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_api/padv3grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/pad_v3_grad_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_tilingkey.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/pad_v3_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_circular_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_circular_huge_width.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/pad_v3_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/config/ascend950/pad_v3_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:50:40Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# CircularPad2dBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCircularPad2dBackwardGetWorkspaceSize` / `aclnnCircularPad2dBackward`，公开两段式 ACLNN；950 的 circular 分支由 `l0op::PadV3Grad` 选择 RegBase `PadV3Grad`。 |
| 计算语义 | 对每个原输入坐标，直接枚举正文及各维左右回卷候选并求和；负 padding 对应裁剪并由边界过滤去除无效候选。 |
| 输入输出 | `gradOutput/self/gradInput` 同 dtype，FP16/BF16/FP32，ND，rank 3 或 4；`padding=[left,right,top,bottom]` INT64；输出 shape 与 `self` 相同。 |
| 执行类型 | 单 AIV Kernel；R-001/R-002 为节点内 SIMT，R-003 为 AIV Vector MicroAPI。无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 SIMT-32；R-002 SIMT-64（超 INT32 地址域）；R-003 SIMD huge-width。 |
| 核心资源 | SIMT 的小型静态 UB 元数据与每线程寄存器候选；SIMD 的双分区 `TBuf`、RegTensor、MTE/V event；16 MiB 系统 workspace 由 Host 声明但 circular body 不读。 |
| 来源闭包 | `source_closed`：ACLNN → L0 RegBase selector → `PadV3Grad` infer/Tiling/POD → template key → SIMT/SIMD entry/body → ascend950 config/CMake 完整闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `G=gradOutput` 的 shape 为 `I`，目标 `gradInput` shape 为 `O`，扩展后每轴 padding 为 `(pL[d],pR[d])`，满足 `O[d]=I[d]-pL[d]-pR[d]`。对输出线性坐标解出的 `o[d]`，正文候选输入坐标是 `i[d]=o[d]+pL[d]`；若 `o[d]>=O[d]-pL[d]`，再加入 `i[d]-O[d]`；若 `o[d]<pR[d]`，再加入 `i[d]+O[d]`。跨维取候选笛卡尔积，过滤落在输入域外的项后求和。

```text
gradInput[o] = Σ gradOutput[c]
c[d] ∈ {o[d]+pL[d], o[d]+pL[d]-O[d], o[d]+pL[d]+O[d]}
且 0 <= c[d] < I[d]。
```

正 padding 产生 circular 回卷候选；负 padding 会让正文坐标偏移，`cutBounds`/地址边界检查丢弃裁剪域之外的候选。每个输出元素只由一个 SIMT thread 或一个 SIMD tile owner 写，因此无需 GM atomic。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | rank 3/4，ND；前向 padded shape | FP16/BF16/FP32 | 与 `self/gradInput` dtype、view format 相同；非连续时 `Contiguous`。最后一维公开约束 FP16/BF16 `(0,16362)`、FP32 `(0,24544)`。 |
| `self` | rank 3/4，ND | 同上 | 只提供前向原输入 shape；与 `gradInput` shape 相同。 |
| `padding` | ACLNN 长度 4；L0 Tensor 长度 `2*rank` | ACLNN INT64 | API 变换为前导零对加 `[top,bottom]`、`[left,right]`，并令 `paddings_contiguous=true`。 |
| `gradInput` | 与 `self` 同 shape/layout | 同上 | 连续核心输出经 `ViewCopy` 写回，支持非连续目标。 |

### 入口与版本

固定源码只有无版本后缀的公开 2D backward 接口。`commonPad2dBackward` 固定传 `mode="circular"`。`l0op::PadV3Grad` 的 circular 分支先检查 `IsRegBase()`：950 为 true，故加入 `PadV3Grad`，不进入旧 `CircularPadGrad`。因此 `CircularPadGrad/README.md` 对 950 标不支持与 ACLNN 文档对 950 标支持不冲突：二者描述的是不同底层节点，950 的真实公开链路由 `PadV3Grad` 的 ascend950 OpDef/config/CMake 支撑。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 层依次完成空指针、dtype、view format、rank、padding 长度和 shape 等式校验；合法空 Tensor 按 API 规则直接返回零 workspace/executor。非空时：

1. `gradOutput` 与 `self` 经 `l0op::Contiguous`；
2. `GetPaddingTensor` 将 4 个值逆向扩成 rank 对齐的连续 pair Tensor；
3. `l0op::PadV3Grad` 先按 `x[d]-padFront[d]-padBack[d]` 推导连续输出；
4. 950 circular selector 加入 `PadV3Grad` AICore 节点；
5. `ViewCopy` 将其输出写入用户 `gradInput`。

Tiling 再读取 input shape/dtype、mode 和 padding Tensor；非 constant 模式 rank 上限为 5，而本 API 输入 rank 3/4。`DimensionCollapseMode` 消除 shape=1 且无净 padding 的轴，并把连续无 padding 轴折叠；随后计算 `outShape/inStride/outStride`，同时验证 circular 的每侧正 padding 不大于推导后的输出轴长。

### 参数、Workspace 与生命周期

`Init` 从 950 平台取得 `GetCoreNumAiv()`、UB block、VReg 和 cache line；`DoTiling` 最终 `SetBlockDim(coreNum_)`。SIMD Route 的调优可能将 `coreNum_` 缩为实际使用核数；SIMT Route 使用全部 AIV 核。

Host 对 `PadV3Grad` 固定声明：

```text
workspaces[0] = 16 * 1024 * 1024 bytes
```

entry 保留 `workspace` 参数，但 `PadV3GradSimt`、`PadV3GradCircular` 以及两类 circular body 都不解引用它；该 16 MiB 属系统 workspace 声明，不是跨 Kernel 中间结果。ACLNN executor 还可能为 `Contiguous/ViewCopy` 分配框架空间，其切片固定仓不可见。POD 和连续中间 Tensor 的 owner 是 executor，生命周期覆盖同一 stream 上的核心节点与 ViewCopy。

### 启动图与失败传播

```text
可选 Contiguous(gradOutput/self)
  -> PadV3Grad(x=gradOutput, paddings, y=continuousResult,
               attrs={circular,true}, workspace=system 16MiB, POD)
       -> 节点内 SIMT body 或 AIV Vector MicroAPI body
  -> ViewCopy(continuousResult, gradInput)
```

核心只有一个 Device 节点：Host launcher 为 `PadV3GradAiCore`，entry 为模板化 `pad_v3_grad<modeName,isBigShape,isSimt,cutMode>`，`blockDim=coreNum_`，输入/输出均为 ND。executor 在同一传入 stream 上保持节点顺序；infer、Tiling 或 launcher 失败会阻断结果/后续 ViewCopy。框架级 event 和清理细节固定仓不可见。

## Route 与参数设计

### 分核与资源计算

共同计算：

```text
inShapeSize  = Π inShape[d]
outShapeSize = Π outShape[d]
outStride[d] = Π outShape[k], k>d
inStride[d]  = Π inShape[k],  k>d
```

对全非负 padding，Host 先尝试 SIMD：若 `inShapeSize<=48Ki elements`，转 SIMT；否则令 `alignNum=blockSize/dtypeBytes`、`lastShapeSizeAlign=align(outShape[last],alignNum)`，并计算：

```text
bufferSize = align((ubSize / (2*dtypeBytes + 2*4)) - alignNum, alignNum)
bufferSize = min(bufferSize, 64KiB/dtypeBytes)
```

仅 `lastShapeSizeAlign>bufferSize` 时保留 SIMD huge-width：`ubAxis=last`、`ubFactor=outTileSize=bufferSize`，再由 `TilingInfoTune` 计算 `ubTotalCount`、`ubPerCount` 和实际核数。其他情况以及含任一负 padding 的情况均走 SIMT。SIMT 用 `inShapeSize/outShapeSize > INT32_MAX` 选择 64-bit 地址模板。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-003 SIMD huge-width | tuple `(Mode=4, IsSIMT=0, CutMode=0)` | 非空、所有 padding `>=0`、`inShapeSize>49152`、且 `align(outShape[last],32/dtypeBytes)>bufferSize` | 任一条件不成立转 SIMT |
| 2 | R-002 SIMT-64 | tuple `(Mode=4, IsBigShape=1, IsSIMT=1, CutMode=任意)` | 未命中 R-003，且 `inShapeSize>INT32_MAX \|\| outShapeSize>INT32_MAX` | 否则 R-001 |
| 3 | R-001 SIMT-32 | tuple `(Mode=4, IsBigShape=0, IsSIMT=1, CutMode=任意)` | 未命中 R-003，且两端总元素均 `<=INT32_MAX`；含空域时也由 body 早退 | 无 fallback |

```text
mode=circular && IsRegBase
  -> PadV3Grad::DoTilingModeCircular
     -> collapse + strides
     -> empty                       -> SIMT
     -> any negative/mixed padding -> SIMT
     -> all nonnegative
          -> inSize<=49152                         -> SIMT
          -> aligned_last_output<=bufferSize       -> SIMT
          -> otherwise                             -> SIMD huge-width
     -> SIMT: bigShape = (inSize>INT32_MAX || outSize>INT32_MAX)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 SIMT-32 | Mode4/Big0/SIMT1 | 950；FP16/BF16/FP32；SIMD 条件不成立；32-bit 元素域 | 单 AIV entry 内 `asc_vf_call(dim3(256))` | `PadV3GradCircularSimt<T>::Process<int32_t>` → dim 1..4 specialization | 288B 静态 UB 元数据；线程私有候选/累加；无 atomic | grid-stride `idx<outputSize`；空域返回 |
| R-002 SIMT-64 | Mode4/Big1/SIMT1 | 同上但任一总元素 > INT32_MAX | 同 R-001 | `Process<int64_t>` | 576B 静态 UB 元数据；64-bit index/magic | 同上 |
| R-003 SIMD huge-width | Mode4/SIMT0/Cut0 | 全非负、非空、大输入、尾轴对齐宽大于 buffer | 单 AIV Vector body | `KernelPadV3GradCircularHugeWidth<T>` | 输入 TBuf 双分区 + FP32/原 dtype 结果 TBuf 双分区；MTE/V event | `dataLen=min(ubFactor, axisRemain)`；核外无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| circular pad 合法性 | 每侧正 pad `<=outShape[d]` | 每轴候选固定最多正文+左右各一份 | 全部 | Tiling 失败 | `CheckModeInputParam` |
| SIMT 小输入阈值 | `inShapeSize<=48*1024` | 小规模直接逐输出枚举，避免 SIMD tile 建立开销 | R-001/R-002 | 命中 SIMT | `CalculateTilingKeyCircular` |
| SIMD 尾宽 | `lastShapeSizeAlign>bufferSize` | huge-width body 切最后轴；较窄尾轴回到 SIMT | R-003 | SIMT | `CalculateTilingKeyCircular` |
| SIMD UB cap | `bufferSize<=64KiB/dtypeBytes` | 两套双分区 TBuf 且每 buffer 有 64KiB 上限 | R-003 | 缩小 factor 后调优 | 同上 |
| SIMD 核利用率 | 使用核比例 `>=0.8` 或 tile 输入字节 `<=4096`，否则调整 axis/factor | 避免低核利用率，同时阻止过小 tile 继续搜索 | R-003 | 调整 `ubAxis/ubFactor` | `CheckTilingInfoSatisfied` / `GetOptimizeTiling` |
| SIMT 地址宽 | `inSize>INT32_MAX \|\| outSize>INT32_MAX` | 线性 offset 和 magic division 必须避免 32-bit 溢出 | R-001/R-002 | 切到 64-bit 模板 | `DoTilingWithSIMTCircular` |

### POD/ABI

`PadV3GradACTilingData` 是显式标准布局字段序列；按自然 8 字节最大对齐推得 `sizeof=408`、`alignof=8`。该结论来自字段类型与声明顺序，固定仓未提供单独 static_assert。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `PadV3GradACTilingData` | `dimNum` | uint8/8 | offset 0 | 轴数 | `FillsAndPrintTilingData` | SIMT specialization/循环，SIMD 外轴循环 | 本 API collapse 后 1..4 |
| 同上 | `ubAxis` | uint8/8 | offset 1 | 轴索引 | 同上 | SIMD tile 解码 | SIMT inactive |
| 同上 | `ubFactor` | uint32/32 | offset 4 | 元素 | 同上 | SIMD 每 tile 轴长 | SIMT inactive |
| 同上 | `ubPerCount` | uint32/32 | offset 8 | tile/核 | 同上 | SIMD `startIdx/endIdx` | SIMT inactive |
| 同上 | `ubTotalCount` | uint32/32 | offset 12 | tile | 同上 | SIMD 核早退/上界 | SIMT inactive |
| 同上 | `outTileSize` | uint32/32 | offset 16 | 元素/分区 | 同上 | 两个 TBuf 分区 stride | SIMT inactive |
| 同上 | `additionTileSize` | uint32/32 | offset 20 | 元素 | 同上 | circular 两 body 不读取 | inactive |
| 同上 | `inShape[8]` | uint64[8]/512 | offset 24 | 元素/轴 | `GetShapesAndDtypes`、collapse | SIMT cutBounds，SIMD 地址 | 只读前 `dimNum` |
| 同上 | `outShape[8]` | uint64[8]/512 | offset 88 | 元素/轴 | `ComputeAfterPaddingsAndStrides` | 输出解码、候选、tile tail | 同上 |
| 同上 | `inStride[8]` | uint64[8]/512 | offset 152 | 元素 stride | 同上 | 输入地址组合 | 同上 |
| 同上 | `outStride[8]` | uint64[8]/512 | offset 216 | 元素 stride | 同上 | 输出线性坐标解码 | 同上 |
| 同上 | `leftPad[8]` | int64[8]/512 | offset 280 | 元素/轴 | `DimensionCollapseMode` | 正文偏移与左回卷候选 | 可正/零/负 |
| 同上 | `rightPad[8]` | int64[8]/512 | offset 344 | 元素/轴 | 同上 | 右回卷候选 | 可正/零/负；总大小 408 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 SIMT-32 | `(4,0,1,*)` | `PadV3GradACTiling::DoTilingModeCircular` | `PadV3GradACTilingData` | `pad_v3_grad` → `LaunchCircularKernelSIMT` | `PadV3GradCircularSimt::Process<int32_t>`、dim 1..4 callee | `PadV3Grad` ascend950 OpDef/binary/CMake |
| R-002 SIMT-64 | `(4,1,1,*)` | 同上 | 同上 | 同上 | `Process<int64_t>` | 同上 |
| R-003 SIMD huge-width | `(4,*,0,0)` | `CalculateTilingKeyCircular` / `TilingInfoTune` | 同上 | `pad_v3_grad` → `LaunchKernelPadV3GradCircularHugeWidth` | `KernelPadV3GradCircularHugeWidth` 全部 methods | 同上 |

## Kernel 实现

### 公共执行流程

`pad_v3_grad` 显式声明 `KERNEL_TYPE_AIV_ONLY`，注册 POD 后按编译期 tuple 分派。SIMT 路线不构造 `TPipe`，以 `asc_vf_call` 在每个 AIV 上发起 256-thread SIMT；SIMD 路线构造 `TPipe` 和 huge-width 类。两路线都采用“一个 owner 完成一个输出”的 gather-and-sum，而非多个输入原子散射到输出。FP16/BF16 都在 FP32 `CastType` 中累加，最终 RNE 饱和转换。

### Route R-001：SIMT-32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/L0 | `aclnn_circular_pad2d_backward.cpp` / `padv3grad.cpp` | `commonPad2dBackward` / `PadV3Grad` | circular 且 `IsRegBase()` | `PadV3GradAiCore` |
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingModeCircular` | 未保留 SIMD，且两端 size `<=INT32_MAX` | `isSimt=true,isBigShape=false` |
| Router | `pad_v3_grad_apt.cpp` | `pad_v3_grad<4,false,true,*>` | `isSimt` | `LaunchCircularKernelSIMT` |
| Body | `pad_v3_grad_simt.h` | `LaunchCircularKernelSIMT` | `isBigShape=false` | `Process<int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
PadV3GradAiCore -> pad_v3_grad<4,false,true,*>
  -> PadV3GradSimt<4,false>
  -> LaunchCircularKernelSIMT<T,false>
  -> PadV3GradCircularSimt<T>::Init
  -> Process<int32_t>
     -> asc_vf_call<SimtComputeCircular{One..Four}>(threads=256)
        -> CalPos -> CalCandidate -> nested candidate sum -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；每 AIV 上 256 threads
for idx = blockIdx*256 + threadIdx; idx < outputSize; idx += blockNum*256:
  outIndex = fast_divmod(idx, outStride, magic, shift)
  inIndex[d] = outIndex[d] + leftPad[d]
  candidates[d] = {inIndex[d]}
  if outIndex[d] >= outShape[d]-leftPad[d]: append inIndex[d]-outShape[d]
  if outIndex[d] < rightPad[d]:             append inIndex[d]+outShape[d]
  total = 0
  for cartesian product c of candidates[0..dimNum-1]:
    if each strided partial offset is within cutBounds: total += cast_fp32(x[c])
  y[idx] = rne_sat_cast<T>(total)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | 候选 strided offset 和 | `inShapeSize` | T/ND | caller | 当前 thread | caller | 标量读；边界由 `cutBounds` |
| y GM | `idx` | `outputSize` | T/ND | 当前 thread | ViewCopy | executor output | 每 idx 单 owner，无 atomic |
| SIMT UB 元数据 | 静态数组：2 个 magic/shift + 7 个 shape/stride/pad/bound | `9*8*4=288B` | uint32/int32 | `Process` 从 POD 填充 | 256 threads | 当前 AIV 调用 | 无 dynamic UB |
| thread locals | `outIndex/inIndex/IdxAndTimes/total` | 编译器寄存器分配固定仓不可见 | U/CastType | thread | 同 thread | 单迭代 | 不跨 thread |
| workspace | 未寻址 | 0 body bytes | N/A | N/A | N/A | Host 系统预留 | 无复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  magic["magics+shifts · 64 B"]:1
  meta["shape/stride/pad/bounds · 224 B"]:2
```

##### 字节账与复用顺序

```text
static_ub = (2 + 7) * 8 * sizeof(uint32) = 288 B
dynamic_ub = N/A
workspace_body = 0 B  # Host 仍声明 16 MiB 系统 workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| UB 元数据 | `Process` 初始化循环 | 所有 SIMT threads | kernel 节点结束 | `asc_vf_call` 参数可见性 | 调用期间只读 |
| thread locals | 当前 idx | 当前候选循环 | grid-stride 下一 idx | 单 thread 顺序 | 不共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元数据→SIMT | `asc_vf_call` 调用边界 | AIV 标量初始化 | SIMT callee | UB arrays | 无额外 barrier 可见于固定 body |
| 候选累加 | thread 私有顺序 | GM scalar loads | 同 thread `total` | 单输出 | FP16/BF16 转 FP32；固定嵌套轴顺序 |
| 输出 | 直接 store | thread total | y GM | 单元素 | BF16 `__float2bfloat16_rn_sat`；FP16 `__float2half_rn_sat`；`idx<outputSize` 保护尾部 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTCircular` | 32/64-bit 选择 |
| Kernel entry/router | `pad_v3_grad_apt.cpp` | `pad_v3_grad<4,false,true,*>` | AIV-only 与 SIMT 分派 |
| Device body/helper | `pad_v3_grad_circular_simt.h` | `Process<int32_t>` / `SimtComputeCircular*` | 256 threads、候选枚举、精度与尾界 |

### Route R-002：SIMT-64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTCircular` | `inShapeSize>INT32_MAX \|\| outShapeSize>INT32_MAX` | `isBigShape=true` |
| Router | `pad_v3_grad_simt.h` | `LaunchCircularKernelSIMT<T,true>` | compile-time true | `Process<int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
pad_v3_grad<4,true,true,*>
  -> LaunchCircularKernelSIMT<T,true>
  -> PadV3GradCircularSimt<T>::Process<int64_t>
  -> SimtComputeCircular{One..Four}<U=int64,GmOffsetType=uint64>
```

##### 源码等价伪码

```text
# 与 R-001 完全相同的候选与 grid-stride 算法，
# 但 shape/stride/pad/cutBounds、linear idx、magic/shift 均为 64-bit。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | 同 R-001，64-bit offset | 输入/输出全域 | T/ND | caller/thread | thread/ViewCopy | 同 R-001 | 防止大元素域截断 |
| SIMT UB 元数据 | 同 9 个长度 8 数组 | `9*8*8=576B` | uint64/int64 | `Process` | SIMT threads | 当前 AIV | 无 dynamic UB |
| thread locals | 同 R-001 | 寄存器占用固定仓不可见 | 64-bit index + CastType | thread | thread | 单迭代 | 无共享 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  magic["magics+shifts · 128 B"]:1
  meta["shape/stride/pad/bounds · 448 B"]:2
```

##### 字节账与复用顺序

```text
static_ub = (2 + 7) * 8 * sizeof(uint64) = 576 B
dynamic_ub = N/A
workspace_body = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| UB 元数据/thread locals | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 64-bit 类型必须贯穿所有地址计算 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 仅地址宽度改变；FP32 累加和 RNE 输出不变 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTCircular` | INT32_MAX guard |
| Kernel entry/router | `pad_v3_grad_simt.h` | `LaunchCircularKernelSIMT<T,true>` | 64-bit specialization |
| Device body/helper | `pad_v3_grad_circular_simt.h` | `Process<int64_t>` | 64-bit magic division与地址 |

### Route R-003：SIMD huge-width

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingModeCircular` | 非空且 `isPadAllPositive_` | 尝试 SIMD |
| Tiling | 同上 | `CalculateTilingKeyCircular` | `inShapeSize>49152 && lastShapeSizeAlign>bufferSize` | `isSimt=false,cutMode=BIG` |
| Router | `pad_v3_grad_apt.cpp` | `PadV3GradCircular<4,0>` | Mode4/SIMT0/Cut0 | huge-width launcher |
| Body | `pad_v3_grad_simd.h` | `LaunchKernelPadV3GradCircularHugeWidth` | dtype binary | `KernelPadV3GradCircularHugeWidth<T>` |

#### Device 数据流

##### 函数级调用链

```text
pad_v3_grad<4,*,false,0>
  -> PadV3GradCircular<4,0>
  -> LaunchKernelPadV3GradCircularHugeWidth<T>
  -> KernelPadV3GradCircularHugeWidth::Init
  -> Process
     -> ProcessOneStep
        -> Duplicate(result,0)
        -> DataCopyPad(center) -> ProcessSelfAdd
        -> CalculateOffsetParams
        -> CopyAndCal
           -> ProcessMiddleData / ProcessLeftData / ProcessRightData
        -> CopyOutputToGM
```

##### 源码等价伪码

```text
# 伪码；tile 双分区由 idx parity 选择
for tileIdx in [blockIdx*ubPerCount, min(...,ubTotalCount)):
  decode outer output indices and last-axis tile start
  dataLen = min(ubFactor, outShape[last]-tileStart)
  res[parity] = 0
  DMA正文 tile -> src[parity]; vector cast/add -> res
  build up to 3 candidate offsets for each outer axis
  for candidate product over outer axes:
    if not all-center: DMA corresponding middle tile; vector add
    if tile intersects left-wrap interval:  DMA left slice; unaligned vector add at leftUbStartIdx
    if tile intersects right-wrap interval: DMA right slice; aligned vector add
  cast FP32 result -> T with CAST_RINT when needed
  DMA valid dataLen -> y[outSelfAddr]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `Σ outerCandidate[d]*inStride[d] + innerOffset` | 每次 `dataLen`、leftLen 或 rightLen | T/ND | caller | MTE2 | caller | `DataCopyPad`, blockCount=1 |
| y GM | `Σ outIndex[d]*outStride[d]` | `dataLen` | T/ND | MTE3 | ViewCopy | executor result | 尾 tile 只写有效元素 |
| `inQueue_` | `(tileIdxRelative&1)*outTileSize` | `2*outTileSize*sizeof(T)` | T | MTE2 | Reg Load/Cast | `TPipe` | TBuf 分成 parity 双区 |
| `resQueue_` | 同 parity 公式 | `2*outTileSize*sizeof(CalType)` | FP32 for FP16/BF16；FP32 for FP32 | Duplicate/Reg Add | Cast/MTE3 | `TPipe` | 双区 |
| RegTensor | repeat offset `k*VRegElems` | `ceil(dataLen/VRegElems)` repeats | T/CalType | MicroAPI Load | Add/Store | VEC scope | mask 保护尾 repeat |
| workspace | 未寻址 | 0 body bytes | N/A | N/A | N/A | Host 系统预留 | 无复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["inQueue tile0 · outTile×sizeof(T)"]:1
  in1["inQueue tile1 · outTile×sizeof(T)"]:1
  res0["res tile0 · outTile×sizeof(CalType)"]:1
  res1["res tile1 · outTile×sizeof(CalType)"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*outTileSize*sizeof(T) + 2*outTileSize*sizeof(CalType)
FP16/BF16: UB_total = 12*outTileSize bytes
FP32:      UB_total = 16*outTileSize bytes
UB_total <= Host ubSize（bufferSize 公式并受 64KiB/单buffer cap）
dynamic_ub = N/A
workspace_body = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| src parity 0/1 | 当前 tile MTE2 | VEC Add/Cast | 相隔 2 tile | `MTE3_MTE2` 等待 + `V_MTE2` event | `idx>1` 才等待旧 parity 输出完成 |
| res parity 0/1 | Duplicate/各候选 Add | Cast 或 MTE3 | 相隔 2 tile | FP32 用 `MTE3_V`；16-bit 用 `MTE3_MTE2` | 输出搬完前不得清零 |
| RegTensor | 每 repeat Load | Add/Store | 下一 repeat | VEC 顺序/mask | tail mask 仅覆盖 `dataLen` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→V | `Set/WaitEvent<MTE2_V>` | 每次 `DataCopyPad` | `Process*Add` | src parity | blockLen 为当前有效 slice |
| V→MTE2 | `Set/WaitEvent<V_MTE2>` | Reg Add | 下一候选 DMA 复用 src | src/result dependency | 候选顺序确定 |
| V→MTE3 | `Set/WaitEvent<V_MTE3>` | 最终 Add/Cast | `DataCopyPad` 输出 | res/src parity | 16-bit `Cast(...CAST_RINT)` |
| parity 复用 | `MTE3_MTE2` 或 `MTE3_V` | 前两个 tile 的输出 | 当前 tile 清零/搬入 | 双区 | `idx>1` guard |
| Vector tail | `UpdateMask` / `dataLen` | 最后 repeat | Reg Store/MTE3 | 有效 lane | left 使用 unaligned load/store；right/center aligned |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | `CalculateTilingKeyCircular` / `TilingInfoTune` | UB factor、核数、huge-width guard |
| Kernel entry/router | `pad_v3_grad_apt.cpp` / `pad_v3_grad_simd.h` | `PadV3GradCircular` / huge-width launcher | tuple 到 body |
| Device body/helper | `pad_v3_grad_circular_huge_width.h` | `ProcessOneStep`、`CopyAndCal`、`Process*Data` | 双区、地址候选、MicroAPI、event、tail |

## 算子特定优化

### Trick T-001：以输出为 owner 的 gather 求和消除 GM atomic

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：所有 circular SIMT 场景；每个 `idx` 由唯一 grid-stride thread 处理。
- Golden 来源：[`conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_circular_simt.h:121-151`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_circular_simt.h#L121-L151)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
     idx += blockNum * blockDim.x) {
    U outIndex[DIM_NUM]{0};
    U inIndex[DIM_NUM]{0};
    GmOffsetType yIdx = idx;
    CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);
```

#### 机制与设计原因

反向也可从每个 `gradOutput` 元素散射并 atomic add，但本实现反过来给每个 `gradInput` 元素一个唯一 owner，枚举其最多三候选/轴并私有累加。这样 GM 输出只有一次普通 store，确定性来自固定候选嵌套顺序；magic/shift 避免每个 thread 对 stride 做普通整数除法。

#### 迁移边界

- 适用：逆映射候选数有严格小上界且输出元素彼此独立的梯度聚合。
- 前提：circular 每侧 pad 不超过输出轴长，候选数最多 3/轴；rank 在 body specialization 范围内。
- 失效：候选爆炸或映射无法局部枚举时需要其他归约设计；固定仓无 atomic fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：仅对真正超宽尾轴启用双分区 SIMD

#### 触发与源码依据

- 适用 Route：R-003；不满足时回 R-001/R-002。
- 触发条件：全非负、输入元素数大于 48 Ki、对齐尾轴宽大于按 UB 推得的 `bufferSize`。
- Golden 来源：[`conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.cpp:334-356`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.cpp#L334-L356)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (inShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > CONST5) {
    DoTilingWithSIMTCircular();
    return;
}
uint64_t alignNum = blockSize_ / dtypeBytes_;
uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1], alignNum);
bufferSize_ = GetSizeOfBlockAlign(ubSize_ / (CONST2 * dtypeBytes_ + CONST2 * FP32_SIZE) - alignNum, alignNum);
if (lastShapeSizeAlign > bufferSize_) {
    cutMode_ = TPL_SIMD_BIG;
    ubAxis_ = dimNum_ - 1;
    ubFactor_ = bufferSize_;
    outTileSize_ = bufferSize_;
    return TilingInfoTune();
}
DoTilingWithSIMTCircular();
```

#### 机制与设计原因

Host 只在尾轴确实需要切块时支付 DMA、双分区和 event 管线成本；窄尾轴或小总量直接逐元素 SIMT。huge-width body 用 parity 双区允许当前 tile 计算与相邻 tile 的 MTE 生命周期交错，并将 16-bit 输入只在 UB 中升精度。

#### 迁移边界

- 适用：最后轴连续且超过单 buffer 的 gather-sum。
- 前提：padding 非负、UB/VReg/blockSize 均可查询，单 buffer 不超过 64 KiB。
- 失效：负/mixed padding、小输入或窄尾轴回 SIMT；不是错误终点。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 SIMT-32 | R-002 SIMT-64 | R-003 SIMD huge-width | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 非 SIMD且 32-bit 域 | 非 SIMD且超 INT32 | 全非负且真正超宽 | 先 SIMD 资源判定，再地址宽度 |
| 模板与 dtype | U=int32，T 三 dtype | U=int64，T 三 dtype | CalType=FP32/FP32 | 16-bit 均 FP32 累加 |
| 循环与分核 | 256-thread grid-stride | 同左，64-bit idx | tile/core 两级 + repeat | 元素 owner 对比连续 tile owner |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 288B UB meta + thread locals | 576B UB meta + thread locals | 双分区 TBuf + RegTensor | 均无 dynamic UB/通信 |
| 内存总占用与复用 | 无数据 tile UB | 同左 | `2*outTile*(sizeofT+4)` | 超宽用 UB 合并 DMA |
| 搬运/Vector/SIMT/NDDMA/通信 API | 标量 GM load/store、SIMT | 同左 | DataCopyPad + MicroAPI | 无 NDDMA/通信 |
| 同步、精度与 tail | thread 顺序，idx guard | 同左 | MTE/V parity event、mask | 均单 owner、RNE 输出 |
| fallback/失败边界 | 无后继 | 地址缩小时为 R-001 | guard 不满足回 SIMT | Tiling 校验失败才终止 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持 |
| Host/Launcher | circular selector 投 `CircularPadGrad` | 同左 | `IsRegBase()` 投 `PadV3Grad` |
| Kernel/资源 | arch32 workspace atomic Vector | 同左 | SIMT 或 MicroAPI SIMD；无 body workspace/atomic |

## 限制与失败边界

- API 层约束与无后缀卡相同：rank 3/4、三 Tensor dtype/format 一致、padding 长度 4、shape 等式成立；最后一维遵守公开上限。
- 950 Tiling 额外验证每侧正 circular padding 不大于推导输出轴长，输出任何轴不得为负；不满足直接失败。
- `PadV3GradACTilingData` 的 `sizeof=408/alignof=8/offset` 为按显式标准布局字段推导，固定仓无 static_assert；编译器寄存器分配、SIMT thread-local spill 与 framework executor 细粒度同步固定仓不可见。
- Host 固定预留 16 MiB 系统 workspace，但 circular kernel body 不读取；不得把它误作跨节点结果或可供 TBuf 使用的空间。
- template key 的数值编码由外部 `GET_TPL_TILING_KEY` 宏生成，固定仓未展开；本卡以完整 tuple 标识，不能臆造数字 key。
- 本卡仅闭合 2D ACLNN 可达 rank 3/4，不把同一通用 body 的 3D API 语义并入。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/circular_pad_grad/docs/aclnnCircularPad2dBackward.md` | 产品表/约束 | 950 公开支持与 API 边界 |
| 入口声明/约束 | `conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.h` | 两段式声明 | C API 参数 |
| 入口实现 | `conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.cpp` | `commonPad2dBackward` | 校验、padding、executor graph |
| 入口实现 | `conversion/pad_v3_grad/op_api/padv3grad.cpp` | `PadV3Grad` / `PadV3GradAiCore` | RegBase circular selector |
| Host/Tiling/Launcher | `conversion/pad_v3_grad/op_host/pad_v3_grad_infershape.cpp` | `InferShape4PadV3Grad` | 输出 shape producer |
| Host/Tiling/Launcher | `conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.cpp` | `PadV3GradACTiling` | collapse、Route、POD、block/workspace |
| Host/Tiling/Launcher | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_struct.h` | `PadV3GradACTilingData` | POD ABI |
| Host/Tiling/Launcher | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_tilingkey.h` | `pad_v3_grad` template args | tuple schema |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/pad_v3_grad_apt.cpp` | `pad_v3_grad` | AIV-only entry/router |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simt.h` | `LaunchCircularKernelSIMT` | 32/64-bit body 选择 |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_circular_simt.h` | `PadV3GradCircularSimt` | SIMT 全实现 |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simd.h` | huge-width launcher | SIMD body 接线 |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_circular_huge_width.h` | `KernelPadV3GradCircularHugeWidth` | SIMD 全实现 |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_common.h` | `BUFFER_NUM` / `CAST_TRAIT_0` | 双分区常量与 MicroAPI cast trait |
| 构建接线 | `conversion/pad_v3_grad/op_host/pad_v3_grad_def.cpp` | `PadV3Grad` OpDef | ascend950 AICore 注册 |
| 构建接线 | `conversion/pad_v3_grad/op_host/config/ascend950/pad_v3_grad_binary.json` | dtype/padding binaries | FP16/BF16/FP32 + INT64 接线 |
| 构建接线 | `conversion/pad_v3_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950→arch35 构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 408B 显式 POD 与逐字段 producer/consumer | 数组长度 8、signed pad、tuple 字段语义 | `PadV3GradACTilingData` |
| 调用顺序 | collapse→stride→Route→key→单节点 | circular 必须先于其他 mode 分派并保持 RegBase selector | `PadV3Grad` / `DoTiling` |
| workspace/内存 | body 不用 workspace；SIMD 双分区公式 | 16 MiB 是系统声明；TBuf parity 不可提前覆盖 | `DoTiling` / `InitBuffer` |
| 同步 | SIMT 单 owner；SIMD MTE2/V/MTE3 event | parity event 和 `idx>1` guard | `ProcessOneStep` / `SetEvent` |
| 精度与尾块 | 16-bit→FP32 sum→RNE sat；32/64-bit 地址分流 | 候选嵌套顺序、mask、`dataLen` | `CopyOut` / `CopyOutputToGM` |
| 失败边界 | padding/out shape 校验、SIMD 可回 SIMT | 不得把负 padding 强行送 SIMD，不得扩为 3D 卡 | `CheckModeInputParam` / `DoTilingModeCircular` |