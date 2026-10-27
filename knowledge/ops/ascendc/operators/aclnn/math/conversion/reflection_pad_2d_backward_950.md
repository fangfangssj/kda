---
type: Operator
title: "ReflectionPad2dBackward 950 ACLNN 算子知识"
description: "说明 aclnnReflectionPad2dBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/docs/aclnnReflectionPad2dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_api/padv3grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_tilingkey.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/pad_v3_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_huge_width.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_normal.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_gather.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/pad_v3_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/config/ascend950/pad_v3_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T09:32:37Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ReflectionPad2dBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnReflectionPad2dBackwardGetWorkspaceSize` / `aclnnReflectionPad2dBackward`，公开两段式 ACLNN；950 由 `IsRegBase()` 分支加入 `PadV3Grad`。 |
| 计算语义 | 为每个原输入坐标收集正文与各轴 reflect 镜像候选的 `gradOutput` 值并求和。 |
| 输入输出 | ND，rank 3/4；FP16/BF16/FP32 可达 AscendC，其中 FP16/BF16 入核前升为 FP32；`padding=[left,right,top,bottom]`。 |
| 执行类型 | 单 AIV Kernel；R-001/R-002 为节点内 SIMT，R-003/R-004/R-005 为 AIV Vector MicroAPI；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 SIMT-32；R-002 SIMT-64；R-003 SIMD huge-width；R-004 SIMD normal-width；R-005 SIMD small-width gather。 |
| 核心资源 | SIMT 静态 UB 元数据+线程寄存器；SIMD 双缓冲 TQue/TBuf、FP32 中间区、索引 RegTensor；Host 固定声明 16MiB workspace，body 不读。 |
| 来源闭包 | `source_closed`：ACLNN→950 RegBase selector→PadV3Grad infer/Tiling/POD→tuple key→SIMT/三个 SIMD body→ascend950 config/CMake 闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 collapse 后第 `d` 轴，设 padded `gradOutput` 长为 `I[d]`，原输入/结果长 `O[d]=I[d]-L[d]-R[d]`。给定输出坐标 `o[d]`，正文输入坐标是 `o[d]+L[d]`；reflect 左/右候选分别是 `L[d]-o[d]` 与 `2*O[d]-o[d]+L[d]-2`，仅在左/右 pad 区条件成立时加入。多轴取候选笛卡尔积：

```text
gradInput[o] = Σ gradOutput[c]
c[d] ∈ {o[d]+L[d], L[d]-o[d], 2*O[d]-o[d]+L[d]-2}
候选必须满足 reflect 区谓词与 0 <= c[d] < I[d]。
```

SIMT 为每个输出线程枚举候选；SIMD 把正文块搬到 UB，对镜像区做 gather/add/fold。两者都是单 owner 写每个输出，无 GM atomic。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | ND，rank 3/4，前向 padded shape | FP16/BF16/FP32；API 还声明 FP64/complex | 与 `self/gradInput` 一致；非连续时 `Contiguous`。 |
| `self` | ND，rank 3/4 | 同上 | 只提供原 shape；与 `gradInput` shape 相同。 |
| `padding` | ACLNN 长度 4；L0 扩展为 `2*rank` | INT64 | `[left,right,top,bottom]`，API 要求非负且每边小于对应 `self` 维。 |
| `gradInput` | 与 `self` 同 shape/layout | 同上 | 连续核心输出经 `ViewCopy` 写回。 |

### 入口与版本

固定仓只有无版本后缀的公开 API。ACLNN 文档明确公开支持 Ascend 950PR/950DT。L0 `PadV3Grad` 在 950 上先命中 `padFlag && IsPadV4GradAicoreRegBaseSupport`，但该分支的 launcher 是 `PadV3GradAiCore`，不是 A2/A3 `PadV4Grad`。`PadV3Grad` OpDef、`SUPPORT_COMPUTE_UNIT="ascend950"`、arch35 Tiling 和 `pad_v3_grad_apt` 共同支撑 950。FP16 reflect/BF16 在 L0 前已 Cast FP32，因而本 API 的 950 核心节点实际 `DTYPE_X=float`。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 层按顺序检查指针、dtype、format、rank、shape、padding 长度/值、BF16 特例和 `gradOutput=self+padding`。非空路线：

1. `gradOutput/self` 连续化，rank 3 在 0 轴 unsqueeze；
2. padding 扩成 rank 对齐 pair Tensor，并传 `mode=reflect,paddings_contiguous=true,padFlag=true`；
3. FP16 reflect/BF16 的 `gradOutput` Cast 到 FP32；
4. 950 RegBase selector 加入 `PadV3Grad` AICore 节点；
5. rank 3 squeeze，必要时 Cast 回原 dtype，最后 ViewCopy。

arch35 Tiling 读取 input shape/dtype、mode 和 padding Tensor，消除 shape=1 且净 padding=0 的轴，合并连续无 pad 轴，然后计算 `outShape/inStride/outStride`。reflect 每边 padding 必须 `< inferredOutShape`，否则 Tiling 失败。

### 参数、Workspace 与生命周期

`Init` 从 950 取 `GetCoreNumAiv()`、UB size、UB block size、VReg size 和 cache line；SIMT 使用所有 AIV 核，SIMD 经 `TilingInfoTune` 可缩减到实际使用核数。Host 固定 `workspaces[0]=16*1024*1024` B；entry 保留 `workspace` 参数，五个 reflect body 均不解引用，因此该空间是系统声明而非跨 Kernel 中间结果。executor 拥有 POD/连续中间张量/workspace，生命期覆盖同 stream 上的所有节点。

### 启动图与失败传播

```text
可选 Contiguous/Unsqueeze/Cast(FP32)
  -> PadV3Grad(x,paddings,y,{reflect,true}, workspace=16MiB, POD)
       -> pad_v3_grad<Mode=2,BigShape,IsSIMT,CutMode>
          -> SIMT body 或一个 AIV Vector body
  -> 可选 Squeeze/Cast(origin dtype)
  -> ViewCopy(result, gradInput)
```

核心只有一个 Device 节点，`blockDim=coreNum_`，输入/输出是 ND，POD 为 `PadV3GradACTilingData`。executor 在传入 stream 上保持顺序；infer/Tiling/launcher 失败阻断后续 Cast/ViewCopy。框架内部 event 和 Contiguous/ViewCopy workspace slice 固定仓不可见。

## Route 与参数设计

### 分核与资源计算

collapse 后：

```text
inShapeSize  = Π inShape[d]
outShape[d]  = inShape[d] - leftPad[d] - rightPad[d]
outShapeSize = Π outShape[d]
inStride/outStride 从尾轴向前乘积。
```

全非负 padding 先尝试 SIMD。`inShapeSize<=48Ki elements` 转 SIMT；否则 Host 以 `vectorSize/dtypeBytes` 对齐尾轴，并以 `ubSize/(2*dtypeBytes+4*FP32_BYTES)` 估算 buffer，单 buffer 封顶 `64KiB/dtypeBytes`。尾轴不可容纳时 R-003；中等宽度由 `DoFindSplitAxisByInput(true)` 得 R-004 或回退 R-003；小宽由 `DoFindSplitAxisByInput(false)` 得 R-005 或回退 R-003。`TilingInfoTune` 用 `ubTotalCount`、`ubPerCount`、核使用率和 4KiB tile 下限调整切轴/因子。负或混合 padding、空域都走 SIMT；本 API 校验屏蔽负 padding，但作为 L0 真实 fallback 仍保留。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 SIMT-32 | `(Mode=2,Big=0,SIMT=1,Cut=*)` | 空/负/混合 pad，或 SIMD selector 退回；且 `2*inSize+1<=INT32_MAX && 2*outSize+1<=INT32_MAX` | 超地址域 R-002 |
| 2 | R-002 SIMT-64 | `(2,1,1,*)` | SIMT 情形且任一 `2*size+1>INT32_MAX` | 无 |
| 3 | R-003 SIMD huge-width | `(2,*,0,0)` | 全 pad>=0、非空、大于 48Ki；尾轴对齐长超 buffer，或 `2*lastAlign>buffer`，或 normal/small 调优后 `ubAxis==last` | 否则 R-004/R-005 |
| 4 | R-004 SIMD normal-width | `(2,*,0,1)` | `inLast>vectorSize/4`，`DoFindSplitAxisByInput(true)` 后 `ubAxis!=last`且 `2*lastAlign<=outTileSize` | R-003 |
| 5 | R-005 SIMD small-width gather | `(2,*,0,2)` | `inLast<=vectorSize/4`，`DoFindSplitAxisByInput(false)` 后 `ubAxis!=last` | R-003 |

```text
reflect -> collapse + strides
  empty/negative/mixed -> SIMT -> 32/64-bit by 2*size+1
  all nonnegative:
    inSize<=49152 or dimNum>5 -> SIMT
    else CalculateTilingKeyMirror:
      tail too large / 2*tail exceeds buffer -> huge-width
      inLast>VReg/4 -> normal candidate -> tune -> normal or huge
      else          -> small gather candidate -> tune -> small or huge
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | Mode2/Big0/SIMT1 | 950，core FP32，SIMT 32-bit 域 | 单 AIV entry 内 256-thread SIMT | `PadV3GradMirrorSimt<float,2>::Process<int32_t>` | 288B 静态 UB 元数据，thread-private sum | grid-stride `idx<outputSize` |
| R-002 | Mode2/Big1/SIMT1 | 同上但超 32-bit 域 | 同上 | `Process<int64_t>` | 576B 元数据，64-bit 地址 | 同上 |
| R-003 | Mode2/SIMT0/Cut0 | huge-width 谓词 | 单 AIV Vector | `KernelPadV3GradMirrorHugeWidth<float,2>` | 3 个双分区 TBuf，MTE/V event | `mDataLen=min(F,remain)` |
| R-004 | Mode2/SIMT0/Cut1 | normal-width 谓词 | 单 AIV Vector | `KernelPadV3GradMirrWithNormalWidth<float,2>` | depth2 in/out queue + FP32 tmpBuf，loop-mode DMA | 切轴 tail |
| R-005 | Mode2/SIMT0/Cut2 | small-width 谓词 | 单 AIV Vector | `PadV3GradGather<float,2>` | depth2 queue、mid/pad TBuf、6个 256B 索引区 | factor tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| reflect padding | `padFront,padBack < outShape` | 每轴只有正文+左+右三候选 | 全部 | Tiling failed | `CheckModeInputParam` |
| SIMT 阈值 | `inShapeSize<=48*1024` | 小数据直接逐输出枚举 | R-001/002 | SIMT | `CalculateTilingKeyMirror` |
| SIMT 地址宽 | `2*inSize+1` 或 `2*outSize+1 > INT32_MAX` | reflect 地址公式含倍增 | R-001/002 | 切 64-bit | `DoTilingWithSIMTMirror` |
| single-buffer cap | `bufferSize<=64KiB/dtypeBytes` | TBuf/queue 分区 API 上界 | R-003..005 | 缩小 factor/切轴 | `CalculateTilingKeyMirror` |
| 核利用率 | used/core>=0.8 或 tile bytes<=4096 | 避免过少核，也防止过小 tile 继续搜索 | R-003..005 | `GetOptimizeTiling` | `CheckTilingInfoSatisfied` |
| max four inner axes | `dimNum-ubAxis<=4` | normal/gather body 的局部折叠专用到 4 轴 | R-004/005 | Host 把 ubAxis 推到 `dimNum-4` | `DoFindSplitAxisByInput` |

### POD/ABI

`PadV3GradACTilingData` 是显式标准布局字段序列；按自然 8B 对齐推得 `sizeof=408, alignof=8`，固定仓未提供单独 `static_assert`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `PadV3GradACTilingData` | `dimNum` | uint8/8 | offset 0 | 轴 | collapse | SIMT specialization/SIMD loop | API 合轴后 1..4 |
| 同上 | `ubAxis` | uint8/8 | offset 1 | 轴 index | selector/tune | SIMD tile 解码 | SIMT inactive |
| 同上 | `ubFactor` | uint32/32 | offset 4 | 切轴元素 | selector/tune | SIMD factor/tail | SIMT inactive |
| 同上 | `ubPerCount` | uint32/32 | offset 8 | tile/核 | `TilingInfoTune` | SIMD start/end | SIMT inactive |
| 同上 | `ubTotalCount` | uint32/32 | offset 12 | tile | 同上 | SIMD 核早退 | SIMT inactive |
| 同上 | `outTileSize` | uint32/32 | offset 16 | 元素/分区 | selector | SIMD InitBuffer | SIMT inactive |
| 同上 | `additionTileSize` | uint32/32 | offset 20 | 元素/字节转换依 route | selector | normal/gather extra space | SIMT inactive |
| 同上 | `inShape[8]` | uint64[8]/512 | offset 24 | 元素/轴 | input shape+collapse | 全 body | 仅前 dimNum 有效 |
| 同上 | `outShape[8]` | uint64[8]/512 | offset 88 | 元素/轴 | `ComputeAfterPaddingsAndStrides` | 输出解码/tail | 同上 |
| 同上 | `inStride[8]` | uint64[8]/512 | offset 152 | 元素 stride | 同上 | GM 地址 | 同上 |
| 同上 | `outStride[8]` | uint64[8]/512 | offset 216 | 元素 stride | 同上 | 输出解码/GM | 同上 |
| 同上 | `leftPad[8]` | int64[8]/512 | offset 280 | 元素/轴 | collapse | reflect 左候选 | L0 可负，API 非负 |
| 同上 | `rightPad[8]` | int64[8]/512 | offset 344 | 元素/轴 | collapse | reflect 右候选 | 同上；总大小 408 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `(2,0,1,*)` | `DoTilingModeMirror/DoTilingWithSIMTMirror` | `PadV3GradACTilingData` | `pad_v3_grad` → `LaunchMirrorKernelSIMT` | `PadV3GradMirrorSimt::Process<int32_t>` | PadV3Grad ascend950 OpDef/config/CMake |
| R-002 | `(2,1,1,*)` | 同上 | 同上 | 同上 | `Process<int64_t>` | 同上 |
| R-003 | `(2,*,0,0)` | `CalculateTilingKeyMirror/TilingInfoTune` | 同上 | `pad_v3_grad` → huge launcher | `KernelPadV3GradMirrorHugeWidth<float,2>` | 同上 |
| R-004 | `(2,*,0,1)` | 同上 | 同上 | `pad_v3_grad` → normal launcher | `KernelPadV3GradMirrWithNormalWidth<float,2>` | 同上 |
| R-005 | `(2,*,0,2)` | 同上 | 同上 | `pad_v3_grad` → gather launcher | `PadV3GradGather<float,2>` | 同上 |

## Kernel 实现

### 公共执行流程

`pad_v3_grad<2,...>` 显式声明 `KERNEL_TYPE_AIV_ONLY`并注册 POD。`isSimt=true` 时进入 `PadV3GradSimt`；否则 `cutMode=0/1/2` 分别进入 huge-width/normal-width/gather。SIMT 以输出为 owner 枚举最多 `3^dimNum` 候选；SIMD 先搬正文块，再对外轴、切分轴和内轴的 mirror 做 gather/add/fold。由于 ACLNN 入核 dtype 是 FP32，这五路都在 FP32 累加；body 中的 FP16/BF16 Cast 分支对此入口 inactive。

### Route R-001：SIMT-32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/L0 | `aclnn_pad2d_backward.cpp` / `padv3grad.cpp` | `commonPad2dBackward/PadV3Grad` | reflect, 950 RegBase | PadV3GradAiCore |
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTMirror` | SIMT 且 `2*size+1<=INT32_MAX` | Big=false |
| Router | `pad_v3_grad_apt.cpp` | `pad_v3_grad<2,false,true,*>` | SIMT | mirror launcher |

#### Device 数据流

##### 函数级调用链

```text
PadV3GradAiCore -> pad_v3_grad<2,false,true,*>
  -> LaunchMirrorKernelSIMT<float,false,2>
  -> PadV3GradMirrorSimt<float,2>::Init -> Process<int32_t>
  -> asc_vf_call<SimtComputeMirror{One..Four}>(threads=256)
```

##### 源码等价伪码

```text
for idx=blockIdx*256+threadIdx; idx<outputSize; idx+=blockNum*256:
  decode outIndex by magic division
  inIndex[d]=outIndex[d]+leftPad[d]
  candidates[d]={inIndex[d]}
  if 0<outIndex[d]<=leftPad[d]: append leftPad[d]-outIndex[d]
  if 0<outShape[d]-outIndex[d]-1<=rightPad[d]:
      append 2*outShape[d]-outIndex[d]+leftPad[d]-2
  total=sum x[cartesian_product(candidates)] after cutBounds checks
  y[idx]=total
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | strided candidate / `idx` | 全域 | FP32/ND | caller/thread | thread/ViewCopy | executor | scalar gather，单 owner |
| SIMT UB arrays | magic/shift + 7 组 shape/stride/pad/bound | `9*8*4=288B` | uint32/int32 | Process | SIMT threads | current AIV | read-only |
| thread locals | index/candidates/total | 寄存器数固定仓不可见 | int32/FP32 | thread | thread | one idx | 不共享 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  magic["magic+shift 64B"]:1
  meta["shape/stride/pad/bounds 224B"]:2
```

##### 字节账与复用顺序

```text
static_ub = 288 B; dynamic_ub = N/A
workspace_body = 0 B  # Host 仍声明 16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| UB arrays | Process init | all 256 threads | kernel end | `asc_vf_call` boundary | 调用期只读 |
| thread locals | current idx | candidate loops | next grid-stride idx | thread order | 不跨 thread |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| metadata→threads | `asc_vf_call` | AIV init | SIMT callee | UB arrays | 固定 body 未见额外 barrier |
| sum/store | thread-private | GM loads | same thread | one y element | FP32，`idx<outputSize` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTMirror` | 32-bit guard |
| Kernel entry/router | `pad_v3_grad_apt.cpp` | Mode2/SIMT1 | AIV-only SIMT dispatch |
| Device body/helper | `pad_v3_grad_mirror_simt.h` | `Process<int32_t>/ReflectDimOffset` | 256 threads，3-way candidate |

### Route R-002：SIMT-64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTMirror` | `2*inSize+1>INT32_MAX \|\| 2*outSize+1>INT32_MAX` | Big=true |
| Router | `pad_v3_grad_simt.h` | `LaunchMirrorKernelSIMT<float,true,2>` | true | `Process<int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
pad_v3_grad<2,true,true,*>
  -> LaunchMirrorKernelSIMT<float,true,2>
  -> PadV3GradMirrorSimt<float,2>::Process<int64_t>
```

##### 源码等价伪码

```text
同 R-001，但 shape/stride/pad/bound、linear idx、magic/shift 和 GM offset 均为 64-bit。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | 64-bit strided offset | 全域 | FP32 | caller/thread | thread | executor | 防截断 |
| SIMT UB arrays | same nine arrays | `9*8*8=576B` | uint64/int64 | Process | threads | current AIV | read-only |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  magic["magic+shift 128B"]:1
  meta["shape/stride/pad/bounds 448B"]:2
```

##### 字节账与复用顺序

```text
static_ub=576 B; dynamic_ub=N/A; workspace_body=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| arrays/locals | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 64-bit 类型必须贯穿 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 只改地址宽，FP32 顺序不变 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | `DoTilingWithSIMTMirror` | doubled-size guard |
| Kernel entry/router | `pad_v3_grad_simt.h` | Big=true | int64 specialization |
| Device body/helper | `pad_v3_grad_mirror_simt.h` | `Process<int64_t>` | 64-bit magic/address |

### Route R-003：SIMD huge-width

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `CalculateTilingKeyMirror` | tail 大于 buffer/两倍 tail 超 buffer/调优后切尾轴 | Cut0 |
| Router | `pad_v3_grad_apt.cpp` | `PadV3GradMirror<2,0,*>` | Cut0 | huge launcher |

#### Device 数据流

##### 函数级调用链

```text
LaunchKernelPadV3GradMirrorHugeWidth<float,2>
  -> KernelPadV3GradMirrorHugeWidth::Init
  -> Process -> ProcessOneStep
     -> ProcessSelfAdd -> CalculateOffsetParams
     -> CopyAndCal -> ProcessMiddle/Left/RightData -> CopyOutputToGM
```

##### 源码等价伪码

```text
for tile idx owned by core:
  decode outer coords and tail-axis start; len=min(F,remain)
  choose parity buffers; zero FP32 result
  DMA正文 x[inSelf:inSelf+len], add to result
  enumerate mirror candidates of outer axes
  for each candidate: DMA middle/left/right source
    reverse-gather reflected lanes with DECREASE_ORDER indices
    masked FP32 add into result
  DMA exactly len FP32 values to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQueue TBuf | parity `(idx&1)*O` | `2*O*4` B | FP32 | MTE2 | V | pipe/tile | double region |
| resQueue TBuf | same | `2*O*4` B | FP32 | Duplicate/Add | MTE3 | pipe/tile | double region |
| tempQueue TBuf | same | `2*O*4` B | FP32 | optional cast/gather | V | pipe/tile | double region |
| GM output | `outSelfAddr` | `mDataLen` | FP32/ND | MTE3 | framework | executor | valid-byte tail |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  in0["in p0 O·4"]:1
  in1["in p1 O·4"]:1
  res0["res p0 O·4"]:1
  res1["res p1 O·4"]:1
  tmp0["tmp p0 O·4"]:1
  tmp1["tmp p1 O·4"]:1
```

##### 字节账与复用顺序

```text
UB_total = 6*outTileSize*4 bytes
workspace_body=0; Host system workspace=16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| parity p0/p1 | MTE2/V | V/MTE3 | idx+2 | `MTE3_V/MTE3_MTE2/MTE2_V/V_MTE2/V_MTE3` event by parity | idx>1 等待上次同 parity |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V | `MTE2_V` | DataCopyPad | ProcessSelf/reflect | in parity | FP32 |
| V→reuse | `V_MTE2` | gather/add | next MTE2 | in/temp | mask=`main/tail` |
| V→GM | `V_MTE3`, parity completion | result | DataCopyPad | res parity | `mDataLen` valid |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | `CalculateTilingKeyMirror/TilingInfoTune` | Cut0/F/O/core |
| Kernel entry/router | `pad_v3_grad_apt.cpp` / `pad_v3_grad_simd.h` | Cut0 launcher | huge class |
| Device body/helper | `pad_v3_grad_mirror_huge_width.h` | `ProcessOneStep/ProcessLeftData/ProcessRightData` | parity event + reverse gather |

### Route R-004：SIMD normal-width

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `CalculateTilingKeyMirror` | `inLast>VReg/4`，normal candidate 且调优后不切尾轴 | Cut1 |
| Router | `pad_v3_grad_simd.h` | `LaunchMirrorKernelNormal` | Cut1 | normal class |

#### Device 数据流

##### 函数级调用链

```text
KernelPadV3GradMirrWithNormalWidth::Process
  -> CopyIn(loop-mode DMA) -> GradGatherProcess
     -> CopyToTmpBuf -> GradAccumulateHighDimBulk
     -> GradProcessLineFromTmpBuf(W reverse gather/add)
  -> CopyOut
```

##### 源码等价伪码

```text
for output tile owned by core:
  decode outIndex on ubAxis; inIndex=outIndex+leftPad
  DMA rectangular padded block, optionally with 2-level loop mode
  copy FP32 input to tmp
  add every valid C/N/D5/H mirror plane/line from GM into tmp
  per row: gather left/right W mirror lanes and add in place
  write compact outShape block to GM, honoring ubFactor tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQueue | tile | depth2×`O*4` B | FP32 padded block | loop-mode MTE2 | tmp/fold | pipe/tile | depth2 |
| outQueue | tile | depth2×`O*4` B | FP32 compact | fold | MTE3 | pipe/tile | depth2 |
| tmpBuf | 0 | `O*4` B | FP32 accumulated | copy/add | W gather | pipe/tile | TBuf in-place |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQueue depth2 · each O·4"]:2
  tmp["tmpBuf O·4"]:1
  outQ["outQueue depth2 · each O·4"]:2
```

##### 字节账与复用顺序

```text
UB_total = (2+1+2)*outTileSize*4 = 20O bytes
workspace_body=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf | CopyToTmpBuf/high-dim add | W gather/final cast | next tile | single AIV Vector order | high-dim 累加完成后才 W fold |
| queue slot | CopyIn/fold | fold/CopyOut | next slot | Alloc/EnQue/DeQue/Free | depth2 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→queue | queue + loop-mode reset | MTE2 | V | rectangular tile | DataCopyPad |
| W fold | `DataCopyGather+Add`, main/tail mask | tmp | output | left/right ranges | FP32，reflect `modeOffset=0` |
| output | queue | V | MTE3 | compact valid tile | `inCopyLen=min(F,remain)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | normal Cut1 | split axis/factor |
| Kernel entry/router | `pad_v3_grad_simd.h` | `LaunchMirrorKernelNormal` | normal class |
| Device body/helper | `pad_v3_grad_mirror_normal.h` | `GradGatherProcess/GradProcessLineFromTmpBuf` | high-dim + W gather |

### Route R-005：SIMD small-width gather

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v3_grad_tiling_arch35.cpp` | `CalculateTilingKeyMirror` | `inLast<=VReg/4`，small candidate 且调优后不切尾轴 | Cut2 |
| Router | `pad_v3_grad_simd.h` | `LaunchMirrorKernelGather<float,true,2>` | Cut2 | gather class |

#### Device 数据流

##### 函数级调用链

```text
PadV3GradGather<float,2>::Process
  -> ComputePositionIndices -> ProcessOneFactor
     -> CopyInAndCast -> FoldOuterAxis -> ProcessUbAxisPad
     -> FoldInnerAxis(FoldW/FoldHW/FoldCHW) -> CopyOutToGm
```

##### 源码等价伪码

```text
for tile:
  decode pos and curFactor=min(F,remaining)
  zero 8 sentinel elements in midBuf
  DMA正文 factor*innerSize to midBuf+8
  enumerate outer-axis mirror combinations and add
  gather-add-scatter left/right mirror of ubAxis
  fold inner W/H/C axes with vector index generators and gather-add
  write factor*outStride[ubAxis] valid FP32 elements to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQue | tile | depth2×`O*4` B | FP32 | MTE2 | mid | pipe/tile | depth2 |
| outputQue | tile | depth2×`O*4` B | FP32 | fold | MTE3 | pipe/tile | depth2 |
| tempMidBuf | `0..7` sentinel, data from 8 | `(O+8)*4` B | FP32 | Duplicate/Copy/Add | fold | pipe/tile | in-place |
| tempPadBuf | 0 | `O*4` B | FP32 | pad copy | scatter/fold | pipe/tile | scratch |
| six index/offset TBuf | independent | `6*256=1536` B | uint32/int32 | index generator | gather | pipe/tile | VReg-sized |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["input depth2 · 2O·4"]:1
  mid["mid (O+8)·4"]:1
  pad["pad O·4"]:1
  outQ["output depth2 · 2O·4"]:1
  idx["3 index · 768B"]:1
  off["3 offset · 768B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2O*4 + 2O*4 + (O+8)*4 + O*4 + 6*256
         = 24O + 1568 bytes
workspace_body=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| midBuf | CopyIn/outer/ub-axis fold | inner fold | next factor | Vector order + local `V_S` waits for scalar transitions | sentinel 8 保留为 0 |
| index TBuf | Generate*Indices | DataCopyGather | next batch | `V_S` event 在标量读取前等待 | 每个 mask 仅含有效 lane |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| gather index | `V_S` event | Vector index generation | scalar/control | index buffers | 256B VReg blocks |
| fold | `DataCopyGather+Add`/gather-add-scatter | mid/pad | output | mapped lanes | FP32, mask tail |
| output | `DataCopyPad` | output queue | GM | valid factor | `curFactor=min(F,remain)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v3_grad_tiling_arch35.cpp` | small Cut2 | split axis/factor |
| Kernel entry/router | `pad_v3_grad_simd.h` | `LaunchMirrorKernelGather` | gather class |
| Device body/helper | `pad_v3_grad_mirror_gather.h` | `ProcessOneFactor/Fold*` | outer/axis/inner fold + exact tail |

## 算子特定优化

### Trick T-001：尾轴双分区流水与反向 Gather 合并 reflect 边界

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：全非负 padding、`inSize>48Ki`，尾轴大宽命中 Cut0
- Golden 来源：[`conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_huge_width.h:168-176`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_huge_width.h#L168-L176)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<T> srcLocal = inQueue_.Get<T>();
LocalTensor<T> src = srcLocal[(idx & 1) * tilingData_->outTileSize];

LocalTensor<CalType> resLocal = resQueue_.Get<CalType>();
LocalTensor<CalType> res = resLocal[(idx & 1) * tilingData_->outTileSize];

LocalTensor<CalType> tempLocal = tempQueue_.Get<CalType>();
LocalTensor<CalType> temp = tempLocal[(idx & 1) * tilingData_->outTileSize];
```

#### 机制与设计原因

Host 把宽尾轴切成 `ubFactor`、以 `ubPerCount` 分给核。Device 对 input/result/temp 都用 `(idx&1)` 双分区，并用 parity event 让当前 MTE2、Vector reverse-gather/add 和前一 tile MTE3 交叠；`DECREASE_ORDER` 索引把 reflect 逆序区直接 gather 到寄存器并加入正文，无需在 UB 物化一份完整反转 Tensor。同步协议必须和双分区一起迁移，否则 `idx+2` 可提前覆盖尚在搬出的结果。

#### 迁移边界

- 适用：可按最内轴切块的 reflect/symmetric gradient gather-and-sum。
- 前提：950 AIV Vector、尾轴大于单 buffer 可容纳范围或 normal/small 调优回退，UB 能容纳 6 个 FP32 tile 分区。
- 失效：小数据/负 padding 改用 SIMT；外轴切分更合适时改用 R-004/R-005。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码的反例是只保留 `(idx&1)` 分区而删除 `idx>1` 的 parity wait：下一个同 parity tile 的 MTE2/V 可与上一个 MTE3 读写同一 UB 子区。

## 模板与实现对比

| 对比维度 | R-001 SIMT32 | R-002 SIMT64 | R-003 huge | R-004 normal | R-005 gather | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | small/fallback 32b | big address | tail-axis large | medium tail | small tail | 先非法/小数据，再 UB 局部性 |
| 模板与 dtype | float/Mode2 | float/Mode2 | float/Mode2/Cut0 | Cut1 | Cut2 | ACLNN core FP32 |
| 循环与分核 | thread grid-stride | 同左 64b | tail tile | outer-axis tile | factor/inner fold | Host tune |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 288B+thread regs | 576B+regs | 6O*4 TBuf | 5O*4 queue/tmp | 24O+1568B | 无 dynamic UB/通信 |
| 内存总占用与复用 | no body ws | 同左 | parity reuse | queue/tmp reuse | mid/index reuse | Host workspace 都是系统 16MiB |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM SIMT | 同左 | Pad+Gather+Add | loop DMA+Gather+Add | Gather+scatter/fold | 无 NDDMA/通信 |
| 同步、精度与 tail | call boundary,FP32 | 同左 | parity events/mask | queues/mask | V_S+mask | 全部 exact tail |
| fallback/失败边界 | big→R002 | none | selector | huge | huge | reflect pad invalid Tiling fail |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，实际 `PadV4Grad` | 同左 | 公开支持，实际 RegBase `PadV3Grad` |
| Host/Launcher | SoC selector 910B | 910_93 | `IsRegBase()`→PadV3Grad |
| Kernel/资源 | 六 shape-specialized AIV body | 同 A2 | SIMT32/64 + 3 SIMD cut modes，16MiB system workspace |

## 限制与失败边界

- API 限制 ND、rank 3/4、dtype/format/shape 一致，padding 长度 4，每边非负且小于原 H/W；BF16 还有 padding<=7/H 无 pad 特例。
- FP16/BF16 可达 AscendC 主节点前后 Cast FP32；DOUBLE/COMPLEX 转 AICPU，其 body 在固定 ops-math 仓不可见，不得当作 950 PadV3Grad route。
- Tiling/POD 支持最多 8 轴，非 constant 最多 5 轴；本 ACLNN 入口 rank 更窄。`outShape<0`、reflect pad 不小于 inferred output、平台资源为 0 都失败。
- SIMD `ubTotalCount/ubPerCount` 窄化为 uint32，固定实现未见独立溢出检查；大元素域只对 SIMT 地址切 64-bit。
- 16MiB Host workspace 在 reflect body 中未解引用；不可据此删除 launcher ABI 中的 workspace，框架系统用途固定仓不可见。
- 文档警告 `gradOutput` 元素数超 `300*1024*1024` 有超时风险；本次未做性能验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.h` | `aclnnReflectionPad2dBackward*` | ACLNN ABI |
| 入口实现 | `conversion/pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.cpp` | `commonPad2dBackward` | 校验/Cast/L0/ViewCopy |
| Host selector | `conversion/pad_v3_grad/op_api/padv3grad.cpp` | `PadV3Grad/IsPadV4GradAicoreRegBaseSupport` | 950 选 PadV3Grad |
| Host/Tiling/Launcher | `conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.cpp` | `PadV3GradACTiling` | collapse/route/POD/block/workspace |
| POD | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_struct.h` | `PadV3GradACTilingData` | ABI |
| key | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_tilingkey.h` | `pad_v3_grad` tuple | mode/big/SIMT/cut |
| Kernel entry/body | `conversion/pad_v3_grad/op_kernel/pad_v3_grad_apt.cpp` | `pad_v3_grad` | AIV-only router |
| Kernel body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simt.h` | `LaunchMirrorKernelSIMT` | 32/64 dispatch |
| Kernel body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_simt.h` | `PadV3GradMirrorSimt` | R-001/002 |
| Kernel body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_simd.h` | three mirror launchers | SIMD dispatch |
| Kernel body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_huge_width.h` | huge class | R-003 |
| Kernel body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_normal.h` | normal class | R-004 |
| Kernel body | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_mirror_gather.h` | gather class | R-005 |
| Kernel helper | `conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_common.h` | common constants/types | FP32 promote/VReg/queue |
| 构建接线 | `conversion/pad_v3_grad/op_host/pad_v3_grad_def.cpp` | `PadV3Grad` | ascend950 OpDef/apt kernel |
| 构建接线 | `conversion/pad_v3_grad/op_host/config/ascend950/pad_v3_grad_binary.json` | PadV3Grad binaries | dtype/padding variants |
| 构建接线 | `conversion/pad_v3_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950→arch35 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 408B explicit struct | 字段顺序、数组长 8、单位和 inactive route | `PadV3GradACTilingData` |
| 调用顺序 | ACLNN→FP32 Cast→PadV3Grad→Cast/ViewCopy | 950 不得改走 PadV4Grad body | `commonPad2dBackward/PadV3Grad` |
| workspace/内存 | body 不读 16MiB，UB 由 cut mode 分配 | launcher workspace ABI 不得删；depth2 必须乘入 | `DoTiling`, route `Init` |
| 同步 | SIMT call boundary；huge parity events；normal queues；gather V_S | 双分区 `idx+2` 不可提前覆盖 | route bodies |
| 精度与尾块 | ACLNN core FP32；mask/grid-stride/exact byte tail | 不得让无效 gather index 参与 Add | `mirror_simt/huge/normal/gather` |
| 失败边界 | reflect pad、shape、resource 失败 | 不得把 AICPU 或 A2/A3 PadV4 body 混入 950 route | `CheckModeInputParam/PadV3Grad` |