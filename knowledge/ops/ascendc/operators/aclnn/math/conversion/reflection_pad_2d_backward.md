---
type: Operator
title: "ReflectionPad2dBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnReflectionPad2dBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_host/pad_v4_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_host/pad_v4_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_mini_h_w_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_h_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_w_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_small_h_large_w_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_large_h_small_w_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_h_w_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_host/pad_v4_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_host/config/ascend910b/pad_v4_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_host/config/ascend910_93/pad_v4_grad_binary.json
    role: implementation
created_at: "2026-08-25T09:32:37Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ReflectionPad2dBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnReflectionPad2dBackwardGetWorkspaceSize` / `aclnnReflectionPad2dBackward`，公开两段式 ACLNN；A2/A3 selector 对 FP32/FP16/BF16 可达 `PadV4Grad`。 |
| 计算语义 | 对前向 reflect-pad 的多对一映射做反向求和：原位、左/上镜像、右/下镜像对应的 `gradOutput` 累加到 `gradInput`。 |
| 输入输出 | `gradOutput/self/gradInput` 同 dtype、ND、rank 3/4；`padding=[left,right,top,bottom]`；AscendC 主节点实际以 FP32 计算。 |
| 执行类型 | 单 `AIV_VECTOR` Kernel，`KERNEL_TYPE_AIV_ONLY`；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 mini-HW；R-002 H-only；R-003 W-only；R-004 small-H/large-W；R-005 large-H/small-W；R-006 generic-HW。 |
| 核心资源 | GM、AIV UB `TQue/TBuf`、转置局部区；R-003/004/005/006 有每核 workspace slice；queue/event 维持 MTE2→V→MTE3。 |
| 来源闭包 | `source_closed`：ACLNN→L0 A2/A3 selector→`PadV4Grad` Tiling/POD→key router→六个 AIV body→A2/A3 OpDef/config 闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设前向原输入空间高宽为 `(Ho,Wo)`，`gradOutput` 高宽为 `(Hi,Wi)=(Ho+top+bottom, Wo+left+right)`。对每个输出坐标 `(h,w)`，反向结果是所有在 reflect 前向中映射到 `(h,w)` 的 padded 坐标求和：

```text
gradInput[n,c,h,w] = Σ gradOutput[n,c,ph,pw]
ph ∈ {h+top, top-h, 2*Ho+top-2-h}
pw ∈ {w+left, left-w, 2*Wo+left-2-w}
仅保留满足各边 reflect 区条件的候选。
```

A2/A3 body 不用 GM atomic；每个 `(n,c)` 或 `(n,c,h)` 由唯一 AIV 核的 tile owner 完成全部累加并写回。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | ND，rank 3/4，高宽是前向 padded shape | FP16/BF16/FP32/FP64/COMPLEX64/COMPLEX128 | 与 `self/gradInput` dtype 一致；非连续时 `Contiguous`。AscendC 卡覆盖 FP16/BF16/FP32 可达路线。 |
| `self` | ND，rank 3/4 | 同上 | 只提供前向原输入 shape；与 `gradInput` shape 相同。 |
| `padding` | ACLNN 长度 4；L0 Tensor 长度 `2*rank` | INT64 | 值为 `[left,right,top,bottom]`，非负，每边小于 `self` 对应维；BF16 还要求 H 无 pad 或所有值 `<=7`。 |
| `gradInput` | 与 `self` 同 shape/layout | 同上 | 连续核心输出经 `ViewCopy` 写回，支持非连续目标。 |

### 入口与版本

固定仓只有无版本后缀的公开 2D backward API。`commonPad2dBackward` 固定传 `mode="reflect"` 和 `padFlag=true`。A2/A3 的 `IsPadV4GradAicoreSupport` 要求 rank `<=4`、dtype 为 FP16/BF16/FP32 且 SoC 为 `ASCEND910B` 或 `ASCEND910_93`，随后 launcher 加入 `PadV4Grad`。FP16 reflect 和 BF16 在此前被 Cast 到 FP32，故 ACLNN 实际可达的 PadV4Grad key 是 `1xxx`；核心结果再 Cast 回原 dtype。DOUBLE/COMPLEX 不命中 AscendC selector，转 AICPU，不进入本 AscendC 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 按顺序检查空指针、dtype 一致与支持集、非私有 format、rank 3/4、`padding.Size()==4`、padding 与 `self` 的 reflect 边界、BF16 特例、`gradOutput` 高宽等式。非空时：

1. `gradOutput/self` 分别 `Contiguous`；rank 3 在 axis 0 `UnsqueezeNd` 成4D；
2. `GetPaddingTensor` 生成 `[0,0,0,0,top,bottom,left,right]`，`paddings_contiguous=true`；
3. FP16 reflect/BF16 的 `gradOutput` Cast 到 FP32；
4. `l0op::PadV3Grad` 在 A2/A3 加入 `PadV4Grad(x,paddings,y,{reflect,true})`；
5. rank 3 结果 `SqueezeNd`，必要时 Cast 回原 dtype，最后 `ViewCopy`。

合法空 Tensor 返回 `workspaceSize=0` 和无核心节点 executor；特定 H/W/C 为 0 时按 API 代码返参数错误。

### 参数、Workspace 与生命周期

Tiling 从 `GetCoreNumAiv()` 和平台 UB 取资源，预留 `11KiB`；`availableUb=ubSize-11KiB`。`blockDim=usedCoreNum`，workspace 是 `workspacePerCore*blockNum + sysWorkspaceSize`。`workspacePerCore` 以字节生产，Device base 内除 `sizeof(T)` 变为元素数。R-001/R-002 为 0；R-003 是 `2*partWs*sizeof(T)`，`partWs=(wPad1<16&&wPad2<16)?64:wPadCopyCount`；R-004/005/006 是 `alignWidth*64*sizeof(T)` 每核。executor 拥有 POD、连续中间 Tensor 和 workspace，生命期覆盖传入 stream 上的所有节点。

### 启动图与失败传播

```text
可选 Contiguous/Unsqueeze/Cast(FP32)
  -> PadV4Grad Host Tiling
  -> pad_v4_grad(x,paddings,y,workspace,POD), blockDim=blockNum
       -> key 对应的单个 AIV body
  -> 可选 Squeeze/Cast(origin dtype)
  -> ViewCopy(continuousResult, gradInput)
```

核心 launch graph 只有一个 Device 节点；workspace 只是该节点的 per-core GM 交接区，不存在跨 Kernel 边。executor 保证同 stream 节点顺序；任意校验、infer、Tiling 或 launcher 失败都阻断后续 Cast/ViewCopy。entry 对 `workspace==nullptr` 或 `GetUserWorkspace(workspace)==nullptr` 直接返回。

## Route 与参数设计

### 分核与资源计算

Host 把 4D `x` 解为 `N,C,Hi,Wi`，输出 `Ho=Hi-top-bottom, Wo=Wi-left-right`，各 H/W 向上对齐 16。一般 `nMulC=N*C`；R-003 改为 `N*C*Hi`。若 `nMulC<=AIVCoreNum`，一核一份；否则 `ncPerCore=floor(nMulC/coreNum), tailNC=nMulC%coreNum`，前 `tailNC` 核多处理一份。UB factor 从扣除对齐 POD 后的 UB 按 route 分母、固定 128/64/16 行或 128 列切分，再按 16 元素/256B 对齐；`ubFactorElement<=0` 直接 Tiling 失败。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 mini-HW | `1000` / `PadV4GradPadMiniHW<float>` | `Hi<=128 && Wi<=112` | 否则下一项 |
| 2 | R-002 H-only | `1110` / `PadV4GradPadH<float>` | `left=right=0 && (top!=0\|\|bottom!=0) && Hi>64` | 下一项 |
| 3 | R-003 W-only | `1101` / `PadV4GradPadW<float>` | `top=bottom=0 && (left!=0\|\|right!=0)` | 下一项 |
| 4 | R-004 small-H/large-W | `1100` / `PadV4GradPadSamllHLargeW<float>` | `Hi<=64 && Wi>112` | 下一项 |
| 5 | R-005 large-H/small-W | `1010` / `PadV4GradLargeHSmallW<float>` | `Hi>128 && Wi<=128` | R-006 |
| 6 | R-006 generic-HW | `1111` / `PadV4GradPadHW<float>` | 上述均不命中 | 无 fallback |

```text
FP32 reflect:
  Hi<=128 && Wi<=112                                  -> R-001
  else if Wpad==0 && Hpad!=0 && Hi>64                 -> R-002
  else if Hpad==0 && Wpad!=0                          -> R-003
  else if Hi<=64 && Wi>112                            -> R-004
  else if Hi>128 && Wi<=128                           -> R-005
  else                                                 -> R-006
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1000 | A2/A3，core dtype FP32，mini 谓词 | 单 AIV | `PadV4GradPadMiniHW<float>::Process` | 整个 H×aligned-W 入 UB，H 累加后转置做 W 累加 | 行宽 DataCopyPad |
| R-002 | 1110 | 只 H pad | 单 AIV | `PadV4GradPadH<float>::Process` | 双输入/输出 queue，对称行 Vector Add | 最后 tile `calCount` |
| R-003 | 1101 | 只 W pad | 单 AIV | `PadV4GradPadW<float>::Process` | 宽边标量累加；中部 DMA；小/large pad 使用 per-core workspace | 中部长度 guard |
| R-004 | 1100 | H 小、W 大 | 单 AIV | `PadV4GradPadSamllHLargeW<float>::Process` | 64 行×W tile 转置，边 tile 写 workspace 后再读 | W tile tail |
| R-005 | 1010 | H 大、W 小 | 单 AIV | `PadV4GradLargeHSmallW<float>::Process` | H 分块、128 列转置，顶/底 workspace | H block tail |
| R-006 | 1111 | 通用 HW | 单 AIV | `PadV4GradPadHW<float>::Process` | 16行×W tile，双队列，上/下边 workspace | H/W 双 tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| A2/A3 selector | SoC `910B/910_93`、rank<=4、dtype FP16/BF16/FP32、reflect | 只有这两平台的 PadV4Grad config | 全部 | RegBase 走 PadV3Grad，其他走旧节点/AICPU | `IsPadV4GradAicoreSupport` |
| UB 预留 | `ubSize>11KiB` 且 `ubFactorElement>0` | Tiling/POD 和 library 预留后才可分 queue | 全部 | Tiling failed | `Tiling4PadV4Grad` / `SplitUb` |
| mini 上界 | `Hi<=128, Wi<=112` | 固定 128 行整面入 UB | R-001 | R-002..006 | `GetTilingKey` |
| workspace 切片 | R-003 按 pad 宽；R-004..006 `alignWidth*64*sizeof(T)` | 保存边界部分累加结果 | R-003..006 | workspace 不足由 launcher 失败 | `FillTilingData` |
| FP32 核心 | FP16 reflect 或 BF16 先 Cast FP32 | PadV4Grad 可达 key 统一为 1xxx，避免低精累加 | 全部 | Cast 失败阻断 | `commonPad2dBackward` |

### POD/ABI

`PadV4GradTilingData` 按字段宣言顺序为 20 个 32-bit 字段后跟 64-bit 字段；自然 8B 对齐下 `sizeof=88, alignof=8`。固定仓未提供 `static_assert/offsetof` 测量。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `PadV4GradTilingData` | `batch` | uint32/32 | offset 0 | 个 | `GetInputInfo` | `Base::Init` | N 窄化为 uint32 |
| 同上 | `channel` | uint32/32 | offset 4 | 个 | 同上 | 分核间接使用 | C |
| 同上 | `height` | uint32/32 | offset 8 | 元素/行 | 同上 | 全 body | Hi |
| 同上 | `width` | uint32/32 | offset 12 | 元素/行 | 同上 | 全 body | Wi |
| 同上 | `alignHeight` | uint32/32 | offset 16 | 元素 | `CeilAlign(Hi,16)` | base 保存 | 部分 body inactive |
| 同上 | `alignWidth` | uint32/32 | offset 20 | 元素 | `CeilAlign(Wi,16)` | workspace/地址 | 16 对齐 |
| 同上 | `outHeight` | uint32/32 | offset 24 | 行 | 输出 shape | 全 body | Ho |
| 同上 | `outWidth` | uint32/32 | offset 28 | 元素/行 | 输出 shape | 全 body | Wo |
| 同上 | `alignOutHeight` | uint32/32 | offset 32 | 元素 | `CeilAlign(Ho,16)` | base 保存 | 部分 inactive |
| 同上 | `alignOutWidth` | uint32/32 | offset 36 | 元素 | `CeilAlign(Wo,16)` | base 保存 | 部分 inactive |
| 同上 | `hPad1` | int32/32 | offset 40 | 行 | paddings[4] | H mirror | API 非负 |
| 同上 | `hPad2` | int32/32 | offset 44 | 行 | paddings[5] | H mirror | API 非负 |
| 同上 | `wPad1` | int32/32 | offset 48 | 元素 | paddings[6] | W mirror | API 非负 |
| 同上 | `wPad2` | int32/32 | offset 52 | 元素 | paddings[7] | W mirror | API 非负 |
| 同上 | `blockNum` | uint32/32 | offset 56 | AIV 核 | `GetUsedCore` | base/核早退 | `<=GetCoreNumAiv()` |
| 同上 | `ubFactorElement` | uint32/32 | offset 60 | 元素 | `SplitUb` | InitBuffer/tile loop | 必须 >0 |
| 同上 | `ncPerCore` | uint32/32 | offset 64 | NC 份/核 | `GetUsedCore` | base | 整除商 |
| 同上 | `tailNC` | uint32/32 | offset 68 | 核 | `GetUsedCore` | base | 前这些核+1 |
| 同上 | `tilingKey` | uint32/32 | offset 72 | key | `GetTilingKey` | 仅日志/base；router 用框架 key | ACLNN 可达 1000/1010/1100/1101/1110/1111 |
| 同上 | `wPadCopyCount` | uint32/32 | offset 76 | 元素 | `ceilAlign(2*max(wPad)+1,elemsPerBlock)` | R-003 | 其他 route inactive |
| 同上 | `workspacePerCore` | uint64/64 | offset 80 | 字节 | `FillTilingData` | base 除 `sizeof(T)` | R-001/002 为 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1000 | `pad_v4_grad_tiling.cpp::GetTilingKey` | `PadV4GradTilingData` | `pad_v4_grad.cpp::pad_v4_grad` | `PadV4GradPadMiniHW<float>` + base | `PadV4Grad` 910b/910_93 config |
| R-002 | 1110 | 同上 | 同上 | 同上 | `PadV4GradPadH<float>` | 同上 |
| R-003 | 1101 | 同上 | 同上 | 同上 | `PadV4GradPadW<float>` | 同上 |
| R-004 | 1100 | 同上 | 同上 | 同上 | `PadV4GradPadSamllHLargeW<float>` | 同上 |
| R-005 | 1010 | 同上 | 同上 | 同上 | `PadV4GradLargeHSmallW<float>` | 同上 |
| R-006 | 1111 | 同上 | 同上 | 同上 | `PadV4GradPadHW<float>` | 同上 |

## Kernel 实现

### 公共执行流程

`pad_v4_grad` 先检查 user workspace，构造 `TPipe`、解析 POD，标记 `KERNEL_TYPE_AIV_ONLY`，然后以 key 实例化 route 类。六个可达 body 共用 `PadV4GradBase<float>::Init`：前 `tailNC` 核处理 `ncPerCore+1`，其他核处理 `ncPerCore`，并将 GM x/y/workspace 绑定到张量。各 body 都按 `CopyIn→Compute→CopyOut` 或“边界计算→workspace→中部拷贝”执行；`DataCopyPad` 保护不对齐行宽。

### Route R-001：mini-HW

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/L0 | `aclnn_pad2d_backward.cpp` / `padv3grad.cpp` | `commonPad2dBackward` / `PadV3Grad` | reflect，A2/A3，FP32 core | `PadV4GradAiCore` |
| Tiling | `pad_v4_grad_tiling.cpp` | `GetTilingKey` | `Hi<=128 && Wi<=112` | key 1000 |
| Router | `pad_v4_grad.cpp` | `TILING_KEY_IS(1000)` | key | `PadV4GradPadMiniHW<float>` |

#### Device 数据流

##### 函数级调用链

```text
Tiling4PadV4Grad -> pad_v4_grad<key=1000>
  -> PadV4GradBase<float>::Init -> PadV4GradPadMiniHW<float>::InitBuffer
  -> Process -> CopyIn -> Compute(H Add -> transpose -> W Add -> transpose back) -> CopyOut
```

##### 源码等价伪码

```text
for nc in core_owned_NC:
  DataCopyPad every input row to xUB[row*ubFactor]
  for top row i:    xUB[(2*top-i),:] += xUB[i,:]
  for bottom row i: xUB[(Hi-2*bottom-1+i),:] += xUB[(Hi-1-i),:]
  compact rows [top, top+Ho) to front
  transpose 16x16 blocks so W becomes contiguous
  fold left/right mirror columns with Vector Add
  compact columns [left,left+Wo), transpose back
  DataCopyPad Ho rows of Wo valid elements to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `(ncOffset+loop)*Hi*Wi + row*Wi` | Hi×Wi | FP32/ND | caller | CopyIn | caller | 行 DataCopyPad |
| xInQueue | 0 | `128*ubFactor*4` B | FP32, row-major | CopyIn | Compute | pipe/单 NC | depth 1 |
| transposeBuf | 0 | `128*ubFactor*4` B | FP32 transposed | Compute | Compute | pipe/单 NC | TBuf 反复转置 |
| yOutQueue | 0 | `128*ubFactor*4` B | FP32 | Compute | CopyOut | pipe/单 NC | depth 1 |
| y GM | `(ncOffset+loop)*Ho*Wo + row*Wo` | Ho×Wo | FP32/ND | CopyOut | framework | executor output | 仅写有效 Wo |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["xInQueue depth=1 · 128·F·4 B"]:1
  trans["transposeBuf · 128·F·4 B"]:1
  outQ["yOutQueue depth=1 · 128·F·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3 * 128 * ubFactorElement * 4 bytes
Host: ubFactorElement=floor_align(floor_align(canUseUb/3/128,32)/4,16)
workspace_body = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xInQueue | CopyIn | Compute | 下一 NC | DeQue/FreeTensor | Compute 完成后才 Free |
| transposeBuf | H compact/transpose | W fold/transpose-back | 同 NC 后续阶段 | 同一 Vector 管线顺序 | 不与 queue 别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | queue EnQue/DeQue | MTE2 | Compute | x tile | `DataCopyPad` 补 0 |
| H/W 累加 | Vector `Add` | x/transpose | 后续 compact | FP32 UB | 仅有效 mirror 范围 |
| UB→GM | queue | Compute | MTE3 | y tile | 逐行 `outWidth*sizeof(float)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v4_grad_tiling.cpp` | `GetTilingKey/SplitUb` | mini 谓词与 3-way UB |
| Kernel entry/router | `pad_v4_grad.cpp` | key 1000 | FP32 class 映射 |
| Device body/helper | `pad_v4_grad_mini_h_w_pad.h` | `Compute/Process` | H/W 累加、transpose、tail |

### Route R-002：H-only

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v4_grad_tiling.cpp` | `GetTilingKey` | W pad=0，H pad≠0，Hi>64 | 1110 |
| Router | `pad_v4_grad.cpp` | key 1110 | FP32 | `PadV4GradPadH<float>` |

#### Device 数据流

##### 函数级调用链

```text
pad_v4_grad -> PadV4GradPadH<float>::Process
  -> CopyGm2UB(two mirror rows) -> ComputeHGrad(Vector Add) -> CopyOut
  -> CopyIn(mid/edge rows) -> Compute(DataCopy) -> CopyOut
```

##### 源码等价伪码

```text
for nc; for width_tile:
  for i<top:    y[top-i,:] = x[i,:] + x[2*top-i,:]
  for i<bottom: y[Ho-bottom-1+i,:] = x[Hi-2*bottom-1+i,:] + x[Hi-1-i,:]
copy untouched middle and first/last rows in 2*F tiles; tail uses remaining elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xInQueue | two regions `[0,F)`/`[F,2F)` | depth2×`2F*4` B | FP32 | CopyGm2UB | Add | pipe/tile | double queue |
| yOutQueue | 0 | depth2×`2F*4` B | FP32 | Add/Copy | CopyOut | pipe/tile | double queue |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["xInQueue depth=2 · each 2F·4 B"]:1
  outQ["yOutQueue depth=2 · each 2F·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*2F*4 + 2*2F*4 = 32F bytes
workspace_body = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | CopyGm2UB | ComputeHGrad | 下一 tile | EnQue/DeQue/FreeTensor | depth=2 算入容量 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mirror | `Add(y,x0,x1,calCount)` | MTE2 | V | pair tile | FP32；`calCount` 尾宽 |
| copy | DataCopyPad | queue | MTE3 | valid bytes | `copyCount` 不对齐安全 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v4_grad_tiling.cpp` | key 1110 | H-only 谓词 |
| Kernel entry/router | `pad_v4_grad.cpp` | `PadV4GradPadH<float>` | class 选择 |
| Device body/helper | `pad_v4_grad_h_pad.h` | `ComputeHGrad/Process` | 成对行累加 |

### Route R-003：W-only

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v4_grad_tiling.cpp` | `GetTilingKey` | H pad=0，W pad≠0 | 1101 |
| Router | `pad_v4_grad.cpp` | key 1101 | FP32 | `PadV4GradPadW<float>` |

#### Device 数据流

##### 函数级调用链

```text
Process -> CopyAndComputeWGrad(boundaries) -> CopyMiddleToOut
  small/large edge -> CopyGm2UB -> ComputeWGrad -> workspace -> CopyInFromWs -> y
```

##### 源码等价伪码

```text
for owned row:
  load left window and right window
  for i<left:  center[2*left-i] += leftPad[i]
  for i<right: center[Wi-2*right-1+i] += rightPad[Wi-1-i]
  resolve overlap of both edge windows
  write valid left/right result; DMA-copy untouched middle
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y queues | route-local | 由 `InitBuffer`, 最大按 F | FP32 | MTE2/V | V/MTE3 | pipe/tile | depth2 |
| per-core workspace | `blockIdx*2*partWs` | `2*partWs*4` B | FP32 | edge compute | workspace reread | executor/kernel | MTE3_MTE2 event |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["edge/middle input queues"]:1
  outQ["edge/middle output queues"]:1
  ws["GM workspace 2·partWs·4 B/core"]:1
```

##### 字节账与复用顺序

```text
workspace_total = blockNum * 2 * partWs * 4 bytes
partWs = (left<16 && right<16) ? 64 : align(2*max(left,right)+1,8)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-core workspace | `Copy2Ws` | `CopyInFromWs` | 下一 owned row | `MTE3_MTE2` Set/Wait | blockIdx slice 独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| edge scalar | `MTE2_S`/`S_MTE3` | DMA/scalar | scalar/DMA | LocalTensor value | FP32 add；overlap 显式累加 |
| workspace | `MTE3_MTE2` | ws write | ws read | per-core slice | middle count<=0 时跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v4_grad_tiling.cpp` | `FillTilingData` | W-only workspace |
| Kernel entry/router | `pad_v4_grad.cpp` | key 1101 | W class |
| Device body/helper | `pad_v4_grad_w_pad.h` | `CopyAndComputeWGrad/Process` | 边界重叠与 workspace |

### Route R-004：small-H/large-W

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v4_grad_tiling.cpp` | `GetTilingKey` | Hi<=64, Wi>112 | 1100 |
| Router | `pad_v4_grad.cpp` | key 1100 | FP32 | small-H/large-W body |

#### Device 数据流

##### 函数级调用链

```text
Process -> CopyGm2UB(W tile over all H) -> ComputeHGrad
  -> CopyOut2Workspace(boundary tiles) -> MTE3_MTE2
  -> implTransposeAndCompute(left/right) -> CopyWs2UB -> copy middle -> y
```

##### 源码等价伪码

```text
for NC and W tile:
  load <=64 rows; fold top/bottom; write Ho×tile to workspace
transpose 16-column left/right blocks, fold W mirrors, write boundary columns
copy remaining middle columns from workspace to y with tail count
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| three queues | 0 | each `64*F*4` B | FP32 | Copy/H fold/transpose | next stage | pipe/tile | depth1 |
| workspace | `blockIdx*alignWi*64` | `alignWi*64*4` B/core | FP32 H-folded | ComputeHGrad | W fold/middle copy | executor/kernel | per-core |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["xIn 64F·4"]:1
  trans["transpose 64F·4"]:1
  outQ["yOut 64F·4"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*64*F*4 bytes
workspace_total = blockNum*alignWidth*64*4 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| workspace tile | `CopyOut2Workspace` | `CopyWs2UB`/middle | next NC | `MTE3_MTE2` | 单核 slice |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| H→W | `MTE3_MTE2` | workspace write | transpose/read | per-core ws | FP32 |
| W tail | `copyCount=min(F,remain)` | tile | y | valid W | DataCopyPad |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v4_grad_tiling.cpp` | key 1100 | H/W guard/UB/workspace |
| Kernel entry/router | `pad_v4_grad.cpp` | key 1100 | class mapping |
| Device body/helper | `pad_v4_grad_small_h_large_w_pad.h` | `ComputeHGrad/implTransposeAndCompute/Process` | 两阶段折叠 |

### Route R-005：large-H/small-W

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v4_grad_tiling.cpp` | `GetTilingKey` | Hi>128, Wi<=128 | 1010 |
| Router | `pad_v4_grad.cpp` | key 1010 | FP32 | large-H/small-W body |

#### Device 数据流

##### 函数级调用链

```text
Process -> CopyGm2UB(16-row top/bottom blocks) -> ComputeHGrad
  -> CopyOut2Ws -> MTE3_MTE2 -> CopyGmAndWorkspace2UB2
  -> implTransposeAndCompute(W fold) -> CopyOut2Gm
```

##### 源码等价伪码

```text
for NC:
  fold top and bottom 16-row boundary blocks, save partials in per-core workspace
  iterate inner H blocks of F rows, combining GM interior and saved boundary
  transpose width<=128, fold left/right reflection, transpose back and write valid Ho×Wo
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| three queues | 0 | each `F*128*4` B | FP32 | MTE2/V | V/MTE3 | pipe/H tile | depth1 |
| workspace | `blockIdx*alignWi*64` | `alignWi*64*4` B/core | FP32 boundary rows | first stage | later H blocks | executor/kernel | per-core |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["xIn F·128·4"]:1
  trans["transpose F·128·4"]:1
  outQ["yOut F·128·4"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*F*128*4 bytes
workspace_total = blockNum*alignWidth*64*4 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| top/bottom ws | `CopyOut2Ws` | `CopyGmAndWs2UB*` | next NC | MTE3_MTE2 + per-core slice | 完成消费后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| boundary handoff | MTE3_MTE2 | boundary result | H block combine | ws | FP32 |
| last H block | `transCount=min(F,remain)` | GM/ws | transpose | valid rows | output only valid Ho |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v4_grad_tiling.cpp` | key 1010 | shape guard |
| Kernel entry/router | `pad_v4_grad.cpp` | key 1010 | class mapping |
| Device body/helper | `pad_v4_grad_large_h_small_w_pad.h` | `ComputeHGrad/implTransposeAndCompute/Process` | H 分块和 W fold |

### Route R-006：generic-HW

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `pad_v4_grad_tiling.cpp` | `GetTilingKey` | 前五路均不命中 | 1111 |
| Router | `pad_v4_grad.cpp` | key 1111 | FP32 | generic HW body |

#### Device 数据流

##### 函数级调用链

```text
Process -> CopyGm2UB(16 rows × W tile) -> ComputeHGrad
  -> CopyOut2Workspace -> MTE3_MTE2
  -> CopyGmAndWorkspace2UB1/2 -> implTransposeAndCompute(left/right)
  -> CopyIn(workspace middle) -> compute(copy) -> CopyOut
```

##### 源码等价伪码

```text
for NC:
  stage top/bottom 16-row partials to per-core workspace
  for W tile and each boundary/interior H region:
    merge GM rows with workspace partials
    transpose 16x16, fold reflected W columns, transpose back
    write only (16-left/right) valid columns and valid H rows
  copy workspace-backed middle ranges with final W/H tails
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xInQueue | two slots | depth2×`16F*4` B | FP32 | MTE2 | Compute | pipe/tile | double buffer |
| yOutQueue | two slots | depth2×`16F*4` B | FP32 | Compute | MTE3 | pipe/tile | double buffer |
| transposeQue | 0 | `16F*4` B | FP32 | transpose | fold | pipe/tile | depth1 |
| workspace | `blockIdx*alignWi*64` | `alignWi*64*4` B/core | FP32 partials | H stage | W/H merge | executor/kernel | per-core |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["xIn depth2 · each 16F·4"]:2
  trans["transpose 16F·4"]:1
  outQ["yOut depth2 · each 16F·4"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*16F*4 + 2*16F*4 + 16F*4 = 320F bytes
workspace_total = blockNum*alignWidth*64*4 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slots | CopyGm/Compute | Compute/CopyOut | parity-next tile | queue ownership | depth2 不提前 Free |
| workspace slice | H stage | W/H merge | next NC | MTE3_MTE2 | blockIdx 独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| queue pipeline | EnQue/DeQue | MTE2/V | V/MTE3 | parity slot | FP32 |
| ws handoff | `MTE3_MTE2` | partial write | reread | per-core ws | H/W tail 用实际 count |
| transpose | 16×16 TransData | local tile | fold/back | transpose queue | 未写 lane 不搬出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `pad_v4_grad_tiling.cpp` | fallback key 1111 | 5-way UB/workspace |
| Kernel entry/router | `pad_v4_grad.cpp` | key 1111 | generic class |
| Device body/helper | `pad_v4_grad_h_w_pad.h` | `ComputeHGrad/implTransposeAndCompute/Process` | 双边两阶段累加 |

## 算子特定优化

### Trick T-001：mini shape 整面驻 UB 并转置后复用 Vector Add

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：FP32 core，`Hi<=128 && Wi<=112`
- Golden 来源：[`conversion/pad_v4_grad/op_kernel/pad_v4_grad_mini_h_w_pad.h:45-58`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v4_grad/op_kernel/pad_v4_grad_mini_h_w_pad.h#L45-L58)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
} else {
    // buffer size (128, this->ubFactorElement)
    pipe->InitBuffer(xInQueue, 1, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
    pipe->InitBuffer(yOutQueue, 1, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
    pipe->InitBuffer(transposeBuf, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
}
```

#### 机制与设计原因

将最多 128 行的整个 NC 面搬入 UB，先在行主序上合并 H 镜像，再以 16×16 `TransData` 使 W 镜像变为连续 Vector Add，最后转回。这避免对每个 W 镜像元素做零散 GM 访问，同时不使用 GM workspace；代价是 `3*128*F*sizeof(T)` UB，因而 Host 以明确的 H/W 上界和 3-way UB 公式保护。

#### 迁移边界

- 适用：两个空间轴都能一次驻 UB 的 reflect-gradient。
- 前提：A2/A3、FP32 core、H<=128、W<=112、`ubFactorElement>0`。
- 失效：超界时必须转 R-002..R-006，不得沿用整面分配。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的反例是在大 shape 上继续按 128 行整面分配：Host `GetTilingKey` 会转到分 H/W 或 workspace route，否则 UB 公式不再成立。

## 模板与实现对比

| 对比维度 | R-001 mini | R-002 H | R-003 W | R-004 smallH | R-005 largeH | R-006 generic | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | H<=128,W<=112 | Wpad=0,H>64 | Hpad=0 | H<=64,W>112 | H>128,W<=128 | fallback | 优先最专用局部性 |
| 模板与 dtype | 全部 FP32 | FP32 | FP32 | FP32 | FP32 | FP32 | API 先升精度 |
| 循环与分核 | NC | NC | NC×H | NC×Wtile | NC×Htile | NC×H/Wtile | 按未 pad 轴分解 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3×128F | 2 个 depth2 queue | queues+scalar | 3×64F | 3×128F | depth2+transpose | 固定行/列块 |
| 内存总占用与复用 | 无 ws | 无 ws | 2*partWs/core | alignW*64/core | 同左 | 同左 | 大 shape 用 GM 交接 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Pad+Transpose+Add | Pad+Add | Pad+scalar/Add | Pad+Transpose+Add | 同左 | 同左 | 无 SIMT/NDDMA |
| 同步、精度与 tail | queue,FP32 | queue,FP32 | hard-event | ws event | ws event | queue+ws event | 全部 valid-byte tail |
| fallback/失败边界 | 超 shape 下跳 | 谓词不符下跳 | 下跳 | 下跳 | 下跳 | 无 | UB<=0 统一失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 API 支持，PadV4Grad `ascend910b` | 公开 API 支持，PadV4Grad `ascend910_93` | 公开 API 支持，但实际改走 RegBase `PadV3Grad`，见独立 `_950` 卡 |
| Host/Launcher | 共用 PadV4 selector/Tiling | 同 A2 | `IsRegBase()` 分支 |
| Kernel/资源 | 六个 AIV body | 同 A2 | SIMT + Vector MicroAPI 五 route，不复用本 body |

## 限制与失败边界

- API 限制 rank 3/4、ND、dtype 一致、padding 长度 4 且每边小于原 H/W；`gradOutput` shape 必须等于 `self` 加 padding。
- FP16/BF16 可达 AscendC 主节点前后有 Cast；DOUBLE/COMPLEX64/COMPLEX128 公开 API 转 AICPU，其 Device body 在固定 ops-math 仓不可见，本卡不将它们补成 AscendC route。
- `GetInputInfo` 将 shape/padding 窄化到 uint32/int32，固定实现未见逐字段溢出防护；超过表示域不可迁移。
- Host 只显式检查 `ubFactorElement>0`；workspace 分配和框架 Cast/ViewCopy 切片由 executor 管理，其内部 offset/event 固定仓不可见。
- 文档警告 `gradOutput` 元素数超过 `300*1024*1024` 有超时风险；本次未做性能验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.h` | `aclnnReflectionPad2dBackward*` | 两段式 ABI |
| 入口实现 | `conversion/pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.cpp` | `commonPad2dBackward` | 校验、Cast、L0、ViewCopy |
| Host selector | `conversion/pad_v3_grad/op_api/padv3grad.cpp` | `PadV3Grad/IsPadV4GradAicoreSupport` | A2/A3 选 PadV4Grad |
| Host/Tiling/Launcher | `conversion/pad_v4_grad/op_host/pad_v4_grad_tiling.cpp` | `Tiling4PadV4Grad` | route/POD/UB/workspace/blockDim |
| POD | `conversion/pad_v4_grad/op_host/pad_v4_grad_tiling.h` | `PadV4GradTilingData` | ABI 字段 |
| Kernel entry/body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad.cpp` | `pad_v4_grad` | key router/AIV-only |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_base.h` | `PadV4GradBase::Init` | 分核/地址基类 |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_mini_h_w_pad.h` | `PadV4GradPadMiniHW` | R-001 |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_h_pad.h` | `PadV4GradPadH` | R-002 |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_w_pad.h` | `PadV4GradPadW` | R-003 |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_small_h_large_w_pad.h` | `PadV4GradPadSamllHLargeW` | R-004 |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_large_h_small_w_pad.h` | `PadV4GradLargeHSmallW` | R-005 |
| Kernel body | `conversion/pad_v4_grad/op_kernel/pad_v4_grad_h_w_pad.h` | `PadV4GradPadHW` | R-006 |
| 构建接线 | `conversion/pad_v4_grad/op_host/pad_v4_grad_def.cpp` | `PadV4Grad` | 910b/910_93 OpDef |
| 构建接线 | `conversion/pad_v4_grad/op_host/config/ascend910b/pad_v4_grad_binary.json` | `PadV4Grad` binaries | A2 dtype/padding binary 变体 |
| 构建接线 | `conversion/pad_v4_grad/op_host/config/ascend910_93/pad_v4_grad_binary.json` | `PadV4Grad` binaries | A3 dtype/padding binary 变体 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 20×32-bit + 1×64-bit 字段 | 顺序、单位、workspace byte→element 转换 | `PadV4GradTilingData/Base::Init` |
| 调用顺序 | ACLNN→Cast→PadV4Grad→Cast/ViewCopy | rank3 Unsqueeze/Squeeze 和 FP32 升精度顺序 | `commonPad2dBackward` |
| workspace/内存 | R-003 和 R-004..006 独立 per-core slice | offset 必须含 `blockIdx`，extent 不得小于 Host 公式 | `FillTilingData`, route `Process` |
| 同步 | queue 所有权 + MTE3_MTE2 hard event | workspace 写后读不可省略 wait | `pad_v4_grad_*_pad.h` |
| 精度与尾块 | ACLNN FP16/BF16 核心 FP32，DataCopyPad 按 valid byte | 不可把 padding lane 写回 GM | `commonPad2dBackward`, `CopyOut*` |
| 失败边界 | UB<=0、shape/padding/dtype 不合法失败 | 不得把 AICPU fallback 补为 PadV4Grad | `Tiling4PadV4Grad`, `PadV3Grad` |