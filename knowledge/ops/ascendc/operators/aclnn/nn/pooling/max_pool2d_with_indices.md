---
type: Operator
title: "MaxPool2dWithIndices A2/A3 ACLNN 算子知识"
description: "说明 aclnnMaxPool2dWithIndices 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool2d_with_indices.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_api/max_pool3d_with_argmax_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_tiling_bigkernel.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2_no_expand_indices.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_big_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/config/ascend910b/max_pool3d_with_argmax_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/config/ascend910_93/max_pool3d_with_argmax_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
created_at: "2026-08-25T18:19:46Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# MaxPool2dWithIndices A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMaxPool2dWithIndicesGetWorkspaceSize` + `aclnnMaxPool2dWithIndices`，公开 ACLNN 两段式接口 |
| 计算语义 | 对每个 N/C 与输出 `(oh,ow)`，在有效的 `kH×kW` 窗口内取最大值，并返回展平到输入 H×W 平面的索引 |
| 输入输出 | `self/out`: FP16/BF16/FP32、3D CHW 或 4D NCHW；`indices`: INT32/INT64；A2/A3 不支持 NHWC，dilation 仅 1 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；R-001 为 transpose+Vector 比较路线，R-002 为逐输出窗口分段归约路线 |
| Kernel 路线 | R-001 NoExpandIndices（key 300001/300002）；R-002 BigKernel（key 311110/311111/311112） |
| 核心资源 | GM；R-001 单深度 TQue、多个 TBuf/mask；R-002 双缓冲输入 Queue、FP32 cast/reduce/index TBuf；框架 workspace 含 16 MiB system workspace，INT64 indices 另经框架 Cast |
| 来源闭包 | `source_closed`：公开 ACLNN→非 RegBase API 分支→2D 到 NCDHW(D=1) 适配→MaxPool3DWithArgmaxV2 launcher→两条 Host 可达 AIV route→entry/body/config 均闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `ih=oh*sH-pH+mh*dH`、`iw=ow*sW-pW+mw*dW`，则 `out[n,c,oh,ow]` 是所有合法 `(ih,iw)` 上的最大输入值，`indices[n,c,oh,ow]=ih*W+iw`。A2/A3 API 校验把 `dH=dW=1` 固定下来。输出尺寸使用向下取整；`ceilMode=true` 时分子加 `stride-1`，且若末窗口起点落到右侧 padding 区域则再减 1。3D CHW 输入在 API 内补 N=1。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| self | CHW/NCHW，3 或 4 维 | FP16/BF16/FP32 | 非空；A2/A3 不支持 NHWC；非连续输入先 `Contiguous` |
| kernelSize | 长度 1 或 2 | INT64 数组 | 元素 > 0；单值复制到 H/W |
| stride | 长度 0、1 或 2 | INT64 数组 | 空数组取 kernelSize；元素 > 0 |
| padding | 长度 1 或 2 | INT64 数组 | `0 <= pad <= kernel/2`；窗口外视为 `-Inf` |
| dilation | 长度 1 或 2 | INT64 数组 | A2/A3 两元素都必须为 1 |
| ceilMode | 标量 | BOOL | 控制输出尺寸取整与末窗口裁剪 |
| out | 与推导 shape 相同 | 与 self 相同 | 非连续输出由 `ViewCopy` 提交 |
| indices | 与 out 相同 | INT32/INT64 | Device 实体输出 INT32；请求 INT64 时 executor 插入 `Cast`，再 `ViewCopy` |

### 入口与版本

固定仓公开文档明确列出 A2/A3 支持。`aclnnMaxPool2dWithIndicesGetWorkspaceSize` 在 `IsRegbase()==false` 时调用 `ExecMaxPool3dWithIndicesGetWorkspaceSize`；这不是另一公开语义，而是当前 2D API 的实际 A2/A3 body：输入被补成 NCDHW，属性变为 `{1,kH,kW}/{1,sH,sW}/{0,pH,pW}/{1,1,1}`。同架构没有更新的公开版本分支。

## Host/Launcher 执行流程

### 参数校验与前处理

顺序为：非空指针→当前架构 dtype 表→self/out 同 dtype 与 format→3/4 维→属性长度和值→输出 shape/out/indices shape。空 `self` 或 `out` 返回 workspace 0 和空执行器路径。非空时 `Contiguous(self)`；3D/4D 分别补成 `[1,C,1,H,W]`/`[N,C,1,H,W]`；L0 分配连续 out/INT32 indices，添加 `MaxPool3DWithArgmaxV2` AICore launcher；随后 squeeze D 与可选 N，必要时 Cast indices，最后分别 `ViewCopy` 到用户输出。

### 参数、Workspace 与生命周期

executor 拥有 Contiguous/Unsqueeze/Squeeze/Cast/ViewCopy 的临时张量及 launcher；第一段接口以 `GetWorkspaceSize()` 返回总 workspace，第二段在同一 `stream` 上执行整个列表。Device 主节点 R-001/R-002 不使用用户算法 workspace，Host 基类仍声明 16 MiB system workspace；INT64 Cast 与非连续 copy 所需空间由 executor 合并计入。具体框架临时 slice 的 offset/extent 在固定仓生成框架内不可见。

### 启动图与失败传播

```text
ACLNN executor (同一 stream)
  Contiguous / shape-view nodes
  -> MaxPool3DWithArgmaxV2(x[N,C,1,H,W], attrs, y5d, idx5d<int32>)
       -> 单个 max_pool3d_with_argmax_v2 AIV kernel
  -> optional Cast(idx5d, int64)
  -> Squeeze/ViewCopy(out, indices)
```

任一校验、临时 tensor、infer-shape 或 launcher 添加失败立即返回并阻断后续节点；`CommonOpExecutorRun` 保持 executor 中的 stream 顺序。框架临时张量由 executor 持有到执行结束，用户 out/indices 的 storage 由调用方持有。

## Route 与参数设计

### 分核与资源计算

共享输入恒有 `D=1,kD=sD=dD=1,pD=0`。R-001 先令 `ncFactor=min(N*C, (32/inputCastBytes)*16)`，`wy/hy/dy` 从 1 开始按 UB 二分放大；最小 tile 字节为两个 input 区、两个 max 区、一个 indices 区、两个 indices-temp 区和三个 mask 区之和，必须 `<= UB_SIZE`。R-002 令 `totalIdx=N*C*Hout*Wout`、`blockFactor=totalIdx/coreNum`、`blockTail=totalIdx%coreNum`，实际核数为 `min(totalIdx,coreNum)`，逐输出分段装入窗口。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 NoExpandIndices | 300001 无 pad；300002 有 pad | priority 10；非 RegBase；D/H/W dilation 全为 1；以最小 `dy=hy=wy=1` 计算的 `tmpTotalSize <= UB_SIZE` | R-002 |
| 2 | R-002 BigKernel | 311110 FP32；311111 FP16；311112 BF16 | priority 11；非 RegBase；D/H/W dilation 全为 1；R-001 未命中 | 失败（2D API 已保证 dilation=1，因此本入口闭合于此） |

```text
if non-RegBase && dilationD/H/W == 1 && minimum_no_expand_tile_bytes <= UB_SIZE:
    R-001, key = any_pad ? 300002 : 300001
else if non-RegBase && dilationD/H/W == 1:
    R-002, key = dtype(fp32/fp16/bf16) -> 311110/311111/311112
else:
    tiling failure  # ACLNN A2/A3 在更早处已拒绝 dilation != 1
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 300001/300002 | A2/A3，NCDHW(D=1)，dilation=1，最小 tile 可放 UB；key 由六侧 pad 是否非零决定 | 单 AIV kernel | `MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, compute_t, PAD_DISABLE/ENABLE>` | input/max/indices TQue depth=1；transpose、index、mask TBuf；Queue + pipe barrier | former/tail core；nc/dy/hy/wy tail；不满足 UB 去 R-002 |
| R-002 | 311110/1/2 | A2/A3，dilation=1，R-001 不可用，key 按 dtype | 单 AIV kernel | `MaxPool3DWithArgmaxBigKernel<T1,T2,false,uint16_t>` | input Queue depth=2；cast/reduce/index buffers；event 与 pipe barrier | 每输出窗口按连续 DHW/HW/W 分段；最后段和 blockTail 显式处理 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 guard | `!IsRegbaseSocVersion(context_)` | legacy POD/body 仅非 RegBase 实例化 | 两条 | 950 V3 分支 | 两个 `IsCapable` |
| 最小 UB tile | `tmpTotalSize <= UB_SIZE` | 即使一个输出也需 input/max/index/temp/mask 全部区 | R-001 | R-002 | `NoExpandIndicesTiling::IsCapable/DoBufferCalculate` |
| dilation | D/H/W 都等于 1 | R-001 的 transpose/连续索引与 R-002 的窗口 body 前提 | 两条 | ACLNN 参数错误 | `CheckShape`、两个 `IsCapable` |
| INT64 indices | 用户 dtype INT64 | legacy Device ABI 固定 int32 | executor 增 Cast | Cast 失败即 API 失败 | `ExecMaxPool3dWithIndicesGetWorkspaceSize` |
| dtype 窄化 | R-002 shape/attrs 写入 uint32 | kernel 地址字段按 32 位消费；固定源码未见 Host 溢出保护 | R-002 | 无显式 fallback | `BigKernelTiling::SetTilingData` |

### POD/ABI

宏生成结构的 `sizeof/alignof/offsetof` 在固定仓不可见；字段顺序即声明顺序。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| NoExpand | `nc,dx,hx,wx` | uint64/64 | 顺序字段 | 元素 | 输入 N*C,D,H,W | `Init/CalCurIdxData` | 2D 时 dx=1 |
| NoExpand | `kd,kh,kw` | uint64/64 | 顺序字段 | 元素 | kernel `{1,kH,kW}` | `PoolMaxAndIndices` | 正数 |
| NoExpand | `sd,sh,sw` | uint64/64 | 顺序字段 | 元素 | stride `{1,sH,sW}` | 地址推进 | 正数 |
| NoExpand | `pf,pb,pl,pr,pt,pd` | uint64/64 | 顺序字段 | 元素 | 六侧 pad | padding/body | 2D front/back=0 |
| NoExpand | `dy,hy,wy` | uint64/64 | 顺序字段 | 元素 | 输出 shape | tile/GM offset | 正数 |
| NoExpand | `ncFactor,ncTail,ncOuter` | uint64/64 | 顺序字段 | 元素/循环 | Host split | core tile loop | tail 0 被改为 factor |
| NoExpand | `dyFactor,dyTail,dyOuter` | uint64/64 | 顺序字段 | 元素/循环 | Host split | tile loop | 2D dy=1 |
| NoExpand | `hyFactor,hyTail,hyOuter` | uint64/64 | 顺序字段 | 元素/循环 | Host split | tile loop | tail 保护 |
| NoExpand | `wyFactor,wyTail,wyOuter` | uint64/64 | 顺序字段 | 元素/循环 | Host split | tile loop | tail 保护 |
| NoExpand | `blockFactor,blockTail,totalIdx,coreNums` | uint64/64 | 顺序字段 | base-block/core | `DoBlockTiling` | `Init/Process` | former/tail core |
| NoExpand | `inputBufferSize` | uint64/64 | 顺序字段 | bytes | `DoBufferCalculate` | `InitBuffer` input queue/TBuf | 已 cache-align |
| NoExpand | `outputMaxBufferSize` | uint64/64 | 顺序字段 | bytes | 同上 | max queue/TBuf | 已 cache-align |
| NoExpand | `outputIndiceBufferSize` | uint64/64 | 顺序字段 | bytes | 同上 | indices queue/TBuf | INT32 |
| NoExpand | `indiceTempBufferSize` | uint64/64 | 顺序字段 | bytes | 同上 | update/template TBuf | 两份 |
| NoExpand | `maskBufferSize` | uint64/64 | 顺序字段 | bytes | 同上 | gt/ge/or masks | 三份，or 条件分配 |
| BigKernel | `inputShapes[3],outShapes[3]` | uint32/32 | 数组顺序字段 | 元素 | `SetTilingData` 强转 | `Prepare` | 存在 64→32 窄化 |
| BigKernel | `kD,kW,kH,sD,sW,sH,pD,pW,pH,dD,dW,dH` | uint32/32 | 顺序字段 | 元素 | inputData attrs | copy/compute/address | 2D 固定 D 轴 |
| BigKernel | `blockFactor,blockTail,totalIdx,coreNums` | uint64/64 | 顺序字段 | 输出/core | `DoUBTiling` | `Init/Process` | former/tail core |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 300001/300002 | `max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp` `IsCapable/DoOpTiling` | `MaxPool3DWithArgmaxV2NoExpandIndicesTilingData` | `op_kernel/max_pool3d_with_argmax_v2.cpp` key branches | `max_pool3d_with_argmax_v2_no_expand_indices.h` | `ascend910b/...binary.json` + module CMake |
| R-002 | 311110/1/2 | `max_pool3d_with_argmax_v2_tiling_bigkernel.cpp` | `MaxPool3DWithArgmaxV2BigKernelTilingData` | 同一 entry key branches | `max_pool3d_with_argmax_big_kernel.h` | 同上 |

## Kernel 实现

### 公共执行流程

两条 route 都只启动 `max_pool3d_with_argmax_v2(x,y,indices,workspace,tiling)` 一个 AIV 节点。router 解析对应 POD 并按 dtype 实例化 body；每核取得连续输出 base-block，读取窗口、按源码 NaN/tie 规则更新最大值与展平 index，最后分别写 y 与 int32 indices。用户 INT64 与非连续 tensor 属于 executor 后处理。

### Route R-001：NoExpandIndices transpose-vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_max_pool2d_with_indices.cpp` | `CheckShape` | 非 RegBase 要求 dilationH/W==1 | 继续或参数错误 |
| API | 同上 | `ExecMaxPool3dWithIndicesGetWorkspaceSize` | 构造 D=1 的 NCDHW 与 3D attrs | launcher |
| Tiling | `op_host/max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp` | `IsCapable` | 非 RegBase、dilation 全 1、`tmpTotalSize<=UB_SIZE` | 300001/300002；否则 R-002 |
| Router | `op_kernel/max_pool3d_with_argmax_v2.cpp` | `max_pool3d_with_argmax_v2` | key 300001/300002 | PAD_DISABLE/ENABLE body |

#### Device 数据流

##### 函数级调用链

```text
MaxPool3DWithArgmaxV2 launcher
  -> max_pool3d_with_argmax_v2
    -> NoExpandIndices::Init
    -> Process -> BaseCompute
      -> CopyInput -> optional BF16->FP32 -> padding -> Transpose
      -> PoolMaxAndIndices (Compare/Select/Max)
      -> transpose back -> optional FP32->BF16 -> CopyMaxOut/CopyIndicesOut
```

##### 源码等价伪码

```text
for baseBlock in core_begin .. core_end:
  derive nc/dy/hy/wy tile and valid tails
  copy padded GM window to inputQueue; BF16 route casts to FP32
  transpose NC and W tiles so vector lanes cover multiple channels/outputs
  for each output d,h and vector-W group:
    max = first window element; index = first valid flattened offset
    for k in 1 .. kd*kh*kw-1:
      gt = x[k] > max; nan = x[k] != x[k]
      update index by gt/NaN mask; max = Max(max,x[k])
  transpose output back; copy valid y and int32 indices tails to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `(nc*D*H*W)+d*H*W+h*W+w` | input shape | T1/NCDHW | caller | CopyInput | caller | DataCopyPad tail |
| inputQue/inputBuffer | tile local | `inputBufferSize` each | T2, transposed staging | CopyInput/Transpose | PoolMax | kernel/loop | Queue depth1 + one TBuf |
| maxQue/maxBuffer | tile local | `outputMaxBufferSize` each | T2 | PoolMax | CopyMaxOut | kernel/loop | Queue depth1 + one TBuf |
| indicesQue/indicesBuffer | tile local | `outputIndiceBufferSize` each | int32 | PoolMax | CopyIndicesOut | kernel/loop | Queue depth1 + one TBuf |
| indices update/template | 0 | `2*indiceTempBufferSize` | int32/float reinterpret | Init/compute | Select | kernel | distinct TBuf |
| gt/ge/or mask | 0 | `3*maskBufferSize` | uint8/uint16 mask | Compare | Select | compute phase | or buffer only when needed |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inputQue d=1 · inputBufferSize"]:1
  inB["inputBuffer · inputBufferSize"]:1
  outQ["maxQue+indicesQue · outputMax+outputIndice"]:1
  outB["max/index TBuf · same capacities"]:1
  idx["2×indiceTempBufferSize"]:2
  mask["3×maskBufferSize"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*inputBufferSize + 2*outputMaxBufferSize + 2*outputIndiceBufferSize
         + 2*indiceTempBufferSize + 3*maskBufferSize
UB_total <= UB_SIZE
dynamic_ub = N/A（AIV TPipe 静态切分）
workspace_total = 16 MiB system workspace + executor framework temporaries
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue tensor | CopyInput/PoolMax | DeQue compute/CopyOut | 下个 baseBlock | EnQue/DeQue/FreeTensor | depth1 串行阶段 |
| mask TBuf | Compare | Select | 下一窗口元素 | `PipeBarrier<PIPE_V>` | 当前 Select 完成后覆盖 |
| BF16 input/output reinterpret | DataCopy/Cast | FP32 compute/DataCopy | 当前 tile 结束 | MTE2→V、V→MTE3 event | 有效长度按 cast loop tail |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue + MTE2_V | DataCopyPad | Cast/Transpose | input tile | 非对齐尾由 pad 参数保护 |
| compare chain | PIPE_V barrier | Compare/Max | Select/下一 k | max/index/mask | BF16 升 FP32；FP16 原类型；NaN mask 单独处理 |
| UB→GM | V_MTE3 event | Cast/compute | DataCopyPad | output tile | BF16 `CAST_ROUND`；仅写 valid tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp` | `IsCapable/DoBufferCalculate` | 最小 UB gate 与所有 buffer 字节公式 |
| Kernel entry/router | `op_kernel/max_pool3d_with_argmax_v2.cpp` | key 300001/300002 | key 到 PAD 模板映射 |
| Device body/helper | `op_kernel/max_pool3d_with_argmax_v2_no_expand_indices.h` | `Init/BaseCompute/PoolMaxAndIndices` | transpose、Compare/Select/Max、tail 和字节分配 |

### Route R-002：BigKernel fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/max_pool3d_with_argmax_v2_tiling_bigkernel.cpp` | `IsCapable` | priority11、非 RegBase、dilation全1 | 按 dtype 取 311110/1/2 |
| Tiling | 同上 | `DoUBTiling` | `totalIdx/blockFactor/blockTail/coreNums` | 单 kernel grid |
| Router | `op_kernel/max_pool3d_with_argmax_v2.cpp` | key 311110/1/2 | FP32/FP16/BF16 实例 | BigKernel body |

#### Device 数据流

##### 函数级调用链

```text
max_pool3d_with_argmax_v2
  -> BigKernel::Init
  -> Process -> Prepare -> BaseCompute
    -> dhwCopyInput / hwCopyInput / wCopyInput
    -> Compute -> ReduceMax + NaN index selection
    -> UpdateMax across chunks
    -> CopyMaxOut + CopyIndicesOut
```

##### 源码等价伪码

```text
for out_idx in core_begin .. core_end:
  clip window against padding; select contiguous DHW, strided-HW, or strided-W copy
  for each chunk <= maxCount:
    DataCopyPad chunk; BF16 cast to FP32
    reduce maximum and index; if NaN exists retain source-defined last-NaN index
    restore chunk-local index to flattened input H*W index and update scalar aggregate
  write one value and one int32 index; process blockTail with one extra output on former cores
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQue | clipped window offset | `2*maxCount*sizeof(T1)` | T1 | copy helpers | Compute | kernel/chunk | depth2 |
| inputCastUB | 0 | `maxCount*sizeof(float)` | float | BF16 cast/index temp | Reduce | kernel | reused by index selection |
| maxUB/nanMaxIndexUB | 0 | 256 bytes each | T2/float | Reduce | scalar GetValue/CopyOut | kernel | event-protected |
| indicesInitUB | 0 | `maxCount*sizeof(float)` | float | NaNIndicesInit | index reduce | kernel | initialized once/reused |
| reduceWorkLocalUB | 0 | `maxCount/16*sizeof(float)` | float | ReduceMax | reduction | kernel | reduction scratch |
| maskNanUB | 0 | `maxCount/4` bytes | mask | Compare | Select | chunk | overwritten per chunk |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inputQue · depth=2 · 2*maxCount*sizeof(T1)"]:2
  cast["inputCast · maxCount*4"]:1
  reduce["reduceWork · maxCount/16*4"]:1
  idx["indicesInit · maxCount*4"]:2
  scalar["max+nanIndex · 512 bytes"]:1
  mask["maskNan · maxCount/4"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*maxCount*sizeof(T1) + maxCount*4 + 256
         + maxCount*4 + (maxCount/16)*4 + maxCount/4 + 256
UB_total <= Tiling 基类提供的 ubSizeNew（maxCount 在 body Init 中按 UB 派生）
dynamic_ub = N/A
workspace_total = 16 MiB system workspace；本 body 的 workspace 参数不读写
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputCastUB | BF16 Cast / index select | Reduce/ScalarCast | 下一 chunk | PIPE_V、V_S event | 当前 scalar index 已读取 |
| inputQue slot | copy helper | Compute | 两 chunk 后 | Queue depth2 | DeQue/FreeTensor 完成 |
| maskNanUB | Compare | Select | 下一 chunk | PIPE_V | Select 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| copy→vector | Queue/PipeBarrier | DataCopyPad | Cast/Reduce | input chunk | blockLen 对齐，最后块 pad |
| vector→scalar | V_S event | Reduce/Select | `GetValue` | max/index | BF16 升 FP32；NaN 单独选 index |
| scalar/vector→GM | S_MTE3/V_MTE3 | scalar index/cast | DataCopyPad | 单元素输出 | BF16 `CAST_RINT`，blockTail 分核 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/max_pool3d_with_argmax_v2_tiling_bigkernel.cpp` | `IsCapable/DoUBTiling/SetTilingData` | fallback 谓词、分核和 uint32 POD |
| Kernel entry/router | `op_kernel/max_pool3d_with_argmax_v2.cpp` | key 311110/1/2 | dtype 模板映射 |
| Device body/helper | `op_kernel/max_pool3d_with_argmax_big_kernel.h` | `Process/Compute/*CopyInput/UpdateMax` | 分段搬运、归约、NaN、同步和 tail |

## 算子特定优化

### Trick T-001：NC/W transpose 后同时计算多个输出

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：非 RegBase、dilation=1、最小/搜索后 tile UB 可容纳。
- Golden 来源：[`max_pool3d_with_argmax_v2_no_expand_indices.h:996-1012`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2_no_expand_indices.h#L996-L1012)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t v = 0; v < vecLoopNum; v++) {
    PipeBarrier<PIPE_V>();
    DataCopyParams InitMaxCopyParams;
    DataCopy(maxLocal[maxDhVecOffset], inputLocal[inputDhVecOffset], InitMaxCopyParams);
    PipeBarrier<PIPE_V>();
    for (uint64_t k = 1; k < kd_ * kh_ * kw; k++) {
```

#### 机制与设计原因

Host 令 NC 与输出 W tile 满足 transpose 对齐，Device 把原本按 NCDHW 分散的同窗口位置转换成连续 vector lane；内层 `k` 只推进窗口元素，`Compare/Select/Max` 同时更新一组输出，减少逐输出标量式地址计算并复用已搬入 UB 的窗口。

#### 迁移边界

- 适用：规则窗口归约且输出/批通道轴可按 transpose 基元对齐。
- 前提：dilation=1、完整 input/max/index/temp/mask 字节账不超过 UB。
- 失效：最小 tile 也放不下时转 R-002。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的可验证反例是忽略完整字节账，仅用逻辑输入 tile 判断 UB：会漏算 output、indices、两个 temp 和三个 mask，Host 因此明确以 `tmpTotalSize` 作 gate。

### Trick T-002：大窗口分段归约而不物化完整窗口

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：R-001 最小 tile UB 不满足且 dilation=1。
- Golden 来源：[`max_pool3d_with_argmax_big_kernel.h:644-668`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_big_kernel.h#L644-L668)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t dhwLoop = 0; dhwLoop < dhwLoops; dhwLoop++) {
    auto dataCount = dhwCopyInput(kernelOffset, curMaxCount);
    auto index = Compute(dataCount);
    UpdateMax(kernelOffset + index, maxValue, maxIndex);
    kernelOffset += curMaxCount;
}
```

#### 机制与设计原因

窗口按 `maxCount` 分段，局部 ReduceMax 后只保留 scalar 最大值与恢复后的全局 index，再读下一段；因此 UB 上界由 chunk 决定，而不是 `kH*kW` 完整窗口。连续 DHW、跨行 HW 和 W-strided 三种 copy helper避免把所有非连续窗口先物化为一个巨大张量。

#### 迁移边界

- 适用：单输出归约窗口大于可用 UB。
- 前提：可恢复 chunk-local index 到输入平面，且定义好相等值与 NaN 的 index 规则。
- 失效：dilation 非 1 不属于 A2/A3 API 有效域。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 NoExpandIndices | R-002 BigKernel | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | priority10，完整最小 tile 可放 UB | priority11，前者失败且 dilation=1 | 先用批量 vector tile，后用必达 chunk fallback |
| 模板与 dtype | PAD×T1/T2，BF16 compute FP32 | dtype key，BF16 compute FP32 | padding 与 dtype 决定实例 |
| 循环与分核 | base-block→nc/dy/hy/wy tile→kernel element | output→window chunk | 输出批量 vs 单输出分段 |
| Queue/TBuf/寄存器 | depth1 queues + transpose/index/mask TBuf | depth2 input + reduce/index TBuf | UB gate 不同 |
| 内存总占用与复用 | 完整显式 `tmpTotalSize` | chunk-size 上界 | 大窗口不物化 |
| 搬运/Vector API | DataCopyPad、Transpose、Compare/Select/Max | 三类 DataCopyPad、Cast、ReduceMax | 连续性与窗口大小 |
| 同步、精度与 tail | Queue/barrier；BF16 FP32；四轴 tail | Queue/event；BF16 FP32；chunk/block tail | 源码各自维护 |
| fallback/失败边界 | UB 不足去 R-002 | dilation!=1 才失败 | API 已提前禁止该失败域 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 支持，但见独立 `_950` 卡 |
| Host/Launcher | 非 RegBase，2D→NCDHW V2 | 与 A2 共用 DAV_2201 路线 | RegBase 直接 MaxPoolWithArgmaxV3 |
| Kernel/资源 | R-001/R-002 AIV | 同 A2 | Gather/BigKernel/NHWC/SIMT 等 arch35 实现 |

## 限制与失败边界

- A2/A3 不支持 NHWC，dilation 只能是 1；输入 NaN/-Inf 在公开文档中标为暂不支持，不能据 Device NaN 分支扩大公开有效域。
- legacy Device indices 固定 INT32；INT64 是 ACLNN executor 的 Cast 结果，不是本主 Kernel ABI。
- R-002 将若干 64 位 shape/attr 强转为 uint32，固定仓未见对应 ACLNN 溢出 guard；超过范围的行为不得推测。
- 框架 Contiguous/Cast/ViewCopy 的内部 Kernel、workspace slice offset 与 binary ABI 属于允许的框架边界；算子自有 selector/POD/body 已闭合。
- 未知 tiling key 在 router 中无输出分支；正常 Host selector 不产生未知 key。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices.md` | 产品支持表、参数表 | A2/A3 公开性与约束 |
| 入口实现 | `pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool2d_with_indices.cpp` | `aclnnMaxPool2dWithIndicesGetWorkspaceSize` | 平台分支、校验、executor 图 |
| L0 launcher | `pooling/max_pool3d_with_argmax_v2/op_api/max_pool3d_with_argmax_v2.cpp` | `MaxPool3DWithArgmaxV2Ncdhw` | 输出分配与 launcher 添加 |
| POD | `pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_tiling_base.h` | 两个 TilingData | ABI 字段 |
| Host/Tiling | `pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp` | `IsCapable/DoBufferCalculate` | R-001 gate/字节账 |
| Host/Tiling | `pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_tiling_bigkernel.cpp` | `IsCapable/DoUBTiling` | R-002 fallback/分核 |
| Kernel entry | `pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2.cpp` | `max_pool3d_with_argmax_v2` | key router 与 AIV_ONLY |
| Kernel body | `pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2_no_expand_indices.h` | `NoExpandIndices` | R-001 完整 body |
| Kernel body | `pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_big_kernel.h` | `BigKernel` | R-002 完整 body |
| 构建接线 | `pooling/max_pool3d_with_argmax_v2/op_host/config/ascend910b/max_pool3d_with_argmax_v2_binary.json` | `MaxPool3DWithArgmaxV2` entries | A2 binary dtype/format 接线；A3 使用同代码族 |
| 构建接线 | `pooling/max_pool3d_with_argmax_v2/op_host/config/ascend910_93/max_pool3d_with_argmax_v2_binary.json` | `MaxPool3DWithArgmaxV2` entries | A3 binary dtype/format 接线 |
| 分发框架 | `common/inc/op_host/tiling_templates_registry.h` | `REGISTER_TILING_TEMPLATE`、`TilingRegistry::DoTilingImpl` | 数字越小优先、首个成功模板终止选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2D 可复用 D=1 的 V2 POD/body | D 轴字段仍必须写 1/0，字段位宽/顺序不变 | `ProcessAttrTo3DParams`、TilingData |
| 调用顺序 | Contiguous→view→main→Cast→ViewCopy | main 失败必须阻断输出 copy | `ExecMaxPool3dWithIndicesGetWorkspaceSize` |
| workspace/内存 | R-001 完整 tile 字节账；R-002 chunk 上界 | depth 和 temp/mask 不得漏乘 | 两个 `InitBuffer` |
| 同步 | Queue/barrier/event 模式可复用 | MTE/Vector/Scalar 依赖不可省 | 两个 body |
| 精度与尾块 | BF16 升 FP32、输出 round、valid-tail 写回 | NaN/index tie 顺序与 blockTail 不可改 | `PoolMaxAndIndices`、`Compute/UpdateMax` |
| 失败边界 | R-001→R-002 的优先级闭合 | 不得把后续 dormant legacy route写成 2D 可达 | 两个 `REGISTER_TILING_TEMPLATE` |