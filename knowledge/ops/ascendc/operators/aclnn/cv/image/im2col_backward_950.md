---
type: Operator
title: "Im2colBackward 950 ACLNN 算子知识"
description: "说明 aclnnIm2colBackwardGetWorkspaceSize/aclnnIm2colBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 SIMT 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/docs/aclnnIm2colBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_api/aclnn_im2col_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_api/aclnn_im2col_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_api/col2im.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/common/inc/op_api/aclnn_check.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_host/col2im_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_host/col2im_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_host/arch35/col2im_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_kernel/arch35/col2im_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_kernel/arch35/col2im_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_kernel/col2im.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_kernel/arch35/col2im_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_host/config/ascend950/col2im_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T20:47:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Im2colBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnIm2colBackwardGetWorkspaceSize` / `aclnnIm2colBackward`；固定仓只有这一组公开版本 |
| 计算语义 | `Col2im`：把 `gradOutput` 中各滑窗位置对输出像素的贡献累加回 `[N,C,H,W]`；是 Im2col 的反向/伴随数据变换，不与前向 Im2col 合并 |
| 输入输出 | `gradOutput` 为二维 `[C·kH·kW,L]` 或三维 `[N,C·kH·kW,L]` ND；`out` 为对应 `[C,H,W]` 或 `[N,C,H,W]` ND；950 支持 FP32/FP16/BF16 |
| 执行类型 | 算子自有单 Device 节点 `SIMT`，入口固定 `KERNEL_TYPE_AIV_ONLY`；1024 个 VF 线程/块，无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001：950 输出元素中心化 SIMT；三个 dtype key 共用控制流，FP16/BF16 以 FP32 累加 |
| 核心资源 | 输入/输出 GM、SIMT 标量/寄存器；无显式 TQue/TBuf/shared buffer；local memory 由 Host 设为 `UB-128 KiB`；workspace 仅为框架 LibApi 区且 entry 要求 user workspace 指针非空 |
| 来源闭包 | `source_closed`：公开 ACLNN、950 API 分支、L0、OpDef、InferShape、arch35 Tiling/POD/key、AIV-only SIMT entry/body、ascend950 binary 与构建接线闭合；通用 Contiguous/Cast/ViewCopy 与 executor 属框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输出空间为 `H=inputSize[0]`、`W=inputSize[1]`，卷积核为 `kH×kW`，膨胀为 `dH,dW`，padding 为 `pH,pW`，stride 为 `sH,sW`。列空间为：

```text
colH = floor((H + 2*pH - dH*(kH-1) - 1) / sH) + 1
colW = floor((W + 2*pW - dW*(kW-1) - 1) / sW) + 1
L = colH * colW
```

ACLNN 先把输入统一成 `[N,C,kH*kW,L]`，Kernel 将前三维中的 `N,C` 展平为 `cIm∈[0,N*C)`。对每个唯一输出元素 `(cIm,h,w)`，令带 padding 的坐标 `hIm=h+pH,wIm=w+pW`，枚举可能覆盖它的列坐标 `(ho,wo)`；仅当 `(hIm-ho*sH)` 与 `(wIm-wo*sW)` 分别能被 `dH,dW` 整除时，得到 `hk,wk` 并累加：

```text
out[cIm,h,w] = Σ gradOutput[cIm,hk,wk,ho,wo]
```

求和只遍历满足窗口覆盖范围、dilation 整除关系的 `(ho,wo)`。每个 SIMT 线程独占一个或多个输出元素，因此不需要 atomic；FP32、FP16、BF16 三条实例都使用 `float` 累加器，最后转换成目标 `D_T`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| gradOutput | `[C*kH*kW,L]` 或 `[N,C*kH*kW,L]`；ND；允许非连续 | FP32/FP16/BF16 | 非空时先 Contiguous；倒数第二维必须被 `kH*kW` 整除，末维必须等于 `colH*colW` |
| inputSize | `aclIntArray`，长度 2 | INT64 | 文档要求两个值大于 0；950 Tiling 也要求 `H,W>0`；API 自身没有在 `CheckArrayValue` 中显式扫描这两个值 |
| kernelSize | `aclIntArray`，长度 2 | INT64 | `kH,kW>0` |
| dilation | `aclIntArray`，长度 2 | INT64 | `dH,dW>0` |
| padding | `aclIntArray`，长度 2 | INT64 | `pH,pW>=0` |
| stride | `aclIntArray`，长度 2 | INT64 | `sH,sW>0` |
| out | `[C,H,W]` 或 `[N,C,H,W]`；ND；允许非连续 | FP32/FP16/BF16 | rank 比 `gradOutput` 多 1，shape 必须精确匹配；README 声明 dtype 同 `gradOutput`，但 API 代码只分别检查二者属于支持集合，未显式检查相等 |
| workspaceSize/executor/stream | ACLNN 两段式句柄 | uint64/指针 | 第一段返回 executor 聚合 workspace；第二段在指定 stream 运行；空输入直接返回 `workspaceSize=0` 的空执行器 |

`FORMAT_FRACTAL_NZ` 不会被拒绝，只记录可能精度失败的 warning；随后 Contiguous、ReFormat 和 ViewCopy 的实际框架行为由运行时决定。

### 入口与版本

固定 Commit 中仅公开 `aclnnIm2colBackwardGetWorkspaceSize` 与 `aclnnIm2colBackward`，没有 V2 或更高版本。950 由 `IsRegBase()` 对 `NpuArch::DAV_3510` 命中；这一路跳过 A2/A3 分支的 NCHW→NC1HWC0→NCHW 两次 TransData，直接把四维 reshape 结果交给 `l0op::Col2im`。公开文档、`OpDef::AddConfig("ascend950")`、`SUPPORT_COMPUTE_UNIT "ascend950"`、arch35 Tiling 与 ascend950 binary config 对 950 支持互相印证。

公开文档同时列出 Atlas A2/A3 支持，但固定 Commit 的 OpDef、CMake、binary config、Tiling 和 Kernel 只接入 ascend950/arch35；因此本卡不把同一 API 的非 RegBase 分支外推成固定源码已闭合的 A2/A3 实现。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段接口按下列顺序执行：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，创建 `OpExecutor`。
2. `CheckParams` 依次检查七个业务指针、NZ warning、当前平台 dtype 集合、输入/输出 rank、五个数组长度和值、`gradOutput` 的通道核面积整除与 `L`、最后检查 `out` shape。
3. `gradOutput->IsEmpty()` 时不加入任何 launcher，返回 workspace 0。
4. 非空输入经 `l0op::Contiguous`。二维输入在 axis 0 增加 N 维，三维输入保持不变；随后 view reshape 成 `[N,C,kH*kW,L]`。
5. 950 的 `IsRegBase()` 为 true，直接调用 `l0op::Col2im`。L0 将 `inputSize` 转成 INT32 tensor，InferShape 生成 `[N,C,H,W]` 中间输出，并加入一个 AICore launcher。
6. 二维原输入对应的结果 squeeze 掉 N 维；随后 CreateView、ReFormat、Cast 到用户 `out` dtype，再 ViewCopy 到用户输出。
7. 任一步返回空指针都会以 `ACLNN_ERR_INNER_NULLPTR` 短路；成功时由 executor 汇总 workspace。第二段只调用 `CommonOpExecutorRun`。

框架生成的 Contiguous、Unsqueeze/Reshape/Squeeze/View、Cast、ViewCopy 的具体 Kernel、临时区公式与内部失败码不在固定仓；本卡把它们标为 ACLNN 框架边界，不把其内部实现伪装成 Col2im 自有 Route。

### 参数、Workspace 与生命周期

`l0op::Col2im` 把 `inputSize` 从 Host `aclIntArray<int64>` 转成 Device `DT_INT32` 常量 tensor，attrs 仍按 `kernelSize,dilation,padding,stride` 传给 launcher。中间输出由 executor 按输入 dtype、storage/original format 分配；executor 还持有所有 view、框架转换节点及临时 tensor，必须存活到第二段在同一 stream 完成。

Tiling 设置 `workspace[0]=PlatformAscendC::GetLibApiWorkSpaceSize()`；固定源码没有给出此平台库区的数值。算子 body 不读写任何 workspace slice，但 entry 先执行 `GetUserWorkspace(workspace)`，其结果为 null 时直接返回，因此 workspace 基址/框架偏移协议不能省略。Tiling 还设置：

```text
localMemorySize = platformUBSize - 128 KiB
totalLength = N*C*H*W
perCoreCeil = ceil(totalLength / totalAivCoreNum)
blockDim = ceil(totalLength / perCoreCeil)
```

`SetLocalMemorySize` 到真实 SIMT launch dynamic-UB 参数的框架传递在固定仓不可见；Device body 没有声明 shared/local array，也没有显式消费该空间。

### 启动图与失败传播

```text
aclnnIm2colBackwardGetWorkspaceSize
  -> CheckParams -> Contiguous -> optional Unsqueeze -> Reshape
  -> A5: l0op::Col2im
    -> ConvertToTensor(inputSize, INT32) -> INFER_SHAPE(Col2im)
    -> ADD_TO_LAUNCHER_LIST_AICORE(Col2im)
      -> TilingForCol2im -> one node K0: col2im<dtype key>
  -> optional Squeeze -> CreateView -> ReFormat -> Cast -> ViewCopy(out)
aclnnIm2colBackward
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

K0 输入为连续 `[N,C,kH*kW,colH*colW]` GM 与 INT32 `outputSize` tensor，输出为 executor 拥有的 `[N,C,H,W]` GM；`blockDim=useCoreNum`，每个 AIV block 内 `asc_vf_call` 启动 1024 个 VF 线程。K0 与前后框架节点共享 executor stream，因而按 executor 列表顺序可见；固定仓没有显式 event。K0 没有跨 Kernel workspace 边；唯一 workspace 关系是框架拥有的 LibApi 区与 entry 非空检查。第一段任一校验/分配/launcher 添加失败都会阻止后续节点；第二段直接传播 `CommonOpExecutorRun` 返回值。entry 自己遇到 user workspace null 时静默 return，固定源码没有把该 Device 早退回传为 ACLNN 错误。

## Route 与参数设计

### 分核与资源计算

`GetPlatformInfo` 使用 AIV 核数 `Ctotal`，并检查 `Ctotal>0`、`UB>0`。非空域的 `T=N*C*H*W>0`：

```text
chunk = ceil(T / Ctotal)                 # 元素/核，int64
useCoreNum = ceil(T / chunk)             # AIV block 数，<= min(T,Ctotal)
threadsPerBlock = 1024                    # VF_MAX_THREAD_NUM
globalThreadStride = useCoreNum * 1024
```

Device 不采用 Host 切好的连续 per-core chunk，而是统一 grid-stride：初始 `idx=threadIdx.x+blockIdx.x*1024`，之后每次加 `gridDim.x*1024`。Host 将 `T` 限制为 `<=INT32_MAX`，再写入 int64 POD；Device 转为 uint32，因而该范围内不截断。H/W、kernel、stride、dilation、padding、colH/colW 则逐项从 int64 转 uint32，固定 Host 没有对每个字段单独检查 `UINT32_MAX`，但 H*W 和 N*C*H*W 的 INT32 上界间接限制正的 H/W。

没有 tile、former/tail core、TQue/TBuf 或显式 shared buffer。SIMT 每线程的循环长度取决于覆盖当前输出像素的 `(ho,wo)` 候选数；最坏情况与 `ceil(kernelExtentH/sH)*ceil(kernelExtentW/sW)` 同阶，但固定源码不建立该上界或备用 Route。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 输出元素中心化 SIMT | `GET_TPL_TILING_KEY(0/1/2)`；`Col2imSimt<float,float/half/bfloat16_t>` | 950/`DAV_3510`；非空；reshape 后 rank=4；dtype FP32→参数0、FP16→参数1、BF16→参数2；Tiling/属性/平台检查均成功 | 其它 dtype、非法属性、rank/shape、资源或 workspace 指针失败；无 Device fallback |

```text
ACLNN empty? yes -> workspace=0，结束
                 no
IsRegBase(DAV_3510)? yes -> direct ND Col2im
GetDataTypeKey(dtype)
  ├─ FP32 -> template dtype argument 0 -> R-001<float,float>
  ├─ FP16 -> template dtype argument 1 -> R-001<float,half>
  ├─ BF16 -> template dtype argument 2 -> R-001<float,bfloat16_t>
  └─ other -> GRAPH_FAILED
```

三个 dtype key 的 POD、地址、线程图、循环、同步和 tail 完全相同，仅 GM 元素类型及最终 cast 不同，故合并为一个 Route；生成后的 raw tiling-key 数值依赖 `GET_TPL_TILING_KEY` 宏，固定仓只证明模板参数 0/1/2，不推测其编码。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 输出元素中心化 SIMT | dtype 参数 0/1/2 | 950；FP32/FP16/BF16；ACLNN shape 约束；Tiling rank4、H/W/attrs 正性、`T<=INT32_MAX`、AIV/UB 可用 | 单 K0；`useCoreNum` blocks × 1024 VF threads；线程 grid-stride 覆盖 T 个输出 | `col2im<dType>` → `Col2imSimt<float,D_T>::Init/Process` → `Col2imSimtCompute<float,D_T>` | GM direct load/store；float scalar accumulator；无 queue/shared/atomic/barrier；workspace 只检查非空 | `idx<T` 是全局 tail；候选窗口端点 clamp；dilation 整除 guard；不满足 Host guard 直接失败，无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 分支 | `IsRegBase()==true`，集合仅含 `DAV_3510` | 950 直接 ND→arch35 SIMT，避免非 RegBase TransData 图 | R-001 | 非 RegBase 进入另一条 ACLNN 图，但固定仓无对应 Col2im build closure | `common/inc/op_api/aclnn_check.h::IsRegBase`; `aclnnIm2colBackwardGetWorkspaceSize` |
| 输入空域 | `gradOutput->IsEmpty()` | 避免 `ceil(T/ceil(T/C))` 的零除与无意义 launch | R-001 | 返回 workspace 0，不 launch | `aclnnIm2colBackwardGetWorkspaceSize` |
| rank4 内部 ABI | reshape 后 `gradOutDimNum==4` | Device 地址公式固定解释为 `[N,C,K,L]` | R-001 | Tiling `GRAPH_FAILED` | `Col2imTiling::GetInputTensorInfo` |
| 总输出 32 位 | `H*W<=INT32_MAX` 且 `N*C*H*W<=INT32_MAX` | `cnt` 与线程 `idx` 为 uint32 | R-001 | Tiling `GRAPH_FAILED` | `Col2imTiling::GetInputTensorInfo` |
| 正除数 | `W,H,sH,sW,dH,dW>0` | magic division 的 divisor 必须非零 | R-001 | API/Tiling 失败 | `CheckArrayValue`; `SetAttrParams`; `GetInputTensorInfo` |
| AIV 与 UB | `GetCoreNumAiv()>0 && ubSize>0` | blockDim 与 localMemory producer | R-001 | Tiling `GRAPH_FAILED` | `Col2imTiling::GetPlatformInfo` |
| local memory 保留 | `SetLocalMemorySize(ubSize-128 KiB)` | 固定留出 DCACHE，余量交给 SIMT runtime | R-001 | `ubSize<=0` 被拒；源码未检查 `ubSize<128 KiB` 的无符号下溢 | `Col2imTiling::GetPlatformInfo` |
| workspace 非空 | `GetUserWorkspace(workspace)!=nullptr` | entry 在 POD 解析和输出写入前执行硬早退 | R-001 | Device 静默 return，无备用路径 | `op_kernel/col2im.cpp::col2im` |
| 全局线程 tail | `idx<cnt` | `useCoreNum*1024` 通常不整除 T | R-001 | 超出线程不读写 | `Col2imSimtCompute` |
| 窗口边界 | start 基于 kernel extent，end 取 `min(div+1,colH/colW)` | 防止访问列空间之外；减少候选循环 | R-001 | 候选为空时写 0 | `Col2imSimtCompute` |
| dilation 命中 | `hK%dH==0 && wK%dW==0` 的 magic-division 等价式 | 仅真实 kernel lattice 点贡献 | R-001 | 当前候选跳过 | `Col2imSimtCompute` |

### POD/ABI

`Col2imRegBaseTilingData` 是 13 个连续 `int64_t` 数据成员，无继承、虚函数或序列化宏。按这些声明的自然布局，`alignof=8`、`sizeof=104 B`，字段 offset 依次为 0 到 96；Host 在 `InitTilingData` 先把全部 104 B 置零，再逐字段赋值。entry 用 `REGISTER_TILING_DEFAULT/GET_TILING_DATA` 得到同名结构。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| Col2imRegBaseTilingData | totalLength | int64/64 | offset 0；结构 104 B/align 8 | 输出元素 | `SetTilingData: totalLength_=N*C*H*W` | `Process→cnt(uint32)` | Host `0<T<=INT32_MAX`；空输入不 launch |
| 同上 | outputSizeH | int64/64 | offset 8 | 元素 | `GetInputTensorInfo: outputSizeArray[0]` | `Process→gradInH/gradInWH` | `>0` |
| 同上 | outputSizeW | int64/64 | offset 16 | 元素 | `GetInputTensorInfo: outputSizeArray[1]` | `Process→gradInW/gradInWH` | `>0` |
| 同上 | kernelSizeH | int64/64 | offset 24 | 元素 | `SetAttrParams: kernel_size[0]` | `Compute→kernelExtentH/gradOutIdx` | `>0` |
| 同上 | kernelSizeW | int64/64 | offset 32 | 元素 | `SetAttrParams: kernel_size[1]` | `Compute→kernelExtentW/gradOutIdx` | `>0` |
| 同上 | dilationH | int64/64 | offset 40 | 元素间距 | `SetAttrParams: dilation[0]` | `Process` 生成除法 magic；`Compute` extent/整除 | `>0` |
| 同上 | dilationW | int64/64 | offset 48 | 元素间距 | `SetAttrParams: dilation[1]` | 同上 W 轴 | `>0` |
| 同上 | paddingH | int64/64 | offset 56 | 元素 | `SetAttrParams: padding[0]` | `Compute: hIm=h+pH` | `>=0` |
| 同上 | paddingW | int64/64 | offset 64 | 元素 | `SetAttrParams: padding[1]` | `Compute: wIm=w+pW` | `>=0` |
| 同上 | strideH | int64/64 | offset 72 | 元素间距 | `SetAttrParams: stride[0]` | `Process` 生成 magic；`Compute` 候选/坐标 | `>0` |
| 同上 | strideW | int64/64 | offset 80 | 元素间距 | `SetAttrParams: stride[1]` | 同上 W 轴 | `>0` |
| 同上 | colH | int64/64 | offset 88 | 列空间行数 | `GetInputTensorInfo` 的卷积输出公式 | `Compute→heightGradOut/gradOutIdx` | API 令 `L=colH*colW`；Tiling 未单独检查正值 |
| 同上 | colW | int64/64 | offset 96 | 列空间列数 | `GetInputTensorInfo` 的卷积输出公式 | `Compute→widthGradOut/gradOutIdx` | 同上 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 输出元素中心化 SIMT | dtype template arg 0/1/2 | `op_host/arch35/col2im_tiling_arch35.cpp::TilingForCol2im/Col2imTiling::DoTiling` | `Col2imRegBaseTilingData` 13 字段；blockDim/localMemory/workspace | `op_kernel/col2im.cpp::col2im<dType>`，AIV-only，三分支实例 | `op_kernel/arch35/col2im_simt.h::Col2imSimt::Init/Process/Col2imSimtCompute` | `col2im_def.cpp` ascend950；`image/col2im/CMakeLists.txt` ascend950+arch35；ascend950 binary 三 dtype |

## Kernel 实现

### 公共执行流程

K0 的 entry 参数顺序为 `(gradOut,outputSize,gradIn,workspace,tiling)`。`outputSize` 只服务 InferShape/Tiling，Device body 不读取它。entry 先限定 AIV-only、验证 user workspace，再解析 POD；按 dtype 模板参数实例化 `Col2imSimt<float,D_T>`。`Init(gradIn,gradOut,...)` 的形参名容易误导：调用点把 API 输出 `gradIn` 传作第一个参数，`Init` 将第二个参数绑定为 `inputXGm_`、第一个绑定为 `yGm_`，最终仍是从 API `gradOut` 读、向 API `gradIn` 写。

每个 AIV block 在 `Process` 中为 W、H*W、stride 和 dilation 生成无符号常量除法的 magic multiplier/shift，然后以 1024 线程进入 `asc_vf_call`。每线程 grid-stride 遍历扁平输出，计算候选列坐标区间、检查 dilation lattice、用 FP32 累加并写唯一输出。没有节点间边、atomic、barrier、queue 或显式 local buffer。

### Route R-001：输出元素中心化 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | image/col2im/op_api/aclnn_im2col_backward.cpp | `CheckDtype/CheckArray/CheckOutShape` | 950 支持 FP32/FP16/BF16；rank2/3；五个数组长度2；K/D/S 正、P 非负；输入 L 与输出 shape 匹配 | 空输入结束；非空进入 950 L0 |
| API selector | common/inc/op_api/aclnn_check.h；aclnn cpp | `IsRegBase`; `aclnnIm2colBackwardGetWorkspaceSize` | `DAV_3510∈regbaseArch` | 跳过 TransData，直接 `l0op::Col2im` |
| Tiling | image/col2im/op_host/arch35/col2im_tiling_arch35.cpp | `DoTiling/GetInputTensorInfo/GetDataTypeKey` | 内部 rank4；H/W>0；`T<=INT32_MAX`；dtype 映射 0/1/2；AIV与UB有效 | 写 POD/key/blockDim/workspace；任一步失败终止 |
| Router | image/col2im/op_kernel/col2im.cpp | `col2im<dType>` | 编译期参数等于 FP32/FP16/BF16 常量 | 分别实例 `Col2imSimt<float,D_T>`；未知参数无 body |

#### Device 数据流

##### 函数级调用链

```text
l0op::Col2im -> ADD_TO_LAUNCHER_LIST_AICORE(Col2im)
  -> TilingForCol2im -> Col2imTiling::DoTiling
  -> K0 col2im<dType>(gradOut,outputSize,gradIn,workspace,tiling)
    -> Col2imSimt<float,D_T>::Init(gradIn,gradOut,workspace,&tilingData)
    -> Col2imSimt<float,D_T>::Process
      -> GetUintDivMagicAndShift for W/HW/stride/dilation
      -> asc_vf_call<Col2imSimtCompute<float,D_T>>(threads=1024,...)
        -> grid-stride output loop -> candidate ho/wo loops -> GM load/FP32 sum -> GM store
  -> executor stream: optional Squeeze/ReFormat/Cast -> ViewCopy to user out
```

##### 源码等价伪码

```text
伪码：Host/Tiling
T = N*C*H*W
blocks = ceil(T / ceil(T / totalAivCores))
POD = {T,H,W,kH,kW,dH,dW,pH,pW,sH,sW,colH,colW}

伪码：每个 AIV block 的 1024 个 VF 线程
预计算 divMagic(W,H*W,sW,sH,dH,dW)
for idx = threadIdx + blockIdx*1024; idx < T; idx += blocks*1024:
    w = idx % W
    h = floor(idx/W) % H
    nc = floor(idx/(H*W))
    wIm = w + pW; hIm = h + pH
    kernelExtentW = (kW-1)*dW + 1
    kernelExtentH = (kH-1)*dH + 1
    woStart = (wIm >= kernelExtentW) ? floor((wIm-kernelExtentW)/sW)+1 : 0
    woEnd = min(floor(wIm/sW)+1, colW)
    hoStart = (hIm >= kernelExtentH) ? floor((hIm-kernelExtentH)/sH)+1 : 0
    hoEnd = min(floor(hIm/sH)+1, colH)
    acc = 0.0f
    for ho in [hoStart,hoEnd):
      for wo in [woStart,woEnd):
        hkRaw = hIm - ho*sH; wkRaw = wIm - wo*sW
        if hkRaw - floor(hkRaw/dH)*dH == 0 and wkRaw - floor(wkRaw/dW)*dW == 0:
            hk=floor(hkRaw/dH); wk=floor(wkRaw/dW)
            in = ((((nc*kH+hk)*kW+wk)*colH+ho)*colW+wo)
            acc += float(gradOut[in])
    gradIn[idx] = D_T(acc)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM gradOut | `((((nc*kH+hk)*kW+wk)*colH+ho)*colW+wo)` | `N*C*kH*kW*colH*colW` 元素 | D_T；连续 `[N,C,kH,kW,colH,colW]` 逻辑视图 | ACLNN Contiguous+Reshape | SIMT 标量 load | executor/整个 K0 | 无块搬运；按贡献位置散读 |
| GM gradIn 中间输出 | `idx=((nc*H+h)*W+w)` | `T=N*C*H*W` 元素 | D_T；连续 `[N,C,H,W]` | 当前 SIMT 线程 | 后续框架 Squeeze/Cast/ViewCopy | executor；K0 后可读 | 每个 idx 唯一 writer，无 atomic |
| 寄存器/线程标量 | `acc,wIm,hIm,start/end,hk,wk,index` | 固定数量标量；编译后字节数固定仓不可见 | `acc:float`，索引:uint32 | 当前线程 | 当前线程 | 单个 idx 迭代 | 下一 idx 覆盖；线程私有 |
| SIMT local memory | 运行时地址不可见 | `platformUBSize-128 KiB` 由 Host 设置 | 固定仓不可见 | `SetLocalMemorySize` | SIMT runtime；body 无显式数组 consumer | 单 K0 launch | 源码未证明实际占用/别名 |
| workspace LibApi 区 | 框架 `GetUserWorkspace` 映射不可见 | `GetLibApiWorkSpaceSize()` | bytes | executor/runtime | entry 非空检查；body 不读写 | 框架/launch | 数值、offset 与可复用时刻固定仓不可见 |

##### 分层内存布局

未使用 AIV TQue/TBuf/LocalTensor，故 AIV UB `block-beta` 为 N/A。真实替代路径是 SIMT 线程直接从 `gradOut` GM 标量读取，以线程私有 FP32 标量累加，再直接写 `gradIn` GM；Host 只向运行时声明 `localMemorySize=UB-128 KiB`，固定 body 没有显式 dynamic UB/shared buffer 分区。

##### 字节账与复用顺序

```text
AIV_TQue/TBuf_UB_total = 0 B
explicit_SIMT_shared_or_dynamic_buffer = 0 B（body 中无声明）
Host_localMemorySize = platformUBSize - 128 KiB
register_bytes = 固定仓不可见；至少包含一个 FP32 accumulator 与 uint32 索引标量
operator_user_workspace_payload = 0 B（无读写）
workspace_total = GetLibApiWorkSpaceSize()（框架数值与 slice 布局固定仓不可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有 `acc` | 当前 idx 开始时置 0 | 当前 idx 的候选循环与最终 store | 同线程下一 idx 再置 0 | 程序顺序 | 每线程串行 grid-stride 循环 |
| GM `gradIn[idx]` | 唯一命中该 idx 的线程 | executor 后续输出整理节点 | 不在 K0 内覆盖 | 单 stream launcher 顺序 | grid-stride 映射必须保持一一覆盖 |
| workspace | 无算子写入者 | entry 只检查映射非空 | 框架决定 | executor/runtime | 不得删除 entry 要求的有效 workspace 基址 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| executor→K0 | 同一 ACLNN executor stream | Contiguous/视图准备 | K0 launch | 连续 gradOut 与输出中间 tensor | 框架内部 event/临时区固定仓不可见 |
| K0 线程分工 | grid-stride 唯一 idx | `blockIdx/threadIdx` | 当前线程 store | `gradIn[idx]` | `idx<cnt` 屏蔽最后不足 1024×blocks 的线程 |
| 候选区间 | start/end clamp | kernel extent 与 padding 坐标 | 双重候选循环 | gradOut 合法列空间 | 空区间产生精确零；end 不超过 colH/colW |
| dilation | magic quotient 余数等价检查 | `hkRaw/wkRaw` | GM index 构造 | kernel lattice | 非整除候选不参与求和 |
| 累加 | 线程私有 `ACC_T=float` | D_T GM 标量读取 | `valv +=` | 一个输出的所有贡献 | FP16/BF16 升为 FP32；FP32 保持 FP32；加法顺序固定为 ho 外层、wo 内层 |
| 输出转换 | `static_cast<D_T>(valv)` | FP32 accumulator | GM store | 最终 out 中间值 | FP16/BF16 rounding mode 未由源码显式指定；无饱和/atomic |
| K0→ViewCopy | executor stream ordering | K0 中间输出 | Squeeze/ReFormat/Cast/ViewCopy | 用户 out | 若用户 dtype 与 gradOutput 不同，额外 Cast 语义/实现是框架边界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | image/col2im/op_host/arch35/col2im_tiling_arch35.cpp | `Col2imTiling::SetAttrParams/GetInputTensorInfo/GetPlatformInfo/SetTilingData/SetTilingKey/DoTiling` | 属性校验、T/colH/colW、AIV/UB、blockDim、POD、dtype key、workspace |
| Kernel entry/router | image/col2im/op_kernel/col2im.cpp | `col2im<dType>` | AIV-only；workspace 早退；POD 解析；0/1/2 到三种 `Col2imSimt<float,D_T>` |
| Device body/helper | image/col2im/op_kernel/arch35/col2im_simt.h | `Col2imSimt::Init/Process`; `Col2imSimtCompute` | GM 绑定、magic division、1024 线程、grid-stride、窗口范围、FP32 累加与唯一 GM store |

## 算子特定优化

### Trick T-001：输出元素中心化规避重叠窗口 Atomic

#### 触发与源码依据

- 适用 Route：R-001 全部 dtype。
- 触发条件：所有有效 950 非空输入；每个 `idx` 唯一映射到一个 `[N,C,H,W]` 输出元素。
- Golden 来源：[`image/col2im/op_kernel/arch35/col2im_simt.h:64-68`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_kernel/arch35/col2im_simt.h#L64-L68)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for(uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < cnt; idx += gridDim.x * blockDim.x) {
    ACC_T valv = static_cast<ACC_T>(0);
    const uint32_t wIm = idx % gradInW + padWidth;
    const uint32_t hIm = Simt::UintDiv(idx, mGW_, shiftGW_) % gradInH + padHeight;
    const uint32_t cIm = Simt::UintDiv(idx, mGWH_, shiftGWH_);
```

#### 机制与设计原因

常见 Col2im 写法从每个列元素向多个输出位置 scatter-add，重叠窗口会让不同线程竞争同一输出，需要 atomic 或额外归并。本实现反转遍历方向：线程先取得唯一输出 `idx`，再在本线程内 gather 它的全部合法列贡献。`idx` 的 grid-stride 步长来自真实 `gridDim.x*blockDim.x`，而 blockDim.x 固定为 1024；这样 K0 内一个输出只写一次，省去输出清零、atomic 冲突和跨线程同步。代价是输入读取为依赖窗口几何的散读，收益边界不能在未真机时量化。

#### 迁移边界

- 适用：反向 unfold/patch merge、多个输入贡献归并到规则网格，且单输出的贡献集合可独立枚举。
- 前提：能由输出坐标构造有限候选范围；单线程 FP32 累加精度和耗时可接受；GM 随机读取成本不压倒 atomic 节省。
- 失效：单输出贡献极多或需要跨输出共享中间结果时，应另设归约/分块 Route；固定仓没有此 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但不能把当前唯一 writer 协议改为列元素 scatter 后仍删除 atomic，否则重叠窗口会发生写冲突。

### Trick T-002：候选区间裁剪与除法 magic 在 VF launch 前复用

#### 触发与源码依据

- 适用 Route：R-001 全部 dtype。
- 触发条件：`W,H,sH,sW,dH,dW` 为正，Host/Tiling guard 已通过。
- Golden 来源：[`image/col2im/op_kernel/arch35/col2im_simt.h:122-137`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/col2im/op_kernel/arch35/col2im_simt.h#L122-L137)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
GetUintDivMagicAndShift(mGW_, shiftGW_, gradInW);
GetUintDivMagicAndShift(mGWH_, shiftGWH_, gradInWH);
GetUintDivMagicAndShift(mSW_, shiftSW_, strideWidth);
GetUintDivMagicAndShift(mSH_, shiftSH_, strideHeight);
GetUintDivMagicAndShift(mDH_, shiftDH_, dilationHeight);
GetUintDivMagicAndShift(mDW_, shiftDW_, dilationWidth);
asc_vf_call<Col2imSimtCompute<ACC_T, D_T>>(
    dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ D_T*)(inputXGm_.GetPhyAddr()), (__gm__ D_T*)(yGm_.GetPhyAddr()),
```

#### 机制与设计原因

六个正除数的 multiplier/shift 在每个 AIV block 的 `Process` 中各生成一次，作为值参数传入 1024 个 VF 线程；线程内坐标拆解、start/end 和 dilation 余数均用 `Simt::UintDiv` 复用它们，而不在每次候选迭代重新构造除法参数。随后 `kernelExtent` 推导出最小 start，end 再 clamp 到 `colH/colW`，先排除绝不覆盖当前输出的列位置，最后才做 dilation lattice 检查和 GM load。这同时改变标量计算量与无效 GM 访问量。

#### 迁移边界

- 适用：大量线程重复用少量运行时正整数作无符号除法，且候选域可先由区间粗筛再由模条件精筛。
- 前提：divisor 非零；输入/坐标受 uint32 范围约束；magic 参数的生命周期覆盖一次 `asc_vf_call`。
- 失效：有符号负坐标、64 位域或 divisor 可为零时，不能直接复用这组 uint32 API；需换匹配的除法与边界实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时若在 dilation 整除检查前读取 GM，会访问并累加不属于 kernel lattice 的位置。

## 模板与实现对比

| 对比维度 | FP32 参数 0 | FP16 参数 1 | BF16 参数 2 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | `gradOutDesc=DT_FLOAT` | `DT_FLOAT16` | `DT_BF16` | 单 switch，无 shape 子 Route |
| 模板与 dtype | `Col2imSimt<float,float>` | `Col2imSimt<float,half>` | `Col2imSimt<float,bfloat16_t>` | ACC_T 对三者都固定 float |
| 循环与分核 | 同一 blocks×1024 grid-stride | 同左 | 同左 | Tiling 与 POD 完全共享 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf；float accumulator | 同左 | 同左 | body 直接 GM 标量访问；local memory 仅 Host 声明 |
| 内存总占用与复用 | GM 元素 4 B；显式 UB 0 B | GM 元素 2 B；显式 UB 0 B | GM 元素 2 B；显式 UB 0 B | dtype 仅影响 GM 容量/转换，不改变地址元素索引 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar load/store、UintDiv | 同左 | 同左 | 不使用 Vector/NDDMA/通信 |
| 同步、精度与 tail | FP32 累加→FP32 | FP16 读→FP32 累加→FP16 | BF16 读→FP32 累加→BF16 | 相同固定 ho/wo 求和序；idx tail 相同 |
| fallback/失败边界 | 无 | 无 | 无 | 未知 dtype 在 Host 失败；entry 未实例化其它参数 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 表写支持；固定 Commit 无 A2 OpDef/config/Tiling/Kernel/build closure，未生成正式卡 | 公开 README/API 表写支持；固定 Commit无 A3 OpDef/config/Tiling/Kernel/build closure，未生成正式卡 | 支持；公开表、ascend950 OpDef/config/CMake 与 arch35 实现闭合 |
| Host/Launcher | ACLNN 非 RegBase 分支包含两次 TransData，但 Col2im 实体固定仓未接入该平台 | 同 A2 | `DAV_3510` 直接 ND Col2im，单自有 Kernel |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | AIV-only SIMT；1024 threads/block；localMemory=UB-128 KiB；三 dtype |

## 限制与失败边界

- API 文档要求 `inputSize` 元素大于 0；API `CheckArrayValue` 没有检查它，最终由 950 Tiling `GetInputTensorInfo` 拒绝非正 H/W。若框架在 Tiling 前使用非法 shape，精确行为固定仓不可见。
- `CheckDtype` 只确认 gradOutput 与 out 各自在支持集合，未检查两者相同；README 要求相同 dtype。不同 dtype 时 final Cast 会改变语义，其具体 Kernel、rounding、workspace 属外部框架边界。
- API 的 `numL0/numL1` 使用 `size_t` 接收含 int64 属性的表达式；极端非法几何可能先产生负值再无符号化。Tiling 计算 `colH/colW` 后也不单独检查正数，只依赖 ACLNN shape/L 闭合。
- `totalLength_` 在乘法之后才检查 `INT32_MAX`，固定源码没有乘前溢出证明；合法 ACLNN tensor shape 通常缩小风险，但不能把未检查的极端维度写成受支持。
- `kernelExtent=(kernelSize-1)*dilation+1`、各字段的 int64→uint32 窄化没有逐项 `UINT32_MAX` guard；仅 H/W/T 有间接 INT32 上界。
- `SetLocalMemorySize(ubSize-128 KiB)` 只检查 `ubSize>0`，未显式检查 `ubSize>=128 KiB`；若平台报告更小 UB，会发生无符号减法下溢。固定 950 实际 UB 数值未写在仓内。
- entry 要求 `GetUserWorkspace(workspace)` 非空却不使用返回指针；null 时静默不写输出。LibApi workspace 的数值、offset、owner 内部实现固定仓不可见。
- FP16/BF16 的 `static_cast<D_T>` rounding mode 固定仓未显式声明；只可确认先用 FP32 按 ho/wo 固定顺序累加。
- 通用 Contiguous、Unsqueeze/Reshape/Squeeze、ReFormat、Cast、ViewCopy 和 executor stream 内部实现不在固定仓；本卡闭合的是算子拥有的 ACLNN 编排与 Col2im 950 核心链路。
- 公开文档列 A2/A3 为支持，但固定源码只构建 ascend950；非 950 不满足 `source_closed`，不能依据文档或 ACLNN 非 RegBase 分支推测实现。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | image/col2im/docs/aclnnIm2colBackward.md | 产品表、原型、参数与错误表 | A2/A3/950 公开性、shape/dtype、两段式语义 |
| 入口声明/约束 | image/col2im/op_api/aclnn_im2col_backward.h | 两个 `ACLNN_API` 原型 | C ABI 与参数顺序 |
| 入口实现 | image/col2im/op_api/aclnn_im2col_backward.cpp | `Check*`; 两段 ACLNN 函数 | 校验、空域、950 selector、executor 图、workspace |
| 入口实现 | image/col2im/op_api/col2im.cpp | `l0op::Col2im` | inputSize INT32 tensor、InferShape、AICore launcher |
| 入口实现 | common/inc/op_api/aclnn_check.h | `IsRegBase` | 950 `DAV_3510` 判定 |
| Host/Tiling/Launcher | image/col2im/op_host/col2im_def.cpp | `ops::Col2im` | 三 dtype ND、ascend950 OpDef 注册 |
| Host/Tiling/Launcher | image/col2im/op_host/col2im_infershape.cpp | `InferShape4Col2im/InferDtype4Col2im` | 常量 output_size 与 `[N,C,H,W]` 输出 |
| Host/Tiling/Launcher | image/col2im/op_host/arch35/col2im_tiling_arch35.cpp | `Col2imTiling` 全流程 | 属性、shape、平台、分核、POD/key/workspace |
| Host/Tiling/Launcher | image/col2im/op_kernel/arch35/col2im_tiling_data.h | `Col2imRegBaseTilingData` | 13 字段 Host/Device ABI |
| Host/Tiling/Launcher | image/col2im/op_kernel/arch35/col2im_tiling_key.h | `ASCENDC_TPL_ARGS_DECL/SEL` | FP32/FP16/BF16 参数 0/1/2 |
| Kernel entry/body | image/col2im/op_kernel/col2im.cpp | `col2im<dType>` | AIV-only entry、workspace guard、三实例路由 |
| Kernel entry/body | image/col2im/op_kernel/arch35/col2im_simt.h | `Col2imSimt/Col2imSimtCompute` | 线程映射、地址、候选裁剪、精度与 tail |
| 构建接线 | image/col2im/op_host/config/ascend950/col2im_binary.json | `Col2im` 三条 `op_list` | ascend950 三 dtype binary |
| 构建接线 | image/col2im/CMakeLists.txt | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR/add_all_modules_sources` | 仅 ascend950、arch35 活动构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 13 个 int64 字段可直接复用为 950 Host↔Device 合同 | 字段顺序、104 B 大小、元素单位、colH/colW 公式与 int64→uint32 消费不可改变 | `Col2imRegBaseTilingData`; `SetTilingData`; `Process` |
| 调用顺序 | ACLNN 950 可复用 Contiguous→view reshape→单 Col2im→输出整理 | `Init(gradIn,gradOut)` 的反直觉传参与 GM 绑定必须按调用点解释；inputSize 仍要转 INT32 tensor | `aclnnIm2colBackwardGetWorkspaceSize`; `l0op::Col2im`; `Col2imSimt::Init` |
| workspace/内存 | 核心算法可保持 direct GM gather、零显式 UB buffer | entry 的有效 workspace 基址、`GetLibApiWorkSpaceSize` 与 `SetLocalMemorySize(UB-128KiB)` 协议不能擅删 | `DoTiling`; `GetPlatformInfo`; `col2im` |
| 同步 | 唯一输出 writer 消除 Kernel 内 atomic/barrier | 必须维持 idx 的全局 grid-stride 一一覆盖；前后框架节点仍依赖 executor stream 顺序 | `Col2imSimtCompute`; `CommonOpExecutorRun` |
| 精度与尾块 | FP16/BF16 使用 FP32 accumulator；候选顺序固定；idx/start/end/dilation 三层 tail 可复用 | 不能改成低精度累加；end clamp、dilation 整除与 `idx<cnt` 不得省略 | `Col2imSimtCompute` |
| 失败边界 | 非空、dtype、rank、shape、属性、T/平台资源检查可作为移植前置条件 | 无 fallback 的失败必须保留；workspace null 静默早退、localMemory 下溢与宽度窄化缺口不得包装成已校验 | `CheckParams`; `DoTiling`; entry `col2im` |
