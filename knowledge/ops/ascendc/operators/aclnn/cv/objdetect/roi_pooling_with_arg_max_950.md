---
type: Operator
title: "RoiPoolingWithArgMax 950 ACLNN 算子知识"
description: "说明 aclnnRoiPoolingWithArgMax 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/docs/aclnnRoiPoolingWithArgMax.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_api/aclnn_roi_pooling_with_arg_max.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_api/aclnn_roi_pooling_with_arg_max.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_api/roi_pooling_with_arg_max.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_host/roi_pooling_with_arg_max_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_host/arch35/roi_pooling_with_arg_max_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_kernel/roi_pooling_with_arg_max.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_host/roi_pooling_with_arg_max_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_host/config/ascend950/roi_pooling_with_arg_max_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T21:00:14Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# RoiPoolingWithArgMax 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnRoiPoolingWithArgMaxGetWorkspaceSize` / `aclnnRoiPoolingWithArgMax`，公开 ACLNN 两段式接口。 |
| 计算语义 | 对每个 `(ROI, channel, pooled_h, pooled_w)` bin 做 NCHW 平面最大池化，同时输出首个最大值的 `h*W+w`。 |
| 输入输出 | `x:[N,C,H,W]`、`rois:[num_rois,5]`，FP32/FP16 ND；`y:[num_rois,C,pooled_h,pooled_w]` 同 `x` dtype，`argmax` 为 INT32。 |
| 执行类型 | 单 Kernel：AIV-only entry 内用 `Simt::VF_CALL` 启动 1024-thread SIMT；无 Cube/AIC 或 MIX_CV 可达节点。 |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16 SIMT。两路仅 dtype key、GM 元素类型和最终 cast 不同。 |
| 核心资源 | 直接 GM load/store；每线程标量寄存器保存坐标、范围、最大值和索引；无显式 AIV UB、dynamic UB、共享内存或跨 Kernel workspace 边；entry 要求 user workspace 非空。 |
| 来源闭包 | `source_closed`：公开 ACLNN→Contiguous/内部输出→Level0 launcher→arch35 Tiling/POD→dtype key→SIMT entry/body→ascend950 构建接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `rois[n]=[batch_idx,x1,y1,x2,y2]`，空间缩放为 `sh/sw`。实现先计算 `roi_x1=x1*sw`、`roi_x2=(x2+1)*sw`，高度同理，并将每个 ROI 均分成 `pooled_h*pooled_w` 个浮点 bin。bin 边界分别做 `floor(left/top)`、`ceil(right/bottom)`，然后裁剪到 `[0,W]×[0,H]`。非空 bin 按 `h` 外层、`w` 内层扫描 `x[batch_idx,c,h,w]`，以严格 `val>max_val` 更新，因此有限值并列时保留行主序第一个位置；输出值转回输入 dtype，索引写为 `h*W+w`。ROI 宽高非正或裁剪后 bin 为空时输出 `0` 和 `-1`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 4D NCHW 语义、ND storage | FP32/FP16 | 必选且与 `rois/y` 同 dtype；公开文档要求非空，源码 ACLNN 对 `x->IsEmpty()` 走零 workspace 空路径。 |
| `rois` | `[num_rois,5]`、ND | 与 `x` 相同 | 每行 `[batch_idx,x1,y1,x2,y2]`；公开文档要求非空、`num_rois<=1024`、坐标非负。 |
| `pooled_h` | 标量属性 | INT64 API→INT64 Host/POD | 必须 `>0`，决定输出第 2 维和 bin 高。 |
| `pooled_w` | 标量属性 | INT64 API→INT64 Host/POD | 必须 `>0`，决定输出第 3 维和 bin 宽。 |
| `spatial_scale_h` | 标量属性 | FP32 | 必须 `>0`，映射 ROI 的高度坐标。 |
| `spatial_scale_w` | 标量属性 | FP32 | 必须 `>0`，映射 ROI 的宽度坐标。 |
| `y` | `[num_rois,C,pooled_h,pooled_w]`、ND | 与 `x` 相同 | 用户输出可非连续；Level0 先产出内部连续 tensor，再 `ViewCopy`。 |
| `argmax` | 与 `y` 同 shape、ND | INT32 | 用户输出可非连续；记录通道平面内 `h*W+w`，空 bin 为 `-1`。 |
| 内部 `roi_actual_num` | `[1]`、ND | INT32 | ACLNN 以 `{num_rois}` 构造；传到 Kernel 但当前 SIMT body 不读取。 |
| 内部 `pool_channel` | 标量属性 | INT64 | ACLNN 取 `x.shape[1]`；OpDef 声明为 Int，当前 arch35 Tiling 不读取该属性，而直接从 `x` shape 取 channels。 |

### 入口与版本

固定仓只公开这一组 `aclnnRoiPoolingWithArgMax` 两段式接口，没有更高版本后缀。公开 ACLNN 文档的产品表只对 Ascend 950PR/950DT 标记支持，对 A2/A3 明确标记不支持；`CheckSocValid` 还要求 `IsRegBase()`。OpDef 仅注册 `ascend950`，CMake 的 `SUPPORT_COMPUTE_UNIT` 也仅为 `ascend950`，因此本文只发布 950 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先校验 `workspaceSize/executor` 公共输出指针和 RegBase SoC，再创建 executor。`CheckParams` 依序检查四个 tensor 非空；`x/rois/y` 仅允许 FP32/FP16 且 dtype 相同，`argmax` 必须 INT32；四者 storage format 必须 ND；`x/y/argmax` 必须 4D，`rois` 必须 2D 且第二维为 5；`y/argmax` shape 必须精确等于 `[num_rois,C,pooled_h,pooled_w]`；四个标量属性必须为正。

若 `x->IsEmpty()`，第一段直接返回 `workspaceSize=0` 和空计算 executor。否则把 `x/rois` 转连续，构造 INT32 `{num_rois}` 的 `roi_actual_num`，从 `x.shape[1]` 生成 `pool_channel`，再调用 Level0 `RoiPoolingWithArgMax`。Level0 分配连续内部 `y` 和 INT32 `argmax`，执行 infer-shape 后添加一个 `ADD_TO_LAUNCHER_LIST_AICORE` 节点；最后分别把内部结果 `ViewCopy` 到用户输出。

### 参数、Workspace 与生命周期

Tiling 从平台读取 AIV core 数 `totalCoreNum` 和 UB 大小；它把 local memory size 设成 `ubSize-128 KiB`，但本算子 SIMT body 没有显式分配或寻址这段 local memory。算子 workspace slice 为 `PlatformAscendC::GetLibApiWorkSpaceSize()`，由 executor 汇总进第一段返回值；Kernel entry 调用 `GetUserWorkspace(workspace)` 并在结果为空时直接返回，之后不再读写该地址。内部连续 tensor、`roi_actual_num`、内部输出和 ViewCopy 节点均由 executor 持有，第二段 `CommonOpExecutorRun` 在调用者提供的同一 stream 上按 executor 图执行。

### 启动图与失败传播

```text
Contiguous(x), Contiguous(rois), ConvertToTensor({num_rois})
  -> Level0 RoiPoolingWithArgMax / infer shape
  -> one roi_pooling_with_arg_max AIV-only entry
       blockDim = min(totalAivCoreNum, outputDataCount)（正数有效域下的等价式）
       threadsPerBlock = 1024, dynamic UB = N/A
       POD = RoiPoolingWithArgMaxRegBaseTilingData
       FP32 -> R-001 SIMT body；FP16 -> R-002 SIMT body
  -> ViewCopy(internal y -> user y)
  -> ViewCopy(internal argmax -> user argmax)
```

只有一个算子 Device 节点，因此没有 Kernel 间 workspace/GM 交接边。executor 的同 stream 执行保证 Contiguous/内部 Kernel/ViewCopy 的顺序与可见性。任一 `CHECK_RET`、Level0 分配/launcher 失败或 Tiling `GRAPH_FAILED` 都阻断后续节点；entry 的 user workspace 空指针分支则静默提前返回，输出不被该 Kernel 写入。

## Route 与参数设计

### 分核与资源计算

令 `Q=roiNumber*channels*poolH*poolW`（输出元素数）。Tiling 计算 `perCore=ceil(Q/totalCoreNum)`，再令 `useCoreNum=ceil(Q/perCore)`，并在非正时钳为 1；对 `Q>0`，该式等价于使用不超过 `totalCoreNum` 且不超过 `Q` 的核数。Kernel 内每个 block 启动 1024 个 SIMT thread，global thread id 为 `threadIdx.x+blockIdx.x*blockDim.x`，以 `gridDim.x*blockDim.x` 为步长遍历 `[0,Q)`，自然覆盖尾部而无需 former/tail core POD。

Host 不按 shape 或资源选择其它算法，只按 `x` dtype 产生模板 key。entry 还为 `poolW/poolH/channels` 计算三组 unsigned division magic/shift，SIMT 用它们把平坦 `idx` 解码为 `(n,c,ph,pw)`，避免循环内普通整数除法。Tiling 虽设置 `localMemorySize=ubSize-128 KiB`，当前 body 直接访问 GM，无 `TQue/TBuf/shared/dynamic UB` 分配公式。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `GET_TPL_TILING_KEY(0)`；`dType=ROI_POOLING_WITH_ARG_MAX_TPL_FP32` | ACLNN 已校验 `x/rois/y` 均为 FP32；`GetDataTypeKey(DT_FLOAT)` 写 `0`。 | dtype 非 FP32 时检查 R-002。 |
| 2 | R-002 FP16 SIMT | `GET_TPL_TILING_KEY(1)`；`dType=ROI_POOLING_WITH_ARG_MAX_TPL_FP16` | ACLNN 已校验 `x/rois/y` 均为 FP16；`GetDataTypeKey(DT_FLOAT16)` 写 `1`。 | 其它 dtype：ACLNN 返回参数错误，或 Tiling `GRAPH_FAILED`。 |

```text
if x.dtype == DT_FLOAT:
  key = GET_TPL_TILING_KEY(0) -> roi_pooling_with_arg_max<0> -> SIMT<float>
else if x.dtype == DT_FLOAT16:
  key = GET_TPL_TILING_KEY(1) -> roi_pooling_with_arg_max<1> -> SIMT<half>
else:
  fail
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 SIMT | dtype key 0 | 950 RegBase；FP32 ND；公开 shape/attr 约束成立 | 单 AIV-only entry→1024-thread SIMT | `roi_pooling_with_arg_max<0>`→`RoiPoolingWithArgMaxCompute<float>`→`RoiPoolingBinRange<float>/RoiPoolingBinMax<float>` | 直接 GM；每线程独占一个 `idx`；无 barrier/atomic | grid-stride 的 `idx<count` 保护尾线程；空 ROI/bin 写 `0/-1`。 |
| R-002 FP16 SIMT | dtype key 1 | 950 RegBase；FP16 ND；同上 | 同一单 Kernel 拓扑 | `roi_pooling_with_arg_max<1>`→`RoiPoolingWithArgMaxCompute<half>`→两个 half helper | FP16 GM load 转 FP32 比较/坐标计算，结果 cast 回 half；其余同 R-001 | 同 R-001；无其它 fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| RegBase SoC | `IsRegBase()==true` | 仅 arch35 SIMT 实现和 binary 接线存在 | R-001/R-002 | `ACLNN_ERR_RUNTIME_ERROR` | `op_api/aclnn_roi_pooling_with_arg_max.cpp::CheckSocValid` |
| dtype key | `DT_FLOAT` 或 `DT_FLOAT16` | entry 只实例化 float/half 两个 body | R-001/R-002 | ACLNN 参数错误/Tiling 失败 | `GetDataTypeKey`、`roi_pooling_with_arg_max<dType>` |
| 正除数 | `poolH>0 && poolW>0`，有效输入还需 `channels>0` | magic/shift 和坐标解码的除数必须为正 | 两路 | ACLNN/Tiling 失败；空 `x` 先短路 | `CheckAttr`、`SetAttrParams`、`GetUintDivMagicAndShift` |
| user workspace | `GetUserWorkspace(workspace)!=nullptr` | entry 在解析 POD 和启动 SIMT 前检查 | 两路 | entry 直接返回，不写输出 | `roi_pooling_with_arg_max` |
| 32 位解码 | `idx` 被 `static_cast<uint32_t>` | magic division 参数和解码量均为 uint32 | 两路 | 固定仓没有大于 UINT32_MAX 输出域的 64 位 fallback | `RoiPoolingWithArgMaxCompute` |
| ROI/bin 空域 | `roi_w<=0 \|\| roi_h<=0`，或裁剪后边界倒置/相等 | 避免非法范围扫描 | 两路 | 写 `y=0,argmax=-1` 后继续 | `RoiPoolingWithArgMaxCompute`、`RoiPoolingBinRange` |

### POD/ABI

`RoiPoolingWithArgMaxRegBaseTilingData` 按声明顺序为六个 `int64_t` 后接两个 `float`；在固定仓采用的自然 8 字节对齐布局下 `sizeof=56`、`alignof=8`。Host 在 `InitTilingData` 中先将完整 56 B 清零，再逐字段覆盖。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `RoiPoolingWithArgMaxRegBaseTilingData` | `channels` | `int64_t`/64 | offset 0；struct 56/align 8 | channel | `SetTilingData: channels_ = x.storageShape[1]` | entry 窄化为 `int32 ch`；magic 除法与 GM channel offset | `>0`；超过 INT32/UINT32 无 fallback。 |
| 同上 | `fmHeight` | `int64_t`/64 | offset 8 | element | `GetInputTensorInfo: x.storageShape[2]` | entry 窄化 `fh`；bin clamp、输入平面地址 | 非空有效域应为正。 |
| 同上 | `fmWidth` | `int64_t`/64 | offset 16 | element | `GetInputTensorInfo: x.storageShape[3]` | entry 窄化 `fw`；bin clamp、`h*W+w` | 非空有效域应为正。 |
| 同上 | `roiNumber` | `int64_t`/64 | offset 24 | ROI | `GetInputTensorInfo: rois.storageShape[0]` | entry 窄化 `rn`；`count=rn*C*poolH*poolW` | 公开文档要求非空且 `<=1024`。 |
| 同上 | `poolH` | `int64_t`/64 | offset 32 | pooled row | `SetAttrParams: *attrs[0]` | entry 窄化 `ph`，magic 除法、bin 高 | 必须 `>0`。 |
| 同上 | `poolW` | `int64_t`/64 | offset 40 | pooled column | `SetAttrParams: *attrs[1]` | entry 窄化 `pw`，magic 除法、bin 宽 | 必须 `>0`。 |
| 同上 | `spatialH` | `float`/32 | offset 48 | ratio | `SetAttrParams: *attrs[2]` | ROI 高度和 bin 边界 | 必须 `>0`。 |
| 同上 | `spatialW` | `float`/32 | offset 52 | ratio | `SetAttrParams: *attrs[3]` | ROI 宽度和 bin 边界 | 必须 `>0`。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 SIMT | `GetDataTypeKey(DT_FLOAT)`→模板 key 0 | `TilingForRoiPoolingWithArgMax`→`RoiPoolingWithArgMaxTiling::DoTiling` | `RoiPoolingWithArgMaxRegBaseTilingData` 全字段 | `roi_pooling_with_arg_max<0>`，AIV-only，`VF_CALL(...dim3{1024,1,1})` | `RoiPoolingWithArgMaxCompute<float>`、`RoiPoolingBinRange<float>`、`RoiPoolingBinMax<float>` | CMake ascend950/arch35；OpDef ascend950；float32 binary 项。 |
| R-002 FP16 SIMT | `GetDataTypeKey(DT_FLOAT16)`→模板 key 1 | 同上 | 同一 POD | `roi_pooling_with_arg_max<1>`，同一 entry | 对应 `<half>` 三个实例 | 同一接线；float16 binary 项。 |

## Kernel 实现

### 公共执行流程

entry 固定声明 `KERNEL_TYPE_AIV_ONLY`，先验证 user workspace，再解析 POD，针对 `poolW/poolH/channels` 生成 magic/shift，并把六个 int64 字段窄化为 int32。dtype 模板分支以 1024 threads 调用唯一 SIMT body。每个线程按 grid-stride 遍历独占的平坦输出索引，解码 `(n,c,ph,pw)`，读取该 ROI，计算 bin 范围，扫描对应输入平面后直接写 `y[idx]` 和 `argmax[idx]`。两 Route 都没有第二个 Kernel、显式 UB 搬运、共享缓存、barrier 或 atomic。

### Route R-001：FP32 直接 GM SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_roi_pooling_with_arg_max.cpp` | `CheckDtype` | `x/rois/y` 同为 `DT_FLOAT` | 进入 Level0。 |
| Tiling | `op_host/arch35/roi_pooling_with_arg_max_tiling.cpp` | `GetDataTypeKey` | `case ge::DT_FLOAT: dataTypeTilingKey_=0` | `GET_TPL_TILING_KEY(0)`。 |
| Router | `op_kernel/roi_pooling_with_arg_max.cpp` | `roi_pooling_with_arg_max<dType>` | `if constexpr(dType==...FP32)` | `VF_CALL<RoiPoolingWithArgMaxCompute<float>>(1024,...)`。 |

#### Device 数据流

##### 函数级调用链

```text
Level0 ADD_TO_LAUNCHER_LIST_AICORE(RoiPoolingWithArgMax)
  -> roi_pooling_with_arg_max<ROI_POOLING_WITH_ARG_MAX_TPL_FP32>
    -> GetUintDivMagicAndShift(poolW/poolH/channels)
    -> Simt::VF_CALL<RoiPoolingWithArgMaxCompute<float>>(dim3{1024,1,1})
      -> grid-stride output loop
        -> RoiPoolingBinRange<float>
        -> RoiPoolingBinMax<float>（仅非空 bin）
        -> direct GM stores y/argmax
```

##### 源码等价伪码

```text
count = rn * C * poolH * poolW
for idx = threadIdx.x + blockIdx.x*1024; idx < count; idx += gridDim.x*1024:
  pw = idx % poolW
  t  = idx / poolW
  ph = t % poolH
  t  = t / poolH
  c  = t % C
  n  = t / C                         # 实际除法用 magic/shift
  roi = rois + n*5
  batch = int64(float(roi[0]))
  roiW = float(roi[3]-roi[1]+1)*sw
  roiH = float(roi[4]-roi[2]+1)*sh
  if roiW<=0 or roiH<=0: y[idx]=0; argmax[idx]=-1; continue
  left/right = clamp(floor/ceil(pw-bin scaled endpoints), 0, W)
  top/bottom = clamp(floor/ceil(ph-bin scaled endpoints), 0, H)
  maxVal=-INF; maxIdx=-1
  if nonempty:
    for h in [top,bottom):
      for w in [left,right):
        val = float(x[(batch*C+c)*H*W + h*W+w])
        if val > maxVal: maxVal=val; maxIdx=h*W+w
  else: maxVal=0
  y[idx]=float(maxVal); argmax[idx]=int32(maxIdx)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `rois` | `n*5 + {0..4}` | 5 elements/ROI；每输出重读 | FP32/row-major | API/Contiguous | 当前 SIMT thread | executor，Kernel 期间只读 | 无显式缓存/合并。 |
| GM `x` | `(batch*C+c)*H*W+h*W+w` | 当前 bin 面积个标量 | FP32/NCHW | API/Contiguous | `RoiPoolingBinMax` | executor，Kernel 期间只读 | 行主序逐元素 load。 |
| GM `y` | `idx` | 1 FP32/thread iteration | FP32/[R,C,PH,PW] | 当前 thread | 后续 ViewCopy | internal tensor 至 ViewCopy 完成 | `idx` 域互斥，无 atomic。 |
| GM `argmax` | `idx` | 1 INT32/thread iteration | INT32/同 y | 当前 thread | 后续 ViewCopy | internal tensor 至 ViewCopy 完成 | 同上。 |
| 标量寄存器 | thread-private | 坐标、边界、`max_val/max_idx` | FP32/int64/uint32 | SIMT thread | 同一 thread | 单次 idx 迭代 | 固定仓不可见具体寄存器字节数。 |
| user workspace | `GetUserWorkspace(workspace)` | `GetLibApiWorkSpaceSize()` | opaque bytes | executor/runtime | entry 仅做 non-null check | 单次 executor run | body 消费 0 B；不可传 null。 |

##### 分层内存布局

N/A：该 SIMT 路线直接进行 GM 标量 load/store，没有显式 AIV UB、shared memory 或 dynamic UB 分配；中间量只保存在每线程标量寄存器中。

##### 字节账与复用顺序

```text
explicit_AIV_UB = 0 B
dynamic_ub = N/A（VF_CALL 未传 dynamic UB 参数）
shared_memory = 0 B（固定仓未见声明）
workspace_total = PlatformAscendC::GetLibApiWorkSpaceSize()
workspace_bytes_consumed_by_body = 0 B；entry 仅要求 userWS != nullptr
register_bytes = 固定仓不可见，只能闭合 thread-private 生命周期
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private `max_val/max_idx` | 当前 idx 初始化/扫描 | 当前 idx 输出 store | 同一线程下一次 grid-stride idx | 单线程程序顺序 | store 完成后才进入下一 idx。 |
| GM output `idx` | 唯一映射 thread | ViewCopy | executor 完成 Kernel 后 | 同 stream Kernel→ViewCopy 顺序 | grid-stride 映射不得重叠。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Kernel 前后 | executor 同 stream 顺序 | Contiguous/Kernel | Kernel/ViewCopy | 内部 GM tensor | 前序完成后可读，Kernel 完成后 ViewCopy。 |
| 线程域 | 无 barrier/atomic | 每个 thread | 自身 store | 唯一 `idx` | `idx<count` 保护最后不完整 stride。 |
| ROI/bin 几何 | FP32 `floorf/ceilf` | 当前 thread | range scan | 边界标量 | 端点裁剪到 `[0,W/H]`；ROI/bin 空域写 `0/-1`。 |
| max scan | 严格 `>` | 行主序 GM load | thread-private accumulator | 单 bin | FP32 输入/比较；并列有限最大值保留首个；NaN 不触发更新。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/roi_pooling_with_arg_max_tiling.cpp` | `SetTilingData/GetDataTypeKey/DoTiling` | blockDim、POD、dtype key、workspace。 |
| Kernel entry/router | `op_kernel/roi_pooling_with_arg_max.cpp` | `roi_pooling_with_arg_max<0>` | AIV-only、userWS guard、1024-thread float VF_CALL。 |
| Device body/helper | `op_kernel/arch35/roi_pooling_with_arg_max_simt.h` | `RoiPoolingWithArgMaxCompute<float>`、两个 helper | flat-index 解码、ROI/bin 几何、最大值扫描、直接 GM 输出。 |

### Route R-002：FP16 输入、FP32 标量计算 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_roi_pooling_with_arg_max.cpp` | `CheckDtype` | `x/rois/y` 同为 `DT_FLOAT16` | 进入 Level0。 |
| Tiling | `op_host/arch35/roi_pooling_with_arg_max_tiling.cpp` | `GetDataTypeKey` | `case ge::DT_FLOAT16: dataTypeTilingKey_=1` | `GET_TPL_TILING_KEY(1)`。 |
| Router | `op_kernel/roi_pooling_with_arg_max.cpp` | `roi_pooling_with_arg_max<dType>` | `else if constexpr(dType==...FP16)` | `VF_CALL<RoiPoolingWithArgMaxCompute<half>>(1024,...)`。 |

#### Device 数据流

##### 函数级调用链

```text
Level0 ADD_TO_LAUNCHER_LIST_AICORE(RoiPoolingWithArgMax)
  -> roi_pooling_with_arg_max<ROI_POOLING_WITH_ARG_MAX_TPL_FP16>
    -> GetUintDivMagicAndShift(poolW/poolH/channels)
    -> Simt::VF_CALL<RoiPoolingWithArgMaxCompute<half>>(dim3{1024,1,1})
      -> RoiPoolingBinRange<half>
      -> RoiPoolingBinMax<half>（half GM load -> float compare）
      -> float max_val cast to half GM y；int64 max_idx cast to int32 GM argmax
```

##### 源码等价伪码

```text
执行与 R-001 完全相同的 grid-stride、索引解码和 bin 扫描；差异为：
  roi coordinate: half GM -> float 后参与 scale/floor/ceil
  feature value:  half GM -> float val，与 float max_val 比较
  output value:   float max_val -> half y[idx]
  argmax:         int64 max_idx -> int32 argmax[idx]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `rois` | `n*5+field` | 5 elements/ROI；每输出重读 | FP16/row-major | API/Contiguous | geometry helper | executor，只读 | 标量 load 后升为 FP32。 |
| GM `x` | `(batch*C+c)*H*W+h*W+w` | bin 面积个标量 | FP16/NCHW | API/Contiguous | max helper | executor，只读 | 标量 load 后升为 FP32。 |
| GM `y` | `idx` | 1 FP16 | FP16/[R,C,PH,PW] | thread cast/store | ViewCopy | internal tensor | 唯一写者。 |
| GM `argmax` | `idx` | 1 INT32 | INT32/同 y | thread cast/store | ViewCopy | internal tensor | 唯一写者。 |
| 标量寄存器 | thread-private | 同 R-001 | FP32/int64/uint32 | 当前 thread | 当前 thread | 单 idx 迭代 | 固定仓不可见具体字节数。 |
| user workspace | opaque | `GetLibApiWorkSpaceSize()` | bytes | runtime | entry non-null check | executor run | body 消费 0 B。 |

##### 分层内存布局

N/A：SIMT 直接 GM 标量 load/store；没有显式 AIV UB、shared memory 或 dynamic UB。

##### 字节账与复用顺序

```text
explicit_AIV_UB = 0 B
dynamic_ub = N/A
workspace_total = PlatformAscendC::GetLibApiWorkSpaceSize()
workspace_bytes_consumed_by_body = 0 B
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private accumulator | 当前 idx | cast/store | 下一 idx | 程序顺序 | 完成两路 store 后覆盖。 |
| GM output `idx` | 唯一 thread | ViewCopy | Kernel 完成后 | executor stream 顺序 | 线程域互斥。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Kernel 前后 | executor 同 stream | 框架节点 | 后续节点 | internal GM | 与 R-001 相同。 |
| 线程域 | 无 barrier/atomic | thread | direct store | `idx` | grid-stride `idx<count` tail。 |
| geometry/max | FP16→FP32 | GM load | FP32 floor/ceil/compare | 坐标与最大值 | 降低 half 中间运算误差；最终 `static_cast<half>`，round mode 的底层细节固定仓不可见。 |
| tie/empty | strict `>`、sentinel | scan/range guard | output | bin | 有限并列取首个；空域 `0/-1`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/roi_pooling_with_arg_max_tiling.cpp` | `GetDataTypeKey/SetTilingData` | FP16 key 与公共 POD/blockDim。 |
| Kernel entry/router | `op_kernel/roi_pooling_with_arg_max.cpp` | `roi_pooling_with_arg_max<1>` | 1024-thread half VF_CALL。 |
| Device body/helper | `op_kernel/arch35/roi_pooling_with_arg_max_simt.h` | `RoiPoolingWithArgMaxCompute<half>`、两个 helper | FP16 GM、FP32 geometry/compare、half 写回。 |

## 算子特定优化

### Trick T-001：magic/shift 解码平坦输出索引

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：所有非空有效调用；`poolW/poolH/channels` 均为正并可安全窄化到 uint32。
- Golden 来源：[`objdetect/roi_pooling_with_arg_max/op_kernel/roi_pooling_with_arg_max.cpp:35-38`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_kernel/roi_pooling_with_arg_max.cpp#L35-L38)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    uint32_t mPoolW = 0, shiftPoolW = 0, mPoolH = 0, shiftPoolH = 0, mCH = 0, shiftCH = 0;
    GetUintDivMagicAndShift(mPoolW, shiftPoolW, static_cast<uint32_t>(tilingData->poolW));
    GetUintDivMagicAndShift(mPoolH, shiftPoolH, static_cast<uint32_t>(tilingData->poolH));
    GetUintDivMagicAndShift(mCH, shiftCH, static_cast<uint32_t>(tilingData->channels));
```

#### 机制与设计原因

entry 每个 block 只计算一次三组除数的 magic/shift，再将它们作为参数传给所有 1024 个 SIMT threads。body 的 `Simt::UintDiv(uidx,m,shift)` 依次得到除以 `poolW`、`poolH` 和 `channels` 的商，并以 `uidx-divW*poolW` 等式恢复余数，从而把平坦输出编号解码为 `(n,c,ph,pw)`。这把热点 grid-stride 循环内的三个不变整数除数显式变成可复用参数；固定源码能证明参数复用和调用替换，未做性能量化。

#### 迁移边界

- 适用：输出逻辑轴为正且可铺平成 uint32 索引的 SIMT elementwise/局部归约映射。
- 前提：除数与 `idx` 必须可无损表示为 uint32，解码轴顺序必须保持 `[n,c,ph,pw]`。
- 失效：输出域或任一除数超出 uint32 时需要新增 64 位解码 Route；本固定仓没有该 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

将大于 `UINT32_MAX` 的 `idx/pool/channel` 直接窄化会改变解码结果；固定仓没有对应 64 位实现或可验证的大尺寸反例。

### Trick T-002：每个输出点独占线程并融合范围计算、归约与双输出

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：所有非空输出；每个 grid-stride `idx` 唯一映射到一个 `(n,c,ph,pw)`。
- Golden 来源：[`objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_simt.h:91-104`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_simt.h#L91-L104)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
  const int64_t count = static_cast<int64_t>(roi_number) * static_cast<int64_t>(channels) * poolH * poolW;
  for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < count;
       idx += gridDim.x * blockDim.x) {
    const uint32_t uidx = static_cast<uint32_t>(idx);
    const uint32_t divW = Simt::UintDiv(uidx, mPoolW, shiftPoolW);
    const uint32_t pw = uidx - divW * upoolW;
    const uint32_t divH = Simt::UintDiv(divW, mPoolH, shiftPoolH);
    const uint32_t ph = divW - divH * upoolH;
    const uint32_t n = Simt::UintDiv(divH, mCH, shiftCH);
    const uint32_t c = divH - n * uchan;

    const __gm__ dT* offset_rois = rois + n * ROI_BOX_ELEMS;
```

#### 机制与设计原因

一个线程在标量寄存器中完成 ROI 读取、bin 边界、bin 内最大值/argmax 归约，并直接提交两路 GM 输出。由于不同线程的 `idx` 不重叠，不需要 atomic、barrier、跨线程归约、UB 中间结果或第二个 Kernel；grid-stride 还让固定 1024-thread block 数覆盖任意 `count` 尾部。该机制减少的是可由源码证明的中间物化和同步，而不是对 GM 输入扫描量作无证据承诺。

#### 迁移边界

- 适用：每个输出的输入窗口互相只读、输出位置独占，且单线程顺序扫描窗口可接受的局部归约。
- 前提：窗口内比较必须保持行主序和严格 `>`，才能维持首个最大值语义；输出 `idx` 不得重叠。
- 失效：若一个 bin 需要跨线程协作或不同输出会更新同一地址，则必须引入 barrier/分级归约/atomic 或额外 Kernel，不能直接复用此同步模型。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把同一 bin 拆给多个线程却仍沿用无同步的 `max_val/max_idx` 会破坏确定性和正确性；固定仓未提供这种替代实现的可验证反例代码。

## 模板与实现对比

| 对比维度 | R-001 FP32 SIMT | R-002 FP16 SIMT | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `x.dtype=FP32`，key 0 | `x.dtype=FP16`，key 1 | ACLNN 同 dtype 检查与 Tiling switch。 |
| 模板与 dtype | `Compute<float>` | `Compute<half>` | binary config 分别提供 float32/float16 项。 |
| 循环与分核 | `useCoreNum` blocks×1024 threads，grid-stride | 相同 | 只按总输出量和 AIV 核数分核。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB；FP32 标量寄存器 | 无 Queue/TBuf/dynamic UB；half load、FP32 标量寄存器 | SIMT 直接 GM。 |
| 内存总占用与复用 | 显式 UB 0 B；workspace body 0 B；每输出读 5 个 ROI 字段并扫描 bin | 同结构，GM 元素为 2 B | 没有 dtype 专属缓存 Route。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `VF_CALL`、GM float load/store、magic UintDiv | `VF_CALL`、GM half load/half store、magic UintDiv | 无 Vector/NDDMA/通信节点。 |
| 同步、精度与 tail | 无线程间同步；FP32 compare/store；grid tail | 无线程间同步；FP16→FP32 compare→FP16；同 tail | dtype 只改变加载/写回精度。 |
| fallback/失败边界 | 无 shape fallback；invalid dtype/attrs/SoC 失败 | 同 | 32 位解码越界没有 Route。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档明确不支持。 | 公开 ACLNN 文档明确不支持。 | Ascend 950PR/950DT 明确支持。 |
| Host/Launcher | 不发布卡。 | 不发布卡。 | RegBase 检查；arch35 Tiling；FP32/FP16 dtype key。 |
| Kernel/资源 | 无公开 ACLNN 可达实现。 | 无公开 ACLNN 可达实现。 | AIV-only entry 内 1024-thread SIMT，直接 GM，无显式 UB。 |

## 限制与失败边界

- 公开文档包含比 ACLNN `CheckParams` 更强的输入约束，例如输入非空、`num_rois<=1024`、ROI 坐标非负及 batch/ROI 维关系；调用者不能把外层未重复检查理解为支持越界值。尤其 `roi_batch_ind` 没有 Device 侧范围检查，非法值会形成越界 GM 地址。
- Kernel 将 POD 的 `channels/H/W/roiNumber/poolH/poolW` 窄化成 int32，并将循环 `idx` 窄化成 uint32 解码；Tiling 没有选择 64 位 Route，也没有显式乘法溢出检查。
- `roi_actual_num` 在 ACLNN 中固定构造为 `{num_rois}`，当前 SIMT body 不读取它；不能据此推导出按 batch 提供不同有效 ROI 数的公开语义。
- `RoiPoolingBinMax` 以 `-INFINITY` 初始化并仅在 `val>max_val` 时更新：有限并列值保留首个；NaN 不更新；若非空 bin 全为 `-INFINITY`，值保持 `-INFINITY` 而索引保持 `-1`。这些边界未在公开文档中单独说明。
- FP16 的最终 `static_cast<half>` 底层舍入模式、`Simt::VF_CALL` runtime 的寄存器分配细节、`GetLibApiWorkSpaceSize()` 的具体字节数及 executor 内部 workspace 子切片均为固定仓不可见。
- ascend950 binary JSON 的 float32 项把内部 `pool_channel` attr 标为 `float`，而 OpDef、Level0 调用和 float16 项均为整数；当前 arch35 Tiling 不消费 attr 4，本文不将该元数据差异扩展为额外 Route。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `objdetect/roi_pooling_with_arg_max/docs/aclnnRoiPoolingWithArgMax.md` | 产品支持表、函数原型、参数表 | 950-only 公开支持、语义、约束。 |
| 入口声明/约束 | `objdetect/roi_pooling_with_arg_max/op_api/aclnn_roi_pooling_with_arg_max.h` | 两个 `ACLNN_API` 声明 | ABI 参数顺序。 |
| 入口实现 | `objdetect/roi_pooling_with_arg_max/op_api/aclnn_roi_pooling_with_arg_max.cpp` | `CheckSocValid/CheckParams/aclnn...GetWorkspaceSize/aclnn...` | 校验、Contiguous、内部参数、executor、ViewCopy。 |
| 入口实现 | `objdetect/roi_pooling_with_arg_max/op_api/roi_pooling_with_arg_max.cpp` | `l0op::RoiPoolingWithArgMax` | 内部输出分配、infer、单 AICore launcher 节点。 |
| Host/Tiling/Launcher | `objdetect/roi_pooling_with_arg_max/op_host/roi_pooling_with_arg_max_infershape.cpp` | `InferShape4RoiPoolingWithArgMax/InferDataType4...` | 输出 shape/dtype 与部分上限检查。 |
| Host/Tiling/Launcher | `objdetect/roi_pooling_with_arg_max/op_host/arch35/roi_pooling_with_arg_max_tiling.cpp` | `RoiPoolingWithArgMaxTiling::*` | POD producer、blockDim、dtype key、local memory/workspace。 |
| Host/Tiling/Launcher | `objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_tiling_data.h` | `RoiPoolingWithArgMaxRegBaseTilingData` | POD 字段顺序和类型。 |
| Host/Tiling/Launcher | `objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | FP32/FP16 模板枚举。 |
| Kernel entry/body | `objdetect/roi_pooling_with_arg_max/op_kernel/roi_pooling_with_arg_max.cpp` | `roi_pooling_with_arg_max<dType>` | AIV-only、workspace guard、magic/shift、VF_CALL router。 |
| Kernel entry/body | `objdetect/roi_pooling_with_arg_max/op_kernel/arch35/roi_pooling_with_arg_max_simt.h` | `RoiPoolingWithArgMaxCompute/BinRange/BinMax` | 线程映射、几何、归约、精度、tail。 |
| 构建接线 | `objdetect/roi_pooling_with_arg_max/op_host/roi_pooling_with_arg_max_def.cpp` | `RoiPoolingWithArgMax::RoiPoolingWithArgMax` | dtype/format/attr、ascend950 注册。 |
| 构建接线 | `objdetect/roi_pooling_with_arg_max/op_host/config/ascend950/roi_pooling_with_arg_max_binary.json` | `op_list` | FP32/FP16 binary 实例。 |
| 构建接线 | `objdetect/roi_pooling_with_arg_max/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR/add_all_modules_sources` | ascend950、arch35、自定义 ACLNN 接线。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六个 int64 shape/attr 后接两个 float scale | 字段顺序、56 B 布局、Host 清零、entry int32/uint32 窄化 | `RoiPoolingWithArgMaxRegBaseTilingData`、`SetTilingData` |
| 调用顺序 | Contiguous→单 Kernel→两路 ViewCopy | executor 同 stream 顺序；任何失败须阻断后续节点 | `aclnn...GetWorkspaceSize`、`l0op::RoiPoolingWithArgMax` |
| workspace/内存 | SIMT 直接 GM，body 不消费 workspace | entry 的 userWS 非空 guard；内部输出保持到 ViewCopy 完成 | `roi_pooling_with_arg_max` |
| 同步 | 每个 `idx` 独占，无 atomic/barrier | grid-stride 映射和 Kernel→ViewCopy stream ordering | `RoiPoolingWithArgMaxCompute` |
| 精度与尾块 | FP32 几何/比较；FP16 路线仅 GM 边界降精度；`idx<count` tail | `floor/ceil/clamp`、strict `>`、空域 `0/-1` 不得改变 | `RoiPoolingBinRange/BinMax/Compute` |
| 失败边界 | SoC/dtype/rank/shape/attr/Tiling 失败路径可复用 | 不得为大索引、非法 batch 或 null workspace 臆造 fallback | `CheckSocValid/CheckParams/DoTiling`、entry guard |
