---
type: Operator
title: "UpsampleNearestExact3dBackward 950 ACLNN 算子知识"
description: "说明 aclnnUpsampleNearestExact3dBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/docs/aclnnUpsampleNearestExact3dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_host/op_api/aclnn_upsample_nearest_exact3d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_host/op_api/aclnn_upsample_nearest_exact3d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_host/op_api/upsample_nearest_exact3d_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/common/inc/op_api/aclnn_check.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/common/src/op_host/tiling_util.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_host/upsample_nearest_exact3d_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_data_copy.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_host/config/ascend950/upsample_nearest_exact3d_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest_exact3d_grad/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T20:56:08Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UpsampleNearestExact3dBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize` + `aclnnUpsampleNearestExact3dBackward`，固定仓唯一公开版本，入口类型 ACLNN |
| 计算语义 | 对每个 `gradInput[n,c,D,H,W]`，累加其最近邻正向映射区间内全部 `gradOut[n,c,d,h,w]`；Exact 边界使用 `ceil(index * scale - 0.5)` |
| 输入输出 | 5-D `gradOut`/`gradInput`，公开 dtype 为 FP32/FP16/BF16；公开格式为 NCDHW/NDHWC，核心 Kernel 统一处理 NCDHW |
| 执行类型 | 单个算子主 Kernel：R-001 为 `AIV_VECTOR` 数据搬运；R-002～R-007 为 AIV-only entry 内 `asc_vf_call` 启动的 SIMT；无 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 identity copy；R-002/R-003 DHW + uint32/uint64；R-004/R-005 CDHW + uint32/uint64；R-006/R-007 NCDHW + uint32/uint64 |
| 核心资源 | R-001：GM + 1 个双缓冲 `TQueBind`；SIMT 路线：GM 直访、线程局部标量/FP32 累加器；主算子固定申请 16 MiB user workspace 但 body 仅判空 |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 launcher→950 Tiling/POD→模板 key→AIV copy/SIMT body→950 binary/构建接线闭合；NDHWC Transpose、Contiguous、ViewCopy 是明确标注的框架前后处理边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令核心 NCDHW 输入 `gradOut` 空间形状为 `(inD,inH,inW)`，输出 `gradInput` 空间形状为 `(outD,outH,outW)`。950 Tiling 取 `scaleD=inD/outD`、`scaleH=inH/outH`、`scaleW=inW/outW`，若 ACLNN 三个 scale 全为正则以显式值覆盖。对输出坐标 `(D,H,W)`：

```text
origD   = min(ceil(D       * scaleD - 0.5), inD)
origDUp = min(ceil((D + 1) * scaleD - 0.5), inD)
origH/HUp、origW/WUp 同理
gradInput[n,c,D,H,W]
  = Σ(d=origD..origDUp-1) Σ(h=origH..origHUp-1) Σ(w=origW..origWUp-1)
      float(gradOut[n,c,d,h,w])
```

FP16/BF16/FP32 输入均在 `float grad` 中累加，最后 `static_cast<T1>` 写回原 dtype。恒等空间尺寸且三轴 scale 均在 `1e-8` 内等于 1 时，不做求和，直接逐字节复制。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOut` | 5-D，NCDHW 或 NDHWC；核心前转 NCDHW | FP32/FP16/BF16 | 非空；shape 必须等于 `[N,C,outputSizeD,H,W]`（NDHWC 为 `[N,D,H,W,C]`）；非连续输入先 `Contiguous` |
| `outputSize` | 长度 3 | INT64 array | 三项均大于 0，并与 `gradOut` 的 D/H/W 相等；即使显式 scales 全正，API 校验阶段仍要求该数组非空且匹配，之后才替换为空数组传给内部算子 |
| `inputSize` | 长度 5 | INT64 array | `[N,C,outD,outH,outW]`，各空间维大于 0；N/C 与 `gradOut` 和 `gradInput` 一致 |
| `scalesD/H/W` | 标量 | double | 若三者全正，统一转为 float 列表并覆盖 Tiling 的 shape 比值；否则内部 scales 列表统一写 0，按 shape 比值计算；有效比值必须 `<=50` |
| `gradInput` | 5-D，与调用者格式一致，可非连续 | 与 `gradOut` 相同 | shape 等于 `inputSize`（NDHWC 调整轴序）；核心结果经可选反向 Transpose 和 `ViewCopy` 提交 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段接口返回完整执行器图及 workspace；950 主算子 Tiling 的一个 slice 固定为 16 MiB |
| `workspace/workspaceSize/executor/stream` | 第二段输入 | opaque | `CommonOpExecutorRun` 在指定 stream 上执行第一段建立的所有节点 |

### 入口与版本

固定 Commit 只公开一组 `aclnnUpsampleNearestExact3dBackward` 两段式接口，没有 V2/V3 版本。公开文档和 README 均明确支持 Ascend 950PR/950DT；OpDef 为 `ascend950` 选择 `upsample_nearest_exact3d_grad_apt`，binary config 为 FP32/BF16/FP16 各提供一个 950 binary，`ascendc_config.json` 同时列出 `ascend950`，证据无冲突。`IsRegBase()` 将 `DAV_3510` 识别为 950，因此 L0 层不插入非 950 使用的 FP16/BF16→FP32 Cast。

## Host/Launcher 执行流程

### 参数校验与前处理

执行顺序固定为：

1. 检查 `workspaceSize`、`executor` 以及 `gradOut/outputSize/inputSize/gradInput` 非空；检查 dtype 属于 FP32/FP16/BF16 且输入输出一致。
2. 检查 `gradOut` 为 5-D、`outputSize.Size()==3`、`inputSize.Size()==5`；检查正空间维、输入/输出 shape 与 NCDHW 或 NDHWC 轴序一致，并计算显式 scale 或 `gradOutDHW/gradInputDHW` 比值，要求三轴均 `<=50`。
3. 950 命中 `IsRegBase()`，`CheckUplimit` 直接成功。由于前一步要求所有空间维正且 N/C 由非空 shape 匹配，源码中的 `gradOut->IsEmpty()` 成功返回分支在合法 950 请求上不可达。
4. `l0op::Contiguous` 物化连续输入；NDHWC 请求使用 `{0,4,1,2,3}` Transpose 成 NCDHW。若三 scale 全正，内部 `outputSize` 改为空数组并传 `[float(scalesD),float(scalesH),float(scalesW)]`；否则传原 `outputSize` 和 `[0,0,0]`。
5. `UpsampleNearestExact3dGradNcdhw` 分配 NCDHW 结果并加入 `UpsampleNearestExact3dGrad` AICore launcher；950 不做 dtype Cast。NDHWC 请求再用 `{0,2,3,4,1}` Transpose，最后 `ViewCopy` 写入调用者可能非连续的 `gradInput`。

任何分配、转换或 launcher 结果为空均立即返回对应 ACLNN 错误，不再建立后续节点。

### 参数、Workspace 与生命周期

`CREATE_EXECUTOR` 拥有第一段构建的 Contiguous、可选 Transpose、主算子、可选反向 Transpose 与 ViewCopy 节点。第一段末尾 `GetWorkspaceSize()` 返回整图需求并 `ReleaseTo(executor)`；第二段的 `workspace/workspaceSize/stream` 交给 `CommonOpExecutorRun`。主算子 950 Tiling 总是写 `workspaces[0]=16*1024*1024`；entry 调用 `GetUserWorkspace(workspace)` 并在空指针时直接返回，但 AIV copy 与 SIMT body 均未读写该 16 MiB。框架节点各自 workspace offset/extent 和 executor 内部 stream 排序在固定仓中不可见，因此不能把 16 MiB 误写成整条 ACLNN 图的唯一 workspace。

### 启动图与失败传播

```text
GetWorkspaceSize
  -> Contiguous(gradOut)
  -> [NDHWC only] Transpose {0,4,1,2,3}
  -> UpsampleNearestExact3dGrad A5 launcher
       Host: Tiling4UpsampleNearest3dGradTiling
         -> Tiling4UpsampleNearest3dGradRegbase
       Device: upsample_nearest_exact3d_grad<schId,isUint32>
         -> one of Nearest3dGradDataCopy / Nearest3dGradSimt
  -> [NDHWC only] Transpose {0,2,3,4,1}
  -> ViewCopy(result, gradInput)
Run
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

主算子是单 Kernel 节点：输入为连续 NCDHW `gradOut`，输出为新分配 NCDHW `gradInput`，POD 为 96 字节 `UpsampleNearest3dGradRegBaseTilingData`，`blockDim=realCoreNum`。与框架前后处理节点之间由 executor 的同一执行图交接临时 GM tensor；其具体 offset、事件和释放时刻固定仓不可见。任何 Host 校验/Tiling/节点创建失败会阻断后继；Device entry 若 user workspace 为空则无输出地返回，固定源码没有 Device 错误码回传。

## Route 与参数设计

### 分核与资源计算

公共量：`inputSize=gradOut.numel`，`outputSize=gradInput.numel`，`outDHW=outD*outH*outW`，`outCDHW=C*outDHW`，`outNCDHW=N*outCDHW`，`maxNum=AIV_coreNum*2048`。`isUint32=(inputSize<=UINT32_MAX && outputSize<=UINT32_MAX)`。

- 恒等 copy：先令 `realCoreNum=AIV_coreNum`；若 `outputSize <= cacheLineNum*AIV_coreNum`，改为 `floor(outputSize/cacheLineNum)`，再至少取 1。`blkProcessNum=floor(outputSize/realCoreNum)`，前 `tailBlockNum=outputSize%realCoreNum` 个 core 多处理 1 元素。`cacheLineNum=128/sizeof(dtype)`，`oneBlockNum=UB_block_bytes/sizeof(dtype)`，`ubFactor=floor((UB_bytes-UB_block_bytes)/(2*sizeof(dtype))/oneBlockNum)*oneBlockNum`。
- SIMT：sch1/2 使用全部 AIV core；sch3 在 `outNCDHW<AIV_coreNum` 时缩为 `outNCDHW` core。随后 `blkProcessNum=floor(allNum/realCoreNum)`，前 `allNum%realCoreNum` 个 core 多一个逻辑坐标。32 位实例每 core 启动 2048 threads，64 位实例启动 1024 threads；每线程以 `blockDim.x` 为步长覆盖本 core 的逻辑域。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 identity copy | `(schId=0,isUint32=0)` | `outD==inD && outH==inH && outW==inW && abs(scaleD-1)<=1e-8 && abs(scaleH-1)<=1e-8 && abs(scaleW-1)<=1e-8` | R-002/R-003 判定 |
| 2 | R-002 DHW-u32 | `(1,1)` | 非 copy，`outDHW >= coreNum*2048`，且输入与输出 numel 均 `<=UINT32_MAX` | 索引超界转 R-003；域不足转 R-004/R-005 |
| 3 | R-003 DHW-u64 | `(1,0)` | 非 copy，`outDHW >= coreNum*2048`，且任一 numel `>UINT32_MAX` | 域不足转 R-004/R-005 |
| 4 | R-004 CDHW-u32 | `(2,1)` | 非 copy，`outDHW < coreNum*2048 && outCDHW >= coreNum*2048`，且两侧 numel 均 `<=UINT32_MAX` | 索引超界转 R-005；域不足转 R-006/R-007 |
| 5 | R-005 CDHW-u64 | `(2,0)` | 与 R-004 相同域条件，且任一 numel `>UINT32_MAX` | 域不足转 R-006/R-007 |
| 6 | R-006 NCDHW-u32 | `(3,1)` | 非 copy，`outDHW < maxNum && outCDHW < maxNum`，且两侧 numel 均 `<=UINT32_MAX` | 索引超界转 R-007 |
| 7 | R-007 NCDHW-u64 | `(3,0)` | 与 R-006 相同域条件，且任一 numel `>UINT32_MAX` | 无后续 Route；校验/Tiling 失败 |

```text
if identity_shape_and_scale:
    schId=0; isUint32=0; ComputeDataCopy()
else:
    if outDHW >= coreNum*2048: schId=1; dimN=N*C; allNum=outDHW
    else if C*outDHW >= coreNum*2048: schId=2; allNum=C*outDHW
    else: schId=3; allNum=N*C*outDHW; realCoreNum=min(coreNum, allNum)
    isUint32 = (gradOut.numel<=UINT32_MAX && gradInput.numel<=UINT32_MAX)
SetTilingKey(GET_TPL_TILING_KEY(schId,isUint32)); SetBlockDim(realCoreNum)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 identity copy | `(0,0)` | 950；任一公开 dtype；三空间轴和 scale 恒等 | 单 AIV Kernel | `Nearest3dGradDataCopy<DTYPE_GRAD_OUTPUT>` | 双缓冲 bind queue；`DataCopyPad` GM↔UB | 前 `tailBlockNum` core +1；最后 tile 取 `min` |
| R-002 DHW-u32 | `(1,1)` | 950；非恒等；`outDHW>=maxNum`；两侧 numel≤UINT32_MAX | 单 entry→SIMT VF | `Nearest3dGradSimt<T,uint32_t,true,1>` | GM 直访；2048 threads；无 shared/atomic/barrier | core 余数前置；线程步进保护 |
| R-003 DHW-u64 | `(1,0)` | R-002 域，任一 numel>UINT32_MAX | 单 entry→SIMT VF | `Nearest3dGradSimt<T,uint64_t,true,1>` | GM 直访；1024 threads | 同左，64 位地址算术 |
| R-004 CDHW-u32 | `(2,1)` | `outDHW<maxNum<=outCDHW`；两侧 numel≤UINT32_MAX | 单 entry→SIMT VF | `Nearest3dGradSimt<T,uint32_t,true,2>` | 2048 threads；每坐标循环 N | 同 R-002 |
| R-005 CDHW-u64 | `(2,0)` | R-004 域，任一 numel>UINT32_MAX | 单 entry→SIMT VF | `Nearest3dGradSimt<T,uint64_t,true,2>` | 1024 threads；每坐标循环 N | 64 位地址算术 |
| R-006 NCDHW-u32 | `(3,1)` | `outDHW<maxNum && outCDHW<maxNum`；两侧 numel≤UINT32_MAX | 单 entry→SIMT VF | `Nearest3dGradSimt<T,uint32_t,true,3>` | 2048 threads；一坐标一输出 | `realCoreNum=min(coreNum,outNCDHW)` |
| R-007 NCDHW-u64 | `(3,0)` | R-006 域，任一 numel>UINT32_MAX | 单 entry→SIMT VF | `Nearest3dGradSimt<T,uint64_t,true,3>` | 1024 threads；一坐标一输出 | 64 位地址算术 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| RegBase 平台 | `IsRegbaseSocVersion(context)` 即 `DAV_3510` | 选择 950 POD/模板与 AIV core 计数，避免落入 Cube MatMul body | 全部 | 非 950 进入旧 Tiling，须 CV Skill | `RunBigKernelTiling`、`IsRegbaseSocVersion` |
| identity epsilon | 三轴 shape 相等且 `abs(scale-1)<=1e-8` | 只有此时输出逐元素等于输入，才能删除求和 | R-001 | SIMT 路线 | `CalTilingData` |
| copy UB 上限 | `ubFactor=floor((UB-UB_block)/(2*dtypeSize)/oneBlockNum)*oneBlockNum` | 给 depth=2 队列留空间并按 UB block 对齐 | R-001 | 固定仓未提供 fallback；平台 UB/块信息异常由 Init 失败 | `ComputeDataCopy` |
| SIMT 域阈值 | 依次比较 `outDHW`、`outCDHW` 与 `coreNum*2048` | 决定由线程覆盖空间域，还是在线程内串行 N/NC | R-002～R-007 | 下一粒度 Route | `CalTilingData` |
| 索引宽度 | 两侧 numel 同时 `<=UINT32_MAX` | 保证所有展平乘加可在 `T2=uint32_t` 范围内；否则用 uint64 | R-002～R-007 | 同 schId 的 u64 Route | `CheckInputParams`、entry `isUint32` 分支 |
| thread 数 | uint32→2048，uint64→1024 | `calleeInt32/64` 的 `LAUNCH_BOUND` 与 `asc_vf_call` 一致 | SIMT | 编译期固定，无 fallback | `Nearest3dGradSimt::Process` |
| user workspace 非空 | `GetUserWorkspace(workspace)!=nullptr` | entry 在任何 body 前统一判空；尽管 body 不消费该区 | 全部 | Device 直接返回、不写输出 | `upsample_nearest_exact3d_grad` |

### POD/ABI

`UpsampleNearest3dGradRegBaseTilingData` 是普通顺序 C++ struct；按字段声明和自然对齐可闭合为 `sizeof=96`、`alignof=8`。Host 在 `Init` 中对全部 96 字节清零，再由 `FillTilingData` 逐字段写值；无 inactive 字段的未初始化读取。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `UpsampleNearest3dGradRegBaseTilingData` | `blkProcessNum` | int64/64 | offset 0 | 逻辑元素/坐标 per core | `allNum/realCoreNum` 或 copy `outSize/realCoreNum` | copy `totalLength`；SIMT `blkProcessNum` | 余数不含在此字段 |
| 同上 | `lenN` | int64/64 | offset 8 | 个 | `dimN`；sch1 已改为 `N*C` | sch1 循环 NC；sch2 循环 N；sch3 解码 N | sch1 语义是 NC |
| 同上 | `lenC` | int64/64 | offset 16 | 个 | 输入 C | sch2/3 地址计算 | sch1 不读 |
| 同上 | `inD` | int64/64 | offset 24 | 元素 | `gradOut.shape[2]` | SIMT 输入边界/stride | copy 不读 |
| 同上 | `inH` | int64/64 | offset 32 | 元素 | `gradOut.shape[3]` | SIMT 输入边界/stride | copy 不读 |
| 同上 | `inW` | int64/64 | offset 40 | 元素 | `gradOut.shape[4]` | SIMT 输入边界/stride | copy 不读 |
| 同上 | `outD` | int64/64 | offset 48 | 元素 | `gradInput.shape[2]` | SIMT 输出坐标解码/stride | copy 仅 Host route 判定使用 |
| 同上 | `outH` | int64/64 | offset 56 | 元素 | `gradInput.shape[3]` | SIMT 输出坐标解码/stride | copy 不读 |
| 同上 | `outW` | int64/64 | offset 64 | 元素 | `gradInput.shape[4]` | SIMT 输出坐标解码/stride | copy 不读 |
| 同上 | `ubFactor` | int32/32 | offset 72 | dtype 元素 | `ComputeDataCopy` 公式 | copy tile 长度/`InitBuffer` | SIMT 清零且不读 |
| 同上 | `tailBlockNum` | int32/32 | offset 76 | core 个数 | `allNum%realCoreNum` | copy/SIMT 前若干 core +1 | `<realCoreNum` |
| 同上 | `scaleD` | float/32 | offset 80 | 输入坐标/输出坐标 | shape 比值或正显式 scale | `ComputeOrig` | copy 仅用于 Host 判定 |
| 同上 | `scaleH` | float/32 | offset 84 | 同上 | 同上 | `ComputeOrig` | 同上 |
| 同上 | `scaleW` | float/32 | offset 88；尾 padding 至 96 | 同上 | 同上 | `ComputeOrig` | 同上 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 identity copy | `(0,0)` | `upsample_nearest3d_grad_tiling_arch35.cpp::CalTilingData/ComputeDataCopy` | RegBase POD 全字段 | `upsample_nearest_exact3d_grad_apt.cpp::upsample_nearest_exact3d_grad<0,0>` | `Nearest3dGradDataCopy::{Init,Process,CopyIn,CopyOut}` | ascend950 binary JSON + `upsample_nearest_exact3d_grad_apt` OpDef |
| R-002/R-003 DHW | `(1,1)/(1,0)` | `CalTilingData` 首个 SIMT 分支 | 同 POD；`lenN=N*C` | entry `<1,1>/<1,0>` | `Nearest3dGradSimt::Process`→`SimtCompute`→`ComputeForSplitDhw` | 同上 |
| R-004/R-005 CDHW | `(2,1)/(2,0)` | `CalTilingData` 第二分支 | 同 POD | entry `<2,1>/<2,0>` | `SimtCompute`→`ComputeForSplitCdhw` | 同上 |
| R-006/R-007 NCDHW | `(3,1)/(3,0)` | `CalTilingData` else | 同 POD | entry `<3,1>/<3,0>` | `SimtCompute`→`ComputeForSplitNcdhw` | 同上 |

## Kernel 实现

### 公共执行流程

所有 950 Route 由同一模板 entry 解析 RegBase POD，并声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。`GetUserWorkspace` 为空则退出；否则 `schId==0` 构造 AIV copy body，`schId!=0` 按 `isUint32` 构造 SIMT body。SIMT 路线为单 Device 节点内部的 `asc_vf_call`，不是第二个 Host launch 节点。三种 SIMT body 都先以乘法 magic/shift 解码展平坐标，再用 Exact 边界计算输入区间；每个输出地址只由一个线程写，无 atomic、barrier 或跨核归约。

### Route R-001：identity copy

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | 三轴 in/out 相等且三 scale 与 1 的差≤`1e-8` | `schId=0,isUint32=0`，调用 `ComputeDataCopy` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `upsample_nearest_exact3d_grad<0,0>` | `if constexpr(schId==0)` | `Nearest3dGradDataCopy<T>` |

#### Device 数据流

##### 函数级调用链

```text
UpsampleNearestExact3dGrad launcher
  -> upsample_nearest_exact3d_grad<0,0>(x,y,workspace,tiling)
    -> Nearest3dGradDataCopy<T>::Init
    -> Nearest3dGradDataCopy<T>::Process
      -> CopyIn -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；单位除特别标注外为 T 元素
base = blockIdx * floor(total/coreNum) + min(blockIdx, total%coreNum)
count = floor(total/coreNum) + (blockIdx < total%coreNum)
for loop in [0,count) step ubFactor:
    valid = min(ubFactor, count-loop)
    q = AllocTensor<uint8_t>()
    DataCopyPad(q, xGM[(base+loop)*sizeof(T)], blockCount=1, blockLen=valid*sizeof(T))
    EnQue(q); q = DeQue<uint8_t>()
    DataCopyPad(yGM[(base+loop)*sizeof(T)], q, blockCount=1, blockLen=valid*sizeof(T))
    FreeTensor(q)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `(totalOffset+loop)*sizeof(T)` bytes | `valid*sizeof(T)` | byte view of NCDHW T | ACLNN input | `CopyIn` | caller/executor；全 Kernel 有效 | tail 用 `DataCopyPad` |
| `dataQue` UB | queue slot | `ubFactor*sizeof(T)` each ×2 | uint8 byte view | `CopyIn` | `CopyOut` | `TPipe`，Alloc→EnQue→DeQue→Free | depth=2；`ubFactor` 按 UB block 对齐 |
| output GM | 同 input offset | `valid*sizeof(T)` | byte view of NCDHW T | `CopyOut` | 后处理/ViewCopy | executor 临时 tensor | 只写有效字节 |
| user workspace | framework `GetUserWorkspace` offset 固定仓不可见 | 16 MiB slice | opaque | Host Tiling | entry 仅判空 | executor | body 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  slot0["dataQue slot0 · ubFactor*sizeof(T) bytes"]:1
  slot1["dataQue slot1 · ubFactor*sizeof(T) bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * ubFactor * sizeof(T)
ubFactor = floor((UB_bytes - UB_block_bytes) / (2*sizeof(T)) / oneBlockNum) * oneBlockNum
UB_total <= UB_bytes - UB_block_bytes <= UB_bytes
dynamic_ub = N/A；AIV TQue 路径
workspace_total(main node) = 16*1024*1024 bytes（仅 entry 判空）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQue` 两个 slot | `CopyIn` | `CopyOut` | 后续 tile `AllocTensor` | `EnQue/DeQue/FreeTensor` | 不得在 `FreeTensor` 前复用；每次 `blockLen` 仅为 valid bytes |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→GM | bind queue + EnQue/DeQue | `CopyIn` | `CopyOut` | 当前 queue slot | 无数值转换；最后 tile `valid=min(...)`；byte blockLen 精确 |
| core tail | 前置余数分配 | Host POD | `Process` | core 地址区间 | 前 `tailBlockNum` core 各多 1 元素，无重叠 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `ComputeDataCopy/CalTilingData` | copy 条件、core/tile/UB 公式 |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<0,0>` | 映射到 `Nearest3dGradDataCopy<T>` |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_data_copy.h` | `Init/Process/CopyIn/CopyOut` | 双缓冲、字节地址、DataCopyPad、tail |

### Route R-002：DHW-u32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | 非 copy；`outDHW>=coreNum*2048`；两侧 numel≤UINT32_MAX | `schId=1,isUint32=1`；`lenN=N*C` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<1,1>` | `isUint32==1` | `Nearest3dGradSimt<T,uint32_t,true,1>` |

#### Device 数据流

##### 函数级调用链

```text
entry<1,1> -> Nearest3dGradSimt<T,uint32_t,true,1>::Init/Process
  -> asc_vf_call<calleeInt32>(2048 threads)
    -> SimtCompute -> ComputeOrig ×6 -> ComputeForSplitDhw
```

##### 源码等价伪码

```text
coreRange = balanced_partition(outDHW)
for local=threadIdx.x; local<coreRange.count; local+=2048:
    flatDHW=coreRange.start+local
    (D,H,W)=magic_div_decode(flatDHW,outH,outW)
    [d0,d1),[h0,h1),[w0,w1)=ceil_exact_ranges(...)
    for nc in [0,N*C):
        grad=float(0)
        for d,h,w in source_box: grad += float(x[nc,d,h,w])
        y[nc,D,H,W]=T(grad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `nc*inD*inH*inW+d*inH*inW+h*inW+w` | 每次 1 T | NCDHW | executor | SIMT load | caller/executor | 无 UB staging |
| output GM | `nc*outD*outH*outW+flatDHW` | 每次 1 T | NCDHW | unique SIMT thread | 后处理 | executor | 无写冲突 |
| thread local | 寄存器生命周期，物理字节固定仓不可见 | 坐标/magic/shift + FP32 `grad` | uint32 + float | thread | same thread | 一次坐标/一次 nc | 无共享/alias |
| user workspace | offset 不可见 | 16 MiB | opaque | Host | entry 判空 | executor | 未消费 |

##### 分层内存布局

N/A：SIMT 路线不初始化 AIV UB/TQue，也未声明 dynamic/shared UB；数据由线程直接 GM load/store，线程局部标量的物理寄存器布局固定仓不可见。

##### 字节账与复用顺序

```text
UB_total = 0（算子 body 未分配 UB）
dynamic_ub = 0 / 未传入（asc_vf_call 仅给 dim3(2048)）
workspace_total(main node) = 16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `grad` | 当前 nc 的 source-box 循环 | 当前输出 store | 下一个 nc 重置为 0 | 单线程程序序 | 一个线程独占当前 flatDHW 的所有 NC 输出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 坐标循环 | 无 barrier/atomic | 唯一 thread | 自身 store | 一个 `(D,H,W)` 的 NC 输出列 | 输入转 FP32 顺序累加，末尾 cast T；线程步长保护 core tail |
| Exact 边界 | `ceilf(idx*scale-0.5f)` + `min(limit)` | thread | source loops | 读区间 | 半像素边界；区间可为空则写 0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | DHW threshold、NC flatten、u32 key |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<1,1>` | uint32 SIMT 实例 |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt.h` + `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `Process/SimtCompute/ComputeForSplitDhw` | 2048 threads、地址、求和和 cast |

### Route R-003：DHW-u64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData` | 非 copy；`outDHW>=coreNum*2048`；输入或输出 numel>`UINT32_MAX` | `schId=1,isUint32=0`；`lenN=N*C` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<1,0>` | `isUint32!=1` | `Nearest3dGradSimt<T,uint64_t,true,1>` |

#### Device 数据流

##### 函数级调用链

```text
entry<1,0> -> Nearest3dGradSimt<T,uint64_t,true,1>::Init/Process
  -> asc_vf_call<calleeInt64>(1024 threads)
    -> SimtCompute -> ComputeOrig ×6 -> ComputeForSplitDhw
```

##### 源码等价伪码

```text
coreRange = balanced_partition(outDHW)             # uint64 index
for local=threadIdx.x; local<coreRange.count; local+=1024:
    flatDHW=coreRange.start+local
    (D,H,W)=uint64_magic_div_decode(flatDHW)
    box=exact_ranges(ceil(index*scale-0.5), input_limits)
    for nc in [0,N*C):
        grad=float(0); for d,h,w in box: grad += float(x[uint64_offset])
        y[uint64_offset]=T(grad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `uint64(nc*inDHW+d*inHW+h*inW+w)` | 每次 1 T | NCDHW | executor | SIMT load | executor | 64 位展平地址 |
| output GM | `uint64(nc*outDHW+flatDHW)` | 每次 1 T | NCDHW | unique thread | 后处理 | executor | 无冲突 |
| thread local | 物理字节不可见 | uint64 坐标/magic/shift + FP32 grad | register scalar | thread | same thread | 当前坐标 | 无 shared/UB |
| user workspace | offset 不可见 | 16 MiB | opaque | Host | entry 判空 | executor | 未消费 |

##### 分层内存布局

N/A：无 AIV UB、dynamic UB 或 shared buffer；GM 直访，线程局部寄存器物理布局固定仓不可见。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0/未传入
workspace_total(main node)=16 MiB（body 未读写）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `grad` | source-box loop | output store | 下一 nc 清零 | 单线程程序序 | flatDHW 在 core/thread 间唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 计算/store | N/A（无 atomic/barrier） | unique thread | 自身 store | NC×当前 DHW | FP32 累加→T；`local<count` 保护尾部 |
| 地址 | uint64 magic division | `GetUintDivMagicAndShift` | `Simt::UintDiv` | 大张量坐标 | 避免 32 位溢出；线程数固定 1024 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData` | 任一 numel 超 UINT32 选 u64，DHW 拆分 |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<1,0>` | uint64 实例 |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt.h` + `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `calleeInt64/ComputeForSplitDhw` | 1024-thread u64 循环 |

### Route R-004：CDHW-u32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | 非 copy；`outDHW<maxNum && C*outDHW>=maxNum`；两侧 numel≤UINT32_MAX | `schId=2,isUint32=1` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<2,1>` | uint32 branch | `Nearest3dGradSimt<T,uint32_t,true,2>` |

#### Device 数据流

##### 函数级调用链

```text
entry<2,1> -> Nearest3dGradSimt<T,uint32_t,true,2>::Process
  -> asc_vf_call<calleeInt32>(2048 threads)
    -> SimtCompute -> ComputeOrig ×6 -> ComputeForSplitCdhw
```

##### 源码等价伪码

```text
coreRange = balanced_partition(C*outDHW)
for local=threadIdx.x; local<coreRange.count; local+=2048:
    flatCDHW=coreRange.start+local
    (C,D,H,W)=magic_div_decode(flatCDHW,outD,outH,outW)
    box=exact_ranges(D,H,W)
    for n in [0,N):
        grad=float(0); for d,h,w in box: grad += float(x[n,C,d,h,w])
        y[n,C,D,H,W]=T(grad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `(n*C+Cidx)*inDHW+d*inHW+h*inW+w` | 1 T/load | NCDHW | executor | thread | executor | uint32 offsets |
| output GM | `n*C*outDHW+flatCDHW` | 1 T/store | NCDHW | unique thread | 后处理 | executor | 每个 C,D,H,W thread 写 N 个独占地址 |
| thread local | register bytes 不可见 | uint32 coords + float grad | scalar | thread | same | 每个 n 重置 | 无 UB/shared |
| user workspace | 不可见 offset | 16 MiB | opaque | Host | entry 判空 | executor | 未消费 |

##### 分层内存布局

N/A：SIMT GM 直访，无 AIV UB/TQue、dynamic UB 或 shared buffer。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0/未传入
workspace_total(main node)=16 MiB（仅非空 guard）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `grad` | 当前 n 的 box loop | `y[n,C,D,H,W]` store | 下一 n 清零 | 程序序 | 每个 flatCDHW 仅一个 thread |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | unique thread | store | 当前 CDHW ×全 N | FP32 累加→T；core 余数前置，thread loop 保护 |
| Exact range | ceilf/min | thread | nested loop | 输入合法区 | 不读取 `>=inD/H/W` 地址 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | CDHW 阈值和 key |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<2,1>` | uint32 sch2 |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `SimtCompute/ComputeForSplitCdhw` | C/D/H/W 解码、外循环 N |

### Route R-005：CDHW-u64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData` | R-004 域条件；输入或输出 numel>`UINT32_MAX` | `schId=2,isUint32=0` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<2,0>` | uint64 branch | `Nearest3dGradSimt<T,uint64_t,true,2>` |

#### Device 数据流

##### 函数级调用链

```text
entry<2,0> -> Nearest3dGradSimt<T,uint64_t,true,2>::Process
  -> asc_vf_call<calleeInt64>(1024 threads)
    -> SimtCompute -> ComputeOrig ×6 -> ComputeForSplitCdhw
```

##### 源码等价伪码

```text
coreRange=balanced_partition(C*outDHW)              # uint64
for local=threadIdx.x; local<coreRange.count; local+=1024:
    (C,D,H,W)=uint64_magic_div_decode(coreRange.start+local)
    for n in [0,N):
        grad=float(0); for d,h,w in exact_box: grad+=float(x[uint64_offset])
        y[uint64_offset]=T(grad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | uint64 NCDHW formula | 1 T/load | NCDHW | executor | SIMT | executor | 大张量地址 |
| output GM | `uint64(n*C*outDHW+flatCDHW)` | 1 T/store | NCDHW | unique thread | 后处理 | executor | 无冲突 |
| thread local | 物理字节不可见 | uint64 coords + float grad | scalar | thread | same | per n | 无 shared/UB |
| user workspace | 不可见 offset | 16 MiB | opaque | Host | entry 判空 | executor | 未消费 |

##### 分层内存布局

N/A：SIMT GM 直访，无显式 local/shared/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0/未传入; workspace_total(main node)=16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `grad` | 当前 n | output store | 下一 n | 程序序 | flatCDHW 独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | thread | store | CDHW×N | FP32 顺序累加→T；1024-thread stride；uint64 offset |
| core tail | balanced remainder | POD | thread loop | core ranges | 前 `tailBlockNum` core +1 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData` | CDHW + u64 |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<2,0>` | uint64 sch2 |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `calleeInt64/ComputeForSplitCdhw` | 1024 threads、N loop、uint64 地址 |

### Route R-006：NCDHW-u32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | 非 copy；`outDHW<maxNum && C*outDHW<maxNum`；两侧 numel≤UINT32_MAX | `schId=3,isUint32=1`；`realCoreNum=min(coreNum,outNCDHW)` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<3,1>` | uint32 branch | `Nearest3dGradSimt<T,uint32_t,true,3>` |

#### Device 数据流

##### 函数级调用链

```text
entry<3,1> -> Nearest3dGradSimt<T,uint32_t,true,3>::Process
  -> asc_vf_call<calleeInt32>(2048 threads)
    -> SimtCompute -> ComputeOrig ×6 -> ComputeForSplitNcdhw
```

##### 源码等价伪码

```text
coreRange=balanced_partition(N*C*outDHW)
for local=threadIdx.x; local<coreRange.count; local+=2048:
    flatNCDHW=coreRange.start+local
    (NC,D,H,W)=magic_div_decode(flatNCDHW,outD,outH,outW)
    grad=float(0)
    for d,h,w in exact_box(D,H,W): grad += float(x[NC,d,h,w])
    y[flatNCDHW]=T(grad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `NC*inDHW+d*inHW+h*inW+w` | 1 T/load | NCDHW | executor | thread | executor | uint32 address |
| output GM | `flatNCDHW` | 1 T/store | NCDHW | unique thread | 后处理 | executor | 每线程唯一 |
| thread local | 物理字节不可见 | uint32 coords + float grad | scalar | thread | same | one output | 无 shared/UB |
| user workspace | offset 不可见 | 16 MiB | opaque | Host | entry 判空 | executor | 未消费 |

##### 分层内存布局

N/A：SIMT 直接访问 GM；无 AIV UB/TQue、dynamic/shared UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0/未传入
workspace_total(main node)=16 MiB（仅 guard）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `grad` | 当前 output 的 box loop | output store | 下一 output 清零 | 程序序 | 展平输出在线程间互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | unique thread | store | 一个 NCDHW 输出 | FP32 累加→T；`local<count` 尾保护 |
| core 数 | `min(coreNum,outNCDHW)` | Tiling | launch | 避免空 core | 非空输入保证至少 1 core |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CalTilingData` | NCDHW fallback、缩 core、u32 key |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<3,1>` | uint32 sch3 |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `ComputeForSplitNcdhw` | 一个 thread 对应一个输出 |

### Route R-007：NCDHW-u64 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData` | R-006 域条件；输入或输出 numel>`UINT32_MAX` | `schId=3,isUint32=0` |
| Router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<3,0>` | uint64 branch | `Nearest3dGradSimt<T,uint64_t,true,3>` |

#### Device 数据流

##### 函数级调用链

```text
entry<3,0> -> Nearest3dGradSimt<T,uint64_t,true,3>::Process
  -> asc_vf_call<calleeInt64>(1024 threads)
    -> SimtCompute -> ComputeOrig ×6 -> ComputeForSplitNcdhw
```

##### 源码等价伪码

```text
coreRange=balanced_partition(N*C*outDHW)            # uint64
for local=threadIdx.x; local<coreRange.count; local+=1024:
    flat=coreRange.start+local
    (NC,D,H,W)=uint64_magic_div_decode(flat)
    grad=float(0); for d,h,w in exact_box: grad+=float(x[uint64_offset])
    y[flat]=T(grad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | uint64 NCDHW formula | 1 T/load | NCDHW | executor | thread | executor | 大张量地址 |
| output GM | uint64 flat NCDHW | 1 T/store | NCDHW | unique thread | 后处理 | executor | 无冲突 |
| thread local | 物理字节不可见 | uint64 coords + float grad | scalar | thread | same | one output | 无 shared/UB |
| user workspace | offset 不可见 | 16 MiB | opaque | Host | entry 判空 | executor | 未消费 |

##### 分层内存布局

N/A：无显式 UB、shared 或 dynamic buffer；GM 直访。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0/未传入; workspace_total(main node)=16 MiB（未消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `grad` | source-box loop | output store | 下一 output | 程序序 | flat output 唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | unique thread | store | 一个 NCDHW 输出 | FP32 累加→T；1024-thread stride；uint64 地址 |
| 失败终点 | 无后续 Route | Tiling/entry | N/A | 全请求 | shape/dtype/core/UB/POD 获取失败即 Host Tiling 失败；workspace 空则 Device 返回 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData` | NCDHW + u64 |
| Kernel entry/router | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `<3,0>` | uint64 sch3 |
| Device body/helper | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `calleeInt64/ComputeForSplitNcdhw` | 1024 threads、大地址、单输出 |

## 算子特定优化

### Trick T-001：恒等映射绕过求和并使用 UB 双缓冲拷贝

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：三轴输入/输出尺寸相等，且三轴有效 scale 与 1.0 的绝对差均不超过 `1e-8`。
- Golden 来源：[`image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp:110-124`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp#L110-L124)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bool isDataCopy = baseTiling.outD == baseTiling.inD && baseTiling.outH == baseTiling.inH &&
                  baseTiling.outW == baseTiling.inW && std::abs(baseTiling.scaleD - 1.0f) <= EPSILON &&
                  std::abs(baseTiling.scaleH - 1.0f) <= EPSILON && std::abs(baseTiling.scaleW - 1.0f) <= EPSILON;
baseTiling.realCoreNum = baseTiling.coreNum;
baseTiling.cacheLineNum = CACHE_LINE / baseTiling.dtypeSize;

if (isDataCopy) {
    OP_LOGI(tilingContext, "enter datacopy");
    baseTiling.schId = 0; // 纯copy模板
    baseTiling.isUint32 = 0;
    ComputeDataCopy();
```

#### 机制与设计原因

恒等最近邻反向的每个求和区间恰含同位置一个元素，Host 用专用 key 删除 `ceilf`、坐标解码、三重 source-box 循环和 FP32 cast/accumulate。Device 将 T 元素视作字节，通过 `DataCopyPad` 精确搬运有效 tail；`BUFFER_NUM=2` 和 `ubFactor` 公式让两个 queue slot 在 UB 中交替使用。源码证明其改变了计算和搬运路径，但本次未测量性能收益。

#### 迁移边界

- 适用：数学上输出逐元素等于输入、且布局和 dtype 不变的恒等特化。
- 前提：950；三轴 shape/scale 同时满足严格 guard；`ubFactor>0` 所需 UB 由平台提供。
- 失效：任一轴非恒等立即进入 R-002～R-007，不能近似复制。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能只比较 shape 而忽略显式 scale；源码同时检查三轴 scale。固定仓未提供除此 guard 外的可验证反例。

### Trick T-002：按输出域粒度切换 DHW/CDHW/NCDHW 线程映射

#### 触发与源码依据

- 适用 Route：R-002～R-007
- 触发条件：非恒等；依次比较 `outDHW`、`C*outDHW` 与 `AIV_coreNum*2048`。
- Golden 来源：[`image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp:125-146`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp#L125-L146)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
} else {
    int64_t allNum = 0;
    if (outDHW >= maxNum) {
        OP_LOGI(tilingContext, "enter simt dhw");
        baseTiling.schId = 1; // dhw 分线程， nc拉成一维
        baseTiling.dimN = baseTiling.dimN * baseTiling.dimC;
        allNum = outDHW;
    } else if (outCDHW >= maxNum) {
        OP_LOGI(tilingContext, "enter simt cdhw");
        baseTiling.schId = CONST_2; // cdhw 分线程, 和竞品一致
        allNum = outCDHW;
    } else {
        OP_LOGI(tilingContext, "enter simt ncdhw");
```

#### 机制与设计原因

空间域足够大时，一个线程负责一个 DHW 坐标并在内部复用该坐标和 source-box，串行写全部 NC；空间不足但 CDHW 足够时把 C 纳入线程域，仅串行 N；更小时把 N/C 都纳入线程域，保证每个线程只写一个输出。选择保持源码优先级，避免在大空间情形过早把 N/C 展平而重复边界计算，也避免小空间情形 core/thread 闲置。该结论是源码机制说明，不包含实测收益。

#### 迁移边界

- 适用：输出坐标互不写冲突、每个输出可独立从输入 box 归约的算子。
- 前提：950 SIMT；各轴与展平 numel 可由选择的 uint32/uint64 表达；每线程 source-box 循环可接受。
- 失效：恒等条件改走 R-001；索引位宽变化时必须选同 schId 的另一模板。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把三个 if/else 分支写成无优先级集合；`outDHW>=maxNum` 必须先于 CDHW。固定仓未提供可验证的性能反例。

### Trick T-003：用 numel guard 专门化 32/64 位坐标与线程数

#### 触发与源码依据

- 适用 Route：R-002～R-007
- 触发条件：输入和输出 numel 同时不超过 `UINT32_MAX` 使用 uint32，否则 uint64。
- Golden 来源：[`image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt.h:81-90`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt.h#L81-L90)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (sizeof(T2) == sizeof(uint64_t)) {
    asc_vf_call<calleeInt64<T1, T2, isExtra, schId>>(
        dim3(THREAD_NUM_B64), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
        blkStartOffset, blkProcessNum, lenN, lenC, mD, shiftD, mH, shiftH, mW, shiftW, lenSrcD, lenSrcH, lenSrcW,
        lenDstD, lenDstH, lenDstW, tilingData->scaleD, tilingData->scaleH, tilingData->scaleW);
} else {
    asc_vf_call<calleeInt32<T1, T2, isExtra, schId>>(
        dim3(THREAD_NUM_B32), (__gm__ T1*)(inputGm.GetPhyAddr()), (__gm__ T1*)(outputGm.GetPhyAddr()),
```

#### 机制与设计原因

Tiling 用两侧总元素数证明展平地址能否由 uint32 表达；模板把该选择传播到所有坐标、magic/shift 和地址乘加。uint32 body 配 2048-thread launch bound，uint64 body 配 1024-thread launch bound，避免小张量承担 64 位索引路径，同时为大张量保留地址正确性。

#### 迁移边界

- 适用：所有地址乘积都被总 numel 上界覆盖的展平 tensor。
- 前提：shape 乘积本身在 Host int64 计算中有效；模板 key 与 Host `isUint32` 必须一致。
- 失效：任一 tensor numel 超 UINT32_MAX 时必须使用 uint64 路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只检查输出而不检查输入会使 source offset 可能溢出；源码要求两者同时满足。固定仓未提供其它可验证反例。

## 模板与实现对比

| 对比维度 | R-001 identity | R-002/003 DHW | R-004/005 CDHW | R-006/007 NCDHW | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | shape+scale 恒等，最高 | 非恒等且 DHW≥core×2048 | DHW不足、CDHW足够 | 前两者均不足 | 从复用最多的空间坐标粒度逐步扩展线程域 |
| 模板与 dtype | `DataCopy<T>`，三 dtype | `Simt<T,u32/u64,true,1>` | `Simt<T,u32/u64,true,2>` | `Simt<T,u32/u64,true,3>` | dtype binary 独立；索引位宽由 numel guard |
| 循环与分核 | core→UB tile | core→thread→NC→D/H/W box | core→thread→N→box | core→thread→box | 域越小，越多 N/C 维进入并行坐标 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth=2 bind queue | thread-local coords+grad | 同左 | 同左 | SIMT 无显式 UB/shared/通信 |
| 内存总占用与复用 | `2*ubFactor*sizeof(T)` | UB=0；grad per NC | UB=0；grad per N | UB=0；grad per output | copy 用 UB 隐藏搬运，SIMT 直接 GM |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad` | `asc_vf_call` 2048/1024 | 同左 | 同左 | 无 NDDMA/通信/Cube |
| 同步、精度与 tail | queue；bitwise copy；tile min | 无 atomic；FP32 累加→T | 同左 | 同左 | 唯一 thread 写唯一地址 |
| fallback/失败边界 | 非恒等转 SIMT | 域小转 CDHW | 域小转 NCDHW | 无后继 | 平台资源/输入/POD/workspace 错误失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，但可达主 Kernel 使用 `TCubeTiling`/`REGIST_MATMUL_OBJ`，本 VV 卡不覆盖 | 同 A2 | 公开支持；本卡覆盖 |
| Host/Launcher | `IsRegBase=false`，FP16/BF16 前后 Cast，旧 Tiling key=1 | 同 A2 | `IsRegBase=true`，保持原 dtype，RegBase Tiling 生成 `(schId,isUint32)` key |
| Kernel/资源 | `UpsampleNearest3dGradND<T>` + 三个 MatMul 对象，需 CV Skill | 同 A2 | AIV copy 或 SIMT，单 Kernel，无 Cube/AIC/MIX_CV |

## 限制与失败边界

- ACLNN 必须同时提供非空 `outputSize` 和 `inputSize`；显式 scale 路径也先校验 `outputSize` 与 `gradOut` shape，再在 executor 内把 `outputSize` 替换为空数组。任一 scale 非正会使三个内部 scale 全部归零并统一按 shape 比值处理。
- API 只检查有效 scale `<=50`，正性由“显式 scale 全正，否则用正 shape 比值”间接保证；950 跳过非 RegBase 的 INT32 逐维上限检查。公开文档另列每维≤2^20、总内存<60 GiB 等约束，但这些约束未在 950 `CheckUplimit` 中实现，调用者仍须遵守公开接口约束。
- `inputSize/outputSize` 与 shape 的连乘在 Host 中使用 int64；固定源码没有显式乘法溢出 guard。POD 的 `blkProcessNum` 为 int64，但 `ubFactor/tailBlockNum/coreNum/UB size` 在 Host 中窄化为 int32。
- R-001 的 `DataCopyExtParams.blockLen` 接收 `length*sizeof(T)`；具体字段位宽和单次 DMA 最大 blockLen 属于 AscendC primitive 外部边界，固定仓不可见。Host 公式未提供 `ubFactor==0` fallback。
- SIMT 顺序累加使用 float；FP16/BF16 写回的精确硬件舍入模式没有在源码中显式指定。乘法 magic/shift helper 的实现属于 SIMT API 边界，固定仓不可见。
- 主算子固定申请 16 MiB user workspace，却只在 entry 判空；完整 ACLNN workspace 还可能包含 Contiguous/Transpose/ViewCopy 临时区。框架内部 slice offset、事件、owner 释放点固定仓不可见。
- NDHWC 的 Transpose、非连续 Tensor 的 Contiguous/ViewCopy 是框架前后处理节点；其 Kernel body 不在本算子固定源码范围内。本卡闭合算子拥有的主语义节点，不把框架节点伪造成算子 Kernel。
- A2/A3 虽公开支持，但旧 body 实例化 `TCubeTiling` 和 MatMul 对象，触发 VV→CV 分流；未生成无后缀卡。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `image/upsample_nearest_exact3d_grad/docs/aclnnUpsampleNearestExact3dBackward.md` | 产品表、函数原型、约束 | 950 公开支持、接口 shape/dtype/format |
| 入口声明/约束 | `image/upsample_nearest_exact3d_grad/op_host/op_api/aclnn_upsample_nearest_exact3d_backward.h` | 两段式 API | 精确 C ABI |
| 入口实现 | `image/upsample_nearest_exact3d_grad/op_host/op_api/aclnn_upsample_nearest_exact3d_backward.cpp` | `CheckParams`、两段入口 | 校验、Contiguous/Transpose/ViewCopy、executor |
| 入口实现 | `image/upsample_nearest_exact3d_grad/op_host/op_api/upsample_nearest_exact3d_grad.cpp` | `UpsampleNearestExact3dGradNcdhw` | 950 不 Cast、结果 shape、launcher |
| 平台选择 | `common/inc/op_api/aclnn_check.h` | `IsRegBase` | DAV_3510 的 ACLNN 950 分支 |
| 平台选择 | `common/src/op_host/tiling_util.cpp` | `IsRegbaseSocVersion` | DAV_3510 的 RegBase Tiling 分支 |
| Host 注册 | `image/upsample_nearest_exact3d_grad/op_host/upsample_nearest_exact3d_grad_def.cpp` | `UpsampleNearestExact3dGrad` | ascend950 APT opFile 和 dtype |
| Host/Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling.cpp` | `RunBigKernelTiling`、IMPL registration | 950 selector 到 RegBase Tiling |
| Host/Tiling | `image/upsample_nearest3d_grad/op_host/upsample_nearest3d_grad_tiling_arch35.cpp` | `CheckInputParams/CalTilingData/DoTiling` | Route、分核、key、workspace、POD producer |
| POD | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_tiling_data.h` | `UpsampleNearest3dGradRegBaseTilingData` | POD 顺序和类型 |
| Route key | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_tiling_key.h` | `UPSAMPLE_NEAREST3D_GRAD_*` | 7 个可达模板组合 |
| Kernel entry/body | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad_apt.cpp` | `upsample_nearest_exact3d_grad<schId,isUint32>` | AIV-only entry 和 body router |
| Kernel entry/body | `image/upsample_nearest_exact3d_grad/op_kernel/upsample_nearest_exact3d_grad.cpp` | `upsample_nearest_exact3d_grad`、`REGIST_MATMUL_OBJ` | A2/A3 可达 Cube 路线的 CV 分流证据 |
| Kernel body | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_data_copy.h` | `Nearest3dGradDataCopy` | R-001 UB/DMA/tail |
| Kernel body | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt.h` | `Nearest3dGradSimt` | block 分区、thread launch、magic producer |
| Kernel body | `image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_simt_base.h` | `SimtCompute`、三个 `ComputeForSplit*` | 坐标、Exact box、FP32 累加、store |
| 构建接线 | `image/upsample_nearest_exact3d_grad/op_host/config/ascend950/upsample_nearest_exact3d_grad_binary.json` | 3 个 binary 条目 | 950 FP32/BF16/FP16 binary |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `UpsampleNearestExact3dGrad` | ascend950 compute unit 与 950 compile options |
| 构建接线 | `image/upsample_nearest_exact3d_grad/op_host/CMakeLists.txt` | `add_modules_sources` | Host/API 模块纳入构建 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 96 字节 RegBase POD，Host 先清零后完整填充 | 字段顺序、int64/int32/float 位宽；sch1 的 `lenN=N*C` 特殊语义 | `UpsampleNearest3dGradRegBaseTilingData`、`FillTilingData` |
| 调用顺序 | ACLNN 前处理→主 Kernel→后处理；主 Kernel 内单一 copy 或 SIMT body | NDHWC 两次 permute、950 不插 Cast、entry workspace guard 必须保留 | `GetWorkspaceSize`、`UpsampleNearestExact3dGradNcdhw`、entry |
| workspace/内存 | 主节点 16 MiB；copy depth=2 UB；SIMT GM 直访 | 不能将主节点 16 MiB 当整图唯一需求；copy `ubFactor` 和有效 byte tail 不可变 | `DoTiling`、`ComputeDataCopy`、`Nearest3dGradDataCopy::Init` |
| 同步 | copy 用 queue 生命周期；SIMT 唯一 writer 无 atomic/barrier | 不能让两个 thread/core 覆盖同一展平输出；框架节点顺序由 executor 管理 | `Process`、`SimtCompute` |
| 精度与尾块 | Exact `ceilf(x*scale-0.5)`；float 顺序累加；前置 core 余数 | 不得换成普通 nearest 边界、不得改变累加/写回顺序；u32/u64 thread 数配对 | `ComputeOrig`、`ComputeForSplit*`、`Nearest3dGradSimt::Process` |
| 失败边界 | Host/Tiling 失败阻断 launch；user workspace 空则 entry 返回 | 不可把未知/非法 key、空 workspace 或非 950 平台静默映射为某条 950 Route | `CheckParams`、`DoTiling`、entry |
