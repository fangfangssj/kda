---
type: Operator
title: "MaxPool 950 ACLNN 算子知识"
description: "说明 aclnnMaxPool 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/docs/aclnnMaxPool.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_api/aclnn_max_pool.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_api/aclnn_max_pool.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_api/max_pool_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/max_pool_v3_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_big_kernel_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_big_kernel_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/max_pool_v3_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_small_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_small_kernel_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_big_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_api/aclnn_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/config/ascend950/max_pool_v3_binary.json
    role: implementation
created_at: "2026-08-25T18:21:23Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MaxPool 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMaxPoolGetWorkspaceSize` + `aclnnMaxPool`，公开 ACLNN 两段式接口；本仓只有该版本。 |
| 计算语义 | 对 NCHW/CHW 输入的每个二维滑窗求最大值；padding lane 以该 dtype 的最小值或负无穷参与。 |
| 输入输出 | `self/out` 同 shape 语义、dtype 与 storage format；公开 950 dtype 为 FP16/FP32/BF16/INT32/INT64/UINT8/INT16/INT8/UINT16，3D CHW 或 4D NCHW/ND。 |
| 执行类型 | 三条公开可达 Route 均为单 Kernel、`AIV_VECTOR`；entry 明确 `KERNEL_TYPE_AIV_ONLY`。 |
| Kernel 路线 | R-001 小窗无 padding（key `300001`）；R-002 小窗 padding/ceil 尾窗（`300002`）；R-003 大窗 fallback（`311110`）。 |
| 核心资源 | GM 输入/输出、双缓冲 VECIN/VECOUT、索引 TBuf；padding 路线增加完整负无穷画布 TBuf；大窗路线用双缓冲 VECIN 与 1024 元素输出 TBuf。MaxPoolV3 Host 固定申请 16 MiB 系统 workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→参数整形/Contiguous→`l0op::MaxPoolV3`→模板 Tiling→POD→`max_pool_v3` router→三条 AIV body→950 构建/二进制配置均可追溯。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入为 `x[n,c,h,w]`。对输出位置 `(ho,wo)`，窗口左上角为 `(ho*sH-tPad, wo*sW-lPad)`，输出为窗口与有效输入域交集中元素的最大值：

```text
y[n,c,ho,wo] = max x[n,c,hi,wi]
hi = ho*sH - tPad + kh, 0 <= kh < kH, 0 <= hi < H
wi = wo*sW - lPad + kw, 0 <= kw < kW, 0 <= wi < W
```

`ceilMode=false` 使用向下取整；`ceilMode=true` 先加 `stride-1`，若最后窗口起点落在下/右 padding 区则再减一。950 ACLNN 将 CHW 在轴 0 补成 `1×C×H×W`，Kernel 中 NCHW 的 `N*C` 被折叠为 `batches`。本卡不覆盖 WithIndices、WithMask、3D pooling 语义；源码中仅 DAV_2201 且大核时的 `MaxPool3DWithArgmaxV2` 分支在 950 不可达。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | CHW 或 NCHW；公开文档为 ND，非连续可接受 | FP16/FP32/BF16/INT32/INT64/UINT8/INT16/INT8/UINT16 | 950 由 `DAV_3510` dtype 表放行；私有 storage format 拒绝，C/H/W 不得为 0。 |
| `kernelShape` | 长度 1 或 2 | INT64 array | 值均大于 0；单值复制到 H/W。 |
| `strides` | 长度 0、1 或 2 | INT64 array | 空数组默认为 1；值大于 0；单值复制到 H/W。 |
| `autoPad` | scalar | INT64 | 只允许 0（NOTSET）。 |
| `pads` | 长度 0、1、2 或 4 | INT64 array | 规范化为 `[top,bottom,left,right]`；两侧之和分别不大于 kH/kW；Host Tiling 进一步要求每一侧 `<` 对应 kernel。 |
| `dilations` | 长度 0、1、2 或 4 | INT64 array | 所有给定值只能为 1；Device 不接收 dilation 字段。 |
| `ceilMode` | scalar | INT64 | 0 为 false，非 0 为 true。 |
| `out` | 与输入同 rank/layout，H/W 为推导值；非连续可接受 | 与 `self` 相同 | 空输入/输出在 ACLNN 层直接返回成功且 workspace=0，不启动 MaxPoolV3。 |

### 入口与版本

公开入口声明位于 `op_api/aclnn_max_pool.h`，实现位于 `op_api/aclnn_max_pool.cpp`。950 上 `Ops::NN::AclnnUtil::IsRegbase()` 为真，故不执行旧平台的 NC1HWC0 TransData；4D ND 被明确解释为 NCHW，再调用 `l0op::MaxPoolV3`。`max_pool_v3_def.cpp` 只为 `ascend950` 注册公开可见 AICore 配置；`CMakeLists.txt` 和 binary config 同时接入 `ascend950` 与 `mc62`。本卡只发布 Prompt 指定的 950/950。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `aclnnMaxPoolGetWorkspaceSize` 创建 executor，`CheckParams` 依次做非空、按 SoC dtype、format/rank、数组长度/值和输出 shape 校验。
2. 空 tensor 将 `workspaceSize=0` 后直接交出 executor；否则 `l0op::Contiguous` 物化连续输入。
3. 950 上 `ExecMaxPool3DCapable` 恒因 `curArch != DAV_2201` 为假，进入 `ExecMaxPoolV3`。CHW 经 `UnsqueezeNd(axis=0)` 成 NCHW；4D 保持原 rank。
4. `kernelShapeOps/stridesOps/padsOps` 生成 4 元素属性；ND 四维输入强制 `data_format="NCHW"`。
5. `l0op::MaxPoolV3` 推 shape、分配连续输出并通过 `ADD_TO_LAUNCHER_LIST_AICORE` 加入 `MaxPoolV3`；最后 `ViewCopy` 写用户 `out`。
6. 第二段 `aclnnMaxPool` 将 workspace、executor 和 stream 交给 `CommonOpExecutorRun`。`Contiguous/Unsqueeze/ViewCopy` 是允许的框架边界；算子拥有的 MaxPoolV3 selector、POD 和 Device 均已闭合。

### 参数、Workspace 与生命周期

`TilingPrepare4MaxPoolV3` 从平台取得 AIV 核数与 UB 字节数。`MaxPoolV3BaseTiling::GetWorkspaceSize` 将该 MaxPoolV3 节点的唯一 workspace slice 设置为 `16*1024*1024` 字节；Kernel 参数虽然带 `workspace`，三条公开可达 body 均不读取它，因此这是运行时系统 workspace。整个 executor 的 `workspaceSize` 还可能包含框架 `Contiguous/ViewCopy` 节点需求，其汇总算法属于 executor 框架边界。workspace 由调用者在两段接口之间分配并保持有效；executor 在第二段运行结束前拥有任务图与中间 tensor 描述。

### 启动图与失败传播

```text
aclnnMaxPoolGetWorkspaceSize
  -> [可选框架 Contiguous / UnsqueezeNd]
  -> MaxPoolV3AiCore: ADD_TO_LAUNCHER_LIST_AICORE(MaxPoolV3)
       -> Tiling4MaxPoolV3 -> priority selector -> POD/key/blockDim
       -> max_pool_v3(x, y, workspace, tiling)  # 单 AIV Kernel
  -> [可选框架 SqueezeNd] -> ViewCopy(out)
aclnnMaxPool(workspace, executor, stream)
  -> CommonOpExecutorRun 按同一 stream 的 executor 顺序执行
```

| 节点/边 | producer / consumer | 介质与范围 | ordering / owner / failure |
|---|---|---|---|
| 输入前处理 | 用户 `self` → `Contiguous`/可选 `UnsqueezeNd` → MaxPoolV3 | executor 管理的连续 tensor；CHW 补 N=1 | executor 同 stream 排序并持有；任何 nullptr 立即使第一段返回错误。 |
| MaxPoolV3 | `MaxPoolV3AiCore` → `max_pool_v3` | x/y GM；POD；16 MiB 系统 workspace（Device 未读） | launcher 失败由 `CommonOpExecutorRun` 返回，阻断后续 ViewCopy。 |
| 输出交接 | `max_pool_v3` → 可选 `SqueezeNd` → `ViewCopy` | 连续输出 GM 到用户 out 的 view/stride | executor 同 stream 顺序保证生产后消费；直到执行完毕均由 executor/调用者 workspace 生命周期覆盖。 |

## Route 与参数设计

### 分核与资源计算

记 `s=sizeof(dtype)`、`V=GetVRegSize()` 字节、`B=GetUbBlockSize()/s` 元素、`C0=platform coreNum`、`Nf=N*C`、`Ho/Wo` 为输出空间。

- 小窗模板：`maxGatherElm=V/2`（s=1/2）或 `V/4`（s=4/8）；`indiceBytes=ceil(kH*kW/maxGatherElm)*V`；`availableElems=(UB-512-indiceBytes)/s`。候选 tile 的无 pad 元素账是 `2*(inRows*align(inCols,B)+align(outRows*outCols,B))`，padding 再加一份 input tile。至少能容纳产生 `paraNum=V/(2*s)` 个输出的最小 tile 才可用。
- 小窗 UB 切分按整批、整 W 切 H、再切 W 的顺序选择；如果总 loop 少于核数，以 `4096/s` 元素步长缩小可用 UB 来提高 tile 数，直到 loop 数达到核数或预算不足。`totalLoop=nLoop*hLoop*wLoop`；`blockFactor=floor(totalLoop/C0)`，`blockTail=totalLoop-blockFactor*C0`，实际核数为 `blockFactor==0 ? blockTail : C0`。
- 大窗模板：`totalIdx=Nf*Ho*Wo`，同样以前 `blockTail` 核多处理一个输出。`maxCount=floor_align((UB-1024*s)/2,V)/s`，恰好对应双 input queue 与 1024 元素输出 TBuf。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 小窗无 padding | priority 0；`300001` | NCHW；`Nf*Ho*Wo >= C0`；`IsBufferCapable()`；四侧 pad 为 0，且 ceil 最后窗口不越输入。 | padding/ceil 越界改 R-002；容量或并行谓词失败继续 priority 1/2。 |
| 2 | R-002 小窗 padding | priority 0；`300002` | 同一小窗 `IsCapable` 成功，且任一 pad 非 0，或 `ceilMode && ((Wo-1)*sW+kW>W \|\| (Ho-1)*sH+kH>H)`。 | 小窗失败继续 priority 1/2。 |
| 3 | R-003 大窗 fallback | priority 2；`311110` | 公开 ACLNN 的 NCHW 输入，且 priority 0 小窗返回 `GRAPH_PARAM_INVALID`；priority 1 NHWC 小窗因 format false；priority 2 `IsCapable` 对 NCHW 恒真。 | 非 NCHW 才继续 priority 4 NHWC；公开 ACLNN 路径不可达。 |

```text
priority 0 MaxPoolV3SmallKernelTiling
  if format != NCHW or Nf*Ho*Wo < C0 or !IsBufferCapable: fallthrough
  else if isPadding: key 300002 -> R-002
  else: key 300001 -> R-001
priority 1 NHWC-small: public ACLNN NCHW => fallthrough
priority 2 MaxPoolV3BigKernelTiling
  if NCHW: key 311110 -> R-003
priority 4 NHWC-big: public ACLNN unreachable
else: GRAPH_FAILED
```

`TilingRegistry` 遍历升序 `std::map`，源码明确“priority 越小越高”；模板 `DoTiling()` 只有返回 `GRAPH_PARAM_INVALID` 才继续下一优先级。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `300001` | 950 公开 dtype、NCHW、小窗 capable、无 pad/ceil 越界 | 单 AIV `max_pool_v3` | `MaxPoolV3SmallKernel<T>`；gather mode 0/1/2/3 | 双 VECIN/VECOUT + index TBuf；queue 同步 | n/h/w 尾 tile；稀疏 H 搬运；不 capable 到 R-003。 |
| R-002 | `300002` | 同上但 `isPadding_` 真 | 单 AIV `max_pool_v3` | `MaxPoolV3SmallPadKernel<T>` | R-001 资源外加 `tmpBuf` 和 256B scatter index；负无穷画布后 gather | h/w 有效交集；无效 lane 填 dtype 最小值。 |
| R-003 | `311110` | NCHW 且小窗模板不 capable | 单 AIV `max_pool_v3` | `MaxPoolV3BigKernel<T>`；窗口全载/分 H/分 W | 双 VECIN + 1024 元素 TBuf；V↔MTE3 event | 每输出裁剪窗口；超 `maxCount` 分段并 merge。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 小窗并行度 | `Nf*Ho*Wo >= C0` | 小窗 route 以 tile loop 分核，过少输出不使用该模板 | R-001/R-002 | R-003 | `MaxPoolV3SmallKernelTiling::IsCapable` |
| 小窗 UB | `CalcBufferSize(min tile)<=availableElems` | depth=2 的输入/输出及 padding 额外画布必须落 UB | R-001/R-002 | R-003 | `IsBufferCapable` |
| 16 位索引域 | 单 tile 输入元素被各切分函数限制到 `<=65535` | s=1/2 用 uint16 index，s=4/8 用 uint32；Host 对所有 dtype仍保守限制 input tile | R-001/R-002 | 缩小 H/W/N tile；最终小窗失败则 R-003 | `CalcSplitMaxRows/Cols/Batch` |
| gather 模式 | `kH*kW`、`Ho*Wo`、`Wo` 与 `maxGatherElm` 的有序判断 | 决定索引在寄存器中覆盖单窗/多批/多行/单行 | R-001/R-002 | mode 3→2→1→0 的源码分支 | `CalcGatherMode` |
| padding copy 模式 | `W<=maxGatherElm`、`W<=4*maxGatherElm` | 决定多行 scatter、单行 scatter 或逐行 copy 到负无穷画布 | R-002 | mode 1→0→2 | `CalcCopyMode` / `CopyAndPad` |
| 大窗全载 | `kH*kW <= maxCount`（Device） | 一个窗口可一次装入 VECIN | R-003 | 分 H；若 `curkW>maxCount` 再分 W | `MaxPoolV3BigKernel::Process/SplitKernelProcess` |
| dtype/平台 | ACLNN `DAV_3510` 表 + OpDef/config `ascend950` | 无 AICPU fallback；entry 编译为 AIV-only | 全部 | ACLNN 参数错误 | `GetDtypeSupportListBySocVersion` / `IsAiCoreSupport` / `max_pool_v3` |
| 空域 | ACLNN 在 `self/out IsEmpty()` 时提前成功 | 避免 `blockDim=0` 与无效 Kernel | 全部 | workspace=0，不启动 Device | `aclnnMaxPoolGetWorkspaceSize` |

### POD/ABI

两种 POD 的所有字段均以声明顺序使用 `TILING_DATA_FIELD_DEF(int64_t, ...)`，源码类型/位宽为有符号 64 位。宏展开定义来自仓外运行时头，固定仓不能证明 `sizeof/alignof/offsetof`，故 ABI 布局栏据实标缺；Host 与 Device 使用同名注册结构，字段语义闭合。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| Small | `hInDim` | int64/64 | 宏生成，offset 不可见 | 元素 | `SetTilingData=inputShape[H]` | tile offset/稀疏 copy | >0（非空路径） |
| Small | `wInDim` | int64/64 | 同上 | 元素 | `inputShape[W]` | 行跨度、GM offset | >0 |
| Small | `nOutDim` | int64/64 | 同上 | 折叠 N*C | `inputData.batches` | 最后 N tile 长度 | >0 |
| Small | `hOutDim` | int64/64 | 同上 | 元素 | `outShape[H]` | h tail/dst offset | >0 |
| Small | `wOutDim` | int64/64 | 同上 | 元素 | `outShape[W]` | w tail/dst offset | >0 |
| Small | `kW` | int64/64 | 同上 | 元素 | `kernelSize[W]` | gather/window loop | >0；Device 窄化为 uint16 |
| Small | `kH` | int64/64 | 同上 | 元素 | `kernelSize[H]` | gather/window loop | >0；Device 窄化为 uint16 |
| Small | `sW` | int64/64 | 同上 | 元素 | `stride[W]` | index/offset | >0；部分 helper 窄化 |
| Small | `sH` | int64/64 | 同上 | 元素 | `stride[H]` | index/offset | >0；部分 helper 窄化 |
| Small | `tPad` | int64/64 | 同上 | 元素 | `pad[top]` | R-002 h 有效交集 | R-001 为 0 |
| Small | `lPad` | int64/64 | 同上 | 元素 | `pad[left]` | R-002 w 有效交集 | R-001 为 0 |
| Small | `blockFactor` | int64/64 | 同上 | tile/core | `totalLoop/coreNum` | 每核 `[start,end)` | 可为 0 |
| Small | `blockTail` | int64/64 | 同上 | core 数 | `totalLoop-blockFactor*coreNum` | 前 tail 核多 1 tile | `[0,coreNum)` |
| Small | `ubFactorN` | int64/64 | 同上 | 折叠 batch/tile | UB 切分 | tile n 长度 | >=1 非空路径 |
| Small | `outUbFactorH` | int64/64 | 同上 | 输出行/tile | UB 二分/整批 | tile rows/gather | >=1 |
| Small | `outUbFactorW` | int64/64 | 同上 | 输出列/tile | UB 二分/整批 | tile cols/gather | >=1 |
| Small | `nLoop` | int64/64 | 同上 | loop | `ceil(Nf/ubFactorN)` | idx→nIdx | >=1 |
| Small | `hLoop` | int64/64 | 同上 | loop | `ceil(Ho/outUbFactorH)` | idx→hIdx | >=1 |
| Small | `wLoop` | int64/64 | 同上 | loop | `ceil(Wo/outUbFactorW)` | idx→wIdx | >=1 |
| Small | `inUbSize` | int64/64 | 同上 | T 元素/queue slot | `DoBlockTiling` | `InitBuffer(inputQue)`；R-002 `tmpBuf` | 已按 block 对齐列容量 |
| Small | `outUbSize` | int64/64 | 同上 | T 元素/queue slot | `DoBlockTiling` | `InitBuffer(maxUBOutput)` | block 对齐 |
| Small | `indiceUbSize` | int64/64 | 同上 | 字节 | `ceil(kH*kW/maxGatherElm)*V` | `InitBuffer(indexBuf)` | 已按 VReg 倍数 |
| Small | `gatherMode` | int64/64 | 同上 | enum | `CalcGatherMode` | `Process` 模板分派 | 0/1/2/3 |
| Small | `copyMode` | int64/64 | 同上 | enum | `CalcCopyMode` | R-002 copy/scatter | 0/1/2；R-001 仅影响稀疏搬运选择之外不读 |
| Small | `onceCopyRow` | int64/64 | 同上 | 行 | `DoBlockTiling` | R-002 多行 scatter | >=1；R-001 不读 |
| Small | `splitMode` | int64/64 | 同上 | enum | UB 切分 | CopyIn/offset | 1=cols,2=rows,3=batch |
| Big | `hInDim` | int64/64 | 宏生成，offset 不可见 | 元素 | `inputShape[H]` | 窗口裁剪/offset | >0 |
| Big | `wInDim` | int64/64 | 同上 | 元素 | `inputShape[W]` | 行跨度/offset | >0 |
| Big | `hOutDim` | int64/64 | 同上 | 元素 | `outShape[H]` | idx→ho | >0 |
| Big | `wOutDim` | int64/64 | 同上 | 元素 | `outShape[W]` | idx→wo | >0 |
| Big | `kW` | int64/64 | 同上 | 元素 | `kernelSize[W]` | 裁剪/分 W | >0 |
| Big | `kH` | int64/64 | 同上 | 元素 | `kernelSize[H]` | 裁剪/分 H | >0 |
| Big | `sW` | int64/64 | 同上 | 元素 | `stride[W]` | 起点公式 | >0 |
| Big | `sH` | int64/64 | 同上 | 元素 | `stride[H]` | 起点公式 | >0 |
| Big | `tPad` | int64/64 | 同上 | 元素 | `pad[top]` | 起点/单输出裁剪 | >=0 |
| Big | `bPad` | int64/64 | 同上 | 元素 | `pad[bottom]` | Device 不直接读取 | inactive consumer |
| Big | `lPad` | int64/64 | 同上 | 元素 | `pad[left]` | 起点/单输出裁剪 | >=0 |
| Big | `rPad` | int64/64 | 同上 | 元素 | `pad[right]` | Device 不直接读取 | inactive consumer |
| Big | `blockFactor` | int64/64 | 同上 | 输出/core | `totalIdx/coreNum` | `[beginIdx,endIdx)` | 可为 0 |
| Big | `blockTail` | int64/64 | 同上 | core 数 | `totalIdx%coreNum` | 前 tail 核多 1 输出 | `[0,coreNum)` |
| Big | `totalIdx` | int64/64 | 同上 | 输出元素 | `Nf*Ho*Wo` | body 不直接读取；支撑 coreNums | inactive consumer |
| Big | `coreNums` | int64/64 | 同上 | 核 | `min(totalIdx,coreNum)` | Device 不读；`SetBlockDim` 消费 | >0 |
| Big | `maxCount` | int64/64 | 同上 | T 元素/slot | UB 公式 | queue 容量、全载/分段 | VReg 字节对齐后折算 |
| Big | `isSigOut` | int64/64 | 同上 | bool | `Ho==1 && Wo==1` | 特化窗口裁剪 | 0/1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 300001 / priority 0 | `max_pool_v3_small_kernel_tiling.cpp::IsCapable/GetTilingKey/DoOpTiling` | `MaxPoolV3SmallKernelTilingData` | `max_pool_v3_apt.cpp::max_pool_v3` | `max_pool_v3_small_kernel.h::MaxPoolV3SmallKernel` + `max_pool_v3_common.h` helpers | `max_pool_v3_def.cpp`、`CMakeLists.txt`、`max_pool_v3_binary.json` |
| R-002 | 300002 / priority 0 | 同上，`InitializationVars` 判 padding | 同一 Small POD | 同一 router | `max_pool_v3_small_kernel_pad.h::MaxPoolV3SmallPadKernel` + common helpers | 同上 |
| R-003 | 311110 / priority 2 | `max_pool_v3_big_kernel_tiling.cpp::IsCapable/DoUBTiling` | `MaxPoolV3BigKernelTilingData` | 同一 router | `max_pool_v3_big_kernel.h::MaxPoolV3BigKernel` + common helpers | 同上 |

## Kernel 实现

### 公共执行流程

三条 Route 都由 `max_pool_v3(x,y,workspace,tiling)` 启动一次，entry 设置 `KERNEL_TYPE_AIV_ONLY`，按 key 解 POD、构造模板类、`Init` 后 `Process`。未知 key 没有 else/error，Kernel 直接结束且不写输出。所有 route 都把输出域静态分成连续编号，再由前 `blockTail` 个 AIV core 多处理一个编号。小窗按 tile 编号映射 `(nIdx,hIdx,wIdx)`；大窗按输出元素编号映射 `(nc,ho,wo)`。

### Route R-001：小窗无 padding

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_max_pool.cpp` | `ExecMaxPoolV3` | 950 regbase；4D ND→`NCHW`；pads 规范化 | `l0op::MaxPoolV3` |
| Host selector | `op_host/arch35/max_pool_v3_small_kernel_tiling.cpp` | `IsCapable` | NCHW && 输出数>=核数 && buffer capable | priority 0 accepted |
| Host key | 同上 | `InitializationVars/GetTilingKey` | 四侧 pad=0 且 ceil 末窗不越界 | 300001 |
| Router | `op_kernel/max_pool_v3_apt.cpp` | `max_pool_v3` | `TILING_KEY_IS(300001)` | `MaxPoolV3SmallKernel<DTYPE_X>` |

#### Device 数据流

##### 函数级调用链

```text
MaxPoolV3AiCore -> max_pool_v3(x,y,workspace,tiling)
  -> MaxPoolV3SmallKernel<T>::Init
  -> Process -> BaseCompute<M,U,GATHER_MODE>
    -> GenGatherIndex{SingleRow|MultiRow|MultiBatch|SingleKernel}
    -> [each tile] CopyInMultiRows[ Sparse]
    -> Compute{SingleRow|MultiRow|MultiBatch|SingleKernel}
       -> MaxPoolSplit{W|H|Batch} or MaxPoolSingleKernelCommon
    -> CopyMaxOut
```

##### 源码等价伪码

```text
# 伪码；idx 范围由 blockFactor/blockTail 给出
colsUB = splitCols ? align((tileW-1)*sW+kW, B) : W
index = gather_index(tileH,tileW,colsUB,sH,sW)
for idx in core_range:
  (nTile,hTile,wTile) = decode(idx,nLoop,hLoop,wLoop)
  n,rows,cols = clamp_last_tile(ubFactorN,tileH,tileW)
  src = nTile*ubFactorN*H*W + hTile*tileH*sH*W + wTile*tileW*sW
  dst = nTile*ubFactorN*Ho*Wo + hTile*tileH*Wo + wTile*tileW
  if sparse-H guard:
    DMA only kH rows per requested output row; logical sH becomes kH in UB
  else:
    DMA n * required input rows/cols
  gather every kH*kW window, max-reduce with exact tail mask
  DMA n*rows*cols valid outputs to y[dst]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | 上述 `src`；行跨度 W | tile 实际 n/inRows/inCols | T/NCHW folded NC | caller | CopyIn | caller；Kernel 全程 | splitCols 用 2D DMA stride，其余连续；sparse 用 loop mode |
| `inputQue_` | slot 0/1 | 每 slot `inUbSize*s` | T/紧凑或行对齐 | CopyIn | Compute | queue Alloc→EnQue→DeQue→Free | depth=2 |
| `indexBuf_` | 0 | `indiceUbSize` bytes | U=uint16(s<=2) else uint32 | GenGatherIndex | gather helper | Kernel 全程 | VReg 整数倍 |
| `maxUBOutput_` | slot 0/1 | 每 slot `outUbSize*s` | T/紧凑输出 tile | Compute | CopyMaxOut | queue 生命周期 | depth=2；只拷有效 n*rows*cols |
| y GM | `dst` | `n*rows*cols` | T/NCHW folded NC | CopyMaxOut | 后续框架 | caller/executor | tail 精确字节数 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ0["VECIN slot0 · inUbSize·s"]:1
  inQ1["VECIN slot1 · inUbSize·s"]:1
  idx["indexBuf · indiceUbSize B"]:1
  outQ0["VECOUT slot0 · outUbSize·s"]:1
  outQ1["VECOUT slot1 · outUbSize·s"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*inUbSize*s + indiceUbSize + 2*outUbSize*s
UB_total <= UB - 512   # Host 的保守预留；tile 容量已含 alignment
dynamic_ub = N/A
workspace_total(MaxPoolV3 node) = 16 MiB system workspace; body 不读
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input queue slot | CopyIn | Compute | `FreeTensor` 后下一 tile | EnQue/DeQue/FreeTensor | depth=2，禁止消费前覆盖 |
| output queue slot | Compute | CopyMaxOut | `FreeTensor` 后下一 tile | EnQue/DeQue/FreeTensor | 仅输出有效 tail |
| indexBuf | tile 循环前 GenGatherIndex | 每 tile Compute | Kernel 结束 | 写后进入后续 Vector scope | shape/stride 对所有该 core tile 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | TQue + `DataCopyPad` | CopyIn | DeQue in Compute | input slot | block/row stride 来自 tile；不把对齐尾部写出 |
| gather/max | `DataCopyGather` + `MicroAPI::Max` | input/index | output register | window lanes | 8-bit gather 先 unpack/pack；BF16 负无穷按位生成；mask 保护 W/H/N tail |
| reduce | `ReduceMaxAll` | max registers | output UB | 单输出/向量输出 | 8-bit 转 half reduce 后 `CAST_RINT` 回 8-bit；BF16 转 FP32 reduce 后回 BF16；其余原 dtype reduce。 |
| UB→GM | TQue + `DataCopyPad` | Compute | CopyMaxOut | output slot | `n*rows*cols` 精确长度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/max_pool_v3_small_kernel_tiling.cpp` | `DoUBTiling/DoBlockTiling/CalcGatherMode` | 优先整批→切 H→切 W，生成 tile/core/gather POD。 |
| Kernel entry/router | `op_kernel/max_pool_v3_apt.cpp` | `max_pool_v3`, key 300001 | POD 到 `MaxPoolV3SmallKernel<DTYPE_X>`。 |
| Device body/helper | `op_kernel/arch35/max_pool_v3_small_kernel.h` | `BaseCompute` 与四个 Compute 方法 | tile 地址、DMA、gather 模式、有效输出提交。 |
| Device body/helper | `op_kernel/arch35/max_pool_v3_common.h` | `MaxPoolImpl/MaxPoolSplit*/ReduceMaxAll` | kH×kW 最大值、dtype 特化与 mask tail。 |

### Route R-002：小窗 padding/ceil 尾窗

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_max_pool.cpp` | `padsOps/ExecMaxPoolV3` | pads→`[top,bottom,left,right]`，NCHW | MaxPoolV3 |
| Host selector | `op_host/arch35/max_pool_v3_small_kernel_tiling.cpp` | `InitializationVars` | 任一 pad 非 0，或 ceil 最后窗口越 H/W | `isPadding_=true` |
| Host key | 同上 | `GetTilingKey` | small capable && isPadding | 300002 |
| Router | `op_kernel/max_pool_v3_apt.cpp` | `max_pool_v3` | key 300002 | `MaxPoolV3SmallPadKernel<T>` |

#### Device 数据流

##### 函数级调用链

```text
max_pool_v3 -> MaxPoolV3SmallPadKernel<T>::Init/Process
  -> BaseCompute<M,U,GATHER_MODE>
    -> GenGatherIndex + optional GenScatterIndex
    -> [tile] compute valid h/w intersection -> CopyInMultiRows[Compact]
    -> Compute* -> CopyAndPad
       -> CustomDuplicate(-inf) -> CustomCopy / Scatter into tmpBuf
       -> MaxPoolSplit* / MaxPoolSingleKernelCommon
    -> CopyMaxOut
```

##### 源码等价伪码

```text
# 伪码
for tile in core_range:
  expected_origin = (hTile*tileH*sH, wTile*tileW*sW)
  valid_h = [max(expected_h-tPad,0), min(expected_h+(rows-1)*sH+kH-tPad,H))
  valid_w = [max(expected_w-lPad,0), min(expected_w+(cols-1)*sW+kW-lPad,W))
  DMA only valid rectangle from x into inputQue
  fill tmpBuf[n, expectedRows, alignCols] with dtype negative infinity/minimum
  scatter/copy valid rectangle to (rowStart,colStart) inside tmpBuf
  gather kH*kW on tmpBuf, max-reduce, enqueue output
  DMA only n*rows*cols to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `nTile*ubN*H*W+hLower*W+wLower` | n×realRows×realCols | T/NCHW | caller | CopyIn | caller | 只读有效交集，不读 padding GM |
| `inputQue_` | 双 slot | `2*inUbSize*s` | T/有效矩形 | CopyIn | CopyAndPad | queue | compact 或行对齐 |
| `tmpBuf_` | 0 | `inUbSize*s` | T/期望含 pad 画布 | CustomDuplicate+copy/scatter | max gather | Kernel 全程、每 tile 覆盖 | 先全写最小值，barrier 后写有效值 |
| `indexBuf_` | 0 | `indiceUbSize` | U | GenGatherIndex | max gather | Kernel 全程 | VReg 倍数 |
| `scatterIndexBuf_` | 0 | 256 bytes | U | GenScatterIndex | CopyAndPad | Kernel 全程 | copyMode=2 时不使用 |
| `maxUBOutput_` | 双 slot | `2*outUbSize*s` | T | Compute | CopyMaxOut | queue | 有效 tail |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["VECIN · depth=2 · 2·inUbSize·s"]:2
  tmp["tmpBuf · inUbSize·s"]:1
  idx["gather index · indiceUbSize B"]:1
  sidx["scatter index · 256 B"]:1
  outQ["VECOUT · depth=2 · 2·outUbSize·s"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*inUbSize*s + 2*outUbSize*s + indiceUbSize + 256
Host CalcBufferSize = 3*input_tile_elements + 2*output_tile_elements
Host additionally reserves 512 B, covering scatterIndexBuf(256 B) and margin
dynamic_ub = N/A
workspace_total(MaxPoolV3 node) = 16 MiB system workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf 全域 | `CustomDuplicate` | scatter/copy then max gather | 下一 tile | `LocalMemBar<VEC_STORE,VEC_STORE>` 后才写有效值 | 总画布长度不超过 inUbSize |
| input/output queue | CopyIn/Compute | CopyAndPad/CopyOut | FreeTensor 后 | TQue 协议 | 同 R-001 |
| scatter index | GenScatterIndex | CopyAndPad | Kernel 结束 | 生成后只读 | copyMode 与 index 公式匹配 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pad 初始化→有效值 | `CustomDuplicate` + LocalMemBar | tmpBuf 负无穷画布 | CustomCopy/scatter | tmpBuf | 浮点用 -inf，整数用最小值，无效 lane 不会胜出。 |
| 有效矩形搬运 | copy/scatter MicroAPI | input queue | max gather | tmpBuf 有效区域 | realRows/realCols 与 rowStart/colStart 精确裁剪四边。 |
| max/output | 同 R-001 | tmpBuf/index | output queue/GM | window/output | H/W/N 尾 tile 和 gather mask；dtype reduce 顺序同 R-001。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/max_pool_v3_small_kernel_tiling.cpp` | `InitializationVars/CalcBufferSize/CalcCopyMode` | padding key、额外 input 画布预算和 copy mode。 |
| Kernel entry/router | `op_kernel/max_pool_v3_apt.cpp` | key 300002 | 选择 Pad body。 |
| Device body/helper | `op_kernel/arch35/max_pool_v3_small_kernel_pad.h` | `BaseCompute/CopyAndPad/Compute*` | 有效交集、负无穷初始化、scatter/copy 与 max。 |
| Device body/helper | `op_kernel/arch35/max_pool_v3_common.h` | `CustomDuplicate/CustomCopyByScatter*/MaxPoolSplit*` | pad 物化及公共最大池化。 |

### Route R-003：大窗逐输出 fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host registry | `common/inc/op_host/tiling_templates_registry.h` | `DoTilingImpl` | priority 0 返回 param-invalid 后升序继续 | priority 2 |
| Host selector | `op_host/arch35/max_pool_v3_big_kernel_tiling.cpp` | `IsCapable` | `inputFormat==NCHW` | accepted |
| Host key | 同上 | `GetTilingKey` | 固定 | 311110 |
| Router | `op_kernel/max_pool_v3_apt.cpp` | `max_pool_v3` | key 311110 | `MaxPoolV3BigKernel<T>` |

#### Device 数据流

##### 函数级调用链

```text
max_pool_v3 -> MaxPoolV3BigKernel<T>::Init/Process
  -> BaseCompute<SPLIT_KERNEL>
    -> [each output] CalcKernelSize
    -> InitOutLocal<CLEAR>
    -> NoSplitKernelProcess -> CopyInMultiRows -> ComputeSingle<MERGE=false>
       or SplitKernelProcess -> CopyInMultiRows/CopyInSingleRow -> ComputeSingle<MERGE=true>
    -> CopyMaxOut (每 1024 个或 core 尾批提交)
```

##### 源码等价伪码

```text
# 伪码
for outIdx in core_contiguous_range:
  (nc,ho,wo) = decode(outIdx,Ho,Wo)
  origin = (ho*sH-tPad, wo*sW-lPad)
  clip origin and (curkH,curkW) to input H/W
  if curkH*curkW <= maxCount:
    DMA curkH rows × curkW compact; max all; store one result in output TBuf
  else:
    initialize output TBuf chunk to dtype minimum
    if curkW <= maxCount: split H into floor(maxCount/curkW) rows per segment
    else: for every H row, split W into maxCount elements
    reduce each segment, merge with prior scalar in output TBuf
  when local index=1023 or core tail: V->MTE3 event, DMA accumulated results to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `nc*H*W+originH*W+originW` | clipped curkH×curkW | T/NCHW folded NC | caller | CopyIn | caller | multi-row compact；超宽逐行分段 |
| `inputQue_` | slot 0/1 | 每 slot `maxCount*s` | T/compact window segment | CopyIn | ComputeSingle | queue | depth=2；最后 segment 精确 dataCount |
| `maxUBOutput_` | circular local index `(outIdx-begin)%1024` | `1024*s` | T/scalar results | Init/Compute merge | CopyMaxOut | core 全程 | 1024 输出一批；分段时原地 scalar merge |
| y GM | `outIdx-localCurIdx` | `(localCurIdx+1)*s` | T/contiguous output | CopyMaxOut | 后续框架 | caller | 每批实际长度 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in0["VECIN slot0 · maxCount·s"]:1
  in1["VECIN slot1 · maxCount·s"]:1
  out["maxUBOutput · 1024·s"]:1
```

##### 字节账与复用顺序

```text
maxCount = floor_align((UB - 1024*s)/2, V) / s
UB_total = 2*maxCount*s + 1024*s <= UB
dynamic_ub = N/A
workspace_total(MaxPoolV3 node) = 16 MiB system workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input slot | CopyIn | ComputeSingle | FreeTensor 后 | TQue | 分段 dataCount<=maxCount |
| output scalar | InitOutLocal/ComputeSingle | MergeMaxRes / CopyMaxOut | 对应 1024 批完成后 | VEC store/load barriers；V_MTE3 event | merge 前已有负无穷或前段结果 |
| 1024 输出批 | ComputeSingle | GM DMA | 下一个 local index 0 前 | `MTE3_V` 防止覆写未完成批；`V_MTE3` 防止 DMA 读未完成 | core 内连续输出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输出批复用 | `MTE3_V` flag | 前一 GM copy | 新批 InitOutLocal | 1024 元素 TBuf | local index 0 才等待。 |
| 计算→GM | `V_MTE3` flag | Vector 写 max TBuf | DataCopyPad | 实际批结果 | local index 1023 或 core 最后输出触发。 |
| 分段 merge | VEC store/load barriers | 前段 scalar | `MergeMaxRes` | 单输出 scalar | 每段 ReduceMax 后与前值 Max；最终只写一次 GM。 |
| reduce/tail | `DuplicateNegInf` + UpdateMask | input segment | ReduceMaxAll | VReg 尾 lane | 补齐 VReg 的 lane 写负无穷；8-bit/BF16 cast 顺序同 R-001。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/max_pool_v3_big_kernel_tiling.cpp` | `DoUBTiling/SetTilingData` | 连续输出分核与 maxCount 公式。 |
| Kernel entry/router | `op_kernel/max_pool_v3_apt.cpp` | key 311110 | Big body 实例。 |
| Device body/helper | `op_kernel/arch35/max_pool_v3_big_kernel.h` | `CalcKernelSize/SplitKernelProcess/ComputeSingle/CopyMaxOut` | 边界裁剪、H/W 分段、原地 merge、事件同步。 |
| Device body/helper | `op_kernel/arch35/max_pool_v3_common.h` | `DuplicateNegInf/ReduceMaxAll/MergeMaxRes` | tail、dtype 特化与跨段最大值。 |

## 算子特定优化

### Trick T-001：小窗模板按输出并行度自适应收缩 UB tile

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：初始 `nLoop*hLoop*wLoop < coreNum` 且非空；每轮仍有 `availableUb_ > 4096/dtypeSize`。
- Golden 来源：[`pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.cpp:278-287`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.cpp#L278-L287)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t ubStep = BLOCK_SPLIT_THREHOLD / dtypeSize;
do {
    DoUBTilingSingle();
    if (nLoop_ * hLoop_ * wLoop_ >= coreNum || isZero_) {
        break;
    }
    availableUb_ -= ubStep;
} while (availableUb_ > ubStep);
```

#### 机制与设计原因

Host 不固定追求最大 tile，而是每轮减少 4096 字节等价的元素预算并重算整批/切 H/切 W。较小 tile 增加 `nLoop*hLoop*wLoop`，直到可覆盖全部 AIV core；这用更多 tile 换取可证明的更高核级并行度。每轮仍通过 `CalcBufferSize` 约束双缓冲与 padding 额外画布，因此不会越过 UB 上界。

#### 迁移边界

- 适用：输出数足够且存在多个合法 UB tile 的逐 tile 独立算子。
- 前提：950、NCHW、小窗模板 capable；tile 之间无跨 tile reduction；缩小后仍满足最小并发/索引上界。
- 失效：最小 tile 仍不满足 UB 或输出数小于核数时使用 R-003。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码只证明该循环保留 buffer-capable 约束后提高 tile 数。

### Trick T-002：padding 只搬有效 GM，并在 UB 物化负无穷画布

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：key 300002；每 tile 先计算 `hLower/hUpper/wLower/wUpper`。
- Golden 来源：[`pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_small_kernel_pad.h:158-178`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_small_kernel_pad.h#L158-L178)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t hUpper =
    min(hIdx * sH * outUbFactorH + (rows - 1) * sH + tilingData_->kH - tilingData_->tPad, tilingData_->hInDim);
int64_t hLower = max(hIdx * sH * outUbFactorH - tilingData_->tPad, (int64_t)0);
int64_t wUpper =
    min(wIdx * sW * outUbFactorW + (cols - 1) * sW + tilingData_->kW - tilingData_->lPad, tilingData_->wInDim);
int64_t wLower = max(wIdx * sW * outUbFactorW - tilingData_->lPad, (int64_t)0);
int64_t realRows = hUpper - hLower;
int64_t realCols = tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : wUpper - wLower;
```

#### 机制与设计原因

Device 从 GM 只取窗口并集与真实输入的交集，再把它复制/散射到预先填为负无穷或整数最小值的 `tmpBuf`。这样无需在 GM 中读取不存在的 padding，也使公共 gather/max helper 可在规则画布上运行。`copyMode` 根据 W 与寄存器覆盖能力选择多行 scatter、单行 scatter 或普通 copy，减少无必要的逐行控制。

#### 迁移边界

- 适用：max/min 等存在天然幺元、且 padding 可在 UB 中廉价生成的滑窗算子。
- 前提：幺元必须严格不改变有效 reduction；UB 能容纳额外 input 大小的 `tmpBuf`；有效交集与目标 offset 可闭合。
- 失效：额外画布使 small buffer incapable 时回退 R-003 的逐输出裁剪，不物化整 tile padding。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码明确以 `GetNegInf<T>`/BF16 位模式初始化，不能任意用 0 替代有符号或浮点 max 的 padding 值。

### Trick T-003：大窗分段 reduction 原地合并并批量回写

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：`kH*kW > maxCount`；每个窗口沿 H 或 W 分段。
- Golden 来源：[`pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_big_kernel.h:239-269`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_big_kernel.h#L239-L269)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (curkW <= maxCount) {
    // 整行搬入
    int64_t hFactor = maxCount / curkW;
    int64_t hLoops = (curkH + hFactor - 1) / hFactor;
    int64_t hTail = curkH - (hLoops - 1) * hFactor;
```

#### 机制与设计原因

当整个窗口不能放入一个 queue slot 时，若一整行可放下则一次搬多行，否则逐行切 W。每段先在寄存器中 ReduceMax，再通过 `MergeMaxRes` 与同一 `maxUBOutput_` scalar 原地取 Max；中间段不落 GM。最多 1024 个最终 scalar 聚合后一次回写，减少小粒度 GM 输出提交。

#### 迁移边界

- 适用：可结合、可分段的标量 reduction，且单段不超过 UB slot。
- 前提：幺元、merge 次序和 dtype reduce 语义等价；输出 TBuf 的事件协议保持不变。
- 失效：窗口全载时使用同一 R-003 内的 `NoSplitKernelProcess`，不做初始化与 merge。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码表明若删除 VEC store/load barrier 或 V↔MTE3 event，会破坏同一输出 scalar 或 1024 元素批的生命周期协议。

## 模板与实现对比

| 对比维度 | R-001 小窗无 pad | R-002 小窗 pad | R-003 大窗 fallback | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | p0 small capable、isPadding=false | p0 small capable、isPadding=true | p2 NCHW fallback | 先让一个 tile 计算多个输出；资源/并行不足改逐输出。 |
| 模板与 dtype | Small<T>, U 按 s 选 16/32 位 | SmallPad<T>, 同 U | Big<T> | dtype 均由 DTYPE_X 实例化。 |
| 循环与分核 | n/h/w tile | n/h/w tile + 有效交集 | 连续输出编号 + 窗口段 | 输出覆盖域不同。 |
| Queue/TBuf | 2×in + index + 2×out | R-001 + tmp input + 256B scatter index | 2×segment input + 1024 scalar TBuf | padding 需要规则画布，大窗只保留单输出状态。 |
| 内存总占用与复用 | `2in+2out+index` | `3in+2out+index+256` | `2maxCount+1024` 元素 | Host 公式与 InitBuffer 一致。 |
| 搬运/Vector API | 2D/loop DMA + gather/max | 有效 GM DMA + duplicate/scatter + gather/max | compact DMA + ReduceMax + scalar merge | 根据窗口规则性与容量选路径。 |
| 同步、精度与 tail | TQue；mask tail | TQue + LocalMemBar；负无穷 pad | TQue + V/MTE events + merge barriers | 同 dtype reduce helper，生命周期不同。 |
| fallback/失败边界 | small incapable→R-003 | small incapable→R-003 | NCHW 恒 capable；未知 key 不写 | Host selector 必须生成已注册 key。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档标为支持，但固定仓无公开 Host/Tiling/Kernel 闭包，未发布卡 | 同 A2 | 公开支持且 source_closed |
| Host/Launcher | API dtype 表含 DAV_2201，但 `max_pool_v3` 活动构建未接 A2 | 同左；固定仓未见 A3 专属接线 | DAV_3510 regbase；直接 ND/NCHW，不经 5HD TransData |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 三条 NCHW AIV route；CMake/config 接 ascend950 |

## 限制与失败边界

- 本卡只覆盖 `aclnnMaxPool` 的二维 MaxPool 语义，不覆盖 WithIndices、WithMask、MaxPool3D、Grad、原地或量化变体。
- ACLNN 校验允许两侧 pad 之和 `<=kernel`，而 MaxPoolV3 Host 对每一侧要求 `0<=pad<kernel`；若某一侧等于 kernel，API shape 检查可能通过但 Tiling 失败。
- Small body 将 kH/kW/stride/tile loop 多处窄化到 uint16；Host 用 65535 元素 tile 上界约束索引域，但固定仓未见对所有单字段窄化的独立显式上限检查，迁移时不得扩大范围。
- `TILING_DATA_FIELD_DEF` 的实际 `sizeof/alignof/offsetof` 由仓外运行时宏生成，固定仓不可见；字段声明顺序、类型、producer/consumer 已闭合。
- unknown TilingKey 在 router 中静默结束且不写输出；正确性依赖 Host selector 与注册 POD 的一致性。
- MaxPoolV3 节点固定申请 16 MiB system workspace，body 不读取；完整 ACLNN workspace 还包括框架节点，由 executor 汇总，固定仓未暴露其内部切片公式。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `pooling/max_pool_v3/docs/aclnnMaxPool.md` | `aclnnMaxPool*` 文档 | 平台、dtype、shape、属性与错误语义。 |
| 入口声明/约束 | `pooling/max_pool_v3/op_api/aclnn_max_pool.h` | 两段式函数声明 | C ABI。 |
| 入口实现 | `pooling/max_pool_v3/op_api/aclnn_max_pool.cpp` | `CheckParams/ExecMaxPoolV3/aclnnMaxPool*` | 校验、前后处理、950 分支与 executor。 |
| 入口实现 | `pooling/max_pool_v3/op_api/max_pool_v3.cpp` | `l0op::MaxPoolV3/MaxPoolV3AiCore` | dtype 分派、infer 与 launcher。 |
| Host 定义 | `pooling/max_pool_v3/op_host/max_pool_v3_def.cpp` | `ops::MaxPoolV3` | ascend950 AICore 注册。 |
| Host/Tiling | `pooling/max_pool_v3/op_host/arch35/max_pool_v3_tiling.cpp` | `Tiling4MaxPoolV3/TilingPrepare4MaxPoolV3` | 模板 registry、AIV core/UB producer。 |
| Host/Tiling | `pooling/max_pool_v3/op_host/arch35/max_pool_v3_tiling_base.cpp` | `MaxPoolV3BaseTiling::*` | shape/attrs/pad/workspace 公共逻辑。 |
| Host/Tiling | `pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.h` | `MaxPoolV3SmallKernelTilingData` | Small POD 定义/注册。 |
| Host/Tiling | `pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.cpp` | `MaxPoolV3SmallKernelTiling::*` | p0 selector、UB/core/gather/copy 参数。 |
| Host/Tiling | `pooling/max_pool_v3/op_host/arch35/max_pool_v3_big_kernel_tiling.h` | `MaxPoolV3BigKernelTilingData` | Big POD 定义/注册。 |
| Host/Tiling | `pooling/max_pool_v3/op_host/arch35/max_pool_v3_big_kernel_tiling.cpp` | `MaxPoolV3BigKernelTiling::*` | p2 fallback 与 maxCount。 |
| Kernel entry/body | `pooling/max_pool_v3/op_kernel/max_pool_v3_apt.cpp` | `max_pool_v3` | AIV-only router 六 key；ACLNN 可达三 key。 |
| Kernel entry/body | `pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_small_kernel.h` | `MaxPoolV3SmallKernel` | 无 pad small body。 |
| Kernel entry/body | `pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_small_kernel_pad.h` | `MaxPoolV3SmallPadKernel` | padding small body。 |
| Kernel entry/body | `pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_big_kernel.h` | `MaxPoolV3BigKernel` | 大窗逐输出 body。 |
| Kernel helper | `pooling/max_pool_v3/op_kernel/arch35/max_pool_v3_common.h` | gather/scatter/max/reduce helpers | dtype、索引、tail、精度与同步。 |
| Host selector framework | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority 升序与 fallthrough 语义。 |
| Host selector framework | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | capability false→`GRAPH_PARAM_INVALID` 与 Tiling 阶段顺序。 |
| ACLNN 平台 helper | `common/inc/op_api/aclnn_util.h` | `AclnnUtil::IsRegbase` | 只有 DAV_3510 被归为 regbase。 |
| 构建接线 | `pooling/max_pool_v3/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35。 |
| 构建接线 | `pooling/max_pool_v3/op_host/config/ascend950/max_pool_v3_binary.json` | `op_type=MaxPoolV3` | ascend950 dtype 二进制实例。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Small/Big 字段均 int64 且 producer/consumer 对应 | 字段顺序、key→POD 类型、元素/字节单位 | 两个 `BEGIN_TILING_DATA_DEF` 与 router |
| 调用顺序 | ACLNN 校验→Contiguous→MaxPoolV3→ViewCopy；Kernel Init→Process | 同 stream producer→consumer 与失败阻断 | `aclnnMaxPoolGetWorkspaceSize/CommonOpExecutorRun` |
| workspace/内存 | 16 MiB system workspace；Small/Pad/Big 字节账 | depth=2、512B 预留、256B scatter index、1024 输出槽 | `GetWorkspaceSize/InitBuffer` |
| 同步 | Small 使用 TQue；Pad 增加 LocalMemBar；Big 增加 V↔MTE3 flags | queue FreeTensor 前不覆盖，GM DMA 前后事件不可删 | 三个 body 的 queue/event 调用 |
| 精度与尾块 | dtype 最小值 padding；8-bit/BF16 特化 reduce；精确 mask | cast 顺序、BF16 位模式、tail mask、只写有效输出 | `GetNegInf/ReduceMaxAll/UpdateMask` |
| 失败边界 | Small 不 capable 落 Big；空 tensor不启动；未知 key 不写 | priority 顺序和注册 key 必须一致 | registry、`IsCapable`、router |
