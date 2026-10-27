---
type: Operator
title: "AntiMxQuant 950 ACLNN 算子知识"
description: "说明 aclnnAntiMxQuant 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/docs/aclnnAntiMxQuant.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_api/aclnn_anti_mx_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_api/anti_mx_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_host/anti_mx_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_host/arch35/anti_mx_quant_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/anti_mx_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_host/config/ascend950/anti_mx_quant_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:21:21Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# AntiMxQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAntiMxQuantGetWorkspaceSize` + `aclnnAntiMxQuant`，公开 ACLNN、无版本后缀；固定仓中 950 仅此公开版本 |
| 计算语义 | 对每个 32 元素量化块执行 `y = x * scale`；`mxscale` 的 `FLOAT8_E8M0` 值表示公式中的 `2^(sf-127)` |
| 输入输出 | `x`: 1–7D ND，FP4 E2M1/E1M2 或 FP8 E4M3FN/E5M2；`mxscale`: 2–8D ND、FP8 E8M0；`y`: 与 `x` 同 shape，FP16/BF16/FP32 |
| 执行类型 | 单节点 `AIV_VECTOR`；entry 显式声明 `KERNEL_TYPE_AIV_ONLY`，可达图中无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001/R-002/R-003：FP8→FP32/BF16/FP16；R-004/R-005/R-006：FP4→FP32/BF16/FP16 |
| 核心资源 | GM `x/mxScale/y`；双缓冲 `VECIN` x/scale 与 `VECOUT` y；单 `VECCALC` scale buffer；MicroAPI `RegTensor`；workspace 为 0 |
| 来源闭包 | `source_closed`：公开 ACLNN→连续化/launcher→950 Tiling/POD→单 AIV entry→`AntiMxQuantTailAxis` body→Ascend 950 binary/CMake 接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `R=rank(x)`，实现只接受归一化后 `axis=R-1`。令 `C=x.shape[-1]`、`row=product(x.shape[0:R-1])`、`b=floor(c/32)`，则逻辑输出为：

```text
y[r, c] = cast_U(cast_compute(x[r, c]) * cast_compute(mxscale[r, b]))
```

公开文档将 `mxscale` 解释成 FP8 E8M0 缩放指数，等价写作 `x * 2^(sf-127)`。存储 shape 把每两个 32 元素块的 scale 放在末尾长度 2 的维度，因此末轴前一维为 `ceil(ceil(C/32)/2)`，展平后的每行 scale 字节数为 `align_up(ceil(C/32), 2)`。FP8 路径把数据与 scale 升到 FP32 相乘；FP4 路径把数据与 scale 转 BF16 后相乘，再按 `dstType` 输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 1–7D / ND | FP4 E2M1、FP4 E1M2、FP8 E5M2、FP8 E4M3FN | 允许空 Tensor；FP4 最后一维必须为偶数且 ACLNN 层拒绝非连续 FP4；FP8 非连续时由 `Contiguous` 处理 |
| `mxscale` | `rank(x)+1`，2–8D / ND | FP8 E8M0 | `shape[axis]=ceil(ceil(x.shape[axis]/32)/2)`，末维为 2，其余维与 `x` 相同；允许空 Tensor；非连续时连续化 |
| `axis` | 标量 | `int64_t` | 文档给出 `[-R,R-1]`，但固定实现 Tiling 进一步只接受尾轴，即 `axis==-1` 或 `axis==R-1` 等价值 |
| `dstType` | 标量 | `int64_t` | `{0,1,27}` 分别对应 FP32、FP16、BF16；L0/OpDef 的默认值为 27，但 ACLNN 原型要求显式传入 |
| `y` | 与 `x` 同 shape / ND | FP16、BF16、FP32 | dtype 必须与 `dstType` 数值完全一致；`ViewCopy` 负责写入用户视图 |

### 入口与版本

公开入口是两段式 `aclnnAntiMxQuantGetWorkspaceSize`/`aclnnAntiMxQuant`。固定 Commit 未出现 `V2` 或其它同语义公开版本，因此这就是 950 实际支持的最新公开版本。第一段构造 executor，加入可能的 `Contiguous`、一个 `AntiMxQuant` AICore launcher 和 `ViewCopy`；第二段以用户 stream 调用 `CommonOpExecutorRun`。公开产品表仅 Ascend 950PR/950DT 为支持，A2/A3 明确为不支持；OpDef、`SUPPORT_COMPUTE_UNIT` 与 `ascend950` binary config 相互印证。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnAntiMxQuantGetWorkspaceSize` 的顺序为：创建 executor；检查 `x/mxscale/y` 非空；检查三者 dtype 及 `y.dtype==dstType`；检查 `dstType∈{0,1,27}`；检查 `rank(x)==rank(y)`、`rank(x)<=7`、`2<=rank(mxscale)<=8`、`rank(mxscale)=rank(x)+1` 与 axis 范围。任一输入 Tensor 为空时返回 `workspaceSize=0`，executor 中没有本算子 Device 节点。非空时先拒绝非连续 FP4 `x`，再对 `x` 和 `mxscale` 调用框架 `Contiguous`，由 `l0op::AntiMxQuant` 分配与 `x` 同 shape、dtype 为 `dstType` 的连续中间输出并加入 AICore launcher，最后用框架 `ViewCopy` 写用户 `y`。

API 层只检查 rank 与 axis 范围；完整 shape 关系、FP4 末维偶数和尾轴限制由 `Tiling4AntiMxQuant::CheckShape` 再检查。Tiling 依次执行 `CheckDtype`、`GetAttr`、`CheckShape`、`GetPlatInfo`，任一步失败即 `GRAPH_FAILED`，不启动 Kernel。

### 参数、Workspace 与生命周期

`GetPlatInfo` 从 950 platform 取得 `GetCoreNumAiv()`、UB 字节数和 VReg 字节数。Tiling 将非末维折叠为 `rowNum`，末维为 `colNum`，按固定 `1×512` 外块划分二维核网格。`SetTilingParams` 把 13 个 `int64_t` 字段复制到 raw tiling buffer，`SetBlockDim(usedCoreNum)`，并令唯一算子 workspace slice 为 0。

ACLNN 第一段报告的是整个 executor 的 workspace；算子自身 Tiling workspace 为 0，但框架 `Contiguous/ViewCopy` 是否需要空间由 executor 汇总，固定算子源码没有把它错误地当成始终为 0。executor 从 `uniqueExecutor` 转交调用者，第二段运行期间持有中间 Tensor 和 launcher；stream ordering 由 `CommonOpExecutorRun` 的框架边界保证。算子单 Kernel 无跨 Kernel workspace 边。

### 启动图与失败传播

```text
aclnnAntiMxQuantGetWorkspaceSize
  -> [可选] l0op::Contiguous(x)
  -> [可选] l0op::Contiguous(mxscale)
  -> l0op::AntiMxQuant
       -> ADD_TO_LAUNCHER_LIST_AICORE(AntiMxQuant, x, mxscale -> out, axis, dstType)
       -> Tiling4AntiMxQuant / AntiMxQuantTailAxisTiling::DoTiling
       -> anti_mx_quant<axisMode=TPL_AXIS_TAIL>
  -> l0op::ViewCopy(out, y)
aclnnAntiMxQuant(workspace, workspaceSize, executor, stream)
  -> CommonOpExecutorRun
```

唯一算子节点的 entry 参数顺序为 `x, mxScale, y, workspace, tiling`，`blockDim=usedCoreNum=rowTileNum*colTileNum`，POD 为 `AntiMxQuantTilingData`。输入输出直接位于 executor 管理的 GM Tensor；workspace 参数被 entry 显式忽略。Tiling、launcher 添加或任一框架节点失败都会使第一段返回错误/空指针并阻断后续节点；第二段把 executor 的运行状态直接返回。固定仓将 `Contiguous/ViewCopy/CommonOpExecutorRun` 作为框架边界，没有算子拥有的跨节点 event 或额外 stream。

## Route 与参数设计

### 分核与资源计算

令 `M=product(x.shape[:-1])`、`C=x.shape[-1]`：

```text
rowBlockLoopNum = M
colBlockLoopNum = ceil_div(C, 512)
initialUsed = max(1, min(totalAiv, M * ceil_div(C, 512)))
```

`AutoTiling` 枚举 `initialUsed` 的候选行切分 `m` 及 `n=floor(initialUsed/m)`，要求 `m<=M`、`n<=ceil_div(C,512)`；用尾块不均衡量 `delta` 排序，`delta` 小者优先，同 `delta` 时较小的列切分 `n` 优先。随后重新计算：

```text
rowNormalBlockNum = ceil_div(M, rowTileNum)
colNormalBlockNum = ceil_div(ceil_div(C,512), colTileNum)
rowTileNum = ceil_div(M, rowNormalBlockNum)
colTileNum = ceil_div(ceil_div(C,512), colNormalBlockNum)
usedCoreNum = rowTileNum * colTileNum
rowTailLen = M - rowNormalBlockNum * (rowTileNum - 1)
colTailLen = C - colNormalBlockNum * 512 * (colTileNum - 1)
```

Host 以每 512 元素粗粒度计算 UB：FP8/FP4 输入分别 512/256 B，FP16/BF16 输出 1024 B，FP32 输出 2048 B，scale 为 16 B；三路 Queue 乘双缓冲，FP8/FP4 scale compute 额外按 64/32 B 计。因此六类 `bytesPer512` 分别为 5216、3168、3168、4640、2624、2624（按 R-001 至 R-006），`K=floor(ubSize/bytesPer512)`，POD `maxUbBlockNum=16*K`，单位为 32 元素块。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP8→FP32 | `GET_TPL_TILING_KEY(TPL_AXIS_TAIL)`；`axisMode=0,T=fp8_e4m3fn_t/fp8_e5m2_t,U=float` | binary dtype 为 FP8 E4M3FN/E5M2 + FP32，且尾轴/shape/Tiling 均通过 | dtype binary 不匹配时查下一行；非法组合失败 |
| 2 | R-002 FP8→BF16 | 同一 tail key；`axisMode=0,T=fp8_e4m3fn_t/fp8_e5m2_t,U=bfloat16_t` | binary dtype 为 FP8 E4M3FN/E5M2 + BF16 | 同上 |
| 3 | R-003 FP8→FP16 | 同一 tail key；`axisMode=0,T=fp8_e4m3fn_t/fp8_e5m2_t,U=half` | binary dtype 为 FP8 E4M3FN/E5M2 + FP16 | 同上 |
| 4 | R-004 FP4→FP32 | 同一 tail key；`axisMode=0,T=fp4x2_e2m1_t/fp4x2_e1m2_t,U=float` | binary dtype 为 FP4 E2M1/E1M2 + FP32，末维为偶数 | 同上 |
| 5 | R-005 FP4→BF16 | 同一 tail key；`axisMode=0,T=fp4x2_e2m1_t/fp4x2_e1m2_t,U=bfloat16_t` | binary dtype 为 FP4 E2M1/E1M2 + BF16，末维为偶数 | 同上 |
| 6 | R-006 FP4→FP16 | 同一 tail key；`axisMode=0,T=fp4x2_e2m1_t/fp4x2_e1m2_t,U=half` | binary dtype 为 FP4 E2M1/E1M2 + FP16，末维为偶数 | 无其它 Route，失败 |

```text
empty x or mxscale -> ACLNN success, no Device launch
non-empty -> validate dtype/shape/axis
  -> binary config selects one of 4 x dtypes × 3 y dtypes
  -> CalcTilingKey always uses GET_TPL_TILING_KEY(TPL_AXIS_TAIL), with axisMode=0
  -> anti_mx_quant<0>
  -> compile-time IsFp8Type/IsFp4Type and U type select R-001...R-006
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | tail key / two FP8→FP32 binaries | 950；合法尾轴；FP8 E4M3FN/E5M2→FP32 | 单 `AIV_VECTOR` | `AntiMxQuantTailAxis<T,float>`；FP32 scale/乘法/输出 | x/scale/y 双 Queue，FP32 scale TBuf；Queue 同步 | DataCopyPad + 偶数 scale block；无 fallback |
| R-002 | tail key / two FP8→BF16 binaries | 同上，输出 BF16 | 单 `AIV_VECTOR` | FP32 乘法后 RINT/SAT cast BF16 | 同 R-001，y 字节减半 | 同上 |
| R-003 | tail key / two FP8→FP16 binaries | 同上，输出 FP16 | 单 `AIV_VECTOR` | FP32 乘法后 RINT/SAT cast FP16 | 同 R-002 | 同上 |
| R-004 | tail key / two FP4→FP32 binaries | 950；合法尾轴；FP4 E2M1/E1M2→FP32；`C%2==0` | 单 `AIV_VECTOR` | `AntiMxQuantTailAxis<T,float>`；BF16 scale/乘法，后升 FP32 | FP4 packed x Queue、BF16 scale TBuf | packed tail pad + CopyOut stride；无 fallback |
| R-005 | tail key / two FP4→BF16 binaries | 同上，输出 BF16 | 单 `AIV_VECTOR` | BF16 scale/乘法，直接 BF16 store | 同 R-004，y 字节减半 | 同上 |
| R-006 | tail key / two FP4→FP16 binaries | 同上，输出 FP16 | 单 `AIV_VECTOR` | BF16 乘法后 RINT/SAT cast FP16 | 同 R-005 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 仅尾轴 | normalized `axis == rank(x)-1` | body 把非末维折成 row，并用连续 32 元素块 scale | 全部 | `GRAPH_FAILED`，无 non-tail fallback | `anti_mx_quant_tiling_arch35.cpp::CheckShape` |
| FP4 packed 对齐 | `C%2==0` | 每字节含两个 FP4，GM offset/stride 除 2 | R-004..006 | `GRAPH_FAILED` | `CheckShape`、`CopyIn` |
| scale 双数 | `scaleColNum=align_up(ceil(C/32),2)`；每个 UB chunk 的 block 数也向偶数对齐 | scale storage 最后一维为 2；MicroAPI 每次按配对布局展开 | 全部 | shape 不符失败；chunk tail 用合法配对 scale/padding | `CheckShape`、`GetGmParams`、`GetUbParams` |
| 512 元素 VF 粒度 | `loopNum2VF=ceil(totalBlockNum/16)` | FP8/FP4 `ComputeData` 每轮消费 16 个 32 元素块 | 全部 | 最后一轮由输入 pad 与 CopyOut 有效长度保护 | `Compute`、两组 `ComputeData` |
| UB 模型 | `K=floor(ubSize/bytesPer512)`，`maxUbBlockNum=16K` | 控制双 Queue 和 scale TBuf 容量 | 全部 | 固定源码没有 `K>0` 或实际向上对齐后总字节的二次检查 | `CalcBytesPerBlock`、`DoTiling`、`Init` |
| ABI 窄化 | `rowTileNum/colTileNum` 先 cast `uint16_t`，POD 再存 `int64_t`；VF loops cast `uint16_t` | split 和 MicroAPI loop 计数位宽有限 | 全部 | 固定源码无显式溢出 fallback | `AutoTiling`、`ComputeScale`、`ComputeData` |

### POD/ABI

`AntiMxQuantTilingData` 由 13 个依次声明的 `int64_t` 字段组成，Host 与 Device 直接包含同一头文件，Host 用该类型的 `sizeof` 整体复制。目标编译器最终生成的 `sizeof/alignof/offsetof` 数值在固定仓中不可见，因此下表只记录可证明的声明序号，不猜测数值 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AntiMxQuantTilingData` | `ubSize` | `int64_t`/64 | 声明序号 0；数值 offset 固定仓不可见 | 字节 | `SetTilingParams: tilingParam_.ubSize` | body 不读取 | inactive 字段，仍序列化 |
| 同上 | `dstType` | `int64_t`/64 | 声明序号 1；同上 | GE dtype 枚举值 | `GetAttr` | `ParseTilingData` 保存但计算用模板 `U` | Device 不据此分支 |
| 同上 | `totalCoreNum` | `int64_t`/64 | 声明序号 2；同上 | AIV core | `GetCoreNumAiv` | `ParseTilingData` 保存但不再使用 | inactive |
| 同上 | `usedCoreNum` | `int64_t`/64 | 声明序号 3；同上 | core | `rowTileNum*colTileNum` | `Process` 越界保护 | 同时作为 `blockDim` |
| 同上 | `rowTileNum` | `int64_t`/64 | 声明序号 4；同上 | core 网格行数 | `AutoTiling/DoTiling` | `GetGmParams/GetUbParams` | 必须大于 0 |
| 同上 | `colTileNum` | `int64_t`/64 | 声明序号 5；同上 | core 网格列数 | `AutoTiling/DoTiling` | `coreIdx%colTileNum` 与尾列判断 | 必须大于 0 |
| 同上 | `rowNum` | `int64_t`/64 | 声明序号 6；同上 | 行 | `CalcAxisSize` | 仅保存，地址用分块字段 | inactive |
| 同上 | `colNum` | `int64_t`/64 | 声明序号 7；同上 | 元素/行 | `CalcAxisSize` | GM offset/stride 与 scale row 宽 | 非空 Device route 应大于 0 |
| 同上 | `colNormalBlockNum` | `int64_t`/64 | 声明序号 8；同上 | 512 元素块/core | `DoTiling` | `GetGmParams/GetUbParams` | 非尾列使用 |
| 同上 | `colTailLen` | `int64_t`/64 | 声明序号 9；同上 | 元素 | `DoTiling` | 尾列有效长度 | 最后列使用 |
| 同上 | `rowNormalBlockNum` | `int64_t`/64 | 声明序号 10；同上 | 行/core | `DoTiling` | GM base 与非尾行核 | 非尾行核使用 |
| 同上 | `rowTailLen` | `int64_t`/64 | 声明序号 11；同上 | 行 | `DoTiling` | 尾行核有效行数 | 最后行核使用 |
| 同上 | `maxUbBlockNum` | `int64_t`/64 | 声明序号 12；同上 | 32 元素块 | `16*floor(ubSize/bytesPer512)` | `Init/GetUbParams` | 必须使分母非零且实际 UB 分配可容纳 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001..003 | `GET_TPL_TILING_KEY(TPL_AXIS_TAIL)` + 6 个 FP8 binary dtype 项 | `Tiling4AntiMxQuant`→`AntiMxQuantTailAxisTiling::DoTiling` | `AntiMxQuantTilingData` | `anti_mx_quant<axisMode=0>` | `AntiMxQuantTailAxis<T,U>`；`anti_mx_quant_common.h` cast traits | `anti_mx_quant_binary.json` + `CMakeLists.txt` ascend950 |
| R-004..006 | 同一 tail key + 6 个 FP4 binary dtype 项 | 同上 | 同上 | 同上 | 同上，FP4 overload | 同上 |

## Kernel 实现

### 公共执行流程

entry 注册 POD，声明 AIV-only，950 `__NPU_ARCH__==3510` 时先保存 overflow SPR；`Init` 暂置 overflow mode 为 0、解析 POD、计算当前 core 的二维 GM base 与 UB chunk，初始化四个 UB 对象；`Process` 遍历 row chunk×col chunk，严格执行 `CopyIn→ComputeScale→ComputeData→CopyOut`。结束后 entry 恢复原 SPR。全部 Route 为单 Kernel，无 workspace、event、atomic、SIMT/NDDMA/通信节点。

### Route R-001：FP8→FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_anti_mx_quant.cpp` | `CheckDtypeValid/CheckAttrValid` | x 为两种 FP8，`dstType=0` 且 y FP32 | 加入 launcher |
| Tiling | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcTilingKey` | 通过公共 shape guard | `GET_TPL_TILING_KEY(TPL_AXIS_TAIL)`；宏展开数值固定仓不可见 |
| Config | `op_host/config/ascend950/anti_mx_quant_binary.json` | `AntiMxQuant_e5m2tofp32/e4m3tofp32` | dtype tuple | 编译 `T,float` binary |
| Router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<axisMode>` | `axisMode==TPL_AXIS_TAIL` | `AntiMxQuantTailAxis<T,float>` |

#### Device 数据流

##### 函数级调用链

```text
Tiling4AntiMxQuant -> DoTiling -> anti_mx_quant<0>(x,mxScale,y,unused,tiling)
  -> AntiMxQuantTailAxis<T,float>::Init
  -> Process
    -> CopyIn -> Compute -> ComputeScale(float*) -> ComputeData(float*) -> CopyOut
```

##### 源码等价伪码

```text
core=(blockIdx/colTileNum, blockIdx%colTileNum)
for each row chunk, col chunk:
  DMA-pad FP8 x rows and compact-copy paired E8M0 scales
  blocks=align_up(valid_32_blocks,2); vfLoops=ceil(rowChunks*blocks/16)
  scale: E8M0 -> BF16 -> FP32, interleave and store scale TBuf
  each VF loop: load 512 FP8 -> eight FP32 regs
                broadcast 16 scales, multiply 32 values/scale in FP32
                interleave order, store 512 FP32 to y Queue
  DMA only valid rows*columns to GM, skipping padded UB lanes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `coreRow*rowNormal*C + coreCol*colNormal*512 + rowLoop*rowChunk*C + colLoop*colChunk` | 有效 `rows*cols` | FP8 ND | caller | `CopyIn` | caller/executor | row-strided pad copy |
| GM scale | `coreRow*rowNormal*scaleCols + coreCol*colNormal*16 + rowLoop*rowChunk*scaleCols + colLoop*colChunkBlocks` | `rows*align_up(blocks,2)` bytes | E8M0 flattened pair layout | caller | `CopyIn` | caller/executor | compact copy |
| `inQueueX_` | Queue slot | `B*32` B/slot | uint8 FP8 bytes | `CopyIn` | `ComputeData` | pipe；Free after compute | depth 2 |
| `inQueueScale_` | Queue slot | `align_up(B,32)` B/slot | uint8 | `CopyIn` | `ComputeScale` | pipe；Free after compute | depth 2、32 B aligned |
| `scaleBuffer_` | TBuf | `ceil(B/256)*256*4` B | FP32 | `ComputeScale` | `ComputeData` | pipe，跨阶段同 tile | 单缓冲 |
| `outQueueY_` | Queue slot | `B*32*4` B/slot | FP32 | `ComputeData` | `CopyOut` | pipe；copy 后 Free | depth 2 |
| GM y | 与 x 逻辑元素 offset 相同 | `rows*validCols*4` B | FP32 ND | `CopyOut` | caller | executor→caller | 不写 padding |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inX["inQueueX · depth=2 · 2·B·32 B"]:1
  inS["inQueueScale · depth=2 · 2·align_up(B,32) B"]:1
  scale["scaleBuffer · ceil(B/256)·256·4 B"]:1
  outY["outQueueY · depth=2 · 2·B·32·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*B*32 + 2*align_up(B,32) + ceil(B/256)*256*4 + 2*B*32*4
Host coarse budget: bytesPer512=5216; B=16*floor(ubSize/5216)
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x Queue slot | `CopyIn` | `ComputeData` | 后续 tile `AllocTensor` | `EnQue→DeQue→FreeTensor` | compute 完成前不 Free |
| scale Queue slot | `CopyIn` | `ComputeScale` | 后续 tile | 同上 | scale 转换完成后 Free |
| scale TBuf | `ComputeScale` | `ComputeData` | 下一 tile `ComputeScale` | 同一 `Compute` 调用顺序 | 当前 `ComputeData` 完成后覆盖 |
| y Queue slot | `ComputeData` | `CopyOut` | 后续 tile | `EnQue→DeQue→FreeTensor` | DMA out 后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VEC | Queue + `DataCopyPad` | `CopyIn` | `DeQue` | x/scale slot | x pad 为 0；scale block 数取偶数 |
| scale | MicroAPI 顺序 | `ComputeScale` | `ComputeData` | scale TBuf | E8M0→BF16→FP32，full mask 处理已搬入/配对范围 |
| multiply | MicroAPI register dataflow | FP8 cast/scale broadcast | interleave/store | RegTensor | FP32 multiply；每 scale 覆盖 32 data |
| VEC→GM | Queue + `DataCopyPad` | y store/`EnQue` | `CopyOut` | y slot | `ubFactorColNum` 限制有效列，`srcStride` 跳 UB pad |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcBytesPerBlock/DoTiling` | 5216 B/512 元素、tail template key、blockDim、workspace 0 |
| Kernel entry/router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<0>` | AIV-only；映射到 `AntiMxQuantTailAxis<T,float>` |
| Device body/helper | `op_kernel/arch35/anti_mx_quant_tail_axis.h` | `ComputeScale(float*)/ComputeData(float*)` | FP32 scale、乘法、重排和 FP32 store |

### Route R-002：FP8→BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `op_api/aclnn_anti_mx_quant.cpp` | `CheckDtypeValid` | FP8，`dstType=27`，y BF16；公共尾轴 guard | tail template key |
| Config/Router | `op_host/config/ascend950/anti_mx_quant_binary.json` | `e5m2tobf16/e4m3tobf16` | dtype tuple + `axisMode=0` | `AntiMxQuantTailAxis<T,bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
anti_mx_quant<0> -> Init -> Process -> CopyIn
  -> ComputeScale(float*) -> ComputeData<T,bfloat16_t>(float*) -> CopyOut
```

##### 源码等价伪码

```text
沿用 R-001 的 core/tile/DMA/FP32 multiply；每组偶/奇 FP32 layout
分别以 ZERO/ONE、SAT、CAST_RINT 转 BF16，Reg::Add 合并互斥 lane，B16 store。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/Queue/scale TBuf | 同 R-001 | x `B*32`/slot；scale `align_up(B,32)`/slot；scale TBuf `ceil(B/256)*1024` | FP8/E8M0/FP32 | CopyIn/ComputeScale | ComputeData | 同 R-001 | x/scale depth 2 |
| `outQueueY_`/GM y | 同 R-001 逻辑 offset | `B*32*2` B/slot；GM 仅有效列 | BF16 | ComputeData/CopyOut | caller | Queue depth 2 | 尾 padding 不搬出 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inX["inQueueX · 2·B·32 B"]:1
  inS["inQueueScale · 2·align_up(B,32) B"]:1
  scale["FP32 scaleBuffer · ceil(B/256)·1024 B"]:1
  outY["outQueueY · 2·B·32·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*B*32 + 2*align_up(B,32) + ceil(B/256)*1024 + 2*B*32*2
Host bytesPer512=3168; B=16*floor(ubSize/3168); workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三 Queue + scale TBuf | 同 R-001 | 同 R-001 | 下一 tile | Queue lifecycle / 函数顺序 | 当前消费者结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | 同 R-001 Queue/MicroAPI | CopyIn/Compute | DeQue/CopyOut | tile buffers | FP32 multiply 后 RINT/SAT 降 BF16；tail 同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcBytesPerBlock` | FP8+B16 为 3168 B/512 |
| Kernel entry/router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<0>` | compile-time `U=bfloat16_t` |
| Device body/helper | `op_kernel/arch35/anti_mx_quant_tail_axis.h` | FP8 `ComputeData` BF16 branch | FP32 multiply，双 layout cast+Add 合并 |

### Route R-003：FP8→FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `op_api/aclnn_anti_mx_quant.cpp` | `CheckDtypeValid` | FP8，`dstType=1`，y FP16；公共尾轴 guard | tail template key |
| Config/Router | `op_host/config/ascend950/anti_mx_quant_binary.json` | `e5m2tofp16/e4m3tofp16` | dtype tuple | `AntiMxQuantTailAxis<T,half>` |

#### Device 数据流

##### 函数级调用链

```text
anti_mx_quant<0> -> Init -> Process -> CopyIn
  -> ComputeScale(float*) -> ComputeData<T,half>(float*) -> CopyOut
```

##### 源码等价伪码

```text
同 R-002，但 FP32 multiply 后用 ZERO/ONE layout、SAT、CAST_RINT 转 FP16，
再 Add 合并互斥 lane 并 B16 store。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 全部资源 | 同 R-002 | 同 R-002；y dtype 改 FP16 | FP8/E8M0/FP32 scale/FP16 y | 同 R-002 | 同 R-002 | 同 R-002 | 同 R-002 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inX["inQueueX · 2·B·32 B"]:1
  inS["inQueueScale · 2·align_up(B,32) B"]:1
  scale["FP32 scaleBuffer · ceil(B/256)·1024 B"]:1
  outY["outQueueY · 2·B·32·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total 与 Host bytesPer512=3168 同 R-002；workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三 Queue + scale TBuf | 同 R-002 | 同 R-002 | 下一 tile | Queue lifecycle / 函数顺序 | 当前消费者结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | Queue/MicroAPI | CopyIn/Compute | DeQue/CopyOut | tile buffers | FP32 multiply 后 RINT/SAT 降 FP16；tail 同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcBytesPerBlock` | FP8+B16 为 3168 B/512 |
| Kernel entry/router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<0>` | compile-time `U=half` |
| Device body/helper | `op_kernel/arch35/anti_mx_quant_tail_axis.h` | FP8 `ComputeData` FP16 branch | FP32 multiply 后 FP16 cast/store |

### Route R-004：FP4→FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_anti_mx_quant.cpp` | `CheckDtypeValid`/连续性分支 | 两种 FP4、FP32，x 连续 | Tiling |
| Tiling | `op_host/arch35/anti_mx_quant_tiling_arch35.cpp` | `CheckShape` | `C%2==0` 且尾轴/scale shape 合法 | tail template key |
| Config/Router | `op_host/config/ascend950/anti_mx_quant_binary.json` | `e2m1tofp32/e1m2tofp32` | dtype tuple | `AntiMxQuantTailAxis<T,float>` |

#### Device 数据流

##### 函数级调用链

```text
anti_mx_quant<0> -> Init -> Process -> CopyIn(FP4 packed)
  -> ComputeScale(bfloat16_t*) -> ComputeData<T,float>(bfloat16_t*) -> CopyOut
```

##### 源码等价伪码

```text
x GM base/stride 除 2；每 VF loop 读取 256 packed bytes并 UNPACK4 成 512 FP4
scale E8M0 -> BF16；每 16 scales 通过 E2B/Interleave 扩为 32-lane broadcast
FP4 -> BF16，BF16 multiply；每 128 BF16 分成两组 FP32、Interleave 恢复顺序
store 512 FP32；CopyOut 只搬有效列并跳过对齐膨胀区
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | 公共逻辑 offset `/2` | `rows*cols/2` B | packed FP4×2 | caller | `CopyIn` | caller | C 为偶数 |
| GM scale | 同 R-001 | `rows*align_up(blocks,2)` B | E8M0 | caller | `CopyIn` | caller | paired |
| `inQueueX_` | slot | `B*16` B/slot | packed bytes | `CopyIn` | FP4 `ComputeData` | depth 2 | DMA pad 到 32 B |
| `inQueueScale_` | slot | `align_up(B,32)` B/slot | uint8 | `CopyIn` | `ComputeScale` | depth 2 | 32 B aligned |
| `scaleBuffer_` | TBuf | `ceil(B/256)*256*2` B | BF16 | `ComputeScale` | `ComputeData` | 单缓冲 | 每轮 256 scales |
| `outQueueY_`/GM y | 公共逻辑 offset | `B*32*4` B/slot；GM 有效 `rows*cols*4` | FP32 | Compute/CopyOut | caller | Queue depth 2 | tail stride |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inX["packed inQueueX · 2·B·16 B"]:1
  inS["inQueueScale · 2·align_up(B,32) B"]:1
  scale["BF16 scaleBuffer · ceil(B/256)·512 B"]:1
  outY["outQueueY · 2·B·32·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*B*16 + 2*align_up(B,32) + ceil(B/256)*256*2 + 2*B*32*4
Host bytesPer512=4640; B=16*floor(ubSize/4640); workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x/scale/y Queue slot | CopyIn/Compute | Compute/CopyOut | 下一 tile | EnQue/DeQue/FreeTensor | 当前消费者完成 |
| BF16 scale TBuf | `ComputeScale` | `ComputeData` | 下一 tile | 同一 Compute 顺序 | 当前 data compute 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| unpack/compute | Queue + register dataflow | packed DMA | FP4 cast/multiply | tile | E8M0→BF16，FP4→BF16，BF16 multiply，再升 FP32 |
| output tail | `DataCopyPad` + `srcStride` | padded y UB | GM | y 有效域 | 不写 UB 补齐元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcBytesPerBlock` | FP4+FP32 为 4640 B/512 |
| Kernel entry/router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<0>` | FP4→FP32 binary template |
| Device body/helper | `op_kernel/arch35/anti_mx_quant_tail_axis.h` | FP4 `ComputeScale/ComputeData` | packed load、BF16 multiply、FP32 expand/store |

### Route R-005：FP4→BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `op_api/aclnn_anti_mx_quant.cpp` / `op_host/arch35/anti_mx_quant_tiling_arch35.cpp` | dtype/shape checks | FP4、BF16、连续且 `C%2==0`、尾轴 | tail template key |
| Config/Router | `op_host/config/ascend950/anti_mx_quant_binary.json` | `e2m1tobf16/e1m2tobf16` | dtype tuple | `AntiMxQuantTailAxis<T,bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
anti_mx_quant<0> -> Init -> Process -> CopyIn
  -> ComputeScale(bfloat16_t*) -> ComputeData<T,bfloat16_t> -> CopyOut
```

##### 源码等价伪码

```text
同 R-004 的 packed load、scale broadcast 与 BF16 multiply；四个 128-lane BF16
结果直接 B16 store，不升 FP32。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/输入 Queue/scale TBuf | 同 R-004 | 同 R-004 | packed FP4/E8M0/BF16 | 同 R-004 | 同 R-004 | 同 R-004 | 同 R-004 |
| `outQueueY_`/GM y | 公共逻辑 offset | `B*32*2` B/slot；GM 有效列 | BF16 | Compute/CopyOut | caller | depth 2 | tail stride |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inX["packed inQueueX · 2·B·16 B"]:1
  inS["inQueueScale · 2·align_up(B,32) B"]:1
  scale["BF16 scaleBuffer · ceil(B/256)·512 B"]:1
  outY["outQueueY · 2·B·32·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*B*16 + 2*align_up(B,32) + ceil(B/256)*512 + 2*B*32*2
Host bytesPer512=2624; B=16*floor(ubSize/2624); workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三 Queue + scale TBuf | 同 R-004 | 同 R-004 | 下一 tile | Queue lifecycle / 顺序 | 当前消费者完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | Queue + register flow | CopyIn/scale/data | CopyOut | tile | BF16 multiply、BF16 直接输出；tail 同 R-004 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcBytesPerBlock` | FP4+B16 为 2624 B/512 |
| Kernel entry/router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<0>` | `U=bfloat16_t` |
| Device body/helper | `op_kernel/arch35/anti_mx_quant_tail_axis.h` | FP4 `ComputeData` BF16 branch | BF16 multiply 结果直接 store |

### Route R-006：FP4→FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `op_api/aclnn_anti_mx_quant.cpp` / `op_host/arch35/anti_mx_quant_tiling_arch35.cpp` | dtype/shape checks | FP4、FP16、连续且 `C%2==0`、尾轴 | tail template key |
| Config/Router | `op_host/config/ascend950/anti_mx_quant_binary.json` | `e2m1tofp16/e1m2tofp16` | dtype tuple | `AntiMxQuantTailAxis<T,half>` |

#### Device 数据流

##### 函数级调用链

```text
anti_mx_quant<0> -> Init -> Process -> CopyIn
  -> ComputeScale(bfloat16_t*) -> ComputeData<T,half> -> CopyOut
```

##### 源码等价伪码

```text
同 R-005 得到四组 BF16 multiply 结果；各组以 SAT、CAST_RINT 转 FP16，
然后 B16 store，CopyOut 仅提交有效列。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 全部资源 | 同 R-005 | 同 R-005；y dtype 为 FP16 | packed FP4/E8M0/BF16 scale/FP16 y | 同 R-005 | 同 R-005 | 同 R-005 | 同 R-005 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inX["packed inQueueX · 2·B·16 B"]:1
  inS["inQueueScale · 2·align_up(B,32) B"]:1
  scale["BF16 scaleBuffer · ceil(B/256)·512 B"]:1
  outY["outQueueY · 2·B·32·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total 与 Host bytesPer512=2624 同 R-005；workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三 Queue + scale TBuf | 同 R-005 | 同 R-005 | 下一 tile | Queue lifecycle / 顺序 | 当前消费者完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | Queue + register flow | CopyIn/scale/data | CopyOut | tile | BF16 multiply 后 RINT/SAT 降 FP16；tail 同 R-004 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `CalcBytesPerBlock` | FP4+B16 为 2624 B/512 |
| Kernel entry/router | `op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<0>` | `U=half` |
| Device body/helper | `op_kernel/arch35/anti_mx_quant_tail_axis.h` | FP4 `ComputeData` FP16 branch | BF16 multiply 后 FP16 cast/store |

## 算子特定优化

### Trick T-001：二维因子分核保留连续列块

#### 触发与源码依据

- 适用 Route：R-001..R-006
- 触发条件：非空输入通过尾轴检查，`usedCoreNum=min(totalAiv,M*ceil(C/512))` 后存在合法 `(m,n)` 切分。
- Golden 来源：[`quant/anti_mx_quant/op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp:102-120`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp#L102-L120)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// Calculate available cores
tilingParam_.usedCoreNum =
    std::min(tilingParam_.totalCoreNum, tilingParam_.rowBlockLoopNum * tilingParam_.colBlockLoopNum);
tilingParam_.usedCoreNum = tilingParam_.usedCoreNum == 0 ? 1 : tilingParam_.usedCoreNum;

// Find split combinations
std::set<int64_t> cutSet = FindSplitCombo(tilingParam_.usedCoreNum);
std::vector<std::vector<int64_t>> allTiling;

// Enumerate row-direction split (m)
for (int64_t m : cutSet) {
    if (m > tilingParam_.rowBlockLoopNum) {
        break;
    }
    int64_t n = tilingParam_.usedCoreNum / m;
    n = n < 1 ? 1 : n;
    if (n > tilingParam_.colBlockLoopNum) {
        continue;
    }
```

#### 机制与设计原因

Host 不把所有核只切在展平元素轴，而是把 `M` 行与 `ceil(C/512)` 列块映射为二维核网格，并只保留不超过两维可分块数的 `(m,n)` 候选。这使每核 Device body 能保持整行 stride 和连续 512 列块 DMA，同时利用行、列两个方向的并行度。该结论只说明源码调度机制，不包含未实测的性能比例。

#### 迁移边界

- 适用：末轴按固定块共享 scale、同时存在行和列并行度的逐元素算子。
- 前提：可把非末维无歧义折叠为行，列块间无归约依赖，候选 core 数/乘法不溢出。
- 失效：非尾轴、跨列依赖或无合法 `(m,n)` 时必须换布局/Route；本算子直接 Tiling 失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP4/FP8 编译期分流控制 scale 精度和 UB

#### 触发与源码依据

- 适用 Route：R-001..R-006
- 触发条件：binary config 已把 `DTYPE_X/DTYPE_Y` 固化，`IsFp8Type<T>()` 在编译期成立或不成立。
- Golden 来源：[`quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h:130-148`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h#L130-L148)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (IsFp8Type<T>()) {
    xBufSize = maxUbBlockNum_ * BLOCK_SIZE * sizeof(uint8_t);
    // Scale buffer: 1 byte per 1×32 block, 32-byte aligned
    scaleBufSize = ((maxUbBlockNum_ + UBBlockSize_ - 1) / UBBlockSize_) * UBBlockSize_ * sizeof(uint8_t);
    yBufSize = maxUbBlockNum_ * BLOCK_SIZE * sizeof(U);
    // ComputeScale: each loop stores vfLen8 floats (256 * 4 = 1024 bytes)
    int64_t maxScaleNum = maxUbBlockNum_;
    int64_t scaleLoopNum = (maxScaleNum + vfLen8 - 1) / vfLen8;
    scaleComputeBufSize = scaleLoopNum * vfLen8 * sizeof(float);
} else {
    xBufSize = maxUbBlockNum_ * BLOCK_SIZE / DIGIT_TWO;
    // Scale buffer: 1 byte per 1×32 block, 32-byte aligned
    scaleBufSize = ((maxUbBlockNum_ + UBBlockSize_ - 1) / UBBlockSize_) * UBBlockSize_ * sizeof(uint8_t);
    yBufSize = maxUbBlockNum_ * BLOCK_SIZE * sizeof(U);
    // ComputeScale: each loop stores vfLen8 bfloat16s (256 * 2 = 512 bytes)
    int64_t maxScaleNum = maxUbBlockNum_;
    int64_t scaleLoopNum = (maxScaleNum + vfLen8 - 1) / vfLen8;
    scaleComputeBufSize = scaleLoopNum * vfLen8 * sizeof(bfloat16_t);
}
```

#### 机制与设计原因

FP4 x 保持一字节两元素的 packed UB 表示，scale 只扩为 BF16；FP8 x 按一字节一元素，scale 扩为 FP32。`if constexpr` 同时消除运行时 dtype 分支和不需要的 scale buffer 宽度，Host 的四种 `bytesPer512` 又把这一差异反馈到 tile 上限。迁移时必须保持 Host 粗粒度模型与 Device 实际对齐公式一致。

#### 迁移边界

- 适用：输入存储位宽决定计算精度和中间 buffer 的多 dtype elementwise 算子。
- 前提：dtype 在 binary 选择时已固定，所有模板实例都有 config 接线。
- 失效：动态 dtype、FP4 非 packed 或精度要求变化时不能复用同一分支。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-003：偶数 scale 块与搬出 stride 隔离尾部 padding

#### 触发与源码依据

- 适用 Route：R-001..R-006 的列尾 chunk，尤其 32 元素块数为奇数或 `C` 非 32 对齐时。
- 触发条件：`ubFactorColBlockNum%2!=0` 或最后 32 元素块不满。
- Golden 来源：[`quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h:331-350`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h#L331-L350)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

int64_t anotherStride =
    ((ubFactorColNum + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE - ubFactorColNum) * sizeof(U) / UBBlockSize_;
int64_t srcStride = scaleIsNotOdd * sizeof(U) + anotherStride;          // scaleIsNotOdd为1时，表示输入 X 需要补一个BlockSize 32，搬出时不需要搬出这个补的block，对应的UBBlock数量就是scaleIsNotOdd * sizeof(U)；
                                                                        // anotherStride 指的是尾block块内的元素，多补到BlockSize的元素不应该搬出，这一部分需要跳过

LocalTensor<U> yLocal = outQueueY_.DeQue<U>();
int64_t yOffset = rowLoopIdx * ubFactorRowNormalNum_ * colNum_ +
                  colLoopIdx * ubFactorColNormalLen_;

DataCopyExtParams copyOutParamY = {
    static_cast<uint16_t>(ubFactorRowNum),
    static_cast<uint32_t>(ubFactorColNum * sizeof(U)),
    static_cast<uint32_t>(srcStride),   // FP8 非偶数Block在数据搬入时会跳一个Block(32个数)，反量化为B16或者B32分别对应2*32字节或者4*32字节
                                        // FP4 在搬入时一个UBBlock是64个数对齐，天然偶数对齐，所以搬入无需跳Block，但是搬出时，非对齐场景，由于量化
                                        // 会有字节膨胀，ub存在多余的 0 不能搬出，这时候需要跳跃这些 0；
    static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(U)), 0};
DataCopyPad(yGm_[yOffset], yLocal, copyOutParamY);
outQueueY_.FreeTensor(yLocal);
```

#### 机制与设计原因

Compute 总把 32 元素块数补成偶数，以匹配 scale 的 pair layout 和每轮 16 块 MicroAPI 管线；CopyIn 把越界 x lane 补 0。CopyOut 的 `srcStride` 同时跳过“奇数块补出的完整 32 元素”和“不满 32 元素尾块补出的 lane”，而 GM blockLen 仍只取 `ubFactorColNum*sizeof(U)`，避免把 UB padding 物化到输出。

#### 迁移边界

- 适用：计算宽度必须向块/配对粒度上取整、但输出保持原逻辑宽度的逐行算子。
- 前提：stride 单位、`UBBlockSize_` 和每行 padding 公式与 DMA API 一致。
- 失效：输出布局含 padding、跨行连续拼接或 block size 变化时必须重算 stride。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP8→FP32 | R-002/R-003 FP8→B16 | R-004 FP4→FP32 | R-005/R-006 FP4→B16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | FP8+FP32 | FP8+BF16/FP16 | FP4+FP32 | FP4+BF16/FP16 | config dtype tuple；全为 `axisMode=0` 的 tail template key |
| 模板与 dtype | `T,float` | `T,bfloat16_t/half` | `T,float` | `T,bfloat16_t/half` | compile-time macro |
| 循环与分核 | 二维核网格、row/col UB loop | 同 R-001 | 同 R-001 | 同 R-001 | dtype 不改变每 VF 512 元素的分核算法 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | FP32 scale/y | FP32 scale、B16 y | BF16 scale、FP32 y | BF16 scale/y | packed 位宽和计算精度 |
| 内存总占用与复用 | 5216 B/512 coarse | 3168 B/512 | 4640 B/512 | 2624 B/512 | Queue depth 2 + scale TBuf |
| 搬运/Vector/SIMT/NDDMA/通信 API | FP8 DINTLV、FP32 cast/Mul | 再降精度 cast | FP4 UNPACK4、BF16 Mul、升 FP32 | FP4 UNPACK4、BF16 Mul | 均无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue；FP32 | Queue；FP32→B16 | Queue；BF16→FP32 | Queue；BF16/→FP16 | 相同 pad/stride tail |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | 无 | 非法 dtype/shape/axis 在 Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档明确不支持 | 公开文档明确不支持 | 公开支持 Ascend 950PR/950DT |
| Host/Launcher | 无公开支持链路 | 无公开支持链路 | `arch35` Tiling、12 dtype binary、公开 ACLNN 闭合 |
| Kernel/资源 | N/A | N/A | 单 AIV_VECTOR；双 Queue + scale TBuf；workspace 0 |

## 限制与失败边界

- 公开 ACLNN 文档把 `axis` 范围写为 `[-R,R-1]`，但固定 Host/Tiling 仅实现尾轴；非尾轴在 `CheckShape` 返回 `GRAPH_FAILED`，不存在 fallback。
- ACLNN 层未完整验证 `mxscale` 各维与 `y.shape==x.shape`；Tiling 对实际 launcher 输入/输出执行完整 storage-shape 检查。调用方仍应遵守公开 shape 约束。
- FP4 `x` 最后一维必须为偶数且必须连续；FP8/scale 可由 executor 连续化。空 `x` 或空 `mxscale` 在 ACLNN 第一段成功短路，不启动 Device。
- Host UB 粗模型按每 512 元素计费；Device 对 scale Queue 向 32 B、scale TBuf 向 256 scale 元素上取整。固定源码没有对实际 `UB_total` 再检查，也没有显式检查 `maxUbBlockNum>0`，迁移时不能省略这一一致性校核。
- `GET_TPL_TILING_KEY(TPL_AXIS_TAIL)` 的实参可确定为 `axisMode=0`，但宏定义来自仓外 AscendC 头，固定仓不能给出最终数值 key；不得把模板参数 0 直接冒充宏展开值。
- `DataCopyExtParams.blockCount`、二维 split 与 MicroAPI loop 存在到 `uint16_t` 的窄化；固定源码没有独立的超界 fallback。950 实际平台资源约束由 platform producer 决定，本次未用真机验证极值。
- 950 entry 会临时把 overflow control SPR 设为 0，并在返回前恢复；迁移不得遗漏成对保存/恢复。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/anti_mx_quant/docs/aclnnAntiMxQuant.md` | 产品表、函数原型、shape 约束 | 950 公开性、API 语义、A2/A3 不支持 |
| 入口实现 | `quant/anti_mx_quant/op_api/aclnn_anti_mx_quant.cpp` | `aclnnAntiMxQuantGetWorkspaceSize/aclnnAntiMxQuant` | 参数检查、空域、连续化、executor |
| 入口实现 | `quant/anti_mx_quant/op_api/anti_mx_quant.cpp` | `l0op::AntiMxQuant` | 输出分配与 AICore launcher |
| 平台注册 | `quant/anti_mx_quant/op_host/anti_mx_quant_def.cpp` | `ops::AntiMxQuant` | dtype tuples、Ascend 950 配置 |
| Host/Tiling | `quant/anti_mx_quant/op_host/arch35/anti_mx_quant_tiling_arch35.cpp` | `Tiling4AntiMxQuant` | dtype/shape/platform guard |
| Host/Tiling | `quant/anti_mx_quant/op_host/arch35/anti_mx_quant_tail_axis_tiling_arch35.cpp` | `AutoTiling/DoTiling` | 二维分核、UB、POD、blockDim/workspace |
| POD | `quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tilingdata.h` | `AntiMxQuantTilingData` | 13 字段 ABI |
| 模板选择 | `quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_struct.h` | `ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL` | `axisMode` 的 tail/non-tail 模板域 |
| Kernel entry/body | `quant/anti_mx_quant/op_kernel/anti_mx_quant.cpp` | `anti_mx_quant<axisMode>` | AIV-only entry/router、SPR |
| Kernel entry/body | `quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h` | `AntiMxQuantTailAxis<T,U>` | 全部搬运、计算、同步、tail |
| Kernel helper | `quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_common.h` | cast traits/constants | dtype traits、VReg/UB block、round/mask |
| 构建接线 | `quant/anti_mx_quant/op_host/config/ascend950/anti_mx_quant_binary.json` | 12 个 `bin_filename` 项 | 4×3 dtype binary 闭合 |
| 构建接线 | `quant/anti_mx_quant/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | 仅 ascend950 + arch35 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 13×`int64_t`、Host/Device 共用定义并按 `sizeof` 直接 memcpy | 字段顺序、单位、inactive 字段仍保留；数值 offset 需由目标 ABI 确认 | `AntiMxQuantTilingData/SetTilingParams` |
| 调用顺序 | `CopyIn→ComputeScale→ComputeData→CopyOut` | Queue 生命周期和 scale producer→consumer | `Process/Compute` |
| workspace/内存 | 算子 workspace 0；三双 Queue + 单 scale TBuf | depth、packed FP4、32 B/256-element 对齐、actual UB 上界 | `DoTiling/Init` |
| 同步 | Queue EnQue/DeQue/Free + 同函数寄存器顺序 | y DMA 完成前不可 Free；scale compute 完成前不可覆盖 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP8 用 FP32 multiply，FP4 用 BF16 multiply；偶数 scale block + stride tail | cast trait、RINT/SAT、512/32 粒度、有效 GM 长度 | `ComputeScale/ComputeData/CopyOut` |
| 失败边界 | 非尾轴、非法 scale shape/dtype、FP4 奇数末维失败 | 不得虚构 non-tail 或其它平台 fallback | `CheckShape/CheckDtype` |
