---
type: Operator
title: "GroupedDynamicMxQuant 950 ACLNN 算子知识"
description: "说明 aclnnGroupedDynamicMxQuantV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/docs/aclnnGroupedDynamicMxQuantV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_api/aclnn_grouped_dynamic_mx_quant_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_api/grouped_dynamic_mx_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_host/grouped_dynamic_mx_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/grouped_dynamic_mx_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_host/config/ascend950/grouped_dynamic_mx_quant_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:30:14Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GroupedDynamicMxQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | CANN 9.1.0 公开两段式 `aclnnGroupedDynamicMxQuantV2GetWorkspaceSize` → `aclnnGroupedDynamicMxQuantV2`；V2 暴露 `scaleAlg`/`dstTypeMax`，是 950 实际支持的最新公开版本。 |
| 计算语义 | 按 `groupIndex` 沿 `x` 第 0 维分组，再对每组每 32 行、每一列生成一个 E8M0 scale；`scaleAlg=0` 使用 OCP 共享指数，`scaleAlg=1` 使用 `amax/FP8_max` 后向上取整到 2 的幂；量化值以 RINT、饱和方式转为 FP8。 |
| 输入输出 | `x:[m,n]` FP16/BF16；`groupIndex:[g]` INT32；`y:[m,n]` FP8 E4M3FN/E5M2；`mxscale:[m/64+g,n,2]` E8M0，均为 ND。 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；entry 显式 `KERNEL_TYPE_AIV_ONLY`，可达 body 为 GM↔UB、MicroAPI RegTensor 与 Queue 流程，无 Cube/AIC、MIX_CV、SIMT、NDDMA 或 AICPU。 |
| Kernel 路线 | R-001：`scaleAlg=0` OCP；R-002：`scaleAlg=1` cuBLAS-style。两者共享分核、POD、UB 与 entry，计算 body 不同。 |
| 核心资源 | 每核三个 depth=2 `TQue`、一个 `TBuf`、MicroAPI RegTensor；单 Kernel，无节点间 workspace。Tiling 固定申请 32 B 框架 workspace，但 kernel body 不读 `workspace`。 |
| 来源闭包 | `source_closed`：公开 V2 ACLNN → 参数校验/连续化 → level-0 launcher → 950 OpDef/Tiling/POD → mode-0 单 AIV entry → OCP/cuBLAS 两个完整 body → Ascend 950 binary/CMake 接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `groupIndex[q]` 是第 `q` 组的结束行，起点为 `q=0 ? 0 : groupIndex[q-1]`。每组按 32 行切块；不足 32 行的最后块只读取有效行。对每个列位置 `c`，R-001 从块内值的 BF16 指数域求最大指数，减去目标 FP8 的 `emax`（E4M3FN 为 8、E5M2 为 15）得到共享 E8M0 scale，再计算 `y=cast_rint_sat(x/scale)`。R-002 先求 `amax`，计算 `s=amax/448`（E4M3FN）或 `amax/57344`（E5M2），把 `s` 的指数按正规/非正规数规则向上取整成 E8M0，再用其倒数缩放并 RINT 饱和转换。

每两个 32-row scale 平面由 `Interleave` 组成 `mxscale[...,2]`；组内 32-row scale 平面数为奇数时补零形成偶数对。第 `q` 组从 pair 索引 `floor(groupStart/64)+q` 开始写，因此组间可留 pad pair；API/InferShape 按 `floor(m/64)+g` 分配第 0 维，与 Device 的该基址公式配套。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[m,n]` / ND | FP16、BF16 | 可非连续，第一段转连续；空 Tensor 直接返回 0 workspace，不启动核心算子。 |
| `groupIndex` | `[g]` / ND | INT32 | 可非连续，第一段转连续；不得为空；文档要求非负、非递减且末值等于 `m`，Device 用 `assert` 检查范围与单调性。 |
| `roundMode` | Host scalar | string | 仅 `"rint"`。 |
| `dstType` | Host scalar | int64 | 35→E5M2，36→E4M3FN，必须与 `y.dtype` 相同。 |
| `blocksize` | Host scalar | int64 | 仅 32。 |
| `scaleAlg` | Host scalar | int64 | 0→OCP；1→cuBLAS-style。 |
| `dstTypeMax` | Host scalar | double→float | V2 预留参数，仅 0.0；ACLNN 层未检查，Tiling `GetAttr` 拒绝非 0.0。 |
| `y` | `[m,n]` / ND | E4M3FN、E5M2 | dtype 与 `dstType` 一致；可非连续，计算后 `ViewCopy`。 |
| `mxscale` | `[m/64+g,n,2]` / ND | E8M0 | 必须连续；最后一维固定 2。 |

### 入口与版本

公开文档同时存在旧入口和 V2。旧 `aclnnGroupedDynamicMxQuant` 把 `scaleAlg=0,dstTypeMax=0` 固定后调用同一 level-0 body；V2 新增这两个实参并允许 `scaleAlg=1`，因此本卡选 V2。README 与 V2 文档均明确 Ascend 950PR/950DT 支持、A2/A3 不支持；OpDef 仅 `AddConfig("ascend950")`，顶层 CMake 仅 `SUPPORT_COMPUTE_UNIT "ascend950"`，与公开表一致。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnGroupedDynamicMxQuantV2GetWorkspaceSize` 依次创建 executor，检查必选指针、rank、shape、dtype、`roundMode`、`blocksize`、`scaleAlg` 与 `y.dtype==dstType`。平台必须满足 `AclnnUtil::IsRegbase()`。`groupIndex` 为空报错；`x` 为空则返回 `workspaceSize=0` 和空执行器流程。非空时对 `x/groupIndex` 执行框架 `Contiguous`，调用 level-0 `GroupedDynamicMxQuant` 分配连续 `yOut/mxScaleOut` 并加入一个 AICore launcher，随后以 `ViewCopy` 写用户输出；`mxscale` 非连续会在加入 launcher 后返回参数错误。

### 参数、Workspace 与生命周期

level-0 根据 `x` 分配 `yOut`，根据 `(m/(2*blocksize)+g,n,2)` 分配 `mxScaleOut`；launcher 属性顺序为 `roundMode,dstType,blocksize,scaleAlg,float(dstTypeMax)`。Tiling 将 `workspaces[0]=32`，该区域由 executor 汇总、调用者分配并在第二段 `CommonOpExecutorRun` 的 stream 生命周期内有效；核心 entry 接收 `workspace` 但没有读取。Contiguous/ViewCopy 属框架节点，其额外 workspace 由 executor 统一计入，固定仓未展开框架内部切片。

### 启动图与失败传播

```text
aclnnGroupedDynamicMxQuantV2GetWorkspaceSize
  -> Contiguous(x), Contiguous(groupIndex)                 # 框架边界
  -> l0op::GroupedDynamicMxQuant
     -> ADD_TO_LAUNCHER_LIST_AICORE(GroupedDynamicMxQuant)
        -> Tiling4GroupedDynamicMxQuant
        -> grouped_dynamic_mx_quant<mode=0>                # blockDim=usedCoreNum, AIV only
  -> ViewCopy(yOut,y), ViewCopy(mxScaleOut,mxscale)         # 框架边界
aclnnGroupedDynamicMxQuantV2(workspace,size,executor,stream)
  -> CommonOpExecutorRun                                   # 同 stream 顺序执行
```

只有一个算子自有 Device 节点，输入为连续 `x/groupIndex`，输出为连续临时 `yOut/mxScaleOut`，POD 为 `GroupedDynamicMxQuantTilingData`，workspace slice 为 `[0,32)` 但 body 不消费。launcher 添加失败返回空输出；第二段任一 executor 节点失败统一转为 `ACLNN_ERR_INNER`，阻断后续 stream 节点。框架临时 Tensor 与 workspace 均由 executor 持有至执行完成。

## Route 与参数设计

### 分核与资源计算

设 `V=GetVRegSize(context)` 字节、`C=GetCoreNumAiv()`、`U=UB_SIZE`、`g=groupSize`、`n=postAxisSize`：

```text
ubFactor = V / sizeof(T) = V/2                 # 每个列 tile 的输入元素数
uo = ceil(n / ubFactor)
tailUbFactor = n - (uo-1)*ubFactor
maxUbAvailable = floor(U / 2 / 3)
maxUbCol = floor(floor(maxUbAvailable/V)/64)*64 # 每次载入的最大行数，64 行对齐
tasks = uo * g
coreData = ceil(tasks/C)
usedCoreNum = ceil(tasks/coreData)
blockFactor = ceil(tasks/usedCoreNum)
tailBlockFactor = tasks-(usedCoreNum-1)*blockFactor
```

每个 task 对应 `(nIdx=floor(task/g), gIdx=task%g)`；每核处理连续 task。每组行数再以 `maxUbCol` 切片。Kernel 实际 `B=align_ub(ubFactor*maxUbCol*2)=align_ub(V*maxUbCol)`，三个 depth=2 Queue 与一个 TBuf 合计 `7B`。Host 的 `maxUbCol` 只显式以 `U/6` 为三个双缓冲 Queue 预算；固定源码未给出独立的 `7B<=U` guard，这一复用实现的移植必须额外核验平台 UB，不能照搬 `/6` 公式到别的平台。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 OCP | `GET_TPL_TILING_KEY(mode=0)`；四个 `DTYPE_X×DTYPE_Y` binary specialization | ACLNN/Host 校验通过且 `scaleAlg==0` | `scaleAlg==1`→R-002；其它值失败 |
| 2 | R-002 cuBLAS-style | 同一 mode-0 entry 与四个 dtype specialization | ACLNN/Host 校验通过且 `scaleAlg==1` | 其它值失败 |

```text
平台 != Regbase/Ascend950 -> ACLNN 失败
dtype/shape/roundMode/blocksize/dstTypeMax 非法 -> ACLNN 或 Tiling 失败
scaleAlg == 0 -> R-001 ComputeOCP
scaleAlg == 1 -> R-002 ComputecuBLAS
else -> 失败（不会到达 Device）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 OCP | mode=0；binary: FP16/BF16→E4M3FN/E5M2 | 950，合法二维输入、`blocksize=32,scaleAlg=0,dstTypeMax=0` | 单 AIV `grouped_dynamic_mx_quant` | `GroupedDynamicMxQuantBaseFP8<DTYPE_X,DTYPE_Y>::ComputeOCP` | 3×双 Queue+calc TBuf；Queue 同步；RegTensor 指数计算 | 列尾补到 32；行尾小于 32；奇数 scale 平面补零；无 fallback |
| R-002 cuBLAS-style | 与 R-001 相同 | 同上但 `scaleAlg=1` | 同一单 AIV 节点 | `GroupedDynamicMxQuantBaseFP8<DTYPE_X,DTYPE_Y>::ComputecuBLAS` | UB 同 R-001；FP32 scale 计算在寄存器 | 同一 tail；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950/Regbase | ACLNN `IsRegbase()` 且 build/config 为 ascend950 | body 使用 arch35 MicroAPI 与 FP8 类型 | 两条 | ACLNN runtime error | `CheckDtypeValid`; `GroupedDynamicMxQuant::GroupedDynamicMxQuant` |
| dtype specialization | `x∈{FP16,BF16}`, `y∈{E4M3FN,E5M2}` | `DTYPE_X/DTYPE_Y` 编译期选择常量、Cast 路径 | 两条 | 参数/tiling 失败 | `CheckDtype`; binary json |
| UB 行块 | `maxUbCol=floor((U/6)/V/64)*64` | 32 行一 scale、两 scale 平面交织，行块需 64 对齐 | 两条 | 固定源码无显式 fallback；若为 0 会导致除零 | `DoTiling`; `Process` |
| 列 tile | `ubFactor=V/2`，尾块 `tailUbFactor` | 输入每元素 2 B，列 tile 对齐向量寄存器宽度 | 两条 | 尾列路线 | `DoTiling`; `Process` |
| groupIndex | Device: `0<=end<=m && start<=end` | 决定 GM 行区间与 scale offset | 两条 | Device `assert` | `Process` |
| scale algorithm | `scaleAlg==0 \|\| scaleAlg==1` | 选择两个不同数值管线 | R-001/R-002 | Tiling 失败 | `GetAttr`; `SplitPostAxisCompute` |
| `dstTypeMax` | `dstTypeMax==0.0f` | 当前 body 不消费该字段 | 两条 | Tiling 失败 | `GetAttr` |

### POD/ABI

`GroupedDynamicMxQuantTilingData` 是 13 个 `int64_t` 后跟一个 `float` 的标准顺序结构：对固定 C++ ABI 可得 `alignof=8,sizeof=112`；前 13 字段 offset 每次加 8，`dstTypeMax` offset 104，尾部补齐 4 B。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `GroupedDynamicMxQuantTilingData` | `totalCoreNum` | int64/64 | off 0 | core | `GetCoreNumAiv()` | 未读取 | 仅日志/ABI |
| 同上 | `usedCoreNum` | int64/64 | off 8 | core | `ceil(tasks/coreData)` | `ParseTilingData`; block guard | >0 |
| 同上 | `blockFactor` | int64/64 | off 16 | task/core | `ceil(tasks/usedCoreNum)` | `Process` 非尾核 loop | 正数 |
| 同上 | `tailBlockFactor` | int64/64 | off 24 | task | `tasks-(usedCoreNum-1)*blockFactor` | `Process` 尾核 loop | 正数 |
| 同上 | `uo` | int64/64 | off 32 | 列 tile | `ceil(n/ubFactor)` | `Process` 判断尾列 | 非空路径 >0 |
| 同上 | `maxUbCol` | int64/64 | off 40 | row | UB 公式后向下对齐 64 | `Process` 行块/scale offset | 必须 >0；源码无显式检查 |
| 同上 | `ubFactor` | int64/64 | off 48 | input element | `V/2` | `Init` buffer、`Process` 列 offset | 向量宽度 |
| 同上 | `tailUbFactor` | int64/64 | off 56 | input element | `n-(uo-1)*ubFactor` | `Process` 尾列有效长度 | `(0,ubFactor]` |
| 同上 | `blockSize` | int64/64 | off 64 | row/scale | 属性，仅 32 | 解析但 body 未读取 | 固定 32 |
| 同上 | `scaleAlg` | int64/64 | off 72 | enum | 属性 0/1 | `SplitPostAxisCompute` | 选择 Route |
| 同上 | `preAxisSize` | int64/64 | off 80 | row | `x.shape[0]` | groupIndex assert/offset | `m` |
| 同上 | `postAxisSize` | int64/64 | off 88 | element/row | `x.shape[1]` | GM stride、Copy 参数 | `n` |
| 同上 | `groupSize` | int64/64 | off 96 | group | `groupIndex.shape[0]` | task→group 映射 | >0 |
| 同上 | `dstTypeMax` | float/32 | off 104；sizeof 112 | scalar | `float(attr)`，仅 0 | 未读取 | inactive，尾填充 4 B |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | mode=0 + `scaleAlg=0` | `Tiling4GroupedDynamicMxQuant`/`DoTiling` | `GroupedDynamicMxQuantTilingData` | `grouped_dynamic_mx_quant<0>` | `Process`→`ComputeOCP` | ascend950 binary json + top CMake |
| R-002 | mode=0 + `scaleAlg=1` | 同上 | 同上 | 同上 | `Process`→`ComputecuBLAS` | 同上 |

`DoTiling` 还计算了按输入/输出 dtype 编码的 `tilingParam.tilingKey=11/12/21/22`，但 `SetTilingKeyParam` 没有使用它，而是始终提交 `GET_TPL_TILING_KEY(mode=0)`；四种 dtype 组合由 binary config 的 `DTYPE_X/DTYPE_Y` specialization 区分，不能把 11/12/21/22 当成可达 runtime key。

## Kernel 实现

### 公共执行流程

entry 保存 arch3510 overflow SPR，构造 `GroupedDynamicMxQuantBaseFP8<DTYPE_X,DTYPE_Y>`，执行 `Init/Process`，最后恢复 SPR。`Init` 绑定四个 GM tensor，解析 POD，并分配 UB。`Process` 将 `uo*groupSize` 个 `(列 tile,group)` task 分给 AIV core；读取 group 边界后把组内行再切成 `maxUbCol`，执行 `CopyIn→SplitPostAxisCompute→CopyOut`。`CopyIn` 把每行有效 `dataLen` 补到 `L=ceil(dataLen/32)*32`；计算以 32 行共享 scale，scale 暂存在 `calcBuf`，相邻两个 scale 平面交织到 `mxScaleQueue`，量化值进入 `outQueue`。

### Route R-001：OCP 共享指数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_grouped_dynamic_mx_quant_v2.cpp` | `CheckDtypeValid` | `scaleAlg==0 \|\| scaleAlg==1` | 允许进入 launcher |
| Tiling | `op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.cpp` | `GetAttr` | `scaleAlg==0` | POD `scaleAlg=0` |
| Router | `op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h` | `SplitPostAxisCompute` | `scaleAlg_==0` | `ComputeOCP` |

#### Device 数据流

##### 函数级调用链

```text
grouped_dynamic_mx_quant<0>(x,groupIndex,y,mxScale,workspace,tiling)
  -> GroupedDynamicMxQuantBaseFP8<DTYPE_X,DTYPE_Y>::Init
  -> Process
     -> CopyIn
     -> SplitPostAxisCompute
        -> ComputeOCP              # 每 32 行或尾行块
        -> Duplicate + Interleave  # scale 平面补偶数并交织
     -> CopyOut
```

##### 源码等价伪码

```text
task = blockIdx*blockFactor + localLoop
g = task % groupSize; colTile = task / groupSize
rowBegin = g ? groupIndex[g-1] : 0; rowEnd = groupIndex[g]
for rowChunk in [rowBegin,rowEnd) step maxUbCol:
  rows = min(maxUbCol,rowEnd-rowChunk)
  valid = colTile==uo-1 ? tailUbFactor : ubFactor
  L = ceil(valid/32)*32
  CopyIn rows × valid T; 每行 UB 补到 L
  for scaleRow in [0,rows) step 32:
    for 每个寄存器列片:
      从两端成对读取最多 32 行，FP16 先 CAST_TRUNC 到 BF16 表示
      expMax = max(exponent_bits); clamp 至目标 FP8 emax 下界
      scaleE8M0 = encode(expMax-target_emax)，处理 zero/Inf
      reciprocal = encode(bias-(expMax-target_emax))
      对每个有效行：FP32/BF16 路径乘 reciprocal，CAST_RINT+SAT 到 FP8
  scale 平面数为奇数则补零；两两 Interleave 为 [...,2]
  CopyOut y 的 rows×valid 与 scale 的 ceil(rows/64)×valid×2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `rowBegin*n+colTile*ubFactor+chunk*n` | `rows×valid` | T/row-major | caller | `CopyIn` | caller | 行 stride `n` |
| GM `groupIndex` | `g`/`g-1` | g | INT32 | caller | `Process::GetValue` | caller | scalar load |
| `inQueue` | slot base | `2×B` | T，行长 L | `CopyIn` | `SplitPostAxisCompute` | pipe；FreeTensor 后复用 | depth=2 |
| `calcBuf` | `scaleRow*L` | `B` | uint8 scale planes | `ComputeOCP` | `Interleave` | pipe 全 Process | TBuf |
| `outQueue` | `row*L+col` | `2×B` | uint8 FP8 | `ComputeOCP` | `CopyOut` | pipe | depth=2 |
| `mxScaleQueue` | `(2*pair+lane)*L+col` | `2×B` | uint8 E8M0 interleaved | `Interleave` | `CopyOut` | pipe | depth=2 |
| GM `y` | 与 x 相同 offset | `rows×valid` | FP8 row-major | `CopyOut` | caller | caller | 只提交有效列 |
| GM `mxscale` | `2*(rowBegin/64+g)*n+2*colTile*ubFactor+(chunk/32)*n` | `ceil(rows/64)×2×valid` | E8M0 `[...,2]` | `CopyOut` | caller | caller | UB pad 不写 GM |
| workspace | `[0,32)` | 32 B | opaque | framework | body 未读 | executor | 无跨 Kernel 边 |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inQ["inQueue · depth=2 · 2B"]:2
  scaleQ["mxScaleQueue · depth=2 · 2B"]:2
  outQ["outQueue · depth=2 · 2B"]:2
  calc["calcBuf · B"]:1
```

##### 字节账与复用顺序

```text
B = align_UB(ubFactor*maxUbCol*sizeof(T)) = align_UB(V*maxUbCol)
UB_total = 2B + 2B + 2B + B = 7B
Host: maxUbCol = floor((floor(U/6)/V)/64)*64
必要条件: 7B <= U；固定源码未单独编码该检查
dynamic_ub = N/A（AIV TPipe）
workspace_total = 32 B（kernel body 不读）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueue` slot | `CopyIn` | `SplitPostAxisCompute` | 后续 rowChunk | EnQue→DeQue→FreeTensor | 计算结束后才 Free |
| `calcBuf[scaleRow*L]` | `ComputeOCP` | `Interleave` | 下一 rowChunk | 同函数顺序；非 Queue | 全部 Interleave 完成后覆盖 |
| output Queue slots | Compute/Interleave | `CopyOut` | 后续 chunk | EnQue→DeQue→FreeTensor | GM copy 完成后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad`+Queue | MTE | `DeQue` | input slot | 有效列 `valid`，行尾补到 L |
| OCP reduction | RegTensor mask | register loads | scale/quant | 每列 32 行 | FP16 先截断为 BF16 指数域；尾列 mask |
| Quant | `CastTrait(SAT,ZEROING,CAST_RINT)` | scaled FP32/BF16 | output pack/store | FP8 lanes | zero/Inf 专门 Select；只 store 有效 lane |
| Scale layout | `Duplicate`+`Interleave` | calcBuf | mxScaleQueue | 两个 32-row scale 平面 | 奇数平面补零；行尾按实际 `blockCount` |
| UB→GM | Queue+`DataCopyPad` | output queues | FreeTensor | y/scale slots | 只搬有效列与 `ceil(rows/64)` scale pair |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.cpp` | `DoTiling`,`SetTilingData` | 产生 core/tile/UB/POD 与 32 B workspace |
| Kernel entry/router | `op_kernel/grouped_dynamic_mx_quant.cpp` | `grouped_dynamic_mx_quant<0>` | AIV-only；实例化 DTYPE_X/DTYPE_Y body |
| Device body/helper | `op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h` | `Process`,`SplitPostAxisCompute`,`ComputeOCP` | group/列/行循环、指数 scale、RINT FP8、交织 scale |

### Route R-002：cuBLAS-style scale

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_grouped_dynamic_mx_quant_v2.cpp` | `CheckDtypeValid` | `scaleAlg==1` 合法 | 允许进入 launcher |
| Tiling | `op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.cpp` | `GetAttr` | `scaleAlg==1` | POD `scaleAlg=1` |
| Router | `op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h` | `SplitPostAxisCompute` | `scaleAlg_==1` | `ComputecuBLAS` |

#### Device 数据流

##### 函数级调用链

```text
grouped_dynamic_mx_quant<0>
  -> Init -> Process -> CopyIn
  -> SplitPostAxisCompute
     -> ComputecuBLAS
     -> Duplicate + Interleave
  -> CopyOut
```

##### 源码等价伪码

```text
沿用 R-001 的 core/group/列 tile/行 chunk 与搬运
对每个最多 32 行的 scale block、每个 vfNum32 列片：
  amax = max(bit_abs(x[row,col]))
  amax_fp32 = cast(amax)
  s = amax_fp32 * (1/448 或 1/57344)
  (E,M) = 拆出 FP32 exponent/mantissa
  if (0<E<254 && M>0) or (E==0 && M>0.5): E += 1
  zero/Inf 选择特殊 E8M0；store scale
  reciprocal = exponent_bits(127-E)
  对各有效行乘 reciprocal，CAST_RINT+SAT 到 FP8
尾列使用 tail mask；随后与 R-001 相同地交织并 CopyOut
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/三个 Queue/calcBuf | 同 R-001 | 同 R-001：`7B` | 同 R-001 | CopyIn/Compute/Interleave | Compute/CopyOut | pipe/caller | 完全共享布局 |
| cuBLAS RegTensor | 寄存器 lane | V 宽寄存器集合，固定仓不公开物理字节分配 | uint16/uint32/float/FP8 | `ComputecuBLAS` | 同方法后继指令 | 当前 `__VEC_SCOPE__` | 不占 TPipe UB 账 |
| workspace | `[0,32)` | 32 B | opaque | framework | body 未读 | executor | N/A |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inQ["inQueue · depth=2 · 2B"]:2
  scaleQ["mxScaleQueue · depth=2 · 2B"]:2
  outQ["outQueue · depth=2 · 2B"]:2
  calc["calcBuf · B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 7B，B=align_UB(V*maxUbCol)
dynamic_ub = N/A
workspace_total = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue/calcBuf | 同 R-001 | 同 R-001 | 下一 chunk | Queue 协议/函数顺序 | 同 R-001 |
| FP32 scale/exponent RegTensor | amax/scale 阶段 | reciprocal 与 quant 阶段 | 下一寄存器列片 | 单个 `__VEC_SCOPE__` 指令顺序 | store scale 且生成 reciprocal 后可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬运/Queue | 同 R-001 | 同 R-001 | 同 R-001 | Queue slots | 同一行/列 tail |
| amax→scale | bit abs、Cast、Muls、Shift/And、mask Select | input regs | scale store/reciprocal | 每列最多 32 行 | FP32 scale；正规/非正规条件向上取整 |
| quant | Mul→FP32→`CAST_RINT,SAT` | reciprocal+input | FP8 store | 有效 lane | FP16 路径 FP32 乘；BF16 路径先 BF16 Mul 再 cast FP32 |
| scale 交织 | 同 R-001 | calcBuf | CopyOut | E8M0 pair | 奇数 scale 平面补零 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.cpp` | `GetAttr`,`SetTilingData` | 校验并传递 `scaleAlg=1` |
| Kernel entry/router | `op_kernel/grouped_dynamic_mx_quant.cpp` | `grouped_dynamic_mx_quant<0>` | 与 R-001 共用 AIV entry |
| Device body/helper | `op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h` | `ComputecuBLAS` | amax/FP8max、指数向上取整、倒数缩放与 tail |

## 算子特定优化

### Trick T-001：行对称成对归约并复用一份列 scale

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：每个 32-row scale block 的寄存器列片归约。
- Golden 来源：[`grouped_dynamic_mx_quant_not_tail_axis_fp8.h:342-359`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h#L342-L359)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint16_t j = 0; j <= static_cast<uint16_t>(blockCountLoop); j++) {
    Reg::LoadAlign(xRegTensor0, xAddr + j * dataLen + i * vfLen);
    Reg::LoadAlign(xRegTensor1, xAddr + (static_cast<uint16_t>(blockCount) - j - 1) * dataLen + i * vfLen);
    // ... 提取两端行的指数 ...
    Reg::Max(expRegTensor, expMaxRegTensor0, expMaxRegTensor1, p0);
    Reg::Max(expMaxRegTensor, expRegTensor, expMaxRegTensor, p0);
}
```

#### 机制与设计原因

从行块两端同时加载并做 `Reg::Max`，每轮把两行并入同一列向量归约；所得 scale 寄存器随后供块内全部有效行量化，不为每行重复求 scale。它改变的是寄存器归约调度与 scale 复用，不改变 32 行共享 scale 的语义。

#### 迁移边界

- 适用：固定小行块、沿列向量化的 group-wise reduction。
- 前提：行地址等距、同一 scale block 不超过 32 行、tail mask 正确。
- 失效：scale 不按列共享或行布局非等距时必须重做归约/搬运。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：scale 平面在 UB 中补偶数后直接交织为最终布局

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：一个 row chunk 的 `ceil(blockCount/32)` 个 scale 平面产生后。
- Golden 来源：[`grouped_dynamic_mx_quant_not_tail_axis_fp8.h:247-255`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h#L247-L255)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (rowNumFullAlign32 % 2 != 0) {
    Duplicate<uint8_t>(tmpBuff[rowNumFullAlign32 * dataLen], (uint8_t)0, 128);
}
for(int64_t i = 0; i < rowNumFullAlign64; i++){
    Interleave(mxScale[2 * i * dataLen], mxScale[(2 * i + 1) * dataLen], tmpBuff[2 * i * dataLen],
        tmpBuff[(2 * i + 1) * dataLen], dataLen);
}
this->mxScaleQueue_.template EnQue(mxScale);
```

#### 机制与设计原因

计算先把每 32 行的 scale 连续写进 `calcBuf`，再把相邻 scale 平面直接交织到输出 Queue；奇数平面先补零，对应最终 `[...,2]` 的 pad lane。这样 CopyOut 已是目标布局，无需额外 GM layout-transform Kernel，也没有跨 Kernel workspace 边。

#### 迁移边界

- 适用：最终布局把两个逻辑 scale 平面放在最内层双 lane 的场景。
- 前提：列 tile 已补齐到 MicroAPI 要求，GM CopyOut 只提交有效列。
- 失效：输出 scale lane 数不是 2 或 scale block 不是 32 行时必须调整 pad/offset。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 OCP | R-002 cuBLAS-style | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `scaleAlg=0` | `scaleAlg=1` | 公开 V2 属性直接选择，无 fallback |
| 模板与 dtype | mode=0；FP16/BF16→两种 FP8 | 相同 | binary specialization 决定常量与 Cast |
| 循环与分核 | task/group/列/行/寄存器 | 相同 | 两 Route 共享 Process |
| Queue/TBuf/寄存器 | 3×depth2 Queue+1 TBuf；OCP exponent regs | 同 UB；更多 FP32 exponent/mantissa regs | scale 算法不同 |
| 内存总占用与复用 | `7B`，scale calcBuf→Interleave | 相同 | body 只改变寄存器计算 |
| 搬运/Vector API | DataCopyPad；And/Max/Sub/Shift/Cast | 同搬运；bit abs/Muls/Shift/And/Select/Cast | OCP exponent vs amax/FP8max |
| 同步、精度与 tail | Queue；FP16→BF16 exponent；RINT SAT | Queue；FP32 scale/ceil；RINT SAT | 两种规范 |
| fallback/失败边界 | 非 0 去 R-002 | 非 1 失败 | Host 限定 0/1 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/V2 文档明确不支持 | 公开 README/V2 文档明确不支持 | Ascend 950PR/950DT 支持 |
| Host/Launcher | 无正式路线 | 无正式路线 | arch35 Tiling、Regbase ACLNN、ascend950 OpDef |
| Kernel/资源 | 无正式路线 | 无正式路线 | AIV-only MicroAPI，3×双 Queue+TBuf |

## 限制与失败边界

- `groupIndex` 的值约束不在 ACLNN Host 逐值验证，Device `assert` 才检查每个结束索引的范围和单调性；末值等于 `m` 来自公开接口约束，body 未单独 assert 最后一个值必须等于 `m`。
- `dstTypeMax` 从 API `double` 窄化到 level-0 `float`；ACLNN `CheckParams` 未检查它，Tiling 仅接受 0.0。
- `maxUbCol` 公式没有显式检查结果大于 0，也只按三个双 Queue 的 `6B` 预算，实际 `InitBuffer` 为 `7B`；迁移到不同 UB/VReg 配置前必须重新证明 `maxUbCol>0 && 7B<=U`。
- OCP 和 cuBLAS body 的 loop 计数窄化为 `uint16_t`；当前 `dataLen` 由寄存器宽度派生，固定 950 路线可达，不能把 POD 放大后复用。
- 核心算子固定 workspace 为 32 B 但 body 不读取；Contiguous/ViewCopy 的内部 workspace、同步实现属于框架边界，固定仓未展开。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/grouped_dynamic_mx_quant/docs/aclnnGroupedDynamicMxQuantV2.md` | V2 prototypes/产品表 | 最新公开入口、语义、950 支持 |
| 入口实现 | `quant/grouped_dynamic_mx_quant/op_api/aclnn_grouped_dynamic_mx_quant_v2.cpp` | `CheckParams`; V2 两段函数 | 校验、连续化、空 Tensor、executor |
| 入口实现 | `quant/grouped_dynamic_mx_quant/op_api/grouped_dynamic_mx_quant.cpp` | `l0op::GroupedDynamicMxQuant` | 输出分配与单 launcher |
| Host/Tiling/Launcher | `quant/grouped_dynamic_mx_quant/op_host/grouped_dynamic_mx_quant_def.cpp` | `GroupedDynamicMxQuant` OpDef | dtype 与 ascend950 注册 |
| Host/Tiling/Launcher | `quant/grouped_dynamic_mx_quant/op_host/arch35/grouped_dynamic_mx_quant_tiling_arch35.cpp` | `Tiling4GroupedDynamicMxQuant` | 校验、分核、tile、workspace、POD |
| Host/Tiling/Launcher | `quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_tilingdata.h` | `GroupedDynamicMxQuantTilingData` | Host/Device ABI |
| Host/Tiling/Launcher | `quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_struct.h` | `GroupedDynamicMxQuant` template args | mode=0/1 模板域；Host 实际取 0 |
| Kernel entry/body | `quant/grouped_dynamic_mx_quant/op_kernel/grouped_dynamic_mx_quant.cpp` | `grouped_dynamic_mx_quant` | AIV-only entry、SPR 生命周期 |
| Kernel entry/body | `quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h` | `GroupedDynamicMxQuantBaseFP8` | 全部循环、搬运、计算、同步、tail |
| Kernel entry/body | `quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_common.h` | FP8 exponent/scale constants | 两条算法的 bit mask、bias、FP8 最大值倒数 |
| 构建接线 | `quant/grouped_dynamic_mx_quant/op_host/config/ascend950/grouped_dynamic_mx_quant_binary.json` | 四个 binary 条目 | FP16/BF16×E4M3FN/E5M2 |
| 构建接线 | `quant/grouped_dynamic_mx_quant/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | 仅 ascend950、arch35 Tiling |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 13×int64+float 的 producer/consumer | 字段顺序、112 B ABI、inactive 字段也必须保留 | `GroupedDynamicMxQuantTilingData` |
| 调用顺序 | `CopyIn→Compute→Interleave→CopyOut` | Queue 生命周期与 scale 补偶数顺序 | `Process`; `SplitPostAxisCompute` |
| workspace/内存 | 3×双 Queue+calc TBuf | `B` 容量、`7B<=U`、GM offset 与 mxscale layout | `Init`; `CopyOut` |
| 同步 | EnQue/DeQue/FreeTensor | 消费前不可覆盖 slot；calcBuf 在 Interleave 前不可覆盖 | `CopyIn`; `SplitPostAxisCompute`; `CopyOut` |
| 精度与尾块 | 两 scale 算法、RINT SAT、zero/Inf、列 mask | FP16/BF16 路径顺序、32-row scale、奇数平面 pad | `ComputeOCP`; `ComputecuBLAS` |
| 失败边界 | Host/Tiling 拒绝非法属性；Device assert groupIndex | 无 fallback，不得静默接受非 0/1 或非零 dstTypeMax | `GetAttr`; `Process` |

- [Interleave](../../../../apis/simd_api/basic_api/memory_vector_compute/data_rearrange_isasi/interleave.md)
