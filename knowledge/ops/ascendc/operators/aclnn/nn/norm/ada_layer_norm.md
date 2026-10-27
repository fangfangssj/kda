---
type: Operator
title: "AdaLayerNorm A2/A3 ACLNN 算子知识"
description: "说明 aclnnAdaLayerNormV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/op_api/aclnn_ada_layer_norm_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/op_api/ada_layer_norm_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/ada_layer_norm_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_host/ada_layer_norm_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/ada_layer_norm_base_v1.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/ada_layer_norm_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/config/ascend910b/ada_layer_norm_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/config/ascend910_93/ada_layer_norm_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:42:35Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# AdaLayerNorm A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAdaLayerNormV2GetWorkspaceSize` / `aclnnAdaLayerNormV2`，公开 ACLNN V2；A2/A3 上 V2 已公开支持，且比同族 `aclnnAdaLayerNorm` 多输出 `mean/rstd`，因此本卡选择 V2。 |
| 计算语义 | 对 `x[B…,S,H]` 沿末维做 LayerNorm，再计算 `out = norm * (1 + scale[B…,H]) + shift[B…,H]`，并输出每行 `mean`、`rstd`。 |
| 输入输出 | `x/scale/shift/out/mean/rstd` 为同 dtype（FP32/FP16/BF16）、ND；可选 `weight/bias` 为 x dtype 或 V2 允许的 FP32。 |
| 执行类型 | 单节点 `AIV_VECTOR`；entry 明确 `KERNEL_TYPE_AIV_ONLY`。 |
| Kernel 路线 | R-001：key 1，非 FP32 weight/bias；R-002：key 2，FP32 或缺省 weight/bias。两条均由 `AdaLayerNormND` 内部按 H 进入多行、单行或切片子路径。 |
| 核心资源 | GM + 3 个 `TBuf`；固定 UB 184320 B；Tiling 为算子节点申请 16 MiB workspace，但 V2 Device body 不读该指针。 |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 launcher→`AdaLayerNormV2` Tiling/POD→AIV entry→`AdaLayerNormND`/helper→binary/CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令一行 `x_r` 的长度为 `H`，`mean_r = sum(x_r)/H`，`var_r = sum((x_r-mean_r)^2)/H`，`rstd_r = 1/sqrt(var_r+epsilon)`。若 `weight`/`bias` 缺省，对应乘法/加法跳过；否则 `norm_r = (x_r-mean_r)*rstd_r*weight+bias`。最终 `out_r = norm_r*(1+scale_b)+shift_b`，其中 `b=floor(r/S)`，同一 B 项的 S 行复用同一组 scale/shift。`mean/rstd` 的逻辑 shape 是 `x` 最后一维改为 1。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| x | `[B…,S,H]`，ND，2–8 维 | FP32/FP16/BF16 | 必选、非空；B、S、H 均大于 0。 |
| scale | `[B…,H]` 或 `[B…,1,H]`，ND | 与 x 相同 | 必选；Device 先加 1。 |
| shift | 同 scale | 与 x 相同 | 必选；自适应偏移。 |
| weightOptional | `[H]`，ND | x dtype 或 FP32 | 可选；与 bias 同时存在时二者 dtype 必须一致。 |
| biasOptional | `[H]`，ND | x dtype 或 FP32 | 可选；约束同 weight。 |
| epsilon | 标量 | Host `double`，传入算子时窄化为 `float` | 默认文档值 `1e-5`；源码未加额外范围检查。 |
| out | 与 x 相同 | 与 x 相同 | 必选；通过 `ViewCopy` 写用户 tensor。 |
| meanOutOptional/rstdOutOptional | `[B…,S,1]`，ND | 与 x 相同 | 可选用户输出；内部算子仍分配并产生两者。 |

### 入口与版本

公开 README/API 支持表同时将 A2、A3 标为支持；`AdaLayerNormV2` OpDef 注册 `ascend910b`、`ascend910_93`，两个 binary config 均提供 FP32/FP16/BF16 ND 实例。旧 `aclnnAdaLayerNorm` 只返回 `out`；本逻辑版本族在 A2/A3 选择公开且实际接线的最新 V2。Quant/Grad 等不同语义不在本卡范围。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段接口依次创建 executor，检查 x/scale/shift/out 非空，检查 dtype，再检查 shape。`CheckShape` 计算 B、S、H，拒绝 B/S/H 非正；验证 scale/shift 的两种合法形状、weight/bias `[H]`、out 与 x 同形、mean/rstd 为末维 1。随后 x/scale/shift 及存在的 weight/bias 经 `l0op::Contiguous`；内部 `AdaLayerNormV2` 分配 out/mean/rstd 并将一个 AICore 节点加入 launcher；最后以 `ViewCopy` 接到用户输出。任何空返回立即结束，不继续添加后继。第二段由 `CommonOpExecutorRun` 在传入 stream 上执行。

### 参数、Workspace 与生命周期

Tiling 将节点 workspace[0] 固定为 `16*1024*1024` B（仅 Quant 且 H>6144 才追加；本卡 V2 不命中）。`aclnn...GetWorkspaceSize` 返回整个 executor 的总 workspace，可能还含 Contiguous/ViewCopy 框架节点，其细分在固定仓不可见。workspace 由调用者在第二段执行期间持有；executor 从第一段转交调用者，并由同一 stream 顺序执行。V2 `ada_layer_norm_v2` entry 虽接收 workspace，但 `gmAddr` 与 `InitV2` 均不绑定或读取它。

### 启动图与失败传播

```text
ACLNN phase 1
  -> optional Contiguous nodes (framework boundary)
  -> AdaLayerNormV2 AIV node: entry ada_layer_norm_v2, blockDim=needCoreNum,
     POD=AdaLayerNormTilingData, GM inputs x/scale/shift/weight/bias,
     GM outputs internal out/mean/rstd, workspace slice requested=16 MiB
  -> ViewCopy(out) -> optional ViewCopy(mean/rstd)
ACLNN phase 2 -> CommonOpExecutorRun(workspace, size, executor, stream)
```

唯一核心语义 Device 节点是单 Kernel；没有 Kernel 间 workspace 边。executor 是所有中间 tensor 和 workspace slice 的 owner，生命周期覆盖整次第二段调用；launcher 添加失败返回空 tuple，ACLNN 随即返回内部错误，后继 ViewCopy 不会建立。

## Route 与参数设计

### 分核与资源计算

设 `rows=B*S`、`Hc32=ceil(H/32)*32`。`singleCoreNum=floor(rows/coreNumAiv)`，`tailNum=rows mod coreNumAiv`，`blockDim = singleCoreNum>0 ? coreNumAiv : tailNum`。Device core `i` 处理 `[singleCoreNum*i+min(i,tailNum), start+singleCoreNum+(i<tailNum)]`。

- H≤3072：`sliceSize=H`，`rowNum=floor(6144/Hc32)`，进入 FastProcess，多行可批处理。
- 3072<H≤6144：`sliceSize=H`，`rowNum=1`，进入 FastProcess 单行路径。
- H>6144：`batch=ceil(H/6144)`，`sliceSize=ceil(ceil(H/batch)/32)*32 < H`，进入 SliceProcess；均值和方差对 H>16384 分别多段扫描。

Host 获取 UB 大小但本版本 selector 不使用该值；没有源码可见的 `UB_total<=ubSizePlatForm` 拒绝分支。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 同 dtype 参数 | key 1，`AdaLayerNormND<X,X,12>` | `isRegBase=false` 且 V2，存在的 weight 或 bias 任一不是 FP32；由 ACLNN dtype 约束知二者为 X dtype | 否则 R-002 |
| 2 | R-002 FP32/缺省参数 | key 2，`AdaLayerNormND<X,float,12>` | `isRegBase=false` 且 V2，所有存在的 weight/bias 均为 FP32；二者都缺省时 `isWeightFloat` 也保持 true | 未知 key 无 body |

```text
isRegBase=false
  -> keyBase=0
  -> (opCode==12 && isWeightFloat) ? key=2 : key=1
  -> key 1: X=FP16/BF16（FP32 X 不可能配非 FP32 weight）
  -> key 2: X=FP32/FP16/BF16
  -> body 内：H>6144 ? SliceProcess : FastProcess(rowNum)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 同 dtype参数 | 1 | A2/A3；X=FP16/BF16；存在非 FP32 weight/bias | 单 AIV Kernel | `AdaLayerNormND<X,X,BASE_V2_OP_CODE>`；Fast 或 Slice | 固定 184320 B UB；event+PipeBarrier | core tail；H/tile tail；未知 key 不写输出 |
| R-002 FP32/缺省参数 | 2 | A2/A3；X 任一公开 dtype；存在参数均 FP32或均缺省 | 单 AIV Kernel | `AdaLayerNormND<X,float,BASE_V2_OP_CODE>`；Fast 或 Slice | 同上；weight/bias GM 按 float 读取 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 多行阈值 | H≤3072 | `rowNum=6144/Hc32`，多行铺入固定 6144 元素分区 | R-001/R-002 Fast | H>3072 转单行或切片 | `AdaLayerNormTiling::SplitCore` |
| 切片阈值 | H>6144 | 单次固定工作域无法容纳整行 | R-001/R-002 Slice | H≤6144 Fast | `SplitCore`; `AdaLayerNormND::Process` |
| 二次扫描阈值 | H>16384 | `xFloat` 的最大连续统计窗口为 16384 float | Slice | H≤16384 一次拷入后算 mean/var | `SliceProcess` |
| dtype key | `isWeightFloat` | entry 模板必须按 GM 实际元素宽度读取 weight/bias | R-001/R-002 | 另一 key | `Init`; `GetTilingKey` |
| 空域 | ACLNN 要求 B,S,H>0 | 防止 rows=0 与 `1/H` | 全部 | ACLNN 161002 | `CheckShape` |
| UB 上界 | 无显式条件 | 固定分配 184320 B；Host 读到 UB 大小但未比较 | 全部 | 固定仓无 fallback | `InitV2`; `TilingPrepareTiling` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 生成的结构最终布局宏实现来自仓外 `tiling/tiling_api.h`，因此固定仓能证明字段顺序和源码类型，不能证明 `sizeof/alignof/offsetof` 数值；不得把自然对齐推测写成 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| AdaLayerNormTilingData | batchSize | int64/64 | 第1字段；offset不可见 | batch | `FillTilingData=batchSize` | `ParseTilingData` | B乘积已由 ACLNN保证正 |
| 同上 | seqLen | int64/64 | 第2字段；offset不可见 | 行/B | `x[-2]` | scaleOffset、batch映射 | >0 |
| 同上 | hiddenDim | int64/64 | 第3字段；offset不可见 | 元素 | `x[-1]` | 所有循环/地址 | >0 |
| 同上 | epsilon | float/32 | 第4字段；offset不可见 | 数值 | attr float | rstd计算 | double 已窄化 |
| 同上 | aveFactor | float/32 | 第5字段；offset不可见 | 1/元素 | `float(1.0/H)` | 均值/方差缩放 | FP32 |
| 同上 | hasWeight | int32/32 | 第6字段；offset不可见 | bool flag | optional非空 | 条件 Mul | 0/1 |
| 同上 | hasBias | int32/32 | 第7字段；offset不可见 | bool flag | optional非空 | 条件 Add | 0/1 |
| 同上 | hasSmooth | int32/32 | 第8字段；offset不可见 | bool flag | V2固定0 | Quant分支；V2不读 | inactive=0 |
| 同上 | singleCoreNum | int64/64 | 第9字段；offset不可见 | 行/core | `rows/coreNum` | rowStart/rowEnd | 可为0 |
| 同上 | tailNum | int64/64 | 第10字段；offset不可见 | core数 | `rows%coreNum` | 前 tail cores 加1行 | `<coreNum` |
| 同上 | sliceSize | int64/64 | 第11字段；offset不可见 | 元素 | `SplitCore` | Fast/Slice选择与 H 循环 | 32对齐仅切片时 |
| 同上 | rowNum | int64/64 | 第12字段；offset不可见 | 行/批 | H≤3072时 `6144/Hc32`，否则1 | FastProcess | ≥1 |
| 同上 | tmpBufferSize | int64/64 | 第13字段；offset不可见 | byte | A2/A3不调用 `DoLayerNormTiling`，保持宏默认值 | A2/A3 body不读 | inactive |
| 同上 | layerNormTiling | `LayerNormSeparateTiling` | 第14字段；定义在仓外头，大小/offset不可见 | API POD | A2/A3不生产 | A2/A3 body不读 | inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key1 | `ada_layer_norm_tiling.cpp::Init/GetTilingKey/SplitCore` | `AdaLayerNormTilingData` | `ada_layer_norm_v2.cpp::ada_layer_norm_v2` | `AdaLayerNormND<X,X,12>`；base/base_v1/util | OpDef 910B/910_93；对应 binary config；V2 CMake 依赖 ada_layer_norm |
| R-002 | key2 | 同上 | 同上 | 同上 | `AdaLayerNormND<X,float,12>`；base/base_v1/util | 同上 |

## Kernel 实现

### 公共执行流程

entry 将 5 路输入、3 路输出写入 `GmAddr`，解析同一 POD，按 key 实例化 `AdaLayerNormND`，调用 `InitV2→Process`。`ParseTilingData` 把 core 映射到连续行域；Process 空域即返回，随后初始化 LocalTensor 别名和 5 个事件。Fast 路先加载 weight/bias；按行批加载 x、scale、shift，在 FP32 中算均值/方差、可选 weight/bias、再做自适应 Mul/Add，最后 cast 并写 out/mean/rstd。Slice 路先对整行统计，再按 slice 重读 x/scale/shift/weight/bias 并输出。所有路径是单 Kernel，无跨节点同步。

### Route R-001：同 dtype weight/bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_ada_layer_norm_v2.cpp` | `CheckDtypeValid` | weight/bias 可为 x dtype；若都存在需相同 | 非 FP32 可达 Host |
| Tiling | `ada_layer_norm_tiling.cpp` | `Init/GetTilingKey` | 任一存在参数 dtype!=FP32→`isWeightFloat=false`；`isRegBase=false` | key1 |
| Router | `ada_layer_norm_v2.cpp` | `ada_layer_norm_v2` | key1 且 X=half/bfloat16 | `AdaLayerNormND<X,X,12>` |

#### Device 数据流

##### 函数级调用链

```text
AdaLayerNormV2 launcher -> ada_layer_norm_v2(..., tiling)
  -> AdaLayerNormND<X,X,12>::InitV2
  -> Process
     -> FastProcess -> MultiCalMeanRstd|SingleLayerNorm -> Adaption -> BaseCopyOut -> CopyMeanRstdOut
     -> or SliceProcess -> ComputeMean/ComputeVar -> BaseSliceCompute -> CopyMeanRstdOut
```

##### 源码等价伪码

```text
rowStart = q*core + min(core, rem); rowEnd = rowStart + q + (core<rem)
if H <= 6144:
  preload optional weight/bias as X then cast to fp32
  for rows in groups of rowNum:
    reuse scale/shift when batch interval unchanged; otherwise DMA and add 1 to scale
    DMA x; cast X->fp32
    if group rows>1: vector per-row mean/variance via ReduceSum+broadcast
    else: scalar mean/rstd after vector reductions
    optional fp32 weight Mul; optional fp32 bias Add
    out = norm*scalePlusOne + shift
    cast fp32->X(CAST_RINT); DMA valid H; batch mean/rstd up to 1024 rows
else:
  for each row: scan x to accumulate mean; rescan to accumulate variance; rstd=1/sqrt(var+eps)
  for h in [0,H) step sliceSize: DMA/cast inputs, normalize, weight/bias, scalePlusOne, shift, cast and DMA tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/GM | `row*H+h` | B*S*H | X/ND | caller | CopyInX/CopyInAndCast | caller | H有效元素 |
| scale/shift GM | `(row/S)*H+h` | B*H each | X/ND | caller | CopyInScaleShift | caller | 同 batch 行复用 |
| tmpBuf | base0 | `7*6144*4=172032` B | fp32别名域 | InitV2 | 全部计算 | kernel | 固定 TBuf，多逻辑视图重叠 |
| reduceBuf | base独立 | `1024*4=4096` B | fp32 | reductions | scalar/broadcast | kernel | 批次上限1024 |
| meanRstdBuf | base独立 | `2*1024*4=8192` B | fp32 | reductions | mean/rstd DMA | kernel | 前后各1024 float |
| out/GM | `row*H+h` | B*S*H | X/ND | BaseCopyOut | user/ViewCopy | executor | 仅有效H |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  tmp["tmpBuf · 172032 B · x/weight/bias/scale/shift/mean/y/tmp aliases"]:2
  reduce["reduceBuf · 4096 B"]:1
  stats["meanRstdBuf · 8192 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 7*6144*4 + 1024*4 + 2*1024*4 = 184320 B
UB_total <= platform UB：Host 未实现该比较，固定仓不能证明显式 guard
dynamic_ub = N/A
workspace_total(node) = 16*1024*1024 B；V2 Device 不消费
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf 0.. | x/各参数/y/tmp 视图 | Fast 或 Slice 各阶段 | 下一行/切片 | PipeBarrier 与 hard event | Slice 的 `tmpFloat=xFloat[16384]` 与参数区重叠，只在统计阶段使用；参数随后才加载 |
| scaleFloat/shiftFloat | CopyInScaleShift | Adaption | batch区间变化 | V_MTE2；缓存区间比较 | batchStart/batchEnd 相同才复用 |
| mean/rstd halves | reductions | CopyMeanRstdOut | 每1024行或核尾 | MTE3_S | batchCount≤1024 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | DataCopyPad + MTE2_V | DMA | vector cast/compute | x/params | FP16/BF16 CAST_NONE升FP32；有效 len |
| vector链 | PipeBarrier V | 每个算术 primitive | 下一 primitive | 重叠 LocalTensor | mean/var/normalize 全 FP32 |
| scalar读 | V_S event | ReduceSum | GetValue | reduce首元素 | rstd 用 scalar sqrt |
| UB→GM | V_MTE3/MTE3_S | cast/vector | DataCopyPad/标量复用 | out/stats | FP32→X CAST_RINT；只写H或batchCount |
| tail | `Min`/blockCount/rightPadding | Host slice/core | Device loops | H与行尾 | 前 tailNum cores 多1行；最后slice取Min |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp` | `SplitCore/GetTilingKey/FillTilingData` | 行分核、H路径、key1与POD |
| Kernel entry/router | `norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2.cpp` | `ada_layer_norm_v2` key1 | X/X模板映射，AIV_ONLY |
| Device body/helper | `norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h` | `AdaLayerNormND::Process/FastProcess/SliceProcess` | 路径、地址、UB和批处理 |
| Device body/helper | `norm/ada_layer_norm/op_kernel/ada_layer_norm_base_v1.h` | `MultiCalMeanRstd/SingleLayerNorm/BaseSliceCompute` | FP32计算、同步、tail与输出 |

### Route R-002：FP32 或缺省 weight/bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_ada_layer_norm_v2.cpp` | `CheckDtypeValid` | V2 允许 weight/bias=FP32；可选参数可空 | Host可达 |
| Tiling | `ada_layer_norm_tiling.cpp` | `Init/GetTilingKey` | 所有存在参数均FP32，或均不存在→`isWeightFloat=true` | key2 |
| Router | `ada_layer_norm_v2.cpp` | `ada_layer_norm_v2` | key2，X=FP32/FP16/BF16 | `AdaLayerNormND<X,float,12>` |

#### Device 数据流

##### 函数级调用链

```text
AdaLayerNormV2 launcher -> ada_layer_norm_v2 key2
  -> AdaLayerNormND<X,float,12>::InitV2 -> Process
  -> 与 R-001 相同 Fast/Slice 图；差异仅 weight/bias GM 按 float 读取，缺省时由 hasWeight/hasBias 跳过
```

##### 源码等价伪码

```text
按 R-001 的 core/H 分支；CopyInOtherData:
  if hasWeight: DMA float weight directly into weightFloat
  if hasBias: DMA float bias directly into biasFloat
其余 mean/var/normalize/adaption/cast/tail 顺序与 R-001 相同。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weight/bias GM | h | H each when present | FP32/ND | caller | CopyInOtherData | caller | 缺省不读 |
| tmpBuf | 同 R-001 | 172032 B | fp32别名域 | InitV2 | body | kernel | 固定 |
| reduce+stats | 独立 | 4096+8192 B | fp32 | reductions | output | kernel | 批≤1024 |
| x/scale/shift/out | 同 R-001 | 逻辑张量 | X/ND | caller/body | body/user | caller/executor | H tail |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  tmp["tmpBuf · 172032 B · fp32 views"]:2
  reduce["reduceBuf · 4096 B"]:1
  stats["meanRstdBuf · 8192 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 184320 B
dynamic_ub = N/A
workspace_total(node) = 16 MiB（Device不读）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpBuf视图 | 同 R-001 | 同 R-001 | 下一批/切片 | hard event/PipeBarrier | key2 不改变别名关系 |
| scale/shift缓存 | CopyInScaleShift | Adaption | batch区间变化 | V_MTE2 | 区间相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| weight/bias | DataCopyPad+MTE2_V | GM FP32 | Mul/Add | 参数区 | 不需 dtype cast |
| x/统计/输出 | 与 R-001 相同 | DMA/vector | vector/DMA | x/out/stats | 计算FP32，输出按X CAST_RINT |
| 缺省参数 | `hasWeight/hasBias` | POD | 分支 | 不存在GM | 跳过对应 Mul/Add，不解引用 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp` | `Init/GetTilingKey` | FP32/缺省→key2 |
| Kernel entry/router | `norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2.cpp` | `AdaLayerNormND<X,float,12>` | key2完整dtype实例 |
| Device body/helper | `norm/ada_layer_norm/op_kernel/ada_layer_norm_base_v1.h` | `CopyInOtherData` | FP32 weight/bias读取与条件跳过 |

## 算子特定优化

### Trick T-001：按 batch 区间复用 scale/shift

#### 触发与源码依据

- 适用 Route：R-001、R-002 的 FastProcess。
- 触发条件：相邻工作批的 `batchStart`、`batchEnd` 同时与上批相等。
- Golden 来源：[`norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h:319-325`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h#L319-L325)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bool scaleShiftReuse = (lastBatchStart == range.batchStart && lastBatchEnd == range.batchEnd);
if (!scaleShiftReuse) {
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    CopyInScaleShift(range.batchStart * hiddenDim, range.batchEnd - range.batchStart, hiddenDim);
}
```

#### 机制与设计原因

scale/shift 只随 B 变化而不随 S 行变化。缓存区间不变时跳过两路 GM→UB 搬运及 `(scale+1)` 预处理；V_MTE2 event 只在确需覆盖缓存前保证前次 vector 消费完成。判断同时比较区间两端，避免跨 B 边界错误复用。

#### 迁移边界

- 适用：参数沿若干行广播，连续 tile 仍落在同一广播区间的 AIV 算子。
- 前提：缓存容量覆盖 `[batchStart,batchEnd)`，行到 batch 的映射与 `row/seqLen` 一致。
- 失效：区间任一端改变时必须重新 CopyIn；SliceProcess 每片显式读取，不使用此缓存判定。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：同一 TBuf 在统计与逐元素阶段错峰别名

#### 触发与源码依据

- 适用 Route：R-001、R-002，尤其 SliceProcess。
- 触发条件：固定 `tmpBuf` 初始化后按不同阶段建立 x、参数、临时值和 y 视图。
- Golden 来源：[`norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h:221-243`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h#L221-L243)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
xFloat = tmpBuf.Get<float>();
weightFloat = xFloat[DATA_COUNT];
biasFloat = weightFloat[DATA_COUNT];
scaleFloat = biasFloat[DATA_COUNT];
shiftFloat = scaleFloat[DATA_COUNT];
meanFloat = shiftFloat[DATA_COUNT];
tmpFloat = xFloat[MAX_X_SIZE];
reduceFloat = reduceBuf.Get<float>();
```

#### 机制与设计原因

`tmpFloat` 从第 16384 个 float 起，与 Fast 路的参数逻辑视图区间发生物理重叠；Slice 路先完成 mean/var 统计，再加载 weight/bias/scale/shift 并进入归一化，因此两个生命周期错峰，避免额外统计 TBuf。Fast 路不用 `tmpFloat`，直接使用参数和 y 视图。该复用把 V2 固定 UB 保持在 184320 B。

#### 迁移边界

- 适用：阶段严格串行、统计临时区在参数计算前失效的 Vector pipeline。
- 前提：保持 `Process` 的 Fast/Slice 分支和调用顺序；不得并发预取会覆盖重叠区的数据。
- 失效：若统计与参数计算交叠，必须拆分物理 buffer 或增加可证明的同步/容量。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 同 dtype参数 | R-002 FP32/缺省参数 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 任一存在参数非FP32 | 参数均FP32或不存在 | `isWeightFloat` |
| 模板与 dtype | `ND<X,X,12>` | `ND<X,float,12>` | GM参数真实dtype |
| 循环与分核 | 相同三档H子路径 | 相同 | `SplitCore` |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 TBuf；无dynamic UB/通信 | 相同 | 固定布局 |
| 内存总占用与复用 | 184320 B；参数需升FP32 | 184320 B；FP32直读 | 模板dtype差异 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、Cast、ReduceSum、Mul/Add | 同，参数cast可省 | AIV Vector |
| 同步、精度与 tail | FP32计算，CAST_RINT输出 | 同 | 输出dtype由X决定 |
| fallback/失败边界 | 未知key无输出 | 未知key无输出 | router无else错误上报 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2公开支持 | V2公开支持 | V2公开支持，见独立 `_950` 卡 |
| Host/Launcher | `isRegBase=false`，key1/2 | 同 A2 | `isRegBase=true`，key11/12/21/22 |
| Kernel/资源 | `AdaLayerNormND`，184320 B固定UB | 同 A2；binary为 ascend910_93 | arch35 FullLoad/Welford 与 MicroAPI |

## 限制与失败边界

- ACLNN 拒绝 x 维数不在 2–8、B/S/H 任一非正、dtype/shape 不匹配；epsilon 源码未限制取值，负值导致的数值语义不在固定仓保证范围。
- `uint32_t(len*sizeof(dtype))` 用于 DMA 长度，Host 未见对应上界或窄化检查；超大 H 的安全范围固定仓未闭合为显式 API 限制。
- Host 读取 UB 容量但不参与 selector；固定 184320 B 是否适配每个具体 A2/A3 SKU 由平台构建假设承担，固定源码没有失败分支。
- POD 宏和 `LayerNormSeparateTiling` 的真实 ABI 实现在固定仓外，故数值 `sizeof/alignof/offsetof` 缺失；A2/A3 路不消费嵌套 tiling。
- router 对未知 key 或编译 dtype 不匹配没有显式错误/兜底 body；正常 Host selector 只产生 key1/2。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md` | 产品支持表/函数原型 | V2公开性、A2/A3支持、接口约束 |
| 入口实现 | `norm/ada_layer_norm_v2/op_host/op_api/aclnn_ada_layer_norm_v2.cpp` | `CheckParams/AdaLayerNormV2Calculate/aclnnAdaLayerNormV2*` | 校验、Contiguous/ViewCopy、executor |
| 入口实现 | `norm/ada_layer_norm_v2/op_host/op_api/ada_layer_norm_v2.cpp` | `l0op::AdaLayerNormV2` | 输出分配与AICore launcher |
| 平台注册 | `norm/ada_layer_norm_v2/op_host/ada_layer_norm_v2_def.cpp` | `ops::AdaLayerNormV2` | 910B/910_93配置与dtype组合 |
| Host/Tiling/Launcher | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.h` | `AdaLayerNormTilingData` | POD字段顺序/类型 |
| Host/Tiling/Launcher | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp` | `AdaLayerNormTiling` | selector、分核、workspace与POD producer |
| Kernel entry/body | `norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2.cpp` | `ada_layer_norm_v2` | AIV entry、key/router |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/ada_layer_norm_base.h` | `AdaLayerNormND` | UB、core/H路径、生命周期 |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/ada_layer_norm_base_v1.h` | `AdaLayerNormND::*` | 计算、搬运、同步、tail |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/ada_layer_norm_util.h` | `CopyInAndCast/CopyOut/GmAddr` | helper和entry参数映射 |
| 构建接线 | `norm/ada_layer_norm_v2/op_host/config/ascend910b/ada_layer_norm_v2_binary.json` | `AdaLayerNormV2 op_list` | A2 dtype binary；A3同构config另有 ascend910_93 文件 |
| 构建接线 | `norm/ada_layer_norm_v2/op_host/config/ascend910_93/ada_layer_norm_v2_binary.json` | `AdaLayerNormV2 op_list` | A3 FP32/FP16/BF16 ND binary配置 |
| 构建接线 | `norm/ada_layer_norm_v2/op_host/CMakeLists.txt` | `add_modules_sources` | V2 Host依赖公共 ada_layer_norm tiling/body |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 公共14字段POD与V2 opCode=12 | 字段顺序/位宽；A2/A3嵌套字段inactive | `AdaLayerNormTilingData`; `FillTilingData` |
| 调用顺序 | InitV2→Process→Fast/Slice | 统计先于重叠参数区覆盖；输出批≤1024 | `AdaLayerNormND::Process` |
| workspace/内存 | 固定184320 B UB；16MiB节点workspace | tmpBuf别名offset、GM地址公式、workspace未读 | `InitV2/InitTensor` |
| 同步 | MTE2/V/S/MTE3事件与PipeBarrier | 不得在消费前覆盖scale/shift或y | `InitEventId`; helpers |
| 精度与尾块 | FP32统计/计算，CAST_RINT回X | `aveFactor=1/H`、core/tile tail与有效H写回 | base_v1计算方法 |
| 失败边界 | ACLNN先校验；launcher失败阻断 | 未知key不得静默映射到任一模板 | `CheckParams`; entry router |
