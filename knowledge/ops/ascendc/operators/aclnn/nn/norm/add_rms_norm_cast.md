---
type: Operator
title: "AddRmsNormCast A2/A3 ACLNN 算子知识"
description: "说明 aclnnAddRmsNormCast 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/docs/aclnnAddRmsNormCast.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_host/add_rms_norm_cast_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_split_d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_single_n.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/rms_norm/op_kernel/rms_norm_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_host/config/ascend910b/add_rms_norm_cast_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:46:01Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# AddRmsNormCast A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAddRmsNormCastGetWorkspaceSize` + `aclnnAddRmsNormCast`，公开两段式 ACLNN；固定仓未暴露版本后缀，提取该公开入口的当前实现。 |
| 计算语义 | `x = cast_T(cast_fp32(x1)+cast_fp32(x2))`（FP16 路线先在 FP16 相加），`rstd=1/sqrt(mean(x_fp32^2)+epsilon)`，`y2=cast_T(x_fp32*rstd*gamma)`，`y1=cast_fp32(y2)`。 |
| 输入输出 | `x1/x2/gamma: ND, FP16/BF16`；`y1/rstd: FP32`；`y2/x: FP16/BF16`。归约维 `N=numel(gamma)`，外层行数 `M=numel(x1)/N`。 |
| 执行类型 | 单 Kernel：`add_rms_norm_cast`，AIV_VECTOR。 |
| Kernel 路线 | R-001 Normal（key 10/30）；R-002 Split-D（11/31）；R-003 Single-N（13/33）。源码中的 key 14 body 对 Host 不可达，不发布为 Route。 |
| 核心资源 | GM；AIV UB `TQue/TBuf`；固定 256 B 用户 workspace 未被三条可达 body 读取；队列/event/pipe barrier。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→OpDef/自动生成 ACLNN bridge 边界→`Tiling4AddRmsNormCast`→POD→entry router→三条可达 body/helper→binary config/CMake。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

把 `x1` 的末尾 `rank(gamma)` 维展平为长度 `N`，其余维展平为 `M` 行。每行先求和 `x=x1+x2`，再计算 `rstd=1/sqrt(sum(x_fp32^2)/N+epsilon)`，最后计算 `y2=x_fp32*rstd*gamma` 并降到输入 dtype，`y1` 是已降精度 `y2` 再升为 FP32 的值。这个最后顺序很重要：源码不是直接把未降精度乘积写入 `y1`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | ND，1–8 维 | FP16/BF16 | 必选；与 `x2/y1/y2/x` 同 shape。公开文档称支持空 Tensor，但本平台 Host 在 `CheckShapeInfo` 拒绝任一零维长度。 |
| `x2` | 同 `x1` | 同 `x1` | 必选，加数。 |
| `gamma` | 等于 `x1` 的末尾若干维 | 同 `x1` | 必选；其元素数为 `N`。 |
| `epsilon` | 标量属性 | double 入口、Host 取 float | 建议/默认 `1e-6`；Tiling 明确要求 `epsilon >= 0`。 |
| `y1Out` | 同 `x1`，连续 ND | FP32 | 降精度 `y2` 的 FP32 表示。 |
| `y2Out` | 同 `x1`，连续 ND | 同 `x1` | RMSNorm 输出。 |
| `rstdOut` | `x1` 去除 gamma 对应末尾维 | FP32 | 每行一个倒均方根。 |
| `xOut` | 同 `x1`，连续 ND | 同 `x1` | 融合 Add 的输出。 |

### 入口与版本

公开入口是无版本后缀的 `aclnnAddRmsNormCast`。两段式第一段完成参数校验、连续化输入并建立 executor，第二段在调用者给定 stream 上执行。固定仓中 ACLNN bridge 由构建框架按 `ACLNNTYPE aclnn` 生成，算子拥有的 OpDef、Tiling、POD、router 和 body 均可见。README/API 产品表明确支持 Atlas A2 与 A3，OpDef 注册 `ascend910b`、`ascend910_93`，两者 binary config 的 FP16/BF16 实例相同，因此合并为本 A2/A3 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 第一段检查空指针、dtype/shape/layout；OpDef 对三个输入声明 `AutoContiguous`，输出不支持非连续。`Tiling4AddRmsNormCast` 继续核对 1–8 维、`x1/x2/y1/y2/x` 完全同形、`gamma` 等于 `x1` 尾维、`rstd` 对应前缀维，并拒绝空 shape 与负 `epsilon`。任一步失败返回 `GRAPH_FAILED`，映射到 ACLNN 参数或 tiling 错误。

### 参数、Workspace 与生命周期

`numCol=N=numel(gamma)`；`numRow=M=prod(x1[0:rank(x1)-rank(gamma)])`；`avgFactor=1/N`。`blockFactor=ceil(M/AIVCoreNum)`，`blockDim=ceil(M/blockFactor)`。Tiling 始终报告 `256 + 16 MiB` workspace，其中 256 B 是用户区、16 MiB 是系统区；三条 Host 可达 body 的 `workspace` 参数均未读取。executor 和 workspace 由 ACLNN 第一段产生/计量，调用者保持至第二段 stream launch 完成。

### 启动图与失败传播

单节点图：ACLNN executor 在给定 stream 上启动 `add_rms_norm_cast(x1,x2,gamma,y1,y2,rstd,x,workspace,tiling)`，`blockDim=useCoreNum`，POD 为 `AddRMSNormCastTilingData`。entry 按 key 构造一个模板对象并同步执行 `Init→Process`。无 Kernel 间 workspace 边；同一 stream 保证本节点对四个输出的提交顺序。未知 key 或在被编译 guard 排除的平台命中 BF16 key 时 entry 无 body、无显式错误回写，因此 Host/key 与 binary config 的一致性是必要条件。

## Route 与参数设计

### 分核与资源计算

每核覆盖 `rowWork=blockFactor`，最后核为 `M-(blockDim-1)*blockFactor`。常规/Single-N 的 `ubFactor` 为 FP16/BF16 8704 元素；Split-D 先取 12096，再令 `colTileNum=ceil(N/12096)`、`ubFactor=ceil(N/(colTileNum*16))*16`。行组固定 `rowFactor=64`。所有形状乘积先落入 `uint32_t` POD，文档仅约束单维不超过 INT32_MAX，固定 Host 未对 `numRow/numCol` 乘积窄化做完整溢出检查。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-002 Split-D | FP16 `11`；BF16 `31` | 初始 `ubFactor=8704`，若 `N>ubFactor` | 否则检查 Single-N |
| 2 | R-003 Single-N | FP16 `13`；BF16 `33` | `N<=8704 && blockFactor==1 && socVersion!=ASCEND310P` | 否则 R-001 |
| 3 | R-001 Normal | FP16 `10`；BF16 `30` | 其余；FP16 且 `N` 已 16 元素对齐时显式落 Normal，其它情况保留初始 Normal | 无 fallback；shape/epsilon 失败即终止 |

```text
if N > 8704: Split-D
else if ceil(M/AIVCoreNum) == 1 and SoC != ASCEND310P: Single-N
else: Normal
key = dtypeKey(FP16=1, BF16=3) * 10 + mode(0/1/3)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 Normal | 10/30 | `N<=8704` 且非 Single-N | 单 AIV Kernel | `KernelAddRmsNormCast<half/bfloat16_t>` | 每行一次全载；Queue + FP32 TBuf；V/MTE event | 每核尾行、每 Vector mask 与 `DataCopyCustom` 尾搬运 |
| R-002 Split-D | 11/31 | `N>8704` | 单 AIV Kernel | `KernelAddRmsNormCastSplitD<T>` | 两遍扫 N；第一遍累加 rstd，第二遍写 y；无用户 workspace | 列尾 `col_tail`、行尾 `row_tail` |
| R-003 Single-N | 13/33 | `N<=8704 && blockFactor==1` | 单 AIV Kernel | `KernelAddRmsNormCastSingleN<T>` | 单个 195584 B `TBuf` 手工切片；显式 event | 每核一行；API tail helper |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 全载上限 | `N<=8704` | Normal/Single-N 的 UB 容量按固定 `ubFactor` 分配 | R-001/R-003 | R-002 |
| 单行并核 | `blockFactor==1 && soc!=ASCEND310P` | 一核一行可用 195584 B unitBuf 消除 Queue 编排 | R-003 | R-001 |
| BF16 编译 guard | 非 NPU arch 3003/3113 | router 对 BF16 实例有预处理保护 | 所有 BF16 key | guard 不成立时无 body；A2/A3 binary 接线必须与编译宏一致 |
| DMA 尾块 | `count % (32/sizeof(T))` | `DataCopyCustom` 对 GM→UB 向上对齐，对 UB→GM 用末 32 B 重写保护有效尾 | 全部 | helper 内 tail 路径 |
| 空 Tensor | Host 任一 `x1` 维为 0 | `Process` 的 `i_o_max-1` 不允许零行 | 全部 | `GRAPH_FAILED` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AddRMSNormCastTilingData` | `num_row` | uint32/32 | 声明序第 1 字段；宏生成的精确 offsetof 固定仓未显式输出 | 行 | `numRow` | 各 body 分核/行循环 | 乘积窄化未完整校验 |
| 同上 | `num_col` | uint32/32 | 第 2 字段 | 元素/行 | `numCol` | 归约长度与 GM stride | 非零 |
| 同上 | `block_factor` | uint32/32 | 第 3 字段 | 行/核 | `ceil(M/core)` | core base、末核行数 | 至少 1 |
| 同上 | `row_factor` | uint32/32 | 第 4 字段 | 行/批 | 常量 64 | 外层行 tile | 尾批缩短 |
| 同上 | `ub_factor` | uint32/32 | 第 5 字段 | 元素 | Route 公式 | UB 容量、列 tile | Route 相关 |
| 同上 | `epsilon` | float/32 | 第 6 字段 | 数值 | attr `*epsilon` | rstd | Host 要求非负 |
| 同上 | `avg_factor` | float/32 | 第 7 字段 | 1/元素 | `1.0/N` | POD 字段未被 body 使用；body 重算 `1/N` | inactive consumer |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10/30 | `add_rms_norm_cast_tiling.cpp::Tiling4AddRmsNormCast` | `AddRMSNormCastTilingData` | `add_rms_norm_cast.cpp::add_rms_norm_cast` | `add_rms_norm_cast.h::KernelAddRmsNormCast`；`rms_norm_base.h` helpers | `ascend910b/...binary.json`、`op_host/CMakeLists.txt` |
| R-002 | 11/31 | 同上 | 同上 | 同上 | `add_rms_norm_cast_split_d.h::KernelAddRmsNormCastSplitD`；base helpers | 同上 |
| R-003 | 13/33 | 同上 | 同上 | 同上 | `add_rms_norm_cast_single_n.h::KernelAddRmsNormCastSingleN`；base helpers | 同上 |

## Kernel 实现

### 公共执行流程

每个 block 绑定连续的行区间。所有 Route 都先得到输入和 gamma，在 FP32 中计算平方和/rstd，并输出 `x/y1/y2/rstd`；FP16 的 Add 在部分路线按 FP16 执行后升精度，BF16 先升 FP32 相加再 `CAST_RINT` 回 BF16 作为 `xOut`。`DataCopyCustom` 在 A2/A3 使用 `DataCopyPad`，以字节 `blockLen=count*sizeof(T)` 处理尾块。

### Route R-001：Normal 全行加载

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | `N<=8704`，未命中 Single-N | mode 0，key 10/30 |
| Router | `op_kernel/add_rms_norm_cast.cpp` | `add_rms_norm_cast` | `TILING_KEY_IS(10/30)` | `KernelAddRmsNormCast<T>` |

#### Device 数据流

##### 函数级调用链

```text
add_rms_norm_cast -> KernelAddRmsNormCast::Init -> Process
  -> SubProcess(row) -> CopyIn(x1,x2), write xOut
  -> Computefp16/Computebf16 -> rstd, y1/y2
  -> CopyOutY
```

##### 源码等价伪码

```text
for each local row:
  load x1[0:N], x2[0:N]
  FP16: xT = x1 + x2; xfp32 = cast(xT)
  BF16: xfp32 = cast(x1)+cast(x2); xT = cast_rint(xfp32)
  store xT -> xOut
  rstd = 1/sqrt(reduce_sum(xfp32*xfp32/N)+epsilon); store rstd
  yT = cast_T(xfp32*rstd*gamma); y1=cast_fp32(yT); store y1,y2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM rows | `blockIdx*blockFactor*N + row*N` | `rowWork*N` | T/ND | ACLNN | Kernel | 调用者 | 连续 |
| `inQueueX` | UB | `1*ubFactor*sizeof(T)` | T | CopyIn x1 | Add/Cast | Queue，一行 | depth 1 |
| `outQueueY` | UB | `1*ubFactor*sizeof(T)` | T | Compute | CopyOut y2 | Queue，一行 | depth 1 |
| `xFp32Buf/tmpBuf/sqxBuf` | UB 独立 TBuf | 各 `ubFactor*4` | FP32 | Cast/compute | reduce/y | Kernel | 阶段复用 |
| `tmpBuf2` | UB | `3*ubFactor*sizeof(T)` | T/重解释 FP32 | Compute | gamma/y1 | Kernel | gamma 位于 `2*ubFactor` T 元素偏移 |
| `reduceFp32Buf` | UB | `64*4` | FP32 | reduce helper | ReduceSum | Kernel | 固定 256 B |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueueX · depth=1 · ubFactor*sizeof(T)"]:1
  outQ["outQueueY · depth=1 · ubFactor*sizeof(T)"]:1
  fp["xFp32/tmp/sqx · 3*ubFactor*4"]:1
  aux["tmpBuf2 + reduce · 3*ubFactor*sizeof(T)+256"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*ubFactor*sizeof(T) + 3*ubFactor*4 + 3*ubFactor*sizeof(T) + 256
dynamic_ub = N/A
workspace_total = 256 B user + 16 MiB system；body 不读 user slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueueX` | CopyIn/Add | Cast/GM xOut | 下一行 | EnQue/DeQue/FreeTensor + MTE3→MTE2 | xOut 搬运完成后复用 |
| `tmpBuf2` | gamma CopyIn、y1 cast | Compute/GM | 下一行组 | MTE2/V 与 V/MTE3 event | gamma 仅在 `inner_progress==0` 刷新 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Add→Cast | `PipeBarrier<PIPE_V>` | Vector Add | Cast | x buffer | FP16 先低精度 Add；BF16 先 FP32 Add 后 RINT |
| xOut | V_MTE3、MTE3_MTE2 | Vector | GM/下一次 MTE2 | inQueueX | `DataCopyPad` 有效 N |
| reduce | `ReduceSumCustom` | square/scale | sqrt/div | sqx/reduceBuf | FP32，tail mask |
| y | MTE2_V、MTE3_V | gamma/rstd/y | Mul/下一轮 | gamma/y1/rstd | y2 降精度后 y1 再升 FP32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | Normal key、核数、UB 参数 |
| Kernel entry/router | `op_kernel/add_rms_norm_cast.cpp` | `add_rms_norm_cast` | 10/30→模板 dtype |
| Device body/helper | `op_kernel/add_rms_norm_cast.h` | `CopyIn/Computefp16/Computebf16` | 四输出、精度和同步 |

### Route R-002：Split-D 两遍列切分

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | `N>8704` | key 11/31；计算对齐 `ubFactor` |
| Router | `op_kernel/add_rms_norm_cast.cpp` | `add_rms_norm_cast` | 11/31 | `KernelAddRmsNormCastSplitD<T>` |

#### Device 数据流

##### 函数级调用链

```text
Process -> SubProcess(row group)
  -> for col tiles: ComputeFormer -> CopyInAndAdd -> ComputeSum
  -> ComputeRstd
  -> for col tiles: ComputeLatter -> CopyInGamma/CopyInX -> ComputeY -> CopyOutY
  -> CopyOutRstd
```

##### 源码等价伪码

```text
for row_group up to 64:
  rstdAccum[:] = 0
  for col_tile: add x1+x2; store xOut; reduce tile square/N into each row accumulator
  rstd = 1/sqrt(rstdAccum+epsilon)
  for col_tile: reload xOut and gamma; y2=cast_T(x*rstd*gamma); y1=cast_fp32(y2)
  store rstd rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQueueX` | UB | `2*ubFactor*sizeof(T)` | T | x1/x2 or xOut | Add/reload | depth1 | 两半分别 x1/x2 |
| `inQueueGamma` | UB | `ubFactor*sizeof(T)` | T | GM | y | depth1 | 每列 tile |
| `outQueueY` | UB | `ubFactor*sizeof(T)` | T | xOut/y2 | GM | depth1 | x 与 y2 分阶段复用 |
| `outQueueRstd` | UB | `64*4` | FP32 | ComputeRstd | GM | depth1 | 行组 |
| FP32 TBuf | UB | `2*ubFactor*4 + 64*8*4 + 64*4` | FP32 | cast/square/reduce | rstd/y | Kernel | sumBuf 按每行 8 FP32 block |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["inQueueX · 2*ubFactor*sizeof(T)"]:1
  gq["gammaQ · ubFactor*sizeof(T)"]:1
  oq["yQ+rstdQ · ubFactor*sizeof(T)+256"]:1
  calc["xFp32+sqx+sum+reduce"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4*ubFactor*sizeof(T) + 2*ubFactor*4 + 64*8*4 + 2*64*4
dynamic_ub = N/A
workspace_total = 256 B user + 16 MiB system；body 不读 user slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `outQueueY` | Add 或 ComputeY | xOut/y2 GM copy | 下一列 tile | EnQue/DeQue/FreeTensor | 对应 GM copy 完成 |
| `xFp32Buf` | Add/Cast 或 reload Cast | square/y | 下一行/列 | PipeBarrier | 同一阶段顺序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 第一遍 | Queue + PIPE_V | Add/Cast | square/reduce | x/sqx | `col_tail` 精确 count |
| 跨遍 | GM `xOut` | 第一遍 | 第二遍 CopyIn | 融合 Add 结果 | 同一 Kernel 顺序，MTE 事件 |
| 标量 rstd | V_S/S_V | UB vector | scalar `GetValue`/Muls | 行 rstd | FP32 |
| 输出 | V_MTE3/MTE3_V | Vector | GM/复用 | y1/y2 | BF16 RINT，FP16 CAST_NONE |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | Split-D tile 公式 |
| Kernel entry/router | `op_kernel/add_rms_norm_cast.cpp` | key 11/31 | dtype 实例 |
| Device body/helper | `op_kernel/add_rms_norm_cast_split_d.h` | `SubProcess/ComputeFormer/ComputeLatter` | 两遍列切分与尾块 |

### Route R-003：Single-N 单行手工 UB

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | `N<=8704 && blockFactor==1 && soc!=ASCEND310P` | key 13/33 |
| Router | `op_kernel/add_rms_norm_cast.cpp` | `add_rms_norm_cast` | 13/33 | `KernelAddRmsNormCastSingleN<T>` |

#### Device 数据流

##### 函数级调用链

```text
Process -> ProcessFp16/ProcessBf16
  -> load x1/x2 -> add/cast -> write xOut
  -> square/scale/reduce -> rstd
  -> reuse x2 slice for gamma -> normalize/multiply -> write y2/y1
```

##### 源码等价伪码

```text
unitBuf[0:195584] split into x1T, x2T/gammaT, xFp32, sqx, tmp
load x1,x2; compute x and start gamma load into dead x2 slice
store x; compute FP32 rstd; normalize; wait gamma; multiply
store y2; store y1 from the route's post-multiply FP32 representation
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `unitBuf` | UB base | 195584 B | byte/T/FP32 views | Init | whole route | Kernel | 单一物理 TBuf |
| `x1Local` | T offset 0 | `ubFactor*T` | T | x1 then y2 | compute/output | route | in-place |
| `x2Local` | T offset `ubFactor` | `ubFactor*T` | T | x2 then gamma | Add then y | route | event 保证覆盖 |
| FP32 regions | float offsets `ubFactor,2*ubFactor,3*ubFactor` | 各约 `ubFactor*4` | FP32 | cast/square/reduce | rstd/y | route | 阶段覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  x1["x1/y2 slice · ubFactor*sizeof(T)"]:1
  x2["x2/gamma slice · ubFactor*sizeof(T)"]:1
  fp["xFp32 · ubFactor*4"]:1
  aux["sqx/tmp · remaining unitBuf"]:1
```

##### 字节账与复用顺序

```text
UB_total = 195584 B
dynamic_ub = N/A
workspace_total = 256 B user + 16 MiB system；body 不读 user slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x2/gamma slice | x2 MTE2 | Add | gamma MTE2 | V_MTE2 + MTE2_V | Add 完成才覆盖 |
| x1/y2 slice | Add | xOut MTE3/Cast | normalized y2 | MTE3_V | xOut 已提交 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 双输入 | 两个 MTE2_V event | GM loads | Add/Cast | x1/x2 | N 有效元素 |
| gamma 覆盖 | V_MTE2/MTE2_V | Add/GM gamma | Mul | x2 slice | 禁止提前覆盖 |
| xOut | V_MTE3/MTE3_V | Add | slice reuse | x1 slice | N 尾块 helper |
| rstd/y | V_S/S_V + PIPE_V | reduce | scalar normalize | FP32 buffers | BF16 RINT；FP16 CAST_NONE |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | 单行谓词 |
| Kernel entry/router | `op_kernel/add_rms_norm_cast.cpp` | key 13/33 | Single-N 实例 |
| Device body/helper | `op_kernel/add_rms_norm_cast_single_n.h` | `ProcessFp16/ProcessBf16` | 195584 B 手工切片与事件协议 |

## 算子特定优化

### Trick T-001：Split-D 用 xOut 作为两遍归约交接

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`N>8704`
- Golden 来源：[`op_kernel/add_rms_norm_cast_split_d.h:148-153`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_split_d.h#L148-L153)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// copy out to workspace && x_out
outQueueY.EnQue(xLocal);
auto x_out = outQueueY.DeQue<T>();
DataCopyCustom<T>(xGm[i_idx * numCol + j_idx * ubFactor], x_out, num);
outQueueY.FreeTensor(x_out);
```

#### 机制与设计原因

第一遍必须产生公开输出 `xOut`，第二遍直接从该输出重载归一化输入，因此不再申请额外的大型用户 workspace。代价是大 N 路线对 x 有一次 UB→GM→UB 往返；这是输出物化兼作阶段交接，而非隐藏临时区。

#### 迁移边界

- 适用：已有公开中间输出且 UB 无法容纳完整归约维的融合算子。
- 前提：同一 Kernel 内 MTE 顺序闭合；输出 dtype/舍入正是后续语义所需。
- 失效：若后续必须消费未降精度中间值，不能复用低精度 `xOut`。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Single-N 覆盖 x2 UB 子区预取 gamma

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：一核一行且 `N<=8704`
- Golden 来源：[`op_kernel/add_rms_norm_cast_single_n.h:89-104`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_single_n.h#L89-L104)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
SetFlag<HardEvent::V_MTE2>(eventVMTE2);
WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
DataCopyCustom<T>(x2Local, gammaGm, numCol); // gammaLocal use x2Local
event_t eventMTE2V3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
SetFlag<HardEvent::MTE2_V>(eventMTE2V3);
```

#### 机制与设计原因

Add 消费完 x2 后，gamma 直接覆盖同一 UB 子区；V→MTE2 事件防止提前覆盖，后续 MTE2→V 事件保护乘法读取。它省去独立 gamma buffer，并让 gamma 搬运与 xOut/rstd 工作交叠。

#### 迁移边界

- 适用：前一输入最后消费点明确、后续只读权重与其容量一致的单 tile 实现。
- 前提：保留两条事件边和相同 buffer offset。
- 失效：多 tile 流水或 x2 在 gamma load 后仍被访问时必须拆 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

删除 V_MTE2/MTE2_V 依赖会使 gamma 覆盖与 Add/乘法竞态；该错误后果由源码中的显式事件闭包支撑。

## 模板与实现对比

| 对比维度 | R-001 Normal | R-002 Split-D | R-003 Single-N | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | N 小、非一核一行 | N>8704，最高优先 | N 小且一核一行 | 先容量，再并核形态 |
| 模板与 dtype | FP16/BF16 | FP16/BF16 | FP16/BF16 | key 十位编码 dtype |
| 循环与分核 | 每核多行，逐行 | 行组×列 tile 两遍 | 每核恰一行 | UB 与并行度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 多 Queue/TBuf | 多 Queue/TBuf + 行累加 | 单 195584 B TBuf | 均无 dynamic UB/通信 |
| 内存总占用与复用 | 固定 ubFactor 多 buffer | tile 容量；xOut 跨遍 | 单 buffer 强复用 | 容量/搬运权衡 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + Vector | 两遍 DataCopyPad + Vector | DataCopyPad + Vector/event | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue/event；行尾 | Queue/event；行/列尾 | 手工 event；N 尾 | y1 均源于降精度 y2 |
| fallback/失败边界 | 最终普通路线 | N 降至上限即不选 | blockFactor>1 即不选 | 无跨 Kernel fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开支持，另见 `_950` 卡 |
| Host/Launcher | `ascend910b`，本卡 Tiling 决策树 | `ascend910_93`，与 A2 同实现/config 形态 | regbase Tiling，keys 100–103/199 |
| Kernel/资源 | AIV Vector，普通 AscendC API | 同 A2 | MicroAPI、NDDMA 分支与不同 workspace |

## 限制与失败边界

- 公开 ACLNN 文档称输入支持空 Tensor，但 A2/A3 `CheckShapeInfo` 明确拒绝任一零维长度；本卡按可达源码记录为空输入失败，不推测运行时补偿。
- `numRow/numCol` 和 POD 为 uint32，Host 对多维乘积的溢出/窄化没有完整显式保护。
- BF16 router 在 NPU arch 3003/3113 被编译排除；A2/A3 发布依赖 binary config 与实际编译宏保持一致。
- 未知 key 无显式错误 body；Host selector 是唯一可达 key producer。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/add_rms_norm_cast/docs/aclnnAddRmsNormCast.md` | 两段式原型 | 公开性、产品与参数 |
| 入口实现 | `norm/add_rms_norm_cast/op_host/add_rms_norm_cast_def.cpp` | `ops::AddRmsNormCast` | dtype/layout、AutoContiguous、平台 |
| Host/Tiling/Launcher | `norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling.cpp` | `Tiling4AddRmsNormCast` | 校验、Route、POD/workspace |
| Host/Tiling/Launcher | `norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling.h` | `AddRMSNormCastTilingData` | POD 声明 |
| Kernel entry/body | `norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast.cpp` | `add_rms_norm_cast` | router |
| Kernel entry/body | `norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast.h` | `KernelAddRmsNormCast` | Normal body |
| Kernel entry/body | `norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_split_d.h` | `KernelAddRmsNormCastSplitD` | Split-D body |
| Kernel entry/body | `norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_single_n.h` | `KernelAddRmsNormCastSingleN` | Single-N body |
| Kernel helper | `norm/rms_norm/op_kernel/rms_norm_base.h` | `DataCopyCustom/ReduceSumCustom` | DMA tail 与归约 |
| 构建接线 | `norm/add_rms_norm_cast/op_host/config/ascend910b/add_rms_norm_cast_binary.json` | `op_list` | FP16/BF16 binary 实例 |
| 构建接线 | `norm/add_rms_norm_cast/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN 生成接线与 rms_norm 依赖 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 7 字段 producer/consumer | 字段顺序、uint32 窄化、float epsilon | `AddRMSNormCastTilingData` |
| 调用顺序 | 单 entry `Init→Process` | key 与模板映射 | `add_rms_norm_cast` |
| workspace/内存 | 三路线不读用户 workspace | Tiling 仍报告 256 B + 系统区 | `Tiling4AddRmsNormCast` |
| 同步 | Queue/PIPE/event 协议 | Single-N 覆盖前的 V_MTE2/MTE2_V | 各 body |
| 精度与尾块 | FP32 rstd；y1 从 y2 升精度 | BF16 RINT、有效 N 的 DataCopy tail | Compute helpers |
| 失败边界 | shape/epsilon 失败；未知 key 无 body | 不把 key14 dormant body当 Route | Host/router |