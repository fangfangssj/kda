---
type: Operator
title: "LayerNormQuant 950 ACLNN 算子知识"
description: "说明 aclnnLayerNormQuant 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/docs/aclnnLayerNormQuant.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/op_api/aclnn_layer_norm_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/op_api/layer_norm_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/layer_norm_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/layer_norm_quant_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/layer_norm_quant_tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/layer_norm_quant_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/layer_norm_quant_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_kernel/layer_norm_quant_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_kernel/arch35/layer_norm_quant_regbase_normal.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_kernel/arch35/layer_norm_quant_regbase_spilt_d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_kernel/arch35/utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/config/ascend950/layer_norm_quant_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T17:58:11Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# LayerNormQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnLayerNormQuantGetWorkspaceSize` + `aclnnLayerNormQuant`，公开 ACLNN、固定仓内唯一版本 |
| 计算语义 | 尾轴 LayerNorm 后静态量化；固定 API 只接受 `quantMode=0`，Device 输出 `res`，不写 `scaleOut` |
| 输入输出 | 950 支持 ND FP16/BF16/FP32 输入族，INT8 `res`；scale/zero point 均只消费第 0 元素 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；entry 声明 `KERNEL_TYPE_AIV_ONLY` 并让 AIC core 直接返回 |
| Kernel 路线 | R-001 FP16 整载；R-002 BF16 整载；R-003 FP32 整载；R-004 FP16 分片；R-005 BF16 分片；R-006 FP32 分片 |
| 核心资源 | GM、depth=1 TQue、三个 FP32 TBuf、dtype-aware mean/variance TBuf；Kernel 不消费 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→L0→RegBase selector/Tiling/POD→AIV-only `_apt` router→六个实际 body 实例→950 binary config 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `D=x.shape[-1]`、`R=product(x.shape[:-1])`。每行计算 `m=sum(x)/D`、`v=sum((x-m)^2)/D`、`y=(x-m)/sqrt(v+epsilon)*gamma+beta`。API 强制静态模式，Device 取 `sInv=(scale[0]==0 ? 1 : 1/scale[0])`、`zp=float(offset[0])`，计算 `q=y*sInv+zp`。不论输入是 FP16、BF16 还是 FP32，q 都先以 `CAST_NONE` 降为 FP16，再夹到 `[-128,127]`；950 非 CCE220 分支以 `CAST_NONE` 转 INT8。静态模式的 `scaleOut` 无有效语义，Kernel 没有绑定或写入它。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，1–8 维；空 Tensor 可返回 | FP16/BF16/FP32 | 950 文档、OpDef、API dtype list 与 binary config 一致 |
| `gamma` | ND，API 最多 2 维，尾维 D | 与 x 相同 | 文档静态约束 `[1,D]`，可见 API 未检查第一维为 1 |
| `beta` | 与 gamma 完全同 shape | 与 x 相同 | Device 每行广播同一 gamma/beta |
| `scale` | `[1]` ND | 与 x 相同 | 0 被替换为 1 后取倒数 |
| `zeroPointsOptional` | 静态模式必选 | INT8 | 源码仅校验与 scale 的 rank 相同，body 只读 `[0]` |
| `quantMode` | 标量 | int | 固定实现仅支持 0 |
| `epsilon` | 标量 | double→float | L0 窄化为 float，默认 `1e-5` |
| `res` | 与 x 相同 | INT8 | 有效输出 |
| `scaleOut` | 静态时临时 `[1]` | FP32 | entry 参数保留但不写 |

### 入口与版本

公开文档明确支持 Ascend 950PR/950DT。OpDef 的 `ascend950` config 支持 FP32/FP16/BF16 并通过 `opFile.value=layer_norm_quant_apt` 选择 950 entry；950 binary config 列出三种 dtype，构建总表也含 `ascend950`。固定仓没有更高版本 API。`common/inc/op_host/tiling_util.h::IsRegbaseSocVersion` 将 950 的 `DAV_3510` 判为 RegBase，因而 `Tiling4LayerNormQuant` 选择 `LayerNormQuantRegTiling::DoTiling`。

## Host/Launcher 执行流程

### 参数校验与前处理

流程与 A2/A3 相同：创建 executor；检查必选指针与静态 zero point；按 950 dtype list 检查 x/gamma/beta/scale；检查 shape 关系和 `quantMode==0`；空 x 返回 workspace 0；否则依次 `Contiguous` 五个输入，调用一个 L0 `LayerNormQuant` 节点，再把临时 INT8 res `ViewCopy` 到调用方输出。静态模式不会复制临时 scaleOut。源码定义了 `CheckPlatform()`，但第一段入口没有调用它；950 可达性来自公开表、OpDef 与 binary/build 接线。

### 参数、Workspace 与生命周期

L0 分配 `res[x.shape]` 和静态 `scaleOut[1]`，launcher 参数顺序为 `(x,gamma,beta,scale,offset)->(res,scaleOut)`，属性为 `(float(epsilon),quantMode)`。Tiling 设置单个算子 workspace 为 `GetLibApiWorkSpaceSize()`；`layer_norm_quant_apt` 的 workspace 参数未读取。ACLNN 总 workspace 由 executor 汇总，可能含 Contiguous/ViewCopy 临时区；其内部 offset/event 在固定算子仓不可见。调用方拥有总 workspace，executor 管理临时 Tensor，并在传入 stream 上顺序执行到第二段返回。

### 启动图与失败传播

```text
ACLNN GetWorkspaceSize
  -> Contiguous × 5                         # 框架边界
  -> l0op::LayerNormQuant
       -> layer_norm_quant_apt::layer_norm_quant
            KERNEL_TASK_TYPE_DEFAULT(AIV_ONLY)
            if coreType==AIC: return
            else route by key to one AIV body
  -> ViewCopy(tempRes -> res)                # 框架边界
ACLNN phase 2 -> CommonOpExecutorRun(stream)
```

只有一个算子 Device 节点，没有跨 Kernel workspace/GM 边。任一参数检查、连续化、launcher 或 Tiling 失败都会阻断后续节点；entry 未知 key 则没有 body 被执行，也没有显式 Device 错误回传。

## Route 与参数设计

### 分核与资源计算

公共分核：`C=ceil(D/32)*32`，`numCore=ceil(R/ceil(R/maxCore))`，普通核 `nl=ceil(R/numCore)` 行，末核 `l=R-nl*(numCore-1)` 行。令输入元素字节 `s=2`（FP16/BF16）或 `4`（FP32）。RegBase 分片容量模型为 `single=(12+2s)*C`、`multi=(s+1)*C`、scalar reserve=256 B；若 `UB-256 < (13+3s)*C`，则 `sliceSize=floor((UB-256)/(13+3s), 256/s)`、`sliceNum=ceil(D/sliceSize)`，否则整载 `sliceSize=C`。整载再通过 `CheckSplit` 生成批行 `firstDimPerTimes`，分片固定为 1。所有 shape/POD 主量窄化到 uint32 前均做 R、D 边界检查。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 整载 | 2310000000 / `KernelLayerNormQuantNormal<half>` | `dtype=FP16 && sliceNum==1` | 容量不足 R-004 |
| 2 | R-002 BF16 整载 | 2210000000 / `...<bfloat16_t>` | `dtype=BF16 && sliceNum==1` | 容量不足 R-005 |
| 3 | R-003 FP32 整载 | 2410000000 / `...<float>` | `dtype=FP32 && sliceNum==1` | 容量不足 R-006 |
| 4 | R-004 FP16 分片 | 2300000000 / `KernelLayerNormQuantSplitD<half>` | `dtype=FP16 && sliceNum>1` | 无下一 Route |
| 5 | R-005 BF16 分片 | 2200000000 / `...<bfloat16_t>` | `dtype=BF16 && sliceNum>1` | 无下一 Route |
| 6 | R-006 FP32 分片 | 2400000000 / `...<float>` | `dtype=FP32 && sliceNum>1` | 无下一 Route |

```text
dtypeOffset = BF16 ? 200000000 : FP16 ? 300000000 : 400000000
fastOffset = (sliceNum==1) ? 10000000 : 0
key = 2000000000 + dtypeOffset + fastOffset
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 2310000000 | 950 FP16，整载 | 单 AIV | `KernelLayerNormQuantNormal<half>` | 多行 queue，FP32统计 | 尾核/尾批 |
| R-002 | 2210000000 | 950 BF16，整载 | 单 AIV | `...<bfloat16_t>` | 同上，BF16 cast | 尾核/尾批 |
| R-003 | 2410000000 | 950 FP32，整载 | 单 AIV | `...<float>` | 输入/gamma/beta用 Adds 复制 | 尾核/尾批 |
| R-004 | 2300000000 | 950 FP16，多片 | 单 AIV | `KernelLayerNormQuantSplitD<half>` | 三遍分片 | `tailSliceSize` |
| R-005 | 2200000000 | 950 BF16，多片 | 单 AIV | `...<bfloat16_t>` | 三遍分片、BF16 cast | tail slice |
| R-006 | 2400000000 | 950 FP32，多片 | 单 AIV | `...<float>` | 三遍分片、FP32 Adds | tail slice |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| RegBase selector | `IsRegbaseSocVersion \|\| npuArch==DAV_5102` | 选择 Reg POD/key/body | 全部 950 | 非 RegBase Tiling | `CanUseRegbase` |
| AIV-only | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；AIC return | 禁止 Cube/AIC 执行 body | 全部 | AIC core 无操作 | `_apt::layer_norm_quant` |
| 整载容量 | `UB-256 >= (13+3s)*C` | FP32时等于真实一行布局；2-byte dtype 对 scalar 区保守预留 256 B（实际128 B） | R-001~003 | 对应 split | `RegTiling::GetTilingSliceInfo` |
| slice repeat | `sliceSize` 向下对齐 `256/s` 元素 | Vector repeat 粒度 | R-004~006 | 源码无 `elemSize==0` 独立失败 guard | 同上 |
| dtype key | BF16→200m、FP16→300m、其他→400m | router 模板实例不同 | 全部 | API/OpDef 已先限制 dtype | `GetDataTypeKey` |
| POD 注册差异 | 头文件注册 220/221/230/231 等，但未列 240/241 | FP32 entry 显式用 `GET_TILING_DATA_WITH_STRUCT` 读取同一 Reg POD | R-003/R-006 | 固定仓无单独 fallback | `REGISTER_TILING_DATA_CLASS`; `_apt` router |

### POD/ABI

固定仓未包含 `BEGIN_TILING_DATA_DEF` 宏展开，因此精确 `sizeof/alignof/offsetof` 不可见；字段顺序、类型、producer 和 consumer 可闭合。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LayerNormQuantRegTilingData` | `numCore` | uint32/32 | 第1字段，offset不可见 | core | `PostLayerNormPtrFunc` | Init 末核判断 | `1..maxCore` |
| 同上 | `numLastDim` | uint32/32 | 第2字段 | element | 同上 | 地址/Vector count | `1..UINT_MAX` |
| 同上 | `numFirstDim` | uint32/32 | 第3字段 | row | 同上 | normal Init 读取后未参与计算；split不读 | inactive |
| 同上 | `nlFirstdimPerCore` | uint32/32 | 第4字段 | row/core | 同上 | 普通核行数、GM base | 非末核 |
| 同上 | `lFirstdimPerCore` | uint32/32 | 第5字段 | row | 同上 | 末核行数 | 末核 |
| 同上 | `firstDimPerTimes` | uint32/32 | 第6字段 | row/batch | `CheckSplit`/置1 | row_step | 至少1 |
| 同上 | `colsAligned` | uint32/32 | 第7字段 | element | `GetTilingBasicInfo` | normal queue行距 | split不读 |
| 同上 | `epsStr` | float/32 | 第8字段 | scalar | attrs[0] | variance | double→float |
| 同上 | `aveStr` | float/32 | 第9字段 | 1/element | `1.0/D` | mean/variance | FP32 |
| 同上 | `sliceNum` | uint32/32 | 第10字段 | slice | `GetTilingSliceInfo` | split loop | fast=1 |
| 同上 | `sliceSize` | uint32/32 | 第11字段 | element | 同上 | queue/DMA | aligned repeat |
| 同上 | `tailSliceSize` | uint32/32 | 第12字段 | element | 同上 | final slice | `D-(n-1)*sliceSize` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 2310000000 | `layer_norm_quant_tiling_arch35.cpp::DoTiling` | Reg POD | `_apt.cpp::layer_norm_quant` | regbase normal half + utils | ascend950 binary/config |
| R-002 | 2210000000 | 同上 | 同上 | 同上 | regbase normal bf16 + utils | 同上 |
| R-003 | 2410000000 | 同上 | 同上 | 同上 | regbase normal float + utils | 同上 |
| R-004 | 2300000000 | 同上 | 同上 | 同上 | regbase split half + utils | 同上 |
| R-005 | 2200000000 | 同上 | 同上 | 同上 | regbase split bf16 + utils | 同上 |
| R-006 | 2400000000 | 同上 | 同上 | 同上 | regbase split float + utils | 同上 |

## Kernel 实现

### 公共执行流程

`_apt` entry 只在 AIV core 执行。每个 block 处理连续行段，GM base=`blockIdx*nl*D`。Init 分配 depth=1 queues 和三个 FP32工作区，从 scale/offset 各搬 32 B 后只读首元素。half/BF16 输入、gamma、beta用 Cast 升 FP32；FP32实例用 `Adds(...,0)` 复制到 FP32工作区。所有路径的最终 q 都别名复用 `x_buf_fp32` 的低精度视图，先变为 half，再饱和转 INT8。

### Route R-001：FP16 整载

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `norm/layer_norm_quant/op_host/layer_norm_quant_tiling_arch35.cpp` | `DoTiling` | FP16→300m，`sliceNum==1`→10m | 2310000000 |
| Router | `norm/layer_norm_quant/op_kernel/layer_norm_quant_apt.cpp` | `layer_norm_quant` | key 2310000000 | normal half |

#### Device 数据流

##### 函数级调用链

```text
_apt::layer_norm_quant -> KernelLayerNormQuantNormal<half>::Init -> Process
  -> FastCopyIn + FastCopyInGammaBeta
  -> FastPrecisionComputeRegbase -> ComputeNormalMean -> ComputeNormalVariance
  -> ComputeNormalLayerNorm -> FastCopyOut
```

##### 源码等价伪码

```text
for row_batch:
  DataCopyPad r rows x and one gamma/beta row
  for valid row: FP16->FP32; mean; std; affine; q=y/scale+offset
                 FP32->FP16(CAST_NONE); clamp; INT8
  DataCopyPad exactly r*D output bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/z | `core*nl*D+batch*r*D` | valid r×D | FP16/INT8 | caller/body | body/caller | launch | 32-B padded x，z valid bytes |
| queues | UB | x `2rA`, z `rA`, gamma+beta `4A` B | FP16/INT8 | DMA/Vector | Vector/DMA | batch | depth1 |
| FP32 TBuf×3 | UB | 各 `4A` B | FP32 | Vector | Vector | row | x区后期 half alias |
| ave/var | UB | 各 64 B | allocation按 T=half；以 FP32 view消费 | ReduceSum | scalar | row | V_S/S_V |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  q["queues · 3*r*A+4*A B"]:2
  fp["3 FP32 TBuf · 12*A B"]:1
  scal["ave+var · 128 B"]:1
```

##### 字节账与复用顺序

```text
A=round_up(sliceSize,32), s=2
UB_total = 3*r*A + 16*A + 128 bytes
实际单行占用 = 19*C+128；Host full-load gate 更保守地要求 UB >= 19*C+256；批行另由 CheckSplit 缩放
dynamic_ub=N/A
workspace_total=LibApi workspace；Kernel 自身0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x FP32 TBuf | cast/variance | normalize | half q | V barriers | 最后 FP32 read 完成 |
| queues | DMA/Vector | Vector/DMA | next batch | DeQue/FreeTensor | depth1按序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→V | MTE2_V | DataCopyPad | compute | input queues | rightPadding，不计入D |
| Reduce→scalar | V_S/S_V | ReduceSum | GetValue/next V | mean/std | FP32 |
| Quant/output | V barriers + queue | affine | cast/CopyOut | shared TBuf/z | CAST_NONE→half；尾批 `row_tail_` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `layer_norm_quant_tiling_arch35.cpp` | `DoTiling` | FP16 fast key/resources |
| Kernel entry/router | `layer_norm_quant_apt.cpp` | key 2310000000 | AIV-only half normal |
| Device body/helper | `arch35/layer_norm_quant_regbase_normal.h`; `utils.h` | normal half | FP32 pipeline、tail、cast |

### Route R-002：BF16 整载

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `layer_norm_quant_tiling_arch35.cpp` | `GetDataTypeKey`; `DoTiling` | BF16 + 1 slice | 2210000000 |
| Router | `layer_norm_quant_apt.cpp` | `layer_norm_quant` | key 2210000000 | normal bf16 |

#### Device 数据流

##### 函数级调用链

```text
_apt entry -> KernelLayerNormQuantNormal<bfloat16_t>::Process
 -> FastCopyIn -> BF16-to-FP32 mean/variance/affine -> FastCopyOut
```

##### 源码等价伪码

```text
同 R-001 批行图；x/gamma/beta 为 BF16 并 CAST_NONE 到 FP32；
q FP32 经 CastFrom32To16<half> 的 CAST_NONE 分支后饱和转 INT8。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/queues | 同 R-001 | `3rA+4A` B | BF16/INT8 | DMA/V | V/DMA | batch | depth1 |
| FP32/scalar | UB | `12A+128` B | FP32 | V | V/S | row | x区half alias |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  q["BF16/INT8 queues · 3*r*A+4*A B"]:2
  fp["FP32 TBuf · 12*A B"]:1
  scal["scalar buffers · 128 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*r*A + 16*A + 128 bytes
dynamic_ub=N/A; workspace_total=LibApi workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x TBuf | BF16 cast | LayerNorm | half quant temp | V barriers | FP32消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Reduce/affine | MTE/V/S flags | BF16 DMA | FP32 pipeline | queues/TBuf | BF16→FP32，q→FP16 CAST_NONE |
| Output | valid row count | Compute | CopyOut | z | last batch/core protected |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `layer_norm_quant_tiling_arch35.cpp` | BF16 key | 2210000000 |
| Kernel entry/router | `layer_norm_quant_apt.cpp` | key 2210000000 | BF16 normal |
| Device body/helper | `layer_norm_quant_regbase_normal.h`; `utils.h` | BF16 instance | cast/sync/tail |

### Route R-003：FP32 整载

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `layer_norm_quant_tiling_arch35.cpp` | `GetDataTypeKey`; `DoTiling` | FP32 + 1 slice | 2410000000 |
| Router | `layer_norm_quant_apt.cpp` | `layer_norm_quant` | key 2410000000 | normal float |

#### Device 数据流

##### 函数级调用链

```text
_apt entry -> KernelLayerNormQuantNormal<float>::Process
 -> FastCopyIn -> Adds-copy FP32 x/gamma/beta -> mean/variance/affine
 -> FP32-to-FP16-to-INT8 -> FastCopyOut
```

##### 源码等价伪码

```text
for batch/row: x32=Adds(x,0); gamma32=Adds(gamma,0); beta32=Adds(beta,0)
mean/std/affine stay FP32; q32 is cast to half, clamp, cast int8; store D bytes.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| queues | UB | x `4rA`, z `rA`, gamma+beta `8A` B | FP32/INT8 | DMA/V | V/DMA | batch | depth1 |
| FP32 TBuf×3 | UB | `12A` B | FP32 | Adds/Vector | Vector | row | x区half alias |
| ave/var | UB | 各128 B | allocation T=float | ReduceSum | scalar | row | total256 B |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  q["FP32 x/gamma/beta + INT8 z queues · 5*r*A+8*A B"]:2
  fp["3 FP32 TBuf · 12*A B"]:1
  scal["ave+var · 256 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 5*r*A + 20*A + 256 bytes
full-load gate for one row = 25*C+256 bytes
CheckSplit producer uses (UB-64-12*C-50)/4 and 5*nl*C；该批行模型与真实固定区 20*C+256 不同，迁移时必须按实际 InitBuffer 复核
dynamic_ub=N/A; workspace_total=LibApi workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x FP32 TBuf | Adds/variance | normalize | half quant view | V barriers | affine已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/compute | Adds + V barriers | FP32 queues | FP32 Vector | TBuf | 无输入 cast，统计/仿射 FP32 |
| Quant/output | FP32→half→INT8 | affine | CopyOut | x alias/z | half CAST_NONE；尾批有效行 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `layer_norm_quant_tiling_arch35.cpp` | dtypeSize=4,key2410m | FP32容量/key |
| Kernel entry/router | `layer_norm_quant_apt.cpp` | key 2410000000 | FP32 normal |
| Device body/helper | `layer_norm_quant_regbase_normal.h`; `utils.h` | float instance | Adds复制、FP16量化桥 |

### Route R-004：FP16 分片

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `layer_norm_quant_tiling_arch35.cpp` | `GetTilingSliceInfo` | s=2 且 `UB-256 < 19*C` | key 2300000000 |
| Router | `layer_norm_quant_apt.cpp` | | key 2300000000 | split half |

#### Device 数据流

##### 函数级调用链

```text
_apt -> KernelLayerNormQuantSplitD<half>::Process
 -> ComputeMeanRegBase/ComputeSliceMeanRegBase
 -> ComputeVarianceRegBase/ComputeSliceSSD
 -> SliceCopyInRegBase/SlicePrecisionComputeRegBase/SliceCopyOut
```

##### 源码等价伪码

```text
for row: pass1 sum FP16 slices; pass2 SSD FP16 slices;
         pass3 load x/gamma/beta per slice, FP32 normalize/affine/quantize, store valid tail.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM slice | `coreBase+row*D+sid*sliceSize` | eleNum | FP16/INT8 | GM/body | body/GM | slice | DataCopyPad valid bytes |
| queues | UB | x2A+zA+gamma2A+beta2A=`7A` | FP16/INT8 | DMA/V | V/DMA | slice | depth1 |
| TBuf | UB | FP32 `12A` + scalar128 | FP32 | V | V/S | slice/row | x alias half |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  q["queues · 7*A B"]:2
  fp["FP32 TBuf · 12*A B"]:1
  scal["scalar · 128 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 19*A + 128 bytes
dynamic_ub=N/A; workspace_total=LibApi workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x queue/TBuf | mean pass | reduce | SSD/output pass | FreeTensor+flags | 每遍消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| slice statistics | sequential loops+V_S | per-slice reduce | scalar accumulator | row stats | FP32累加 |
| output tail | DataCopyPad blockLen=eleNum | quant | GM | tail | exact valid elements |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `layer_norm_quant_tiling_arch35.cpp` | slice gate | FP16 slice/key |
| Kernel entry/router | `layer_norm_quant_apt.cpp` | key2300m | half split |
| Device body/helper | `layer_norm_quant_regbase_spilt_d.h`; `utils.h` | split half | three-pass/tail |

### Route R-005：BF16 分片

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `layer_norm_quant_tiling_arch35.cpp` | slice gate + BF16 key | s=2，多片 | 2200000000 |
| Router | `layer_norm_quant_apt.cpp` | | key2200m | split bf16 |

#### Device 数据流

##### 函数级调用链

```text
_apt -> KernelLayerNormQuantSplitD<bfloat16_t>::Process
 -> BF16 slice mean -> BF16 slice SSD -> BF16 slice normalize/quantize/output
```

##### 源码等价伪码

```text
同 R-004 三遍分片；每次 BF16 输入/参数 Cast 到 FP32；q 经 half 再到 INT8。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/queues | 同 R-004 | `7A` B | BF16/INT8 | DMA/V | V/DMA | slice | depth1 |
| FP32/scalar | UB | `12A+128` B | FP32 | V | V/S | slice | alias |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  q["BF16/INT8 queues · 7*A B"]:2
  fp["FP32 work · 12*A B"]:1
  scal["scalar · 128 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 19*A + 128 bytes
dynamic_ub=N/A; workspace_total=LibApi workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 x work | BF16 cast/SSD | normalize | half q | V barriers | FP32 read done |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| stats/affine | Cast+ReduceSum+flags | BF16 slices | FP32 scalar/vector | queues/TBuf | BF16→FP32 |
| quant/tail | half bridge+DataCopyPad | FP32 q | GM | tail slice | only eleNum valid |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `layer_norm_quant_tiling_arch35.cpp` | BF16 split key | 2200000000 |
| Kernel entry/router | `layer_norm_quant_apt.cpp` | key2200m | bf16 split |
| Device body/helper | `layer_norm_quant_regbase_spilt_d.h`; `utils.h` | split bf16 | cast/three-pass/tail |

### Route R-006：FP32 分片

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `layer_norm_quant_tiling_arch35.cpp` | `dtypeSize=4`; slice gate | `UB-256<25*C` | 2400000000 |
| Router | `layer_norm_quant_apt.cpp` | | key2400m | split float |

#### Device 数据流

##### 函数级调用链

```text
_apt -> KernelLayerNormQuantSplitD<float>::Process
 -> FP32 slice mean -> FP32 slice SSD -> FP32 slice normalize/affine
 -> FP16 bridge -> INT8 -> tail-safe output
```

##### 源码等价伪码

```text
for row: scan FP32 x slices for sum; rescan for SSD;
for slice: Adds-copy x/gamma/beta, normalize/affine in FP32, cast half, clamp/cast int8, store.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM slice | core/row/sid formula | eleNum | FP32/INT8 | GM/body | body/GM | slice | blockLen=`eleNum*sizeof(T)` in, `eleNum` out |
| queues | UB | x4A+zA+gamma4A+beta4A=`13A` | FP32/INT8 | DMA/V | V/DMA | slice | depth1 |
| TBuf | UB | FP32 work12A+scalar256 | FP32 | V | V/S | slice | x half alias after affine |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  q["FP32/INT8 queues · 13*A B"]:2
  fp["FP32 work · 12*A B"]:1
  scal["ave+var · 256 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 25*A + 256 bytes
UB_total <= platform UB by the exact RegBase slice gate
dynamic_ub=N/A
workspace_total=LibApi workspace；Kernel自身0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x TBuf | Adds/SSD | normalize | half q | V barriers | FP32 affine完成 |
| queues | three passes | reduce/output | next pass/slice | FreeTensor | depth1 released |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| stats | V_S/S_V flags | FP32 ReduceSum | scalar accumulation | mean/SSD | all FP32 |
| tail/quant | DataCopyPad + V barriers | FP32 affine | INT8 GM | last slice | FP32→half→INT8；`tailSliceSize` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `layer_norm_quant_tiling_arch35.cpp` | dtypeSize4/slice/key | FP32 split |
| Kernel entry/router | `layer_norm_quant_apt.cpp` | key2400m | float split |
| Device body/helper | `layer_norm_quant_regbase_spilt_d.h`; `utils.h` | split float | Adds、three-pass、tail |

## 算子特定优化

### Trick T-001：950 entry 显式裁掉 AIC，只在 AIV 执行

#### 触发与源码依据

- 适用 Route：R-001 至 R-006
- 触发条件：950 `_apt` entry 被调度；core type 为 AIC 时立即返回
- Golden 来源：[`norm/layer_norm_quant/op_kernel/layer_norm_quant_apt.cpp:18-25`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_kernel/layer_norm_quant_apt.cpp#L18-L25)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
extern "C" __global__ __aicore__ void layer_norm_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                                                       GM_ADDR offset, GM_ADDR z, GM_ADDR scale_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(LayerNormQuantRegTilingData, tilingData, tiling);
```

#### 机制与设计原因

LayerNorm、归约、仿射和量化全部由 Vector primitive 实现，不需要 Cube。entry 用任务类型和运行时 core-type guard 双重限制执行域，使同一二进制入口不会在 AIC 上重复写输出；这也锁定本算子属于 VV/AIV，而非 MIX_CV。

#### 迁移边界

- 适用：950 上完全由 AIV Vector 实现的单 Kernel。
- 前提：Host blockDim/二进制任务类型与 AIV 调度一致，body 不引入 Cube/MIX_CV。
- 失效：一旦可达 body 增加 AIC/Cube 节点，不能复用该 AIV-only入口或继续归类 VV。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓中的真实防护就是 AIC 直接返回；移除 guard 可能让错误 core type 进入 Vector body，源码未提供可验证的性能/数值反例。

### Trick T-002：按 dtype 字节数精确缩放 950 分片

#### 触发与源码依据

- 适用 Route：全部；容量不足时落到 R-004/R-005/R-006
- 触发条件：`UB-256 < singleRowBufferSize+multiRowBufferSize`
- Golden 来源：[`norm/layer_norm_quant/op_host/layer_norm_quant_tiling_arch35.cpp:68-92`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_host/layer_norm_quant_tiling_arch35.cpp#L68-L92)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
ge::graphStatus LayerNormQuantRegTiling::GetTilingSliceInfo()
{
    uint32_t singleRowSizePerElem = fp32BufNum * sizeof(uint32_t) + fp16BufNum * dtypeSize; // 3*4 + 2*2  --  3*4 + 2*4
    uint32_t multiRowSizePerElem = fp16BufNumForMulRow * dtypeSize + i8BufNumForMulRow * sizeof(uint8_t); // 2*2 +1  --  2*4 +1

    OP_CHECK_IF(colsAligned > (UINT_MAX / (singleRowSizePerElem + multiRowSizePerElem)),
                    OP_LOGE(context->GetNodeName(), "RowBufferSize invalid!"),
                    return ge::GRAPH_FAILED);
    uint32_t singleRowBufferSize = singleRowSizePerElem * colsAligned;
    uint32_t multiRowBufferSize = multiRowSizePerElem * colsAligned;

    if ((maxUbSize - MEAN_AND_VAR_SIZE) < (singleRowBufferSize + multiRowBufferSize)) {
        uint32_t oneRepeatElemCount = 256U / dtypeSize;
        uint32_t elemSize = roundDownReg((maxUbSize - MEAN_AND_VAR_SIZE) / (singleRowSizePerElem + multiRowSizePerElem),
                                      oneRepeatElemCount);
        tilingData.set_sliceNum(CeilDiv(numCol, elemSize));
        tilingData.set_sliceSize(elemSize);
        tilingData.set_tailSliceSize(numCol - (tilingData.get_sliceNum() - 1) * elemSize);
    } else {
        tilingData.set_sliceNum(1);
        tilingData.set_sliceSize(colsAligned);
        tilingData.set_tailSliceSize(numCol);
    }
    return ge::GRAPH_SUCCESS;
}
```

#### 机制与设计原因

分片预算把三个 FP32工作区固定计为 12 B/元素，同时让 x/gamma/beta 随 s=2/4 缩放，输出固定 1 B/元素；再按一个 256-byte Vector repeat 能容纳的元素数对齐 slice。FP32 路径因此使用 25 B/元素，FP16/BF16 使用 19 B/元素，避免用半精度容量错误放大 FP32 tile。

#### 迁移边界

- 适用：同一模板覆盖多输入 dtype，但计算工作区固定 FP32 的 Vector 算子。
- 前提：InitBuffer 与 Host 的 `(13+3s)` 逐元素模型一致；Host 固定预留 256 B，而 2-byte dtype 的 ave/var 实际合计 128 B、FP32 才是 256 B。
- 失效：整载多行还需单独核对 `CheckSplit` 的批行公式，尤其 FP32 的固定区不能只套分片公式。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 FP32 当 2 B 参与 `oneRepeatElemCount` 会产生超 UB slice；固定仓通过 `GetDataTypeSize` 明确区分 2/4 B。

### Trick T-003：分片三遍扫描换取轴长无关的 UB 上界

#### 触发与源码依据

- 适用 Route：R-004、R-005、R-006
- 触发条件：`sliceNum>1`
- Golden 来源：[`norm/layer_norm_quant/op_kernel/arch35/layer_norm_quant_regbase_spilt_d.h:66-91`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/layer_norm_quant/op_kernel/arch35/layer_norm_quant_regbase_spilt_d.h#L66-L91)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    __aicore__ inline void Process()
    {
        for (uint32_t pid = 0; pid < row_work; pid++) {
            uint32_t row_offset = pid * num_last_dim;

            float mean = ComputeMeanRegBase(row_offset);
            float variance = ComputeVarianceRegBase(row_offset, mean);

            for (uint32_t sid = 0; sid < num_slice; sid++) {
                uint32_t slice_offset = 0;
                uint64_t col_offset = 0;
                uint32_t eleNum = 0;
                GetSliceOffsetAndSizeRegBase(row_offset, sid, slice_offset, col_offset, eleNum);

                SliceCopyInRegBase(slice_offset, col_offset, eleNum);

                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SlicePrecisionComputeRegBase(eleNum, mean, variance);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                SliceCopyOut(col_offset, eleNum);
            }
        }
    }
```

#### 机制与设计原因

mean、SSD、最终输出分为三遍 slice，统计量仅保留为 FP32 scalar；因此 UB 总量随 A 而非整个 D 增长，且最终结果逐片直接写 GM，不物化整行 y。代价是 x 被读取三次，源码没有提供性能百分比。

#### 迁移边界

- 适用：归一化尾轴超过 UB，统计量可用少量 scalar 交接。
- 前提：三遍顺序不能重排；tail 的 `eleNum` 同时驱动 DMA 与所有 Vector count。
- 失效：整行可驻 UB 时 R-001~R-003 可批多行并复用 gamma/beta。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在全行 mean/variance 完成前输出任一 slice 会使用不完整统计量；源码严格先完成前两遍。

## 模板与实现对比

| 对比维度 | R-001 FP16 fast | R-002 BF16 fast | R-003 FP32 fast | R-004 FP16 split | R-005 BF16 split | R-006 FP32 split | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | s2/1片 | s2/1片 | s4/1片 | s2/多片 | s2/多片 | s4/多片 | dtype+UB |
| 模板与 dtype | normal half | normal bf16 | normal float | split half | split bf16 | split float | key offsets |
| 循环与分核 | core→batch→row | 同左 | 同左 | core→row→3 pass | 同左 | 同左 | resident axis |
| Queue/TBuf | batch queues | 同左 | 4-B queues | slice queues | 同左 | 4-B queues | dtype bytes |
| 内存总占用 | `3rA+16A+128` | 同左 | `5rA+20A+256` | `19A+128` | 同左 | `25A+256` | actual InitBuffer |
| 搬运/Vector API | DataCopyPad/Cast | 同左 | Adds copy | 3-pass | 3-pass | 3-pass/Adds | dtype body |
| 同步、精度与 tail | FP32→half | 同左 | FP32→half | slice tail | slice tail | slice tail | 所有输出 int8 |
| fallback/失败边界 | split | split | split | none | none | none | unknown key no body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | FP16/BF16 | FP16/BF16 | FP16/BF16/FP32 |
| Host/Launcher | 非 RegBase POD/key | 非 RegBase POD/key | DAV_3510 RegBase POD，dtype-aware slice |
| Kernel/资源 | 通用 `layer_norm_quant.cpp` | 同左 | `_apt` AIV-only，FP32实例与 dtype-aware queues |

## 限制与失败边界

- 固定 ACLNN 只支持 `quantMode=0`；动态量化公式/`scaleOut` 不可达，Kernel 不写 `scale_out`。
- 文档称 gamma 静态 shape `[1,D]`、zero point shape 等于 scale；可见 API 分别只检查 gamma 至多2维+尾维，以及 zero point 与 scale rank 相同。Device 最终只读 scale/offset 首元素。
- `epsilon` double→float；FP16/BF16输入升 FP32，FP32输入用 Adds 复制，全部 q 最终都经 FP16 桥接后转 INT8。
- Reg POD 的 `REGISTER_TILING_DATA_CLASS` 列表没有 2400000000/2410000000，但 950 Tiling 会生产这两个 FP32 key，router 也用 `GET_TILING_DATA_WITH_STRUCT` 显式解析并实例化 float body；固定仓未给出该注册差异的说明或 fallback，复用构建元数据时必须保留审计。
- FP32 fast 的真实 UB 为 `5rA+20A+256`，而 `CheckSplit` 使用共享常量公式；卡中不把该 Host 公式等同为严格实际字节账。
- `LayerNormQuantRegTiling::DoTiling` 调用 `GetTilingSliceInfo()` 后没有检查返回值；helper 内的溢出/资源失败可能不在这一层立即传播。
- `elemSize==0`、减法下溢等极端 UB 资源异常没有独立清晰 guard；正常 950 平台 UB 是必要前提。
- 框架 Contiguous/ViewCopy、tiling macro 展开与 LibApi workspace 内部不可见；本次未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/layer_norm_quant/docs/aclnnLayerNormQuant.md` | ACLNN prototypes | 950公开支持与参数 |
| 入口实现 | `norm/layer_norm_quant/op_host/op_api/aclnn_layer_norm_quant.cpp` | GetWorkspaceSize/CheckParams | API graph/失败边界 |
| L0 launcher | `norm/layer_norm_quant/op_host/op_api/layer_norm_quant.cpp` | `l0op::LayerNormQuant` | 单节点与输出分配 |
| 平台注册 | `norm/layer_norm_quant/op_host/layer_norm_quant_def.cpp` | ascend950 config | dtype与 `_apt` 选择 |
| RegBase判定 | `common/inc/op_host/tiling_util.h` | `IsRegbaseSocVersion` | DAV_3510 |
| POD | `norm/layer_norm_quant/op_host/layer_norm_quant_tiling.h` | RegTilingData | 字段与key注册 |
| 公共Tiling | `norm/layer_norm_quant/op_host/layer_norm_quant_tiling_base.h` | Post/CheckSplit | 分核/批行 |
| selector | `norm/layer_norm_quant/op_host/layer_norm_quant_tiling.cpp` | CanUseRegbase/Tiling4 | 950跳转RegTiling/workspace |
| 950 Tiling | `norm/layer_norm_quant/op_host/layer_norm_quant_tiling_arch35.cpp` | DoTiling/GetTilingSliceInfo | 六key与dtype容量 |
| Kernel entry | `norm/layer_norm_quant/op_kernel/layer_norm_quant_apt.cpp` | `layer_norm_quant` | AIV-only六路router |
| Fast body | `norm/layer_norm_quant/op_kernel/arch35/layer_norm_quant_regbase_normal.h` | Normal<T> | 三dtype整载 |
| Split body | `norm/layer_norm_quant/op_kernel/arch35/layer_norm_quant_regbase_spilt_d.h` | SplitD<T> | 三dtype分片 |
| Helper | `norm/layer_norm_quant/op_kernel/arch35/utils.h` | Compute/cast helpers | dtype分支/量化 |
| 构建接线 | `norm/layer_norm_quant/op_host/config/ascend950/layer_norm_quant_binary.json` | `op_list` | 三dtype binary |
| 构建总表 | `scripts/kernel/binary_config/ascendc_config.json` | LayerNormQuant | ascend950 compute unit |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Reg POD驱动六key | Reg字段顺序不同于非Reg；FP32 key注册差异需审计 | `LayerNormQuantRegTilingData` |
| 调用顺序 | Contiguous→单AIV→ViewCopy | 静态不复制scaleOut，AIC return | ACLNN/L0/apt entry |
| workspace/内存 | Kernel workspace0；UB精确公式 | FP32 fast按真实InitBuffer复核 | Tiling+Init |
| 同步 | MTE2_V、V_S/S_V、V barrier、queues | scalar读写和alias不能越过flag | body methods |
| 精度与尾块 | FP32统计/仿射，统一half量化桥 | dtype复制分支、tail eleNum | utils/split/normal |
| 失败边界 | shape/quantMode/overflow失败 | unknown key无fallback；AIC不执行 | CheckParams/router |