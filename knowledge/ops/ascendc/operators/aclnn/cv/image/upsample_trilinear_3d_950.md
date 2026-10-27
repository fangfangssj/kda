---
type: Operator
title: "UpsampleTrilinear3d 950 ACLNN 算子知识"
description: "说明 aclnnUpsampleTrilinear3d* 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel SIMT 启动图与实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_host/op_api/aclnn_upsample_trilinear_3d.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_host/op_api/upsample_trilinear_3d.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_host/resize_upsample_trilinear_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_host/resize_upsample_trilinear_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_kernel/resize_upsample_trilinear_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_simt_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_host/config/ascend950/resize_upsample_trilinear_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-24T20:52:50Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UpsampleTrilinear3d 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnUpsampleTrilinear3dGetWorkspaceSize` / `aclnnUpsampleTrilinear3d`；内部主节点 `ResizeUpsampleTrilinear` |
| 计算语义 | 对 NCDHW 的 D/H/W 三轴做三线性插值，每个输出由至多 8 个输入邻点加权 |
| 输入输出 | 5D；FLOAT32/FLOAT16/BFLOAT16；ND 按 NCDHW，公开入口也接受 NCDHW/NDHWC |
| 执行类型 | 单 Kernel `AIV_ONLY + SIMT`；无 AIC/Cube、MIX_CV、NDDMA 或通信节点 |
| Kernel 路线 | R-001 `uint32_t` 索引；R-002 `uint64_t` 索引 |
| 核心资源 | GM 输入/输出与线程局部标量；无显式 UB/shared/dynamic UB；主 Kernel 不消费 workspace |
| 来源闭包 | `source_closed`：950 公开支持域内 ACLNN→格式前处理→950 Tiling/POD→SIMT entry/body→构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对输出 `(n,c,od,oh,ow)`，每轴源坐标在 `alignCorners=true` 时为 `o*scale`，否则为 `max(0,(o+0.5)*scale-0.5)`。每轴取得 `i0=floor(src)`、`i1=min(i0+1,in-1)` 及权重 `(1-frac,frac)`，按 D→H→W 嵌套完成八邻点加权。FLOAT16/BFLOAT16 输入加载后转为 `float`，坐标、权重和累加均为 FP32，最后 `static_cast<T>` 写回。若零权邻点是 Inf/NaN，Kernel 显式输出 NaN。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 5D；ND/NCDHW/NDHWC | FLOAT32/FLOAT16/BFLOAT16 | C/D/H/W>0；950 不支持 DOUBLE；非连续输入先 `Contiguous` |
| `outputSize` | 长度 3，D/H/W | INT64 array | 三项均>0，决定输出空间轴 |
| `alignCorners` | 标量 | bool | `true` 用角像素对齐；否则 half-pixel |
| `scalesD/H/W` | 标量 | double→float | 显式输出倍率≤50；输入/输出缩小比也≤50 |
| `out` | 5D，与 self format 相同 | 与 self 相同 | N/C 与 self 相同，D/H/W 与 outputSize 相同 |

### 入口与版本

固定 Commit 仅公开这一组无版本后缀接口。公开产品表明确支持 Ascend 950PR/950DT，950 特别说明不支持 DOUBLE。第一段生成 workspace/executor，第二段用调用方 stream 执行。`UpsampleTrilinear3dNcdhw` 在 DAV_3510、三种 AICore dtype 且三轴双向 scale≤50 时加入 `ResizeUpsampleTrilinear`；这些条件与公开支持域一致。超出公开倍率域才落入固定仓不可见的 AICPU 分支，不作为正式 Route。

## Host/Launcher 执行流程

### 参数校验与前处理

顺序为：检查 `self/outputSize/out` 非空；按 DAV_3510 白名单校验 dtype 且输入输出相同；检查 5D、outputSize 长度 3、format 为 ND/NCDHW/NDHWC；检查 C/D/H/W 和输出 D/H/W 均正，输出 shape 精确匹配。N=0 使 tensor empty 时直接返回 workspace 0。非空输入先 `Contiguous`。NDHWC 在主节点前按 `{0,4,1,2,3}` 转成 NCDHW，主节点后按 `{0,2,3,4,1}` 转回；ND/NCDHW 直接进入主节点，最后 `ViewCopy` 到 out。通用 Contiguous/Transpose/ViewCopy 是框架边界，节点添加失败立即阻断后继。

### 参数、Workspace 与生命周期

API 在三个显式 scale 均正时构造原倍率数组和倒数数组。`AsComputeScale` 得到输入步长，`ComputeCheckScale` 得到反向倍率，L0 用逐轴 `supportScale=max(scale,checkScale)` 判断 950 SIMT 支持域。950 Tiling 固定报告 16 MiB workspace；entry 明确 `(void)workspace`，主 Kernel 不读写它。executor 拥有临时连续/转置 tensor 与任务；调用者拥有第一段返回的 Device workspace，并保持至第二段 stream 完成。

### 启动图与失败传播

```text
GetWorkspaceSize
  -> Contiguous(self)
  -> [NDHWC] Transpose to NCDHW
  -> ResizeUpsampleTrilinear: one AIV-only SIMT kernel
  -> [NDHWC] Transpose back
  -> ViewCopy(out)
aclnnUpsampleTrilinear3d -> CommonOpExecutorRun(..., stream)
```

主节点 `blockDim=realCoreNum=min(outSize,coreNumAiv)`，使用单份 `ResizeUpsampleTrilinearRegBaseTilingData`。executor 保证同 stream 节点顺序；任一前处理或 launcher 添加返回空指针，ACLNN 第一段返回内部错误，不形成可执行图。

## Route 与参数设计

### 分核与资源计算

`outSize=N*C*outD*outH*outW`（元素）。`realCoreNum=min(outSize,coreNumAiv)`；`blkProcessNum=floor(outSize/realCoreNum)`；`tailBlockNum=outSize%realCoreNum`。前 `tailBlockNum` 个 block 各多处理一个元素。每个 block 内固定 512 threads，线程 `t` 遍历 `t,t+512,...<blkProcessNum`。Host 同时检查 input/output 总元素及 `inD*inH*inW`、`outD*outH*outW` 能否安全落入 uint32，选择索引模板。显式 UB/L1/L0 与 dynamic UB 均为 N/A；16 MiB workspace 不被主 entry 使用。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 32-bit SIMT | `(schId=1,isInt32=1)` | input/output 元素数和输入/输出 D*H*W 均在 `(0,UINT32_MAX]` | R-002 |
| 2 | R-002 64-bit SIMT | `(schId=1,isInt32=0)` | 上述任一量不能安全使用 uint32；其余 950 参数校验仍通过 | Tiling 失败 |

```text
DAV_3510 + dtype in {fp32,fp16,bf16} + public scale domain
  -> schId=1
  -> all address products fit uint32 ? R-001 : R-002
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `(1,1)` | 950；三种 dtype；四项范围≤UINT32_MAX | 单 AIV-only SIMT | `ResizeUpsampleTrilinearSimt<DTYPE_INPUT,uint32_t>`→`calleeInt32` | GM 直读写；512 threads；无 barrier/atomic/UB | 前 remainder blocks +1；线程循环 guard |
| R-002 | `(1,0)` | 950；三种 dtype；至少一项超 uint32 | 单 AIV-only SIMT | `ResizeUpsampleTrilinearSimt<DTYPE_INPUT,uint64_t>`→`calleeInt64` | 同上，地址/除法索引为 64 bit | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype | `float/fp16/bf16` | 950 binary 仅构建三种 DTYPE_INPUT | 两 Route | ACLNN 参数错误 | `CheckDtypeValid` / `IsInputDtypeSupported` |
| 双向 scale | 每轴 source step 与反向倍率都≤50 | 950 SIMT 主路线范围 | 两 Route | 公开支持域外 AICPU；不发布 | `UpsampleTrilinear3dNcdhw::CheckScales` |
| 索引位宽 | 四项正范围均≤UINT32_MAX | 32-bit 地址乘积不能溢出 | R-001 | R-002 | `IsShapeSizeWithinLimit` / `IsDimProductWithinLimit` |
| AIV 核数 | `coreNumAiv>0` | 至少一个 block | 两 Route | Tiling 失败 | `RegbaseTiling::Init` |
| 特殊值 | 零权邻点是 Inf/NaN | 保留 IEEE 特殊值语义 | 两 Route | 输出 NaN | `CheckSpecialZeroWeight` |

### POD/ABI

该普通结构按固定宽度类型自然对齐可推导 `sizeof=80, alignof=8`；固定仓没有 `static_assert`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| RegBase | `blkProcessNum` | int64/64 | off 0 | element/block | `outSize/realCoreNum` | `Process` | >0 |
| RegBase | `inD` | int64/64 | off 8 | element | input dim2 | 地址计算 | >0 |
| RegBase | `inH` | int64/64 | off 16 | element | input dim3 | 地址计算 | >0 |
| RegBase | `inW` | int64/64 | off 24 | element | input dim4 | 地址计算 | >0 |
| RegBase | `outD` | int64/64 | off 32 | element | output dim2 | 输出反解 | >0 |
| RegBase | `outH` | int64/64 | off 40 | element | output dim3 | 输出反解 | >0 |
| RegBase | `outW` | int64/64 | off 48 | element | output dim4 | 输出反解 | >0 |
| RegBase | `tailBlockNum` | int32/32 | off 56 | block | `outSize%realCoreNum` | block partition | `[0,realCoreNum)` |
| RegBase | `alignCorners` | int32/32 | off 60 | flag | attr→0/1 | `ComputeSourceIndex` | 0/1 |
| RegBase | `scaleD` | float/32 | off 64 | source/output | `ComputeScaleValue` | D 坐标 | ≤50 |
| RegBase | `scaleH` | float/32 | off 68 | source/output | `ComputeScaleValue` | H 坐标 | ≤50 |
| RegBase | `scaleW` | float/32 | off 72 | source/output | `ComputeScaleValue` | W 坐标 | ≤50；off 76..79 padding |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `(1,1)` | `tiling_arch35.cpp::DoTiling` | RegBase POD | `resize_upsample_trilinear_apt.cpp::<1,1>` | `simt.h::Process`→`simt_base.h::calleeInt32/SimtCompute` | ascend950 binary + ascendc_config |
| R-002 | `(1,0)` | 同上 | 同一 POD | `resize_upsample_trilinear_apt.cpp::<1,0>` | `simt.h::Process`→`calleeInt64/SimtCompute` | 同上 |

## Kernel 实现

### 公共执行流程

entry 固定 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，解析 RegBase POD；`schId=1` 时按 `isInt32` 构造索引模板。每个 block 得到一个连续输出区，SIMT threads 以 512 步长遍历。线程将线性输出索引用 magic division 反解为 `nc,d,h,w`，计算三轴邻点/权重，GM 标量加载八点，FP32 加权后直接写 GM；每个输出只有一个 writer。

### Route R-001：32-bit SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/upsample_trilinear_3d.cpp` | `UpsampleTrilinear3dNcdhw` | DAV_3510 + dtype + scale | `ResizeUpsampleTrilinear` |
| Tiling | `op_host/resize_upsample_trilinear_tiling_arch35.cpp` | `CheckInputParams` | 四项范围 fit uint32 | `isInt32=1` |
| Router | `op_kernel/resize_upsample_trilinear_apt.cpp` | `<1,1>` | `schId==1,isInt32==1` | `Simt<T,uint32_t>` |

#### Device 数据流

##### 函数级调用链

```text
resize_upsample_trilinear<1,1>
  -> ResizeUpsampleTrilinearSimt<T,uint32_t>::Init/Process
    -> GetUintDivMagicAndShift(W/H/D)
    -> asc_vf_call<calleeInt32>(512 threads)
      -> SimtCompute -> ComputeOutIndex -> ComputeInterpParams
        -> LoadInputValues -> CheckSpecialZeroWeight/ComputeTrilinearValue
        -> outputGm[y]
```

##### 源码等价伪码

```text
# 源码等价伪码
base = block*floor(outSize/blocks) + min(block,remainder)
count = floor(outSize/blocks) + (block<remainder)
for local = threadIdx.x; local < count; local += 512:
  y = base + local
  (nc,d,h,w) = magic_div_decompose(y,outW,outH,outD)
  each axis: src -> clamp(i0,i1), weight0/weight1
  load eight input values as float
  v = NaN if zero-weight neighbor is Inf/NaN else FP32 trilinear sum
  output[y] = cast<T>(v)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `nc*inD*inH*inW+d*inH*inW+h*inW+w` | input shape | T/NCDHW | caller/framework | SIMT thread | caller/stream | 邻点可重复读取 |
| GM output | `y` | output shape | T/NCDHW | one thread | framework/caller | caller/stream | 无写冲突 |
| thread locals | 8 float values、6 weights、indices | 固定仓未给寄存器字节数 | uint32/float | thread | same thread | thread lifetime | 不共享 |
| workspace | entry 不读取 | Tiling 固定报告 16 MiB | bytes | Host | N/A | executor/caller | 不是算法中间量 |
| UB/shared/dynamic UB | N/A：GM 直接访问 | 0 个显式分配 | N/A | N/A | N/A | N/A | 无 `InitBuffer` |

##### 分层内存布局

```text
N/A（未使用 AIV UB）：GM input -> SIMT thread-local indices/FP32 values -> GM output。
```

##### 字节账与复用顺序

```text
explicit_UB_total = dynamic_ub = 0
workspace_reported = 16*1024*1024 bytes
workspace_consumed_by_main_kernel = 0
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | current thread | current interpolation | next loop iteration | program order | 上次结果已写 GM |
| GM output[y] | unique thread | downstream framework | 不由本 Kernel 覆盖 | kernel completion | block/thread partitions disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| block 分段 | arithmetic | POD | block | continuous output slice | 前 remainder blocks +1 |
| thread 循环 | `idx+=blockDim.x` | thread | same thread | local slice | `idx<count` guard |
| 计算/写回 | N/A：无 barrier/atomic | scalar GM loads | scalar GM store | one output | FP32 权重/累加，cast<T> |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_arch35.cpp` | `CalTilingData/DoTiling` | block/remainder/key/workspace |
| Kernel entry/router | `resize_upsample_trilinear_apt.cpp` | `<1,1>` | AIV-only、uint32 |
| Device body/helper | `simt.h` / `simt_base.h` | `Process/calleeInt32/SimtCompute` | 512-thread GM direct interpolation |

### Route R-002：64-bit SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `tiling_arch35.cpp` | `CheckInputParams` | 任一总元素/空间积不 fit uint32 | `isInt32=0` |
| Router | `resize_upsample_trilinear_apt.cpp` | `<1,0>` | `schId=1,isInt32=0` | `Simt<T,uint64_t>` |

#### Device 数据流

##### 函数级调用链

```text
resize_upsample_trilinear<1,0>
  -> ResizeUpsampleTrilinearSimt<T,uint64_t>::Process
    -> asc_vf_call<calleeInt64>(512 threads)
      -> SimtCompute<T,uint64_t> -> same interpolation helpers
```

##### 源码等价伪码

```text
# 源码等价伪码
与 R-001 相同；所有输出索引、维度乘积、magic-div 参数和 GM offset 使用 uint64_t。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | 同 R-001，offset 为 uint64 | full tensor | T/NCDHW | caller/thread | thread/caller | caller/stream | output disjoint |
| thread locals | 同 R-001，索引为 uint64 | 固定仓未给字节数 | uint64/float | thread | thread | thread lifetime | 不共享 |
| workspace/UB | 同 R-001；main 不消费/不分配 | 0 explicit | N/A | N/A | N/A | N/A | N/A |

##### 分层内存布局

```text
N/A（未使用 AIV UB）：GM input -> uint64 SIMT thread locals -> GM output。
```

##### 字节账与复用顺序

```text
explicit_UB_total = dynamic_ub = 0
workspace_reported = 16 MiB; main kernel consumed = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals/output | 同 R-001 | 同 R-001 | 同 R-001 | program order/kernel completion | uint64 地址域有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all stages | 同 R-001 | thread | thread | one output | 仅索引位宽不同 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_arch35.cpp` | `isInt32` | overflow-safe selection |
| Kernel entry/router | `resize_upsample_trilinear_apt.cpp` | `<1,0>` | uint64 route |
| Device body/helper | `simt_base.h` | `calleeInt64/SimtCompute` | same compute with 64-bit addressing |

## 算子特定优化

### Trick T-001：Host 证明索引范围后静态选择 32/64 位

#### 触发与源码依据

- 适用 Route：R-001/R-002
- 触发条件：input/output 总元素及输入/输出 D*H*W 是否全部≤UINT32_MAX
- Golden 来源：[`resize_upsample_trilinear_tiling_arch35.cpp:202-206`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_host/resize_upsample_trilinear_tiling_arch35.cpp#L202-L206)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
baseTiling_.isInt32 = static_cast<uint64_t>(
    IsShapeSizeWithinLimit(inputSize, uint32Max) && IsShapeSizeWithinLimit(outputSize, uint32Max) &&
    IsDimProductWithinLimit(baseTiling_.inD, baseTiling_.inH, baseTiling_.inW, uint32Max) &&
    IsDimProductWithinLimit(baseTiling_.outD, baseTiling_.outH, baseTiling_.outW, uint32Max));
```

#### 机制与设计原因

Host 覆盖全部 Device 地址乘积后把位宽编码进模板 key。常见范围避免每元素运行时位宽判断；任一乘积超界则选择 uint64，避免静默窄化。

#### 迁移边界

- 适用：维度在 launch 期间固定的逐元素 SIMT。
- 前提：范围检查覆盖所有地址中间量。
- 失效：任一乘积超 UINT32_MAX 必须走 R-002。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只检查输出元素数而忽略输入/输出空间乘积，会让 32-bit 中间地址溢出。

### Trick T-002：每 block 复用三轴 magic division 参数

#### 触发与源码依据

- 适用 Route：R-001/R-002
- 触发条件：进入 `Process`，D/H/W 在 block 内恒定
- Golden 来源：[`resize_upsample_trilinear_simt.h:75-96`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_simt.h#L75-L96)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
GetUintDivMagicAndShift(mW, shiftW, lenDstW);
GetUintDivMagicAndShift(mH, shiftH, lenDstH);
GetUintDivMagicAndShift(mD, shiftD, lenDstD);
```

#### 机制与设计原因

每个 block 只生成一次 W/H/D 的 multiplier/shift，线程用 `Simt::UintDiv` 从 y 反解坐标，避免为每个输出重新构造除法参数；参数位宽随 Route 匹配。

#### 迁移边界

- 适用：固定除数的线性索引反解。
- 前提：D/H/W>0，magic 参数与索引位宽一致。
- 失效：逐元素变化的除数不能共享。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 32-bit | R-002 64-bit | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 四项范围 fit uint32 | 否则 | 完整地址乘积证明 |
| 模板与 dtype | `<T,uint32_t>`，三种 T | `<T,uint64_t>` | dtype body相同，索引分离 |
| 循环与分核 | block 连续分段、512-thread stride | 同 | outSize/remainder |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB；uint32 locals | 同，uint64 locals | GM-direct SIMT |
| 内存总占用与复用 | 16 MiB report，main 消费0 | 同 | workspace 是保留值 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM scalar + `asc_vf_call<calleeInt32>` | `calleeInt64` | 索引范围 |
| 同步、精度与 tail | 无 barrier/atomic；FP32；remainder | 同 | 单 writer |
| fallback/失败边界 | 超界转 R-002 | Tiling 参数失败 | 不窄化 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；非本卡范围 | 公开支持；非本卡范围 | 公开支持 |
| Host/Launcher | DAV_2201 selector | DAV_2201 selector | DAV_3510 RegBase selector |
| Kernel/资源 | 三轴 Matmul 路线 | 同 | AIV-only SIMT，GM direct |

## 限制与失败边界

- 950 不支持 DOUBLE；输入输出必须同 dtype/format、5D，N/C 与 D/H/W 约束见公开文档。
- 公开支持域要求显式倍率≤50且输入/输出缩小比≤50；域外请求可达仓内不可见 AICPU，不属于本卡可复用路线。
- NDHWC 由 executor 的框架 Transpose 转换，主 Kernel 只处理 NCDHW 线性地址。
- 16 MiB workspace 是 950 Tiling 固定报告值，entry 不消费；不得解释成 SIMT dynamic UB。
- 固定仓不可见寄存器分配字节数以及通用 Contiguous/Transpose/ViewCopy 内部 workspace。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnUpsampleTrilinear3d.md` | 两段式 API | 950 平台/dtype/shape/scale |
| 入口实现 | `op_host/op_api/aclnn_upsample_trilinear_3d.cpp` | `GetWorkspaceSize` | 校验、前处理、executor |
| 入口实现 | `op_host/op_api/upsample_trilinear_3d.cpp` | `UpsampleTrilinear3dNcdhw` | DAV_3510 selector |
| Host/Tiling/Launcher | `op_host/resize_upsample_trilinear_tiling.cpp` | `Tiling4UpsampleTrilinear` | arch35 dispatch |
| Host/Tiling/Launcher | `op_host/resize_upsample_trilinear_tiling_arch35.cpp` | RegBase tiling | key/POD/block/workspace |
| Kernel entry/body | `op_kernel/resize_upsample_trilinear_apt.cpp` | template entry | AIV-only router |
| Kernel entry/body | `op_kernel/arch35/resize_upsample_trilinear_simt.h` | `Process` | block partition、SIMT launch |
| Kernel entry/body | `op_kernel/arch35/resize_upsample_trilinear_simt_base.h` | `SimtCompute` | 坐标、加载、计算、tail |
| 构建接线 | `op_host/config/ascend950/resize_upsample_trilinear_binary.json` | six binaries | dtype/format binaries |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `ResizeUpsampleTrilinear` | ascend950 compute unit、auto_sync=false |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 12-field RegBase POD | 字段顺序/位宽/NCDHW 维序 | `ResizeUpsampleTrilinearRegBaseTilingData` |
| 调用顺序 | Contiguous→可选 Transpose→main→Transpose→ViewCopy | NDHWC permutations | `upsampleTrilinear3dCompute` |
| workspace/内存 | main GM direct | 16 MiB report≠Kernel消费 | `DoTiling` / apt entry |
| 同步 | 每输出单 writer，无 barrier/atomic | executor stream 顺序 | `SimtCompute` |
| 精度与尾块 | FP32 权重/累加、cast back、remainder | Inf/NaN 零权规则 | `CheckSpecialZeroWeight` |
| 失败边界 | 支持域内 selector/Tiling 失败阻断 | 不把域外 AICPU 冒充 SIMT fallback | API/L0 selector |
