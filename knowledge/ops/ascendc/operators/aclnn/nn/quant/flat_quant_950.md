---
type: Operator
title: "FlatQuant 950 ACLNN 算子知识"
description: "说明 aclnnFlatQuantV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/docs/aclnnFlatQuantV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_host/op_api/aclnn_flat_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_host/op_api/flat_quant.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_host/flat_quant_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_host/flat_quant_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_kernel/flat_quant_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_kernel/flat_quant_high.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_kernel/arch35/flat_quant_high_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_kernel/arch35/flat_quant_basic_cmct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_kernel/arch35/block_scheduler_flat_quant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/cmct/kernel/kernel_flat_quant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/cmct/block/block_flat_quant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/cmct/epilogue/block_epilogue_flat_quant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/cmct/utils/constant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_host/flat_quant_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T18:30:20Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# FlatQuant 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | V2 `aclnnFlatQuantV2GetWorkspaceSize` / `aclnnFlatQuantV2`，公开 ACLNN 两段式入口 |
| 计算语义 | `z_k=P1@(x_k@P2)`（per-group 允许 P2 为空），随后做 INT4 per-token 或 FLOAT4_E2M1、32 元素一组的 MX 量化；V2 新增 `dstTypeMax` |
| 输入输出 | FP16/BF16 ND 输入；INT4/INT32+FP32 scale，或 FP4_E2M1+[K,ceil(MN/64),2] FP8_E8M0 scale |
| 执行类型 | R-001/R-002 为 Matmul API Cube + AIV Vector 混合；R-003 为显式单 Kernel `MIX_CV`，AIC MMAD/Fixpipe 到 UB、AIV MicroAPI epilogue |
| Kernel 路线 | R-001 key4 INT4；R-002 key4 非 16 对齐 FP4；R-003 key6 16 对齐 FP4 CMCT |
| 核心资源 | key4 使用 GM workspace x1/x2 与 UB；key6 的 Host 仍分配同公式 workspace，但 Device 不消费该地址，而使用 L1/L0A/L0B/L0C、共享 UB/SSBUF 与寄存器 |
| 来源闭包 | `source_closed`：V2 ACLNN→Host/Tiling/POD→`flat_quant_apt`→High/CMCT body→AIC/AIV 同步、POD 与构建选项闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

先对每个 batch 求 `u_k=x_k@P2`，再求 `z_k=P1@u_k`。INT4 路与 A2/A3 相同：`scale=max(abs(z))*clipRatio/7`，经 FP32 乘法、RINT、INT4 cast。FP4 路把 `[M,N]` 展平，每 32 个元素求共享指数并输出两个 32 元素组对应的 2-byte scale 块；`dstTypeMax=0` 用目标精度默认上界，6/7 走位级 carry 规则，其它 `[6,12]` 值先乘 `invDstTypeMax` 后取向上指数。最终乘反 scale、`CAST_RINT` 到 FP4，并把 scale 转置为 `[K,ceil(MN/64),2]`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[K,M,N]` ND | FP16/BF16 | `0<K<=262144`、`0<M,N<=256`、N 偶数；先连续化 |
| `kroneckerP1` | `[M,M]` ND | 同 x | 必选、非空 |
| `kroneckerP2` | `[N,N]`；仅 FP4 per-group 可 `[0,0]` | 同 x | `hasP2` 由两维均大于 0 得出 |
| `clipRatio` | Host double | FP64→FP32 attr | `(0,1]`；INT4 使用；FP4 body 不读取 |
| `dstTypeMax` | Host double | FP64→FP32 attr | 0 或 `[6,12]`；仅 FP4 有效；Host 同时生成倒数 |
| `out` | `[K,M,N]` INT4；`[K,M,N/8]` INT32；`[K,M*N]` FP4 | INT4/INT32/FP4_E2M1 | INT32 是 INT4 物理 tensor 的 view |
| `quantScale` | `[K]` 或 `[K,ceil(MN/64),2]` ND | FP32 或 FP8_E8M0 | dtype 必须与 out 模式配套，API 仅分别检查平台允许集，shape 检查补充约束 |

### 入口与版本

公开 V2 文档只支持 Ascend 950PR/950DT，A2/A3 明确为不支持；它比 V1 多 `dstTypeMax`，所以 950 取 V2。V2 与 V1 共用内部 `l0op::FlatQuant`，但 V2 将用户值传入 POD。DAV_3510 的 API dtype 表放行 FP4_E2M1/FP8_E8M0；V1 即使在文档表中也支持 950，但不是本平台最新公开版本。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 检查非空、DAV_3510 dtype 集、shape 与 ND；`CheckRatioValid` 和 `CheckMaxValid` 再验证两个属性。三个输入分别 `Contiguous`，`l0op::FlatQuant` 对非 FP4 输出分配内部 INT4 out 和 FP32 scale；FP4 直接使用用户输出。之后可选 INT32 view，最后两个 `ViewCopy`。任一失败立即终止 executor 构造。

### 参数、Workspace 与生命周期

所有 950 FP4 key（4 和 6）workspace 都是 `useAiv*(4*M*N*2 + 8*mA*N*4)+48MiB`；key4 INT4 使用相同可变两 slice但固定余量为 16MiB。key6 虽由 Host 分配该 workspace，CMCT `Params` 不接收 `workspaceGM`，Device 不消费这块地址。workspace/executor 必须活到 stream 执行结束。

### 启动图与失败传播

```text
aclnnFlatQuantV2GetWorkspaceSize
 -> Check/Contiguous -> l0op::FlatQuant -> FlatQuant AICore launcher
 -> [INT32 only: INT4 view] -> ViewCopy(out,scale)
aclnnFlatQuantV2 -> CommonOpExecutorRun(stream)
 -> one flat_quant kernel, blockDim=aicNum
    key4: MatmulR -> GM x1 -> MatmulL -> GM x2 -> Vector quant
    key6: AIC two MMADs -> Fixpipe UB -> flag -> AIV MX epilogue -> GM
```

Host/Tiling/launcher 失败不提交核心节点；key4 Matmul tiling失败返回 Tiling failure；unknown key 或 dtype/key 组合在 router 中无输出。

## Route 与参数设计

### 分核与资源计算

`mA=align16(M)`、`nA=align16(N)`。DAV_3510 非 FP4 强制 key4；FP4 在 M、N 都为 16 倍数时 key6，否则 key4。Host 以 `iterBatch=min((512KiB-N²*2-M²*2)/(mA*nA*2)/2, 64KiB/max(mA,nA)/64/2, 256KiB/(mA*nA*4)/2, ceil(K/aicNum))`，下限 1。key6 scheduler 取 `baseM=M*iterBatch, baseN=N, baseK=64`，每核按主轮/尾轮分 batch；AIV 的硬件 blockIdx 除 task ratio 映射回 AIC block。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-003 对齐 FP4 CMCT | 6 | out FP4 且 `M%16==0 && N%16==0` | R-002 |
| 2 | R-002 非对齐 FP4 HighV2 | 4 | out FP4 且至少一维非 16 对齐 | 无设备 fallback |
| 3 | R-001 INT4 High | 4 | out INT4/INT32，DAV_3510 | 无设备 fallback |

```text
if out==FP4: key=(M%16==0 && N%16==0 ? 6 : 4)
else if arch==DAV_3510: key=4
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 INT4 High | 4 | 950，FP16/BF16→INT4/INT32 | single Cube+Vector mixed | `FlatQuantHigh<T>` + 两个 GM/ND/`MDL_CFG` `matmul::Matmul`（右乘 T→T，左乘 T→FP32） | GM x1/x2，148KiB UB；API/program-order sync | M/N/K tail by Matmul + row chunks |
| R-002 FP4 HighV2 | 4 | 950 FP4，M/N 非全16对齐 | single Cube+RegBase Vector mixed | `FlatQuantHighV2<T>` | GM x1/x2，192KiB UB；MicroAPI | MN>32768 时 M 分两段；32-group mask |
| R-003 FP4 CMCT | 6 | 950 FP4，M/N 都16对齐 | single `MIX_CV`: AIC→AIV | `KernelFlatQuant<BlockMmadFlatQuant,BlockEpilogueFlatQuant,Scheduler>` | L1/L0↔共享 UB，flag 基址 8/9 与 `+16` 双轮；SSBUF | iterBatch 尾核；P2 可跳过 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| CMCT 对齐 | `M%16==0 && N%16==0` | BlockMmad/Fixpipe 以 16 cube block 操作 | R-003 | R-002 | `DetermineMmMode` |
| AIC:AIV 比例 | DAV_3510 `aivNum==2*aicNum` | MIX kernel task ratio/双 AIV | 全部 | Tiling failure | `TilingPrepareTiling` |
| iterBatch L1/L0/L0C | 三个容量最小值 | 同时容纳 P1/P2、x 双份和 FP32 L0C 双份 | R-003 | 至少1；Tiling Matmul失败 | `CalculateIterBatch` |
| FP4 UB split | `M*N>32768` | HighV2 `MN_SIZE` FP32 slice上限 | R-002 | 两段 M：128 与 M-128 | `FlatQuantHighV2::tiling/SplitQuant` |
| `dstTypeMax` 分支 | 0；6/7；其它(6,12] | 三种 exponent 生成规则 | R-002/3 | API 拒绝区间外 | `ComputeMxQuant` |

### POD/ABI

宏与 `TCubeTiling` 完整定义在固定仓不可见，不能可靠给出 `sizeof/alignof/offsetof`；声明顺序如下。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `FlatQuantTilingData` | `dataType` | uint8/8 | offset/sizeof 不可见 | enum | 未赋值 | 当前 body 不读 | 不得依赖 |
| 同上 | `hasP2` | uint8/8 | 同上 | bool | P2 两维>0 | R-003 scheduler/BlockMmad | R-001/2 body未使用 |
| 同上 | `K` | int64/64 | 同上 | batch | x dim0 | 全部 | 1..262144 |
| 同上 | `M` | int64/64 | 同上 | element | x dim1 | 全部 | 1..256 |
| 同上 | `N` | int64/64 | 同上 | element | x dim2 | 全部 | 1..256，偶数 |
| 同上 | `iterBatch` | int64/64 | 同上 | batch/tile | 容量公式 | R-003 scheduler | >=1；key4 inactive |
| 同上 | `clipRatio` | float/32 | 同上 | scalar | attr | R-001 `FlatQuantHigh` | FP4 inactive |
| 同上 | `dstTypeMax` | float/32 | 同上 | scalar | attr或0 | R-002/3 epilogue | 0或[6,12] |
| 同上 | `invDstTypeMax` | float/32 | 同上 | reciprocal | `1/dstTypeMax`或0 | R-002/3 generic branch | 0 when dst=0 |
| 同上 | `matmulTilingR` | `TCubeTiling` | 类型布局不可见 | Matmul POD | key4 `GetTCubeTiling` | `matmulR` | key6 inactive |
| 同上 | `matmulTilingL` | `TCubeTiling` | 类型布局不可见 | Matmul POD | key4 `GetTCubeTiling` | `matmulL` | key6 inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key4 + non-FP4 | `flat_quant_tiling.cpp` | POD + 2 TCube | `flat_quant_apt.cpp` | `flat_quant_high.h` | `flat_quant_def.cpp`、`ascendc_config.json` |
| R-002 | key4 + FP4 | 同上 | 同上 | 同上 | `arch35/flat_quant_high_v2.h` | 同上 |
| R-003 | key6 + FP4 | `DetermineMmMode/CalculateIterBatch` | POD | `flat_quant_apt.cpp` | `flat_quant_basic_cmct.h`→CMCT kernel/block/epilogue+scheduler | 同上，`ENABLE_CV_COMM_VIA_SSBUF=true` |

## Kernel 实现

### 公共执行流程

三条路线都由 `flat_quant_apt.cpp::flat_quant` 根据编译期 `DTYPE_OUT` 与 key 分发。key4 注册两个 Matmul 对象，先右乘后左乘，再 Vector 量化。key6 实例化 RowMajor、`MatmulFlatQuant<>`、`BuiltInFlatQuantScheduler`；AIC 与 AIV 共享 POD，AIC 的 Fixpipe 直接写 epilogue 提供的 UB/SSBUF，避免 GM 中间结果。

### Route R-001：INT4 High

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_flat_quant.cpp` | DAV_3510 dtype map | out INT4/INT32 | internal INT4 |
| Tiling | `flat_quant_tiling.cpp` | `DetermineMmMode` | DAV_3510 non-FP4 | key4 |
| Router | `flat_quant_apt.cpp` | non-FP4/key4 | `FlatQuantHigh<DTYPE_X>` | route |

#### Device 数据流

##### 函数级调用链

```text
flat_quant -> FlatQuantHigh::Process -> ProcessHighK
 -> matmulR(x,P2)->x1(T) -> matmulL(P1,x1)->x2(FP32)
 -> ComputeMaxValue/SplitQuant -> INT4 out + FP32 scale
```

##### 源码等价伪码

```text
each core owns K interval aligned to 4
right GEMM batches <=4; left GEMM each batch; write FP32 ring
row-chunk amax; scale=amax*clip/7; FP32 multiply; RINT half; INT4 store
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace x2 | 0 | `useAiv*8*mA*N*4` | FP32 ND | matmulL | AIV | kernel | `k%8` ring |
| workspace x1 | after x2 | `useAiv*4*M*N*2` | T ND | matmulR | matmulL | kernel | `k%4` ring |
| UB | fixed | 148KiB | FP32/half/INT4 | x2 | Vector/GM | AIV | row chunks |
| L1/L0 | Matmul API internal | TCube POD | Cube | Matmul | Matmul | object | fixed repo boundary |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM x/P2"] --> R["Cube MatmulR"] --> X1["GM workspace T"] --> L["Cube MatmulL"] --> X2["GM workspace FP32"] --> U["AIV UB"] --> O["INT4/scale GM"]
```

##### 字节账、交接边与复用顺序

```text
useAiv=min(ceil(K/4),aivNum)
workspace_total=useAiv*(4*M*N*2+8*mA*N*4)+16MiB; UB=148KiB
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| x1 | matmulR | ring slice | T ND | matmulL | `PipeBarrier<PIPE_ALL>` | workspace | batch<=4 |
| x2 | matmulL | ring slice | FP32 ND | Vector | program order+MTE events | workspace | consume before reuse |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| two GEMM | `PipeBarrier<PIPE_ALL>` | matmulR | matmulL | x1 | FP16/BF16 |
| GM→UB | MTE2_V event | x2 | Vector | UB | FP32→half only for max path |
| quant/store | V/MTE3 events | Vector | GM | out/scale | RINT；真实 M*N |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `flat_quant_tiling.cpp` | `GetTCubeTiling/CalculateWorkspace` | key4 POD/slices |
| Kernel entry/router | `flat_quant_apt.cpp` | non-FP4 key4 | High body |
| Device body/helper | `flat_quant_high.h` | `ProcessHighK/SplitQuant` | two GEMM + per-token quant |

### Route R-002：非对齐 FP4 HighV2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `flat_quant_tiling.cpp` | FP4 branch | `M%16!=0 \|\| N%16!=0` | key4 |
| Router | `flat_quant_apt.cpp` | FP4/key4 | `FlatQuantHighV2<T>` | route |

#### Device 数据流

##### 函数级调用链

```text
ProcessHighK(two Matmul API) -> Quant/SplitQuant
 -> FP32 GM->UB -> BF16 CAST_RINT
 -> ExpMaxVf* -> ScaleVf* -> QuantVf -> TransLayoutVf -> GM
```

##### 源码等价伪码

```text
z FP32 -> BF16; each 32-group reduce exponent/max
choose default/dynamic6or7/generic scale; multiply deqscale; RINT FP4 pack
repack scales to 2 bytes per 64 logical elements; if MN>32768 process M rows [0,128) and [128,M)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 | 同 R-001 | 可变 slices +48MiB | T/FP32 | Matmul | Vector | kernel | ring |
| UB `bufQueue` | static slices | 192KiB | 32768 FP32 + BF16 + packed out/emax/scales | GM | MicroAPI/GM | AIV | each split clears/reuses |
| registers | VF repeat | vector length dependent | BF16/u16/u32/FP4 | UB | UB | VF call | UpdateMask tail |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM"] --> C["two Cube Matmul"] --> W["workspace FP32"] --> U["UB FP32→BF16"] --> R["RegBase exp/scale/FP4"] --> O["GM FP4+FP8 scale"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total=useAiv*(4*M*N*2+8*mA*N*4)+48MiB
UB_total=192KiB; split threshold M*N>32768
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| x2 | matmulL | per-core ring | FP32 ND | CopyInputFromGm2Ub | call/MTE events | workspace | cmpM*N<=32768 per split |
| scale layout | ScaleVf | ceil(MN/32) bytes | E8M0 intermediate | TransLayoutVf | VF order | UB | mask/64 pairing |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM/UB | V_MTE2、MTE2_V | Matmul x2 | AIV | xF32 | exact blockLen cmpM*N*4 |
| MX | VF calls/program order | exponent | quant | UB/registers | 32 group，RINT FP4 |
| store | V_MTE3/MTE3_V | packed UB | GM | out/scale | output bytes MN/2；scale tail mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `flat_quant_tiling.cpp` | key4 FP4 + workspace | unaligned route |
| Kernel entry/router | `flat_quant_apt.cpp` | FP4 key4 | HighV2 |
| Device body/helper | `arch35/flat_quant_high_v2.h` | `computeMxQuant/QuantVf/TransLayoutVf` | three scale rules and packing |

### Route R-003：对齐 FP4 CMCT MIX_CV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `flat_quant_tiling.cpp` | FP4 aligned branch | M,N both multiple16 | key6 |
| Router | `flat_quant_apt.cpp` | FP4/key6 | `FlatQuantCmctKernel<DTYPE_X, DTYPE_OUT, DTYPE_QUANT_SCALE, layout::RowMajor>` | CMCT |
| Template | `flat_quant_basic_cmct.h` | aliases | `MatmulFlatQuant<>`, BuiltIn scheduler, FlatQuant epilogue | actual instance |

#### Device 数据流

##### 函数级调用链

```text
FlatQuantCmctKernel -> KernelFlatQuant::Run
AIC: BlockMmad::operator -> x@P2 (optional) -> FixpipeToL1 -> P1@u -> CopyOut(Fixpipe to shared UB)
AIV: BlockEpilogueFlatQuant::operator -> ComputeMxQuant -> TransLayout -> GM
```

##### 源码等价伪码

```text
for scheduler tile on AIC: load x/P2 L1; K0=64 ping-pong MMAD; Fixpipe T back to L1
load P1; per iterBatch second MMAD FP32; Fixpipe row-major to UB; set round flag
matching AIV subblock waits; for each batch in tile do 32-group MX quant and GM stores; signal AIC before next reuse
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM workspace | Host 分配、无 Device offset | `useAiv*(4*M*N*2+8*mA*N*4)+48MiB` | 未使用 | Host | 无 | executor 生命周期 | `workspaceGM` 形参未进入 CMCT Params |
| L1 | static `TOTAL_L1_SIZE` | A pingpong `2*align(iterBatch*M,16)*align(N,16)` + P2 + P1 | T NZ | MTE2/Fixpipe | L0 | AIC | p1/p2 first-round reuse |
| L0A/L0B | static total，half pingpong | each half `L0A_SIZE/2/sizeof(T)` elements | T cube | LoadData | Mmad | AIC | flags 6/7 |
| L0C | total，half when tile fits | `TOTAL_L0C_SIZE` FP32 | FP32 | Mmad | Fixpipe | AIC | FIX/M flags 0/1 |
| shared UB/SSBUF | `ubLocal_` | `TOTAL_UB_SIZE` | BF16 row-major then packed | AIC Fixpipe | AIV epilogue | MIX kernel | round parity prevents overwrite |
| AIV UB slices | same total UB | 64K BF16 + 32K packed + 2K-scale slices | mixed | shared result | MicroAPI | AIV | iterBatch offset |

##### 分层内存布局

```mermaid
flowchart LR
  GM["GM x/P1/P2"] --> L1["L1 pingpong"] --> AB["L0A/L0B"] --> MM["MMAD→FP32 L0C"]
  MM --> FP["Fixpipe row-major shared UB/SSBUF"] --> VF["AIV RegBase MX epilogue"] --> OUT["GM FP4/scale"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total=useAiv*(4*M*N*2+8*mA*N*4)+48MiB（Device key6 不消费）
L0A/L0B each use two half-capacity pingpong; L0C total hardware capacity
UB=TOTAL_UB_SIZE; logical epilogue slices: 64K BF16 +32K int8 +2K u16 +2K u16 +2K int8 +2K int8
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| AIC→AIV tile | `BlockMmad::CopyOut` | UB offset 0；`iterBatch*M*N` 16-bit elements | row-major | epilogue `Quant` | AIC 用 `SYNC_AIC_AIV_FLAG=9` 或 `9+16`；AIV parity 等待基址 9 | shared UB | 下一轮 AIC 等 `SYNC_AIV_AIC_FLAG=8` 或 `8+16` |
| P2 optional | GM→L1 | `N*N*T` | T NZ | first MMAD | MTE2/MTE1 flags | AIC L1 | `hasP2=false` skips first MMAD |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| L1/L0 | MTE/FIX/M flags 0/1/6/7 | DMA/MMAD/Fixpipe | next pipeline | pingpong buffers | baseK=64，last curK0 |
| AIC→AIV | `CrossCoreSet/WaitFlag<AIC_SYNC_AIV_MODE_4>` | AIC | matching AIV parity | shared UB | flag 9 / `9+16` |
| AIV→AIC | same mode，反向 flag | AIV MTE3 | next AIC round | UB reuse | flag 8 / `8+16`；last round skips signal |
| tail scheduler | mainTailBatch/mainTailBlock | scheduler | block body | batch domain | tail cores use B or B-1 |
| quant | MicroAPI RINT | BF16 UB | FP4 GM | group32 | masks incomplete vector repeat |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `flat_quant_tiling.cpp` | `CalculateIterBatch/SetTilingContextAndSaveData` | key6, blockDim, POD |
| Kernel entry/router | `flat_quant_apt.cpp`、`flat_quant_basic_cmct.h` | `FlatQuantCmctKernel` | complete instance |
| Device body/helper | `matmul/common/cmct/kernel/kernel_flat_quant.h` | `KernelFlatQuant::Run` | AIC/AIV graph + cross flags |
| Device body/helper | `matmul/common/cmct/block/block_flat_quant.h` | `BlockMmad<MatmulFlatQuant>` | L1/L0/MMAD/Fixpipe |
| Device body/helper | `matmul/common/cmct/epilogue/block_epilogue_flat_quant.h` | `BlockEpilogueFlatQuant` | MX epilogue and GM stores |

## 算子特定优化

### Trick T-001：对齐 shape 切到无 GM 中间的 CMCT epilogue

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：FP4 输出；M/N 是否都为 16 倍数
- Golden 来源：[`op_host/flat_quant_tiling.cpp:236-245`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_host/flat_quant_tiling.cpp#L236-L245)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (outDtype == ge::DT_FLOAT4_E2M1) {
    if ((N % CEIL_SIZE == 0) && (M % CEIL_SIZE == 0)) {
        return MM_HIGH_MODE_ALIGN;
    }
    return MM_HIGH_MODE;
}

auto compileInfo = tilingContext_->GetCompileInfo<FlatQuantCompileInfo>();
if (compileInfo != nullptr && compileInfo->npuArch == NpuArch::DAV_3510) {
    return MM_HIGH_MODE;
}
```

#### 机制与设计原因

16 对齐时 key6 让第二次 MMAD 的 Fixpipe 直接写共享 UB，AIV 就地量化；非对齐 key4 用 GM x1/x2 承接 Matmul API，保留通用 tail 能力。

#### 迁移边界

- 适用：950 FP4、M/N 16 对齐、相同 CMCT layout。
- 前提：SSBUF CV 通信构建选项、2:1 核比、完整 flag 协议。
- 失效：任一维非对齐必须用 key4；其它平台不可只凭 arch35 文件启用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 key6 用到非对齐 shape 会绕过 Host 明确设置的 tail fallback。

### Trick T-002：AIC/AIV 双轮共享 UB 协议

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：key6 CMCT，存在下一 scheduler round
- Golden 来源：[`matmul/common/cmct/kernel/kernel_flat_quant.h:176-194`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/cmct/kernel/kernel_flat_quant.h#L176-L194)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if ASCEND_IS_AIC {
    // sync with aiv
    if (roundIdx > 0) {
        if ((roundIdx & 1) == 1) {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIV_AIC_FLAG);
        } else {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
        }
    }
    int64_t offsetA = batchOffset * m_ * k_;
    blockMmadOp.template operator()<AscendC::LocalTensor<AType>>(
        ubLocal, aGlobal_[offsetA], p1Global_, p2Global_, blockShape, tileIdx < blockNum);
    // skip last round
    if (roundIdx % 2 == 0) {
        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    } else {
        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
    }
}
```

#### 机制与设计原因

round 奇偶选择两套 flag，AIC 写共享 UB 后唤醒对应 AIV sub-block；AIV 完成 GM store 后才允许 AIC 下一轮覆盖，从而把 Cube→Vector 交接留在片上。

#### 迁移边界

- 适用：`AIC_SYNC_AIV_MODE_4`、task ratio 2、轮次 parity scheduler。
- 前提：AIV epilogue 全部 store 完成后发反向 flag。
- 失效：改变 sub-block 选择或 round 分配时需同步修改两侧 flag。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只保留 AIC→AIV flag 会允许下一轮提前覆盖 UB；源码明确存在反向等待。

### Trick T-003：`dstTypeMax` 三段 exponent specialization

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：`dstTypeMax==0`、`==6/7`、或其它 `[6,12]`
- Golden 来源：[`op_kernel/arch35/flat_quant_high_v2.h:218-229`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/flat_quant/op_kernel/arch35/flat_quant_high_v2.h#L218-L229)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (dstTypeMax_ == ZERO_FLOAT) {
    AscendC::VF_CALL<ScaleVf>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeateScaleCount);
} else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
    AscendC::VF_CALL<ScaleVfDynamic>(
        scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeateScaleCount, addValueBit_);
} else {
    AscendC::VF_CALL<ScaleVfcuBLAS>(
        scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeateScaleHalfCount, invDstTypeMax_);
}

__ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)yTensor.GetPhyAddr();
AscendC::VF_CALL<QuantVf>(yAddr, xAddr, deScaleAddr, MNCeil, repeatCount);
```

#### 机制与设计原因

0 可直接从 BF16 exponent 得默认 MX scale；6/7 用预计算尾数 carry bit；其它值需要 FP32 乘倒数、检查尾数并向上取 exponent，避免把不同数值规则合并成一个近似路径。

#### 迁移边界

- 适用：FP4_E2M1 + FP8_E8M0 scale。
- 前提：保持 NaN/zero select、RINT 和 group32。
- 失效：目标格式、group size 或 scale 编码改变。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

用通用 `1/dstTypeMax` 路替代 0、6、7 会改变 exponent/舍入；源码提供专门分支作为反例。

## 模板与实现对比

| 对比维度 | R-001 INT4 High | R-002 FP4 HighV2 | R-003 FP4 CMCT | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | non-FP4 DAV3510 | FP4 非对齐 | FP4 16对齐 | dtype+alignment |
| 执行族/节点图 | Matmul+Vector | Matmul+RegBase | explicit MIX_CV | router/body |
| M/N/K、tile 与 scheduler | K每4，row split | K每4，MN>32768两段 | iterBatch，baseK64 | capacity |
| workspace/L1/L0/UB | x1/x2+16MiB,148K | x1/x2+48MiB,192K | no user slice, hardware L1/L0/UB | handoff location |
| AIC/AIV 或 kernel 交接 | GM/program order | GM/program order | UB flags 双轮 | alignment permits on-chip |
| 同步、精度与 tail | FP32→half→INT4 | BF16/MicroAPI FP4 | Fixpipe T→MicroAPI | output semantics |
| fallback/失败边界 | unknown无输出 | aligned去R-003 | unaligned去R-002 | Host-only selector |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 明确不支持 | V2 明确不支持 | V2 公开支持 |
| Host/Launcher | V1 五 key | V1 五 key | V2 key4/6，DAV3510 |
| Kernel/资源 | `flat_quant.cpp` | 同左 | `flat_quant_apt.cpp`；CMCT+SSBUF |

## 限制与失败边界

- key4 FP4 body `FlatQuantHighV2::ProcessHighK` 无条件调用右 Matmul，未读取 `hasP2`；公开 API 允许 FP4 的 P2 为 `[0,0]`。固定源码不能证明“空 P2 + 非 16 对齐 FP4”安全，复用时必须保留为明确风险边界，不得推测其行为。
- key6 的 scheduler/body 显式支持 `hasP2=false`，跳过第一次 MMAD并直接把 x 送入左乘。
- POD exact ABI 和 key4 Matmul API 内部 tile 细节不在固定仓宏展开中；不得补造。
- 未执行运行验证；性能收益只描述数据路径变化，不给数值结论。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/flat_quant/docs/aclnnFlatQuantV2.md` | `aclnnFlatQuantV2*` | 950/V2 平台与语义 |
| 入口实现 | `quant/flat_quant/op_host/op_api/aclnn_flat_quant.cpp` | V2 GetWorkspaceSize/Run | 校验、executor |
| 入口实现 | `quant/flat_quant/op_host/op_api/flat_quant.cpp` | `l0op::FlatQuant` | launcher/临时输出 |
| Host/Tiling/Launcher | `quant/flat_quant/op_host/flat_quant_tiling.h` | POD | 字段声明 |
| Host/Tiling/Launcher | `quant/flat_quant/op_host/flat_quant_tiling.cpp` | selector/workspace/iterBatch | key4/6 |
| Kernel entry/body | `quant/flat_quant/op_kernel/flat_quant_apt.cpp` | `flat_quant` | 950 router |
| Kernel entry/body | `quant/flat_quant/op_kernel/flat_quant_high.h` | `FlatQuantHigh` | INT4 key4 |
| Kernel entry/body | `quant/flat_quant/op_kernel/arch35/flat_quant_high_v2.h` | `FlatQuantHighV2` | FP4 key4 |
| Kernel entry/body | `quant/flat_quant/op_kernel/arch35/flat_quant_basic_cmct.h` | `FlatQuantCmctKernel` | CMCT instance |
| Kernel entry/body | `quant/flat_quant/op_kernel/arch35/block_scheduler_flat_quant.h` | built-in scheduler | iterBatch/tail |
| Kernel entry/body | `matmul/common/cmct/kernel/kernel_flat_quant.h` | `KernelFlatQuant` | launch-internal AIC/AIV graph |
| Kernel entry/body | `matmul/common/cmct/block/block_flat_quant.h` | `BlockMmad` | L1/L0/MMAD/Fixpipe |
| Kernel entry/body | `matmul/common/cmct/epilogue/block_epilogue_flat_quant.h` | `BlockEpilogueFlatQuant` | MX quant/store |
| Kernel entry/body | `matmul/common/cmct/utils/constant.h` | `SYNC_*_FLAG/FLAG_ID_MAX` | cross-core flag IDs |
| 构建接线 | `quant/flat_quant/op_host/flat_quant_def.cpp` | ascend950 aicConfig | apt opFile、softsync |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `FlatQuant` | ascend950、SSBUF option |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 字段 producer/consumer | 位宽/顺序、未知 offset | `FlatQuantTilingData` |
| 调用顺序 | Check→Contiguous→FlatQuant→ViewCopy | V2 dstTypeMax 传递 | V2 GetWorkspaceSize |
| workspace/内存 | key4 x2/x1 rings；key6 UB handoff | offsets、dtype、16/48MiB | `CalculateWorkspace`、各 Init |
| 同步 | key6 双向双轮 flags | task ratio2、store后反向 signal | `KernelFlatQuant::Run` |
| 精度与尾块 | 3 类 scale specialization、RINT、group32 | NaN/zero、mask、layout | `ComputeMxQuant/QuantVf` |
| 失败边界 | Host selector；非对齐空P2不可证 | 不补 fallback | `DetermineMmMode/ProcessHighK` |