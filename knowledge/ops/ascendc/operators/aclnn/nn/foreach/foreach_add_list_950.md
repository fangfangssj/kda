---
type: Operator
title: "ForeachAddList 950 ACLNN 算子知识"
description: "说明 aclnnForeachAddListV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/docs/aclnnForeachAddListV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/op_api/aclnn_foreach_add_list_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/op_api/foreach_add_list_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/arch35/foreach_add_list_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/arch35/foreach_add_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/foreach_add_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/arch35/foreach_add_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:28:50Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddListV2GetWorkspaceSize` / `aclnnForeachAddListV2`，公开 ACLNN 两段式接口；950 取最新 V2 |
| 计算语义 | `y_i[j] = x1_i[j] + x2_i[j] * alpha` |
| 输入输出 | 最多公开支持 50 个 Tensor；ND；FP16/FP32/INT32/BF16；逐位置 shape/dtype 相同；ACL Scalar 在 API 内物化为 Tensor |
| 执行类型 | 单 AIV entry 内按每个 Tensor 片段调用 `SIMT` VF；未发现 Cube/AIC 或 `MIX_CV` |
| Kernel 路线 | R-001..R-008：四种 dtype × 32/64 位索引路径 |
| 核心资源 | GM ListTensor 描述符与数据；32 位路径 1024 threads，64 位路径 512 threads；无 AIV TQue/TBuf 或 body-owned shared buffer；Host 仍设置 `localMemorySize=UBplat-128KiB`；无算子 workspace |
| 来源闭包 | `source_closed`：公开 V2 API→L0 launcher→arch35 selector/POD→模板 key→SIMT body→950 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

Host 将所有 Tensor 看作按列表顺序拼接的扁平元素域，再把连续区间分给 AIV core；Device 恢复每核首尾 Tensor 与元素 offset。每个 core 对自己覆盖的每个 Tensor 片段发起一次 SIMT `VF_CALL`，线程以 `threadIdx + k*threadNum` 遍历片段，逐元素计算公式。FP16/BF16 存储路线显式转成 FP32 中间值，最后按 round-to-nearest helper 降精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList，单 Tensor 0–8 维，ND | FP16/FP32/INT32/BF16 | 列表内 dtype 一致；950 文档给出最大 Tensor 数 50；不支持空/非连续 Tensor |
| `x2` | 列表等长，逐位置 shape 相同 | 与 `x1` 相同 | 第二加数 |
| `alpha` | ACL Scalar | FP16 数据公开文档列 FP16/FP32/DOUBLE；FP32/BF16 列 FP32/DOUBLE；INT32 列 INT32/INT64 | API 实现实际对 FP16 只接受 FP16/DOUBLE，见失败边界；物化后 BF16 为 FP32，其余与 x1 dtype 相同 |
| `out` | 列表等长，逐位置 shape 与 x1 相同，ND | 与 `x1` 相同 | 公开要求连续；最大 Tensor 数 50 |

### 入口与版本

固定仓公开旧版 Tensor-alpha 接口和新版 Scalar-alpha 接口；950 产品表同时支持 V2，故只提取 `aclnnForeachAddListV2`。V2 的 API 前处理最终仍调用同一个内部 op type `ForeachAddList`；950 由 `arch35` Tiling 与 Kernel 替代 A2/A3 公共 Vector body。

## Host/Launcher 执行流程

### 参数校验与前处理

`ForeachAddListV2CheckParams` 依次校验非空指针、当前 arch/dtype、列表和逐 Tensor shape、format。空 TensorList 直接令 `workspaceSize=0` 并返回。非空时 executor 对 x1/x2 每个 Tensor调用框架 `Contiguous`，然后将 ACL Scalar 转为单元素 Tensor：BF16 数据强制 FP32；其他 dtype 转成首个 x1 Tensor 的 dtype。公开合同仍声明非连续 Tensor 不支持，内部连续化不能作为扩展公开能力的依据。

### 参数、Workspace 与生命周期

`l0op::ForeachAddListV2` 通过 `ADD_TO_LAUNCHER_LIST_AICORE` 添加唯一主 op。arch35 Tiling 把主 op workspace 设为 0；API 返回的 executor 总 workspace 可能仍包含框架 Contiguous 节点，其公式在固定仓边界外。所有临时 Tensor 和 launcher 由第一段创建的 executor 持有，第二段在用户给定 stream 上运行。

### 启动图与失败传播

```text
V2 first phase
  -> validate
  -> optional framework Contiguous nodes for x1/x2
  -> Scalar-to-Tensor
  -> ForeachAddList AICore launcher
V2 second phase / same stream
  -> foreach_add_list<schMode><<<blockDim=needCoreNum>>>
     -> for each tensor slice owned by this core: one SIMT VF_CALL
```

主 op 是一个 Device entry；entry 内的多次 `VF_CALL` 是同一 Kernel 内按 Tensor 顺序执行的 SIMT 子调用，不形成 GM/workspace 跨 Kernel 边。框架前置节点由 executor/stream 排序；失败会阻断后续 launcher。主节点无 workspace producer/consumer，输出直接写用户 `out` GM。

## Route 与参数设计

### 分核与资源计算

平台解析得到 `coreNum=GetCoreNumAiv()`、原始 `UBplat`；要求 `UBplat>128 KiB`，随后 `localMemorySize=UBplat-128 KiB`。该设置未被 SIMT body 用作显式 shared/dynamic UB 分配。

设 `N=sum(numel(x1_i))`。若 `N=0`，POD 写 `needCoreNum=0`，launch `blockDim=1`，entry 立即返回。否则先取 `C0=min(coreNum,ceil(N/1024))`，再算 `P=ceil32(ceil(N/C0))` 个元素，最终 `C=ceil(N/P)`。每核取得扁平区间 `[core*P, min((core+1)*P,N)-1]`，Host 两次扫描 `tensorDataCountList` 得到首/尾 Tensor 与含尾 offset。`tensorNum>256`、平台/UB 信息失败或 local-memory 设置失败均返回 `GRAPH_FAILED`；公开 950 限制更严，为 50 个 Tensor。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16/i32 | `schMode=0`; `IDX_T=int32_t` | FP16 且当前 Tensor 片段 `0<localCount<=INT32_MAX` | localCount 更大→R-002 |
| 2 | R-002 FP16/i64 | `schMode=0`; `IDX_T=int64_t` | FP16 且 `localCount>INT32_MAX` | 无更大索引 fallback |
| 3 | R-003 FP32/i32 | `schMode=1`; `IDX_T=int32_t` | FP32 且 `0<localCount<=INT32_MAX` | →R-004 |
| 4 | R-004 FP32/i64 | `schMode=1`; `IDX_T=int64_t` | FP32 且 `localCount>INT32_MAX` | 无更大索引 fallback |
| 5 | R-005 INT32/i32 | `schMode=2`; `IDX_T=int32_t` | INT32 且 `0<localCount<=INT32_MAX` | →R-006 |
| 6 | R-006 INT32/i64 | `schMode=2`; `IDX_T=int64_t` | INT32 且 `localCount>INT32_MAX` | 无更大索引 fallback |
| 7 | R-007 BF16/i32 | `schMode=3`; `IDX_T=int32_t` | BF16 且 `0<localCount<=INT32_MAX` | →R-008 |
| 8 | R-008 BF16/i64 | `schMode=3`; `IDX_T=int64_t` | BF16 且 `localCount>INT32_MAX` | 无更大索引 fallback |

```text
dtype -> schMode: FP16 0 / FP32 1 / INT32 2 / BF16 3
entry reads alpha and calls dtype-specific Process
for each core-owned tensor slice:
  localCount<=0 -> skip
  else localCount<=INT32_MAX -> IDX_T=int32, threads=1024
  else -> IDX_T=int64, threads=512
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 0/i32 | FP16, `0<L<=INT32_MAX`, 950 | 单 entry→FP16 Process→VF | `OpForeachAddListFp16CastSimt<int32_t>` | 1024 threads；GM direct；无 barrier/body-owned UB | thread grid-stride 保护 L |
| R-002 | 0/i64 | FP16, `L>INT32_MAX`, 950 | 同左 | `...<int64_t>` | 512 threads | int64 尾保护 |
| R-003 | 1/i32 | FP32, `0<L<=INT32_MAX` | entry→FP32 Process→VF | `OpForeachAddListDirectSimt<float,int32_t>` | 1024 threads | grid-stride |
| R-004 | 1/i64 | FP32, `L>INT32_MAX` | 同左 | `...<float,int64_t>` | 512 threads | grid-stride |
| R-005 | 2/i32 | INT32, `0<L<=INT32_MAX` | entry→INT32 Process→VF | `OpForeachAddListDirectSimt<int32_t,int32_t>` | 1024 threads | grid-stride |
| R-006 | 2/i64 | INT32, `L>INT32_MAX` | 同左 | `...<int32_t,int64_t>` | 512 threads | grid-stride |
| R-007 | 3/i32 | BF16, `0<L<=INT32_MAX` | entry→BF16 Process→VF | `OpForeachAddListBf16CastSimt<int32_t>` | 1024 threads；FP32 thread-local intermediates | RN downcast |
| R-008 | 3/i64 | BF16, `L>INT32_MAX` | 同左 | `...<int64_t>` | 512 threads；FP32 thread-local intermediates | RN downcast |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| DCache 保留 | `UBplat>128KiB`; local memory=`UBplat-128KiB` | 固定留出 DCache | 全部 | Tiling 失败 | `ForeachAddListTilingFunc` |
| 每核至少约 1024 元素 | `C0=min(coreNum,ceil(N/1024))` | 控制外层 core 数 | 全部 | `N=0` 单 block 空返回 | 同上 |
| 32 元素 core 对齐 | `P=ceil32(ceil(N/C0))` | 统一核心切片边界 | 全部 | 无备用 Route | 同上 |
| 列表数 | Host 实现 `tensorNum<=256`；公开合同 `<=50` | POD data array 256；950 公共限制更严 | 全部 | Host >256 失败；51–256 不属于公开支持 | 同上；API 文档 |
| 索引宽度 | `L<=INT32_MAX` | 32 位索引减少宽寄存器压力，允许 1024 threads | 奇数 Route | `L>INT32_MAX` 转偶数 Route | 四个 `Process` 函数 |
| 线程数 | `sizeof(IDX_T)==4 ? 1024 : 512` | 源码按索引宽度选择 launch bound | 全部 | 编译期固定 | `THREAD_NUM_VF` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAddListTilingData` | `needCoreNum` | int32/32 | offset 0 | core | Tiling 最终 C；空域 0 | entry/四个 Process | `0..coreNum` |
| 同上 | `tensorCount` | int32/32 | offset 4 | Tensor 数 | `tensorNum` 窄化 | Device 固定源码不读取 | Host 检查 <=256 |
| 同上 | `totalElements` | int64/64 | offset 8 | 元素 | `sum(numel)` | Device 固定源码不读取 | int64 累加，无显式溢出检查 |
| 同上 | `tensorDataCountList` | int64[256] | offset 16 | 元素/每 Tensor | shape size | 四个 Process | 未用项不读 |
| 同上 | `tensorStartList` | uint16[80] | offset 2064 | Tensor index/每核 | flattened coreStart 扫描 | Process | 950 80 core 数组 |
| 同上 | `tensorEndList` | uint16[80] | offset 2224 | Tensor index/每核 | flattened coreEnd 扫描 | Process | 含尾 |
| 同上 | `tensorStartOffsetList` | int64[80] | offset 2384 | 元素/每核 | `coreStart-accumulated` | Process | 首 Tensor offset |
| 同上 | `tensorEndOffsetList` | int64[80] | offset 3024 | 元素/每核 | `coreEnd-accumulated` | Process 后加 1 转开区间 | Host 存含尾 offset |

`ForeachAddListTilingDataHost` 与 Device class 字段、顺序、数组长度一致；按固定字段自然对齐，二者 `alignof=8`、`sizeof=3664`。固定仓没有显式 `static_assert`，复用时必须保持此布局。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-002 | key0 + L guard | `ForeachAddListTilingFunc` | `ForeachAddListTilingData` | `foreach_add_list<0>` | `ForeachAddListProcessFp16`; `OpForeachAddListFp16CastSimt<i32/i64>` | `add_kernel_sources(KERNEL_SRC arch35/..., ascend950)` |
| R-003/R-004 | key1 + L guard | 同上 | 同上 | `foreach_add_list<1>` | `ForeachAddListProcessFp32`; direct float SIMT | 同上 |
| R-005/R-006 | key2 + L guard | 同上 | 同上 | `foreach_add_list<2>` | `ForeachAddListProcessInt32`; direct int SIMT | 同上 |
| R-007/R-008 | key3 + L guard | 同上 | 同上 | `foreach_add_list<3>` | `ForeachAddListProcessBf16`; BF16 cast SIMT | 同上 |

## Kernel 实现

### 公共执行流程

entry 先解析 3664 字节 POD，并在 `needCoreNum==0` 或 `blockIdx>=needCoreNum` 时返回。模板参数 `schMode` 编译期选择 alpha 读取类型和四个 Process 之一。Process 以 ListTensor 描述符取得每个 Tensor 的真实 GM 指针，计算本核在首尾 Tensor 中的半开区间 `[localStart,localEnd)`；每个非空片段发起一个 VF。SIMT 内不使用 shared/dynamic UB、Queue、atomic 或 barrier，每个线程只写唯一的 `y[index]`，不同 core 的 flattened 区间不重叠。

### Route R-001：FP16 / 32 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_add_list_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key0 |
| Router | `op_kernel/arch35/foreach_add_list.cpp` | `foreach_add_list<0>` | `schMode==FP16` | 读取 half alpha 并转 float |
| Device | `op_kernel/arch35/foreach_add_list_simt.h` | `ForeachAddListProcessFp16` | `0<L<=INT32_MAX` | 1024-thread i32 VF |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<0> -> manual half-alpha bit expansion -> ForeachAddListProcessFp16
  -> ListTensorDesc::GetDataPtr<half> -> VF_CALL<OpForeachAddListFp16CastSimt<int32_t>>(1024)
```

##### 源码等价伪码

```text
alpha_fp32 = entry's coded half-bit expansion
for each owned tensor slice L:
  for idx=threadIdx; idx<L; idx+=1024:
    a=float(x1[idx]); b=float(x2[idx]); y[idx]=half_rn(a+b*alpha_fp32)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x1/x2/y | list pointer `t` + `localStart+idx` | L FP16 elements | FP16/ND | caller | SIMT thread | Tensor | thread-exclusive y index |
| thread local | scalar `x1_val/x2_val/result` | 3 FP32 values | register allocation fixed仓不可见 | load/ALU | downcast | one loop iteration | overwritten next idx |
| launch local memory | N/A（body 无命名子区） | `UBplat-128KiB` provision；0 个源码可见已用字节 | N/A | Host Tiling | launch/runtime；body 无显式 consumer | 单 Kernel | 不得将 provision 误作已分配 Queue |
| workspace | N/A | 0 | N/A | Host Tiling | 无 consumer | 单 Kernel | 未使用 |

##### 分层内存布局

N/A：本 Route 不使用 AIV `TQue/TBuf` 或显式 SIMT shared/dynamic UB；数据从 GM 直接进入 thread-local scalar/register，并直接写回 GM。

##### 字节账与复用顺序

```text
AIV_UB_total=0 个 body-owned 显式字节；launch localMemorySize=UBplat-128KiB；workspace_total=0。
register_bytes=固定仓编译器分配不可见；每线程源码级活跃值为 x1_val、x2_val、result、index。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local scalar | current loop load/ALU | current loop store | `idx+=threadNum` | 单线程程序顺序 | 每线程 y 地址不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMD-thread loop | grid-stride loop，无 barrier | each thread | own store | unique index | FP16→FP32，表达式 `a+b*alpha`，`__float2half_rn`；`idx<L` 尾保护 |
| Alpha decode | entry 程序顺序 | GM half load | VF call | alpha scalar | 手工位域扩展；subnormal 分支按固定源码执行 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_add_list_tiling_arch35.cpp` | `ForeachAddListTilingFunc` | key0、core slice |
| Kernel entry/router | `op_kernel/arch35/foreach_add_list.cpp` | `foreach_add_list<0>` | alpha 转换与 Process |
| Device body/helper | `op_kernel/arch35/foreach_add_list_simt.h` | `OpForeachAddListFp16CastSimt<int32_t>` | 1024 threads、FP32 中间、RN |

### Route R-002：FP16 / 64 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | arch35 Tiling/entry | key0 | FP16 | FP16 Process |
| Device | `foreach_add_list_simt.h` | `ForeachAddListProcessFp16` | `L>INT32_MAX` | `IDX_T=int64_t`, 512 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<0> -> ForeachAddListProcessFp16
  -> VF_CALL<OpForeachAddListFp16CastSimt<int64_t>>(512)
```

##### 源码等价伪码

```text
for int64 idx=threadIdx; idx<L; idx+=512: y[idx]=half_rn(float(x1[idx])+float(x2[idx])*alpha_fp32)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | tensor base + 64-bit idx | L FP16 | ND | caller | SIMT | Tensor | 512-way grid stride |
| local | index + FP32 scalars | compiler allocation不可见 | scalar | thread | thread | iteration | no shared state |

##### 分层内存布局

N/A：GM direct + thread-local register path。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0；register bytes固定仓不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | loop | store | next 64-bit idx | program order | idx<L |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| loop | 512-thread grid stride | load | store | unique y | 与 R-001 同精度；int64 tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_add_list_tiling_arch35.cpp` | `ForeachAddListTilingFunc` | key/core segment |
| Kernel entry/router | `op_kernel/arch35/foreach_add_list.cpp` | `foreach_add_list<0>` | FP16 route |
| Device body/helper | `op_kernel/arch35/foreach_add_list_simt.h` | `...Fp16CastSimt<int64_t>` | 512 threads, 64-bit loop |

### Route R-003：FP32 / 32 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | arch35 Tiling/entry | key1 / `schMode==FP32` | FP32 | read FP32 alpha |
| Device | `ForeachAddListProcessFp32` | `0<L<=INT32_MAX` | 1024 threads | direct float body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<1> -> ForeachAddListProcessFp32 -> VF_CALL<OpForeachAddListDirectSimt<float,int32_t>>(1024)
```

##### 源码等价伪码

```text
for int32 idx=threadIdx; idx<L; idx+=1024: y[idx]=x1[idx]+x2[idx]*alpha
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+localStart+idx | L FP32 | ND | caller | thread | Tensor | direct scalar load/store |
| UB/workspace | N/A | 0 explicit | N/A | N/A | N/A | N/A | no allocation |

##### 分层内存布局

N/A：GM direct。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | arithmetic | GM store | next idx | program order | disjoint idx |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| direct | no barrier/atomic | GM load | GM store | unique y | FP32 expression；`idx<L` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | `GetTilingKeyByDtype/ForeachAddListTilingFunc` | key1 |
| Kernel entry/router | arch35 entry cpp | `foreach_add_list<1>` | FP32 alpha |
| Device body/helper | arch35 simt h | `OpForeachAddListDirectSimt<float,int32_t>` | 1024 threads |

### Route R-004：FP32 / 64 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | arch35 Tiling/entry | key1 | FP32 | FP32 Process |
| Device | simt header | `ForeachAddListProcessFp32` | `L>INT32_MAX` | 512-thread i64 body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<1> -> ForeachAddListProcessFp32 -> VF_CALL<OpForeachAddListDirectSimt<float,int64_t>>(512)
```

##### 源码等价伪码

```text
for int64 idx=threadIdx; idx<L; idx+=512: y[idx]=x1[idx]+x2[idx]*alpha
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | 64-bit base offset | L FP32 | ND | caller | thread | Tensor | direct |

##### 分层内存布局

N/A：GM direct + thread locals。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0；register bytes不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| locals | loop | store | next idx | program order | idx<L |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| direct | no barrier | load | store | y[idx] | FP32；64-bit tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | `ForeachAddListTilingFunc` | core slice |
| Kernel entry/router | arch35 entry cpp | `foreach_add_list<1>` | key1 |
| Device body/helper | arch35 simt h | `...DirectSimt<float,int64_t>` | 512 threads |

### Route R-005：INT32 / 32 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | arch35 Tiling/entry | key2 / INT32 | read int32 alpha | INT32 Process |
| Device | simt header | `ForeachAddListProcessInt32` | `0<L<=INT32_MAX` | 1024-thread i32 body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<2> -> ForeachAddListProcessInt32 -> VF_CALL<OpForeachAddListDirectSimt<int32_t,int32_t>>(1024)
```

##### 源码等价伪码

```text
for int32 idx=threadIdx; idx<L; idx+=1024: y[idx]=x1[idx]+x2[idx]*alpha_int32
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+idx | L INT32 | ND | caller | thread | Tensor | direct |

##### 分层内存布局

N/A：无显式 UB/shared buffer。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | ALU | GM store | next idx | program order | disjoint y |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| direct | no barrier/atomic | GM load | GM store | unique y | INT32；源码未给出溢出扩展语义；`idx<L` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | key2 producer | dtype selector |
| Kernel entry/router | arch35 entry cpp | `foreach_add_list<2>` | int32 alpha |
| Device body/helper | arch35 simt h | `...DirectSimt<int32_t,int32_t>` | 1024 threads |

### Route R-006：INT32 / 64 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | arch35 Tiling/entry | key2 | INT32 | INT32 Process |
| Device | simt header | `ForeachAddListProcessInt32` | `L>INT32_MAX` | 512-thread i64 body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<2> -> ForeachAddListProcessInt32 -> VF_CALL<OpForeachAddListDirectSimt<int32_t,int64_t>>(512)
```

##### 源码等价伪码

```text
for int64 idx=threadIdx; idx<L; idx+=512: y[idx]=x1[idx]+x2[idx]*alpha_int32
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+64-bit idx | L INT32 | ND | caller | thread | Tensor | direct |

##### 分层内存布局

N/A：GM direct。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | ALU | store | next idx | program order | idx<L |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| direct | no barrier | load | store | y[idx] | INT32；64-bit tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | `ForeachAddListTilingFunc` | flattened slice |
| Kernel entry/router | arch35 entry cpp | `foreach_add_list<2>` | key2 |
| Device body/helper | arch35 simt h | `...DirectSimt<int32_t,int64_t>` | 512 threads |

### Route R-007：BF16 / 32 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | arch35 Tiling/entry | key3 / BF16 | alpha read as FP32 | BF16 Process |
| Device | simt header | `ForeachAddListProcessBf16` | `0<L<=INT32_MAX` | 1024-thread i32 cast body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<3> -> ForeachAddListProcessBf16
  -> VF_CALL<OpForeachAddListBf16CastSimt<int32_t>>(1024)
  -> CastBf16ToFloat×2 -> FP32 expression -> CastFloatToBf16
```

##### 源码等价伪码

```text
for int32 idx=threadIdx; idx<L; idx+=1024:
  a=__bfloat162float(x1[idx]); b=__bfloat162float(x2[idx])
  y[idx]=__float2bfloat16_rn(a+b*alpha_fp32)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+idx | L BF16 | ND | caller | thread | Tensor | direct |
| thread local | a,b,result FP32 | 3 scalars | compiler allocation不可见 | cast/ALU | downcast | iteration | overwritten next idx |

##### 分层内存布局

N/A：无显式 UB/shared buffer，FP32 中间为 thread local。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0；register bytes固定仓不可见。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | casts | ALU/downcast | next idx | program order | unique output |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast compute | SIMT callee helpers | GM BF16 | GM BF16 | thread local | FP32 intermediate, RN downcast, `idx<L` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | key3 producer | BF16 selector |
| Kernel entry/router | arch35 entry cpp | `foreach_add_list<3>` | FP32 alpha |
| Device body/helper | arch35 simt h | `...Bf16CastSimt<int32_t>` | FP32 compute/RN/1024 threads |

### Route R-008：BF16 / 64 位索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | arch35 Tiling/entry | key3 | BF16 | BF16 Process |
| Device | simt header | `ForeachAddListProcessBf16` | `L>INT32_MAX` | 512-thread i64 cast body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_list<3> -> ForeachAddListProcessBf16 -> VF_CALL<OpForeachAddListBf16CastSimt<int64_t>>(512)
```

##### 源码等价伪码

```text
for int64 idx=threadIdx; idx<L; idx+=512:
  y[idx]=bf16_rn(float(x1[idx])+float(x2[idx])*alpha_fp32)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | base+64-bit idx | L BF16 | ND | caller | thread | Tensor | direct |
| thread local | FP32 a/b/result + int64 idx | compiler allocation不可见 | scalar | thread | thread | iteration | no cross-thread sharing |

##### 分层内存布局

N/A：GM direct + thread locals。

##### 字节账与复用顺序

```text
body-owned explicit UB=0；launch localMemorySize=UBplat-128KiB；workspace=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| locals | cast/ALU | downcast/store | next idx | program order | idx<L |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast compute | no barrier | GM load | GM store | y[idx] | FP32 intermediate, RN downcast, 64-bit tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | `ForeachAddListTilingFunc` | core/tensor offset |
| Kernel entry/router | arch35 entry cpp | `foreach_add_list<3>` | key3 |
| Device body/helper | arch35 simt h | `...Bf16CastSimt<int64_t>` | 512 threads, 64-bit loop |

## 算子特定优化

### Trick T-001：按索引宽度选择 1024/512 线程

#### 触发与源码依据

- 适用 Route：R-001..R-008。
- 触发条件：每个 core-owned Tensor 片段 `L<=INT32_MAX` 使用 int32，否则 int64。
- Golden 来源：[`foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_simt.h:42-47`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_simt.h#L42-L47)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// uint32_t path: 1024 threads for maximum parallelism (low register pressure)
// uint64_t path: 512 threads to balance parallelism and register pressure
template <typename IDX_T>
constexpr uint32_t THREAD_NUM_VF = (sizeof(IDX_T) == 4) ? 1024 : 512;
```

#### 机制与设计原因

常见片段使用 32 位计数和 1024 threads；只有超出 `INT32_MAX` 才承担 64 位索引并把线程数降为 512。模板把索引位宽、`LAUNCH_BOUND`、VF launch dim 和循环步长保持一致，避免在常见路线中无条件使用宽索引。

#### 迁移边界

- 适用：元素独立、单片段可能跨 32 位范围的 SIMT grid-stride 算子。
- 前提：Host/POD 用 int64 保存真实长度；两条 specialization 均构建；地址计算支持对应索引宽度。
- 失效：存在跨线程协作、固定 block size 或 shared-memory 布局依赖时需重新证明线程数。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码的可验证错误边界是对 `L>INT32_MAX` 强制窄化：会改变循环上界/offset；Process 明确转入 int64 specialization。

### Trick T-002：跨 Tensor 扁平切核并在 Device 恢复边界

#### 触发与源码依据

- 适用 Route：全部。
- 触发条件：`totalElements>0`；按 `P=ceil32(ceil(N/C0))` 分连续 flattened interval。
- Golden 来源：[`foreach/foreach_add_list/op_host/arch35/foreach_add_list_tiling_arch35.cpp:133-165`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_list/op_host/arch35/foreach_add_list_tiling_arch35.cpp#L133-L165)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t needCoreNum = std::min(coreNum, Ops::Base::CeilDiv(totalElements, SINGLE_CORE_MIN_ELEMENTS));
needCoreNum = std::max(needCoreNum, static_cast<int64_t>(1));

int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, needCoreNum);
perCoreElements = Ops::Base::CeilDiv(perCoreElements, ALIGN_SIZE) * ALIGN_SIZE;
needCoreNum = Ops::Base::CeilDiv(totalElements, perCoreElements);
```

#### 机制与设计原因

Host 不以“一个 Tensor 一个 core”绑定任务，而是把整个列表拼成一条逻辑元素轴；大 Tensor 可跨 core，小 Tensor 可由同一 core 连续处理。POD 只传首尾 Tensor 与 offset，Device 不需要全局 flatten 搬运或 workspace 物化。

#### 迁移边界

- 适用：列表元素彼此独立且逐 Tensor 地址可由 ListTensorDesc 获取的逐元素算子。
- 前提：每核区间互斥、首尾 offset 为含尾/半开区间转换一致、数组覆盖平台 core 数。
- 失效：Tensor 间有依赖、需要保持整 Tensor 原子性或输出地址别名时不能直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001/2 FP16 | R-003/4 FP32 | R-005/6 INT32 | R-007/8 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key0 + L guard | key1 + L guard | key2 + L guard | key3 + L guard | dtype→schMode，再按 L 分索引 |
| 模板与 dtype | half storage/FP32 local | float direct | int32 direct | bf16 storage/FP32 local | 低精度显式升精度 |
| 循环与分核 | flattened core slice + per-Tensor VF | 同左 | 同左 | 同左 | 公共 POD |
| threads/local | i32 1024；i64 512 | 同左 | 同左 | 同左 | 索引宽度 guard |
| 内存总占用与复用 | GM direct；thread local | GM direct | GM direct | GM direct；thread local FP32 | 无显式 UB/workspace |
| SIMT API | FP16 cast VF | direct float VF | direct int VF | BF16 cast VF | dtype 语义 |
| 同步、精度与 tail | no barrier；FP32→half RN | no barrier；FP32 | no barrier；INT32 | no barrier；FP32→bf16 RN | 每线程唯一输出，idx<L |
| fallback/失败边界 | i32→i64 | i32→i64 | i32→i64 | i32→i64 | 未知 schMode 不应由 Host 产生 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 公开 V2 支持 |
| Host/Launcher | arch22 公共 Tiling | arch22 公共 Tiling | arch35 独立 flattened Tiling，主 workspace 0 |
| Kernel/资源 | AIV Vector Queue/UB | AIV Vector Queue/UB | SIMT VF，GM direct，无显式 UB Queue |

## 限制与失败边界

- 公开 950 最大 Tensor 数是 50；arch35 Host 结构容量和检查上限为 256。51–256 仅是实现容量信号，不属于公开支持。
- V2 文档写 FP16 数据时 Scalar 可为 FP16/FP32/DOUBLE，但固定 API 实现的 `FOREACH_ADD_LIST_FLOAT16_SUPPORT_LIST` 只有 FP16/DOUBLE，会拒绝 FP32 Scalar；本卡按实际固定实现记录此发布差异，不推测修复。
- V2 文档不支持空 Tensor/非连续 Tensor。API 空 TensorList 会短路；arch35 Tiling 对 `totalElements==0` 安全返回空域。两者都不构成对单个空 Tensor 的公开支持扩展。
- FP16 entry 的 half→float 使用显式位域拼装；对 `exp==0 && mant!=0` 的源码分支没有执行规范化循环。本卡只记录固定 body 行为，不推测其与框架参考值的数值一致性。
- `SetLocalMemorySize(UB-128KiB)` 被 Host 设置，但 SIMT body 没有可见的 dynamic/shared UB consumer；不把该容量计入实际显式 buffer。
- `totalElements` 以 int64 累加、POD tensorCount 窄化为 int32，固定仓未提供总元素溢出检查。INT32 算术溢出语义、编译器是否融合浮点乘加、寄存器实际字节数均不可见。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_add_list/docs/aclnnForeachAddListV2.md` | V2 prototypes/product table | 950 公开支持、Scalar/shape/list 约束 |
| 入口实现 | `foreach/foreach_add_list/op_host/op_api/aclnn_foreach_add_list_v2.cpp` | `ExecForeachAddListV2GetWorkspaceSize` | 校验、连续化、Scalar 转 Tensor、executor |
| Launcher | `foreach/foreach_add_list/op_host/op_api/foreach_add_list_v2.cpp` | `l0op::ForeachAddListV2` | 单主 op launcher |
| Host/POD | `foreach/foreach_add_list/op_host/arch35/foreach_add_list_tiling_arch35.h` | `ForeachAddListTilingDataHost` | Host POD/compile info |
| Host/Tiling | `foreach/foreach_add_list/op_host/arch35/foreach_add_list_tiling_arch35.cpp` | `ForeachAddListTilingFunc` | dtype key、core slice、local memory、workspace |
| OpDef/平台 | `foreach/foreach_add_list/op_host/foreach_add_list_def.cpp` | `ForeachAddList::ForeachAddList` | ascend950 注册与 dtype/layout |
| Host 构建 | `foreach/foreach_add_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950 选择 arch35 Tiling |
| Kernel entry/body | `foreach/foreach_add_list/op_kernel/arch35/foreach_add_list.cpp` | `foreach_add_list<schMode>` | POD parse、alpha、dtype router |
| Kernel body | `foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_simt.h` | four Process/four VF bodies | thread mapping、index guard、precision |
| POD | `foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_tiling_data.h` | `ForeachAddListTilingData` | Device POD |
| Template key | `foreach/foreach_add_list/op_kernel/arch35/foreach_add_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 0..3 实例选择 |
| 构建接线 | `foreach/foreach_add_list/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 选择独立 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 3664 B Host/Device 同序 POD | 256 Tensor、80 core 数组、含尾 end offset | 两个 TilingData 定义 |
| 调用顺序 | entry dtype specialization→逐 Tensor VF_CALL | alpha 读取类型与 schMode 对应 | `foreach_add_list`; four Process |
| workspace/内存 | 主 op workspace 0，GM direct | DCache 保留不等于 body buffer | `ForeachAddListTilingFunc` |
| 同步 | 无 barrier/atomic；线程与 core 写域互斥 | flattened interval 必须不重叠 | Process + grid-stride bodies |
| 精度与尾块 | FP16/BF16 FP32 中间+RN；i32/i64 tail guard | `L<=INT32_MAX` 分界、1024/512 threads | VF bodies；`THREAD_NUM_VF` |
| 失败边界 | platform/UB/tensorNum/localMemory failures | 公开 50 Tensor 上限；API dtype 差异 | Tiling checks；API support lists |