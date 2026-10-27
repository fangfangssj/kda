---
type: Operator
title: "ForeachRoundOffNumber 950 ACLNN 算子知识"
description: "说明 aclnnForeachRoundOffNumberV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/docs/aclnnForeachRoundOffNumberV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/op_api/aclnn_foreach_round_off_number_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/op_api/foreach_round_off_number_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/arch35/foreach_round_off_number_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/foreach_round_off_number_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_host/config/ascend950/foreach_round_off_number_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:59:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachRoundOffNumber 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachRoundOffNumberV2GetWorkspaceSize` / `aclnnForeachRoundOffNumberV2`，公开 ACLNN V2 |
| 计算语义 | 公开语义为按 `roundMode` 舍入；固定 950 body 实际忽略 `roundMode`，三种 dtype 均只做 ties-to-even |
| 输入输出 | `x[]/out[]`：FP32/FP16/BF16；`roundMode`：INT8/INT64 scalar，经 executor 转 DT_INT8 Tensor |
| 执行类型 | 单 Kernel，SIMT |
| Kernel 路线 | R-001 FP32 `rintf`；R-002 FP16 `hrint`；R-003 BF16 FP32 中转 `rintf` |
| 核心资源 | GM TensorList/POD；1024 threads/block；无显式 shared/dynamic UB buffer、Queue 或 workspace |
| 来源闭包 | `source_closed`：公开 V2 ACLNN → L0 launcher → arch35 tiling/plain POD → 编译期 schMode router → SIMT VF body → 950 CMake/config 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开接口定义 `y_i[j]=round(x_i[j],roundMode)`，模式 1–6 分别为 rint/floor/ceil/round/trunc/odd。固定 Commit 的 950 `OpForeachRoundOffNumberSimt` 没有读取传入的 `roundMode`：FP32 调 `rintf`，FP16 调 `hrint`，BF16 转 FP32 后调 `rintf` 再转回 BF16。因此本卡把“公开契约”和“当前 950 body 实际行为”同时保留，不能把未实现模式推测成已支持。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，ND、每 Tensor 0–8 维 | FP32/FP16/BF16，列表内一致 | API 非空；空列表直接成功；实现会尝试 Contiguous |
| `roundMode` | 单个 `aclScalar` | INT8/INT64 | 转为 DT_INT8 Tensor并传至 Kernel，但 950 body 不读取 |
| `out` | TensorList，与对应输入 shape/dtype 相同 | FP32/FP16/BF16 | 输出 Tensor 由调用方拥有；无 ViewCopy |

### 入口与版本

V2 比旧 `aclnnForeachRoundOffNumber` 更新，区别是 `roundMode` 使用 `aclScalar`，故 950 采用 V2。V2 文档公开支持 Ascend 950PR/950DT；OpDef `ascend950` config、Host CMake `arch35` 映射、ascend950 binary config 和 Kernel CMake arch35 source 形成一致平台证据。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 层与 A2/A3 共用实现：创建 executor；校验非空、950 regbase arch、dtype/shape/format；空列表早退；逐输入调用 `Contiguous`；将 scalar 转成 DT_INT8 Tensor；添加一个 `ForeachRoundOffNumber` launcher。校验或分配失败直接返回，不启动后续 Kernel。

### 参数、Workspace 与生命周期

arch35 Tiling 把 workspace slice 0 设为 0。executor workspace 仍可能包含 `Contiguous` 框架前处理的空间，固定仓未展开其切片。Tiling 把可用 local memory 设为 `平台UB-128KiB`，但 SIMT body 没有声明或索引任何 dynamic/shared UB；该值如何映射到运行时 launch 参数属于框架边界。Tensor/launcher 由 executor 持有至第二段 stream 执行结束。

### 启动图与失败传播

```text
V2 GetWorkspaceSize
  -> optional Contiguous nodes
  -> ConvertToTensor(roundMode, DT_INT8)
  -> one ForeachRoundOffNumber launcher
       -> arch35 ForeachRoundOffNumberTilingFunc
       -> foreach_round_off_number<schMode><<<needCoreNum blocks>>>
            -> VF_CALL<OpForeachRoundOffNumberSimt<T>>(1024 threads)
```

核心算子为单 SIMT Device 节点，无节点间 workspace 边。executor stream 保证前处理先于核心 Kernel；任一前驱失败阻断 launcher。核心 Kernel 输入为 GM TensorList 描述符、roundMode GM Tensor（未消费）、输出 TensorList 和 POD。

## Route 与参数设计

### 分核与资源计算

Host 统计 `totalElements=sum_i numel(x_i)`。若非零，`needCoreNum=min(AIV核数,ceil(totalElements/1024))`；若为零，保留平台核数，随后至少钳为 1。`blockDim=needCoreNum`。每 block 固定 1024 个 SIMT thread；对每个 Tensor，global thread id 为 `blockIdx*1024+threadIdx`，全网步长为 `needCoreNum*1024`。Host 保留 128KiB DCache 后令 local memory 为 `UB-128KiB`，workspace=0。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 rint | schMode/key 0 | 首个动态输入 dtype 为 `DT_FLOAT` | 其它 schMode |
| 2 | R-002 FP16 hrint | schMode/key 1 | dtype 为 `DT_FLOAT16` | schMode 2 或失败 |
| 3 | R-003 BF16 FP32 中转 | schMode/key 2 | dtype 为 `DT_BF16` | 其它 dtype 在 ACLNN/OpDef 拒绝 |

```text
DT_FLOAT -> key/schMode 0 -> Process<float> -> rintf
DT_FLOAT16 -> key/schMode 1 -> Process<half> -> hrint
DT_BF16 -> key/schMode 2 -> Process<bfloat16_t> -> float+rintf+bf16
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | FP32、950、校验通过 | 单 SIMT Kernel | `Process<float>`/`Op...<float>` | GM 直读直写，无显式 UB/barrier | `idx<count` guard |
| R-002 FP16 | 1 | FP16、950、校验通过 | 单 SIMT Kernel | `Process<half>`/`Op...<half>` | 同上 | 同上 |
| R-003 BF16 | 2 | BF16、950、校验通过 | 单 SIMT Kernel | `Process<bfloat16_t>` | 每线程寄存器/临时值中转 FP32 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| DCache 预留 | `ubSize>128KiB` | Host 从 local memory 配额扣除 DCache | 全部 | Tiling 失败 |
| Tensor 数上限 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | Tiling 失败 |
| 每核最低粒度 | `ceil(totalElements/1024)` | 与每 block 1024 threads 对应 | 全部 | 至少 1 block |
| 空 Tensor | `count==0` | 防止地址解析/线程循环 | 全部 | continue |
| 尾元素 | `idx<count` | grid-stride 最后一轮保护 | 全部 | thread 不写 |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachRoundOffNumberTilingData` | `needCoreNum` | int32/32 | offset 0 | block | `ForeachRoundOffNumberTilingFunc` | body 未读取；blockDim 已单独设置 | >=1 |
| 同上 | `tensorCount` | int32/32 | offset 4 | Tensor 个数 | `tensorNum` 窄化 | `Process`/SIMT Tensor 循环 | <=256 |
| 同上 | `totalElements` | int64/64 | offset 8 | 元素 | 累加所有 numel | body 未读取 | Host 分核使用 |
| 同上 | `tensorElements[256]` | int64[256] | offset 16；结构总大小 2064、align 8 | 每 Tensor 元素 | Host 逐项 `GetShapeSize()` | SIMT `count=tensorElements[t]` | inactive 尾项不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | `ForeachRoundOffNumberTilingFunc` | plain struct POD | `foreach_round_off_number<0>` | `Op...Simt<float>` | arch35 source for ascend950 |
| R-002 FP16 | 1 | 同上 | 同上 | `foreach_round_off_number<1>` | `Op...Simt<half>` | 同上 |
| R-003 BF16 | 2 | 同上 | 同上 | `foreach_round_off_number<2>` | `Op...Simt<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 用 `schMode` 编译期分派 dtype。`Process<T>` 从 GM POD 取 Tensor 数与每 Tensor 元素数，用 `VF_CALL` 启动 1024-thread SIMT body。每个线程对每个非空 Tensor 解析输入/输出 TensorList 中的实际 GM 指针，并以全局线程总数为 stride 处理元素。没有 block barrier、atomic、Queue、显式 dynamic UB 或 workspace。

### Route R-001：FP32 rintf

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_round_off_number_tiling.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT` | key 0 |
| Router | `op_kernel/arch35/foreach_round_off_number.cpp` | `foreach_round_off_number<schMode>` | `schMode==0` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_round_off_number<0>
  -> Process<float>
    -> VF_CALL<OpForeachRoundOffNumberSimt<float>>(Dim3(1024), ...)
      -> SimtGetTensorAddr<float>
      -> rintf
```

##### 源码等价伪码

```text
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0 continue
  xData/yData = parse TensorList descriptor[t]
  tid=blockIdx*threadNum+threadIdx
  stride=threadNum*blockNum
  for idx=tid; idx<count; idx+=stride:
    yData[idx]=rintf(xData[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | TensorList `tensorPtr[t]+idx` | 1 FP32/thread iteration | FP32 ND | caller | SIMT thread | caller | scalar GM load |
| y GM | 同逻辑 idx | 1 FP32 | FP32 ND | SIMT thread | caller | caller | scalar GM store |
| register/temp | thread local `xVal` | 1 FP32 | FP32 | load | `rintf`/store | 当前迭代 | 下一迭代覆盖 |
| dynamic UB/workspace | N/A | 0 explicit bytes | N/A | N/A | N/A | N/A | Host local-memory 请求未被 body 索引 |

##### 分层内存布局

N/A：该 SIMT body 仅使用 GM 与线程局部值，没有显式 dynamic/shared UB buffer。

##### 字节账与复用顺序

```text
UB_total = N/A（非 AIV Queue 路径）
dynamic_ub_explicit = 0
Host SetLocalMemorySize = platform_UB - 128KiB（body 未消费）
workspace_total = 0（不含可选 Contiguous 框架 workspace）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `xVal` | GM load | `rintf` | 下一 idx | 单线程顺序 | 无跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| grid-stride | 独立 thread | GM input | GM output | 唯一 idx | `idx<count` tail guard |
| round | `rintf` | `xVal` | store | thread local | ties-to-even；roundMode 未读取 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_round_off_number_tiling.cpp` | `ForeachRoundOffNumberTilingFunc` | key 0、blockDim、POD |
| Kernel entry/router | `op_kernel/arch35/foreach_round_off_number.cpp` | `foreach_round_off_number<0>` | 编译期 FP32 分派 |
| Device body/helper | `op_kernel/arch35/foreach_round_off_number_simt.h` | `Op...Simt<float>` | grid-stride `rintf` |

### Route R-002：FP16 hrint

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_round_off_number_tiling.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key 1 |
| Router | `op_kernel/arch35/foreach_round_off_number.cpp` | `foreach_round_off_number<schMode>` | `schMode==1` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_round_off_number<1>
  -> Process<half>
    -> VF_CALL<OpForeachRoundOffNumberSimt<half>>(1024 threads)
      -> SimtGetTensorAddr<half>
      -> hrint
```

##### 源码等价伪码

```text
按公共 grid-stride 循环：
  half xVal=xData[idx]
  yData[idx]=hrint(xVal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensorPtr[t]+idx` | 1 FP16 load/store | FP16 ND | caller/thread | thread/caller | caller | scalar access |
| thread local | `xVal` | 1 half | half | GM load | `hrint` | 当前迭代 | 下一迭代覆盖 |
| dynamic UB/workspace | N/A | 0 explicit | N/A | N/A | N/A | N/A | 无共享缓冲 |

##### 分层内存布局

N/A：仅 GM 与线程局部 half 值。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub_explicit = 0
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `xVal` | GM load | `hrint` | 下一 idx | 单线程顺序 | 无共享状态 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| round/store | `hrint` | half load | half store | 当前 idx | ties-to-even；`idx<count`；mode 未读取 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_round_off_number_tiling.cpp` | `GetTilingKeyByDtype` | FP16→1 |
| Kernel entry/router | `op_kernel/arch35/foreach_round_off_number.cpp` | schMode 1 | `Process<half>` |
| Device body/helper | `op_kernel/arch35/foreach_round_off_number_simt.h` | `Op...Simt<half>` | `hrint` 与 grid stride |

### Route R-003：BF16 FP32 中转 rintf

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_round_off_number_tiling.cpp` | `GetTilingKeyByDtype` | `DT_BF16` | key 2 |
| Router | `op_kernel/arch35/foreach_round_off_number.cpp` | `foreach_round_off_number<schMode>` | `schMode==2` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_round_off_number<2>
  -> Process<bfloat16_t>
    -> VF_CALL<OpForeachRoundOffNumberSimt<bfloat16_t>>(1024 threads)
      -> SimtGetTensorAddr<bfloat16_t>
      -> static_cast<float> -> rintf -> static_cast<bfloat16_t>
```

##### 源码等价伪码

```text
按公共 grid-stride 循环：
  bf16 xVal=xData[idx]
  yData[idx]=bf16(rintf(float(xVal)))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `tensorPtr[t]+idx` | 1 BF16 load/store | BF16 ND | caller/thread | thread/caller | caller | scalar access |
| thread locals | `xVal` + FP32 conversion/result | 固定仓不暴露寄存器字节数 | BF16/FP32 | load/cast | rintf/cast/store | 当前迭代 | 下一迭代覆盖 |
| dynamic UB/workspace | N/A | 0 explicit | N/A | N/A | N/A | N/A | 无共享缓冲 |

##### 分层内存布局

N/A：BF16→FP32→BF16 仅在线程表达式中，固定仓无显式 UB 分配。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub_explicit = 0
workspace_total = 0
register_bytes = 固定仓不可见，仅能证明生命周期局限于当前 thread/idx
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local conversion | BF16→FP32 | `rintf`/BF16 cast | 下一 idx | 单线程表达式顺序 | 无跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| convert/round | scalar casts + `rintf` | BF16 load | BF16 store | 当前 idx | FP32 中舍入，再量化 BF16；`idx<count` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_round_off_number_tiling.cpp` | key 2 | BF16 selector |
| Kernel entry/router | `op_kernel/arch35/foreach_round_off_number.cpp` | schMode 2 | BF16 模板实例 |
| Device body/helper | `op_kernel/arch35/foreach_round_off_number_simt.h` | `Op...Simt<bfloat16_t>` | FP32 中转 rintf |

## 算子特定优化

### Trick T-001：统一 1024-thread grid-stride 覆盖 TensorList

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003
- 触发条件：任意非空 Tensor；blockDim 由总元素数按 1024 元素/核估算
- Golden 来源：[`foreach_round_off_number_simt.h:67-73`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number_simt.h#L67-L73)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
    AscendC::Simt::GetThreadIdx());
uint64_t stride = static_cast<uint64_t>(
    AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
```

#### 机制与设计原因

同一 grid 对列表中的每个 Tensor 重用，线程直接从 GM 标量加载、计算和写回，不物化 Tensor 拼接缓冲，也不需要每 Tensor 单独 launch。`idx<count` 同时处理任意长度与尾部；Host 的 1024 元素/核估算与 `THREAD_NUM=1024` 相配。

#### 迁移边界

- 适用：各 Tensor 之间独立、逐元素、无跨元素归约的 TensorList 算子。
- 前提：TensorList 描述符 ABI 可由 `SimtGetTensorAddr` 解析，输出不存在重叠写。
- 失效：跨 Tensor 依赖、广播、归约或需要 block 协作时必须添加同步/中间内存并重做分核。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 0 | key 1 | key 2 | dtype 唯一决定 |
| 模板与 dtype | `float` | `half` | `bfloat16_t` | 编译期 schMode |
| 循环与分核 | 1024-thread grid stride | 同左 | 同左 | 统一 TensorList 遍历 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread FP32 值 | thread half 值 | thread BF16/FP32 值 | 无显式 UB/Queue |
| 内存总占用与复用 | 固定仓不可见寄存器 | 同左 | 同左 | 当前 idx 生命周期 |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM+rintf | GM+hrint | GM+casts+rintf | SIMT math 函数 |
| 同步、精度与 tail | 无 barrier；FP32 | 无 barrier；FP16 | FP32 中转 | `idx<count` |
| fallback/失败边界 | 无备用 Route | 无备用 Route | 无备用 Route | ACLNN/OpDef 限定 dtype |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 公开支持 V2；固定 body 有 mode 语义冲突 |
| Host/Launcher | arch22 common tiling | 同 A2 | arch35 plain POD、自定义 tiling |
| Kernel/资源 | AIV Queue/UB | AIV Queue/UB | SIMT 1024 threads，无显式 UB |

## 限制与失败边界

- 950 body 完全不读取 `roundMode`，只实现 ties-to-even；这与公开 V2 对模式 2–6 及其它值的语义冲突。源码闭包结论是“行为已闭合且冲突可定位”，不是声称所有公开模式已正确实现。
- ACLNN 内部 dtype 列表还包含 `DT_INT32`，但公开文档、OpDef、ascend950 binary config 与 schMode 模板只发布三种浮点 dtype；INT32 不属于公开可达 Route。
- `GetTilingKeyByDtype` 的 default 返回 0，但 ACLNN/OpDef 应先拒绝不支持 dtype；若绕过上层，未知 dtype 可能误路由 FP32。
- `tensorNum` 从 uint64 窄化到 int32 前只检查 `<=256`，因此该窄化在可达域安全；每 Tensor/总元素累加使用 int64，固定仓未提供乘法溢出额外 guard。
- Host 设置 `localMemory=UB-128KiB`，而 body 无显式消费者；运行时如何使用该配置固定仓不可见。
- executor 的 `Contiguous` 内部 launch/workspace 为框架边界；本次知识提取未执行运行验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_round_off_number/docs/aclnnForeachRoundOffNumberV2.md` | V2 产品支持与原型 | 950 公开支持、接口契约 |
| 入口实现 | `foreach/foreach_round_off_number/op_host/op_api/aclnn_foreach_round_off_number_v2.cpp` | `ExecForeachRoundOffNumberV2GetWorkspaceSize` | 校验、前处理、executor |
| L0 launcher | `foreach/foreach_round_off_number/op_host/op_api/foreach_round_off_number_v2.cpp` | `ForeachRoundOffNumberV2` | 单 Kernel launcher |
| Host/Tiling | `foreach/foreach_round_off_number/op_host/arch35/foreach_round_off_number_tiling.cpp` | `ForeachRoundOffNumberTilingFunc` | blockDim、key、local memory、workspace |
| 平台注册 | `foreach/foreach_round_off_number/op_host/foreach_round_off_number_def.cpp` | `ForeachRoundOffNumber` | ascend950 OpDef 与 dtype |
| Host 构建 | `foreach/foreach_round_off_number/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 |
| binary config | `foreach/foreach_round_off_number/op_host/config/ascend950/foreach_round_off_number_binary.json` | `op_list` | 950 dtype 实例 |
| POD | `foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number_tiling_data.h` | `ForeachRoundOffNumberTilingData` | plain ABI 字段 |
| Kernel entry | `foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number.cpp` | `foreach_round_off_number<schMode>` | 编译期 router |
| Kernel body | `foreach/foreach_round_off_number/op_kernel/arch35/foreach_round_off_number_simt.h` | `Process/Op...Simt/SimtGetTensorAddr` | threads、地址、math、tail |
| 构建接线 | `foreach/foreach_round_off_number/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 使用 arch35 source |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | plain 2064B POD | 字段 offset、256 Tensor 上限 | `ForeachRoundOffNumberTilingData` |
| 调用顺序 | 前处理→单 SIMT Kernel | executor stream 与失败阻断 | V2 executor/L0 launcher |
| workspace/内存 | 核心 Kernel workspace=0 | local-memory 预留 128KiB；无显式 UB | `ForeachRoundOffNumberTilingFunc` |
| 同步 | 每 idx 独占，无 barrier | 不可引入重叠输出或跨线程依赖 | `Op...Simt` |
| 精度与尾块 | FP32 rintf、FP16 hrint、BF16 FP32 中转 | 当前 body 不消费 mode；`idx<count` | `Op...Simt` |
| 失败边界 | UB>128KiB、Tensor<=256 | dtype default 0 依赖上层校验 | Host Tiling/API checks |