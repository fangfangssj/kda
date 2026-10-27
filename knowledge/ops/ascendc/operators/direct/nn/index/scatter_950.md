---
type: Operator
title: "Scatter 950 Direct Invoke 算子知识"
description: "说明 cann_bench::scatter、scatter_npu 与 Host 可达 launch_scatter* 在 950 上的接口语义、Launcher 参数生成、Route 选择、多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/README.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_api/aclnn_scatter.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_host/arch35/scatter_elements_v2_asc_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_kernel/scatter_elements_v2_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_kernel/arch35/scatter_elements.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_kernel/arch35/scatter_elements_deterministic.h"
    role: implementation
created_at: "2026-08-24T19:31:26Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Scatter 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | Python `cann_bench.scatter` → `torch.ops.cann_bench.scatter` → `scatter_npu`；Host 直调子入口为 `launch_scatter`、`launch_scatter_copy`、`launch_scatter_copy_simt`、`launch_scatter_winner`、`launch_scatter_serial`、`launch_scatter_owned`、`launch_scatter_mul_float_fused`、`launch_scatter_add_f32_fast`、`launch_scatter_atomic_max_i32`。入口类型为 `Direct Invoke`。 |
| 计算语义 | 先以 `data` 初始化 `y`，再沿 `dim` 用 `indices` 把 `updates` 映射到 `y`；重复目标按 `update/add/multiply/amax/amin` 执行覆盖或归并。 |
| 输入输出 | 原始 `data/indices/updates` 可非连续，Device 看到的是各自的 contiguous 副本；`indices` 为 `int32/int64`，`data` 与 `updates` 同 dtype，支持 `float16/float32/bfloat16/int32/int64`；输出 shape 与 `data` 相同，但实际输出是按 `data.contiguous()` 分配的连续张量。 |
| 执行类型 | AIV MTE/Vector 节点与 AIV 上的 SIMT VF 节点；含 `AtomicAdd/AtomicMax`。没有可达 Cube/AIC 或 `MIX_CV` 节点。 |
| Kernel 路线 | 按实现重要性为 R-001 winner update；R-005 FP16 add；R-003 atomic int32 amax；R-002 owned update；R-008 FP32 multiply；R-006 FP32 atomic add；R-004 owned extrema；R-007 owned add；R-009 owned multiply；R-012 独立 copy；R-010 通用直调 add；R-011 融合 FP32 multiply。 |
| 核心资源 | `data/indices/updates/output` GM；winner `int32[dataNum]` GM；FP16 add accumulator `float[dataNum]` GM；固定 16 MiB workspace 对象但当前 body 不消费；AIV copy 32 KiB TBuf；向量 cast 48 KiB TBuf；SIMT dynamic UB 为 N/A。 |
| 来源闭包 | `direct_implementation_closed`：Python/Torch 注册、`scatter_npu` selector、全部 Host 可达 `launch_scatter*`、POD、entry、SIMT/helper、MTE/Vector body 与构建接线均已闭合。官方 `ScatterElementsV2` 仅是相关语义映射，不是本交付件的实现来源。 |
| 验证状态 | 已按固定交付源码闭合 schema、Host producer、POD、全部可达 launcher 与 Device body；本卡不把交付报告中的运行现象写成算子契约。 |

## 接口与语义

### 计算语义

设输入 rank 为 `r`，归一化轴为 `d`。对 `indices` 的线性位置 `s` 解码为坐标 `q=(q0,…,q(r-1))`，目标坐标为：

```text
p[k] = q[k]                    , k != d
p[d] = uint64(indices[s])
dst  = Σ p[k] * dataStride[k]
y    = clone(data)
```

随后按 `updates[s]` 更新 `y[dst]`。`update` 的 winner 路线以最大线性 `s` 为最终写者；owned 路线按 `mid=0..midAxis-1` 顺序覆盖，同样得到最后线性位置。`add`、`multiply`、`amax`、`amin` 分别执行加、乘、最大、最小归并。浮点 owned 极值显式传播 NaN：旧值为 NaN 时保留旧值，新值为 NaN 时写入新值。

把 contiguous shape 折叠为：

```text
preAxis     = product(indices.shape[0:d])
midAxis     = indices.shape[d]
afterAxis   = product(indices.shape[d+1:r])
dataMidAxis = data.shape[d]
src(pre,mid,after) = (pre * midAxis + mid) * afterAxis + after
dst(pre,mid,after) = (pre * dataMidAxis + indices[src]) * afterAxis + after
```

owned 路线以 `(pre, after)` 为互斥 owner，在单线程内顺序遍历全部 `mid`，因此不同 owner 不会写同一目标地址。

### 重复索引与归约顺序契约

`indices` 在 `dim` 上的值决定目标 middle 坐标；它不会改变 `pre` 和 `after`。因而两个更新只有在 `(pre, after, indices[src])` 三者都相同时才冲突。实现不能把“`indices` 元素不同”当成无冲突的充分条件。

| mode | 公共入口的冲突消解 | 重复目标的可见顺序/数值特性 |
|---|---|---|
| `update` | winner 表对 `winners[dst]` 做 `AtomicMax(linear)`，或 owner 线程串行遍历 `mid` | 两路都以最大 contiguous 源线性位置为最终写者；这个结论依赖单轴替换与 contiguous 线性化不变量。 |
| `add` FP32 | `AtomicAdd` | 不丢失更新，但浮点加法的合并顺序由 atomic 调度决定，不应声称 bitwise deterministic。 |
| `add` FP16 | S1 在 FP32 accumulator 上选 atomic tail1/atomic pre1/owned | 只在 S2 降回 FP16；atomic 变体顺序不固定，owned 变体按 `mid` 固定。 |
| `add` BF16/I32/I64 | owner 串行 `mid` | BF16 每次经 FP32 计算后立即降回 BF16；它不等价于 FP32 中间累加。整数路线不是饱和算术。 |
| `multiply` | 全 dtype 均用 owner；FP32 只特化了前置 SIMT clone | 按 `mid=0..midAxis-1` 左折叠；FP16/BF16 每次乘法后降回原 dtype。 |
| `amax/amin` | int32 amax 可用 atomic 快路，其余用 owner | owned 浮点路线遇到第一个 NaN 后保留该 NaN；非 NaN 相等时选后来值，因此 `+0/-0` 的 bit pattern 可受源顺序影响。 |

所有路线都假设 `0 <= indices[s] < dataMidAxis`。源码中的 `static_cast<uint64_t>(indices[s])` 不是边界处理：负值会变成巨大无符号偏移。若上层不能保证值域，安全实现需先行校验 kernel + 错误标志，而不是在写入 kernel 内静默 clamp 或丢弃，否则会改变算子语义。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `data` | rank 1–8；任意输入 stride，入口转 contiguous | `float16/float32/bfloat16/int32/int64` | 提供输出初值；决定 Device guard 和输出 dtype。 |
| `dim` | 标量 | `int64` Host 值 | 负值先加 rank；归一化后必须在 `[0, rank)`。 |
| `indices` | 与 `updates` shape 完全一致；rank 与 `data` 相同；入口转 contiguous | `int32/int64` | 非 `dim` 轴长度不得大于 `data` 对应轴；源码不检查索引值域。 |
| `updates` | 与 `indices` shape 完全一致；入口转 contiguous | 与 `data` 相同 | 作为散射值。 |
| `reduce` | 可空字符串 | `str?` | 空或缺省为 `update`；还接受 `update/add/multiply/amax/amin`；其他值抛错。 |
| `y` | 与 contiguous `data` shape 相同 | 与 `data` 相同 | 新分配输出；不是原地接口，也不恢复原输入的非连续 stride。 |

### 入口与版本

`cann_bench.scatter` 由 Python 包转发到 Torch 自定义算子。`TORCH_LIBRARY_FRAGMENT` 声明 schema，Meta 实现返回 `empty_like(data)`，PrivateUse1 实现绑定 `scatter_npu`。该入口不是 ACLNN，也不是官方 `ScatterElementsV2` 的转换壳。对非连续 `data`，Meta 的 `empty_like(data)` 可保留原 memory-format，而 PrivateUse1 返回基于 `data.contiguous()` 的连续输出；因此图编译/形状推导不应把 Meta stride 当成 Device 输出 stride 契约。

本卡冻结 CANN `9.1.0`、950（`ascend950`）。交付件顶层构建把 `ascend950` 映射为 Bisheng `dav-3510` 并把 Device TU 编译为 `-xasc`，因此 950 直调卡由交付件自身接线成立。固定 Commit 的官方 `index/scatter_elements_v2` 只用于语义相关性；不把其平台声明、Host selector 或 kernel body归属于本直调实现。

## Host/Launcher 执行流程

### 参数校验与前处理

`scatter_npu` 的顺序是：建立 `OptionalDeviceGuard(data.device())`；检查 rank、三输入 rank、data/update dtype、`dim`、indices/update shape 与非轴维上界；分别把三输入转 contiguous；映射 data/index dtype tag；解析 reduce mode；分配 output、固定 16 MiB byte workspace、可选 winner 或 FP32 accumulator；计算 general/winner/serial 三份参数；选择路线；取得当前 NPU stream 并通过 `OpCommand::RunOpApi("Scatter", acl_call)` 执行 launcher 闭包。

源码没有显式检查三输入是否位于同一设备、索引元素是否在 `[0,data.shape[dim])`、shape 乘积是否溢出 `int64`。这些条件必须由调用者保证，否则 Device 地址计算可能越界。

#### Host checked arithmetic 与安全退化

当把本卡路线落成可靠 Host selector 时，shape 乘积与字节数不能继续沿用源码的未检查 `int64_t` 乘法。最小安全契约是：

```text
checked_product(shape[lo:hi]) -> uint64 value or reject
checked_mul(preAxis, afterAxis) -> ownerCount or reject
checked_mul(midAxis, afterAxis) -> updateStride or reject
checked_mul(dataNum, sizeof(DATA_T)) -> outputBytes or reject
winnerBytes = checked_mul(dataNum, 4)
accBytes    = checked_mul(dataNum, 4)
```

- 任一 shape 为负、乘积超过 tensor/storage 可表示范围，或中间地址公式不可表示时，应在 launch 前拒绝。这不是 route fallback，因为任何 Device route 都无法安全消费已回绕的 POD。
- winner 密度谓词中的 `4*updatesNum` 也必须用溢出安全比较，例如改写为 `ceil(dataNum/4) <= updatesNum`；不能先执行可溢出乘法再判断。
- `updatesNum <= INT32_MAX`、`afterAxis <= UINT32_MAX`和 `midAxis*afterAxis <= UINT32_MAX` 是 winner 紧凑表示 guard；失配可安全退到 owned update。
- winner copy/init 的 32 KiB 别名 TBuf 无法容纳 8192 个 `int64` data 元素。在不改 Device tile 前，Host 应额外要求 `dataTag != INT64`，失配退到 owned update。
- Vector FP16 cast 还需平台 UB 不小于 48 KiB；资源不足时应选 count-guarded SIMT cast。MTE copy 需 32 KiB，没有等价的已闭合 Host 退化，因此应视为平台先决条件或补实 SIMT clone selector。
- `OptionalDeviceGuard(data.device())` 不能代替同设备检查。`indices` 或 `updates` 若在另一设备，其 `.contiguous()` 仍属于原设备，却会把裸指针交给 `data` 的 stream。可靠 Host 必须在取指针前检查三者 device 相同。

### 参数、Workspace 与生命周期

`dataNum=product(dataShape)`，`updatesNum=product(indexShape)`。AIV 核数 `C=GetCoreNumAiv()`，不可用时按 1；`coresByUpdates=min(ceil(updatesNum/512),C)`；`perCoreData=max(ceil(dataNum/C),1024)`；`coresByData=ceil(dataNum/perCoreData)`；最终 `blockDim=max(coresByUpdates,coresByData,1)`。

每次调用持有 output 与 16 MiB byte workspace。公共 `scatter_npu` 生成 `wsPtr`，但闭包没有引用它，也不调用 `launch_scatter`；因此当前公共 launch 图只承担分配成本，不消费这块 workspace。独立 caller 传入 `launch_scatter` 的 workspace 参数也被 wrapper 显式忽略。winner 仅在 winner update 路线分配，生命周期覆盖 copy/init→mark→apply；FP16 accumulator 仅在 FP16 add 路线分配，生命周期覆盖 S0→S1→S2。

闭包 `[=]` 仅捕获裸指针与 POD，**没有**捕获 `winners/accumulator/workspace` Tensor 对象。这些局部 Tensor 因 C++ 作用域至少存活到 `RunOpApi` 返回，此时同流节点已提交；返回后的异步存储生命期依赖 Torch NPU allocator/current-stream 契约，源码没有显式 event、`recordStream` 或 host sync。

### 启动图与失败传播

```text
cann_bench.scatter -> scatter_npu -> same current NPU stream
  update && useWinner
    -> scatter_copy_winner_init_entry
    -> scatter_winner_mark_entry
    -> scatter_apply_entry
  update && !useWinner
    -> scatter_copy_mte_entry -> scatter_owned_entry<UPDATE>
  amax && int32 && preAxis==1 && isPow2(afterAxis)
    -> scatter_copy_mte_entry -> scatter_atomic_max_i32_entry
  amax/amin fallback
    -> scatter_copy_mte_entry -> scatter_owned_entry<AMAX/AMIN>
  add && fp16
    -> half init cast -> half add variant -> half final cast
  add && fp32
    -> scatter_copy_mte_entry -> scatter_fast_add_f32_entry
  add && other dtype
    -> scatter_copy_mte_entry -> scatter_owned_entry<ADD>
  multiply && fp32
    -> scatter_copy_simt_entry -> scatter_owned_entry<MULTIPLY>
  multiply && other dtype
    -> scatter_copy_mte_entry -> scatter_owned_entry<MULTIPLY>
```

独立 Host 入口另外允许：`launch_scatter` 只在 key `1000100` 时启动 general FP32/int32 add；`launch_scatter_mul_float_fused` 启动单个 fused copy+multiply；两个 copy launcher 可单独克隆张量。`scatter_npu` 当前不调用 `launch_scatter` 或 `launch_scatter_mul_float_fused`。

节点均以 `<<<blockDim, nullptr, stream>>>` 同流提交，GM 临时对象是节点间交接介质；`nullptr` 表示没有 dynamic UB。wrapper 返回 `void`，闭包固定返回 0，没有逐 launch 同步错误检查；异步设备错误由运行时边界传播，而不是阻断 wrapper 中后续提交。

### Direct launcher ABI 与失配行为

公共入口在 Host 层已把 data/index tag 限制在合法集合；独立 launcher 本身不是自校验 ABI。不能统一概括为“未知 tag 不 launch”：

| launcher 族 | 非法 data tag | 非 `INT32` index tag | 非法 mode/key |
|---|---|---|---|
| `launch_scatter_copy*` | 不 launch | N/A | N/A |
| `launch_scatter_winner` | copy/apply 可不 launch，但 mark 仍可提交，属部分图而非安全 no-op | 按 int64 实例解读 | N/A |
| `launch_scatter_owned` | 不 launch | 按 int64 实例解读 | 不 launch |
| FP16 serial / FP32 add / int32 amax / fused FP32 multiply | FP16 serial 的非 FP16 data tag 返回；其余 data dtype 由入口固定 | 按 int64 实例解读 | serial 忽略 reduction 实参，因此只能按声明用于 FP16 add |
| `launch_scatter` | 仅 key 参与分发，data/reduction 实参被忽略 | key `1000100` 强制 body 按 int32 index 解读 | key 非 `1000100` 时不 launch |

因此 direct caller 的安全 fallback 必须发生在进入 wrapper 之前：验证 tag/mode/POD/extent，只在可证明的 selector guard 失配时转到 owned/copy 等闭合路线。不得把部分图、错类型解读或 no-launch 当成计算 fallback。

## Route 与参数设计

### 分核与资源计算

所有 SIMT entry 先取 `perCore=ceil(total/GetBlockNum())`，block `b` 覆盖 `[b*perCore,min(total,(b+1)*perCore))`，再由 512 个 VF 线程以 `threadIdx.x + n*blockDim.x` 遍历；atomic max 快路使用 1024 线程。copy MTE 把 `perCore` 向上对齐到 `32/sizeof(DATA_T)` 元素，单 tile 为 `32768/sizeof(DATA_T)` 元素。

general POD 的 `loopLength` 从 `ubRaw` 经 `min(max(ubRaw-131072,0),65536)-640` 后按 32 字节元素数向下对齐，但当前 general SIMT body 不读取该字段。实际 AIV 固定分配分别为 copy 32 KiB、winner copy/init 32 KiB、FP16 vector cast 48 KiB，Host selector 没有按这些常量做 UB capability 检查。

### Route 特征与重要性

下表按阅读与实现复用重要性降序排列；`P0`～`P11` 不是 Host runtime 优先级。真实控制流位置由“Host 顺序”列保留：H0 是入口的 copy/init 前置选择，H1～H5 是 `mode` 的 `if/else if` 短路链，D1～D3 是不经 `scatter_npu` 的独立 launcher 入口。

#### 可达性分层

| 层级 | Route | 可达证据与边界 |
|---|---|---|
| `active` | R-001～R-009 | 都由 `scatter_npu` 的 H0～H5 在某一合法 mode/dtype/shape 组合上选中；R-002/R-004 同时承担 winner/atomic guard 失配的语义 fallback。 |
| `active` 共享节点 / `conditional` 独立入口 | R-012 | MTE copy 是 UPDATE-owned、extrema、FP32 add 和非 FP32 multiply 的 H0 前置节点，SIMT copy 是 FP32 multiply 前置节点；两个 launcher 也可由受控 direct caller 单独调用。 |
| `conditional` | R-010 | `launch_scatter` 只对 key `1000100` 提交 general FP32/I32 ADD。公共 `scatter_npu` 生成 general POD 但不调该 launcher；只有显式 direct caller 在先初始化 output 并满足 POD 安全契约时可达。 |
| `conditional` | R-011 | `launch_scatter_mul_float_fused` 是导出的 FP32/I32-or-I64 单 Kernel launcher；公共 selector 未调用，需 direct caller 显式组图。 |

本 TU 还有两个 `dormant` body，不伪装成上表 Route：

| dormant 节点 | 不可达原因 | 重新启用所需条件 |
|---|---|---|
| `scatter_winner_init_entry -> InitWinners` | `launch_scatter_winner` 已改用 `scatter_copy_winner_init_entry` 融合 clone + winner=-1，没有 wrapper 提交独立 init entry | 需显式 wrapper/launch 节点，并保持 `init -> mark -> apply` 同流顺序；若拆分后仍需另一节点完成 `data -> output` clone。 |
| `scatter_mul_float_entry -> SerialMulFloat` | 没有对应 launch 声明/wrapper；公共 FP32 multiply 实际调用 `scatter_owned_entry<float,...,MULTIPLY>` | 需新增 ABI 声明与 wrapper，在它之前完成 output clone，并证明其与已活跃 owned body 的价值差异。 |

| 重要性 | Route | Route 特征 | 适用域与实现前提 | Host 顺序、短路与 fallback |
|---|---|---|---|---|
| P0 | R-001 winner update | copy/init、winner mark、apply 三节点；用 `AtomicMax` 保留最大线性更新位置 | UPDATE；compact-indexable；满足 dense 或 small-sparse winner 谓词；当前 selector 未排除 INT64，但其 32 KiB init TBuf 不能承载满 8192 元素 INT64 tile | H0 命中 `fusedWinnerInit` 而跳过通用 copy；H1 先判 `useWinner`，命中后短路结束 UPDATE 分支；稳健 Host 需把 INT64 或资源 guard 失配退到 R-002 |
| P1 | R-005 FP16 add | FP16→FP32、FP32 归并、FP32→FP16 三阶段；cast 和 S1 均有 shape guard | ADD + FP16；需 `4*dataNum` 字节 accumulator；Vector cast 需 48 KiB UB | H0 命中 `fusedFp16Add` 而跳过通用 copy；H4 的第一个 dtype 分支，命中后不再检查 R-006/R-007；内部 guard 选 Vector/SIMT 和 tail1/pre1/owned，无跨 Route fallback |
| P2 | R-003 atomic int32 amax | 1024-thread signed `AtomicMax`；2 幂 tail 用位运算化简地址 | AMAX + INT32；`preAxis==1 && IsPow2(afterAxis)` | H0 走 MTE copy；H2 先检查本 fast path，命中后短路 AMAX 分支，未命中转 R-004 |
| P3 | R-002 owned update | `(pre,after)` 互斥 owner 并行，单 owner 按 `mid` 顺序覆盖 | UPDATE 且 R-001 的 winner 谓词失败 | H0 走 MTE copy；H1 内层 `else` fallback，执行后结束 UPDATE 分支 |
| P4 | R-008 FP32 multiply | SIMT clone 避开 MTE copy，随后用 FP32 owned 按源顺序累乘 | MULTIPLY + FP32 | H0 命中 `simtCopyFp32Mul`；H5 进入 MULTIPLY 的 owned launcher；dtype 已决定本 Route，无 Device fallback |
| P5 | R-006 FP32 atomic add | MTE clone 后以 512-thread `AtomicAdd` 直接归并 | ADD + FP32 | H0 走 MTE copy；H4 在 R-005 失配后检查 FP32，命中后不再进入 R-007 |
| P6 | R-004 owned extrema | owner 互斥、`mid` 有序；浮点极值显式传播 NaN | AMIN 的全部支持 dtype，或 AMAX 且 R-003 失配 | H0 走 MTE copy；AMAX 从 H2 内层 `else` fallback 进入，AMIN 从 H3 直接进入；执行后短路后续 mode 分支 |
| P7 | R-007 owned add | 非 FP16/FP32 add 的 owner 归并；BF16 每步经 FP32 再降精度 | ADD + BF16/INT32/INT64 | H0 走 MTE copy；H4 在 R-005、R-006 均失配后进入最终 `else`，无后续 fallback |
| P8 | R-009 owned multiply | MTE clone + owner 互斥累乘 | MULTIPLY + FP16/BF16/INT32/INT64 | H0 未命中 FP32 SIMT-copy 特化而走 MTE copy；H5 与 R-008 共用 owned launcher，dtype 决定本 Route，无后续 fallback |
| P9 | R-012 独立 copy | 五种 dtype 的 MTE tile copy 或 SIMT grid-stride copy；也是主路线的共享前置节点 | caller 提供合法 dtype tag、extent 和同流 output | H0 由主入口在 fused/simt 特化之后选 MTE 或 SIMT copy，该节点不终止后续 Route；D3 可独立调用，未知 tag 不 launch |
| P10 | R-010 通用直调 add | general POD 解码 + FP32 atomic add | 独立 `launch_scatter`；key `1000100`；caller 已初始化 output | D1；不经 H0～H5；key 不等时短路为 no-launch，无 fallback |
| P11 | R-011 融合 FP32 multiply | 单 Kernel 内每 owner 先 clone data-mid，再顺序乘 | 独立 `launch_scatter_mul_float_fused`；FP32 + I32/I64 index | D2；不经 H0～H5；index tag 非 I32 会按 I64 实例解读，故 caller 必须先校验；无 wrapper 内 fallback |

### Route 选择

| 重要性 | Route | Host 顺序 | Key/模板 | 精确选择条件 | 短路/fallback |
|---|---|---|---|---|---|
| P0 | R-001 winner update | H0 fused init→H1 update 首分支 | `useWinner` | 源码实际谓词为 `mode==UPDATE && ((ownerCount<=8192 && dataNum<=4*updatesNum) \|\| (ownerCount<=4 && dataNum<=1100000)) && updatesNum<=INT32_MAX && afterAxis<=UINT32_MAX && midAxis*afterAxis<=UINT32_MAX`；安全落地还要求计算无溢出且当前 Device tile 下 `dataTag!=INT64` | 命中结束 H1；任一选择/安全 guard 失败转 R-002 |
| P1 | R-005 FP16 add | H0 skip copy→H4 add 第 1 dtype 分支 | cast + 三种 S1 | `mode==ADD && dataTag==FLOAT16`；内部先按 `dataNum%8192==0` 选 Vector/Scalar cast，再按 `afterAxis==1 && IsPow2(midAxis)`、否则 `preAxis==1`、否则 owned 选 S1 | 命中结束 H4；内部变体互为 fallback，不落到其他 Route |
| P2 | R-003 atomic int32 amax | H0 MTE copy→H2 amax 首分支 | `AtomicMaxInt32Power2Tail` | `mode==AMAX && dataTag==INT32 && preAxis==1 && IsPow2(afterAxis)` | 命中结束 H2；未命中转 R-004 |
| P3 | R-002 owned update | H0 MTE copy→H1 update fallback | `OwnedScatter<DATA_T,INDEX_T,UPDATE>` | `mode==UPDATE && !useWinner` | H1 最终 `else`；执行后结束，无后续 fallback |
| P4 | R-008 FP32 multiply | H0 SIMT copy→H5 multiply | SIMT copy + `OwnedScatter<float,...,MULTIPLY>` | `mode==MULTIPLY && dataTag==FLOAT32` | H5 的 dtype 特化解读；无 Device fallback |
| P5 | R-006 FP32 atomic add | H0 MTE copy→H4 add 第 2 dtype 分支 | `FastAddF32` | `mode==ADD && dataTag==FLOAT32` | R-005 失配后命中并结束 H4；非 FP32 继续 R-007 |
| P6 | R-004 owned extrema | H0 MTE copy→H2 amax fallback / H3 amin | `OwnedScatter<...,AMAX/AMIN>` | `mode==AMIN`，或 `mode==AMAX` 且 R-003 未命中 | H2 内层 fallback 或 H3 直达；无后续 fallback |
| P7 | R-007 owned add | H0 MTE copy→H4 add 最终 fallback | `OwnedScatter<...,ADD>` | `mode==ADD && dataTag in {BFLOAT16,INT32,INT64}` | R-005/R-006 均失配后命中；无后续 fallback |
| P8 | R-009 owned multiply | H0 MTE copy→H5 multiply | `OwnedScatter<...,MULTIPLY>` | `mode==MULTIPLY && dataTag!=FLOAT32` | H5 的 dtype 通用解读；无 Device fallback |
| P9 | R-012 独立 copy | H0 共享前置 / D3 | `scatter_copy_mte_entry` / `scatter_copy_simt_entry` | 主入口按 fused/simt guard 选 copy，或 caller 独立调用相应 launcher；dtype tag 必须是五种支持值 | H0 copy 后继续 H1～H5；D3 未知 tag 不 launch |
| P10 | R-010 通用直调 add | D1 | key `1000100` | 独立调用 `launch_scatter` 且 `tiling.key==1000100`；该 key 对应非大域、FP32 data、int32 index、ADD | key 不等则 no-launch；无 fallback |
| P11 | R-011 融合 FP32 multiply | D2 | `SerialMulFloatFused<INDEX_T>` | 独立调用 `launch_scatter_mul_float_fused`，caller 保证 FP32 data 与 I32/I64 index | 非 I32 index tag 会当 I64；无 wrapper 内 fallback |

general key 的 Host producer 公式必须和 R-010 的唯一 launcher 分支一起阅读：

```text
suffix = elementBytes
if mode in {ADD, MULTIPLY}: suffix = (dataTag == FLOAT32 ? 0 : 1)
key = 1000000 + 100*mode + (indexTag == INT64 ? 1000 : 0) + suffix
if updatesNum > INT32_MAX or dataNum > INT32_MAX: key += 10000

launch_scatter: key == 1000100 -> GeneralAddF32<int32_t>; otherwise no launch
```

因此 `1000100` 精确表示 ADD + FP32 + I32 index + 两个 extent 均不超过 `INT32_MAX`。`dataDtype/indexDtype/reduction` 虽仍出现在 launcher ABI 中，body 分发不再校验它们；所以 direct caller 必须保证 key 与实际指针类型相符。

general stride producer 只合并 `dim` 之后 data/index shape 相同的连续尾部：

```text
j = 从 rank-1 向前找到的最后一个 dataShape[j] != indexShape[j] 的轴；全同则 j=-1
combAxis = max(j+1, dim+1)
if combAxis <= rank-1:
    tail = product(dataShape[combAxis:rank])
    dataCur  = dataShape[0:combAxis]  + [tail]
    indexCur = indexShape[0:combAxis] + [tail]
    reducedRank = combAxis + 1
else:
    dataCur/indexCur = 原 shape
    reducedRank = rank

stride[i] = product(curShape[i+1:reducedRank]); stride[reducedRank-1] = 1
```

合并不穿过 `dim`，所以 `dim` 字段不需重映射。但 `reducedRank==8` 时 POD 的 7 项 stride 无法容纳末轴；这一 Host producer 安全边界对公共入口也生效，不因 R-010 未被公共 selector launch 而消失。

#### Entry/body/template delta

| Route | entry 与模板实例 | 相对共享 body 的决定性 delta |
|---|---|---|
| R-001 | `copy_winner_init<DATA_T>` + `winner_mark<INDEX_T>` + `apply<DATA_T>`；5 data × 2 index | data 类型只影响 clone/apply，index 类型只影响 mark；冲突协议固定为 int32 winner `AtomicMax(linear)`。 |
| R-005 | S0/S2 固定 half↔float；S1 仅按 `INDEX_T` 实例 I32/I64 | `dataNum%8192` 选 Vector/SIMT cast；`after==1 && pow2(mid)` / `pre==1` / 其他选三个不同 S1 body，但 accumulator 均为 FP32。 |
| R-003 | `AtomicMaxInt32Power2Tail<INDEX_T>`；data/update/output 固定 int32 | 从通用地址解码改为 `afterMask` + `afterShift`，VF 线程数由 512 增为 1024。 |
| R-002/R-004/R-007/R-008/R-009 | `OwnedScatter<DATA_T,INDEX_T,MODE>`；MODE 在 wrapper 降为编译期常量 | owner/`mid` 骨架共享；delta 只是 UPDATE/ADD/MULTIPLY/AMAX/AMIN 的 read-modify-write、浮点 cast/NaN 规则，以及 R-008 前置 SIMT clone。 |
| R-006 | `FastAddF32<INDEX_T>`；data 固定 float | 以 update 元素为并行单元并 `AtomicAdd`，不使用 owner 串行 `mid`。 |
| R-012 | `copy_mte_entry<DATA_T>` 或 `copy_simt_entry<DATA_T>`；5 data 类型 | MTE 变体有 32 KiB TBuf、32B 分块与 pad tail；SIMT 变体无显式 UB，以 count guard 处理尾部。 |
| R-010 | `GeneralAddF32<int32_t>` 单一实例 | 使用 general rank/stride POD 逐轴解码，而非 serial pre/mid/after POD；output 必须由 caller 预初始化。 |
| R-011 | `SerialMulFloatFused<INDEX_T>`；float × I32/I64 | 每个 owner 在同一 VF 线程内先遍历 `dataMidAxis` clone 整个 owner slice，再遍历 `midAxis` 完成乘法，不依赖前置 copy kernel。 |

```text
UPDATE -> winner predicate ? R-001 : R-002
AMAX   -> int32 && pre==1 && pow2(after) ? R-003 : R-004
AMIN   -> R-004
ADD    -> fp16 ? R-005 : fp32 ? R-006 : R-007
MUL    -> fp32 ? R-008 : R-009
direct launch_scatter(key) -> key==1000100 ? R-010 : no-launch
direct fused/copy launchers -> R-011/R-012
```

### 模板/实现执行矩阵

| 重要性 | Route | Host 顺序 | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|---|---|
| P0 | R-001 winner update | H0→H1-first | Host winner predicate | 五种 data、两种 index 都有模板实例；源码 selector 满足 compact/dense 或 small-sparse 即选 | 3 Kernel | `scatter_copy_winner_init_entry<DATA_T>` → `MarkWinners<INDEX_T>` → `ApplyWinners<DATA_T>` | 32 KiB alias TBuf；winner GM；同流；mark 用 `AtomicMax` | count guard 处理空 tile；INT64 实例存在但容量不安全，应由 Host 退 R-002 |
| P1 | R-005 FP16 add | H0→H4-first | cast/S1 selector | data FP16 | 3 Kernel，2×3 个图变体 | Vector 或 SIMT cast；S1 为 tail1 atomic/pre1 atomic/owned；Vector 或 SIMT final | accumulator FP32 GM；vector 节点 48 KiB；atomic/owned | `dataNum%8192` 决定 cast tail 策略；变体内 fallback |
| P2 | R-003 atomic int32 amax | H0→H2-first | pow2 tail | data int32，`pre=1`，after 非零 2 幂 | MTE copy → atomic max | `AtomicMaxInt32Power2Tail<INDEX_T>` | 32 KiB copy；1024-thread `AtomicMax` | 不命中转 R-004 |
| P3 | R-002 owned update | H0→H1-fallback | mode selector | UPDATE 且非 winner | MTE copy → owned | `OwnedScatter<DATA_T,INDEX_T,UPDATE>` | copy 32 KiB；owner 无 atomic | MTE pad tail；owner count 可为 0 |
| P4 | R-008 FP32 multiply | H0-SIMT→H5 | dtype selector | FP32 | SIMT copy → owned | `CopyDataSimt<float>` → `OwnedScatter<float,...,MULTIPLY>` | 无 dynamic UB；同流可见 | count 防空；无 Device fallback |
| P5 | R-006 FP32 atomic add | H0→H4-second | dtype selector | data FP32 | MTE copy → fast add | `FastAddF32<INDEX_T>` | 32 KiB copy；512-thread `AtomicAdd` | count 防空；非 FP32 续行 R-007 |
| P6 | R-004 owned extrema | H0→H2-fallback / H3 | mode selector | AMIN 任意支持 dtype；AMAX 非 R-003 | MTE copy → owned | `OwnedScatter<...,AMAX/AMIN>` | owner 独占；浮点 NaN 分支 | MTE pad tail；无后续 fallback |
| P7 | R-007 owned add | H0→H4-final | dtype selector | BF16/I32/I64 | MTE copy → owned | `OwnedScatter<...,ADD>` | owner 独占；BF16 每步经 FP32 再降精度 | MTE pad tail；无后续 fallback |
| P8 | R-009 owned multiply | H0-MTE→H5 | dtype selector | FP16/BF16/I32/I64 | MTE copy → owned | `OwnedScatter<...,MULTIPLY>` | 32 KiB copy；owner 独占 | MTE pad tail；无 Device fallback |
| P9 | R-012 独立 copy | H0 shared / D3 | dtype dispatch | 五种 dtype | 1 Kernel | MTE 或 SIMT copy | 32 KiB TBuf或无 dynamic UB | H0 继续主 Route；D3 未知 tag 无 launch |
| P10 | R-010 通用直调 add | D1 | `1000100` | caller 提供匹配 POD 与已初始化 output | 1 Kernel | `GeneralAddF32<int32_t>` | 512-thread atomic；workspace 忽略 | 其他 key 无 launch；rank-8 stride 边界见限制 |
| P11 | R-011 融合 FP32 multiply | D2 | direct launcher | FP32 data、I32/I64 index | 1 Kernel | `SerialMulFloatFused<INDEX_T>` | 每 owner 先复制完整 data-mid，再顺序乘；无 UB | owner count 防空；无 fallback |

### 性能与安全 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| winner 紧凑索引 | `updatesNum<=INT32_MAX` 且两个 stride 乘积在 `uint32` 内 | winner 存 `int32` 线性位置，apply 转 `uint32_t` | R-001 | R-002 | `scatter_npu` winner predicate |
| winner 密度 | `(owner<=8192 && D<=4A) \|\| (owner<=4 && D<=1100000)` | winner 表按 dataNum 初始化和扫描 | R-001 | R-002 | `scatter_npu` |
| FP16 Vector cast | `dataNum%8192==0` | Vector entry 每次固定处理 8192 元素，没有尾 mask | R-005 | SIMT cast 变体 | `launch_scatter_serial` |
| FP16 tail1 | `after==1 && mid!=0 && IsPow2(mid)` | 用 `src>>midShift` 避免除法，AtomicAdd 处理重复目标 | R-005 | pre1/owned S1 | `ParallelAddHalfTail1` |
| FP16 pre1 | 上一条件失败且 `pre==1` | 线性 src 只需解码 middle/after | R-005 | owned S1 | `ParallelAddHalfPre1` |
| atomic max 地址简化 | `pre==1 && IsPow2(after)` | `after=src&(after-1)`，dst 用左移 | R-003 | R-004 | `AtomicMaxInt32Power2Tail` |
| MTE 对齐 | 每核范围按 `32/sizeof(T)` 对齐 | `DataCopy` 走整 32B block，末段用 `DataCopyPad` | R-002/3/4/6/7/9/12 | pad tail | `scatter_copy_mte_entry` |
| UB 容量 | 实际平台 UB 至少 32 KiB；Vector cast 至少 48 KiB | 固定 `InitBuffer` | 含 AIV TBuf 的 Route | Host 未检查，资源不足属于 launch/build 边界 | 三个 AIV Vector entry |
| SIMT 线程数 | 普通 512，atomic max 1024 | `LAUNCH_BOUND` 与 `asc_vf_call(dim3(...))` 一致 | SIMT routes | 无运行时 fallback | SIMT helper/entry |

### POD/ABI

下表 ABI 按源码标准宽度字段及自然对齐推导；数组是一项字段，不与其他字段合并。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScatterElementsV2AscTilingData` | `dataStride` | `uint64_t[7]` | offset 0，56 B | 元素 stride | `calc_scatter_tiling` reduced contiguous strides | `GeneralAddF32` | reduced rank=8 时 producer 与 consumer 都需第 8 项，超出 POD 容量 |
| 同上 | `indicesStride` | `uint64_t[7]` | offset 56，56 B | 元素 stride | 同上 | `GeneralAddF32` 解码 | 同上 |
| 同上 | `updatesStride` | `uint64_t[7]` | offset 112，56 B | 元素 stride | 同上 | 当前 body 不读取 | inactive |
| 同上 | `loopLength` | `int64_t` | offset 168 | 元素 | UB 公式 | 当前 body 不读取 | inactive |
| 同上 | `allAxis` | `int64_t` | offset 176 | 元素 | `A=Numel(indices)` | `launch_scatter` 形成 total | 负值钳为 0 |
| 同上 | `dataAxis` | `int64_t` | offset 184 | 元素 | `D=Numel(data)` | 当前 body 不读取 | inactive |
| 同上 | `updatesAxis` | `int64_t` | offset 192 | 元素 | `A` | 当前 body 不读取 | inactive |
| 同上 | `preAxis` | `int64_t` | offset 200 | 元素组 | 固定 1 | 当前 body 不读取 | inactive |
| 同上 | `midAxis` | `int64_t` | offset 208 | 元素组 | 固定 1 | 当前 body 不读取 | inactive |
| 同上 | `afterAxis` | `int64_t` | offset 216 | 元素组 | 固定 1 | 当前 body 不读取 | inactive |
| 同上 | `indicesUsedCoreNum` | `int64_t` | offset 224 | core | 固定 0 | 当前 body 不读取 | inactive |
| 同上 | `indicesNormBlockData` | `int64_t` | offset 232 | 元素 | 固定 0 | 当前 body 不读取 | inactive |
| 同上 | `indicesTailBlockData` | `int64_t` | offset 240 | 元素 | 固定 0 | 当前 body 不读取 | inactive |
| 同上 | `baseS` | `int64_t` | offset 248 | 元素 | 固定 1 | 当前 body 不读取 | inactive |
| 同上 | `baseA` | `int64_t` | offset 256 | 元素 | 固定 1 | 当前 body 不读取 | inactive |
| 同上 | `isDeterministic` | `int64_t` | offset 264 | bool 整数 | 固定 0 | 当前 body 不读取 | inactive |
| 同上 | `rank` | `int16_t` | offset 272 | 轴数 | reduced rank | `GeneralAddF32` loop bound | 有 int16 窄化；入口 rank≤8 |
| 同上 | `dim` | `int16_t` | offset 274 | 轴号 | normalized dim | `GeneralAddF32` | 入口已校验 |
| 同上 | `sortSharedBufSize` | `uint32_t` | offset 276；结构 size/align 280/8 | 字节 | 固定 0 | 当前 body 不读取 | inactive |
| `ScatterLaunchTiling` | `data` | 上述结构 | offset 0，280 B | POD | `calc_scatter_tiling` | `launch_scatter`/general entry | 仅 key 1000100 消费部分字段 |
| 同上 | `key` | `uint64_t` | offset 280 | selector | key 编码 | `launch_scatter` | 未知 key 无 launch |
| 同上 | `blockDim` | `uint32_t` | offset 288 | block | 核数公式 | 所有 launcher launch config | 后有 4 B padding |
| 同上 | `workspaceBytes` | `uint64_t` | offset 296；结构 size/align 304/8 | 字节 | 固定 16 MiB | 当前 Device 不读取 | inactive |
| `ScatterWinnerTilingData` | `updatesNum` | `uint64_t` | offset 0 | 元素 | `Numel(indexShape)` | mark total | winner route需≤INT32_MAX |
| 同上 | `dataNum` | `uint64_t` | offset 8 | 元素 | `Numel(dataShape)` | init/apply total | winner 表 extent |
| 同上 | `dataStride` | `uint64_t[8]` | offset 16，64 B | 元素 stride | full-rank contiguous stride | `MarkWinners` | rank≤8 |
| 同上 | `indexStride` | `uint64_t[8]` | offset 80，64 B | 元素 stride | full-rank contiguous stride | `MarkWinners` | rank≤8 |
| 同上 | `rank` | `int32_t` | offset 144 | 轴数 | shape size | mark axis loop | 1–8 |
| 同上 | `dim` | `int32_t` | offset 148；结构 size/align 152/8 | 轴号 | normalized dim | mark axis replacement | 已校验 |
| `ScatterSerialTilingData` | `preAxis` | `uint64_t` | offset 0 | owner 前缀数 | `ProductRange(0,dim)` | owner/fast/fused helpers | 乘法溢出未检查 |
| 同上 | `midAxis` | `uint64_t` | offset 8 | update 轴长 | `indicesShape[dim]` | mid loop/total | 可为 0 |
| 同上 | `afterAxis` | `uint64_t` | offset 16 | owner 尾元素数 | `ProductRange(dim+1,rank)` | 地址、owner total | 可为 0 |
| 同上 | `dataMidAxis` | `uint64_t` | offset 24 | data 轴长 | `dataShape[dim]` | dst stride | 索引必须小于此值 |
| 同上 | `dataNum` | `uint64_t` | offset 32 | 元素 | `Numel(dataShape)` | cast total | 乘法溢出未检查 |
| 同上 | `afterShift` | `uint32_t` | offset 40 | bit | `FloorLog2(afterAxis)` | atomic max | 仅 2 幂 guard 下有效 |
| 同上 | `midShift` | `uint32_t` | offset 44；结构 size/align 48/8 | bit | `FloorLog2(midAxis)` | FP16 tail1 | 仅 2 幂 guard 下有效 |

### Route 源码闭包

| 重要性 | Route | Host 顺序 | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|---|
| P0 | R-001 | H0→H1-first | winner predicate | `scatter_plugin.cpp::scatter_npu` | winner POD | `launch_scatter_winner` 三节点 | copy/init、mark、apply helpers | scatter op CMake + top-level CMake |
| P1 | R-005 | H0→H4-first | FP16 + cast/S1 guards | `scatter_npu` + serial producer | serial POD | `launch_scatter_serial` 三节点 | four cast bodies + three add helpers | 同上 |
| P2 | R-003 | H0→H2-first | int32/pow2 | `scatter_npu` | serial POD | copy + atomic max launcher | MTE copy + `AtomicMaxInt32Power2Tail` | 同上 |
| P3 | R-002 | H0→H1-fallback | update fallback | `scatter_npu` + `calc_scatter_serial` | serial POD | copy launcher + `launch_scatter_owned` | MTE copy + `OwnedScatter<...,UPDATE>` | 同上 |
| P4 | R-008 | H0-SIMT→H5 | FP32 multiply | `scatter_npu` | serial POD | SIMT copy + owned launcher | `CopyDataSimt` + `OwnedScatter<...,MULTIPLY>` | 同上 |
| P5 | R-006 | H0→H4-second | FP32 add | `scatter_npu` | serial POD | copy + fast-add launcher | MTE copy + `FastAddF32` | 同上 |
| P6 | R-004 | H0→H2-fallback / H3 | extrema fallback | `scatter_npu` + `calc_scatter_serial` | serial POD | copy launcher + `launch_scatter_owned` | MTE copy + `OwnedScatter<...,AMAX/AMIN>` | 同上 |
| P7 | R-007 | H0→H4-final | add fallback | `scatter_npu` + `calc_scatter_serial` | serial POD | copy launcher + `launch_scatter_owned` | MTE copy + `OwnedScatter<...,ADD>` | 同上 |
| P8 | R-009 | H0-MTE→H5 | multiply fallback | `scatter_npu` + `calc_scatter_serial` | serial POD | copy launcher + `launch_scatter_owned` | MTE copy + `OwnedScatter<...,MULTIPLY>` | 同上 |
| P9 | R-012 | H0 shared / D3 | dtype tag | `scatter_npu` / direct caller | scalar total/tag | MTE/SIMT copy launcher | copy entry/helper | 同上 |
| P10 | R-010 | D1 | key 1000100 | `calc_scatter_tiling` / direct caller | launch POD | `launch_scatter` | `GeneralAddF32<int32_t>` | 同上 |
| P11 | R-011 | D2 | direct launcher | direct caller + serial POD | serial POD | fused launcher | `SerialMulFloatFused` | 同上 |

## Kernel 实现

### 公共执行流程

所有 entry 都声明 `KERNEL_TYPE_AIV_ONLY`。MTE/Vector 路线直接在 AIV entry 使用 `TPipe/TBuf/DataCopy/Cast`；SIMT 路线由 AIV entry 调 `asc_vf_call`，普通 helper 以 512 线程执行，atomic max 以 1024 线程执行。SIMT 节点不声明 shared/dynamic UB、warp cooperative group 或 barrier；冲突目标只通过 `AtomicAdd/AtomicMax` 协调。跨 Kernel 依赖由同一 stream 的提交顺序和 GM 对象承载。以下展开按各共享实现族中最高重要性降序排列；同一 body 的 Route 保持合并，族内 Route 及表行仍按 P0～P11 顺序。

### P0 · Route R-001：winner update

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | winner 完整谓词 | `launch_scatter_winner`，否则 R-002 |
| Launcher | `op_kernel/scatter_kernel.cpp` | `launch_scatter_winner` | data/index tag | 三个模板 entry 顺序提交 |

#### Device 数据流

##### 函数级调用链

```text
launch_scatter_winner
  -> scatter_copy_winner_init_entry<DATA_T>: data -> output; sharedBuf -> winners=-1
  -> scatter_winner_mark_entry<INDEX_T> -> MarkWinners -> AtomicMax(winners[dst], linear)
  -> scatter_apply_entry<DATA_T> -> ApplyWinners -> output[dst]=updates[winner]
```

##### 源码等价伪码

```text
for each data tile: shared=data; output=shared; shared=-1; winners=shared
parallel for linear update: decode full-rank coord; replace dim by indices[linear]; atomicMax winner[dst]
parallel for dst in data: if winner[dst]>=0: output[dst]=updates[uint32(winner[dst])]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| data/output GM | `begin+off` | `dataNum` | DATA_T contiguous | copy/init | caller/apply | caller/per call | pad tail |
| winners GM | `dst` | `4*dataNum` B | int32 contiguous | init/mark | apply | Host temp/三节点 | 初值 -1，atomic target |
| sharedBuf UB | 0 | 32768 B | DATA_T 后 alias int32 | copy/init | output/winner GM | entry/tile | 同一 TBuf 顺序覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  shared["sharedBuf · TBuf VECCALC · 32768 B · dataLocal/winnerLocal alias"]
```

##### 字节账与复用顺序

```text
UB_total(init) = 8192 * sizeof(int32) = 32768 B
dynamic_ub(mark/apply) = N/A; SIMT VF 直接访问 GM
workspace_total = 0 B（winner GM 是独立 Host tensor）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| sharedBuf | DataCopyPad data | output DataCopyPad | Duplicate(-1) | 每步 `PipeBarrier<PIPE_ALL>` | DATA_T tile 字节不得超过 32768 B |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init tile | PipeBarrier | GM→UB/output/duplicate | 下一阶段 | alias TBuf | `DataCopyPad` 处理元素 tail |
| mark | AtomicMax | 512 SIMT threads | apply（同流后继 kernel） | winner[dst] | 最大 linear 决定最后写者 |
| apply | 同流顺序 | mark kernel | apply kernel | winner GM | winner<0 保留 data |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | winner 资源与选择谓词 |
| Kernel router | `op_kernel/scatter_kernel.cpp` | `launch_scatter_winner` | dtype/index 两轴实例化 |
| Device body | 同上 | `scatter_copy_winner_init_entry`、`MarkWinners`、`ApplyWinners` | 三阶段、alias、atomic、last-writer |

### P1 · Route R-005：FP16 add 三阶段

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | ADD && FP16 | 分配 FP32 accumulator，跳过 output clone |
| Launcher | `op_kernel/scatter_kernel.cpp` | `launch_scatter_serial` | `dataNum%8192` 与 S1 两级 guard | S0/S1/S2 三节点 |

#### Device 数据流

##### 函数级调用链

```text
S0 data(FP16) -> accumulator(FP32): vector cast if aligned, else SIMT scalar cast
S1 updates(FP16) -> accumulator(FP32): tail1 AtomicAdd / pre1 AtomicAdd / owned sequential add
S2 accumulator(FP32) -> output(FP16): vector CAST_RINT if aligned, else SIMT scalar cast
```

##### 源码等价伪码

```text
acc[:] = float(data[:])
if after==1 && pow2(mid): parallel src, dst=(src>>midShift)*dataMidAxis+index[src], atomicAdd(acc[dst],float(update[src]))
else if pre==1: parallel src, decode after, atomicAdd(acc[index[src]*afterAxis+after],float(update[src]))
else: parallel owner, sequential mid, acc[dst]+=float(update[src])
output[:] = half_rint(acc[:])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| accumulator GM | linear/dst | `4*dataNum` B | FP32 | S0/S1 | S1/S2 | Host temp/三节点 | 不与 output alias |
| halfBuf UB | 0 | 16384 B | half[8192] | S0/S2 copy/cast | cast/output | vector entry/tile | 独立 TBuf |
| floatBuf UB | 0 | 32768 B | float[8192] | cast/copy | output/cast | vector entry/tile | 独立 TBuf |
| SIMT S1 | GM direct | N/A | index/half/float | caller/S0 | atomic或owner | kernel | dynamic UB N/A |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  half["halfBuf · 8192*2 = 16384 B"]:1
  float["floatBuf · 8192*4 = 32768 B"]:2
```

##### 字节账与复用顺序

```text
UB_total(vector S0/S2)=16384+32768=49152 B
dynamic_ub(SIMT S0/S1/S2)=N/A
workspace_total=4*dataNum B（独立 accumulator tensor；固定 16 MiB workspace 仍不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| halfBuf/floatBuf | DataCopy/Cast | Cast/DataCopy | 下一 tile | 每阶段 PipeBarrier | 仅整 8192 元素进入 vector route |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| S0 vector | DataCopyPad→Cast NONE→DataCopyPad | data | accumulator | two TBuf | 精确升 FP32；固定 8192 |
| S1 | AtomicAdd或owner | updates | accumulator | repeated dst | 全程 FP32 accumulate |
| S2 vector | DataCopyPad→Cast RINT→DataCopyPad | accumulator | output | two TBuf | 最后一次降 FP16；固定 8192 |
| scalar fallback | SIMT scalar cast | GM | GM | direct | count guard处理任意尾部 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | accumulator分配与跳过clone |
| Kernel router | `op_kernel/scatter_kernel.cpp` | `launch_scatter_serial` | 2×3 topology selector |
| Device body | 同上 | vector cast entries、scalar cast helpers、三种 half-add helpers | 48 KiB UB、FP32 accumulation、tail |

### P2/P5 · Route R-003/R-006：atomic fast families

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | R-003 int32/pow2；R-006 FP32 add | copy 后对应 atomic launcher |
| Router | `op_kernel/scatter_kernel.cpp` | atomic/fast add launchers | index tag | I32/I64 index实例 |

#### Device 数据流

##### 函数级调用链

```text
scatter_copy_mte_entry
  -> R-003 scatter_atomic_max_i32_entry -> AtomicMaxInt32Power2Tail(1024 threads)
  -> R-006 scatter_fast_add_f32_entry -> FastAddF32(512 threads)
```

##### 源码等价伪码

```text
R-003: after=src&(afterAxis-1); dst=(uint64(index[src])<<afterShift)+after; atomicMax(output[dst],updates[src])
R-006: decode pre/after from contiguous src; dst=(pre*dataMidAxis+uint64(index[src]))*afterAxis+after; atomicAdd(output[dst],updates[src])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| copyBuf UB | 0 | 32768 B | int32/float | copy | output clone | copy entry | single TBuf |
| output GM | 公式 dst | dataNum | int32/float | copy | atomic helper | Host output | atomic conflict domain为相同 dst |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  copy["copyBuf · 32768 B"]
```

##### 字节账与复用顺序

```text
UB_total(copy)=32768 B; dynamic_ub(atomic)=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| copyBuf | DataCopy | DataCopy output | next tile | PipeBarrier | copy-out before overwrite |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| R-003 | `Simt::AtomicMax` | 1024 threads | kernel completion | int32 output | 2幂位运算，无除法 |
| R-006 | `Simt::AtomicAdd` | 512 threads | kernel completion | float output | 重复目标求和次序由 atomic 调度决定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | guard 与优先级 |
| Kernel router | `op_kernel/scatter_kernel.cpp` | `launch_scatter_atomic_max_i32`、`launch_scatter_add_f32_fast` | index分发、线程域 total |
| Device body | 同上 | `AtomicMaxInt32Power2Tail`、`FastAddF32` | 地址化简与 atomic |

### P3/P6/P7/P8 · Route R-002/R-004/R-007/R-009：owned families

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | update fallback、extrema fallback、非 FP16/FP32 add、非 FP32 multiply | MTE copy 后 `launch_scatter_owned` |
| Router | `op_kernel/scatter_kernel.cpp` | `launch_scatter_owned` | reduction→compile-time MODE，data/index tag→模板类型 | `scatter_owned_entry` |

#### Device 数据流

##### 函数级调用链

```text
launch_scatter_copy -> scatter_copy_mte_entry<DATA_T>
launch_scatter_owned -> scatter_owned_entry<DATA_T,INDEX_T,MODE>
  -> OwnedScatter: each thread owns disjoint (pre,after), loops mid in source order
```

##### 源码等价伪码

```text
clone data to output
parallel owner in [0,preAxis*afterAxis):
  decode pre,after
  for mid=0..midAxis-1:
    src=(pre*midAxis+mid)*afterAxis+after
    dst=(pre*dataMidAxis+uint64(indices[src]))*afterAxis+after
    output[dst] = update/reduce(output[dst],updates[src])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| copyBuf UB | 0 | 32768 B | DATA_T view | MTE input | MTE output | copy entry/tile | single TBuf |
| indices/updates GM | `src` | updatesNum | contiguous | caller | owner helper | caller/call | read-only |
| output GM | `dst` | dataNum | DATA_T | copy | owner/caller | Host output | owner-disjoint |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  copy["copyBuf · TBuf VECCALC · 32768 B"]
```

##### 字节账与复用顺序

```text
UB_total(copy) = 32768 B
dynamic_ub(owned) = N/A
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| copyBuf | GM→UB | UB→GM | 下一 tile | 两次 PipeBarrier | 下一 tile 前 copy-out 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| clone | DataCopy/Pad + barrier | data | output | copyBuf/output | 非整 32B 末段 pad API |
| owned reduce | owner partition，无 atomic | 每个 `(pre,after)` 线程 | 同线程下一 mid | output[dst] | FP16/BF16 add/mul 每步转 float 后降回 DATA_T；extrema显式 NaN次序 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | fallback dispatch |
| Kernel router | `op_kernel/scatter_kernel.cpp` | `launch_scatter_copy`、`launch_scatter_owned` | 五 data×两 index×五 MODE lowering |
| Device body | 同上 | `scatter_copy_mte_entry`、`OwnedScatter` | copy tail、owner地址与精度顺序 |

### P4/P9/P10/P11 · Route R-008/R-012/R-010/R-011：其余 Direct/SIMT 路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | FP32 multiply | R-008 SIMT copy + owned |
| Direct | `op_kernel/scatter_kernel.cpp` | copy launchers | caller-selected dtype/extent | R-012 MTE 或 SIMT copy |
| Direct | 同上 | `launch_scatter` | key==1000100 | R-010 general add，否则 no-launch |
| Direct | 同上 | fused launcher | caller-selected FP32/index tag | R-011；非 I32 index tag 会按 I64 实例解读，所以 tag 校验是 caller 契约 |

#### Device 数据流

##### 函数级调用链

```text
R-008 CopyDataSimt<float> -> OwnedScatter<float,...,MULTIPLY>
R-012 scatter_copy_mte_entry<DATA_T> or CopyDataSimt<DATA_T>
R-010 GeneralAddF32<int32_t> -> AtomicAdd(output[decoded dst],updates[linear])
R-011 SerialMulFloatFused -> per owner clone all dataMidAxis -> per owner sequential multiply
```

##### 源码等价伪码

```text
R-008: parallel linear copy; same-stream owner multiply
R-012: per-core MTE tile copy with pad, or per-core SIMT strided copy
R-010: parallel update; full/reduced rank decode; atomicAdd to caller-initialized output
R-011: parallel owner; for data-mid copy self to output; for update-mid multiply indexed output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| SIMT copy | `begin+i` | dataNum | DATA_T GM | data | output | one kernel | no UB |
| MTE copyBuf | 0 | 32768 B | DATA_T view | data | output | tile | 32B alignment/pad |
| general add | decoded dst | allAxis updates | FP32/I32 index | caller | output | one kernel | AtomicAdd |
| fused multiply | owner/data-mid | full dataNum + updatesNum | FP32 | self/updates | output | one kernel | owner disjoint |

##### 分层内存布局

R-008/R-010/R-011 与 SIMT R-012：`N/A`，实际替代路径是 SIMT VF 直接访问 GM；MTE R-012 使用单个 32768 B `copyBuf`。

##### 字节账与复用顺序

```text
UB_total(SIMT/general/fused)=0 B explicit local allocation
UB_total(MTE copy)=32768 B
dynamic_ub=N/A
workspace_total=0 B（launch_scatter 的 workspace 参数被忽略）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| MTE copyBuf | GM→UB | UB→GM | next tile | PipeBarrier | copy-out complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| R-008 | same stream | SIMT copy | owned multiply | output GM | FP32 multiply source order per owner |
| R-012 | barrier或N/A | copy | caller | output | MTE pad或SIMT count guard |
| R-010 | AtomicAdd | update threads | completion | output[dst] | caller必须先初始化 output |
| R-011 | owner partition | per-owner clone | same thread multiply | owner slice | no cross-owner conflict |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | `op_plugin/scatter_plugin.cpp` | `scatter_npu` | R-008选择 |
| Kernel router | `op_kernel/scatter_kernel.cpp` | `launch_scatter*` wrappers | direct入口、key/dtype/index分发 |
| Device body | 同上 | `CopyDataSimt`、`GeneralAddF32`、`SerialMulFloatFused`、copy MTE | 地址、atomic、融合与tail |

## 算子特定优化

### Trick T-001：winner 的 copy/init TBuf 别名融合

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：winner predicate 命中。
- 转换片段定位：`scatter_kernel.cpp:400-440`；相关官方源码：[`scatter_elements.h`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_kernel/arch35/scatter_elements.h)

**交付件源码（连续逐字摘录，不归因于官方映射）**：

```cpp
TBuf<TPosition::VECCALC> sharedBuf;
pipe.InitBuffer(sharedBuf, tileElems * sizeof(int32_t));
auto dataLocal = sharedBuf.Get<DATA_T>();
auto winnerLocal = sharedBuf.Get<int32_t>();
DataCopyPad(dataLocal, srcGm[begin + off], dataCp, dataPad);
PipeBarrier<PIPE_ALL>();
DataCopyPad(dstGm[begin + off], dataLocal, dataCp);
PipeBarrier<PIPE_ALL>();
Duplicate(winnerLocal, static_cast<int32_t>(-1), elems);
```

#### 机制与设计原因

一个 AIV kernel 同时完成 `output=data` 与 `winners=-1`，省去独立 winner-init launch。`dataLocal` 被 output copy 消费后，经过 pipe barrier 才以 `winnerLocal` 视图覆盖同一 32 KiB TBuf，因此不同时物化两块 UB。

#### 迁移边界

- 适用：两个等长 GM 初始化阶段能串行消费同一 tile 的场景。
- 前提：别名区容量覆盖两种视图的最大实际 tile 字节数，并保留 copy-out→overwrite barrier。
- 失效：容量或阶段顺序不成立时必须拆成独立 buffer/kernel；当前 `int64` 满 tile 会超过 32 KiB，不能直接复用该参数。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 `Duplicate` 移到 output copy-out 前会破坏 data tile；对 `int64` 仍按 8192 元素搬入 32 KiB TBuf 会越过分配容量。

### Trick T-003：FP16 add 只在边界 cast，FP32 中间累加

#### 触发与源码依据

- 适用 Route：R-005。
- 触发条件：ADD 且 data dtype 为 FP16。
- 转换片段定位：`scatter_kernel.cpp:963-1004`；相关官方入口：[`scatter_elements_v2_apt.cpp`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_kernel/scatter_elements_v2_apt.cpp)

**交付件源码（连续逐字摘录，不归因于官方映射）**：

```cpp
if (vecCast) {
    scatter_half_init_vec_entry<<<bd, nullptr, stream>>>(self, accumulator, tiling.dataNum);
} else {
    scatter_half_cast_to_f32_entry<<<bd, nullptr, stream>>>(self, accumulator, tiling.dataNum);
}
// S1: reduce updates into accumulator
```

#### 机制与设计原因

S0 直接从 self 生成 FP32 accumulator，避免单独初始化 output；S1 在 FP32 中完成全部更新；S2 仅一次降为 FP16。整 8192 元素 tile 用 Vector Cast，非整 tile用 count-guarded SIMT scalar cast，避免无 tail mask 的固定 Vector 调用。

#### 迁移边界

- 适用：低精度输入但希望把归并中间态保存在 FP32 的 scatter-add。
- 前提：有 `4*dataNum` 字节额外 GM，Vector 变体还需 48 KiB UB。
- 失效：内存预算不足时不能照搬该三阶段图；非 8192 对齐必须走 tail-safe cast。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

逐更新把 FP32 和降回 FP16交替执行会改变舍入顺序；固定源码明确把降精度放在 S2。

### Trick T-002：按 `(pre,after)` owner 消除非必要 atomic

#### 触发与源码依据

- 适用 Route：R-002/R-004/R-007/R-008/R-009/R-011，以及 R-005 owned S1。
- 触发条件：使用 serial POD 的 owner 路线。
- 转换片段定位：`scatter_kernel.cpp:120-178`；相关官方源码：[`scatter_elements_deterministic.h`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_elements_v2/op_kernel/arch35/scatter_elements_deterministic.h)

**交付件源码（连续逐字摘录，不归因于官方映射）**：

```cpp
uint64_t srcBase = pre * td.midAxis * td.afterAxis + after;
uint64_t dstBase = pre * td.dataMidAxis * td.afterAxis + after;
for (uint64_t mid = 0; mid < td.midAxis; ++mid) {
    uint64_t src = srcBase + mid * td.afterAxis;
    uint64_t dst = dstBase + static_cast<uint64_t>(indices[src]) * td.afterAxis;
    DATA_T value = updates[src];
```

#### 机制与设计原因

目标地址的 `pre` 与 `after` 分量不受 index 改写；把每个 `(pre,after)` 交给一个线程后，所有可能冲突的 `mid` 更新都在同一线程按源序完成。这样 update/multiply/extrema及部分 add 不需 atomic，还固定了重复目标的顺序。

#### 迁移边界

- 适用：仅一个轴由 index 替换且布局可折为 contiguous `pre/mid/after`。
- 前提：不同 owner 的目标切片确实不相交，索引值合法。
- 失效：需要跨 owner 的地址变换时必须改用 atomic、winner或其他冲突消解。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若把 `mid` 并行化却仍使用普通 read-modify-write，重复 index 会丢失更新；源码用 atomic fast route 或 owner串行规避该问题。

## 模板与实现对比

| 对比维度 | winner | FP16 add | atomic fast | owned | direct/copy | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/重要性 | P0；update 且 compact+dense/small | P1；FP16 add | P2/P5；int32 pow2 amax / FP32 add | P3/P6/P7/P8；各 mode fallback | P4/P9/P10/P11；FP32 multiply 或 caller 显式入口 | 冲突域、dtype与地址可化简性 |
| 模板与 dtype | 五 data×两 index | half/float accumulator×两 index | 固定 int32或float data | 五 data×两 index×MODE | FP32 + I32/I64 index；copy 五 dtype | 编译期 MODE/type 降低运行时分支 |
| 循环与分核 | 更新并行+输出并行 | cast tile或SIMT；S1三分支 | update并行 | owner并行、mid串行 | SIMT copy、update或owner并行 | 在并行度与冲突消解间选择 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 32 KiB alias TBuf；SIMT无dynUB | vector 48 KiB；SIMT无dynUB | copy 32 KiB；atomic无dynUB | copy 32 KiB；owned无dynUB | fused/general无dynUB；copy可32 KiB | 无 Queue/通信 window |
| 内存总占用与复用 | winner `4D` GM | accumulator `4D` GM + 48 KiB | output + copyBuf | output + copyBuf | 无额外GM或caller管理 | winner/accumulator按路线条件分配 |
| 搬运/Vector/SIMT API | DataCopyPad/Duplicate/AtomicMax | Cast/DataCopyPad/AtomicAdd | AtomicMax/Add | DataCopy/Pad + scalar SIMT | SIMT/DataCopyPad/AtomicAdd或owner融合 | tail和冲突决定 primitive |
| 同步、精度与 tail | pipe + same stream；last winner | pipe/atomic/owner；FP32累加；scalar tail | atomic；copy pad | pipe + same stream；source-order | same stream、same kernel或atomic | 只依赖源码明确协议 |
| fallback/失败边界 | 不满足转 owned | 非对齐转 scalar变体 | guard失败转 owned | 非法 data/mode 可 no-launch，非 I32 index tag 却会当 I64 | general key 不符 no-launch；fused 的非 I32 index tag 当 I64；主 FP32 multiply 无 Device fallback | 不存在隐式通用 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 非本卡范围 | 非本卡范围 | 交付件构建显式接受 `ascend950`；本卡平台为 `950` |
| Host/Launcher | 未提取 | 未提取 | `scatter_npu` 与全部 `launch_scatter*` 共用本卡实现 |
| Kernel/资源 | 未提取 | 未提取 | Bisheng `dav-3510`、`-xasc`；AIV/Vector+SIMT，无 Cube/MIX_CV |

官方 `index/scatter_elements_v2` 在这里只是相关映射。它不是交付件的转换来源；其公开平台声明无论支持与否，都不能覆盖本直调 CMake/launcher 的 950 事实。

## 限制与失败边界

- 索引值域未校验；负 `int32/int64` 被转换为 `uint64`，越界值会形成越界 GM 地址。
- `Product/Numel/ProductRange` 的 `int64` 乘法没有溢出检查，随后还会写入 `uint64` POD。
- R-001 的 `sharedBuf` 固定 32768 B，但 `tileElems=8192`；`DATA_T=int64` 的满 tile需要 65536 B。这是源码可见容量缺口，int64 winner 路线不能按当前参数安全复用。
- general POD 只有 7 项 stride。reduced rank 保持 8 时，`calc_scatter_tiling` 就会执行 `dataStride[7]=indicesStride[7]=updatesStride[7]=1`，随后 R-010 body 也会读 axis 7。虽然公共 `scatter_npu` 不调用 R-010，它在所有路线选择前都会生成 general tiling，所以这是公共 Host producer 边界，不只是独立 direct body 边界。安全实现应把数组扩到 8 并同步 ABI，或在生成前保证尾维压缩后 `reducedRank<=7`；不能仅依赖“不 launch R-010”。
- Host 未按固定 32 KiB/48 KiB `InitBuffer` 做 UB capability guard；950 资源必须满足实际 entry 分配。
- 固定 16 MiB workspace 每次分配但所有已闭合 Device body均不消费；`launch_scatter` 也忽略 workspace 参数。
- R-010 只有 key `1000100` 有实例，且不负责 `data→output` 初始化；其他 key静默 no-launch。R-011/R-012 由 caller 负责 dtype/POD/extent 一致性；尤其 R-011 的非 I32 index tag 会被当作 I64，而非 no-launch。
- wrapper 不检查每个异步 launch的即时返回，`acl_call` 固定返回 0；设备错误依赖运行时后续边界暴露。
- 本地固定仓没有给定官方 Commit 对象，因而未从该对象摘录 `index/scatter_elements_v2` 技术事实；只保留带 Commit 的相关映射指针。此缺口不影响以交付件闭合的 Direct Invoke Route。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| Python入口 | `cann_bench/__init__.py` | `scatter` | Python→Torch custom op |
| Torch声明/Host实现 | `csrc/ops/scatter/op_plugin/scatter_plugin.cpp` | schema、`scatter_meta`、`scatter_npu` | 校验、临时对象、selector、launch graph |
| Launcher/POD声明 | `csrc/ops/scatter/op_kernel/scatter_launch.h` | 全部 `launch_scatter*` | Direct Invoke ABI |
| POD | `csrc/ops/scatter/op_kernel/scatter_tilingdata.h` | 三类 TilingData、tag/mode | 字段、位宽、布局 |
| Host producer/Kernel | `csrc/ops/scatter/op_kernel/scatter_kernel.cpp` | `calc_scatter_*`、全部 wrapper/entry/helper | 参数、Route、AIV/SIMT/atomic/body |
| 算子接线 | `csrc/ops/scatter/CMakeLists.txt` | `register_direct_launch_op` | Device/Host TU 注册 |
| 平台构建 | `CMakeLists.txt` | `NPU_ARCH`、`BISHENG_NPU_ARCH`、targets | ascend950→dav-3510 与共享库 |

官方相关映射固定为 `cann/ops-nn@955f8b33985b6d72989ebff341bb9c2c564a51bc:index/scatter_elements_v2`，不作为上表交付件实现链路的一环。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | contiguous pre/mid/after、winner full strides、general reduced strides | 字段顺序、位宽、自然对齐、rank数组容量 | 三类 TilingData |
| 调用顺序 | clone/init→reduce/apply；FP16 S0→S1→S2 | 同流 producer→consumer顺序 | `scatter_npu`、launch wrappers |
| workspace/内存 | winner与accumulator按路线独立分配；copy/cast固定 TBuf | extent、alias barrier、32/48 KiB容量 | Host temp分配与 `InitBuffer` |
| 同步 | MTE/Vector用 PipeBarrier；跨kernel同stream；冲突用atomic或owner | 不得提前覆盖alias TBuf或打乱同流节点 | copy/init/cast/atomic helpers |
| 精度与尾块 | FP16 add在 FP32 累加后一次 RINT；owned浮点按源码每步cast；MTE pad | cast顺序、NaN规则、atomic顺序、tail guard | `OwnedScatter`、R-005 entries |
| 失败边界 | key/tag 失配可能 no-launch、部分 launch 或当作 I64；index/overflow/UB 由 Host/caller 保证 | 不能把静默 no-launch、错类型解读说成通用 fallback | launch wrappers与 Host 校验 |