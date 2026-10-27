---
type: Operator
title: "Xlog1py 950 ACLNN 算子知识"
description: "说明 aclnnXlog1py 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/docs/aclnnXlog1py.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_api/aclnn_xlog1py.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_api/xlog1py.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_host/arch35/xlog1py_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_host/arch35/xlog1py_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/arch35/xlog1py_tiling_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/arch35/xlog1py_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/xlog1py_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/arch35/xlog1py_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_host/xlog1py_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_host/config/ascend950/xlog1py_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T14:37:43Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Xlog1py 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnXlog1pyGetWorkspaceSize` + `aclnnXlog1py`，公开 ACLNN 两段式接口，无其它公开版本族 |
| 计算语义 | `z_i = 0`（`x_i == 0`），否则 `z_i = x_i * log(1 + y_i)`；`y_i` 为 NaN 时返回 `y_i` |
| 输入输出 | `x/y -> z`，ND，FLOAT/FLOAT16/BFLOAT16，同 dtype；`x/y` 可 broadcast，公开维数上限 8 |
| 执行类型 | 单 Kernel：`AIV_VECTOR`，输入搬运使用 NDDMA `MultiCopyParams`，计算使用 `__simd_vf__`/Register API；无 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001/R-002：归一化 rank `<=4` 的 FP32/FP16-BF16；R-003/R-004：归一化 rank `5..8` 的 FP32/FP16-BF16 |
| 核心资源 | GM 输入输出、3 或 4 个 VECCALC `TBuf`、寄存器张量/Mask、MTE2/V/MTE3 event；Xlog1py 节点声明 16 MiB workspace 但 Kernel 参数未消费 |
| 来源闭包 | `source_closed`：公开 ACLNN、L0 launcher、950 Host/Tiling/POD、rank router、AIV Kernel body、NDDMA/Vector helper 与构建接线均闭合；`Contiguous/ViewCopy` 为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 broadcast 后的每个输出坐标，Kernel 先计算 `log(1+y)` 与 `x*log(1+y)`，再用 `x==0` 的 mask 选择零，最后用 `y==y` 的 mask 检测 NaN：非 NaN 取前述结果，NaN 取原始 `y`。因此最终语义为：

```text
mul = x * ln(y + 1)
res0 = (x == 0) ? 0 : mul
z = (y == y) ? res0 : y
```

`x/y` 的逻辑坐标按尾对齐 broadcast；Host 将 size-1 维 stride 预计算为 0，NDDMA 据此把广播值展开到连续 UB tile。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | ND，0..8 维；可非连续 | FLOAT/FLOAT16/BFLOAT16 | 与 `y` 可 broadcast；与 `y/z` 同 dtype；允许空 Tensor；禁止 private format |
| `y` | ND，0..8 维；可非连续 | 同 `x` | `log1p` 自变量；NaN 输出保留该 NaN；允许空 Tensor |
| `z` | ND，公开文档为 1..8 维；可非连续 | 同 `x` | shape 应等于 `x/y` broadcast shape；L0 先生成连续临时结果，再由 `ViewCopy` 写入 `z` |
| `workspaceSize` | 标量 | `uint64_t` | 第一段返回 executor 聚合 workspace；空输入时固定为 0 |
| `executor` | Host 对象 | `aclOpExecutor*` | 第一段构造并移交，第二段在指定 stream 执行 |
| `workspace`/`stream` | Device 地址/运行流 | `void*`/`aclrtStream` | 第二段交给 `CommonOpExecutorRun`；生命周期由调用者与 executor 框架管理 |

### 入口与版本

固定版本只公开 `aclnnXlog1pyGetWorkspaceSize` 和 `aclnnXlog1py`。`README.md` 与接口文档均明确 Ascend 950PR/950DT 支持，A2/A3 不支持；L0 `IsAiCoreSupport` 进一步要求 `NpuArch::DAV_3510`。OpDef 只注册 `ascend950`，binary config 只含 FLOAT16/FLOAT32/BF16，CMake 只设置 `SUPPORT_COMPUTE_UNIT "ascend950"` 与 `arch35` tiling。因此本卡只覆盖 950，不将 A2/A3 实现信号补齐。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段创建 `aclOpExecutor`，失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. `CheckParams` 依次检查 `x/y/z` 非空指针、三者 dtype 一致且属于 FLOAT/FLOAT16/BFLOAT16、storage format 非 private、维数不超过 8。
3. 若 `x` 或 `y` 为空，直接返回 `workspaceSize=0` 与空计算 executor；此路径不做 broadcast 推导，也不启动 Xlog1py Kernel。
4. 非空时依次加入 `Contiguous(x)`、`Contiguous(y)`；这是框架边界，固定算子目录未展开其可能的 no-op/搬运实现。
5. L0 `Xlog1py` 使用 `BroadcastInferShape` 推导结果 shape，以 `x` dtype 分配连续临时输出，再检查 DAV_3510 与 dtype，最后通过 `ADD_TO_LAUNCHER_LIST_AICORE(Xlog1py, OP_INPUT(x,y), OP_OUTPUT(out))` 加入核心节点。
6. `ViewCopy(opResult,z)` 加入最终写回步骤；其非连续输出处理属于框架边界。任一步返回空指针都会终止第一段。

### 参数、Workspace 与生命周期

核心 Tiling 从平台解析得到 AIV core 数 `coreNum` 和每核 UB 字节 `ubSize`。FLOAT 使用 `phys_nodes=3`，FLOAT16/BFLOAT16 使用 `phys_nodes=4`；每个物理 UB buffer 的容量为 `P = floor((ubSize/phys_nodes)/32)*32` 字节。Tiling 无条件给 Xlog1py 节点登记 `16*1024*1024` 字节 workspace，但 `xlog1py` entry 的 `workspace` 参数未被 Kernel body 读取；这 16 MiB 是 executor 管理的节点 reservation，不是 Kernel 的中间数据边。第一段最终返回 `uniqueExecutor->GetWorkspaceSize()`，其中还可包含 `Contiguous/ViewCopy` 的框架 workspace，固定算子目录未公开其聚合公式。

executor 在第一段 `ReleaseTo(executor)` 后由调用者持有；workspace 由调用者在 Device 分配并保持到第二段完成；`CommonOpExecutorRun` 将任务提交到传入 stream，同一 executor 内节点的顺序和失败短路由框架维护。

### 启动图与失败传播

```text
aclnnXlog1pyGetWorkspaceSize
  -> Contiguous(x)                       # 框架边界，可能 no-op/搬运
  -> Contiguous(y)                       # 框架边界，可能 no-op/搬运
  -> l0op::Xlog1py
       -> Xlog1py Tiling                 # POD、key、blockDim、16 MiB reservation
       -> xlog1py<4|8>(x_c, y_c, tmp_z, workspace, tiling)
  -> ViewCopy(tmp_z, user_z)             # 框架边界
aclnnXlog1py(workspace, size, executor, stream)
  -> CommonOpExecutorRun                 # 按 executor 顺序提交
```

核心 launch graph 是单 Device 节点。其 entry 为 `xlog1py<RANK>`，`blockDim=tiling.multicore.num_cores`，输入为两个 contiguous GM Tensor，输出为 L0 分配的连续 GM Tensor，POD 为 `Xlog1pyTilingData<4|8>`。`workspace` 虽作为参数传入却无 consumer；核心节点不存在 producer→consumer workspace 边。核心节点失败会阻断后续 `ViewCopy`；框架内部错误清理细节在固定算子目录不可见。

## Route 与参数设计

### 分核与资源计算

Host 将输入/输出 storage shape 左侧补 1 到共同 raw rank，再删除所有输入和输出都为 1 的维；若全部删除则归一化 shape 为 `[1]`。每维 `max_bro_shape[d]` 取所有 padded shape 的最大值，size-1 输入维的 stride 为 0，其它维 stride 为后缀乘积。

设 `U=ubSize`，`N=3`（FP32）或 `4`（FP16/BF16），`P=floor((U/N)/32)*32` 字节，`C=P/4` 个 FP32 计算元素。从最内维向外扫描，维护 `inner=product(max_bro_shape[k+1:])`：首个满足 `max_bro_shape[k]*inner>C` 的轴成为 split axis，`a_i=floor(C/inner)`、`a_o=ceil(dim[k]/a_i)`、`a_i_tail=(dim[k] mod a_i) or a_i`；若全域可装入则 `axis=0,a_i=dim[0],a_o=1`。总 tile 数为 `product(dim[0:axis])*a_o`，`num_cores=min(total_tiles,coreNum)`，每核主 tile 数 `tiles_main=floor(total_tiles/num_cores)`，前 `cores_tail=total_tiles mod num_cores` 个核多处理一个 tile。

归一化 rank `<=4` 映射到 POD/Kernel rank 4，左补 1；rank `5..8` 映射到 rank 8。每 tile 有效计算数 `count=a_i_seg*product(max_bro_shape[axis+1:])`，最后 split tile 使用 `a_i_tail`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 rank4 FP32 | `XLOG1PY_RANK_4`; `Xlog1pyKernel<float,4>` | DAV_3510，非空，dtype FLOAT，归一化 rank `<=4` | dtype 半精度转 R-002；rank 大于 4 转 R-003/R-004 |
| 2 | R-002 rank4 FP16/BF16 | `XLOG1PY_RANK_4`; `Xlog1pyKernel<half\|bfloat16_t,4>` | DAV_3510，非空，dtype FLOAT16 或 BF16，归一化 rank `<=4` | rank 大于 4 转 R-004；其它 dtype 失败 |
| 3 | R-003 rank8 FP32 | `XLOG1PY_RANK_8`; `Xlog1pyKernel<float,8>` | DAV_3510，非空，dtype FLOAT，归一化 rank `5..8` | 半精度转 R-004；rank 超 8 在 ACLNN 检查失败 |
| 4 | R-004 rank8 FP16/BF16 | `XLOG1PY_RANK_8`; `Xlog1pyKernel<half\|bfloat16_t,8>` | DAV_3510，非空，dtype FLOAT16 或 BF16，归一化 rank `5..8` | 其它 dtype 或 rank 超 8 失败 |

```text
if x.empty || y.empty: return success without Device route
require DAV_3510 and dtype in {FLOAT,FLOAT16,BF16}
normalize_and_squeeze_shapes()
if normalized_rank <= 4:
    key = XLOG1PY_RANK_4
    route = FLOAT ? R-001 : R-002
else:
    key = XLOG1PY_RANK_8
    route = FLOAT ? R-003 : R-004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `XLOG1PY_RANK_4` | DAV_3510，FLOAT，normalized rank 1..4 | 单 AIV `xlog1py<4>` | `Xlog1pyKernel<float,4>::ProcessFP32` + `Xlog1pyVF<float>` | 3P UB；4D NDDMA input；MTE2→V→MTE3 | `a_i_tail` + VF mask；无 fallback |
| R-002 | 同上 | DAV_3510，FLOAT16/BF16，normalized rank 1..4 | 单 AIV `xlog1py<4>` | `Xlog1pyKernel<T,4>::ProcessFP16`（两种 T 共用） | 4P UB；输入升 FP32、结果 RINT 降精度 | 同上 |
| R-003 | `XLOG1PY_RANK_8` | DAV_3510，FLOAT，normalized rank 5..8 | 单 AIV `xlog1py<8>` | `Xlog1pyKernel<float,8>::ProcessFP32` | 3P UB；最多 5D NDDMA，必要时外层拆分多次 copy | 同上 |
| R-004 | 同上 | DAV_3510，FLOAT16/BF16，normalized rank 5..8 | 单 AIV `xlog1py<8>` | `Xlog1pyKernel<T,8>::ProcessFP16` | 4P UB；最多 5D NDDMA 外层拆分；FP32 计算 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 | `npuArch == DAV_3510` | binary、tiling 与 Kernel 只接线 950 | 全部 | L0 返回 `nullptr`，第一段报错 | `op_api/xlog1py.cpp::IsAiCoreSupport` |
| dtype | FLOAT/FLOAT16/BF16 且三者一致 | binary 实例和 buffer/cast 路线只有这三类 | 全部 | ACLNN 参数失败 | `CheckDtypeValid`/`GetShapeInfo` |
| rank key | `mapped=(rank_<=4)?4:8` | POD 数组和 Kernel 模板 rank 固定为 4/8 | R-001/2 vs R-003/4 | raw rank >8 在 ACLNN 失败 | `Xlog1pyTiling::RunTiling` |
| UB 分片 | `count <= P/4` | VF 始终以 FP32 计算，float buffer 容量是 `P/4` 元素 | 全部 | 向外选择 split axis；无独立 fallback | `FindSplitAxis` |
| NDDMA 维数 | `ND=min(RANK,5)` | `MultiCopyParams` 单次最多覆盖模板中的 4D/5D | R-003/4 | `RANK-axis>5` 时外层循环拆成多次 NDDMA | `Init`/`CopyInOne` |
| VF repeat 窄化 | `repeatTimes=CeilDivision(count,VL)` 后写 `uint16_t`；`count` 写 `uint32_t` | VF ABI 位宽固定 | 全部 | 固定源码没有显式溢出 guard/fallback | `CallXlog1pyVF` |
| tile 复用 | 下一 tile 前等待 `MTE3_MTE2` | 单 buffer、无 double buffer，必须等前一输出 DMA 后再覆盖 | 全部 | 无 fallback，串行复用 | `Xlog1pyKernel::Process` |

### POD/ABI

所有成员都是 `int64_t`，自然对齐 8 字节；固定定义没有序列化宏或显式 pack。按标准布局可得 `Xlog1pyTilingData<4>` 为 320 B、`Xlog1pyTilingData<8>` 为 544 B。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `SplitResult` | `axis` | `int64_t`/64 | size 32, align 8, offset 0 | 维索引 | `FindSplitAxis`; `DoTilingAndSet` 再加 `delta` | split、坐标、NDDMA | 0..R-1 |
| `SplitResult` | `a_i` | 同上 | offset 8 | 元素 | `per_buf_elems/inner` 或整轴长 | tile 长度/坐标/NDDMA | 至少 1 |
| `SplitResult` | `a_o` | 同上 | offset 16 | tile 数 | `ceil(dim/a_i)` | total tile 与 flat 解码 | 至少 1 |
| `SplitResult` | `a_i_tail` | 同上 | offset 24 | 元素 | `dim%a_i`，整除取 `a_i` | 最后 split tile | 1..a_i |
| `MultiCoreResult` | `num_cores` | `int64_t`/64 | size 32, align 8, offset 0 | AIV core | `min(total_tiles,coreNum)` | Host `SetBlockDim` 间接消费 | 1..coreNum |
| `MultiCoreResult` | `total_tiles` | 同上 | offset 8 | tile | `outer_prod*a_o` | 仅日志；Device 不读 | 非空路径 >=1 |
| `MultiCoreResult` | `tiles_main` | 同上 | offset 16 | tile/core | `total_tiles/num_cores` | `GetCoreRange` | >=1 |
| `MultiCoreResult` | `cores_tail` | 同上 | offset 24 | core 数 | `total_tiles%num_cores` | 前若干核多一 tile | `<num_cores` |
| `Xlog1pyTilingData<R>` | `split` | `SplitResult`/256 | offset 0 | 复合 | `FindSplitAxis`/delta | Kernel 全流程 | 见上 |
| 同上 | `multicore` | `MultiCoreResult`/256 | offset 32 | 复合 | `MultiCoreSplit` | `Process/GetCoreRange` | 见上 |
| 同上 | `rank` | `int64_t`/64 | offset 64 | 维数 | `rank_` | Device 未读取 | 1..8 |
| 同上 | `per_buf_bytes` | 同上 | offset 72 | 字节 | `floor((ubSize/phys_nodes)/32)*32` | `InitBuffer` | 32 B 对齐 |
| 同上 | `max_bro_shape[R]` | `int64_t[R]` | offset 80；R4 32 B/R8 64 B | 元素/维 | 左补 1 后写 normalized shape | inner、坐标、NDDMA loop | 每维正值（非空 ACLNN） |
| 同上 | `num_inputs` | `int64_t`/64 | R4 offset 112；R8 offset 144 | 个数 | normal input 数，固定 2 | Device 未读取 | 固定 2 |
| 同上 | `num_outputs` | 同上 | R4 offset 120；R8 offset 152 | 个数 | normal output 数，固定 1 | Device 未读取 | 固定 1 |
| 同上 | `input_shapes[2][R]` | `int64_t` 数组 | R4 offset 128；R8 offset 160 | 元素/维 | normalized input shape，未用槽填 1 | 当前 Device body 未读取 | 两个槽均活动 |
| 同上 | `input_strides[2][R]` | `int64_t` 数组 | R4 offset 192；R8 offset 288 | 元素 | size-1 维 0，否则后缀乘积 | GM offset/NDDMA stride | broadcast 依赖 0 stride |
| 同上 | `output_shapes[1][R]` | `int64_t` 数组 | R4 offset 256；R8 offset 416 | 元素/维 | normalized output shape | CopyOut count | 一个活动槽 |
| 同上 | `output_strides[1][R]` | `int64_t` 数组 | R4 offset 288；R8 offset 480 | 元素 | 后缀乘积，size-1 维 0 | GM output offset | 输出按连续 shape 生成 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `XLOG1PY_RANK_4` + FLOAT binary | `GetShapeInfo/DoTilingAndSet<4>` | `Xlog1pyTilingData<4>` | `xlog1py<4>` | `Xlog1pyKernel<float,4>::ProcessFP32`, `Xlog1pyVF<float>` | CMake ascend950/arch35 + FLOAT32 binary |
| R-002 | 同 key + FLOAT16/BF16 binary | 同上，`phys_nodes=4` | 同 POD | `xlog1py<4>` | `Xlog1pyKernel<T,4>::ProcessFP16` | FLOAT16/BF16 binary |
| R-003 | `XLOG1PY_RANK_8` + FLOAT binary | `GetShapeInfo/DoTilingAndSet<8>` | `Xlog1pyTilingData<8>` | `xlog1py<8>` | `Xlog1pyKernel<float,8>::ProcessFP32` | FLOAT32 binary |
| R-004 | 同 key + FLOAT16/BF16 binary | 同上，`phys_nodes=4` | 同 POD | `xlog1py<8>` | `Xlog1pyKernel<T,8>::ProcessFP16` | FLOAT16/BF16 binary |

## Kernel 实现

### 公共执行流程

`xlog1py<RANK>` 将 `x/y` 和 `z` 组成 GM 指针数组，声明 `KERNEL_TYPE_AIV_ONLY`，按模板 rank 解析 POD，再构造对应 `Xlog1pyKernel<DTYPE_X,RANK>`。`Init` 绑定 GM，分配 TBuf，构造每个输入的 NDDMA loop/stride；`Process` 用 `GetBlockIdx` 取得连续 tile 区间。每 tile 依次 NDDMA 读 `x/y`、等待 MTE2→V、执行 FP32 VF 链、等待 V→MTE3、以单 block `DataCopyPad` 回写。tile 之间用 MTE3→MTE2 event 保护同一组 UB buffer。核心 launch graph始终只有一个 AIV Kernel；NDDMA 是该节点内部搬运阶段，不是第二个 Kernel。

### Route R-001：rank4 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0 | `op_api/xlog1py.cpp` | `IsAiCoreSupport` | DAV_3510 且 FLOAT | 加入 AICore launcher |
| Tiling | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `RunTiling` | normalized `rank_<=4` | `DoTilingAndSet<4>` + `XLOG1PY_RANK_4` |
| Router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<4>` | `RANK==4`, binary `DTYPE_X=float` | `Xlog1pyKernel<float,4>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(Xlog1py)
  -> xlog1py<4>(x,y,z,workspace,tiling)
    -> Xlog1pyKernel<float,4>::Init
    -> Xlog1pyKernel<float,4>::Process
      -> CopyInOne(x) -> CopyInOne(y)
      -> CallXlog1pyVF -> Xlog1pyVF<float>
      -> CopyOutOne(z)
```

##### 源码等价伪码

```text
core [start,end) = balanced_tiles(blockIdx, tiles_main, cores_tail)
for flat in [start,end):
  seg = last_split_tile(flat) ? a_i_tail : a_i
  count = seg * product(shape[axis+1:])
  coord = decode(flat); coord[axis] = (flat % a_o) * a_i
  if not first: wait previous MTE3 -> next MTE2
  NDDMA x[sum(coord*stride_x)] -> bufX; NDDMA y[...] -> bufY
  event MTE2 -> V
  VF by VL lanes: (y+1) -> ln -> *x -> x==0 select -> y-is-NaN select
  event V -> MTE3
  DataCopyPad bufZ -> z[sum(coord*stride_z)], blockLen=count*4
  if not last: set MTE3 -> next MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `sum(coord[d]*input_stride[d])` | 输入 storage | float/ND | caller | NDDMA | caller；全 Kernel | stride 0 实现 broadcast |
| `buf_[0]/B_X` | UB 子区 | P B | float/连续 tile | NDDMA x | VF | pipe；逐 tile | depth 1 |
| `buf_[1]/B_Y` | UB 子区 | P B | float | NDDMA y | VF | 同上 | depth 1 |
| `buf_[2]/B_Z` | UB 子区 | P B | float | VF | MTE3 | 同上 | depth 1 |
| GM z | `sum(coord[d]*output_stride[d])` | `count*4` B/tile | float/ND | MTE3 | ViewCopy | executor 临时 Tensor | 单 block、有效 tail |
| workspace | entry 参数 | Xlog1py node 16 MiB reservation | opaque | executor | 无 Kernel consumer | 第二段执行期 | 未使用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  bx["buf0 B_X · depth=1 · P bytes"]
  by["buf1 B_Y · depth=1 · P bytes"]
  bz["buf2 B_Z · depth=1 · P bytes"]
```

##### 字节账与复用顺序

```text
UB_total = P + P + P = 3P
P = floor((U/3)/32)*32
UB_total <= U
dynamic_ub = N/A（AIV TBuf）
workspace_total(core node reservation) = 16 MiB；Kernel 实际读取 = 0 B
=> count <= P/4 float elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三个 `buf_` | NDDMA/VF | VF/MTE3 | 下一 tile | MTE2_V、V_MTE3、跨 tile MTE3_MTE2 | 无 double buffer，必须串行 event |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input→VF | `MTE2_V` | 两次 NDDMA | VF 前 wait | B_X/B_Y | float 原精度；broadcast 已展开 |
| VF | Register mask | `UpdateMask<float>(count)` | 所有 VF primitive | 最后 repeat lane | `repeatTimes=ceil(count/VL)` |
| VF→output | `V_MTE3` | StoreAlign | CopyOut 前 wait | B_Z | 只回写 `count` 有效元素 |
| tile reuse | `MTE3_MTE2` | 上一 CopyOut | 下一 CopyIn 前 wait | 全部单缓冲 | split tail 用 `a_i_tail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `DoTilingAndSet<4>` | P、split、分核、POD、blockDim |
| Kernel entry/router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<4>` | rank4 POD 到 `Xlog1pyKernel<DTYPE_X,4>` |
| Device body/helper | `op_kernel/arch35/xlog1py_kernel.h` | `ProcessFP32/CopyInOne/Xlog1pyVF` | 4D NDDMA、FP32 VF、event 与 tail |

### Route R-002：rank4 FP16/BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0 | `op_api/xlog1py.cpp` | `IsAiCoreSupport` | DAV_3510，dtype FLOAT16/BF16 | 对应 binary launcher |
| Tiling | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `GetShapeInfo/RunTiling` | `phys_nodes=4` 且 rank `<=4` | `DoTilingAndSet<4>` |
| Router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<4>` | `DTYPE_X` 为半精度类型 | `Xlog1pyKernel<T,4>` |

#### Device 数据流

##### 函数级调用链

```text
xlog1py<4> -> Xlog1pyKernel<T,4>::Process
  -> CopyInOne(x -> B_TEMP_X), CopyInOne(y -> B_TEMP_Y)
  -> Cast T->float into B_X/B_Y
  -> CallXlog1pyVF(dst=B_X, srcX=B_X, srcY=B_Y)
  -> Cast float->T(CAST_RINT) into B_TEMP_X
  -> CopyOutOne(B_TEMP_X -> z)
```

##### 源码等价伪码

```text
for assigned tile:
  NDDMA T x/y into temp buffers; wait MTE2_V
  cast x,y to float
  fused VF in-place on float-x buffer
  cast result to T using CAST_RINT, reusing temp-x buffer
  wait V_MTE3; copy count*sizeof(T) to GM
  guard next tile with MTE3_MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| B_TEMP_X (`buf0`) | UB | P B | T | NDDMA x；后为 downcast | Cast；MTE3 | pipe/tile | 输入后复用为输出 |
| B_X (`buf1`) | UB | P B | float | Cast x；后 VF | VF/downcast | pipe/tile | VF 原地 dst=srcX |
| B_Y (`buf2`) | UB | P B | float | Cast y | VF | pipe/tile | depth 1 |
| B_TEMP_Y (`buf3`) | UB | P B | T | NDDMA y | Cast y | pipe/tile | Cast 后不再读 |
| GM x/y/z | stride 坐标 | 每 tile输入/输出有效数 | T/ND | caller/Kernel | Kernel/ViewCopy | GM 生命周期 | broadcast stride 0 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  tx["buf0 tempX/outputT · P bytes"]
  fx["buf1 floatX/result · P bytes"]
  fy["buf2 floatY · P bytes"]
  ty["buf3 tempY · P bytes"]
```

##### 字节账与复用顺序

```text
UB_total = 4P
P = floor((U/4)/32)*32
UB_total <= U
workspace_total(core reservation) = 16 MiB；Kernel consumer = 0 B
=> float compute capacity = P/4 elements；T input capacity P/2 elements，不构成更紧上限
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| B_TEMP_X | NDDMA x | Cast x | downcast result | 同一 V 流顺序 | Cast 已消费输入 |
| B_X | Cast x | VF reg load | VF StoreAlign 原地覆盖 | 单 repeat 内先 LoadAlign 后 StoreAlign | dst 与 srcX 同址 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input | `MTE2_V` | 两路 T NDDMA | Cast 前 | temp buffers | 仅有效 tile count |
| precision | `Cast(...CAST_NONE)` | T input | FP32 VF | B_X/B_Y | FLOAT16/BF16 均升到 FP32 |
| output precision | `Cast(...CAST_RINT)` | FP32 result | MTE3 | B_TEMP_X | 统一 RINT 降到原 dtype |
| reuse/tail | `V_MTE3`、`MTE3_MTE2`、mask | V/MTE3 | CopyOut/下一 tile | 四个 buffer | `a_i_tail` 与 VF mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `GetShapeInfo/DoTilingAndSet<4>` | `phys_nodes=4`，P 与 rank4 POD |
| Kernel entry/router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<4>` | dtype binary 注入 `DTYPE_X` |
| Device body/helper | `op_kernel/arch35/xlog1py_kernel.h` | `ProcessFP16` | 双升精度、VF、RINT、两处原地复用 |

### Route R-003：rank8 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `RunTiling` | normalized rank `5..8`，FLOAT | `DoTilingAndSet<8>` + rank8 key |
| Router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<8>` | `RANK!=4`, `DTYPE_X=float` | `Xlog1pyKernel<float,8>` |

#### Device 数据流

##### 函数级调用链

```text
xlog1py<8> -> Xlog1pyKernel<float,8>::Init
  -> build MultiCopyParams<float,5> and nddmaOuterIters
  -> Process -> CopyInOne (one 5D copy or outer-loop multiple 5D copies)
  -> ProcessFP32 -> Xlog1pyVF -> CopyOutOne
```

##### 源码等价伪码

```text
ND=5
for each input:
  encode last min(8-axis,5) dimensions in MultiCopyParams
  outerIters = product of remaining leading dimensions from split axis
for tile:
  for oi in outerIters:
    elem_adj = decode oi with input strides (0 stride broadcasts)
    NDDMA up-to-5D GM[base+elem_adj] -> UB[oi*inner]
  execute the same FP32 VF and contiguous output copy as R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input NDDMA suffix | GM `base+elem_adj`；UB `oi*inner` | 最多 5D loop | float/ND→连续 | MTE2 | VF | tile | 无 padding；broadcast stride 0 |
| NDDMA outer loop | `oi in [0,nddmaOuterIters)` | split suffix超过 5D 时启用 | 索引循环 | Device scalar | NDDMA | tile | 多次 copy 拼接同一 UB |
| B_X/B_Y/B_Z | UB 子区 | 各 P B | float | NDDMA/VF | VF/MTE3 | tile | 与 R-001 相同 |
| GM z | output stride offset | `count*4` B | float/连续 | MTE3 | ViewCopy | executor | 单 block CopyOut |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  bx["buf0 B_X · P bytes · outer NDDMA拼接"]
  by["buf1 B_Y · P bytes · outer NDDMA拼接"]
  bz["buf2 B_Z · P bytes"]
```

##### 字节账与复用顺序

```text
UB_total = 3P <= U；P=floor((U/3)/32)*32
NDDMA intermediate = 0 B（直接拼入 B_X/B_Y）
workspace core reservation = 16 MiB，Kernel读取0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| B_X/B_Y 分段 | 各次 NDDMA | 完整 VF | 下一 tile | 所有 CopyIn 后统一 MTE2_V | `oi*inner` 覆盖互不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 多 NDDMA→VF | 单一 `MTE2_V` event | 该 tile 全部 input copy | VF | 拼接后的完整 B_X/B_Y | FP32 |
| VF/output/tile | 与 R-001 相同 | V/MTE3 | CopyOut/下一 tile | B_Z/全部 UB | mask + `a_i_tail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `DoTilingAndSet<8>` | rank8 POD、split、stride |
| Kernel entry/router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<8>` | rank8 实例 |
| Device body/helper | `op_kernel/arch35/xlog1py_kernel.h` | `Init/CopyInOne<float>` | 5D NDDMA 参数与外层拆分 |

### Route R-004：rank8 FP16/BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `GetShapeInfo/RunTiling` | normalized rank `5..8`，FLOAT16/BF16 | `phys_nodes=4` + rank8 key |
| Router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<8>` | `DTYPE_X` 非 float | `Xlog1pyKernel<T,8>` |

#### Device 数据流

##### 函数级调用链

```text
xlog1py<8> -> Init(MultiCopyParams<T,5>, four TBuf)
  -> CopyInOne x/y with optional outer NDDMA split
  -> ProcessFP16: T->float casts -> in-place fused VF -> CAST_RINT to T
  -> CopyOutOne
```

##### 源码等价伪码

```text
for assigned rank8 tile:
  materialize broadcast T inputs by one/multiple 5D NDDMA copies
  wait all MTE2; cast both inputs to float
  run fused FP32 VF in B_X; cast_rint result into reused B_TEMP_X
  wait V; copy valid T bytes to output
  wait previous MTE3 before next tile overwrite
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| B_TEMP_X/B_TEMP_Y | UB，outer copy `oi*inner` | 各 P B | T/连续 | NDDMA | Cast | tile | x buffer 后复用输出 |
| B_X/B_Y | UB | 各 P B | float | Cast | VF | tile | B_X 原地结果 |
| GM input | `base+elem_adj` | 1..多次 5D region | T/ND | caller | NDDMA | Kernel | 0 stride broadcast |
| GM output | output stride | `count*sizeof(T)` | T/ND | MTE3 | ViewCopy | executor | tail 精确字节数 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  tx["buf0 T x / T output · P bytes"]
  fx["buf1 FP32 x/result · P bytes"]
  fy["buf2 FP32 y · P bytes"]
  ty["buf3 T y · P bytes"]
```

##### 字节账与复用顺序

```text
UB_total = 4P <= U；P=floor((U/4)/32)*32
outer-NDDMA额外UB = 0 B
workspace core reservation = 16 MiB，Kernel读取0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| B_TEMP_X | 多次 NDDMA x | Cast | downcast output | MTE2_V + V 顺序 | 全部 outer copy 完成后才 Cast |
| B_X | Cast x | VF load | VF store | VF 单线程段内顺序 | 原地地址一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA | `MTE2_V` | 所有 outer copies | 两次 Cast | T input buffers | 输入 dtype 保持 |
| compute | Cast + VF | T buffers | downcast | FP32 buffers/registers | FP32 `Ln/Mul/Compare/Select` |
| output | `CAST_RINT` + `V_MTE3` | FP32 VF | CopyOut | T output buffer | 精确 `count*sizeof(T)` |
| reuse | `MTE3_MTE2` | prior CopyOut | next CopyIn | 四缓冲 | `a_i_tail`/mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/xlog1py_tiling_arch35.cpp` | `GetShapeInfo/DoTilingAndSet<8>` | 四 buffer 容量与 rank8 POD |
| Kernel entry/router | `op_kernel/xlog1py_apt.cpp` | `xlog1py<8>` | 半精度 rank8 binary 实例 |
| Device body/helper | `op_kernel/arch35/xlog1py_kernel.h` | `CopyInOne/ProcessFP16` | 高维 NDDMA、FP32 升降精度与复用 |

## 算子特定优化

### Trick T-001：半精度路径双重原地复用并保持 FP32 计算

#### 触发与源码依据

- 适用 Route：R-002、R-004
- 触发条件：`NEED_CAST = !std::is_same_v<T,float>`，即 FLOAT16/BFLOAT16 binary
- Golden 来源：[`op_kernel/arch35/xlog1py_kernel.h:212-226`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/arch35/xlog1py_kernel.h#L212-L226)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr int B_TEMP_X = 0, B_X = 1, B_Y = 2, B_TEMP_Y = 3;
CopyInOne(coord, inX, B_TEMP_X, a_i_seg);
CopyInOne(coord, inY, B_TEMP_Y, a_i_seg);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
AscendC::Cast(buf_[B_X].template Get<float>(), buf_[B_TEMP_X].template Get<T>(),
              AscendC::RoundMode::CAST_NONE, count);
AscendC::Cast(buf_[B_Y].template Get<float>(), buf_[B_TEMP_Y].template Get<T>(),
              AscendC::RoundMode::CAST_NONE, count);
CallXlog1pyVF(B_X, B_X, B_Y, count);
AscendC::Cast(buf_[B_TEMP_X].template Get<T>(), buf_[B_X].template Get<float>(),
              AscendC::RoundMode::CAST_RINT, count);
```

#### 机制与设计原因

两路半精度输入先各占一个 T buffer，再升成 FP32；VF 的 dst 与 FP32 x buffer 同址，结果覆盖已消费的 x；最后又把结果降精度写回已消费的原始 x temp buffer。这样四个 P-byte buffer 同时承担输入 staging、FP32 计算和输出 staging，无需第五个结果 buffer，同时 `Ln/Mul/Compare/Select` 全部保持 FP32。`CAST_NONE` 用于升精度，`CAST_RINT` 明确控制降精度舍入。

#### 迁移边界

- 适用：逐元素双输入融合、输入可在计算前完全消费、结果可覆盖其中一路 staging 的半精度算子。
- 前提：950 AIV、FLOAT16/BFLOAT16、FP32 VF 支持、MTE2_V 与 V_MTE3 顺序不变。
- 失效：需要保留原始 x 供后续阶段、多输出共享 x、或 VF 不允许 dst/srcX 同址时必须新增 buffer/改路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从源码约束可知，若在 Cast/VF 消费前覆盖 B_TEMP_X/B_X 会破坏输入，但固定仓没有对应错误实现。

### Trick T-002：用零 stride NDDMA 直接把 broadcast tile 物化到计算 buffer

#### 触发与源码依据

- 适用 Route：全部；R-003/R-004 在 suffix 超过 5D 时额外触发 outer loop
- 触发条件：输入归一化 shape 某维为 1 时 Host stride 为 0；rank8 且 `RANK-split.axis>5` 时 `nddmaOuterIters>1`
- Golden 来源：[`op_kernel/arch35/xlog1py_kernel.h:252-267`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/arch35/xlog1py_kernel.h#L252-L267)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                               AscendC::NdDmaConfig::unsetPad, false };
if constexpr (RANK <= 5) {
    AscendC::DataCopy<T, ND, cfg>(buf_[slot].template Get<T>(), gmIn_[inputIdx][off], params);
} else {
    AscendC::LocalTensor<T> buf = buf_[slot].template Get<T>();
    int64_t elem_base = off;
    for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
        int64_t elem_adj = 0, tmp = oi;
        for (int64_t d = RANK - nddma_dims_ - 1; d >= k; d--) {
            int64_t sz = (d == k) ? a_i_seg : dstShape[d];
            elem_adj += (tmp % sz) * td_->input_strides[inputIdx][d];
            tmp /= sz;
        }
        AscendC::DataCopy<T, ND, cfg>(buf[oi * inner], gmIn_[inputIdx][elem_base + elem_adj], params);
    }
}
```

#### 机制与设计原因

Host 预先把 broadcast 维 stride 设为 0，Device 将目标 broadcast shape 写入 NDDMA loop size，因此 DMA 直接把重复值展开到连续 UB。rank8 超过单次 ND 参数覆盖范围时，外层坐标只调整 GM 地址并把每个片段直接拼到 `buf[oi*inner]`；没有额外 layout workspace，也不需要 VF 内逐元素做坐标除模。

#### 迁移边界

- 适用：尾对齐 broadcast、输出 tile 连续、输入 stride 可用 0 表示复制维的逐元素算子。
- 前提：`MultiCopyParams<T,ND>` 支持对应零 stride/loop，单次最多 5D，外层拼接容量不超过 P。
- 失效：非规则索引、动态 gather 或不能用 stride 表达的广播需改用其它搬运/计算路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能把高于 5D 的后缀直接塞入一个 `MultiCopyParams<T,5>`，官方实现通过 outer loop 显式拆分。

### Trick T-003：单次 VF 调用融合特殊值语义

#### 触发与源码依据

- 适用 Route：全部
- 触发条件：每个 tile 在升为/本来就是 FP32 后调用 `Xlog1pyVF<float>`
- Golden 来源：[`op_kernel/arch35/xlog1py_kernel.h:306-328`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/xlog1py/op_kernel/arch35/xlog1py_kernel.h#L306-L328)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// S1: Adds(y, 1.0) → y+1
AscendC::Reg::Adds(regYp1, regY, scalar_one, mask);

// S2: Ln(y+1) → log(1+y)
AscendC::Reg::Ln(regLogY, regYp1, mask);

// S3: Mul(x, log_y) → mul_res
AscendC::Reg::Mul(regMul, regX, regLogY, mask);

// S4: Compare(x, 0, EQ) → mask_eq
AscendC::Reg::Sub(regZero, regX, regX, mask);
AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(maskEq, regX, regZero, mask);

// S5: Select(mask_eq, 0, mul_res) → res0
AscendC::Reg::Select<float>(regRes0, regZero, regMul, maskEq);

// S6: Compare(y, y, EQ) → mask_nan (NaN≠NaN → mask=0)
AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(maskNaN, regY, regY, mask);

// S7: Select(mask_nan, res0, y) → final
AscendC::Reg::Select<float>(regFinal, regRes0, regY, maskNaN);

AscendC::Reg::StoreAlign(dstAddr, regFinal, aReg, mask);
```

#### 机制与设计原因

`Adds→Ln→Mul→Compare→Select→Compare→Select` 全部停留在寄存器，既实现 `x==0` 强制零，又用 `y==y` 的 IEEE NaN 性质选择原始 y，避免把七个语义阶段物化到 UB/GM 或拆成多个 Kernel。`mask` 同时约束所有步骤与最终 StoreAlign，尾 repeat 不写无效 lane。

#### 迁移边界

- 适用：需要 log1p、零乘数特殊值和 NaN 透传的逐元素融合。
- 前提：FP32 Register `Ln/Compare/Select` 可用，调用侧提供 `VL`、`ceil(count/VL)` 与有效 mask。
- 失效：改变 NaN/零符号语义、要求不同舍入或复数 dtype 时不可直接复用该链。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；只计算 `x*log1p(y)` 而省略两个 Select 会改变 `x==0` 与 `y=NaN` 的公开语义。

## 模板与实现对比

| 对比维度 | R-001 rank4 FP32 | R-002 rank4 FP16/BF16 | R-003 rank8 FP32 | R-004 rank8 FP16/BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | rank<=4/FLOAT | rank<=4/半精度 | rank5..8/FLOAT | rank5..8/半精度 | rank 决定 POD/NDDMA 模板，dtype 决定 buffer/精度 |
| 模板与 dtype | `<float,4>` | `<T,4>` | `<float,8>` | `<T,8>` | binary 注入 `DTYPE_X` |
| 循环与分核 | tile 均分 | 同 | tile 均分 + NDDMA outer | 同 | core 公式共用 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 TBuf + FP32 reg | 4 TBuf + FP32 reg | 3 TBuf + FP32 reg | 4 TBuf + FP32 reg | 无 TQue/dynamic UB/通信 |
| 内存总占用与复用 | 3P | 4P，双重原地 | 3P，无额外高维 buffer | 4P，双重原地 | `phys_nodes` 与 `NUM_BUF` 配对 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 4D NDDMA + VF | 同 + Cast | 最多5D NDDMA拆分 + VF | 同 + Cast | 无 SIMT/通信 |
| 同步、精度与 tail | FP32，3 event | FP32计算/RINT输出 | 同 R-001 | 同 R-002 | `a_i_tail` + VF mask |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | 无 | 平台/dtype/rank 不支持均 Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持 | 公开 README/API 明确不支持 | Ascend 950PR/950DT 支持 |
| Host/Launcher | 不生成正式卡 | 不生成正式卡 | DAV_3510 检查，arch35 Tiling，rank4/rank8 key |
| Kernel/资源 | 无活动接线 | 无活动接线 | AIV-only，NDDMA + FP32 VF，3P/4P UB |

## 限制与失败边界

- 仅 950 DAV_3510；A2/A3 公开明确不支持，不能仅凭通用 ACLNN 声明推断支持。
- 仅 FLOAT/FLOAT16/BFLOAT16、ND、raw rank 不超过 8、三者同 dtype、非 private format；固定 ACLNN `CheckParams` 未显式比较 `z` shape 与 broadcast shape，最终写回约束落在 `ViewCopy` 框架边界。
- 非空路径由 `BroadcastInferShape` 检查 `x/y` 可广播；空输入在该检查前短路，固定源码未展示空输入 shape 不兼容时的额外校验。
- Host 的 shape 乘积、stride、tile 数均用 `int64_t`，固定源码没有显式乘法溢出检查；VF 又把 `count` 窄化到 `uint32_t`、repeat 窄化到 `uint16_t`，没有显式 guard/fallback。
- Tiling 给核心节点固定登记 16 MiB workspace，但 Kernel 不读取 `workspace`；其保留原因在固定源码中不可见。ACLNN 总 workspace 还可能包含 Contiguous/ViewCopy，聚合细节属于框架边界。
- `evVtoMTE2` 被申请并传给处理函数，但固定 body 未使用；实际复用依靠 `MTE2_V`、`V_MTE3`、`MTE3_MTE2`。
- 固定源码未公开 AscendC NDDMA/Register primitive 的内部实现；本卡闭合到算子传入的 loop、stride、mask、repeat 与 event 参数，不推测 primitive 内部微架构。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/xlog1py/docs/aclnnXlog1py.md` | `aclnnXlog1py*` 文档 | 950 支持、语义、dtype/rank/layout 与错误 |
| 入口实现 | `math/xlog1py/op_api/aclnn_xlog1py.cpp` | `CheckParams`、两段式入口 | 校验、空域、Contiguous/L0/ViewCopy、executor |
| L0 Launcher | `math/xlog1py/op_api/xlog1py.cpp` | `IsAiCoreSupport`、`Xlog1pyAiCore` | DAV_3510、broadcast shape、AICore node |
| Host/Tiling | `math/xlog1py/op_host/arch35/xlog1py_tiling_arch35.cpp` | `PadAndSqueeze`、`FindSplitAxis`、`DoTilingAndSet` | 全部 Host 参数与 workspace |
| Host/Tiling ABI | `math/xlog1py/op_host/arch35/xlog1py_tiling_arch35.h` | `Xlog1pyCompileInfo`、`Xlog1pyTiling` | 平台 core/UB compile info 与 Tiling 类接口 |
| POD | `math/xlog1py/op_kernel/arch35/xlog1py_tiling_struct.h` | `Xlog1pyTilingData<R>` | ABI 字段与 rank4/rank8 布局 |
| key selector | `math/xlog1py/op_kernel/arch35/xlog1py_struct.h` | `XLOG1PY_RANK_4/8`、`ASCENDC_TPL_SEL` | rank key 的合法实例集合 |
| Kernel entry/body | `math/xlog1py/op_kernel/xlog1py_apt.cpp` | `xlog1py<RANK>` | AIV-only router |
| Kernel entry/body | `math/xlog1py/op_kernel/arch35/xlog1py_kernel.h` | `Xlog1pyKernel`、`Xlog1pyVF` | NDDMA、UB、VF、同步、精度、tail |
| 平台注册 | `math/xlog1py/op_host/xlog1py_def.cpp` | `Xlog1py::Xlog1py` | ascend950 OpDef 与 dtype |
| binary config | `math/xlog1py/op_host/config/ascend950/xlog1py_binary.json` | `op_list` | 三种 dtype binary |
| 构建接线 | `math/xlog1py/CMakeLists.txt` | `add_all_modules_sources` | ascend950、arch35、xlog1py_apt 活动接线 |

每行文件均已列入 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | rank4=320 B、rank8=544 B；全 `int64_t` | 字段顺序、数组 rank、stride 单位、delta 左补 | `Xlog1pyTilingData<R>` |
| 调用顺序 | Contiguous x/y → Xlog1py → ViewCopy；Kernel CopyIn→VF→CopyOut | 空域短路与 executor stream 顺序 | `aclnnXlog1pyGetWorkspaceSize`/`Process` |
| workspace/内存 | P 按 3/4 个物理 buffer 分割；核心 reservation 16 MiB | P 的 32B 对齐、3P/4P、workspace 未被 body 使用 | `DoTilingAndSet`/`Init` |
| 同步 | MTE2_V、V_MTE3、MTE3_MTE2 | 单缓冲不得提前覆盖；最后由 Kernel/stream 完成保证可见 | `ProcessFP32/ProcessFP16` |
| 精度与尾块 | 半精度升 FP32、RINT 降精度；`a_i_tail` + VF mask | NaN 透传、x==0 选择零、mask/repeat/VL | `Xlog1pyVF`/`GetUBSplitRange` |
| 失败边界 | DAV_3510/dtype/rank/broadcast/format 检查 | 无 Device fallback，不得扩到 A2/A3/其它 dtype | `CheckParams`/`IsAiCoreSupport` |
