---
type: Operator
title: "CircularPad2d A2/A3 ACLNN 算子知识"
description: "说明 aclnnCircularPad2d 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/docs/aclnnCircularPad2d.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/op_api/aclnn_circular_pad2d.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/op_api/aclnn_circular_pad2d.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3/op_api/padv3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3/op_host/pad_v3_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_2d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/config/ascend910b/circular_pad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/config/ascend910_93/circular_pad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-24T23:50:18Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# CircularPad2d A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCircularPad2dGetWorkspaceSize` + `aclnnCircularPad2d`，公开两段式 ACLNN；固定仓未发现更高版本入口。 |
| 计算语义 | 对最后两维作 signed circular padding：正值循环扩展，负值裁剪；概念索引为 `y[..., oh, ow] = x[..., mod(oh-top,H), mod(ow-left,W)]`，有效域由 Host/Tiling guard 限定。 |
| 输入输出 | `self/out` 为 3D 或 4D ND，dtype 同为 `int8/float16/bfloat16/float32/int32`；`padding=[left,right,top,bottom]` 为 4 个 `int64`。 |
| 执行类型 | 单节点 `AIV_VECTOR` 数据搬运 Kernel；Tiling 使用 `GetCoreNumAiv()`，Device 仅含 MTE/Queue 搬运与事件同步，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 小 shape 整页路线；R-002 大 shape 分条带路线。每条路线按 dtype 展开 5 个 key。 |
| 核心资源 | GM `x/y/workspace`，一个 `TQueBind<VECIN,VECOUT,1>` 双缓冲（总 192 KiB UB），外加 executor workspace；无计算寄存器、SIMT dynamic UB 或通信窗口。 |
| 来源闭包 | `source_closed`：公开 ACLNN 经 `PadV3(mode="circular")` 的 L0 分派闭合到 `CircularPad` Host/Tiling、POD、`circular_pad` entry 和全部 2D body/helper、平台 binary 配置。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设输入最后两维为 `H×W`，`padding=[left,right,top,bottom]`，输出为 `OH=H+top+bottom`、`OW=W+left+right`。正 padding 从相对边界循环取值，负 padding 先裁掉对应边界；前导维保持不变。对输出有效坐标，源码实现等价于按周期映射回输入最后两维，数据仅搬运、不作算术变换，因此 dtype 位模式和确定性均保持。

3D 输入把第 0 维视为独立 page，4D 输入把前两维乘积视为 page 数。API 在空 Tensor 上不启动 Kernel：3D 空 Tensor 一律失败；4D 仅允许第 0 维为 0 且其余三维非 0，此时 workspace 为 0、executor 为空计算流程。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 3D/4D，公开文档为 ND；实现拒绝 private storage format，并要求 view format 与 `out` 相同 | `int8/float16/bfloat16/float32/int32` | 非连续输入先经框架 `Contiguous`；最后一维 `inputW < ubSize/(2*tSize)`。 |
| `padding` | 长度 4，顺序 `[left,right,top,bottom]` | `int64` | API 将其扩为 `2*rank` 的连续 padding Tensor，前导维补 0；Tiling 对正、负、混合符号分别校验，输出维不得为负。 |
| `out` | rank、view format 与 `self` 相同；最后两维为 `OH×OW` | 与 `self` 相同 | 非连续输出由框架 `ViewCopy` 写回；文档额外限定 `OW`：int8 `(0,98176)`，f16/bf16 `(0,49088)`，int32/f32 `(0,24544)`。 |
| `workspaceSize` | 标量输出 | `uint64_t` | 返回 executor 中全部节点所需 Device workspace；`CircularPad` 节点自身请求见后文。 |
| `executor` | 两段式执行器 | `aclOpExecutor**` | 第一段创建并保存 `Contiguous/CircularPad/ViewCopy` 流程，第二段在给定 stream 上执行。 |

### 入口与版本

公开声明为 `aclnnCircularPad2dGetWorkspaceSize` 和 `aclnnCircularPad2d`。第一段完成空指针、dtype、format、rank、padding 长度和输出 shape 检查；第二段调用 `CommonOpExecutorRun`。固定仓只存在这一公开版本族。

API 实现调用 `l0op::PadV3(..., "circular", true, ...)`，但 `PadV3` 的 circular 分支并不进入通用 PadV3/AICPU：`l0op::PadV3` 在 `mode == "circular"` 时直接 `ADD_TO_LAUNCHER_LIST_AICORE(CircularPad, ...)`。因此核心算法闭合到本目录的 `CircularPad` Tiling/Kernel，而非 arch35 PadV3 body。官方文档明确 A2、A3 支持，明确 Ascend 950PR/950DT 不支持；A2 对应 `ascend910b`、A3 对应 `ascend910_93` 的 OpDef/binary/build 接线。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序如下：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，`CHECK_NOT_NULL` 检查 `self/padding/out`，随后创建 executor。
2. `CheckDtypeValid` 检查五种 dtype 且输入输出一致；`CheckFormat` 拒绝 private storage format并要求 view format 相同；`CheckShape` 要求 rank 3/4、padding 长度 4、前导维不变及 `OH/OW` 精确匹配。
3. 空域按前述规则提前结束；否则 `InputPreprocess` 将 `self` 连续化。
4. `GetPaddingTensor` 逆序扩展为 rank 对：4D 得到 `[0,0,0,0,top,bottom,left,right]`，3D 得到 `[0,0,top,bottom,left,right]`；常量值 0 只为满足 PadV3 L0 形参，circular 分支不把它传给 `CircularPad`。
5. `PadV3` 先用常量 padding 推导输出 shape，随后 circular 特判添加单个 `CircularPad` AIV 节点；最终 `ViewCopy(pad2dResult,out)` 处理非连续输出。
6. 任一步返回空指针或失败码即停止构造；第二段只运行已构造 executor。

### 参数、Workspace 与生命周期

Tiling 从平台取得 `coreNum=GetCoreNumAiv()`、`ubSize` 与 `sysWorkspaceSize`。令 `tasks=∏self.shape[0:rank-2]`、`S=sizeof(dtype)`、`align32(e)=ceil(e*S/32)*32/S`、`innerH=H+min(top,0)+min(bottom,0)`、`innerW=W+min(left,0)+min(right,0)`：

- 小 shape：`workspaceLen=innerH*(align32(max(left,0))+align32(innerW)+align32(max(right,0)))` 个 Tiling 元素。
- 大 shape：先把非正左右 padding 的 strip 宽设为 `32/S`，再执行源码的 `strip = strip>W ? signedPadding : strip`，`workspaceLen=innerH*(leftStrip+rightStrip)`。
- `CircularPad` 节点请求 `tasks*workspaceLen*sizeof(float)+sysWorkspaceSize` 字节；这是源码精确公式，因固定乘 4，int8/f16/bf16 会保守多申请。Device 通过 `GetUserWorkspace(workspace)` 跳过系统区，再按 `workspaceLen` 个 `T`/page 使用。
- executor 最终 `GetWorkspaceSize()` 还汇总框架 `Contiguous/ViewCopy` 的需求；其内部切片算法属于允许的框架边界，固定仓不可见。workspace 由调用方持有至第二段 stream 执行结束，executor 持有节点和临时 Tensor 描述。

### 启动图与失败传播

```text
aclnnCircularPad2dGetWorkspaceSize
  -> [可选框架节点] Contiguous(self)
  -> [单个算子节点] CircularPad(x, expandedPaddings, y_tmp)
       Host: Tiling4CircularPad
       entry: circular_pad<<<useCoreNum, stream>>>
       POD: CircularPadCommonTilingData
       workspace: sys prefix + tasks*workspaceLen*4 bytes
  -> [框架节点] ViewCopy(y_tmp, out)
aclnnCircularPad2d
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

核心语义 launch graph 只有一个 Device 节点，无 Kernel 间边。`Contiguous` 和 `ViewCopy` 与 `CircularPad` 由同一 executor/stream 排序；其生成式 launcher、临时 Tensor 物理 offset 和失败清理为框架边界。`ADD_TO_LAUNCHER_LIST_AICORE` 失败使 `PadV3` 返回 `nullptr`，ACLNN 第一段返回 `ACLNN_ERR_INNER_NULLPTR`；Tiling 任一 guard 失败则该节点不启动，后续 `ViewCopy` 不应消费无效结果。

## Route 与参数设计

### 分核与资源计算

对 CircularPad2d，扩展 padding 的前导对均为 0，故 `front=back=0`、`dPad=false`。`DivCore` 计算：

```text
perCoreTaskNum = tasks / coreNum
tailTaskNum    = tasks % coreNum
useCoreNum     = perCoreTaskNum>0 ? coreNum : tailTaskNum
core k start   = perCoreTaskNum*k + min(k,tailTaskNum)
core k count   = perCoreTaskNum + (k<tailTaskNum ? 1 : 0)
```

每个 task 是一个 `H×W` page。Host 路线阈值为 `innerH*align32(OW) < ubSize/(2*S)`；Kernel 固定初始化两个各 `192KiB/2=96KiB` 的 Queue buffer。小 shape 把一个裁剪后内页装入 UB；大 shape 按 `rowsNum=96KiB/align32(loopW)/S` 分批。所有宽度对齐均以 32 字节为粒度。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 小 shape 整页 | int8 `211`；f16 `221`；bf16 `231`；f32 `241`；int32 `251` | dtype 映射成功，所有输入 guard 成功，且 `innerH*align32(OW) < ubSize/(2*S)` | 条件为假进入 R-002；任一 guard 失败则 Tiling 失败。 |
| 2 | R-002 大 shape 条带 | int8 `212`；f16 `222`；bf16 `232`；f32 `242`；int32 `252` | dtype 映射成功，所有输入 guard 成功，且上述乘积 `>= ubSize/(2*S)` | 无其他 fallback；未知 key 的 entry 直接返回。 |

```text
CheckDtype -> CalculateParams -> CheckInput
  -> innerH*align32(OW) < ubSize/(2*S) ? shapeType=1 : shapeType=2
  -> key = 200 + 10*dataType + shapeType
  -> DivCore -> SetTilingData -> SetBlockDim(useCoreNum)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 小 shape 整页 | `211/221/231/241/251` | A2/A3；五种 dtype；3D/4D；signed padding/shape/UB guard 成功；整页严格小于半 UB | 单 `circular_pad` AIV 节点 | `CircularPad2D<T>::Init2D` → `ProcessSmallShape` → `PadLeftAndRightSmallShape<HASLR>` → `CopyToOutSmallShape<HASLR>` | 中间裁剪页及 LR 环绕片写 workspace；Queue 双缓冲；workspace 写后以 MTE3→MTE2 event 再读。 | top/bottom 从 UB 页首尾回填；无 fallback，未知 key 返回。 |
| R-002 大 shape 条带 | `212/222/232/242/252` | 同上，但整页大于等于半 UB | 单 `circular_pad` AIV 节点 | `CircularPad2D<T>::Init2D` → `ProcessBigShape` → `PadLeftAndRightBigShape<HASLR>` → `CopyToOutBigShape` | workspace 仅物化左右 strip；中心直接 x→y；每阶段以 MTE3→MTE2 event 保证 GM 可读。 | 行批 `loop+tail`；top/bottom 再从 y 首尾搬运；无 fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 输入行 UB 上界 | `inputW < ubSize/(2*S)`（严格） | 任一行必须能进入 96 KiB Queue buffer | 两条 Route | Tiling `GRAPH_FAILED` | `circular_pad_tiling.cpp::CheckInput` |
| 整页阈值 | `innerH*align32(OW) < ubSize/(2*S)` | R-001 一次在 UB 保留裁剪后页面；等号归 R-002 | R-001/R-002 | R-002 | `CircularPadTiling::DoOpTiling` |
| shape 一致 | `OH==H+top+bottom && OW==W+left+right`，2D 下 `outputL==inputL` | 地址公式直接依赖 Host shape | 两条 Route | Tiling 失败 | `CircularPadTiling::CheckInput` |
| signed 左右 padding | 四个按符号组合的边界式 | 防止裁剪越界或正 padding 超过剩余周期域 | 两条 Route | Tiling 失败 | `CheckLeftAndRight` |
| signed 上下 padding | 与左右同构的四个边界式 | 防止行选择越界 | 两条 Route | Tiling 失败 | `CheckTopAndBottom` |
| AIV 核数/UB | `GetCoreNumAiv()>0 && ubSize>0` | 明确选择 AIV，排除 Cube/AIC 执行 | 两条 Route | TilingPrepare 失败 | `TilingPrepare4CircularPad` |
| dtype/router guard | key dtype 十位为 1..5；bf16 仅在非 NPU arch 3003/3113 编译 | 保证 `T` 与 binary dtype 一致 | 两条 Route | unsupported dtype Tiling 失败；未知 key return | `CheckDtype`；`circular_pad` |

### POD/ABI

`CircularPadCommonTilingData` 由 `BEGIN_TILING_DATA_DEF` 宏声明。固定仓能确认字段顺序、每字段 `int64_t` 和 Host/Device 同名访问；宏定义位于外部 CANN 头，故精确 `sizeof/alignof/offsetof` 固定仓不可见。下表的 2D inactive 字段仍被 Host 写入，但 2D body 不读取。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CircularPadCommonTilingData` | `inputH` | `int64_t/64` | 声明序 0；精确 offset 不可见 | 元素/行 | storage shape 倒数第 2 维 | `InitCommon`，地址/循环 | `>0`（非空路径） |
| 同上 | `inputW` | `int64_t/64` | 声明序 1；精确 offset 不可见 | 元素/行 | storage shape 最后 1 维 | `InitCommon`，行 stride | 严格小于半 UB 元素数 |
| 同上 | `outputH` | `int64_t/64` | 声明序 2；精确 offset 不可见 | 行 | 输出倒数第 2 维 | `InitCommon`，top/bottom offset | `H+top+bottom` |
| 同上 | `outputW` | `int64_t/64` | 声明序 3；精确 offset 不可见 | 元素/行 | 输出最后 1 维 | `InitCommon`，GM blockLen | `W+left+right` 且文档上界 |
| 同上 | `left` | `int64_t/64` | 声明序 4；精确 offset 不可见 | 元素 | 扩展 padding 尾 4 项 | LR helpers | signed guard |
| 同上 | `right` | `int64_t/64` | 声明序 5；精确 offset 不可见 | 元素 | 同上 | LR helpers | signed guard |
| 同上 | `top` | `int64_t/64` | 声明序 6；精确 offset 不可见 | 行 | 同上 | top/bottom helpers | signed guard |
| 同上 | `bottom` | `int64_t/64` | 声明序 7；精确 offset 不可见 | 行 | 同上 | top/bottom helpers | signed guard |
| 同上 | `front` | `int64_t/64` | 声明序 8；精确 offset 不可见 | page | 扩展 padding 中为 0 | 2D inactive | 必须为 0 才产生 2D key |
| 同上 | `back` | `int64_t/64` | 声明序 9；精确 offset 不可见 | page | 扩展 padding 中为 0 | 2D inactive | 必须为 0 |
| 同上 | `inputL` | `int64_t/64` | 声明序 10；精确 offset 不可见 | page | rank 3/4 的倒数第 3 维 | 2D inactive | 仅 shape 一致检查间接使用 |
| 同上 | `outputL` | `int64_t/64` | 声明序 11；精确 offset 不可见 | page | 输出倒数第 3 维 | 2D inactive | 等于 `inputL` |
| 同上 | `perCoreTaskNum` | `int64_t/64` | 声明序 12；精确 offset 不可见 | task/core | `tasks/coreNum` | `Init2D` 后按 tail 修正 | 可为 0，实际启动核各自修正 |
| 同上 | `tailTaskNum` | `int64_t/64` | 声明序 13；精确 offset 不可见 | task/core | `tasks%coreNum` | `Init2D` 分配前若干核 | `[0,coreNum)` |
| 同上 | `workspaceLen` | `int64_t/64` | 声明序 14；精确 offset 不可见 | `T` 元素/page | Route 公式 | `Init2D` GM base/stride | Host 分配公式按 float 字节放大 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 小 shape 整页 | `211/221/231/241/251` | `circular_pad_tiling.cpp::DoOpTiling/Tiling4CircularPad` | `CircularPadCommonTilingData` 全字段 | `circular_pad.cpp::circular_pad` | `circular_pad_2d.h::ProcessSmallShape`；`circular_pad.h`；`circular_pad_common.h` | `circular_pad_def.cpp`；A2/A3 binary JSON；`ascendc_config.json` |
| R-002 大 shape 条带 | `212/222/232/242/252` | 同上 | 同上 | 同上 | `circular_pad_2d.h::ProcessBigShape`；`circular_pad.h`；`circular_pad_common.h` | 同上 |

## Kernel 实现

### 公共执行流程

`circular_pad` 用 `GET_TILING_DATA` 解析 POD，构造一个 `TPipe`，以 `GetUserWorkspace` 跳过系统 workspace，随后按 key 实例化 `CircularPad2D<int8_t/half/bfloat16_t/float/int32_t>`。`Init2D` 调用公共 `Init`，固定初始化一个双缓冲 `TQueBind`，按 blockId 切分 page，并给 `xGM/yGM/workspaceGM` 绑定本核连续区间。两条路线都是纯搬运：GM→UB、Queue EnQue/DeQue、UB→GM；无 dtype cast、Vector 算术、Atomic 或跨核共享写。

### Route R-001：小 shape 整页

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/circular_pad/op_host/op_api/aclnn_circular_pad2d.cpp` | `aclnnCircularPad2dGetWorkspaceSize` | 非空、校验成功，构造 rank 对 padding，`mode="circular"` | `l0op::PadV3` |
| L0 | `conversion/pad_v3/op_api/padv3.cpp` | `PadV3` | `mode == CIRCULAR_MODE` | `CircularPadAiCore`，添加 `CircularPad` 节点 |
| Tiling | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling` | `innerH*align32(OW) < ubSize/(2*S)` | `shapeType=1`，key 个位为 1 |
| Router | `conversion/circular_pad/op_kernel/circular_pad.cpp` | `circular_pad` | key `211/221/231/241/251` | 对应 `CircularPad2D<T>::ProcessSmallShape` |

#### Device 数据流

##### 函数级调用链

```text
CircularPad launcher -> circular_pad(x,paddings,y,workspace,tiling)
  -> CircularPad2D<T>::Init2D
    -> CircularPad<T>::Init -> CircularPadCommon::InitCommon
  -> CircularPad2D<T>::ProcessSmallShape
    -> CircularPad<T>::PadLeftAndRightSmallShape<HASLR>
    -> CircularPadCommon::MTE3ToMTE2Sync
    -> CircularPad2D<T>::CopyToOutSmallShape<HASLR>
      -> CircularPad<T>::CopyToOutSmallShapeOnePage
```

##### 源码等价伪码

```text
# 伪码；page 属于当前 AIV 核
for page in core_pages:
  if left>0 or right>0:
    crop = x[page, -min(top,0):H+min(bottom,0), -min(left,0):W+min(right,0)]
    DMA crop -> UB -> workspace[page, :, leftAlign:leftAlign+align32(innerW)]
    if right>0: DMA UB 中每行首部 -> workspace 右 strip
    MTE3->MTE2
    if left>0: DMA workspace 中每行尾部 -> workspace 左 strip
  MTE3->MTE2
  src = workspace[:, leftAlign-left : leftAlign-left+OW] if hasLR else cropped x
  DMA 全 innerH 行 src -> UB -> y[page, top:top+innerH, :]
  if top>0:    DMA UB 最后 top 行 -> y[page, :top, :]
  if bottom>0: DMA UB 最前 bottom 行 -> y[page, OH-bottom:, :]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `xGM_` | `x + inputLen*startTask` | `inputLen*coreTaskCount` 元素 | `T`，page-major ND | 调用方/Contiguous | LR/crop DMA | 调用方/executor；Kernel 只读 | 行源 stride 为 `inputW` |
| GM `workspaceGM_` | user workspace `+ workspaceLen*startTask` 个 `T` | `workspaceLen*coreTaskCount` 个 `T` | 每 page 为 `[innerH, leftAlign+innerWAlign+rightAlign]` | LR DMA | 输出 DMA | 调用方 owner；当前 Kernel 内有效 | 各行 32B 对齐；无跨核重叠 |
| GM `yGM_` | `y + outputLen*startTask` | `outputLen*coreTaskCount` 元素 | `T`，page-major ND | 输出及 top/bottom DMA | ViewCopy/调用方 | executor/调用方 | 每 page 独占 |
| UB `queBind_` | Queue 管理 | `2*(192KiB/2)=192KiB` | `T` LocalTensor | GM→UB `DataCopyPad` | UB→GM `DataCopyPad` | `TPipe`，Alloc→EnQue→DeQue→Free | depth=2；每 buffer 96KiB |

`DataCopyExtParams` 的 `blockLen` 为字节；GM 侧 stride 由源码以字节表达，LocalTensor 侧 stride 在需要非零时按 32B block 表达（如右 strip 的 `srcStride=(innerWAlign-rightAlign)*S/32`）。每次只写 `blockCount` 个有效行，alignment padding 不提交为逻辑输出。

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind buffer 0 · 96 KiB · GM→UB/UB→GM"]:1
  q1["queBind buffer 1 · 96 KiB · GM→UB/UB→GM"]:1
```

##### 字节账与复用顺序

```text
UB_total = BUFFER_NUM * (UB_SIZE/BUFFER_NUM) = 2 * 96 KiB = 192 KiB
Host route guard: innerH*align32(OW)*S < ubSize/2
dynamic_ub = N/A（AIV Queue 静态分配）
workspace_node = tasks*workspaceLen*4 + sysWorkspaceSize bytes
workspace_effective_T = tasks*workspaceLen*S bytes（其余为 Host 保守余量）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `queBind_` buffer | 每次 GM→UB `DataCopyPad` | 对应 UB→GM | `FreeTensor` 后下一 DMA | EnQue/DeQue/FreeTensor | 单次 extent 不超过 96KiB |
| workspace 中央/右 strip | `PadLeftAndRightSmallShape` 首轮 | 左 strip 构造及最终输出 | Kernel 结束后可复用 | `MTE3ToMTE2Sync` | 每核 workspace page 不重叠 |
| 输出中间行 | `CopyToOutSmallShapeOnePage` | 后续 ViewCopy | stream 后继节点 | executor stream 顺序 | Kernel 成功返回 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→GM | Queue `EnQue/DeQue/FreeTensor` | MTE2 | MTE3 | 当前 LocalTensor | 原 dtype 位拷贝，无 cast；有效 blockLen 控制尾字节 |
| workspace 右→左依赖 | `PipeBarrier<PIPE_MTE3>` + `MTE3_MTE2` event | workspace 写 | workspace 读 | 同一 page strip | 防止 GM 回读早于写完成 |
| LR 完成→输出 | `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>` | workspace MTE3 | 输出阶段 MTE2 | workspace page | 只同步本核；核间地址不重叠 |
| page 尾行 | `blockCount=innerH/top/bottom` | DMA | y | 最后有效行 | 无无效行写出；top/bottom 只在正值时执行 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling` | 严格半 UB 阈值选择 shapeType 1，产生 workspaceLen。 |
| Kernel entry/router | `conversion/circular_pad/op_kernel/circular_pad.cpp` | `circular_pad` keys `*1` | 五 dtype key 映射到 `ProcessSmallShape`。 |
| Device body/helper | `conversion/circular_pad/op_kernel/circular_pad_2d.h` | `ProcessSmallShape/CopyToOutSmallShape` | 2D page 分核、中央输出与 top/bottom 回填。 |
| Device body/helper | `conversion/circular_pad/op_kernel/circular_pad.h` | `PadLeftAndRightSmallShape/CalculateLeftAndRightParams` | workspace 的中央、左右 strip 构造及 DMA 参数。 |

### Route R-002：大 shape 条带

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/L0 | `aclnn_circular_pad2d.cpp`；`padv3.cpp` | `aclnnCircularPad2dGetWorkspaceSize`；`PadV3` | 同 R-001，circular 特判 | `CircularPad` 节点 |
| Tiling | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling` | `innerH*align32(OW) >= ubSize/(2*S)` | `shapeType=2`，key 个位为 2 |
| Router | `conversion/circular_pad/op_kernel/circular_pad.cpp` | `circular_pad` | key `212/222/232/242/252` | `CircularPad2D<T>::ProcessBigShape` |

#### Device 数据流

##### 函数级调用链

```text
CircularPad launcher -> circular_pad
  -> CircularPad2D<T>::Init2D
  -> CircularPad2D<T>::ProcessBigShape
    -> CircularPad<T>::PadLeftAndRightBigShape<HASLR>
      -> CircularPad<T>::CopyLines
    -> CircularPadCommon::MTE3ToMTE2Sync
    -> CircularPad2D<T>::CopyToOutBigShape
      -> CircularPad<T>::CopyToOutBigShapeOnePage
        -> CopyWSToOutOnce -> CopyOnePage
        -> MTE3ToMTE2Sync
        -> CopyInToOutOnce -> CopyOnePage
        -> MTE3ToMTE2Sync
        -> PadTopAndBottomOnce -> CopyOnePage
```

##### 源码等价伪码

```text
# 伪码
if hasLR:
  for every core-local page and row batch of rowsNum:
    DMA input 每行末端 leftStrip -> UB -> workspace 每行左槽
    DMA input 每行起始 rightStrip -> UB -> workspace 每行右槽
MTE3->MTE2
for page in core_pages:
  if left>0:  分批 DMA workspace 左槽 -> y 中间区左端
  if right>0: 分批 DMA workspace 右槽 -> y 中间区右端
  MTE3->MTE2
  if holeW>0: 分批 DMA cropped x 中央区 -> y 中间区 hole
  MTE3->MTE2
  if top>0:    分批 DMA y 中间区末 top 行 -> y 顶部
  if bottom>0: 分批 DMA y 中间区首 bottom 行 -> y 底部

rowsNum = 96KiB / align32(loopW) / S
loop = loopH / rowsNum; tail = loopH % rowsNum
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `xGM_` | page base `inputLen*page`，行 base 加裁剪 offset | 本核 pages | `T` ND | 调用方/Contiguous | strip/center DMA | 调用方，只读 | 以 `inputW` 为行 stride |
| GM `workspaceGM_` | `workspaceLen*page + row*(leftStrip+rightStrip)` | `innerH*(leftStrip+rightStrip)` 个 `T`/page | 紧凑双 strip | `CopyLines` | `CopyWSToOutOnce` | Kernel 内临时 | strip 宽 32B 对齐或取真实正 padding |
| GM `yGM_` | `outputLen*page + row*OW + col` | `OH*OW` 个 `T`/page | `T` ND | strip、center、top/bottom DMA | ViewCopy | executor/调用方 | 阶段间 event；y→y 无别名重叠假设外推 |
| UB `queBind_` | Queue 管理 | 两个 96KiB buffer | `T` LocalTensor | 每个条带 GM→UB | 条带 UB→GM | `TPipe` | 同一物理 Queue 跨所有阶段复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind buffer 0 · 96 KiB · 当前行批"]:1
  q1["queBind buffer 1 · 96 KiB · 下一/另一行批"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * 96 KiB = 192 KiB
rowsNum = 96KiB / align32(loopW) / S
dynamic_ub = N/A（AIV Queue 静态分配）
workspaceLen = innerH*(leftStrip+rightStrip) T-elements/page
workspace_node = tasks*workspaceLen*4 + sysWorkspaceSize bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `queBind_` 任一 buffer | `CopyLines/CopyOnePage` 当前 GM→UB | 当前 UB→GM | `FreeTensor` 后下个 batch | Queue 协议 | `rowsNum*align32(loopW)*S <=96KiB` |
| workspace strip | `PadLeftAndRightBigShape` | `CopyWSToOutOnce` | Kernel 结束 | 入口处 MTE3→MTE2 event | page/core 切片互斥 |
| y 中间区 | strip + center 阶段 | `PadTopAndBottomOnce` 的 y→y 读 | top/bottom 写后稳定 | 两次 MTE3→MTE2 event | 中间区完成后才回读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 行批搬运 | Queue 协议 | GM→UB | UB→GM | 96KiB LocalTensor | `loop` 满批 + `tail` 行，blockLen 为真实字节 |
| workspace→y | MTE3→MTE2 event | strip workspace 写 | workspace 回读 | workspace strips | 无 cast/round |
| strip→center | MTE3→MTE2 event | y 左右写 | 后续中心/回读 | y 中间区 | `holeW>0` 才搬中心 |
| center→top/bottom | MTE3→MTE2 event | y 中间区写 | y→y padding | y 中间区 | top/bottom 分别用末/首有效行；各自 `loop+tail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling` | shapeType 2、strip 宽及紧凑 workspace 公式。 |
| Kernel entry/router | `conversion/circular_pad/op_kernel/circular_pad.cpp` | `circular_pad` keys `*2` | 五 dtype key 映射到 `ProcessBigShape`。 |
| Device body/helper | `conversion/circular_pad/op_kernel/circular_pad_2d.h` | `ProcessBigShape/CopyToOutBigShape` | page 循环与阶段顺序。 |
| Device body/helper | `conversion/circular_pad/op_kernel/circular_pad.h` | `PadLeftAndRightBigShape/CopyLines/CopyOnePage/CopyWSToOutOnce/CopyInToOutOnce/PadTopAndBottomOnce` | strip、中心、上下回填的地址、批次和同步。 |

## 算子特定优化

### Trick T-001：整页与条带双路线缩减大 shape workspace

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：`innerH*align32(OW) < ubSize/(2*S)` 走整页；否则只物化左右 strip。
- Golden 来源：[`conversion/circular_pad/op_host/circular_pad_tiling.cpp:174-184`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_tiling.cpp#L174-L184)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (inOutputH * outputWAlign < (int64_t)ubSize / BUFFER_NUM / tSize) {
    shapeType = DIM_1;
    workspaceLen = inOutputH * (leftAlign + inOutputWAlign + rightAlign);
} else {
    shapeType = DIM_2;
    leftAlign = left > 0 ? leftAlign : ALIGN / tSize;
    rightAlign = right > 0 ? rightAlign : ALIGN / tSize;
    leftAlign = leftAlign > inputW ? left : leftAlign;
    rightAlign = rightAlign > inputW ? right : rightAlign;
    workspaceLen = inOutputH * (leftAlign + rightAlign);
}
```

#### 机制与设计原因

小 shape 允许一个 inner page 进入半 UB，workspace 保存“左对齐区 + 裁剪后的中心 + 右对齐区”，最终一次行块搬运同时提供中心与上下循环来源。页面超过半 UB 后，R-002 不再物化中心区，只保存左右环绕 strip；中心直接从 `x` 搬到 `y`，上下 padding 再从已经完成的 `y` 中间区回读。该选择把大 shape 的每页 workspace 从与 `innerW` 成正比降为与左右 strip 宽之和成正比，同时用 `loop+tail` 保证每个 UB 行批不超过 96KiB。

#### 迁移边界

- 适用：二维周期 padding、输出中心可直接由输入裁剪区搬运且上下 padding 可从已写输出回读的实现。
- 前提：AIV MTE 可执行所用 `DataCopyPad`；行宽能进入半 UB；workspace/y 阶段间具有 MTE3→MTE2 可见性；各核 page 互斥。
- 失效：整页不满足阈值必须走 R-002；输入单行不满足半 UB guard 时没有 fallback，Tiling 失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的真实失败边界是忽略 `inputW < ubSize/(2*S)` 仍尝试启动：Tiling 明确返回 `GRAPH_FAILED`；固定仓未提供绕过该 guard 的可验证正确反例。

### Trick T-002：大 shape 以输出中间区作为上下循环源

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`top>0` 或 `bottom>0`，且 strip 与中心写 y 后完成两次 MTE3→MTE2 同步。
- Golden 来源：[`conversion/circular_pad/op_kernel/circular_pad.h:286-312`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad.h#L286-L312)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (top_ > 0) {
    copyParamsIn.offset = pageIdxOut * outputLen_ +
        (outputH_ - pBottom_ - top_) * outputW_;
    copyParamsOut.offset = pageIdxOut * outputLen_;
    CopyOnePage(yGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
}
```

#### 机制与设计原因

R-002 先把输出中间的 `innerH` 行写全，再直接以 `yGM_` 为源和目的：top 从中间区末端取，bottom 从中间区首端取。这样无需在 workspace 再保留完整的上下来源。`CopyToOutBigShapeOnePage` 在 strip、center 后分别执行 `MTE3ToMTE2Sync`，因此后续 MTE2 回读看到已经写入的 y 数据；`CopyOnePage` 继续按 96KiB 行批处理尾行。

#### 迁移边界

- 适用：输出中间区与循环边界源值相同、且同核先写后读的 padding/复制算子。
- 前提：源/目的地址公式不破坏尚未读取的数据，且存在 GM 写→GM 读的显式事件顺序。
- 失效：若中间区尚未完整或 y 的读写区发生不受支持的覆盖关系，必须保留独立中间缓冲；本实现没有该 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

省略 `MTE3ToMTE2Sync` 会违背源码明确的 producer→consumer 顺序；固定仓未提供无同步仍正确的可验证反例。

## 模板与实现对比

| 对比维度 | R-001 小 shape 整页 | R-002 大 shape 条带 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 整页严格 `<` 半 UB | 整页 `>=` 半 UB | 等号归大 shape，避免单 buffer 恰满边界。 |
| 模板与 dtype | `CircularPad2D<T>::ProcessSmallShape`，5 dtype | `CircularPad2D<T>::ProcessBigShape`，5 dtype | dtype 只改变 `S`、key 十位和 C++ `T`。 |
| 循环与分核 | page/core；整页 DMA | page/core + 行批 `loop/tail` | page 不能整块进入 UB 时切行。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 双缓冲 `TQueBind` 192KiB；其余 N/A | 相同 | Kernel 固定资源相同。 |
| 内存总占用与复用 | workspace 保存中央+LR；UB 保存整页 | workspace 仅 LR；UB 跨 strip/center/top-bottom 复用 | 大 shape 避免中心物化。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad` 整页/多 block | `DataCopyPad` 行批 | 均为 AIV MTE 数据搬运，无算术 Vector/SIMT/NDDMA/通信。 |
| 同步、精度与 tail | Queue + workspace event；top/bottom blockCount | Queue + 多次 GM event；行批 tail | 均位拷贝，无 cast/round。 |
| fallback/失败边界 | 阈值不满足转 R-002 | 无后继；未知 key 返回 | 输入行超半 UB时 Tiling 失败。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；公开文档与 `ascend910b` binary/OpDef 接线一致 | 支持；公开文档与 `ascend910_93` binary/OpDef 接线一致 | 明确不支持；未生成 `_950` 卡 |
| Host/Launcher | 共用 `CircularPad` Tiling，平台注册 `ascend910b` | 共用同一实现，平台注册 `ascend910_93` | 无公开 ACLNN 支持，不以其它目录信号补齐 |
| Kernel/资源 | 同一 `circular_pad` AIV body、同一 192KiB Queue 配置 | 同左 | N/A |

## 限制与失败边界

- API 仅支持 3D/4D、五种 dtype、长度 4 的 padding；输入输出 dtype/rank/view format 与输出 shape 必须一致。
- Tiling 对 signed padding 的四种符号组合逐维校验；没有 Device 侧纠错或 fallback。输出推导为负、输入行达到半 UB、平台 AIV 核数/UB 不可得都会失败。
- 大 shape 对非正左右 padding 的 strip 宽先置 `32/S`，再可能赋回 signed padding；固定源码未见独立的 `workspaceLen>=0` 或无符号转换保护。复用实现时不得脱离原 guard 和合法输入域解释该公式。
- `workspaceSize` 的 executor 总切片、`Contiguous/ViewCopy` 内部实现、Tiling 宏展开后的精确 ABI offset，以及 `GetUserWorkspace` 的系统区物理布局属于固定仓不可见的框架边界。
- entry 对未知 key 直接返回，不写错误码；调用链依靠 Host 只生成十个 2D 合法 key。BF16 在 A2/A3 可达，Kernel guard 排除的 NPU arch 3003/3113 属于本卡平台外。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/circular_pad/docs/aclnnCircularPad2d.md` | 两段式原型、产品支持表 | A2/A3 支持、950 不支持、公开 dtype/shape/宽度约束 |
| 入口声明/约束 | `conversion/circular_pad/op_host/op_api/aclnn_circular_pad2d.h` | `aclnnCircularPad2d*` | C ABI 与参数语义 |
| 入口实现 | `conversion/circular_pad/op_host/op_api/aclnn_circular_pad2d.cpp` | `CheckParams/GetPaddingTensor/aclnnCircularPad2dGetWorkspaceSize` | 校验、连续化、padding 扩展、executor 流程 |
| 入口实现 | `conversion/pad_v3/op_api/padv3.cpp` | `PadV3/CircularPadAiCore` | circular 特判到 `CircularPad` launcher |
| 入口实现 | `conversion/pad_v3/op_host/pad_v3_infershape.cpp` | `InferShape4PadV3/PadV3Infershape<int64_t>` | 常量 padding 的输出 shape 推导 |
| Host/Tiling/Launcher | `conversion/circular_pad/op_host/circular_pad_def.cpp` | `CircularPad` OpDef | dtype、A2/A3 平台注册 |
| Host/Tiling/Launcher | `conversion/circular_pad/op_host/circular_pad_common_tiling.h` | `CircularPadCommonTilingData` | POD 字段、compile info |
| Host/Tiling/Launcher | `conversion/circular_pad/op_host/circular_pad_common_tiling.cpp` | `GetShapeAttrsInfo/DivCore/SetTilingData/GetWorkspaceSize` | shape、分核、workspace、POD producer |
| Host/Tiling/Launcher | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling/Tiling4CircularPad/TilingPrepare4CircularPad` | guard、Route、AIV 核数与 UB |
| Kernel entry/body | `conversion/circular_pad/op_kernel/circular_pad.cpp` | `circular_pad` | key router 与 dtype 实例 |
| Kernel entry/body | `conversion/circular_pad/op_kernel/circular_pad_common.h` | `CircularPadCommon/MTE3ToMTE2Sync` | 公共字段、对齐与事件 |
| Kernel entry/body | `conversion/circular_pad/op_kernel/circular_pad.h` | `CircularPad<T>` 及全部 copy helper | 双路线搬运、workspace、tail |
| Kernel entry/body | `conversion/circular_pad/op_kernel/circular_pad_2d.h` | `CircularPad2D<T>` | 2D 分核、路线入口与输出阶段 |
| 构建接线 | `conversion/circular_pad/op_host/config/ascend910b/circular_pad_binary.json` | `CircularPad` binaries | A2 五 dtype binary |
| 构建接线 | `conversion/circular_pad/op_host/config/ascend910_93/circular_pad_binary.json` | `CircularPad` binaries | A3 五 dtype binary |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `CircularPad` | 活动 compute unit 列表与 auto_sync |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 个 `int64_t` 字段的声明顺序与同名 Host/Device访问 | 不得重排；2D 的 front/back 必须为 0；宏展开 ABI 需由目标 CANN 头确认 | `CircularPadCommonTilingData` |
| 调用顺序 | `Contiguous→CircularPad→ViewCopy`，单 Device Kernel | circular 必须在 `PadV3` L0 中特判到 `CircularPad`，不能误走 PadV3 AICPU/arch35 body | `aclnnCircularPad2dGetWorkspaceSize`；`PadV3` |
| workspace/内存 | 小路线保存整页+LR，大路线只保存 LR | Host 乘 `sizeof(float)` 的分配公式、Device 按 `T` stride、系统 workspace 前缀不能混淆 | `GetWorkspaceSize`；`Init2D` |
| 同步 | Queue 生命周期及 MTE3→MTE2 events | workspace/y 先写后读前不可省略 event | `MTE3ToMTE2Sync`；`Process*Shape` |
| 精度与尾块 | 全程 dtype 位拷贝；blockLen/row tail 保护有效区 | 不得引入 cast；LocalTensor stride 的 32B block 单位与 GM stride 字节单位不能混用 | `CalculateLeftAndRightParams`；`CopyLines/CopyOnePage` |
| 失败边界 | 超行宽、shape/padding/dtype/platform guard 均 Host 失败 | 不得把未知 key 的 silent return 当作 fallback | `CheckInput/CheckDtype/circular_pad` |
