---
type: Operator
title: "CircularPad2dBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnCircularPad2dBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/docs/aclnnCircularPad2dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_api/padv3grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/circular_pad_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad_2d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/circular_pad_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/config/ascend910b/circular_pad_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/config/ascend910_93/circular_pad_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:50:40Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# CircularPad2dBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCircularPad2dBackwardGetWorkspaceSize` / `aclnnCircularPad2dBackward`，公开两段式 ACLNN；A2/A3 的 circular 分支由 `l0op::PadV3Grad` 选择 `CircularPadGrad`。 |
| 计算语义 | 把 `gradOutput` 中所有经 circular 映射到同一原输入坐标的梯度求和到 `gradInput`；负 padding 对应前向裁剪，裁掉位置的梯度为 0。 |
| 输入输出 | `gradOutput/self/gradInput` 为同 dtype 的 ND Tensor，dtype 为 FP16/BF16/FP32，rank 为 3 或 4；`padding=[left,right,top,bottom]` 为 INT64；`gradInput.shape=self.shape`。 |
| 执行类型 | 单 Kernel，节点为 `AIV_VECTOR`；FP16/BF16 在 FP32 中累加，FP32 原类型累加。 |
| Kernel 路线 | R-001 small-shape 整页驻 UB；R-002 big-shape 分行搬运。每条路线各有 FP32、FP16、BF16 模板实例。 |
| 核心资源 | 输入/输出 GM、每任务 FP32 workspace、深度 2 的 `TQueBind`、MTE2/MTE3/VEC pipe event、workspace 原子加。 |
| 来源闭包 | `source_closed`：ACLNN 校验/前处理 → `PadV3Grad` 平台 selector → `CircularPadGrad` arch32 Tiling/POD → key router → 2D body/helper → A2/A3 构建接线均闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令原输入空间为 `S=[...,H,W]`，前向 circular padding 为 `p=[l,r,t,b]`，前向输出也是反向输入 `G=gradOutput`。对 `gradInput[...,h,w]`：

```text
gradInput[..., h, w] = Σ G[..., oh, ow]
其中 circular_map(oh - t, H) = h，circular_map(ow - l, W) = w。
```

正 padding 会使同一 `(h,w)` 最多从正文、上下回卷和左右回卷的组合位置收到多项；负 padding 表示前向裁剪，反向先把完整 `gradInput` 清零，再只回写未裁剪区域。A2/A3 body 先在 FP32 workspace 中合并上下、左右贡献，再写最终 dtype。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | rank 3/4，ND；shape 等于前向 circular pad 输出 | FP16/BF16/FP32 | 与 `self/gradInput` 同 dtype、同 view format；非连续时经 `l0op::Contiguous`。最后一维公开约束：FP16/BF16 `(0,16362)`，FP32 `(0,24544)`。 |
| `self` | rank 3/4，ND | 同 `gradOutput` | 仅提供前向原输入 shape；shape 与 `gradInput` 相同；非连续时也进入 `Contiguous` 框架节点。 |
| `padding` | 长度 4 | INT64 | 顺序 `[left,right,top,bottom]`；API 将其扩为与 rank 匹配的连续成对 Tensor：未 pad 的前导轴填 0。 |
| `gradInput` | 与 `self` 同 shape/layout | 同 `gradOutput` | 输出；非连续目标由 `ViewCopy` 从连续中间结果写回。 |

### 入口与版本

固定源码只公开这一组无版本后缀的 2D backward 两段式接口。`commonPad2dBackward` 固定以 `mode="circular"`、`paddingsContiguous=true`、`padFlag=true` 调用 `l0op::PadV3Grad`。该 L0 selector 在 `mode == "circular"` 时，A2/A3 的 `IsRegBase()==false` 分支无条件落到 `CircularPadGradAiCore`；950 走另一实现，见独立 `_950` 卡。公开产品表明确 A2、A3 支持。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckNotNull` 检查四个 Tensor/数组指针；`OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`。
2. `CheckDtypeValid` 限定 FP32/FP16/BF16 且三 Tensor dtype 相同；`CheckFormat` 拒绝 private storage format，并要求三者 view format 相同。
3. `CheckShape` 要求 `self.shape==gradInput.shape`、rank 为 3 或 4、三者 rank 相同、padding 长度为 4，并验证 `gradOutput.H=self.H+t+b`、`gradOutput.W=self.W+l+r`。
4. 空 Tensor 时，除 batch 第 0 维外出现零维会报错；可接受的空域直接返回 `workspaceSize=0`，executor 中不加入核心节点。
5. 非空时依次对 `gradOutput`、`self` 调 `Contiguous`。`GetPaddingTensor` 从后向前构造每轴成对 padding，例如 rank 4 得到 `[0,0,0,0,t,b,l,r]`。
6. `l0op::PadV3Grad` 推导连续输出并将 `CircularPadGrad` 加入 AICore launcher；最后 `ViewCopy` 把连续结果提交到用户 `gradInput`。

### 参数、Workspace 与生命周期

`CircularPadGrad` Tiling 读取 AIV 核数、UB 字节数和系统 workspace。对每个前导逻辑页分配：

```text
inputWAlign = ceil(inputW * sizeof(T1) / 32) * 32 / sizeof(T1)
inOutputH  = inputH - max(top,0) - max(bottom,0)
workspaceLen_small = inputH    * (8 + inputWAlign + 8)  # FP32 元素
workspaceLen_big   = inOutputH * (8 + inputWAlign + 8)  # FP32 元素
operator_workspace = totalTasks * workspaceLen * 4 bytes
total_workspace    = operator_workspace + sysWorkspaceSize
```

`GetUserWorkspace(workspace)` 跳过系统区后，每核绑定自己任务页的 FP32 slice；不同核不共享 slice。ACLNN executor 同时拥有 `Contiguous/ViewCopy` 等框架节点可能需要的空间，第一段接口以 `uniqueExecutor->GetWorkspaceSize()` 返回总量，固定仓不展开框架节点的内部切片。

### 启动图与失败传播

```text
可选 Contiguous(gradOutput) ─┐
可选 Contiguous(self)       ├─ 同一 executor/stream 顺序
ConvertToTensor(padding) ───┘
  -> CircularPadGrad(x=gradOutput, paddings, y=continuousResult,
                     userWorkspace, CircularPadCommonTilingData)
  -> ViewCopy(continuousResult, gradInput)
```

核心是单 Device 节点：Host 由 `ADD_TO_LAUNCHER_LIST_AICORE(CircularPadGrad,...)` 添加，entry 为 `circular_pad_grad`，`blockDim=useCoreNum`，输入 `x/paddings`、输出 `y`，每个逻辑页对应 FP32 workspace slice。节点与 `ViewCopy` 由同一 executor 在传入 stream 上保持顺序；任一创建/launcher 返回空或失败即返回 `ACLNN_ERR_INNER_NULLPTR`，后续节点不成立。框架 executor 的更细 event 与临时 Tensor owner 固定仓不可见。

## Route 与参数设计

### 分核与资源计算

`totalTasks` 是输入最后两维之前各维乘积；2D API 下即 rank-3 的首维或 rank-4 的前两维乘积。`perCoreTaskNum=totalTasks/coreNum`，`tailTaskNum=totalTasks%coreNum`，使用核数在任务不足时缩为 `tailTaskNum`。Device 对 `blockId<tailTaskNum` 的核多分一页，并以 `startIdx=base*blockId+min(blockId,tailTaskNum)` 绑定连续页。

FP32 的 `usedUBSize=ubSize/2`；FP16/BF16 的 `usedUBSize=ubSize/6`。若 `inputH*inputWAlign < usedUBSize/sizeof(T1)` 走 small，否则走 big。注意判断为严格小于。共同 guard 还要求 `inputW < usedUBSize/sizeof(T1)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 small-shape | FP32 `211`；FP16 `221`；BF16 `231` | `mode=circular`、2D key 基数 200，且 `inputH*inputWAlign < usedUBSize/tSize` | 条件不成立进入 R-002；校验失败终止 |
| 2 | R-002 big-shape | FP32 `212`；FP16 `222`；BF16 `232` | 同上且 small 谓词为假，同时 `inputW < usedUBSize/tSize` | 无 fallback；失败 |

```text
mode == circular && !IsRegBase()
  -> CircularPadGrad
     -> CheckDtype / CalculateParams / CheckInput
     -> shapeType = (inputH * inputWAlign < usedUBSize / tSize) ? 1 : 2
     -> key = 200 + 10 * dataType + shapeType
        dataType: FP32=1, FP16=2, BF16=3
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 small-shape | 211/221/231 | A2/A3；FP32/FP16/BF16；整页满足严格 UB 谓词 | 单 `circular_pad_grad` | `CircularPadGrad2D<float,float,false>` 或 `<half,float,true>` / `<bfloat16_t,float,true>`；`ProcessSmallShape` | 整页搬入 UB；FP32 workspace 原子合并；阶段间 MTE3→MTE2 | 每核页数按余数分配；无下一 fallback |
| R-002 big-shape | 212/222/232 | A2/A3；同 dtype；整页不驻 UB但单行通过 guard | 单 `circular_pad_grad` | 同三模板；`ProcessBigShape` | 按 `rowsNum` 分批搬运；同一 workspace 布局与原子合并 | 行循环有 `tail=lines%rowsNum`；无下一 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 单行 UB | `inputW < usedUBSize/tSize` | 至少一行及其 cast buffer 必须可在 UB 中处理 | R-001/R-002 | Tiling 失败 | `CircularPadGradTiling::CheckInput` |
| 整页 UB | `inputH*inputWAlign < usedUBSize/tSize` | small 一次搬入整页；FP16/BF16 同时保留输入和 FP32 队列 | R-001 | 否则 R-002 | `CircularPadGradTiling::DoOpTiling` |
| circular 正 pad 上界 | 左右同正时 `l+r<=2W/3`，单侧正时该侧 `<=W/2`；H 同理 | 限定同坐标 wrap 候选与 workspace 合并范围 | 两 Route | Tiling 失败 | `CheckLeftAndRight` / `CheckTopAndBottom` |
| 输出 shape | `outputH=inputH-top-bottom` 且 `outputW=inputW-left-right` | 保证 API 结果 shape 与 Device 偏移一致 | 两 Route | Tiling 失败 | `CircularPadGradTiling::CheckInput` |
| 32B 对齐 | `inputWAlign=align32(inputW*sizeof(T1))/sizeof(T1)` | DMA blockLen/stride 与 workspace 每行两侧各留 32B | 两 Route | 公式内处理，无 fallback | `CalculateParams` / `InitCommon` |

### POD/ABI

`CircularPadCommonTilingData` 由 `TILING_DATA_FIELD_DEF` 生成；15 个字段均为 `int64_t` 并按声明顺序序列化。宏实现不在固定仓，因此精确 `sizeof/alignof/offsetof` 标为固定仓不可见；Host producer 与 Device consumer 一致使用同一注册类。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CircularPadCommonTilingData` | `inputH` | int64/64 | 第 1 字段；精确 offsetof 固定仓不可见 | 元素/行 | `SetTilingData(inputH)` | `InitCommon`、上下合并 | >0 |
| 同上 | `inputW` | int64/64 | 第 2 字段 | 元素/行 | `SetTilingData(inputW)` | 对齐、DMA、地址 | 受单行 UB guard |
| 同上 | `outputH` | int64/64 | 第 3 字段 | 行 | `SetTilingData(outputH)` | `outputLen_`、输出偏移 | `inputH-top-bottom` |
| 同上 | `outputW` | int64/64 | 第 4 字段 | 元素/行 | `SetTilingData(outputW)` | 输出 stride/长度 | `inputW-left-right` |
| 同上 | `left` | int64/64 | 第 5 字段 | 元素 | `SetTilingData(left)` | 正/负 pad 分解 | API padding[0] |
| 同上 | `right` | int64/64 | 第 6 字段 | 元素 | `SetTilingData(right)` | 正/负 pad 分解 | API padding[1] |
| 同上 | `top` | int64/64 | 第 7 字段 | 行 | `SetTilingData(top)` | 正/负 pad 分解 | API padding[2] |
| 同上 | `bottom` | int64/64 | 第 8 字段 | 行 | `SetTilingData(bottom)` | 正/负 pad 分解 | API padding[3] |
| 同上 | `front` | int64/64 | 第 9 字段 | 页 | `SetTilingData(front)` | 2D Route 不读取 | 2D 下为 0/inactive |
| 同上 | `back` | int64/64 | 第 10 字段 | 页 | `SetTilingData(back)` | 2D Route 不读取 | 2D 下为 0/inactive |
| 同上 | `inputL` | int64/64 | 第 11 字段 | 页 | `SetTilingData(inputL)` | 2D Route 不读取 | inactive |
| 同上 | `outputL` | int64/64 | 第 12 字段 | 页 | `SetTilingData(outputL)` | 2D Route 不读取 | inactive |
| 同上 | `perCoreTaskNum` | int64/64 | 第 13 字段 | 页/核 | `DivCore` 后写入 | `Init2D`，尾核可加 1 | 基础商 |
| 同上 | `tailTaskNum` | int64/64 | 第 14 字段 | 核/余数页 | `DivCore` 后写入 | `Init2D` 分配余数 | `<coreNum` |
| 同上 | `workspaceLen` | int64/64 | 第 15 字段 | FP32 元素/页 | `DoOpTiling` | 每页 workspace stride | 含两侧各 8 FP32 元素 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 small-shape | 211/221/231 | `circular_pad_grad_tiling.cpp::DoOpTiling` | `CircularPadCommonTilingData` | `circular_pad_grad.cpp::circular_pad_grad` | `CircularPadGrad2D::ProcessSmallShape`、`CircularPadGrad` helpers、`CircularPadCommon` | `circular_pad_grad/CMakeLists.txt` + `circular_pad_grad_def.cpp` |
| R-002 big-shape | 212/222/232 | 同上 | 同上 | 同上 | `CircularPadGrad2D::ProcessBigShape`、分行 helpers | 同上 |

## Kernel 实现

### 公共执行流程

entry 解出 POD、创建 `TPipe`、取得 user workspace，再按 2D key 实例化三种 dtype 模板。`Init2D` 将 block 映射为连续页区间；若任一 padding 为负，先把本核整个输出区置零。两 Route 都执行：上下贡献写 FP32 workspace → MTE3/MTE2 同步 → 左右回卷贡献原子加到 workspace → 同步 → 取有效矩形写输出。`p*=max(pad,0)` 决定需要求和的 wrap 区，`n*=min(pad,0)` 决定输出写入偏移。

### Route R-001：small-shape 整页驻 UB

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_circular_pad2d_backward.cpp` | `commonPad2dBackward` | `mode="circular"` | 进入 `l0op::PadV3Grad` |
| L0 selector | `conversion/pad_v3_grad/op_api/padv3grad.cpp` | `PadV3Grad` | `mode=="circular" && !IsRegBase()` | `CircularPadGradAiCore` |
| Tiling | `op_host/arch32/circular_pad_grad_tiling.cpp` | `DoOpTiling` | `inputH*inputWAlign < usedUBSize/tSize` | `shapeType=1`，key 211/221/231 |
| Router | `op_kernel/circular_pad_grad.cpp` | `circular_pad_grad` | `TILING_KEY_IS(211/221/231)` | 三 dtype 的 `ProcessSmallShape` |

#### Device 数据流

##### 函数级调用链

```text
CircularPadGradAiCore -> circular_pad_grad(x,paddings,y,workspace,tiling)
  -> CircularPadGrad2D<T1,float,ISCAST>::Init2D
  -> ProcessSmallShape
     -> AddTopAndBottomSmallShape
        -> AddTopAndBottomSmallShapeNoCast / Cast
     -> MTE3ToMTE2Sync
     -> AddLeftAndRightSmallShape -> AddLeftAndRightOnce
     -> MTE3ToMTE2Sync
     -> CopyToOutSmallShape -> CopyToOutSmallShapeOnePage
```

##### 源码等价伪码

```text
# 伪码；page 为本核连续逻辑页
for page in assigned_pages:
  if any(pad < 0): zero(y[page])
  ub = DMA_PAD(x[page], inputH rows, inputW bytes -> inputWAlign)
  fp32 = cast_if_needed(ub)
  workspace[page][center_rows] = fp32[pTop : pTop+inOutputH]
  atomic_add workspace[page][bottom_wrap_rows] += fp32[top_rows]       if top>0
  atomic_add workspace[page][top_wrap_rows]    += fp32[bottom_rows]    if bottom>0
  barrier MTE3->MTE2
  gather left/right aligned strips from workspace into UB
  atomic_add workspace[opposite-side strips] += gathered strips
  barrier MTE3->MTE2
  result = workspace[page][rows, pLeft:pLeft+inOutputW]
  cast FP32 -> T1 with CAST_ROUND if needed
  DMA result -> y[page][offsetH:, offsetW:]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `startIdx*inputLen + page*inputLen` | `inputH*inputW` | T1/ND | caller | top/bottom copy | caller；kernel 只读 | 行搬入 pad 到 `inputWAlign` |
| workspace GM | `startIdx*workspaceLen + page*workspaceLen` | `workspaceLen` | FP32，行 stride `inputWAlign+16` | 上下阶段 | 左右阶段与 CopyOut | executor user workspace；节点结束失效 | 每行前后各 8 FP32=32B guard |
| y GM | `startIdx*outputLen + page*outputLen + (-nTop)*outputW + (-nLeft)` | 有效矩形 `inOutputH*inOutputW` | T1/ND | CopyOut | ViewCopy/用户 | caller | 负 pad 未覆盖区预清零 |
| `queBindT2_` FP32/no-cast | queue 内部 | depth 2 × `UB_SIZE/2` bytes | FP32 | DMA/VEC | DMA/VEC | `TPipe`/kernel | FP32 Route 总计 `UB_SIZE` |
| `queBindT1_` cast input | queue 内部 | depth 2 × `UB_SIZE/6` bytes | FP16/BF16 | MTE2 | Cast | `TPipe`/kernel | 仅 ISCAST |
| `queBindT2_` cast result | queue 内部 | depth 2 × `UB_SIZE/3` bytes | FP32 | Cast/累加 | MTE3 | `TPipe`/kernel | 仅 ISCAST |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["ISCAST: queBindT1 · depth=2 · 2×UB/6 B"]:1
  calc["queBindT2 · depth=2 · FP32:2×UB/2 B; ISCAST:2×UB/3 B"]:2
```

##### 字节账与复用顺序

```text
FP32: UB_total = 2 * (UB_SIZE/2) = UB_SIZE
FP16/BF16: UB_total = 2*(UB_SIZE/6) + 2*(UB_SIZE/3) = UB_SIZE
workspace_total = totalTasks * workspaceLen * 4 + sysWorkspaceSize
=> small 要求整页有效元素严格小于单 buffer 容量
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot 0/1 | `DataCopyPad`/`Cast` | 下游 Cast 或 MTE3 | 下一页/下一 strip | `EnQue/DeQue/FreeTensor` | Free 后才复用 |
| workspace page | 上下中心写与 atomic wrap | 左右合并、最终 CopyOut | 节点完成后 | 两次 `MTE3ToMTE2Sync` | 每核页区不重叠 |
| `wLRLocal` | 左右 strip 搬入 | atomic Add 搬出 | 下一 page | `EnQue/DeQue/FreeTensor` + MTE3 barrier | 左输出完成后才写右输出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入→计算 | TQue bind | MTE2 `DataCopyPad` | `DeQue`/Cast | queue slot | 行尾 pad 不进入有效输出 |
| 上下→左右 | `SetFlag/WaitFlag<MTE3_MTE2>` | workspace MTE3 写/atomic add | 左右 MTE2 读 | workspace page | 保证读到完整上下和 |
| wrap 合并 | `SetAtomicAdd<float>` | top/bottom/left/right DMA | workspace | FP32 slice | FP16/BF16 先 `CAST_NONE` 到 FP32 |
| workspace→输出 | queue + `CAST_ROUND` | FP32 workspace | y MTE3 | 有效矩形 | 按 `blockCount` 和 32B 对齐处理尾行；只写有效宽 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_grad_tiling.cpp` | `DoOpTiling` | dtype、small 谓词、workspaceLen、key |
| Kernel entry/router | `circular_pad_grad.cpp` | key 211/221/231 | 映射三 dtype small 模板 |
| Device body/helper | `arch32/circular_pad_grad_2d.h` | `ProcessSmallShape` | 上下→左右→输出顺序 |
| Device body/helper | `arch32/circular_pad_grad.h` | `AddTopAndBottomSmallShapeCast`、`AddLeftAndRightOnce` | FP32 升精度、workspace atomic 合并 |

### Route R-002：big-shape 分行搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/L0 | 同 R-001 | `commonPad2dBackward` / `PadV3Grad` | circular 且非 RegBase | `CircularPadGrad` |
| Tiling | `op_host/arch32/circular_pad_grad_tiling.cpp` | `DoOpTiling` | `inputH*inputWAlign >= usedUBSize/tSize` 且单行 guard 通过 | `shapeType=2`，key 212/222/232 |
| Router | `op_kernel/circular_pad_grad.cpp` | `circular_pad_grad` | key 212/222/232 | `ProcessBigShape` |

#### Device 数据流

##### 函数级调用链

```text
circular_pad_grad
  -> CircularPadGrad2D<T1,float,ISCAST>::Init2D
  -> ProcessBigShape
     -> AddTopAndBottomBigShape -> CopyGMToGMLines -> CopyGMToGMOnce*
     -> MTE3ToMTE2Sync
     -> AddLeftAndRightBigShape -> AddLeftAndRightLines -> AddLeftAndRightOnce
     -> MTE3ToMTE2Sync
     -> CopyToOutBigShapeBig -> CopyToOutBigShapeOnePage -> CopyGMToGMLinesOnce
```

##### 源码等价伪码

```text
# 伪码
rowsNum = T1UBSize / align32(lineW*sizeof(T1))
for page in assigned_pages:
  for each full rowsNum batch and final tail rows:
    DMA/cast input center rows -> FP32 workspace
  barrier
  atomic_add top rows -> bottom wrap; bottom rows -> top wrap
  barrier
  rowsLR = T2UBSize / ((leftAlign+rightAlign)*sizeof(float))
  for each full rowsLR batch and tail:
    DMA left/right strips -> UB; atomic_add to opposite strips in workspace
  barrier
  for each output rowsNum batch and tail:
    DMA workspace valid rectangle -> UB; cast if needed; DMA -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/workspace/y GM | 同 R-001 | workspace 每页仅 `inOutputH*(inputWAlign+16)` | T1/FP32/T1 | 分行 helper | 后续 helper | 同 R-001 | 跨 batch 以 `strideLoop` 推进 |
| 行输入 UB | batch index × aligned row | `rowsNum*alignedRow` | T1 或 FP32 | `CopyGMToGMLines` | Cast/MTE3 | queue slot | 最后一批 `tail=lines%rowsNum` |
| 左右 UB | 右 strip 后接左 strip | `rowsLR*(rightAlign+leftAlign)` | FP32 | `AddLeftAndRightOnce` | atomic MTE3 | `queBindT2_` | 分母仅在至少一侧正 pad 时使用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["queBindT1 (ISCAST) · depth=2 · 2×UB/6 B"]:1
  calc["queBindT2 · depth=2 · FP32/no-cast=UB; cast=2×UB/3 B"]:2
```

##### 字节账与复用顺序

```text
UB_total 与 R-001 相同，均等于硬编码 UB_SIZE=192 KiB
rowsNum = floor(T1UBSize / (align32(lineW*sizeof(T1))))
rowsLR  = floor(T2UBSize / ((leftAlign+rightAlign)*sizeof(float)))
workspace_total = totalTasks * inOutputH * (inputWAlign+16) * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 行 queue slot | 当前行 batch MTE2/Cast | workspace MTE3 | 下一个 batch | queue Free | full/tail blockCount 分别设置 |
| workspace page | 中心/上下/左右阶段 | 输出分行读取 | kernel 后 | MTE3→MTE2 event | 阶段严格串行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分行搬运 | TQue | MTE2 | Cast/MTE3 | UB batch | full loop 后显式处理 `tail` |
| workspace 阶段 | `MTE3ToMTE2Sync` | 每阶段最后写 | 下一阶段首读 | FP32 workspace | 与 R-001 同累加/round 顺序 |
| 输出 | `CopyGMToGMLinesOnce` | workspace | y | 有效矩形 | full rows 与 tail rows 各设置 `blockCount` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_grad_tiling.cpp` | `DoOpTiling` | big key 与缩短的 workspace 高度 |
| Kernel entry/router | `circular_pad_grad.cpp` | key 212/222/232 | 映射 big body |
| Device body/helper | `arch32/circular_pad_grad_2d.h` | `ProcessBigShape` | 三阶段调用顺序 |
| Device body/helper | `arch32/circular_pad_grad.h` | `CopyGMToGMLines`、`AddLeftAndRightLines`、`CopyGMToGMLinesOnce` | 行批次、tail、地址推进 |

## 算子特定优化

### Trick T-001：FP16/BF16 梯度在 FP32 workspace 中合并

#### 触发与源码依据

- 适用 Route：R-001、R-002 的 FP16/BF16 key。
- 触发条件：模板 `ISCAST=true`。
- Golden 来源：[`conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h:206-241`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h#L206-L241)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
auto inLocal = queBindT1_.AllocTensor<T1>();
DataCopyPad(inLocal, xGM_[i * inputLen_], params.paramsIn, padParmsT1);
queBindT1_.EnQue<QuePosition::GM, QuePosition::VECIN, T1>(inLocal);
inLocal = queBindT1_.DeQue<QuePosition::GM, QuePosition::VECIN, T1>();
auto inLocalT2 = queBindT2_.AllocTensor<T2>();
Cast(inLocalT2, inLocal, RoundMode::CAST_NONE, inputH_ * inputWAlign_);
```

#### 机制与设计原因

输入只在 GM/输入 queue 中保留 16-bit，进入累加前升为 FP32；上下和左右 wrap 都对 FP32 workspace 做 atomic add，最后一次性 `CAST_ROUND` 回目标 dtype。这避免每次 wrap 合并都先发生 16-bit 舍入；代价是 workspace 恒按 4 字节元素计。

#### 迁移边界

- 适用：多个来源映射到同一输出且输出为 FP16/BF16 的梯度聚合。
- 前提：有 FP32 workspace，UB 能同时容纳双缓冲输入和 FP32 结果。
- 失效：FP32 输入无需 cast；workspace 或 UB 不满足时本实现直接 Tiling 失败，没有低精度 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：small/big 专门化避免整页超 UB

#### 触发与源码依据

- 适用 Route：R-001/R-002。
- 触发条件：严格比较 `inputH*inputWAlign` 与 dtype 对应单 buffer 容量。
- Golden 来源：[`conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp:150-166`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp#L150-L166)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (dataType == TYPE_MODE1) {
    usedUBSize = ubSize / BUFFER_NUM;
} else {
    usedUBSize = ubSize / BUFFER_NUM / DIM_3;
}
if (inputH * inputWAlign < usedUBSize / tSize) {
    shapeType = DIM_1;
    workspaceLen = inputH * (ALIGN / WTYPE_SIZE + inputWAlign + ALIGN / WTYPE_SIZE);
} else {
    shapeType = DIM_2;
    workspaceLen = inOutputH * (ALIGN / WTYPE_SIZE + inputWAlign + ALIGN / WTYPE_SIZE);
}
```

#### 机制与设计原因

整页可驻留时一次搬入可复用同页数据；否则复用同一 UB queue 按多行批次流式处理，并将 workspace 高度缩到去除正 top/bottom 后的 `inOutputH`。选择发生在 Host，Device 主循环无需动态判断整页是否可放入。

#### 迁移边界

- 适用：二维页式数据、单行可放 UB 但整页大小跨度大的场景。
- 前提：行宽通过单行 guard；workspace 行布局保持两侧 32B guard。
- 失效：单行本身不进 UB 时没有更细粒度 fallback，Tiling 失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 small-shape | R-002 big-shape | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 整页严格小于单 buffer 容量 | small 谓词为假且单行通过 | Host 以整页是否驻 UB 分流 |
| 模板与 dtype | 三 dtype 模板，shapeType=1 | 同模板，shapeType=2 | dtype 只改变 cast 和 UB 划分 |
| 循环与分核 | 每核按页，页内整块 | 每核按页，页内按 `rowsNum` | 大页避免超 UB |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 深度 2 TQueBind；无 dynamic UB/通信 | 同一物理配置反复复用 | 两 Route 总 UB 都为 192 KiB |
| 内存总占用与复用 | workspace 含 `inputH` 行 | workspace 仅 `inOutputH` 行 | big 去除无需保存的正 pad 行 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad`、`Cast`、atomic DMA | 相同 API 加行批次循环 | 均为 AIV Vector |
| 同步、精度与 tail | 阶段 event；页级有效宽 | 阶段 event；显式尾行 batch | FP16/BF16 均 FP32 累加 |
| fallback/失败边界 | 超整页容量转 R-002 | 单行超容量直接失败 | 无 AICPU/Cube fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 支持，但使用独立 `_950` 卡 |
| Host/Launcher | `ascend910b`，arch32 `CircularPadGrad` | `ascend910_93`，同 arch32 实现 | RegBase selector 改投 `PadV3Grad` |
| Kernel/资源 | key 211/212/221/222/231/232，192 KiB UB 常量 | 同左 | SIMT/MicroAPI SIMD，不使用本卡 body |

## 限制与失败边界

- API rank 仅 3/4、dtype 仅 FP16/BF16/FP32、padding 长度必须为 4，shape 关系必须完全成立；private storage format 被拒绝。
- 公开文档最后一维上限比内部 README/Tiling 理论边界更保守；复用时以 ACLNN 文档 `(16362/24544)` 为发布约束。
- 正 padding 还受 A2/A3 Tiling 的 1/2、2/3 比例约束；单行无法放入 dtype 对应 UB slice 时失败，无 fallback。
- `CircularPadCommonTilingData` 宏展开后的精确 `sizeof/alignof/offsetof`、executor 框架节点 workspace 切片和 event 细节固定仓不可见。
- entry 对未知 key 直接返回，不写输出；正常 Host 仅产生列出的 2D key。本次未执行运行验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/circular_pad_grad/docs/aclnnCircularPad2dBackward.md` | 产品表/接口约束 | A2/A3 支持、dtype/rank/宽度约束 |
| 入口声明/约束 | `conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.h` | 两段式声明 | 参数 ABI |
| 入口实现 | `conversion/circular_pad_grad/op_api/aclnn_circular_pad2d_backward.cpp` | `commonPad2dBackward` | 校验、padding 转换、executor graph |
| 入口实现 | `conversion/pad_v3_grad/op_api/padv3grad.cpp` | `PadV3Grad` / `CircularPadGradAiCore` | A2/A3 平台 selector |
| Host/Tiling/Launcher | `conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp` | `Tiling4CircularPadGrad` | key、UB、workspace、blockDim |
| Host/Tiling/Launcher | `conversion/circular_pad/op_host/circular_pad_common_tiling.h` | `CircularPadCommonTilingData` | POD 声明 |
| Host/Tiling/Launcher | `conversion/circular_pad/op_host/circular_pad_common_tiling.cpp` | `GetShapeAttrsInfo` / `DivCore` | shape、分核、POD producer |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/circular_pad_grad.cpp` | `circular_pad_grad` | key router |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad_2d.h` | `CircularPadGrad2D` | 2D launch body |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h` | `CircularPadGrad` | 搬运、cast、atomic、tail helpers |
| Kernel entry/body | `conversion/circular_pad/op_kernel/circular_pad_common.h` | `CircularPadCommon` | 公共地址、对齐、event |
| 构建接线 | `conversion/circular_pad_grad/op_host/circular_pad_grad_def.cpp` | `CircularPadGrad` OpDef | A2/A3 注册 |
| 构建接线 | `conversion/circular_pad_grad/op_host/config/ascend910b/circular_pad_grad_binary.json` | `CircularPadGrad` binaries | A2 三 dtype ND 二进制接线 |
| 构建接线 | `conversion/circular_pad_grad/op_host/config/ascend910_93/circular_pad_grad_binary.json` | `CircularPadGrad` binaries | A3 三 dtype ND 二进制接线 |
| 构建接线 | `conversion/circular_pad_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | A2/A3→arch32 构建映射 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 个 int64 字段的 producer/consumer 顺序 | 字段顺序、workspace 单位为 FP32 元素 | `CircularPadCommonTilingData` |
| 调用顺序 | 上下→左右→输出 | 两个阶段间 MTE3→MTE2 同步不可删除 | `ProcessSmallShape/ProcessBigShape` |
| workspace/内存 | 每页带左右 32B guard 的 FP32 layout | `workspaceLen`、核间 slice 不重叠 | `DoOpTiling` / `Init2D` |
| 同步 | queue 生命周期、atomic 开关、硬 event | atomic 关闭与下一阶段等待顺序 | `AddTopAndBottom*` / `MTE3ToMTE2Sync` |
| 精度与尾块 | 16-bit→FP32 累加→round cast；big 显式尾行 | cast 次序、有效宽和负 pad 零区 | `CopyGMToGMOnceCast*` |
| 失败边界 | 未知 key 返回、Host 校验失败无 fallback | 不得扩展到 3D key 或假设 AICPU fallback | `circular_pad_grad` / `CheckInput` |