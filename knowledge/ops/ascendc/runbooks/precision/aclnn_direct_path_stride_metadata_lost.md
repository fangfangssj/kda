---
type: "Runbook"
title: aclnn 直调路径 stride 丢失，kernel 按物理全量算
description: 1D stride=2 输入下算子对全部物理元素求范数（actual=物理全量范数）；tiling 取 GetStorageShape 元素数且 kernel 连续读 GM，而 aclnn 直调路径 GetDynamicInputStride 拿不到 stride、AutoContiguous 不生效——修 kernel 无效，须在 aclnn 层显式 Contiguous（端到端验证未闭环）。
tags: [task.debugging, topic.precision, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# aclnn 直调路径 stride 丢失，kernel 按物理全量算

## 现象

在 Ascend950PR（CANN 9.1.0）上为 BenchNoHarnessForeachNorm 做 ST Real 全量回归，新增的非连续 Tensor 用例失败，其余 73 个用例全部通过。该用例构造：1D 逻辑视图 shape=[512]、元素间 stride=2、物理存储 storage=[1024]、FP32、p=2（L2 范数），用 `aclCreateTensor(viewDims=[512], strides=[2], storageDims=[1024])` 创建输入。结果：

```
  [FAIL] 标量[0]: golden=22.619326, actual=32.008255, rel_err=4.15e-01
  [FAIL] MERE=4.15e-01 (threshold=1.22e-04), MARE=4.15e-01 (threshold=1.22e-03)
```

golden=22.619326 是按 512 个逻辑元素（物理位置 0,2,4,...,1022）计算的 L2 范数；actual=32.008255 与"全部 1024 个物理元素参与计算"的范数一致——kernel 把 stride 间隔里的无效元素也算了进去。

## 适用场景

- 工程形态：aclnn 两段式接口 + AscendC 自定义算子（op_host tiling + op_kernel），ST Real 精度回归。
- API 路径：`aclCreateTensor`（view/stride/storage 三组描述）、tiling 侧 `gert::TilingContext::GetDynamicInputShape/GetDynamicInputStride`、`gert::StorageShape::GetStorageShape/GetOriginShape`、算子定义 `.AutoContiguous()`、aclnn 层 `l0op::Contiguous`、kernel 侧 `SetGlobalBuffer + DataCopyPad`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有以连续布局假设实现、又被以带 stride 的 aclTensor 直调的算子。
- 路线优先级：safety_fallback（诊断完成；修复方向明确但端到端验证未闭环）。fallback 声明：连续输入场景不受影响；非连续输入在本卡证据范围内仍失败。
- 明确不适用 / 未验证场景：GE 图模式路径下 AutoContiguous/GetDynamicInputStride 的行为未验证；多维（ndim>1）非连续未验证。

## 判别方法（机制链与探针）

按以下链条逐层判别（每步都有实测输出支撑）：

1. **数值判别**：actual 是否等于"物理全量"的范数。golden（逻辑 512 元素）=22.619326，actual=32.008255 与物理 1024 元素全量范数吻合——kernel 在按物理存储计算，不是精度漂移。
2. **tiling 侧证据**：原 tiling 用物理 shape 取元素数：

```cpp
    for (size_t k = 0; k < tensorNum; k++) {
        auto shape = context->GetDynamicInputShape(0, k);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t elemCount = shape->GetStorageShape().GetShapeSize();   // 非连续时=1024（物理）
        tiling->elemCounts[k] = elemCount;
        totalElem += elemCount;
    }
```

3. **kernel 侧证据**：GM 缓冲区按连续区间设置、`DataCopyPad` 连续搬运：

```cpp
            __gm__ D_T_X* tensorGM = GetTensorAddr(static_cast<uint16_t>(k), xPtr_);
            GlobalTensor<D_T_X> inTensorGM;
            inTensorGM.SetGlobalBuffer(tensorGM + localStart, localCount);
            // ...
            DataCopyPad(inputLocal, inTensorGM[chunkStart], copyParams, padParams);
```

4. **修复尝试 A（tiling 改逻辑 shape + TilingData 传 stride + kernel stride 分支）**：TilingData 增加 `strides[MAX_TENSOR_NUM]`/`maxStride`，tiling 改用 `GetOriginShape().GetShapeSize()` 并按 storage/origin 推 stride，kernel 增加 stride>1 分支（拷物理跨度→向量 Cast 升 FP32→标量提取）。结果 **actual 仍为 32.008255，一字不变**。
5. **修复尝试 B（改用专用 stride API）**：tiling 改用 `context->GetDynamicInputStride(0, k)`（连续返回 nullptr、非连续返回 `Stride*`）：

```cpp
    for (size_t k = 0; k < tensorNum; k++) {
        auto shape = context->GetDynamicInputShape(0, k);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        // 物理元素数 (StorageShape): 非连续张量包含 stride 间隔的无效元素
        int64_t storageSize = shape->GetStorageShape().GetShapeSize();

        // 检测非连续: GetDynamicInputStride 对连续张量返回 nullptr,
        // 对非连续张量返回 Stride 指针 (aclnn 路径中 OriginShape == StorageShape)
        int64_t stride = 1;
        const gert::Stride* strideInfo = context->GetDynamicInputStride(0, k);
        if (strideInfo != nullptr && strideInfo->GetDimNum() >= 1) {
            stride = strideInfo->GetStride(0);
            if (stride < 1) stride = 1;
        }
        tiling->strides[k] = stride;
        if (stride > tiling->maxStride) {
            tiling->maxStride = stride;
        }

        // 逻辑元素数 = 物理元素数 / stride (1D 非连续时 stride > 1)
        int64_t elemCount = (stride > 1) ? (storageSize / stride) : storageSize;
        tiling->elemCounts[k] = elemCount;
        totalElem += elemCount;
    }
```

结果 **actual 仍为 32.008255，一字不变** → stride 信息在 aclnn→L0 链路上没有传到 tiling（`GetDynamicInputStride` 在该路径对非连续张量也返回 nullptr 的观测推断）。
6. **算子定义侧证据**：`op_host/*_def.cpp` 中输入已声明 `.AutoContiguous()`，但 aclnn 直调路径不自动生效；aclnn L2 实现中"autogen 代码已处理 Contiguous"的注释与实测不符。
7. **结论方向**：问题不在 kernel/tiling 代码，而在 **stride 元信息在 aclnn 直调链路丢失**；修复位置应上移到 aclnn L2 层——对 TensorList 每个张量显式 `l0op::Contiguous(xTensor, executor)` 后重建 TensorList 再调 L0（该修复已实现，端到端验证因套件超时/隔离问题未闭环，列为遗留优先项）。

判别口诀：**多轮 host/kernel 修复后 actual 完全不变，说明修改没有到达真正生效的层——先查输入描述（shape/stride）在框架内的传递路径，再改 kernel。**

## 根因

算子按连续布局假设实现（tiling 取 `GetStorageShape().GetShapeSize()` 物理元素数 + kernel `SetGlobalBuffer/DataCopyPad` 连续读），而 aclnn 直调路径下 stride 元信息丢失（`GetDynamicInputStride` 返回 nullptr、`AutoContiguous()` 不生效、OriginShape 与 StorageShape 同为物理 shape），导致非连续输入被当作物理连续全量数据计算。

## 验证信号

本卡为诊断规则（允许没有成功修复），信号如下：

- 触发：非连续 stride=2 用例 FAIL，golden=22.619326, actual=32.008255, rel_err=4.15e-01，MERE/MARE=4.15e-01 超阈 1.22e-04/1.22e-03；其余 73 用例 PASS（来源平台：950；证据等级：tool_observed）。
- 修复尝试 A 后：同用例 FAIL 输出数值完全不变（actual=32.008255）（来源平台：950；证据等级：tool_observed）。
- 修复尝试 B（GetDynamicInputStride）后：同用例 FAIL 输出数值仍完全不变（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 端到端修复（aclnn 层 Contiguous 后该用例通过）的验证未完成，本卡不宣称该修复已生效。
- 只覆盖 1D stride 场景与 aclnn 直调路径；GE 图模式下 AutoContiguous/GetDynamicInputStride 行为未验证（未验证，不推断）。
- "actual=物理全量范数"的等值验证是数值推断（按 golden 与实际缓冲区内容对比），未单独打印物理全量范数。

**未决项**：

- 优先项：完成 aclnn 层显式 `l0op::Contiguous` 的端到端验证（该用例通过 + 全量无回归），并确认 `GetDynamicInputStride` 在 aclnn 直调路径返回 nullptr 的确切机制层原因。
- ndim>1 非连续、负 stride/offset 场景未覆盖。
## 坏实践

- 反复修改 kernel/tiling 的 stride 处理而不验证 tiling 实际收到的输入描述——两次结构性修复均因元信息未到达而无效。
- 相信代码注释（"autogen 已处理 Contiguous"）而不实测。

## 误诊

- 修复尝试 A 阶段曾推断 `GetOriginShape()` 会返回逻辑视图 [512]——实测该路径下 OriginShape 与 StorageShape 一致（都是物理 shape），推断错误，由"actual 不变"证伪。
- 曾怀疑算子定义 `.AutoContiguous()` 会自动转连续——实测不生效，证伪。
