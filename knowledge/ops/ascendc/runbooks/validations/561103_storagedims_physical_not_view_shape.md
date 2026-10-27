---
type: "Runbook"
title: 561103 非连续输入：storageDims 须传物理 shape
description: aclnn 测试程序用 aclCreateTensor 构造 stride≥2 的 slice 非连续输入时，storageDims 传逻辑 view shape 会让 GetWorkspaceSize 存储校验越界报 561103；storageDims 必须描述物理缓冲区大小。
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:39:38Z'
updated_at: '2026-09-17T01:39:38Z'
platforms: ["950"]
---

# 561103 非连续输入：storageDims 须传物理 shape

## 现象

为一个已完成开发的自定义 aclnn 算子（elementwise sigmoid，fp32/fp16/bf16）补齐 ST 测试的非连续 Tensor 用例：C++ 原生测试程序里，SLICE 模式构造 1D 逻辑 shape `{1003}`、strides `{2}` 的输入（物理缓冲区按 `2 * 1003 = 2006` 个元素分配），TRANSPOSE 模式构造 2D shape `{37, 29}`、strides `{1, 37}` 的输入。测试程序先 `aclnnXxxGetWorkspaceSize(...)` 再 `aclnnXxx(...)`。

在 NPU 上跑 Real 模式全量用例，结果 99/102：三个 dtype 的 SLICE 用例全部失败，报错原文一致：

```
[Real-NC] L3_fp32_nc_slice (dtype=float32, shape_size=1003, ncMode=1)
  [FAIL] GetWorkspaceSize 失败: 561103

[Real-NC] L3_fp16_nc_slice (dtype=float16, shape_size=1003, ncMode=1)
  [FAIL] GetWorkspaceSize 失败: 561103

[Real-NC] L3_bf16_nc_slice (dtype=bfloat16, shape_size=1003, ncMode=1)
  [FAIL] GetWorkspaceSize 失败: 561103
```

而同样走非连续路径的 TRANSPOSE 用例（三个 dtype）全部 PASS，连续输入用例、64M 大 shape 用例、L2 异常用例也全部通过；Mock 模式（不真正调用算子）102/102 通过。当时测试代码里的张量构造函数 `CreateAclTensorWithStrides` 把**逻辑 view shape** 直接当作 `aclCreateTensor` 的 storageDims（第 7/8 个参数）传入。

## 适用场景

- 工程形态：aclnn 自定义算子的 C++ 原生 ST 测试程序（也可用于任何需要手工构造非连续 `aclTensor` 的 acl 调用方）。
- API 路径：`aclCreateTensor(viewDims, viewDimNum, dataType, strides, offset, format, storageDims, storageDimNum, address)` → `aclnnXxxGetWorkspaceSize` 的入参校验链。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何走 aclnn 且测试需要构造 stride≠1 非连续输入的算子。
- 路线优先级：recommended_fastpath（语义正确性修复，非绕过）。
- 明确不适用 / 未验证场景：连续 tensor（storageDims=viewDims 恰好等价，无此问题）；算子本体缺陷（如 tiling 未实现）导致的 561103（另有根因，本卡不覆盖）；高维组合非连续视图（同时 slice+transpose）未验证；其他 NPU 架构未验证。

## 根因

`aclCreateTensor` 的 storageDims 参数描述的是**底层物理缓冲区的形状（元素总数）**，GetWorkspaceSize 阶段会用它校验"按 strides/offset 展开的全部访问是否落在声明存储内"；把它误传成逻辑 view shape 后，对 stride≥2 的视图必然越界。

机制链（本会话实测推导）：

1. SLICE 用例：逻辑 `{1003}`、strides `{2}`、offset 0 → 最大访问偏移 `(1003-1) × 2 = 2004` 个元素，物理缓冲区实际有 2006 个元素，但声明的 storageDims `{1003}` 只有 1003 → 2004 ≥ 1003，校验检出越界，`GetWorkspaceSize` 返回 561103。
2. TRANSPOSE 用例：逻辑 `{37, 29}`、strides `{1, 37}` → 物理大小 `37 × 29 = 1073` 与逻辑大小**相等**（转置不扩缓冲），storageDims 传逻辑 shape 时总元素数恰好正确 → 不触发。
3. 这解释了"同为非连续、TRANSPOSE 过而 SLICE 挂"的判别特征：**凡是物理缓冲区大于逻辑视图的非连续构造（slice/stride 采样）才会触发；物理=逻辑的（transpose/permute）不会**。
4. Mock 模式不调用真实 GetWorkspaceSize 校验，所以 Mock 全绿掩盖了该问题，只在 Real 模式暴露。

## 解决配方

修复前（`tests/st/test_aclnn_bench_no_harness_sigmoid.cpp`，storageDims 误传逻辑 shape——两个分支同样错误）：

```cpp
// 创建带自定义 stride 的非连续 ACL Tensor
int CreateAclTensorWithStrides(const std::vector<uint8_t>& physicalData,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& strides,
                                int64_t storageOffset,
                                int64_t physicalNumElements,
                                aclDataType dataType,
                                size_t elemSize,
                                void** deviceAddr,
                                aclTensor** tensor) {
    size_t physicalBytes = static_cast<size_t>(physicalNumElements) * elemSize;

    if (physicalBytes == 0) {
        *deviceAddr = nullptr;
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                                   storageOffset, aclFormat::ACL_FORMAT_ND,
                                   shape.data(), shape.size(), *deviceAddr);
        return ACL_SUCCESS;
    }

    auto ret = aclrtMalloc(deviceAddr, physicalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, physicalBytes, physicalData.data(), physicalBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        return ret;
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                               storageOffset, aclFormat::ACL_FORMAT_ND,
                               shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
```

修复后（storageShape 统一传物理缓冲区大小的一维 shape）：

```cpp
// 创建带自定义 stride 的非连续 ACL Tensor
int CreateAclTensorWithStrides(const std::vector<uint8_t>& physicalData,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& strides,
                                int64_t storageOffset,
                                int64_t physicalNumElements,
                                aclDataType dataType,
                                size_t elemSize,
                                void** deviceAddr,
                                aclTensor** tensor) {
    size_t physicalBytes = static_cast<size_t>(physicalNumElements) * elemSize;

    if (physicalBytes == 0) {
        *deviceAddr = nullptr;
        // storageShape 使用物理缓冲区大小 (1D), 确保非连续视图的越界访问被正确检测
        std::vector<int64_t> storageShape = {physicalNumElements};
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                                   storageOffset, aclFormat::ACL_FORMAT_ND,
                                   storageShape.data(), 1, *deviceAddr);
        return ACL_SUCCESS;
    }

    auto ret = aclrtMalloc(deviceAddr, physicalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, physicalBytes, physicalData.data(), physicalBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        return ret;
    }

    // storageShape 使用物理缓冲区大小 (1D), 确保非连续视图的所有元素都在合法范围内
    std::vector<int64_t> storageShape = {physicalNumElements};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                               storageOffset, aclFormat::ACL_FORMAT_ND,
                               storageShape.data(), 1, *deviceAddr);
    return ACL_SUCCESS;
}
```

配套的非连续布局计算（SLICE 物理缓冲区 = 2×逻辑大小；TRANSPOSE 物理 = 逻辑大小，这是根因判别的数据来源）：

```cpp
// 根据非连续模式计算 stride 和物理缓冲区大小
struct NonContiguousLayout {
    std::vector<int64_t> strides;   // 逻辑 shape 对应的 stride
    int64_t storageOffset = 0;      // 存储偏移 (元素数)
    int64_t physicalNumElements = 0; // 物理缓冲区大小 (元素数)
};

NonContiguousLayout ComputeNonContiguousLayout(const std::vector<int64_t>& shape,
                                                 NonContiguousMode mode) {
    NonContiguousLayout layout;
    int64_t logicalSize = GetShapeSize(shape);

    if (mode == NonContiguousMode::SLICE) {
        // 1D slice: stride=2, 隔元素取值
        // 物理 buffer 大小 = 2 * logicalSize
        layout.strides = {2};
        layout.storageOffset = 0;
        layout.physicalNumElements = 2 * logicalSize;
    } else if (mode == NonContiguousMode::TRANSPOSE) {
        // 2D transpose: TestCase shape = 转置后的 view shape (N, M)
        // 原始 (M, N) 连续 stride = (N, 1), 转置后 stride = (1, N)
        // 其中 N = shape[0] (转置 view 的第一维 = 原始第二维)
        int64_t origRowStride = shape[0];  // N = 原始行宽
        layout.strides = {1, origRowStride};
        layout.storageOffset = 0;
        layout.physicalNumElements = GetShapeSize(shape);  // N * M = logicalSize
    } else {
        layout.strides = ComputeStrides(shape);
        // ... 连续布局分支
    }
    return layout;
}
```

判别 / 定位方法：

1. 先看失败分布：同为非连续，TRANSPOSE 过而 SLICE 挂 → 差异只在"物理缓冲区是否大于逻辑视图"，与算子的非连续支持逻辑无关。
2. 手工算一遍最大访问偏移：`maxOffset = offset + Σ (dim_i - 1) × stride_i`，与传入的 storageDims 总元素数比较；本例 2004 > 1003 即越界。
3. 确认被测算子声明支持非连续输入（需求文档/实现里有 `l0op::Contiguous` 处理），排除"算子不支持"方向。

## 设计决策

- 为什么 storageShape 用一维 `{physicalNumElements}` 而不是复刻视图维度：物理缓冲区本来就是一维线性分配，storageDims 的作用是声明存储总量，一维物理大小对任意视图都成立，也避免为每种非连续模式推导等价的高维存储形状。
- 为什么空 buffer 分支也要同步修：空 tensor（physicalBytes==0）同样构造 `storageShape={physicalNumElements}`，保持两条分支语义一致，避免空 tensor 非连续用例踩同一个坑。
- 为什么不把 strides 改回连续再测：那是绕过被测能力（非连续输入支持），测试就失去意义；修张量描述符才是对测试脚手架 bug 的实质修复。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 手工构造非连续 `aclTensor` 时按三件套自查：viewDims/strides/offset 描述**视图**，storageDims 描述**物理缓冲区**；两者只在连续 tensor 时才相等。
- 写非连续用例时同时保留"物理缓冲区大于逻辑"（slice/stride 采样）与"物理=逻辑"（transpose）两类模式：只有前者能暴露 storageDims 误传。
- Real 模式回归必须跑：Mock 模式不触发 GetWorkspaceSize 真实校验，这类描述符错误在 Mock 下不可见。

## 验证信号

以下信号来自该算子 ST 测试补齐会话：SLICE 用例首轮 Real 模式失败定位后修改张量构造函数，再全量回归。

- 修复前：Real 模式 99/102，SLICE 非连续用例 fp32/fp16/bf16 三例均 `[FAIL] GetWorkspaceSize 失败: 561103`，TRANSPOSE 三例 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后：重编译 Real 模式全量 102/102 通过，`Real-NC` 六个非连续用例（SLICE+TRANSPOSE × 3 dtype）全部 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后：Mock 模式回归 102/102 通过，测试报告"总计: 102, 通过: 102, 失败: 0, 跳过: 0"（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"storageDims 误传逻辑 shape"这一根因；同码 561103 在其他场景另有根因（如 tiling 未实现），需按本卡判别方法另行定位。
- `storageShape={physicalNumElements}` 一维写法在本会话的 1D SLICE / 2D TRANSPOSE 上验证；更高维、带非零 storageOffset 或组合非连续（slice 后再 transpose）的视图未验证。
- GetWorkspaceSize 内部用 storageDims 做校验的具体逻辑（按元素数还是按字节）未读实现验证，机制链是从修复前后行为反推。

**未决项**：

- 复现并追根因到机制层：读 aclnn GetWorkspaceSize 的入参校验实现，确认 561103 的官方语义与校验用 storageDims 的确切规则（本卡机制链为行为反推）。
- 非零 storageOffset、组合非连续视图下 storageDims 的推荐写法未验证。
## 坏实践

- 把 `shape.data(), shape.size()` 同时当 viewDims 和 storageDims 传给 `aclCreateTensor`（连续时恰好等价，非连续时静默埋雷）。
- 只用 TRANSPOSE 模式做非连续覆盖就认为测试完备——它物理=逻辑，暴露不了 storageDims 误传。
- 遇到 561103 直接归因为"算子不支持非连续输入"而给算子加 Contiguous 预处理或放弃用例。

## 误诊

- 最初按错误码猜"算子不支持 1D stride=2 非连续 / stride 配置非法"：核对算子需求与实现（op_api 侧有 `l0op::Contiguous` 处理）后排除；真因在测试侧张量描述符（由修复生效证实）。
