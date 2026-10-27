---
type: "Runbook"
title: "非连续输入 ST 构造：转置 stride 张量+gather golden"
description: C++ ST 用 aclCreateTensor(viewShape, viewStrides, storageShape) 构造转置型真非连续输入、按 stride gather 量化后数据计算 golden，配套空 tensor/最小/非对齐/最大元素数边界用例矩阵，Mock 113/113 + Real 115/115 验证。
tags: [task.testing, topic.data_movement.layout_transform, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:38:37Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# 非连续输入 ST 构造：转置 stride 张量+gather golden

## 现象

在 Ascend950PR（CANN 9.1.0）上为 BenchNoHarnessSwiGlu（SwiGLU）做 ST 测试工程扩展：需要为"非连续输入 + 边界 shape"建立可信的 C++ 原生测试——此前 ST 只覆盖连续输入，`aclCreateTensor` 也只按连续 stride 构造张量，无法验证任何非连续处理链路。扩展完成后新增 40 条用例（非连续 12 + 空 tensor 9 + 最小 shape 6 + 非对齐 9 + 最大元素数 4，覆盖 FP16/FP32/BF16），Mock 模式 113/113、Real NPU 模式 115/115 全部通过，CPU golden 自测 9/9（含 3 项新增非连续/空 tensor 校验）。

## 适用场景

- 工程形态：aclnn 自定义算子 C++ 原生 ST（Mock：CPU golden 自测；Real：NPU 输出 vs CPU golden）。
- API 路径：`aclCreateTensor`（viewShape/viewStrides/storageShape 三元组）、`aclrtMalloc/aclrtMemcpy`、测试侧自定义 golden。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`——所有需要非连续输入/边界 shape 覆盖的 aclnn 算子 ST。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：本配方验证的是"框架 Contiguous 转换 + kernel"全链路的正确性（当时 kernel 的 TilingKey=2 自有非连续路径尚未启用）；自有非连续 kernel 路径启用后应复用同一套用例矩阵重新回归（接续点）。另一种非连续构造形态是 padded 存储（每行补 PAD 列、strides=[cols+PAD,1]，见同域 padded 存储构造卡），适合验证 Contiguous 转换；本卡转置视图更贴近框架最常见的用户态非连续（`tensor.T`）。

## 解决配方

四个组件（verbatim 取自该会话 ST 写入事件）：

**1. 转置构造非连续视图**——原始连续 tensor 的 stride 交换两维，view shape 同步交换：

```cpp
static std::vector<int64_t> ComputeTransposedStrides(const std::vector<int64_t>& origShape,
                                                      int64_t dimA, int64_t dimB) {
    auto strides = ComputeStrides(origShape);
    std::swap(strides[dimA], strides[dimB]);
    return strides;
}
```

**2. 创建带自定义 stride 的 aclTensor**——物理存储按 storageShape 分配，逻辑视图按 viewShape+viewStrides 描述：

```cpp
static int CreateAclTensorWithStrides(const std::vector<uint8_t>& hostData,
                                       const std::vector<int64_t>& viewShape,
                                       const std::vector<int64_t>& viewStrides,
                                       const std::vector<int64_t>& storageShape,
                                       aclDataType dataType,
                                       void** deviceAddr,
                                       aclTensor** tensor) {
    size_t typeSize = 0;
    switch (dataType) {
        case ACL_FLOAT16: typeSize = 2; break;
        case ACL_FLOAT:   typeSize = 4; break;
        case ACL_BF16:    typeSize = 2; break;
        default:          typeSize = 4; break;
    }
    size_t storageElemCount = static_cast<size_t>(GetShapeSize(storageShape));
    size_t totalBytes = storageElemCount * typeSize;

    auto ret = aclrtMalloc(deviceAddr, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, totalBytes, hostData.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    *tensor = aclCreateTensor(viewShape.data(), static_cast<int64_t>(viewShape.size()), dataType,
                               viewStrides.data(), 0, aclFormat::ACL_FORMAT_ND,
                               storageShape.data(), static_cast<int64_t>(storageShape.size()), *deviceAddr);
    return ACL_SUCCESS;
}
```

**3. stride 感知的 gather/scatter**——golden 侧把物理存储按 stride 收集为逻辑连续（反向 scatter 用于需要回写的场合）：

```cpp
static void GatherToContiguous(const float* storage,
                                const std::vector<int64_t>& viewShape,
                                const std::vector<int64_t>& viewStrides,
                                float* contiguous) {
    int64_t rank = static_cast<int64_t>(viewShape.size());
    int64_t total = GetShapeSize(viewShape);
    if (total == 0 || rank == 0) return;

    std::vector<int64_t> indices(rank, 0);
    for (int64_t linearIdx = 0; linearIdx < total; linearIdx++) {
        int64_t offset = 0;
        for (int64_t d = 0; d < rank; d++) {
            offset += indices[d] * viewStrides[d];
        }
        contiguous[linearIdx] = storage[offset];

        // 递增坐标（row-major 顺序）
        for (int64_t d = rank - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < viewShape[d]) break;
            indices[d] = 0;
        }
    }
}
```

**4. 非连续 golden 与用例矩阵**——golden 必须从"发往设备的物理存储字节"decode 后 gather 再计算（保持 golden 输入 ≡ kernel 输入，见[ST golden 须用 dtype 转换后数据](golden_input_dtype_aligns_npu_input.md)）：

```cpp
    std::vector<float> golden(outputElemCount);
    if (tc.isNonContiguous) {
        // 非连续: decode 物理存储 → gather 为逻辑连续 → golden
        size_t storageElemCount = static_cast<size_t>(tc.storageSize);
        auto quantizedStorage = DecodeData(inputBytes.data(), storageElemCount, aclDtype);
        std::vector<float> gatheredInput(inputElemCount);
        GatherToContiguous(quantizedStorage.data(), tc.inputShape, tc.inputStrides, gatheredInput.data());
        ComputeSwiGluGolden(gatheredInput.data(), tc.inputShape, tc.dim, golden.data());
    } else {
        auto quantizedInput = DecodeData(inputBytes.data(), inputElemCount, aclDtype);
        ComputeSwiGluGolden(quantizedInput.data(), tc.inputShape, tc.dim, golden.data());
    }
```

用例注册（`TestCase` 增加 `isNonContiguous/storageSize/inputStrides/storageShape` 字段后）：

```cpp
    auto addNonContCase = [&](const std::string& name,
                              const std::vector<int64_t>& origShape,
                              int64_t transposeDimA, int64_t transposeDimB,
                              const std::string& dtype, int64_t dim,
                              double vmin, double vmax) {
        TestCase tc;
        tc.name = name;
        tc.dtype = dtype;
        tc.dim = dim;
        tc.valueMin = vmin;
        tc.valueMax = vmax;
        tc.fromCsv = false;
        tc.isNonContiguous = true;

        // 转置后的 view shape = 交换两个维度
        tc.inputShape = origShape;
        std::swap(tc.inputShape[transposeDimA], tc.inputShape[transposeDimB]);

        // 转置后的 stride = 原始 stride 交换两个维度
        tc.inputStrides = ComputeTransposedStrides(origShape, transposeDimA, transposeDimB);

        // 物理存储元素数 = 原始 shape 的元素数
        tc.storageSize = GetShapeSize(origShape);
        ...
    };
    // 2D transpose(0,1): origShape=[4,8] → viewShape=[8,4], dim=-1（fp16/fp32/bf16 各一）
    addNonContCase("noncont_fp16_2d_trans_[8,4]_dim-1",  {4, 8},    0, 1, "float16",  -1, -1.0, 1.0);
```

配套边界用例：空 tensor（`[4,0]/[0,8]/[2,0,4]`，Real 模式 `inputElemCount==0` 快速 PASS）、最小 shape（`[2,2]` dim=-1/0）、非对齐 shape（如 `[4,6]`，元素数非 32B 对齐）、最大元素数（大 shape 多核）。

## 设计决策

- 为什么用 transpose 构造而非手写任意 stride：转置视图是框架最常交付的非连续形态（用户侧 `tensor.T`），物理存储仍是合法 row-major 原始数据，golden 的 gather 逻辑可独立自测；任意 stride 需另行证明其物理布局合法。
- 为什么物理存储按 storageShape 整体分配、而不是按视图元素数分配：stride 视图的合法访问范围以物理存储为准（offset 可达 `max(indices[d]*viewStrides[d])`），按视图元素数分配会越界。
- 为什么 golden 从 decode 后的物理存储 gather：与"golden 输入 ≡ kernel 输入"铁律一致（见[ST golden 须用 dtype 转换后数据](golden_input_dtype_aligns_npu_input.md)），非连续下还要叠加 stride gather 这一步。
- 空 tensor 在 Real 模式直接判 PASS：`aclnn` 侧对空输入有快速返回路径，比对无意义；Mock 模式仍验证 CPU golden 对空输入的健全性。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0；ST 工程 `tests/st/`（CMake 双模式：`build-mock` USE_MOCK=ON / `build-real` 直连 aclnn）。
- 相关文件（与事件流读写一一对应）：
  - `tests/st/test_aclnn_bench_no_harness_swi_glu.cpp`——非连续辅助函数、CreateAclTensorWithStrides、RunMockTest/RunRealTest 非连续分支、GetBuiltinTestCases 用例矩阵。
  - `tests/st/run.sh --mock|--real`——双模式入口。

## 完整代码

核心四个函数与用例注册的完整 verbatim 版本见上节《解决配方》（`ComputeTransposedStrides` / `CreateAclTensorWithStrides` / `GatherToContiguous` / `ScatterFromContiguous` 与 `addNonContCase`；`ScatterFromContiguous` 与 `GatherToContiguous` 同构，方向相反，按 stride 散布回物理存储）。

## 性能影响

不适用（测试工程配方，无运行时性能主张）。

## 预防

- 写非连续 ST 前先确认算子当前实际走的链路（框架 Contiguous 还是自有 kernel 路径），避免"用例全绿"造成覆盖错觉。
- 非连续 golden 一律走"decode 物理存储 → gather → 计算"，不要对 gather 前的数据直接算。

## 验证信号

以下信号来自该算子 ST 测试工程开发会话。

- Mock 模式：113/113 通过、0 失败 0 跳过；CPU golden 自测 9/9（含新增负值/非连续/空 tensor 校验）（来源平台：950；证据等级：tool_observed）。
- Real NPU 模式：115/115 通过、0 失败 0 跳过，含 `noncont_fp16/fp32/bf16_2d/3d_trans_*` 12 例、`empty_*` 9 例、`min_*` 6 例、`unaligned_*` 9 例、`max_*` 4 例（来源平台：950；证据等级：tool_observed）。
- 用例构造正确性旁证：转置视图经框架 Contiguous 链路回读后与 gather golden 全部匹配（若 stride/存储构造错误，比对会立即失配）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本配方验证的是"非连续输入 → 框架 Contiguous → 连续 kernel"链路；自有 TilingKey=2 kernel 路径启用后需用同一矩阵重回归（该启用工作跨会话）。
- transpose 形态覆盖 2D/3D；更高维、负 stride、重叠视图（as_strided 类）未构造未验证。
- 空 tensor 快速 PASS 的语义依赖算子 L2 层确有空输入早退——无该路径的算子须实测。
- 仅 Ascend950PR + CANN 9.1.0 实测；其他平台/版本 `aclCreateTensor` 的 strides 语义未验证（未验证，推断可迁移——stride 视图语义为 aclnn 框架行为）。

**未决项**：

- 待自有非连续 kernel 路径（TilingKey=2）启用后，用本矩阵重跑 Real 回归——这是下个会话的直接接续点。
- 高维/负 stride/重叠视图用例未构造（未尝试）。
## 坏实践

- 按视图元素数（而非 storageShape）分配物理存储——stride 视图访问越界。
- 非连续 golden 直接用编码前 FP32 原数据计算——量化 + 布局双重不一致。

## 误诊

- 无（本卡为成功实现的配方提取；实现过程中无失败-重试闭环）。
