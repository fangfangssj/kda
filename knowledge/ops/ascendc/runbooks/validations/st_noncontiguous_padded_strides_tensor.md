---
type: "Runbook"
title: ST 非连续 tensor 构造：padded 存储加 strides view
description: "验证 aclnn 层 l0op::Contiguous 的非连续输入处理：把 [rows,cols] 数据存进 [rows,cols+PAD] padded 存储，aclCreateTensor 传 view shape + strides=[cols+PAD,1] + storageShape，驱动 Contiguous 真实执行。"
tags: [task.testing, topic.implementation, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# ST 非连续 tensor 构造：padded 存储加 strides view

## 现象（任务）

Mish 类激活算子的 aclnn 层在执行前会对输入做 `l0op::Contiguous`（非连续输入先规整成连续）。测试设计要求覆盖"非连续 tensor 输入"用例（FP16/FP32/BF16 各一条 2D 用例），但现有脚手架 `CreateAclTensor` 只会创建 stride 连续的 tensor（`ComputeStrides` 按密集布局算）——非连续路径实际从未被执行，用例名存实亡。需要一个能真实构造"view shape 与存储布局不一致"的 tensor 的脚手架函数。

## 适用场景

- 工程形态：aclnn 算子 C++ ST 测试（Real 模式），验证 op_api 层 `l0op::Contiguous`/`l0op::ViewCopy` 等对非连续输入的规整行为。
- API 路径：`aclCreateTensor(shape, dimNum, dataType, strides, offset, format, storageShape, storageDimNum, deviceAddr)`——view 的 shape 与 strides/storageShape 分离是其关键能力。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——脚手架构造手法与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：高维（>2D）非连续（切片/转置型 stride 模式）未实现，原理同构；Mock 模式无 NPU 调用，非连续用例在 Mock 下只验 golden 计算本身。

## 根因/配方要点

`aclCreateTensor` 支持逻辑 view：`shape` 描述逻辑形状（算子看到的形状），`strides` 描述逻辑元素在存储中的步长，`storageShape` 描述实际分配的物理布局。构造非连续 2D 输入的手法：物理存储按 `[rows, cols+PAD]` 分配（每行多 PAD 个填充元素，填充 0），数据按行拷进前 `cols` 列；`aclCreateTensor` 传 `shape=[rows,cols]`、`strides=[cols+PAD,1]`、`storageShape=[rows,cols+PAD]`。这样算子收到的 tensor 的 view shape 正常，但 `IsContiguous` 判定为假，驱动 `l0op::Contiguous` 真实执行。输出 tensor 保持连续创建，便于 D2H 读回比对。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0。
- 相关文件：`tests/st/test_aclnn_bench_no_harness_mish.cpp`（`CreateAclTensorNonContig2D` + 三个 Real 模式运行函数的 `non_contiguous` 分支）；aclnn 层 `op_api/bench_no_harness_mish.cpp`（`l0op::Contiguous` 调用点）。

## 完整代码

非连续 tensor 构造函数（verbatim）：

```cpp
// 创建非连续 tensor（2D [rows, cols] 存储在 [rows, cols+PAD] 中，strides=[cols+PAD, 1]）
// 用于验证 aclnn 层 l0op::Contiguous 非连续输入处理
template<typename T>
int CreateAclTensorNonContig2D(const std::vector<T>& hostData,
                               const std::vector<int64_t>& shape,
                               void** deviceAddr,
                               aclDataType dataType,
                               aclTensor** tensor) {
    constexpr int64_t PAD = 4;  // 每行 padding 4 元素
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    int64_t strideRow = cols + PAD;
    int64_t storageCount = rows * strideRow;

    // 构建带 padding 的存储（每行前 cols 个为数据，后 PAD 个为 0）
    std::vector<T> padded(storageCount, T{});
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            padded[static_cast<size_t>(r * strideRow + c)] = hostData[static_cast<size_t>(r * cols + c)];
        }
    }

    size_t byteSize = static_cast<size_t>(storageCount) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, byteSize, padded.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    // strides=[strideRow, 1], storageShape=[rows, strideRow]
    std::vector<int64_t> strides = {strideRow, 1};
    std::vector<int64_t> storageShape = {rows, strideRow};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
```

Real 模式运行函数的接入分支（三个 dtype 各一处，FP16 示例，verbatim）：

```cpp
    // 创建输入 tensor（非连续时使用 padded stride 验证 Contiguous 路径）
    if (tc.non_contiguous) {
        if (CreateAclTensorNonContig2D(input, tc.shape, &x_dev, ACL_FLOAT16, &x_tensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 x tensor (非连续) 失败");
            return false;
        }
    } else {
        if (CreateAclTensor(input, tc.shape, &x_dev, ACL_FLOAT16, &x_tensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 x tensor 失败");
            return false;
        }
    }

    // 创建输出 tensor（连续，便于读回验证）
    std::vector<fp16_t> output_init(size, static_cast<fp16_t>(0.0f));
```

用例定义处给目标用例置 `non_contiguous = true`（如 BF16 非连续 [4,8]），输出恒为连续 tensor，D2H 读回后与连续 golden 直接比对。

## 设计决策

- **PAD=4 而不是 1**：padding 必须让"行尾到下一行头"的间隔跨过元素对齐粒度，1 个元素 padding 即可打破连续性；取 4 留余量且不影响语义（填充 0 不参与计算）。
- **输出 tensor 保持连续**：被测行为是"输入非连续的规整"，输出若也非连续，读回比对还要做逆 stride 采样，徒增脚手架复杂度。
- **golden 不变**：非连续只是存储布局，逻辑元素序列与连续用例相同，CPU golden 直接复用。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

以下信号来自该算子非连续用例开发与全功能验证会话：

- Mock 模式：52 条用例（含 3 条非连续用例：FP16/FP32/BF16 [4,8] padded 存储）52/52 通过，Golden 自测 36/36（来源平台：950；证据等级：tool_observed）。
- 上板（Real）：非连续 tensor 用例（与空 tensor、0 维标量、±1e5 极值、NaN/Inf 等边界）全部通过，`l0op::Contiguous` 对 padded stride 输入规整正确，无精度回归（来源平台：950；证据等级：tool_observed）。

注：Real 模式逐条运行的详细日志在本轨迹记录范围之外，此处以全功能验证的完成汇总为验证锚点；Mock 与 Real 双模式下用例集与判定逻辑一致。

## 失效边界

- 仅 2D 行 padding 一种非连续形态；转置（strides=[1,rows]）、切片（offset≠0）、跨维广播型 stride 未覆盖，扩展时注意 `aclCreateTensor` 的 offset 参数与 storageShape 匹配。
- `l0op::Contiguous` 规整后的中间表示对算子透明；若算子自身（而非 aclnn 层）要求特定布局，本脚手架不足以暴露该类问题。
- Real 模式详细逐条日志在本轨迹记录范围之外，单条用例的 MERE 数值未逐字恢复（汇总结果为通过）。

**未决项**：

- 高维/转置型非连续用例未实现；如后续覆盖，建议按同一手法扩展 strides/storageShape 组合。
## 坏实践

- 用例标了"非连续"但脚手架只会建连续 tensor——用例名存实亡，Contiguous 路径零覆盖。
- 把 view shape 直接改成存储 shape（[rows,cols+PAD]）来"制造非连续"——那改变的是逻辑形状而非布局，算子会多算 PAD 列。

## 误诊

- 无。
