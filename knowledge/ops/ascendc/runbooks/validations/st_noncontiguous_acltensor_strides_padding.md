---
type: "Runbook"
title: ST 构造真非连续 aclTensor 验证 Contiguous 路径
description: "用物理 padding buffer + aclCreateTensor(逻辑 shape, strides, storage_dims) 构造真实非连续 aclTensor，让 ST 真正测到算子 L2 层 l0op::Contiguous 的非连续输入处理路径；避免\"用例标了 non_contiguous 实际仍走连续创建\"的假覆盖。"
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# ST 构造真非连续 aclTensor 验证 Contiguous 路径

## 现象

自定义算子（MoE finalize routing 类）的 C++ ST 用例表中有 non_contiguous 标记的用例（验证算子对非连续输入的处理），但测试工程最初的 Real 模式仍用连续 tensor 创建函数构造输入——名义上是非连续用例，实际只跑了连续路径，联调无法验证算子 L2 层 `l0op::Contiguous`（把非连续 aclTensor 转连续）是否正确工作。任务是在不改动算子的前提下，让 ST 在 Real 模式下真正构造物理非连续的 aclTensor，并确认算子输出仍与连续 golden 一致。

## 适用场景

- 工程形态：算子仓开发 C++ ST（Real 模式，NPU 执行 + CPU golden 比对）。
- API 路径：`aclCreateTensor`（shape/strides/offset/format/storage_dims 参数组）、`aclrtMalloc`/`aclrtMemcpy`（物理 buffer 上载）、算子 L2 层 `l0op::Contiguous`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（任何需要覆盖非连续输入路径的 aclnn 算子 ST）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：3D 输入的非连续构造（本实现仅支持 2D）；转置/切片型非连续（仅实现行主序 H 维 padding 一种形态）；Mock/CPU golden 侧的非连续索引计算（见设计决策）。

## 解决配方

思路：为 2D 输入的 H 维度加 padding 列——物理存储 shape 为 (rows, H+8)、逻辑 shape 为 (rows, H)、strides 为 (H+8, 1)，padding 列填 0；`aclCreateTensor` 用逻辑 shape + strides + 物理 storage_dims 描述。这样算子收到的 aclTensor 真实非连续，L2 层 `l0op::Contiguous` 必须正确处理才能得到与连续 golden 一致的输出。

## 完整代码

非连续 tensor 创建函数（tests/st/test_aclnn_<op>.cpp，完整实现）：

```cpp
// 创建非连续 aclTensor（仅在 H 维度后加 padding，模拟 ND 非连续存储）
// 物理 shape: (rows, H + PAD_COLS)，逻辑 shape: (rows, H)，strides: (H + PAD_COLS, 1)
// 仅支持 2D shape；padding 区域填 0
constexpr int64_t NON_CONTIG_PAD_COLS = 8;

static int CreateAclTensorNonContig(const std::vector<float>& hostData,
                                    const std::vector<int64_t>& shape,
                                    DType dtype,
                                    void** deviceAddr, aclTensor** tensor) {
    if (shape.size() != 2) return -1;
    int64_t rows = shape[0];
    int64_t H = shape[1];
    int64_t phys_H = H + NON_CONTIG_PAD_COLS;
    int64_t phys_total = rows * phys_H;

    size_t elem_size;
    aclDataType acl_dt;
    switch (dtype) {
        case DType::FP16: elem_size = sizeof(_Float16); acl_dt = ACL_FLOAT16; break;
        case DType::FP32: elem_size = sizeof(float);    acl_dt = ACL_FLOAT;   break;
        case DType::BF16: elem_size = sizeof(uint16_t); acl_dt = ACL_BF16;    break;
        default: return -1;
    }

    size_t phys_bytes = static_cast<size_t>(phys_total) * elem_size;
    std::vector<uint8_t> phys_buf(phys_bytes, 0);

    // 填充逻辑数据到物理 buffer 前 H 列（padding 区域保持 0）
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < H; j++) {
            float v = hostData[static_cast<size_t>(i * H + j)];
            size_t phys_offset = static_cast<size_t>(i * phys_H + j) * elem_size;
            if (dtype == DType::FP16) {
                _Float16 h = static_cast<_Float16>(v);
                std::memcpy(&phys_buf[phys_offset], &h, elem_size);
            } else if (dtype == DType::FP32) {
                std::memcpy(&phys_buf[phys_offset], &v, elem_size);
            } else {  // BF16
                uint16_t bits = FloatToBf16Bits(v);
                std::memcpy(&phys_buf[phys_offset], &bits, elem_size);
            }
        }
    }

    auto ret = aclrtMalloc(deviceAddr, phys_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, phys_bytes, phys_buf.data(), phys_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    // 逻辑 shape (rows, H)，strides (phys_H, 1)，storage_dims (rows, phys_H)
    std::vector<int64_t> strides = {phys_H, 1};
    std::vector<int64_t> storage_dims = {rows, phys_H};

    *tensor = aclCreateTensor(shape.data(), static_cast<int32_t>(shape.size()),
                              acl_dt, strides.data(), 0,
                              ACL_FORMAT_ND,
                              storage_dims.data(), static_cast<int32_t>(storage_dims.size()),
                              *deviceAddr);
    return ACL_SUCCESS;
}
```

调用点（Real 模式 RunTest 中按用例标志分支）：

```cpp
    // expanded_permuted_rows (必选, 按 dtype)
    // 非连续用例: 创建带 padding 的非连续 tensor（Real 模式专用）
    if (input.non_contiguous) {
        if (CreateAclTensorNonContig(input.expanded_permuted_rows, expanded_shape, tc.dtype,
                                     &dev_expanded, &t_expanded) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 expanded_permuted_rows (non_contiguous) tensor 失败");
            ok = false; goto cleanup;
        }
        LOG_PRINT("  expanded_permuted_rows: 非连续 tensor (pad_cols=%lld)",
                  static_cast<long long>(NON_CONTIG_PAD_COLS));
    } else {
        if (CreateAclTensorByDType(input.expanded_permuted_rows, expanded_shape, tc.dtype,
                                   &dev_expanded, &t_expanded) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 expanded_permuted_rows tensor 失败");
            ok = false; goto cleanup;
        }
    }
```

`aclCreateTensor` 参数语义要点：第 1/2 参数是逻辑 shape 与其维数；第 4 参数 strides 按元素计（本例 (H+8, 1)）；第 5 参数 offset=0；第 7/8 参数 storage_dims 描述物理存储 shape（本例 (rows, H+8)），设备指针按物理存储分配（`phys_total * elem_size` 字节）。

## 设计决策

- **只对 2D 必选输入做 H 维 padding**：非连续形态取最简单的行主序 padding（strides[0] > H），足以触发 L2 层 Contiguous 转换路径；转置/切片型非连续未做。
- **Mock 模式保持连续 golden，Real 模式真非连续**：曾考虑让 Mock 的 ComputeGolden 也支持 padding 索引（按物理 stride 取数），因破坏 golden 公式的简单性而放弃；最终分工为 Mock 验证公式正确性、Real 验证非连续处理路径，两者互补。
- **padding 列填 0**：便于在输出异常时区分"读到了 padding"与"读错行"（读到 0 会使输出偏向 0，特征明显）。
- 当前方案属 recommended_fastpath（覆盖最常见非连续形态且实现量最小）。

## 系统信息

- 目标芯片：Ascend950PR（arch35）；CANN 9.1.0；ST 构建入口 `cd tests/st && bash run.sh`（Real 模式）。
- 相关文件（与实际读写一一对应）：
  - `tests/st/test_aclnn_bench_no_harness_moe_finalize_routing.cpp`（CreateAclTensorNonContig 定义与 RunTest 调用分支）
  - `tests/st/testcases/L1_test_cases.csv`（non_contiguous 用例定义）

## 验证信号

本卡信号来自非连续构造实现后的两轮 Real 模式全量 ST（一轮全功能验证、一轮联调复验，Ascend950PR + CANN 9.1.0）。

- Real 模式日志出现 `expanded_permuted_rows: 非连续 tensor (pad_cols=8)` 且用例 PASS：fp16 非连续用例 MERE=1.85e-08、MARE=9.39e-04（threshold=9.77e-04，524288 elems）；fp32 非连续用例 MERE=1.56e-07、MARE=4.38e-04；bf16 非连续用例 MERE=0.00e+00、MARE=0.00e+00（来源平台：950；证据等级：tool_observed）。
- 非连续用例指标与对应连续路径用例一致（如 fp16 连续用例同为 MERE=1.85e-08/MARE=9.39e-04），说明 Contiguous 转换无精度损失（来源平台：950；证据等级：tool_observed）。
- 全量 57/57 通过（非连续用例与全部其他用例无冲突）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅支持 2D shape（函数对非 2D 直接返回 -1）；3D 输入（drop_pad 模式的 (E, C, H) 形态）的非连续构造未实现。
- 仅覆盖行主序 H 维 padding 一种非连续形态；转置型（stride 交换）、切片型（offset≠0）非连续未验证。
- 仅在 Ascend950PR + CANN 9.1.0 验证；`aclCreateTensor` strides/storage_dims 语义在其他 CANN 版本未验证（未验证，推断接口契约稳定）。

**未决项**：

- 3D 输入与转置型非连续的构造未实现（接续点：扩展 CreateAclTensorNonContig 支持 N 维 strides）。
- padding 列填 0 的"读到 padding 会偏 0"特征只用于人工判读，未做成自动断言（如校验输出不含 padding 值混入）。
## 坏实践

- 用例表标了 non_contiguous 但创建仍走连续函数——测试名与实际路径不符，联调会漏掉 Contiguous 路径的缺陷（本卡要解决的正是这个）。
- 只在 Mock/CPU 侧模拟非连续（改 golden 索引）而不在 Real 侧构造真非连续 tensor——测不到算子/框架的实际转换代码。

## 误诊

无（本卡为实现模式；实现后非连续用例首轮即通过）。
