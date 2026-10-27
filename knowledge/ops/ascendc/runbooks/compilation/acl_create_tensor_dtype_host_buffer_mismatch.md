---
type: "Runbook"
title: aclCreateTensor声明dtype与host数据不符—静默错值
description: "aclnn 示例用 std::vector<float> 主机数据但 aclCreateTensor 声明 ACL_FLOAT16：模板按 sizeof(float)=4 分配/搬运，设备侧按 fp16 位模式重解释，输出量级完全错误的\"合法值\"且进程 PASS 退出、无任何报错；dataType 声明、host 数据类型、字节数三者必须逐字节一致，并以手算锚点值验证示例输出。"
tags: [task.testing, topic.implementation.api_layer.c_api, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T01:58:00Z'
updated_at: '2026-09-17T01:58:00Z'
platforms: ["950"]
---

# aclCreateTensor声明dtype与host数据不符—静默错值

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子 BenchNoHarnessConv2d 编写 aclnn（eager）调用示例：输入 x[1,4,8,8]、filter[4,4,3,3]、bias[4] 全部初始化为 1.0f/0.5f 的 `std::vector<float>` 主机数据，经 `CreateAclTensor` 模板函数 `aclrtMalloc` + `aclrtMemcpy` 上设备并 `aclCreateTensor`。示例编译成功、运行"成功"（退出码 0、打印 `执行结果: PASS`），但输出值是量级完全错误的数：

```
output shape: [1, 4, 8, 8]
result[0] is: 8932104.000000
result[1] is: 8933128.000000
result[4] is: 207934560.000000
```

正确结果应为 corner≈16.5、edge≈24.5（x/filter 全 1.0、bias 0.5、K=3×3、C_in=4）。整个流程没有任何报错或警告——dtype 声明与数据布局不符在 aclnn 链路上是**静默**的。当时的示例代码把 aclCreateTensor 的 dataType 参数写成了 `ACL_FLOAT16`，而模板 `T=float`。

## 适用场景

- 工程形态：aclnn 调用示例 / 自研 C++ 测试脚手架（host 数据 → aclrtMemcpy → aclCreateTensor → 两段式调用 → 拷回比对）。
- API 路径：`aclCreateTensor(shape, dims, dataType, strides, offset, format, ...)` 与 `aclrtMalloc/aclrtMemcpy` 的字节数语义（`GetShapeSize(shape) * sizeof(T)`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——aclTensor 的 dataType 声明与设备缓冲区字节布局一致性语义与平台无关；本会话在 950 上验证。
- 路线优先级：recommended_fastpath（示例/脚手架统一用与 host 数据一致的 dtype 声明 + 手算锚点值验证）。
- 明确不适用 / 未验证场景：框架（torch_npu/GE）托管 dtype 的路径不涉及手工声明；int32/int64 整型数据同类错配未单独实验（机制相同）。

## 根因

`aclCreateTensor` 的 `dataType` 参数只描述元数据，**不搬运也不校验**设备缓冲区内容；缓冲区字节数由调用方决定。示例的模板函数按 `sizeof(T)`（float=4 字节）分配并拷贝了 `numel*4` 字节，但把张量声明为 `ACL_FLOAT16`（2 字节/元素）——kernel 按 fp16 读该缓冲区时，把 fp32 位模式两两截段重解释，得到随机量级的"合法 fp16 值"；输出侧同理。全程无错误码，因为对 ACL 而言每个 API 调用都自洽成功。

机制链（以本例）：x 元素 1.0f = 0x3F800000；相邻两 fp32 组成的 8 字节被读作 4 个 fp16（0x0000、0x3F80、0x0000、0x3F80...），参与卷积后输出落到 1e6~1e8 量级——正是观测到的 8932104.0/207934560.0。

## 解决配方

让 dataType 声明、host 数据类型、字节数三者一致。本例最简修复是把示例统一为 float32（与 `std::vector<float>` 数据匹配）：

```cpp
    // x: [1, 4, 8, 8] float32
    std::vector<int64_t> xShape = {1, 4, 8, 8};
    int64_t xSize = GetShapeSize(xShape);
    std::vector<float> xHostData(xSize, 1.0f);
    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // filter: [4, 4, 3, 3] float32
    std::vector<int64_t> filterShape = {4, 4, 3, 3};
    int64_t filterSize = GetShapeSize(filterShape);
    std::vector<float> filterHostData(filterSize, 1.0f);
    aclTensor* filter = nullptr;
    void* filterDeviceAddr = nullptr;
    ret = CreateAclTensor(filterHostData, filterShape, &filterDeviceAddr, aclDataType::ACL_FLOAT, &filter);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // bias: [4] float32
    std::vector<int64_t> biasShape = {4};
    std::vector<float> biasHostData = {0.5f, 0.5f, 0.5f, 0.5f};
    aclTensor* bias = nullptr;
    void* biasDeviceAddr = nullptr;
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // y: [1, 4, 8, 8] float32
    // H_out = (8 + 1 + 1 - 1*(3-1) - 1) / 1 + 1 = 8, W_out = 8
    std::vector<int64_t> yShape = {1, 4, 8, 8};
    int64_t ySize = GetShapeSize(yShape);
    std::vector<float> yHostData(ySize, 0.0f);
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
```

若必须用 fp16 示例，则 host 侧先显式转 half（如 `Float2Half`）再按 `numel*2` 字节搬运，dataType 声明 ACL_FLOAT16——两种方式择一，关键是逐字节一致。

模板函数本身的字节数语义（修复前后未变，供对照）：

```cpp
template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}
```

## 设计决策

- 为何选"示例统一 float32"而不是"数据转 half 保 ACL_FLOAT16"：示例首要目标是演示调用流程且结果可人工验证；float 路径与 add_example 模板（float 数据 + ACL_FLOAT）一致，最少惊讶。真 fp16 数据路径属精度测试职责，不在示例。
- 保留 `sizeof(T)` 模板字节数设计：只要 dataType 与 T 匹配该设计自洽；错配风险靠"dtype 声明必须与 host 数据类型一致"的约定消除。

## 验证信号

以下信号来自该算子 aclnn 示例的输出验证闭环（Ascend950PR + CANN 9.1.0）。

- 修复前：示例输出 result[0]=8932104.0、result[4]=207934560.0 等量级错误值，无任何报错、退出 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后（全部改 `ACL_FLOAT`）：输出 corner=16.5、edge=24.5，与手算一致（corner：3×3 核在 (0,0) 处覆盖 4 个有效像素 × C_in=4 × 1.0 + bias 0.5 = 16.5；edge：6 个有效像素 × 4 × 1.0 + 0.5 = 24.5）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对"调用方自管 host 数据 + aclCreateTensor"路径；输出错值还有其它根因（tiling 缓存、kernel bug 等），先用"手算锚点值"确认是否 dtype/布局层问题再深入。
- 静默性意味着 CI 里只看退出码/无异常的示例验证发现不了此类错配——必须有值断言。
- bf16/fp16 host 侧转换的舍入模式（RN/RTZ）与设备端一致性未在本例验证。
- 仅 Ascend950PR + CANN 9.1.0 实测；机制属 ACL 元数据与缓冲区语义，其他平台未验证，推断可迁移。

**未决项**：

- 无。
## 坏实践

- 示例只打印结果不设期望值（"调用流程通了就行"）——本例流程"全通"但输出全错。
- 把 `sizeof(T)` 字节数当作与 dataType 参数无关的实现细节随手传。

## 误诊

- 无（本现象一次定位到 dtype 错配；输出量级 1e6~1e8 与位模式重解释机制吻合）。
