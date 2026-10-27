---
type: API
title: asc_set_nthbit
description: 将uint64_t数据的指定二进制位置为1，其余位保持不变。返回修改后的uint64_t整数。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/scalar_compute/scalar_bit/asc_set_nthbit.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/scalar_compute/compute/scalar_bit.h
- include/c_api/asc_simd.h
---

# asc_set_nthbit

## 功能说明

头文件路径为：`"c_api/scalar_compute/compute/scalar_bit.h"`。

将`uint64_t`数据的指定二进制位置为1，其余位保持不变。返回修改后的`uint64_t`整数。

## 函数原型

```c
__aicore__ inline uint64_t asc_set_nthbit(uint64_t bits,
                                          int64_t idx)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| bits | 输入 | 待置位的64位无符号整数，取值范围为[0, 2^64-1]。执行此计算后输入数据不变。 |
| idx | 输入 | 位索引，指定需要设置为1的二进制位的位置。实际位索引按`(idx % 64 + 64) % 64`计算，索引范围为[0, 63]。 |

## 返回值说明

返回执行计算得到的结果，数据类型为`uint64_t`。

## 约束说明

当参数`idx > 63`或者`idx < 0`时，位索引的计算逻辑为`(idx % 64 + 64) % 64`。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {
template <typename T>
void print_data(const char* label, const std::vector<T>& values)
{
    std::cout << label << ":";
    const size_t count = values.size() < 8 ? values.size() : 8;
    for (size_t i = 0; i < count; ++i) std::cout << ' ' << +values[i];
    if (values.size() > count) std::cout << " ...";
    std::cout << std::endl;
}

template <typename T>
bool compare_data(const std::vector<T>& actual, const std::vector<T>& expected, double tolerance = 0.0)
{
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == expected[i]) continue;
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        if (diff > tolerance || diff < -tolerance) return false;
    }
    return true;
}

constexpr uint32_t ELEMENTS = 8;

__global__ __vector__ void asc_set_nthbit_kernel(__gm__ int64_t* output)
{
    asc_init();
    output[0] = static_cast<int64_t>(asc_set_nthbit(8, 1));
    asc_sync();
}
} // namespace

int main()
{
    std::vector<int64_t> input = {0};
    std::vector<int64_t> golden = {10};
    input.resize(ELEMENTS, 0);
    golden.resize(ELEMENTS, 0);
    std::vector<int64_t> output(ELEMENTS, -1);
    aclInit(nullptr);
    aclrtSetDevice(0);
    int64_t* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), (ELEMENTS) * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(output_device, input.size() * sizeof(int64_t), input.data(), input.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    asc_set_nthbit_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), output.size() * sizeof(int64_t), output_device, output.size() * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", input);
    print_data("Output", output);
    print_data("Golden", golden);
    const bool passed = compare_data(output, golden);
    std::cout << (passed ? "[Success] asc_set_nthbit passed." : "[Failed] asc_set_nthbit failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
