---
type: API
title: asc_get_mask_spr
description: 从SPR（Special Purpose Register）中读取mask值，并按数据类型格式化后返回。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/reg_compute/reg_mask/asc_get_mask_spr.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/compute/reg_mask.h
- include/c_api/asc_simd.h
---

# asc_get_mask_spr

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_mask.h"`。

从SPR（Special Purpose Register）中读取mask值，并按数据类型格式化后返回。

- 对于b16数据类型：从SPR读取mask共128bit数据，将每个bit扩展为2bit，拼成256bit掩码寄存器返回。
- 对于b32数据类型：从SPR读取mask共64bit数据，将每个bit扩展为4bit，拼成256bit掩码寄存器返回。

本接口为Reg矢量计算接口，仅在AIV上生效。

## 函数原型

```c
__simd_callee__ inline vector_bool asc_get_mask_spr_b16()
__simd_callee__ inline vector_bool asc_get_mask_spr_b32()
```

## 参数说明

无

## 返回值说明

vector_bool，掩码寄存器。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../overview.md#通用约束)。
- 本接口需在Vector Function（`__simd_vf__`标记的函数）内调用。
- [asc_set_vector_mask](../../vector_compute/vector_mask_config/asc_set_vector_mask.md)用于在VF场景外设置掩码寄存器，在Reg矢量计算配置时，通过`asc_get_mask_spr_b16`或`asc_get_mask_spr_b32`配合获取掩码寄存器的值。更推荐用户使用VF场景内的接口（[asc_create_mask](asc_create_mask.md)和[asc_update_mask.md](asc_update_mask.md)）完成Reg矢量计算掩码配置。

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

template <typename T>
bool compare_range_data(const std::vector<T>& actual, const std::vector<T>& expected,
    size_t begin, size_t count, double tolerance = 0.0)
{
    if (begin + count > actual.size() || begin + count > expected.size()) return false;
    for (size_t i = begin; i < begin + count; ++i) {
        if (actual[i] == expected[i]) continue;
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        if (diff > tolerance || diff < -tolerance) return false;
    }
    return true;
}

constexpr uint16_t ONE_REPEAT_ELEMENTS = 256 / sizeof(uint16_t);
constexpr uint16_t REPEAT_TIMES = 2;
constexpr uint32_t BUFFER_ELEMENTS = ONE_REPEAT_ELEMENTS * REPEAT_TIMES;
constexpr uint32_t BUFFER_BYTES = BUFFER_ELEMENTS * sizeof(uint16_t);

__simd_vf__ inline void get_mask_spr_vf(__ubuf__ uint16_t* output,
                                uint16_t one_repeat_size, uint16_t repeat_time)
{
    vector_bool mask16 = asc_get_mask_spr_b16();
    vector_uint16_t dst_reg;
    for (uint16_t repeat_index = 0; repeat_index < repeat_time; ++repeat_index) {
        asc_duplicate_scalar(dst_reg, static_cast<uint16_t>(1), mask16);
        asc_storealign(output + repeat_index * one_repeat_size, dst_reg, mask16);
    }
}

__global__ __vector__ void asc_get_mask_spr_kernel(__gm__ uint16_t* output, __gm__ uint16_t* input)
{
    asc_init();
    __ubuf__ uint16_t output_local[BUFFER_ELEMENTS];
    asc_copy_gm2ub_align(output_local, input, BUFFER_BYTES);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_set_vector_mask(0, 0xffffffffULL);
    asc_vf_call<get_mask_spr_vf>(output_local, ONE_REPEAT_ELEMENTS, REPEAT_TIMES);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(output, output_local, BUFFER_BYTES);
    asc_sync();
}
} // namespace

int main()
{
    std::vector<uint16_t> input(BUFFER_ELEMENTS), output(BUFFER_ELEMENTS, 0xffff), golden(BUFFER_ELEMENTS);
    for (uint32_t index = 0; index < BUFFER_ELEMENTS; ++index) {
        input[index] = static_cast<uint16_t>(index + 2);
    }
    golden = input;
    for (uint16_t repeat_index = 0; repeat_index < REPEAT_TIMES; ++repeat_index) {
        for (uint32_t index = 0; index < 32; ++index) {
            golden[repeat_index * ONE_REPEAT_ELEMENTS + index] = 1;
        }
    }
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint16_t* input_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&input_device), BUFFER_BYTES,
        ACL_MEM_MALLOC_HUGE_FIRST);
    uint16_t* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), BUFFER_BYTES,
        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(input_device, BUFFER_BYTES, input.data(), BUFFER_BYTES,
        ACL_MEMCPY_HOST_TO_DEVICE);
    asc_get_mask_spr_kernel<<<1, 0>>>(output_device, input_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BUFFER_BYTES, output_device, BUFFER_BYTES,
        ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input elements", input);
    print_data("Output elements", output);
    print_data("Golden elements", golden);
    const bool passed = compare_data(output, golden);
    std::cout << (passed ? "[Success] asc_get_mask_spr_b16 passed." : "[Failed] asc_get_mask_spr_b16 failed.") << std::endl;
    aclrtFree(input_device);
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
