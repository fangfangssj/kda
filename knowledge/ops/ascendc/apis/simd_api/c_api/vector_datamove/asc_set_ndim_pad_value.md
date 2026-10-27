---
type: API
title: asc_set_ndim_pad_value
description: 本接口用于设置asc_ndim_copy_gm2ub在常数填充模式下使用的填充值。当padding_mode设置为true时，后续调用asc_ndim_copy_gm2ub将使用该值填充搬运数据的边界区域。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_datamove/asc_set_ndim_pad_value.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/vector_datamove/vector_datamove.h
- include/c_api/asc_simd.h
---

# asc_set_ndim_pad_value

## 功能说明

头文件路径为：`"c_api/vector_datamove/vector_datamove.h"`。

本接口用于设置[asc_ndim_copy_gm2ub](asc_ndim_copy_gm2ub.md)在常数填充模式下使用的填充值。当`padding_mode`设置为`true`时，后续调用`asc_ndim_copy_gm2ub`将使用该值填充搬运数据的边界区域。

填充值的数据类型应为本接口支持的数据类型，并与`asc_ndim_copy_gm2ub`搬运的数据类型一致。

本接口仅在AIV上执行有效，在AIC上调用为空操作。

## 函数原型

```c
__aicore__ inline void asc_set_ndim_pad_value(<dtype> pad_value)
```

### dtype支持数据类型

`dtype`取值为：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`。

### 函数原型典型示例

```c
// 示例：int32_t类型填充值设置。
__aicore__ inline void asc_set_ndim_pad_value(int32_t pad_value)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| pad_value | 输入 | 常数填充值。当`padding_mode`设置为`true`时，后续调用`asc_ndim_copy_gm2ub`将使用该值填充搬运数据的边界区域。对于本接口直接支持的数据类型，`pad_value`的数据类型应与搬运数据类型一致。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 本接口仅在AIV上生效，非AIV调用直接返回。
- 本接口仅设置填充值，实际填充由后续调用的多维数据搬运接口在搬运过程中取用该填充值完成，须先调用本接口写入配置，再调用对应搬运接口。
- 常数填充模式下，还需配置各维左右两侧的填充元素数。`loop0`的左右填充元素数通过`asc_ndim_copy_gm2ub`的`loop0_lp_count`和`loop0_rp_count`参数配置，`loop1`至`loop4`的左右填充元素数通过[asc_set_ndim_pad_count](asc_set_ndim_pad_count.md)配置。
- 填充值配置会持续生效，直至重新配置。若后续搬运任务使用不同的填充值，应在调用`asc_ndim_copy_gm2ub`前调用本接口重新配置。
- 填充值仅在搬运接口选择常数填充模式时被取用；若选为最近值填充模式，本接口设置不生效。

## 调用示例

将以下代码保存为`example.asc`后，执行对应的编译运行命令。


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

constexpr uint32_t INPUT_BYTES = 6;
constexpr uint32_t OUTPUT_BYTES = 20;

__global__ __vector__ void asc_set_ndim_pad_value_kernel(__gm__ uint8_t* output, __gm__ uint8_t* input)
{
    asc_init();
    __ubuf__ uint8_t local[32];
    asc_set_ndim_loop0_stride(1, 1);
    asc_set_ndim_loop1_stride(5, 3);
    asc_set_ndim_loop2_stride(0, 0);
    asc_set_ndim_loop3_stride(0, 0);
    asc_set_ndim_loop4_stride(0, 0);
    asc_set_ndim_pad_value(static_cast<uint8_t>(42));
    asc_ndim_pad_count_config config;
    config.loop1_lp_count = 1;
    config.loop1_rp_count = 1;
    asc_set_ndim_pad_count(config);
    asc_ndim_copy_gm2ub(local, input, 3, 2, 1, 1, 1, 1, 1, true,
        asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(output, local, OUTPUT_BYTES);
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

void print_data(const char* name, const std::vector<uint8_t>& data)
{
    std::cout << name << ":";
    const uint32_t count = data.size() < 32 ? data.size() : 32;
    for (uint32_t i = 0; i < count; ++i) std::cout << ' ' << +data[i];
    if (data.size() > count) std::cout << " ...";
    std::cout << std::endl;
}
} // namespace

int main()
{
    std::vector<uint8_t> input(INPUT_BYTES), output(OUTPUT_BYTES, 0), golden(OUTPUT_BYTES, 0);
    for (uint32_t i = 0; i < INPUT_BYTES; ++i) input[i] = static_cast<uint8_t>(i + 1);
    for (uint32_t i = 0; i < OUTPUT_BYTES; ++i) golden[i] = 42;
    for (uint32_t row = 0; row < 2; ++row) {
        for (uint32_t column = 0; column < 3; ++column) golden[(row + 1) * 5 + column + 1] = input[row * 3 + column];
    }
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint8_t *input_device = nullptr, *output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&input_device), INPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&output_device), OUTPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(input_device, INPUT_BYTES, input.data(), INPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_device, OUTPUT_BYTES, output.data(), OUTPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_set_ndim_pad_value_kernel<<<1, 0>>>(output_device, input_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), OUTPUT_BYTES, output_device, OUTPUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", input);
    print_data("Output", output);
    print_data("Golden", golden);
    const bool passed = output == golden;
    std::cout << (passed ? "[Success] asc_set_ndim_pad_value passed." : "[Failed] asc_set_ndim_pad_value failed.") << std::endl;
    aclrtFree(input_device);
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
