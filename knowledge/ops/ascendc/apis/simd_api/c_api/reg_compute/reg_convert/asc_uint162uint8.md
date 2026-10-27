---
type: API
title: asc_uint162uint8
description: 将uint16_t类型数据转换为uint8_t类型，无舍入模式，支持饱和/非饱和模式。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/reg_compute/reg_convert/asc_uint162uint8.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/reg_convert.h
- include/c_api/asc_simd.h
---

# asc_uint162uint8

## 功能说明

头文件路径为：`"c_api/reg_compute/reg_convert.h"`。

将`uint16_t`类型数据转换为`uint8_t`类型，无舍入模式，支持饱和/非饱和模式。

关于饱和/非饱和模式的详细说明，请参见[舍入模式与饱和模式](rounding_mode.md)。

由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个`VL`大小的数据分为两部分，根据参数`dst_pos`选择写入`dst`索引为偶数的位置或奇数的位置。

```python
# 位置判断仅用于说明接口语义，实际位置参数为编译期常量。
# VL表示矢量数据寄存器位宽，取值256字节。
def asc_uint162uint8(dst, src, mask, dst_pos):
    offset = 0 if dst_pos == ASC_POSITION_EVEN else 1
    for i in range(128):
        dst[2 * i] = 0
        dst[2 * i + 1] = 0
        if mask[i]:
            dst[2 * i + offset] = uint8_t(src[i])
```

本接口为Reg矢量计算接口，仅在AIV上生效。

## 函数原型

```c
// 通过引用参数输出结果（占位符形式）
__simd_callee__ inline void asc_uint162uint8<sat_mode>(vector_uint8_t& dst,
                                                       vector_uint16_t src,
                                                       vector_bool mask,
                                                       std::integral_constant<asc_position_mode, asc_position_mode::EVEN> dst_pos)

__simd_callee__ inline void asc_uint162uint8<sat_mode>(vector_uint8_t& dst,
                                                       vector_uint16_t src,
                                                       vector_bool mask,
                                                       std::integral_constant<asc_position_mode, asc_position_mode::ODD> dst_pos)

// 通过函数返回值返回结果（占位符形式）
__simd_callee__ inline vector_uint8_t asc_uint162uint8<sat_mode>(vector_uint16_t src,
                                                                 vector_bool mask,
                                                                 std::integral_constant<asc_position_mode, asc_position_mode::EVEN> dst_pos)

__simd_callee__ inline vector_uint8_t asc_uint162uint8<sat_mode>(vector_uint16_t src,
                                                                 vector_bool mask,
                                                                 std::integral_constant<asc_position_mode, asc_position_mode::ODD> dst_pos)
```

**占位符说明如下：**

- `<sat_mode>`表示饱和/非饱和模式，为空时表示非饱和模式，取值为`_sat`时表示饱和模式。

### 函数原型典型示例

```c
// 示例：偶数分块位置、非饱和模式。
__simd_callee__ inline void asc_uint162uint8(vector_uint8_t& dst,
                                             vector_uint16_t src,
                                             vector_bool mask,
                                             std::integral_constant<asc_position_mode, asc_position_mode::EVEN> dst_pos)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。|
| src | 输入 | 源操作数（矢量数据寄存器）。|
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |
| dst_pos | 输入 | 位置选择标签（编译期标签分发，通过编译期重载选择对应实现），取值如下：<br>&bull; `ASC_POSITION_EVEN`：选择将结果写入目的操作数索引为偶数的位置，其他位置清零<br>&bull; `ASC_POSITION_ODD`：选择将结果写入目的操作数索引为奇数的位置，其他位置清零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../../defs/type/data_type_definition.md)。

## 返回值说明

- 通过引用参数输出结果的函数原型无返回值。
- 通过函数返回值输出结果的函数原型返回计算结果，返回值类型与对应引用输出函数原型中`dst`参数的类型一致（去除引用）。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../overview.md#通用约束)。
- 位置选择标签参数仅能使用编译期常量，编译器据此在编译期分发至对应的重载。
- 位置选择标签选择目的操作数的写入地址，其他位置清零。
- `src`与`dst`的数据类型需要与函数原型匹配。
- `mask`掩码位为0时，`dst`对应元素置0。
- 结果写入`dst`的奇数索引位置时，偶数索引位置置零；结果写入偶数索引位置时，奇数索引位置置零。
- 使用饱和或非饱和模式时，需要配置`ctrl`寄存器，详细说明请参见[asc_set_ctrl](../../spr/asc_set_ctrl.md)。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[__NPU_ARCH__](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```c
#include <cstdint>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "acl/acl.h"

#define CHECK_ACL(call)                                                        \
    do {                                                                       \
        const aclError status = (call);                                        \
        if (status != ACL_SUCCESS) {                                           \
            std::cerr << #call << " failed: " << status << std::endl;            \
            return 1;                                                          \
        }                                                                      \
    } while (false)

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

constexpr uint32_t BUFFER_BYTES = 256;
constexpr uint32_t OUTPUT_BYTES = 2 * BUFFER_BYTES;
__simd_vf__ inline void convert(__ubuf__ uint8_t* output, __ubuf__ uint8_t* input)
{
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bool store_mask = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t dst_0;
    vector_uint16_t src_0;
    asc_loadalign(src_0, reinterpret_cast<__ubuf__ uint16_t*>(input));
    asc_uint162uint8_sat(dst_0, src_0, mask, ASC_POSITION_EVEN);
    asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(output + BUFFER_BYTES), dst_0, store_mask);
    vector_uint8_t dst_1;
    vector_uint16_t src_1;
    asc_loadalign(src_1, reinterpret_cast<__ubuf__ uint16_t*>(input));
    asc_uint162uint8(dst_1, src_1, mask, ASC_POSITION_EVEN);
    asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(output), dst_1, store_mask);
}
__global__ __vector__ void asc_uint162uint8_kernel(__gm__ uint8_t* output, __gm__ uint8_t* input)
{
    asc_init();
    const asc_override_strategy old_strategy = asc_get_saturation_strategy();
    asc_set_saturation_strategy(asc_override_strategy::USE_API);
    __ubuf__ uint8_t output_local[OUTPUT_BYTES];
    __ubuf__ uint8_t input_local[BUFFER_BYTES];
    asc_copy_gm2ub_align(input_local, input, BUFFER_BYTES);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    convert(output_local, input_local);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(output, output_local, OUTPUT_BYTES);
    asc_sync();
    asc_set_saturation_strategy(old_strategy);
}
} // namespace
int main()
{
    constexpr uint8_t input_pattern[] = {0x05, 0x01};
    constexpr uint8_t golden_pattern[] = {0x05, 0x00};
    constexpr uint8_t saturated_pattern[] = {0xff, 0x00};
    std::vector<uint8_t> input(BUFFER_BYTES);
    std::vector<uint8_t> output(OUTPUT_BYTES, 0xff);
    std::vector<uint8_t> golden(OUTPUT_BYTES);
    for (size_t i = 0; i < BUFFER_BYTES; ++i) {
        input[i] = input_pattern[i % sizeof(input_pattern)];
        golden[i] = golden_pattern[i % sizeof(golden_pattern)];
        golden[BUFFER_BYTES + i] = saturated_pattern[i % sizeof(saturated_pattern)];
    }
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    uint8_t* input_device = nullptr;
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&input_device), BUFFER_BYTES * sizeof(uint8_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* output_device = nullptr;
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&output_device), OUTPUT_BYTES * sizeof(uint8_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(input_device, input.size() * sizeof(uint8_t), input.data(), input.size() * sizeof(uint8_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
    asc_uint162uint8_kernel<<<1, 0>>>(output_device, input_device);
    CHECK_ACL(aclrtSynchronizeDevice());
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(uint8_t), output_device, output.size() * sizeof(uint8_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    std::cout << "Conversion: vector_uint16_t -> vector_uint8_t" << std::endl;
    print_data("Input (raw bytes)", input);
    const std::vector<uint8_t> truncated(output.begin(), output.begin() + BUFFER_BYTES);
    const std::vector<uint8_t> saturated(output.begin() + BUFFER_BYTES, output.end());
    const std::vector<uint8_t> truncated_golden(golden.begin(), golden.begin() + BUFFER_BYTES);
    const std::vector<uint8_t> saturated_golden(golden.begin() + BUFFER_BYTES, golden.end());
    print_data("Non-saturated output (raw bytes)", truncated);
    print_data("Non-saturated golden (raw bytes)", truncated_golden);
    print_data("Saturated output (raw bytes)", saturated);
    print_data("Saturated golden (raw bytes)", saturated_golden);
    const bool truncated_passed = compare_data(truncated, truncated_golden);
    const bool saturated_passed = compare_data(saturated, saturated_golden);
    std::cout << "Non-saturated: " << (truncated_passed ? "passed" : "failed") << std::endl;
    std::cout << "Saturated: " << (saturated_passed ? "passed" : "failed") << std::endl;
    const bool passed = truncated_passed && saturated_passed;
    std::cout << (passed ? "[Success] asc_uint162uint8/asc_uint162uint8_sat/ASC_POSITION_EVEN passed."
                          : "[Failed] asc_uint162uint8/asc_uint162uint8_sat/ASC_POSITION_EVEN failed.")
              << std::endl;
    CHECK_ACL(aclrtFree(input_device));
    CHECK_ACL(aclrtFree(output_device));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return passed ? 0 : 1;
}
```
