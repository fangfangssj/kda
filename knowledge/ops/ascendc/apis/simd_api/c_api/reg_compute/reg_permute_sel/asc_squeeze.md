---
type: API
title: asc_squeeze
description: 将src中被mask选择的有效元素作为返回值返回或依次复制到dst中。有效元素在计算结果中从低到高连续排列，剩余位置元素置为0。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/reg_compute/reg_permute_sel/asc_squeeze.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/compute/reg_permute_sel.h
- include/c_api/asc_simd.h
---

# asc_squeeze

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_permute_sel.h"`。

将`src`中被`mask`选择的有效元素作为返回值返回或依次复制到`dst`中。有效元素在计算结果中从低到高连续排列，剩余位置元素置为0。

本接口不会将有效数据大小保存至AR寄存器。如果需要筛选有效元素并将其连续搬出至Unified Buffer（UB），请参考[asc_squeeze_and_storeunalign](asc_squeeze_and_storeunalign.md)。

本接口为Reg矢量搬运接口，仅在AIV上生效。

## 函数原型

```c
// 通过函数返回值返回结果（占位符形式）
__simd_callee__ inline vector_<dtype> asc_squeeze(vector_<dtype> src,
                                                  vector_bool mask)

// 通过引用参数输出结果（占位符形式）
__simd_callee__ inline void asc_squeeze(vector_<dtype>& dst,
                                        vector_<dtype> src,
                                        vector_bool mask)
```

### dtype支持数据类型

`dtype`取值为：`int8_t`、`uint8_t`、`hifloat8_t`、`fp8_e8m0_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`。

### 函数原型典型示例

```c
// 示例：对half矢量数据寄存器执行按掩码压缩
__simd_callee__ inline vector_half asc_squeeze(vector_half src,
                                               vector_bool mask)

__simd_callee__ inline void asc_squeeze(vector_half& dst,
                                        vector_half src,
                                        vector_bool mask)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数（矢量数据寄存器），仅用于无返回值原型。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../../defs/type/data_type_definition.md)。

## 返回值说明

- 通过函数返回值返回结果的函数原型返回按掩码压缩后的结果，类型为矢量数据寄存器，与`src`的数据类型一致。
- 通过引用参数输出结果的函数原型无返回值。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../overview.md#通用约束)。
- `src`为矢量数据寄存器；无返回值原型中的`dst`为矢量数据寄存器。
- `mask`比特位为1的`src`元素按原顺序紧凑排列到计算结果低位；`mask`比特位为0的`src`元素不参与压缩，计算结果中压缩结果之后的剩余高位统一写0。

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

constexpr uint32_t ELEMENT_COUNT = 64;

__simd_vf__ inline void squeeze(__ubuf__ float* dst, __ubuf__ float* src)
{
    vector_float dst_reg;
    vector_float src_reg;
    uint32_t count = ELEMENT_COUNT;
    vector_bool mask = asc_update_mask_b32(count);
    asc_loadalign(src_reg, src);
    asc_squeeze(dst_reg, src_reg, mask);
    asc_storealign(dst, dst_reg, mask);
}

__global__ __vector__ void asc_squeeze_kernel(__gm__ float* dst, __gm__ float* src)
{
    asc_init();
    __ubuf__ float dst_local[ELEMENT_COUNT];
    __ubuf__ float src_local[ELEMENT_COUNT];
    asc_copy_gm2ub_align(src_local, src, ELEMENT_COUNT * sizeof(float));
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    squeeze(dst_local, src_local);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(dst, dst_local, ELEMENT_COUNT * sizeof(float));
    asc_sync();
}
} // namespace

int main()
{
    std::vector<float> src(ELEMENT_COUNT);
    std::vector<float> output(ELEMENT_COUNT, 0.0f);
    std::vector<float> golden(ELEMENT_COUNT);
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) src[i] = golden[i] = static_cast<float>(i) * 0.25f;
    aclInit(nullptr);
    aclrtSetDevice(0);
    float* src_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&src_device), (ELEMENT_COUNT) * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST);
    float* dst_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dst_device), (ELEMENT_COUNT) * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(src_device, src.size() * sizeof(float), src.data(), src.size() * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    asc_squeeze_kernel<<<1, 0>>>(dst_device, src_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), output.size() * sizeof(float), dst_device, output.size() * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", src);
    print_data("Output", output);
    print_data("Golden", golden);
    const bool passed = compare_data(output, golden);
    std::cout << (passed ? "[Success] asc_squeeze passed." : "[Failed] asc_squeeze failed.") << std::endl;
    aclrtFree(dst_device);
    aclrtFree(src_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```

- [AscSqueeze：固定 reference 调用约束](../../../../../guides/regbase/asc_squeeze_reference_contract.md)
- [比较与选择：固定 reference 调用约束](../../../../../guides/regbase/compare_and_select_reference_contract.md)
