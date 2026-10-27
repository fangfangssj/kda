---
type: API
title: asc_set_va_reg
description: 用于设置transpose的16个DataBlock地址，将操作数地址序列与地址寄存器关联，接口要求前8个和后8个地址序列与地址寄存器分别关联。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_permute_sel/asc_set_va_reg.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_permute_sel.h
- include/c_api/asc_simd.h
---

# asc_set_va_reg

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_permute_sel.h"`。

用于设置转置的16个[DataBlock](../../general_description_and_constraints.md)地址，将操作数地址序列与地址寄存器关联，接口要求前8个和后8个地址序列与地址寄存器分别关联。

## 函数原型

```cpp
// 占位符形式
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr,
                                      __ubuf__ <dtype>** src_array)
```

### dtype支持数据类型

`dtype`取值为：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`half`、`int32_t`、`uint32_t`、`float`。

### 函数原型典型示例

```c
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr,
                                      __ubuf__ int8_t** src_array)
```

## 参数说明

**表1** 参数说明

|  参数名  |  输入/输出  |  描述  |
| ------------ | ------------ | ------------ |
| addr | 输入 | 地址寄存器，类型为ub_addr8_t，可取值为：<br>&bull; VA0<br>&bull; VA1<br>&bull; VA2<br>&bull; VA3<br>&bull; VA4<br>&bull; VA5<br>&bull; VA6<br>&bull; VA7<br>数字代表寄存器顺序，每个地址寄存器只能关联8个地址，使用方法请参考调用示例|
| src_array | 输入 | 操作数地址序列。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址对齐约束请参考[存储单元说明](../../general_description_and_constraints.md#存储单元说明)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_description_and_constraints.md#通用地址重叠约束)。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```cpp
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {

constexpr uint32_t MATRIX_SIDE = 16;
constexpr uint32_t MATRIX_ELEMENTS = MATRIX_SIDE * MATRIX_SIDE;
constexpr uint32_t TOTAL_ELEMENTS = 2 * 32 * 16 * 16;
constexpr uint32_t OFFSET = 16 * 16 * 16;

__global__ __vector__ void transto5hd_kernel(__gm__ uint16_t* matrix_out,
                                             __gm__ uint16_t* matrix_in,
                                             __gm__ uint16_t* layout_out,
                                             __gm__ uint16_t* layout_in)
{
    asc_init();
    __ubuf__ uint16_t src[TOTAL_ELEMENTS];
    __ubuf__ uint16_t dst[TOTAL_ELEMENTS];
    __ubuf__ half* src_list[16];
    __ubuf__ half* dst_list[16];

    // 场景一：一次repeat完成16x16矩阵转置，stride为0时从地址列表起点读取。
    asc_copy_gm2ub_align(src, matrix_in, MATRIX_ELEMENTS * sizeof(uint16_t));
    asc_sync_mte2(0);
    for (int i = 0; i < 16; ++i) {
        src_list[i] = reinterpret_cast<__ubuf__ half*>(src + i * 16);
        dst_list[i] = reinterpret_cast<__ubuf__ half*>(dst + i * 16);
    }
    asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
    asc_set_va_reg(VA0, dst_list);
    asc_set_va_reg(VA1, dst_list + 8);
    asc_set_va_reg(VA2, src_list);
    asc_set_va_reg(VA3, src_list + 8);
    asc_transto5hd_b16(VA0, VA2, 1, 0, 0);
    asc_sync_vec(0);
    asc_copy_ub2gm_align(matrix_out, dst, MATRIX_ELEMENTS * sizeof(uint16_t));
    asc_sync_mte3(0);

    // 场景二：N=2、C=32、H=W=16，四组C0=16通道分别处理。
    asc_copy_gm2ub_align(src, layout_in, TOTAL_ELEMENTS * sizeof(uint16_t));
    asc_sync_mte2(0);
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 16; ++i) {
            src_list[i] = reinterpret_cast<__ubuf__ half*>(src + OFFSET * j + i * 16 * 16);
            dst_list[i] = reinterpret_cast<__ubuf__ half*>(dst + OFFSET * j + i * 16);
        }
        asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
        asc_set_va_reg(VA0, dst_list);
        asc_set_va_reg(VA1, dst_list + 8);
        asc_set_va_reg(VA2, src_list);
        asc_set_va_reg(VA3, src_list + 8);
        // 16次repeat沿HW方向推进，src_stride=1、dst_stride=16，单位均为DataBlock。
        asc_transto5hd_b16(VA0, VA2, 16, 16, 1);
    }
    asc_sync_vec(0);
    asc_copy_ub2gm_align(layout_out, dst, TOTAL_ELEMENTS * sizeof(uint16_t));
    asc_sync_mte3(0);
}

void check(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        std::cerr << operation << " failed: " << status << std::endl;
        std::exit(1);
    }
}

} // namespace

int main()
{
    std::vector<uint16_t> matrix_in(MATRIX_ELEMENTS), matrix_out(MATRIX_ELEMENTS);
    std::vector<uint16_t> layout_in(TOTAL_ELEMENTS), layout_out(TOTAL_ELEMENTS);
    for (uint32_t i = 0; i < MATRIX_ELEMENTS; ++i) matrix_in[i] = i;
    for (uint32_t i = 0; i < TOTAL_ELEMENTS; ++i) layout_in[i] = i;

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "aclrtSetDevice");
    uint16_t *matrix_in_device = nullptr, *matrix_out_device = nullptr;
    uint16_t *layout_in_device = nullptr, *layout_out_device = nullptr;
    check(aclrtMalloc(reinterpret_cast<void**>(&matrix_in_device), MATRIX_ELEMENTS * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc matrix_in");
    check(aclrtMalloc(reinterpret_cast<void**>(&matrix_out_device), MATRIX_ELEMENTS * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc matrix_out");
    check(aclrtMalloc(reinterpret_cast<void**>(&layout_in_device), TOTAL_ELEMENTS * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc layout_in");
    check(aclrtMalloc(reinterpret_cast<void**>(&layout_out_device), TOTAL_ELEMENTS * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc layout_out");
    check(aclrtMemcpy(matrix_in_device, MATRIX_ELEMENTS * sizeof(uint16_t), matrix_in.data(), MATRIX_ELEMENTS * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy matrix_in");
    check(aclrtMemcpy(layout_in_device, TOTAL_ELEMENTS * sizeof(uint16_t), layout_in.data(), TOTAL_ELEMENTS * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy layout_in");
    transto5hd_kernel<<<1, 0>>>(matrix_out_device, matrix_in_device, layout_out_device, layout_in_device);
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
    check(aclrtMemcpy(matrix_out.data(), MATRIX_ELEMENTS * sizeof(uint16_t), matrix_out_device, MATRIX_ELEMENTS * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy matrix_out");
    check(aclrtMemcpy(layout_out.data(), TOTAL_ELEMENTS * sizeof(uint16_t), layout_out_device, TOTAL_ELEMENTS * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy layout_out");

    bool matrix_passed = true;
    for (uint32_t row = 0; row < 16; ++row) {
        for (uint32_t col = 0; col < 16; ++col) {
            matrix_passed &= matrix_out[col * 16 + row] == matrix_in[row * 16 + col];
        }
    }
    bool layout_passed = true;
    for (uint32_t n = 0; n < 2; ++n) {
        for (uint32_t c = 0; c < 32; ++c) {
            for (uint32_t hw = 0; hw < 256; ++hw) {
                uint32_t src_index = (n * 32 + c) * 256 + hw;
                uint32_t dst_index = ((n * 2 + c / 16) * 256 + hw) * 16 + c % 16;
                layout_passed &= layout_out[dst_index] == layout_in[src_index];
            }
        }
    }
    std::cout << "16x16 transpose: " << (matrix_passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "NCHW to NC1HWC0: " << (layout_passed ? "PASS" : "FAIL") << std::endl;
    check(aclrtFree(matrix_in_device), "free matrix_in");
    check(aclrtFree(matrix_out_device), "free matrix_out");
    check(aclrtFree(layout_in_device), "free layout_in");
    check(aclrtFree(layout_out_device), "free layout_out");
    check(aclrtResetDevice(0), "aclrtResetDevice");
    check(aclFinalize(), "aclFinalize");
    return matrix_passed && layout_passed ? 0 : 1;
}
```
