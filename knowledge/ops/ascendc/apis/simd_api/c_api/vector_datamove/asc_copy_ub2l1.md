---
type: API
title: asc_copy_ub2l1
description: 提供数据搬运功能，将数据从Unified Buffer (UB)搬运到L1 Buffer。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_datamove/asc_copy_ub2l1.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/vector_datamove/vector_datamove.h
- include/c_api/asc_simd.h
---

# asc_copy_ub2l1

## 功能说明

头文件路径为：`"c_api/vector_datamove/vector_datamove.h"`。

提供数据搬运功能，将数据从Unified Buffer（UB）搬运到L1 Buffer。

本接口支持以下两种数据搬运方式：

- 连续数据搬运

    将一段连续数据从UB搬运到L1 Buffer，如图1所示。

    **图1**  连续搬运示意图
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/c_api/figures/asc_copy_ub2l1_continuous.png "连续搬运示意图")

- 高维切分数据搬运

    将多个连续传输数据块从UB搬运到L1 Buffer。每个数据块的长度以及相邻数据块之间的间隔均以DataBlock（32字节）为单位指定，如图2所示。

    **图2**  高维切分搬运示意图
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/c_api/figures/asc_copy_ub2l1_highdim_split.png "高维切分搬运示意图")

本接口仅在AIV上生效。

## 函数原型

### 连续数据搬运

```c
__aicore__ inline void asc_copy_ub2l1(__cbuf__ void* dst,
                                      __ubuf__ void* src,
                                      uint32_t size)
```

### 高维切分数据搬运

```c
__aicore__ inline void asc_copy_ub2l1(__cbuf__ void* dst,
                                      __ubuf__ void* src,
                                      uint16_t burst_count,
                                      uint16_t burst_len,
                                      uint16_t src_gap,
                                      uint16_t dst_gap)
```

## 参数说明

### 连续数据搬运

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L1 Buffer的起始地址。需要32字节对齐。 |
| src | 输入 | 源UB的起始地址。需要32字节对齐。 |
| size | 输入 | 搬运数据大小，单位为字节。取值范围：[32, 4095×32]，且需为32的整数倍。 |

### 高维切分数据搬运

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L1 Buffer的起始地址。需要32字节对齐。 |
| src | 输入 | 源UB的起始地址。需要32字节对齐。 |
| burst_count | 输入 | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| burst_len | 输入 | 待搬运的每个连续传输数据块的长度，单位为DataBlock（32字节）。取值范围：[1, $2^{16}−1$]。 |
| src_gap | 输入 | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔），单位为DataBlock（32字节）。取值范围：[0, $2^{16}−1$]。 |
| dst_gap | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔），单位为DataBlock（32字节）。取值范围：[0, $2^{16}−1$]。 |

## 返回值说明

无

## 流水类型

PIPE_MTE3

## 约束说明

### 通用约束

- 本接口在非AIV上调用直接返回。
- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。
- 如果本指令与其他指令存在数据依赖，需要插入[同步指令](../sync/system_sync_overview.md)，保证多个指令串行化，防止出现异常数据。

### 连续搬运模式约束

- `size`需为32的整数倍，否则`size/32`截断后实际搬运字节数不足，多出非整32字节部分被丢弃。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


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

namespace {

constexpr uint32_t DIM = 16;
constexpr uint32_t ELEMENTS = DIM * DIM;
constexpr uint32_t INPUT_BYTES = ELEMENTS * sizeof(half);
constexpr uint64_t SYNC_ID = 8;

__global__ __mix__(1, 2) void asc_copy_ub2l1_kernel(
    __gm__ float* output, __gm__ uint16_t* a, __gm__ uint16_t* b)
{
    asc_init();
    __ubuf__ half a_ub[ELEMENTS], b_ub[ELEMENTS];
    __cbuf__ half a_l1[ELEMENTS], b_l1[ELEMENTS];
    __ca__ half a_l0[ELEMENTS];
    __cb__ half b_l0[ELEMENTS];
    __cc__ float c_l0[ELEMENTS];
    constexpr int64_t l1_ready = 2;
    if ASC_IS_AIV {
        asc_copy_gm2ub_align(a_ub, reinterpret_cast<__gm__ half*>(a), INPUT_BYTES);
        asc_copy_gm2ub_align(b_ub, reinterpret_cast<__gm__ half*>(b), INPUT_BYTES);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // Copy two contiguous Nz matrices from UB to the L1 buffers consumed by AIC.
        asc_copy_ub2l1(a_l1, a_ub, INPUT_BYTES);
        asc_copy_ub2l1(b_l1, b_ub, INPUT_BYTES);
        asc_sync_block_arrive(PIPE_MTE3, l1_ready);
        asc_sync_intra_arrive(PIPE_MTE3, SYNC_ID);
    }
    if ASC_IS_AIC {
        asc_sync_intra_wait(PIPE_MTE1, SYNC_ID);
        // AIC starts reading L1 only after both AIV MTE3 pipelines arrive.
        asc_sync_block_wait(PIPE_MTE1, l1_ready);
        asc_copy_l12l0a(a_l0, a_l1, 0, 0, 1, 1, 1, 1);
        asc_copy_l12l0b_transpose(b_l0, b_l1, 0, 0, 1, 1, 1, 1);
        asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
        asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
        asc_mmad(c_l0, a_l0, b_l0, DIM, DIM, DIM, asc_unit_flag_mode::DISABLE, false, false, true);
        asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2gm(output, c_l0, DIM, DIM, DIM, DIM, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
            asc_unit_flag_mode::DISABLE, QuantMode_t::NoQuant, asc_relu_pre_mode::NONE, false, true, false, false);
        asc_sync_pipe(PIPE_FIX);
    }
    asc_sync_pipe(PIPE_ALL);
}

void print_row(const char* name, const std::vector<float>& data)
{
    std::cout << name << ":";
    for (uint32_t i = 0; i < 8; ++i) std::cout << ' ' << data[i];
    std::cout << " ..." << std::endl;
}
} // namespace

int main()
{
    constexpr uint16_t half_bits[] = {0x3c00, 0x4000, 0x4200, 0x4400};
    std::vector<uint16_t> a(ELEMENTS), b(ELEMENTS, 0);
    std::vector<float> input(ELEMENTS), output(ELEMENTS, 0), golden(ELEMENTS);
    for (uint32_t row = 0; row < DIM; ++row) {
        for (uint32_t column = 0; column < DIM; ++column) {
            const uint32_t value = (row + column) % 4;
            a[row * DIM + column] = half_bits[value];
            input[row * DIM + column] = static_cast<float>(value + 1);
            golden[row * DIM + column] = input[row * DIM + column];
        }
        b[row * DIM + row] = half_bits[0];
    }
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint16_t *a_device = nullptr, *b_device = nullptr;
    float* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&a_device), INPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&b_device), INPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&output_device), output.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(a_device, INPUT_BYTES, a.data(), INPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(b_device, INPUT_BYTES, b.data(), INPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_copy_ub2l1_kernel<<<1, 0>>>(output_device, a_device, b_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), output.size() * sizeof(float), output_device, output.size() * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    print_row("Input A row 0", input);
    print_row("Output row 0", output);
    print_row("Golden row 0", golden);
    const bool passed = output == golden;
    std::cout << (passed ? "[Success] asc_copy_ub2l1 passed." : "[Failed] asc_copy_ub2l1 failed.") << std::endl;
    aclrtFree(a_device);
    aclrtFree(b_device);
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
