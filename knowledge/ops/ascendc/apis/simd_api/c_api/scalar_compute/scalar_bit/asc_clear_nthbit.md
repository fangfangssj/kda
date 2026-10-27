---
type: API
title: asc_clear_nthbit
description: 对一个64位无符号整数执行位清除操作，将bits中第idx位清零，其余位保持不变。返回修改后的64位无符号整数。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/scalar_compute/scalar_bit/asc_clear_nthbit.md
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

# asc_clear_nthbit

## 功能说明

头文件路径为：`"c_api/scalar_compute/compute/scalar_bit.h"`。

对一个64位无符号整数执行位清除操作，将`bits`中第`idx`位清零，其余位保持不变。返回修改后的64位无符号整数。

本接口为标量位运算，运行在标量流水上，AIC与AIV均可调用，行为一致。

## 函数原型

```c
__aicore__ inline uint64_t asc_clear_nthbit(uint64_t bits,
                                            int64_t idx)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| `bits` | 输入 | 64位无符号整数。取值范围为[0，2^64−1]。 |
| `idx` | 输入 | 位索引。实际位索引按`(idx % 64 + 64) % 64`计算，范围为[0，63]。 |

## 返回值说明

返回位清零后的结果（uint64_t类型）。

## 流水类型

PIPE_S

## 约束说明

当`idx`大于63或小于0时，实际清除的位索引按`(idx % 64 + 64) % 64`计算。

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
template <typename T>
void print_data(const char* label, const std::vector<T>& values)
{
    std::cout << label << ":";
    const size_t count = values.size() < 8 ? values.size() : 8;
    for (size_t i = 0; i < count; ++i) std::cout << ' ' << +values[i];
    if (values.size() > count) std::cout << " ...";
    std::cout << std::endl;
}

constexpr uint32_t ELEMENTS = 8;

__global__ __vector__ void asc_clear_nthbit_kernel(__gm__ uint64_t* output)
{
    asc_init();
    // 确保普通Scalar读取到GM中的最新输入。
    asc_dcci_single(output);
    asc_sync_data_barrier(mem_dsb_t::DSB_ALL);
    output[0] = asc_clear_nthbit(output[0], 1);
    // 将普通Scalar写产生的Dirty Cache Line写回GM。
    asc_dcci_single(output);
    asc_sync_data_barrier(mem_dsb_t::DSB_ALL);
}
} // namespace

int main()
{
    std::vector<uint64_t> input = {15};
    std::vector<uint64_t> golden = {13};
    input.resize(ELEMENTS, 0);
    golden.resize(ELEMENTS, 0);
    std::vector<uint64_t> output(ELEMENTS, 0xffffffffffffffffULL);
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint64_t* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), (ELEMENTS) * sizeof(uint64_t),
        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(output_device, input.size() * sizeof(uint64_t), input.data(), input.size() * sizeof(uint64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    asc_clear_nthbit_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), output.size() * sizeof(uint64_t), output_device, output.size() * sizeof(uint64_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", input);
    print_data("Output", output);
    print_data("Golden", golden);
    const bool passed = output == golden;
    std::cout << (passed ? "[Success] asc_clear_nthbit passed." : "[Failed] asc_clear_nthbit failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
