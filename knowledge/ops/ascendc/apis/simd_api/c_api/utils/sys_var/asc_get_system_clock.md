---
type: API
title: asc_get_system_clock
description: 读取当前系统时钟计数器，返回int64_t类型的当前时间值（单位：微秒）。该接口在底层读取系统cycle计数器后，按当前芯片的时钟频率进行换算，直接返回微秒级时间值，无需调用方自行换算。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_var/asc_get_system_clock.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/utils/sys_var.h
- include/c_api/asc_simd.h
---

# asc_get_system_clock

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

读取当前系统时钟计数器，返回`int64_t`类型的当前时间值（单位：微秒）。该接口在底层读取系统cycle计数器后，按当前芯片的时钟频率进行换算，直接返回微秒级时间值，无需调用方自行换算。


- 针对Ascend 950PR&950DT系列产品，时钟频率为1GHz，换算系数为1000，即`clock = cycle / 1000`。



- 针对Atlas A3系列产品，时钟频率为50MHz，换算系数为50，即`clock = cycle / 50`。



- 针对Atlas A2系列产品，时钟频率为50MHz，换算系数为50，即`clock = cycle / 50`。


本接口与[asc_get_system_cycle](asc_get_system_cycle.md)的区别在于：`asc_get_system_cycle`返回原始cycle计数值，需要调用方根据芯片频率自行换算为时间；`asc_get_system_clock`在接口内部完成换算，直接返回微秒级时间值。本接口在AIC与AIV上均可调用，返回值含义一致。

## 函数原型

```c
__aicore__ inline int64_t asc_get_system_clock()
```

## 参数说明

无

## 返回值说明

返回`int64_t`类型的当前系统时间值，单位为微秒。该值由系统cycle计数器读数除以芯片对应的换算系数得到，随系统时钟持续递增。

## 流水类型

PIPE_S

## 约束说明

- 本接口为只读查询接口，读取的是系统时钟计数器并按芯片频率换算，不修改任何寄存器或存储状态，可在AIC与AIV上下文中调用。
- 返回值随系统时钟持续递增，同一核函数（Kernel）内多次调用返回值单调不减。
- 本接口与`asc_get_system_cycle`读取的是同一个底层计数器，区别仅在于本接口在内部完成了频率换算，返回值为微秒，而`asc_get_system_cycle`返回原始cycle值。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```


```cpp
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {
constexpr uint32_t ELEMENTS = 16;
constexpr uint32_t BYTES = ELEMENTS * sizeof(uint64_t);

void print_data(const char* label, const std::vector<uint64_t>& data)
{
    std::cout << label << ":";
    for (uint32_t i = 0; i < 8; ++i) std::cout << ' ' << data[i];
    std::cout << " ..." << std::endl;
}

void print_hex(const char* label, const std::vector<uint64_t>& data)
{
    std::cout << label << ":" << std::hex;
    for (uint32_t i = 0; i < 4; ++i) std::cout << " 0x" << data[i];
    std::cout << std::dec << std::endl;
}

__global__ __vector__ void asc_get_system_clock_kernel(__gm__ uint64_t* output)
{
    asc_init();
    const uint64_t before = static_cast<uint64_t>(asc_get_system_clock());
    for (volatile uint32_t i = 0; i < 256; ++i) {}
    const uint64_t after = static_cast<uint64_t>(asc_get_system_clock());
    output[0] = before;
    output[1] = after;
    output[2] = after - before;
    asc_dcci_single(output);
}

} // namespace

int main()
{
    std::vector<uint64_t> output(ELEMENTS, ~0ULL);
    aclInit(nullptr);
    aclrtSetDevice(0);
    uint64_t* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(output_device, BYTES, output.data(), BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_get_system_clock_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BYTES, output_device, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Observed", output);
    print_hex("Observed hex", output);
    const bool passed = output[1] > output[0] && output[2] > 0;
    std::cout << (passed ? "[Success] asc_get_system_clock passed." : "[Failed] asc_get_system_clock failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
