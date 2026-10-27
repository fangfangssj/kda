---
type: API
title: asc_get_ctrl
description: 读取CTRL寄存器（控制寄存器）特定比特位上的值。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/spr/asc_get_ctrl.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/spr/spr.h
- include/c_api/asc_simd.h
---

# asc_get_ctrl

## 功能说明

头文件路径为：`"c_api/spr/spr.h"`。

读取CTRL寄存器（控制寄存器）特定比特位上的值。

## 函数原型

```c
__aicore__ inline int64_t asc_get_ctrl()
```

## 参数说明

无

## 返回值说明

CTRL寄存器的值。具体含义请参考[asc_set_ctrl](asc_set_ctrl.md)中的描述。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。


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

__global__ __vector__ void asc_get_ctrl_kernel(__gm__ uint64_t* output)
{
    asc_init();
    const uint64_t before = static_cast<uint64_t>(asc_get_ctrl());
    const uint64_t changed = before ^ (1ULL << 53);
    asc_set_ctrl(changed);
    const uint64_t after = static_cast<uint64_t>(asc_get_ctrl());
    asc_set_ctrl(before);
    const uint64_t restored = static_cast<uint64_t>(asc_get_ctrl());
    output[0] = before;
    output[1] = after;
    output[2] = restored;
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
    asc_get_ctrl_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BYTES, output_device, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Observed", output);
    print_hex("Observed hex", output);
    const bool passed = ((output[0] ^ output[1]) & (1ULL << 53)) != 0 && output[2] == output[0];
    std::cout << (passed ? "[Success] asc_get_ctrl passed." : "[Failed] asc_get_ctrl failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
