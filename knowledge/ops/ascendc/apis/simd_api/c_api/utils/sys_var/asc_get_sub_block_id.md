---
type: API
title: asc_get_sub_block_id
description: 获取AI Core上Vector核的ID。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_var/asc_get_sub_block_id.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/utils/sys_var.h
- include/c_api/asc_simd.h
---

# asc_get_sub_block_id

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

在[分离模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下获取逻辑AI Core上Cube Core（AIC）或者Vector Core（AIV）的逻辑索引。

## 函数原型

```c
__aicore__ inline int64_t asc_get_sub_block_id()
```

## 参数说明

无

## 返回值说明

返回逻辑AI Core上Cube Core（AIC）或者Vector Core（AIV）的逻辑索引。

SubBlockIdx的取值范围为[0, SubBlockNum)，SubBlockNum可以通过[GetSubBlockNum](asc_get_sub_block_num.md)获取。

## 流水类型

PIPE_S

## 约束说明

本接口为只读查询接口，不修改任何寄存器或存储状态。

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

__global__ __vector__ void asc_get_sub_block_id_kernel(__gm__ uint64_t* output)
{
    asc_init();
    const uint64_t sub_id = static_cast<uint64_t>(asc_get_sub_block_id());
    const uint64_t sub_num = static_cast<uint64_t>(asc_get_sub_block_num());
    output[0] = sub_id;
    output[1] = sub_id;
    output[2] = sub_num;
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
    asc_get_sub_block_id_kernel<<<1, 0>>>(output_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BYTES, output_device, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Observed", output);
    print_hex("Observed hex", output);
    const bool passed = output[2] > 0 && output[1] < output[2];
    std::cout << (passed ? "[Success] asc_get_sub_block_id passed." : "[Failed] asc_get_sub_block_id failed.") << std::endl;
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```
